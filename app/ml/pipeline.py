"""Inference for the hybrid resume-job matcher.

Supports two artifact formats:
  v2  pipeline_v2.joblib  trained in Colab (resume_job_matcher.py), features exactly as there
  v3  pipeline_v3.joblib  trained by training/train.py: better cleaning, skills, experience,
                          optional learning-to-rank model and tuned label thresholds

The same functions here are used by training/train.py, so training and serving cannot drift.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from app.ml import features as F

V2_CFG = {'clean': 2, 'skills': 2, 'exp': 2}
RAW_V2 = ['t_nofit', 't_potential', 't_good', 'b_nofit', 'b_potential', 'b_good',
          'bert_cos', 'tfidf_cos', 'skill_score', 'skill_jaccard', 'n_matched', 'exp_score']
RAW_V3_EXTRA = ['n_jd_skills', 'years_required_f', 'years_found_f', 'years_gap']
WJ_V2 = ['t0', 't2', 'texp', 'b2', 'bcos', 'tcos', 'sk']
WJ_V3_EXTRA = ['yf']
RAW_V4_EXTRA = ['w_skill_score', 'req_coverage', 'pref_coverage', 'n_required']
WJ_V4_EXTRA = ['wsk', 'req']


def feature_names(cfg: dict) -> list[str]:
    v3 = cfg.get('exp', 2) == 3
    v4 = cfg.get('skills', 2) == 4
    raw = RAW_V2 + (RAW_V3_EXTRA if v3 else []) + (RAW_V4_EXTRA if v4 else [])
    wj = WJ_V2 + (WJ_V3_EXTRA if v3 else []) + (WJ_V4_EXTRA if v4 else [])
    return raw + [f'{k}_{s}' for k in wj for s in ('pct', 'z')]


# ---------------------------------------------------------------- building blocks (shared with training)

def make_matcher(cfg: dict, catalog=None) -> F.SkillMatcher:
    from app.ml.skills_catalog import CATALOG_V3, CATALOG_V4
    if cfg['skills'] == 2:
        return F.SkillMatcher(2, catalog)
    return F.SkillMatcher(3, catalog or (CATALOG_V4 if cfg['skills'] == 4 else CATALOG_V3))


def pair_rule_features(df: pd.DataFrame, matcher: F.SkillMatcher, exp_version: int,
                       weighted: bool = False) -> pd.DataFrame:
    """Skill + experience columns for (resume_text, job_description_text) pairs of cleaned text."""
    jd_cache: dict[str, set] = {}
    imp_cache: dict[str, dict] = {}
    res_cache: dict[str, set] = {}
    rows = []
    for r, j in zip(df.resume_text, df.job_description_text):
        rs = res_cache.get(r)
        if rs is None:
            rs = res_cache[r] = matcher.extract(r)
        js = jd_cache.get(j)
        if js is None:
            js = jd_cache[j] = matcher.extract(j)
        inter = rs & js
        row = {'skill_score': len(inter) / len(js) if js else 0.0,
               'skill_jaccard': len(inter) / len(rs | js) if (rs | js) else 0.0,
               'n_matched': len(inter), 'n_jd_skills': len(js),
               'matched_skills': sorted(inter), 'missing_skills': sorted(js - rs)}
        row.update(F.experience(r, j, exp_version))
        if weighted:
            imp = imp_cache.get(j)
            if imp is None:
                imp = imp_cache[j] = F.jd_skill_importance(j, matcher)
            row.update(F.weighted_skill_features(imp, rs))
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def tfidf_pair(tfidf, df):
    R, J = tfidf.transform(df.resume_text), tfidf.transform(df.job_description_text)
    return sp.hstack([R.multiply(J), abs(R - J)]).tocsr(), np.asarray(R.multiply(J).sum(1)).ravel()


def bert_pair(RB, JB):
    return np.hstack([RB * JB, np.abs(RB - JB)]), (RB * JB).sum(1).astype(float)


def within_job(groups, cols: dict) -> np.ndarray:
    d = pd.DataFrame(cols)
    d['g'] = groups
    out = []
    for c in cols:
        grp = d.groupby('g')[c]
        out += [grp.rank(pct=True).values[:, None],
                ((d[c] - grp.transform('mean')) / (grp.transform('std').fillna(1) + 1e-6)).values[:, None]]
    return np.hstack(out)


def stage2_matrix(df: pd.DataFrame, p_tfidf, p_bert, bcos, tcos, cfg: dict) -> np.ndarray:
    raw = [p_tfidf, p_bert, bcos, tcos, df.skill_score, df.skill_jaccard, df.n_matched, df.exp_score]
    wj = {'t0': p_tfidf[:, 0], 't2': p_tfidf[:, 2], 'texp': p_tfidf[:, 2] + .5 * p_tfidf[:, 1],
          'b2': p_bert[:, 2], 'bcos': bcos, 'tcos': tcos, 'sk': df.skill_score.values}
    if cfg.get('exp', 2) == 3:
        need = df.years_required.astype(float)
        have = df.years_found.astype(float)
        gap = (have - need).fillna(0.0)
        raw += [df.n_jd_skills, need.fillna(-1), have.fillna(-1), gap]
        wj['yf'] = have.fillna(0).values
    if cfg.get('skills', 2) == 4:
        raw += [df.w_skill_score, df.req_coverage, df.pref_coverage, df.n_required]
        wj['wsk'] = df.w_skill_score.values
        wj['req'] = df.req_coverage.values
    return np.hstack([np.c_[tuple(raw)], within_job(df.job_description_text.values, wj)])


def tiers_from_score(score: np.ndarray, thresholds) -> np.ndarray:
    """0/1/2 tiers from an ordering score and two cut points."""
    t1, t2 = thresholds
    return np.where(score >= t2, 2, np.where(score >= t1, 1, 0))


# ---------------------------------------------------------------- the pipeline

class MatcherPipeline:
    """Loads a trained artifact once and scores (resume, job) pairs."""

    def __init__(self, artifacts_dir: Path, pipeline_file: str = 'pipeline_v2.joblib',
                 embedding_cache_size: int = 20000, device: str | None = None, load_bert: bool = True):
        artifacts_dir = Path(artifacts_dir)
        p = joblib.load(artifacts_dir / pipeline_file)
        self.pipeline_file = pipeline_file
        self.version = p.get('version', 2)
        self.cfg = p.get('cfg', V2_CFG)
        self.known_words = frozenset(p.get('known_words', ()))
        self.tfidf = p['tfidf']
        self.s1_tfidf = p['s1_tfidf']
        self.s1_bert = p['s1_bert']
        self.stage2 = p['stage2']
        self.stage2_name = p['stage2_name']
        self.ranker = p.get('ranker')
        self.ranker_name = p.get('ranker_name')
        self.thresholds = p.get('thresholds')
        self.bert_model_name = p['bert_model_name']
        self.labels = [k for k, _ in sorted(p['labels'].items(), key=lambda kv: kv[1])]
        self.metrics = p.get('metrics', {})
        self.matcher = make_matcher(self.cfg, p.get('catalog') if self.cfg['skills'] >= 3 else p['skills'])
        self.skills = self.matcher.skills
        # Explanations always use the v4 catalog + hierarchy + required/preferred weighting,
        # independent of which features the ranking model was trained on.
        self.explain_matcher = make_matcher({'skills': 4})
        self._jd_imp_cache: dict[str, dict] = {}
        self.feature_names = feature_names(self.cfg)
        assert self.stage2.n_features_in_ == len(self.feature_names), 'artifact / feature mismatch'

        # Embedding cache, seeded with precomputed vectors for all dataset texts.
        self._emb: OrderedDict[str, np.ndarray] = OrderedDict()
        self._emb_max = max(embedding_cache_size, 1)
        cache_file = artifacts_dir / p.get('embeddings_file', 'embeddings.joblib')
        if cache_file.exists():
            for t, v in joblib.load(cache_file).items():
                self._emb[t] = np.asarray(v, dtype=np.float32)
        self._lock = threading.Lock()

        self._device = device
        self._bert = None
        if load_bert:
            self._load_bert()

    # ------------------------------------------------------------ text
    def clean(self, text) -> str:
        return F.clean(text, self.cfg['clean'], self.known_words)

    def extract_skills(self, text: str) -> set[str]:
        return self.matcher.extract(text)

    def jd_importance(self, jd_clean: str) -> dict:
        imp = self._jd_imp_cache.get(jd_clean)
        if imp is None:
            imp = F.jd_skill_importance(F.strip_jd_footer(jd_clean), self.explain_matcher)
            if len(self._jd_imp_cache) > 1000:
                self._jd_imp_cache.clear()
            self._jd_imp_cache[jd_clean] = imp
        return imp

    def years_required(self, jd_clean: str):
        return F.experience('', jd_clean, self.cfg['exp'])['years_required']

    # ------------------------------------------------------------ embeddings
    def _load_bert(self):
        if self._bert is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = self._device or ('cuda' if torch.cuda.is_available() else 'cpu')
            try:   # use the local copy if already downloaded; avoids network calls on every start
                self._bert = SentenceTransformer(self.bert_model_name, device=dev, local_files_only=True)
            except Exception:
                self._bert = SentenceTransformer(self.bert_model_name, device=dev)
        return self._bert

    @property
    def bert_loaded(self) -> bool:
        return self._bert is not None

    def embed(self, texts) -> np.ndarray:
        """Mean-pooled, L2-normalised 200-word-chunk embeddings (Colab embed_mean), batched."""
        texts = list(texts)
        with self._lock:
            todo = [t for t in dict.fromkeys(texts) if t not in self._emb]
        if todo:
            bert = self._load_bert()
            all_chunks, owner = [], []
            for i, t in enumerate(todo):
                c = F_chunks(t)
                all_chunks += c
                owner += [i] * len(c)
            E = bert.encode(all_chunks, normalize_embeddings=True, batch_size=64,
                            convert_to_numpy=True, show_progress_bar=False)
            owner = np.asarray(owner)
            with self._lock:
                for i, t in enumerate(todo):
                    e = E[owner == i].mean(axis=0)
                    self._emb[t] = (e / np.linalg.norm(e)).astype(np.float32)
        with self._lock:
            out = []
            for t in texts:
                self._emb.move_to_end(t)
                out.append(self._emb[t])
            while len(self._emb) > self._emb_max:
                self._emb.popitem(last=False)
        return np.vstack(out)

    @property
    def cache_size(self) -> int:
        return len(self._emb)

    # ------------------------------------------------------------ scoring
    def score_pairs(self, resumes, jobs) -> pd.DataFrame:
        """Score already-cleaned (resume, job) pairs. Within-job features compare each resume with
        the other resumes for the same job text in this call, exactly as in training."""
        df = pd.DataFrame({'resume_text': list(resumes), 'job_description_text': list(jobs)})
        if df.empty:
            raise ValueError('no pairs to score')
        df = pd.concat([df, pair_rule_features(df, self.matcher, self.cfg['exp'],
                                               weighted=self.cfg['skills'] == 4)], axis=1)

        Xt, tcos = tfidf_pair(self.tfidf, df)
        p_tfidf = self.s1_tfidf.predict_proba(Xt)
        Xb, bcos = bert_pair(self.embed(df.resume_text), self.embed(df.job_description_text))
        p_bert = self.s1_bert.predict_proba(Xb)

        S = stage2_matrix(df, p_tfidf, p_bert, bcos, tcos, self.cfg)
        proba = self.stage2.predict_proba(S)
        df['tfidf_score'] = tcos
        df['bert_score'] = bcos
        names = self.feature_names
        for name, col in zip(names, S.T):
            if name.endswith(('_pct', '_z')):
                df[name] = col
        df['p_no_fit'], df['p_potential_fit'], df['p_good_fit'] = proba.T

        # ordering score: learning-to-rank model if trained, else P(Good) + 0.5 * P(Potential)
        df['rank_score'] = (self.ranker.predict(S) if self.ranker is not None
                            else proba[:, 2] + 0.5 * proba[:, 1])
        tier = (tiers_from_score(df.rank_score.values, self.thresholds) if self.thresholds is not None
                else proba.argmax(1))
        df['label'] = [self.labels[i] for i in tier]
        df['confidence'] = proba[np.arange(len(df)), tier]
        df.attrs['features'] = S

        # explanation layer (does not affect scores)
        if self.cfg['skills'] != 4:
            expl = [F.weighted_skill_features(self.jd_importance(j), self.explain_matcher.extract(r))
                    for r, j in zip(df.resume_text, df.job_description_text)]
            for col in ('w_skill_score', 'req_coverage', 'pref_coverage', 'n_required', 'n_core', 'n_core_matched',
                        'skill_details'):
                df[col] = [e[col] for e in expl]
        df['matched_skills'] = [[d['skill'] for d in det if d['status'] == 'matched'] for det in df.skill_details]
        df['missing_skills'] = [[d['skill'] for d in det if d['status'] == 'missing'] for det in df.skill_details]
        return df

    def rank(self, resumes: list[str], job_description: str, ranking: str = 'model',
             core_guard: bool = True) -> pd.DataFrame:
        """Clean, score and sort many resumes against one job (best first)."""
        jd = self.clean(job_description)
        res = [self.clean(r) for r in resumes]
        df = finalize(self.score_pairs(res, [jd] * len(res)), ranking, core_guard)
        df['input_index'] = np.arange(len(df))
        return df.sort_values('rank_score', ascending=False, kind='stable').reset_index(drop=True)


FORMULA_WEIGHTS = {'skills': 0.40, 'experience': 0.25, 'semantic': 0.20, 'keyword': 0.15}
CORE_GUARD_MIN = 3     # JD must list at least this many required technologies for the guard to apply


def finalize(df: pd.DataFrame, ranking: str = 'model', core_guard: bool = True) -> pd.DataFrame:
    """Add the readable score breakdown, choose the ordering score and apply the core-skill label rule.

    ranking='model'   : order by the trained model (best measured NDCG@10 on the test split)
    ranking='formula' : order by 0.40*skills + 0.25*experience + 0.20*semantic + 0.15*keyword
    core_guard        : a resume that matches none of the JD's core technologies (when it lists >= 3)
                        is never labelled Good Fit; it is capped at Potential Fit. Ordering is unchanged.
    `rank_score` is always exactly the score the list is sorted by.
    """
    if ranking not in ('model', 'formula'):
        raise ValueError(f'unknown ranking {ranking!r}')
    df = df.copy()
    grp = df.groupby('job_description_text')
    for col, out in (('bert_score', 'semantic_scaled'), ('tfidf_score', 'keyword_scaled')):
        lo, hi = grp[col].transform('min'), grp[col].transform('max')
        df[out] = ((df[col] - lo) / (hi - lo).where(hi > lo)).fillna(1.0)
    w = FORMULA_WEIGHTS
    df['formula_score'] = (w['skills'] * df.w_skill_score + w['experience'] * df.exp_score +
                           w['semantic'] * df.semantic_scaled + w['keyword'] * df.keyword_scaled)
    df['model_score'] = df['rank_score']
    if ranking == 'formula':
        df['rank_score'] = df['formula_score']
    df['ranked_by'] = ranking
    df['label_capped'] = False
    if core_guard:
        cap = (df.n_core >= CORE_GUARD_MIN) & (df.n_core_matched == 0) & (df.label == 'Good Fit')
        df.loc[cap, 'label'] = 'Potential Fit'
        df.loc[cap, 'confidence'] = df.loc[cap, 'p_potential_fit']
        df['label_capped'] = cap
    return df


def F_chunks(text: str, size: int = 200) -> list[str]:
    w = text.split()
    return [' '.join(w[i:i + size]) for i in range(0, len(w), size)] or ['']


# backwards-compatible helpers used elsewhere
clean = F.clean_v2
chunks = F_chunks


def years(text: str):
    return F.years_v2(text)
