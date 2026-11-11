"""Retrain the matcher locally (no Colab needed) and write artifacts/pipeline_v<N>.joblib.

Steps
  1. ablation: v2 features -> +cleaning -> +skills -> +experience -> +weighted skills (Random Forest each time)
  2. 7-algorithm comparison on the final features (selected by grouped CV, as in Colab)
  3. ordering: classifier score vs XGBoost learning-to-rank (selected by grouped-CV NDCG@10)
  4. labels: argmax vs thresholds on the ordering score (selected by grouped-CV macro-F1)

All choices use grouped 5-fold CV on the train split only. The official test split
(jobs never seen in training) is only used for reporting.

Usage:  .venv/bin/python training/train.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, ndcg_score,
                             precision_score, recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.ml import features as F  # noqa: E402
from app.ml.pipeline import (F_chunks, bert_pair, feature_names, make_matcher, pair_rule_features,  # noqa: E402
                             stage2_matrix, tfidf_pair, tiers_from_score)

warnings.filterwarnings('ignore')
ART = ROOT / 'artifacts'
LABELS = {'No Fit': 0, 'Potential Fit': 1, 'Good Fit': 2}
BERT_NAME = 'all-MiniLM-L6-v2'
CV = GroupKFold(5)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ metrics

def ndcg10(groups, y, score):
    d = pd.DataFrame({'g': groups, 'y': y, 's': score})
    return float(np.mean([ndcg_score([g.y.values], [g.s.values], k=10) for _, g in d.groupby('g') if len(g) > 1]))


def top3_hit(groups, y, score):
    d = pd.DataFrame({'g': groups, 'y': y, 's': score})
    return float((d.sort_values('s', ascending=False).groupby('g').head(3).y > 0).mean())


def report(y, pred, proba, groups, score):
    pb = proba[:, 1] + proba[:, 2]
    return {'Accuracy': accuracy_score(y, pred), 'Precision': precision_score(y, pred, average='macro'),
            'Recall': recall_score(y, pred, average='macro'), 'F1 (macro)': f1_score(y, pred, average='macro'),
            'ROC-AUC': roc_auc_score(y, proba, multi_class='ovr'), 'NDCG@10': ndcg10(groups, y, score),
            'Top-3 hit': top3_hit(groups, y, score),
            'Shortlist acc': accuracy_score(y > 0, pb > 0.5), 'Shortlist AUROC': roc_auc_score(y > 0, pb)}


# ------------------------------------------------------------------ embeddings

class Embedder:
    def __init__(self):
        self.cache = {}
        for f in ('embeddings.joblib', 'embeddings_v3.joblib'):
            if (ART / f).exists():
                self.cache.update(joblib.load(ART / f))
        self.model = None

    def ensure(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if not todo:
            return
        log(f'embedding {len(todo)} new texts with {BERT_NAME} (CPU)...')
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self.model = SentenceTransformer(BERT_NAME, device='cpu', local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(BERT_NAME, device='cpu')
        chunks, owner = [], []
        for i, t in enumerate(todo):
            c = F_chunks(t)
            chunks += c
            owner += [i] * len(c)
        E = self.model.encode(chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        owner = np.asarray(owner)
        for i, t in enumerate(todo):
            e = E[owner == i].mean(0)
            self.cache[t] = (e / np.linalg.norm(e)).astype(np.float32)

    def matrix(self, texts):
        return np.vstack([self.cache[t] for t in texts])


# ------------------------------------------------------------------ stage 1

def stage1(tr, te, groups, y_tr, emb: Embedder):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=2, max_features=50000, sublinear_tf=True)
    tfidf.fit(pd.concat([tr.resume_text, tr.job_description_text]))
    Xt_tr, tcos_tr = tfidf_pair(tfidf, tr)
    Xt_te, tcos_te = tfidf_pair(tfidf, te)
    emb.ensure(pd.concat([tr.resume_text, tr.job_description_text, te.resume_text, te.job_description_text]))
    Xb_tr, bcos_tr = bert_pair(emb.matrix(tr.resume_text), emb.matrix(tr.job_description_text))
    Xb_te, bcos_te = bert_pair(emb.matrix(te.resume_text), emb.matrix(te.job_description_text))
    s1_t = LogisticRegression(max_iter=3000, C=3, class_weight='balanced')
    s1_b = LogisticRegression(max_iter=3000, C=1, class_weight='balanced')
    oof_t = cross_val_predict(s1_t, Xt_tr, y_tr, cv=CV, groups=groups, method='predict_proba', n_jobs=-1)
    oof_b = cross_val_predict(s1_b, Xb_tr, y_tr, cv=CV, groups=groups, method='predict_proba', n_jobs=-1)
    s1_t.fit(Xt_tr, y_tr)
    s1_b.fit(Xb_tr, y_tr)
    return dict(tfidf=tfidf, s1_tfidf=s1_t, s1_bert=s1_b,
                tr=(oof_t, oof_b, bcos_tr, tcos_tr),
                te=(s1_t.predict_proba(Xt_te), s1_b.predict_proba(Xb_te), bcos_te, tcos_te))


def rf():
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1)


def all_models():
    from xgboost import XGBClassifier
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight='balanced')),
        'Naive Bayes': make_pipeline(StandardScaler(), GaussianNB()),
        'SVM (RBF)': make_pipeline(StandardScaler(), SVC(C=1, class_weight='balanced', probability=True, random_state=42)),
        'Random Forest': rf(),
        'Gradient Boosting': HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, class_weight='balanced', random_state=42),
        'Neural Net (MLP)': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=2000, random_state=42)),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42, n_jobs=-1),
    }


def evaluate(model, S_tr, y_tr, g_tr, S_te, y_te, g_te):
    oof = cross_val_predict(clone(model), S_tr, y_tr, cv=CV, groups=g_tr, method='predict_proba')
    m = clone(model).fit(S_tr, y_tr)
    pr = m.predict_proba(S_te)
    cv = {'CV macro-F1': f1_score(y_tr, oof.argmax(1), average='macro'),
          'CV NDCG@10': ndcg10(g_tr, y_tr, oof[:, 2] + .5 * oof[:, 1])}
    return m, oof, pr, {**cv, **report(y_te, pr.argmax(1), pr, g_te, pr[:, 2] + .5 * pr[:, 1])}


# ------------------------------------------------------------------ main

def main(quick: bool):
    tr0 = pd.read_parquet(ART / 'train_features.parquet')[['resume_text', 'job_description_text', 'label']]
    te0 = pd.read_parquet(ART / 'test_features.parquet')[['resume_text', 'job_description_text', 'label']]
    y_tr, y_te = tr0.label.map(LABELS).values, te0.label.map(LABELS).values
    known = F.build_known_words(pd.concat([tr0.resume_text, tr0.job_description_text]).drop_duplicates())
    emb = Embedder()

    texts = {2: (tr0, te0)}
    c3 = {t: F.clean_v3(t, known) for t in pd.concat([tr0.resume_text, tr0.job_description_text,
                                                      te0.resume_text, te0.job_description_text]).unique()}
    texts[3] = tuple(d.assign(resume_text=d.resume_text.map(c3), job_description_text=d.job_description_text.map(c3))
                     for d in (tr0, te0))
    log(f'clean v3 changed {sum(k != v for k, v in c3.items())} of {len(c3)} texts')

    s1 = {}
    for v in (2, 3):
        tr, te = texts[v]
        log(f'stage 1 (clean v{v})...')
        s1[v] = stage1(tr, te, tr.job_description_text.factorize()[0], y_tr, emb)

    configs = [('v2 (Colab)', {'clean': 2, 'skills': 2, 'exp': 2}),
               ('+ cleaning', {'clean': 3, 'skills': 2, 'exp': 2}),
               ('+ skills', {'clean': 3, 'skills': 3, 'exp': 2}),
               ('+ experience', {'clean': 3, 'skills': 3, 'exp': 3}),
               ('+ weighted skills', {'clean': 3, 'skills': 4, 'exp': 3})]
    rows, built = [], {}
    for name, cfg in configs:
        tr, te = texts[cfg['clean']]
        g_tr, g_te = tr.job_description_text.factorize()[0], te.job_description_text.values
        matcher = make_matcher(cfg)
        w = cfg['skills'] == 4
        tr = pd.concat([tr, pair_rule_features(tr, matcher, cfg['exp'], weighted=w)], axis=1)
        te = pd.concat([te, pair_rule_features(te, matcher, cfg['exp'], weighted=w)], axis=1)
        S_tr = stage2_matrix(tr, *s1[cfg['clean']]['tr'], cfg)
        S_te = stage2_matrix(te, *s1[cfg['clean']]['te'], cfg)
        assert S_tr.shape[1] == len(feature_names(cfg))
        _, _, _, met = evaluate(rf(), S_tr, y_tr, g_tr, S_te, y_te, g_te)
        rows.append({'step': name, **met})
        built[name] = (cfg, tr, te, S_tr, S_te, g_tr, g_te, matcher)
        log(f'{name:14} CV F1 {met["CV macro-F1"]:.3f}  CV NDCG {met["CV NDCG@10"]:.3f} | test F1 {met["F1 (macro)"]:.3f}  NDCG {met["NDCG@10"]:.3f}')
    ablation = pd.DataFrame(rows).set_index('step').round(3)

    # ---- final feature set: best mean of CV macro-F1 and CV NDCG@10 (labels and ranking both matter)
    best_step = (ablation['CV macro-F1'] + ablation['CV NDCG@10']).idxmax()
    cfg, tr, te, S_tr, S_te, g_tr, g_te, matcher = built[best_step]
    log(f'final features: {best_step} {cfg}')

    # ---- 7-model comparison
    models = {'Random Forest': rf()} if quick else all_models()
    comp, fitted = [], {}
    for name, m in models.items():
        fitted[name] = evaluate(m, S_tr, y_tr, g_tr, S_te, y_te, g_te)
        comp.append({'model': name, **fitted[name][3]})
        log(f'{name:20} CV F1 {fitted[name][3]["CV macro-F1"]:.3f} | test acc {fitted[name][3]["Accuracy"]:.3f} F1 {fitted[name][3]["F1 (macro)"]:.3f}')
    comp = pd.DataFrame(comp).set_index('model').round(3)
    clf_name = comp['CV macro-F1'].idxmax()
    clf, oof_clf, pr_clf, _ = fitted[clf_name]
    log(f'classifier selected by CV: {clf_name}')

    # ---- ordering: classifier score vs learning-to-rank
    from xgboost import XGBRanker
    order = tr.assign(_g=g_tr).sort_values('_g', kind='stable').index.values   # rankers need contiguous groups

    def make_ranker():
        return XGBRanker(objective='rank:ndcg', n_estimators=300, max_depth=3, learning_rate=0.05,
                         subsample=0.8, lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=10,
                         ndcg_exp_gain=False, random_state=42, n_jobs=-1)
    oof_rank = np.zeros(len(y_tr))
    for f_tr, f_va in CV.split(S_tr, y_tr, g_tr):
        f_tr = np.sort(f_tr)[np.argsort(g_tr[np.sort(f_tr)], kind='stable')]
        r = make_ranker().fit(S_tr[f_tr], y_tr[f_tr], qid=g_tr[f_tr])
        oof_rank[f_va] = r.predict(S_tr[f_va])
    ranker = make_ranker().fit(S_tr[order], y_tr[order], qid=g_tr[order])
    te_rank = ranker.predict(S_te)
    clf_oof_score = oof_clf[:, 2] + .5 * oof_clf[:, 1]
    clf_te_score = pr_clf[:, 2] + .5 * pr_clf[:, 1]
    cv_ndcg = {'classifier': ndcg10(g_tr, y_tr, clf_oof_score), 'ranker': ndcg10(g_tr, y_tr, oof_rank)}
    use_ranker = cv_ndcg['ranker'] > cv_ndcg['classifier']
    log(f'CV NDCG@10 classifier {cv_ndcg["classifier"]:.3f} vs ranker {cv_ndcg["ranker"]:.3f} -> '
        f'{"ranker" if use_ranker else "classifier"} | test NDCG classifier {ndcg10(g_te, y_te, clf_te_score):.3f} '
        f'ranker {ndcg10(g_te, y_te, te_rank):.3f}')
    oof_score, te_score = (oof_rank, te_rank) if use_ranker else (clf_oof_score, clf_te_score)

    # ---- labels: argmax vs thresholds tuned on OOF ordering score
    qs = np.quantile(oof_score, np.linspace(0.02, 0.98, 49))
    best = (-1, None)
    for i, t1 in enumerate(qs):
        for t2 in qs[i + 1:]:
            f = f1_score(y_tr, tiers_from_score(oof_score, (t1, t2)), average='macro')
            if f > best[0]:
                best = (f, (float(t1), float(t2)))
    cv_argmax = f1_score(y_tr, oof_clf.argmax(1), average='macro')
    use_thr = best[0] > cv_argmax
    thresholds = best[1] if use_thr else None
    log(f'CV macro-F1 argmax {cv_argmax:.3f} vs thresholds {best[0]:.3f} -> {"thresholds" if use_thr else "argmax"}')

    # ---- final test report
    pred = tiers_from_score(te_score, thresholds) if use_thr else pr_clf.argmax(1)
    final = report(y_te, pred, pr_clf, g_te, te_score)
    final.update({'CV macro-F1': best[0] if use_thr else cv_argmax, 'CV NDCG@10': max(cv_ndcg.values())})
    log('FINAL (official test split):', {k: round(v, 3) for k, v in final.items()})
    print(confusion_matrix(y_te, pred))

    comp.to_csv(ART / f'model_comparison_{TAG}.csv')
    summary = pd.concat([ablation, pd.DataFrame([{**final}], index=['+ ranking & thresholds (final)']).round(3)])
    summary.to_csv(ART / f'ablation_{TAG}.csv')
    print('\nABLATION (Random Forest unless noted)\n', summary[['CV macro-F1', 'CV NDCG@10', 'Accuracy', 'F1 (macro)',
                                                            'ROC-AUC', 'NDCG@10', 'Top-3 hit', 'Shortlist AUROC']].to_string())
    print('\nMODEL COMPARISON (final features)\n', comp.to_string())

    # ---- save
    joblib.dump({t: emb.cache[t] for t in c3.values()}, ART / 'embeddings_v3.joblib')
    catalog = getattr(matcher, 'catalog', None)
    joblib.dump({'version': VERSION, 'cfg': cfg, 'known_words': sorted(known), 'catalog': catalog,
                 'tfidf': s1[cfg['clean']]['tfidf'], 's1_tfidf': s1[cfg['clean']]['s1_tfidf'],
                 's1_bert': s1[cfg['clean']]['s1_bert'], 'stage2': clf, 'stage2_name': clf_name,
                 'ranker': ranker if use_ranker else None, 'ranker_name': 'XGBoost LambdaMART (rank:ndcg)' if use_ranker else None,
                 'thresholds': thresholds, 'bert_model_name': BERT_NAME, 'labels': LABELS,
                 'skills': matcher.skills, 'embeddings_file': 'embeddings_v3.joblib',
                 'feature_names': feature_names(cfg),
                 'metrics': {'test': {k: round(v, 4) for k, v in final.items()},
                             'ablation': summary.reset_index().to_dict('records'),
                             'model_comparison': comp.reset_index().to_dict('records')}},
                ART / f'pipeline_{TAG}.joblib', compress=3)
    log(f'saved artifacts/pipeline_{TAG}.joblib, ablation_{TAG}.csv, model_comparison_{TAG}.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='only Random Forest in the model comparison')
    ap.add_argument('--version', type=int, default=3, help='artifact version number -> artifacts/pipeline_v<N>.joblib')
    a = ap.parse_args()
    VERSION, TAG = a.version, f'v{a.version}'
    main(a.quick)
