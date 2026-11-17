"""Candidate mode: how well does ONE resume fit ONE job, and what should the candidate change?

The model's stage-2 features compare each resume with the other applicants for the same job
(within-job percentiles / z-scores), so a resume scored alone always looks like the "best" applicant.
To keep the label honest, the resume is scored together with a reference pool: the training-set
resumes most similar to the job description (by BERT embedding), about the size of an applicant
pool in training (~22 per job). The label, probabilities and position come from that comparison.
Skill gaps, experience and keyword suggestions are absolute (they don't depend on the pool).
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from app.explain import MAX_LIST, _fmt_list, _ordinal, _yrs, add_positions
from app.ml import features as F
from app.ml.pipeline import MatcherPipeline
from app.parsing import guess_candidate_name
from app.ranker import RankError, _candidate, _importance_lists, _num_or_none
from app.schemas import Comparison, FitCheckResponse, JobInfo, SkillGap, Suggestion

MAX_KEYWORDS = 10
SHORT_RESUME_WORDS = 150
MIN_RESUME_WORDS = 30            # below this there is nothing meaningful to compare
_PRIORITY = {'high': 0, 'medium': 1, 'low': 2}
_HAS_DIGIT = re.compile(r'\d')
# salary / benefits / legal / hiring boilerplate and job-ad filler: never useful resume advice
_BOILERPLATE = frozenset('''
salary base pay compensation benefits benefit bonus insurance medical dental vision pto vacation holiday holidays
leave accommodation accommodations eligible eligibility equal opportunity employer disability disabilities veteran
veterans gender race religion color national origin sexual orientation age status protected law laws applicable
applicant applicants application applications apply candidate candidates hire hiring hired offer sign designation
position positions role roles job jobs company organization people person employee employees location locations
office remote onsite hybrid relocation visa sponsorship travel range per hour hourly annual annually year years
month months week day days time full part contract permanent temporary including include includes etc please
want wants need needs way ways following follow help helps helping world passion passionate study studies
responsibility responsibilities ability abilities able strong work working works looking join great new good best
like make makes well also within across use using used opportunities ideal plus must preferred required requirements
qualifications skills knowledge understanding understand environment environments related similar various multiple others
employment field fields senior junior
'''.split())


class ReferencePool:
    """Cleaned training resumes + their (cached) embeddings, loaded once on first use."""

    def __init__(self, pipe: MatcherPipeline, source: Path):
        self.pipe, self.source = pipe, Path(source)
        self._lock = threading.Lock()
        self.raw: list[str] | None = None        # original text (what pipe.rank expects)
        self.texts: list[str] | None = None      # cleaned text (embedding cache key)
        self.emb: np.ndarray | None = None

    def _load(self):
        with self._lock:
            if self.texts is None:
                raw = pd.read_parquet(self.source, columns=['resume_text']).resume_text.dropna().unique()
                by_clean: dict[str, str] = {}
                for r in raw:
                    c = self.pipe.clean(r)
                    if c and c not in by_clean:
                        by_clean[c] = r
                self.emb = self.pipe.embed(list(by_clean))
                self.raw = list(by_clean.values())
                self.texts = list(by_clean)

    def nearest(self, jd_clean: str, k: int, exclude: str = '') -> list[str]:
        """Raw text of the k reference resumes most similar to the job (skipping a copy of the cleaned `exclude`)."""
        self._load()
        sims = self.emb @ self.pipe.embed([jd_clean])[0]
        out = []
        for i in np.argsort(-sims):
            if self.texts[i] != exclude:
                out.append(self.raw[i])
                if len(out) == k:
                    break
        return out


# ---------------------------------------------------------------- keywords

def missing_keywords(pipe: MatcherPipeline, jd_clean: str, resume_clean: str, known_skills: set[str],
                     k: int = MAX_KEYWORDS) -> list[str]:
    """Highest-weighted TF-IDF terms of the job description that never appear in the resume.

    A term must occur at least twice in the job description (one-off words are often a city, the company
    or boilerplate). Skips salary/benefits/legal and filler wording, numbers, skills already reported in the skill gap (and their plural/singular
    forms), and single words already covered by a chosen phrase."""
    vocab = pipe.tfidf.get_feature_names_out()
    jd_clean = F.strip_jd_footer(jd_clean)
    jd_vec = pipe.tfidf.transform([jd_clean])
    res_terms = set(pipe.tfidf.transform([resume_clean]).indices)
    res_words = set(resume_clean.lower().split())
    jd_low = ' ' + ' '.join(jd_clean.lower().split()) + ' '
    skill_forms = {f for s in known_skills for f in (s, s + 's', s.rstrip('s'))}
    order = jd_vec.indices[np.argsort(-jd_vec.data)]
    chosen: list[str] = []
    for idx in order:
        term = vocab[idx]
        words = term.split()
        if (idx in res_terms or _HAS_DIGIT.search(term) or len(term) < 3 or term in skill_forms
                or any(w in _BOILERPLATE for w in words) or jd_low.count(f' {term} ') < 2):
            continue
        if len(words) == 1 and (term in res_words or any(term in c.split() for c in chosen)):
            continue
        if len(words) > 1 and (all(w in res_words for w in words) or any(w in skill_forms for w in words)):
            continue
        chosen = [c for c in chosen if not (len(c.split()) == 1 and c in words)]
        chosen.append(term)
        if len(chosen) == k:
            break
    return chosen


# ---------------------------------------------------------------- advice

def _skill_suggestions(details: list[dict]) -> list[Suggestion]:
    out = []
    req_tech = [d for d in details if d['importance'] == 'required' and d['kind'] == 'technology']
    miss_tech = [d['skill'] for d in req_tech if d['status'] == 'missing']
    for s in miss_tech[:5]:
        out.append(Suggestion(priority='high', area='skills',
                              text=f'Add {s} if you have used it: it is a core requirement of this job and your '
                                   f'resume does not mention it. If you have not, it is a skill worth learning '
                                   f'for this role.'))
    if len(miss_tech) > 5:
        out.append(Suggestion(priority='high', area='skills',
                              text=f'Other core technologies not found: {_fmt_list(miss_tech[5:])}.'))
    for d in details:
        if d['status'] == 'related' and d['importance'] != 'soft':
            out.append(Suggestion(priority='medium', area='skills',
                                  text=f'The job asks for {d["skill"]}; your resume shows the related {d["via"]}. '
                                       f'If you have used {d["skill"]} itself, name it explicitly.'))
    comp = [d['skill'] for d in details if d['importance'] == 'required' and d['kind'] == 'competency'
            and d['status'] == 'missing']
    if comp:
        out.append(Suggestion(priority='medium', area='skills',
                              text=f'Show {_fmt_list(comp)} with a concrete example (a project, task or result), '
                                   f'not just a keyword.'))
    via = [d for d in details if d['status'] == 'matched' and d['via'] and d['importance'] != 'soft']
    if via:
        pairs = _fmt_list([f'{d["skill"]} (you list {d["via"]})' for d in via])
        out.append(Suggestion(priority='low', area='skills',
                              text=f'Name these skills directly; recruiters and keyword filters may not infer them: '
                                   f'{pairs}.'))
    pref = [d['skill'] for d in details if d['importance'] == 'preferred' and d['status'] == 'missing']
    if pref:
        out.append(Suggestion(priority='low', area='skills',
                              text=f'Nice-to-have skills you could add if true: {_fmt_list(pref)}.'))
    soft = [d['skill'] for d in details if d['importance'] == 'soft' and d['status'] == 'missing']
    if soft:
        out.append(Suggestion(priority='low', area='skills',
                              text=f'The job mentions {_fmt_list(soft)}; show them through an achievement '
                                   f'(e.g. "led a team of 4", "presented results to clients").'))
    return out


def _experience_suggestion(row) -> Suggestion | None:
    need, have = row.years_required, row.years_found
    need = None if need is None or pd.isna(need) else float(need)
    have = None if have is None or pd.isna(have) else float(have)
    if need is not None and have is None:
        return Suggestion(priority='high', area='experience',
                          text=f'The job asks for {_yrs(need)}+ years of experience, but no years or work dates '
                               f'were found. State your total experience ("{_yrs(need)}+ years of ...") and add '
                               f'start-end dates to each job (e.g. "Jan 2020 - Mar 2023").')
    if need is not None and have < need:
        return Suggestion(priority='medium', area='experience',
                          text=f'The job asks for {_yrs(need)}+ years; your resume shows {_yrs(have)}. Include all '
                               f'relevant experience with dates (internships, freelance, long projects), or '
                               f'expect this to be a gap.')
    if need is None and have is None:
        return Suggestion(priority='low', area='experience',
                          text='Add start-end dates to each job so your total experience is clear.')
    return None


def build_advice(row, n: int, keywords: list[str], resume_clean: str) -> tuple[list[str], list[Suggestion]]:
    """(strengths, suggestions) for one scored row; n = resumes in the comparison (pool + this one)."""
    details = list(row.skill_details)
    strengths: list[str] = []
    suggestions = _skill_suggestions(details)

    core = [d for d in details if d['importance'] == 'required' and d['kind'] == 'technology']
    have_core = [d['skill'] for d in core if d['status'] == 'matched']
    if have_core:
        strengths.append(f'Has {len(have_core)} of {len(core)} core technologies: {_fmt_list(have_core)}')
    comp = [d for d in details if d['importance'] == 'required' and d['kind'] == 'competency']
    have_comp = [d['skill'] for d in comp if d['status'] == 'matched']
    if have_comp:
        strengths.append(f'Shows {len(have_comp)} of {len(comp)} required competencies: {_fmt_list(have_comp)}')
    need, have = row.years_required, row.years_found
    if need is not None and not pd.isna(need) and have is not None and not pd.isna(have) and have >= need:
        strengths.append(f'Meets the experience requirement ({_yrs(have)} years, {_yrs(need)} required)')

    top = max(1, round(n * 0.25))
    if row.semantic_pos <= top:
        strengths.append(f'Content is closely related to this job ({_ordinal(row.semantic_pos)} of {n} '
                         f'on semantic similarity)')
    elif row.semantic_pos > n / 2:
        suggestions.append(Suggestion(priority='medium', area='content',
                                      text='Overall, your resume is less about this job\'s work than most comparable '
                                           'resumes. Put your most relevant roles and projects first and describe '
                                           'them in terms of the tasks this job lists.'))
    if row.keyword_pos <= top:
        strengths.append(f'Uses much of the job\'s vocabulary ({_ordinal(row.keyword_pos)} of {n} on keyword '
                         f'similarity)')
    if keywords and row.keyword_pos > top:
        suggestions.append(Suggestion(priority='medium', area='keywords',
                                      text=f'Use the job\'s own wording where it is true for you. Terms from the job '
                                           f'description missing from your resume: {", ".join(keywords[:MAX_LIST])}.'))

    exp = _experience_suggestion(row)
    if exp:
        suggestions.append(exp)
    if len(resume_clean.split()) < SHORT_RESUME_WORDS:
        suggestions.append(Suggestion(priority='medium', area='content',
                                      text=f'Your resume is short ({len(resume_clean.split())} words). Describe what '
                                           f'you did in each role, the tools you used and the results.'))
    if getattr(row, 'label_capped', False):
        suggestions.insert(0, Suggestion(priority='high', area='skills',
                                         text=f'None of the job\'s {row.n_core} core technologies were found, so the '
                                              f'label is capped at Potential Fit. Adding the ones you know matters '
                                              f'most.'))
    if not strengths:
        strengths.append('No strong matches with this job were found')
    suggestions.sort(key=lambda s: _PRIORITY[s.priority])        # stable: keeps order within a priority
    return strengths, suggestions


def _headline(label: str, rank: int, n: int, pct: float) -> str:
    lead = {'Good Fit': 'Strong match for this job.',
            'Potential Fit': 'Partial match: worth applying after closing the main gaps below.',
            'No Fit': 'Weak match for this job as the resume stands.'}[label]
    return (f'{lead} Your resume would rank {_ordinal(rank)} of {n} against the {n - 1} most similar resumes '
            f'in our database (better than {pct:.0f}% of them).')


# ---------------------------------------------------------------- main entry

def check_fit(pipe: MatcherPipeline, pool: ReferencePool, job_description: str, resume_text: str,
              pool_size: int, filename: str = '', core_guard: bool = True) -> FitCheckResponse:
    resume_clean = pipe.clean(resume_text)
    n_words = len(resume_clean.split())
    if n_words < MIN_RESUME_WORDS:
        raise RankError(f'the resume has too little text to judge ({n_words} words; at least {MIN_RESUME_WORDS} needed)')
    jd_clean = pipe.clean(job_description)
    if not jd_clean:
        raise RankError('the job description contains no usable text')
    warnings: list[str] = []
    jd_skills = sorted(pipe.jd_importance(jd_clean))
    if not jd_skills:
        warnings.append(f'No skills from the {len(pipe.skills)}-skill catalog were found in the job description; '
                        'skill gaps will be empty and the result relies on text similarity.')

    refs = pool.nearest(jd_clean, pool_size, exclude=resume_clean)
    df = add_positions(pipe.rank([resume_text] + refs, job_description, ranking='model', core_guard=core_guard))
    n = len(df)
    pos = int(np.flatnonzero(df.input_index.values == 0)[0])
    row = next(df.iloc[[pos]].itertuples(index=False))
    others = df.rank_score.drop(index=pos)
    pct = round(100.0 * float((others < row.rank_score).mean()), 1)
    rank = pos + 1

    name = guess_candidate_name(resume_text, filename or 'resume')
    cand = _candidate(row, rank, filename or 'resume', [], name, 0)
    keywords = missing_keywords(pipe, jd_clean, resume_clean, set(jd_skills))
    strengths, suggestions = build_advice(row, n, keywords, resume_clean)
    det = cand.skill_details or []
    return FitCheckResponse(
        candidate_name=name[0], name_source=name[1],
        label=cand.label, confidence=cand.confidence, label_capped=cand.label_capped,
        probabilities=cand.probabilities,
        match_score=round(100 * float(row.formula_score)),
        headline=_headline(cand.label, rank, n, pct),
        comparison=Comparison(pool_size=n - 1, rank_in_pool=rank, percentile=pct,
                              description=f'Scored together with the {n - 1} training-set resumes most similar to '
                                          f'this job, because the model judges a resume relative to other '
                                          f'applicants for the same job.'),
        score_breakdown=cand.score_breakdown, scores=cand.scores, experience=cand.experience,
        skills=SkillGap(matched=[d for d in det if d.status == 'matched'],
                        related=[d for d in det if d.status == 'related'],
                        missing=[d for d in det if d.status == 'missing']),
        missing_keywords=keywords, strengths=strengths, suggestions=suggestions,
        job=JobInfo(skills_required=jd_skills, years_required=_num_or_none(pipe.years_required(jd_clean)),
                    **_importance_lists(pipe, jd_clean)),
        warnings=warnings,
        model=f'{pipe.pipeline_file}: {pipe.stage2_name}'
              + (f', ordered by {pipe.ranker_name}' if pipe.ranker is not None else '')
              + f'; compared with {n - 1} reference resumes')
