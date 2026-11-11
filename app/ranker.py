"""Rank many resumes against one job description and build the API response."""
from __future__ import annotations

import pandas as pd

from app.config import MIN_RELIABLE_BATCH
from app.explain import add_positions, why_not_selected, why_selected
from app.ml import features as F
from app.ml.pipeline import MatcherPipeline
from app.schemas import (Candidate, Experience, JobInfo, Probabilities, RankResponse, ScoreBreakdown, Scores,
                         SkillDetail, Skipped, Summary)


class RankError(ValueError):
    """Input problem the client can fix (reported as HTTP 422)."""


def _num_or_none(v, ndigits=1):
    if v is None or pd.isna(v):
        return None
    v = round(float(v), ndigits)
    return int(v) if v.is_integer() else v


def _years_source(row) -> str | None:
    if row.years_found is None or pd.isna(row.years_found):
        return None
    dated = getattr(row, 'years_dated', None)
    if dated is not None and not pd.isna(dated) and float(dated) == float(row.years_found):
        return 'work dates'
    return 'stated'


def _unique_ids(ids: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i in ids:
        if i in seen:
            seen[i] += 1
            out.append(f'{i} ({seen[i]})')
        else:
            seen[i] = 0
            out.append(i)
    return out


def _candidate(row, rank: int, rid: str, reasons: list[str]) -> Candidate:
    return Candidate(
        rank=rank, resume_id=rid, label=row.label,
        confidence=round(float(row.confidence), 4),
        rank_score=round(float(row.rank_score), 4),
        label_capped=bool(row.label_capped),
        probabilities=Probabilities(no_fit=round(float(row.p_no_fit), 4),
                                    potential_fit=round(float(row.p_potential_fit), 4),
                                    good_fit=round(float(row.p_good_fit), 4)),
        matched_skills=list(row.matched_skills), missing_skills=list(row.missing_skills),
        experience=Experience(required_years=_num_or_none(row.years_required),
                              found_years=_num_or_none(row.years_found),
                              found_from=_years_source(row),
                              score=round(float(row.exp_score), 3)),
        scores=Scores(keyword_similarity=round(float(row.tfidf_score), 4),
                      semantic_similarity=round(float(row.bert_score), 4),
                      skill_coverage=round(float(row.w_skill_score), 4)),
        score_breakdown=ScoreBreakdown(skills=round(float(row.w_skill_score), 3),
                                       experience=round(float(row.exp_score), 3),
                                       semantic=round(float(row.semantic_scaled), 3),
                                       keyword=round(float(row.keyword_scaled), 3),
                                       formula_score=round(float(row.formula_score), 4),
                                       model_score=round(float(row.model_score), 4)),
        skill_details=[SkillDetail(**{k: d[k] for k in ('skill', 'importance', 'kind', 'status', 'via')})
                       for d in row.skill_details],
        reasons=reasons)


def _importance_lists(pipe: MatcherPipeline, jd_clean: str) -> dict:
    imp = pipe.jd_importance(jd_clean)
    out = {f'{k}_skills': sorted(s for s, (i, _) in imp.items() if i == k) for k in ('required', 'preferred', 'soft')}
    for kind, key in (('technology', 'required_technologies'), ('competency', 'required_competencies')):
        out[key] = sorted(s for s, (i, _) in imp.items() if i == 'required' and F.skill_kind(s) == kind)
    return out


def rank_resumes(pipe: MatcherPipeline, job_description: str,
                 resumes: list[tuple[str, str]], top_k: int,
                 skipped: list[Skipped] | None = None, total_received: int | None = None,
                 ranking: str = 'model', core_guard: bool = True) -> RankResponse:
    """resumes: list of (resume_id, raw_text). Resumes that fail parsing go in `skipped`."""
    skipped = list(skipped or [])
    if total_received is None:
        total_received = len(resumes) + len(skipped)
    warnings: list[str] = []

    ids = _unique_ids([rid for rid, _ in resumes])
    usable = []
    for rid, text in zip(ids, (t for _, t in resumes)):
        if pipe.clean(text):
            usable.append((rid, text))
        else:
            skipped.append(Skipped(resume_id=rid, reason='empty text'))
    if not usable:
        raise RankError('none of the resumes contained usable text')

    jd_clean = pipe.clean(job_description)
    jd_skills = sorted(pipe.jd_importance(jd_clean))
    if not jd_skills:
        warnings.append(f'No skills from the {len(pipe.skills)}-skill catalog were found in the job description; '
                        'skill matching and skill explanations will be empty.')
    n = len(usable)
    if n < MIN_RELIABLE_BATCH:
        warnings.append(f'Only {n} resume(s) scored. The model compares candidates with each other '
                        f'and was trained with ~22 per job; rankings are less reliable below '
                        f'{MIN_RELIABLE_BATCH}.')
    k = min(top_k, n)
    if top_k > n:
        warnings.append(f'top_k={top_k} is larger than the number of scored resumes ({n}).')

    df = pipe.rank([t for _, t in usable], job_description, ranking=ranking, core_guard=core_guard)
    df = add_positions(df)
    df['resume_id'] = [usable[i][0] for i in df.input_index]
    cutoff = float(df.rank_score.iloc[k - 1])

    top, rest = [], []
    for pos, row in enumerate(df.itertuples(index=False), start=1):
        if pos <= k:
            top.append(_candidate(row, pos, row.resume_id, why_selected(row, n)))
        else:
            rest.append(_candidate(row, pos, row.resume_id, why_not_selected(row, n, k)))

    counts = df.label.value_counts()
    return RankResponse(
        job=JobInfo(skills_required=jd_skills, years_required=_num_or_none(pipe.years_required(jd_clean)),
                    **_importance_lists(pipe, jd_clean)),
        summary=Summary(total_received=total_received,
                        scored=n, skipped=len(skipped), top_k=k, shortlist_cutoff=round(cutoff, 4),
                        ranked_by=ranking, core_guard=core_guard,
                        label_counts={lab: int(counts.get(lab, 0)) for lab in pipe.labels}),
        top_k=top, not_selected=rest, skipped=skipped, warnings=warnings,
        model=(f'{pipe.pipeline_file}: TF-IDF + {pipe.bert_model_name} stage-1 Logistic Regressions -> '
               f'{pipe.stage2_name}' + (f', ordered by {pipe.ranker_name}' if pipe.ranker is not None else '')))
