"""Rank many resumes against one job description and build the API response."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config import MIN_RELIABLE_BATCH
from app.explain import add_positions, why_not_selected, why_selected
from app.ml import features as F
from app.filters import failures, passed_line, resolve
from app.ml.pipeline import MatcherPipeline, finalize, normalize_weights, sort_ranked
from app.parsing import guess_candidate_name
from app.schemas import (Candidate, Experience, Filters, JobInfo, Probabilities, RankResponse, ScoreBreakdown, Scores,
                         SkillDetail, Skipped, Summary, Weights)


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


def _candidate(row, rank: int, rid: str, reasons: list[str], name: tuple[str, str], index: int) -> Candidate:
    return Candidate(
        rank=rank, resume_id=rid, candidate_name=name[0], name_source=name[1], upload_index=index,
        label=row.label,
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


@dataclass
class ScoredBatch:
    """Everything expensive about one ranking request (BERT, model), kept so /rerank can reuse it."""
    scored: pd.DataFrame                  # pipe.score output in input order, plus resume_id
    usable: list[tuple]                   # (resume_id, raw text, (name, name_source), upload index)
    job: JobInfo
    skipped: list[Skipped]
    total_received: int
    warnings: list[str]
    _skills: list[set[str]] | None = field(default=None, repr=False)

    def resume_skills(self, pipe: MatcherPipeline) -> list[set[str]]:
        """Catalog skills of each resume (explanation matcher), computed on first use."""
        if self._skills is None:
            self._skills = [pipe.explain_matcher.extract(t) for t in self.scored.resume_text]
        return self._skills


def score_batch(pipe: MatcherPipeline, job_description: str, resumes: list[tuple[str, str]],
                skipped: list[Skipped] | None = None, total_received: int | None = None,
                indices: list[int] | None = None) -> ScoredBatch:
    """resumes: list of (resume_id, raw_text). Resumes that fail parsing go in `skipped`.
    indices: position of each resume in the original request (default 0..n-1), echoed as `upload_index`."""
    skipped = list(skipped or [])
    if total_received is None:
        total_received = len(resumes) + len(skipped)
    warnings: list[str] = []

    ids = _unique_ids([rid for rid, _ in resumes])
    indices = list(range(len(resumes))) if indices is None else list(indices)
    usable = []
    for rid, (orig_id, text), idx in zip(ids, resumes, indices):
        if pipe.clean(text):
            usable.append((rid, text, guess_candidate_name(text, orig_id), idx))
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
    scored = pipe.score([u[1] for u in usable], job_description)
    scored['resume_id'] = [u[0] for u in usable]
    job = JobInfo(skills_required=jd_skills, years_required=_num_or_none(pipe.years_required(jd_clean)),
                  **_importance_lists(pipe, jd_clean))
    return ScoredBatch(scored=scored, usable=usable, job=job, skipped=skipped,
                       total_received=total_received, warnings=warnings)


def build_response(pipe: MatcherPipeline, batch: ScoredBatch, top_k: int, ranking: str | None = None,
                   core_guard: bool = True, weights: Weights | None = None,
                   filters: Filters | None = None) -> RankResponse:
    """Order, filter and explain a scored batch. Cheap: no BERT, no model calls.

    ranking None -> "formula" if custom weights are given, else "model".
    Resumes that fail a filter go to `filtered_out` (their scores are unchanged). Ranks: resumes that pass
    are 1..m in score order; filtered-out resumes continue from m+1, also in score order."""
    warnings = list(batch.warnings)
    w = weights.model_dump(exclude_none=True) if weights is not None else {}
    if ranking is None:
        ranking = 'formula' if w else 'model'
    if w and ranking == 'model':
        warnings.append('Custom weights change formula_score only; this list is ordered by the model. '
                        'Send ranking="formula" to order by your weights.')
    try:
        used_weights = normalize_weights(w)
    except ValueError as e:
        raise RankError(str(e)) from e
    if filters is not None and not filters.active:
        filters = None

    df = add_positions(sort_ranked(finalize(batch.scored, ranking, core_guard, w)))
    must, more = resolve(pipe, filters)
    warnings += more
    skills = batch.resume_skills(pipe) if must else None

    rows = []                                        # (row, checks, failure reasons) in score order
    for row in df.itertuples(index=False):
        checks = ([m.check(skills[row.input_index], batch.usable[row.input_index][1]) for m in must]
                  if must else None)
        rows.append((row, checks, failures(row, checks, filters)))
    passing = [r for r in rows if not r[2]]
    n, m = len(df), len(passing)
    k = min(top_k, m)
    if filters is not None and m == 0:
        warnings.append('No resume passes the filters; all are listed under filtered_out.')
    elif top_k > m:
        what = 'resumes that pass the filters' if filters is not None else 'scored resumes'
        warnings.append(f'top_k={top_k} is larger than the number of {what} ({m}).')

    top, rest, out = [], [], []
    for rank, (row, checks, fails) in enumerate(passing + [r for r in rows if r[2]], start=1):
        _, _, name, idx = batch.usable[row.input_index]
        if fails:
            reasons = fails + why_not_selected(row, n, k)[:-1]
        elif rank <= k:
            reasons = why_selected(row, n)
        else:
            reasons = why_not_selected(row, n, k)
        if not fails:
            line = passed_line(checks, filters)
            if line:
                reasons = reasons + [line]
        cand = _candidate(row, rank, row.resume_id, reasons, name, idx)
        cand.must_have, cand.passed_filters = checks, not fails
        (out if fails else top if rank <= k else rest).append(cand)

    counts = df.label.value_counts()
    return RankResponse(
        job=batch.job,
        summary=Summary(total_received=batch.total_received, scored=n, skipped=len(batch.skipped), top_k=k,
                        shortlist_cutoff=round(float(passing[k - 1][0].rank_score), 4) if k else None,
                        ranked_by=ranking, core_guard=core_guard,
                        weights={key: round(v, 4) for key, v in used_weights.items()},
                        filters=filters, filtered_out=len(out),
                        label_counts={lab: int(counts.get(lab, 0)) for lab in pipe.labels}),
        top_k=top, not_selected=rest, filtered_out=out, skipped=batch.skipped, warnings=warnings,
        model=(f'{pipe.pipeline_file}: TF-IDF + {pipe.bert_model_name} stage-1 Logistic Regressions -> '
               f'{pipe.stage2_name}' + (f', ordered by {pipe.ranker_name}' if pipe.ranker is not None else '')))


def rank_resumes(pipe: MatcherPipeline, job_description: str,
                 resumes: list[tuple[str, str]], top_k: int,
                 skipped: list[Skipped] | None = None, total_received: int | None = None,
                 ranking: str | None = 'model', core_guard: bool = True,
                 indices: list[int] | None = None, weights: Weights | None = None,
                 filters: Filters | None = None) -> RankResponse:
    """Score and rank in one call (score_batch + build_response)."""
    batch = score_batch(pipe, job_description, resumes, skipped, total_received, indices)
    return build_response(pipe, batch, top_k, ranking, core_guard, weights, filters)
