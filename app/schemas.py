from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import DEFAULT_TOP_K

Label = Literal['No Fit', 'Potential Fit', 'Good Fit']


class ResumeIn(BaseModel):
    id: str | None = Field(None, description='Your identifier; defaults to resume_<n>')
    text: str = Field(..., min_length=1)


Ranking = Literal['model', 'formula']
RANKING_HELP = ('"model" (default): trained ranker, best measured accuracy. '
                '"formula": 0.40*skills + 0.25*experience + 0.20*semantic + 0.15*keyword, fully transparent.')
CORE_GUARD_HELP = ('Never label a resume Good Fit if it matches none of the job\'s core technologies '
                   '(when the job lists 3 or more). Does not change the order.')


class RankRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    resumes: list[ResumeIn] = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1)
    ranking: Ranking = Field('model', description=RANKING_HELP)
    core_guard: bool = Field(True, description=CORE_GUARD_HELP)


class Experience(BaseModel):
    required_years: float | None
    found_years: float | None
    found_from: Literal['stated', 'work dates'] | None = Field(
        None, description='"stated" = an "N years" phrase, "work dates" = summed job date ranges')
    score: float


class Scores(BaseModel):
    keyword_similarity: float = Field(..., description='TF-IDF cosine')
    semantic_similarity: float = Field(..., description='BERT cosine')
    skill_coverage: float = Field(..., description='weighted share of the job\'s skills found (required > preferred > soft)')


class SkillDetail(BaseModel):
    skill: str
    importance: Literal['required', 'preferred', 'soft']
    kind: Literal['technology', 'competency', 'soft']
    status: Literal['matched', 'related', 'missing']
    via: str | None = Field(None, description='resume skill that provides the evidence, e.g. aws via "redshift"')


class ScoreBreakdown(BaseModel):
    """Readable components, each 0-1 (semantic/keyword scaled within this request), plus both overall scores.
    The list is sorted by `rank_score`, which equals `formula_score` or `model_score` depending on `ranking`."""
    skills: float = Field(..., description='weighted skill coverage (core technologies > competencies/preferred > soft)')
    experience: float
    semantic: float
    keyword: float
    formula_score: float = Field(..., description='0.40*skills + 0.25*experience + 0.20*semantic + 0.15*keyword')
    model_score: float = Field(..., description='trained ranker score (relative; compare within one request only)')


class Probabilities(BaseModel):
    no_fit: float
    potential_fit: float
    good_fit: float


class Candidate(BaseModel):
    rank: int
    resume_id: str
    label: Label
    confidence: float = Field(..., description='model probability of the assigned label')
    rank_score: float = Field(..., description='the exact score this list is sorted by (see summary.ranked_by)')
    label_capped: bool = Field(False, description='label lowered to Potential Fit by core_guard')
    probabilities: Probabilities
    matched_skills: list[str]
    missing_skills: list[str]
    experience: Experience
    scores: Scores
    score_breakdown: ScoreBreakdown
    skill_details: list[SkillDetail] | None = None
    reasons: list[str]


class JobInfo(BaseModel):
    skills_required: list[str] = Field(..., description='all catalog skills found in the job description')
    required_skills: list[str] | None = None
    required_technologies: list[str] | None = None
    required_competencies: list[str] | None = None
    preferred_skills: list[str] | None = None
    soft_skills: list[str] | None = None
    years_required: float | None


class Summary(BaseModel):
    total_received: int
    scored: int
    skipped: int
    top_k: int
    shortlist_cutoff: float
    ranked_by: Ranking
    core_guard: bool
    label_counts: dict[str, int]


class Skipped(BaseModel):
    resume_id: str
    reason: str


class RankResponse(BaseModel):
    job: JobInfo
    summary: Summary
    top_k: list[Candidate]
    not_selected: list[Candidate]
    skipped: list[Skipped]
    warnings: list[str]
    model: str
