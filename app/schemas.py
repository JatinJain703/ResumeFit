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


RANKING_DEFAULT_HELP = RANKING_HELP + ' Default: "formula" when custom weights are sent, else "model".'


class Weights(BaseModel):
    """Formula weights. Any non-negative numbers (e.g. slider values 0-100); they are scaled to sum to 1.
    Omitted ones keep their default."""
    skills: float | None = Field(None, ge=0, description='default 0.40')
    experience: float | None = Field(None, ge=0, description='default 0.25')
    semantic: float | None = Field(None, ge=0, description='default 0.20')
    keyword: float | None = Field(None, ge=0, description='default 0.15')


class Filters(BaseModel):
    """Must-have rules. Resumes that fail any are moved to `filtered_out`; scores are not changed."""
    min_years: float | None = Field(None, ge=0, le=60, description='minimum years of experience found in the resume')
    unknown_years: Literal['keep', 'exclude'] = Field(
        'keep', description='with min_years: keep or exclude resumes where no years could be found')
    must_have_skills: list[str] = Field(default_factory=list, max_length=20,
                                        description='every one must be in the resume, e.g. ["python", "aws"]')

    @property
    def active(self) -> bool:
        return self.min_years is not None or bool(self.must_have_skills)


class RankRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    resumes: list[ResumeIn] = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1)
    ranking: Ranking | None = Field(None, description=RANKING_DEFAULT_HELP)
    core_guard: bool = Field(True, description=CORE_GUARD_HELP)
    weights: Weights | None = None
    filters: Filters | None = None


class RerankRequest(BaseModel):
    """Re-rank the resumes of an earlier /rank or /rank/upload call with new options. No re-upload, no re-scoring."""
    session_id: str
    top_k: int = Field(DEFAULT_TOP_K, ge=1)
    ranking: Ranking | None = Field(None, description=RANKING_DEFAULT_HELP)
    core_guard: bool = Field(True, description=CORE_GUARD_HELP)
    weights: Weights | None = None
    filters: Filters | None = None


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


class MustHaveCheck(BaseModel):
    skill: str = Field(..., description='the must-have as resolved (catalog name, or your text)')
    found: bool
    via: str | None = Field(None, description='more specific skill that satisfies it, e.g. aws via "redshift"')
    matched_as: Literal['catalog', 'text'] = Field(
        ..., description='"catalog": skill catalog with aliases; "text": not in the catalog, plain text search')


class Candidate(BaseModel):
    rank: int
    resume_id: str
    candidate_name: str = Field(..., description='name guessed from the first lines of the resume, else from the file name')
    name_source: Literal['resume', 'filename'] = 'filename'
    upload_index: int = Field(..., description='0-based position of this resume in the request (files / resumes list)')
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
    must_have: list[MustHaveCheck] | None = Field(None, description='one entry per must-have skill filter')
    passed_filters: bool = True
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
    shortlist_cutoff: float | None = Field(..., description='rank_score of the last shortlisted resume '
                                                         '(null if none passes the filters)')
    ranked_by: Ranking
    core_guard: bool
    weights: dict[str, float] = Field(..., description='formula weights used (sum to 1)')
    filters: Filters | None = None
    filtered_out: int = Field(0, description='scored resumes that failed a filter')
    label_counts: dict[str, int]


class Skipped(BaseModel):
    resume_id: str
    reason: str


class RankResponse(BaseModel):
    job: JobInfo
    summary: Summary
    top_k: list[Candidate]
    not_selected: list[Candidate]
    filtered_out: list[Candidate] = Field(default_factory=list,
                                          description='failed a filter; reasons say which (sorted by score)')
    skipped: list[Skipped]
    warnings: list[str]
    model: str
    session_id: str | None = Field(None, description='send to POST /rerank to re-rank instantly with new weights/'
                                                     'filters (kept in server memory for a limited time)')


# ---------------------------------------------------------------- candidate mode: one resume vs one job

class FitCheckRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    resume_text: str = Field(..., min_length=1)
    core_guard: bool = Field(True, description=CORE_GUARD_HELP)


class Suggestion(BaseModel):
    priority: Literal['high', 'medium', 'low']
    area: Literal['skills', 'experience', 'keywords', 'content']
    text: str


class SkillGap(BaseModel):
    """The job's catalog skills split by what the resume shows."""
    matched: list[SkillDetail]
    related: list[SkillDetail] = Field(..., description='half credit: a close substitute is in the resume')
    missing: list[SkillDetail]


class Comparison(BaseModel):
    pool_size: int = Field(..., description='reference resumes this resume was compared with')
    rank_in_pool: int = Field(..., description='position among pool_size + 1 resumes (1 = best)')
    percentile: float = Field(..., description='share of reference resumes that scored lower (0-100)')
    description: str


class FitCheckResponse(BaseModel):
    candidate_name: str
    name_source: Literal['resume', 'filename']
    label: Label
    confidence: float
    label_capped: bool
    probabilities: Probabilities
    match_score: int = Field(..., description='0-100, the transparent formula score (skills, experience, '
                                              'semantic, keyword) x 100')
    headline: str
    comparison: Comparison
    score_breakdown: ScoreBreakdown
    scores: Scores
    experience: Experience
    skills: SkillGap
    missing_keywords: list[str] = Field(..., description='important words of the job description absent from the resume')
    strengths: list[str]
    suggestions: list[Suggestion] = Field(..., description='most important first')
    job: JobInfo
    warnings: list[str]
    model: str
