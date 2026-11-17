"""ResumeFit API: rank many resumes against one job description (recruiter mode),
or check how well one resume fits one job (candidate mode, /fit-check)."""
from __future__ import annotations

import csv
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.fit_check import ReferencePool, check_fit
from app.ml.pipeline import MatcherPipeline
from app.parsing import ParseError, extract_text
from app.ranker import RankError, build_response, score_batch
from app.schemas import (CORE_GUARD_HELP, RANKING_DEFAULT_HELP, Filters, FitCheckRequest, FitCheckResponse, Ranking,
                         RankRequest, RankResponse, RerankRequest, Skipped, Weights)
from app.sessions import SessionStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.time()
    app.state.pipe = MatcherPipeline(config.ARTIFACTS_DIR, pipeline_file=config.PIPELINE_FILE,
                                     embedding_cache_size=config.EMBEDDING_CACHE_SIZE,
                                     load_bert=config.LOAD_BERT_AT_STARTUP)
    app.state.load_seconds = round(time.time() - t, 2)
    app.state.pool = ReferencePool(app.state.pipe, config.REFERENCE_RESUMES)   # loaded on first /fit-check
    app.state.sessions = SessionStore(config.SESSION_MAX, config.SESSION_TTL_SECONDS)
    yield


app = FastAPI(title='ResumeFit API', version='1.0.0', lifespan=lifespan,
              description='Hybrid resume-job matcher: TF-IDF + BERT + skills, two-stage model. '
                          'Recruiters: send one job description and many resumes; get the top k with reasons. '
                          'Candidates: send one resume and one job description to /fit-check; get a fit label, '
                          'skill gaps and suggestions.')
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                   allow_methods=['GET', 'POST'], allow_headers=['*'])


def _pipe(request: Request) -> MatcherPipeline:
    return request.app.state.pipe


def _check_count(n: int):
    if n > config.MAX_RESUMES:
        raise HTTPException(413, f'too many resumes: {n} (max {config.MAX_RESUMES})')


def _read_upload(f: UploadFile, what: str) -> str:
    data = f.file.read(config.MAX_FILE_BYTES + 1)
    if len(data) > config.MAX_FILE_BYTES:
        raise HTTPException(413, f'{what} file larger than {config.MAX_FILE_BYTES // 2**20} MB')
    try:
        return extract_text(f.filename, data)
    except ParseError as e:
        raise HTTPException(422, f'{what} file: {e}')


def _job_description(text: str, file: UploadFile | None) -> str:
    """Job description from form text, an uploaded file, or both (joined)."""
    parts = [text.strip()] if text.strip() else []
    if file is not None and file.filename:
        parts.append(_read_upload(file, 'job description'))
    if not parts:
        raise HTTPException(422, 'send the job description as text (job_description) or as a file (jd_file)')
    return '\n\n'.join(parts)


@app.get('/health')
def health(request: Request):
    pipe = _pipe(request)
    return {'status': 'ok', 'bert_loaded': pipe.bert_loaded,
            'embedding_cache': pipe.cache_size, 'startup_seconds': request.app.state.load_seconds}


@app.get('/model-info')
def model_info(request: Request):
    pipe = _pipe(request)
    comparison = pipe.metrics.get('model_comparison')
    if comparison is None:
        f = config.ARTIFACTS_DIR / 'model_comparison.csv'
        comparison = []
        if f.exists():
            with f.open() as fh:
                comparison = [{k: (float(v) if k != 'model' else v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return {
        'pipeline_file': pipe.pipeline_file,
        'version': pipe.version,
        'final_model': pipe.stage2_name,
        'ranking_model': pipe.ranker_name,
        'label_rule': ('thresholds on the ranking score, tuned by cross-validation' if pipe.thresholds is not None
                       else 'most probable class'),
        'stage1': ['Logistic Regression on TF-IDF pairwise features (product + |difference|)',
                   f'Logistic Regression on {pipe.bert_model_name} pairwise features'],
        'embedding_model': pipe.bert_model_name,
        'labels': pipe.labels,
        'n_skills_in_catalog': len(pipe.skills),
        'n_stage2_features': int(pipe.stage2.n_features_in_),
        'ranking_score': (pipe.ranker_name if pipe.ranker is not None else 'P(Good Fit) + 0.5 * P(Potential Fit)'),
        'test_set_results': comparison,
        'final_test_metrics': pipe.metrics.get('test'),
        'ablation': pipe.metrics.get('ablation'),
        'notes': ['Evaluated on job descriptions never seen in training (official split).',
                  'Rank with 10+ resumes per job; the model compares candidates with each other.'],
    }


def _rank(request: Request, jd: str, resumes, top_k: int, ranking, core_guard: bool,
          weights: Weights | None, filters: Filters | None, **batch_kw) -> RankResponse:
    """Score once, keep the batch for /rerank, build the response."""
    pipe = _pipe(request)
    try:
        batch = score_batch(pipe, jd, resumes, **batch_kw)
        resp = build_response(pipe, batch, top_k, ranking, core_guard, weights, filters)
    except RankError as e:
        raise HTTPException(422, str(e))
    resp.session_id = request.app.state.sessions.put(batch)
    return resp


@app.post('/rank', response_model=RankResponse)
def rank_json(body: RankRequest, request: Request):
    """Rank resumes given as plain text. Optional custom formula `weights` and must-have `filters`."""
    _check_count(len(body.resumes))
    resumes = [(r.id or f'resume_{i}', r.text) for i, r in enumerate(body.resumes, start=1)]
    return _rank(request, body.job_description, resumes, body.top_k, body.ranking, body.core_guard,
                 body.weights, body.filters)


def _form_options(weight_skills, weight_experience, weight_semantic, weight_keyword,
                  min_years, unknown_years, must_have) -> tuple[Weights | None, Filters | None]:
    w = {'skills': weight_skills, 'experience': weight_experience, 'semantic': weight_semantic,
         'keyword': weight_keyword}
    weights = Weights(**w) if any(v is not None for v in w.values()) else None
    skills = [x.strip() for x in (must_have or '').split(',') if x.strip()]
    if len(skills) > 20:
        raise HTTPException(422, 'at most 20 must-have skills')
    filters = Filters(min_years=min_years, unknown_years=unknown_years, must_have_skills=skills)
    return weights, (filters if filters.active else None)


@app.post('/rank/upload', response_model=RankResponse)
def rank_upload(request: Request,
                files: list[UploadFile] = File(..., description='Resumes as PDF, DOCX or TXT'),
                job_description: str = Form('', description='Job description as text (or send jd_file)'),
                jd_file: UploadFile | None = File(None, description='Job description as a PDF, DOCX or TXT file'),
                top_k: int = Form(config.DEFAULT_TOP_K, ge=1),
                ranking: Ranking | None = Form(None, description=RANKING_DEFAULT_HELP),
                core_guard: bool = Form(True, description=CORE_GUARD_HELP),
                weight_skills: float | None = Form(None, ge=0, description='formula weight (default 0.40)'),
                weight_experience: float | None = Form(None, ge=0, description='formula weight (default 0.25)'),
                weight_semantic: float | None = Form(None, ge=0, description='formula weight (default 0.20)'),
                weight_keyword: float | None = Form(None, ge=0, description='formula weight (default 0.15)'),
                min_years: float | None = Form(None, ge=0, le=60, description='filter: minimum years of experience'),
                unknown_years: Literal['keep', 'exclude'] = Form('keep', description='with min_years: resumes '
                                                                 'with no years found are kept or excluded'),
                must_have: str = Form('', description='filter: comma-separated skills, e.g. "python, aws"')):
    """Rank uploaded resume files. The job description can be text, a file, or both (joined).
    Unreadable resumes are listed under `skipped`. Use the returned `session_id` with /rerank."""
    _check_count(len(files))
    weights, filters = _form_options(weight_skills, weight_experience, weight_semantic, weight_keyword,
                                     min_years, unknown_years, must_have)
    jd = _job_description(job_description, jd_file)
    resumes, indices, skipped = [], [], []
    for i, f in enumerate(files):
        name = f.filename or f'file_{i + 1}'
        data = f.file.read(config.MAX_FILE_BYTES + 1)
        if len(data) > config.MAX_FILE_BYTES:
            skipped.append(Skipped(resume_id=name, reason=f'file larger than {config.MAX_FILE_BYTES // 2**20} MB'))
            continue
        try:
            resumes.append((name, extract_text(name, data)))
            indices.append(i)
        except ParseError as e:
            skipped.append(Skipped(resume_id=name, reason=str(e)))
    if not resumes:
        raise HTTPException(422, {'message': 'no readable resumes', 'skipped': [s.model_dump() for s in skipped]})
    return _rank(request, jd, resumes, top_k, ranking, core_guard, weights, filters,
                 skipped=skipped, total_received=len(files), indices=indices)


@app.post('/rerank', response_model=RankResponse)
def rerank(body: RerankRequest, request: Request):
    """Re-rank the resumes of an earlier /rank or /rank/upload call with new weights, filters, top_k,
    ranking or core_guard. Instant: nothing is uploaded or re-scored (no BERT, no model).
    Returns 404 if the session expired; then send the resumes again."""
    batch = request.app.state.sessions.get(body.session_id)
    if batch is None:
        raise HTTPException(404, 'session not found or expired; rank the resumes again')
    try:
        resp = build_response(_pipe(request), batch, body.top_k, body.ranking, body.core_guard,
                              body.weights, body.filters)
    except RankError as e:
        raise HTTPException(422, str(e))
    resp.session_id = body.session_id
    return resp


# ---------------------------------------------------------------- candidate mode

def _fit(request: Request, jd: str, resume: str, filename: str, core_guard: bool) -> FitCheckResponse:
    try:
        return check_fit(_pipe(request), request.app.state.pool, jd, resume, config.FIT_POOL_SIZE,
                         filename=filename, core_guard=core_guard)
    except RankError as e:
        raise HTTPException(422, str(e))


@app.post('/fit-check', response_model=FitCheckResponse)
def fit_check_json(body: FitCheckRequest, request: Request):
    """Candidate mode with plain text: how well does this resume fit this job, and what should change?"""
    return _fit(request, body.job_description, body.resume_text, '', body.core_guard)


@app.post('/fit-check/upload', response_model=FitCheckResponse)
def fit_check_upload(request: Request,
                     resume: UploadFile = File(..., description='Your resume as PDF, DOCX or TXT'),
                     job_description: str = Form('', description='Job description as text (or send jd_file)'),
                     jd_file: UploadFile | None = File(None, description='Job description as a PDF, DOCX or TXT file'),
                     core_guard: bool = Form(True, description=CORE_GUARD_HELP)):
    """Candidate mode with files: one resume file + the job description as text and/or a file.
    Returns a fit label, a 0-100 match score, the position among similar resumes, skill gaps,
    missing keywords, strengths and prioritised suggestions."""
    jd = _job_description(job_description, jd_file)
    text = _read_upload(resume, 'resume')
    return _fit(request, jd, text, resume.filename or 'resume', core_guard)
