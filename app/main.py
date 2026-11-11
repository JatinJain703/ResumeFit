"""ResumeFit API: rank many resumes against one job description."""
from __future__ import annotations

import csv
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from app import config
from app.ml.pipeline import MatcherPipeline
from app.parsing import ParseError, extract_text
from app.ranker import RankError, rank_resumes
from app.schemas import CORE_GUARD_HELP, RANKING_HELP, Ranking, RankRequest, RankResponse, Skipped


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.time()
    app.state.pipe = MatcherPipeline(config.ARTIFACTS_DIR, pipeline_file=config.PIPELINE_FILE,
                                     embedding_cache_size=config.EMBEDDING_CACHE_SIZE,
                                     load_bert=config.LOAD_BERT_AT_STARTUP)
    app.state.load_seconds = round(time.time() - t, 2)
    yield


app = FastAPI(title='ResumeFit API', version='1.0.0', lifespan=lifespan,
              description='Hybrid resume-job matcher: TF-IDF + BERT + skills, two-stage model. '
                          'Send one job description and many resumes; get the top k with reasons.')


def _pipe(request: Request) -> MatcherPipeline:
    return request.app.state.pipe


def _check_count(n: int):
    if n > config.MAX_RESUMES:
        raise HTTPException(413, f'too many resumes: {n} (max {config.MAX_RESUMES})')


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


@app.post('/rank', response_model=RankResponse)
def rank_json(body: RankRequest, request: Request):
    """Rank resumes given as plain text."""
    _check_count(len(body.resumes))
    resumes = [(r.id or f'resume_{i}', r.text) for i, r in enumerate(body.resumes, start=1)]
    try:
        return rank_resumes(_pipe(request), body.job_description, resumes, body.top_k,
                            ranking=body.ranking, core_guard=body.core_guard)
    except RankError as e:
        raise HTTPException(422, str(e))


@app.post('/rank/upload', response_model=RankResponse)
def rank_upload(request: Request,
                job_description: str = Form(..., min_length=1),
                files: list[UploadFile] = File(..., description='Resumes as PDF, DOCX or TXT'),
                top_k: int = Form(config.DEFAULT_TOP_K, ge=1),
                ranking: Ranking = Form('model', description=RANKING_HELP),
                core_guard: bool = Form(True, description=CORE_GUARD_HELP)):
    """Rank uploaded resume files. Unreadable files are listed under `skipped`."""
    _check_count(len(files))
    resumes, skipped = [], []
    for i, f in enumerate(files, start=1):
        name = f.filename or f'file_{i}'
        data = f.file.read(config.MAX_FILE_BYTES + 1)
        if len(data) > config.MAX_FILE_BYTES:
            skipped.append(Skipped(resume_id=name, reason=f'file larger than {config.MAX_FILE_BYTES // 2**20} MB'))
            continue
        try:
            resumes.append((name, extract_text(name, data)))
        except ParseError as e:
            skipped.append(Skipped(resume_id=name, reason=str(e)))
    if not resumes:
        raise HTTPException(422, {'message': 'no readable resumes', 'skipped': [s.model_dump() for s in skipped]})
    try:
        return rank_resumes(_pipe(request), job_description, resumes, top_k,
                            skipped=skipped, total_received=len(files), ranking=ranking, core_guard=core_guard)
    except RankError as e:
        raise HTTPException(422, str(e))
