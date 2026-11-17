import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.getenv('RESUMEFIT_ARTIFACTS', ROOT / 'artifacts'))
# pipeline_v3.joblib = retrained locally (training/train.py); pipeline_v2.joblib = original Colab model
PIPELINE_FILE = os.getenv('RESUMEFIT_PIPELINE') or (
    'pipeline_v3.joblib' if (ARTIFACTS_DIR / 'pipeline_v3.joblib').exists() else 'pipeline_v2.joblib')

DEFAULT_TOP_K = 10
MAX_RESUMES = int(os.getenv('RESUMEFIT_MAX_RESUMES', 500))
MAX_FILE_BYTES = int(os.getenv('RESUMEFIT_MAX_FILE_MB', 5)) * 1024 * 1024
MIN_RELIABLE_BATCH = 10          # within-job features were trained on ~22 candidates per job
EMBEDDING_CACHE_SIZE = int(os.getenv('RESUMEFIT_EMBEDDING_CACHE', 20000))
LOAD_BERT_AT_STARTUP = os.getenv('RESUMEFIT_LAZY_BERT', '0') != '1'
# Browser origins allowed to call the API (comma-separated). Default: common local frontend dev servers.
CORS_ORIGINS = [o.strip() for o in os.getenv(
    'RESUMEFIT_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000'
).split(',') if o.strip()]
# Candidate mode (/fit-check): one resume is scored together with this many reference resumes from the
# training set (those most similar to the job), because the model's features compare applicants for a job.
FIT_POOL_SIZE = int(os.getenv('RESUMEFIT_FIT_POOL', 24))
REFERENCE_RESUMES = ARTIFACTS_DIR / 'train_features.parquet'
# /rerank: scored batches kept in memory so weights/filters can change without re-uploading
SESSION_MAX = int(os.getenv('RESUMEFIT_SESSIONS', 20))
SESSION_TTL_SECONDS = int(os.getenv('RESUMEFIT_SESSION_TTL', 3600))
