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
