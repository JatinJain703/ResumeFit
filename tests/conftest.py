import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Dataset texts are already in artifacts/embeddings.joblib, so most tests run without BERT.
os.environ.setdefault('RESUMEFIT_LAZY_BERT', '1')

from app.ml.pipeline import MatcherPipeline  # noqa: E402


@pytest.fixture(scope='session')
def pipe():
    return MatcherPipeline(ROOT / 'artifacts', load_bert=False)


@pytest.fixture(scope='session')
def test_df():
    return pd.read_parquet(ROOT / 'artifacts' / 'test_features.parquet')


@pytest.fixture(scope='session')
def big_job(test_df):
    """The test job with the most candidates: (job_text, candidates_df)."""
    jd = test_df.job_description_text.value_counts().index[0]
    return jd, test_df[test_df.job_description_text == jd].reset_index(drop=True)
