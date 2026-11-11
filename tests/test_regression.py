"""The backend must reproduce the Colab results on the official test set."""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# From artifacts/model_comparison.csv, Random Forest row
COLAB = {'accuracy': 0.583, 'macro_f1': 0.549, 'roc_auc': 0.757}


def test_reproduces_colab_test_metrics(pipe, test_df):
    df = pipe.score_pairs(test_df.resume_text, test_df.job_description_text)
    y = test_df.label.map({l: i for i, l in enumerate(pipe.labels)}).values
    proba = df[['p_no_fit', 'p_potential_fit', 'p_good_fit']].values
    pred = proba.argmax(1)
    assert accuracy_score(y, pred) == pytest.approx(COLAB['accuracy'], abs=0.001)
    assert f1_score(y, pred, average='macro') == pytest.approx(COLAB['macro_f1'], abs=0.001)
    assert roc_auc_score(y, proba, multi_class='ovr') == pytest.approx(COLAB['roc_auc'], abs=0.001)


def test_rank_single_job_matches_batch_scoring(pipe, big_job):
    """Ranking one job on its own gives the same scores as scoring it inside the full test set."""
    jd, cands = big_job
    ranked = pipe.rank(cands.resume_text.tolist(), jd)
    batch = pipe.score_pairs(cands.resume_text, [jd] * len(cands))
    np.testing.assert_allclose(ranked.sort_values('input_index').rank_score.values,
                               batch.rank_score.values, atol=1e-9)
    assert ranked.rank_score.is_monotonic_decreasing


@pytest.mark.slow
def test_live_bert_matches_colab_embeddings(pipe, test_df):
    """Batched CPU encoding here vs. per-text GPU encoding in Colab: vectors should agree closely."""
    from app.ml.pipeline import MatcherPipeline
    from tests.conftest import ROOT
    fresh = MatcherPipeline(ROOT / 'artifacts', load_bert=True)
    fresh._emb.clear()
    texts = test_df.resume_text.drop_duplicates().head(20).tolist()
    new = fresh.embed(texts)
    old = np.vstack([pipe._emb[t] for t in texts])
    cos = (new * old).sum(1)
    assert cos.min() > 0.999, cos.min()


def test_v3_backend_reproduces_training_metrics(test_df):
    """pipeline_v3.joblib scored through the API code must match what training/train.py reported."""
    from app.ml.pipeline import MatcherPipeline
    from tests.conftest import ROOT
    p3 = MatcherPipeline(ROOT / 'artifacts', pipeline_file='pipeline_v3.joblib', load_bert=False)
    df = p3.score_pairs([p3.clean(t) for t in test_df.resume_text], [p3.clean(t) for t in test_df.job_description_text])
    y = test_df.label.map({l: i for i, l in enumerate(p3.labels)}).values
    pred = df.label.map({l: i for i, l in enumerate(p3.labels)}).values
    expected = p3.metrics['test']
    assert accuracy_score(y, pred) == pytest.approx(expected['Accuracy'], abs=0.001)
    assert f1_score(y, pred, average='macro') == pytest.approx(expected['F1 (macro)'], abs=0.001)
    from training.train import ndcg10
    assert ndcg10(test_df.job_description_text.values, y, df.rank_score.values) == pytest.approx(expected['NDCG@10'], abs=0.001)
