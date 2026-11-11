"""Evaluate ranking settings with NDCG@10, Precision@10 and Recall@10 (plus label accuracy / macro-F1).

Sources
  --dataset                 the official test split (71 unseen jobs, dataset labels)
  --folders DIR [DIR ...]   human-labelled jobs, one folder per job:
                              DIR/job_description.txt
                              DIR/resumes/<file>.pdf|.docx|.txt
                              DIR/labels.csv   columns: file,label   (label = No Fit | Potential Fit | Good Fit)
                            Rows with any other label (e.g. blank) are scored but not evaluated.

Create a blank labels.csv to fill in:   python scripts/evaluate.py --init DIR
Examples:
  python scripts/evaluate.py --dataset
  python scripts/evaluate.py --folders labelled/job_*
  python scripts/evaluate.py --folders labelled/job_* --relevant potential   # count Potential Fit as relevant too
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, ndcg_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.ml.pipeline import MatcherPipeline, finalize  # noqa: E402
from app.parsing import SUPPORTED, ParseError, extract_text  # noqa: E402

LABELS = {'No Fit': 0, 'Potential Fit': 1, 'Good Fit': 2}
K = 10
SETTINGS = [('model', True), ('model', False), ('formula', True), ('formula', False)]


def job_metrics(y: np.ndarray, score: np.ndarray, min_rel: int) -> dict:
    order = np.argsort(-score, kind='stable')
    k = min(K, len(y))
    rel = y >= min_rel
    top = rel[order[:k]]
    return {'NDCG@10': ndcg_score([y], [score], k=K) if len(y) > 1 else np.nan,
            'P@10': top.mean(),
            'R@10': top.sum() / rel.sum() if rel.sum() else np.nan}


def evaluate(df: pd.DataFrame, min_rel: int, pipelines: dict) -> pd.DataFrame:
    """df: resume_text, job_description_text (raw), y (int or -1 for unlabelled)."""
    rows = []
    for pname, pipe in pipelines.items():
        scored = pipe.score_pairs([pipe.clean(t) for t in df.resume_text],
                                  [pipe.clean(t) for t in df.job_description_text])
        scored['y'] = df.y.values
        scored['job'] = df.job.values
        for ranking, guard in SETTINGS:
            d = finalize(scored, ranking, guard)
            lab = d[d.y >= 0]
            per_job = [job_metrics(g.y.values, g.rank_score.values, min_rel)
                       for _, g in lab.groupby('job') if len(g) > 1]
            pred = lab.label.map(LABELS).values
            rows.append({'pipeline': pname, 'ranking': ranking, 'core_guard': guard,
                         **pd.DataFrame(per_job).mean().round(3).to_dict(),
                         'label acc': round(accuracy_score(lab.y, pred), 3),
                         'label macro-F1': round(f1_score(lab.y, pred, average='macro'), 3),
                         'jobs': lab.job.nunique(), 'labelled resumes': len(lab)})
    return pd.DataFrame(rows)


def load_folders(dirs) -> pd.DataFrame:
    rows = []
    for d in map(Path, dirs):
        jd = (d / 'job_description.txt').read_text()
        lab_file = next((f for f in (d / 'labels.csv', d / 'true_labels.csv') if f.exists()), None)
        labels = {}
        if lab_file:
            with open(lab_file) as fh:
                labels = {r['file']: r.get('label') or r.get('true_label') for r in csv.DictReader(fh)}
        for f in sorted((d / 'resumes').iterdir()):
            if f.suffix.lower() not in SUPPORTED:
                continue
            try:
                text = extract_text(f.name, f.read_bytes())
            except ParseError as e:
                print(f'skip {f}: {e}')
                continue
            rows.append({'job': str(d), 'job_description_text': jd, 'resume_text': text,
                         'y': LABELS.get((labels.get(f.name) or '').strip(), -1)})
    return pd.DataFrame(rows)


def init_labels(d: Path):
    out = d / 'labels.csv'
    if out.exists():
        sys.exit(f'{out} already exists')
    files = sorted(f.name for f in (d / 'resumes').iterdir() if f.suffix.lower() in SUPPORTED)
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'label'])
        w.writerows([f, ''] for f in files)
    print(f'wrote {out} with {len(files)} rows; fill label with No Fit / Potential Fit / Good Fit')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', action='store_true')
    ap.add_argument('--folders', nargs='+')
    ap.add_argument('--init', type=Path, help='write a blank labels.csv for a job folder')
    ap.add_argument('--relevant', choices=['good', 'potential'], default='good',
                    help='what counts as relevant for P@10/R@10: Good Fit only (default) or Potential Fit and above')
    ap.add_argument('--pipelines', nargs='+', default=['pipeline_v3.joblib'],
                    help='artifact files to compare, e.g. pipeline_v2.joblib pipeline_v3.joblib')
    a = ap.parse_args()
    if a.init:
        return init_labels(a.init)
    if not (a.dataset or a.folders):
        ap.error('give --dataset and/or --folders')
    pipes = {p: MatcherPipeline(ROOT / 'artifacts', pipeline_file=p, load_bert=bool(a.folders)) for p in a.pipelines}
    min_rel = 2 if a.relevant == 'good' else 1
    pd.set_option('display.width', 200)
    for name, df in ([('official test split (dataset labels)', None)] if a.dataset else []) + \
                    ([('human-labelled folders', None)] if a.folders else []):
        if name.startswith('official'):
            t = pd.read_parquet(ROOT / 'artifacts' / 'test_features.parquet')
            df = pd.DataFrame({'job': t.job_description_text, 'job_description_text': t.job_description_text,
                               'resume_text': t.resume_text, 'y': t.label.map(LABELS)})
        else:
            df = load_folders(a.folders)
        relevant = 'Good Fit' if min_rel == 2 else 'Potential Fit or better'
        print(f'\n=== {name} - relevant = {relevant}')
        print(evaluate(df, min_rel, pipes).to_string(index=False))


if __name__ == '__main__':
    main()
