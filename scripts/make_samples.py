"""Write a demo job + 100 resume files from the official TEST split (jobs the model never saw).

samples/<name>/job_description.txt
samples/<name>/resumes/resume_001.txt ...
samples/<name>/true_labels.csv     (dataset label for this job, or "other job" for filler resumes)

Usage: python scripts/make_samples.py [--n 100] [--job-index 0]
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--job-index', type=int, default=0, help='0 = test job with the most candidates')
args = ap.parse_args()

df = pd.read_parquet(ROOT / 'artifacts' / 'test_features.parquet')
jd = df.job_description_text.value_counts().index[args.job_index]
cands = df[df.job_description_text == jd].drop_duplicates('resume_text')
rows = list(zip(cands.resume_text, cands.label))
if len(rows) < args.n:   # top up with resumes that were labelled for other jobs
    extra = df[~df.resume_text.isin(cands.resume_text)].drop_duplicates('resume_text')
    rows += [(t, 'other job (no label)') for t in extra.resume_text.sample(args.n - len(rows), random_state=0)]
rows = rows[:args.n]

out = ROOT / 'samples' / f'test_job_{args.job_index}'
(out / 'resumes').mkdir(parents=True, exist_ok=True)
(out / 'job_description.txt').write_text(jd)
with open(out / 'true_labels.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['file', 'true_label'])
    for i, (text, label) in enumerate(rows, start=1):
        name = f'resume_{i:03d}.txt'
        (out / 'resumes' / name).write_text(text)
        w.writerow([name, label])
print(f'wrote {len(rows)} resumes ({len(cands)} labelled for this job) to {out}')
