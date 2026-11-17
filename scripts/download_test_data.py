"""Download one job description + 50 resumes from the Hugging Face dataset as PDFs, for testing the app.

Dataset: https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit (test split by default,
i.e. jobs the model never saw during training).

By default it picks the job whose resumes have the most even mix of Good / Potential / No Fit labels,
so a good ranking is visible (Good Fits near the top).

Output (default ~/ResumeFit_test_data/job_<index>/):
  job_description.pdf      upload this as the job description (also .txt, to paste)
  resumes/01_Aarav_Mehta.pdf ... 50 resume PDFs to upload
  answer_key.csv           dataset label of each resume for this job, to compare with the app's ranking

The dataset resumes are anonymised, so by default a made-up name is written on the first line of each
resume (and used in the file name) so the app can show candidate names. Use --no-names to skip that.

Usage:
  .venv/bin/python scripts/download_test_data.py                    # 50 resumes, job with the best label mix
  .venv/bin/python scripts/download_test_data.py --job-index 3      # a specific job (0 = most resumes)
  .venv/bin/python scripts/download_test_data.py --n 30 --out ~/Desktop/rf_test --split train
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd

REPO = 'cnamuangtoun/resume-job-description-fit'

FIRST = ['Aarav', 'Priya', 'Rahul', 'Ananya', 'Vikram', 'Sneha', 'Arjun', 'Kavya', 'Rohan', 'Isha', 'Michael',
         'Sarah', 'David', 'Emily', 'James', 'Olivia', 'Daniel', 'Sophia', 'Wei', 'Mei', 'Carlos', 'Lucia',
         'Ahmed', 'Fatima', 'Kwame', 'Amara', 'Liam', 'Chloe', 'Noah', 'Hannah']
LAST = ['Mehta', 'Sharma', 'Verma', 'Iyer', 'Reddy', 'Nair', 'Gupta', 'Kapoor', 'Joshi', 'Desai', 'Johnson',
        'Williams', 'Brown', 'Miller', 'Davis', 'Garcia', 'Wilson', 'Chen', 'Wang', 'Lopez', 'Khan', 'Hassan',
        'Mensah', 'Okafor', 'Murphy', 'Martin', 'Anderson', 'Thomas', 'Lee', 'Clark']

# Core PDF fonts are Latin-1 only: map common Unicode punctuation, drop the rest.
_UNI = str.maketrans({'•': '-', '●': '-', '▪': '-', '◦': '-', '·': '-', '–': '-', '—': '-', '‘': "'", '’': "'",
                      '“': '"', '”': '"', '…': '...', ' ': ' ', '​': '', '\t': '    '})


def load_split(split: str) -> pd.DataFrame:
    """Download <split>.csv from Hugging Face (cached after the first run)."""
    from huggingface_hub import hf_hub_download
    print(f'downloading {split}.csv from huggingface.co/datasets/{REPO} ...')
    path = hf_hub_download(REPO, f'{split}.csv', repo_type='dataset')
    df = pd.read_csv(path)
    return df[['resume_text', 'job_description_text', 'label']].dropna()


def write_pdf(text: str, path: Path, title: str | None = None):
    from fpdf import FPDF
    pdf = FPDF(format='A4')
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()
    if title:
        pdf.set_font('Helvetica', 'B', 16)
        pdf.multi_cell(0, 9, title.translate(_UNI).encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(3)
    pdf.set_font('Helvetica', size=10)
    clean = str(text).translate(_UNI).encode('latin-1', 'replace').decode('latin-1')
    for para in clean.splitlines() or ['']:
        # wrapmode CHAR so long unbroken tokens (URLs, glued words) never overflow the page
        pdf.multi_cell(0, 5, para.strip() or ' ', wrapmode='CHAR', new_x='LMARGIN', new_y='NEXT')
    pdf.output(str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=50, help='number of resumes (default 50)')
    ap.add_argument('--job-index', type=int, default=None,
                    help='jobs sorted by number of labelled resumes, 0 = most. Default: best label mix')
    ap.add_argument('--split', choices=['test', 'train'], default='test', help='test = jobs the model never saw')
    ap.add_argument('--out', type=Path, default=Path.home() / 'ResumeFit_test_data')
    ap.add_argument('--no-names', action='store_true', help='do not add made-up candidate names')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    df = load_split(a.split)
    df = df.drop_duplicates(['resume_text', 'job_description_text'])
    jobs = df.job_description_text.value_counts()
    if a.job_index is None:   # job with the largest smallest-label count (most balanced), among jobs with >= n/2
        mix = pd.crosstab(df.job_description_text, df.label).reindex(jobs.index)
        ok = mix[mix.sum(axis=1) >= a.n // 2]
        a.job_index = jobs.index.get_loc((ok if len(ok) else mix).min(axis=1).idxmax())
    if not 0 <= a.job_index < len(jobs):
        raise SystemExit(f'--job-index must be 0..{len(jobs) - 1}')
    jd = jobs.index[a.job_index]
    own = df[df.job_description_text == jd].drop_duplicates('resume_text')
    own = own.sample(min(a.n, len(own)), random_state=a.seed)   # random subset keeps the label mix
    rows = [(t, lab) for t, lab in zip(own.resume_text, own.label)]
    if len(rows) < a.n:   # top up with resumes that were labelled for other jobs (most will be poor fits)
        pool = df[~df.resume_text.isin(own.resume_text)].drop_duplicates('resume_text')
        extra = pool.resume_text.sample(min(a.n - len(rows), len(pool)), random_state=a.seed)
        rows += [(t, 'not labelled (resume from another job)') for t in extra]
    random.Random(a.seed).shuffle(rows)   # so the file order gives no hint of the label

    rng = random.Random(a.seed)
    names = [f'{f} {l}' for f in FIRST for l in LAST]
    rng.shuffle(names)

    out = a.out.expanduser() / f'job_{a.job_index}'
    res_dir = out / 'resumes'
    res_dir.mkdir(parents=True, exist_ok=True)
    for old in res_dir.glob('*.pdf'):
        old.unlink()

    (out / 'job_description.txt').write_text(jd)
    write_pdf(jd, out / 'job_description.pdf', title='Job Description')

    with open(out / 'answer_key.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'candidate_name', 'dataset_label'])
        for i, (text, label) in enumerate(rows, start=1):
            name = None if a.no_names else names[i - 1]
            fname = f'{i:02d}_{name.replace(" ", "_")}.pdf' if name else f'resume_{i:02d}.pdf'
            write_pdf(text, res_dir / fname, title=name)
            w.writerow([fname, name or '', label])
            print(f'\r  writing resumes {i}/{len(rows)}', end='', flush=True)
    print()

    counts = pd.Series([lab for _, lab in rows]).value_counts()
    print(f'\nSaved to {out}')
    print(f'  job_description.pdf / .txt   ({len(jd.split())} words)')
    print(f'  resumes/                     {len(rows)} PDFs')
    print('  answer_key.csv               dataset labels:')
    for lab, n in counts.items():
        print(f'      {n:3d}  {lab}')
    print('\nIn the app: upload job_description.pdf, select all PDFs in resumes/, click "Rank candidates".')


if __name__ == '__main__':
    main()
