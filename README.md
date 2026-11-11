# ResumeFit

**Give it one job description and up to 500 resumes. It returns the top-k candidates, a label for each
(Good Fit / Potential Fit / No Fit), and plain-English reasons why each one was or wasn't selected.**

ResumeFit is a hybrid machine-learning resume matcher served as a FastAPI backend. It combines keyword
similarity (TF-IDF), semantic similarity (BERT sentence embeddings), skill matching and
experience parsing, then uses a trained classifier and a learning-to-rank model to sort candidates.

- Dataset: [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
  (6,241 train / 1,759 test resume–job pairs, 3 labels)
- Default model: `artifacts/pipeline_v3.joblib` (included in this repo, no training needed to run)
- Input: PDF / DOCX / TXT resume files, or plain text as JSON
- Output: JSON with ranked candidates, labels, confidence, score breakdown, skill-by-skill evidence and reasons

---

## Contents
1. [Quick start](#quick-start)
2. [How it works](#how-it-works)
3. [Results](#results)
4. [API](#api)
5. [Example response](#example-response)
6. [Training and evaluation](#training-and-evaluation)
7. [Project layout](#project-layout)
8. [Configuration](#configuration)
9. [Limitations](#limitations)

---

## Quick start

Requires **Python 3.12**. Commands assume Linux/macOS (on Windows use `.venv\Scripts\python`).

```bash
git clone https://github.com/JatinJain703/ResumeFit.git
cd ResumeFit

# 1. environment (uv is fastest; plain venv + pip also works)
uv venv --python 3.12 .venv                 # or: python3.12 -m venv .venv
# CPU-only PyTorch keeps the install small
uv pip install --python .venv/bin/python torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt

# 2. run the API
.venv/bin/uvicorn app.main:app --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI, where you can upload files and try every endpoint.

On the first start the BERT model `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) is downloaded once. Later
starts use the local copy.

> `scikit-learn==1.6.1` is pinned on purpose: the model files were pickled with that version.

### Try it with demo data

```bash
# build a demo job folder from the test split: 1 job description + 100 resumes + their dataset labels
.venv/bin/python scripts/make_samples.py --job-index 4
cd samples/test_job_4

curl -X POST localhost:8000/rank/upload \
     -F "job_description=<job_description.txt" -F top_k=10 \
     $(for f in resumes/*.txt; do printf -- '-F files=@%s ' "$f"; done) > result.json
```

`true_labels.csv` holds the dataset labels so you can compare them with `result.json`.

---

## How it works

```
 Job description + N resumes (PDF / DOCX / TXT)
            │
            ▼
 1. Parsing         pypdf / python-docx / text decode; unreadable or scanned files → "skipped"
 2. Cleaning        fixes PDF artefacts: glued words ("Engineeringwith"), glued dates ("01/2018to")
 3. JD analysis     required core technologies, competencies, preferred skills, soft skills,
                    years required (recruiter/company footer ignored)
 4. Pair features   for every (JD, resume) pair
      • TF-IDF (1-2 grams, 50k)        → keyword similarity + Logistic Regression on the pair
      • BERT all-MiniLM-L6-v2           → semantic similarity + Logistic Regression on the pair
      • Skill matching (394-skill catalog with aliases, e.g. k8s → kubernetes, Postgres → postgresql)
      • Experience: stated years ("5+ years") and years summed from work date ranges
 5. Within-job      percentile and z-score of each feature among all resumes for this job
                    (a resume is judged relative to the other applicants)
 6. Stage 2         Gradient Boosting classifier → label + probabilities
                    XGBoost LambdaMART ranker   → order of candidates
 7. Guard           core_guard: no "Good Fit" if the resume has none of the JD's core technologies
 8. Explanations    reasons, per-skill evidence, score breakdown
            │
            ▼
 JSON: top_k[], not_selected[], skipped[], warnings[]
```

### Stage 1 – pairwise models
The resume vector **r** and job vector **j** are combined as `[r ⊙ j, |r − j|]` (element-wise product and
absolute difference). A Logistic Regression learns which overlaps and gaps matter. This is done separately for
TF-IDF and for BERT. Out-of-fold predictions (GroupKFold by job) feed stage 2, so stage 2 never trains on
leaked scores.

### Stage 2 – classifier + ranker
32 features: stage-1 probabilities, cosine similarities, skill coverage, experience (years required, found, gap),
and within-job percentile/z-scores of the key signals.
- **Gradient Boosting** (scikit-learn `HistGradientBoostingClassifier`) gives the label and class probabilities.
- **XGBoost ranker** (`rank:ndcg`) gives the order. It was chosen over sorting by classifier probability
  because it ranked better in cross-validation.

### Explanation layer
Explanations use their own skill rules, separate from the model features:
- **Required vs preferred vs soft**: based on cue phrases ("is a plus", "nice to have") and section
  headers ("Preferred Qualifications:").
- **Technologies vs competencies**: a missing tool (Airflow) counts more than a missing broad area
  (data engineering).
- **Skill hierarchy, evidence flows upward only**: Redshift ⇒ AWS, MySQL ⇒ SQL. Never
  "data engineering" ⇒ Airflow, and never Azure ⇒ AWS. Only SQL dialects, NoSQL stores and BI tools give
  partial credit for each other.
- **No double counting**: if a JD lists "AWS (Redshift, EC2)", AWS isn't counted on top of its services.

---

## Results

All numbers are on the **official test split: 71 jobs that never appear in training**. Every
modelling decision was made with **grouped 5-fold cross-validation on the train split (grouped by job)**.
The test split was only used for reporting.

| Model | Accuracy | Macro-F1 | ROC-AUC | NDCG@10 | Top-3 hit | Shortlist AUROC* |
|---|---|---|---|---|---|---|
| v2 – original Colab model (Random Forest) | 0.585 | 0.550 | 0.757 | 0.688 | 0.607 | 0.809 |
| **v3 – default** (Gradient Boosting + XGBoost ranker) | **0.606** | **0.570** | **0.776** | **0.726** | **0.673** | **0.869** |

\* Shortlist = Good Fit or Potential Fit vs No Fit.

### What helped (ablation, `artifacts/ablation_v4.csv`)

| Step | CV macro-F1 | CV NDCG@10 | Test macro-F1 | Test NDCG@10 |
|---|---|---|---|---|
| v2 features (Colab) | 0.520 | 0.703 | 0.550 | 0.688 |
| + better text cleaning | 0.523 | 0.702 | 0.553 | 0.686 |
| + 394-skill catalog | 0.521 | 0.704 | 0.558 | 0.685 |
| **+ experience from work dates** | **0.537** | **0.727** | 0.601 | 0.724 |
| + weighted required/preferred skills | 0.534 | 0.725 | 0.590 | 0.730 |
| + ranker & model choice (final) | 0.541 | 0.737 | 0.570 | 0.726 |

The biggest gain came from experience parsing. Years of experience are now found for 98.9% of resumes, up from 41.6%.

### Algorithms compared for stage 2 (`artifacts/model_comparison_v3.csv`)
Logistic Regression, Naive Bayes, SVM (RBF), Random Forest, Gradient Boosting, MLP and XGBoost were all
compared. Gradient Boosting had the best **CV** macro-F1 (0.541), so it was selected, even though Random
Forest happens to score higher on the test split. Choosing by test score would leak test information.

### Tried and rejected (measured, not adopted)
| Idea | Result |
|---|---|
| Tuned per-class label thresholds | CV macro-F1 0.473 vs 0.541 for argmax |
| Fixed formula `0.40·skills + 0.25·exp + 0.20·semantic + 0.15·keyword` as the ranking | test NDCG@10 0.625 vs 0.726 (still available as `ranking=formula`) |
| Pretrained ms-marco cross-encoder | NDCG@10 0.573 alone, ~10 pairs/s on CPU |
| Pushing resumes with zero core skills to the bottom | NDCG@10 0.713 vs 0.726 |

### Why not a random train/test split?
The dataset reuses only 280 different JDs and 642 different resumes across 6,241 training pairs.
With a random 80/20 split, 99.7% of test rows share their JD with training and 98.9% share their
resume, so accuracy around 78% just measures memorisation. Splitting by job gives lower but honest numbers.

### A note on label quality
Some dataset labels are noisy. For example, strong senior data engineers are labelled "No Fit" for test
job #4. `scripts/evaluate.py` therefore also supports **human-labelled** job folders. See
[Training and evaluation](#training-and-evaluation).

---

## API

| Method | Endpoint | Input | Purpose |
|---|---|---|---|
| POST | `/rank/upload` | multipart form (below) | Rank uploaded resume files |
| POST | `/rank` | JSON (below) | Rank resumes you already have as text |
| GET | `/model-info` | – | Model version, metrics, algorithm comparison, ablation |
| GET | `/health` | – | Status |

### `POST /rank/upload` (multipart/form-data)

| Field | Type | Default | Notes |
|---|---|---|---|
| `job_description` | text | required | the JD as plain text |
| `files` | file, repeated | required | `.pdf`, `.docx`, `.txt`, `.md`; ≤ 5 MB each; ≤ 500 files |
| `top_k` | int | 10 | size of the shortlist |
| `ranking` | `model` \| `formula` | `model` | trained ranker, or the transparent weighted formula |
| `core_guard` | bool | `true` | cap "Good Fit" → "Potential Fit" when the resume matches 0 of ≥3 core technologies |

### `POST /rank` (JSON)

```json
{
  "job_description": "Senior Data Engineer ... Python, SQL, AWS (Redshift, EC2), Airflow ...",
  "resumes": [ {"id": "alice", "text": "..."}, {"id": "bob", "text": "..."} ],
  "top_k": 10,
  "ranking": "model",
  "core_guard": true
}
```

### From a browser (JavaScript)

```js
const form = new FormData();
form.append("job_description", jdText);
for (const f of files) form.append("files", f);   // same key once per file
form.append("top_k", 10);
const res = await fetch("http://localhost:8000/rank/upload", { method: "POST", body: form });
const data = await res.json();
```
A frontend on another origin (e.g. `localhost:5173`) needs `CORSMiddleware` added in `app/main.py`.

### Status codes
- `200` – ranked. Unreadable files don't fail the request; they are listed in `skipped` with a reason.
- `413` – more than 500 resumes.
- `422` – invalid input, or no resume could be read.

### Tips
- Send **10 or more resumes** per job. Part of the model compares applicants with each other, so a
  warning is returned for smaller batches.
- 100 new resumes take ~20 s on a laptop CPU, mostly BERT. Resumes already seen are cached and a repeat
  takes ~1 s.

---

## Example response

Shortened output for test job #4 (Senior Data Engineer, 100 resumes):

```json
{
  "job": {
    "required_technologies": ["airflow", "aws", "aws emr", "ec2", "nosql", "python", "rds", "redshift", "sql"],
    "required_competencies": ["big data", "cloud computing", "data engineering", "data modeling", "..."],
    "preferred_skills": ["dbt"],
    "years_required": 8.0
  },
  "summary": {
    "total_received": 100, "scored": 100, "skipped": 0, "top_k": 10,
    "label_counts": {"Good Fit": 32, "Potential Fit": 29, "No Fit": 39},
    "ranked_by": "model", "core_guard": true
  },
  "top_k": [
    {
      "rank": 1,
      "resume_id": "resume_057.txt",
      "label": "Good Fit",
      "confidence": 0.6431,
      "probabilities": {"no_fit": 0.0587, "potential_fit": 0.2983, "good_fit": 0.6431},
      "experience": {"required_years": 8.0, "found_years": 15.5, "found_from": "work dates", "score": 1.0},
      "score_breakdown": {"skills": 0.222, "experience": 1.0, "semantic": 0.759, "keyword": 0.324,
                          "formula_score": 0.5393, "model_score": 1.8604},
      "skill_details": [
        {"skill": "sql", "importance": "required", "kind": "technology", "status": "matched", "via": null},
        {"skill": "cloud computing", "importance": "required", "kind": "competency", "status": "matched", "via": "azure"},
        {"skill": "airflow", "importance": "required", "kind": "technology", "status": "missing", "via": null}
      ],
      "reasons": [
        "Core technologies: 1 of 9 (sql) - 26th of 100 applicants on weighted skill coverage",
        "Competencies: 5 of 7 (dimensional modeling, cloud computing (via azure), data engineering, ...)",
        "Resume content is closely related to the job in meaning (19th of 100 on semantic similarity)",
        "Meets the experience requirement (15.5 years from work dates, 8 required)",
        "Gaps to check in interview: missing core technologies: airflow, aws emr, ec2, nosql, python, rds (+2 more)"
      ]
    }
  ],
  "not_selected": [ "... same shape, with reasons such as 'Missing core technologies: ...' and 'Overall, ranked below the top 10 by the model score'" ],
  "skipped": [ {"resume_id": "scan.pdf", "reason": "too little text extracted (scanned/image PDF? OCR is not supported yet)"} ],
  "warnings": []
}
```

Field guide:
- `label`, `confidence`, `probabilities` – stage-2 classifier output.
- `rank_score` – always the exact score the list is sorted by (`model_score` or `formula_score`).
- `label_capped` – `true` if `core_guard` lowered the label.
- `score_breakdown` – skills, experience, semantic and keyword on a 0–1 scale, for display.
- `skill_details[].status` – `matched`, `related` (partial credit) or `missing`. `via` shows the evidence
  (e.g. AWS matched via Redshift).

---

## Training and evaluation

### Retrain (optional – the trained model is already in `artifacts/`)
```bash
.venv/bin/python training/train.py        # ~10 min on CPU
```
This runs the ablation, compares 7 algorithms, picks ranker vs classifier ordering and argmax vs tuned
thresholds, all by grouped CV. It writes `pipeline_v3.joblib`, `ablation_v3.csv` and `model_comparison_v3.csv`.
Training and the API import the same `app/ml/features.py` and `app/ml/pipeline.py`, so features can't drift.

The original Colab notebook is kept as `resume_job_matcher.py`. It produced `pipeline_v2.joblib`.

### Evaluate
```bash
# official test split, both models, all ranking settings → NDCG@10, P@10, R@10, label accuracy / macro-F1
.venv/bin/python scripts/evaluate.py --dataset --pipelines pipeline_v2.joblib pipeline_v3.joblib

# your own human-labelled jobs
.venv/bin/python scripts/evaluate.py --init labelled/job_1        # writes labels.csv to fill in
.venv/bin/python scripts/evaluate.py --folders labelled/job_*      # add --relevant potential to count Potential Fit as relevant
```
Human-labelled folder format: `job_description.txt`, `resumes/` (PDF/DOCX/TXT), and `labels.csv`
(`file,label` with `No Fit` / `Potential Fit` / `Good Fit`).

### Tests
```bash
.venv/bin/python -m pytest -m "not slow"   # reproduces v2 and v3 test metrics, feature unit tests, API tests
.venv/bin/python -m pytest -m slow         # checks local BERT embeddings match the Colab ones (downloads BERT)
```

---

## Project layout

```
app/
  main.py              FastAPI app and endpoints
  ranker.py            runs the pipeline for one JD + many resumes, builds the response
  explain.py           reasons for selected / not-selected resumes
  parsing.py           PDF / DOCX / TXT text extraction
  schemas.py           request / response models (Pydantic)
  config.py            limits and paths (RESUMEFIT_* env vars)
  ml/
    pipeline.py        model loading, feature building, scoring, ranking (shared with training)
    features.py        cleaning, skill matching, experience parsing, JD skill importance
    skills_catalog.py  394 skills, aliases, context rules, hierarchy, preferred-cue phrases
training/
  train.py             retraining, ablation, model comparison
  train_log*.txt       logs of the training runs behind the numbers above
scripts/
  make_samples.py      build a demo job folder from the test split
  evaluate.py          NDCG@10 / P@10 / R@10 on the dataset or human-labelled folders
tests/                 regression, feature and API tests
artifacts/
  pipeline_v3.joblib   default model (Gradient Boosting + XGBoost ranker)
  pipeline_v2.joblib   original Colab model (Random Forest), used by regression tests
  embeddings*.joblib   cached BERT vectors of all dataset texts (fast tests / retraining)
  train_features.parquet, test_features.parquet   dataset texts and labels
  ablation_*.csv, model_comparison*.csv           result tables
resume_job_matcher.py  original Google Colab notebook (export)
```

---

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `RESUMEFIT_PIPELINE` | `pipeline_v3.joblib` | model file in `artifacts/` (`pipeline_v2.joblib` for the Colab model) |
| `RESUMEFIT_ARTIFACTS` | `./artifacts` | artifacts directory |
| `RESUMEFIT_MAX_RESUMES` | 500 | max resumes per request |
| `RESUMEFIT_MAX_FILE_MB` | 5 | max size per file |
| `RESUMEFIT_EMBEDDING_CACHE` | 20000 | number of BERT vectors kept in memory |
| `RESUMEFIT_LAZY_BERT` | 0 | `1` = load BERT on first request instead of at startup |

---

## Limitations

- **No OCR**: scanned or image-only PDFs are skipped with a reason.
- **English, mostly tech/office jobs**: the skill catalog and training data are English, and best covered
  for IT, data, healthcare and business roles.
- **Noisy labels**: dataset labels are imperfect. Treat the label as a guide and the ranking plus reasons as
  the main output. Use human-labelled jobs to judge quality.
- **CPU latency**: BERT dominates (~0.2 s per new resume). Use a GPU or background jobs for very large batches.
- **No CORS / auth yet**: add `CORSMiddleware` for a browser frontend and authentication before exposing it publicly.
- Resumes are processed in memory and not stored. Only BERT vectors are cached while the server runs.
