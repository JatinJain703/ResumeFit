# ResumeFit

**Give it one job description and up to 500 resumes. It returns the top-k candidates, a label for each
(Good Fit / Potential Fit / No Fit), and plain-English reasons why each one was or wasn't selected.**
**Candidates can also check one resume against one job and get their skill gaps and concrete suggestions.**

ResumeFit is a hybrid machine-learning resume matcher: a FastAPI backend plus a React web app. It combines
keyword similarity (TF-IDF), semantic similarity (BERT sentence embeddings), skill matching and
experience parsing, then uses a trained classifier and a learning-to-rank model to sort candidates.

- Dataset: [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
  (6,241 train / 1,759 test resume–job pairs, 3 labels)
- Default model: `artifacts/pipeline_v3.joblib` (included in this repo, no training needed to run)
- Input: PDF / DOCX / TXT resume files, or plain text as JSON
- Output: JSON with ranked candidates, labels, confidence, score breakdown, skill-by-skill evidence and reasons
- Two modes: **recruiter** (`/rank`: many resumes → shortlist) and **candidate** (`/fit-check`: one resume →
  fit label, skill gaps, missing keywords, prioritised suggestions)
- Web app: home page, *Rank candidates*, *Check my resume* and *How it works* pages ([Web frontend](#web-frontend))

### Features

| Feature | For | What it does | API |
|---|---|---|---|
| Ranked shortlist with reasons | recruiters | top-k candidates, Good / Potential / No Fit labels, skill-by-skill evidence, plain-English reasons | `/rank`, `/rank/upload` |
| Adjustable weights | recruiters | your own weights for skills, experience, meaning and keywords (sliders), or the trained AI ranking | `/rerank` ([details](#recruiter-controls-custom-weights-must-have-filters-post-rerank)) |
| Must-have filters | recruiters | required skills (synonyms and sub-skills understood) and minimum years; failures listed with the reason | `/rerank` |
| Instant re-ranking | recruiters | resumes are scored once; changing weights or filters re-orders in ~0.3 s without re-uploading | `/rerank` |
| Resume fit check | job seekers | one resume vs one job: fit label, 0–100 match score, position among similar resumes, skill gap, missing keywords, prioritised suggestions | `/fit-check` ([details](#candidate-mode-post-fit-checkupload-and-post-fit-check)) |
| Real files | both | PDF / DOCX / TXT resumes; job description as text or a file; candidate names read from the resume | all |
| Model transparency | both | test metrics, algorithm comparison and ablation served by the API and shown in the app | `/model-info` |

---

## Contents
1. [Quick start](#quick-start)
   - [Web frontend](#web-frontend)
2. [How it works](#how-it-works)
3. [Results](#results)
4. [API](#api)
   - [Candidate mode (`/fit-check`)](#candidate-mode-post-fit-checkupload-and-post-fit-check)
   - [Recruiter controls (weights, filters, `/rerank`)](#recruiter-controls-custom-weights-must-have-filters-post-rerank)
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

> **macOS:** PyTorch and XGBoost each load their own OpenMP runtime, and on a Mac the two deadlock on the
> first request that needs BERT (the server stays up but never answers). `app/__init__.py` sets
> `OMP_NUM_THREADS=1` automatically on macOS. If you import the code some other way, run
> `OMP_NUM_THREADS=1 .venv/bin/uvicorn app.main:app --port 8000`.

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

### Test PDFs for the web frontend

```bash
# downloads the dataset from Hugging Face and writes 1 job description + 50 resumes as PDFs
.venv/bin/python scripts/download_test_data.py            # -> ~/ResumeFit_test_data/job_<n>/
.venv/bin/python scripts/download_test_data.py --n 30 --job-index 3 --out ~/Desktop/rf_test
```

By default it picks the test job (never seen in training) with the most even mix of Good / Potential / No Fit
resumes. The dataset resumes are anonymised, so a made-up name is put on the first line of each PDF so the app
can show names (`--no-names` to skip). `answer_key.csv` has the dataset label of every resume.

---

### Web frontend

A React (Vite) app in `frontend/` with four pages (hash routes, so it works on any static host):

| Page | Route | What it does |
|---|---|---|
| Home | `#/` | What ResumeFit does, the two modes, how it works, features, live test metrics |
| Rank candidates | `#/recruiter` | JD (text or PDF) + resume PDFs → top-k cards with label, reasons, skill chips, score bars, original PDF viewer. A **Tune the ranking** side panel (AI model vs your weights, 4 weight sliders, must-have skills with suggestions from the job, minimum years, shortlist size) re-ranks instantly via `/rerank`; failed candidates appear under "filtered out" with the reason |
| Check my resume | `#/candidate` | One resume + a JD → match score gauge, label, position among similar resumes, prioritised suggestions, strengths, score breakdown, skill gap (have / related / missing) and missing job keywords (`/fit-check/upload`) |
| How it works | `#/how-it-works` | Pipeline diagram, the four signals and the formula, algorithm comparison and ablation tables (from `/model-info`), limitations |

Results stay when you switch pages. Works on phones (collapsing menu, single-column layout). Needs Node 18+.

```bash
# with the API running on port 8000 (see above), in a second terminal:
cd frontend
npm install
npm run dev            # open http://localhost:5173
```

The API address defaults to `http://localhost:8000`; to change it, copy `frontend/.env.example` to
`frontend/.env` and edit `VITE_API_URL`. `npm run build` writes a static site to `frontend/dist/`.

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
 JSON: top_k[], not_selected[], filtered_out[], skipped[], warnings[], session_id
            │
            ▼
 9. Re-rank (/rerank)   new weights / must-have filters / top_k applied to the stored scores: no BERT, ~0.3 s
```

**Candidate mode** (`/fit-check`) runs the same pipeline for one resume, scored together with the 24 most
similar training resumes (so the within-job features stay meaningful), then turns the skill gap, experience
and missing job keywords into prioritised suggestions.

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
- **Capitalised names survive cleaning**: the cleaner splits "PowerShell" into "Power Shell" (that is what
  the model was trained on), so the explanation matcher also accepts the split spelling of names like
  PowerShell, TensorFlow, MySQL, GitHub, DevOps, iOS and B2B (`CASED_NAMES` in `app/ml/skills_catalog.py`).

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
| POST | `/rerank` | JSON | Re-rank the same resumes instantly with new weights / filters / top_k |
| POST | `/fit-check/upload` | multipart form | Candidate mode: one resume file vs one job |
| POST | `/fit-check` | JSON | Candidate mode with plain text |
| GET | `/model-info` | – | Model version, metrics, algorithm comparison, ablation |
| GET | `/health` | – | Status |

### `POST /rank/upload` (multipart/form-data)

| Field | Type | Default | Notes |
|---|---|---|---|
| `job_description` | text | – | the JD as plain text (send this, `jd_file`, or both) |
| `jd_file` | file | – | the JD as `.pdf`, `.docx` or `.txt` |
| `files` | file, repeated | required | `.pdf`, `.docx`, `.txt`, `.md`; ≤ 5 MB each; ≤ 500 files |
| `top_k` | int | 10 | size of the shortlist |
| `ranking` | `model` \| `formula` | `model` | trained ranker, or the transparent weighted formula |
| `core_guard` | bool | `true` | cap "Good Fit" → "Potential Fit" when the resume matches 0 of ≥3 core technologies |
| `weight_skills`, `weight_experience`, `weight_semantic`, `weight_keyword` | number ≥ 0 | 0.40 / 0.25 / 0.20 / 0.15 | custom formula weights (any scale, e.g. slider 0–100; normalised to sum to 1) |
| `min_years` | number | – | filter: minimum years of experience |
| `unknown_years` | `keep` \| `exclude` | `keep` | with `min_years`: what to do with resumes where no years were found |
| `must_have` | text | – | filter: comma-separated skills, e.g. `python, aws, k8s` |

`ranking` defaults to `formula` when any weight is sent, otherwise `model`.

### `POST /rank` (JSON)

```json
{
  "job_description": "Senior Data Engineer ... Python, SQL, AWS (Redshift, EC2), Airflow ...",
  "resumes": [ {"id": "alice", "text": "..."}, {"id": "bob", "text": "..."} ],
  "top_k": 10,
  "ranking": "model",
  "core_guard": true,
  "weights": {"skills": 50, "experience": 30, "semantic": 20, "keyword": 0},
  "filters": {"min_years": 3, "unknown_years": "keep", "must_have_skills": ["python", "aws"]}
}
```
`weights` and `filters` are optional. Every rank response includes a `session_id`.

### Recruiter controls: custom weights, must-have filters, `POST /rerank`

**Weights.** The transparent formula `skills·w1 + experience·w2 + semantic·w3 + keyword·w4` can use the
recruiter's own weights (any non-negative numbers, normalised to sum to 1). With custom weights the list is
ordered by this formula unless you ask for `"ranking": "model"`; then the weights only change `formula_score`
and a warning says so. The trained model stays the default when no weights are sent.

**Filters** decide who may be shortlisted; they never change a score.
- `min_years`: years found in the resume (stated or from work dates) must be at least this. Resumes with no years
  found are kept unless `unknown_years: "exclude"`.
- `must_have_skills`: each must be in the resume. Names go through the skill catalog, so `k8s` = Kubernetes,
  `PowerShell` = powershell, and a more specific skill counts (`aws` is satisfied by Redshift or EC2). A
  broader or sibling skill does not (Azure doesn't satisfy AWS). Names not in the catalog (`Go`, `R`) are matched
  as plain text, with a warning; 1–2-letter names are case-sensitive.

Resumes that fail go to `filtered_out` with the reason first ("Filtered out: missing must-have skills: java").
Ranks: the resumes that pass are 1…m; filtered-out resumes continue from m+1. Every candidate has
`passed_filters` and, with skill filters, `must_have: [{skill, found, via, matched_as}]`.

**`POST /rerank`.** Scoring (BERT + model) is the slow part; ordering and filtering are not. Each rank request
keeps its scored batch in server memory and returns a `session_id`. Send it back with new options and get a
new ranking with no upload and no re-scoring, so it's fast enough for sliders:

```json
POST /rerank
{"session_id": "Xq3...", "top_k": 5, "ranking": "formula",
 "weights": {"skills": 60, "experience": 20, "semantic": 20, "keyword": 0},
 "filters": {"min_years": 5, "must_have_skills": ["Java", "AWS"]}}
```

The response is the same as `/rank`. Measured on 50 PDF resumes: `/rank/upload` 18 s (new resumes, CPU),
`/rerank` 0.3 s. Sessions are kept for 1 hour after last use, at most 20 at a time
(`RESUMEFIT_SESSION_TTL`, `RESUMEFIT_SESSIONS`). `404` means the session expired, so rank again (the BERT
cache makes a repeat quick).

### From a browser (JavaScript)

```js
const form = new FormData();
form.append("job_description", jdText);
for (const f of files) form.append("files", f);   // same key once per file
form.append("top_k", 10);
const res = await fetch("http://localhost:8000/rank/upload", { method: "POST", body: form });
const data = await res.json();
```
CORS is enabled for `localhost:5173` (Vite) and `localhost:3000` by default. For another frontend URL, set
`RESUMEFIT_CORS_ORIGINS` (comma-separated), e.g. `RESUMEFIT_CORS_ORIGINS=https://myapp.example.com`.

### Candidate mode: `POST /fit-check/upload` and `POST /fit-check`

"How well does my resume fit this job, and what should I change?" One resume and one job description in;
a fit label, a 0–100 match score, the resume's position among similar resumes, the skill gap, missing
keywords, strengths and prioritised suggestions out.

| Field (multipart) | Type | Notes |
|---|---|---|
| `resume` | file | `.pdf`, `.docx`, `.txt`; at least 30 words |
| `job_description` / `jd_file` | text / file | as for `/rank/upload` (one or both) |
| `core_guard` | bool | as for `/rank/upload` |

JSON version: `{"job_description": "...", "resume_text": "...", "core_guard": true}`.

```bash
curl -X POST localhost:8000/fit-check/upload \
  -F "resume=@my_resume.pdf" -F "jd_file=@job_description.pdf"
```

Response (shortened):
```json
{
  "candidate_name": "Michael Wilson",
  "label": "No Fit", "confidence": 0.69, "match_score": 33,
  "headline": "Weak match for this job as the resume stands. Your resume would rank 10th of 25 against the 24 most similar resumes in our database (better than 62% of them).",
  "comparison": {"pool_size": 24, "rank_in_pool": 10, "percentile": 62.5, "description": "..."},
  "skills": {"matched": [{"skill": "sql server", "importance": "required", ...}],
             "related": [{"skill": "cassandra", "status": "related", "via": "mongodb", ...}],
             "missing": [{"skill": "aws", "importance": "required", "kind": "technology", ...}, ...]},
  "missing_keywords": ["large scale", "data processing", "cloud", ...],
  "strengths": ["Has 1 of 9 core technologies: sql server", "Uses much of the job's vocabulary (6th of 25 ...)"],
  "suggestions": [
    {"priority": "high", "area": "skills", "text": "Add aws if you have used it: it is a core requirement ..."},
    {"priority": "medium", "area": "skills", "text": "The job asks for cassandra; your resume shows the related mongodb. If you have used cassandra itself, name it explicitly."},
    {"priority": "medium", "area": "experience", "text": "The job asks for 8+ years; your resume shows 4. ..."}
  ],
  "score_breakdown": {...}, "scores": {...}, "experience": {...}, "job": {...}, "warnings": [], "model": "..."
}
```

**How one resume is judged.** The model's stage-2 features compare a resume with the other applicants for the
same job, so a resume can't be scored meaningfully on its own. `/fit-check` scores it together with a
**reference pool**: the 24 training-set resumes most similar to the job description (BERT cosine; 24 ≈ the
average applicant pool in training; `RESUMEFIT_FIT_POOL`). The label, probabilities and position come from that
comparison. Skill gaps, experience and keywords don't depend on the pool. On 131 unseen test pairs (2 per test
job) this gives the right label more often than scoring the resume alone:

| One resume scored… | Accuracy | macro-F1 |
|---|---|---|
| alone | 0.603 | 0.397 (almost everything → No Fit) |
| with the reference pool | **0.664** | **0.545** |

Suggestions are rules over the explanation layer, most important first:
- **high** – missing core technologies; experience required but no years or dates found; label capped by `core_guard`
- **medium** – related-but-not-exact skills ("name cassandra if you used it"); competencies without evidence;
  fewer years than required; content less related to the job than most comparable resumes; job wording missing
  from the resume; very short resume
- **low** – skills only implied by a more specific one (aws via redshift); missing nice-to-have and soft skills

`missing_keywords` are the job description's highest-weighted TF-IDF terms (words and 2-word phrases) that appear at
least twice in the job and nowhere in the resume, excluding skills already in the gap list and salary/benefits/
legal wording.

### Status codes
- `200` – ranked. Unreadable files don't fail the request; they are listed in `skipped` with a reason.
- `413` – more than 500 resumes.
- `422` – invalid input, or no resume could be read.
- `404` – `/rerank` session not found or expired.

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
    "ranked_by": "model", "core_guard": true,
    "weights": {"skills": 0.4, "experience": 0.25, "semantic": 0.2, "keyword": 0.15},
    "filters": null, "filtered_out": 0
  },
  "top_k": [
    {
      "rank": 1,
      "resume_id": "resume_057.txt",
      "candidate_name": "resume_057",
      "name_source": "filename",
      "upload_index": 56,
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
  "filtered_out": [],
  "session_id": "Xq3vB0...",
  "warnings": []
}
```

Field guide:
- `candidate_name` – taken from the first lines of the resume (a short line of 2–4 capitalised words that
  is not a heading or job title), else from the file name; `name_source` says which. The dataset resumes are
  anonymised, so they fall back to the file name.
- `upload_index` – 0-based position of the resume in the request, so a client can match it to its file.
- `label`, `confidence`, `probabilities` – stage-2 classifier output.
- `rank_score` – always the exact score the list is sorted by (`model_score` or `formula_score`).
- `label_capped` – `true` if `core_guard` lowered the label.
- `score_breakdown` – skills, experience, semantic and keyword on a 0–1 scale, for display.
- `skill_details[].status` – `matched`, `related` (partial credit) or `missing`. `via` shows the evidence
  (e.g. AWS matched via Redshift).
- `summary.weights` – the formula weights used (sum to 1); `summary.filters` / `summary.filtered_out` – active
  filters and how many resumes failed them.
- `filtered_out` – candidates that failed a filter, reason first; `passed_filters` and `must_have`
  (`[{skill, found, via, matched_as}]`) on every candidate when filters are used.
- `session_id` – send to `POST /rerank` to re-rank the same resumes with new options.

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
  fit_check.py         candidate mode: one resume vs one job, reference pool, gaps, keywords, suggestions
  explain.py           reasons for selected / not-selected resumes
  filters.py           recruiter must-have filters (min years, must-have skills)
  sessions.py          in-memory store of scored batches for /rerank
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
  make_samples.py      build a demo job folder (text files) from the test split
  download_test_data.py  download 1 JD + 50 resumes from Hugging Face as PDFs, for the frontend
  evaluate.py          NDCG@10 / P@10 / R@10 on the dataset or human-labelled folders
tests/                 regression, feature and API tests
artifacts/
  pipeline_v3.joblib   default model (Gradient Boosting + XGBoost ranker)
  pipeline_v2.joblib   original Colab model (Random Forest), used by regression tests
  embeddings*.joblib   cached BERT vectors of all dataset texts (fast tests / retraining)
  train_features.parquet, test_features.parquet   dataset texts and labels
  ablation_*.csv, model_comparison*.csv           result tables
frontend/              React + Vite web app: src/pages/ (Home, Recruiter, Candidate, HowItWorks),
                       src/components/, src/api.js (all API calls), src/router.js (hash routes)
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
| `RESUMEFIT_CORS_ORIGINS` | `http://localhost:5173,…:3000` | browser origins allowed to call the API (comma-separated) |
| `RESUMEFIT_SESSIONS` | 20 | scored batches kept in memory for `/rerank` |
| `RESUMEFIT_SESSION_TTL` | 3600 | seconds a `/rerank` session lives after its last use |
| `RESUMEFIT_FIT_POOL` | 24 | reference resumes a `/fit-check` resume is compared with |

---

## Limitations

- **No OCR**: scanned or image-only PDFs are skipped with a reason.
- **English, mostly tech/office jobs**: the skill catalog and training data are English, and best covered
  for IT, data, healthcare and business roles.
- **Noisy labels**: dataset labels are imperfect. Treat the label as a guide and the ranking plus reasons as
  the main output. Use human-labelled jobs to judge quality.
- **CPU latency**: BERT dominates (~0.2 s per new resume). Use a GPU or background jobs for very large batches.
- **No authentication or streaming size limits**: upload size is checked after a file is read into memory, and
  the JSON endpoint has no text-length limit. Fine for a demo; add auth and limits before exposing it publicly.
- **Small batches in `/rank`**: the relative (within-job) features compare a resume with the rest of its batch,
  so very small batches are unreliable (one resume alone is labelled No Fit 91% of the time). The API warns
  below 10 resumes. To judge a single resume, use `/fit-check`, which adds a reference pool.
- **Candidate-mode pool**: the reference resumes come from the training set (642 resumes, mostly IT, data,
  healthcare and business). For jobs far from those fields the closest resumes are still loosely related,
  so the position and label are less meaningful there; the skill gap and suggestions are unaffected.
- Resumes are processed in memory and not written to disk. BERT vectors are cached, and for `/rerank` the last 20
  scored batches (including resume text) are kept in memory for up to an hour. Restarting the server clears both.
- `/rerank` sessions live in one server process; with several workers, use sticky sessions or one worker.
