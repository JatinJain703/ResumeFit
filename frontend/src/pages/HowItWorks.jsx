import { useEffect, useState } from 'react'
import { modelInfo } from '../api.js'
import Link from '../components/Link.jsx'

const FLOW = [
  { title: 'Input', items: ['Job description (text / PDF)', 'Resumes (PDF / DOCX / TXT)'] },
  { title: 'Signals', items: ['TF-IDF keyword similarity', 'BERT meaning (all-MiniLM-L6-v2)', '394-skill catalog + hierarchy', 'Years from dates / text'] },
  { title: 'Stage 1', items: ['Logistic Regression on TF-IDF pairs', 'Logistic Regression on BERT pairs'] },
  { title: 'Stage 2', items: ['32 features incl. comparison with the other applicants', 'Gradient Boosting → label', 'XGBoost LambdaMART → order'] },
  { title: 'Output', items: ['Top-k shortlist', 'Good / Potential / No Fit', 'Reasons + skill evidence'] },
]

const COLS = ['Accuracy', 'F1 (macro)', 'ROC-AUC', 'NDCG@10', 'Top-3 hit']

function MetricTable({ rows, nameKey, highlight }) {
  const best = Object.fromEntries(COLS.map((c) => [c, Math.max(...rows.map((r) => r[c] ?? 0))]))
  return (
    <div className="table-wrap">
      <table className="metrics">
        <thead><tr><th>{nameKey === 'model' ? 'Algorithm' : 'Step'}</th>{COLS.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[nameKey]} className={highlight?.(r) ? 'hl' : ''}>
              <td>{r[nameKey]}</td>
              {COLS.map((c) => (
                <td key={c} className={r[c] === best[c] ? 'best' : ''}>{r[c] == null ? '–' : r[c].toFixed(3)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function HowItWorks() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => { modelInfo().then(setInfo).catch((e) => setError(e.message)) }, [])

  return (
    <div className="page">
      <div className="page-head">
        <span className="mode-tag">Under the hood</span>
        <h1>How ResumeFit works</h1>
        <p className="muted">A hybrid pipeline: classic text similarity, a sentence-embedding model, a skill catalog
          and experience parsing, combined by two trained models.</p>
      </div>

      <section className="flow" aria-label="Pipeline">
        {FLOW.map((f, i) => (
          <div key={f.title} className="flow-step">
            <div className="flow-box">
              <h3>{f.title}</h3>
              <ul>{f.items.map((x) => <li key={x}>{x}</li>)}</ul>
            </div>
            {i < FLOW.length - 1 && <span className="flow-arrow" aria-hidden>→</span>}
          </div>
        ))}
      </section>

      <section className="explain-grid">
        <div className="panel">
          <h3 className="panel-title">The four signals</h3>
          <dl className="defs">
            <dt>Skills</dt>
            <dd>Weighted share of the job’s skills found in the resume. Required skills count 1, preferred 0.5,
              soft 0.25. A specific skill proves its parent (Redshift → AWS); a close substitute earns half
              (MySQL for PostgreSQL).</dd>
            <dt>Experience</dt>
            <dd>Years found (from work dates or “N years” phrases) divided by years required, capped at 1.</dd>
            <dt>Meaning</dt>
            <dd>Cosine similarity of BERT embeddings of the resume and the job: catches related wording.</dd>
            <dt>Keywords</dt>
            <dd>TF-IDF cosine similarity: how much of the job’s specific vocabulary the resume uses.</dd>
          </dl>
        </div>
        <div className="panel">
          <h3 className="panel-title">Two ways to order</h3>
          <p><strong>AI model</strong> (default): an XGBoost learning-to-rank model trained to put Good Fits first.
            It uses all 32 features, including how each resume compares with the other applicants.</p>
          <p><strong>Your weights</strong>: a transparent formula you control.</p>
          <pre className="formula">score = 0.40·skills + 0.25·experience{'\n'}      + 0.20·meaning + 0.15·keywords</pre>
          <p className="muted small">Must-have filters (skills, minimum years) never change scores; they only decide who
            can be shortlisted.</p>
          <h3 className="panel-title spaced">Checking one resume</h3>
          <p className="small">The model judges a resume relative to other applicants, so in <Link to="/candidate" className="link">Check my resume</Link> your
            resume is scored together with the 24 most similar resumes from the training data. This raised label accuracy
            for a single resume from 0.60 to 0.66.</p>
        </div>
      </section>

      <section className="section">
        <h2 className="section-heading left">Results on unseen jobs</h2>
        <p className="muted">Dataset: <a className="link" href="https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit"
          target="_blank" rel="noreferrer">Resume–Job Description Fit</a> (Hugging Face). Trained on 6,241 resume–job
          pairs; tested on 71 job descriptions the model never saw. NDCG@10 measures ranking quality (1 = perfect order);
          Top-3 hit is the share of the top 3 that are Good or Potential Fit.</p>
        {error && <div className="note warn">Model results need the API: {error}</div>}
        {info && (
          <>
            <div className="metrics-band compact">
              <div><strong>{info.final_test_metrics['NDCG@10'].toFixed(3)}</strong><span>NDCG@10</span></div>
              <div><strong>{info.final_test_metrics['Accuracy'].toFixed(3)}</strong><span>accuracy (3 labels)</span></div>
              <div><strong>{info.final_test_metrics['ROC-AUC'].toFixed(3)}</strong><span>ROC-AUC</span></div>
              <div><strong>{Math.round(info.final_test_metrics['Top-3 hit'] * 100)}%</strong><span>top-3 hit</span></div>
            </div>
            <h3 className="panel-title spaced">Algorithms compared for stage 2</h3>
            <MetricTable rows={info.test_set_results} nameKey="model" highlight={(r) => r.model === info.final_model} />
            <p className="muted small">Highlighted: the algorithm in use ({info.final_model}, chosen by cross-validation on
              the training jobs, not by test results). Bold: best in each column. The final system adds the {info.ranking_model} ranker.</p>
            {info.ablation?.length > 0 && (
              <>
                <h3 className="panel-title spaced">What each improvement added</h3>
                <MetricTable rows={info.ablation} nameKey="index" />
                <p className="muted small">Feature steps are cumulative, starting from the original Colab model, each measured with a Random Forest. The last row is the final system (best feature set + chosen classifier + ranker + label rule).</p>
              </>
            )}
          </>
        )}
      </section>

      <section className="panel limits">
        <h3 className="panel-title">Limitations</h3>
        <ul>
          <li>Dataset labels are noisy; treat labels as a guide and read the reasons.</li>
          <li>English only; skills are best covered for IT, data, healthcare and business roles.</li>
          <li>Scanned (image-only) PDFs can’t be read; there is no OCR.</li>
          <li>Within-batch comparison means rankings are most reliable with 10 or more resumes.</li>
        </ul>
      </section>
    </div>
  )
}
