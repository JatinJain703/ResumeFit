import { useEffect, useState } from 'react'
import { modelInfo } from '../api.js'
import Link from '../components/Link.jsx'

const PREVIEW = [
  { rank: 1, name: 'Priya Sharma', label: 'Good Fit', cls: 'good', score: 92, skills: ['python', 'aws', 'airflow'] },
  { rank: 2, name: 'David Chen', label: 'Good Fit', cls: 'good', score: 87, skills: ['sql', 'spark', 'aws'] },
  { rank: 3, name: 'Amara Okafor', label: 'Potential Fit', cls: 'potential', score: 71, skills: ['python', 'etl'] },
]

const STEPS = [
  { n: 1, title: 'Read', text: 'PDF, DOCX or text resumes and job descriptions are parsed and cleaned. Names, skills and years of experience are extracted.' },
  { n: 2, title: 'Compare', text: 'Each resume is compared with the job in three ways: shared keywords (TF-IDF), meaning (BERT embeddings) and skills from a 394-skill catalog.' },
  { n: 3, title: 'Score', text: 'A two-stage model (logistic regressions → gradient boosting) labels each resume, and an XGBoost learning-to-rank model orders them.' },
  { n: 4, title: 'Explain', text: 'Every result comes with plain-English reasons: which skills matched, what is missing, and how experience compares.' },
]

const FEATURES = [
  { icon: '⚖️', title: 'Adjustable weights', text: 'Decide how much skills, experience, meaning and keywords count. The list re-orders instantly.' },
  { icon: '✅', title: 'Must-have filters', text: 'Require skills like "Python" or "AWS" and a minimum number of years. Synonyms and sub-skills are understood.' },
  { icon: '💬', title: 'Reasons, not just scores', text: 'See why each candidate was shortlisted or not, skill by skill.' },
  { icon: '🧭', title: 'Resume coaching', text: 'Candidates get a fit label, their skill gaps and prioritised suggestions for a specific job.' },
  { icon: '📄', title: 'Works with real files', text: 'Upload PDFs directly, with the job description as text or a PDF. View each resume from the results.' },
  { icon: '🧪', title: 'Measured, not guessed', text: 'Evaluated on 71 job descriptions the model never saw, against 7 algorithms.' },
]

const pct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`)

export default function Home() {
  const [info, setInfo] = useState(null)
  useEffect(() => { modelInfo().then(setInfo).catch(() => {}) }, [])
  const m = info?.final_test_metrics

  return (
    <div className="home">
      <section className="hero">
        <div className="hero-text">
          <span className="eyebrow">AI resume screening · explainable</span>
          <h1>Find the right candidates.<br /><span className="accent-text">Fix the right gaps.</span></h1>
          <p className="lead">
            ResumeFit compares resumes with a job description using keywords, meaning and skills. Recruiters get a
            ranked shortlist with reasons. Candidates find out how well their resume fits and what to improve.
          </p>
          <div className="cta-row">
            <Link to="/recruiter" className="btn primary">Rank candidates</Link>
            <Link to="/candidate" className="btn big">Check my resume</Link>
          </div>
        </div>

        <div className="hero-preview" aria-hidden>
          <div className="preview-head">
            <span className="muted small">Senior Data Engineer · 48 resumes</span>
            <span className="pill">Top 3</span>
          </div>
          {PREVIEW.map((p) => (
            <div key={p.rank} className="preview-row">
              <span className="rank sm">#{p.rank}</span>
              <div className="preview-who">
                <strong>{p.name}</strong>
                <div className="chips tight">{p.skills.map((s) => <span key={s} className="chip ok">{s}</span>)}</div>
              </div>
              <div className="preview-score">
                <span className={`badge ${p.cls}`}>{p.label}</span>
                <span className="bar-track"><span className="bar-fill" style={{ width: `${p.score}%` }} /></span>
              </div>
            </div>
          ))}
          <p className="preview-reason">“Has 6 of 7 core technologies; meets the experience requirement (9 years from work dates, 5 required)”</p>
        </div>
      </section>

      <section className="modes">
        <Link to="/recruiter" className="mode-card">
          <span className="mode-tag">For recruiters</span>
          <h2>Rank candidates</h2>
          <p>Upload one job description and up to 500 resumes. Get the top candidates with labels, scores and
            reasons, then tune weights and must-have filters live.</p>
          <ul className="ticks">
            <li>Top-k shortlist with Good / Potential / No Fit labels</li>
            <li>Skill-by-skill evidence and gaps</li>
            <li>Sliders and filters that re-rank instantly</li>
          </ul>
          <span className="mode-go">Start ranking →</span>
        </Link>
        <Link to="/candidate" className="mode-card alt">
          <span className="mode-tag">For job seekers</span>
          <h2>Check my resume</h2>
          <p>Upload your resume and the job you want. See your fit, how you compare with similar resumes, and
            exactly what to add or change.</p>
          <ul className="ticks">
            <li>Fit label and 0–100 match score</li>
            <li>Matched, related and missing skills</li>
            <li>Prioritised, concrete suggestions</li>
          </ul>
          <span className="mode-go">Check my fit →</span>
        </Link>
      </section>

      <section className="section">
        <h2 className="section-heading">How it works</h2>
        <div className="steps">
          {STEPS.map((s) => (
            <div key={s.n} className="step-card">
              <span className="step">{s.n}</span>
              <h3>{s.title}</h3>
              <p>{s.text}</p>
            </div>
          ))}
        </div>
        <p className="center"><Link to="/how-it-works" className="link">Model details and results →</Link></p>
      </section>

      <section className="section">
        <h2 className="section-heading">Features</h2>
        <div className="features">
          {FEATURES.map((f) => (
            <div key={f.title} className="feature">
              <span className="feature-icon" aria-hidden>{f.icon}</span>
              <h3>{f.title}</h3>
              <p>{f.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section metrics-band">
        <div>
          <strong>{m ? m['NDCG@10'].toFixed(3) : '0.726'}</strong>
          <span>NDCG@10 ranking quality</span>
        </div>
        <div>
          <strong>{m ? pct(m['Top-3 hit']) : '67%'}</strong>
          <span>of top-3 picks are Good or Potential Fit</span>
        </div>
        <div>
          <strong>{m ? pct(m['Shortlist acc']) : '76%'}</strong>
          <span>accuracy on fit vs no fit</span>
        </div>
        <div>
          <strong>{info?.n_skills_in_catalog ?? 394}</strong>
          <span>skills recognised</span>
        </div>
      </section>
      <p className="muted small center">Measured on the test split of the Resume–Job Description Fit dataset: 71 unseen job descriptions.</p>
    </div>
  )
}
