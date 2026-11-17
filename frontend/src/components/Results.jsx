import { useState } from 'react'
import CandidateCard from './CandidateCard.jsx'
import { LABEL_CLASS } from '../constants.js'

function Chips({ title, items, kind }) {
  if (!items?.length) return null
  return (
    <div className="chips">
      <span className="chips-label">{title}</span>
      {items.map((s) => <span key={s} className={`chip ${kind}`}>{s}</span>)}
    </div>
  )
}

const WEIGHT_NAMES = { skills: 'skills', experience: 'experience', semantic: 'meaning', keyword: 'keywords' }

function orderLine(summary) {
  if (summary.ranked_by === 'model') return 'Ordered by the AI ranking model'
  const w = Object.entries(summary.weights).filter(([, v]) => v > 0)
    .map(([k, v]) => `${WEIGHT_NAMES[k]} ${Math.round(v * 100)}%`)
  return `Ordered by your weights: ${w.join(' · ')}`
}

function CandidateTable({ rows, onView, why }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>#</th><th>Candidate</th><th>Label</th><th>{why}</th><th /></tr></thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.resume_id}>
              <td>{c.rank}</td>
              <td><strong>{c.candidate_name}</strong><br /><span className="muted small">{c.resume_id}</span></td>
              <td><span className={`badge ${LABEL_CLASS[c.label]}`}>{c.label}</span></td>
              <td className="small">{c.reasons[0]}</td>
              <td><button type="button" className="link" onClick={() => onView(c)}>View</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Collapsible({ title, children, tone = '' }) {
  const [open, setOpen] = useState(false)
  return (
    <div className={`panel rest ${tone}`}>
      <button type="button" className="link" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        {open ? '▾' : '▸'} {title}
      </button>
      {open && children}
    </div>
  )
}

export default function Results({ data, onView, busy }) {
  const { job, summary, top_k: top, not_selected: rest, filtered_out: out = [], skipped, warnings } = data

  return (
    <section className={`results ${busy ? 'busy' : ''}`} aria-live="polite" aria-busy={busy}>
      <div className="panel job-summary">
        <div className="panel-head">
          <h2>What the job asks for</h2>
          {job.years_required != null && <span className="pill">{job.years_required}+ years</span>}
        </div>
        <Chips title="Required" items={job.required_technologies ?? job.skills_required} kind="neutral" />
        <Chips title="Also required" items={job.required_competencies} kind="neutral" />
        <Chips title="Preferred" items={job.preferred_skills} kind="soft" />
        <Chips title="Soft skills" items={job.soft_skills} kind="soft" />
        <div className="stats">
          <div><strong>{summary.scored}</strong><span>resumes ranked</span></div>
          <div><strong className="good-text">{summary.label_counts['Good Fit'] ?? 0}</strong><span>Good Fit</span></div>
          <div><strong className="potential-text">{summary.label_counts['Potential Fit'] ?? 0}</strong><span>Potential Fit</span></div>
          <div><strong className="nofit-text">{summary.label_counts['No Fit'] ?? 0}</strong><span>No Fit</span></div>
        </div>
      </div>

      {warnings.length > 0 && (
        <div className="note warn">{warnings.map((w, i) => <p key={i}>{w}</p>)}</div>
      )}
      {skipped.length > 0 && (
        <div className="note error">
          <p><strong>{skipped.length} file{skipped.length === 1 ? '' : 's'} could not be read:</strong></p>
          <ul>{skipped.map((s) => <li key={s.resume_id}>{s.resume_id}: {s.reason}</li>)}</ul>
        </div>
      )}

      <div className="results-head">
        <h2 className="section-title">Top {top.length} candidate{top.length === 1 ? '' : 's'}</h2>
        <span className="muted small">
          {orderLine(summary)}
          {summary.filtered_out > 0 && ` · ${summary.filtered_out} filtered out`}
        </span>
      </div>
      {top.length === 0 && (
        <div className="panel empty">No candidate passes the current filters. Loosen the must-haves or the minimum experience.</div>
      )}
      <div className="cards">
        {top.map((c) => <CandidateCard key={c.resume_id} c={c} onView={() => onView(c)} />)}
      </div>

      {rest.length > 0 && (
        <Collapsible title={`The other ${rest.length} candidate${rest.length === 1 ? '' : 's'} that pass`}>
          <CandidateTable rows={rest} onView={onView} why="Why not shortlisted" />
        </Collapsible>
      )}
      {out.length > 0 && (
        <Collapsible tone="filtered" title={`${out.length} filtered out by your must-haves`}>
          <CandidateTable rows={out} onView={onView} why="Failed filter" />
        </Collapsible>
      )}
    </section>
  )
}
