import { LABEL_CLASS } from '../constants.js'

function Experience({ exp }) {
  if (exp.found_years == null && exp.required_years == null) return null
  const found = exp.found_years == null ? 'not found' : `${exp.found_years} yrs`
  const need = exp.required_years == null ? '' : ` / ${exp.required_years} required`
  return <span className="meta-item">Experience: <strong>{found}</strong>{need}</span>
}

function Bar({ label, value }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div className="bar">
      <span className="bar-label">{label}</span>
      <span className="bar-track"><span className="bar-fill" style={{ width: `${pct}%` }} /></span>
      <span className="bar-value">{pct}%</span>
    </div>
  )
}

export default function CandidateCard({ c, onView }) {
  const missingRequired = (c.skill_details || [])
    .filter((d) => d.status === 'missing' && d.importance === 'required').map((d) => d.skill)
  const related = (c.skill_details || []).filter((d) => d.status === 'related')

  return (
    <article className="card">
      <div className="card-top">
        <span className="rank">#{c.rank}</span>
        <div className="who">
          <h3>{c.candidate_name}</h3>
          <span className="muted file-ref" title={c.resume_id}>{c.resume_id}</span>
        </div>
        <div className="label-box">
          <span className={`badge ${LABEL_CLASS[c.label]}`}>{c.label}</span>
          <span className="muted small">{Math.round(c.confidence * 100)}% confidence</span>
        </div>
      </div>

      <div className="card-body">
        <div className="reasons">
          <h4>Why this candidate</h4>
          <ul>{c.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        </div>

        <div className="side">
          <div className="bars">
            <Bar label="Skills" value={c.score_breakdown.skills} />
            <Bar label="Experience" value={c.score_breakdown.experience} />
            <Bar label="Meaning" value={c.score_breakdown.semantic} />
            <Bar label="Keywords" value={c.score_breakdown.keyword} />
          </div>
          <div className="meta"><Experience exp={c.experience} /></div>
        </div>
      </div>

      {(c.matched_skills.length > 0 || c.missing_skills.length > 0 || c.must_have?.length > 0) && (
        <div className="chips-block">
          {c.must_have?.length > 0 && (
            <div className="chips">
              <span className="chips-label">Must-haves</span>
              {c.must_have.map((m) => (
                <span key={m.skill} className={`chip ${m.found ? 'ok' : 'miss strong'}`}
                      title={m.matched_as === 'text' ? 'not in the skill catalog: matched as plain text' : undefined}>
                  {m.found ? '✓' : '✗'} {m.skill}{m.via ? ` (via ${m.via})` : ''}
                </span>
              ))}
            </div>
          )}
          {c.matched_skills.length > 0 && (
            <div className="chips">
              <span className="chips-label">Has</span>
              {c.matched_skills.map((s) => <span key={s} className="chip ok">{s}</span>)}
              {related.map((d) => (
                <span key={d.skill} className="chip partial" title={`related evidence: ${d.via}`}>{d.skill} ≈ {d.via}</span>
              ))}
            </div>
          )}
          {c.missing_skills.length > 0 && (
            <div className="chips">
              <span className="chips-label">Missing</span>
              {c.missing_skills.map((s) => (
                <span key={s} className={`chip miss ${missingRequired.includes(s) ? 'strong' : ''}`}>{s}</span>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="card-actions">
        <span className="muted small">
          Match score {Math.round(c.score_breakdown.formula_score * 100)}/100
        </span>
        <button type="button" className="btn" onClick={onView}>View resume</button>
      </div>
    </article>
  )
}
