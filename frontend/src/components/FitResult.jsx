import { LABEL_CLASS } from '../constants.js'

const PRIORITY_TEXT = { high: 'Do first', medium: 'Worth doing', low: 'Nice to have' }
const AREA_TEXT = { skills: 'Skills', experience: 'Experience', keywords: 'Wording', content: 'Content' }

function Gauge({ value, cls }) {
  const r = 52
  const c = 2 * Math.PI * r
  return (
    <div className={`gauge ${cls}`} role="img" aria-label={`Match score ${value} out of 100`}>
      <svg viewBox="0 0 120 120" width="132" height="132">
        <circle cx="60" cy="60" r={r} className="gauge-track" />
        <circle cx="60" cy="60" r={r} className="gauge-fill"
                strokeDasharray={`${(value / 100) * c} ${c}`} transform="rotate(-90 60 60)" />
      </svg>
      <div className="gauge-text"><strong>{value}</strong><span>/ 100</span></div>
    </div>
  )
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

function SkillColumn({ title, items, kind, empty }) {
  return (
    <div className="gap-col">
      <h4>{title} <span className="muted">({items.length})</span></h4>
      {items.length === 0 ? <p className="muted small">{empty}</p> : (
        <div className="chips tight">
          {items.map((d) => (
            <span key={d.skill} className={`chip ${kind} ${d.importance === 'required' && kind === 'miss' ? 'strong' : ''}`}
                  title={`${d.importance} ${d.kind}${d.via ? ` · via ${d.via}` : ''}`}>
              {d.skill}{d.via ? (kind === 'partial' ? ` ≈ ${d.via}` : ` (via ${d.via})`) : ''}
              {d.importance !== 'required' && <small> · {d.importance}</small>}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export default function FitResult({ data, resumeUrl }) {
  const cls = LABEL_CLASS[data.label]
  const cmp = data.comparison
  const exp = data.experience
  const groups = ['high', 'medium', 'low'].map((p) => [p, data.suggestions.filter((s) => s.priority === p)])
    .filter(([, list]) => list.length)

  return (
    <section className="results fit" aria-live="polite">
      <div className={`panel fit-hero ${cls}`}>
        <Gauge value={data.match_score} cls={cls} />
        <div className="fit-summary">
          <div className="fit-title">
            <span className={`badge big ${cls}`}>{data.label}</span>
            <span className="muted small">{Math.round(data.confidence * 100)}% confidence</span>
            {data.label_capped && <span className="pill warn">capped: no core technology found</span>}
          </div>
          {data.name_source === 'resume' && <span className="muted small">{data.candidate_name}</span>}
          <h2>{data.headline.split('. ')[0]}.</h2>
          <div className="compare">
            <div className="compare-track" aria-hidden>
              <span className="compare-marker" style={{ left: `${cmp.percentile}%` }} />
            </div>
            <div className="compare-labels"><span>weakest</span><span>strongest</span></div>
            <p className="small">
              You would rank <strong>#{cmp.rank_in_pool} of {cmp.pool_size + 1}</strong> among the most similar resumes in our
              database: better than <strong>{Math.round(cmp.percentile)}%</strong> of them.
            </p>
          </div>
          {resumeUrl && <a className="link small" href={resumeUrl} target="_blank" rel="noreferrer">Open my resume ↗</a>}
        </div>
      </div>

      {data.warnings.length > 0 && <div className="note warn">{data.warnings.map((w, i) => <p key={i}>{w}</p>)}</div>}

      <div className="fit-grid">
        <div className="panel">
          <h3 className="panel-title">What to improve</h3>
          {groups.length === 0 && <p className="muted">No changes needed for this job. Nice work.</p>}
          {groups.map(([p, list]) => (
            <div key={p} className="sugg-group">
              <span className={`prio ${p}`}>{PRIORITY_TEXT[p]}</span>
              <ol className="sugg">
                {list.map((s, i) => (
                  <li key={i}><span className="area">{AREA_TEXT[s.area]}</span>{s.text}</li>
                ))}
              </ol>
            </div>
          ))}
        </div>

        <div className="fit-side">
          <div className="panel">
            <h3 className="panel-title">Strengths</h3>
            <ul className="ticks">{data.strengths.map((s, i) => <li key={i}>{s}</li>)}</ul>
          </div>
          <div className="panel">
            <h3 className="panel-title">Score breakdown</h3>
            <div className="bars">
              <Bar label="Skills" value={data.score_breakdown.skills} />
              <Bar label="Experience" value={data.score_breakdown.experience} />
              <Bar label="Meaning" value={data.score_breakdown.semantic} />
              <Bar label="Keywords" value={data.score_breakdown.keyword} />
            </div>
            <p className="meta">
              Experience: <strong>{exp.found_years == null ? 'not found' : `${exp.found_years} years`}</strong>
              {exp.found_from && <span> ({exp.found_from})</span>}
              {exp.required_years != null && <> · job asks for <strong>{exp.required_years}+</strong></>}
            </p>
            <p className="muted small">Meaning and keywords are relative to the similar resumes you were compared with.</p>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3 className="panel-title">Skill gap</h3>
        <div className="gap-grid">
          <SkillColumn title="You have" items={data.skills.matched} kind="ok" empty="None of the job’s skills were found." />
          <SkillColumn title="Related" items={data.skills.related} kind="partial" empty="–" />
          <SkillColumn title="Missing" items={data.skills.missing} kind="miss" empty="Nothing missing." />
        </div>
        {data.missing_keywords.length > 0 && (
          <div className="kw">
            <h4>Words from the job that your resume doesn’t use</h4>
            <div className="chips tight">
              {data.missing_keywords.map((k) => <span key={k} className="chip soft">{k}</span>)}
            </div>
            <p className="muted small">Use them where they are true for you; recruiters and screening software look for the job’s own wording.</p>
          </div>
        )}
      </div>
    </section>
  )
}
