import { useState } from 'react'

const SLIDERS = [
  { key: 'skills', label: 'Skills', hint: 'share of the job’s skills found in the resume' },
  { key: 'experience', label: 'Experience', hint: 'years found vs years required' },
  { key: 'semantic', label: 'Meaning', hint: 'BERT similarity of the whole resume to the job' },
  { key: 'keyword', label: 'Keywords', hint: 'TF-IDF overlap with the job’s wording' },
]

export default function TunePanel({ value: t, onChange, job, busy, error, isDefault, onReset, maxK }) {
  const [skillInput, setSkillInput] = useState('')
  const set = (patch) => onChange({ ...t, ...patch })
  const sum = SLIDERS.reduce((a, s) => a + t.weights[s.key], 0)
  const total = sum || 1

  function addSkills(text) {
    const add = text.split(',').map((s) => s.trim()).filter(Boolean)
    const have = new Set(t.mustHave.map((s) => s.toLowerCase()))
    const next = [...t.mustHave, ...add.filter((s) => !have.has(s.toLowerCase()))].slice(0, 20)
    if (next.length !== t.mustHave.length) set({ mustHave: next })
    setSkillInput('')
  }

  const suggestions = (job.required_technologies ?? job.skills_required ?? [])
    .filter((s) => !t.mustHave.some((m) => m.toLowerCase() === s)).slice(0, 8)

  return (
    <aside className="panel tune" aria-label="Tune the ranking">
      <div className="panel-head">
        <h2>Tune the ranking {busy && <span className="spinner dark" aria-label="Updating" />}</h2>
        {!isDefault && <button type="button" className="link" onClick={onReset}>Reset</button>}
      </div>

      <div className="tune-group">
        <h3>Order by</h3>
        <div className="segmented full" role="radiogroup" aria-label="Ranking method">
          <button type="button" role="radio" aria-checked={t.mode === 'model'}
                  className={t.mode === 'model' ? 'active' : ''} onClick={() => set({ mode: 'model' })}>
            AI model
          </button>
          <button type="button" role="radio" aria-checked={t.mode === 'formula'}
                  className={t.mode === 'formula' ? 'active' : ''} onClick={() => set({ mode: 'formula' })}>
            My weights
          </button>
        </div>
        <p className="muted small">
          {t.mode === 'model'
            ? 'The trained ranking model (most accurate in testing). Move a slider to use your own weights.'
            : 'A transparent weighted score using the sliders below.'}
        </p>
      </div>

      <div className={`tune-group ${t.mode === 'model' ? 'dim' : ''}`}>
        <h3>Weights</h3>
        {SLIDERS.map((s) => (
          <label key={s.key} className="slider" title={s.hint}>
            <span className="slider-top">
              <span>{s.label}</span>
              <span className="slider-val">{Math.round((t.weights[s.key] / total) * 100)}%</span>
            </span>
            <input type="range" min={0} max={100} step={5} value={t.weights[s.key]}
                   aria-label={`${s.label} weight`}
                   onChange={(e) => set({ mode: 'formula', weights: { ...t.weights, [s.key]: Number(e.target.value) } })} />
          </label>
        ))}
        {sum === 0 && <p className="warn-text small">Set at least one weight above 0.</p>}
      </div>

      <div className="tune-group">
        <h3>Must-have skills</h3>
        <div className="chips tight">
          {t.mustHave.map((s) => (
            <span key={s} className="chip must">
              {s}
              <button type="button" aria-label={`Remove ${s}`}
                      onClick={() => set({ mustHave: t.mustHave.filter((x) => x !== s) })}>×</button>
            </span>
          ))}
        </div>
        <form className="inline-add" onSubmit={(e) => { e.preventDefault(); addSkills(skillInput) }}>
          <input type="text" value={skillInput} placeholder="e.g. Python, AWS" aria-label="Add a must-have skill"
                 onChange={(e) => setSkillInput(e.target.value)} />
          <button type="submit" className="btn ghost" disabled={!skillInput.trim()}>Add</button>
        </form>
        {suggestions.length > 0 && (
          <div className="chips tight suggest">
            <span className="muted small">From the job:</span>
            {suggestions.map((s) => (
              <button key={s} type="button" className="chip add" onClick={() => addSkills(s)}>+ {s}</button>
            ))}
          </div>
        )}
      </div>

      <div className="tune-group">
        <h3>Minimum experience</h3>
        <div className="row">
          <input type="number" min={0} max={60} step={0.5} className="num" value={t.minYears} placeholder="any"
                 aria-label="Minimum years of experience"
                 onChange={(e) => set({ minYears: e.target.value })} />
          <span className="muted">years</span>
        </div>
        {t.minYears !== '' && (
          <label className="check">
            <input type="checkbox" checked={t.unknownYears === 'exclude'}
                   onChange={(e) => set({ unknownYears: e.target.checked ? 'exclude' : 'keep' })} />
            Also exclude resumes where no experience could be found
          </label>
        )}
      </div>

      <div className="tune-group">
        <h3>Shortlist size</h3>
        <div className="row">
          <input type="number" min={1} max={Math.max(1, maxK)} className="num" value={t.topK}
                 aria-label="Shortlist size"
                 onChange={(e) => set({ topK: Math.max(1, parseInt(e.target.value, 10) || 1) })} />
          <span className="muted">of {maxK}</span>
        </div>
      </div>

      {error && <div className="note error" role="alert">{error}</div>}
    </aside>
  )
}
