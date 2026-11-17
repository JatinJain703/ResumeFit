import { useEffect, useRef, useState } from 'react'
import { rankResumes, rerank } from '../api.js'
import JobInput from '../components/JobInput.jsx'
import ResumeDropzone from '../components/ResumeDropzone.jsx'
import Results from '../components/Results.jsx'
import ResumeViewer from '../components/ResumeViewer.jsx'
import TunePanel from '../components/TunePanel.jsx'
import { DEFAULT_TUNE, isDefaultTune } from '../constants.js'
import { useElapsed, useObjectUrls } from '../hooks.js'

function tuneRequest(t) {
  const filters = {}
  if (t.mustHave.length) filters.must_have_skills = t.mustHave
  if (t.minYears !== '' && !Number.isNaN(Number(t.minYears))) {
    filters.min_years = Number(t.minYears)
    filters.unknown_years = t.unknownYears
  }
  return {
    ranking: t.mode,
    weights: t.mode === 'formula' ? t.weights : undefined,
    filters: Object.keys(filters).length ? filters : undefined,
  }
}

export default function Recruiter({ onApiUp }) {
  const [jdMode, setJdMode] = useState('text')
  const [jdText, setJdText] = useState('')
  const [jdFile, setJdFile] = useState(null)
  const [resumes, setResumes] = useState([])
  const [topK, setTopK] = useState(5)

  const [loading, setLoading] = useState(false)
  const elapsed = useElapsed(loading)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)       // { data, files, sessionId }
  const [viewing, setViewing] = useState(null)

  const [tune, setTune] = useState(DEFAULT_TUNE)
  const [tuning, setTuning] = useState(false)
  const [tuneError, setTuneError] = useState('')
  const firstTune = useRef(true)

  // one blob URL per submitted resume, so "View resume" opens the original PDF
  const urls = useObjectUrls(result?.files)

  const jdReady = jdMode === 'text' ? jdText.trim().length >= 30 : Boolean(jdFile)
  const canSubmit = jdReady && resumes.length > 0 && topK >= 1 && !loading

  async function submit(e) {
    e.preventDefault()
    if (!canSubmit) return
    setLoading(true); setError(''); setResult(null); setTuneError('')
    const files = [...resumes]
    try {
      const data = await rankResumes({ jdText, jdFile: jdMode === 'file' ? jdFile : null, resumes: files, topK })
      firstTune.current = true
      setTune({ ...DEFAULT_TUNE, topK })
      setResult({ data, files, sessionId: data.session_id })
      onApiUp?.()
      requestAnimationFrame(() => document.getElementById('rec-results')?.scrollIntoView({ behavior: 'smooth' }))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // re-rank on every tuning change (debounced; stale responses are aborted)
  const sessionId = result?.sessionId
  useEffect(() => {
    if (!sessionId) return
    if (firstTune.current) { firstTune.current = false; return }
    const ctrl = new AbortController()
    const timer = setTimeout(async () => {
      setTuning(true)
      try {
        const data = await rerank({ sessionId, topK: tune.topK, ...tuneRequest(tune), signal: ctrl.signal })
        setResult((r) => (r && r.sessionId === sessionId ? { ...r, data } : r))
        setTuneError('')
      } catch (err) {
        if (err.name === 'AbortError') return
        setTuneError(err.status === 404
          ? 'This ranking expired on the server (sessions last 1 hour). Click "Rank candidates" again; repeated resumes are fast.'
          : err.message)
      } finally {
        if (!ctrl.signal.aborted) setTuning(false)
      }
    }, 250)
    return () => { clearTimeout(timer); ctrl.abort() }
  }, [tune, sessionId])

  return (
    <div className="page">
      <div className="page-head">
        <span className="mode-tag">For recruiters</span>
        <h1>Rank candidates</h1>
        <p className="muted">Upload a job description and the resumes you received. ResumeFit returns the best
          candidates with a label and the reasons. Then tune the weights and must-haves; the list updates instantly.</p>
      </div>

      <form className="inputs" onSubmit={submit}>
        <JobInput step={1} mode={jdMode} onModeChange={setJdMode} text={jdText} onTextChange={setJdText}
                  file={jdFile} onFileChange={setJdFile} />
        <ResumeDropzone files={resumes} onChange={setResumes} />

        <div className="submit-row">
          <label className="topk">
            <span className="step">3</span> Shortlist size
            <input type="number" min={1} max={Math.max(1, resumes.length || 50)} value={topK}
                   onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value, 10) || 1))} />
          </label>
          <button type="submit" className="btn primary" disabled={!canSubmit}>
            {loading ? <><span className="spinner" aria-hidden /> Ranking… {elapsed}s</> : 'Rank candidates'}
          </button>
        </div>
        {!jdReady && jdMode === 'text' && jdText.length > 0 && (
          <p className="muted small">The job description looks too short. Paste the full posting.</p>
        )}
        {loading && (
          <p className="muted small">Reading PDFs and comparing each resume with the job. New resumes take about 0.2–0.4 s each.</p>
        )}
        {error && <div className="note error" role="alert">{error}</div>}
      </form>

      <div id="rec-results">
        {result && (
          <div className="results-layout">
            <TunePanel value={tune} onChange={setTune} job={result.data.job} busy={tuning} error={tuneError}
                       isDefault={isDefaultTune(tune, topK)} onReset={() => setTune({ ...DEFAULT_TUNE, topK })}
                       maxK={result.data.summary.scored} />
            <Results data={result.data} onView={setViewing} busy={tuning} />
          </div>
        )}
      </div>

      {viewing && (
        <ResumeViewer candidate={viewing} url={urls[viewing.upload_index]} onClose={() => setViewing(null)} />
      )}
    </div>
  )
}
