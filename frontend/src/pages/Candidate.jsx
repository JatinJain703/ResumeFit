import { useState } from 'react'
import { fitCheck } from '../api.js'
import FitResult from '../components/FitResult.jsx'
import JobInput from '../components/JobInput.jsx'
import SingleFileDrop from '../components/SingleFileDrop.jsx'
import { useElapsed, useObjectUrls } from '../hooks.js'

export default function Candidate({ onApiUp }) {
  const [resume, setResume] = useState(null)
  const [jdMode, setJdMode] = useState('text')
  const [jdText, setJdText] = useState('')
  const [jdFile, setJdFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const elapsed = useElapsed(loading)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)        // { data, files: [resume] }
  const [resumeUrl] = useObjectUrls(result?.files)

  const jdReady = jdMode === 'text' ? jdText.trim().length >= 30 : Boolean(jdFile)
  const canSubmit = jdReady && resume && !loading

  async function submit(e) {
    e.preventDefault()
    if (!canSubmit) return
    setLoading(true); setError('')
    try {
      const data = await fitCheck({ jdText, jdFile: jdMode === 'file' ? jdFile : null, resume })
      setResult({ data, files: [resume] })
      onApiUp?.()
      requestAnimationFrame(() => document.getElementById('fit-results')?.scrollIntoView({ behavior: 'smooth' }))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="page-head">
        <span className="mode-tag alt">For job seekers</span>
        <h1>Check my resume</h1>
        <p className="muted">See how well your resume fits a specific job, how it compares with similar resumes,
          and what to add or change before you apply.</p>
      </div>

      <form className="inputs" onSubmit={submit}>
        <SingleFileDrop step={1} title="Your resume" file={resume} onChange={setResume}
                        hint="PDF, DOCX or TXT · one file" />
        <JobInput step={2} mode={jdMode} onModeChange={setJdMode} text={jdText} onTextChange={setJdText}
                  file={jdFile} onFileChange={setJdFile} />
        <div className="submit-row end">
          <button type="submit" className="btn primary" disabled={!canSubmit}>
            {loading ? <><span className="spinner" aria-hidden /> Checking… {elapsed}s</> : 'Check my fit'}
          </button>
        </div>
        {!jdReady && jdMode === 'text' && jdText.length > 0 && (
          <p className="muted small">The job description looks too short. Paste the full posting.</p>
        )}
        {error && <div className="note error" role="alert">{error}</div>}
      </form>

      <div id="fit-results">
        {result && <FitResult data={result.data} resumeUrl={resumeUrl} />}
      </div>
      {result && (
        <p className="muted small center">
          Change the job description or upload a new version of your resume, then check again.
        </p>
      )}
    </div>
  )
}
