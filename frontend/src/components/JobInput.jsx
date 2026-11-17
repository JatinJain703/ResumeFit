import { useRef } from 'react'

const JD_ACCEPT = '.pdf,.txt,.docx'

export default function JobInput({ step, mode, onModeChange, text, onTextChange, file, onFileChange }) {
  const inputRef = useRef(null)

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{step != null && <span className="step">{step}</span>} Job description</h2>
        <div className="segmented" role="tablist" aria-label="Job description format">
          <button type="button" role="tab" aria-selected={mode === 'text'}
                  className={mode === 'text' ? 'active' : ''} onClick={() => onModeChange('text')}>
            Paste text
          </button>
          <button type="button" role="tab" aria-selected={mode === 'file'}
                  className={mode === 'file' ? 'active' : ''} onClick={() => onModeChange('file')}>
            Upload PDF
          </button>
        </div>
      </div>

      {mode === 'text' ? (
        <textarea
          className="jd-text"
          placeholder="Paste the full job description here: responsibilities, required skills, years of experience…"
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          rows={9}
        />
      ) : (
        <div className="jd-file">
          <input ref={inputRef} type="file" accept={JD_ACCEPT} hidden
                 onChange={(e) => { onFileChange(e.target.files[0] || null); e.target.value = '' }} />
          {file ? (
            <div className="file-row">
              <span className="file-icon" aria-hidden>PDF</span>
              <span className="file-name">{file.name}</span>
              <span className="muted">{(file.size / 1024).toFixed(0)} KB</span>
              <button type="button" className="link" onClick={() => onFileChange(null)}>Remove</button>
            </div>
          ) : (
            <button type="button" className="drop small" onClick={() => inputRef.current.click()}>
              <strong>Choose the job description file</strong>
              <span className="muted">PDF (or TXT / DOCX)</span>
            </button>
          )}
        </div>
      )}
    </section>
  )
}
