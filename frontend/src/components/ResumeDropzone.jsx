import { useRef, useState } from 'react'

const isPdf = (f) => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')

export default function ResumeDropzone({ files, onChange }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [rejected, setRejected] = useState([])

  function add(list) {
    const incoming = Array.from(list)
    const pdfs = incoming.filter(isPdf)
    setRejected(incoming.filter((f) => !isPdf(f)).map((f) => f.name))
    const key = (f) => `${f.name}|${f.size}`
    const have = new Set(files.map(key))
    onChange([...files, ...pdfs.filter((f) => !have.has(key(f)))])
  }

  return (
    <section className="panel">
      <div className="panel-head">
        <h2><span className="step">2</span> Resumes <span className="muted">(PDF)</span></h2>
        {files.length > 0 && (
          <button type="button" className="link" onClick={() => { onChange([]); setRejected([]) }}>Clear all</button>
        )}
      </div>

      <input ref={inputRef} type="file" accept=".pdf,application/pdf" multiple hidden
             onChange={(e) => { add(e.target.files); e.target.value = '' }} />
      <button
        type="button"
        className={`drop ${dragging ? 'dragging' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); add(e.dataTransfer.files) }}
      >
        <strong>Drop resume PDFs here or click to browse</strong>
        <span className="muted">Select many at once. 10 or more give the most reliable ranking.</span>
      </button>

      {rejected.length > 0 && (
        <p className="note warn">Ignored (not PDF): {rejected.join(', ')}</p>
      )}

      {files.length > 0 && (
        <>
          <p className="count">
            {files.length} resume{files.length === 1 ? '' : 's'} selected
            {files.length < 10 && <span className="warn-text"> · add {10 - files.length} more for a reliable ranking</span>}
          </p>
          <ul className="file-list">
            {files.map((f, i) => (
              <li key={`${f.name}|${f.size}`} className="file-row">
                <span className="file-icon" aria-hidden>PDF</span>
                <span className="file-name" title={f.name}>{f.name}</span>
                <span className="muted">{(f.size / 1024).toFixed(0)} KB</span>
                <button type="button" className="icon-btn" aria-label={`Remove ${f.name}`}
                        onClick={() => onChange(files.filter((_, j) => j !== i))}>×</button>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
