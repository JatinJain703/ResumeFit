import { useRef, useState } from 'react'

const OK = ['.pdf', '.docx', '.txt']

export default function SingleFileDrop({ file, onChange, step, title, hint }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')

  function pick(list) {
    const f = list?.[0]
    if (!f) return
    if (!OK.some((ext) => f.name.toLowerCase().endsWith(ext))) {
      setError(`"${f.name}" is not a PDF, DOCX or TXT file.`)
      return
    }
    setError('')
    onChange(f)
  }

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{step != null && <span className="step">{step}</span>} {title}</h2>
        {file && <button type="button" className="link" onClick={() => onChange(null)}>Remove</button>}
      </div>
      <input ref={inputRef} type="file" accept={OK.join(',')} hidden
             onChange={(e) => { pick(e.target.files); e.target.value = '' }} />
      {file ? (
        <div className="file-row picked">
          <span className="file-icon" aria-hidden>{file.name.split('.').pop().toUpperCase()}</span>
          <span className="file-name" title={file.name}>{file.name}</span>
          <span className="muted">{(file.size / 1024).toFixed(0)} KB</span>
          <button type="button" className="link" onClick={() => inputRef.current.click()}>Change</button>
        </div>
      ) : (
        <button type="button" className={`drop small ${dragging ? 'dragging' : ''}`}
                onClick={() => inputRef.current.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files) }}>
          <strong>Drop your resume here or click to browse</strong>
          <span className="muted">{hint}</span>
        </button>
      )}
      {error && <p className="note error">{error}</p>}
    </section>
  )
}
