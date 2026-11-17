import { useEffect } from 'react'

export default function ResumeViewer({ candidate, url, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => { window.removeEventListener('keydown', onKey); document.body.style.overflow = '' }
  }, [onClose])

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={`Resume of ${candidate.candidate_name}`}
           onClick={(e) => e.stopPropagation()}>
        <header className="modal-head">
          <div>
            <strong>#{candidate.rank} {candidate.candidate_name}</strong>
            <span className="muted"> · {candidate.resume_id}</span>
          </div>
          <div className="modal-actions">
            <a className="btn ghost" href={url} target="_blank" rel="noreferrer">Open in new tab</a>
            <a className="btn ghost" href={url} download={candidate.resume_id}>Download</a>
            <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>×</button>
          </div>
        </header>
        <iframe title={`Resume of ${candidate.candidate_name}`} src={url} className="pdf-frame" />
      </div>
    </div>
  )
}
