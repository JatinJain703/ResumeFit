export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

/** Turn any FastAPI error body into one readable sentence. */
function errorMessage(status, body) {
  const d = body?.detail
  if (typeof d === 'string') return d
  if (d?.message) {
    const skipped = (d.skipped || []).map((s) => `${s.resume_id}: ${s.reason}`).join('; ')
    return skipped ? `${d.message} (${skipped})` : d.message
  }
  if (Array.isArray(d)) return d.map((e) => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('; ')
  return `Request failed (HTTP ${status})`
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(`${API_URL}${path}`, options)
  } catch {
    throw new ApiError(`Cannot reach the ResumeFit API at ${API_URL}. Is the backend running?`, 0)
  }
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new ApiError(errorMessage(res.status, body), res.status)
  return body
}

const json = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

function appendJd(form, jdText, jdFile) {
  if (jdFile) form.append('jd_file', jdFile)
  else form.append('job_description', jdText)
}

/**
 * POST /rank/upload - score many resumes once. The response has a session_id for rerank().
 * @param {{jdText: string, jdFile: File|null, resumes: File[], topK: number}} input
 */
export function rankResumes({ jdText, jdFile, resumes, topK }) {
  const form = new FormData()
  appendJd(form, jdText, jdFile)
  for (const f of resumes) form.append('files', f)
  form.append('top_k', String(topK))
  return request('/rank/upload', { method: 'POST', body: form })
}

/**
 * POST /rerank - instant re-ordering of an earlier ranking with new weights / filters.
 * @param {{sessionId: string, topK: number, ranking: 'model'|'formula', weights?: object, filters?: object}} input
 */
export function rerank({ sessionId, topK, ranking, weights, filters, signal }) {
  return request('/rerank', {
    ...json({ session_id: sessionId, top_k: topK, ranking, weights, filters }),
    signal,
  })
}

/** POST /fit-check/upload - one resume against one job, with gaps and suggestions. */
export function fitCheck({ jdText, jdFile, resume }) {
  const form = new FormData()
  appendJd(form, jdText, jdFile)
  form.append('resume', resume)
  return request('/fit-check/upload', { method: 'POST', body: form })
}

export const modelInfo = () => request('/model-info')
export const checkHealth = () => request('/health')
