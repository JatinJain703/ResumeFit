import { useEffect, useState } from 'react'

/** Seconds since `running` became true (0 when not running). */
export function useElapsed(running) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (!running) return
    const t0 = Date.now()
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 500)
    return () => { clearInterval(id); setElapsed(0) }
  }, [running])
  return elapsed
}

/**
 * Blob URLs for Files (e.g. to open an uploaded PDF), revoked when the files change or on unmount.
 * Created inside the effect so React StrictMode's mount/unmount/mount cycle never leaves a revoked URL in use.
 */
export function useObjectUrls(files) {
  const [urls, setUrls] = useState([])
  useEffect(() => {
    const made = (files || []).map((f) => URL.createObjectURL(f))
    // oxlint-disable-next-line react/set-state-in-effect -- syncing with the browser's blob URL store
    setUrls(made)
    return () => made.forEach((u) => URL.revokeObjectURL(u))
  }, [files])
  return urls
}
