import { useEffect, useState } from 'react'

/** Tiny hash router: #/recruiter -> "/recruiter". Works on any static host, no server config. */
export const ROUTES = ['/', '/recruiter', '/candidate', '/how-it-works']

function current() {
  const path = window.location.hash.replace(/^#/, '') || '/'
  return ROUTES.includes(path) ? path : '/'
}

export function useRoute() {
  const [route, setRoute] = useState(current)
  useEffect(() => {
    const onChange = () => {
      setRoute(current())
      window.scrollTo({ top: 0 })
    }
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}
