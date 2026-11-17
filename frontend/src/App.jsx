import { useEffect, useState } from 'react'
import { API_URL, checkHealth } from './api.js'
import Link from './components/Link.jsx'
import { useRoute } from './router.js'
import Home from './pages/Home.jsx'
import Recruiter from './pages/Recruiter.jsx'
import Candidate from './pages/Candidate.jsx'
import HowItWorks from './pages/HowItWorks.jsx'

const NAV = [
  { to: '/recruiter', label: 'Rank candidates' },
  { to: '/candidate', label: 'Check my resume' },
  { to: '/how-it-works', label: 'How it works' },
]

export default function App() {
  const route = useRoute()
  const [apiUp, setApiUp] = useState(null)
  const [menuOpen, setMenuOpen] = useState(false)
  // keep pages mounted once visited, so uploads and results survive navigation
  const [visited, setVisited] = useState(() => new Set([route]))
  if (!visited.has(route)) setVisited(new Set(visited).add(route))      // derived during render

  useEffect(() => {
    checkHealth().then(() => setApiUp(true)).catch(() => setApiUp(false))
  }, [])

  const page = (path, el) => visited.has(path) && <div hidden={route !== path}>{el}</div>

  return (
    <div className="shell">
      <header className="nav">
        <div className="nav-inner">
          <Link to="/" className="brand" aria-label="ResumeFit home" onClick={() => setMenuOpen(false)}>
            <img src="/favicon.svg" alt="" width="28" height="28" />
            <span>ResumeFit</span>
          </Link>
          <button type="button" className="icon-btn menu-btn" aria-label="Menu" aria-expanded={menuOpen}
                  onClick={() => setMenuOpen((o) => !o)}>☰</button>
          <nav className={menuOpen ? 'open' : ''}>
            {NAV.map((n) => (
              <Link key={n.to} to={n.to} className={route === n.to ? 'active' : ''} onClick={() => setMenuOpen(false)}
                    aria-current={route === n.to ? 'page' : undefined}>{n.label}</Link>
            ))}
            <span className={`status ${apiUp === false ? 'down' : apiUp ? 'up' : ''}`}
                  title={apiUp === false ? `Start the backend at ${API_URL}` : API_URL}>
              <span className="dot" aria-hidden />
              {apiUp == null ? 'Checking API' : apiUp ? 'API online' : 'API offline'}
            </span>
          </nav>
        </div>
      </header>

      <main className="app">
        {page('/', <Home />)}
        {page('/recruiter', <Recruiter onApiUp={() => setApiUp(true)} />)}
        {page('/candidate', <Candidate onApiUp={() => setApiUp(true)} />)}
        {page('/how-it-works', <HowItWorks />)}
      </main>

      <footer className="footer">
        <div className="footer-inner">
          <span><strong>ResumeFit</strong> · hybrid resume–job matching with explainable results</span>
          <span className="muted">TF-IDF · BERT (all-MiniLM-L6-v2) · 394-skill catalog · Gradient Boosting + XGBoost ranker</span>
        </div>
      </footer>
    </div>
  )
}
