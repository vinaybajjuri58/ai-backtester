import React, { useState } from 'react'
import StrategyInput from './components/StrategyInput'
import ResultsDashboard from './components/ResultsDashboard'

function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#0a0a0a' }}>
      {/* Header */}
      <header className="border-b" style={{ borderColor: '#1e1e1e' }}>
        <div className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ backgroundColor: '#06b6d4' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0a0a0a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
            </div>
            <h1 className="text-xl font-semibold tracking-tight" style={{ color: '#e5e5e5' }}>
              AI Backtester
            </h1>
          </div>
          <span className="text-xs font-medium px-2.5 py-1 rounded-full" style={{ backgroundColor: '#111', color: '#737373', border: '1px solid #1e1e1e' }}>
            v1.0
          </span>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <StrategyInput
          onResults={setResults}
          loading={loading}
          setLoading={setLoading}
          setError={setError}
        />

        {error && (
          <div className="mt-6 p-4 rounded-lg border" style={{ backgroundColor: '#1a0a0a', borderColor: '#3f1111', color: '#ef4444' }}>
            <p className="text-sm">{error}</p>
          </div>
        )}

        {results && (
          <ResultsDashboard results={results} />
        )}
      </main>
    </div>
  )
}

export default App
