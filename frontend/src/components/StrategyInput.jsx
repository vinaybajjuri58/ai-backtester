import React, { useState } from 'react'
import axios from 'axios'

const ASSETS = ['BTC/USDT', 'ETH/USDT']
const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
const LOOKBACKS = ['3 months', '6 months', '1 year', '2 years']

const EXAMPLES = [
  "Buy BTC when RSI drops below 30, sell when it crosses above 70",
  "Buy when 10-day EMA crosses above 50-day EMA, sell on cross below",
  "Buy when price touches lower Bollinger Band, sell at middle band",
  "Buy when MACD line crosses above signal line, sell on cross below",
  "Buy on 20-day high breakout, sell on 20-day low breakdown",
]

function StrategyInput({ onResults, loading, setLoading, setError }) {
  const [hypothesis, setHypothesis] = useState('')
  const [asset, setAsset] = useState('BTC/USDT')
  const [timeframe, setTimeframe] = useState('1d')
  const [lookback, setLookback] = useState('1 year')

  const handleSubmit = async () => {
    if (!hypothesis.trim()) {
      setError('Please enter a trading strategy')
      return
    }

    setLoading(true)
    setError(null)
    onResults(null)

    try {
      const response = await axios.post('/api/backtest', {
        hypothesis: hypothesis.trim(),
        asset,
        timeframe,
        lookback,
      })
      onResults(response.data)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Backtest failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const selectStyle = {
    backgroundColor: '#111',
    borderColor: '#1e1e1e',
    color: '#e5e5e5',
  }

  return (
    <div className="rounded-xl border p-6" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
      <div className="mb-5">
        <h2 className="text-lg font-medium mb-1" style={{ color: '#e5e5e5' }}>
          Strategy Hypothesis
        </h2>
        <p className="text-sm" style={{ color: '#737373' }}>
          Describe your trading strategy in plain English
        </p>
      </div>

      {/* Text area */}
      <textarea
        value={hypothesis}
        onChange={(e) => setHypothesis(e.target.value)}
        placeholder="e.g., Buy BTC when RSI drops below 30, sell when it crosses above 70"
        rows={4}
        className="w-full rounded-lg border px-4 py-3 text-sm resize-none focus:outline-none focus:ring-1 transition-all"
        style={{
          backgroundColor: '#0a0a0a',
          borderColor: '#1e1e1e',
          color: '#e5e5e5',
          focusRingColor: '#06b6d4',
        }}
        onFocus={(e) => e.target.style.borderColor = '#06b6d4'}
        onBlur={(e) => e.target.style.borderColor = '#1e1e1e'}
      />

      {/* Example strategies */}
      <div className="mt-3 flex flex-wrap gap-2">
        {EXAMPLES.map((ex, i) => (
          <button
            key={i}
            onClick={() => setHypothesis(ex)}
            className="text-xs px-3 py-1.5 rounded-full border transition-colors cursor-pointer hover:border-cyan-800"
            style={{ backgroundColor: '#0a0a0a', borderColor: '#1e1e1e', color: '#737373' }}
          >
            {ex.length > 50 ? ex.slice(0, 50) + '...' : ex}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="mt-5 grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div>
          <label className="block text-xs font-medium mb-1.5" style={{ color: '#737373' }}>Asset</label>
          <select
            value={asset}
            onChange={(e) => setAsset(e.target.value)}
            className="w-full rounded-lg border px-3 py-2.5 text-sm focus:outline-none"
            style={selectStyle}
          >
            {ASSETS.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium mb-1.5" style={{ color: '#737373' }}>Timeframe</label>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="w-full rounded-lg border px-3 py-2.5 text-sm focus:outline-none"
            style={selectStyle}
          >
            {TIMEFRAMES.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium mb-1.5" style={{ color: '#737373' }}>Lookback</label>
          <select
            value={lookback}
            onChange={(e) => setLookback(e.target.value)}
            className="w-full rounded-lg border px-3 py-2.5 text-sm focus:outline-none"
            style={selectStyle}
          >
            {LOOKBACKS.map(l => <option key={l} value={l}>{l}</option>)}
          </select>
        </div>

        <div className="flex items-end">
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full rounded-lg px-4 py-2.5 text-sm font-medium transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              backgroundColor: loading ? '#0e7490' : '#06b6d4',
              color: '#0a0a0a',
            }}
            onMouseEnter={(e) => { if (!loading) e.target.style.backgroundColor = '#22d3ee' }}
            onMouseLeave={(e) => { if (!loading) e.target.style.backgroundColor = '#06b6d4' }}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                Running...
              </span>
            ) : (
              'Run Backtest'
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

export default StrategyInput
