import React, { useState } from 'react'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'

/* ─── Strategy Classification Badge ─── */
function ClassificationBadge({ classification }) {
  if (!classification) return null

  const isMR = classification.classification === 'mean_reversion'
  const isTF = classification.classification === 'trend_following'

  const colors = isMR
    ? { bg: '#0c1f3f', border: '#1e3a5f', text: '#60a5fa', icon: '↩️' }
    : isTF
      ? { bg: '#1a2e0a', border: '#2d4a14', text: '#86efac', icon: '📈' }
      : { bg: '#1e1e1e', border: '#333', text: '#737373', icon: '❓' }

  const [expanded, setExpanded] = useState(false)

  return (
    <div className="rounded-xl border p-5" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl">{colors.icon}</span>
            <span
              className="text-sm font-semibold px-3 py-1.5 rounded-full"
              style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}`, color: colors.text }}
            >
              {classification.label}
            </span>
            <span
              className="text-xs px-2 py-0.5 rounded-full"
              style={{
                backgroundColor: classification.confidence === 'high' ? '#052e16' : '#1a1a0a',
                color: classification.confidence === 'high' ? '#22c55e' : '#eab308',
                border: `1px solid ${classification.confidence === 'high' ? '#14532d' : '#3f3f11'}`,
              }}
            >
              {classification.confidence} confidence
            </span>
          </div>
          <p className="text-sm leading-relaxed" style={{ color: '#a3a3a3' }}>
            {classification.description}
          </p>
        </div>
      </div>

      {classification.reasoning && (
        <div className="mt-3">
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs flex items-center gap-1 cursor-pointer transition-colors"
            style={{ color: '#737373' }}
            onMouseEnter={e => e.target.style.color = '#a3a3a3'}
            onMouseLeave={e => e.target.style.color = '#737373'}
          >
            <svg
              width="12" height="12" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round"
              style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}
            >
              <polyline points="6 9 12 15 18 9"/>
            </svg>
            {expanded ? 'Hide' : 'Show'} reasoning
          </button>
          {expanded && (
            <div className="mt-2 text-xs leading-relaxed rounded-lg p-3" style={{ backgroundColor: '#0a0a0a', color: '#737373', border: '1px solid #1e1e1e' }}>
              💡 {classification.reasoning}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Metric Card ─── */
function MetricCard({ label, value, suffix = '', positive }) {
  const color = positive === undefined ? '#e5e5e5'
    : positive ? '#22c55e' : '#ef4444'

  return (
    <div className="rounded-lg border p-4" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
      <p className="text-xs font-medium mb-1" style={{ color: '#737373' }}>{label}</p>
      <p className="text-xl font-semibold tabular-nums" style={{ color }}>
        {value}{suffix}
      </p>
    </div>
  )
}

/* ─── Custom Tooltip ─── */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border px-3 py-2 text-xs" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
      <p style={{ color: '#737373' }}>{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color }}>
          {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : entry.value}
        </p>
      ))}
    </div>
  )
}

/* ─── Walk-Forward Table ─── */
function WalkForwardTable({ data }) {
  if (!data) return null
  const { in_sample, out_of_sample, robustness_score } = data

  const rows = [
    { metric: 'Total Return', is: in_sample.total_return, oos: out_of_sample.total_return, suffix: '%' },
    { metric: 'Sharpe Ratio', is: in_sample.sharpe_ratio, oos: out_of_sample.sharpe_ratio, suffix: '' },
    { metric: 'Win Rate', is: in_sample.win_rate, oos: out_of_sample.win_rate, suffix: '%' },
    { metric: 'Max Drawdown', is: in_sample.max_drawdown, oos: out_of_sample.max_drawdown, suffix: '%' },
    { metric: 'Total Trades', is: in_sample.total_trades, oos: out_of_sample.total_trades, suffix: '' },
  ]

  return (
    <div className="rounded-xl border p-6" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-base font-medium" style={{ color: '#e5e5e5' }}>Walk-Forward Analysis</h3>
        <span className="text-xs px-2.5 py-1 rounded-full" style={{
          backgroundColor: robustness_score > 50 ? '#052e16' : '#1a0a0a',
          color: robustness_score > 50 ? '#22c55e' : '#ef4444',
          border: `1px solid ${robustness_score > 50 ? '#14532d' : '#3f1111'}`,
        }}>
          Robustness: {robustness_score}%
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: '1px solid #1e1e1e' }}>
              <th className="text-left py-2 font-medium" style={{ color: '#737373' }}>Metric</th>
              <th className="text-right py-2 font-medium" style={{ color: '#06b6d4' }}>In-Sample (70%)</th>
              <th className="text-right py-2 font-medium" style={{ color: '#a78bfa' }}>Out-of-Sample (30%)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.metric} style={{ borderBottom: '1px solid #1e1e1e' }}>
                <td className="py-2.5" style={{ color: '#a3a3a3' }}>{r.metric}</td>
                <td className="text-right py-2.5 tabular-nums" style={{ color: '#e5e5e5' }}>{r.is}{r.suffix}</td>
                <td className="text-right py-2.5 tabular-nums" style={{ color: '#e5e5e5' }}>{r.oos}{r.suffix}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ─── Code Viewer ─── */
function CodeViewer({ code }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-xl border" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-5 cursor-pointer"
      >
        <h3 className="text-base font-medium" style={{ color: '#e5e5e5' }}>Generated Python Code</h3>
        <svg
          width="16" height="16" viewBox="0 0 24 24" fill="none"
          stroke="#737373" strokeWidth="2" strokeLinecap="round"
          style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}
        >
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </button>
      {open && (
        <div className="px-5 pb-5">
          <pre className="text-xs leading-relaxed overflow-x-auto max-h-96 overflow-y-auto" style={{
            backgroundColor: '#0a0a0a',
            border: '1px solid #1e1e1e',
            borderRadius: '8px',
            padding: '16px',
          }}>
            <code style={{ color: '#a5f3fc' }}>{code}</code>
          </pre>
          <button
            onClick={() => navigator.clipboard.writeText(code)}
            className="mt-3 text-xs px-3 py-1.5 rounded-lg border transition-colors cursor-pointer"
            style={{ borderColor: '#1e1e1e', color: '#737373', backgroundColor: '#0a0a0a' }}
          >
            Copy to clipboard
          </button>
        </div>
      )}
    </div>
  )
}

/* ─── Main Dashboard ─── */
function ResultsDashboard({ results }) {
  if (!results) return null

  const { metrics, equity_curve, monte_carlo, walk_forward, generated_code, strategy_rules, classification } = results

  // Build Monte Carlo chart data
  const mcData = monte_carlo?.x_axis?.map((x, i) => ({
    trade: x,
    p5: monte_carlo.percentile_5[i],
    p25: monte_carlo.percentile_25[i],
    p50: monte_carlo.percentile_50[i],
    p75: monte_carlo.percentile_75[i],
    p95: monte_carlo.percentile_95[i],
  })) || []

  return (
    <div className="mt-8 space-y-6">
      {/* Strategy Classification — TOP OF RESULTS */}
      <ClassificationBadge classification={classification} />

      {/* Strategy Summary */}
      {strategy_rules && (
        <div className="rounded-xl border p-5" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
          <h3 className="text-sm font-medium mb-2" style={{ color: '#06b6d4' }}>
            Parsed Strategy: {strategy_rules.description}
          </h3>
          <div className="flex flex-wrap gap-4 text-xs" style={{ color: '#a3a3a3' }}>
            <span>Entry: {strategy_rules.entry_condition}</span>
            <span>•</span>
            <span>Exit: {strategy_rules.exit_condition}</span>
          </div>
        </div>
      )}

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Total Return" value={metrics.total_return} suffix="%" positive={metrics.total_return > 0} />
        <MetricCard label="Win Rate" value={metrics.win_rate} suffix="%" positive={metrics.win_rate > 50} />
        <MetricCard label="Sharpe Ratio" value={metrics.sharpe_ratio} positive={metrics.sharpe_ratio > 0} />
        <MetricCard label="Profit Factor" value={metrics.profit_factor} positive={metrics.profit_factor > 1} />
        <MetricCard label="Max Drawdown" value={metrics.max_drawdown} suffix="%" positive={false} />
        <MetricCard label="Total Trades" value={metrics.total_trades} />
        <MetricCard label="Avg Win" value={metrics.avg_win} suffix="%" positive={true} />
        <MetricCard label="Avg Loss" value={metrics.avg_loss} suffix="%" positive={false} />
      </div>

      {/* Equity Curve */}
      <div className="rounded-xl border p-6" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
        <h3 className="text-base font-medium mb-4" style={{ color: '#e5e5e5' }}>Equity Curve</h3>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={equity_curve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
            <XAxis
              dataKey="date"
              stroke="#404040"
              tick={{ fontSize: 11, fill: '#737373' }}
              tickFormatter={(v) => v?.split(' ')[0] || v}
              interval="preserveStartEnd"
            />
            <YAxis
              stroke="#404040"
              tick={{ fontSize: 11, fill: '#737373' }}
              tickFormatter={(v) => `$${v.toLocaleString()}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="equity"
              stroke="#06b6d4"
              strokeWidth={2}
              dot={false}
              name="Equity"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Monte Carlo Simulation */}
      {mcData.length > 0 && (
        <div className="rounded-xl border p-6" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
          <h3 className="text-base font-medium mb-1" style={{ color: '#e5e5e5' }}>Monte Carlo Simulation</h3>
          <p className="text-xs mb-4" style={{ color: '#737373' }}>1,000 simulations — confidence bands (5th–95th percentile)</p>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={mcData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
              <XAxis dataKey="trade" stroke="#404040" tick={{ fontSize: 11, fill: '#737373' }} label={{ value: 'Trade #', position: 'insideBottom', offset: -5, fill: '#737373', fontSize: 11 }} />
              <YAxis stroke="#404040" tick={{ fontSize: 11, fill: '#737373' }} tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="p95" stroke="none" fill="#06b6d4" fillOpacity={0.08} name="95th pctl" />
              <Area type="monotone" dataKey="p75" stroke="none" fill="#06b6d4" fillOpacity={0.12} name="75th pctl" />
              <Area type="monotone" dataKey="p50" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.18} strokeWidth={1.5} name="Median" />
              <Area type="monotone" dataKey="p25" stroke="none" fill="#0a0a0a" fillOpacity={0.5} name="25th pctl" />
              <Area type="monotone" dataKey="p5" stroke="none" fill="#0a0a0a" fillOpacity={0.7} name="5th pctl" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Walk-Forward Analysis */}
      <WalkForwardTable data={walk_forward} />

      {/* Generated Code */}
      <CodeViewer code={generated_code} />

      {/* Expectancy card */}
      <div className="rounded-xl border p-5 text-center" style={{ backgroundColor: '#111', borderColor: '#1e1e1e' }}>
        <p className="text-xs mb-1" style={{ color: '#737373' }}>Per-Trade Expectancy</p>
        <p className="text-3xl font-bold tabular-nums" style={{ color: metrics.expectancy > 0 ? '#22c55e' : '#ef4444' }}>
          {metrics.expectancy > 0 ? '+' : ''}{metrics.expectancy}%
        </p>
      </div>
    </div>
  )
}

export default ResultsDashboard
