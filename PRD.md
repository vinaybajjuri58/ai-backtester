# Product Requirements Document (PRD)
## AI Backtester — Agentic Backtesting Platform

**Version:** 1.0
**Last Updated:** 2026-02-23
**Author:** Professor (Product Owner) + Dot (Technical Author)
**Status:** Phase 1 MVP — In Development

---

## 1. Executive Summary

AI Backtester is an **agentic, AI-powered backtesting platform** that transforms plain English trading hypotheses into fully-executed, statistically-validated backtests. The platform eliminates the coding barrier in quantitative trading — users describe a strategy in natural language, and an AI agent translates, executes, analyzes, and reports results with institutional-grade metrics.

**One-liner:** *"Describe your strategy in English → Get a complete backtest with Monte Carlo, walk-forward, and strategy classification — zero code required."*

### Why "Agentic"?
Unlike traditional backtesting tools that require manual coding, AI Backtester uses an **AI agent pipeline**:
1. **Understands** your hypothesis (NLP → structured rules)
2. **Classifies** the strategy type (mean reversion vs trend following)
3. **Generates** executable backtest code
4. **Executes** the backtest on real market data
5. **Analyzes** results with statistical rigor
6. **Reports** findings with actionable insights

The AI agent makes decisions at each step — choosing appropriate indicators, setting reasonable defaults, handling edge cases — just like a junior quant analyst would.

---

## 2. Problem Statement

### The Gap
- **Retail traders** have hypotheses but can't code backtests
- **Algo traders** spend 80% of their time on boilerplate code, not alpha generation
- **Existing tools** (QuantConnect, Backtrader, Pine Script) require programming knowledge
- **No-code tools** (Composer) are too simplistic — no Monte Carlo, no walk-forward, no strategy classification

### The Pain
| User Segment | Pain Point |
|---|---|
| Retail Trader | "I think RSI < 30 is a buy signal, but I can't code a backtest to prove it" |
| Algo Developer | "I spend 2 hours writing boilerplate for every 10-minute hypothesis" |
| Fund Manager | "I need to evaluate 50 strategy ideas per week — coding each one is insane" |
| Educator | "Teaching students to backtest requires teaching Python first" |

### Our Solution
A platform where the input is **plain English** and the output is **institutional-grade backtest results** — complete with strategy classification, statistical validation, and production-ready code.

---

## 3. Target Users

### Primary: Independent Algo Traders
- Have trading experience and strategy intuition
- Know what Sharpe ratio and drawdown mean
- Don't want to (or can't) write Python for every idea
- Want rapid hypothesis validation (minutes, not hours)

### Secondary: Quant Developers
- Use it as a rapid prototyping tool
- Generate boilerplate code, then customize
- Save time on initial validation before deep-diving

### Tertiary: Trading Educators & Students
- Teach backtesting concepts without coding prerequisites
- Students can test ideas from textbooks instantly

### Future: Fund Managers / Research Teams
- Batch-test strategy ideas from research pipelines
- Screen hypotheses before allocating developer time

---

## 4. Product Vision & Phases

```
Phase 1 (MVP) ← WE ARE HERE
  Plain English → Backtest → Metrics + Charts + Classification

Phase 2: Iteration & Optimization
  Auto-iterate strategies, parameter optimization, multi-run comparison

Phase 3: Hypothesis Generation Engine
  AI scrapes internet for trading ideas → auto-backtest → rank

Phase 4: Live Trading Bridge
  Connect validated strategies to live execution (paper → real)

Phase 5: Strategy Marketplace
  Users share/sell strategies, leaderboard, subscription model
```

---

## 5. Phase 1 (MVP) — Detailed Requirements

### 5.1 Core User Flow

```
┌──────────────────────────────────────────────────┐
│  USER INPUT                                       │
│  ┌─────────────────────────────────────────────┐ │
│  │ "Buy BTC when RSI drops below 30,           │ │
│  │  sell when it crosses above 70"              │ │
│  └─────────────────────────────────────────────┘ │
│  Asset: BTC/USDT  Timeframe: 1d  Lookback: 1yr  │
│  [▶ Run Backtest]                                 │
└──────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│  AI AGENT PIPELINE (backend)                      │
│                                                   │
│  1. Parse hypothesis → structured rules           │
│  2. Classify: Mean Reversion / Trend Following    │
│  3. Fetch real OHLCV market data                  │
│  4. Calculate technical indicators                │
│  5. Generate & execute vectorized backtest        │
│  6. Compute performance metrics                   │
│  7. Run Monte Carlo simulation (1000 paths)       │
│  8. Run walk-forward analysis (70/30 split)       │
│  9. Generate standalone Python code               │
│  10. Return everything to frontend                │
└──────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│  RESULTS DASHBOARD                                │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │ 🏷️ TREND FOLLOWING | High Confidence     │    │
│  │ "MA crossovers are classic trend-following│    │
│  │  signals — they ride momentum..."         │    │
│  └──────────────────────────────────────────┘    │
│                                                   │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐               │
│  │+12% │ │ 58% │ │ 1.4 │ │-8.2%│               │
│  │Retrn│ │WinRt│ │Shrpe│ │MaxDD│               │
│  └─────┘ └─────┘ └─────┘ └─────┘               │
│                                                   │
│  📈 Equity Curve Chart                           │
│  📊 Monte Carlo Confidence Bands                 │
│  📋 Walk-Forward: In-Sample vs Out-of-Sample     │
│  💻 Generated Python Code (collapsible)          │
│  📌 Per-Trade Expectancy                         │
└──────────────────────────────────────────────────┘
```

### 5.2 Functional Requirements

#### FR-1: Strategy Input
| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | Text area accepting plain English trading hypothesis | P0 |
| FR-1.2 | Asset selector dropdown (BTC/USDT, ETH/USDT) | P0 |
| FR-1.3 | Timeframe selector (1m, 5m, 15m, 1h, 4h, 1d) | P0 |
| FR-1.4 | Lookback period selector (3mo, 6mo, 1yr, 2yr) | P0 |
| FR-1.5 | Pre-built example strategy chips (click to populate) | P1 |
| FR-1.6 | Input validation with clear error messages | P1 |
| FR-1.7 | Loading state with spinner during execution | P0 |

#### FR-2: AI Strategy Translation
| ID | Requirement | Priority |
|---|---|---|
| FR-2.1 | LLM-based parser (Claude API) for complex hypotheses | P0 |
| FR-2.2 | Rule-based fallback parser (no API key needed) | P0 |
| FR-2.3 | Support strategy types: RSI, MA Crossover, Bollinger, MACD, Breakout | P0 |
| FR-2.4 | Extract: entry condition, exit condition, indicators, parameters | P0 |
| FR-2.5 | Handle ambiguous input with sensible defaults | P1 |
| FR-2.6 | Return structured JSON with parsed rules | P0 |

#### FR-3: Strategy Classification
| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | Classify every strategy as **Mean Reversion** or **Trend Following** | P0 |
| FR-3.2 | Provide human-readable reasoning for classification | P0 |
| FR-3.3 | Confidence level (high/medium/low) | P1 |
| FR-3.4 | Context-aware classification (e.g., Bollinger can be either depending on usage) | P0 |
| FR-3.5 | Display classification prominently in results with color coding | P0 |

#### FR-4: Backtest Execution
| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | Fetch real OHLCV data from yfinance (BTC-USD, ETH-USD) | P0 |
| FR-4.2 | Calculate technical indicators using pandas-ta | P0 |
| FR-4.3 | Vectorized backtest execution (not bar-by-bar for speed) | P0 |
| FR-4.4 | Proper position tracking (long/flat/exit) | P0 |
| FR-4.5 | Handle intraday data limits (yfinance: 7d for 1m, 60d for 5m/15m) | P1 |
| FR-4.6 | Support 4h timeframe via 1h resampling | P1 |

#### FR-5: Metrics Calculation
| ID | Requirement | Priority |
|---|---|---|
| FR-5.1 | Win Rate (%) | P0 |
| FR-5.2 | Sharpe Ratio (annualized) | P0 |
| FR-5.3 | Profit Factor (gross profit / gross loss) | P0 |
| FR-5.4 | Maximum Drawdown (%) | P0 |
| FR-5.5 | Total Return (%) | P0 |
| FR-5.6 | Total Number of Trades | P0 |
| FR-5.7 | Average Win / Average Loss (%) | P0 |
| FR-5.8 | Per-Trade Expectancy (%) | P0 |
| FR-5.9 | Color-coded metrics (green = good, red = bad) | P1 |

#### FR-6: Monte Carlo Simulation
| ID | Requirement | Priority |
|---|---|---|
| FR-6.1 | 1,000 simulations by shuffling trade return sequence | P0 |
| FR-6.2 | Compute percentile bands: 5th, 25th, 50th (median), 75th, 95th | P0 |
| FR-6.3 | Area chart visualization with confidence bands | P0 |
| FR-6.4 | Handle edge cases (< 2 trades → flat line with explanation) | P1 |

#### FR-7: Walk-Forward Analysis
| ID | Requirement | Priority |
|---|---|---|
| FR-7.1 | 70/30 in-sample / out-of-sample split | P0 |
| FR-7.2 | Compute metrics for both periods independently | P0 |
| FR-7.3 | Robustness score (OOS performance / IS performance ratio) | P0 |
| FR-7.4 | Comparison table with color-coded robustness indicator | P0 |
| FR-7.5 | Minimum data requirement check (50 bars IS, 20 bars OOS) | P1 |

#### FR-8: Code Generation
| ID | Requirement | Priority |
|---|---|---|
| FR-8.1 | Generate standalone, copy-paste-ready Python backtest code | P0 |
| FR-8.2 | Include comments explaining each section | P1 |
| FR-8.3 | Collapsible code viewer in UI | P0 |
| FR-8.4 | Copy-to-clipboard button | P1 |
| FR-8.5 | Code uses same libraries as backend (pandas, yfinance, pandas-ta) | P0 |

#### FR-9: Results Dashboard
| ID | Requirement | Priority |
|---|---|---|
| FR-9.1 | Strategy classification badge (MR/TF) with reasoning | P0 |
| FR-9.2 | Metrics grid (2x4 cards) | P0 |
| FR-9.3 | Interactive equity curve chart (Recharts) | P0 |
| FR-9.4 | Monte Carlo area chart with confidence bands | P0 |
| FR-9.5 | Walk-forward comparison table | P0 |
| FR-9.6 | Generated code viewer (collapsible) | P0 |
| FR-9.7 | Per-trade expectancy highlight card | P1 |

### 5.3 Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Backtest execution time | < 15 seconds for daily data, < 30s for intraday |
| NFR-2 | Frontend load time | < 2 seconds |
| NFR-3 | Concurrent users (MVP) | 5-10 simultaneous |
| NFR-4 | Data freshness | Real-time market data (yfinance) |
| NFR-5 | Browser support | Chrome, Firefox, Safari (latest) |
| NFR-6 | Mobile responsive | Basic responsiveness (not primary target) |
| NFR-7 | Error handling | User-friendly error messages, no raw tracebacks |

### 5.4 Design Requirements

| Aspect | Specification |
|---|---|
| Theme | Dark mode only (primary: #0a0a0a) |
| Style | Minimalistic — clean lines, generous whitespace |
| Accent Color | Cyan (#06b6d4) for interactive elements |
| Typography | System font stack (Inter-like), tabular numbers for metrics |
| Cards | Subtle borders (#1e1e1e), no heavy shadows |
| Charts | Minimal grid lines, cyan accent, dark tooltips |
| Color Coding | Green (#22c55e) = positive, Red (#ef4444) = negative |
| Classification Badge | Purple/Violet for Mean Reversion, Cyan/Blue for Trend Following |

---

## 6. Technical Architecture

### 6.1 System Overview

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   React + Vite   │────▶│   Python FastAPI      │────▶│   Data Sources   │
│   (Frontend)     │◀────│   (Backend)            │◀────│                  │
│                  │     │                        │     │  • yfinance      │
│  • TailwindCSS   │     │  • agent.py (LLM/NLP) │     │  • Binance API   │
│  • Recharts      │     │  • backtest.py         │     │  • Claude API    │
│  • Axios         │     │  • executor.py         │     │                  │
│                  │     │  • models.py           │     │                  │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
```

### 6.2 Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Frontend | React 18 + Vite 6 | Fast dev, HMR, modern tooling |
| Styling | TailwindCSS v4 | Utility-first, dark mode native |
| Charts | Recharts | React-native charting, clean API |
| HTTP Client | Axios | Promise-based, interceptors |
| Backend | Python FastAPI | Async, auto-docs, Pydantic validation |
| AI/LLM | Claude API (claude-sonnet-4-20250514) | Best-in-class code understanding |
| Backtest Engine | Vectorized (pandas/numpy) | 10-100x faster than bar-by-bar |
| Indicators | pandas-ta | Comprehensive TA library |
| Data | yfinance | Free, reliable, good coverage |
| Statistics | numpy + scipy | Monte Carlo, statistical tests |
| Validation | Pydantic v2 | Type-safe request/response models |

### 6.3 API Endpoints

#### `POST /api/backtest`
**Request:**
```json
{
  "hypothesis": "Buy BTC when RSI drops below 30, sell when it crosses above 70",
  "asset": "BTC/USDT",
  "timeframe": "1d",
  "lookback": "1 year"
}
```

**Response:**
```json
{
  "strategy_rules": {
    "strategy_type": "rsi",
    "description": "RSI strategy: Buy when RSI < 30, Sell when RSI > 70",
    "entry_condition": "RSI(14) crosses below 30",
    "exit_condition": "RSI(14) crosses above 70",
    "indicators": [{"name": "rsi", "params": {"period": 14}}],
    "parameters": {"entry_threshold": 30, "exit_threshold": 70, "rsi_period": 14}
  },
  "classification": {
    "classification": "mean_reversion",
    "label": "Mean Reversion",
    "description": "RSI is inherently a mean-reversion indicator...",
    "reasoning": "Buying oversold and selling overbought is classic mean reversion...",
    "confidence": "high"
  },
  "metrics": {
    "win_rate": 62.5,
    "sharpe_ratio": 1.42,
    "profit_factor": 2.1,
    "max_drawdown": -12.3,
    "total_return": 34.5,
    "total_trades": 16,
    "avg_win": 4.2,
    "avg_loss": -2.1,
    "expectancy": 1.8
  },
  "equity_curve": [
    {"date": "2025-02-23", "equity": 10000.0, "price": 50000.0},
    ...
  ],
  "monte_carlo": {
    "percentile_5": [10000, ...],
    "percentile_25": [10000, ...],
    "percentile_50": [10000, ...],
    "percentile_75": [10000, ...],
    "percentile_95": [10000, ...],
    "x_axis": [0, 1, 2, ...]
  },
  "walk_forward": {
    "in_sample": {"total_return": 28.5, "sharpe_ratio": 1.6, ...},
    "out_of_sample": {"total_return": 18.2, "sharpe_ratio": 1.1, ...},
    "robustness_score": 63.9
  },
  "generated_code": "# AI Backtester - Generated Strategy Code\nimport pandas as pd..."
}
```

### 6.4 Strategy Parser Pipeline

```
Plain English Input
        │
        ▼
┌──────────────────┐
│ Try Claude API    │──── API Key present? ──── Yes ──▶ LLM Parse
│ (agent.py)       │                                        │
└──────────────────┘                                        │
        │ No                                                │
        ▼                                                   ▼
┌──────────────────┐                              ┌──────────────────┐
│ Rule-Based Parser │                              │ Structured JSON   │
│ (regex patterns)  │────────────────────────────▶│ {strategy_type,   │
│                   │                              │  entry, exit,     │
│ Supports:         │                              │  indicators,      │
│ • RSI             │                              │  parameters}      │
│ • MA Crossover    │                              └──────────────────┘
│ • Bollinger       │                                       │
│ • MACD            │                                       ▼
│ • Breakout        │                              ┌──────────────────┐
└──────────────────┘                              │ classify_strategy │
                                                   │ (MR vs TF)       │
                                                   └──────────────────┘
```

---

## 7. Strategy Classification System

### 7.1 Classification Logic

The platform classifies every strategy into one of two fundamental categories:

| Category | Definition | Examples |
|---|---|---|
| **Mean Reversion** | Bets that price will return to a mean/average after deviating | RSI oversold/overbought, Bollinger bounce, Z-score |
| **Trend Following** | Bets that current momentum will continue | MA crossover, MACD, breakout, Donchian channels |

### 7.2 Classification Rules

| Strategy Type | Default Classification | Context-Dependent? |
|---|---|---|
| RSI | Mean Reversion | No — RSI is inherently mean-reverting |
| MA Crossover | Trend Following | No — crossovers follow momentum |
| Bollinger Band | **Depends on mode** | Yes — bounce = MR, breakout = TF |
| MACD | Trend Following | No — momentum indicator |
| Price Breakout | Trend Following | No — Donchian/Turtle style |

### 7.3 Why It Matters

Knowing the classification helps traders:
- **Set expectations**: MR strategies have higher win rates but smaller gains; TF strategies have lower win rates but larger gains
- **Portfolio construction**: Combine MR + TF strategies for diversification (low correlation)
- **Market regime awareness**: MR works in ranging markets, TF works in trending markets
- **Risk management**: Different stop-loss approaches for each type

---

## 8. Phase 2 — Iteration & Optimization (Future)

### 8.1 Features
| Feature | Description |
|---|---|
| Auto-Iteration | Agent tweaks parameters and re-runs to find optimal configuration |
| Parameter Optimization | Grid search, random search, Bayesian optimization |
| Multi-Strategy Comparison | Run 5 strategies side-by-side, rank by metrics |
| Strategy Combination | Test portfolio of 2-3 strategies together (Carver-style) |
| Custom Exit Rules | Stop-loss, take-profit, trailing stops, time-based exits |
| Transaction Costs | Include slippage, fees, spread in backtest |
| Position Sizing | Kelly criterion, fixed fractional, volatility-targeted |

### 8.2 Agentic Iteration Loop
```
User Hypothesis
      │
      ▼
  Run Backtest (V1)
      │
      ▼
  Agent Analyzes Results
      │
      ├── Sharpe < 0.5? → "Try adjusting RSI period from 14 to 21"
      ├── Win rate < 40%? → "Consider adding trend filter (200 SMA)"
      ├── Max DD > 20%? → "Add stop-loss at 2% per trade"
      └── Profit factor < 1? → "Strategy is losing — try opposite signal"
      │
      ▼
  Auto-Generate V2 with Improvements
      │
      ▼
  Run Backtest (V2)
      │
      ▼
  Compare V1 vs V2 → Present Best
```

---

## 9. Phase 3 — Hypothesis Generation Engine (Future)

### 9.1 Concept
An AI "idea factory" that runs 24/7, scraping the internet for trading ideas, auto-backtesting them, and ranking by performance.

### 9.2 Sources
| Source | What We Extract |
|---|---|
| Reddit (r/algotrading, r/stocks) | Strategy discussions, indicator ideas |
| Twitter/X (quant traders) | Market observations, pattern claims |
| Academic Papers (arXiv, SSRN) | Research hypotheses, factor models |
| Trading Forums | Community strategies, indicator combinations |
| News Events | "Buy the rumor, sell the news" patterns |

### 9.3 Pipeline
```
Scrape Sources (daily)
      │
      ▼
Extract Trading Hypotheses (LLM)
      │
      ▼
Deduplicate & Prioritize
      │
      ▼
Auto-Backtest (batch, async)
      │
      ▼
Rank by: Sharpe, Robustness Score, # Trades
      │
      ▼
Present Top 10 to User (weekly digest)
```

---

## 10. Phase 4 — Live Trading Bridge (Future)

### 10.1 Flow
```
Validated Strategy (backtest passed thresholds)
      │
      ▼
Paper Trading (2-4 weeks, simulated execution)
      │
      ▼
Performance Matches Backtest? (within 1σ)
      │
  Yes ▼              No → Flag for review
Live Trading (real capital)
      │
      ▼
Continuous Monitoring
  • Real-time P&L tracking
  • Drawdown alerts
  • Strategy degradation detection
  • Auto-pause if metrics breach thresholds
```

### 10.2 Broker Integrations (Planned)
| Broker | Asset Class | Region |
|---|---|---|
| Binance | Crypto | Global |
| Alpaca | US Equities | US |
| IBKR | Multi-asset | Global |
| Zerodha (Kite) | Indian Equities/F&O | India |

---

## 11. Phase 5 — Strategy Marketplace (Future)

### 11.1 Concept
Users can share, sell, and subscribe to profitable strategies. A leaderboard ranks strategies by live performance.

### 11.2 Revenue Model
| Model | Description |
|---|---|
| Freemium | 5 free backtests/month, unlimited on Pro |
| Pro Subscription | $29/month — unlimited backtests, Monte Carlo, walk-forward, code export |
| API Access | $99/month — programmatic backtest execution |
| Marketplace Cut | 20% fee on strategy sales |
| Enterprise | Custom pricing for fund managers / teams |

---

## 12. Competitive Analysis

| Platform | Approach | Strengths | Our Edge |
|---|---|---|---|
| **QuantConnect** | Code-first (C#/Python) | Deep, institutional-grade | We're hypothesis-first — no coding |
| **Composer** | No-code visual builder | Simple, approachable | We have AI translation + statistical validation |
| **TradingView** | Pine Script indicators | Massive community, charts | We do NLP → full statistical backtest |
| **Backtrader** | Python library | Flexible, open-source | We abstract away all coding |
| **Alpaca Backtest** | API-based | Free, broker-integrated | We add Monte Carlo + walk-forward + classification |
| **StrategyQuant** | Genetic algorithm | Auto-generates strategies | We start from human intuition, not brute force |

### Key Differentiator
**No other platform does: Plain English → Strategy Classification → Full Statistical Backtest → Generated Code.**

We sit at the intersection of:
- 🧠 AI (LLM understanding of trading strategies)
- 📊 Quant (institutional metrics: Monte Carlo, walk-forward, Sharpe)
- 🎨 UX (minimalistic dark UI, zero-code experience)

---

## 13. Success Metrics

### Phase 1 (MVP)
| Metric | Target |
|---|---|
| Backtest execution success rate | > 95% |
| Strategy parse accuracy (5 supported types) | > 90% |
| Average backtest time | < 15 seconds |
| User completes first backtest | < 2 minutes from landing |

### Phase 2+
| Metric | Target |
|---|---|
| Monthly Active Users | 1,000 (6 months post-launch) |
| Backtests per user per month | > 10 |
| Pro conversion rate | > 5% |
| User retention (30-day) | > 40% |

---

## 14. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM misparses strategy | Wrong backtest results | Rule-based fallback + validation checks |
| yfinance rate limits | Data fetch failures | Caching layer, fallback to Binance API |
| Overfitting appearance | User trusts bad strategy | Walk-forward + Monte Carlo + robustness score |
| User inputs garbage | Wasted compute | Input validation + sensible defaults + "couldn't parse" error |
| Claude API downtime | No LLM parsing | Rule-based fallback works without any API |
| Security (code execution) | Server compromise | Sandboxed execution (Docker in Phase 2), vectorized backtest (no eval) |

---

## 15. Open Questions

| # | Question | Status |
|---|---|---|
| 1 | Which LLM to use long-term? (Claude vs GPT-4 vs fine-tuned open-source) | Using Claude, evaluating |
| 2 | Sandbox execution model for Phase 2? (Docker per session vs shared pool) | Leaning Docker per session |
| 3 | Pricing tiers for freemium model | TBD |
| 4 | Mobile-first or desktop-first? | Desktop-first (traders use monitors) |
| 5 | Support short-selling in backtests? | Phase 2 |
| 6 | Multi-asset portfolio backtesting? | Phase 3 |
| 7 | Data vendor for equities beyond yfinance? (Polygon, Alpha Vantage) | TBD |

---

## 16. Appendix

### A. Supported Strategy Patterns (MVP)

| Pattern | Example Input | Parsed As |
|---|---|---|
| RSI | "Buy when RSI goes below 30, sell above 70" | RSI mean reversion |
| MA Crossover | "Buy when 10 EMA crosses above 50 EMA" | MA crossover (trend) |
| Bollinger Bounce | "Buy at lower Bollinger Band, sell at middle" | Bollinger mean reversion |
| Bollinger Breakout | "Buy when price breaks above upper BB" | Bollinger trend |
| MACD | "Buy when MACD crosses signal line" | MACD trend |
| Breakout | "Buy on 20-day high breakout" | Donchian breakout (trend) |

### B. Metrics Definitions

| Metric | Formula | Good Value |
|---|---|---|
| Win Rate | winning_trades / total_trades × 100 | > 50% (MR), > 40% (TF) |
| Sharpe Ratio | mean(returns) / std(returns) × √(bars_per_year) | > 1.0 |
| Profit Factor | gross_profit / gross_loss | > 1.5 |
| Max Drawdown | max(peak - trough) / peak × 100 | < 20% |
| Expectancy | mean(all_trade_returns) × 100 | > 0% |

### C. File Structure

```
ai-backtester/
├── PRD.md                  ← This document
├── README.md               ← Setup instructions
├── backend/
│   ├── main.py             ← FastAPI app, CORS, endpoint
│   ├── agent.py            ← LLM + rule-based parser + classification
│   ├── backtest.py         ← Engine: data fetch, indicators, signals, metrics, MC, WF
│   ├── executor.py         ← Standalone code generation
│   ├── models.py           ← Pydantic schemas
│   ├── requirements.txt    ← Python dependencies
│   └── venv/               ← Virtual environment
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js      ← Vite + Tailwind + API proxy
│   └── src/
│       ├── main.jsx
│       ├── index.css
│       ├── App.jsx
│       └── components/
│           ├── StrategyInput.jsx
│           └── ResultsDashboard.jsx
```

---

*This is a living document. Updated as the product evolves.*
*Last reviewed: 2026-02-23*
