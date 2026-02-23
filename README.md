# AI Backtester

An AI-powered backtesting platform that translates plain English trading strategies into fully functional backtests with real market data.

## Architecture

- **Frontend**: React + Vite + TailwindCSS v4 (dark theme)
- **Backend**: Python FastAPI

## Features

- 🤖 Plain English strategy input → structured trading rules (LLM or rule-based)
- 📊 Real market data from Yahoo Finance (BTC, ETH)
- 📈 Equity curve visualization
- 🎲 Monte Carlo simulation (1,000 iterations)
- 🔄 Walk-forward analysis (70/30 in-sample/out-of-sample split)
- 📋 Auto-generated standalone Python backtest code
- 📐 Full metrics: Win Rate, Sharpe, Profit Factor, Max Drawdown, etc.

## Supported Strategies

The rule-based parser handles:
- **RSI**: "Buy when RSI drops below 30, sell when it crosses above 70"
- **Moving Average Crossover**: "Buy when 10-day EMA crosses above 50-day EMA"
- **Bollinger Bands**: "Buy when price touches lower Bollinger Band"
- **MACD**: "Buy when MACD crosses above signal line"
- **Breakout**: "Buy on 20-day high breakout"

Set `ANTHROPIC_API_KEY` env var to enable LLM-powered parsing for any strategy description.

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## API

### POST /api/backtest

Request:
```json
{
  "hypothesis": "Buy BTC when RSI drops below 30, sell when it crosses above 70",
  "asset": "BTC/USDT",
  "timeframe": "1d",
  "lookback": "1 year"
}
```

Response includes: metrics, equity_curve, monte_carlo, walk_forward, generated_code, strategy_rules

## Environment Variables

- `ANTHROPIC_API_KEY` (optional) — enables Claude-powered strategy parsing
