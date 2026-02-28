# AI Backtester Backend

A FastAPI-based backend for translating plain English trading hypotheses into structured trading rules and running comprehensive backtests against historical market data.

## Overview

The backend powers an AI-driven trading strategy backtesting system that:
- Converts natural language trading ideas into executable strategies
- Fetches historical market data from Yahoo Finance
- Runs vectorized backtests with performance metrics
- Provides statistical validation through Monte Carlo and Walk-Forward Analysis
- Generates standalone Python code for strategies

## Architecture

```
backend/
├── main.py        # FastAPI application & API routes
├── models.py      # Pydantic schemas for type safety
├── agent.py       # LLM strategy translation & classification
├── backtest.py    # Backtest engine (data, indicators, metrics)
├── executor.py    # Standalone Python code generation
└── requirements.txt
```

## Core Capabilities

### 1. Natural Language Strategy Translation (`agent.py`)

Converts plain English trading hypotheses into structured rules using:
- **Primary**: Claude API (Anthropic) for intelligent parsing
- **Fallback**: Rule-based regex parser for offline operation

**Supported Strategy Types:**
| Strategy | Type | Description |
|----------|------|-------------|
| RSI | Mean Reversion | Buy oversold (<30), sell overbought (>70) |
| MA Crossover | Trend Following | Fast MA crosses above/below slow MA |
| Bollinger Bands | Mean Reversion / Trend Following | Bounce from bands or breakout |
| MACD | Trend Following | MACD line crosses signal line |
| Price Breakout | Trend Following | Breaks N-period highs/lows |

**Strategy Classification:**
- Automatically classifies strategies as "Mean Reversion" or "Trend Following"
- Provides confidence scores and detailed reasoning
- Explains why indicators fit each classification

### 2. Market Data & Backtesting (`backtest.py`)

**Data Sources:**
- Yahoo Finance via `yfinance` library
- Assets: BTC/USDT, ETH/USDT (mapped to BTC-USD, ETH-USD)
- Timeframes: 1m, 5m, 15m, 1h, 4h, 1d
- Lookback: 1 month to 2 years

**Technical Indicators:**
- RSI (Relative Strength Index)
- SMA/EMA (Simple/Exponential Moving Averages)
- Bollinger Bands
- MACD
- Rolling High/Low (breakout detection)

**Performance Metrics:**
| Metric | Description |
|--------|-------------|
| Win Rate | % of profitable trades |
| Sharpe Ratio | Risk-adjusted returns |
| Profit Factor | Gross profit / Gross loss |
| Max Drawdown | Largest peak-to-trough decline |
| Total Return | Overall strategy return |
| Avg Win/Loss | Average winning/losing trade |
| Expectancy | Expected return per trade |

**Statistical Validation:**
- **Monte Carlo Simulation**: 1000 simulations shuffling trade sequences to test robustness
- **Walk-Forward Analysis**: 70/30 in-sample/out-of-sample split with robustness scoring

### 3. Code Generation (`executor.py`)

Generates clean, standalone Python scripts from strategy rules:
- Complete, runnable Python code
- Includes data fetching, indicator calculation, backtest logic
- Outputs performance metrics
- Optional matplotlib visualization (equity curve + drawdown)
- Ready for local execution or further customization

### 4. API Layer (`main.py`)

**Endpoints:**
```
GET  /                Health check
POST /api/backtest    Run full backtest pipeline
```

**CORS Enabled:** Configured for localhost:5173 (Vite dev server)

## Data Models (`models.py`)

**Request:**
```python
BacktestRequest {
    hypothesis: str   # Natural language strategy description
    asset: str        # Trading pair (e.g., "BTC/USDT")
    timeframe: str    # 1m, 5m, 15m, 1h, 4h, 1d
    lookback: str     # 1 month, 3 months, 6 months, 1 year, 2 years
}
```

**Response:**
```python
BacktestResponse {
    metrics: dict           # Performance metrics
    equity_curve: list      # Portfolio value over time
    monte_carlo: dict       # MC simulation percentiles
    walk_forward: dict      # IS/OOS analysis results
    generated_code: str     # Standalone Python code
    strategy_rules: dict    # Structured strategy definition
    classification: dict    # Strategy type & reasoning
}
```

## Getting Started

### Installation

```bash
cd backend
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in either the **project root** or the **backend folder**:

```bash
# Option A: Project root (recommended) - ai-backtester/.env
OPENAI_API_KEY=sk-proj-...

# Option B: Backend folder - ai-backtester/backend/.env
OPENAI_API_KEY=sk-proj-...
```

Both locations work. The backend checks for `.env` in this order:
1. `backend/.env`
2. `ai-backtester/.env` (project root)

**Example `.env` content:**

```bash
# Option 1: Anthropic Claude (tried first)
ANTHROPIC_API_KEY=sk-ant-api03-...

# Option 2: OpenAI (tried second) - RECOMMENDED
OPENAI_API_KEY=sk-proj-...
# OPENAI_MODEL=gpt-4o-mini  # optional

# Option 3: Kimi K2.5 via Moonshot (tried third)
MOONSHOT_API_KEY=sk-...

# Option 4: Kimi via OpenRouter or other proxy
KIMI_API_KEY=sk-or-v1-...
KIMI_BASE_URL=https://openrouter.ai/api/v1
KIMI_MODEL=moonshotai/kimi-k2.5
```

**LLM Priority:**
1. Claude (Anthropic) - tried first if `ANTHROPIC_API_KEY` is set
2. OpenAI - tried second if `OPENAI_API_KEY` is set
3. Kimi K2.5 - tried third if any Kimi key is set (`KIMI_API_KEY`, `MOONSHOT_API_KEY`, or `OPENROUTER_API_KEY`)
4. Rule-based parser - always available as fallback

**Kimi Configuration Options:**

| Variable | Default | Description |
|----------|---------|-------------|
| `KIMI_API_KEY` / `MOONSHOT_API_KEY` / `OPENROUTER_API_KEY` | - | API key (tried in that order) |
| `KIMI_BASE_URL` | `https://api.moonshot.cn/v1` | API base URL |
| `KIMI_MODEL` | `kimi-k2.5` | Model name |

**Common Setups:**

```bash
# OpenAI (Default - recommended)
OPENAI_API_KEY=sk-proj-...

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-api03-...

# Direct Moonshot (China)
MOONSHOT_API_KEY=your-moonshot-key

# OpenRouter (Global)
KIMI_API_KEY=sk-or-v1-...
KIMI_BASE_URL=https://openrouter.ai/api/v1
KIMI_MODEL=moonshotai/kimi-k2.5

# Custom proxy
KIMI_API_KEY=your-key
KIMI_BASE_URL=https://your-proxy.com/v1
```

Without any API keys, the system uses the rule-based regex parser.

### Running the Server

```bash
python main.py
# OR
uvicorn main:app --reload --port 8000
```

Server runs at `http://localhost:8000`

### API Usage Example

```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "hypothesis": "Buy when RSI drops below 30, sell when it crosses above 70",
    "asset": "BTC/USDT",
    "timeframe": "1d",
    "lookback": "1 year"
  }'
```

## Dependencies

- **fastapi** - Modern web framework
- **uvicorn** - ASGI server
- **httpx** - Async HTTP client (LLM API calls)
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **yfinance** - Yahoo Finance data
- **pandas-ta** - Technical indicators
- **pydantic** - Data validation

## Design Notes

- **Vectorized Backtesting**: Uses pandas operations for speed
- **Async/Await**: Non-blocking LLM API calls
- **Multi-LLM Support**: Claude and Kimi K2.5 with automatic fallback
- **Resilient**: Fallback parsers ensure operation without external APIs
- **Modular**: Clear separation between parsing, backtesting, and code generation
