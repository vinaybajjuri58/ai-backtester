"""Pydantic schemas for the AI Backtester API."""

from pydantic import BaseModel
from typing import Optional


class BacktestRequest(BaseModel):
    hypothesis: str
    asset: str
    timeframe: str
    lookback: str


class StrategyRules(BaseModel):
    entry_condition: str
    exit_condition: str
    indicators: list[dict]
    parameters: dict
    strategy_type: str
    description: str


class StrategyClassification(BaseModel):
    classification: str          # "mean_reversion" | "trend_following" | "unknown"
    label: str                   # "Mean Reversion" | "Trend Following"
    description: str             # Why this strategy is classified this way
    reasoning: str               # Detailed reasoning for the classification
    confidence: str              # "high" | "medium" | "low"


class Metrics(BaseModel):
    win_rate: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    total_return: float
    total_trades: int
    avg_win: float
    avg_loss: float
    expectancy: float


class MonteCarloResult(BaseModel):
    percentile_5: list[float]
    percentile_25: list[float]
    percentile_50: list[float]
    percentile_75: list[float]
    percentile_95: list[float]
    x_axis: list[int]


class WalkForwardResult(BaseModel):
    in_sample: dict
    out_of_sample: dict
    robustness_score: float


class BacktestResponse(BaseModel):
    metrics: dict
    equity_curve: list[dict]
    monte_carlo: dict
    walk_forward: dict
    generated_code: str
    strategy_rules: dict
    classification: dict         # Strategy classification info
    last_3_trades: list[dict]    # Last 3 trades for verification
