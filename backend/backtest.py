"""Backtest Engine - Fetches data, runs backtests, computes metrics."""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from typing import Tuple
from datetime import datetime, timedelta


ASSET_MAP = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
}

TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "1h",  # yfinance doesn't support 4h, we'll resample
    "1d": "1d",
}

LOOKBACK_MAP = {
    "1 month": 30,
    "3 months": 90,
    "6 months": 180,
    "1 year": 365,
    "2 years": 730,
}


def fetch_data(asset: str, timeframe: str, lookback: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    ticker = ASSET_MAP.get(asset, "BTC-USD")
    days = LOOKBACK_MAP.get(lookback, 365)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    yf_interval = TIMEFRAME_MAP.get(timeframe, "1d")

    # yfinance has limits on intraday data, adjust accordingly
    if yf_interval in ("1m", "5m", "15m"):
        # Max 60 days for intraday < 1h, 730 for 1h
        if yf_interval == "1m":
            max_days = 7
        elif yf_interval == "5m":
            max_days = 60
        else:
            max_days = 60
        days = min(days, max_days)
        start_date = end_date - timedelta(days=days)

    if yf_interval == "1h":
        max_days = 730
        days = min(days, max_days)
        start_date = end_date - timedelta(days=days)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=yf_interval,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Resample for 4h if needed
    if timeframe == "4h" and yf_interval == "1h":
        df = df.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    return df


def calculate_indicators(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Calculate technical indicators based on strategy rules."""
    strategy_type = rules.get("strategy_type", "rsi")
    params = rules.get("parameters", {})
    indicators = rules.get("indicators", [])

    if strategy_type == "rsi":
        period = params.get("rsi_period", 14)
        df["rsi"] = ta.rsi(df["close"], length=period)

    elif strategy_type == "confluence_rsi_ema":
        # RSI for mean-reversion entry
        rsi_period = params.get("rsi_period", 14)
        df["rsi"] = ta.rsi(df["close"], length=rsi_period)
        # EMAs for trend confirmation
        fast_period = params.get("fast_period", 10)
        slow_period = params.get("slow_period", 50)
        df["fast_ema"] = ta.ema(df["close"], length=fast_period)
        df["slow_ema"] = ta.ema(df["close"], length=slow_period)

    elif strategy_type == "ma_crossover":
        fast = params.get("fast_period", 10)
        slow = params.get("slow_period", 50)
        ma_type = params.get("ma_type", "sma")
        if ma_type == "ema":
            df["fast_ma"] = ta.ema(df["close"], length=fast)
            df["slow_ma"] = ta.ema(df["close"], length=slow)
        else:
            df["fast_ma"] = ta.sma(df["close"], length=fast)
            df["slow_ma"] = ta.sma(df["close"], length=slow)

    elif strategy_type == "bollinger":
        period = params.get("bb_period", 20)
        std = params.get("bb_std", 2.0)
        bb = ta.bbands(df["close"], length=period, std=std)
        if bb is not None:
            df["bb_upper"] = bb.iloc[:, 2]
            df["bb_middle"] = bb.iloc[:, 1]
            df["bb_lower"] = bb.iloc[:, 0]

    elif strategy_type == "macd":
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        signal = params.get("signal_period", 9)
        macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        if macd_df is not None:
            df["macd"] = macd_df.iloc[:, 0]
            df["macd_hist"] = macd_df.iloc[:, 1]
            df["macd_signal"] = macd_df.iloc[:, 2]

    elif strategy_type == "breakout":
        period = params.get("breakout_period", 20)
        df["high_breakout"] = df["high"].rolling(window=period).max()
        df["low_breakout"] = df["low"].rolling(window=period).min()

    return df


def generate_signals(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Generate buy/sell signals based on strategy rules."""
    strategy_type = rules.get("strategy_type", "rsi")
    params = rules.get("parameters", {})

    df["signal"] = 0

    if strategy_type == "rsi":
        entry = params.get("entry_threshold", 30)
        exit_val = params.get("exit_threshold", 70)
        df.loc[df["rsi"] < entry, "signal"] = 1
        df.loc[df["rsi"] > exit_val, "signal"] = -1

    elif strategy_type == "confluence_rsi_ema":
        # Confluence: RSI oversold + EMA uptrend
        rsi_entry = params.get("rsi_entry_threshold", 30)
        df["confluence_long"] = (df["rsi"] < rsi_entry) & (df["fast_ema"] > df["slow_ema"])
        # Signal = 1 when confluence condition is met
        df.loc[df["confluence_long"], "signal"] = 1
        # Exit signal (optional - not used with SL/TP)
        df.loc[~df["confluence_long"], "signal"] = -1

    elif strategy_type == "ma_crossover":
        df.loc[df["fast_ma"] > df["slow_ma"], "signal"] = 1
        df.loc[df["fast_ma"] < df["slow_ma"], "signal"] = -1

    elif strategy_type == "bollinger":
        mode = params.get("mode", "mean_reversion")
        if mode == "mean_reversion":
            df.loc[df["close"] < df["bb_lower"], "signal"] = 1
            df.loc[df["close"] > df["bb_middle"], "signal"] = -1
        else:
            df.loc[df["close"] > df["bb_upper"], "signal"] = 1
            df.loc[df["close"] < df["bb_middle"], "signal"] = -1

    elif strategy_type == "macd":
        df.loc[df["macd"] > df["macd_signal"], "signal"] = 1
        df.loc[df["macd"] < df["macd_signal"], "signal"] = -1

    elif strategy_type == "breakout":
        df.loc[df["close"] > df["high_breakout"].shift(1), "signal"] = 1
        df.loc[df["close"] < df["low_breakout"].shift(1), "signal"] = -1

    return df


def run_backtest(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Run vectorized backtest and return equity curve + trades."""
    df = df.copy()
    df["position"] = 0
    df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)

    # Calculate returns
    df["market_return"] = df["close"].pct_change()
    df["strategy_return"] = df["position"].shift(1) * df["market_return"]
    df["strategy_return"] = df["strategy_return"].fillna(0)

    # Equity curve (starting with $10,000)
    initial_capital = 10000
    df["equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()

    # Extract individual trades with timestamps
    trades = []
    in_trade = False
    entry_price = 0
    entry_idx = None
    entry_time = None

    for i in range(1, len(df)):
        prev_pos = df["position"].iloc[i - 1]
        curr_pos = df["position"].iloc[i]
        current_time = df.index[i]

        if not in_trade and curr_pos == 1:
            in_trade = True
            entry_price = df["close"].iloc[i]
            entry_idx = i
            entry_time = current_time
        elif in_trade and curr_pos != 1:
            exit_price = df["close"].iloc[i]
            pnl_pct = (exit_price - entry_price) / entry_price
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": i,
                "entry_time": entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(entry_time, 'strftime') else str(entry_time),
                "exit_time": current_time.strftime("%Y-%m-%d %H:%M") if hasattr(current_time, 'strftime') else str(current_time),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl_pct": float(pnl_pct),
                "exit_reason": "SIGNAL",
            })
            in_trade = False

    # Close any open position at the end
    if in_trade and entry_idx is not None:
        final_price = float(df["close"].iloc[-1])
        final_time = df.index[-1]
        pnl_pct = (final_price - entry_price) / entry_price
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": len(df) - 1,
            "entry_time": entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(entry_time, 'strftime') else str(entry_time),
            "exit_time": final_time.strftime("%Y-%m-%d %H:%M") if hasattr(final_time, 'strftime') else str(final_time),
            "entry_price": float(entry_price),
            "exit_price": float(final_price),
            "pnl_pct": float(pnl_pct),
            "exit_reason": "END",
        })

    return df, trades


def run_backtest_with_sl_tp(df: pd.DataFrame, sl_pct: float = 0.01, tp_pct: float = 0.02) -> Tuple[pd.DataFrame, list]:
    """Run bar-by-bar backtest with Stop Loss and Take Profit exits.
    
    Final SL/TP rules (Long-only):
    - Entry: When confluence condition becomes true, enter at that bar's close
      entry_price = close[i], entry_time = index[i]
    - SL/TP levels: sl_price = entry_price * (1 - sl_pct), tp_price = entry_price * (1 + tp_pct)
    - Exit check: Start from i+1 (next bar) to avoid "enter and exit on same candle" artifacts
    - SL hit if low[j] <= sl_price
    - TP hit if high[j] >= tp_price
    - If both hit in same bar: SL takes precedence (conservative)
    
    Args:
        df: DataFrame with 'signal', 'open', 'high', 'low', 'close' columns
        sl_pct: Stop loss percentage (e.g., 0.01 = 1%)
        tp_pct: Take profit percentage (e.g., 0.02 = 2%)
    
    Returns:
        Tuple of (DataFrame with equity curve, list of trades)
    """
    df = df.copy()
    initial_capital = 10000
    
    trades = []
    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    entry_idx = None
    entry_time = None
    sl_price = 0.0
    tp_price = 0.0
    
    equity = [initial_capital]
    position_series = [0]
    
    for i in range(1, len(df)):
        current_bar = df.iloc[i]
        current_time = df.index[i]
        prev_equity = equity[-1]
        
        if position == 0:
            # Not in position - check for entry signal
            if current_bar["signal"] == 1:
                # Enter long at close of signal bar
                position = 1
                entry_price = float(current_bar["close"])
                entry_idx = i
                entry_time = current_time
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
                # Don't check SL/TP on entry bar - wait for next bar
        else:
            # In position - check for SL/TP exit (evaluating current bar)
            bar_high = float(current_bar["high"])
            bar_low = float(current_bar["low"])
            bar_close = float(current_bar["close"])
            
            sl_hit = bar_low <= sl_price
            tp_hit = bar_high >= tp_price
            
            exit_price = None
            exit_reason = None
            
            if sl_hit and tp_hit:
                # Both hit in same bar - SL takes precedence (conservative)
                exit_price = sl_price
                exit_reason = "SL_same_bar"
            elif sl_hit:
                exit_price = sl_price
                exit_reason = "SL"
            elif tp_hit:
                exit_price = tp_price
                exit_reason = "TP"
            
            if exit_price is not None:
                # Close the trade
                pnl_pct = (exit_price - entry_price) / entry_price
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_time": entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(entry_time, 'strftime') else str(entry_time),
                    "exit_time": current_time.strftime("%Y-%m-%d %H:%M") if hasattr(current_time, 'strftime') else str(current_time),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "pnl_pct": float(pnl_pct),
                    "exit_reason": exit_reason,
                })
                
                position = 0
                entry_price = 0.0
                entry_idx = None
                entry_time = None
        
        # Record position and equity (mark-to-market)
        position_series.append(position)
        if position == 0:
            equity.append(equity[-1] if len(equity) > 0 else initial_capital)
        else:
            # Calculate current equity based on close price
            bar_close = float(current_bar["close"])
            unrealized_pnl = (bar_close - entry_price) / entry_price
            new_equity = initial_capital * (1 + unrealized_pnl)
            equity.append(new_equity)
    
    # Close any open position at the end
    if position == 1 and entry_idx is not None:
        final_price = float(df["close"].iloc[-1])
        final_time = df.index[-1]
        pnl_pct = (final_price - entry_price) / entry_price
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": len(df) - 1,
            "entry_time": entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(entry_time, 'strftime') else str(entry_time),
            "exit_time": final_time.strftime("%Y-%m-%d %H:%M") if hasattr(final_time, 'strftime') else str(final_time),
            "entry_price": float(entry_price),
            "exit_price": float(final_price),
            "pnl_pct": float(pnl_pct),
            "exit_reason": "END",
        })
    
    df["position"] = position_series[:len(df)]
    df["equity"] = equity[:len(df)]
    df["strategy_return"] = df["equity"].pct_change().fillna(0)
    
    return df, trades


def calculate_metrics(df: pd.DataFrame, trades: list) -> dict:
    """Calculate all performance metrics."""
    if not trades:
        return {
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "total_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win = np.mean(wins) * 100 if wins else 0
    avg_loss = np.mean(losses) * 100 if losses else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Sharpe ratio (annualized)
    returns = df["strategy_return"].dropna()
    if len(returns) > 1 and returns.std() > 0:
        # Estimate annualization factor from data frequency
        if len(df) > 1:
            time_diff = (df.index[-1] - df.index[0]).total_seconds()
            bar_seconds = time_diff / len(df)
            bars_per_year = 365.25 * 24 * 3600 / bar_seconds if bar_seconds > 0 else 252
        else:
            bars_per_year = 252
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0

    # Max drawdown
    equity = df["equity"]
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() * 100

    # Total return
    total_return = ((df["equity"].iloc[-1] / df["equity"].iloc[0]) - 1) * 100

    # Expectancy
    expectancy = np.mean(pnls) * 100 if pnls else 0

    return {
        "win_rate": round(float(win_rate), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "profit_factor": round(float(profit_factor), 2),
        "max_drawdown": round(float(max_dd), 2),
        "total_return": round(float(total_return), 2),
        "total_trades": len(trades),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "expectancy": round(float(expectancy), 2),
    }


def get_equity_curve(df: pd.DataFrame) -> list:
    """Generate equity curve data points for charting."""
    # Downsample to max 500 points for frontend
    step = max(1, len(df) // 500)
    sampled = df.iloc[::step]

    points = []
    for idx, row in sampled.iterrows():
        ts = idx.strftime("%Y-%m-%d %H:%M") if hasattr(idx, 'strftime') else str(idx)
        points.append({
            "date": ts,
            "equity": round(float(row["equity"]), 2),
            "price": round(float(row["close"]), 2),
        })
    return points


def run_monte_carlo(trades: list, n_simulations: int = 1000, initial_capital: float = 10000) -> dict:
    """Run Monte Carlo simulation by shuffling trade returns."""
    if not trades or len(trades) < 2:
        x_axis = list(range(10))
        flat = [initial_capital] * 10
        return {
            "percentile_5": flat,
            "percentile_25": flat,
            "percentile_50": flat,
            "percentile_75": flat,
            "percentile_95": flat,
            "x_axis": x_axis,
        }

    pnls = np.array([t["pnl_pct"] for t in trades])
    n_trades = len(pnls)

    # Simulate equity paths
    all_paths = np.zeros((n_simulations, n_trades + 1))
    all_paths[:, 0] = initial_capital

    for i in range(n_simulations):
        shuffled = np.random.permutation(pnls)
        for j in range(n_trades):
            all_paths[i, j + 1] = all_paths[i, j] * (1 + shuffled[j])

    # Compute percentiles at each step
    x_axis = list(range(n_trades + 1))
    p5 = np.percentile(all_paths, 5, axis=0).tolist()
    p25 = np.percentile(all_paths, 25, axis=0).tolist()
    p50 = np.percentile(all_paths, 50, axis=0).tolist()
    p75 = np.percentile(all_paths, 75, axis=0).tolist()
    p95 = np.percentile(all_paths, 95, axis=0).tolist()

    # Downsample if too many points
    max_points = 200
    if len(x_axis) > max_points:
        step = len(x_axis) // max_points
        x_axis = x_axis[::step]
        p5 = p5[::step]
        p25 = p25[::step]
        p50 = p50[::step]
        p75 = p75[::step]
        p95 = p95[::step]

    return {
        "percentile_5": [round(v, 2) for v in p5],
        "percentile_25": [round(v, 2) for v in p25],
        "percentile_50": [round(v, 2) for v in p50],
        "percentile_75": [round(v, 2) for v in p75],
        "percentile_95": [round(v, 2) for v in p95],
        "x_axis": x_axis,
    }


def run_walk_forward(df_original: pd.DataFrame, rules: dict, split_ratio: float = 0.7) -> dict:
    """Run walk-forward analysis with in-sample/out-of-sample split."""
    n = len(df_original)
    split_idx = int(n * split_ratio)

    if split_idx < 50 or (n - split_idx) < 20:
        return {
            "in_sample": {"total_return": 0, "sharpe_ratio": 0, "win_rate": 0, "max_drawdown": 0, "total_trades": 0},
            "out_of_sample": {"total_return": 0, "sharpe_ratio": 0, "win_rate": 0, "max_drawdown": 0, "total_trades": 0},
            "robustness_score": 0,
        }
    
    # Check if SL/TP mode
    params = rules.get("parameters", {})
    sl_pct = params.get("sl_pct")
    tp_pct = params.get("tp_pct")
    use_sl_tp = sl_pct is not None and tp_pct is not None
    
    # In-sample
    df_is = df_original.iloc[:split_idx].copy()
    df_is = generate_signals(df_is, rules)
    if use_sl_tp:
        df_is, trades_is = run_backtest_with_sl_tp(df_is, sl_pct=sl_pct, tp_pct=tp_pct)
    else:
        df_is, trades_is = run_backtest(df_is)
    metrics_is = calculate_metrics(df_is, trades_is)

    # Out-of-sample
    df_oos = df_original.iloc[split_idx:].copy()
    df_oos = generate_signals(df_oos, rules)
    if use_sl_tp:
        df_oos, trades_oos = run_backtest_with_sl_tp(df_oos, sl_pct=sl_pct, tp_pct=tp_pct)
    else:
        df_oos, trades_oos = run_backtest(df_oos)
    metrics_oos = calculate_metrics(df_oos, trades_oos)

    # Robustness score (ratio of OOS to IS performance)
    if metrics_is["total_return"] != 0:
        robustness = min(metrics_oos["total_return"] / metrics_is["total_return"], 2.0) if metrics_is["total_return"] > 0 else 0
    else:
        robustness = 0

    return {
        "in_sample": {
            "total_return": metrics_is["total_return"],
            "sharpe_ratio": metrics_is["sharpe_ratio"],
            "win_rate": metrics_is["win_rate"],
            "max_drawdown": metrics_is["max_drawdown"],
            "total_trades": metrics_is["total_trades"],
        },
        "out_of_sample": {
            "total_return": metrics_oos["total_return"],
            "sharpe_ratio": metrics_oos["sharpe_ratio"],
            "win_rate": metrics_oos["win_rate"],
            "max_drawdown": metrics_oos["max_drawdown"],
            "total_trades": metrics_oos["total_trades"],
        },
        "robustness_score": round(float(robustness) * 100, 1),
    }


async def execute_backtest(hypothesis: str, asset: str, timeframe: str, lookback: str, rules: dict) -> dict:
    """Main backtest execution pipeline."""
    # 1. Fetch data
    df = fetch_data(asset, timeframe, lookback)

    # 2. Calculate indicators
    df = calculate_indicators(df, rules)
    df = df.dropna()

    if len(df) < 30:
        raise ValueError("Not enough data points after indicator calculation. Try a longer lookback period.")

    # 3. Generate signals
    df = generate_signals(df, rules)

    # 4. Run backtest (use SL/TP if sl_pct and tp_pct are specified)
    strategy_type = rules.get("strategy_type", "rsi")
    params = rules.get("parameters", {})
    
    sl_pct = params.get("sl_pct")
    tp_pct = params.get("tp_pct")
    
    if sl_pct is not None and tp_pct is not None:
        # Use bar-by-bar SL/TP backtest
        df, trades = run_backtest_with_sl_tp(df, sl_pct=sl_pct, tp_pct=tp_pct)
    else:
        # Use standard vectorized backtest
        df, trades = run_backtest(df)

    # 5. Calculate metrics
    metrics = calculate_metrics(df, trades)

    # 6. Get equity curve
    equity_curve = get_equity_curve(df)

    # 7. Monte Carlo
    monte_carlo = run_monte_carlo(trades)

    # 8. Walk-forward analysis (need fresh data with indicators)
    df_wf = fetch_data(asset, timeframe, lookback)
    df_wf = calculate_indicators(df_wf, rules)
    df_wf = df_wf.dropna()
    walk_forward = run_walk_forward(df_wf, rules)

    # 9. Get last 3 trades for verification
    last_3_trades = trades[-3:] if len(trades) >= 3 else trades

    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "monte_carlo": monte_carlo,
        "last_3_trades": last_3_trades,
        "walk_forward": walk_forward,
    }
