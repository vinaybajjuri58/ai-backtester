"""Code Generation - Generates standalone Python backtest code from trading rules."""


def generate_backtest_logic(strategy_type: str, use_sl_tp: bool = False) -> str:
    """Generate the appropriate backtest logic based on strategy type.
    
    Args:
        strategy_type: The type of strategy
        use_sl_tp: If True, use bar-by-bar SL/TP backtest regardless of strategy type
    """
    
    if use_sl_tp or strategy_type == "confluence_rsi_ema":
        # SL/TP backtest logic for confluence with entry/exit times
        return '''# Bar-by-bar backtest with Stop Loss and Take Profit
# Entry: at close of signal bar
# Exit: SL/TP checked from next bar onward (avoid same-bar exit)
# Conservative: if both SL and TP hit in same bar, SL takes precedence

trades = []
position = 0  # 0 = flat, 1 = long
entry_price = 0.0
entry_time = None
sl_price = 0.0
tp_price = 0.0

equity = [INITIAL_CAPITAL]

for i in range(1, len(df)):
    current_bar = df.iloc[i]
    current_time = df.index[i]
    
    if position == 0:
        # Check for entry signal
        if current_bar["signal"] == 1:
            position = 1
            entry_price = float(current_bar["close"])
            entry_time = current_time
            sl_price = entry_price * (1 - SL_PCT)
            tp_price = entry_price * (1 + TP_PCT)
            # Don't check SL/TP on entry bar - wait for next bar
    else:
        # In position - check SL/TP (evaluating current bar)
        bar_high = float(current_bar["high"])
        bar_low = float(current_bar["low"])
        
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
            pnl_pct = (exit_price - entry_price) / entry_price
            trades.append({
                "signal_type": "LONG",
                "entry_time": entry_time.strftime("%Y-%m-%d %H:%M UTC"),
                "entry_time_ist": (entry_time + pd.Timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M IST"),
                "exit_time": current_time.strftime("%Y-%m-%d %H:%M UTC"),
                "exit_time_ist": (current_time + pd.Timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M IST"),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl_pct": float(pnl_pct),
                "exit_reason": exit_reason,
            })
            position = 0
            entry_price = 0.0
            entry_time = None
    
    # Mark-to-market equity
    if position == 1:
        bar_close = float(current_bar["close"])
        unrealized = (bar_close - entry_price) / entry_price
        equity.append(INITIAL_CAPITAL * (1 + unrealized))
    else:
        if len(equity) > 0:
            equity.append(equity[-1])
        else:
            equity.append(INITIAL_CAPITAL)

# Close any open position at end
if position == 1:
    final_price = float(df["close"].iloc[-1])
    final_time = df.index[-1]
    pnl_pct = (final_price - entry_price) / entry_price
    trades.append({
        "signal_type": "LONG",
        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M UTC"),
        "entry_time_ist": (entry_time + pd.Timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M IST"),
        "exit_time": final_time.strftime("%Y-%m-%d %H:%M UTC"),
        "exit_time_ist": (final_time + pd.Timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M IST"),
        "entry_price": float(entry_price),
        "exit_price": float(final_price),
        "pnl_pct": float(pnl_pct),
        "exit_reason": "END",
    })

df["equity"] = equity[:len(df)]
df["strategy_return"] = pd.Series(equity).pct_change().fillna(0).iloc[:len(df)]'''
    else:
        # Standard vectorized backtest for other strategies
        return '''# Vectorized backtest
df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
df["market_return"] = df["close"].pct_change()
df["strategy_return"] = df["position"].shift(1) * df["market_return"]
df["strategy_return"] = df["strategy_return"].fillna(0)
df["equity"] = INITIAL_CAPITAL * (1 + df["strategy_return"]).cumprod()

# Extract trades
trades = []
in_trade = False
entry_price = 0

for i in range(1, len(df)):
    prev_pos = df["position"].iloc[i - 1]
    curr_pos = df["position"].iloc[i]
    if not in_trade and curr_pos == 1:
        in_trade = True
        entry_price = df["close"].iloc[i]
    elif in_trade and curr_pos != 1:
        exit_price = df["close"].iloc[i]
        pnl_pct = (exit_price - entry_price) / entry_price
        trades.append(float(pnl_pct))
        in_trade = False'''


def generate_code(rules: dict, asset: str, timeframe: str, lookback: str) -> str:
    """Generate clean, readable Python backtest code from the trading rules."""
    strategy_type = rules.get("strategy_type", "rsi")
    params = rules.get("parameters", {})
    description = rules.get("description", "Trading Strategy")

    code = f'''"""
{description}
Auto-generated by AI Backtester
Asset: {asset} | Timeframe: {timeframe} | Lookback: {lookback}
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta

# ============================================================
# 1. Configuration
# ============================================================
'''

    asset_map = {"BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD"}
    ticker = asset_map.get(asset, "BTC-USD")
    lookback_map = {"1 month": 30, "3 months": 90, "6 months": 180, "1 year": 365, "2 years": 730}
    days = lookback_map.get(lookback, 365)

    code += f'TICKER = "{ticker}"\n'
    code += f'INTERVAL = "{timeframe}"\n'
    code += f'LOOKBACK_DAYS = {days}\n'
    code += f'INITIAL_CAPITAL = 10000\n\n'

    if strategy_type == "rsi":
        code += f'RSI_PERIOD = {params.get("rsi_period", 14)}\n'
        code += f'ENTRY_THRESHOLD = {params.get("entry_threshold", 30)}\n'
        code += f'EXIT_THRESHOLD = {params.get("exit_threshold", 70)}\n'
    elif strategy_type == "ma_crossover":
        code += f'FAST_PERIOD = {params.get("fast_period", 10)}\n'
        code += f'SLOW_PERIOD = {params.get("slow_period", 50)}\n'
        code += f'MA_TYPE = "{params.get("ma_type", "sma")}"\n'
    elif strategy_type == "bollinger":
        code += f'BB_PERIOD = {params.get("bb_period", 20)}\n'
        code += f'BB_STD = {params.get("bb_std", 2.0)}\n'
        code += f'BB_MODE = "{params.get("mode", "mean_reversion")}"\n'
    elif strategy_type == "macd":
        code += f'MACD_FAST = {params.get("fast_period", 12)}\n'
        code += f'MACD_SLOW = {params.get("slow_period", 26)}\n'
        code += f'MACD_SIGNAL = {params.get("signal_period", 9)}\n'
    elif strategy_type == "breakout":
        code += f'BREAKOUT_PERIOD = {params.get("breakout_period", 20)}\n'
    elif strategy_type == "confluence_rsi_ema":
        code += f'RSI_PERIOD = {params.get("rsi_period", 14)}\n'
        code += f'RSI_ENTRY = {params.get("rsi_entry_threshold", 30)}\n'
        code += f'FAST_EMA = {params.get("fast_period", 10)}\n'
        code += f'SLOW_EMA = {params.get("slow_period", 50)}\n'
    
    # Add SL/TP config if specified (works with any strategy)
    sl_pct = params.get("sl_pct")
    tp_pct = params.get("tp_pct")
    if sl_pct is not None:
        code += f'SL_PCT = {sl_pct}\n'
    if tp_pct is not None:
        code += f'TP_PCT = {tp_pct}\n'
    use_sl_tp = sl_pct is not None and tp_pct is not None

    code += '''
# ============================================================
# 2. Fetch Market Data
# ============================================================
end_date = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_DAYS)

df = yf.download(
    TICKER,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    interval=INTERVAL,
    progress=False,
)

# Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
print(f"Fetched {len(df)} bars of {TICKER} data")

# ============================================================
# 3. Calculate Indicators
# ============================================================
'''

    if strategy_type == "rsi":
        code += 'df["rsi"] = ta.rsi(df["close"], length=RSI_PERIOD)\n'
    elif strategy_type == "ma_crossover":
        code += '''if MA_TYPE == "ema":
    df["fast_ma"] = ta.ema(df["close"], length=FAST_PERIOD)
    df["slow_ma"] = ta.ema(df["close"], length=SLOW_PERIOD)
else:
    df["fast_ma"] = ta.sma(df["close"], length=FAST_PERIOD)
    df["slow_ma"] = ta.sma(df["close"], length=SLOW_PERIOD)
'''
    elif strategy_type == "bollinger":
        code += '''bb = ta.bbands(df["close"], length=BB_PERIOD, std=BB_STD)
df["bb_upper"] = bb.iloc[:, 2]
df["bb_middle"] = bb.iloc[:, 1]
df["bb_lower"] = bb.iloc[:, 0]
'''
    elif strategy_type == "macd":
        code += '''macd_df = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
df["macd"] = macd_df.iloc[:, 0]
df["macd_hist"] = macd_df.iloc[:, 1]
df["macd_signal"] = macd_df.iloc[:, 2]
'''
    elif strategy_type == "breakout":
        code += '''df["high_breakout"] = df["high"].rolling(window=BREAKOUT_PERIOD).max()
df["low_breakout"] = df["low"].rolling(window=BREAKOUT_PERIOD).min()
'''
    elif strategy_type == "confluence_rsi_ema":
        code += '''df["rsi"] = ta.rsi(df["close"], length=RSI_PERIOD)
df["fast_ema"] = ta.ema(df["close"], length=FAST_EMA)
df["slow_ema"] = ta.ema(df["close"], length=SLOW_EMA)
df["confluence_long"] = (df["rsi"] < RSI_ENTRY) & (df["fast_ema"] > df["slow_ema"])
'''

    code += '''
df = df.dropna()

# ============================================================
# 4. Generate Trading Signals
# ============================================================
df["signal"] = 0
'''

    if strategy_type == "rsi":
        code += '''df.loc[df["rsi"] < ENTRY_THRESHOLD, "signal"] = 1   # Buy signal
df.loc[df["rsi"] > EXIT_THRESHOLD, "signal"] = -1  # Sell signal
'''
    elif strategy_type == "ma_crossover":
        code += '''df.loc[df["fast_ma"] > df["slow_ma"], "signal"] = 1   # Buy signal
df.loc[df["fast_ma"] < df["slow_ma"], "signal"] = -1  # Sell signal
'''
    elif strategy_type == "bollinger":
        code += '''if BB_MODE == "mean_reversion":
    df.loc[df["close"] < df["bb_lower"], "signal"] = 1    # Buy at lower band
    df.loc[df["close"] > df["bb_middle"], "signal"] = -1  # Sell at middle band
else:
    df.loc[df["close"] > df["bb_upper"], "signal"] = 1    # Buy breakout
    df.loc[df["close"] < df["bb_middle"], "signal"] = -1  # Sell at middle band
'''
    elif strategy_type == "macd":
        code += '''df.loc[df["macd"] > df["macd_signal"], "signal"] = 1   # Buy signal
df.loc[df["macd"] < df["macd_signal"], "signal"] = -1  # Sell signal
'''
    elif strategy_type == "breakout":
        code += '''df.loc[df["close"] > df["high_breakout"].shift(1), "signal"] = 1   # Buy breakout
df.loc[df["close"] < df["low_breakout"].shift(1), "signal"] = -1  # Sell breakdown
'''
    elif strategy_type == "confluence_rsi_ema":
        code += '''# Confluence entry: RSI oversold + EMA uptrend
df.loc[df["confluence_long"], "signal"] = 1
df.loc[~df["confluence_long"], "signal"] = -1
'''

    code += '''
# ============================================================
# 5. Run Backtest
# ============================================================

''' + generate_backtest_logic(strategy_type, use_sl_tp=use_sl_tp) + '''

# ============================================================
# 6. Calculate Metrics
# ============================================================
wins = [t for t in trades if t > 0]
losses = [t for t in trades if t <= 0]

total_return = ((df["equity"].iloc[-1] / INITIAL_CAPITAL) - 1) * 100
win_rate = len(wins) / len(trades) * 100 if trades else 0
avg_win = np.mean(wins) * 100 if wins else 0
avg_loss = np.mean(losses) * 100 if losses else 0
profit_factor = sum(wins) / abs(sum(losses)) if losses else 0

peak = df["equity"].cummax()
drawdown = (df["equity"] - peak) / peak
max_drawdown = drawdown.min() * 100

returns = df["strategy_return"].dropna()
sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

print(f"\\n{'='*50}")
print(f"BACKTEST RESULTS")
print(f"{'='*50}")
print(f"Total Return:   {total_return:.2f}%")
print(f"Win Rate:        {win_rate:.2f}%")
print(f"Sharpe Ratio:    {sharpe:.2f}")
print(f"Profit Factor:   {profit_factor:.2f}")
print(f"Max Drawdown:    {max_drawdown:.2f}%")
print(f"Total Trades:    {len(trades)}")
print(f"Avg Win:         {avg_win:.2f}%")
print(f"Avg Loss:        {avg_loss:.2f}%")
print(f"Expectancy:      {np.mean(trades)*100:.2f}%" if trades else "Expectancy:      N/A")

# ============================================================
# 7. Plot Results (optional - requires matplotlib)
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(df.index, df["equity"], label="Strategy Equity", color="#06b6d4")
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(df.index, drawdown * 100, 0, alpha=0.3, color="red")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150)
    print("\\nChart saved to backtest_results.png")
except ImportError:
    print("\\nInstall matplotlib to generate charts: pip install matplotlib")
'''

    return code
