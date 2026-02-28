"""LLM Strategy Translation - Converts plain English to structured trading rules.

Supports multiple LLM providers:
- Anthropic Claude (via ANTHROPIC_API_KEY)
- Moonshot AI Kimi K2.5 (via MOONSHOT_API_KEY)

Falls back to rule-based parsing if no LLM is available.
"""

import os
import json
import re
import httpx
from typing import Optional


# ─── LLM Configuration ─────────────────────────────────────────────

LLM_PROMPT_TEMPLATE = """You are a quantitative trading strategy parser. Convert the following plain English trading hypothesis into a structured JSON format.

Hypothesis: "{hypothesis}"

Return ONLY valid JSON (no markdown, no code blocks) with this exact structure:
{{
    "strategy_type": "rsi|ma_crossover|bollinger|macd|breakout|confluence_rsi_ema|custom",
    "description": "Brief description of the strategy",
    "entry_condition": "Human-readable entry condition",
    "exit_condition": "Human-readable exit condition",
    "indicators": [
        {{"name": "indicator_name", "params": {{"period": 14, ...}}}}
    ],
    "parameters": {{
        "entry_threshold": value,
        "exit_threshold": value,
        "sl_pct": 0.01,        // Optional: Stop loss percentage (e.g., 0.01 = 1%)
        "tp_pct": 0.02,        // Optional: Take profit percentage (e.g., 0.02 = 2%)
        ... any other relevant parameters
    }}
}}

Indicator names must be one of: rsi, sma, ema, bbands, macd, atr, stoch
Parameter keys should be descriptive. Include all numeric thresholds from the hypothesis.

IMPORTANT: If the user mentions stop loss (SL) or take profit (TP) percentages, include them as sl_pct and tp_pct in parameters.
Examples:
- "stop loss 1%" → "sl_pct": 0.01
- "take profit 3%" → "tp_pct": 0.03"""


# ─── Strategy Classification ───────────────────────────────────────

STRATEGY_CLASSIFICATIONS = {
    "rsi": {
        "default": "mean_reversion",
        "description": "RSI is inherently a mean-reversion indicator — it identifies overbought/oversold conditions expecting price to revert to the mean.",
        "reasoning": {
            "mean_reversion": "Buying oversold (low RSI) and selling overbought (high RSI) is classic mean reversion — betting price returns to its average.",
            "trend_following": "Some traders use RSI breakouts (e.g., RSI > 50 = bullish) as trend confirmation.",
        },
    },
    "ma_crossover": {
        "default": "trend_following",
        "description": "Moving average crossovers are classic trend-following signals — they identify and ride momentum when a shorter MA overtakes a longer one.",
        "reasoning": {
            "trend_following": "Fast MA crossing above slow MA indicates upward momentum — you're following the trend, not betting against it.",
        },
    },
    "bollinger": {
        "mean_reversion": {
            "description": "Bollinger Band bounce strategies are mean reversion — buying at the lower band expects price to revert to the middle band (mean).",
            "reasoning": "Price touching the lower band is ~2σ below the mean — statistically likely to revert. You're betting on normalization.",
        },
        "breakout": {
            "description": "Bollinger Band breakout strategies are trend-following — buying above the upper band expects momentum continuation.",
            "reasoning": "Price breaking above the upper band signals strong momentum. You're following the trend expecting continuation.",
        },
    },
    "macd": {
        "default": "trend_following",
        "description": "MACD is a trend-following momentum indicator — crossovers signal the beginning of new momentum phases.",
        "reasoning": {
            "trend_following": "MACD line crossing above signal indicates accelerating bullish momentum — a classic trend-following entry.",
        },
    },
    "breakout": {
        "default": "trend_following",
        "description": "Price breakout strategies are trend-following — breaking a N-day high/low signals new momentum you aim to ride.",
        "reasoning": {
            "trend_following": "New highs/lows indicate strong directional momentum. Breakout traders follow the trend expecting continuation (Donchian/Turtle style).",
        },
    },
    "confluence_rsi_ema": {
        "default": "trend_following",
        "description": "RSI + EMA confluence combines mean-reversion entry (oversold RSI) with trend confirmation (EMA alignment) for higher-probability entries.",
        "reasoning": {
            "trend_following": "EMA(10) > EMA(50) confirms uptrend, while RSI < 30 ensures buying at temporary weakness within the trend — confluence increases entry quality.",
        },
    },
}


def classify_strategy(rules: dict) -> dict:
    """Classify a strategy as mean reversion or trend following with reasoning."""
    strategy_type = rules.get("strategy_type", "rsi")
    params = rules.get("parameters", {})

    classification_info = STRATEGY_CLASSIFICATIONS.get(strategy_type)
    if not classification_info:
        return {
            "classification": "unknown",
            "label": "Unknown",
            "description": "Strategy type could not be classified.",
            "reasoning": "Insufficient information to determine strategy nature.",
            "confidence": "low",
        }

    # Bollinger has mode-dependent classification
    if strategy_type == "bollinger":
        mode = params.get("mode", "mean_reversion")
        info = classification_info.get(mode, classification_info.get("mean_reversion"))
        classification = "mean_reversion" if mode == "mean_reversion" else "trend_following"
        return {
            "classification": classification,
            "label": "Mean Reversion" if classification == "mean_reversion" else "Trend Following",
            "description": info["description"],
            "reasoning": info["reasoning"],
            "confidence": "high",
        }

    # All other strategies
    default_class = classification_info["default"]
    label = "Mean Reversion" if default_class == "mean_reversion" else "Trend Following"
    description = classification_info["description"]
    reasoning = classification_info["reasoning"].get(default_class, "")

    return {
        "classification": default_class,
        "label": label,
        "description": description,
        "reasoning": reasoning,
        "confidence": "high",
    }


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Extract and parse JSON from text response."""
    try:
        # Try to extract JSON if wrapped in code blocks
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
        return None
    except json.JSONDecodeError:
        return None


async def translate_hypothesis_claude(hypothesis: str) -> Optional[dict]:
    """Use Anthropic Claude API to translate hypothesis to structured trading rules."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    prompt = LLM_PROMPT_TEMPLATE.format(hypothesis=hypothesis)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if response.status_code == 200:
                data = response.json()
                text = data["content"][0]["text"].strip()
                return _extract_json_from_text(text)
            else:
                print(f"Claude API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Claude translation failed: {e}")
        return None


async def translate_hypothesis_openai(hypothesis: str) -> Optional[dict]:
    """Use OpenAI API to translate hypothesis to structured trading rules.
    
    Env vars:
    - OPENAI_API_KEY (required)
    - OPENAI_MODEL (optional, defaults to gpt-4o-mini)
    - OPENAI_BASE_URL (optional, for Azure or proxies)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not base_url.endswith("/chat/completions"):
        base_url = base_url.rstrip("/") + "/chat/completions"
    
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = LLM_PROMPT_TEMPLATE.format(hypothesis=hypothesis)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if response.status_code == 200:
                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()
                return _extract_json_from_text(text)
            else:
                print(f"OpenAI API error: {response.status_code} - {response.text}")
                print(f"  Model: {model}")
            return None
    except Exception as e:
        print(f"OpenAI translation failed: {e}")
        return None


async def translate_hypothesis_kimi(hypothesis: str) -> Optional[dict]:
    """Use Kimi K2.5 API to translate hypothesis to structured trading rules.
    
    Supports:
    - Direct Moonshot API (api.moonshot.cn)
    - OpenRouter (openrouter.ai) and other OpenAI-compatible proxies
    
    Env vars:
    - KIMI_API_KEY or MOONSHOT_API_KEY or OPENROUTER_API_KEY
    - KIMI_BASE_URL (optional, defaults to Moonshot)
    - KIMI_MODEL (optional, defaults to kimi-k2.5)
    """
    # Try different API key env vars
    api_key = (
        os.environ.get("KIMI_API_KEY") or 
        os.environ.get("MOONSHOT_API_KEY") or 
        os.environ.get("OPENROUTER_API_KEY")
    )
    if not api_key:
        return None

    # Get base URL (support OpenRouter and other proxies)
    base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    if not base_url.endswith("/chat/completions"):
        base_url = base_url.rstrip("/") + "/chat/completions"
    
    # Get model name
    model = os.environ.get("KIMI_MODEL", "kimi-k2.5")
    
    prompt = LLM_PROMPT_TEMPLATE.format(hypothesis=hypothesis)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            # OpenRouter requires extra headers
            if "openrouter" in base_url:
                headers["HTTP-Referer"] = os.environ.get("APP_URL", "http://localhost:8000")
                headers["X-Title"] = "AI Backtester"
            
            response = await client.post(
                base_url,
                headers=headers,
                json={
                    "model": model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if response.status_code == 200:
                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()
                return _extract_json_from_text(text)
            else:
                print(f"Kimi API error: {response.status_code} - {response.text}")
                print(f"  Base URL: {base_url}")
                print(f"  Model: {model}")
            return None
    except Exception as e:
        print(f"Kimi translation failed: {e}")
        return None


def _extract_sl_tp(hypothesis: str) -> tuple[float, float]:
    """Extract SL and TP percentages from hypothesis text.
    Returns (sl_pct, tp_pct) - defaults to (None, None) if not specified.
    """
    h = hypothesis.lower()
    
    # Try various SL patterns
    sl_patterns = [
        r'stop\s*loss\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'sl\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'sl\s*(\d+(?:\.\d+)?)\s*%',
    ]
    sl_pct = None
    for pattern in sl_patterns:
        match = re.search(pattern, h)
        if match:
            sl_pct = float(match.group(1)) / 100
            break
    
    # Try various TP patterns
    tp_patterns = [
        r'take\s*profit\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'tp\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'tp\s*(\d+(?:\.\d+)?)\s*%',
    ]
    tp_pct = None
    for pattern in tp_patterns:
        match = re.search(pattern, h)
        if match:
            tp_pct = float(match.group(1)) / 100
            break
    
    return sl_pct, tp_pct


def parse_hypothesis_fallback(hypothesis: str) -> dict:
    """Rule-based parser for common trading strategy patterns."""
    h = hypothesis.lower().strip()
    
    # Check for SL/TP in any hypothesis
    sl_pct, tp_pct = _extract_sl_tp(hypothesis)

    # Confluence: RSI + EMA
    if ('confluence' in h or ('rsi' in h and 'ema' in h)) and ('both' in h or 'and' in h):
        # Extract RSI parameters
        rsi_period_match = re.search(r'(\d+)\s*(?:period|day|bar)?\s*rsi', h)
        rsi_period = int(rsi_period_match.group(1)) if rsi_period_match else 14
        rsi_threshold_match = re.search(r'rsi\s*(?:is\s*)?(?:below|under|<)\s*(\d+)', h)
        rsi_threshold = int(rsi_threshold_match.group(1)) if rsi_threshold_match else 30
        
        # Extract EMA parameters
        ema_matches = re.findall(r'ema\s*\(?\s*(\d+)\s*\)?', h)
        if len(ema_matches) >= 2:
            fast_ema = int(ema_matches[0])
            slow_ema = int(ema_matches[1])
            if fast_ema > slow_ema:
                fast_ema, slow_ema = slow_ema, fast_ema
        else:
            fast_ema, slow_ema = 10, 50
        
        # Extract SL/TP if mentioned
        sl_match = re.search(r'stop\s*loss\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%', h)
        sl_pct = float(sl_match.group(1)) / 100 if sl_match else 0.01
        tp_match = re.search(r'take\s*profit\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%', h)
        tp_pct = float(tp_match.group(1)) / 100 if tp_match else 0.02
        
        result = {
            "strategy_type": "confluence_rsi_ema",
            "description": f"RSI({rsi_period}) < {rsi_threshold} + EMA({fast_ema}) > EMA({slow_ema}) confluence" + (f" with SL {sl_pct*100:.1f}% / TP {tp_pct*100:.1f}%" if sl_pct and tp_pct else ""),
            "entry_condition": f"RSI({rsi_period}) < {rsi_threshold} AND EMA({fast_ema}) > EMA({slow_ema})",
            "exit_condition": f"Stop Loss {sl_pct*100:.1f}% OR Take Profit {tp_pct*100:.1f}%" if (sl_pct and tp_pct) else "Signal change",
            "indicators": [
                {"name": "rsi", "params": {"period": rsi_period}},
                {"name": "ema", "params": {"period": fast_ema}},
                {"name": "ema", "params": {"period": slow_ema}},
            ],
            "parameters": {
                "rsi_period": rsi_period,
                "rsi_entry_threshold": rsi_threshold,
                "fast_period": fast_ema,
                "slow_period": slow_ema,
            },
        }
        if sl_pct:
            result["parameters"]["sl_pct"] = sl_pct
        if tp_pct:
            result["parameters"]["tp_pct"] = tp_pct
        return result

    # RSI strategies
    rsi_match = re.search(r'rsi\s*(?:drops?\s*)?(?:below|under|<)\s*(\d+)', h)
    rsi_exit = re.search(r'rsi\s*(?:crosses?\s*)?(?:above|over|>)\s*(\d+)', h)
    rsi_sell_below = re.search(r'sell.*rsi\s*(?:drops?\s*)?(?:below|under|<)\s*(\d+)', h)
    rsi_buy_above = re.search(r'buy.*rsi\s*(?:crosses?\s*)?(?:above|over|>)\s*(\d+)', h)

    # Also check "when rsi is below X buy" pattern
    if not rsi_match:
        rsi_match = re.search(r'rsi\s*(?:is\s*)?(?:below|under|<)\s*(\d+).*buy', h)
    if not rsi_exit:
        rsi_exit = re.search(r'rsi\s*(?:is\s*)?(?:above|over|>)\s*(\d+).*sell', h)

    if rsi_match or rsi_exit or rsi_sell_below or rsi_buy_above:
        rsi_period_match = re.search(r'(\d+)\s*(?:period|day|bar)?\s*rsi', h)
        rsi_period = int(rsi_period_match.group(1)) if rsi_period_match else 14
        entry_val = int(rsi_match.group(1)) if rsi_match else 30
        exit_val = int(rsi_exit.group(1)) if rsi_exit else 70

        result = {
            "strategy_type": "rsi",
            "description": f"RSI strategy: Buy when RSI < {entry_val}, Sell when RSI > {exit_val}" + (f" with SL {sl_pct*100:.1f}% / TP {tp_pct*100:.1f}%" if sl_pct and tp_pct else ""),
            "entry_condition": f"RSI({rsi_period}) crosses below {entry_val}",
            "exit_condition": f"Stop Loss {sl_pct*100:.1f}% OR Take Profit {tp_pct*100:.1f}%" if (sl_pct and tp_pct) else f"RSI({rsi_period}) crosses above {exit_val}",
            "indicators": [{"name": "rsi", "params": {"period": rsi_period}}],
            "parameters": {
                "entry_threshold": entry_val,
                "exit_threshold": exit_val,
                "rsi_period": rsi_period,
            },
        }
        if sl_pct:
            result["parameters"]["sl_pct"] = sl_pct
        if tp_pct:
            result["parameters"]["tp_pct"] = tp_pct
        return result

    # Moving Average Crossover
    ma_cross = re.search(
        r'(\d+)\s*(?:period|day|bar)?\s*(?:sma|ema|ma|moving\s*average).*(?:cross|above|over).*(\d+)\s*(?:period|day|bar)?\s*(?:sma|ema|ma|moving\s*average)',
        h,
    )
    if not ma_cross:
        ma_cross = re.search(
            r'(?:sma|ema|ma)\s*\(?(\d+)\)?\s*(?:cross|above|over).*(?:sma|ema|ma)\s*\(?(\d+)\)?',
            h,
        )
    # Also "short MA crosses above long MA" or "X day crosses Y day"
    if not ma_cross:
        ma_cross = re.search(r'(\d+)\s*(?:day|period).*cross.*(\d+)\s*(?:day|period)', h)

    if ma_cross or 'moving average' in h or 'crossover' in h or ('ma ' in h and 'cross' in h) or ('ema' in h and 'cross' in h):
        ma_type = "ema" if "ema" in h else "sma"
        if ma_cross:
            fast = int(ma_cross.group(1))
            slow = int(ma_cross.group(2))
            if fast > slow:
                fast, slow = slow, fast
        else:
            fast, slow = 10, 50

        result = {
            "strategy_type": "ma_crossover",
            "description": f"{ma_type.upper()} Crossover: {fast}/{slow}" + (f" with SL {sl_pct*100:.1f}% / TP {tp_pct*100:.1f}%" if sl_pct and tp_pct else ""),
            "entry_condition": f"{ma_type.upper()}({fast}) crosses above {ma_type.upper()}({slow})",
            "exit_condition": f"Stop Loss {sl_pct*100:.1f}% OR Take Profit {tp_pct*100:.1f}%" if (sl_pct and tp_pct) else f"{ma_type.upper()}({fast}) crosses below {ma_type.upper()}({slow})",
            "indicators": [
                {"name": ma_type, "params": {"period": fast}},
                {"name": ma_type, "params": {"period": slow}},
            ],
            "parameters": {
                "fast_period": fast,
                "slow_period": slow,
                "ma_type": ma_type,
            },
        }
        if sl_pct:
            result["parameters"]["sl_pct"] = sl_pct
        if tp_pct:
            result["parameters"]["tp_pct"] = tp_pct
        return result

    # Bollinger Band strategies
    if 'bollinger' in h or 'bband' in h or 'bb ' in h:
        bb_period_match = re.search(r'(\d+)\s*(?:period|day|bar)', h)
        bb_period = int(bb_period_match.group(1)) if bb_period_match else 20
        bb_std_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:std|deviation|sigma)', h)
        bb_std = float(bb_std_match.group(1)) if bb_std_match else 2.0

        # Determine if mean reversion or breakout
        if 'lower' in h or ('below' in h and 'band' in h) or 'bounce' in h or 'revert' in h or 'mean' in h or 'touch' in h:
            entry_cond = f"Price touches lower Bollinger Band ({bb_period}, {bb_std}σ)"
            exit_cond = f"Price reaches middle Bollinger Band ({bb_period})"
            bb_mode = "mean_reversion"
        else:
            entry_cond = f"Price breaks above upper Bollinger Band ({bb_period}, {bb_std}σ)"
            exit_cond = f"Price falls below middle Bollinger Band ({bb_period})"
            bb_mode = "breakout"

        result = {
            "strategy_type": "bollinger",
            "description": f"Bollinger Band {bb_mode}: {bb_period} period, {bb_std}σ" + (f" with SL {sl_pct*100:.1f}% / TP {tp_pct*100:.1f}%" if sl_pct and tp_pct else ""),
            "entry_condition": entry_cond,
            "exit_condition": f"Stop Loss {sl_pct*100:.1f}% OR Take Profit {tp_pct*100:.1f}%" if (sl_pct and tp_pct) else exit_cond,
            "indicators": [{"name": "bbands", "params": {"period": bb_period, "std": bb_std}}],
            "parameters": {
                "bb_period": bb_period,
                "bb_std": bb_std,
                "mode": bb_mode,
            },
        }
        if sl_pct:
            result["parameters"]["sl_pct"] = sl_pct
        if tp_pct:
            result["parameters"]["tp_pct"] = tp_pct
        return result

    # MACD strategies
    if 'macd' in h:
        fast_match = re.search(r'fast\s*(?:period)?\s*(\d+)', h)
        slow_match = re.search(r'slow\s*(?:period)?\s*(\d+)', h)
        signal_match = re.search(r'signal\s*(?:period)?\s*(\d+)', h)
        fast_p = int(fast_match.group(1)) if fast_match else 12
        slow_p = int(slow_match.group(1)) if slow_match else 26
        signal_p = int(signal_match.group(1)) if signal_match else 9

        result = {
            "strategy_type": "macd",
            "description": f"MACD Crossover: {fast_p}/{slow_p}/{signal_p}" + (f" with SL {sl_pct*100:.1f}% / TP {tp_pct*100:.1f}%" if sl_pct and tp_pct else ""),
            "entry_condition": f"MACD line crosses above signal line",
            "exit_condition": f"Stop Loss {sl_pct*100:.1f}% OR Take Profit {tp_pct*100:.1f}%" if (sl_pct and tp_pct) else f"MACD line crosses below signal line",
            "indicators": [
                {"name": "macd", "params": {"fast": fast_p, "slow": slow_p, "signal": signal_p}}
            ],
            "parameters": {
                "fast_period": fast_p,
                "slow_period": slow_p,
                "signal_period": signal_p,
            },
        }
        if sl_pct:
            result["parameters"]["sl_pct"] = sl_pct
        if tp_pct:
            result["parameters"]["tp_pct"] = tp_pct
        return result

    # Price breakout strategies
    if 'breakout' in h or 'break above' in h or 'break below' in h or 'high' in h:
        period_match = re.search(r'(\d+)\s*(?:period|day|bar|candle)', h)
        period = int(period_match.group(1)) if period_match else 20

        result = {
            "strategy_type": "breakout",
            "description": f"Price Breakout: {period}-period high/low" + (f" with SL {sl_pct*100:.1f}% / TP {tp_pct*100:.1f}%" if sl_pct and tp_pct else ""),
            "entry_condition": f"Price breaks above {period}-period high",
            "exit_condition": f"Stop Loss {sl_pct*100:.1f}% OR Take Profit {tp_pct*100:.1f}%" if (sl_pct and tp_pct) else f"Price breaks below {period}-period low",
            "indicators": [],
            "parameters": {
                "breakout_period": period,
            },
        }
        if sl_pct:
            result["parameters"]["sl_pct"] = sl_pct
        if tp_pct:
            result["parameters"]["tp_pct"] = tp_pct
        return result

    # Default: RSI with standard params
    result = {
        "strategy_type": "rsi",
        "description": "Default RSI strategy (couldn't parse specific rules)" + (f" with SL {sl_pct*100:.1f}% / TP {tp_pct*100:.1f}%" if sl_pct and tp_pct else ""),
        "entry_condition": "RSI(14) crosses below 30",
        "exit_condition": f"Stop Loss {sl_pct*100:.1f}% OR Take Profit {tp_pct*100:.1f}%" if (sl_pct and tp_pct) else "RSI(14) crosses above 70",
        "indicators": [{"name": "rsi", "params": {"period": 14}}],
        "parameters": {
            "entry_threshold": 30,
            "exit_threshold": 70,
            "rsi_period": 14,
        },
    }
    if sl_pct:
        result["parameters"]["sl_pct"] = sl_pct
    if tp_pct:
        result["parameters"]["tp_pct"] = tp_pct
    return result


async def translate_hypothesis_llm(hypothesis: str) -> Optional[dict]:
    """Try LLM providers in order: Claude -> OpenAI -> Kimi.
    
    Returns the first successful result, or None if all fail.
    """
    # Try Claude first
    result = await translate_hypothesis_claude(hypothesis)
    if result:
        print("Using Claude for strategy translation")
        return result
    
    # Try OpenAI
    result = await translate_hypothesis_openai(hypothesis)
    if result:
        print("Using OpenAI for strategy translation")
        return result
    
    # Fall back to Kimi K2.5
    result = await translate_hypothesis_kimi(hypothesis)
    if result:
        print("Using Kimi K2.5 for strategy translation")
        return result
    
    return None


async def translate_hypothesis(hypothesis: str) -> dict:
    """Main entry point: try LLM first, fall back to rule-based parser.
    
    Priority:
    1. Anthropic Claude (if ANTHROPIC_API_KEY is set)
    2. Moonshot Kimi K2.5 (if MOONSHOT_API_KEY is set)
    3. Rule-based regex parser (always available)
    """
    result = await translate_hypothesis_llm(hypothesis)
    if result:
        return result
    
    print("Using rule-based parser for strategy translation")
    return parse_hypothesis_fallback(hypothesis)
