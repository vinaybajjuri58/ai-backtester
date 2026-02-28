"""FastAPI application for AI Backtester."""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback

from models import BacktestRequest, BacktestResponse
from agent import translate_hypothesis, classify_strategy
from backtest import execute_backtest
from executor import generate_code

app = FastAPI(title="AI Backtester", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Backtester API"}


@app.post("/api/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    try:
        # 1. Translate hypothesis to structured rules
        rules = await translate_hypothesis(request.hypothesis)

        # 2. Classify strategy (mean reversion vs trend following)
        classification = classify_strategy(rules)

        # 3. Run backtest
        results = await execute_backtest(
            hypothesis=request.hypothesis,
            asset=request.asset,
            timeframe=request.timeframe,
            lookback=request.lookback,
            rules=rules,
        )

        # 4. Generate standalone code
        code = generate_code(rules, request.asset, request.timeframe, request.lookback)

        return BacktestResponse(
            metrics=results["metrics"],
            equity_curve=results["equity_curve"],
            monte_carlo=results["monte_carlo"],
            walk_forward=results["walk_forward"],
            generated_code=code,
            strategy_rules=rules,
            classification=classification,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
