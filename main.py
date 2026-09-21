"""
main.py
 
FastAPI wrapper around dcf_engine.py. This is the "mouth" that lets your
Next.js frontend talk to the DCF calculation engine over HTTP.
 
Run locally:
    uvicorn main:app --reload
 
Then open http://127.0.0.1:8000/docs for an interactive test page —
FastAPI auto-generates this, no extra work needed.
 
Install: pip install fastapi uvicorn yfinance pandas numpy
"""
 
import time
from typing import Optional
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
 
from dcf_engine import fetch_company_data, run_dcf, sensitivity_analysis, find_implied_growth_rate
 
 
app = FastAPI(title="DCF Valuation API")
 
# Allows your Next.js dev server (localhost:3000) to call this API from the
# browser. Wildcard "*" is fine for a personal project; tighten to your
# actual domain once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# ---------------------------------------------------------------------------
# Simple in-memory cache so re-running the same ticker doesn't hammer Yahoo
# Finance every time a slider moves. Free, no database needed. Resets when
# the server restarts — that's fine for a personal project.
# ---------------------------------------------------------------------------
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes
 
 
def get_company_data_cached(ticker: str) -> dict:
    ticker = ticker.upper()
    now = time.time()
 
    if ticker in _CACHE:
        cached_at, data = _CACHE[ticker]
        if now - cached_at < _CACHE_TTL_SECONDS:
            return data
 
    try:
        data = fetch_company_data(ticker)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Couldn't fetch data for '{ticker}': {e}")
 
    _CACHE[ticker] = (now, data)
    return data
 
 
# ---------------------------------------------------------------------------
# Request schema — every assumption the frontend sliders can control
# ---------------------------------------------------------------------------
 
class DCFRequest(BaseModel):
    ticker: str
    growth_rate: float = 0.08
    projection_years: int = 5
    terminal_method: str = "gordon"          # "gordon" or "exit_multiple"
    terminal_growth: float = 0.025
    exit_multiple: float = 12.0
    cost_of_debt: Optional[float] = None     # None = use company's real calculated rate
    wacc_override: Optional[float] = None    # let user override auto-WACC
    multistage: bool = False
    high_growth_years: int = 3
    fade_years: int = 5
    buyback_rate: float = 0.0                # 0 = no buyback assumption
 
 
class SensitivityRequest(BaseModel):
    ticker: str
    growth_rate: float = 0.08
    projection_years: int = 5
    terminal_method: str = "gordon"
    multistage: bool = False
    high_growth_years: int = 3
    fade_years: int = 5
    wacc_min: float = 0.06
    wacc_max: float = 0.12
    wacc_steps: int = 13
    terminal_growth_min: float = 0.01
    terminal_growth_max: float = 0.04
    terminal_growth_steps: int = 7
 
 
class ImpliedGrowthRequest(BaseModel):
    ticker: str
    terminal_growth: float = 0.025
    projection_years: int = 5
    terminal_method: str = "gordon"
    exit_multiple: float = 12.0
    cost_of_debt: Optional[float] = None
    wacc_override: Optional[float] = None
    multistage: bool = False
    high_growth_years: int = 3
    fade_years: int = 5
 
 
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
 
@app.get("/")
def health_check():
    return {"status": "ok"}
 
 
@app.post("/api/dcf")
def calculate_dcf(req: DCFRequest):
    """Run a single DCF for a ticker with the given assumptions."""
    data = get_company_data_cached(req.ticker)
 
    try:
        result = run_dcf(
            data,
            growth_rate=req.growth_rate,
            projection_years=req.projection_years,
            terminal_method=req.terminal_method,
            terminal_growth=req.terminal_growth,
            exit_multiple=req.exit_multiple,
            cost_of_debt=req.cost_of_debt,
            wacc_override=req.wacc_override,
            multistage=req.multistage,
            high_growth_years=req.high_growth_years,
            fade_years=req.fade_years,
            buyback_rate=req.buyback_rate,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
 
    result["company_name"] = data["company_name"]
    return result
 
 
@app.post("/api/sensitivity")
def calculate_sensitivity(req: SensitivityRequest):
    """Return a WACC x terminal-growth grid of implied prices for a heatmap."""
    data = get_company_data_cached(req.ticker)
 
    grid = sensitivity_analysis(
        data,
        growth_rate=req.growth_rate,
        projection_years=req.projection_years,
        terminal_method=req.terminal_method,
        wacc_range=(req.wacc_min, req.wacc_max, req.wacc_steps),
        terminal_growth_range=(req.terminal_growth_min, req.terminal_growth_max, req.terminal_growth_steps),
        multistage=req.multistage,
        high_growth_years=req.high_growth_years,
        fade_years=req.fade_years,
    )
 
    # Convert the DataFrame into a plain JSON-friendly shape for the frontend
    return {
        "wacc_labels": grid.index.tolist(),
        "terminal_growth_labels": grid.columns.tolist(),
        "values": grid.values.tolist(),   # 2D array: values[wacc_index][growth_index]
    }
 
 
@app.post("/api/implied-growth")
def calculate_implied_growth(req: ImpliedGrowthRequest):
    """
    Reverse-DCF: what growth rate does the current market price already
    imply, given this WACC and terminal growth? Answers "what does the
    market believe" instead of "is this stock mispriced under my assumption."
    """
    data = get_company_data_cached(req.ticker)
 
    try:
        result = find_implied_growth_rate(
            data,
            wacc=req.wacc_override,
            terminal_growth=req.terminal_growth,
            projection_years=req.projection_years,
            terminal_method=req.terminal_method,
            exit_multiple=req.exit_multiple,
            cost_of_debt=req.cost_of_debt,
            multistage=req.multistage,
            high_growth_years=req.high_growth_years,
            fade_years=req.fade_years,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
 
    return result
 
