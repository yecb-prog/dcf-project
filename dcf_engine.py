"""
dcf_engine.py
 
A self-contained DCF valuation engine.
 
Pipeline:
  ticker -> fetch_company_data() -> run_dcf() -> verdict + implied price
                                  -> sensitivity_analysis() -> WACC x growth grid
 
Designed to be dropped straight into a Python API (FastAPI/Flask) as the
calculation layer. Each function returns plain dicts/DataFrames so it's
easy to json-serialize for a Next.js frontend.
 
Install: pip install yfinance pandas numpy curl_cffi
"""
 
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import yfinance as yf
 
# yfinance scrapes Yahoo Finance rather than using an official API, and
# Yahoo has tightened bot detection significantly -- plain requests from
# cloud server IPs (Render, AWS, etc.) get blocked/rate-limited far more
# than requests from a home connection. curl_cffi impersonates a real
# browser's TLS fingerprint, which is the current standard workaround.
#
# IMPORTANT: create a NEW session per request rather than one shared global
# session. Yahoo Finance uses a "crumb" security token tied to a session's
# cookies; FastAPI runs requests concurrently across threads, and sharing
# one session object across concurrent requests can corrupt that crumb,
# causing "Invalid Crumb" 401 errors. A fresh session per call avoids this.
try:
    from curl_cffi import requests as cffi_requests
    _CURL_CFFI_AVAILABLE = True
except ImportError:
    _CURL_CFFI_AVAILABLE = False
 
 
def _new_session():
    if _CURL_CFFI_AVAILABLE:
        return cffi_requests.Session(impersonate="chrome")
    return None  # falls back to yfinance's default session
 
 
# ---------------------------------------------------------------------------
# 1. DATA INGESTION
# ---------------------------------------------------------------------------
 
def _first_available(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """yfinance renames line items across versions, so try a list of aliases."""
    for name in candidates:
        if df is not None and name in df.index:
            return df.loc[name]
    return None
 
 
def fetch_company_data(ticker: str, _retries: int = 3) -> dict:
    """
    Pull everything needed for a DCF from Yahoo Finance, with automatic
    retry. Yahoo's "crumb" auth token occasionally fails on the first
    attempt (a widely-reported, ongoing issue with yfinance -- see
    github.com/ranaroussi/yfinance/issues -- not specific to this app) but
    frequently succeeds on a retry with a fresh session. A short delay
    between attempts gives Yahoo's session negotiation room to recover.
    """
    last_error = None
    for attempt in range(_retries):
        try:
            return _fetch_company_data_once(ticker)
        except Exception as e:
            last_error = e
            if attempt < _retries - 1:
                time.sleep(1.5)
    raise last_error
 
 
def _fetch_company_data_once(ticker: str) -> dict:
    session = _new_session()
    t = yf.Ticker(ticker, session=session) if session else yf.Ticker(ticker)
    info = t.info or {}
 
    income = t.financials          # annual income statement (most recent col first)
    balance = t.balance_sheet
    cashflow = t.cashflow
 
    # --- historical free cash flow (most recent 3-4 years, oldest -> newest) ---
    op_cf = _first_available(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = _first_available(cashflow, ["Capital Expenditure", "Capital Expenditures"])
 
    if op_cf is None or capex is None:
        raise ValueError(f"Could not find cash flow line items for {ticker}")
 
    fcf_hist = (op_cf + capex).dropna()  # capex is already negative in yfinance
    fcf_hist = fcf_hist.iloc[::-1]        # oldest -> newest
 
    # --- historical revenue CAGR: a real, ticker-specific anchor point for
    # the growth rate slider, so the person isn't guessing blind against a
    # generic default. This is shown as a hint, never used to auto-set the
    # slider -- the growth assumption stays entirely user-controlled. ---
    revenue = _first_available(income, ["Total Revenue", "Operating Revenue"])
    revenue_cagr = None
    try:
        if revenue is not None:
            rev_hist = revenue.dropna().iloc[::-1]  # oldest -> newest
            if len(rev_hist) >= 2:
                oldest = float(rev_hist.iloc[0])
                newest = float(rev_hist.iloc[-1])
                years = len(rev_hist) - 1
                if oldest > 0 and newest > 0:
                    revenue_cagr = (newest / oldest) ** (1 / years) - 1
    except Exception:
        pass  # leave as None; frontend just won't show the hint
 
    ebit = _first_available(income, ["EBIT", "Operating Income"])
    tax_provision = _first_available(income, ["Tax Provision", "Income Tax Expense"])
    pretax_income = _first_available(income, ["Pretax Income", "Income Before Tax"])
 
    # effective tax rate from most recent year, fallback to 21% (US statutory)
    try:
        tax_rate = float(tax_provision.iloc[0] / pretax_income.iloc[0])
        if not (0 <= tax_rate <= 0.6):
            tax_rate = 0.21
    except Exception:
        tax_rate = 0.21
 
    total_debt = _first_available(balance, ["Total Debt"])
    cash = _first_available(balance, ["Cash And Cash Equivalents", "Cash"])
 
    # --- real cost of debt: effective interest rate the company actually
    # pays, derived from its own income statement, instead of a flat
    # assumed number applied to every company regardless of credit quality ---
    interest_expense = _first_available(
        income, ["Interest Expense", "Interest Expense Non Operating", "Net Interest Income"]
    )
    total_debt_value = float(total_debt.iloc[0]) if total_debt is not None else 0.0
 
    cost_of_debt = 0.05  # fallback: used if data is missing/unusable
    try:
        if interest_expense is not None and total_debt_value > 0:
            implied_rate = abs(float(interest_expense.iloc[0])) / total_debt_value
            # sanity bounds: real corporate borrowing rates don't sit outside
            # roughly 1%-15% in practice, so clamp rather than trust a wild
            # number caused by messy/incomplete reported data
            if 0.01 <= implied_rate <= 0.15:
                cost_of_debt = implied_rate
    except Exception:
        pass  # keep the 5% fallback
 
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares_outstanding = info.get("sharesOutstanding")
 
    # market_cap sometimes comes back missing from yfinance's .info (more
    # common from cloud/server IPs than from a home connection). Fall back
    # to computing it ourselves -- price x shares -- rather than crashing
    # downstream in calculate_wacc().
    market_cap = info.get("marketCap")
    if market_cap is None and current_price is not None and shares_outstanding is not None:
        market_cap = current_price * shares_outstanding
 
    if current_price is None or shares_outstanding is None or market_cap is None:
        raise ValueError(
            f"Missing critical price/share data for {ticker} -- yfinance may be "
            f"rate-limiting or this ticker may not have complete data available."
        )
 
    data = {
        "ticker": ticker.upper(),
        "company_name": info.get("shortName", ticker),
        "current_price": current_price,
        "shares_outstanding": shares_outstanding,
        "market_cap": market_cap,
        "beta": info.get("beta") or 1.0,
        "total_debt": total_debt_value,
        "cash": float(cash.iloc[0]) if cash is not None else 0.0,
        "tax_rate": tax_rate,
        "cost_of_debt": cost_of_debt,
        "fcf_history": fcf_hist.astype(float).tolist(),   # oldest -> newest
        "latest_ebitda": info.get("ebitda"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "exchange": info.get("exchange"),
        "pe_ratio": info.get("trailingPE"),
        "revenue_cagr": revenue_cagr,
    }
    return data
 
 
# ---------------------------------------------------------------------------
# 2. WACC
# ---------------------------------------------------------------------------
 
def calculate_wacc(
    market_cap: float,
    total_debt: float,
    cost_of_debt: float,
    tax_rate: float,
    beta: float,
    risk_free_rate: float = 0.042,   # ~10Y Treasury, update as needed
    equity_risk_premium: float = 0.045,
) -> float:
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    total_value = market_cap + total_debt
    if total_value == 0:
        return cost_of_equity
 
    e_weight = market_cap / total_value
    d_weight = total_debt / total_value
 
    wacc = (e_weight * cost_of_equity) + (d_weight * cost_of_debt * (1 - tax_rate))
    return wacc
 
 
# ---------------------------------------------------------------------------
# 3. FCF PROJECTION + TERMINAL VALUE
# ---------------------------------------------------------------------------
 
def project_fcf(last_fcf: float, growth_rate: float, years: int = 5) -> list[float]:
    return [last_fcf * (1 + growth_rate) ** i for i in range(1, years + 1)]
 
 
def project_fcf_multistage(
    last_fcf: float,
    high_growth_rate: float,
    high_growth_years: int,
    fade_years: int,
    terminal_growth: float,
) -> list[float]:
    """
    Three-phase growth, the standard way analysts model mature companies:
 
      Phase 1 (high_growth_years):  constant high_growth_rate
      Phase 2 (fade_years):         growth rate glides linearly down to terminal_growth
      Phase 3:                      handled separately by terminal_value() at terminal_growth
 
    Returns the projected FCF for phase 1 + phase 2 combined (oldest -> newest).
    This avoids the common naive-DCF trap of assuming a high growth rate holds
    constant right up until the terminal value kicks in.
    """
    fcf_list = []
    fcf = last_fcf
 
    # Phase 1: high growth, flat rate
    for _ in range(high_growth_years):
        fcf = fcf * (1 + high_growth_rate)
        fcf_list.append(fcf)
 
    # Phase 2: linear fade from high_growth_rate down to terminal_growth
    if fade_years > 0:
        for step in range(1, fade_years + 1):
            fade_rate = high_growth_rate + (terminal_growth - high_growth_rate) * (step / fade_years)
            fcf = fcf * (1 + fade_rate)
            fcf_list.append(fcf)
 
    return fcf_list
 
 
def terminal_value(
    final_year_fcf: float,
    wacc: float,
    method: str = "gordon",
    terminal_growth: float = 0.025,
    exit_multiple: float = 12.0,
    final_ebitda: float | None = None,
) -> float:
    if method == "gordon":
        if wacc <= terminal_growth:
            raise ValueError("WACC must exceed terminal growth rate for Gordon Growth.")
        return final_year_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    elif method == "exit_multiple":
        if final_ebitda is None:
            raise ValueError("final_ebitda required for exit_multiple method.")
        return final_ebitda * exit_multiple
    else:
        raise ValueError("method must be 'gordon' or 'exit_multiple'")
 
 
# ---------------------------------------------------------------------------
# 4. FULL DCF RUN
# ---------------------------------------------------------------------------
 
def project_shares_outstanding(current_shares: float, buyback_rate: float, years: int) -> list[float]:
    """Shares outstanding declining each year due to net buybacks (oldest -> newest)."""
    return [current_shares * (1 - buyback_rate) ** i for i in range(1, years + 1)]
 
 
def find_implied_growth_rate(
    data: dict,
    wacc: float | None = None,
    terminal_growth: float = 0.025,
    projection_years: int = 5,
    terminal_method: str = "gordon",
    exit_multiple: float = 12.0,
    cost_of_debt: float | None = None,
    multistage: bool = False,
    high_growth_years: int = 3,
    fade_years: int = 5,
    tolerance: float = 0.01,
    max_iterations: int = 100,
) -> dict:
    """
    Reverse-DCF: instead of assuming a growth rate and checking whether the
    stock looks over/underpriced, solve for the growth rate that makes the
    DCF's implied price exactly equal today's market price.
 
    This reframes the question from "is this stock mispriced under MY
    assumption" to "what growth rate is the market currently assuming" —
    a more defensible way to present a verdict, since the person reading it
    judges the plausibility of that growth number themselves.
 
    Uses binary search over the growth rate, holding WACC and terminal
    growth fixed. Assumes higher growth -> higher implied price, which
    holds for companies with positive free cash flow (the normal case).
    """
    wacc_value = wacc or calculate_wacc(
        market_cap=data["market_cap"],
        total_debt=data["total_debt"],
        cost_of_debt=cost_of_debt if cost_of_debt is not None else data["cost_of_debt"],
        tax_rate=data["tax_rate"],
        beta=data["beta"],
    )
    current_price = data["current_price"]
 
    def price_for_growth(g: float) -> float:
        result = run_dcf(
            data,
            growth_rate=g,
            wacc_override=wacc_value,
            terminal_growth=terminal_growth,
            projection_years=projection_years,
            terminal_method=terminal_method,
            exit_multiple=exit_multiple,
            cost_of_debt=cost_of_debt,
            multistage=multistage,
            high_growth_years=high_growth_years,
            fade_years=fade_years,
        )
        return result["implied_price_per_share"]
 
    lo, hi = -0.5, 1.0  # search growth rates from -50% to +100%
    mid = 0.0
    converged = False
 
    for i in range(max_iterations):
        mid = (lo + hi) / 2
        price = price_for_growth(mid)
 
        if abs(price - current_price) < tolerance:
            converged = True
            break
        if price > current_price:
            hi = mid
        else:
            lo = mid
 
    return {
        "ticker": data["ticker"],
        "company_name": data["company_name"],
        "current_price": current_price,
        "implied_growth_rate": mid,
        "wacc_used": wacc_value,
        "converged": converged,
    }
 
 
def run_dcf(
    data: dict,
    growth_rate: float = 0.08,
    projection_years: int = 5,
    terminal_method: str = "gordon",
    terminal_growth: float = 0.025,
    exit_multiple: float = 12.0,
    cost_of_debt: float | None = None,
    wacc_override: float | None = None,
    multistage: bool = False,
    high_growth_years: int = 3,
    fade_years: int = 5,
    buyback_rate: float = 0.0,
) -> dict:
    """
    Run a full DCF given company data from fetch_company_data().
 
    cost_of_debt: None (default) uses the company's own real effective
    interest rate, calculated from its income statement in
    fetch_company_data(). Pass a number to override with your own assumption.
 
    Two modes:
      multistage=False (default): flat `growth_rate` for `projection_years`.
      multistage=True: `growth_rate` acts as the high-growth rate for
        `high_growth_years`, then fades linearly to `terminal_growth` over
        `fade_years` before the terminal value kicks in. Total projection
        length = high_growth_years + fade_years.
    """
    last_fcf = data["fcf_history"][-1]
    resolved_cost_of_debt = cost_of_debt if cost_of_debt is not None else data["cost_of_debt"]
 
    wacc = wacc_override or calculate_wacc(
        market_cap=data["market_cap"],
        total_debt=data["total_debt"],
        cost_of_debt=resolved_cost_of_debt,
        tax_rate=data["tax_rate"],
        beta=data["beta"],
    )
 
    if multistage:
        projected = project_fcf_multistage(
            last_fcf,
            high_growth_rate=growth_rate,
            high_growth_years=high_growth_years,
            fade_years=fade_years,
            terminal_growth=terminal_growth,
        )
    else:
        projected = project_fcf(last_fcf, growth_rate, projection_years)
 
    pv_fcf = [cf / (1 + wacc) ** (i + 1) for i, cf in enumerate(projected)]
    total_years = len(projected)
 
    tv = terminal_value(
        final_year_fcf=projected[-1],
        wacc=wacc,
        method=terminal_method,
        terminal_growth=terminal_growth,
        exit_multiple=exit_multiple,
        final_ebitda=data.get("latest_ebitda"),
    )
    pv_tv = tv / (1 + wacc) ** total_years
 
    enterprise_value = sum(pv_fcf) + pv_tv
    equity_value = enterprise_value - data["total_debt"] + data["cash"]
 
    # Standard DCF answer: value TODAY, divided by TODAY's share count.
    # This is the theoretically correct number — buybacks don't change it,
    # because FCF already accounts for the cash available to fund them.
    implied_price = equity_value / data["shares_outstanding"]
 
    # Buyback-adjusted view (a common practitioner heuristic, not textbook
    # DCF): if the company keeps retiring `buyback_rate`% of shares per
    # year, the SAME total equity value gets split across a smaller share
    # count by the end of the projection window. This estimates what your
    # per-share value looks like if you hold through that share shrinkage.
    if buyback_rate > 0:
        future_shares = data["shares_outstanding"] * (1 - buyback_rate) ** total_years
        implied_price_buyback_adjusted = equity_value / future_shares
    else:
        implied_price_buyback_adjusted = implied_price
 
    current_price = data["current_price"]
    upside_pct = (implied_price - current_price) / current_price
 
    if upside_pct > 0.10:
        verdict = "UNDERPRICED"
    elif upside_pct < -0.10:
        verdict = "OVERPRICED"
    else:
        verdict = "FAIRLY PRICED"
 
    return {
        "ticker": data["ticker"],
        "wacc": wacc,
        "cost_of_debt_used": resolved_cost_of_debt,
        "tax_rate": data["tax_rate"],
        "sector": data.get("sector"),
        "industry": data.get("industry"),
        "exchange": data.get("exchange"),
        "pe_ratio": data.get("pe_ratio"),
        "market_cap": data.get("market_cap"),
        "revenue_cagr": data.get("revenue_cagr"),
        "multistage": multistage,
        "projection_years": total_years,
        "projected_fcf": projected,
        "pv_fcf": pv_fcf,
        "terminal_value": tv,
        "pv_terminal_value": pv_tv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "implied_price_per_share": implied_price,
        "implied_price_buyback_adjusted": implied_price_buyback_adjusted,
        "buyback_rate": buyback_rate,
        "current_price": current_price,
        "upside_pct": upside_pct,
        "verdict": verdict,
    }
 
 
# ---------------------------------------------------------------------------
# 5. SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------------
 
def sensitivity_analysis(
    data: dict,
    growth_rate: float = 0.08,
    projection_years: int = 5,
    terminal_method: str = "gordon",
    wacc_range: tuple[float, float, int] = (0.06, 0.12, 13),   # (min, max, steps)
    terminal_growth_range: tuple[float, float, int] = (0.01, 0.04, 7),
    exit_multiple: float = 12.0,
    multistage: bool = False,
    high_growth_years: int = 3,
    fade_years: int = 5,
) -> pd.DataFrame:
    """
    Returns a DataFrame: rows = WACC values, columns = terminal growth values,
    cells = implied price per share. Perfect for a heatmap on the frontend.
    """
    wacc_values = np.linspace(*wacc_range)
    growth_values = np.linspace(*terminal_growth_range)
 
    grid = pd.DataFrame(index=[f"{w:.1%}" for w in wacc_values],
                         columns=[f"{g:.1%}" for g in growth_values], dtype=float)
 
    for w in wacc_values:
        for g in growth_values:
            try:
                result = run_dcf(
                    data,
                    growth_rate=growth_rate,
                    projection_years=projection_years,
                    terminal_method=terminal_method,
                    terminal_growth=g,
                    exit_multiple=exit_multiple,
                    wacc_override=w,
                    multistage=multistage,
                    high_growth_years=high_growth_years,
                    fade_years=fade_years,
                )
                grid.loc[f"{w:.1%}", f"{g:.1%}"] = result["implied_price_per_share"]
            except ValueError:
                grid.loc[f"{w:.1%}", f"{g:.1%}"] = np.nan
 
    grid.index.name = "WACC"
    grid.columns.name = "Terminal Growth"
    return grid
 
 
# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------
 
def print_dcf(result: dict) -> None:
    label = "MULTI-STAGE" if result["multistage"] else "FLAT GROWTH"
    print(f"\n--- {result['ticker']} DCF ({label}, {result['projection_years']}yr) ---")
    print(f"WACC:                        {result['wacc']:.2%}")
    print(f"Enterprise Value:            ${result['enterprise_value']:,.0f}")
    print(f"Equity Value:                ${result['equity_value']:,.0f}")
    print(f"Implied Price/Share:         ${result['implied_price_per_share']:.2f}   <- honest DCF answer")
    if result["buyback_rate"] > 0:
        print(f"Buyback-Adjusted Price:      ${result['implied_price_buyback_adjusted']:.2f}   "
              f"<- if {result['buyback_rate']:.1%}/yr buybacks hold, {result['projection_years']}yr out")
    print(f"Current Price:               ${result['current_price']:.2f}")
    print(f"Upside/Downside:             {result['upside_pct']:.1%}")
    print(f"Verdict:                     {result['verdict']}")
 
 
if __name__ == "__main__":
    ticker = "AAPL"
    company = fetch_company_data(ticker)
 
    # Old approach: flat 8% growth for 5 years straight, then terminal value
    flat_result = run_dcf(company, growth_rate=0.08, projection_years=5)
    print_dcf(flat_result)
 
    # Multi-stage: 12% growth for 3 years, fading to 2.5% terminal growth
    # over the next 5 years, PLUS a 3%/yr buyback assumption (Apple has
    # historically retired roughly this much of its float annually)
    multistage_result = run_dcf(
        company,
        growth_rate=0.12,
        terminal_growth=0.025,
        multistage=True,
        high_growth_years=3,
        fade_years=5,
        buyback_rate=0.03,
    )
    print_dcf(multistage_result)
 
    print("\n--- Sensitivity grid (multi-stage, implied price per share) ---")
    grid = sensitivity_analysis(
        company,
        growth_rate=0.12,
        multistage=True,
        high_growth_years=3,
        fade_years=5,
    )
    print(grid.round(2))
 
