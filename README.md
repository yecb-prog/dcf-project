# Intrinsic Value Desk — Backend
 
A DCF (Discounted Cash Flow) equity valuation engine and API, built in Python with FastAPI. Given a ticker, it pulls live financial data, runs a full discounted cash flow model, and returns an intrinsic value estimate, a sensitivity grid, and a reverse-DCF "market-implied growth rate."
 
**Live API:** https://dcf-project.onrender.com
**Frontend / live demo:** https://dcf-frontend.vercel.app
**Frontend repo:** [dcf-frontend](https://github.com/yecb-prog/dcf-frontend)
 
## What it does
 
- Pulls real financial statements (income statement, balance sheet, cash flow) via `yfinance`
- Projects free cash flow forward, either at a flat growth rate or a **multi-stage** model (high growth fading to a terminal rate — avoids the unrealistic "cliff" of naive single-rate DCFs)
- Calculates WACC per company from real inputs: live beta, market cap, and **real cost of debt** (derived from the company's own reported interest expense ÷ total debt, not a flat assumption)
- Calculates real effective **tax rate** from the company's own income statement
- Computes enterprise value, equity value, and implied price per share
- Verdict logic: flags a stock as UNDERPRICED / OVERPRICED / FAIRLY PRICED based on a ±10% band vs. current market price
- **Buyback-adjusted view**: a secondary, clearly-labeled metric showing what per-share value looks like if a buyback assumption holds — kept separate from the core verdict, since buybacks don't change intrinsic value today, only how it's split per share over time
- **Sensitivity analysis**: a full WACC × terminal-growth grid (13×7 = 91 individual DCF runs), so you can see how fragile or robust the verdict is across a range of reasonable assumptions
- **Reverse-DCF**: instead of only asking "is this stock mispriced under my assumption," solves (via binary search) for the growth rate the current market price already implies — answers "what does the market believe," a more defensible framing than a single-point verdict
## Tech stack
 
Python · FastAPI · yfinance · pandas · numpy · curl_cffi (for reliable data fetching from cloud hosts)
 
## API endpoints
 
| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `POST /api/dcf` | Run a single DCF valuation for a ticker with given assumptions |
| `POST /api/sensitivity` | Return the WACC × terminal-growth sensitivity grid |
| `POST /api/implied-growth` | Reverse-DCF: solve for the market-implied growth rate |
 
Full request/response schemas are visible at `/docs` (FastAPI's auto-generated interactive API docs) when running locally.
 
## Running locally
 
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
 
Visit `http://127.0.0.1:8000/docs` to test endpoints interactively.
 
## Known limitations
 
This is a personal/portfolio project, and some simplifications are intentional trade-offs rather than oversights:
 
- **Beta** is used as-is from Yahoo Finance, rather than unlevering/relevering against a set of comparable companies the way professional equity research does
- **Cost of debt** is calculated from the most recent single year of reported interest expense, which can be skewed by one-off events (e.g. a company's effective tax rate can swing significantly year to year due to items like valuation allowance changes)
- **Growth rate and terminal growth are entirely user-supplied**, by design — the tool deliberately does not auto-guess a "correct" growth assumption, since the whole point is letting the user reason about and stress-test their own assumptions
- yfinance scrapes Yahoo Finance rather than using an official licensed API; `curl_cffi` is used to reduce (not eliminate) the risk of being rate-limited, particularly from cloud server IPs
## Why these design choices
 
A standard DCF tends to flag mature, cash-rich, buyback-heavy companies (Apple) and pure-growth/narrative stocks (Palantir) as overpriced, and safety/dividend-premium stocks (Coca-Cola, Clorox) as overpriced too — this is a well-documented, widely-discussed limitation of the method itself, not a bug in this implementation. The buyback-adjusted view and reverse-DCF feature were both added specifically to make those limitations visible and explainable, rather than presenting a single confident number as gospel.
 
