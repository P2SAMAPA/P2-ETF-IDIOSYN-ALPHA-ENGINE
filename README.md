# P2-ETF-IDIOSYN-ALPHA-ENGINE

**Dynamic Beta Hedging & Idiosyncratic Alpha — Ranking ETFs by the Return That Cannot Be Explained by Market, Duration, or Commodity Exposure**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-IDIOSYN-ALPHA-ENGINE/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-IDIOSYN-ALPHA-ENGINE/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-p2--etf--idiosyn--alpha--engine--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-idiosyn-alpha-engine-results)

---

## Overview

Most return-based engines are contaminated by **systematic beta** — on a strong up-day, high-beta ETFs rank first simply because they amplify the market. This engine strips that noise by estimating each ETF's time-varying exposure to three orthogonal risk benchmarks and ranking on what remains.

Three output signals:

| Signal | What it measures | How to use it |
|---|---|---|
| **Idiosyncratic return (ε)** | Today's residual after removing beta × benchmark | Trading signal — what's outperforming right now |
| **Jensen's alpha (α)** | Rolling OLS intercept annualised, 63d + 126d | Structural signal — persistent edge above factor exposure |
| **Conviction** | Top 5 in *both* rankings simultaneously | Highest confidence — genuine alpha + active outperformance today |

---

## Methodology

### Step 1 — Three orthogonal benchmarks

```
SPY = equity direction risk
AGG = duration / interest rate risk
GLD = commodity / inflation / safe-haven risk
```

These three are largely uncorrelated and together explain ~70–85% of most ETF return variance.

### Step 2 — DCC-GARCH(1,1) time-varying betas

```
r_etf(t) = α + β_SPY(t)·r_SPY(t) + β_AGG(t)·r_AGG(t) + β_GLD(t)·r_GLD(t) + ε(t)
```

**DCC-GARCH** (Dynamic Conditional Correlation GARCH) updates betas daily using exponential weighting on recent observations. During stress regimes, betas can change dramatically intra-month — rolling OLS misses this. Falls back to rolling OLS if fitting fails.

### Step 3 — Idiosyncratic return

```
ε(t) = r_etf(t) - [ β_SPY(t)·r_SPY(t) + β_AGG(t)·r_AGG(t) + β_GLD(t)·r_GLD(t) ]
```

### Step 4 — Jensen's alpha

Rolling OLS intercept estimated separately on 63-day and 126-day windows, annualised (×252). Combined: 40% × α_63d + 60% × α_126d.

### Step 5 — Cross-sectional z-score and conviction

Idiosyncratic returns and alpha are each z-scored cross-sectionally within the universe. Conviction ETFs = top 5 in both rankings simultaneously.

---

## Output schema

Results pushed daily to `P2SAMAPA/p2-etf-idiosyn-alpha-engine-results`:

| Column | Description |
|---|---|
| `run_date` | Date (YYYY-MM-DD) |
| `universe` | `fi`, `equity`, or `combined` |
| `ticker` | ETF ticker |
| `idio_return` | Raw residual ε today |
| `idio_zscore` | Cross-sectional z-score of ε |
| `idio_rank` | Rank by idio_zscore (1 = highest) |
| `jensen_alpha_63d` | Annualised OLS intercept, 63d window |
| `jensen_alpha_126d` | Annualised OLS intercept, 126d window |
| `jensen_alpha_combined` | Weighted average (40% × 63d + 60% × 126d) |
| `alpha_rank` | Rank by alpha_combined (1 = highest) |
| `conviction` | True if in top 5 of both rankings |
| `conviction_rank` | Rank among conviction ETFs |
| `beta_SPY` / `beta_AGG` / `beta_GLD` | Time-varying DCC-GARCH betas |
| `r_squared_63d` | OLS R² on 63-day window |
| `beta_method` | `DCC-GARCH` or `OLS-63d` |
| `systematic_return` | β·r_benchmark (what was explained) |
| `actual_return` | Raw ETF log return today |

---

## CPU budget

| Step | Time |
|---|---|
| Data load (HF + yfinance AGG) | ~2 min |
| DCC-GARCH (20 ETFs × 252d) | ~15–20 min |
| Rolling OLS (63d + 126d) | ~2 min |
| Scoring + push | ~1 min |
| **Total** | **~20–25 min** |

Well within the 6-hour GitHub Actions free tier limit.

---

## Setup

Add `HF_TOKEN` to GitHub repo secrets (`Settings → Secrets → Actions`).

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
```

---

## File structure

```
P2-ETF-IDIOSYN-ALPHA-ENGINE/
├── config.py           # All constants
├── data_manager.py     # HF master data + yfinance AGG benchmark
├── idiosyn_model.py    # DCC-GARCH betas, OLS alpha, idio return, conviction scoring
├── trainer.py          # Pipeline orchestrator
├── push_results.py     # HuggingFace push
├── streamlit_app.py    # Dashboard — three-column hero + universe tabs
├── us_calendar.py      # NYSE calendar
├── requirements.txt
└── .github/workflows/daily_run.yml
```
