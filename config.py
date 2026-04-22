# config.py — P2-ETF-IDIOSYN-ALPHA-ENGINE
# Dynamic Beta Hedging & Idiosyncratic Alpha Engine
# All constants in one place — edit here only.

# ── Universe ──────────────────────────────────────────────────────────────────
FI_TICKERS     = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
                   "XLY", "XLP", "XLU", "GDX", "XME", "IWM"]
ALL_TICKERS    = FI_TICKERS + EQUITY_TICKERS

# ── Three orthogonal benchmark factors ───────────────────────────────────────
# SPY = equity direction risk
# AGG = duration / interest rate risk  (sourced via yfinance — free)
# GLD = commodity / inflation / safe-haven risk
BENCHMARKS = ["SPY", "AGG", "GLD"]
BENCHMARK_LABELS = {
    "SPY": "Equity",
    "AGG": "Duration",
    "GLD": "Commodity",
}

# ── HuggingFace repos ─────────────────────────────────────────────────────────
HF_MASTER_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_MASTER_FILE   = "master_data.parquet"
HF_RESULTS_REPO  = "P2SAMAPA/p2-etf-idiosyn-alpha-engine-results"
HF_RESULTS_FILE  = "idiosyn_alpha_results.parquet"

# ── Rolling OLS windows (trading days) ───────────────────────────────────────
OLS_WINDOWS = [63, 126]          # short and medium — used for rolling beta + alpha
PRIMARY_WINDOW = 63              # window used for today's idiosyncratic return

# ── DCC-GARCH settings ────────────────────────────────────────────────────────
# DCC-GARCH gives time-varying betas that update daily.
# DCC parameters (a, b) are re-estimated via MLE on each daily run using
# scipy L-BFGS-B optimisation of the Engle (2002) DCC log-likelihood.
# This adds ~10-15 min to runtime but gives optimal parameters for the
# current 252-day window rather than using fixed typical values.
# Falls back to rolling OLS if arch fitting or MLE fails.
USE_DCC_GARCH      = True
DCC_LOOKBACK_DAYS  = 252         # data window fed to DCC-GARCH
GARCH_P            = 1           # GARCH(p,q) order
GARCH_Q            = 1
# DCC_A and DCC_B are no longer fixed — estimated daily via MLE.
DCC_MLE_X0         = [0.05, 0.93]       # MLE starting point (typical values)
DCC_MLE_BOUNDS     = [(1e-4, 0.4), (1e-4, 0.9999)]  # bounds for (a, b)
DCC_MLE_MAXITER    = 200

# ── Scoring ───────────────────────────────────────────────────────────────────
# Idiosyncratic return (ε): cross-sectional z-score of today's residual
# Jensen's alpha: annualised OLS intercept (63d and 126d, weighted average)
# Conviction score: ETFs appearing in top N of BOTH rankings simultaneously
ALPHA_WINDOW_WEIGHTS = {63: 0.4, 126: 0.6}   # longer window weighted more
TOP_N_CONVICTION     = 5                       # top N in each ranking for conviction filter
ANNUALISE_FACTOR     = 252                     # daily → annual

# ── Output columns ────────────────────────────────────────────────────────────
OUTPUT_COLS = [
    "run_date", "universe", "ticker",
    # Today's idiosyncratic signal
    "idio_return",          # raw residual ε today
    "idio_zscore",          # cross-sectional z-score of ε (main ranking signal 1)
    "idio_rank",            # rank by idio_zscore within universe
    # Structural Jensen's alpha
    "jensen_alpha_63d",     # annualised OLS intercept, 63-day window
    "jensen_alpha_126d",    # annualised OLS intercept, 126-day window
    "jensen_alpha_combined",# weighted average of above (main ranking signal 2)
    "alpha_rank",           # rank by jensen_alpha_combined within universe
    # Conviction (intersection signal)
    "conviction",           # True if in top N of BOTH rankings
    "conviction_rank",      # rank among conviction ETFs (by combined score)
    # Beta exposures (from DCC-GARCH or OLS fallback)
    "beta_SPY", "beta_AGG", "beta_GLD",
    # Model quality
    "r_squared_63d",        # OLS R² on 63-day window
    "beta_method",          # "DCC-GARCH" or "OLS-63d"
    # Systematic return (for reference)
    "systematic_return",    # beta_SPY*r_SPY + beta_AGG*r_AGG + beta_GLD*r_GLD
    "actual_return",        # raw ETF return today
]

# ── CPU budget ────────────────────────────────────────────────────────────────
# GitHub free tier: 2 vCPU, 7GB RAM, 6h limit.
# DCC-GARCH on 20 ETFs × 3 benchmarks: ~15-20 min.
# Total expected: ~25 min including data load + OLS + push.
MAX_RUNTIME_MINUTES = 300
