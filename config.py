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
USE_DCC_GARCH      = True
DCC_LOOKBACK_DAYS  = 252         # data window fed to DCC-GARCH
GARCH_P            = 1           # GARCH(p,q) order
GARCH_Q            = 1
DCC_MLE_X0         = [0.05, 0.93]       # MLE starting point (typical values)
DCC_MLE_BOUNDS     = [(1e-4, 0.4), (1e-4, 0.9999)]  # bounds for (a, b)
DCC_MLE_MAXITER    = 200

# ── Scoring ───────────────────────────────────────────────────────────────────
ALPHA_WINDOW_WEIGHTS = {63: 0.4, 126: 0.6}   # longer window weighted more
TOP_N_CONVICTION     = 5                       # top N in combined ranking for conviction filter
ANNUALISE_FACTOR     = 252                     # daily → annual

# ── Output columns ────────────────────────────────────────────────────────────
OUTPUT_COLS = [
    "run_date", "universe", "ticker",
    # Today's idiosyncratic signal
    "idio_return",          # raw residual ε today
    "idio_zscore",          # cross-sectional z-score of ε (main ranking signal 1)
    "idio_rank_pct",        # PERCENTILE rank by idio_return within universe (v2.0 - prevents tie churn)
    # Structural Jensen's alpha
    "jensen_alpha_63d",     # annualised OLS intercept, 63-day window
    "jensen_alpha_126d",    # annualised OLS intercept, 126-day window
    "jensen_alpha_combined",# weighted average of above (main ranking signal 2)
    "alpha_rank_pct",       # PERCENTILE rank by jensen_alpha_combined within universe (v2.0 - prevents tie churn)
    # Conviction (continuous soft-score intersection signal)
    "conviction",           # True if in top N of combined score
    "conviction_rank",      # rank among conviction ETFs (by combined score)
    "conviction_score",     # Continuous 0-1 weighted score of Idio + Alpha percentiles (v2.0)
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
MAX_RUNTIME_MINUTES = 300
