# idiosyn_model.py — P2-ETF-IDIOSYN-ALPHA-ENGINE
# Core engine:
#   1. DCC-GARCH time-varying betas (with OLS fallback)
#   2. Rolling OLS Jensen's alpha (63d + 126d)
#   3. Today's idiosyncratic return (ε = actual - systematic)
#   4. Cross-sectional z-score ranking
#   5. Conviction score — intersection of idio ranking + alpha ranking
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import (
    BENCHMARKS, OLS_WINDOWS, PRIMARY_WINDOW,
    USE_DCC_GARCH, DCC_LOOKBACK_DAYS,
    GARCH_P, GARCH_Q,
    ALPHA_WINDOW_WEIGHTS, ANNUALISE_FACTOR,
    TOP_N_CONVICTION,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DCC-GARCH time-varying beta estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_dcc_betas(
    etf_series: pd.Series,
    bench_df: pd.DataFrame,
    lookback: int = DCC_LOOKBACK_DAYS,
) -> dict[str, float]:
    """
    Fit a DCC-GARCH(1,1) model to estimate today's time-varying betas.
    Returns {benchmark: beta} for the most recent observation.

    Falls back to rolling OLS if arch is unavailable or fitting fails.
    The DCC approach:
      - Fit univariate GARCH(1,1) to each series for conditional variance
      - Apply DCC correlation update equations to get Σ_t
      - Beta_i = Cov(etf, bench_i) / Var(bench_i) from Σ_t
    """
    try:
        from arch import arch_model
    except ImportError:
        log.warning("arch library not available — falling back to OLS")
        return _ols_betas(etf_series, bench_df, PRIMARY_WINDOW)

    # Align and take lookback window
    common = etf_series.index.intersection(bench_df.index)
    etf_w  = etf_series.loc[common].iloc[-lookback:]
    bch_w  = bench_df[BENCHMARKS].loc[common].iloc[-lookback:]

    if len(etf_w) < 60:
        return _ols_betas(etf_series, bench_df, PRIMARY_WINDOW)

    try:
        # ── Step 1: fit univariate GARCH(1,1) to each series, get std residuals ──
        all_series = {f"etf": etf_w}
        for b in BENCHMARKS:
            all_series[b] = bch_w[b]

        std_resids = {}
        cond_vols  = {}
        for name, series in all_series.items():
            s = series.dropna() * 100  # scale to percent for numerical stability
            am = arch_model(s, vol="Garch", p=GARCH_P, q=GARCH_Q, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            std_resids[name] = res.std_resid.values
            cond_vols[name]  = res.conditional_volatility.values

        # Trim to common length
        min_len = min(len(v) for v in std_resids.values())
        for k in std_resids:
            std_resids[k] = std_resids[k][-min_len:]
            cond_vols[k]  = cond_vols[k][-min_len:]

        # ── Step 2: DCC correlation recursion ──────────────────────────────────
        # Q_bar = unconditional correlation of std residuals
        keys = ["etf"] + BENCHMARKS
        n    = len(keys)
        T    = min_len

        e_mat = np.column_stack([std_resids[k] for k in keys])  # T × n
        Q_bar = np.cov(e_mat.T)  # n × n unconditional

        dcc_a = 0.05
        dcc_b = 0.93
        Q_t   = Q_bar.copy()

        # Iterate DCC to get final Q_T
        for t in range(1, T):
            e_t = e_mat[t].reshape(-1, 1)
            Q_t = (1 - dcc_a - dcc_b) * Q_bar + dcc_a * (e_t @ e_t.T) + dcc_b * Q_t

        # Convert Q_T to correlation R_T
        D_inv = np.diag(1.0 / np.sqrt(np.diag(Q_t)))
        R_t   = D_inv @ Q_t @ D_inv

        # ── Step 3: conditional covariance Σ_T ────────────────────────────────
        d_t   = np.array([cond_vols[k][-1] for k in keys]) / 100  # back to decimal
        D_t   = np.diag(d_t)
        Sigma = D_t @ R_t @ D_t  # n × n conditional covariance

        # ── Step 4: betas = Cov(etf, bench_i) / Var(bench_i) ──────────────────
        etf_idx = 0
        betas   = {}
        for i, b in enumerate(BENCHMARKS):
            b_idx = keys.index(b)
            cov_eb = Sigma[etf_idx, b_idx]
            var_b  = Sigma[b_idx, b_idx]
            betas[b] = cov_eb / var_b if var_b > 1e-10 else 0.0

        log.debug(f"DCC betas: {betas}")
        return betas

    except Exception as exc:
        log.warning(f"DCC-GARCH failed ({exc}) — using OLS fallback")
        return _ols_betas(etf_series, bench_df, PRIMARY_WINDOW)


def _ols_betas(
    etf_series: pd.Series,
    bench_df: pd.DataFrame,
    window: int,
) -> dict[str, float]:
    """Simple rolling OLS beta (fallback)."""
    common = etf_series.index.intersection(bench_df.index)
    y = etf_series.loc[common].iloc[-window:].values
    X = bench_df[BENCHMARKS].loc[common].iloc[-window:].values
    mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    if mask.sum() < 20:
        return {b: 0.0 for b in BENCHMARKS}
    reg = LinearRegression(fit_intercept=True).fit(X[mask], y[mask])
    return dict(zip(BENCHMARKS, reg.coef_))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling OLS — Jensen's alpha + betas for multiple windows
# ─────────────────────────────────────────────────────────────────────────────

def rolling_ols_alpha(
    etf_series: pd.Series,
    bench_df: pd.DataFrame,
    windows: list[int] = OLS_WINDOWS,
) -> dict[int, dict]:
    """
    For each window, fit OLS on the most recent N days.
    Returns {window: {alpha_annualised, betas, r_squared}}.
    """
    results = {}
    common  = etf_series.index.intersection(bench_df.index)

    for w in windows:
        y = etf_series.loc[common].iloc[-w:].values
        X = bench_df[BENCHMARKS].loc[common].iloc[-w:].values
        mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))

        if mask.sum() < 15:
            results[w] = {"alpha": np.nan, "betas": {b: np.nan for b in BENCHMARKS}, "r2": np.nan}
            continue

        reg   = LinearRegression(fit_intercept=True).fit(X[mask], y[mask])
        y_hat = reg.predict(X[mask])
        ss_res = np.sum((y[mask] - y_hat) ** 2)
        ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
        r2    = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

        results[w] = {
            "alpha": reg.intercept_ * ANNUALISE_FACTOR,   # daily → annual
            "betas": dict(zip(BENCHMARKS, reg.coef_)),
            "r2":    r2,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Today's idiosyncratic return for one ETF
# ─────────────────────────────────────────────────────────────────────────────

def compute_idio_return(
    actual_return: float,
    betas: dict[str, float],
    bench_returns_today: dict[str, float],
) -> tuple[float, float]:
    """
    ε = r_etf - (β_SPY·r_SPY + β_AGG·r_AGG + β_GLD·r_GLD)
    Returns (systematic_return, idiosyncratic_return).
    """
    systematic = sum(betas.get(b, 0.0) * bench_returns_today.get(b, 0.0)
                     for b in BENCHMARKS)
    idio = actual_return - systematic
    return systematic, idio


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-sectional z-score
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    mean, std = series.mean(), series.std()
    if std < 1e-10 or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return (series - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full universe scoring for one date
# ─────────────────────────────────────────────────────────────────────────────

def score_universe(
    tickers: list[str],
    etf_returns: pd.DataFrame,
    bench_returns: pd.DataFrame,
    run_date: str,
    universe: str,
) -> pd.DataFrame:
    """
    For each ticker in the universe on the latest available date:
      - Estimate DCC-GARCH or OLS betas
      - Compute idiosyncratic return
      - Compute Jensen's alpha (63d + 126d)
      - Cross-sectional z-score all signals
      - Determine conviction (top N in both rankings)
    Returns a DataFrame of output rows.
    """
    # Latest date with data
    latest = etf_returns.index.max()
    bench_today = {b: bench_returns.loc[latest, b]
                   for b in BENCHMARKS if latest in bench_returns.index}

    rows = []
    for ticker in tickers:
        if ticker not in etf_returns.columns:
            continue
        etf_s = etf_returns[ticker].dropna()
        if len(etf_s) < 30:
            continue

        actual_ret = float(etf_s.iloc[-1]) if latest in etf_s.index else np.nan
        if np.isnan(actual_ret):
            continue

        # ── Betas: DCC-GARCH (or OLS fallback) ────────────────────────────────
        beta_method = "OLS-63d"
        if USE_DCC_GARCH:
            betas = estimate_dcc_betas(etf_s, bench_returns)
            beta_method = "DCC-GARCH"
        else:
            betas = _ols_betas(etf_s, bench_returns, PRIMARY_WINDOW)

        # ── Idiosyncratic return ───────────────────────────────────────────────
        sys_ret, idio_ret = compute_idio_return(actual_ret, betas, bench_today)

        # ── Rolling OLS alpha ──────────────────────────────────────────────────
        ols = rolling_ols_alpha(etf_s, bench_returns)
        a63  = ols.get(63,  {}).get("alpha", np.nan)
        a126 = ols.get(126, {}).get("alpha", np.nan)
        r2   = ols.get(63,  {}).get("r2",    np.nan)

        # Combined alpha: weighted average of 63d and 126d
        w63, w126 = ALPHA_WINDOW_WEIGHTS[63], ALPHA_WINDOW_WEIGHTS[126]
        if not np.isnan(a63) and not np.isnan(a126):
            alpha_combined = w63 * a63 + w126 * a126
        elif not np.isnan(a63):
            alpha_combined = a63
        elif not np.isnan(a126):
            alpha_combined = a126
        else:
            alpha_combined = np.nan

        row = {
            "run_date":            run_date,
            "universe":            universe,
            "ticker":              ticker,
            "idio_return":         round(idio_ret, 6),
            "idio_zscore":         np.nan,          # filled after cross-section
            "idio_rank":           np.nan,
            "jensen_alpha_63d":    round(a63,  4) if not np.isnan(a63)  else np.nan,
            "jensen_alpha_126d":   round(a126, 4) if not np.isnan(a126) else np.nan,
            "jensen_alpha_combined": round(alpha_combined, 4) if not np.isnan(alpha_combined) else np.nan,
            "alpha_rank":          np.nan,
            "conviction":          False,
            "conviction_rank":     np.nan,
            "beta_SPY":            round(betas.get("SPY", np.nan), 4),
            "beta_AGG":            round(betas.get("AGG", np.nan), 4),
            "beta_GLD":            round(betas.get("GLD", np.nan), 4),
            "r_squared_63d":       round(r2, 4) if not np.isnan(r2) else np.nan,
            "beta_method":         beta_method,
            "systematic_return":   round(sys_ret, 6),
            "actual_return":       round(actual_ret, 6),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Cross-sectional z-scores ───────────────────────────────────────────────
    valid_idio  = df["idio_return"].notna()
    valid_alpha = df["jensen_alpha_combined"].notna()

    df.loc[valid_idio,  "idio_zscore"] = cross_sectional_zscore(
        df.loc[valid_idio, "idio_return"]).values

    # ── Rankings ──────────────────────────────────────────────────────────────
    df = df.sort_values("idio_zscore", ascending=False)
    df["idio_rank"] = range(1, len(df) + 1)

    df = df.sort_values("jensen_alpha_combined", ascending=False)
    df["alpha_rank"] = range(1, len(df) + 1)

    # ── Conviction: top N in BOTH rankings ────────────────────────────────────
    top_idio  = set(df.nsmallest(TOP_N_CONVICTION, "idio_rank")["ticker"])
    top_alpha = set(df.nsmallest(TOP_N_CONVICTION, "alpha_rank")["ticker"])
    conviction_set = top_idio & top_alpha
    df["conviction"] = df["ticker"].isin(conviction_set)

    # Rank conviction ETFs by combined score (mean of both z-scores)
    df["_combined"] = (
        df["idio_zscore"].fillna(0) +
        cross_sectional_zscore(df["jensen_alpha_combined"].fillna(df["jensen_alpha_combined"].median()))
    )
    conv_df = df[df["conviction"]].sort_values("_combined", ascending=False)
    df.loc[conv_df.index, "conviction_rank"] = range(1, len(conv_df) + 1)
    df = df.drop(columns=["_combined"])

    log.info(f"  {universe}: {len(df)} ETFs scored, "
             f"{df['conviction'].sum()} conviction, "
             f"top idio: {df.nsmallest(1,'idio_rank')['ticker'].values}")
    return df
