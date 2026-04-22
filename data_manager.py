# data_manager.py — P2-ETF-IDIOSYN-ALPHA-ENGINE
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from huggingface_hub import hf_hub_download

from config import (
    ALL_TICKERS, FI_TICKERS, EQUITY_TICKERS,
    BENCHMARKS, HF_MASTER_REPO, HF_MASTER_FILE,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Master data — price columns → log returns
# ─────────────────────────────────────────────────────────────────────────────

def load_master_data() -> pd.DataFrame:
    log.info(f"Loading master data from {HF_MASTER_REPO}")
    path = hf_hub_download(
        repo_id=HF_MASTER_REPO,
        filename=HF_MASTER_FILE,
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    log.info(f"Master data: {df.shape}, {df.index.min().date()} → {df.index.max().date()}")
    return df


def extract_log_returns(master: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Compute log returns from bare price columns (SPY, TLT, GLD, ...)."""
    available = [t for t in tickers if t in master.columns]
    if not available:
        raise ValueError(f"No tickers found in master_data columns: {list(master.columns)[:10]}")
    prices  = master[available].apply(pd.to_numeric, errors="coerce")
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    log.info(f"Log returns: {returns.shape}, tickers: {available}")
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# 2. Benchmark returns — AGG may not be in master_data, fetch via yfinance
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark_returns(
    master: pd.DataFrame,
    start: str = "2007-01-01",
) -> pd.DataFrame:
    """
    Build a DataFrame of daily log returns for SPY, AGG, GLD.
    SPY and GLD are in master_data (price columns).
    AGG may or may not be present — fetch from yfinance as fallback.
    """
    bench_returns = {}

    for b in BENCHMARKS:
        if b in master.columns:
            prices = master[b].apply(pd.to_numeric, errors="coerce")
            ret    = np.log(prices / prices.shift(1)).dropna()
            bench_returns[b] = ret
            log.info(f"  Benchmark {b}: loaded from master_data")
        else:
            log.info(f"  Benchmark {b}: not in master_data — fetching from yfinance")
            try:
                raw = yf.download(b, start=start, progress=False, auto_adjust=True)
                if raw.empty:
                    raise ValueError(f"yfinance returned empty for {b}")
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index)
                ret = np.log(close / close.shift(1)).dropna()
                ret.name = b
                bench_returns[b] = ret
                log.info(f"  Benchmark {b}: {len(ret)} rows from yfinance")
            except Exception as e:
                log.warning(f"  Benchmark {b} failed: {e} — filling with zeros")
                bench_returns[b] = pd.Series(0.0, index=master.index, name=b)

    bench_df = pd.DataFrame(bench_returns)
    bench_df.index = pd.to_datetime(bench_df.index)
    bench_df = bench_df.sort_index()
    log.info(f"Benchmark returns ready: {bench_df.shape}")
    return bench_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Align all return series on common dates
# ─────────────────────────────────────────────────────────────────────────────

def align_all(
    etf_returns: pd.DataFrame,
    bench_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join ETF returns and benchmark returns on common trading dates."""
    common = etf_returns.index.intersection(bench_returns.index)
    etf_a   = etf_returns.loc[common].dropna(how="all")
    bench_a = bench_returns.loc[common].dropna(how="all")
    # Keep only dates where all 3 benchmarks are present
    valid = bench_a[BENCHMARKS].notna().all(axis=1)
    etf_a   = etf_a.loc[valid]
    bench_a = bench_a.loc[valid]
    log.info(f"Aligned: {len(etf_a)} common dates")
    return etf_a, bench_a


# ─────────────────────────────────────────────────────────────────────────────
# 4. Universe helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_universe_tickers(universe: str) -> list[str]:
    if universe == "fi":
        return FI_TICKERS
    elif universe == "equity":
        return EQUITY_TICKERS
    return ALL_TICKERS
