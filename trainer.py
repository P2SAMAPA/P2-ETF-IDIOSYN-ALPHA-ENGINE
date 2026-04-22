# trainer.py — P2-ETF-IDIOSYN-ALPHA-ENGINE
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    ALL_TICKERS, FI_TICKERS, EQUITY_TICKERS,
    OUTPUT_COLS, MAX_RUNTIME_MINUTES,
)
from data_manager import (
    load_master_data, extract_log_returns,
    load_benchmark_returns, align_all,
    get_universe_tickers,
)
from idiosyn_model import score_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

UNIVERSES = {
    "fi":       FI_TICKERS,
    "equity":   EQUITY_TICKERS,
    "combined": ALL_TICKERS,
}


def main():
    t0       = time.time()
    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    log.info(f"=== P2-ETF-IDIOSYN-ALPHA-ENGINE | Run date: {run_date} ===")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("Step 1/4: Loading data...")
    master       = load_master_data()
    etf_returns  = extract_log_returns(master, ALL_TICKERS)
    bench_returns = load_benchmark_returns(master)
    etf_r, bench_r = align_all(etf_returns, bench_returns)
    log.info(f"  Data ready: {etf_r.shape} ETF returns, "
             f"{bench_r.shape} benchmark returns, "
             f"latest: {etf_r.index.max().date()}")

    # ── 2. Score all universes ─────────────────────────────────────────────────
    log.info("Step 2/4: Scoring all universes...")
    all_results = []

    for universe, tickers in UNIVERSES.items():
        log.info(f"  Universe: {universe} ({len(tickers)} ETFs)")
        t_univ = time.time()
        df = score_universe(
            tickers=[t for t in tickers if t in etf_r.columns],
            etf_returns=etf_r,
            bench_returns=bench_r,
            run_date=run_date,
            universe=universe,
        )
        if not df.empty:
            all_results.append(df)
            log.info(f"  {universe} done in {time.time()-t_univ:.1f}s")

    if not all_results:
        log.error("No results produced — aborting")
        return

    # ── 3. Assemble final output ──────────────────────────────────────────────
    log.info("Step 3/4: Assembling output...")
    final = pd.concat(all_results, ignore_index=True)
    for col in OUTPUT_COLS:
        if col not in final.columns:
            final[col] = np.nan
    final = final[OUTPUT_COLS]
    log.info(f"  Final shape: {final.shape}")

    # ── 4. Push to HuggingFace ─────────────────────────────────────────────────
    log.info("Step 4/4: Pushing to HuggingFace...")
    from push_results import push_to_hf
    push_to_hf(final, run_date)

    elapsed = (time.time() - t0) / 60
    log.info(f"=== COMPLETE | {elapsed:.1f} min ===")
    if elapsed > MAX_RUNTIME_MINUTES:
        log.warning(f"Exceeded budget of {MAX_RUNTIME_MINUTES} min!")


if __name__ == "__main__":
    main()
