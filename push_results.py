# push_results.py — P2-ETF-IDIOSYN-ALPHA-ENGINE
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from config import HF_RESULTS_REPO, HF_RESULTS_FILE

log = logging.getLogger(__name__)


def push_to_hf(df: pd.DataFrame, run_date: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN not set — skipping push")
        return

    api = HfApi(token=token)

    # Try to load and append existing results
    try:
        from huggingface_hub import hf_hub_download
        existing_path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=HF_RESULTS_FILE,
            repo_type="dataset",
            token=token,
        )
        existing = pd.read_parquet(existing_path)
        existing = existing[existing["run_date"] != run_date]
        combined = pd.concat([existing, df], ignore_index=True)
        # Keep last 252 run dates
        dates = sorted(combined["run_date"].unique())
        if len(dates) > 252:
            combined = combined[combined["run_date"].isin(dates[-252:])]
        log.info(f"Appended to existing: {combined.shape}")
    except Exception as e:
        log.info(f"First run (no existing results): {e}")
        combined = df

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / HF_RESULTS_FILE
        combined.to_parquet(out, index=False)
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=HF_RESULTS_FILE,
            repo_id=HF_RESULTS_REPO,
            repo_type="dataset",
            commit_message=f"Idiosyncratic alpha results — {run_date}",
        )
        log.info(f"Pushed to {HF_RESULTS_REPO}/{HF_RESULTS_FILE}")
