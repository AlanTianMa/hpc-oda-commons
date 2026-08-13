#!/usr/bin/env python3
"""
MoE XGBoost vs Flat XGBoost — production settings comparison.

Runs both models on Kestrel with production SVD/OHE and logs results.
Checkpoints after each model so a crash doesn't lose the first result.

Usage:
    python docs/benchmarking/run_moe_vs_flat.py
    python docs/benchmarking/run_moe_vs_flat.py --resume

Output:
    workspace/moe_vs_flat_checkpoint.json
    workspace/moe_vs_flat.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "workspace" / "data" / "datasets" / "nlr_kestrel" / "data.parquet"
CHECKPOINT_PATH = REPO_ROOT / "workspace" / "moe_vs_flat_checkpoint.json"
LOG_PATH = REPO_ROOT / "workspace" / "moe_vs_flat.log"

sys.path.insert(0, str(REPO_ROOT / "src"))

from hpc_oda_commons.models.job_runtime_xgboost.model import (
    JobRuntimeXGBoostConfig,
    JobRuntimeXGBoostModel,
)
from hpc_oda_commons.models.experimental.moe_xgboost_model import (
    MoEXGBoostConfig,
    MoEXGBoostModel,
)

# ---------------------------------------------------------------------------
# Configuration — production defaults
# ---------------------------------------------------------------------------
N_WINDOWS = 120
TEST_WINDOW_HOURS = 6
TRAINING_LOOKBACK_DAYS = 120
MAX_SVD = 256
MAX_OHE = 2048
ESTIMATOR_N_JOBS = 12


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("moe_vs_flat")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def save_checkpoint(results: dict, status: str = "running") -> None:
    data = {"status": status, "timestamp": datetime.now(timezone.utc).isoformat(), "results": results}
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(CHECKPOINT_PATH)


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text()).get("results", {})
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE vs Flat XGBoost comparison")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 70)
    log.info("MoE XGBoost vs Flat XGBoost — Production Settings")
    log.info("SVD=%d, OHE=%d, windows=%d, lookback=%dd, n_jobs=%d",
             MAX_SVD, MAX_OHE, N_WINDOWS, TRAINING_LOOKBACK_DAYS, ESTIMATOR_N_JOBS)
    log.info("=" * 70)

    # --- Load data ---
    log.info("Loading data...")
    table = pq.read_table(DATA_PATH)
    lo = datetime(2025, 1, 1, tzinfo=timezone.utc)
    hi = datetime(2025, 6, 26, tzinfo=timezone.utc) + timedelta(days=1)
    sc = table.column("submit_time")
    ec = table.column("end_time")
    mask = pc.and_(
        pc.less(sc, pa.scalar(hi, type=sc.type)),
        pc.greater_equal(ec, pa.scalar(lo, type=ec.type)),
    )
    df = table.filter(mask).to_pandas()
    rows = df.to_dict("records")
    log.info("Loaded: %s rows, span: %s days",
             f"{len(rows):,}", (df["submit_time"].max() - df["submit_time"].min()).days)

    # --- Load checkpoint ---
    results = load_checkpoint() if args.resume else {}
    if args.resume:
        log.info("Resuming: %d models already completed", len(results))

    # --- Run Flat XGBoost ---
    if "flat_xgboost" not in results:
        log.info("")
        log.info("=" * 70)
        log.info("Running: Flat XGBoost (no MoE, no decay)")
        log.info("=" * 70)

        flat_config = JobRuntimeXGBoostConfig(
            n_windows=N_WINDOWS,
            test_window_hours=TEST_WINDOW_HOURS,
            training_lookback_days=TRAINING_LOOKBACK_DAYS,
            max_svd_components=MAX_SVD,
            target_max_one_hot_width=MAX_OHE,
        )
        flat_model = JobRuntimeXGBoostModel(flat_config)

        flat_start = time.time()
        flat_result = flat_model.evaluate(rows)
        flat_time = (time.time() - flat_start) / 60

        results["flat_xgboost"] = {
            "mae": flat_result["mae"],
            "rmse": flat_result["rmse"],
            "scored": flat_result["summary"]["rows_scored"],
            "time_min": round(flat_time, 1),
        }

        log.info("Flat XGBoost: MAE=%s s, RMSE=%s s, scored=%s, time=%.0fmin",
                 f"{flat_result['mae']:,.0f}", f"{flat_result['rmse']:,.0f}",
                 f"{flat_result['summary']['rows_scored']:,}", flat_time)

        save_checkpoint(results)
        log.info("Checkpoint saved (flat complete)")
    else:
        log.info("Flat XGBoost: already completed (MAE=%s s)", f"{results['flat_xgboost']['mae']:,.0f}")

    # --- Run MoE XGBoost ---
    if "moe_xgboost" not in results:
        log.info("")
        log.info("=" * 70)
        log.info("Running: MoE XGBoost (user+WC routing, decay=0.05)")
        log.info("=" * 70)

        moe_config = MoEXGBoostConfig(
            n_windows=N_WINDOWS,
            test_window_hours=TEST_WINDOW_HOURS,
            training_lookback_days=TRAINING_LOOKBACK_DAYS,
            max_svd_components=MAX_SVD,
            target_max_one_hot_width=MAX_OHE,
            time_decay_rate=0.05,
            estimator_n_jobs=ESTIMATOR_N_JOBS,
        )
        moe_model = MoEXGBoostModel(moe_config)

        moe_start = time.time()
        moe_result = moe_model.evaluate(rows, verbose=True)
        moe_time = (time.time() - moe_start) / 60

        # Store per-bin details
        bin_details = moe_result["summary"].get("bin_details", [])

        results["moe_xgboost"] = {
            "mae": moe_result["mae"],
            "rmse": moe_result["rmse"],
            "scored": moe_result["summary"]["rows_scored"],
            "time_min": round(moe_time, 1),
            "bins": bin_details,
        }

        log.info("MoE XGBoost: MAE=%s s, RMSE=%s s, scored=%s, time=%.0fmin",
                 f"{moe_result['mae']:,.0f}", f"{moe_result['rmse']:,.0f}",
                 f"{moe_result['summary']['rows_scored']:,}", moe_time)

        save_checkpoint(results)
        log.info("Checkpoint saved (MoE complete)")
    else:
        log.info("MoE XGBoost: already completed (MAE=%s s)", f"{results['moe_xgboost']['mae']:,.0f}")

    # --- Summary ---
    log.info("")
    log.info("=" * 70)
    log.info("COMPARISON RESULTS")
    log.info("=" * 70)

    flat = results["flat_xgboost"]
    moe = results["moe_xgboost"]

    log.info("")
    log.info("%-30s %10s %10s %10s %10s", "Model", "MAE", "RMSE", "Scored", "Time")
    log.info("-" * 75)
    log.info("%-30s %10s %10s %10s %9.0fmin",
             "Flat XGBoost", f"{flat['mae']:,.0f}s", f"{flat['rmse']:,.0f}s",
             f"{flat['scored']:,}", flat["time_min"])
    log.info("%-30s %10s %10s %10s %9.0fmin",
             "MoE XGBoost (decay=0.05)", f"{moe['mae']:,.0f}s", f"{moe['rmse']:,.0f}s",
             f"{moe['scored']:,}", moe["time_min"])

    mae_imp = (moe["mae"] - flat["mae"]) / flat["mae"] * 100
    rmse_imp = (moe["rmse"] - flat["rmse"]) / flat["rmse"] * 100

    log.info("")
    log.info("MAE improvement:  %+.1f%%", mae_imp)
    log.info("RMSE improvement: %+.1f%%", rmse_imp)

    save_checkpoint(results, status="complete")
    log.info("")
    log.info("Checkpoint: %s", CHECKPOINT_PATH)
    log.info("Log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
