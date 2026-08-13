#!/usr/bin/env python3
"""
Global decay rate grid search.

Evaluates decay rates [0.0, 0.01, 0.02, 0.03, 0.04, 0.05] using the full
User+WC MoE pipeline (all bins). Reports overall MAE for each rate.

Writes progressive checkpoints so the run can be resumed after a crash.

Usage:
    python docs/benchmarking/run_decay_grid_search.py
    python docs/benchmarking/run_decay_grid_search.py --resume

Output:
    workspace/decay_grid_search_checkpoint.json   (progressive checkpoint)
    workspace/decay_grid_search.log               (log file)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "workspace" / "data" / "datasets" / "nlr_kestrel" / "data.parquet"
CHECKPOINT_PATH = REPO_ROOT / "workspace" / "decay_grid_search_checkpoint.json"
LOG_PATH = REPO_ROOT / "workspace" / "decay_grid_search.log"

sys.path.insert(0, str(REPO_ROOT / "src"))

from hpc_oda_commons.models.experimental.xgboost_adjusted_model import (
    ExperimentalXGBoostAdjustedConfig,
    ExperimentalXGBoostAdjustedModel,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_WINDOWS = 120
TEST_WINDOW_HOURS = 6
TRAINING_LOOKBACK_DAYS = 120

DECAY_RATES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

POWER_USER_PERCENTILE = 0.99
ESTIMATOR_N_JOBS = 12

BIN_EDGES_H = [0, 2, 4, 24, 48, float("inf")]
BIN_LABELS = ["<=2h", "2-4h", "4-24h", "24-48h", ">48h"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("decay_grid")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(results: dict, status: str = "running") -> None:
    data = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rates_completed": len(results),
        "results": results,
    }
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(CHECKPOINT_PATH)


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        return data.get("results", {})
    return {}


# ---------------------------------------------------------------------------
# Model config factory
# ---------------------------------------------------------------------------
def make_config(decay_rate: float) -> ExperimentalXGBoostAdjustedConfig:
    return ExperimentalXGBoostAdjustedConfig(
        n_windows=N_WINDOWS,
        test_window_hours=TEST_WINDOW_HOURS,
        training_lookback_days=TRAINING_LOOKBACK_DAYS,
        max_svd_components=64,
        target_max_one_hot_width=512,
        random_state=42,
        n_estimators=200,
        max_depth=12,
        learning_rate=0.03,
        min_child_weight=5,
        gamma=0.1,
        time_decay_rate=decay_rate,
        estimator_n_jobs=ESTIMATOR_N_JOBS,
    )


# ---------------------------------------------------------------------------
# Bin assignment
# ---------------------------------------------------------------------------
def assign_bin(row: dict) -> str:
    wc_h = (row.get("requested_seconds") or 0) / 3600
    for i in range(len(BIN_EDGES_H) - 1):
        if wc_h <= BIN_EDGES_H[i + 1]:
            return BIN_LABELS[i]
    return BIN_LABELS[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Global decay rate grid search")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 70)
    log.info("Global decay rate grid search (User+WC MoE)")
    log.info("Rates to test: %s", DECAY_RATES)
    log.info("=" * 70)

    # --- Load data ---
    log.info("Loading data from %s", DATA_PATH)
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
    rows_all = df.to_dict("records")
    log.info("Loaded: %s rows, span: %s days",
             f"{len(rows_all):,}",
             (df["submit_time"].max() - df["submit_time"].min()).days)

    # --- Build bins ---
    user_counts = Counter(r.get("user") for r in rows_all)
    threshold = np.percentile(list(user_counts.values()), POWER_USER_PERCENTILE * 100)
    power_users = {u for u, c in user_counts.items() if c >= threshold}

    all_bins: dict[str, list] = {}
    for row in rows_all:
        user = row.get("user")
        bl = assign_bin(row)
        if user in power_users:
            key = f"power {user[:7]}/{bl}"
        else:
            key = f"non-power/{bl}"
        all_bins.setdefault(key, []).append(row)

    valid_bins = {k: v for k, v in all_bins.items() if len(v) >= 100}
    sorted_bins = sorted(valid_bins.items(), key=lambda x: -len(x[1]))
    total_rows = sum(len(v) for v in valid_bins.values())

    cumulative = 0
    top_bins = []
    for name, rows in sorted_bins:
        cumulative += len(rows)
        top_bins.append((name, rows))
        if cumulative / total_rows >= 0.90:
            break

    log.info("Power users: %d", len(power_users))
    log.info("Bins: %d (covering %.1f%% of valid rows)", len(top_bins), cumulative / total_rows * 100)
    for name, rows in top_bins:
        log.info("  %-35s %10s rows", name, f"{len(rows):,}")

    # --- Load checkpoint if resuming ---
    if args.resume:
        results = load_checkpoint()
        completed_rates = {float(k) for k in results.keys()}
        log.info("Resuming: %d rates already completed", len(completed_rates))
    else:
        results = {}
        completed_rates = set()
        log.info("Starting fresh (use --resume to continue from checkpoint)")

    # --- Config summary ---
    log.info("Config: SVD=%d, OHE=%d, windows=%d, lookback=%dd, n_jobs=%d",
             64, 512, N_WINDOWS, TRAINING_LOOKBACK_DAYS, ESTIMATOR_N_JOBS)
    log.info("Total: %d rates × %d bins per rate = %d model evaluations",
             len(DECAY_RATES), len(top_bins),
             len(DECAY_RATES) * len(top_bins))

    # --- Evaluate each rate ---
    total_start = time.time()
    rates_to_run = [r for r in DECAY_RATES if r not in completed_rates]
    log.info("Rates to run: %d / %d", len(rates_to_run), len(DECAY_RATES))

    for rate_idx, rate in enumerate(rates_to_run):
        elapsed_total = (time.time() - total_start) / 60
        if rate_idx > 0:
            avg_per_rate = elapsed_total / rate_idx
            remaining = avg_per_rate * (len(rates_to_run) - rate_idx)
        else:
            remaining = 0

        log.info("")
        log.info("=" * 70)
        log.info("[Rate %d/%d | %.0fmin elapsed | ~%.0fmin remaining]",
                 len(completed_rates) + rate_idx + 1, len(DECAY_RATES),
                 elapsed_total, remaining)
        log.info("Testing decay_rate=%.2f across all %d bins", rate, len(top_bins))
        log.info("=" * 70)

        rate_start = time.time()
        total_scored = 0
        weighted_mae_sum = 0.0
        weighted_se_sum = 0.0  # for RMSE: sum of (rmse^2 * scored)
        bin_results = {}

        for bin_idx, (bin_name, bin_rows) in enumerate(top_bins):
            bin_start = time.time()
            try:
                config = make_config(rate)
                model = ExperimentalXGBoostAdjustedModel(config)
                payload = model.evaluate(bin_rows)
                mae = payload["mae"]
                rmse = payload["rmse"]
                scored = payload["summary"]["rows_scored"]
                bin_time = (time.time() - bin_start) / 60

                total_scored += scored
                weighted_mae_sum += mae * scored
                weighted_se_sum += (rmse ** 2) * scored
                bin_results[bin_name] = {"mae": mae, "rmse": rmse, "scored": scored}

                log.info("  [%d/%d] %-30s scored=%6s MAE=%8s RMSE=%8s s (%.1fmin)",
                         bin_idx + 1, len(top_bins), bin_name,
                         f"{scored:,}", f"{mae:,.0f}", f"{rmse:,.0f}", bin_time)
            except Exception as e:
                log.error("  [%d/%d] %-30s FAILED: %s",
                          bin_idx + 1, len(top_bins), bin_name, e)
                bin_results[bin_name] = {"mae": None, "rmse": None, "scored": 0}

        # Compute weighted MAE and RMSE across all bins
        overall_mae = weighted_mae_sum / total_scored if total_scored > 0 else None
        overall_rmse = (weighted_se_sum / total_scored) ** 0.5 if total_scored > 0 else None
        rate_elapsed = (time.time() - rate_start) / 60

        results[str(rate)] = {
            "rate": rate,
            "overall_mae": overall_mae,
            "overall_rmse": overall_rmse,
            "total_scored": total_scored,
            "time_min": round(rate_elapsed, 1),
            "bins": bin_results,
        }

        log.info("")
        log.info("  >> rate=%.2f: overall MAE=%s s, RMSE=%s s, scored=%s, time=%.0fmin",
                 rate, f"{overall_mae:,.0f}" if overall_mae else "N/A",
                 f"{overall_rmse:,.0f}" if overall_rmse else "N/A",
                 f"{total_scored:,}", rate_elapsed)

        save_checkpoint(results)
        log.info("  Checkpoint saved (%d/%d rates complete)",
                 len(results), len(DECAY_RATES))

    # --- Final summary ---
    total_elapsed = (time.time() - total_start) / 60
    log.info("")
    log.info("=" * 70)
    log.info("GRID SEARCH COMPLETE — %.0f minutes total", total_elapsed)
    log.info("=" * 70)
    log.info("")
    log.info("%-10s %12s %12s %12s %10s", "Rate", "Overall MAE", "Overall RMSE", "Scored", "Time")
    log.info("-" * 60)

    best_rate = None
    best_mae = float("inf")
    flat_mae = None

    for rate in DECAY_RATES:
        r = results.get(str(rate))
        if r and r["overall_mae"]:
            mae = r["overall_mae"]
            rmse = r.get("overall_rmse")
            rmse_str = f"{rmse:,.0f}s" if rmse else "N/A"
            log.info("%-10.2f %12s %12s %12s %9.0fmin",
                     rate, f"{mae:,.0f}s", rmse_str, f"{r['total_scored']:,}", r["time_min"])
            if rate == 0.0:
                flat_mae = mae
            if mae < best_mae:
                best_mae = mae
                best_rate = rate

    if flat_mae and best_rate is not None:
        improvement = (best_mae - flat_mae) / flat_mae * 100
        log.info("")
        log.info("Best rate: %.2f (MAE=%s s, %+.1f%% vs flat)",
                 best_rate, f"{best_mae:,.0f}", improvement)

    save_checkpoint(results, status="complete")
    log.info("")
    log.info("Checkpoint: %s", CHECKPOINT_PATH)
    log.info("Log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
