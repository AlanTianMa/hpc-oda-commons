#!/usr/bin/env python3
"""Run the corrected 2 x 2 x 2 MoE ablation on NLR Kestrel.

The original notebook split the complete table before rolling evaluation. Each
subset therefore built a different test grid, so its MoE metrics were not
comparable with the single-model baseline. This runner keeps one shared table
and lets the production MoE route jobs *inside* every shared rolling window.

Dimensions:
  - user routing: pooled users vs per-user experts for power users
  - wallclock routing: one bin vs data-derived requested-wallclock bins
  - time weighting: flat vs exponential decay at rate 0.05

Every experiment uses the current submission-time feature policy. The runner
refuses to compare results unless split times, exact test-row indices, scored
row counts, and true targets match the baseline.

Usage:
    python docs/benchmarking/run_user_moe_experiments.py
    python docs/benchmarking/run_user_moe_experiments.py --resume

Results:
    workspace/user_moe_experiments_corrected.json
    workspace/user_moe_experiments_corrected.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from hpc_oda_commons.benchmarking.hpc.slice import slice_to_window
from hpc_oda_commons.models.feature_policy import RUNTIME_PREDICTION_FEATURE_FIELDS
from hpc_oda_commons.models.job_runtime_moe_xgboost.model import (
    MoEXGBoostConfig,
    MoEXGBoostModel,
)
from hpc_oda_commons.models.rolling_tabular.base import RollingTabularModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "workspace" / "user_moe_experiments_corrected.json"
DEFAULT_LOG = REPO_ROOT / "workspace" / "user_moe_experiments_corrected.log"

WINDOW_START = "2025-03-29"
WINDOW_END = "2025-06-26"

# Deliberately unavailable at submission time (or a pure identifier). The
# current feature policy must keep these out even though Kestrel carries them.
FORBIDDEN_FEATURES = frozenset(
    {
        "job_id",
        "job_state",
        "exit_code",
        "allocated_cpus",
        "num_cores_alloc",
        "num_nodes_alloc",
        "start_time",
        "end_time",
        "runtime_seconds",
    }
)


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    user_routing: bool
    wallclock_routing: bool
    decay_rate: float


# Preserve the old notebook's factorial ordering while correcting its method.
EXPERIMENTS = (
    Experiment("1", "All/Single/Flat", False, False, 0.0),
    Experiment("2", "All/MoE-WC/Flat", False, True, 0.0),
    Experiment("3", "All/Single/Decay", False, False, 0.05),
    Experiment("4", "All/MoE-WC/Decay", False, True, 0.05),
    Experiment("5", "UserMoE/Single/Flat", True, False, 0.0),
    Experiment("6", "UserMoE/MoE-WC/Flat", True, True, 0.0),
    Experiment("7", "UserMoE/Single/Decay", True, False, 0.05),
    Experiment("8", "UserMoE/MoE-WC/Decay", True, True, 0.05),
)


class _MatchedFlatXGBoostModel(RollingTabularModel):
    """One estimator with exactly the same hyperparameters as each MoE expert."""

    _evaluate_desc = "rolling/xgboost-matched-flat"
    _log_prefix = "xgboost_matched_flat"

    def __init__(self, config: MoEXGBoostConfig) -> None:
        super().__init__(config)

    @staticmethod
    def _check_dependencies() -> None:
        MoEXGBoostModel._check_dependencies()

    def _new_regressor(self, n_train: int) -> Any:
        from xgboost import XGBRegressor

        del n_train
        cfg: MoEXGBoostConfig = self.config  # type: ignore[assignment]
        return XGBRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            gamma=cfg.gamma,
            random_state=cfg.random_state,
            n_jobs=cfg.estimator_n_jobs,
            verbosity=0,
        )


class _RoutingAblationMoEModel(MoEXGBoostModel):
    """Production MoE with either routing dimension optionally disabled."""

    def __init__(
        self,
        config: MoEXGBoostConfig,
        *,
        user_routing: bool,
        wallclock_routing: bool,
    ) -> None:
        super().__init__(config)
        self._use_user_routing = user_routing
        self._use_wallclock_routing = wallclock_routing

    def _power_users(self, train_rows: list[dict[str, Any]]) -> frozenset[str]:
        if not self._use_user_routing:
            return frozenset()
        return super()._power_users(train_rows)

    def _wallclock_edges(self, train_rows: list[dict[str, Any]]) -> tuple[float, ...]:
        if not self._use_wallclock_routing:
            return (math.inf,)
        return super()._wallclock_edges(train_rows)


def _setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("corrected_user_moe_experiments")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _resolve_data_path(value: Path | None) -> Path:
    if value is not None:
        path = value.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = (
        REPO_ROOT / "workspace" / "data" / "datasets" / "nlr_kestrel" / "data.parquet",
        REPO_ROOT / "data" / "datasets" / "nlr_kestrel" / "data.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Kestrel parquet not found; pass --data-path (checked: "
        + ", ".join(str(path) for path in candidates)
        + ")"
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hash_array(values: list[float]) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _test_rows_fingerprint(windows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for window in windows:
        digest.update(str(window["split_time"]).encode())
        digest.update(str(window["split_end_time"]).encode())
        digest.update(str(window["status"]).encode())
        if window["status"] == "ok":
            indices = np.asarray(window["test_row_indices"], dtype="<i8")
            digest.update(indices.tobytes())
    return digest.hexdigest()


def _split_times(windows: list[dict[str, Any]]) -> list[str]:
    return [str(window["split_time"]) for window in windows]


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _configuration(args: argparse.Namespace, data_path: Path) -> dict[str, Any]:
    return {
        "data_path": str(data_path),
        "window_start": args.window_start,
        "window_end": args.window_end,
        "n_windows": args.n_windows,
        "test_window_hours": args.test_window_hours,
        "training_lookback_days": args.training_lookback_days,
        "max_svd_components": args.max_svd_components,
        "target_max_one_hot_width": args.target_max_one_hot_width,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "power_user_percentile": args.power_user_percentile,
        "min_expert_rows": args.min_expert_rows,
        "n_wallclock_bins": args.n_wallclock_bins,
        "min_cluster_fraction": args.min_cluster_fraction,
        "estimator_n_jobs": args.estimator_n_jobs,
        "window_n_jobs": args.window_n_jobs,
        "random_state": 42,
    }


def _model_config(args: argparse.Namespace, experiment: Experiment) -> MoEXGBoostConfig:
    return MoEXGBoostConfig(
        n_windows=args.n_windows,
        test_window_hours=args.test_window_hours,
        training_lookback_days=args.training_lookback_days,
        max_svd_components=args.max_svd_components,
        target_max_one_hot_width=args.target_max_one_hot_width,
        random_state=42,
        window_n_jobs=args.window_n_jobs,
        time_decay_rate=experiment.decay_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        estimator_n_jobs=args.estimator_n_jobs,
        power_user_percentile=args.power_user_percentile,
        min_expert_rows=args.min_expert_rows,
        n_wallclock_bins=args.n_wallclock_bins,
        min_cluster_fraction=args.min_cluster_fraction,
    )


def _build_model(
    config: MoEXGBoostConfig,
    experiment: Experiment,
) -> RollingTabularModel:
    if not experiment.user_routing and not experiment.wallclock_routing:
        return _MatchedFlatXGBoostModel(config)
    return _RoutingAblationMoEModel(
        config,
        user_routing=experiment.user_routing,
        wallclock_routing=experiment.wallclock_routing,
    )


def _validate_features(model: RollingTabularModel, rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible, ignored = model.feature_field_report(rows)
    leaked = sorted(FORBIDDEN_FEATURES & set(eligible))
    if leaked:
        raise AssertionError(f"post-hoc fields eligible as features: {', '.join(leaked)}")
    if not set(eligible) <= RUNTIME_PREDICTION_FEATURE_FIELDS:
        raise AssertionError("eligible fields exceed the submission-time allowlist")
    return {
        "eligible": eligible,
        "ignored": ignored,
        "forbidden_present_but_ignored": sorted(FORBIDDEN_FEATURES & set(ignored)),
    }


def _reference_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows_scored": payload["summary"]["rows_scored"],
        "windows_scored": payload["summary"]["windows_scored"],
        "split_times": _split_times(payload["windows"]),
        "test_rows_sha256": _test_rows_fingerprint(payload["windows"]),
        "y_true_sha256": _hash_array(payload["_y_true"]),
    }


def _assert_comparable(payload: dict[str, Any], reference: dict[str, Any]) -> None:
    current = _reference_from_payload(payload)
    for key in (
        "rows_scored",
        "windows_scored",
        "split_times",
        "test_rows_sha256",
        "y_true_sha256",
    ):
        if current[key] != reference[key]:
            raise AssertionError(f"experiment is not comparable to baseline: {key} differs")


def _summarize(
    experiment: Experiment,
    payload: dict[str, Any],
    elapsed_minutes: float,
) -> dict[str, Any]:
    y_true = np.asarray(payload["_y_true"], dtype=float)
    y_pred = np.asarray(payload["_y_pred"], dtype=float)
    absolute_error = np.abs(y_true - y_pred)
    routing = payload["summary"].get("moe_routing")
    return {
        "key": experiment.key,
        "label": experiment.label,
        "user_routing": experiment.user_routing,
        "wallclock_routing": experiment.wallclock_routing,
        "decay_rate": experiment.decay_rate,
        "mae": float(payload["mae"]),
        "rmse": float(payload["rmse"]),
        "median_absolute_error": float(np.median(absolute_error)),
        "underprediction_ratio": float(np.mean(y_pred < y_true) * 100.0),
        "rows_scored": int(payload["summary"]["rows_scored"]),
        "windows_scored": int(payload["summary"]["windows_scored"]),
        "elapsed_minutes": round(elapsed_minutes, 3),
        "test_rows_sha256": _test_rows_fingerprint(payload["windows"]),
        "y_true_sha256": _hash_array(payload["_y_true"]),
        "routing": routing,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="skip completed experiments")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--window-start", default=WINDOW_START)
    parser.add_argument("--window-end", default=WINDOW_END)
    parser.add_argument("--n-windows", type=int, default=120)
    parser.add_argument("--test-window-hours", type=int, default=6)
    parser.add_argument("--training-lookback-days", type=int, default=60)
    parser.add_argument("--max-svd-components", type=int, default=256)
    parser.add_argument("--target-max-one-hot-width", type=int, default=2048)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-weight", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--power-user-percentile", type=float, default=0.99)
    parser.add_argument("--min-expert-rows", type=int, default=100)
    parser.add_argument("--n-wallclock-bins", type=int, default=5)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.02)
    parser.add_argument("--estimator-n-jobs", type=int, default=1)
    parser.add_argument("--window-n-jobs", type=int, default=1)
    parser.add_argument(
        "--quiet-model",
        action="store_true",
        help="suppress per-window model progress",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    log_path = args.log_path.expanduser().resolve()
    logger = _setup_logging(log_path)
    data_path = _resolve_data_path(args.data_path)
    config = _configuration(args, data_path)

    previous = _load_checkpoint(checkpoint_path) if args.resume else None
    if previous is not None and previous.get("configuration") != config:
        raise ValueError(
            "checkpoint configuration differs from this run; use matching arguments "
            "or a different --checkpoint path"
        )

    state: dict[str, Any] = previous or {
        "schema_version": "user-moe-corrected.v1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "configuration": config,
        "data": {},
        "feature_policy": {},
        "reference": None,
        "results": {},
    }

    logger.info("=" * 78)
    logger.info("Corrected User MoE 2 x 2 x 2 ablation")
    logger.info("git=%s", state["git_commit"])
    logger.info("data=%s", data_path)
    logger.info("window=%s through %s", args.window_start, args.window_end)
    logger.info(
        "split=%d x %dh, lookback=%dd, SVD=%d, OHE=%d",
        args.n_windows,
        args.test_window_hours,
        args.training_lookback_days,
        args.max_svd_components,
        args.target_max_one_hot_width,
    )
    logger.info("=" * 78)

    table = pq.read_table(data_path)
    sliced = slice_to_window(table, args.window_start, args.window_end)
    rows = sliced.to_pylist()
    state["data"] = {
        "source_rows": table.num_rows,
        "slice_rows": sliced.num_rows,
        "columns": sliced.column_names,
    }
    logger.info(
        "loaded %s rows from %s-row source",
        f"{len(rows):,}",
        f"{table.num_rows:,}",
    )

    baseline_config = _model_config(args, EXPERIMENTS[0])
    feature_report = _validate_features(_MatchedFlatXGBoostModel(baseline_config), rows)
    state["feature_policy"] = feature_report
    logger.info("eligible features: %s", ", ".join(feature_report["eligible"]))
    logger.info(
        "post-hoc columns present but ignored: %s",
        ", ".join(feature_report["forbidden_present_but_ignored"]) or "(none present)",
    )
    _atomic_write(checkpoint_path, state)

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        if experiment.key in state["results"]:
            logger.info("[%d/8] %s: cached", index, experiment.label)
            continue

        logger.info("")
        logger.info("[%d/8] running %s", index, experiment.label)
        model = _build_model(_model_config(args, experiment), experiment)
        started = time.time()
        payload = model.evaluate(
            rows,
            verbose=not args.quiet_model,
            capture_artifacts=True,
        )
        elapsed_minutes = (time.time() - started) / 60.0

        if state["reference"] is None:
            state["reference"] = _reference_from_payload(payload)
        else:
            _assert_comparable(payload, state["reference"])

        result = _summarize(experiment, payload, elapsed_minutes)
        state["results"][experiment.key] = result
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(checkpoint_path, state)
        logger.info(
            "[%d/8] %s: MAE=%ss RMSE=%ss median_AE=%ss scored=%s time=%.1fmin",
            index,
            experiment.label,
            f"{result['mae']:,.1f}",
            f"{result['rmse']:,.1f}",
            f"{result['median_absolute_error']:,.1f}",
            f"{result['rows_scored']:,}",
            elapsed_minutes,
        )

    reference = state["reference"]
    for result in state["results"].values():
        if result["rows_scored"] != reference["rows_scored"]:
            raise AssertionError("completed results do not share a scored population")
        if result["test_rows_sha256"] != reference["test_rows_sha256"]:
            raise AssertionError("completed results do not share exact test rows")
        if result["y_true_sha256"] != reference["y_true_sha256"]:
            raise AssertionError("completed results do not share true targets")

    state["status"] = "complete"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(checkpoint_path, state)
    logger.info("")
    logger.info("All eight experiments complete and same-row checks passed.")
    logger.info("results=%s", checkpoint_path)
    logger.info("log=%s", log_path)


if __name__ == "__main__":
    main()
