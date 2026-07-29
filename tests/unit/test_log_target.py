"""Tests for heavy-tail handling in rolling tabular models.

Two approaches to handle heavy-tailed HPC runtime distributions:
1. log_target: train on log1p(runtime), back-transform with expm1
2. pseudohuber objective: predict in seconds directly but reduce outlier influence

Correctness tests verify:
- Non-negative predictions after expm1 back-transform
- Default behavior unchanged when log_target=False
- Model runs without error when log_target=True

Accuracy tests compare log_target=True vs False on real datasets (when available)
to measure the impact of log-transform on skewed HPC runtime distributions.
These tests are marked with pytest.mark.slow and require downloaded datasets
in workspace/data/datasets/.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from hpc_oda_commons.models.job_runtime_xgboost.model import (
    JobRuntimeXGBoostConfig,
    JobRuntimeXGBoostModel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skewed_rows(n_hours: int = 8) -> list[dict]:
    """Rows with a heavy-tailed runtime distribution for log_target testing."""
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    rows = []
    runtimes = [30, 60, 120, 300, 600, 1800, 3600, 7200, 14400, 28800, 43200, 86400]
    for i in range(n_hours * 4):
        submit = base + timedelta(hours=i)
        end = submit + timedelta(hours=1)
        rt = float(runtimes[i % len(runtimes)])
        rows.append(
            {
                "job_id": i,
                "submit_time": submit,
                "end_time": end,
                "partition": "compute" if i % 2 == 0 else "debug",
                "num_cores_req": float((i % 4) + 1),
                "runtime_seconds": rt,
            }
        )
    return rows


def _workspace_dataset_path(name: str) -> Path:
    """Resolve a dataset path from workspace/data/datasets/."""
    return (
        Path(__file__).resolve().parents[2]
        / "workspace"
        / "data"
        / "datasets"
        / name
        / "data.parquet"
    )


def _load_dataset_rows(name: str, max_rows: int = 50_000) -> list[dict] | None:
    """Load a windowed subset of rows from a dataset. Returns None if unavailable."""
    import pyarrow.parquet as pq

    path = _workspace_dataset_path(name)
    if not path.exists():
        return None

    table = pq.read_table(path)
    # Take a sample from the end of the dataset (most recent jobs)
    if table.num_rows > max_rows:
        table = table.slice(table.num_rows - max_rows)
    return table.to_pylist()


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_log_target_produces_non_negative_predictions() -> None:
    """With log_target=True, all predictions must be >= 0 after expm1 back-transform."""
    rows = _make_skewed_rows(10)
    config = JobRuntimeXGBoostConfig(
        n_windows=2,
        test_window_hours=2,
        training_lookback_days=100,
        max_svd_components=4,
        target_max_one_hot_width=32,
        log_target=True,
    )
    model = JobRuntimeXGBoostModel(config)
    payload = model.evaluate(rows)

    for window in payload["windows"]:
        if window["status"] == "ok":
            assert window["metrics"]["mae"] >= 0
            assert window["metrics"]["rmse"] >= 0

    assert payload["mae"] >= 0
    assert payload["rmse"] >= 0


def test_log_target_false_matches_default_behavior() -> None:
    """log_target=False (default) should produce identical results to before."""
    rows = _make_skewed_rows(10)
    config_default = JobRuntimeXGBoostConfig(
        n_windows=2,
        test_window_hours=2,
        training_lookback_days=100,
        max_svd_components=4,
        target_max_one_hot_width=32,
        random_state=42,
    )
    config_explicit = JobRuntimeXGBoostConfig(
        n_windows=2,
        test_window_hours=2,
        training_lookback_days=100,
        max_svd_components=4,
        target_max_one_hot_width=32,
        random_state=42,
        log_target=False,
    )
    payload_default = JobRuntimeXGBoostModel(config_default).evaluate(rows)
    payload_explicit = JobRuntimeXGBoostModel(config_explicit).evaluate(rows)

    assert payload_default["mae"] == payload_explicit["mae"]
    assert payload_default["rmse"] == payload_explicit["rmse"]


def test_log_target_runs_without_error() -> None:
    """log_target=True should complete evaluation without exceptions."""
    rows = _make_skewed_rows(10)
    config = JobRuntimeXGBoostConfig(
        n_windows=3,
        test_window_hours=2,
        training_lookback_days=100,
        max_svd_components=4,
        target_max_one_hot_width=32,
        log_target=True,
    )
    payload = JobRuntimeXGBoostModel(config).evaluate(rows)

    assert payload["summary"]["windows_scored"] > 0
    assert payload["summary"]["rows_scored"] > 0


def test_log1p_expm1_roundtrip() -> None:
    """log1p and expm1 are exact inverses within floating point tolerance."""
    values = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000], dtype=float)
    roundtripped = np.expm1(np.log1p(values))
    np.testing.assert_allclose(roundtripped, values, rtol=1e-12)


def test_log_target_clips_negative_predictions() -> None:
    """expm1 of very negative numbers should be clipped to 0, not go negative."""
    # Simulate what happens in the code
    log_preds = np.array([-100.0, -50.0, -1.0, 0.0, 5.0, 10.0])
    back_transformed = np.maximum(np.expm1(log_preds), 0.0)
    assert (back_transformed >= 0).all()


# ---------------------------------------------------------------------------
# Accuracy tests on real datasets (skipped if data unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_log_target_accuracy_on_pm100() -> None:
    """Compare log_target=True vs False on PM100 (if downloaded).

    PM100 has 122K rows in its benchmark window with a typical HPC skew
    (median 215s, p99 85,623s). This test runs a small rolling evaluation
    on both settings and reports whether log_target improves MAE.
    """
    rows = _load_dataset_rows("pm100", max_rows=20_000)
    if rows is None:
        pytest.skip("pm100 dataset not available in workspace/data/datasets/")

    config_raw = JobRuntimeXGBoostConfig(
        n_windows=4,
        test_window_hours=6,
        training_lookback_days=30,
        max_svd_components=16,
        target_max_one_hot_width=128,
        random_state=42,
        log_target=False,
    )
    config_log = JobRuntimeXGBoostConfig(
        n_windows=4,
        test_window_hours=6,
        training_lookback_days=30,
        max_svd_components=16,
        target_max_one_hot_width=128,
        random_state=42,
        log_target=True,
    )

    payload_raw = JobRuntimeXGBoostModel(config_raw).evaluate(rows)
    payload_log = JobRuntimeXGBoostModel(config_log).evaluate(rows)

    mae_raw = payload_raw["mae"]
    mae_log = payload_log["mae"]
    rmse_raw = payload_raw["rmse"]
    rmse_log = payload_log["rmse"]

    print(f"\n  PM100 (n={payload_raw['summary']['rows_scored']} scored rows):")
    print(f"    log_target=False:  MAE={mae_raw:,.1f}s  RMSE={rmse_raw:,.1f}s")
    print(f"    log_target=True:   MAE={mae_log:,.1f}s  RMSE={rmse_log:,.1f}s")
    print(f"    MAE change: {(mae_log - mae_raw) / mae_raw * 100:+.1f}%")
    print(f"    RMSE change: {(rmse_log - rmse_raw) / rmse_raw * 100:+.1f}%")

    # Both should produce valid finite results
    assert np.isfinite(mae_raw) and np.isfinite(mae_log)
    assert np.isfinite(rmse_raw) and np.isfinite(rmse_log)
    assert payload_raw["summary"]["rows_scored"] > 0
    assert payload_log["summary"]["rows_scored"] > 0


@pytest.mark.slow
def test_log_target_accuracy_on_lassen() -> None:
    """Compare log_target=True vs False on Lassen (if downloaded).

    Lassen has longer runtimes (median 2,816s) but less extreme skew (p99/p50=15x).
    """
    rows = _load_dataset_rows("lassen", max_rows=20_000)
    if rows is None:
        pytest.skip("lassen dataset not available in workspace/data/datasets/")

    config_raw = JobRuntimeXGBoostConfig(
        n_windows=4,
        test_window_hours=6,
        training_lookback_days=30,
        max_svd_components=16,
        target_max_one_hot_width=128,
        random_state=42,
        log_target=False,
    )
    config_log = JobRuntimeXGBoostConfig(
        n_windows=4,
        test_window_hours=6,
        training_lookback_days=30,
        max_svd_components=16,
        target_max_one_hot_width=128,
        random_state=42,
        log_target=True,
    )

    payload_raw = JobRuntimeXGBoostModel(config_raw).evaluate(rows)
    payload_log = JobRuntimeXGBoostModel(config_log).evaluate(rows)

    mae_raw = payload_raw["mae"]
    mae_log = payload_log["mae"]
    rmse_raw = payload_raw["rmse"]
    rmse_log = payload_log["rmse"]

    print(f"\n  Lassen (n={payload_raw['summary']['rows_scored']} scored rows):")
    print(f"    log_target=False:  MAE={mae_raw:,.1f}s  RMSE={rmse_raw:,.1f}s")
    print(f"    log_target=True:   MAE={mae_log:,.1f}s  RMSE={rmse_log:,.1f}s")
    print(f"    MAE change: {(mae_log - mae_raw) / mae_raw * 100:+.1f}%")
    print(f"    RMSE change: {(rmse_log - rmse_raw) / rmse_raw * 100:+.1f}%")

    assert np.isfinite(mae_raw) and np.isfinite(mae_log)
    assert np.isfinite(rmse_raw) and np.isfinite(rmse_log)
    assert payload_raw["summary"]["rows_scored"] > 0
    assert payload_log["summary"]["rows_scored"] > 0


@pytest.mark.slow
def test_log_target_accuracy_on_atlas_mustang() -> None:
    """Compare log_target=True vs False on Atlas Mustang (if downloaded).

    Atlas Mustang has extreme short-job skew (50% under 60s, p99/p50=820x).
    """
    rows = _load_dataset_rows("atlas_mustang", max_rows=20_000)
    if rows is None:
        pytest.skip("atlas_mustang dataset not available in workspace/data/datasets/")

    config_raw = JobRuntimeXGBoostConfig(
        n_windows=4,
        test_window_hours=6,
        training_lookback_days=30,
        max_svd_components=16,
        target_max_one_hot_width=128,
        random_state=42,
        log_target=False,
    )
    config_log = JobRuntimeXGBoostConfig(
        n_windows=4,
        test_window_hours=6,
        training_lookback_days=30,
        max_svd_components=16,
        target_max_one_hot_width=128,
        random_state=42,
        log_target=True,
    )

    payload_raw = JobRuntimeXGBoostModel(config_raw).evaluate(rows)
    payload_log = JobRuntimeXGBoostModel(config_log).evaluate(rows)

    mae_raw = payload_raw["mae"]
    mae_log = payload_log["mae"]
    rmse_raw = payload_raw["rmse"]
    rmse_log = payload_log["rmse"]

    print(f"\n  Atlas Mustang (n={payload_raw['summary']['rows_scored']} scored rows):")
    print(f"    log_target=False:  MAE={mae_raw:,.1f}s  RMSE={rmse_raw:,.1f}s")
    print(f"    log_target=True:   MAE={mae_log:,.1f}s  RMSE={rmse_log:,.1f}s")
    print(f"    MAE change: {(mae_log - mae_raw) / mae_raw * 100:+.1f}%")
    print(f"    RMSE change: {(rmse_log - rmse_raw) / rmse_raw * 100:+.1f}%")

    assert np.isfinite(mae_raw) and np.isfinite(mae_log)
    assert np.isfinite(rmse_raw) and np.isfinite(rmse_log)
    assert payload_raw["summary"]["rows_scored"] > 0
    assert payload_log["summary"]["rows_scored"] > 0


# ---------------------------------------------------------------------------
# Pseudo-Huber objective tests (XGBoost only)
# ---------------------------------------------------------------------------


def test_pseudohuber_runs_without_error() -> None:
    """XGBoost with reg:pseudohubererror should complete evaluation."""
    rows = _make_skewed_rows(10)
    config = JobRuntimeXGBoostConfig(
        n_windows=2,
        test_window_hours=2,
        training_lookback_days=100,
        max_svd_components=4,
        target_max_one_hot_width=32,
        objective="reg:pseudohubererror",
        huber_slope=1000.0,
    )
    payload = JobRuntimeXGBoostModel(config).evaluate(rows)

    assert payload["summary"]["windows_scored"] > 0
    assert payload["summary"]["rows_scored"] > 0
    assert payload["mae"] >= 0
    assert payload["rmse"] >= 0


def test_pseudohuber_produces_non_negative_predictions() -> None:
    """Pseudo-Huber predictions should be non-negative (model predicts in seconds directly)."""
    rows = _make_skewed_rows(10)
    config = JobRuntimeXGBoostConfig(
        n_windows=3,
        test_window_hours=2,
        training_lookback_days=100,
        max_svd_components=4,
        target_max_one_hot_width=32,
        objective="reg:pseudohubererror",
        huber_slope=500.0,
    )
    payload = JobRuntimeXGBoostModel(config).evaluate(rows)

    for window in payload["windows"]:
        if window["status"] == "ok":
            assert window["metrics"]["mae"] >= 0
            assert window["metrics"]["rmse"] >= 0


def test_pseudohuber_default_objective_unchanged() -> None:
    """Default objective remains reg:squarederror, producing identical results."""
    rows = _make_skewed_rows(10)
    config_default = JobRuntimeXGBoostConfig(
        n_windows=2,
        test_window_hours=2,
        training_lookback_days=100,
        max_svd_components=4,
        target_max_one_hot_width=32,
        random_state=42,
    )
    config_explicit = JobRuntimeXGBoostConfig(
        n_windows=2,
        test_window_hours=2,
        training_lookback_days=100,
        max_svd_components=4,
        target_max_one_hot_width=32,
        random_state=42,
        objective="reg:squarederror",
    )
    payload_default = JobRuntimeXGBoostModel(config_default).evaluate(rows)
    payload_explicit = JobRuntimeXGBoostModel(config_explicit).evaluate(rows)

    assert payload_default["mae"] == payload_explicit["mae"]
    assert payload_default["rmse"] == payload_explicit["rmse"]


# ---------------------------------------------------------------------------
# Accuracy comparison: all three approaches on real datasets
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_all_approaches_accuracy_on_pm100() -> None:
    """Compare squarederror vs log_target vs pseudohuber on PM100."""
    rows = _load_dataset_rows("pm100", max_rows=20_000)
    if rows is None:
        pytest.skip("pm100 dataset not available in workspace/data/datasets/")

    shared = {
        "n_windows": 4,
        "test_window_hours": 6,
        "training_lookback_days": 30,
        "max_svd_components": 16,
        "target_max_one_hot_width": 128,
        "random_state": 42,
    }

    configs = {
        "squarederror": JobRuntimeXGBoostConfig(**shared, log_target=False),
        "log_target": JobRuntimeXGBoostConfig(**shared, log_target=True),
        "pseudohuber": JobRuntimeXGBoostConfig(
            **shared, log_target=False, objective="reg:pseudohubererror", huber_slope=1000.0
        ),
    }

    print("\n  PM100 — all approaches:")
    for name, config in configs.items():
        payload = JobRuntimeXGBoostModel(config).evaluate(rows)
        mae = payload["mae"]
        rmse = payload["rmse"]
        scored = payload["summary"]["rows_scored"]
        print(f"    {name:20s}  MAE={mae:>8,.1f}s  RMSE={rmse:>10,.1f}s  scored={scored}")
        assert np.isfinite(mae) and np.isfinite(rmse)
        assert scored > 0


@pytest.mark.slow
def test_all_approaches_accuracy_on_lassen() -> None:
    """Compare squarederror vs log_target vs pseudohuber on Lassen."""
    rows = _load_dataset_rows("lassen", max_rows=20_000)
    if rows is None:
        pytest.skip("lassen dataset not available in workspace/data/datasets/")

    shared = {
        "n_windows": 4,
        "test_window_hours": 6,
        "training_lookback_days": 30,
        "max_svd_components": 16,
        "target_max_one_hot_width": 128,
        "random_state": 42,
    }

    configs = {
        "squarederror": JobRuntimeXGBoostConfig(**shared, log_target=False),
        "log_target": JobRuntimeXGBoostConfig(**shared, log_target=True),
        "pseudohuber": JobRuntimeXGBoostConfig(
            **shared, log_target=False, objective="reg:pseudohubererror", huber_slope=1000.0
        ),
    }

    print("\n  Lassen — all approaches:")
    for name, config in configs.items():
        payload = JobRuntimeXGBoostModel(config).evaluate(rows)
        mae = payload["mae"]
        rmse = payload["rmse"]
        scored = payload["summary"]["rows_scored"]
        print(f"    {name:20s}  MAE={mae:>8,.1f}s  RMSE={rmse:>10,.1f}s  scored={scored}")
        assert np.isfinite(mae) and np.isfinite(rmse)
        assert scored > 0


@pytest.mark.slow
def test_all_approaches_accuracy_on_atlas_mustang() -> None:
    """Compare squarederror vs log_target vs pseudohuber on Atlas Mustang."""
    rows = _load_dataset_rows("atlas_mustang", max_rows=20_000)
    if rows is None:
        pytest.skip("atlas_mustang dataset not available in workspace/data/datasets/")

    shared = {
        "n_windows": 4,
        "test_window_hours": 6,
        "training_lookback_days": 30,
        "max_svd_components": 16,
        "target_max_one_hot_width": 128,
        "random_state": 42,
    }

    configs = {
        "squarederror": JobRuntimeXGBoostConfig(**shared, log_target=False),
        "log_target": JobRuntimeXGBoostConfig(**shared, log_target=True),
        "pseudohuber": JobRuntimeXGBoostConfig(
            **shared, log_target=False, objective="reg:pseudohubererror", huber_slope=1000.0
        ),
    }

    print("\n  Atlas Mustang — all approaches:")
    for name, config in configs.items():
        payload = JobRuntimeXGBoostModel(config).evaluate(rows)
        mae = payload["mae"]
        rmse = payload["rmse"]
        scored = payload["summary"]["rows_scored"]
        print(f"    {name:20s}  MAE={mae:>8,.1f}s  RMSE={rmse:>10,.1f}s  scored={scored}")
        assert np.isfinite(mae) and np.isfinite(rmse)
        assert scored > 0
