"""
XGBoost model for job runtime prediction with rolling evaluation.

Subclasses the neutral RollingTabularModel base (shared rolling evaluation +
OHE/SVD preprocessing) and supplies the XGBoost regressor and hyperparameters.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from hpc_oda_commons.models.rolling_tabular.base import (
    RollingTabularConfig,
    RollingTabularModel,
)


@dataclass(frozen=True)
class JobRuntimeXGBoostConfig(RollingTabularConfig):
    """Rolling/preprocessing config plus XGBoost hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    # XGBoost objective function. Options:
    #   "reg:squarederror" — default, minimizes squared error
    #   "reg:pseudohubererror" — reduces outlier influence without transforming the target;
    #     the model still predicts in seconds directly but very long jobs don't dominate
    #     training gradients. Controlled by huber_slope (larger = more like squarederror).
    objective: str = "reg:squarederror"
    # Slope parameter for pseudohubererror. The transition point between quadratic (small
    # errors) and linear (large errors) behavior. Only used when objective is pseudohuber.
    huber_slope: float = 1000.0


class JobRuntimeXGBoostModel(RollingTabularModel):
    """
    XGBoost model for job runtime prediction with rolling evaluation.

    Public API:
    - evaluate(): rolling train/test evaluation with daily preprocessing cache
    - build_split_plan(): preview split windows without running evaluation
    - analyze_preprocessing(): profile categorical features and preview OHE/SVD config
    """

    _evaluate_desc = "rolling/xgboost"
    _log_prefix = "xgboost"

    def __init__(self, config: JobRuntimeXGBoostConfig | None = None) -> None:
        super().__init__(config or JobRuntimeXGBoostConfig())

    @staticmethod
    def _check_dependencies() -> None:
        missing: list[str] = []
        for package in ("xgboost", "sklearn"):
            if importlib.util.find_spec(package) is None:
                missing.append(package)
        if missing:
            missing_list = ", ".join(missing)
            raise RuntimeError(
                "Missing optional model dependencies: "
                f'{missing_list}. Install with `pip install -e ".[dev]"`.'
            )

    def _new_regressor(self, n_train: int) -> Any:
        from xgboost import XGBRegressor

        _ = n_train  # XGBoost does not size-adapt; the seam is shared with subclasses

        kwargs: dict[str, Any] = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "random_state": self.config.random_state,
            "objective": self.config.objective,
            "n_jobs": 1,
            "verbosity": 0,
        }
        if self.config.objective == "reg:pseudohubererror":
            kwargs["huber_slope"] = self.config.huber_slope

        return XGBRegressor(**kwargs)
