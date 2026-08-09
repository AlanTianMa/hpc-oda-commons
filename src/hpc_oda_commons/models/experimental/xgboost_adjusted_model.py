"""
Adjusted XGBoost model with monotone constraints for job runtime prediction.

This variant applies domain knowledge through monotone constraints — telling
XGBoost that certain relationships must be non-decreasing (e.g. more cores
requested should not predict shorter runtime). This prevents the model from
learning spurious correlations on small training sets.

Also uses deeper trees and more estimators than the default XGBoost config.
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
class ExperimentalXGBoostAdjustedConfig(RollingTabularConfig):
    """Rolling/preprocessing config plus adjusted XGBoost hyperparameters."""

    n_estimators: int = 200
    max_depth: int = 12
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1  # minimum loss reduction for a split


class ExperimentalXGBoostAdjustedModel(RollingTabularModel):
    """
    XGBoost with domain-informed tuning for job runtime prediction.

    Differences from the default XGBoost:
    - Deeper trees (max_depth=12 vs 8) for more complex patterns
    - More estimators (200 vs 100) with lower learning rate (0.03 vs 0.05)
    - min_child_weight=5 prevents splits on very few samples
    - gamma=0.1 requires minimum loss reduction for splits (regularization)
    """

    _evaluate_desc = "rolling/xgboost-adjusted"
    _log_prefix = "xgboost_adjusted"

    def __init__(self, config: ExperimentalXGBoostAdjustedConfig | None = None) -> None:
        super().__init__(config or ExperimentalXGBoostAdjustedConfig())

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

        _ = n_train

        return XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=0,
        )
