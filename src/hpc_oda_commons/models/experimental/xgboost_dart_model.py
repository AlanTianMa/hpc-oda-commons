"""
XGBoost DART model for job runtime prediction with rolling evaluation.

DART (Dropout Additive Regression Trees) randomly drops some trees during
training. This prevents later trees from over-specializing on specific training
examples and can improve generalization. The model still predicts in seconds
directly — the difference is in how trees are combined during training.
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
class ExperimentalXGBoostDartConfig(RollingTabularConfig):
    """Rolling/preprocessing config plus XGBoost DART hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    # DART-specific parameters
    rate_drop: float = 0.1  # fraction of trees to drop during each boosting round
    skip_drop: float = 0.5  # probability of skipping the dropout procedure entirely


class ExperimentalXGBoostDartModel(RollingTabularModel):
    """
    XGBoost with DART booster for job runtime prediction.

    DART randomly drops trees during training to prevent over-specialization.
    rate_drop controls what fraction of trees are dropped each round.
    skip_drop controls how often dropout is skipped entirely.
    """

    _evaluate_desc = "rolling/xgboost-dart"
    _log_prefix = "xgboost_dart"

    def __init__(self, config: ExperimentalXGBoostDartConfig | None = None) -> None:
        super().__init__(config or ExperimentalXGBoostDartConfig())

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
            booster="dart",
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            rate_drop=self.config.rate_drop,
            skip_drop=self.config.skip_drop,
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=0,
        )
