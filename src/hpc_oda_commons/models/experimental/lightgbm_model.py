"""
LightGBM model for job runtime prediction with rolling evaluation.

Subclasses the shared RollingTabularModel base. LightGBM grows trees leaf-wise
(best-first) rather than depth-wise, which often produces more accurate models
on large datasets. It also handles categorical features more efficiently than
XGBoost's one-hot encoding approach via native categorical support.
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
class ExperimentalLightGBMConfig(RollingTabularConfig):
    """Rolling/preprocessing config plus LightGBM hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 8
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8


class ExperimentalLightGBMModel(RollingTabularModel):
    """
    LightGBM model for job runtime prediction with rolling evaluation.

    Public API:
    - evaluate(): rolling train/test evaluation with daily preprocessing cache
    - build_split_plan(): preview split windows without running evaluation
    - analyze_preprocessing(): profile categorical features and preview OHE/SVD config
    """

    _evaluate_desc = "rolling/lightgbm"
    _log_prefix = "lightgbm"

    def __init__(self, config: ExperimentalLightGBMConfig | None = None) -> None:
        super().__init__(config or ExperimentalLightGBMConfig())

    @staticmethod
    def _check_dependencies() -> None:
        missing: list[str] = []
        for package in ("lightgbm", "sklearn"):
            if importlib.util.find_spec(package) is None:
                missing.append(package)
        if missing:
            missing_list = ", ".join(missing)
            raise RuntimeError(
                "Missing optional model dependencies: "
                f"{missing_list}. Install with `pip install lightgbm`."
            )

    def _new_regressor(self, n_train: int) -> Any:
        from lightgbm import LGBMRegressor

        _ = n_train

        return LGBMRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=-1,
        )
