"""A generic ExperimentEnvironment backed by a GP emulator fit on any
tabular dataset: pick numeric input columns, an objective column to
maximize, and (optionally) a constraint column with a threshold.

This is the piece that makes "bring your own data" possible -- the exact
same optimization/benchmark code that runs on the CSTR simulator
(titrate/physics/) and the Reizman Suzuki dataset (reizman_suzuki_env.py,
which is a thin wrapper around this class) runs unmodified on whatever
table a user hands it, because they all implement ExperimentEnvironment.

Unlike the CSTR/Reizman environments, objective values here are NOT forced
into a [0, 1] "yield fraction" -- an arbitrary uploaded column has no such
guarantee, so values stay in their native units. Acquisition functions
(Expected Improvement, probability of feasibility) only compare relative
magnitudes, so this doesn't affect the optimization itself, only display
formatting (handled by the caller).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from titrate.environments.base import EvaluationResult, ExperimentEnvironment
from titrate.surrogate.gp_model import GPSurrogate


class TabularEmulatorEnvironment(ExperimentEnvironment):
    def __init__(
        self,
        data: pd.DataFrame,
        input_columns: list[str],
        objective_column: str,
        dimension_names: tuple[str, ...] | None = None,
        constraint_column: str | None = None,
        constraint_max: float = float("inf"),
        clip_range: tuple[float, float] | None = None,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        if len(input_columns) < 1:
            raise ValueError("Need at least one input column.")

        X = data[input_columns].to_numpy(dtype=float)
        y = data[objective_column].to_numpy(dtype=float)

        self.dimension_names = tuple(dimension_names or input_columns)
        self.clip_range = clip_range
        self.bounds = np.column_stack([X.min(axis=0), X.max(axis=0)])
        self.n_real_experiments = len(data)
        self.objective_column = objective_column

        self._emulator = GPSurrogate(self.bounds, random_state=random_state).fit(X, y)
        self._X_train, self._y_train = X, y

        self._constraint_emulator = None
        if constraint_column is not None:
            c = data[constraint_column].to_numpy(dtype=float)
            self._constraint_emulator = GPSurrogate(self.bounds, random_state=random_state).fit(X, c)
            self.constraint_max = constraint_max
            self.constraint_name = constraint_column
        else:
            self.constraint_max = float("inf")
            self.constraint_name = "none"

    def _clip(self, value: float) -> float:
        if self.clip_range is None:
            return value
        return float(np.clip(value, self.clip_range[0], self.clip_range[1]))

    def evaluate_noiseless(self, x: np.ndarray) -> EvaluationResult:
        mean, _ = self._emulator.predict(np.atleast_2d(x))
        return EvaluationResult(objective=self._clip(mean[0]), constraint_value=self._predict_constraint(x))

    def evaluate(self, x: np.ndarray, rng: np.random.Generator) -> EvaluationResult:
        """Sample from the emulator's own predictive distribution -- its
        posterior std at x is a principled noise estimate (higher where the
        real data was sparser), rather than an arbitrary assumed noise level."""
        mean, std = self._emulator.predict(np.atleast_2d(x))
        sample = float(rng.normal(mean[0], std[0]))
        return EvaluationResult(objective=self._clip(sample), constraint_value=self._predict_constraint(x))

    def _predict_constraint(self, x: np.ndarray) -> float:
        if self._constraint_emulator is None:
            return 0.0
        mean, _ = self._constraint_emulator.predict(np.atleast_2d(x))
        return float(mean[0])

    def emulator_holdout_rmse(self, n_splits: int = 5, random_state: int = 0) -> float:
        """K-fold cross-validated RMSE of the emulator against the real
        measurements it was fit on -- the honest accuracy check for a
        benchmark built on a fitted proxy rather than a closed-form model."""
        n_splits = min(n_splits, len(self._X_train))
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        squared_errors = []
        for train_idx, test_idx in kfold.split(self._X_train):
            fold_gp = GPSurrogate(self.bounds, random_state=random_state).fit(
                self._X_train[train_idx], self._y_train[train_idx]
            )
            mean, _ = fold_gp.predict(self._X_train[test_idx])
            squared_errors.extend((mean - self._y_train[test_idx]) ** 2)
        return float(np.sqrt(np.mean(squared_errors)))
