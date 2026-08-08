"""A real-data experiment environment: a GP emulator fit on real Suzuki-Miyaura
cross-coupling flow-chemistry data, wrapped behind the same ExperimentEnvironment
interface as the CSTR simulator.

Data: Reizman, B. J.; Wang, Y.-M.; Buchwald, S. L.; Jensen, K. F. "Suzuki-Miyaura
cross-coupling optimization enabled by automated feedback." React. Chem. Eng.
2016, 1, 658-666. See data/README.md for provenance.

Why an emulator, not the raw table directly: a sequential optimization strategy
needs to query points that weren't in the original 96-experiment dataset. Fitting
a GP regression model on the real measurements and treating its predictions as
the queryable "ground truth" is the same technique the Summit benchmarking
package (Felton et al., 2021) uses for exactly this purpose -- it turns a fixed
real dataset into a continuously queryable benchmark function, at the cost of
the emulator's own regression error, which is reported (not hidden) via
`emulator_holdout_rmse()` below and in the README's real-data validation section.

This dataset has no purity/impurity specification, so unlike the CSTR
environment there is no real engineering constraint here -- constraint_max is
set to +inf (always satisfied), which makes constrained BO on this environment
mathematically reduce to plain BO. This is stated explicitly rather than
inventing a constraint that isn't in the source data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from titrate.environments.base import EvaluationResult, ExperimentEnvironment
from titrate.surrogate.gp_model import GPSurrogate

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reizman_suzuki_case1.csv"
DEFAULT_CATALYST = "P1-L4"  # the most-sampled catalyst in the dataset (37 of 96 runs)


def _load_catalyst_subset(catalyst: str = DEFAULT_CATALYST) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, skiprows=[1])  # row 1 is a "TYPE" metadata row, not data
    return df[df["catalyst"] == catalyst].reset_index(drop=True)


class ReizmanSuzukiEnvironment(ExperimentEnvironment):
    def __init__(self, catalyst: str = DEFAULT_CATALYST, random_state: int = 0) -> None:
        super().__init__()
        self.catalyst = catalyst
        data = _load_catalyst_subset(catalyst)
        self.n_real_experiments = len(data)

        self.dimension_names = ("residence_time_s", "temperature_C", "catalyst_loading_mol_pct")
        X = data[["t_res", "temperature", "catalyst_loading"]].to_numpy(dtype=float)
        y = data["yld"].to_numpy(dtype=float) / 100.0  # yield reported as 0-100%, rescale to [0,1]

        self.bounds = np.column_stack([X.min(axis=0), X.max(axis=0)])
        self.constraint_max = float("inf")  # no purity/impurity spec in this dataset -- see module docstring
        self.constraint_name = "none (unconstrained real-data benchmark)"

        self._emulator = GPSurrogate(self.bounds, random_state=random_state).fit(X, y)
        self._X_train, self._y_train = X, y

    def evaluate_noiseless(self, x: np.ndarray) -> EvaluationResult:
        mean, _ = self._emulator.predict(np.atleast_2d(x))
        return EvaluationResult(objective=float(np.clip(mean[0], 0.0, 1.0)), constraint_value=0.0)

    def evaluate(self, x: np.ndarray, rng: np.random.Generator) -> EvaluationResult:
        """Sample from the emulator's own predictive distribution -- its
        posterior std at x is a principled noise estimate (higher where the
        real data was sparser), rather than an arbitrary assumed noise level."""
        mean, std = self._emulator.predict(np.atleast_2d(x))
        sample = rng.normal(mean[0], std[0])
        return EvaluationResult(objective=float(np.clip(sample, 0.0, 1.0)), constraint_value=0.0)

    def emulator_holdout_rmse(self, n_splits: int = 5, random_state: int = 0) -> float:
        """K-fold cross-validated RMSE of the emulator against the real
        measurements it was fit on -- the honest accuracy check for a
        benchmark built on a fitted proxy rather than a closed-form model."""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        squared_errors = []
        for train_idx, test_idx in kfold.split(self._X_train):
            fold_gp = GPSurrogate(self.bounds, random_state=random_state).fit(
                self._X_train[train_idx], self._y_train[train_idx]
            )
            mean, _ = fold_gp.predict(self._X_train[test_idx])
            squared_errors.extend((mean - self._y_train[test_idx]) ** 2)
        return float(np.sqrt(np.mean(squared_errors)))
