"""CSTR experiment environment used by the synthetic benchmark.

All observations from this environment are simulated. The reactor equations live
in ``titrate.physics.reactor``. Relative Gaussian measurement noise is added as a
modeling assumption and is not fit to a specific instrument.

The search domain limits temperature, residence time, and catalyst loading. The
impurity specification is a nonlinear process constraint evaluated from the reactor
model.
"""

from __future__ import annotations

import numpy as np

from titrate.environments.base import EvaluationResult, ExperimentEnvironment
from titrate.physics.reactor import REFERENCE_PARAMS, ReactionParameters, cstr_yield

DEFAULT_TEMPERATURE_BOUNDS = (320.0, 400.0)  # K
DEFAULT_RESIDENCE_TIME_BOUNDS = (0.1, 5.0)  # hr
DEFAULT_CATALYST_BOUNDS = (0.0, 2.0)  # mol%
DEFAULT_IMPURITY_MAX = 0.05  # mol/L byproduct concentration (purity spec)
DEFAULT_NOISE_RELATIVE_STD = 0.025  # ~2.5% relative measurement noise


class CSTREnvironment(ExperimentEnvironment):
    def __init__(
        self,
        params: ReactionParameters = REFERENCE_PARAMS,
        noise_relative_std: float = DEFAULT_NOISE_RELATIVE_STD,
        impurity_max: float = DEFAULT_IMPURITY_MAX,
        temperature_bounds: tuple[float, float] = DEFAULT_TEMPERATURE_BOUNDS,
        residence_time_bounds: tuple[float, float] = DEFAULT_RESIDENCE_TIME_BOUNDS,
        catalyst_bounds: tuple[float, float] = DEFAULT_CATALYST_BOUNDS,
    ) -> None:
        super().__init__()
        self.params = params
        self.noise_relative_std = noise_relative_std
        self.dimension_names = (
            "temperature_K",
            "residence_time_hr",
            "catalyst_loading_mol_pct",
        )
        self.bounds = np.array(
            [temperature_bounds, residence_time_bounds, catalyst_bounds], dtype=float
        )
        self.constraint_max = impurity_max
        self.constraint_name = "impurity_concentration_mol_per_L"

    def evaluate_noiseless(self, x: np.ndarray) -> EvaluationResult:
        temperature, residence_time, catalyst_loading = np.asarray(x, dtype=float)
        out = cstr_yield(temperature, residence_time, catalyst_loading, self.params)
        return EvaluationResult(objective=out.yield_, constraint_value=out.impurity)

    def evaluate(self, x: np.ndarray, rng: np.random.Generator) -> EvaluationResult:
        clean = self.evaluate_noiseless(x)
        noisy_objective = clean.objective * (
            1.0 + rng.normal(0.0, self.noise_relative_std)
        )
        noisy_constraint = clean.constraint_value * (
            1.0 + rng.normal(0.0, self.noise_relative_std)
        )
        return EvaluationResult(
            objective=float(np.clip(noisy_objective, 0.0, 1.0)),
            constraint_value=float(max(noisy_constraint, 0.0)),
        )
