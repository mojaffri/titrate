"""The V1 experiment source: the physics CSTR simulator, wrapped as an
ExperimentEnvironment with measurement noise added.

This is entirely simulated data -- there are no real experimental
measurements here. The physics is real (see titrate.physics.reactor); the
measurement noise is a modeling choice meant to mimic realistic
experimental/analytical reproducibility (~2-3% relative error), not fit to
any real instrument.

Bounds represent real engineering constraints on the problem:
  - temperature: material/safety ceiling on a liquid-phase reaction
  - residence time: reactor size / throughput cost
  - catalyst loading: catalyst cost
  - impurity constraint: a product purity specification (not a box bound --
    this is what makes constrained BO, not just bounded BO, relevant)
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
