"""Shared interface for simulated and data-backed experiment environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution


@dataclass(frozen=True)
class EvaluationResult:
    """Objective and constraint values returned by one experiment."""

    objective: float
    constraint_value: float


@dataclass(frozen=True)
class OptimumInfo:
    """Offline optimum used as the benchmark scoring reference."""

    x: np.ndarray
    objective: float
    constraint_value: float


class ExperimentEnvironment(ABC):
    """Interface queried by optimization and benchmark routines."""

    dimension_names: tuple[str, ...]
    bounds: np.ndarray  # shape (n_dims, 2), rows are (low, high)
    constraint_max: float
    constraint_name: str
    objective_direction: str = "maximize"
    constraint_operator: str = "<="

    def __init__(self) -> None:
        self._true_optimum_cache: OptimumInfo | None = None

    @abstractmethod
    def evaluate(self, x: np.ndarray, rng: np.random.Generator) -> EvaluationResult:
        """Evaluate one point with the environment's observation model."""

    @abstractmethod
    def evaluate_noiseless(self, x: np.ndarray) -> EvaluationResult:
        """Evaluate one point without observation noise for benchmark scoring."""

    @property
    def n_dims(self) -> int:
        return len(self.dimension_names)

    def is_feasible(self, constraint_value: float) -> bool:
        if self.constraint_operator == "<=":
            return constraint_value <= self.constraint_max
        if self.constraint_operator == ">=":
            return constraint_value >= self.constraint_max
        raise ValueError(f"Unsupported constraint operator: {self.constraint_operator}")

    def feasible_mask(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.constraint_operator == "<=":
            return values <= self.constraint_max
        return values >= self.constraint_max

    def objective_score(self, values: np.ndarray | float) -> np.ndarray | float:
        """Return objective values on a common higher-is-better scale."""
        return values if self.objective_direction == "maximize" else -np.asarray(values)

    def true_optimum(self) -> OptimumInfo:
        """Compute and cache the constrained optimum of the noiseless environment."""
        if self._true_optimum_cache is not None:
            return self._true_optimum_cache

        def negative_objective(x: np.ndarray) -> float:
            value = self.evaluate_noiseless(np.asarray(x)).objective
            return -float(self.objective_score(value))

        def constraint_fn(x: np.ndarray) -> float:
            return self.evaluate_noiseless(np.asarray(x)).constraint_value

        if self.constraint_operator == "<=":
            constraint = NonlinearConstraint(constraint_fn, -np.inf, self.constraint_max)
        else:
            constraint = NonlinearConstraint(constraint_fn, self.constraint_max, np.inf)
        result = differential_evolution(
            negative_objective,
            bounds=self.bounds,
            constraints=(constraint,),
            seed=0,
            tol=1e-10,
            polish=True,
            maxiter=2000,
        )
        best_eval = self.evaluate_noiseless(result.x)
        self._true_optimum_cache = OptimumInfo(
            x=result.x,
            objective=best_eval.objective,
            constraint_value=best_eval.constraint_value,
        )
        return self._true_optimum_cache
