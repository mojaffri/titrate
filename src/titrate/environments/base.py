"""Common interface every 'experiment source' implements.

The physics-based CSTR simulator (V1) and, later, a real published HTE
dataset (V2) both implement this interface. The optimization and benchmark
code only ever talks to an ExperimentEnvironment, so swapping the underlying
experiment source requires zero changes to the BO loop, acquisition
function, or benchmark harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution


@dataclass(frozen=True)
class EvaluationResult:
    """One experiment's outcome: the objective plus its paired constraint."""

    objective: float
    constraint_value: float


@dataclass(frozen=True)
class OptimumInfo:
    """The best feasible point, found offline, used only for benchmark scoring."""

    x: np.ndarray
    objective: float
    constraint_value: float


class ExperimentEnvironment(ABC):
    """Abstract 'black box' an optimization strategy queries one point at a time."""

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
        """Run one noisy 'experiment' at x. This is what an optimizer sees."""

    @abstractmethod
    def evaluate_noiseless(self, x: np.ndarray) -> EvaluationResult:
        """Ground-truth evaluation. Used only for benchmark scoring / optimum
        search -- never exposed to an optimizer under test."""

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
        return values <= self.constraint_max if self.constraint_operator == "<=" else values >= self.constraint_max

    def objective_score(self, values: np.ndarray | float) -> np.ndarray | float:
        """Return values on a common higher-is-better scale."""
        return values if self.objective_direction == "maximize" else -np.asarray(values)

    def true_optimum(self) -> OptimumInfo:
        """Global constrained optimum of the noiseless objective, computed once
        via differential evolution and cached. This is the scoring reference
        for every benchmark metric -- no optimizer under test ever sees it."""
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
