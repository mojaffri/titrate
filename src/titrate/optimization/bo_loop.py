"""Sequential constrained Bayesian optimization driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from titrate.baselines.lhs import propose as lhs_propose
from titrate.environments.base import ExperimentEnvironment
from titrate.optimization.acquisition import (
    constrained_expected_improvement,
    expected_improvement,
    probability_of_feasibility,
)
from titrate.surrogate.gp_model import GPSurrogate


@dataclass
class BOResult:
    X: np.ndarray
    objectives: np.ndarray
    constraints: np.ndarray


@dataclass
class Recommendation:
    x: np.ndarray
    acquisition_value: float
    y_best: float
    gp_objective: GPSurrogate
    gp_constraint: GPSurrogate | None
    predicted_mean: float
    predicted_std: float
    probability_feasible: float


def maximize_acquisition(
    acquisition_fn: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    rng: np.random.Generator,
    n_restarts: int = 10,
    observed_X: np.ndarray | None = None,
    min_distance: float = 1e-3,
    n_fallback_candidates: int = 1024,
) -> tuple[np.ndarray, float]:
    """Maximize an acquisition function with local restarts and a sampled fallback."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_dims, 2)")
    if np.any(~np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each bound must contain finite low < high values")
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1")
    if n_fallback_candidates < 1:
        raise ValueError("n_fallback_candidates must be at least 1")
    if min_distance < 0:
        raise ValueError("min_distance cannot be negative")

    observations = None if observed_X is None else np.asarray(observed_X, dtype=float)
    if observations is not None:
        if observations.ndim != 2 or observations.shape[1] != bounds.shape[0]:
            raise ValueError("observed_X must have shape (n_observations, n_dims)")
        if np.any(~np.isfinite(observations)):
            raise ValueError("observed_X must contain only finite values")

    scipy_bounds = list(zip(bounds[:, 0], bounds[:, 1]))
    best_x: np.ndarray | None = None
    best_value = -np.inf
    start_points = lhs_propose(bounds, n_restarts, rng)

    for x0 in start_points:
        result = minimize(
            lambda x: -acquisition_fn(x),
            x0,
            bounds=scipy_bounds,
            method="L-BFGS-B",
        )
        value = -float(result.fun) if np.isfinite(result.fun) else -np.inf
        if result.success and value > best_value:
            best_value = value
            best_x = np.asarray(result.x, dtype=float)

    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-12)
    candidate_is_novel = True
    if observations is not None and len(observations) and best_x is not None:
        candidate_is_novel = bool(
            np.linalg.norm((best_x - observations) / span, axis=1).min() >= min_distance
        )

    if best_x is not None and candidate_is_novel:
        return np.clip(best_x, bounds[:, 0], bounds[:, 1]), best_value

    candidates = lhs_propose(bounds, n_fallback_candidates, rng)
    if observations is not None and len(observations):
        distances = np.linalg.norm(
            (candidates[:, None, :] - observations[None, :, :]) / span,
            axis=2,
        ).min(axis=1)
        novel = distances >= min_distance
        if novel.any():
            candidates = candidates[novel]

    values = np.array([acquisition_fn(x) for x in candidates], dtype=float)
    values[~np.isfinite(values)] = -np.inf
    if values.size and np.isfinite(values).any():
        index = int(np.argmax(values))
        best_x = candidates[index]
        best_value = float(values[index])

    if best_x is None:
        raise RuntimeError("acquisition optimization produced no finite candidate")

    return np.clip(best_x, bounds[:, 0], bounds[:, 1]), best_value


def recommend_next_point(
    env: ExperimentEnvironment,
    X: np.ndarray,
    objectives: np.ndarray,
    constraints: np.ndarray,
    rng: np.random.Generator,
    use_constraint: bool = True,
    xi: float = 0.01,
    n_acquisition_restarts: int = 10,
) -> Recommendation:
    """Fit the current surrogate models and recommend one unobserved experiment."""
    X = np.asarray(X, dtype=float)
    objectives = np.asarray(objectives, dtype=float)
    constraints = np.asarray(constraints, dtype=float)

    if X.ndim != 2 or X.shape[1] != env.bounds.shape[0]:
        raise ValueError("X must have shape (n_observations, n_dims)")
    if objectives.ndim != 1 or constraints.ndim != 1:
        raise ValueError("objectives and constraints must be one-dimensional")
    if len(X) != len(objectives) or len(X) != len(constraints):
        raise ValueError("X, objectives and constraints must contain the same number of rows")
    if len(X) < 2:
        raise ValueError("at least two observations are required before fitting a GP")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(objectives)):
        raise ValueError("X and objectives must contain only finite values")
    if n_acquisition_restarts < 1:
        raise ValueError("n_acquisition_restarts must be at least 1")
    if xi < 0:
        raise ValueError("xi cannot be negative")

    gp_objective = GPSurrogate(env.bounds).fit(X, objectives)

    feasible_mask = env.feasible_mask(constraints)
    maximize = env.objective_direction == "maximize"
    if use_constraint and feasible_mask.any():
        feasible_values = objectives[feasible_mask]
        y_best = float(feasible_values.max() if maximize else feasible_values.min())
    elif use_constraint:
        y_best = float(objectives.min() if maximize else objectives.max())
    else:
        y_best = float(objectives.max() if maximize else objectives.min())

    gp_constraint = None
    if use_constraint:
        if np.any(~np.isfinite(constraints)) and np.isfinite(env.constraint_max):
            raise ValueError("constraints must be finite for a finite engineering constraint")
        gp_constraint = GPSurrogate(env.bounds).fit(X, constraints)

        def acquisition(x: np.ndarray) -> float:
            mean, std = gp_objective.predict(np.atleast_2d(x))
            c_mean, c_std = gp_constraint.predict(np.atleast_2d(x))
            value = constrained_expected_improvement(
                mean,
                std,
                y_best,
                c_mean,
                c_std,
                env.constraint_max,
                xi,
                maximize=maximize,
                constraint_operator=env.constraint_operator,
            )
            return float(value[0])

    else:

        def acquisition(x: np.ndarray) -> float:
            mean, std = gp_objective.predict(np.atleast_2d(x))
            value = expected_improvement(mean, std, y_best, xi, maximize=maximize)
            return float(value[0])

    x_next, acquisition_value = maximize_acquisition(
        acquisition,
        env.bounds,
        rng,
        n_acquisition_restarts,
        observed_X=X,
    )
    predicted_mean, predicted_std = gp_objective.predict(np.atleast_2d(x_next))
    probability_feasible = 1.0
    if gp_constraint is not None and np.isfinite(env.constraint_max):
        c_mean, c_std = gp_constraint.predict(np.atleast_2d(x_next))
        probability_feasible = float(
            probability_of_feasibility(
                c_mean,
                c_std,
                env.constraint_max,
                operator=env.constraint_operator,
            )[0]
        )

    return Recommendation(
        x=x_next,
        acquisition_value=acquisition_value,
        y_best=y_best,
        gp_objective=gp_objective,
        gp_constraint=gp_constraint,
        predicted_mean=float(predicted_mean[0]),
        predicted_std=float(predicted_std[0]),
        probability_feasible=probability_feasible,
    )


def run_bo(
    env: ExperimentEnvironment,
    n_initial: int,
    n_iterations: int,
    rng: np.random.Generator,
    use_constraint: bool = True,
    xi: float = 0.01,
    n_acquisition_restarts: int = 10,
) -> BOResult:
    """Run an LHS bootstrap followed by sequential Bayesian optimization."""
    if n_initial < 2:
        raise ValueError("n_initial must be at least 2")
    if n_iterations < 0:
        raise ValueError("n_iterations cannot be negative")
    if n_acquisition_restarts < 1:
        raise ValueError("n_acquisition_restarts must be at least 1")
    if xi < 0:
        raise ValueError("xi cannot be negative")

    X, objectives, constraints = _bootstrap(env, n_initial, rng)

    for _ in range(n_iterations):
        recommendation = recommend_next_point(
            env,
            X,
            objectives,
            constraints,
            rng,
            use_constraint,
            xi,
            n_acquisition_restarts,
        )
        result = env.evaluate(recommendation.x, rng)

        X = np.vstack([X, recommendation.x])
        objectives = np.append(objectives, result.objective)
        constraints = np.append(constraints, result.constraint_value)

    return BOResult(X=X, objectives=objectives, constraints=constraints)


def _bootstrap(
    env: ExperimentEnvironment,
    n_initial: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = lhs_propose(env.bounds, n_initial, rng)
    objectives = np.empty(n_initial)
    constraints = np.empty(n_initial)
    for i, x in enumerate(X):
        result = env.evaluate(x, rng)
        objectives[i] = result.objective
        constraints[i] = result.constraint_value
    return X, objectives, constraints
