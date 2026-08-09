"""Sequential (constrained) Bayesian optimization driver.

Algorithm, at each iteration:
  1. Fit a GP on all objective observations so far (and a second GP on
     constraint observations, if constraint-aware).
  2. Compute y_best: the best *feasible* observed objective. If no feasible
     point has been observed yet, fall back to the worst observed objective
     so the acquisition is dominated by the feasibility term (Step 3) and
     actively searches for a feasible region rather than degenerating.
  3. Maximize the acquisition function (constrained EI, or plain EI if
     constraint-unaware) over the bounds via multi-start L-BFGS-B.
  4. Query the environment at that point (adds real measurement noise),
     append to the dataset, repeat.

The first `n_initial` points come from a Latin Hypercube bootstrap (BO needs
some data before a GP fit is meaningful).

Steps 1-3 are exposed separately as `recommend_next_point` so a caller (e.g.
the web demo) can show "here's what we'd run next and why" before actually
spending an experiment on it.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    X: np.ndarray  # shape (n_total, n_dims), all queried points in order
    objectives: np.ndarray  # noisy observed objective at each point
    constraints: np.ndarray  # noisy observed constraint value at each point


@dataclass
class Recommendation:
    x: np.ndarray  # the proposed next experiment
    acquisition_value: float  # (constrained) EI at x -- "expected improvement"
    y_best: float  # best feasible objective the recommendation is measured against
    gp_objective: GPSurrogate
    gp_constraint: GPSurrogate | None  # None when use_constraint=False
    predicted_mean: float
    predicted_std: float
    probability_feasible: float


def maximize_acquisition(
    acquisition_fn,
    bounds: np.ndarray,
    rng: np.random.Generator,
    n_restarts: int = 10,
    observed_X: np.ndarray | None = None,
    min_distance: float = 1e-3,
    n_fallback_candidates: int = 1024,
) -> tuple[np.ndarray, float]:
    """Robust multi-start maximization with a finite randomized fallback."""
    scipy_bounds = list(zip(bounds[:, 0], bounds[:, 1]))
    best_x = None
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
            best_x = result.x

    candidate_is_novel = True
    if observed_X is not None and len(observed_X):
        span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-12)
        if best_x is not None:
            candidate_is_novel = bool(
                np.linalg.norm((best_x - np.asarray(observed_X)) / span, axis=1).min()
                >= min_distance
            )
    if best_x is not None and candidate_is_novel:
        return np.clip(best_x, bounds[:, 0], bounds[:, 1]), best_value

    # Preserve the established multi-start path when it succeeds. This broad
    # search is only a safety net for solver failure or a near-duplicate.
    candidates = lhs_propose(bounds, n_fallback_candidates, rng)
    if observed_X is not None and len(observed_X):
        distances = np.linalg.norm(
            (candidates[:, None, :] - np.asarray(observed_X)[None, :, :]) / span,
            axis=2,
        ).min(axis=1)
        novel = distances >= min_distance
        if novel.any():
            candidates = candidates[novel]
    values = np.array([acquisition_fn(x) for x in candidates], dtype=float)
    values[~np.isfinite(values)] = -np.inf
    if values.size and np.isfinite(values).any():
        index = int(np.argmax(values))
        best_x, best_value = candidates[index], float(values[index])
    if best_x is None:
        raise RuntimeError("Acquisition optimization produced no finite candidate.")
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
    """Fit GP(s) on the current dataset and pick the next point to query,
    without spending an experiment on it. This is the "what should I run
    next, and why" step, reused by both run_bo and the live demo."""
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
        gp_constraint = GPSurrogate(env.bounds).fit(X, constraints)

        def acquisition(x: np.ndarray) -> float:
            mean, std = gp_objective.predict(np.atleast_2d(x))
            c_mean, c_std = gp_constraint.predict(np.atleast_2d(x))
            value = constrained_expected_improvement(
                mean, std, y_best, c_mean, c_std, env.constraint_max, xi,
                maximize=maximize, constraint_operator=env.constraint_operator,
            )
            return float(value[0])
    else:

        def acquisition(x: np.ndarray) -> float:
            mean, std = gp_objective.predict(np.atleast_2d(x))
            value = expected_improvement(mean, std, y_best, xi, maximize=maximize)
            return float(value[0])

    x_next, acquisition_value = maximize_acquisition(
        acquisition, env.bounds, rng, n_acquisition_restarts, observed_X=X
    )
    predicted_mean, predicted_std = gp_objective.predict(np.atleast_2d(x_next))
    probability_feasible = 1.0
    if gp_constraint is not None and np.isfinite(env.constraint_max):
        c_mean, c_std = gp_constraint.predict(np.atleast_2d(x_next))
        probability_feasible = float(probability_of_feasibility(
            c_mean, c_std, env.constraint_max, operator=env.constraint_operator
        )[0])
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
    """Run a full BO trajectory: n_initial LHS points + n_iterations adaptive queries."""
    X, objectives, constraints = _bootstrap(env, n_initial, rng)

    for _ in range(n_iterations):
        recommendation = recommend_next_point(
            env, X, objectives, constraints, rng, use_constraint, xi, n_acquisition_restarts
        )
        result = env.evaluate(recommendation.x, rng)

        X = np.vstack([X, recommendation.x])
        objectives = np.append(objectives, result.objective)
        constraints = np.append(constraints, result.constraint_value)

    return BOResult(X=X, objectives=objectives, constraints=constraints)


def _bootstrap(
    env: ExperimentEnvironment, n_initial: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = lhs_propose(env.bounds, n_initial, rng)
    objectives = np.empty(n_initial)
    constraints = np.empty(n_initial)
    for i, x in enumerate(X):
        result = env.evaluate(x, rng)
        objectives[i] = result.objective
        constraints[i] = result.constraint_value
    return X, objectives, constraints
