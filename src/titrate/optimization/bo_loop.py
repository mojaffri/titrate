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
)
from titrate.surrogate.gp_model import GPSurrogate


@dataclass
class BOResult:
    X: np.ndarray  # shape (n_total, n_dims), all queried points in order
    objectives: np.ndarray  # noisy observed objective at each point
    constraints: np.ndarray  # noisy observed constraint value at each point


def _maximize_acquisition(
    acquisition_fn,
    bounds: np.ndarray,
    rng: np.random.Generator,
    n_restarts: int = 10,
) -> np.ndarray:
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
        value = -result.fun
        if value > best_value:
            best_value = value
            best_x = result.x
    return np.clip(best_x, bounds[:, 0], bounds[:, 1])


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
        gp_objective = GPSurrogate(env.bounds).fit(X, objectives)

        feasible_mask = constraints <= env.constraint_max
        if use_constraint and feasible_mask.any():
            y_best = float(objectives[feasible_mask].max())
        elif use_constraint:
            y_best = float(objectives.min())
        else:
            y_best = float(objectives.max())

        if use_constraint:
            gp_constraint = GPSurrogate(env.bounds).fit(X, constraints)

            def acquisition(x: np.ndarray) -> float:
                mean, std = gp_objective.predict(np.atleast_2d(x))
                c_mean, c_std = gp_constraint.predict(np.atleast_2d(x))
                value = constrained_expected_improvement(
                    mean, std, y_best, c_mean, c_std, env.constraint_max, xi
                )
                return float(value[0])
        else:

            def acquisition(x: np.ndarray) -> float:
                mean, std = gp_objective.predict(np.atleast_2d(x))
                value = expected_improvement(mean, std, y_best, xi)
                return float(value[0])

        x_next = _maximize_acquisition(acquisition, env.bounds, rng, n_acquisition_restarts)
        result = env.evaluate(x_next, rng)

        X = np.vstack([X, x_next])
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
