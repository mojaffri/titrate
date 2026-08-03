"""Benchmark scoring metrics.

Every strategy under test picks points using noisy observations, exactly as
it would in real experimentation. Scoring, however, always re-evaluates the
ground-truth (noiseless) simulator at the chosen points. This matters: a
strategy that got a lucky high noise draw should not look better than it
really is -- ground-truth scoring mirrors what would happen if you actually
tried to reproduce the recommended condition in the lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from titrate.environments.base import ExperimentEnvironment


@dataclass
class TrajectoryScore:
    true_objectives: np.ndarray  # ground-truth objective at each queried x
    true_constraints: np.ndarray  # ground-truth constraint value at each queried x
    is_feasible: np.ndarray  # ground-truth feasibility at each queried x
    best_feasible_so_far: np.ndarray  # running best feasible ground-truth objective (NaN until the first feasible point)


def score_trajectory(env: ExperimentEnvironment, X: np.ndarray) -> TrajectoryScore:
    true_objectives = np.array([env.evaluate_noiseless(x).objective for x in X])
    true_constraints = np.array([env.evaluate_noiseless(x).constraint_value for x in X])
    is_feasible = true_constraints <= env.constraint_max

    best_so_far = np.full(len(X), np.nan)
    running_best = -np.inf
    for i in range(len(X)):
        if is_feasible[i] and true_objectives[i] > running_best:
            running_best = true_objectives[i]
        if running_best > -np.inf:
            best_so_far[i] = running_best

    return TrajectoryScore(true_objectives, true_constraints, is_feasible, best_so_far)


def experiments_to_threshold(
    best_feasible_so_far: np.ndarray, true_optimum: float, fraction: float
) -> float:
    """First 1-indexed experiment count at which best_feasible_so_far reaches
    `fraction` of true_optimum. Returns NaN if never reached within budget."""
    target = fraction * true_optimum
    reached = np.where(best_feasible_so_far >= target)[0]
    if len(reached) == 0:
        return float("nan")
    return float(reached[0] + 1)


def simple_regret(best_feasible_so_far: np.ndarray, true_optimum: float) -> np.ndarray:
    """true_optimum - best_feasible_so_far at each iteration. Iterations with
    no feasible point yet (NaN) are scored at full regret (zero credit)."""
    filled = np.where(np.isnan(best_feasible_so_far), 0.0, best_feasible_so_far)
    return true_optimum - filled


def constraint_violation_rate(is_feasible: np.ndarray) -> float:
    return float(1.0 - is_feasible.mean())


def calibration_curve(
    predict_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    nominal_coverages: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> dict[float, float]:
    """For each nominal central-interval coverage, the empirical fraction of
    held-out points that actually fall inside the GP's predictive interval.
    A well-calibrated GP has empirical coverage close to the nominal value."""
    mean, std = predict_fn(X_test)
    std = np.maximum(std, 1e-9)
    empirical = {}
    for coverage in nominal_coverages:
        z = norm.ppf(0.5 + coverage / 2.0)
        lower, upper = mean - z * std, mean + z * std
        covered = (y_test >= lower) & (y_test <= upper)
        empirical[coverage] = float(np.mean(covered))
    return empirical


def summarize_experiments_to_threshold(
    trials: pd.DataFrame,
    true_optimum: float,
    fractions: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """One row per (strategy, seed) trial: experiments-to-X% for each
    fraction, plus the final best feasible yield reached."""
    rows = []
    for (strategy, seed), group in trials.groupby(["strategy", "seed"]):
        group = group.sort_values("iteration")
        best_so_far = group["best_feasible_so_far"].to_numpy()
        row = {"strategy": strategy, "seed": seed}
        for fraction in fractions:
            row[f"experiments_to_{int(fraction * 100)}pct"] = experiments_to_threshold(
                best_so_far, true_optimum, fraction
            )
        final = best_so_far[-1] if len(best_so_far) else np.nan
        row["final_best_feasible_yield"] = final
        rows.append(row)
    return pd.DataFrame(rows)
