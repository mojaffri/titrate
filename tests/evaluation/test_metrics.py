import numpy as np
import pandas as pd
import pytest

from titrate.environments.cstr_env import CSTREnvironment
from titrate.evaluation.metrics import (
    calibration_curve,
    constraint_violation_rate,
    experiments_to_threshold,
    score_trajectory,
    simple_regret,
    summarize_experiments_to_threshold,
)


def test_experiments_to_threshold_on_known_trace():
    trace = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
    true_optimum = 1.0
    assert experiments_to_threshold(trace, true_optimum, 0.5) == 3  # 0.6 first >= 0.5
    assert experiments_to_threshold(trace, true_optimum, 0.9) == 5
    assert np.isnan(experiments_to_threshold(trace, true_optimum, 0.99))


def test_simple_regret_treats_nan_as_full_regret():
    trace = np.array([np.nan, np.nan, 0.5])
    regret = simple_regret(trace, true_optimum=1.0)
    assert regret[0] == pytest.approx(1.0)
    assert regret[1] == pytest.approx(1.0)
    assert regret[2] == pytest.approx(0.5)


def test_constraint_violation_rate():
    feasible = np.array([True, True, False, True, False])
    assert constraint_violation_rate(feasible) == pytest.approx(0.4)


def test_score_trajectory_best_so_far_is_nondecreasing_and_feasible_only():
    env = CSTREnvironment()
    # A mix of clearly feasible (low T) and clearly infeasible (high T, long tau) points.
    X = np.array(
        [
            [400.0, 5.0, 2.0],  # likely infeasible (high impurity)
            [320.0, 0.5, 0.0],  # feasible, low yield
            [340.0, 5.0, 2.0],  # feasible, higher yield
            [325.0, 0.5, 0.0],  # feasible but worse than previous best
        ]
    )
    score = score_trajectory(env, X)
    valid = score.best_feasible_so_far[~np.isnan(score.best_feasible_so_far)]
    assert np.all(np.diff(valid) >= -1e-9)
    for i in range(len(X)):
        if not np.isnan(score.best_feasible_so_far[i]):
            assert score.is_feasible[: i + 1].any()


def test_summarize_experiments_to_threshold_shape():
    df = pd.DataFrame(
        {
            "strategy": ["s1"] * 5 + ["s2"] * 5,
            "seed": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "iteration": [1, 2, 3, 4, 5] * 2,
            "best_feasible_so_far": [0.1, 0.3, 0.5, 0.8, 0.95, np.nan, 0.2, 0.4, 0.6, 0.9],
        }
    )
    summary = summarize_experiments_to_threshold(df, true_optimum=1.0, fractions=(0.9,))
    assert set(summary["strategy"]) == {"s1", "s2"}
    assert "experiments_to_90pct" in summary.columns
    assert summary.loc[summary["strategy"] == "s1", "experiments_to_90pct"].iloc[0] == 5


def test_calibration_curve_perfect_gaussian_is_well_calibrated():
    rng = np.random.default_rng(0)
    n = 5000
    true_mean = rng.normal(0, 1, n)
    true_std = np.full(n, 1.0)
    y_test = true_mean + rng.normal(0, 1, n)  # matches the assumed std exactly

    def predict_fn(_):
        return true_mean, true_std

    empirical = calibration_curve(predict_fn, np.zeros(n), y_test, nominal_coverages=(0.5, 0.9))
    assert empirical[0.5] == pytest.approx(0.5, abs=0.03)
    assert empirical[0.9] == pytest.approx(0.9, abs=0.03)
