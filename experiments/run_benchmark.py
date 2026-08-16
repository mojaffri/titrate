"""Regenerates every number and figure in results/ from scratch.

Usage:
    python experiments/run_benchmark.py

This is the single source of truth for the README's reported results: no
number in the README is hand-typed from anywhere else. Runs are seeded and
deterministic, so re-running this script reproduces results/*.csv exactly
(figures may differ by a few pixels of matplotlib layout only).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from titrate.environments.cstr_env import CSTREnvironment  # noqa: E402
from titrate.evaluation.benchmark import run_benchmark  # noqa: E402
from titrate.evaluation.metrics import (  # noqa: E402
    calibration_curve,
    constraint_violation_rate,
    summarize_experiments_to_threshold,
)
from titrate.evaluation.plotting import (  # noqa: E402
    plot_calibration,
    plot_convergence,
    plot_experiments_to_threshold,
    plot_final_yield_distribution,
)
from titrate.surrogate.gp_model import GPSurrogate  # noqa: E402

BUDGET = 25
N_SEEDS = 40
BO_N_INITIAL = 5
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_calibration_check(env: CSTREnvironment, rng: np.random.Generator) -> dict[float, float]:
    """Fit a GP on a modest random training set, then check whether its
    predictive intervals are calibrated against a large held-out test set."""
    n_train, n_test = 25, 300
    X_train = rng.uniform(env.bounds[:, 0], env.bounds[:, 1], size=(n_train, env.n_dims))
    y_train = np.array([env.evaluate(x, rng).objective for x in X_train])

    X_test = rng.uniform(env.bounds[:, 0], env.bounds[:, 1], size=(n_test, env.n_dims))
    y_test = np.array([env.evaluate_noiseless(x).objective for x in X_test])

    gp = GPSurrogate(env.bounds).fit(X_train, y_train)
    return calibration_curve(gp.predict, X_test, y_test)


def main() -> None:
    warnings.filterwarnings("ignore", category=Warning, module="sklearn")
    RESULTS_DIR.mkdir(exist_ok=True)

    env = CSTREnvironment()
    optimum = env.true_optimum()
    print(f"True constrained optimum: yield={optimum.objective:.4f} at x={optimum.x}")
    print(f"  impurity at optimum: {optimum.constraint_value:.4f} (max {env.constraint_max})")

    start = time.time()
    trials = run_benchmark(env, budget=BUDGET, n_seeds=N_SEEDS, bo_n_initial=BO_N_INITIAL)
    elapsed = time.time() - start
    print(f"Benchmark finished in {elapsed:.1f}s ({len(trials)} rows)")

    trials.to_csv(RESULTS_DIR / "benchmark_trials.csv", index=False)

    summary = summarize_experiments_to_threshold(trials, true_optimum=optimum.objective)
    summary.to_csv(RESULTS_DIR / "benchmark_summary.csv", index=False)

    headline = {}
    for strategy, group in summary.groupby("strategy"):
        wall_clock = trials.loc[trials["strategy"] == strategy, "wall_clock_seconds"]
        headline[strategy] = {
            "median_experiments_to_90pct": float(group["experiments_to_90pct"].median()),
            "median_experiments_to_95pct": float(group["experiments_to_95pct"].median()),
            "median_experiments_to_99pct": float(group["experiments_to_99pct"].median()),
            "median_final_best_feasible_yield": float(group["final_best_feasible_yield"].median()),
            "constraint_violation_rate": constraint_violation_rate(
                trials.loc[trials["strategy"] == strategy, "is_feasible"].to_numpy()
            ),
            "mean_wall_clock_seconds_per_trial": float(wall_clock.groupby(trials["seed"]).first().mean())
            if not wall_clock.empty
            else None,
            "n_seeds_reached_90pct": int(group["experiments_to_90pct"].notna().sum()),
            "n_seeds": int(len(group)),
        }

    calibration = run_calibration_check(env, np.random.default_rng(2024))
    headline["gp_calibration"] = {str(k): v for k, v in calibration.items()}
    headline["true_optimum"] = {
        "yield": optimum.objective,
        "x": optimum.x.tolist(),
        "dimension_names": list(env.dimension_names),
        "impurity_at_optimum": optimum.constraint_value,
        "impurity_max": env.constraint_max,
    }
    headline["benchmark_config"] = {
        "budget": BUDGET,
        "n_seeds": N_SEEDS,
        "bo_n_initial": BO_N_INITIAL,
    }
    (RESULTS_DIR / "headline_results.json").write_text(json.dumps(headline, indent=2))

    plot_convergence(trials, optimum.objective, RESULTS_DIR / "convergence.png")
    plot_experiments_to_threshold(summary, 95, RESULTS_DIR / "experiments_to_95pct.png")
    plot_final_yield_distribution(summary, optimum.objective, RESULTS_DIR / "final_yield_distribution.png")
    plot_calibration(calibration, RESULTS_DIR / "gp_calibration.png")

    print("\n=== Headline results (median over seeds) ===")
    for strategy in ("random", "grid", "lhs", "bo_unconstrained", "bo_constrained"):
        row = headline[strategy]
        print(
            f"{strategy:20s} exp->90%: {row['median_experiments_to_90pct']:.1f}  "
            f"exp->95%: {row['median_experiments_to_95pct']:.1f}  "
            f"final yield: {row['median_final_best_feasible_yield']:.4f}  "
            f"constraint violation rate: {row['constraint_violation_rate']:.3f}"
        )
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
