"""Multi-seed benchmark harness: runs every strategy across many seeds at a
fixed experiment budget and returns one tidy DataFrame scored against
ground truth (see titrate.evaluation.metrics)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from titrate.baselines import grid_search, lhs, random_search
from titrate.environments.base import ExperimentEnvironment
from titrate.evaluation.metrics import score_trajectory
from titrate.optimization.bo_loop import run_bo

STRATEGY_NAMES = ("random", "grid", "lhs", "bo_unconstrained", "bo_constrained")


def _propose_points(strategy: str, env: ExperimentEnvironment, budget: int, seed: int, bo_n_initial: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return random_search.propose(env.bounds, budget, rng)
    if strategy == "grid":
        return grid_search.propose(env.bounds, budget, rng)
    if strategy == "lhs":
        return lhs.propose(env.bounds, budget, rng)
    if strategy == "bo_unconstrained":
        return run_bo(
            env, n_initial=bo_n_initial, n_iterations=budget - bo_n_initial, rng=rng, use_constraint=False
        ).X
    if strategy == "bo_constrained":
        return run_bo(
            env, n_initial=bo_n_initial, n_iterations=budget - bo_n_initial, rng=rng, use_constraint=True
        ).X
    raise ValueError(f"Unknown strategy: {strategy!r}")


def run_trial(
    strategy: str, env: ExperimentEnvironment, budget: int, seed: int, bo_n_initial: int = 5
) -> pd.DataFrame:
    """Run one (strategy, seed) trial and score it against ground truth.
    Returns one row per iteration (1..budget)."""
    start = time.time()
    X = _propose_points(strategy, env, budget, seed, bo_n_initial)
    elapsed = time.time() - start

    score = score_trajectory(env, X)
    n = len(X)
    return pd.DataFrame(
        {
            "strategy": strategy,
            "seed": seed,
            "iteration": np.arange(1, n + 1),
            "true_objective": score.true_objectives,
            "is_feasible": score.is_feasible,
            "best_feasible_so_far": score.best_feasible_so_far,
            "wall_clock_seconds": elapsed,
        }
    )


def run_benchmark(
    env: ExperimentEnvironment,
    budget: int,
    n_seeds: int,
    strategies: tuple[str, ...] = STRATEGY_NAMES,
    bo_n_initial: int = 5,
    seed_start: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every strategy across n_seeds seeds at the same fixed budget."""
    frames = []
    for strategy in strategies:
        for seed in range(seed_start, seed_start + n_seeds):
            if verbose:
                print(f"[benchmark] {strategy} seed={seed}")
            frames.append(run_trial(strategy, env, budget, seed, bo_n_initial))
    return pd.concat(frames, ignore_index=True)
