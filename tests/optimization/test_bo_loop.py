import numpy as np

from titrate.environments.cstr_env import CSTREnvironment
from titrate.optimization.bo_loop import run_bo


def test_bo_loop_runs_end_to_end_and_respects_bounds():
    env = CSTREnvironment()
    rng = np.random.default_rng(0)
    result = run_bo(env, n_initial=5, n_iterations=3, rng=rng)

    assert result.X.shape == (8, 3)
    assert result.objectives.shape == (8,)
    assert result.constraints.shape == (8,)
    for dim in range(3):
        assert np.all(result.X[:, dim] >= env.bounds[dim, 0] - 1e-6)
        assert np.all(result.X[:, dim] <= env.bounds[dim, 1] + 1e-6)


def test_bo_loop_runs_unconstrained_variant():
    env = CSTREnvironment()
    rng = np.random.default_rng(1)
    result = run_bo(env, n_initial=5, n_iterations=3, rng=rng, use_constraint=False)
    assert result.X.shape == (8, 3)


def test_bo_loop_is_reproducible_with_same_seed():
    env = CSTREnvironment()
    result_a = run_bo(env, n_initial=5, n_iterations=3, rng=np.random.default_rng(7))
    result_b = run_bo(env, n_initial=5, n_iterations=3, rng=np.random.default_rng(7))
    assert np.allclose(result_a.X, result_b.X)
    assert np.allclose(result_a.objectives, result_b.objectives)


def test_bo_improves_best_feasible_objective_over_bootstrap():
    """Not a strict guarantee for any single seed, but with enough
    iterations BO should typically improve on its own LHS bootstrap."""
    env = CSTREnvironment()
    rng = np.random.default_rng(3)
    result = run_bo(env, n_initial=5, n_iterations=10, rng=rng)

    feasible = result.constraints <= env.constraint_max
    bootstrap_best = result.objectives[:5][feasible[:5]].max() if feasible[:5].any() else -np.inf
    overall_best = result.objectives[feasible].max() if feasible.any() else -np.inf
    assert overall_best >= bootstrap_best
