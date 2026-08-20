import numpy as np
import pytest

from titrate.environments.cstr_env import CSTREnvironment
from titrate.optimization.bo_loop import maximize_acquisition, run_bo


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
    env = CSTREnvironment()
    rng = np.random.default_rng(3)
    result = run_bo(env, n_initial=5, n_iterations=10, rng=rng)

    feasible = result.constraints <= env.constraint_max
    bootstrap_best = result.objectives[:5][feasible[:5]].max() if feasible[:5].any() else -np.inf
    overall_best = result.objectives[feasible].max() if feasible.any() else -np.inf
    assert overall_best >= bootstrap_best


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_initial": 1, "n_iterations": 1}, "n_initial"),
        ({"n_initial": 5, "n_iterations": -1}, "n_iterations"),
        ({"n_initial": 5, "n_iterations": 1, "xi": -0.1}, "xi"),
        (
            {"n_initial": 5, "n_iterations": 1, "n_acquisition_restarts": 0},
            "n_acquisition_restarts",
        ),
    ],
)
def test_bo_loop_rejects_invalid_run_parameters(kwargs, message):
    env = CSTREnvironment()
    with pytest.raises(ValueError, match=message):
        run_bo(env, rng=np.random.default_rng(0), **kwargs)


def test_acquisition_optimizer_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="low < high"):
        maximize_acquisition(
            lambda x: float(-np.sum(x**2)),
            np.array([[1.0, 1.0]]),
            np.random.default_rng(0),
        )
