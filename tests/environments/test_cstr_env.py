import numpy as np
import pytest

from titrate.environments.cstr_env import CSTREnvironment


@pytest.fixture
def env() -> CSTREnvironment:
    return CSTREnvironment()


def test_bounds_shape_matches_dimensions(env: CSTREnvironment):
    assert env.bounds.shape == (env.n_dims, 2)
    assert env.n_dims == 3


def test_evaluate_noiseless_matches_evaluate_in_expectation(env: CSTREnvironment):
    x = np.array([360.0, 1.0, 1.0])
    clean = env.evaluate_noiseless(x)
    rng = np.random.default_rng(0)
    samples = [env.evaluate(x, rng).objective for _ in range(4000)]
    assert np.mean(samples) == pytest.approx(clean.objective, abs=0.01)


def test_evaluate_is_stochastic(env: CSTREnvironment):
    x = np.array([360.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    values = {env.evaluate(x, rng).objective for _ in range(20)}
    assert len(values) > 1


def test_evaluate_respects_output_bounds(env: CSTREnvironment):
    x = np.array([400.0, 5.0, 2.0])
    rng = np.random.default_rng(0)
    for _ in range(200):
        result = env.evaluate(x, rng)
        assert 0.0 <= result.objective <= 1.0
        assert result.constraint_value >= 0.0


def test_true_optimum_is_feasible(env: CSTREnvironment):
    opt = env.true_optimum()
    assert env.is_feasible(opt.constraint_value)


def test_true_optimum_is_within_bounds(env: CSTREnvironment):
    opt = env.true_optimum()
    for value, (low, high) in zip(opt.x, env.bounds):
        assert low - 1e-6 <= value <= high + 1e-6


def test_true_optimum_beats_a_dense_feasible_grid(env: CSTREnvironment):
    """The DE-found optimum should not be beaten by any point on a dense
    feasible grid -- a cheap regression check against the analytic model."""
    opt = env.true_optimum()
    Ts = np.linspace(*env.bounds[0], 40)
    taus = np.linspace(*env.bounds[1], 40)
    cats = np.linspace(*env.bounds[2], 15)
    best_grid = -1.0
    for T in Ts:
        for tau in taus:
            for c in cats:
                result = env.evaluate_noiseless(np.array([T, tau, c]))
                if result.constraint_value <= env.constraint_max:
                    best_grid = max(best_grid, result.objective)
    assert opt.objective >= best_grid - 1e-3


def test_true_optimum_is_cached(env: CSTREnvironment):
    first = env.true_optimum()
    second = env.true_optimum()
    assert first is second


def test_constraint_binds_at_optimum(env: CSTREnvironment):
    """Sanity check on the chosen impurity_max: it should actually change the
    answer (constraint active), not be a no-op bound."""
    opt = env.true_optimum()
    assert opt.constraint_value == pytest.approx(env.constraint_max, abs=1e-3)
