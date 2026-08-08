import numpy as np
import pytest

from titrate.environments.reizman_suzuki_env import ReizmanSuzukiEnvironment


@pytest.fixture(scope="module")
def env() -> ReizmanSuzukiEnvironment:
    return ReizmanSuzukiEnvironment()


def test_loads_real_data_with_expected_shape(env: ReizmanSuzukiEnvironment):
    assert env.n_real_experiments > 30  # P1-L4 subset has 37 real experiments
    assert env.n_dims == 3
    assert env.bounds.shape == (3, 2)


def test_unconstrained_by_construction(env: ReizmanSuzukiEnvironment):
    assert env.constraint_max == float("inf")
    result = env.evaluate_noiseless(env.bounds.mean(axis=1))
    assert env.is_feasible(result.constraint_value)


def test_evaluate_noiseless_within_unit_interval(env: ReizmanSuzukiEnvironment):
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(env.bounds[:, 0], env.bounds[:, 1])
        result = env.evaluate_noiseless(x)
        assert 0.0 <= result.objective <= 1.0


def test_evaluate_is_stochastic(env: ReizmanSuzukiEnvironment):
    x = env.bounds.mean(axis=1)
    rng = np.random.default_rng(0)
    values = {env.evaluate(x, rng).objective for _ in range(20)}
    assert len(values) > 1


def test_emulator_holdout_rmse_is_reasonable(env: ReizmanSuzukiEnvironment):
    """Sanity bound, not a tight assertion: with 37 real points and real
    experimental noise, some regression error is expected and fine -- this
    just guards against something being badly broken (e.g. RMSE near 0.5,
    which would mean the emulator learned nothing)."""
    rmse = env.emulator_holdout_rmse()
    assert 0.0 < rmse < 0.35


def test_true_optimum_within_bounds(env: ReizmanSuzukiEnvironment):
    optimum = env.true_optimum()
    for value, (low, high) in zip(optimum.x, env.bounds):
        assert low - 1e-6 <= value <= high + 1e-6
    assert 0.0 <= optimum.objective <= 1.0
