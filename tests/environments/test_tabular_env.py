import numpy as np
import pandas as pd
import pytest

from titrate.environments.tabular_env import TabularEmulatorEnvironment


def make_data(n=30, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-5, 5, n)
    y = -((x1 - 5) ** 2) - (x2 ** 2) + rng.normal(0, 0.5, n)  # smooth, has an interior max
    c = x1 + x2  # arbitrary constraint quantity
    return pd.DataFrame({"a": x1, "b": x2, "yield": y, "impurity": c})


def test_basic_fit_and_bounds():
    data = make_data()
    env = TabularEmulatorEnvironment(data, input_columns=["a", "b"], objective_column="yield")
    assert env.n_dims == 2
    assert env.n_real_experiments == len(data)
    assert env.bounds[0, 0] == pytest.approx(data["a"].min())
    assert env.bounds[0, 1] == pytest.approx(data["a"].max())


def test_dimension_names_default_to_input_columns():
    data = make_data()
    env = TabularEmulatorEnvironment(data, input_columns=["a", "b"], objective_column="yield")
    assert env.dimension_names == ("a", "b")


def test_custom_dimension_names():
    data = make_data()
    env = TabularEmulatorEnvironment(
        data, input_columns=["a", "b"], objective_column="yield", dimension_names=("Alpha", "Beta")
    )
    assert env.dimension_names == ("Alpha", "Beta")


def test_no_constraint_column_means_unconstrained():
    data = make_data()
    env = TabularEmulatorEnvironment(data, input_columns=["a", "b"], objective_column="yield")
    assert env.constraint_max == float("inf")
    result = env.evaluate_noiseless(np.array([5.0, 0.0]))
    assert env.is_feasible(result.constraint_value)


def test_constraint_column_is_used():
    data = make_data()
    env = TabularEmulatorEnvironment(
        data,
        input_columns=["a", "b"],
        objective_column="yield",
        constraint_column="impurity",
        constraint_max=2.0,
    )
    assert env.constraint_max == 2.0
    assert env.constraint_name == "impurity"


def test_clip_range_bounds_output():
    data = make_data()
    env = TabularEmulatorEnvironment(
        data, input_columns=["a", "b"], objective_column="yield", clip_range=(-10.0, 0.0)
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.uniform(env.bounds[:, 0], env.bounds[:, 1])
        result = env.evaluate_noiseless(x)
        assert -10.0 <= result.objective <= 0.0


def test_no_clip_range_allows_any_value():
    data = make_data()
    env = TabularEmulatorEnvironment(data, input_columns=["a", "b"], objective_column="yield")
    assert env.clip_range is None


def test_recovers_approximate_interior_optimum():
    """The synthetic function peaks near (5, 0) -- the emulator's optimum
    should land in the right neighborhood, not require exact recovery."""
    data = make_data(n=60)
    env = TabularEmulatorEnvironment(data, input_columns=["a", "b"], objective_column="yield")
    optimum = env.true_optimum()
    assert 2.0 < optimum.x[0] < 8.0
    assert -3.0 < optimum.x[1] < 3.0


def test_single_input_column_works():
    data = make_data()
    env = TabularEmulatorEnvironment(data, input_columns=["a"], objective_column="yield")
    assert env.n_dims == 1
    result = env.evaluate_noiseless(np.array([5.0]))
    assert isinstance(result.objective, float)
