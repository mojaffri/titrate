import numpy as np
import pandas as pd
import pytest

from titrate.environments.tabular_env import TabularEmulatorEnvironment
from titrate.environments.validation import validate_experiment_table


def test_constant_input_is_rejected() -> None:
    data = pd.DataFrame({"x": np.ones(10), "y": np.arange(10.0)})
    with pytest.raises(ValueError, match="must vary"):
        validate_experiment_table(data, ["x"], "y")


def test_nonfinite_rows_are_reported_and_duplicates_are_averaged() -> None:
    data = pd.DataFrame(
        {"x": [0, 0, 1, 2, 3, 4, 5, 6, np.inf], "y": [1, 3, 2, 3, 4, 5, 6, 7, 8]}
    )
    clean, health = validate_experiment_table(data, ["x"], "y")
    assert len(clean) == 8
    assert health.dropped_rows == 1
    assert health.duplicate_rows == 1
    env = TabularEmulatorEnvironment(data, ["x"], "y")
    assert len(env._X_train) == 7


def test_minimize_and_greater_than_constraint_semantics() -> None:
    data = pd.DataFrame({"x": np.arange(8.0), "cost": np.arange(8.0) ** 2, "purity": np.arange(8.0)})
    env = TabularEmulatorEnvironment(
        data,
        ["x"],
        "cost",
        constraint_column="purity",
        constraint_max=4.0,
        objective_direction="minimize",
        constraint_operator=">=",
    )
    assert env.is_feasible(5.0)
    assert not env.is_feasible(3.0)
    assert env.objective_score(2.0) > env.objective_score(3.0)
