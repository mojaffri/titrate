import numpy as np

from titrate.evaluation.model_comparison import (
    build_cstr_dataset,
    build_learning_curve,
    compare_surrogates,
    held_out_split,
)


def test_held_out_comparison_reports_accuracy_and_uncertainty() -> None:
    bounds = np.array([[0.0, 1.0], [-1.0, 1.0]])
    rng = np.random.default_rng(9)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(60, 2))
    y = np.sin(2.0 * X[:, 0]) + 0.2 * X[:, 1]
    X_train, X_test, y_train, y_test = held_out_split(X, y, seed=9)

    result = compare_surrogates(
        bounds,
        X_train,
        y_train,
        X_test,
        y_test,
        seed=9,
        torch_epochs=25,
        torch_hidden_sizes=(16, 16),
        torch_patience=8,
        mc_samples=8,
        gp_restarts=0,
    )

    assert {metric.model for metric in result.metrics} == {"Gaussian process", "PyTorch MLP"}
    assert all(np.isfinite(metric.rmse) for metric in result.metrics)
    assert all(np.isfinite(metric.r2) for metric in result.metrics)
    assert all(0.0 <= metric.interval_95_coverage <= 1.0 for metric in result.metrics)
    assert result.gp_mean.shape == y_test.shape
    assert result.torch_std.shape == y_test.shape


def test_learning_curve_uses_requested_valid_sizes() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(4)
    X = rng.uniform(0.0, 1.0, size=(48, 2))
    y = X[:, 0] ** 2 + X[:, 1]

    points = build_learning_curve(
        bounds,
        X[:36],
        y[:36],
        X[36:],
        y[36:],
        (12, 24, 100),
        seed=4,
        torch_epochs=12,
        mc_samples=5,
    )

    assert {point.train_samples for point in points} == {12, 24}
    assert len(points) == 4


def test_cstr_dataset_is_reproducible_and_noiseless() -> None:
    first_X, first_y = build_cstr_dataset(25, seed=11)
    second_X, second_y = build_cstr_dataset(25, seed=11)

    np.testing.assert_allclose(first_X, second_X)
    np.testing.assert_allclose(first_y, second_y)
