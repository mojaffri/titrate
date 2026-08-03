import numpy as np
import pytest

from titrate.surrogate.gp_model import GPSurrogate

BOUNDS = np.array([[320.0, 400.0], [0.1, 5.0], [0.0, 2.0]])


def make_training_data(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n, 3))
    # simple smooth synthetic function of the (scaled) inputs
    X_scaled = (X - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    y = np.sin(X_scaled[:, 0] * 3) + 0.5 * X_scaled[:, 1] - 0.2 * X_scaled[:, 2] ** 2
    return X, y


def test_predict_before_fit_raises():
    gp = GPSurrogate(BOUNDS)
    with pytest.raises(RuntimeError):
        gp.predict(np.array([[360.0, 1.0, 1.0]]))


def test_predict_output_shapes():
    X, y = make_training_data()
    gp = GPSurrogate(BOUNDS).fit(X, y)
    mean, std = gp.predict(X)
    assert mean.shape == (len(X),)
    assert std.shape == (len(X),)


def test_uncertainty_low_at_observed_points_high_far_away():
    X, y = make_training_data(n=15)
    gp = GPSurrogate(BOUNDS).fit(X, y)
    _, std_at_data = gp.predict(X)

    far_point = np.array([[BOUNDS[0, 1], BOUNDS[1, 0], BOUNDS[2, 1]]])
    _, std_far = gp.predict(far_point)

    assert std_at_data.mean() < std_far[0]


def test_fit_recovers_training_points_reasonably_well():
    X, y = make_training_data(n=25)
    gp = GPSurrogate(BOUNDS).fit(X, y)
    mean, _ = gp.predict(X)
    rmse = np.sqrt(np.mean((mean - y) ** 2))
    assert rmse < 0.3 * (y.max() - y.min())
