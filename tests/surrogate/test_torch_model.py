import numpy as np

from titrate.surrogate.torch_model import TorchSurrogate


def _dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    bounds = np.array([[0.0, 1.0], [-2.0, 2.0]])
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(96, 2))
    y = np.sin(2.5 * X[:, 0]) + 0.35 * X[:, 1] ** 2
    return bounds, X, y


def test_torch_surrogate_predicts_with_uncertainty() -> None:
    bounds, X, y = _dataset()
    model = TorchSurrogate(bounds, hidden_sizes=(32, 32), dropout=0.1, random_state=3, device="cpu")
    history = model.fit(X, y, epochs=120, patience=20, batch_size=32)

    mean, std = model.predict(X[:8], mc_samples=20)

    assert mean.shape == (8,)
    assert std.shape == (8,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    assert (std > 0).all()
    assert history.best_epoch >= 0


def test_torch_surrogate_round_trip(tmp_path) -> None:
    bounds, X, y = _dataset()
    model = TorchSurrogate(bounds, hidden_sizes=(16, 16), dropout=0.0, random_state=4, device="cpu")
    model.fit(X, y, epochs=100, patience=15, batch_size=32)

    artifact = tmp_path / "surrogate.pt"
    model.save(artifact)
    loaded = TorchSurrogate.load(artifact, device="cpu")

    expected, _ = model.predict(X[:5], mc_samples=3)
    actual, _ = loaded.predict(X[:5], mc_samples=3)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
