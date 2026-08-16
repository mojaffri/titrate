import numpy as np
import torch

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
    expected_reference = model.reference_distribution
    actual_reference = loaded.reference_distribution
    np.testing.assert_allclose(actual_reference[0], expected_reference[0])
    np.testing.assert_allclose(actual_reference[1], expected_reference[1])
    assert loaded.artifact_version == 2


def test_torch_surrogate_loads_legacy_v1_artifact(tmp_path) -> None:
    bounds, X, y = _dataset()
    model = TorchSurrogate(bounds, hidden_sizes=(8, 8), dropout=0.0, random_state=5, device="cpu")
    model.fit(X, y, epochs=20, patience=5, batch_size=32)
    current_artifact = tmp_path / "current.pt"
    legacy_artifact = tmp_path / "legacy.pt"
    model.save(current_artifact)

    checkpoint = torch.load(current_artifact, map_location="cpu", weights_only=False)
    checkpoint["artifact_version"] = 1
    checkpoint.pop("reference_x_mean")
    checkpoint.pop("reference_x_std")
    torch.save(checkpoint, legacy_artifact)

    legacy = TorchSurrogate.load(legacy_artifact, device="cpu")
    reference_mean, reference_std = legacy.reference_distribution
    assert legacy.artifact_version == 1
    np.testing.assert_allclose(reference_mean, 0.5)
    np.testing.assert_allclose(reference_std, np.sqrt(1.0 / 12.0))
