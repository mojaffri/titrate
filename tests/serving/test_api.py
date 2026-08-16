from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from titrate.serving import api
from titrate.surrogate.torch_model import TorchSurrogate


@pytest.fixture(scope="module")
def model_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("api-model")
    artifact = directory / "surrogate.pt"
    bounds = np.array([[320.0, 400.0], [0.1, 5.0], [0.0, 2.0]])
    rng = np.random.default_rng(17)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(56, 3))
    scaled = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    y = 0.3 + 0.2 * scaled[:, 0] + 0.4 * scaled[:, 1] - 0.1 * scaled[:, 2] ** 2
    model = TorchSurrogate(
        bounds,
        hidden_sizes=(16, 16),
        dropout=0.1,
        random_state=17,
        device="cpu",
    )
    model.fit(X, y, epochs=30, patience=8, batch_size=28)
    model.save(artifact)
    return artifact


@pytest.fixture()
def configured_client(monkeypatch: pytest.MonkeyPatch, model_artifact: Path) -> TestClient:
    monkeypatch.setenv("TITRATE_MODEL_PATH", str(model_artifact))
    api.get_model.cache_clear()
    api.prediction_monitor.reset()
    yield TestClient(api.app)
    api.get_model.cache_clear()
    api.prediction_monitor.reset()


def test_health_and_metadata_load_real_artifact(configured_client: TestClient) -> None:
    health = configured_client.get("/health")
    metadata = configured_client.get("/metadata")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model": "loaded"}
    assert metadata.status_code == 200
    assert metadata.json()["n_features"] == 3
    assert metadata.json()["artifact_version"] == 2
    assert metadata.json()["max_batch_size"] == 256
    assert len(metadata.json()["monitoring_reference"]["scaled_feature_mean"]) == 3


def test_predict_returns_batch_means_and_uncertainty(configured_client: TestClient) -> None:
    response = configured_client.post(
        "/predict",
        json={"points": [[360.0, 2.0, 1.0], [380.0, 4.0, 0.5]], "mc_samples": 12},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "pytorch-mlp-mc-dropout"
    assert len(payload["predictions"]) == 2
    assert all(np.isfinite(item["mean"]) for item in payload["predictions"])
    assert all(item["std"] > 0 for item in payload["predictions"])

    monitoring = configured_client.get("/monitoring")
    assert monitoring.status_code == 200
    assert monitoring.json()["window_size"] == 2
    assert monitoring.json()["total_predictions"] == 2
    assert monitoring.json()["mean_predictive_uncertainty"] > 0
    assert monitoring.json()["drift_detected"] is False


def test_prometheus_metrics_expose_service_and_model_signals(configured_client: TestClient) -> None:
    configured_client.post(
        "/predict",
        json={"points": [[360.0, 2.0, 1.0]], "mc_samples": 6},
    )
    response = configured_client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "titrate_http_requests_total" in response.text
    assert "titrate_http_request_duration_seconds" in response.text
    assert "titrate_predictions_total" in response.text
    assert "titrate_predictive_uncertainty" in response.text


@pytest.mark.parametrize(
    "payload",
    [
        {"points": [], "mc_samples": 10},
        {"points": [[360.0, 2.0]], "mc_samples": 10},
        {"points": [[500.0, 2.0, 1.0]], "mc_samples": 10},
        {"points": [[360.0, 2.0, 1.0]], "mc_samples": 1},
    ],
)
def test_predict_rejects_invalid_requests(configured_client: TestClient, payload: dict[str, object]) -> None:
    response = configured_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_model_is_degraded_and_inference_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TITRATE_MODEL_PATH", str(tmp_path / "missing.pt"))
    api.get_model.cache_clear()
    client = TestClient(api.app)

    assert client.get("/health").json() == {"status": "degraded", "model": "unavailable"}
    assert client.get("/metadata").status_code == 503
    assert client.post("/predict", json={"points": [[360.0, 2.0, 1.0]]}).status_code == 503
    api.get_model.cache_clear()


def test_corrupt_model_is_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corrupt_artifact = tmp_path / "corrupt.pt"
    corrupt_artifact.write_bytes(b"not a torch artifact")
    monkeypatch.setenv("TITRATE_MODEL_PATH", str(corrupt_artifact))
    api.get_model.cache_clear()
    client = TestClient(api.app)

    assert client.get("/health").json() == {"status": "degraded", "model": "unavailable"}
    assert client.get("/metadata").status_code == 503
    api.get_model.cache_clear()
