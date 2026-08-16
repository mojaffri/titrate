"""FastAPI service for a trained PyTorch surrogate artifact.

Set TITRATE_MODEL_PATH to the .pt artifact produced by TorchSurrogate.save.
The service is intentionally small so it can run locally, in Docker, or on a
managed container platform such as AWS App Runner/ECS.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Awaitable, Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest
from pydantic import BaseModel, Field

from titrate.serving.monitoring import InferenceMonitor
from titrate.surrogate.torch_model import TorchSurrogate

METRICS_REGISTRY = CollectorRegistry()
HTTP_REQUESTS = Counter(
    "titrate_http_requests_total",
    "HTTP requests handled by the inference service.",
    ("method", "route", "status_class"),
    registry=METRICS_REGISTRY,
)
HTTP_DURATION = Histogram(
    "titrate_http_request_duration_seconds",
    "Inference-service HTTP request latency.",
    ("method", "route"),
    registry=METRICS_REGISTRY,
)
PREDICTIONS = Counter(
    "titrate_predictions_total",
    "Individual model predictions returned.",
    registry=METRICS_REGISTRY,
)
PREDICTIVE_UNCERTAINTY = Histogram(
    "titrate_predictive_uncertainty",
    "MC-dropout predictive standard deviation.",
    registry=METRICS_REGISTRY,
)
MONITOR_WINDOW_SIZE = Gauge(
    "titrate_monitor_window_size",
    "Predictions retained in the bounded monitoring window.",
    registry=METRICS_REGISTRY,
)
DRIFT_SCORE = Gauge(
    "titrate_drift_score",
    "Maximum standardized feature-mean shift in the monitoring window.",
    registry=METRICS_REGISTRY,
)
DRIFT_DETECTED = Gauge(
    "titrate_drift_detected",
    "One when the rolling input window exceeds the configured drift threshold.",
    registry=METRICS_REGISTRY,
)

prediction_monitor = InferenceMonitor(
    max_window=int(os.getenv("TITRATE_MONITOR_WINDOW", "2048")),
    minimum_drift_samples=int(os.getenv("TITRATE_DRIFT_MIN_SAMPLES", "25")),
    drift_threshold=float(os.getenv("TITRATE_DRIFT_THRESHOLD", "0.75")),
)


class PredictRequest(BaseModel):
    points: list[list[float]] = Field(min_length=1, max_length=256)
    mc_samples: int = Field(default=50, ge=2, le=500)


class Prediction(BaseModel):
    mean: float
    std: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model: str = "pytorch-mlp-mc-dropout"


@lru_cache(maxsize=1)
def get_model() -> TorchSurrogate:
    model_path = Path(os.getenv("TITRATE_MODEL_PATH", "artifacts/torch_surrogate.pt"))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path!r}. Set TITRATE_MODEL_PATH to a trained .pt file."
        )
    try:
        return TorchSurrogate.load(model_path)
    except Exception as exc:
        raise RuntimeError("The configured model artifact could not be loaded.") from exc


def _model_or_503() -> TorchSurrogate:
    try:
        return get_model()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Model unavailable.") from exc


app = FastAPI(
    title="Titrate Surrogate API",
    version="0.2.0",
    description="Uncertainty-aware inference for chemical-process surrogate models.",
)


@app.middleware("http")
async def observe_http_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Record bounded-cardinality request counts and latency."""

    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", "unmatched")
        HTTP_REQUESTS.labels(
            method=request.method,
            route=route,
            status_class=f"{status_code // 100}xx",
        ).inc()
        HTTP_DURATION.labels(method=request.method, route=route).observe(perf_counter() - started)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_model()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError):
        return {"status": "degraded", "model": "unavailable"}
    return {"status": "ok", "model": "loaded"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    model = _model_or_503()
    reference_mean, reference_std = model.reference_distribution
    return {
        "n_features": int(model.bounds.shape[0]),
        "bounds": model.bounds.tolist(),
        "device": str(model.device),
        "model": "pytorch-mlp-mc-dropout",
        "artifact_version": model.artifact_version,
        "max_batch_size": 256,
        "monitoring_reference": {
            "scaled_feature_mean": reference_mean.tolist(),
            "scaled_feature_std": reference_std.tolist(),
        },
    }


@app.get("/monitoring")
def monitoring() -> dict[str, object]:
    """Return process-local rolling drift and uncertainty diagnostics."""

    model = _model_or_503()
    reference_mean, reference_std = model.reference_distribution
    snapshot = prediction_monitor.snapshot(reference_mean, reference_std)
    MONITOR_WINDOW_SIZE.set(snapshot.window_size)
    DRIFT_SCORE.set(snapshot.max_feature_mean_shift)
    DRIFT_DETECTED.set(float(snapshot.drift_detected))
    return snapshot.to_dict()


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus-compatible service and model metrics."""

    return Response(generate_latest(METRICS_REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = _model_or_503()

    points = np.asarray(request.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != model.bounds.shape[0]:
        raise HTTPException(
            status_code=422,
            detail=f"Expected points with {model.bounds.shape[0]} features each.",
        )
    if not np.isfinite(points).all():
        raise HTTPException(status_code=422, detail="All input values must be finite.")

    low, high = model.bounds[:, 0], model.bounds[:, 1]
    if np.any(points < low) or np.any(points > high):
        raise HTTPException(
            status_code=422,
            detail="One or more inputs fall outside the model training bounds.",
        )

    mean, std = model.predict(points, mc_samples=request.mc_samples)
    prediction_monitor.record(model.scale_inputs(points), mean, std)
    PREDICTIONS.inc(len(mean))
    for uncertainty in std:
        PREDICTIVE_UNCERTAINTY.observe(float(uncertainty))
    return PredictResponse(
        predictions=[Prediction(mean=float(mu), std=float(sigma)) for mu, sigma in zip(mean, std)]
    )
