"""FastAPI service for a trained PyTorch surrogate artifact.

Set TITRATE_MODEL_PATH to the .pt artifact produced by TorchSurrogate.save.
The service is intentionally small so it can run locally, in Docker, or on a
managed container platform such as AWS App Runner/ECS.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from titrate.surrogate.torch_model import TorchSurrogate


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
    return {
        "n_features": int(model.bounds.shape[0]),
        "bounds": model.bounds.tolist(),
        "device": str(model.device),
        "model": "pytorch-mlp-mc-dropout",
        "artifact_version": 1,
        "max_batch_size": 256,
    }


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
    return PredictResponse(
        predictions=[Prediction(mean=float(mu), std=float(sigma)) for mu, sigma in zip(mean, std)]
    )
