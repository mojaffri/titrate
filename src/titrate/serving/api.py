"""FastAPI service for a trained PyTorch surrogate artifact.

Set TITRATE_MODEL_PATH to the .pt artifact produced by TorchSurrogate.save.
The service is intentionally small so it can run locally, in Docker, or on a
managed container platform such as AWS App Runner/ECS.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from titrate.surrogate.torch_model import TorchSurrogate


class PredictRequest(BaseModel):
    points: list[list[float]] = Field(min_length=1)
    mc_samples: int = Field(default=50, ge=2, le=500)


class Prediction(BaseModel):
    mean: float
    std: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model: str = "pytorch-mlp-mc-dropout"


@lru_cache(maxsize=1)
def get_model() -> TorchSurrogate:
    model_path = os.getenv("TITRATE_MODEL_PATH", "artifacts/torch_surrogate.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at {model_path!r}. Set TITRATE_MODEL_PATH to a trained .pt file."
        )
    return TorchSurrogate.load(model_path)


app = FastAPI(
    title="Titrate Surrogate API",
    version="0.2.0",
    description="Uncertainty-aware inference for chemical-process surrogate models.",
)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_model()
    except FileNotFoundError:
        return {"status": "degraded", "model": "missing"}
    return {"status": "ok", "model": "loaded"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "n_features": int(model.bounds.shape[0]),
        "bounds": model.bounds.tolist(),
        "device": str(model.device),
        "model": "pytorch-mlp-mc-dropout",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

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
