"""Fair, reproducible comparison utilities for the GP and PyTorch surrogates.

The comparison intentionally uses one held-out split and identical training
points for both models. The Gaussian process remains Titrate's default for
small-data Bayesian optimization; the neural surrogate is an additional
scalable model whose strengths should be demonstrated rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from titrate.baselines.lhs import propose as lhs_propose
from titrate.environments.cstr_env import CSTREnvironment
from titrate.surrogate.gp_model import GPSurrogate
from titrate.surrogate.torch_model import TorchSurrogate, TrainingHistory


@dataclass(frozen=True)
class ModelMetrics:
    """Held-out accuracy, uncertainty, and fit-time measurements."""

    model: str
    rmse: float
    mae: float
    r2: float
    mean_predictive_std: float
    interval_95_coverage: float
    fit_seconds: float


@dataclass
class ComparisonResult:
    """Models and evidence produced from a single fair held-out comparison."""

    gp: GPSurrogate
    torch: TorchSurrogate
    torch_history: TrainingHistory
    metrics: tuple[ModelMetrics, ModelMetrics]
    X_test: np.ndarray
    y_test: np.ndarray
    gp_mean: np.ndarray
    gp_std: np.ndarray
    torch_mean: np.ndarray
    torch_std: np.ndarray


@dataclass(frozen=True)
class LearningCurvePoint:
    """One model's held-out score at one training-set size."""

    model: str
    train_samples: int
    rmse: float
    mae: float
    r2: float


def build_cstr_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic, noiseless space-filling CSTR dataset.

    Noise is excluded so the lab measures surrogate approximation quality,
    rather than rewarding either model for fitting a particular noise draw.
    """

    if n_samples < 25:
        raise ValueError("n_samples must be at least 25 for a held-out comparison.")
    env = CSTREnvironment()
    rng = np.random.default_rng(seed)
    X = lhs_propose(env.bounds, n_samples, rng)
    y = np.array([env.evaluate_noiseless(point).objective for point in X], dtype=float)
    return X, y


def held_out_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a reproducible shuffled split without leaking test observations."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of observations.")
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_fraction)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _metrics(
    model: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    std: np.ndarray,
    fit_seconds: float,
) -> ModelMetrics:
    interval_95 = np.abs(actual - predicted) <= 1.96 * np.maximum(std, 1e-12)
    return ModelMetrics(
        model=model,
        rmse=float(mean_squared_error(actual, predicted) ** 0.5),
        mae=float(mean_absolute_error(actual, predicted)),
        r2=float(r2_score(actual, predicted)),
        mean_predictive_std=float(np.mean(std)),
        interval_95_coverage=float(np.mean(interval_95)),
        fit_seconds=float(fit_seconds),
    )


def compare_surrogates(
    bounds: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
    torch_epochs: int = 120,
    torch_hidden_sizes: tuple[int, ...] = (64, 64),
    torch_patience: int = 25,
    mc_samples: int = 50,
    gp_restarts: int = 1,
) -> ComparisonResult:
    """Fit both models on identical observations and score one held-out set."""

    started = perf_counter()
    gp = GPSurrogate(
        bounds,
        random_state=seed,
        n_restarts_optimizer=gp_restarts,
    ).fit(X_train, y_train)
    gp_seconds = perf_counter() - started
    gp_mean, gp_std = gp.predict(X_test)

    torch_model = TorchSurrogate(
        bounds,
        hidden_sizes=torch_hidden_sizes,
        dropout=0.1,
        random_state=seed,
        device="cpu",
    )
    started = perf_counter()
    history = torch_model.fit(
        X_train,
        y_train,
        epochs=torch_epochs,
        patience=torch_patience,
        batch_size=min(64, len(X_train)),
    )
    torch_seconds = perf_counter() - started
    torch_mean, torch_std = torch_model.predict(X_test, mc_samples=mc_samples)

    return ComparisonResult(
        gp=gp,
        torch=torch_model,
        torch_history=history,
        metrics=(
            _metrics("Gaussian process", y_test, gp_mean, gp_std, gp_seconds),
            _metrics("PyTorch MLP", y_test, torch_mean, torch_std, torch_seconds),
        ),
        X_test=np.asarray(X_test),
        y_test=np.asarray(y_test),
        gp_mean=gp_mean,
        gp_std=gp_std,
        torch_mean=torch_mean,
        torch_std=torch_std,
    )


def build_learning_curve(
    bounds: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_sizes: tuple[int, ...],
    *,
    seed: int = 42,
    torch_epochs: int = 80,
    mc_samples: int = 30,
) -> list[LearningCurvePoint]:
    """Compare held-out error as each model receives more observations."""

    points: list[LearningCurvePoint] = []
    valid_sizes = sorted({size for size in sample_sizes if 5 <= size <= len(X_train)})
    if not valid_sizes:
        raise ValueError("At least one sample size must fit inside the training set.")

    for size in valid_sizes:
        result = compare_surrogates(
            bounds,
            X_train[:size],
            y_train[:size],
            X_test,
            y_test,
            seed=seed + size,
            torch_epochs=torch_epochs,
            torch_hidden_sizes=(48, 48),
            torch_patience=max(12, torch_epochs // 5),
            mc_samples=mc_samples,
            gp_restarts=0,
        )
        for metric in result.metrics:
            points.append(
                LearningCurvePoint(
                    model=metric.model,
                    train_samples=size,
                    rmse=metric.rmse,
                    mae=metric.mae,
                    r2=metric.r2,
                )
            )
    return points
