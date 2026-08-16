"""Train and evaluate a PyTorch surrogate on the physics-based CSTR environment.

This script creates a reproducible supervised-learning dataset from the
existing simulator, holds out a test set, trains the neural surrogate, reports
RMSE/MAE/R2, saves a versioned model artifact, and optionally logs the run to
MLflow when MLFLOW_TRACKING_URI is configured.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from titrate.baselines.lhs import propose as lhs_propose
from titrate.environments.cstr_env import CSTREnvironment
from titrate.surrogate.torch_model import TorchSurrogate


def build_dataset(env: CSTREnvironment, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = lhs_propose(env.bounds, n_samples, rng)
    y = np.array([env.evaluate(x, rng).objective for x in X], dtype=float)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/torch_surrogate.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("results/torch_surrogate_metrics.json"))
    args = parser.parse_args()

    env = CSTREnvironment()
    X, y = build_dataset(env, args.samples, args.seed)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(X))
    split = int(0.8 * len(order))
    train_idx, test_idx = order[:split], order[split:]

    model = TorchSurrogate(env.bounds, random_state=args.seed)
    history = model.fit(X[train_idx], y[train_idx], epochs=args.epochs)
    predicted, uncertainty = model.predict(X[test_idx], mc_samples=100)

    metrics = {
        "samples": args.samples,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "rmse": float(mean_squared_error(y[test_idx], predicted) ** 0.5),
        "mae": float(mean_absolute_error(y[test_idx], predicted)),
        "r2": float(r2_score(y[test_idx], predicted)),
        "mean_predictive_std": float(np.mean(uncertainty)),
        "best_epoch": int(history.best_epoch),
        "seed": args.seed,
    }

    model.save(args.artifact)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("titrate-cstr-surrogate")
        with mlflow.start_run():
            mlflow.log_params({"samples": args.samples, "epochs": args.epochs, "seed": args.seed})
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, float)})
            mlflow.log_artifact(str(args.artifact))
            mlflow.log_artifact(str(args.metrics))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
