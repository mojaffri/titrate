"""PyTorch neural-network surrogate with uncertainty estimates.

This complements the Gaussian-process surrogate already used by Titrate. The
GP remains the default for small-data Bayesian optimization, while this model
provides a scalable deep-learning baseline for larger experiment sets and a
production-friendly artifact that can be served independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = n_features
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_features, hidden), nn.GELU(), nn.Dropout(dropout)])
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int


class TorchSurrogate:
    """Scaled MLP surrogate with MC-dropout predictive uncertainty.

    Inputs are scaled from physical bounds to [0, 1]. Targets are standardized.
    ``predict`` returns mean and standard deviation in original target units,
    matching the interface used by ``GPSurrogate``.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        hidden_sizes: tuple[int, ...] = (128, 128, 64),
        dropout: float = 0.10,
        random_state: int = 0,
        device: str | None = None,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dimensions, 2).")
        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("Every input dimension must have a non-zero range.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.dropout = float(dropout)
        self.random_state = int(random_state)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._model = _MLP(self.bounds.shape[0], self.hidden_sizes, self.dropout).to(self.device)
        self._y_mean = 0.0
        self._y_std = 1.0
        self._fitted = False

    def _scale_x(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        low = self.bounds[:, 0].astype(np.float32)
        high = self.bounds[:, 1].astype(np.float32)
        return (X - low) / (high - low)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 500,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        validation_fraction: float = 0.2,
        patience: int = 50,
    ) -> TrainingHistory:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of observations.")
        if len(X) < 5:
            raise ValueError("TorchSurrogate requires at least 5 observations.")

        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) if float(y.std()) > 1e-8 else 1.0
        X_scaled = self._scale_x(X)
        y_scaled = (y - self._y_mean) / self._y_std

        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(len(X_scaled))
        n_val = max(1, int(round(len(indices) * validation_fraction)))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        if len(train_idx) == 0:
            train_idx, val_idx = indices[:-1], indices[-1:]

        train_ds = TensorDataset(
            torch.tensor(X_scaled[train_idx], dtype=torch.float32),
            torch.tensor(y_scaled[train_idx], dtype=torch.float32),
        )
        generator = torch.Generator().manual_seed(self.random_state)
        train_loader = DataLoader(
            train_ds,
            batch_size=min(batch_size, len(train_ds)),
            shuffle=True,
            generator=generator,
        )
        X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32, device=self.device)
        y_val = torch.tensor(y_scaled[val_idx], dtype=torch.float32, device=self.device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_fn = nn.MSELoss()
        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        best_epoch = 0
        stale_epochs = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(epochs):
            self._model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self._model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(X_batch)
            epoch_loss /= len(train_ds)

            self._model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self._model(X_val), y_val).item())
            train_losses.append(epoch_loss)
            val_losses.append(val_loss)

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self._model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.to(self.device)
        self._fitted = True
        return TrainingHistory(train_losses, val_losses, best_epoch)

    def predict(self, X: np.ndarray, mc_samples: int = 50) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("TorchSurrogate.predict called before fit().")
        if mc_samples < 2:
            raise ValueError("mc_samples must be at least 2 to estimate uncertainty.")

        X_tensor = torch.tensor(self._scale_x(np.atleast_2d(X)), dtype=torch.float32, device=self.device)
        self._model.train()  # keep dropout active for Monte Carlo uncertainty
        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(mc_samples):
                prediction = self._model(X_tensor).detach().cpu().numpy()
                samples.append(prediction)
        stacked = np.stack(samples, axis=0)
        mean = stacked.mean(axis=0) * self._y_std + self._y_mean
        std = stacked.std(axis=0, ddof=1) * self._y_std
        return mean.astype(float), np.maximum(std, 1e-9).astype(float)

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted TorchSurrogate.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bounds": self.bounds,
                "hidden_sizes": self.hidden_sizes,
                "dropout": self.dropout,
                "random_state": self.random_state,
                "y_mean": self._y_mean,
                "y_std": self._y_std,
                "state_dict": self._model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "TorchSurrogate":
        checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
        model = cls(
            bounds=np.asarray(checkpoint["bounds"], dtype=float),
            hidden_sizes=tuple(checkpoint["hidden_sizes"]),
            dropout=float(checkpoint["dropout"]),
            random_state=int(checkpoint["random_state"]),
            device=device,
        )
        model._model.load_state_dict(checkpoint["state_dict"])
        model._y_mean = float(checkpoint["y_mean"])
        model._y_std = float(checkpoint["y_std"])
        model._fitted = True
        model._model.to(model.device)
        return model
