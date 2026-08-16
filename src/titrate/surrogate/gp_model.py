"""Gaussian Process surrogate wrapper: input scaling + output standardization
around scikit-learn's GaussianProcessRegressor.

Chosen over a neural network deliberately: with the small experiment budgets
this project targets (tens of points, not thousands), a GP gives calibrated
posterior uncertainty out of the box and is the standard small-data
surrogate choice in the Bayesian optimization literature. Titrate's separate
PyTorch surrogate complements this model for larger supervised datasets; it
does not replace the GP in the small-data optimization loop.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class GPSurrogate:
    """Fits a GP on inputs scaled to [0, 1]^d (from `bounds`) and outputs
    standardized to zero mean / unit variance -- both are necessary for
    stable marginal-likelihood optimization when raw dimensions have very
    different physical scales (e.g. temperature ~350 vs. catalyst ~1)."""

    def __init__(
        self,
        bounds: np.ndarray,
        random_state: int = 0,
        n_restarts_optimizer: int = 5,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dimensions, 2).")
        if not np.isfinite(self.bounds).all():
            raise ValueError("bounds must contain only finite values.")
        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("Every input dimension must have a non-zero range.")
        if n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer must be non-negative.")
        n_dims = self.bounds.shape[0]
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1.0))
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        self._y_mean = 0.0
        self._y_std = 1.0
        self._fitted = False

    def _scale_x(self, X: np.ndarray) -> np.ndarray:
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return (X - low) / (high - low)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) if y.std() > 1e-8 else 1.0
        y_scaled = (y - self._y_mean) / self._y_std
        self._gp.fit(self._scale_x(X), y_scaled)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) in the original (unscaled) output units."""
        if not self._fitted:
            raise RuntimeError("GPSurrogate.predict called before fit().")
        X = np.atleast_2d(X)
        mean_scaled, std_scaled = self._gp.predict(self._scale_x(X), return_std=True)
        mean = mean_scaled * self._y_std + self._y_mean
        std = std_scaled * self._y_std
        return mean, std
