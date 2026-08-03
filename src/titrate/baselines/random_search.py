"""Uniform random sampling baseline."""

from __future__ import annotations

import numpy as np


def propose(bounds: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_points uniformly at random within bounds.

    Args:
        bounds: shape (n_dims, 2), rows (low, high).
        n_points: number of points to draw.
        rng: seeded random generator (determines reproducibility).

    Returns:
        Array of shape (n_points, n_dims).
    """
    low, high = bounds[:, 0], bounds[:, 1]
    return rng.uniform(low, high, size=(n_points, bounds.shape[0]))
