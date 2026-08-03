"""Latin Hypercube Sampling baseline -- space-filling, non-adaptive."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def propose(bounds: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an n_points Latin Hypercube design scaled to bounds.

    Args:
        bounds: shape (n_dims, 2), rows (low, high).
        n_points: number of points to draw.
        rng: seeded random generator (determines reproducibility).

    Returns:
        Array of shape (n_points, n_dims).
    """
    n_dims = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=n_dims, seed=rng)
    unit_samples = sampler.random(n=n_points)
    return qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
