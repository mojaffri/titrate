"""Fixed Cartesian-grid sampling baseline.

A regular grid only has n_per_dim ** n_dims possible sizes, so an exact
budget is generally unreachable. We pick the smallest n_per_dim whose full
grid covers at least n_points, then take a reproducible random subset of
that grid of exactly n_points -- this keeps every point grid-structured
while still honoring the shared experiment budget used by every other
strategy in the benchmark.
"""

from __future__ import annotations

import numpy as np


def propose(bounds: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n_points from a Cartesian grid over bounds.

    Args:
        bounds: shape (n_dims, 2), rows (low, high).
        n_points: number of points to return.
        rng: seeded random generator, used only to subset an oversized grid.

    Returns:
        Array of shape (n_points, n_dims).
    """
    n_dims = bounds.shape[0]
    n_per_dim = 2
    while n_per_dim**n_dims < n_points:
        n_per_dim += 1

    axes = [np.linspace(low, high, n_per_dim) for low, high in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([m.ravel() for m in mesh], axis=1)

    if grid.shape[0] == n_points:
        return grid
    selected_idx = rng.choice(grid.shape[0], size=n_points, replace=False)
    return grid[selected_idx]
