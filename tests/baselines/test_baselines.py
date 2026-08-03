import numpy as np
import pytest

from titrate.baselines import grid_search, lhs, random_search

BOUNDS = np.array([[320.0, 400.0], [0.1, 5.0], [0.0, 2.0]])
STRATEGIES = [random_search.propose, lhs.propose, grid_search.propose]


@pytest.mark.parametrize("propose", STRATEGIES)
def test_shape_matches_request(propose):
    rng = np.random.default_rng(0)
    points = propose(BOUNDS, 25, rng)
    assert points.shape == (25, 3)


@pytest.mark.parametrize("propose", STRATEGIES)
def test_points_respect_bounds(propose):
    rng = np.random.default_rng(1)
    points = propose(BOUNDS, 30, rng)
    for dim in range(BOUNDS.shape[0]):
        assert np.all(points[:, dim] >= BOUNDS[dim, 0])
        assert np.all(points[:, dim] <= BOUNDS[dim, 1])


@pytest.mark.parametrize("propose", STRATEGIES)
def test_reproducible_with_same_seed(propose):
    points_a = propose(BOUNDS, 20, np.random.default_rng(42))
    points_b = propose(BOUNDS, 20, np.random.default_rng(42))
    assert np.array_equal(points_a, points_b)


def test_random_search_differs_across_seeds():
    a = random_search.propose(BOUNDS, 20, np.random.default_rng(1))
    b = random_search.propose(BOUNDS, 20, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_grid_search_handles_non_perfect_cube_budget():
    rng = np.random.default_rng(0)
    points = grid_search.propose(BOUNDS, 25, rng)
    assert points.shape == (25, 3)
    unique_rows = {tuple(row) for row in points}
    assert len(unique_rows) == 25


def test_grid_search_exact_cube_uses_full_grid():
    rng = np.random.default_rng(0)
    points = grid_search.propose(BOUNDS, 27, rng)  # 3^3
    assert points.shape == (27, 3)
    for dim in range(3):
        assert len(np.unique(points[:, dim])) == 3


def test_lhs_covers_each_dimension_bin():
    """A defining LHS property: each 1D projection lands one point per bin."""
    rng = np.random.default_rng(3)
    n = 15
    points = lhs.propose(BOUNDS, n, rng)
    for dim in range(BOUNDS.shape[0]):
        edges = np.linspace(BOUNDS[dim, 0], BOUNDS[dim, 1], n + 1)
        bin_idx = np.digitize(points[:, dim], edges) - 1
        assert sorted(bin_idx) == list(range(n))
