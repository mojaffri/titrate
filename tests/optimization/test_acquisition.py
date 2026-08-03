import numpy as np

from titrate.optimization.acquisition import (
    constrained_expected_improvement,
    expected_improvement,
    probability_of_feasibility,
)


def test_ei_is_never_negative():
    mean = np.array([-1.0, 0.0, 0.5, 1.0, 5.0])
    std = np.array([0.1, 0.0, 1.0, 2.0, 0.01])
    ei = expected_improvement(mean, std, y_best=0.5)
    assert np.all(ei >= 0.0)


def test_ei_zero_std_matches_clipped_improvement():
    mean = np.array([0.2, 0.8])
    std = np.array([1e-12, 1e-12])
    y_best = 0.5
    xi = 0.0
    ei = expected_improvement(mean, std, y_best, xi=xi)
    expected = np.maximum(mean - y_best - xi, 0.0)
    assert np.allclose(ei, expected, atol=1e-6)


def test_ei_increases_with_mean_at_fixed_std():
    std = np.array([0.1])
    ei_low = expected_improvement(np.array([0.3]), std, y_best=0.5)
    ei_high = expected_improvement(np.array([0.9]), std, y_best=0.5)
    assert ei_high > ei_low


def test_ei_increases_with_uncertainty_when_mean_below_best():
    mean = np.array([0.4])
    ei_low_std = expected_improvement(mean, np.array([0.01]), y_best=0.5)
    ei_high_std = expected_improvement(mean, np.array([1.0]), y_best=0.5)
    assert ei_high_std > ei_low_std


def test_probability_of_feasibility_in_unit_interval():
    pf = probability_of_feasibility(
        constraint_mean=np.array([-1.0, 0.0, 0.05, 1.0]),
        constraint_std=np.array([0.5, 0.5, 0.01, 0.5]),
        constraint_max=0.05,
    )
    assert np.all(pf >= 0.0) and np.all(pf <= 1.0)


def test_probability_of_feasibility_decreases_as_constraint_mean_rises():
    constraint_std = np.array([0.1, 0.1, 0.1])
    pf = probability_of_feasibility(
        constraint_mean=np.array([0.0, 0.05, 0.2]),
        constraint_std=constraint_std,
        constraint_max=0.05,
    )
    assert pf[0] > pf[1] > pf[2]


def test_constrained_ei_never_exceeds_unconstrained_ei():
    mean = np.array([0.6, 0.7, 0.9])
    std = np.array([0.1, 0.2, 0.05])
    y_best = 0.5
    constraint_mean = np.array([0.06, 0.1, 0.02])
    constraint_std = np.array([0.02, 0.05, 0.01])
    constraint_max = 0.05

    ei = expected_improvement(mean, std, y_best)
    cei = constrained_expected_improvement(
        mean, std, y_best, constraint_mean, constraint_std, constraint_max
    )
    assert np.all(cei <= ei + 1e-12)


def test_constrained_ei_near_zero_when_clearly_infeasible():
    mean = np.array([0.95])
    std = np.array([0.05])
    y_best = 0.5
    constraint_mean = np.array([0.5])  # far above constraint_max
    constraint_std = np.array([0.01])  # confidently infeasible
    constraint_max = 0.05

    cei = constrained_expected_improvement(
        mean, std, y_best, constraint_mean, constraint_std, constraint_max
    )
    assert cei[0] < 1e-6
