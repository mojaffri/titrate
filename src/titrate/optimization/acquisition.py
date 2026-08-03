"""Acquisition functions, implemented from scratch on top of GP posterior
mean/std (not called from a black-box BO library).

Expected Improvement (maximization):
    EI(x) = E[max(f(x) - y_best - xi, 0)]
          = (mu - y_best - xi) * Phi(z) + sigma * phi(z),   z = (mu - y_best - xi) / sigma

Constrained EI (Gardner et al., 2014, "Bayesian Optimization with Inequality
Constraints"): weight EI by the GP's own estimated probability that the
constraint is satisfied at x, using the constraint GP's posterior:
    P_feasible(x) = P(g(x) <= g_max) = Phi((g_max - mu_g) / sigma_g)
    constrained_EI(x) = EI(x) * P_feasible(x)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """EI for maximizing an objective, given GP posterior mean/std."""
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-9)
    improvement = mean - y_best - xi
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0.0)


def probability_of_feasibility(
    constraint_mean: np.ndarray,
    constraint_std: np.ndarray,
    constraint_max: float,
) -> np.ndarray:
    """P(constraint_value(x) <= constraint_max) under the constraint GP posterior."""
    constraint_mean = np.asarray(constraint_mean, dtype=float)
    constraint_std = np.maximum(np.asarray(constraint_std, dtype=float), 1e-9)
    z = (constraint_max - constraint_mean) / constraint_std
    return norm.cdf(z)


def constrained_expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    y_best: float,
    constraint_mean: np.ndarray,
    constraint_std: np.ndarray,
    constraint_max: float,
    xi: float = 0.01,
) -> np.ndarray:
    """EI(x) * P_feasible(x) -- see module docstring."""
    ei = expected_improvement(mean, std, y_best, xi)
    p_feasible = probability_of_feasibility(constraint_mean, constraint_std, constraint_max)
    return ei * p_feasible
