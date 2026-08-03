"""Arrhenius reaction-rate kinetics for the Titrate CSTR model.

All rate constants are first-order, expressed in hr^-1. Temperatures are in
Kelvin, activation energies in J/mol, matching the gas constant R below.
"""

from __future__ import annotations

import numpy as np

GAS_CONSTANT = 8.314  # J / (mol * K)


def arrhenius_rate_constant(
    pre_exponential_factor: float,
    activation_energy: float,
    temperature: float | np.ndarray,
) -> float | np.ndarray:
    """k(T) = A * exp(-Ea / (R * T))."""
    return pre_exponential_factor * np.exp(
        -activation_energy / (GAS_CONSTANT * temperature)
    )


def catalyzed_rate_constant(
    base_rate_constant: float | np.ndarray,
    catalyst_loading: float | np.ndarray,
    promotion_strength: float,
    saturation_constant: float,
) -> float | np.ndarray:
    """Langmuir-type catalytic promotion of a base rate constant.

    k_eff = k_base * (1 + beta * C_cat / (1 + Kc * C_cat))

    This saturates at high catalyst loading (diminishing returns), which is
    the physically realistic behavior for surface/homogeneous catalysis
    rather than an unbounded linear speedup.
    """
    promotion = (
        promotion_strength
        * catalyst_loading
        / (1.0 + saturation_constant * catalyst_loading)
    )
    return base_rate_constant * (1.0 + promotion)
