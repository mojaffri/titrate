"""Steady-state CSTR model for a parallel first-order reaction network.

Reaction network (liquid phase, constant density, constant volume CSTR):

    A -> B   (desired product),   rate constant k1(T, C_cat)
    A -> C   (undesired byproduct), rate constant k2(T)

k1 and k2 follow Arrhenius behavior with Ea2 > Ea1, so the side reaction is
more temperature-sensitive than the desired one: raising temperature always
speeds up both reactions, but it erodes selectivity toward B. That trade-off
against residence time (more tau -> more conversion, cheaper/smaller
reactor) is what gives the yield surface a genuine interior optimum instead
of a boundary solution.

Parameter values below (REFERENCE_PARAMS) are chosen to be physically
plausible in order of magnitude for a liquid-phase organic reaction (e.g. an
esterification-like system: Ea ~ 50-80 kJ/mol, residence times of minutes to
a few hours, moderate temperatures well below solvent boiling points) -- they
are illustrative, not fit to a specific published system, and this is stated
explicitly wherever the model is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from titrate.physics.kinetics import arrhenius_rate_constant, catalyzed_rate_constant


@dataclass(frozen=True)
class ReactionParameters:
    """Physical constants for the A -> B (desired) / A -> C (side) network."""

    pre_exp_desired: float = 5.8e7  # A1, hr^-1
    activation_energy_desired: float = 50_000.0  # Ea1, J/mol
    pre_exp_side: float = 4.72e10  # A2, hr^-1
    activation_energy_side: float = 75_000.0  # Ea2, J/mol (> Ea1 by design)
    catalyst_promotion_strength: float = 3.0  # beta, dimensionless
    catalyst_saturation_constant: float = 2.0  # Kc, (mol%)^-1
    feed_concentration: float = 1.0  # C_A0, mol/L (basis concentration)


REFERENCE_PARAMS = ReactionParameters()


@dataclass(frozen=True)
class ReactorOutput:
    """Steady-state CSTR outputs for one operating condition."""

    conversion: float
    selectivity: float
    yield_: float
    impurity: float


def cstr_yield(
    temperature: float | np.ndarray,
    residence_time: float | np.ndarray,
    catalyst_loading: float | np.ndarray,
    params: ReactionParameters = REFERENCE_PARAMS,
) -> ReactorOutput:
    """Solve the steady-state CSTR mass balance at one operating point.

    Args:
        temperature: T, Kelvin.
        residence_time: tau = V/Q, hours.
        catalyst_loading: C_cat, mol%.
        params: reaction network constants.

    Returns:
        ReactorOutput with conversion X, selectivity S, yield Y = X*S, and
        the byproduct (impurity) concentration C_A0 * X * (1 - S).
    """
    k1_base = arrhenius_rate_constant(
        params.pre_exp_desired, params.activation_energy_desired, temperature
    )
    k2 = arrhenius_rate_constant(
        params.pre_exp_side, params.activation_energy_side, temperature
    )
    k1_eff = catalyzed_rate_constant(
        k1_base,
        catalyst_loading,
        params.catalyst_promotion_strength,
        params.catalyst_saturation_constant,
    )

    k_total_tau = (k1_eff + k2) * residence_time
    conversion = k_total_tau / (1.0 + k_total_tau)
    selectivity = k1_eff / (k1_eff + k2)
    yield_ = conversion * selectivity
    impurity = params.feed_concentration * conversion * (1.0 - selectivity)

    return ReactorOutput(
        conversion=float(conversion),
        selectivity=float(selectivity),
        yield_=float(yield_),
        impurity=float(impurity),
    )

