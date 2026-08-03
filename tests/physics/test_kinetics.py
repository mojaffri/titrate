import numpy as np
import pytest

from titrate.physics.kinetics import arrhenius_rate_constant, catalyzed_rate_constant


def test_arrhenius_increases_with_temperature():
    k_low = arrhenius_rate_constant(1e7, 50_000.0, 320.0)
    k_high = arrhenius_rate_constant(1e7, 50_000.0, 400.0)
    assert k_high > k_low


def test_arrhenius_higher_activation_energy_is_more_temperature_sensitive():
    A = 1e9
    T_low, T_high = 320.0, 400.0
    ratio_low_Ea = arrhenius_rate_constant(A, 40_000.0, T_high) / arrhenius_rate_constant(
        A, 40_000.0, T_low
    )
    ratio_high_Ea = arrhenius_rate_constant(A, 80_000.0, T_high) / arrhenius_rate_constant(
        A, 80_000.0, T_low
    )
    assert ratio_high_Ea > ratio_low_Ea


def test_arrhenius_vectorized_over_temperature_array():
    T = np.array([320.0, 360.0, 400.0])
    k = arrhenius_rate_constant(1e7, 50_000.0, T)
    assert k.shape == T.shape
    assert np.all(np.diff(k) > 0)


def test_catalyzed_rate_constant_never_decreases_base_rate():
    base = 2.0
    for c in [0.0, 0.5, 1.0, 2.0]:
        k_eff = catalyzed_rate_constant(base, c, promotion_strength=3.0, saturation_constant=2.0)
        assert k_eff >= base


def test_catalyzed_rate_constant_zero_loading_equals_base():
    base = 2.0
    k_eff = catalyzed_rate_constant(base, 0.0, promotion_strength=3.0, saturation_constant=2.0)
    assert k_eff == pytest.approx(base)


def test_catalyzed_rate_constant_increasing_in_loading():
    base = 2.0
    loadings = [0.0, 0.5, 1.0, 1.5, 2.0]
    values = [
        catalyzed_rate_constant(base, c, promotion_strength=3.0, saturation_constant=2.0)
        for c in loadings
    ]
    assert np.all(np.diff(values) > 0)


def test_catalyzed_rate_constant_saturates():
    """Diminishing returns: the marginal gain shrinks as loading increases."""
    base = 2.0
    k = lambda c: catalyzed_rate_constant(base, c, promotion_strength=3.0, saturation_constant=2.0)
    early_gain = k(0.5) - k(0.0)
    late_gain = k(2.0) - k(1.5)
    assert late_gain < early_gain
