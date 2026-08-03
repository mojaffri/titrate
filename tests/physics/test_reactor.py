import numpy as np

from titrate.physics.reactor import REFERENCE_PARAMS, cstr_yield


def test_outputs_are_physically_bounded():
    for T in [320.0, 360.0, 400.0]:
        for tau in [0.1, 1.0, 5.0]:
            for c in [0.0, 1.0, 2.0]:
                out = cstr_yield(T, tau, c)
                assert 0.0 <= out.conversion <= 1.0
                assert 0.0 <= out.selectivity <= 1.0
                assert 0.0 <= out.yield_ <= 1.0
                assert out.impurity >= 0.0


def test_conversion_limit_as_residence_time_to_zero():
    out = cstr_yield(360.0, 1e-6, 1.0)
    assert out.conversion < 0.01


def test_conversion_limit_as_residence_time_grows_large():
    out = cstr_yield(360.0, 1e6, 1.0)
    assert out.conversion > 0.999


def test_conversion_monotonic_increasing_in_residence_time():
    taus = [0.1, 0.5, 1.0, 2.0, 5.0]
    conversions = [cstr_yield(360.0, t, 1.0).conversion for t in taus]
    assert np.all(np.diff(conversions) > 0)


def test_selectivity_independent_of_residence_time():
    """Selectivity is a ratio of rate constants only; tau must not affect it."""
    selectivities = [cstr_yield(360.0, t, 1.0).selectivity for t in [0.1, 1.0, 5.0]]
    assert np.allclose(selectivities, selectivities[0])


def test_selectivity_decreases_with_temperature():
    """Ea_side > Ea_desired by construction, so raising T erodes selectivity."""
    selectivities = [cstr_yield(T, 1.0, 1.0).selectivity for T in [320.0, 360.0, 400.0]]
    assert np.all(np.diff(selectivities) < 0)


def test_catalyst_improves_selectivity_at_fixed_temperature():
    """Catalyst only promotes the desired pathway (k1), so it should raise S."""
    s_no_cat = cstr_yield(360.0, 1.0, 0.0).selectivity
    s_with_cat = cstr_yield(360.0, 1.0, 2.0).selectivity
    assert s_with_cat > s_no_cat


def test_yield_has_interior_optimum_in_temperature():
    """At fixed tau/Ccat, yield vs T is non-monotonic: an interior optimum
    exists strictly inside the bounds, not at either boundary."""
    Ts = np.linspace(320.0, 400.0, 200)
    yields = [cstr_yield(T, 1.0, 1.0).yield_ for T in Ts]
    argmax = np.argmax(yields)
    assert 0 < argmax < len(Ts) - 1


def test_optimal_temperature_shifts_with_residence_time_constraint():
    """The T that maximizes yield depends on tau (a real, coupled trade-off),
    not just on T in isolation."""
    Ts = np.linspace(320.0, 400.0, 200)

    def best_T(tau):
        yields = [cstr_yield(T, tau, 2.0).yield_ for T in Ts]
        return Ts[np.argmax(yields)]

    T_star_short_tau = best_T(0.5)
    T_star_long_tau = best_T(5.0)
    assert T_star_short_tau > T_star_long_tau


def test_reference_params_are_the_default():
    out_default = cstr_yield(360.0, 1.0, 1.0)
    out_explicit = cstr_yield(360.0, 1.0, 1.0, params=REFERENCE_PARAMS)
    assert out_default == out_explicit
