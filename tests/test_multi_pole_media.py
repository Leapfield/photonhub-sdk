"""Multi-pole / Drude medium model (schema 1.17, NUMERICS.md §19).

Engine-side physics is gated in engine tests + validation/test_tier3_multipole;
these pin the client model: wire shape and back-compat omission, the pole
budget, wire-order accessors, and the multi-pole eps evaluator (incl. the
negative metal branch the mode stack must not silently sqrt).
"""

import math

import pytest

import photonhub as ph
from photonhub.components.structures import MAX_ADE_POLES


def _lp(f0=2.0e14, de=1.0, lw=0.0):
    return ph.LorentzPole(resonance_frequency_hz=f0, delta_eps=de,
                          linewidth_hz=lw)


def test_wire_omits_unset_pole_fields():
    m = ph.Medium(permittivity=2.25)
    d = m.model_dump(mode="json", exclude_none=True)
    assert "lorentz" not in d and "poles" not in d and "drude" not in d
    # empty lists canonicalize to the omitted form
    m2 = ph.Medium(permittivity=2.25, poles=(), drude=())
    d2 = m2.model_dump(mode="json", exclude_none=True)
    assert d2 == d


def test_wire_carries_pole_lists():
    m = ph.Medium(
        permittivity=2.0,
        lorentz=_lp(1.0e14, 3.0, 5.0e12),
        poles=(_lp(2.0e14, 0.5),),
        drude=(ph.DrudePole(plasma_frequency_hz=1.5e14, linewidth_hz=1e13),))
    d = m.model_dump(mode="json", exclude_none=True)
    assert d["lorentz"]["resonance_frequency_hz"] == 1.0e14
    assert d["poles"][0]["delta_eps"] == 0.5
    assert d["drude"][0]["plasma_frequency_hz"] == 1.5e14


def test_pole_budget_enforced_across_all_fields():
    poles = tuple(_lp(1.0e14 + i * 1e13, 0.1) for i in range(MAX_ADE_POLES))
    ph.Medium(permittivity=2.0, poles=poles)  # exactly at the budget: fine
    with pytest.raises(ValueError, match="at most"):
        ph.Medium(permittivity=2.0, lorentz=_lp(), poles=poles)
    with pytest.raises(ValueError, match="at most"):
        ph.Medium(permittivity=2.0, poles=poles,
                  drude=(ph.DrudePole(plasma_frequency_hz=1e14),))


def test_dispersive_flag_and_wire_order():
    assert not ph.Medium(permittivity=2.0).is_dispersive
    assert ph.Medium(permittivity=2.0, lorentz=_lp()).is_dispersive
    assert ph.Medium(permittivity=2.0, poles=(_lp(),)).is_dispersive
    assert ph.Medium(
        permittivity=2.0,
        drude=(ph.DrudePole(plasma_frequency_hz=1e14),)).is_dispersive

    m = ph.Medium(permittivity=2.0, lorentz=_lp(1.0e14, 1.0),
                  poles=(_lp(2.0e14, 0.5), _lp(3.0e14, 0.25)))
    fs = [p.resonance_frequency_hz for p in m.all_lorentz_poles()]
    assert fs == [1.0e14, 2.0e14, 3.0e14]  # legacy first, then the list


def test_permittivity_at_hz_sums_poles():
    m1 = ph.Medium(permittivity=2.0, lorentz=_lp(2.0e14, 1.0, 1.0e13))
    m2 = ph.Medium(permittivity=2.0, poles=(_lp(2.0e14, 1.0, 1.0e13),))
    f = 1.5e14
    assert m1.permittivity_at_hz(f) == pytest.approx(
        m2.permittivity_at_hz(f), rel=1e-15)

    both = ph.Medium(permittivity=2.0,
                     lorentz=_lp(2.0e14, 1.0, 1.0e13),
                     poles=(_lp(3.0e14, 0.5, 1.0e13),))
    one = ph.Medium(permittivity=2.0, lorentz=_lp(2.0e14, 1.0, 1.0e13))
    other = ph.Medium(permittivity=2.0, lorentz=_lp(3.0e14, 0.5, 1.0e13))
    assert both.permittivity_at_hz(f) == pytest.approx(
        one.permittivity_at_hz(f) + other.permittivity_at_hz(f) - 2.0,
        rel=1e-12)


def test_drude_goes_negative_below_plasma():
    m = ph.Medium(permittivity=2.25,
                  drude=(ph.DrudePole(plasma_frequency_hz=4.5e14,
                                      linewidth_hz=2.0e13),))
    f = 1.934e14
    eps = m.permittivity_at_hz(f)
    w, wp, g = (2 * math.pi * f, 2 * math.pi * 4.5e14, 2 * math.pi * 2.0e13)
    assert eps == pytest.approx(2.25 - wp * wp / (w * w + g * g), rel=1e-12)
    assert eps < -1.0
    # collisionless limit: exactly eps_inf - (fp/f)^2
    m0 = ph.Medium(permittivity=2.25,
                   drude=(ph.DrudePole(plasma_frequency_hz=4.5e14),))
    assert m0.permittivity_at_hz(f) == pytest.approx(
        2.25 - (4.5e14 / f) ** 2, rel=1e-12)


def test_undamped_resonance_still_raises():
    m = ph.Medium(permittivity=2.0, poles=(_lp(2.0e14, 1.0, 0.0),))
    with pytest.raises(ValueError, match="resonance"):
        m.permittivity_at_hz(2.0e14)


def test_simulation_dispersive_policies_see_new_fields():
    """The subpixel default flip and PML stabilization must key on poles/drude
    too, not just the legacy field."""
    m = ph.Medium(permittivity=4.0,
                  drude=(ph.DrudePole(plasma_frequency_hz=4.0e14,
                                      linewidth_hz=2.0e13),))
    sim = ph.Simulation(
        size_um=(0.4, 0.4, 0.4),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=10),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="periodic"),
        structures=[ph.Structure(
            geometry=ph.Box(center_um=(0.2, 0.2, 0.2),
                            size_um=(0.1, 0.1, 0.1)),
            medium=m)],
        sources=[ph.PointDipole(
            center_um=(0.2, 0.2, 0.2), polarization="Ez",
            source_time=ph.GaussianPulse(freq0_hz=1.934e14,
                                         fwidth_hz=4.0e13))],
    )
    # dispersive scene -> subpixel default OFF (the §19 auto-default rule)
    assert sim.subpixel is False


def test_pec_medium_model():
    m = ph.Medium(permittivity=1.0, pec=True)
    d = m.model_dump(mode="json", exclude_none=True)
    assert d["pec"] is True
    # False canonicalizes to the omitted form (byte-back-compat)
    off = ph.Medium(permittivity=2.0, pec=False)
    assert "pec" not in off.model_dump(mode="json", exclude_none=True)
    assert not off.is_dispersive
    with pytest.raises(ValueError, match="PEC"):
        ph.Medium(permittivity=1.0, pec=True,
                  lorentz=_lp(2.0e14, 1.0))
    with pytest.raises(ValueError, match="PEC"):
        ph.Medium(permittivity=1.0, pec=True, conductivity_s_per_m=5.0)
    with pytest.raises(ValueError, match="PEC"):
        ph.Medium(permittivity=1.0, pec=True,
                  drude=(ph.DrudePole(plasma_frequency_hz=1e14),))
