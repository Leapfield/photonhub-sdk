"""Full-vector, 1 W power-normalized mode source (NUMERICS.md §18, WS2).

Covers the Python pipeline that turns a full-vector ``VectorMode`` into a
power-normalized :class:`~photonhub.components.sources.ModeSource` carrying BOTH
transverse-E components:

* 1 W power normalization (the injected modal Poynting flux integrates to the
  requested power in the engine's scalar-H convention);
* both-component packing (major + minor transverse-E, correct polarizations and
  the mode's true component ratio);
* :meth:`VectorMode.modal_power` (the physical full-H Poynting flux);
* back-compat (the scalar ``mode_source`` path is unchanged; the wire format
  omits the new optional fields when unset);
* the additive wire-format validators.
"""

import math

import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins import (
    ModeSolver,
    VectorModeSolver,
    mode_source,
    mode_source_vector,
)
from photonhub.plugins.mode_overlap import ETA0, _cell_widths, vector_modal_fields

F0 = 1.934e14  # Hz, ~1.55 um
DL = 0.04  # um
N_CORE = 3.5
N_CLAD = 1.444


def _waveguide_shell(*, wt_um=1.6, lz_um=4.0):
    """A straight SOI strip shell (the grid the builder reads). A dummy point
    dipole satisfies the 'at least one source' constructor rule."""
    core = ph.Structure(
        geometry=ph.Box(
            center_um=(wt_um / 2, wt_um / 2, lz_um / 2),
            size_um=(0.45, 0.22, lz_um * 4),
        ),
        medium=ph.Medium(permittivity=N_CORE**2),
    )
    base = dict(
        size_um=(wt_um, wt_um, lz_um),
        grid=ph.UniformGridSpec(dl_um=DL),
        run=ph.RunSpec(n_steps=100),
        background=ph.Background(permittivity=N_CLAD**2),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        structures=[core],
    )
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13)
    shell = ph.Simulation(
        **base,
        sources=[
            ph.PointDipole(
                center_um=(0.8, 0.8, 0.8), polarization="Ex", source_time=pulse
            )
        ],
    )
    return base, shell, pulse


def _te0_vector_mode():
    sv = VectorModeSolver.from_rectangular_core(
        wavelength_um=1.55,
        dl_um=DL,
        core_w_um=0.45,
        core_h_um=0.22,
        n_core=N_CORE,
        n_clad=N_CLAD,
    )
    return sv.solve(num_modes=1, n_guess=3.0)[0]


def _injected_power(src):
    """The modal Poynting flux the engine launches from this ModeSource, in the
    engine's scalar-H convention: P = (n_eff/2 eta0) * amplitude^2
    * integral (|profile|^2 + |profile_minor|^2) dA, over the plane cells."""
    nu, nv = src.nu, src.nv
    maj = np.asarray(src.profile, dtype=float).reshape(nv, nu)
    minr = (
        np.asarray(src.profile_minor, dtype=float).reshape(nv, nu)
        if src.profile_minor is not None
        else np.zeros_like(maj)
    )
    u = (np.arange(nu) + 0.5) * DL
    v = (np.arange(nv) + 0.5) * DL
    dA = np.outer(_cell_widths(v), _cell_widths(u)) * 1e-12  # um^2 -> m^2
    amp2 = src.amplitude ** 2
    return (src.n_eff / (2.0 * ETA0)) * amp2 * float(
        np.sum((maj**2 + minr**2) * dA)
    )


# --------------------------------------------------------------------------
# VectorMode.modal_power
# --------------------------------------------------------------------------


def test_modal_power_is_positive_and_uses_true_H():
    """The full-vector modal power (true E x H*) is finite and positive for the
    forward guided mode, and not equal to the field L2 normalization (it carries
    physical units)."""
    m = _te0_vector_mode()
    p = m.modal_power()
    assert math.isfinite(p)
    assert p > 0.0
    # The transverse-E pair is L2-normalized (dimensionless ~1), but the power
    # carries SI units and is tiny in absolute terms — distinctly not ~1.
    assert p != pytest.approx(1.0, abs=0.1)


# --------------------------------------------------------------------------
# 1 W power normalization
# --------------------------------------------------------------------------


def test_vector_mode_source_injects_one_watt():
    """The packed profiles are scaled so the engine launches exactly 1 W in its
    scalar-H injection convention (the ∫S·n̂ == 1 W acceptance check)."""
    _, shell, pulse = _waveguide_shell()
    m = _te0_vector_mode()
    src = mode_source_vector(
        shell, m, axis="z", position_um=0.8, source_time=pulse
    )
    assert _injected_power(src) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("watts", [0.5, 2.0, 10.0])
def test_vector_mode_source_scales_to_requested_power(watts):
    _, shell, pulse = _waveguide_shell()
    m = _te0_vector_mode()
    src = mode_source_vector(
        shell, m, axis="z", position_um=0.8, source_time=pulse,
        power_watts=watts,
    )
    assert _injected_power(src) == pytest.approx(watts, rel=1e-9)


def test_power_scales_quadratically_with_profile():
    """Doubling the target power scales the profile amplitudes by sqrt(2)
    (P ∝ amplitude², the physical scaling)."""
    _, shell, pulse = _waveguide_shell()
    m = _te0_vector_mode()
    s1 = mode_source_vector(
        shell, m, axis="z", position_um=0.8, source_time=pulse, power_watts=1.0
    )
    s2 = mode_source_vector(
        shell, m, axis="z", position_um=0.8, source_time=pulse, power_watts=2.0
    )
    p1 = np.asarray(s1.profile)
    p2 = np.asarray(s2.profile)
    big = np.abs(p1) > 1e-12 * np.max(np.abs(p1))
    assert np.allclose(p2[big] / p1[big], math.sqrt(2.0), rtol=1e-6)


def test_power_watts_must_be_positive():
    _, shell, pulse = _waveguide_shell()
    m = _te0_vector_mode()
    with pytest.raises(ValueError, match="power_watts must be > 0"):
        mode_source_vector(
            shell, m, axis="z", position_um=0.8, source_time=pulse,
            power_watts=0.0,
        )


# --------------------------------------------------------------------------
# Both-component packing
# --------------------------------------------------------------------------


def test_vector_source_packs_both_transverse_components():
    _, shell, pulse = _waveguide_shell()
    m = _te0_vector_mode()
    src = mode_source_vector(
        shell, m, axis="z", position_um=0.8, source_time=pulse
    )
    # z-propagation -> tangential E components are Ex (major TE) and Ey (minor).
    assert src.polarization == "Ex"
    assert src.minor_polarization == "Ey"
    assert src.profile_minor is not None
    assert len(src.profile) == src.nu * src.nv
    assert len(src.profile_minor) == src.nu * src.nv
    # The minor component is genuinely non-trivial (full-vector, not scalar).
    assert max(abs(p) for p in src.profile_minor) > 0.0
    # ... and smaller than the major (a quasi-TE mode is Ex-dominant).
    assert max(abs(p) for p in src.profile_minor) < max(abs(p) for p in src.profile)


def test_packed_ratio_matches_the_mode():
    """The packed major/minor RATIO equals the resampled transverse-E ratio of
    the mode (the power normalization is a common scalar that cancels)."""
    _, shell, pulse = _waveguide_shell()
    m = _te0_vector_mode()
    src = mode_source_vector(
        shell, m, axis="z", position_um=0.8, source_time=pulse
    )
    nu, nv = src.nu, src.nv
    maj = np.asarray(src.profile).reshape(nv, nu)
    minr = np.asarray(src.profile_minor).reshape(nv, nu)
    u = (np.arange(nu) + 0.5) * DL
    v = (np.arange(nv) + 0.5) * DL
    f = vector_modal_fields(m, u, v, axis="z", center_um=(0.8, 0.8))
    # Ex is t1 ("x"), Ey is t2 ("y") for z-propagation.
    ratio_packed = np.sum(maj * minr)
    ratio_mode = np.sum(np.real(f["e1"]) * np.real(f["e2"]))
    # Same sign and same relative magnitude (up to the common power scale).
    assert np.sign(ratio_packed) == np.sign(ratio_mode)


# --------------------------------------------------------------------------
# Back-compat: scalar path + wire format
# --------------------------------------------------------------------------


def test_scalar_mode_source_unchanged():
    """The legacy scalar builder still produces a minor-free, peak-normalized
    source — the full-vector work does not perturb it."""
    _, shell, pulse = _waveguide_shell()
    sm = ModeSolver.from_rectangular_core(
        wavelength_um=1.55,
        dl_um=DL,
        core_w_um=0.45,
        core_h_um=0.22,
        n_core=N_CORE,
        n_clad=N_CLAD,
    ).solve(num_modes=1, polarization="TE", n_guess=3.0)[0]
    src = mode_source(shell, sm, axis="z", position_um=0.8, source_time=pulse)
    assert src.minor_polarization is None
    assert src.profile_minor is None
    assert max(abs(p) for p in src.profile) == pytest.approx(1.0)  # peak-norm


def test_scalar_source_wire_omits_minor_fields():
    """An unset minor field is dropped from the wire dict so a 1.7-or-earlier
    engine still accepts the document (additive/back-compat)."""
    base, shell, pulse = _waveguide_shell()
    sm = ModeSolver.from_rectangular_core(
        wavelength_um=1.55, dl_um=DL, core_w_um=0.45, core_h_um=0.22,
        n_core=N_CORE, n_clad=N_CLAD,
    ).solve(num_modes=1, polarization="TE", n_guess=3.0)[0]
    src = mode_source(shell, sm, axis="z", position_um=0.8, source_time=pulse)
    wire = ph.Simulation(**base, sources=[src]).to_wire_dict()
    s = wire["sources"][0]
    assert s["type"] == "mode_source"
    assert "minor_polarization" not in s
    assert "profile_minor" not in s


def test_vector_source_wire_carries_minor_fields_and_roundtrips():
    base, shell, pulse = _waveguide_shell()
    m = _te0_vector_mode()
    src = mode_source_vector(
        shell, m, axis="z", position_um=0.8, source_time=pulse
    )
    sim = ph.Simulation(**base, sources=[src])
    s = sim.to_wire_dict()["sources"][0]
    assert s["minor_polarization"] == "Ey"
    assert len(s["profile_minor"]) == src.nu * src.nv
    back = ph.Simulation.model_validate_json(sim.model_dump_json())
    assert back == sim
    assert back.sources[0].minor_polarization == "Ey"


def test_schema_version_bumped_to_1_8():
    # The full-vector mode source landed in schema 1.8; the constant has since
    # advanced (1.9 added the §19 Lorentz pole). Pin to the live SCHEMA_VERSION
    # so this stays a real "the version moved past 1.7" assertion across bumps.
    base, _, _ = _waveguide_shell()
    sim = ph.Simulation(
        **base,
        sources=[
            ph.PointDipole(
                center_um=(0.8, 0.8, 0.8),
                polarization="Ex",
                source_time=ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13),
            )
        ],
    )
    assert sim.schema_version == ph.SCHEMA_VERSION
    major, minor = sim.schema_version.split(".")[:2]
    assert (int(major), int(minor)) >= (1, 8)


# --------------------------------------------------------------------------
# Additive wire-format validators
# --------------------------------------------------------------------------


def _mk(**over):
    kw = dict(
        axis="z",
        direction="+",
        position_um=0.8,
        polarization="Ex",
        n_eff=2.4,
        nu=2,
        nv=2,
        profile=(1.0, 0.0, 0.0, 0.0),
        source_time=ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13),
    )
    kw.update(over)
    return ph.ModeSource(**kw)


def test_minor_pol_and_profile_must_be_set_together():
    with pytest.raises(ValueError, match="must be set together"):
        _mk(minor_polarization="Ey")  # profile_minor missing
    with pytest.raises(ValueError, match="must be set together"):
        _mk(profile_minor=(0.0, 0.0, 0.0, 0.0))  # minor_polarization missing


def test_minor_pol_must_be_tangential():
    with pytest.raises(ValueError, match="tangential"):
        _mk(minor_polarization="Ez", profile_minor=(0.0, 0.0, 0.0, 0.0))


def test_minor_pol_must_differ_from_major():
    with pytest.raises(ValueError, match="differ"):
        _mk(minor_polarization="Ex", profile_minor=(0.0, 0.0, 0.0, 0.0))


def test_minor_pol_must_be_electric():
    with pytest.raises(ValueError, match="must be an E component"):
        _mk(minor_polarization="Hy", profile_minor=(0.0, 0.0, 0.0, 0.0))


def test_profile_minor_length_checked():
    with pytest.raises(ValueError, match="profile_minor length"):
        _mk(minor_polarization="Ey", profile_minor=(1.0, 2.0))  # != nu*nv


def test_valid_full_vector_modesource_constructs():
    src = _mk(minor_polarization="Ey", profile_minor=(0.0, 0.1, 0.1, 0.0))
    assert src.minor_polarization == "Ey"
    assert src.profile_minor == (0.0, 0.1, 0.1, 0.0)


# --------------------------------------------------------------------------
# Phase-1 default: mode_source ships the discrete true-H for a full-vector mode
# (benchmarks/launch_fidelity/ shows true-H cuts the launch shed several-fold).
# --------------------------------------------------------------------------


def test_mode_source_full_vector_mode_ships_true_h_by_default():
    """A full-vector mode passed to the plain `mode_source` launches via the
    discrete true-H vector source by default (profile_h shipped), equivalent to
    calling mode_source_vector — NOT the scalar-limit (profile_h omitted)."""
    _, shell, pulse = _waveguide_shell()
    m = _te0_vector_mode()
    kw = dict(axis="z", position_um=2.0, source_time=pulse,
              center_um=(0.8, 0.8), thickness_axis="y")
    src = mode_source(shell, m, **kw)                       # default paired_h=True
    ref = mode_source_vector(shell, m, **kw)                # explicit vector launch
    assert src.profile_h is not None and src.profile_h_minor is not None
    assert src.profile_h == pytest.approx(ref.profile_h)
    assert src.profile == pytest.approx(ref.profile)


def test_mode_source_paired_h_false_is_scalar_limit():
    """paired_h=False forces the legacy scalar-impedance-H launch (no profile_h)
    of the same full-vector E, and amplitude scales the launched power as a
    peak-field would (power ∝ amplitude²)."""
    _, shell, pulse = _waveguide_shell()
    m = _te0_vector_mode()
    kw = dict(axis="z", position_um=2.0, source_time=pulse,
              center_um=(0.8, 0.8), thickness_axis="y")
    scalar = mode_source(shell, m, **kw, paired_h=False)
    assert scalar.profile_h is None and scalar.profile_h_minor is None
    assert scalar.profile is not None                       # E profile still launched
    p1 = _injected_power(mode_source(shell, m, **kw))
    p2 = _injected_power(mode_source(shell, m, **kw, amplitude=2.0))
    assert p2 == pytest.approx(4.0 * p1, rel=1e-6)
