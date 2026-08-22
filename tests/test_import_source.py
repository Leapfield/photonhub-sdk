"""Import source — resampling, packaging, and sheet wiring (engine-free).

The Huygens-sheet launch physics is engine-validated through the mode and
Gaussian-beam paths (same ``equivalence_current_source`` machinery); what the
import path adds — and what these tests pin — is the resampling of a
user-supplied field map onto the plane's true Yee sample locations, the
quasi-plane-wave H fill, the VectorMode packaging contract, and the
source-builder wiring.
"""

import math

import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins import import_field, import_source
from photonhub.plugins.mode_overlap import ETA0

C0 = 2.99792458e8
WL_UM = 1.55
F0 = C0 / (WL_UM * 1e-6)

DL = 0.05
SX, SY, SZ = 2.0, 1.6, 1.0
XC, YC = SX / 2, SY / 2
N_BG = 1.444


def _sim():
    return ph.Simulation(
        size_um=(SX, SY, SZ),
        grid=ph.UniformGridSpec(dl_um=DL),
        run=ph.RunSpec(n_steps=10),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=6,
        background=ph.Background(permittivity=N_BG**2),
        sources=[ph.PointDipole(
            center_um=(XC, YC, SZ / 2), polarization="Ex",
            source_time=ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0))],
    )


def _pulse():
    return ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0)


def _gauss_data(w=0.4, tilt=2.0, n_data=161, extent=0.75):
    """A smooth analytic profile (Gaussian with a linear phase ramp) on a fine
    user grid: fine enough that bilinear resampling error << the tolerance."""
    c = np.linspace(-extent, extent, n_data)
    H, V = np.meshgrid(c, c)
    prof = np.exp(-(H**2 + V**2) / w**2) * np.exp(1j * tilt * H)
    return c, prof


def _analytic(fn, H, V):
    return fn(H, V)


def test_resampling_matches_analytic_profile():
    sim = _sim()
    w, tilt = 0.4, 2.0
    c, prof = _gauss_data(w, tilt)

    def f(H, V):
        return np.exp(-(H**2 + V**2) / w**2) * np.exp(1j * tilt * H)

    field = import_field(
        sim, axis="z", e_h=prof, e_v=0.3j * prof,
        coords_h_um=c, coords_v_um=c, source_time=_pulse())

    # reconstruct the sample grids the module used and compare against the
    # analytic profile (bilinear on a 161-point grid: error ~ (dx/w)^2 ~ 5e-4)
    from photonhub.plugins.gaussian_beam import _plane_grids
    h_node, v_node, _, grids = _plane_grids(
        sim, "z", h_center=XC, v_center=YC, half_w=0.75, half_v=0.75, dl=DL)
    H1, V1 = grids["mid_node"]
    H2, V2 = grids["node_mid"]

    want_ex = f(H1 - XC, V1 - YC)
    want_ey = 0.3j * f(H2 - XC, V2 - YC)
    norm = math.sqrt(float(np.sum(np.abs(want_ex)**2 + np.abs(want_ey)**2)))
    np.testing.assert_allclose(field.ex, want_ex / norm, atol=2e-3)
    np.testing.assert_allclose(field.ey, want_ey / norm, atol=2e-3)

    # packaging contract
    assert field.yee_staggered
    assert field.n_eff == pytest.approx(N_BG)
    l2 = float(np.sum(np.abs(field.ex)**2 + np.abs(field.ey)**2))
    assert l2 == pytest.approx(1.0, rel=1e-12)
    assert np.all(field.ez == 0) and np.all(field.hz == 0)


def test_plane_wave_h_fill_is_exact_pairing():
    sim = _sim()
    c, prof = _gauss_data()
    field = import_field(
        sim, axis="z", e_h=prof, e_v=0.5 * prof,
        coords_h_um=c, coords_v_um=c, source_time=_pulse())
    y0 = N_BG / ETA0
    # h_h = -(n/eta0) e_v sampled at the same points; h_v = +(n/eta0) e_h.
    # ex/hy share a grid and ey/hx share a grid, so the pairing is exact
    # (up to the joint normalization, which cancels in the ratio).
    np.testing.assert_allclose(field.hy, y0 * field.ex, rtol=1e-12)
    # hx is -y0 * (e_v resampled at hx's grid); ey is the same resample
    np.testing.assert_allclose(field.hx, -y0 * field.ey, rtol=1e-12)


def test_explicit_h_is_used_verbatim():
    sim = _sim()
    c, prof = _gauss_data()
    field = import_field(
        sim, axis="z", e_h=prof, e_v=0 * prof,
        h_h=0.25 * prof, h_v=1.5j * prof,
        coords_h_um=c, coords_v_um=c, source_time=_pulse())
    # with E along h only and H supplied explicitly, hy/ex keep the imported
    # ratio (both sampled on the same grid, joint normalization cancels)
    ratio = field.hy / np.where(np.abs(field.ex) > 1e-9, field.ex, np.nan)
    finite = np.isfinite(ratio)
    assert finite.any()
    np.testing.assert_allclose(ratio[finite], 1.5j, rtol=1e-9)


def test_import_source_builds_dipole_sheet():
    sim = _sim()
    c, prof = _gauss_data()
    dips = import_source(
        sim, axis="z", position_um=0.5, source_time=_pulse(),
        e_h=prof, e_v=0 * prof, coords_h_um=c, coords_v_um=c)
    assert len(dips) > 50                      # J and M sheets
    z = sorted({d.center_um[2] for d in dips})
    assert all(abs(zz - 0.5) <= DL for zz in z)
    pols = {d.polarization for d in dips}
    assert pols & {"Ex", "Ey"} and pols & {"Hx", "Hy"}

    # direction and power knobs pass through
    dips_b = import_source(
        sim, axis="z", position_um=0.5, source_time=_pulse(),
        e_h=prof, e_v=0 * prof, coords_h_um=c, coords_v_um=c,
        direction="-", power_watts=2.0)
    assert len(dips_b) > 50
    # raw-units launch
    dips_r = import_source(
        sim, axis="z", position_um=0.5, source_time=_pulse(),
        e_h=prof, e_v=0 * prof, coords_h_um=c, coords_v_um=c,
        power_watts=None)
    assert len(dips_r) > 50


def test_guards():
    sim = _sim()
    c, prof = _gauss_data()
    kw = dict(coords_h_um=c, coords_v_um=c, source_time=_pulse())
    with pytest.raises(ValueError, match="axis"):
        import_field(sim, axis="w", e_h=prof, e_v=prof, **kw)
    with pytest.raises(ValueError, match="both h_h and h_v"):
        import_field(sim, axis="z", e_h=prof, e_v=prof, h_h=prof, **kw)
    with pytest.raises(ValueError, match="shape"):
        import_field(sim, axis="z", e_h=prof[:-1], e_v=prof, **kw)
    with pytest.raises(ValueError, match="ascending"):
        import_field(sim, axis="z", e_h=prof, e_v=prof,
                     coords_h_um=c[::-1], coords_v_um=c,
                     source_time=_pulse())
    with pytest.raises(ValueError, match="identically zero"):
        import_field(sim, axis="z", e_h=0 * prof, e_v=0 * prof, **kw)
    with pytest.raises(ValueError, match="non-finite"):
        bad = prof.copy()
        bad[0, 0] = np.nan
        import_field(sim, axis="z", e_h=bad, e_v=prof, **kw)
    with pytest.raises(ValueError, match="direction"):
        import_source(sim, axis="z", position_um=0.5, source_time=_pulse(),
                      e_h=prof, e_v=prof, coords_h_um=c, coords_v_um=c,
                      direction="up")
    with pytest.raises(ValueError, match="power_watts"):
        import_source(sim, axis="z", position_um=0.5, source_time=_pulse(),
                      e_h=prof, e_v=prof, coords_h_um=c, coords_v_um=c,
                      power_watts=0.0)


def test_offcenter_placement_and_window_clip():
    sim = _sim()
    c, prof = _gauss_data(extent=0.5)
    # place near a corner: window must clip to the domain and still sample
    field = import_field(
        sim, axis="z", e_h=prof, e_v=0 * prof, coords_h_um=c, coords_v_um=c,
        center_um=(0.4, 0.4), source_time=_pulse())
    assert field.ex.shape[0] > 2 and field.ex.shape[1] > 2
    l2 = float(np.sum(np.abs(field.ex)**2 + np.abs(field.ey)**2))
    assert l2 == pytest.approx(1.0, rel=1e-12)
