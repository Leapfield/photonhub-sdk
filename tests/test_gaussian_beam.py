"""Gaussian-beam excitation source: analytic construction invariants + two
engine-gated launch checks.

The construction tests pin the beam physics (shape, paired H, phase convention,
Yee registration, symmetry folding); the engine tests pin what only a real run
can show — that the Huygens sheet launches this beam FORWARD, that it then
propagates like Maxwell (checked against exact angular-spectrum propagation, not
against the paraxial formulas), and that an offset waist actually focuses inside
the domain, which is the feature the old `gaussian_mode` + §18 hack could not
express at all.
"""
import math
import warnings

import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins import gaussian_beam, gaussian_beam_source
from photonhub.plugins._constants import ETA0
from photonhub.plugins.eq_current_source import _launched_power
from photonhub.runners.phsolver import find_solver

C0 = 2.99792458e8
LAM = 1.55
F0 = C0 / (LAM * 1e-6)
DL = 0.05
W0 = 1.2
ZR_VAC = math.pi * W0 ** 2 / LAM          # Rayleigh range in vacuum


def _pulse():
    return ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0)


def _sim(size=(8.0, 8.0, 7.0), symmetry=(0, 0, 0), eps=1.0, dl=DL, grid=None,
         **kw):
    """A bare shell: the beam only reads grid / size / symmetry / background."""
    return ph.Simulation(
        size_um=size, grid=grid or ph.UniformGridSpec(dl_um=dl),
        run={"n_steps": 10},
        background=ph.Background(permittivity=eps),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"), pml_num_layers=12,
        symmetry=symmetry,
        sources=[ph.PointDipole(center_um=(size[0] / 2, size[1] / 2, 1.5),
                                polarization="Ey", source_time=_pulse())],
        **kw)


def _plane_coords(beam, center, dl=DL):
    """The window's ``(h_node, v_node)`` ladders, recovered from the beam's own
    recorded grid snap the way ``mode_launch`` does."""
    nv, nh = beam.ex.shape
    h0 = center[0] + beam.center_offset_um[0] - 0.5 * (nh - 1) * dl
    v0 = center[1] + beam.center_offset_um[1] - 0.5 * (nv - 1) * dl
    return h0 + np.arange(nh) * dl, v0 + np.arange(nv) * dl


# --------------------------------------------------------------------------- #
# The beam profile
# --------------------------------------------------------------------------- #
def test_profile_is_the_requested_gaussian_with_flat_phase_at_the_waist():
    sim = _sim()
    b = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                      polarization="Ey", n=1.0)
    h, v = _plane_coords(b, (4.0, 4.0))
    # Ey (the v component) is sampled at (h, v + dl/2)
    H, V = np.meshgrid(h - 4.0, v + 0.5 * DL - 4.0, indexing="xy")
    want = np.exp(-(H / W0) ** 2 - (V / W0) ** 2)
    got = np.abs(b.ey)
    assert got.max() > 0
    assert np.allclose(got / got.max(), want / want.max(), atol=1e-12)
    # At its waist, at normal incidence, the beam is real (flat phase).
    assert np.abs(np.angle(b.ey)).max() < 1e-12


def test_mfd_is_twice_the_waist():
    sim = _sim()
    a = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM, n=1.0)
    b = gaussian_beam(sim, axis="z", mfd_um=2 * W0, wavelength_um=LAM, n=1.0)
    assert np.allclose(a.ex, b.ex)


def test_elliptical_waist_sizes_each_in_plane_axis():
    sim = _sim(size=(12.0, 12.0, 7.0))
    b = gaussian_beam(sim, axis="z", waist_um=(1.0, 2.5), wavelength_um=LAM,
                      polarization="Ey", n=1.0)
    h, v = _plane_coords(b, (6.0, 6.0))
    H, V = np.meshgrid(h - 6.0, v + 0.5 * DL - 6.0, indexing="xy")
    want = np.exp(-(H / 1.0) ** 2 - (V / 2.5) ** 2)
    got = np.abs(b.ey)
    assert np.allclose(got / got.max(), want / want.max(), atol=1e-12)


def test_default_window_is_three_field_radii_and_is_domain_clipped():
    sim = _sim(size=(12.0, 12.0, 7.0))
    b = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM, n=1.0)
    h, _ = _plane_coords(b, (6.0, 6.0))
    assert (h[-1] - h[0]) / 2 == pytest.approx(3 * W0, abs=DL)
    # A silly request cannot blow the window past the domain.
    big = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM, n=1.0,
                        half_w_um=1e4, half_v_um=1e4)
    assert big.ex.shape[1] <= round(12.0 / DL) + 1


# --------------------------------------------------------------------------- #
# Polarization
# --------------------------------------------------------------------------- #
def test_polarization_names_pick_the_in_plane_component():
    sim = _sim()
    # z-cut -> in-plane axes (x, y); "Ey" is the second (vertical) one.
    b = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                      polarization="Ey", n=1.0)
    assert np.abs(b.ey).max() > 0
    assert np.abs(b.ex).max() < 1e-15 * np.abs(b.ey).max()
    a = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                      polarization="Ex", n=1.0)
    assert np.abs(a.ex).max() > 0 and np.abs(a.ey).max() < 1e-15 * np.abs(a.ex).max()


def test_pol_angle_splits_the_two_transverse_components():
    sim = _sim()
    b = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                      pol_angle=math.pi / 4, n=1.0)
    px = float(np.sum(np.abs(b.ex) ** 2))
    py = float(np.sum(np.abs(b.ey) ** 2))
    assert px == pytest.approx(py, rel=1e-9)
    assert b.te_fraction == pytest.approx(0.5, abs=1e-9)


def test_polarization_along_the_propagation_axis_is_rejected():
    sim = _sim()
    with pytest.raises(ValueError, match="not tangential"):
        gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                      polarization="Ez", n=1.0)


# --------------------------------------------------------------------------- #
# The paired H, power, and Yee registration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pol", ["Ex", "Ey"])
def test_paired_h_is_the_forward_huygens_pair(pol):
    """H = (n/eta0) k_hat x E: at normal incidence H_v = +(n/eta0) E_h and
    H_h = -(n/eta0) E_v, which is what makes the sheet radiate one-sided."""
    n = 1.45
    sim = _sim(eps=n ** 2)
    b = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                      polarization=pol)
    y0 = n / ETA0
    assert np.allclose(b.hy, y0 * b.ex, atol=1e-18)
    assert np.allclose(b.hx, -y0 * b.ey, atol=1e-18)
    assert np.abs(b.ez).max() == 0.0 and np.abs(b.hz).max() == 0.0
    assert b.n_eff == pytest.approx(n)


def test_launched_power_is_forward_and_matches_the_scalar_limit():
    n = 1.45
    b = gaussian_beam(_sim(eps=n ** 2), axis="z", waist_um=W0,
                      wavelength_um=LAM, polarization="Ey")
    # The transverse-E pair is jointly L2-normalized, so the discrete Poynting
    # sum is exactly (n / 2 eta0) * dl^2 in m^2.
    assert _launched_power(b, DL) == pytest.approx(
        n / (2 * ETA0) * (DL * 1e-6) ** 2, rel=1e-12)


def test_components_sit_on_their_own_yee_sublattices():
    """E_h/H_v at (h+1/2, v) and E_v/H_h at (h, v+1/2) — the engine's in-plane
    Yee offsets. Centre the beam ON a node: the node-sampled component then peaks
    on a single centre row while the half-cell-sampled one straddles it."""
    sim = _sim()
    b = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                      polarization="Ey", n=1.0, center_um=(4.0, 4.0))
    h, v = _plane_coords(b, (4.0, 4.0))
    # Ey ~ exp(-(h-4)^2/w^2 - (v+dl/2-4)^2/w^2): peaks at the h node nearest 4.0
    # and straddles in v.
    col = np.abs(b.ey).max(axis=0)
    assert h[int(np.argmax(col))] == pytest.approx(4.0, abs=1e-9)
    row = np.abs(b.ey).max(axis=1)
    top2 = np.sort(row)[-2:]
    assert top2[0] == pytest.approx(top2[1], rel=1e-12)   # two equal, straddling


def test_sheet_stamps_on_the_two_engine_planes():
    sim = _sim()
    pulse = _pulse()
    dips = gaussian_beam_source(sim, axis="z", position_um=1.5,
                                source_time=pulse, waist_um=W0,
                                polarization="Ey", n=1.0)
    assert len(dips) > 1000
    k0 = round(1.5 / DL)
    for d in dips:
        x, y, z = d.center_um
        if d.polarization.startswith("H"):
            assert z == pytest.approx((k0 - 0.5) * DL, abs=1e-9)
        else:
            assert z == pytest.approx(k0 * DL, abs=1e-9)
        for c in (x, y):
            frac = (c / DL) % 1.0
            assert min(frac, abs(frac - 0.5), abs(frac - 1.0)) < 1e-6
    assert {"Ex", "Ey"} & {d.polarization for d in dips}
    assert {"Hx", "Hy"} & {d.polarization for d in dips}


def test_power_watts_scales_the_dipole_amplitudes():
    sim = _sim()
    pulse = _pulse()
    kw = dict(axis="z", position_um=1.5, source_time=pulse, waist_um=W0,
              polarization="Ey", n=1.0)
    a1 = max(d.amplitude for d in gaussian_beam_source(sim, power_watts=1.0, **kw))
    a4 = max(d.amplitude for d in gaussian_beam_source(sim, power_watts=4.0, **kw))
    assert a4 / a1 == pytest.approx(2.0, rel=1e-9)       # power proportional to amplitude^2


# --------------------------------------------------------------------------- #
# Offset waist and off-normal launch — the phase-carrying features
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [2.5, -2.5])
def test_offset_waist_grows_the_spot_and_curves_the_wavefront(d):
    sim = _sim(size=(14.0, 14.0, 7.0))
    b = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                      polarization="Ey", n=1.0, waist_distance_um=d)
    h, v = _plane_coords(b, (7.0, 7.0))
    H, V = np.meshgrid(h - 7.0, v + 0.5 * DL - 7.0, indexing="xy")
    w = W0 * math.sqrt(1 + (d / ZR_VAC) ** 2)
    want = np.exp(-(H ** 2 + V ** 2) / w ** 2)
    got = np.abs(b.ey)
    assert np.allclose(got / got.max(), want / want.max(), atol=1e-12)
    # Wavefront curvature, in the sheet's own phasor convention (forward is
    # e^{-ik.r}): a CONVERGING beam (waist ahead, d < 0) LEADS in phase off axis
    # — the rim has further to go to the focus, so it must leave earlier — and a
    # diverging one lags. Backwards, and the beam defocuses where it should focus.
    k = 2 * math.pi / LAM
    inv_r = d / (d * d + ZR_VAC ** 2)
    want_phase = -0.5 * k * inv_r * (H ** 2 + V ** 2)
    m = got > 1e-3 * got.max()
    residual = np.angle(b.ey * np.exp(-1j * want_phase))[m]
    assert np.abs(residual).max() < 1e-9         # phase is exactly the curvature
    # ... and it leads/lags the way the focus direction demands.
    r2 = 4.0
    off = np.unravel_index(np.argmin(np.abs(H ** 2 + V ** 2 - r2)), H.shape)
    assert abs(want_phase[off]) < math.pi        # unwrapped at this radius
    assert np.angle(b.ey[off]) * (1 if d < 0 else -1) > 0.1


def test_tilt_puts_the_angular_spectrum_centroid_at_k_sin_theta():
    """The invariant of an off-normal launch. (The amplitude centroid's walk-off
    is NOT tan(theta) for a tightly focused beam — it is <kx/kz>, which Jensen
    pushes above tan(theta) — so the spectrum centroid is what to assert.)"""
    sim = _sim(size=(16.0, 16.0, 7.0))
    k = 2 * math.pi / LAM
    for deg in (5.0, 10.0, 20.0):
        th = math.radians(deg)
        b = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                          polarization="Ey", n=1.0, angle_theta=th,
                          angle_phi=0.0, half_w_um=7.0, half_v_um=7.0)
        # conj: the module builds in the sheet's e^{+iwt} convention.
        p = np.abs(np.fft.fft2(np.conj(b.ey))) ** 2
        kx = 2 * math.pi * np.fft.fftfreq(b.ey.shape[1], d=DL)
        assert float((p.sum(0) * kx).sum() / p.sum()) == \
            pytest.approx(k * math.sin(th), rel=2e-3)
        assert b.n_eff == pytest.approx(math.cos(th))      # axial phase constant


def test_angle_phi_steers_and_direction_minus_keeps_its_meaning():
    """`angle_phi` is measured about the LAUNCH direction, so a '-' launch must
    pre-rotate the azimuth by pi (the sheet reverses the whole k vector)."""
    sim = _sim(size=(16.0, 16.0, 7.0))
    th = math.radians(10.0)
    kw = dict(axis="z", waist_um=W0, wavelength_um=LAM, polarization="Ey",
              n=1.0, angle_theta=th, half_w_um=7.0, half_v_um=7.0)
    fwd = gaussian_beam(sim, angle_phi=0.0, direction="+", **kw)
    bwd = gaussian_beam(sim, angle_phi=0.0, direction="-", **kw)
    flip = gaussian_beam(sim, angle_phi=math.pi, direction="+", **kw)
    assert np.allclose(bwd.ey, flip.ey, atol=1e-15)
    assert not np.allclose(bwd.ey, fwd.ey)


# --------------------------------------------------------------------------- #
# Symmetry, background index, broadband, and input validation
# --------------------------------------------------------------------------- #
def test_symmetry_plane_clips_the_window_and_keeps_the_open_quadrant():
    full = _sim(size=(12.0, 12.0, 7.0))
    quarter = _sim(size=(6.0, 6.0, 7.0), symmetry=(1, -1, 0))
    bf = gaussian_beam(full, axis="z", waist_um=W0, wavelength_um=LAM,
                       polarization="Ey", n=1.0)
    bq = gaussian_beam(quarter, axis="z", waist_um=W0, wavelength_um=LAM,
                       polarization="Ey", n=1.0, center_um=(0.0, 0.0))
    hq, vq = _plane_coords(bq, (0.0, 0.0))
    assert hq[0] == pytest.approx(0.0, abs=1e-12)
    assert vq[0] == pytest.approx(0.0, abs=1e-12)
    # The folded beam is the open quadrant of the full one (peak-normalized).
    hf, vf = _plane_coords(bf, (6.0, 6.0))
    ih = [int(np.argmin(np.abs(hf - 6.0 - t))) for t in hq]
    iv = [int(np.argmin(np.abs(vf - 6.0 - t))) for t in vq]
    sub = np.abs(bf.ey)[np.ix_(iv, ih)]
    assert np.allclose(sub / sub.max(), np.abs(bq.ey) / np.abs(bq.ey).max(),
                       atol=1e-12)


def _graded(size, dl, graded_axes=("x", "y")):
    """A §15 graded grid: `dl` everywhere except a 2x-coarsened outer third on
    each named axis, so the beam window genuinely straddles two spacings."""
    coords = {}
    for ax in graded_axes:
        L = size["xyz".index(ax)]
        q, x = [0.0], 0.0
        while x < L - 1e-9:
            step = dl if x < 0.66 * L else 2 * dl
            x += step
            q.append(round(x, 12))
        coords[ax] = tuple(q[:-1])
    return ph.GradedGridSpec(dl_um=dl, coords=coords)


def test_degenerate_graded_grid_reproduces_the_uniform_beam():
    size = (12.0, 12.0, 7.0)
    n = round(12.0 / DL)
    deg = ph.GradedGridSpec(dl_um=DL, coords={
        "x": tuple(float(i * DL) for i in range(n)),
        "y": tuple(float(i * DL) for i in range(n))})
    kw = dict(axis="z", waist_um=W0, wavelength_um=LAM, polarization="Ey",
              n=1.0, half_w_um=4.0, half_v_um=4.0)
    u = gaussian_beam(_sim(size=size), **kw)
    g = gaussian_beam(_sim(size=size, grid=deg), **kw)
    assert g.x_coords_um is not None and u.x_coords_um is None
    assert g.ey.shape == u.ey.shape
    assert np.allclose(g.ey, u.ey, atol=1e-14)


def test_beam_samples_a_graded_window_on_its_true_nodes():
    size = (12.0, 12.0, 7.0)
    grid = _graded(size, DL)
    sim = _sim(size=size, grid=grid)
    b = gaussian_beam(sim, axis="z", waist_um=W0, wavelength_um=LAM,
                      polarization="Ey", n=1.0, half_w_um=5.0, half_v_um=5.0)
    assert b.x_coords_um is not None and len(b.x_coords_um) == b.ey.shape[1]
    dq = np.diff(np.asarray(b.x_coords_um))
    assert dq.max() > 1.5 * dq.min()             # genuinely nonuniform window
    # placement metadata is the graded form (ladder midpoint), not lo + (n-1)dl/2
    assert b.center_offset_um[0] == pytest.approx(
        0.5 * float(b.x_coords_um[0] + b.x_coords_um[-1]), abs=1e-12)
    # Ey is sampled at (h_node, v_mid) on THOSE nodes, not on a dl ladder.
    nodes = np.asarray(b.x_coords_um) + 6.0
    v_nodes = np.asarray(b.y_coords_um) + 6.0
    v_mid = v_nodes + 0.5 * np.append(np.diff(v_nodes), np.diff(v_nodes)[-1])
    H, V = np.meshgrid(nodes - 6.0, v_mid - 6.0, indexing="xy")
    want = np.exp(-(H / W0) ** 2 - (V / W0) ** 2)
    got = np.abs(b.ey)
    assert np.allclose(got / got.max(), want / want.max(), atol=1e-12)
    # ...and the sheet stamps on those same nodes.
    dips = gaussian_beam_source(sim, axis="z", position_um=1.5,
                                source_time=_pulse(), waist_um=W0,
                                polarization="Ey", n=1.0, half_w_um=5.0,
                                half_v_um=5.0)
    allowed = set(np.round(np.concatenate([nodes, nodes + 0.5 * np.append(
        np.diff(nodes), np.diff(nodes)[-1])]), 9))
    for d in dips:
        assert round(d.center_um[0], 9) in allowed


def test_index_defaults_to_the_simulation_background():
    b = gaussian_beam(_sim(eps=1.45 ** 2), axis="z", waist_um=W0,
                      wavelength_um=LAM)
    assert b.n_eff == pytest.approx(1.45)


def test_frequency_defaults_to_the_pulse_centre():
    b = gaussian_beam(_sim(), axis="z", waist_um=W0, source_time=_pulse(), n=1.0)
    assert b.wavelength_um == pytest.approx(C0 / F0 * 1e6)


def test_broadband_builds_one_sheet_per_frequency():
    sim = _sim()
    pulse = _pulse()
    freqs = (0.95 * F0, F0, 1.05 * F0)
    kw = dict(axis="z", position_um=1.5, source_time=pulse, waist_um=W0,
              polarization="Ey", n=1.0, waist_distance_um=2.0)
    one = gaussian_beam_source(sim, **kw)
    many = gaussian_beam_source(sim, freqs_hz=freqs, **kw)
    assert len(many) == pytest.approx(3 * len(one), rel=0.25)
    bands = {d.source_time.band_freqs_hz for d in many}
    assert bands == {tuple(sorted(float(f) for f in freqs))}
    assert {d.source_time.carrier_index for d in many} == {0, 1, 2}
    # A single entry is the plain single-frequency launch.
    assert len(gaussian_beam_source(sim, freqs_hz=(F0,), **kw)) == len(one)


@pytest.mark.parametrize("kw, match", [
    (dict(), "exactly one of waist_um"),
    (dict(waist_um=1.0, mfd_um=2.0), "exactly one of waist_um"),
    (dict(waist_um=-1.0), "must be > 0"),
    (dict(waist_um=1.0, polarization="Ex", pol_angle=0.1), "at most one of"),
    (dict(waist_um=1.0, n=0.0), "index n must be > 0"),
    (dict(waist_um=1.0, angle_theta=math.pi / 2), "angle_theta"),
    (dict(waist_um=1.0, direction="up"), "direction"),
    (dict(waist_um=1.0, axis="q"), "axis"),
])
def test_bad_inputs_raise(kw, match):
    sim = _sim()
    call = dict(axis="z", wavelength_um=LAM, n=1.0)
    call.update(kw)
    with pytest.raises(ValueError, match=match):
        gaussian_beam(sim, **call)


def test_beam_frozen_far_off_the_pulse_centre_warns():
    """The sheet phases its half-cell straddle at the pulse centre, so a beam
    built at a materially different wavelength is launched detuned."""
    sim = _sim()
    kw = dict(axis="z", position_um=1.5, source_time=_pulse(), waist_um=W0,
              polarization="Ey", n=1.0)
    with pytest.warns(UserWarning, match="detuned"):
        gaussian_beam_source(sim, wavelength_um=1.2 * LAM, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # on the centre: silent
        gaussian_beam_source(sim, wavelength_um=LAM, **kw)
        gaussian_beam_source(sim, **kw)


def test_beam_entirely_off_the_domain_raises():
    sim = _sim()
    with pytest.raises(ValueError, match="identically zero"):
        gaussian_beam(sim, axis="z", waist_um=1e-3, wavelength_um=LAM, n=1.0,
                      center_um=(-50.0, -50.0), half_w_um=0.2, half_v_um=0.2)


# --------------------------------------------------------------------------- #
# Engine-gated: the launch itself
# --------------------------------------------------------------------------- #
def _angular_spectrum(u, dl, dz, lam):
    """Exact scalar vacuum propagation over ``dz`` of a RECORDED monitor plane.

    Recorded ``field_dft`` phasors carry forward propagation as ``e^{+ikz}`` —
    the CONJUGATE of the convention the source sheet's per-dipole phases use
    (see the gaussian_beam module docstring), so the textbook kernel applies
    here unconjugated."""
    ny, nx = u.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dl)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dl)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k = 2 * np.pi / lam
    kz2 = k * k - KX ** 2 - KY ** 2
    prop = np.where(kz2 > 0, np.exp(1j * np.sqrt(np.maximum(kz2, 0.0)) * dz), 0.0)
    return np.fft.ifft2(np.fft.fft2(u) * prop)


def _beam_run(monitor_offsets, *, z_src=1.5, size=(8.0, 8.0, 7.0), fields=None,
              **beam):
    pulse = _pulse()
    base = dict(size_um=size, grid=ph.UniformGridSpec(dl_um=DL),
                run={"n_steps": 2600}, background=ph.Background(permittivity=1.0),
                boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
                pml_num_layers=12)
    shell = ph.Simulation(**base, sources=[ph.PointDipole(
        center_um=(size[0] / 2, size[1] / 2, z_src), polarization="Ey",
        source_time=pulse)])
    dips = gaussian_beam_source(shell, axis="z", position_um=z_src,
                                source_time=pulse, polarization="Ey", n=1.0,
                                power_watts=1.0, **beam)
    nx = round(size[0] / DL)
    lo, hi = 1.25 * DL, (nx - 1 + 0.25) * DL
    mons, zs = [], {}
    for i, d in enumerate(monitor_offsets):
        zp = (round((z_src + d) / DL - 0.25) + 0.25) * DL
        mons.append(ph.FieldDftMonitor(
            name=f"p{i}", center_um=(0.5 * (lo + hi), 0.5 * (lo + hi), zp),
            size_um=(hi - lo, hi - lo, 0.0),
            fields=fields or ("Ex", "Ey", "Hx", "Hy"), freqs_hz=(F0,)))
        zs[f"p{i}"] = zp - z_src
    sim = ph.Simulation(**base, sources=tuple(dips), monitors=tuple(mons))
    return ph.run_local(sim, solver_path=find_solver(), quiet=True,
                        timeout=3600), zs


def _comp(data, name, letter):
    a = data[name].sel(component=letter)
    if "f" in a.dims:
        a = a.isel(f=0)
    return a.squeeze(drop=True).transpose("y", "x")


def _flux(data, name):
    from photonhub.plugins.mode_overlap import _cell_widths
    e1, e2 = _comp(data, name, "Ex"), _comp(data, name, "Ey")
    h1, h2 = _comp(data, name, "Hx"), _comp(data, name, "Hy")
    cx = np.asarray(e1.coords["x"].values, float)
    cy = np.asarray(e1.coords["y"].values, float)
    s = 0.5 * np.real(e1.values * np.conj(h2.values)
                      - e2.values * np.conj(h1.values))
    return float(np.sum(s * np.outer(_cell_widths(cy), _cell_widths(cx))))


def _fit_spot(data, name, center):
    """1/e field radii of |Ey| from a log-quadratic fit over the plane."""
    a = _comp(data, name, "Ey")
    A = np.abs(a.values)
    X, Y = np.meshgrid(np.asarray(a.coords["x"].values, float) - center,
                       np.asarray(a.coords["y"].values, float) - center,
                       indexing="xy")
    m = A > 0.05 * A.max()
    M = np.stack([np.ones(m.sum()), X[m], X[m] ** 2, Y[m], Y[m] ** 2], axis=1)
    c, *_ = np.linalg.lstsq(M, np.log(A[m]), rcond=None)
    return 1 / math.sqrt(-c[2]), 1 / math.sqrt(-c[4])


@pytest.mark.skipif(find_solver() is None, reason="needs a phsolver binary")
def test_launch_is_forward_lossless_and_propagates_like_maxwell():
    """The beam goes the right way, keeps its power, and the field two and a half
    microns downstream is what EXACT vacuum diffraction of the near plane gives.
    (Compared against angular-spectrum propagation, not against w(z): at
    w0 = 0.77 lambda the paraxial spot formula is itself ~7% off one Rayleigh
    range out — see the module note.)"""
    data, zs = _beam_run((0.5, 3.0, -0.7), waist_um=W0)
    near, far, back = _flux(data, "p0"), _flux(data, "p1"), -_flux(data, "p2")
    assert near > 0 and far > 0
    assert back / near < 0.01                       # one-sided launch
    assert abs(1 - far / near) < 0.005              # no loss in free space

    un, uf = _comp(data, "p0", "Ey").values, _comp(data, "p1", "Ey").values
    pred = _angular_spectrum(un, DL, zs["p1"] - zs["p0"], LAM)
    w = np.abs(uf) / np.abs(uf).max()
    m = w > 0.02
    overlap = abs(np.vdot(pred[m], uf[m])) / (
        np.linalg.norm(pred[m]) * np.linalg.norm(uf[m]))
    assert overlap > 0.998


@pytest.mark.skipif(find_solver() is None, reason="needs a phsolver binary")
def test_offset_waist_focuses_inside_the_domain():
    """`waist_distance_um = -2` puts the waist two microns downstream: the beam
    must NARROW to w0 there and re-expand after. This is the launch the §18
    real-profile mode source cannot express (the converging wavefront is pure
    phase), and it is the check that catches a flipped phase convention — with
    the sign backwards the beam simply keeps diverging."""
    data, zs = _beam_run((0.05, 2.0, 3.5), waist_um=W0, waist_distance_um=-2.0,
                         half_w_um=4.0, half_v_um=4.0)
    w_in = _fit_spot(data, "p0", 4.0)
    w_focus = _fit_spot(data, "p1", 4.0)
    w_out = _fit_spot(data, "p2", 4.0)
    # launched wide, waisted at the focus, expanding again
    assert min(w_in) > 1.15 * max(w_focus)
    assert min(w_out) > 1.10 * max(w_focus)
    for w in w_focus:
        assert w == pytest.approx(W0, rel=0.10)
    for w in w_in:                                   # w(2 um) off the waist
        assert w == pytest.approx(W0 * math.sqrt(1 + (2.0 / ZR_VAC) ** 2),
                                  rel=0.10)
