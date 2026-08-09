"""Discrete-Yee mode solver: per-frequency bank + the ``use_yee`` cross-section
dispatch (now the library default for sources and monitors)."""
import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins import (
    mode_bank_on_cross_section,
    solve_mode_on_cross_section,
    solve_yee_mode,
    solve_yee_mode_bank,
)

C0 = 2.99792458e8


def _strip_sim(dl=0.02):
    """Si strip (450x220 nm, n=3.48) in SiO2 (n=1.444), TE0 along z."""
    core = ph.Structure(
        geometry=ph.Box(center_um=(0.9, 0.7, 3.0), size_um=(0.45, 0.22, 6.0)),
        medium=ph.Medium(permittivity=3.48 ** 2))
    return ph.Simulation(
        size_um=(1.8, 1.4, 6.0), grid=ph.UniformGridSpec(dl_um=dl),
        run={"n_steps": 1}, background=ph.Background(permittivity=1.444 ** 2),
        structures=(core,), boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=8,
        sources=[ph.PointDipole(
            center_um=(0.9, 0.7, 1.0), polarization="Ex",
            source_time=ph.GaussianPulse(freq0_hz=C0 / 1.55e-6, fwidth_hz=1e13))])


_WIN = dict(h_center_um=0.9, v_center_um=0.7, half_w_um=0.55, half_v_um=0.5, dl_um=0.02)


def test_yee_bank_matches_single_solve_per_lambda():
    """Each bank entry equals a direct single-λ solve, and n_eff disperses correctly."""
    sim = _strip_sim()
    freqs = [C0 / 1.50e-6, C0 / 1.55e-6, C0 / 1.60e-6]
    bank = solve_yee_mode_bank(sim, "z", 1.0, freqs, "TE", 0, **_WIN)
    assert set(bank) == {float(f) for f in freqs}
    for f in freqs:
        one = solve_yee_mode(sim, "z", 1.0, C0 / float(f) * 1e6, "TE", 0, **_WIN)
        assert bank[float(f)].n_eff == pytest.approx(one.n_eff, abs=1e-9)
    ne = [bank[float(f)].n_eff for f in freqs]     # 1.50, 1.55, 1.60 µm
    assert ne[0] > ne[1] > ne[2]                   # normal waveguide dispersion
    assert 2.0 < ne[1] < 3.0                        # a sane strip-TE0 n_eff


def test_cross_section_defaults_to_yee():
    """solve_mode_on_cross_section defaults to the engine-consistent Yee solver;
    use_yee=False falls back to the (distinct) FLM VectorModeSolver."""
    sim = _strip_sim()
    default = solve_mode_on_cross_section(sim, "z", 1.0, 1.55, "TE", 0, **_WIN)
    yee = solve_yee_mode(sim, "z", 1.0, 1.55, "TE", 0, **_WIN)
    flm = solve_mode_on_cross_section(sim, "z", 1.0, 1.55, "TE", 0, use_yee=False, **_WIN)
    assert default.n_eff == pytest.approx(yee.n_eff, abs=1e-9)   # default IS Yee
    assert abs(default.n_eff - flm.n_eff) > 1e-4                 # a different operator
    assert flm.n_eff > 1.444 and yee.n_eff > 1.444              # both guided


def test_mode_bank_dispatch_matches_yee_bank():
    """mode_bank_on_cross_section(use_yee=True, the default) == solve_yee_mode_bank."""
    sim = _strip_sim()
    freqs = [C0 / 1.53e-6, C0 / 1.57e-6]
    viadispatch = mode_bank_on_cross_section(sim, "z", 1.0, freqs, "TE", 0, **_WIN)
    direct = solve_yee_mode_bank(sim, "z", 1.0, freqs, "TE", 0, **_WIN)
    for f in freqs:
        assert viadispatch[float(f)].n_eff == pytest.approx(direct[float(f)].n_eff, abs=1e-9)


def test_wall_cell_normal_component_is_harmonic():
    """The KFJ sampler must apply the HARMONIC average to the component NORMAL
    to a wall even when the wall is contained in a single sample column (the
    normal used to come from the gradient of the brightest-material fill, which
    is 1 in BOTH bulks — so a single-column wall read a ZERO gradient and every
    component silently got the arithmetic average; two-column walls worked,
    making the defect registration-dependent)."""
    from photonhub.plugins.yee_mode import _kfj_at_offset

    dl = 0.02
    eps_hi, eps_lo = 3.48 ** 2, 1.444 ** 2
    # x-normal wall at x = (k + 0.4)*dl: strictly inside ONE node column
    # (block [k-1/2, k+1/2]*dl), fill f = 0.9 of the high-eps side.
    k = 20
    wall_x = (k + 0.4) * dl
    core = ph.Structure(
        geometry=ph.Box(center_um=(wall_x / 2, 0.7, 3.0),
                        size_um=(wall_x, 10.0, 6.0)),      # Si for x < wall_x
        medium=ph.Medium(permittivity=eps_hi))
    sim = ph.Simulation(
        size_um=(1.8, 1.4, 6.0), grid=ph.UniformGridSpec(dl_um=dl),
        run={"n_steps": 1}, background=ph.Background(permittivity=eps_lo),
        structures=(core,), boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=8,
        sources=[ph.PointDipole(
            center_um=(0.9, 0.7, 1.0), polarization="Ex",
            source_time=ph.GaussianPulse(freq0_hz=C0 / 1.55e-6, fwidth_hz=1e13))])

    ss = 10                     # 0.4 offset is exactly resolvable at ss=10
    epar, eh, ev = _kfj_at_offset(sim, "z", 3.0, 0.0, 0.0, 40, 40, dl,
                                  0.0, 0.0, ss, lambda s: float(s.medium.permittivity))
    iv = 20                     # a row well inside the (y-infinite) slab
    f = 0.9
    arith = f * eps_hi + (1 - f) * eps_lo
    harm = 1.0 / (f / eps_hi + (1 - f) / eps_lo)
    col = np.argmin(np.abs(epar[iv, :] - arith))            # the wall column
    assert epar[iv, col] == pytest.approx(arith, rel=1e-6)  # fill really is 0.9
    # the component ALONG the wall normal (x == h) must be harmonic ...
    assert eh[iv, col] == pytest.approx(harm, rel=1e-6)
    # ... and the tangential (y == v) component arithmetic.
    assert ev[iv, col] == pytest.approx(arith, rel=1e-6)


def test_kfj_reduce_handles_three_media_cells():
    """Regression (both twins): the diagonal-KFJ reduction used a two-phase
    (emax, emin, fill-of-emax) blend, which mis-assigns every INTERMEDIATE medium
    to emin — at a real air / BOX / Si triple junction the BOX sub-cells were
    counted as air. For isotropic constituents sharing one interface normal the
    Kottke/KFJ construction gives EXACTLY eps_par = <eps> and eps_perp = <1/eps>^-1
    for ANY number of media (the two-phase forms are only the N=2 case), so the
    reduction must average all sub-samples.
    """
    from photonhub.plugins.kfj_smoothing import _kfj_tensor_reduce
    from photonhub.plugins.yee_mode import _kfj_reduce

    AIR, BOX, SI = 1.0, 2.085, 12.067
    nh = nv = 3
    ss, dl = 4, 0.05
    # A horizontal air / BOX / Si stack whose MIDDLE cell row (iv=1) straddles all
    # three media: sub-rows 4=AIR, 5=BOX, 6..7=SI -> 4 AIR, 4 BOX, 8 SI sub-cells.
    eps_fine = np.empty((nv * ss, nh * ss), dtype=np.float64)
    eps_fine[0:5, :] = AIR
    eps_fine[5, :] = BOX
    eps_fine[6:, :] = SI

    mid = eps_fine[ss:2 * ss, 0:ss]          # the triple-junction cell's sub-block
    want_par = mid.mean()
    want_perp = 1.0 / np.mean(1.0 / mid)
    # The OLD two-phase reduction, for contrast: emax=SI, emin=AIR, so it counts
    # the BOX sub-row as AIR. Assert the scene actually distinguishes the two.
    f_old = (mid == mid.max()).mean()
    old_par = f_old * mid.max() + (1.0 - f_old) * mid.min()
    assert abs(old_par - want_par) > 0.2, "scene must actually distinguish the two"

    par_t, (exx, eyy, ezz) = _kfj_tensor_reduce(eps_fine, nh, nv, ss, dl)
    assert par_t[1, 1] == pytest.approx(want_par, rel=1e-12)
    assert ezz[1, 1] == pytest.approx(want_par, rel=1e-12)

    par_y, along_h, along_v = _kfj_reduce(eps_fine, nh, nv, ss, dl)
    assert par_y[1, 1] == pytest.approx(want_par, rel=1e-12)

    # eps_perp enters via d = eps_perp - eps_par on the NORMAL component. The
    # interface is horizontal (eps varies along v), so v carries the harmonic mean
    # and h stays arithmetic.
    assert along_v[1, 1] == pytest.approx(want_perp, rel=1e-9)
    assert along_h[1, 1] == pytest.approx(want_par, rel=1e-9)
    assert eyy[1, 1] == pytest.approx(want_perp, rel=1e-9)
    assert exx[1, 1] == pytest.approx(want_par, rel=1e-9)
