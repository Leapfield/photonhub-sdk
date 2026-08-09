"""``solve_modes_by_freq`` — the readout-side broadband (Tidy3D ``num_freqs``)
helper: auto-solve the FDE mode at each monitor frequency and fill a
``{freq_hz: Mode}`` map for ``mode_monitor`` / :class:`ModeMonitor`. Pure
Python (no engine), so it runs anywhere the FDE solver does."""

import math

import pytest

import photonhub as ph
from photonhub.plugins import (
    ModeSolver,
    VectorModeSolver,
    mode_monitor,
    solve_modes_by_freq,
)
from photonhub.plugins.mode_devices import C0

DL = 0.04  # um
N_CORE = 3.5
N_CLAD = 1.444
# O-band-ish triple: 1.26, 1.31, 1.36 um (the bilayer-taper sweep band).
WAVELENGTHS = (1.26, 1.31, 1.36)
FREQS = tuple(C0 / (wl * 1e-6) for wl in WAVELENGTHS)


def _solver(wavelength_um=1.31):
    return ModeSolver.from_rectangular_core(
        wavelength_um=wavelength_um,
        dl_um=DL,
        core_w_um=0.45,
        core_h_um=0.22,
        n_core=N_CORE,
        n_clad=N_CLAD,
    )


def test_at_wavelength_shares_cross_section():
    s = _solver(1.31)
    s2 = s.at_wavelength(1.26)
    assert s2.wavelength_um == pytest.approx(1.26)
    assert s2.eps is s.eps  # shared by reference, no re-rasterization
    assert (s2.dl_x_um, s2.dl_y_um) == (s.dl_x_um, s.dl_y_um)


def test_solve_modes_by_freq_keys_and_count():
    s = _solver()
    mbf = solve_modes_by_freq(s, FREQS, polarization="TE", n_guess=3.0)
    assert tuple(mbf.keys()) == FREQS  # input order preserved, float keys
    assert all(isinstance(m, ph.plugins.Mode) for m in mbf.values())
    # Each mode is solved AT its own frequency (wavelength), not the solver's.
    for f, m in mbf.items():
        assert m.wavelength_um == pytest.approx(C0 / f * 1e6)


def test_solve_modes_by_freq_neff_disperses_across_band():
    """The whole point: n_eff drifts across the band. A confined guided mode is
    MORE confined (higher n_eff) at the shorter wavelength / higher frequency."""
    s = _solver()
    mbf = solve_modes_by_freq(s, FREQS, polarization="TE", n_guess=3.0)
    neff = [mbf[f].n_eff for f in FREQS]  # FREQS ascending -> wavelength desc
    assert neff[0] > neff[-1]  # higher freq (1.26 um) is more confined
    assert max(neff) - min(neff) > 1e-3  # a real, resolvable drift


def test_solve_modes_by_freq_feeds_mode_monitor():
    base = dict(
        size_um=(1.6, 1.6, 4.0),
        grid=ph.UniformGridSpec(dl_um=DL),
        run=ph.RunSpec(n_steps=10),
        background=ph.Background(permittivity=N_CLAD**2),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
    )
    pulse = ph.GaussianPulse(freq0_hz=FREQS[1], fwidth_hz=2e13)
    sim = ph.Simulation(
        **base,
        sources=[ph.PointDipole(center_um=(0.8, 0.8, 0.5),
                                polarization="Ey", source_time=pulse)],
    )
    s = _solver()
    mbf = solve_modes_by_freq(s, FREQS, polarization="TE", n_guess=3.0)
    mm = mode_monitor(
        sim, mbf[FREQS[1]], axis="z", position_um=2.0,
        freqs_hz=FREQS, name="out", modes_by_freq=mbf,
    )
    assert mm.modes_by_freq is mbf
    assert mm.field_monitor.freqs_hz == FREQS


def test_solve_modes_by_freq_rejects_bad_inputs():
    s = _solver()
    with pytest.raises(ValueError, match="non-empty"):
        solve_modes_by_freq(s, [])
    with pytest.raises(ValueError, match="mode_index"):
        solve_modes_by_freq(s, FREQS, mode_index=-1)
    with pytest.raises(ValueError, match="> 0 Hz"):
        solve_modes_by_freq(s, [-1.0])


def test_solve_modes_by_freq_works_for_vector_solver():
    vs = VectorModeSolver.from_rectangular_core(
        wavelength_um=1.31, dl_um=DL, core_w_um=0.45, core_h_um=0.22,
        n_core=N_CORE, n_clad=N_CLAD,
    )
    mbf = solve_modes_by_freq(vs, FREQS, n_guess=3.0)
    assert tuple(mbf.keys()) == FREQS
    assert all(math.isfinite(m.n_eff) for m in mbf.values())
