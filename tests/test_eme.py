"""Minimal eigenmode-expansion (EME) propagator — correctness invariants.

Validates ``photonhub.plugins.eme`` on the building blocks (star product,
interface, propagation) and end-to-end on a straight guide and a tapered guide.
The load-bearing checks are *internal physics invariants* that hold regardless of
mesh accuracy:

* a **straight** (constant-cross-section) guide transmits perfectly
  (``T == 1``, ``R == 0`` to machine precision) — the matched-interface
  pass-through;
* every interface S-matrix is **unitary** (energy-conserving) on the guided
  basis, so any lossless cascade has ``T_total + R_total == 1``;
* a gentle **taper** is highly transmitting and converges toward ``T -> 1`` with
  vanishing reflection as the staircase is refined (the adiabatic limit).

A cross-check against the Tidy3D FDTD reference for the canonical 0.45->0.80 um
taper (``T ~= 0.997``) lives in ``benchmarks/eme/taper_eme.py`` — EME sits a hair
above it because a guided-mode basis cannot radiate (documented in the module).
"""

import numpy as np
import pytest

from photonhub.plugins.eme import (
    EMEResult,
    cascade,
    interface_smatrix,
    propagation_smatrix,
    rectangular_base_section,
    run_eme,
    run_eme_band,
    star_product,
    waveguide_section,
)

# Coarse + small for speed: we test the cascade machinery, not mode accuracy.
WL_UM = 1.31
DL_UM = 0.05
CORE_H_UM = 0.22
N_CORE, N_CLAD = 3.5, 1.444
WIN_W_UM, WIN_H_UM = 2.0, 1.3
MARGIN = 0.05  # drop near-cutoff (under-resolved) modes


def _sec(core_w_um, length_um=0.0, num_modes=2):
    return waveguide_section(
        wavelength_um=WL_UM,
        dl_um=DL_UM,
        core_w_um=core_w_um,
        core_h_um=CORE_H_UM,
        n_core=N_CORE,
        n_clad=N_CLAD,
        window_w_um=WIN_W_UM,
        window_h_um=WIN_H_UM,
        num_modes=num_modes,
        length_um=length_um,
        neff_margin=MARGIN,
    )


def _taper_sections(n_slabs, *, w_in=0.45, w_out=0.80, taper_len=3.0, num_modes=2):
    secs = [_sec(w_in, 0.0, num_modes)]
    for k in range(n_slabs):
        wk = w_in + (w_out - w_in) * ((k + 0.5) / n_slabs)
        secs.append(_sec(wk, taper_len / n_slabs, num_modes))
    secs.append(_sec(w_out, 0.0, num_modes))
    return secs


def _block(s):
    s11, s12, s21, s22 = s
    return np.block([[s11, s12], [s21, s22]])


def _unitarity_error(s):
    m = _block(s)
    return float(np.abs(m.conj().T @ m - np.eye(m.shape[0])).max())


# --- fixtures (cache the expensive mode solves) ----------------------------


@pytest.fixture(scope="module")
def modes_narrow():
    return _sec(0.45).modes


@pytest.fixture(scope="module")
def modes_wide():
    return _sec(0.80).modes


# --- star product (pure linear algebra) ------------------------------------


def _passthrough(n):
    z = np.zeros((n, n), dtype=complex)
    i = np.eye(n, dtype=complex)
    return z, i, i, z


def _random_smatrix(n, seed):
    rng = np.random.default_rng(seed)
    blocks = [
        (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.1
        for _ in range(4)
    ]
    return tuple(blocks)


def test_star_product_passthrough_is_identity_element():
    s = _random_smatrix(3, seed=1)
    pt = _passthrough(3)
    left = star_product(pt, s)
    right = star_product(s, pt)
    for got, want in zip(left, s):
        assert np.allclose(got, want, atol=1e-12)
    for got, want in zip(right, s):
        assert np.allclose(got, want, atol=1e-12)


def test_cascade_single_segment_is_itself():
    s = _random_smatrix(2, seed=2)
    for got, want in zip(cascade([s]), s):
        assert np.allclose(got, want)


def test_cascade_requires_a_segment():
    with pytest.raises(ValueError):
        cascade([])


# --- propagation -----------------------------------------------------------


def test_propagation_is_diagonal_phase_and_unitary(modes_wide):
    s11, s12, s21, s22 = propagation_smatrix(modes_wide, length_um=1.7)
    n = len(modes_wide)
    assert np.allclose(s11, 0) and np.allclose(s22, 0)
    # forward/backward are equal diagonal phase, |phase| == 1 (lossless)
    assert np.allclose(s12, s21)
    assert np.allclose(np.abs(np.diag(s12)), 1.0)
    assert np.allclose(s12 - np.diag(np.diag(s12)), 0)  # strictly diagonal
    assert _unitarity_error((s11, s12, s21, s22)) < 1e-12


# --- interface -------------------------------------------------------------


def test_single_mode_identical_interface_is_exact(modes_wide):
    # one channel each side: pass-through is machine-exact (G is the scalar 1).
    m = modes_wide[:1]
    s11, s12, s21, s22 = interface_smatrix(m, m, DL_UM, DL_UM)
    assert abs(s11[0, 0]) < 1e-12
    assert abs(s22[0, 0]) < 1e-12
    assert s21[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert s12[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_identical_interface_is_passthrough(modes_wide):
    n = len(modes_wide)
    s = interface_smatrix(modes_wide, modes_wide, DL_UM, DL_UM)
    s11, s12, s21, s22 = s
    # Unitarity (energy conservation) is exact at any resolution.
    assert _unitarity_error(s) < 1e-10
    # The matched step is a pass-through up to the FDE's residual reciprocity
    # error in the multimode off-diagonals (-> 0 as the grid is refined); coarse
    # here for speed, so use a resolution-appropriate tolerance.
    assert np.allclose(s11, 0.0, atol=2e-3)
    assert np.allclose(s22, 0.0, atol=2e-3)
    assert np.allclose(s21, np.eye(n), atol=2e-3)
    assert np.allclose(s12, np.eye(n), atol=2e-3)


def test_interface_is_unitary(modes_narrow, modes_wide):
    n = min(len(modes_narrow), len(modes_wide))
    s = interface_smatrix(modes_narrow[:n], modes_wide[:n], DL_UM, DL_UM)
    assert _unitarity_error(s) < 1e-10


def test_interface_requires_equal_mode_counts(modes_wide):
    if len(modes_wide) < 2:
        pytest.skip("need >=2 modes to form an unequal pair")
    with pytest.raises(ValueError):
        interface_smatrix(modes_wide[:1], modes_wide[:2], DL_UM, DL_UM)


# --- straight guide: the load-bearing invariant ----------------------------


@pytest.mark.parametrize("num_modes", [1, 2])
def test_straight_guide_transmits_perfectly(num_modes):
    secs = [
        _sec(0.6, 0.0, num_modes),
        _sec(0.6, 2.0, num_modes),
        _sec(0.6, 0.0, num_modes),
    ]
    r = run_eme(secs)
    assert r.transmission == pytest.approx(1.0, abs=1e-7)
    assert r.reflection < 1e-12
    assert r.energy_balance() == pytest.approx(1.0, abs=1e-9)


def test_straight_guide_multisection_still_unity():
    # several identical interior slabs — many interfaces, must stay unity
    secs = [_sec(0.6, 0.0)] + [_sec(0.6, 0.5) for _ in range(5)] + [_sec(0.6, 0.0)]
    r = run_eme(secs)
    assert r.transmission == pytest.approx(1.0, abs=1e-6)
    assert r.reflection < 1e-10


# --- taper: transmission, energy conservation, adiabatic convergence -------


def test_taper_is_highly_transmitting_and_lossless_basis():
    r = run_eme(_taper_sections(8))
    assert r.transmission > 0.99
    assert r.reflection < 1e-3
    # guided basis is unitary by construction -> exactly lossless
    assert r.energy_balance() == pytest.approx(1.0, abs=1e-9)
    assert isinstance(r, EMEResult)


def test_taper_converges_toward_adiabatic_limit():
    coarse = run_eme(_taper_sections(2))
    fine = run_eme(_taper_sections(16))
    # refining the staircase -> higher transmission, lower reflection (adiabatic)
    assert fine.transmission > coarse.transmission
    assert fine.reflection < coarse.reflection
    assert fine.transmission > 0.999


def test_taper_energy_conserved_every_refinement():
    for n_slabs in (1, 3, 9):
        r = run_eme(_taper_sections(n_slabs))
        assert r.energy_balance() == pytest.approx(1.0, abs=1e-9)


def test_multimode_taper_shows_conversion_but_conserves_energy():
    # a short, strong taper drives real TE0->TE1 conversion; with a guided basis
    # the fundamental loses power to the higher mode while total power is kept.
    secs = _taper_sections(6, w_in=0.5, w_out=1.2, taper_len=1.0, num_modes=3)
    r = run_eme(secs)
    if r.n_modes < 2:
        pytest.skip("resolution did not yield a multimode basis")
    assert r.transmission < r.transmitted_power()  # power leaks out of mode 0
    assert r.transmitted_power() <= 1.0 + 1e-9
    assert r.energy_balance() == pytest.approx(1.0, abs=1e-9)


# --- run_eme bookkeeping & guards ------------------------------------------


def test_run_eme_truncates_to_common_basis():
    # narrow guide supports fewer modes than the wide one; the cascade uses the
    # common minimum and never errors on the rectangular mismatch.
    a = _sec(0.45, 0.0, num_modes=4)
    b = _sec(0.95, 1.0, num_modes=4)
    c = _sec(0.45, 0.0, num_modes=4)
    n_common = min(len(a.modes), len(b.modes), len(c.modes))
    r = run_eme([a, b, c])
    assert r.n_modes == n_common
    assert r.s21.shape == (n_common, n_common)


def test_run_eme_respects_n_modes_cap():
    secs = [_sec(0.6, 0.0, 3), _sec(0.6, 1.0, 3), _sec(0.6, 0.0, 3)]
    r = run_eme(secs, n_modes=1)
    assert r.n_modes == 1
    assert r.transmission == pytest.approx(1.0, abs=1e-7)


def test_run_eme_requires_two_sections():
    with pytest.raises(ValueError):
        run_eme([_sec(0.6, 1.0)])


def test_run_eme_rejects_grid_mismatch():
    a = _sec(0.6, 0.0)
    # a section on a different transverse grid (different window) must be rejected
    bad = waveguide_section(
        wavelength_um=WL_UM,
        dl_um=DL_UM,
        core_w_um=0.6,
        core_h_um=CORE_H_UM,
        n_core=N_CORE,
        n_clad=N_CLAD,
        window_w_um=WIN_W_UM + 0.5,
        window_h_um=WIN_H_UM,
        num_modes=2,
        length_um=1.0,
        neff_margin=MARGIN,
    )
    with pytest.raises(ValueError):
        run_eme([a, bad])


# --- band sweep (rectangular_base_section + run_eme_band) -------------------


def _base_taper(n_slabs, *, w_in=0.45, w_out=0.80, taper_len=3.0):
    """Same staircased taper as ``_taper_sections`` but as (solver, length)
    base sections for :func:`run_eme_band` (rasterized once, no solve yet)."""
    common = dict(dl_um=DL_UM, core_h_um=CORE_H_UM, n_core=N_CORE,
                  n_clad=N_CLAD, window_w_um=WIN_W_UM, window_h_um=WIN_H_UM)
    secs = [rectangular_base_section(core_w_um=w_in, length_um=0.0, **common)]
    for k in range(n_slabs):
        wk = w_in + (w_out - w_in) * ((k + 0.5) / n_slabs)
        secs.append(rectangular_base_section(
            core_w_um=wk, length_um=taper_len / n_slabs, **common))
    secs.append(rectangular_base_section(core_w_um=w_out, length_um=0.0, **common))
    return secs


def _max_smatrix_diff(a, b):
    return max(
        float(np.max(np.abs(x - y)))
        for x, y in ((a.s11, b.s11), (a.s12, b.s12),
                     (a.s21, b.s21), (a.s22, b.s22))
    )


@pytest.mark.parametrize("backend", ["serial", "thread"])
def test_run_eme_band_matches_per_wavelength(backend):
    # run_eme_band must reproduce the per-wavelength waveguide_section + run_eme
    # cascade to machine precision (the only difference is at_wavelength reuse vs
    # a fresh raster — both give the same wavelength-independent permittivity).
    lams = [1.27, 1.31, 1.35]
    common = dict(dl_um=DL_UM, core_h_um=CORE_H_UM, n_core=N_CORE,
                  n_clad=N_CLAD, window_w_um=WIN_W_UM, window_h_um=WIN_H_UM)

    def ref(wl):
        secs = [waveguide_section(core_w_um=w, length_um=L, num_modes=2,
                                  wavelength_um=wl, neff_margin=MARGIN, **common)
                for w, L in ((0.45, 0.0), (0.60, 1.5), (0.80, 0.0))]
        return run_eme(secs)

    bases = [rectangular_base_section(core_w_um=w, length_um=L, **common)
             for w, L in ((0.45, 0.0), (0.60, 1.5), (0.80, 0.0))]
    band = run_eme_band(bases, lams, num_modes=2, neff_margin=MARGIN,
                        n_clad=N_CLAD, backend=backend, workers=2)

    assert set(band) == set(lams)
    for wl in lams:
        assert _max_smatrix_diff(ref(wl), band[wl]) < 1e-9


def test_run_eme_band_is_lossless_and_per_wavelength_keyed():
    bases = _base_taper(4)
    lams = [1.28, 1.34]
    band = run_eme_band(bases, lams, num_modes=2, neff_margin=MARGIN,
                        n_clad=N_CLAD, backend="serial")
    for wl in lams:
        r = band[wl]
        assert isinstance(r, EMEResult)
        # guided-basis cascade conserves energy and transmits strongly
        assert r.energy_balance() == pytest.approx(1.0, abs=1e-6)
        assert r.transmission > 0.95


def test_run_eme_band_guards():
    bases = _base_taper(2)
    # needs >= 2 sections
    with pytest.raises(ValueError):
        run_eme_band(bases[:1], [1.31], num_modes=2)
    # wavelengths must be non-empty
    with pytest.raises(ValueError):
        run_eme_band(bases, [], num_modes=2)
    # neff_margin > 0 requires n_clad
    with pytest.raises(ValueError):
        run_eme_band(bases, [1.31], num_modes=2, neff_margin=0.05)
    # unknown backend
    with pytest.raises(ValueError):
        run_eme_band(bases, [1.31], num_modes=2, backend="cuda")
