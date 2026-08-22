"""One-call waveguide analysis — geometry, materials, and dispersion pins.

The underlying full-vector FDE physics is gated in
``validation/test_tier2a_vector_modesolver.py``; these tests pin what the
convenience layer adds: cross-section building (strip delegation, rib
raster), materials-library resolution, the material-aware group index, and
the result accessors.
"""

import math

import numpy as np
import pytest

from photonhub import materials
from photonhub.plugins import rectangular_waveguide
from photonhub.plugins.waveguide import _rib_eps

WL = 1.55
W, H = 0.45, 0.22
N_SI, N_CLAD = 3.48, 1.444
DL = 0.02


def test_strip_matches_direct_solver_and_accessors():
    wg = rectangular_waveguide(
        wavelength_um=WL, core_w_um=W, core_h_um=H,
        core=N_SI, clad=N_CLAD, dl_um=DL, num_modes=2)
    # 450x220 SOI strip TE0 at 1550 nm ~ 2.36 (literature ~2.37); the FDE
    # gates pin the absolute accuracy, this pins the wiring
    te0 = wg.fundamental("TE")
    assert 2.2 < np.real(te0.n_eff) < 2.6
    assert te0.te_fraction > 0.7
    assert wg.n_eff[0] >= wg.n_eff[1]          # descending
    assert wg.n_group is None                  # not requested
    assert wg.loss_db_per_cm is None           # straight + lossless
    assert "n_eff" in wg.summary()
    assert wg.n_core == pytest.approx(N_SI)


def test_fundamental_polarization_selection_and_missing():
    wg = rectangular_waveguide(
        wavelength_um=WL, core_w_um=W, core_h_um=H,
        core=N_SI, clad=N_CLAD, dl_um=DL, num_modes=2)
    pols = set(wg.polarizations)
    if "TM" in pols:
        assert wg.fundamental("TM").polarization == "TM"
    with pytest.raises(ValueError, match="polarization must be TE or TM"):
        wg.fundamental("TEM")
    # a single-mode request that only returns TE has no TM to offer
    wg1 = rectangular_waveguide(
        wavelength_um=WL, core_w_um=W, core_h_um=H,
        core=N_SI, clad=N_CLAD, dl_um=DL, num_modes=1)
    if all(p == "TE" for p in wg1.polarizations):
        with pytest.raises(ValueError, match="no TM mode"):
            wg1.fundamental("TM")


def test_rib_raster_geometry():
    eps = _rib_eps(
        wavelength_um=WL, dl_um=0.05, core_w_um=0.6, core_h_um=0.3,
        slab_h_um=0.1, n_core=3.0, n_clad=1.5,
        window_w_um=2.0, window_h_um=1.2)
    ny, nx = eps.shape
    assert nx % 2 == 1 and ny % 2 == 1
    ec, ecl = 9.0, 2.25
    # cladding at the corners, core in the middle of the stack
    assert eps[0, 0] == pytest.approx(ecl)
    assert eps[-1, -1] == pytest.approx(ecl)
    assert eps.max() == pytest.approx(ec)
    # the slab extends across the FULL width at a height inside [0, slab_h]:
    # find a row whose min is core eps (fully slab-filled row)
    slab_rows = [iy for iy in range(ny) if eps[iy].min() >= ec - 1e-9]
    assert slab_rows, "no fully-slab row found"
    # and above the slab the wings are cladding while the core column stays
    core_rows = [iy for iy in range(ny)
                 if eps[iy, nx // 2] >= ec - 1e-9 and eps[iy, 0] <= ecl + 1e-9]
    assert core_rows, "no core-above-slab row found"


def test_rib_mode_sits_between_slab_and_strip():
    common = dict(wavelength_um=WL, core_w_um=W, core_h_um=H,
                  core=N_SI, clad=N_CLAD, dl_um=DL, num_modes=1)
    strip = rectangular_waveguide(**common)
    rib = rectangular_waveguide(slab_h_um=0.09, **common)
    n_strip = float(np.real(strip.fundamental("TE").n_eff))
    n_rib = float(np.real(rib.fundamental("TE").n_eff))
    # the slab raises the average cladding-side index -> rib TE0 above strip
    assert n_rib > n_strip
    assert n_rib < N_SI


def test_materials_by_name_and_entry():
    wg = rectangular_waveguide(
        wavelength_um=1.31, core_w_um=W, core_h_um=H,
        core="cSi", clad=materials.get("SiO2"), dl_um=DL, num_modes=1)
    n_si = math.sqrt(float(
        materials.get("cSi").medium(wavelength_um=1.31).permittivity))
    assert wg.n_core == pytest.approx(n_si, rel=1e-12)
    assert 2.0 < np.real(wg.fundamental("TE").n_eff) < n_si


def test_material_aware_group_index_exceeds_waveguide_only():
    kw = dict(wavelength_um=1.55, core_w_um=W, core_h_um=H,
              dl_um=DL, num_modes=1, group_index=True)
    # float indices -> built-in waveguide-dispersion-only n_g
    wg_float = rectangular_waveguide(core=N_SI, clad=N_CLAD, **kw)
    ng_wg = float(wg_float.n_group[0])
    assert 3.0 < ng_wg < 6.0
    # materials -> adds silicon's material dispersion (dn/dlambda < 0 raises
    # n_g at 1.55 by ~ +0.2-0.5 for SOI)
    wg_mat = rectangular_waveguide(core="cSi", clad="SiO2", **kw)
    ng_mat = float(wg_mat.n_group[0])
    assert np.isfinite(ng_mat)
    assert ng_mat > ng_wg - 0.05
    assert abs(ng_mat - ng_wg) < 2.0


def test_guards():
    kw = dict(wavelength_um=WL, core_w_um=W, core_h_um=H, dl_um=DL)
    with pytest.raises(ValueError, match="exceed"):
        rectangular_waveguide(core=1.4, clad=N_CLAD, **kw)
    with pytest.raises(ValueError, match="slab_h_um"):
        rectangular_waveguide(core=N_SI, clad=N_CLAD, slab_h_um=H, **kw)
    with pytest.raises(ValueError, match=">= 1"):
        rectangular_waveguide(core=0.5, clad=0.4, **kw)
    with pytest.raises(KeyError):
        rectangular_waveguide(core="unobtainium", clad=N_CLAD, **kw)
