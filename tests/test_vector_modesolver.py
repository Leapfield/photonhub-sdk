"""Full-vectorial FDE solver (VectorModeSolver / VectorMode) — physics + frozen
API surface.

Validates ``photonhub.plugins.VectorModeSolver`` on the canonical SOI strip: the
full-vector fundamental quasi-TE ``n_eff ~= 2.71`` (the design-doc FDTD cross-check
was 2.70; the semi-vec self-value 2.72 slightly over-confines), a quasi-TM mode
below it, the group index, and the six-component field surface. The load-bearing
<1% accuracy gate against the analytic slab lives in
``validation/test_tier2a_vector_modesolver.py``.
"""

import numpy as np
import pytest
import xarray as xr

from photonhub.plugins import VectorMode, VectorModeSolver

WL_UM = 1.31
DL_UM = 0.025
CORE_W_UM, CORE_H_UM = 0.45, 0.22
N_SI, N_SIO2 = 3.5, 1.444


@pytest.fixture(scope="module")
def strip() -> VectorModeSolver:
    return VectorModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM, core_w_um=CORE_W_UM,
        core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2)


@pytest.fixture(scope="module")
def strip_modes(strip):
    return strip.solve(num_modes=2)


# --- physics ---------------------------------------------------------------

def test_te0_neff_full_vector(strip_modes):
    te0 = strip_modes[0]
    # Full-vector fundamental ~ 2.66 with subpixel smoothing on (the default) —
    # the converged value, matching tidy3d's KFJ mode solver (~2.65); the
    # staircase / semi-vec value (~2.70-2.72) sits high.
    assert te0.n_eff == pytest.approx(2.66, abs=0.05)
    assert N_SIO2 < te0.n_eff < N_SI                    # genuinely guided
    assert isinstance(te0, VectorMode)
    assert te0.polarization == "TE" and te0.te_fraction > 0.5
    assert te0.core_fraction(CORE_W_UM, CORE_H_UM) > 0.6


def test_subpixel_flag_smooths_and_shifts_neff():
    """Subpixel smoothing (default on) raster-averages the high-contrast walls,
    shifting n_eff off the staircase value; subpixel=False reproduces the hard
    sample. The smoothed cross-section carries intermediate (non-binary) eps."""
    kw = dict(wavelength_um=WL_UM, dl_um=DL_UM, core_w_um=CORE_W_UM,
              core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2)
    soft = VectorModeSolver.from_rectangular_core(**kw, subpixel=True)
    hard = VectorModeSolver.from_rectangular_core(**kw, subpixel=False)
    # Hard sample is binary core/clad; subpixel introduces boundary cells strictly
    # between the two permittivities.
    eps_vals = set(np.round(hard.eps.ravel(), 6))
    assert eps_vals == {round(N_SIO2 ** 2, 6), round(N_SI ** 2, 6)}
    between = (soft.eps > N_SIO2 ** 2 + 1e-6) & (soft.eps < N_SI ** 2 - 1e-6)
    assert between.any(), "subpixel produced no intermediate-eps boundary cells"
    # The two rasterizations give measurably different (both guided) n_eff.
    n_soft = soft.solve(num_modes=1)[0].n_eff
    n_hard = hard.solve(num_modes=1)[0].n_eff
    assert N_SIO2 < n_soft < N_SI and N_SIO2 < n_hard < N_SI
    assert abs(n_soft - n_hard) > 1e-3


def test_subpixel_method_tensor_default_and_options():
    """``from_rectangular_core`` defaults to the KFJ subpixel **tensor**;
    ``subpixel_method="volume"`` and ``subpixel=False`` select the scalar rasters.
    The tensor solver carries a genuine anisotropic boundary tensor (εxx, εyy
    differ from εzz where an interface is partially filled), and all three rasters
    return a guided fundamental."""
    kw = dict(wavelength_um=WL_UM, dl_um=DL_UM, core_w_um=CORE_W_UM,
              core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2)
    tensor = VectorModeSolver.from_rectangular_core(**kw)                # default
    volume = VectorModeSolver.from_rectangular_core(**kw,
                                                    subpixel_method="volume")
    stair = VectorModeSolver.from_rectangular_core(**kw, subpixel=False)
    assert tensor._is_tensor
    assert not volume._is_tensor and not stair._is_tensor
    # Harmonic normal vs arithmetic tangential -> εyy and εxx both diverge from
    # εzz somewhere on the partially-filled wall cells (the anisotropy KFJ adds).
    assert np.any(np.abs(tensor._eyy - tensor._ezz) > 1e-6)
    assert np.any(np.abs(tensor._exx - tensor._ezz) > 1e-6)
    # The scalar rasters are isotropic (all three components equal).
    assert np.array_equal(volume._exx, volume._ezz)
    for solver in (tensor, volume, stair):
        n_eff = solver.solve(num_modes=1)[0].n_eff
        assert N_SIO2 < n_eff < N_SI


def test_tm0_below_te0_and_classified(strip_modes):
    te0, tm0 = strip_modes
    assert tm0.n_eff < te0.n_eff                        # TM less confined
    assert N_SIO2 < tm0.n_eff < N_SI
    assert tm0.polarization == "TM" and tm0.te_fraction < 0.5


def test_modes_ordered_descending(strip_modes):
    neffs = [m.n_eff for m in strip_modes]
    assert neffs == sorted(neffs, reverse=True)


def test_group_index(strip):
    m = strip.solve(num_modes=1, group_index=True)[0]
    assert m.n_group is not None and np.isfinite(m.n_group)
    assert m.n_group > m.n_eff                          # normal waveguide dispersion
    # Without the flag, n_group is None (group index is opt-in).
    assert strip.solve(num_modes=1)[0].n_group is None


# --- six-component field surface -------------------------------------------

def test_six_components_present_and_complex(strip_modes):
    te0 = strip_modes[0]
    for comp in (te0.ex, te0.ey, te0.ez, te0.hx, te0.hy, te0.hz):
        assert comp.shape == te0.shape
        assert np.iscomplexobj(comp)
    # The transverse-E pair is jointly L2-normalized.
    power = float(np.sum(np.abs(te0.ex) ** 2 + np.abs(te0.ey) ** 2))
    assert power == pytest.approx(1.0, abs=1e-6)


def test_field_dataarray(strip_modes):
    te0 = strip_modes[0]
    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        da = te0.field_dataarray(comp)
        assert isinstance(da, xr.DataArray)
        assert da.dims == ("y", "x") and da.name == comp
        assert da.shape == te0.shape
    da = te0.field_dataarray("Ex")
    # Real-space coords in microns, centered on the origin.
    assert float(da.coords["x"].values.mean()) == pytest.approx(0.0, abs=1e-9)
    assert float(da.coords["y"].values.mean()) == pytest.approx(0.0, abs=1e-9)
    assert {"n_eff", "wavelength_um", "polarization", "te_fraction"} <= set(da.attrs)
    with pytest.raises(ValueError):
        te0.field_dataarray("Qz")


def test_reported_coords_honour_center_offset_and_graded_ladders():
    """Regression: field_dataarray / core_fraction rebuilt a uniform ladder from
    dl_x_um and IGNORED both center_offset_um (the window-placement shift a
    grid-snapped cross-section solve applies) and the graded x_coords_um/y_coords_um
    ladders — so a dispatcher-solved mode reported coordinates displaced from where
    mode_overlap.vector_modal_fields actually places it, and core_fraction measured
    a box offset from the real core. Both must use the same reconstruction."""
    n = 9
    z = np.zeros((n, n), dtype=np.complex128)
    e = np.ones((n, n), dtype=np.complex128)

    def mk(**kw):
        return VectorMode(n_eff=2.0, n_group=None, ex=e, ey=z, ez=z,
                          hx=z, hy=e, hz=z, wavelength_um=1.55,
                          dl_x_um=0.1, dl_y_um=0.1, **kw)

    # (a) center_offset_um shifts the reported ladder by exactly the offset.
    base = mk()
    off = mk(center_offset_um=(0.25, -0.15))
    x0 = base.field_dataarray("Ex").coords["x"].values
    y0 = base.field_dataarray("Ex").coords["y"].values
    x1 = off.field_dataarray("Ex").coords["x"].values
    y1 = off.field_dataarray("Ex").coords["y"].values
    assert np.allclose(x1 - x0, 0.25)
    assert np.allclose(y1 - y0, -0.15)

    # (b) a GRADED mode reports its true node ladder, not the dl_x_um pitch.
    xc = np.linspace(-0.8, 0.8, n)      # pitch 0.2 != dl_x_um 0.1
    yc = np.linspace(-0.6, 0.6, n)
    grad = mk(x_coords_um=xc, y_coords_um=yc)
    assert np.allclose(grad.field_dataarray("Ex").coords["x"].values, xc)
    assert np.allclose(grad.field_dataarray("Ex").coords["y"].values, yc)

    # (c) core_fraction's box follows the same coordinates: with the mode shifted
    # +0.25 um in x, a narrow box on the requested centre captures strictly less
    # energy than it does for the unshifted mode (uniform |E| here).
    assert off.core_fraction(0.4, 10.0) < base.core_fraction(0.4, 10.0)


def test_vectormode_is_frozen(strip_modes):
    with pytest.raises(Exception):
        strip_modes[0].n_eff = 1.0  # type: ignore[misc]


# --- input validation ------------------------------------------------------

def test_constructor_validation():
    good = np.full((8, 8), N_SI ** 2)
    with pytest.raises(ValueError):
        VectorModeSolver(np.ones(10), DL_UM, DL_UM, WL_UM)            # not 2-D
    with pytest.raises(ValueError):
        VectorModeSolver(np.full((4, 4), 0.5), DL_UM, DL_UM, WL_UM)   # eps < 1
    bad = good.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        VectorModeSolver(bad, DL_UM, DL_UM, WL_UM)                    # non-finite
    with pytest.raises(ValueError):
        VectorModeSolver(good, 0.0, DL_UM, WL_UM)                     # dl <= 0
    with pytest.raises(ValueError):
        VectorModeSolver(good, DL_UM, DL_UM, 0.0)                     # wl <= 0
    with pytest.raises(ValueError):
        VectorModeSolver(good, DL_UM, DL_UM, WL_UM, x_symmetry="bogus")


def test_solve_rejects_bad_args(strip):
    with pytest.raises(ValueError):
        strip.solve(num_modes=0)


def test_oversized_grid_is_capped():
    n = int(np.sqrt(VectorModeSolver.MAX_UNKNOWNS)) + 20
    eps = np.full((n, n), N_SIO2 ** 2)
    ms = VectorModeSolver(eps, DL_UM, DL_UM, WL_UM)
    with pytest.raises(ValueError):
        ms.solve()


# --- §20 half-domain symmetry plane ----------------------------------------

def _full_half_te0(dl):
    kw = dict(wavelength_um=WL_UM, dl_um=dl, core_w_um=CORE_W_UM,
              core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2,
              window_w_um=2.0, window_h_um=1.5, subpixel=False)
    full = VectorModeSolver.from_rectangular_core(**kw)
    half = VectorModeSolver.from_rectangular_core(**kw, x_min_symmetry="pec")
    return full, half, full.solve(num_modes=1, n_guess=3.0)[0], \
        half.solve(num_modes=1, n_guess=3.0)[0]


def test_half_domain_te0_is_guided_te():
    # The width-center is an odd / electric (PEC) symmetry plane for the
    # width-even TE0; the half solve must still return a guided TE mode.
    _, _, _, te0_half = _full_half_te0(0.02)
    assert te0_half.polarization == "TE"
    assert N_SIO2 < te0_half.n_eff < N_SI
    assert te0_half.te_fraction > 0.9


def test_half_domain_halves_the_unknowns():
    full, half, _, _ = _full_half_te0(0.04)
    assert half.eps.shape[0] == full.eps.shape[0]             # height unchanged
    assert half.eps.shape[1] == (full.eps.shape[1] + 1) // 2  # width halved
    assert half.eps.size < 0.55 * full.eps.size


def test_half_domain_converges_to_full_mode():
    # The PEC fold at the sliced centre reconstructs the full EVEN mode; the
    # finite-dl gap is the stencil's first-order wall error, so it must SHRINK
    # as dl halves (the repo's "parity = convergence" bar). Not bit-exact.
    _, _, f4, h4 = _full_half_te0(0.04)
    _, _, f2, h2 = _full_half_te0(0.02)
    gap4 = abs(f4.n_eff - h4.n_eff)
    gap2 = abs(f2.n_eff - h2.n_eff)
    assert gap2 < gap4                  # converging toward the full mode
    assert gap2 < 0.04                  # within ~1.5% at dl = 0.02


def test_half_domain_pmc_selects_the_complementary_mode():
    # A WIDE guide supports the width-EVEN TE0 and the width-ODD TE1. The odd /
    # electric (PEC) plane selects the even mode; the even / magnetic (PMC) plane
    # selects the odd one — the complementary parity. Both are guided, and each
    # half solve matches the corresponding full-domain mode (first-order wall).
    kw = dict(wavelength_um=WL_UM, dl_um=0.02, core_w_um=1.2,
              core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2,
              window_w_um=3.0, window_h_um=1.5, subpixel=False)
    full = VectorModeSolver.from_rectangular_core(**kw).solve(
        num_modes=2, n_guess=3.0)
    pec = VectorModeSolver.from_rectangular_core(
        **kw, x_min_symmetry="pec").solve(num_modes=1, n_guess=3.0)[0]
    pmc = VectorModeSolver.from_rectangular_core(
        **kw, x_min_symmetry="pmc").solve(num_modes=1, n_guess=3.0)[0]
    assert N_SIO2 < pmc.n_eff < N_SI                     # PMC mode is guided
    assert abs(pec.n_eff - full[0].n_eff) < 0.04         # PEC ≈ even fundamental
    assert abs(pmc.n_eff - full[1].n_eff) < 0.04         # PMC ≈ odd first mode
    assert abs(pec.n_eff - pmc.n_eff) > 0.05             # genuinely complementary


def test_half_domain_rejects_bad_x_min_symmetry():
    with pytest.raises(ValueError, match="x_min_symmetry"):
        VectorModeSolver.from_rectangular_core(
            wavelength_um=WL_UM, dl_um=0.05, core_w_um=CORE_W_UM,
            core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2,
            x_min_symmetry="bogus")


# --- bent waveguide + leaky (complex n_eff) modes --------------------------
#
# A moderately-confined SiN-like strip (n=2.0 in oxide) on a 5x3 um window — the
# same cross-section the Tidy3D bent-mode benchmark uses (benchmarks/tidy3d/
# bent_modes), so the corrected cylindrical/centrifugal bend operator is exercised
# in the regime it was validated against. The window must be wide enough (and the
# radii not TOO tight) to hold the outward-pushed bend mode; tighter bends radiate
# (the loss sits above the PML floor). The straight (bend_radius=None) solve stays
# bit-for-bit the original; the bent solve uses the generalized eigenproblem
# A h = beta0^2 B h with the centrifugal weight B = diag((R/r)^2), plus a
# tangential PML so n_eff goes complex (the imaginary part is the bend loss).

BEND_KW = dict(wavelength_um=1.55, dl_um=0.035, core_w_um=0.7, core_h_um=0.4,
               n_core=2.0, n_clad=1.444, window_w_um=5.0, window_h_um=3.0)


@pytest.fixture(scope="module")
def bend_solver() -> VectorModeSolver:
    return VectorModeSolver.from_rectangular_core(**BEND_KW)


def test_straight_path_unchanged_lossless(bend_solver):
    """With ``bend_radius_um=None`` (the default) the mode is the straight,
    lossless one: real n_eff, exactly-zero loss, and the new bend attributes
    advertise 'not a bend'. This is the guarantee the existing straight tests
    rely on."""
    m = bend_solver.solve(num_modes=1)[0]
    assert m.bend_radius_um is None
    assert m.k_eff == 0.0
    assert m.loss_db_per_cm == 0.0
    assert m.n_eff_complex == complex(m.n_eff, 0.0)
    assert N_SIO2 < m.n_eff < 2.0


def test_bend_loss_increases_as_radius_decreases(bend_solver):
    """Bend-loss MONOTONICITY: a tighter bend radiates more, so Im(n_eff)
    (loss) grows as R shrinks — the headline physical check."""
    radii = [9.0, 12.0, 15.0]
    losses = [bend_solver.solve(num_modes=1, bend_radius_um=R)[0].loss_db_per_cm
              for R in radii]
    # All clearly lossy (well above the PML floor) and strictly ordered.
    assert all(L > 1.0 for L in losses), losses
    assert losses[0] > losses[1] > losses[2], losses


def test_bend_neff_is_complex_and_positive_loss(bend_solver):
    """A tight bend has a genuinely complex modal index: positive k_eff (loss,
    not gain) and a positive loss in dB/cm. The eigenvalue went complex via the
    PML, which the straight solve never does."""
    straight = bend_solver.solve(num_modes=1)[0].n_eff
    m = bend_solver.solve(num_modes=1, bend_radius_um=9.0)[0]
    assert m.bend_radius_um == 9.0
    assert m.k_eff > 0.0                                 # loss, not gain
    assert m.loss_db_per_cm > 0.0
    assert m.n_eff_complex.imag == m.k_eff
    # The bend mode sits in the guided band, pushed UP from the straight index by
    # the curvature (the centrifugal map). A lossy bend mode is a leaky resonance
    # whose field is shifted radially outward (it does NOT stay core-box-confined
    # like a straight mode — that delamination is the bend loss), so we check it is
    # the right *guided* eigenpair by its index, not by core-box overlap.
    assert N_SIO2 < m.n_eff < 2.0                        # genuinely guided band
    assert m.n_eff > straight + 1e-3                     # curvature-shifted upward


def test_bend_recovers_straight_limit(bend_solver):
    """As R -> infinity the bent mode recovers the straight mode: Re(n_eff)
    converges to the straight value and the loss falls to the (near-zero) PML
    noise floor — orders of magnitude below a real bend's loss (hundreds-plus
    dB/cm). At R=2000 um the centrifugal weight (R/r)^2 -> 1 so the eigenproblem
    is the straight one to ~1e-6."""
    straight = bend_solver.solve(num_modes=1)[0].n_eff
    wide = bend_solver.solve(num_modes=1, bend_radius_um=2000.0)[0]
    assert wide.n_eff == pytest.approx(straight, abs=2e-3)
    # Effectively lossless: at/below the PML floor, far under a real bend's loss.
    assert wide.loss_db_per_cm < 1.0


def test_bend_real_neff_trends_above_straight(bend_solver):
    """Re(n_eff(R)) trends as expected: the centrifugal (cylindrical) curvature
    term pushes the mode outward, so a tighter bend has a higher Re(n_eff),
    monotonically approaching the straight value from above as R grows."""
    straight = bend_solver.solve(num_modes=1)[0].n_eff
    n_tight = bend_solver.solve(num_modes=1, bend_radius_um=9.0)[0].n_eff
    n_loose = bend_solver.solve(num_modes=1, bend_radius_um=15.0)[0].n_eff
    assert n_tight > n_loose > straight - 1e-4
    assert n_tight - straight > 1e-3                     # a measurable shift


def test_bend_field_dataarray_carries_loss(bend_solver):
    """The bend metadata (radius, k_eff, loss) rides along on the field
    DataArray attrs so plotting/inspection sees it; the straight DataArray does
    not carry them."""
    m = bend_solver.solve(num_modes=1, bend_radius_um=9.0)[0]
    da = m.field_dataarray("Ex")
    assert da.attrs["bend_radius_um"] == 9.0
    assert da.attrs["k_eff"] == m.k_eff
    assert da.attrs["loss_db_per_cm"] == m.loss_db_per_cm
    straight_da = bend_solver.solve(num_modes=1)[0].field_dataarray("Ex")
    assert "bend_radius_um" not in straight_da.attrs


def test_bend_rejects_zero_radius(bend_solver):
    with pytest.raises(ValueError):
        bend_solver.solve(num_modes=1, bend_radius_um=0.0)
    with pytest.raises(ValueError):
        bend_solver.solve(num_modes=1, num_pml=-1)


# --- exports (additive; the frozen semi-vec surface is untouched) ----------

def test_exports_are_additive():
    import photonhub.plugins as plugins
    assert {"VectorMode", "VectorModeSolver", "Mode", "ModeSolver"} <= set(
        plugins.__all__)
    from photonhub.plugins import Mode, ModeSolver  # the frozen surface still works
    assert Mode is not None and ModeSolver is not None
