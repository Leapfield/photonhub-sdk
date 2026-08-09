"""Mode ⇄ mode overlap (coupling efficiency) — physics pins for
``photonhub.plugins.mode_overlap.mode_overlap`` and the analytic
``gaussian_mode`` builder.

Unlike ``test_mode_overlap.py`` (which projects a recorded FDTD field *plane*
onto one mode), these exercise the **mode-to-mode** overlap: two FDE modes, or an
FDE mode vs an analytic Gaussian. The Gaussian path is checked against CLOSED
FORM (offset and waist-mismatch overlap integrals), which validates the whole
chain — ``gaussian_mode`` rasterization, the common-grid resample, and the
overlap quadrature — to ~1e-3. The waveguide path pins self-overlap == 1,
power-orthogonal modes == 0, polarization selectivity, symmetry, and the bounded
power-coupling vs field-overlap distinction.
"""

import warnings

import numpy as np
import pytest

from photonhub.plugins import (
    ModeSolver,
    VectorModeSolver,
    gaussian_mode,
    mode_overlap,
    mode_overlap_matrix,
)
from photonhub.plugins.mode_overlap import ModeOverlap

WL_UM = 1.31
DL_UM = 0.025
N_SI, N_SIO2 = 3.5, 1.444


# --------------------------------------------------------------------------- #
# Fixtures: a small SOI strip's TE0 (scalar + full-vector), reused across tests.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def te0() -> "object":
    return ModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM, core_w_um=0.45, core_h_um=0.22,
        n_core=N_SI, n_clad=N_SIO2).solve(num_modes=1, polarization="TE")[0]


@pytest.fixture(scope="module")
def te0_wide() -> "object":
    return ModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM, core_w_um=0.80, core_h_um=0.22,
        n_core=N_SI, n_clad=N_SIO2).solve(num_modes=1, polarization="TE")[0]


@pytest.fixture(scope="module")
def vmodes() -> list:
    """TE0/TE1/TM0 of a wider multimode guide (full-vector, shared window so they
    sit on a common grid — power-orthogonal eigenmodes of one operator)."""
    return list(VectorModeSolver.from_rectangular_core(
        wavelength_um=1.55, dl_um=0.03, core_w_um=1.2, core_h_um=0.22,
        n_core=3.48, n_clad=1.444, window_w_um=2.4, window_h_um=1.4
    ).solve(num_modes=4))


# --------------------------------------------------------------------------- #
# Self-overlap is exactly 1 (the normalization pin) for every mode kind.
# --------------------------------------------------------------------------- #
def test_self_overlap_scalar_is_one(te0):
    r = mode_overlap(te0, te0)
    assert isinstance(r, ModeOverlap)
    assert r.power == pytest.approx(1.0, abs=1e-9)
    assert r.field == pytest.approx(1.0, abs=1e-9)
    assert abs(r.amplitude) == pytest.approx(1.0, abs=1e-9)
    assert r.mismatch_db == pytest.approx(0.0, abs=1e-6)
    assert float(r) == pytest.approx(r.power)  # __float__ → power


def test_self_overlap_vector_is_one(vmodes):
    for m in vmodes:
        r = mode_overlap(m, m)
        assert r.power == pytest.approx(1.0, abs=1e-6)
        assert r.field == pytest.approx(1.0, abs=1e-9)


def test_self_overlap_gaussian_is_one():
    g = gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM, mfd_um=1.0)
    r = mode_overlap(g, g)
    assert r.power == pytest.approx(1.0, abs=1e-9)
    assert r.field == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Power-orthogonal eigenmodes of the same guide → ~0; self → 1 (overlap matrix).
# --------------------------------------------------------------------------- #
def test_orthogonal_modes_decouple(vmodes):
    n = len(vmodes)
    for i in range(n):
        for j in range(n):
            r = mode_overlap(vmodes[i], vmodes[j])
            if i == j:
                assert r.power == pytest.approx(1.0, abs=1e-6)
            else:
                # different eigenmodes (incl. TE vs TM) carry ~no mutual power.
                assert r.power < 5e-3
                assert r.field < 5e-3


def test_cross_polarization_is_zero(te0):
    """An x-polarized waveguide TE0 and a y-polarized (TM) Gaussian share no
    transverse field — overlap is identically zero."""
    g_tm = gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM, mfd_um=1.0,
                         polarization="TM")
    r = mode_overlap(te0, g_tm)
    assert r.power == pytest.approx(0.0, abs=1e-12)
    assert r.field == pytest.approx(0.0, abs=1e-12)
    assert r.mismatch_db == float("inf")


# --------------------------------------------------------------------------- #
# Symmetry, boundedness, and the scalar power==field identity.
# --------------------------------------------------------------------------- #
def test_overlap_is_symmetric(te0, te0_wide):
    a = mode_overlap(te0, te0_wide)
    b = mode_overlap(te0_wide, te0)
    assert a.power == pytest.approx(b.power, rel=1e-9)
    assert a.field == pytest.approx(b.field, rel=1e-9)


def test_scalar_power_equals_field(te0):
    """For scalar modes the scalar-H impedance factors cancel, so the power
    coupling reduces exactly to the field overlap — independent of the index
    mismatch (here n_eff≈2.7 guide vs n=1.0 air Gaussian) and argument order."""
    g = gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM, mfd_um=0.7, n=1.0)
    r1 = mode_overlap(te0, g)
    r2 = mode_overlap(g, te0)
    assert r1.power == pytest.approx(r1.field, abs=1e-10)
    assert r1.power == pytest.approx(r2.power, abs=1e-10)


def test_power_bounded_over_gaussian_sweep(te0):
    """Power coupling stays in [0, 1] (to discretization precision) across a wide
    MFD / background-index / argument-order sweep — the headline efficiency does
    not blow past 1 the way the bare symmetrized overlap does."""
    worst = 0.0
    for n in (1.0, 1.444):
        for mfd in np.linspace(0.3, 2.0, 18):
            g = gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM,
                              mfd_um=float(mfd), n=n)
            worst = max(worst, mode_overlap(te0, g).power,
                        mode_overlap(g, te0).power)
    assert worst <= 1.0 + 1e-6


def test_vector_power_bounded(vmodes):
    """Dissimilar full-vector pairs may overshoot 1 only by the small documented
    discretization margin (≲1e-3), never grossly."""
    for i in range(len(vmodes)):
        for j in range(len(vmodes)):
            assert mode_overlap(vmodes[i], vmodes[j]).power <= 1.0 + 2e-3


def test_method_is_snyder_love_for_vector_pairs(vmodes):
    """Two full-vector modes use the exact Snyder & Love / Tidy3D power coupling."""
    r = mode_overlap(vmodes[0], vmodes[1])
    assert r.method == "snyder_love"
    assert mode_overlap(vmodes[0], vmodes[0]).method == "snyder_love"


def test_method_is_geomean_when_a_scalar_is_involved(te0, vmodes):
    """Any scalar/Gaussian operand falls back to the bounded geometric mean (the
    scalar-H Snyder–Love form would over-count by the inverse-Fresnel factor)."""
    g = gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM, mfd_um=1.0)
    assert mode_overlap(te0, g).method == "geomean"          # scalar + Gaussian
    assert mode_overlap(g, g).method == "geomean"            # Gaussian + Gaussian
    assert mode_overlap(vmodes[0], g).method == "geomean"    # vector + Gaussian (mixed)


def test_vector_power_is_the_snyder_love_value():
    """For a full-vector pair, .power == |¼(conj(A)+B)|²/(P_a P_b) computed by hand
    from the modes' true (E, H) — i.e. the rigorous two-term coupling, NOT the
    geometric mean (the two differ here, proving the accurate form is used).

    Uses the TE0 of two DIFFERENT-width guides (dissimilar but non-orthogonal), so
    the Snyder–Love and geomean forms genuinely diverge."""
    common = dict(wavelength_um=1.55, dl_um=0.03, core_h_um=0.22, n_core=3.48,
                  n_clad=1.444, window_w_um=2.4, window_h_um=1.4)
    a = VectorModeSolver.from_rectangular_core(core_w_um=0.45, **common).solve(
        num_modes=1)[0]
    b = VectorModeSolver.from_rectangular_core(core_w_um=0.80, **common).solve(
        num_modes=1)[0]
    from photonhub.plugins.mode_overlap import (
        _mode_plane_fields, _union_grid, _cell_widths)
    c1, c2 = _union_grid(a, b, (0.0, 0.0), (0.0, 0.0))
    ma = _mode_plane_fields(a, c1, c2, axis="z", center_um=(0.0, 0.0),
                            thickness_axis=None)
    mb = _mode_plane_fields(b, c1, c2, axis="z", center_um=(0.0, 0.0),
                            thickness_axis=None)
    dA = np.outer(_cell_widths(c2), _cell_widths(c1))
    A = np.sum((ma["e1"] * np.conj(mb["h2"]) - ma["e2"] * np.conj(mb["h1"])) * dA)
    B = np.sum((mb["e1"] * np.conj(ma["h2"]) - mb["e2"] * np.conj(ma["h1"])) * dA)
    pa = 0.5 * np.real(np.sum((ma["e1"] * np.conj(ma["h2"])
                               - ma["e2"] * np.conj(ma["h1"])) * dA))
    pb = 0.5 * np.real(np.sum((mb["e1"] * np.conj(mb["h2"])
                               - mb["e2"] * np.conj(mb["h1"])) * dA))
    form_b = float(np.abs(0.25 * (np.conj(A) + B)) ** 2 / (pa * pb))
    geomean = float(np.abs(A) * np.abs(B) / (4.0 * pa * pb))
    r = mode_overlap(a, b)
    assert r.power == pytest.approx(form_b, abs=1e-9)
    assert abs(form_b - geomean) > 1e-4    # the two forms genuinely differ here


def test_union_grid_covers_graded_mode_true_span():
    """F12 regression: a GRADED-window VectorMode's array pitch is NOT its scalar
    dl_x_um — it carries the true node ladder in x_coords_um/y_coords_um, and
    coordinate-aware consumers (vector_modal_fields) prefer it. _union_grid must
    too: reconstructing a uniform window of extent (nx-1)*dl_base is far narrower
    than the mode's real span, so the resample zero-fills (clips) the mode tails
    out of the overlap. Here the ladder spans ±0.8/±0.6 um while a dl_base=0.1
    reconstruction would give only ±0.2 um."""
    from photonhub.plugins.mode_overlap import _union_grid
    from photonhub.plugins.vector_modes import VectorMode

    nx = ny = 5
    z = np.zeros((ny, nx), dtype=np.complex128)
    xc = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])   # true (graded) node ladders,
    yc = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])   # much wider than (nx-1)*dl_base
    m = VectorMode(n_eff=2.0, n_group=None, ex=z + 1, ey=z, ez=z,
                   hx=z, hy=z + 1, hz=z, wavelength_um=1.55,
                   dl_x_um=0.1, dl_y_um=0.1, x_coords_um=xc, y_coords_um=yc)
    gx, gy = _union_grid(m, m, (0.0, 0.0), (0.0, 0.0))
    # The union grid must reach the true ladder span, not the ±0.2 um a dl_base
    # reconstruction (the pre-fix bug) would produce.
    assert gx[0] == pytest.approx(-0.8) and gx[-1] == pytest.approx(0.8)
    assert gy[0] == pytest.approx(-0.6) and gy[-1] == pytest.approx(0.6)


def test_vector_power_differs_from_field(vmodes):
    """For full-vector modes the impedance-aware power coupling is genuinely
    distinct from the impedance-blind field overlap (here for two nearby TE
    modes the two readings differ by a few %)."""
    # vmodes[0]=TE0, vmodes[1]=TE1; use TE0 vs a re-solved nearby guide instead so
    # they are not orthogonal — compare TE0 against itself on a coarser grid mix.
    a = VectorModeSolver.from_rectangular_core(
        wavelength_um=1.55, dl_um=0.03, core_w_um=0.45, core_h_um=0.22,
        n_core=3.48, n_clad=1.444, window_w_um=2.4, window_h_um=1.4
    ).solve(num_modes=1)[0]
    b = VectorModeSolver.from_rectangular_core(
        wavelength_um=1.55, dl_um=0.03, core_w_um=0.55, core_h_um=0.22,
        n_core=3.48, n_clad=1.444, window_w_um=2.4, window_h_um=1.4
    ).solve(num_modes=1)[0]
    r = mode_overlap(a, b)
    assert abs(r.power - r.field) > 1e-3


# --------------------------------------------------------------------------- #
# Gaussian builder vs CLOSED FORM — validates the whole overlap pipeline.
# --------------------------------------------------------------------------- #
def test_gaussian_mfd_waist_equivalence():
    """MFD is the 1/e² intensity diameter, w0 = MFD/2 — so mfd_um=1.0 and
    waist_um=0.5 build the identical beam (overlap exactly 1)."""
    g_mfd = gaussian_mode(wavelength_um=WL_UM, dl_um=0.02, mfd_um=1.0)
    g_w = gaussian_mode(wavelength_um=WL_UM, dl_um=0.02, waist_um=0.5)
    assert g_mfd.shape == g_w.shape
    assert np.allclose(g_mfd.field, g_w.field)
    assert mode_overlap(g_mfd, g_w).field == pytest.approx(1.0, abs=1e-12)


def test_gaussian_waist_mismatch_matches_closed_form():
    """Two concentric Gaussians of field-radii w1, w2 have the analytic field
    overlap |F|² = (2 w1 w2 / (w1²+w2²))². Pins gaussian_mode + the quadrature."""
    w1, w2 = 0.5, 1.0
    g1 = gaussian_mode(wavelength_um=WL_UM, dl_um=0.02, waist_um=w1,
                       window_um=8.0)
    g2 = gaussian_mode(wavelength_um=WL_UM, dl_um=0.02, waist_um=w2,
                       window_um=8.0)
    expect = (2.0 * w1 * w2 / (w1 ** 2 + w2 ** 2)) ** 2
    r = mode_overlap(g1, g2)
    assert r.field == pytest.approx(expect, abs=2e-3)
    # same medium ⇒ power == field == the closed form.
    assert r.power == pytest.approx(expect, abs=2e-3)


@pytest.mark.parametrize("d_um", [0.0, 0.2, 0.4, 0.6])
def test_gaussian_lateral_offset_matches_closed_form(d_um):
    """Two identical Gaussians (field radius w) offset by d couple as
    |F|² = exp(-d²/w²). Pins the center_b lateral-misalignment path."""
    w = 0.5
    g = gaussian_mode(wavelength_um=WL_UM, dl_um=0.02, waist_um=w, window_um=8.0)
    r = mode_overlap(g, g, center_b=(d_um, 0.0))
    assert r.field == pytest.approx(np.exp(-(d_um ** 2) / w ** 2), abs=2e-3)


def test_cubic_resample_beats_linear_on_different_grids():
    """When the two modes live on DIFFERENT grids, the default bicubic resample is
    far closer to the closed form than bilinear; on a SHARED grid both are exact
    (no resampling happens)."""
    w1, w2 = 0.5, 1.0
    exact = (2 * w1 * w2 / (w1 ** 2 + w2 ** 2)) ** 2
    g1 = gaussian_mode(wavelength_um=WL_UM, dl_um=0.04, waist_um=w1, window_um=8.0)
    g2 = gaussian_mode(wavelength_um=WL_UM, dl_um=0.02, waist_um=w2, window_um=8.0)
    err_cubic = abs(mode_overlap(g1, g2, interp="cubic").power - exact)
    err_linear = abs(mode_overlap(g1, g2, interp="linear").power - exact)
    assert err_cubic < 1e-5            # bicubic ~3e-7 here
    assert err_linear > 1e-4           # bilinear ~6e-4 here
    assert err_cubic < err_linear / 50
    # same grid → identical (resample is a no-op either way)
    s1 = gaussian_mode(wavelength_um=WL_UM, dl_um=0.02, waist_um=w1, window_um=8.0)
    s2 = gaussian_mode(wavelength_um=WL_UM, dl_um=0.02, waist_um=w2, window_um=8.0)
    assert mode_overlap(s1, s2, interp="cubic").power == pytest.approx(
        mode_overlap(s1, s2, interp="linear").power, abs=1e-12)
    assert mode_overlap(s1, s2).power == pytest.approx(exact, abs=1e-9)


def test_interp_validation(te0):
    with pytest.raises(ValueError):
        mode_overlap(te0, te0, interp="quadratic")


def test_offset_reduces_coupling(te0):
    """A lateral fibre offset monotonically lowers the waveguide coupling."""
    g = gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM, mfd_um=1.0, n=N_SIO2)
    p0 = mode_overlap(te0, g).power
    p1 = mode_overlap(te0, g, center_b=(0.3, 0.0)).power
    p2 = mode_overlap(te0, g, center_b=(0.6, 0.0)).power
    assert p0 > p1 > p2


def test_gaussian_elliptical_and_normalization():
    g = gaussian_mode(wavelength_um=WL_UM, dl_um=0.02, mfd_um=(1.2, 0.6))
    # L2-normalized profile, centered (peak at the middle cell).
    assert np.sum(g.field ** 2) == pytest.approx(1.0, abs=1e-9)
    ny, nx = g.field.shape
    assert g.field[ny // 2, nx // 2] == pytest.approx(g.field.max())
    # wider in x than y (MFDx > MFDy): the 1/e width along x exceeds that along y.
    cut_x = g.field[ny // 2, :]
    cut_y = g.field[:, nx // 2]
    half_x = np.sum(cut_x > cut_x.max() / np.e) * g.dl_x_um
    half_y = np.sum(cut_y > cut_y.max() / np.e) * g.dl_y_um
    assert half_x > half_y


# --------------------------------------------------------------------------- #
# Amplitude carries phase / sign; grid override; input validation.
# --------------------------------------------------------------------------- #
def test_amplitude_tracks_sign(te0):
    """Flipping a mode's sign flips the complex field amplitude (phase π) but not
    the (squared) power/field magnitudes."""
    import dataclasses
    flipped = dataclasses.replace(te0, field=-te0.field)
    r = mode_overlap(te0, flipped)
    assert r.amplitude.real == pytest.approx(-1.0, abs=1e-6)
    assert r.field == pytest.approx(1.0, abs=1e-9)
    assert r.power == pytest.approx(1.0, abs=1e-6)


def test_coupling_magnitude_equals_power(vmodes, te0):
    """The complex .coupling always satisfies |coupling|² == power (vector pair via
    snyder_love, scalar/Gaussian via geomean)."""
    g = gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM, mfd_um=0.8)
    for a, b in [(vmodes[0], vmodes[1]), (vmodes[0], vmodes[0]), (te0, g), (g, g)]:
        r = mode_overlap(a, b)
        assert abs(r.coupling) ** 2 == pytest.approx(r.power, abs=1e-9)
    # self-overlap is a clean unit coupling
    assert mode_overlap(vmodes[0], vmodes[0]).coupling == pytest.approx(1.0 + 0j, abs=1e-6)


def test_coupling_carries_propagation_phase(vmodes):
    """A global e^{iθ} phase on one mode (what propagation e^{-iβL} does) advances
    .coupling's phase by θ but leaves its magnitude — the property an S-matrix needs."""
    import dataclasses
    m = vmodes[0]
    for theta in (0.4, 1.9, -2.3):
        ph = np.exp(1j * theta)
        mp = dataclasses.replace(
            m, ex=m.ex * ph, ey=m.ey * ph, ez=m.ez * ph,
            hx=m.hx * ph, hy=m.hy * ph, hz=m.hz * ph)
        r = mode_overlap(m, mp)
        assert np.angle(r.coupling) == pytest.approx(theta, abs=1e-3)
        assert abs(r.coupling) ** 2 == pytest.approx(1.0, abs=1e-6)


def test_coupling_sign_for_flipped_scalar(te0):
    import dataclasses
    flipped = dataclasses.replace(te0, field=-te0.field)
    assert mode_overlap(te0, flipped).coupling == pytest.approx(-1.0 + 0j, abs=1e-6)


def test_explicit_grid_matches_auto(te0, te0_wide):
    x = (np.arange(81) - 40) * DL_UM
    y = (np.arange(61) - 30) * DL_UM
    r_grid = mode_overlap(te0, te0_wide, grid=(x, y))
    r_auto = mode_overlap(te0, te0_wide)
    assert r_grid.power == pytest.approx(r_auto.power, abs=5e-3)


# --------------------------------------------------------------------------- #
# Backward / contra-propagating direction (C).
# --------------------------------------------------------------------------- #
def test_backward_direction_is_orthogonal_to_self(vmodes):
    """A forward mode and the SAME mode's backward partner are power-orthogonal —
    overlap ~0 — while the co-propagating self-overlap is 1."""
    assert mode_overlap(vmodes[0], vmodes[0], direction="+").power == pytest.approx(
        1.0, abs=1e-6)
    assert mode_overlap(vmodes[0], vmodes[0], direction="-").power < 1e-6


def test_direction_default_unchanged(vmodes):
    """direction='+' (default) is the existing co-propagating coupling."""
    a = mode_overlap(vmodes[0], vmodes[1])
    b = mode_overlap(vmodes[0], vmodes[1], direction="+")
    assert a.power == pytest.approx(b.power, rel=1e-12)


def test_direction_validation(te0):
    with pytest.raises(ValueError):
        mode_overlap(te0, te0, direction="backward")


def test_backward_direction_scalar_falls_back_to_two_term(te0):
    """Regression: the geomean power form is direction-blind (|A||B| is
    unchanged by flipping mode_b's H), so a SCALAR mode used to read power == 1
    against its own backward partner while the docstring promised ~0. The
    contra-propagating scalar case now falls back (with a warning) to the
    two-term snyder_love form, which carries the forward/backward cancellation."""
    with pytest.warns(UserWarning, match="direction-blind"):
        r = mode_overlap(te0, te0, direction="-")
    assert r.method == "snyder_love"
    assert r.power < 1e-6
    # the co-propagating scalar path is untouched (bounded geomean, no warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = mode_overlap(te0, te0, direction="+")
    assert f.method == "geomean"
    assert f.power == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Coupling matrix / S-matrix block (B).
# --------------------------------------------------------------------------- #
def test_overlap_matrix_is_identity_for_orthonormal_basis(vmodes):
    """The complex coupling matrix of a mode set with itself is ~identity (|M|²),
    and the diagonal is unit-phase 1+0j."""
    M = mode_overlap_matrix(vmodes, vmodes)
    assert M.shape == (len(vmodes), len(vmodes))
    assert np.iscomplexobj(M)
    P = np.abs(M) ** 2
    assert np.allclose(np.diag(P), 1.0, atol=1e-5)
    off = P - np.diag(np.diag(P))
    assert np.max(off) < 5e-3
    assert M[0, 0] == pytest.approx(1.0 + 0j, abs=1e-6)


def test_overlap_matrix_quantities_and_shape(te0, te0_wide):
    a = [te0, te0_wide]
    g = gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM, mfd_um=1.0)
    b = [te0_wide, g]
    cpl = mode_overlap_matrix(a, b)                       # complex
    pwr = mode_overlap_matrix(a, b, quantity="power")     # real
    assert cpl.shape == (2, 2) and np.iscomplexobj(cpl)
    assert pwr.shape == (2, 2) and pwr.dtype == np.float64
    # |coupling|² == power entrywise
    assert np.allclose(np.abs(cpl) ** 2, pwr, atol=1e-9)
    # entries equal the scalar mode_overlap
    assert pwr[0, 0] == pytest.approx(mode_overlap(te0, te0_wide).power, abs=1e-12)
    with pytest.raises(ValueError):
        mode_overlap_matrix(a, b, quantity="bogus")


def test_overlap_matrix_forwards_kwargs(vmodes):
    """kwargs (e.g. direction) reach every pair."""
    M = mode_overlap_matrix(vmodes[:1], vmodes[:1], direction="-")
    assert np.abs(M[0, 0]) ** 2 < 1e-6                    # backward self ~ 0


def test_invalid_inputs(te0):
    with pytest.raises(ValueError):
        mode_overlap(te0, "not a mode")
    with pytest.raises(ValueError):
        mode_overlap(te0, te0, axis="x")  # non-z auto-grid needs explicit grid
    with pytest.raises(ValueError):
        gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM)  # neither mfd nor waist
    with pytest.raises(ValueError):
        gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM, mfd_um=1.0, waist_um=0.5)
    with pytest.raises(ValueError):
        gaussian_mode(wavelength_um=WL_UM, dl_um=DL_UM, mfd_um=-1.0)
