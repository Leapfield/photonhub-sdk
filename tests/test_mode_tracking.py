"""Mode tracking (``photonhub.plugins.mode_tracking``) — overlap matching and
crossing-aware labeling.

The load-bearing test is a controllable **n_eff crossing**: two weakly-coupled
cores whose widths are swept in opposite directions. Their localized modes swap
n_eff ordering at the crossing, so the solver's "mode 0" jumps from one core to
the other — but overlap tracking follows each physical mode continuously (each
stays localized on its own core, with a smooth monotonic n_eff curve). A taper
(no crossing) must track to the identity, confirming tracking does not disturb
the easy case.
"""

import numpy as np
import pytest

from photonhub.plugins import eme
from photonhub.plugins.mode_tracking import (
    match_modes,
    reorder_to_tracks,
    track_modes,
    transverse_overlap,
)
from photonhub.plugins.vector_modes import VectorModeSolver

WL_UM = 1.31
DL_UM = 0.05
N_CORE, N_CLAD = 3.5, 1.444
CORE_H_UM = 0.22

# --- two-core crossing geometry --------------------------------------------
WIN_W_UM, WIN_H_UM = 3.6, 1.2
X_A, X_B = -0.8, 0.8  # core centers (wide gap -> weak coupling -> clean crossing)


def _grid():
    nx = int(round(WIN_W_UM / DL_UM)) | 1  # odd
    ny = int(round(WIN_H_UM / DL_UM)) | 1
    xs = (np.arange(nx) - (nx - 1) / 2) * DL_UM
    ys = (np.arange(ny) - (ny - 1) / 2) * DL_UM
    return xs, ys


def _eps_two_core(w_a, w_b):
    xs, ys = _grid()
    eps = np.full((ys.size, xs.size), N_CLAD**2)
    for xc, w in ((X_A, w_a), (X_B, w_b)):
        mask = (np.abs(xs[None, :] - xc) <= w / 2) & (np.abs(ys[:, None]) <= CORE_H_UM / 2)
        eps[mask] = N_CORE**2
    return eps


def _left_fraction(mode):
    """Fraction of transverse-E energy on the left (core A) half."""
    xs, _ = _grid()
    p = np.abs(mode.ex) ** 2 + np.abs(mode.ey) ** 2
    return float(p[:, xs < 0].sum() / p.sum())


@pytest.fixture(scope="module")
def crossing_planes():
    # 8 planes; t straddles but never lands on the t=0.5 degeneracy
    n = 8
    planes = []
    for i in range(n):
        t = (i + 0.5) / n
        w_a, w_b = 0.5 - 0.2 * t, 0.3 + 0.2 * t
        planes.append(VectorModeSolver(_eps_two_core(w_a, w_b), DL_UM, DL_UM, WL_UM).solve(num_modes=2))
    return planes


@pytest.fixture(scope="module")
def taper_planes():
    common = dict(wavelength_um=WL_UM, dl_um=DL_UM, core_h_um=CORE_H_UM,
                  n_core=N_CORE, n_clad=N_CLAD, window_w_um=2.0, window_h_um=1.3,
                  neff_margin=0.05)
    return [eme.waveguide_section(core_w_um=0.45 + 0.35 * (k / 8), num_modes=2,
                                  **common).modes for k in range(9)]


# --- overlap + match primitives --------------------------------------------


def test_transverse_overlap_self_is_orthonormal(crossing_planes):
    modes = crossing_planes[0]
    o = transverse_overlap(modes, modes)
    n = len(modes)
    assert np.allclose(np.abs(np.diag(o)), 1.0, atol=1e-6)
    off = np.abs(o - np.diag(np.diag(o)))
    assert off.max() < 0.05  # distinct localized modes are near-orthogonal
    assert np.abs(o).max() <= 1.0 + 1e-9


def test_transverse_overlap_grid_mismatch_raises(crossing_planes, taper_planes):
    with pytest.raises(ValueError):
        transverse_overlap(crossing_planes[0], taper_planes[0])


def test_transverse_overlap_spacing_mismatch_raises(crossing_planes):
    """Same point count but different dl is a different grid — the point-wise
    correlation would compare fields at different physical locations."""
    from dataclasses import replace

    modes = crossing_planes[0]
    rescaled = [replace(m, dl_x_um=m.dl_x_um * 2.0) for m in modes]
    with pytest.raises(ValueError, match="spacing"):
        transverse_overlap(modes, rescaled)


def test_match_modes_identity(crossing_planes):
    modes = crossing_planes[0]
    assign, sim, _ = match_modes(modes, modes)
    assert list(assign) == list(range(len(modes)))
    assert np.allclose(sim, 1.0, atol=1e-6)


def test_match_modes_detects_reversal(crossing_planes):
    modes = list(crossing_planes[0])
    assign, sim, _ = match_modes(modes, modes[::-1])
    n = len(modes)
    assert list(assign) == list(range(n - 1, -1, -1))
    assert np.allclose(sim, 1.0, atol=1e-6)


# --- crossing: the load-bearing case ---------------------------------------


def test_naive_ordering_mislabels_at_crossing(crossing_planes):
    # The solver's mode 0 (highest n_eff) flips which core it lives on.
    left0 = [_left_fraction(p[0]) for p in crossing_planes]
    assert max(left0) > 0.9 and min(left0) < 0.1  # jumps from core A to core B


def test_tracking_follows_physical_modes_through_crossing(crossing_planes):
    tr = track_modes(crossing_planes, min_similarity=0.3)
    assert tr.n_tracks == 2
    assert tr.has_reordering  # a swap occurred -> naive order would mislabel

    # each track stays localized on its own core across the whole sweep
    left = {
        t: [_left_fraction(crossing_planes[k][tr.mode_of[t, k]])
            for k in range(tr.n_planes)]
        for t in range(2)
    }
    # one track ~entirely left (core A), the other ~entirely right (core B)
    track_means = sorted(np.mean(left[t]) for t in range(2))
    assert track_means[0] < 0.05 and track_means[1] > 0.95

    # n_eff along each track is smooth & monotonic (one falls, one rises),
    # unlike the kinked naive max/min curves
    for t in range(2):
        ne = tr.neff[t]
        assert not np.any(np.isnan(ne))
        diffs = np.diff(ne)
        assert np.all(diffs <= 1e-9) or np.all(diffs >= -1e-9)
    # the two tracks actually cross (orders flip between first and last plane)
    assert np.sign(tr.neff[0, 0] - tr.neff[1, 0]) != np.sign(tr.neff[0, -1] - tr.neff[1, -1])

    # links are confident (weak coupling -> modes stay distinguishable)
    assert tr.min_confidence(0) > 0.8 and tr.min_confidence(1) > 0.8


def test_mode_of_records_the_swap(crossing_planes):
    tr = track_modes(crossing_planes, min_similarity=0.3)
    # track 0 is solver-mode 0 early and solver-mode 1 late (or vice versa)
    row = tr.mode_of[0]
    assert row[0] != row[-1]
    assert set(np.unique(row)) <= {0, 1}


# --- taper: tracking must be the identity ----------------------------------


def test_taper_tracks_to_identity(taper_planes):
    tr = track_modes(taper_planes)
    assert not tr.has_reordering
    # the fundamental track is solver-mode 0 at every plane
    assert np.all(tr.mode_of[0] == 0)


# --- reorder helper --------------------------------------------------------


def test_reorder_to_tracks_gives_consistent_basis(crossing_planes):
    tr = track_modes(crossing_planes, min_similarity=0.3)
    reordered = reorder_to_tracks(crossing_planes, tr)
    # every plane now has the full-track modes in the same physical order
    assert all(len(p) == len(tr.full_tracks) for p in reordered)
    # position 0 across all planes is the same physical (core-A) mode
    left_pos0 = [_left_fraction(p[0]) for p in reordered]
    assert max(left_pos0) - min(left_pos0) < 0.1  # consistent, no flip


def test_track_modes_empty_raises():
    with pytest.raises(ValueError):
        track_modes([])
