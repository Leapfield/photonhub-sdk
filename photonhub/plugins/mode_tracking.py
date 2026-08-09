"""Mode tracking — a consistent labeling of FDE modes across a sequence of planes.

A finite-difference mode solver returns the modes of each cross-section sorted by
descending ``n_eff``. That order is **not** a stable identity: as a geometry (or
wavelength, or bend radius) is swept, two modes can swap ``n_eff`` ordering at a
**crossing**, so "mode 0" silently changes which physical mode it is. Reporting a
mode's ``n_eff(z)`` curve, a per-mode S-matrix, or interpolating modes between
sampled planes (the CVCS efficiency trick) all require following each *physical*
mode through the sweep regardless of the solver's per-plane ordering.

Mode tracking establishes that correspondence by **field overlap**: between
adjacent planes, each mode is matched to the most-similar mode on the previous
plane (a global assignment, not greedy), giving a set of continuous "tracks". It
also fixes the arbitrary per-solve eigenvector sign/phase so that consecutive
modes along a track are phase-aligned — required for any interpolation between
planes.

This is the prerequisite for CVCS (continuously-varying-cross-section) modelling
and for the per-mode interpretation of an EME (:mod:`photonhub.plugins.eme`)
cascade. It is deliberately general: the planes can be the z-sections of an EME
device, or the same cross-section re-solved across a wavelength band or a bend
radius — anything that yields a sequence of mode sets **on a common transverse
grid**.

Similarity metric
=================
Modes are compared by the normalized Hermitian transverse-E correlation

    O[i, j] = <E_i, E_j> / (|E_i| |E_j|),
    <E_i, E_j> = integral (Ex_i* Ex_j + Ey_i* Ey_j) dA,   |O[i, j]| in [0, 1] ,

which is 1 for an identical field shape and ~0 for an orthogonal one (a pure
shape correlation, independent of the EME power inner product, hence robust at
any resolution). The complex phase of ``O`` gives the relative sign/phase used to
align modes along a track. The assignment maximizing the total ``|O|`` between two
planes is found with the Hungarian algorithm
(:func:`scipy.optimize.linear_sum_assignment`).

Limitation: an *exact* crossing (degenerate modes) is genuinely ambiguous — the
two modes are an arbitrary delocalized mix there. Sample finely enough that
adjacent planes stay clearly distinguishable and avoid landing a plane on the
degeneracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .vector_modes import VectorMode

__all__ = [
    "TrackingResult",
    "transverse_overlap",
    "match_modes",
    "track_modes",
    "reorder_to_tracks",
]


def _transverse_e_stack(modes: Sequence[VectorMode]) -> np.ndarray:
    """``(N, 2*npts)`` complex array of each mode's transverse E field
    ``[Ex | Ey]``, L2-normalized per mode."""
    ex = np.array([m.ex.ravel() for m in modes], dtype=complex)
    ey = np.array([m.ey.ravel() for m in modes], dtype=complex)
    field = np.concatenate([ex, ey], axis=1)  # (N, 2*npts)
    norm = np.sqrt(np.sum(np.abs(field) ** 2, axis=1, keepdims=True))
    norm[norm == 0.0] = 1.0
    return field / norm


def transverse_overlap(
    modes_a: Sequence[VectorMode], modes_b: Sequence[VectorMode]
) -> np.ndarray:
    """Complex Hermitian transverse-E correlation ``O[i, j]`` between two mode
    sets (same transverse grid). ``|O[i, j]|`` is in ``[0, 1]`` (1 == identical
    field shape); ``O``'s phase carries the relative sign/phase."""
    fa = _transverse_e_stack(modes_a)
    fb = _transverse_e_stack(modes_b)
    if fa.shape[1] != fb.shape[1]:
        raise ValueError(
            "mode sets are on different transverse grids "
            f"({fa.shape[1] // 2} vs {fb.shape[1] // 2} points); tracking needs "
            "a common grid"
        )
    # Same point count is not the same grid: also require equal spacings
    # (matching run_eme's check), else the point-wise correlation compares
    # fields at different physical locations.
    if len(modes_a) and len(modes_b):
        a0, b0 = modes_a[0], modes_b[0]
        if not (np.isclose(a0.dl_x_um, b0.dl_x_um)
                and np.isclose(a0.dl_y_um, b0.dl_y_um)):
            raise ValueError(
                f"mode sets have different grid spacings "
                f"(({a0.dl_x_um}, {a0.dl_y_um}) vs ({b0.dl_x_um}, "
                f"{b0.dl_y_um}) um); tracking needs a common grid")
    return fa.conj() @ fb.T  # (na, nb), already normalized


def match_modes(
    modes_a: Sequence[VectorMode],
    modes_b: Sequence[VectorMode],
    min_similarity: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best assignment of ``modes_b`` onto ``modes_a`` by maximum
    ``|transverse_overlap|`` (Hungarian / global optimum).

    Returns ``(assign, sim, phase)`` each of length ``len(modes_b)``:
    ``assign[j]`` is the index in ``modes_a`` matched to ``modes_b[j]`` (``-1`` if
    unmatched — extra mode, or below ``min_similarity``); ``sim[j] = |O|`` the
    match quality; ``phase[j] = O / |O|`` the unit relative phase.
    """
    overlap = transverse_overlap(modes_a, modes_b)
    mag = np.abs(overlap)
    rows, cols = linear_sum_assignment(-mag)  # maximize total |overlap|
    n_b = len(modes_b)
    assign = np.full(n_b, -1, dtype=int)
    sim = np.zeros(n_b, dtype=float)
    phase = np.ones(n_b, dtype=complex)
    for i, j in zip(rows, cols):
        if mag[i, j] >= min_similarity:
            assign[j] = i
            sim[j] = mag[i, j]
            phase[j] = overlap[i, j] / mag[i, j] if mag[i, j] > 0 else 1.0
    return assign, sim, phase


@dataclass
class TrackingResult:
    """Tracks following each physical mode across a sequence of ``n_planes``
    planes. Arrays are ``(n_tracks, n_planes)``; an absent track at a plane (the
    mode cut off, or had not appeared yet) is marked ``-1`` / ``nan``.

    Attributes
    ----------
    mode_of:
        ``mode_of[t, k]`` = the solver-mode index of track ``t`` at plane ``k``
        (``-1`` if absent). This is the relabeling: read row ``t`` to follow one
        physical mode through the sweep.
    neff:
        ``n_eff`` of track ``t`` at plane ``k`` (``nan`` if absent) — the smooth,
        crossing-aware dispersion curve.
    confidence:
        ``|overlap|`` of the match that linked this track to the previous plane
        (``nan`` at the track's first plane). Low values flag an unreliable link
        (near a degeneracy / under-sampling).
    phase:
        Unit complex factor to multiply that solver mode by so it is sign/phase
        aligned with the track's previous plane (cumulative). For real lossless
        modes this is +-1.
    """

    n_planes: int
    n_tracks: int
    mode_of: np.ndarray
    neff: np.ndarray
    confidence: np.ndarray
    phase: np.ndarray

    @property
    def has_reordering(self) -> bool:
        """True if at any plane the tracks are not in solver (descending-n_eff)
        order — i.e. a crossing/swap occurred and naive ordering would mislabel."""
        for k in range(self.n_planes):
            present = self.mode_of[:, k]
            present = present[present >= 0]
            if list(present) != sorted(present):
                return True
        return False

    @property
    def full_tracks(self) -> List[int]:
        """Track ids present at *every* plane (a clean basis for reordering)."""
        return [t for t in range(self.n_tracks) if bool(np.all(self.mode_of[t] >= 0))]

    def min_confidence(self, track: int) -> float:
        """Weakest adjacent-plane link along ``track`` (ignores its start
        ``nan``) — a single-number reliability score for the track."""
        c = self.confidence[track]
        c = c[~np.isnan(c)]
        return float(c.min()) if c.size else float("nan")


def track_modes(
    mode_sets: Sequence[Sequence[VectorMode]], min_similarity: float = 0.3
) -> TrackingResult:
    """Follow each physical mode across ``mode_sets`` (one mode list per plane, on
    a common transverse grid) by chaining adjacent-plane overlap matches.

    Plane 0 seeds one track per mode; each later plane's modes inherit the track
    of their best match on the previous plane (a mode whose best match is below
    ``min_similarity``, or an unmatched extra mode, seeds a new track).
    """
    n_planes = len(mode_sets)
    if n_planes == 0:
        raise ValueError("mode_sets is empty")

    # records: (track, plane, mode_idx, n_eff, confidence, phase)
    records: List[Tuple[int, int, int, float, float, complex]] = []
    n0 = len(mode_sets[0])
    cur_track = list(range(n0))
    cur_phase = [1.0 + 0j] * n0
    next_track = n0
    for t in range(n0):
        records.append((t, 0, t, mode_sets[0][t].n_eff, np.nan, 1.0 + 0j))

    for k in range(1, n_planes):
        assign, sim, phase = match_modes(
            mode_sets[k - 1], mode_sets[k], min_similarity
        )
        new_track = [-1] * len(mode_sets[k])
        new_phase = [1.0 + 0j] * len(mode_sets[k])
        for j in range(len(mode_sets[k])):
            i = assign[j]
            if i >= 0:
                tr = cur_track[i]
                # align so consecutive modes along the track are phase-continuous
                ph = cur_phase[i] * np.conj(phase[j])
                new_track[j] = tr
                new_phase[j] = ph
                records.append((tr, k, j, mode_sets[k][j].n_eff, sim[j], ph))
            else:
                tr = next_track
                next_track += 1
                new_track[j] = tr
                records.append(
                    (tr, k, j, mode_sets[k][j].n_eff, np.nan, 1.0 + 0j)
                )
        cur_track, cur_phase = new_track, new_phase

    n_tracks = next_track
    mode_of = np.full((n_tracks, n_planes), -1, dtype=int)
    neff = np.full((n_tracks, n_planes), np.nan, dtype=float)
    confidence = np.full((n_tracks, n_planes), np.nan, dtype=float)
    phase_arr = np.zeros((n_tracks, n_planes), dtype=complex)
    for tr, k, j, ne, cf, ph in records:
        mode_of[tr, k] = j
        neff[tr, k] = ne
        confidence[tr, k] = cf
        phase_arr[tr, k] = ph
    return TrackingResult(
        n_planes=n_planes,
        n_tracks=n_tracks,
        mode_of=mode_of,
        neff=neff,
        confidence=confidence,
        phase=phase_arr,
    )


def reorder_to_tracks(
    mode_sets: Sequence[Sequence[VectorMode]], tracking: TrackingResult
) -> List[List[VectorMode]]:
    """Re-order each plane's modes into a consistent track order, keeping only the
    tracks present at every plane (:attr:`TrackingResult.full_tracks`).

    The result has the same number of modes at every plane, in the same physical
    order — a consistent basis for an EME cascade (so ``S21[0, 0]`` is always the
    same tracked mode) or for plane-to-plane interpolation.
    """
    full = tracking.full_tracks
    return [
        [mode_sets[k][tracking.mode_of[t, k]] for t in full]
        for k in range(tracking.n_planes)
    ]
