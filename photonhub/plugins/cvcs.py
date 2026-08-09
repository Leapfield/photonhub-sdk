"""CVCS — continuously-varying-cross-section EME via mode interpolation.

A naive EME staircases a z-varying device into ``N`` constant-cross-section
sections and solves the FDE modes of *every* one — ``N`` eigensolves, the
dominant cost. CVCS exploits the fact that, in a **smoothly varying** (adiabatic)
region, the modes evolve smoothly with z: solve the modes at only a few **key
planes** and **interpolate** them onto a fine sub-slicing, so the dense cascade
costs ``K`` eigensolves (``K << N``) plus cheap array interpolation.

This is the efficiency capstone on top of :mod:`photonhub.plugins.eme` and
:mod:`photonhub.plugins.mode_tracking`. Tracking is the prerequisite: to
interpolate "the same" mode between two key planes, the modes must first be put
in correspondence (and sign/phase aligned) — :func:`interpolate_mode` aligns the
pair it is given, and :func:`cvcs_sections` tracks + reorders the key planes into
a consistent basis before interpolating each track.

Method
======
Between two key planes A, B (tracked so index ``m`` is the same physical mode),
the mode at fractional position ``s in [0, 1]`` is the renormalized linear blend

    e_m(s) = normalize[ (1-s) e_m^A + s * align * e_m^B ] ,   n_eff_m(s) = (1-s) n_eff_m^A + s n_eff_m^B

(and likewise for ``h``), where ``align`` is the unit phase that makes A and B
co-phased (from their transverse-E overlap). The interpolated planes feed
:func:`photonhub.plugins.eme.run_eme` exactly like solved ones — the EME interface
re-orthonormalizes each basis, so the blend only needs to be *approximately* right.

Conversion (the fixed multimode basis)
======================================
CVCS **does** capture inter-mode conversion, *provided the whole modal basis is
carried through* — :func:`cvcs_sections` matches the key planes with a full
permutation and never drops a mode, so a mode the fundamental converts into stays
in the basis even if it is near-cutoff at the narrow planes and only well-confined
at the wide ones. With that fixed basis, interpolating between key planes and
cascading reproduces a full multimode staircase (e.g. a symmetric taper's
TE0->TE2 conversion) at ``K << N`` solves. (The conversion was never an
interpolation problem — an earlier version *dropped* the conversion target by
keeping only globally-present "tracks", which made it return the adiabatic limit.)

Validity / scope
================
Interpolation reproduces the true modes to ``O((key spacing)^2)``, so accuracy is
set by the **key-plane density relative to how fast the modes evolve**: a smoothly
varying cross-section needs few key planes; a region where the modes change quickly
needs more. The hard case is a **sharp avoided crossing** (near-degenerate modes
swapping over a tiny z-window): interpolation cannot resolve the rapid mode
variation there, so it needs near-staircase key-plane density — at which point the
plain :mod:`~photonhub.plugins.eme` staircase is the simpler tool. CVCS's ``N/K``
eigensolve saving is realized wherever the modes are smooth (the common case);
see ``benchmarks/eme/cvcs_taper.py``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence

import numpy as np

from .eme import Section
from .mode_tracking import reorder_to_tracks, track_modes, transverse_overlap
from .vector_modes import VectorMode

__all__ = ["interpolate_mode", "interpolate_plane", "cvcs_sections"]

_COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")


def interpolate_mode(mode_a: VectorMode, mode_b: VectorMode, s: float) -> VectorMode:
    """Linear blend of two corresponding modes at fraction ``s`` in ``[0, 1]``
    (``s = 0`` -> ``mode_a``, ``s = 1`` -> ``mode_b``).

    ``mode_b`` is sign/phase aligned to ``mode_a`` (via their transverse-E
    overlap) before blending, so the two add constructively. All six field
    components are interpolated and the result is renormalized to the
    :class:`VectorMode` convention (transverse-E L2 = 1); ``n_eff`` is linearly
    interpolated. Intended for *adjacent* tracked modes (see module docstring).
    """
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"s must be in [0, 1], got {s}")
    if mode_a.shape != mode_b.shape:
        raise ValueError(
            f"modes are on different grids ({mode_a.shape} vs {mode_b.shape})"
        )
    # Same POINT COUNT is not the same GRID: a blend of two modes with
    # different spacings is meaningless (matching run_eme's spacing check).
    if not (np.isclose(mode_a.dl_x_um, mode_b.dl_x_um)
            and np.isclose(mode_a.dl_y_um, mode_b.dl_y_um)):
        raise ValueError(
            f"modes have different grid spacings "
            f"(({mode_a.dl_x_um}, {mode_a.dl_y_um}) vs "
            f"({mode_b.dl_x_um}, {mode_b.dl_y_um}) um); interpolation needs "
            "one common transverse grid")
    if s == 0.0:
        return mode_a
    if s == 1.0:
        return mode_b
    overlap = complex(transverse_overlap([mode_a], [mode_b])[0, 0])
    align = np.conj(overlap) / abs(overlap) if abs(overlap) > 0 else 1.0 + 0j
    blended = {
        c: (1.0 - s) * getattr(mode_a, c) + s * align * getattr(mode_b, c)
        for c in _COMPONENTS
    }
    norm = np.sqrt(np.sum(np.abs(blended["ex"]) ** 2 + np.abs(blended["ey"]) ** 2))
    if norm == 0.0:
        raise ValueError("interpolated mode has zero transverse-E norm")
    blended = {c: v / norm for c, v in blended.items()}
    return replace(
        mode_a,
        n_eff=(1.0 - s) * mode_a.n_eff + s * mode_b.n_eff,
        n_group=None,
        **blended,
    )


def interpolate_plane(
    modes_a: Sequence[VectorMode], modes_b: Sequence[VectorMode], s: float
) -> List[VectorMode]:
    """Interpolate a whole **corresponding** (tracked, same-order, same-count)
    mode set at fraction ``s``. Mode ``m`` of the result blends ``modes_a[m]``
    with ``modes_b[m]``."""
    if len(modes_a) != len(modes_b):
        raise ValueError(
            f"plane mode counts differ ({len(modes_a)} vs {len(modes_b)}); "
            "track + reorder to a common basis first"
        )
    return [interpolate_mode(a, b, s) for a, b in zip(modes_a, modes_b)]


def cvcs_sections(
    key_planes: Sequence[Sequence[VectorMode]],
    key_z_um: Sequence[float],
    n_subslices: int,
    min_confidence: float = 0.0,
) -> List[Section]:
    """Build a dense interpolated EME section list from a few solved key planes,
    keeping a **fixed multimode basis** so inter-mode conversion is represented.

    The key planes are put in correspondence with a *full permutation* match (every
    mode is matched to one on the previous plane — nothing is dropped), so a mode
    that the fundamental converts into (even one that is near-cutoff / "born" only
    at the wider planes) stays in the basis and the conversion is captured. This is
    what lets CVCS reproduce a multimode staircase (e.g. a symmetric taper's
    TE0->TE2 conversion) at ``K << N`` solves, not just the adiabatic limit.

    Parameters
    ----------
    key_planes:
        ``K`` mode lists (``K >= 2``), the FDE modes solved at the key
        cross-sections, all on the **same transverse grid** and with the **same
        number of modes** (solve a ``num_modes`` every key plane supports).
    key_z_um:
        The ``K`` monotonically increasing z-positions (microns) of the key
        planes. The device spans ``[key_z_um[0], key_z_um[-1]]``.
    n_subslices:
        Number of interpolated, uniformly spaced propagation slices spanning the
        device. Choose ``n_subslices >> K`` (the point of CVCS).
    min_confidence:
        Optional guard (default ``0.0`` = off): if ``> 0``, reject when any
        adjacent key-plane match is weaker than this overlap — a coarse check that
        the key planes are dense enough to correspond. It is conservative (a
        near-cutoff *higher* mode can have a weak link without spoiling the
        result), so the real validation is convergence in the number of key
        planes; leave it off unless you want a hard floor.

    Returns
    -------
    list[Section]
        The first/last sections are the (solved) key endpoints as length-0 ports;
        the interior is ``n_subslices`` interpolated sections. Feed directly to
        :func:`photonhub.plugins.eme.run_eme`.
    """
    k = len(key_planes)
    if k < 2:
        raise ValueError("need at least two key planes")
    if len(key_z_um) != k:
        raise ValueError("key_z_um must have one position per key plane")
    z = np.asarray(key_z_um, dtype=float)
    if np.any(np.diff(z) <= 0):
        raise ValueError("key_z_um must be strictly increasing")
    if n_subslices < 1:
        raise ValueError("n_subslices must be >= 1")
    counts = [len(p) for p in key_planes]
    if len(set(counts)) != 1:
        raise ValueError(
            f"key planes have varying mode counts {counts}; CVCS needs a fixed "
            "modal basis — solve a num_modes every key plane supports (lower "
            "num_modes or raise neff_margin so the count is constant)"
        )

    # Full-permutation correspondence (min_similarity=0 -> nothing dropped), so the
    # whole multimode basis is carried through births/crossings.
    tracking = track_modes(key_planes, min_similarity=0.0)
    keys = reorder_to_tracks(key_planes, tracking)
    if len(keys[0]) != counts[0]:  # safety: a fixed-count basis must keep every mode
        raise ValueError("internal: tracking dropped a mode despite a fixed count")
    if min_confidence > 0.0:
        worst = min(tracking.min_confidence(t) for t in tracking.full_tracks)
        if worst < min_confidence:
            raise ValueError(
                f"key planes too sparse: weakest adjacent mode correspondence is "
                f"{worst:.2f} < min_confidence={min_confidence} (a key-plane gap "
                "likely skips a mode crossing). Add key planes between them."
            )

    total = float(z[-1] - z[0])
    dz = total / n_subslices
    sections: List[Section] = [Section(modes=keys[0], length_um=0.0)]
    for i in range(n_subslices):
        z_mid = z[0] + (i + 0.5) * dz
        j = int(np.searchsorted(z, z_mid) - 1)
        j = min(max(j, 0), k - 2)
        s = (z_mid - z[j]) / (z[j + 1] - z[j])
        modes = interpolate_plane(keys[j], keys[j + 1], float(s))
        sections.append(Section(modes=modes, length_um=dz))
    sections.append(Section(modes=keys[-1], length_um=0.0))
    return sections
