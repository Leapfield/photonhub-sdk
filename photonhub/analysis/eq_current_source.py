"""source-current (Huygens) mode source, per-cell phased-dipole source planes.

Replaces the single-scalar-aux-carrier mode launch with the textbook
source-current pair built from the engine's own discrete Yee eigenmode
(:func:`~photonhub.analysis.yee_mode.solve_yee_mode`):

    J = n̂ × H_mode   stamped as electric dipoles at the E Yee points of plane k0
    M = −n̂ × E_mode  stamped as magnetic dipoles at the H Yee points half a cell
                      upstream (the TF/SF straddle)

Every dipole carries the mode's per-cell **complex** field value: amplitude from
|A| and the phase from arg(A) plus two half-offsets the aux-line architecture
cannot represent, the half-cell spatial phase e^{iβ·dl/2} between the E and H
planes, and the half-step temporal phase e^{iω·dt/2} (the engine samples both
dipole types at (s+½)·dt while the J source plane acts at the E time level). This is the
per-cell discrete Huygens construction, whose shed field vanishes with
resolution, translated onto the engine's dipole conventions (dipoles: E −= (dt/ε)·A·g,
H −= (dt/μ0)·A·g, g = env·cos(2πf0(t−t0)+phase)).

Measured vs the aux-line launch (straight Si strip, GPU):
flux loss 0.44→0.09 % @20 nm (and FALLING with resolution where the aux-line launch floors),
backward 0.20→0.003 %, near-source p_in placement wobble 0.24→0.11 %.

The trade: a cloud of ~10–40 k point dipoles instead of one ModeSource (linear
per-step cost, negligible next to the curl updates) and a frozen single-frequency
profile (like a single-frequency mode source; the broadband carriers remain the
tool for wide-band launches).
"""
from __future__ import annotations

import math
from typing import List, Mapping, Optional, Tuple

import numpy as np

from ..components import PointDipole
# the band test of Simulation.check_runnable: the clip must keep what it keeps
from ..components.simulation import _BAND_EDGE_TOL_UM
from ..viz import _geometry as _geom
from ._constants import C0 as _C0, engine_dt_s

_AXES = "xyz"


def _launched_power(mode, dl_um: float, wh_um=None, wv_um=None,
                    fold_low: Tuple[bool, bool] = (False, False),
                    keep: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
    """Forward modal Poynting power of the discrete mode at unit amplitude (W for
    fields in V/m / A/m on the dl grid). ``wh_um``/``wv_um`` = optional per-node
    cell widths for a GRADED window (the area element is then the outer product
    of the dual widths); ``None`` keeps the uniform ``dl^2`` element
    bit-identically.

    ``keep`` = optional ``(keep_1, keep_2)`` boolean masks ``[iv, ih]`` naming
    the window entries whose dipoles the sheet actually stamps: ``keep_1`` for
    the ``ex·hy*`` term (the E_h/H_v pair at ``(h+½, v)``) and ``keep_2`` for
    ``ey·hx*`` (the E_v/H_h pair at ``(h, v+½)``). Entries the domain clips
    (the interior-margin test in ``_build_sheet``) contribute nothing to the
    launched power, so the normalization is the power of the sheet as placed.
    ``None`` or all-true masks keep every historical expression bit-identically.

    ``fold_low`` = (h folded, v folded): the window's MIN face on that in-plane
    axis sits ON a symmetry plane (``window_nodes``' ``h_bc``/``v_bc`` is
    ``"pec"``/``"pmc"``; a ``"periodic"`` window is the whole period, never
    folded).
    The half-domain power integral then weights an ON-PLANE node row by half a
    cell, the full-width first entry (uniform ``dl``, graded ``dual[0]=dq[0]``)
    spills ``dl/2`` into the MIRROR half and over-counts a fold-antinode mode by
    ~dl/(2 w_eff) (the same quadrature bug fixed in the modal readout,
    mode_overlap._overlap_terms). Only the NODE-registered Poynting term on the
    folded axis is halved: in the (h, v) window frame ``ex·hy*`` sits +½-cell
    along h (node along v) and ``ey·hx*`` node along h (+½ along v). The
    normalization ``power_watts`` is thereby the mode's exact power through the
    MODELED (half) domain. Default (False, False) is bit-identical to the
    historical integral."""
    s1 = np.real(np.asarray(mode.ex) * np.conj(np.asarray(mode.hy)))
    s2 = np.real(np.asarray(mode.ey) * np.conj(np.asarray(mode.hx)))
    if keep is not None:
        k1 = np.asarray(keep[0], dtype=bool)
        k2 = np.asarray(keep[1], dtype=bool)
        if k1.shape != s1.shape or k2.shape != s2.shape:
            raise ValueError(
                f"keep masks {k1.shape}/{k2.shape} do not match the mode window {s1.shape}")
        if not (k1.all() and k2.all()):
            s1 = np.where(k1, s1, 0.0)
            s2 = np.where(k2, s2, 0.0)
    nv, nh = s1.shape
    if wh_um is None:
        wh = np.full(nh, float(dl_um))
        wv = np.full(nv, float(dl_um))
        if fold_low == (False, False):
            dA = (dl_um * 1e-6) ** 2
            return float(0.5 * np.sum(s1 - s2) * dA)
    else:
        wh = np.asarray(wh_um, dtype=float)
        wv = np.asarray(wv_um, dtype=float)
        if fold_low == (False, False):
            dA = np.outer(wv, wh) * 1e-12                    # um^2 -> m^2
            return float(0.5 * np.sum((s1 - s2) * dA))
    def _halved(w: np.ndarray, folded: bool) -> np.ndarray:
        if not folded or w.size < 1:
            return w
        wf = w.copy()
        wf[0] *= 0.5
        return wf
    wh_n = _halved(wh, fold_low[0])   # node-registered along h
    wv_n = _halved(wv, fold_low[1])   # node-registered along v
    dA_1 = np.outer(wv_n, wh) * 1e-12   # ex·hy*: +½ along h, node along v
    dA_2 = np.outer(wv, wh_n) * 1e-12   # ey·hx*: node along h, +½ along v
    return float(0.5 * (np.sum(s1 * dA_1) - np.sum(s2 * dA_2)))


# Largest fraction of a sheet's peak amplitude allowed to sit inside a
# PML/absorber band on a TRANSVERSE axis when the sheet is BUILT. It only
# decides which windows fail here, at build time, with a window-specific
# message: a sheet with any dipole in the band is still refused when it is
# run (Simulation.check_runnable) unless the caller opts in
# with allow_sources_in_boundary=True. The failure this rejects at once is a
# WINDOW SIZED INTO the absorber (the 08-14 edge-coupler auto15 frame put 30 %
# of the beam's peak amplitude inside the transverse PML, sized for 20*dl
# instead of 20 LOCAL cells).
_ABSORBER_AMPLITUDE_TOL = 0.05



def _absorbing_interval(sim, axis_index: int):
    """``(lo_um, hi_um, boundary)`` of the axis' nonabsorbing interior, or
    ``None`` when the axis has no absorbing boundary, or when ``sim`` is a
    duck-typed shell without the resolver (unit-test rigs; a real
    :class:`Simulation` always has it)."""
    if not hasattr(sim, "_nonabsorbing_bounds_um"):
        return None
    lo, hi, boundary = sim._nonabsorbing_bounds_um(axis_index)
    if boundary not in ("pml", "absorber"):
        return None
    return float(lo), float(hi), boundary


def _reject_absorber_overlap(sim, dips, *, a: int, ih_ax: int, iv_ax: int,
                             aJ: float, aM: float) -> None:
    """Refuse a Huygens sheet that meaningfully overlaps PML/absorber cells.

    Dipoles inside an absorbing band drive a damped, coordinate-stretched
    medium: the run completes and the launch is silently wrong (power AND
    profile) instead of failing. Boundary layers are counted in realized LOCAL
    cells (graded axes carve physically thick bands from coarse face cells),
    which is exactly how such a sheet ends up there by accident.

    Two rules:

    - the J/M sheet PLANES inside the propagation axis' band is always an
      error, the entire launch is inside the absorber;
    - on a transverse in-plane axis, in-band dipoles are an error only when
      one carries more than ``_ABSORBER_AMPLITUDE_TOL`` of its own sheet's
      (J or M) peak amplitude, window tails against the wall are expected
      and harmless.
    """
    span = _absorbing_interval(sim, a)
    if span is not None:
        lo, hi, boundary = span
        for name, pos in (("J (electric)", aJ), ("M (magnetic)", aM)):
            if not (lo + 1e-12 < pos < hi - 1e-12):
                raise ValueError(
                    f"the {name} sheet plane at {pos:.6g} um on axis "
                    f"'{_AXES[a]}' lies inside the {boundary} band: the "
                    f"nonabsorbing interior is ({lo:.6g}, {hi:.6g}) um "
                    f"(boundary layers are counted in realized LOCAL cells, "
                    f"so a graded axis' band is thicker than layers*dl). A "
                    f"sheet inside the absorber launches silently wrong "
                    f"fields — move position_um into the interior.")
    if not dips:
        return
    peaks = {}
    for d in dips:
        kind = d.polarization[0]                        # 'E' (J) / 'H' (M)
        peaks[kind] = max(peaks.get(kind, 0.0), d.amplitude)
    problems = []
    for axis_index in (ih_ax, iv_ax):
        span = _absorbing_interval(sim, axis_index)
        if span is None:
            continue
        lo, hi, boundary = span
        worst, count = 0.0, 0
        for d in dips:
            # In-band means STRICTLY beyond an interval edge: a symmetry-folded
            # axis has lo = 0.0 with no lower band at all, and its on-plane
            # dipole row sits exactly AT 0.0 (it can carry the beam's peak);
            # a window edge exactly ON the first absorber node is likewise
            # not yet inside the band's cells.
            c = d.center_um[axis_index]
            if not (c < lo - 1e-9 or c > hi + 1e-9):
                continue
            count += 1
            worst = max(worst, d.amplitude / peaks[d.polarization[0]])
        if worst > _ABSORBER_AMPLITUDE_TOL:
            problems.append(
                f"axis '{_AXES[axis_index]}': {count} dipoles beyond the "
                f"nonabsorbing interval ({lo:.6g}, {hi:.6g}) um carry up to "
                f"{worst:.1%} of the sheet's peak amplitude inside the "
                f"{boundary} band")
    if problems:
        raise ValueError(
            "the Huygens sheet window reaches into absorbing boundary cells "
            "with significant amplitude — the launch would be silently wrong "
            "(the band is carved from realized LOCAL cells, so on a graded "
            "axis it is thicker than layers*dl): " + "; ".join(problems) +
            ". Shrink the window (half_w_um / half_v_um / window_sigmas), "
            "enlarge the domain, or re-center the beam. A smaller tail in the "
            "band builds, but the run is refused until every dipole clears "
            "the layers.")


def equivalence_current_source(
    sim,
    mode,
    *,
    axis: str,
    position_um: float,
    source_time,
    direction: str = "+",
    h_center_um: float,
    v_center_um: float,
    half_w_um: float,
    half_v_um: float,
    power_watts: Optional[float] = 1.0,
    amplitude_threshold: float = 1e-6,
    extra_j_phase: float = 0.0,
    _window_origin: Optional[Tuple[float, float]] = None,
    modes_by_freq: Optional[Mapping[float, object]] = None,
    _clip_to_interior: Tuple[bool, bool] = (False, False),
) -> List[PointDipole]:
    """Build the phased-dipole Huygens source planes launching ``mode`` along ``axis``.

    ``mode`` must be the **discrete Yee eigenmode** from :func:`solve_yee_mode`
    solved with the SAME window arguments (``h_center_um``/``v_center_um``/
    ``half_w_um``/``half_v_um``/the simulation's ``dl``), they are used here to
    re-derive the window's grid registration exactly. ``source_time`` is the
    shared :class:`GaussianPulse` envelope (its ``phase`` is overridden per
    dipole). ``power_watts`` scales the launched modal power (``None`` = leave
    the mode's own units): the forward Poynting power, on the engine's own
    discrete quadrature, of the dipoles the source plane actually stamps. Window nodes
    the domain clips (the interior margin of a quarter cell at each wall) are
    left out of the normalization, so the placed source plane carries exactly
    ``power_watts``. A plain periodic in-plane axis has no such margin: its
    window is the whole period and every column is stamped, the single
    column of a one-cell (quasi-2-D) axis included. The normalization is the
    power the Yee scheme carries through the planes downstream, the continuum
    integral times ``cos(beta*dl/2)`` of the source plane's own cell,
    so without symmetry a ``PowerMonitor`` or a port's ``mode_power`` downstream
    reads ``power_watts`` back in watts. Under symmetry planes this builder is
    the primitive: ``power_watts`` is the power into the modeled part, so a
    source plane centered on k planes launches ``2^k * power_watts`` into the whole
    device, which is what the ``PowerMonitor`` and the port report. The
    launches built on it (``mode_launch``, ``gaussian_beam_source``,
    ``import_source``, ``thin_lens_source``) take the whole device's power
    and pass ``power_watts / 2^k`` here. ``amplitude_threshold`` compares
    each field with its own peak; a sample is kept when either field reaches
    that fraction, and its electric and magnetic currents are kept or dropped
    together. Power is normalized on the kept samples. Entries outside the
    domain are dropped.

    Returns the list of :class:`PointDipole` (electric + magnetic) to put in
    ``Simulation.sources``. Single-frequency profile (band-centre), like a
    single-frequency (``num_freqs=1``) mode source.

    **Broadband (``num_freqs`` analogue).** Pass
    ``modes_by_freq`` (``{freq_hz: Yee mode}``, N >= 2, each solved AT that
    frequency with the SAME window arguments) to launch a BROADBAND mode: this
    builds one dipole source plane per frequency, source plane k stamped from the mode at
    ``freqs[k]`` (its own ``n_eff``/fields drive the geometric + half-step
    phases) and driven by the partition-of-unity WINDOWED carrier for that
    frequency (``source_time`` copied with ``band_freqs_hz=sorted_freqs``,
    ``carrier_index=k``). Because the windows sum to 1, the source planes reconstruct
    the full source pulse while the per-frequency mode dominates near its own
    sample, the eq-current twin of the broadband ModeSource, now on the
    graded grid too. Each source plane is normalized to the full ``power_watts`` (the
    window, not an amplitude split, apportions the band, exactly like the broadband ModeSource). The
    concatenation of all N source planes is returned. ``modes_by_freq`` None or with a
    single entry falls through to the single-frequency path (``mode`` is used),
    bit-identical to the pre-broadband build.

    Symmetry planes (half-domain sims) are honored automatically, matching
    :func:`solve_yee_mode`'s window rule: the source plane is clipped at the plane,
    ON-plane dipoles are kept at 1x amplitude (they are self-mirror, the
    boundary supplies the image; the engine's own half==full tests use
    unchanged amplitudes, and odd-parity components are exactly 0 there by the
    mode's parity BC), and half-cell-offset dipoles get their mirror partner
    from the boundary. ``power_watts`` is the modeled-part launch power.
    Full-plane public flux and modal readouts return ``2^k * power_watts`` for
    k in-plane symmetry planes. ``RunResult.wire()`` retains the recorded
    modeled-part flux. Transmission ratios are normalization-invariant,
    so T needs no factor bookkeeping."""
    if modes_by_freq is not None and len(modes_by_freq) >= 2:
        freqs = sorted(float(f) for f in modes_by_freq)
        dips: List[PointDipole] = []
        for k, fk in enumerate(freqs):
            # Sheet k: the mode solved AT freqs[k] drives the geometric/temporal
            # phases (its own n_eff/frequency); the shared pulse is windowed to
            # this carrier (band = sorted freqs, index = k). The windows sum to
            # 1, so the N sheets reconstruct the source pulse.
            st_k = source_time.model_copy(
                update={"band_freqs_hz": tuple(freqs), "carrier_index": k})
            dips.extend(_build_sheet(
                sim, modes_by_freq[fk], axis=axis, position_um=position_um,
                source_time=st_k, direction=direction, h_center_um=h_center_um,
                v_center_um=v_center_um, half_w_um=half_w_um,
                half_v_um=half_v_um, power_watts=power_watts,
                amplitude_threshold=amplitude_threshold,
                extra_j_phase=extra_j_phase, _window_origin=_window_origin,
                sheet_freq_hz=fk, _clip_to_interior=_clip_to_interior))
        return dips
    return _build_sheet(
        sim, mode, axis=axis, position_um=position_um, source_time=source_time,
        direction=direction, h_center_um=h_center_um, v_center_um=v_center_um,
        half_w_um=half_w_um, half_v_um=half_v_um, power_watts=power_watts,
        amplitude_threshold=amplitude_threshold, extra_j_phase=extra_j_phase,
        _window_origin=_window_origin, sheet_freq_hz=None,
        _clip_to_interior=_clip_to_interior)


def _build_sheet(
    sim,
    mode,
    *,
    axis: str,
    position_um: float,
    source_time,
    direction: str = "+",
    h_center_um: float,
    v_center_um: float,
    half_w_um: float,
    half_v_um: float,
    power_watts: Optional[float] = 1.0,
    amplitude_threshold: float = 1e-6,
    extra_j_phase: float = 0.0,
    _window_origin: Optional[Tuple[float, float]] = None,
    sheet_freq_hz: Optional[float] = None,
    _clip_to_interior: Tuple[bool, bool] = (False, False),
) -> List[PointDipole]:
    """One phased-dipole Huygens sheet (the single-frequency build). The
    geometric (beta), half-cell and half-step temporal phases use
    ``sheet_freq_hz``, the frequency this sheet's ``mode`` was solved at ,
    defaulting to ``source_time.freq0_hz`` (the plain single-frequency case,
    bit-identical). ``source_time`` is stamped on every dipole (broadband sheets
    pass a copy carrying ``band_freqs_hz``/``carrier_index``).
    ``amplitude_threshold`` compares each field with its own peak, keeps or
    drops paired currents together, and normalizes power on the kept pairs."""
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    if not math.isfinite(amplitude_threshold) or not 0.0 <= amplitude_threshold < 1.0:
        raise ValueError(
            "amplitude_threshold must be finite and in [0, 1); "
            f"got {amplitude_threshold!r}")
    # The frequency this sheet's geometry/phase is tuned to (the mode's own
    # solve frequency for a broadband sheet; the pulse centre otherwise).
    sheet_freq_hz = (float(source_time.freq0_hz) if sheet_freq_hz is None
                     else float(sheet_freq_hz))
    dl = getattr(sim.grid, "dl_um", None)
    if not dl:
        raise ValueError(
            "equivalence_current_source needs the grid's base dl_um")
    a = _geom.axis_index(axis)
    h_letter, v_letter = _geom.in_plane_axes(axis)     # SORTED pair (h < v)
    ih_ax, iv_ax = _AXES.index(h_letter), _AXES.index(v_letter)
    # in_plane_axes returns the SORTED (h, v) pair, which is right-handed
    # (ĥ×v̂ = +â) for x- and z-cuts but LEFT-handed for a y-cut ((x, z):
    # x̂×ẑ = −ŷ). The Huygens sheet below stamps J = n̂×H and M = −n̂×E using the
    # right-handed component formulas; on a left-handed cut the M=−n̂×E sheet
    # comes out sign-flipped relative to J, so the one-sided cancellation selects
    # the WRONG side and the mode launches BACKWARD (verified: +y eq-current
    # launch put ~98% of the flux toward −y). The frame handedness s = sign of the
    # permutation (h, v, axis) restores M = −n̂×E in the physical frame (J is
    # already correct); s = +1 for x/z, −1 for y.
    hand = 1.0 if (ih_ax - iv_ax) * (iv_ax - a) * (a - ih_ax) > 0 else -1.0

    from ..components.grid import graded_primary_spacings

    def _ladder(idx):
        q = sim._axis_coords_um(idx) if hasattr(sim, "_axis_coords_um") else None
        if q is None:
            return None, None                       # uniform axis
        q = np.asarray(q, dtype=float)
        return q, np.asarray(graded_primary_spacings(tuple(q)), dtype=float)

    ladders = [_ladder(i) for i in range(3)]
    graded_any = any(q is not None for q, _ in ladders)
    qa, dqa = ladders[a]

    # The engine timestep, honoring the simulation's OWN courant (RunSpec,
    # user-settable): the half-step temporal phase below de-tunes silently if
    # this is computed with a hardcoded 0.99 while the run uses another value.
    # Deliberately a HARD attribute access — a sim-like object without a run
    # spec must fail loudly here, not launch with a silently guessed phase.
    # Graded grids use the engine's generalized Courant over the
    # per-axis MINIMUM spacings; uniform keeps the legacy formula verbatim.
    if not graded_any:
        dt = engine_dt_s(dl, float(sim.run.courant))
    else:
        mins = [float(np.min(dq)) if dq is not None else float(dl)
                for _, dq in ladders]
        inv = sum(1.0 / (m * 1e-6) ** 2 for m in mins)
        dt = float(sim.run.courant) / (_C0 * math.sqrt(inv))
    w0 = 2.0 * math.pi * sheet_freq_hz
    lam_um = _C0 / sheet_freq_hz * 1e6
    beta = float(mode.n_eff) * 2.0 * math.pi / lam_um  # 1/um
    # Propagation-axis placement: the J sheet on the E plane at node k0, the M
    # sheet on the H nodes half a LOCAL cell upstream (direction '+') or
    # downstream ('-'). On a graded propagation axis k0 is the nearest primary
    # node and every half-cell is the LOCAL primal width; wJ/wM are the widths
    # the sheet current densities are smeared over (dual at the E node, primal
    # at the H node) — the graded generalization of the uniform 1/dl amplitude.
    if qa is None:
        k0 = int(round(position_um / dl))
        kM = k0 - 1 if direction == "+" else k0        # H-node cell of the M sheet
        aM = (kM + 0.5) * dl
        aJ = k0 * dl
        halfcell_a = None                              # marker: uniform phase
        wJ = wM = float(dl)
    else:
        k0 = int(np.argmin(np.abs(qa - position_um)))
        kM = max(k0 - 1, 0) if direction == "+" else k0
        aJ = float(qa[k0])
        aM = float(qa[kM] + 0.5 * dqa[kM])
        halfcell_a = abs(aJ - aM)
        wJ = float(0.5 * (dqa[max(k0 - 1, 0)] + dqa[k0]))   # dual width at E
        wM = float(dqa[kM])                                  # primal width at H
    # J-sheet extra phase: half-cell spatial e^{i beta dl/2} — the incident H is
    # evaluated half a cell UPSTREAM of the E plane for EITHER direction, so this
    # does not flip sign — plus the half-step temporal e^{i w dt/2}. Reversing the
    # launch flips the mode's H relative to E (Poynting reversal): an extra pi on
    # the J sheet only. The M sheet moves to the downstream H nodes.
    ph_j = ((beta * dl / 2.0 if halfcell_a is None else beta * halfcell_a)
            + w0 * dt / 2.0 + extra_j_phase
            + (0.0 if direction == "+" else math.pi))

    ex, ey = np.asarray(mode.ex), np.asarray(mode.ey)  # [iv, ih], native stagger
    hx, hy = np.asarray(mode.hx), np.asarray(mode.hy)
    nv, nh = ex.shape
    # Window registration — the SHARED snap+symmetry-clip rule (bit-identical
    # to the Yee mode solve's, or the dipoles would carry a mode solved on a
    # shifted grid). window_nodes returns the sim's OWN node ladders (graded
    # axes carry their true nonuniform coordinates). With a symmetry plane
    # on an in-plane min face the window starts ON the plane.
    #   _window_origin (mode_launch, uniform-only) overrides the floor-snap
    #   with the origin the mode RECORDED (center_offset_um) — exact grid
    #   multiples, immune to the float-boundary sensitivity of re-deriving
    #   half_w from the mode.
    from .yee_mode import (is_plain_periodic_axis, min_face_symmetry_bcs,
                           window_nodes)
    h_dq = v_dq = None
    if _window_origin is not None:
        h_lo, v_lo = _window_origin
        sh_bc, sv_bc = min_face_symmetry_bcs(sim, axis)
        h_bc = sh_bc if h_lo == 0.0 else None
        v_bc = sv_bc if v_lo == 0.0 else None
    else:
        h_nodes, h_dq, h_bc, v_nodes, v_dq, v_bc = window_nodes(
            sim, axis, h_center=h_center_um, half_w=half_w_um,
            v_center=v_center_um, half_v=half_v_um, dl=dl)
        h_lo, v_lo = float(h_nodes[0]), float(v_nodes[0])
    graded_t = h_dq is not None or v_dq is not None
    if graded_t:
        if (nv, nh) != (len(v_nodes), len(h_nodes)):
            raise ValueError(
                f"mode window mismatch: the mode arrays are {(nv, nh)} but the "
                f"window ladder is {(len(v_nodes), len(h_nodes))} — solve the "
                "mode with the SAME window arguments on the SAME simulation")
        h_node = np.asarray(h_nodes, dtype=float)
        v_node = np.asarray(v_nodes, dtype=float)
        h_mid = h_node + 0.5 * (h_dq if h_dq is not None else dl)
        v_mid = v_node + 0.5 * (v_dq if v_dq is not None else dl)
    dom = sim.size_um
    # A symmetry min face is NOT an interior-margin face: on-plane dipoles
    # are SELF-MIRROR at 1x amplitude (the engine's own half==full tests drive
    # a single on-plane dipole at unchanged amplitude — no half-weights), and odd-parity components carry a mode value of exactly
    # 0 there (the parity BC pins them), so keeping the on-plane row is both
    # required (even components peak there) and safe. min_lo[i] therefore
    # admits coordinate 0 on symmetry axes only. Margins are LOCAL cell widths
    # on graded axes (uniform axes keep the 0.25*dl quarter cell verbatim).
    if not graded_any:
        min_lo = [0.25 * dl] * 3
        max_hi = [dom[i] - 0.25 * dl for i in range(3)]
    else:
        min_lo, max_hi = [], []
        for i in range(3):
            _, dq_i = ladders[i]
            lo_w = float(dq_i[0]) if dq_i is not None else float(dl)
            hi_w = float(dq_i[-1]) if dq_i is not None else float(dl)
            min_lo.append(0.25 * lo_w)
            max_hi.append(dom[i] - 0.25 * hi_w)
    # A plain periodic in-plane axis has no margin either: coordinate 0 is an
    # interior node of the wrapped ladder (the engine admits it, and on a
    # one-cell axis it is the ONLY node — dropping the node-registered
    # E_v/H_h dipoles there would launch the TM slab family at zero).
    if h_bc is not None or is_plain_periodic_axis(sim, h_letter):
        min_lo[ih_ax] = -1e-9
    if v_bc is not None or is_plain_periodic_axis(sim, v_letter):
        min_lo[iv_ax] = -1e-9
    # Its upper margin is the REALIZED period: the window spans every realized
    # cell, and a size_um that rounds up to one more cell would otherwise clip
    # that cell's column against the nominal extent.
    if hasattr(sim, "_realized_um"):
        for i_ax, letter in ((ih_ax, h_letter), (iv_ax, v_letter)):
            if is_plain_periodic_axis(sim, letter):
                _, dq_i = ladders[i_ax]
                hi_w = float(dq_i[-1]) if dq_i is not None else float(dl)
                max_hi[i_ax] = float(sim._realized_um()[i_ax]) - 0.25 * hi_w
    # A window the SDK sized itself (a Gaussian beam's default window) is
    # clipped to the absorber-free interior of each flagged in-plane axis: its
    # tail entries inside the PML/absorber layers stamp no dipole, as the
    # domain wall's do, so the run is not refused for them and the
    # normalization counts only the placed dipoles. An explicit window is the
    # caller's and is never clipped here. The clip may drop only a tail: an
    # entry it drops above the build-time tolerance is refused below, as an
    # unclipped window with that amplitude in the band would be.
    domain_lo, domain_hi = list(min_lo), list(max_hi)
    clipped = []
    for flag, i_ax in zip(_clip_to_interior, (ih_ax, iv_ax)):
        span = _absorbing_interval(sim, i_ax) if flag else None
        if span is not None:
            min_lo[i_ax] = max(min_lo[i_ax], span[0] - _BAND_EDGE_TOL_UM)
            max_hi[i_ax] = min(max_hi[i_ax], span[1] + _BAND_EDGE_TOL_UM)
            clipped.append((i_ax, span))
    # The in-plane Yee points of every window entry, with the SAME float
    # arithmetic the stamping loop below uses, and which of them pass the
    # interior-margin test. A window the domain clips (the far wall row) stamps
    # no dipole there, so those entries carry no launched power: the
    # normalization below counts only what is placed. A plain periodic axis
    # clips nothing: its window is the whole period (a one-cell axis is the
    # single node 0, admitted above), so every column is kept.
    if not graded_t:
        h_pts = h_lo + np.arange(nh) * dl
        v_pts = v_lo + np.arange(nv) * dl
        h_mid_pts = h_pts + 0.5 * dl
        v_mid_pts = v_pts + 0.5 * dl
    else:
        h_pts, v_pts = h_node, v_node
        h_mid_pts, v_mid_pts = h_mid, v_mid

    def _inside(pts, i_ax):
        return (min_lo[i_ax] < pts) & (pts < max_hi[i_ax])

    in_h_node, in_h_mid = _inside(h_pts, ih_ax), _inside(h_mid_pts, ih_ax)
    in_v_node, in_v_mid = _inside(v_pts, iv_ax), _inside(v_mid_pts, iv_ax)
    keep = (in_v_node[:, None] & in_h_mid[None, :],    # ex·hy*: E_h/H_v at (h+½, v)
            in_v_mid[:, None] & in_h_node[None, :])    # ey·hx*: E_v/H_h at (h, v+½)
    # E and H have different units. Compare each to its own peak, then retain
    # the paired currents together at their shared Yee position.
    e_peak = max(float(np.abs(ex).max()), float(np.abs(ey).max()))
    h_peak = max(float(np.abs(hx).max()), float(np.abs(hy).max()))
    if e_peak == 0.0 and h_peak == 0.0:
        raise ValueError("cannot launch a sheet where both E and H are zero everywhere")
    if e_peak == 0.0 or h_peak == 0.0:
        raise ValueError("a one-sided launch needs both fields: E and H must each be nonzero")
    envelope = (
        np.maximum(np.abs(ex) / e_peak, np.abs(hy) / h_peak),
        np.maximum(np.abs(ey) / e_peak, np.abs(hx) / h_peak),
    )
    cutoff_masks = tuple(weight >= amplitude_threshold for weight in envelope)
    keep = tuple(placed & retained for placed, retained in zip(keep, cutoff_masks))
    if clipped:
        def _in_domain(pts, i_ax):
            return (domain_lo[i_ax] < pts) & (pts < domain_hi[i_ax])
        placed = (_in_domain(v_pts, iv_ax)[:, None] & _in_domain(h_mid_pts, ih_ax)[None, :],
                  _in_domain(v_mid_pts, iv_ax)[:, None] & _in_domain(h_pts, ih_ax)[None, :])
        # per clipped axis, the entries only that axis' clip drops
        pts_of = {ih_ax: (h_mid_pts[None, :], h_pts[None, :]),
                  iv_ax: (v_pts[:, None], v_mid_pts[:, None])}
        origin = tuple(getattr(sim, "origin_um", (0.0, 0.0, 0.0)))
        faults = []
        for i_ax, (lo, hi, boundary) in clipped:
            worst = 0.0
            for field, dom_mask, pts in zip((ex, ey), placed, pts_of[i_ax]):
                amp = np.abs(field)
                peak = float(amp[dom_mask].max()) if dom_mask.any() else 0.0
                dropped = dom_mask & np.broadcast_to(
                    (pts <= lo - _BAND_EDGE_TOL_UM) | (pts >= hi + _BAND_EDGE_TOL_UM), dom_mask.shape)
                if peak > 0.0 and dropped.any():
                    worst = max(worst, float(amp[dropped].max()) / peak)
            if worst > _ABSORBER_AMPLITUDE_TOL:
                faults.append(
                    f"axis '{_AXES[i_ax]}': {worst:.1%} of the peak amplitude lies in the "
                    f"{boundary} layers outside ({lo + origin[i_ax]:.6g}, "
                    f"{hi + origin[i_ax]:.6g}) um")
        if faults:
            raise ValueError(
                "the default window reaches into absorbing boundary cells with "
                "significant amplitude, " + "; ".join(faults) + ": clipping it to "
                "the interior would cut the launch itself, not a tail. Enlarge the "
                "domain, re-center the beam, or set half_w_um / half_v_um to a "
                "window that fits.")
    if power_watts is not None:
        # fold-aware half-domain power (see _launched_power): a window
        # whose min face sits ON a symmetry plane (h_bc/v_bc from the shared
        # window rule) half-weights the on-plane node row, so power_watts is
        # the mode's exact power through the MODELED (half) domain.
        # A "periodic" window (the whole period of a plain periodic axis) has
        # no on-plane row to halve: every cell is a whole cell of the period.
        fold = (h_bc in ("pec", "pmc"), v_bc in ("pec", "pmc"))
        if graded_t:
            from .yee_mode import dual_spacings
            wh = dual_spacings(h_dq) if h_dq is not None else np.full(nh, dl)
            wv = dual_spacings(v_dq) if v_dq is not None else np.full(nv, dl)
            p0 = _launched_power(mode, dl, wh_um=wh, wv_um=wv, fold_low=fold,
                                 keep=keep)
        else:
            p0 = _launched_power(mode, dl, fold_low=fold, keep=keep)
        # p0 is the continuum Poynting integral of the profile. The sheet
        # stamps E on the J plane and H half a local cell away, so it launches
        # the wave at amplitude s in its OWN cell, and the power the Yee scheme
        # carries through every plane downstream is s^2 * p0 * cos(beta*half)
        # (the conserved staggered flux). Normalizing on that
        # product makes power_watts the watts a PowerMonitor and the port's
        # modal readout report, whatever cell either sits in; on p0 alone they
        # read cos(beta*half) * power_watts (0.96 W at 8 cells per wavelength).
        half = 0.5 * dl if halfcell_a is None else halfcell_a
        p0 *= math.cos(beta * half)
        s = math.sqrt(float(power_watts) / abs(p0)) if p0 else 1.0
        ex, ey, hx, hy = ex * s, ey * s, hx * s, hy * s
    dips: List[PointDipole] = []

    def add(comp_letter: str, kind: str, hpos: float, vpos: float, apos: float,
            val: complex, phi_extra: float, w_um: float, retained: bool):
        if not retained or val == 0:
            return
        c = [0.0, 0.0, 0.0]
        c[a], c[ih_ax], c[iv_ax] = apos, hpos, vpos
        for i in range(3):                              # interior only
            if not (min_lo[i] < c[i] < max_hi[i]):
                return
        phi = math.atan2(val.imag, val.real) + phi_extra
        dips.append(PointDipole(
            center_um=tuple(c), polarization=f"{kind}{comp_letter}",
            amplitude=float(abs(val) / (w_um * 1e-6)),
            source_time=source_time.model_copy(
                update={"phase": float(math.fmod(phi, 2.0 * math.pi))})))

    for iv in range(nv):
        for ih in range(nh):
            if not graded_t:
                hh, vv = h_lo + ih * dl, v_lo + iv * dl
                hm, vm = hh + 0.5 * dl, vv + 0.5 * dl
            else:
                hh, vv = float(h_node[ih]), float(v_node[iv])
                hm, vm = float(h_mid[ih]), float(v_mid[iv])
            # M sheet at the H Yee points of cell kM (a + dl/2), M = −n̂×E with
            # the frame-handedness sign `hand` (see above; +1 on x/z, −1 on y):
            #   M_h (H_h dipole at (h, v+1/2)) carries  +hand·E_v = ey
            #   M_v (H_v dipole at (h+1/2, v)) carries  -hand·E_h = ex, phase pi
            pair1 = bool(cutoff_masks[0][iv, ih])
            pair2 = bool(cutoff_masks[1][iv, ih])
            add(h_letter, "H", hh, vm, aM, hand * ey[iv, ih], 0.0, wM, pair2)
            add(v_letter, "H", hm, vv, aM, hand * ex[iv, ih], math.pi, wM, pair1)
            # J sheet at the E Yee points of plane k0:
            #   J_h (E_h dipole at (h+1/2, v)) carries  -H_v = hy, phase pi (+ph_j)
            #   J_v (E_v dipole at (h, v+1/2)) carries  +H_h = hx        (+ph_j)
            add(h_letter, "E", hm, vv, aJ, hy[iv, ih], math.pi + ph_j, wJ, pair1)
            add(v_letter, "E", hh, vm, aJ, hx[iv, ih], ph_j, wJ, pair2)
    _reject_absorber_overlap(sim, dips, a=a, ih_ax=ih_ax, iv_ax=iv_ax,
                             aJ=aJ, aM=aM)
    return dips
