"""Resolution of the declarative ``Simulation`` fields: ``wlens_um``, ``ports``
and ``source`` (design spec ``docs/superpowers/specs/2026-09-10-setup-api-
simplification-design.md`` §4.2 steps 2, 7 and 8).

A :class:`~photonhub.Simulation` given these fields solves each port's mode on
its own grid at construction and carries the launch and the readout planes as
ordinary sources and monitors, so the wire document is exactly what the
hand-built ``mode_launch`` / ``mode_monitor`` pipeline produces. The
``Simulation`` keeps what was resolved in a private :class:`Resolved` so
:meth:`photonhub.RunResult.transmission` can read a port back and so a
``with_*`` copy can resolve again on a new grid.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ._bounds import geometry_bounds_um
from .authoring import GaussianBeam, Port
from .source_time import GaussianPulse, _C0_M_PER_S
from .._compat import caller_stacklevel

#: The ``Simulation`` fields this module resolves; never on the wire.
DECLARATIVE_FIELDS = ("wlens_um", "wlen0_um", "ports", "source")
_AXES = ("x", "y", "z")
_SIGN = {"+": 1.0, "-": -1.0}


@dataclass(frozen=True)
class Fold:
    """The symmetry fold a fitted simulation applied (design spec §4.5): the
    folded axes (index to the mirror plane's position in the user's frame),
    the dropped ports and the kept ports they mirror, and the user monitors
    the fold clipped or that span the plane, returned unfolded."""

    planes: Dict[int, float]
    mirrored_ports: Dict[str, str]
    unfolded_monitors: Tuple[str, ...]


@dataclass(frozen=True)
class Resolved:
    """What a declarative simulation resolved to, kept beside the model."""

    freqs_hz: Tuple[float, ...]
    pulse: GaussianPulse
    ports: Tuple[Port, ...]
    port_monitors: Dict[str, Any]          # port name -> analysis ModeMonitor
    driven: Optional[str]                  # the driven port's name, if a port is driven
    user_sources: Tuple[Any, ...]          # what the caller passed as sources=
    user_monitors: Tuple[Any, ...]         # what the caller passed as monitors=
    # port name -> (half_w_um, half_v_um), the mode window each port was solved on
    port_windows_um: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def band(wlens_um: Union[float, Sequence[float]],
         wlen0_um: Optional[float]) -> Tuple[Tuple[float, ...], float, GaussianPulse]:
    """Readout frequencies (in the order the wavelengths were given, so a
    monitor's ``f`` axis follows the caller's list), the centre wavelength and
    the pulse for a wavelength list. One wavelength gets a short pulse of one
    tenth of its frequency; a band gets :meth:`GaussianPulse.for_band` centred
    on ``wlen0_um``, by default the mean of the extremes (the notebooks'
    rule)."""
    if isinstance(wlens_um, (int, float)):
        wl = (float(wlens_um),)
    else:
        wl = tuple(float(w) for w in wlens_um)
    if not wl or any(not w > 0.0 for w in wl):
        raise ValueError("wlens_um must be one or more positive wavelengths in microns")
    if len(set(wl)) != len(wl):
        raise ValueError("wlens_um must not repeat a wavelength")
    wlen0 = float(wlen0_um) if wlen0_um is not None else 0.5 * (min(wl) + max(wl))
    if not wlen0 > 0.0:
        raise ValueError("wlen0_um must be positive")
    freqs = tuple(_C0_M_PER_S / (w * 1e-6) for w in wl)
    if len(wl) == 1:
        f0 = _C0_M_PER_S / (wlen0 * 1e-6)
        pulse = GaussianPulse(freq0_hz=f0, fwidth_hz=0.1 * f0)
    else:
        pulse = GaussianPulse.for_band(wlens_um=(min(wl), max(wl)), wlen0_um=wlen0)
    return freqs, wlen0, pulse


def infer_out_direction(shell, port: Port) -> str:
    """The side of the port plane that faces the wall: where the plane sits
    relative to the structures' bounding-box centre along the port axis."""
    if port.out_direction is not None:
        return port.out_direction
    a = _AXES.index(port.axis)
    lo, hi = math.inf, -math.inf
    for s in shell.structures:
        b = geometry_bounds_um(s.geometry)[a]
        lo, hi = min(lo, b[0]), max(hi, b[1])
    if not (lo < hi):
        raise ValueError(
            f"port {port.name!r}: no structures to infer out_direction from; pass "
            "out_direction='+' or '-'")
    # a guide that runs through the wall is bounded by the domain
    lo, hi = max(lo, 0.0), min(hi, float(shell.size_um[a]))
    mid = 0.5 * (lo + hi)
    if abs(port.plane_um - mid) <= 1e-9:
        raise ValueError(
            f"port {port.name!r}: its plane sits on the device centre along {port.axis}; "
            "pass out_direction='+' or '-'")
    return "+" if port.plane_um > mid else "-"


# The margin beyond the core that a port window and the domain's clearance
# take by default, in wavelengths in the background: two thirds. The phase-5
# study swept it on a strip (T within 1e-4 of the wide-window value from 0.5 um
# at 1.55 um in oxide, 1e-5 from 0.7 um) and on the Y-junction of notebook 37
# (arm ratio flat to 0.004 dB at 0.54 um, 0.003 at 0.6, 0.0009 at 0.7).
MARGIN_WLENS = 2.0 / 3.0


def default_margin_um(wlen0_um: float, n_background: float) -> float:
    """The default margin beyond the core: ``MARGIN_WLENS`` wavelengths in the
    background (``wlen0_um / n_background``)."""
    return MARGIN_WLENS * float(wlen0_um) / float(n_background)


# How far the driven port's launch sits behind its readout plane, in wavelengths
# in the background. This is a LENGTH and never a cell count: NUMERICS §18.2a
# leaves the longitudinal E_z/H_z out of the injection (they re-form from the
# transverse fields downstream) and §18.3 bounds the residual by the mismatch
# between the FDE continuous mode and the FDTD discrete one, so what the launch
# injects that is NOT the discrete mode co-propagates and interferes with it at
# the readout plane, at a phase set by the distance in MICRONS (it is not a
# near field: measured, it does not decay over two wavelengths). A standoff
# counted in CELLS samples that interference at a mesh-dependent phase, the
# monitor's P_in drifts, and T = P_out / P_in reports a loss that GROWS under
# refinement — the one direction a convergence study cannot absorb.
# Measured on a lossless 400 x 220 nm silicon strip in oxide at 1550 nm: a
# ten-cell standoff read -0.0000259 dB at 12 cells per wavelength, +0.0000503 at
# 20, +0.0001390 at 28 and +0.0001591 at 32; the same 0.37163 um standoff held
# -0.0000259, -0.0000104 and -0.0000142 across 12, 20 and 28. See NUMERICS §18.6,
# which also records what a standoff does NOT fix.
#
# One third, not the window's two thirds. Sweeping the standoff on that strip at
# a FIXED mesh gives a bounded OSCILLATION, not a convergence: +0.00015 dB at
# 0.10 um, +0.00002 at 0.30, -0.00008 at 0.50, -0.00014 at 0.72, -0.00008 at
# 1.00, +0.00009 at 1.40 — period about one vacuum wavelength, amplitude about
# 1.4e-04 dB. So no standoff is "correct" and the value is chosen on three
# practical grounds instead: it is a LENGTH (the fix), it is sized to the
# wavelength so it travels between scenes, and at 0.358 um here it is close to
# what the old ten-cell rule already gave at the meshes in use (0.372 um at 12
# cells per wavelength, 0.279 at 16), so existing coarse-mesh work barely moves
# while the fine-mesh drift it was hiding goes away. Two thirds would sit on the
# measured trough of that oscillation.
LAUNCH_STANDOFF_WLENS = 1.0 / 3.0


def default_source_offset_um(wlen0_um: float, n_background: float) -> float:
    """The default standoff between the driven port's readout plane and its
    launch: ``LAUNCH_STANDOFF_WLENS`` wavelengths in the background. A port may
    override it with ``source_offset_um``. See NUMERICS §18.6."""
    return LAUNCH_STANDOFF_WLENS * float(wlen0_um) / float(n_background)


@dataclass(frozen=True)
class WindowPlan:
    """How a port's default mode window was sized (NUMERICS §18.8)."""

    half_w_um: float
    half_v_um: float
    n_clad_w: float                # the cladding the mode decays into across its width
    n_clad_v: float                # ... and across its thickness
    wlen_um: float                 # the longest wavelength the port reads
    method: str                    # "strip", "rib", "slab" or "margin"
    n_eff_estimate: Optional[float] = None       # the effective-index n_eff at wlen_um
    n_eff_ratio: Optional[float] = None          # that n_eff over the estimate at the band centre
    n_ref: float = 1.0             # the claddings' painted index, for the neighbour check

    @property
    def n_clad(self) -> float:
        return max(self.n_clad_w, self.n_clad_v)


class _Painter:
    """Which structure fills a point (paint order, last containing wins,
    NUMERICS §9), with each structure's bounding box as a cheap first test:
    a metasurface of thousands of posts is probed at a few dozen points per
    port."""

    def __init__(self, shell, items=None):
        self.shell = shell
        self.items = (items if items is not None
                      else [(geometry_bounds_um(st.geometry), st) for st in reversed(tuple(shell.structures))])

    def near(self, centre_um, reach_um: float) -> "_Painter":
        """The same painter over only the structures within ``reach_um`` of a
        point (a port's probes stay inside that box), paint order kept."""
        items = [(b, st) for b, st in self.items
                 if all(b[i][0] - reach_um <= float(centre_um[i]) <= b[i][1] + reach_um for i in range(3))]
        return _Painter(self.shell, items)

    def structure_at(self, point_um):
        import types

        import numpy as np

        from ..viz.eps import eps_at_points

        probe = types.SimpleNamespace(background=types.SimpleNamespace(permittivity=-1.0))
        hh = np.array([[float(point_um[1])]])
        vv = np.array([[float(point_um[2])]])
        for b, st in self.items:
            if not all(b[i][0] - 1e-9 <= float(point_um[i]) <= b[i][1] + 1e-9 for i in range(3)):
                continue
            if float(eps_at_points(probe, "x", hh, vv, float(point_um[0]), structures=[st])[0, 0]) != -1.0:
                return st
        return None

    def index(self, point_um, freq_hz: float) -> float:
        st = self.structure_at(point_um)
        eps = (float(st.medium.permittivity_at_hz(freq_hz)) if st is not None
               else float(self.shell.background.permittivity))
        return math.sqrt(max(eps, 1.0))

    def painted(self, point_um) -> float:
        """The index as the rasterizer paints it (the high-frequency permittivity)."""
        st = self.structure_at(point_um)
        eps = float(st.medium.permittivity) if st is not None else float(self.shell.background.permittivity)
        return math.sqrt(max(eps, 1.0))


def _port_axes(port: Port) -> Tuple[int, int]:
    """(width axis, thickness axis) of a port, as indices."""
    t = _AXES.index(port.thickness_axis)
    w = [i for i in range(3) if i not in (_AXES.index(port.axis), t)][0]
    return w, t


# How far a face probe walks out of the core looking for the cladding (a
# sloped sidewall widens the core at its foot), and its step.
_PROBE_REACH_UM = 0.6
_PROBE_STEP_UM = 0.01


def _probe_faces(painter: _Painter, port: Port, freq_hz: float):
    """The port's cross-section as the window rule reads it: ``n_core``, the
    claddings below and above the core, the cladding beside it, and the
    thickness of a slab beside it (0 when there is none, a rib otherwise).
    Each face is walked outward from the port's nominal core until the
    material changes, so a sloped sidewall finds its cladding; a side that
    stays in the core material for ``_PROBE_REACH_UM`` is a rib's slab."""
    w, t = _port_axes(port)
    centre = [float(c) for c in port.center_um]
    half_w, half_t = 0.5 * float(port.width_um), 0.5 * float(port.thickness_um)
    if port.medium is not None:
        n_core = math.sqrt(max(float(port.medium.permittivity_at_hz(freq_hz)), 1.0))
    else:
        n_core = painter.index(centre, freq_hz)

    def point(dw: float, dt: float):
        q = list(centre)
        q[w] += dw
        q[t] += dt
        return q

    def walk(axis_w: bool, sign: float, height: float):
        """(cladding index, point) out of one face, or (None, None) when the
        core material continues past the reach."""
        start = (half_w if axis_w else half_t) + _PROBE_STEP_UM
        n_steps = int(round(_PROBE_REACH_UM / _PROBE_STEP_UM))
        for k in range(n_steps + 1):
            d = sign * (start + k * _PROBE_STEP_UM)
            q = point(d, height) if axis_w else point(height, d)
            n = painter.index(q, freq_hz)
            if n < n_core - 1e-6:
                return n, q
        return None, None

    below, qb = walk(False, -1.0, 0.0)
    above, qa = walk(False, 1.0, 0.0)
    below = below if below is not None else n_core
    above = above if above is not None else n_core
    heights = (-(half_t - _PROBE_STEP_UM), 0.0, half_t - _PROBE_STEP_UM) \
        if half_t > _PROBE_STEP_UM else (0.0,)
    beside, slab_seen, refs = [], False, [q for q in (qb, qa) if q is not None]
    for sign in (-1.0, 1.0):
        for z in heights:
            n, q = walk(True, sign, z)
            if n is None:
                slab_seen = True
            else:
                beside.append(n)
                refs.append(q)
    t_slab = 0.0
    if slab_seen:
        # the slab's thickness, counted a probe reach beside the core
        n_t = 400
        dw = half_w + _PROBE_REACH_UM
        for sign in (-1.0, 1.0):
            inside = sum(1 for k in range(n_t)
                         if painter.index(point(sign * dw, -half_t + (k + 0.5) * 2 * half_t / n_t), freq_hz)
                         >= n_core - 1e-6)
            t_slab = max(t_slab, inside * 2 * half_t / n_t)
    n_beside = max(beside) if beside else max(below, above)
    n_ref = max([painter.painted(q) for q in refs] or [1.0])
    return n_core, below, above, n_beside, t_slab, n_ref


def plan_default_window(shell, port: Port, wlen0_um: float, wlen_max_um: float,
                        n_background: float, painter: Optional[_Painter] = None) -> WindowPlan:
    """The default mode window of ``port`` from the port-window rule (decision
    0018, NUMERICS §18.8): beyond each core face a pad of 1.2 times the -30 dB
    bound, ``1.2 * ln(1000) / (2 k0 sqrt(n_eff^2 - n_clad^2))``, rounded up to
    0.05 um, at the longest wavelength the port reads, with the cladding the
    mode decays into on that side.

    ``n_eff`` is an effective-index estimate for the port's mode:
    - a strip: the slab of its thickness, then the slab of its width;
    - a rib (the core material continues beside it): the same two steps with
      the slab beside it, of its measured thickness, as the side cladding;
    - a periodic axis across the guide (a quasi-2D slab): the one slab left.
    When none applies (a cladding at or above the core's index, a mode the
    estimate finds cut off) the window takes the older margin, two thirds of
    a wavelength in the background, and the port's own solve sizes it
    (``refine_window``)."""
    from .mode_window import MODE_WINDOW_FACTOR, _pad_um, _round_up, _slab_neff

    half_w, half_t = 0.5 * float(port.width_um), 0.5 * float(port.thickness_um)
    painter = (painter or _Painter(shell)).near(port.center_um,
                                                 max(half_w, half_t) + 2.0 * _PROBE_REACH_UM)
    wlen, wlen0 = float(wlen_max_um), float(wlen0_um)
    w_axis, t_axis = _port_axes(port)
    periodic = {i: getattr(getattr(shell, "boundaries", None), _AXES[i], None) in ("periodic", "bloch")
                for i in (w_axis, t_axis)}
    family, order = port.family_index()
    across = "TM" if family == "TE" else "TE"

    def estimate(wl: float):
        """(n_eff, lateral cladding, vertical cladding, method) at ``wl``."""
        n_core, below, above, beside, t_slab, n_ref = _probe_faces(painter, port, _C0_M_PER_S / (wl * 1e-6))
        vertical = max(below, above)
        if periodic[w_axis]:          # uniform across the width: the thickness slab
            return _slab_neff(2 * half_t, n_core, below, above, wl, family, order), vertical, vertical, "slab", n_ref
        if periodic[t_axis]:          # uniform through the thickness: the width slab
            return _slab_neff(2 * half_w, n_core, beside, beside, wl, across, order), beside, beside, "slab", n_ref
        if not n_core > max(vertical, beside if t_slab == 0.0 else vertical):
            raise ValueError("no cladding below the core's index")
        n_centre = _slab_neff(2 * half_t, n_core, below, above, wl, family, 0)
        if t_slab > 0.0:              # a rib: the slab beside it is the side cladding
            if t_slab >= 2 * half_t - 1e-6:
                raise ValueError("the core material runs through the whole thickness beside the port")
            n_side = _slab_neff(t_slab, n_core, below, above, wl, family, 0)
            return _slab_neff(2 * half_w, n_centre, n_side, n_side, wl, across, order), n_side, vertical, "rib", n_ref
        return _slab_neff(2 * half_w, n_centre, beside, beside, wl, across, order), beside, vertical, "strip", n_ref

    try:
        n_max, clad_w, clad_v, method, n_ref = estimate(wlen)
        n_0 = estimate(wlen0)[0]
    except ValueError:
        n_core, below, above, beside, _, n_ref = _probe_faces(painter, port, _C0_M_PER_S / (wlen * 1e-6))
        pad = default_margin_um(wlen0, n_background)
        return WindowPlan(half_w + pad, half_t + pad, beside, max(below, above), wlen, "margin", n_ref=n_ref)
    pad_w = _round_up(_pad_um(n_max, clad_w, wlen, MODE_WINDOW_FACTOR))
    pad_v = _round_up(_pad_um(n_max, clad_v, wlen, MODE_WINDOW_FACTOR))
    return WindowPlan(round(half_w + pad_w, 10), round(half_t + pad_v, 10), clad_w, clad_v, wlen, method,
                      n_max, n_max / n_0, n_ref)


def refine_window(plan: WindowPlan, port: Port, n_eff_solved: float) -> Tuple[float, float]:
    """The window the port's own mode solve asks for: the rule's pads from the
    solved ``n_eff`` (at the band centre, carried to the longest wavelength by
    the estimate's ratio when there is one), never smaller than the planned
    window. A solved ``n_eff`` at or below a cladding (a mode that is not
    guided there) leaves that side's planned window."""
    from .mode_window import MODE_WINDOW_FACTOR, _pad_um, _round_up

    n_eff = float(n_eff_solved) * (plan.n_eff_ratio if plan.n_eff_ratio is not None else 1.0)
    out = []
    for half, core, clad in ((plan.half_w_um, 0.5 * float(port.width_um), plan.n_clad_w),
                             (plan.half_v_um, 0.5 * float(port.thickness_um), plan.n_clad_v)):
        if n_eff > clad:
            half = max(half, round(core + _round_up(_pad_um(n_eff, clad, plan.wlen_um, MODE_WINDOW_FACTOR)), 10))
        out.append(half)
    return out[0], out[1]


def default_window(port: Port, wlen0_um: float, n_background: float, shell=None,
                   wlen_max_um: Optional[float] = None) -> Tuple[float, float]:
    """``(half_w_um, half_v_um)``: the port's ``window_um``, else the default.
    With ``shell`` (the simulation the port sits in) the default is the
    port-window rule's estimate (:func:`plan_default_window`; the Simulation
    then refines it from the port's own solve); without it, the core plus two
    thirds of a wavelength in the background, the older margin."""
    if port.window_um is not None:
        return port.window_um
    if port.thickness_um is None:
        raise ValueError(
            f"port {port.name!r}: pass thickness_um (the default mode window is the core "
            "plus a margin) or window_um=(half_w_um, half_v_um)")
    if shell is not None:
        plan = plan_default_window(shell, port, wlen0_um, wlen_max_um or wlen0_um, n_background)
        return plan.half_w_um, plan.half_v_um
    pad = default_margin_um(wlen0_um, n_background)
    return 0.5 * float(port.width_um) + pad, 0.5 * float(port.thickness_um) + pad


def neighbour_in_window(shell, port: Port, half_w_um: float, half_v_um: float,
                        plan: WindowPlan) -> Optional[Tuple[float, Tuple[float, float, float]]]:
    """A point of the window, outside the port's core, whose painted index is
    above the claddings' painted index: another guide or layer the window
    reaches, as ``(index, point)``, or None (decision 0017: a window should not
    reach a neighbouring guide). Sampled on a 25 x 25 grid at the port's plane;
    the slab of a rib or of a periodic axis is not a neighbour, so the scan is
    then across the thickness only."""
    import numpy as np

    from ..viz import _geometry as geom
    from ..viz.eps import eps_at_points

    w, t = _port_axes(port)
    a = _AXES.index(port.axis)
    boundaries = getattr(shell, "boundaries", None)
    periodic = {i: getattr(boundaries, _AXES[i], None) in ("periodic", "bloch") for i in (w, t)}
    # the slab of a rib or of a periodic axis is not a neighbour: scan across
    # the direction the guide is bounded in
    across = 0.0 if plan.method == "rib" or periodic[w] else half_w_um
    through = 0.0 if periodic[t] else half_v_um
    uu, vv = np.meshgrid(np.linspace(-across, across, 25), np.linspace(-through, through, 25))
    pts = np.zeros(uu.shape + (3,))
    pts[..., :] = port.center_um
    pts[..., w] += uu
    pts[..., t] += vv
    h_letter, _ = geom.in_plane_axes(port.axis)
    h_i = _AXES.index(h_letter)
    v_i = [i for i in range(3) if i not in (a, h_i)][0]
    eps = eps_at_points(shell, port.axis, pts[..., h_i], pts[..., v_i], float(port.center_um[a]))
    outside = (np.abs(uu) > 0.5 * float(port.width_um) + 0.02) | (np.abs(vv) > 0.5 * float(port.thickness_um) + 0.02)
    hot = outside & (np.sqrt(np.maximum(eps, 1.0)) > plan.n_ref + 0.01)
    if not hot.any():
        return None
    i = np.unravel_index(int(np.argmax(np.where(hot, eps, -np.inf))), eps.shape)
    return float(np.sqrt(eps[i])), tuple(float(c) for c in pts[i])


def clip_default_window(shell, port: Port, half_w_um: float, half_v_um: float) -> Tuple[float, float]:
    """A default window kept inside the boundary-layer-free interior of the
    shell, one background cell short of each open face (the window origin
    snaps down to a cell): what lies beyond the face the layers absorb, and a
    launch stamped there would only draw the in-the-layers warning. A
    symmetry plane on the low side is left to the solver's own clip. A port
    given ``window_um`` is never touched."""
    if port.window_um is not None:
        return half_w_um, half_v_um
    dl_bg = float(shell.grid.dl_um)
    h_c, v_c = port.hv_center_um()
    out = []
    for letter, centre, half in zip([a for a in _AXES if a != port.axis], (h_c, v_c), (half_w_um, half_v_um)):
        i = _AXES.index(letter)
        if getattr(shell.boundaries, letter) in ("periodic", "bloch"):
            # no layers to stay out of; the period bounds the window (a plain
            # periodic axis is solved as its whole period with the periodic
            # closure whatever this half is, yee_mode._periodic_window)
            out.append(float(half))
            continue
        lo, hi = shell._open_interval_um(i)
        room = hi - dl_bg - centre
        if shell.symmetry[i] == 0:
            room = min(room, centre - lo - dl_bg)
        clipped = min(float(half), room)
        core = 0.5 * float(port.width_um if letter == [a for a in _AXES if a != port.axis][0] else port.thickness_um or 0.0)
        if clipped <= core:
            raise ValueError(
                f"port {port.name!r}: the domain leaves no room for its mode window along {letter} "
                f"({room:.3g} um from the core's centre to the boundary layers); raise the clearance "
                "or give window_um")
        out.append(clipped)
    return out[0], out[1]


def _warn_user(message: str) -> None:
    """A UserWarning attributed to the caller's own line, outside this package
    and pydantic's validation frames."""
    warnings.warn(message, UserWarning, stacklevel=caller_stacklevel())


def port_cell_um(shell, port: Port, half_w_um: float, half_v_um: float) -> float:
    """The port's own cell: the finest primary spacing of the shell's grid
    across the port window at its plane, the uniform cell on a uniform grid.
    The mode is solved on it and the launch offset is counted in it (phase-5
    plan, refinements 1 and 2)."""
    grid = shell.grid
    dl = float(grid.dl_um)
    coords = getattr(grid, "coords", None)
    if coords is None:
        return dl
    import numpy as np
    h_c, v_c = port.hv_center_um()
    best = dl
    for letter, centre, half in zip([a for a in _AXES if a != port.axis], (h_c, v_c), (half_w_um, half_v_um)):
        q = getattr(coords, letter)
        if q is None:
            continue
        q = np.asarray(q, dtype=float)
        dq = np.diff(q)
        overlap = (q[:-1] < centre + half) & (q[1:] > centre - half)
        if overlap.any():
            best = min(best, float(dq[overlap].min()))
    return best


def paraxial_beam_source(shell, beam: GaussianBeam, wlen0: float, pulse, n_bg: float):
    """A ``paraxial=True`` beam ships as ONE profile source (NUMERICS §18): the
    Gaussian sampled over the plane once, the medium's index as its ``n_eff``,
    where the cell-by-cell launch is a pair of dipoles per cell of the window.
    ``power_watts`` sets the beam power through the plane in watts, as a
    :class:`~photonhub.PowerMonitor` reads it: the whole, unfolded device's
    power (NUMERICS §20.8). The amplitude is the whole Gaussian's, so a
    profile centred on k §20 symmetry planes carries ``power_watts / 2^k``
    into the modeled part with no factor of its own. ``n`` defaults to the
    background index."""
    from ..analysis.mode_devices import _TRANSVERSE, mode_source
    from ..analysis.mode_overlap import gaussian_mode
    t1, t2 = _TRANSVERSE[beam.axis]
    n = float(beam.n) if beam.n is not None else float(n_bg)
    pol = beam.polarization
    if pol is None or pol == "E" + t1:
        family = "TE"
    elif pol == "E" + t2:
        family = "TM"
    else:
        raise ValueError(
            f"GaussianBeam: polarization {pol!r} is not tangential to the {beam.axis}-normal "
            f"plane; use E{t1} or E{t2}")
    if beam.waist_um is not None:
        w = beam.waist_um
        w1, w2 = (float(w[0]), float(w[1])) if isinstance(w, (tuple, list)) else (float(w), float(w))
    else:
        m = beam.mfd_um
        w1, w2 = (0.5 * float(m[0]), 0.5 * float(m[1])) if isinstance(m, (tuple, list)) else (0.5 * float(m), 0.5 * float(m))
    window = (float(shell.size_um[_AXES.index(t1)]), float(shell.size_um[_AXES.index(t2)]))
    profile = gaussian_mode(wlen_um=float(wlen0), dl_um=float(shell.grid.dl_um), waist_um=(w1, w2), n=n,
                            polarization=family, window_um=window)
    # the peak field (V/m) of a paraxial Gaussian carrying power_watts through the plane in a
    # medium of index n: P = (n / 2 eta0) |E0|^2 pi w1 w2 / 2, the waists in metres
    eta0 = 376.730313668
    amplitude = math.sqrt(4.0 * float(beam.power_watts) * eta0 / (n * math.pi * (w1 * 1e-6) * (w2 * 1e-6)))
    if beam.center_um is not None:
        center = (float(beam.center_um[_AXES.index(t1)]), float(beam.center_um[_AXES.index(t2)]))
    else:
        # the domain centre, or the mirror plane (0) on a §20-folded axis: the
        # device's centre either way, where the unfolded simulation puts it
        from ..analysis.gaussian_beam import _default_center
        center = (_default_center(shell, t1), _default_center(shell, t2))
    return mode_source(shell, profile, axis=beam.axis, position_um=float(beam.position_um), source_time=pulse,
                       direction=beam.direction, amplitude=amplitude, center_um=center)


def resolve(shell, *, ports: Sequence[Port], source, wlens_um, wlen0_um,
            sources: Sequence[Any], monitors: Sequence[Any]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Resolved]:
    """Solve the ports on ``shell`` (a Simulation without declarative fields)
    and return ``(sources, monitors, resolved)`` for the final model: the
    launch first, then the caller's sources; the port readout planes first (in
    port order), then the caller's monitors."""
    from ..analysis.gaussian_beam import gaussian_beam_source
    from ..analysis.kfj_smoothing import solve_mode_on_cross_section
    from ..analysis.mode_devices import mode_launch, mode_monitor

    port_list = [p if isinstance(p, Port) else Port(**p) for p in (ports or ())]
    driven: Optional[str] = None
    beam: Optional[GaussianBeam] = None
    if isinstance(source, GaussianBeam):
        beam = source
    elif isinstance(source, Port):
        if source.name not in {p.name for p in port_list}:
            port_list.append(source)
        driven = source.name
    elif isinstance(source, str):
        if source not in {p.name for p in port_list}:
            raise ValueError(
                f"source={source!r} names no port; the ports are "
                f"{[p.name for p in port_list]}")
        driven = source
    elif source is not None:
        raise ValueError("source must be a port name, a Port or a GaussianBeam")
    if (port_list or beam is not None) and wlens_um is None:
        raise ValueError("ports= and a declared source= need wlens_um= (the readout wavelengths)")
    if wlens_um is None:
        return tuple(sources), tuple(monitors), Resolved((), None, (), {}, None, tuple(sources), tuple(monitors))  # type: ignore[arg-type]

    freqs, wlen0, pulse = band(wlens_um, wlen0_um)
    names = [p.name for p in port_list]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate port name(s): {dup}")
    monitor_names = [p.monitor_name for p in port_list]
    if len(set(monitor_names)) != len(monitor_names):
        raise ValueError(f"two ports share a monitor name: {sorted(monitor_names)}")
    user_names = {getattr(m, "name", None) for m in monitors}
    clash = sorted(set(monitor_names) & user_names)
    if clash:
        raise ValueError(
            f"monitor name(s) {clash} are used by both a port and a monitor (a dumped "
            "declarative Simulation already carries its port planes: rebuild it from the "
            "caller's monitors, or copy it with the with_* methods)")

    n_bg = math.sqrt(float(shell.background.permittivity))
    wlen_max = _C0_M_PER_S / min(freqs) * 1e6          # the longest wavelength a port reads
    painter = _Painter(shell)
    port_monitors: Dict[str, Any] = {}
    port_windows: Dict[str, Tuple[float, float]] = {}
    launch: Tuple[Any, ...] = ()
    resolved_ports = []
    for port in port_list:
        out_dir = infer_out_direction(shell, port)
        port = port.with_out_direction(out_dir)
        resolved_ports.append(port)
        family, index = port.family_index()
        h_c, v_c = port.hv_center_um()
        plan = None
        if port.window_um is None:
            default_window(port, wlen0, n_bg)                 # the thickness_um check
            plan = plan_default_window(shell, port, wlen0, wlen_max, n_bg, painter=painter)
            half_w, half_v = clip_default_window(shell, port, plan.half_w_um, plan.half_v_um)
        else:
            half_w, half_v = port.window_um

        def solve(hw: float, hv: float, port=port, family=family, index=index, h_c=h_c, v_c=v_c):
            # One engine-consistent Yee solve at the band centre, carrying its grid
            # placement, is the launch's mode; the readout re-solves it at every
            # frequency on first use (mode_monitor's default per-frequency bank).
            cell = port_cell_um(shell, port, hw, hv)
            dl = float(port.dl_um) if port.dl_um is not None else cell
            return solve_mode_on_cross_section(
                shell, port.axis, port.plane_um, wlen0, family, index,
                h_center_um=h_c, v_center_um=v_c, half_w_um=hw, half_v_um=hv,
                dl_um=dl, supersample=port.supersample, num_modes=port.num_modes)

        central = solve(half_w, half_v)
        if plan is not None:
            # the rule's pads from the port's own solved n_eff: a second solve
            # only when that asks for a wider window than the estimate gave
            wanted = refine_window(plan, port, float(central.n_eff))
            want = clip_default_window(shell, port, *wanted)
            if want[0] > half_w + 1e-9 or want[1] > half_v + 1e-9:
                half_w, half_v = want
                central = solve(half_w, half_v)
            if half_w < wanted[0] - 1e-9 or half_v < wanted[1] - 1e-9:
                _warn_user(
                    f"port {port.name!r}: the domain clips its default mode window from "
                    f"({wanted[0]:.3g}, {wanted[1]:.3g}) to ({half_w:.3g}, {half_v:.3g}) um, so the mode is "
                    "no longer 30 dB down at the window's edge; widen the domain across the guide or "
                    "give window_um")
            near = neighbour_in_window(shell, port, half_w, half_v, plan)
            if near is not None:
                origin = tuple(float(o) for o in getattr(shell, "origin_um", (0.0, 0.0, 0.0)))
                at = tuple(round(c + o, 3) for c, o in zip(near[1], origin))
                _warn_user(
                    f"port {port.name!r}: its mode window reaches another guide or layer (index "
                    f"{near[0]:.3g} at {at} um); place the port where the guides are more than two pads "
                    "apart, or give window_um enclosing both")
        port_windows[port.name] = (float(half_w), float(half_v))
        # The driven port's plane is read in the launch direction (the power it
        # sends into the device, the reference of every transmission); every
        # other port's plane is read outward (the power leaving through it).
        read_dir = port.in_direction if driven == port.name else out_dir
        port_monitors[port.name] = mode_monitor(
            shell, central, axis=port.axis, position_um=port.plane_um, freqs_hz=freqs,
            name=port.monitor_name, direction=read_dir, thickness_axis=port.thickness_axis)
        if driven == port.name:
            offset = (float(port.source_offset_um) if port.source_offset_um is not None
                      else default_source_offset_um(wlen0, n_bg))
            src_pos = port.plane_um + _SIGN[out_dir] * offset
            length = float(shell.size_um[_AXES.index(port.axis)])
            if not (0.0 < src_pos < length):
                raise ValueError(
                    f"port {port.name!r}: launch plane {port.axis}={src_pos:.4g} um falls "
                    f"outside the domain (0, {length:.4g}); reduce source_offset_um or move "
                    "the port inward")
            # The launch injects the band-centre mode as one sheet (a sheet per
            # frequency would multiply the dipole count by the wavelength count);
            # the readout above projects every frequency onto its own mode.
            launch = tuple(mode_launch(
                shell, central, axis=port.axis, position_um=src_pos, source_time=pulse,
                direction=port.in_direction))
    if beam is not None and beam.paraxial:
        launch = (paraxial_beam_source(shell, beam, wlen0, pulse, n_bg),)
    elif beam is not None:
        launch = tuple(gaussian_beam_source(
            shell, axis=beam.axis, position_um=float(beam.position_um), source_time=pulse,
            waist_um=beam.waist_um, mfd_um=beam.mfd_um, direction=beam.direction,
            power_watts=beam.power_watts, center_um=beam.hv_center_um(), n=beam.n,
            polarization=beam.polarization, pol_angle_rad=beam.pol_angle_rad,
            waist_distance_um=beam.waist_distance_um, angle_theta_rad=beam.angle_theta_rad,
            angle_phi_rad=beam.angle_phi_rad, half_w_um=beam.half_w_um,
            half_v_um=beam.half_v_um, window_sigmas=beam.window_sigmas,
            amplitude_threshold=beam.amplitude_threshold))
    out_sources = launch + tuple(sources)
    out_monitors = tuple(port_monitors[p.name].field_monitor for p in resolved_ports) + tuple(monitors)
    resolved = Resolved(freqs, pulse, tuple(resolved_ports), port_monitors, driven,
                        tuple(sources), tuple(monitors), port_windows)
    return out_sources, out_monitors, resolved
