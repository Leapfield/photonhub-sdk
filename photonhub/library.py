"""Parametric PIC component library — the "~one line per component" sugar
layer on top of the geometry primitives in :mod:`photonhub.components.structures`.

Each builder returns a small :class:`Component` bundling the emitted
:class:`~photonhub.Structure` geometry with its :class:`Port` s — the planes
where a mode source / mode monitor will later attach. Builders are
*position-agnostic*: they take a ``center_um`` and place geometry relative to
it, so the caller positions the device inside their (corner-origin) domain.

Coordinate convention (defaults): in-plane propagation runs along ``axis``
(default ``"x"``), the slab thickness runs along ``thickness_axis`` (default
``"z"``), and the remaining in-plane axis carries the waveguide width. This is
the standard SOI strip-waveguide layout.

>>> import photonhub as ph
>>> from photonhub.library import straight, ring
>>> wg = straight(length_um=10.0)                 # one Box + 2 ports
>>> res = ring(radius_um=5.0, gap_um=0.2)         # ring Cylinder + bus Box + 2 ports
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from .components.base import AxisName, Vec3Um
from .components.structures import Box, Cylinder, Medium, PolySlab, Structure

__all__ = [
    "Port",
    "Component",
    "SILICON",
    "straight",
    "bend",
    "taper",
    "cosine_taper",
    "crossing",
    "cosine_taper_crossing",
    "coupler",
    "ring",
    "bragg_grating",
    "y_branch",
]

# Default material/geometry for the SOI strip platform (NUMERICS.md / Phase-2
# MVP). n ~= 3.5 silicon core, 220 nm slab, 450 nm strip width.
# NOTE: n = 3.5 is a generic library default; the GDS benchmark suite
# (benchmarks/gds/) uses the convention n = 3.478 (permittivity 3.478**2,
# Si at 1.55 um) — pass an explicit medium when matching those results.
SILICON = Medium(permittivity=12.25)
DEFAULT_WIDTH_UM = 0.45
DEFAULT_THICKNESS_UM = 0.22

_AXES: Tuple[AxisName, AxisName, AxisName] = ("x", "y", "z")
_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Port:
    """A plane where a mode source / mode monitor attaches.

    ``axis`` is the local propagation axis at the port; ``width_um`` is the
    waveguide width there (sizes the transverse mode window).
    """

    name: str
    center_um: Tuple[float, float, float]
    axis: str
    width_um: float


@dataclass(frozen=True)
class Component:
    """The geometry a builder emits plus the ports it exposes."""

    structures: Tuple[Structure, ...]
    ports: Tuple[Port, ...]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _third_axis(a: AxisName, b: AxisName) -> AxisName:
    """The axis that is neither ``a`` nor ``b``."""
    for ax in _AXES:
        if ax != a and ax != b:
            return ax
    raise ValueError(f"could not find third axis distinct from {a!r}, {b!r}")


def _vec(center: Vec3Um, **offsets: float) -> Tuple[float, float, float]:
    """``center`` shifted by per-axis offsets given as ``x=``/``y=``/``z=``."""
    out = list(center)
    for ax, d in offsets.items():
        out[_INDEX[ax]] += d
    return (out[0], out[1], out[2])


def _planar_axes(
    prop_axis: AxisName, thickness_axis: AxisName
) -> Tuple[AxisName, AxisName]:
    """Return ``(width_axis, thickness_axis)`` validating they are distinct."""
    if prop_axis == thickness_axis:
        raise ValueError(
            f"propagation axis {prop_axis!r} must differ from thickness axis "
            f"{thickness_axis!r}"
        )
    width_axis = _third_axis(prop_axis, thickness_axis)
    return width_axis, thickness_axis


def _slab_box(
    *,
    center: Vec3Um,
    prop_axis: AxisName,
    width_axis: AxisName,
    thickness_axis: AxisName,
    length_um: float,
    width_um: float,
    thickness_um: float,
    medium: Medium,
) -> Structure:
    """A strip-waveguide ``Box`` of the given length/width/thickness centered
    on ``center`` with the stated axis roles."""
    size = [0.0, 0.0, 0.0]
    size[_INDEX[prop_axis]] = length_um
    size[_INDEX[width_axis]] = width_um
    size[_INDEX[thickness_axis]] = thickness_um
    return Structure(
        geometry=Box(center_um=center, size_um=(size[0], size[1], size[2])),
        medium=medium,
    )


# ---------------------------------------------------------------------------
# 1. straight
# ---------------------------------------------------------------------------


def straight(
    length_um: float,
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis: AxisName = "x",
    thickness_axis: AxisName = "z",
) -> Component:
    """A straight strip waveguide: one ``Box`` with input/output ports at the
    two ends along ``axis``."""
    width_axis, thickness_axis = _planar_axes(axis, thickness_axis)
    structure = _slab_box(
        center=center_um,
        prop_axis=axis,
        width_axis=width_axis,
        thickness_axis=thickness_axis,
        length_um=length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    half = length_um / 2.0
    ports = (
        Port("in", _vec(center_um, **{axis: -half}), axis, width_um),
        Port("out", _vec(center_um, **{axis: +half}), axis, width_um),
    )
    return Component(structures=(structure,), ports=ports)


# ---------------------------------------------------------------------------
# 2. bend (90-degree annular sector)
# ---------------------------------------------------------------------------


def bend(
    radius_um: float,
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    thickness_axis: AxisName = "z",
) -> Component:
    """A 90-degree waveguide bend: an annular-sector ``Cylinder`` extruded
    along ``thickness_axis`` (the slab normal), with ``inner_radius`` /
    ``radius`` straddling ``radius_um +/- width_um/2`` and a pi/2 sweep.

    ``center_um`` is the bend's *center of curvature*. The cylinder sweeps the
    first quadrant in the transverse (u, v) plane, so the two arc ends point
    along the two in-plane axes. With the default ``thickness_axis="z"`` the
    transverse plane is (x, y): the start end (angle 0) sits at +u and
    propagates along that axis; the stop end (angle pi/2) sits at +v.
    """
    u_axis, v_axis = _bend_plane_axes(thickness_axis)
    geometry = Cylinder(
        axis=thickness_axis,
        center_um=center_um,
        radius_um=radius_um + width_um / 2.0,
        inner_radius_um=radius_um - width_um / 2.0,
        length_um=thickness_um,
        angle_start=0.0,
        angle_stop=math.pi / 2.0,
    )
    structure = Structure(geometry=geometry, medium=medium)
    # Arc endpoints on the centerline radius. Start end faces +u, propagating
    # along u_axis; stop end faces +v, propagating along v_axis.
    p_start = _vec(center_um, **{u_axis: radius_um})
    p_stop = _vec(center_um, **{v_axis: radius_um})
    ports = (
        Port("in", p_start, u_axis, width_um),
        Port("out", p_stop, v_axis, width_um),
    )
    return Component(structures=(structure,), ports=ports)


def _bend_plane_axes(thickness_axis: AxisName) -> Tuple[AxisName, AxisName]:
    """The (u, v) transverse axes for a cylinder extruded along
    ``thickness_axis`` (u = lower-indexed of the remaining two)."""
    remaining = [ax for ax in _AXES if ax != thickness_axis]
    return remaining[0], remaining[1]


# ---------------------------------------------------------------------------
# 3. taper (trapezoidal PolySlab)
# ---------------------------------------------------------------------------


def taper(
    length_um: float,
    width1_um: float,
    width2_um: float,
    *,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis: AxisName = "x",
    thickness_axis: AxisName = "z",
) -> Component:
    """A linear width taper: a trapezoidal ``PolySlab`` (4 vertices) whose
    cross-section width grows ``width1_um`` -> ``width2_um`` along ``axis``.
    Ports at each end with the local widths."""
    width_axis, thickness_axis = _planar_axes(axis, thickness_axis)
    # PolySlab vertices are (u, v) in the two transverse axes of the EXTRUSION
    # axis (= thickness_axis): u = lower-indexed, v = higher-indexed. Here the
    # in-plane polygon lives in (prop, width) so we map those two roles onto
    # (u, v) by axis index, then emit vertices in increasing-index order.
    poly_axes = [ax for ax in _AXES if ax != thickness_axis]  # the two (u, v)
    half_len = length_um / 2.0
    # offsets of each role relative to center, keyed by axis
    half1 = width1_um / 2.0
    half2 = width2_um / 2.0
    # Build the four corners in (prop, width) space then project to (u, v).
    corners_pw = (
        (-half_len, -half1),  # input bottom
        (+half_len, -half2),  # output bottom
        (+half_len, +half2),  # output top
        (-half_len, +half1),  # input top
    )
    pw_to_axis = {axis: 0, width_axis: 1}
    vertices = []
    for corner in corners_pw:
        u = corner[pw_to_axis[poly_axes[0]]]
        v = corner[pw_to_axis[poly_axes[1]]]
        # add the center offset for these in-plane axes
        u += center_um[_INDEX[poly_axes[0]]]
        v += center_um[_INDEX[poly_axes[1]]]
        vertices.append((u, v))
    t_center = center_um[_INDEX[thickness_axis]]
    slab_bounds = (t_center - thickness_um / 2.0, t_center + thickness_um / 2.0)
    geometry = PolySlab(
        axis=thickness_axis,
        vertices_um=tuple(vertices),
        slab_bounds_um=slab_bounds,
    )
    structure = Structure(geometry=geometry, medium=medium)
    ports = (
        Port("in", _vec(center_um, **{axis: -half_len}), axis, width1_um),
        Port("out", _vec(center_um, **{axis: +half_len}), axis, width2_um),
    )
    return Component(structures=(structure,), ports=ports)


# ---------------------------------------------------------------------------
# 3b. cosine taper (curved-sidewall "beam shaping" PolySlab)
# ---------------------------------------------------------------------------


def _cosine_lens_half(s: float, length: float, w1: float, w2: float, w_m: float) -> float:
    """Half-width of a convex cosine ("beam shaping") taper at arc position ``s``
    (0 at the ``w1`` end, ``length`` at the ``w2`` end).

    The FULL width follows ``W(x) = W_m*cos(pi*x/(2*L0))`` (Chandran et al., Opt.
    Lett. 45, 6230 (2020), Eq. 1), with the peak ``W_m`` reached *inside* the
    taper — a convex lens that focuses the mode — and ``w1``/``w2`` at the two
    ends. ``L0`` is set from the taper length via Eq. 2,
    ``L_t = (2*L0/pi)*(arccos(w1/W_m) + arccos(w2/W_m))``. With ``W_m`` equal to
    the wider end the peak slides to that end and the shape reduces to a
    monotonic cosine taper (the default)."""
    if w_m <= 0.0:
        return 0.0
    theta1 = math.acos(max(-1.0, min(1.0, w1 / w_m)))
    theta2 = math.acos(max(-1.0, min(1.0, w2 / w_m)))
    denom = theta1 + theta2
    if denom < 1e-12:  # w1 == w2 == w_m -> straight
        return 0.5 * w1
    l0 = math.pi * length / (2.0 * denom)
    x1 = -(2.0 * l0 / math.pi) * theta1  # peak (W_m) sits at x = 0
    x = x1 + s
    return 0.5 * w_m * math.cos(math.pi * x / (2.0 * l0))


def cosine_taper(
    length_um: float,
    width1_um: float,
    width2_um: float,
    *,
    peak_width_um: float | None = None,
    n_points: int = 48,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis: AxisName = "x",
    thickness_axis: AxisName = "z",
) -> Component:
    """A cosine ("beam shaping") width taper: a curved-sidewall ``PolySlab``
    whose full width follows ``W(x) = W_m*cos(pi*x/(2*L0))`` from ``width1_um`` to
    ``width2_um`` along ``axis`` (Chandran et al., Opt. Lett. 45, 6230 (2020)).
    ``peak_width_um`` (``W_m``) is the maximum width reached inside the taper —
    the convex lens that focuses the beam; it must be >= both end widths and
    defaults to the wider end (a monotonic cosine taper). The sidewall is
    discretized into ``n_points`` samples per edge; ports carry the end widths."""
    width_axis, thickness_axis = _planar_axes(axis, thickness_axis)
    half_len = length_um / 2.0
    w_m = peak_width_um if peak_width_um is not None else max(width1_um, width2_um)
    if w_m < max(width1_um, width2_um) - 1e-12:
        raise ValueError(
            f"peak_width_um ({w_m}) must be >= both end widths "
            f"({width1_um}, {width2_um})"
        )

    poly_axes = [ax for ax in _AXES if ax != thickness_axis]  # (u, v)
    pw_to_axis = {axis: 0, width_axis: 1}

    ts = [i / (n_points - 1) for i in range(n_points)]  # 0 at width1 end -> 1

    def half_at(t: float) -> float:
        return _cosine_lens_half(t * length_um, length_um, width1_um, width2_um, w_m)

    # bottom edge left->right (increasing prop), then top edge right->left: CCW.
    corners_pw = [(-half_len + t * length_um, -half_at(t)) for t in ts]
    corners_pw += [(-half_len + t * length_um, +half_at(t)) for t in reversed(ts)]

    vertices = []
    for prop_off, width_off in corners_pw:
        role = {axis: prop_off, width_axis: width_off}
        u = role[poly_axes[0]] + center_um[_INDEX[poly_axes[0]]]
        v = role[poly_axes[1]] + center_um[_INDEX[poly_axes[1]]]
        vertices.append((u, v))

    t_center = center_um[_INDEX[thickness_axis]]
    slab_bounds = (t_center - thickness_um / 2.0, t_center + thickness_um / 2.0)
    geometry = PolySlab(
        axis=thickness_axis,
        vertices_um=tuple(vertices),
        slab_bounds_um=slab_bounds,
    )
    structure = Structure(geometry=geometry, medium=medium)
    ports = (
        Port("in", _vec(center_um, **{axis: -half_len}), axis, width1_um),
        Port("out", _vec(center_um, **{axis: +half_len}), axis, width2_um),
    )
    return Component(structures=(structure,), ports=ports)


# ---------------------------------------------------------------------------
# 4. crossing (two boxes at 90 degrees)
# ---------------------------------------------------------------------------


def crossing(
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    arm_length_um: float = 3.0,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    thickness_axis: AxisName = "z",
) -> Component:
    """A waveguide crossing: two ``Box`` waveguides crossed at 90 degrees in
    the slab plane. Four ports, one per arm end."""
    a_axis, b_axis = _bend_plane_axes(thickness_axis)  # the two in-plane axes
    arm_a = _slab_box(
        center=center_um,
        prop_axis=a_axis,
        width_axis=b_axis,
        thickness_axis=thickness_axis,
        length_um=arm_length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    arm_b = _slab_box(
        center=center_um,
        prop_axis=b_axis,
        width_axis=a_axis,
        thickness_axis=thickness_axis,
        length_um=arm_length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    half = arm_length_um / 2.0
    ports = (
        Port(f"{a_axis}-", _vec(center_um, **{a_axis: -half}), a_axis, width_um),
        Port(f"{a_axis}+", _vec(center_um, **{a_axis: +half}), a_axis, width_um),
        Port(f"{b_axis}-", _vec(center_um, **{b_axis: -half}), b_axis, width_um),
        Port(f"{b_axis}+", _vec(center_um, **{b_axis: +half}), b_axis, width_um),
    )
    return Component(structures=(arm_a, arm_b), ports=ports)


# ---------------------------------------------------------------------------
# 4b. cosine-taper crossing (beam-shaped ultra-compact crossing)
# ---------------------------------------------------------------------------


def cosine_taper_crossing(
    *,
    wg_width_um: float,
    junction_width_um: float,
    peak_width_um: float,
    taper_length_um: float,
    arm_length_um: float,
    n_points: int = 48,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    thickness_axis: AxisName = "z",
) -> Component:
    """A beam-shaped ultra-compact waveguide crossing (Chandran et al., Opt.
    Lett. 45, 6230 (2020)). Four convex cosine tapers connect the single-mode
    routing guide (``wg_width_um`` = W_in) to a solid central junction of side
    ``junction_width_um`` (= W_out); each taper bulges to a peak width
    ``peak_width_um`` (= W_m) in its middle — the lens that focuses the beam
    across the intersection so it couples to the through arm instead of
    scattering into the cross arms.

    Each arm runs center -> port along one in-plane axis: the shared central
    ``Box`` junction (W_out square), a cosine taper of ``taper_length_um`` (= L_t)
    from ``junction_width_um`` at the junction to ``wg_width_um`` at the routing
    end (peaking at ``peak_width_um`` inside), then a straight routing stub out to
    the port at ``arm_length_um`` from center. Four ports, each ``wg_width_um``
    wide. ``arm_length_um`` must be >= ``junction_width_um/2 + taper_length_um``.
    The paper's footprint is ``junction_width_um + 2*taper_length_um``.
    """
    inner = junction_width_um / 2.0 + taper_length_um  # end of the shaped region
    if arm_length_um < inner - 1e-9:
        raise ValueError(
            f"arm_length_um ({arm_length_um}) must be >= junction_width_um/2 + "
            f"taper_length_um ({inner})"
        )
    a_axis, b_axis = _bend_plane_axes(thickness_axis)  # two in-plane axes
    half_j = junction_width_um / 2.0

    structures: list[Structure] = [
        _slab_box(
            center=center_um,
            prop_axis=a_axis,
            width_axis=b_axis,
            thickness_axis=thickness_axis,
            length_um=junction_width_um,
            width_um=junction_width_um,
            thickness_um=thickness_um,
            medium=medium,
        )
    ]
    ports: list[Port] = []
    for prop_axis in (a_axis, b_axis):
        for sign in (-1.0, +1.0):
            taper_center = _vec(
                center_um, **{prop_axis: sign * (half_j + taper_length_um / 2.0)}
            )
            # width1 is the -half_len (lower-coordinate) end of the taper.
            if sign > 0:  # inner (junction) end at -half_len
                w1, w2 = junction_width_um, wg_width_um
            else:  # inner (junction) end at +half_len
                w1, w2 = wg_width_um, junction_width_um
            arm = cosine_taper(
                taper_length_um,
                w1,
                w2,
                peak_width_um=peak_width_um,
                n_points=n_points,
                thickness_um=thickness_um,
                medium=medium,
                center_um=taper_center,
                axis=prop_axis,
                thickness_axis=thickness_axis,
            )
            structures.extend(arm.structures)

            stub_len = arm_length_um - inner
            if stub_len > 1e-9:
                stub_center = _vec(
                    center_um, **{prop_axis: sign * (inner + stub_len / 2.0)}
                )
                structures.append(
                    _slab_box(
                        center=stub_center,
                        prop_axis=prop_axis,
                        width_axis=_third_axis(prop_axis, thickness_axis),
                        thickness_axis=thickness_axis,
                        length_um=stub_len,
                        width_um=wg_width_um,
                        thickness_um=thickness_um,
                        medium=medium,
                    )
                )
            port_name = f"{prop_axis}{'+' if sign > 0 else '-'}"
            port_center = _vec(center_um, **{prop_axis: sign * arm_length_um})
            ports.append(Port(port_name, port_center, prop_axis, wg_width_um))
    return Component(structures=tuple(structures), ports=tuple(ports))


# ---------------------------------------------------------------------------
# 5. coupler (two parallel straights, edge-to-edge gap)
# ---------------------------------------------------------------------------


def coupler(
    length_um: float,
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    gap_um: float = 0.2,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis: AxisName = "x",
    thickness_axis: AxisName = "z",
) -> Component:
    """A directional coupler: two parallel straight waveguides separated
    edge-to-edge by ``gap_um`` (along the in-plane width axis). Four ports —
    two per guide at the ends."""
    width_axis, thickness_axis = _planar_axes(axis, thickness_axis)
    # center-to-center spacing = one width + the edge-to-edge gap
    offset = (width_um + gap_um) / 2.0
    top_center = _vec(center_um, **{width_axis: +offset})
    bot_center = _vec(center_um, **{width_axis: -offset})
    top = _slab_box(
        center=top_center,
        prop_axis=axis,
        width_axis=width_axis,
        thickness_axis=thickness_axis,
        length_um=length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    bot = _slab_box(
        center=bot_center,
        prop_axis=axis,
        width_axis=width_axis,
        thickness_axis=thickness_axis,
        length_um=length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    half = length_um / 2.0
    ports = (
        Port("top_in", _vec(top_center, **{axis: -half}), axis, width_um),
        Port("top_out", _vec(top_center, **{axis: +half}), axis, width_um),
        Port("bot_in", _vec(bot_center, **{axis: -half}), axis, width_um),
        Port("bot_out", _vec(bot_center, **{axis: +half}), axis, width_um),
    )
    return Component(structures=(top, bot), ports=ports)


# ---------------------------------------------------------------------------
# 6. ring (full-sweep annular Cylinder + bus Box)
# ---------------------------------------------------------------------------


def ring(
    radius_um: float,
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    gap_um: float = 0.2,
    bus_width_um: float | None = None,
    bus_length_um: float | None = None,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis_normal: AxisName = "z",
) -> Component:
    """An all-pass ring resonator: a full-sweep annular ``Cylinder`` (inner =
    ``radius - width/2``, outer = ``radius + width/2``) plus a straight bus
    ``Box`` one ``gap_um`` away (edge-to-edge) from the ring's outer edge.

    ``axis_normal`` is the slab normal (the ring/cylinder extrusion axis).
    The bus runs along the lower-indexed in-plane axis; it is offset along the
    higher-indexed in-plane axis. Two ports on the bus (input + through)."""
    if bus_width_um is None:
        bus_width_um = width_um
    bus_axis, offset_axis = _bend_plane_axes(axis_normal)
    outer_r = radius_um + width_um / 2.0
    # ring sits centered on center_um in the slab plane
    ring_geom = Cylinder(
        axis=axis_normal,
        center_um=center_um,
        radius_um=outer_r,
        inner_radius_um=radius_um - width_um / 2.0,
        length_um=thickness_um,
        angle_start=0.0,
        angle_stop=2.0 * math.pi,
    )
    ring_structure = Structure(geometry=ring_geom, medium=medium)

    # Bus: parallel to bus_axis, offset along offset_axis so its inner edge is
    # gap_um from the ring's outer edge.
    bus_offset = outer_r + gap_um + bus_width_um / 2.0
    if bus_length_um is None:
        # span the ring diameter by default
        bus_length_um = 2.0 * outer_r
    bus_center = _vec(center_um, **{offset_axis: bus_offset})
    bus_structure = _slab_box(
        center=bus_center,
        prop_axis=bus_axis,
        width_axis=offset_axis,
        thickness_axis=axis_normal,
        length_um=bus_length_um,
        width_um=bus_width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    half = bus_length_um / 2.0
    ports = (
        Port("in", _vec(bus_center, **{bus_axis: -half}), bus_axis, bus_width_um),
        Port(
            "through",
            _vec(bus_center, **{bus_axis: +half}),
            bus_axis,
            bus_width_um,
        ),
    )
    return Component(structures=(ring_structure, bus_structure), ports=ports)


# ---------------------------------------------------------------------------
# 7. Bragg grating (sidewall-corrugated straight waveguide)
# ---------------------------------------------------------------------------


def bragg_grating(
    *,
    wg_width_um: float,
    corrugation_um: float,
    period_um: float,
    n_periods: int,
    arm_length_um: float,
    duty: float = 0.5,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis: AxisName = "x",
    thickness_axis: AxisName = "z",
) -> Component:
    """A sidewall-corrugated waveguide Bragg grating (Wang et al., Opt. Express
    20, 15547 (2012), strip variant). A straight strip of width ``wg_width_um``
    whose two sidewalls corrugate SYMMETRICALLY (equal ± on each side, so the
    average width — hence the average effective index — is constant): over a
    ``period_um`` the full width alternates between ``wg_width + 2*corrugation``
    (wide, a ``duty`` fraction of the period) and ``wg_width - 2*corrugation``
    (narrow). ``n_periods`` teeth open a photonic stopband at the Bragg
    wavelength ``λ_B ≈ 2*n_eff*period``; ``corrugation_um`` (the per-side tooth
    amplitude) sets the coupling coefficient κ, hence the stopband width. Plain
    routing stubs of the nominal ``wg_width`` run out to the two ports at
    ``±arm_length_um`` (``in`` at −axis, ``through`` at +axis)."""
    width_axis, thickness_axis = _planar_axes(axis, thickness_axis)
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")
    if corrugation_um * 2.0 >= wg_width_um:
        raise ValueError(
            f"corrugation_um ({corrugation_um}) too large: narrow segment width "
            f"{wg_width_um - 2*corrugation_um} would be non-positive")
    grating_len = n_periods * period_um
    half_g = grating_len / 2.0
    wide_w = wg_width_um + 2.0 * corrugation_um
    narrow_w = wg_width_um - 2.0 * corrugation_um
    wide_len = duty * period_um
    narrow_len = period_um - wide_len

    structures: list[Structure] = []
    # Each period = one wide tooth (centred at the period start) then a narrow
    # gap. Segments are axis-aligned boxes, so subpixel staircasing is exact.
    for p in range(n_periods):
        p0 = -half_g + p * period_um
        wide_center = _vec(center_um, **{axis: p0 + wide_len / 2.0})
        structures.append(_slab_box(
            center=wide_center, prop_axis=axis, width_axis=width_axis,
            thickness_axis=thickness_axis, length_um=wide_len, width_um=wide_w,
            thickness_um=thickness_um, medium=medium))
        narrow_center = _vec(center_um, **{axis: p0 + wide_len + narrow_len / 2.0})
        structures.append(_slab_box(
            center=narrow_center, prop_axis=axis, width_axis=width_axis,
            thickness_axis=thickness_axis, length_um=narrow_len, width_um=narrow_w,
            thickness_um=thickness_um, medium=medium))

    # Routing stubs (nominal width) from each grating end out to the ports.
    for sign in (-1.0, +1.0):
        stub_len = arm_length_um - half_g
        if stub_len > 1e-9:
            stub_center = _vec(center_um, **{axis: sign * (half_g + stub_len / 2.0)})
            structures.append(_slab_box(
                center=stub_center, prop_axis=axis, width_axis=width_axis,
                thickness_axis=thickness_axis, length_um=stub_len,
                width_um=wg_width_um, thickness_um=thickness_um, medium=medium))
    ports = (
        Port("in", _vec(center_um, **{axis: -arm_length_um}), axis, wg_width_um),
        Port("through", _vec(center_um, **{axis: +arm_length_um}), axis, wg_width_um),
    )
    return Component(structures=tuple(structures), ports=ports)


# ---------------------------------------------------------------------------
# 8. Y-branch (asymmetric S-curve 1x2 splitter, arbitrary power ratio)
# ---------------------------------------------------------------------------


def _sbend_phi(rho1: float, rho2: float, straight_um: float, y_target: float) -> float:
    """Half-turn angle φ of a two-tangent-arc S-bend (radii ``rho1`` then
    ``rho2``, with an optional ``straight_um`` tangent section between them) that
    starts and ends horizontal and produces a net lateral offset ``y_target``:
    ``(rho1+rho2)(1-cos φ) + straight*sin φ = y_target``. Solved by bisection on
    (0, π) where the left side is monotone increasing for our small straights."""
    lo, hi = 1e-5, math.pi - 1e-5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = (rho1 + rho2) * (1.0 - math.cos(mid)) + straight_um * math.sin(mid)
        if val < y_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sbend_centerline(rho1, rho2, straight_um, y_target, n):
    """Sample ``n`` (x, y) centerline points of a two-arc S-bend from (0, 0)
    heading +x to (X, ``y_target``) heading +x. Arc 1 (radius ``rho1``) curves
    toward +y by φ, an optional straight tangent section, then arc 2 (radius
    ``rho2``) curves back to horizontal. ``y_target`` may be negative (mirrored)."""
    side = 1.0 if y_target >= 0 else -1.0
    yt = abs(y_target)
    phi = _sbend_phi(rho1, rho2, straight_um, yt)
    n1 = max(2, n // 2)
    pts = []
    # arc 1: t in [0, phi], centre (0, rho1)
    for i in range(n1):
        t = phi * i / (n1 - 1)
        pts.append((rho1 * math.sin(t), rho1 * (1.0 - math.cos(t))))
    # straight tangent section at heading phi
    ax, ay = pts[-1]
    if straight_um > 1e-12:
        ax2, ay2 = ax + straight_um * math.cos(phi), ay + straight_um * math.sin(phi)
        pts.append((ax2, ay2))
        ax, ay = ax2, ay2
    # arc 2: heading phi -> 0, centre C2 = A + rho2*(sin phi, -cos phi)
    c2x, c2y = ax + rho2 * math.sin(phi), ay - rho2 * math.cos(phi)
    n2 = max(2, n - len(pts))
    for i in range(1, n2 + 1):
        th = phi * (1.0 - i / n2)
        pts.append((c2x - rho2 * math.sin(th), c2y + rho2 * math.cos(th)))
    return [(px, side * py) for px, py in pts]


def _cos_sbend(dy: float, length: float, n: int):
    """Raised-cosine S-bend centerline from (0, 0) to (``length``, ``dy``),
    horizontal (zero-slope) at both ends — the standard low-loss access-waveguide
    bend used to fan the two outputs apart until they decouple."""
    n = max(2, n)
    return [(length * i / (n - 1), dy * 0.5 * (1.0 - math.cos(math.pi * i / (n - 1))))
            for i in range(n)]


def _ribbon_polygon(centerline, width_um):
    """Constant-width ribbon (list of (prop, transverse) vertices, CCW) around a
    centerline: offset each sample ±width/2 along the local normal, left edge
    forward then right edge backward."""
    half = width_um / 2.0
    n = len(centerline)
    left, right = [], []
    for i in range(n):
        px, py = centerline[i]
        if i == 0:
            dx, dy = centerline[1][0] - px, centerline[1][1] - py
        elif i == n - 1:
            dx, dy = px - centerline[i - 1][0], py - centerline[i - 1][1]
        else:
            dx, dy = (centerline[i + 1][0] - centerline[i - 1][0],
                      centerline[i + 1][1] - centerline[i - 1][1])
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm  # left normal
        left.append((px + half * nx, py + half * ny))
        right.append((px - half * nx, py - half * ny))
    return left + right[::-1]


def y_branch(
    *,
    wg_width_um: float,
    splitting_length_um: float,
    gap_um: float,
    arm_length_um: float,
    top_radii: Tuple[float, float],
    bot_radii: Tuple[float, float],
    top_straight_um: float = 0.0,
    bot_straight_um: float = 0.0,
    output_offset_final_um: float | None = None,
    fanout_length_um: float = 0.0,
    region_halfwidth_um: float | None = None,
    solid_junction: bool = False,
    branch_tip_frac: float | None = None,
    tip_gap_um: float = 0.0,
    n_points: int = 60,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis: AxisName = "x",
    thickness_axis: AxisName = "z",
) -> Component:
    """An asymmetric S-curve 1x2 Y-junction splitter with an arbitrary power
    splitting ratio (Lin & Shi, Opt. Express 27, 14338 (2019)). In the paper's
    widened ``solid_junction`` construction, ``top_radii=(r1, r2)`` and
    ``bot_radii=(R1, R2)`` describe TWO successive S-shaped OUTER-BOUNDARY
    segments per side: the first expands the input edge from ``w0/2`` to the
    published half-width ``w`` and the second contracts it to the outer edge of
    the output guide. Each S segment is itself made from two equal-radius tangent
    arcs. The paper's straight ``l`` lies between the two top-boundary S segments;
    any remaining length is a straight lead-in at the input width. This is not an
    output-arm-centreline construction: treating (r1, r2) as the two arcs of one
    centreline S-bend erases most of the paper's split-ratio asymmetry.

    ``solid_junction=False`` retains the generic constant-width two-arm routing
    construction. There, each arm follows one two-radius centreline S-bend.

    Because the paper's two outputs sit only ``gap_um`` apart they would remain
    evanescently coupled (a directional coupler) and the per-arm power would slosh
    between them — so for a clean split-ratio readout the arms are fanned apart
    with a raised-cosine access bend over ``fanout_length_um`` to a decoupled
    ``output_offset_final_um`` (default = the junction offset, i.e. no fanout).
    Three ports: ``in`` (−axis), ``o_top`` / ``o_bot`` (+axis, offset in the width
    axis)."""
    width_axis, thickness_axis = _planar_axes(axis, thickness_axis)
    poly_axes = [ax for ax in _AXES if ax != thickness_axis]  # (u, v)
    offset = (wg_width_um + gap_um) / 2.0
    y_final = output_offset_final_um if output_offset_final_um is not None else offset
    half_L = splitting_length_um / 2.0
    t_center = center_um[_INDEX[thickness_axis]]
    slab_bounds = (t_center - thickness_um / 2.0, t_center + thickness_um / 2.0)

    def _to_uv(prop_off, trans_off):
        role = {axis: prop_off, width_axis: trans_off}
        return (role[poly_axes[0]] + center_um[_INDEX[poly_axes[0]]],
                role[poly_axes[1]] + center_um[_INDEX[poly_axes[1]]])

    structures: list[Structure] = []

    def _add_arm(radii, straight, sign):
        rho1, rho2 = radii
        y_out = sign * offset
        yf = sign * y_final
        # 1. junction S-bend: merged centre -> ±offset over the splitting region
        cl = _sbend_centerline(rho1, rho2, straight, y_out, n_points)
        cl = [(px - half_L, py) for px, py in cl]
        # 2. straight at ±offset out to the end of the paper's splitting region
        if half_L - cl[-1][0] > 1e-9:
            cl.append((half_L, y_out))
        # 3. fanout access bend ±offset -> ±y_final (decouple the arms)
        if abs(yf - y_out) > 1e-9 and fanout_length_um > 1e-9:
            fb = _cos_sbend(yf - y_out, fanout_length_um, max(2, n_points // 2))
            cl += [(half_L + fx, y_out + fy) for fx, fy in fb[1:]]
        # 4. straight to the output port
        cl.append((arm_length_um, yf))
        poly_pw = _ribbon_polygon(cl, wg_width_um)
        verts = [_to_uv(px, py) for px, py in poly_pw]
        structures.append(Structure(
            geometry=PolySlab(axis=thickness_axis, vertices_um=tuple(verts),
                              slab_bounds_um=slab_bounds), medium=medium))

    def _region_outer(radii, straight, sign):
        """Paper-faithful outer boundary made from two complete S segments.

        Segment 1 uses ``radii[0]`` twice and moves from the input edge
        ``±w0/2`` to the extremum ``±w``. Segment 2 uses ``radii[1]`` twice and
        returns to the output outer edge ``±(w0 + D/2)``. The specified straight
        lies between the S segments. They are right-aligned to the region exit;
        for an asymmetric small ``r1`` the leftover length is therefore a long
        input-width lead-in, exactly the mechanism visible in Fig. 1(c).
        """
        if region_halfwidth_um is None:
            raise ValueError("solid_junction requires region_halfwidth_um (paper w)")
        w = float(region_halfwidth_um)
        y_in = 0.5 * wg_width_um
        y_peak = w
        y_out = wg_width_um + 0.5 * gap_um
        if not (y_peak > y_out >= y_in > 0.0):
            raise ValueError(
                "solid_junction requires w > w0 + D/2 >= w0/2; "
                f"got w={w}, w0={wg_width_um}, D={gap_um}")

        rho1, rho2 = radii
        for name, radius, delta in (
            ("first", rho1, y_peak - y_in),
            ("second", rho2, y_peak - y_out),
        ):
            # A two-equal-arc S can move laterally by at most 4R (phi -> pi).
            if radius <= 0.0 or delta > 4.0 * radius + 1e-12:
                raise ValueError(
                    f"{name} paper S segment cannot span {delta:.6g} um with "
                    f"radius {radius:.6g} um")
        first = _sbend_centerline(
            rho1, rho1, 0.0, sign * (y_peak - y_in), n_points)
        second = _sbend_centerline(
            rho2, rho2, 0.0, -sign * (y_peak - y_out), n_points)
        used = first[-1][0] + straight + second[-1][0]
        lead = splitting_length_um - used
        if lead < -1e-6:
            raise ValueError(
                "paper boundary does not fit splitting_length_um: "
                f"S1+straight+S2={used:.6g} > L={splitting_length_um:.6g}")
        lead = max(0.0, lead)  # tolerate the paper's rounded L=2.32 um

        pts = [(-half_L, sign * y_in)]

        def extend(seq):
            for point in seq:
                if math.hypot(point[0] - pts[-1][0], point[1] - pts[-1][1]) > 1e-12:
                    pts.append(point)

        x = -half_L + lead
        extend([(x, sign * y_in)])
        extend([(x + px, sign * y_in + py) for px, py in first])
        x += first[-1][0]
        extend([(x + straight, sign * y_peak)])
        x += straight
        extend([(x + px, sign * y_peak + py) for px, py in second])
        # Rounded paper parameters leave a few nm; close on the exact L plane.
        extend([(half_L, sign * y_out)])
        return pts

    def _add_solid_half(radii, straight, sign):
        # A SOLID region half: the published S-curve OUTER boundary plus the
        # central triangular branch gap. Fig. 1(c) places the sharp branch point
        # at the tangent between the two equal-r2 arcs of the TOP final S segment.
        # An explicit branch_tip_frac remains available for generic callers, but
        # the paper spec deliberately leaves it unset so the geometry derives it.
        outer = _region_outer(radii, straight, sign)
        if branch_tip_frac is None:
            r2 = top_radii[1]
            gap_rise = float(region_halfwidth_um) - (wg_width_um + gap_um / 2.0)
            final_s = _sbend_centerline(r2, r2, 0.0, gap_rise, n_points)
            x_b = half_L - 0.5 * final_s[-1][0]
        else:
            if not 0.0 <= branch_tip_frac <= 1.0:
                raise ValueError("branch_tip_frac must lie in [0, 1]")
            x_b = -half_L + branch_tip_frac * splitting_length_um
        half_tip = sign * tip_gap_um / 2.0
        inner = [(-half_L, 0.0), (x_b, half_tip), (half_L, sign * gap_um / 2.0)]
        poly_pw = outer + inner[::-1]
        verts = [_to_uv(px, py) for px, py in poly_pw]
        structures.append(Structure(
            geometry=PolySlab(axis=thickness_axis, vertices_um=tuple(verts),
                              slab_bounds_um=slab_bounds), medium=medium))

    def _add_access_arm(sign):
        # Constant-wg_width output arm from the region end (+half_L, ±offset) through
        # the decoupling fanout to the port. Meets the region half exactly (the
        # region half is wg_width wide, centred on ±offset, at +half_L).
        cl = [(half_L, sign * offset)]
        if abs(y_final - offset) > 1e-9 and fanout_length_um > 1e-9:
            fb = _cos_sbend(sign * (y_final - offset), fanout_length_um, max(2, n_points // 2))
            cl += [(half_L + fx, sign * offset + fy) for fx, fy in fb[1:]]
        cl.append((arm_length_um, sign * y_final))
        poly_pw = _ribbon_polygon(cl, wg_width_um)
        verts = [_to_uv(px, py) for px, py in poly_pw]
        structures.append(Structure(
            geometry=PolySlab(axis=thickness_axis, vertices_um=tuple(verts),
                              slab_bounds_um=slab_bounds), medium=medium))

    if solid_junction:
        _add_solid_half(top_radii, top_straight_um, +1.0)
        _add_solid_half(bot_radii, bot_straight_um, -1.0)
        _add_access_arm(+1.0)
        _add_access_arm(-1.0)
    else:
        _add_arm(top_radii, top_straight_um, +1.0)
        _add_arm(bot_radii, bot_straight_um, -1.0)

    # Input routing stub (single wg_width guide) feeding the merged junction.
    structures.append(_slab_box(
        center=_vec(center_um, **{axis: -0.5 * (arm_length_um + half_L)}),
        prop_axis=axis, width_axis=width_axis, thickness_axis=thickness_axis,
        length_um=arm_length_um - half_L, width_um=wg_width_um,
        thickness_um=thickness_um, medium=medium))

    ports = (
        Port("in", _vec(center_um, **{axis: -arm_length_um}), axis, wg_width_um),
        Port("o_top", _vec(center_um, **{axis: arm_length_um, width_axis: +y_final}),
             axis, wg_width_um),
        Port("o_bot", _vec(center_um, **{axis: arm_length_um, width_axis: -y_final}),
             axis, wg_width_um),
    )
    return Component(structures=tuple(structures), ports=ports)
