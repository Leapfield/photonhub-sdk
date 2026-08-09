"""Cut-plane geometry shared by every 2D view and the 3D builder (design §5).

Given a constant-coordinate cut plane ``axis=a, value=v0`` (a in {x, y, z}),
project the scene's analytic geometry onto the two in-plane axes:

- :func:`box_rectangle` — a :class:`Box` intersected with the plane is a
  rectangle in the other two axes (or ``None`` when the plane misses it).
- :func:`sphere_circle` — a :class:`Sphere` intersected with the plane is a
  circle of radius ``sqrt(radius**2 - d**2)`` (or ``None`` when ``d >=
  radius``).
- :func:`point_in_plane` — a point feature (e.g. ``PointDipole.center_um``)
  drawn when it lies within ~half a cell of the plane.
- :func:`axis_line_in_plane` — an axis-aligned line feature (``PlaneWave`` /
  ``FluxMonitor``: ``axis`` + ``position_um``) drawn only when its own axis is
  one of the two in-plane axes.

:class:`Box` and :class:`Sphere` are exact on every cut; the extruded
:class:`PolySlab` / :class:`Cylinder` (added at the documented seam) are exact
on the cut perpendicular to their extrusion ``axis`` and approximated by a
bounding rectangle on a parallel cut (see :func:`polyslab_cross_section` /
:func:`cylinder_cross_section`). Each new geometry adds one ``*_cross_section``
helper plus a branch in :func:`structure_patch_spec`.

The helpers return plain numbers/tuples (a small tagged-union of patch specs)
so they stay reusable from matplotlib (2D) and plotly (3D) without importing
either. The recognized patch-spec kinds are:

- ``("rect", (x0, y0, w, h))`` — Box, or a parallel-cut bounding box.
- ``("circle", (cx, cy, r))`` — Sphere, or a full solid Cylinder disk.
- ``("polygon", ((h0, v0), (h1, v1), ...))`` — PolySlab cross-section, or a
  Cylinder wedge/annular-sector approximated by an arc polygon.
- ``("annulus", (cx, cy, r_outer, r_inner))`` — a full Cylinder ring.
"""

import math
from typing import Optional, Tuple

_AXES = "xyz"


def axis_index(axis: str) -> int:
    """Index 0/1/2 of an axis letter; raises on anything but x/y/z."""
    try:
        return _AXES.index(axis)
    except ValueError:  # pragma: no cover - guarded by the callers
        raise ValueError(f"axis must be one of 'x', 'y', 'z', got {axis!r}")


def in_plane_axes(axis: str) -> Tuple[str, str]:
    """The two axis letters that remain in the cut plane, in (horizontal,
    vertical) order: the natural pair for a constant-``axis`` slice. For a
    z-cut this is ('x', 'y'); for x it is ('y', 'z'); for y it is ('x', 'z')."""
    a = axis_index(axis)
    rest = [i for i in range(3) if i != a]
    return _AXES[rest[0]], _AXES[rest[1]]


def select_plane(x, y, z) -> Tuple[str, float]:
    """Resolve exactly one of x/y/z (microns) to ``(axis, value)``; not
    exactly one set -> ValueError (design §9)."""
    given = [(ax, v) for ax, v in (("x", x), ("y", y), ("z", z)) if v is not None]
    if len(given) != 1:
        raise ValueError(
            "exactly one of x=, y=, z= must be given (the constant-coordinate "
            f"cut plane, in microns); got {len(given)}"
        )
    return given[0]


# --------------------------------------------------------------------------- #
# Structure cross-sections.
# --------------------------------------------------------------------------- #

def box_rectangle(
    center_um: Tuple[float, float, float],
    size_um: Tuple[float, float, float],
    axis: str,
    value: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Cross-section of a box on the plane, as ``(x0, y0, width, height)`` in
    the in-plane axes' order, or ``None`` if the plane does not intersect.

    Intersects iff ``|value - center[axis]| <= size[axis]/2`` (closed, matching
    the §9 closed-containment rasterization rule)."""
    a = axis_index(axis)
    if abs(value - center_um[a]) > size_um[a] / 2.0:
        return None
    h_ax, v_ax = (_AXES.index(c) for c in in_plane_axes(axis))
    x0 = center_um[h_ax] - size_um[h_ax] / 2.0
    y0 = center_um[v_ax] - size_um[v_ax] / 2.0
    return (x0, y0, size_um[h_ax], size_um[v_ax])


def sphere_circle(
    center_um: Tuple[float, float, float],
    radius_um: float,
    axis: str,
    value: float,
) -> Optional[Tuple[float, float, float]]:
    """Cross-section of a sphere on the plane, as ``(cx, cy, r)`` in the
    in-plane axes' order, or ``None`` if the plane does not intersect.

    Intersects iff ``d = |value - center[axis]| < radius``; the cut radius is
    ``sqrt(radius**2 - d**2)`` (design §5)."""
    a = axis_index(axis)
    d = abs(value - center_um[a])
    if d >= radius_um:
        return None
    h_ax, v_ax = (_AXES.index(c) for c in in_plane_axes(axis))
    r = math.sqrt(radius_um * radius_um - d * d)
    return (center_um[h_ax], center_um[v_ax], r)


def _slab_extent(slab_bounds_um: Tuple[float, float]) -> Tuple[float, float]:
    """``(lo, hi)`` axial extent of a PolySlab from its (lo, hi) bounds (already
    ordered lo<hi by the model validator)."""
    lo, hi = slab_bounds_um
    return (lo, hi)


def polyslab_section(
    geom_axis: str,
    vertices_um,
    slab_bounds_um: Tuple[float, float],
    cut_axis: str,
    value: float,
):
    """Cut-plane spec of a PolySlab extruded along ``geom_axis``;
    ``cut_axis``/``value`` is the cut plane.

    - Cut PERPENDICULAR to the extrusion axis (``cut_axis == geom_axis`` and
      ``slab_lo <= value <= slab_hi``): the exact cross-section is the reference
      polygon ``vertices_um`` mapped into the two in-plane axes ->
      ``("polygon", ((h, v), ...))``. The vertices are stored in ``(u, v)`` =
      (lower-indexed, higher-indexed) transverse order, which already matches
      the ``(h, v)`` in-plane order this module uses, so no swap is needed.
      ``sidewall_angle`` is NOT applied — the reference polygon is drawn as-is
      (the slanted-wall taper to the cut height is deferred; see the caveat in
      :class:`~photonhub.components.structures.PolySlab`).
    - Cut PARALLEL to the extrusion axis (the cut axis is one of the two
      transverse axes): an APPROXIMATE bounding rectangle spanning
      ``[slab_lo, slab_hi]`` along the geometry axis and the polygon's extent
      along the surviving transverse axis, but only when ``value`` falls inside
      the polygon's span on the cut axis. The exact (possibly tilted) wall
      profile is deferred. -> ``("rect", (x0, y0, w, h))``.

    Returns ``None`` when the plane misses the slab."""
    lo, hi = _slab_extent(slab_bounds_um)
    verts = [(float(u), float(v)) for u, v in vertices_um]
    u_ax, v_ax = in_plane_axes(geom_axis)  # the two transverse axes (u, v order)

    if cut_axis == geom_axis:
        # Perpendicular cut: exact polygon when within the axial extent.
        if not (lo <= value <= hi):
            return None
        return ("polygon", tuple(verts))

    # Parallel cut: bounding-rectangle approximation.
    h_ax, v_ax2 = in_plane_axes(cut_axis)
    # Which transverse axis is the cut axis, and which survives in-plane?
    if cut_axis == u_ax:
        cut_coords = [u for u, _ in verts]      # span tested against the cut
        keep_coords = [v for _, v in verts]     # surviving transverse extent
        keep_axis = v_ax
    else:  # cut_axis == v_ax
        cut_coords = [v for _, v in verts]
        keep_coords = [u for u, _ in verts]
        keep_axis = u_ax
    if not (min(cut_coords) <= value <= max(cut_coords)):
        return None  # plane misses the polygon's transverse span
    # Build the rect in (h_ax, v_ax2) order. One in-plane axis is the geometry
    # axis (extent [lo, hi]); the other is the surviving transverse axis.
    spans = {geom_axis: (lo, hi),
             keep_axis: (min(keep_coords), max(keep_coords))}
    h_lo, h_hi = spans[h_ax]
    v_lo, v_hi = spans[v_ax2]
    return ("rect", (h_lo, v_lo, h_hi - h_lo, v_hi - v_lo))


def _arc_polygon(cx, cy, r_outer, r_inner, a0, a1, n=48):
    """Vertices of an annular sector (wedge) from ``a0`` to ``a1`` radians,
    outer radius ``r_outer`` >= inner ``r_inner`` >= 0, as ``((h, v), ...)``.
    A solid sector (``r_inner == 0``) is the outer arc closed through the
    centre; an annular sector traces the outer arc out and the inner arc back."""
    sweep = a1 - a0
    steps = max(2, int(math.ceil(abs(sweep) / (2.0 * math.pi) * n)))
    outer = [(cx + r_outer * math.cos(a0 + sweep * t / steps),
              cy + r_outer * math.sin(a0 + sweep * t / steps))
             for t in range(steps + 1)]
    if r_inner <= 0.0:
        return tuple(outer + [(cx, cy)])
    inner = [(cx + r_inner * math.cos(a1 - sweep * t / steps),
              cy + r_inner * math.sin(a1 - sweep * t / steps))
             for t in range(steps + 1)]
    return tuple(outer + inner)


def cylinder_section(
    geom_axis: str,
    center_um: Tuple[float, float, float],
    radius_um: float,
    inner_radius_um: float,
    length_um: float,
    angle_start: float,
    angle_stop: float,
    cut_axis: str,
    value: float,
    full_sweep_tol: float = 1e-9,
):
    """Cut-plane spec of a Cylinder (annular sector) extruded along
    ``geom_axis`` with axial length ``length_um`` centred at ``center_um``.

    - Cut PERPENDICULAR to the axis (``cut_axis == geom_axis``, ``value`` within
      the axial extent ``center +- length/2``): the cross-section is EXACT —
      a full sweep + solid -> ``("circle", (cx, cy, r))``; a full sweep with
      ``inner_radius>0`` -> ``("annulus", (cx, cy, r_outer, r_inner))``; a
      partial sweep -> ``("polygon", arc-vertices)`` (a wedge, annular when
      ``inner_radius>0``). Angles use ``atan2(v, u)`` in the (u, v) transverse
      plane, matching the rasterizer.
    - Cut PARALLEL to the axis: an APPROXIMATE bounding rectangle spanning
      ``[axial_lo, axial_hi]`` and the cylinder DIAMETER on the surviving
      transverse axis, when the plane lies within +-radius of the centre on the
      cut axis. The exact curved-wall profile is deferred. ->
      ``("rect", (x0, y0, w, h))``.

    Returns ``None`` when the plane misses the cylinder."""
    a = axis_index(geom_axis)
    half = length_um / 2.0
    axial_lo = center_um[a] - half
    axial_hi = center_um[a] + half
    u_ax, v_ax = in_plane_axes(geom_axis)
    u_i, v_i = _AXES.index(u_ax), _AXES.index(v_ax)
    cu, cv = center_um[u_i], center_um[v_i]

    if cut_axis == geom_axis:
        if not (axial_lo <= value <= axial_hi):
            return None
        sweep = angle_stop - angle_start
        is_full = sweep >= 2.0 * math.pi - full_sweep_tol
        if is_full and inner_radius_um <= 0.0:
            return ("circle", (cu, cv, radius_um))
        if is_full:
            return ("annulus", (cu, cv, radius_um, inner_radius_um))
        return ("polygon",
                _arc_polygon(cu, cv, radius_um, inner_radius_um,
                             angle_start, angle_stop))

    # Parallel cut: bounding-box approximation across the full radius.
    h_ax, v_ax2 = in_plane_axes(cut_axis)
    if cut_axis == u_ax:
        center_on_cut, keep_center, keep_axis = cu, cv, v_ax
    else:  # cut_axis == v_ax
        center_on_cut, keep_center, keep_axis = cv, cu, u_ax
    if abs(value - center_on_cut) > radius_um:
        return None
    spans = {geom_axis: (axial_lo, axial_hi),
             keep_axis: (keep_center - radius_um, keep_center + radius_um)}
    h_lo, h_hi = spans[h_ax]
    v_lo, v_hi = spans[v_ax2]
    return ("rect", (h_lo, v_lo, h_hi - h_lo, v_hi - v_lo))


def structure_patch_spec(geometry, axis: str, value: float):
    """Dispatch a geometry to its cut-plane patch spec (a tagged tuple, see the
    module docstring), or ``None`` when the plane misses the geometry. Unknown
    geometry types return ``None`` (forward-compatible: a later track adds its
    own branch rather than crashing this one)."""
    gtype = getattr(geometry, "type", None)
    if gtype == "box":
        rect = box_rectangle(geometry.center_um, geometry.size_um, axis, value)
        return ("rect", rect) if rect is not None else None
    if gtype == "sphere":
        circ = sphere_circle(geometry.center_um, geometry.radius_um, axis, value)
        return ("circle", circ) if circ is not None else None
    if gtype == "polyslab":
        return polyslab_section(
            geometry.axis, geometry.vertices_um, geometry.slab_bounds_um,
            axis, value)
    if gtype == "cylinder":
        return cylinder_section(
            geometry.axis, geometry.center_um, geometry.radius_um,
            geometry.inner_radius_um, geometry.length_um,
            geometry.angle_start, geometry.angle_stop, axis, value)
    return None


# --------------------------------------------------------------------------- #
# Point / line projections (sources, monitors).
# --------------------------------------------------------------------------- #

def point_in_plane(
    center_um: Tuple[float, float, float],
    axis: str,
    value: float,
    half_thickness_um: float,
) -> Optional[Tuple[float, float]]:
    """In-plane ``(h, v)`` location of a point feature, drawn only when it lies
    within ``half_thickness_um`` of the plane (≈ half a cell); else ``None``."""
    a = axis_index(axis)
    if abs(value - center_um[a]) > half_thickness_um:
        return None
    h_ax, v_ax = (_AXES.index(c) for c in in_plane_axes(axis))
    return (center_um[h_ax], center_um[v_ax])


def axis_line_in_plane(
    feature_axis: str,
    position_um: float,
    cut_axis: str,
) -> Optional[Tuple[str, float]]:
    """A feature that is a full plane perpendicular to ``feature_axis`` at
    ``position_um`` (PlaneWave / FluxMonitor) appears as a full-span LINE in a
    cut only when ``feature_axis`` lies in the cut plane — i.e. is one of the
    two in-plane axes. Returns ``(orientation_axis, position)`` where
    ``orientation_axis`` is the in-plane axis the line is constant along, or
    ``None`` when the feature's plane is parallel to the cut (out of plane,
    design §5/§9)."""
    if feature_axis == cut_axis:
        return None
    return (feature_axis, position_um)
