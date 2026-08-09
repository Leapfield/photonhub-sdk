"""``plot_eps()`` — the rasterized permittivity the solver actually samples on
a cut plane, as a heatmap (design §3, §9).

Rebuilds the realized grid with the SAME helpers ``cost.py`` uses
(:func:`photonhub.components.grid.realized_cells` /
:func:`graded_primary_spacings`), samples ε at cell centers under the
NUMERICS §9 last-structure-wins rule, and renders with ``pcolormesh`` on the
real µm node coordinates so graded meshes are correct (never assumed uniform).

Every Phase-2 geometry is rasterized here so the preview equals what the solver
samples: ``Box`` and ``Sphere`` are exact on any cut, and the extruded
``Cylinder`` (annular sector) and ``PolySlab`` (point-in-polygon) reproduce the
engine's ``cylinder_contains`` / ``polyslab_contains`` predicates
(``engine/src/cpu_ref/reference_solver.cpp``) per pixel — both on a cut
PERPENDICULAR to the extrusion axis (the exact in-plane cross-section) and on a
cut ALONG it (the rectangular bands where the plane crosses the solid).

Caveat: subpixel smoothing (``Simulation.subpixel``) is NOT reflected here —
this is the §9 hard point sample. Engine-faithful ε (incl. subpixel) via a
future ``phsolver --mesh`` subcommand is deferred (design §11). The PolySlab
``sidewall_angle`` taper IS applied (§17.6), mirroring the engine: this module
feeds the client mode solver's cross-section, so painting the un-tapered
reference polygon would solve launch/readout modes on a different guide than the
engine propagates. (``plot``'s polygon OUTLINE is still the reference polygon —
the outline and this ε sample intentionally differ on a tapered slab.)
"""

import warnings
from typing import List, Tuple

import numpy as np

from ..components.grid import graded_primary_spacings, realized_cells
from . import _geometry as geom
from . import _style

_AXES = "xyz"


def axis_nodes_um(sim, axis_index: int) -> np.ndarray:
    """Primary-node coordinates (microns) for an axis of the realized grid:
    the graded ``coords`` array extended by the §15.1 replicate-last closing
    node, or a uniform ``n*dl`` ladder. Length ``n_cells + 1`` (cell edges)."""
    dl = sim.grid.dl_um
    q = sim._axis_coords_um(axis_index)
    if q is None:
        n = realized_cells(sim.size_um[axis_index], dl)
        return np.arange(n + 1, dtype=np.float64) * dl
    # Graded: n cells from len(coords) nodes; the closing node is q[-1] + the
    # replicated final spacing (cost.py / Simulation._realized_um use this).
    nodes = list(q)
    nodes.append(q[-1] + graded_primary_spacings(q)[-1])
    return np.asarray(nodes, dtype=np.float64)


def axis_cell_centers_um(nodes: np.ndarray) -> np.ndarray:
    """Cell centers (microns) from edge nodes — where ε is sampled."""
    return 0.5 * (nodes[:-1] + nodes[1:])


def realized_shape(sim) -> Tuple[int, int, int]:
    """(nx, ny, nz) realized cell counts, matching cost.py's grid."""
    out = []
    for i in range(3):
        q = sim._axis_coords_um(i)
        if q is None:
            out.append(realized_cells(sim.size_um[i], sim.grid.dl_um))
        else:
            out.append(len(q))
    return tuple(out)


def sample_eps_plane(sim, axis: str, value: float):
    """Sample the hard ε on the cut plane at cell centers (design §9
    last-structure-wins). Returns ``(h_nodes, v_nodes, eps2d)`` where the node
    arrays are µm cell edges for ``pcolormesh`` and ``eps2d`` has shape
    ``(n_v, n_h)`` (row = vertical axis, as pcolormesh expects)."""
    a = geom.axis_index(axis)
    h_axis_letter, v_axis_letter = geom.in_plane_axes(axis)
    h_idx = _AXES.index(h_axis_letter)
    v_idx = _AXES.index(v_axis_letter)

    h_nodes = axis_nodes_um(sim, h_idx)
    v_nodes = axis_nodes_um(sim, v_idx)
    h_centers = axis_cell_centers_um(h_nodes)
    v_centers = axis_cell_centers_um(v_nodes)

    # Start from the background everywhere, then paint structures in list order
    # (last containing structure wins, §9 closed containment).
    eps = np.full((v_centers.size, h_centers.size),
                  float(sim.background.permittivity), dtype=np.float64)

    # Build the fixed-axis coordinate of every sample point (constant = value).
    HH, VV = np.meshgrid(h_centers, v_centers)  # shape (n_v, n_h)
    # Per-sample 3D coordinate, only the three components we need to test.
    for structure in sim.structures:
        g = structure.geometry
        eps_val = float(structure.medium.permittivity)
        gtype = getattr(g, "type", None)
        if gtype == "box":
            c, s = g.center_um, g.size_um
            # Plane must intersect the box along the fixed axis.
            if abs(value - c[a]) > s[a] / 2.0:
                continue
            inside = (
                (np.abs(HH - c[h_idx]) <= s[h_idx] / 2.0)
                & (np.abs(VV - c[v_idx]) <= s[v_idx] / 2.0)
            )
            eps[inside] = eps_val
        elif gtype == "sphere":
            c, r = g.center_um, g.radius_um
            d_axis = value - c[a]
            if abs(d_axis) >= r:
                continue
            dh = HH - c[h_idx]
            dv = VV - c[v_idx]
            inside = (dh * dh + dv * dv + d_axis * d_axis) <= r * r
            eps[inside] = eps_val
        elif gtype == "cylinder":
            inside = _cylinder_inside(g, axis, value, HH, VV, h_idx, v_idx)
            if inside is not None:
                eps[inside] = eps_val
        elif gtype == "polyslab":
            inside = _polyslab_inside(g, axis, value, HH, VV, h_idx, v_idx)
            if inside is not None:
                eps[inside] = eps_val
        # Unknown geometry types are skipped (forward-compatible).
    return h_nodes, v_nodes, eps


def _cylinder_inside(g, cut_axis, value, HH, VV, h_idx, v_idx):
    """Boolean mask of the §9 hard sample for a ``Cylinder`` on the cut plane,
    or ``None`` if the plane misses the cylinder entirely. Mirrors the engine's
    ``cylinder_contains`` (annular sector, closed axial extent), evaluated
    per pixel over the ``(HH, VV)`` in-plane meshgrid.

    - Cut PERPENDICULAR to the extrusion axis: a point is inside iff the plane
      is within the axial extent AND ``inner_radius ≤ r ≤ radius`` (r measured
      from the centre in the transverse plane) AND its ``atan2`` angle lies in
      the ``[angle_start, angle_stop]`` sweep (full ring is atan2-free).
    - Cut ALONG the axis: the plane slices the annulus into one or two
      rectangular bands. The point is inside iff its axial coordinate is within
      the extent AND the transverse offsets satisfy the same
      ``inner ≤ r ≤ outer`` + sweep test (here one transverse component is the
      fixed ``value - center`` offset, the other varies over the plane)."""
    a = geom.axis_index(g.axis)              # cylinder's own extrusion axis
    cut_a = geom.axis_index(cut_axis)
    center = g.center_um
    half = g.length_um / 2.0
    ro = float(g.radius_um)
    ri = float(g.inner_radius_um)
    sweep = float(g.angle_stop - g.angle_start)
    full = sweep >= 2.0 * np.pi - 1e-9

    # The two transverse axes of the EXTRUSION axis, in (u, v) order — angles
    # are measured atan2(dv, du) exactly as the engine does.
    u_letter, v_letter = geom.in_plane_axes(g.axis)
    u_i = _AXES.index(u_letter)
    v_i = _AXES.index(v_letter)

    # Build per-pixel transverse offsets (du, dv) from the cylinder centre, plus
    # the axial coordinate, expressing each of the three world axes as either a
    # constant (the cut value) or one of the meshgrid arrays HH/VV.
    def world(comp_axis):
        if comp_axis == cut_a:
            return value
        if comp_axis == h_idx:
            return HH
        if comp_axis == v_idx:
            return VV
        return None  # unreachable: the three axes partition into cut/h/v

    ra = world(a)
    if np.isscalar(ra) and (ra < center[a] - half or ra > center[a] + half):
        return None  # perpendicular-ish cut wholly outside the axial extent
    du = world(u_i) - center[u_i]
    dv = world(v_i) - center[v_i]

    d2 = du * du + dv * dv
    inside = (d2 <= ro * ro) & (d2 >= ri * ri)
    # Closed axial extent (only constrains when the axial coord varies in-plane).
    if not np.isscalar(ra):
        inside = inside & (ra >= center[a] - half) & (ra <= center[a] + half)
    if not full:
        rel = np.mod(np.arctan2(dv, du) - g.angle_start, 2.0 * np.pi)
        inside = inside & (rel <= sweep)
    inside = np.broadcast_to(inside, HH.shape)
    return inside if inside.any() else None


def _polyslab_inside(g, cut_axis, value, HH, VV, h_idx, v_idx):
    """Boolean mask of the §9 hard sample for a ``PolySlab`` on the cut plane,
    or ``None`` if the plane misses it. Mirrors the engine's
    ``polyslab_contains`` (closed axial extent + even-odd point-in-polygon,
    plus the §17.6 ``sidewall_angle`` taper).

    - Cut PERPENDICULAR to the extrusion axis: in-plane point-in-polygon over
      the full ``(HH, VV)`` grid when the plane is within ``slab_bounds_um``.
    - Cut ALONG the axis: one in-plane coordinate is the extrusion axis (within
      ``slab_bounds_um``), the other is a transverse axis; the surviving
      transverse coordinate plus the fixed ``value`` form the polygon-space
      ``(pu, pv)`` tested against the polygon.

    §17.6 taper: for ``sidewall_angle != 0`` the reference polygon is
    eroded/dilated by the signed-distance band ``delta = (r_a - z_ref)*tan(angle)``
    (rounded-corner offset), with ``z_ref`` set by ``reference_plane``. This
    mirrors ``reference_solver.cpp::polyslab_contains`` — without it the client
    mode solver painted the UN-tapered polygon while the engine propagated the
    tapered one, so launch/readout reference modes were solved on a different
    guide than the run. At ``sidewall_angle == 0`` this is a bit-exact no-op."""
    a = geom.axis_index(g.axis)
    cut_a = geom.axis_index(cut_axis)
    lo, hi = g.slab_bounds_um
    verts = [(float(u), float(v)) for u, v in g.vertices_um]

    u_letter, v_letter = geom.in_plane_axes(g.axis)
    u_i = _AXES.index(u_letter)
    v_i = _AXES.index(v_letter)

    def world(comp_axis):
        if comp_axis == cut_a:
            return value
        if comp_axis == h_idx:
            return HH
        if comp_axis == v_idx:
            return VV
        return None

    ra = world(a)  # axial coordinate (scalar on a perpendicular cut)
    if np.isscalar(ra) and (ra < lo or ra > hi):
        return None
    pu = world(u_i)  # polygon-space coords, matching engine (u, v) order
    pv = world(v_i)
    inside = _point_in_polygon_vec(verts, pu, pv)
    inside = np.broadcast_to(inside, HH.shape).copy()

    angle = float(getattr(g, "sidewall_angle", 0.0) or 0.0)
    if angle != 0.0:
        # §17.6, mirroring polyslab_contains: phi = signed distance (<0 inside);
        # a point survives iff phi <= -delta.
        ref = str(getattr(g, "reference_plane", "middle") or "middle").lower()
        z_ref = lo if ref == "bottom" else (hi if ref == "top" else (lo + hi) / 2.0)
        delta = (np.asarray(ra, dtype=np.float64) - z_ref) * np.tan(angle)
        dist = _dist_to_polygon_boundary_vec(verts, pu, pv)
        phi = np.where(inside, -dist, dist)
        inside = np.broadcast_to(phi <= -delta, HH.shape).copy()

    if not np.isscalar(ra):
        axial_ok = np.broadcast_to((ra >= lo) & (ra <= hi), HH.shape)
        inside &= axial_ok
    return inside if inside.any() else None


def _dist_to_polygon_boundary_vec(verts, pu, pv):
    """Vectorized UNSIGNED distance from ``(pu, pv)`` to the polygon's edge set —
    the §17.6 mirror of the engine's ``dist_to_polygon_boundary`` (min over edges
    of the distance to the closest point on each clamped segment, which is what
    gives the rounded-corner offset)."""
    pu = np.asarray(pu, dtype=np.float64)
    pv = np.asarray(pv, dtype=np.float64)
    shape = np.broadcast(pu, pv).shape
    best = np.full(shape, np.inf, dtype=np.float64)
    n = len(verts)
    for i in range(n):
        ax, ay = verts[i - 1]          # j = i - 1 (wraps), matching the engine
        bx, by = verts[i]
        ex, ey = bx - ax, by - ay
        wx, wy = pu - ax, pv - ay
        ee = ex * ex + ey * ey
        t = (wx * ex + wy * ey) / ee if ee > 0.0 else np.zeros(shape)
        t = np.clip(t, 0.0, 1.0)
        ddx = pu - (ax + t * ex)
        ddy = pv - (ay + t * ey)
        best = np.minimum(best, ddx * ddx + ddy * ddy)
    return np.sqrt(best)


def _point_in_polygon_vec(verts, pu, pv):
    """Vectorized even-odd point-in-polygon (the engine's §17.5 ``ray_cross``
    rule) over numpy-broadcastable ``pu``/``pv``. ``verts`` is a list of
    ``(u, v)`` tuples. Returns a boolean array broadcast to ``pu/pv``."""
    pu = np.asarray(pu, dtype=np.float64)
    pv = np.asarray(pv, dtype=np.float64)
    inside = np.zeros(np.broadcast(pu, pv).shape, dtype=bool)
    n = len(verts)
    j = n - 1
    for i in range(n):
        ui, vi = verts[i]
        uj, vj = verts[j]
        cond = (vi > pv) != (vj > pv)
        # Avoid divide-by-zero where the edge is horizontal (cond is False there
        # so the result is masked out anyway); guard the denominator.
        denom = vj - vi
        denom = denom if denom != 0.0 else np.nan
        xcross = (uj - ui) * (pv - vi) / denom + ui
        inside ^= cond & (pu < xcross)
        j = i
    return inside


def plot_eps(sim, x=None, y=None, z=None, *, ax=None, cmap=None,
             legend=True, grid=False, **kw):
    """Render the rasterized ε heatmap on the selected cut plane. See the
    module docstring for the subpixel caveat. Returns the matplotlib ``Axes``.

    Exactly one of x/y/z (microns) selects the constant-coordinate cut plane.
    The plot uses the real µm node coordinates (``pcolormesh``), so graded
    meshes render with correct, non-uniform cell widths. ``grid=True`` overlays
    the realized cell edges — useful to confirm features are well resolved."""
    import matplotlib.pyplot as plt

    axis, value = geom.select_plane(x, y, z)
    if ax is None:
        _, ax = plt.subplots()

    # Cut outside the domain -> empty Axes + warning, never raise (design §9).
    a = geom.axis_index(axis)
    realized = sim._realized_um()
    if not (0.0 <= value <= realized[a]):
        warnings.warn(
            f"{axis}={value} um is outside the realized domain "
            f"[0, {realized[a]:.6g}] um on that axis; nothing to draw",
            UserWarning, stacklevel=2)
        _finish_axes(ax, axis)
        return ax

    h_nodes, v_nodes, eps = sample_eps_plane(sim, axis, value)
    vmin, vmax = _style.eps_norm(
        [sim.background.permittivity]
        + [s.medium.permittivity for s in sim.structures]
    )
    mesh = ax.pcolormesh(h_nodes, v_nodes, eps, cmap=cmap or _style.EPS_CMAP,
                         vmin=vmin, vmax=vmax, shading="flat", **kw)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label("permittivity ε (hard sample)")

    if grid:
        _style.draw_grid(ax, sim, axis)

    drew = _style.draw_overlays(ax, sim, axis, value)
    if legend:
        _style.add_legend(ax, source=drew["source"], monitor=drew["monitor"],
                          pml=drew["pml"], structure=bool(sim.structures))
    _finish_axes(ax, axis)
    return ax


def _finish_axes(ax, axis: str) -> None:
    h_ax, v_ax = geom.in_plane_axes(axis)
    ax.set_xlabel(f"{h_ax} (µm)")
    ax.set_ylabel(f"{v_ax} (µm)")
    ax.set_aspect("equal")
    ax.set_title(f"ε  ({axis}-cut)")
