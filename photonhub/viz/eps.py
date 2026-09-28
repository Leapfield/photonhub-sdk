"""``plot_index()`` shows permittivity sampled on the mesh on a cut plane
as a heatmap.

Rebuilds the realized grid with the SAME helpers ``cost.py`` uses
(:func:`photonhub.components.grid.realized_cells` /
:func:`graded_primary_spacings`), samples ε at cell centers under the
NUMERICS §9 last-structure-wins rule, and renders with ``pcolormesh`` on the
real µm node coordinates so graded meshes are correct (never assumed uniform).

Every supported geometry is sampled on the mesh here so the preview equals what the solver
samples: ``Box`` and ``Sphere`` are exact on any cut, and the extruded
``Cylinder`` (annular sector) and ``Polygon`` (point-in-polygon) reproduce the
engine's ``cylinder_contains`` / ``polyslab_contains`` predicates
(``engine/src/cpu_ref/reference_solver.cpp``) per pixel, both on a cut
PERPENDICULAR to the extrusion axis (the exact in-plane cross-section) and on a
cut ALONG it (the rectangular bands where the plane crosses the solid).

Subpixel smoothing IS reflected: when ``Simulation.subpixel`` is on, the view
shows the §16 volume-fraction average the solver samples on the mesh rather than the §9
hard point sample, so the drawn boundary is the one that runs. It is the
isotropic average, resolved on a sub-sample lattice; the per-component
harmonic term of the ``tensor``/``contour`` methods cannot be shown as one
scalar (see :func:`smooth_eps_plane`). Pass ``subpixel=False`` for the hard
sample. A ``phsolver --mesh`` dump of the engine's own coefficients is still
the only way to compare byte for byte. The Polygon
``sidewall_angle`` taper IS applied (§17.6), mirroring the engine: this module
feeds the client mode solver's cross-section, so painting the un-tapered
reference polygon would solve launch/readout modes on a different guide than the
engine propagates. (``plot``'s polygon OUTLINE is still the reference polygon, the outline and this ε sample intentionally differ on a tapered slab.)
"""

import math
import warnings
from typing import List, Optional, Tuple

import numpy as np

from ..components.grid import (graded_primary_spacings, realized_cells,
                               sim_axis_min_cells)
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
        n = realized_cells(sim.size_um[axis_index], dl,
                           sim_axis_min_cells(sim, axis_index))
        return np.arange(n + 1, dtype=np.float64) * dl
    # Graded: n cells from len(coords) nodes; the closing node is q[-1] + the
    # replicated final spacing (cost.py / Simulation._realized_um use this).
    nodes = list(q)
    nodes.append(q[-1] + graded_primary_spacings(q)[-1])
    return np.asarray(nodes, dtype=np.float64)


def axis_cell_centers_um(nodes: np.ndarray) -> np.ndarray:
    """Cell centers (microns) from edge nodes, where ε is sampled."""
    return 0.5 * (nodes[:-1] + nodes[1:])


def realized_shape(sim) -> Tuple[int, int, int]:
    """(nx, ny, nz) realized cell counts, matching cost.py's grid."""
    out = []
    for i in range(3):
        q = sim._axis_coords_um(i)
        if q is None:
            out.append(realized_cells(sim.size_um[i], sim.grid.dl_um,
                                      sim_axis_min_cells(sim, i)))
        else:
            out.append(len(q))
    return tuple(out)


def sample_eps_plane(sim, axis: str, value: float, *, subpixel: bool = False,
                     supersample: Optional[int] = None):
    """Sample ε on the cut plane at cell centers. Returns ``(h_nodes, v_nodes,
    eps2d)`` where the node arrays are µm cell edges for ``pcolormesh`` and
    ``eps2d`` has shape ``(n_v, n_h)`` (row = vertical axis, as pcolormesh
    expects).

    ``subpixel=False`` (the default, and what every physics caller wants) is
    the §9 hard point sample: one last-structure-wins material per cell
    centre. ``subpixel=True`` returns the §16 VOLUME-FRACTION AVERAGE instead,
    the smoothed ε the solver samples on the mesh when ``Simulation.subpixel`` is on;
    see :func:`smooth_eps_plane` for what that does and does not reproduce."""
    a = geom.axis_index(axis)
    h_axis_letter, v_axis_letter = geom.in_plane_axes(axis)
    h_idx = _AXES.index(h_axis_letter)
    v_idx = _AXES.index(v_axis_letter)

    h_nodes = axis_nodes_um(sim, h_idx)
    v_nodes = axis_nodes_um(sim, v_idx)
    h_centers = axis_cell_centers_um(h_nodes)
    v_centers = axis_cell_centers_um(v_nodes)

    HH, VV = np.meshgrid(h_centers, v_centers)  # shape (n_v, n_h)
    eps = eps_at_points(sim, axis, HH, VV, float(value))
    if subpixel:
        eps = smooth_eps_plane(sim, axis, float(value), h_nodes, v_nodes, eps,
                               supersample=supersample)
    return h_nodes, v_nodes, eps


def eps_at_points(sim, axis: str, HH, VV, WW, structures=None):
    """§9 last-structure-wins ε at arbitrary points, painted structure by
    structure in list order (closed containment).

    ``HH``/``VV`` are the cut's in-plane coordinates and ``WW`` the coordinate
    ALONG the cut axis. ``WW`` is a float for the plane itself and an array,
    shaped like ``HH``, for points spread through a cell's depth, which is
    what makes the §16 volume fraction computable from a 2D view at all, since
    the smoothing weight is a fraction of the three-dimensional voxel.

    ``structures`` restricts the paint to a subset, in the SAME relative order
    (a caller that has already culled by bounding box passes the survivors).
    A structure outside its own bounding box contains nothing, so culling
    never changes the answer, only the cost, which matters because the cost
    is points times structures, and a metasurface has thousands."""
    a = geom.axis_index(axis)
    h_axis_letter, v_axis_letter = geom.in_plane_axes(axis)
    h_idx = _AXES.index(h_axis_letter)
    v_idx = _AXES.index(v_axis_letter)

    # Start from the background everywhere, then paint structures in list order
    # (last containing structure wins, §9 closed containment).
    eps = np.full(HH.shape, float(sim.background.permittivity),
                  dtype=np.float64)
    flat_w = np.isscalar(WW)

    for structure in (sim.structures if structures is None else structures):
        g = structure.geometry
        eps_val = float(structure.medium.permittivity)
        gtype = getattr(g, "type", None)
        if gtype == "box":
            c, s = g.center_um, g.size_um
            axial = np.abs(WW - c[a]) <= s[a] / 2.0
            if flat_w:
                if not axial:
                    continue          # plane misses the box along the cut axis
                axial = True
            elif not axial.any():
                continue
            inside = (
                (np.abs(HH - c[h_idx]) <= s[h_idx] / 2.0)
                & (np.abs(VV - c[v_idx]) <= s[v_idx] / 2.0)
            )
            if axial is not True:
                inside = inside & axial
            pd = getattr(structure.medium, "permittivity_data", None)
            if pd is not None:
                # §10.3 custom medium: trilinear node grid over the box
                # extent (mirrors the engine's eps_data_at) — previews show
                # the real profile, and downstream homogeneity checks (e.g.
                # diffraction_orders' n inference) correctly see a
                # non-uniform plane instead of the scalar mean.
                shape = pd.shape
                vals = np.asarray(pd.values, dtype=np.float64).reshape(shape)

                def _frac(coord, cd, sd, n):
                    f = ((np.asarray(coord, dtype=np.float64)
                          - (cd - sd / 2.0)) / sd) if sd > 0 else 0.0
                    return np.clip(f, 0.0, 1.0) * (n - 1)

                q = [None, None, None]
                q[h_idx] = _frac(HH, c[h_idx], s[h_idx], shape[h_idx])
                q[v_idx] = _frac(VV, c[v_idx], s[v_idx], shape[v_idx])
                q[a] = _frac(WW, c[a], s[a], shape[a])
                i0, w = [], []
                for d in range(3):
                    qq = np.asarray(q[d], dtype=np.float64)
                    lo = np.minimum(qq.astype(np.int64), shape[d] - 2)
                    i0.append(lo)
                    w.append(qq - lo)
                acc = np.zeros_like(HH, dtype=np.float64)
                for bx in (0, 1):
                    for by in (0, 1):
                        for bz in (0, 1):
                            wt = ((w[0] if bx else 1.0 - w[0])
                                  * (w[1] if by else 1.0 - w[1])
                                  * (w[2] if bz else 1.0 - w[2]))
                            acc = acc + wt * vals[i0[0] + bx, i0[1] + by,
                                                  i0[2] + bz]
                eps[inside] = np.broadcast_to(acc, eps.shape)[inside]
            else:
                eps[inside] = eps_val
        elif gtype == "sphere":
            c, r = g.center_um, g.radius_um
            d_axis = WW - c[a]
            if flat_w and abs(d_axis) >= r:
                continue
            dh = HH - c[h_idx]
            dv = VV - c[v_idx]
            inside = (dh * dh + dv * dv + d_axis * d_axis) <= r * r
            eps[inside] = eps_val
        elif gtype == "cylinder":
            inside = _cylinder_inside(g, axis, WW, HH, VV, h_idx, v_idx)
            if inside is not None:
                eps[inside] = eps_val
        elif gtype == "polyslab":
            inside = _polyslab_inside(g, axis, WW, HH, VV, h_idx, v_idx)
            if inside is not None:
                eps[inside] = eps_val
        # Unknown geometry types are skipped (forward-compatible).
    return eps


def geometry_bbox(g):
    """Axis-aligned world bounding box ``(lo3, hi3)`` of a geometry, or
    ``None`` when the kind is unknown (never cull what you cannot bound)."""
    gtype = getattr(g, "type", None)
    if gtype == "box":
        c, s = g.center_um, g.size_um
        return ([c[i] - s[i] / 2.0 for i in range(3)],
                [c[i] + s[i] / 2.0 for i in range(3)])
    if gtype == "sphere":
        c, r = g.center_um, float(g.radius_um)
        return ([c[i] - r for i in range(3)], [c[i] + r for i in range(3)])
    if gtype == "cylinder":
        a = geom.axis_index(g.axis)
        c, r = g.center_um, float(g.radius_um)
        half = float(g.length_um) / 2.0
        lo = [c[i] - r for i in range(3)]
        hi = [c[i] + r for i in range(3)]
        lo[a], hi[a] = c[a] - half, c[a] + half
        return lo, hi
    if gtype == "polyslab":
        a = geom.axis_index(g.axis)
        u_i, v_i = (_AXES.index(x) for x in geom.in_plane_axes(g.axis))
        us = [float(u) for u, _ in g.vertices_um]
        vs = [float(v) for _, v in g.vertices_um]
        slab_lo, slab_hi = g.slab_bounds_um
        # §17.6: a sidewall taper dilates the reference polygon by at most
        # the slab thickness times tan(angle), so the bound must include it.
        angle = float(getattr(g, "sidewall_angle", 0.0) or 0.0)
        pad = abs(math.tan(angle)) * (slab_hi - slab_lo) if angle else 0.0
        lo = [0.0, 0.0, 0.0]
        hi = [0.0, 0.0, 0.0]
        lo[a], hi[a] = slab_lo, slab_hi
        lo[u_i], hi[u_i] = min(us) - pad, max(us) + pad
        lo[v_i], hi[v_i] = min(vs) - pad, max(vs) + pad
        return lo, hi
    return None


def _cull(structures, lo3, hi3):
    """The structures whose bounding box overlaps the region, in list order."""
    out = []
    for st in structures:
        bb = geometry_bbox(st.geometry)
        if bb is None:
            out.append(st)
            continue
        blo, bhi = bb
        if all(blo[i] <= hi3[i] and bhi[i] >= lo3[i] for i in range(3)):
            out.append(st)
    return out


def _faces_cross(sim, axis_index_: int, lo: float, hi: float) -> bool:
    """Whether any structure has a bounding-box face strictly inside
    ``(lo, hi)`` on an axis, the cheap test for "this cut grazes a surface",
    which is the only case needing the extra front/back gate samples."""
    for st in sim.structures:
        bb = geometry_bbox(st.geometry)
        if bb is None:
            return True
        for face in (bb[0][axis_index_], bb[1][axis_index_]):
            if lo < face < hi:
                return True
    return False


# Cell-block edge for the smoothing pass. Small enough that a tile of a dense
# lattice sees a handful of bodies, large enough that the per-tile numpy
# overhead stays amortised.
_SMOOTH_TILE_CELLS = 48

# Sub-sample budget for the §16 smoothing: the largest per-axis division whose
# total point count stays under this. A picture is not a solve, and the
# polygon predicate costs one pass per vertex per point.
_SMOOTH_POINT_BUDGET = 400_000
_SMOOTH_MAX_DIVISIONS = 8


def smooth_eps_plane(sim, axis: str, value: float, h_nodes, v_nodes, hard,
                     *, supersample: Optional[int] = None):
    """The §16 volume-fraction average of ε over each cell, given the §9 hard
    sample ``hard`` on the same grid.

    Each cell's ε becomes the mean of the material over its PRIMARY voxel,
    sampled on an ``supersample``-per-axis lattice through all three axes. For
    one interface against a homogeneous surround that is exactly the §16.3
    isotropic average, and it is exactly the tangential entry ε∥ of the §16.8
    diagonal KFJ tensor, so it is the smoothed permittivity under
    ``subpixel_method="volume"`` and the tangential half of the tensor
    methods.

    What it is NOT: the per-component tensor. Under ``tensor``/``contour`` the
    component along the interface normal sees the HARMONIC mean ε⊥, which is
    lower than this; a single scalar picture cannot show three different
    coefficients at once. Conductivity is not smoothed either, matching the
    engine. Treat this as the smoothed ε the mesh sampler builds, resolved to
    the sub-sample lattice rather than in closed form.

    Only partially filled cells are sampled; a cell whose neighbourhood is one
    material has fill fraction 1 and keeps its hard value, which is the §16.4
    bit-identity floor. A cell is taken as partially filled when the material
    changes across the plane OR between the cell's own front and back faces
    along the cut axis. A feature smaller than one cell that the hard sample
    misses entirely is missed here too. ``supersample=None`` picks the finest
    division that fits the point budget."""
    if not sim.structures:
        return hard
    a = geom.axis_index(axis)
    w_nodes = axis_nodes_um(sim, a)
    w_lo, w_hi = _cell_bounds(w_nodes, value)

    h_idx = _AXES.index(geom.in_plane_axes(axis)[0])
    v_idx = _AXES.index(geom.in_plane_axes(axis)[1])
    HH, VV = np.meshgrid(axis_cell_centers_um(h_nodes),
                         axis_cell_centers_um(v_nodes))
    # A cell is partially filled either because the material changes ACROSS
    # the plane, or because a face crosses the cell along the CUT axis — a cut
    # grazing the top of a slab is uniform in plane and still half empty. The
    # second case costs two extra samples, at the cell's own front and back.
    edge = _interface_cells(hard)
    if _faces_cross(sim, a, w_lo, w_hi):
        # The cut grazes a surface: a cell can be uniform in plane and still
        # half empty through its depth, which the in-plane test cannot see.
        inset = 0.25 * (w_hi - w_lo)
        edge = edge | (eps_at_points(sim, axis, HH, VV, w_lo + inset)
                       != eps_at_points(sim, axis, HH, VV, w_hi - inset))
    n_cells = int(edge.sum())
    if n_cells == 0:
        return hard

    div = supersample or max(
        2, min(_SMOOTH_MAX_DIVISIONS,
               int(round((_SMOOTH_POINT_BUDGET / n_cells) ** (1.0 / 3.0)))))
    offs = (np.arange(div, dtype=np.float64) + 0.5) / div
    w_sub = w_lo + offs * (w_hi - w_lo)

    out = hard.copy()
    # Walk the plane in cell blocks so each block can drop the structures whose
    # bounding box misses it. Cost is points times structures, and a dense
    # lattice has thousands of bodies of which a block sees a handful.
    t = _SMOOTH_TILE_CELLS
    for r0 in range(0, edge.shape[0], t):
        for c0 in range(0, edge.shape[1], t):
            block = edge[r0:r0 + t, c0:c0 + t]
            if not block.any():
                continue
            br, bc = np.nonzero(block)
            rows, cols = br + r0, bc + c0
            here = _cull(sim.structures,
                         _corner(h_nodes, v_nodes, w_lo, cols.min(),
                                 rows.min(), axis, sim, 0),
                         _corner(h_nodes, v_nodes, w_hi, cols.max() + 1,
                                 rows.max() + 1, axis, sim, 1))
            if not here:
                continue
            h_sub = h_nodes[cols][:, None] + offs[None, :] * (
                h_nodes[cols + 1] - h_nodes[cols])[:, None]
            v_sub = v_nodes[rows][:, None] + offs[None, :] * (
                v_nodes[rows + 1] - v_nodes[rows])[:, None]
            HHs = np.repeat(np.repeat(h_sub[:, None, :], div, axis=1)[..., None],
                            div, axis=3)          # (n, div_v, div_h, div_w)
            VVs = np.repeat(np.repeat(v_sub[:, :, None], div, axis=2)[..., None],
                            div, axis=3)
            WWs = np.broadcast_to(w_sub, HHs.shape)
            sampled = eps_at_points(sim, axis, HHs, VVs, WWs, structures=here)
            out[rows, cols] = sampled.reshape(rows.size, -1).mean(axis=1)
    return out


def _corner(h_nodes, v_nodes, w, h_i, v_i, axis, sim, which):
    """One corner of a cell block as a world (x, y, z) triple, for culling."""
    a = geom.axis_index(axis)
    h_ax, v_ax = geom.in_plane_axes(axis)
    out = [0.0, 0.0, 0.0]
    out[a] = float(w)
    out[_AXES.index(h_ax)] = float(h_nodes[min(h_i, h_nodes.size - 1)])
    out[_AXES.index(v_ax)] = float(v_nodes[min(v_i, v_nodes.size - 1)])
    return out


def _interface_cells(hard):
    """Cells that touch a material change, dilated by one so a voxel whose
    own centre sits in the bulk but whose face is cut still gets sampled."""
    differs = np.zeros(hard.shape, dtype=bool)
    differs[:, :-1] |= hard[:, :-1] != hard[:, 1:]
    differs[:, 1:] |= hard[:, :-1] != hard[:, 1:]
    differs[:-1, :] |= hard[:-1, :] != hard[1:, :]
    differs[1:, :] |= hard[:-1, :] != hard[1:, :]
    out = differs.copy()
    out[:, :-1] |= differs[:, 1:]
    out[:, 1:] |= differs[:, :-1]
    out[:-1, :] |= differs[1:, :]
    out[1:, :] |= differs[:-1, :]
    return out


def _cell_bounds(nodes, value):
    """The [lo, hi) extent of the cell containing ``value`` on an axis, clamped
    into the realized domain."""
    i = int(np.searchsorted(nodes, float(value), side="right") - 1)
    i = min(max(i, 0), nodes.size - 2)
    return float(nodes[i]), float(nodes[i + 1])


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
    sweep = float(g.angle_stop_rad - g.angle_start_rad)
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
        rel = np.mod(np.arctan2(dv, du) - g.angle_start_rad, 2.0 * np.pi)
        inside = inside & (rel <= sweep)
    inside = np.broadcast_to(inside, HH.shape)
    return inside if inside.any() else None


def _polyslab_inside(g, cut_axis, value, HH, VV, h_idx, v_idx):
    """Boolean mask of the §9 hard sample for a ``Polygon`` on the cut plane,
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
    mirrors ``reference_solver.cpp::polyslab_contains``, without it the client
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
    """Vectorized UNSIGNED distance from ``(pu, pv)`` to the polygon's edge set ,
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


def plot_index(sim, x=None, y=None, z=None, *, ax=None, cmap=None,
             legend=True, grid=False, unfold=True, subpixel=None,
             supersample=None, **kw):
    """Render a heatmap of ε sampled on the mesh on the selected cut plane.
    Returns the matplotlib ``Axes``.

    Exactly one of x/y/z (microns) selects the constant-coordinate cut plane.
    The plot uses the real µm node coordinates (``pcolormesh``), so graded
    meshes render with correct, non-uniform cell widths. ``grid=True`` overlays
    the realized cell edges, useful to confirm features are well resolved.
    ``unfold`` (the default) mirrors the sampled half back across each in-plane
    §20 symmetry plane, so the heatmap shows the whole device; ``unfold=False``
    shows the reduced domain the solver steps.

    ``subpixel`` follows ``Simulation.subpixel`` by default, so the view shows
    what the run samples on the mesh: the §16 volume-fraction average when the run
    smooths, the §9 hard point sample when it does not. Force either with
    ``True``/``False``, and raise ``supersample`` for a finer fill fraction at
    proportional cost (see :func:`smooth_eps_plane` for what the scalar
    average does and does not reproduce)."""
    import matplotlib.pyplot as plt

    axis, value = geom.select_plane(x, y, z)
    owns_figure = ax is None
    if ax is None:
        _, ax = plt.subplots()
    # Which way round the plane reads best, decided once from the domain's own
    # proportions and then true of every helper below (design §5).
    spans = [sim._realized_um()[_AXES.index(l)] for l in geom.in_plane_axes(axis)]
    with geom.displayed_transposed(_style.prefer_long_axis_horizontal(*spans)):
        return _plot_index_on(ax, sim, axis, value, owns_figure, cmap=cmap,
                              legend=legend, grid=grid, unfold=unfold,
                              subpixel=subpixel, supersample=supersample, **kw)


def _plot_index_on(ax, sim, axis, value, owns_figure, *, cmap, legend, grid,
                   unfold, subpixel, supersample, **kw):
    """:func:`plot_index`'s body, with the plane's display order already
    settled so every helper it calls agrees on it."""
    before = set(ax.get_children())

    # The cut is given in the user's frame; the scene is sampled, reflected
    # and drawn in the corner frame, then moved into the user's (spec §4.4).
    a = geom.axis_index(axis)
    origin = geom.frame_origin(sim)
    oh, ov = geom.plane_offsets(origin, axis)
    value_c = value - origin[a]
    # Cut outside the domain -> empty Axes + warning, never raise (design §9).
    realized = sim._realized_um()
    if not (0.0 <= value_c <= realized[a]):
        warnings.warn(
            f"{axis}={value} um is outside the realized domain "
            f"[{origin[a]:.6g}, {origin[a] + realized[a]:.6g}] um on that axis; nothing to draw",
            UserWarning, stacklevel=2)
        _finish_axes(ax, axis, value, sim, None)
        return ax

    smoothed = bool(sim.subpixel) if subpixel is None else bool(subpixel)
    h_nodes, v_nodes, eps = sample_eps_plane(sim, axis, value_c,
                                             subpixel=smoothed,
                                             supersample=supersample)
    vmin, vmax = _style.eps_norm(
        [sim.background.permittivity]
        + [s.medium.permittivity for s in sim.structures]
    )
    mirror_h, mirror_v = _style.unfold_in_plane(sim, axis, unfold)
    mesh = None
    for sign_h, sign_v in _style.mirror_copies(mirror_h, mirror_v):
        # Each unfolded quadrant is the SAMPLED half reflected: negate the node
        # ladder and reverse it (pcolormesh wants monotonic coordinates), and
        # reverse the ε rows/columns to match.
        hn, block = (h_nodes, eps) if sign_h == 1 else (
            -h_nodes[::-1], eps[:, ::-1])
        vn, block = (v_nodes, block) if sign_v == 1 else (
            -v_nodes[::-1], block[::-1, :])
        mesh = ax.pcolormesh(hn, vn, block, cmap=cmap or _style.EPS_CMAP,
                             vmin=vmin, vmax=vmax, shading="flat", **kw)
    # The index axis and its ticks hang off the bar's LEFT side, so the bar
    # stands further off the picture than matplotlib's default would put it.
    cbar = ax.figure.colorbar(mesh, ax=ax, pad=_style.EPS_COLORBAR_PAD)
    # ε on one side, n on the other, a mark across the bar at each material.
    _style.finish_eps_colorbar(
        cbar, [sim.background.permittivity]
        + [s.medium.permittivity for s in sim.structures], vmin, vmax,
        "permittivity ε " + ("(subpixel average)" if smoothed
                             else "(hard sample)"))

    if grid:
        _style.draw_grid(ax, sim, axis, unfold=unfold)

    drew = _style.draw_overlays(ax, sim, axis, value_c, unfold=unfold)
    drew_symmetry = _style.draw_symmetry_planes(ax, sim, axis, mirror_h,
                                                mirror_v)
    # into the user's frame, the limits with it
    _style.translate_artists(ax, before, oh, ov)
    h_i, v_i = (_AXES.index(l) for l in geom.in_plane_axes(axis))
    ax.set_xlim(oh - (realized[h_i] if mirror_h else 0.0), oh + realized[h_i])
    ax.set_ylim(ov - (realized[v_i] if mirror_v else 0.0), ov + realized[v_i])
    if legend:
        _style.add_legend(ax, source=drew["source"], monitor=drew["monitor"],
                          pml=drew["pml"], structure=bool(sim.structures),
                          symmetry=drew_symmetry,
                          source_kinds=drew["source_kinds"],
                          monitor_kinds=drew["monitor_kinds"],
                          mode_monitor=drew["mode_monitor"],
                          flux_monitor=drew["flux_monitor"],
                          loc=_style.legend_corner(
                              _unfolded_occupancy(
                                  eps != float(sim.background.permittivity),
                                  mirror_h, mirror_v),
                              *[abs(hi - lo) for lo, hi in
                                (ax.get_xlim(), ax.get_ylim())]))
    _finish_axes(ax, axis, value, sim,
                 "subpixel average" if smoothed else "hard sample")
    if owns_figure:
        h_lo, h_hi = ax.get_xlim()
        v_lo, v_hi = ax.get_ylim()
        _style.size_figure_for_plane(ax, h_hi - h_lo, v_hi - v_lo)
    return ax


def _unfolded_occupancy(occ, mirror_h: bool, mirror_v: bool):
    """The occupancy mesh of the whole VIEW: the sampled half plus its
    mirror images, laid out as the picture is."""
    if mirror_h:
        occ = np.concatenate([occ[:, ::-1], occ], axis=1)
    if mirror_v:
        occ = np.concatenate([occ[::-1, :], occ], axis=0)
    return occ


def _finish_axes(ax, axis: str, value: float, sim, sample) -> None:
    h_ax, v_ax = geom.in_plane_axes(axis)
    ax.set_xlabel(f"{h_ax} (µm)")
    ax.set_ylabel(f"{v_ax} (µm)")
    realized = sim._realized_um()
    h_i, v_i = _AXES.index(h_ax), _AXES.index(v_ax)
    mirror_h, mirror_v = _style.unfold_in_plane(sim, axis, True)
    stretch = _style.set_view_aspect(ax, realized[h_i] * (2 if mirror_h else 1),
                                     realized[v_i] * (2 if mirror_v else 1))
    note = _style.join_notes(_style.mesh_summary(sim), sample,
                             _style.stretch_note(v_ax, stretch))
    _style.set_titles(ax, f"ε at {_style.cut_label(axis, value)}", note)


def plot_eps(*args, **kwargs):
    """Deprecated alias for :func:`plot_index` (renamed 2026-09)."""
    import warnings

    warnings.warn(
        "photonhub.viz.plot_eps was renamed to plot_index; the old name will be "
        "removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_index(*args, **kwargs)
