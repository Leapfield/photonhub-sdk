"""Polygon shapes for adjoint shape optimization with exact area averaging.

:class:`PolygonProfile` renders a polygon whose vertices move onto the pixel
boxes of a :class:`~photonhub.inverse_design.DesignRegion`: each pixel takes
the exact fraction of its area that the polygon covers, so the rendered
permittivity is a continuous, piecewise-smooth function of the vertex
positions (continuous material averaging, as in A. Michaels and
E. Yablonovitch, Opt. Express 26, 31717 (2018), Eq. 14 with two materials).
Its Jacobian is sparse and exact. :func:`curvature_penalty` is a
radius-of-curvature penalty for such vertex chains.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.special import expit

if TYPE_CHECKING:  # pragma: no cover
    from .inverse_design import DesignRegion

# Base vertices closer than this (in pixels) to a pixel face are placed on it,
# so a polygon drawn on the faces is not left a rounding error to one side.
_FACE_SNAP_PIXELS = 1e-9
# The rendered area may leave [0, 1] by rounding; beyond this the polygon
# overlaps itself.
_AREA_ROUNDING = 1e-9
# Forward-difference Jacobian entries below this, in pixels of area per pixel
# of displacement, are rounding of the column sums and are dropped.
_DIFFERENCE_FLOOR = 1e-6
# Candidate edge pairs the self-crossing test holds in memory at once.
_PAIR_BLOCK = 1 << 20


def _crossing_edges(points: np.ndarray) -> np.ndarray:
    """Pairs of non-adjacent edges of a closed polygon that touch or cross.

    Returns ``(k, 4)`` vertex indices into ``points``, the start and end of
    one edge, then of the other, ordered by the first edge. Repeated
    consecutive vertices are dropped first, so a zero-length edge does not
    make its two neighbours meet. A sweep along the axis on which fewer
    edges overlap finds every pair whose extents overlap on both axes; two
    such closed segments meet unless one has both ends of the other
    strictly on one side of its line, which also settles collinear pairs.
    The cost is ``O(n log n)`` plus the overlapping pairs, which are taken
    in blocks of at most ``_PAIR_BLOCK``."""
    keep = np.flatnonzero(np.any(points != np.roll(points, 1, axis=0), axis=1))
    n = keep.size
    if n < 4:
        return np.empty((0, 4), dtype=np.int64)
    nxt = np.roll(keep, -1)
    P = points[keep]
    Q = np.roll(P, -1, axis=0)
    lo, hi = np.minimum(P, Q), np.maximum(P, Q)
    sweeps = []
    for c in (0, 1):
        # Sorted by their low end, the edges overlapping edge i along c and
        # starting after it are the next ones, up to the first that starts
        # beyond its high end.
        order = np.argsort(lo[:, c], kind="stable")
        stop = np.searchsorted(lo[order, c], hi[order, c], side="right")
        sweeps.append((order, np.maximum(stop - np.arange(n) - 1, 0), 1 - c))
    order, count, other = min(sweeps, key=lambda sweep: int(sweep[1].sum()))

    def side(o, d, r):
        return np.sign(d[:, 0] * (r[:, 1] - o[:, 1]) - d[:, 1] * (r[:, 0] - o[:, 0]))

    found = []
    total = np.cumsum(count)
    first = 0
    while first < n:
        done = total[first - 1] if first else 0
        last = min(max(first + 1, int(np.searchsorted(total, done + _PAIR_BLOCK, side="right"))), n)
        span = count[first:last]
        size = int(span.sum())
        if size:
            pos = np.repeat(np.arange(first, last), span)
            ia = order[pos]
            ib = order[pos + 1 + np.arange(size) - np.repeat(np.cumsum(span) - span, span)]
            gap = (ia - ib) % n
            ok = ((gap != 1) & (gap != n - 1)
                  & (lo[ia, other] <= hi[ib, other]) & (lo[ib, other] <= hi[ia, other]))
            ia, ib = ia[ok], ib[ok]
            da, db = Q[ia] - P[ia], Q[ib] - P[ib]
            meet = ((side(P[ia], da, P[ib]) * side(P[ia], da, Q[ib]) <= 0)
                    & (side(P[ib], db, P[ia]) * side(P[ib], db, Q[ia]) <= 0))
            found.append(np.stack([np.minimum(ia, ib)[meet], np.maximum(ia, ib)[meet]], axis=1))
        first = last
    pairs = np.concatenate(found) if found else np.empty((0, 2), dtype=np.int64)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return np.stack([keep[pairs[:, 0]], nxt[pairs[:, 0]], keep[pairs[:, 1]], nxt[pairs[:, 1]]],
                    axis=1)


def _edge_pieces(anchor: np.ndarray, frac: np.ndarray):
    """Cut every edge of a closed polygon at the integer grid lines.

    Vertex ``m`` sits at ``anchor[m] + frac[m]`` in pixel units, ``anchor``
    integer. Each edge is handled in coordinates relative to the anchor of
    its start vertex, so a small displacement in ``frac`` keeps its full
    precision. Returns a dict of per-piece arrays: ``edge`` (start vertex),
    ``i``/``j`` (pixel), ``du`` (signed extent along the first axis), ``h``
    (mean height above the bottom of row ``j``), ``ta``/``tb`` (edge
    parameter range), ``on_u``/``on_v`` (piece lies on a grid line normal to
    that axis), and the edge vector ``D``.
    """
    m_count = anchor.shape[0]
    nxt = np.roll(np.arange(m_count), -1)
    P = frac
    Q = (anchor[nxt] - anchor).astype(float) + frac[nxt]
    D = Q - P
    edges = np.flatnonzero((D[:, 0] != 0) | (D[:, 1] != 0))
    e_parts, t_parts, u_parts, v_parts = [], [], [], []
    for e_end, t_end, pt in ((edges, 0.0, P), (edges, 1.0, Q)):
        e_parts.append(e_end)
        t_parts.append(np.full(e_end.size, t_end))
        u_parts.append(pt[e_end, 0])
        v_parts.append(pt[e_end, 1])
    for c in (0, 1):
        o = 1 - c
        lo = np.minimum(P[edges, c], Q[edges, c])
        hi = np.maximum(P[edges, c], Q[edges, c])
        first = np.floor(lo).astype(np.int64) + 1
        last = np.ceil(hi).astype(np.int64) - 1
        count = np.maximum(last - first + 1, 0)
        total = int(count.sum())
        if total == 0:
            continue
        owner = np.repeat(edges, count)
        start = np.repeat(np.cumsum(count) - count, count)
        k = np.repeat(first, count) + (np.arange(total) - start)
        t = (k - P[owner, c]) / D[owner, c]
        line = k.astype(float)
        other = P[owner, o] + t * D[owner, o]
        e_parts.append(owner)
        t_parts.append(t)
        u_parts.append(line if c == 0 else other)
        v_parts.append(other if c == 0 else line)
    e = np.concatenate(e_parts)
    t = np.concatenate(t_parts)
    u = np.concatenate(u_parts)
    v = np.concatenate(v_parts)
    order = np.lexsort((t, e))
    e, t, u, v = e[order], t[order], u[order], v[order]
    a = np.arange(e.size - 1)
    good = (e[a] == e[a + 1]) & (t[a + 1] > t[a])
    a = a[good]
    b = a + 1
    edge = e[a]
    um = 0.5 * (u[a] + u[b])
    vm = 0.5 * (v[a] + v[b])
    fi = np.floor(um)
    fj = np.floor(vm)
    on_u = (u[a] == u[b]) & (u[a] == np.floor(u[a]))
    on_v = (v[a] == v[b]) & (v[a] == np.floor(v[a]))
    return dict(
        edge=edge, nxt=nxt[edge],
        i=anchor[edge, 0] + fi.astype(np.int64),
        j=anchor[edge, 1] + fj.astype(np.int64),
        du=u[b] - u[a],
        h=np.clip(0.5 * (v[a] + v[b]) - fj, 0.0, 1.0),
        ta=t[a], tb=t[b], on_u=on_u, on_v=on_v, D=D[edge])


def _orientation(anchor: np.ndarray, frac: np.ndarray) -> float:
    """+1 for a counterclockwise polygon in pixel coordinates, -1 otherwise."""
    pts = (anchor - anchor[0]).astype(float) + frac
    nxt = np.roll(pts, -1, axis=0)
    area2 = float(np.sum(pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1]))
    if area2 == 0:
        raise ValueError("the polygon has zero area")
    return 1.0 if area2 > 0 else -1.0


def _render_planar(anchor, frac, n_u: int, n_v: int) -> np.ndarray:
    """Exact covered fraction of each unit pixel ``[i, i+1) x [j, j+1)``.

    By Green's theorem a closed polygon's indicator is the signed sum, over
    its edges, of the region below each edge. Each piece of an edge inside
    pixel ``(i, j)`` adds its trapezoid to that pixel and its full column
    width to every pixel below it; the second part is accumulated from the
    top of each column down."""
    s = _orientation(anchor, frac)
    pc = _edge_pieces(anchor, frac)
    i, j = pc["i"], pc["j"]
    weight = -s * pc["du"]
    col = (i >= 0) & (i < n_u)
    inside = col & (j >= 0) & (j < n_v)
    partial = np.bincount(i[inside] * n_v + j[inside], weights=weight[inside] * pc["h"][inside],
                          minlength=n_u * n_v).reshape(n_u, n_v)
    row = np.minimum(j, n_v) - 1
    below = col & (row >= 0)
    cover = np.bincount(i[below] * n_v + row[below], weights=weight[below],
                        minlength=n_u * n_v).reshape(n_u, n_v)
    return partial + np.cumsum(cover[:, ::-1], axis=1)[:, ::-1]


def _boundary_matrix(anchor, frac, n_u: int, n_v: int) -> sparse.csr_matrix:
    """Exact ``d(pixel area)/d(vertex coordinate)`` in pixel units.

    Moving a polygon's boundary changes the area inside a fixed pixel by the
    integral, over the boundary inside it, of the boundary's normal velocity
    (the transport theorem). A vertex drags its two edges with the hat
    weights ``t`` and ``1 - t``, so a piece ``[ta, tb]`` of an edge with
    vector ``D`` contributes ``s D_v int(phi)`` to ``d/du`` and
    ``-s D_u int(phi)`` to ``d/dv`` of its pixel, ``s`` the orientation.
    A piece lying on a grid line belongs to both pixels beside it; there the
    area is not differentiable and each takes half, the mean of the two
    one-sided derivatives. Rows are ``i * n_v + j``; columns ``2 m + c``."""
    s = _orientation(anchor, frac)
    pc = _edge_pieces(anchor, frac)
    ta, tb = pc["ta"], pc["tb"]
    i1 = 0.5 * (tb * tb - ta * ta)
    i0 = (tb - ta) - i1
    du_n = s * pc["D"][:, 1]          # weight of a move along u
    dv_n = -s * pc["D"][:, 0]         # weight of a move along v
    rows, cols, vals = [], [], []

    def emit(i, j, vertex, comp, value, scale=1.0):
        ok = (i >= 0) & (i < n_u) & (j >= 0) & (j < n_v) & (value != 0)
        rows.append(i[ok] * n_v + j[ok])
        cols.append(2 * vertex[ok] + comp)
        vals.append(scale * value[ok])

    for vertex, integral in ((pc["edge"], i0), (pc["nxt"], i1)):
        for comp, normal, on, axis in ((0, du_n, pc["on_u"], "i"), (1, dv_n, pc["on_v"], "j")):
            value = normal * integral
            i, j = pc["i"], pc["j"]
            emit(i[~on], j[~on], vertex[~on], comp, value[~on])
            # On a grid line the piece's own pixel is the one past the line.
            emit(i[on], j[on], vertex[on], comp, value[on], 0.5)
            if axis == "i":
                emit(i[on] - 1, j[on], vertex[on], comp, value[on], 0.5)
            else:
                emit(i[on], j[on] - 1, vertex[on], comp, value[on], 0.5)
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_u * n_v, 2 * anchor.shape[0]))


def curvature_radii(points_um) -> np.ndarray:
    """Radius of curvature at every interior point of an open chain, in um.

    At each point a quadratic parametric curve ``(x(t), y(t))`` passes
    through the point at ``t = 0`` and its two neighbours at ``t = -1`` and
    ``+1``, and the radius is that of the curve at ``t = 0``:
    ``R = (x'^2 + y'^2)^(3/2) / |x' y'' - y' x''|`` with
    ``x' = (x+ - x-)/2`` and ``x'' = x+ - 2 x0 + x-``. Three collinear
    points give ``inf``. Returns ``len(points) - 2`` radii.

    The fit spaces the three points equally in ``t``, so the radius assumes
    the two neighbours are roughly equally far from the point; otherwise it
    is biased. On a circle of radius ``R``, neighbours at -0.1 and +0.5 rad
    read 1.77 ``R``, and a sharp corner with one neighbour far away reads
    as a gentle bend."""
    pts = np.asarray(points_um, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3 or not np.all(np.isfinite(pts)):
        raise ValueError("points_um must be at least three finite (x, y) points")
    _, _, num, den = _quadratic_fit(pts)
    with np.errstate(divide="ignore"):
        return np.where(den == 0, np.inf, num ** 1.5 / np.abs(den))


def _quadratic_fit(pts):
    prev, mid, nxt = pts[:-2], pts[1:-1], pts[2:]
    d1 = 0.5 * (nxt - prev)
    d2 = nxt - 2.0 * mid + prev
    num = d1[:, 0] ** 2 + d1[:, 1] ** 2
    den = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return d1, d2, num, den


def curvature_penalty(points_um, *, r0_um: float, alpha: float, kappa_per_um: float
                      ) -> Tuple[float, np.ndarray]:
    """Logistic penalty on small radii of curvature along a vertex chain.

    ``P = alpha * sum_i 1 / (1 + exp(-kappa (r0 - R_i)))`` over the
    interior points of the chain, with ``R_i`` from
    :func:`curvature_radii`: a radius well below ``r0_um`` costs about
    ``alpha``, one well above costs nearly nothing, and ``kappa_per_um``
    sets how sharp the step is (``1/kappa`` is its width in um). This is
    the penalty of Michaels and Yablonovitch, Opt. Express 26, 31717 (2018),
    Appendix B, with their fit read as parameter values -1, 0 and 1 at the
    three points. Returns ``(P, dP/dpoints)``, the gradient exact with
    shape ``points_um.shape``; a straight run of three points has an
    infinite radius and contributes neither value nor gradient.

    Each ``R_i`` assumes the point's two neighbours are roughly equally far
    from it (see :func:`curvature_radii`): on a circle, neighbours at -0.1
    and +0.5 rad read 1.77 times the true radius. Where a movable chain
    meets a fixed vertex far away, as a taper meets a fixed guide, the
    corner there is therefore barely penalized; :meth:`PolygonProfile.pad_ends`
    adds fixed vertices along the guides at the chain's own spacing.
    """
    if not (np.isfinite(r0_um) and np.isfinite(alpha) and np.isfinite(kappa_per_um)
            and kappa_per_um > 0):
        raise ValueError("r0_um and alpha must be finite and kappa_per_um positive")
    pts = np.asarray(points_um, dtype=float)
    radii = curvature_radii(pts)
    d1, d2, num, den = _quadratic_fit(pts)
    bent = np.isfinite(radii)
    z = np.where(bent, kappa_per_um * (r0_um - np.where(bent, radii, 0.0)), -np.inf)
    value = float(alpha * np.sum(expit(z)))
    grad = np.zeros_like(pts)
    live = bent & (z > -700.0)
    if not np.any(live):
        return value, grad
    d1, d2, num, den, z = d1[live], d2[live], num[live], den[live], z[live]
    k = np.abs(den) / num ** 1.5
    # d sigma / d k = sigma'(z) * kappa / k**2, since R = 1/k.
    scale = alpha * expit(z) * expit(-z) * kappa_per_um / k ** 2
    sign = np.sign(den)
    dk_dden = sign / num ** 1.5
    dk_dnum = -1.5 * np.abs(den) / num ** 2.5
    xp, yp = d1[:, 0], d1[:, 1]
    xpp, ypp = d2[:, 0], d2[:, 1]
    # Partials of den = x' y'' - y' x'' and num = x'^2 + y'^2 with respect
    # to the previous, middle and next point.
    dden = {("prev", 0): -0.5 * ypp - yp, ("mid", 0): 2.0 * yp, ("next", 0): 0.5 * ypp - yp,
            ("prev", 1): xp + 0.5 * xpp, ("mid", 1): -2.0 * xp, ("next", 1): xp - 0.5 * xpp}
    dnum = {("prev", 0): -xp, ("mid", 0): 0.0 * xp, ("next", 0): xp,
            ("prev", 1): -yp, ("mid", 1): 0.0 * yp, ("next", 1): yp}
    centre = np.flatnonzero(live) + 1
    for where, offset in (("prev", -1), ("mid", 0), ("next", 1)):
        for c in (0, 1):
            dk = dk_dden * dden[(where, c)] + dk_dnum * dnum[(where, c)]
            np.add.at(grad[:, c], centre + offset, scale * dk)
    return value, grad


@dataclass(frozen=True)
class PolygonProfile:
    """A polygon rendered as exact area fractions on a pixel-box region.

    The polygon lies in the plane of ``plane = (a, b)``, two of x, y and z,
    and is extruded through the region along the third axis. Each pixel of
    ``region`` takes as its density the exact fraction of its box (face
    offset included) that the polygon covers, the continuous material
    averaging of Michaels and Yablonovitch, Opt. Express 26, 31717 (2018),
    Eq. 14, for two materials. Moving a vertex changes the density only in
    the pixels its two edges cross, continuously everywhere (the render has
    no staircase jumps) and with a continuous derivative, except for a move
    across a pixel face of an edge lying on that face (see below); the second
    derivative jumps where the vertex crosses a pixel face or one of its
    edges sweeps over a pixel corner, so a central difference across such a
    point converges at first order only.

    ``vertices_um`` lists the base vertices as ``(a, b)`` pairs in microns.
    Without ``mirror`` they are the whole polygon, closed from the last
    vertex back to the first. With ``mirror = (axis, value)``, ``axis`` one
    of the plane's axes, they are an open chain and the polygon is the
    chain followed by its mirror image about ``axis = value`` in reverse
    order, so the shape is symmetric by construction; a chain vertex and
    its image move together, mirrored. The parameters are vertex
    displacements in microns, zero at the base polygon: for each vertex
    index in ``movable`` (default every vertex, or every chain vertex), one
    displacement along each axis in ``move`` (default both plane axes, in
    plane order), so ``p.reshape(len(movable), len(move))`` with the
    default order. Vertices not listed stay fixed, for example the ends of
    a taper where it meets fixed guides.

    :meth:`density` renders, :meth:`jacobian` returns the exact
    ``d density / d p`` as a ``scipy.sparse`` matrix of shape
    ``(region.n_params, num_params)`` in flat ``density(...).ravel()`` order;
    pass the two as ``expand`` and ``jacobian`` to
    :func:`parametric_value_and_gradient` and :func:`optimize_parametric`,
    which chain the sparse matrix without densifying it.
    :meth:`difference_jacobian` is the forward difference of the exact
    render at a tiny step, for checking. The area of a pixel is not
    differentiable when an edge lies exactly on a pixel face and moves
    across it; :meth:`jacobian` then returns the mean of the two one-sided
    derivatives. A base vertex within 1e-9 pixel of a face is placed on
    it.

    The region must use pixel boxes with quarter-cell faces (the
    :class:`DesignRegion` default); build it with :meth:`on_grid`, which
    merges equal pixels into a few boxes (``merge_boxes``) so a large region
    stays cheap to set up. A polygon that leaves the region is cut at its
    boundary; structures outside the region are the caller's. The polygon
    must be simple: :meth:`planar_density`, :meth:`density` and
    :meth:`jacobian` test every pair of non-adjacent edges and raise,
    naming the first pair, when two touch or cross (for 400 vertices a
    small fraction of the render's cost); pass ``check_simple=False`` to
    skip the test when the polygon is known to stay simple. They also raise
    when a pixel's covered fraction leaves ``[0, 1]``. With ``mirror`` the
    test covers the chain and its image, so a chain crossing the axis
    raises too.
    """

    region: "DesignRegion"
    vertices_um: Tuple[Tuple[float, float], ...]
    plane: Tuple[str, str] = ("x", "y")
    movable: Optional[Tuple[int, ...]] = None
    move: Optional[str] = None
    mirror: Optional[Tuple[str, float]] = None

    def __post_init__(self) -> None:
        plane = tuple(self.plane)
        if (len(plane) != 2 or any(a not in ("x", "y", "z") for a in plane)
                or plane[0] == plane[1]):
            raise ValueError("plane must be two different axes among x, y, z")
        object.__setattr__(self, "plane", plane)
        if self.region.parametrization != "boxes" or self.region.box_face_offset_cells != 0.25:
            raise ValueError("PolygonProfile renders pixel boxes with quarter-cell faces; "
                             "use parametrization='boxes' and box_face_offset_cells=0.25")
        verts = np.asarray(self.vertices_um, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < (2 if self.mirror else 3):
            raise ValueError("vertices_um must be (a, b) pairs: at least three, or two "
                             "for a mirrored chain")
        if not np.all(np.isfinite(verts)):
            raise ValueError("vertices_um must be finite")
        object.__setattr__(self, "vertices_um", tuple(tuple(float(c) for c in v) for v in verts))
        if self.mirror is not None:
            axis, value = self.mirror
            if axis not in plane or not np.isfinite(value):
                raise ValueError("mirror must be (axis, value) with axis one of the plane's axes")
            object.__setattr__(self, "mirror", (axis, float(value)))
        n = verts.shape[0]
        movable = tuple(range(n)) if self.movable is None else tuple(int(k) for k in self.movable)
        if not movable or len(set(movable)) != len(movable) or any(k < 0 or k >= n for k in movable):
            raise ValueError("movable must list unique vertex indices")
        object.__setattr__(self, "movable", movable)
        move = "".join(plane) if self.move is None else self.move
        if not move or len(set(move)) != len(move) or any(c not in plane for c in move):
            raise ValueError(f"move must name one or both plane axes {plane}")
        object.__setattr__(self, "move", move)
        # Base polygon in pixel units: integer anchors and exact fractions.
        full = verts
        if self.mirror is not None:
            c = plane.index(self.mirror[0])
            image = verts[::-1].copy()
            image[:, c] = 2.0 * self.mirror[1] - image[:, c]
            full = np.concatenate([verts, image])
        lo, pitch = self._grid()
        pix = (full - lo) / pitch
        snapped = np.round(pix)
        pix = np.where(np.abs(pix - snapped) <= _FACE_SNAP_PIXELS, snapped, pix)
        anchor = np.floor(pix).astype(np.int64)
        object.__setattr__(self, "_anchor", anchor)
        object.__setattr__(self, "_base_frac", pix - anchor)
        object.__setattr__(self, "_map", self._parameter_map(n, full.shape[0]))
        _orientation(anchor, pix - anchor)

    # -- construction -------------------------------------------------------

    @classmethod
    def on_grid(
        cls,
        *,
        center_um: Tuple[float, float, float],
        size_um: Tuple[float, float, float],
        dl_um: float,
        vertices_um,
        plane: Tuple[str, str] = ("x", "y"),
        movable: Optional[Tuple[int, ...]] = None,
        move: Optional[str] = None,
        mirror: Optional[Tuple[str, float]] = None,
        eps_min: float = 1.0,
        eps_max: float = 12.25,
    ) -> "PolygonProfile":
        """Profile on a region of one cell per pixel in the plane and one
        pixel across the third axis, with equal pixels merged into boxes.

        The region snaps as :meth:`DesignRegion.on_grid` does. For a
        quasi-2D run (one periodic cell along the third axis) center the
        region on 0 along that axis with size ``2 * dl_um``: its
        quarter-cell faces at -0.75 and 1.25 cells then hold every Yee
        sample of the single plane, and ``region.monitor(freq_hz,
        simulation=sim)`` collapses its monitor onto that plane."""
        from .inverse_design import DesignRegion

        plane = tuple(plane)
        if len(plane) != 2 or any(a not in ("x", "y", "z") for a in plane) or plane[0] == plane[1]:
            raise ValueError("plane must be two different axes among x, y, z")
        idx = ["xyz".index(a) for a in plane]
        shape = tuple(max(int(round(s / dl_um)), 1) if a in idx else 1
                      for a, s in enumerate(size_um))
        region = DesignRegion.on_grid(
            center_um=center_um, size_um=size_um, dl_um=dl_um, shape=shape,
            eps_min=eps_min, eps_max=eps_max, merge_boxes=True)
        return cls(region=region, vertices_um=tuple(map(tuple, vertices_um)), plane=plane,
                   movable=movable, move=move, mirror=mirror)

    def pad_ends(self, *, spacing_um: float, count: int) -> "PolygonProfile":
        """The same profile with ``count`` fixed vertices added on each end
        edge of the chain, ``spacing_um`` apart, so the curvature penalty
        sees the corners where the chain meets its fixed ends.

        For a mirrored chain whose first and last vertices are fixed, for
        example far inside the guides a taper joins: the new vertices lie on
        the edge from vertex 1 back toward vertex 0, at ``k * spacing_um``
        from vertex 1 for ``k = 1 .. count``, and likewise from the
        second-to-last vertex toward the last. The polygon at ``p = 0`` and
        the order of ``p`` are unchanged. A move of vertex 1 or of the
        second-to-last vertex now tilts only the short edge to the nearest
        new vertex instead of the whole end edge, so the render away from
        ``p = 0`` and those vertices' Jacobian columns change; every other
        column is the same. With ``spacing_um`` about the movable vertices'
        own spacing, the radius at each junction no longer averages over a
        neighbour far away (see :func:`curvature_radii`)."""
        if self.mirror is None:
            raise ValueError("pad_ends needs a mirrored open chain")
        if not (np.isfinite(spacing_um) and spacing_um > 0) or int(count) != count or count < 1:
            raise ValueError("spacing_um must be positive and count a positive integer")
        count = int(count)
        verts = np.asarray(self.vertices_um, dtype=float)
        n = verts.shape[0]
        if n < 4 or 0 in self.movable or n - 1 in self.movable:
            raise ValueError("pad_ends needs at least four chain vertices, the first and "
                             "last of them fixed")
        k = spacing_um * np.arange(1, count + 1)[:, None]
        pads = []
        for inner, outer in ((1, 0), (n - 2, n - 1)):
            edge = verts[outer] - verts[inner]
            length = float(np.hypot(*edge))
            if count * spacing_um >= length:
                raise ValueError(f"{count} vertices {spacing_um} um apart do not fit on the "
                                 f"{length:.6g} um edge from vertex {inner} to vertex {outer}")
            pads.append(verts[inner] + k * edge / length)
        new = np.concatenate([verts[:1], pads[0][::-1], verts[1:n - 1], pads[1], verts[n - 1:]])
        return replace(self, vertices_um=tuple(map(tuple, new)),
                       movable=tuple(m + count for m in self.movable))

    # -- geometry -----------------------------------------------------------

    def _grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Low pixel face and pixel pitch along the two plane axes, um."""
        r = self.region
        idx = ["xyz".index(a) for a in self.plane]
        lo = np.array([(r.cells[a][0] + r.box_face_offset_cells) * r.dl_um for a in idx])
        pitch = np.array([r.cells_per_pixel[a] * r.dl_um for a in idx])
        return lo, pitch

    def _parameter_map(self, n_chain: int, n_full: int) -> sparse.csr_matrix:
        """``d (full-polygon pixel coordinates) / d p``, shape ``(2 n_full, n)``."""
        _, pitch = self._grid()
        rows, cols, vals = [], [], []
        mirror_c = self.plane.index(self.mirror[0]) if self.mirror is not None else None
        for slot, (k, axis) in enumerate((k, a) for k in self.movable for a in self.move):
            c = self.plane.index(axis)
            copies = [(k, 1.0)]
            if self.mirror is not None:
                copies.append((n_full - 1 - k, -1.0 if c == mirror_c else 1.0))
            for m, sign in copies:
                rows.append(2 * m + c)
                cols.append(slot)
                vals.append(sign / pitch[c])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(2 * n_full, self.num_params))

    @property
    def num_params(self) -> int:
        """Number of displacement parameters, ``len(movable) * len(move)``."""
        return len(self.movable) * len(self.move)

    @property
    def num_vertices(self) -> int:
        """Vertices of the full polygon, mirror image included."""
        return self._anchor.shape[0]

    def _params(self, p) -> np.ndarray:
        params = np.asarray(p, dtype=float)
        if params.shape != (self.num_params,) or not np.all(np.isfinite(params)):
            raise ValueError(f"p must be {self.num_params} finite displacements in um")
        return params

    def _frac(self, params: np.ndarray) -> np.ndarray:
        """Vertex fractions at ``params``: base fraction plus displacement."""
        return self._base_frac + (self._map @ params).reshape(-1, 2)

    def polygon_um(self, p) -> np.ndarray:
        """The full polygon at ``p``, shape ``(num_vertices, 2)`` in plane order."""
        lo, pitch = self._grid()
        return lo + (self._anchor + self._frac(self._params(p))) * pitch

    def chain_um(self, p) -> np.ndarray:
        """The vertices listed in ``vertices_um`` at ``p`` (the chain, without
        its mirror image), shape ``(len(vertices_um), 2)``."""
        return self.polygon_um(p)[:len(self.vertices_um)]

    def _shape(self) -> Tuple[int, int, int]:
        idx = ["xyz".index(a) for a in self.plane]
        third = 3 - sum(idx)
        return self.region.shape[idx[0]], self.region.shape[idx[1]], self.region.shape[third]

    def _to_region(self, planar: np.ndarray) -> np.ndarray:
        a, b = ("xyz".index(c) for c in self.plane)
        c = 3 - a - b
        n_c = self.region.shape[c]
        spread = np.broadcast_to(planar[:, :, None], planar.shape + (n_c,))
        return np.ascontiguousarray(np.transpose(spread, np.argsort([a, b, c])))

    def _region_rows(self) -> np.ndarray:
        """Flat region index of every ``(i, j, k)`` in plane order."""
        a, b = ("xyz".index(c) for c in self.plane)
        c = 3 - a - b
        index = np.arange(self.region.n_params).reshape(self.region.shape)
        return np.transpose(index, (a, b, c))

    def _vertex_name(self, m: int) -> str:
        n_chain = len(self.vertices_um)
        return f"vertex {m}" if m < n_chain else \
            f"the mirror image of vertex {2 * n_chain - 1 - m}"

    def _require_simple(self, frac: np.ndarray) -> None:
        """Raise, naming the first pair, if two non-adjacent edges meet."""
        pairs = _crossing_edges(self._anchor + frac)
        if pairs.size:
            a0, a1, b0, b1 = (self._vertex_name(int(m)) for m in pairs[0])
            more = f" ({pairs.shape[0]} such pairs)" if pairs.shape[0] > 1 else ""
            raise ValueError(
                f"the polygon overlaps itself at p: its edge from {a0} to {a1} touches "
                f"or crosses its edge from {b0} to {b1}{more}; pass check_simple=False "
                "only for a polygon known to stay simple")

    def planar_density(self, p, *, check_simple: bool = True) -> np.ndarray:
        """Covered fraction of each pixel in the plane, shape ``(n_a, n_b)``.
        Raises if two non-adjacent edges touch or cross (skipped with
        ``check_simple=False``) or a pixel's fraction leaves ``[0, 1]``."""
        n_a, n_b, _ = self._shape()
        frac = self._frac(self._params(p))
        if check_simple:
            self._require_simple(frac)
        rho = _render_planar(self._anchor, frac, n_a, n_b)
        if np.any(rho < -_AREA_ROUNDING) or np.any(rho > 1 + _AREA_ROUNDING):
            raise ValueError("the polygon overlaps itself at p: its rendered area leaves [0, 1]")
        return np.clip(rho, 0.0, 1.0)

    def density(self, p, *, check_simple: bool = True) -> np.ndarray:
        """Density grid of shape ``region.shape``: the exact covered fraction
        of each pixel box, the same along the third axis. Raises as
        :meth:`planar_density` does."""
        return self._to_region(self.planar_density(p, check_simple=check_simple))

    def jacobian(self, p, *, check_simple: bool = True) -> sparse.csr_matrix:
        """Exact ``d density / d p``, a ``scipy.sparse`` CSR matrix of shape
        ``(region.n_params, num_params)`` in flat ``density(...).ravel()``
        order, per micron of displacement. Raises if two non-adjacent edges
        touch or cross, unless ``check_simple=False``."""
        n_a, n_b, n_c = self._shape()
        frac = self._frac(self._params(p))
        if check_simple:
            self._require_simple(frac)
        g = _boundary_matrix(self._anchor, frac, n_a, n_b)
        return self._expand_rows((g @ self._map).tocoo(), n_b, n_c)

    def _expand_rows(self, planar: sparse.coo_matrix, n_b: int, n_c: int) -> sparse.csr_matrix:
        rows3 = self._region_rows()
        i, j = np.divmod(planar.row, n_b)
        rows = rows3[i, j, :].reshape(-1)
        cols = np.repeat(planar.col, n_c)
        vals = np.repeat(planar.data, n_c)
        return sparse.csr_matrix((vals, (rows, cols)),
                                 shape=(self.region.n_params, self.num_params))

    def difference_jacobian(self, p, step_um: Optional[float] = None) -> sparse.csr_matrix:
        """Forward difference of the exact render, ``(density(p + h e_j) -
        density(p)) / h``, as a sparse matrix like :meth:`jacobian`.

        ``step_um`` defaults to ``1e-8`` of a cell. Entries below 1e-6 of a
        pixel of area per pixel of displacement are rounding of the column
        sums and are dropped. For checking :meth:`jacobian`; it renders
        the polygon ``num_params + 1`` times and does not test that it is
        simple. A step that reverses the polygon's orientation, as one
        larger than the thickness of a sliver can, gives a wrong column: on
        a sliver 2^-30 pixel thick the default step is off by 160%, where
        :meth:`jacobian` stays exact."""
        params = self._params(p)
        h = 1e-8 * self.region.dl_um if step_um is None else float(step_um)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("step_um must be positive and finite")
        n_a, n_b, n_c = self._shape()
        base = _render_planar(self._anchor, self._frac(params), n_a, n_b).ravel()
        _, pitch = self._grid()
        rows, cols, vals = [], [], []
        for j in range(self.num_params):
            q = params.copy()
            q[j] += h
            col = (_render_planar(self._anchor, self._frac(q), n_a, n_b).ravel() - base) / h
            floor = _DIFFERENCE_FLOOR / pitch[self.plane.index(self.move[j % len(self.move)])]
            keep = np.flatnonzero(np.abs(col) > floor)
            rows.append(keep)
            cols.append(np.full(keep.size, j))
            vals.append(col[keep])
        planar = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(n_a * n_b, self.num_params))
        return self._expand_rows(planar, n_b, n_c)

    # -- curvature ----------------------------------------------------------

    def curvature_penalty(self, p, *, r0_um: float, alpha: float, kappa_per_um: float
                          ) -> Tuple[float, np.ndarray]:
        """:func:`curvature_penalty` of the vertex chain at ``p`` and its
        gradient with respect to ``p``.

        With ``mirror`` the chain is ``vertices_um`` (the image has the same
        radii and is not counted again); without it, the closed polygon,
        every vertex with its two neighbours. Fixed vertices count as
        neighbours and as points. Use it as ``penalty`` of
        :func:`optimize_parametric`::

            penalty=lambda p: profile.curvature_penalty(
                p, r0_um=0.15, alpha=..., kappa_per_um=...)
        """
        params = self._params(p)
        poly = self.polygon_um(params)
        n_chain = len(self.vertices_um)
        if self.mirror is not None:
            pts = poly[:n_chain]
            index = np.arange(n_chain)
        else:
            pts = np.concatenate([poly[-1:], poly, poly[:1]])
            index = np.concatenate([[n_chain - 1], np.arange(n_chain), [0]])
        value, g_pts = curvature_penalty(pts, r0_um=r0_um, alpha=alpha,
                                         kappa_per_um=kappa_per_um)
        g_vertex = np.zeros((poly.shape[0], 2))
        np.add.at(g_vertex, index, g_pts)
        # d(polygon_um)/dp = pitch * map; only the chain's own vertices count.
        _, pitch = self._grid()
        if self.mirror is not None:
            g_vertex[n_chain:] = 0.0
        grad = self._map.T @ (g_vertex * pitch).ravel()
        return value, np.asarray(grad, dtype=float)
