"""Compile :class:`~photonhub.components.monitors.FarFieldMonitor`
declarations into the face monitors the solver records, and check them.

Client side only: the faces are ordinary ``field_dft`` monitors, and the
recipe :func:`~photonhub.analysis.far_field` reads them back with (the faces,
their outward normals, the medium they sit in, the declared geometry) is
client state, written beside the run's wire document like declared ports
(:func:`~photonhub.components.frame.client_state`).

How a surface is recorded, on the grid lines ``q`` of each axis (with the
closing node, the mesher's lines as the solver realizes them):

* A face normal to ``a`` sits on the grid line ``k`` nearest its plane and
  is recorded one cell thick, base nodes ``k`` and ``k + 1`` (its region runs
  from the quarter point of cell ``k`` to that of cell ``k + 1``, which every
  component snaps to), so E and H can be brought onto one surface.
* Box faces cover base nodes ``lo - 1 .. hi + 1`` along each in-plane axis,
  one cell past the box, at every cell. A single plane covers the nodes
  nearest its extent (or the PML-free interior) and may take an in-plane
  stride from ``pts_per_wlen``.
* An open box (``exclude``) records the faces kept exactly as the closed box
  would, one cell past the box on the open side too; the recipe keeps the
  declared plane of each face left out, and ``far_field`` cuts the kept
  faces there (not on a grid line, which would move the open edge with the
  mesh).
* A plane's edge window spans the plane's declared extent (its
  ``size_um``, or the PML-free interior), mirrored across a symmetry
  plane.
* With symmetry planes (each at coordinate 0 of the run's frame, the
  domain's first grid line) only the kept part is recorded: along such an
  axis a face starts on the plane (node 0), and the box face on the far side
  is not recorded (``far_field`` reads it as the mirror image of the near
  one). The recipe keeps the planes.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..analysis._constants import _TANGENTIAL as FACE_FIELDS
from ..constants import c0 as _C0
from ._bounds import geometry_bounds_um
from .grid import _YEE_OFFSETS_BY_COMPONENT as _YEE_OFFSETS
from .grid import EngineSnap
from .monitors import FarFieldMonitor, ProfileMonitor, far_field_face_name

_AXES = "xyz"

#: The graded-mesh scope the projection's asymmetry is shown for (with
#: SOURCE_CLEARANCE_WAVELENGTHS): every box face at least this many
#: wavelengths (in its medium) from the nearest grid line where the cell
#: size changes, along its normal, on either side, measured from both of the
#: face's recorded sample planes.
FACE_CLEARANCE_WAVELENGTHS = 0.35
#: ... and every source at least this many wavelengths from the nearest
#: grid line where the cell size changes: a point source along each axis, a
#: source plane (a launch sheet, a plane wave, a mode source, a TF/SF face)
#: along its normal. Nearer, the field a source radiates is itself
#: asymmetric: in a vacuum dipole box at 30
#: background cells per wavelength the front/back asymmetry reaches 1.085 %
#: of the peak with the dipole just past the edge of a region refined three
#: times (0.92 % twice), against at most about 0.7 % for sources clear of
#: every change.
SOURCE_CLEARANCE_WAVELENGTHS = 0.1
#: A grid line where the two cells either side differ by more than this
#: fraction is a change of cell size. A grading gentler than this per cell
#: never counts, however far it stretches the cells in total; the graded
#: scope was measured on the SDK mesher's grading of up to 1.4 per cell.
CHANGE_RTOL = 0.01
#: A source or structure this many cells from a face's innermost recorded
#: sample: the readout's error grows sharply within about one cell.
NEAR_CELLS = 2


def _who(mon: FarFieldMonitor) -> str:
    return f"FarFieldMonitor {mon.name!r}"


def _side(axis: str, sign: float) -> str:
    return f"{'+' if sign > 0 else '-'}{axis}"


def _quarter(q: np.ndarray, i: int) -> float:
    """The quarter point of cell ``i``: a region edge there snaps to base node
    ``i`` on every Yee sublattice."""
    return float(q[i] + 0.25 * (q[i + 1] - q[i]))


def _nearest(q: np.ndarray, value: float) -> int:
    """The base node (not the closing one) nearest ``value``."""
    return int(np.argmin(np.abs(q[:-1] - float(value))))


def changes(q: np.ndarray, rtol: float = CHANGE_RTOL) -> np.ndarray:
    """The grid lines of ``q`` (with its closing node) where the cell size
    changes: the cells either side differ by more than ``rtol``."""
    d = np.diff(np.asarray(q, dtype=np.float64))
    if d.size < 2:
        return np.empty(0)
    return np.asarray(q)[np.nonzero(np.abs(d[1:] / d[:-1] - 1.0) > rtol)[0] + 1]


def distance_to_change(q: np.ndarray, lo: float, hi: Optional[float] = None) -> float:
    """The distance (um) from the interval ``lo .. hi`` (a point when ``hi``
    is None) along one axis to the nearest grid line of ``q`` where the cell
    size changes (:func:`changes`); 0 when one lies inside it, ``inf`` when
    the axis has none."""
    ch = changes(q)
    if not ch.size:
        return math.inf
    hi = lo if hi is None else hi
    return float(np.min(np.maximum(0.0, np.maximum(lo - ch, ch - hi))))


def face_clearance(q: np.ndarray, k: int) -> float:
    """A box face's clearance along its normal (um): from its two recorded
    sample planes, ``q[k]`` and ``q[k + 1]``, to the nearest grid line where
    the cell size changes, on either side; ``inf`` when there is none."""
    return min(distance_to_change(q, float(q[k])), distance_to_change(q, float(q[k + 1])))


def _cull(structures, lo, hi):
    """The structures whose outer bounds reach the region ``lo .. hi``."""
    out = []
    for st in structures:
        try:
            b = geometry_bounds_um(st.geometry)
        except (TypeError, ValueError, AttributeError):
            out.append(st)                      # never cull what cannot be bounded
            continue
        if all(b[a][1] >= lo[a] and b[a][0] <= hi[a] for a in range(3)):
            out.append(st)
    return out


class _Face:
    """One face being compiled: its normal axis and outward sign, its normal
    base node ``k`` and its in-plane base-node spans, its stride."""

    def __init__(self, mon: FarFieldMonitor, axis: str, sign: float, k: int,
                 spans: Dict[str, Tuple[int, int]], box: bool):
        self.mon, self.axis, self.sign, self.k, self.spans, self.box = mon, axis, sign, k, spans, box
        self.name = far_field_face_name(mon.name, axis, sign)
        self.stride = {a: 1 for a in _AXES}
        self.n: Optional[np.ndarray] = None

    @property
    def where(self) -> str:
        return f"face {self.name!r} (normal to {_side(self.axis, self.sign)})"

    def node_span(self, a: str) -> Tuple[int, int]:
        return (self.k, self.k + 1) if a == self.axis else self.spans[a]

    def region(self, q) -> Tuple[List[float], List[float]]:
        """The recorded slab, ``lo`` and ``hi`` per axis (um): from the first
        node to the end of the cell of the last (where its half-offset
        components sit)."""
        lo, hi = [], []
        for a in _AXES:
            i0, i1 = self.node_span(a)
            lo.append(float(q[a][i0]))
            hi.append(float(q[a][i1 + 1]))
        return lo, hi

    def innermost(self, q) -> Tuple[int, float]:
        """The base node and coordinate of the sample on the side away from
        the direction the face radiates into."""
        i = self.k if self.sign > 0 else self.k + 1
        return i, float(q[self.axis][i])

    def monitor(self, q) -> ProfileMonitor:
        center, size = [0.0] * 3, [0.0] * 3
        for ia, a in enumerate(_AXES):
            i0, i1 = self.node_span(a)
            e0, e1 = _quarter(q[a], i0), _quarter(q[a], i1)
            center[ia], size[ia] = 0.5 * (e0 + e1), e1 - e0
        stride = tuple(self.stride[a] for a in _AXES)
        return ProfileMonitor(
            name=self.name, center_um=tuple(center), size_um=tuple(size),
            fields=FACE_FIELDS[self.axis], freqs_hz=self.mon.freqs_hz,
            interval_space=None if stride == (1, 1, 1) else stride,
            apodization=self.mon.apodization)


def _lines(sim) -> Dict[str, np.ndarray]:
    """Each axis's grid lines with the closing node: the mesher's lines (a
    graded axis) or the uniform ladder, as the solver realizes them."""
    from ..viz.eps import axis_nodes_um
    return {a: np.asarray(axis_nodes_um(sim, i), dtype=np.float64) for i, a in enumerate(_AXES)}


def _symmetry(sim) -> Tuple[int, int, int]:
    return tuple(int(v) for v in (getattr(sim, "symmetry", None) or (0, 0, 0)))  # type: ignore[return-value]


def _user(sim, i: int, value: float) -> float:
    """A coordinate along axis ``i`` in the user's frame, for a message."""
    return float(value) + float(getattr(sim, "origin_um", (0.0, 0.0, 0.0))[i])


def _kept_end(sim, who: str, q: np.ndarray, i: int, lo_c: float, hi_c: float,
              what: str) -> int:
    """Along axis ``i``, which has a symmetry plane (coordinate 0 of the
    run's frame, its first grid line ``q[0]``), the base node the kept part
    of the extent ``lo_c .. hi_c`` ends on. The run records the kept side
    only and the far side is read as its mirror image, so the extent must
    start on the plane (within half its first cell) or reach across it
    symmetrically (its two ends nearest mirror-image grid lines)."""
    a = _AXES[i]
    tol = 0.5 * float(q[1] - q[0])
    plane = f"the symmetry plane at {a} = {_user(sim, i, 0.0):.6g} um"
    span = f"{a} = {_user(sim, i, lo_c):.6g} .. {_user(sim, i, hi_c):.6g} um"
    if hi_c <= tol:
        raise ValueError(
            f"{who}: {what} ({span}) lies on {plane} or in its mirrored half, which the "
            "run does not model. Place it on the kept side")
    if lo_c > tol:
        raise ValueError(
            f"{who}: {what} ({span}) does not reach {plane}. The sources there have mirror "
            "images on the other side, so a surface that encloses or faces them must reach "
            "across the plane: centre it on the plane, or start it on the plane")
    end = _nearest(q, hi_c)
    if lo_c < -tol and _nearest(q, -lo_c) != end:
        raise ValueError(
            f"{who}: {what} ({span}) crosses {plane} off centre. The run records the kept "
            "side only and the far side is read as its mirror image, so the part across "
            "the plane must mirror the part on the kept side: centre it on the plane, or "
            "start it on the plane")
    return end


def _faces(sim, mon: FarFieldMonitor, q) -> List[_Face]:
    """The faces of ``mon`` on the grid lines ``q``, before any check (the
    faces a box leaves out among them). With symmetry planes, the kept part
    only: along a mirrored axis a face starts on the plane (node 0), and the
    box face on the far side of the plane is not among them."""
    who = _who(mon)
    n_cells = {a: q[a].size - 1 for a in _AXES}
    sym = _symmetry(sim)
    if mon.kind == "box":
        flat = [a for a in _AXES if n_cells[a] == 1]
        if flat:
            raise ValueError(
                f"{who}: axis {flat[0]!r} is one cell deep (a quasi-2-D scene), so a box "
                "around the source cannot close along it; a 2-D near-to-far projection "
                "needs the 2-D Green's function, which far_field does not implement. Use a "
                "full 3-D scene, or a FarFieldMonitor.plane")
        lo, hi = {}, {}
        for i, a in enumerate(_AXES):
            c, s = float(mon.center_um[i]), float(mon.size_um[i])
            if sym[i]:
                # the kept half: from the plane (node 0) to the plus face,
                # which lies past the plane's first half cell, off node 0
                lo[a] = 0
                hi[a] = _kept_end(sim, who, q[a], i, c - 0.5 * s, c + 0.5 * s, "the box")
                if hi[a] + 2 > n_cells[a]:
                    raise ValueError(
                        f"{who}: the box reaches the edge of the domain along {a} "
                        f"({c - 0.5 * s:.6g} to {c + 0.5 * s:.6g} um); its faces need a cell "
                        "past the box on every side inside the domain")
                continue
            lo[a], hi[a] = _nearest(q[a], c - 0.5 * s), _nearest(q[a], c + 0.5 * s)
            if hi[a] - lo[a] < 2:
                raise ValueError(
                    f"{who}: the box spans {hi[a] - lo[a]} cell(s) along {a} on this grid "
                    f"({c - 0.5 * s:.6g} to {c + 0.5 * s:.6g} um); it needs at least 2")
            # one cell past the box in plane (node lo - 1 .. hi + 1, whose
            # half-offset components sit in cell hi + 1), and the plus face's
            # second sample on node hi + 1
            if lo[a] - 1 < 0 or hi[a] + 2 > n_cells[a]:
                raise ValueError(
                    f"{who}: the box reaches the edge of the domain along {a} "
                    f"({c - 0.5 * s:.6g} to {c + 0.5 * s:.6g} um); its faces need a cell "
                    "past the box on every side inside the domain")
        faces = []
        for ia, a in enumerate(_AXES):
            # across a symmetry plane a face starts on it; elsewhere one cell
            # past the box
            spans = {t: (0 if sym[_AXES.index(t)] else lo[t] - 1, hi[t] + 1)
                     for t in _AXES if t != a}
            for kp, sign in ((lo[a], -1.0), (hi[a], 1.0)):
                if sign < 0 and sym[ia]:
                    continue                   # the mirror image of the plus face
                faces.append(_Face(mon, a, sign, kp, spans, box=True))
        return faces
    a = mon.axis
    ia = _AXES.index(a)
    k = _nearest(q[a], mon.position_um)
    if sym[ia] and k == 0:
        raise ValueError(
            f"{who}: the plane at {a} = {_user(sim, ia, mon.position_um):.6g} um lies on the "
            f"symmetry plane at {a} = {_user(sim, ia, 0.0):.6g} um or in its mirrored half, "
            "which the run does not model; a plane normal to a symmetry plane must sit on "
            "the kept side, past its first cell")
    if k + 2 > n_cells[a]:
        raise ValueError(
            f"{who}: the plane at {a} = {mon.position_um:.6g} um sits on the last cell of "
            "the domain; it is recorded one cell thick and needs the next cell inside the domain")
    spans = {}
    for i, t in enumerate(_AXES):
        if t == a:
            continue
        if mon.size_um is None:
            open_lo, open_hi, _kind = sim._nonabsorbing_bounds_um(i)
            tol = 1e-9 * max(1.0, abs(open_hi))
            nodes = np.nonzero(q[t][:-1] >= open_lo - tol)[0]
            ends = np.nonzero(q[t][1:] <= open_hi + tol)[0]
            if nodes.size == 0 or ends.size == 0 or ends[-1] < nodes[0]:
                raise ValueError(f"{who}: the domain interior along {t} holds no cell")
            spans[t] = (int(nodes[0]), int(ends[-1]))
        elif sym[i]:
            c, s = float(mon.center_um[i]), float(mon.size_um[i])
            spans[t] = (0, _kept_end(sim, who, q[t], i, c - 0.5 * s, c + 0.5 * s, "the plane"))
        else:
            c, s = float(mon.center_um[i]), float(mon.size_um[i])
            i0, i1 = _nearest(q[t], c - 0.5 * s), _nearest(q[t], c + 0.5 * s)
            if i1 <= i0:
                raise ValueError(
                    f"{who}: the plane spans no cell along {t} on this grid "
                    f"({c - 0.5 * s:.6g} to {c + 0.5 * s:.6g} um)")
            spans[t] = (i0, i1)
    return [_Face(mon, a, 1.0 if mon.direction == "+" else -1.0, k, spans, box=False)]


def _check_layers(sim, face: _Face, q) -> None:
    """Refuse a face that reaches into a PML or absorber layer, or past the
    domain."""
    lo, hi = face.region(q)
    for i, a in enumerate(_AXES):
        open_lo, open_hi, kind = sim._nonabsorbing_bounds_um(i)
        tol = 1e-9 * max(1.0, abs(open_hi))
        if lo[i] < open_lo - tol or hi[i] > open_hi + tol:
            layer = {"pml": "PML", "absorber": "absorber"}.get(kind)
            what = (f"the {layer} layers, which leave {a} = {open_lo:.6g} .. {open_hi:.6g} um"
                    if layer else f"the domain, {a} = {open_lo:.6g} .. {open_hi:.6g} um")
            raise ValueError(
                f"{_who(face.mon)}: {face.where} records {a} = {lo[i]:.6g} .. {hi[i]:.6g} um, "
                f"which reaches past {what}. A face inside an absorbing layer records "
                "fields the layer has already damped; make the surface smaller, or the "
                "domain larger")


def _sample_medium(sim, face: _Face, q, freqs: np.ndarray) -> np.ndarray:
    """The index of the one medium the face's recorded slab sits in, per
    frequency; ``ValueError`` naming the face when the slab crosses a
    structure or a medium boundary, or sits in a medium the projection
    cannot use. The slab's cells (the cells of its two samples along the
    normal, every cell of its extent in plane) are sampled at their centres
    with the sampling ``far_field`` and ``diffraction_orders`` share."""
    from ..analysis.diffraction import _infer_n_medium, _plane_media
    from ..analysis.near_field import _LOSS_TANGENT_MAX, _loss_tangent, _unsupported_medium
    from ..viz import _geometry as _geom

    who = f"{_who(face.mon)}: {face.where}"
    axis = face.axis
    h, v = _geom.in_plane_axes(axis)
    centres = {}
    for t in (h, v):
        i0, i1 = face.node_span(t)
        nodes = q[t][i0:i1 + 2]
        centres[t] = 0.5 * (nodes[:-1] + nodes[1:])
    points = tuple(np.meshgrid(centres[h], centres[v]))
    lo, hi = face.region(q)
    view = SimpleNamespace(background=sim.background,
                           structures=_cull(sim.structures, lo, hi))
    found = []
    for cell in (face.k, face.k + 1):
        pos = 0.5 * float(q[axis][cell] + q[axis][cell + 1])
        try:
            n_s, _ = _infer_n_medium(view, axis, pos, freqs, wording="far_field", points=points)
        except ValueError as exc:
            raise ValueError(
                f"{who} crosses a structure or a medium boundary at {axis} = {pos:.6g} um: "
                f"{exc}. Move the surface into one homogeneous medium") from None
        media = [m for m in _plane_media(view, axis, pos, points) if m is not None]
        for m in media:
            why = _unsupported_medium(m)
            if why:
                raise ValueError(
                    f"{who} sits in {why} medium at {axis} = {pos:.6g} um, and a far-field "
                    "projection needs an isotropic dielectric")
        tan_d = max((_loss_tangent(m, freqs) for m in media), default=0.0)
        if tan_d > _LOSS_TANGENT_MAX:
            raise ValueError(
                f"{who} sits in a lossy medium at {axis} = {pos:.6g} um (Im eps / Re eps up "
                f"to {tan_d:.3g}, above {_LOSS_TANGENT_MAX:g}); the projection ignores the "
                "absorption on the way to the far zone. Move the surface into a lossless medium")
        if not np.all(n_s >= 1.0):
            raise ValueError(f"{who} sits in a medium of index below 1 at {axis} = {pos:.6g} um; "
                             "far_field projects into a dielectric of index >= 1")
        found.append((pos, n_s))
    if not np.allclose(found[0][1], found[1][1], rtol=1e-9, atol=0.0):
        raise ValueError(
            f"{who} crosses a medium boundary along its normal: index "
            f"{_fmt(found[0][1])} at {axis} = {found[0][0]:.6g} um, {_fmt(found[1][1])} at "
            f"{found[1][0]:.6g} um. Move the surface into one homogeneous medium")
    return found[0][1]


def _fmt(n) -> str:
    arr = np.atleast_1d(np.asarray(n, dtype=np.float64))
    if arr.size and np.all(arr == arr[0]):
        return f"{float(arr[0]):.6g}"
    return f"{float(arr.min()):.6g}..{float(arr.max()):.6g}"


def _stride(face: _Face, q, wlen_um: float) -> None:
    """A plane's in-plane stride: the largest whole number of cells that
    keeps ``pts_per_wlen`` samples per wavelength ``wlen_um`` (its medium's,
    the shortest projected) across the largest cell of its extent."""
    ppw = face.mon.pts_per_wlen
    if ppw is None:
        return
    for t in _AXES:
        if t == face.axis:
            continue
        i0, i1 = face.node_span(t)
        widest = float(np.max(np.diff(q[t][i0:i1 + 2])))
        face.stride[t] = max(1, int(math.floor(wlen_um / (ppw * widest) + 1e-9)))


def _point_label(sim, c) -> str:
    """A point in the user's frame, for a message."""
    o = getattr(sim, "origin_um", (0.0, 0.0, 0.0))
    return "(" + ", ".join(f"{float(v) + float(o[a]):.4g}" for a, v in enumerate(c)) + ") um"


def _snapped_label(origin, declared, snapped) -> str:
    """Where the solver runs a source, for a message: the coordinates
    ``snapped`` (um, the corner frame; ``origin`` shifts each into the
    user's) to 4 decimals, more where needed until they differ from
    ``declared`` printed the same way, so the label never repeats the
    declared position."""
    o = [float(v) for v in origin]

    def fixed(c, d):
        return ", ".join(f"{float(v) + o[a]:.{d}f}" for a, v in enumerate(c))

    d = 4
    while d < 12 and fixed(snapped, d) == fixed(declared, d):
        d += 1
    return fixed(snapped, d)


def _yee_point(sim, q, snap: EngineSnap, k: int) -> List[float]:
    """Where the engine runs point dipole ``k`` (um, the corner frame): the
    node of its component's Yee sublattice it snaps to along each axis,
    on the grid lines ``q``."""
    src = sim.sources[k]
    off = _YEE_OFFSETS[src.polarization]
    out = []
    for a in range(3):
        qa = q[_AXES[a]]
        i = snap.node(a, src.center_um[a], off[a])
        out.append(float(qa[i] + off[a] * (qa[i + 1] - qa[i])))
    return out


def _plane_span(q, axis: str, k0: int, before: bool) -> Tuple[float, float]:
    """Along its normal, where a TF/SF plane injects (um): its E row on grid
    line ``k0`` and the H row of its correction half a cell before it
    (``before``: the side below ``k0``; else above)."""
    qa = q[axis]
    k0 = min(max(int(k0), 1), qa.size - 2)       # validate refuses a plane outside
    if before:
        return float(qa[k0 - 1] + 0.5 * (qa[k0] - qa[k0 - 1])), float(qa[k0])
    return float(qa[k0]), float(qa[k0] + 0.5 * (qa[k0 + 1] - qa[k0]))


def _source_regions(sim, q) -> List[Tuple[str, List[float], List[float], Tuple[int, ...]]]:
    """``(label, lo, hi, axes)`` of where each source injects (um, the
    corner frame), with the axes along which its position is checked
    against a change of cell size: a point dipole (all three axes); a sheet
    of dipoles, the wire form of a port's or a beam's launch (the axis it is
    thin along); a plane wave's or mode source's plane (its normal; unbounded
    in it); each face of a TF/SF box (its normal). Every source the
    simulation runs is seen, the launches that ``ports=`` and ``source=``
    add included. Labels name a source by kind and place, not by index.

    Measured where the solver puts each source on the grid lines ``q``
    (:class:`~photonhub.components.grid.EngineSnap`, the indices the
    solver runs at), not where it is declared: each dipole at
    the node of its component's Yee sublattice nearest its centre, ties
    half away from zero (a dipole declared on a grid node two cells from a
    sample runs half a cell nearer it along its polarization); a plane
    wave's or mode source's plane, and each face of a TF/SF box, on its
    snapped grid line with the H row half a cell before it on the side the
    corrections take."""
    snap = EngineSnap(sim)
    origin = getattr(sim, "origin_um", (0.0, 0.0, 0.0))
    out = []
    for first, last in sim._dipole_groups():
        pts = [_yee_point(sim, q, snap, k) for k in range(first, last + 1)]
        lo = [min(p[a] for p in pts) for a in range(3)]
        hi = [max(p[a] for p in pts) for a in range(3)]
        if first == last:
            declared = [float(v) for v in sim.sources[first].center_um]
            label = f"the point dipole at {_point_label(sim, declared)}"
            if any(abs(lo[a] - declared[a]) > 1e-9 for a in range(3)):
                label += (f" (run at ({_snapped_label(origin, declared, lo)}) um, the nearest "
                          f"{sim.sources[first].polarization} point of the grid)")
            out.append((label, lo, hi, (0, 1, 2)))
            continue
        widths = [(hi[a] - lo[a]) / sim._coarsest_cell_um(a) for a in range(3)]
        thin = int(np.argmin(widths))
        out.append((f"the {sim._dipole_group_label(first, last)}", lo, hi, (thin,)))
    for src in sim.sources:
        kind = getattr(src, "type", type(src).__name__)
        if kind in ("plane_wave", "mode_source"):
            a = _AXES.index(src.axis)
            k0 = snap.plane(a, src.position_um)
            lo, hi = [-math.inf] * 3, [math.inf] * 3
            # the TF side lies past k0 in the direction of travel; the H
            # correction half a cell before it
            lo[a], hi[a] = _plane_span(q, src.axis, k0, before=src.direction == "+")
            where = float(src.position_um) + float(origin[a])
            label = f"the {kind.replace('_', ' ')} at {src.axis} = {where:.4g} um"
            plane = float(q[src.axis][min(max(k0, 0), q[src.axis].size - 1)])
            if abs(plane - float(src.position_um)) > 1e-9:
                line = _snapped_label([origin[a]], [float(src.position_um)], [plane])
                label += f" (injected on the grid line {src.axis} = {line} um)"
            out.append((label, lo, hi, (a,)))
        elif kind == "tfsf_box":
            klo, khi = snap.tfsf_planes(src.center_um, src.size_um)
            for a in range(3):
                for sg in (-1.0, 1.0):
                    lo = [float(q[_AXES[j]][min(max(klo[j], 0), q[_AXES[j]].size - 1)])
                          for j in range(3)]
                    hi = [float(q[_AXES[j]][min(max(khi[j], 0), q[_AXES[j]].size - 1)])
                          for j in range(3)]
                    # the face's E row on its plane, the H row half a cell
                    # outside the box
                    k = klo[a] if sg < 0 else khi[a]
                    lo[a], hi[a] = _plane_span(q, _AXES[a], k, before=sg < 0)
                    out.append((f"the TF/SF box's {_side(_AXES[a], sg)} face", lo, hi, (a,)))
    return out


def _structure_label(sim, i: int) -> str:
    st = sim.structures[i]
    return f"structure {i}" + (f" ({st.name!r})" if getattr(st, "name", None) else "")


def _box_gap(lo1, hi1, lo2, hi2) -> float:
    """Distance between two axis-aligned boxes (0 when they touch or
    overlap)."""
    return float(math.sqrt(sum(max(0.0, lo2[a] - hi1[a], lo1[a] - hi2[a]) ** 2 for a in range(3))))


def _advice(sim, faces: Sequence[_Face], q, wlen_um: float, sources) -> List[str]:
    """The build warnings of one far-field surface (messages)."""
    mon = faces[0].mon
    who = _who(mon)
    out: List[str] = []
    bounds = []
    for i, st in enumerate(sim.structures):
        try:
            b = geometry_bounds_um(st.geometry)
        except (TypeError, ValueError, AttributeError):
            continue
        bounds.append((i, [b[a][0] for a in range(3)], [b[a][1] for a in range(3)]))
    half = 0.5 * wlen_um
    for face in faces:
        a = _AXES.index(face.axis)
        lo, hi = face.region(q)
        k_in, s_in = face.innermost(q)
        qa = q[face.axis]
        above = float(qa[min(k_in + NEAR_CELLS, qa.size - 1)] - s_in)
        below = float(s_in - qa[max(k_in - NEAR_CELLS, 0)])
        near = []
        # in plane: within the face's extent widened by two of its cells
        wide_lo, wide_hi = list(lo), list(hi)
        for t in range(3):
            if t == a:
                continue
            qt = q[_AXES[t]]
            i0, i1 = face.node_span(_AXES[t])
            wide_lo[t] = float(qt[max(i0 - NEAR_CELLS, 0)])
            wide_hi[t] = float(qt[min(i1 + 1 + NEAR_CELLS, qt.size - 1)])

        def gap_along(slo, shi):
            if any(shi[t] < wide_lo[t] or slo[t] > wide_hi[t] for t in range(3) if t != a):
                return None
            if slo[a] > s_in:
                return float(slo[a] - s_in), above
            if shi[a] < s_in:
                return float(s_in - shi[a]), below
            return 0.0, above

        for label, slo, shi, _axes in sources:
            g = gap_along(slo, shi)
            if g is not None and g[0] < g[1]:
                near.append(f"{label} at {g[0]:.3g} um")
        for i, blo, bhi in bounds:
            inside = all(blo[t] <= lo[t] and bhi[t] >= hi[t] for t in range(3))
            if inside:
                continue                         # the face sits within it (its medium)
            g = gap_along(blo, bhi)
            if g is not None and g[0] < g[1]:
                near.append(f"{_structure_label(sim, i)} at {g[0]:.3g} um")
        if near:
            out.append(
                f"{who}: {face.where} has {', '.join(near)} from its innermost sample "
                f"({face.axis} = {s_in + float(getattr(sim, 'origin_um', (0, 0, 0))[a]):.6g} um), "
                f"within {NEAR_CELLS} cells; the projection's "
                "error grows sharply with a source or a structure that close to a recorded "
                "sample. Move the surface away from it")
            continue
        closest = None
        for i, blo, bhi in bounds:
            if all(blo[t] <= lo[t] and bhi[t] >= hi[t] for t in range(3)):
                # the face inside a structure (its medium): to that structure's boundary
                d = min(min(lo[t] - blo[t], bhi[t] - hi[t]) for t in range(3))
            else:
                d = _box_gap(lo, hi, blo, bhi)
            if d < half and (closest is None or d < closest[0]):
                closest = (d, i)
        if closest is not None:
            out.append(
                f"{who}: {face.where} is {closest[0]:.3g} um from {_structure_label(sim, closest[1])} "
                f"(its bounding box), closer than half a wavelength in the face's medium "
                f"({half:.3g} um): the evanescent near field there is not radiation, and the "
                "projection reads it as if it were. Move the surface away from the structure")
    # graded meshes: changes of cell size near a box face's normal samples,
    # or near a source
    clear = FACE_CLEARANCE_WAVELENGTHS * wlen_um
    for face in faces:
        if not face.box:
            continue
        d = face_clearance(q[face.axis], face.k)
        if d < clear * (1.0 - 1e-9):        # at the threshold itself: covered
            out.append(
                f"{who}: {face.where} is {d:.3g} um ({d / wlen_um:.3g} wavelengths in its "
                f"medium) from a grid line where the cell size changes, along {face.axis}. "
                f"{_COVERED} A box face is covered when it is at least "
                f"{FACE_CLEARANCE_WAVELENGTHS:g} wavelengths ({clear:.3g} um) from such a change, "
                "measured along its normal from both of its recorded samples, on either "
                "side: end a refinement inside the box with that room to spare, or carry it "
                "that far past every face")
    near_change = SOURCE_CLEARANCE_WAVELENGTHS * wlen_um
    for label, slo, shi, axes in sources:
        hits = []
        for i in axes:
            if not (math.isfinite(slo[i]) and math.isfinite(shi[i])):
                continue
            d = distance_to_change(q[_AXES[i]], slo[i], shi[i])
            if d < near_change * (1.0 - 1e-9):
                hits.append(f"{d:.3g} um along {_AXES[i]}")
        if hits:
            rule = (f"A point source is covered when it is at least "
                    f"{SOURCE_CLEARANCE_WAVELENGTHS:g} wavelengths from every such change, "
                    "along each axis" if len(axes) == 3 else
                    "A source plane (a launch, a plane wave, a mode source, a TF/SF face) is "
                    f"covered when it is at least {SOURCE_CLEARANCE_WAVELENGTHS:g} wavelengths "
                    "from every such change along its normal")
            out.append(
                f"{who}: a source is {', '.join(hits)} from a grid line where the cell size "
                f"changes ({label}), nearer than {SOURCE_CLEARANCE_WAVELENGTHS:g} wavelengths "
                f"in the surface's medium ({near_change:.3g} um). {_COVERED} {rule}; nearer, "
                "the field it radiates is itself asymmetric, which no projection removes "
                f"(measured just past the edge of a refined region: {_EDGE_MEASURED}). Keep "
                "sources that far clear of where a refinement ends")
    return out


#: The covered range the graded-mesh warnings mark the edge of.
_COVERED = ("On a graded mesh the projection's front/back asymmetry is shown to stay "
            "within 2 % of the peak (measured up to 0.93 %) at 30 background cells per "
            "wavelength, for refinements 2 and 3 times as fine as the background, only in "
            "the covered range.")
#: What was measured just outside it, sources beside a change of cell size.
_EDGE_MEASURED = ("up to 1.085 % at three times as fine and 0.92 % at twice as fine, at "
                  "30 background cells per wavelength")


def far_field_advice(sim, surfaces) -> Tuple[str, ...]:
    """The build warnings of the compiled far-field surfaces
    (``surfaces`` from :func:`compile_far_field_monitors`), as messages,
    judged against every source ``sim`` runs: called once its sources are
    final, the launches of ``ports=`` / ``source=`` included."""
    if not surfaces:
        return ()
    q = _lines(sim)
    sources = _source_regions(sim, q)
    advice: List[str] = []
    for faces, wl_max in surfaces:
        advice.extend(_advice(sim, faces, q, wl_max, sources))
    return tuple(dict.fromkeys(advice))


def compile_far_field_monitors(sim) -> Tuple[tuple, Dict[str, dict], list]:
    """``(monitors, recipes, surfaces)``: ``sim.monitors`` with each
    :class:`FarFieldMonitor` replaced by its face monitors (in place, in
    order), the recipe of each by name, and the compiled surfaces that
    :func:`far_field_advice` judges once the sources are final. Raises
    ``ValueError`` naming the face for what the projection cannot read."""
    q = _lines(sim)
    mons = list(sim.monitors)
    taken = {}
    for m in mons:
        if not isinstance(m, FarFieldMonitor):
            taken[m.name.lower()] = m.name
    out, recipes, surfaces = [], {}, []
    for m in mons:
        if not isinstance(m, FarFieldMonitor):
            out.append(m)
            continue
        who = _who(m)
        if m.name.lower() in taken:
            raise ValueError(f"{who}: another monitor is named {taken[m.name.lower()]!r}; "
                             "far_field(data, name) reads a far-field surface by its name")
        sym = _symmetry(sim)
        for side in m.exclude:
            i = _AXES.index(side[1])
            if sym[i]:
                raise ValueError(
                    f"{who}: exclude leaves out the {side} face, which is normal to the "
                    f"symmetry plane at {side[1]} = {_user(sim, i, 0.0):.6g} um. The run "
                    f"records the +{side[1]} face only and the -{side[1]} face is read as its "
                    "mirror image, so the box is open on both sides of the plane or on "
                    "neither: keep both faces, or drop the symmetry plane")
        freqs = np.asarray(m.freqs_hz, dtype=np.float64)
        faces = _faces(sim, m, q)
        faces = [f for f in faces if _side(f.axis, f.sign) not in m.exclude]
        n_by_face = []
        for face in faces:
            _check_layers(sim, face, q)
            face.n = _sample_medium(sim, face, q, freqs)
            n_by_face.append(face.n)
        if any(not np.allclose(n, n_by_face[0], rtol=1e-9, atol=0.0) for n in n_by_face[1:]):
            listing = "; ".join(f"{f.where}: index {_fmt(f.n)}" for f in faces)
            raise ValueError(
                f"{who}: the faces sit in different media ({listing}). A box that straddles "
                "an interface radiates into a layered space, which needs the layered "
                "Green's function; enclose the source in one medium")
        n_f = n_by_face[0]
        # the medium wavelength: the shortest sets a plane's stride, the
        # longest every distance the warnings judge
        wl = _C0 / (freqs * n_f) * 1e6
        for face in faces:
            _stride(face, q, float(np.min(wl)))
        for face in faces:
            if face.name.lower() in taken:
                raise ValueError(f"{who}: its {face.where} is named like the monitor "
                                 f"{taken[face.name.lower()]!r}; rename one of them")
            taken[face.name.lower()] = face.name
        taken[m.name.lower()] = m.name
        out.extend(face.monitor(q) for face in faces)
        surfaces.append((faces, float(np.max(wl))))
        window = _window_extent(sim, m, sym) if m.kind == "plane" and m.window else None
        recipes[m.name] = recipe(m, faces, n_f, symmetry=sym, window=window)
    return tuple(out), recipes, surfaces


def _declared_plane(mon: FarFieldMonitor, axis: str, sign: float) -> float:
    """The plane a box face is declared on, along its normal (um, the wire
    frame): the box's centre plus or minus half its size."""
    i = _AXES.index(axis)
    return float(mon.center_um[i]) + (0.5 if sign > 0 else -0.5) * float(mon.size_um[i])


def _window_extent(sim, mon: FarFieldMonitor, sym) -> List[list]:
    """``[[axis, fraction, lo_um, hi_um], ...]`` of a plane's edge window,
    one per in-plane axis in xyz order, over the plane's declared extent
    along it (its ``size_um``, or the PML-free interior; across a symmetry
    plane, the mirrored extent), not the recorded samples, which snap to
    the grid and differ between a run with symmetry planes and the full
    run."""
    out = []
    for t, frac in zip((a for a in _AXES if a != mon.axis), mon.window):
        i = _AXES.index(t)
        if mon.size_um is None:
            lo, hi, _kind = sim._nonabsorbing_bounds_um(i)
        else:
            c, size = float(mon.center_um[i]), float(mon.size_um[i])
            lo, hi = c - 0.5 * size, c + 0.5 * size
        lo, hi = float(lo), float(hi)
        if sym[i]:
            lo = -hi
        out.append([t, float(frac), lo, hi])
    return out


def recipe(mon: FarFieldMonitor, faces: Sequence[_Face], n_f: np.ndarray, *,
           symmetry=(0, 0, 0), window=None) -> dict:
    """The client-state record of one far-field surface (JSON-ready):
    ``far_field`` reads its faces and medium; the geometry is in the wire
    frame, as declared. Only when they apply: ``symmetry``, the run's
    symmetry planes, which ``far_field`` mirrors the faces through;
    ``exclude`` and ``open`` (``[axis, outward sign, plane_um]``, the
    declared plane of each face left out), where ``far_field`` cuts an open
    box's kept faces; ``window`` (``[axis, fraction, lo_um, hi_um]`` per
    in-plane axis), the plane's edge window over its declared extent."""
    out = {"kind": mon.kind,
           "faces": [[f.name, f.axis, float(f.sign)] for f in faces],
           "freqs_hz": [float(f) for f in mon.freqs_hz],
           "n_medium": [float(v) for v in np.atleast_1d(n_f)]}
    if mon.kind == "box":
        out["center_um"] = [float(v) for v in mon.center_um]
        out["size_um"] = [float(v) for v in mon.size_um]
    else:
        out.update(axis=mon.axis, position_um=float(mon.position_um), direction=mon.direction)
    if any(symmetry):
        out["symmetry"] = [int(v) for v in symmetry]
    if mon.kind == "box" and mon.exclude:
        out["exclude"] = list(mon.exclude)
        out["open"] = [[side[1], 1.0 if side[0] == "+" else -1.0,
                        _declared_plane(mon, side[1], 1.0 if side[0] == "+" else -1.0)]
                       for side in mon.exclude]
    if window is not None:
        out["window"] = window
    return out
