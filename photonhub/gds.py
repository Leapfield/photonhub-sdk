"""GDS layout import — turn a GDSII layout into PhotonHub structures.

A GDS file is a 2-D layout: ordered polygons each tagged by an integer
``(layer, datatype)`` pair, optionally organized into a hierarchy of cell
references (instances with translation/rotation/magnification). A photonic
device is built from that 2-D drawing by **extruding** each layer to a slab of a
fixed z-thickness filled with one material — the "layer stack".

:func:`import_gds` reads the file (via the optional ``gdstk`` dependency),
flattens any cell hierarchy into a flat polygon list, and emits one
:class:`~photonhub.PolySlab` :class:`~photonhub.Structure` per polygon on each
requested layer, using that layer's z-extent and medium. Polygon winding is
normalized to counter-clockwise (the orientation :class:`PolySlab` and the
rasterizer expect).

This is the client-side analogue of Tidy3D's ``Geometry.from_gds`` paired with a
``LayerStack``. It is what the GDS benchmark suite (``benchmarks/gds/``) uses to
build devices from the JPPhotonics ``fdtd-pipeline`` layouts (arXiv:2506.16665).

>>> import photonhub as ph
>>> from photonhub.gds import import_gds, GdsLayer
>>> si = ph.Medium(permittivity=3.478**2)
>>> structures = import_gds(
...     "crossing.gds",
...     [GdsLayer(layer=(1, 0), medium=si, zmin_um=0.0, thickness_um=0.22),
...      GdsLayer(layer=(2, 0), medium=si, zmin_um=0.0, thickness_um=0.15)],
... )
>>> sim = ph.Simulation(..., structures=structures)

Limitations (v1). Each polygon is extruded independently; polygons with holes
(even-odd fill) are not specially handled — for the strip/rib SOI layouts this
targets, every drawn shape is a simple filled region. Curved sidewalls are a
single global ``sidewall_angle`` per layer (matching ``PolySlab``); arbitrary
per-edge tapering is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from .components.base import AxisName
from .components.structures import Box, Cylinder, Medium, PolySlab, Sphere, Structure

__all__ = ["GdsLayer", "import_gds", "export_gds", "read_gds_cell_names"]

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class GdsLayer:
    """One GDS ``(layer, datatype)`` mapped to an extruded slab of one medium.

    ``zmin_um`` / ``thickness_um`` give the slab extent along the extrusion axis
    (the :func:`import_gds` ``axis``, default ``z``); the slab spans
    ``[zmin_um, zmin_um + thickness_um]``. ``sidewall_angle`` (radians) and
    ``reference_plane`` are forwarded to every :class:`PolySlab` emitted for this
    layer (see :class:`~photonhub.PolySlab`)."""

    layer: Tuple[int, int]
    medium: Medium
    zmin_um: float
    thickness_um: float
    sidewall_angle: float = 0.0
    reference_plane: str = "middle"

    def __post_init__(self) -> None:
        if self.thickness_um <= 0.0:
            raise ValueError(
                f"GdsLayer thickness_um must be > 0, got {self.thickness_um}"
            )

    @property
    def slab_bounds_um(self) -> Tuple[float, float]:
        return (self.zmin_um, self.zmin_um + self.thickness_um)


def _import_gdstk():
    """Import the optional ``gdstk`` GDSII reader with a helpful error."""
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "import_gds needs the optional 'gdstk' dependency to read GDSII "
            "files. Install it with `pip install gdstk` (the same reader Tidy3D "
            "and gdsfactory use)."
        ) from exc
    return gdstk


def _signed_area(points) -> float:
    """The signed polygon area (shoelace, halved); > 0 for counter-clockwise.
    Callers compare ``abs()`` of this against ``min_area_um2`` directly."""
    n = len(points)
    acc = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        acc += x0 * y1 - x1 * y0
    return 0.5 * acc


def _normalize_ring(points) -> Optional[List[Tuple[float, float]]]:
    """Clean one polygon's point ring into a CCW list of ``(u, v)`` tuples.

    Drops a duplicated closing vertex if present and reverses clockwise rings so
    the result is counter-clockwise. Returns ``None`` for a degenerate ring
    (< 3 distinct vertices or zero area)."""
    ring = [(float(p[0]), float(p[1])) for p in points]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        return None
    area2 = _signed_area(ring)
    if area2 == 0.0:
        return None
    if area2 < 0.0:  # clockwise -> reverse to CCW
        ring.reverse()
    return ring


def _select_cell(gdstk, lib, cell_name: Optional[str], gds_path: str):
    """Pick the cell to import: the named one, or the single top-level cell."""
    if cell_name is not None:
        for cell in lib.cells:
            if cell.name == cell_name:
                return cell
        have = ", ".join(repr(c.name) for c in lib.cells)
        raise ValueError(
            f"cell {cell_name!r} not found in {gds_path}; cells: {have}"
        )
    tops = lib.top_level()
    if not tops:
        raise ValueError(f"{gds_path} contains no cells")
    if len(tops) > 1:
        names = ", ".join(repr(c.name) for c in tops)
        raise ValueError(
            f"{gds_path} has multiple top-level cells ({names}); pass "
            "cell_name= to choose one"
        )
    return tops[0]


def import_gds(
    gds_path: Union[str, Path],
    layers: Sequence[GdsLayer],
    *,
    cell_name: Optional[str] = None,
    axis: AxisName = "z",
    flatten: bool = True,
    min_area_um2: float = 0.0,
) -> Tuple[Structure, ...]:
    """Import a GDSII layout as a tuple of extruded :class:`Structure`.

    Parameters
    ----------
    gds_path:
        Path to the ``.gds`` file. Coordinates are converted to MICRONS on
        read regardless of the file's user unit (a GDS authored in nm or mm
        imports at its true physical size).
    layers:
        The layers to import, as :class:`GdsLayer` specs (each maps a GDS
        ``(layer, datatype)`` to a z-slab + medium). A GDS layer present in the
        file but absent from this list is ignored; a spec whose layer is absent
        from the file simply yields no structures.
    cell_name:
        Which cell to import. ``None`` (default) uses the file's single
        top-level cell (raising if there are several — pass a name to choose).
    axis:
        Extrusion axis = slab normal (default ``"z"``: the GDS drawing plane is
        ``(x, y)``). The two GDS coordinate columns map to the two transverse
        axes of ``axis`` in index order, so ``axis="z"`` keeps GDS ``x,y`` as
        the device ``x,y``.
    flatten:
        Resolve cell references (instances) into polygons first (default True).
        Required for hierarchical layouts; with ``False`` only the chosen cell's
        own polygons are read.
    min_area_um2:
        Drop polygons whose absolute area is below this (default 0 = keep all) —
        a guard against zero-area slivers from boolean ops.

    Returns
    -------
    tuple[Structure, ...]
        One :class:`PolySlab` structure per polygon, grouped in the given
        ``layers`` order (file order within a layer). Paint order is last-wins
        (NUMERICS.md §9); same-material overlaps from a flattened hierarchy are
        therefore harmless.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if not layers:
        raise ValueError("import_gds: pass at least one GdsLayer")

    gdstk = _import_gdstk()
    gds_path = str(gds_path)
    if not Path(gds_path).is_file():
        raise FileNotFoundError(gds_path)

    # unit=1e-6: convert coordinates into MICRONS regardless of the file's own
    # user unit. A bare read_gds returns coordinates in whatever unit the file
    # was authored with (e.g. an nm-authored file comes back 1000x too large),
    # silently scaling the device; PhotonHub geometry is always microns.
    lib = gdstk.read_gds(gds_path, unit=1e-6)
    cell = _select_cell(gdstk, lib, cell_name, gds_path)
    if flatten:
        # Work on a copy so the library cell is not mutated; flatten() resolves
        # all references (applying their transforms) into this cell's polygons.
        cell = cell.copy(cell.name + "__phflat").flatten()

    # Bucket the cell's polygons by (layer, datatype) once.
    by_key: dict = {}
    for poly in cell.polygons:
        by_key.setdefault((poly.layer, poly.datatype), []).append(poly)

    structures: List[Structure] = []
    for spec in layers:
        slab_bounds = spec.slab_bounds_um
        for poly in by_key.get(spec.layer, ()):
            ring = _normalize_ring(poly.points)
            if ring is None:
                continue
            if min_area_um2 > 0.0 and abs(_signed_area(ring)) < min_area_um2:
                continue
            geometry = PolySlab(
                axis=axis,
                vertices_um=tuple(ring),
                slab_bounds_um=slab_bounds,
                sidewall_angle=spec.sidewall_angle,
                reference_plane=spec.reference_plane,
            )
            structures.append(Structure(geometry=geometry, medium=spec.medium))
    return tuple(structures)


# ---------------------------------------------------------------------------
# export (the reciprocal of import_gds)
# ---------------------------------------------------------------------------


def _medium_key(m: Medium):
    """A hashable identity for a medium (for grouping structures into layers)."""
    lz = m.lorentz
    return (
        m.permittivity,
        m.conductivity_s_per_m,
        None if lz is None else (lz.resonance_frequency_hz, lz.delta_eps, lz.linewidth_hz),
    )


def _slab_bounds_of(geometry, axis_i: int, axis: AxisName) -> Tuple[float, float]:
    """The ``[lo, hi]`` extent of a geometry along the extrusion axis."""
    if isinstance(geometry, Box):
        c = geometry.center_um[axis_i]
        h = geometry.size_um[axis_i] / 2.0
        return (c - h, c + h)
    if isinstance(geometry, PolySlab):
        if geometry.axis != axis:
            raise ValueError(
                f"PolySlab extruded along {geometry.axis!r} cannot be exported on "
                f"the {axis!r} drawing plane (its cross-section is not in-plane)"
            )
        return geometry.slab_bounds_um
    if isinstance(geometry, Cylinder):
        if geometry.axis != axis:
            raise ValueError(
                f"Cylinder extruded along {geometry.axis!r} cannot be exported on "
                f"the {axis!r} drawing plane"
            )
        c = geometry.center_um[axis_i]
        h = geometry.length_um / 2.0
        return (c - h, c + h)
    raise ValueError(
        f"export_gds cannot represent geometry {type(geometry).__name__} as a "
        "top-down layer polygon (only Box, PolySlab, Cylinder)"
    )


def _to_gds_polygons(gdstk, geometry, u: int, v: int, tol_um: float):
    """Convert one geometry's in-plane cross-section to gdstk polygon(s), with
    ``(u, v)`` the drawing-plane axis indices."""
    if isinstance(geometry, Box):
        cu, cv = geometry.center_um[u], geometry.center_um[v]
        hu, hv = geometry.size_um[u] / 2.0, geometry.size_um[v] / 2.0
        return [gdstk.rectangle((cu - hu, cv - hv), (cu + hu, cv + hv))]
    if isinstance(geometry, PolySlab):
        # vertices_um are (lower-index, higher-index) transverse coords = (u, v)
        return [gdstk.Polygon([(float(a), float(b)) for a, b in geometry.vertices_um])]
    if isinstance(geometry, Cylinder):
        cu, cv = geometry.center_um[u], geometry.center_um[v]
        inner = None if geometry.inner_radius_um == 0.0 else float(geometry.inner_radius_um)
        sweep = geometry.angle_stop - geometry.angle_start
        full = sweep >= 2.0 * 3.141592653589793 - 1e-9
        return [gdstk.ellipse(
            (cu, cv), float(geometry.radius_um),
            inner_radius=inner,
            initial_angle=0.0 if full else float(geometry.angle_start),
            final_angle=0.0 if full else float(geometry.angle_stop),
            tolerance=tol_um,
        )]
    raise ValueError(
        f"export_gds cannot represent geometry {type(geometry).__name__}"
    )


def export_gds(
    structures,
    gds_path: Union[str, Path],
    *,
    layers: Optional[Sequence[GdsLayer]] = None,
    axis: AxisName = "z",
    cell_name: str = "TOP",
    unit: float = 1e-6,
    precision: float = 1e-9,
    cylinder_tolerance_um: float = 1e-3,
    max_points: int = 0,
    z_tol_um: float = 1e-6,
) -> Tuple[GdsLayer, ...]:
    """Write PhotonHub structures OUT to a ``.gds`` layout — the reciprocal of
    :func:`import_gds`. Returns the :class:`GdsLayer` stack that maps the file
    back to structures (hand it straight to :func:`import_gds` to round-trip).

    ``structures`` is a sequence of :class:`Structure` or a
    :class:`~photonhub.library.Component`. Each structure's in-plane cross-section
    on the ``axis`` drawing plane becomes one polygon: a :class:`Box` -> a
    rectangle, a :class:`PolySlab` -> its vertex polygon, a :class:`Cylinder` ->
    a faceted (annular/wedge) polygon at ``cylinder_tolerance_um``. Structures
    are grouped into GDS layers by their ``(z-extent, medium)``:

    * ``layers=None`` (default) auto-assigns ``(1, 0), (2, 0), ...`` in
      first-occurrence order, one per distinct ``(slab, medium)``.
    * pass an explicit ``layers`` list (e.g. the one :func:`import_gds` used) to
      pin the layer numbers; every structure must match one of them.

    Gotchas handled: ``max_points=0`` disables gdstk's default 199-vertex polygon
    fracturing (a tolerance-faceted ring stays one polygon); ``unit``/``precision``
    default to the 1 nm database grid :func:`import_gds` reads. Paint order is not
    representable in GDS (last-wins is lost); interleaved cross-layer paint orders
    are flattened to layer order. ``Sphere`` and out-of-plane-extruded geometries
    are rejected (no top-down layer representation).
    """
    if axis not in _AXIS_INDEX:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    structs = list(getattr(structures, "structures", structures))
    if not structs:
        raise ValueError("export_gds: no structures to write")

    gdstk = _import_gdstk()
    ai = _AXIS_INDEX[axis]
    u, v = [i for i in range(3) if i != ai]

    # Per-structure: slab extent, medium, gdstk polygon(s).
    entries = []
    for s in structs:
        lo, hi = _slab_bounds_of(s.geometry, ai, axis)
        polys = _to_gds_polygons(gdstk, s.geometry, u, v, cylinder_tolerance_um)
        entries.append((lo, hi, s.medium, polys))

    def _matches(lo, hi, gl: GdsLayer) -> bool:
        glo, ghi = gl.slab_bounds_um
        return abs(lo - glo) <= z_tol_um and abs(hi - ghi) <= z_tol_um

    if layers is None:
        # auto-infer one layer per distinct (rounded slab, medium)
        assigned: dict = {}
        out_layers: List[GdsLayer] = []
        for lo, hi, med, _ in entries:
            key = (round(lo, 6), round(hi, 6), _medium_key(med))
            if key not in assigned:
                n = len(out_layers) + 1
                assigned[key] = (n, 0)
                out_layers.append(GdsLayer(layer=(n, 0), medium=med, zmin_um=lo, thickness_um=hi - lo))
        layer_of = {i: assigned[(round(lo, 6), round(hi, 6), _medium_key(med))]
                    for i, (lo, hi, med, _) in enumerate(entries)}
    else:
        out_layers = list(layers)
        layer_of = {}
        for i, (lo, hi, med, _) in enumerate(entries):
            match = [gl for gl in out_layers
                     if _matches(lo, hi, gl) and _medium_key(gl.medium) == _medium_key(med)]
            if not match:
                raise ValueError(
                    f"structure {i} (z=[{lo:.4g},{hi:.4g}], eps="
                    f"{med.permittivity:.4g}) matches none of the provided layers"
                )
            layer_of[i] = match[0].layer

    lib = gdstk.Library(unit=unit, precision=precision)
    cell = lib.new_cell(cell_name)
    for i, (lo, hi, med, polys) in enumerate(entries):
        L, D = layer_of[i]
        for p in polys:
            p.layer = L
            p.datatype = D
            cell.add(p)
    lib.write_gds(str(gds_path), max_points=max_points)
    return tuple(out_layers)


def read_gds_cell_names(gds_path: Union[str, Path]) -> Tuple[str, ...]:
    """List every cell name in a GDS file (top-level first) — a discovery
    helper for picking ``cell_name`` / layers before :func:`import_gds`."""
    gdstk = _import_gdstk()
    gds_path = str(gds_path)
    if not Path(gds_path).is_file():
        raise FileNotFoundError(gds_path)
    lib = gdstk.read_gds(gds_path)
    tops = {c.name for c in lib.top_level()}
    ordered = [c.name for c in lib.cells if c.name in tops]
    ordered += [c.name for c in lib.cells if c.name not in tops]
    return tuple(ordered)
