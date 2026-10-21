"""gdsfactory adapter: a layout ``Component`` and its layer stack in, PhotonHub
structures, ports and an S-matrix study out.

Install the optional extra with ``pip install "photonhub[gdsfactory]"``. This
module imports without gdsfactory; the functions that need it import it when
they run.

>>> import gdsfactory as gf
>>> import photonhub as ph
>>> from photonhub.integrations import gdsfactory as phgf
>>> gf.gpdk.PDK.activate()
>>> mmi = gf.components.mmi1x2()
>>> device = phgf.import_component(mmi, wlen_um=1.55)
>>> device.plot()                                  # preview the extruded scene
>>> plan = phgf.plan_component_smatrix(mmi, wlens_um=(1.5, 1.55, 1.6),
...                                    mesh=ph.Mesh(cells_per_wlen=12))
>>> plan.quote()                                   # cost before compute
>>> result = plan.run()                            # local CPU solver
>>> result.to_touchstone("mmi", version=2)

How a layout maps to a scene
============================
* **Layers.** Each level of a gdsfactory ``LayerStack`` becomes extruded
  :class:`~photonhub.Polygon` structures: the level's shapes in this component
  (derived layers included), its ``zmin`` and ``thickness``, its sidewall angle
  (degrees in gdsfactory, radians here, positive narrows upward in both) and
  its ``width_to_z`` reference plane (0, 0.5 or 1 for bottom, middle or top).
  A plain list of :class:`~photonhub.GDSLayer` is the simple alternative: each
  ``(layer, datatype)`` extruded with its own medium.
* **Paint order.** A lower gdsfactory ``mesh_order`` wins where levels
  overlap, so levels are painted from the highest ``mesh_order`` down.
* **Materials.** ``materials=`` maps a material name to a
  :class:`~photonhub.Medium`, a :class:`~photonhub.materials.Material` or a
  refractive index. Other names go through :data:`MATERIAL_ALIASES` to the
  material library, evaluated at ``wlen_um`` as a non-dispersive medium. A drawn
  level whose material maps to nothing is refused.
* **Background.** The cladding medium: ``background=``, or the stack's ``clad``
  level, or its ``box`` level. Levels with no shapes in the component (the
  wafer-wide substrate, box and cladding, unused metal layers) are not drawn
  and are listed in :attr:`ImportedComponent.omitted_levels`.
* **Ports.** Each optical port becomes a :class:`~photonhub.Port` at the port
  centre, on the level its layer belongs to, at that level's mid-height. Its
  direction comes from the orientation (0, 90, 180 or 270 degrees), and its
  mode window follows the port-window rule. The S-matrix is referenced at the
  gdsfactory port centres, so models compose port to port in a circuit.
* **Guide extensions.** Each port's guide is extended outward by
  ``extension_um`` with its own cross-section, so it runs through the
  absorbing layers instead of ending inside the domain.
* **Holes.** A polygon with holes can be split into hole-free pieces only on
  a vertical-wall level. A closed ring on the generic PDK's sloped core needs
  a layer stack with ``sidewall_angle=0`` for that level.

S-parameters from this path inherit the convergence of the simulation and the
mesh you choose. Check them with a mesh refinement before you rely on them.
"""

from __future__ import annotations

import math
import types
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .. import materials as _materials
from .._compat import caller_stacklevel
from ..components.authoring import Domain, Mesh, Port
from ..components.medium import Background
from ..components.structures import Box, Medium, Polygon, Structure
from ..gds import GDSLayer, _normalize_ring

__all__ = [
    "MATERIAL_ALIASES",
    "ImportedComponent",
    "ImportedLevel",
    "import_component",
    "plan_component_smatrix",
    "run_component_smatrix",
]

_EXTRA_HINT = 'pip install "photonhub[gdsfactory]"'
# orientation in degrees -> (axis, the side of the plane facing the wall)
_ORIENTATIONS = {0: ("x", "+"), 90: ("y", "+"), 180: ("x", "-"), 270: ("y", "-")}
_REFERENCE_PLANES = {0.0: "bottom", 0.5: "middle", 1.0: "top"}
_ANGLE_TOL_DEG = 1e-6
_FACE_TOL_UM = 2e-3          # two database units of a 1 nm layout grid
_TOUCH_TOL_UM = 1e-6
_EXTENSION_WLENS = 3.0       # default guide extension, in vacuum wavelengths
_PREVIEW_CELLS_PER_WLEN = 8.0
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

#: gdsfactory material names (lower case) and the :mod:`photonhub.materials`
#: entry each one maps to. Metals are left out: the library holds them as
#: dispersive-only, so a drawn metal level needs ``materials=``.
MATERIAL_ALIASES: Mapping[str, str] = types.MappingProxyType({
    "si": "Si", "silicon": "Si", "c-si": "Si",
    "sio2": "SiO2", "oxide": "SiO2", "silica": "SiO2",
    "sin": "Si3N4", "si3n4": "Si3N4", "nitride": "Si3N4",
    "ge": "Ge", "germanium": "Ge",
    "air": "Vacuum", "vacuum": "Vacuum",
    "gaas": "GaAs",
    "inp": "InP",
    "aln": "AlN",
    "tio2": "TiO2",
    "al2o3": "Sapphire", "sapphire": "Sapphire",
    "pmma": "PMMA",
    "caf2": "CaF2",
})

MaterialSpec = Union[Medium, "_materials.Material", float, int]


def _import_gdsfactory():
    """Import gdsfactory, or say which extra installs it."""
    try:
        import gdsfactory
    except ImportError as exc:
        raise ImportError(
            "photonhub.integrations.gdsfactory needs the optional gdsfactory "
            f"package: {_EXTRA_HINT}") from exc
    return gdsfactory


# ---------------------------------------------------------------------------
# public values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportedLevel:
    """One drawn level of an imported component: its material, the medium it
    resolved to, and the extrusion every polygon on it takes.

    ``sidewall_angle`` is in radians with the meaning of
    :attr:`photonhub.Polygon.sidewall_angle` (positive narrows toward +z), and
    ``reference_plane`` is where the drawn outline applies. ``mesh_order`` is
    the gdsfactory priority (lower wins), ``None`` for a :class:`GDSLayer` list,
    which paints in list order. ``num_polygons`` counts this level's polygons,
    the port extensions included."""

    name: str
    material: Optional[str]
    medium: Medium
    zmin_um: float
    thickness_um: float
    sidewall_angle: float
    reference_plane: str
    mesh_order: Optional[int]
    num_polygons: int


@dataclass(frozen=True)
class ImportedComponent:
    """A layout component as PhotonHub values: what :func:`import_component`
    returns.

    ``structures`` and ``ports`` are in the layout's own frame (microns, the
    drawing plane is ``x, y``, the stack runs along ``z``), the same two fields
    as :class:`photonhub.library.Component`, so ``ph.export_gds(device, path)``
    accepts it. ``background`` is the cladding. ``levels`` lists the drawn
    levels in paint order; ``omitted_levels`` maps every level that drew
    nothing to the reason. ``wlen_um`` is the wavelength the materials were
    evaluated at and ``extension_um`` how far each port's guide was extended.
    """

    name: str
    structures: Tuple[Structure, ...]
    ports: Tuple[Port, ...]
    background: Background
    levels: Tuple[ImportedLevel, ...]
    omitted_levels: Mapping[str, str]
    wlen_um: float
    extension_um: float

    def simulation(self, *, wlens_um, mesh: Mesh, domain: Optional[Domain] = None,
                   run=None, **simulation_kwargs: Any):
        """A :class:`~photonhub.Simulation` of this component, fitted around it.

        The domain is fitted with ``domain`` (default ``ph.Domain()``), the mesh
        is ``mesh`` (no default: the resolution is your convergence choice),
        and the ports are declared, so their modes are solved at construction.
        No source is set unless you pass ``source=``; other keywords go to the
        :class:`~photonhub.Simulation` constructor. This standalone scene uses
        the plain domain fit around the readout planes. A study built by hand from it
        must leave each launch standoff plus the port margin beyond those
        planes, as :func:`plan_component_smatrix` does. Warns when the fitted
        domain still had to continue a port's guide (``extension_um`` too short to
        reach through the absorbing layers) or when the materials were
        evaluated outside the simulated band."""
        from ..components.simulation import Simulation

        args: Dict[str, Any] = dict(
            domain=domain if domain is not None else Domain(), mesh=mesh,
            wlens_um=wlens_um, background=self.background,
            structures=self.structures, ports=self.ports)
        if run is not None:
            args["run"] = run
        clash = sorted(set(args) & set(simulation_kwargs))
        if clash:
            raise TypeError(f"simulation() sets {clash} itself; pass them as its own arguments")
        wl = (float(wlens_um),) if isinstance(wlens_um, (int, float)) else tuple(float(w) for w in wlens_um)
        if wl and not (min(wl) * (1 - 1e-9) <= self.wlen_um <= max(wl) * (1 + 1e-9)):
            warnings.warn(
                f"the materials of {self.name!r} were evaluated at {self.wlen_um:g} um, "
                f"outside the simulated wavelengths {min(wl):g} to {max(wl):g} um; "
                "import the component again at a wavelength inside the band",
                UserWarning, stacklevel=caller_stacklevel())
        sim = Simulation(**args, **simulation_kwargs)
        # The adapter emits polygons only. A Box in the fitted scene is a
        # rectangular continuation; a symmetry fold may drop other structures.
        if any(isinstance(s.geometry, Box) for s in sim.structures):
            warnings.warn(
                f"the guide extension of {self.extension_um:g} um did not reach through the "
                "absorbing layers, so a rectangular continuation was added at a port; pass a "
                "larger extension_um to import_component",
                UserWarning, stacklevel=caller_stacklevel())
        return sim

    def plot(self, z: Optional[float] = None, *, ax=None, **kwargs: Any):
        """Preview the scene: a cross-section at height ``z`` (default the first
        port's centre height) of a simulation fitted at ``wlen_um`` on a coarse
        preview mesh. Returns the matplotlib ``Axes`` of
        :meth:`photonhub.Simulation.plot`; for another cut or a 3D view call
        :meth:`simulation` and plot that."""
        sim = self.simulation(wlens_um=(self.wlen_um,),
                              mesh=Mesh(cells_per_wlen=_PREVIEW_CELLS_PER_WLEN))
        if z is None:
            if self.ports:
                z = float(self.ports[0].center_um[2])
            else:
                lv = self.levels[-1]
                z = lv.zmin_um + 0.5 * lv.thickness_um
        return sim.plot(z=z, ax=ax, **kwargs)


# ---------------------------------------------------------------------------
# reading the layout
# ---------------------------------------------------------------------------


def _int_like(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) or type(v).__name__.startswith(("int", "uint"))


def _layer_tuple(obj) -> Optional[Tuple[int, int]]:
    """``(layer, datatype)`` of a layer spec: a pair, an object with integer
    ``layer`` and ``datatype`` (a layer enum, a ``LayerInfo``), or a logical
    layer wrapping one. ``None`` when it names no single drawn layer (a derived
    layer)."""
    for _ in range(4):
        if obj is None:
            return None
        if isinstance(obj, (tuple, list)):
            if len(obj) == 2 and all(_int_like(v) for v in obj):
                return int(obj[0]), int(obj[1])
            return None
        lay, dt = getattr(obj, "layer", None), getattr(obj, "datatype", None)
        if _int_like(lay) and _int_like(dt):
            return int(lay), int(dt)
        obj = lay
    return None


@dataclass
class _Level:
    """A layer-stack level being imported."""

    name: str
    material: Optional[str]
    zmin_um: float
    thickness_um: float
    sidewall_angle: float
    reference_plane: str
    mesh_order: Optional[int]
    layer: Optional[Tuple[int, int]]          # the logical layer, None when derived
    port_layer: Optional[Tuple[int, int]]     # the layer its derived shapes are drawn on
    shapes_spec: Any                          # an object with get_shapes(component), or None
    medium: Optional[Medium] = None
    rings: Optional[List[List[Tuple[float, float]]]] = None


def _reference_plane(width_to_z, name: str) -> str:
    try:
        key = float(width_to_z if width_to_z is not None else 0.0)
    except (TypeError, ValueError):
        key = math.nan
    for value, plane in _REFERENCE_PLANES.items():
        if abs(key - value) <= 1e-9:
            return plane
    raise ValueError(
        f"level {name!r}: width_to_z={width_to_z!r} is not 0, 0.5 or 1; a PhotonHub polygon "
        "takes its outline at the bottom, middle or top of the slab")


def _levels_from_stack(layer_stack) -> Tuple[List[_Level], bool]:
    """The levels of a gdsfactory ``LayerStack`` or of a :class:`GDSLayer`
    list, and whether they came from a stack (with names to take the
    background from)."""
    stack_levels = getattr(layer_stack, "layers", None)
    if isinstance(stack_levels, Mapping):
        out = []
        for name, lv in stack_levels.items():
            zmin, thickness = float(lv.zmin), float(lv.thickness)
            if thickness == 0.0:
                raise ValueError(f"level {name!r}: thickness is 0")
            if thickness < 0.0:         # extends downward from zmin
                zmin, thickness = zmin + thickness, -thickness
            spec = lv.layer
            mesh_order = getattr(lv, "mesh_order", None)
            out.append(_Level(
                name=str(name), material=getattr(lv, "material", None),
                zmin_um=zmin, thickness_um=thickness,
                sidewall_angle=math.radians(float(getattr(lv, "sidewall_angle", 0.0) or 0.0)),
                reference_plane=_reference_plane(getattr(lv, "width_to_z", 0.0), str(name)),
                mesh_order=None if mesh_order is None else int(mesh_order),
                layer=_layer_tuple(spec),
                port_layer=_layer_tuple(getattr(lv, "derived_layer", None)),
                shapes_spec=spec if hasattr(spec, "get_shapes") else None))
        return out, True
    try:
        items = list(layer_stack)
    except TypeError:
        items = []
    if not items or not all(isinstance(g, GDSLayer) for g in items):
        raise TypeError(
            "layer_stack must be a gdsfactory LayerStack, a non-empty sequence of "
            "photonhub.GDSLayer, or None for the active PDK's stack")
    seen: Dict[Tuple[int, int], int] = {}
    out = []
    for i, g in enumerate(items):
        key = (int(g.layer[0]), int(g.layer[1]))
        name = f"{key[0]}/{key[1]}"
        if key in seen:
            name += f"#{i}"
        seen[key] = i
        lo, hi = g.slab_bounds_um
        level = _Level(name=name, material=None, zmin_um=float(lo), thickness_um=float(hi - lo),
                       sidewall_angle=float(g.sidewall_angle), reference_plane=g.reference_plane,
                       mesh_order=None, layer=key, port_layer=key, shapes_spec=None)
        level.medium = g.medium
        out.append(level)
    return out, False


def _is_layout_component(component) -> bool:
    """A real gdsfactory component (it carries a KLayout layout), as opposed to
    a duck-typed stand-in."""
    return hasattr(component, "kcl")


def _region_rings(region, dbu: float, level: _Level) -> List[List[Tuple[float, float]]]:
    """Hole-free outlines, in microns, of a KLayout region's merged polygons.
    A polygon with holes is split into hole-free pieces on a vertical-wall
    level and refused on a sloped one."""
    rings: List[List[Tuple[float, float]]] = []
    for poly in region.merged().each():
        stack = [poly]
        splits = 0
        while stack:
            p = stack.pop()
            if p.holes() == 0:
                rings.append([(pt.x * dbu, pt.y * dbu) for pt in p.each_point_hull()])
                continue
            if level.sidewall_angle != 0.0:
                raise ValueError(
                    f"level {level.name!r}: a polygon with holes on a sloped-wall level cannot be "
                    "extruded (splitting it would put a sloped face on an internal seam); set the "
                    "level's sidewall_angle to 0 or split the shape in the layout")
            splits += 1
            if splits > 10_000:
                raise ValueError(f"level {level.name!r}: could not split a polygon with holes")
            stack.extend(p.split())
    return rings


def _component_rings(component, level: _Level, gf_lazy: Callable[[], Any]) -> List[List[Tuple[float, float]]]:
    """The outlines of ``level`` in ``component``, in microns."""
    if _is_layout_component(component):
        spec = level.shapes_spec
        if spec is None:
            if level.layer is None:
                return []                     # a level with no layer draws nothing
            spec = gf_lazy().technology.LogicalLayer(layer=level.layer)
        return _region_rings(spec.get_shapes(component), float(component.kcl.dbu), level)
    if level.layer is None:
        if level.shapes_spec is None:
            return []
        raise ValueError(
            f"level {level.name!r} is a derived layer; reading it needs a gdsfactory component")
    by_layer = component.get_polygons_points(merge=True, by="tuple")
    return [[(float(p[0]), float(p[1])) for p in pts] for pts in by_layer.get(level.layer, ())]


def _normalize_materials(materials) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in dict(materials or {}).items():
        k = str(key).strip().lower()
        if k in out:
            raise ValueError(f"materials= names {key!r} twice (names match case-insensitively)")
        out[k] = value
    return out


def _as_medium(value, wlen_um: float, what: str) -> Medium:
    if isinstance(value, Medium):
        return value
    if isinstance(value, _materials.Material):
        return value.medium(wlen_um=wlen_um)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = float(value)
        if not n >= 1.0:
            raise ValueError(f"{what}: a refractive index must be >= 1, got {value!r}")
        return Medium(permittivity=n * n)
    raise TypeError(
        f"{what}: a material maps to a ph.Medium, a ph.materials.Material or a refractive "
        f"index, got {type(value).__name__}")


def _resolve_medium(material, overrides: Mapping[str, Any], wlen_um: float, what: str) -> Medium:
    if material is None:
        raise ValueError(f"{what} has no material; pass materials= or use a level that names one")
    key = str(material).strip().lower()
    if key in overrides:
        return _as_medium(overrides[key], wlen_um, what)
    alias = MATERIAL_ALIASES.get(key)
    hint = f"pass materials={{{str(material)!r}: ph.Medium(permittivity=...)}}"
    if alias is None:
        raise ValueError(f"{what}: material {material!r} has no PhotonHub medium; {hint}")
    try:
        return _materials.get(alias).medium(wlen_um=wlen_um)
    except ValueError as exc:
        raise ValueError(
            f"{what}: material {material!r} maps to the library entry {alias!r}, which cannot be "
            f"evaluated at {wlen_um:g} um ({exc}); {hint}") from exc


def _plain_background(medium: Medium, what: str) -> Background:
    if medium != Medium(permittivity=medium.permittivity):
        raise ValueError(
            f"{what}: the background must be a lossless, non-dispersive, isotropic medium; "
            "pass background= with a plain permittivity")
    return Background(permittivity=medium.permittivity)


def _resolve_background(background, levels: Sequence[_Level], from_stack: bool,
                        overrides: Mapping[str, Any], wlen_um: float) -> Tuple[Background, Optional[_Level]]:
    """The background, and the stack level it came from (when it did)."""
    if background is not None:
        if isinstance(background, Background):
            return background, None
        if isinstance(background, str):
            return _plain_background(
                _resolve_medium(background, overrides, wlen_um, "background"), "background"), None
        return _plain_background(_as_medium(background, wlen_um, "background"), "background"), None
    if from_stack:
        by_name = {lv.name: lv for lv in levels}
        for name in ("clad", "box"):
            if name in by_name:
                lv = by_name[name]
                medium = _resolve_medium(lv.material, overrides, wlen_um, f"level {name!r}")
                return _plain_background(medium, f"level {name!r}"), lv
    raise ValueError(
        "no background: pass background= (a ph.Background, a medium, a material name or a "
        "refractive index); a layer stack supplies one from a level named 'clad' or 'box'")


def _port_attrs(port) -> Tuple[str, Tuple[float, float], Optional[float], float, Optional[Tuple[int, int]], str]:
    name = str(port.name)
    center = tuple(float(c) for c in port.center)
    orientation = getattr(port, "orientation", None)
    layer = _layer_tuple(getattr(port, "layer_info", None)) or _layer_tuple(getattr(port, "layer", None))
    return (name, (center[0], center[1]), None if orientation is None else float(orientation),
            float(port.width), layer, str(getattr(port, "port_type", "optical")))


def _orientation(name: str, orientation: Optional[float]) -> Tuple[str, str]:
    if orientation is None:
        raise ValueError(f"port {name!r} has no orientation")
    deg = orientation % 360.0
    for key, value in _ORIENTATIONS.items():
        if min(abs(deg - key), abs(deg - key - 360.0)) <= _ANGLE_TOL_DEG:
            return value
    raise ValueError(
        f"port {name!r}: orientation {orientation:g} deg is not 0, 90, 180 or 270; PhotonHub "
        "ports are normal to x or y (rotate the component or add a straight)")


def _port_level(name: str, layer: Optional[Tuple[int, int]], drawn: Sequence[_Level],
                port_level) -> _Level:
    by_name = {lv.name: lv for lv in drawn}
    chosen = port_level.get(name) if isinstance(port_level, Mapping) else port_level
    if chosen is not None:
        if chosen not in by_name:
            raise ValueError(
                f"port {name!r}: port_level {chosen!r} is not a drawn level of this component "
                f"(drawn: {sorted(by_name)})")
        return by_name[chosen]
    candidates = [lv for lv in drawn if layer is not None and lv.port_layer == layer]
    if not candidates:
        candidates = [lv for lv in drawn if layer is not None and lv.layer == layer]
    if not candidates:
        raise ValueError(
            f"port {name!r} on layer {layer} matches no drawn level; pass port_level= with one "
            f"of {sorted(by_name)}")
    if len(candidates) > 1 and all(lv.mesh_order is not None for lv in candidates):
        best = min(lv.mesh_order for lv in candidates)
        candidates = [lv for lv in candidates if lv.mesh_order == best]
    if len(candidates) > 1:
        raise ValueError(
            f"port {name!r} on layer {layer} matches the levels "
            f"{[lv.name for lv in candidates]}; pass port_level= to choose one")
    return candidates[0]


def _extension_ring(center: Tuple[float, float], axis: str, side: str, width_um: float,
                    length_um: float) -> List[Tuple[float, float]]:
    """A rectangle continuing a port's guide outward from its centre."""
    x, y = center
    s = 1.0 if side == "+" else -1.0
    h = 0.5 * width_um
    if axis == "x":
        x1 = x + s * length_um
        return [(min(x, x1), y - h), (max(x, x1), y - h), (max(x, x1), y + h), (min(x, x1), y + h)]
    y1 = y + s * length_um
    return [(x - h, min(y, y1)), (x + h, min(y, y1)), (x + h, max(y, y1)), (x - h, max(y, y1))]


def _extend_outline(rings: List[List[Tuple[float, float]]], center: Tuple[float, float], axis: str,
                    side: str, width_um: float, length_um: float) -> bool:
    """Grow the outline that ends at a port by the port's extension, in place.

    The extension joins the guide's outline (one polygon, no seam) where an
    edge of that outline lies on the port's face and spans the port's width:
    a separate rectangle would meet the guide at a seam, and on a sloped-wall
    level both sides of a seam slope. Returns False when no outline has such
    an edge."""
    a = _AXIS_INDEX[axis]
    t = 1 - a
    face, c, h = center[a], center[t], 0.5 * width_um
    out = face + (length_um if side == "+" else -length_um)
    tol = _FACE_TOL_UM
    for ring in rings:
        n = len(ring)
        for i in range(n):
            p, q = ring[i], ring[(i + 1) % n]
            if abs(p[a] - face) > tol or abs(q[a] - face) > tol:
                continue
            lo, hi = min(p[t], q[t]), max(p[t], q[t])
            if lo > c - h + tol or hi < c + h - tol:
                continue
            d = 1.0 if q[t] > p[t] else -1.0          # the edge's direction along t
            bump = []
            for along, tv in ((face, c - d * h), (out, c - d * h), (out, c + d * h), (face, c + d * h)):
                pt = [0.0, 0.0]
                pt[a], pt[t] = along, tv
                bump.append((pt[0], pt[1]))
            # keep the outline free of repeated vertices where the guide is as
            # wide as the port
            if abs(bump[0][t] - p[t]) <= tol:
                bump[0] = None
            if abs(bump[-1][t] - q[t]) <= tol:
                bump[-1] = None
            ring[i + 1:i + 1] = [b for b in bump if b is not None]
            return True
    return False


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


def import_component(
    component,
    layer_stack=None,
    *,
    wlen_um: float,
    materials: Optional[Mapping[str, MaterialSpec]] = None,
    background=None,
    port_names: Optional[Sequence[str]] = None,
    mode: Union[str, Mapping[str, str]] = "TE0",
    port_level: Union[None, str, Mapping[str, str]] = None,
    extension_um: Optional[float] = None,
) -> ImportedComponent:
    """Turn a gdsfactory ``Component`` into PhotonHub structures and ports.

    Parameters
    ----------
    component:
        A gdsfactory ``Component``.
    layer_stack:
        A gdsfactory ``LayerStack``; ``None`` (default) takes the active PDK's.
        Or a sequence of :class:`~photonhub.GDSLayer`, each ``(layer,
        datatype)`` extruded with its own medium (then pass ``background=``).
    wlen_um:
        The wavelength, in microns, at which library materials are evaluated
        (non-dispersive media). Use the centre of the band you simulate.
    materials:
        Material name to a :class:`~photonhub.Medium`, a
        :class:`~photonhub.materials.Material` or a refractive index, matched
        case-insensitively; it takes precedence over :data:`MATERIAL_ALIASES`.
    background:
        The cladding: a :class:`~photonhub.Background`, a medium, a material
        name or a refractive index. ``None`` takes the stack's ``clad`` level,
        else its ``box`` level.
    port_names:
        The optical ports to keep, by name (default all, in component order).
        A port left out keeps its guide ending at the component's edge.
    mode:
        The port mode, ``"TE0"`` by default, or a mapping from port name to mode.
    port_level:
        The level a port's mode sits on, when its layer matches several: a
        level name for every port, or a mapping from port name to level name.
    extension_um:
        How far each kept port's guide is extended outward, in microns
        (default three times ``wlen_um``). It must reach through the absorbing
        layers of the simulation you build.

    Returns
    -------
    ImportedComponent
        The structures, ports and background, ready for
        :meth:`ImportedComponent.simulation`.

    Raises
    ------
    ImportError
        When gdsfactory is needed (``layer_stack=None`` or a real component)
        and is not installed.
    ValueError
        For a drawn level with no medium, a port that is not normal to ``x`` or
        ``y``, a port that does not sit on the face of the device it points
        through, or a port whose level is ambiguous. A polygon with holes on a
        sloped-wall level is also refused: a closed ring on the generic PDK's
        core needs a stack with ``sidewall_angle=0`` for that level.
    """
    wlen_um = float(wlen_um)
    if not (wlen_um > 0.0 and math.isfinite(wlen_um)):
        raise ValueError(f"wlen_um must be a positive wavelength in microns, got {wlen_um}")
    if extension_um is None:
        extension_um = _EXTENSION_WLENS * wlen_um
    extension_um = float(extension_um)
    if not (extension_um > 0.0 and math.isfinite(extension_um)):
        raise ValueError(f"extension_um must be > 0, got {extension_um}")

    gf_module: List[Any] = []

    def gf_lazy():
        if not gf_module:
            gf_module.append(_import_gdsfactory())
        return gf_module[0]

    if layer_stack is None:
        layer_stack = gf_lazy().get_active_pdk().layer_stack
        if layer_stack is None:
            raise ValueError("the active gdsfactory PDK has no layer stack; pass layer_stack=")
    levels, from_stack = _levels_from_stack(layer_stack)
    overrides = _normalize_materials(materials)
    name = str(getattr(component, "name", "component"))

    # --- ports ----------------------------------------------------------------
    optical = [_port_attrs(p) for p in component.ports]
    optical = [p for p in optical if p[5] == "optical"]
    if port_names is not None:
        wanted = [str(n) for n in port_names]
        have = [p[0] for p in optical]
        unknown = [n for n in wanted if n not in have]
        if unknown:
            raise ValueError(f"port_names {unknown} are not optical ports of {name!r} ({have})")
        optical = [p for p in optical if p[0] in wanted]
    if isinstance(mode, Mapping):
        unknown = sorted(set(mode) - {p[0] for p in optical})
        if unknown:
            raise ValueError(f"mode= names ports {unknown} that are not imported")
    if isinstance(port_level, Mapping):
        unknown = sorted(set(port_level) - {p[0] for p in optical})
        if unknown:
            raise ValueError(f"port_level= names ports {unknown} that are not imported")

    # --- the drawn shapes of each level (before any extension) ------------------
    for lv in levels:
        lv.rings = [r for r in (_normalize_ring(ring) for ring in _component_rings(component, lv, gf_lazy))
                    if r is not None]
    drawn = [lv for lv in levels if lv.rings]
    omitted = {lv.name: "no shapes in this component" for lv in levels if not lv.rings}
    if not drawn:
        raise ValueError(f"{name!r} draws nothing on any level of the layer stack")

    background_value, background_level = _resolve_background(
        background, levels, from_stack, overrides, wlen_um)
    for lv in drawn:
        if lv.medium is None:
            lv.medium = _resolve_medium(lv.material, overrides, wlen_um, f"level {lv.name!r}")

    # --- ports onto levels, and the face check ---------------------------------
    xs = [x for lv in drawn for ring in lv.rings for x, _ in ring]
    ys = [y for lv in drawn for ring in lv.rings for _, y in ring]
    bbox = {"x": (min(xs), max(xs)), "y": (min(ys), max(ys))}
    ports: List[Port] = []
    port_geo: List[Tuple[str, Tuple[float, float], str, str, float, _Level]] = []
    for pname, center, orientation, width, layer, _ in optical:
        axis, side = _orientation(pname, orientation)
        face = bbox[axis][1] if side == "+" else bbox[axis][0]
        plane = center[_AXIS_INDEX[axis]]
        if abs(plane - face) > _FACE_TOL_UM:
            raise ValueError(
                f"port {pname!r} at {axis}={plane:g} um points {side}{axis} but the device's "
                f"{side}{axis} face is at {face:g} um; a port must sit on the face it points "
                "through (extend the port in the layout first)")
        lv = _port_level(pname, layer, drawn, port_level)
        port_mode = mode.get(pname, "TE0") if isinstance(mode, Mapping) else mode
        ports.append(Port(
            pname, (center[0], center[1], lv.zmin_um + 0.5 * lv.thickness_um), axis, width,
            out_direction=side, thickness_um=lv.thickness_um, thickness_axis="z",
            mode=port_mode, medium=lv.medium))
        port_geo.append((pname, center, axis, side, width, lv))

    # --- guide extensions -------------------------------------------------------
    if port_geo and _is_layout_component(component):
        # the port's own cross-section, rib slabs included
        extended = gf_lazy().components.extend_ports(
            component, port_names=[p[0] for p in port_geo], length=extension_um)
        for lv in drawn:
            lv.rings = [r for r in (_normalize_ring(ring) for ring in _component_rings(extended, lv, gf_lazy))
                        if r is not None]
    else:
        for pname, center, axis, side, width, lv in port_geo:
            if _extend_outline(lv.rings, center, axis, side, width, extension_um):
                continue
            if lv.sidewall_angle != 0.0:
                warnings.warn(
                    f"port {pname!r}: no outline edge of level {lv.name!r} spans the port on its "
                    "face, so its extension is a separate polygon that meets the guide at a sloped "
                    "seam; draw the port's guide to the face in the layout",
                    UserWarning, stacklevel=caller_stacklevel())
            lv.rings.append(_normalize_ring(_extension_ring(center, axis, side, width, extension_um)))

    # --- structures, in paint order ----------------------------------------------
    if from_stack:
        order = sorted(drawn, key=lambda lv: -(lv.mesh_order if lv.mesh_order is not None else 0))
    else:
        order = list(drawn)
    structures: List[Structure] = []
    imported_levels: List[ImportedLevel] = []
    for lv in order:
        bounds = (lv.zmin_um, lv.zmin_um + lv.thickness_um)
        for ring in lv.rings:
            structures.append(Structure(
                geometry=Polygon(axis="z", vertices_um=tuple(ring), slab_bounds_um=bounds,
                                 sidewall_angle=lv.sidewall_angle, reference_plane=lv.reference_plane),
                medium=lv.medium))
        imported_levels.append(ImportedLevel(
            name=lv.name, material=lv.material, medium=lv.medium, zmin_um=lv.zmin_um,
            thickness_um=lv.thickness_um, sidewall_angle=lv.sidewall_angle,
            reference_plane=lv.reference_plane, mesh_order=lv.mesh_order,
            num_polygons=len(lv.rings)))

    _warn_dropped_wafer_levels(levels, drawn, background_value, background_level, overrides, wlen_um)

    return ImportedComponent(
        name=name, structures=tuple(structures), ports=tuple(ports), background=background_value,
        levels=tuple(imported_levels), omitted_levels=types.MappingProxyType(omitted),
        wlen_um=wlen_um, extension_um=extension_um)


def _warn_dropped_wafer_levels(levels, drawn, background: Background, background_level: Optional[_Level],
                               overrides, wlen_um: float) -> None:
    """Warn when a wafer-wide level (one on the same layer as the level the
    background came from) has another medium and sits against the device: the
    single background cannot represent it."""
    if background_level is None or background_level.layer is None:
        return
    lo = min(lv.zmin_um for lv in drawn)
    hi = max(lv.zmin_um + lv.thickness_um for lv in drawn)
    dropped = []
    for lv in levels:
        if lv.rings or lv is background_level or lv.layer != background_level.layer:
            continue
        if lv.zmin_um > hi + _TOUCH_TOL_UM or lv.zmin_um + lv.thickness_um < lo - _TOUCH_TOL_UM:
            continue
        try:
            medium = _resolve_medium(lv.material, overrides, wlen_um, f"level {lv.name!r}")
        except (ValueError, TypeError):
            continue
        if medium != Medium(permittivity=background.permittivity):
            dropped.append(f"{lv.name!r} ({lv.material})")
    if dropped:
        warnings.warn(
            f"the scene uses one background medium, so the wafer-wide level(s) {', '.join(dropped)} "
            "that sit against the device are not modelled; add them as structures with a Domain "
            "whose walls you choose, or pick another background",
            UserWarning, stacklevel=caller_stacklevel())


# ---------------------------------------------------------------------------
# S-matrix study
# ---------------------------------------------------------------------------


def _imported(component, layer_stack, *, wlen_um, materials, background, port_names, mode,
              port_level, extension_um) -> ImportedComponent:
    if isinstance(component, ImportedComponent):
        given = {"layer_stack": layer_stack, "materials": materials, "background": background,
                 "port_names": port_names, "port_level": port_level, "extension_um": extension_um}
        set_options = sorted(k for k, v in given.items() if v is not None)
        if mode != "TE0":
            set_options.append("mode")
        if set_options:
            raise ValueError(
                f"{set_options} apply when importing; the component is already imported "
                "(call import_component with them instead)")
        return component
    return import_component(
        component, layer_stack, wlen_um=wlen_um, materials=materials, background=background,
        port_names=port_names, mode=mode, port_level=port_level, extension_um=extension_um)


def _study_domain(imported: ImportedComponent, domain: Optional[Domain], mesh: Mesh,
                  wlen0_um: float, launch_wlen0_um: float) -> Domain:
    """Fit readout planes far enough from the walls for every port's launch."""
    from ..components.declarative import default_source_offset_um
    from ..components.grid import _structure_index

    domain = domain if domain is not None else Domain()
    n_bg = math.sqrt(float(imported.background.permittivity))
    n_max = max([_structure_index(s, wlen0_um) for s in imported.structures] + [n_bg])
    if mesh.dl_um is not None and not isinstance(mesh.dl_um, tuple):
        dl_bg = float(mesh.dl_um)
        cpw = None
    elif mesh.cells_per_wlen is not None:
        cpw = float(mesh.cells_per_wlen)
        dl_bg = wlen0_um / ((n_max if mesh.uniform else n_bg) * cpw)
    else:
        cpw = None
        dl_bg = min(float(v) for v in mesh.dl_um if v is not None)

    margins = []
    for port in imported.ports:
        if cpw is None:
            port_cell = dl_bg
        elif mesh.uniform:
            port_cell = dl_bg
        else:
            n_port = n_max if port.medium is None else max(
                _structure_index(Structure(
                    geometry=Box(center_um=(0.0, 0.0, 0.0), size_um=(1.0, 1.0, 1.0)),
                    medium=port.medium), wlen0_um), n_bg)
            port_cell = wlen0_um / (n_port * cpw)
        standoff = (float(port.source_offset_um) if port.source_offset_um is not None
                    else default_source_offset_um(launch_wlen0_um, n_bg))
        base_margin = (float(domain.port_margin_um) if domain.port_margin_um is not None
                       else 10.0 * port_cell)
        # The fitted extent rounds to whole boundary cells. One extra cell
        # preserves the requested margin at the realized PML inner face.
        rounding_cell = mesh.spacing(_AXIS_INDEX[port.axis]) or dl_bg
        margins.append(standoff + base_margin + rounding_cell)
    if not margins:
        return domain
    return domain.model_copy(update={"port_margin_um": max(margins)})


def plan_component_smatrix(
    component,
    layer_stack=None,
    *,
    wlens_um,
    mesh: Mesh,
    domain: Optional[Domain] = None,
    run=None,
    source_time=None,
    drive: Optional[Sequence[str]] = None,
    materials: Optional[Mapping[str, MaterialSpec]] = None,
    background=None,
    port_names: Optional[Sequence[str]] = None,
    mode: Union[str, Mapping[str, str]] = "TE0",
    port_level: Union[None, str, Mapping[str, str]] = None,
    extension_um: Optional[float] = None,
    simulation_kwargs: Optional[Mapping[str, Any]] = None,
):
    """Prepare the S-matrix study of a component without running anything.

    Imports ``component`` (see :func:`import_component` for ``layer_stack``
    and the import options; an :class:`ImportedComponent` is used as it is),
    with materials evaluated at the band centre of ``wlens_um``; builds the
    fitted :meth:`ImportedComponent.simulation` with ``mesh``, ``domain``,
    ``run`` and ``simulation_kwargs``; and plans one driven simulation per port
    with :func:`photonhub.analysis.plan_smatrix` at the frequencies of
    ``wlens_um``. ``source_time`` defaults to the pulse the scene's own ports
    take for ``wlens_um`` (a single wavelength gets a short pulse).

    ``mesh`` has no default: the resolution is your convergence choice, and the
    S-parameters inherit it. Call ``quote()`` on the plan for the cost, then
    ``run()``.

    The study fit leaves each launch plane at least the default ten port cells
    from the absorbing layers. When ``domain.port_margin_um`` is set, that
    margin is measured from the launch plane. A symmetry fold that mirrors
    away any port is refused because it would pair that port's drive with its
    mirror image; every port must lie on every symmetry plane. The symmetry
    must match each port mode's tangential electric-field parity (``-1`` is the
    odd electric/PEC mirror and ``+1`` is the even magnetic/PMC mirror), or the
    run can read a wrong mode and report plausible but incorrect S-parameters.

    Returns
    -------
    photonhub.analysis.SMatrixPlan
    """
    from ..analysis.smatrix_driver import plan_smatrix
    from ..components.declarative import band

    freqs, wlen0, pulse = band(wlens_um, None)
    imported = _imported(component, layer_stack, wlen_um=wlen0, materials=materials,
                         background=background, port_names=port_names, mode=mode,
                         port_level=port_level, extension_um=extension_um)
    from ..constants import c0

    launch_wlen0 = (c0 / float(source_time.freq0_hz) * 1e6
                    if source_time is not None else wlen0)
    study_domain = _study_domain(imported, domain, mesh, wlen0, launch_wlen0)
    sim = imported.simulation(wlens_um=wlens_um, mesh=mesh, domain=study_domain, run=run,
                              **dict(simulation_kwargs or {}))
    if sim._fold is not None and sim._fold.mirrored_ports:
        names = sorted(sim._fold.mirrored_ports)
        raise ValueError(f"an S-matrix study cannot mirror away ports {names}; "
                         "remove that symmetry plane or place every port on it")
    if sim._fold is not None:
        for axis, symmetry in enumerate(sim.symmetry):
            if symmetry == 0:
                continue
            plane_user_um = sim._fold.planes[axis]
            plane_um = plane_user_um - sim.origin_um[axis]
            off_plane = sorted(port.name for port in sim.ports
                               if abs(port.center_um[axis] - plane_um) > 1e-6)
            if off_plane:
                raise ValueError(
                    f"an S-matrix study needs every port centre on symmetry plane "
                    f"{'xyz'[axis]} = {plane_user_um:g} um; ports {off_plane} lie off it")
    # the fitted scene's ports, already in its corner frame; by default the
    # pulse the scene's own ports take (one wavelength gets a short pulse)
    return plan_smatrix(sim, sim.ports, freqs_hz=freqs, drive=drive,
                        source_time=source_time if source_time is not None else pulse)


def run_component_smatrix(
    component,
    layer_stack=None,
    *,
    wlens_um,
    mesh: Mesh,
    domain: Optional[Domain] = None,
    run=None,
    source_time=None,
    drive: Optional[Sequence[str]] = None,
    materials: Optional[Mapping[str, MaterialSpec]] = None,
    background=None,
    port_names: Optional[Sequence[str]] = None,
    mode: Union[str, Mapping[str, str]] = "TE0",
    port_level: Union[None, str, Mapping[str, str]] = None,
    extension_um: Optional[float] = None,
    simulation_kwargs: Optional[Mapping[str, Any]] = None,
    runner: Union[str, Callable[[Dict[str, Any]], Mapping[str, Any]]] = "local",
    path_dir=None,
    max_workers: int = 1,
    allow_partial: bool = False,
    **runner_kwargs: Any,
):
    """One call from a component to its S-matrix: :func:`plan_component_smatrix`
    then :meth:`~photonhub.analysis.SMatrixPlan.run`.

    ``runner="local"`` (default) runs on the local solver; ``runner="cloud"``
    submits through ``ph.cloud.Batch``, which quotes each job and holds it to
    your spending ceiling. Pass ``device`` and ``solver`` through the extra
    keywords for the cloud service; a callable runner is used as
    :meth:`~photonhub.analysis.SMatrixPlan.run` describes. ``path_dir``,
    ``max_workers``, ``allow_partial`` and any other keyword go to that method.
    ``mesh`` has no default (see :func:`plan_component_smatrix`).
    The symmetry must match each port mode's tangential electric-field parity
    (``-1`` is the odd electric/PEC mirror and ``+1`` is the even magnetic/PMC
    mirror), or the run can read a wrong mode and report plausible but
    incorrect S-parameters.

    Returns
    -------
    photonhub.analysis.SMatrixResult
        ``S`` has dims ``(port_out, port_in, f)`` labelled with the port names.
    """
    plan = plan_component_smatrix(
        component, layer_stack, wlens_um=wlens_um, mesh=mesh, domain=domain, run=run,
        source_time=source_time, drive=drive, materials=materials, background=background,
        port_names=port_names, mode=mode, port_level=port_level, extension_um=extension_um,
        simulation_kwargs=simulation_kwargs)
    return plan.run(runner, path_dir=path_dir, max_workers=max_workers,
                    allow_partial=allow_partial, **runner_kwargs)
