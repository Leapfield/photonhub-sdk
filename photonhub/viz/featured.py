"""Export a simulation and its recorded fields for the PhotonHub 3D viewer.

The exporter turns a :class:`~photonhub.Simulation` plus its run data into
one JSON document, the contract between Python and the viewer:

- ``structures``: the top-view outline rings of every structure (union of the
  structures sharing a z-extent, clipped to the non-PML interior), with their
  ``z0``/``z1``;
- ``ports``: positions, outward directions, roles and labels (from the
  notebook's port dict, the same one the GDS sidecar fills);
- ``planes``: one recorded field plane as |E| plus one phasor (the dominant
  E component, or the component or pair named by ``phase_component``),
  normalised to the 99.7th percentile, downsampled to at most
  ``max_samples`` along its long axis;
- the placement rule (``"surface"`` when a horizontal plane lies inside a
  structure, else ``"cut"``) and the periodic-extension rule for the
  quasi-2D examples (structures spanning a periodic axis are widened to
  ``periodic_extent_um`` and the plane is tiled with the source's Bloch
  phase).

The exporter carries NO style: colours, camera, lighting and bloom live in
the viewer's ``STYLE`` object, so every device renders the same way.

shapely is required for the polygon union/clip and is part of the optional
``photonhub[viz]`` extra; it is imported lazily with an install hint.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import c0
from . import _geometry as geom

C0_M_PER_S = c0
_SHAPELY_HINT = (
    "export_scene requires shapely (the optional viz extra). Install it with:\n"
    "    pip install photonhub[viz]"
)
_ABSORBING = ("pml", "absorber")
_PERIODIC = ("periodic", "bloch")
_LUTS = ("inferno",)


def _column_axis(sim, ax: str, periodic_extent_um: float) -> bool:
    """True for a periodic in-plane axis thinner than the displayed extent: a
    quasi-2D column, which is widened to the extent and tiled. A periodic axis
    wider than that (a large periodic-padded device) is shown as it is."""
    if ax == "z" or _boundary_kind(sim, ax) not in _PERIODIC:
        return False
    return float(sim.size_um["xyz".index(ax)]) < periodic_extent_um


def _require_shapely():
    try:
        import shapely
        from shapely.geometry import Polygon as SPolygon  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without shapely
        raise ImportError(_SHAPELY_HINT) from exc
    return shapely


# --------------------------------------------------------------------------- #
# Scene container.
# --------------------------------------------------------------------------- #

@dataclass
class Scene:
    """The exported scene document. ``write(path)`` stores it as compact JSON."""

    doc: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the scene document itself. Mutating it also changes this scene."""
        return self.doc

    def to_json(self) -> str:
        """Serialize the scene as compact JSON. Reject non-finite numeric values."""
        return json.dumps(self.doc, separators=(",", ":"), allow_nan=False)

    def write(self, path) -> Path:
        """Write compact scene JSON to ``path``, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @property
    def size_bytes(self) -> int:
        """UTF-8 byte length of the compact serialized scene document."""
        return len(self.to_json().encode("utf-8"))


# --------------------------------------------------------------------------- #
# Grid / boundary helpers.
# --------------------------------------------------------------------------- #

def _corner_frame(data, name, sim):
    """The recorded plane as the engine wrote it, in the wire's corner frame the
    exporter works in (design spec §4.4): a result hands out user-frame
    coordinates, and the plane of a folded device unfolded, while the
    exporter mirrors a §20 half plane itself."""
    from ..components.frame import wire_array
    return wire_array(data, name)


def _plane_names(sim, data, plane) -> list[str]:
    """The monitors ``plane`` names: itself, the group ``{plane}_0``,
    ``{plane}_1``, ... that :meth:`ProfileMonitor.sections` records, or, when
    ``plane`` is a list, the monitors it lists."""
    if not isinstance(plane, str):
        names = [str(n) for n in plane]
        if not names:
            raise ValueError("plane lists no monitors")
        return names
    have = {getattr(m, "name", None) for m in getattr(sim, "monitors", ())}
    if isinstance(data, Mapping):
        have |= set(data)
    if plane in have:
        return [plane]
    group = []
    while f"{plane}_{len(group)}" in have:
        group.append(f"{plane}_{len(group)}")
    return group or [plane]


def _dl_um(sim, plane_da=None) -> tuple[float, float, float]:
    """Cell size per axis: the uniform spacing, or, on a graded grid, the
    spacing of the recorded plane's coordinates at the domain edge (the PML
    is always laid on the coarse outer cells)."""
    grid = sim.grid
    if getattr(grid, "type", None) == "uniform":
        return (float(grid.dl_um),) * 3
    out = []
    for ax in "xyz":
        if plane_da is not None and ax in plane_da.coords and plane_da.sizes.get(ax, 1) > 1:
            c = np.asarray(plane_da.coords[ax].values, dtype=float)
            out.append(float(c[1] - c[0]))
        else:
            out.append(float(grid.dl_um))
    return tuple(out)


def _boundary_kind(sim, ax: str) -> str:
    return str(getattr(sim.boundaries, ax))


def _layer_thickness_um(sim, ax: str, dl: float) -> float:
    kind = _boundary_kind(sim, ax)
    if kind == "pml":
        return float(sim.pml_num_layers) * dl
    if kind == "absorber":
        return float(sim.absorber_num_layers) * dl
    return 0.0


def _interior(sim, dl3, periodic_extent_um: float):
    """Per-axis (lo, hi) of the region shown: inside the boundary layers on
    absorbing axes, the full domain on the others, and the periodic extent
    (centred on the domain) on periodic axes."""
    lo, hi = [], []
    for i, ax in enumerate("xyz"):
        size = float(sim.size_um[i])
        t = _layer_thickness_um(sim, ax, dl3[i])
        if _column_axis(sim, ax, periodic_extent_um):
            c = size / 2.0
            lo.append(c - periodic_extent_um / 2.0)
            hi.append(c + periodic_extent_um / 2.0)
        else:
            lo.append(t)
            hi.append(size - t)
    # NUMERICS.md §20: on a symmetry axis the min face is a mirror, not a
    # wall, and only half the device was simulated. The figure shows the whole
    # device: the shown region is unfolded about that face.
    for i, s in enumerate(_symmetry(sim)):
        if s != 0:
            lo[i] = -hi[i]
    return lo, hi


def _symmetry(sim) -> tuple[int, int, int]:
    sym = getattr(sim, "symmetry", None) or (0, 0, 0)
    return (int(sym[0]), int(sym[1]), int(sym[2]))


# --------------------------------------------------------------------------- #
# Structures -> top-view rings.
# --------------------------------------------------------------------------- #

def _circle(cx: float, cy: float, r: float, n: int = 64):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def _footprint(structure):
    """``(polygon, z0, z1)`` of one structure seen from above, or None for a
    geometry the standard does not draw. Polygons and cylinders must be
    extruded along z, the figure's up axis."""
    shapely = _require_shapely()
    from shapely.geometry import Polygon as SPolygon

    g = structure.geometry
    kind = getattr(g, "type", None)
    if kind == "box":
        cx, cy, cz = g.center_um
        sx, sy, sz = g.size_um
        return (shapely.box(cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2),
                cz - sz / 2, cz + sz / 2)
    if kind == "polyslab":
        if g.axis != "z":
            raise ValueError(
                f"export_scene draws polygons extruded along z; this one is "
                f"extruded along {g.axis!r}")
        lo, hi = g.slab_bounds_um
        return SPolygon([(float(u), float(v)) for u, v in g.vertices_um]), float(lo), float(hi)
    if kind == "cylinder":
        if g.axis != "z":
            raise ValueError(
                f"export_scene draws cylinders extruded along z; this one is "
                f"extruded along {g.axis!r}")
        cx, cy, cz = g.center_um
        outer = SPolygon(_circle(cx, cy, g.radius_um))
        if g.inner_radius_um > 0:
            outer = outer.difference(SPolygon(_circle(cx, cy, g.inner_radius_um)))
        sweep = g.angle_stop_rad - g.angle_start_rad
        if sweep < 2 * math.pi - 1e-9:
            wedge = SPolygon([(cx, cy)] + [
                (cx + 2 * g.radius_um * math.cos(g.angle_start_rad + sweep * k / 48),
                 cy + 2 * g.radius_um * math.sin(g.angle_start_rad + sweep * k / 48))
                for k in range(49)])
            outer = outer.intersection(wedge)
        return outer, cz - g.length_um / 2, cz + g.length_um / 2
    if kind == "sphere":
        cx, cy, cz = g.center_um
        return SPolygon(_circle(cx, cy, g.radius_um)), cz - g.radius_um, cz + g.radius_um
    raise ValueError(f"export_scene does not know how to draw geometry {kind!r}")


def _extend_periodic(poly, sim, dl3, periodic_extent_um: float):
    """Widen a footprint that spans the domain along a periodic axis to the
    displayed periodic extent (the quasi-2D column becomes a plate). A footprint
    that does not span it (a pillar in its unit cell) is repeated at the period
    instead, as the walls repeat it and as the plane's field is tiled."""
    from shapely import affinity
    from shapely.ops import unary_union

    minx, miny, maxx, maxy = poly.bounds
    for i, ax in enumerate("xy"):
        if not _column_axis(sim, ax, periodic_extent_um):
            continue
        size = float(sim.size_um[i])
        lo, hi = (minx, maxx) if ax == "x" else (miny, maxy)
        if lo <= dl3[i] and hi >= size - dl3[i]:
            factor = periodic_extent_um / size
            centre = (float(sim.size_um[0]) / 2.0, float(sim.size_um[1]) / 2.0)
            poly = affinity.scale(poly, xfact=factor if ax == "x" else 1.0,
                                  yfact=factor if ax == "y" else 1.0, origin=centre)
        elif size > 0.0:
            n_rep = int(math.ceil(periodic_extent_um / size)) + 1
            poly = unary_union([affinity.translate(poly, xoff=m * size if ax == "x" else 0.0,
                                                   yoff=m * size if ax == "y" else 0.0)
                                for m in range(-n_rep, n_rep + 1)])
    return poly


def _is_air(medium) -> bool:
    """A plain vacuum/air medium (no loss, poles, anisotropy or PEC): such a
    structure carves the background (air above a substrate) and is not drawn."""
    if getattr(medium, "pec", None) or getattr(medium, "permittivity_xyz", None):
        return False
    if any(getattr(medium, k, None) for k in ("lorentz", "poles", "drude")):
        return False
    if float(getattr(medium, "conductivity_s_per_m", 0.0) or 0.0) > 0.0:
        return False
    return float(getattr(medium, "permittivity", 1.0)) <= 1.0 + 1e-9


def _structures(sim, dl3, lo, hi, periodic_extent_um: float, simplify_um: float, only=None):
    """The bodies the picture draws, in the user's frame: ``(z0, z1, polygon,
    eps)`` per z extent, cladding left out. :func:`_serialize_bodies` puts them
    in the scene's frame; the caller reads their z extents first, so the origin
    and a horizontal plane's placement follow what is actually drawn. ``only``
    lists the indices of ``sim.structures`` to draw (air carve-outs always
    carve); ``None`` draws them all."""
    shapely = _require_shapely()
    from shapely.ops import unary_union

    clip = shapely.box(lo[0], lo[1], hi[0], hi[1])
    sym = _symmetry(sim)
    by_extent: dict[tuple[float, float], list] = {}
    eps_by_extent: dict[tuple[float, float], float] = {}
    last_body: dict[tuple[float, float], int] = {}
    carves: list[tuple[float, float, Any, int]] = []

    def _mirrored(poly):
        """The structure completed across whichever symmetry faces the run used
        (a structure drawn whole across a face is unchanged by the union)."""
        from shapely import affinity
        if sym[0]:
            poly = unary_union([poly, affinity.scale(poly, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0, 0.0))])
        if sym[1]:
            poly = unary_union([poly, affinity.scale(poly, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0, 0.0))])
        return poly

    for idx, s in enumerate(sim.structures):
        poly, z0, z1 = _footprint(s)
        poly = _mirrored(_extend_periodic(poly, sim, dl3, periodic_extent_um))
        if _is_air(s.medium):
            # An air structure carves whatever it sits in: a hole etched through
            # a membrane, a trench in a slab. Where it hangs in the background
            # instead (an air box above a substrate) nothing shares its extent
            # and it draws nothing, as before.
            carves.append((z0, z1, poly, idx))
            continue
        if only is not None and idx not in only:
            continue
        key = (round(z0, 6), round(z1, 6))
        by_extent.setdefault(key, []).append(poly)
        eps_by_extent[key] = float(getattr(s.medium, "permittivity", 1.0))
        last_body[key] = idx
    bodies = []
    for (z0, z1), polys in sorted(by_extent.items()):
        merged = unary_union(polys).intersection(clip)
        # §9 last-wins: only an air structure listed AFTER the body it overlaps
        # replaces its material, and only where their z extents meet.
        cut = [p for (a0, a1, p, i) in carves
               if i > last_body[(round(z0, 6), round(z1, 6))]
               and a1 > z0 + 1e-9 and a0 < z1 - 1e-9]
        if cut and not merged.is_empty:
            merged = merged.difference(unary_union(cut))
        if merged.is_empty:
            continue
        bodies.append((z0, z1, merged.simplify(simplify_um), eps_by_extent[(z0, z1)]))

    return [b for b in bodies if not _is_cladding(*b, bodies)]


def _serialize_bodies(bodies, origin):
    out = []
    for z0, z1, merged, eps in bodies:
        geoms = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        rings, holes = [], []
        for g in geoms:
            if g.is_empty or g.geom_type != "Polygon":
                continue
            rings.append([[round(x - origin[0], 4), round(y - origin[1], 4)]
                          for x, y in g.exterior.coords])
            holes.append([[[round(x - origin[0], 4), round(y - origin[1], 4)]
                           for x, y in r.coords] for r in g.interiors])
        if rings:
            body = {"rings": rings, "z0": round(z0 - origin[2], 4),
                    "z1": round(z1 - origin[2], 4), "kind": "core", "eps": eps}
            if any(holes):        # optional: a viewer without hole support still draws the bodies
                body["holes"] = holes
            out.append(body)
    return out


def _is_cladding(z0: float, z1: float, poly, eps: float, bodies) -> bool:
    """Whether this body is another body's cladding: it surrounds a guide of
    higher index, above, below and all around it.

    The picture shows what guides the light. The medium around a guide is not
    drawn (the background never is), and drawn it would be worse than absent:
    an upper cladding stands in front of every cross-section, and a horizontal
    field plane inside the guide would be lifted onto the cladding's top face,
    far above the device. A stack of layers (a buried oxide under a grating,
    a substrate under that) surrounds nothing and is unaffected."""
    for oz0, oz1, other, other_eps in bodies:
        if other is poly or other_eps <= eps + 1e-9:
            continue
        if oz0 >= z0 - 1e-9 and oz1 <= z1 + 1e-9 and poly.covers(other):
            return True
    return False


# --------------------------------------------------------------------------- #
# The field plane.
# --------------------------------------------------------------------------- #

def _select_frequency(da, freq_hz: float | None):
    if "f" not in da.dims:
        return da, float(da.attrs.get("freqs_hz", [float("nan")])[0])
    f = np.asarray(da.coords["f"].values, dtype=float)
    if freq_hz is None:
        if f.size > 1:
            raise ValueError(
                f"monitor {da.name!r} records {f.size} frequencies; pass freq_hz= "
                "to choose the one to draw")
        freq_hz = float(f[0])
    i = int(np.argmin(np.abs(f - float(freq_hz))))
    return da.isel(f=i), float(f[i])


def _box_average(a: np.ndarray, fv: int, fu: int) -> np.ndarray:
    if fv == 1 and fu == 1:
        return a
    nv = (a.shape[0] // fv) * fv
    nu = (a.shape[1] // fu) * fu
    a = a[:nv, :nu]
    return a.reshape(nv // fv, fv, nu // fu, fu).mean(axis=(1, 3))


# The phase animation draws Re[F e^{-i w t}], so box averaging must leave the travelling wave resolved. A
# sample that spans half a guided wavelength averages the phasor away, and past 180 degrees per sample the
# wave aliases and appears to run backwards: a 59 um device at 181 samples did exactly that in its silicon.
_PHASE_STEP_MAX_DEG = 90.0     # at most a quarter wave per sample: four samples per guided wavelength
_MIN_AXIS_SAMPLES = 64         # an axis gives up samples to the other one no further than this
_MAX_AXIS_SAMPLES = 1024       # and no axis grows past this: a texture every WebGL device takes


def _phase_step_deg(phasor: np.ndarray, amp: np.ndarray, axis: int) -> float:
    """The median phase advance per recorded cell along ``axis``, over the bright part of the plane.
    ``phasor`` is one component ``[v, u]`` or a stack of them ``[c, v, u]``, whose advances add by power."""
    stack = phasor if phasor.ndim == 3 else phasor[None]
    a = np.moveaxis(stack, axis + 1, -1)
    step = np.abs(np.angle((a[..., 1:] * np.conj(a[..., :-1])).sum(axis=0)))
    w = np.moveaxis(amp, axis, -1)
    bright = np.minimum(w[..., 1:], w[..., :-1]) > 0.3 * float(w.max() or 1.0)
    return float(np.degrees(np.median(step[bright]))) if bright.any() else 0.0


def _resolve_phase(amp, phasor, fv: int, fu: int, max_samples: int) -> tuple[int, int]:
    """Box factors ``(fv, fu)`` that keep the wave resolved along the axis it travels.

    An axis whose factor would advance the phase more than ``_PHASE_STEP_MAX_DEG`` per sample keeps more
    samples. The plane's budget stays what a square plane has, ``max_samples ** 2``: the other axis pays,
    down to ``_MIN_AXIS_SAMPLES``, and past that the travelling axis takes what the budget and
    ``_MAX_AXIS_SAMPLES`` leave; a device hundreds of wavelengths long stays under-resolved, as it was.
    A plane that is already resolved, which is every device a few tens of wavelengths long, is returned
    unchanged."""
    nv0, nu0 = amp.shape
    f = [fv, fu]
    for ax in (0, 1):
        step = _phase_step_deg(phasor, amp, ax)
        if step * f[ax] <= _PHASE_STEP_MAX_DEG:
            continue
        other, n0, m0 = 1 - ax, (nv0, nu0)[ax], (nv0, nu0)[1 - ax]
        f[ax] = max(1, int(_PHASE_STEP_MAX_DEG // step)) if step > 0 else f[ax]
        budget = max_samples * max_samples
        while (n0 // f[ax]) * (m0 // f[other]) > budget and m0 // (f[other] + 1) >= _MIN_AXIS_SAMPLES:
            f[other] += 1
        while (n0 // f[ax]) * (m0 // f[other]) > budget or n0 // f[ax] > _MAX_AXIS_SAMPLES:
            f[ax] += 1
    return f[0], f[1]


def _phase_names(phase_component) -> tuple[str, ...]:
    if phase_component is None:
        return ()
    names = (phase_component,) if isinstance(phase_component, str) else tuple(str(c) for c in phase_component)
    if not 1 <= len(names) <= 2 or len(set(names)) != len(names):
        raise ValueError(f"phase_component names one recorded component or a pair of them, not {phase_component!r}")
    return names


def _major_amplitude(stack: np.ndarray) -> np.ndarray:
    """The largest instantaneous magnitude of the field the stacked components make up: |F| for one
    component, the semi-major axis of the ellipse a pair traces."""
    if stack.shape[0] == 1:
        return np.abs(stack[0])
    a, b = stack
    return np.sqrt((np.abs(a) ** 2 + np.abs(b) ** 2 + np.abs(a * a + b * b)) / 2.0)


def _major_axis_phasor(stack: np.ndarray) -> np.ndarray:
    """The pair's field along the major axis of its ellipse, as one phasor. A device that turns the
    polarization hands the light from one component to the other, so neither follows it end to end; the
    major axis does. Its sign is a convention, which :func:`_agree_on_sign` settles."""
    a, b = stack
    return _agree_on_sign(_major_amplitude(stack) * np.exp(0.5j * np.angle(a * a + b * b)))


def _agree_on_sign(f: np.ndarray) -> np.ndarray:
    """``f`` with every sample's free sign chosen to agree with its neighbours. The viewer draws |Re F| and
    interpolates F between samples, so two neighbours of opposite sign would draw a dark seam that is not in
    the field. Grown from the brightest sample outward, brightest first, so that a disagreement that cannot
    be avoided (around a polarization singularity) falls where the field is darkest."""
    import heapq

    out = np.array(f, dtype=complex)
    mag = np.abs(out)
    nv, nu = out.shape
    done = np.zeros(out.shape, dtype=bool)
    start = tuple(int(k) for k in np.unravel_index(int(np.argmax(mag)), out.shape))
    heap = [(-float(mag[start]), start, start)]
    while heap:
        _, (i, j), (pi, pj) = heapq.heappop(heap)
        if done[i, j]:
            continue
        if (out[i, j] * np.conj(out[pi, pj])).real < 0.0:
            out[i, j] = -out[i, j]
        done[i, j] = True
        for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= ni < nv and 0 <= nj < nu and not done[ni, nj]:
                heapq.heappush(heap, (-float(mag[ni, nj]), (ni, nj), (i, j)))
    return out


def _spread_fronts(stack: np.ndarray, axis: int, spacing: int) -> np.ndarray:
    """``stack`` with its wave fronts ``spacing`` times further apart along ``axis`` (1 = u, 0 = v).

    A device a hundred wavelengths long, drawn a few hundred pixels long, has fronts a pixel or two apart:
    the animation shimmers and the shape of the mode under it cannot be read. The carrier is the
    power-weighted phase advance from one line of the plane to the next; all of it but one part in
    ``spacing`` is taken off. The envelope, the mode's lobes and the beating between modes are untouched,
    because only the phase common to a whole line changes. It follows the wave that carries the power, so it
    is for a travelling wave: a reflected one would have its fronts crowded instead."""
    a = np.moveaxis(stack, axis + 1, -1)
    advance = np.angle((a[..., 1:] * np.conj(a[..., :-1])).sum(axis=(0, 1)))
    carrier = np.concatenate([[0.0], np.cumsum(advance)])
    return np.moveaxis(a * np.exp(-1j * carrier * (1.0 - 1.0 / spacing)), -1, axis + 1)


def _propagated_plane(sim, data, monitor: str, dz_um: float):
    """The recorded plane's complex E reconstructed ``dz_um`` downstream of it
    (:func:`~photonhub.analysis.propagate_plane`), as a field DataArray at
    that height, so it can be drawn like a monitor plane."""
    import xarray as xr

    from ..analysis.propagate import propagate_plane

    da0 = _corner_frame(data, monitor, sim)
    spatial = [ax for ax in ("x", "y", "z") if ax in da0.dims]
    axis = [ax for ax in spatial if da0.sizes[ax] == 1][0]
    # a negative offset looks upstream of the recorded +axis wave
    res = propagate_plane(sim, data, monitor, [dz_um])
    u1, u2 = res["axes"]
    position = float(da0.coords[axis].values[0]) + dz_um
    # propagate_plane keeps the in-plane coordinates a result hands out, the
    # user's frame; the exporter crops and places in the wire's corner frame
    # (a plain mapping of planes is in the wire frame already, as wire_array reads it)
    from ..components.frame import frame_origin
    origin = frame_origin(sim) if callable(getattr(data, "wire", None)) else (0.0, 0.0, 0.0)
    vals = np.stack([res["e1"][0], res["e2"][0], res["en"][0]], axis=1)       # [f, c, n1, n2]
    coords = {"f": ("f", np.asarray(res["freqs_hz"], dtype=float)),
              "component": [f"E{u1}", f"E{u2}", f"E{axis}"],
              axis: (axis, np.asarray([position], dtype=float)),
              u1: (u1, np.asarray(res["coords1_um"], dtype=float) - origin["xyz".index(u1)]),
              u2: (u2, np.asarray(res["coords2_um"], dtype=float) - origin["xyz".index(u2)])}
    return xr.DataArray(vals[:, :, None], dims=("f", "component", axis, u1, u2), coords=coords,
                        name=f"{monitor} +{dz_um:g} um")


def _plane(sim, da, *, freq_hz, dl3, lo, hi, origin, structures_z, max_samples: int,
           periodic_extent_um: float, exclude_um=(), exclude_radius_um: float = 0.15,
           phase_component: str | Sequence[str] | None = None, front_spacing: int = 1):
    da, f_used = _select_frequency(da, freq_hz)
    spatial = [ax for ax in ("x", "y", "z") if ax in da.dims]
    normals = [ax for ax in spatial if da.sizes[ax] == 1]
    if len(normals) != 1:
        raise ValueError(
            f"monitor {da.name!r} is not a plane: sizes "
            f"{ {ax: da.sizes[ax] for ax in spatial} } (exactly one axis must have 1 sample)")
    axis = normals[0]
    h_ax, v_ax = geom.in_plane_axes(axis)          # (u, v)
    position = float(da.coords[axis].values[0])
    da = da.isel({axis: 0})

    comps = [str(c) for c in da.coords["component"].values]
    vals = np.asarray(da.transpose("component", v_ax, h_ax).values)  # [c, v, u]
    # |E| sums the E components only: a monitor that also records H (for a
    # phase_component such as Hz) must not add it to the envelope.
    e_idx = [i for i, c in enumerate(comps) if c.startswith("E")] or list(range(len(comps)))
    power = (np.abs(vals) ** 2).sum(axis=(1, 2))
    amp = np.sqrt((np.abs(vals[e_idx]) ** 2).sum(axis=0)).astype(float)
    shown = _phase_names(phase_component) or (comps[max(e_idx, key=lambda i: power[i])],)
    for name in shown:
        if name not in comps:
            raise ValueError(f"phase_component {name!r} is not recorded by monitor {da.name!r} "
                             f"(it records {comps}); add it to the monitor's fields=")
    phasor = vals[[comps.index(name) for name in shown]].astype(complex)   # [c, v, u]: one component, or a pair
    u = np.asarray(da.coords[h_ax].values, dtype=float)
    v = np.asarray(da.coords[v_ax].values, dtype=float)

    # Crop to the shown region along both in-plane axes.
    iu = "xyz".index(h_ax)
    iv = "xyz".index(v_ax)
    ku = np.where((u >= lo[iu] - 1e-9) & (u <= hi[iu] + 1e-9))[0]
    kv = np.where((v >= lo[iv] - 1e-9) & (v <= hi[iv] + 1e-9))[0]
    u, v = u[ku], v[kv]
    amp, phasor = amp[np.ix_(kv, ku)], phasor[:, kv][:, :, ku]
    if front_spacing > 1:
        along_u = (u[-1] - u[0]) >= (v[-1] - v[0])          # the fronts are spread along the plane's long axis
        if _column_axis(sim, h_ax if along_u else v_ax, periodic_extent_um):
            raise ValueError("front_spacing spreads the fronts along the plane's long axis, which is a periodic "
                             "axis here: the tiled periods would no longer join")
        phasor = _spread_fronts(phasor, 1 if along_u else 0, front_spacing)

    # The colour scale comes from the field, not from a point source's singular
    # cell: samples within ``exclude_radius_um`` of a dipole are left out of the
    # percentile (they are still drawn) and, because a periodic axis would
    # repeat them into a row of hot spots, in-filled from their row before
    # tiling.
    keep = np.ones(amp.shape, dtype=bool)
    iu_ax, iv_ax = "xyz".index(h_ax), "xyz".index(v_ax)
    for pt in exclude_um:
        du = u[None, :] - pt[iu_ax]
        dv = v[:, None] - pt[iv_ax]
        keep &= (du * du + dv * dv) > exclude_radius_um ** 2
    peak = float(np.percentile(amp[keep] if keep.any() else amp, 99.7)) or 1.0
    phase_peak = peak
    if phase_component is not None:
        mag = _major_amplitude(phasor)
        phase_peak = float(np.percentile(mag[keep] if keep.any() else mag, 99.7)) or 1.0
    periodic_in_plane = any(_column_axis(sim, ax, periodic_extent_um) for ax in (h_ax, v_ax))
    if periodic_in_plane and not keep.all():
        for r in np.where(~keep.all(axis=1))[0]:
            good = keep[r]
            if good.any():
                amp[r, ~good] = amp[r, good].mean()
                phasor[:, r, ~good] = phasor[:, r, good].mean(axis=1, keepdims=True)
    # A travelling wave has a flat |E| (nothing reflected to beat against): its
    # static frame is a phase snapshot, otherwise the figure is a blank plane.
    body = amp[keep] if keep.any() else amp
    flat = bool(body.size > 8 and body.std() < 0.2 * body.mean())

    # Periodic tiling: along a periodic in-plane axis the recorded column is one
    # period of the field, so it is repeated (with the source's Bloch phase per
    # period) until the displayed extent is covered, then cropped to it.
    k_bloch = tuple(sim.bloch_k_per_um or (0.0, 0.0, 0.0))
    for which, ax in (("u", h_ax), ("v", v_ax)):
        i = "xyz".index(ax)
        if not _column_axis(sim, ax, periodic_extent_um):
            continue
        coord = u if which == "u" else v
        period = float(sim.size_um[i])
        if coord[-1] - coord[0] >= periodic_extent_um or period <= 0:
            continue
        c = period / 2.0
        n_rep = int(math.ceil(periodic_extent_um / period)) + 1
        coords, amps, phs = [], [], []
        for m in range(-n_rep, n_rep + 1):
            shift = np.exp(1j * k_bloch[i] * m * period)
            coords.append(coord + m * period)
            amps.append(amp if which == "u" else amp)
            phs.append(phasor * shift)
        axis_i = 1 if which == "u" else 0
        coord_t = np.concatenate(coords)
        amp_t = np.concatenate(amps, axis=axis_i)
        ph_t = np.concatenate(phs, axis=axis_i + 1)
        keep = np.where((coord_t >= c - periodic_extent_um / 2.0 - 1e-9)
                        & (coord_t <= c + periodic_extent_um / 2.0 + 1e-9))[0]
        if which == "u":
            u, amp, phasor = coord_t[keep], amp_t[:, keep], ph_t[:, :, keep]
        else:
            v, amp, phasor = coord_t[keep], amp_t[keep, :], ph_t[:, keep, :]

    # Unfold about a symmetry face (NUMERICS.md §20): |E| is even; each
    # animated component's phasor carries its parity — a PEC mirror (-1) keeps
    # the component normal to the face and flips the tangential ones, a PMC
    # (+1) the reverse. The recorded samples start at the first cell centre
    # past the face, so the mirrored copy adds no duplicate row.
    sym = _symmetry(sim)
    for which, ax in (("u", h_ax), ("v", v_ax)):
        i = "xyz".index(ax)
        if sym[i] == 0:
            continue
        sign = np.ones((len(shown), 1, 1))
        for c, name in enumerate(shown):
            normal = name.endswith(ax)
            sign[c] = 1.0 if normal == (sym[i] == -1) else -1.0
            if name.startswith("H"):
                sign[c] = -sign[c]          # H is a pseudovector: a mirror flips what it keeps for E
        if which == "u":
            u = np.concatenate([-u[::-1], u])
            amp = np.concatenate([amp[:, ::-1], amp], axis=1)
            phasor = np.concatenate([sign * phasor[:, :, ::-1], phasor], axis=2)
        else:
            v = np.concatenate([-v[::-1], v])
            amp = np.concatenate([amp[::-1, :], amp], axis=0)
            phasor = np.concatenate([sign * phasor[:, ::-1, :], phasor], axis=1)

    # Downsample by box averaging so each axis has <= max_samples. Each axis is
    # capped on its own: a plane along a long device keeps its short axis at
    # full resolution instead of dividing it by the long axis's factor.
    fu = max(1, math.ceil(amp.shape[1] / max_samples))
    fv = max(1, math.ceil(amp.shape[0] / max_samples))
    fv, fu = _resolve_phase(amp, phasor, fv, fu, max_samples)
    if fu > 1 or fv > 1:
        amp = _box_average(amp, fv, fu)
        phasor = np.stack([_box_average(c.real, fv, fu) + 1j * _box_average(c.imag, fv, fu) for c in phasor])
        u = _box_average(u[None, :], 1, fu)[0]
        v = _box_average(v[:, None], fv, 1)[:, 0]
    # A pair becomes one phasor only now: each component is a smooth field and averages cleanly, while the
    # major axis is defined up to a sign and would not.
    field = phasor[0] if len(shown) == 1 else _major_axis_phasor(phasor)
    re, im = field.real, field.imag

    # float64 before rounding: a rounded float32 prints with 17 digits in JSON
    amp = np.asarray(amp, dtype=np.float64) / peak
    re = np.asarray(re, dtype=np.float64) / phase_peak
    im = np.asarray(im, dtype=np.float64) / phase_peak

    # Placement: a horizontal plane inside a structure is painted on that
    # structure's top face; anything else is a framed cut.
    placement, top_z = "cut", None
    if axis == "z":
        tol = dl3[2] / 2.0
        containing = [z1 for (z0, z1) in structures_z if z0 - tol <= position <= z1 + tol]
        if containing:
            placement, top_z = "surface", round(max(containing) - origin[2], 4)

    ou, ov = origin["xyz".index(h_ax)], origin["xyz".index(v_ax)]
    plane = {
        "monitor": str(da.name), "placement": placement, "axis": axis,
        "position": round(position - origin["xyz".index(axis)], 4),
        "u_axis": h_ax, "v_axis": v_ax,
        "u0": round(float(u[0]) - ou, 4), "u1": round(float(u[-1]) - ou, 4),
        "v0": round(float(v[0]) - ov, 4), "v1": round(float(v[-1]) - ov, 4),
        "nu": int(amp.shape[1]), "nv": int(amp.shape[0]),
        "component": "+".join(shown),
        "amp": np.round(amp, 3).ravel().tolist(),      # 3 decimals: finer than the 256-level LUT,
        "re": np.round(re, 3).ravel().tolist(),        # and it keeps a 256 x 256 plane near 1 MB
        "im": np.round(im, 3).ravel().tolist(),
    }
    if top_z is not None:
        plane["top_z"] = top_z
    if front_spacing > 1:
        plane["front_spacing"] = int(front_spacing)
    plane["static_frame"] = "snapshot" if flat else "envelope"
    return plane, f_used


# --------------------------------------------------------------------------- #
# Ports, sources, LUTs.
# --------------------------------------------------------------------------- #

def _ports(ports: Mapping[str, Mapping[str, Any]] | None, port_in, port_through,
           offset, origin) -> list[dict[str, Any]]:
    if not ports:
        return []
    if port_in is not None and port_in not in ports:
        raise KeyError(f"port_in {port_in!r} is not one of the ports {list(ports)}")
    if port_through is not None and port_through not in ports:
        raise KeyError(f"port_through {port_through!r} is not one of the ports {list(ports)}")
    out = []
    for name, p in ports.items():
        cx, cy = p["center"]
        dx, dy = p["outward"]
        role = "in" if name == port_in else ("through" if name == port_through else "other")
        label = p.get("label") or ("in" if role == "in" else ("through" if role == "through" else name))
        out.append({"name": str(name), "role": role, "label": str(label),
                        "at": [round(float(cx) + offset[0] - origin[0], 4),
                            round(float(cy) + offset[1] - origin[1], 4)],
                        "out": [round(float(dx)), round(float(dy))],
                        "width": float(p.get("width", 0.5))})
    return out


def _sim_ports(sim) -> dict[str, dict[str, Any]]:
    """The simulation's own in-plane ports as the port dict ``export_scene``
    draws, in the user's frame: each port's centre and guide width as
    declared, and the side of its plane that faces the wall as the simulation
    resolved it. A port a symmetry fold dropped (it reads through its mirror
    image) is drawn too, facing its image's wall, mirrored when the fold plane
    is normal to the port. A port normal to z is not drawn on a plan view."""
    from ..components.authoring import Port
    from ..components.declarative import infer_out_direction

    frame = geom.frame_origin(sim)
    # the resolved side of every port the simulation kept, by name
    record = getattr(sim, "_declarative", None)
    kept = {p.name: p for p in (record.ports if record is not None else ())}
    stored = {p.name: p for p in getattr(sim, "ports", ()) or ()}
    sides = {n: (p.out_direction or infer_out_direction(sim, p)) for n, p in {**stored, **kept}.items()}
    inputs = getattr(sim, "_inputs", None) or {}
    declared = [p if isinstance(p, Port) else Port(**p) for p in (inputs.get("ports") or ())]
    source = inputs.get("source")
    if isinstance(source, Port) and source.name not in {p.name for p in declared}:
        declared.append(source)
    if not declared:                                 # a simulation read back: its stored ports
        declared = [dataclasses.replace(p, center_um=tuple(c + o for c, o in zip(p.center_um, frame)))
                    for p in stored.values()]
    fold = getattr(sim, "_fold", None)
    by_name = {p.name: p for p in declared}
    out: dict[str, dict[str, Any]] = {}
    for p in declared:
        if p.axis == "z":
            continue
        a = "xyz".index(p.axis)
        side = sides.get(p.name)
        if side is None and fold is not None and p.name in fold.mirrored_ports:
            image = fold.mirrored_ports[p.name]
            side = sides.get(image)
            plane = fold.planes.get(a)
            if side is not None and plane is not None and image in by_name and (
                    (float(p.center_um[a]) - plane) * (float(by_name[image].center_um[a]) - plane) < 0.0):
                side = "-" if side == "+" else "+"
        if side is None:
            continue
        s = 1 if side == "+" else -1
        out[p.name] = {"center": (float(p.center_um[0]), float(p.center_um[1])),
                       "outward": (s, 0) if p.axis == "x" else (0, s), "width": float(p.width_um)}
    return out


def _sources(sim, origin, max_dipoles: int = 4) -> list[dict[str, Any]]:
    """Point dipoles worth a glyph. A mode launch is thousands of dipoles on a
    plane, a source the figure never draws, so more than ``max_dipoles`` of
    them export nothing."""
    dipoles = [s for s in sim.sources if getattr(s, "type", None) == "point_dipole"]
    if len(dipoles) > max_dipoles:
        return []
    out = []
    for s in dipoles:
        if True:
            x, y, z = s.center_um
            out.append({"kind": "dipole", "at": [round(x - origin[0], 4), round(y - origin[1], 4),
                                                 round(z - origin[2], 4)],
                            "polarization": str(getattr(s, "polarization", ""))})
    return out


def _luts() -> dict[str, list[list[int]]]:
    import matplotlib as mpl

    return {name: [[round(255 * c) for c in mpl.colormaps[name](i / 255.0)[:3]]
                   for i in range(256)] for name in _LUTS}


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #

def export_scene(
    sim,
    data,
    *,
    plane: str | Sequence[str],
    freq_hz: float | None = None,
    ports: Mapping[str, Mapping[str, Any]] | None = None,
    port_in: str | None = None,
    port_through: str | None = None,
    ports_offset_um: tuple[float, float] = (0.0, 0.0),
    label: str | None = None,
    caption: str | None = None,
    mode: str = "TE0",
    periodic_extent_um: float = 3.0,
    max_samples: int = 256,
    simplify_um: float = 0.002,
    propagate_um: float | None = None,
    propagate_extent_um: float | None = None,
    phase_component: str | Sequence[str] | None = None,
    front_spacing: int = 1,
    structures: Sequence[int] | None = None,
) -> Scene:
    """Build the featured-figure scene of ``sim`` from its run ``data``.

    ``plane`` names the field-DFT monitor (a :class:`~photonhub.ProfileMonitor`
    with one zero-size axis) to draw; ``freq_hz`` picks the frequency when the
    monitor records several. Naming ``port_in`` or ``port_through`` without
    ``ports`` draws the simulation's own ports (``sim.ports``), each facing the
    wall the simulation infers for it. ``ports`` is otherwise a port dict, ``name ->
    {"center": (x, y), "outward": (dx, dy), "width": w}`` in the notebook's
    own coordinates: the user's frame of a simulation fitted with ``domain=``
    (its ``origin_um`` is taken off here), shifted by ``ports_offset_um`` when
    the layout was placed at an offset of its own. ``port_in`` is
    labelled "in" and its arrow points inward; ``port_through`` is labelled
    "through"; every other port carries its own name unless it declares a
    ``label``. ``caption`` defaults to "|E|, λ = … nm" plus ", TE₀ in" when
    ports are given. Quasi-2D runs (periodic transverse axes thinner than
    ``periodic_extent_um``) are widened to that extent, a structure that
    does not span the period (a pillar in its unit cell) repeated at it; a
    wider periodic domain is shown as it is. ``propagate_um`` draws the recorded plane's field
    reconstructed that far downstream of it instead (its plane-wave spectrum
    propagated through the homogeneous medium it sits in, a lens's focal
    spot floating above the device); ``propagate_extent_um`` crops that
    plane to a square of this side, centred on the shown region.
``phase_component`` names the recorded component whose phase animates the
figure; by default it is the E component carrying the most power over the
plane. A guide that turns in the plane (a bend, a ring) rotates its field
with it, so no single Cartesian E component follows the light around the
turn: record ``Hz`` on the plane and pass ``phase_component="Hz"``, the one
component a rotation about the plane's normal leaves unchanged. A device that
turns the polarization hands the light from one component to another, so that
no single one is lit from end to end: pass the pair, ``phase_component=("Ey",
"Ez")``, and the figure animates the field along the major axis of the
ellipse the two trace, which is the whole transverse field of a guided mode.
``front_spacing`` draws the wave fronts that many times further apart along
the plane's long axis, and the caption says so. It is for a device some
hundred wavelengths long, whose fronts are otherwise a pixel or two apart:
the envelope, the mode's lobes and the beating between modes are untouched,
only the carrier common to a whole line of the plane is slowed. It follows the
wave that carries the power, so it suits a travelling wave, not a cavity. The still
figure is |E| either way.
``structures`` lists the indices of ``sim.structures`` to draw, for a figure
that shows the guide alone and not a substrate several microns of oxide below
it; by default every structure is drawn. Air carve-outs carve and claddings
are left out either way, and the scene's height and a plan view's placement
follow the bodies that are drawn.

    ``plane`` may also name a group recorded with
    :meth:`~photonhub.ProfileMonitor.sections`: the monitors ``{plane}_0``,
    ``{plane}_1``, ... are drawn together, each a cross-section on its own
    brightness scale, each sampled at ``max_samples / sqrt(n)`` per axis so
    the scene keeps its size budget. A list of monitor names draws those
    planes together the same way: one plan view inside each layer of a
    stacked device paints the field on every layer's top face. The viewer draws a device much longer
    than it is wide with its long axis compressed and says so in the caption.

    Returns a :class:`Scene`; ``Scene.write(path)`` stores the JSON the viewer
    loads."""
    names = _plane_names(sim, data, plane)
    if isinstance(front_spacing, bool) or int(front_spacing) != front_spacing or front_spacing < 1:
        raise ValueError(f"front_spacing is a whole number of at least 1, not {front_spacing!r}")
    front_spacing = int(front_spacing)
    if len(names) > 1 and propagate_um is not None:
        raise ValueError(f"propagate_um draws one plane; {plane!r} names a group of {len(names)}")
    da = _corner_frame(data, names[0], sim)
    dl3 = _dl_um(sim, da)
    lo, hi = _interior(sim, dl3, periodic_extent_um)
    plane_da, lo_p, hi_p = da, lo, hi
    if propagate_um is not None:
        plane_da = _propagated_plane(sim, data, plane, float(propagate_um))
        if propagate_extent_um is not None:
            half = float(propagate_extent_um) / 2.0
            lo_p = [max(a, (a + b) / 2.0 - half) for a, b in zip(lo, hi)]
            hi_p = [min(b, (a + b) / 2.0 + half) for a, b in zip(lo, hi)]

    # Origin: the shown region's centre in x, y; the drawn bodies' mid-height in
    # z (or the plane's own height / the domain centre without structures). A
    # carve-out is not a body and neither is a cladding: neither sets the origin
    # nor holds a plane.
    only = None if structures is None else {int(i) for i in structures}
    if only is not None and not all(0 <= i < len(sim.structures) for i in only):
        raise ValueError(f"structures lists indices into sim.structures (0 to {len(sim.structures) - 1}), "
                         f"not {sorted(only)}")
    bodies = _structures(sim, dl3, lo, hi, periodic_extent_um, simplify_um, only=only)
    zs = [(z0, z1) for z0, z1, _, _ in bodies]
    if zs:
        z_mid = 0.5 * (min(z for z, _ in zs) + max(z for _, z in zs))
    else:
        z_mid = float(sim.size_um[2]) / 2.0
    origin = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, z_mid)

    body_docs = _serialize_bodies(bodies, origin)
    dipoles = [tuple(float(c) for c in s.center_um) for s in sim.sources
               if getattr(s, "type", None) == "point_dipole"]
    per_plane = max_samples if len(names) == 1 else max(64, int(max_samples / math.sqrt(len(names))))
    plane_docs = []
    for i, name in enumerate(names):
        pda = plane_da if i == 0 else _corner_frame(data, name, sim)
        doc_i, f_used = _plane(sim, pda, freq_hz=freq_hz, dl3=dl3, lo=lo_p, hi=hi_p,
                               origin=origin, structures_z=zs, max_samples=per_plane,
                               periodic_extent_um=periodic_extent_um,
                               exclude_um=dipoles if len(dipoles) <= 4 else (),
                               phase_component=phase_component, front_spacing=front_spacing)
        plane_docs.append(doc_i)
    plane_doc = plane_docs[0]
    if not zs and plane_doc["axis"] != "z":
        # Without structures the z origin is the domain centre; keep the
        # plane's own height as the reference for a horizontal plane.
        pass
    frame = geom.frame_origin(sim)
    if ports is None and (port_in is not None or port_through is not None):
        ports = _sim_ports(sim)
        if not ports:
            raise ValueError("export_scene: port_in/port_through name a port, but the simulation has no "
                             "in-plane ports (a port normal to z is not drawn); pass ports=")
        ports_offset_um = (0.0, 0.0)                     # the simulation's ports are in its own frame
    port_docs = _ports(ports, port_in, port_through,
                       (ports_offset_um[0] - frame[0], ports_offset_um[1] - frame[1]), origin)
    wavelength_um = C0_M_PER_S / f_used * 1e6 if f_used and math.isfinite(f_used) else None
    if caption is None:
        caption = f"|E|, λ = {wavelength_um * 1e3:.0f} nm" if wavelength_um else "|E|"
        if port_docs:
            caption += f", {mode[:2]}<sub>{mode[2:] or '0'}</sub> in"
        if len(plane_docs) > 1:
            stacked = all(d["axis"] == "z" for d in plane_docs)     # one plan view per layer of a stack
            caption += f", each {'layer' if stacked else 'cross-section'} on its own scale"
    if front_spacing > 1:       # on a caption of the caller's too: the figure no longer shows the true wavelength
        caption += f", wave fronts drawn {front_spacing}× further apart"

    doc = {
        "version": 1,
        "label": label or "",
        "caption": caption,
        "mode": mode,
        "wavelength_um": round(wavelength_um, 6) if wavelength_um else None,
        "interior": [round(lo[0] - origin[0], 4), round(hi[0] - origin[0], 4),
                  round(lo[1] - origin[1], 4), round(hi[1] - origin[1], 4)],
        "z_range": [round(lo[2] - origin[2], 4), round(hi[2] - origin[2], 4)],
        "structures": body_docs,
        "ports": port_docs,
        "sources": _sources(sim, origin),
        "planes": plane_docs,
        "luts": _luts(),
    }
    return Scene(doc)


__all__ = ["Scene", "export_scene"]
