"""``export_scene()`` — the featured-figure scene of a run, for the standard
3D "journal schematic" viewer (docs/superpowers/specs/2026-09-04-featured-
figure-design.md).

The exporter turns a :class:`~photonhub.Simulation` plus its run data into
one JSON document — the contract between Python and the viewer:

- ``structures``: the top-view outline rings of every structure (union of the
  structures sharing a z-extent, clipped to the non-PML interior), with their
  ``z0``/``z1``;
- ``ports``: positions, outward directions, roles and labels (from the
  notebook's port dict — the same one the GDS sidecar fills);
- ``planes``: one recorded field plane as |E| plus the dominant component's
  phasor, normalised to the 99.7th percentile, downsampled to at most
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

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import _geometry as geom

C0_M_PER_S = 2.99792458e8
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
        return self.doc

    def to_json(self) -> str:
        return json.dumps(self.doc, separators=(",", ":"), allow_nan=False)

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @property
    def size_bytes(self) -> int:
        return len(self.to_json().encode("utf-8"))


# --------------------------------------------------------------------------- #
# Grid / boundary helpers.
# --------------------------------------------------------------------------- #

def _dl_um(sim, plane_da=None) -> tuple[float, float, float]:
    """Cell size per axis: the uniform spacing, or — on a graded grid — the
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
    extruded along z — the figure's up axis."""
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
        sweep = g.angle_stop - g.angle_start
        if sweep < 2 * math.pi - 1e-9:
            wedge = SPolygon([(cx, cy)] + [
                (cx + 2 * g.radius_um * math.cos(g.angle_start + sweep * k / 48),
                 cy + 2 * g.radius_um * math.sin(g.angle_start + sweep * k / 48))
                for k in range(49)])
            outer = outer.intersection(wedge)
        return outer, cz - g.length_um / 2, cz + g.length_um / 2
    if kind == "sphere":
        cx, cy, cz = g.center_um
        return SPolygon(_circle(cx, cy, g.radius_um)), cz - g.radius_um, cz + g.radius_um
    raise ValueError(f"export_scene does not know how to draw geometry {kind!r}")


def _extend_periodic(poly, sim, dl3, periodic_extent_um: float):
    """Widen a footprint that spans the domain along a periodic axis to the
    displayed periodic extent (the quasi-2D column becomes a plate)."""
    from shapely import affinity

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


def _structures(sim, dl3, lo, hi, periodic_extent_um: float, origin, simplify_um: float):
    shapely = _require_shapely()
    from shapely.ops import unary_union

    clip = shapely.box(lo[0], lo[1], hi[0], hi[1])
    sym = _symmetry(sim)
    by_extent: dict[tuple[float, float], list] = {}
    eps_by_extent: dict[tuple[float, float], float] = {}
    for s in sim.structures:
        if _is_air(s.medium):
            continue                      # a carve-out of the background, not a body
        poly, z0, z1 = _footprint(s)
        poly = _extend_periodic(poly, sim, dl3, periodic_extent_um)
        # A half-domain build may hold only the half of a structure on the
        # simulated side; its mirror image completes it (a structure drawn
        # whole across the face is unchanged by the union).
        from shapely import affinity
        if sym[0]:
            poly = unary_union([poly, affinity.scale(poly, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0, 0.0))])
        if sym[1]:
            poly = unary_union([poly, affinity.scale(poly, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0, 0.0))])
        key = (round(z0, 6), round(z1, 6))
        by_extent.setdefault(key, []).append(poly)
        eps_by_extent[key] = float(getattr(s.medium, "permittivity", 1.0))
    out = []
    for (z0, z1), polys in sorted(by_extent.items()):
        merged = unary_union(polys).intersection(clip)
        if merged.is_empty:
            continue
        merged = merged.simplify(simplify_um)
        geoms = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        rings = []
        for g in geoms:
            if g.is_empty or g.geom_type != "Polygon":
                continue
            rings.append([[round(x - origin[0], 4), round(y - origin[1], 4)]
                          for x, y in g.exterior.coords])
        if rings:
            out.append({"rings": rings, "z0": round(z0 - origin[2], 4),
                            "z1": round(z1 - origin[2], 4), "kind": "core",
                            "eps": eps_by_extent[(z0, z1)]})
    return out


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


def _propagated_plane(sim, data, monitor: str, dz_um: float):
    """The recorded plane's complex E reconstructed ``dz_um`` downstream of it
    (:func:`~photonhub.analysis.propagate_plane`), as a field DataArray at
    that height, so it can be drawn like a monitor plane."""
    import xarray as xr

    from ..analysis.propagate import propagate_plane

    da0 = data[monitor]
    spatial = [ax for ax in ("x", "y", "z") if ax in da0.dims]
    axis = [ax for ax in spatial if da0.sizes[ax] == 1][0]
    res = propagate_plane(sim, data, monitor, [abs(dz_um)], direction="+" if dz_um >= 0 else "-")
    u1, u2 = res["axes"]
    position = float(da0.coords[axis].values[0]) + dz_um
    vals = np.stack([res["e1"][0], res["e2"][0], res["en"][0]], axis=1)       # [f, c, n1, n2]
    coords = {"f": ("f", np.asarray(res["freqs_hz"], dtype=float)),
              "component": [f"E{u1}", f"E{u2}", f"E{axis}"],
              axis: (axis, np.asarray([position], dtype=float)),
              u1: (u1, np.asarray(res["coords1_um"], dtype=float)),
              u2: (u2, np.asarray(res["coords2_um"], dtype=float))}
    return xr.DataArray(vals[:, :, None], dims=("f", "component", axis, u1, u2), coords=coords,
                        name=f"{monitor} +{dz_um:g} um")


def _plane(sim, da, *, freq_hz, dl3, lo, hi, origin, structures_z, max_samples: int,
           periodic_extent_um: float, exclude_um=(), exclude_radius_um: float = 0.15):
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
    if np.iscomplexobj(vals):
        amp = np.sqrt((np.abs(vals) ** 2).sum(axis=0))
        dominant = int(np.argmax((np.abs(vals) ** 2).sum(axis=(1, 2))))
        phasor = vals[dominant]
    else:
        amp = np.sqrt((vals.astype(float) ** 2).sum(axis=0))
        dominant = int(np.argmax((vals ** 2).sum(axis=(1, 2))))
        phasor = vals[dominant].astype(complex)
    u = np.asarray(da.coords[h_ax].values, dtype=float)
    v = np.asarray(da.coords[v_ax].values, dtype=float)

    # Crop to the shown region along both in-plane axes.
    iu = "xyz".index(h_ax)
    iv = "xyz".index(v_ax)
    ku = np.where((u >= lo[iu] - 1e-9) & (u <= hi[iu] + 1e-9))[0]
    kv = np.where((v >= lo[iv] - 1e-9) & (v <= hi[iv] + 1e-9))[0]
    u, v = u[ku], v[kv]
    amp, phasor = amp[np.ix_(kv, ku)], phasor[np.ix_(kv, ku)]

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
    periodic_in_plane = any(_column_axis(sim, ax, periodic_extent_um) for ax in (h_ax, v_ax))
    if periodic_in_plane and not keep.all():
        for r in np.where(~keep.all(axis=1))[0]:
            good = keep[r]
            if good.any():
                amp[r, ~good] = amp[r, good].mean()
                phasor[r, ~good] = phasor[r, good].mean()
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
        ph_t = np.concatenate(phs, axis=axis_i)
        keep = np.where((coord_t >= c - periodic_extent_um / 2.0 - 1e-9)
                        & (coord_t <= c + periodic_extent_um / 2.0 + 1e-9))[0]
        if which == "u":
            u, amp, phasor = coord_t[keep], amp_t[:, keep], ph_t[:, keep]
        else:
            v, amp, phasor = coord_t[keep], amp_t[keep, :], ph_t[keep, :]

    # Unfold about a symmetry face (NUMERICS.md §20): |E| is even; the
    # dominant component's phasor carries its parity — a PEC mirror (-1) keeps
    # the component normal to the face and flips the tangential ones, a PMC
    # (+1) the reverse. The recorded samples start at the first cell centre
    # past the face, so the mirrored copy adds no duplicate row.
    sym = _symmetry(sim)
    for which, ax in (("u", h_ax), ("v", v_ax)):
        i = "xyz".index(ax)
        if sym[i] == 0:
            continue
        normal = comps[dominant].endswith(ax)
        sign = 1.0 if normal == (sym[i] == -1) else -1.0
        if which == "u":
            u = np.concatenate([-u[::-1], u])
            amp = np.concatenate([amp[:, ::-1], amp], axis=1)
            phasor = np.concatenate([sign * phasor[:, ::-1], phasor], axis=1)
        else:
            v = np.concatenate([-v[::-1], v])
            amp = np.concatenate([amp[::-1, :], amp], axis=0)
            phasor = np.concatenate([sign * phasor[::-1, :], phasor], axis=0)

    # Downsample by box averaging so the long axis has <= max_samples.
    factor = max(1, math.ceil(max(amp.shape) / max_samples))
    if factor > 1:
        amp = _box_average(amp, factor, factor)
        re = _box_average(phasor.real, factor, factor)
        im = _box_average(phasor.imag, factor, factor)
        u = _box_average(u[None, :], 1, factor)[0]
        v = _box_average(v[:, None], factor, 1)[:, 0]
    else:
        re, im = phasor.real, phasor.imag

    # float64 before rounding: a rounded float32 prints with 17 digits in JSON
    amp = np.asarray(amp, dtype=np.float64) / peak
    re = np.asarray(re, dtype=np.float64) / peak
    im = np.asarray(im, dtype=np.float64) / peak

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
        "component": comps[dominant],
        "amp": np.round(amp, 3).ravel().tolist(),      # 3 decimals: finer than the 256-level LUT,
        "re": np.round(re, 3).ravel().tolist(),        # and it keeps a 256 x 256 plane near 1 MB
        "im": np.round(im, 3).ravel().tolist(),
    }
    if top_z is not None:
        plane["top_z"] = top_z
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


def _sources(sim, origin, max_dipoles: int = 4) -> list[dict[str, Any]]:
    """Point dipoles worth a glyph. A mode launch is thousands of dipoles on a
    plane — a source the figure never draws — so more than ``max_dipoles`` of
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
    plane: str,
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
) -> Scene:
    """Build the featured-figure scene of ``sim`` from its run ``data``.

    ``plane`` names the field-DFT monitor (a :class:`~photonhub.ProfileMonitor`
    with one zero-size axis) to draw; ``freq_hz`` picks the frequency when the
    monitor records several. ``ports`` is the notebook's port dict — ``name ->
    {"center": (x, y), "outward": (dx, dy), "width": w}`` in the layout's
    coordinates, shifted by ``ports_offset_um`` into the simulation's (pass the
    same origin the notebook used to place the layout). ``port_in`` is
    labelled "in" and its arrow points inward; ``port_through`` is labelled
    "through"; every other port carries its own name unless it declares a
    ``label``. ``caption`` defaults to "|E|, λ = … nm" plus ", TE₀ in" when
    ports are given. Quasi-2D runs (periodic transverse axes thinner than
    ``periodic_extent_um``) are widened to that extent; a wider periodic
    domain is shown as it is. ``propagate_um`` draws the recorded plane's field
    reconstructed that far downstream of it instead (its plane-wave spectrum
    propagated through the homogeneous medium it sits in — a lens's focal
    spot floating above the device); ``propagate_extent_um`` crops that
    plane to a square of this side, centred on the shown region.

    Returns a :class:`Scene`; ``Scene.write(path)`` stores the JSON the viewer
    loads."""
    da = data[plane]
    dl3 = _dl_um(sim, da)
    lo, hi = _interior(sim, dl3, periodic_extent_um)
    plane_da, lo_p, hi_p = da, lo, hi
    if propagate_um is not None:
        plane_da = _propagated_plane(sim, data, plane, float(propagate_um))
        if propagate_extent_um is not None:
            half = float(propagate_extent_um) / 2.0
            lo_p = [max(a, (a + b) / 2.0 - half) for a, b in zip(lo, hi)]
            hi_p = [min(b, (a + b) / 2.0 + half) for a, b in zip(lo, hi)]

    # Origin: the shown region's centre in x, y; the structures' mid-height in z
    # (or the plane's own height / the domain centre without structures).
    zs = []
    for s in sim.structures:
        if _is_air(s.medium):
            continue                      # a carve-out is not a body: it neither sets the origin nor holds a plane
        _, z0, z1 = _footprint(s)
        zs.append((z0, z1))
    if zs:
        z_mid = 0.5 * (min(z for z, _ in zs) + max(z for _, z in zs))
    else:
        z_mid = float(sim.size_um[2]) / 2.0
    origin = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, z_mid)

    structures = _structures(sim, dl3, lo, hi, periodic_extent_um, origin, simplify_um)
    dipoles = [tuple(float(c) for c in s.center_um) for s in sim.sources
               if getattr(s, "type", None) == "point_dipole"]
    plane_doc, f_used = _plane(sim, plane_da, freq_hz=freq_hz, dl3=dl3, lo=lo_p, hi=hi_p,
                               origin=origin, structures_z=zs, max_samples=max_samples,
                               periodic_extent_um=periodic_extent_um,
                               exclude_um=dipoles if len(dipoles) <= 4 else ())
    if not zs and plane_doc["axis"] != "z":
        # Without structures the z origin is the domain centre; keep the
        # plane's own height as the reference for a horizontal plane.
        pass
    port_docs = _ports(ports, port_in, port_through, ports_offset_um, origin)
    wavelength_um = C0_M_PER_S / f_used * 1e6 if f_used and math.isfinite(f_used) else None
    if caption is None:
        caption = f"|E|, λ = {wavelength_um * 1e3:.0f} nm" if wavelength_um else "|E|"
        if port_docs:
            caption += f", {mode[:2]}<sub>{mode[2:] or '0'}</sub> in"

    doc = {
        "version": 1,
        "label": label or "",
        "caption": caption,
        "mode": mode,
        "wavelength_um": round(wavelength_um, 6) if wavelength_um else None,
        "interior": [round(lo[0] - origin[0], 4), round(hi[0] - origin[0], 4),
                  round(lo[1] - origin[1], 4), round(hi[1] - origin[1], 4)],
        "z_range": [round(lo[2] - origin[2], 4), round(hi[2] - origin[2], 4)],
        "structures": structures,
        "ports": port_docs,
        "sources": _sources(sim, origin),
        "planes": [plane_doc],
        "luts": _luts(),
    }
    return Scene(doc)


__all__ = ["Scene", "export_scene"]
