"""Translation between the user's frame and the wire frame (design spec §4.4).

A simulation fitted around its device is built in the user's coordinates; the
wire document is corner-anchored. :func:`translate` moves every positional
field of a value by a 3-vector, and is the one place the fit, the results and
the plotters call, so the inventory of positional fields lives here:

- ``Box``, ``Sphere``, ``Cylinder``: ``center_um``;
- ``Polygon``: ``vertices_um`` (natural (u, v) order of its transverse axes)
  and ``slab_bounds_um`` (along ``axis``);
- ``Structure``: its geometry (a ``PermittivityArray`` medium is sampled over
  the geometry and carries no position);
- ``PointDipole``, ``TimeMonitor``, ``ProfileMonitor``, ``TFSFBox``:
  ``center_um``; a ``ProfileMonitor``'s ``mode_port`` centre (natural order);
- ``PlaneWave``, ``ModeSource``: ``position_um`` along ``axis``; a
  ``ModeSource``'s ``mode_solve`` centre (natural order);
- ``PowerMonitor``: ``position_um`` along ``axis`` and its window centre in
  the plane's CYCLIC order;
- ``Port``, ``GaussianBeam``: ``center_um``;
- ``SnapshotMonitor`` (whole domain), ``GradedMesh`` (corner-anchored) and
  ``DesignRegion`` (indexed in cells) do not move.

A ``RunResult`` hands out arrays with ``origin_um`` added to their spatial
coordinates, so the user reads and plots in the frame they drew in. The
analysis readers that place a monitor's window on a recorded plane work in the
wire frame and take the origin off again through :func:`wire_array`.

:func:`mirror` reflects the same inventory through a plane, and
:func:`mirror_image_missing` checks a structure set for the mirror symmetry
the fold of a declared ``symmetry`` plane relies on (design spec §4.5).

The wire document carries none of that client state: not ``origin_um``, not
the fold, not the declared ports and wavelengths. :func:`client_state` records
it beside the wire document a run writes (``client.json`` next to
``sim.json``, a dataset in an HDF5 bundle, a file in the cloud client's cache),
bound to that document (:func:`wire_digest`), and :func:`restore_client_state` puts it back
on a :class:`~photonhub.Simulation` reloaded from the wire, so a reloaded or
resumed result reads in the same frame, unfolds the same planes and reads the
same ports as the result the run returned.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
import threading
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from .authoring import GaussianBeam, Port
from .monitors import PowerMonitor, ProfileMonitor, SnapshotMonitor, TimeMonitor
from .sources import ModeSource, PlaneWave, PointDipole, TfsfBox
from .structures import Box, Cylinder, Medium, Polygon, Sphere, Structure
from .._compat import caller_stacklevel

Vec3 = Tuple[float, float, float]
_AXES = "xyz"


def _natural(axis: str) -> Tuple[int, int]:
    a = _AXES.index(axis)
    return tuple(i for i in range(3) if i != a)  # type: ignore[return-value]


def _cyclic(axis: str) -> Tuple[int, int]:
    a = _AXES.index(axis)
    return (a + 1) % 3, (a + 2) % 3


def _add3(c, d) -> Vec3:
    return (float(c[0]) + d[0], float(c[1]) + d[1], float(c[2]) + d[2])


def translate(value: Any, delta: Sequence[float]) -> Any:
    """``value`` moved by ``delta`` (microns, x, y, z). Tuples and lists are
    mapped element-wise; ``None`` and values without a position pass through."""
    d = (float(delta[0]), float(delta[1]), float(delta[2]))
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        out = [translate(v, d) for v in value]
        return type(value)(out) if isinstance(value, tuple) else out
    if isinstance(value, (Box, Sphere, Cylinder, PointDipole, TimeMonitor, TfsfBox)):
        return value.model_copy(update={"center_um": _add3(value.center_um, d)})
    if isinstance(value, Polygon):
        u, v = _natural(value.axis)
        a = _AXES.index(value.axis)
        verts = tuple((float(p[0]) + d[u], float(p[1]) + d[v]) for p in value.vertices_um)
        lo, hi = value.slab_bounds_um
        return value.model_copy(update={"vertices_um": verts,
                                        "slab_bounds_um": (float(lo) + d[a], float(hi) + d[a])})
    if isinstance(value, Structure):
        return value.model_copy(update={"geometry": translate(value.geometry, d)})
    if isinstance(value, ProfileMonitor):
        update: dict = {"center_um": _add3(value.center_um, d)}
        if value.mode_port is not None:
            normal = [i for i, s in enumerate(value.size_um) if float(s) == 0.0]
            if len(normal) == 1:
                u, v = _natural(_AXES[normal[0]])
                mp = value.mode_port
                update["mode_port"] = mp.model_copy(update={
                    "center_um": (float(mp.center_um[0]) + d[u], float(mp.center_um[1]) + d[v])})
        return value.model_copy(update=update)
    if isinstance(value, PowerMonitor):
        a = _AXES.index(value.axis)
        update = {"position_um": float(value.position_um) + d[a]}
        if value.center_um is not None:
            u, v = _cyclic(value.axis)
            update["center_um"] = (float(value.center_um[0]) + d[u], float(value.center_um[1]) + d[v])
        return value.model_copy(update=update)
    if isinstance(value, PlaneWave):
        a = _AXES.index(value.axis)
        return value.model_copy(update={"position_um": float(value.position_um) + d[a]})
    if isinstance(value, ModeSource):
        a = _AXES.index(value.axis)
        update = {"position_um": float(value.position_um) + d[a]}
        if value.mode_solve is not None:
            u, v = _natural(value.axis)
            ms = value.mode_solve
            update["mode_solve"] = ms.model_copy(update={
                "center_um": (float(ms.center_um[0]) + d[u], float(ms.center_um[1]) + d[v])})
        return value.model_copy(update=update)
    if isinstance(value, Port):
        return dataclasses.replace(value, center_um=_add3(value.center_um, d))
    if isinstance(value, GaussianBeam):
        a = _AXES.index(value.axis)
        return dataclasses.replace(
            value, position_um=float(value.position_um) + d[a],
            center_um=None if value.center_um is None else _add3(value.center_um, d))
    if isinstance(value, SnapshotMonitor):
        return value
    return value


def frame_origin(sim) -> Vec3:
    """The user-frame position of the wire's corner (``Simulation.origin_um``),
    ``(0, 0, 0)`` for a hand-built or ingested scene or for no simulation."""
    o = getattr(sim, "origin_um", None)
    return (float(o[0]), float(o[1]), float(o[2])) if o is not None else (0.0, 0.0, 0.0)


def to_wire_frame(da, sim):
    """A result array's spatial coordinates moved from the user's frame back
    into the wire's corner frame: ``origin_um`` taken off ``x``, ``y``, ``z``."""
    origin = frame_origin(sim)
    if not any(origin):
        return da
    shift = {ax: da.coords[ax] - origin[i] for i, ax in enumerate(_AXES) if ax in da.coords}
    return da.assign_coords(shift)


def to_user_frame(da, sim):
    """The inverse of :func:`to_wire_frame`: ``origin_um`` added to ``x``,
    ``y``, ``z``."""
    origin = frame_origin(sim)
    if not any(origin):
        return da
    shift = {ax: da.coords[ax] + origin[i] for i, ax in enumerate(_AXES) if ax in da.coords}
    return da.assign_coords(shift)


def wire_array(data, name: str):
    """The recorded array ``name`` in the wire frame, as the engine wrote it.
    A ``RunResult`` hands it over through ``wire()`` (its ``data[name]`` is
    the user's view: user-frame coordinates, a folded plane unfolded); a plain
    mapping of arrays (a test's synthetic planes) is in the wire frame
    already."""
    raw = getattr(data, "wire", None)
    if callable(raw):
        return raw(name)
    return to_wire_frame(data[name], getattr(data, "simulation", None))


_FLIP = {"+": "-", "-": "+"}
_REFERENCE_FLIP = {"bottom": "top", "top": "bottom", "middle": "middle"}
_TWO_PI = 2.0 * math.pi


def _reflect3(c, a: int, plane: float) -> Vec3:
    out = [float(c[0]), float(c[1]), float(c[2])]
    out[a] = 2.0 * plane - out[a]
    return (out[0], out[1], out[2])


def mirror(value: Any, axis: str, plane: float) -> Any:
    """``value`` reflected through the plane ``axis = plane`` (microns): the
    inventory :func:`translate` covers, with a sector's angles, a slanted
    sidewall, a port's or a source's direction turned over where the reflection
    turns them. Tuples and lists are mapped element-wise; ``None`` and values
    without a position pass through."""
    a = _AXES.index(axis)
    p = float(plane)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        out = [mirror(v, axis, p) for v in value]
        return type(value)(out) if isinstance(value, tuple) else out
    if isinstance(value, (Box, Sphere, PointDipole, TimeMonitor, TfsfBox)):
        return value.model_copy(update={"center_um": _reflect3(value.center_um, a, p)})
    if isinstance(value, Cylinder):
        update: dict = {"center_um": _reflect3(value.center_um, a, p)}
        sweep = float(value.angle_stop_rad) - float(value.angle_start_rad)
        if value.axis != axis and sweep < _TWO_PI - 1e-9:
            u, _ = _natural(value.axis)
            # angles are atan2(v, u); (u, v) -> (-u, v) sends t to pi - t and
            # (u, v) -> (u, -v) sends t to -t; the sweep runs the other way
            start = (math.pi - float(value.angle_stop_rad)) if a == u else -float(value.angle_stop_rad)
            start = start % _TWO_PI
            update["angle_start_rad"], update["angle_stop_rad"] = start, start + sweep
        return value.model_copy(update=update)
    if isinstance(value, Polygon):
        if value.axis == axis:
            lo, hi = value.slab_bounds_um
            return value.model_copy(update={
                "slab_bounds_um": (2.0 * p - float(hi), 2.0 * p - float(lo)),
                "sidewall_angle": -float(value.sidewall_angle),
                "reference_plane": _REFERENCE_FLIP[value.reference_plane]})
        u, _ = _natural(value.axis)
        i = 0 if a == u else 1
        verts = []
        for q in value.vertices_um:
            q = [float(q[0]), float(q[1])]
            q[i] = 2.0 * p - q[i]
            verts.append((q[0], q[1]))
        return value.model_copy(update={"vertices_um": tuple(reversed(verts))})   # counter-clockwise again
    if isinstance(value, Structure):
        return value.model_copy(update={"geometry": mirror(value.geometry, axis, p)})
    if isinstance(value, ProfileMonitor):
        update = {"center_um": _reflect3(value.center_um, a, p)}
        if value.mode_port is not None:
            normal = [i for i, sz in enumerate(value.size_um) if float(sz) == 0.0]
            if len(normal) == 1 and a != normal[0]:
                nat = _natural(_AXES[normal[0]])
                i = nat.index(a)
                c = [float(value.mode_port.center_um[0]), float(value.mode_port.center_um[1])]
                c[i] = 2.0 * p - c[i]
                update["mode_port"] = value.mode_port.model_copy(update={"center_um": (c[0], c[1])})
        return value.model_copy(update=update)
    if isinstance(value, PowerMonitor):
        if value.axis == axis:
            return value.model_copy(update={"position_um": 2.0 * p - float(value.position_um)})
        if value.center_um is not None:
            cyc = _cyclic(value.axis)
            i = cyc.index(a)
            c = [float(value.center_um[0]), float(value.center_um[1])]
            c[i] = 2.0 * p - c[i]
            return value.model_copy(update={"center_um": (c[0], c[1])})
        return value
    if isinstance(value, PlaneWave):
        if value.axis != axis:
            return value
        update = {"position_um": 2.0 * p - float(value.position_um)}
        if getattr(value, "direction", None) in _FLIP:
            update["direction"] = _FLIP[value.direction]
        return value.model_copy(update=update)
    if isinstance(value, ModeSource):
        update = {}
        if value.axis == axis:
            update["position_um"] = 2.0 * p - float(value.position_um)
            if getattr(value, "direction", None) in _FLIP:
                update["direction"] = _FLIP[value.direction]
        elif value.mode_solve is not None:
            nat = _natural(value.axis)
            i = nat.index(a)
            c = [float(value.mode_solve.center_um[0]), float(value.mode_solve.center_um[1])]
            c[i] = 2.0 * p - c[i]
            update["mode_solve"] = value.mode_solve.model_copy(update={"center_um": (c[0], c[1])})
        return value.model_copy(update=update) if update else value
    if isinstance(value, Port):
        out = value.out_direction
        if value.axis == axis and out in _FLIP:
            out = _FLIP[out]
        return dataclasses.replace(value, center_um=_reflect3(value.center_um, a, p), out_direction=out)
    if isinstance(value, GaussianBeam):
        changes: dict = {}
        if value.axis == axis:
            changes["position_um"] = 2.0 * p - float(value.position_um)
            if value.direction in _FLIP:
                changes["direction"] = _FLIP[value.direction]
        if value.center_um is not None:
            changes["center_um"] = _reflect3(value.center_um, a, p)
        return dataclasses.replace(value, **changes) if changes else value
    return value


def _close(x, y, tol: float) -> bool:
    return abs(float(x) - float(y)) <= tol


def _same_points(a, b, tol: float) -> bool:
    """Two vertex lists describe the same point set within ``tol``: aligned by
    a coarse sort first, then paired one to one when the sort disagrees."""
    if len(a) != len(b):
        return False
    key = lambda q: (round(float(q[0]), 6), round(float(q[1]), 6))
    sa, sb = sorted(a, key=key), sorted(b, key=key)
    if all(_close(x[0], y[0], tol) and _close(x[1], y[1], tol) for x, y in zip(sa, sb)):
        return True
    free = list(b)
    for q in a:
        for i, r in enumerate(free):
            if _close(q[0], r[0], tol) and _close(q[1], r[1], tol):
                del free[i]
                break
        else:
            return False
    return True


def _same_geometry(g, h, tol: float) -> bool:
    if type(g) is not type(h):
        return False
    if isinstance(g, (Box, Sphere)):
        if not all(_close(x, y, tol) for x, y in zip(g.center_um, h.center_um)):
            return False
        return (all(_close(x, y, tol) for x, y in zip(g.size_um, h.size_um)) if isinstance(g, Box)
                else _close(g.radius_um, h.radius_um, tol))
    if isinstance(g, Cylinder):
        if g.axis != h.axis or not all(_close(x, y, tol) for x, y in zip(g.center_um, h.center_um)):
            return False
        if not (_close(g.radius_um, h.radius_um, tol) and _close(g.inner_radius_um, h.inner_radius_um, tol)
                and _close(g.length_um, h.length_um, tol)):
            return False
        sg, sh = g.angle_stop_rad - g.angle_start_rad, h.angle_stop_rad - h.angle_start_rad
        if not _close(sg, sh, tol):
            return False
        if sg >= _TWO_PI - 1e-9:
            return True                                    # a full ring: the start angle is immaterial
        d = (float(g.angle_start_rad) - float(h.angle_start_rad)) % _TWO_PI
        return min(d, _TWO_PI - d) <= tol
    if isinstance(g, Polygon):
        return (g.axis == h.axis and g.reference_plane == h.reference_plane
                and _close(g.sidewall_angle, h.sidewall_angle, tol)
                and all(_close(x, y, tol) for x, y in zip(g.slab_bounds_um, h.slab_bounds_um))
                and _same_points(g.vertices_um, h.vertices_um, tol))
    return g == h


def mirror_image_missing(structures: Sequence[Structure], axis: str, plane: float,
                         tol: float = 1e-9) -> Optional[Structure]:
    """The first structure whose mirror image through ``axis = plane`` is not
    in the set (same medium, geometry equal within ``tol`` microns; polygon
    vertices as point sets), or ``None`` when the set is mirror-symmetric. A
    structure symmetric about the plane is its own image. A structure whose
    medium is a permittivity array cannot be checked and raises."""
    structures = list(structures)
    for st in structures:
        if getattr(st.medium, "permittivity_data", None) is not None:
            raise ValueError(
                f"structure {st.name or structures.index(st)}: a permittivity array cannot be checked for "
                f"mirror symmetry about {axis} = {plane:.6g} um; build the half domain by hand (size_um)")
    for st in structures:
        image = mirror(st, axis, plane)
        if not any(other.medium == st.medium and _same_geometry(image.geometry, other.geometry, tol)
                   for other in structures):
            return st
    return None


# --------------------------------------------------------------------------- #
# Client state beside the wire document
# --------------------------------------------------------------------------- #

#: The file a runner writes beside ``sim.json`` when the simulation carries
#: client state the wire document does not.
CLIENT_STATE_FILE = "client.json"
_CLIENT_STATE_FORMAT = "photonhub-client-state-1"
_MEDIUM_KEY = "__medium__"


def _json_value(value):
    if isinstance(value, Medium):
        return {_MEDIUM_KEY: value.model_dump(mode="json")}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _python_value(value):
    if isinstance(value, dict) and set(value) == {_MEDIUM_KEY}:
        return Medium.model_validate(value[_MEDIUM_KEY])
    if isinstance(value, list):
        return tuple(_python_value(v) for v in value)
    return value


def _dataclass_json(value) -> dict:
    return {f.name: _json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}


def _dataclass_from_json(cls, data):
    if not isinstance(data, dict):
        raise ValueError(f"a {cls.__name__} record must be an object")
    return cls(**{k: _python_value(v) for k, v in data.items()})


def _source_json(source):
    if source is None:
        return None
    if isinstance(source, str):
        return {"port": source}
    if isinstance(source, Port):
        return {"port_object": _dataclass_json(source)}
    if isinstance(source, GaussianBeam):
        return {"beam": _dataclass_json(source)}
    raise ValueError(f"source of type {type(source).__name__} has no record")


def _source_from_json(data):
    if data is None:
        return None
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError("the source record must name one kind")
    (kind, value), = data.items()
    if kind == "port" and isinstance(value, str):
        return value
    if kind == "port_object":
        return _dataclass_from_json(Port, value)
    if kind == "beam":
        return _dataclass_from_json(GaussianBeam, value)
    raise ValueError(f"unknown source kind {kind!r}")


def wire_digest(sim) -> str:
    """SHA-256 of ``sim``'s canonical wire document
    (``to_wire_json(indent=0)``): what a client-state record is bound to.
    Canonical, so the binding holds whatever bytes the document was stored as
    (text-mode line endings, a service's own serialization) and fails for any
    other simulation. A record written by an earlier release keeps binding
    only while this serialization of an old document does not change, which
    the schema governance already requires of golden specs; a checked-in
    record (``photonhub/tests/data/client_frame``) pins it."""
    return hashlib.sha256(sim.to_wire_json(indent=0).encode("utf-8")).hexdigest()


def client_state(sim, *, wire: Optional[dict] = None) -> Optional[dict]:
    """The client-only state of ``sim`` as a JSON-ready record, bound to its
    wire document (:func:`wire_digest`); None when there is none to keep (a
    hand-built scene: corner frame, no fold, no declared ports or
    wavelengths).

    The record holds ``origin_um``, the symmetry fold, and for a declarative
    simulation its readout wavelengths, its resolved ports (in the wire frame,
    as the simulation holds them), the driven port and the source as given.
    It does not hold the solved port modes: :func:`restore_client_state`
    solves them again, on first use, on the reloaded simulation's own grid.
    """
    origin = frame_origin(sim)
    fold = getattr(sim, "_fold", None)
    decl = getattr(sim, "_declarative", None)
    wlens = getattr(sim, "wlens_um", None)
    wlen0 = getattr(sim, "wlen0_um", None)
    if not any(origin) and fold is None and wlens is None and wlen0 is None:
        return None
    if wire is not None:
        # The runner can add a CPU-only default or omit an unsupported key.
        # Bind the client frame to the document actually submitted.
        from .simulation import Simulation

        digest = wire_digest(Simulation.from_wire_json(json.dumps(wire)))
    else:
        digest = wire_digest(sim)
    state: dict = {
        "format": _CLIENT_STATE_FORMAT,
        "wire_sha256": digest,
        "origin_um": list(origin),
        "wlens_um": _json_value(wlens),
        "wlen0_um": None if wlen0 is None else float(wlen0),
        "run_transits": getattr(sim, "_run_transits", None),
    }
    if fold is not None:
        state["fold"] = {
            "planes": {str(int(a)): float(p) for a, p in fold.planes.items()},
            "mirrored_ports": dict(fold.mirrored_ports),
            "unfolded_monitors": list(fold.unfolded_monitors),
        }
    if wlens is not None and decl is not None:
        state["declared"] = {
            "ports": [_dataclass_json(p) for p in decl.ports],
            "driven": decl.driven,
            "source": _source_json(getattr(sim, "source", None)),
            # the launch comes first among the sources (declarative.resolve)
            "launch_sources": len(sim.sources) - len(decl.user_sources),
        }
    return state


class _ReloadedPortMonitors(Mapping):
    """``{port name: ModeMonitor}`` for a declarative simulation reloaded from
    its wire document. The names are known at once; the modes are solved on
    first use, by the same :func:`~photonhub.components.declarative.resolve`
    the construction ran, on the reloaded simulation's grid and structures,
    so a transmission read from a reload is the one the run returned."""

    def __init__(self, sim, ports, source, wlens_um, wlen0_um):
        self._sim = sim
        self._ports = tuple(ports)
        self._source = source
        self._wlens_um = wlens_um
        self._wlen0_um = wlen0_um
        self._solved: Optional[dict] = None
        self._lock = threading.Lock()

    # a lock does not pickle or deep-copy; a copy gets its own
    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _monitors(self) -> dict:
        with self._lock:
            if self._solved is None:
                from . import declarative as _decl
                _, _, resolved = _decl.resolve(
                    self._sim, ports=self._ports, source=self._source,
                    wlens_um=self._wlens_um, wlen0_um=self._wlen0_um,
                    sources=(), monitors=())
                self._solved = dict(resolved.port_monitors)
            return self._solved

    def __getitem__(self, name):
        return self._monitors()[name]

    def __iter__(self):
        return iter([p.name for p in self._ports])

    def __len__(self) -> int:
        return len(self._ports)


class _UnrecordedInputs(Mapping):
    """Stands in for the fields as the user gave them on a simulation
    reloaded from a result, which were never stored. ``with_changes``
    rebuilds a fitted simulation from those fields in the user's frame; a
    reload has its origin back but not them, and a copy made in the corner
    frame would place new geometry off by ``origin_um``. Refuse instead."""

    _MESSAGE = (
        "this Simulation was reloaded from a result: the fields as you gave "
        "them (the Domain, the Mesh, your structures and ports in your frame) "
        "are not stored with a run, so with_changes cannot rebuild it. Build "
        "it again from your script and change that, or reopen the result with "
        "RunResult(path, client_state=False) to edit it in the wire's corner "
        "frame")

    def __getitem__(self, name):
        raise ValueError(self._MESSAGE)

    def __iter__(self):
        raise ValueError(self._MESSAGE)

    def keys(self):
        raise ValueError(self._MESSAGE)

    def __len__(self) -> int:
        return 0


def _parse_client_state(sim, state):
    """Validate ``state`` against ``sim``, the simulation loaded from its wire
    document; return what :func:`restore_client_state` sets. Raises
    ValueError, TypeError or KeyError on anything it cannot use."""
    from . import declarative as _decl

    if not isinstance(state, dict):
        raise ValueError("the record is not a JSON object")
    if state.get("format") != _CLIENT_STATE_FORMAT:
        raise ValueError(f"unknown format {state.get('format')!r}")
    if state.get("wire_sha256") != wire_digest(sim):
        raise ValueError("it was written for a different simulation")
    origin = tuple(float(v) for v in state["origin_um"])
    if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
        raise ValueError("origin_um must be three finite numbers")
    fold = None
    if state.get("fold") is not None:
        f = state["fold"]
        fold = _decl.Fold(
            planes={int(a): float(p) for a, p in f["planes"].items()},
            mirrored_ports={str(k): str(v) for k, v in f["mirrored_ports"].items()},
            unfolded_monitors=tuple(str(m) for m in f["unfolded_monitors"]))
    wlens = _python_value(state.get("wlens_um"))
    wlen0 = None if state.get("wlen0_um") is None else float(state["wlen0_um"])
    transits = None if state.get("run_transits") is None else float(state["run_transits"])
    declared = None
    if state.get("declared") is not None:
        d = state["declared"]
        if wlens is None:
            raise ValueError("declared ports need wlens_um")
        ports = tuple(_dataclass_from_json(Port, p) for p in d["ports"])
        driven = d.get("driven")
        if driven is not None and driven not in {p.name for p in ports}:
            raise ValueError(f"the driven port {driven!r} is not among the ports")
        source = _source_from_json(d.get("source"))
        n_launch = int(d["launch_sources"])
        if not 0 <= n_launch <= len(sim.sources):
            raise ValueError("launch_sources does not fit the sources")
        generated = {p.monitor_name for p in ports}
        missing = generated - {getattr(m, "name", None) for m in sim.monitors}
        if missing:
            raise ValueError(f"the port monitors {sorted(missing)} are not in the document")
        freqs, _, pulse = _decl.band(wlens, wlen0)
        resolved = _decl.Resolved(
            freqs_hz=freqs, pulse=pulse, ports=ports,
            port_monitors=_ReloadedPortMonitors(sim, ports, driven, wlens, wlen0),
            driven=driven, user_sources=tuple(sim.sources[n_launch:]),
            user_monitors=tuple(m for m in sim.monitors
                                if getattr(m, "name", None) not in generated))
        declared = (ports, source, resolved)
    return origin, fold, (wlens, wlen0, transits), declared


def restore_client_state(sim, state, *, where: str) -> bool:
    """Put the client state ``state`` (read from ``where``) back on ``sim``, a
    :class:`~photonhub.Simulation` just loaded from its wire document.
    Returns True when it was restored.

    A record that does not belong to that simulation (another run's
    leftover, a copy beside a different ``sim.json``), of another format, or
    unreadable restores nothing: this warns and returns False, and the result
    then reads as the wire document alone describes it (corner-frame
    coordinates, a plane across a symmetry plane left as recorded, no
    declared ports). A restored simulation refuses ``with_changes``: the
    fields as the user gave them are not part of the record.
    """
    try:
        origin, fold, (wlens, wlen0, transits), declared = _parse_client_state(sim, state)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        _warn_not_restored(where, str(exc))
        return False
    object.__setattr__(sim, "origin_um", origin)
    object.__setattr__(sim, "wlens_um", wlens)
    object.__setattr__(sim, "wlen0_um", wlen0)
    sim._fold = fold
    sim._run_transits = transits
    sim._inputs = _UnrecordedInputs()
    if declared is not None:
        ports, source, resolved = declared
        object.__setattr__(sim, "ports", ports)
        object.__setattr__(sim, "source", source)
        sim._declarative = resolved
    return True


def loads_client_state(text: Optional[str], *, where: str):
    """The record in ``text`` (None for none). Unreadable JSON warns, naming
    ``where``, and gives None."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        _warn_not_restored(where, f"unreadable JSON: {exc}")
        return None


def read_client_state(path):
    """The record in the file at ``path``, or None when there is no such file
    (a symlink counts as none). An unreadable file warns and gives None."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        _warn_not_restored(str(path), str(exc))
        return None
    return loads_client_state(text, where=str(path))


def _warn_not_restored(where: str, why: str) -> None:
    warnings.warn(
        f"could not restore the client state in {where} ({why}); the result "
        "reads in the wire's corner frame, a plane across a symmetry plane "
        "stays as recorded and declared ports are not available",
        UserWarning, stacklevel=caller_stacklevel())


def write_json_atomic(path, record) -> Path:
    """Write ``record`` as JSON to ``path`` through a temporary file in the
    same directory and an atomic rename, so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_client_state(sim, spec_path, *, wire: Optional[dict] = None) -> Optional[Path]:
    """Write ``sim``'s client state beside the wire document at ``spec_path``
    (:data:`CLIENT_STATE_FILE` in the same directory), or remove a record
    left there by an earlier run when ``sim`` has none. Best effort: a failure warns, since it costs a later reload its
    frame and ports, never the run."""
    spec_path = Path(spec_path)
    out = spec_path.with_name(CLIENT_STATE_FILE)
    try:
        state = client_state(sim, wire=wire)
        if state is None:
            out.unlink(missing_ok=True)
            return None
        return write_json_atomic(out, state)
    except (OSError, TypeError, ValueError) as exc:
        warnings.warn(
            f"could not write the client state beside {spec_path} ({exc}); a "
            "reload of this result will read in the wire's corner frame, with "
            "no symmetry unfolding and no declared ports",
            UserWarning, stacklevel=caller_stacklevel())
        return None
