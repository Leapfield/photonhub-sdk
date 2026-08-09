"""Data core for the local viz/agent stack — **no plotly, no presentation**.

Pure data operations over a result bundle: load, catalog, slider metadata, plane
slicing, field statistics, spectra, time-series, and cut-plane structure outlines.
The plotly figure builders live in :mod:`photonhub.viz.figures`; the MCP/agent path
imports only this module. This keeps the data↔presentation seam structural.

Design notes (see ``desktop/PLAN.md``):
- ``plot_field`` in ``photonhub.viz`` returns a *matplotlib* Axes; the web path instead
  reads raw arrays from the ``xarray.DataArray`` and builds plotly in ``figures``.
- Geometry is not in the manifest; ``run_local`` writes ``sim.json`` next to
  ``manifest.json``, so the scene + overlays are reconstructed from there.
- The slider *catalog* is manifest-only (no blob load). **Caveat (memmap TODO):**
  ``SimulationData[name]`` currently reads the whole monitor ``.bin`` into RAM and
  caches it (see ``data.py``); large volumetric monitors are not yet sliced lazily.
  PLAN §2/§10 — tracked, not yet implemented.
"""

from __future__ import annotations

import json
import hashlib
import math
import warnings
from dataclasses import asdict
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..data import SimulationData

# complex -> real reducers for the `val` control
_REDUCE = {"real": np.real, "imag": np.imag, "abs": np.abs, "phase": np.angle}
_SPATIAL = ("x", "y", "z")
_CIRC = [2 * math.pi * i / 48 for i in range(49)]  # circle sampling for overlays
_C = 299_792_458.0  # speed of light, for f(Hz) <-> wavelength(nm)
_DIFF_ARRAY_DETAIL_LIMIT = 256
_DIFF_ENTRY_LIMIT = 2000

# sim.json -> (file identity, Simulation), so overlays/scene don't re-parse every
# request and a replacement under the same path is picked up even when its mtime
# is deliberately preserved. Keyed by resolved path string.
_SIM_CACHE: dict[str, tuple[tuple[int, int, int, int, int], Any]] = {}


# --------------------------------------------------------------------------- #
# Loading + catalog
# --------------------------------------------------------------------------- #

def load_result(path: str | Path) -> SimulationData:
    """Open a result bundle (a dir with ``manifest.json``, a ``manifest.json``, or a
    ``.h5``). Cheap to open; a monitor's blob is read on first access (see the
    module memmap caveat)."""
    return SimulationData(path)


def _kind(monitor_manifest: dict) -> str:
    """Classify a monitor for the UI: 'plane' (field over a region → heatmap),
    'timeseries' (field at a point → line vs time), or 'spectrum' (flux/DFT)."""
    mtype = monitor_manifest.get("type", "")
    dims = monitor_manifest.get("dims", [])
    if mtype == "flux":
        return "spectrum"
    shape = monitor_manifest.get("shape", [])
    sizes = {str(d): int(shape[i]) for i, d in enumerate(dims) if i < len(shape)}
    spatial_rank = sum(sizes.get(axis, 0) > 1 for axis in _SPATIAL)
    if mtype == "field_dft" and spatial_rank == 0:
        return "field_spectrum"
    if spatial_rank >= 2:
        return "plane"
    if spatial_rank == 1:
        return "profile"
    if "freq" in dims or "f" in dims:
        return "spectrum"
    return "timeseries"


def session(data: SimulationData, result_id: Optional[str] = None,
            run_id: Optional[str] = None) -> dict:
    """Top-level metadata for the UI/agent: run stats, grid, provenance, monitor
    catalog, abort state, and whether a 3D scene is available."""
    geometry = geometry_status(data)
    payload = {
        "result_id": result_id,
        # Stable durable identity is distinct from the ephemeral result revision.
        # Legacy/external bundles have no ledger run id and remain fully viewable.
        "run_id": run_id,
        "output_dir": str(data.output_dir),
        "monitors": monitor_catalog(data),
        "run": data.manifest.get("run", {}),
        "grid": data.manifest.get("grid", {}),
        "provenance": data.manifest.get("provenance", {}),
        "aborted": data.aborted,
        "abort_reason": data.abort_reason,
        "geometry": geometry,
        "has_scene": geometry["status"] in {"matched", "unverified"},
    }
    # Legacy/external bundles keep their historical response shape.  A port
    # catalog is published only when a checksum-acceptable sim.json actually
    # carries the schema-1.16 authoring recipe.
    if payload["has_scene"]:
        sim = sim_for(data)
        ports = modal_port_summaries_from_sim(sim) if sim is not None else []
        if ports:
            payload["ports"] = ports
    return payload


def geometry_status(data: SimulationData) -> dict:
    """Trust state for geometry overlays reconstructed from sibling ``sim.json``.

    A modified spec beside immutable field binaries is worse than no overlay: it
    draws persuasive but false structure boundaries over the result.  Newer
    manifests carry the engine's exact input SHA-256, so fail closed when it does
    not match.  Legacy bundles without a hash remain viewable but explicitly
    report ``unverified``.
    """
    path = data.output_dir / "sim.json"
    if not path.is_file():
        return {"status": "missing", "expected_sha256": None, "actual_sha256": None}
    try:
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
    except OSError as exc:
        return {"status": "invalid", "error": str(exc),
                "expected_sha256": None, "actual_sha256": None}
    expected = data.manifest.get("provenance", {}).get("input_sha256")
    if expected and str(expected).lower() != actual:
        return {"status": "mismatch", "expected_sha256": str(expected),
                "actual_sha256": actual}
    try:
        load_sim_file(path)
    except Exception as exc:
        return {"status": "invalid", "error": str(exc),
                "expected_sha256": str(expected) if expected else None,
                "actual_sha256": actual}
    return {"status": "matched" if expected else "unverified",
            "expected_sha256": str(expected) if expected else None,
            "actual_sha256": actual}


def recorded_spec(data: SimulationData) -> dict:
    """The raw input document this result actually ran (sibling ``sim.json``).

    Gated on geometry provenance exactly like the scene overlays: a
    mismatched or unparseable sibling would show persuasive but false
    settings beside the recorded outputs, so it is withheld with the reason
    instead of served.

    The gate and the served document come from ONE read: hashing one read
    and serving another would let a file swapped mid-request ride out as
    "matched" — the exact substitution the gate exists to stop — and a file
    deleted in that window would surface as a bare 500 instead of a
    withheld-with-reason payload.
    """
    path = data.output_dir / "sim.json"
    if not path.is_file():
        return {"available": False, "geometry_status": "missing",
                "reason": "this bundle has no sibling sim.json"}
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return {"available": False, "geometry_status": "invalid",
                "reason": str(exc)}
    expected = data.manifest.get("provenance", {}).get("input_sha256")
    actual = hashlib.sha256(raw_bytes).hexdigest()
    if expected and str(expected).lower() != actual:
        return {"available": False, "geometry_status": "mismatch",
                "reason": "the sibling sim.json is not the input recorded by "
                          "this result bundle"}
    status = "matched" if expected else "unverified"
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
        sim, _ = parse_sim_spec(raw)
    except Exception:
        return {"available": False, "geometry_status": "invalid",
                "reason": "the sibling sim.json failed to parse"}
    # Mode-source provenance evaluated against the RECORDED document itself,
    # so the read-only Sources page reports the truth (a recorded source with
    # a matching solve recipe is "fresh", never "legacy · unverified").
    return {"available": True, "geometry_status": status, "spec": raw,
            "mode_source_statuses": mode_source_statuses(sim)}


def monitor_catalog(data: SimulationData) -> list[dict]:
    """Per-monitor slider metadata, read from the MANIFEST only (no blob load)."""
    cat = []
    for m in data.manifest.get("monitors", []):
        comps = [str(c) for c in m.get("components", [])]
        e: dict[str, Any] = {
            "name": m["name"],
            "type": m.get("type"),
            "kind": _kind(m),
            "dims": list(m.get("dims", [])),
            "shape": list(m.get("shape", [])),
            "components": comps,
            "derived": _derived_components(comps),
        }
        if "freqs_hz" in m:
            e["freqs_hz"] = list(m["freqs_hz"])
        if "sample_steps" in m:
            e["sample_steps"] = list(m["sample_steps"])
        if "axis" in m:
            e["axis"] = m["axis"]
        if m.get("type") == "field_dft":
            e["normalization"] = "phasor / (first-source A0*S(f))"
        elif m.get("type") == "flux":
            e["normalization"] = "signed power / |first-source A0*S(f)|²"
            e["sign_convention"] = f"positive toward +{m.get('axis', '?')}"
        elif m.get("type") in {"field_time", "field_snapshot"}:
            e["timing_convention"] = "H samples lag E by dt/2"
        cat.append(e)
    return cat


def _derived_components(comps: list[str]) -> list[str]:
    """Magnitude/intensity options when a full vector is present (mirrors
    viz.interactive._field_component_options)."""
    out = []
    if {"Ex", "Ey", "Ez"} <= set(comps):
        out += ["E", "intensity"]
    if {"Hx", "Hy", "Hz"} <= set(comps):
        out += ["H"]
    return out


def _component_array(da, field: str):
    """Select a component (direct Ex.. or derived E/intensity/H) from a DataArray
    that still carries the ``component`` dim."""
    if "component" not in da.dims:
        return da
    if field in ("E", "H", "intensity"):
        axis = "E" if field in ("E", "intensity") else "H"
        comps = [f"{axis}{c}" for c in "xyz"]
        mag = np.sqrt(sum(np.abs(da.sel(component=c)) ** 2 for c in comps))
        return mag ** 2 if field == "intensity" else mag
    return da.sel(component=field)


def meta(data: SimulationData, monitor: str) -> dict:
    """Full slider ranges for a monitor — kind, components, value reducers, freqs,
    times, and per-axis spatial coords (for the cut-plane). Loads the DataArray
    (cached) to read its real coords; the UI fetches this once on selection."""
    mm = next((m for m in data.manifest.get("monitors", []) if m["name"] == monitor), {})
    da = data[monitor]
    comps = [str(c) for c in da.coords["component"].values] if "component" in da.coords else []
    complex_ = bool(np.iscomplexobj(da.values))
    out: dict[str, Any] = {
        "name": monitor,
        "kind": _kind(mm),
        "components": comps,
        "derived": _derived_components(comps),
        "vals": ["real", "imag", "abs", "phase"] if complex_ else ["real", "abs"],
        "normalization": str(da.attrs.get("normalization", "")),
        "timing_convention": "H samples lag E by dt/2; spatial components use raw Yee locations",
    }
    if "f" in da.coords:
        out["freqs"] = [{"hz": float(x), "nm": _C / float(x) * 1e9}
                        for x in da.coords["f"].values]
    if "t" in da.coords:
        steps = list(da.attrs.get("sample_steps", []))
        out["times"] = [{"s": float(x), "step": int(steps[i]) if i < len(steps) else i}
                        for i, x in enumerate(da.coords["t"].values)]
    axes: dict[str, Any] = {}
    spatial_coords: dict[str, Any] = {}
    for ax in _SPATIAL:
        if ax in da.coords:
            c = da.coords[ax].values
            spatial_coords[ax] = c
            if da.coords[ax].size > 1:
                axes[ax] = {"min": float(c[0]), "max": float(c[-1]), "n": int(len(c))}
    out["axes"] = axes
    out["volumetric"] = len(axes) == 3
    # A DFT port plane is represented with all three spatial dimensions and one
    # singleton normal axis.  Record that plane explicitly: without it the old
    # UI omitted a cut axis and slice_plane defaulted to z, turning an x-normal
    # yz port into a one-cell-wide xy strip.
    singleton = [ax for ax, c in spatial_coords.items() if len(c) == 1]
    if len(spatial_coords) == 3 and len(singleton) == 1:
        normal = singleton[0]
        out["plane"] = {"axis": normal, "pos": float(spatial_coords[normal][0])}
    return out


# --------------------------------------------------------------------------- #
# Slicing + numeric outputs (shared by figures.py and the MCP tools)
# --------------------------------------------------------------------------- #

def _isel_nearest(array, dim: str, value: float):
    """Select the nearest recorded coordinate without a monotonic-index assumption."""
    coords = np.asarray(array.coords[dim].values, dtype=float)
    if coords.size == 0 or not np.all(np.isfinite(coords)):
        raise ValueError(f"coordinate {dim!r} has no finite recorded samples")
    index = int(np.argmin(np.abs(coords - float(value))))
    return array.isel({dim: index})

def slice_plane(data: SimulationData, monitor: str, *, field: str = "Ex",
                val: str = "real", freq: Optional[float] = None,
                time: Optional[float] = None, axis: Optional[str] = None,
                pos: Optional[float] = None):
    """Reduce a monitor to a 2D plane (component → freq/time → cut plane). Returns
    ``(arr2d, reduced_z, row_axis, col_axis, resolved)`` where ``resolved`` echoes
    the chosen freq_hz/time_s/cut. Raises ValueError for non-planar monitors."""
    da = data[monitor]
    # Select cheap scalar axes and the cut plane while a raw result is still a
    # disk-backed memmap.  Derived E/H/intensity used to be formed over the
    # entire 3-D, all-time/all-frequency monitor first, materializing GBs just
    # to display one plane.
    arr = da
    resolved: dict[str, Any] = {}
    if "f" in arr.dims:
        f = float(arr.coords["f"].values[0]) if freq is None else float(freq)
        arr = _isel_nearest(arr, "f", f)
        resolved["freq_hz"] = float(arr.coords["f"].values)
    if "t" in arr.dims:
        t = float(arr.coords["t"].values[-1]) if time is None else float(time)
        arr = _isel_nearest(arr, "t", t)
        resolved["time_s"] = float(arr.coords["t"].values)
    spatial = [d for d in arr.dims if d in _SPATIAL]
    if len(spatial) == 3:
        singleton = [d for d in spatial if arr.coords[d].size == 1]
        cut_axis = axis or (singleton[0] if len(singleton) == 1 else "z")
        if cut_axis not in spatial:
            raise ValueError(
                f"cut axis {cut_axis!r} is not present in monitor {monitor!r} "
                f"(spatial dims: {tuple(spatial)})")
        coords = arr.coords[cut_axis].values
        cpos = float(coords[len(coords) // 2]) if pos is None else float(pos)
        arr = _isel_nearest(arr, cut_axis, cpos)
        resolved["cut"] = {"axis": cut_axis, "value_um": float(arr.coords[cut_axis].values)}
        spatial = [d for d in arr.dims if d in _SPATIAL]
    arr = _component_array(arr, field)
    if len(spatial) != 2:
        raise ValueError(
            f"monitor {monitor!r} is not a 2D plane (dims after slicing: {tuple(arr.dims)}); "
            "use a time-series or spectrum view")
    z = _REDUCE.get(val, np.real)(arr.values)
    return arr, z, spatial[0], spatial[1], resolved


def field_stats(data: SimulationData, monitor: str, *, field: str = "Ex",
                val: str = "abs", freq: Optional[float] = None,
                time: Optional[float] = None, axis: Optional[str] = None,
                pos: Optional[float] = None) -> dict:
    """Summary statistics of a displayed field cut-plane.

    ``sample_sum_squares`` is deliberately named as a discrete display-space
    diagnostic: it has no cell-area quadrature, material weighting, or physical
    energy-density constants and therefore must not be interpreted as energy.
    Guards all-NaN planes (aborted/non-finite runs).
    """
    arr, z, row, col, resolved = slice_plane(
        data, monitor, field=field, val=val, freq=freq, time=time, axis=axis, pos=pos)
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)
    out: dict[str, Any] = {
        "monitor": monitor, "field": field, "val": val,
        "plane_axes": [row, col], "shape": [int(s) for s in z.shape],
        "aborted": bool(data.aborted), **resolved,
    }
    if not finite.any():
        out["error"] = "all values are non-finite (NaN/inf) — run may have diverged"
        return out
    clean = np.where(finite, z, np.nan)
    pk = np.unravel_index(int(np.nanargmax(np.abs(clean))), clean.shape)
    out.update({
        "min": float(np.nanmin(clean)), "max": float(np.nanmax(clean)),
        "mean": float(np.nanmean(clean)),
        "abs_max": float(np.nanmax(np.abs(clean))),
        "sample_sum_squares": float(np.nansum(clean * clean)),
        "nonfinite_count": int(clean.size - np.count_nonzero(finite)),
        "peak_at_um": {col: float(arr.coords[col].values[pk[1]]),
                       row: float(arr.coords[row].values[pk[0]])},
    })
    return out


def spectrum_values(data: SimulationData, monitor: str) -> dict:
    """A flux/DFT monitor's spectrum as plain numbers (wavelength_nm + value),
    sorted by wavelength."""
    da = data[monitor]
    if "f" not in da.dims:
        raise ValueError(f"monitor {monitor!r} has no frequency axis (not a spectrum)")
    f = np.asarray(da.coords["f"].values, dtype=float)
    y = np.real(da.values) if np.iscomplexobj(da.values) else np.asarray(da.values, dtype=float)
    lam = _C / f * 1e9
    order = np.argsort(lam)
    return {
        "monitor": monitor,
        "wavelength_nm": lam[order].tolist(),
        "freq_hz": f[order].tolist(),
        "value": np.asarray(y, dtype=float)[order].tolist(),
    }


# --------------------------------------------------------------------------- #
# Immutable-history A/B comparison
# --------------------------------------------------------------------------- #

def _pointer_token(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def json_pointer_diff(a: Any, b: Any, path: str = "") -> list[dict]:
    """Exact recursive JSON diff using RFC 6901 pointer paths.

    This deliberately applies no numerical tolerance: it describes the resolved
    canonical request/provenance records, not a physics acceptance criterion.
    """
    out: list[dict] = []
    truncated = False
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )

    def summary(value: list | dict) -> dict:
        digest = hashlib.sha256()
        for chunk in encoder.iterencode(value):
            digest.update(chunk.encode("utf-8"))
        return {
            "type": "array" if isinstance(value, list) else "object",
            "items": len(value), "sha256": digest.hexdigest(),
        }

    def leaf(value: Any) -> Any:
        # Added/removed collections can themselves contain giant mode-profile
        # arrays. The pointer already identifies the changed subtree; an exact
        # canonical hash keeps the response bounded without pretending detail.
        return summary(value) if isinstance(value, (list, dict)) else value

    def emit(change: dict) -> None:
        nonlocal truncated
        if len(out) >= _DIFF_ENTRY_LIMIT:
            truncated = True
            return
        out.append(change)

    def walk(left: Any, right: Any, current: str) -> None:
        if truncated:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right), key=str):
                if truncated:
                    break
                child = f"{current}/{_pointer_token(key)}"
                if key not in left:
                    emit({"path": child, "kind": "added", "a": None,
                          "b": leaf(right[key])})
                elif key not in right:
                    emit({"path": child, "kind": "removed",
                          "a": leaf(left[key]), "b": None})
                else:
                    walk(left[key], right[key], child)
            return
        if isinstance(left, list) and isinstance(right, list):
            if max(len(left), len(right)) > _DIFF_ARRAY_DETAIL_LIMIT:
                if left != right:
                    emit({
                        "path": current or "", "kind": "changed",
                        "a": summary(left), "b": summary(right),
                        "summarized": (
                            "large arrays are represented by exact JSON hashes"),
                    })
                return
            for index in range(max(len(left), len(right))):
                if truncated:
                    break
                child = f"{current}/{index}"
                if index >= len(left):
                    emit({"path": child, "kind": "added", "a": None,
                          "b": leaf(right[index])})
                elif index >= len(right):
                    emit({"path": child, "kind": "removed",
                          "a": leaf(left[index]), "b": None})
                else:
                    walk(left[index], right[index], child)
            return
        # JSON booleans are not numbers, and integer-vs-float is a real lexical
        # type change even though Python considers True == 1 and 1 == 1.0.
        if type(left) is type(right) and left == right:
            return
        change = {
            "path": current or "", "kind": "changed",
            "a": left, "b": right,
        }
        if (isinstance(left, (int, float)) and not isinstance(left, bool)
                and isinstance(right, (int, float)) and not isinstance(right, bool)
                and math.isfinite(float(left)) and math.isfinite(float(right))):
            change["delta"] = float(right) - float(left)
        emit(change)

    walk(a, b, path)
    if truncated:
        out.append({
            "path": path or "", "kind": "truncated", "a": None,
            "b": f"diff limited to {_DIFF_ENTRY_LIMIT} entries",
            "summarized": f"diff limited to {_DIFF_ENTRY_LIMIT} entries",
        })
    return out


def _metric_pair(a: Any, b: Any) -> dict:
    out = {"a": a, "b": b, "delta": None, "relative_change": None}
    if (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)
            and math.isfinite(float(a)) and math.isfinite(float(b))):
        out["delta"] = float(b) - float(a)
        if float(a) != 0:
            out["relative_change"] = (float(b) - float(a)) / abs(float(a))
    return out


def _num_cells(grid: dict) -> Optional[int]:
    shape = grid.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        return None
    try:
        return math.prod(int(value) for value in shape)
    except (TypeError, ValueError):
        return None


def _json_safe_vector(values: Any) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for value in np.asarray(values).reshape(-1):
        number = float(value)
        out.append(number if math.isfinite(number) else None)
    return out


def _monitor_map(data: SimulationData) -> dict[str, dict]:
    monitors = data.manifest.get("monitors", [])
    if not isinstance(monitors, list):
        return {}
    return {
        str(item.get("name")): item for item in monitors
        if isinstance(item, dict) and item.get("name")
    }


def _field_grid(data: SimulationData, name: str) -> dict:
    """Manifest-derived field signature; never opens the monitor binary.

    ``SimulationData[name]`` currently materializes the complete raw array, so a
    comparison eligibility check must reconstruct only its coordinate contract
    from the already-verified manifest.  The historical field endpoint performs
    the actual artifact checksum before any numerical slice is loaded.
    """
    monitor = _monitor_map(data).get(name)
    if monitor is None:
        raise KeyError(name)
    raw_dims = [str(value) for value in monitor.get("dims", [])]
    raw_shape = [int(value) for value in monitor.get("shape", [])]
    if len(raw_dims) != len(raw_shape):
        raise ValueError(
            f"monitor {name!r} dimensions and shape have different lengths")
    sizes = dict(zip(raw_dims, raw_shape))
    dims: list[str] = []
    shape: list[int] = []
    for dim, size in zip(raw_dims, raw_shape):
        if dim == "complex":
            continue
        dims.append("f" if dim == "freq" else "t" if dim == "sample" else dim)
        shape.append(size)
    coords: dict[str, list[float]] = {}
    if "freq" in sizes:
        coords["f"] = [float(value) for value in monitor.get("freqs_hz", [])]
    if "sample" in sizes:
        dt = float(monitor.get("dt_s", data.manifest.get("run", {}).get("dt_s")))
        coords["t"] = [
            float(value) * dt for value in monitor.get("sample_steps", [])]
    origin = list(monitor.get("origin_cells", (0, 0, 0)))
    stride = list(monitor.get("interval_space", (1, 1, 1)))
    if len(origin) != 3 or len(stride) != 3:
        raise ValueError(
            f"monitor {name!r} has invalid origin_cells/interval_space")
    for axis, offset, step in zip(("x", "y", "z"), origin, stride):
        if axis in sizes:
            coords[axis] = [
                float(value) for value in data._axis_coord_um(  # noqa: SLF001
                    axis, int(offset), sizes[axis], int(step))
            ]
    components = [str(value) for value in monitor.get("components", [])]
    return {
        "dims": dims, "shape": shape,
        "components": components, "coords": coords,
    }


def _plane_semantics(monitor: dict) -> dict:
    dims = list(monitor.get("dims", []))
    shape = list(monitor.get("shape", []))
    sizes = {str(dim): int(shape[index]) for index, dim in enumerate(dims)
             if index < len(shape)}
    varying = [axis for axis in _SPATIAL if sizes.get(axis, 0) > 1]
    singleton = [axis for axis in _SPATIAL if axis in sizes and sizes[axis] == 1]
    return {
        "volumetric": len(varying) == 3,
        "normal_axis": singleton[0] if len(varying) == 2 and singleton else None,
    }


def _has_shared_recorded_sample(a: list[float], b: list[float],
                                relative_tolerance: float = 1e-9) -> bool:
    for value in a:
        for candidate in b:
            scale = max(float.fromhex("0x0.0000000000001p-1022"),
                        abs(value), abs(candidate))
            if abs(value - candidate) <= relative_tolerance * scale:
                return True
    return False


def _requested_monitor(record: dict, name: str) -> dict:
    spec = record.get("request", {}).get("spec", {}).get("document", {})
    monitors = spec.get("monitors", []) if isinstance(spec, dict) else []
    return next((item for item in monitors
                 if isinstance(item, dict) and item.get("name") == name), {})


def _monitor_semantics(monitor: dict, requested: dict) -> dict:
    """Effective interpretation keys, including semantics derived by the reader.

    Engine v1 manifests encode normalization/timing through monitor type rather
    than repeating those prose strings.  The request is also needed for DFT
    apodization, which changes absolute field magnitudes but is not emitted in
    the result manifest.
    """
    mtype = monitor.get("type")
    if mtype == "flux":
        return {
            "normalization": monitor.get("normalization")
            or "signed power / |first-source A0*S(f)|^2",
            "sign_convention": monitor.get("sign_convention")
            or f"positive toward +{monitor.get('axis', '?')}",
        }
    if mtype == "field_dft":
        return {
            "normalization": monitor.get("normalization")
            or "phasor / (first-source A0*S(f))",
            "apodization": requested.get("apodization"),
        }
    if mtype in {"field_time", "field_snapshot"}:
        return {
            "timing_convention": monitor.get("timing_convention")
            or "H samples lag E by dt/2",
        }
    return {}


def compare_run_data(a_record: dict, b_record: dict,
                     a_data: SimulationData, b_data: SimulationData, *,
                     include_numerical: bool = True,
                     include_specs: bool = True) -> dict:
    """Compare two sealed runs without changing the viewer's current result.

    Spectra keep their independent wavelength grids; a pointwise delta appears
    only when those grids are exactly equal.  Field entries report whether the
    disk-backed grids are exactly compatible and provide immutable-history URLs
    for fetching matching slices.  Resampling is intentionally not invented.
    """
    a_summary = a_record.get("summary") or {}
    b_summary = b_record.get("summary") or {}
    a_run, b_run = a_summary.get("run", {}), b_summary.get("run", {})
    a_grid, b_grid = a_summary.get("grid", {}), b_summary.get("grid", {})
    a_reported = a_summary.get("provenance", {})
    b_reported = b_summary.get("provenance", {})

    def comparison_provenance(record: dict, reported: dict) -> dict:
        request = record.get("request") or {}
        return {
            "requested": {
                "execution": request.get("execution", {}),
                "solver": request.get("solver_requested", {}),
            },
            "reported": reported,
        }

    a_prov = comparison_provenance(a_record, a_reported)
    b_prov = comparison_provenance(b_record, b_reported)

    reasons: list[str] = []
    for label, record, data in (("A", a_record, a_data), ("B", b_record, b_data)):
        if record.get("status") != "completed":
            reasons.append(f"run {label} is {record.get('status')}, not completed")
        if bool(data.aborted):
            reasons.append(
                f"run {label} was solver-aborted ({data.abort_reason or 'unknown reason'})")
    comparable = not reasons

    a_total = (a_record.get("integrity") or {}).get("total_bytes")
    b_total = (b_record.get("integrity") or {}).get("total_bytes")
    metrics = {
        "num_cells": _metric_pair(_num_cells(a_grid), _num_cells(b_grid)),
        "steps_run": _metric_pair(a_run.get("steps_run"), b_run.get("steps_run")),
        "wall_seconds": _metric_pair(a_run.get("wall_seconds"), b_run.get("wall_seconds")),
        "setup_seconds": _metric_pair(a_run.get("setup_seconds"), b_run.get("setup_seconds")),
        "mcells_per_s": _metric_pair(a_run.get("mcells_per_s"), b_run.get("mcells_per_s")),
        "output_bytes": _metric_pair(a_total, b_total),
    }

    a_monitors, b_monitors = _monitor_map(a_data), _monitor_map(b_data)
    spectra: list[dict] = []
    fields: list[dict] = []
    for name in sorted(set(a_monitors) | set(b_monitors)):
        am, bm = a_monitors.get(name), b_monitors.get(name)
        types = {str(item.get("type", "")) for item in (am, bm) if item}
        family = "spectrum" if "flux" in types else (
            "field" if any(item.startswith("field_") for item in types) else None)
        if family is None:
            continue
        entry_reasons = list(reasons)
        if am is None:
            entry_reasons.append("monitor exists only in run B")
        if bm is None:
            entry_reasons.append("monitor exists only in run A")
        if am is not None and bm is not None:
            if am.get("type") != bm.get("type"):
                entry_reasons.append(
                    f"monitor types differ ({am.get('type')} vs {bm.get('type')})")
            if family == "spectrum":
                if am.get("axis") != bm.get("axis"):
                    entry_reasons.append(
                        f"flux sign axes differ ({am.get('axis')} vs {bm.get('axis')})")
            if family == "field":
                a_kind, b_kind = _kind(am), _kind(bm)
                if a_kind != "plane" or b_kind != "plane":
                    entry_reasons.append(
                        "historical side-by-side field comparison supports only "
                        f"2D plane monitors ({a_kind} vs {b_kind})")
                elif _plane_semantics(am) != _plane_semantics(bm):
                    entry_reasons.append(
                        "field monitors have different plane/volume orientation semantics")
            a_semantics = _monitor_semantics(
                am, _requested_monitor(a_record, name))
            b_semantics = _monitor_semantics(
                bm, _requested_monitor(b_record, name))
            if a_semantics != b_semantics:
                entry_reasons.append(
                    "monitor normalization/timing/apodization semantics differ")

        if family == "spectrum":
            entry: dict[str, Any] = {
                "name": name, "compatible": not entry_reasons,
                "reasons": entry_reasons, "grids_equal": None,
                "a": None, "b": None, "delta": None,
            }
            if not entry_reasons:
                af = np.asarray(am.get("freqs_hz", []), dtype=float)
                bf = np.asarray(bm.get("freqs_hz", []), dtype=float)
                ax = _C / af * 1e9
                bx = _C / bf * 1e9
                grids_equal = bool(np.array_equal(ax, bx))
                entry["grids_equal"] = grids_equal
                if include_numerical:
                    av = spectrum_values(a_data, name)
                    bv = spectrum_values(b_data, name)
                    ay = np.asarray(av["value"], dtype=float)
                    by = np.asarray(bv["value"], dtype=float)
                    entry["a"] = {
                        "wavelength_nm": _json_safe_vector(ax),
                        "value": _json_safe_vector(ay),
                    }
                    entry["b"] = {
                        "wavelength_nm": _json_safe_vector(bx),
                        "value": _json_safe_vector(by),
                    }
                    if grids_equal:
                        entry["delta"] = _json_safe_vector(by - ay)
                if not grids_equal:
                    entry["reasons"].append(
                        "wavelength grids differ; traces are returned independently without interpolation")
            spectra.append(entry)
            continue

        entry = {
            "name": name, "compatible": False, "reasons": entry_reasons,
            "side_by_side": False,
            "grids_equal": None, "shared_components": [],
            "a_endpoint": None, "b_endpoint": None,
        }
        if not entry_reasons:
            ag, bg = _field_grid(a_data, name), _field_grid(b_data, name)
            grids_equal = bool(
                ag["dims"] == bg["dims"] and ag["shape"] == bg["shape"]
                and ag["coords"] == bg["coords"]
            )
            shared = sorted(set(ag["components"]) & set(bg["components"]))
            samples_match = True
            for dim, label in (("f", "frequency"), ("t", "time")):
                av = ag["coords"].get(dim)
                bv = bg["coords"].get(dim)
                if (av is None) != (bv is None) or (
                    av is not None and bv is not None
                    and not _has_shared_recorded_sample(av, bv)
                ):
                    entry["reasons"].append(
                        f"field monitors have no shared recorded {label} sample")
                    samples_match = False
            entry["grids_equal"] = grids_equal
            entry["shared_components"] = shared
            if not grids_equal:
                entry["reasons"].append(
                    "field sampling grids differ; no resampling or difference field was produced")
            if not shared:
                entry["reasons"].append("field monitors have no shared components")
            # Different grids still support scientifically honest side-by-side
            # views. Only subtraction/difference fields require exact alignment.
            entry["side_by_side"] = bool(shared) and samples_match
            entry["compatible"] = not entry["reasons"]
        entry["a_endpoint"] = (
            f"/api/runs/{a_record['run_id']}/monitor/{name}/field" if am else None)
        entry["b_endpoint"] = (
            f"/api/runs/{b_record['run_id']}/monitor/{name}/field" if bm else None)
        fields.append(entry)

    def public_record(record: dict, data: SimulationData) -> dict:
        out = {key: record.get(key) for key in (
            "run_id", "created_at", "started_at", "finished_at", "status",
            "device", "output_dir", "spec_sha256", "summary", "integrity",
        )}
        summary = record.get("summary") or {}
        out.update({
            "run": summary.get("run", {}), "grid": summary.get("grid", {}),
            "provenance": summary.get("provenance", {}),
            "aborted": bool((summary.get("run") or {}).get("aborted", False)),
            "abort_reason": (summary.get("run") or {}).get("abort_reason") or None,
            "requested_execution": record.get("request", {}).get("execution", {}),
            "solver_requested": record.get("request", {}).get("solver_requested", {}),
        })
        if include_specs:
            out["monitors"] = monitor_catalog(data)
            out["spec"] = record.get("request", {}).get("spec", {}).get("document")
            out["manifest"] = summary
            out["session"] = session(
                data, result_id=None, run_id=record.get("run_id"))
        else:
            # The compare UI fetches selected monitor metadata/data lazily.  Keep
            # only the geometry trust needed for its overlay toggle here.
            out["session"] = {
                "run_id": record.get("run_id"), "result_id": None,
                "output_dir": str(data.output_dir),
                "run": summary.get("run", {}), "grid": summary.get("grid", {}),
                "provenance": summary.get("provenance", {}),
                "aborted": bool(data.aborted),
                "abort_reason": data.abort_reason,
                "geometry": geometry_status(data),
            }
        return out

    a_spec = a_record.get("request", {}).get("spec", {}).get("document", {})
    b_spec = b_record.get("request", {}).get("spec", {}).get("document", {})
    shared_spectra = [
        {"name": item["name"], "kind": "spectrum", "type": "flux"}
        for item in spectra if item.get("compatible")
    ]
    shared_fields = [
        {"name": item["name"], "kind": "field",
         "components": item.get("shared_components", []),
         "grids_equal": item.get("grids_equal")}
        for item in fields if item.get("side_by_side")
    ]
    eligibility_reasons = list(reasons)
    if not shared_spectra and not shared_fields:
        eligibility_reasons.append(
            "runs share no compatible spectrum or side-by-side field monitor")
    eligible = not reasons and bool(shared_spectra or shared_fields)
    return {
        "a": public_record(a_record, a_data), "b": public_record(b_record, b_data),
        "comparable": comparable, "reasons": reasons,
        "spec_diff": json_pointer_diff(a_spec, b_spec),
        "provenance_diff": json_pointer_diff(a_prov, b_prov),
        "metrics": metrics, "spectra": spectra, "fields": fields,
        "compatibility": {
            "compatible": eligible, "eligible": eligible,
            "reasons": eligibility_reasons,
            "shared_spectra": shared_spectra,
            "shared_fields": shared_fields,
        },
    }


def timeseries_values(data: SimulationData, monitor: str, *, field: str = "Ex",
                      val: str = "real") -> dict:
    """A point/time monitor's component vs time as plain numbers."""
    da = _component_array(data[monitor], field)
    if "t" not in da.dims:
        raise ValueError(f"monitor {monitor!r} has no time axis (not a time series)")
    if "component" in da.dims:  # safety: collapse any stray non-time dims
        raise ValueError(f"monitor {monitor!r}: ambiguous component for time series")
    t = np.asarray(da.coords["t"].values, dtype=float)
    y = _REDUCE.get(val, np.real)(da.values)
    return {"monitor": monitor, "field": field, "val": val,
            "time_s": t.tolist(), "value": np.asarray(y, dtype=float).tolist()}


def timeseries_fft_values(data: SimulationData, monitor: str, *,
                          field: str = "Ex") -> dict:
    """Discrete power spectrum of a recorded time series, in dB re its peak.

    The spectrum is of the RECORDED WINDOW — resolution is 1/T_record and a
    truncated ringdown leaks — so the payload carries ``resolution_hz`` and
    ``nyquist_hz`` for an honest display.  The mean is removed first (the DC
    bin is meaningless for a pulse-driven probe) and the DC bin is dropped.
    """
    da = _component_array(data[monitor], field)
    if "t" not in da.dims:
        raise ValueError(f"monitor {monitor!r} has no time axis (not a time series)")
    if "component" in da.dims:
        raise ValueError(f"monitor {monitor!r}: ambiguous component for time series")
    t = np.asarray(da.coords["t"].values, dtype=float)
    y = np.asarray(np.real(da.values), dtype=float)
    if t.size < 4:
        raise ValueError(
            f"monitor {monitor!r} recorded only {t.size} samples; a spectrum "
            "needs at least 4")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"monitor {monitor!r} has non-increasing time samples")
    y = y - float(np.mean(y))
    power = np.abs(np.fft.rfft(y)) ** 2
    freq = np.fft.rfftfreq(y.size, d=dt)
    keep = freq > 0.0
    freq, power = freq[keep], power[keep]
    peak = float(power.max()) if power.size else 0.0
    if peak <= 0.0:
        raise ValueError(
            f"monitor {monitor!r} is identically zero after removing the "
            "mean; no spectrum to show")
    db = 10.0 * np.log10(np.maximum(power / peak, 1e-30))
    return {
        "monitor": monitor, "field": field,
        "freq_hz": freq.tolist(),
        "wavelength_nm": (_C / freq * 1e9).tolist(),
        "psd_db": db.tolist(),
        "resolution_hz": 1.0 / (y.size * dt),
        "nyquist_hz": 0.5 / dt,
        "samples": int(y.size),
    }


def line_profile_values(data: SimulationData, monitor: str, *, field: str = "Ex",
                        val: str = "real", freq: Optional[float] = None,
                        time: Optional[float] = None) -> dict:
    """A rank-1 field monitor as coordinate/value arrays.

    Singleton spatial axes are squeezed explicitly; a field-DTF line is not a
    malformed heatmap.  Frequency/time selection happens before derived vector
    magnitude so a disk-backed result stays slice-sized.
    """
    da = data[monitor]
    arr = da
    resolved: dict[str, Any] = {}
    if "f" in arr.dims:
        chosen = float(arr.coords["f"].values[0]) if freq is None else float(freq)
        arr = _isel_nearest(arr, "f", chosen)
        resolved["freq_hz"] = float(arr.coords["f"].values)
    if "t" in arr.dims:
        chosen = float(arr.coords["t"].values[-1]) if time is None else float(time)
        arr = _isel_nearest(arr, "t", chosen)
        resolved["time_s"] = float(arr.coords["t"].values)
    for axis in _SPATIAL:
        if axis in arr.dims and arr.sizes[axis] == 1:
            arr = arr.isel({axis: 0}, drop=True)
    arr = _component_array(arr, field)
    spatial = [axis for axis in _SPATIAL if axis in arr.dims]
    if len(spatial) != 1:
        raise ValueError(
            f"monitor {monitor!r} is not a 1D profile (remaining spatial dims: {spatial})")
    axis = spatial[0]
    values = _REDUCE.get(val, np.real)(arr.values)
    return {
        "monitor": monitor, "field": field, "val": val, "axis": axis,
        "coord_um": np.asarray(arr.coords[axis].values, dtype=float).tolist(),
        "value": np.asarray(values, dtype=float).tolist(), **resolved,
    }


def field_spectrum_values(data: SimulationData, monitor: str, *, field: str = "Ex",
                          val: str = "abs") -> dict:
    """A rank-0 field DFT monitor as field amplitude/phase versus wavelength."""
    arr = data[monitor]
    for axis in _SPATIAL:
        if axis in arr.dims and arr.sizes[axis] == 1:
            arr = arr.isel({axis: 0}, drop=True)
    arr = _component_array(arr, field)
    if "f" not in arr.dims:
        raise ValueError(f"monitor {monitor!r} has no frequency axis")
    leftovers = [d for d in arr.dims if d != "f"]
    if leftovers:
        raise ValueError(
            f"monitor {monitor!r} is not a point field spectrum (extra dims: {leftovers})")
    f = np.asarray(arr.coords["f"].values, dtype=float)
    values = np.asarray(_REDUCE.get(val, np.abs)(arr.values), dtype=float)
    wavelength = _C / f * 1e9
    order = np.argsort(wavelength)
    return {
        "monitor": monitor, "field": field, "val": val,
        "wavelength_nm": wavelength[order].tolist(),
        "freq_hz": f[order].tolist(), "value": values[order].tolist(),
        "normalization": str(arr.attrs.get("normalization", "")),
    }


# --------------------------------------------------------------------------- #
# Modal ports — persisted DFT-plane recipes -> one driven scattering column
# --------------------------------------------------------------------------- #

_PORT_TRANSVERSE = {
    "x": ("y", "z"),
    "y": ("z", "x"),
    "z": ("x", "y"),
}
_PORT_DB_FLOOR = 1.0e-30  # finite JSON/plot value for an exactly zero channel


def _modal_port_entries(sim) -> list[tuple[Any, Any, str]]:
    """``[(FieldDftMonitor, ModePort, normal_axis)]`` in monitor order."""
    if sim is None:
        return []
    from ..components.monitors import FieldDftMonitor

    entries = []
    for monitor in sim.monitors:
        if not isinstance(monitor, FieldDftMonitor) or monitor.mode_port is None:
            continue
        zero_axes = [
            axis for axis, size in enumerate(monitor.size_um)
            if float(size) == 0.0
        ]
        # A parsed Simulation has already run the cross-field validator.  Keep
        # this guard because tests/agent callers may pass a model-like object.
        if len(zero_axes) != 1:
            raise ValueError(
                f"modal port {monitor.name!r} must be a plane with exactly one "
                "zero size_um axis")
        entries.append((monitor, monitor.mode_port, _SPATIAL[zero_axes[0]]))
    return entries


def assert_modal_ports_ready(sim) -> None:
    """Fail before execution when declared ports cannot yield one S-column.

    The schema deliberately permits an unbound port while a user is authoring
    it. Execution is a stronger boundary: all ports must share one frequency
    grid, exactly one solved ModeSource must be the sole active excitation, and
    every requested cross-section solve must fit the Workbench resource guard.
    """
    from ..components.sources import ModeSource

    entries = _modal_port_entries(sim)
    if not entries:
        return
    driven_entries = [
        entry for entry in entries if entry[1].source_index is not None
    ]
    if len(driven_entries) != 1:
        raise ValueError(
            "modal results require exactly one source-linked driven port; "
            f"found {len(driven_entries)}")
    driven_monitor, driven_recipe, driven_axis = driven_entries[0]
    reference_freqs = tuple(float(value) for value in driven_monitor.freqs_hz)
    mismatched = [
        monitor.name for monitor, _, _ in entries
        if tuple(float(value) for value in monitor.freqs_hz) != reference_freqs
    ]
    if mismatched:
        raise ValueError(
            "modal results require every port to use the driven port's exact "
            f"frequency grid; mismatched ports: {mismatched}")

    source_index = int(driven_recipe.source_index)
    if source_index >= len(sim.sources):
        raise ValueError(
            f"driven port {driven_monitor.name!r} references missing source "
            f"index {source_index}")
    if source_index != 0:
        raise ValueError(
            f"driven port {driven_monitor.name!r} must use source_index 0 "
            "because the engine normalizes every DFT monitor by the first "
            "wire-order source")
    source = sim.sources[source_index]
    if not isinstance(source, ModeSource) or source.mode_solve is None:
        raise ValueError(
            f"driven port {driven_monitor.name!r} must reference a solved "
            "ModeSource with mode_solve provenance")
    active_sources = [
        index for index, candidate in enumerate(sim.sources)
        if abs(float(getattr(candidate, "amplitude", 1.0))) > 0.0
    ]
    if active_sources != [source_index]:
        labels = [index + 1 for index in active_sources]
        raise ValueError(
            "modal S-parameters require the driven ModeSource to be the only "
            f"active excitation; active sources: {labels}")

    # Use the actual mixed-Yee reference planes when a full Simulation is
    # available. Lightweight unit doubles fall back to nominal positions.
    normal_index = _SPATIAL.index(driven_axis)
    port_position = float(driven_monitor.center_um[normal_index])
    source_position = float(source.position_um)
    if hasattr(sim, "grid"):
        from ..components.grid import snap_mixed_plane
        port_position, _ = snap_mixed_plane(
            sim, normal_index, port_position)
        source_position, _ = snap_mixed_plane(
            sim, normal_index, source_position)
    travel = 1.0 if source.direction == "+" else -1.0
    if (port_position - source_position) * travel <= 0.0:
        raise ValueError(
            f"driven port {driven_monitor.name!r} must lie downstream of its "
            f"ModeSource plane along {source.direction}{driven_axis}")

    # Match the guided-source editor's per-window guard and add an aggregate
    # bound for a multi-port sweep. This keeps a GET from becoming an
    # unbounded sparse-eigensolver workload.
    total_frames = 0
    total_cell_frames = 0
    if hasattr(sim, "grid"):
        from ..components.monitors import mode_port_trial_modes
        from ..plugins.yee_mode import window_nodes
        for monitor, recipe, axis in entries:
            h_nodes, _, _, v_nodes, _, _ = window_nodes(
                sim, axis,
                h_center=float(recipe.center_um[0]),
                half_w=float(recipe.size_um[0]) / 2.0,
                v_center=float(recipe.center_um[1]),
                half_v=float(recipe.size_um[1]) / 2.0,
                dl=float(recipe.dl_um),
            )
            solve_cells = len(h_nodes) * len(v_nodes)
            if solve_cells > 50_000:
                raise ValueError(
                    f"modal port {monitor.name!r} resolves to {solve_cells:,} "
                    "cross-section cells; reduce its solve span or increase "
                    "mode grid spacing (Workbench limit: 50,000)")
            trial_modes = mode_port_trial_modes(
                recipe.modes, recipe.num_modes)
            if trial_modes >= 2 * solve_cells - 1:
                raise ValueError(
                    f"modal port {monitor.name!r} requests {trial_modes} trial "
                    f"modes on only {solve_cells} cross-section cells; reduce "
                    "trial modes or enlarge the solve window")
            frames = len(monitor.freqs_hz)
            total_frames += frames
            total_cell_frames += solve_cells * frames
    if total_frames > 1_024 or total_cell_frames > 20_000_000:
        raise ValueError(
            "modal port sweep is too large for interactive post-processing; "
            f"requested {total_frames:,} port-frequency solves over "
            f"{total_cell_frames:,} cell-frames (limits: 1,024 and 20,000,000)")


def modal_port_summaries_from_sim(sim) -> list[dict]:
    """Cheap result-navigation metadata from ``sim.json``; no field blobs/FDE."""
    summaries = []
    for monitor, port, axis in _modal_port_entries(sim):
        modes = list(port.modes)
        # Keep the compact fields for older clients, but also expose the
        # ordered channel list so mixed TE/TM ports are never mislabeled.
        family = modes[0].polarization
        summaries.append({
            "name": monitor.name,
            "monitor_name": monitor.name,
            "axis": axis,
            "position_um": float(monitor.center_um[_SPATIAL.index(axis)]),
            "out_direction": port.out_direction,
            "polarization": family,
            "mode_indices": [int(mode.mode_index) for mode in modes],
            "frequency_count": len(monitor.freqs_hz),
            "modes": [
                {
                    "polarization": mode.polarization,
                    "mode_index": int(mode.mode_index),
                }
                for mode in modes
            ],
            "source_index": port.source_index,
        })
    return summaries


def modal_port_monitor_names(data: SimulationData) -> list[str]:
    """Names of every result artifact needed by :func:`modal_port_results`."""
    sim = sim_for(data)
    if sim is None:
        raise FileNotFoundError(
            "modal results require a checksum-acceptable sim.json")
    names = [monitor.name for monitor, _, _ in _modal_port_entries(sim)]
    if not names:
        raise ValueError("this result defines no modal ports")
    return names


def modal_port_results(data: SimulationData) -> dict:
    """Resolve every saved modal port into one complex driven S-column.

    A schema-1.16 modal port remains an ordinary four-tangential-field DFT
    plane to the native engine.  Here its saved Yee solve recipe is replayed at
    every recorded frequency, one virtual :class:`ModeMonitor` / :class:`SPort`
    is built per requested TE/TM family channel, and ``S_ij = b_i/a_j`` is
    evaluated against the one source-linked driven channel.

    The returned object is JSON-native and matches the desktop Results contract.
    ``power`` is ``|S|^2``; ``db`` is ``10 log10(power)`` with an explicit
    -300 dB serialization floor at exactly zero.  Phase remains referenced to
    each recorded monitor plane; no de-embedding is inferred.
    """
    from ..components.sources import ModeSource
    from ..components.grid import snap_mixed_plane
    from ..plugins.mode_devices import ModeMonitor
    from ..plugins.smatrix import SPort, smatrix
    from ..plugins.yee_mode import solve_yee_port_mode_bank
    from . import _geometry as geom

    sim = sim_for(data)
    if sim is None:
        raise FileNotFoundError(
            "modal results require a checksum-acceptable sim.json")
    entries = _modal_port_entries(sim)
    if not entries:
        raise ValueError("this result defines no modal ports")
    assert_modal_ports_ready(sim)

    manifest_by_name = {
        str(item.get("name")): item
        for item in data.manifest.get("monitors", [])
        if isinstance(item, dict)
    }
    missing = [monitor.name for monitor, _, _ in entries
               if monitor.name not in manifest_by_name]
    if missing:
        raise ValueError(
            f"result bundle is missing modal-port monitor data: {missing}")
    artifact_frequency_mismatches = []
    for monitor, _, _ in entries:
        raw_freqs = manifest_by_name[monitor.name].get("freqs_hz")
        try:
            artifact_freqs = tuple(float(value) for value in raw_freqs)
        except (TypeError, ValueError):
            artifact_freqs = ()
        if artifact_freqs != tuple(
                float(value) for value in monitor.freqs_hz):
            artifact_frequency_mismatches.append(monitor.name)
    if artifact_frequency_mismatches:
        raise ValueError(
            "modal-port artifact frequency grids must exactly match sim.json; "
            f"mismatched monitors: {artifact_frequency_mismatches}")

    driven_entries = [entry for entry in entries if entry[1].source_index is not None]
    if len(driven_entries) != 1:
        raise ValueError(
            "modal results require exactly one source-linked driven port; "
            f"found {len(driven_entries)}")
    driven_monitor, driven_recipe, driven_axis = driven_entries[0]
    reference_freqs = tuple(float(value) for value in driven_monitor.freqs_hz)
    mismatched = [
        monitor.name for monitor, _, _ in entries
        if tuple(float(value) for value in monitor.freqs_hz) != reference_freqs
    ]
    if mismatched:
        raise ValueError(
            "modal results require every port to use the driven port's exact "
            f"frequency grid; mismatched ports: {mismatched}"
        )
    source_index = int(driven_recipe.source_index)
    if source_index >= len(sim.sources):
        raise ValueError(
            f"driven port {driven_monitor.name!r} references missing source "
            f"index {source_index}")
    source = sim.sources[source_index]
    if not isinstance(source, ModeSource) or source.mode_solve is None:
        raise ValueError(
            f"driven port {driven_monitor.name!r} must reference a solved "
            "ModeSource with mode_solve provenance")
    source_status = next((
        item for item in mode_source_statuses(sim)
        if item["source_index"] == source_index
    ), None)
    if source_status is not None and source_status["status"] == "stale":
        raise StaleModeSourceError([source_status])
    from ..components.monitors import mode_port_physical_polarization
    driven_mode = (
        mode_port_physical_polarization(
            source.mode_solve.polarization,
            driven_axis,
            driven_recipe.thickness_axis,
        ),
        int(source.mode_solve.mode_index),
    )

    virtual_ports = []
    virtual_by_key = {}
    public_ports = []
    for port_index, (monitor, recipe, axis) in enumerate(entries):
        axis_index = _SPATIAL.index(axis)
        position_um = float(monitor.center_um[axis_index])
        natural_axes = geom.in_plane_axes(axis)
        center_by_axis = dict(zip(natural_axes, map(float, recipe.center_um)))
        thickness_axis = recipe.thickness_axis or natural_axes[1]
        # The Yee port solver always returns VectorMode arrays in its natural
        # raster frame: mode-x is the editor's horizontal axis and mode-y is
        # its vertical axis.  ``recipe.thickness_axis`` changes only which
        # solver family a physical TE/TM request selects; it does not rotate
        # those returned arrays.  Project them with the natural vertical axis
        # so ModeMonitor preserves that raster orientation.
        projection_thickness_axis = natural_axes[1]
        requested = [
            (mode.polarization, int(mode.mode_index))
            for mode in recipe.modes
        ]
        snapped_position, normal_dl = snap_mixed_plane(
            sim, axis_index, position_um)
        bank = solve_yee_port_mode_bank(
            sim, axis, snapped_position, monitor.freqs_hz,
            modes=requested,
            h_center_um=float(recipe.center_um[0]),
            v_center_um=float(recipe.center_um[1]),
            half_w_um=float(recipe.size_um[0]) / 2.0,
            half_v_um=float(recipe.size_um[1]) / 2.0,
            dl_um=float(recipe.dl_um),
            supersample=int(recipe.supersample),
            num_modes=recipe.num_modes,
            thickness_axis=thickness_axis,
        )
        if not bank:
            raise ValueError(f"modal port {monitor.name!r} solved no frequencies")
        first_frequency = next(iter(bank))
        overlap_center = tuple(
            center_by_axis[name] for name in _PORT_TRANSVERSE[axis])

        public = {
            "name": monitor.name,
            "monitor_name": monitor.name,
            "axis": axis,
            "position_um": float(snapped_position),
            "out_direction": recipe.out_direction,
            "modes": [],
        }
        public_ports.append(public)
        for mode_index, mode_key in enumerate(requested):
            modes_by_freq = {frequency: per_mode[mode_key]
                             for frequency, per_mode in bank.items()}
            runtime_monitor = ModeMonitor(
                field_monitor=monitor,
                mode=modes_by_freq[first_frequency],
                axis=axis,
                center_um=overlap_center,
                direction=recipe.out_direction,
                thickness_axis=projection_thickness_axis,
                modes_by_freq=modes_by_freq,
                dl_um=float(normal_dl),
                simulation=sim,
                per_freq_modes=False,
            )
            virtual_name = f"port{port_index}:mode{mode_index}"
            runtime_port = SPort(
                virtual_name, runtime_monitor,
                out_direction=recipe.out_direction,
            )
            virtual_ports.append(runtime_port)
            virtual_by_key[(monitor.name, mode_key)] = (runtime_port, public)

    driven_runtime = virtual_by_key.get((driven_monitor.name, driven_mode))
    if driven_runtime is None:
        raise ValueError(
            f"driven port {driven_monitor.name!r} does not request its launched "
            f"{driven_mode[0]}{driven_mode[1]} mode")
    driven_runtime_port = driven_runtime[0]
    column = smatrix(virtual_ports, driven_runtime_port.name, data)

    for (_port_name, mode_key), (runtime_port, public) in virtual_by_key.items():
        values = column[(runtime_port.name, driven_runtime_port.name)]
        samples = []
        for frequency, value in sorted(
                values.items(), key=lambda item: _C / float(item[0]) * 1e9):
            sij = complex(value)
            power = float(abs(sij) ** 2)
            samples.append({
                "polarization": mode_key[0],
                "mode_index": int(mode_key[1]),
                "wavelength_nm": float(_C / float(frequency) * 1e9),
                "freq_hz": float(frequency),
                "power": power,
                "db": float(10.0 * math.log10(max(power, _PORT_DB_FLOOR))),
                "phase_deg": float(math.degrees(math.atan2(sij.imag, sij.real))),
                "s_re": float(sij.real),
                "s_im": float(sij.imag),
            })
        public["modes"].extend(samples)

    return {
        "driven_port": driven_monitor.name,
        "driven_mode": {
            "polarization": driven_mode[0],
            "mode_index": driven_mode[1],
        },
        "normalization": (
            "outgoing modal amplitude / driven incident modal amplitude; "
            "power = |S|^2"
        ),
        "reference_plane": "recorded monitor planes; no phase de-embedding",
        "ports": public_ports,
    }


# --------------------------------------------------------------------------- #
# Geometry (sim.json) — data only; figures.py turns outlines into traces
# --------------------------------------------------------------------------- #

def sim_for(data: SimulationData):
    """The Simulation parsed from the bundle's identity-cached ``sim.json``, or
    None when the bundle carries no geometry."""
    p = data.output_dir / "sim.json"
    if geometry_status(data)["status"] not in {"matched", "unverified"}:
        return None
    key = str(p.resolve())
    stat = p.stat()
    identity = (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    hit = _SIM_CACHE.get(key)
    if hit is None or hit[0] != identity:
        from ..components.simulation import Simulation
        sim = Simulation.from_wire_json(p.read_text())
        if len(_SIM_CACHE) >= 8 and key not in _SIM_CACHE:
            _SIM_CACHE.pop(next(iter(_SIM_CACHE)))  # bound: evict oldest
        _SIM_CACHE[key] = (identity, sim)
        return sim
    return hit[1]


def _spec_loops(spec) -> list:
    """A §5 cut-plane patch spec -> closed (h, v) point loops (one per ring)."""
    kind = spec[0]
    if kind == "rect":
        h0, v0, w, hh = spec[1]
        return [[(h0, v0), (h0 + w, v0), (h0 + w, v0 + hh), (h0, v0 + hh), (h0, v0)]]
    if kind == "polygon":
        verts = list(spec[1])
        return [verts + [verts[0]]] if verts else []
    if kind == "circle":
        cu, cv, r = spec[1]
        return [[(cu + r * math.cos(t), cv + r * math.sin(t)) for t in _CIRC]]
    if kind == "annulus":
        cu, cv, r, ir = spec[1]
        return [[(cu + r * math.cos(t), cv + r * math.sin(t)) for t in _CIRC],
                [(cu + ir * math.cos(t), cv + ir * math.sin(t)) for t in _CIRC]]
    return []


def structure_outlines(sim, cut_axis: str, value: float) -> Optional[dict]:
    """Structure cross-sections on a cut plane as (h, v) point loops, tagged with
    the in-plane axes. Pure geometry — ``figures`` maps these into the heatmap
    frame. Returns None when there's no geometry/plane."""
    if sim is None:
        return None
    from . import _geometry as geom
    try:
        h_axis, v_axis = geom.in_plane_axes(cut_axis)
    except ValueError:
        return None
    loops: list = []
    for st in getattr(sim, "structures", []):
        try:
            spec = geom.structure_patch_spec(st.geometry, cut_axis, value)
        except Exception:
            continue  # trust-but-don't-crash on an unknown geometry
        if spec:
            loops.extend(_spec_loops(spec))
    return {"h_axis": h_axis, "v_axis": v_axis, "loops": loops}


def eps_plane_sim(sim, axis: str = "z", value: float = 0.0) -> dict:
    """Permittivity on a cut plane of a ``Simulation`` spec (hard ε, last-structure-
    wins): the in-plane axes, their µm cell centers, and the eps array (row=v, col=h).
    Takes a sim directly so it serves both a result bundle and a live preview spec."""
    if sim is None:
        raise FileNotFoundError("no geometry — the simulation defines no scene")
    from .eps import sample_eps_plane
    from . import _geometry as geom
    h_nodes, v_nodes, eps2d = sample_eps_plane(sim, axis, float(value))
    h_nodes = np.asarray(h_nodes, dtype=float)
    v_nodes = np.asarray(v_nodes, dtype=float)
    h_axis, v_axis = geom.in_plane_axes(axis)
    return {
        "h_axis": h_axis, "v_axis": v_axis,
        "h_um": ((h_nodes[:-1] + h_nodes[1:]) / 2).tolist(),
        "v_um": ((v_nodes[:-1] + v_nodes[1:]) / 2).tolist(),
        "eps": np.asarray(eps2d, dtype=float).tolist(),
    }


def eps_plane(data: SimulationData, axis: str = "z", value: float = 0.0) -> dict:
    """Permittivity cut plane of a result bundle's geometry (via its ``sim.json``)."""
    sim = sim_for(data)
    if sim is None:
        raise FileNotFoundError("no sim.json next to the result bundle — geometry unavailable")
    return eps_plane_sim(sim, axis, value)


# --------------------------------------------------------------------------- #
# Preview — render a Simulation *spec file* (the input as data) with no results.
# The spec file (``sim.json`` = ``Simulation.to_wire_json``) is the narrow waist:
# the viewer, run_local, and the cloud all consume it. Watching the file (not the
# Python code) gives live authoring with no code execution.
# --------------------------------------------------------------------------- #

def load_sim_file(path: str | Path):
    """Parse a ``Simulation`` from a spec file (``sim.json``). The input *is* data —
    no Python evaluation, just JSON in."""
    from ..components.simulation import Simulation
    return Simulation.from_wire_json(Path(path).read_text())


def parse_sim_spec(spec: Any):
    """Validate a JSON-level simulation spec and return ``(sim, warnings)``.

    The desktop workbench uses this as its single validation boundary.  Keeping
    parsing here means the GUI, file preview, and run path all exercise the same
    pydantic/wire-ingest rules as :meth:`Simulation.from_wire_json`; the UI never
    grows a second, subtly different solver schema.
    """
    from ..components.simulation import Simulation

    if not isinstance(spec, dict):
        raise ValueError("simulation spec must be a JSON object")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = Simulation.from_wire_json(json.dumps(spec))
    messages = []
    for item in caught:
        text = str(item.message)
        if text not in messages:
            messages.append(text)
    return sim, messages


def mode_source_input_sha256(sim, source_index: int, recipe=None) -> str:
    """Hash the canonical inputs that can change a solved mode profile.

    This is intentionally conservative: renaming or changing any structure can
    mark the profile stale even when that structure is far from the source
    plane.  A false-positive re-solve is safe; silently reusing a profile from
    different material/grid inputs is not.  Numerically irrelevant run,
    monitor, PML, and source-amplitude settings are excluded.
    """
    from ..components.sources import ModeSource

    if source_index < 0 or source_index >= len(sim.sources):
        raise IndexError(f"source_index {source_index} is out of range")
    source = sim.sources[source_index]
    if not isinstance(source, ModeSource):
        raise TypeError(f"sources[{source_index}] is not a ModeSource")
    if recipe is None:
        recipe = source.mode_solve
    if recipe is None:
        raise ValueError("mode source has no solve recipe")
    if hasattr(recipe, "model_dump"):
        recipe_data = recipe.model_dump(mode="json", exclude={"input_sha256"})
    elif isinstance(recipe, dict):
        recipe_data = {
            key: value for key, value in recipe.items()
            if key != "input_sha256"
        }
    else:
        raise TypeError("mode solve recipe must be an object")

    pulse = {"freq0_hz": float(source.source_time.freq0_hz)}
    if int(recipe_data.get("num_freqs", 1)) > 1:
        pulse["fwidth_hz"] = float(source.source_time.fwidth_hz)
    material_inputs = {
        "size_um": [float(value) for value in sim.size_um],
        "grid": sim.grid.model_dump(
            mode="json", by_alias=True, exclude_none=True),
        "background": sim.background.model_dump(
            mode="json", by_alias=True, exclude_none=True),
        # Structure names are authoring metadata with no rasterization
        # semantics; renaming a card must not force a mode re-solve.
        "structures": [
            structure.model_dump(
                mode="json", by_alias=True, exclude_none=True,
                exclude={"name"},
            )
            for structure in sim.structures
        ],
        "symmetry": [int(value) for value in sim.symmetry],
    }
    source_inputs = {
        "axis": source.axis,
        "position_um": float(source.position_um),
        # Direction changes the paired-H packing even though it does not change
        # the eigenvalue problem, so it belongs in the resolved-profile hash.
        "direction": source.direction,
        "source_time": pulse,
    }
    canonical = json.dumps(
        {
            "fingerprint_version": 1,
            "simulation": material_inputs,
            "source": source_inputs,
            "mode_solve": recipe_data,
        },
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def mode_source_statuses(sim) -> list[dict]:
    """Fresh/stale/legacy provenance for every ModeSource in source order."""
    from ..components.sources import ModeSource

    statuses = []
    for source_index, source in enumerate(sim.sources):
        if not isinstance(source, ModeSource):
            continue
        provenance = source.mode_solve
        if provenance is None:
            statuses.append({
                "source_index": source_index,
                "status": "legacy",
                "message": (
                    "Solved profile has no saved solve recipe; it is preserved "
                    "but its geometry/grid provenance cannot be verified."),
                "expected_sha256": None,
                "actual_sha256": None,
            })
            continue
        expected = provenance.input_sha256
        actual = mode_source_input_sha256(sim, source_index, provenance)
        fresh = expected == actual
        statuses.append({
            "source_index": source_index,
            "status": "fresh" if fresh else "stale",
            "message": (
                "Solved profile matches the current geometry and grid."
                if fresh else
                "Geometry, grid, source placement/carrier, or solve settings "
                "changed after this profile was solved. Re-solve it before running."
            ),
            "expected_sha256": expected,
            "actual_sha256": actual,
        })
    return statuses


class StaleModeSourceError(ValueError):
    """Raised before execution when one or more solved profiles are stale."""

    def __init__(self, statuses: list[dict]):
        self.statuses = statuses
        indices = ", ".join(str(item["source_index"] + 1) for item in statuses)
        super().__init__(
            f"Mode source {indices} has a stale solved profile; re-solve it "
            "against the current geometry and grid before running.")


def assert_no_stale_mode_sources(sim) -> None:
    """Fail closed for stale provenance while allowing unverified legacy specs."""
    stale = [
        item for item in mode_source_statuses(sim)
        if item["status"] == "stale"
    ]
    if stale:
        raise StaleModeSourceError(stale)


class ModeSourceSolveError(ValueError):
    """Actionable validation failure from :func:`solve_mode_source`.

    ``field`` is deliberately UI-sized (for example ``"center_um"`` rather
    than a pydantic-internal path) so the HTTP facade can attach the message to
    the corresponding Configure mode control.
    """

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(message)


def solve_mode_source(sim, source_index: int, settings: dict):
    """Re-solve one canonical ``ModeSource`` from practical port settings.

    The Workbench must never ask a user to edit the coupled profile arrays by
    hand.  This operation takes the same compact controls exposed by commercial
    port/mode-source editors, solves an engine-consistent full-vector Yee mode,
    and replaces exactly one source with the canonical, power-normalized
    profile arrays the native solver consumes.

    ``center_um`` / ``size_um`` use the natural horizontal/vertical axes of the
    cross-section (``viz._geometry.in_plane_axes(axis)``).  The solve wavelength
    is authoritative for both the eigenmode and the returned Gaussian carrier;
    bandwidth, offset, phase, and the existing top-level source amplitude are
    preserved.  ``num_freqs > 1`` is the Tidy3D-style broadband option: modes
    are sampled over the carrier's +/-2-sigma frequency band.

    Returns ``(updated_simulation, summary)``.  The helper is presentation-free
    so the local HTTP facade and future agent/MCP surfaces share one workflow.
    """
    from ..components.sources import ModeSolveProvenance, ModeSource
    from ..components.source_time import GaussianPulse
    from ..components.grid import graded_primary_spacings, realized_cells
    from ..plugins.mode_devices import mode_launch
    from ..plugins.yee_mode import (
        solve_yee_mode,
        solve_yee_mode_bank,
        window_nodes,
    )
    from . import _geometry as geom

    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise ModeSourceSolveError(
            "source_index", "source_index must be an integer")
    if source_index < 0 or source_index >= len(sim.sources):
        raise ModeSourceSolveError(
            "source_index",
            f"source_index {source_index} is outside 0..{len(sim.sources) - 1}",
        )
    current = sim.sources[source_index]
    if not isinstance(current, ModeSource):
        raise ModeSourceSolveError(
            "source_index",
            f"sources[{source_index}] is {current.type}, not a mode_source",
        )
    if not isinstance(settings, dict):
        raise ModeSourceSolveError("settings", "mode solve settings must be an object")

    def finite_float(key: str, default=None, *, positive: bool = False) -> float:
        raw = settings.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ModeSourceSolveError(key, f"{key} must be a number") from None
        if not math.isfinite(value):
            raise ModeSourceSolveError(key, f"{key} must be finite")
        if positive and not value > 0.0:
            raise ModeSourceSolveError(key, f"{key} must be greater than 0")
        return value

    def bounded_int(key: str, default, lo: int, hi: int, label: str) -> int:
        raw = settings.get(key, default)
        if isinstance(raw, bool):
            raise ModeSourceSolveError(key, f"{label} must be an integer")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ModeSourceSolveError(key, f"{label} must be an integer") from None
        if isinstance(raw, float) and raw != value:
            raise ModeSourceSolveError(key, f"{label} must be an integer")
        if not lo <= value <= hi:
            raise ModeSourceSolveError(
                key, f"{label} must be between {lo} and {hi}")
        return value

    axis = str(settings.get("axis", current.axis)).lower()
    if axis not in ("x", "y", "z"):
        raise ModeSourceSolveError("axis", "axis must be x, y, or z")
    direction = str(settings.get("direction", current.direction))
    if direction not in ("+", "-"):
        raise ModeSourceSolveError("direction", "direction must be '+' or '-'")

    h_axis, v_axis = geom.in_plane_axes(axis)
    inferred_pol = "TE" if current.polarization == "E" + h_axis else "TM"
    polarization = str(settings.get("polarization", inferred_pol)).upper()
    if polarization not in ("TE", "TM"):
        raise ModeSourceSolveError(
            "polarization", "polarization must be TE or TM")
    mode_index = bounded_int("mode_index", 0, 0, 31, "mode index")

    default_wavelength = _C / float(current.source_time.freq0_hz) * 1e6
    wavelength_um = finite_float(
        "wavelength_um", default_wavelength, positive=True)
    position_um = finite_float(
        "position_um", current.position_um)
    dl_um = finite_float("dl_um", sim.grid.dl_um, positive=True)
    supersample = bounded_int(
        "supersample", 8, 1, 16, "supersample")
    num_freqs = bounded_int(
        "num_freqs", len(current.freqs_hz or ()) or 1,
        1, 11, "frequency samples")

    raw_num_modes = settings.get("num_modes")
    num_modes = None
    if raw_num_modes not in (None, ""):
        num_modes = bounded_int("num_modes", raw_num_modes, 1, 32, "mode count")
        if num_modes <= mode_index:
            raise ModeSourceSolveError(
                "num_modes", "mode count must be greater than mode index")

    realized = sim._realized_um()
    h_i, v_i, axis_i = ("xyz".index(name) for name in (h_axis, v_axis, axis))

    def pair(key: str, default, *, positive: bool = False) -> tuple[float, float]:
        raw = settings.get(key, default)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ModeSourceSolveError(
                key, f"{key} must contain [horizontal, vertical] values")
        values = []
        for item in raw:
            try:
                value = float(item)
            except (TypeError, ValueError):
                raise ModeSourceSolveError(
                    key, f"{key} values must be numbers") from None
            if not math.isfinite(value):
                raise ModeSourceSolveError(key, f"{key} values must be finite")
            if positive and not value > 0.0:
                raise ModeSourceSolveError(
                    key, f"{key} values must be greater than 0")
            values.append(value)
        return values[0], values[1]

    default_center = (realized[h_i] / 2.0, realized[v_i] / 2.0)
    # A compact default avoids turning an old profile with no solve provenance
    # into an unexpectedly enormous eigenproblem.  Users can deliberately grow
    # the window up to the realized cross-section bounds.
    default_size = (min(realized[h_i], 4.0), min(realized[v_i], 2.0))
    center_um = pair("center_um", default_center)
    size_um = pair("size_um", default_size, positive=True)

    tol = 1e-12 * max(1.0, realized[h_i], realized[v_i])
    for label, center, span, extent in (
            (h_axis, center_um[0], size_um[0], realized[h_i]),
            (v_axis, center_um[1], size_um[1], realized[v_i])):
        lo, hi = center - span / 2.0, center + span / 2.0
        if lo < -tol or hi > extent + tol:
            raise ModeSourceSolveError(
                "center_um",
                f"mode window on {label} spans [{lo:.6g}, {hi:.6g}] um, "
                f"outside the realized domain [0, {extent:.6g}] um; "
                "move the center or reduce the size",
            )

    if position_um < -tol or position_um > realized[axis_i] + tol:
        raise ModeSourceSolveError(
            "position_um",
            f"source plane {axis}={position_um:.6g} um is outside the realized "
            f"domain [0, {realized[axis_i]:.6g}] um",
        )

    coords = getattr(sim.grid, "coords", None)
    if (coords is not None and not math.isclose(
            dl_um, float(sim.grid.dl_um), rel_tol=1e-12, abs_tol=1e-15)):
        raise ModeSourceSolveError(
            "dl_um",
            "a custom mode-solve spacing is not supported on a graded "
            f"Simulation; omit dl_um or use the canonical grid.dl_um "
            f"({sim.grid.dl_um:.6g} um)",
        )

    # Mirror the native §18 injection-plane bound so Configure fails next to
    # the position control rather than much later in phsolver preflight.
    q_axis = sim._axis_coords_um(axis_i)
    if q_axis is None:
        n_axis = realized_cells(sim.size_um[axis_i], sim.grid.dl_um)
        plane_index = int(math.floor(position_um / sim.grid.dl_um + 0.5))
        plane_coord = lambda k: k * sim.grid.dl_um
    else:
        q_values = list(q_axis)
        closing = q_values[-1] + graded_primary_spacings(tuple(q_values))[-1]
        nodes = q_values + [closing]
        # ``min`` over (distance, -index) reproduces the native higher-index
        # tie break without relying on floating iteration order.
        plane_index = min(
            range(len(nodes)), key=lambda k: (abs(nodes[k] - position_um), -k))
        n_axis = len(q_values)
        plane_coord = lambda k: nodes[k]
    boundary = getattr(sim.boundaries, axis)
    layers = (sim.pml_num_layers if boundary == "pml" else
              sim.absorber_num_layers if boundary == "absorber" else 0)
    lower_layers = 0 if sim.symmetry[axis_i] != 0 else layers
    lower_index = lower_layers + 1
    upper_index = min(n_axis - 2, n_axis - layers - 1)
    if lower_index > upper_index:
        raise ModeSourceSolveError(
            "position_um",
            f"the {axis} domain has no valid interior mode-source plane after "
            f"reserving {layers} {boundary} layers",
        )
    if not lower_index <= plane_index <= upper_index:
        raise ModeSourceSolveError(
            "position_um",
            f"source plane snaps to {axis}-grid index {plane_index}, inside the "
            f"{boundary} or edge guard; choose a position between "
            f"{plane_coord(lower_index):.6g} and "
            f"{plane_coord(upper_index):.6g} um",
        )

    h_nodes, _, _, v_nodes, _, _ = window_nodes(
        sim, axis, h_center=center_um[0], half_w=size_um[0] / 2.0,
        v_center=center_um[1], half_v=size_um[1] / 2.0, dl=dl_um,
    )
    solve_cells = len(h_nodes) * len(v_nodes)
    if solve_cells > 50_000:
        raise ModeSourceSolveError(
            "size_um",
            f"mode window resolves to {len(h_nodes)} x {len(v_nodes)} = "
            f"{solve_cells:,} cells; reduce its size or increase dl_um "
            "(Workbench limit: 50,000 cross-section cells)",
        )

    carrier_hz = _C / (wavelength_um * 1e-6)
    if num_freqs > 1 and not float(current.source_time.fwidth_hz) > 0.0:
        raise ModeSourceSolveError(
            "num_freqs",
            "broadband mode solving requires source bandwidth greater than 0 Hz",
        )
    pulse_data = current.source_time.model_dump(mode="python")
    pulse_data["freq0_hz"] = carrier_hz
    pulse = GaussianPulse.model_validate(pulse_data)

    solve_kwargs = {
        "h_center_um": center_um[0],
        "v_center_um": center_um[1],
        "half_w_um": size_um[0] / 2.0,
        "half_v_um": size_um[1] / 2.0,
        "dl_um": dl_um,
        "supersample": supersample,
        "num_modes": num_modes,
    }
    frequency_samples = [carrier_hz]
    modes_by_freq = None
    if num_freqs == 1:
        mode = solve_yee_mode(
            sim, axis, position_um, wavelength_um, polarization, mode_index,
            **solve_kwargs,
        )
    else:
        half_band = 2.0 * float(pulse.fwidth_hz)
        if carrier_hz <= half_band:
            raise ModeSourceSolveError(
                "num_freqs",
                "the +/-2-sigma mode-solve band reaches zero frequency; "
                "reduce source bandwidth or use one frequency sample",
            )
        frequency_samples = [float(f) for f in np.linspace(
            carrier_hz - half_band, carrier_hz + half_band, num_freqs)]
        if len(set(frequency_samples)) != num_freqs:
            raise ModeSourceSolveError(
                "num_freqs",
                "source bandwidth is too narrow to resolve distinct mode "
                "frequencies at this carrier; increase bandwidth or use one "
                "frequency sample",
            )
        solve_frequencies = sorted(set(frequency_samples + [carrier_hz]))
        solved = solve_yee_mode_bank(
            sim, axis, position_um, solve_frequencies,
            polarization, mode_index, **solve_kwargs,
        )
        mode = solved[carrier_hz]
        modes_by_freq = {f: solved[f] for f in frequency_samples}

    # Keep the Workbench document compact: use the auxiliary ModeSource rather
    # than expanding a full Huygens sheet into thousands of PointDipoles.  The
    # solved profile is always normalized at 1 W; amplitude stays the explicit
    # top-level multiplier it was before re-solving.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        solved_sources = mode_launch(
            sim, mode, axis=axis, position_um=position_um,
            source_time=pulse, direction=direction, power_watts=1.0,
            center_um=center_um, thickness_axis=v_axis,
            modes_by_freq=modes_by_freq, launch="aux",
        )
    if len(solved_sources) != 1 or not isinstance(solved_sources[0], ModeSource):
        raise RuntimeError("auxiliary mode solve did not produce one ModeSource")
    solved_source = solved_sources[0]
    source_data = solved_source.model_dump(mode="python")
    source_data["amplitude"] = current.amplitude
    solved_source = ModeSource.model_validate(source_data)

    sources = list(sim.sources)
    sources[source_index] = solved_source
    # Build the provenance before cross-validating the whole Simulation. A
    # source-linked modal port intentionally rejects a ModeSource whose solve
    # provenance is absent, so validating this short-lived intermediate source
    # would fail even though the very next operation attaches that provenance.
    interim = sim.model_copy(update={"sources": tuple(sources)})
    recipe_data = {
        "solver": "yee",
        "polarization": polarization,
        "mode_index": mode_index,
        "wavelength_um": wavelength_um,
        "center_um": center_um,
        "size_um": size_um,
        "dl_um": dl_um,
        "supersample": supersample,
        "num_modes": num_modes,
        "num_freqs": num_freqs,
    }
    recipe = ModeSolveProvenance.model_validate({
        **recipe_data,
        "input_sha256": mode_source_input_sha256(
            interim, source_index, recipe_data),
    })
    source_data = solved_source.model_dump(mode="python")
    source_data["mode_solve"] = recipe
    solved_source = ModeSource.model_validate(source_data)
    sources[source_index] = solved_source

    # Preserve a driven-port binding when the re-solve is still on the same
    # plane axis. Direction and launched channel are coupled constraints, so
    # update them atomically with the source. If the user changes axis, keep
    # the physical port plane untouched and explicitly unbind it; silently
    # rotating a measurement plane would be a much more surprising edit.
    from ..components.monitors import (
        FieldDftMonitor,
        mode_port_physical_polarization,
        mode_port_required_trial_modes,
    )

    rebound_ports = []
    unlinked_ports = []
    monitors = []
    for monitor in sim.monitors:
        if (not isinstance(monitor, FieldDftMonitor)
                or monitor.mode_port is None
                or monitor.mode_port.source_index != source_index):
            monitors.append(monitor)
            continue
        zero_axes = [
            index for index, size in enumerate(monitor.size_um)
            if float(size) == 0.0
        ]
        port_axis = _SPATIAL[zero_axes[0]] if len(zero_axes) == 1 else None
        downstream = (
            (float(monitor.center_um[_SPATIAL.index(axis)]) - position_um)
            * (1.0 if direction == "+" else -1.0)
            if port_axis == axis else -1.0
        )
        if port_axis != axis or downstream <= 1e-12:
            port = monitor.mode_port.model_copy(update={"source_index": None})
            monitors.append(monitor.model_copy(update={"mode_port": port}))
            unlinked_ports.append(monitor.name)
            continue
        launched = (
            mode_port_physical_polarization(
                polarization, axis, monitor.mode_port.thickness_axis),
            mode_index,
        )
        modes = list(monitor.mode_port.modes)
        if not any((mode.polarization, int(mode.mode_index)) == launched
                   for mode in modes):
            from ..components.monitors import PortMode
            modes.insert(0, PortMode(
                polarization=launched[0], mode_index=mode_index))
        minimum_modes = mode_port_required_trial_modes(modes)
        trial_modes = monitor.mode_port.num_modes
        if trial_modes is not None:
            trial_modes = max(int(trial_modes), minimum_modes)
        port = monitor.mode_port.model_copy(update={
            "out_direction": "-" if direction == "+" else "+",
            "modes": tuple(modes),
            "num_modes": trial_modes,
        })
        monitors.append(monitor.model_copy(update={"mode_port": port}))
        rebound_ports.append(monitor.name)

    updated = sim._validated_copy({
        "sources": tuple(sources),
        "monitors": tuple(monitors),
    })
    summary = {
        "source_index": source_index,
        "solver": "yee",
        "polarization": polarization,
        "mode_index": mode_index,
        "wavelength_um": wavelength_um,
        "n_eff": float(mode.n_eff),
        "te_fraction": float(mode.te_fraction),
        "profile_shape": [int(solved_source.nv), int(solved_source.nu)],
        "mode_window_shape": [int(mode.shape[0]), int(mode.shape[1])],
        "frequency_samples": len(frequency_samples),
        "frequency_samples_hz": frequency_samples,
        "has_minor": solved_source.profile_minor is not None,
        "has_true_h": solved_source.profile_h is not None,
    }
    if rebound_ports:
        summary["rebound_ports"] = rebound_ports
    if unlinked_ports:
        summary["unlinked_ports"] = unlinked_ports
    return updated, summary


def append_mode_source(sim, settings: dict, seed: dict):
    """Solve and append a guided source without publishing a dummy profile.

    A transient 1x1 ModeSource exists only in this immutable local copy so the
    established solve path can preserve amplitude/source-time semantics.  The
    caller receives either a fully solved canonical source or an exception;
    the input simulation is never mutated and no placeholder can leak into the
    Workbench workspace.
    """
    from ..components.source_time import GaussianPulse
    from ..components.sources import ModeSource
    from . import _geometry as geom

    if not isinstance(settings, dict):
        raise ModeSourceSolveError("settings", "mode solve settings must be an object")
    if not isinstance(seed, dict):
        raise ModeSourceSolveError(
            "seed", "append requires a seed object with amplitude and source_time")
    axis = str(settings.get("axis", "x")).lower()
    if axis not in ("x", "y", "z"):
        raise ModeSourceSolveError("axis", "axis must be x, y, or z")
    direction = str(settings.get("direction", "+"))
    if direction not in ("+", "-"):
        raise ModeSourceSolveError("direction", "direction must be '+' or '-'")
    try:
        position_um = float(settings.get(
            "position_um", sim._realized_um()["xyz".index(axis)] / 2.0))
        amplitude = float(seed.get("amplitude", 1.0))
    except (TypeError, ValueError):
        raise ModeSourceSolveError(
            "seed", "seed amplitude and source position must be numbers") from None
    if not math.isfinite(position_um):
        raise ModeSourceSolveError("position_um", "position_um must be finite")
    if not math.isfinite(amplitude):
        raise ModeSourceSolveError("seed.amplitude", "amplitude must be finite")
    pulse_data = seed.get("source_time")
    if hasattr(pulse_data, "model_dump"):
        pulse_data = pulse_data.model_dump(mode="python")
    if not isinstance(pulse_data, dict):
        raise ModeSourceSolveError(
            "seed.source_time", "seed.source_time must be a GaussianPulse object")
    try:
        pulse = GaussianPulse.model_validate(pulse_data)
    except ValueError as exc:
        raise ModeSourceSolveError(
            "seed.source_time", f"invalid GaussianPulse seed: {exc}") from exc

    h_axis, _ = geom.in_plane_axes(axis)
    try:
        transient = ModeSource(
            axis=axis, direction=direction, position_um=position_um,
            polarization="E" + h_axis, amplitude=amplitude,
            n_eff=1.0, nu=1, nv=1, profile=(1.0,), source_time=pulse,
        )
    except ValueError as exc:
        raise ModeSourceSolveError(
            "seed.source_time", f"invalid mode-source seed: {exc}") from exc
    appended_index = len(sim.sources)
    local = sim._validated_copy({"sources": tuple(sim.sources) + (transient,)})
    return solve_mode_source(local, appended_index, settings)


_CPU_STARTER = {
    "id": "soi-waveguide-cpu-quickstart",
    "title": "SOI waveguide CPU quickstart",
    "profile": "First run · CPU and GPU",
    "description": (
        "A compact silicon-on-insulator waveguide with field, flux, and time "
        "monitors. It is deliberately small enough for a first local CPU run."
    ),
    "provenance": "PhotonHub built-in beta quickstart",
    "fidelity_note": (
        "A workflow smoke test, not a benchmark or calibrated device result — "
        "verify the edit → run → results → save loop before spending cloud "
        "credits."
    ),
    # No reference row: the estimate bar already reports the planned size.
    "reference": "",
}

_FRESNEL_SLAB_RESOURCE = ("examples", "fresnel_slab.sim.json")
_FRESNEL_SLAB_STARTER = {
    "id": "fresnel-slab-tmm-cpu",
    "title": "Fresnel slab vs analytic theory",
    "profile": "CPU example · seconds",
    "description": (
        "A 0.25 µm ε=4 dielectric slab under a normal-incidence broadband "
        "plane wave, with upstream/downstream flux planes for reflectance "
        "and transmittance — the classic textbook interference spectrum."
    ),
    "provenance": (
        "PhotonHub notebook gallery scene — examples/notebooks/"
        "02_fresnel_slab.ipynb and 10_cloud_gpu_run.ipynb"
    ),
    "fidelity_note": (
        "5,120 cells, seconds on any laptop CPU. The transverse span is four "
        "periodic cells, so the 3D view shows a thin pillar; the physics is "
        "the laterally infinite slab. R = (P0_up − P_up)/P0_up needs the "
        "empty-domain reference run; T's downstream shape is directly "
        "comparable."
    ),
    "reference": (
        "Analytic transfer-matrix across the band: T spans 0.6446–0.7753, "
        "R spans 0.2247–0.3554, R + T = 1 (lossless). Notebook 02 reproduces "
        "both to ~5e-4 at this grid."
    ),
}

_MODE_CONVERTER_RESOURCE = ("examples", "mode_converter.sim.json")
_MODE_CONVERTER_STARTER = {
    "id": "gds-mode-converter-matched-res10",
    "title": "TE0→TE1 GDS mode converter",
    "profile": "GPU example · matched res10",
    "description": (
        "Exact gdsfactory generic-PDK mode-converter geometry: a 0.5 µm TE0 "
        "input couples across a 0.15 µm gap into the TE1 mode of a 1.2 µm bus."
    ),
    "provenance": (
        "JPPhotonics/fdtd-pipeline@622e0a9 · Liu & Poon, arXiv:2506.16665"
    ),
    "fidelity_note": (
        "Uses the benchmark's matched res10 materials, uniform grid, 1500–1600 nm "
        "sweep, true-H guided-mode profile, and port planes. To keep the editor "
        "interactive it uses one auxiliary ModeSource instead of thousands of "
        "equivalence-current dipoles. Four physical modal ports are ready in "
        "Design and Results; o3 resolves both TE0 and TE1 from one raw DFT plane."
    ),
    "reference": (
        "Matched res25 headline at 1550 nm: PhotonHub 45.74% vs Tidy3D 45.69% "
        "TE0→TE1 conversion (+0.05 percentage points)."
    ),
}


def default_starter() -> dict:
    """User-facing provenance for the desktop ``New`` starter.

    Return a copy so API state can retain or discard the metadata without ever
    mutating the module-level source of truth.
    """
    return dict(_CPU_STARTER)


def example_starters() -> list[dict]:
    """Packaged examples that users may opt into after the CPU quickstart.

    CPU-runnable entries come first: the beta admits every user to free local
    CPU, while the GPU-oriented scenes assume metered cloud credit."""
    return [dict(_FRESNEL_SLAB_STARTER), dict(_MODE_CONVERTER_STARTER)]


@lru_cache(maxsize=1)
def _mode_converter_sim_resource():
    """Parse the packaged, GPU-oriented interactive benchmark once.

    The canonical matched-res25 input expands its equivalence-current launch
    into nearly 30,000 sources, which is appropriate for the benchmark runner
    but unusable as a structured GUI document.  This generated resource is the
    repo's matched-res10 scene with ``USE_EQ_SOURCE=False``: the same GDS,
    materials, grid, band, true-H mode, and monitor planes in one ModeSource.
    """
    from ..components.simulation import Simulation

    resource = resources.files(__package__).joinpath(*_MODE_CONVERTER_RESOURCE)
    return Simulation.from_wire_json(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _fresnel_slab_sim_resource():
    """Parse the packaged CPU-quick Fresnel-slab benchmark scene once."""
    from ..components.simulation import Simulation

    resource = resources.files(__package__).joinpath(*_FRESNEL_SLAB_RESOURCE)
    return Simulation.from_wire_json(resource.read_text(encoding="utf-8"))


def example_sim(example_id: str):
    """Return a packaged opt-in Workbench example by stable id."""
    if example_id == _FRESNEL_SLAB_STARTER["id"]:
        return _fresnel_slab_sim_resource(), dict(_FRESNEL_SLAB_STARTER)
    if example_id == _MODE_CONVERTER_STARTER["id"]:
        return _mode_converter_sim_resource(), dict(_MODE_CONVERTER_STARTER)
    raise KeyError(f"unknown Workbench example: {example_id}")


def default_sim():
    """A small, runnable SOI waveguide starter for desktop ``New``.

    The default must stay a CPU-friendly first-run check. Larger, scientifically
    grounded scenes are available through :func:`example_sim`, but silently
    making one of them the first document would turn a successful install into
    a multi-hour CPU run on typical beta laptops.
    """
    from ..components.simulation import SCHEMA_VERSION

    spec = {
        "schema_version": SCHEMA_VERSION,
        "size_um": [4.0, 2.0, 1.2],
        "grid": {"type": "uniform", "dl_um": 0.05},
        # Covers the pulse offset/tail plus conservative SOI group delay to the
        # output monitor. A shorter 400-step run ends while still injecting.
        "run": {"n_steps": 1000, "courant": 0.99, "shutoff": 1.0e-5},
        "background": {"permittivity": 1.0},
        # The compact 24-cell z span needs an 8-layer slab on each face;
        # schema-default 12+12 would touch and fail the engine overlap gate.
        "pml_num_layers": 8,
        "subpixel": True,
        "subpixel_method": "contour",
        "structures": [{
            "geometry": {
                "type": "box", "center_um": [2.0, 1.0, 0.6],
                "size_um": [4.0, 0.45, 0.22],
            },
            "medium": {"permittivity": 12.25, "conductivity_s_per_m": 0.0},
        }],
        "boundaries": {"x": "pml", "y": "pml", "z": "pml"},
        "sources": [{
            "type": "point_dipole", "center_um": [0.7, 1.0, 0.6],
            "polarization": "Ey", "amplitude": 1.0,
            "source_time": {
                "type": "gaussian_pulse", "freq0_hz": 193.4e12,
                "fwidth_hz": 40.0e12, "offset": 5.0, "phase": 0.0,
            },
        }],
        "monitors": [
            {
                "type": "field_dft", "name": "field",
                "center_um": [2.0, 1.0, 0.6], "size_um": [4.0, 2.0, 0.0],
                "fields": ["Ex", "Ey", "Ez"],
                "freqs_hz": [170.0e12, 193.4e12, 210.0e12],
            },
            {
                "type": "flux", "name": "output_flux", "axis": "x",
                "position_um": 3.5,
                "freqs_hz": [170.0e12, 193.4e12, 210.0e12],
            },
            {
                "type": "field_time", "name": "probe",
                "center_um": [3.2, 1.0, 0.6],
                "fields": ["Ex", "Ey", "Ez"], "interval_steps": 1,
            },
        ],
    }
    return parse_sim_spec(spec)[0]


def sim_payload(sim) -> dict:
    """Canonical editable spec plus an engine-faithful pre-run estimate."""
    estimate = sim.cost_estimate()
    return {
        "spec": sim.to_wire_dict(),
        "estimate": asdict(estimate),
        "mode_source_statuses": mode_source_statuses(sim),
        **sim_overview(sim),
    }


def sim_overview(sim) -> dict:
    """Preview metadata for a spec (no results): per-axis domain bounds (µm, for the
    ε slider) and the structures/monitors/sources it defines."""
    domain: dict[str, Any] = {}
    try:
        from .eps import axis_nodes_um
        for i, ax in enumerate(_SPATIAL):
            n = np.asarray(axis_nodes_um(sim, i), dtype=float)
            domain[ax] = {"min": float(n[0]), "max": float(n[-1])}
    except Exception:
        domain = {}

    def _named(items, prefix: str) -> list[dict]:
        # every monitor/source carries a `type` Literal; type(it).__name__ is just a
        # defensive fallback for an odd spec.
        return [{"name": str(getattr(it, "name", "") or f"{prefix} {i + 1}"),
                 "type": str(getattr(it, "type", None) or type(it).__name__)}
                for i, it in enumerate(items or [])]

    return {
        "domain_um": domain,
        "structures": len(getattr(sim, "structures", []) or []),
        "monitors": _named(getattr(sim, "monitors", []), "Monitor"),
        "sources": _named(getattr(sim, "sources", []), "Source"),
    }


# ---------------------------------------------------------------------------
# GDS layout import (workbench Structures editor)
# ---------------------------------------------------------------------------

# Decoded upload ceiling. The benchmark layouts are tens of KB; even dense
# foundry exports stay far below this. The cap exists so a mistaken pick of a
# multi-GB file fails fast instead of being buffered into memory.
GDS_MAX_BYTES = 64 * 1024 * 1024


class GdsImportError(ValueError):
    """A user-correctable GDS import problem, tagged for the HTTP facade."""

    def __init__(self, message: str, code: str = "gds_invalid"):
        super().__init__(message)
        self.code = code


def _gds_module():
    from .. import gds
    try:
        gds._import_gdstk()
    except ImportError as exc:
        raise GdsImportError(str(exc), code="gds_reader_missing") from exc
    return gds


def _gds_read_cell(gds, gds_path: str, cell_name: Optional[str]):
    """Read + flatten the requested cell; map reader errors to GdsImportError.

    Unlike :func:`photonhub.gds.import_gds`, a file with several top-level cells
    is not an error here: inspection defaults to the first top-level cell and
    returns the full cell list so the dialog can offer the choice. The strict
    ambiguity check still applies at import time, where the dialog always
    passes its selected cell explicitly.
    """
    gdstk = gds._import_gdstk()
    try:
        lib = gdstk.read_gds(gds_path, unit=1e-6)
    except Exception as exc:
        raise GdsImportError(f"not a readable GDSII file: {exc}") from exc
    tops = {c.name for c in lib.top_level()}
    cells = [c.name for c in lib.cells if c.name in tops]
    cells += [c.name for c in lib.cells if c.name not in tops]
    if not cells:
        raise GdsImportError("the GDS file contains no cells")
    chosen = cells[0] if cell_name is None else cell_name
    for cell in lib.cells:
        if cell.name == chosen:
            return cell.copy(cell.name + "__phinspect").flatten(), chosen, cells
    have = ", ".join(repr(name) for name in cells)
    raise GdsImportError(
        f"cell {chosen!r} not found; cells: {have}", code="gds_cell_not_found")


def gds_inspect(gds_path: str, cell_name: Optional[str] = None) -> dict:
    """Catalog a GDS file for the import dialog: cells and per-layer stats.

    Coordinates are reported in microns on the file's drawing plane (the two
    transverse axes of the eventual extrusion axis). ``bbox_um`` entries are
    ``[[u_min, v_min], [u_max, v_max]]``; the top-level bbox is the union over
    all listed layers, or ``None`` for a cell with no polygons.
    """
    gds = _gds_module()
    cell, chosen, cells = _gds_read_cell(gds, gds_path, cell_name)

    per_layer: dict[tuple, dict] = {}
    for poly in cell.polygons:
        entry = per_layer.setdefault((poly.layer, poly.datatype), {
            "polygons": 0, "vertices": 0,
            "lo": [math.inf, math.inf], "hi": [-math.inf, -math.inf],
        })
        pts = np.asarray(poly.points, dtype=float)
        entry["polygons"] += 1
        entry["vertices"] += int(pts.shape[0])
        entry["lo"] = [min(entry["lo"][i], float(pts[:, i].min())) for i in range(2)]
        entry["hi"] = [max(entry["hi"][i], float(pts[:, i].max())) for i in range(2)]

    layers = [{
        "layer": [int(layer), int(datatype)],
        "polygons": entry["polygons"],
        "vertices": entry["vertices"],
        "bbox_um": [entry["lo"], entry["hi"]],
    } for (layer, datatype), entry in sorted(per_layer.items())]

    bbox = None
    if layers:
        bbox = [
            [min(l["bbox_um"][0][i] for l in layers) for i in range(2)],
            [max(l["bbox_um"][1][i] for l in layers) for i in range(2)],
        ]
    return {"cells": cells, "cell": chosen, "layers": layers, "bbox_um": bbox}


def gds_import_structures(
    gds_path: str,
    layers: list,
    *,
    cell_name: Optional[str] = None,
    axis: str = "z",
    offset_um=(0.0, 0.0),
    min_area_um2: float = 0.0,
    name_prefix: str = "gds",
) -> dict:
    """Convert selected GDS layers into editable structure wire dicts.

    ``layers`` entries are dicts mirroring :class:`photonhub.gds.GdsLayer` plus a
    medium: ``{"layer": [l, d], "zmin_um", "thickness_um", "permittivity",
    "conductivity_s_per_m"?, "sidewall_angle"? (rad), "reference_plane"?}``.
    ``offset_um`` translates every polygon on the drawing plane (u, v) — the
    dialog uses it to center a layout inside the domain. The result's
    ``structures`` are wire-schema dicts ready to append to a simulation spec.
    """
    from ..components.structures import Medium

    gds = _gds_module()
    if axis not in ("x", "y", "z"):
        raise GdsImportError(f"axis must be one of x/y/z, got {axis!r}")
    if not isinstance(layers, list) or not layers:
        raise GdsImportError("select at least one GDS layer to import")
    try:
        du, dv = (float(offset_um[0]), float(offset_um[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise GdsImportError("offset_um must be a [du, dv] pair of numbers") from exc
    if not (math.isfinite(du) and math.isfinite(dv)):
        raise GdsImportError("offset_um must be finite")
    prefix = str(name_prefix or "gds").strip() or "gds"

    def _layer_spec(index: int, raw) -> "gds.GdsLayer":
        if not isinstance(raw, dict):
            raise GdsImportError(f"layer {index + 1} must be an object")
        pair = raw.get("layer")
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(v, (int, float)) and float(v).is_integer()
                           for v in pair)):
            raise GdsImportError(
                f"layer {index + 1} needs a [layer, datatype] integer pair")
        try:
            permittivity = float(raw.get("permittivity", 1.0))
            conductivity = float(raw.get("conductivity_s_per_m", 0.0))
            zmin = float(raw.get("zmin_um", 0.0))
            thickness = float(raw.get("thickness_um", 0.0))
            sidewall = float(raw.get("sidewall_angle", 0.0))
        except (TypeError, ValueError) as exc:
            raise GdsImportError(f"layer {index + 1}: {exc}") from exc
        if not all(map(math.isfinite, (permittivity, conductivity, zmin,
                                       thickness, sidewall))):
            raise GdsImportError(f"layer {index + 1} contains a non-finite value")
        if permittivity < 1.0:
            raise GdsImportError(
                f"layer {index + 1} permittivity must be >= 1 (vacuum)")
        if conductivity < 0.0:
            raise GdsImportError(f"layer {index + 1} conductivity must be >= 0")
        if thickness <= 0.0:
            raise GdsImportError(f"layer {index + 1} thickness must be > 0 µm")
        reference_plane = raw.get("reference_plane", "middle")
        if reference_plane not in ("bottom", "middle", "top"):
            raise GdsImportError(
                f"layer {index + 1} reference plane must be bottom/middle/top")
        return gds.GdsLayer(
            layer=(int(pair[0]), int(pair[1])),
            medium=Medium(permittivity=permittivity,
                          conductivity_s_per_m=conductivity),
            zmin_um=zmin, thickness_um=thickness,
            sidewall_angle=sidewall, reference_plane=reference_plane,
        )

    specs = [_layer_spec(i, raw) for i, raw in enumerate(layers)]

    # Validate readability + cell choice up front so a wrong cell name reports
    # gds_cell_not_found (import_gds's own ValueError is indistinguishable
    # from other invalid-input failures).
    _gds_read_cell(gds, gds_path, cell_name)

    structures: list[dict] = []
    per_layer: list[dict] = []
    for spec in specs:
        # One import_gds call per layer keeps the polygon -> layer attribution
        # exact even when min_area filtering drops slivers.
        try:
            imported = gds.import_gds(
                gds_path, [spec], cell_name=cell_name, axis=axis,
                min_area_um2=min_area_um2)
        except (ValueError, OSError) as exc:
            # gdstk reports malformed GDSII as a bare OSError; FileNotFoundError
            # is its subclass and shares the user-facing mapping.
            raise GdsImportError(str(exc) or "not a readable GDSII file") from exc
        layer_tag = f"L{spec.layer[0]}_{spec.layer[1]}"
        dumped = []
        for i, structure in enumerate(imported):
            data = structure.model_dump(
                mode="json", by_alias=True, exclude_none=True)
            data["name"] = (f"{prefix}_{layer_tag}" if len(imported) == 1
                            else f"{prefix}_{layer_tag}_p{i + 1}")
            if du or dv:
                data["geometry"]["vertices_um"] = [
                    [u + du, v + dv]
                    for u, v in data["geometry"]["vertices_um"]]
            dumped.append(data)
        structures.extend(dumped)
        per_layer.append({"layer": [spec.layer[0], spec.layer[1]],
                          "count": len(dumped)})

    if not structures:
        raise GdsImportError(
            "the selected layers contain no polygons in this cell",
            code="gds_empty")
    return {"structures": structures, "per_layer": per_layer,
            "count": len(structures)}
