"""Load a phsolver output directory into xarray.

Output contract (NUMERICS.md section 6, engine/include/phcore/output.h): one
raw little-endian float32 ``.bin`` per monitor plus ``manifest.json``::

    {
      "manifest_version": "1",          # OUTPUT contract version (gate here)
      "schema_version": "1.1.0-alpha.1",  # echo of the INPUT spec version
      "monitors": [
        {"name": "probe", "type": "field_time", "file": "probe.bin",
         "dtype": "float32", "shape": [n_samples, n_components],
         "dims": ["sample", "component"], "components": ["Ez"],
         "sample_steps": [1, 2, 3], "dt_s": 1.234e-17},
        {"name": "final", "type": "field_snapshot", "file": "final.bin",
         "dtype": "float32", "shape": [n_samples, n_comp, nz, ny, nx],
         "dims": ["sample", "component", "z", "y", "x"],
         "components": ["Ex", "Ez"], "sample_steps": [1600],
         "dt_s": 1.234e-17},
        {"name": "slab", "type": "field_dft", "file": "slab.bin",
         "dtype": "float32", "shape": [n_freqs, n_comp, nz, ny, nx, 2],
         "dims": ["freq", "component", "z", "y", "x", "complex"],
         "components": ["Ex", "Hy"], "freqs_hz": [1.934e14],
         "origin_cells": [i0, j0, k0],   # optional; region low corner (x,y,z)
         "dt_s": 1.234e-17},
        {"name": "reflection", "type": "flux", "file": "reflection.bin",
         "dtype": "float32", "shape": [n_freqs], "dims": ["freq"],
         "axis": "z", "freqs_hz": [1.784e14, 1.934e14], "dt_s": 1.234e-17}
      ],
      "run": {"n_steps": 1000, "steps_run": 1000, "dt_s": 1.234e-17,
              "wall_seconds": 1.2, "mcells_per_s": 800.0, "aborted": false,
              "abort_reason": "", "shut_off": false},
      "grid": {"shape": [nx, ny, nz], "dl_um": 0.05, "size_um": [...]},
      "provenance": {"solver_version": "...", "device_name": "...", ...}
    }

Frequency-domain monitors (NUMERICS.md section 12) are emitted as float32
``[re, im]`` pairs in binary order ``[freq][component][k][j][i][re,im]``
(de-pitched) and reconstructed here as ``complex64`` DataArrays with dims
``('f', 'component', 'z', 'y', 'x')``; flux monitors are one float32 power
per frequency, dims ``('f',)``, positive toward +axis. The engine writes both
with the section-12 normalization, phasors divided by ``A0 * S(f)`` (the
first wire-order source's amplitude times its unit-amplitude analytic
spectrum), flux therefore scaled by ``1/|A0*S(f)|^2``. This reader multiplies
``A0`` back in (``field_dft`` by ``A0``, ``flux`` by ``A0^2``) whenever it
knows the simulation, handed over by the runner, or read from the
``sim.json`` beside the outputs or inside the HDF5 bundle, so a
frequency-domain array is the continuous-wave response to the sources as
declared: fields in V/m and A/m, flux in watts. Without the simulation the
engine's unit-amplitude values come back unchanged. Either way the
``normalization`` attr says which convention the array carries and
``norm_amplitude`` holds ``A0`` (None when unknown). On a domain folded by
§20 symmetry planes ``data[name]`` reports a flux as the whole device's
power, the modeled part times 2 for every plane that cuts the monitor plane
(NUMERICS §20.8, attr ``symmetry_factor``); ``data.wire(name)`` keeps the
modeled part.
field_dft coordinates are base-node positions with
NO per-component Yee half-cell destaggering applied, see the warning on
``_load_field_dft`` before hand-computing E x H* flux from them.

Aborted runs (NUMERICS.md section 7: ``divergence`` / ``non_finite_energy``)
still write a complete manifest with truncated monitor data; loading one
emits a ``UserWarning`` and surfaces ``aborted`` / ``abort_reason`` here and
in every DataArray's attrs, so partial fields are never silently presented
as healthy data. (``run_local`` raises before loading; the direct
``RunResult(path)`` post-mortem path warns instead so diverged runs
stay inspectable.)

Snapshot binaries are de-pitched (rows of exactly nx), sample-major order
``[sample][component][k][j][i]``. ``sample_steps`` stores ``step = s + 1``,
so E-field sample times are ``step * dt_s`` (H lags by ``dt_s / 2``; raw
non-colocated Yee values).
"""

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import xarray as xr

from .components.base import (
    _MONITOR_NAME_MAX_LENGTH,
    _is_portable_filename_token,
    _monitor_name_key,
)
from .constants import c0

_TIME_DIMS = ("t", "component")
_SNAPSHOT_DIMS = ("t", "component", "z", "y", "x")
# Free-space speed of light (m/s), identical to the engine's kC0, so the
# ``wlen_um`` coordinate round-trips with ``GaussianPulse.for_band``.
_C0_M_PER_S = c0


def _wlen_um(freqs_hz) -> np.ndarray:
    """Free-space wavelength (microns) for each frequency: the second
    coordinate every frequency-domain array carries beside ``f``."""
    return _C0_M_PER_S / np.asarray(freqs_hz, dtype=np.float64) * 1e6


_DFT_DIMS = ("f", "component", "z", "y", "x")
_FLUX_DIMS = ("f",)
_MAX_SHAPE_DIM = (1 << 31) - 1
_RESERVED_RESULT_FILES = {"sim.json", "manifest.json", "solver-events.jsonl",
                          "client.json"}

# NUMERICS.md section 12 normalization, surfaced on every frequency-domain
# DataArray so absolute-magnitude use is never silent. NOTE for apodized
# monitors (ProfileMonitor(apodization=...), §12): the phasors are a WINDOWED
# DFT but are still normalized by the FULL-pulse A0*S(f) — the window is not
# divided out — so their absolute magnitudes are NOT comparable to an
# unapodized monitor's (or to another window's); only ratios within the same
# apodization are.
_DFT_NORMALIZATION = (
    "phasors normalized by A0*S(f): the first wire-order source's amplitude "
    "times the unit-amplitude analytic pulse spectrum (NUMERICS.md section "
    "12, e^{-i omega t} convention) — the response to a UNIT-amplitude "
    "harmonic drive, not restored to the declared amplitude because the "
    "simulation was not available to the reader; an apodized monitor's "
    "windowed-DFT phasors keep this full-pulse normalization, so their "
    "absolute magnitudes are not comparable to unapodized monitors")
_DFT_ABSOLUTE = (
    "continuous-wave phasors of the sources as declared, in V/m and A/m "
    "(e^{-i omega t}): the engine's A0*S(f)-normalized phasors (NUMERICS.md "
    "section 12) multiplied back by A0, the first wire-order source's "
    "amplitude (attr norm_amplitude); an apodized monitor's windowed-DFT "
    "phasors are not comparable in magnitude to unapodized monitors")
_FLUX_NORMALIZATION = (
    "power normalized by 1/|A0*S(f)|^2 (shared normalized phasors, "
    "NUMERICS.md section 12) — the response to a UNIT-amplitude harmonic "
    "drive, NOT absolute watts because the simulation was not available to "
    "the reader; positive values flow toward +axis")
_FLUX_ABSOLUTE = (
    "time-averaged power in watts for the sources as declared: the engine's "
    "1/|A0*S(f)|^2-normalized flux (NUMERICS.md section 12) multiplied back "
    "by A0^2, the first wire-order source's amplitude squared (attr "
    "norm_amplitude); positive values flow toward +axis")


def validate_monitor_manifest_entry(
    entry: dict, manifest: Optional[dict] = None, *,
    require_explicit_dims: bool = False,
) -> int:
    """Validate one v1 monitor contract without opening its numerical blob.

    The durable run ledger calls this same validator before it publishes a
    completed seal.  Keeping the shape/dimension contract here prevents a run
    from being marked completed only to fail the reader on first selection.
    The returned value is the declared number of raw float32 values.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"monitor manifest entry must be an object: {entry!r}")
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"monitor manifest entry without a name: {entry!r}")
    dtype = entry.get("dtype", "float32")
    if dtype != "float32":
        raise ValueError(f"monitor {name!r}: unsupported dtype {dtype!r}")
    raw_shape = entry.get("shape")
    if not isinstance(raw_shape, list):
        raise ValueError(f"monitor {name!r}: shape must be a list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           or value > _MAX_SHAPE_DIM
           for value in raw_shape):
        raise ValueError(f"monitor {name!r}: invalid shape {raw_shape!r}")
    shape = tuple(raw_shape)
    kind = entry.get("type")
    components = entry.get("components", [] if kind == "flux" else None)
    if not isinstance(components, list):
        raise ValueError(f"monitor {name!r}: components must be a list")
    allowed_components = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    if kind != "flux" and (
        not components or any(not isinstance(value, str)
                              or value not in allowed_components
                              for value in components)
    ):
        raise ValueError(
            f"monitor {name!r}: components must be non-empty field labels")
    if len(set(components)) != len(components):
        raise ValueError(f"monitor {name!r}: components must be unique")
    manifest_dims = entry.get("dims")
    if manifest_dims is not None and not isinstance(manifest_dims, list):
        raise ValueError(f"monitor {name!r}: dims must be a list")

    if kind in ("field_time", "field_snapshot"):
        expected_dims = (_TIME_DIMS if kind == "field_time" else _SNAPSHOT_DIMS)
        expected_wire_dims = ["sample", *expected_dims[1:]]
        if manifest_dims is None:
            if require_explicit_dims:
                raise ValueError(
                    f"monitor {name!r}: completed engine manifest must declare dims")
            manifest_dims = expected_wire_dims
        else:
            normalized = ("t",) + tuple(manifest_dims[1:])
            if tuple(manifest_dims[:1]) != ("sample",) or normalized != expected_dims:
                raise ValueError(
                    f"monitor {name!r}: manifest dims {manifest_dims} != expected "
                    f"{expected_wire_dims}")
        if len(shape) != len(expected_dims):
            raise ValueError(
                f"monitor {name!r}: shape {shape} has {len(shape)} dims, "
                f"expected {len(expected_dims)}")
        sample_steps = entry.get("sample_steps")
        if not isinstance(sample_steps, list):
            raise ValueError(f"monitor {name!r}: sample_steps must be a list")
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value < 0 or value > _MAX_SHAPE_DIM for value in sample_steps):
            raise ValueError(
                f"monitor {name!r}: sample_steps must be non-negative 32-bit integers")
        if any(current <= previous
               for previous, current in zip(sample_steps, sample_steps[1:])):
            raise ValueError(
                f"monitor {name!r}: sample_steps must be strictly increasing")
        if shape[0] != len(sample_steps) or shape[1] != len(components):
            raise ValueError(
                f"monitor {name!r}: shape {shape} inconsistent with "
                f"{len(sample_steps)} samples x {len(components)} components")
    elif kind == "field_dft":
        if len(shape) != 6 or shape[-1] != 2:
            raise ValueError(
                f"monitor {name!r}: field_dft shape {shape} must be "
                "[freq, component, z, y, x, 2]")
        if any(value < 1 for value in shape[2:5]):
            raise ValueError(
                f"monitor {name!r}: field_dft spatial extents must be positive")
        expected_wire_dims = [
            "freq", "component", "z", "y", "x", "complex"]
        if manifest_dims is None:
            if require_explicit_dims:
                raise ValueError(
                    f"monitor {name!r}: completed engine manifest must declare dims")
            manifest_dims = expected_wire_dims
        else:
            normalized = ("f",) + tuple(manifest_dims[1:-1])
            if (tuple(manifest_dims[:1]) != ("freq",)
                    or tuple(manifest_dims[-1:]) != ("complex",)
                    or normalized != _DFT_DIMS):
                raise ValueError(
                    f"monitor {name!r}: manifest dims {manifest_dims} != "
                    "expected ['freq', 'component', 'z', 'y', 'x', 'complex']")
        freqs = entry.get("freqs_hz")
        if not isinstance(freqs, list) or not freqs:
            raise ValueError(
                f"monitor {name!r}: frequency-domain manifest entry has no "
                "'freqs_hz'")
        try:
            numeric_freqs = [float(value) for value in freqs]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"monitor {name!r}: freqs_hz must contain numbers") from exc
        if any(not math.isfinite(value) or value <= 0 for value in numeric_freqs):
            raise ValueError(
                f"monitor {name!r}: freqs_hz must contain finite positive numbers")
        if len(set(numeric_freqs)) != len(numeric_freqs):
            raise ValueError(f"monitor {name!r}: freqs_hz must be unique")
        if shape[0] != len(freqs) or shape[1] != len(components):
            raise ValueError(
                f"monitor {name!r}: shape {shape} inconsistent with "
                f"{len(freqs)} freqs x {len(components)} components")
        origin = entry.get("origin_cells", (0, 0, 0))
        stride = entry.get("interval_space", (1, 1, 1))
        if not isinstance(origin, (list, tuple)) or len(origin) != 3:
            raise ValueError(
                f"monitor {name!r}: origin_cells {origin} must be [i0, j0, k0]")
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value < 0 for value in origin):
            raise ValueError(
                f"monitor {name!r}: origin_cells must be non-negative integers")
        if (not isinstance(stride, (list, tuple)) or len(stride) != 3
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1 for value in stride)):
            raise ValueError(
                f"monitor {name!r}: interval_space {stride} must contain three "
                "positive integer strides")
    elif kind == "flux":
        if len(shape) != 1:
            raise ValueError(f"monitor {name!r}: flux shape {shape} must be [freq]")
        if manifest_dims is None:
            if require_explicit_dims:
                raise ValueError(
                    f"monitor {name!r}: completed engine manifest must declare dims")
            manifest_dims = ["freq"]
        elif tuple(manifest_dims) != ("freq",):
            raise ValueError(
                f"monitor {name!r}: manifest dims {manifest_dims} != expected ['freq']")
        freqs = entry.get("freqs_hz")
        if not isinstance(freqs, list) or not freqs:
            raise ValueError(
                f"monitor {name!r}: frequency-domain manifest entry has no "
                "'freqs_hz'")
        try:
            numeric_freqs = [float(value) for value in freqs]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"monitor {name!r}: freqs_hz must contain numbers") from exc
        if any(not math.isfinite(value) or value <= 0 for value in numeric_freqs):
            raise ValueError(
                f"monitor {name!r}: freqs_hz must contain finite positive numbers")
        if len(set(numeric_freqs)) != len(numeric_freqs):
            raise ValueError(f"monitor {name!r}: freqs_hz must be unique")
        if shape[0] != len(freqs):
            raise ValueError(
                f"monitor {name!r}: shape {shape} inconsistent with {len(freqs)} freqs")
        if entry.get("axis") not in {"x", "y", "z"}:
            raise ValueError(
                f"monitor {name!r}: flux manifest entry has no valid 'axis'")
    else:
        raise ValueError(f"monitor {name!r}: unknown type {kind!r}")

    if manifest is not None:
        run = manifest.get("run", {}) if isinstance(manifest, dict) else {}
        monitors = manifest.get("monitors", []) if isinstance(manifest, dict) else []
        dt = run.get("dt_s") if isinstance(run, dict) else None
        if dt is None and isinstance(monitors, list):
            dt = next((item.get("dt_s") for item in monitors
                       if isinstance(item, dict) and item.get("dt_s") is not None), None)
        try:
            dt_value = float(dt)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"monitor {name!r}: manifest has no valid dt_s") from exc
        if not math.isfinite(dt_value) or dt_value <= 0:
            raise ValueError(
                f"monitor {name!r}: manifest dt_s must be finite and positive")
        if kind in {"field_time", "field_snapshot"}:
            max_step = run.get("steps_run", run.get("n_steps"))
            if (isinstance(max_step, int) and not isinstance(max_step, bool)
                    and max_step >= 0
                    and any(step > max_step for step in entry.get("sample_steps", []))):
                raise ValueError(
                    f"monitor {name!r}: sample_steps exceed completed run steps")

        spatial_sizes = {
            dim: shape[index] for index, dim in enumerate(manifest_dims or [])
            if dim in {"x", "y", "z"} and index < len(shape)
        }
        if spatial_sizes:
            grid = manifest.get("grid", {})
            if not isinstance(grid, dict):
                raise ValueError(f"monitor {name!r}: grid must be an object")
            grid_shape = grid.get("shape")
            if (not isinstance(grid_shape, list) or len(grid_shape) != 3
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           or value < 1 or value > _MAX_SHAPE_DIM
                           for value in grid_shape)):
                raise ValueError(
                    f"monitor {name!r}: grid shape must contain three positive "
                    "32-bit cell counts")
            grid_sizes = dict(zip(("x", "y", "z"), grid_shape))
            if kind == "field_snapshot":
                for axis, count in spatial_sizes.items():
                    if count != grid_sizes[axis]:
                        raise ValueError(
                            f"monitor {name!r}: snapshot {axis} extent {count} "
                            f"does not match grid shape {grid_sizes[axis]}")
            origin = (entry.get("origin_cells", (0, 0, 0))
                      if kind == "field_dft" else (0, 0, 0))
            stride = (entry.get("interval_space", (1, 1, 1))
                      if kind == "field_dft" else (1, 1, 1))
            origins = dict(zip(("x", "y", "z"), origin))
            strides = dict(zip(("x", "y", "z"), stride))
            for axis, count in spatial_sizes.items():
                offset, step = int(origins[axis]), int(strides[axis])
                if count and offset + (count - 1) * step >= grid_sizes[axis]:
                    raise ValueError(
                        f"monitor {name!r}: {axis} samples exceed grid extent")
            coords = grid.get("coords_um")
            dl = grid.get("dl_um")
            if coords is not None and not isinstance(coords, dict):
                raise ValueError(f"monitor {name!r}: grid coords_um must be an object")

            def validate_dl() -> None:
                try:
                    dl_value = float(dl)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"monitor {name!r}: spatial data requires grid dl_um") from exc
                if not math.isfinite(dl_value) or dl_value <= 0:
                    raise ValueError(
                        f"monitor {name!r}: grid dl_um must be finite and positive")

            if coords is None:
                validate_dl()
            else:
                needs_dl = False
                for axis, count in spatial_sizes.items():
                    if axis not in coords:
                        needs_dl = True
                        continue
                    values = coords[axis]
                    if not isinstance(values, list):
                        raise ValueError(
                            f"monitor {name!r}: grid coords_um[{axis!r}] must be a list")
                    try:
                        numeric = [float(value) for value in values]
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"monitor {name!r}: grid coords_um[{axis!r}] must "
                            "contain numbers") from exc
                    if any(not math.isfinite(value) for value in numeric):
                        raise ValueError(
                            f"monitor {name!r}: grid coords_um[{axis!r}] must "
                            "contain finite numbers")
                    offset, step = int(origins[axis]), int(strides[axis])
                    stop = offset + count * step
                    if len(values[offset:stop:step]) != count:
                        raise ValueError(
                            f"monitor {name!r}: grid coords_um[{axis!r}] is too short")
                if needs_dl:
                    validate_dl()

    expected = 1
    for value in shape:
        expected *= value
        if expected > np.iinfo(np.intp).max // np.dtype("float32").itemsize:
            raise ValueError(
                f"monitor {name!r}: shape product exceeds platform limits")
    return expected


def validate_result_manifest_contract(
    manifest: dict, *, raw_files: bool, require_top_level: bool = False,
    strict_engine: bool = False,
) -> list[dict]:
    """Validate the reader-visible v1 manifest envelope and monitor identity.

    This is shared by :class:`RunResult` and the durable ledger seal so a
    completed run cannot be published with a manifest the reader would reject.
    Raw result directories additionally require unique, safe, non-reserved blob
    filenames; HDF5 stores monitor arrays by group name instead.
    """
    if not isinstance(manifest, dict):
        raise ValueError("result manifest must be a JSON object")
    mv = str(manifest.get("manifest_version", "1"))
    major = mv.split(".", 1)[0]
    if not (major.isascii() and major.isdigit() and int(major) == 1):
        raise ValueError(
            f"unsupported manifest_version {mv!r}; this reader supports major "
            "version 1 only")
    if require_top_level:
        for block in ("run", "grid", "provenance"):
            if not isinstance(manifest.get(block), dict):
                raise ValueError(f"manifest {block} is not a JSON object")
    monitors = manifest.get("monitors", [])
    if not isinstance(monitors, list):
        raise ValueError("manifest monitors is not a list")

    names: set[str] = set()
    files: set[str] = set()
    for entry in monitors:
        if not isinstance(entry, dict):
            raise ValueError(f"monitor manifest entry must be an object: {entry!r}")
        name = entry.get("name")
        if (
            not isinstance(name, str)
            or not _is_portable_filename_token(
                name, max_length=_MONITOR_NAME_MAX_LENGTH
            )
        ):
            raise ValueError(f"manifest monitor entry has an unsafe name: {name!r}")
        name_key = _monitor_name_key(name)
        if name_key in names:
            raise ValueError(f"duplicate monitor name in manifest: {name!r}")
        names.add(name_key)
        if raw_files:
            filename = entry.get("file")
            if (not isinstance(filename, str)
                    or not _is_portable_filename_token(filename, max_length=255)
                    or Path(filename).is_absolute()
                    or Path(filename).name != filename
                    or _monitor_name_key(filename) in _RESERVED_RESULT_FILES):
                raise ValueError(
                    f"monitor {name!r} has an unsafe result filename: {filename!r}")
            filename_key = _monitor_name_key(filename)
            if filename_key in files:
                raise ValueError(
                    f"multiple monitors alias result filename {filename!r}")
            if strict_engine and filename != f"{name}.bin":
                raise ValueError(
                    f"monitor {name!r}: engine result filename must be "
                    f"{name + '.bin'!r}, got {filename!r}")
            files.add(filename_key)
        validate_monitor_manifest_entry(
            entry, manifest, require_explicit_dims=strict_engine)
    return monitors


class RunResult:
    """Lazy, dict-like view of one solver output directory.

    ``data["probe"]`` returns an :class:`xarray.DataArray`:

    - time series: dims ``('t', 'component')``, ``t`` in seconds;
    - snapshots: dims ``('t', 'component', 'z', 'y', 'x')``, spatial
      coordinates in microns (Yee-node base coordinates, ``i * dl_um``).

    A result reloaded from disk reads like the one the run returned: the
    runners write the client state the wire document does not carry (the
    user-frame origin of a fitted domain, the half a symmetry plane dropped,
    declared ports and wavelengths) beside ``sim.json``, and :attr:`simulation`
    restores it, so coordinates, planes made whole across a symmetry plane,
    :meth:`transmission` and :meth:`reflection` agree with the live result.
    ``client_state=False`` reads the result as the wire document alone
    describes it: coordinates in the corner frame, planes as recorded, no
    declared ports.
    """

    def __init__(self, path: Union[str, Path], *, simulation=None,
                 client_state: bool = True):
        path = Path(path)
        # The Simulation this result came from, when the runner hands it over;
        # otherwise loaded lazily from the ``sim.json`` the runners write beside
        # the outputs (see the ``simulation`` property).
        self._simulation = simulation
        self._client_state = bool(client_state)
        self._simulation_failed = False
        # Source can be a raw-output directory / manifest.json, OR a single
        # HDF5 file (photonhub.hdf5) / a directory holding one. HDF5 reuses
        # every reconstruction path below: only the raw-blob read differs.
        self.manifest_path: Optional[Path] = None
        self._h5_path: Optional[Path] = None
        self.manifest_sha256: Optional[str] = None
        self.manifest: dict = self._open(path)
        if not isinstance(self.manifest, dict):
            raise ValueError(
                f"result manifest must be a JSON object: {self._source}")

        # Output-contract version gate (distinct from the input-spec echo in
        # "schema_version"); absent in older draft manifests => assume v1.
        mv = str(self.manifest.get("manifest_version", "1"))
        if mv.split(".", 1)[0] != "1":
            raise ValueError(
                f"unsupported manifest_version {mv!r} in {self._source}; "
                "this reader supports major version 1 only"
            )

        run = self.manifest.get("run", {})
        if not isinstance(run, dict):
            raise ValueError(
                f"manifest 'run' block must be an object: {self._source}")
        self._run: dict = dict(run)
        if self._run.get("aborted"):
            warnings.warn(
                "loading output of an ABORTED run (reason: "
                f"{self._run.get('abort_reason') or 'unknown'}); monitor data "
                "may be partial or non-finite",
                UserWarning, stacklevel=2)

        self._entries: Dict[str, dict] = {}
        monitors = self.manifest.get("monitors", [])
        if not isinstance(monitors, list):
            raise ValueError(
                f"manifest 'monitors' must be a list: {self._source}")
        monitors = validate_result_manifest_contract(
            self.manifest, raw_files=self._h5_path is None)
        for entry in monitors:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"manifest monitor entry must be an object: {entry!r}")
            name = entry.get("name")
            if (
                not isinstance(name, str)
                or not _is_portable_filename_token(
                    name, max_length=_MONITOR_NAME_MAX_LENGTH
                )
            ):
                raise ValueError(f"manifest monitor entry without a name: {entry}")
            if name in self._entries:
                raise ValueError(f"duplicate monitor name in manifest: {name!r}")
            if self._h5_path is None:
                filename = entry.get("file")
                if (not isinstance(filename, str)
                        or not _is_portable_filename_token(
                            filename, max_length=255
                        )
                        or filename == "manifest.json"):
                    raise ValueError(
                        f"monitor {name!r} has an unsafe result filename: "
                        f"{filename!r}")
            validate_monitor_manifest_entry(entry, self.manifest)
            self._entries[name] = dict(entry)
        self._cache: Dict[str, xr.DataArray] = {}
        self._norm_amplitude_known = False
        self._norm_amplitude: Optional[float] = None

    @property
    def _source(self) -> Path:
        """The file this view was loaded from (the .h5 or the manifest.json),
        for error messages."""
        return self._h5_path or self.manifest_path

    def _open(self, path: Path) -> dict:
        """Resolve ``path`` to a manifest dict, choosing the raw-directory or
        HDF5 backend and setting output_dir / manifest_path / _h5_path."""
        if path.is_file() and path.suffix == ".h5":
            return self._open_h5(path)
        # A directory holding an .h5 but no manifest.json is the HDF5 case too.
        if path.is_dir() and not (path / "manifest.json").is_file():
            h5s = sorted(path.glob("*.h5"))
            if h5s:
                return self._open_h5(h5s[0])
        self.manifest_path = (path if path.name == "manifest.json"
                              else path / "manifest.json")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"no manifest.json or .h5 found at: {path}")
        self.output_dir = self.manifest_path.parent
        raw = self.manifest_path.read_bytes()
        # Bind the parsed dictionary to the exact bytes read. Durable-history
        # callers compare this digest to the sealed artifact after loading.
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def _h5_sim_json(self) -> Optional[str]:
        """The ``sim.json`` text ``convert_to_hdf5`` packed into the bundle,
        or None for a bundle written without one."""
        return self._h5_text("sim_json")

    def _h5_text(self, dataset: str) -> Optional[str]:
        import h5py

        with h5py.File(self._h5_path, "r") as f:
            if dataset not in f:
                return None
            return str(f[dataset].asstr()[()])

    def _open_h5(self, h5_path: Path) -> dict:
        import h5py

        self._h5_path = h5_path
        self.output_dir = h5_path.parent
        with h5py.File(h5_path, "r") as f:
            fmt = f.attrs.get("format")
            if fmt is not None and not str(fmt).startswith("photonhub-hdf5-1"):
                raise ValueError(
                    f"unsupported HDF5 format {fmt!r} in {h5_path}; this reader "
                    "supports photonhub-hdf5-1")
            mj = f.attrs.get("manifest_json")
            if mj is None:
                raise ValueError(
                    f"{h5_path} is not a PhotonHub HDF5 file (no manifest_json "
                    "attribute); convert one with photonhub.convert_to_hdf5")
            return json.loads(mj)

    @property
    def dt_s(self) -> float:
        """Recorded time step in seconds.

        Read the run metadata first, then the first monitor entry with a time
        step. Raise ``ValueError`` when neither records ``dt_s``."""
        dt = self.manifest.get("run", {}).get("dt_s")
        if dt is None:
            for entry in self._entries.values():
                if "dt_s" in entry:
                    dt = entry["dt_s"]
                    break
        if dt is None:
            raise ValueError(f"manifest has no 'dt_s' key: {self._source}")
        return float(dt)

    @property
    def provenance(self) -> dict:
        """Return a shallow copy of the recorded provenance, or an empty dict.

        Available fields depend on the solver and manifest version. This
        property reports stored metadata; it does not verify a build or device."""
        return dict(self.manifest.get("provenance", {}))

    @property
    def run(self) -> dict:
        """The manifest's run block (n_steps, dt_s, wall_seconds,
        mcells_per_s, aborted, abort_reason)."""
        return dict(self._run)

    @property
    def aborted(self) -> bool:
        """True when the solver aborted this run (NUMERICS.md section 7)."""
        return bool(self._run.get("aborted", False))

    @property
    def abort_reason(self) -> Optional[str]:
        """The section-7 reason string (``divergence`` /
        ``non_finite_energy``), or None for a healthy run."""
        reason = self._run.get("abort_reason")
        return str(reason) if reason else None

    @property
    def shut_off(self) -> bool:
        """True when the run ended early via NUMERICS.md section 7 auto-shutoff
        after the energy rule (field energy below ``run.shutoff`` of peak, or
        the fp16 plateau rule) and any DFT guard passed: a clean finish, not
        an abort."""
        return bool(self._run.get("shut_off", False))

    @property
    def stop_reason(self) -> Optional[str]:
        """Stop rule: ``dft_decay_estimate``, ``energy_decay``,
        ``fp16_plateau``, ``step_cap`` or ``aborted``; None for older runs."""
        value = self._run.get("stop_reason")
        return str(value) if value is not None else None

    @property
    def dft_shutoff(self) -> Optional[float]:
        """Effective DFT decay estimate threshold; None in older manifests."""
        value = self._run.get("dft_shutoff")
        return float(value) if value is not None else None

    @property
    def steps_run(self) -> Optional[int]:
        """Steps actually taken (``<= run.n_steps``; fewer when auto-shutoff or
        an abort ended the run early). None for older manifests without the
        key."""
        n = self._run.get("steps_run")
        return int(n) if n is not None else None

    @property
    def monitor_names(self) -> List[str]:
        """Monitor names in manifest order, as a new list."""
        return list(self._entries)

    def keys(self) -> List[str]:
        """Monitor names in manifest order, as a new list."""
        return self.monitor_names

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, name: str) -> xr.DataArray:
        """The recorded array in the user's frame. On a simulation that folded
        a declared ``symmetry`` (design spec §4.5) a monitor the fold clipped,
        or that spans the mirror plane, comes back unfolded: the plane the
        whole device would have recorded. A flux on a folded domain is the
        whole device's power through the plane (NUMERICS §20.8; its
        ``symmetry_factor`` attr is the 2^k it carries, and is absent when no
        plane cuts the monitor). :meth:`wire` is the array
        as the engine wrote it, the modeled part."""
        raw = self._raw(name)
        sim = self._known_simulation()
        if raw.attrs.get("kind") == "flux":
            return self._whole_device_flux(name, raw, sim)
        fold = getattr(sim, "_fold", None) if sim is not None else None
        if fold is None or name not in fold.unfolded_monitors:
            return raw
        views = self.__dict__.setdefault("_unfolded", {})
        if name not in views:
            from .analysis.near_field import unfold_symmetry
            from .components import frame as _frame
            full = unfold_symmetry(_frame.to_wire_frame(raw, sim), sim)
            views[name] = _frame.to_user_frame(full, sim)
        return views[name]

    def _raw(self, name: str) -> xr.DataArray:
        if name not in self._cache:
            if name not in self._entries:
                raise KeyError(
                    f"unknown monitor {name!r}; available: {self.monitor_names}"
                )
            self._cache[name] = self._load(self._entries[name])
        return self._cache[name]

    def _whole_device_flux(self, name: str, raw: xr.DataArray, sim) -> xr.DataArray:
        """A flux as the whole, unfolded device reads it: the modeled part
        times 2 for every §20 symmetry plane that cuts the monitor's plane
        (NUMERICS §20.8). Unchanged, bit for bit, when no plane cuts it."""
        factor = self._flux_symmetry_factor(name, sim)
        if factor == 1.0:
            return raw
        views = self.__dict__.setdefault("_unfolded", {})
        if name not in views:
            wide = raw.values.astype(np.float64) * factor
            whole = raw.copy(data=wide.astype(raw.dtype))
            whole.attrs["symmetry_factor"] = factor
            whole.attrs["power"] = (
                "the whole, unfolded device's power through the plane: the "
                "modeled part (RunResult.wire) times symmetry_factor, 2 for every "
                "symmetry plane that cuts the monitor plane (NUMERICS 20.8)")
            views[name] = whole
        return views[name]

    def _flux_symmetry_factor(self, name: str, sim) -> float:
        """2^k for the k §20 symmetry planes whose normal lies in the plane of
        flux monitor ``name`` and which its window reaches. A full plane
        reaches every one. On a domain fitted with ``domain=`` a window
        reaches a plane when the fold clipped it there, from a window
        symmetric about the plane (the fit refuses an edge on the plane, an
        asymmetric crossing and an edge inside the first cell beside an even
        plane), so its low edge sits on the mirror. On a half
        domain built by hand it reaches the plane when it holds the node row
        on the mirror (the engine's cell-centre membership,
        ``flux_window_range``). A window that stops short of the mirror reads
        its own region and gets no factor, like a port off the plane. Without
        the simulation, or on a §4 ``pmc`` wall (not a fold), 1. Raises for a
        fitted window on the kept side of an even plane whose edge lies inside
        the first cell of the grid the run used (a copy re-meshed after the
        fit, which refuses it, can bring one back): that flux has no
        whole-device reading."""
        sym = getattr(sim, "symmetry", None) if sim is not None else None
        if not sym or not any(sym):
            return 1.0
        mon = next((m for m in getattr(sim, "monitors", ())
                    if getattr(m, "type", None) == "flux"
                    and getattr(m, "name", None) == name), None)
        if mon is None:
            return 1.0
        a = "xyz".index(mon.axis)
        fitted = getattr(sim, "_fold", None) is not None
        coords = self.manifest.get("grid", {}).get("coords_um") or {}
        factor = 1.0
        for i, b in enumerate(((a + 1) % 3, (a + 2) % 3)):   # the cyclic (u, v) of the window
            if sym[b] == 0:
                continue
            if mon.center_um is not None:
                lo = float(mon.center_um[i]) - 0.5 * float(mon.size_um[i])

                def first_cell() -> float:
                    # the first cell of the grid the engine ran (the manifest's)
                    q = coords.get("xyz"[b])
                    return float(q[1]) - float(q[0]) if q is not None and len(q) > 1 else self._grid_dl_um()

                if fitted:
                    if lo > 1e-6:
                        dq = first_cell() if sym[b] == 1 else 0.0
                        if sym[b] == 1 and lo <= 0.5 * dq + 1e-9 * dq:
                            # the fit refuses this window (§20.8); a copy re-meshed
                            # afterwards can bring it back: refuse the reading
                            raise ValueError(
                                f"monitor {name!r}: its window starts {lo:.6g} um from the even (PMC) "
                                f"symmetry plane on {'xyz'[b]}, inside the first cell of the grid the "
                                f"run used, whose centre is {0.5 * dq:.6g} um from the plane. The run "
                                "counted the row on the plane at half weight where the run without the "
                                "plane counts it whole, so this flux has no whole-device reading. "
                                "RunResult.wire(name) is the modeled part; move the window's edge more "
                                f"than {0.5 * dq:.6g} um from the plane, or make it symmetric about it, "
                                "and run again")
                        continue                      # wholly on the kept side
                else:
                    dq = first_cell()
                    if 0.5 * dq < lo - 1e-9 * dq:
                        continue                      # the window starts above the mirror row
            factor *= 2.0
        return factor

    def wire(self, name: str) -> xr.DataArray:
        """The recorded array in the wire's corner frame, exactly the region
        the engine wrote (the simulated half of a folded domain): what the
        analysis readers that place a monitor window on a plane work on. A
        flux here is the modeled part's power, without the whole-device
        factor ``data[name]`` carries (NUMERICS §20.8)."""
        from .components import frame as _frame
        return _frame.to_wire_frame(self._raw(name), self._known_simulation())

    def __repr__(self) -> str:
        return f"RunResult({str(self.output_dir)!r}, monitors={self.monitor_names})"

    # -- visualization (photonhub.viz; docs/viz-layer-design.md) ------------

    @property
    def simulation(self):
        """The :class:`~photonhub.Simulation` this result came from: passed by
        the runner, or loaded from the ``sim.json`` the runners write beside
        the outputs (or the copy inside an HDF5 bundle). A loaded one gets back
        the client state its run recorded beside it (``client.json``): the
        user-frame origin, the symmetry-plane record, and the declared ports
        and wavelengths, whose modes are solved again on first use. A record
        that belongs to another simulation warns and restores nothing, and a
        simulation restored this way refuses ``with_changes`` (the fields as
        given are not recorded). None when no simulation is available (a bare
        output directory)."""
        if self._simulation is None:
            from .components import Simulation
            from .components import frame as _frame
            text = self._h5_sim_json() if self._h5_path is not None else None
            if text is not None:
                sim = Simulation.from_wire_json(text)
                if self._client_state:
                    where = f"{self._h5_path} (client_json)"
                    state = _frame.loads_client_state(
                        self._h5_text("client_json"), where=where)
                    if state is not None:
                        _frame.restore_client_state(sim, state, where=where)
                self._simulation = sim
                return sim
            candidate = getattr(self, "output_dir", None)
            spec = candidate / "sim.json" if candidate is not None else None
            if spec is not None and spec.is_file():
                sim = Simulation.from_file(spec)
                if self._client_state:
                    state_path = self._client_state_path(spec)
                    state = (_frame.read_client_state(state_path)
                             if state_path is not None else None)
                    if state is not None:
                        _frame.restore_client_state(sim, state, where=str(state_path))
                self._simulation = sim
        return self._simulation

    @staticmethod
    def _client_state_path(spec: Path) -> Optional[Path]:
        """The client-state record for the ``sim.json`` at ``spec``: the
        ``client.json`` beside it, or, for a cloud job's cache directory, the
        record the cloud client kept beside its cache (the record says which
        simulation it belongs to, and the restore checks that)."""
        from .components import frame as _frame
        beside = spec.with_name(_frame.CLIENT_STATE_FILE)
        if beside.is_file() and not beside.is_symlink():
            return beside
        from .cloud.cache import stored_client_state_for_result
        return stored_client_state_for_result(spec.parent)

    def _known_simulation(self):
        """:attr:`simulation`, or None (with one warning) when it cannot be
        loaded: the arrays then keep the engine's unit-amplitude normalization
        and the wire's corner frame instead of failing to read. Looked up
        once: a result with no simulation is not searched again per array."""
        if self._simulation is None and not self._simulation_failed:
            try:
                sim = self.simulation
                if sim is None:
                    self._simulation_failed = True   # none to find; do not look again
                return sim
            except (OSError, ValueError) as exc:
                self._simulation_failed = True
                warnings.warn(
                    f"could not load the simulation of {self._source} ({exc}); "
                    "frequency-domain arrays keep the engine's unit-amplitude "
                    "normalization and coordinates stay in the wire's corner "
                    "frame", UserWarning, stacklevel=3)
        return self._simulation

    @property
    def norm_amplitude(self) -> Optional[float]:
        """``A0``, the amplitude of the first wire-order source: the
        engine's section-12 normalization amplitude, which this reader
        multiplies back into every frequency-domain array (``A0`` on
        ``field_dft`` phasors, ``A0^2`` on flux). None when the simulation is
        not known, in which case the arrays keep the engine's unit-amplitude
        normalization (their ``normalization`` attr says so)."""
        if not self._norm_amplitude_known:
            a0 = None
            sim = self._known_simulation()
            sources = getattr(sim, "sources", ()) if sim is not None else ()
            if sources:
                a0 = float(getattr(sources[0], "amplitude", 1.0))
            self._norm_amplitude = a0
            self._norm_amplitude_known = True
        return self._norm_amplitude

    def _restore_amplitude(self, data: np.ndarray, attrs: dict, *,
                           power: bool) -> np.ndarray:
        """Undo the engine's per-unit-amplitude normalization: ``A0^2`` on a
        flux (``power=True``), ``A0`` on phasors. Records the convention the
        array ends up with in ``attrs``. Computed in double precision and
        cast back to the stored dtype; ``A0 == 1`` (a unit first source) and
        an unknown ``A0`` return the array untouched, bit for bit."""
        a0 = self.norm_amplitude
        attrs["norm_amplitude"] = a0
        if a0 is None:
            attrs["normalization"] = _FLUX_NORMALIZATION if power else _DFT_NORMALIZATION
            return data
        attrs["normalization"] = _FLUX_ABSOLUTE if power else _DFT_ABSOLUTE
        if a0 == 1.0:
            return data
        scale = a0 * a0 if power else a0
        wide = np.complex128 if np.iscomplexobj(data) else np.float64
        return (data.astype(wide) * scale).astype(data.dtype)

    @property
    def port_names(self) -> List[str]:
        """The ports of a declarative simulation, in declaration order; a port
        the symmetry reduction dropped is listed after them and read through its
        image."""
        sim = self.simulation
        if sim is None:
            return []
        fold = getattr(sim, "_fold", None)
        # the names without solving: a reloaded simulation solves its port
        # modes on first use, and listing them needs none
        decl = getattr(sim, "_declarative", None)
        names = list(decl.port_monitors) if decl is not None else []
        return names + (list(fold.mirrored_ports) if fold is not None else [])

    def transmission(self, port: str, **kwargs):
        """Modal power transmission into ``port`` from the driven port of a
        simulation built with ``ports=`` and ``source=``: an ``xarray.DataArray``
        over ``f`` with a ``wlen_um`` coordinate (see
        :func:`photonhub.analysis.transmission_spectrum`, whose keywords pass
        through). The power the device sends back into the driven port is
        :meth:`reflection`.

        **Limitation: a beam-driven simulation has no transmission.** With
        ``source=`` a :class:`~photonhub.GaussianBeam` there is no driven port
        whose plane reads the launched power in the launch direction, and the
        beam's ``power_watts`` is not a per-frequency reference: the beam's
        source amplitudes are fixed at the band centre, so the power it
        launches drifts across the band, and a flux plane at the beam would
        also count whatever the device reflects back through it. This method
        therefore raises for a beam-driven result. Read the absolute modal
        power of a port instead, ``data.simulation.port_monitors[name]
        .mode_power(data)`` (flux-commensurate with a
        :class:`~photonhub.PowerMonitor`; on a folded domain both, and the
        beam's ``power_watts``, are the whole device's power, NUMERICS
        §20.8), and normalize it against a
        reference run of your own, for example the same beam launched into
        the bare background with a :class:`~photonhub.PowerMonitor` across
        its path."""
        from .analysis.mode_devices import transmission_spectrum

        sim = self.simulation
        monitors = sim.port_monitors if sim is not None else {}
        if not monitors:
            raise ValueError(
                "this result's simulation declares no ports (built by hand, or loaded "
                "from the wire); use photonhub.analysis.transmission on your own "
                "ModeMonitor objects")
        if port not in monitors:
            fold = getattr(sim, "_fold", None)
            if fold is not None and port in fold.mirrored_ports:
                # a port in the mirrored half of the fold: its image's reading (§4.5)
                image = fold.mirrored_ports[port]
                t = self.transmission(image, **kwargs)
                t = t.copy()
                t.attrs["mirror_of"] = image
                t.name = port
                return t
            raise KeyError(f"no port named {port!r}; the ports are {list(monitors)}")
        driven = sim.driven_port
        if driven is None:
            raise ValueError("the simulation drives no port (source= is not a port), so "
                             "there is no launched power to normalize by; read "
                             "simulation.port_monitors[name].mode_power(data) and "
                             "normalize against a reference run (see transmission's "
                             "docstring: a beam's power_watts is not a per-frequency "
                             "reference)")
        return transmission_spectrum(monitors[port], monitors[driven], self, **kwargs)

    def reflection(self, port: Optional[str] = None, **kwargs):
        """Modal power reflection at the driven port of a simulation built with
        ``ports=`` and ``source=``: the driven port's monitor read against the
        launch direction (the power the device sends back toward the source)
        over the same monitor read in the launch direction (the launched
        power), as an ``xarray.DataArray`` over ``f`` with a ``wlen_um``
        coordinate (see :func:`photonhub.analysis.reflection_spectrum`, whose
        keywords pass through). The two readings share one plane and one mode,
        so ``R + T = 1`` across the band is the energy check for a lossless
        device.

        ``port`` defaults to the driven port, and naming any other port raises
        ``ValueError``, a port mirrored by a symmetry plane included. Every port
        that is not driven reads outward, the power leaving the device, so its
        monitor read the other way measures the wave arriving from its own
        boundary (nothing, with no source there), not the power reflected into
        it. To read the reflection at another port, drive that port
        (``source=``) and run again. Like :meth:`transmission` it has no
        launched power to divide by in a beam-driven simulation, and raises
        (see the limitation there)."""
        from .analysis.mode_devices import reflection_spectrum

        sim = self.simulation
        monitors = sim.port_monitors if sim is not None else {}
        if not monitors:
            raise ValueError(
                "this result's simulation declares no ports (built by hand, or loaded "
                "from the wire); use photonhub.analysis.reflection on your own "
                "ModeMonitor objects")
        driven = sim.driven_port
        if driven is None:
            raise ValueError("the simulation drives no port (source= is not a port), so "
                             "there is no launched power to normalize the reflection by; "
                             "read simulation.port_monitors[name].mode_power(data, "
                             "direction=...) and normalize against a reference run (see "
                             "RunResult.transmission's docstring: a beam's power_watts is "
                             "not a per-frequency reference)")
        if port is None:
            port = driven
        if port != driven:
            if port not in self.port_names:
                raise KeyError(f"no port named {port!r}; the ports are {self.port_names}")
            raise ValueError(
                f"reflection is read at the driven port ({driven!r}), and {port!r} is not "
                "driven: a port that is not driven reads the power leaving the device, so "
                "read the other way it measures the wave arriving from its own boundary, "
                f"not a reflection. Drive {port!r} (source={port!r}) to read its reflection")
        return reflection_spectrum(monitors[driven], monitors[driven], self, **kwargs)

    def plot_field(self, monitor, field="Ex", x=None, y=None, z=None, *,
                   freq=None, val="real", structures=True, simulation=None,
                   ax=None, cmap=None, unfold=True, **kw):
        """Heatmap of a field component on a 2D slice of ``self[monitor]``.

        Thin delegation: the rendering lives in :func:`photonhub.viz.plot_field`
        (imported lazily so matplotlib loads only when a plot is requested).
        ``field`` is Ex..Hz or a derived 'E'/'intensity'/'H'; ``freq=`` is
        required for a multi-frequency DFT monitor; ``val`` selects
        real/imag/abs/phase for complex data. ``unfold`` (the default) mirrors
        a NUMERICS §20 half domain back into the whole device, with each
        component's parity about the plane; it needs ``simulation=``. Returns
        a matplotlib ``Axes``."""
        from .viz import plot_field as _plot_field
        if simulation is None:
            simulation = self.simulation
        return _plot_field(self, monitor, field=field, x=x, y=y, z=z, freq=freq,
                           val=val, structures=structures, simulation=simulation,
                           ax=ax, cmap=cmap, unfold=unfold, **kw)

    def preview(self, monitor, *, simulation=None):
        """Interactive Jupyter scrubber over a recorded field monitor: component,
        real/imag/abs/phase, frequency, and (volumetric monitors) the cut plane,
        with optional structure outlines (``simulation=``). Requires the
        ``photonhub[viz]`` extra and a notebook. See
        :func:`photonhub.viz.interactive_field`."""
        from .viz import interactive_field as _interactive_field
        if simulation is None:
            simulation = self.simulation
        return _interactive_field(self, monitor, simulation=simulation)

    # -- internals ----------------------------------------------------------

    def _grid_dl_um(self) -> float:
        grid = self.manifest.get("grid", {})
        if "dl_um" in grid:
            return float(grid["dl_um"])
        raise ValueError(f"manifest grid block has no 'dl_um': {grid}")

    def _axis_coord_um(self, axis: str, origin: int, n: int,
                       stride: int = 1) -> np.ndarray:
        """Yee-node base coordinates (microns) for `n` recorded cells starting at
        cell `origin` along `axis`, sampled every `stride` cells (NUMERICS.md
        section 12 interval_space; stride 1 = every cell). Graded runs (section
        15.10) carry the per-axis 'coords_um' arrays in the manifest; uniform
        runs fall back to the i * dl_um rule."""
        coords = self.manifest.get("grid", {}).get("coords_um")
        if coords is not None and axis in coords:
            q = np.asarray(coords[axis], dtype=np.float64)
            base = q[origin:origin + n * stride:stride]
        else:
            base = (float(origin) + np.arange(n, dtype=np.float64) * stride) \
                * self._grid_dl_um()
        # A simulation fitted around its device (design spec §4.4) was built in
        # the user's frame and records the wire corner in origin_um; report the
        # coordinates in that frame. A document loaded from the wire has (0,0,0).
        return base + self._frame_origin(axis)

    def _frame_origin(self, axis: str) -> float:
        sim = self._known_simulation()
        origin = getattr(sim, "origin_um", None) if sim is not None else None
        return float(origin["xyz".index(axis)]) if origin is not None else 0.0

    def _load(self, entry: dict) -> xr.DataArray:
        name = entry["name"]
        validate_monitor_manifest_entry(entry, self.manifest)
        dtype = entry.get("dtype", "float32")
        if dtype != "float32":
            raise ValueError(f"monitor {name!r}: unsupported dtype {dtype!r}")
        # Exactly the keys output.cpp emits — no aliases for never-shipped
        # draft manifests.
        kind = entry.get("type")
        if kind in ("field_time", "field_snapshot"):
            return self._load_time_domain(entry)
        if kind == "field_dft":
            return self._load_field_dft(entry)
        if kind == "flux":
            return self._load_flux(entry)
        raise ValueError(f"monitor {name!r}: unknown type {kind!r}")

    def _read_raw(self, entry: dict, expected: int) -> np.ndarray:
        """The monitor's raw float32 array (from the .bin, or the HDF5
        ``/monitors/<name>`` dataset), flat and length-checked against the
        manifest shape. Both backends store the identical little-endian bytes,
        so reconstruction downstream is bit-identical."""
        name = entry["name"]
        if self._h5_path is not None:
            import h5py
            with h5py.File(self._h5_path, "r") as f:
                raw = np.asarray(f["monitors"][name][...], dtype="<f4").ravel()
            source = f"{self._h5_path} [/monitors/{name}]"
        else:
            filename = entry.get("file")
            if (not isinstance(filename, str)
                    or not _is_portable_filename_token(filename, max_length=255)
                    or filename == "manifest.json"):
                raise ValueError(
                    f"monitor {name!r} has an unsafe result filename: "
                    f"{filename!r}")
            bin_path = self.output_dir / filename
            if bin_path.is_symlink() or not bin_path.is_file():
                raise ValueError(
                    f"monitor {name!r}: result file is missing or unsafe: "
                    f"{bin_path}")
            # Keep raw bundles disk-backed.  ``np.fromfile`` eagerly copied an
            # entire multi-GB snapshot before the viz service could select one
            # frequency/time/cut plane, contradicting the local slice-on-demand
            # desktop architecture.  A read-only memmap preserves the public
            # DataArray API while NumPy/xarray selections touch only requested
            # pages.  HDF5 remains eager for now because its dataset lifetime
            # cannot safely outlive the context manager without a dedicated
            # store abstraction.
            byte_size = bin_path.stat().st_size
            if byte_size != expected * np.dtype("<f4").itemsize:
                raise ValueError(
                    f"monitor {name!r}: {bin_path} holds {byte_size // 4} "
                    f"float32 values, manifest shape "
                    f"{tuple(int(n) for n in entry['shape'])} needs {expected}"
                )
            # mmap cannot map a zero-byte file; aborted interval-0 snapshots
            # legitimately carry shape[0] == 0 and an empty blob.
            raw = (np.empty((0,), dtype="<f4") if expected == 0 else
                   np.memmap(bin_path, dtype="<f4", mode="r", shape=(expected,)))
            source = str(bin_path)
        if raw.size != expected:
            raise ValueError(
                f"monitor {name!r}: {source} holds {raw.size} "
                f"float32 values, manifest shape "
                f"{tuple(int(n) for n in entry['shape'])} needs {expected}"
            )
        return raw

    def _freqs_hz(self, entry: dict) -> np.ndarray:
        name = entry["name"]
        freqs = entry.get("freqs_hz")
        if not freqs:
            raise ValueError(
                f"monitor {name!r}: frequency-domain manifest entry has no "
                "'freqs_hz'"
            )
        return np.asarray([float(f) for f in freqs], dtype=np.float64)

    def _common_attrs(self, name: str, kind: str) -> dict:
        return {
            "monitor": name,
            "kind": kind,
            "dt_s": self.dt_s,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason or "",
            "provenance": self.provenance,
        }

    def _load_time_domain(self, entry: dict) -> xr.DataArray:
        name = entry["name"]
        shape = tuple(int(n) for n in entry["shape"])
        components = list(entry["components"])
        sample_steps = [int(s) for s in entry["sample_steps"]]

        kind = entry.get("type")
        if kind == "field_time":
            kind, dims = "time_series", _TIME_DIMS
        else:
            kind, dims = "snapshot", _SNAPSHOT_DIMS
        manifest_dims = entry.get("dims")
        if manifest_dims is not None:
            # The engine names the leading dim "sample" (NUMERICS.md section
            # 6); the xarray dim is "t" with coordinates in seconds.
            normalized = ("t",) + tuple(manifest_dims[1:])
            if tuple(manifest_dims[:1]) != ("sample",) or normalized != dims:
                raise ValueError(
                    f"monitor {name!r}: manifest dims {manifest_dims} != expected {list(dims)}"
                )
        if len(shape) != len(dims):
            raise ValueError(
                f"monitor {name!r}: shape {shape} has {len(shape)} dims, expected {len(dims)}"
            )
        if shape[0] != len(sample_steps) or shape[1] != len(components):
            raise ValueError(
                f"monitor {name!r}: shape {shape} inconsistent with "
                f"{len(sample_steps)} samples x {len(components)} components"
            )

        data = self._read_raw(entry, int(np.prod(shape))).reshape(shape)

        dt = self.dt_s
        coords = {
            "t": ("t", np.asarray(sample_steps, dtype=np.float64) * dt,
                  {"units": "s", "long_name": "E-field sample time (step * dt)"}),
            "component": list(components),
        }
        if kind == "snapshot":
            for axis, n in zip(("z", "y", "x"), shape[2:]):
                coords[axis] = (axis, self._axis_coord_um(axis, 0, n),
                                {"units": "um"})

        attrs = self._common_attrs(name, kind)
        attrs["sample_steps"] = sample_steps
        return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs, name=name)

    def _load_field_dft(self, entry: dict) -> xr.DataArray:
        """NUMERICS.md section 12 field_dft: float32 [re, im] pairs in binary
        order [freq][component][k][j][i][re,im], reconstructed as complex64
        with dims ('f', 'component', 'z', 'y', 'x'). Spatial coordinates are
        Yee cell base coordinates (index * dl_um, offset by the optional
        'origin_cells' region corner). The section-12 snapping rule makes
        origin_cells EXACT for every listed component (the validator rejects
        specs whose per-component snaps disagree).

        .. warning:: The coordinates are BASE-NODE positions for EVERY
           component: the per-component Yee half-cell offsets (NUMERICS.md
           section 1.1, E and H components live at staggered points, H also
           half a step earlier in time) are NOT applied here, as for
           snapshots. Combining raw components by hand, in particular an
           E x H* Poynting flux from a field_dft plane, without first
           colocating/destaggering them gives a systematically wrong answer
           (this exact mistake once mis-diagnosed a mode-launch 'reflection'
           that modal projection showed was 30x smaller). Use the
           colocation/destagger handling in ``photonhub.analysis.mode_overlap``
           (what ``mode_devices``' ``mode_power(..., colocate=True,
           destagger_dl=dl)`` applies), or a flux monitor, which the engine
           computes on the staggered grid correctly, before hand-computing
           power."""
        name = entry["name"]
        shape = tuple(int(n) for n in entry["shape"])
        components = list(entry["components"])
        freqs = self._freqs_hz(entry)

        if len(shape) != 6 or shape[-1] != 2:
            raise ValueError(
                f"monitor {name!r}: field_dft shape {shape} must be "
                "[freq, component, z, y, x, 2]"
            )
        manifest_dims = entry.get("dims")
        if manifest_dims is not None:
            # Exactly the dim names output.cpp emits — no aliases. The
            # engine names the leading dim "freq"; the xarray dim is "f"
            # with coordinates in Hz. The trailing [re, im] pair dim (named
            # "complex" by the engine's output.cpp) is consumed by the
            # complex64 reconstruction.
            normalized = ("f",) + tuple(manifest_dims[1:-1])
            if (tuple(manifest_dims[:1]) != ("freq",)
                    or tuple(manifest_dims[-1:]) != ("complex",)
                    or normalized != _DFT_DIMS):
                raise ValueError(
                    f"monitor {name!r}: manifest dims {manifest_dims} != "
                    "expected ['freq', 'component', 'z', 'y', 'x', 'complex']"
                )
        if shape[0] != len(freqs) or shape[1] != len(components):
            raise ValueError(
                f"monitor {name!r}: shape {shape} inconsistent with "
                f"{len(freqs)} freqs x {len(components)} components"
            )

        raw = self._read_raw(entry, int(np.prod(shape)))
        # Adjacent [re, im] float32 pairs are exactly numpy's complex64
        # memory layout — a view, not a lossy round trip through complex128.
        data = raw.view("<c8").reshape(shape[:-1])

        origin = entry.get("origin_cells", (0, 0, 0))
        if len(origin) != 3:
            raise ValueError(
                f"monitor {name!r}: origin_cells {origin} must be the region "
                "low corner as [i0, j0, k0] (x, y, z cell indices)"
            )
        # §12 interval_space: per-axis spatial stride (x, y, z). Absent => every
        # cell (stride 1). Strides the reconstructed coordinates so a decimated
        # plane reads back at the right physical positions.
        stride = entry.get("interval_space", (1, 1, 1))
        if len(stride) != 3:
            raise ValueError(
                f"monitor {name!r}: interval_space {stride} must be "
                "[sx, sy, sz] (x, y, z strides)"
            )
        coords = {
            "f": ("f", freqs, {"units": "Hz"}),
            "wlen_um": ("f", _wlen_um(freqs), {"units": "um"}),
            "component": components,
        }
        # origin_cells / interval_space are (x, y, z); the spatial block is
        # (z, y, x), so reverse both alongside it.
        for axis, n, o, s in zip(("z", "y", "x"), shape[2:5],
                                 reversed(list(origin)), reversed(list(stride))):
            coords[axis] = (axis, self._axis_coord_um(axis, int(o), n, int(s)),
                            {"units": "um"})

        attrs = self._common_attrs(name, "field_dft")
        attrs["freqs_hz"] = [float(f) for f in freqs]
        data = self._restore_amplitude(data, attrs, power=False)
        return xr.DataArray(data, dims=_DFT_DIMS, coords=coords, attrs=attrs,
                            name=name)

    def _load_flux(self, entry: dict) -> xr.DataArray:
        """NUMERICS.md section 12 flux: one float32 power per frequency
        (fp64-accumulated in the engine), dims ('f',), positive toward
        +axis. The engine's shared 1/|A0*S(f)|^2 normalization is undone
        here (``A0^2`` multiplied back in) so the value is watts for the
        sources as declared; see :meth:`norm_amplitude`."""
        name = entry["name"]
        shape = tuple(int(n) for n in entry["shape"])
        freqs = self._freqs_hz(entry)

        if len(shape) != 1:
            raise ValueError(
                f"monitor {name!r}: flux shape {shape} must be [freq]"
            )
        manifest_dims = entry.get("dims")
        if manifest_dims is not None and tuple(manifest_dims) != ("freq",):
            # Exactly the dim name output.cpp emits — no aliases.
            raise ValueError(
                f"monitor {name!r}: manifest dims {manifest_dims} != "
                "expected ['freq']"
            )
        if shape[0] != len(freqs):
            raise ValueError(
                f"monitor {name!r}: shape {shape} inconsistent with "
                f"{len(freqs)} freqs"
            )

        data = self._read_raw(entry, shape[0])

        attrs = self._common_attrs(name, "flux")
        attrs["freqs_hz"] = [float(f) for f in freqs]
        data = self._restore_amplitude(data, attrs, power=True)
        # output.cpp emits "axis" unconditionally for flux entries (it throws
        # on a spec lookup miss): without it the sign convention of the
        # reported power is unrecoverable from the artifact.
        if "axis" not in entry:
            raise ValueError(
                f"monitor {name!r}: flux manifest entry has no 'axis' (the "
                "plane normal; required — output.h manifest contract)"
            )
        attrs["axis"] = entry["axis"]
        coords = {"f": ("f", freqs, {"units": "Hz"}),
                  "wlen_um": ("f", _wlen_um(freqs), {"units": "um"})}
        return xr.DataArray(data, dims=_FLUX_DIMS, coords=coords, attrs=attrs,
                            name=name)
