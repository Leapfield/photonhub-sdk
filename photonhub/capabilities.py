"""Client/engine capability-manifest synchronization.

The pydantic models provide early structural validation, while ``phsolver
validate`` remains authoritative for grid- and device-specific constraints.
This module pins the coarse feature list emitted by ``phsolver
--capabilities`` and exposes a drift check so the client and engine cannot
silently advertise different surfaces.
"""

import json
import subprocess
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from .components.simulation import SUPPORTED_SCHEMA_MAJOR
from ._compat import caller_stacklevel


# Public compatibility constant used by callers comparing the solver manifest.
SCHEMA_MAJOR = SUPPORTED_SCHEMA_MAJOR


_implicit_dft_warning_sent = False
_implicit_dft_warning_lock = threading.Lock()


def warn_implicit_dft_shutoff_unsupported(target: str) -> None:
    """Explain once per process why a frequency monitor uses energy-only stop."""
    global _implicit_dft_warning_sent
    with _implicit_dft_warning_lock:
        if _implicit_dft_warning_sent:
            return
        _implicit_dft_warning_sent = True
        warnings.warn(
            f"{target} does not advertise dft_shutoff; recorded DFT/flux "
            "decay is not estimated and the energy-only stop is used",
            UserWarning, stacklevel=caller_stacklevel())


def omit_unsupported_dft_shutoff(wire: dict, target: str) -> dict:
    """Drop the optional guard on a target without the capability."""
    if "dft_shutoff" in wire.get("run", {}):
        wire["run"].pop("dft_shutoff")
        warnings.warn(
            f"{target} does not advertise dft_shutoff; the run will use "
            "energy-only auto-shutoff", UserWarning, stacklevel=caller_stacklevel())
    return wire

# The feature flags the v1 engine advertises via ``phsolver --capabilities``
# (the solver's capability manifest). Pinned here so the drift test
# fails the build the moment the engine manifest changes without the client
# being updated in lockstep — in EITHER direction.
#
# 2026-07 manifest refresh: the list had frozen at Phase 1a-1 while the engine
# shipped nine more features; both sides now advertise the full surface. Each
# name mirrors its wire surface. A drift-test failure
# against an OLDER binary means that binary predates this refresh — rebuild it.
ENGINE_ADVERTISED_FEATURES = frozenset({
    "uniform_grid", "point_dipole", "gaussian_pulse", "pec", "periodic",
    "field_time", "field_snapshot",
    # Phase 1a-1
    "structures", "lossy_media", "pml", "plane_wave", "field_dft", "flux",
    # Shipped since (2026-07 manifest refresh)
    "graded_grid",      # nonuniform coords (schema 1.2)
    "subpixel",         # volume/tensor/tensor_full
    "mode_source",      # incl. broadband modes_by_freq
    "lorentz_media",    # single-pole ADE dispersion
    "multi_pole_media",  # multi-pole + Drude ADE
    "pec_media",  # PEC structure material
    "symmetry",         # PEC/PMC symmetry planes
    "absorber",         # adiabatic absorber boundary
    "magnetic_dipole",  # PointDipole polarization Hz
    "apodization",      # DFT time window
    "interval_space",   # DFT spatial decimation
    "mode_port",        # post-processing metadata (schema 1.16)
    # Schema 1.18 (2026-08)
    "anisotropic_media",   # diagonal eps tensor
    "bloch",               # Bloch boundaries (CPU)
    "oblique_plane_wave",  # stage C constant-k tilt
    "tfsf_box",            # closed TF/SF box (CPU)
    # Schema 1.19 (2026-08)
    "field_precision_fp16",  # fp16 field storage lane
    # Schema 1.20 (2026-08)
    "dft_precision_narrow",  # narrowed field_dft accumulator lanes
    "dft_shutoff",  # CPU-only recorded-result decay estimate
    # Schema 1.22 (2026-10), CPU reference only (the
    # custom_media_coords_or_conductivity row of CPU_ONLY_FEATURES refuses
    # them on a GPU)
    "custom_media_coords",        # permittivity_data.coords_um
    "custom_media_conductivity",  # permittivity_data.conductivity_s_per_m
})

_PROBE_TIMEOUT_S = 30.0


def engine_capabilities(
    solver_path: Union[str, Path, None] = None,
) -> Optional[dict]:
    """Parse ``phsolver --capabilities`` into a dict, or ``None`` when no solver
    binary is configured/found.

    Locates the binary the same way a run does (explicit arg, ``$PHOTONHUB_SOLVER``,
    ``PATH``, then the in-repo build dir) via :func:`photonhub.find_solver`.
    Raises on a present-but-broken binary (non-zero exit or unparseable output)
    — a silent fallback would defeat the point of the drift gate.
    """
    # Lazy import: runners.local imports the components package, which imports
    # this module — importing find_solver at module load would be a cycle.
    from .runners.local import find_solver
    from .runners.phsolver import _solver_subprocess_env

    solver = find_solver(solver_path)
    if solver is None:
        return None
    proc = subprocess.run(
        [str(solver), "--capabilities"],
        capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
        env=_solver_subprocess_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{solver} --capabilities exited {proc.returncode}: "
            f"{proc.stderr.strip()[:400]}")
    return json.loads(proc.stdout)


def engine_feature_drift(
    solver_path: Union[str, Path, None] = None,
) -> Optional[set]:
    """The symmetric difference between the engine's advertised feature set and
    :data:`ENGINE_ADVERTISED_FEATURES`. Empty set = in sync; ``None`` when no
    solver binary is available (so callers can skip rather than fail). The CI
    drift gate asserts this is the empty set.
    """
    caps = engine_capabilities(solver_path)
    if caps is None:
        return None
    advertised = set(caps.get("features", []))
    return advertised ^ set(ENGINE_ADVERTISED_FEATURES)


# --- device capability exclusions ----------------------------------------- #
#
# ``phsolver --capabilities`` is a flat list with no device dimension, and
# ``phsolver validate`` takes no ``--device``,
# so neither can express that a shipped feature is CPU-reference-only.
# ``GpuSolver::run`` (the engine's GPU solver) therefore throws
# SpecError on these features at the moment the GPU run starts, which on the
# paid cloud path is after a worker has been provisioned against a bound quote.
#
# The table below is the machine-checkable client mirror of those throws,
# so a GPU-targeted run is refused before it can cost anything. The engine
# throws stay as the backstop: this is defence in depth, not a replacement, and
# the engine remains the authority on its own spec.
#
# ``oblique_plane_wave`` needs no row of its own: an oblique wave's
# transverse phase advance IS a Bloch phase, and the engine's validation rejects ``angle_theta != 0`` unless the transverse axis is
# ``bloch`` -- so the ``bloch`` row already covers it on both solvers.


@dataclass(frozen=True)
class CpuOnlyFeature:
    """One shipped feature the GPU backend refuses, with the engine's message.

    ``message`` is copied verbatim from the matching ``throw SpecError`` in
    ``GpuSolver::run``, so the client and the engine say the same thing about
    the same spec. ``tests/test_device_capabilities.py`` pins every message
    against the engine source: editing a throw without editing this table
    fails the suite, in either direction.
    """

    name: str
    section: str
    message: str
    detect: Callable[[dict], bool]

    def used_by(self, spec: dict) -> bool:
        """True when the wire spec ``spec`` uses this feature."""
        return bool(self.detect(spec))


def _has_custom_media_grid_keys(spec: dict) -> bool:
    # Schema 1.22 keys on ``permittivity_data`` (node coordinates and a
    # per-node conductivity grid), matching the engine's own loop over
    # ``spec_.structures`` for them. Kept
    # apart from the ``permittivity_data`` row, which a GPU port of
    # box-spanning custom media removes; this one stays until the GPU carries
    # the coordinates and the conductivity grid.
    for st in spec.get("structures") or ():
        pd = ((st or {}).get("medium") or {}).get("permittivity_data") or {}
        if (pd.get("coords_um") is not None
                or pd.get("conductivity_s_per_m") is not None):
            return True
    return False


def _has_boundary(spec: dict, condition: str) -> bool:
    boundaries = spec.get("boundaries") or {}
    return any(boundaries.get(axis) == condition for axis in ("x", "y", "z"))


def _has_source_type(spec: dict, type_name: str) -> bool:
    return any((s or {}).get("type") == type_name
               for s in spec.get("sources") or ())


def _has_cw_source(spec: dict) -> bool:
    # The engine also tests ``norm_pulse.cw``, but norm_pulse is filled from the
    # first source in wire order, so scanning every
    # source covers that arm too -- there is no separate wire field.
    return any(((s or {}).get("source_time") or {}).get("type") == "cw"
               for s in spec.get("sources") or ())


def _has_permittivity_data(spec: dict) -> bool:
    # Matches the engine's loop over ``spec_.structures`` only: a background
    # permittivity carries no data array, and the GPU throw does not test one.
    return any(((st or {}).get("medium") or {}).get("permittivity_data")
               is not None for st in spec.get("structures") or ())


CPU_ONLY_FEATURES = (
    CpuOnlyFeature(
        name="custom_media_coords_or_conductivity",
        section="custom media node coordinates and conductivity grid",
        message=("custom media node coordinates and conductivity grids "
                 "(permittivity_data coords_um, conductivity_s_per_m) are "
                 "supported on the CPU reference solver only in this "
                 "release. A domain fit gives coords_um to the guide "
                 "extension it adds behind a port whose guide is a "
                 "box-spanning grid; draw that guide through the domain "
                 "walls so the fit adds none, or run with --device cpu"),
        detect=_has_custom_media_grid_keys,
    ),
    CpuOnlyFeature(
        name="dft_shutoff",
        section="frequency-domain decay stop",
        message=("dft_shutoff is supported on the CPU reference solver only "
                 "in this release; run with --device cpu"),
        detect=lambda spec: (spec.get("run", {}).get("dft_shutoff") or 0) > 0,
    ),
    CpuOnlyFeature(
        name="bloch",
        section="Bloch boundaries",
        message=("bloch boundaries are supported on the CPU reference solver "
                 "only in this release — run with --device cpu"),
        detect=lambda spec: _has_boundary(spec, "bloch"),
    ),
    CpuOnlyFeature(
        name="tfsf_box",
        section="total-field/scattered-field box",
        message=("tfsf_box sources are supported on the CPU reference solver "
                 "only in this release — run with --device cpu"),
        detect=lambda spec: _has_source_type(spec, "tfsf_box"),
    ),
    CpuOnlyFeature(
        name="permittivity_data",
        section="custom permittivity data",
        message=("permittivity_data (custom media) is supported on the CPU "
                 "reference solver only in this release — run with "
                 "--device cpu"),
        detect=_has_permittivity_data,
    ),
    CpuOnlyFeature(
        name="pmc",
        section="PMC boundaries",
        message=("pmc boundaries are supported on the CPU reference solver "
                 "only in this release — run with --device cpu"),
        detect=lambda spec: _has_boundary(spec, "pmc"),
    ),
    CpuOnlyFeature(
        name="cw",
        section="continuous-wave sources",
        message=("cw sources are supported on the CPU reference solver only "
                 "in this release — run with --device cpu"),
        detect=_has_cw_source,
    ),
)

# Name -> feature, for callers that want to look one up.
CPU_ONLY_FEATURES_BY_NAME = {f.name: f for f in CPU_ONLY_FEATURES}


def selects_gpu(device, *, unset_may_be_gpu: bool = False) -> bool:
    """True when ``device`` names a GPU in either device grammar.

    Local (``device_args`` in ``runners/phsolver.py``): ``cpu`` / ``gpu`` /
    ``gpu:N`` / ``gpu:all`` / ``gpu:N,M,...``. Cloud (the device validator in
    ``cloud/run.py``): ``cpu`` / ``gpu`` / ``gpu:<id>``. Both grammars put the
    backend first, so one split settles it for both.

    ``device=None`` means CPU locally (phsolver's own default) but *service
    policy* on the cloud, which no client can see. Cloud callers therefore pass
    ``unset_may_be_gpu=True``, so an unset device is treated as possibly-GPU
    rather than assumed safe.
    """
    if device is None:
        return unset_may_be_gpu
    return str(device).strip().split(":", 1)[0] == "gpu"


def unsupported_on_device(spec, device, *, unset_may_be_gpu: bool = False):
    """The :data:`CPU_ONLY_FEATURES` ``spec`` uses that ``device`` cannot run.

    ``spec`` is a :class:`~photonhub.Simulation` or an already-built wire dict.
    Returns an empty tuple whenever ``device`` is not a GPU selector, so this is
    safe to call unconditionally on any run path.
    """
    if not selects_gpu(device, unset_may_be_gpu=unset_may_be_gpu):
        return ()
    wire = spec.to_wire_dict() if hasattr(spec, "to_wire_dict") else spec
    return tuple(f for f in CPU_ONLY_FEATURES if f.used_by(wire))


def check_device_support(spec, device, *, unset_may_be_gpu: bool = False,
                         note: Optional[str] = None) -> None:
    """Raise before a GPU run that ``GpuSolver::run`` would throw ``SpecError`` on.

    The guard the paid path needs: callers run this *before* binding a quote or
    spawning a solver, so a CPU-reference-only feature costs a client-side
    exception instead of a provisioned GPU worker. ``note`` is appended by
    callers that can say something the core message cannot, such as the cloud
    path confirming that no job was submitted.
    """
    blocked = unsupported_on_device(
        spec, device, unset_may_be_gpu=unset_may_be_gpu)
    if not blocked:
        return
    # Lazy import: the components package imports this module, so pulling the
    # runners package in at module load would be a cycle.
    from .runners.phsolver import SolverRunError

    selector = "unset (the service picks)" if device is None else repr(device)
    plural = "feature is" if len(blocked) == 1 else "features are"
    lines = [
        f"device {selector} cannot run this simulation: "
        f"{len(blocked)} {plural} supported on the CPU reference solver only "
        "in this release, so the run was refused before it started."
    ]
    lines += [f"  - {f.name} ({f.section}): {f.message}" for f in blocked]
    lines.append(
        'Run it on the CPU with device="cpu", or drop the feature. '
        "The engine rejects the same spec on the GPU.")
    if note:
        lines.append(note)
    raise SolverRunError("\n".join(lines))
