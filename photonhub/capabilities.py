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
            UserWarning, stacklevel=2)


def omit_unsupported_dft_shutoff(wire: dict, target: str) -> dict:
    """Drop the optional guard on a target without the capability."""
    if "dft_shutoff" in wire.get("run", {}):
        wire["run"].pop("dft_shutoff")
        warnings.warn(
            f"{target} does not advertise dft_shutoff; the run will use "
            "energy-only auto-shutoff", UserWarning, stacklevel=2)
    return wire

# The feature flags the v1 engine advertises via ``phsolver --capabilities``
# (engine/src/main/phsolver.cpp cmd_capabilities). Pinned here so the drift test
# fails the build the moment the engine manifest changes without the client
# being updated in lockstep — in EITHER direction.
#
# 2026-07 manifest refresh: the list had frozen at Phase 1a-1 while the engine
# shipped nine more features; both sides now advertise the full surface. Each
# name mirrors its wire surface (section = NUMERICS.md). A drift-test failure
# against an OLDER binary means that binary predates this refresh — rebuild it.
ENGINE_ADVERTISED_FEATURES = frozenset({
    "uniform_grid", "point_dipole", "gaussian_pulse", "pec", "periodic",
    "field_time", "field_snapshot",
    # Phase 1a-1 (NUMERICS.md §9-§13)
    "structures", "lossy_media", "pml", "plane_wave", "field_dft", "flux",
    # Shipped since (2026-07 manifest refresh)
    "graded_grid",      # §15 nonuniform coords (schema 1.2)
    "subpixel",         # §16 volume/tensor/tensor_full
    "mode_source",      # §18 incl. broadband modes_by_freq
    "lorentz_media",
    "multi_pole_media",  # §19 multi-pole + Drude ADE
    "pec_media",  # §10.1 PEC structure material    # §19 single-pole ADE dispersion
    "symmetry",         # §20 PEC/PMC symmetry planes
    "absorber",         # §21 adiabatic absorber boundary
    "magnetic_dipole",  # §5 PointDipole polarization Hz
    "apodization",      # §12 DFT time window
    "interval_space",   # §12 DFT spatial decimation
    "mode_port",        # §12 post-processing metadata (schema 1.16)
    # Schema 1.18 (2026-08)
    "anisotropic_media",   # §10.2 diagonal eps tensor
    "bloch",               # §22 Bloch boundaries (CPU)
    "oblique_plane_wave",  # §22 stage C constant-k tilt
    "tfsf_box",            # §13.5 closed TF/SF box (CPU)
    # Schema 1.19 (2026-08)
    "field_precision_fp16",  # §23 fp16 field storage lane
    # Schema 1.20 (2026-08)
    "dft_precision_narrow",  # §12.6 narrowed field_dft accumulator lanes
    "dft_shutoff",  # §7 CPU-only recorded-result decay estimate
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
# ``phsolver validate`` takes no ``--device`` (engine/src/main/phsolver.cpp),
# so neither can express that a shipped feature is CPU-reference-only.
# ``GpuSolver::run`` (the GPU solver in engine/src/kernels/) therefore throws
# SpecError on these features at the moment the GPU run starts, which on the
# paid cloud path is after a worker has been provisioned against a bound quote.
#
# The table below is the machine-checkable client mirror of those throws,
# so a GPU-targeted run is refused before it can cost anything. The engine
# throws stay as the backstop: this is defence in depth, not a replacement, and
# the engine remains the authority on its own spec.
#
# ``oblique_plane_wave`` (NUMERICS 22.1) needs no row of its own: an oblique
# wave's transverse phase advance IS a Bloch phase, and engine/src/core/
# resolve.cpp rejects ``angle_theta != 0`` unless the transverse axis is
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


def _has_boundary(spec: dict, condition: str) -> bool:
    boundaries = spec.get("boundaries") or {}
    return any(boundaries.get(axis) == condition for axis in ("x", "y", "z"))


def _has_source_type(spec: dict, type_name: str) -> bool:
    return any((s or {}).get("type") == type_name
               for s in spec.get("sources") or ())


def _has_cw_source(spec: dict) -> bool:
    # The engine also tests ``norm_pulse.cw``, but norm_pulse is filled from the
    # first source in wire order (engine/src/io/spec_io.cpp), so scanning every
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
        name="dft_shutoff",
        section="NUMERICS 7",
        message=("dft_shutoff is supported on the CPU reference solver only "
                 "in this release; run with --device cpu"),
        detect=lambda spec: (spec.get("run", {}).get("dft_shutoff") or 0) > 0,
    ),
    CpuOnlyFeature(
        name="bloch",
        section="NUMERICS 22",
        message=("bloch boundaries are supported on the CPU reference solver "
                 "only in this release — run with --device cpu"),
        detect=lambda spec: _has_boundary(spec, "bloch"),
    ),
    CpuOnlyFeature(
        name="tfsf_box",
        section="NUMERICS 13.5",
        message=("tfsf_box sources are supported on the CPU reference solver "
                 "only in this release — run with --device cpu"),
        detect=lambda spec: _has_source_type(spec, "tfsf_box"),
    ),
    CpuOnlyFeature(
        name="permittivity_data",
        section="NUMERICS 10.3",
        message=("permittivity_data (custom media) is supported on the CPU "
                 "reference solver only in this release — run with "
                 "--device cpu"),
        detect=_has_permittivity_data,
    ),
    CpuOnlyFeature(
        name="pmc",
        section="NUMERICS 4-PMC",
        message=("pmc boundaries are supported on the CPU reference solver "
                 "only in this release — run with --device cpu"),
        detect=lambda spec: _has_boundary(spec, "pmc"),
    ),
    CpuOnlyFeature(
        name="cw",
        section="NUMERICS 5-CW",
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
