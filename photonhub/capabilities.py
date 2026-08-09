"""Client/engine capability-manifest synchronization.

The pydantic models provide early structural validation, while ``phsolver
validate`` remains authoritative for grid- and device-specific constraints.
This module pins the coarse feature list emitted by ``phsolver
--capabilities`` and exposes a drift check so the client and engine cannot
silently advertise different surfaces.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Union

from .components.simulation import SUPPORTED_SCHEMA_MAJOR


# Public compatibility constant used by callers comparing the solver manifest.
SCHEMA_MAJOR = SUPPORTED_SCHEMA_MAJOR

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
    "lorentz_media",    # §19 single-pole ADE dispersion
    "symmetry",         # §20 PEC/PMC symmetry planes
    "absorber",         # §21 adiabatic absorber boundary
    "magnetic_dipole",  # §5 PointDipole polarization Hz
    "apodization",      # §12 DFT time window
    "interval_space",   # §12 DFT spatial decimation
    "mode_port",        # §12 post-processing metadata (schema 1.16)
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
