"""Run phsolver as a local subprocess.

The solver streams JSON-lines events on stdout (NUMERICS.md section 7), e.g.
``{"event": "progress", "step": 100, ...}`` and on failure
``{"event": "error", "reason": "divergence"}``; outputs land in the output
directory as monitor binaries plus ``manifest.json`` (photonhub.data).
"""

import json
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Callable, Optional, Union

from ..components import Simulation
from ..data import RunResult
from .phsolver import (
    SolverRunError,
    find_solver,
    phsolver_run_cmd,
    run_phsolver,
)
from .progress import default_renderer

# The phsolver process layer (discovery, command grammar, subprocess + event
# contract) lives in .phsolver, shared with the cloud executor. SolverRunError
# and find_solver are re-exported here so the established imports
# `from photonhub.runners.local import SolverRunError / find_solver` keep working.
# The --device grammar (incl. multi-GPU gpu:all / gpu:N,M,...) lives in
# .phsolver.device_args, the single definition shared with the cloud executor.
__all__ = ["run_local", "find_solver", "SolverRunError"]


class _DecayTrail:
    """Watch the engine's JSON-lines events for a field-decay PLATEAU.

    A pulsed current whose spectrum has nonzero DC content (e.g. a PointDipole
    GaussianPulse with fwidth >~ 0.25*freq0) deposits net charge on its source
    cells; the resulting static field is not a wave, no PML drains it on run
    timescales, and it anchors the engine's NUMERICS section-7 energy-decay
    ratio at a level that can sit orders of magnitude above ``run.shutoff``
    (measured: fwidth = 0.3*f0 pins decay at ~1.2e-2 vs the 1e-5 default —
    the run silently rides its full step cap, ~25x the physical duration).
    Perfectly trapped modes (periodic/PEC scenes, quasi-2D) produce the same
    flat trace, and from the scalar energy alone the two cannot be told
    apart, so the ENGINE must not stop early on its own — but the user
    deserves to know their cap-length run was flat. This advisory is
    diagnosis-only: it changes no run semantics."""

    # A trace is "flat" when the decay improved by less than 1 % across the
    # trailing half of the post-source progress checks, with at least 20 such
    # checks — thousands of steps at every progress cadence, far beyond any
    # transient dip.
    MIN_CHECKS = 20
    IMPROVE = 0.01

    def __init__(self):
        self.shutoff = None
        self.decays = []
        self.done = None

    def feed(self, event: dict) -> None:
        kind = event.get("event")
        if kind == "start":
            self.shutoff = event.get("shutoff")
        elif kind == "progress":
            d = event.get("field_decay")
            if d is not None and event.get("phase") == "ringdown":
                self.decays.append(float(d))
        elif kind == "done":
            self.done = event

    def plateau_advisory(self):
        if not self.done or self.done.get("shut_off"):
            return None
        shutoff = self.shutoff or 0.0
        if shutoff <= 0.0:
            return None          # auto-shutoff disabled: cap runs are asked-for
        if len(self.decays) < self.MIN_CHECKS:
            return None
        tail = self.decays[len(self.decays) // 2:]
        first, last = tail[0], tail[-1]
        if not (last > 10.0 * shutoff and first > 0.0):
            return None
        if last < (1.0 - self.IMPROVE) * first:
            return None          # still draining; the cap was simply short
        return (
            f"the field decay plateaued at {last:.2e} and never reached "
            f"run.shutoff = {shutoff:g}, so the run used its full step cap "
            f"({self.done.get('steps_run')} steps). Likely causes: a static "
            "charge residue from a broadband source (narrow the pulse, e.g. "
            "fwidth <= 0.2*freq0) or a trapped/lossless mode (set an explicit "
            "run_time_s / n_steps, or shutoff=0 to silence this).")


def run_local(
    sim: Simulation,
    output_dir: Union[str, Path, None] = None,
    solver_path: Union[str, Path, None] = None,
    progress: Optional[Callable[[dict], None]] = None,
    timeout: Optional[float] = None,
    device: Union[str, None] = None,
    quiet: bool = False,
    log_file: Union[str, Path, None] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunResult:
    """Run ``phsolver run sim.json --output <dir>`` and load the results.

    ``progress`` (if given) receives every parsed JSON-lines event dict as it
    arrives, and takes over the human surface. When ``progress`` is ``None`` a
    default live status line (field decay vs. the shutoff threshold, phase,
    stability, throughput, ETA) is rendered to stderr; pass ``quiet=True`` to
    silence it. Either way the child runs with ``--progress none`` so Python is
    the only thing drawing the status line.
    ``log_file`` (if given) is forwarded to ``phsolver --log-file`` so the engine
    mirrors the full JSON-lines event stream (start/progress/done/error — field
    decay, phase, stability, throughput) to that path *as it runs*. The engine
    writes it directly, so the record survives even if this process is killed or
    crashes; it is independent of ``progress``/``quiet``. The parent directory is
    created if needed.
    ``timeout`` (seconds) kills the solver and raises. Outputs go to
    ``output_dir`` (created if needed) or a fresh persistent temp directory.
    ``device`` selects the backend — ``"cpu"`` (default when ``None``), ``"gpu"``,
    ``"gpu:N"``, ``"gpu:all"``, or ``"gpu:N,M,..."`` (multi-GPU z-decomposition;
    see ``engine/docs/multi-gpu-decomposition.md``); it is passed to ``phsolver
    --device``. Selection is vendor-neutral: ``"gpu"`` runs on whichever GPU the
    ``phsolver`` binary was built for — AMD (HIP/ROCm) or NVIDIA (native CUDA) —
    so running on NVIDIA is just a matter of pointing ``$PHOTONHUB_SOLVER`` at a
    CUDA build (see ``docs/nvidia-gpu.md``). GPU↔CPU equivalence follows the
    tolerances in NUMERICS §8 (including ``ModeSource`` §18), not a blanket
    bit-exact guarantee. Hardware records are dated snapshots: the then-current
    suite passed 29/29 on an NVIDIA RTX A4000 on 2026-06-27 and 47/47 on an AMD
    MI300X on 2026-07-12. Those counts do not attest a newer source inventory;
    current refresh status is tracked in ``GPU_TODO.md`` and
    ``MI300X_VERIFY.md``. CI compiles both GPU paths; hardware equivalence needs
    the ``nvidia-equivalence`` workflow or a GPU box.
    ``cancel_event`` is an optional :class:`threading.Event`; setting it
    terminates the solver subprocess and raises :class:`SolverRunError`.  It is
    used by interactive callers such as the desktop Stop button.
    """
    solver = find_solver(solver_path)
    if solver is None:
        raise SolverRunError(
            "phsolver engine binary not found. Local runs need the engine; "
            "pip installs the Python client only. Either run on the cloud "
            "instead (ph.cloud.run_quoted(sim, max_usd=...) — no engine "
            "needed), or point this client at an engine: pass solver_path=, "
            "set $PHOTONHUB_SOLVER, or put phsolver on PATH. Invited beta "
            "participants receive a standalone headless solver archive "
            "(https://leapfield.app/docs/get-started/headless-solver/); the "
            "copy inside the desktop app is locked to that app. Developers "
            "with the source tree: "
            "cmake -S engine -B build && cmake --build build"
        )

    for index, axis, band in sim.point_sources_in_boundary_layers():
        center = tuple(sim.sources[index].center_um)
        warnings.warn(
            f"sources[{index}] (point_dipole at {center} um) lies inside the "
            f"boundary layers on axis '{axis}' ({band:g} um thick on each "
            "side): the engine will run it, but the boundary absorbs the "
            "source in place and the recorded spectra are physically "
            "meaningless. Move the source into the interior or thin the "
            "boundary layers.",
            stacklevel=2,
        )

    out_dir = Path(output_dir) if output_dir is not None else Path(
        tempfile.mkdtemp(prefix="photonhub-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "sim.json"
    # to_wire_json (not a raw model_dump_json) so the canonical wire rules
    # apply — notably the omission of an unset pml_num_layers, which keeps
    # Phase-0-style specs consumable by schema-1.0 phsolver binaries.
    spec_path.write_text(sim.to_wire_json() + "\n", encoding="utf-8")

    log_path = None
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = phsolver_run_cmd(solver, spec_path, out_dir, device, log_file=log_path)

    # Stream events through the shared phsolver runner — the single source of
    # truth for the subprocess + wire-event + error/timeout contract (also used
    # by the cloud executor). A caller-supplied progress callback owns the human
    # surface; otherwise render a default live status line unless silenced. The
    # child runs --progress none regardless, so Python is the only human surface.
    renderer = default_renderer() if (progress is None and not quiet) else None
    decay_trail = _DecayTrail()

    def _on_event(event: dict) -> None:
        decay_trail.feed(event)
        if progress is not None:
            progress(event)
        elif renderer is not None:
            renderer(event)

    run_phsolver(cmd, on_event=_on_event, timeout=timeout,
                 cancel_event=cancel_event)
    advisory = decay_trail.plateau_advisory()
    if advisory:
        warnings.warn(advisory, stacklevel=2)

    # "Solver lies" guard: a clean exit with a missing/malformed manifest or
    # .bin is still a solver failure — surface it as SolverRunError so callers
    # have a single exception surface, not a raw FileNotFoundError/ValueError.
    try:
        return RunResult(out_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        raise SolverRunError(
            f"phsolver exited cleanly but its outputs are unreadable: {e}") from e
