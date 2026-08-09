"""The provider-agnostic job executor core.

``execute(spec)`` runs a wire-JSON spec through ``phsolver`` and returns a
result bundle + metrics. Every executor surface — the RunPod serverless handler,
the ``python -m photonhub.executor`` CLI, and a local reference run — calls this
one function, and it drives the solver through ``runners.phsolver`` (the same
layer ``run_local`` uses) so the JSON-lines event stream, error handling, and
"solver lies" guard are identical local and cloud.

The bundle is the shared :mod:`photonhub.bundle` format — the exact executed
``sim.json``, ``manifest.json``, and monitor ``*.bin`` files in one gzip tar
— that ``web/cache.py`` extracts and ``SimulationData`` reads.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..bundle import pack_bundle
from ..runners.phsolver import (
    EventCb,
    SolverRunError,
    find_solver,
    phsolver_run_cmd,
    run_phsolver,
)


@dataclass
class ExecResult:
    """Outcome of one ``execute`` call. Exactly one of ``bundle`` (in-memory
    tar.gz bytes) or ``bundle_path`` (a tar.gz streamed to disk) is set,
    depending on how ``execute`` was called."""
    bundle: Optional[bytes] = None
    bundle_path: Optional[Path] = None
    metrics: dict = field(default_factory=dict)   # the phsolver `done` event
    manifest: dict = field(default_factory=dict)  # parsed manifest.json


def execute(spec: dict, *, device: str = "cpu",
            timeout: Optional[float] = None, on_event: EventCb = None,
            solver_path=None, bundle_path=None) -> ExecResult:
    """Run one job to completion: a wire-JSON ``spec`` dict → ``phsolver`` → a
    result bundle.

    ``spec`` is a canonical wire dict (``Simulation.to_wire_dict()``); it is NOT
    re-validated or mutated here — the engine owns validation. ``device`` is a
    resolved role the worker can honor (``"cpu"`` / ``"gpu"``); the cloud
    ``"gpu:<target>"`` form is resolved to a plain role before it reaches a
    worker. ``on_event`` receives every JSON-lines event as it streams. With
    ``bundle_path`` the result tar.gz is streamed straight to that file (cheap
    for the large bundles a worker writes); otherwise it is returned in memory as
    ``ExecResult.bundle`` bytes (for an inline serverless response).

    Returns an :class:`ExecResult` (bundle bytes or path + the ``done`` metrics +
    the parsed manifest). Raises :class:`SolverRunError` on a missing solver, an
    error event, a timeout, a nonzero exit, or a clean exit that left no readable
    ``manifest.json`` (the executor's "solver lies" guard). The scratch dir is
    always cleaned up.
    """
    solver = find_solver(solver_path)
    if solver is None:
        raise SolverRunError(
            "phsolver binary not found: set $PHOTONHUB_SOLVER, put phsolver on "
            "PATH, or pass solver_path=")

    base = Path(tempfile.mkdtemp(prefix="photonhub-exec-"))
    try:
        out_dir = base / "out"
        out_dir.mkdir()
        spec_path = base / "sim.json"
        spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")

        cmd = phsolver_run_cmd(solver, spec_path, out_dir, device)
        metrics = run_phsolver(cmd, on_event=on_event, timeout=timeout)

        # Executor "solver lies" guard — mirror run_local: a clean exit with no
        # readable manifest.json is still a failure (single exception surface).
        try:
            manifest = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SolverRunError(
                "phsolver exited cleanly but wrote no readable manifest.json: "
                f"{e}") from e

        # Preserve the exact bytes phsolver hashed.  Consumers can therefore
        # prove that manifest.provenance.input_sha256 names the input they are
        # displaying, even when a coordinator reserializes the submitted JSON.
        # Copy only after the solver has finished so the immutable input cannot
        # be changed underneath the running process.
        shutil.copyfile(spec_path, out_dir / "sim.json")

        if bundle_path is not None:
            pack_bundle(out_dir, dest=bundle_path)
            return ExecResult(bundle_path=Path(bundle_path), metrics=metrics,
                              manifest=manifest)
        return ExecResult(bundle=pack_bundle(out_dir), metrics=metrics,
                          manifest=manifest)
    finally:
        shutil.rmtree(base, ignore_errors=True)
