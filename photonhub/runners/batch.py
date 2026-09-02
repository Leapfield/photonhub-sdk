"""Batch and asynchronous local runs.

The master plan pulls this surface forward to Phase 1: parameter sweeps are the
dominant real usage pattern and the API *shape* binds to the cloud backend
later, so designing it now is cheap and retrofitting after Phase 3 is not. The
shape mirrors common cloud job handles / ``web.Batch`` so the local and cloud paths
read identically::

    job = ph.submit(sim)                 # returns immediately
    data = job.result()                     # blocks; RunResult

    batch = ph.Batch({"w20": sim20, "w40": sim40})
    batch_data = batch.run(path_dir="sweep") # blocks until all finish
    for name, sim_data in batch_data.items():  # successful runs only
        ...
    batch_data.errors                       # {name: exception} failures

Local backend: each simulation is an independent :func:`run_local` subprocess
(ROCm-crash isolation; identical local/cloud file protocol). ``max_workers``
multiplexes the subprocesses — on the cloud this becomes a fan-out across GPUs;
locally it defaults to 1 (serial) so a CPU box is not oversubscribed. A failed
simulation is captured per-name (partial-failure semantics) and never aborts
the rest of the batch.
"""

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from ..components import Simulation
from ..data import RunResult
from .local import SolverRunError, run_local

# A batch key becomes an output subdirectory name, so it must be filesystem
# safe — the same rule the engine applies to monitor names (components/base.py).
def _check_batch_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"batch keys must be non-empty strings, got {name!r}")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(
            f"batch key {name!r} must be usable as a directory name "
            "(no '/' or '\\', not '.' or '..')")
    return name


class Job:
    """Handle to a single asynchronous local run (the forward-compatible shape
    a cloud job handle will also satisfy). Created by :func:`submit`; the
    work runs on a daemon thread so the call returns immediately."""

    def __init__(self, fn: Callable[[], RunResult],
                 name: Optional[str] = None, job_id: Optional[str] = None,
                 cancel_event: Optional[threading.Event] = None):
        self.name = name
        # Remote jobs expose the service identifier for cancellation/resume;
        # local jobs leave this as None while retaining the same handle type.
        self.job_id = job_id
        self._cancel_event = cancel_event
        self._done = threading.Event()
        self._data: Optional[RunResult] = None
        self._exc: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, args=(fn,), name=f"photonhub-job-{name or ''}",
            daemon=True)
        self._thread.start()

    def _run(self, fn: Callable[[], RunResult]) -> None:
        try:
            self._data = fn()
        except BaseException as exc:  # captured; re-raised in result()
            self._exc = exc
        finally:
            self._done.set()

    @property
    def done(self) -> bool:
        """True once the run has finished (successfully or not)."""
        return self._done.is_set()

    def cancel(self) -> None:
        """Ask the solver to stop. No-op when the run has already finished or
        the job was created without a cancel event; a cancelled run raises
        :class:`SolverRunError` from :meth:`result`."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    def result(self, timeout: Optional[float] = None) -> RunResult:
        """Block until the run finishes and return its :class:`RunResult`,
        re-raising any :class:`SolverRunError` in the caller's thread. Raises
        :class:`TimeoutError` if ``timeout`` elapses first (the run keeps
        going; call again)."""
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"job {self.name or ''!r} not finished after {timeout} s")
        if self._exc is not None:
            raise self._exc
        assert self._data is not None
        return self._data


def submit(
    sim: Simulation,
    output_dir: Union[str, Path, None] = None,
    solver_path: Union[str, Path, None] = None,
    progress: Optional[Callable[[dict], None]] = None,
    timeout: Optional[float] = None,
    name: Optional[str] = None,
    log_file: Union[str, Path, None] = None,
    device: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Job:
    """Start ``sim`` on a background thread and return a :class:`Job` handle
    immediately; collect the result with ``job.result()``.

    Takes :func:`run_local`'s arguments (``log_file`` mirrors the engine event
    stream to disk, ``device`` selects cpu/gpu backends) EXCEPT ``quiet``: a
    background job never draws the live status line (it would fight foreground
    output and other jobs) — pass ``progress`` to consume events instead.
    ``cancel_event`` (or :meth:`Job.cancel`) terminates the solver early;
    the job then raises :class:`SolverRunError` from ``result()``."""
    ev = cancel_event if cancel_event is not None else threading.Event()
    return Job(
        lambda: run_local(sim, output_dir=output_dir, solver_path=solver_path,
                          progress=progress, timeout=timeout, quiet=True,
                          log_file=log_file, device=device, cancel_event=ev),
        name=name, cancel_event=ev)


class BatchResults:
    """Results of a :meth:`Batch.run`. Dict-like over the **successful** runs
    (``batch_data[name]`` / ``items()`` / iteration), with failures captured in
    :attr:`errors`. Local failures are :class:`SolverRunError`; a remote batch
    may also retain a resumable timeout carrying its service job id. Indexing a
    failed name re-raises that exception; indexing an unknown name raises
    ``KeyError``."""

    def __init__(self, results: Dict[str, RunResult],
                 errors: Dict[str, Exception], path: Path,
                 names: List[str]):
        self._results = results
        self._errors = errors
        self.path = path
        self._names = list(names)

    def __getitem__(self, name: str) -> RunResult:
        if name in self._results:
            return self._results[name]
        if name in self._errors:
            raise self._errors[name]
        raise KeyError(
            f"unknown batch entry {name!r}; entries: {self._names}")

    def __contains__(self, name: str) -> bool:
        return name in self._results

    def __iter__(self) -> Iterator[str]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def items(self) -> Iterator[Tuple[str, RunResult]]:
        """Iterate (name, RunResult) over the successful runs only."""
        return iter(self._results.items())

    def keys(self) -> List[str]:
        """Names of the successful runs."""
        return list(self._results)

    @property
    def names(self) -> List[str]:
        """Every submitted name, in submission order (success or failure)."""
        return list(self._names)

    @property
    def errors(self) -> Dict[str, Exception]:
        """Per-name exceptions for failed or resumably timed-out runs."""
        return dict(self._errors)

    @property
    def succeeded(self) -> List[str]:
        return list(self._results)

    @property
    def failed(self) -> List[str]:
        return list(self._errors)

    def __repr__(self) -> str:
        return (f"BatchResults({str(self.path)!r}, "
                f"succeeded={self.succeeded}, failed={self.failed})")


class Batch:
    """A named collection of simulations run together. Keys are stable across
    the local and (future) cloud backends and become output subdirectory
    names, so they must be filesystem-safe."""

    def __init__(self, simulations: Mapping[str, Simulation]):
        if not simulations:
            raise ValueError("Batch needs at least one simulation")
        validated: Dict[str, Simulation] = {}
        for name, sim in simulations.items():
            _check_batch_name(name)
            if not isinstance(sim, Simulation):
                raise TypeError(
                    f"batch entry {name!r} is {type(sim).__name__}, "
                    "expected a Simulation")
            validated[name] = sim
        self.simulations = validated

    def quote(self, **kwargs):
        """Per-name :class:`~photonhub.cost.CostEstimate` plus the batch total
        (the plan's per-batch upfront estimate). Returns
        ``(per_sim: dict, total_usd: float)``."""
        per_sim = {name: sim.cost_estimate(**kwargs)
                   for name, sim in self.simulations.items()}
        total = sum(e.usd for e in per_sim.values())
        return per_sim, total

    def run(
        self,
        path_dir: Union[str, Path, None] = None,
        solver_path: Union[str, Path, None] = None,
        max_workers: int = 1,
        progress: Optional[Callable[[str, dict], None]] = None,
        timeout: Optional[float] = None,
        device: Optional[str] = None,
    ) -> BatchResults:
        """Run every simulation, writing ``<path_dir>/<name>/`` per entry, and
        block until all finish. ``max_workers`` runs that many concurrently
        (default 1 = serial). ``progress`` (if given) receives ``(name,
        event)`` for each solver event; the per-run live status line is
        always suppressed (concurrent entries would interleave).
        ``device`` selects the backend for every entry, as in
        :func:`run_local`. Per-simulation failures are captured in
        the returned :attr:`BatchResults.errors`, not raised."""
        base = (Path(path_dir) if path_dir is not None
                else Path(tempfile.mkdtemp(prefix="photonhub-batch-")))
        base.mkdir(parents=True, exist_ok=True)

        def _run_one(name: str, sim: Simulation) -> RunResult:
            cb = ((lambda ev: progress(name, ev)) if progress is not None
                  else None)
            # quiet: concurrent entries would interleave their live status
            # lines; the batch-level (name, event) callback is the surface.
            return run_local(sim, output_dir=base / name,
                             solver_path=solver_path, progress=cb,
                             timeout=timeout, quiet=True, device=device)

        results: Dict[str, RunResult] = {}
        errors: Dict[str, SolverRunError] = {}
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
            futs = {ex.submit(_run_one, n, s): n
                    for n, s in self.simulations.items()}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    results[name] = fut.result()
                except SolverRunError as exc:
                    # Partial-failure semantics: one bad sim never sinks the
                    # batch. Non-SolverRunError (programming bugs) still
                    # propagate and abort the run.
                    errors[name] = exc
        return BatchResults(results, errors, base, list(self.simulations))
