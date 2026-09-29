"""Cloud run entry points, the prime directive: ``ph.cloud.submit`` returns the
**same** :class:`~photonhub.runners.batch.Job` as the local path, so
``job = ph.cloud.submit(sim); data = job.result()`` reads identically whether
local or cloud, and a server-side failure surfaces as the same
:class:`SolverRunError`.

Where the two differ, it is to keep the spend bounded: every submission is
quoted first and refused above ``max_usd`` (default $5) unless the caller opts
out with ``max_usd=None``; ``job.cancel()`` asks the service to cancel the
job; and ``wait_timeout_s`` bounds only the local wait, never the job.
"""

from __future__ import annotations

import functools
import math
import re
import threading
import time
import warnings
from typing import Callable, Optional

from .._compat import caller_stacklevel
from ..bundle import BundleError
from ..capabilities import check_device_support
from ..data import RunResult
from ..runners.batch import Job
from ..runners.local import SolverRunError
from . import cache
from ._ids import validate_job_id
from .client import HttpClient
from .config import CloudConfig, CloudError, get_config

ProgressCb = Optional[Callable[[dict], None]]

# A curated-GPU id in `device="gpu:<id>"` (ids come from `ph.cloud.gpus()`). Lowercase
# slug, distinct from the local path's numeric `gpu:N` device index; legacy or
# development catalogs may expose other ids.
_GPU_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")


class _DefaultCeiling(float):
    """The default ``max_usd``, a float that remembers it was not given, so an
    explicit ceiling passed beside a ``quote_id`` (which is bound without
    quoting again, so no ceiling could be checked) is refused, not dropped."""

    def __repr__(self) -> str:
        return float.__repr__(self)


#: The dollar ceiling every paid entry point applies unless told otherwise:
#: the beta's credit.
DEFAULT_MAX_USD = _DefaultCeiling(5.0)


class CloudJobTimeout(TimeoutError):
    """The client stopped waiting (``wait_timeout_s``) while the service job
    is still running, and billing."""

    def __init__(self, job_id: str, timeout: float):
        self.job_id = job_id
        self.wait_timeout_s = timeout
        self.timeout = timeout   # the name before wait_timeout_s
        super().__init__(
            f"stopped waiting for cloud job {job_id!r} after {timeout} s; the "
            "job is still running and billing: resume it with "
            "ph.cloud.resume(job_id) or ask the service to cancel it with "
            "ph.cloud.cancel(job_id)")


def _poll_timeout(timeout: Optional[float]) -> Optional[float]:
    if timeout is None:
        return None
    if isinstance(timeout, bool):
        raise ValueError(
            "wait_timeout_s must be a finite non-negative number or None")
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "wait_timeout_s must be a finite non-negative number or None") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "wait_timeout_s must be a finite non-negative number or None")
    return value


def _renamed_timeout(fn):
    """Accept ``timeout=``, the old name of ``wait_timeout_s=``, with a
    warning that says what the value never did: stop the service job. The
    package-wide switch for renamed keywords (``photonhub._compat``) is off,
    but this one warns regardless, because the old name read like a limit on
    the spend. The signature keeps the new name."""
    label = f"ph.cloud.{fn.__qualname__}()"

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "timeout" in kwargs:
            if "wait_timeout_s" in kwargs:
                raise TypeError(
                    f"{label} got both 'timeout' and 'wait_timeout_s'; "
                    "'wait_timeout_s' is the current spelling")
            warnings.warn(
                f"{label}: 'timeout' was renamed to 'wait_timeout_s'. It bounds "
                "how long this client waits, nothing more: when it runs out the "
                "service job keeps running and billing, so ask the service to "
                "cancel it with job.cancel() or ph.cloud.cancel(job_id). The old "
                "spelling will be removed in a future release.",
                FutureWarning, stacklevel=caller_stacklevel())
            kwargs["wait_timeout_s"] = kwargs.pop("timeout")
        return fn(*args, **kwargs)

    wrapper.__legacy_keywords__ = {"timeout": "wait_timeout_s"}  # type: ignore[attr-defined]
    return wrapper


def _submitted_job_id(response: object) -> str:
    try:
        candidate = response["job_id"]  # type: ignore[index]
        return validate_job_id(candidate)
    except (KeyError, TypeError, ValueError) as exc:
        raise CloudError("service returned an invalid job_id") from exc


def _validate_quote_id(quote_id: Optional[str]) -> Optional[str]:
    """Validate an opaque server quote id before placing it in a submission."""
    if quote_id is None:
        return None
    if not isinstance(quote_id, str) or not quote_id.strip():
        raise ValueError("quote_id must be a non-empty string or None")
    return quote_id


def _validate_web_device(device: Optional[str]) -> Optional[str]:
    """Reject a bad ``device`` before submitting, so a typo fails fast client-side
    with a clear message (mirrors the local runner). The cloud grammar is
    ``cpu`` / ``gpu`` / ``gpu:<id>`` where ``<id>`` is a curated GPU from
    ``ph.gpus()``, the platform resolves it to a provider + a plain ``gpu`` on
    the worker."""
    if device is None:
        return None
    d = device.strip()
    if d in ("cpu", "gpu"):
        return d
    if d.startswith("gpu:") and _GPU_ID.fullmatch(d[4:]):
        return d
    raise SolverRunError(
        f"invalid device {device!r}: expected 'cpu', 'gpu', or 'gpu:<id>' "
        "(an id from ph.gpus())")


# Said by every cloud entry point that refuses a spec on capability grounds, so
# the caller knows the refusal cost nothing.
_NO_JOB_SUBMITTED = "No job was submitted and no quote was bound."


def _check_cloud_device_support(sim, device, *, entry=None) -> dict:
    """Build a cloud wire spec and refuse unsupported required features.

    The optional DFT guard is omitted with a warning before this check.
    ``GpuSolver::run`` throws ``SpecError`` on the other features in
    ``capabilities.CPU_ONLY_FEATURES``, but only once a worker is already
    running against a bound quote. Every cloud entry point therefore checks the
    same table client-side first. An unset device is treated as possibly-GPU:
    the service picks the backend, and the client cannot see that policy.

    ``entry`` names the batch entry at fault, since a batch preflights every
    name before submitting any of them.
    """
    note = _NO_JOB_SUBMITTED
    if entry is not None:
        note = f"Batch entry {entry!r} was rejected. {note}"
    from ..capabilities import (omit_unsupported_dft_shutoff,
                                warn_implicit_dft_shutoff_unsupported)
    wire = sim.to_wire_dict()
    if (sim.run.dft_shutoff is None and sim.run.shutoff > 0 and
            any(m.type in ("field_dft", "flux") for m in sim.monitors)):
        warn_implicit_dft_shutoff_unsupported("Cloud solver")
    omit_unsupported_dft_shutoff(wire, "Cloud solver")
    check_device_support(wire, device, unset_may_be_gpu=True, note=note)
    return wire


def _poll_and_download(http: HttpClient, cfg: CloudConfig, job_id: str, *,
                       progress: ProgressCb, timeout: Optional[float],
                       stop: Optional[threading.Event] = None) -> object:
    timeout = _poll_timeout(timeout)
    deadline = (time.monotonic() + timeout) if timeout is not None else None
    interval = cfg.poll_interval_s
    while True:
        # Set once the service accepted a cancel (Job.cancel): stop waiting
        # for a state change that only confirms it.
        if stop is not None and stop.is_set():
            raise SolverRunError(f"cloud job {job_id} was cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise CloudJobTimeout(job_id, timeout)
        try:
            st = http.get_job(job_id, deadline=deadline)
        except TimeoutError as exc:
            raise CloudJobTimeout(job_id, timeout) from exc
        if deadline is not None and time.monotonic() >= deadline:
            raise CloudJobTimeout(job_id, timeout)
        if not isinstance(st, dict) or not isinstance(st.get("state"), str):
            raise CloudError(
                f"service returned an invalid status for cloud job {job_id}")
        state = st["state"]
        if state not in (
                "queued", "provisioning", "running", "succeeded", "failed",
                "cancelled"):
            raise CloudError(
                f"service returned unknown state {state!r} for cloud job "
                f"{job_id}", job_id=job_id)
        if progress and st.get("progress"):
            progress(st["progress"])
            if deadline is not None and time.monotonic() >= deadline:
                raise CloudJobTimeout(job_id, timeout)
        if state == "succeeded":
            break
        if state == "failed":
            err = st.get("error") or {}
            if not isinstance(err, dict):
                err = {"reason": "invalid service error response"}
            reason = err.get("reason")
            if not isinstance(reason, str) or not reason:
                reason = "unknown"
            else:
                reason = reason[:1024]
            stderr_tail = err.get("stderr_tail")
            if not isinstance(stderr_tail, str):
                stderr_tail = None
            elif len(stderr_tail) > 65_536:
                stderr_tail = stderr_tail[-65_536:]
            raise SolverRunError(
                f"cloud job {job_id} failed: {reason}",
                stderr_tail=stderr_tail)
        if state == "cancelled":
            raise SolverRunError(f"cloud job {job_id} was cancelled")
        sleep_for = interval
        if deadline is not None:
            sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
        if sleep_for > 0:
            if stop is not None:
                stop.wait(sleep_for)
            else:
                time.sleep(sleep_for)
        interval = min(interval * 1.5, cfg.poll_backoff_max_s)
    try:
        return cache.download_bundle(http, cfg, job_id)
    except (BundleError, OSError) as exc:
        raise CloudError(
            f"cloud job {job_id} returned an invalid result bundle: {exc}",
            job_id=job_id) from exc


def _stored_simulation(cfg: CloudConfig, job_id: str):
    """The simulation submitted as ``job_id``, recovered from the local spec
    cache, or None when no usable spec was kept.

    The engine normalizes every ``field_dft`` phasor and flux per unit
    amplitude of the first wire-order source (NUMERICS.md section 12) and the
    reader multiplies ``A0`` back in only when it knows the simulation. The
    submitting calls hand theirs over; a resume is given an id alone, so it
    reads the spec the client stored when it submitted that job. Without one
    the arrays keep the engine's values and say so in their ``normalization``
    attr, exactly as before this store existed.

    The client state stored beside it (the user-frame origin, a symmetry
    fold, declared ports) goes back on the recovered simulation, so the
    resumed result also reads in the submitting call's frame; a record that
    does not match the spec warns and restores nothing.
    """
    path = cache.stored_spec(cfg, job_id)
    if path is None:
        return None
    from ..components import Simulation
    from ..components import frame as _frame

    try:
        sim = Simulation.from_file(path)
    except (OSError, ValueError) as exc:
        warnings.warn(
            f"could not read the cached spec of cloud job {job_id} ({exc}); "
            "its frequency-domain arrays keep the engine's unit-amplitude "
            "normalization", UserWarning, stacklevel=caller_stacklevel())
        return None
    state_path = cache.stored_client_state(cfg, job_id)
    if state_path is not None:
        state = _frame.read_client_state(state_path)
        if state is not None:
            _frame.restore_client_state(sim, state, where=str(state_path))
    return sim


def _store_submission(cfg: CloudConfig, job_id: str, sim, wire: dict) -> None:
    """Keep exactly the document the service accepted, and the client state
    the document does not carry, so a later resume of this job restores the
    amplitude and the frame this call is about to restore.

    It runs after the paid submission and before the caller holds a handle,
    so nothing it raises may escape: a lost handle would leave a billing job
    the caller cannot name. Any failure warns with the job id instead."""
    stored = None
    try:
        stored = cache.store_spec(cfg, job_id, wire)
        if stored is not None:
            cache.store_client_state(cfg, job_id, sim, wire=wire)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        lost = ("its frame (origin, symmetry fold, ports)" if stored is not None
                else "its absolute units and its frame")
        try:
            warnings.warn(
                f"cloud job {job_id} was submitted, but storing its "
                f"{'client state' if stored is not None else 'spec'} locally "
                f"failed ({type(exc).__name__}: {exc}); pass simulation= to a "
                f"later ph.cloud.resume({job_id!r}) to keep {lost}",
                UserWarning, stacklevel=caller_stacklevel())
        except Exception:  # noqa: BLE001, S110 - warnings turned into errors (-W error)
            pass


def _finish_cloud_job(http: HttpClient, cfg: CloudConfig, job_id: str, *,
                      progress: ProgressCb = None,
                      timeout: Optional[float] = None,
                      simulation=None,
                      stop: Optional[threading.Event] = None) -> RunResult:
    # Every path that owns the Simulation passes it; a resume knows only the
    # id, so recover it from the spec stored at submit time. Both then restore
    # the first source's amplitude, and one paid job reads the same absolute
    # numbers whichever way it is collected.
    if simulation is None:
        simulation = _stored_simulation(cfg, job_id)
    # A validated, sealed cache entry is exactly what a successful poll +
    # download would produce, so an already-fetched result loads without the
    # service round-trip. This keeps a paid, downloaded run loadable through
    # ph.cloud.resume(job_id) offline and after service-side job expiry.
    cached = cache.completed_result(cfg, job_id)
    if cached is not None:
        try:
            return RunResult(cached, simulation=simulation)
        except (OSError, ValueError, KeyError, TypeError):
            # Corruption the structural seal cannot see: drop the entry and
            # re-fetch this already-paid job from the service.
            cache.invalidate(cfg, job_id)
    try:
        bundle_dir = _poll_and_download(
            http, cfg, job_id, progress=progress, timeout=timeout, stop=stop)
    except (SolverRunError, CloudJobTimeout):
        raise
    except CloudError as exc:
        if exc.job_id is None:
            exc.job_id = job_id
        raise
    try:
        return RunResult(bundle_dir, simulation=simulation)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Do not preserve a completion marker for outputs the public reader
        # rejects; resume can safely re-fetch this already-paid job.
        cache.invalidate(cfg, job_id)
        raise CloudError(
            f"cloud job {job_id} returned unreadable outputs: {exc}",
            job_id=job_id) from exc


def _bind_quote(http: HttpClient, sim, *, device, solver,
                quote_id: Optional[str], max_usd: Optional[float],
                preflight_wire: Optional[dict] = None):
    """``(device, solver, quote_id)`` to submit with. A caller's ``quote_id``
    is an accepted quote and is bound as given, and ``max_usd=None`` is the
    explicit unquoted opt-out; otherwise the job is quoted first, on the same
    client it is submitted with, and refused above ``max_usd`` or the
    account's available balance (:func:`~photonhub.cloud.preflight`). Every
    refusal happens before any request."""
    from .actions import _finite_non_negative, _preflight

    if max_usd is not None:
        _finite_non_negative(max_usd, "max_usd")
    if quote_id is not None:
        if max_usd is not None and not isinstance(max_usd, _DefaultCeiling):
            raise ValueError(
                "pass quote_id= or max_usd=, not both: a quote_id is bound as "
                "given, without quoting again, so no ceiling can be checked "
                "against it; check its price with ph.cloud.preflight(sim, "
                "max_usd=...) and pass the quote_id it returns")
        return device, solver, quote_id
    if max_usd is None:
        return device, solver, None
    if device is None:
        raise ValueError(
            "a quoted submission needs a device: leave device at its default "
            "'gpu' or name one, or pass max_usd=None to submit unquoted and "
            "let the service choose")
    accepted = _preflight(lambda: http, sim, device=device, solver=solver,
                          max_usd=max_usd, preflight_wire=preflight_wire)
    return accepted.device, accepted.solver, accepted.quote_id


def _quoted(sim, *, device, solver, max_usd):
    """The accepted quote of :func:`run_quoted` and :func:`submit_quoted`,
    and the wire document it priced, which is the one they submit."""
    from . import actions

    device = _validate_web_device(device)
    if device is None:
        raise ValueError("cloud preflight requires an explicit device")
    wire = _check_cloud_device_support(sim, device)
    accepted = actions._preflight(
        lambda: actions.HttpClient(get_config()), sim, device=device,
        solver=solver, max_usd=max_usd, preflight_wire=wire)
    return accepted, wire


def _cancel_service_job(cfg: CloudConfig, job_id: str,
                        stop: threading.Event):
    """Ask the service to cancel ``job_id`` (what stops its spend). Returns
    ``(record, stopped)``: the service's reply, and whether it says the job
    is cancelled. Only then does the local wait end; a job that finished just
    before the cancel arrived is still collected, and a refusal raises with
    the wait untouched, since the job is still running."""
    record = HttpClient(cfg).cancel_job(job_id)
    stopped = isinstance(record, dict) and record.get("state") == "cancelled"
    if stopped:
        stop.set()
    return record, stopped


def _cloud_run(sim, *, name=None, device=None, solver=None,
               progress: ProgressCb = None,
               timeout: Optional[float] = None,
               quote_id: Optional[str] = None,
               cfg: Optional[CloudConfig] = None,
               max_usd: Optional[float] = None,
               preflight_wire: Optional[dict] = None) -> RunResult:
    device = _validate_web_device(device)
    wire = preflight_wire if preflight_wire is not None else _check_cloud_device_support(sim, device)
    sim.check_runnable()
    timeout = _poll_timeout(timeout)
    quote_id = _validate_quote_id(quote_id)
    cfg = cfg or get_config()
    http = HttpClient(cfg)
    device, solver, quote_id = _bind_quote(
        http, sim, device=device, solver=solver, quote_id=quote_id,
        max_usd=max_usd, preflight_wire=wire)
    resp = http.submit_job(
        wire, name=name, device=device, solver=solver, quote_id=quote_id)
    job_id = _submitted_job_id(resp)
    _store_submission(cfg, job_id, sim, wire)
    return _finish_cloud_job(
        http, cfg, job_id, progress=progress, timeout=timeout, simulation=sim)


@_renamed_timeout
def run(sim, *, max_usd: Optional[float] = DEFAULT_MAX_USD, name=None,
        device: Optional[str] = "gpu", solver=None, progress: ProgressCb = None,
        wait_timeout_s: Optional[float] = None,
        quote_id: Optional[str] = None) -> RunResult:
    """Submit ``sim`` to the cloud under a dollar ceiling and block until its
    result is ready. Returns a :class:`RunResult`; raises
    :class:`SolverRunError` if the run fails, :class:`CloudError` for
    transport, auth or result-transfer problems and for a refused quote.

    **Spend-safe by default.** The job is quoted first and submitted bound to
    that quote only when it is within ``max_usd`` (default $5) and the
    account's available balance, and when the quote's grid matches the one
    ``sim`` realizes here; otherwise nothing is submitted (see
    :func:`~photonhub.cloud.preflight`). Pass the ``quote_id`` of a quote you
    already accepted (from :func:`~photonhub.cloud.preflight` or
    :func:`~photonhub.cloud.estimate`) to bind it as given, without quoting
    again. ``max_usd=None`` is the explicit opt-out: an unquoted submission
    with no ceiling.

    ``wait_timeout_s`` bounds only how long this call waits. When it runs out
    it raises :class:`CloudJobTimeout` and **the service job keeps running,
    and billing**: ask the service to cancel it with
    ``ph.cloud.cancel(job_id)`` or collect it
    with :func:`resume`. ``timeout`` is its old name and warns. ``device``
    defaults to ``"gpu"`` as in every cloud call; ``"gpu:<id>"`` picks a GPU
    from :func:`~photonhub.cloud.gpus`. ``solver`` pins a solver version or
    commit (default: latest). The check covers the quoted grid only: the
    solver image that runs the job is versioned separately from the service
    that quotes it."""
    return _cloud_run(sim, name=name, device=device, solver=solver,
                      progress=progress, timeout=wait_timeout_s,
                      quote_id=quote_id, max_usd=max_usd)


@_renamed_timeout
def run_quoted(sim, *, max_usd: float = DEFAULT_MAX_USD, name=None,
               device: str = "gpu", solver=None, progress: ProgressCb = None,
               wait_timeout_s: Optional[float] = None) -> RunResult:
    """Preflight and submit one quote-bound job under a hard dollar ceiling.

    :func:`run` does the same by default; this form has no unquoted opt-out
    (``max_usd`` must be a number) and takes no caller quote: it verifies a finite server quote, that the quote's
    ``cells_per_axis`` and ``dt_s`` match the grid ``sim`` realizes locally
    when the service reports them (a quote that omits them warns and is still
    submitted), ``max_usd`` (default $5), and the account's *available*
    balance before binding that quote id to the submission.  A quote for a
    different grid,
    from a service whose SDK resolves the document differently, is refused
    with :class:`CloudError`; the solver image that runs the job is versioned
    separately, so this checks the quote, not the run.  The service remains
    the final authority for quote expiry and concurrent balance changes.
    """
    accepted, wire = _quoted(sim, device=device, solver=solver, max_usd=max_usd)
    return _cloud_run(
        sim, name=name, device=accepted.device, solver=accepted.solver,
        progress=progress, timeout=wait_timeout_s, quote_id=accepted.quote_id,
        preflight_wire=wire)


@_renamed_timeout
def submit(sim, *, max_usd: Optional[float] = DEFAULT_MAX_USD, name=None,
           device: Optional[str] = "gpu", solver=None,
           progress: ProgressCb = None,
           wait_timeout_s: Optional[float] = None,
           quote_id: Optional[str] = None,
           _preflighted_wire: Optional[dict] = None) -> Job:
    """Submit ``sim`` and return the same :class:`Job` handle type as local
    ``ph.submit`` once the service accepts the job. Polling and download
    continue in the background; collect with ``job.result()``.

    Spend-safe by default, exactly as :func:`run`: quoted first and refused
    above ``max_usd`` (default $5) or the available balance; a ``quote_id``
    you already accepted is bound as given; ``max_usd=None`` is the explicit
    unquoted opt-out. ``job.cancel()`` asks the service to cancel the job and
    returns its reply; a service that refuses raises ``CloudError``, and the
    job keeps running.
    ``wait_timeout_s`` bounds only how long ``job.result()`` polls: the
    service job keeps running, and billing, after it runs out.
    ``timeout`` is its old name and warns."""
    device = _validate_web_device(device)
    wire = (_preflighted_wire if _preflighted_wire is not None else
            _check_cloud_device_support(sim, device))
    sim.check_runnable()
    timeout = _poll_timeout(wait_timeout_s)
    quote_id = _validate_quote_id(quote_id)
    cfg = get_config()
    http = HttpClient(cfg)
    device, solver, quote_id = _bind_quote(
        http, sim, device=device, solver=solver, quote_id=quote_id,
        max_usd=max_usd, preflight_wire=wire)
    # Submit before returning so the handle owns the real service id. Transport
    # and authentication failures therefore surface synchronously, before a
    # background thread can hide them.
    resp = http.submit_job(
        wire, name=name, device=device, solver=solver, quote_id=quote_id)
    job_id = _submitted_job_id(resp)
    _store_submission(cfg, job_id, sim, wire)
    stop = threading.Event()
    return Job(
        lambda: _finish_cloud_job(
            http, cfg, job_id, progress=progress, timeout=timeout,
            simulation=sim, stop=stop),
        name=name, job_id=job_id,
        on_cancel=lambda: _cancel_service_job(cfg, job_id, stop))


@_renamed_timeout
def submit_quoted(sim, *, max_usd: float = DEFAULT_MAX_USD, name=None,
                  device: str = "gpu", solver=None,
                  progress: ProgressCb = None,
                  wait_timeout_s: Optional[float] = None) -> Job:
    """Async form of :func:`run_quoted`, returning the accepted service id."""
    accepted, wire = _quoted(sim, device=device, solver=solver, max_usd=max_usd)
    return submit(
        sim, name=name, device=accepted.device, solver=accepted.solver,
        progress=progress, wait_timeout_s=wait_timeout_s,
        quote_id=accepted.quote_id, _preflighted_wire=wire)


@_renamed_timeout
def resume(job_id: str, *, simulation=None, progress: ProgressCb = None,
           wait_timeout_s: Optional[float] = None) -> Job:
    """Return a handle that resumes an already-submitted service job.

    If the job's result bundle was already downloaded and validated, the
    handle loads it straight from the local cache, no service round-trip, so a finished run stays loadable offline and after the service ages the
    job out. Otherwise it resumes polling exactly as before. ``job.cancel()``
    asks the service to cancel the job; ``wait_timeout_s`` bounds only the wait
    (``timeout`` is its old name and warns).

    The result reads in the same absolute units and the same frame as the
    call that submitted the job: this client stores the wire spec of every job
    it submits, so the reader knows the simulation and undoes the engine's
    per-unit-amplitude normalization (NUMERICS.md section 12) here too, and it
    stores beside it the client state the spec does not carry (the user-frame
    origin of a fitted domain, the half a symmetry plane dropped, declared
    ports), so coordinates, whole planes and ``transmission()`` agree. Both are stored
    once the service returns the job id, so a job submitted from another
    machine, through a cache that has since been cleared, or by a submission
    whose response was lost has neither: pass ``simulation=`` then, and
    whenever you still hold the simulation. It takes precedence over the
    stored spec. With neither, the frequency-domain arrays carry the engine's
    unit-amplitude values and say so in their ``normalization`` attr, and the
    coordinates are the wire's corner frame."""
    if simulation is not None:
        from ..components import Simulation

        if not isinstance(simulation, Simulation):
            raise TypeError(
                "simulation must be the photonhub.Simulation the job was "
                f"submitted from, got {type(simulation).__name__}")
    job_id = validate_job_id(job_id)
    timeout = _poll_timeout(wait_timeout_s)
    cfg = get_config()
    http = HttpClient(cfg)
    stop = threading.Event()
    return Job(
        lambda: _finish_cloud_job(
            http, cfg, job_id, progress=progress, timeout=timeout,
            simulation=simulation, stop=stop),
        name=job_id, job_id=job_id,
        on_cancel=lambda: _cancel_service_job(cfg, job_id, stop))
