"""Cloud run entry points — the prime directive: ``ph.web.run_async`` returns the
**same** :class:`~photonhub.runners.batch.Job` as the local path, so
``job = ph.web.run_async(sim); data = job.result()`` reads identically whether
local or cloud, and a server-side failure surfaces as the same
:class:`SolverRunError`.
"""

from __future__ import annotations

import math
import re
import time
from typing import Callable, Optional

from ..bundle import BundleError
from ..data import SimulationData
from ..runners.batch import Job
from ..runners.local import SolverRunError
from . import cache
from ._ids import validate_job_id
from .client import HttpClient
from .config import WebConfig, WebError, get_config

ProgressCb = Optional[Callable[[dict], None]]

# A curated-GPU id in `device="gpu:<id>"` (the beta uses "mi300x"). Lowercase
# slug, distinct from the local path's numeric `gpu:N` device index; legacy or
# development catalogs may expose other ids.
_GPU_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")


class WebJobTimeout(TimeoutError):
    """Polling timed out while the service job remains active."""

    def __init__(self, job_id: str, timeout: float):
        self.job_id = job_id
        self.timeout = timeout
        super().__init__(
            f"cloud job {job_id!r} not finished after {timeout} s; "
            "resume it with ph.web.resume(job_id) or cancel it with "
            "ph.web.cancel(job_id)")


def _poll_timeout(timeout: Optional[float]) -> Optional[float]:
    if timeout is None:
        return None
    if isinstance(timeout, bool):
        raise ValueError("timeout must be a finite non-negative number or None")
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "timeout must be a finite non-negative number or None") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("timeout must be a finite non-negative number or None")
    return value


def _submitted_job_id(response: object) -> str:
    try:
        candidate = response["job_id"]  # type: ignore[index]
        return validate_job_id(candidate)
    except (KeyError, TypeError, ValueError) as exc:
        raise WebError("service returned an invalid job_id") from exc


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
    ``ph.gpus()`` — the platform resolves it to a provider + a plain ``gpu`` on
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


def _poll_and_download(http: HttpClient, cfg: WebConfig, job_id: str, *,
                       progress: ProgressCb, timeout: Optional[float]) -> object:
    timeout = _poll_timeout(timeout)
    deadline = (time.monotonic() + timeout) if timeout is not None else None
    interval = cfg.poll_interval_s
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise WebJobTimeout(job_id, timeout)
        try:
            st = http.get_job(job_id, deadline=deadline)
        except TimeoutError as exc:
            raise WebJobTimeout(job_id, timeout) from exc
        if deadline is not None and time.monotonic() >= deadline:
            raise WebJobTimeout(job_id, timeout)
        if not isinstance(st, dict) or not isinstance(st.get("state"), str):
            raise WebError(
                f"service returned an invalid status for cloud job {job_id}")
        state = st["state"]
        if state not in (
                "queued", "provisioning", "running", "succeeded", "failed",
                "cancelled"):
            raise WebError(
                f"service returned unknown state {state!r} for cloud job "
                f"{job_id}", job_id=job_id)
        if progress and st.get("progress"):
            progress(st["progress"])
            if deadline is not None and time.monotonic() >= deadline:
                raise WebJobTimeout(job_id, timeout)
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
            time.sleep(sleep_for)
        interval = min(interval * 1.5, cfg.poll_backoff_max_s)
    try:
        return cache.download_bundle(http, cfg, job_id)
    except (BundleError, OSError) as exc:
        raise WebError(
            f"cloud job {job_id} returned an invalid result bundle: {exc}",
            job_id=job_id) from exc


def _finish_cloud_job(http: HttpClient, cfg: WebConfig, job_id: str, *,
                      progress: ProgressCb = None,
                      timeout: Optional[float] = None) -> SimulationData:
    try:
        bundle_dir = _poll_and_download(
            http, cfg, job_id, progress=progress, timeout=timeout)
    except (SolverRunError, WebJobTimeout):
        raise
    except WebError as exc:
        if exc.job_id is None:
            exc.job_id = job_id
        raise
    try:
        return SimulationData(bundle_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Do not preserve a completion marker for outputs the public reader
        # rejects; resume can safely re-fetch this already-paid job.
        cache.invalidate(cfg, job_id)
        raise WebError(
            f"cloud job {job_id} returned unreadable outputs: {exc}",
            job_id=job_id) from exc


def _cloud_run(sim, *, name=None, device=None, solver=None,
               progress: ProgressCb = None,
               timeout: Optional[float] = None,
               quote_id: Optional[str] = None,
               cfg: Optional[WebConfig] = None) -> SimulationData:
    device = _validate_web_device(device)
    timeout = _poll_timeout(timeout)
    quote_id = _validate_quote_id(quote_id)
    cfg = cfg or get_config()
    http = HttpClient(cfg)
    resp = http.submit_job(
        sim.to_wire_dict(), name=name, device=device, solver=solver,
        quote_id=quote_id)
    job_id = _submitted_job_id(resp)
    return _finish_cloud_job(
        http, cfg, job_id, progress=progress, timeout=timeout)


def run(sim, *, name=None, device=None, solver=None, progress: ProgressCb = None,
        timeout: Optional[float] = None,
        quote_id: Optional[str] = None) -> SimulationData:
    """Submit ``sim`` to the cloud and block until its result is ready. Returns a
    :class:`SimulationData`; raises :class:`SolverRunError` if the run fails,
    :class:`WebError` for transport/auth/result-transfer problems. ``solver`` pins
    a specific solver version/commit (default: latest). Pass the ``quote_id`` from
    a device-matched server estimate to bind the submission to that accepted quote."""
    return _cloud_run(sim, name=name, device=device, solver=solver,
                      progress=progress, timeout=timeout, quote_id=quote_id)


def run_quoted(sim, *, max_usd: float = 5.0, name=None, device: str = "gpu",
               solver=None, progress: ProgressCb = None,
               timeout: Optional[float] = None) -> SimulationData:
    """Preflight and submit one quote-bound job under a hard dollar ceiling.

    Unlike the compatibility-level :func:`run`, this helper cannot submit an
    unquoted job: it verifies a finite server quote, ``max_usd`` (default $5),
    and the account's *available* balance before binding that quote id to the
    submission.  The service remains the final authority for quote expiry and
    concurrent balance changes.
    """
    from .actions import preflight

    accepted = preflight(
        sim, device=device, solver=solver, max_usd=max_usd,
    )
    return _cloud_run(
        sim, name=name, device=accepted.device, solver=accepted.solver,
        progress=progress, timeout=timeout, quote_id=accepted.quote_id)


def run_async(sim, *, name=None, device=None, solver=None,
              progress: ProgressCb = None,
              timeout: Optional[float] = None,
              quote_id: Optional[str] = None) -> Job:
    """Submit ``sim`` and return the same :class:`Job` handle type as local
    ``ph.run_async`` once the service accepts the job. Polling and download
    continue in the background; collect with ``job.result()``. ``solver`` pins a
    specific solver version/commit (default: latest). Pass the ``quote_id`` from a
    server estimate to bind the submission to that quote."""
    device = _validate_web_device(device)
    timeout = _poll_timeout(timeout)
    quote_id = _validate_quote_id(quote_id)
    cfg = get_config()
    http = HttpClient(cfg)
    # Submit before returning so the handle owns the real service id. Transport
    # and authentication failures therefore surface synchronously, before a
    # background thread can hide them.
    resp = http.submit_job(
        sim.to_wire_dict(), name=name, device=device, solver=solver,
        quote_id=quote_id)
    job_id = _submitted_job_id(resp)
    return Job(
        lambda: _finish_cloud_job(
            http, cfg, job_id, progress=progress, timeout=timeout),
        name=name, job_id=job_id)


def run_quoted_async(sim, *, max_usd: float = 5.0, name=None,
                     device: str = "gpu", solver=None,
                     progress: ProgressCb = None,
                     timeout: Optional[float] = None) -> Job:
    """Async form of :func:`run_quoted`, returning the accepted service id."""
    from .actions import preflight

    accepted = preflight(
        sim, device=device, solver=solver, max_usd=max_usd,
    )
    return run_async(
        sim, name=name, device=accepted.device, solver=accepted.solver,
        progress=progress, timeout=timeout, quote_id=accepted.quote_id)


def resume(job_id: str, *, progress: ProgressCb = None,
           timeout: Optional[float] = None) -> Job:
    """Return a handle that resumes polling an already-submitted service job."""
    job_id = validate_job_id(job_id)
    timeout = _poll_timeout(timeout)
    cfg = get_config()
    http = HttpClient(cfg)
    return Job(
        lambda: _finish_cloud_job(
            http, cfg, job_id, progress=progress, timeout=timeout),
        name=job_id, job_id=job_id)
