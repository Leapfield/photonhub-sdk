"""Cloud batch, submit N simulations and assemble the SAME
:class:`~photonhub.runners.batch.BatchResults` the local path returns, so per-name
partial failures (``batch_data.errors[name]``) work identically. This realizes
the local-backend docstring's promise that on the cloud a Batch "becomes a
fan-out across GPUs": each name is an independent job the coordinator spreads
across capacity.
"""

from __future__ import annotations

import math
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional

from .._compat import caller_stacklevel
from ..components import Simulation
from ..data import SimulationMismatchError
from ..runners.batch import BatchResults, _check_batch_name
from ..runners.local import SolverRunError
from .actions import _check_quote_grid
from .client import HttpClient
from .config import CloudError, get_config
from ._defaults import resolve as resolve_defaults
from .run import (
    CloudJobTimeout,
    _check_cloud_device_support,
    _finish_cloud_job,
    _poll_timeout,
    _renamed_timeout,
    _store_submission,
    _submitted_job_id,
    _validate_quote_id,
    _validate_web_device,
)


class _TotalCeilingRefused(CloudError):
    """A whole-batch quote refused by the total spend ceiling."""

    def __init__(self, total_quote_usd: float, limit_usd: float):
        self.total_quote_usd = total_quote_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"summed quote (${total_quote_usd:.4f}) exceeds max_usd_total "
            f"ceiling (${limit_usd:.4f}); nothing was submitted")


def _spend_limit(value, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(limit) or limit < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return limit


def _spend_limits(max_usd, names) -> dict[str, float]:
    """Resolve a scalar or per-name mapping into one limit for every job."""
    if max_usd is None:
        raise ValueError(
            "cloud Batch.run requires max_usd: pass one per-job limit or a "
            "mapping with one limit for every batch entry")
    if isinstance(max_usd, Mapping):
        expected = set(names)
        actual = set(max_usd)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected, key=repr)
            raise ValueError(
                "max_usd mapping keys must exactly match batch entries; "
                f"missing={missing}, extra={extra}")
        return {
            name: _spend_limit(max_usd[name], label=f"max_usd[{name!r}]")
            for name in names
        }
    limit = _spend_limit(max_usd, label="max_usd")
    return {name: limit for name in names}


@dataclass(frozen=True)
class _AcceptedQuote:
    quote_id: str
    usd: float
    expires_at: Optional[datetime]
    refresh_margin_s: float

    def needs_refresh(self) -> bool:
        return (self.expires_at is not None and
                self.expires_at <= datetime.now(timezone.utc)
                + timedelta(seconds=self.refresh_margin_s))


def _accepted_quote(http: HttpClient, name: str, sim: Simulation,
                    limit: float, device: Optional[str], wire: dict,
                    solver=None) -> _AcceptedQuote:
    """Get and validate one server quote without submitting the job."""
    estimate = http.estimate(wire, device=device, solver=solver)
    if not isinstance(estimate, dict):
        raise CloudError(
            f"server estimate for batch entry {name!r} was not an object")
    usd = estimate.get("usd")
    if (isinstance(usd, bool) or not isinstance(usd, (int, float))
            or not math.isfinite(float(usd)) or float(usd) < 0):
        raise CloudError(
            f"server estimate for batch entry {name!r} has no finite "
            f"non-negative 'usd' value (got {usd!r})")
    try:
        quote_id = _validate_quote_id(estimate.get("quote_id"))
    except ValueError as exc:
        raise CloudError(
            f"server estimate for batch entry {name!r} has no usable "
            "'quote_id'") from exc
    if quote_id is None:
        raise CloudError(
            f"server estimate for batch entry {name!r} has no usable "
            "'quote_id'")
    _check_quote_grid(sim, estimate,
                      context=f"estimate for batch entry {name!r}")
    if float(usd) > limit:
        raise CloudError(
            f"server estimate for batch entry {name!r} (${float(usd):.4f}) "
            f"exceeds its max_usd limit (${limit:.4f})")
    expires_at = estimate.get("expires_at")
    if expires_at is not None:
        try:
            expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                raise ValueError("expiry must include a timezone")
        except (AttributeError, TypeError, ValueError) as exc:
            raise CloudError(
                f"server estimate for batch entry {name!r} has an invalid 'expires_at'") from exc
    # Fix the margin at receipt, rather than shrinking it on every check (which
    # would never refresh before expiry). A short-lived quote must remain usable.
    margin_s = (0.0 if expires_at is None else
                min(60.0, max(0.0, (expires_at - datetime.now(timezone.utc)).total_seconds()) / 4))
    return _AcceptedQuote(quote_id, float(usd), expires_at, margin_s)


class Batch:
    """Named simulations submitted as independent cloud jobs.

    Construction validates names and simulation types and makes no requests.
    ``quote`` returns local cost estimates. ``run`` validates all service quotes
    and spend limits before submitting jobs in parallel. Successful results and
    per-name failures share the local ``BatchResults`` interface."""
    def __init__(self, simulations: Mapping[str, Simulation]):
        if not simulations:
            raise ValueError("Batch needs at least one simulation")
        validated = {}
        for name, sim in simulations.items():
            _check_batch_name(name)
            if not isinstance(sim, Simulation):
                raise TypeError(
                    f"batch entry {name!r} is {type(sim).__name__}, "
                    "expected a Simulation")
            validated[name] = sim
        self.simulations = validated

    def quote(self, **kwargs):
        """Per-name CostEstimate + the batch total (local, deterministic)."""
        per = {name: sim.cost_estimate(**kwargs)
               for name, sim in self.simulations.items()}
        total = sum(e.usd for e in per.values())
        return per, total

    @_renamed_timeout
    def run(self, *, device="gpu", solver=None, max_workers: int = 4,
            wait_timeout_s: Optional[float] = None, max_usd=None,
            max_usd_total=None, on_submit=None, on_complete=None) -> BatchResults:
        """Quote every entry, then submit within its accepted spend limit.

        ``max_usd`` is required and is either one per-job limit applied to all
        entries or a mapping whose keys exactly match the batch. All estimates
        are validated before any job is submitted, including, when the service
        reports a quote's ``cells_per_axis`` and ``dt_s``, that they match the
        grid its simulation realizes locally (a quote that omits them warns
        and is still submitted), and each accepted quote id is bound to its
        corresponding submission. ``device`` and ``solver`` default as in
        :func:`~photonhub.cloud.run` (the service's only GPU, and on a GPU the
        commit of the locally installed solver), resolved once for the whole
        batch and sent unchanged on every estimate, refresh and submission.
        ``wait_timeout_s`` bounds only how long each entry is
        waited for: an entry past it is reported as a
        :class:`~photonhub.cloud.CloudJobTimeout` and its service job keeps
        running, and billing (``timeout`` is its old name and warns).
        ``max_usd_total`` optionally caps the sum of accepted quotes. A quote
        above either ceiling in the initial pass refuses the whole batch.
        Batches larger than service concurrency run in waves: set
        ``max_workers`` to the service's active-job limit (1 for the beta).
        Immediately before submission, quotes near expiry are refreshed and
        both ceilings checked again against accepted spend. The refresh margin
        is one quarter of the remaining lifetime when the quote is received,
        capped at 60 seconds and fixed for that quote.
        A missing ``expires_at`` means no expiry. A refresh refusal fails only
        that entry; earlier jobs may already have been charged.
        ``on_submit(name, job_id, quote_usd)`` runs on a worker thread after
        submission. ``on_complete(name, result, error)`` runs on the calling
        thread as each entry finishes. Submit hooks have their own lock.
        ``on_submit`` exceptions are reported as warnings, never raised,
        because the job is already paid. ``on_complete`` exceptions propagate.
        If collection is interrupted, queued entries are cancelled before the
        exception is raised; entries already running are waited for first.
        A submit-time CloudError carries its accepted ``quote_usd`` because
        the service may have accepted a job before the response was lost.
        """
        total_limit = (None if max_usd_total is None else
                       _spend_limit(max_usd_total, label="max_usd_total"))
        timeout = _poll_timeout(wait_timeout_s)
        device = _validate_web_device(device)
        # Every entry is checked against the device before ANY of them is
        # quoted: one entry the GPU cannot run must not cost the others a
        # round-trip, let alone a bound quote.
        wires = {}
        for name, sim in self.simulations.items():
            wires[name] = _check_cloud_device_support(sim, device, entry=name)
            sim.check_runnable()
        limits = _spend_limits(max_usd, self.simulations)
        cfg = get_config()
        quote_http = HttpClient(cfg)
        # Once per batch: every estimate, refresh and submission below names
        # this one device and solver, so each bound quote stays valid.
        device, solver = resolve_defaults(quote_http, device, solver)
        quotes = {
            name: _accepted_quote(
                quote_http, name, sim, limits[name], device, wires[name], solver)
            for name, sim in self.simulations.items()
        }
        total_quote = sum(quote.usd for quote in quotes.values())
        if total_limit is not None and total_quote > total_limit + 1e-9:
            raise _TotalCeilingRefused(total_quote, total_limit)
        completed = {}
        submit_lock = threading.Lock()
        spend_lock = threading.Lock()
        accepted_usd = 0.0

        def _one(item):
            nonlocal accepted_usd
            name, sim = item
            quote = quotes[name]
            submission_attempted = False
            try:
                http = HttpClient(cfg)
                # Refresh, reserve and submit under one lock so concurrent
                # workers cannot spend the same remaining budget. Polling and
                # downloads run outside it. Ambiguous submissions stay reserved.
                with spend_lock:
                    if quote.needs_refresh():
                        quote = _accepted_quote(
                            http, name, sim, limits[name], device, wires[name], solver)
                    if quote.needs_refresh():
                        raise CloudError(f"fresh quote for batch entry {name!r} expires too soon")
                    if total_limit is not None and accepted_usd + quote.usd > total_limit + 1e-9:
                        raise CloudError(
                            f"quote for batch entry {name!r} (${quote.usd:.4f}) "
                            f"exceeds remaining max_usd_total "
                            f"(${max(0.0, total_limit - accepted_usd):.4f}); entry was not submitted")
                    accepted_usd += quote.usd
                    submission_attempted = True
                    try:
                        response = http.submit_job(wires[name], name=name, device=device,
                                                   solver=solver, quote_id=quote.quote_id)
                    except CloudError as exc:
                        if exc.status_code is not None and 400 <= exc.status_code < 500:
                            accepted_usd -= quote.usd
                        raise
                job_id = _submitted_job_id(response)
                _store_submission(cfg, job_id, sim, wires[name])
            except CloudError as e:
                e.quote_usd = quote.usd if submission_attempted else 0.0
                e.submission_attempted = submission_attempted
                return name, None, e
            except (SolverRunError, CloudJobTimeout) as e:
                return name, None, e
            if on_submit is not None:
                with submit_lock:
                    try:
                        on_submit(name, job_id, quote.usd)
                    except Exception as exc:
                        try:
                            warnings.warn(
                                f"cloud job {job_id} was submitted, but on_submit failed "
                                f"({type(exc).__name__}: {exc})",
                                UserWarning, stacklevel=caller_stacklevel())
                        except Exception:
                            pass
            try:
                return name, _finish_cloud_job(http, cfg, job_id,
                                               timeout=timeout, simulation=sim), None
            except (SolverRunError, CloudJobTimeout, CloudError, SimulationMismatchError) as e:
                return name, None, e

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futures = [ex.submit(_one, item) for item in self.simulations.items()]
            try:
                for future in as_completed(futures):
                    name, data, err = future.result()
                    completed[name] = (data, err)
                    if on_complete is not None:
                        on_complete(name, data, err)
            except BaseException:
                for future in futures:
                    future.cancel()
                ex.shutdown(cancel_futures=True)
                raise

        results = {name: completed[name][0] for name in self.simulations
                   if completed[name][1] is None}
        errors = {name: completed[name][1] for name in self.simulations
                  if completed[name][1] is not None}

        return BatchResults(results, errors, cfg.cache_dir,
                         list(self.simulations))
