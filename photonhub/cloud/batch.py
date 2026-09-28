"""Cloud batch, submit N simulations and assemble the SAME
:class:`~photonhub.runners.batch.BatchResults` the local path returns, so per-name
partial failures (``batch_data.errors[name]``) work identically. This realizes
the local-backend docstring's promise that on the cloud a Batch "becomes a
fan-out across GPUs": each name is an independent job the coordinator spreads
across capacity.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Optional

from ..components import Simulation
from ..runners.batch import BatchResults, _check_batch_name
from ..runners.local import SolverRunError
from .actions import _check_quote_grid
from .client import HttpClient
from .config import CloudError, get_config
from .run import (
    CloudJobTimeout,
    _check_cloud_device_support,
    _cloud_run,
    _poll_timeout,
    _renamed_timeout,
    _validate_quote_id,
    _validate_web_device,
)


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


def _accepted_quote(http: HttpClient, name: str, sim: Simulation,
                    limit: float, device: Optional[str], wire: dict) -> str:
    """Get and validate one server quote without submitting the job."""
    estimate = http.estimate(wire, device=device)
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
    return quote_id


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
    def run(self, *, device="gpu", max_workers: int = 4,
            wait_timeout_s: Optional[float] = None, max_usd=None) -> BatchResults:
        """Quote every entry, then submit within its accepted spend limit.

        ``max_usd`` is required and is either one per-job limit applied to all
        entries or a mapping whose keys exactly match the batch. All estimates
        are validated before any job is submitted, including, when the service
        reports a quote's ``cells_per_axis`` and ``dt_s``, that they match the
        grid its simulation realizes locally (a quote that omits them warns
        and is still submitted), and each accepted quote id is bound to its
        corresponding submission. ``device`` defaults to ``"gpu"`` as in every
        cloud call. ``wait_timeout_s`` bounds only how long each entry is
        waited for: an entry past it is reported as a
        :class:`~photonhub.cloud.CloudJobTimeout` and its service job keeps
        running, and billing (``timeout`` is its old name and warns).
        """
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
        quotes = {
            name: _accepted_quote(
                quote_http, name, sim, limits[name], device, wires[name])
            for name, sim in self.simulations.items()
        }
        results = {}
        errors = {}

        def _one(item):
            name, sim = item
            try:
                return name, _cloud_run(sim, name=name, device=device,
                                        timeout=timeout, quote_id=quotes[name],
                                        cfg=cfg, preflight_wire=wires[name]), None
            except (SolverRunError, CloudJobTimeout, CloudError) as e:
                return name, None, e

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            for name, data, err in ex.map(_one, self.simulations.items()):
                if err is None:
                    results[name] = data
                else:
                    errors[name] = err

        return BatchResults(results, errors, cfg.cache_dir,
                         list(self.simulations))
