"""One-shot cloud actions that don't need a Job handle.

The paid path deliberately has a first-class preflight instead of making every
caller re-implement money parsing.  A :class:`CloudPreflight` is bound to one
simulation + device quote and checks both a caller ceiling and the account's
*available* balance (never the larger balance that may include reserved funds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from ._ids import validate_job_id
from .client import HttpClient
from .config import WebError, get_config


@dataclass(frozen=True)
class CloudPreflight:
    """A quote that passed the caller's ceiling and available-balance checks.

    ``quote_id`` is intentionally omitted from ``repr`` so displaying this
    object in a notebook does not publish the opaque accepted-quote token.
    ``quote`` retains the service response for audit fields such as cell count,
    step count, rate, and expiry.
    """

    device: str
    solver: str | None
    max_usd: float
    quote_usd: float
    available_usd: float
    remaining_usd: float
    quote_id: str = field(repr=False)
    quote: dict[str, Any] = field(repr=False)

    def summary(self) -> str:
        expiry = self.quote.get("expires_at")
        suffix = f"; expires {expiry}" if expiry else ""
        return (
            f"cloud preflight: quote ${self.quote_usd:.12f} <= "
            f"limit ${self.max_usd:.2f}; available ${self.available_usd:.6f}; "
            f"remaining after quote ${self.remaining_usd:.6f}{suffix}"
        )


def _finite_non_negative(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return amount


def _service_amount(payload: dict, key: str, *, context: str) -> float:
    """Read a service dollar field, accepting its integer micro-dollar twin."""
    value = payload.get(key)
    if value is not None:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0):
            raise WebError(
                f"service {context} {key!r} must be a finite non-negative number")
        return float(value)

    micros_key = key.removesuffix("_usd") + "_micros"
    value = payload.get(micros_key)
    if (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise WebError(
            f"service {context} has no usable {key!r} or {micros_key!r}")
    return value / 1_000_000


def _normalise_job_costs(record: dict) -> dict:
    """Add dollar twins for stable integer micro-dollar history fields."""
    out = dict(record)
    for stem in ("quote", "actual", "refunded"):
        dollars = f"{stem}_usd"
        micros = f"{stem}_micros"
        if dollars not in out:
            value = out.get(micros)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                out[dollars] = value / 1_000_000
    return out


def whoami() -> dict:
    return HttpClient(get_config()).whoami()


def account() -> dict:
    """Balance/usage for the configured account (micro-USD + dollar fields)."""
    return HttpClient(get_config()).account()


def estimate(sim, *, device=None, solver=None) -> dict:
    """Server-side quote bound to ``sim``, device, and solver ref."""
    # Lazy import avoids an import cycle while keeping estimate/run on exactly
    # the same public device grammar.
    from .run import _validate_web_device

    device = _validate_web_device(device)
    return HttpClient(get_config()).estimate(
        sim.to_wire_dict(), device=device, solver=solver)


def preflight(
    sim, *, device: str = "gpu", solver=None, max_usd: float = 5.0,
) -> CloudPreflight:
    """Get a device/solver-bound quote and enforce the spend/balance limit.

    This call does not submit a job.  ``max_usd`` defaults to the beta's $5
    credit ceiling.  The account must report ``available_usd`` (or exact
    ``available_micros``); falling back to total balance would be unsafe because
    funds reserved by active jobs are not spendable.
    """
    from .run import _validate_quote_id, _validate_web_device

    device = _validate_web_device(device)
    if device is None:
        raise ValueError("cloud preflight requires an explicit device")
    limit = _finite_non_negative(max_usd, "max_usd")
    http = HttpClient(get_config())
    account_payload = http.account()
    if not isinstance(account_payload, dict):
        raise WebError("service account response was not an object")
    available = _service_amount(
        account_payload, "available_usd", context="account response")

    quote = http.estimate(sim.to_wire_dict(), device=device, solver=solver)
    if not isinstance(quote, dict):
        raise WebError("service estimate response was not an object")
    quote_usd = _service_amount(quote, "usd", context="estimate")
    try:
        quote_id = _validate_quote_id(quote.get("quote_id"))
    except ValueError as exc:
        raise WebError("service estimate has no usable 'quote_id'") from exc
    if quote_id is None:
        raise WebError("service estimate has no usable 'quote_id'")
    if quote_usd > limit:
        raise WebError(
            f"server quote ${quote_usd:.6f} exceeds max_usd ${limit:.6f}; "
            "no job was submitted")
    if quote_usd > available:
        raise WebError(
            f"server quote ${quote_usd:.6f} exceeds available balance "
            f"${available:.6f}; no job was submitted")
    return CloudPreflight(
        device=device,
        solver=solver,
        max_usd=limit,
        quote_usd=quote_usd,
        available_usd=available,
        remaining_usd=available - quote_usd,
        quote_id=quote_id,
        quote=dict(quote),
    )


def create_api_key(name: str = "default") -> dict:
    """Mint a new API key (the plaintext ``token`` is returned exactly once)."""
    return HttpClient(get_config()).create_api_key(name)


def cancel(job_id: str) -> dict:
    job_id = validate_job_id(job_id)
    return HttpClient(get_config()).cancel_job(job_id)


def list_jobs() -> list[dict]:
    """Recent service jobs, including normalized quote/actual/refund dollars.

    The retention window and ordering are service policy.  This is the recovery
    surface for finding a paid job id after a notebook or process exits.
    """
    jobs = HttpClient(get_config()).list_jobs()
    return [_normalise_job_costs(record) for record in jobs]


def job_status(job_id: str) -> dict:
    """One service job's current state, progress, and cost metadata."""
    job_id = validate_job_id(job_id)
    record = HttpClient(get_config()).get_job(job_id)
    if not isinstance(record, dict):
        raise WebError("service job status response was not an object", job_id=job_id)
    return _normalise_job_costs(record)


def gpus() -> list:
    """The curated menu of GPUs you can run on, each a dict like
    ``{"id": "mi300x", "vendor": "AMD", "arch": "gfx942",
    "gpu_mem_gb": 192}``. Pass an id to
    ``ph.web.run(sim, device="gpu:<id>")``;
    bare ``device="gpu"`` lets the platform pick a default. The platform manages
    which providers back each entry — that stays an internal detail."""
    return HttpClient(get_config()).list_gpus()
