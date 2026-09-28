"""One-shot cloud actions that don't need a Job handle.

The paid path deliberately has a first-class preflight instead of making every
caller re-implement money parsing.  A :class:`CloudPreflight` is bound to one
simulation + device quote and checks both a caller ceiling and the account's
*available* balance (never the larger balance that may include reserved funds).
It also compares the grid the quote reports with the one this client realizes
and refuses a quote for a different grid; :func:`estimate` applies the same
check. That covers the quoted grid only: the solver image that runs the job is
versioned separately from the service that quotes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any
import warnings

from .._compat import caller_stacklevel
from ._ids import validate_job_id
from .client import HttpClient
from .config import CloudError, get_config
from .run import DEFAULT_MAX_USD

#: Relative tolerance on the quoted time step. A service that realizes the same
#: grid computes dt from the same Courant formula, so it agrees to rounding;
#: a different active-axis count moves dt by at least sqrt(3/2).
_QUOTE_DT_REL_TOL = 1e-6


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
        """Format the accepted quote, spend limit, and available-balance checks.

        Include expiry when the service supplied it. The quote identifier is
        omitted. This method makes no request and reserves no funds."""
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
            raise CloudError(
                f"service {context} {key!r} must be a finite non-negative number")
        return float(value)

    micros_key = key.removesuffix("_usd") + "_micros"
    value = payload.get(micros_key)
    if (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise CloudError(
            f"service {context} has no usable {key!r} or {micros_key!r}")
    return value / 1_000_000


def _dt_with_axes_floored(sim, axes) -> float:
    """The dt ``sim`` would get with ``axes`` floored at four cells, as a
    resolver without quasi-2-D support does: the same Courant formula as
    :meth:`Simulation.cost_estimate`, with those axes counted."""
    from ..cost import _cells_and_min_spacing_um, _dt_seconds

    counts, min_spacing = _cells_and_min_spacing_um(sim)
    floored = [max(4, n) if axis in axes else n
               for axis, n in zip("xyz", counts)]
    return _dt_seconds(sim, floored, min_spacing)


def _one_cell_hint(axes, one_cell_count: int) -> str:
    """Name the one-cell periodic ``axes`` a quote disagrees on.

    ``one_cell_count`` is how many axes are one periodic cell locally: one
    leaves a quasi-2-D scene, two a quasi-1-D one, three no named reduction.
    """
    if len(axes) == 1:
        named = f"{axes[0]} is a single periodic cell"
    else:
        named = (f"{', '.join(axes[:-1])} and {axes[-1]} are single "
                 "periodic cells")
    scene = {1: " (a quasi-2-D scene)", 2: " (a quasi-1-D scene)"}.get(
        one_cell_count, "")
    return (f"{named} here{scene}, which a version without one-cell periodic "
            "axes floors at four cells and counts in the Courant limit")


def _check_quote_grid(sim, quote: dict, *, context: str = "estimate",
                      unchecked: str = "submitting unchecked") -> None:
    """Refuse a quote whose grid differs from the one ``sim`` realizes here.

    The service quotes the wire document with its own bundled SDK, which can
    differ in version from this client in either direction and then resolve
    the same document to a different grid: one without quasi-2-D support
    floors a one-cell periodic axis (NUMERICS.md section 1) at four cells and
    counts it in the Courant limit. Compare the quote's ``cells_per_axis``
    and ``dt_s`` with :meth:`Simulation.cost_estimate`, which matches the
    engine's resolver. Any cell-count difference, or a dt outside
    ``_QUOTE_DT_REL_TOL``, raises :class:`CloudError` before submission.
    A quote without these fields warns and does not block; ``unchecked`` ends
    that warning with what happens next.

    This checks the QUOTED grid only. The solver image that runs the job is
    versioned separately from the service that quotes it, so the check cannot
    confirm the grid the worker's engine resolves.
    """
    local = sim.cost_estimate()
    local_cells = tuple(int(n) for n in local.cells_per_axis)
    # Axes that are one plain periodic cell here (a quasi-2-D scene): the
    # ones a service without quasi-2-D support floors at four cells and
    # counts in the Courant limit.
    boundaries = getattr(sim, "boundaries", None)
    one_cell_periodic = [
        axis for axis, n in zip("xyz", local_cells)
        if n == 1 and getattr(boundaries, axis, None) == "periodic"]
    problems = []
    missing = []
    hint_axes = []

    cells = quote.get("cells_per_axis")
    if cells is None:
        missing.append("cells_per_axis")
    else:
        if (not isinstance(cells, (list, tuple)) or len(cells) != 3
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1
                       for n in cells)):
            raise CloudError(
                f"service {context} 'cells_per_axis' must be three positive "
                f"integers (got {cells!r}); no job was submitted")
        for axis, quoted, realized in zip("xyz", cells, local_cells):
            if quoted != realized:
                problems.append(
                    f"{axis} axis has {quoted} cells in the quote and "
                    f"{realized} locally")
                # Four is the floor a resolver without one-cell support
                # applies; any other count has some other cause.
                if axis in one_cell_periodic and quoted == 4:
                    hint_axes.append(axis)

    dt = quote.get("dt_s")
    if dt is None:
        missing.append("dt_s")
    else:
        dt_value = None
        if not isinstance(dt, bool) and isinstance(dt, (int, float)):
            try:
                dt_value = float(dt)
            except OverflowError:  # a JSON integer beyond the float range
                dt_value = None
        if dt_value is None or not math.isfinite(dt_value) or dt_value <= 0:
            raise CloudError(
                f"service {context} 'dt_s' must be a finite positive number "
                f"(got {dt!r}); no job was submitted")
        if not math.isclose(dt_value, local.dt_s, rel_tol=_QUOTE_DT_REL_TOL,
                            abs_tol=0.0):
            # Nine digits: a mismatch just past the tolerance must not print
            # two equal values and a ratio of 1.
            problems.append(
                f"dt is {dt_value:.9g} s in the quote and {local.dt_s:.9g} s "
                f"locally (ratio {dt_value / local.dt_s:.9g}; the tolerance "
                f"is {_QUOTE_DT_REL_TOL:g} relative)")
            # The local dt leaves a one-cell periodic axis out of the Courant
            # limit. Name those axes only when the quoted dt is the one this
            # grid gets with them floored at four cells and counted.
            if one_cell_periodic and math.isclose(
                    dt_value, _dt_with_axes_floored(sim, one_cell_periodic),
                    rel_tol=_QUOTE_DT_REL_TOL, abs_tol=0.0):
                hint_axes.extend(a for a in one_cell_periodic
                                 if a not in hint_axes)

    local_grid = (f"{' x '.join(str(n) for n in local_cells)} cells, "
                  f"dt {local.dt_s:.6g} s")
    if problems:
        cause = ("the service and this client resolve the same document to "
                 "different grids, which points at a version difference "
                 "between the service's SDK and this one, in either "
                 "direction")
        if hint_axes:
            cause += "; " + _one_cell_hint(sorted(hint_axes),
                                           len(one_cell_periodic))
        raise CloudError(
            f"service {context} describes a different grid than this "
            f"simulation realizes ({local_grid}): {'; '.join(problems)}. "
            f"No job was submitted: {cause}.")
    if missing:
        warnings.warn(
            f"service {context} omits {', '.join(repr(k) for k in missing)}, "
            "so the client cannot confirm the service realizes this "
            f"simulation's grid ({local_grid}); {unchecked}",
            UserWarning, stacklevel=caller_stacklevel())


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
    """Return the service identity associated with the configured API key.

    This makes an authenticated request. Response-body or socket-read timeouts
    can propagate as ``TimeoutError``. Other transport and service errors raise
    ``CloudError``."""
    return HttpClient(get_config()).whoami()


def account() -> dict:
    """Balance/usage for the configured account (micro-USD + dollar fields)."""
    return HttpClient(get_config()).account()


def estimate(sim, *, device="gpu", solver=None) -> dict:
    """Server-side quote bound to ``sim``, device, and solver ref.

    As in :func:`preflight`, a quote whose ``cells_per_axis`` or ``dt_s``
    differ from the grid ``sim`` realizes locally raises :class:`CloudError`
    instead of returning its price, and a quote that omits them warns. The
    check belongs here because the ``quote_id`` can be passed to
    :func:`~photonhub.cloud.run` or :func:`~photonhub.cloud.submit`, which
    receive only the id and so cannot check the grid themselves. It covers
    the quoted grid only: the solver image that runs the job is versioned
    separately from the service that quotes it.
    """
    # Imported here, as everywhere in this module, to keep estimate and run on
    # exactly the same public device grammar and the same capability guard.
    from .run import _check_cloud_device_support, _validate_web_device

    device = _validate_web_device(device)
    wire = _check_cloud_device_support(sim, device)
    sim.check_runnable()
    quote = HttpClient(get_config()).estimate(
        wire, device=device, solver=solver)
    if not isinstance(quote, dict):
        raise CloudError("service estimate response was not an object")
    _check_quote_grid(sim, quote,
                      unchecked="a job bound to this quote runs unchecked")
    return quote


def preflight(
    sim, *, device: str = "gpu", solver=None, max_usd: float = DEFAULT_MAX_USD,
) -> CloudPreflight:
    """Get a device/solver-bound quote and enforce the spend/balance limit.

    This call does not submit a job.  ``max_usd`` defaults to the beta's $5
    credit ceiling.  The account must report ``available_usd`` (or exact
    ``available_micros``); falling back to total balance would be unsafe because
    funds reserved by active jobs are not spendable.  When the quote reports
    ``cells_per_axis`` and ``dt_s``, they must match the grid ``sim`` realizes
    locally, before any price check; a quote that omits them warns.  This
    checks the quoted grid only: the solver image that runs the job is
    versioned separately from the service that quotes it.
    """
    return _preflight(lambda: HttpClient(get_config()), sim, device=device,
                      solver=solver, max_usd=max_usd)


def _preflight(http_factory, sim, *, device, solver, max_usd,
               preflight_wire=None) -> CloudPreflight:
    """:func:`preflight` on the client ``http_factory()`` returns, made only
    once the local checks pass: the default quote step of the run and submit
    entry points uses the same client they submit with."""
    from .run import (
        _check_cloud_device_support, _validate_quote_id, _validate_web_device,
    )

    device = _validate_web_device(device)
    if device is None:
        raise ValueError("cloud preflight requires an explicit device")
    # Before the account and estimate round-trips: a spec the GPU cannot run
    # must not reach a quote at all, and neither may one the solver refuses
    # (no source, overlapping boundary slabs, a flux plane in the layers, a
    # run shorter than a cw ramp), the rules run_local checks before it runs.
    wire = (preflight_wire if preflight_wire is not None else
            _check_cloud_device_support(sim, device))
    sim.check_runnable()
    limit = _finite_non_negative(max_usd, "max_usd")
    http = http_factory()
    account_payload = http.account()
    if not isinstance(account_payload, dict):
        raise CloudError("service account response was not an object")
    available = _service_amount(
        account_payload, "available_usd", context="account response")

    quote = http.estimate(wire, device=device, solver=solver)
    if not isinstance(quote, dict):
        raise CloudError("service estimate response was not an object")
    quote_usd = _service_amount(quote, "usd", context="estimate")
    try:
        quote_id = _validate_quote_id(quote.get("quote_id"))
    except ValueError as exc:
        raise CloudError("service estimate has no usable 'quote_id'") from exc
    if quote_id is None:
        raise CloudError("service estimate has no usable 'quote_id'")
    # The grid comes before the price: a quote for a different grid prices
    # the wrong run, so its dollar checks would be misleading.
    _check_quote_grid(sim, quote)
    if quote_usd > limit:
        raise CloudError(
            f"server quote ${quote_usd:.6f} exceeds max_usd ${limit:.6f}; "
            "no job was submitted")
    if quote_usd > available:
        raise CloudError(
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
    """Request cancellation of a cloud job and return the service response.

    Validate ``job_id`` before the request. This does not wait for a terminal
    state or determine the final charge or refund. Use ``job_status`` to inspect
    the recorded state and costs."""
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
        raise CloudError("service job status response was not an object", job_id=job_id)
    return _normalise_job_costs(record)


def gpus() -> list:
    """The curated menu of GPUs you can run on, each a dict like
    ``{"id": ..., "vendor": ..., "arch": ..., "gpu_mem_gb": ...}``. Pass an id to
    ``ph.cloud.run(sim, device="gpu:<id>")``;
    bare ``device="gpu"`` lets the platform pick a default. The platform manages
    which providers back each entry, that stays an internal detail."""
    return HttpClient(get_config()).list_gpus()
