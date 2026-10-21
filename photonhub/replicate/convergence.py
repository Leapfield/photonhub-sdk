"""Automated convergence gate — run a device at escalating resolution until the
figure-of-merit stops moving, and stamp the evidence into the result.

The user's standing bar is *convergence demonstrated, not assumed*. Nothing in
the stack looped build -> run -> extract -> drift -> stop; the pieces existed
(the benchmark framework's convergence metric had zero callers,
``spec.resolutions`` was parsed and ignored). This wires them together.

Two design points learned from this repo's own history:

* the stop rule is on the SUCCESSIVE-rung drift (finest vs previous), not the
  coarsest-to-finest total — appending a finer rung can only grow the total, so
  it cannot express "the answer stopped moving";
* one flat step is not proof (a metric can oscillate through a flat pair), so
  ``patience`` consecutive sub-tolerance steps are required (default 2). This is
  the automated form of the dl-refinement the benchmarks did by eye.

The core :func:`auto_converge` is solver-agnostic — you inject ``make_run`` (a
resolution -> ``(Simulation, extract)`` factory) and ``run`` — so it is unit
testable without an engine. :func:`converge_through_transmission` wires it to the
replication build + a local/GPU run for the through-port transmission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Context, Decimal, MAX_EMAX, MIN_EMIN, localcontext
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "Rung",
    "ConvergenceReport",
    "auto_converge",
    "converge_through_transmission",
]


@dataclass(frozen=True)
class Rung:
    """One rung of the resolution ladder."""

    cells_per_wavelength: float
    dl_um: float
    metric: float
    cost_usd: float
    run_dir: Optional[str] = None


@dataclass(frozen=True)
class ConvergenceReport:
    """The convergence evidence stamped alongside a result."""

    converged: bool
    metric_name: str
    ladder: Tuple[Rung, ...]
    tol: float
    patience: int
    stop_reason: str  # "tol" | "budget" | "estimate_unavailable" | "diverged" | "exhausted"
    total_cost_usd: float
    budget_detail: Optional[str] = None
    budget_cause: Optional[str] = None
    _budgeted: bool = field(default=False, repr=False, compare=False)
    _spent_usd: float = field(default=0.0, repr=False, compare=False)

    @property
    def finest(self) -> Optional[float]:
        return self.ladder[-1].metric if self.ladder else None

    @property
    def drift_successive(self) -> Optional[float]:
        """|finest - previous rung| in the metric's own units."""
        if len(self.ladder) < 2:
            return None
        return abs(self.ladder[-1].metric - self.ladder[-2].metric)

    @property
    def drift_total(self) -> Optional[float]:
        """|finest - coarsest| — a report number, NOT the stop rule."""
        if len(self.ladder) < 2:
            return None
        return abs(self.ladder[-1].metric - self.ladder[0].metric)

    def to_dict(self) -> dict:
        result = {
            "converged": self.converged,
            "metric_name": self.metric_name,
            "tol": self.tol,
            "patience": self.patience,
            "stop_reason": self.stop_reason,
            "finest": self.finest,
            "drift_successive": self.drift_successive,
            "drift_total": self.drift_total,
            "total_cost_usd": self.total_cost_usd,
            "ladder": [
                {
                    "cells_per_wavelength": r.cells_per_wavelength,
                    "dl_um": r.dl_um,
                    "metric": r.metric,
                    "cost_usd": r.cost_usd,
                }
                for r in self.ladder
            ],
        }
        if self.budget_detail is not None:
            result["budget_detail"] = self.budget_detail
        return result

    def summary(self) -> str:
        if self.finest is None and (self.budget_detail is not None or self._budgeted):
            finest = "none"
        else:
            finest = f"{self.finest:.5g}"
        head = (
            f"convergence[{self.metric_name}]: "
            f"{'CONVERGED' if self.converged else 'NOT CONVERGED'} "
            f"({self.stop_reason}); finest={finest}"
        )
        if self.drift_successive is not None:
            head += (
                f", last-step drift={self.drift_successive:.4g} (tol {self.tol:g}), "
                f"total drift={self.drift_total:.4g}"
            )
        head += (
            f"; {_money(Decimal(str(self.total_cost_usd)))}" if self._budgeted
            else f"; ${self.total_cost_usd:.2f}"
        )
        if self.budget_detail is not None:
            head += f"; {self.budget_detail}"
        rows = "\n".join(
            f"    {r.cells_per_wavelength:>5g} c/λ  dl={r.dl_um:.4f} µm  "
            f"metric={r.metric:.5g}  "
            f"{_money(Decimal(str(r.cost_usd))) if self._budgeted else f'${r.cost_usd:.2f}'}"
            for r in self.ladder
        )
        return head + "\n" + rows


def _budget_limit(budget_usd: float) -> Optional[Decimal]:
    """Validate a supplied budget before a run can start."""
    message = "budget_usd must be a nonnegative real number (positive infinity means no cap)"
    if isinstance(budget_usd, (bool, Fraction)) or not isinstance(budget_usd, (Real, Decimal)):
        raise ValueError(message)
    try:
        if isinstance(budget_usd, Decimal):
            limit = budget_usd
        elif isinstance(budget_usd, Integral):
            limit = Decimal(int(budget_usd))
        else:
            limit = Decimal(str(float(budget_usd)))
    except (OverflowError, TypeError, ValueError):
        raise ValueError(message) from None
    if limit.is_infinite() and limit > 0:
        return None
    if not limit.is_finite() or limit < 0:
        raise ValueError(message)
    return limit


def _money(amount: Decimal) -> str:
    if amount == 0:
        return "$0.00"
    if Decimal("0.01") <= abs(amount) < Decimal("100000000"):
        return f"${amount:.2f}"
    return f"${amount:.6g}"


def _refusal_amounts(limit: Decimal, spent: Decimal, quote: Decimal) -> tuple[str, str, str]:
    """Use one precision and notation for all amounts in a refusal."""
    amounts = tuple(abs(amount) if amount == 0 else amount
                    for amount in (limit, spent, quote))

    def distinct(rendered):
        return all(amounts[i] == amounts[j] or rendered[i] != rendered[j]
                   for i in range(3) for j in range(i + 1, 3))

    max_places = max(2, *(-amount.as_tuple().exponent for amount in amounts))
    if any(abs(amount) >= Decimal("100000000") for amount in amounts) or max_places > 20:
        max_digits = max(len("".join(map(str, amount.as_tuple().digits)).rstrip("0")) or 1
                         for amount in amounts)
        for precision in range(max(1, max_digits) + 1):
            rendered = tuple(f"${amount:.{precision}e}" for amount in amounts)
            if distinct(rendered):
                return rendered
    else:
        for places in range(2, max_places + 1):
            rendered = tuple(f"${amount:.{places}f}" for amount in amounts)
            if distinct(rendered):
                return rendered
    raise AssertionError("budget amounts could not be distinguished")


def _estimate_description(raw) -> str:
    try:
        return repr(raw)[:80]
    except BaseException:
        return f"<{type(raw).__name__}>"


def _exact_sum(left: Decimal, right: Decimal) -> Decimal:
    """Add nonnegative amounts exactly, independent of ambient precision."""
    digits = max(left.adjusted(), right.adjusted()) - min(
        left.as_tuple().exponent, right.as_tuple().exponent) + 2
    with localcontext(Context(prec=max(28, digits), Emax=MAX_EMAX, Emin=MIN_EMIN)):
        return left + right


def _budget_quote(sim, cost_of, limit: Decimal, spent: Decimal) -> tuple[float, Decimal, Optional[str], Optional[Exception]]:
    """Return the planning estimate, refusal detail, and original cause."""
    try:
        raw = cost_of(sim) if cost_of is not None else sim.cost_estimate().usd
    except Exception as exc:
        if cost_of is not None:
            raise  # A caller's cost function error is not a budget refusal.
        cause = exc
        return 0.0, Decimal(0), (
            f"estimate unavailable: budget {_money(limit)}, spent {_money(spent)}, "
            f"estimate call raised {type(exc).__name__}: {str(exc)[:80]}"
        ), cause
    try:
        if isinstance(raw, (bool, Fraction)) or not isinstance(raw, (Real, Decimal)):
            raise ValueError("invalid estimate")
        cost = float(raw)
        if not math.isfinite(cost) or cost < 0 or (cost == 0 and raw != 0):
            raise ValueError("invalid estimate")
        quote = Decimal(str(cost))
    except Exception as exc:
        cause = ValueError(f"estimate returned {_estimate_description(raw)}")
        cause.__cause__ = exc
        return 0.0, Decimal(0), (
            f"estimate unavailable: budget {_money(limit)}, spent {_money(spent)}, "
            f"estimate returned {_estimate_description(raw)}"
        ), cause
    fits = _exact_sum(spent, quote) <= limit
    if not fits:
        budget_text, spent_text, quote_text = _refusal_amounts(limit, spent, quote)
        return cost, quote, (
            f"budget exhausted: budget {budget_text}, spent {spent_text}, "
            f"estimate {quote_text} did not fit"
        ), None
    return cost, quote, None, None


def _record_spent(exc: BaseException, amount: float) -> None:
    """Best-effort exception metadata must never replace the original error."""
    try:
        exc.spent_usd = amount
    except BaseException:
        pass
    try:
        note = f"Replication planning estimate spent: {_money(Decimal(str(amount)))}"
        try:
            exc.add_note(note)
        except BaseException:
            try:
                object.__setattr__(exc, "__notes__", [*getattr(exc, "__notes__", ()), note])
            except BaseException:
                pass
    except BaseException:
        pass


def auto_converge(
    make_run: Callable[[float], Tuple[object, Callable[[object], float]]],
    run: Callable[[object], object],
    *,
    ladder: Sequence[float],
    tol: float,
    metric_name: str = "metric",
    patience: int = 2,
    budget_usd: Optional[float] = None,
    cost_of: Optional[Callable[[object], float]] = None,
    on_rung: Optional[Callable[[Rung], None]] = None,
    _on_estimate_unavailable: Optional[Callable[[Exception], None]] = None,
) -> ConvergenceReport:
    """Escalate through ``ladder`` (cells-per-wavelength, sorted ascending),
    stopping when ``patience`` consecutive successive-rung drifts fall below
    ``tol`` (converged) or the ``budget_usd`` would be exceeded.

    ``make_run(cpw)`` returns ``(sim, extract)``; ``run(sim)`` returns the run
    data; ``extract(data)`` returns the scalar metric. ``cost_of(sim)`` returns a
    dollar estimate (defaults to ``sim.cost_estimate().usd`` when available).
    A finite budget applies per call and bounds the SDK's planning estimate,
    not a provider's bill. A caller that retries must subtract prior spending.
    Each run needs a usable estimate that fits before it starts. A refused
    first run returns an empty ladder. Spent includes each admitted run, even
    if it fails, plus earlier runs when build or callback work is interrupted.
    Exceptions retain their identity; spent is attached when possible and
    recorded in an exception note where supported.
    """
    def _cost(sim) -> float:
        if cost_of is not None:
            return float(cost_of(sim))
        est = getattr(sim, "cost_estimate", None)
        if est is None:
            return 0.0
        return float(est().usd)

    rungs: List[Rung] = []
    total_cost = 0.0
    limit = _budget_limit(budget_usd) if budget_usd is not None else None
    spent = Decimal(0)
    started_spent = Decimal(0)
    budget_detail = None
    budget_cause = None
    below = 0
    stop_reason = "exhausted"
    for cpw in sorted(ladder):
        try:
            sim, extract = make_run(cpw)
        except BaseException as exc:
            if limit is not None:
                _record_spent(exc, total_cost)
            raise
        if limit is None:
            cost = _cost(sim)
        else:
            try:
                cost, quote, budget_detail, budget_cause = _budget_quote(
                    sim, cost_of, limit, spent)
            except BaseException as exc:
                _record_spent(exc, total_cost)
                raise
            if budget_detail is not None:
                stop_reason = "estimate_unavailable" if budget_cause is not None else "budget"
                if budget_cause is not None and _on_estimate_unavailable is not None:
                    _on_estimate_unavailable(budget_cause)
                break
            spent = _exact_sum(spent, quote)
            total_cost = float(spent)
        if math.isfinite(cost) and cost >= 0:
            started_spent = _exact_sum(started_spent, Decimal(str(cost)))
        try:
            data = run(sim)
            metric = float(extract(data))
        except Exception as exc:  # e.g. the solver aborts on a late-time divergence
            if limit is not None:
                _record_spent(exc, total_cost)
            if not rungs:
                raise  # nothing usable yet — surface the failure
            # the finest rung blew up (a known hazard for dispersive scenes at
            # fine dl); keep the coarser stable rungs and stop here rather than
            # crash the whole ladder.
            stop_reason = "diverged"
            if on_rung is not None:
                try:
                    on_rung(Rung(cells_per_wavelength=float(cpw), dl_um=0.0,
                                 metric=float("nan"), cost_usd=cost,
                                 run_dir=f"FAILED: {type(exc).__name__}"))
                except BaseException as callback_exc:
                    if limit is not None:
                        _record_spent(callback_exc, total_cost)
                    raise
            break
        except BaseException as exc:
            if limit is not None:
                _record_spent(exc, total_cost)
            raise
        dl = float(getattr(getattr(sim, "grid", None), "dl_um", 0.0) or 0.0)
        rung = Rung(cells_per_wavelength=float(cpw), dl_um=dl, metric=metric, cost_usd=cost)
        rungs.append(rung)
        if limit is None:
            total_cost += cost
        if on_rung is not None:
            try:
                on_rung(rung)
            except BaseException as exc:
                if limit is not None:
                    _record_spent(exc, total_cost)
                raise
        if len(rungs) >= 2:
            drift = abs(rungs[-1].metric - rungs[-2].metric)
            below = below + 1 if drift < tol else 0
            if below >= patience:
                stop_reason = "tol"
                break

    converged = stop_reason == "tol"
    return ConvergenceReport(
        converged=converged,
        metric_name=metric_name,
        ladder=tuple(rungs),
        tol=tol,
        patience=patience,
        stop_reason=stop_reason,
        total_cost_usd=total_cost,
        budget_detail=budget_detail,
        budget_cause=(f"{type(budget_cause).__name__}: {budget_cause}"
                      if budget_cause is not None else None),
        _budgeted=limit is not None,
        _spent_usd=float(started_spent),
    )


def _band_centre_index(wavelengths_um: Sequence[float], center_um: float) -> int:
    return min(range(len(wavelengths_um)), key=lambda i: abs(wavelengths_um[i] - center_um))


def converge_through_transmission(
    spec,
    *,
    run,
    ladder: Optional[Sequence[float]] = None,
    tol_pp: Optional[float] = None,
    patience: int = 2,
    budget_usd: Optional[float] = None,
    build_kwargs: Optional[dict] = None,
    on_rung: Optional[Callable[[Rung], None]] = None,
) -> ConvergenceReport:
    """Convergence gate on the band-centre THROUGH-port transmission fraction —
    the primary quantity (insertion loss is ``-10log10`` of it). Drift is judged
    in percentage points (``tol_pp``); a spec's ``convergence.tol_pp`` is the
    default.

    ``run`` is applied to each resolution's ``Simulation``. A finite
    ``budget_usd`` applies per call and bounds the SDK's planning estimate,
    not a provider's bill. A caller that retries must subtract prior spending.
    Spent includes each admitted run, including failed runs, and earlier runs
    if build or callback work is interrupted.
    """
    from .build import build_simulation

    ladder = ladder if ladder is not None else spec.convergence.ladder_cpw
    tol_pp = tol_pp if tol_pp is not None else spec.convergence.tol_pp
    bkw = dict(build_kwargs or {})

    def make_run(cpw):
        built = build_simulation(spec, cells_per_wavelength=cpw, **bkw)
        ic = _band_centre_index(list(built.wavelengths_um), spec.optical.center_um)

        def extract(data):
            trans = built.transmissions(data)
            return trans["through"][built.freqs_hz[ic]]

        return built.sim, extract

    return auto_converge(
        make_run,
        run,
        ladder=ladder,
        tol=tol_pp / 100.0,   # percentage points -> transmission fraction
        metric_name="through_transmission",
        patience=patience,
        budget_usd=budget_usd,
        on_rung=on_rung,
    )
