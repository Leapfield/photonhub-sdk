"""Automated convergence gate — run a device at escalating resolution until the
figure-of-merit stops moving, and stamp the evidence into the result.

The user's standing bar is *convergence demonstrated, not assumed*. Nothing in
the stack looped build -> run -> extract -> drift -> stop; the pieces existed
(``benchmarks/gds/framework/metrics.convergence`` had zero callers,
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
    stop_reason: str  # "tol" | "budget" | "exhausted"
    total_cost_usd: float

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
        return {
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

    def summary(self) -> str:
        head = (
            f"convergence[{self.metric_name}]: "
            f"{'CONVERGED' if self.converged else 'NOT CONVERGED'} "
            f"({self.stop_reason}); finest={self.finest:.5g}"
        )
        if self.drift_successive is not None:
            head += (
                f", last-step drift={self.drift_successive:.4g} (tol {self.tol:g}), "
                f"total drift={self.drift_total:.4g}"
            )
        head += f"; ${self.total_cost_usd:.2f}"
        rows = "\n".join(
            f"    {r.cells_per_wavelength:>5g} c/λ  dl={r.dl_um:.4f} µm  "
            f"metric={r.metric:.5g}  ${r.cost_usd:.2f}"
            for r in self.ladder
        )
        return head + "\n" + rows


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
) -> ConvergenceReport:
    """Escalate through ``ladder`` (cells-per-wavelength, sorted ascending),
    stopping when ``patience`` consecutive successive-rung drifts fall below
    ``tol`` (converged) or the ``budget_usd`` would be exceeded.

    ``make_run(cpw)`` returns ``(sim, extract)``; ``run(sim)`` returns the run
    data; ``extract(data)`` returns the scalar metric. ``cost_of(sim)`` returns a
    dollar estimate (defaults to ``sim.cost_estimate().usd`` when available).
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
    below = 0
    stop_reason = "exhausted"
    for cpw in sorted(ladder):
        sim, extract = make_run(cpw)
        cost = _cost(sim)
        if budget_usd is not None and rungs and (total_cost + cost) > budget_usd:
            stop_reason = "budget"
            break
        try:
            data = run(sim)
            metric = float(extract(data))
        except Exception as exc:  # e.g. the solver aborts on a late-time divergence
            if not rungs:
                raise  # nothing usable yet — surface the failure
            # the finest rung blew up (a known hazard for dispersive scenes at
            # fine dl); keep the coarser stable rungs and stop here rather than
            # crash the whole ladder.
            stop_reason = "diverged"
            if on_rung is not None:
                on_rung(Rung(cells_per_wavelength=float(cpw), dl_um=0.0,
                             metric=float("nan"), cost_usd=cost,
                             run_dir=f"FAILED: {type(exc).__name__}"))
            break
        dl = float(getattr(getattr(sim, "grid", None), "dl_um", 0.0) or 0.0)
        rung = Rung(cells_per_wavelength=float(cpw), dl_um=dl, metric=metric, cost_usd=cost)
        rungs.append(rung)
        total_cost += cost
        if on_rung is not None:
            on_rung(rung)
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

    ``run`` is the run function (e.g. ``photonhub.run_local`` or a GPU runner)
    applied to each rung's ``Simulation``.
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
