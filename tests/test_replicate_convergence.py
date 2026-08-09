"""Tests for the convergence gate logic (auto_converge), driven by a mock
build+run so no engine is needed. Verifies the successive-drift stop rule with
patience, the budget stop, the exhausted case, and the report."""

import pytest

from photonhub.replicate.convergence import ConvergenceReport, Rung, auto_converge


class _FakeSim:
    """Minimal stand-in with a cost_estimate().usd."""

    def __init__(self, dl_um, usd):
        self.grid = type("G", (), {"dl_um": dl_um})()
        self._usd = usd

    def cost_estimate(self):
        return type("C", (), {"usd": self._usd})()


def _make_run_factory(values_by_cpw, usd=1.0):
    """make_run(cpw) -> (sim, extract) returning the scripted metric."""
    def make_run(cpw):
        sim = _FakeSim(dl_um=1.0 / cpw, usd=usd)

        def extract(_data):
            return values_by_cpw[cpw]

        return sim, extract

    return make_run


def _run(_sim):
    return None  # the mock extract ignores data


def test_converges_when_metric_flattens():
    # metric settles: successive drifts 0.02, 0.002, 0.0005 -> two below tol(0.005)
    values = {10: 0.90, 15: 0.92, 20: 0.922, 25: 0.9225, 30: 0.9226}
    report = auto_converge(
        _make_run_factory(values), _run,
        ladder=[10, 15, 20, 25, 30], tol=0.005, patience=2,
        metric_name="through_transmission",
    )
    assert report.converged
    assert report.stop_reason == "tol"
    # drifts: 10->15=0.02, 15->20=0.002 (<tol, below=1), 20->25=0.0005 (<tol,
    # below=2 => stop). Stops at cpw=25 without paying for cpw=30.
    assert report.ladder[-1].cells_per_wavelength == 25
    assert report.drift_successive == pytest.approx(0.0005, abs=1e-9)


def test_patience_prevents_early_stop_on_single_flat_step():
    # a flat pair (15->20 drift 0) then a jump (20->25 drift 0.02): must NOT
    # declare converged at the flat pair with patience=2
    values = {10: 0.80, 15: 0.90, 20: 0.90, 25: 0.92, 30: 0.9201}
    report = auto_converge(
        _make_run_factory(values), _run,
        ladder=[10, 15, 20, 25, 30], tol=0.005, patience=2,
    )
    # convergence only when 25->30 (0.0001) follows a sub-tol step; but 20->25 is
    # 0.02 (not sub-tol), so the two sub-tol steps are 15->20 and 25->30 which are
    # not consecutive => only converges if a later consecutive pair appears.
    assert report.ladder[-1].cells_per_wavelength == 30
    # 25->30 is one sub-tol step; needs a second consecutive one -> not converged
    assert not report.converged
    assert report.stop_reason == "exhausted"


def test_budget_stops_before_expensive_rung():
    values = {10: 0.9, 20: 0.91, 30: 0.911}
    # each rung $5; budget $12 admits 10 and 20, refuses 30
    report = auto_converge(
        _make_run_factory(values, usd=5.0), _run,
        ladder=[10, 20, 30], tol=1e-9, patience=2, budget_usd=12.0,
    )
    assert report.stop_reason == "budget"
    assert [r.cells_per_wavelength for r in report.ladder] == [10, 20]
    assert report.total_cost_usd == pytest.approx(10.0)


def test_report_fields_and_summary():
    values = {10: 0.90, 20: 0.92, 30: 0.9205}
    report = auto_converge(
        _make_run_factory(values), _run,
        ladder=[10, 20, 30], tol=0.005, patience=1,
    )
    d = report.to_dict()
    assert d["metric_name"] == "metric"
    assert len(d["ladder"]) >= 2
    assert d["drift_total"] == pytest.approx(abs(report.ladder[-1].metric - report.ladder[0].metric))
    text = report.summary()
    assert "convergence[" in text
    assert "c/λ" in text


def test_diverging_finest_rung_keeps_stable_rungs():
    # the run() blows up at the finest rung (cpw=30) -> the ladder keeps the
    # stable coarser rungs and stops with stop_reason 'diverged', not a crash
    values = {10: 0.90, 20: 0.93, 30: None}  # None => raise at cpw=30

    def make_run(cpw):
        sim = _FakeSim(dl_um=1.0 / cpw, usd=1.0)

        def extract(_data):
            v = values[cpw]
            if v is None:
                raise RuntimeError("solver reported an error: divergence")
            return v

        return sim, extract

    report = auto_converge(
        make_run, _run, ladder=[10, 20, 30], tol=1e-9, patience=2,
    )
    assert report.stop_reason == "diverged"
    assert not report.converged
    # the two stable rungs survive; the diverged one is not recorded as a rung
    assert [r.cells_per_wavelength for r in report.ladder] == [10, 20]
    assert report.finest == pytest.approx(0.93)


def test_first_rung_failure_reraises():
    def make_run(cpw):
        def extract(_d):
            raise RuntimeError("divergence")
        return _FakeSim(dl_um=0.1, usd=1.0), extract

    with pytest.raises(RuntimeError, match="divergence"):
        auto_converge(make_run, _run, ladder=[10, 20], tol=1e-9, patience=2)


def test_on_rung_callback_fires_per_rung():
    seen = []
    values = {10: 0.9, 20: 0.91, 30: 0.911}
    auto_converge(
        _make_run_factory(values), _run,
        ladder=[10, 20, 30], tol=1e-9, patience=2,
        on_rung=lambda r: seen.append(r.cells_per_wavelength),
    )
    assert seen == [10, 20, 30]
