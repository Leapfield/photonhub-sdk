"""Paid benchmark runners bind the accepted server quote to submission."""

from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import pytest
import photonhub as ph

from .helpers import REPO_ROOT


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quoted_web(monkeypatch, *, result=None):
    seen = {"estimate_devices": [], "run_kwargs": []}
    result = result if result is not None else object()
    monkeypatch.setattr(ph.web, "configure", lambda: None)
    monkeypatch.setattr(ph.web, "gpus", lambda: [{"id": "mi300x"}])

    def estimate(sim, *, device=None):
        seen["estimate_devices"].append(device)
        return {"usd": 0.25, "quote_id": "quote-server-42"}

    monkeypatch.setattr(ph.web, "estimate", estimate)

    def run(sim, **kwargs):
        seen["run_kwargs"].append(kwargs)
        return result

    monkeypatch.setattr(ph.web, "run", run)
    return seen, result


def test_resonator_runner_submits_the_accepted_quote(monkeypatch):
    cloud = _load("_quote_test_resonator_cloud",
                  "benchmarks/resonators/_cloud.py")
    seen, expected = _quoted_web(monkeypatch)
    assert cloud.run_sim(object(), device="gpu", max_usd=1.0) is expected
    assert seen["run_kwargs"][0]["quote_id"] == "quote-server-42"
    assert seen["estimate_devices"] == [seen["run_kwargs"][0]["device"]]


def test_metasurface_runner_submits_the_accepted_quote(monkeypatch):
    metalens = _load("_quote_test_metalens",
                     "benchmarks/metasurface/metalens_meta_atom.py")
    seen, expected = _quoted_web(monkeypatch)
    assert metalens._run(object(), "gpu", max_usd=1.0) is expected
    assert seen["run_kwargs"][0]["quote_id"] == "quote-server-42"
    assert seen["estimate_devices"] == [seen["run_kwargs"][0]["device"]]


def test_gds_runner_submits_the_accepted_quote(monkeypatch):
    gds = _load("_quote_test_gds", "benchmarks/gds/run_ph_gpu.py")
    data = SimpleNamespace(
        provenance={}, steps_run=1, shut_off=False, aborted=False)
    seen, _ = _quoted_web(monkeypatch, result=data)
    monkeypatch.setattr(
        ph.web, "account",
        lambda: {"email": "test@example.invalid", "available_usd": 10.0})
    monkeypatch.setattr(gds, "readout_and_save", lambda *args, **kwargs: 0)
    args = SimpleNamespace(
        force=False, preflight=False, max_usd=1.0, yes=True,
        automesh=False, timeout=30.0, save=None)
    scene = SimpleNamespace(sim=object())
    assert gds.run_cloud(scene, "mode_converter", 10, args) == 0
    assert seen["run_kwargs"][0]["quote_id"] == "quote-server-42"
    assert seen["estimate_devices"] == [seen["run_kwargs"][0]["device"]]


@pytest.mark.parametrize(
    ("estimate", "error_match"),
    [
        pytest.param({"usd": 0.25}, "quote_id", id="missing-quote-id"),
        pytest.param(
            {"usd": 0.25, "quote_id": "   "}, "quote_id",
            id="blank-quote-id"),
        pytest.param(
            {"usd": 0.25, "quote_id": 42}, "quote_id",
            id="non-string-quote-id"),
        pytest.param(
            {"usd": True, "quote_id": "quote-server-42"}, "usd",
            id="boolean-usd"),
        pytest.param(
            {"usd": -0.01, "quote_id": "quote-server-42"}, "usd",
            id="negative-usd"),
        pytest.param(
            {"usd": "0.25", "quote_id": "quote-server-42"}, "usd",
            id="non-number-usd"),
        pytest.param(
            {"usd": float("inf"), "quote_id": "quote-server-42"}, "usd",
            id="non-finite-usd"),
    ],
)
@pytest.mark.parametrize("runner", ["resonator", "metasurface", "gds"])
def test_paid_runner_refuses_invalid_quote_without_submitting(
        monkeypatch, runner, estimate, error_match):
    monkeypatch.setattr(ph.web, "configure", lambda: None)
    monkeypatch.setattr(ph.web, "gpus", lambda: [{"id": "mi300x"}])
    monkeypatch.setattr(
        ph.web, "estimate", lambda sim, *, device=None: estimate)
    submissions = []
    monkeypatch.setattr(
        ph.web, "run", lambda *args, **kwargs: submissions.append(kwargs))

    if runner == "resonator":
        module = _load("_quote_reject_resonator", "benchmarks/resonators/_cloud.py")
        with pytest.raises(RuntimeError, match=error_match):
            module.run_sim(object(), device="gpu", max_usd=1.0)
    elif runner == "metasurface":
        module = _load(
            "_quote_reject_metalens",
            "benchmarks/metasurface/metalens_meta_atom.py")
        with pytest.raises(RuntimeError, match=error_match):
            module._run(object(), "gpu", max_usd=1.0)
    else:
        module = _load("_quote_reject_gds", "benchmarks/gds/run_ph_gpu.py")
        monkeypatch.setattr(
            ph.web, "account",
            lambda: {"email": "test@example.invalid", "available_usd": 10.0})
        args = SimpleNamespace(
            force=False, preflight=False, max_usd=1.0, yes=True,
            automesh=False, timeout=30.0, save=None)
        assert module.run_cloud(
            SimpleNamespace(sim=object()), "mode_converter", 10, args) == 3

    assert submissions == []


def test_cloud_beta_notebook_has_opt_in_and_combined_five_dollar_guard():
    notebook = json.loads(
        (REPO_ROOT / "examples/notebooks/10_cloud_gpu_run.ipynb").read_text(
            encoding="utf-8"))
    code = ["".join(cell.get("source", [])) for cell in notebook["cells"]
            if cell.get("cell_type") == "code"]
    setup = next(source for source in code if "RUN_PAID =" in source)
    preflight = next(source for source in code if "total_quote = sum" in source)
    submit = next(source for source in code if "empty_job = ph.web.run_async" in source)

    assert 'PHOTONHUB_RUN_PAID") == "1"' in setup
    assert 'PHOTONHUB_MAX_TOTAL_USD", "5.00"' in setup
    assert preflight.count("ph.web.preflight(") == 2
    assert "if total_quote > MAX_TOTAL_USD" in preflight
    assert "if total_quote > available" in preflight
    assert submit.index("elif not RUN_PAID") < submit.index("ph.web.run_async")
    assert submit.count("quote_id=checks[") == 2
    assert "ph.web.estimate(" not in submit
