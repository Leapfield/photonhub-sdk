"""Self-contained tests for the cloud client (``ph.web``) with a mocked HTTP
transport — no server, no photonhub-cloud dependency. Proves the cloud path
returns the same SimulationData/Job/SolverRunError as the local path."""

import importlib
import io
import json
import struct
import tarfile
from pathlib import Path

import numpy as np
import pytest

import photonhub as ph
from photonhub.runners.batch import BatchData, Job
from photonhub.runners.local import SolverRunError

# the `run` submodule (the name `photonhub.web.run` is shadowed by the run
# function re-export, so reach the module object explicitly)
_runmod = importlib.import_module("photonhub.web.run")
_actions = importlib.import_module("photonhub.web.actions")
_batchmod = importlib.import_module("photonhub.web.batch")


@pytest.mark.parametrize(
    ("given", "normalized"),
    [("cpu", "cpu"), ("gpu", "gpu"), ("gpu:a4000", "gpu:a4000"),
     ("gpu:mi300x", "gpu:mi300x"), (" gpu:h100 ", "gpu:h100"),
     (None, None)],
)
def test_validate_web_device_accepts_and_normalizes(given, normalized):
    assert _runmod._validate_web_device(given) == normalized


@pytest.mark.parametrize("bad", ["tpu", "gpu:", "gpu:BAD!", "", "gpu:0:1", "cpu:0"])
def test_validate_web_device_rejects(bad):
    with pytest.raises(SolverRunError, match="invalid device"):
        _runmod._validate_web_device(bad)


def test_gpus_returns_curated_menu(monkeypatch, configured):
    menu = [{"id": "a4000", "vendor": "NVIDIA", "arch": "sm_86", "gpu_mem_gb": 16},
            {"id": "mi300x", "vendor": "AMD", "arch": "gfx942", "gpu_mem_gb": 192}]

    class _FakeGpus:
        def __init__(self, cfg):
            pass

        def list_gpus(self):
            return menu

    monkeypatch.setattr(_actions, "HttpClient", _FakeGpus)
    assert ph.web.gpus() == menu
    assert ph.gpus() == menu  # top-level convenience alias


def test_estimate_normalizes_and_binds_selected_device(monkeypatch, configured):
    seen = []

    class _FakeEstimate:
        def __init__(self, cfg):
            pass

        def estimate(self, spec, *, device=None, solver=None):
            seen.append(device)
            return {"usd": 0.25, "quote_id": "quote-device"}

    monkeypatch.setattr(_actions, "HttpClient", _FakeEstimate)
    quote = ph.web.estimate(_make_sim(), device=" gpu:mi300x ")
    assert quote["quote_id"] == "quote-device"
    assert seen == ["gpu:mi300x"]


def _bundle_bytes() -> bytes:
    """A tar.gz of a minimal manifest.json + probe.bin, as the server returns."""
    manifest = {
        "manifest_version": "1",
        "monitors": [{"name": "probe", "type": "field_time", "file": "probe.bin",
                      "dtype": "float32", "shape": [5, 2],
                      "dims": ["sample", "component"], "components": ["Ez", "Hx"],
                      "sample_steps": [1, 2, 3, 4, 5], "dt_s": 1e-16}],
        "run": {"n_steps": 5, "steps_run": 5, "dt_s": 1e-16},
        "grid": {"shape": [4, 4, 4], "dl_um": 0.05},
        "provenance": {"solver_version": "fake", "device_name": "fake"},
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in [("manifest.json",
                            json.dumps(manifest).encode()),
                           ("probe.bin", struct.pack("<10f", *range(10)))]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeHttp:
    """Stands in for web.client.HttpClient. ``mode`` picks the job outcome."""

    def __init__(self, cfg, mode="ok"):
        self.cfg = cfg
        self.mode = mode

    def submit_job(self, spec, *, name=None, **kw):
        # Unique per-name ids: real submissions never share a job_id, and a
        # shared id made concurrent Batch entries fetch into ONE cache dir
        # (the CI-flaky torn-manifest read; extraction is atomic now, but the
        # fake should model reality regardless).
        return {"job_id": name or "job-1"}

    def estimate(self, spec, *, device=None, solver=None):
        return {"usd": 0.01, "quote_id": "quote-test-1"}

    def get_job(self, job_id, *, deadline=None):
        if self.mode == "running":
            return {"state": "running", "progress": {"step": 1, "total": 5}}
        if self.mode == "fail":
            return {"state": "failed",
                    "error": {"reason": "divergence", "stderr_tail": "boom"}}
        return {"state": "succeeded"}

    def download_result(self, job_id):
        return _bundle_bytes()


@pytest.fixture(autouse=True)
def _isolated_web_env(monkeypatch):
    """No test here may see the developer's real cloud credentials: ph.web
    falls back to $PHOTONHUB_*/$PHOTONHUB_* (web/config.py get_config), so a
    leaked key makes the "unconfigured" tests submit REAL paid cloud jobs.
    Scrub both prefixes and clear the module-global config around every test;
    tests that want env values set them explicitly via monkeypatch.setenv."""
    for prefix in ("SIMUPOD", "PHOTONHUB"):
        for suffix in ("API_KEY", "URL", "CACHE_DIR"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    ph.web.reset()
    yield
    ph.web.reset()


@pytest.fixture
def configured(tmp_path):
    ph.web.configure(api_key="ph_test_x", url="http://localhost:0",
                     cache_dir=tmp_path / "cache", poll_interval_s=0.0,
                     poll_backoff_max_s=0.0)
    yield
    ph.web.reset()


def _patch(monkeypatch, mode):
    factory = lambda cfg: FakeHttp(cfg, mode)
    monkeypatch.setattr(_runmod, "HttpClient", factory)
    monkeypatch.setattr(_batchmod, "HttpClient", factory)


def _make_sim():
    return ph.Simulation(
        size_um=(0.2, 0.2, 0.2), grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        sources=[ph.PointDipole(center_um=(0.1, 0.1, 0.1), polarization="Ez",
                                source_time=ph.GaussianPulse(freq0_hz=1.934e14,
                                                             fwidth_hz=4e13))],
        monitors=[ph.FieldTimeMonitor(name="probe", center_um=(0.15, 0.1, 0.1),
                                      fields=["Ez"])])


def test_unconfigured_raises_weberror():
    ph.web.reset()
    with pytest.raises(ph.web.WebError):
        ph.web.run(_make_sim())


def test_run_returns_simulationdata(monkeypatch, configured):
    _patch(monkeypatch, "ok")
    data = ph.web.run(_make_sim())
    assert isinstance(data, ph.SimulationData)
    assert "probe" in data.monitor_names
    np.testing.assert_array_equal(
        data["probe"].values.ravel(), np.arange(10, dtype="float32"))


def test_run_async_is_same_job_type(monkeypatch, configured):
    _patch(monkeypatch, "ok")
    job = ph.web.run_async(_make_sim())
    assert isinstance(job, Job)                 # the SAME handle as ph.run_async
    assert job.job_id == "job-1"
    data = job.result(timeout=5)
    assert isinstance(data, ph.SimulationData)
    assert job.done is True


def test_run_async_surfaces_submission_failure_synchronously(monkeypatch,
                                                             configured):
    class _RejectSubmit(FakeHttp):
        def submit_job(self, *args, **kwargs):
            raise ph.web.WebError("authentication rejected", status_code=401)

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _RejectSubmit(cfg))
    with pytest.raises(ph.web.WebError, match="authentication rejected"):
        ph.web.run_async(_make_sim())


def test_post_submit_transport_error_carries_resumable_job_id(
        monkeypatch, configured):
    class _PollBreaks(FakeHttp):
        def get_job(self, job_id, *, deadline=None):
            raise ph.web.WebError("connection lost")

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _PollBreaks(cfg))
    job = ph.web.run_async(_make_sim())
    with pytest.raises(ph.web.WebError, match="connection lost") as exc:
        job.result(timeout=5)
    assert exc.value.job_id == job.job_id == "job-1"


def test_invalid_result_bundle_is_weberror_with_resumable_job_id(
        monkeypatch, configured):
    class _BadBundle(FakeHttp):
        def download_result(self, job_id):
            return b"not a gzip tar"

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _BadBundle(cfg))
    job = ph.web.run_async(_make_sim())
    with pytest.raises(ph.web.WebError, match="invalid result bundle") as exc:
        job.result(timeout=5)
    assert exc.value.job_id == job.job_id == "job-1"


def test_progress_callback_oserror_is_not_mislabeled_as_cloud_error(
        monkeypatch, configured):
    _patch(monkeypatch, "running")

    def broken_progress(_):
        raise OSError("log sink full")

    job = ph.web.run_async(_make_sim(), progress=broken_progress)
    with pytest.raises(OSError, match="log sink full"):
        job.result(timeout=5)


def test_provisioning_is_a_normal_active_service_state(monkeypatch, configured):
    class _ProvisioningThenDone(FakeHttp):
        polls = 0

        def get_job(self, job_id, *, deadline=None):
            self.polls += 1
            if self.polls == 1:
                return {"state": "provisioning"}
            return {"state": "succeeded"}

    instance = _ProvisioningThenDone(None)
    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: instance)
    result = ph.web.run_async(_make_sim()).result(timeout=5)
    assert isinstance(result, ph.SimulationData)
    assert instance.polls == 2


def test_unknown_service_state_fails_resumably_instead_of_hanging(
        monkeypatch, configured):
    class _UnknownState(FakeHttp):
        def get_job(self, job_id, *, deadline=None):
            return {"state": "teleporting"}

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _UnknownState(cfg))
    job = ph.web.run_async(_make_sim())
    with pytest.raises(ph.web.WebError, match="unknown state") as exc:
        job.result(timeout=5)
    assert exc.value.job_id == "job-1"


@pytest.mark.parametrize("entrypoint", ["run", "run_async"])
def test_run_entrypoints_bind_accepted_quote_to_submission(
        monkeypatch, configured, entrypoint):
    seen = []

    class _CaptureQuote(FakeHttp):
        def submit_job(self, spec, *, name=None, quote_id=None, **kw):
            seen.append(quote_id)
            return {"job_id": "job-quoted"}

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _CaptureQuote(cfg))
    result = getattr(ph.web, entrypoint)(
        _make_sim(), quote_id="quote-accepted-42")
    if isinstance(result, Job):
        result = result.result(timeout=5)
    assert isinstance(result, ph.SimulationData)
    assert seen == ["quote-accepted-42"]


def test_invalid_quote_id_fails_before_submission(monkeypatch, configured):
    class _MustNotSubmit(FakeHttp):
        def submit_job(self, *args, **kwargs):
            raise AssertionError("invalid quote id must fail before submission")

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _MustNotSubmit(cfg))
    with pytest.raises(ValueError, match="quote_id"):
        ph.web.run_async(_make_sim(), quote_id="   ")


@pytest.mark.parametrize("entrypoint", ["run_quoted", "run_quoted_async"])
def test_single_job_spend_guard_checks_balance_and_binds_quote(
        monkeypatch, configured, entrypoint):
    submissions = []

    class _GuardedHttp(FakeHttp):
        def account(self):
            return {"available_usd": 5.0, "balance_usd": 5.0}

        def estimate(self, spec, *, device=None, solver=None):
            assert device == "gpu"
            return {
                "usd": 0.25,
                "quote_id": "quote-beta-five",
                "expires_at": "2099-01-01T00:00:00Z",
            }

        def submit_job(self, spec, *, name=None, quote_id=None, device=None,
                       **kwargs):
            submissions.append((quote_id, device))
            return {"job_id": "job-guarded"}

    factory = lambda cfg: _GuardedHttp(cfg)
    monkeypatch.setattr(_actions, "HttpClient", factory)
    monkeypatch.setattr(_runmod, "HttpClient", factory)

    result = getattr(ph.web, entrypoint)(_make_sim())
    if isinstance(result, Job):
        result = result.result(timeout=5)
    assert isinstance(result, ph.SimulationData)
    assert submissions == [("quote-beta-five", "gpu")]


def test_preflight_is_inspectable_without_leaking_quote_id(
        monkeypatch, configured):
    class _PreflightHttp(FakeHttp):
        def account(self):
            return {"available_micros": 5_000_000}

        def estimate(self, spec, *, device=None, solver=None):
            return {
                "usd": 0.25,
                "quote_id": "quote-secret-never-display",
                "expires_at": "2099-01-01T00:00:00Z",
            }

    monkeypatch.setattr(_actions, "HttpClient", lambda cfg: _PreflightHttp(cfg))
    check = ph.web.preflight(_make_sim())
    assert isinstance(check, ph.web.CloudPreflight)
    assert check.quote_usd == pytest.approx(0.25)
    assert check.available_usd == pytest.approx(5.0)
    assert check.remaining_usd == pytest.approx(4.75)
    assert "quote-secret-never-display" not in repr(check)
    assert "quote-secret-never-display" not in check.summary()
    assert "limit $5.00" in check.summary()


@pytest.mark.parametrize(
    ("account", "quote", "match"),
    [
        ({"available_usd": 5.0}, {"usd": 5.01, "quote_id": "q"},
         "exceeds max_usd"),
        ({"available_usd": 0.20}, {"usd": 0.25, "quote_id": "q"},
         "exceeds available balance"),
        ({"balance_usd": 5.0}, {"usd": 0.25, "quote_id": "q"},
         "available_usd"),
        ({"available_usd": 5.0}, {"usd": 0.25}, "quote_id"),
        ({"available_usd": 5.0}, {"usd": float("nan"), "quote_id": "q"},
         "finite non-negative"),
        ({"available_usd": "5.0"}, {"usd": 0.25, "quote_id": "q"},
         "finite non-negative"),
        ({"available_usd": 5.0}, {"usd": "0.25", "quote_id": "q"},
         "finite non-negative"),
    ],
)
def test_single_job_spend_guard_fails_before_submission(
        monkeypatch, configured, account, quote, match):
    submissions = []

    class _RejectedPreflight(FakeHttp):
        def account(self):
            return account

        def estimate(self, spec, *, device=None, solver=None):
            return quote

    monkeypatch.setattr(
        _actions, "HttpClient", lambda cfg: _RejectedPreflight(cfg))
    monkeypatch.setattr(
        _runmod, "HttpClient",
        lambda cfg: submissions.append(cfg) or FakeHttp(cfg))
    with pytest.raises(ph.web.WebError, match=match):
        ph.web.run_quoted(_make_sim())
    assert submissions == []


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf"), "x"])
def test_single_job_spend_guard_rejects_invalid_limit_before_network(
        monkeypatch, configured, bad):
    calls = []
    monkeypatch.setattr(
        _actions, "HttpClient", lambda cfg: calls.append(cfg))
    with pytest.raises(ValueError, match="max_usd"):
        ph.web.preflight(_make_sim(), max_usd=bad)
    assert calls == []


def test_public_job_history_and_status_normalize_costs(
        monkeypatch, configured):
    records = [{
        "job_id": "job-1",
        "state": "succeeded",
        "quote_micros": 250_000,
        "actual_micros": 125_000,
        "refunded_micros": 125_000,
    }]

    class _HistoryHttp:
        def __init__(self, cfg):
            pass

        def list_jobs(self):
            return records

        def get_job(self, job_id):
            assert job_id == "job-1"
            return records[0]

    monkeypatch.setattr(_actions, "HttpClient", _HistoryHttp)
    history = ph.web.list_jobs()
    assert history[0]["quote_usd"] == pytest.approx(0.25)
    assert history[0]["actual_usd"] == pytest.approx(0.125)
    assert history[0]["refunded_usd"] == pytest.approx(0.125)
    status = ph.web.job_status("job-1")
    assert status["state"] == "succeeded"
    assert status["actual_usd"] == pytest.approx(0.125)
    assert records[0].keys() == {
        "job_id", "state", "quote_micros", "actual_micros", "refunded_micros"}


def test_public_job_status_rejects_bad_service_shape(monkeypatch, configured):
    class _BadStatus:
        def __init__(self, cfg):
            pass

        def get_job(self, job_id):
            return []

    monkeypatch.setattr(_actions, "HttpClient", _BadStatus)
    with pytest.raises(ph.web.WebError, match="not an object") as exc:
        ph.web.job_status("job-1")
    assert exc.value.job_id == "job-1"


def test_submission_rejects_unsafe_service_job_id(monkeypatch, configured):
    class _BadId(FakeHttp):
        def submit_job(self, *args, **kwargs):
            return {"job_id": "../outside"}

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _BadId(cfg))
    with pytest.raises(ph.web.WebError, match="invalid job_id"):
        ph.web.run_async(_make_sim())


def test_failure_raises_solverrunerror(monkeypatch, configured):
    _patch(monkeypatch, "fail")
    job = ph.web.run_async(_make_sim())
    with pytest.raises(SolverRunError) as ei:
        job.result(timeout=5)
    assert "divergence" in str(ei.value)
    assert ei.value.stderr_tail == "boom"


def test_failed_status_ignores_non_string_stderr_tail(monkeypatch, configured):
    class _BadErrorShape(FakeHttp):
        def get_job(self, job_id, *, deadline=None):
            return {
                "state": "failed",
                "error": {"reason": "solver failed", "stderr_tail": {"x": 1}},
            }

    monkeypatch.setattr(
        _runmod, "HttpClient", lambda cfg: _BadErrorShape(cfg))
    job = ph.web.run_async(_make_sim())
    with pytest.raises(SolverRunError, match="solver failed") as exc:
        job.result(timeout=5)
    assert exc.value.stderr_tail is None


def test_caller_wait_timeout_can_retry_same_job(monkeypatch, configured):
    class _EventuallySucceeds(FakeHttp):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.polls = 0

        def get_job(self, job_id, *, deadline=None):
            import time
            self.polls += 1
            time.sleep(0.01)
            return {"state": "running" if self.polls < 3 else "succeeded"}

    monkeypatch.setattr(
        _runmod, "HttpClient", lambda cfg: _EventuallySucceeds(cfg))
    job = ph.web.run_async(_make_sim())
    with pytest.raises(TimeoutError):
        job.result(timeout=0.001)
    assert isinstance(job.result(timeout=5), ph.SimulationData)
    assert job.done is True


def test_internal_poll_timeout_exposes_resumable_job_id(monkeypatch, configured):
    _patch(monkeypatch, "running")
    job = ph.web.run_async(_make_sim(), timeout=0.01)
    with pytest.raises(ph.web.WebJobTimeout) as exc:
        job.result(timeout=1)
    assert exc.value.job_id == job.job_id == "job-1"


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan"), True])
def test_internal_poll_timeout_rejects_invalid_values(monkeypatch, configured,
                                                      timeout):
    class _MustNotSubmit(FakeHttp):
        def submit_job(self, *args, **kwargs):
            raise AssertionError("invalid timeout must fail before submission")

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _MustNotSubmit(cfg))
    with pytest.raises(ValueError, match="timeout"):
        ph.web.run_async(_make_sim(), timeout=timeout)


def test_zero_internal_timeout_is_immediate_and_resumable(monkeypatch,
                                                          configured):
    _patch(monkeypatch, "ok")
    job = ph.web.run_async(_make_sim(), timeout=0)
    with pytest.raises(ph.web.WebJobTimeout) as exc:
        job.result(timeout=5)
    assert exc.value.job_id == "job-1"


def test_resume_polls_existing_job_without_resubmitting(monkeypatch, configured):
    class _ResumeOnly(FakeHttp):
        def submit_job(self, *args, **kwargs):
            raise AssertionError("resume must not submit a new job")

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _ResumeOnly(cfg))
    job = ph.web.resume("existing-42")
    assert job.job_id == "existing-42"
    assert isinstance(job.result(timeout=5), ph.SimulationData)


def test_timeout_then_resume_same_id_without_duplicate_submit(monkeypatch,
                                                              configured):
    state = {"ready": False, "submits": 0, "downloads": 0}

    class _Stateful(FakeHttp):
        def submit_job(self, *args, **kwargs):
            state["submits"] += 1
            return {"job_id": "job-stateful"}

        def get_job(self, job_id, *, deadline=None):
            return {"state": "succeeded" if state["ready"] else "running"}

        def download_result(self, job_id):
            state["downloads"] += 1
            return _bundle_bytes()

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _Stateful(cfg))
    first = ph.web.run_async(_make_sim(), timeout=0.005)
    with pytest.raises(ph.web.WebJobTimeout) as exc:
        first.result(timeout=5)
    assert exc.value.job_id == first.job_id == "job-stateful"

    state["ready"] = True
    resumed = ph.web.resume(exc.value.job_id)
    assert isinstance(resumed.result(timeout=5), ph.SimulationData)
    assert state == {"ready": True, "submits": 1, "downloads": 1}


@pytest.mark.parametrize(
    "job_id", ["", " ../outside", "../outside", "/tmp/result", "job?x=1",
               "job#fragment", "job/name", "job\\name"],
)
def test_resume_rejects_unsafe_job_ids(configured, job_id):
    with pytest.raises(ValueError, match="job_id"):
        ph.web.resume(job_id)


def test_cancel_rejects_unsafe_job_id_before_configuration():
    ph.web.reset()
    with pytest.raises(ValueError, match="job_id"):
        ph.web.cancel("../outside")


def test_cancel_posts_once_and_returns_cost_status(monkeypatch, configured):
    calls = []

    class _FakeCancel:
        def __init__(self, cfg):
            pass

        def cancel_job(self, job_id):
            calls.append(job_id)
            return {
                "job_id": job_id, "state": "cancelled",
                "actual_micros": 250_000, "refunded_micros": 750_000,
            }

    monkeypatch.setattr(_actions, "HttpClient", _FakeCancel)
    response = ph.web.cancel("job-cancel-me")
    assert calls == ["job-cancel-me"]
    assert response == {
        "job_id": "job-cancel-me", "state": "cancelled",
        "actual_micros": 250_000, "refunded_micros": 750_000,
    }


def test_concurrent_same_job_fetch_is_atomic(tmp_path):
    """N threads fetching the SAME job id must all see a complete bundle —
    the pre-atomic cache let two extractions interleave in one directory and
    a reader could hit a torn manifest.json (the CI JSONDecodeError flake)."""
    import concurrent.futures as cf

    from photonhub.web import cache as _cache
    from photonhub.web.config import WebConfig

    cfg = WebConfig(url="http://localhost:0", api_key="ph_test_x",
                    cache_dir=tmp_path / "cache")

    class _SlowHttp:
        def download_result(self, job_id):
            import time
            time.sleep(0.01)          # widen the extract window
            return _bundle_bytes()

    def fetch(_):
        out = _cache.download_bundle(_SlowHttp(), cfg, "job-shared")
        return json.loads((out / "manifest.json").read_text())

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        manifests = list(ex.map(fetch, range(16)))
    assert all(m["manifest_version"] == "1" for m in manifests)
    # no stray tmp dirs left behind
    assert not list((tmp_path / "cache").glob("*.tmp-*"))


def test_stale_cache_without_private_marker_is_redownloaded(tmp_path):
    from photonhub.web import cache as _cache
    from photonhub.web.config import WebConfig

    cfg = WebConfig(url="http://localhost:0", api_key="ph_test_x",
                    cache_dir=tmp_path / "cache")
    out = _cache.job_dir(cfg, "job-stale")
    extract_bundle(_bundle_bytes(), out)
    assert not (out / COMPLETION_MARKER).exists()
    calls = []

    class _CountingHttp:
        def download_result(self, job_id):
            calls.append(job_id)
            return _bundle_bytes()

    result = _cache.download_bundle(_CountingHttp(), cfg, "job-stale")
    assert calls == ["job-stale"]
    assert result == out
    assert (out / COMPLETION_MARKER).is_file()
    assert _cache._is_complete(out)


def test_batch_partial_failure(monkeypatch, configured):
    # good sims succeed; a sim whose name marker == "boom" fails. Drive it by
    # routing each name through a mode-specific fake.
    factory = lambda cfg: FakeHttp(cfg, "ok")
    monkeypatch.setattr(_runmod, "HttpClient", factory)
    monkeypatch.setattr(_batchmod, "HttpClient", factory)
    bd = ph.web.Batch({"a": _make_sim(), "b": _make_sim()}).run(max_usd=1)
    assert isinstance(bd, BatchData)
    assert isinstance(bd["a"], ph.SimulationData)
    assert isinstance(bd["b"], ph.SimulationData)


# --------------------------------------------------------------------------
# Gaps not covered by the six tests above: cache path-traversal guard,
# config env-var precedence, client 4xx->WebError mapping, a batch where one
# name genuinely fails, and the cancelled-job path.
# --------------------------------------------------------------------------


class _NameRoutedHttp:
    """Per-name FakeHttp: routes each job to a mode chosen by the submitted
    ``name`` so a Batch can mix successes and failures in one run. Submission
    returns the name as the job_id; get_job keys its outcome off that id."""

    def __init__(self, cfg, fail_names=()):
        self.cfg = cfg
        self.fail_names = set(fail_names)

    def submit_job(self, spec, *, name=None, **kw):
        return {"job_id": name or "job-1"}

    def estimate(self, spec, *, device=None, solver=None):
        return {"usd": 0.01, "quote_id": "quote-test-1"}

    def get_job(self, job_id, *, deadline=None):
        if job_id in self.fail_names:
            return {"state": "failed",
                    "error": {"reason": "divergence", "stderr_tail": "boom"}}
        return {"state": "succeeded"}

    def download_result(self, job_id):
        return _bundle_bytes()


def test_batch_mixed_success_and_failure(monkeypatch, configured):
    # "b" genuinely fails; the batch must surface it in .errors while "a"
    # succeeds, and indexing the failed name re-raises its SolverRunError.
    factory = lambda cfg: _NameRoutedHttp(cfg, fail_names={"b"})
    monkeypatch.setattr(_runmod, "HttpClient", factory)
    monkeypatch.setattr(_batchmod, "HttpClient", factory)
    bd = ph.web.Batch({"a": _make_sim(), "b": _make_sim()}).run(max_usd=1)
    assert isinstance(bd["a"], ph.SimulationData)
    assert "a" in bd and "b" not in bd          # __contains__ = successes only
    assert set(bd.errors) == {"b"}
    assert isinstance(bd.errors["b"], SolverRunError)
    with pytest.raises(SolverRunError):
        _ = bd["b"]                             # indexing re-raises


def test_batch_preserves_resumable_timeout_and_partial_success(monkeypatch,
                                                               configured):
    class _MixedTimeout(_NameRoutedHttp):
        def get_job(self, job_id, *, deadline=None):
            if job_id == "b":
                return {"state": "running"}
            return {"state": "succeeded"}

    factory = lambda cfg: _MixedTimeout(cfg)
    monkeypatch.setattr(_runmod, "HttpClient", factory)
    monkeypatch.setattr(_batchmod, "HttpClient", factory)
    bd = ph.web.Batch({"a": _make_sim(), "b": _make_sim()}).run(
        timeout=0.01, max_usd=1)
    assert isinstance(bd["a"], ph.SimulationData)
    assert isinstance(bd.errors["b"], ph.web.WebJobTimeout)
    assert bd.errors["b"].job_id == "b"
    with pytest.raises(ph.web.WebJobTimeout):
        _ = bd["b"]


def test_batch_preserves_post_submit_transport_error_and_partial_success(
        monkeypatch, configured):
    class _MixedTransport(_NameRoutedHttp):
        def get_job(self, job_id, *, deadline=None):
            if job_id == "b":
                raise ph.web.WebError("connection lost")
            return {"state": "succeeded"}

    factory = lambda cfg: _MixedTransport(cfg)
    monkeypatch.setattr(_runmod, "HttpClient", factory)
    monkeypatch.setattr(_batchmod, "HttpClient", factory)
    bd = ph.web.Batch({"a": _make_sim(), "b": _make_sim()}).run(max_usd=1)
    assert isinstance(bd["a"], ph.SimulationData)
    assert isinstance(bd.errors["b"], ph.web.WebError)
    assert bd.errors["b"].job_id == "b"


def test_cloud_batch_matches_local_construction_validation():
    with pytest.raises(ValueError, match="at least one"):
        ph.web.Batch({})
    with pytest.raises(TypeError, match="expected a Simulation"):
        ph.web.Batch({"not-a-sim": object()})


def test_cloud_batch_requires_spend_limit_before_network(configured):
    with pytest.raises(ValueError, match="requires max_usd"):
        ph.web.Batch({"a": _make_sim()}).run()


def test_cloud_batch_binds_each_preflight_quote(monkeypatch, configured):
    next_quote = iter(["quote-a", "quote-b"])
    submissions = {}
    estimated_devices = []

    class _QuotedBatch(_NameRoutedHttp):
        def estimate(self, spec, *, device=None, solver=None):
            estimated_devices.append(device)
            return {"usd": 0.25, "quote_id": next(next_quote)}

        def submit_job(self, spec, *, name=None, quote_id=None, device=None,
                       **kw):
            submissions[name] = (quote_id, device)
            return {"job_id": name}

    factory = lambda cfg: _QuotedBatch(cfg)
    monkeypatch.setattr(_runmod, "HttpClient", factory)
    monkeypatch.setattr(_batchmod, "HttpClient", factory)
    result = ph.web.Batch({"a": _make_sim(), "b": _make_sim()}).run(
        device=" gpu:mi300x ", max_usd={"a": 0.30, "b": 0.40})
    assert set(result.succeeded) == {"a", "b"}
    assert estimated_devices == ["gpu:mi300x", "gpu:mi300x"]
    assert submissions == {
        "a": ("quote-a", "gpu:mi300x"),
        "b": ("quote-b", "gpu:mi300x"),
    }


@pytest.mark.parametrize(
    "estimate",
    [
        pytest.param({"usd": 0.25}, id="missing-quote"),
        pytest.param({"usd": 0.25, "quote_id": " "}, id="blank-quote"),
        pytest.param({"usd": 0.25, "quote_id": 42}, id="non-string-quote"),
        pytest.param(
            {"usd": float("nan"), "quote_id": "quote-secret-never-log"},
            id="nan-cost"),
        pytest.param(
            {"usd": -0.01, "quote_id": "quote-secret-never-log"},
            id="negative-cost"),
    ],
)
def test_cloud_batch_invalid_quote_fails_before_any_submit(
        monkeypatch, configured, estimate):
    submissions = []

    class _InvalidQuote(FakeHttp):
        def estimate(self, spec, *, device=None, solver=None):
            return estimate

        def submit_job(self, *args, **kwargs):
            submissions.append(kwargs)
            return {"job_id": "must-not-submit"}

    factory = lambda cfg: _InvalidQuote(cfg)
    monkeypatch.setattr(_runmod, "HttpClient", factory)
    monkeypatch.setattr(_batchmod, "HttpClient", factory)
    with pytest.raises(ph.web.WebError, match="estimate") as exc:
        ph.web.Batch({"a": _make_sim()}).run(max_usd=1)
    assert "quote-secret-never-log" not in str(exc.value)
    assert submissions == []


def test_cloud_batch_over_limit_fails_before_any_submit(monkeypatch,
                                                         configured):
    submissions = []

    class _OverLimit(FakeHttp):
        def estimate(self, spec, *, device=None, solver=None):
            return {"usd": 0.51, "quote_id": "quote-over"}

        def submit_job(self, *args, **kwargs):
            submissions.append(kwargs)
            return {"job_id": "must-not-submit"}

    factory = lambda cfg: _OverLimit(cfg)
    monkeypatch.setattr(_runmod, "HttpClient", factory)
    monkeypatch.setattr(_batchmod, "HttpClient", factory)
    with pytest.raises(ph.web.WebError, match="exceeds"):
        ph.web.Batch({"a": _make_sim(), "b": _make_sim()}).run(max_usd=0.50)
    assert submissions == []


def test_cloud_batch_limit_mapping_must_cover_exact_entries(configured):
    batch = ph.web.Batch({"a": _make_sim(), "b": _make_sim()})
    with pytest.raises(ValueError, match="exactly match"):
        batch.run(max_usd={"a": 1.0})


def test_cancelled_job_raises_solverrunerror(monkeypatch, configured):
    class _Cancelled(FakeHttp):
        def get_job(self, job_id, *, deadline=None):
            return {"state": "cancelled"}

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _Cancelled(cfg))
    job = ph.web.run_async(_make_sim())
    with pytest.raises(SolverRunError) as ei:
        job.result(timeout=5)
    assert "cancelled" in str(ei.value)


# --- bundle.extract_bundle path-traversal guard (used by the cloud cache) ---

from photonhub.bundle import (  # noqa: E402
    BundleError,
    COMPLETION_MARKER,
    extract_bundle,
    extract_bundle_file,
    pack_bundle,
)


def _tar_with_member(name: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        data = b"x"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _tar_with_files(files) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_safe_extract_rejects_parent_traversal(tmp_path):
    dest = tmp_path / "job"
    with pytest.raises(ValueError, match="unsafe path"):
        extract_bundle(_tar_with_member("../escape.txt"), dest)
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_rejects_prefix_sibling_traversal(tmp_path):
    dest = tmp_path / "job"
    outside = tmp_path / "job-escape" / "payload.txt"
    with pytest.raises(ValueError, match="unsafe path"):
        extract_bundle(_tar_with_member("../job-escape/payload.txt"), dest)
    assert not outside.exists()


def test_safe_extract_rejects_absolute_path(tmp_path):
    dest = tmp_path / "job"
    outside = tmp_path / "outside.txt"
    with pytest.raises(ValueError, match="unsafe path"):
        extract_bundle(_tar_with_member(f"/{outside.name}"), dest)


def test_safe_extract_accepts_normal_bundle(tmp_path):
    dest = tmp_path / "job"
    extract_bundle(_bundle_bytes(), dest)
    assert (dest / "manifest.json").is_file()
    assert (dest / "probe.bin").is_file()


def test_safe_extract_wraps_invalid_archive(tmp_path):
    with pytest.raises(BundleError, match="invalid result bundle"):
        extract_bundle(b"not a gzip tar", tmp_path / "job")


def test_safe_extract_enforces_compressed_expanded_and_member_limits(tmp_path):
    bundle = _tar_with_files([("a.bin", b"ab"), ("b.bin", b"cd")])
    with pytest.raises(BundleError, match="compressed-byte"):
        extract_bundle(
            bundle, tmp_path / "compressed",
            max_compressed_bytes=len(bundle) - 1)
    with pytest.raises(BundleError, match="expanded-byte"):
        extract_bundle(
            bundle, tmp_path / "expanded", max_expanded_bytes=3)
    with pytest.raises(BundleError, match="member limit"):
        extract_bundle(bundle, tmp_path / "members", max_members=1)


def test_safe_extract_counts_hidden_pax_metadata_toward_limits(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(
            fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo("payload.bin")
        info.size = 1
        info.pax_headers = {"comment": "x" * 100_000}
        tar.addfile(info, io.BytesIO(b"x"))
    # The visible file is exactly one byte. Without accounting for the PAX
    # extension that tarfile consumes internally, this incorrectly succeeded.
    with pytest.raises(BundleError, match="metadata|expanded-byte"):
        extract_bundle(
            buf.getvalue(), tmp_path / "pax", max_expanded_bytes=1)


def test_safe_extract_rejects_nul_from_pax_path(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(
            fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo("placeholder.bin")
        info.size = 1
        info.pax_headers = {"path": "bad\x00name"}
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(BundleError, match="unsafe path"):
        extract_bundle(buf.getvalue(), tmp_path / "nul")


def test_streaming_file_extractor_enforces_limits(tmp_path):
    archive = tmp_path / "result.tar.gz"
    archive.write_bytes(_tar_with_files([("payload.bin", b"abcd")]))
    with pytest.raises(BundleError, match="expanded-byte"):
        extract_bundle_file(
            archive, tmp_path / "out", max_expanded_bytes=3)


def test_bundle_cannot_supply_or_repack_completion_marker(tmp_path):
    with pytest.raises(BundleError, match="unsafe path"):
        extract_bundle(
            _tar_with_member(COMPLETION_MARKER), tmp_path / "reject")

    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    (source / COMPLETION_MARKER).write_text("forged", encoding="utf-8")
    packed = pack_bundle(source)
    assert isinstance(packed, bytes)
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as tar:
        assert tar.getnames() == ["manifest.json"]


def test_packer_does_not_emit_pax_metadata_for_every_ordinary_file(tmp_path):
    source = tmp_path / "many"
    source.mkdir()
    for index in range(2_050):
        (source / f"m{index:04d}.bin").touch()
    packed = pack_bundle(source)
    assert isinstance(packed, bytes)
    dest = tmp_path / "unpacked"
    extract_bundle(packed, dest, max_members=100_000)
    assert len(list(dest.iterdir())) == 2_050


def test_safe_extract_rejects_symlink_member(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("manifest.json")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside.json"
        tar.addfile(info)
    with pytest.raises(ValueError, match="unsafe path"):
        extract_bundle(buf.getvalue(), tmp_path / "job")


def test_safe_extract_rejects_duplicate_member(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for payload in (b"first", b"second"):
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe path"):
        extract_bundle(buf.getvalue(), tmp_path / "job")


def test_safe_extract_does_not_follow_existing_destination_symlink(tmp_path):
    dest = tmp_path / "job"
    dest.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"keep")
    (dest / "manifest.json").symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe existing"):
        extract_bundle(_tar_with_member("manifest.json"), dest)
    assert outside.read_bytes() == b"keep"


def test_safe_extract_rejects_symlink_destination_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "job"
    dest.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe result destination"):
        extract_bundle(_tar_with_member("manifest.json"), dest)
    assert not (outside / "manifest.json").exists()


def test_cache_job_dir_rejects_path_syntax(tmp_path):
    from photonhub.web import cache as _cache

    cfg = _config.WebConfig(
        url="http://localhost:0", api_key="ph_x", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="job_id"):
        _cache.job_dir(cfg, "../escape")


# --- config env-var precedence (configure / get_config) -------------------

from photonhub.web import config as _config  # noqa: E402


def test_configure_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("PHOTONHUB_API_KEY", "ph_env")
    monkeypatch.setenv("PHOTONHUB_URL", "https://env-host:9/")
    try:
        cfg = ph.web.configure(api_key="ph_explicit",
                               url="https://explicit:1/")
        assert cfg.api_key == "ph_explicit"     # explicit beats env
        assert cfg.url == "https://explicit:1"  # trailing slash stripped
    finally:
        ph.web.reset()


def test_configure_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("PHOTONHUB_API_KEY", "ph_env")
    monkeypatch.setenv("PHOTONHUB_URL", "https://env-host:9/")
    try:
        cfg = ph.web.configure()
        assert cfg.api_key == "ph_env"
        assert cfg.url == "https://env-host:9"
    finally:
        ph.web.reset()


def test_configure_requires_url_when_unset(monkeypatch):
    # A key without a URL must fail loudly at configure() time — never fall
    # back to localhost and later die with a raw connection-refused.
    monkeypatch.delenv("PHOTONHUB_URL", raising=False)
    monkeypatch.delenv("PHOTONHUB_URL", raising=False)
    with pytest.raises(ph.web.WebError, match=r"no service URL.*PHOTONHUB_URL"):
        ph.web.configure(api_key="ph_x")


def test_get_config_autoconfigure_requires_url(monkeypatch):
    # The exact half-configured state a new user hits: $PHOTONHUB_API_KEY set
    # (e.g. pasted from the operator) but no $PHOTONHUB_URL. The lazy
    # get_config() path must surface the same actionable message.
    monkeypatch.setenv("PHOTONHUB_API_KEY", "ph_env")
    monkeypatch.delenv("PHOTONHUB_URL", raising=False)
    monkeypatch.delenv("PHOTONHUB_URL", raising=False)
    ph.web.reset()
    with pytest.raises(ph.web.WebError, match=r"no service URL.*PHOTONHUB_URL"):
        _config.get_config()


def test_configure_missing_key_raises_weberror(monkeypatch):
    monkeypatch.delenv("PHOTONHUB_API_KEY", raising=False)
    ph.web.reset()
    with pytest.raises(ph.web.WebError, match="no API key"):
        ph.web.configure()


def test_configure_explicit_empty_key_does_not_fall_back_to_env(monkeypatch):
    monkeypatch.setenv("PHOTONHUB_API_KEY", "ph_env")
    with pytest.raises(ph.web.WebError, match=r"no API key.*PHOTONHUB_API_KEY"):
        ph.web.configure(api_key="")


def test_get_config_unconfigured_names_both_env_vars():
    # The message a user with neither variable set sees; it must name both.
    ph.web.reset()
    with pytest.raises(ph.web.WebError,
                       match=r"not configured.*\$PHOTONHUB_API_KEY \+ \$PHOTONHUB_URL"):
        _config.get_config()


@pytest.mark.parametrize(
    "bad", ["api.example.test", "http://api.example.test", "https://u:p@host"])
def test_configure_rejects_malformed_env_url_as_weberror(monkeypatch, bad):
    # An env-sourced URL is user configuration, not a programming error, so it
    # must surface as the WebError the notebook preview and docs promise.
    monkeypatch.setenv("PHOTONHUB_API_KEY", "ph_env")
    monkeypatch.setenv("PHOTONHUB_URL", bad)
    with pytest.raises(ph.web.WebError, match=r"PHOTONHUB_URL is not a usable"):
        ph.web.configure()


def test_config_repr_omits_api_key():
    # A bare get_config() in a notebook cell must not write the live key into
    # committed output.
    cfg = ph.web.configure(api_key="ph_live_SUPERSECRET",
                           url="https://x.example")
    try:
        assert "SUPERSECRET" not in repr(cfg)
    finally:
        ph.web.reset()


@pytest.mark.parametrize(
    "url",
    ["", "localhost:8000", "ftp://example.invalid", "https://u:p@host/api",
     "https://host:bad/api", "https://host:99999/api",
     "https://host/api?q=1", "https://host/api#fragment", "https://host/ bad"],
)
def test_configure_rejects_unsafe_or_ambiguous_url(url):
    with pytest.raises(ValueError, match="url"):
        ph.web.configure(api_key="ph_x", url=url)


def test_configure_requires_https_away_from_loopback():
    with pytest.raises(ValueError, match="require HTTPS"):
        ph.web.configure(api_key="ph_x", url="http://api.example.test")
    cfg = ph.web.configure(
        api_key="ph_x", url="http://devbox.internal:8000",
        allow_insecure_http=True)
    assert cfg.allow_insecure_http is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("poll_interval_s", -1), ("poll_interval_s", float("nan")),
     ("poll_backoff_max_s", float("inf")), ("request_timeout_s", 0),
     ("request_timeout_s", True)],
)
def test_configure_rejects_invalid_timing(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        ph.web.configure(api_key="ph_x", url="http://localhost:8000", **kwargs)


def test_configure_rejects_backoff_cap_below_initial_interval():
    with pytest.raises(ValueError, match="poll_backoff_max_s"):
        ph.web.configure(
            api_key="ph_x", url="http://localhost:8000",
            poll_interval_s=2, poll_backoff_max_s=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_bundle_download_bytes", 0),
     ("max_bundle_extract_bytes", -1),
     ("max_bundle_members", True),
     ("max_bundle_members", 1.5)],
)
def test_configure_rejects_invalid_bundle_resource_limits(field, value):
    with pytest.raises(ValueError, match=field):
        ph.web.configure(
            api_key="ph_x", url="http://localhost:8000", **{field: value})


# --- client error -> WebError mapping -------------------------------------

from photonhub.web.client import HttpClient, _parse_detail  # noqa: E402
import http.client  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


def _cfg():
    return _config.WebConfig(url="http://localhost:0", api_key="ph_x",
                             cache_dir=Path("/tmp"), request_timeout_s=0.01)


@pytest.mark.parametrize(
    "payload",
    [{}, {"jobs": {}}, {"jobs": ["not-an-object"]}],
)
def test_client_list_jobs_rejects_invalid_history_shape(monkeypatch, payload):
    client = HttpClient(_cfg())
    monkeypatch.setattr(client, "get_json", lambda path: payload)
    with pytest.raises(ph.web.WebError, match="invalid 'jobs' list"):
        client.list_jobs()


def test_client_list_jobs_uses_read_only_history_route(monkeypatch):
    client = HttpClient(_cfg())
    seen = []
    records = [{"job_id": "job-1", "state": "succeeded"}]
    monkeypatch.setattr(
        client, "get_json", lambda path: seen.append(path) or {"jobs": records})
    assert client.list_jobs() == records
    assert seen == ["/v1/jobs"]


def test_client_4xx_maps_to_weberror_with_detail(monkeypatch):
    body = io.BytesIO(json.dumps({"detail": "bad spec"}).encode())
    err = urllib.error.HTTPError("http://x/v1/jobs", 400, "Bad Request", {}, body)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(err))
    with pytest.raises(ph.web.WebError) as ei:
        HttpClient(_cfg()).get_json("/v1/jobs")
    assert ei.value.status_code == 400
    assert ei.value.body == "bad spec"          # parsed from JSON "detail"
    assert body.closed


def test_client_caps_http_error_body(monkeypatch):
    client_module = importlib.import_module("photonhub.web.client")
    monkeypatch.setattr(client_module, "_MAX_ERROR_BODY_BYTES", 4)
    err = urllib.error.HTTPError(
        "http://x/v1/jobs", 400, "Bad Request", {}, io.BytesIO(b"abcdef"))
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(err))
    with pytest.raises(ph.web.WebError) as exc:
        HttpClient(_cfg()).get_json("/v1/jobs")
    assert exc.value.body == "response error body exceeded the 1 MiB limit"


def test_client_closes_every_retried_http_error_response(monkeypatch):
    client_module = importlib.import_module("photonhub.web.client")
    bodies = []

    def _fail(req, timeout=None):
        body = io.BytesIO(b'{"detail": "temporary"}')
        bodies.append(body)
        raise urllib.error.HTTPError(
            "http://x/v1/jobs", 503, "Unavailable", {}, body)

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    monkeypatch.setattr(client_module.time, "sleep", lambda delay: None)
    with pytest.raises(ph.web.WebError) as exc:
        HttpClient(_cfg()).get_json("/v1/jobs")
    assert exc.value.status_code == 503
    assert len(bodies) == 3
    assert all(body.closed for body in bodies)


def test_client_caps_json_response_body(monkeypatch):
    client_module = importlib.import_module("photonhub.web.client")
    monkeypatch.setattr(client_module, "_MAX_JSON_BODY_BYTES", 4)
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _DownloadResponse(b'{"x": 1}'))
    with pytest.raises(ph.web.WebError, match="response limit"):
        HttpClient(_cfg()).get_json("/v1/account")


def test_https_redirect_handler_rejects_plaintext_downgrade():
    client_module = importlib.import_module("photonhub.web.client")
    handler = client_module._NoHttpsDowngrade()
    request = urllib.request.Request("https://api.example.test/result")
    with pytest.raises(urllib.error.HTTPError, match="unsafe redirect"):
        handler.redirect_request(
            request, io.BytesIO(), 302, "Found", {},
            "http://objects.example.test/signed")


def test_client_network_error_maps_to_weberror(monkeypatch):
    err = urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(err))
    with pytest.raises(ph.web.WebError, match="failed"):
        HttpClient(_cfg()).get_json("/v1/account")


def test_client_marks_bearer_header_unredirected(monkeypatch):
    seen = {}

    def _open(req, timeout=None):
        seen["headers"] = {k.lower(): v for k, v in req.headers.items()}
        seen["unredirected"] = {
            k.lower(): v for k, v in req.unredirected_hdrs.items()
        }
        return io.BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _open)
    assert HttpClient(_cfg()).get_json("/v1/jobs/x/result") == {}
    assert "authorization" not in seen["headers"]
    assert seen["unredirected"]["authorization"] == "Bearer ph_x"


class _DownloadResponse(io.BytesIO):
    def __init__(self, data: bytes, content_length=None):
        super().__init__(data)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)


def test_client_stream_download_enforces_actual_byte_limit(monkeypatch,
                                                           tmp_path):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _DownloadResponse(b"abcdef"))
    dest = tmp_path / "result.tar.gz"
    with pytest.raises(ph.web.WebError, match="download limit"):
        HttpClient(_cfg()).get_to_file("/v1/result", dest, max_bytes=5)
    assert not dest.exists()


def test_client_stream_download_rejects_truncated_content_length(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _DownloadResponse(b"abc", content_length=5))
    dest = tmp_path / "result.tar.gz"
    with pytest.raises(ph.web.WebError, match="Content-Length"):
        HttpClient(_cfg()).get_to_file("/v1/result", dest, max_bytes=10)
    assert not dest.exists()


def test_client_stream_download_maps_incomplete_read_to_weberror(
        monkeypatch, tmp_path):
    class _Incomplete(_DownloadResponse):
        def read(self, size=-1):
            raise http.client.IncompleteRead(b"ab", 5)

    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _Incomplete(b""))
    dest = tmp_path / "result.tar.gz"
    with pytest.raises(ph.web.WebError, match="failed while reading"):
        HttpClient(_cfg()).get_to_file("/v1/result", dest, max_bytes=10)
    assert not dest.exists()


@pytest.mark.parametrize("method", ["POST", "DELETE"])
def test_client_does_not_retry_mutating_requests(monkeypatch, method):
    calls = []

    def _fail(req, timeout=None):
        calls.append(req.get_method())
        raise urllib.error.URLError("ambiguous network failure")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    client = HttpClient(_cfg())
    with pytest.raises(ph.web.WebError, match="ambiguous network failure"):
        if method == "POST":
            client.post_json("/v1/jobs", {"spec": {}})
        else:
            client.delete("/v1/jobs/x")
    assert calls == [method]


def test_client_clips_request_timeout_to_poll_deadline(monkeypatch):
    seen = []

    def _open(req, timeout=None):
        seen.append(timeout)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _open)
    deadline = _runmod.time.monotonic() + 0.05
    assert HttpClient(_cfg()).get_json("/v1/jobs/x", deadline=deadline) == {}
    assert len(seen) == 1
    assert 0 < seen[0] <= 0.05


def test_client_retry_backoff_cannot_overrun_deadline(monkeypatch):
    client_module = importlib.import_module("photonhub.web.client")
    clock = {"now": 100.0}
    sleeps = []
    calls = []

    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock["now"])

    def _sleep(delay):
        sleeps.append(delay)
        clock["now"] += delay

    def _fail(req, timeout=None):
        calls.append(timeout)
        raise urllib.error.URLError("retry me")

    monkeypatch.setattr(client_module.time, "sleep", _sleep)
    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    with pytest.raises(TimeoutError, match="deadline"):
        HttpClient(_cfg()).get_json("/v1/jobs/x", deadline=100.2)
    assert len(calls) == 1
    assert calls[0] == pytest.approx(0.01)
    assert sleeps == [pytest.approx(0.2)]


def test_parse_detail_variants():
    assert _parse_detail(json.dumps({"detail": "boom"})) == "boom"
    assert _parse_detail(json.dumps({"x": 1})) == {"x": 1}  # no detail -> obj
    assert _parse_detail("not json") == "not json"          # falls back to text


def test_https_context_falls_back_to_certifi_only_when_rootless(monkeypatch):
    """A default context that already trusts CAs is used untouched; a root-less
    one (python.org macOS build without its certificate step) is topped up
    from certifi's bundle. Verification itself is never relaxed."""
    import ssl as _ssl

    from photonhub.web import client as _client

    class _Ctx:
        def __init__(self, n_ca):
            self.n_ca = n_ca
            self.loaded = []

        def cert_store_stats(self):
            return {"x509": self.n_ca, "crl": 0, "x509_ca": self.n_ca}

        def load_verify_locations(self, cafile=None):
            self.loaded.append(cafile)

    healthy = _Ctx(140)
    monkeypatch.setattr(_ssl, "create_default_context", lambda: healthy)
    assert _client._https_context() is healthy
    assert healthy.loaded == []                      # untouched

    rootless = _Ctx(0)
    monkeypatch.setattr(_ssl, "create_default_context", lambda: rootless)
    ctx = _client._https_context()
    assert ctx is rootless
    import certifi
    assert rootless.loaded == [certifi.where()]      # topped up, not disabled
