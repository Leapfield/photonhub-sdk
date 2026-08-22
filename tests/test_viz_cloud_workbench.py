"""Cloud-GPU Workbench contracts with a fully mocked metered service.

These tests must never inherit or exercise developer cloud credentials.  Every
cloud operation reached by an endpoint is replaced before the TestClient is
created; submit/cancel/resume are in-process fakes only.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from photonhub import web
from photonhub.viz import service


def _app_client(run_root: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    return TestClient(create_app(run_root=run_root))


def _spec() -> dict:
    return service.default_sim().to_wire_dict()


class _FailedDownload:
    job_id = "job-exact-five"

    def result(self):
        raise web.WebError("mock download interrupted", job_id=self.job_id)


def _accepted(*, quote_usd: float = 5.0, available_usd: float = 5.0):
    return web.CloudPreflight(
        device="gpu:mi300x",
        solver="abcdef1234567890",
        max_usd=5.0,
        quote_usd=quote_usd,
        available_usd=available_usd,
        remaining_usd=available_usd - quote_usd,
        quote_id="opaque-accepted-quote",
        quote={
            "quote_id": "opaque-accepted-quote",
            "usd": quote_usd,
            "num_cells": 64,
            "num_steps": 5,
            "expires_at": "2999-07-21T20:23:04Z",
        },
    )


@pytest.fixture(autouse=True)
def _pinned_source_cloud_solver(monkeypatch):
    monkeypatch.setenv("PHOTONHUB_CLOUD_SOLVER", "abcdef1234567890")


def test_cloud_status_is_actionable_without_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(
        web, "get_config",
        lambda: (_ for _ in ()).throw(web.WebError("set PHOTONHUB_API_KEY + PHOTONHUB_URL")),
    )
    monkeypatch.setattr(
        web, "account", lambda: pytest.fail("unconfigured status must not call account"))

    with _app_client(tmp_path / "runs") as client:
        response = client.get("/api/cloud/status")

    assert response.status_code == 200
    assert response.json() == {
        "configured": False,
        "reachable": False,
        "account": None,
        "gpus": [],
        "gpu_menu_available": False,
        "warning": None,
        "error": "set PHOTONHUB_API_KEY + PHOTONHUB_URL",
        "max_usd": 5.0,
    }


def test_cloud_status_tolerates_missing_gpu_menu(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "get_config", lambda: object())
    monkeypatch.setattr(web, "account", lambda: {
        "balance_micros": 5_000_000,
        "available_micros": 4_000_000,
        "reserved_micros": 1_000_000,
    })
    monkeypatch.setattr(
        web, "gpus",
        lambda: (_ for _ in ()).throw(web.WebError("not found", status_code=404)),
    )

    with _app_client(tmp_path / "runs") as client:
        payload = client.get("/api/cloud/status").json()

    assert payload["reachable"] is True
    assert payload["gpu_menu_available"] is False
    assert payload["gpus"] == []
    assert payload["account"] == {
        "balance_usd": 5.0, "available_usd": 4.0, "reserved_usd": 1.0,
    }
    assert "no approved /v1/gpus menu" in payload["warning"]


@pytest.mark.parametrize("device", [None, "", "gpu"])
def test_workbench_preflight_rejects_non_curated_gpu_device(
        monkeypatch, tmp_path, device):
    monkeypatch.setattr(
        web, "preflight",
        lambda *_args, **_kwargs: pytest.fail(
            "an unapproved device must not reach the metered service"),
    )
    payload = {"spec": _spec(), "max_usd": 5}
    if device is not None:
        payload["device"] = device

    with _app_client(tmp_path / "runs") as client:
        response = client.post("/api/cloud/preflight", json=payload)

    assert response.status_code == 400
    assert "explicitly approved gpu:<id>" in response.json()["detail"]


def test_exact_five_dollar_quote_is_bound_once_and_never_serialized(
        monkeypatch, tmp_path):
    calls = {"preflight": 0, "submit": 0}

    def fake_preflight(_sim, *, device, solver, max_usd):
        calls["preflight"] += 1
        assert device == "gpu:mi300x"
        assert solver == "abcdef1234567890"
        assert max_usd == 5.0
        return _accepted()

    def fake_run_async(_sim, *, device, solver, quote_id, **_kwargs):
        calls["submit"] += 1
        assert device == "gpu:mi300x"
        assert solver == "abcdef1234567890"
        assert quote_id == "opaque-accepted-quote"
        return _FailedDownload()

    monkeypatch.setattr(web, "preflight", fake_preflight)
    monkeypatch.setattr(web, "run_async", fake_run_async)
    monkeypatch.setattr(web, "job_status", lambda _job_id: {
        "job_id": "job-exact-five", "state": "succeeded",
        "actual_micros": 4_500_000, "refunded_micros": 500_000,
    })

    run_root = tmp_path / "runs"
    with _app_client(run_root) as client:
        quote_response = client.post("/api/cloud/preflight", json={
            "spec": _spec(), "device": "gpu:mi300x", "max_usd": 5,
        })
        assert quote_response.status_code == 200
        quote = quote_response.json()
        assert quote["quote_usd"] == 5.0
        assert quote["available_usd"] == 5.0
        assert quote["remaining_usd"] == 0.0
        assert "quote_id" not in quote
        token = quote["token"]

        invalid_name = client.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": token, "name": "",
        })
        assert invalid_name.status_code == 400

        submitted = client.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": token,
        })
        assert submitted.status_code == 200
        assert submitted.json()["id"] == "job-exact-five"

        # A browser retry or double click cannot resubmit the accepted quote.
        duplicate = client.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": token,
        })
        assert duplicate.status_code == 409

        deadline = time.monotonic() + 2
        current = None
        while time.monotonic() < deadline:
            current = client.get("/api/cloud/run/status").json()
            if current["download_status"] == "failed":
                break
            time.sleep(0.01)
        assert current is not None
        assert current["id"] == "job-exact-five"
        assert current["status"] == "succeeded"
        assert current["actual_usd"] == 4.5
        assert current["refunded_usd"] == 0.5
        assert current["resumable"] is True

        # A failed download is still an already-paid, resumable job. A fresh
        # quote must not create a second charge for the same Workbench setup.
        fresh = client.post("/api/cloud/preflight", json={
            "spec": _spec(), "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        blocked = client.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": fresh["token"],
        })
        assert blocked.status_code == 409
        assert "resume" in blocked.json()["detail"]

    assert calls == {"preflight": 2, "submit": 1}
    recovery = (run_root / "cloud-job-recovery.json").read_text(encoding="utf-8")
    assert "opaque-accepted-quote" not in recovery
    assert token not in recovery


def test_insufficient_available_balance_and_over_five_cap_submit_nothing(
        monkeypatch, tmp_path):
    calls = {"preflight": 0, "submit": 0}

    def insufficient(*_args, **_kwargs):
        calls["preflight"] += 1
        raise web.WebError(
            "server quote $5.000000 exceeds available balance $4.990000; "
            "no job was submitted")

    monkeypatch.setattr(web, "preflight", insufficient)
    monkeypatch.setattr(
        web, "run_async",
        lambda *_args, **_kwargs: calls.__setitem__("submit", calls["submit"] + 1),
    )

    with _app_client(tmp_path / "runs") as client:
        insufficient_response = client.post("/api/cloud/preflight", json={
            "spec": _spec(), "device": "gpu:mi300x", "max_usd": 5,
        })
        assert insufficient_response.status_code == 409
        assert "no job was submitted" in insufficient_response.json()["detail"]

        too_high = client.post("/api/cloud/preflight", json={
            "spec": _spec(), "device": "gpu:mi300x", "max_usd": 5.01,
        })
        assert too_high.status_code == 400

    assert calls == {"preflight": 1, "submit": 0}


def test_ambiguous_submit_retains_job_id_and_forbids_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "preflight", lambda *_args, **_kwargs: _accepted(quote_usd=1))

    def ambiguous(*_args, **_kwargs):
        raise web.WebError("connection lost after accept", job_id="job-ambiguous")

    monkeypatch.setattr(web, "run_async", ambiguous)
    monkeypatch.setattr(web, "job_status", lambda _job_id: {
        "job_id": "job-ambiguous", "state": "provisioning",
    })

    with _app_client(tmp_path / "runs") as client:
        quote = client.post("/api/cloud/preflight", json={
            "spec": _spec(), "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        submitted = client.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": quote["token"],
        })
        assert submitted.status_code == 424
        assert "job-ambiguous" in submitted.json()["detail"]

        recovered = client.get("/api/cloud/run/status").json()
        assert recovered["id"] == "job-ambiguous"
        assert recovered["status"] == "provisioning"
        duplicate = client.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": quote["token"],
        })
        assert duplicate.status_code == 409


def test_no_id_submit_is_persisted_before_post_and_requires_exact_name_reattach(
        monkeypatch, tmp_path):
    run_root = tmp_path / "runs"
    attempts = []
    monkeypatch.setattr(web, "preflight", lambda *_args, **_kwargs: _accepted(quote_usd=1))

    def ambiguous(*_args, **_kwargs):
        record = json.loads(
            (run_root / "cloud-job-recovery.json").read_text(encoding="utf-8"))
        assert record["id"] is None
        assert record["unresolved_submission"] is True
        assert record["name"].startswith("workbench-")
        attempts.append(record["name"])
        raise web.WebError("connection lost during paid submit")

    monkeypatch.setattr(web, "run_async", ambiguous)

    with _app_client(run_root) as first:
        quote = first.post("/api/cloud/preflight", json={
            "spec": _spec(), "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        response = first.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": quote["token"],
        })
        assert response.status_code == 424
        assert "new submissions are blocked" in response.json()["detail"]
        current = first.get("/api/cloud/run/status").json()
        assert current["id"] is None
        assert current["unresolved_submission"] is True
        assert current["name"] == attempts[0]

    # A fresh sidecar restores the name-only duplicate-charge guard.
    with _app_client(run_root) as restarted:
        restored = restarted.get("/api/cloud/run/status").json()
        assert restored["status"] == "unknown"
        assert restored["unresolved_submission"] is True
        assert restored["name"] == attempts[0]

        new_quote = restarted.post("/api/cloud/preflight", json={
            "spec": _spec(), "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        blocked = restarted.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": new_quote["token"],
        })
        assert blocked.status_code == 409
        assert len(attempts) == 1

        # There is deliberately no client-side "nothing was charged" escape:
        # one eventually-consistent history response cannot prove absence.
        assert restarted.post(
            "/api/cloud/run/clear-ambiguous", json={"confirmed_no_job": True}
        ).status_code == 404

        released = threading.Event()

        class FoundJob:
            job_id = "job-found-by-name"

            def result(self):
                released.wait(2)
                raise web.WebError("mock remote job still running", job_id=self.job_id)

        monkeypatch.setattr(web, "job_status", lambda job_id: {
            "job_id": job_id, "name": attempts[0], "state": "running",
        })
        monkeypatch.setattr(web, "list_jobs", lambda: [{
            "job_id": "job-found-by-name",
            "name": attempts[0],
            "state": "running",
        }])
        monkeypatch.setattr(web, "resume", lambda job_id: (
            FoundJob() if job_id == "job-found-by-name" else pytest.fail(job_id)))

        wrong = restarted.post("/api/cloud/run/resume", json={
            "job_id": "job-found-by-name", "name": "some-other-job",
        })
        assert wrong.status_code == 409
        attached = restarted.post("/api/cloud/run/resume", json={
            "job_id": "job-found-by-name", "name": attempts[0],
        })
        assert attached.status_code == 200
        assert attached.json()["id"] == "job-found-by-name"
        assert attached.json()["unresolved_submission"] is False
        # The canonical simulation survived the no-id recovery and is retained
        # for checksum-sealing when this exact job eventually finishes.
        recovery = json.loads(
            (run_root / "cloud-job-recovery.json").read_text(encoding="utf-8"))
        assert recovery["id"] == "job-found-by-name"
        assert recovery["spec"] == service.default_sim().to_wire_dict()
        assert len(attempts) == 1
        released.set()


@pytest.mark.parametrize(
    ("case", "expected_detail"),
    [
        ("zero", "found no exact job"),
        ("multiple", "found 2 exact jobs"),
        ("selected_id_changed", "selected job id is no longer"),
        ("status_race", "service job changed while attachment was being verified"),
    ],
)
def test_ambiguous_attach_requeries_history_and_fails_closed(
        monkeypatch, tmp_path, case, expected_detail):
    monkeypatch.setattr(
        web, "preflight", lambda *_args, **_kwargs: _accepted(quote_usd=1))
    monkeypatch.setattr(
        web, "run_async",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            web.WebError("connection lost during paid submit")),
    )

    with _app_client(tmp_path / "runs") as client:
        quote = client.post("/api/cloud/preflight", json={
            "spec": _spec(), "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        assert client.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": quote["token"],
        }).status_code == 424
        ambiguous_name = client.get("/api/cloud/run/status").json()["name"]
        selected_id = "job-selected"

        if case == "zero":
            history = []
        elif case == "multiple":
            history = [
                {"job_id": selected_id, "name": ambiguous_name},
                {"job_id": "job-duplicate", "name": ambiguous_name},
            ]
        elif case == "selected_id_changed":
            history = [{"job_id": "job-replaced", "name": ambiguous_name}]
        else:
            history = [{"job_id": selected_id, "name": ambiguous_name}]
        monkeypatch.setattr(web, "list_jobs", lambda: history)

        status_calls = []

        def job_status(job_id):
            status_calls.append(job_id)
            remote_id = "job-replaced-after-list" if case == "status_race" else job_id
            return {
                "job_id": remote_id,
                "name": ambiguous_name,
                "state": "running",
            }

        monkeypatch.setattr(web, "job_status", job_status)
        monkeypatch.setattr(
            web, "resume",
            lambda _job_id: pytest.fail("a non-unique attach must not resume"),
        )

        response = client.post("/api/cloud/run/resume", json={
            "job_id": selected_id, "name": ambiguous_name,
        })

    assert response.status_code == 409
    assert expected_detail in response.json()["detail"]
    assert status_calls == ([selected_id] if case == "status_race" else [])


def test_cloud_cancel_polls_actual_and_refunded_cost(monkeypatch, tmp_path):
    released = threading.Event()

    class ActiveJob:
        job_id = "job-cancelled"

        def result(self):
            released.wait(2)
            raise RuntimeError("mock job cancelled")

    status = {"state": "running"}

    monkeypatch.setattr(web, "preflight", lambda *_args, **_kwargs: _accepted(quote_usd=1))
    monkeypatch.setattr(web, "run_async", lambda *_args, **_kwargs: ActiveJob())
    monkeypatch.setattr(web, "job_status", lambda _job_id: {
        "job_id": "job-cancelled", **status,
        "actual_micros": 250_000, "refunded_micros": 750_000,
    })

    def cancel(job_id):
        assert job_id == "job-cancelled"
        status["state"] = "cancelled"
        released.set()
        return {
            "job_id": job_id, "state": "cancelled",
            "actual_micros": 250_000, "refunded_micros": 750_000,
        }

    monkeypatch.setattr(web, "cancel", cancel)

    with _app_client(tmp_path / "runs") as client:
        quote = client.post("/api/cloud/preflight", json={
            "spec": _spec(), "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        client.post("/api/cloud/run", json={
            "spec": _spec(), "preflight_token": quote["token"],
        })
        cancelled = client.post("/api/cloud/run/cancel", json={})

        assert cancelled.status_code == 200
        assert cancelled.json()["id"] == "job-cancelled"
        assert cancelled.json()["status"] == "cancelled"
        polled = client.get("/api/cloud/run/status").json()
        assert polled["actual_usd"] == 0.25
        assert polled["refunded_usd"] == 0.75


def test_cloud_result_opens_and_last_job_can_resume_after_restart(
        monkeypatch, tmp_path):
    run_root = tmp_path / "runs"
    cloud_spec = _spec()
    cloud_spec["monitors"] = []
    cloud_sim, _ = service.parse_sim_spec(cloud_spec)
    # photonhub-cloud stores and executes this deterministic representation;
    # Workbench must bind the archive to the same bytes, not its pretty save
    # representation.
    canonical = cloud_sim.to_wire_json(indent=0)
    cache_dir = tmp_path / "cloud-cache" / "job-resume"
    cache_dir.mkdir(parents=True)
    cache_dir.joinpath("manifest.json").write_text(json.dumps({
        "manifest_version": "1", "schema_version": "1.0.0",
        "run": {"n_steps": 5, "steps_run": 5, "dt_s": 1e-16,
                "aborted": False, "abort_reason": ""},
        "grid": {"shape": [2, 2, 2], "dl_um": 0.1},
        "provenance": {
            "input_sha256": __import__("hashlib").sha256(
                canonical.encode("utf-8")).hexdigest(),
        },
        "monitors": [],
    }) + "\n", encoding="utf-8")
    fake_data = service.load_result(cache_dir)

    class DownloadedJob:
        job_id = "job-resume"

        def result(self):
            return fake_data

    monkeypatch.setattr(web, "preflight", lambda *_args, **_kwargs: _accepted(quote_usd=1))
    monkeypatch.setattr(web, "run_async", lambda *_args, **_kwargs: _FailedDownload())
    monkeypatch.setattr(web, "job_status", lambda job_id: {
        "job_id": job_id, "state": "succeeded", "actual_micros": 1_000_000,
    })

    with _app_client(run_root) as first:
        quote = first.post("/api/cloud/preflight", json={
            "spec": cloud_spec, "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        first.post("/api/cloud/run", json={
            "spec": cloud_spec, "preflight_token": quote["token"],
        })

    # Simulate a fresh sidecar: the accepted service id survives, but the
    # downloaded result/session is intentionally process-local until resumed.
    monkeypatch.setattr(web, "resume", lambda job_id: (
        DownloadedJob() if job_id == "job-exact-five" else pytest.fail(job_id)))

    original_session = service.session
    activation_calls = 0

    def fail_once_after_seal(data, result_id, run_id=None):
        nonlocal activation_calls
        activation_calls += 1
        recovery_path = run_root / "cloud-job-recovery.json"
        if activation_calls <= 2:
            # Both cloud-archive activations run before the result session is
            # bound, so the durable job->run mapping must still be on disk.
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            assert recovery["ledger_run_id"] == run_id
        else:
            # Later /api/session rebuilds happen after the opened result
            # resolved the recovery record.
            assert not recovery_path.exists()
        if activation_calls == 1:
            raise RuntimeError("injected crash window after cloud seal")
        return original_session(data, result_id, run_id=run_id)

    monkeypatch.setattr(service, "session", fail_once_after_seal)

    with _app_client(run_root) as restarted:
        recovered = restarted.get("/api/cloud/run/status").json()
        assert recovered["id"] == "job-exact-five"
        assert recovered["resumable"] is True

        resumed = restarted.post("/api/cloud/run/resume", json={
            "job_id": "job-exact-five",
        })
        assert resumed.status_code == 200
        deadline = time.monotonic() + 2
        interrupted = None
        while time.monotonic() < deadline:
            interrupted = restarted.get("/api/cloud/run/status").json()
            if interrupted.get("download_status") == "failed":
                break
            time.sleep(0.01)
        assert interrupted is not None
        assert interrupted["resumable"] is True
        sealed_id = interrupted["ledger_run_id"]
        completed_before_retry = [
            item for item in restarted.get("/api/runs").json()["runs"]
            if item["recorded_status"] == "completed"
        ]
        assert [item["run_id"] for item in completed_before_retry] == [sealed_id]

        # Resume reuses the already-sealed mapping instead of creating an
        # orphan second archive after the injected post-seal failure.
        assert restarted.post("/api/cloud/run/resume", json={
            "job_id": "job-exact-five",
        }).status_code == 200
        deadline = time.monotonic() + 2
        completed = None
        while time.monotonic() < deadline:
            completed = restarted.get("/api/cloud/run/status").json()
            if completed.get("session"):
                break
            time.sleep(0.01)
        assert completed is not None
        assert completed["status"] == "succeeded"
        assert completed["download_status"] == "completed"
        archive = Path(completed["session"]["output_dir"])
        assert archive.parent == (run_root / "bundles").resolve()
        assert archive != cache_dir.resolve()
        assert completed["ledger_run_id"] == completed["session"]["run_id"]
        assert restarted.get("/api/session").json()["output_dir"] == str(archive)
        assert archive.joinpath("sim.json").read_text(encoding="utf-8") == canonical
        assert completed["session"]["geometry"]["status"] == "matched"

        history = restarted.get("/api/runs").json()["runs"]
        assert len([item for item in history
                    if item["recorded_status"] == "completed"]) == 1
        sealed = next(item for item in history
                      if item["run_id"] == completed["ledger_run_id"])
        assert sealed["recorded_status"] == "completed"
        assert sealed["device"] == "cloud:gpu:mi300x"
        assert {item["path"] for item in sealed["integrity"]["artifacts"]} == {
            "manifest.json", "sim.json", "solver-events.jsonl",
        }

        # Cloud plots now cross the same local SHA-256 boundary as CPU plots.
        manifest_path = archive / "manifest.json"
        raw = manifest_path.read_bytes()
        manifest_path.write_bytes(raw.replace(b'"n_steps": 5', b'"n_steps": 6'))
        tampered = restarted.get("/api/session")
        assert tampered.status_code == 409
        assert "integrity" in tampered.json()["detail"]


def test_opened_cloud_result_restores_idle_after_restart(monkeypatch, tmp_path):
    run_root = tmp_path / "runs"
    cloud_spec = _spec()
    cloud_spec["monitors"] = []
    cloud_sim, _ = service.parse_sim_spec(cloud_spec)
    canonical = cloud_sim.to_wire_json(indent=0)
    cache_dir = tmp_path / "cloud-cache" / "job-settled"
    cache_dir.mkdir(parents=True)
    cache_dir.joinpath("manifest.json").write_text(json.dumps({
        "manifest_version": "1", "schema_version": "1.0.0",
        "run": {"n_steps": 5, "steps_run": 5, "dt_s": 1e-16,
                "aborted": False, "abort_reason": ""},
        "grid": {"shape": [2, 2, 2], "dl_um": 0.1},
        "provenance": {
            "input_sha256": __import__("hashlib").sha256(
                canonical.encode("utf-8")).hexdigest(),
        },
        "monitors": [],
    }) + "\n", encoding="utf-8")
    fake_data = service.load_result(cache_dir)

    class DownloadedJob:
        job_id = "job-settled"

        def result(self):
            return fake_data

    monkeypatch.setattr(web, "preflight", lambda *_args, **_kwargs: _accepted(quote_usd=1))
    monkeypatch.setattr(web, "run_async", lambda *_args, **_kwargs: DownloadedJob())
    monkeypatch.setattr(web, "job_status", lambda job_id: {
        "job_id": job_id, "state": "succeeded", "actual_micros": 1_000_000,
    })

    with _app_client(run_root) as first:
        quote = first.post("/api/cloud/preflight", json={
            "spec": cloud_spec, "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        assert first.post("/api/cloud/run", json={
            "spec": cloud_spec, "preflight_token": quote["token"],
        }).status_code == 200
        deadline = time.monotonic() + 2
        completed = None
        while time.monotonic() < deadline:
            completed = first.get("/api/cloud/run/status").json()
            if completed.get("session"):
                break
            time.sleep(0.01)
        assert completed is not None
        assert completed["status"] == "succeeded"
        assert completed["download_status"] == "completed"
        assert completed["error"] is None

    # The bound result session is the recovery's goal state: once it exists
    # the sealed ledger run owns the evidence and no record may survive to
    # warn "resume the existing cloud job" on every subsequent launch.
    assert not (run_root / "cloud-job-recovery.json").exists()

    with _app_client(run_root) as restarted:
        assert restarted.get("/api/cloud/run/status").json() == {
            "status": "idle", "download_status": "idle",
        }
        history = restarted.get("/api/runs").json()["runs"]
        assert [item["run_id"] for item in history
                if item["recorded_status"] == "completed"] == [
            completed["ledger_run_id"]]


def _real_solver():
    import photonhub as ph
    try:
        return ph.find_solver()
    except ph.SolverRunError:
        return None


@pytest.mark.skipif(_real_solver() is None,
                    reason="no phsolver binary found (build the engine first)")
def test_completed_cloud_run_reopens_from_history_in_fresh_offline_session(
        monkeypatch, tmp_path):
    """The whole beta journey: run on the cloud, quit the Workbench, reopen it
    with the service unreachable — the paid result must come back from the
    durable archive with its monitor DATA intact. Uses a real solver output
    shaped like the deployed coordinator's bundle (no sim.json member, so the
    archive step must reconstruct and digest-verify the executed spec)."""
    import subprocess

    import photonhub as ph

    run_root = tmp_path / "runs"
    freqs = [1.8e14, 1.934e14]
    cloud_sim = ph.Simulation(
        size_um=(1.2, 0.8, 0.8),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=20, shutoff=0.0),
        boundaries=ph.Boundaries(x="pml", y="periodic", z="pml"),
        pml_num_layers=6,
        sources=[ph.PointDipole(
            center_um=(0.5, 0.4, 0.4), polarization="Ez",
            source_time=ph.GaussianPulse(freq0_hz=1.934e14,
                                         fwidth_hz=4.0e13))],
        monitors=[
            ph.FieldTimeMonitor(name="probe", center_um=(0.6, 0.4, 0.4),
                                fields=["Ez"]),
            ph.FluxMonitor(name="flux", axis="x", position_um=0.75,
                           freqs_hz=freqs)],
    )
    cloud_spec = cloud_sim.to_wire_dict()

    # The coordinator stores and executes exactly these no-newline bytes;
    # phsolver hashes the file it reads, so provenance.input_sha256 matches
    # the workbench's deterministic reconstruction — as on the live service.
    cache_dir = tmp_path / "cloud-cache" / "job-fresh-open"
    cache_dir.mkdir(parents=True)
    spec_path = tmp_path / "coordinator-spec.json"
    spec_path.write_bytes(cloud_sim.to_wire_json(indent=0).encode("utf-8"))
    solver_proc = subprocess.run(
        [str(_real_solver()), "run", str(spec_path), "--output",
         str(cache_dir), "--progress", "none"],
        capture_output=True, text=True, timeout=300)
    assert solver_proc.returncode == 0, solver_proc.stderr
    assert not (cache_dir / "sim.json").exists()  # deployed-worker shape
    fake_data = service.load_result(cache_dir)

    class DownloadedJob:
        job_id = "job-fresh-open"

        def result(self):
            return fake_data

    monkeypatch.setattr(
        web, "preflight", lambda *_args, **_kwargs: _accepted(quote_usd=1))
    monkeypatch.setattr(
        web, "run_async", lambda *_args, **_kwargs: DownloadedJob())
    monkeypatch.setattr(web, "job_status", lambda job_id: {
        "job_id": job_id, "state": "succeeded", "actual_micros": 1_000_000,
    })

    with _app_client(run_root) as first:
        quote = first.post("/api/cloud/preflight", json={
            "spec": cloud_spec, "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        assert first.post("/api/cloud/run", json={
            "spec": cloud_spec, "preflight_token": quote["token"],
        }).status_code == 200
        deadline = time.monotonic() + 10
        completed = None
        while time.monotonic() < deadline:
            completed = first.get("/api/cloud/run/status").json()
            if completed.get("session"):
                break
            time.sleep(0.02)
        assert completed and completed["download_status"] == "completed"
        run_id = completed["ledger_run_id"]
        first_probe = first.get("/api/monitor/probe/timeseries").json()
        first_flux = first.get("/api/monitor/flux/spectrum").json()

    # Fresh session, dead service, and the download cache is GONE (the user
    # cleared ~/.cache): the reload must live entirely on the durable archive.
    import shutil

    shutil.rmtree(cache_dir)

    def _no_cloud(*_args, **_kwargs):
        raise AssertionError("fresh-session reload must not touch the service")

    for entry in ("preflight", "run_async", "job_status", "resume", "cancel"):
        monkeypatch.setattr(web, entry, _no_cloud)

    with _app_client(run_root) as fresh:
        assert fresh.get("/api/cloud/run/status").json() == {
            "status": "idle", "download_status": "idle",
        }
        history = fresh.get("/api/runs").json()["runs"]
        sealed = next(item for item in history if item["run_id"] == run_id)
        assert sealed["recorded_status"] == "completed"
        assert sealed["device"] == "cloud:gpu:mi300x"

        reopened = fresh.post(f"/api/runs/{run_id}/open")
        assert reopened.status_code == 200
        session_payload = reopened.json()
        assert session_payload["run_id"] == run_id
        names = {m["name"] for m in session_payload["monitors"]}
        assert {"probe", "flux"} <= names

        assert fresh.get("/api/monitor/probe/timeseries").json() == first_probe
        assert fresh.get("/api/monitor/flux/spectrum").json() == first_flux
        # The archived spec is loadable as a model and equals the submission.
        spec_response = fresh.get("/api/result/spec")
        assert spec_response.status_code == 200

        # And the run is a usable starting point for further design work: the
        # durable request spec recovers an editable workspace equal to the
        # submitted simulation.
        recovered = fresh.post(f"/api/runs/{run_id}/workspace")
        assert recovered.status_code == 200
        recovered_sim, _ = service.parse_sim_spec(
            recovered.json()["spec"])
        assert recovered_sim.to_wire_dict() == cloud_spec


def test_ledger_open_resolves_restored_cloud_recovery(monkeypatch, tmp_path):
    run_root = tmp_path / "runs"
    cloud_spec = _spec()
    cloud_spec["monitors"] = []
    cloud_sim, _ = service.parse_sim_spec(cloud_spec)
    canonical = cloud_sim.to_wire_json(indent=0)
    cache_dir = tmp_path / "cloud-cache" / "job-sealed-unopened"
    cache_dir.mkdir(parents=True)
    cache_dir.joinpath("manifest.json").write_text(json.dumps({
        "manifest_version": "1", "schema_version": "1.0.0",
        "run": {"n_steps": 5, "steps_run": 5, "dt_s": 1e-16,
                "aborted": False, "abort_reason": ""},
        "grid": {"shape": [2, 2, 2], "dl_um": 0.1},
        "provenance": {
            "input_sha256": __import__("hashlib").sha256(
                canonical.encode("utf-8")).hexdigest(),
        },
        "monitors": [],
    }) + "\n", encoding="utf-8")
    fake_data = service.load_result(cache_dir)

    class DownloadedJob:
        job_id = "job-sealed-unopened"

        def result(self):
            return fake_data

    monkeypatch.setattr(web, "preflight", lambda *_args, **_kwargs: _accepted(quote_usd=1))
    monkeypatch.setattr(web, "run_async", lambda *_args, **_kwargs: DownloadedJob())
    monkeypatch.setattr(web, "job_status", lambda job_id: {
        "job_id": job_id, "state": "succeeded", "actual_micros": 1_000_000,
    })

    original_session = service.session
    activation_calls = 0

    def fail_once_after_seal(data, result_id, run_id=None):
        nonlocal activation_calls
        activation_calls += 1
        if activation_calls == 1:
            raise RuntimeError("injected crash window after cloud seal")
        return original_session(data, result_id, run_id=run_id)

    monkeypatch.setattr(service, "session", fail_once_after_seal)

    with _app_client(run_root) as first:
        quote = first.post("/api/cloud/preflight", json={
            "spec": cloud_spec, "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        assert first.post("/api/cloud/run", json={
            "spec": cloud_spec, "preflight_token": quote["token"],
        }).status_code == 200
        deadline = time.monotonic() + 2
        interrupted = None
        while time.monotonic() < deadline:
            interrupted = first.get("/api/cloud/run/status").json()
            if (interrupted.get("download_status") == "failed"
                    and interrupted.get("status") == "succeeded"
                    and interrupted.get("ledger_run_id")):
                break
            time.sleep(0.01)
        assert interrupted is not None
        sealed_id = interrupted["ledger_run_id"]

    # Sealed but never opened: this restart prompt is genuinely useful and
    # must keep restoring exactly as before.
    assert (run_root / "cloud-job-recovery.json").is_file()

    with _app_client(run_root) as restarted:
        restored = restarted.get("/api/cloud/run/status").json()
        assert restored["id"] == "job-sealed-unopened"
        assert restored["ledger_run_id"] == sealed_id
        assert "Workbench restarted" in restored["error"]

        # Opening the sealed local evidence resolves the recovery without a
        # paid resume; the prompt must not survive this or any later launch.
        opened = restarted.post(f"/api/runs/{sealed_id}/open")
        assert opened.status_code == 200
        assert opened.json()["run_id"] == sealed_id
        assert restarted.get("/api/cloud/run/status").json() == {
            "status": "idle", "download_status": "idle",
        }
        assert not (run_root / "cloud-job-recovery.json").exists()

    with _app_client(run_root) as second_restart:
        assert second_restart.get("/api/cloud/run/status").json() == {
            "status": "idle", "download_status": "idle",
        }


def test_cloud_archive_rejects_bundled_execution_input_from_another_simulation(
        monkeypatch, tmp_path):
    submitted_spec = _spec()
    submitted_spec["monitors"] = []
    executed_spec = dict(submitted_spec)
    executed_spec["run"] = {**submitted_spec["run"], "n_steps": 6}
    executed_sim, _ = service.parse_sim_spec(executed_spec)
    executed_bytes = (json.dumps(executed_sim.to_wire_dict()) + "\n").encode()

    cache_dir = tmp_path / "cloud-cache" / "job-wrong-input"
    cache_dir.mkdir(parents=True)
    cache_dir.joinpath("sim.json").write_bytes(executed_bytes)
    cache_dir.joinpath("manifest.json").write_text(json.dumps({
        "manifest_version": "1", "schema_version": "1.0.0",
        "run": {"n_steps": 6, "steps_run": 6, "dt_s": 1e-16,
                "aborted": False, "abort_reason": ""},
        "grid": {"shape": [2, 2, 2], "dl_um": 0.1},
        "provenance": {
            "input_sha256": __import__("hashlib").sha256(
                executed_bytes).hexdigest(),
        },
        "monitors": [],
    }) + "\n", encoding="utf-8")
    fake_data = service.load_result(cache_dir)

    class WrongInputJob:
        job_id = "job-wrong-input"

        def result(self):
            return fake_data

    monkeypatch.setattr(
        web, "preflight", lambda *_args, **_kwargs: _accepted(quote_usd=1))
    monkeypatch.setattr(web, "run_async", lambda *_args, **_kwargs: WrongInputJob())
    monkeypatch.setattr(web, "job_status", lambda job_id: {
        "job_id": job_id, "state": "succeeded", "actual_micros": 1_000_000,
    })

    with _app_client(tmp_path / "runs") as client:
        quote = client.post("/api/cloud/preflight", json={
            "spec": submitted_spec, "device": "gpu:mi300x", "max_usd": 5,
        }).json()
        assert client.post("/api/cloud/run", json={
            "spec": submitted_spec, "preflight_token": quote["token"],
        }).status_code == 200
        deadline = time.monotonic() + 2
        current = None
        while time.monotonic() < deadline:
            current = client.get("/api/cloud/run/status").json()
            if current.get("download_status") == "failed":
                break
            time.sleep(0.01)
        assert current is not None
        assert current["download_status"] == "failed"
        assert "differs from the submitted immutable simulation" in current["error"]
        assert not [item for item in client.get("/api/runs").json()["runs"]
                    if item["recorded_status"] == "completed"]


def test_cloud_history_is_sanitized_and_resumable(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "get_config", lambda: object())
    monkeypatch.setattr(web, "list_jobs", lambda: [{
        "job_id": "job-history", "name": "workbench-abc",
        "state": "succeeded", "device": "gpu", "quote_micros": 5,
        "actual_micros": 4, "refunded_micros": 1,
        "quote_id": "must-not-leak", "download_url": "must-not-leak",
    }])

    with _app_client(tmp_path / "runs") as client:
        payload = client.get("/api/cloud/jobs").json()

    assert payload["error"] is None
    assert payload["jobs"] == [{
        "id": "job-history", "name": "workbench-abc",
        "status": "succeeded", "device": "gpu", "progress": None,
        "quote_usd": 0.000005, "actual_usd": 0.000004,
        "refunded_usd": 0.000001, "created_at": None,
        "updated_at": None, "finished_at": None, "error": None,
    }]
