import hashlib
import json
import socket
import sys
from pathlib import Path

import pytest

import photonhub as ph

# CI runs pytest with ``working-directory: photonhub``. Keep the repository's
# runtime smoke module importable there without relying on the caller's cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import beta_workbench_smoke


def _installed_mac_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(beta_workbench_smoke.sys, "platform", "darwin")
    monkeypatch.setattr(beta_workbench_smoke.platform, "machine", lambda: "arm64")
    root = tmp_path / "PhotonHub Workbench.app/Contents/Resources"
    files = {
        "app.asar": b"sealed electron application",
        "ui/index.html": b"<!doctype html><title>PhotonHub</title>",
        "solver/phsolver": b"native solver fixture",
        "sidecar/photonhub-serve-viz": b"frozen sidecar fixture",
    }
    records = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if relative in {"solver/phsolver", "sidecar/photonhub-serve-viz"}:
            path.chmod(0o755)
        records.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    by_path = {record["path"]: record for record in records}
    manifest = {
        "status": "candidate",
        "product": {
            "name": "PhotonHub Workbench",
            "app_id": "com.photonhub.workbench",
            "version": "0.0.1",
        },
        "build": {"platform": "macos", "arch": "arm64", "build_id": "fixture"},
        "solver": {
            "relative_path": "solver/phsolver",
            "sha256": by_path["solver/phsolver"]["sha256"],
            "gpu": False,
            "workbench_authorization_required": True,
        },
        "sidecar": {
            "relative_path": "sidecar/photonhub-serve-viz",
            "sha256": by_path["sidecar/photonhub-serve-viz"]["sha256"],
        },
        "artifacts": sorted(records, key=lambda record: record["path"]),
    }
    (root / "release.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_health_is_minimal_and_never_echoes_launch_secrets():
    assert beta_workbench_smoke.health_is_ready({"ok": True})
    assert not beta_workbench_smoke.health_is_ready({
        "ok": True, "launch_token": "fresh-launch",
    })
    assert not beta_workbench_smoke.health_is_ready({
        "ok": True, "launch_capability": "fresh-launch",
    })
    assert not beta_workbench_smoke.health_is_ready({
        "ok": True, "run_root": "/private/archive",
    })


def test_require_plotly_figure_rejects_empty_payload():
    figure = {"data": [{"type": "scatter"}], "layout": {"title": "ready"}}
    assert (
        beta_workbench_smoke.require_plotly_figure(figure, "fixture")
        is figure
    )
    with pytest.raises(
        beta_workbench_smoke.WorkbenchSmokeError,
        match="did not return a renderable figure",
    ):
        beta_workbench_smoke.require_plotly_figure(
            {"data": [], "layout": {}},
            "fixture",
        )


def test_source_checkout_env_prepends_repo_package():
    env = beta_workbench_smoke.source_checkout_env(
        {"PYTHONPATH": "/tmp/stale-checkout", "KEEP": "yes"}
    )

    expected = Path(beta_workbench_smoke.__file__).resolve().parents[1] / "photonhub"
    assert env["PYTHONPATH"].split(beta_workbench_smoke.os.pathsep) == [
        str(expected),
        "/tmp/stale-checkout",
    ]
    assert env["KEEP"] == "yes"


def test_installed_resources_accepts_only_manifest_bound_native_payload(
    tmp_path, monkeypatch
):
    root = _installed_mac_fixture(tmp_path, monkeypatch)

    installed = beta_workbench_smoke.installed_resources(root)

    assert installed.root == root.resolve()
    assert installed.solver == (root / "solver/phsolver").resolve()
    assert installed.sidecar == (root / "sidecar/photonhub-serve-viz").resolve()
    (root / "solver/phsolver").write_bytes(b"tampered")
    with pytest.raises(beta_workbench_smoke.WorkbenchSmokeError, match="size changed"):
        beta_workbench_smoke.installed_resources(root)


def test_installed_resources_rejects_prepackage_stage(tmp_path, monkeypatch):
    root = _installed_mac_fixture(tmp_path, monkeypatch)
    (root / "app.asar").unlink()

    with pytest.raises(beta_workbench_smoke.WorkbenchSmokeError, match="packaged app.asar"):
        beta_workbench_smoke.installed_resources(root)


def test_installed_env_removes_all_developer_runtime_overrides(tmp_path):
    solver = tmp_path / "phsolver"
    env = beta_workbench_smoke.installed_env(
        solver,
        tmp_path / "run-root",
        {
            "SIMUPOD_SOLVER": "stale",
            "PYTHONHOME": "stale",
            "PYTHONPATH": "stale",
            "PHOTONHUB_PY": "stale",
            "VIRTUAL_ENV": "stale",
            "KEEP": "yes",
        },
    )

    assert env["PHOTONHUB_SOLVER"] == str(solver)
    assert env["MPLBACKEND"] == "Agg"
    assert env["MPLCONFIGDIR"] == str(tmp_path / "run-root/.matplotlib")
    assert env["KEEP"] == "yes"
    for name in ("SIMUPOD_SOLVER", "PYTHONHOME", "PYTHONPATH", "PHOTONHUB_PY", "VIRTUAL_ENV"):
        assert name not in env


def test_free_loopback_port_returns_bindable_port():
    port = beta_workbench_smoke.free_loopback_port()
    assert 0 < port < 65536
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_poll_completed_run_accepts_normal_lifecycle(monkeypatch):
    class Client:
        def __init__(self):
            self.payloads = iter(
                [
                    {"status": "queued"},
                    {"status": "running"},
                    {"status": "completed", "run_id": "run-1"},
                ]
            )

        def get(self, _path):
            return next(self.payloads)

    monkeypatch.setattr(beta_workbench_smoke.time, "sleep", lambda _delay: None)
    result = beta_workbench_smoke.poll_completed_run(Client(), 10)
    assert result == {"status": "completed", "run_id": "run-1"}


def test_poll_completed_run_surfaces_terminal_failure():
    class Client:
        def get(self, _path):
            return {"status": "failed", "error": "injected solver failure"}

    with pytest.raises(beta_workbench_smoke.WorkbenchSmokeError, match="injected"):
        beta_workbench_smoke.poll_completed_run(Client(), 1)


def test_workbench_http_smoke_with_current_solver(tmp_path):
    solver = ph.find_solver()
    if solver is None:
        pytest.skip("needs a current phsolver build")
    summary = beta_workbench_smoke.run_workbench_smoke(
        sys.executable,
        solver,
        tmp_path / "workbench-smoke",
        startup_timeout_s=20,
        run_timeout_s=60,
    )
    assert summary["ok"] is True
    assert summary["run_status"] == "completed"
    assert summary["geometry_integrity"] == "matched"
    assert summary["workspace_reopened"] is True
    assert summary["workspace_restored_automatically"] is True
    assert summary["workspace_recovered_from_result"] is True
    assert summary["design_visualization_rendered"] is True
    assert summary["result_scene_rendered"] is True
    assert summary["result_monitor_visualization_rendered"] is True
    assert summary["run_root_reused_after_restart"] is True
    assert summary["launch_capabilities_distinct"] is True
