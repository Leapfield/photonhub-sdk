"""Loopback-only browser security contracts for the local workbench API."""

import io
import threading

import pytest


def _client(tmp_path, *, base_url="http://testserver", release_identity=None):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    return TestClient(
        create_app(
            run_root=tmp_path / "runs", release_identity=release_identity,
        ),
        base_url=base_url,
    )


def test_rejects_untrusted_host(tmp_path):
    with _client(tmp_path) as client:
        response = client.get(
            "/api/health",
            headers={"Host": "attacker.example"},
        )

    assert response.status_code == 400
    assert response.text == "Invalid host header"


def test_rejects_cross_origin_browser_mutation(tmp_path):
    with _client(tmp_path, base_url="http://localhost") as client:
        evil_origin = client.post(
            "/api/workspace/new",
            headers={"Origin": "https://attacker.example"},
            json={},
        )
        other_loopback_origin = client.post(
            "/api/workspace/new",
            headers={"Origin": "http://localhost:9999"},
            json={},
        )
        other_scheme = client.post(
            "/api/workspace/new",
            headers={"Origin": "https://localhost"},
            json={},
        )
        fetch_metadata_only = client.post(
            "/api/workspace/new",
            headers={"Sec-Fetch-Site": "cross-site"},
            json={},
        )

    assert evil_origin.status_code == 403
    assert evil_origin.json()["detail"] == "cross-origin browser mutation blocked"
    assert other_loopback_origin.status_code == 403
    assert other_scheme.status_code == 403
    assert fetch_metadata_only.status_code == 403
    assert fetch_metadata_only.json()["detail"] == "cross-site browser mutation blocked"


def test_allows_same_origin_browser_mutation(tmp_path):
    with _client(tmp_path, base_url="http://localhost") as client:
        response = client.post(
            "/api/workspace/new",
            headers={
                "Origin": "http://localhost",
                "Sec-Fetch-Site": "same-origin",
            },
            json={},
        )

    assert response.status_code == 200
    assert response.json()["dirty"] is True


def test_normal_header_free_test_client_remains_usable(tmp_path):
    with _client(tmp_path) as client:
        health = client.get("/api/health")
        mutation = client.post("/api/workspace/new", json={})

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert mutation.status_code == 200


def test_installed_release_identity_is_reported_without_artifact_paths(tmp_path):
    identity = {
        "manifest_version": "1",
        "status": "candidate",
        "product": {"name": "PhotonHub Workbench", "version": "0.0.1",
                    "app_id": "com.photonhub.workbench"},
        "source": {"git_sha": "123456789abc"},
        "contracts": {"input_schema_version": "1", "result_schema_version": "1"},
        "physics": {"physics_release_id": "physics-beta",
                    "validation_profile_id": "mac-arm64-candidate"},
        "build": {"build_id": "a" * 64, "created_at": "2026-07-22T00:00:00Z",
                  "platform": "macos", "arch": "arm64", "compiler": "AppleClang",
                  "openmp_runtime": "libomp"},
        "solver": {"relative_path": "solver/phsolver", "name": "phsolver",
                   "version": "0.0.1", "git_sha": "123456789abc", "schema_major": 1,
                   "gpu": False, "sha256": "b" * 64, "capabilities": {},
                   "capabilities_sha256": "c" * 64},
        "sidecar": {"relative_path": "sidecar/photonhub-serve-viz",
                    "version": "0.0.1", "sha256": "d" * 64},
        "legal": {"community_eula_version": "beta", "privacy_version": "beta",
                  "third_party_notices_sha256": "e" * 64},
        "artifacts": [{"path": "solver/phsolver", "bytes": 1, "sha256": "b" * 64}],
    }
    with _client(tmp_path, release_identity=identity) as client:
        health = client.get("/api/health")
        response = client.get("/api/release")

    assert health.json()["release_build_id"] == "a" * 64
    assert response.status_code == 200
    assert response.json()["product"]["name"] == "PhotonHub Workbench"
    assert "artifacts" not in response.json()


def test_release_identity_route_is_absent_in_source_mode(tmp_path):
    with _client(tmp_path) as client:
        response = client.get("/api/release")
    assert response.status_code == 404


def test_desktop_api_requires_per_launch_capability_and_never_echoes_it(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    capability = "fresh-desktop-capability"
    requested = threading.Event()
    with TestClient(create_app(
            run_root=tmp_path / "runs", launch_token=capability,
            shutdown_callback=requested.set)) as client:
        public_root = client.get("/")
        missing_health = client.get("/api/health")
        wrong_health = client.get("/api/health", headers={
            "X-PhotonHub-Launch-Capability": "stale-desktop-capability",
        })
        missing_local = client.post("/api/workspace/new", json={})
        missing_cloud = client.get("/api/cloud/status")
        duplicate = client.get("/api/health", headers=[
            ("X-PhotonHub-Launch-Capability", capability),
            ("X-PhotonHub-Launch-Capability", capability),
        ])
        assert not requested.is_set()

        headers = {"X-PhotonHub-Launch-Capability": capability}
        health = client.get("/api/health", headers=headers)
        local = client.post("/api/workspace/new", headers=headers, json={})
        accepted = client.post(
            "/api/shutdown", headers=headers,
        )

    assert public_root.status_code == 200
    assert capability not in public_root.text
    for response in (
        missing_health, wrong_health, missing_local, missing_cloud, duplicate,
    ):
        assert response.status_code == 401
        assert response.json() == {"detail": "desktop API capability required"}
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert "launch_token" not in health.json()
    assert "launch_capability" not in health.json()
    assert "run_root" not in health.json()
    assert capability not in health.text
    assert local.status_code == 200
    assert accepted.status_code == 200
    assert accepted.json() == {"ok": True}
    assert requested.is_set()


def test_desktop_capability_wraps_future_event_and_websocket_routes(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import WebSocket
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    from photonhub.viz.server import create_app

    capability = "event-capability"
    app = create_app(run_root=tmp_path / "runs", launch_token=capability)

    @app.get("/api/events")
    def events():
        return StreamingResponse(iter([b"data: ready\n\n"]), media_type="text/event-stream")

    @app.websocket("/api/events/ws")
    async def events_ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("ready")
        await websocket.close()

    with TestClient(app) as client:
        denied_event = client.get("/api/events")
        allowed_event = client.get("/api/events", headers={
            "X-PhotonHub-Launch-Capability": capability,
        })
        with pytest.raises(WebSocketDisconnect) as denied_ws:
            with client.websocket_connect("/api/events/ws"):
                pass
        with client.websocket_connect(
            "/api/events/ws",
            headers={"X-PhotonHub-Launch-Capability": capability},
        ) as websocket:
            assert websocket.receive_text() == "ready"

    assert denied_event.status_code == 401
    assert allowed_event.status_code == 200
    assert allowed_event.text == "data: ready\n\n"
    assert denied_ws.value.code == 4401


def test_shutdown_route_is_absent_outside_owned_desktop_launch(tmp_path):
    with _client(tmp_path) as client:
        response = client.post(
            "/api/shutdown",
            headers={"X-PhotonHub-Launch-Capability": "anything"},
        )

    assert response.status_code == 404


def test_parent_eof_requests_native_shutdown_without_a_signal(monkeypatch):
    from photonhub.viz import server

    class ClosedParentPipe:
        def isatty(self):
            return False

        def read(self):
            return ""

    requested = threading.Event()
    monkeypatch.setattr("sys.stdin", ClosedParentPipe())
    server._watch_parent(requested.set)
    assert requested.wait(1)


def test_desktop_launch_capability_uses_private_stdin_framing(monkeypatch):
    from photonhub.viz import server

    capability = "a" * 64
    monkeypatch.setattr("sys.stdin", io.StringIO(capability + "\nremaining"))

    assert server._read_desktop_launch_capability() == capability
    assert __import__("sys").stdin.read() == "remaining"


@pytest.mark.parametrize("value", [
    "short\n",
    "g" * 64 + "\n",
    "a" * 64,
    "a" * 65 + "\n",
])
def test_desktop_launch_capability_rejects_invalid_stdin(monkeypatch, value):
    from photonhub.viz import server

    monkeypatch.setattr("sys.stdin", io.StringIO(value))
    with pytest.raises(ValueError, match="launch capability"):
        server._read_desktop_launch_capability()
