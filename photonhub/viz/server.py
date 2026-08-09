"""Local viz HTTP facade — ``photonhub serve-viz <result-dir>``.

A localhost-only FastAPI server over the data core (:mod:`service`) and the plotly
builders (:mod:`figures`). It serves a bundle's scene, field slices, spectra, and
time-series as plotly JSON. The interactive viewer is the PhotonHub desktop app (or
the dev React UI), which talks to this same API — this server is headless.

``result-dir`` is optional: launch empty, then Open a bundle or Preview a spec file
in the app. The same server also renders a live ``Simulation`` spec under
``/api/preview/*`` (the input as data — watched, not executed).

Run:  python -m photonhub.viz.server [result-dir] [--port N] [--no-open]
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import threading
import time
import uuid
import warnings
import webbrowser
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from . import checks, figures, service
from .ledger import LedgerError, LedgerNotFound, RunLedger
from .notebook import NotebookKernel, WorkbenchHooks


_WORKSPACE_RECOVERY_VERSION = 1
_CLOUD_RECOVERY_VERSION = 1
_WORKBENCH_CLOUD_MAX_USD = 5.0
_MAX_CLOUD_SPEC_BYTES = 16 * 1024 * 1024
_PINNED_CLOUD_SOLVER = re.compile(r"[0-9a-f]{12,64}\Z")
_DESKTOP_CAPABILITY_HEADER = b"x-photonhub-launch-capability"
_DESKTOP_CAPABILITY_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class _DesktopCapabilityMiddleware:
    """Authenticate every desktop-owned API request before route dispatch.

    Static UI assets intentionally remain public on loopback so Chromium can
    bootstrap the renderer.  Once Electron supplies a launch capability,
    however, every HTTP *and* WebSocket path under ``/api`` fails closed.  The
    dedicated header is attached only by Electron's main-process proxy; it is
    never placed in a page, URL, response, command line, or child environment.
    """

    def __init__(self, app, launch_capability: Optional[str] = None):
        self.app = app
        self.launch_capability = launch_capability

    async def __call__(self, scope, receive, send):
        scope_type = scope.get("type")
        path = scope.get("path", "")
        protected = path == "/api" or path.startswith("/api/")
        if (self.launch_capability is None or not protected
                or scope_type not in {"http", "websocket"}):
            await self.app(scope, receive, send)
            return

        supplied_values = [
            value for name, value in scope.get("headers", [])
            if name.lower() == _DESKTOP_CAPABILITY_HEADER
        ]
        authenticated = (
            len(supplied_values) == 1
            and hmac.compare_digest(
                supplied_values[0], self.launch_capability.encode("ascii"))
        )
        if authenticated:
            await self.app(scope, receive, send)
            return

        if scope_type == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return

        payload = b'{"detail":"desktop API capability required"}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        })
        await send({"type": "http.response.body", "body": payload})


def _workbench_cloud_solver(release_identity: Optional[dict]) -> str:
    """Select the immutable cloud solver ref from trusted shell state."""
    value = None
    if release_identity is not None:
        solver = release_identity.get("solver")
        if isinstance(solver, dict):
            value = solver.get("git_sha")
    else:
        # Source-development only. Installed builds always use the verified
        # release manifest and never inherit a launcher override.
        value = os.environ.get("PHOTONHUB_CLOUD_SOLVER")
    if not isinstance(value, str) or not _PINNED_CLOUD_SOLVER.fullmatch(value):
        raise ValueError(
            "Workbench cloud execution has no immutable solver release; "
            "repair the installation or configure PHOTONHUB_CLOUD_SOLVER in source development"
        )
    return value


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_size),
        int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _read_file_snapshot(path: Path) -> tuple[bytes, dict]:
    """Read one stable file snapshot and return its user-facing identity.

    The SHA-256 binds Save to the exact bytes the workbench opened, while mtime
    and size make conflicts explainable in the UI.  The before/after signatures
    reject a file that is replaced or rewritten while it is being read.
    """
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read()
        after_fd = os.fstat(stream.fileno())
    after_path = path.stat()
    if (_stat_signature(before) != _stat_signature(after_fd)
            or _stat_signature(before) != _stat_signature(after_path)
            or len(raw) != int(before.st_size)):
        raise OSError(f"{path} changed while it was being read")
    return raw, {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mtime_ns": int(after_path.st_mtime_ns),
        "size": int(after_path.st_size),
    }


def _file_identity(path: Path) -> dict:
    return _read_file_snapshot(path)[1]


def _same_file_identity(expected: object, actual: object) -> bool:
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    return all(expected.get(key) == actual.get(key)
               for key in ("sha256", "mtime_ns", "size"))

_INDEX = """<!doctype html><meta charset=utf-8><title>PhotonHub viz API</title>
<body style="font:14px system-ui;max-width:640px;margin:48px auto;color:#333;line-height:1.6">
<h2>PhotonHub — local viz API</h2>
<p>This is the headless data server (the HTTP facade). The interactive viewer is the
PhotonHub desktop app, or the dev React UI which proxies to <code>/api</code>.</p>
<p>Result endpoints: <a href="/api/session">/api/session</a> ·
<code>/api/monitor/{name}/{meta,field,spectrum,timeseries}</code> ·
<a href="/api/scene">/api/scene</a> · <code>/api/eps</code></p>
<p>Preview (a live spec file): <code>POST /api/preview</code> ·
<code>/api/preview/{status,scene,eps}</code></p>
</body>"""


def create_app(result_dir: Optional[str | Path] = None,
               ui_dir: Optional[str | Path] = None,
               run_root: Optional[str | Path] = None,
               launch_token: Optional[str] = None,
               shutdown_callback: Optional[Callable[[], None]] = None,
               release_identity: Optional[dict] = None):
    """Build the FastAPI app. The loaded bundle lives in a mutable holder so the UI
    can open a different result dir without restarting; ``result_dir`` is optional —
    the app launches empty (Open a bundle or use Preview). If ``ui_dir`` is given the
    built web UI is served at ``/`` (same-origin with the API, for the packaged
    desktop app); otherwise ``/`` is a headless info page."""
    from contextlib import asynccontextmanager

    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    initial_data = service.load_result(result_dir) if result_dir else None
    # Direct/library callers (notably TestClient) get an isolated history.  The
    # CLI resolves a persistent default before calling us, and the packaged shell
    # passes its user-data run root explicitly.
    if run_root is None:
        run_root = tempfile.mkdtemp(prefix="photonhub-run-ledger-")
    ledger = RunLedger(run_root)
    workspace_recovery_path = ledger.root / "workspace-recovery.json"
    cloud_recovery_path = ledger.root / "cloud-job-recovery.json"

    def _persist_workspace(pv: dict) -> None:
        """Atomically retain the last schema-valid canonical workspace."""
        payload = {
            "version": _WORKSPACE_RECOVERY_VERSION,
            "path": pv.get("path"),
            "file_identity": pv.get("file_identity"),
            "dirty": bool(pv.get("dirty", False)),
            "warnings": list(pv.get("warnings", [])),
            "starter": pv.get("starter"),
            "spec": pv["sim"].to_wire_dict(),
        }
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=ledger.root,
                prefix=".workspace-recovery.", suffix=".tmp", delete=False,
            ) as tmp:
                json.dump(
                    payload, tmp, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False,
                )
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            os.replace(tmp_name, workspace_recovery_path)
        except Exception:
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
            raise

    def _restore_workspace() -> Optional[dict]:
        if not workspace_recovery_path.is_file():
            return None
        try:
            record = json.loads(workspace_recovery_path.read_text(encoding="utf-8"))
            if (not isinstance(record, dict)
                    or record.get("version") != _WORKSPACE_RECOVERY_VERSION):
                raise ValueError("unsupported workspace recovery record")
            sim, parse_warnings = service.parse_sim_spec(record.get("spec"))
            path = record.get("path")
            if path is not None and not isinstance(path, str):
                raise ValueError("workspace recovery path is invalid")
            identity = record.get("file_identity")
            if identity is not None and (
                    not isinstance(identity, dict)
                    or not isinstance(identity.get("sha256"), str)
                    or not isinstance(identity.get("mtime_ns"), int)
                    or not isinstance(identity.get("size"), int)):
                raise ValueError("workspace recovery file identity is invalid")
            restored_warnings = list(record.get("warnings") or [])
            for message in parse_warnings:
                if message not in restored_warnings:
                    restored_warnings.append(message)
            mtime = (float(identity["mtime_ns"]) / 1e9
                     if isinstance(identity, dict) else 0.0)
            return {
                "path": path,
                "mtime": mtime,
                "observed_mtime": mtime,
                "file_identity": identity,
                "observed_identity": identity,
                "observed_fs_error": None,
                "sim": sim,
                "error": None,
                "dirty": bool(record.get("dirty", True)),
                "external_change": False,
                "warnings": restored_warnings,
                "starter": record.get("starter"),
            }
        except Exception as exc:  # corrupt recovery must not brick the app
            warnings.warn(
                f"ignoring unreadable workspace recovery record "
                f"{workspace_recovery_path}: {exc}", RuntimeWarning,
            )
            return None

    restored_preview = _restore_workspace()

    def _restore_cloud_job() -> Optional[dict]:
        """Restore only the resumable identity/cost summary of the last job.

        API keys and accepted quote ids never enter this file.  A restored job
        deliberately has no live handle or result session; the operator must
        resume that already-paid service id before its result can be opened.
        """
        if not cloud_recovery_path.is_file():
            return None
        try:
            record = json.loads(cloud_recovery_path.read_text(encoding="utf-8"))
            if (not isinstance(record, dict)
                    or record.get("version") != _CLOUD_RECOVERY_VERSION):
                raise ValueError("unsupported cloud recovery record")
            job_id = record.get("id")
            unresolved_submission = bool(record.get("unresolved_submission", False))
            if job_id is not None and (
                    not isinstance(job_id, str) or not job_id.strip()
                    or len(job_id) > 256 or any(ch.isspace() for ch in job_id)):
                raise ValueError("cloud recovery job id is invalid")
            name = record.get("name")
            if not isinstance(name, str) or not name.strip() or len(name) > 128:
                name = None
            if job_id is None and (not unresolved_submission or name is None):
                raise ValueError("cloud recovery record has no resolvable job identity")
            cloud_sim = None
            if record.get("spec") is not None:
                cloud_sim, _ = service.parse_sim_spec(record["spec"])
            ledger_run_id = record.get("ledger_run_id")
            if ledger_run_id is not None:
                ledger.validate_run_id(ledger_run_id)
            status = record.get("status")
            if status not in {
                    "submitted", "queued", "provisioning", "running",
                    "succeeded", "failed", "cancelled", "unknown"}:
                status = "unknown"
            return {
                "id": job_id,
                "name": name,
                "status": status,
                "device": str(record.get("device") or "gpu"),
                "quote_usd": record.get("quote_usd"),
                "actual_usd": record.get("actual_usd"),
                "refunded_usd": record.get("refunded_usd"),
                "max_usd": record.get("max_usd"),
                "available_usd": record.get("available_usd"),
                "remaining_usd": record.get("remaining_usd"),
                "quote_expires_at": record.get("quote_expires_at"),
                "progress": None,
                "error": (
                    f"The submit response did not contain a job id. Check cloud "
                    f"history for {name!r} and resume that exact job if present. "
                    f"If it is absent, do not submit again; contact the beta operator "
                    f"to resolve the possibly billed request."
                    if unresolved_submission else
                    "Workbench restarted before this result was opened; resume "
                    "the existing cloud job to poll/download it."
                ),
                "poll_error": None,
                "download_status": "failed",
                "session": None,
                "output_dir": record.get("output_dir"),
                "submitted_at": record.get("submitted_at"),
                "finished_at": record.get("finished_at"),
                "cancel_requested": bool(record.get("cancel_requested", False)),
                "unresolved_submission": unresolved_submission,
                "ledger_run_id": ledger_run_id,
                "sim": cloud_sim,
                "recovery_error": None,
                "handle": None,
                "thread": None,
            }
        except Exception as exc:
            warnings.warn(
                f"ignoring unreadable cloud recovery record "
                f"{cloud_recovery_path}: {exc}", RuntimeWarning,
            )
            return None

    restored_cloud_job = _restore_cloud_job()
    state: dict = {
        # One immutable pointer is the result snapshot.  Publishing data and
        # revision as separate dict entries lets a concurrent request observe
        # a new bundle under the previous revision.
        "result": (
            initial_data,
            uuid.uuid4().hex if initial_data is not None else None,
            None,  # stable ledger run id; external/legacy bundles have none
        ),
        "result_seq": 0,
        "preview": restored_preview,
        "workspace_seq": 0,
        "job": None,
        "cloud_job": restored_cloud_job,
        "cloud_preflights": {},
        "cloud_submit_in_progress": False,
    }
    execution_lock = threading.Lock()
    job_lock = threading.Lock()
    result_lock = threading.Lock()
    workspace_lock = threading.Lock()
    cloud_lock = threading.Lock()
    modal_cache_lock = threading.Lock()
    modal_result_cache: dict[object, dict] = {}

    @asynccontextmanager
    async def lifespan(_app):
        yield
        # Uvicorn SIGTERM/Ctrl-C and Electron's parent-watch path all converge
        # here. Ask the shared process layer to terminate/kill phsolver, then
        # keep the sidecar alive long enough for that worker to reap its child.
        thread = None
        with job_lock:
            job = state["job"]
            if job is not None and job["status"] in {"queued", "running"}:
                job["cancel_event"].set()
                thread = job.get("thread")
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=4.0)

    app = FastAPI(title="photonhub viz", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )

    @app.middleware("http")
    async def reject_cross_origin_mutation(request: Request, call_next):
        """Keep browser writes confined to the loopback workbench origin.

        Host validation prevents DNS-rebinding requests from reaching the API.
        Origin is the stronger browser signal for mutation requests; when it is
        absent, Sec-Fetch-Site still catches cross-site form/navigation traffic.
        Header-free local clients remain usable in source/browser launches;
        Electron-owned launches are wrapped by the capability middleware.
        """
        if request.method.upper() not in {"GET", "HEAD", "OPTIONS", "TRACE"}:
            origin = request.headers.get("origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                    origin_host = (parsed.hostname or "").lower()
                    request_host = (request.url.hostname or "").lower()
                    origin_port = parsed.port or (
                        443 if parsed.scheme.lower() == "https" else 80)
                    request_port = request.url.port or (
                        443 if request.url.scheme.lower() == "https" else 80)
                    local_origin = (
                        parsed.scheme.lower() == request.url.scheme.lower()
                        and parsed.username is None
                        and parsed.password is None
                        and not parsed.path
                        and not parsed.query
                        and not parsed.fragment
                        and origin_host == request_host
                        and origin_port == request_port
                        and origin_host in {"127.0.0.1", "localhost", "testserver"}
                    )
                except ValueError:
                    local_origin = False
                if not local_origin:
                    return JSONResponse(
                        {"detail": "cross-origin browser mutation blocked"},
                        status_code=403,
                    )
            elif (request.headers.get("sec-fetch-site", "").strip().lower()
                  == "cross-site"):
                return JSONResponse(
                    {"detail": "cross-site browser mutation blocked"},
                    status_code=403,
                )
        return await call_next(request)

    def _data(rev: Optional[str] = None, *, monitor: Optional[str] = None,
              verify_metadata: bool = False):
        """Return one atomic open-result snapshot, verifying ledger evidence.

        External/legacy bundles retain their existing best-effort behavior.  A
        ledger-backed result is different: every provenance or numerical read
        checks the sealed record and the selected artifact before using the
        already-open ``SimulationData`` object.
        """
        data, result_id, run_id = state["result"]
        if data is None:
            raise HTTPException(404, "no result open — open a bundle (POST /api/open) or use Preview")
        if rev and rev != result_id:
            raise HTTPException(409, "stale result revision — the open bundle changed")
        if run_id is not None and (verify_metadata or monitor is not None):
            try:
                ledger.verify_bundle_metadata(run_id)
                if monitor is not None:
                    item = next((
                        value for value in data.manifest.get("monitors", [])
                        if isinstance(value, dict) and value.get("name") == monitor
                    ), None)
                    if item is None:
                        raise KeyError(
                            f"run {run_id} has no monitor {monitor!r}")
                    ledger.verify_artifact(run_id, str(item.get("file")))
            except LedgerError as exc:
                raise HTTPException(
                    409, f"open historical run {run_id} failed integrity checks: {exc}")
        return data

    def _bind_sealed_manifest(record: dict, data) -> None:
        expected = next((
            item.get("sha256")
            for item in (record.get("integrity") or {}).get("artifacts", [])
            if isinstance(item, dict) and item.get("path") == "manifest.json"
        ), None)
        actual = getattr(data, "manifest_sha256", None)
        if not expected or actual != expected:
            raise LedgerError(
                "loaded manifest bytes do not match the sealed run evidence")

    def _numerical_result(rev: Optional[str], monitor: str, render):
        """Materialize one numerical response inside a verify/use/verify guard."""
        data, result_id, run_id = state["result"]
        if data is None:
            raise HTTPException(404, "no result open")
        if rev and rev != result_id:
            raise HTTPException(409, "stale result revision — the open bundle changed")
        filename = None
        before = None
        if run_id is not None:
            try:
                ledger.verify_bundle_metadata(run_id)
                item = next((
                    value for value in data.manifest.get("monitors", [])
                    if isinstance(value, dict) and value.get("name") == monitor
                ), None)
                if item is None:
                    raise KeyError(f"run {run_id} has no monitor {monitor!r}")
                filename = str(item.get("file"))
                before = ledger.verify_artifact(run_id, filename)
            except LedgerError as exc:
                raise HTTPException(
                    409, f"open historical run {run_id} failed integrity checks: {exc}")
        payload = render(data)
        if run_id is not None and filename is not None and before is not None:
            try:
                after = ledger.verify_artifact(run_id, filename)
            except LedgerError as exc:
                raise HTTPException(409, str(exc))
            if after != before:
                raise HTTPException(
                    409, f"sealed artifact {filename!r} changed during numerical read")
        return payload

    def _multi_numerical_result(
        rev: Optional[str],
        monitor_names,
        render,
        *,
        cache: Optional[dict[object, dict]] = None,
        cache_lock=None,
    ):
        """Materialize a response that consumes several sealed monitor blobs.

        Modal decomposition reads every physical port plane.  Treat that set as
        one numerical transaction: verify the ledger metadata and every selected
        artifact before use, then verify the same artifact tokens again after
        rendering.  External/legacy bundles retain their existing best-effort
        behavior because they have no ledger run id.
        """
        data, result_id, run_id = state["result"]
        if data is None:
            raise HTTPException(404, "no result open")
        if rev and rev != result_id:
            raise HTTPException(
                409, "stale result revision — the open bundle changed")

        def evict_sim_cache() -> None:
            sim_path = data.output_dir / "sim.json"
            service._SIM_CACHE.pop(str(sim_path.resolve()), None)

        def external_source(value) -> Path:
            source = getattr(value, "_h5_path", None)
            if source is None:
                source = getattr(value, "manifest_path", None)
            if source is None:
                raise FileNotFoundError(
                    f"open result at {value.output_dir} has no readable source")
            return Path(source)

        def external_input_paths(value, selected_names) -> tuple[Path, ...]:
            """Files whose bytes determine an external modal decomposition."""
            paths = [
                external_source(value),
                value.output_dir / "sim.json",
            ]
            # HDF5 embeds both the manifest and every monitor array in the one
            # source file. Raw bundles store selected arrays beside the
            # manifest and must bind each one independently.
            if getattr(value, "_h5_path", None) is None:
                names_set = set(selected_names)
                for item in value.manifest.get("monitors", []):
                    if (not isinstance(item, dict)
                            or str(item.get("name")) not in names_set):
                        continue
                    filename = item.get("file")
                    if isinstance(filename, str) and filename:
                        paths.append(value.output_dir / filename)
            return tuple(dict.fromkeys(paths))

        def external_identities(paths) -> tuple[tuple[str, tuple], ...]:
            return tuple(
                (str(path), _stat_signature(path.stat()))
                for path in paths
            )

        before: dict[str, object] = {}
        if run_id is not None:
            try:
                ledger.verify_bundle_metadata(run_id)
                # modal post-processing consumes sim.json as numerical input,
                # while manifest.json resolves monitor names to artifacts. Keep
                # their tokens in the same before/after transaction as blobs.
                for filename in ("sim.json", "manifest.json"):
                    before[filename] = ledger.verify_artifact(run_id, filename)
            except LedgerError as exc:
                evict_sim_cache()
                raise HTTPException(
                    409,
                    f"open historical run {run_id} failed integrity checks: {exc}",
                ) from exc

        # An external cached request gets a fresh, identity-bound SimulationData
        # below while holding the single-flight lock. Sealed runs and uncached
        # callers can resolve names from their immutable open snapshot here.
        external_cached = (
            run_id is None and cache is not None and cache_lock is not None)
        names = []
        if not external_cached:
            # Parse the recipe only after verifying its sealed bytes. This
            # ordering also prevents a rejected tampered sim.json from entering
            # service's parsed-simulation cache before the integrity check.
            try:
                names = list(dict.fromkeys(
                    str(name) for name in monitor_names(data)))
            except Exception:
                evict_sim_cache()
                raise
            if not names:
                evict_sim_cache()
                raise ValueError(
                    "numerical result requires at least one monitor")
        if run_id is not None:
            try:
                manifest_by_name = {
                    str(item.get("name")): item
                    for item in data.manifest.get("monitors", [])
                    if isinstance(item, dict)
                }
                for name in names:
                    item = manifest_by_name.get(name)
                    if item is None:
                        raise LedgerError(
                            f"run {run_id} has no monitor {name!r}")
                    filename = item.get("file")
                    if not isinstance(filename, str) or not filename:
                        raise LedgerError(
                            f"run {run_id} monitor {name!r} has no artifact file")
                    before[filename] = ledger.verify_artifact(run_id, filename)
            except LedgerError as exc:
                evict_sim_cache()
                raise HTTPException(
                    409,
                    f"open historical run {run_id} failed integrity checks: {exc}",
                ) from exc

        cache_key: object = result_id
        sentinel = object()
        cached = sentinel
        cache_guard = False
        numerical_data = data
        external_paths: tuple[Path, ...] = ()
        external_before: tuple[tuple[str, tuple], ...] = ()
        try:
            if cache is not None and cache_lock is not None:
                # Single-flight identical decompositions. In development React
                # StrictMode may abort/restart the same GET; holding this lock
                # through render and the post-use integrity check prevents two
                # sparse eigensolve banks from running concurrently.
                cache_lock.acquire()
                cache_guard = True
            if external_cached:
                # The open SimulationData intentionally caches monitor arrays.
                # Re-open its cheap manifest/HDF5 envelope for each identity
                # check so a replaced blob can never be recomputed through the
                # old object's DataArray cache.
                source_path = external_source(data)
                base_paths = (
                    source_path,
                    data.output_dir / "sim.json",
                )
                base_before = external_identities(base_paths)
                numerical_data = service.load_result(source_path)
                if external_identities(base_paths) != base_before:
                    raise HTTPException(
                        409, "external result changed while it was being opened")
                names = list(dict.fromkeys(
                    str(name) for name in monitor_names(numerical_data)))
                if not names:
                    raise ValueError(
                        "numerical result requires at least one monitor")
                external_paths = external_input_paths(numerical_data, names)
                external_before = external_identities(external_paths)
                cache_key = (result_id, external_before)
            if cache is not None and cache_lock is not None:
                cached = cache.get(cache_key, sentinel)
            payload = (
                render(numerical_data) if cached is sentinel else cached)
            if (external_cached
                    and external_identities(external_paths) != external_before):
                evict_sim_cache()
                raise HTTPException(
                    409, "external result changed during numerical read")
            if run_id is not None:
                for filename, token in before.items():
                    try:
                        after = ledger.verify_artifact(run_id, filename)
                    except LedgerError as exc:
                        evict_sim_cache()
                        raise HTTPException(409, str(exc)) from exc
                    if after != token:
                        evict_sim_cache()
                        raise HTTPException(
                            409,
                            f"sealed artifact {filename!r} changed during numerical read",
                        )
            if cached is sentinel and cache is not None:
                cache[cache_key] = payload
                while len(cache) > 4:
                    cache.pop(next(iter(cache)))
            return payload
        except Exception:
            # A race can replace sim.json after the initial check and make the
            # parser fail. Do not retain a parse from an unverified interval.
            evict_sim_cache()
            raise
        finally:
            if cache_guard:
                cache_lock.release()

    @app.get("/api/health")
    def health():
        # An authenticated 200 is itself the desktop ownership proof. Never
        # echo the capability (or filesystem ownership details) in health.
        return {"ok": True, "release_build_id": (
                    release_identity.get("build", {}).get("build_id")
                    if release_identity is not None else None)}

    @app.get("/api/release")
    def desktop_release():
        """Report the content-bound identity of an installed Workbench build."""
        if release_identity is None:
            raise HTTPException(404, "desktop release identity is not configured")
        # Return only identity/capability metadata. The full artifact inventory
        # remains on disk in release.json and is verified before the HTTP server
        # starts; exposing hundreds of local resource paths adds no UI value.
        return {
            key: release_identity[key]
            for key in (
                "manifest_version", "status", "product", "source", "contracts",
                "physics", "build", "solver", "sidecar", "legal",
            )
        }

    @app.post("/api/shutdown")
    def shutdown():
        """Ask the owning desktop sidecar to enter normal ASGI shutdown.

        This route is deliberately unavailable to library/browser launches. A
        packaged Electron main process supplies a fresh capability on every
        launch through the outer API middleware.  The
        callback flips Uvicorn's normal ``should_exit`` flag, so FastAPI's
        lifespan cancels and reaps an active solver before the process exits.
        """
        if launch_token is None or shutdown_callback is None:
            raise HTTPException(404, "desktop shutdown is not configured")
        shutdown_callback()
        return {"ok": True}

    @app.get("/api/session")
    def get_session():
        d, result_id, run_id = state["result"]
        if d is None:  # launched empty — the UI shows an Open / Preview prompt
            return {"monitors": [], "run": {}, "grid": {}, "provenance": {},
                    "aborted": False, "abort_reason": None, "has_scene": False,
                    "output_dir": None, "result_id": None, "run_id": None,
                    "geometry": {"status": "missing", "expected_sha256": None,
                                 "actual_sha256": None}}
        if run_id is not None:
            # Small and cacheable, but catches a bundle changed after it was
            # explicitly opened from history.
            _data(result_id, verify_metadata=True)
        return service.session(d, result_id, run_id=run_id)

    @app.post("/api/open")
    def open_bundle(dir: str = Body(..., embed=True)):
        # Reserve arrival order before loading or deriving the session.  Result
        # bundles can be large, so completion order is not user-intent order.
        with result_lock:
            state["result_seq"] += 1
            request_seq = state["result_seq"]
        try:
            data = service.load_result(dir)
        except Exception as e:
            raise HTTPException(400, f"cannot open {dir!r}: {e}")
        result_id = uuid.uuid4().hex
        result_session = service.session(data, result_id, run_id=None)
        with result_lock:
            if request_seq != state["result_seq"]:
                raise HTTPException(409, "stale result open — a newer selection won")
            state["result"] = (data, result_id, None)
        return result_session

    # --- Workbench: schema-backed authoring, validation, saving, and run -----
    # These endpoints intentionally accept/return the canonical Simulation wire
    # document.  The desktop form is a projection over this IR, not a parallel
    # GUI-only model, and its Advanced JSON view keeps every additive solver
    # field accessible even before a bespoke control is added.
    def _validation_error(exc: Exception):
        issues = None
        errors = getattr(exc, "errors", None)
        if callable(errors):
            try:
                issues = errors(include_url=False, include_context=False)
            except TypeError:
                issues = errors()
        detail = {"message": str(exc)}
        if issues:
            detail["issues"] = issues
        raise HTTPException(422, detail=detail)

    def _workspace_payload(pv: dict):
        return {
            "path": pv.get("path") or "",
            "mtime": pv.get("mtime", 0.0),
            "file_identity": pv.get("file_identity"),
            "error": pv.get("error"),
            "dirty": bool(pv.get("dirty", False)),
            "external_change": bool(pv.get("external_change", False)),
            "warnings": list(pv.get("warnings", [])),
            "starter": pv.get("starter"),
            **service.sim_payload(pv["sim"]),
        }

    def _assert_no_stale_mode_sources(sim) -> None:
        try:
            service.assert_no_stale_mode_sources(sim)
        except service.StaleModeSourceError as exc:
            raise HTTPException(409, detail={
                "code": "stale_mode_source",
                "message": str(exc),
                "mode_source_statuses": exc.statuses,
            }) from exc

    def _assert_modal_ports_ready(sim) -> None:
        try:
            service.assert_modal_ports_ready(sim)
        except ValueError as exc:
            raise HTTPException(422, detail={
                "code": "modal_ports_not_ready",
                "message": str(exc),
            }) from exc

    def _begin_workspace_update() -> int:
        """Claim arrival order before any potentially slow parsing/meshing.

        Browser request cancellation does not stop a synchronous FastAPI worker
        thread.  The sequence makes workspace publication last-request-wins,
        rather than whichever parse happens to finish last.
        """
        with workspace_lock:
            state["workspace_seq"] += 1
            return state["workspace_seq"]

    def _publish_workspace(pv: dict, request_seq: int):
        with workspace_lock:
            if request_seq != state["workspace_seq"]:
                raise HTTPException(409, "stale workspace update — a newer edit won")
            _persist_workspace(pv)
            state["preview"] = pv
        return pv

    def _set_workspace(sim, *, path: Optional[str] = None, mtime: float = 0.0,
                       file_identity: Optional[dict] = None,
                       dirty: bool = False, warnings_: Optional[list] = None,
                       starter: Optional[dict] = None,
                       request_seq: Optional[int] = None):
        if request_seq is None:
            request_seq = _begin_workspace_update()
        pv = {
            "path": path,
            "mtime": mtime,
            "observed_mtime": mtime,
            "file_identity": file_identity,
            "observed_identity": file_identity,
            "observed_fs_error": None,
            "sim": sim,
            "error": None,
            "dirty": dirty,
            "external_change": False,
            "warnings": warnings_ or [],
            "starter": starter,
        }
        return _publish_workspace(pv, request_seq)

    def _update_workspace_sim(sim, *, request_seq: int, dirty: bool,
                              warnings_: Optional[list] = None):
        with workspace_lock:
            if request_seq != state["workspace_seq"]:
                raise HTTPException(409, "stale workspace update — a newer edit won")
            current = state["preview"] or {}
            pv = {
                "path": current.get("path"),
                "mtime": current.get("mtime", 0.0),
                "observed_mtime": current.get(
                    "observed_mtime", current.get("mtime", 0.0)),
                "file_identity": current.get("file_identity"),
                "observed_identity": current.get(
                    "observed_identity", current.get("file_identity")),
                "observed_fs_error": current.get("observed_fs_error"),
                "sim": sim,
                "error": (current.get("error")
                          if (current.get("external_change") or
                              current.get("observed_fs_error")) else None),
                "dirty": dirty,
                "external_change": bool(
                    current.get("external_change", False) or
                    current.get("observed_fs_error")),
                "warnings": warnings_ or [],
                "starter": current.get("starter"),
            }
            _persist_workspace(pv)
            state["preview"] = pv
        return pv

    def _preserve_workspace_sim(sim, *, request_seq: int,
                                warnings_: Optional[list] = None):
        """Durably preserve a recovery snapshot without inventing dirty state.

        The renderer can be ahead of the normal validation debounce, so a
        different canonical spec is necessarily an unsaved edit.  When the
        canonical spec is unchanged, retain the authoritative workspace dirty
        bit and all file/path identity metadata instead of turning a clean Save
        into an unsaved recovery draft.
        """
        with workspace_lock:
            if request_seq != state["workspace_seq"]:
                raise HTTPException(409, "stale workspace update — a newer edit won")
            current = state["preview"]
            if current is None:
                raise HTTPException(404, "no simulation workspace open")
            dirty = bool(
                current.get("dirty", False)
                or current["sim"].to_wire_dict() != sim.to_wire_dict()
            )
            pv = {
                "path": current.get("path"),
                "mtime": current.get("mtime", 0.0),
                "observed_mtime": current.get(
                    "observed_mtime", current.get("mtime", 0.0)),
                "file_identity": current.get("file_identity"),
                "observed_identity": current.get(
                    "observed_identity", current.get("file_identity")),
                "observed_fs_error": current.get("observed_fs_error"),
                "sim": sim,
                "error": (current.get("error")
                          if current.get("external_change") or
                              current.get("observed_fs_error") else None),
                "dirty": dirty,
                "external_change": bool(
                    current.get("external_change", False) or
                    current.get("observed_fs_error")),
                "warnings": warnings_ or [],
                "starter": current.get("starter"),
            }
            _persist_workspace(pv)
            state["preview"] = pv
        return pv

    @app.get("/api/workspace/schema")
    def workspace_schema():
        from ..components.simulation import Simulation
        return Simulation.model_json_schema()

    def _solver_snapshot():
        from ..runners.phsolver import _solver_subprocess_env, find_solver
        try:
            solver = find_solver()
        except Exception as exc:
            return {"available": False, "error": str(exc), "info": {},
                    "capabilities": {}}
        if solver is None:
            return {"available": False, "error": "phsolver binary not found",
                    "info": {}, "capabilities": {}}

        def probe(command: str):
            try:
                result = subprocess.run(
                    [str(solver), command], capture_output=True, text=True, timeout=10,
                    env=_solver_subprocess_env())
                if result.returncode != 0:
                    return {"error": (result.stderr or result.stdout).strip()}
                return json.loads(result.stdout)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return {"error": str(exc)}

        return {"available": True, "path": str(solver), "info": probe("info"),
                "capabilities": probe("capabilities")}

    @app.get("/api/solver")
    def solver_info():
        return _solver_snapshot()

    @app.post("/api/workspace/new")
    def new_workspace():
        request_seq = _begin_workspace_update()
        return _workspace_payload(_set_workspace(
            service.default_sim(), dirty=True,
            starter=service.default_starter(), request_seq=request_seq))

    @app.get("/api/workspace/examples")
    def workspace_examples():
        """List packaged, opt-in simulations without changing the draft."""
        return {"examples": service.example_starters()}

    @app.post("/api/workspace/example")
    def workspace_example(payload: dict = Body(...)):
        """Replace the draft with one explicitly selected packaged example."""
        request_seq = _begin_workspace_update()
        example_id = payload.get("id")
        if not isinstance(example_id, str) or not example_id.strip():
            raise HTTPException(422, "example id must be a non-empty string")
        try:
            sim, starter = service.example_sim(example_id)
        except KeyError:
            raise HTTPException(404, detail={
                "code": "example_not_found",
                "message": f"unknown Workbench example: {example_id}",
            })
        return _workspace_payload(_set_workspace(
            sim, dirty=True, starter=starter, request_seq=request_seq))

    @app.post("/api/workspace/from-result")
    def workspace_from_result():
        request_seq = _begin_workspace_update()
        # A ledger-backed result remains inside the integrity boundary even
        # when it is used only to seed a new editable setup.  In particular,
        # sim.json and manifest.json must still match the sealed inventory.
        data = _data(verify_metadata=True)
        sim = service.sim_for(data)
        if sim is None:
            raise HTTPException(
                409, "result geometry is missing or failed its input checksum; "
                     "the setup cannot be reconstructed safely")
        geometry = service.geometry_status(data)
        warning = ("Loaded a new unsaved setup from this result's checksum-matched sim.json."
                   if geometry["status"] == "matched" else
                   "Loaded a new unsaved setup from a legacy sim.json with no recorded input checksum; verify its provenance before reuse.")
        return _workspace_payload(_set_workspace(
            sim, dirty=True,
            warnings_=[warning],
            request_seq=request_seq,
        ))

    @app.get("/api/workspace")
    def get_workspace():
        if state["preview"] is None:
            raise HTTPException(404, "no simulation workspace open")
        return _workspace_payload(_reload_preview())

    @app.post("/api/workspace/validate")
    def validate_workspace(spec: dict = Body(..., embed=True)):
        request_seq = _begin_workspace_update()
        try:
            sim, messages = service.parse_sim_spec(spec)
        except Exception as exc:  # pydantic emits the authoritative field paths
            _validation_error(exc)
        return _workspace_payload(_update_workspace_sim(
            sim, request_seq=request_seq, dirty=True, warnings_=messages))

    @app.post("/api/workspace/monitor-check")
    def monitor_check_workspace(spec: dict = Body(..., embed=True)):
        """Advisory monitor findings for a draft spec (checks.py).

        Read-only: unlike ``validate`` this never touches the shared
        workspace state, so the UI can poll it while editing.  The engine's
        ``preflight`` remains the authoritative hard gate; these findings
        flag setups that would run fine and record silent garbage.
        """
        try:
            sim, _ = service.parse_sim_spec(spec)
        except Exception as exc:
            _validation_error(exc)
        return checks.monitor_checks(sim)

    @app.post("/api/workspace/recovery-preserve")
    def preserve_workspace_for_recovery(spec: dict = Body(..., embed=True)):
        """Validate and fsync the renderer snapshot for native shutdown.

        This endpoint is intentionally separate from ordinary editing
        validation: a clean, unchanged saved workspace must remain clean, while
        a renderer snapshot newer than the sidecar workspace is preserved as a
        dirty draft.  A missing workspace remains an authoritative 404 and is
        handled by the renderer as "nothing to preserve."
        """
        request_seq = _begin_workspace_update()
        try:
            sim, messages = service.parse_sim_spec(spec)
        except Exception as exc:
            _validation_error(exc)
        return _workspace_payload(_preserve_workspace_sim(
            sim, request_seq=request_seq, warnings_=messages))

    @app.post("/api/workspace/auto-grid")
    def auto_grid_workspace(payload: dict = Body(...)):
        """Compile the guided mesh settings into the canonical graded grid.

        Auto meshing deliberately remains a workflow operation instead of a
        second wire-format grid type: the saved Simulation contains the exact
        resolved coordinates that the native engine will execute.
        """
        request_seq = _begin_workspace_update()
        try:
            sim, messages = service.parse_sim_spec(payload.get("spec"))
        except Exception as exc:
            _validation_error(exc)

        raw = payload.get("settings")
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise HTTPException(422, "auto-grid settings must be an object")
        try:
            wavelength_nm = float(raw.get("wavelength_nm", 1550.0))
            steps_per_wvl = float(raw.get("steps_per_wvl", 20.0))
            max_grading = float(raw.get("max_grading", 1.3))
            min_nodes = int(raw.get("min_nodes", 4))
            axes = "".join(dict.fromkeys(str(raw.get("axes", "xyz")).lower()))
            if not wavelength_nm > 0:
                raise ValueError("target wavelength must be greater than 0 nm")
            if not steps_per_wvl > 0:
                raise ValueError("steps per wavelength must be greater than 0")
            if not axes or any(axis not in "xyz" for axis in axes):
                raise ValueError("graded axes must contain one or more of x, y, z")
            if min_nodes < 4:
                raise ValueError("minimum nodes must be at least 4")

            def positive_optional(key: str):
                value = raw.get(key)
                if value is None or value == "" or float(value) == 0:
                    return None
                parsed = float(value)
                if parsed < 0:
                    raise ValueError(f"{key} must be 0/blank or greater than 0")
                return parsed

            regions = []
            for index, region in enumerate(raw.get("refine_regions") or []):
                if not isinstance(region, (list, tuple)) or len(region) != 4:
                    raise ValueError(
                        f"refinement region {index + 1} must be [axis, lo_um, hi_um, dl_um]")
                axis, lo, hi, dl = region
                axis = str(axis).lower()
                lo, hi, dl = float(lo), float(hi), float(dl)
                if axis not in "xyz" or len(axis) != 1:
                    raise ValueError(f"refinement region {index + 1} has an invalid axis")
                if hi <= lo or dl <= 0:
                    raise ValueError(
                        f"refinement region {index + 1} requires hi > lo and dl > 0")
                regions.append((axis, lo, hi, dl))

            meshed = sim.with_auto_grid(
                wavelength_um=wavelength_nm / 1000.0,
                steps_per_wvl=steps_per_wvl,
                max_grading=max_grading,
                axes=axes,
                refine_pad_um=positive_optional("refine_pad_um"),
                min_nodes=min_nodes,
                dl_min_um=positive_optional("dl_min_um"),
                refine_regions=regions,
                snap_interfaces=bool(raw.get("snap_interfaces", True)),
                feature_ceil=bool(raw.get("feature_ceil", True)),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, {"message": str(exc), "issues": [
                {"loc": ["auto_grid"], "msg": str(exc), "type": "value_error"}
            ]})

        return _workspace_payload(_update_workspace_sim(
            meshed, request_seq=request_seq, dirty=True, warnings_=messages))

    @app.post("/api/workspace/mode-source/solve")
    def solve_workspace_mode_source(payload: dict = Body(...)):
        """Compile practical port controls into one canonical ModeSource.

        Solved field arrays remain part of the execution IR returned in
        ``spec`` but are intentionally absent from the request controls.  This
        is the Workbench equivalent of a Tidy3D/Lumerical port-mode solve, not
        an invitation to hand-edit tens of thousands of coupled samples.
        """
        request_seq = _begin_workspace_update()
        if not isinstance(payload, dict):
            raise HTTPException(422, "mode-source solve payload must be an object")
        try:
            sim, messages = service.parse_sim_spec(payload.get("spec"))
        except Exception as exc:
            _validation_error(exc)
        try:
            append = payload.get("append", False)
            if not isinstance(append, bool):
                raise service.ModeSourceSolveError(
                    "append", "append must be a boolean")
            if append:
                if payload.get("source_index") is not None:
                    raise service.ModeSourceSolveError(
                        "source_index",
                        "source_index must be null when appending a mode source",
                    )
                solved, summary = service.append_mode_source(
                    sim, payload.get("settings"), payload.get("seed"),
                )
            else:
                solved, summary = service.solve_mode_source(
                    sim, payload.get("source_index"), payload.get("settings"),
                )
        except service.ModeSourceSolveError as exc:
            raise HTTPException(422, detail={
                "message": str(exc),
                "issues": [{
                    "loc": ["mode_source", exc.field],
                    "msg": str(exc),
                    "type": "value_error",
                }],
            }) from exc
        except ImportError as exc:
            raise HTTPException(503, detail={
                "code": "mode_solver_unavailable",
                "message": f"the local Yee mode solver is unavailable: {exc}",
            }) from exc
        except (RuntimeError, ValueError) as exc:
            # The eigensolver's most common runtime error is an unavailable
            # requested TE/TM index.  Return it beside Configure rather than as
            # an opaque server failure.
            raise HTTPException(422, detail={
                "message": str(exc),
                "issues": [{
                    "loc": ["mode_source", "mode_index"],
                    "msg": str(exc),
                    "type": "mode_solve_error",
                }],
            }) from exc

        response = _workspace_payload(_update_workspace_sim(
            solved, request_seq=request_seq, dirty=True, warnings_=messages))
        response["mode_source_summary"] = summary
        return response

    def _gds_payload_path(payload: dict, stack) -> str:
        """Materialize the request's GDS bytes as a readable file path.

        Uploads arrive base64-encoded in JSON because the Electron API bridge
        forwards string bodies only (no multipart). A ``path`` is accepted as
        the programmatic/desktop-native alternative — the same local trust
        boundary as ``POST /api/preview``.
        """
        import base64
        import binascii

        content = payload.get("content_base64")
        path = payload.get("path")
        if content is not None:
            if not isinstance(content, str):
                raise HTTPException(422, "content_base64 must be a string")
            if len(content) > service.GDS_MAX_BYTES * 4 // 3 + 4:
                raise HTTPException(413, detail={
                    "code": "gds_too_large",
                    "message": "GDS upload exceeds the 64 MB import limit",
                })
            try:
                raw = base64.b64decode(content, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(
                    422, "content_base64 is not valid base64") from exc
            handle = stack.enter_context(tempfile.NamedTemporaryFile(
                prefix="workbench-import-", suffix=".gds"))
            handle.write(raw)
            handle.flush()
            return handle.name
        if not isinstance(path, str) or not path.strip():
            raise HTTPException(
                422, "provide the GDS as content_base64 or a local file path")
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise HTTPException(404, detail={
                "code": "gds_not_found",
                "message": f"no GDS file at {resolved}",
            })
        if resolved.stat().st_size > service.GDS_MAX_BYTES:
            raise HTTPException(413, detail={
                "code": "gds_too_large",
                "message": "GDS file exceeds the 64 MB import limit",
            })
        return str(resolved)

    def _gds_error(exc: service.GdsImportError) -> HTTPException:
        status = 503 if exc.code == "gds_reader_missing" else 422
        return HTTPException(status, detail={
            "code": exc.code, "message": str(exc)})

    @app.post("/api/gds/inspect")
    def gds_inspect(payload: dict = Body(...)):
        """Catalog an uploaded/local GDS: cells + per-layer polygon stats.

        A pure query — the workspace draft is untouched. The dialog calls this
        again when the user switches cells (layer content is per-cell).
        """
        cell_name = payload.get("cell_name")
        if cell_name is not None and not isinstance(cell_name, str):
            raise HTTPException(422, "cell_name must be a string")
        with contextlib.ExitStack() as stack:
            gds_path = _gds_payload_path(payload, stack)
            try:
                return service.gds_inspect(gds_path, cell_name=cell_name)
            except service.GdsImportError as exc:
                raise _gds_error(exc) from exc

    @app.post("/api/gds/import")
    def gds_import(payload: dict = Body(...)):
        """Convert selected GDS layers into structure wire dicts.

        Returns structures for the renderer to append through its ordinary
        editing path (mutate + validate) so undo/dirty semantics stay uniform;
        the server-side workspace draft is intentionally not touched here.
        """
        cell_name = payload.get("cell_name")
        if cell_name is not None and not isinstance(cell_name, str):
            raise HTTPException(422, "cell_name must be a string")
        axis = payload.get("axis", "z")
        try:
            min_area = float(payload.get("min_area_um2", 0.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "min_area_um2 must be a number") from exc
        with contextlib.ExitStack() as stack:
            gds_path = _gds_payload_path(payload, stack)
            try:
                return service.gds_import_structures(
                    gds_path, payload.get("layers"),
                    cell_name=cell_name, axis=axis,
                    offset_um=payload.get("offset_um") or (0.0, 0.0),
                    min_area_um2=min_area,
                    name_prefix=payload.get("name_prefix", "gds"),
                )
            except service.GdsImportError as exc:
                raise _gds_error(exc) from exc

    @app.post("/api/workspace/save")
    def save_workspace(payload: dict = Body(...)):
        request_seq = _begin_workspace_update()
        spec = payload.get("spec")
        force = payload.get("force", False)
        if not isinstance(force, bool):
            raise HTTPException(422, "force must be a boolean")
        requested_identity = payload.get("expected_identity")
        if requested_identity is not None and (
                not isinstance(requested_identity, dict)
                or not isinstance(requested_identity.get("sha256"), str)
                or not isinstance(requested_identity.get("mtime_ns"), int)
                or not isinstance(requested_identity.get("size"), int)):
            raise HTTPException(422, "expected_identity is invalid")
        with workspace_lock:
            current_workspace = state["preview"] or {}
            current_path = current_workspace.get("path")
        path = payload.get("path") or current_path
        if not path:
            raise HTTPException(400, "choose a .json path for this simulation")
        try:
            sim, messages = service.parse_sim_spec(spec)
        except Exception as exc:
            _validation_error(exc)
        target = Path(path).expanduser().resolve()
        if target.suffix.lower() != ".json":
            raise HTTPException(400, "simulation specs must use a .json filename")
        if not target.parent.is_dir():
            raise HTTPException(400, f"parent directory does not exist: {target.parent}")
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=target.parent,
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as tmp:
                tmp.write(sim.to_wire_json() + "\n")
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
        except OSError as exc:
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
            raise HTTPException(400, f"cannot save {str(target)!r}: {exc}")
        # Commit the file and the matching clean workspace as one serialized
        # publication. A newer edit that arrived while this save was parsing or
        # writing its temp file must prevent the stale os.replace side effect,
        # not merely turn the eventual HTTP response into a 409.
        with workspace_lock:
            if request_seq != state["workspace_seq"]:
                Path(tmp_name).unlink(missing_ok=True)
                raise HTTPException(409, "stale workspace update — a newer edit won")
            current_workspace = state["preview"] or {}
            same_workspace_file = bool(
                current_workspace.get("path")
                and Path(current_workspace["path"]).expanduser().resolve() == target)
            expected_identity = requested_identity
            if expected_identity is None and same_workspace_file:
                expected_identity = current_workspace.get("file_identity")
            if not force and same_workspace_file and expected_identity is not None:
                try:
                    actual_identity = _file_identity(target)
                except OSError:
                    actual_identity = None
                if not _same_file_identity(expected_identity, actual_identity):
                    Path(tmp_name).unlink(missing_ok=True)
                    conflicted = {
                        **current_workspace,
                        "observed_identity": actual_identity,
                        "observed_mtime": (
                            float(actual_identity["mtime_ns"]) / 1e9
                            if isinstance(actual_identity, dict) else 0.0),
                        "external_change": True,
                        "error": (
                            "the simulation file changed on disk; reload it, "
                            "save to a different path, or explicitly overwrite it"
                        ),
                    }
                    _persist_workspace(conflicted)
                    state["preview"] = conflicted
                    raise HTTPException(409, detail={
                        "code": "external_change",
                        "message": conflicted["error"],
                        "expected_identity": expected_identity,
                        "actual_identity": actual_identity,
                    })
            try:
                # Saving to the currently-open file is a compare-and-swap:
                # identity verification above protects an unforced replace.
                # A new/different Save As target is instead published with a
                # hard-link create, which is an atomic no-clobber operation.
                # This closes the exists-check race where another application
                # could create the selected file between a prompt and replace.
                if force or same_workspace_file:
                    os.replace(tmp_name, target)
                else:
                    try:
                        os.link(tmp_name, target)
                    except FileExistsError:
                        Path(tmp_name).unlink(missing_ok=True)
                        raise HTTPException(409, detail={
                            "code": "target_exists",
                            "message": (
                                f"{target.name} already exists; confirm overwrite "
                                "or choose a different Save As path"
                            ),
                            "path": str(target),
                        })
                    try:
                        Path(tmp_name).unlink()
                    except OSError as exc:
                        warnings.warn(
                            f"saved {target} but could not remove temporary link "
                            f"{tmp_name}: {exc}", RuntimeWarning,
                        )
                identity = _file_identity(target)
                mtime = float(identity["mtime_ns"]) / 1e9
            except HTTPException:
                raise
            except OSError as exc:
                Path(tmp_name).unlink(missing_ok=True)
                raise HTTPException(400, f"cannot save {str(target)!r}: {exc}")
            pv = {
                "path": str(target), "mtime": mtime, "observed_mtime": mtime,
                "file_identity": identity, "observed_identity": identity,
                "observed_fs_error": None,
                "sim": sim,
                "error": None, "dirty": False, "external_change": False,
                "warnings": messages,
                "starter": (state["preview"] or {}).get("starter"),
            }
            _persist_workspace(pv)
            state["preview"] = pv
        return _workspace_payload(pv)

    @app.post("/api/workspace/preflight")
    def preflight_workspace(spec: dict = Body(..., embed=True)):
        """Run the authoritative native ``phsolver validate`` gate.

        The normal workbench validator is intentionally labelled *schema
        valid*: pydantic catches the portable model constraints, while this
        endpoint resolves exact Yee snapping and engine-only limits before a
        potentially expensive run.
        """
        try:
            sim, _ = service.parse_sim_spec(spec)
        except Exception as exc:
            _validation_error(exc)
        _assert_no_stale_mode_sources(sim)
        _assert_modal_ports_ready(sim)
        try:
            from ..runners.phsolver import _solver_subprocess_env, find_solver
            solver = find_solver()
        except Exception as exc:
            raise HTTPException(424, str(exc))
        if solver is None:
            raise HTTPException(
                424, "phsolver binary not found; build the engine or set PHOTONHUB_SOLVER")
        with tempfile.TemporaryDirectory(prefix="photonhub-preflight-") as td:
            spec_path = Path(td) / "sim.json"
            spec_path.write_text(sim.to_wire_json() + "\n", encoding="utf-8")
            try:
                result = subprocess.run(
                    [str(solver), "validate", str(spec_path)],
                    capture_output=True, text=True, timeout=30,
                    env=_solver_subprocess_env(),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise HTTPException(500, f"engine preflight failed to start: {exc}")
        stdout = result.stdout.strip()
        parsed = None
        if stdout:
            try:
                parsed = json.loads(stdout.splitlines()[-1])
            except json.JSONDecodeError:
                parsed = None
        if result.returncode != 0:
            detail = (result.stderr or stdout or "phsolver rejected the spec").strip()
            raise HTTPException(422, {"message": detail, "engine": parsed})
        return {"ok": True, "engine": parsed, "solver": str(solver)}

    def _public_job(job: Optional[dict] = None):
        job = state["job"] if job is None else job
        if job is None:
            return {"status": "idle"}
        return {k: v for k, v in job.items()
                if k not in {"cancel_event", "thread", "sim"}}

    def _active_run_id() -> Optional[str]:
        with job_lock:
            job = state["job"]
            if job is not None and job["status"] in {"queued", "running"}:
                return str(job["id"])
        return None

    @app.post("/api/run")
    def start_run(payload: dict = Body(...)):
        # Cross-target starts are serialized through job publication. This is
        # intentionally held until the local worker exists so a simultaneous
        # cloud request cannot pass both endpoints' idle checks.
        with execution_lock:
            return _start_run_locked(payload)

    def _start_run_locked(payload: dict):
        with job_lock:
            cloud_current = state.get("cloud_job")
            if (state.get("cloud_submit_in_progress")
                    or (cloud_current is not None and (
                        cloud_current.get("status") in {
                            "submitted", "queued", "provisioning", "running",
                            "unknown"}
                        or cloud_current.get("download_status") in {
                            "polling", "downloading"}))):
                raise HTTPException(
                    409, "a cloud simulation is active; resume/cancel it before "
                         "starting a local run")
            current = state["job"]
            if current is not None and current["status"] in {"queued", "running"}:
                raise HTTPException(409, "a simulation is already running")
            raw_spec = payload.get("spec")
            if raw_spec is None:
                if state["preview"] is None:
                    raise HTTPException(400, "open or create a simulation first")
                sim = state["preview"]["sim"]
            else:
                try:
                    sim, _ = service.parse_sim_spec(raw_spec)
                except Exception as exc:
                    _validation_error(exc)
            _assert_no_stale_mode_sources(sim)
            _assert_modal_ports_ready(sim)

            device = str(payload.get("device") or "cpu").strip()
            try:
                from ..runners.phsolver import device_args
                device_args(device)
            except Exception as exc:
                raise HTTPException(400, str(exc))
            timeout_s = payload.get("timeout_s")
            if timeout_s in (None, "", 0):
                timeout_s = None
            else:
                try:
                    timeout_s = float(timeout_s)
                except (TypeError, ValueError):
                    raise HTTPException(400, "timeout_s must be a positive number")
                if not math.isfinite(timeout_s) or timeout_s <= 0:
                    raise HTTPException(400, "timeout_s must be a positive number")

            # The workbench names a result after the document it ran, so the
            # Design and Results headers read the same. Display text only: the
            # ledger sanitises it before it becomes a path component.
            label = payload.get("label")
            label = str(label)[:120] if isinstance(label, str) and label.strip() else None

            parent = payload.get("output_parent")
            output_parent = None
            if parent:
                output_parent = Path(str(parent)).expanduser().resolve()
                if not output_parent.is_dir():
                    raise HTTPException(400, f"output parent is not a directory: {output_parent}")

            job_id = uuid.uuid4().hex
            canonical_spec = sim.to_wire_json() + "\n"
            with workspace_lock:
                preview = state["preview"] or {}
                preview_sim = preview.get("sim")
                workspace_path = (
                    preview.get("path")
                    if (preview_sim is not None
                        and preview_sim.to_wire_json() + "\n" == canonical_spec)
                    else None
                )
            try:
                request = ledger.create_request(
                    run_id=job_id,
                    canonical_spec=canonical_spec,
                    device=device,
                    timeout_s=timeout_s,
                    solver=_solver_snapshot(),
                    estimate=service.sim_payload(sim)["estimate"],
                    workspace_path=workspace_path,
                    output_parent=output_parent,
                    label=label,
                )
            except Exception as exc:
                raise HTTPException(500, f"cannot create durable run record: {exc}")
            output_dir = Path(request["output_dir"])
            cancel_event = threading.Event()
            job = {
                "id": job_id,
                "run_id": job_id,
                "status": "queued",
                "device": device,
                "output_dir": str(output_dir) if output_dir else None,
                "log_file": str(output_dir / "solver-events.jsonl"),
                "progress": None,
                "events_seen": 0,
                "error": None,
                "session": None,
                "started_at": None,
                "finished_at": None,
                "cancel_event": cancel_event,
                "sim": sim,
            }
            state["job"] = job

            def worker():
                from ..runners.local import run_local

                try:
                    ledger.append_event(job_id, "running")
                except Exception as exc:
                    job["error"] = f"durable run ledger could not record start: {exc}"
                    try:
                        ledger.seal(job_id, "failed", error={
                            "type": type(exc).__name__, "message": str(exc),
                            "phase": "ledger_start",
                        })
                    except Exception as seal_exc:
                        job["error"] += f"; terminal sealing also failed: {seal_exc}"
                    job["finished_at"] = time.time()
                    job["status"] = "failed"
                    return

                job["started_at"] = time.time()
                job["status"] = "running"

                def on_progress(event: dict):
                    job["events_seen"] += 1
                    job["progress"] = event

                sealed_completed = False
                try:
                    run_local(
                        sim, output_dir=output_dir, device=device,
                        timeout=timeout_s, progress=on_progress, quiet=True,
                        log_file=output_dir / "solver-events.jsonl",
                        cancel_event=cancel_event,
                    )
                    # Hash every declared result artifact before either the ledger
                    # or the UI is allowed to call this run completed.
                    job["progress"] = {
                        **(job.get("progress") or {}),
                        "phase": "finalizing checksums",
                    }
                    sealed = ledger.seal(job_id, "completed")
                    sealed_completed = True
                    # Publish only a fresh reader bound to the exact manifest
                    # bytes captured by the terminal seal. The solver-returned
                    # object may predate a last-moment manifest replacement.
                    data = service.load_result(output_dir)
                    _bind_sealed_manifest(sealed, data)
                    ledger.verify_bundle_metadata(job_id)
                    result_id = uuid.uuid4().hex
                    result_session = service.session(
                        data, result_id, run_id=job_id)
                except Exception as exc:  # surfaced verbatim to the local-only UI
                    if sealed_completed:
                        # The immutable lifecycle is authoritative once its
                        # completed terminal commits. Activation can fail (for
                        # example, immediate external tampering), but must not
                        # contradict that durable state in /api/run/status.
                        job["error"] = (
                            "completed result was sealed but could not be "
                            f"activated: {exc}")
                        terminal_status = "completed"
                    else:
                        job["error"] = str(exc)
                        terminal_status = (
                            "cancelled" if cancel_event.is_set() else "failed")
                        try:
                            ledger.seal(job_id, terminal_status, error={
                                "type": type(exc).__name__, "message": str(exc),
                            })
                        except Exception as seal_exc:
                            job["error"] += (
                                f"; durable terminal sealing failed: {seal_exc}")
                    job["finished_at"] = time.time()
                    # Terminal status is the publish flag; assign it last so a
                    # poll can never stop on a half-populated terminal job.
                    job["status"] = terminal_status
                else:
                    # Publish the result pointer atomically and advance the
                    # selection sequence so an already-loading older Open
                    # request cannot replace this later completion.
                    with result_lock:
                        state["result_seq"] += 1
                        state["result"] = (data, result_id, job_id)
                    job["output_dir"] = str(data.output_dir)
                    job["session"] = result_session
                    job["finished_at"] = time.time()
                    job["status"] = "completed"

            thread = threading.Thread(target=worker, name=f"photonhub-run-{job_id[:8]}", daemon=True)
            job["thread"] = thread
            thread.start()
            return _public_job(job)

    @app.get("/api/run/status")
    def run_status():
        return _public_job()

    @app.post("/api/run/cancel")
    def cancel_run():
        with job_lock:
            job = state["job"]
            if job is None or job["status"] not in {"queued", "running"}:
                raise HTTPException(409, "no simulation is running")
            job["cancel_event"].set()
            return _public_job(job)

    # --- Workbench notebook: scriptable setup/run/analysis over the same ----
    # authoritative workspace and run machinery the GUI uses.  Cells are user
    # Python executed locally (the user's own machine, their own privileges);
    # the endpoints sit under /api and are therefore capability-protected in
    # packaged desktop launches.
    def _hook_detail(exc) -> str:
        detail = getattr(exc, "detail", None)
        if isinstance(detail, dict):
            return str(detail.get("message") or detail)
        return str(detail if detail is not None else exc)

    def _notebook_get_spec() -> dict:
        with workspace_lock:
            pv = state["preview"]
            if pv is None:
                raise LookupError(
                    "no simulation workspace open — create or open a setup in "
                    "Design first")
            return pv["sim"].to_wire_dict()

    def _notebook_apply_spec(spec: dict) -> list:
        request_seq = _begin_workspace_update()
        # pydantic errors propagate verbatim into the cell traceback — they
        # are the same authoritative messages the GUI validation shows.
        sim, messages = service.parse_sim_spec(spec)
        try:
            _update_workspace_sim(
                sim, request_seq=request_seq, dirty=True, warnings_=messages)
        except HTTPException as exc:
            raise RuntimeError(_hook_detail(exc)) from exc
        return messages

    def _notebook_run_local(*, spec, device, timeout_s, on_progress):
        payload = {"device": device, "timeout_s": timeout_s}
        if spec is not None:
            payload["spec"] = spec
        try:
            with execution_lock:
                job = _start_run_locked(payload)
        except HTTPException as exc:
            raise RuntimeError(_hook_detail(exc)) from exc
        job_id = job["id"]
        while True:
            with job_lock:
                current = state["job"]
                if current is None or current.get("id") != job_id:
                    raise RuntimeError(
                        "the run was superseded by another job")
                status = current["status"]
                progress = current.get("progress")
                error = current.get("error")
                session = current.get("session")
            if on_progress is not None:
                on_progress(progress, status)
            if status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.25)
        if status != "completed" or session is None:
            raise RuntimeError(error or f"simulation run {status}")
        data, result_id, run_id = state["result"]
        if data is None or result_id != session.get("result_id"):
            raise RuntimeError(
                "the completed result is no longer the open result")
        return data, session

    def _notebook_cancel_active_run() -> None:
        with job_lock:
            job = state["job"]
            if job is not None and job["status"] in {"queued", "running"}:
                job["cancel_event"].set()

    def _notebook_get_result():
        data, result_id, run_id = state["result"]
        if data is None:
            return None
        return data, service.session(data, result_id, run_id=run_id)

    notebook_kernel = NotebookKernel(
        WorkbenchHooks(
            get_spec=_notebook_get_spec,
            apply_spec=_notebook_apply_spec,
            run_local=_notebook_run_local,
            cancel_active_run=_notebook_cancel_active_run,
            get_result=_notebook_get_result,
        ),
        persist_path=ledger.root / "workbench-notebook.json",
    )

    def _notebook_op(operation):
        try:
            return operation()
        except LookupError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/notebook")
    def notebook_state():
        return notebook_kernel.snapshot()

    @app.post("/api/notebook/cells")
    def notebook_add_cell(payload: dict = Body(default={})):
        after_id = payload.get("after_id")
        if after_id is not None and not isinstance(after_id, str):
            raise HTTPException(422, "after_id must be a cell id string")
        code = payload.get("code", "")
        if not isinstance(code, str):
            raise HTTPException(422, "code must be a string")
        return _notebook_op(
            lambda: notebook_kernel.add_cell(after_id=after_id, code=code))

    @app.post("/api/notebook/cells/{cell_id}")
    def notebook_update_cell(cell_id: str, payload: dict = Body(...)):
        code = payload.get("code")
        if not isinstance(code, str):
            raise HTTPException(422, "code must be a string")
        return _notebook_op(
            lambda: notebook_kernel.update_cell(cell_id, code))

    @app.post("/api/notebook/cells/{cell_id}/delete")
    def notebook_delete_cell(cell_id: str):
        return _notebook_op(lambda: notebook_kernel.delete_cell(cell_id))

    @app.post("/api/notebook/cells/{cell_id}/move")
    def notebook_move_cell(cell_id: str, payload: dict = Body(...)):
        index = payload.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise HTTPException(422, "index must be an integer")
        return _notebook_op(lambda: notebook_kernel.move_cell(cell_id, index))

    @app.post("/api/notebook/cells/{cell_id}/run")
    def notebook_run_cell(cell_id: str, payload: dict = Body(default={})):
        code = payload.get("code")
        if code is not None and not isinstance(code, str):
            raise HTTPException(422, "code must be a string")
        return _notebook_op(lambda: notebook_kernel.run_cell(cell_id, code))

    @app.post("/api/notebook/run-all")
    def notebook_run_all():
        return notebook_kernel.run_all()

    @app.post("/api/notebook/interrupt")
    def notebook_interrupt():
        return notebook_kernel.interrupt()

    @app.post("/api/notebook/restart")
    def notebook_restart(payload: dict = Body(default={})):
        return notebook_kernel.restart(
            clear_outputs=bool(payload.get("clear_outputs", False)))

    # --- Metered cloud GPU: explicit quote acceptance + resumable download --
    def _cloud_amount(payload: object, stem: str) -> Optional[float]:
        if not isinstance(payload, dict):
            return None
        value = payload.get(f"{stem}_usd")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) >= 0):
            return float(value)
        micros = payload.get(f"{stem}_micros")
        if isinstance(micros, int) and not isinstance(micros, bool) and micros >= 0:
            return micros / 1_000_000
        return None

    def _cloud_account(payload: object) -> dict:
        return {
            "balance_usd": _cloud_amount(payload, "balance"),
            "available_usd": _cloud_amount(payload, "available"),
            "reserved_usd": _cloud_amount(payload, "reserved"),
        }

    def _cloud_job_resolved(job: dict) -> bool:
        """A succeeded job with its downloaded result session bound is settled
        local evidence: the sealed ledger run owns it from here, so a restart
        must boot idle instead of asking to resume this job forever."""
        return (job.get("status") == "succeeded"
                and job.get("session") is not None)

    def _discard_cloud_recovery(job: Optional[dict] = None) -> None:
        try:
            cloud_recovery_path.unlink(missing_ok=True)
        except OSError as exc:
            if job is not None:
                job["recovery_error"] = (
                    f"could not clear cloud job recovery: {exc}")
        else:
            if job is not None:
                job["recovery_error"] = None

    def _resolve_cloud_recovery_for_run(run_id: str) -> None:
        """Opening a run's sealed result locally is the recovery's goal state.

        Only a restored, watcher-less record pointing at this exact run is
        cleared; live jobs resolve themselves on completion, and unresolved
        submissions keep their duplicate-charge guard.
        """
        with cloud_lock:
            job = state.get("cloud_job")
            if (job is None or job.get("thread") is not None
                    or job.get("session") is not None
                    or job.get("unresolved_submission")
                    or job.get("download_status") in {"polling", "downloading"}):
                return
            ledger_run_id = job.get("ledger_run_id")
            if (not isinstance(ledger_run_id, str)
                    or ledger_run_id.lower() != run_id):
                return
            state["cloud_job"] = None
            _discard_cloud_recovery()

    def _persist_cloud_job(job: dict) -> None:
        """Best-effort, secret-free recovery of an accepted service job id."""
        if _cloud_job_resolved(job):
            _discard_cloud_recovery(job)
            return
        payload = {
            "version": _CLOUD_RECOVERY_VERSION,
            **{key: job.get(key) for key in (
                "id", "status", "device", "quote_usd", "actual_usd",
                "refunded_usd", "max_usd", "available_usd", "remaining_usd",
                "quote_expires_at", "output_dir", "submitted_at", "finished_at",
                "cancel_requested", "name", "unresolved_submission",
                "ledger_run_id",
            )},
        }
        sim = job.get("sim")
        if sim is not None:
            payload["spec"] = sim.to_wire_dict()
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=ledger.root,
                prefix=".cloud-job-recovery.", suffix=".tmp", delete=False,
            ) as tmp:
                json.dump(
                    payload, tmp, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False,
                )
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            os.replace(tmp_name, cloud_recovery_path)
            job["recovery_error"] = None
        except Exception as exc:
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
            # Never turn an already accepted paid submission into an HTTP
            # failure that might tempt a retry.  Keep the service id visible.
            job["recovery_error"] = f"could not persist cloud job recovery: {exc}"

    def _public_cloud_job(job: Optional[dict] = None) -> dict:
        job = state["cloud_job"] if job is None else job
        if job is None:
            return {"status": "idle", "download_status": "idle"}
        public = {key: job.get(key) for key in (
            "id", "status", "device", "quote_usd", "actual_usd",
            "refunded_usd", "max_usd", "available_usd", "remaining_usd",
            "quote_expires_at", "progress", "error", "poll_error",
            "download_status", "session", "output_dir", "submitted_at",
            "finished_at", "cancel_requested", "recovery_error", "name",
            "unresolved_submission",
            "ledger_run_id",
        )}
        public["resumable"] = bool(
            job.get("id") and not job.get("session")
            and job.get("status") not in {"failed", "cancelled"}
            and job.get("download_status") not in {"polling", "downloading"}
        )
        return public

    def _public_cloud_history(record: object) -> Optional[dict]:
        if not isinstance(record, dict):
            return None
        job_id = record.get("job_id", record.get("id"))
        if not isinstance(job_id, str) or not job_id:
            return None
        error = record.get("error")
        if isinstance(error, dict):
            error = error.get("reason") or error.get("message")
        if error is not None and not isinstance(error, str):
            error = str(error)
        return {
            "id": job_id,
            "name": record.get("name") if isinstance(record.get("name"), str) else None,
            "status": record.get("state", record.get("status", "unknown")),
            "device": record.get("device"),
            "progress": record.get("progress") if isinstance(record.get("progress"), dict) else None,
            "quote_usd": _cloud_amount(record, "quote"),
            "actual_usd": _cloud_amount(record, "actual"),
            "refunded_usd": _cloud_amount(record, "refunded"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "finished_at": record.get("finished_at"),
            "error": error,
        }

    def _apply_cloud_remote(job: dict, remote: object) -> None:
        if not isinstance(remote, dict):
            return
        remote_status = remote.get("state", remote.get("status"))
        if remote_status == "completed":
            remote_status = "succeeded"
        if remote_status in {
                "submitted", "queued", "provisioning", "running",
                "succeeded", "failed", "cancelled", "unknown"}:
            # A locally opened result is stronger than a stale remote poll.
            if job.get("session") is None:
                job["status"] = remote_status
        for stem in ("quote", "actual", "refunded"):
            amount = _cloud_amount(remote, stem)
            if amount is not None:
                job[f"{stem}_usd"] = amount
        if isinstance(remote.get("progress"), dict):
            job["progress"] = remote["progress"]
        if remote_status == "succeeded" and job.get("download_status") == "polling":
            job["download_status"] = "downloading"
        if remote_status in {"failed", "cancelled"}:
            job["finished_at"] = job.get("finished_at") or time.time()

    def _cloud_error_status(exc: Exception, *, spend_check: bool = False) -> int:
        message = str(exc).lower()
        if spend_check and any(word in message for word in (
                "exceeds max", "exceeds available", "available balance")):
            return 409
        status = getattr(exc, "status_code", None)
        if status in {400, 404, 409, 422}:
            return int(status)
        return 424

    def _cloud_sim(raw_spec: object):
        if raw_spec is None:
            raise HTTPException(400, "cloud execution requires a simulation spec")
        try:
            sim, _ = service.parse_sim_spec(raw_spec)
        except Exception as exc:
            _validation_error(exc)
        _assert_no_stale_mode_sources(sim)
        _assert_modal_ports_ready(sim)
        canonical = sim.to_wire_json()
        return sim, hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cloud_limit(raw: object) -> float:
        if isinstance(raw, bool):
            raise HTTPException(400, "max_usd must be a finite number from 0 to 5")
        try:
            limit = float(raw)
        except (TypeError, ValueError):
            raise HTTPException(400, "max_usd must be a finite number from 0 to 5")
        if (not math.isfinite(limit) or limit < 0
                or limit > _WORKBENCH_CLOUD_MAX_USD):
            raise HTTPException(
                400, "Workbench beta max_usd must be between $0 and $5.00")
        return limit

    def _quote_expired(accepted) -> bool:
        raw = getattr(accepted, "quote", {}).get("expires_at")
        if not isinstance(raw, str) or not raw:
            return False
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed <= datetime.now(timezone.utc)
        except ValueError:
            # The service remains authoritative when an older deployment omits
            # or changes this optional display field.
            return False

    def _public_cloud_preflight(token: str, accepted) -> dict:
        quote = getattr(accepted, "quote", {})
        out = {
            "token": token,
            "device": accepted.device,
            "max_usd": accepted.max_usd,
            "quote_usd": accepted.quote_usd,
            "available_usd": accepted.available_usd,
            "remaining_usd": accepted.remaining_usd,
        }
        # Explicit whitelist: never serialize the opaque accepted quote id.
        if isinstance(quote, dict):
            for key in (
                    "expires_at", "num_cells", "num_steps", "tcell_steps",
                    "rate_usd_per_tcell_step", "device_memory_bytes",
                    "output_bytes"):
                value = quote.get(key)
                if value is not None:
                    out[key] = value
        return out

    def _copy_cloud_artifact(source_root: Path, target_root: Path,
                             name: str) -> None:
        """Copy one stable regular file into a private ledger bundle."""
        if (not isinstance(name, str) or not name or name in {".", ".."}
                or "/" in name or "\\" in name or Path(name).name != name):
            raise LedgerError(f"unsafe cloud result artifact name: {name!r}")
        source = source_root / name
        try:
            before = source.lstat()
        except OSError as exc:
            raise LedgerError(f"cannot stat cloud artifact {name!r}: {exc}") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise LedgerError(f"cloud artifact {name!r} is not a regular file")
        flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
        tmp_name = None
        try:
            fd = os.open(source, flags)
            with os.fdopen(fd, "rb") as src, tempfile.NamedTemporaryFile(
                "wb", dir=target_root, prefix=f".{name}.", suffix=".tmp",
                delete=False,
            ) as dst:
                tmp_name = dst.name
                opened = os.fstat(src.fileno())
                if _stat_signature(opened) != _stat_signature(before):
                    raise LedgerError(f"cloud artifact {name!r} changed while opening")
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
                after_fd = os.fstat(src.fileno())
            after_path = source.lstat()
            if (_stat_signature(after_fd) != _stat_signature(before)
                    or _stat_signature(after_path) != _stat_signature(before)):
                raise LedgerError(f"cloud artifact {name!r} changed while copying")
            os.replace(tmp_name, target_root / name)
            tmp_name = None
        except (LedgerError, OSError) as exc:
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
            if isinstance(exc, LedgerError):
                raise
            raise LedgerError(f"cannot archive cloud artifact {name!r}: {exc}") from exc

    def _read_cloud_artifact(source_root: Path, name: str, *,
                             max_bytes: int) -> bytes:
        """Read one bounded, stable, no-follow cloud artifact snapshot."""
        if (not isinstance(name, str) or not name or name in {".", ".."}
                or "/" in name or "\\" in name or Path(name).name != name):
            raise LedgerError(f"unsafe cloud result artifact name: {name!r}")
        source = source_root / name
        try:
            before = source.lstat()
        except OSError as exc:
            raise LedgerError(f"cannot stat cloud artifact {name!r}: {exc}") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise LedgerError(f"cloud artifact {name!r} is not a regular file")
        if before.st_size > max_bytes:
            raise LedgerError(
                f"cloud artifact {name!r} exceeds the {max_bytes}-byte limit")
        flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
        try:
            fd = os.open(source, flags)
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if _stat_signature(opened) != _stat_signature(before):
                    raise LedgerError(f"cloud artifact {name!r} changed while opening")
                raw = stream.read(max_bytes + 1)
                after_fd = os.fstat(stream.fileno())
            after_path = source.lstat()
        except (LedgerError, OSError) as exc:
            if isinstance(exc, LedgerError):
                raise
            raise LedgerError(f"cannot read cloud artifact {name!r}: {exc}") from exc
        if len(raw) > max_bytes:
            raise LedgerError(
                f"cloud artifact {name!r} exceeds the {max_bytes}-byte limit")
        if (_stat_signature(after_fd) != _stat_signature(before)
                or _stat_signature(after_path) != _stat_signature(before)
                or len(raw) != int(before.st_size)):
            raise LedgerError(f"cloud artifact {name!r} changed while reading")
        return raw

    def _write_cloud_archive_bytes(target_root: Path, name: str,
                                   value: bytes) -> None:
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=target_root, prefix=f".{name}.", suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(value)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            os.replace(tmp_name, target_root / name)
        except OSError as exc:
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
            raise LedgerError(f"cannot write archived {name}: {exc}") from exc

    def _write_cloud_archive_text(target_root: Path, name: str, value: str) -> None:
        try:
            raw = value.encode("utf-8")
        except UnicodeError as exc:
            raise LedgerError(f"cannot encode archived {name}: {exc}") from exc
        _write_cloud_archive_bytes(target_root, name, raw)

    def _archive_cloud_result(job: dict, downloaded):
        """Copy, checksum-seal, and reopen a paid result before publication."""
        request = None
        existing_id = job.get("ledger_run_id")
        if isinstance(existing_id, str):
            record = ledger.get_run(existing_id)
            recorded_status = record.get("recorded_status")
            if recorded_status == "completed":
                ledger.verify_bundle_metadata(existing_id)
                archived = service.load_result(record["output_dir"])
                _bind_sealed_manifest(record, archived)
                ledger.verify_bundle_metadata(existing_id)
                return archived, existing_id
            if recorded_status in {"queued", "running"}:
                # Recovery may land here after a crash anywhere between request
                # creation, copying, and the completed terminal commit. Reuse
                # the same durable identity and overwrite only its unsealed
                # private bundle files.
                request = record
            else:
                job["ledger_run_id"] = None

        sim = job.get("sim")
        if sim is None:
            raise LedgerError(
                "the original cloud simulation is unavailable; Workbench cannot "
                "seal this result's input provenance")
        source_root = Path(downloaded.output_dir).expanduser().absolute()
        try:
            resolved_source = source_root.resolve(strict=True)
            source_stat = source_root.lstat()
        except OSError as exc:
            raise LedgerError(f"cloud result directory is unavailable: {exc}") from exc
        if (resolved_source != source_root or stat.S_ISLNK(source_stat.st_mode)
                or not stat.S_ISDIR(source_stat.st_mode)):
            raise LedgerError("cloud result path is not a trusted regular directory")
        manifest = getattr(downloaded, "manifest", None)
        if not isinstance(manifest, dict):
            raise LedgerError("cloud result has no parsed manifest")
        source_spec = source_root / "sim.json"
        try:
            source_spec.lstat()
        except FileNotFoundError:
            # Current Hot Aisle bundles predate carrying sim.json, but the
            # coordinator contract is deterministic: it stores and executes
            # these exact no-newline bytes.  The digest check below makes this
            # compatibility reconstruction fail closed if that contract drifts.
            canonical_spec_bytes = sim.to_wire_json(indent=0).encode("utf-8")
        except OSError as exc:
            raise LedgerError(f"cannot inspect cloud execution input: {exc}") from exc
        else:
            # New executors bundle the exact bytes phsolver hashed. Prefer that
            # evidence over reconstructing a provider-specific serialization,
            # and independently require semantic equality with the submitted
            # immutable Simulation.
            canonical_spec_bytes = _read_cloud_artifact(
                source_root, "sim.json", max_bytes=_MAX_CLOUD_SPEC_BYTES)
            try:
                executed_sim = type(sim).from_wire_json(canonical_spec_bytes)
            except Exception as exc:
                raise LedgerError(
                    f"cloud result executed sim.json is invalid: {exc}") from exc
            if executed_sim.to_wire_dict() != sim.to_wire_dict():
                raise LedgerError(
                    "cloud result executed sim.json differs from the submitted "
                    "immutable simulation")
        try:
            canonical_spec = canonical_spec_bytes.decode("utf-8")
        except UnicodeError as exc:
            raise LedgerError(
                f"cloud result executed sim.json is not UTF-8: {exc}") from exc
        provenance = manifest.get("provenance")
        input_sha = (provenance.get("input_sha256")
                     if isinstance(provenance, dict) else None)
        reconstructed_sha = hashlib.sha256(canonical_spec_bytes).hexdigest()
        if (not isinstance(input_sha, str)
                or input_sha.lower() != reconstructed_sha):
            raise LedgerError(
                "cloud result input provenance does not match the deterministic "
                "service execution request")
        monitors = manifest.get("monitors")
        if not isinstance(monitors, list):
            raise LedgerError("cloud result manifest monitors is not a list")
        artifact_names = {"manifest.json"}
        for monitor in monitors:
            if not isinstance(monitor, dict):
                raise LedgerError("cloud result manifest has a non-object monitor")
            filename = monitor.get("file")
            if not isinstance(filename, str):
                raise LedgerError("cloud result monitor has no artifact filename")
            artifact_names.add(filename)

        with workspace_lock:
            preview = state.get("preview") or {}
            preview_sim = preview.get("sim")
            workspace_path = (
                preview.get("path")
                if (preview_sim is not None
                    and preview_sim.to_wire_dict() == sim.to_wire_dict())
                else None
            )
        if request is None:
            request = ledger.create_request(
                run_id=None,
                canonical_spec=canonical_spec,
                device=f"cloud:{job.get('device') or 'gpu'}",
                timeout_s=None,
                solver={
                    "available": True,
                    "path": "PhotonHub cloud service",
                    "info": {
                        "job_id": job.get("id"),
                        "quote_usd": job.get("quote_usd"),
                        "actual_usd": job.get("actual_usd"),
                        "refunded_usd": job.get("refunded_usd"),
                    },
                    "capabilities": {},
                },
                estimate=service.sim_payload(sim)["estimate"],
                workspace_path=workspace_path,
                # Cloud results are named after their document too, when the
                # run is still traceable to a saved workspace.
                label=Path(workspace_path).name if workspace_path else None,
            )
        run_id = request["run_id"]
        target_root = Path(request["output_dir"])
        sealed_completed = False
        try:
            # The cloud-job -> ledger identity is durable before any completed
            # terminal can commit. A restart therefore reuses, rather than
            # orphaning and duplicating, the local evidence record.
            with cloud_lock:
                if state.get("cloud_job") is not job:
                    raise LedgerError("cloud job changed before archive publication")
                job["ledger_run_id"] = run_id
                _persist_cloud_job(job)
                if job.get("recovery_error"):
                    raise LedgerError(job["recovery_error"])
            if request.get("recorded_status") == "queued":
                ledger.append_event(run_id, "running", detail={
                    "cloud_job_id": job.get("id"), "phase": "archiving download",
                })
            for name in sorted(artifact_names):
                _copy_cloud_artifact(source_root, target_root, name)
            _write_cloud_archive_bytes(
                target_root, "sim.json", canonical_spec_bytes)
            _write_cloud_archive_text(
                target_root, "solver-events.jsonl",
                json.dumps({
                    "event": "cloud_result_archived",
                    "job_id": job.get("id"),
                    "at": datetime.now(timezone.utc).isoformat(),
                }, sort_keys=True, separators=(",", ":")) + "\n",
            )
            # Parse before committing an immutable completed terminal. Sealing
            # then binds every exact byte that this fresh reader will expose.
            service.load_result(target_root)  # parse gate before immutable terminal
            sealed = ledger.seal(run_id, "completed")
            sealed_completed = True
            archived = service.load_result(target_root)
            _bind_sealed_manifest(sealed, archived)
            ledger.verify_bundle_metadata(run_id)
            return archived, run_id
        except Exception as exc:
            if not sealed_completed:
                try:
                    ledger.seal(run_id, "failed", error={
                        "type": type(exc).__name__, "message": str(exc),
                        "phase": "cloud_archive",
                    })
                except Exception:
                    pass
            raise

    def _watch_cloud_job(job: dict, handle) -> None:
        try:
            downloaded = handle.result()
            data, ledger_run_id = _archive_cloud_result(job, downloaded)
            result_id = uuid.uuid4().hex
            result_session = service.session(
                data, result_id, run_id=ledger_run_id)
            with cloud_lock:
                if state.get("cloud_job") is not job:
                    return
                job["ledger_run_id"] = ledger_run_id
                job["output_dir"] = str(data.output_dir)
                _persist_cloud_job(job)
        except Exception as exc:
            with cloud_lock:
                if state.get("cloud_job") is not job:
                    return
                job["download_status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = time.time()
            try:
                from .. import web as cloud_web
                remote = cloud_web.job_status(str(job["id"]))
            except Exception as poll_exc:
                with cloud_lock:
                    if state.get("cloud_job") is job:
                        job["status"] = (
                            job.get("status")
                            if job.get("status") in {"failed", "cancelled", "succeeded"}
                            else "unknown")
                        job["poll_error"] = str(poll_exc)
                        _persist_cloud_job(job)
            else:
                with cloud_lock:
                    if state.get("cloud_job") is job:
                        _apply_cloud_remote(job, remote)
                        job["poll_error"] = None
                        _persist_cloud_job(job)
            return

        # A watcher superseded by an explicit Resume must never publish its
        # stale result over the newer user-selected job.
        with cloud_lock:
            if state.get("cloud_job") is not job:
                return
        try:
            from .. import web as cloud_web
            final_remote = cloud_web.job_status(str(job["id"]))
        except Exception as exc:
            final_remote = None
            final_poll_error = str(exc)
        else:
            final_poll_error = None

        with result_lock:
            state["result_seq"] += 1
            state["result"] = (data, result_id, ledger_run_id)
        with cloud_lock:
            if state.get("cloud_job") is not job:
                return
            if final_remote is not None:
                _apply_cloud_remote(job, final_remote)
            job["status"] = "succeeded"
            job["download_status"] = "completed"
            job["session"] = result_session
            job["output_dir"] = str(data.output_dir)
            job["error"] = None
            job["poll_error"] = final_poll_error
            job["finished_at"] = time.time()
            _persist_cloud_job(job)

    def _start_cloud_watcher(job: dict, handle) -> None:
        thread = threading.Thread(
            target=_watch_cloud_job, args=(job, handle),
            name=f"photonhub-cloud-{str(job['id'])[:8]}", daemon=True,
        )
        job["handle"] = handle
        job["thread"] = thread
        thread.start()

    @app.get("/api/cloud/status")
    def cloud_status():
        from .. import web as cloud_web

        try:
            cloud_web.get_config()
        except Exception as exc:
            return {
                "configured": False, "reachable": False, "account": None,
                "gpus": [], "gpu_menu_available": False, "warning": None,
                "error": str(exc), "max_usd": _WORKBENCH_CLOUD_MAX_USD,
            }
        try:
            account = cloud_web.account()
        except Exception as exc:
            return {
                "configured": True, "reachable": False, "account": None,
                "gpus": [], "gpu_menu_available": False, "warning": None,
                "error": str(exc), "max_usd": _WORKBENCH_CLOUD_MAX_USD,
            }

        warning = None
        gpu_menu_available = True
        try:
            raw_gpus = cloud_web.gpus()
            gpus = [item for item in raw_gpus if isinstance(item, dict)]
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                gpus = []
                gpu_menu_available = False
                warning = (
                    "This service has no approved /v1/gpus menu. Cloud GPU is "
                    "disabled until the operator promotes an explicit device.")
            else:
                gpus = []
                gpu_menu_available = False
                warning = f"GPU menu unavailable: {exc}"
        public_account = _cloud_account(account)
        if public_account["available_usd"] is None:
            warning = (warning + " " if warning else "") + (
                "Account response has no usable available balance; paid "
                "preflight will fail closed.")
        return {
            "configured": True, "reachable": True,
            "account": public_account, "gpus": gpus,
            "gpu_menu_available": gpu_menu_available, "warning": warning,
            "error": None, "max_usd": _WORKBENCH_CLOUD_MAX_USD,
        }

    @app.post("/api/cloud/preflight")
    def cloud_preflight(payload: dict = Body(...)):
        from .. import web as cloud_web

        sim, canonical_sha = _cloud_sim(payload.get("spec"))
        limit = _cloud_limit(payload.get("max_usd", _WORKBENCH_CLOUD_MAX_USD))
        device = str(payload.get("device") or "").strip()
        if not device.startswith("gpu:") or len(device) <= 4:
            raise HTTPException(
                400,
                "Workbench cloud execution requires an explicitly approved "
                "gpu:<id> device",
            )
        try:
            solver_ref = _workbench_cloud_solver(release_identity)
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        try:
            accepted = cloud_web.preflight(
                sim, device=device, solver=solver_ref, max_usd=limit,
            )
        except Exception as exc:
            raise HTTPException(
                _cloud_error_status(exc, spend_check=True), str(exc))

        token = uuid.uuid4().hex
        with cloud_lock:
            preflights = state["cloud_preflights"]
            # Bound this in-memory, secret-bearing set.  Consumed records are
            # discarded first; the oldest unused record is invalidated if the
            # user keeps requesting fresh quotes without accepting them.
            while len(preflights) >= 32:
                oldest = next(iter(preflights))
                preflights.pop(oldest, None)
            preflights[token] = {
                "accepted": accepted, "canonical_sha": canonical_sha,
                "consumed": False, "created_at": time.time(),
            }
        return _public_cloud_preflight(token, accepted)

    @app.post("/api/cloud/run")
    def start_cloud_run(payload: dict = Body(...)):
        # Serialize the paid POST through durable service-id publication. The
        # lock also closes the local-vs-cloud simultaneous-start race.
        with execution_lock:
            return _start_cloud_run_locked(payload)

    def _start_cloud_run_locked(payload: dict):
        from .. import web as cloud_web

        sim, canonical_sha = _cloud_sim(payload.get("spec"))
        token = payload.get("preflight_token")
        if not isinstance(token, str) or not token:
            raise HTTPException(400, "preflight_token is required")
        requested_name = payload.get("name")
        if requested_name is None:
            requested_name = f"workbench-{uuid.uuid4().hex[:12]}"
        if (not isinstance(requested_name, str) or not requested_name.strip()
                or len(requested_name) > 128):
            raise HTTPException(400, "cloud job name must be 1 to 128 characters")
        requested_name = requested_name.strip()

        with job_lock:
            local_current = state.get("job")
            if local_current is not None and local_current.get("status") in {
                    "queued", "running"}:
                raise HTTPException(
                    409, "a local simulation is active; stop it before "
                         "submitting a cloud run")

        with cloud_lock:
            if state["cloud_submit_in_progress"]:
                raise HTTPException(409, "a cloud submission is already in progress")
            current = state.get("cloud_job")
            if current is not None and (
                    current.get("status") in {
                        "submitted", "queued", "provisioning", "running", "unknown"
                    }
                    or current.get("download_status") in {"polling", "downloading"}
                    or (
                        current.get("id") and current.get("session") is None
                        and current.get("status") not in {"failed", "cancelled"}
                    )):
                raise HTTPException(
                    409, f"cloud job {current.get('id')} is still active; "
                         "resume or cancel it before submitting another")
            record = state["cloud_preflights"].get(token)
            if record is None:
                raise HTTPException(409, "unknown or expired cloud preflight token")
            if record["consumed"]:
                raise HTTPException(409, "this cloud quote was already accepted")
            if record["canonical_sha"] != canonical_sha:
                raise HTTPException(
                    409, "simulation changed after cloud preflight; request a new quote")
            accepted = record["accepted"]
            if _quote_expired(accepted):
                record["consumed"] = True
                raise HTTPException(409, "cloud quote expired; request a new quote")
            # Consume before any network write.  If the POST outcome is
            # ambiguous, retrying this token could create a duplicate paid job.
            record["consumed"] = True
            state["cloud_submit_in_progress"] = True

        cloud_job = {
            "id": None, "name": requested_name, "status": "submitted",
            "device": accepted.device, "quote_usd": accepted.quote_usd,
            "actual_usd": None, "refunded_usd": None,
            "max_usd": accepted.max_usd,
            "available_usd": accepted.available_usd,
            "remaining_usd": accepted.remaining_usd,
            "quote_expires_at": accepted.quote.get("expires_at"),
            "progress": None, "error": None, "poll_error": None,
            "download_status": "polling", "session": None, "output_dir": None,
            "submitted_at": time.time(), "finished_at": None,
            "cancel_requested": False, "handle": None, "thread": None,
            # Persist this intent before crossing the paid service boundary.
            # If the process dies during POST, recovery blocks another submit
            # and gives the generated name needed to reconcile via history.
            "unresolved_submission": True,
            "ledger_run_id": None,
            "sim": sim,
        }

        with cloud_lock:
            state["cloud_job"] = cloud_job
            _persist_cloud_job(cloud_job)
            if cloud_job.get("recovery_error"):
                state["cloud_submit_in_progress"] = False
                cloud_job["status"] = "failed"
                cloud_job["download_status"] = "failed"
                cloud_job["unresolved_submission"] = False
                cloud_job["error"] = (
                    "Cloud submission was not attempted because Workbench could "
                    "not persist its duplicate-charge guard. Request a new quote.")
                raise HTTPException(500, cloud_job["error"])

        def on_progress(event: dict):
            if not isinstance(event, dict):
                return
            with cloud_lock:
                cloud_job["progress"] = event
                if cloud_job.get("status") in {
                    "submitted", "queued", "provisioning"
                }:
                    cloud_job["status"] = "running"

        try:
            handle = cloud_web.run_async(
                sim, name=cloud_job["name"], device=accepted.device,
                solver=accepted.solver, quote_id=accepted.quote_id,
                progress=on_progress,
            )
            job_id = getattr(handle, "job_id", None)
            if not isinstance(job_id, str) or not job_id:
                raise cloud_web.WebError("cloud submission returned no usable job id")
            cloud_job["id"] = job_id
            cloud_job["unresolved_submission"] = False
        except Exception as exc:
            ambiguous_id = getattr(exc, "job_id", None)
            with cloud_lock:
                state["cloud_submit_in_progress"] = False
                cloud_job.update({
                    "id": ambiguous_id if isinstance(ambiguous_id, str) and ambiguous_id else None,
                    "status": "unknown", "download_status": "failed",
                    "error": (
                        str(exc) if isinstance(ambiguous_id, str) and ambiguous_id
                        else f"Submit outcome unknown for {requested_name!r}. "
                             "Refresh cloud history and attach that exact name if "
                             "present. If absent, do not submit again; contact the "
                             f"beta operator. Service error: {exc}"
                    ),
                    "finished_at": time.time(),
                    "unresolved_submission": not (
                        isinstance(ambiguous_id, str) and bool(ambiguous_id)),
                })
                # Even without a returned id the request may have crossed the
                # service boundary. Persist the generated name and block a new
                # paid submit until the tester resolves it through history.
                state["cloud_job"] = cloud_job
                _persist_cloud_job(cloud_job)
            detail = (
                f"cloud submit outcome is unknown for job {ambiguous_id}; "
                "resume that id instead of submitting again"
                if isinstance(ambiguous_id, str) and ambiguous_id else
                f"cloud submit outcome is unknown and new submissions are blocked. "
                f"Check cloud history for {requested_name!r}: {exc}"
            )
            raise HTTPException(_cloud_error_status(exc), detail)

        with cloud_lock:
            state["cloud_job"] = cloud_job
            state["cloud_submit_in_progress"] = False
            _persist_cloud_job(cloud_job)
            _start_cloud_watcher(cloud_job, handle)
            return _public_cloud_job(cloud_job)

    @app.get("/api/cloud/run/status")
    def cloud_run_status():
        from .. import web as cloud_web

        with cloud_lock:
            job = state.get("cloud_job")
            job_id = job.get("id") if job else None
        if not job or not isinstance(job_id, str):
            return _public_cloud_job(job)
        try:
            remote = cloud_web.job_status(job_id)
        except Exception as exc:
            with cloud_lock:
                if state.get("cloud_job") is job:
                    job["poll_error"] = str(exc)
                    return _public_cloud_job(job)
                return _public_cloud_job()
        with cloud_lock:
            if state.get("cloud_job") is job:
                _apply_cloud_remote(job, remote)
                job["poll_error"] = None
                if job.get("status") in {"succeeded", "failed", "cancelled"}:
                    _persist_cloud_job(job)
                return _public_cloud_job(job)
            return _public_cloud_job()

    @app.post("/api/cloud/run/cancel")
    def cancel_cloud_run():
        from .. import web as cloud_web

        with cloud_lock:
            job = state.get("cloud_job")
            if (job is None or not isinstance(job.get("id"), str)
                    or job.get("status") not in {
                        "submitted", "queued", "provisioning", "running", "unknown"}):
                raise HTTPException(409, "no cancellable cloud simulation is active")
            job_id = str(job["id"])
            job["cancel_requested"] = True
        try:
            response = cloud_web.cancel(job_id)
        except Exception as exc:
            with cloud_lock:
                if state.get("cloud_job") is job:
                    job["poll_error"] = str(exc)
                    _persist_cloud_job(job)
            raise HTTPException(_cloud_error_status(exc), str(exc))
        with cloud_lock:
            if state.get("cloud_job") is job:
                _apply_cloud_remote(job, response)
                _persist_cloud_job(job)
                return _public_cloud_job(job)
            return _public_cloud_job()

    @app.post("/api/cloud/run/resume")
    def resume_cloud_run(payload: dict = Body(...)):
        with execution_lock:
            return _resume_cloud_run_locked(payload)

    def _resume_cloud_run_locked(payload: dict):
        from .. import web as cloud_web

        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise HTTPException(400, "job_id is required")
        requested_name = payload.get("name")
        if requested_name is not None and (
                not isinstance(requested_name, str) or not requested_name.strip()
                or len(requested_name) > 128):
            raise HTTPException(400, "name must be a non-empty string when provided")
        with job_lock:
            local_current = state.get("job")
            if local_current is not None and local_current.get("status") in {
                    "queued", "running"}:
                raise HTTPException(
                    409, "a local simulation is active; stop it before "
                         "resuming a cloud run")
        with cloud_lock:
            current = state.get("cloud_job")
            if (current is not None
                    and current.get("download_status") in {"polling", "downloading"}):
                raise HTTPException(
                    409, f"cloud job {current.get('id')} is already being downloaded")
            ambiguous_name = None
            if current and current.get("id") == job_id:
                previous = current
            elif (current and current.get("id") is None
                    and current.get("unresolved_submission")):
                ambiguous_name = current.get("name")
                if requested_name != ambiguous_name:
                    raise HTTPException(
                        409, "the selected history job name does not match the "
                             "unresolved Workbench submission")
                previous = current
            else:
                previous = {}
        if ambiguous_name is not None:
            try:
                raw_history = cloud_web.list_jobs()
            except Exception as exc:
                raise HTTPException(_cloud_error_status(exc), str(exc))
            if not isinstance(raw_history, list):
                raise HTTPException(
                    502, "cloud history did not return a job list; attachment "
                         "remains blocked")
            history_matches = [
                record for record in raw_history
                if isinstance(record, dict)
                and record.get("name") == ambiguous_name
            ]
            if len(history_matches) != 1:
                count = len(history_matches)
                detail = (
                    f"cloud history found no exact job named {ambiguous_name!r}"
                    if count == 0 else
                    f"cloud history found {count} exact jobs named "
                    f"{ambiguous_name!r}"
                )
                raise HTTPException(
                    409, f"{detail}; attachment remains blocked. Do not submit "
                         "again; refresh history and contact the beta operator")
            history_record = history_matches[0]
            history_job_id = history_record.get(
                "job_id", history_record.get("id"))
            if not isinstance(history_job_id, str) or not history_job_id:
                raise HTTPException(
                    409, "the sole exact-name cloud history record has no usable "
                         "job id; attachment remains blocked")
            if history_job_id != job_id:
                raise HTTPException(
                    409, "cloud history changed while attaching: the selected "
                         "job id is no longer the sole exact-name match. Refresh "
                         "history before trying again")
            try:
                remote = cloud_web.job_status(job_id)
            except Exception as exc:
                raise HTTPException(_cloud_error_status(exc), str(exc))
            remote_job_id = (
                remote.get("job_id", remote.get("id"))
                if isinstance(remote, dict) else None
            )
            if (remote_job_id != job_id
                    or remote.get("name") != ambiguous_name):
                raise HTTPException(
                    409, "the service job changed while attachment was being "
                         "verified; its job id and name no longer match the sole "
                         "Workbench history record")
        if previous.get("sim") is None:
            raise HTTPException(
                409, "Workbench does not have the immutable submitted simulation "
                     "for this history job; reopen the task that submitted it")
        try:
            handle = cloud_web.resume(job_id)
        except Exception as exc:
            raise HTTPException(_cloud_error_status(exc), str(exc))
        resumed = {
            "id": job_id, "name": previous.get("name") or job_id,
            "status": previous.get("status") or "unknown",
            "device": previous.get("device") or "gpu",
            "quote_usd": previous.get("quote_usd"),
            "actual_usd": previous.get("actual_usd"),
            "refunded_usd": previous.get("refunded_usd"),
            "max_usd": previous.get("max_usd"),
            "available_usd": previous.get("available_usd"),
            "remaining_usd": previous.get("remaining_usd"),
            "quote_expires_at": previous.get("quote_expires_at"),
            "progress": previous.get("progress"), "error": None,
            "poll_error": None, "download_status": "polling", "session": None,
            "output_dir": previous.get("output_dir"),
            "submitted_at": previous.get("submitted_at") or time.time(),
            "finished_at": None, "cancel_requested": False,
            "handle": None, "thread": None,
            "unresolved_submission": False,
            "ledger_run_id": previous.get("ledger_run_id"),
            "sim": previous.get("sim"),
        }
        with cloud_lock:
            state["cloud_job"] = resumed
            _persist_cloud_job(resumed)
            _start_cloud_watcher(resumed, handle)
            return _public_cloud_job(resumed)

    @app.get("/api/cloud/jobs")
    def list_cloud_jobs():
        from .. import web as cloud_web

        try:
            cloud_web.get_config()
        except Exception as exc:
            return {"configured": False, "jobs": [], "error": str(exc)}
        try:
            raw = cloud_web.list_jobs()
        except Exception as exc:
            return {"configured": True, "jobs": [], "error": str(exc)}
        jobs = [item for item in (
            _public_cloud_history(record) for record in raw) if item is not None]
        return {"configured": True, "jobs": jobs, "error": None}

    # --- Immutable local run history + pure A/B comparison -----------------
    def _history_record(run_id: str) -> dict:
        try:
            return ledger.get_run(
                run_id, active=bool(_active_run_id() == str(run_id).lower()))
        except LedgerNotFound:
            raise HTTPException(404, f"unknown run id: {run_id}")
        except LedgerError as exc:
            raise HTTPException(409, str(exc))

    def _history_data(run_id: str):
        record = _history_record(run_id)
        if record.get("recorded_status") != "completed":
            raise HTTPException(
                409, f"run {run_id} has no completed result bundle "
                     f"(status: {record.get('status')})")
        try:
            ledger.verify_bundle_metadata(run_id)
            data = service.load_result(record["output_dir"])
            _bind_sealed_manifest(record, data)
            # Close the verify/load race: the parsed bytes are bound above, and
            # the path must still contain the sealed metadata after the read.
            ledger.verify_bundle_metadata(run_id)
            return record, data
        except (LedgerError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(409, f"historical run {run_id} failed integrity/load checks: {exc}")

    def _monitor_artifact(run_id: str, data, name: str):
        monitor = next((item for item in data.manifest.get("monitors", [])
                        if isinstance(item, dict) and item.get("name") == name), None)
        if monitor is None:
            raise HTTPException(404, f"run {run_id} has no monitor {name!r}")
        filename = monitor.get("file")
        try:
            token = ledger.verify_artifact(run_id, str(filename))
        except LedgerError as exc:
            raise HTTPException(409, str(exc))
        return str(filename), token

    def _historical_numerical(run_id: str, data, name: str, render):
        filename, before = _monitor_artifact(run_id, data, name)
        payload = render()
        try:
            after = ledger.verify_artifact(run_id, filename)
        except LedgerError as exc:
            raise HTTPException(409, str(exc))
        if after != before:
            raise HTTPException(
                409, f"sealed artifact {filename!r} changed during numerical read")
        return payload

    @app.get("/api/runs")
    def list_runs(limit: int = 100, cursor: Optional[str] = None):
        try:
            return ledger.list_runs(
                limit=limit, cursor=cursor, active_run_id=_active_run_id())
        except (LedgerError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/runs/{run_id}")
    def get_run_record(run_id: str):
        return _history_record(run_id)

    @app.post("/api/runs/{run_id}/workspace")
    def workspace_from_run(run_id: str):
        """Recover an editable copy of any immutable run request.

        Failed, cancelled, and interrupted attempts have no openable result, but
        their canonical request spec is still durable evidence and should never
        be stranded after an application restart.
        """
        request_seq = _begin_workspace_update()
        record = _history_record(run_id)
        spec = record.get("spec")
        if not isinstance(spec, dict):
            raise HTTPException(409, f"run {run_id} has no recoverable simulation spec")
        try:
            sim, messages = service.parse_sim_spec(spec)
        except Exception as exc:
            raise HTTPException(
                409, f"run {run_id} recorded an invalid simulation spec: {exc}")
        warning = (
            f"Recovered a new unsaved setup from {record.get('status', 'recorded')} "
            f"run {run_id}."
        )
        return _workspace_payload(_set_workspace(
            sim, dirty=True, warnings_=[warning, *messages],
            request_seq=request_seq,
        ))

    @app.post("/api/runs/{run_id}/open")
    def open_historical_run(run_id: str):
        # Same arrival-order contract as manual Open: a slow integrity pass for
        # history A may not replace a newer user selection B when it finishes.
        with result_lock:
            state["result_seq"] += 1
            request_seq = state["result_seq"]
        record, data = _history_data(run_id)
        result_id = uuid.uuid4().hex
        result_session = service.session(data, result_id, run_id=record["run_id"])
        with result_lock:
            if request_seq != state["result_seq"]:
                raise HTTPException(409, "stale result open — a newer selection won")
            state["result"] = (data, result_id, record["run_id"])
        _resolve_cloud_recovery_for_run(record["run_id"])
        return result_session

    @app.get("/api/compare")
    def compare_historical_runs(a: str, b: str):
        if a.lower() == b.lower():
            raise HTTPException(400, "choose two distinct run ids for A/B comparison")
        a_record, a_data = _history_data(a)
        b_record, b_data = _history_data(b)

        try:
            payload = service.compare_run_data(
                a_record, b_record, a_data, b_data,
                include_numerical=False, include_specs=False,
            )
            # The initial comparison is deliberately metadata-only.  The UI
            # fetches a selected spectrum/field through run-addressed routes,
            # where that artifact receives an independent checksum pass.
            payload.pop("spectra", None)
            payload.pop("fields", None)
            return payload
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/runs/{run_id}/monitor/{name}/meta")
    def get_historical_meta(run_id: str, name: str):
        _, data = _history_data(run_id)
        try:
            return service.meta(data, name)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/runs/{run_id}/monitor/{name}/spectrum")
    def get_historical_spectrum(run_id: str, name: str):
        _, data = _history_data(run_id)
        try:
            return _historical_numerical(
                run_id, data, name,
                lambda: figures.spectrum_figure(data, name))
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/runs/{run_id}/monitor/{name}/field")
    def get_historical_field(
        run_id: str, name: str, field: str = "Ex", val: str = "real",
        freq: Optional[float] = None, time: Optional[float] = None,
        axis: Optional[str] = None, pos: Optional[float] = None,
        cmap: Optional[str] = None, structures: bool = True,
    ):
        _, data = _history_data(run_id)
        try:
            return _historical_numerical(
                run_id, data, name,
                lambda: figures.field_figure(
                    data, name, field=field, val=val, freq=freq, time=time,
                    axis=axis, pos=pos, cmap=cmap, structures=structures,
                ))
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/runs/{run_id}/monitor/{name}/stats")
    def get_historical_stats(
        run_id: str, name: str, field: str = "Ex", val: str = "abs",
        freq: Optional[float] = None, time: Optional[float] = None,
        axis: Optional[str] = None, pos: Optional[float] = None,
    ):
        _, data = _history_data(run_id)
        try:
            return _historical_numerical(
                run_id, data, name,
                lambda: service.field_stats(
                    data, name, field=field, val=val, freq=freq, time=time,
                    axis=axis, pos=pos,
                ))
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/scene")
    def get_scene(rev: Optional[str] = None):
        try:
            return figures.scene_figure(_data(rev, verify_metadata=True))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/eps")
    def get_eps(axis: str = "z", pos: float = 0.0, rev: Optional[str] = None):
        try:
            return figures.eps_figure(
                _data(rev, verify_metadata=True), axis, pos)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    # --- Preview: render a live Simulation spec file (the input as data) ------
    # The viewer watches the file (not Python code) — POST a path, then poll
    # /status; the sim is re-parsed when the file's mtime changes.
    def _reload_preview():
        """The preview holder, re-parsing the spec file if its mtime changed. On a
        parse/IO error the *last good* sim is kept and ``error`` is set (so the viewer
        keeps rendering the last valid spec); raises 404 when no preview is open."""
        with workspace_lock:
            current = state["preview"]
            if current is None:
                raise HTTPException(404, "no preview open — POST /api/preview {path}")
            pv = dict(current)
            observed_seq = state["workspace_seq"]

        def claim_observed_file_event():
            """Claim only if this poll still observes the current workspace.

            Filesystem stat/read happens outside the lock. A GUI mutation that
            arrived meanwhile must win even if this older poll finishes later;
            a watcher is an observation, not a newer user intent.
            """
            with workspace_lock:
                if observed_seq != state["workspace_seq"]:
                    latest = state["preview"]
                    return None, dict(latest) if latest is not None else None
                state["workspace_seq"] += 1
                return state["workspace_seq"], None

        def latest_if_superseded():
            with workspace_lock:
                if observed_seq == state["workspace_seq"]:
                    return None
                latest = state["preview"]
                return dict(latest) if latest is not None else None
        if not pv.get("path"):
            return pv
        try:
            file_stat = Path(pv["path"]).stat()
        except OSError as e:
            observed_error = f"{type(e).__name__}:{getattr(e, 'errno', None)}:{e}"
            if observed_error == pv.get("observed_fs_error"):
                return latest_if_superseded() or pv
            request_seq, latest = claim_observed_file_event()
            if request_seq is None:
                return latest
            failed = {
                **pv, "observed_fs_error": observed_error,
                "external_change": True, "error": str(e),
            }
            return _publish_workspace(failed, request_seq)
        comparison_identity = (
            pv.get("observed_identity")
            if pv.get("external_change") or pv.get("error")
            else pv.get("file_identity"))
        if (not pv.get("observed_fs_error")
                and isinstance(comparison_identity, dict)
                and int(file_stat.st_mtime_ns) == comparison_identity.get("mtime_ns")
                and int(file_stat.st_size) == comparison_identity.get("size")):
            return latest_if_superseded() or pv

        # A detected filesystem edit is itself a newer workspace event. Claim
        # its sequence before parsing so a concurrent, later GUI edit can win.
        request_seq, latest = claim_observed_file_event()
        if request_seq is None:
            return latest
        try:
            file_bytes, identity = _read_file_snapshot(Path(pv["path"]))
        except OSError as e:
            failed = {
                **pv,
                "observed_fs_error": (
                    f"{type(e).__name__}:{getattr(e, 'errno', None)}:{e}"),
                "external_change": True,
                "error": str(e),
            }
            return _publish_workspace(failed, request_seq)
        mtime = float(identity["mtime_ns"]) / 1e9
        if pv.get("dirty"):
            conflicted = {
                **pv,
                "observed_mtime": mtime,
                "observed_identity": identity,
                "observed_fs_error": None,
                "external_change": True,
                "error": (
                    "the file changed on disk while this workspace has unsaved edits; "
                    "reload it, save to a different path, or explicitly overwrite it"
                ),
            }
            return _publish_workspace(conflicted, request_seq)
        try:  # keep the last good sim on a mid-edit parse error, but report it
            raw = json.loads(file_bytes.decode("utf-8"))
            sim, messages = service.parse_sim_spec(raw)
            reloaded = {
                **pv, "sim": sim, "warnings": messages, "mtime": mtime,
                "observed_mtime": mtime,
                "file_identity": identity,
                "observed_identity": identity,
                "observed_fs_error": None,
                "error": None, "external_change": False,
            }
        except Exception as e:  # noqa: BLE001 — surface to the UI
            reloaded = {
                **pv, "observed_mtime": mtime,
                "observed_identity": identity,
                "observed_fs_error": None,
                "error": f"{type(e).__name__}: {e}",
            }
        return _publish_workspace(reloaded, request_seq)

    def _preview_payload(pv):
        return _workspace_payload(pv)

    @app.post("/api/preview")
    def open_preview(path: str = Body(..., embed=True)):
        request_seq = _begin_workspace_update()
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise HTTPException(400, f"not a file: {path!r}")
        try:
            file_bytes, identity = _read_file_snapshot(p)
            raw = json.loads(file_bytes.decode("utf-8"))
            sim, messages = service.parse_sim_spec(raw)
        except Exception as e:
            raise HTTPException(400, f"cannot parse {path!r}: {e}")
        pv = _set_workspace(
            sim, path=str(p),
            mtime=float(identity["mtime_ns"]) / 1e9,
            file_identity=identity,
            dirty=False, warnings_=messages,
            request_seq=request_seq)
        return _preview_payload(pv)

    @app.get("/api/preview/status")
    def preview_status():
        return _preview_payload(_reload_preview())

    @app.get("/api/preview/scene")
    def preview_scene():
        sim = _reload_preview()["sim"]  # may 404 (no preview) — keep outside the try
        try:
            return figures.scene_figure_from_sim(sim)
        except Exception as e:  # noqa: BLE001 — render arbitrary in-progress specs defensively
            raise HTTPException(400, f"cannot render scene: {e}")

    @app.get("/api/preview/eps")
    def preview_eps(axis: str = "z", pos: float = 0.0):
        sim = _reload_preview()["sim"]
        try:
            return figures.eps_figure_from_sim(sim, axis, pos)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/result/checks")
    def get_result_checks(rev: Optional[str] = None):
        """Advisory data-health findings over every monitor of the open result."""
        data = _data(rev)
        if not data.manifest.get("monitors"):
            return checks.result_checks(data)

        def _all_monitor_names(value):
            return [item["name"]
                    for item in value.manifest.get("monitors", [])
                    if isinstance(item, dict) and item.get("name")]

        try:
            return _multi_numerical_result(
                rev, _all_monitor_names, checks.result_checks)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/result/spec")
    def get_result_spec(rev: Optional[str] = None):
        """The recorded input document (sibling sim.json), provenance-gated.

        Results-mode settings sections render this — never the live design
        draft, which may have diverged since the run.
        """
        return service.recorded_spec(_data(rev))

    @app.get("/api/monitor/{name}/meta")
    def get_meta(name: str, rev: Optional[str] = None):
        try:
            # Slider/catalog metadata comes entirely from the sealed manifest;
            # do not hash a potentially multi-GB field blob until a numerical
            # field/stats request actually reads it.
            return service.meta(_data(rev, verify_metadata=True), name)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/monitor/{name}/field")
    def get_field(name: str, field: str = "Ex", val: str = "real",
                  freq: Optional[float] = None, time: Optional[float] = None,
                  axis: Optional[str] = None, pos: Optional[float] = None,
                  cmap: Optional[str] = None, structures: bool = True,
                  rev: Optional[str] = None):
        try:
            return _numerical_result(
                rev, name,
                lambda data: figures.field_figure(
                    data, name, field=field, val=val,
                    freq=freq, time=time, axis=axis, pos=pos,
                    cmap=cmap, structures=structures))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/monitor/{name}/stats")
    def get_field_stats(name: str, field: str = "Ex", val: str = "abs",
                        freq: Optional[float] = None, time: Optional[float] = None,
                        axis: Optional[str] = None, pos: Optional[float] = None,
                        rev: Optional[str] = None):
        try:
            return _numerical_result(
                rev, name,
                lambda data: service.field_stats(
                    data, name, field=field, val=val, freq=freq, time=time,
                    axis=axis, pos=pos))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/monitor/{name}/spectrum")
    def get_spectrum(name: str, rev: Optional[str] = None):
        try:
            return _numerical_result(
                rev, name, lambda data: figures.spectrum_figure(data, name))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/monitor/{name}/profile")
    def get_profile(name: str, field: str = "Ex", val: str = "real",
                    freq: Optional[float] = None, time: Optional[float] = None,
                    rev: Optional[str] = None):
        try:
            return _numerical_result(
                rev, name,
                lambda data: figures.profile_figure(
                    data, name, field=field, val=val, freq=freq, time=time))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/monitor/{name}/field-spectrum")
    def get_field_spectrum(name: str, field: str = "Ex", val: str = "abs",
                           rev: Optional[str] = None):
        try:
            return _numerical_result(
                rev, name,
                lambda data: figures.field_spectrum_figure(
                    data, name, field=field, val=val))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/monitor/{name}/timeseries")
    def get_timeseries(name: str, field: str = "Ex", val: str = "real",
                       rev: Optional[str] = None):
        try:
            return _numerical_result(
                rev, name,
                lambda data: figures.timeseries_figure(
                    data, name, field=field, val=val))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/monitor/{name}/fft")
    def get_timeseries_fft(name: str, field: str = "Ex",
                           rev: Optional[str] = None):
        try:
            return _numerical_result(
                rev, name,
                lambda data: figures.timeseries_fft_figure(
                    data, name, field=field))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/ports/modal")
    def get_modal_ports(rev: Optional[str] = None):
        """Return one driven multimode S-column for the open result bundle."""
        try:
            return _multi_numerical_result(
                rev,
                service.modal_port_monitor_names,
                service.modal_port_results,
                cache=modal_result_cache,
                cache_lock=modal_cache_lock,
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except service.StaleModeSourceError as exc:
            raise HTTPException(409, detail={
                "code": "stale_mode_source",
                "message": str(exc),
                "mode_source_statuses": exc.statuses,
            }) from exc
        except (ImportError, KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    # Serve the built UI at / (packaged app, same-origin), else a headless info page.
    if ui_dir and Path(ui_dir).is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    else:
        if ui_dir:
            import warnings
            warnings.warn(f"--ui-dir {str(ui_dir)!r} is not a directory; serving the info page")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return _INDEX

    # Add this last so it is the outermost application middleware. In an
    # Electron-owned launch, no API route (including health, cloud endpoints,
    # future streaming responses, or future WebSockets) can run first.
    app.add_middleware(
        _DesktopCapabilityMiddleware, launch_capability=launch_token,
    )
    return app


def _watch_parent(request_shutdown: Callable[[], None]) -> None:
    """Exit when the spawning parent dies. A parent (e.g. the Electron main) holds
    our stdin pipe open; EOF on stdin means it's gone, so we shut down rather than
    orphan. No-op when stdin is a TTY (standalone ``serve-viz`` in a terminal)."""
    import sys

    if sys.stdin is None or sys.stdin.isatty():
        return

    def watch() -> None:
        try:
            sys.stdin.read()  # blocks until EOF (parent closed the pipe / died)
        except Exception:
            pass
        # Do not synthesize POSIX signals: Uvicorn's native shutdown flag works
        # the same way in a frozen Windows sidecar and runs the ASGI lifespan.
        request_shutdown()

    threading.Thread(target=watch, daemon=True).start()


def _read_desktop_launch_capability() -> str:
    """Read one Electron capability from the private inherited stdin pipe."""
    import sys

    if sys.stdin is None or sys.stdin.isatty():
        raise ValueError("desktop launch capability requires a private stdin pipe")
    value = sys.stdin.readline(66)
    if not value.endswith("\n"):
        raise ValueError("desktop launch capability framing is invalid")
    capability = value[:-1]
    if not _DESKTOP_CAPABILITY_PATTERN.fullmatch(capability):
        raise ValueError("desktop launch capability is invalid")
    return capability


def _persistent_run_root(run_root: Optional[str | Path] = None) -> Path:
    """Resolve the CLI/desktop archive; direct ``create_app`` stays isolated."""
    configured = run_root or os.environ.get("PHOTONHUB_RUN_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    default = (Path.home() / ".photonhub" / "runs").resolve()
    if not default.exists():
        # Installs that predate the SimuPod → PhotonHub rename keep their run
        # history here. Adopt it in place rather than moving user data, and only
        # when the current archive has not been created yet.
        legacy = (Path.home() / ".simupod" / "runs").resolve()
        if legacy.is_dir():
            return legacy
    return default


def _load_desktop_release(path: str | Path) -> dict:
    """Validate an installed release identity and every bound resource byte."""
    from ..release import load_release_manifest

    manifest_path = Path(path).expanduser().resolve()
    return load_release_manifest(manifest_path, artifact_root=manifest_path.parent)


def serve_viz(result_dir: Optional[str | Path] = None, *, port: int = 8765,
              open_browser: bool = True, ui_dir: Optional[str | Path] = None,
              watch_parent: bool = False,
              run_root: Optional[str | Path] = None,
              launch_token: Optional[str] = None,
              release_manifest: Optional[str | Path] = None) -> None:
    """Start the local viz server on 127.0.0.1 and (optionally) open a browser."""
    import uvicorn

    server_ref: dict = {}

    def request_shutdown() -> None:
        server = server_ref.get("server")
        if server is not None:
            server.should_exit = True

    release_identity = (
        _load_desktop_release(release_manifest) if release_manifest else None
    )
    app = create_app(
        result_dir, ui_dir=ui_dir, run_root=_persistent_run_root(run_root),
        launch_token=launch_token, shutdown_callback=request_shutdown,
        release_identity=release_identity)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_ref["server"] = server
    if watch_parent:
        _watch_parent(request_shutdown)
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        webbrowser.open(url)
    print(f"[photonhub] viz server → {url}  (Ctrl-C to stop)")
    server.run()


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="photonhub serve-viz")
    ap.add_argument("result_dir", nargs="?", default=None,
                    help="a result bundle (dir with manifest.json); optional — "
                         "launch empty and Open/Preview in the app")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    ap.add_argument("--ui-dir", default=None, help="serve a built web UI at / (packaged app)")
    ap.add_argument(
        "--run-root", default=None,
        help="persistent run-ledger/archive directory "
             "(default: $PHOTONHUB_RUN_ROOT or ~/.photonhub/runs)",
    )
    ap.add_argument("--watch-parent", action="store_true",
                    help="exit when the parent process that spawned us dies")
    ap.add_argument(
        "--launch-capability-stdin", action="store_true", help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--release-manifest", default=None,
        help="validate and report an installed Workbench release.json",
    )
    a = ap.parse_args(argv)
    try:
        launch_capability = (
            _read_desktop_launch_capability()
            if a.launch_capability_stdin else None
        )
        serve_viz(
            a.result_dir, port=a.port, open_browser=not a.no_open,
            ui_dir=a.ui_dir, watch_parent=a.watch_parent,
            run_root=a.run_root, launch_token=launch_capability,
            release_manifest=a.release_manifest,
        )
    except (OSError, ValueError) as exc:
        if a.release_manifest:
            print(
                "PhotonHub Workbench installation verification failed; "
                f"repair or reinstall the application: {exc}",
                file=__import__("sys").stderr,
            )
            return 2
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
