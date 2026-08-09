"""Durable, append-only local run history for the desktop workbench.

The viewer's ``result_id`` is deliberately ephemeral: it is a revision guard for
one mutable UI selection.  A ledger ``run_id`` is the opposite -- a stable UUID
that names one immutable execution request and its terminal evidence forever.

SQLite is part of Python's standard library and gives this local-only service
transactional, crash-safe appends on every supported desktop platform.  The
tables are protected by UPDATE/DELETE triggers; application history is expressed
only by new request, event, and terminal rows.  Solver artifacts remain ordinary
files so multi-GB fields are never copied into the database.  A terminal row
seals their sizes and streaming SHA-256 digests instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..data import (
    validate_monitor_manifest_entry,
    validate_result_manifest_contract,
)


_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_TERMINAL = {"completed", "failed", "cancelled"}
_STATUSES = {"queued", "running", *_TERMINAL}
_BASE_ARTIFACTS = ("sim.json", "manifest.json", "solver-events.jsonl")
_DTYPE_BYTES = {
    "float32": 4, "<f4": 4, "f4": 4,
    "float64": 8, "<f8": 8, "f8": 8,
    "complex64": 8, "<c8": 8, "c8": 8,
}


class LedgerError(RuntimeError):
    """The durable run record could not be created or sealed safely."""


class LedgerNotFound(KeyError):
    """No immutable request exists for a run id."""


def utc_now() -> str:
    """RFC3339 UTC timestamp with a stable, JSON-friendly spelling."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z")


def _json_text(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stat_signature(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(file_stat.st_dev), int(file_stat.st_ino), int(file_stat.st_size),
        int(file_stat.st_mtime_ns), int(file_stat.st_ctime_ns),
    )


def _windows_change_time(path: Path) -> Optional[int]:
    """Return NTFS FILE_BASIC_INFO.ChangeTime for safe checksum caching.

    Python's ``st_ctime`` is historically the creation time on Windows. The
    native change timestamp is the metadata signal we need to detect same-size
    in-place rewrites without rehashing multi-GB fields on every GUI slider
    request. Failure returns ``None`` and the caller safely disables caching.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class FILE_BASIC_INFO(ctypes.Structure):
            _fields_ = [
                ("CreationTime", ctypes.c_longlong),
                ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong),
                ("ChangeTime", ctypes.c_longlong),
                ("FileAttributes", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        create.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        create.restype = wintypes.HANDLE
        get_info = kernel32.GetFileInformationByHandleEx
        get_info.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        close = kernel32.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        handle = create(
            str(path), 0, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            return None
        try:
            info = FILE_BASIC_INFO()
            if not get_info(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
                return None
            return int(info.ChangeTime)
        finally:
            close(handle)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _hash_regular_file(
    path: Path, *, capture: bool = False,
) -> tuple[os.stat_result, int, str, Optional[bytes]]:
    """Hash one stable, non-symlink inode and optionally retain its exact bytes."""
    try:
        before = path.lstat()
    except OSError as exc:
        raise LedgerError(f"cannot stat artifact {path.name!r}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise LedgerError(
            f"artifact {path.name!r} is not a regular non-symlink file")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise LedgerError(f"cannot open artifact {path.name!r}: {exc}") from exc
    digest = hashlib.sha256()
    size = 0
    captured = bytearray() if capture else None
    try:
        stream = os.fdopen(fd, "rb")
    except Exception:
        os.close(fd)
        raise
    try:
        with stream:
            opened = os.fstat(stream.fileno())
            if _stat_signature(opened) != _stat_signature(before):
                raise LedgerError(
                    f"artifact {path.name!r} changed while it was opened")
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
                if captured is not None:
                    captured.extend(block)
            after_fd = os.fstat(stream.fileno())
    except Exception:
        # ``fdopen`` owns and closes fd after it succeeds.
        raise
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise LedgerError(
            f"artifact {path.name!r} disappeared while hashing: {exc}") from exc
    signature = _stat_signature(before)
    if (_stat_signature(after_fd) != signature
            or _stat_signature(after_path) != signature
            or size != int(before.st_size)):
        raise LedgerError(f"artifact {path.name!r} changed while hashing")
    return after_path, size, digest.hexdigest(), (
        bytes(captured) if captured is not None else None)


def _safe_artifact_name(value: Any) -> str:
    """Return a bundle-relative leaf filename or fail closed.

    Engine manifests currently use one file per monitor in the bundle root.  Do
    not let a corrupted/imported manifest turn sealing into an arbitrary file
    reader, and do not follow a symlink even when its textual name is harmless.
    """
    if not isinstance(value, str) or not value or "\x00" in value:
        raise LedgerError(f"unsafe result artifact name: {value!r}")
    if value in {".", ".."} or Path(value).is_absolute():
        raise LedgerError(f"unsafe result artifact name: {value!r}")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise LedgerError(f"unsafe result artifact name: {value!r}")
    return value


def _monitor_contract_issues(expected_spec: dict, manifest: dict) -> list[str]:
    """Bind result monitor identity/interpretation to the hashed request."""
    requested = expected_spec.get("monitors", []) if isinstance(expected_spec, dict) else []
    reported = manifest.get("monitors", []) if isinstance(manifest, dict) else []
    if not isinstance(requested, list):
        return ["immutable request monitors is not a list"]
    if not isinstance(reported, list):
        return []  # the manifest envelope validator reports this precisely

    def monitor_map(items: list) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                out.setdefault(item["name"], item)
        return out

    def contract(item: dict, *, result: bool) -> dict:
        kind = item.get("type")
        value: dict[str, Any] = {"type": kind}
        if kind in {"field_time", "field_snapshot", "field_dft"}:
            value["components"] = list(item.get(
                "components" if result else "fields", []))
        if kind in {"field_dft", "flux"}:
            value["freqs_hz"] = list(item.get("freqs_hz", []))
        if kind == "field_dft":
            value["interval_space"] = list(
                item.get("interval_space") or (1, 1, 1))
        if kind == "flux":
            value["axis"] = item.get("axis")
        return value

    request_map, result_map = monitor_map(requested), monitor_map(reported)
    issues: list[str] = []
    missing = sorted(set(request_map) - set(result_map))
    unexpected = sorted(set(result_map) - set(request_map))
    if missing:
        issues.append(f"result manifest is missing requested monitors: {missing}")
    if unexpected:
        issues.append(f"result manifest has unrequested monitors: {unexpected}")
    for name in sorted(set(request_map) & set(result_map)):
        expected = contract(request_map[name], result=False)
        actual = contract(result_map[name], result=True)
        if actual != expected:
            issues.append(
                f"result monitor {name!r} contract does not match immutable request: "
                f"expected {expected!r}, got {actual!r}")
    return issues


class RunLedger:
    """Append-only run requests, lifecycle events, and terminal evidence."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bundles_dir = self.root / "bundles"
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "runs.sqlite3"
        self._verify_lock = threading.Lock()
        # A successful checksum may be reused only while every cheap identity and
        # change signal exposed by the filesystem is unchanged.  In particular,
        # ctime catches a rewrite followed by restoring the old mtime.
        self._verified: dict[
            tuple[str, str], tuple[int, int, int, int, Optional[int], str]
        ] = {}
        # Terminal rows are application-enforced immutable. Cache their already
        # hash-verified artifact index so interactive field sliders do not parse
        # a large request/terminal JSON document on every slice request.
        self._artifact_records: dict[str, dict[str, dict]] = {}
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS run_requests (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    output_dir TEXT NOT NULL UNIQUE,
                    device TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    request_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL REFERENCES run_requests(run_id),
                    seq INTEGER NOT NULL,
                    at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued','running','completed','failed','cancelled')
                    ),
                    detail_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq)
                );

                CREATE TABLE IF NOT EXISTS run_terminals (
                    run_id TEXT PRIMARY KEY REFERENCES run_requests(run_id),
                    finished_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('completed','failed','cancelled')
                    ),
                    terminal_sha256 TEXT NOT NULL,
                    terminal_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS run_requests_created
                    ON run_requests(created_at DESC, run_id DESC);

                CREATE TRIGGER IF NOT EXISTS immutable_requests_update
                BEFORE UPDATE ON run_requests BEGIN
                    SELECT RAISE(ABORT, 'run_requests is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS immutable_requests_delete
                BEFORE DELETE ON run_requests BEGIN
                    SELECT RAISE(ABORT, 'run_requests is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS immutable_events_update
                BEFORE UPDATE ON run_events BEGIN
                    SELECT RAISE(ABORT, 'run_events is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS immutable_events_delete
                BEFORE DELETE ON run_events BEGIN
                    SELECT RAISE(ABORT, 'run_events is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS immutable_terminals_update
                BEFORE UPDATE ON run_terminals BEGIN
                    SELECT RAISE(ABORT, 'run_terminals is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS immutable_terminals_delete
                BEFORE DELETE ON run_terminals BEGIN
                    SELECT RAISE(ABORT, 'run_terminals is append-only');
                END;
                """
            )

    @staticmethod
    def validate_run_id(run_id: str) -> str:
        value = str(run_id).lower()
        if not _RUN_ID.fullmatch(value):
            raise LedgerNotFound(run_id)
        return value

    @staticmethod
    def _label_slug(label: Optional[str]) -> str:
        """A filesystem-safe stem for a run directory, or "" to fall back.

        The caller's label is a display name typed by a person, so it is never
        trusted as a path component: only letters, digits, dot, dash and
        underscore survive, leading dots and dashes are dropped so the result
        can neither hide the directory nor look like a flag, and the length is
        capped well below any filename limit.
        """
        stem = re.sub(r"\.sim\.json$|\.json$", "", str(label or ""), flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
        return cleaned[:64]

    def _output_dir(self, run_id: str, created_at: str,
                    output_parent: Optional[str | Path],
                    label: Optional[str] = None) -> Path:
        if output_parent is None:
            parent = self.bundles_dir
        else:
            parent = Path(output_parent).expanduser().resolve()
            if not parent.is_dir():
                raise LedgerError(f"output parent is not a directory: {parent}")
        stamp = created_at.replace("-", "").replace(":", "")[:15]
        # A named run carries its document's name so the result reads as the
        # same thing the designer ran; the run id still disambiguates repeats.
        slug = self._label_slug(label)
        name = f"{slug}-{stamp}-{run_id[:8]}" if slug else f"photonhub-run-{stamp}-{run_id}"
        output = (parent / name).resolve()
        if output.parent != parent:
            raise LedgerError(f"run output directory escapes its parent: {output}")
        try:
            output.mkdir(mode=0o700, parents=False, exist_ok=False)
        except OSError as exc:
            raise LedgerError(f"cannot create run output directory {output}: {exc}") from exc
        return output

    def create_request(
        self,
        *,
        run_id: Optional[str],
        canonical_spec: str,
        device: str,
        timeout_s: Optional[float],
        solver: dict,
        estimate: dict,
        workspace_path: Optional[str] = None,
        output_parent: Optional[str | Path] = None,
        label: Optional[str] = None,
    ) -> dict:
        """Persist the exact execution request and initial queued event atomically."""
        rid = self.validate_run_id(run_id or uuid.uuid4().hex)
        created_at = utc_now()
        raw_spec = canonical_spec.encode("utf-8")
        try:
            document = json.loads(canonical_spec)
        except json.JSONDecodeError as exc:  # defensive: caller should pass wire JSON
            raise LedgerError(f"canonical simulation spec is not JSON: {exc}") from exc
        spec_sha256 = _sha256_bytes(raw_spec)
        output = self._output_dir(rid, created_at, output_parent, label)
        try:
            request = {
                "contract": "photonhub.run_request/1",
                "run_id": rid,
                "created_at": created_at,
                "output_dir": str(output),
                "execution": {"device": device, "timeout_s": timeout_s},
                "source": {"workspace_path": workspace_path, "label": label or None},
                "spec": {"sha256": spec_sha256, "document": document},
                "estimate": estimate,
                "solver_requested": solver,
            }
            request_json = _json_text(request)
            request_sha256 = _sha256_bytes(request_json.encode("utf-8"))
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO run_requests "
                    "(run_id,created_at,output_dir,device,spec_sha256,request_sha256,request_json) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (rid, created_at, str(output), device, spec_sha256,
                     request_sha256, request_json),
                )
                conn.execute(
                    "INSERT INTO run_events (run_id,seq,at,status,detail_json) "
                    "VALUES (?,0,?,'queued','{}')",
                    (rid, created_at),
                )
        except Exception:
            # The UUID directory is new and empty here; leave no misleading
            # unindexed bundle when the durable request transaction failed.
            try:
                output.rmdir()
            except OSError:
                pass
            raise
        return self.get_run(rid)

    def append_event(self, run_id: str, status: str,
                     detail: Optional[dict] = None, *, at: Optional[str] = None) -> dict:
        rid = self.validate_run_id(run_id)
        if status not in _STATUSES:
            raise LedgerError(f"invalid run status: {status!r}")
        if status in _TERMINAL:
            raise LedgerError("terminal lifecycle events must be committed through seal()")
        with self._connect() as conn:
            # Serialize the read-check-append transition across connections.
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM run_terminals WHERE run_id=?", (rid,)
            ).fetchone() is not None:
                raise LedgerError(f"run {rid} already has an immutable terminal record")
            row = conn.execute(
                "SELECT seq,status FROM run_events WHERE run_id=? "
                "ORDER BY seq DESC LIMIT 1",
                (rid,),
            ).fetchone()
            if row is None:
                raise LedgerNotFound(rid)
            current = str(row["status"])
            if current != "queued" or status != "running":
                raise LedgerError(
                    f"invalid run lifecycle transition {current!r} -> {status!r}")
            conn.execute(
                "INSERT INTO run_events (run_id,seq,at,status,detail_json) "
                "VALUES (?,?,?,?,?)",
                (rid, int(row["seq"]) + 1, at or utc_now(), status,
                 _json_text(detail or {})),
            )
        return self.get_run(rid)

    def _request_row(self, conn: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM run_requests WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise LedgerNotFound(run_id)
        return row

    @staticmethod
    def _decode_hashed_json(raw: str, expected_sha256: str, label: str) -> dict:
        actual = _sha256_bytes(raw.encode("utf-8"))
        if actual != expected_sha256:
            raise LedgerError(f"{label} failed its stored SHA-256 verification")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"{label} is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise LedgerError(f"{label} is not a JSON object")
        return value

    @staticmethod
    def _trusted_output_dir(output_dir: Path) -> Path:
        """Reject a missing/replaced archive directory before reading children."""
        try:
            file_stat = output_dir.lstat()
            resolved = output_dir.resolve(strict=True)
        except OSError as exc:
            raise LedgerError(
                f"run output directory is unavailable: {output_dir}: {exc}") from exc
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISDIR(file_stat.st_mode):
            raise LedgerError(f"run output path is not a regular directory: {output_dir}")
        # The path was canonicalized before it was inserted.  A different value
        # now means some directory component was replaced by a symlink.
        if resolved != output_dir:
            raise LedgerError(
                f"run output directory no longer resolves to its recorded path: {output_dir}")
        return output_dir

    def _artifact_inventory(self, output_dir: Path, *, completed: bool,
                            expected_spec_sha256: str,
                            expected_spec: dict) -> tuple[list[dict], dict, list[str]]:
        manifest: dict = {}
        issues: list[str] = []
        artifacts: list[dict] = []
        try:
            output_dir = self._trusted_output_dir(output_dir)
        except LedgerError as exc:
            issues.append(str(exc))
            return [], manifest, issues
        manifest_path = output_dir / "manifest.json"
        try:
            manifest_stat, manifest_size, manifest_digest, manifest_raw = \
                _hash_regular_file(manifest_path, capture=True)
            artifacts.append({
                "path": "manifest.json", "bytes": manifest_size,
                "mtime_ns": manifest_stat.st_mtime_ns,
                "device": int(manifest_stat.st_dev),
                "inode": int(manifest_stat.st_ino),
                "sha256": manifest_digest,
            })
            loaded = json.loads((manifest_raw or b"").decode("utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                issues.append("manifest.json is not a JSON object")
        except (LedgerError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if completed or manifest_path.exists():
                issues.append(f"manifest.json is unreadable or unsafe: {exc}")

        try:
            validate_result_manifest_contract(
                manifest, raw_files=True, require_top_level=True,
                strict_engine=True)
        except (TypeError, ValueError) as exc:
            issues.append(f"invalid result manifest contract: {exc}")
        issues.extend(_monitor_contract_issues(expected_spec, manifest))

        names = set(_BASE_ARTIFACTS)
        names.discard("manifest.json")
        expected_bytes: dict[str, int] = {}
        artifact_owners: dict[str, str] = {}
        monitors = manifest.get("monitors", [])
        if not isinstance(monitors, list):
            issues.append("manifest monitors is not a list")
            monitors = []
        for block in ("run", "grid", "provenance"):
            if not isinstance(manifest.get(block), dict):
                issues.append(f"manifest {block} is not a JSON object")
        for item in monitors:
            try:
                if not isinstance(item, dict):
                    raise LedgerError(f"invalid monitor manifest entry: {item!r}")
                try:
                    validate_monitor_manifest_entry(
                        item, manifest, require_explicit_dims=True)
                except (KeyError, TypeError, ValueError) as exc:
                    raise LedgerError(
                        f"invalid monitor manifest contract: {exc}") from exc
                filename = _safe_artifact_name(item.get("file"))
                if filename in _BASE_ARTIFACTS:
                    raise LedgerError(
                        f"monitor artifact {filename!r} aliases a reserved bundle file")
                previous_owner = artifact_owners.get(filename)
                if previous_owner is not None:
                    raise LedgerError(
                        f"monitor artifact {filename!r} is shared by monitors "
                        f"{previous_owner!r} and {item.get('name')!r}")
                artifact_owners[filename] = str(item.get("name"))
                names.add(filename)
                dtype = str(item.get("dtype", ""))
                itemsize = _DTYPE_BYTES.get(dtype)
                shape = item.get("shape")
                if itemsize is None:
                    raise LedgerError(
                        f"monitor artifact {filename!r} has unsupported dtype {dtype!r}")
                if not isinstance(shape, list) or not shape:
                    raise LedgerError(
                        f"monitor artifact {filename!r} has invalid shape {shape!r}")
                elements = 1
                for value in shape:
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise LedgerError(
                            f"monitor artifact {filename!r} has invalid shape {shape!r}")
                    elements *= value
                byte_count = elements * itemsize
                expected_bytes[filename] = byte_count
            except (LedgerError, OverflowError) as exc:
                issues.append(str(exc))

        for name in sorted(names):
            path = output_dir / name
            try:
                file_stat, size, digest, _ = _hash_regular_file(path)
            except LedgerError as exc:
                if completed:
                    issues.append(f"required artifact {name!r} is missing: {exc}")
                continue
            artifacts.append({
                "path": name, "bytes": size, "mtime_ns": file_stat.st_mtime_ns,
                "device": int(file_stat.st_dev), "inode": int(file_stat.st_ino),
                "sha256": digest,
            })

        actual_bytes = {item["path"]: int(item["bytes"]) for item in artifacts}
        for name, expected in expected_bytes.items():
            if name in actual_bytes and actual_bytes[name] != expected:
                issues.append(
                    f"monitor artifact {name!r} has {actual_bytes[name]} bytes; "
                    f"manifest dtype/shape requires {expected}")

        sim_entry = next((item for item in artifacts if item["path"] == "sim.json"), None)
        if sim_entry is None:
            if completed:
                issues.append("sim.json is unavailable for input provenance verification")
        elif sim_entry["sha256"] != expected_spec_sha256:
            issues.append(
                "sim.json SHA-256 does not match the immutable execution request")

        input_sha = manifest.get("provenance", {}).get("input_sha256") \
            if isinstance(manifest.get("provenance", {}), dict) else None
        if completed and str(input_sha or "").lower() != expected_spec_sha256:
            issues.append(
                "manifest provenance input_sha256 does not match the immutable request")
        return artifacts, manifest, issues

    def seal(self, run_id: str, status: str, *, error: Optional[dict] = None) -> dict:
        """Hash the stopped bundle and append its one immutable terminal record.

        Completed runs fail closed on missing/unsafe artifacts or mismatched input
        provenance.  Failed/cancelled attempts still preserve whatever safe partial
        artifacts exist, plus any inventory issues, for post-mortem inspection.
        """
        rid = self.validate_run_id(run_id)
        if status not in _TERMINAL:
            raise LedgerError(f"terminal status required, got {status!r}")

        def require_predecessor(current: str) -> None:
            allowed = ({"running"} if status == "completed"
                       else {"queued", "running"})
            if current not in allowed:
                raise LedgerError(
                    f"invalid run lifecycle transition {current!r} -> {status!r}")

        with self._connect() as conn:
            request_row = self._request_row(conn, rid)
            if conn.execute(
                "SELECT 1 FROM run_terminals WHERE run_id=?", (rid,)
            ).fetchone() is not None:
                raise LedgerError(f"run {rid} already has an immutable terminal record")
            latest = conn.execute(
                "SELECT seq,status FROM run_events WHERE run_id=? "
                "ORDER BY seq DESC LIMIT 1", (rid,),
            ).fetchone()
            if latest is None:
                raise LedgerError(f"run {rid} has no lifecycle event")
            require_predecessor(str(latest["status"]))

        request = self._decode_hashed_json(
            request_row["request_json"], request_row["request_sha256"],
            f"run {rid} request record",
        )
        request_spec = request.get("spec")
        expected_spec = (request_spec.get("document")
                         if isinstance(request_spec, dict) else None)
        execution = request.get("execution")
        if (request.get("run_id") != rid
                or request.get("output_dir") != request_row["output_dir"]
                or not isinstance(execution, dict)
                or execution.get("device") != request_row["device"]
                or not isinstance(expected_spec, dict)
                or request_spec.get("sha256") != request_row["spec_sha256"]):
            raise LedgerError(
                f"run {rid} immutable request has an invalid spec binding")

        output_dir = Path(request_row["output_dir"])
        artifacts, manifest, issues = self._artifact_inventory(
            output_dir, completed=status == "completed",
            expected_spec_sha256=str(request_row["spec_sha256"]),
            expected_spec=expected_spec,
        )
        if status == "completed" and issues:
            raise LedgerError("cannot seal completed run: " + "; ".join(issues))

        run_summary = manifest.get("run", {})
        grid_summary = manifest.get("grid", {})
        provenance_summary = manifest.get("provenance", {})
        monitor_summary = manifest.get("monitors", [])

        # The manifest artifact itself is the full sealed evidence. Keep the
        # terminal's query summary deliberately bounded so history polling does
        # not duplicate graded-grid coordinates or long sample/frequency arrays.
        compact_grid = ({key: value for key, value in grid_summary.items()
                         if key != "coords_um"}
                        if isinstance(grid_summary, dict) else {})
        monitor_keys = {
            "name", "type", "file", "dtype", "shape", "dims", "components",
            "axis", "origin_cells", "interval_space",
        }
        compact_monitors = ([
            {key: value for key, value in item.items() if key in monitor_keys}
            for item in monitor_summary if isinstance(item, dict)
        ] if isinstance(monitor_summary, list) else [])

        with self._connect() as conn:
            # A queued->running append may race the potentially long hashing
            # pass. Re-read the predecessor and started_at under one write lock,
            # then derive the immutable terminal from that final lifecycle.
            conn.execute("BEGIN IMMEDIATE")
            self._request_row(conn, rid)
            if conn.execute(
                "SELECT 1 FROM run_terminals WHERE run_id=?", (rid,)
            ).fetchone() is not None:
                raise LedgerError(f"run {rid} already has an immutable terminal record")
            row = conn.execute(
                "SELECT seq,status FROM run_events WHERE run_id=? "
                "ORDER BY seq DESC LIMIT 1",
                (rid,),
            ).fetchone()
            if row is None:
                raise LedgerError(f"run {rid} has no lifecycle event")
            require_predecessor(str(row["status"]))
            started = conn.execute(
                "SELECT at FROM run_events WHERE run_id=? AND status='running' "
                "ORDER BY seq LIMIT 1", (rid,),
            ).fetchone()
            finished_at = utc_now()
            terminal = {
                "contract": "photonhub.run_result/1",
                "run_id": rid,
                "request_sha256": request_row["request_sha256"],
                "status": status,
                "started_at": started["at"] if started is not None else None,
                "finished_at": finished_at,
                "error": error,
                "summary": {
                    "run": run_summary if isinstance(run_summary, dict) else {},
                    "grid": compact_grid,
                    "provenance": (provenance_summary
                                   if isinstance(provenance_summary, dict) else {}),
                    "monitors": compact_monitors,
                },
                "integrity": {
                    "algorithm": "sha256",
                    "spec_matches_manifest": not any(
                        "input_sha256" in issue or "sim.json SHA-256" in issue
                        for issue in issues
                    ),
                    "issues": issues,
                    "total_bytes": sum(int(item["bytes"]) for item in artifacts),
                    "artifacts": artifacts,
                },
            }
            terminal_json = _json_text(terminal)
            terminal_sha = _sha256_bytes(terminal_json.encode("utf-8"))
            seq = int(row["seq"]) + 1
            conn.execute(
                "INSERT INTO run_terminals "
                "(run_id,finished_at,status,terminal_sha256,terminal_json) "
                "VALUES (?,?,?,?,?)",
                (rid, finished_at, status, terminal_sha, terminal_json),
            )
            conn.execute(
                "INSERT INTO run_events (run_id,seq,at,status,detail_json) "
                "VALUES (?,?,?,?,?)",
                (rid, seq, finished_at, status,
                 _json_text({"terminal_sha256": terminal_sha})),
            )
        # Do not seed the verification cache from the sealing pass.  The first
        # historical numerical read performs an independent checksum pass.
        return self.get_run(rid)

    @staticmethod
    def _snapshot(request_row: sqlite3.Row, event_row: sqlite3.Row,
                  terminal_row: Optional[sqlite3.Row], *, active: bool,
                  detail: bool) -> dict:
        request: Optional[dict] = None
        if detail:
            request = RunLedger._decode_hashed_json(
                request_row["request_json"], request_row["request_sha256"],
                f"run {request_row['run_id']} request record",
            )
            request_spec = request.get("spec", {})
            execution = request.get("execution", {})
            if (
                request.get("run_id") != request_row["run_id"]
                or request.get("created_at") != request_row["created_at"]
                or request.get("output_dir") != request_row["output_dir"]
                or not isinstance(execution, dict)
                or execution.get("device") != request_row["device"]
                or not isinstance(request_spec, dict)
                or request_spec.get("sha256") != request_row["spec_sha256"]
            ):
                raise LedgerError(
                    f"run {request_row['run_id']} request columns do not match "
                    "its hashed record")
        event_status = str(event_row["status"])
        status = event_status
        if event_status in {"queued", "running"} and not active:
            status = "interrupted"
        terminal = None
        if terminal_row is not None:
            terminal = RunLedger._decode_hashed_json(
                terminal_row["terminal_json"], terminal_row["terminal_sha256"],
                f"run {request_row['run_id']} terminal record",
            )
            try:
                event_detail = json.loads(event_row["detail_json"])
            except json.JSONDecodeError as exc:
                raise LedgerError(
                    f"run {request_row['run_id']} terminal event is invalid JSON: {exc}") from exc
            if (
                terminal.get("run_id") != request_row["run_id"]
                or terminal.get("request_sha256") != request_row["request_sha256"]
                or terminal.get("status") != terminal_row["status"]
                or terminal.get("finished_at") != terminal_row["finished_at"]
                or event_status != terminal_row["status"]
                or not isinstance(event_detail, dict)
                or event_detail.get("terminal_sha256") != terminal_row["terminal_sha256"]
            ):
                raise LedgerError(
                    f"run {request_row['run_id']} terminal columns/event do not match "
                    "its hashed record")
        summary = terminal.get("summary") if terminal else None
        run_block = (summary or {}).get("run", {})
        error_detail = terminal.get("error") if terminal else None
        error_text = (error_detail.get("message") if isinstance(error_detail, dict)
                      else error_detail)
        result = {
            "run_id": request_row["run_id"],
            "created_at": request_row["created_at"],
            "started_at": None,
            "finished_at": terminal_row["finished_at"] if terminal_row else None,
            "status": status,
            "recorded_status": event_status,
            "device": request_row["device"],
            "output_dir": request_row["output_dir"],
            "spec_sha256": request_row["spec_sha256"],
            "request_sha256": request_row["request_sha256"],
            "terminal_sha256": (
                terminal_row["terminal_sha256"] if terminal_row else None),
            "error": error_text,
            "error_detail": error_detail,
            "summary": summary,
            # Additive flattened blocks keep the HTTP/UI contract convenient;
            # ``summary`` remains the immutable terminal document's exact view.
            "run": (summary or {}).get("run", {}),
            "grid": (summary or {}).get("grid", {}),
            "provenance": (summary or {}).get("provenance", {}),
            "monitors": (summary or {}).get("monitors", []),
            "aborted": bool(run_block.get("aborted", False)),
            "abort_reason": run_block.get("abort_reason") or None,
            "integrity": terminal.get("integrity") if terminal else None,
            "integrity_status": "sealed" if terminal else "unsealed",
        }
        # Started time is intentionally an event, never an UPDATE of the request.
        # The detailed reader fills it with one tiny indexed query below.
        if terminal:
            result["started_at"] = terminal.get("started_at")
        if detail:
            assert request is not None
            result["request"] = request
            result["terminal"] = terminal
            result["spec"] = request.get("spec", {}).get("document")
            result["manifest"] = summary
            result["latest_event"] = {
                "seq": int(event_row["seq"]), "at": event_row["at"],
                "status": event_status, "detail": json.loads(event_row["detail_json"]),
            }
        return result

    def get_run(self, run_id: str, *, active: bool = False) -> dict:
        rid = self.validate_run_id(run_id)
        with self._connect() as conn:
            request = self._request_row(conn, rid)
            event = conn.execute(
                "SELECT * FROM run_events WHERE run_id=? ORDER BY seq DESC LIMIT 1",
                (rid,),
            ).fetchone()
            terminal = conn.execute(
                "SELECT * FROM run_terminals WHERE run_id=?", (rid,),
            ).fetchone()
            if event is None:
                raise LedgerError(f"run {rid} has no lifecycle event")
            return self._snapshot(request, event, terminal, active=active, detail=True)

    def list_runs(self, *, limit: int = 100, cursor: Optional[str] = None,
                  active_run_id: Optional[str] = None) -> dict:
        limit = max(1, min(int(limit), 500))
        args: list[Any] = []
        where = ""
        if cursor:
            # Cursor is the exact (created_at,run_id) pair returned by this API.
            try:
                created, rid = cursor.split("|", 1)
                rid = self.validate_run_id(rid)
            except (ValueError, LedgerNotFound) as exc:
                raise LedgerError("invalid run-history cursor") from exc
            where = "WHERE (r.created_at < ? OR (r.created_at = ? AND r.run_id < ?))"
            args.extend([created, created, rid])
        args.append(limit + 1)
        query = f"""
            SELECT r.run_id, r.created_at, r.output_dir, r.device,
                   r.spec_sha256, r.request_sha256,
                   e.seq AS event_seq, e.at AS event_at,
                   e.status AS event_status, e.detail_json AS event_detail,
                   t.finished_at AS terminal_finished_at,
                   t.status AS terminal_status,
                   t.terminal_sha256, t.terminal_json
            FROM run_requests r
            JOIN run_events e ON e.run_id=r.run_id AND e.seq=(
                SELECT MAX(e2.seq) FROM run_events e2 WHERE e2.run_id=r.run_id
            )
            LEFT JOIN run_terminals t ON t.run_id=r.run_id
            {where}
            ORDER BY r.created_at DESC, r.run_id DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        more = len(rows) > limit
        rows = rows[:limit]
        runs = []
        for row in rows:
            event = {
                "seq": row["event_seq"], "at": row["event_at"],
                "status": row["event_status"], "detail_json": row["event_detail"],
            }
            terminal = None
            if row["terminal_json"] is not None:
                terminal = {
                    "finished_at": row["terminal_finished_at"],
                    "status": row["terminal_status"],
                    "terminal_sha256": row["terminal_sha256"],
                    "terminal_json": row["terminal_json"],
                }
            runs.append(self._snapshot(
                row, event, terminal,
                active=bool(active_run_id and row["run_id"] == active_run_id),
                detail=False,
            ))
        next_cursor = None
        if more and runs:
            tail = runs[-1]
            next_cursor = f"{tail['created_at']}|{tail['run_id']}"
        return {"runs": runs, "next_cursor": next_cursor, "root": str(self.root)}

    def output_dir(self, run_id: str) -> Path:
        rid = self.validate_run_id(run_id)
        with self._connect() as conn:
            row = self._request_row(conn, rid)
            return Path(row["output_dir"])

    def artifact(self, run_id: str, name: str) -> Optional[dict]:
        rid = self.validate_run_id(run_id)
        with self._verify_lock:
            index = self._artifact_records.get(rid)
        if index is None:
            record = self.get_run(rid)
            integrity = record.get("integrity") or {}
            index = {
                str(item["path"]): dict(item)
                for item in integrity.get("artifacts", [])
                if isinstance(item, dict) and item.get("path")
            }
            with self._verify_lock:
                self._artifact_records[rid] = index
        item = index.get(name)
        return dict(item) if item is not None else None

    def verify_artifact(
        self, run_id: str, name: str,
    ) -> tuple[int, int, int, int, Optional[int], str]:
        """Re-hash one selected artifact before historical numerical use."""
        expected = self.artifact(run_id, name)
        if expected is None:
            raise LedgerError(f"run {run_id} has no sealed artifact {name!r}")
        safe = _safe_artifact_name(name)
        path = self._trusted_output_dir(self.output_dir(run_id)) / safe
        try:
            file_stat = path.lstat()
            mode = file_stat.st_mode
        except OSError as exc:
            raise LedgerError(f"sealed artifact {safe!r} is missing: {exc}") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise LedgerError(f"sealed artifact {safe!r} is not a regular file")
        if file_stat.st_size != int(expected["bytes"]):
            raise LedgerError(f"sealed artifact {safe!r} failed size verification")
        if (expected.get("device") is not None
                and int(expected["device"]) != int(file_stat.st_dev)):
            raise LedgerError(f"sealed artifact {safe!r} was replaced (device changed)")
        if (expected.get("inode") is not None
                and int(expected["inode"]) != int(file_stat.st_ino)):
            raise LedgerError(f"sealed artifact {safe!r} was replaced (inode changed)")
        key = (self.validate_run_id(run_id), safe)
        with self._verify_lock:
            cached = self._verified.get(key)
        change_token = (int(file_stat.st_ctime_ns) if os.name != "nt"
                        else _windows_change_time(path))
        signature = (
            int(file_stat.st_dev), int(file_stat.st_ino), int(file_stat.st_size),
            int(file_stat.st_mtime_ns), change_token,
            str(expected["sha256"]),
        )
        cache_safe = change_token is not None
        if cache_safe and cached == signature:
            return signature
        hashed_stat, size, digest, _ = _hash_regular_file(path)
        if size != int(expected["bytes"]) or digest != expected["sha256"]:
            raise LedgerError(f"sealed artifact {safe!r} failed integrity verification")
        after_change = (int(hashed_stat.st_ctime_ns) if os.name != "nt"
                        else _windows_change_time(path))
        if cache_safe:
            if after_change is None or after_change != change_token:
                raise LedgerError(
                    f"sealed artifact {safe!r} changed while hashing")
        signature = (
            int(hashed_stat.st_dev), int(hashed_stat.st_ino),
            int(hashed_stat.st_size), int(hashed_stat.st_mtime_ns),
            after_change, str(expected["sha256"]),
        )
        if cache_safe:
            with self._verify_lock:
                self._verified[key] = signature
        return signature

    def verify_bundle_metadata(self, run_id: str) -> None:
        """Verify the small provenance-bearing files before loading a history item."""
        self.verify_artifact(run_id, "sim.json")
        self.verify_artifact(run_id, "manifest.json")

    def spec(self, run_id: str) -> dict:
        return dict(self.get_run(run_id)["request"]["spec"]["document"])
