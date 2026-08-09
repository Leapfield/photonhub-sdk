"""Validated, atomic local cache for cloud result bundles.

A cache entry is reusable only after the downloaded archive has passed the
wire-format resource limits, its manifest references a complete set of safe
raw files, and a private completion marker has been written. The marker is not
accepted from an archive, so a crash or hostile payload cannot make a partial
directory look complete.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

from ..bundle import (
    BundleError,
    COMPLETION_MARKER,
    extract_bundle_file,
)
from ._ids import validate_job_id
from .client import HttpClient
from .config import WebConfig

_CACHE_LOCKS = tuple(threading.Lock() for _ in range(64))
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def job_dir(cfg: WebConfig, job_id: str) -> Path:
    return Path(cfg.cache_dir) / validate_job_id(job_id)


def _safe_leaf(value, label: str) -> str:
    if (not isinstance(value, str) or not value
            or value in (".", "..", COMPLETION_MARKER)
            or "/" in value or "\\" in value or "\x00" in value):
        raise BundleError(f"invalid {label} in result manifest: {value!r}")
    return value


def _shape_bytes(value, name: str) -> int:
    if not isinstance(value, list) or not value:
        raise BundleError(
            f"monitor {name!r} has an invalid shape in result manifest")
    count = 1
    for dimension in value:
        if (isinstance(dimension, bool) or not isinstance(dimension, int)
                or dimension < 0):
            raise BundleError(
                f"monitor {name!r} has an invalid shape in result manifest")
        count *= dimension
    return count * 4  # the output contract stores little-endian float32


def _validate_payload(out: Path) -> None:
    """Validate the cheap, structural portion of the raw output contract.

    This deliberately uses file metadata rather than reading every monitor
    array into memory. :class:`SimulationData` performs the monitor-type and
    coordinate-level checks lazily when callers access data.
    """
    if out.is_symlink() or not out.is_dir():
        raise BundleError("result cache entry is not a safe directory")
    manifest_path = out / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise BundleError("result bundle has no regular manifest.json")
    try:
        if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise BundleError(
                "result manifest exceeds the 64 MiB safety limit")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except BundleError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise BundleError(f"result bundle has an invalid manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BundleError("result bundle manifest must be a JSON object")
    monitors = manifest.get("monitors", [])
    if not isinstance(monitors, list):
        raise BundleError("result bundle manifest 'monitors' must be a list")

    names: set[str] = set()
    files: set[str] = set()
    for entry in monitors:
        if not isinstance(entry, dict):
            raise BundleError("result manifest monitor entries must be objects")
        name = _safe_leaf(entry.get("name"), "monitor name")
        filename = _safe_leaf(entry.get("file"), f"file for monitor {name!r}")
        if filename == "manifest.json":
            raise BundleError(
                f"monitor {name!r} uses the reserved manifest filename")
        if name in names:
            raise BundleError(f"duplicate monitor name in result manifest: {name!r}")
        if filename in files:
            raise BundleError(f"duplicate monitor file in result manifest: {filename!r}")
        if entry.get("dtype", "float32") != "float32":
            raise BundleError(
                f"monitor {name!r} has unsupported result dtype")
        expected_bytes = _shape_bytes(entry.get("shape"), name)
        data_path = out / filename
        if data_path.is_symlink() or not data_path.is_file():
            raise BundleError(
                f"monitor {name!r} result file is missing or unsafe")
        try:
            actual_bytes = data_path.stat().st_size
        except OSError as exc:
            raise BundleError(
                f"monitor {name!r} result file is unreadable") from exc
        if actual_bytes != expected_bytes:
            raise BundleError(
                f"monitor {name!r} result file has {actual_bytes} bytes; "
                f"manifest shape requires {expected_bytes}")
        names.add(name)
        files.add(filename)


def _is_complete(out: Path) -> bool:
    marker = out / COMPLETION_MARKER
    if (out.is_symlink() or marker.is_symlink() or not marker.is_file()):
        return False
    try:
        _validate_payload(out)
    except (BundleError, OSError):
        return False
    return True


def _remove_path(path: Path) -> None:
    """Remove a cache path without ever following a directory symlink."""
    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    else:
        shutil.rmtree(path, ignore_errors=True)


def invalidate(cfg: WebConfig, job_id: str) -> None:
    """Remove an unreadable cached result so a later resume can re-fetch it."""
    _remove_path(job_dir(cfg, job_id))


def _download_archive(http: HttpClient, cfg: WebConfig, job_id: str,
                      archive: Path) -> None:
    stream = getattr(http, "download_result_to", None)
    if callable(stream):
        stream(job_id, archive, max_bytes=cfg.max_bundle_download_bytes)
        return

    # Compatibility for injected/test transports implementing the original
    # byte-returning protocol. The production HttpClient always streams.
    data = http.download_result(job_id)
    if not isinstance(data, bytes):
        raise BundleError("result download did not return bytes")
    if len(data) > cfg.max_bundle_download_bytes:
        raise BundleError(
            "result bundle exceeds the configured download-byte limit")
    archive.write_bytes(data)


def download_bundle(http: HttpClient, cfg: WebConfig, job_id: str) -> Path:
    """Return a locally cached, validated result directory for ``job_id``."""
    out = job_dir(cfg, job_id)
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)

    # The stripe prevents same-process threads from racing over a stale entry;
    # random private paths plus atomic rename preserve cross-process safety.
    lock = _CACHE_LOCKS[hash(str(out)) % len(_CACHE_LOCKS)]
    with lock:
        if _is_complete(out):
            return out

        archive_fd, archive_name = tempfile.mkstemp(
            dir=parent, prefix=f".{out.name}.download-", suffix=".tar.gz")
        os.close(archive_fd)
        archive = Path(archive_name)
        tmp = Path(tempfile.mkdtemp(dir=parent, prefix=f"{out.name}.tmp-"))
        try:
            _download_archive(http, cfg, job_id, archive)
            extract_bundle_file(
                archive,
                tmp,
                max_compressed_bytes=cfg.max_bundle_download_bytes,
                max_expanded_bytes=cfg.max_bundle_extract_bytes,
                max_members=cfg.max_bundle_members,
            )
            _validate_payload(tmp)
            (tmp / COMPLETION_MARKER).write_text(
                "photonhub-result-v1\n", encoding="ascii")

            try:
                os.replace(tmp, out)
            except OSError:
                # A different process may have completed the same result while
                # this one downloaded. Never replace its validated winner.
                if _is_complete(out):
                    return out
                _remove_path(out)
                try:
                    os.replace(tmp, out)
                except OSError:
                    if _is_complete(out):
                        return out
                    raise
            return out
        finally:
            try:
                archive.unlink()
            except FileNotFoundError:
                pass
            _remove_path(tmp)
