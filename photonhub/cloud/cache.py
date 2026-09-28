"""Validated, atomic local cache for cloud result bundles.

A cache entry is reusable only after the downloaded archive has passed the
wire-format resource limits, its manifest references a complete set of safe
raw files, and a private completion marker has been written. The marker is not
accepted from an archive, so a crash or hostile payload cannot make a partial
directory look complete.

Beside those result directories the client also keeps the wire spec it
submitted under each job id (:func:`store_spec`). ``ph.cloud.resume`` is handed
an id and nothing else, so without it the reader cannot know the simulation and
leaves every frequency-domain array on the engine's per-unit-amplitude
normalization (NUMERICS.md section 12) while the submitting call returns
absolute watts and V/m for the same paid job.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Optional

from ..bundle import (
    BundleError,
    COMPLETION_MARKER,
    extract_bundle_file,
)
from ._ids import validate_job_id
from .client import HttpClient
from .config import CloudConfig

_CACHE_LOCKS = tuple(threading.Lock() for _ in range(64))
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


#: Submitted wire specs live beside the job directories under a name no job id
#: can take: ``validate_job_id`` requires a leading letter or digit, so this
#: store can never collide with a result directory. Keeping the specs OUTSIDE
#: the job directory also keeps them out of the way of ``download_bundle``,
#: which publishes a result by atomically renaming a freshly extracted
#: directory over whatever stood there.
_SPEC_DIR = ".specs"
#: The client state of each submitted simulation (the user-frame origin, the
#: symmetry-plane record, declared ports: what the wire spec does not carry),
#: beside the specs and bound to the simulation it belongs to; same naming
#: rule, so no collision.
_CLIENT_STATE_DIR = ".client-state"


def job_dir(cfg: CloudConfig, job_id: str) -> Path:
    return Path(cfg.cache_dir) / validate_job_id(job_id)


def spec_path(cfg: CloudConfig, job_id: str) -> Path:
    """Where the wire spec submitted as ``job_id`` is kept."""
    return Path(cfg.cache_dir) / _SPEC_DIR / f"{validate_job_id(job_id)}.json"


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
    array into memory. :class:`RunResult` performs the monitor-type and
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


def store_spec(cfg: CloudConfig, job_id: str, spec: dict) -> Optional[Path]:
    """Keep the wire spec submitted as ``job_id``, written atomically.

    This is what lets ``ph.cloud.resume(job_id)`` return the same absolute
    values as the call that submitted the job: the reader needs the simulation
    to undo the engine's per-unit-amplitude normalization, and a resume has
    only an id. Best effort by design. A cache directory that cannot be
    written costs the amplitude restoration on a later resume, never the
    submitted job, so this warns and returns None instead of raising.
    """
    out = spec_path(cfg, job_id)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        handle_fd, tmp_name = tempfile.mkstemp(
            dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                json.dump(spec, handle)
                handle.write("\n")
            os.replace(tmp, out)
        except BaseException:
            _remove_path(tmp)
            raise
    except (OSError, TypeError, ValueError) as exc:
        warnings.warn(
            f"could not cache the spec of cloud job {job_id} ({exc}); "
            "ph.cloud.resume of this job will read the engine's "
            "unit-amplitude normalization instead of absolute watts and V/m",
            UserWarning, stacklevel=2)
        return None
    return out


def stored_spec(cfg: CloudConfig, job_id: str) -> Optional[Path]:
    """The wire spec kept for ``job_id``, or None when none was: a job
    submitted before this store existed, from another machine, or with a cache
    directory that could not be written."""
    return stored_spec_for_result(job_dir(cfg, job_id))


def stored_spec_for_result(result_dir) -> Optional[Path]:
    """The wire spec kept for the job whose downloaded result is ``result_dir``.

    The reverse of :func:`spec_path`, for a reader holding a path and no
    :class:`CloudConfig`: the Workbench opens a job's cache directory by name
    (``photonhub.viz.service.load_result``), and a result directory names its
    job. Returns None for a directory that is not a cloud job entry, or one
    whose spec was never stored.

    The caller decides whether the recovered document is the one this result
    ran; a directory name alone is not evidence, and a spec that is not the
    executed input would silently rescale the numbers it restores.
    """
    result_dir = Path(result_dir)
    try:
        job_id = validate_job_id(result_dir.name)
    except ValueError:
        return None
    path = result_dir.parent / _SPEC_DIR / f"{job_id}.json"
    return path if (not path.is_symlink() and path.is_file()) else None


def client_state_path(cfg: CloudConfig, job_id: str) -> Path:
    """Where the client state of the simulation submitted as ``job_id`` is kept."""
    return Path(cfg.cache_dir) / _CLIENT_STATE_DIR / f"{validate_job_id(job_id)}.json"


def store_client_state(cfg: CloudConfig, job_id: str, sim,
                       *, wire: Optional[dict] = None) -> Optional[Path]:
    """Keep ``sim``'s client state for ``job_id`` (bound to its wire
    document), so ``ph.cloud.resume(job_id)``, and a ``RunResult`` of the
    job's cache directory, read the result in the frame the submitting call
    returned it in: user-frame coordinates, a plane across a symmetry plane
    made whole, the declared ports. Nothing is written for a simulation
    without client state. Best effort, like :func:`store_spec`: a failure
    warns and costs a later reload its frame, never the job."""
    from ..components.frame import client_state, write_json_atomic

    try:
        state = client_state(sim, wire=wire)
        if state is None:
            return None
        return write_json_atomic(client_state_path(cfg, job_id), state)
    except (OSError, TypeError, ValueError) as exc:
        warnings.warn(
            f"could not cache the client state of cloud job {job_id} ({exc}); "
            "ph.cloud.resume of this job will read in the wire's corner frame, "
            "with no symmetry unfolding and no declared ports",
            UserWarning, stacklevel=2)
        return None


def stored_client_state(cfg: CloudConfig, job_id: str) -> Optional[Path]:
    """The client state kept for ``job_id``, or None when none was (a
    hand-built scene, or a job submitted without this store)."""
    path = client_state_path(cfg, job_id)
    return path if (not path.is_symlink() and path.is_file()) else None


def stored_client_state_for_result(result_dir) -> Optional[Path]:
    """The client state kept for the job whose downloaded result is
    ``result_dir``, for a reader holding a path and no :class:`CloudConfig`
    (``RunResult`` of a cache directory). None when the directory is not a
    cloud job entry or no record was kept. The record carries its own
    binding to the simulation it belongs to, which the reader checks: a
    directory name alone is not evidence."""
    result_dir = Path(result_dir)
    try:
        job_id = validate_job_id(result_dir.name)
    except ValueError:
        return None
    path = result_dir.parent / _CLIENT_STATE_DIR / f"{job_id}.json"
    return path if (not path.is_symlink() and path.is_file()) else None


def invalidate(cfg: CloudConfig, job_id: str) -> None:
    """Remove an unreadable cached result so a later resume can re-fetch it.

    The stored spec stays: the re-fetched result needs it for exactly the same
    reason the discarded one did."""
    _remove_path(job_dir(cfg, job_id))


def completed_result(cfg: CloudConfig, job_id: str) -> Optional[Path]:
    """The validated, sealed local result directory for ``job_id``, or None.

    A directory is returned only when a prior download passed the full
    structural validation and published the private completion marker, so the
    entry is exactly what :func:`download_bundle` would return without
    touching the network."""
    out = job_dir(cfg, job_id)
    lock = _CACHE_LOCKS[hash(str(out)) % len(_CACHE_LOCKS)]
    with lock:
        return out if _is_complete(out) else None


def _download_archive(http: HttpClient, cfg: CloudConfig, job_id: str,
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


def download_bundle(http: HttpClient, cfg: CloudConfig, job_id: str) -> Path:
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
