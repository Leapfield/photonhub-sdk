"""Atomic records for a resumable study directory."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from ..runners.batch import _check_batch_name


STUDY_FORMAT = "photonhub.study/1"
JOBS_FORMAT = "photonhub.study_jobs/1"


def _write_json(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                      dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _read_jobs(path: Path) -> dict:
    if not path.exists():
        return {"format": JOBS_FORMAT, "jobs": {}, "measurements": {},
                "submissions": []}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"corrupt jobs.json: {exc}") from exc
    if isinstance(document, dict) and "submissions" not in document:
        raise ValueError("jobs.json is missing submissions history")
    if (not isinstance(document, dict) or document.get("format") != JOBS_FORMAT
            or not isinstance(document.get("jobs"), dict)
            or not isinstance(document.get("measurements"), dict)
            or not isinstance(document.get("submissions"), list)):
        raise ValueError(f"jobs.json must have format {JOBS_FORMAT}")
    if any(not isinstance(key, str) or not isinstance(value, dict)
           for section in ("jobs", "measurements")
           for key, value in document[section].items()):
        raise ValueError("jobs.json has an invalid job or measurement record")
    for digest, record in document["jobs"].items():
        if (record.get("status") not in ("succeeded", "failed", "submitted", "timed_out")
                or record.get("backend") not in ("local", "cloud")
                or not isinstance(record.get("job"), str)
                or not isinstance(record.get("job_id", ""), str)):
            raise ValueError(f"jobs.json has an invalid job record for {digest!r}")
        try:
            _check_batch_name(record["job"])
        except ValueError as exc:
            raise ValueError(f"jobs.json has an unsafe job name for {digest!r}") from exc
        runs = (path.parent / "runs").resolve()
        target = (runs / record["job"]).resolve()
        if (not runs.is_relative_to(path.parent.resolve()) or target.parent != runs
                or (runs / record["job"]).is_symlink()):
            raise ValueError(f"jobs.json has an unsafe job name for {digest!r}")
        if record["backend"] == "cloud" and record["status"] in ("submitted", "timed_out"):
            if not record.get("job_id"):
                raise ValueError(f"jobs.json has a cloud job without a job_id for {digest!r}")
        quote = record.get("quote_usd")
        if quote is not None and (isinstance(quote, bool) or not isinstance(quote, (int, float))
                                  or not math.isfinite(float(quote)) or quote < 0):
            raise ValueError(f"jobs.json has an invalid quote_usd for {digest!r}")
        if record["backend"] == "cloud" and record.get("job_id") and quote is None:
            raise ValueError(f"jobs.json has a cloud job without quote_usd for {digest!r}")
    for key, record in document["measurements"].items():
        if (record.get("status") not in ("succeeded", "measure_failed")
                or not isinstance(record.get("measured"), dict)):
            raise ValueError(f"jobs.json has an invalid measurement for {key!r}")
    for entry in document["submissions"]:
        if (not isinstance(entry, dict) or
                (set(entry) - {"solver"}) not in ({"digest", "job", "job_id", "quote_usd", "at"},
                                   {"digest", "job", "job_id", "quote_usd", "at", "outcome"})):
            raise ValueError("jobs.json has an invalid submission")
        if (not isinstance(entry["digest"], str) or len(entry["digest"]) != 64
                or any(char not in "0123456789abcdef" for char in entry["digest"])
                or not isinstance(entry["job_id"], str)
                or (not entry["job_id"] and entry.get("outcome") != "unknown")
                or (entry.get("outcome") == "unknown" and entry["job_id"])
                or ("outcome" in entry and entry["outcome"] != "unknown")
                or not isinstance(entry["at"], str) or not entry["at"]):
            raise ValueError("jobs.json has an invalid submission")
        try:
            _check_batch_name(entry["job"])
        except ValueError as exc:
            raise ValueError("jobs.json has an invalid submission") from exc
        quote = entry["quote_usd"]
        if (isinstance(quote, bool) or not isinstance(quote, (int, float))
                or not math.isfinite(float(quote)) or quote < 0):
            raise ValueError("jobs.json has an invalid submission")
    return document


class StudyStore:
    """Read and update a single study directory."""

    def __init__(self, path, *, create=False):
        self.path = Path(path) if path is not None else None
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.document = (_read_jobs(self.path / "jobs.json") if self.path is not None
                         else {"format": JOBS_FORMAT, "jobs": {}, "measurements": {},
                               "submissions": []})

    @staticmethod
    @contextmanager
    def directory_lock(path):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a+b") as handle:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"study directory {directory} is already in use") from exc
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @property
    def jobs(self):
        return self.document["jobs"]

    @property
    def measurements(self):
        return self.document["measurements"]

    def pin(self, samples, *, write=True) -> None:
        from .. import __version__

        definitions = [param.to_dict() for param in samples.parameters]
        target = self.path / "study.json"
        if target.exists():
            try:
                document = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"corrupt study.json: {exc}") from exc
            if (not isinstance(document, dict) or
                    document.get("format") != STUDY_FORMAT or
                    not isinstance(document.get("parameters"), list)):
                raise ValueError(f"study.json must have format {STUDY_FORMAT}")
            previous = document["parameters"]
            for index in range(max(len(previous), len(definitions))):
                old = previous[index] if index < len(previous) else None
                new = definitions[index] if index < len(definitions) else None
                if old != new:
                    name = (new or old).get("name", index) if isinstance(new or old, dict) else index
                    raise ValueError(
                        f"parameter {name!r} differs from study.json: "
                        f"old={old!r}, new={new!r}")
        elif write:
            _write_json(target, {"format": STUDY_FORMAT,
                                 "parameters": definitions,
                                 "photonhub_version": __version__})

    def write_samples(self, samples) -> None:
        _write_json(self.path / "samples.json", samples.to_dict())

    def put_job(self, digest: str, record: dict) -> None:
        with self.lock:
            self.jobs[digest] = record
            _write_json(self.path / "jobs.json", self.document)

    def put_submission(self, digest: str, job: str, job_id: str,
                       quote_usd: float, at: str, *, outcome=None, solver=None) -> None:
        with self.lock:
            entry = {
                "digest": digest, "job": job, "job_id": job_id,
                "quote_usd": quote_usd, "at": at, "solver": solver}
            if outcome is not None:
                entry["outcome"] = outcome
            self.document["submissions"].append(entry)
            _write_json(self.path / "jobs.json", self.document)

    def put_measurement(self, key: str, record: dict) -> None:
        with self.lock:
            self.measurements[key] = record
            _write_json(self.path / "jobs.json", self.document)
