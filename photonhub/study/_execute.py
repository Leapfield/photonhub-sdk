"""Build, deduplicate, execute and resume design-study samples."""

from __future__ import annotations

import math
import numbers
import shutil
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps

from .. import cloud as cloud_module
from ..cloud.batch import _TotalCeilingRefused, _spend_limit
from ..cloud.config import CloudError
from .._compat import caller_stacklevel
from ..cloud.run import CloudJobTimeout, DEFAULT_MAX_USD, _check_cloud_device_support
from ..components import Simulation
from ..data import RunResult
from ..runners.batch import Batch, _check_batch_name
from ..runners.local import SolverRunError
from ..spec_hash import _sha256_hex, spec_sha256
from ._outcome import RESERVED, SampleOutcome
from ._store import StudyStore
from .sampling import Samples
from .table import build_table, save


def model_sha256(sim: Simulation) -> str:
    return _sha256_hex(sim.model_dump_json().encode("utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("\r", " ")[:500]


def _validate(samples, build, measure=None, path=None):
    if not isinstance(samples, Samples):
        raise TypeError(f"samples must be Samples, got {type(samples).__name__}")
    if not callable(build):
        raise TypeError("build must be callable")
    if measure is not None and not callable(measure):
        raise TypeError("measure must be callable or None")
    if path is None:
        raise ValueError("path is required for a study run")


def _build_rows(samples, build, *, cloud=False, device=None):
    rows = []
    groups = {}
    first_error = None
    names = samples.sample_names
    for index, params in enumerate(samples):
        try:
            sim = build(params)
        except Exception as exc:  # one bad sample does not stop other rows
            if first_error is None:
                first_error = exc
            rows.append({"error": _error(exc)})
            continue
        if not isinstance(sim, Simulation):
            raise TypeError(f"build returned {type(sim).__name__}, expected a Simulation")
        try:
            sim.check_runnable()
            if cloud:
                _check_cloud_device_support(sim, device or "gpu")
        except Exception as exc:
            if first_error is None:
                first_error = exc
            rows.append({"error": f"not runnable: {_error(exc)}"})
            continue
        digest = spec_sha256(sim)
        if digest not in groups:
            groups[digest] = {"job": f"{names[index]}_{digest[:8]}",
                              "sim": sim, "rows": []}
        groups[digest]["rows"].append(index)
        rows.append({"sim": sim, "spec": digest, "model": model_sha256(sim)})
    if first_error is not None and not groups:
        raise first_error
    return rows, groups


def _decide(groups, jobs, retry_failed):
    reused, to_run, to_collect, skipped_failed = [], [], [], []
    for digest, group in groups.items():
        record = jobs.get(digest)
        job = group["job"]
        if record and record.get("status") == "succeeded":
            reused.append(job)
        elif (record and record.get("status") in ("submitted", "timed_out")
              and record.get("backend") == "cloud" and record.get("job_id")):
            to_collect.append(job)
        elif record and record.get("status") == "failed" and not retry_failed:
            skipped_failed.append(job)
        else:
            to_run.append(job)
    return tuple(reused), tuple(to_run), tuple(to_collect), tuple(skipped_failed)


def _recorded_names(groups, jobs):
    for digest, group in groups.items():
        if digest in jobs:
            job = _check_batch_name(jobs[digest]["job"])
            group["job"] = job


@dataclass(frozen=True)
class StudyPlan:
    """A read-only preview of a study run.

    ``num_samples`` counts rows, ``num_jobs`` unique solver inputs and
    ``num_build_failed`` rows whose builder failed or whose simulation was
    not runnable. ``reused`` lists succeeded
    jobs in the directory, ``to_run`` jobs that would run or submit,
    ``to_collect`` already submitted cloud jobs awaiting collection and
    ``skipped_failed`` failed jobs held unless retried. ``estimate_usd`` sums
    local cost estimates for ``to_run`` and is not a service quote.
    """

    num_samples: int
    num_jobs: int
    num_build_failed: int
    reused: tuple[str, ...]
    to_run: tuple[str, ...]
    to_collect: tuple[str, ...]
    skipped_failed: tuple[str, ...]
    estimate_usd: float


def _make_plan(samples, rows, groups, store, retry_failed):
    _recorded_names(groups, store.jobs)
    decisions = _decide(groups, store.jobs, retry_failed)
    by_job = {group["job"]: group["sim"] for group in groups.values()}
    return StudyPlan(len(samples), len(groups),
                     sum("error" in row for row in rows), *decisions,
                     sum(by_job[job].cost_estimate().usd for job in decisions[1]))


def plan(samples, build, *, path=None, retry_failed=False,
         cloud=False, device=None) -> StudyPlan:
    """Preview the jobs a study would run without writing or executing.

    ``samples`` supplies parameter rows and ``build(params)`` returns a
    Simulation for each row. ``path`` optionally reads an existing study
    directory for resume decisions. ``retry_failed`` includes failed jobs in
    ``to_run``. Build and runnability failures count as failed rows; if no row
    is runnable, the first error is raised. The estimate is local, not a
    cloud quote.
    ``cloud`` and ``device`` apply the same per-row checks as ``run``.
    This toolkit makes no physics claim about the resulting samples.
    """
    if not isinstance(samples, Samples):
        raise TypeError(f"samples must be Samples, got {type(samples).__name__}")
    if not callable(build):
        raise TypeError("build must be callable")
    rows, groups = _build_rows(samples, build, cloud=cloud, device=device)
    store = StudyStore(path)
    if path is not None:
        store.pin(samples, write=False)
    return _make_plan(samples, rows, groups, store, retry_failed)


def _measurement(measure, result, names):
    if measure is None:
        return {"status": "succeeded", "measured": {}, "error": "",
                "updated_at": _now()}
    try:
        values = measure(result)
        if not isinstance(values, Mapping):
            raise TypeError("measure must return a mapping")
        measured = {}
        for name, value in values.items():
            if not isinstance(name, str) or not name or name in RESERVED or name in names:
                raise ValueError(f"measured name {name!r} is empty, reserved or a parameter")
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"measured {name!r} must be a finite real number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"measured {name!r} must be a finite real number")
            measured[name] = number
        return {"status": "succeeded", "measured": measured, "error": "",
                "updated_at": _now()}
    except Exception as exc:
        return {"status": "measure_failed", "measured": {}, "error": _error(exc),
                "updated_at": _now()}


def _record(group, status, *, error="", backend="", job_id="", quote_usd=None,
            result=None, solver=None):
    version = (str(result.manifest.get("provenance", {}).get("solver_version", ""))
               if result is not None else "")
    return {"job": group["job"], "status": status, "error": error,
            "solver_version": version, "backend": backend, "job_id": job_id,
            "quote_usd": quote_usd, "solver": solver,
            "output_dir": f"runs/{group['job']}" if backend == "local" else "",
            "updated_at": _now()}


def _exclusive_run(function):
    @wraps(function)
    def locked(*args, **kwargs):
        if kwargs.get("path") is None:
            return function(*args, **kwargs)
        with StudyStore.directory_lock(kwargs["path"]):
            return function(*args, **kwargs)
    return locked


@_exclusive_run
def run(samples, build, *, path, measure=None, cloud=False, retry_failed=False,
        max_workers=1, progress=None, quiet=False, solver_path=None, device=None,
        timeout=None, solver=None, max_usd=DEFAULT_MAX_USD, max_usd_total=5.0,
        wait_timeout_s=None):
    """Run or resume a study and return one results-table row per sample.

    ``samples`` contains parameter rows; ``build(params)`` makes each row's
    Simulation and optional ``measure(result)`` returns named finite scalars.
    ``path`` stores samples, jobs, measurements and the latest table. Identical
    solver inputs run once. Succeeded jobs and measurements are reused; failed
    jobs and measurements are retried only with ``retry_failed=True``. Submitted
    or timed-out cloud jobs are collected on a call with ``cloud=True``.
    A local call leaves their submitted or timed-out status unchanged.
    Without ``measure``, succeeded rows have no stored measurement. A later
    call with ``measure`` reads stored output without running the solver again.

    ``cloud`` chooses cloud execution; otherwise ``solver_path`` selects the
    local solver. ``solver`` selects the cloud build and is recorded with each
    job and submission; resuming with a different cloud build is refused.
    ``max_workers``, ``device``, ``timeout`` (local) and
    ``wait_timeout_s`` (cloud) control the runners. ``progress`` receives study
    event dictionaries; ``quiet`` suppresses the one-line stderr job reports.
    The cloud path quotes every new job before the first submission. Its
    ``max_usd`` per-job ceiling defaults to $5, as for every cloud entry point;
    ``max_usd_total`` defaults to $5 and counts all submitted jobs recorded in
    this directory, including quotes from failed or locally retried jobs.
    Submission history is append-only, including ambiguous submit outcomes
    whose job id is unknown. Those failed jobs require ``retry_failed=True``
    to submit again; inspect ``ph.cloud.list_jobs()`` before retrying.
    A definite service refusal leaves rows pending with its error. If all new
    submissions are refused, the table is saved and a CloudError is raised.
    Both ceilings must be finite and non-negative. If either quote ceiling is
    exceeded during initial quoting, the new batch is refused before submission.
    Stale quotes are refreshed before each submission and checked against the
    remaining budget; a refresh refusal leaves that entry pending. A submitted
    job is collected without another submission.
    Cloud completion hooks record status before measurements run. An exclusive
    directory lock prevents concurrent runs on the same path. This toolkit
    makes no physics claim about the returned measurements.
    """
    _validate(samples, build, measure, path)
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if cloud:
        if max_usd is None or max_usd_total is None:
            raise ValueError("a cloud study is always quoted: max_usd and max_usd_total must be finite")
        _spend_limit(max_usd, label="max_usd")
        _spend_limit(max_usd_total, label="max_usd_total")
    store = StudyStore(path, create=True)
    if cloud:
        records = [record for record in store.jobs.values()
                   if record.get("backend") == "cloud"]
        records.extend(store.document["submissions"])
        for record in records:
            recorded_solver = record.get("solver")
            if recorded_solver != solver:
                guidance = ("resume with solver=None to collect existing jobs; "
                            "if the service requires an explicit solver for new jobs, "
                            "use a new study path" if recorded_solver is None else
                            f"use solver={recorded_solver!r} or a new study path")
                raise ValueError(
                    "cannot resume cloud study with a different solver: "
                    f"recorded solver={recorded_solver!r}, passed solver={solver!r}; "
                    f"{guidance}")
    store.pin(samples)
    rows, groups = _build_rows(samples, build, cloud=cloud, device=device)
    preview = _make_plan(samples, rows, groups, store, retry_failed)
    store.write_samples(samples)
    if progress is not None:
        progress({"event": "study_plan", "num_samples": preview.num_samples,
                  "num_jobs": preview.num_jobs, "num_reused": len(preview.reused),
                  "num_to_run": len(preview.to_run), "num_to_collect": len(preview.to_collect),
                  "num_skipped_failed": len(preview.skipped_failed),
                  "num_build_failed": preview.num_build_failed})
    by_job = {group["job"]: (digest, group) for digest, group in groups.items()}
    finished = 0
    total = len(preview.to_run) + (len(preview.to_collect) if cloud else 0)
    call_errors = {}
    refused = []

    def report(job, status, error=""):
        nonlocal finished
        finished += 1
        if progress is not None:
            progress({"event": "study_job", "job": job, "status": status,
                      "done": finished, "total": total})
        if not quiet:
            detail = f": {error}" if error else ""
            print(f"study: [{finished}/{total}] {job} {status}{detail}", file=sys.stderr)

    def measure_job(digest, group, result=None):
        rep_model = model_sha256(group["sim"])
        for index in group["rows"]:
            row = rows[index]
            key = f"{digest}:{row['model']}"
            previous = store.measurements.get(key)
            if previous and (previous.get("status") == "succeeded" or not retry_failed):
                continue
            if measure is None:
                continue
            try:
                data = result
                if data is None:
                    if store.jobs[digest]["backend"] == "cloud":
                        data = cloud_module.resume(store.jobs[digest]["job_id"],
                                                   simulation=row["sim"]).result()
                    else:
                        data = RunResult(store.path / "runs" / group["job"],
                                         simulation=row["sim"])
                elif row["model"] != rep_model:
                    data = RunResult(data.output_dir, simulation=row["sim"])
                measurement = _measurement(measure, data, samples.names)
            except CloudError as exc:
                call_errors[index] = _error(exc)
                continue
            except Exception as exc:
                measurement = {"status": "measure_failed", "measured": {},
                               "error": _error(exc), "updated_at": _now()}
            store.put_measurement(key, measurement)

    if cloud:
        committed = sum(float(entry["quote_usd"])
                        for entry in store.document["submissions"])
        remaining = float(max_usd_total) - committed
        if progress is not None:
            progress({"event": "study_quote", "max_usd": float(max_usd),
                      "max_usd_total": float(max_usd_total),
                      "committed_usd": committed})
        for job in preview.to_collect:
            digest, group = by_job[job]
            old = store.jobs[digest]
            try:
                data = cloud_module.resume(old["job_id"], simulation=group["sim"],
                                           wait_timeout_s=wait_timeout_s).result()
                status, error = "succeeded", ""
            except (SolverRunError, CloudJobTimeout, CloudError) as exc:
                data = None
                if isinstance(exc, CloudError) and exc.status_code == 404:
                    status = "failed"
                    error = f"the service no longer knows job {old['job_id']}"
                else:
                    status = ("timed_out" if isinstance(exc, CloudJobTimeout) else
                              old["status"] if isinstance(exc, CloudError) else "failed")
                    error = _error(exc)
            record = _record(group, status, error=error, backend="cloud", solver=solver,
                             job_id=old["job_id"], quote_usd=old.get("quote_usd"),
                             result=data)
            store.put_job(digest, record)
            if data is not None:
                measure_job(digest, group, data)
            report(job, status)
        if preview.to_run and remaining < -1e-9:
            raise CloudError(
                f"study ceiling max_usd_total (${float(max_usd_total):.4f}) "
                f"is below already submitted quotes (${committed:.4f}); nothing was submitted")
        if preview.to_run:
            batch = cloud_module.Batch({job: by_job[job][1]["sim"]
                                        for job in preview.to_run})
            submitted = {}

            def on_submit(job, job_id, usd):
                digest, group = by_job[job]

                def warn_record_failure(exc):
                    try:
                        warnings.warn(
                            f"could not record cloud job {job_id!r} in {store.path}: {_error(exc)}",
                            UserWarning, stacklevel=caller_stacklevel())
                    except Exception:
                        pass

                try:
                    store.put_submission(digest, job, job_id, usd, _now(), solver=solver)
                except Exception as exc:
                    warn_record_failure(exc)
                submitted[job] = (job_id, usd)
                try:
                    store.put_job(digest, _record(group, "submitted", backend="cloud", solver=solver,
                                                   job_id=job_id, quote_usd=usd))
                except Exception as exc:
                    warn_record_failure(exc)

            def on_complete(job, data, error):
                digest, group = by_job[job]
                old = store.jobs.get(digest, {})
                new_id, new_quote = submitted.get(job, ("", None))
                if isinstance(error, CloudError) and not new_id:
                    if (getattr(error, "submission_attempted", True) and
                            (error.status_code is None or error.status_code >= 500)):
                        quote = error.quote_usd
                        store.put_submission(digest, job, "", quote, _now(),
                                             outcome="unknown", solver=solver)
                        message = (f"submission outcome unknown: {_error(error)}; "
                                   "inspect ph.cloud.list_jobs() before retry; "
                                   "retry_failed=True resubmits")
                        store.put_job(digest, _record(group, "failed", error=message,
                                                      backend="cloud", solver=solver, quote_usd=quote))
                        report(job, "failed", message)
                    else:
                        reason = _error(error)
                        refused.append(reason)
                        for index in group["rows"]:
                            call_errors[index] = reason
                        report(job, "pending", reason)
                    return
                status = ("succeeded" if error is None else
                          "timed_out" if isinstance(error, CloudJobTimeout) else
                          "submitted" if isinstance(error, CloudError)
                          else "failed")
                record = _record(group, status, error=_error(error) if error else "",
                                 backend="cloud", solver=solver, job_id=new_id or old.get("job_id", ""),
                                 quote_usd=(new_quote if new_id else old.get("quote_usd")),
                                 result=data)
                store.put_job(digest, record)
                report(job, status)

            try:
                batch_result = batch.run(device=device or "gpu", solver=solver, max_workers=max_workers,
                                         wait_timeout_s=wait_timeout_s, max_usd=max_usd,
                                         max_usd_total=max(0.0, remaining), on_submit=on_submit,
                                         on_complete=on_complete)
            except _TotalCeilingRefused as exc:
                raise CloudError(
                    f"study ceiling max_usd_total (${float(max_usd_total):.4f}); "
                    f"committed spend from this directory (${committed:.4f}); "
                    f"this call's summed quote (${exc.total_quote_usd:.4f}); "
                    "nothing was submitted") from exc
            for job in preview.to_run:
                if job in batch_result:
                    digest, group = by_job[job]
                    measure_job(digest, group, batch_result[job])
    else:
        runs = (store.path / "runs").resolve()
        if not runs.is_relative_to(store.path.resolve()):
            raise ValueError("runs directory must stay inside the study directory")
        for job in preview.to_run:
            digest, group = by_job[job]
            old = store.jobs.get(digest)
            if old and old.get("status") == "failed":
                target = (runs / job).resolve()
                if target.parent != runs or target.name != job or (runs / job).is_symlink():
                    raise ValueError(f"job {job!r} has an unsafe output directory")
                if target.exists():
                    shutil.rmtree(target)
        if preview.to_run:
            batch = Batch({job: by_job[job][1]["sim"] for job in preview.to_run})

            def on_complete(job, data, error):
                digest, group = by_job[job]
                status = "succeeded" if error is None else "failed"
                store.put_job(digest, _record(group, status,
                                               error=_error(error) if error else "",
                                               backend="local", result=data))
                if data is not None:
                    measure_job(digest, group, data)
                report(job, status)

            batch.run(path_dir=store.path / "runs", solver_path=solver_path,
                      max_workers=max_workers, timeout=timeout, device=device,
                      on_complete=on_complete)
    for job in preview.reused:
        digest, group = by_job[job]
        measure_job(digest, group)
    outcomes = []
    for index, row in enumerate(rows):
        if "error" in row:
            outcomes.append(SampleOutcome("build_failed", error=row["error"]))
            continue
        digest = row["spec"]
        group = groups[digest]
        record = store.jobs.get(digest, {})
        status = record.get("status", "pending")
        measurement = store.measurements.get(f"{digest}:{row['model']}", {})
        if status == "succeeded":
            status = measurement.get("status", "succeeded")
        if index in call_errors:
            status = "pending"
        outcomes.append(SampleOutcome(
            status, job=record.get("job", group["job"]), spec_sha256=digest,
            measured=measurement.get("measured", {}) if status == "succeeded" else {},
            error=(call_errors[index] if index in call_errors else
                   measurement.get("error", "") if status == "measure_failed" else
                   record.get("error", "")),
            solver_version=record.get("solver_version", ""),
            backend=record.get("backend", ""), job_id=record.get("job_id", ""),
            quote_usd=record.get("quote_usd")))
    table = build_table(samples, outcomes)
    save(table, store.path / "table.json")
    if progress is not None:
        progress({"event": "study_done",
                  "num_succeeded": sum(o.status == "succeeded" for o in outcomes),
                  "num_failed": sum(o.status in ("failed", "measure_failed", "build_failed")
                                    for o in outcomes),
                  "num_pending": sum(o.status == "pending" for o in outcomes),
                  "num_samples": len(samples)})
    if refused and len(refused) == len(preview.to_run):
        noun = "submission" if len(refused) == 1 else "submissions"
        raise CloudError(f"{len(refused)} {noun} refused; first reason: {refused[0]}")
    return table
