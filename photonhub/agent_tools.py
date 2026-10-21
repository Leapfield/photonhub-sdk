"""Local, read-only contracts for agents inspecting specs and results."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import __version__, cost, schema
from . import capabilities as capability_rules
from .components.simulation import SCHEMA_VERSION, Simulation
from .data import RunResult
from .error_codes import CodedError, issue, issue_from_finding
from .runners.local import _execution_wire_json
from .runners.phsolver import (
    SolverRunError,
    _solver_subprocess_env,
    device_args,
    find_solver,
)
from .viz import checks, service

__all__ = [
    "Capabilities",
    "ConvergenceLocation",
    "ConvergenceSummary",
    "CpuOnlyInfo",
    "DeviceInfo",
    "EngineStatus",
    "EstimateAssumptions",
    "Issue",
    "LocalEstimate",
    "ResultMonitor",
    "ResultSummary",
    "SafetyInfo",
    "SchemaInfo",
    "SecondsRange",
    "SolverInfo",
    "ValidationReport",
    "capabilities",
    "convergence_report",
    "estimate",
    "get_schema",
    "result_summary",
    "validate_spec",
]


class StrictModel(BaseModel):
    """Forbid unknown fields in local, read-only tool output."""
    model_config = ConfigDict(extra="forbid")


class Issue(StrictModel):
    """A stable finding from a local, read-only tool."""
    id: str
    severity: str
    path: str | None = None
    monitor: str | None = None
    message: str
    fix: str
    detail: dict[str, Any] | None = None


class SchemaInfo(StrictModel):
    """Local, read-only schema output for photonhub.schema/1."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contract: str = "photonhub.schema/1"
    schema_version: str
    schema_major: int
    schema_sha256: str
    definitions: list[str]
    definition: str | None
    schema_: dict[str, Any] = Field(alias="schema")


class SolverInfo(StrictModel):
    """Solver facts from local, read-only capability probes."""
    found: bool
    path: str | None = None
    version: str | None = None
    git_sha: str | None = None
    gpu_available: bool | None = None
    gpu_platform: str | None = None
    openmp: bool | None = None
    devices: list[Any] | None = None
    features: list[str] | None = None
    feature_drift: list[str] | None = None
    error: str | None = None


class DeviceInfo(StrictModel):
    """Availability of one local device selector."""
    selector: str
    available: bool
    note: str


class CpuOnlyInfo(StrictModel):
    """A feature restricted to the local CPU solver."""
    name: str
    section: str
    message: str


class SafetyInfo(StrictModel):
    """Side effect flags for a local, read-only capability report."""
    spends_money: bool = False
    network: bool = False
    cloud_submission: bool = False


class Capabilities(StrictModel):
    """Local, read-only capability output for photonhub.capabilities/1."""
    contract: str = "photonhub.capabilities/1"
    sdk_version: str
    schema_version: str
    schema_major: int
    solver: SolverInfo
    devices: list[DeviceInfo]
    cpu_only_features: list[CpuOnlyInfo]
    pinned_features: list[str]
    safety: SafetyInfo = Field(default_factory=SafetyInfo)


class EngineStatus(StrictModel):
    """Compact outcome of a local solver validation probe."""
    ran: bool = False
    solver: str | None = None
    result: Any = None


class ValidationReport(StrictModel):
    """Local validation output for photonhub.validation/1."""
    contract: str = "photonhub.validation/1"
    ok: bool
    authoritative: bool
    device: str
    schema_version: str
    input_sha256: str | None = None
    input_sha256_basis: str | None = None
    stages: dict[str, str]
    issues: list[Issue]
    counts: dict[str, int]
    engine: EngineStatus


class SecondsRange(StrictModel):
    """Estimated local CPU time bounds in seconds."""
    low: float
    high: float


class EstimateAssumptions(StrictModel):
    """Throughput assumptions for a local, read-only estimate."""
    cpu_mcells_per_s: list[float]
    basis: str


class LocalEstimate(StrictModel):
    """Local, read-only planning output for photonhub.estimate/1."""
    contract: str = "photonhub.estimate/1"
    cells_per_axis: tuple[int, int, int]
    num_cells: int
    num_steps: int
    dt_s: float
    cell_steps: int
    tcell_steps: float
    memory_bytes: int
    output_bytes: int
    cpu_seconds: SecondsRange
    assumptions: EstimateAssumptions
    notes: list[str]


class ResultMonitor(StrictModel):
    """One recorded monitor in a local result summary."""
    name: str
    type: str | None = None
    kind: str | None = None
    shape: list[int] | None = None
    components: list[str] | None = None
    n_freqs: int | None = None
    axis: str | None = None


class ConvergenceLocation(StrictModel):
    """Location of a report saved beside the local result."""
    found: bool
    path: str | None = None


class ResultSummary(StrictModel):
    """Local, read-only result output for photonhub.result_summary/1."""
    contract: str = "photonhub.result_summary/1"
    output_dir: str
    input_sha256: str | None = None
    input_sha256_source: str | None = None
    geometry_status: str
    manifest_version: str | None = None
    schema_version: str | None = None
    solver: dict[str, Any]
    device: dict[str, Any]
    run: dict[str, Any]
    grid: dict[str, Any]
    monitors: list[ResultMonitor]
    issues: list[Issue]
    counts: dict[str, int]
    convergence: ConvergenceLocation


class ConvergenceSummary(StrictModel):
    """Local, read-only output for photonhub.convergence_report/1."""
    contract: str = "photonhub.convergence_report/1"
    found: bool
    path: str | None = None
    relation: str | None = None
    format: str | None = None
    converged: bool | None = None
    metric_name: str | None = None
    stop_reason: str | None = None
    tol: float | None = None
    patience: int | None = None
    finest: float | None = None
    drift_successive: float | None = None
    drift_total: float | None = None
    total_cost_usd: float | None = None
    ladder: list[dict[str, Any]] | None = None
    keys: list[str] | None = None
    message: str | None = None


def _counts(issues: list[dict]) -> dict[str, int]:
    return {severity: sum(item["severity"] == severity for item in issues)
            for severity in ("error", "warning", "info")}


def _load_spec(spec: str | Path | dict | Simulation) -> dict:
    if isinstance(spec, Simulation):
        return spec.to_wire_dict()
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, Path):
        path = spec
    elif isinstance(spec, str):
        stripped = spec.lstrip()
        if stripped.startswith(("{", "[")) or not stripped.lower().endswith(".json"):
            try:
                value = json.loads(spec)
            except json.JSONDecodeError as exc:
                raise CodedError(
                    "spec-not-json",
                    f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
                ) from exc
            if not isinstance(value, dict):
                raise CodedError("spec-not-json", "Simulation spec must be a JSON object")
            return value
        path = Path(spec).expanduser()
    else:
        raise CodedError("spec-not-json", "Simulation spec must be a JSON object or .json file")
    if path.suffix.lower() != ".json":
        raise CodedError("spec-not-json", "Simulation spec must be a JSON object or .json file")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CodedError("spec-unreadable", str(exc)) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CodedError(
            "spec-not-json",
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from exc
    if not isinstance(value, dict):
        raise CodedError("spec-not-json", "Simulation spec must be a JSON object")
    return value


def get_schema(definition: str | None = None) -> SchemaInfo:
    """Return the local, read-only photonhub.schema/1 schema document."""
    text = schema.schema_text()
    document = json.loads(text)
    names = sorted(document.get("$defs", {}))
    if definition is not None and definition not in names:
        raise CodedError("unknown-definition", f"Unknown schema definition: {definition}",
                         f"Choose one of: {', '.join(names)}")
    return SchemaInfo(schema_version=SCHEMA_VERSION, schema_major=int(SCHEMA_VERSION.split('.')[0]),
                      schema_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      definitions=names, definition=definition,
                      schema=document if definition is None else document["$defs"][definition])


def capabilities() -> Capabilities:
    """Return local, read-only photonhub.capabilities/1 solver facts."""
    solver = SolverInfo(found=False)
    try:
        path = find_solver()
        if path is not None:
            solver.found = True
            solver.path = str(path)
            def probe(arg: str) -> dict:
                completed = subprocess.run([str(path), arg], capture_output=True, text=True,
                                           timeout=30, env=_solver_subprocess_env(), check=False)
                if completed.returncode:
                    raise RuntimeError(completed.stderr.strip() or completed.stdout.strip()
                                       or f"{arg} exited {completed.returncode}")
                return json.loads(completed.stdout)

            info = probe("info")
            solver.version = info.get("version")
            solver.git_sha = info.get("git_sha")
            solver.gpu_available = info.get("gpu") is True
            solver.gpu_platform = info.get("gpu_platform")
            solver.openmp = info.get("cpu", {}).get("openmp")
            solver.devices = info.get("devices")
            advertised = probe("--capabilities")
            solver.features = advertised.get("features")
            solver.feature_drift = sorted(
                set(solver.features or []) ^ capability_rules.ENGINE_ADVERTISED_FEATURES)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError,
            subprocess.SubprocessError) as exc:
        solver.error = f"{type(exc).__name__}: {exc}"
    if not solver.found and solver.error is None:
        solver.error = "Local solver not found"
    devices = [DeviceInfo(selector="cpu", available=solver.found,
                          note="Local solver available" if solver.found else "Local solver not found"),
               DeviceInfo(selector="gpu", available=solver.gpu_available is True and bool(solver.devices),
                          note="Reported by local solver" if solver.gpu_available and solver.devices
                          else "GPU unavailable or no device reported")]
    return Capabilities(sdk_version=__version__, schema_version=SCHEMA_VERSION,
                        schema_major=int(SCHEMA_VERSION.split('.')[0]), solver=solver,
                        devices=devices,
                        cpu_only_features=[CpuOnlyInfo(name=f.name, section=f.section, message=f.message)
                                           for f in capability_rules.CPU_ONLY_FEATURES],
                        pinned_features=sorted(capability_rules.ENGINE_ADVERTISED_FEATURES))


def _loc_path(loc: tuple, raw: dict, kind: str) -> tuple[str | None, bool, str | None, str | None, int]:
    """Project a validation location onto wire keys and retain branch evidence."""
    parts = []
    node: Any = raw
    matched_branch = False
    branch_root = None
    branch_label = None
    for index, part in enumerate(loc):
        if isinstance(part, int):
            if isinstance(node, (list, tuple)):
                if 0 <= part < len(node):
                    node = node[part]
                parts.append(part)
            continue
        if isinstance(node, dict) and part in node:
            node = node[part]
            parts.append(part)
        elif (index == len(loc) - 1 and kind in {"missing", "extra_forbidden"}
              and "[" not in str(part) and "(" not in str(part)):
            parts.append(part)
        else:
            branch_root = branch_root or _path_string(parts)
            if branch_label is None:
                branch_label = str(part)
            if isinstance(node, dict) and part == node.get("type"):
                matched_branch = True
    if kind in {"union_tag_invalid", "union_tag_not_found"}:
        parts.append("type")
    return _path_string(parts), matched_branch, branch_root, branch_label, len(parts)


def _path_string(parts: list) -> str | None:
    path = ""
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else ("." if path else "") + str(part)
    return path or None


def _validation_issues(exc: ValidationError, raw: dict) -> list[dict]:
    ids = {"missing": "spec-missing-field", "extra_forbidden": "spec-unknown-field",
           "literal_error": "spec-invalid-choice", "enum": "spec-invalid-choice",
           "union_tag_invalid": "spec-invalid-choice"}
    errors = [(error, *_loc_path(error["loc"], raw, error["type"]))
              for error in exc.errors()]
    matching_roots = {root for _, _, matched, root, _, _ in errors if matched and root}
    invalid_tag_roots = {path.rpartition(".")[0] for error, path, _, _, _, _ in errors
                         if error["type"] == "union_tag_invalid"}
    deepest = {}
    for _, _, matched, root, label, depth in errors:
        if root and root not in matching_roots and not matched:
            deepest[root, label] = max(deepest.get((root, label), 0), depth)
    winning_depth = {root: max(depth for (candidate, _), depth in deepest.items()
                               if candidate == root) for root, _ in deepest}
    out = []
    for error, path, matched, root, label, _ in errors:
        if ((root in matching_roots and not matched) or
                (root in invalid_tag_roots and error["type"] != "union_tag_invalid") or
                (root and root not in matching_roots and root not in invalid_tag_roots and
                 deepest[root, label] < winning_depth[root])):
            continue
        kind = error["type"]
        if kind == "missing" and error["loc"] and isinstance(error["loc"][-1], int):
            out.append(issue("spec-constraint", f"List element {error['loc'][-1]} is missing", path=path))
            continue
        id_ = ids.get(kind)
        if kind == "union_tag_not_found":
            id_ = "spec-missing-field"
        if id_ is None:
            id_ = ("spec-wrong-type" if kind.endswith(("_type", "_parsing")) else
                   "spec-constraint" if kind in {
                       "value_error", "assertion_error", "greater_than", "greater_than_equal",
                       "less_than", "less_than_equal", "too_short", "too_long",
                       "finite_number", "multiple_of"} else "spec-invalid")
        out.append(issue(id_, error["msg"], path=path))
    return out


def _hide_temp_path(value: Any, path: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(path), "sim.json")
    if isinstance(value, list):
        return [_hide_temp_path(item, path) for item in value]
    if isinstance(value, dict):
        return {key: _hide_temp_path(item, path) for key, item in value.items()}
    return value


def validate_spec(spec: str | Path | dict | Simulation, device: str = "cpu",
                  engine: bool = True) -> ValidationReport:
    """Return local, read-only photonhub.validation/1 checks for a spec."""
    names = ("parse", "construct", "preflight", "runnable", "device", "advisories", "engine")
    stages = dict.fromkeys(names, "skipped")
    issues: list[dict] = []
    status = EngineStatus()
    digest = None
    basis = None
    try:
        device_args(device)
    except (SolverRunError, AttributeError) as exc:
        issues.append(issue("invalid-argument", str(exc)))
        return _validation_report(device, stages, issues, digest, basis, status)
    try:
        raw = _load_spec(spec)
    except CodedError as exc:
        stages["parse"] = "fail"
        detail = {}
        if exc.id == "spec-not-json" and "line " in exc.message:
            match = re.search(r"line (\d+), column (\d+)", exc.message)
            if match:
                detail = {"line": int(match[1]), "column": int(match[2])}
        issues.append(issue(exc.id, exc.message, **detail))
        return _validation_report(device, stages, issues, digest, basis, status)
    stages["parse"] = "pass"
    try:
        sim, construction_warnings = service.parse_sim_spec(raw)
    except ValidationError as exc:
        stages["construct"] = "fail"
        issues.extend(_validation_issues(exc, raw))
        return _validation_report(device, stages, issues, digest, basis, status)
    except (ValueError, TypeError) as exc:
        stages["construct"] = "fail"
        issues.append(issue("spec-invalid", str(exc)))
        return _validation_report(device, stages, issues, digest, basis, status)
    stages["construct"] = "pass"
    issues.extend(issue("construction-warning", message) for message in construction_warnings)
    before = len(issues)
    for check, id_ in ((service.assert_no_stale_mode_sources, "mode-source-stale"),
                       (service.assert_modal_ports_ready, "modal-port-not-ready")):
        try:
            check(sim)
        except ValueError as exc:
            issues.append(issue(id_, str(exc)))
    stages["preflight"] = "fail" if len(issues) > before else "pass"
    problems = sim._runnable_problems()
    monitor_index = {m.name: i for i, m in enumerate(sim.monitors)}
    for id_, path, message in problems:
        monitor = sim.monitors[int(path[9:-1])].name if (
            id_ == "flux-plane-in-boundary" and path and path.startswith("monitors[")) else None
        issues.append(issue(id_, message, path=path, monitor=monitor))
    stages["runnable"] = "fail" if problems else "pass"
    device_wire = sim.to_wire_dict()
    if capability_rules.selects_gpu(device) and "dft_shutoff" in device_wire.get("run", {}):
        device_wire["run"].pop("dft_shutoff")
        issues.append(issue(
            "cpu-only-feature",
            "The GPU run drops dft_shutoff and uses the energy-only stop.",
            severity="warning", feature="dft_shutoff", section="run"))
    features = capability_rules.unsupported_on_device(device_wire, device)
    for feature in features:
        issues.append(issue("cpu-only-feature", feature.message,
                            feature=feature.name, section=feature.section))
    stages["device"] = "fail" if features else "pass"
    try:
        capability_rules.check_device_support(device_wire, device)
    except SolverRunError:
        pass
    else:
        wire_json = sim.to_wire_json()
        basis = "canonical"
        if capability_rules.selects_gpu(device):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                wire_json, _ = _execution_wire_json(sim, device, None)
            basis = "local-execution"
        elif engine:
            try:
                solver_for_hash = find_solver()
            except (OSError, SolverRunError):
                solver_for_hash = None
            if solver_for_hash is not None:
                needs_caps = ((sim.run.dft_shutoff or 0) > 0 or
                              (sim.run.dft_shutoff is None and sim.run.shutoff > 0 and
                               any(m.type in ("field_dft", "flux") for m in sim.monitors)))
                try:
                    caps = (capability_rules.engine_capabilities(solver_for_hash)
                            if needs_caps else None)
                except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
                    issues.append(issue("engine-error", f"Capability probe failed: {exc}. "
                                        "A local run on this solver will fail the same probe."))
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        wire_json, _ = _execution_wire_json(sim, device, caps)
                    basis = "local-execution"
        digest = hashlib.sha256((wire_json + "\n").encode()).hexdigest()
    if sim.sources:
        try:
            findings = checks.monitor_checks(sim)["findings"]
            issues.extend(issue_from_finding(f, path=f"monitors[{monitor_index[f['monitor']]}]"
                                             if f.get("monitor") in monitor_index else None)
                          for f in findings)
        except Exception as exc:  # noqa: BLE001
            issues.append(issue("advisory-skipped", f"Setup advisories could not run: {exc}"))
            stages["advisories"] = "unavailable"
        else:
            stages["advisories"] = "pass"
    if not engine:
        pass
    else:
        try:
            solver = find_solver()
        except Exception as exc:  # noqa: BLE001
            solver = None
            issues.append(issue("engine-error", str(exc)))
            stages["engine"] = "unavailable"
        if solver is None and stages["engine"] == "skipped":
            stages["engine"] = "unavailable"
            issues.append(issue("engine-unavailable", "Local solver is unavailable"))
        elif solver is not None:
            status.solver = str(solver)
            try:
                with tempfile.TemporaryDirectory(prefix="photonhub-validate-") as directory:
                    path = Path(directory) / "sim.json"
                    path.write_text(sim.to_wire_json() + "\n", encoding="utf-8")
                    result = subprocess.run([str(solver), "validate", str(path)],
                                            capture_output=True, text=True, timeout=30,
                                            env=_solver_subprocess_env(), check=False)
                status.ran = True
                lines = result.stdout.strip().splitlines()
                try:
                    raw_result = _hide_temp_path(json.loads(lines[-1]), path) if lines else None
                except json.JSONDecodeError:
                    raw_result = None
                status.result = ({key: raw_result[key] for key in
                                  ("ok", "schema_version", "grid", "dt_s", "n_steps",
                                   "est_bytes", "errors") if key in raw_result}
                                 if isinstance(raw_result, dict) else None)
                if result.returncode == 1:
                    errors = status.result.get("errors", []) if isinstance(status.result, dict) else []
                    if not errors:
                        errors = (result.stderr or result.stdout or "Engine rejected the spec").splitlines()
                    for error in errors:
                        issues.append(issue("engine-rejected", str(error).replace(str(path), "sim.json")))
                    stages["engine"] = "fail"
                elif result.returncode:
                    message = result.stderr.strip() or result.stdout.strip() or (
                        f"Local solver exited {result.returncode}")
                    issues.append(issue("engine-error", message.replace(str(path), "sim.json")))
                    stages["engine"] = "unavailable"
                else:
                    stages["engine"] = "pass"
            except (OSError, subprocess.TimeoutExpired) as exc:
                issues.append(issue("engine-error", str(exc).replace(str(path), "sim.json")))
                stages["engine"] = "unavailable"
    return _validation_report(device, stages, issues, digest, basis, status)


def _validation_report(device, stages, issues, digest, basis, engine_status):
    return ValidationReport(ok=not any(i["severity"] == "error" for i in issues),
                            authoritative=stages["engine"] in {"pass", "fail"},
                            device=device, schema_version=SCHEMA_VERSION,
                            input_sha256=digest, input_sha256_basis=basis,
                            stages=stages, issues=issues,
                            counts=_counts(issues), engine=engine_status)


def estimate(spec: str | Path | dict | Simulation, cpu_mcells_per_s_low: float = 20.0,
             cpu_mcells_per_s_high: float = 300.0, device: str = "cpu") -> LocalEstimate:
    """Return local, read-only photonhub.estimate/1 planning bounds."""
    try:
        device_args(device)
    except (SolverRunError, AttributeError) as exc:
        raise CodedError("invalid-argument", str(exc)) from exc
    if (not math.isfinite(cpu_mcells_per_s_low) or not math.isfinite(cpu_mcells_per_s_high)
            or cpu_mcells_per_s_low <= 0 or cpu_mcells_per_s_high <= 0
            or cpu_mcells_per_s_low > cpu_mcells_per_s_high):
        raise CodedError(
            "invalid-argument", "CPU throughput bounds must be positive, finite, and low <= high")
    try:
        sim, _ = service.parse_sim_spec(_load_spec(spec))
    except (CodedError, ValueError, TypeError, ValidationError) as exc:
        message = exc.message if isinstance(exc, CodedError) else str(exc)
        raise CodedError("spec-rejected", message) from exc
    q = cost.quote(sim)
    cell_steps = q.num_cells * q.num_steps
    return LocalEstimate(cells_per_axis=q.cells_per_axis, num_cells=q.num_cells,
                         num_steps=q.num_steps, dt_s=q.dt_s, cell_steps=cell_steps,
                         tcell_steps=q.tcell_steps, memory_bytes=q.device_memory_bytes,
                         output_bytes=q.output_bytes,
                         cpu_seconds=SecondsRange(low=cell_steps / (cpu_mcells_per_s_high * 1e6),
                                                  high=cell_steps / (cpu_mcells_per_s_low * 1e6)),
                         assumptions=EstimateAssumptions(
                             cpu_mcells_per_s=[cpu_mcells_per_s_low, cpu_mcells_per_s_high],
                             basis=("This is a planning range, not a measurement of this machine; "
                                    "setup time is excluded; pass a measured run.mcells_per_s "
                                    "as both bounds for a point.")),
                         notes=["This estimates full duration; auto-shutoff can finish sooner.",
                                "memory_bytes is the padded device-layout model from photonhub.cost.quote.",
                                "The cloud quote is the only binding price.",
                                "No network call was made."])


def _result(data_or_dir: RunResult | str | Path) -> RunResult:
    return data_or_dir if isinstance(data_or_dir, RunResult) else service.load_result(data_or_dir)


def _convergence_path(directory: Path) -> tuple[Path | None, str | None]:
    path = directory / "convergence.json"
    return (path, "same-directory") if path.is_file() else (None, None)


def convergence_report(data_or_dir: RunResult | str | Path) -> ConvergenceSummary:
    """Return local, read-only photonhub.convergence_report/1 evidence."""
    data = _result(data_or_dir)
    path, relation = _convergence_path(Path(data.output_dir))
    if path is None:
        return ConvergenceSummary(
            found=False,
            message="No convergence report is saved with this result, so nothing here shows "
                    "the result is mesh-converged; to attach one, save a ConvergenceReport's "
                    "to_dict() as convergence.json in the result directory.")
    base = {"found": True, "path": str(path), "relation": relation}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ConvergenceSummary(**base, format="unrecognized", keys=[],
                                  message=f"Convergence report could not be parsed: {exc}")
    required = {"converged", "metric_name", "ladder", "tol", "patience", "stop_reason"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        return ConvergenceSummary(**base, format="unrecognized",
                                  keys=sorted(payload) if isinstance(payload, dict) else [])
    fields = {key: payload.get(key) for key in ("converged", "metric_name", "stop_reason", "tol",
               "patience", "finest", "drift_successive", "drift_total", "total_cost_usd", "ladder")}
    try:
        return ConvergenceSummary(**base, format="photonhub.convergence_report", **fields)
    except ValidationError:
        return ConvergenceSummary(**base, format="unrecognized", keys=sorted(payload))


def result_summary(data_or_dir: RunResult | str | Path,
                   data_checks: bool = True) -> ResultSummary:
    """Return local, read-only photonhub.result_summary/1 findings and provenance."""
    data = _result(data_or_dir)
    manifest = data.manifest
    directory = Path(data.output_dir)
    provenance = manifest.get("provenance", {})
    recorded_hash = provenance.get("input_sha256")
    sim_path = directory / "sim.json"
    source = "manifest" if recorded_hash else None
    if not recorded_hash and sim_path.is_file():
        recorded_hash = hashlib.sha256(sim_path.read_bytes()).hexdigest()
        source = "sim.json"
    geometry = service.geometry_status(data)["status"]
    issues = []
    run = manifest.get("run", {})
    if run.get("aborted"):
        issues.append(issue("run-aborted", "Run was aborted", abort_reason=run.get("abort_reason")))
    sim = None
    if geometry in {"matched", "unverified"} and sim_path.is_file():
        try:
            sim = service.sim_for(data)
        except (OSError, ValueError):
            pass
    if (run.get("stop_reason") == "step_cap" and run.get("shut_off") is False
            and sim is not None and sim.run.shutoff > 0 and
            (sim.run.dft_shutoff or 0) > 0):
        detail = {key: run[key] for key in ("dft_hold", "energy_rule_passed_at_cap") if key in run}
        issues.append(issue("step-cap-reached", "Run reached the step cap before auto-shutoff", **detail))
    monitor_index = {entry["name"]: i for i, entry in enumerate(manifest.get("monitors", []))}
    if data_checks:
        issues.extend(issue_from_finding(f, path=f"monitors[{monitor_index[f['monitor']]}]"
                                         if f.get("monitor") in monitor_index else None)
                      for f in checks.result_checks(data)["findings"])
    if sim is not None:
        spec_monitor_index = {m.name: i for i, m in enumerate(sim.monitors)}
        try:
            issues.extend(issue_from_finding(f, path=f"monitors[{spec_monitor_index[f['monitor']]}]"
                                             if f.get("monitor") in spec_monitor_index else None)
                          for f in checks.monitor_checks(sim)["findings"])
        except Exception as exc:  # noqa: BLE001
            issues.append(issue("advisory-skipped", f"Setup advisories could not run: {exc}"))
    monitors = [ResultMonitor(name=e["name"], type=e.get("type"), kind=service._kind(e),
                              shape=e.get("shape"), components=e.get("components"),
                              n_freqs=len(e["freqs_hz"]) if "freqs_hz" in e else None,
                              axis=e.get("axis")) for e in manifest.get("monitors", [])]
    convergence_path, _ = _convergence_path(directory)
    return ResultSummary(output_dir=str(directory), input_sha256=recorded_hash,
                         input_sha256_source=source, geometry_status=geometry,
                         manifest_version=str(manifest.get("manifest_version", "1")),
                         schema_version=manifest.get("schema_version"),
                         solver={"version": provenance.get("solver_version"),
                                 "git_sha": provenance.get("git_sha"),
                                 "runtime_version": provenance.get("runtime_version")},
                         device={"backend": provenance.get("backend"),
                                 "name": provenance.get("device_name"),
                                 "vendor": provenance.get("device_vendor"),
                                 "arch": provenance.get("device_arch")},
                         run=run, grid=manifest.get("grid", {}), monitors=monitors,
                         issues=issues, counts=_counts(issues),
                         convergence=ConvergenceLocation(
                             found=convergence_path is not None,
                             path=str(convergence_path) if convergence_path else None))
