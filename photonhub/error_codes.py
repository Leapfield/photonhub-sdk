"""Stable issue codes for local validation and result inspection.

An issue has id, severity, path, monitor, message, fix, and optional detail.
This extends the viz/checks.py finding shape. IDs are never renamed or reused;
new IDs are added to the registry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

__all__ = ["CODES", "CodedError", "ErrorCode", "error_catalog", "issue", "issue_from_finding"]


@dataclass(frozen=True)
class ErrorCode:
    id: str
    severity: str
    category: str
    title: str
    fix: str


_GROUPS = {
    "spec": {
        "spec-unreadable": "Use a readable .json spec file.",
        "spec-not-json": "Provide a JSON object or a path to a readable .json file.",
        "spec-missing-field": "Add the required field.",
        "spec-unknown-field": "Remove the unsupported field.",
        "spec-wrong-type": "Use the field type required by the schema.",
        "spec-invalid-choice": "Choose a value allowed by the schema.",
        "spec-constraint": "Change the value to satisfy the spec constraint.",
        "spec-invalid": "Check the simulation schema and correct the spec.",
        "construction-warning": "Review the construction warning before running.",
        "mode-source-stale": "Re-solve the mode source against the current geometry and grid.",
        "modal-port-not-ready": "Prepare the modal port before running.",
    },
    "runnable": {
        "no-sources": "Add a source or a sourced port.",
        "boundary-layers-overlap": "Enlarge the domain, refine the mesh, or reduce boundary layers.",
        "flux-plane-in-boundary": "Move the flux plane outside the boundary layers.",
        "cw-ramp-longer-than-run": "Lengthen the run or reduce ramp cycles.",
        "source-in-boundary": "Move the source out of the boundary layers.",
    },
    "device": {"cpu-only-feature": "Choose the CPU device for this feature."},
    "engine": {
        "engine-rejected": "Correct the spec according to the engine error.",
        "engine-unavailable": "Install or build the local solver to run engine validation.",
        "engine-error": "Check the local solver and retry validation.",
    },
    "setup-advisory": {
        "advisory-skipped": "Inspect the setup advisory error and retry the summary.",
        "no-monitors": "Add a monitor if recorded output is needed.",
        "absorbing-band": "Move the monitor into the nonabsorbing interior.",
        "out-of-band": "Choose frequencies within the source band.",
        "redundant-frequencies": "Reduce the number of recorded frequencies.",
        "aliasing": "Sample often enough to resolve the signal.",
        "output-size": "Reduce the monitor output size if storage is limited.",
        "apodization-window": "Adjust the recording window to include the signal.",
    },
    "result-advisory": {
        "read-error": "Check the monitor output file.",
        "all-zero": "Check the monitor placement and recording interval.",
        "non-finite": "Inspect the run for divergence or incomplete output.",
        "flux-above-unity": "Check resolution, normalization, and source bandwidth.",
    },
    "run": {
        "run-aborted": "Inspect the abort reason and rerun if needed.",
        "step-cap-reached": "Increase the run duration or inspect the stop criteria.",
    },
    "tool": {
        "no-result-loaded": "Call open_result(result_dir) first or pass result_dir.",
        "result-unreadable": "Provide a readable result bundle.",
        "unknown-monitor": "Choose a monitor from list_monitors.",
        "unknown-definition": "Choose a definition name from get_schema().definitions.",
        "invalid-argument": "Provide valid argument values.",
        "spec-rejected": "Call validate_spec to inspect and correct the spec.",
        "solver-not-found": "Install or build the local solver.",
        "solver-run-failed": "Inspect the local solver error and retry.",
        "internal-error": "Report this error with the tool input and message.",
    },
}

_INFO_IDS = {"engine-unavailable", "no-monitors", "advisory-skipped"}
_WARNING_IDS = {
    "construction-warning", "engine-error", "step-cap-reached", "absorbing-band",
    "out-of-band", "redundant-frequencies", "aliasing", "output-size",
    "apodization-window", "all-zero", "flux-above-unity",
}

_TITLES = {
    "spec-unreadable": "Spec unreadable",
    "spec-not-json": "Invalid JSON spec",
    "spec-missing-field": "Required field missing",
    "spec-unknown-field": "Unknown spec field",
    "spec-wrong-type": "Wrong field type",
    "spec-invalid-choice": "Invalid choice",
    "spec-constraint": "Spec constraint",
    "spec-invalid": "Invalid spec",
    "construction-warning": "Construction warning",
    "mode-source-stale": "Stale mode source",
    "modal-port-not-ready": "Modal port not ready",
    "no-sources": "No sources",
    "boundary-layers-overlap": "Boundary layers overlap",
    "flux-plane-in-boundary": "Flux plane in boundary",
    "cw-ramp-longer-than-run": "CW ramp too long",
    "source-in-boundary": "Source in boundary",
    "cpu-only-feature": "CPU-only feature",
    "engine-rejected": "Engine rejected spec",
    "engine-unavailable": "Engine unavailable",
    "engine-error": "Engine error",
    "no-monitors": "No monitors",
    "advisory-skipped": "Setup advisories skipped",
    "absorbing-band": "Absorbing band",
    "out-of-band": "Outside source band",
    "redundant-frequencies": "Redundant frequencies",
    "aliasing": "Aliasing risk",
    "output-size": "Large output",
    "apodization-window": "Apodization window",
    "read-error": "Monitor read error",
    "all-zero": "All-zero monitor",
    "non-finite": "Non-finite values",
    "flux-above-unity": "Flux above unity",
    "run-aborted": "Run aborted",
    "step-cap-reached": "Step cap reached",
    "no-result-loaded": "No result loaded",
    "result-unreadable": "Result unreadable",
    "unknown-monitor": "Unknown monitor",
    "unknown-definition": "Unknown definition",
    "invalid-argument": "Invalid argument",
    "spec-rejected": "Spec rejected",
    "solver-not-found": "Solver not found",
    "solver-run-failed": "Solver run failed",
    "internal-error": "Internal error",
}

CODES: dict[str, ErrorCode] = {
    id_: ErrorCode(id_, "info" if id_ in _INFO_IDS else
                   "warning" if id_ in _WARNING_IDS else "error",
                   category, _TITLES[id_], fix)
    for category, entries in _GROUPS.items()
    for id_, fix in entries.items()
}


def issue(id: str, message: str, *, path=None, monitor=None, severity=None,
          fix=None, **detail) -> dict:
    """Create a registered issue, preserving caller detail and overrides."""
    code = CODES[id]
    result = {"id": id, "severity": severity or code.severity, "path": path,
              "monitor": monitor, "message": message, "fix": fix or code.fix}
    if detail:
        result["detail"] = detail
    return result


def issue_from_finding(finding: dict, *, path=None) -> dict:
    """Map a setup, result, or repair-carrying finding to an issue."""
    id_ = finding["id"]
    code = CODES.get(id_)
    result = {"id": id_, "severity": finding["severity"], "path": path,
              "monitor": finding.get("monitor"), "message": finding["message"],
              "fix": finding.get("repair") or (code.fix if code else "")}
    if finding.get("detail"):
        result["detail"] = finding["detail"]
    return result


def error_catalog() -> list[dict]:
    """Return the JSON-serializable registry ordered by stable ID."""
    return [asdict(CODES[id_]) for id_ in sorted(CODES)]


class CodedError(ValueError):
    """Tool failure carrying a stable issue ID and a JSON string form."""

    def __init__(self, id: str, message: str, fix: str | None = None):
        code = CODES[id]
        self.id = id
        self.message = message
        self.fix = fix or code.fix
        super().__init__(message)

    def __str__(self) -> str:
        return json.dumps({"id": self.id, "severity": CODES[self.id].severity,
                           "message": self.message, "fix": self.fix})
