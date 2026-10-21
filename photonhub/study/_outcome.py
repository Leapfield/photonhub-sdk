"""The per-sample outcome the execution layer hands the results table.

Shared by :mod:`photonhub.study._execute` (which produces outcomes) and
:mod:`photonhub.study.table` (which turns them into the table). Kept in its own
module so neither side owns the other's file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

#: Every status a sample row can carry in the results table.
#:
#: - ``succeeded``: the job ran and ``measure`` (if any) returned scalars.
#: - ``failed``: the solver or the service reported a failure.
#: - ``measure_failed``: the job ran but ``measure`` raised or returned a
#:   value that is not a finite real number.
#: - ``build_failed``: ``build`` raised for this row; no job exists.
#: - ``submitted``: a cloud job was submitted and is not collected yet.
#: - ``timed_out``: the wait for a cloud job ended; the job keeps running and
#:   a later run collects it without submitting again.
#: - ``pending``: not run (for example, the call was refused or interrupted).
STATUSES = (
    "succeeded", "failed", "measure_failed", "build_failed", "submitted",
    "timed_out", "pending",
)

#: Column names the results table uses for its own fields. A parameter or a
#: measured quantity may not take one of these names.
RESERVED = frozenset({
    "sample", "sample_name", "status", "error", "job", "job_id",
    "spec_sha256", "solver_version", "backend", "quote_usd",
})

BACKENDS = ("local", "cloud", "")


@dataclass(frozen=True)
class SampleOutcome:
    """What happened to one sample row.

    ``job`` is the job name ("" when no job exists, as for ``build_failed``),
    ``spec_sha256`` the job's spec digest, ``measured`` the scalars
    ``measure`` returned (empty unless ``succeeded``), ``error`` a one-line
    reason for any non-success status, ``solver_version`` the solver's
    reported version ("" when unknown), ``backend`` ``"local"`` or
    ``"cloud"``, ``job_id`` the service job id of a cloud job and
    ``quote_usd`` its accepted quote."""

    status: str
    job: str = ""
    spec_sha256: str = ""
    measured: Mapping[str, float] = field(default_factory=dict)
    error: str = ""
    solver_version: str = ""
    backend: str = ""
    job_id: str = ""
    quote_usd: Optional[float] = None

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(
                f"status must be one of {STATUSES}, got {self.status!r}")
        if self.backend not in BACKENDS:
            raise ValueError(
                f"backend must be one of {BACKENDS}, got {self.backend!r}")
        for label in ("job", "spec_sha256", "error", "solver_version", "job_id"):
            if not isinstance(getattr(self, label), str):
                raise TypeError(f"{label} must be a str")
        if self.quote_usd is not None:
            if isinstance(self.quote_usd, bool) or not isinstance(
                    self.quote_usd, (int, float)):
                raise TypeError("quote_usd must be a number or None")
            object.__setattr__(self, "quote_usd", float(self.quote_usd))
        object.__setattr__(
            self, "measured", MappingProxyType(dict(self.measured)))
