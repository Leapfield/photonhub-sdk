"""Validation for opaque service job identifiers.

Job ids cross both an HTTP path boundary and a local cache-path boundary. Keep
the accepted alphabet deliberately small so a compromised or misconfigured
service cannot turn an id into traversal, query, or fragment syntax.
"""

from __future__ import annotations

import re


_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_job_id(job_id: object) -> str:
    """Return a safe job id or raise :class:`ValueError`.

    IDs are opaque; callers must not trim or otherwise reinterpret them.
    """
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise ValueError(
            "job_id must be 1-128 ASCII letters, digits, '.', '_', or '-', "
            "starting with a letter or digit"
        )
    return job_id
