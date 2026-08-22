"""Environment-variable helpers shared by PhotonHub process boundaries."""

from __future__ import annotations

import os
from collections.abc import Mapping

_CREDENTIAL_ENV_SUFFIXES = (
    "_ACCESS_KEY",
    "_API_KEY",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_TOKEN",
    "_URL",
)


def env(suffix: str) -> str | None:
    """Read one ``PHOTONHUB_<suffix>`` environment variable."""
    return os.environ.get(f"PHOTONHUB_{suffix}")


def is_credential_env_name(name: str) -> bool:
    """Return whether *name* is a conventionally named credential or URL.

    Matching complete underscore-delimited suffixes keeps the policy broad
    enough to catch credentials from any provider without stripping unrelated
    variables that merely contain words such as ``TOKEN`` or ``URL``.
    """
    normalized = name.upper()
    return any(
        len(normalized) > len(suffix) and normalized.endswith(suffix)
        for suffix in _CREDENTIAL_ENV_SUFFIXES
    )


def without_credentials(environment: Mapping[str, str]) -> dict[str, str]:
    """Copy *environment* without credential-bearing variables."""
    return {
        name: value
        for name, value in environment.items()
        if not is_credential_env_name(name)
    }
