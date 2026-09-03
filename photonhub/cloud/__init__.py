"""Cloud client for the PhotonHub metered compute API.

Reads identically to the local path — same ``Job`` / ``RunResult`` /
``SolverRunError`` — only the namespace differs:

>>> import photonhub as ph
>>> ph.cloud.configure(api_key="ph_live_...", url="https://<api-host>")
>>> job = ph.cloud.submit_quoted(sim, max_usd=5.00)
>>> data = job.result()             # RunResult, same as local
>>> probe = data["probe"]           # xarray.DataArray
"""

from .actions import (
    CloudPreflight,
    account,
    cancel,
    create_api_key,
    estimate,
    gpus,
    job_status,
    list_jobs,
    preflight,
    whoami,
)
from .batch import Batch
from .client import HttpClient
from .config import CloudConfig, CloudError, configure, get_config, reset
from .run import (
    CloudJobTimeout,
    resume,
    run,
    submit,
    run_quoted,
    submit_quoted,
)

__all__ = [
    "configure",
    "get_config",
    "reset",
    "CloudConfig",
    "CloudError",
    "CloudPreflight",
    "HttpClient",
    "run",
    "submit",
    "run_quoted",
    "submit_quoted",
    "resume",
    "CloudJobTimeout",
    "Batch",
    "estimate",
    "preflight",
    "account",
    "whoami",
    "create_api_key",
    "cancel",
    "gpus",
    "list_jobs",
    "job_status",
]


# --- deprecated aliases (2026-09 cross-solver rename; remove in 0.2) ---------
_RENAMED = {"run_async": "submit", "run_quoted_async": "submit_quoted"}


def __getattr__(name):
    replacement = _RENAMED.get(name)
    if replacement is not None:
        import warnings

        warnings.warn(
            f"photonhub.cloud.{name} was renamed to photonhub.cloud.{replacement}; "
            "the old alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[replacement]
    raise AttributeError(f"module 'photonhub.cloud' has no attribute {name!r}")
