"""Cloud client for the PhotonHub metered compute API.

Reads identically to the local path — same ``Job`` / ``SimulationData`` /
``SolverRunError`` — only the namespace differs:

>>> import photonhub as ph
>>> ph.web.configure(api_key="ph_live_...", url="https://<api-host>")
>>> job = ph.web.run_quoted_async(sim, max_usd=5.00)
>>> data = job.result()             # SimulationData, same as local
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
from .config import WebConfig, WebError, configure, get_config, reset
from .run import (
    WebJobTimeout,
    resume,
    run,
    run_async,
    run_quoted,
    run_quoted_async,
)

__all__ = [
    "configure",
    "get_config",
    "reset",
    "WebConfig",
    "WebError",
    "CloudPreflight",
    "HttpClient",
    "run",
    "run_async",
    "run_quoted",
    "run_quoted_async",
    "resume",
    "WebJobTimeout",
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
