"""The GPU and solver build a cloud GPU call uses when the caller names neither.

The managed GPU service runs only a curated ``gpu:<id>`` from its menu
(``GET /v1/gpus``) and only a solver commit it approved for that GPU, and it
never chooses either itself. The client therefore fills both in before it
quotes:

* ``device="gpu"``, every cloud call's default, becomes ``gpu:<id>`` when the
  menu lists exactly one GPU. With several it refuses and names them. An
  empty or unreachable menu leaves ``"gpu"`` for the service to answer, and
  its refusal now carries the service's reason. The menu is read once per
  call, in one attempt with a ``_MENU_DEADLINE_S``-second timeout, never
  retried. The timeout bounds connecting and each read from the socket, not
  the name lookup or the whole response.
* ``solver=None`` on a GPU device becomes the ``git_sha`` that the solver a
  local run would use (:func:`~photonhub.find_solver`) reports in
  ``phsolver info``, so a local CPU run and a cloud GPU run name the same
  build. Only a full 40-character lowercase commit is sent: it is what a
  release solver reports and what the service's approved builds are keyed
  by. A source build reports a 12-character id (``git rev-parse
  --short=12``) and a build from modified sources ``<sha>-dirty``; with
  either, with no solver installed, or with one that gives no build id, the
  call stops before any request and says to pass ``solver=`` or install a
  release solver. The service
  still decides: a commit it has not approved for that GPU is refused with
  its own reason. The SDK carries no commit of its own.

A call resolves once, before its quote, and submits with the same pair, so
the quote it binds stays valid; a batch resolves once for all its entries.
Explicit values are never changed, and a CPU run (or ``device=None``) keeps
its ``solver`` as given.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from .config import CloudError

#: The only solver build the default sends: a full lowercase git commit, the
#: form the service's approved builds are keyed by.
_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
# An abbreviated commit, such as the 12 characters a source build stamps.
_SHORT_COMMIT = re.compile(r"[0-9a-f]{4,39}")
# A curated GPU id, as `run._GPU_ID` accepts it.
_GPU_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")

#: The timeout, in seconds, of the one attempt a call makes to read the GPU
#: menu: it bounds connecting and each socket read, not the name lookup or
#: the whole response.
_MENU_DEADLINE_S = 10.0

_RELEASE_SOLVER = (
    "a release solver (https://leapfield.ai/docs/get-started/headless-solver/)")
_LEAD = ("a cloud GPU run with solver=None sends the commit of the locally "
         "installed solver")
_NO_REQUEST = "No request was sent."


def is_gpu_device(device: Optional[str]) -> bool:
    """Whether a validated cloud ``device`` asks for a GPU."""
    return isinstance(device, str) and (
        device == "gpu" or device.startswith("gpu:"))


def _local_solver():
    """``(path, git_sha)`` of the solver :func:`~photonhub.find_solver`
    selects for a local run, ``(None, None)`` when none is found. A broken
    ``solver_path``-style override raises ``SolverRunError``."""
    from ..runners.phsolver import _solver_git_sha, find_solver

    path = find_solver()
    if path is None:
        return None, None
    return path, _solver_git_sha(path)


def installed_solver_commit(*, closing: str = _NO_REQUEST) -> str:
    """The full git commit of the locally installed solver, the default
    ``solver`` of a cloud GPU call.

    Raises :class:`CloudError` when no solver is installed, the lookup fails,
    the solver gives no build id (it does not answer ``phsolver info``, as
    the Workbench's solver does not without its launch authorization), or it
    reports anything but a full 40-character lowercase commit: a source
    build's 12-character id, which the service cannot match to an approved
    build, or a ``<sha>-dirty`` build from modified sources. Each refusal
    ends with ``closing``, which says what was sent."""
    from ..runners.local import SolverRunError

    try:
        path, sha = _local_solver()
    except SolverRunError as exc:
        raise CloudError(
            f"{_LEAD}, and the solver lookup failed ({exc}); fix it, install "
            f"{_RELEASE_SOLVER} or pass solver='<full commit>' explicitly. "
            f"{closing}") from exc
    if path is None:
        raise CloudError(
            f"{_LEAD}, and no solver is installed: install {_RELEASE_SOLVER} "
            f"or pass solver='<full commit>' explicitly. {closing}")
    if not sha:
        raise CloudError(
            f"{_LEAD}, and the solver at {path} gave no build id: it did not "
            "answer `phsolver info` with one. Pass solver='<full commit>' "
            f"explicitly, or install {_RELEASE_SOLVER}. {closing}")
    if isinstance(sha, str) and _FULL_COMMIT.fullmatch(sha):
        return sha
    if isinstance(sha, str) and _SHORT_COMMIT.fullmatch(sha):
        raise CloudError(
            f"{_LEAD}, and this solver build ({path}) reports a short build "
            f"id, {sha!r}, that the cloud cannot match: it needs the full "
            "40-character commit. Pass solver='<full commit>' explicitly, or "
            f"install {_RELEASE_SOLVER}, which reports its full commit. "
            f"{closing}")
    raise CloudError(
        f"{_LEAD}, and {path} reports git_sha {sha!r}, not a full 40-character "
        "lowercase commit (a build from modified sources reports "
        f"'<sha>-dirty'); install {_RELEASE_SOLVER} or pass "
        f"solver='<full commit>' explicitly. {closing}")


def menu_gpu_ids(menu) -> list:
    """The distinct, well-formed GPU ids of a ``GET /v1/gpus`` menu, in its
    order. Malformed entries are ignored."""
    ids = []
    if not isinstance(menu, list):
        return ids
    for entry in menu:
        gpu_id = entry.get("id") if isinstance(entry, dict) else None
        if (isinstance(gpu_id, str) and _GPU_ID.fullmatch(gpu_id)
                and gpu_id not in ids):
            ids.append(gpu_id)
    return ids


def default_gpu_device(http) -> str:
    """``gpu:<id>`` for the one GPU the service's menu lists, ``"gpu"`` when
    the menu is empty or cannot be read. Several GPUs raise
    :class:`CloudError` naming them, since the client cannot choose.

    The menu is read in one attempt with a ``_MENU_DEADLINE_S``-second
    timeout, never retried: the menu only refines the default, and the
    service answers a bare ``"gpu"`` itself."""
    try:
        menu = http.list_gpus(
            deadline=time.monotonic() + _MENU_DEADLINE_S, retry=False)
    except (CloudError, OSError):  # TimeoutError is an OSError
        return "gpu"
    ids = menu_gpu_ids(menu)
    if not ids:
        return "gpu"
    if len(ids) > 1:
        choices = ", ".join(f"'gpu:{gpu_id}'" for gpu_id in ids)
        raise CloudError(
            f"the service offers {len(ids)} GPUs, so device='gpu' is "
            f"ambiguous: pass device= with one of {choices}. No request "
            "was sent beyond reading the GPU menu.")
    return f"gpu:{ids[0]}"


def resolve(http, device: Optional[str], solver):
    """``(device, solver)`` for one cloud call, from a validated ``device``.

    The solver default is checked first, since it needs no request; the GPU
    menu is read only for a bare ``"gpu"``, once. Explicit values pass
    unchanged, an explicit ``solver`` in any form: the service judges it."""
    if not is_gpu_device(device):
        return device, solver
    if solver is None:
        solver = installed_solver_commit()
    if device == "gpu":
        device = default_gpu_device(http)
    return device, solver
