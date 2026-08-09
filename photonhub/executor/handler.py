"""RunPod serverless handler — the serverless entrypoint that wraps execute().

``import runpod`` is done lazily (only inside the RunPod-facing functions), so
this module imports without runpod present, the executor core + CLI stay
dependency-free, and the pure :func:`handle` logic is directly unit-testable.
One image serves serverless and non-serverless.

RunPod invokes ``handler(event)`` with
``event["input"] = {"spec": <wire-json dict>, "params": {"device","timeout_s"}}``.
We run the job and return the result bundle inline (base64) + metrics +
provenance, or ``{"error": {"reason","stderr_tail"}}``. A result too large for
the inline response needs a non-serverless provider until the cloud endpoint
wires object storage (the CLI path streams big bundles straight to disk).
"""

import base64
import os
from typing import Callable, Optional

from ..runners.phsolver import SolverRunError
from .core import execute

#: Bytes above which a serverless result cannot be returned inline.
_DEFAULT_MAX_INLINE = 8 * 1024 * 1024


def _error(reason: str, stderr_tail: Optional[str] = None) -> dict:
    # Deliberately NOT keyed under "error": RunPod serverless treats a top-level
    # "error" key as a job failure and nulls the whole output, which would hide
    # the real message. "ok": False keeps the envelope in the output so the
    # executor can surface it.
    return {"ok": False, "reason": reason, "stderr_tail": stderr_tail}


def handle(event: dict, *,
           progress: Optional[Callable[[dict], None]] = None) -> dict:
    """Pure serverless logic: run one job from a RunPod-style ``event`` and
    return the result dict. No runpod dependency, so it is directly testable.

    On success returns ``{"ok": True, "metrics", "provenance", "bundle_b64"}``;
    on failure ``{"ok": False, "reason", "stderr_tail"}`` — including a result
    too large for the inline response. (Not keyed under ``"error"`` on purpose;
    see :func:`_error`.)
    """
    inp = event.get("input") or {}
    spec = inp.get("spec")
    if not isinstance(spec, dict):
        return _error("input.spec must be a wire-JSON object")
    params = inp.get("params") or {}
    device = params.get("device", "gpu")
    timeout = params.get("timeout_s")

    try:
        result = execute(spec, device=device, timeout=timeout,
                         on_event=progress)
    except SolverRunError as e:
        return _error(str(e), getattr(e, "stderr_tail", None))

    bundle = result.bundle
    limit = int(os.environ.get("RESULT_MAX_INLINE_BYTES", _DEFAULT_MAX_INLINE))
    if len(bundle) > limit:
        return _error(
            f"result {len(bundle)} B exceeds the inline limit ({limit} B); run "
            "large jobs on a non-serverless GPU, or configure object storage on "
            "the endpoint")
    return {"ok": True,
            "metrics": result.metrics,
            "provenance": result.manifest.get("provenance", {}),
            "bundle_b64": base64.b64encode(bundle).decode("ascii")}


def handler(event: dict) -> dict:
    """RunPod serverless entrypoint: stream phsolver progress to RunPod's status
    channel, then return :func:`handle`'s result."""
    import runpod  # lazy: only the serverless path needs it

    def progress(ev: dict) -> None:
        try:
            runpod.serverless.progress_update(event, ev)
        except Exception:
            pass  # progress is best-effort; never fail a job over a status post

    return handle(event, progress=progress)


def main() -> None:
    import runpod
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
