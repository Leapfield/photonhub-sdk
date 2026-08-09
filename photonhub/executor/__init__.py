"""photonhub.executor — the provider-agnostic job executor.

``execute(spec)`` runs a wire-JSON spec through phsolver and returns a result
bundle, reusing ``runners.phsolver`` so the wire-event/error contract is
identical to ``run_local``. Thin wrappers sit on top:

  - ``handler.py``   — RunPod serverless handler (``import runpod`` isolated there)
  - ``python -m photonhub.executor`` — CLI for pods / k8s / bare-metal / Hot Aisle

Both call ``execute()``; one image serves serverless and non-serverless.
"""

from .core import ExecResult, execute

__all__ = ["execute", "ExecResult"]
