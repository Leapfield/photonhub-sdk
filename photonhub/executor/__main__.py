"""CLI executor: ``python -m photonhub.executor --spec sim.json --out bundle.tar.gz``.

Reads a wire-JSON spec, runs the job via :func:`execute`, writes the result
bundle (gzip tar) to ``--out``, and streams the phsolver JSON-lines events to
stdout so a pod / k8s / bare-metal / Hot Aisle harness can follow progress.
Exit 0 ok; exit 2 + a final ``{"event":"error",...}`` line on failure. Used by
every non-serverless provider and is free of any ``runpod`` dependency.
"""

import argparse
import json
import sys
from pathlib import Path

from ..runners.phsolver import SolverRunError
from .core import execute


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m photonhub.executor")
    ap.add_argument("--spec", required=True, help="wire-JSON spec file")
    ap.add_argument("--out", required=True, help="result bundle output path (.tar.gz)")
    ap.add_argument("--device", default="cpu", help="cpu | gpu | gpu:N")
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds before the solver is killed")
    args = ap.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    def emit(event: dict) -> None:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()

    try:
        # Stream the bundle straight to --out (no in-RAM copy of a large result).
        result = execute(spec, device=args.device, timeout=args.timeout,
                         on_event=emit, bundle_path=args.out)
    except SolverRunError as e:
        emit({"event": "error", "reason": str(e)})
        return 2

    emit({"event": "bundle", "path": str(result.bundle_path),
          "bytes": Path(args.out).stat().st_size, "metrics": result.metrics})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
