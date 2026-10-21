"""Dependency-light console entry point for the optional viz HTTP server."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    """Run the server or explain how to add its optional dependencies."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--notebook-worker" in args:
        if args != ["--notebook-worker"]:
            print("invalid notebook worker arguments", file=sys.stderr)
            return 2
        from .viz.notebook_worker import worker_main
        return worker_main()
    try:
        from .viz.server import _main as run

        return run(args)
    except ModuleNotFoundError as exc:
        if exc.name not in {"fastapi", "uvicorn"}:
            raise
        print(
            "photonhub-serve-viz requires optional server support; install it "
            "with `pip install 'photonhub[server]'`.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
