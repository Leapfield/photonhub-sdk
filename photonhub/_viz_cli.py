"""Dependency-light console entry point for the optional viz HTTP server."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    """Run the server or explain how to add its optional dependencies."""
    try:
        from .viz.server import _main as run

        return run(argv)
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
