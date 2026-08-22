"""Dependency-light console entry point for the optional MCP server."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    """Load MCP support on demand and give base installs an actionable error."""
    try:
        from .mcp_server import main as run
    except ModuleNotFoundError as exc:
        if exc.name != "mcp":
            raise
        print(
            "photonhub-mcp requires optional MCP support; install it with "
            "`pip install 'photonhub[mcp]'`.",
            file=sys.stderr,
        )
        return 2
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
