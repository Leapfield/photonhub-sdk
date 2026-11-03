"""Dependency-light console entry point for the optional MCP server."""

from __future__ import annotations

import argparse
import sys

from ._skill import read_skill


def build_parser() -> argparse.ArgumentParser:
    """The ``photonhub-mcp`` command line. It needs no MCP support, so
    ``--help`` and option errors work in a base install too."""
    parser = argparse.ArgumentParser(
        prog="photonhub-mcp",
        description=(
            "Serve PhotonHub's result-inspection and local-run tools to an MCP "
            "client over stdio. Point the client's command at this program."
        ),
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        help=(
            "a result directory to open at start (default: $PHOTONHUB_RESULT_DIR); "
            "without one, the client calls open_result first"
        ),
    )
    skills = parser.add_mutually_exclusive_group()
    skills.add_argument("--print-skill", action="store_true",
                        help="print the packaged PhotonHub agent skill and exit")
    skills.add_argument("--print-import-skill", action="store_true",
                        help="print the packaged skill for rebuilding a simulation "
                             "from another tool and exit")
    return parser


def main(argv=None) -> int:
    """Load MCP support on demand and give base installs an actionable error."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)  # --help exits 0, an unknown option exits 2
    if args.print_skill or args.print_import_skill:
        try:
            print(read_skill("photonhub-import") if args.print_import_skill else read_skill(),
                  end="")
        except (FileNotFoundError, OSError):
            print("photonhub-mcp: packaged skill file is missing", file=sys.stderr)
            return 2
        return 0
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
