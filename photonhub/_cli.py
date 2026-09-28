"""The ``photonhub`` console command.

`pip install photonhub` gives you the client; the solver engine is a separate,
invitation-gated download. These subcommands are how it gets onto a machine
that has no browser and no package manager, and how you find out which binary
is actually being used when something looks wrong.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from . import solver_install

_INFO_TIMEOUT_S = 30.0


def _print_record(record: dict, *, verb: str) -> None:
    print(f"{verb} {record['path']}")
    described = [f"{k} {record[k]}" for k in ("version", "git_sha", "platform")
                 if record.get(k)]
    if described:
        print("  " + ", ".join(described))
    print("  Nothing else to set: the client finds this automatically.")


def _install(args) -> int:
    if bool(args.url) == bool(args.archive):
        print("photonhub install-solver: give exactly one of --url or --archive",
              file=sys.stderr)
        return 2
    if not args.sha256:
        # Said before anything is unpacked, so it can be acted on.
        print("photonhub install-solver: no --sha256 given, so nothing can "
              "confirm this is the archive we published.", file=sys.stderr)
    try:
        if args.url:
            record = solver_install.install_url(
                args.url, expected_sha256=args.sha256, force=args.force)
        else:
            record = solver_install.install_archive(
                args.archive, expected_sha256=args.sha256, force=args.force)
    except (solver_install.SolverInstallError, OSError) as exc:
        print(f"photonhub install-solver: {exc}", file=sys.stderr)
        return 1
    _print_record(record, verb="Installed")
    if not args.sha256:
        if "provenance" in record.get("checked", []):
            print("  Checked against the archive's own provenance.json only, "
                  "which catches a truncated download.", file=sys.stderr)
        else:
            print("  The archive carried no provenance.json, so its contents "
                  "were not checked at all.", file=sys.stderr)
    return 0


def _link(args) -> int:
    try:
        record = solver_install.link_solver(args.path)
    except (solver_install.SolverInstallError, OSError) as exc:
        print(f"photonhub link-solver: {exc}", file=sys.stderr)
        return 1
    _print_record(record, verb="Using")
    return 0


def _run_info(found) -> int:
    from .runners.phsolver import _solver_subprocess_env

    try:
        proc = subprocess.run(
            [str(found), "info"], capture_output=True, text=True,
            timeout=_INFO_TIMEOUT_S, env=_solver_subprocess_env())
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"photonhub which-solver: could not run `{found} info`: {exc}",
              file=sys.stderr)
        return 1
    if proc.returncode != 0:
        print(f"photonhub which-solver: `{found} info` exited "
              f"{proc.returncode}: {proc.stderr.strip()[:400]}", file=sys.stderr)
        return 1
    print(proc.stdout.strip())
    return 0


def _stale_reason(record: dict) -> str:
    """Why a recorded binary cannot be used: gone, or present but not runnable."""
    from pathlib import Path

    return "is not executable" if Path(record["path"]).exists() else "is missing"


def _warn_if_writable_by_others() -> None:
    for path in solver_install.writable_by_others():
        print(f"photonhub which-solver: warning: {path} is writable by other "
              "users, who could change which solver runs. Run "
              f"`chmod go-w {path}`.", file=sys.stderr)


def _which(args) -> int:
    from ._env import env
    from .runners.phsolver import _source_match_required, find_solver

    _warn_if_writable_by_others()
    try:
        found = find_solver()
    except Exception as exc:  # a bad override should explain itself, not traceback
        print(f"photonhub which-solver: {exc}", file=sys.stderr)
        return 1
    stale = solver_install.stale_record()
    record = solver_install.installed_record()
    notes = []
    if stale:
        notes.append(f"the recorded solver {stale['path']} {_stale_reason(stale)}, "
                     "so it was skipped. Install again, or run "
                     "`photonhub uninstall-solver`.")
    elif (record and record["path"] != str(found) and not env("SOLVER")
          and _source_match_required()):
        notes.append(f"the recorded solver {record['path']} was skipped because "
                     "PHOTONHUB_REQUIRE_SOURCE_MATCH=1 and it was not built from "
                     "this source tree.")
    if found is None:
        print("No solver found.")
        for note in notes:
            print(f"  note: {note}")
        print("  Install one with `photonhub install-solver --url <link>`, or "
              "point at an existing binary with `photonhub link-solver <path>`.")
        return 1
    if env("SOLVER"):
        source = "$PHOTONHUB_SOLVER"
    elif record and record["path"] == str(found):
        how = "link-solver" if record.get("source") == "link" else "install-solver"
        source = f"recorded by `photonhub {how}`"
    else:
        source = "found on PATH or in a source checkout"
    print(f"{found}")
    print(f"  selected via: {source}")
    for note in notes:
        print(f"  note: {note}")
    if args.info:
        return _run_info(found)
    return 0


def _uninstall(args) -> int:
    try:
        forgot = solver_install.uninstall()
    except OSError as exc:
        print(f"photonhub uninstall-solver: {exc}", file=sys.stderr)
        return 1
    if forgot:
        print("Forgot the recorded solver. The files were left in place.")
        return 0
    print("No solver was recorded.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="photonhub",
        description="PhotonHub client utilities. The solver engine is a "
                    "separate, invitation-gated download.")
    subcommands = parser.add_subparsers(dest="command")

    install = subcommands.add_parser(
        "install-solver",
        help="download or unpack the solver engine and start using it",
        description="Unpack a solver archive into a managed directory and "
                    "record it, so no environment variable is needed.",
        epilog="If the link needs a bearer credential, put it in "
               "$PHOTONHUB_SOLVER_TOKEN rather than on the command line. It is "
               "sent only to the host in --url, never across a redirect, and "
               "every redirect must stay on https.")
    source = install.add_argument_group("where the archive comes from")
    source.add_argument("--url", help="link issued for you, normally short-lived")
    source.add_argument("--archive", help="an archive already on this machine")
    install.add_argument(
        "--sha256", help="the digest published with the archive, or the line "
                         "of its .sha256 file; pass it whenever you have it")
    install.add_argument("--force", action="store_true",
                         help="replace an install of the same version")
    install.set_defaults(func=_install)

    link = subcommands.add_parser(
        "link-solver", help="use a solver you already unpacked, where it is",
        description="Record an existing solver binary without copying it.")
    link.add_argument("path", help="the phsolver binary, or the directory holding it")
    link.set_defaults(func=_link)

    which = subcommands.add_parser(
        "which-solver", help="show which solver the client will run, and why")
    which.add_argument(
        "--info", action="store_true",
        help="also run the solver's `info` and print its build description")
    which.set_defaults(func=_which)

    forget = subcommands.add_parser(
        "uninstall-solver", help="forget the recorded solver, leaving the files")
    forget.set_defaults(func=_uninstall)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
