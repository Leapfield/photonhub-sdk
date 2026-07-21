#!/usr/bin/env python3
"""Install and smoke-test the exact wheel and sdist in clean environments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import venv


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / ".github/scripts/smoke_install.py"


def python_path(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "PYTHONHOME",
        "PYTHONOPTIMIZE",
        "PYTHONPATH",
        "PYTHONWARNINGS",
        "SIMUPOD_SOLVER",
    ):
        environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def run(python: Path, *arguments: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(python), *arguments],
        cwd=python.parents[2],
        env=clean_environment(),
        check=not capture,
        capture_output=capture,
        text=True,
    )


def create_environment(root: Path, name: str) -> Path:
    destination = root / name
    venv.EnvBuilder(with_pip=True, clear=True).create(destination)
    return python_path(destination)


def install_and_smoke(python: Path, artifact: Path) -> None:
    run(python, "-m", "pip", "install", str(artifact))
    run(python, "-m", "pip", "check")
    run(python, str(SMOKE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    wheels = sorted(args.dist.resolve().glob("simupod-*-py3-none-any.whl"))
    sdists = sorted(args.dist.resolve().glob("simupod-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one pure-Python wheel and one sdist")

    with tempfile.TemporaryDirectory(prefix="simupod-artifact-smoke-") as tmp:
        root = Path(tmp)

        wheel_python = create_environment(root, "wheel")
        install_and_smoke(wheel_python, wheels[0])
        missing_extra = run(
            wheel_python,
            "-c",
            "from simupod.cli import mcp; raise SystemExit(mcp())",
            capture=True,
        )
        if missing_extra.returncode != 2 or "simupod[app]" not in missing_extra.stderr:
            raise RuntimeError("base-wheel CLI did not explain the optional app extra")

        sdist_python = create_environment(root, "sdist")
        install_and_smoke(sdist_python, sdists[0])

        app_python = create_environment(root, "wheel-app")
        run(app_python, "-m", "pip", "install", f"{wheels[0]}[app]")
        run(app_python, "-m", "pip", "check")
        run(
            app_python,
            "-c",
            "from importlib.metadata import entry_points; "
            "import simupod.mcp_server, simupod.viz.server; "
            "eps={e.name:e.value for e in entry_points(group='console_scripts') "
            "if e.name.startswith('simupod-')}; "
            "assert eps == {'simupod-mcp':'simupod.cli:mcp', "
            "'simupod-serve-viz':'simupod.cli:serve_viz'}, eps",
        )

    print("exact wheel, sdist, CLI, and app-extra artifact smokes passed")


if __name__ == "__main__":
    main()
