#!/usr/bin/env python3
"""Print the SDK version and optionally require a matching v<version> tag."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def package_version() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    if "version" in project:
        return project["version"]
    source = (ROOT / "simupod/__about__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.M)
    if not match:
        raise ValueError("could not read simupod version")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    args = parser.parse_args()
    version = package_version()
    if args.tag and args.tag != "v" + version:
        raise SystemExit(
            f"release tag {args.tag!r} does not match package version v{version}"
        )
    print(version)


if __name__ == "__main__":
    main()
