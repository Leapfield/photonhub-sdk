#!/usr/bin/env python3
"""Inspect release archives before they reach the credentialed publish job."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from package_version import package_version


FORBIDDEN_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".dylib", ".hip", ".so", ".dll"}


def safe(names: list[str]) -> None:
    for raw in names:
        path = PurePosixPath(raw)
        lowered = {part.lower() for part in path.parts}
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path {raw}")
        if "engine" in lowered or path.name == "phsolver":
            raise ValueError(f"private solver content present: {raw}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"native source or binary present: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    version = package_version()
    wheels = list(args.dist.glob(f"simupod-{version}-*.whl"))
    sdists = list(args.dist.glob(f"simupod-{version}.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one simupod wheel and one source archive")

    wheel = wheels[0]
    dist_info = f"simupod-{version}.dist-info"
    with zipfile.ZipFile(wheel) as bundle:
        names = bundle.namelist()
        safe(names)
        required = {
            "simupod/py.typed",
            "simupod/schemas/simulation_v1.json",
            f"{dist_info}/METADATA",
            f"{dist_info}/licenses/LICENSE",
        }
        if missing := required - set(names):
            raise ValueError(f"wheel is missing {sorted(missing)}")
        metadata = BytesParser(policy=default).parsebytes(
            bundle.read(f"{dist_info}/METADATA")
        )
    if metadata.get("Name") != "simupod" or metadata.get("Version") != version:
        raise ValueError("wheel name/version metadata does not match the source")
    if metadata.get("License-Expression") != "Apache-2.0":
        raise ValueError("wheel must carry the Apache-2.0 license expression")

    sdist = sdists[0]
    prefix = f"simupod-{version}/"
    with tarfile.open(sdist, "r:gz") as bundle:
        names = bundle.getnames()
        safe(names)
    required = {
        prefix + "LICENSE",
        prefix + "README.md",
        prefix + "pyproject.toml",
        prefix + "simupod/py.typed",
        prefix + "simupod/schemas/simulation_v1.json",
    }
    if missing := required - set(names):
        raise ValueError(f"source archive is missing {sorted(missing)}")
    print(f"verified release artifacts for simupod {version}")


if __name__ == "__main__":
    main()
