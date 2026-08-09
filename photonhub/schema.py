"""Emit / check the generated JSON Schema (schemas/GOVERNANCE.md rule 1).

The pydantic models are the single source of truth;
``schemas/simulation_v1.json`` is generated:

    python -m photonhub.schema emit  [path]   # (re)write the schema file
    python -m photonhub.schema check [path]   # exit 1 if the file is stale

Default path: <repo root>/schemas/simulation_v1.json, resolved relative to
this source tree (development checkouts; pass an explicit path otherwise).
"""

import argparse
import json
import sys
from pathlib import Path

from .components import Simulation

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "simulation_v1.json"


def schema_text() -> str:
    """Canonical serialization: 2-space indent, sorted keys, trailing newline."""
    return json.dumps(Simulation.model_json_schema(), indent=2, sort_keys=True) + "\n"


def emit(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema_text(), encoding="utf-8")


def check(path: Path) -> int:
    """0 if ``path`` matches the models, 1 otherwise (CI gate)."""
    expected = schema_text()
    if not path.is_file():
        print(f"schema check FAILED: {path} does not exist; "
              "run 'python -m photonhub.schema emit'", file=sys.stderr)
        return 1
    if path.read_text(encoding="utf-8") != expected:
        print(f"schema check FAILED: {path} is stale; "
              "run 'python -m photonhub.schema emit'", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m photonhub.schema",
        description="Generate or verify schemas/simulation_v1.json from the "
                    "pydantic models.")
    parser.add_argument("command", choices=("emit", "check"))
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_SCHEMA_PATH)
    args = parser.parse_args(argv)
    if args.command == "emit":
        emit(args.path)
        print(f"wrote {args.path}")
        return 0
    return check(args.path)


if __name__ == "__main__":
    sys.exit(main())
