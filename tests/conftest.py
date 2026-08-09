import os
from pathlib import Path

import pytest

from .helpers import (
    EXAMPLE_SPEC,
    FRESNEL_SPEC,
    PACKAGE_ROOT,
    REPO_ROOT,
    make_pw_sim,
    make_sim,
)

# Never let package tests silently exercise an ignored binary built from a
# different commit. Explicit solver paths/environment overrides remain usable.
os.environ.setdefault("PHOTONHUB_REQUIRE_SOURCE_MATCH", "1")
# A large host CPU count can make tiny solver subprocesses slower by orders of
# magnitude through OpenMP oversubscription. Keep test runs bounded and let an
# explicit caller setting win.
os.environ.setdefault("OMP_NUM_THREADS", str(min(os.cpu_count() or 1, 8)))


@pytest.fixture
def example_spec_path() -> Path:
    if not EXAMPLE_SPEC.is_file():
        pytest.skip(f"golden example not found: {EXAMPLE_SPEC}")
    return EXAMPLE_SPEC


@pytest.fixture
def fresnel_spec_path() -> Path:
    if not FRESNEL_SPEC.is_file():
        pytest.skip(f"golden example not found: {FRESNEL_SPEC}")
    return FRESNEL_SPEC


@pytest.fixture
def subprocess_env() -> dict:
    """Environment for `python -m photonhub.schema ...` subprocesses so the
    in-tree package is importable without installation."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.fixture
def tiny_sim():
    return make_sim()
