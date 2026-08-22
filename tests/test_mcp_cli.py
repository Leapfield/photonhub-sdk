"""Optional console commands remain safe and actionable in a base install."""

from __future__ import annotations

import subprocess
import sys

import pytest

from .helpers import PACKAGE_ROOT


def test_console_entry_points_use_dependency_light_wrappers():
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'photonhub-mcp = "photonhub._mcp_cli:main"' in pyproject
    assert 'photonhub-serve-viz = "photonhub._viz_cli:main"' in pyproject


def test_base_install_missing_mcp_exits_cleanly(subprocess_env):
    code = """
import importlib.abc
import sys

class BlockMcp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mcp" or fullname.startswith("mcp."):
            raise ModuleNotFoundError("No module named 'mcp'", name="mcp")
        return None

sys.meta_path.insert(0, BlockMcp())
from photonhub._mcp_cli import main
raise SystemExit(main([]))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=subprocess_env,
    )
    assert proc.returncode == 2
    assert "pip install 'photonhub[mcp]'" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""


def test_mcp_extra_loads_the_server():
    pytest.importorskip("mcp")
    from photonhub import mcp_server

    assert callable(mcp_server.main)
    assert mcp_server.mcp.name == "photonhub"


def test_mcp_preload_accepts_result_dir_environment(monkeypatch):
    pytest.importorskip("mcp")
    from photonhub import mcp_server

    loaded = object()
    seen = []
    monkeypatch.setenv("PHOTONHUB_RESULT_DIR", "/example/result")
    monkeypatch.setattr(
        mcp_server.service,
        "load_result",
        lambda path: seen.append(path) or loaded,
    )
    monkeypatch.setattr(mcp_server.mcp, "run", lambda: None)
    mcp_server._state["data"] = None

    try:
        assert mcp_server.main([]) == 0
        assert seen == ["/example/result"]
        assert mcp_server._state["data"] is loaded
    finally:
        mcp_server._state["data"] = None


@pytest.mark.parametrize("missing", ["fastapi", "uvicorn"])
def test_base_install_missing_server_dependency_exits_cleanly(
    subprocess_env, missing
):
    code = f"""
import importlib.abc
import sys

class BlockDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {missing!r} or fullname.startswith({missing!r} + "."):
            raise ModuleNotFoundError(
                "No module named " + {missing!r}, name={missing!r}
            )
        return None

sys.meta_path.insert(0, BlockDependency())
from photonhub._viz_cli import main
raise SystemExit(main(["--no-open"]))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=subprocess_env,
    )
    assert proc.returncode == 2
    assert "pip install 'photonhub[server]'" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""
