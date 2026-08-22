"""MCP facade — the agent tools over the viz core (see ``desktop/PLAN.md``).

Exposes the local result-inspection + run capabilities as MCP tools so any MCP
client (Claude Code, Cursor, Claude Desktop) can inspect simulation results and
*run* simulations — all on the user's machine. Unlike a docs-search MCP, these
are real tools: the data and the compute are local.

Run standalone:  photonhub-mcp [result-dir]
Or point an MCP client's command at it (stdio transport).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ._env import env
from .viz import service

mcp = FastMCP("photonhub")
_state: dict = {"data": None}


def _data():
    if _state["data"] is None:
        raise ValueError("no result loaded — call open_result(result_dir) first")
    return _state["data"]


@mcp.tool()
def open_result(result_dir: str) -> dict:
    """Load an FDTD result bundle (a directory with manifest.json) as the current
    result. Returns a summary: run stats, monitors, and whether a 3D scene exists."""
    _state["data"] = service.load_result(result_dir)
    return service.session(_state["data"])


@mcp.tool()
def describe_result() -> dict:
    """Summary of the currently-loaded result — run stats (Mcells/s, setup/solve
    seconds, steps), grid, provenance (device), and the monitor list."""
    return service.session(_data())


@mcp.tool()
def list_monitors() -> list:
    """The current result's monitors with slider metadata: components, frequencies,
    time samples, and spatial axes."""
    return service.monitor_catalog(_data())


@mcp.tool()
def field_stats(monitor: str, field: str = "Ex", val: str = "abs",
                freq: Optional[float] = None, time: Optional[float] = None,
                axis: Optional[str] = None, pos: Optional[float] = None) -> dict:
    """Statistics of a displayed field cut-plane — extrema, mean, peak location
    (µm), non-finite count, and an unweighted ``sample_sum_squares`` diagnostic.
    The latter is not a physical energy integral. This is the numeric window to
    reason over *without* the large raw array. ``field`` is Ex..Hz or a raw-Yee
    derived E/intensity/H norm; ``val`` is real/imag/abs/phase; ``axis``+``pos``
    pick the cut plane of a volumetric monitor (µm)."""
    return service.field_stats(_data(), monitor, field=field, val=val, freq=freq,
                               time=time, axis=axis, pos=pos)


@mcp.tool()
def get_spectrum(monitor: str) -> dict:
    """A flux/DFT monitor's spectrum as plain numbers: wavelength_nm + value."""
    return service.spectrum_values(_data(), monitor)


@mcp.tool()
def run_simulation(spec: str, output_dir: str, device: str = "cpu") -> dict:
    """Run an FDTD simulation locally and load its result. ``spec`` is a Simulation
    wire-JSON string or a path to a .json file; ``device`` is 'cpu' (default) or
    'gpu'. Returns the result directory, monitors, and run stats."""
    from . import Simulation, run_local

    text = spec if spec.lstrip().startswith("{") else Path(spec).expanduser().read_text()
    sim = Simulation.from_wire_json(text)
    data = run_local(sim, output_dir=output_dir, device=device)
    _state["data"] = data
    return {
        "output_dir": str(data.output_dir),
        "monitors": list(data.monitor_names),
        "run": data.manifest.get("run", {}),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    preload = argv[0] if argv else env("RESULT_DIR")
    if preload:
        try:
            _state["data"] = service.load_result(preload)
        except Exception as e:  # pragma: no cover - convenience preload
            print(f"[photonhub-mcp] could not preload {preload!r}: {e}", file=sys.stderr)
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
