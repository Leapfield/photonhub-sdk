"""MCP facade for the local agent tools over the viz core.

Exposes local result inspection and run capabilities as MCP tools. Data and
compute stay on the user's machine.

Run standalone:  photonhub-mcp [result-dir]
Or point an MCP client's command at it (stdio transport).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import _mcp_tools
from ._env import env
from .error_codes import CodedError
from .runners.phsolver import SolverRunError, device_args
from .viz import service

mcp = FastMCP("photonhub", instructions=(
    "Call capabilities first. "
    "Call validate_spec before run_simulation. "
    "Read result_summary before quoting a number. "
    "No tool spends money or calls the network."
))
_state: dict = {"data": None}


def _data():
    if _state["data"] is None:
        raise CodedError("no-result-loaded", "no result loaded — call open_result(result_dir) first")
    return _state["data"]


_mcp_tools.register(mcp, _data)


@mcp.tool()
@_mcp_tools.coded_errors
def open_result(result_dir: str) -> dict:
    """Load an FDTD result bundle (a directory with manifest.json) as the current
    result. Returns a summary: run stats, monitors, and whether a 3D scene exists."""
    _state["data"] = service.load_result(result_dir)
    return service.session(_state["data"])


@mcp.tool()
@_mcp_tools.coded_errors
def describe_result() -> dict:
    """Summary of the currently-loaded result: run stats (Mcells/s, setup/solve
    seconds, steps), grid, provenance (device), and the monitor list."""
    return service.session(_data())


@mcp.tool()
@_mcp_tools.coded_errors
def list_monitors() -> list:
    """The current result's monitors with slider metadata: components, frequencies,
    time samples, and spatial axes."""
    return service.monitor_catalog(_data())


@mcp.tool()
@_mcp_tools.coded_errors
def field_stats(monitor: str, field: str = "Ex", val: str = "abs",
                freq: Optional[float] = None, time: Optional[float] = None,  # noqa: UP045
                axis: Optional[str] = None, pos: Optional[float] = None) -> dict:  # noqa: UP045
    """Statistics of a displayed field cut-plane: extrema, mean, peak location
    (µm), non-finite count, and an unweighted ``sample_sum_squares`` diagnostic.
    The latter is not a physical energy integral. This is the numeric window to
    reason over *without* the large raw array. ``field`` is Ex..Hz or a raw-Yee
    derived E/intensity/H norm; ``val`` is real/imag/abs/phase; ``axis``+``pos``
    pick the cut plane of a volumetric monitor (µm)."""
    return service.field_stats(_data(), monitor, field=field, val=val, freq=freq,
                               time=time, axis=axis, pos=pos)


@mcp.tool()
@_mcp_tools.coded_errors
def get_spectrum(monitor: str) -> dict:
    """A flux/DFT monitor's spectrum as plain numbers: wavelength_nm + value."""
    return service.spectrum_values(_data(), monitor)


@mcp.tool()
@_mcp_tools.coded_errors
def run_simulation(spec: str, output_dir: str, device: str = "cpu") -> dict:
    """Run an FDTD simulation locally and load its result. ``spec`` is a Simulation
    wire-JSON string or a path to a .json file; ``device`` is 'cpu' (default) or
    'gpu'. Returns the result directory, monitors, and run stats."""
    from . import Simulation, run_local

    try:
        device_args(device)
    except (SolverRunError, AttributeError) as exc:
        raise CodedError("invalid-argument", str(exc)) from exc
    try:
        text = spec if spec.lstrip().startswith("{") else Path(spec).expanduser().read_text()
    except (OSError, UnicodeError) as exc:
        raise CodedError("spec-rejected", str(exc)) from exc
    try:
        sim = Simulation.from_wire_json(text)
        sim.check_runnable()
    except (ValueError, TypeError, KeyError) as exc:
        message = exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)
        raise CodedError("spec-rejected", message) from exc
    data = run_local(sim, output_dir=output_dir, device=device)
    _state["data"] = data
    return {
        "output_dir": str(data.output_dir),
        "monitors": list(data.monitor_names),
        "run": data.manifest.get("run", {}),
    }


def main(argv=None) -> int:
    from ._mcp_cli import build_parser

    args = build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    preload = args.result_dir or env("RESULT_DIR")
    if preload:
        # A result named at start that does not load is a configuration error:
        # stop with a non-zero status rather than serve without it.
        try:
            _state["data"] = service.load_result(preload)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"photonhub-mcp: cannot open result {preload!r}: {e}", file=sys.stderr)
            return 2
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
