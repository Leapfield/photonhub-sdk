"""Workbench notebook kernel — scriptable setup, run, and analysis.

One persistent Python namespace executes user cells sequentially in a spawned
worker process. The ``wb`` bridge object connects scripts to the same
authoritative workspace and run machinery the GUI uses: ``wb.apply(spec)``
publishes through the ordinary workspace sequence (so the 3D preview refreshes),
and ``wb.run()`` produces the same sealed, ledger-backed result as the Run
dialog.

Trust model: cells are arbitrary user Python executed locally with the user's
own privileges — the same trust level as the user running a script in a
terminal next to the app.  The HTTP endpoints live under ``/api`` and are
therefore behind the desktop capability middleware in packaged builds; nothing
here is reachable cross-origin (see the middleware in :mod:`server`).
"""

from __future__ import annotations

import ast
import base64
import copy
import io
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import atexit
from pathlib import Path
from typing import Any, Callable, Optional

from .notebook_channel import ChannelError, read_frame, write_frame, worker_environment

_NOTEBOOK_DOC_VERSION = 1
_STREAM_LIMIT = 200_000       # chars kept per stream output
_REPR_LIMIT = 20_000          # chars kept for a result repr
_LIVE_OUTPUT_LIMIT = 30_000_000  # total rich-output bytes retained in a cell

_WELCOME_CODE = """\
# PhotonHub notebook — the live workspace is available as `wb`.
#   wb.spec              current Simulation spec (dict, same as the Export tab)
#   wb.apply(spec)       validate + push a spec to the workbench (3D view updates)
#   res = wb.run()       run the current setup locally; blocks, returns the result
#   res = wb.result()    the currently open result, if any
#   res.monitors         monitor names; res[name] -> xarray.DataArray
#   wb.show(fig)         display a plotly figure dict or matplotlib figure
# The two starter cells below are a runnable tour: edit-and-apply, then
# run-and-plot. Run this cell first, then them, or "Run all".
spec = wb.spec
print("domain size (um):", spec["size_um"])
print("structures:", len(spec.get("structures", [])))
"""

_TOUR_APPLY_CODE = """\
# The whole workbench edits this one dict. Change it here and push it back —
# the 3D view, Setup panels, and the cost estimate adopt it immediately (an
# invalid spec raises and changes nothing).
spec = wb.spec
spec["run"]["n_steps"] = 1200            # try a slightly longer run
wb.apply(spec)
print("applied: n_steps ->", wb.spec["run"]["n_steps"])
"""

_TOUR_RUN_CODE = """\
# Run the current setup on the local CPU solver — the same sealed path as the
# Run dialog (seconds for the starter scene) — then plot a monitor inline.
res = wb.run()
print("monitors:", res.monitors)
name = "output_flux" if "output_flux" in res.monitors else res.monitors[0]
arr = res[name]
print(name, "| dims:", arr.dims, "| shape:", tuple(arr.shape))
if arr.ndim == 1 and "f" in arr.dims:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(np.asarray(arr["f"].values) / 1e12, np.asarray(arr.values), "o-")
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel(f"{name} (relative)")
    ax.set_title("plotted from the Workbench notebook")
    wb.show(fig)
"""


class WorkbenchHooks:
    """Server-side capabilities a kernel needs; supplied by ``create_app``.

    Every callable raises plain ``LookupError`` / ``ValueError`` /
    ``RuntimeError`` so cell tracebacks stay framework-free.
    """

    def __init__(self, *,
                 get_spec: Callable[[], dict],
                 apply_spec: Callable[[dict], list],
                 run_local: Callable[..., tuple[Any, dict]],
                 cancel_active_run: Callable[[], None],
                 get_result: Callable[[], Optional[tuple[Any, dict]]]):
        self.get_spec = get_spec
        self.apply_spec = apply_spec
        self.run_local = run_local
        self.cancel_active_run = cancel_active_run
        self.get_result = get_result


class WorkbenchResult:
    """Thin analysis handle over a result bundle.

    ``result[name]`` returns the monitor's ``xarray.DataArray`` exactly as the
    library does; ``session`` is the same catalog the Results UI shows.
    """

    def __init__(self, data, session: dict):
        self._data = data
        self.session = session

    @property
    def monitors(self) -> list[str]:
        return [str(m.get("name")) for m in self.session.get("monitors", [])]

    @property
    def output_dir(self) -> str:
        return str(self.session.get("output_dir") or self._data.output_dir)

    def __getitem__(self, name: str):
        return self._data[name]

    def __repr__(self) -> str:
        run = self.session.get("run", {}) or {}
        steps = run.get("steps_run")
        stats = f", steps_run={steps}" if steps is not None else ""
        return (f"WorkbenchResult(monitors={self.monitors!r}{stats}, "
                f"output_dir={self.output_dir!r})")


class Workbench:
    """``wb`` — the bridge between notebook cells and the Workbench GUI."""

    def __init__(self, kernel: "NotebookKernel"):
        self._kernel = kernel

    @property
    def spec(self) -> dict:
        """A deep copy of the current canonical workspace spec."""
        return copy.deepcopy(self._kernel._hooks.get_spec())

    def apply(self, spec: dict) -> None:
        """Validate ``spec`` and publish it as the live workspace draft.

        The GUI adopts the new workspace and refreshes the 3D scene. Schema
        problems raise with the same pydantic messages the GUI shows.
        """
        if not isinstance(spec, dict):
            raise TypeError("wb.apply expects the simulation spec as a dict")
        messages = self._kernel._hooks.apply_spec(copy.deepcopy(spec))
        for message in messages or []:
            print(f"warning: {message}")
        self._kernel.bump_workspace_rev()

    def run(self, spec: Optional[dict] = None, *, device: str = "cpu",
            timeout_s: Optional[float] = None) -> WorkbenchResult:
        """Run a simulation locally and block until it finishes.

        Without ``spec`` the current workspace setup runs, exactly like the Run
        dialog. The run is recorded in the durable ledger; interrupting the
        cell cancels the solver.
        """
        if spec is not None and not isinstance(spec, dict):
            raise TypeError("wb.run expects the simulation spec as a dict")
        from ..runners.phsolver import device_args
        device = str(device or "cpu").strip()
        device_args(device)
        if timeout_s in (None, "", 0):
            timeout_s = None
        else:
            try:
                timeout_s = float(timeout_s)
            except (TypeError, ValueError) as exc:
                raise ValueError("timeout_s must be a positive number") from exc
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                raise ValueError("timeout_s must be a positive number")
        try:
            data, session = self._kernel._hooks.run_local(
                spec=copy.deepcopy(spec) if spec is not None else None,
                device=device, timeout_s=timeout_s,
                on_progress=self._print_progress)
        except KeyboardInterrupt:
            print("interrupt — cancelling the running simulation…")
            self._kernel._hooks.cancel_active_run()
            raise
        return WorkbenchResult(data, session)

    def result(self) -> Optional[WorkbenchResult]:
        """The currently open result (after ``wb.run`` or opening a bundle)."""
        loaded = self._kernel._hooks.get_result()
        if loaded is None:
            return None
        data, session = loaded
        return WorkbenchResult(data, session)

    def show(self, figure: Any) -> None:
        """Display a figure inline: a plotly ``{data, layout}`` dict, an object
        with ``to_plotly_json()``, or a matplotlib figure."""
        output = _figure_output(figure)
        if output is None:
            raise TypeError(
                "wb.show expects a plotly {'data': [...], 'layout': {...}} dict, "
                "an object with to_plotly_json(), or a matplotlib figure")
        self._kernel.emit_output(output)
        _release_pyplot_figure(figure)

    def _print_progress(self, progress: Optional[dict], status: str) -> None:
        self._kernel.print_progress_line(_progress_text(progress, status))

    def __repr__(self) -> str:
        return ("Workbench(wb.spec, wb.apply(spec), wb.run(device='cpu'), "
                "wb.result(), wb.show(figure))")


def _progress_text(progress: Optional[dict], status: str) -> str:
    if not isinstance(progress, dict):
        return f"run {status}"
    step = progress.get("step")
    n_steps = progress.get("n_steps") or progress.get("num_steps")
    phase = progress.get("phase")
    parts = [f"run {status}"]
    if phase:
        parts.append(str(phase))
    if isinstance(step, (int, float)) and isinstance(n_steps, (int, float)) and n_steps:
        parts.append(f"step {int(step)}/{int(n_steps)}")
    rate = progress.get("mcells_per_s")
    if isinstance(rate, (int, float)):
        parts.append(f"{float(rate):.1f} Mcells/s")
    return " · ".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated {len(text) - limit} characters]"


def _safe_hook_message(message: str) -> str:
    """Do not let a hook exception return the sidecar's private cloud key."""
    try:
        from ..cloud import config
        secrets = [getattr(config._CONFIG, "api_key", None)]
        secrets.extend(value for name, value in os.environ.items()
                       if name.upper().endswith(("_KEY", "_SECRET", "_TOKEN", "_CREDENTIAL")))
        for value in secrets:
            if isinstance(value, str) and len(value) >= 8:
                message = message.replace(value, "[redacted]")
    except Exception:
        return "Workbench operation failed"
    return message


def _figure_output(value: Any, *, raise_errors: bool = False) -> Optional[dict]:
    """Convert a user object into a figure/image output dict, or ``None``."""
    candidate = value
    to_plotly = getattr(candidate, "to_plotly_json", None)
    if callable(to_plotly):
        try:
            candidate = to_plotly()
        except Exception:
            if raise_errors:
                raise
            return None
    if isinstance(candidate, dict) and isinstance(candidate.get("data"), list):
        try:  # keep the wire JSON-clean (numpy arrays -> lists)
            cleaned = json.loads(json.dumps(candidate, default=_json_fallback))
        except Exception:
            # A plain container can resemble a figure but still have keys or
            # cycles that JSON cannot represent. Its last value is its repr.
            if raise_errors and callable(to_plotly):
                raise
            return None
        return {"type": "figure", "figure": {
            "data": cleaned.get("data", []),
            "layout": cleaned.get("layout", {}) or {},
        }}
    savefig = getattr(value, "savefig", None)
    if callable(savefig):
        buffer = io.BytesIO()
        try:
            savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        except Exception:
            if raise_errors:
                raise
            return None
        return {"type": "image", "mime": "image/png",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii")}
    return None


def _release_pyplot_figure(value: Any) -> None:
    """Drop an already-displayed matplotlib figure from pyplot's registry.

    ``_capture_pyplot_figures`` sweeps every figure still open at the end of a
    cell, so a figure that was displayed explicitly — ``wb.show(fig)``, or a
    trailing bare ``fig`` — would otherwise be rendered a second time.  The
    Figure object itself stays usable (``fig.savefig(...)`` still works); only
    pyplot's book-keeping entry goes, and the end-of-cell sweep closes every
    figure anyway.
    """
    if not callable(getattr(value, "savefig", None)):
        return                      # plotly dicts and friends: nothing to do
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    try:
        if getattr(value, "number", None) in plt.get_fignums():
            plt.close(value)
    except Exception:  # noqa: BLE001 — displaying must never fail on cleanup
        pass


def _json_fallback(value: Any):
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


_LINE_MAGIC_RE = re.compile(r"^\s*%[A-Za-z][A-Za-z0-9_]*(?:\s.*)?$")
_CELL_MAGIC_RE = re.compile(r"^\s*%%[A-Za-z][A-Za-z0-9_]*(?:\s.*)?$")
_SHELL_ESCAPE_RE = re.compile(r"^\s*!\s*\S.*$")


def _rewrite_magics(code: str) -> tuple[str, list[tuple[int, str]],
                                        list[tuple[int, str]]]:
    """Blank out IPython line magics so a pasted gallery cell parses.

    Every notebook under ``examples/notebooks`` opens with ``%matplotlib
    inline`` — a line this kernel has no reason to honour (the backend is
    already Agg and figures are captured automatically) and no reason to choke
    on.  Magic lines are blanked rather than deleted so traceback line numbers
    keep pointing at the line the user actually wrote.  Cell magics and shell
    escapes change what the whole cell means, so they are reported rather than
    quietly dropped.  Returns ``(rewritten source, ignored, unsupported)``,
    each entry a ``(1-based line number, text)`` pair.
    """
    lines = code.splitlines(keepends=True)
    ignored: list[tuple[int, str]] = []
    unsupported: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if _CELL_MAGIC_RE.match(line) or _SHELL_ESCAPE_RE.match(line):
            unsupported.append((index + 1, line.strip()))
        elif _LINE_MAGIC_RE.match(line):
            ignored.append((index + 1, line.strip()))
            lines[index] = "\n" if line.endswith("\n") else ""
    return "".join(lines), ignored, unsupported


def _parse_cell(code: str, filename: str) -> tuple[ast.Module, list[str]]:
    """Parse a cell, tolerating the IPython line magics notebooks open with.

    Valid Python parses untouched, so a ``%`` inside a string literal is never
    even considered.  Beyond that, a magic is only ever blamed when the
    interpreter's own first complaint lands on the magic line itself — any
    other syntax error is reported exactly as Python reported it, rather than
    being rewritten into a guess.
    """
    try:
        return ast.parse(code, filename=filename), []
    except SyntaxError as original:
        rewritten, ignored, unsupported = _rewrite_magics(code)
        flagged = {line for line, _ in ignored} | {line for line, _ in unsupported}
        if original.lineno not in flagged:
            raise                     # a real syntax error somewhere else
        if unsupported:
            _, text = unsupported[0]
            raise SyntaxError(
                "the Workbench notebook runs plain Python, and "
                f"{text!r} is an IPython cell magic or shell escape. Use the "
                "Python equivalent (subprocess.run for a shell command), or "
                "run that cell in Jupyter.") from None
        # Line numbers survive blanking, so anything still broken reports
        # against the line the user wrote.
        return ast.parse(rewritten, filename=filename), [t for _, t in ignored]


def _capture_pyplot_figures(limit: int = 1000) -> list[dict]:
    """Collect and close any figures pyplot accumulated during the cell."""
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    outputs = []
    try:
        for num in list(plt.get_fignums()):
            if len(outputs) >= limit:
                break
            output = _figure_output(plt.figure(num))
            if output is not None:
                outputs.append(output)
        plt.close("all")
    except Exception:
        return outputs
    return outputs


class NotebookKernel:
    """Sequential cell executor with a persistent namespace.

    HTTP threads mutate the document and queue under ``_lock``. One daemon
    scheduler serializes execution through a lazily spawned interpreter.
    """

    def __init__(self, hooks: WorkbenchHooks,
                 persist_path: Optional[Path] = None,
                 execution_available: bool = True):
        self._hooks = hooks
        self.execution_available = execution_available
        self._persist_path = Path(persist_path) if persist_path else None
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._cells: list[dict] = []
        self._queue: list[str] = []
        self._execution_count = 0
        self._workspace_rev = 0
        self._generation = 0
        self._running_cell: Optional[str] = None
        self._interrupted_cell: Optional[str] = None
        self._interrupt_sequence = 0
        self._execute_sent = False
        self._code_running = False
        self._interrupt_timer: Optional[threading.Event] = None
        self._live: Optional[dict] = None   # live capture for the running cell
        self._process: Optional[subprocess.Popen] = None
        self._start_lock = threading.Lock()
        self._active_run_process: Optional[subprocess.Popen] = None
        self._forced_restart_count: Optional[int] = None
        self._hook_slot = threading.Lock()
        self._hook_pending = 0
        self._hook_pending_process: Optional[subprocess.Popen] = None
        self._channel_write_lock = threading.Lock()
        self._stopping = False
        if not self._restore():
            self._cells = [self._new_cell(_WELCOME_CODE),
                           self._new_cell(_TOUR_APPLY_CODE),
                           self._new_cell(_TOUR_RUN_CODE)]
        self._worker = self._spawn_worker()
        atexit.register(self.close)

    # ------------------------------------------------------------- document
    def _new_cell(self, code: str = "") -> dict:
        return {"id": uuid.uuid4().hex[:12], "code": code, "status": "idle",
                "execution_count": None, "outputs": [], "elapsed_s": None}

    def _cell(self, cell_id: str) -> dict:
        for cell in self._cells:
            if cell["id"] == cell_id:
                return cell
        raise LookupError(f"unknown notebook cell: {cell_id}")

    def snapshot(self) -> dict:
        with self._lock:
            cells = []
            for cell in self._cells:
                view = {key: cell[key] for key in
                        ("id", "code", "status", "execution_count", "elapsed_s")}
                view["outputs"] = self._outputs_view(cell)
                cells.append(view)
            return {
                "cells": cells,
                "execution_available": self.execution_available,
                "kernel": {
                    "status": "busy" if (self._running_cell or self._queue) else "idle",
                    "running_cell": self._running_cell,
                    "queue": list(self._queue),
                    "execution_count": self._execution_count,
                },
                "workspace_rev": self._workspace_rev,
            }

    def _outputs_view(self, cell: dict) -> list:
        live = self._live
        if live is None or live["cell_id"] != cell["id"]:
            return list(cell["outputs"])
        outputs = list(live["outputs"])
        for name in ("stdout", "stderr"):
            text = live[name].getvalue()
            if text:
                outputs.append({"type": "stream", "name": name,
                                "text": _truncate(text, _STREAM_LIMIT)})
        return outputs

    def add_cell(self, *, after_id: Optional[str] = None,
                 code: str = "") -> dict:
        with self._lock:
            cell = self._new_cell(code)
            index = len(self._cells)
            if after_id is not None:
                index = self._cells.index(self._cell(after_id)) + 1
            self._cells.insert(index, cell)
            self._persist()
            return self.snapshot()

    def update_cell(self, cell_id: str, code: str) -> dict:
        with self._lock:
            self._cell(cell_id)["code"] = str(code)
            self._persist()
            return self.snapshot()

    def delete_cell(self, cell_id: str) -> dict:
        with self._lock:
            cell = self._cell(cell_id)
            self._cells.remove(cell)
            if cell_id in self._queue:
                self._queue.remove(cell_id)
            if not self._cells:
                self._cells.append(self._new_cell())
            self._persist()
            return self.snapshot()

    def move_cell(self, cell_id: str, index: int) -> dict:
        with self._lock:
            cell = self._cell(cell_id)
            self._cells.remove(cell)
            self._cells.insert(max(0, min(int(index), len(self._cells))), cell)
            self._persist()
            return self.snapshot()

    # ------------------------------------------------------------ execution
    def run_cell(self, cell_id: str, code: Optional[str] = None) -> dict:
        if not self.execution_available:
            raise PermissionError("notebook execution is disabled")
        with self._lock:
            cell = self._cell(cell_id)
            if code is not None:
                cell["code"] = str(code)
            if cell_id != self._running_cell and cell_id not in self._queue:
                cell["status"] = "queued"
                self._queue.append(cell_id)
                self._wake.notify_all()
            self._persist()
            return self.snapshot()

    def run_all(self) -> dict:
        if not self.execution_available:
            raise PermissionError("notebook worker is unavailable; cell execution is disabled")
        with self._lock:
            for cell in self._cells:
                if cell["id"] != self._running_cell and cell["id"] not in self._queue:
                    cell["status"] = "queued"
                    self._queue.append(cell["id"])
            self._wake.notify_all()
            return self.snapshot()

    def interrupt(self) -> dict:
        with self._lock:
            for cell_id in self._queue:
                try:
                    self._cell(cell_id)["status"] = "idle"
                except LookupError:
                    pass
            self._queue.clear()
            cell_id = self._running_cell
            process = self._process
            generation = self._generation
            sent = self._execute_sent
            count = self._execution_count
            code_running = self._code_running
            if cell_id is not None:
                self._interrupted_cell = cell_id
                self._interrupt_sequence += 1
        if cell_id is not None and process is not None:
            self._cancel_notebook_run(process)
            try:
                if sent and code_running:
                    self._signal_interrupt(process)
            except (OSError, ValueError):
                with self._lock:
                    current = (self._generation == generation
                               and self._running_cell == cell_id
                               and self._execution_count == count
                               and self._code_running
                               and self._process is process)
                    if current:
                        self._forced_restart_count = count
                if current:
                    self._kill_process(expected=process)
            else:
                if not sent or not code_running:
                    return self.snapshot()
                with self._lock:
                    if (self._interrupt_timer is not None
                            or self._execution_count != count):
                        return self.snapshot()
                    done = threading.Event()
                    self._interrupt_timer = done
                def fallback():
                    if done.wait(2):
                        return
                    with self._lock:
                        stuck = (self._generation == generation
                                 and self._running_cell == cell_id
                                 and self._execution_count == count
                                 and self._code_running
                                 and self._interrupt_timer is done
                                 and self._process is process)
                        if stuck:
                            self._forced_restart_count = count
                    if stuck:
                        self._kill_process(expected=process)
                threading.Thread(target=fallback, name="notebook-interrupt-bound",
                                 daemon=True).start()
        return self.snapshot()

    @staticmethod
    def _signal_interrupt(process: subprocess.Popen) -> None:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)

    def restart(self, *, clear_outputs: bool = False) -> dict:
        with self._lock:
            self._cancel_interrupt_timer()
            self._generation += 1
            self._forced_restart_count = None
            self._wake.notify_all()
            for cell_id in self._queue:
                try:
                    self._cell(cell_id)["status"] = "idle"
                except LookupError:
                    pass
            self._queue.clear()
            if self._running_cell is not None:
                cell = self._cell(self._running_cell)
                cell["status"] = "error"
                cell["outputs"] = [{
                    "type": "error", "ename": "KernelRestart",
                    "evalue": "The kernel restarted while this cell was running.",
                    "traceback": "The kernel restarted while this cell was running.",
                }]
            self._running_cell = None
            self._execute_sent = False
            self._code_running = False
            self._live = None
            if clear_outputs:
                for cell in self._cells:
                    cell.update(status="idle", execution_count=None,
                                outputs=[], elapsed_s=None)
            self._persist()
            snap = self.snapshot()
        with self._lock:
            old_process = self._process
        self._cancel_notebook_run(old_process)
        self._kill_process(expected=old_process)
        with self._lock:
            self._worker = self._spawn_worker()
        return snap

    def bump_workspace_rev(self) -> None:
        with self._lock:
            self._workspace_rev += 1

    def emit_output(self, output: dict) -> None:
        with self._lock:
            live = self._live
            if live is None or live["rich_truncated"]:
                return
            size = len(json.dumps(output, ensure_ascii=False).encode("utf-8"))
            if (len(live["outputs"]) >= 999
                    or live["rich_bytes"] + size > _LIVE_OUTPUT_LIMIT):
                live["rich_truncated"] = True
                live["outputs"].append({"type": "stream", "name": "stderr",
                                        "text": "… [truncated notebook outputs]"})
                return
            live["rich_bytes"] += size
            live["outputs"].append(output)

    def _cancel_interrupt_timer(self) -> None:
        if self._interrupt_timer is not None:
            self._interrupt_timer.set()
            self._interrupt_timer = None

    # ---------------------------------------------------------------- worker
    def _spawn_worker(self) -> threading.Thread:
        worker = threading.Thread(
            target=self._worker_loop, args=(self._generation,),
            name="workbench-notebook", daemon=True)
        worker.start()
        return worker

    def _worker_loop(self, generation: int) -> None:
        while True:
            # A late interrupt can land after the cell finished; it must not
            # silently kill the worker thread while it waits for more work.
            try:
                with self._lock:
                    if generation != self._generation:
                        return
                    while not self._queue:
                        self._wake.wait()
                        if generation != self._generation:
                            return
                    cell_id = self._queue.pop(0)
                    try:
                        cell = self._cell(cell_id)
                    except LookupError:
                        continue
                    self._execution_count += 1
                    count = self._execution_count
                    self._forced_restart_count = None
                    cell["status"] = "running"
                    cell["execution_count"] = count
                    cell["outputs"] = []
                    cell["elapsed_s"] = None
                    self._running_cell = cell_id
                    self._execute_sent = False
                    self._code_running = False
                    self._cancel_interrupt_timer()
                    self._live = {"cell_id": cell_id, "outputs": [],
                                  "stdout": io.StringIO(), "stderr": io.StringIO(),
                                  "rich_bytes": 0, "rich_truncated": False,
                                  "last_progress": None, "last_progress_at": 0.0}
                    code = cell["code"]
                try:
                    outputs, error, elapsed = self._execute_remote(code, count, generation)
                except KeyboardInterrupt:
                    outputs, error, elapsed = [{
                        "type": "error", "ename": "KeyboardInterrupt",
                        "evalue": "interrupted",
                        "traceback": "KeyboardInterrupt",
                    }], True, 0.0
                except (ChannelError, OSError, ValueError, TypeError,
                        subprocess.SubprocessError) as exc:
                    with self._lock:
                        forced = self._forced_restart_count == count
                        interrupted = self._interrupted_cell == cell_id
                        self._forced_restart_count = None
                        self._interrupted_cell = None
                    label = "KernelRestart" if forced or interrupted else "KernelUnavailable"
                    message = ("Notebook restarted after interrupt; Python variables were lost."
                               if forced or interrupted else str(exc))
                    outputs, error, elapsed = [{
                        "type": "error", "ename": label,
                        "evalue": message, "traceback": label,
                    }], True, 0.0
                    with self._lock:
                        current = generation == self._generation
                        failed_process = self._process if current else None
                    if current:
                        self._cancel_notebook_run(failed_process)
                        self._kill_process(expected=failed_process)
                with self._lock:
                    if generation != self._generation:
                        return
                    if self._forced_restart_count == count:
                        message = "Notebook restarted after interrupt; Python variables were lost."
                        outputs = [{"type": "error", "ename": "KernelRestart",
                                    "evalue": message, "traceback": "KernelRestart"}]
                        error = True
                        elapsed = 0.0
                        self._forced_restart_count = None
                    self._interrupted_cell = None
                    self._cancel_interrupt_timer()
                    self._running_cell = None
                    self._execute_sent = False
                    self._code_running = False
                    self._live = None
                    try:
                        cell = self._cell(cell_id)
                    except LookupError:
                        continue
                    cell["outputs"] = outputs
                    cell["status"] = "error" if error else "ok"
                    cell["elapsed_s"] = elapsed
                    self._persist()
            except KeyboardInterrupt:
                continue

    def _start_process(self) -> subprocess.Popen:
        # The start lock serializes spawns, but HTTP snapshots and interrupts
        # can still acquire the kernel lock during a cold frozen start.
        with self._start_lock:
            with self._lock:
                if self._process is not None and self._process.poll() is None:
                    return self._process
                if self._stopping:
                    raise ChannelError("notebook kernel is shutting down")
                generation = self._generation
            frozen = getattr(sys, "frozen", False)
            command = ([sys.executable, "--notebook-worker"] if frozen
                       else [sys.executable, "-P", "-m", "photonhub._viz_cli", "--notebook-worker"])
            environment = worker_environment(os.environ)
            if not frozen:
                environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
            try:
                process = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, env=environment,
                    close_fds=True, start_new_session=(os.name != "nt"),
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
                )
            except OSError:
                raise ChannelError("notebook worker could not start; retry the cell") from None
            with self._lock:
                stale = self._stopping or generation != self._generation
                if not stale:
                    self._process = process
            if stale:
                self._terminate_process(process)
                raise ChannelError("notebook worker start was interrupted; retry the cell")
            ready: queue.Queue = queue.Queue(maxsize=1)
            def read_ready():
                try:
                    ready.put(read_frame(process.stdout))
                except BaseException as exc:
                    ready.put(exc)
            threading.Thread(target=read_ready, name="notebook-worker-startup",
                             daemon=True).start()
            try:
                greeting = ready.get(timeout=60)
            except queue.Empty:
                self._kill_process(expected=process)
                raise ChannelError("notebook worker did not become ready; retry the cell") from None
            if greeting != {"type": "ready"}:
                self._kill_process(expected=process)
                raise ChannelError("notebook worker mode is unavailable; retry the cell")
            with self._lock:
                if self._process is not process or self._stopping:
                    raise ChannelError("notebook worker start was interrupted; retry the cell")
            return process

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        try:
            if process.poll() is None:
                if os.name == "nt":
                    import ctypes
                    directory = ctypes.create_unicode_buffer(32768)
                    if not ctypes.windll.kernel32.GetSystemDirectoryW(directory, len(directory)):
                        raise OSError("Windows system directory is unavailable")
                    tool = os.path.join(directory.value, "taskkill.exe")
                    subprocess.run([tool, "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   env=worker_environment(os.environ), timeout=2, check=False)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        finally:
            # The worker is reaped before closing streams; detached cell
            # children cannot retain the private channel descriptors.
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass

    def _kill_process(self, *, expected=None) -> None:
        with self._lock:
            process = self._process
            if process is None or (expected is not None and process is not expected):
                return
            self._process = None
        self._terminate_process(process)

    def _cancel_notebook_run(self, process) -> None:
        with self._lock:
            owned = process is not None and self._active_run_process is process
        if owned:
            try:
                self._hooks.cancel_active_run()
            except Exception:
                # Interrupt, restart, and shutdown must still reap the worker.
                pass

    def close(self) -> None:
        with self._lock:
            self._stopping = True
            self._cancel_interrupt_timer()
            self._generation += 1
            self._queue.clear()
            self._wake.notify_all()
            process = self._process
        self._cancel_notebook_run(process)
        self._kill_process(expected=process)

    def _execute_remote(self, code: str, count: int,
                        generation: Optional[int] = None) -> tuple[list, bool, float]:
        with self._lock:
            if generation is not None and generation != self._generation:
                raise ChannelError("notebook execution was restarted")
        if len(code) > 1_000_000 or len(code.encode("utf-8")) > 30_000_000:
            message = "notebook cell source is too large"
            return ([{"type": "error", "ename": "NotebookSourceLimit",
                      "evalue": message, "traceback": message}], True, 0.0)
        process = self._start_process()
        if process.stdin is None or process.stdout is None:
            raise ChannelError("notebook worker has no private channel")
        with self._lock:
            if generation is not None and generation != self._generation:
                raise ChannelError("notebook execution was restarted")
            if (self._interrupted_cell is not None
                    and self._running_cell == self._interrupted_cell):
                return ([{"type": "error", "ename": "KeyboardInterrupt",
                          "evalue": "interrupted", "traceback": "KeyboardInterrupt"}], True, 0.0)
        with self._channel_write_lock:
            write_frame(process.stdin, {"type": "execute", "code": code, "count": count})
        with self._lock:
            current = (self._process is process and
                       (generation is None or generation == self._generation))
            if current:
                self._execute_sent = True
                self._code_running = True
            pending_interrupt = current and self._interrupted_cell == self._running_cell
        if pending_interrupt:
            self._signal_interrupt(process)
        while True:
            message = read_frame(process.stdout)
            kind = message.get("type")
            if kind == "code_done":
                if set(message) != {"type", "count"} or message["count"] != count:
                    raise ChannelError("invalid notebook code completion")
                with self._lock:
                    self._code_running = False
                    self._cancel_interrupt_timer()
                continue
            if kind == "live":
                if (set(message) != {"type", "name", "text"}
                        or message["name"] not in {"stdout", "stderr"}
                        or not isinstance(message["text"], str)
                        or len(message["text"]) > 8192):
                    raise ChannelError("invalid notebook live output")
                with self._lock:
                    if self._live is not None:
                        buffer = self._live[message["name"]]
                        if buffer.tell() < _STREAM_LIMIT:
                            buffer.write(message["text"][:_STREAM_LIMIT - buffer.tell()])
                continue
            if kind == "output":
                if set(message) != {"type", "output"} or not self._valid_output(message["output"]):
                    raise ChannelError("invalid notebook rich output")
                self.emit_output(message["output"])
                continue
            if kind == "call":
                self._handle_worker_call(process, message, count=count)
                continue
            if (kind != "result" or set(message) != {"type", "count", "outputs", "error", "elapsed_s"}
                    or type(message["count"]) is not int
                    or message["count"] != count):
                raise ChannelError("unknown notebook worker message")
            outputs = message["outputs"]
            if (not isinstance(outputs, list) or len(outputs) > 1000
                    or not all(self._valid_output(item) for item in outputs)
                    or type(message["error"]) is not bool
                    or type(message["elapsed_s"]) not in {int, float}
                    or not 0 <= message["elapsed_s"] < 1_000_000):
                raise ChannelError("invalid notebook worker result")
            return outputs, message["error"], message["elapsed_s"]

    @staticmethod
    def _valid_output(item: object) -> bool:
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            return False
        kind = item["type"]
        if kind == "stream":
            return (set(item) == {"type", "name", "text"}
                    and item["name"] in {"stdout", "stderr"}
                    and isinstance(item["text"], str) and len(item["text"]) <= 220_000)
        if kind == "result":
            return (set(item) == {"type", "text"} and isinstance(item["text"], str)
                    and len(item["text"]) <= 22_000)
        if kind == "error":
            return (set(item) == {"type", "ename", "evalue", "traceback"}
                    and all(isinstance(item[k], str) for k in ("ename", "evalue", "traceback"))
                    and len(item["ename"]) <= 100 and len(item["evalue"]) <= 22_000
                    and len(item["traceback"]) <= 220_000)
        if kind == "figure":
            return (set(item) == {"type", "figure"} and isinstance(item["figure"], dict)
                    and isinstance(item["figure"].get("data"), list)
                    and isinstance(item["figure"].get("layout"), dict))
        if kind == "image":
            if (set(item) != {"type", "mime", "data"} or item["mime"] != "image/png"
                    or not isinstance(item["data"], str) or len(item["data"]) > 31_000_000):
                return False
            try:
                return base64.b64decode(item["data"], validate=True).startswith(b"\x89PNG\r\n\x1a\n")
            except (ValueError, base64.binascii.Error):
                return False
        return False

    def _handle_worker_call(self, process: subprocess.Popen, message: dict,
                            *, count: Optional[int] = None) -> None:
        with self._lock:
            arrival_interrupt_sequence = self._interrupt_sequence
        name, args = message.get("name"), message.get("args")
        call_id = message.get("id")
        schemas = {"get_spec": set(), "apply_spec": {"spec"},
                   "run_local": {"spec", "device", "timeout_s"},
                   "cancel_run": set(), "get_result": set()}
        if (set(message) != {"type", "id", "name", "args"}
                or type(call_id) is not int or not 0 < call_id < 1_000_000_000
                or name not in schemas
                or not isinstance(args, dict) or set(args) != schemas[name]):
            raise ChannelError("invalid notebook worker call")
        if name == "apply_spec" and not isinstance(args["spec"], dict):
            raise ChannelError("invalid notebook spec")
        if name == "run_local":
            from ..runners.phsolver import SolverRunError, device_args
            try:
                if args["spec"] is not None and not isinstance(args["spec"], dict):
                    raise ValueError
                if not isinstance(args["device"], str) or len(args["device"]) > 80:
                    raise ValueError
                device_args(args["device"])
                timeout = args["timeout_s"]
                if timeout is not None and (type(timeout) not in {int, float}
                                            or not math.isfinite(timeout)
                                            or timeout < 0):
                    raise ValueError
            except (TypeError, ValueError, SolverRunError):
                raise ChannelError("invalid local run request") from None
        def send(value):
            with self._channel_write_lock:
                write_frame(process.stdin, value)

        def call_is_live():
            # Called with _lock held. A later interrupt abandons a queued call,
            # while a call issued by still-running cleanup after an interrupt lives.
            return (count is None or
                    (self._process is process
                     and self._running_cell is not None
                     and self._execution_count == count
                     and self._code_running
                     and self._interrupt_sequence == arrival_interrupt_sequence))

        def interrupted_reply():
            return {"type": "reply", "id": call_id, "ok": False,
                    "error_type": "KeyboardInterrupt", "message": "interrupted"}

        def handle():
            with self._lock:
                live = call_is_live()
                if live and name == "run_local":
                    self._active_run_process = process
            if not live:
                try:
                    send(interrupted_reply())
                except (OSError, ValueError, ChannelError):
                    pass
                return
            try:
                if name == "get_spec":
                    value = self._hooks.get_spec()
                elif name == "apply_spec":
                    value = self._hooks.apply_spec(args["spec"])
                    self.bump_workspace_rev()
                elif name == "run_local":
                    def progress(value, status):
                        if process.poll() is not None:
                            self._cancel_notebook_run(process)
                            raise ChannelError("notebook worker stopped during the local run")
                        if (value is None or isinstance(value, dict)) and isinstance(status, str):
                            send({"type": "progress", "id": call_id,
                                  "value": value, "status": status[:100]})
                            with self._lock:
                                live = self._live
                                if live is not None:
                                    now = time.monotonic()
                                    if (now - live["last_progress_at"] >= 1.0
                                            and live["stdout"].tell() < _STREAM_LIMIT):
                                        live["last_progress_at"] = now
                                        remaining = _STREAM_LIMIT - live["stdout"].tell()
                                        live["stdout"].write((_progress_text(value, status) + "\n")[:remaining])
                    data, session = self._hooks.run_local(
                        spec=args["spec"], device=args["device"],
                        timeout_s=args["timeout_s"], on_progress=progress)
                    value = {"output_dir": str(data.output_dir), "session": session}
                elif name == "cancel_run":
                    value = self._hooks.cancel_active_run()
                else:
                    loaded = self._hooks.get_result()
                    value = ({"output_dir": str(loaded[0].output_dir), "session": loaded[1]}
                             if loaded is not None else None)
                reply = {"type": "reply", "id": call_id, "ok": True, "value": value}
            except (LookupError, ValueError, TypeError, RuntimeError) as exc:
                reply = {"type": "reply", "id": call_id,
                         "ok": False, "error_type": type(exc).__name__,
                         "message": _safe_hook_message(str(exc))[:20000]}
            except Exception:
                if name == "run_local":
                    self._cancel_notebook_run(process)
                reply = {"type": "reply", "id": call_id,
                         "ok": False, "error_type": "RuntimeError",
                         "message": "Workbench operation failed"}
            finally:
                if name == "run_local":
                    with self._lock:
                        if self._active_run_process is process:
                            self._active_run_process = None
            try:
                send(reply)
            except ChannelError:
                try:
                    send({"type": "reply", "id": call_id, "ok": False,
                          "error_type": "ValueError",
                          "message": "Workbench response is too large for a notebook cell"})
                except (OSError, ValueError, ChannelError):
                    self._kill_process(expected=process)
            except (OSError, ValueError):
                self._kill_process(expected=process)

        with self._lock:
            if self._hook_pending_process is not process:
                self._hook_pending_process = process
                self._hook_pending = 0
            if self._hook_pending >= 4:
                raise ChannelError("too many notebook hook calls")
            self._hook_pending += 1
        def bounded_handle():
            try:
                with self._lock:
                    cancel_owned_run = (name == "cancel_run"
                                        and self._active_run_process is process)
                if cancel_owned_run:
                    # Cancellation must reach the idempotent hook while the
                    # run hook still owns the regular call slot.
                    handle()
                elif self._hook_slot.acquire(timeout=2):
                    try:
                        handle()
                    finally:
                        self._hook_slot.release()
                else:
                    try:
                        with self._lock:
                            live = call_is_live()
                        send(({"type": "reply", "id": call_id, "ok": False,
                               "error_type": "RuntimeError",
                               "message": "Previous Workbench call is still finishing; try again."}
                              if live else interrupted_reply()))
                    except (OSError, ValueError, ChannelError):
                        self._kill_process(expected=process)
            finally:
                with self._lock:
                    if self._hook_pending_process is process:
                        self._hook_pending -= 1
        try:
            threading.Thread(target=bounded_handle, name=f"notebook-hook-{name}", daemon=True).start()
        except RuntimeError:
            with self._lock:
                if self._hook_pending_process is process:
                    self._hook_pending -= 1
            send({"type": "reply", "id": call_id, "ok": False,
                  "error_type": "RuntimeError", "message": "Workbench operation could not start"})

    # ----------------------------------------------------------- persistence
    def _persist(self) -> None:
        """Atomically retain cell sources (outputs are session-scoped)."""
        if self._persist_path is None:
            return
        payload = {
            "version": _NOTEBOOK_DOC_VERSION,
            "cells": [{"id": cell["id"], "code": cell["code"],
                       "execution_count": cell["execution_count"]}
                      for cell in self._cells],
        }
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=self._persist_path.parent,
                    prefix=f".{self._persist_path.name}.", suffix=".tmp",
                    delete=False) as tmp:
                json.dump(payload, tmp, ensure_ascii=False)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            os.replace(tmp_name, self._persist_path)
        except (OSError, TypeError, ValueError):
            # Persistence is a convenience; execution must not fail with it.
            pass

    def _restore(self) -> bool:
        if self._persist_path is None or not self._persist_path.is_file():
            return False
        try:
            record = json.loads(self._persist_path.read_text(encoding="utf-8"))
            if (not isinstance(record, dict)
                    or record.get("version") != _NOTEBOOK_DOC_VERSION):
                raise ValueError("unsupported notebook document")
            cells = []
            for item in record.get("cells", []):
                if not isinstance(item, dict) or not isinstance(item.get("code"), str):
                    raise ValueError("invalid notebook cell record")
                cell = self._new_cell(item["code"])
                if isinstance(item.get("id"), str) and item["id"]:
                    cell["id"] = item["id"][:32]
                count = item.get("execution_count")
                if isinstance(count, int) and count > 0:
                    cell["execution_count"] = count
                    self._execution_count = max(self._execution_count, count)
                cells.append(cell)
            if not cells:
                return False
            self._cells = cells
            return True
        except (OSError, ValueError):
            return False


def _error_output(exc: BaseException, count: int) -> dict:
    frames = traceback.extract_tb(exc.__traceback__)
    # Drop kernel frames; keep everything from the user's cell inward.
    start = next((i for i, frame in enumerate(frames)
                  if frame.filename.startswith("<cell")), 0)
    rendered = "".join(traceback.format_list(frames[start:]))
    header = f"Traceback (cell {count}, most recent call last):\n" if rendered else ""
    try:
        message = str(exc)
    except BaseException:  # noqa: BLE001 — even exception formatting is user code
        message = "<exception message could not be formatted>"
        body = f"{type(exc).__name__}: {message}\n"
    else:
        try:
            body = "".join(traceback.format_exception_only(type(exc), exc))
        except BaseException:  # noqa: BLE001 — custom exception metadata can also fail
            body = f"{type(exc).__name__}: {message}\n"
    return {
        "type": "error",
        "ename": type(exc).__name__,
        "evalue": _truncate(message, _REPR_LIMIT),
        "traceback": _truncate(header + rendered + body, _STREAM_LIMIT),
    }
