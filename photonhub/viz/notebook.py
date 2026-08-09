"""Workbench notebook kernel — scriptable setup, run, and analysis.

One persistent Python namespace executes user cells sequentially on a dedicated
worker thread.  The ``wb`` bridge object connects scripts to the same
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
import ctypes
import io
import json
import os
import tempfile
import threading
import time
import traceback
import uuid
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Optional

_NOTEBOOK_DOC_VERSION = 1
_STREAM_LIMIT = 200_000       # chars kept per stream output
_REPR_LIMIT = 20_000          # chars kept for a result repr

_WELCOME_CODE = """\
# PhotonHub Workbench notebook — the live workspace is available as `wb`.
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


def _figure_output(value: Any) -> Optional[dict]:
    """Convert a user object into a figure/image output dict, or ``None``."""
    candidate = value
    to_plotly = getattr(candidate, "to_plotly_json", None)
    if callable(to_plotly):
        try:
            candidate = to_plotly()
        except Exception:
            return None
    if isinstance(candidate, dict) and isinstance(candidate.get("data"), list):
        try:  # keep the wire JSON-clean (numpy arrays -> lists)
            cleaned = json.loads(json.dumps(candidate, default=_json_fallback))
        except (TypeError, ValueError):
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
            return None
        return {"type": "image", "mime": "image/png",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii")}
    return None


def _json_fallback(value: Any):
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


def _capture_pyplot_figures() -> list[dict]:
    """Collect and close any figures pyplot accumulated during the cell."""
    import sys
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    outputs = []
    try:
        for num in list(plt.get_fignums()):
            output = _figure_output(plt.figure(num))
            if output is not None:
                outputs.append(output)
        plt.close("all")
    except Exception:
        return outputs
    return outputs


class NotebookKernel:
    """Sequential cell executor with a persistent namespace.

    Thread model: HTTP threads mutate the document and queue under ``_lock``;
    one daemon worker executes cells.  ``restart`` abandons a stuck worker
    (threads cannot be killed) — the generation counter keeps an abandoned
    worker from publishing into the new session.
    """

    def __init__(self, hooks: WorkbenchHooks,
                 persist_path: Optional[Path] = None):
        # A GUI-less matplotlib backend: cells run on server threads where a
        # macOS/Windows native-window backend would be unusable or unsafe.
        os.environ.setdefault("MPLBACKEND", "Agg")
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            # `plt.show()` ends nearly every plotting cell people paste in
            # (every notebook in examples/notebooks does). On Agg it warns
            # "FigureCanvasAgg is non-interactive, and thus cannot be shown"
            # — which is both noise and a lie here: _capture_pyplot_figures
            # collects the figure and the panel renders it inline. Silence
            # only that message, so genuine matplotlib warnings still show.
            warnings.filterwarnings(
                "ignore", message="FigureCanvasAgg is non-interactive",
                category=UserWarning)
        except Exception:
            pass
        self._hooks = hooks
        self._persist_path = Path(persist_path) if persist_path else None
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._cells: list[dict] = []
        self._queue: list[str] = []
        self._execution_count = 0
        self._workspace_rev = 0
        self._generation = 0
        self._running_cell: Optional[str] = None
        self._live: Optional[dict] = None   # live capture for the running cell
        self._namespace = self._fresh_namespace()
        if not self._restore():
            self._cells = [self._new_cell(_WELCOME_CODE),
                           self._new_cell(_TOUR_APPLY_CODE),
                           self._new_cell(_TOUR_RUN_CODE)]
        self._worker = self._spawn_worker()

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
            worker = self._worker
            running = self._running_cell is not None
        if running and worker.is_alive():
            _async_raise(worker.ident, KeyboardInterrupt)
        return self.snapshot()

    def restart(self, *, clear_outputs: bool = False) -> dict:
        with self._lock:
            self._generation += 1
            for cell_id in self._queue:
                try:
                    self._cell(cell_id)["status"] = "idle"
                except LookupError:
                    pass
            self._queue.clear()
            abandoned = self._worker if self._running_cell is not None else None
            if self._running_cell is not None:
                cell = self._cell(self._running_cell)
                cell["status"] = "error"
                cell["outputs"] = [{
                    "type": "error", "ename": "KernelRestart",
                    "evalue": "The kernel restarted while this cell was running.",
                    "traceback": "The kernel restarted while this cell was running.",
                }]
            self._running_cell = None
            self._live = None
            self._namespace = self._fresh_namespace()
            if clear_outputs:
                for cell in self._cells:
                    cell.update(status="idle", execution_count=None,
                                outputs=[], elapsed_s=None)
            self._worker = self._spawn_worker()
            self._persist()
            snap = self.snapshot()
        if abandoned is not None and abandoned.is_alive():
            # Best effort: a compute-bound abandoned cell honors the async
            # KeyboardInterrupt at its next bytecode boundary.
            _async_raise(abandoned.ident, KeyboardInterrupt)
        return snap

    def bump_workspace_rev(self) -> None:
        with self._lock:
            self._workspace_rev += 1

    def emit_output(self, output: dict) -> None:
        with self._lock:
            live = self._live
            if live is not None:
                live["outputs"].append(output)

    def print_progress_line(self, text: str) -> None:
        """Rate-limited progress line for long ``wb.run`` calls."""
        now = time.monotonic()
        live = self._live
        if live is None:
            return
        if text == live.get("last_progress") and now - live.get("last_progress_at", 0.0) < 5.0:
            return
        if now - live.get("last_progress_at", 0.0) < 1.0:
            return
        live["last_progress"] = text
        live["last_progress_at"] = now
        print(text)

    # ---------------------------------------------------------------- worker
    def _fresh_namespace(self) -> dict:
        namespace: dict[str, Any] = {"__name__": "__workbench__", "wb": Workbench(self)}
        try:
            import numpy
            namespace["np"] = numpy
        except Exception:
            pass
        return namespace

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
                    cell["status"] = "running"
                    cell["execution_count"] = count
                    cell["outputs"] = []
                    cell["elapsed_s"] = None
                    self._running_cell = cell_id
                    self._live = {"cell_id": cell_id, "outputs": [],
                                  "stdout": io.StringIO(), "stderr": io.StringIO(),
                                  "last_progress": None, "last_progress_at": 0.0}
                    code = cell["code"]
                    namespace = self._namespace
                    live = self._live
                try:
                    outputs, error, elapsed = self._execute(
                        code, count, namespace, live)
                except KeyboardInterrupt:
                    outputs, error, elapsed = [{
                        "type": "error", "ename": "KeyboardInterrupt",
                        "evalue": "interrupted",
                        "traceback": "KeyboardInterrupt",
                    }], True, 0.0
                with self._lock:
                    if generation != self._generation:
                        return
                    self._running_cell = None
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

    def _execute(self, code: str, count: int, namespace: dict,
                 live: dict) -> tuple[list, bool, float]:
        outputs: list[dict] = []
        error = False
        started = time.monotonic()
        stdout, stderr = live["stdout"], live["stderr"]
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    tree = ast.parse(code, filename=f"<cell {count}>")
                    trailing = None
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        trailing = ast.Expression(tree.body[-1].value)
                        tree.body = tree.body[:-1]
                    if tree.body:
                        exec(compile(tree, f"<cell {count}>", "exec"), namespace)  # noqa: S102 — the notebook exists to run user code locally
                    value = None
                    if trailing is not None:
                        value = eval(compile(trailing, f"<cell {count}>", "eval"),  # noqa: S307
                                     namespace)
                    if value is not None:
                        namespace["_"] = value
                        figure = _figure_output(value)
                        if figure is not None:
                            live["outputs"].append(figure)
                        else:
                            live["outputs"].append({
                                "type": "result",
                                "text": _truncate(repr(value), _REPR_LIMIT)})
                except BaseException as exc:  # noqa: BLE001 — every user error becomes a cell error output
                    error = True
                    live["outputs"].append(_error_output(exc, count))
                live["outputs"].extend(_capture_pyplot_figures())
        finally:
            elapsed = time.monotonic() - started
        for name, buffer in (("stdout", stdout), ("stderr", stderr)):
            text = buffer.getvalue()
            if text:
                outputs.append({"type": "stream", "name": name,
                                "text": _truncate(text, _STREAM_LIMIT)})
        outputs.extend(live["outputs"])
        # streams first, then rich outputs/errors — stable, readable order
        return outputs, error, elapsed

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
    body = "".join(traceback.format_exception_only(type(exc), exc))
    return {
        "type": "error",
        "ename": type(exc).__name__,
        "evalue": _truncate(str(exc), _REPR_LIMIT),
        "traceback": _truncate(header + rendered + body, _STREAM_LIMIT),
    }


def _async_raise(thread_ident: Optional[int], exc_type: type) -> None:
    """Raise ``exc_type`` asynchronously in another thread (CPython only).

    This is the standard interpreter facility behind notebook interrupts; it
    lands at the next bytecode boundary, so C extensions finish their current
    call first.
    """
    if thread_ident is None:
        return
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_ident), ctypes.py_object(exc_type))
