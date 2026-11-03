"""Credential-free interpreter mode for Workbench notebook cells."""

from __future__ import annotations

import ast
import io
import json
import os
import queue
import signal
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout

from .notebook_channel import ChannelError, MAX_FRAME_BYTES, read_frame, write_frame
from .notebook import (Workbench, _STREAM_LIMIT, _REPR_LIMIT, _LIVE_OUTPUT_LIMIT, _parse_cell,
                       _figure_output, _release_pyplot_figure, _truncate,
                       _capture_pyplot_figures, _error_output)


class _NoCellInput:
    def isatty(self):
        return False

    def read(self, *_args, **_kwargs):
        raise EOFError("Workbench notebook cells have no interactive stdin")

    def readline(self, *_args, **_kwargs):
        raise EOFError("Workbench notebook cells have no interactive stdin")

    def __iter__(self):
        return self

    def __next__(self):
        return self.readline()

    def fileno(self):
        raise io.UnsupportedOperation("fileno")


def _private_channel():
    """Keep the wire away from cell and descendant standard descriptors."""
    read_fd, write_fd = os.dup(0), os.dup(1)
    os.set_inheritable(read_fd, False)
    os.set_inheritable(write_fd, False)
    incoming = os.fdopen(read_fd, "rb", buffering=0)
    outgoing = os.fdopen(write_fd, "wb", buffering=0)
    with open(os.devnull, "r+b", buffering=0) as null:
        for fd in (0, 1, 2):
            os.dup2(null.fileno(), fd)
    sys.stdin = _NoCellInput()
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
    return incoming, outgoing


def worker_main() -> int:
    incoming, outgoing = _private_channel()
    executing = False
    outgoing_lock = threading.Lock()
    execution_thread = threading.get_ident()
    sending = False
    pending_interrupt = False

    def send_frame(message):
        nonlocal sending, pending_interrupt
        on_execution_thread = threading.get_ident() == execution_thread
        if on_execution_thread:
            sending = True
        try:
            with outgoing_lock:
                write_frame(outgoing, message)
        finally:
            if on_execution_thread:
                sending = False
                # Deliver even if serialization rejects the message, so a
                # deferred interrupt cannot escape during cell completion.
                if pending_interrupt:
                    pending_interrupt = False
                    raise KeyboardInterrupt

    def interrupt(_signum, _frame):
        nonlocal pending_interrupt
        if executing:
            if sending:
                pending_interrupt = True
                return
            pending_interrupt = False
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, interrupt)
    _watch_parent_process()
    try:
        import matplotlib
        import warnings
        matplotlib.use("Agg", force=True)
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive",
                                category=UserWarning)
    except ImportError:
        pass
    from ..cloud import config as cloud_config
    cloud_config.set_notebook_worker_mode()

    # SIGINT can abandon a blocking RPC at any byte of a frame. Keep the
    # partial header/body on a thread that cannot receive Python signals;
    # the execution thread waits only for complete messages. Bounding the
    # queue also bounds read-ahead when the cell is busy or interrupted.
    incoming_frames = queue.Queue(maxsize=1)
    def read_frames():
        while True:
            try:
                message = read_frame(incoming)
            except (ChannelError, OSError) as exc:
                incoming_frames.put(ChannelError(str(exc)))
                return
            incoming_frames.put(message)
    threading.Thread(target=read_frames, name="notebook-channel-reader", daemon=True).start()

    def receive_frame():
        while True:
            try:
                # A finite wait also lets Windows dispatch its signal handler.
                message = incoming_frames.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        if isinstance(message, ChannelError):
            raise message
        return message

    class ProxyHooks:
        next_id = 0
        def call(self, name, args=None, progress=None):
            self.next_id += 1
            call_id = self.next_id
            send_frame({"type": "call", "id": call_id,
                        "name": name, "args": args or {}})
            while True:
                response = receive_frame()
                if (response.get("type") in {"progress", "reply"}
                        and response.get("id") != call_id):
                    continue  # a reply to a call interrupted by the cell
                if response.get("type") == "progress" and progress is not None:
                    progress(response.get("value"), response.get("status"))
                    continue
                if response.get("type") != "reply":
                    raise ChannelError("invalid parent reply")
                if response.get("ok") is True:
                    return response.get("value")
                kind = response.get("error_type")
                message = response.get("message", "Workbench operation failed")
                if kind == "LookupError":
                    raise LookupError(message)
                if kind == "ValueError":
                    raise ValueError(message)
                if kind == "TypeError":
                    raise TypeError(message)
                if kind == "KeyboardInterrupt":
                    raise KeyboardInterrupt(message)
                raise RuntimeError(message)

        def get_spec(self):
            return self.call("get_spec")

        def apply_spec(self, spec):
            return self.call("apply_spec", {"spec": spec})

        def run_local(self, *, spec, device, timeout_s, on_progress):
            descriptor = self.call("run_local", {
                "spec": spec, "device": device, "timeout_s": timeout_s,
            }, on_progress)
            return _load_descriptor(descriptor)

        def cancel_active_run(self):
            self.call("cancel_run")

        def get_result(self):
            descriptor = self.call("get_result")
            return _load_descriptor(descriptor) if descriptor is not None else None

    class WorkerContext:
        _hooks = ProxyHooks()
        def emit_output(self, output):
            if len(self._live["outputs"]) >= 997:
                self._live["outputs_truncated"] = True
                return
            if not _fits_output(output):
                raise ValueError("notebook figure is too large to display")
            size = len(json.dumps(output, ensure_ascii=False).encode("utf-8"))
            if self._live["output_bytes"] + size > _LIVE_OUTPUT_LIMIT:
                self._live["outputs_truncated"] = True
                return
            self._live["output_bytes"] += size
            self._live["outputs"].append(output)
            send_frame({"type": "output", "output": output})
        def bump_workspace_rev(self):
            pass
        def print_progress_line(self, value):
            now = time.monotonic()
            live = self._live
            if (live["last_progress"] == value
                    and now - live["last_progress_at"] < 5):
                return
            if now - live["last_progress_at"] < 1:
                return
            live["last_progress"] = value
            live["last_progress_at"] = now
            print(value)

    kernel = WorkerContext()
    namespace = {"__name__": "__workbench__", "wb": Workbench(kernel)}
    try:
        import numpy
        namespace["np"] = numpy
    except ImportError:
        pass

    class LiveStringIO(io.StringIO):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.pending = []
            self.pending_size = 0
            self.last_sent = time.monotonic()
            self.total_chars = 0
            self.pending_lock = threading.Lock()
            self.timer = None
        def write(self, value):
            self.total_chars += len(value)
            retained = value[:max(0, _STREAM_LIMIT - self.tell())]
            super().write(retained)
            if retained:
                with self.pending_lock:
                    self.pending.append(retained)
                    self.pending_size += len(retained)
                    if self.timer is None:
                        self.timer = threading.Timer(0.05, self.flush_live)
                        self.timer.daemon = True
                        self.timer.start()
            now = time.monotonic()
            if self.pending_size >= 8192 or now - self.last_sent >= 0.1:
                self.flush_live()
            return len(value)
        def flush_live(self):
            with self.pending_lock:
                if self.timer is not None:
                    self.timer.cancel()
                    self.timer = None
                if not self.pending:
                    return
                data = "".join(self.pending)
                self.pending.clear()
                self.pending_size = 0
                self.last_sent = time.monotonic()
                for offset in range(0, len(data), 8192):
                    send_frame({"type": "live", "name": self.name,
                                "text": data[offset:offset + 8192]})

    send_frame({"type": "ready"})
    while True:
        try:
            message = receive_frame()
        except ChannelError:
            return 0
        if message.get("type") in {"progress", "reply"}:
            continue  # the parent finished an interrupted local hook
        if (set(message) != {"type", "code", "count"}
                or message["type"] != "execute"
                or not isinstance(message["code"], str)
                or type(message["count"]) is not int
                or not 0 < message["count"] < 1_000_000_000):
            return 2
        count = message["count"]
        if len(message["code"]) > 1_000_000:
            send_frame({"type": "result", "count": count,
                "outputs": [_limit_error("notebook cell source is too large")],
                "error": True, "elapsed_s": 0.0})
            continue
        live = {"outputs": [], "stdout": LiveStringIO("stdout"),
                "stderr": LiveStringIO("stderr"),
                "last_progress": None, "last_progress_at": 0.0,
                "outputs_truncated": False, "output_bytes": 0}
        kernel._live = live
        try:
            executing = True
            def code_done():
                nonlocal executing
                executing = False
                send_frame({"type": "code_done", "count": count})
            outputs, error, elapsed = execute_cell(
                message["code"], count, namespace, live, on_code_done=code_done)
            live["stdout"].flush_live()
            live["stderr"].flush_live()
            for output in outputs:
                if output.get("type") == "stream" and live[output["name"]].total_chars > _STREAM_LIMIT:
                    output["text"] += (f"\n… [truncated {live[output['name']].total_chars - _STREAM_LIMIT} characters]")
            if any(not _fits_output(item) for item in outputs):
                outputs = [item for item in outputs if _fits_output(item)]
                outputs.append(_limit_error("notebook output or figure is too large to display"))
                error = True
            retained = []
            retained_bytes = 0
            truncated = live["outputs_truncated"]
            for item in outputs:
                size = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
                if len(retained) >= 999 or retained_bytes + size > _LIVE_OUTPUT_LIMIT:
                    truncated = True
                    continue
                retained.append(item)
                retained_bytes += size
            if truncated:
                retained.append({"type": "stream", "name": "stderr",
                                 "text": "… [truncated notebook outputs]"})
            outputs = retained
            result = {"type": "result", "count": count, "outputs": outputs,
                      "error": error, "elapsed_s": elapsed}
            try:
                send_frame(result)
            except ChannelError:
                send_frame({"type": "result", "count": count,
                    "outputs": [_limit_error("notebook output or figure is too large to display")],
                    "error": True, "elapsed_s": elapsed})
        except (ChannelError, BrokenPipeError, OSError):
            return 2
        finally:
            executing = False
            kernel._live = None


def _limit_error(message):
    return {"type": "error", "ename": "NotebookOutputLimit",
            "evalue": message, "traceback": message}


def _fits_output(output):
    from .notebook import NotebookKernel
    if not NotebookKernel._valid_output(output):
        return False
    try:
        import json
        return len(json.dumps(output, ensure_ascii=False).encode("utf-8")) < MAX_FRAME_BYTES - 1024
    except (TypeError, ValueError, OverflowError):
        return False


def _load_descriptor(descriptor):
    if (not isinstance(descriptor, dict)
            or set(descriptor) != {"output_dir", "session"}
            or not isinstance(descriptor["output_dir"], str)
            or not isinstance(descriptor["session"], dict)):
        raise ChannelError("invalid result descriptor")
    from . import service
    return service.load_result(descriptor["output_dir"]), descriptor["session"]


def _watch_parent_process() -> None:
    """Exit after an abrupt sidecar death, including while a cell is running."""
    parent_pid = os.getppid()
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x00100000, False, parent_pid)
        if not handle:
            os._exit(0)
        def watch():
            try:
                if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) == 0:
                    os._exit(0)
            finally:
                kernel.CloseHandle(handle)
    else:
        def watch():
            while True:
                time.sleep(0.25)
                if os.getppid() != parent_pid:
                    os._exit(0)
    threading.Thread(target=watch, name="notebook-parent-watch", daemon=True).start()


def execute_cell(code: str, count: int, namespace: dict,
             live: dict, on_code_done=None) -> tuple[list, bool, float]:
    outputs: list[dict] = []
    error = False
    started = time.monotonic()
    stdout, stderr = live["stdout"], live["stderr"]
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                tree, ignored_magics = _parse_cell(
                    code, filename=f"<cell {count}>")
                for magic in ignored_magics:
                    print(f"ignored IPython magic: {magic}",
                          file=sys.stderr)
                trailing = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    trailing = ast.Expression(tree.body[-1].value)
                    tree.body = tree.body[:-1]
                if tree.body:
                    exec(compile(tree, f"<cell {count}>", "exec"), namespace)  # noqa: S102 — the notebook exists to run user code locally
                value = None
                figure = None
                display_text = None
                render_figure = False
                if trailing is not None:
                    value = eval(compile(trailing, f"<cell {count}>", "eval"),  # noqa: S307
                                 namespace)
                if value is not None:
                    namespace["_"] = value
                    render_figure = callable(getattr(value, "savefig", None))
                    if render_figure:
                        display_text = _truncate(repr(value), _REPR_LIMIT)
                    else:
                        figure = _figure_output(value, raise_errors=True)
                        if figure is None:
                            display_text = _truncate(repr(value), _REPR_LIMIT)
            except BaseException as exc:  # noqa: BLE001 — every user error becomes a cell error output
                error = True
                live["outputs"].append(_error_output(exc, count))
            finally:
                if on_code_done is not None:
                    on_code_done()
            if not error and value is not None:
                if render_figure:
                    try:
                        figure = _figure_output(value, raise_errors=True)
                    except BaseException as exc:  # noqa: BLE001 — display errors belong to the cell
                        error = True
                        live["outputs"].append(_error_output(exc, count))
                if not error and figure is not None:
                    live["outputs"].append(figure)
                    _release_pyplot_figure(value)
                elif not error:
                    live["outputs"].append({
                        "type": "result",
                        "text": display_text if display_text is not None else ""})
            remaining = max(0, 997 - len(live["outputs"]))
            figures = _capture_pyplot_figures(limit=remaining + 1)
            if len(figures) > remaining:
                live["outputs_truncated"] = True
            live["outputs"].extend(figures[:remaining])
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
