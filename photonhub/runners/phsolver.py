"""The phsolver process layer — single source of truth for invoking phsolver.

Both the local runner (``run_local``) and the cloud executor
(``photonhub.executor``) sit on this module: solver discovery (``find_solver``),
the run-command grammar (``device_args`` / ``phsolver_run_cmd``), and the
subprocess + JSON-lines event stream + stderr/timeout/exit-code contract
(``run_phsolver``). Driving the solver through here keeps those semantics
byte-identical local vs cloud; output interpretation (load a ``SimulationData``
vs package a result bundle) is the caller's job.

phsolver streams JSON-lines on stdout (NUMERICS.md section 7): ``start`` →
repeated ``progress`` → terminal ``done`` or ``error``. Exit codes are the
contract: 0 ok, 1 spec error, 2 runtime/solver error.
"""

import json
import os
import signal
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional, Union

from .._env import env

_STDERR_TAIL_CHARS = 4000
_CLOUD_CREDENTIAL_ENV = (
    "PHOTONHUB_API_KEY", "PHOTONHUB_URL",
    "PHOTONHUB_API_KEY", "PHOTONHUB_URL",
)

EventCb = Optional[Callable[[dict], None]]


def _solver_subprocess_env() -> dict:
    """Copy the process environment without cloud credentials.

    The Workbench sidecar needs the account to make cloud API calls, but native
    ``phsolver`` children never do. Keeping this at the shared process seam
    prevents API keys from spreading to local CPU/GPU solver processes or
    appearing in their crash diagnostics.
    """
    child_env = os.environ.copy()
    for name in _CLOUD_CREDENTIAL_ENV:
        child_env.pop(name, None)
    return child_env


class _WindowsJob:
    """Own a Windows subprocess tree until the root process is reaped.

    ``Popen.terminate()`` only terminates the process named by the handle on
    Windows.  A solver helper would therefore survive Workbench Stop/Exit.  A
    kill-on-close Job Object gives this process layer the Windows equivalent of
    the POSIX process group used below, without adding a runtime dependency.
    """

    def __init__(self, proc: subprocess.Popen):
        # Import lazily: these types/APIs do not exist on non-Windows hosts.
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            # JOBOBJECTINFOCLASS.JobObjectExtendedLimitInformation = 9;
            # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000.
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = 0x00002000
            if not kernel32.SetInformationJobObject(
                    handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            process_handle = wintypes.HANDLE(int(proc._handle))  # type: ignore[attr-defined]
            if not kernel32.AssignProcessToJobObject(handle, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            kernel32.CloseHandle(handle)
            raise

        self._kernel32 = kernel32
        self._handle = handle
        self._lock = threading.Lock()

    def close(self) -> None:
        """Close once; any process still in the job is terminated by Windows."""
        with self._lock:
            if self._handle is None:
                return
            handle, self._handle = self._handle, None
        self._kernel32.CloseHandle(handle)


def _taskkill_process_tree(pid: int) -> None:
    """Best-effort Windows fallback when Job Object assignment was unavailable."""
    try:
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=_solver_subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError):
        pass


class _ProcessTreeOwner:
    """Platform process-tree lifetime used by timeout, Stop, and app exit."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.windows_job = None
        if os.name == "nt":
            try:
                self.windows_job = _WindowsJob(proc)
            except OSError:
                # Some managed hosts disallow nested jobs.  ``taskkill /T`` is
                # retained as the emergency fallback for those environments.
                self.windows_job = None

    def stop(self, *, force: bool = False) -> None:
        if os.name == "posix":
            if self.proc.poll() is not None:
                return
            try:
                os.killpg(
                    self.proc.pid,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            except ProcessLookupError:
                pass
            return

        if os.name == "nt":
            if force:
                if self.windows_job is not None:
                    # Works even when the root has exited but a descendant kept
                    # one of our stdout/stderr pipe handles open.
                    self.windows_job.close()
                    return
                _taskkill_process_tree(self.proc.pid)
                if self.proc.poll() is None:
                    try:
                        self.proc.kill()
                    except OSError:
                        pass
                return
            if self.proc.poll() is not None:
                return
            # CREATE_NEW_PROCESS_GROUP lets console builds receive Ctrl-Break as
            # a graceful first request. Frozen/windowed deployments may have no
            # console; fall back to terminating the root and let the Job Object
            # force deadline below guarantee descendant cleanup.
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                try:
                    os.kill(self.proc.pid, ctrl_break)
                    return
                except OSError:
                    pass
            try:
                self.proc.terminate()
            except OSError:
                pass
            return

        if self.proc.poll() is None:
            (self.proc.kill if force else self.proc.terminate)()

    def close(self) -> None:
        if self.windows_job is not None:
            self.windows_job.close()


def _process_group_popen_kwargs() -> dict:
    """Return the native process-group flags for a solver root process."""
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)}
    return {}


class SolverRunError(RuntimeError):
    """phsolver could not be found, failed, or reported an error event."""

    def __init__(self, message: str, *, returncode: Optional[int] = None,
                 stderr_tail: Optional[str] = None):
        text = message
        if returncode is not None:
            text += f" (exit code {returncode})"
        if stderr_tail:
            text += "\n--- stderr (tail) ---\n" + stderr_tail
        super().__init__(text)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


def device_args(device: Union[str, None]) -> list:
    """Validate a device selector and map it to the phsolver ``--device`` flag,
    or ``[]`` when unset (the solver then defaults to CPU). Accepts ``"cpu"``,
    ``"gpu"``, ``"gpu:N"`` (N a local device index), ``"gpu:all"`` (every visible
    GPU), or ``"gpu:N,M,..."`` (an explicit multi-GPU set — the engine splits the
    grid along z across those devices) — the engine CLI grammar
    (engine/src/main/phsolver.cpp). Rejected here so a typo fails fast with a
    clear message rather than at the solver. Shared by the local runner and the
    cloud executor so the device grammar has one definition.

    (The cloud ``device="gpu:<target>"`` form — a curated GPU id, not an index —
    is resolved to a plain ``gpu`` on the worker by the platform; only ``cpu`` /
    ``gpu`` ever reach this on a worker.)
    """
    if device is None:
        return []
    d = device.strip()
    ok = d in ("cpu", "gpu")
    if not ok and d.startswith("gpu:"):
        tail = d[4:]
        if tail == "all":
            ok = True
        elif tail != "":
            parts = tail.split(",")
            ok = all(p.isdigit() and p != "" for p in parts)
    if not ok:
        raise SolverRunError(
            f"invalid device {device!r}: expected 'cpu', 'gpu', 'gpu:N', "
            "'gpu:all', or 'gpu:N,M,...'")
    return ["--device", d]


def _as_executable(path) -> Optional[Path]:
    p = Path(path)
    return p if p.is_file() and os.access(p, os.X_OK) else None


def _repo_build_if_current(repo_root: Path) -> Optional[Path]:
    """Return the implicit in-tree solver, optionally requiring HEAD parity.

    Test suites set ``PHOTONHUB_REQUIRE_SOURCE_MATCH=1`` so an ignored binary from
    another checkout/commit cannot create convincing integration failures. An
    explicit ``solver_path``, environment override, or PATH entry remains the
    caller's deliberate choice and is never filtered here.
    """
    solver = _as_executable(repo_root / "build" / "phsolver")
    if solver is None or os.environ.get("PHOTONHUB_REQUIRE_SOURCE_MATCH") != "1":
        return solver
    try:
        expected = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env=_solver_subprocess_env(),
        ).stdout.strip()
        info = subprocess.run(
            [str(solver), "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env=_solver_subprocess_env(),
        )
        actual = str(json.loads(info.stdout).get("git_sha", ""))[:12]
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    return solver if actual == expected else None


def find_solver(solver_path=None) -> Optional[Path]:
    """Locate the phsolver binary: explicit argument, then $PHOTONHUB_SOLVER
    (legacy $SIMUPOD_SOLVER still accepted), then PATH, then the in-repo default
    build directory. An explicit argument or environment override that does not
    exist is an error, not a fallthrough. Returns None only when nothing is
    configured and no binary is found."""
    if solver_path is not None:
        p = _as_executable(solver_path)
        if p is None:
            raise SolverRunError(
                f"solver_path is not an executable file: {solver_path}")
        return p
    override = env("SOLVER")  # $PHOTONHUB_SOLVER, legacy $SIMUPOD_SOLVER
    if override:
        p = _as_executable(override)
        if p is None:
            raise SolverRunError(
                f"$PHOTONHUB_SOLVER is not an executable file: {override}")
        return p
    on_path = shutil.which("phsolver")
    if on_path:
        return Path(on_path)
    # repo root / build / phsolver, for in-tree development checkouts
    return _repo_build_if_current(Path(__file__).resolve().parents[3])


def phsolver_run_cmd(solver, spec_path, out_dir, device=None, log_file=None) -> list:
    """Build the ``phsolver run ...`` argv shared by the local runner and the
    cloud executor — one definition of the engine CLI invocation, so a flag
    change can't silently diverge between the two paths. ``--progress none`` is
    forced (Python is the only human surface); ``device`` and ``log_file`` are
    appended when given."""
    cmd = [str(solver), "run", str(spec_path), "--output", str(out_dir),
           "--progress", "none"]
    cmd += device_args(device)
    if log_file is not None:
        cmd += ["--log-file", str(log_file)]
    return cmd


def run_phsolver(cmd: list, *, on_event: EventCb = None,
                 timeout: Optional[float] = None,
                 cancel_event: Optional[threading.Event] = None) -> dict:
    """Run a ``phsolver run ...`` command to completion.

    Streams every parsed JSON-lines event to ``on_event`` (non-JSON chatter is
    tolerated, never fatal). Enforces ``timeout`` by killing the child. When a
    caller supplies ``cancel_event``, setting it terminates the child and raises
    :class:`SolverRunError` with a cancellation message. Raises the same error
    surface on an emitted ``error`` event, timeout, or nonzero exit. Returns the
    terminal ``done`` event dict on success, or ``{}`` if none was emitted. The
    caller is responsible for interpreting the outputs the solver wrote (the
    "solver lies" guard: a clean exit with unreadable outputs is still a
    failure).
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=_solver_subprocess_env(),
        **_process_group_popen_kwargs(),
    )
    process_tree = _ProcessTreeOwner(proc)

    def stop_process(*, force: bool = False) -> None:
        process_tree.stop(force=force)

    # Drain stderr off-thread so a large stderr can't deadlock the stdout loop.
    stderr_chunks: list = []
    stderr_thread = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
    stderr_thread.start()

    timed_out = threading.Event()
    cancelled = threading.Event()
    process_done = threading.Event()
    watchdog = None
    cancel_waiter = None
    if timeout is not None:
        def _kill():
            timed_out.set()
            stop_process(force=True)
        watchdog = threading.Timer(timeout, _kill)
        watchdog.daemon = True
        watchdog.start()

    # The desktop run loop needs a real Stop action.  Keep cancellation in this
    # shared process layer so local, GUI, and future executor callers cannot
    # leave an orphaned solver behind.  The waiter is daemonized because a
    # never-set caller event should not delay normal process shutdown.
    if cancel_event is not None:
        def _cancel_when_requested():
            while not process_done.is_set():
                if not cancel_event.wait(0.05):
                    continue
                if proc.poll() is None:
                    cancelled.set()
                    stop_process()

                    def _kill_if_needed():
                        stop_process(force=True)

                    timer = threading.Timer(2.0, _kill_if_needed)
                    timer.daemon = True
                    timer.start()
                return

        cancel_waiter = threading.Thread(
            target=_cancel_when_requested, name="photonhub-cancel-waiter", daemon=True)
        cancel_waiter.start()

    error_event = None
    done_event: dict = {}
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON chatter is tolerated, never fatal
            if not isinstance(event, dict):
                continue
            if on_event is not None:
                on_event(event)
            kind = event.get("event")
            if kind == "error":
                error_event = event
            elif kind == "done":
                done_event = event
        returncode = proc.wait()
    finally:
        process_done.set()
        if watchdog is not None:
            watchdog.cancel()
        if cancel_waiter is not None:
            cancel_waiter.join(timeout=1.0)
        proc.stdout.close()
        stderr_thread.join(timeout=5.0)
        proc.stderr.close()
        # On Windows this closes the kill-on-close Job Object after the root was
        # reaped. Any accidentally surviving helper is cleaned up here too.
        process_tree.close()

    stderr_tail = ("".join(c for c in stderr_chunks if c))[-_STDERR_TAIL_CHARS:]

    if timed_out.is_set():
        raise SolverRunError(
            f"phsolver timed out after {timeout} s and was killed",
            stderr_tail=stderr_tail)
    if cancelled.is_set() or (cancel_event is not None and cancel_event.is_set()):
        raise SolverRunError("phsolver run cancelled", stderr_tail=stderr_tail)
    if error_event is not None:
        raise SolverRunError(
            f"solver reported an error: {error_event.get('reason', error_event)}",
            returncode=returncode, stderr_tail=stderr_tail)
    if returncode != 0:
        raise SolverRunError("phsolver exited with an error",
                             returncode=returncode, stderr_tail=stderr_tail)
    return done_event
