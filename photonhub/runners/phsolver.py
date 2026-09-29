"""The phsolver process layer, single source of truth for invoking phsolver.

Both the local runner (``run_local``) and the cloud executor
(``photonhub.executor``) sit on this module: solver discovery (``find_solver``),
the run-command grammar (``device_args`` / ``phsolver_run_cmd``), and the
subprocess + JSON-lines event stream + stderr/timeout/exit-code contract
(``run_phsolver``). Driving the solver through here keeps those semantics
byte-identical local vs cloud; output interpretation (load a ``RunResult``
vs package a result bundle) is the caller's job.

phsolver streams JSON-lines on stdout (NUMERICS.md section 7): ``start`` →
repeated ``progress`` → terminal ``done`` or ``error``. Exit codes are the
contract: 0 ok, 1 spec error, 2 runtime/solver error.
"""

import json
import os
import shutil
import signal
import subprocess
import threading
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from .._env import env, without_credentials
from .._compat import caller_stacklevel

_STDERR_TAIL_CHARS = 4000

EventCb = Optional[Callable[[dict], None]]


# The Workbench/packaging issuer hands a release solver its one-launch
# capability through these two variables (docs/desktop-solver-authorization.md).
# They are the solver's OWN credential — the gated binary refuses to start
# without them — so the generic scrub below must not eat them, even though
# the secret's name ends in a credential suffix.
_SOLVER_AUTHORIZATION_ENV = (
    "PHOTONHUB_SOLVER_AUTHORIZATION_FILE",
    "PHOTONHUB_SOLVER_LAUNCH_SECRET",
)


def _solver_subprocess_env() -> dict:
    """Copy the process environment without cloud credentials.

    The Workbench sidecar needs the account to make cloud API calls, but native
    ``phsolver`` children never do. Keeping this at the shared process seam
    prevents API keys from spreading to local CPU/GPU solver processes or
    appearing in their crash diagnostics.

    The solver's launch authorization is the one deliberate exception: without
    it, every SDK-spawned invocation of an installed (auth-required) solver is
    denied, which surfaced in the candidate gate as ``--capabilities``
    denials, four ``run_local`` failures, and preflight 422s, all one bug.
    """
    scrubbed = without_credentials(os.environ)
    for name in _SOLVER_AUTHORIZATION_ENV:
        if name in os.environ:
            scrubbed[name] = os.environ[name]
    return scrubbed


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
    GPU), or ``"gpu:N,M,..."`` (an explicit multi-GPU set, the engine splits the
    grid along z across those devices), the engine CLI grammar
    (engine/src/main/phsolver.cpp). Rejected here so a typo fails fast with a
    clear message rather than at the solver. Shared by the local runner and the
    cloud executor so the device grammar has one definition.

    (The cloud ``device="gpu:<target>"`` form, a curated GPU id, not an index ,
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


def _source_match_required() -> bool:
    return os.environ.get("PHOTONHUB_REQUIRE_SOURCE_MATCH") == "1"


def _repo_build_if_current(repo_root: Path) -> Optional[Path]:
    """Return the implicit in-tree solver, optionally requiring HEAD parity.

    Test suites set ``PHOTONHUB_REQUIRE_SOURCE_MATCH=1`` so an ignored binary from
    another checkout/commit cannot create convincing integration failures. An
    explicit ``solver_path``, environment override, or PATH entry remains the
    caller's deliberate choice and is never filtered here. A solver recorded by
    ``photonhub install-solver`` is filtered the same way (see ``find_solver``).
    """
    solver = _as_executable(repo_root / "build" / "phsolver")
    if solver is None or not _source_match_required():
        return solver
    return solver if _built_from_checkout(solver, repo_root) else None


def _built_from_checkout(solver: Path, repo_root: Path) -> bool:
    """Whether ``solver info`` reports the revision checked out at ``repo_root``.

    The reported ``git_sha`` must equal ``HEAD`` exactly, in either form a
    build stamps: the 12 characters cmake records from the checkout, or the
    full SHA a ``PHCORE_GIT_SHA_OVERRIDE`` build records (release
    candidates, container images, GPU boxes). Anything else never matches, a
    binary configured from a dirty tree (``<sha>-dirty``) included: its
    sources are not the commit's, and the edits it was built from may since
    have been reverted.

    False when either side cannot be read: no git, not a checkout, or a binary
    that does not answer ``info`` with a ``git_sha``.
    """
    heads = _checkout_git_shas(repo_root)
    reported = _solver_git_sha(solver)
    return bool(heads) and reported in heads


def _checkout_git_shas(repo_root: Path) -> Tuple[str, ...]:
    """``HEAD`` at ``repo_root`` in both forms an engine build stamps into its
    ``info``: ``git rev-parse --short=12`` first, then the full SHA. Empty
    outside a checkout."""
    shas = []
    for args in (["--short=12", "HEAD"], ["HEAD"]):
        try:
            sha = subprocess.run(
                ["git", "rev-parse", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
                env=_solver_subprocess_env(),
            ).stdout.strip()
        except (OSError, ValueError, subprocess.SubprocessError):
            return ()
        if not sha:
            return ()
        shas.append(sha)
    return tuple(shas)


def _solver_git_sha(solver: Path) -> Optional[str]:
    """The ``git_sha`` a solver binary reports through ``info``, verbatim (a
    ``-dirty`` suffix kept), or None when it does not answer with one."""
    try:
        info = subprocess.run(
            [str(solver), "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env=_solver_subprocess_env(),
        )
        sha = json.loads(info.stdout).get("git_sha")
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError,
            json.JSONDecodeError):
        return None
    return str(sha) if sha else None


def source_match_status(repo_root) -> dict:
    """Which solver :func:`find_solver` resolves, and whether it is the build of
    the revision checked out at ``repo_root``.

    Returns ``{"solver", "git_sha", "checkout", "matched", "note"}``:
    ``solver`` is the resolved path or None, ``git_sha`` what it reports,
    ``checkout`` the ``HEAD`` it is compared with (short form), and
    ``matched`` whether the solver reports ``HEAD`` exactly, short or full.
    When nothing resolves, ``note`` says why, naming an in-repository build
    that the source-match guard set aside. The test suites print this in their
    header and, when a caller asks for the guard, fail the session on it
    instead of letting every solver-backed test skip.
    """
    repo_root = Path(repo_root)
    heads = _checkout_git_shas(repo_root)
    checkout = heads[0] if heads else None
    status = {"solver": None, "git_sha": None, "checkout": checkout,
              "matched": False, "note": None}
    try:
        solver = find_solver()
    except SolverRunError as exc:
        status["note"] = str(exc)
        return status
    if solver is None:
        build = _as_executable(repo_root / "build" / "phsolver")
        if build is not None:
            status["note"] = (
                f"{build} reports git_sha {_solver_git_sha(build)!r}, not the "
                f"checkout's {checkout!r}; rebuild it (reconfigure first: the "
                "revision is stamped when cmake configures)")
        else:
            status["note"] = "no phsolver binary found (build the engine first)"
        return status
    status["solver"] = solver
    status["git_sha"] = _solver_git_sha(solver)
    status["matched"] = bool(heads) and status["git_sha"] in heads
    return status


def source_match_summary(status: dict) -> str:
    """One line for a :func:`source_match_status` result: the solver path and
    the revision it reports against the checkout's, or why none resolved."""
    if status["solver"] is None:
        return f"none ({status['note']})"
    verdict = "source-matched" if status["matched"] else "NOT source-matched"
    return (f"{status['solver']} git_sha {status['git_sha']} "
            f"(checkout {status['checkout']}: {verdict})")


# --- the test suites' session gate (photonhub/tests and validation conftests)

_GATE_CALLER = "_PHOTONHUB_SOURCE_MATCH_CALLER"


def record_source_match_caller() -> None:
    """Record whether the caller set ``PHOTONHUB_REQUIRE_SOURCE_MATCH``, before
    a test conftest defaults it to ``1`` for discovery filtering. Kept in the
    environment so both conftests of one run read one answer, and scoped to
    this process: a record inherited from another process (a parent pytest)
    is replaced by this process's own reading of the variable."""
    mine = f"{os.getpid()}:"
    if not os.environ.get(_GATE_CALLER, "").startswith(mine):
        os.environ[_GATE_CALLER] = mine + os.environ.get(
            "PHOTONHUB_REQUIRE_SOURCE_MATCH", "")


def source_match_gate(repo_root) -> Optional[str]:
    """Why a test session must stop, or None. Every solver-backed test skips
    when no source-matched solver is found, and a skip exits 0, so a gate run
    against a stale build would report green with no engine test executed.
    The session stops instead when the caller set
    ``PHOTONHUB_REQUIRE_SOURCE_MATCH=1`` (the conftests' own default only
    filters discovery) and the solver the run would use, however it was
    selected, is not a build of the checkout. ``PHOTONHUB_ALLOW_NO_SOLVER=1``
    runs the session anyway."""
    if os.environ.get(_GATE_CALLER) != f"{os.getpid()}:1":
        return None
    if os.environ.get("PHOTONHUB_ALLOW_NO_SOLVER") == "1":
        return None
    status = source_match_status(repo_root)
    if status["matched"]:
        return None
    return ("PHOTONHUB_REQUIRE_SOURCE_MATCH=1 and the solver is not a build of "
            f"this checkout: {source_match_summary(status)}. Every solver-backed "
            "test would skip. Rebuild build/phsolver from a clean tree, or set "
            "PHOTONHUB_ALLOW_NO_SOLVER=1 to run without one.")


def find_solver(solver_path=None) -> Optional[Path]:
    """Locate the phsolver binary.

    Resolution order is an explicit argument, ``$PHOTONHUB_SOLVER``, a solver
    recorded by ``photonhub install-solver`` or ``photonhub link-solver``,
    ``PATH``, then the in-repository build directory. An explicit argument or
    environment override that does not exist is an error, not a fallthrough.
    Returns ``None`` only when nothing is configured and no binary is found.

    A recorded install beats ``PATH`` because recording one is a deliberate
    act, and someone who has run the installer should not be silently served a
    different binary that happens to be earlier on the path. The environment
    variable still beats both, so a one-off override needs no uninstall.

    With ``PHOTONHUB_REQUIRE_SOURCE_MATCH=1`` a recorded install is used only
    when its ``info`` reports the revision this source tree has checked out,
    the same test the in-repository build gets. The record lives in the
    user's cache, so to a test run it is discovered, not chosen, and a released
    binary there must not stand in for the build under test. Skipped, the
    lookup carries on to ``PATH`` and the in-repository build.
    """
    if solver_path is not None:
        p = _as_executable(solver_path)
        if p is None:
            raise SolverRunError(
                f"solver_path is not an executable file: {solver_path}")
        return p
    override = env("SOLVER")
    if override:
        p = _as_executable(override)
        if p is None:
            raise SolverRunError(
                f"$PHOTONHUB_SOLVER is not an executable file: {override}")
        return p
    from ..solver_install import installed_solver

    repo_root = Path(__file__).resolve().parents[3]
    recorded = installed_solver()
    if recorded is not None and (
            not _source_match_required()
            or _built_from_checkout(recorded, repo_root)):
        return recorded
    on_path = shutil.which("phsolver")
    if on_path:
        return Path(on_path)
    # repo root / build / phsolver, for in-tree development checkouts
    return _repo_build_if_current(repo_root)


def phsolver_run_cmd(solver, spec_path, out_dir, device=None, log_file=None) -> list:
    """Build the ``phsolver run ...`` argv shared by the local runner and the
    cloud executor, one definition of the engine CLI invocation, so a flag
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

    stderr_text = "".join(c for c in stderr_chunks if c)
    stderr_tail = stderr_text[-_STDERR_TAIL_CHARS:]

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
    for line in stderr_text.splitlines():
        if line.startswith("PHOTONHUB_DFT_GUARD_IGNORED: "):
            warnings.warn(line.split(": ", 1)[1], UserWarning, stacklevel=caller_stacklevel())
    return done_event
