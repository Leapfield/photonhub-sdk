"""Default human-readable renderer for phsolver progress events.

``run_local`` streams JSON-lines events (NUMERICS.md section 7) from phsolver;
this module turns them into a one-line, TTY-aware status display on stderr. The
line format mirrors the engine CLI's renderer
(``engine/include/phcore/status_line.h``) — keep the two in sync.

``run_local`` passes ``--progress none`` to the child so phsolver stays silent
on its own stderr and Python owns the human surface (it knows the real terminal
state and whether the caller supplied a custom callback).
"""

import sys
from typing import Optional, TextIO


def _sci(v) -> str:
    try:
        return f"{float(v):.1e}"
    except (TypeError, ValueError):
        return "?"


def _dur(s) -> str:
    if s is None:
        return "?"
    s = float(s)
    if s < 0:
        return "?"
    if s >= 3600:
        h = int(s // 3600)
        m = int((s - h * 3600) // 60)
        return f"{h}h{m:02d}m"
    if s >= 60:
        m = int(s // 60)
        sec = int(s - m * 60)
        return f"{m}m{sec:02d}s"
    if s >= 10:
        return f"{s:.0f}s"
    return f"{s:.1f}s"


def _gib(b) -> str:
    return f"{float(b) / (1 << 30):.2f}"


def _bar(pct: int, width: int = 10) -> str:
    pct = max(0, min(100, pct))
    filled = pct * width // 100
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def format_progress(e: dict) -> str:
    """Build the one-line status body for a ``progress`` event (no control codes)."""
    step = int(e.get("step", 0))
    total = int(e.get("total", 0))
    pct = int(100 * step / total) if total else 0
    shutoff = e.get("shutoff", 0.0) or 0.0

    parts = [f"{pct:3d}% {_bar(pct)}", f"step {step}/{total}"]
    parts.append(f"t={float(e.get('t_s', 0.0)) * 1e12:.1f}ps")
    parts.append("[injecting]" if e.get("phase") == "injecting" else "[ringdown]")

    decay = f"decay {_sci(e.get('field_decay', 1.0))}"
    if shutoff > 0:
        decay += f" -> {_sci(shutoff)}"
    parts.append(decay)

    parts.append("stable" if e.get("stable", True) else "UNSTABLE")

    tput = f"{float(e.get('mcells_per_s', 0.0)):.0f} Mcells/s"
    avg = e.get("mcells_per_s_avg")
    if avg:
        tput += f" (avg {float(avg):.0f})"
    parts.append(tput)

    eta = e.get("eta_s")
    if eta is not None and float(eta) >= 0:
        parts.append(f"ETA {'<=' if shutoff > 0 else ''}{_dur(eta)}")

    used = e.get("mem_used_bytes")
    tot = e.get("mem_total_bytes")
    if used is not None and tot is not None:
        parts.append(f"mem {_gib(used)}/{_gib(tot)} GiB")

    return "  ".join(parts)


class TerminalRenderer:
    """Stateful renderer: feed it every event dict and it draws to ``stream``.

    On a TTY it redraws a single status line in place (carriage return + clear);
    otherwise it prints one plain line per progress event. The terminal
    ``done``/``error`` events finish the display with a summary line.
    """

    def __init__(self, stream: Optional[TextIO] = None):
        self.stream = stream if stream is not None else sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started = False
        self.last: dict = {}

    def __call__(self, event: dict) -> None:
        kind = event.get("event")
        if kind == "start":
            self._start(event)
        elif kind == "progress":
            self._progress(event)
        elif kind == "done":
            self._done(event)
        elif kind == "error":
            self._error(event)

    def _w(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _clear(self) -> None:
        if self.tty and self.started:
            self._w("\r\033[K")

    def _start(self, e: dict) -> None:
        head = "ph"
        dev = e.get("device")
        if dev:
            head += f" · {dev}"
            arch = e.get("arch")
            if arch:
                head += f" {arch}"
        grid = e.get("grid") or []
        if len(grid) == 3:
            mcells = grid[0] * grid[1] * grid[2] / 1e6
            head += f" · {grid[0]}x{grid[1]}x{grid[2]} ({mcells:.1f} Mcells)"
        self._w(head + "\n")

    def _progress(self, e: dict) -> None:
        self.last = e
        line = format_progress(e)
        self._w("\r\033[K" + line if self.tty else line + "\n")
        self.started = True

    def _done(self, e: dict) -> None:
        self._clear()
        wall = _dur(e.get("wall_s"))
        tput = float(e.get("mcells_per_s", 0.0))
        if e.get("shut_off"):
            total = self.last.get("total")
            tail = f"/{total}" if total else ""
            decay = _sci(e.get("field_decay", self.last.get("field_decay", 1.0)))
            self._w(
                f"ph · done in {wall} · shutoff at step "
                f"{e.get('steps_run')}{tail} (decay {decay})  "
                f"avg {tput:.0f} Mcells/s\n"
            )
        else:
            steps = e.get("steps_run")
            tail = f" · {steps} steps" if steps else ""
            self._w(f"ph · done in {wall}{tail}  avg {tput:.0f} Mcells/s\n")

    def _error(self, e: dict) -> None:
        self._clear()
        self._w(f"ph · ERROR: {e.get('reason', 'unknown')}\n")


def default_renderer(stream: Optional[TextIO] = None) -> TerminalRenderer:
    """Return a progress callback (callable on each event dict) that renders to
    ``stream`` — ``sys.stderr`` by default."""
    return TerminalRenderer(stream)
