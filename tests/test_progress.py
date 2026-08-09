"""Default progress renderer: the line formatter and the TTY-aware
TerminalRenderer (driven against a StringIO, i.e. non-tty), plus run_local
passing ``--progress none`` to the child so Python owns the human surface."""

import io
import stat
import sys
import textwrap

import pytest

import photonhub as ph
from photonhub.runners.progress import (
    TerminalRenderer,
    default_renderer,
    format_progress,
)


def _progress_event(**over):
    e = {
        "event": "progress", "step": 300, "total": 1000, "t_s": 1.16e-12,
        "field_energy": 4.2e-3, "mcells_per_s": 423.0, "mcells_per_s_avg": 410.0,
        "field_decay": 4.2e-3, "shutoff": 1e-5, "phase": "ringdown",
        "stable": True, "eta_s": 31.0,
        "mem_used_bytes": 1150 * (1 << 20), "mem_total_bytes": 192 * (1 << 30),
    }
    e.update(over)
    return e


class TestFormat:
    def test_full_line_has_every_field(self):
        s = format_progress(_progress_event())
        assert "step 300/1000" in s
        assert "[ringdown]" in s
        assert "decay 4.2e-03" in s
        assert "-> 1.0e-05" in s          # threshold shown when shutoff>0
        assert "stable" in s and "UNSTABLE" not in s
        assert "Mcells/s" in s
        assert "ETA <=31s" in s           # "<=" ceiling: shutoff may end sooner
        assert "mem " in s

    def test_disabled_shutoff_drops_threshold_and_ceiling(self):
        s = format_progress(_progress_event(
            shutoff=0.0, phase="injecting", stable=False,
            mem_used_bytes=None, mem_total_bytes=None))
        assert "[injecting]" in s
        assert "UNSTABLE" in s
        assert "->" not in s              # no threshold arrow
        assert "ETA <=" not in s and "ETA 31s" in s
        assert "mem " not in s            # omitted on CPU


class TestTerminalRenderer:
    def test_non_tty_emits_plain_lines_and_shutoff_summary(self):
        buf = io.StringIO()
        r = TerminalRenderer(buf)
        assert r.tty is False
        r({"event": "start", "device": "cpu", "grid": [8, 8, 64],
           "shutoff": 1e-5})
        r(_progress_event(step=20, total=1000))
        r(_progress_event(step=40, total=1000, field_decay=1e-4))
        r({"event": "done", "wall_s": 1.5, "mcells_per_s": 400.0,
           "steps_run": 40, "shut_off": True, "field_decay": 9e-6})
        out = buf.getvalue()
        assert "8x8x64" in out
        assert "\r" not in out            # non-tty: no in-place redraw codes
        assert "step 20/1000" in out
        assert "shutoff at step 40" in out
        assert "done in 1.5s" in out

    def test_done_without_shutoff(self):
        buf = io.StringIO()
        r = TerminalRenderer(buf)
        r({"event": "progress", "step": 5, "total": 5, "t_s": 0.0})
        r({"event": "done", "wall_s": 0.5, "mcells_per_s": 100.0,
           "steps_run": 5, "shut_off": False})
        out = buf.getvalue()
        assert "done in 0.5s" in out
        assert "shutoff" not in out

    def test_error_event_renders_reason(self):
        buf = io.StringIO()
        default_renderer(buf)({"event": "error", "reason": "divergence"})
        assert "ERROR: divergence" in buf.getvalue()


# Fake phsolver that reports the --progress value it was launched with.
FAKE_REPORT_PROGRESS = textwrap.dedent("""\
    #!{python}
    import json, sys
    argv = sys.argv
    i = argv.index("--progress") if "--progress" in argv else -1
    val = argv[i + 1] if i >= 0 else "MISSING"
    print(json.dumps({{"event": "error", "reason": "progress=" + val}}),
          flush=True)
    sys.exit(1)
""")


def _fake_solver(tmp_path, body):
    path = tmp_path / "phsolver-fake"
    path.write_text(body.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_run_local_passes_progress_none(tmp_path, tiny_sim):
    solver = _fake_solver(tmp_path, FAKE_REPORT_PROGRESS)
    with pytest.raises(ph.SolverRunError, match="progress=none"):
        ph.run_local(tiny_sim, output_dir=tmp_path / "out", solver_path=solver,
                     quiet=True)
