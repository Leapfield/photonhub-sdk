"""run_local: solver discovery, event streaming, and error paths (driven by a
fake phsolver script), plus a real-binary integration test that skips when no
phsolver is built."""

import json
import stat
import sys
import textwrap

import numpy as np
import pytest

import photonhub as ph
from photonhub.runners.local import find_solver

FAKE_OK = textwrap.dedent("""\
    #!{python}
    import json, sys
    assert sys.argv[1] == "run" and sys.argv[3] == "--output", sys.argv
    sim = json.load(open(sys.argv[2]))
    assert sim["sources"], "sim.json must carry sources"
    out = sys.argv[4]
    print("phsolver fake starting")  # non-JSON chatter must be tolerated
    for step in (1, 3, 5):
        print(json.dumps({{"event": "progress", "step": step, "total": 5,
                           "field_energy": 1e-30 * step}}), flush=True)
    import struct
    data = struct.pack("<10f", *range(10))  # 5 samples x 2 components
    open(out + "/probe.bin", "wb").write(data)
    manifest = {{
        "manifest_version": "1",
        "monitors": [{{"name": "probe", "type": "field_time",
                       "file": "probe.bin", "dtype": "float32",
                       "shape": [5, 2], "dims": ["sample", "component"],
                       "components": ["Ez", "Hx"],
                       "sample_steps": [1, 2, 3, 4, 5], "dt_s": 1e-16}}],
        "run": {{"n_steps": 5, "dt_s": 1e-16, "wall_seconds": 0.01,
                 "mcells_per_s": 1.0, "aborted": False, "abort_reason": ""}},
        "grid": {{"shape": [4, 4, 4], "dl_um": 0.05}},
        "provenance": {{"solver_version": "fake", "device_name": "fake"}},
    }}
    json.dump(manifest, open(out + "/manifest.json", "w"))
    print(json.dumps({{"event": "done", "wall_seconds": 0.01}}), flush=True)
""")

# "Solver lies": exit 0 with a done event but no outputs written at all.
FAKE_OK_NO_MANIFEST = textwrap.dedent("""\
    #!{python}
    import json
    print(json.dumps({{"event": "done", "wall_seconds": 0.01}}), flush=True)
""")

# "Solver lies": exit 0 with a done event but a malformed manifest.
FAKE_OK_BAD_MANIFEST = textwrap.dedent("""\
    #!{python}
    import json, sys
    out = sys.argv[4]
    open(out + "/manifest.json", "w").write("{{not json")
    print(json.dumps({{"event": "done", "wall_seconds": 0.01}}), flush=True)
""")

FAKE_DIVERGES = textwrap.dedent("""\
    #!{python}
    import json, sys
    print(json.dumps({{"event": "progress", "step": 1, "total": 5}}), flush=True)
    print(json.dumps({{"event": "error", "reason": "divergence"}}), flush=True)
    sys.exit(3)
""")

FAKE_CRASHES = textwrap.dedent("""\
    #!{python}
    import sys
    sys.stderr.write("phsolver: catastrophic kaboom\\n")
    sys.exit(2)
""")

FAKE_HANGS = textwrap.dedent("""\
    #!{python}
    import time
    time.sleep(30)
""")

FAKE_REPORT_CLOUD_ENV = textwrap.dedent("""\
    #!{python}
    import json, os, sys
    names = [name for name in (
        "PHOTONHUB_API_KEY", "PHOTONHUB_URL",
        "PHOTONHUB_API_KEY", "PHOTONHUB_URL",
    ) if name in os.environ]
    print(json.dumps({{"event": "error", "reason":
                      "cloud env=" + (",".join(names) if names else "NONE")}}),
          flush=True)
    sys.exit(1)
""")


def fake_solver(tmp_path, body, name="phsolver-fake"):
    path = tmp_path / name
    path.write_text(body.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class TestDiscovery:
    def test_explicit_path_wins(self, tmp_path):
        solver = fake_solver(tmp_path, FAKE_OK)
        assert find_solver(solver) == solver

    def test_explicit_missing_path_raises(self, tmp_path):
        with pytest.raises(ph.SolverRunError, match="solver_path"):
            find_solver(tmp_path / "no-such-binary")

    def test_env_var_used_when_no_arg(self, tmp_path, monkeypatch):
        solver = fake_solver(tmp_path, FAKE_OK)
        monkeypatch.setenv("PHOTONHUB_SOLVER", str(solver))
        assert find_solver() == solver

    def test_broken_env_var_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PHOTONHUB_SOLVER", str(tmp_path / "missing"))
        with pytest.raises(ph.SolverRunError, match="PHOTONHUB_SOLVER"):
            find_solver()

    def test_legacy_env_var_still_honored(self, tmp_path, monkeypatch):
        # back-compat: $SIMUPOD_SOLVER is still read when $PHOTONHUB_SOLVER is unset
        monkeypatch.delenv("PHOTONHUB_SOLVER", raising=False)
        monkeypatch.setenv("SIMUPOD_SOLVER", str(tmp_path / "missing"))
        with pytest.raises(ph.SolverRunError):
            find_solver()


class TestRunLocal:
    def test_success_streams_events_and_loads_outputs(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_OK)
        events = []
        out = tmp_path / "out"
        data = ph.run_local(tiny_sim, output_dir=out, solver_path=solver,
                            progress=events.append)

        steps = [e["step"] for e in events if e.get("event") == "progress"]
        assert steps == [1, 3, 5]
        assert events[-1]["event"] == "done"

        assert (out / "sim.json").is_file()
        written = json.loads((out / "sim.json").read_text())
        assert written == tiny_sim.to_wire_dict()

        da = data["probe"]
        assert da.dims == ("t", "component")
        assert da.shape == (5, 2)
        np.testing.assert_allclose(da.coords["t"].values,
                                   np.arange(1, 6) * 1e-16)

    def test_temp_output_dir_created_when_unset(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_OK)
        data = ph.run_local(tiny_sim, solver_path=solver)
        assert data.output_dir.is_dir()
        assert "probe" in data

    def test_error_event_raises(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_DIVERGES)
        with pytest.raises(ph.SolverRunError, match="divergence"):
            ph.run_local(tiny_sim, output_dir=tmp_path / "out", solver_path=solver)

    def test_nonzero_exit_raises_with_stderr_tail(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_CRASHES)
        with pytest.raises(ph.SolverRunError, match="kaboom") as exc_info:
            ph.run_local(tiny_sim, output_dir=tmp_path / "out", solver_path=solver)
        assert exc_info.value.returncode == 2

    def test_timeout_kills_solver(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_HANGS)
        with pytest.raises(ph.SolverRunError, match="timed out"):
            ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                         solver_path=solver, timeout=0.5)

    def test_solver_child_does_not_inherit_cloud_credentials(
            self, tmp_path, tiny_sim, monkeypatch):
        solver = fake_solver(tmp_path, FAKE_REPORT_CLOUD_ENV)
        for name in ("PHOTONHUB_API_KEY", "PHOTONHUB_URL",
                     "PHOTONHUB_API_KEY", "PHOTONHUB_URL"):
            monkeypatch.setenv(name, "must-not-reach-phsolver")
        with pytest.raises(ph.SolverRunError, match="cloud env=NONE"):
            ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                         solver_path=solver)

    def test_clean_exit_without_manifest_raises_solver_run_error(
            self, tmp_path, tiny_sim):
        # Exit 0 + done event but no outputs: still a SolverRunError, not a
        # raw FileNotFoundError, so callers have one exception surface.
        solver = fake_solver(tmp_path, FAKE_OK_NO_MANIFEST)
        with pytest.raises(ph.SolverRunError, match="unreadable"):
            ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                         solver_path=solver)

    def test_clean_exit_with_malformed_manifest_raises_solver_run_error(
            self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_OK_BAD_MANIFEST)
        with pytest.raises(ph.SolverRunError, match="unreadable"):
            ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                         solver_path=solver)


# Reports back which --device the runner passed (or NONE), via an error event —
# lets us assert passthrough without writing a full fake manifest.
FAKE_REPORT_DEVICE = textwrap.dedent("""\
    #!{python}
    import json, sys
    argv = sys.argv
    i = argv.index("--device") if "--device" in argv else -1
    dev = argv[i + 1] if i >= 0 else "NONE"
    print(json.dumps({{"event": "error", "reason": "saw device=" + dev}}),
          flush=True)
    sys.exit(1)
""")

# Reports back the --log-file path the runner passed (or NONE), via an error
# event — lets us assert passthrough without a full fake manifest.
FAKE_REPORT_LOG = textwrap.dedent("""\
    #!{python}
    import json, sys
    argv = sys.argv
    i = argv.index("--log-file") if "--log-file" in argv else -1
    path = argv[i + 1] if i >= 0 else "NONE"
    print(json.dumps({{"event": "error", "reason": "saw logfile=" + path}}),
          flush=True)
    sys.exit(1)
""")


class TestDevice:
    def test_device_args_builds_flag(self):
        # device_args is shared between the local runner and the cloud executor
        # (runners.phsolver), so the device grammar has one definition.
        from photonhub.runners.phsolver import device_args
        assert device_args(None) == []
        assert device_args("cpu") == ["--device", "cpu"]
        assert device_args("gpu") == ["--device", "gpu"]
        assert device_args("gpu:3") == ["--device", "gpu:3"]
        assert device_args(" gpu:0 ") == ["--device", "gpu:0"]  # trimmed
        # Multi-GPU z-decomposition selectors.
        assert device_args("gpu:all") == ["--device", "gpu:all"]
        assert device_args("gpu:0,1") == ["--device", "gpu:0,1"]
        assert device_args("gpu:0,2,5") == ["--device", "gpu:0,2,5"]

    @pytest.mark.parametrize(
        "bad",
        ["tpu", "gpu:", "gpu:x", "GPU", "", "cpu:0", "gpu:0,", "gpu:0,x",
         "gpu:,1", "gpu:All"])
    def test_device_args_rejects_bad(self, bad):
        from photonhub.runners.phsolver import device_args
        with pytest.raises(ph.SolverRunError, match="invalid device"):
            device_args(bad)

    def test_run_local_passes_device_through(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_REPORT_DEVICE)
        with pytest.raises(ph.SolverRunError, match="saw device=gpu:1"):
            ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                         solver_path=solver, device="gpu:1")

    def test_run_local_omits_device_when_unset(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_REPORT_DEVICE)
        with pytest.raises(ph.SolverRunError, match="saw device=NONE"):
            ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                         solver_path=solver)


class TestLogFile:
    def test_passes_log_file_through(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_REPORT_LOG)
        log = tmp_path / "run.jsonl"
        with pytest.raises(ph.SolverRunError) as exc_info:
            ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                         solver_path=solver, log_file=log)
        assert f"saw logfile={log}" in str(exc_info.value)

    def test_omits_log_file_when_unset(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_REPORT_LOG)
        with pytest.raises(ph.SolverRunError, match="saw logfile=NONE"):
            ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                         solver_path=solver)

    def test_creates_log_parent_dir(self, tmp_path, tiny_sim):
        # run_local mkdirs the log's parent before spawning, so a nested path
        # works without the caller pre-creating it.
        solver = fake_solver(tmp_path, FAKE_REPORT_LOG)
        log = tmp_path / "nested" / "logs" / "run.jsonl"
        with pytest.raises(ph.SolverRunError, match="saw logfile="):
            ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                         solver_path=solver, log_file=log)
        assert log.parent.is_dir()


def _real_solver():
    try:
        return find_solver()
    except ph.SolverRunError:
        return None


@pytest.mark.skipif(_real_solver() is None,
                    reason="no phsolver binary found (build the engine first)")
def test_integration_real_solver(tmp_path, tiny_sim):
    events = []
    data = ph.run_local(tiny_sim, output_dir=tmp_path / "out",
                        progress=events.append, timeout=300)
    assert "probe" in data and "final" in data

    probe = data["probe"]
    assert probe.dims == ("t", "component")
    assert list(probe.coords["component"].values) == ["Ez"]
    assert probe.sizes["t"] == 5  # interval_steps=1, n_steps=5

    snap = data["final"]
    assert snap.dims == ("t", "component", "z", "y", "x")
    # size 0.2 um / dl 0.05 -> 4 cells/axis (the n >= 4 floor)
    assert snap.sizes["x"] == snap.sizes["y"] == snap.sizes["z"] == 4
    assert snap.sizes["t"] == 1  # interval_steps=0: final step only
    assert np.isfinite(snap.values).all()


@pytest.mark.skipif(_real_solver() is None,
                    reason="no phsolver binary found (build the engine first)")
def test_integration_real_solver_nonfinite_abort(tmp_path):
    # NUMERICS.md section 7 end-to-end with the REAL engine: a 1e42 A/m^2
    # dipole overflows the fp32 fields, the solver emits
    # {"event": "error", "reason": "non_finite_energy"}, and run_local
    # surfaces exactly that reason string.
    from .helpers import make_sim

    sim = make_sim(
        run=ph.RunSpec(n_steps=200),
        sources=[ph.PointDipole(
            center_um=(0.1, 0.1, 0.1), polarization="Ez", amplitude=1.0e42,
            source_time=ph.GaussianPulse(freq0_hz=2.0e14, fwidth_hz=1.0e14,
                                         offset=1.0),
        )],
    )
    with pytest.raises(ph.SolverRunError, match="non_finite_energy"):
        ph.run_local(sim, output_dir=tmp_path / "out", timeout=300)


@pytest.mark.skipif(_real_solver() is None,
                    reason="no phsolver binary found (build the engine first)")
def test_integration_mesh_override_runs_on_cpu(tmp_path):
    # End-to-end: a geometry-based MeshOverride refines the mesh around a central
    # region (a graded GridSpec, NUMERICS §15) and the REAL CPU reference solver
    # runs that graded mesh to completion with finite fields. The override-
    # produced graded mesh is what the engine actually consumes; the GPU runs the
    # same graded path (GpuEquivalence.GradedMeshMatchesCpu, box-verified) for the
    # periodic/PEC + materials + dipole subset — see GPU_TODO §3.
    f0 = 1.934e14  # ~1.55 um
    base = ph.Simulation(
        size_um=(2.0, 2.0, 2.0),
        grid=ph.UniformGridSpec(dl_um=0.1),
        run=ph.RunSpec(n_steps=40),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="periodic"),
        sources=[ph.PointDipole(
            center_um=(1.0, 1.0, 1.0), polarization="Ez",
            source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=4.0e13))],
        monitors=[
            ph.FieldTimeMonitor(name="probe", center_um=(1.3, 1.0, 1.0),
                                fields=["Ez"]),
            ph.FieldSnapshotMonitor(name="final", fields=["Ez", "Hx"]),
        ],
    )
    override = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.6, 0.6, 0.6)),
        dl_um=0.04)
    sim = base.with_mesh_overrides(override, steps_per_wvl=12.0, max_grading=1.3)
    # The mesh override produced a genuinely graded grid (per-axis coords).
    assert sim.grid.type == "graded"
    assert min(graded := list(  # finest cell is well below the 0.1 um base dl
        sim.grid.coords.x[i + 1] - sim.grid.coords.x[i]
        for i in range(len(sim.grid.coords.x) - 1))) <= 0.05
    # Periodic boundaries on a graded axis require EQUAL first/last primary
    # spacings (NUMERICS.md §15.2 replicate-closure seam; the engine's
    # validate hard-rejects otherwise). with_mesh_overrides derived
    # periodic_axes="xyz" from the boundaries, so the generated mesh closes
    # seam-symmetrically — this scene end-to-ends that path on the engine.
    for q in (sim.grid.coords.x, sim.grid.coords.y, sim.grid.coords.z):
        df, dl_ = q[1] - q[0], q[-1] - q[-2]
        assert abs(df - dl_) <= 1e-12 * max(df, dl_)

    data = ph.run_local(sim, output_dir=tmp_path / "out", timeout=300)
    assert "probe" in data and "final" in data
    snap = data["final"]
    assert np.isfinite(snap.values).all()
    # The graded mesh has more cells than the 20-cell uniform base would (the
    # central refinement), confirming the override reached the solver's grid.
    assert snap.sizes["x"] > 20


@pytest.mark.skipif(_real_solver() is None,
                    reason="no phsolver binary found (build the engine first)")
def test_integration_log_file_records_events(tmp_path, tiny_sim):
    # The REAL engine mirrors its JSON-lines event stream to --log-file: the
    # file must hold a start event, at least one progress event (carrying the
    # convergence/decay diagnostics we log for), and a terminal done event,
    # each a parseable JSON object on its own line. The parent dir is created
    # by run_local.
    log = tmp_path / "logs" / "run.jsonl"
    ph.run_local(tiny_sim, output_dir=tmp_path / "out", log_file=log,
                 timeout=300)

    assert log.is_file()
    events = [json.loads(line) for line in log.read_text().splitlines()
              if line.strip()]
    kinds = [e.get("event") for e in events]
    assert kinds[0] == "start"
    assert "progress" in kinds
    assert kinds[-1] == "done"

    prog = next(e for e in events if e.get("event") == "progress")
    assert "field_decay" in prog and "step" in prog
