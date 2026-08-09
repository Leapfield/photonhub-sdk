"""Batch / run_async local runner surface (driven by the fake phsolver from
test_run_local). Covers the job handle, dict-like BatchData, and the
partial-failure semantics that keep one bad sim from sinking a sweep."""

import stat
import sys
import textwrap
import time

import pytest

import photonhub as ph
from .helpers import make_sim
from .test_run_local import FAKE_DIVERGES, FAKE_OK, fake_solver

# Sleeps briefly, then behaves like FAKE_OK — lets us observe a not-yet-done
# job and a result(timeout) that fires while the run is still going.
FAKE_SLOW_OK = textwrap.dedent("""\
    #!{python}
    import json, sys, time
    time.sleep(0.8)
    out = sys.argv[4]
    import struct
    open(out + "/probe.bin", "wb").write(struct.pack("<10f", *range(10)))
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
        "provenance": {{"solver_version": "fake"}},
    }}
    json.dump(manifest, open(out + "/manifest.json", "w"))
    print(json.dumps({{"event": "done"}}), flush=True)
""")

# Diverges only for "large" sims (size_um[0] >= 1.0); otherwise behaves like
# FAKE_OK. Lets one Batch.run mix a success and a failure to exercise the
# partial-failure path within a single batch.
FAKE_SELECTIVE = textwrap.dedent("""\
    #!{python}
    import json, struct, sys
    sim = json.load(open(sys.argv[2]))
    out = sys.argv[4]
    if sim["size_um"][0] >= 1.0:
        print(json.dumps({{"event": "error", "reason": "divergence"}}), flush=True)
        sys.exit(3)
    open(out + "/probe.bin", "wb").write(struct.pack("<10f", *range(10)))
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
        "provenance": {{"solver_version": "fake"}},
    }}
    json.dump(manifest, open(out + "/manifest.json", "w"))
    print(json.dumps({{"event": "done"}}), flush=True)
""")


class TestRunAsync:
    def test_returns_handle_and_result(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_OK)
        job = ph.run_async(tiny_sim, output_dir=tmp_path / "o",
                           solver_path=solver, name="j0")
        data = job.result(timeout=30)
        assert job.done
        assert job.name == "j0"
        assert "probe" in data

    def test_result_times_out_while_running(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_SLOW_OK)
        job = ph.run_async(tiny_sim, output_dir=tmp_path / "o",
                           solver_path=solver)
        # The run is still sleeping; a short wait must time out, not block.
        with pytest.raises(TimeoutError):
            job.result(timeout=0.05)
        assert not job.done
        # The same job still completes when we wait it out.
        assert "probe" in job.result(timeout=30)

    def test_failure_reraised_in_caller(self, tmp_path, tiny_sim):
        solver = fake_solver(tmp_path, FAKE_DIVERGES)
        job = ph.run_async(tiny_sim, output_dir=tmp_path / "o",
                           solver_path=solver)
        with pytest.raises(ph.SolverRunError, match="divergence"):
            job.result(timeout=30)


class TestBatchConstruction:
    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            ph.Batch({})

    @pytest.mark.parametrize("bad", ["a/b", "a\\b", ".", ".."])
    def test_unsafe_key_rejected(self, bad, tiny_sim):
        with pytest.raises(ValueError, match="directory name"):
            ph.Batch({bad: tiny_sim})

    def test_non_simulation_value_rejected(self):
        with pytest.raises(TypeError, match="expected a Simulation"):
            ph.Batch({"a": object()})

    def test_estimate_cost_totals(self, tiny_sim):
        batch = ph.Batch({"a": tiny_sim, "b": make_sim(monitors=[])})
        per_sim, total = batch.estimate_cost()
        assert set(per_sim) == {"a", "b"}
        assert total == pytest.approx(sum(e.usd for e in per_sim.values()))


class TestBatchRun:
    def test_all_succeed(self, tmp_path):
        solver = fake_solver(tmp_path, FAKE_OK)
        batch = ph.Batch({"w20": make_sim(monitors=[]),
                          "w40": make_sim(monitors=[])})
        bd = batch.run(path_dir=tmp_path / "sweep", solver_path=solver)
        assert sorted(bd.keys()) == ["w20", "w40"]
        assert bd.errors == {}
        assert len(bd) == 2
        assert dict(bd.items()).keys() == {"w20", "w40"}
        assert (tmp_path / "sweep" / "w20" / "manifest.json").is_file()
        assert "probe" in bd["w20"]

    def test_partial_failure_is_captured(self, tmp_path):
        # One batch, one solver: the small sim succeeds, the large one diverges.
        solver = fake_solver(tmp_path, FAKE_SELECTIVE)
        bd = ph.Batch({
            "good": make_sim(monitors=[]),                       # 0.2 um
            "bad": make_sim(monitors=[], size_um=(2.0, 2.0, 2.0)),  # diverges
        }).run(path_dir=tmp_path / "s", solver_path=solver)

        assert bd.succeeded == ["good"]
        assert bd.failed == ["bad"]
        assert set(bd.names) == {"good", "bad"}   # both submitted
        assert "probe" in bd["good"]
        # Indexing a failed name re-raises its captured error.
        with pytest.raises(ph.SolverRunError, match="divergence"):
            _ = bd["bad"]

    def test_unknown_name_keyerror(self, tmp_path):
        solver = fake_solver(tmp_path, FAKE_OK)
        bd = ph.Batch({"a": make_sim(monitors=[])}).run(
            path_dir=tmp_path / "s", solver_path=solver)
        with pytest.raises(KeyError):
            _ = bd["nope"]

    def test_progress_tagged_with_name(self, tmp_path):
        solver = fake_solver(tmp_path, FAKE_OK)
        seen = []
        ph.Batch({"a": make_sim(monitors=[])}).run(
            path_dir=tmp_path / "s", solver_path=solver,
            progress=lambda name, ev: seen.append((name, ev.get("event"))))
        assert ("a", "progress") in seen

    def test_concurrent_workers_complete_all(self, tmp_path):
        solver = fake_solver(tmp_path, FAKE_SLOW_OK)
        sims = {f"s{i}": make_sim(monitors=[]) for i in range(3)}
        t0 = time.time()
        bd = ph.Batch(sims).run(path_dir=tmp_path / "s", solver_path=solver,
                                max_workers=3)
        elapsed = time.time() - t0
        assert len(bd) == 3
        # 3 x 0.8s sleeps run concurrently should finish well under the 2.4s
        # serial sum (generous bound to avoid CI flakiness).
        assert elapsed < 2.0
