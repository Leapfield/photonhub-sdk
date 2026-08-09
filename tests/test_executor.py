"""photonhub.executor — the provider-agnostic executor core, proven end-to-end on
CPU. execute() runs phsolver via the SAME run_phsolver helper as run_local, and
its result bundle round-trips through SimulationData exactly like a cloud job
would. Skips when no phsolver is built.
"""

import base64
import hashlib
import io
import json
import tarfile

import pytest

import photonhub as ph
from photonhub.executor import execute
from photonhub.executor import __main__ as executor_cli
from photonhub.executor.handler import handle
from photonhub.bundle import extract_bundle
from photonhub.runners.phsolver import SolverRunError
from photonhub.runners.local import find_solver

needs_solver = pytest.mark.skipif(
    find_solver() is None, reason="phsolver not built")


@needs_solver
def test_execute_bundle_roundtrips_through_simulationdata(tiny_sim, tmp_path):
    events = []
    result = execute(tiny_sim.to_wire_dict(), device="cpu",
                     on_event=events.append)

    # The shared run_phsolver streamed events and captured the `done` metrics.
    assert result.metrics.get("event") == "done"
    assert any(e.get("event") == "progress" for e in events)
    assert result.manifest["manifest_version"] == "1"
    assert result.manifest["provenance"]["backend"] == "cpu"
    assert "probe" in [m["name"] for m in result.manifest["monitors"]]

    # The bundle is the web/cache.py gzip-tar format: exact executed sim.json,
    # flat manifest.json, and monitor binaries.
    dest = tmp_path / "bundle"
    dest.mkdir()
    with tarfile.open(fileobj=io.BytesIO(result.bundle), mode="r:gz") as tar:
        names = tar.getnames()
    extract_bundle(result.bundle, dest)
    assert "sim.json" in names
    assert "manifest.json" in names
    assert any(n.endswith(".bin") for n in names)
    assert all("/" not in n and not n.startswith("..") for n in names)  # flat & safe
    executed = (dest / "sim.json").read_bytes()
    assert json.loads(executed) == tiny_sim.to_wire_dict()
    assert hashlib.sha256(executed).hexdigest() == (
        result.manifest["provenance"]["input_sha256"])

    # And SimulationData reads it exactly like a local run's output dir.
    data = ph.SimulationData(dest)
    assert data.provenance["backend"] == "cpu"


@needs_solver
def test_execute_missing_solver_raises(tiny_sim, tmp_path):
    with pytest.raises(SolverRunError):
        execute(tiny_sim.to_wire_dict(), device="cpu",
                solver_path=tmp_path / "does-not-exist")


@needs_solver
def test_cli_writes_bundle(tiny_sim, tmp_path):
    spec = tmp_path / "sim.json"
    spec.write_text(json.dumps(tiny_sim.to_wire_dict()), encoding="utf-8")
    out = tmp_path / "bundle.tar.gz"
    rc = executor_cli.main(
        ["--spec", str(spec), "--out", str(out), "--device", "cpu"])
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0
    with tarfile.open(out, mode="r:gz") as tar:
        assert "manifest.json" in tar.getnames()


@needs_solver
def test_handler_returns_inline_bundle(tiny_sim, tmp_path):
    # The RunPod handler's pure logic — no runpod dependency — runs the job and
    # returns the inline bundle the cloud poll/result path understands.
    event = {"input": {"spec": tiny_sim.to_wire_dict(),
                       "params": {"device": "cpu"}}}
    seen = []
    out = handle(event, progress=seen.append)
    assert out["ok"] is True
    assert out["provenance"]["backend"] == "cpu"
    assert out["metrics"].get("event") == "done"
    assert any(e.get("event") == "progress" for e in seen)

    dest = tmp_path / "b"
    dest.mkdir()
    extract_bundle(base64.b64decode(out["bundle_b64"]), dest)
    assert ph.SimulationData(dest).provenance["backend"] == "cpu"


def test_handler_rejects_non_dict_spec():
    out = handle({"input": {"spec": "not-a-dict"}})
    assert out["ok"] is False and "wire-JSON object" in out["reason"]
