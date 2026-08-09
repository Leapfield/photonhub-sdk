"""Immutable desktop run history, integrity, and A/B contracts."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from photonhub.viz import server, service
from photonhub.viz.ledger import LedgerError, RunLedger


def _canonical(size_x: float = 4.0, *, monitor: str = "flux",
               freqs=(190e12, 200e12)) -> str:
    spec = service.default_sim().to_wire_dict()
    spec["size_um"][0] = size_x
    if monitor == "flux":
        spec["monitors"] = [{
            "type": "flux", "name": "port", "axis": "x",
            "position_um": min(3.5, size_x - 0.25),
            "freqs_hz": list(freqs),
        }]
    elif monitor == "field":
        spec["monitors"] = [{
            "type": "field_dft", "name": "field",
            "center_um": [1.0, 1.0, 0.0],
            "size_um": [0.2, 0.2, 0.0],
            "fields": ["Ex"], "freqs_hz": list(freqs),
        }]
    elif monitor == "time":
        spec["monitors"] = [{
            "type": "field_time", "name": "probe",
            "center_um": [1.0, 1.0, 0.5],
            "fields": ["Ex"], "interval_steps": 1,
        }]
    elif monitor == "none":
        spec["monitors"] = []
    else:
        raise AssertionError(f"unsupported test monitor kind: {monitor}")
    sim, _ = service.parse_sim_spec(spec)
    return sim.to_wire_json() + "\n"


def _request(ledger: RunLedger, canonical: str, *, output_parent=None,
             timeout_s=None) -> dict:
    return ledger.create_request(
        run_id=None, canonical_spec=canonical, device="cpu", timeout_s=timeout_s,
        solver={"available": True, "path": "/test/phsolver",
                "info": {"git_sha": "abc123"}, "capabilities": {}},
        estimate={"num_cells": 8}, output_parent=output_parent,
    )


def _write_flux_bundle(record: dict, canonical: str, *,
                       freqs=(190e12, 200e12), values=(0.4, 0.5),
                       aborted=False) -> Path:
    out = Path(record["output_dir"])
    (out / "sim.json").write_text(canonical, encoding="utf-8")
    (out / "solver-events.jsonl").write_text(
        json.dumps({"event": "done"}) + "\n", encoding="utf-8")
    np.asarray(values, dtype="<f4").tofile(out / "port.bin")
    manifest = {
        "manifest_version": "1", "schema_version": "1.0.0",
        "run": {
            "n_steps": 100, "steps_run": 90, "dt_s": 1e-16,
            "wall_seconds": 2.0, "setup_seconds": 0.2,
            "mcells_per_s": 3.0, "aborted": aborted,
            "abort_reason": "divergence" if aborted else "",
        },
        "grid": {"shape": [2, 2, 2], "dl_um": 0.1,
                 "size_um": [0.2, 0.2, 0.2]},
        "provenance": {
            "solver_version": "test", "git_sha": "abc123",
            "backend": "cpu", "device_name": "cpu_ref",
            "input_sha256": record["spec_sha256"],
        },
        "monitors": [{
            "name": "port", "type": "flux", "file": "port.bin",
            "dtype": "float32", "shape": [len(values)], "dims": ["freq"],
            "components": [], "freqs_hz": list(freqs), "axis": "x",
            "dt_s": 1e-16,
        }],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8")
    return out


def _write_plane_bundle(record: dict, canonical: str, *, dl_um: float) -> Path:
    out = Path(record["output_dir"])
    (out / "sim.json").write_text(canonical, encoding="utf-8")
    (out / "solver-events.jsonl").write_text(
        json.dumps({"event": "done"}) + "\n", encoding="utf-8")
    # [freq, component, z, y, x, complex] = [1,1,1,2,2,2]
    np.arange(8, dtype="<f4").tofile(out / "field.bin")
    manifest = {
        "manifest_version": "1", "schema_version": "1.0.0",
        "run": {"n_steps": 5, "steps_run": 5, "dt_s": 1e-16,
                "wall_seconds": 1.0, "aborted": False, "abort_reason": ""},
        "grid": {"shape": [2, 2, 1], "dl_um": dl_um,
                 "size_um": [2 * dl_um, 2 * dl_um, dl_um]},
        "provenance": {"input_sha256": record["spec_sha256"]},
        "monitors": [{
            "name": "field", "type": "field_dft", "file": "field.bin",
            "dtype": "float32", "shape": [1, 1, 1, 2, 2, 2],
            "dims": ["freq", "component", "z", "y", "x", "complex"],
            "components": ["Ex"], "freqs_hz": [193.4e12],
            "origin_cells": [0, 0, 0], "dt_s": 1e-16,
        }],
    }
    (out / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return out


def _write_time_bundle(record: dict, canonical: str, *, steps=(1,)) -> Path:
    out = Path(record["output_dir"])
    (out / "sim.json").write_text(canonical, encoding="utf-8")
    (out / "solver-events.jsonl").write_text(
        json.dumps({"event": "done"}) + "\n", encoding="utf-8")
    np.arange(len(steps), dtype="<f4").tofile(out / "probe.bin")
    manifest = {
        "manifest_version": "1", "schema_version": "1.0.0",
        "run": {"n_steps": 10, "steps_run": 10, "dt_s": 1e-16,
                "aborted": False, "abort_reason": ""},
        "grid": {"shape": [2, 2, 2], "dl_um": 0.1},
        "provenance": {"input_sha256": record["spec_sha256"]},
        "monitors": [{
            "name": "probe", "type": "field_time", "file": "probe.bin",
            "dtype": "float32", "shape": [len(steps), 1],
            "dims": ["sample", "component"], "components": ["Ex"],
            "sample_steps": list(steps), "dt_s": 1e-16,
        }],
    }
    (out / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return out


def test_append_only_ledger_persists_and_rejects_mutation(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    run_id = record["run_id"]
    assert record["recorded_status"] == "queued"
    ledger.append_event(run_id, "running")
    _write_flux_bundle(record, canonical)
    sealed = ledger.seal(run_id, "completed")
    assert sealed["status"] == "completed"
    assert sealed["run"]["steps_run"] == 90
    assert sealed["integrity"]["spec_matches_manifest"] is True
    assert {item["path"] for item in sealed["integrity"]["artifacts"]} == {
        "sim.json", "manifest.json", "solver-events.jsonl", "port.bin",
    }

    reopened = RunLedger(tmp_path / "ledger")
    listing = reopened.list_runs()
    assert listing["root"] == str((tmp_path / "ledger").resolve())
    assert [item["run_id"] for item in listing["runs"]] == [run_id]
    assert reopened.get_run(run_id)["spec"] == json.loads(canonical)
    assert reopened.get_run(run_id)["terminal"]["request_sha256"] == record["request_sha256"]

    with sqlite3.connect(reopened.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE run_requests SET device='gpu' WHERE run_id=?", (run_id,))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM run_events WHERE run_id=?", (run_id,))


def test_hashed_records_and_cached_artifacts_detect_out_of_band_changes(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    ledger.seal(record["run_id"], "completed")

    # The first numerical verification populates the stat-signature cache.
    ledger.verify_artifact(record["run_id"], "port.bin")
    artifact = out / "port.bin"
    old = artifact.stat()
    artifact.write_bytes(b"x" * old.st_size)
    os.utime(artifact, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(LedgerError, match="integrity verification"):
        ledger.verify_artifact(record["run_id"], "port.bin")

    # Triggers enforce normal application writes; read-time hash validation also
    # detects a record edited by a tool that first removed that enforcement.
    with sqlite3.connect(ledger.db_path) as conn:
        conn.execute("DROP TRIGGER immutable_requests_update")
        conn.execute(
            "UPDATE run_requests SET request_json=? WHERE run_id=?",
            ("{}", record["run_id"]),
        )
    with pytest.raises(LedgerError, match="request record failed"):
        ledger.get_run(record["run_id"])


def test_repeated_artifact_verification_uses_verified_terminal_index(
        tmp_path, monkeypatch):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    _write_flux_bundle(record, canonical)
    ledger.seal(record["run_id"], "completed")
    ledger.verify_artifact(record["run_id"], "port.bin")
    monkeypatch.setattr(
        ledger, "get_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached numerical reads must not parse full records")),
    )
    ledger.verify_artifact(record["run_id"], "port.bin")


def test_replaced_output_directory_symlink_is_rejected(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    ledger.seal(record["run_id"], "completed")
    moved = out.with_name(out.name + "-moved")
    out.rename(moved)
    out.symlink_to(moved, target_is_directory=True)
    with pytest.raises(LedgerError, match="not a regular directory"):
        ledger.verify_bundle_metadata(record["run_id"])


def test_default_archive_is_unique_and_unfinished_run_is_interrupted(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    first = _request(ledger, _canonical())
    second = _request(ledger, _canonical())
    assert first["run_id"] != second["run_id"]
    assert first["output_dir"] != second["output_dir"]
    assert Path(first["output_dir"]).parent == ledger.bundles_dir
    assert ledger.get_run(first["run_id"])["status"] == "interrupted"
    assert ledger.get_run(first["run_id"], active=True)["status"] == "queued"


def test_request_serialization_failure_leaves_no_orphan_output_directory(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    with pytest.raises(ValueError, match="JSON compliant"):
        _request(ledger, _canonical(), timeout_s=float("nan"))
    assert list(ledger.bundles_dir.iterdir()) == []


def test_lifecycle_rejects_out_of_order_or_repeated_transitions(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    with pytest.raises(LedgerError, match="queued.*queued"):
        ledger.append_event(record["run_id"], "queued")
    with pytest.raises(LedgerError, match="queued.*completed"):
        ledger.seal(record["run_id"], "completed")
    ledger.append_event(record["run_id"], "running")
    with pytest.raises(LedgerError, match="running.*running"):
        ledger.append_event(record["run_id"], "running")
    _write_flux_bundle(record, canonical)
    assert ledger.seal(record["run_id"], "completed")["status"] == "completed"


def test_seal_racing_queued_to_running_records_final_started_at(
        tmp_path, monkeypatch):
    ledger = RunLedger(tmp_path / "ledger")
    record = _request(ledger, _canonical())
    entered = threading.Event()
    release = threading.Event()
    original = ledger._artifact_inventory  # noqa: SLF001

    def delayed_inventory(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(ledger, "_artifact_inventory", delayed_inventory)
    result: dict = {}

    def seal_failed():
        result["record"] = ledger.seal(
            record["run_id"], "failed",
            error={"type": "test", "message": "intentional"},
        )

    thread = threading.Thread(target=seal_failed)
    thread.start()
    assert entered.wait(2)
    ledger.append_event(record["run_id"], "running")
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    terminal = result["record"]["terminal"]
    assert terminal["started_at"] is not None
    with sqlite3.connect(ledger.db_path) as conn:
        statuses = [row[0] for row in conn.execute(
            "SELECT status FROM run_events WHERE run_id=? ORDER BY seq",
            (record["run_id"],),
        )]
    assert statuses == ["queued", "running", "failed"]


def test_history_listing_does_not_decode_large_request_documents(
        tmp_path, monkeypatch):
    ledger = RunLedger(tmp_path / "ledger")
    record = _request(ledger, _canonical())

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("listing must not decode request_json")

    monkeypatch.setattr(RunLedger, "_decode_hashed_json", staticmethod(fail_decode))
    listing = ledger.list_runs()
    assert listing["runs"][0]["run_id"] == record["run_id"]


def test_completed_seal_fails_closed_on_input_or_artifact_tamper(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    (out / "sim.json").write_text(_canonical(5.0), encoding="utf-8")
    with pytest.raises(LedgerError, match="immutable execution request"):
        ledger.seal(record["run_id"], "completed")
    failed = ledger.seal(
        record["run_id"], "failed",
        error={"type": "LedgerError", "message": "input mismatch"},
    )
    assert failed["status"] == "failed"
    assert failed["integrity"]["spec_matches_manifest"] is False


def test_completed_seal_rejects_monitor_byte_count_mismatch(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    (out / "port.bin").write_bytes(b"\x00" * 4)  # manifest declares two float32s
    with pytest.raises(LedgerError, match="dtype/shape requires 8"):
        ledger.seal(record["run_id"], "completed")


def test_completed_seal_rejects_reader_invalid_monitor_contract(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][0]["dims"] = ["wrong"]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(LedgerError, match="invalid monitor manifest contract"):
        ledger.seal(record["run_id"], "completed")


def test_completed_seal_rejects_invalid_monitor_scalars_and_top_level_blocks(
        tmp_path):
    for mutation, message in (
        (lambda manifest: manifest["monitors"][0].update(
            {"freqs_hz": ["bad", "worse"]}), "freqs_hz"),
        (lambda manifest: manifest.update({"run": []}),
         "manifest run is not a JSON object"),
        (lambda manifest: manifest.update({"monitors": {}}),
         "manifest monitors is not a list"),
    ):
        ledger = RunLedger(tmp_path / message.replace(" ", "_"))
        canonical = _canonical()
        record = _request(ledger, canonical)
        ledger.append_event(record["run_id"], "running")
        out = _write_flux_bundle(record, canonical)
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        mutation(manifest)
        manifest_path.write_text(json.dumps(manifest) + "\n")
        with pytest.raises(LedgerError, match=message):
            ledger.seal(record["run_id"], "completed")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.update({"manifest_version": "2"}),
         "unsupported manifest_version"),
        (lambda manifest: manifest["monitors"][0].update(
            {"name": "../unsafe"}), "unsafe name"),
        (lambda manifest: manifest["monitors"].append(
            dict(manifest["monitors"][0])), "duplicate monitor name"),
        (lambda manifest: manifest["monitors"][0].update(
            {"shape": [10**100]}), "invalid shape"),
        (lambda manifest: manifest["monitors"][0].pop("dims"),
         "must declare dims"),
    ],
)
def test_completed_seal_rejects_reader_invalid_manifest_envelope(
        tmp_path, mutation, message):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(LedgerError, match=message):
        ledger.seal(record["run_id"], "completed")


def test_completed_seal_binds_monitor_inventory_to_request(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][0]["name"] = "different"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(LedgerError, match="missing requested monitors"):
        ledger.seal(record["run_id"], "completed")


def test_completed_seal_rejects_monitor_artifact_alias(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    spec = json.loads(_canonical())
    second = dict(spec["monitors"][0])
    second["name"] = "port_2"
    spec["monitors"].append(second)
    sim, _ = service.parse_sim_spec(spec)
    canonical = sim.to_wire_json() + "\n"
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    alias = dict(manifest["monitors"][0])
    alias["name"] = "port_2"
    manifest["monitors"].append(alias)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(LedgerError, match="alias result filename|shared by monitors"):
        ledger.seal(record["run_id"], "completed")


def test_completed_seal_requires_engine_monitor_filename_mapping(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    (out / "port.bin").rename(out / "renamed.bin")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][0]["file"] = "renamed.bin"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(LedgerError, match="engine result filename"):
        ledger.seal(record["run_id"], "completed")


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ((10**400,), "32-bit integers"),
        ((2, 2), "strictly increasing"),
        ((11,), "exceed completed run steps"),
    ],
)
def test_completed_seal_rejects_unusable_time_coordinates(tmp_path, steps, message):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical(monitor="time")
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    _write_time_bundle(record, canonical, steps=steps)
    with pytest.raises(LedgerError, match=message):
        ledger.seal(record["run_id"], "completed")


def test_completed_seal_rejects_empty_dft_spatial_extent(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical(monitor="field", freqs=(193.4e12,))
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_plane_bundle(record, canonical, dl_um=0.1)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][0]["shape"] = [1, 1, 0, 2, 2, 2]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    (out / "field.bin").write_bytes(b"")
    with pytest.raises(LedgerError, match="spatial extents must be positive"):
        ledger.seal(record["run_id"], "completed")


def test_request_models_reject_duplicate_monitor_coordinates():
    duplicate_fields = json.loads(_canonical(monitor="field", freqs=(193.4e12,)))
    duplicate_fields["monitors"][0]["fields"] = ["Ex", "Ex"]
    with pytest.raises(Exception, match="fields must be unique"):
        service.parse_sim_spec(duplicate_fields)

    duplicate_freqs = json.loads(_canonical())
    duplicate_freqs["monitors"][0]["freqs_hz"] = [190e12, 190e12]
    with pytest.raises(Exception, match="freqs_hz must be unique"):
        service.parse_sim_spec(duplicate_freqs)


def test_completed_seal_rejects_invalid_or_out_of_grid_field_coordinates(tmp_path):
    for name, mutate, message in (
        ("non_numeric", lambda manifest: manifest["grid"].update({
            "coords_um": {"x": [0.0, "bad"], "y": [0.0, 0.1], "z": [0.0]}}),
         "must contain numbers"),
        ("extent", lambda manifest: manifest["monitors"][0].update(
            {"origin_cells": [1, 0, 0]}), "samples exceed grid extent"),
    ):
        ledger = RunLedger(tmp_path / name)
        canonical = _canonical(monitor="field", freqs=(193.4e12,))
        record = _request(ledger, canonical)
        ledger.append_event(record["run_id"], "running")
        out = _write_plane_bundle(record, canonical, dl_um=0.1)
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        mutate(manifest)
        manifest_path.write_text(json.dumps(manifest) + "\n")
        with pytest.raises(LedgerError, match=message):
            ledger.seal(record["run_id"], "completed")


def test_sealed_artifact_replacement_is_rejected_even_with_same_bytes(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    ledger.seal(record["run_id"], "completed")
    original = (out / "port.bin").read_bytes()
    replacement = out / "replacement.bin"
    replacement.write_bytes(original)
    replacement.replace(out / "port.bin")
    with pytest.raises(LedgerError, match="inode changed"):
        ledger.verify_artifact(record["run_id"], "port.bin")


def test_compare_keeps_independent_spectrum_grids_and_exact_spec_diff(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical_a = _canonical(4.0, freqs=(190e12, 200e12))
    canonical_b = _canonical(5.0, freqs=(190e12, 205e12))
    a = _request(ledger, canonical_a)
    b = _request(ledger, canonical_b, timeout_s=3.0)
    for record in (a, b):
        ledger.append_event(record["run_id"], "running")
    _write_flux_bundle(a, canonical_a, freqs=(190e12, 200e12), values=(0.4, 0.5))
    _write_flux_bundle(b, canonical_b, freqs=(190e12, 205e12), values=(0.45, 0.55))
    ledger.seal(a["run_id"], "completed")
    ledger.seal(b["run_id"], "completed")
    ar, br = ledger.get_run(a["run_id"]), ledger.get_run(b["run_id"])
    comparison = service.compare_run_data(
        ar, br, service.load_result(ar["output_dir"]),
        service.load_result(br["output_dir"]),
    )
    assert comparison["compatibility"]["eligible"] is True
    assert comparison["compatibility"]["shared_spectra"][0]["name"] == "port"
    spectrum = comparison["spectra"][0]
    assert spectrum["grids_equal"] is False
    assert spectrum["delta"] is None
    assert any("without interpolation" in reason for reason in spectrum["reasons"])
    assert any(item["path"] == "/size_um/0" for item in comparison["spec_diff"])
    assert any(item["path"] == "/requested/execution/timeout_s"
               for item in comparison["provenance_diff"])
    assert service.json_pointer_diff(True, 1)[0]["kind"] == "changed"
    assert service.json_pointer_diff(1, 1.0)[0]["kind"] == "changed"
    summarized = service.json_pointer_diff(list(range(300)), [*range(299), 999])
    assert summarized[0]["summarized"].startswith("large arrays")
    assert summarized[0]["a"]["items"] == 300
    bounded = service.json_pointer_diff(
        {str(i): 0 for i in range(2100)},
        {str(i): 1 for i in range(2100)},
    )
    assert len(bounded) == 2001
    assert bounded[-1]["kind"] == "truncated"


def test_field_planes_remain_side_by_side_when_sampling_grids_differ(
        tmp_path, monkeypatch):
    ledger = RunLedger(tmp_path / "ledger")
    canonical = _canonical(monitor="field", freqs=(193.4e12,))
    a, b = _request(ledger, canonical), _request(ledger, canonical)
    for record in (a, b):
        ledger.append_event(record["run_id"], "running")
    _write_plane_bundle(a, canonical, dl_um=0.1)
    _write_plane_bundle(b, canonical, dl_um=0.12)
    ledger.seal(a["run_id"], "completed")
    ledger.seal(b["run_id"], "completed")
    ar, br = ledger.get_run(a["run_id"]), ledger.get_run(b["run_id"])
    a_data = service.load_result(ar["output_dir"])
    b_data = service.load_result(br["output_dir"])
    monkeypatch.setattr(
        type(a_data), "__getitem__",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("comparison eligibility must not load a field binary")),
    )
    comparison = service.compare_run_data(ar, br, a_data, b_data)
    assert comparison["compatibility"]["eligible"] is True
    assert comparison["compatibility"]["shared_fields"][0]["name"] == "field"
    field = comparison["fields"][0]
    assert field["side_by_side"] is True
    assert field["grids_equal"] is False
    assert field["compatible"] is False  # subtraction remains unavailable
    assert any("no resampling" in reason for reason in field["reasons"])


def test_field_comparison_requires_shared_recorded_frequency(tmp_path):
    ledger = RunLedger(tmp_path / "ledger")
    canonical_a = _canonical(monitor="field", freqs=(193.4e12,))
    canonical_b = _canonical(monitor="field", freqs=(220e12,))
    a = _request(ledger, canonical_a)
    b = _request(ledger, canonical_b)
    for record in (a, b):
        ledger.append_event(record["run_id"], "running")
    _write_plane_bundle(a, canonical_a, dl_um=0.1)
    _write_plane_bundle(b, canonical_b, dl_um=0.1)
    manifest_path = Path(b["output_dir"], "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][0]["freqs_hz"] = [220e12]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    ledger.seal(a["run_id"], "completed")
    ledger.seal(b["run_id"], "completed")
    ar, br = ledger.get_run(a["run_id"]), ledger.get_run(b["run_id"])
    comparison = service.compare_run_data(
        ar, br, service.load_result(ar["output_dir"]),
        service.load_result(br["output_dir"]),
    )
    field = comparison["fields"][0]
    assert field["side_by_side"] is False
    assert any("no shared recorded frequency" in reason
               for reason in field["reasons"])
    assert comparison["compatibility"]["shared_fields"] == []


def test_history_and_compare_endpoints_do_not_change_current_result(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    ledger = RunLedger(tmp_path / "runs")
    ca, cb = _canonical(4.0), _canonical(5.0)
    a, b = _request(ledger, ca), _request(ledger, cb)
    for record, canonical, values in (
        (a, ca, (0.4, 0.5)), (b, cb, (0.45, 0.55)),
    ):
        ledger.append_event(record["run_id"], "running")
        _write_flux_bundle(record, canonical, values=values)
        ledger.seal(record["run_id"], "completed")

    with TestClient(create_app(run_root=tmp_path / "runs")) as client:
        history = client.get("/api/runs").json()
        assert history["root"] == str((tmp_path / "runs").resolve())
        assert len(history["runs"]) == 2
        opened = client.post(f"/api/runs/{a['run_id']}/open", json={})
        assert opened.status_code == 200
        revision = opened.json()["result_id"]
        assert opened.json()["run_id"] == a["run_id"]
        compared = client.get(
            "/api/compare", params={"a": a["run_id"], "b": b["run_id"]})
        assert compared.status_code == 200
        comparison = compared.json()
        assert comparison["compatibility"]["eligible"] is True
        assert "spectra" not in comparison and "fields" not in comparison
        assert "spec" not in comparison["a"]
        assert "manifest" not in comparison["a"]
        assert client.get("/api/session").json()["result_id"] == revision
        assert client.post("/api/workspace/from-result", json={}).status_code == 200
        sim_path = Path(a["output_dir"], "sim.json")
        original_sim = sim_path.read_bytes()
        sim_path.write_bytes(b"{}\n")
        unsafe_reuse = client.post("/api/workspace/from-result", json={})
        assert unsafe_reuse.status_code == 409
        assert "integrity" in unsafe_reuse.json()["detail"]
        sim_path.write_bytes(original_sim)
        spectrum = client.get(
            f"/api/runs/{a['run_id']}/monitor/port/spectrum")
        assert spectrum.status_code == 200
        assert spectrum.json()["data"][0]["name"] == "port"
        # Once opened, the generic result routes must retain the ledger's
        # integrity boundary rather than becoming an unchecked legacy view.
        Path(a["output_dir"], "port.bin").write_bytes(b"tampered")
        assert client.get(
            f"/api/runs/{a['run_id']}/monitor/port/meta").status_code == 200
        assert client.get("/api/monitor/port/meta").status_code == 200
        current = client.get("/api/monitor/port/spectrum")
        assert current.status_code == 409
        assert "integrity" in current.json()["detail"]


def test_run_api_rejects_nonfinite_timeout_without_creating_bundle(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    root = tmp_path / "runs"
    with TestClient(create_app(run_root=root)) as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        for value in ("nan", "inf", "-inf"):
            response = client.post(
                "/api/run",
                json={"spec": spec, "device": "cpu", "timeout_s": value},
            )
            assert response.status_code == 400
    assert list((root / "bundles").iterdir()) == []


def test_history_load_rechecks_manifest_after_parsing(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    ledger = RunLedger(tmp_path / "runs")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    ledger.seal(record["run_id"], "completed")
    original_load = service.load_result

    def load_then_replace(path):
        data = original_load(path)
        manifest_path = Path(path, "manifest.json")
        manifest_path.write_text(manifest_path.read_text() + " ")
        return data

    monkeypatch.setattr(service, "load_result", load_then_replace)
    with TestClient(create_app(run_root=tmp_path / "runs")) as client:
        opened = client.post(f"/api/runs/{record['run_id']}/open", json={})
    assert opened.status_code == 409
    assert "integrity" in opened.json()["detail"]


def test_numerical_response_is_discarded_if_artifact_changes_during_read(
        tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz import figures
    from photonhub.viz.server import create_app

    ledger = RunLedger(tmp_path / "runs")
    canonical = _canonical()
    record = _request(ledger, canonical)
    ledger.append_event(record["run_id"], "running")
    out = _write_flux_bundle(record, canonical)
    ledger.seal(record["run_id"], "completed")
    original_figure = figures.spectrum_figure

    def render_then_rewrite(data, name):
        payload = original_figure(data, name)
        artifact = out / "port.bin"
        artifact.write_bytes(artifact.read_bytes())
        return payload

    monkeypatch.setattr(figures, "spectrum_figure", render_then_rewrite)
    with TestClient(create_app(run_root=tmp_path / "runs")) as client:
        response = client.get(
            f"/api/runs/{record['run_id']}/monitor/port/spectrum")
    assert response.status_code == 409
    assert "changed during numerical read" in response.json()["detail"]


def test_run_is_ledgered_before_worker_finishes_and_terminal_before_completed(
        tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.runners import local
    from photonhub.viz.server import create_app

    def fake_run(sim, *, output_dir=None, log_file=None, **_kwargs):
        canonical = sim.to_wire_json() + "\n"
        # Read the request hash from the exact bytes the native solver would see.
        spec_sha = __import__("hashlib").sha256(canonical.encode()).hexdigest()
        out = Path(output_dir)
        out.joinpath("sim.json").write_text(canonical)
        Path(log_file).write_text(json.dumps({"event": "done"}) + "\n")
        out.joinpath("manifest.json").write_text(json.dumps({
            "manifest_version": "1", "schema_version": "1.0.0",
            "run": {"n_steps": 1, "steps_run": 1, "dt_s": 1e-16,
                    "aborted": False, "abort_reason": ""},
            "grid": {"shape": [1, 1, 1], "dl_um": 0.1},
            "provenance": {"input_sha256": spec_sha}, "monitors": [],
        }) + "\n")
        time.sleep(0.15)
        return service.load_result(out)

    monkeypatch.setattr(local, "run_local", fake_run)
    with TestClient(create_app(run_root=tmp_path / "runs")) as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        spec["monitors"] = []
        started = client.post("/api/run", json={"spec": spec, "device": "cpu"})
        assert started.status_code == 200
        run_id = started.json()["run_id"]
        early = client.get("/api/runs").json()["runs"]
        assert any(item["run_id"] == run_id for item in early)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = client.get("/api/run/status").json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed"
        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["recorded_status"] == "completed"
        assert detail["terminal"]["integrity"]["artifacts"]


def test_post_seal_activation_failure_does_not_contradict_completed_ledger(
        tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.runners import local
    from photonhub.viz.server import create_app

    def fake_run(sim, *, output_dir=None, log_file=None, **_kwargs):
        canonical = sim.to_wire_json() + "\n"
        out = Path(output_dir)
        out.joinpath("sim.json").write_text(canonical)
        Path(log_file).write_text(json.dumps({"event": "done"}) + "\n")
        out.joinpath("manifest.json").write_text(json.dumps({
            "manifest_version": "1", "schema_version": "1.0.0",
            "run": {"n_steps": 1, "steps_run": 1, "dt_s": 1e-16,
                    "aborted": False, "abort_reason": ""},
            "grid": {"shape": [1, 1, 1], "dl_um": 0.1},
            "provenance": {"input_sha256": __import__("hashlib").sha256(
                canonical.encode()).hexdigest()},
            "monitors": [],
        }) + "\n")
        return object()

    monkeypatch.setattr(local, "run_local", fake_run)
    monkeypatch.setattr(
        service, "load_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("intentional activation failure")),
    )
    with TestClient(create_app(run_root=tmp_path / "runs")) as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        spec["monitors"] = []
        run_id = client.post(
            "/api/run", json={"spec": spec, "device": "cpu"}).json()["run_id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = client.get("/api/run/status").json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed"
        assert "sealed but could not be activated" in job["error"]
        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["recorded_status"] == "completed"


def test_failed_and_cancelled_attempts_receive_terminal_records(
        tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.runners import local
    from photonhub.runners.phsolver import SolverRunError
    from photonhub.viz.server import create_app

    def wait_terminal(client):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = client.get("/api/run/status").json()
            if job["status"] not in {"queued", "running"}:
                return job
            time.sleep(0.01)
        raise AssertionError("run did not reach a terminal status")

    with TestClient(create_app(run_root=tmp_path / "runs")) as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]

        monkeypatch.setattr(
            local, "run_local",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                SolverRunError("intentional solver failure")),
        )
        failed_id = client.post(
            "/api/run", json={"spec": spec, "device": "cpu"}).json()["run_id"]
        assert wait_terminal(client)["status"] == "failed"
        failed = client.get(f"/api/runs/{failed_id}").json()
        assert failed["recorded_status"] == "failed"
        assert failed["terminal"]["error"]["message"] == "intentional solver failure"

        entered = threading.Event()

        def block_until_cancel(*_args, cancel_event=None, **_kwargs):
            entered.set()
            assert cancel_event is not None and cancel_event.wait(2)
            raise SolverRunError("cancelled by test")

        monkeypatch.setattr(local, "run_local", block_until_cancel)
        cancelled_id = client.post(
            "/api/run", json={"spec": spec, "device": "cpu"}).json()["run_id"]
        assert entered.wait(2)
        assert client.post("/api/run/cancel", json={}).status_code == 200
        assert wait_terminal(client)["status"] == "cancelled"
        cancelled = client.get(f"/api/runs/{cancelled_id}").json()
        assert cancelled["recorded_status"] == "cancelled"
        assert cancelled["terminal"]["status"] == "cancelled"


class TestPersistentRunRoot:
    """The default archive location, including the pre-rename fallback."""

    def test_defaults_to_photonhub_archive(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PHOTONHUB_RUN_ROOT", raising=False)
        monkeypatch.delenv("SIMUPOD_RUN_ROOT", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
        assert server._persistent_run_root() == (tmp_path / ".photonhub" / "runs").resolve()

    def test_adopts_legacy_simupod_archive_when_present(self, tmp_path, monkeypatch):
        # An install predating the rename keeps its run history in ~/.simupod.
        monkeypatch.delenv("PHOTONHUB_RUN_ROOT", raising=False)
        monkeypatch.delenv("SIMUPOD_RUN_ROOT", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
        (tmp_path / ".simupod" / "runs").mkdir(parents=True)
        assert server._persistent_run_root() == (tmp_path / ".simupod" / "runs").resolve()

    def test_current_archive_wins_over_legacy(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PHOTONHUB_RUN_ROOT", raising=False)
        monkeypatch.delenv("SIMUPOD_RUN_ROOT", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
        (tmp_path / ".simupod" / "runs").mkdir(parents=True)
        (tmp_path / ".photonhub" / "runs").mkdir(parents=True)
        assert server._persistent_run_root() == (tmp_path / ".photonhub" / "runs").resolve()

    def test_explicit_env_override_beats_both(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
        (tmp_path / ".simupod" / "runs").mkdir(parents=True)
        monkeypatch.setenv("PHOTONHUB_RUN_ROOT", str(tmp_path / "explicit"))
        assert server._persistent_run_root() == (tmp_path / "explicit").resolve()
