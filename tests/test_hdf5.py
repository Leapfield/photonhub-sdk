"""HDF5 packing (photonhub.hdf5) and the SimulationData HDF5 backend.

The contract: loading the .h5 is bit-identical to loading the raw output
directory, because the .h5 carries the same manifest and the same raw monitor
bytes and reuses the same reconstruction path. So the core tests convert a
hand-built raw directory and assert per-monitor xarray identity.
"""

import json
import os

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("h5py")

from photonhub.data import SimulationData

from photonhub import convert_to_hdf5

DT = 9.53e-17
DL_UM = 0.25
NX, NY, NZ = 4, 3, 2


@pytest.fixture
def full_dir(tmp_path):
    """A raw output directory exercising every monitor kind — time series,
    a complex field_dft, and flux — so the complex64 view and the flux/dft
    metadata are covered by the round trip."""
    np.arange(6, dtype="<f4").tofile(tmp_path / "probe.bin")    # 3 x (Ex,Ez)
    # field_dft: [freq, comp, z, y, x, 2] = [2,1,1,1,1,2] -> 4 floats = 2
    # complex (one per freq).
    np.arange(4, dtype="<f4").tofile(tmp_path / "slab.bin")
    np.array([0.5, -0.25], dtype="<f4").tofile(tmp_path / "flux.bin")  # 2 freqs
    manifest = {
        "manifest_version": "1",
        "schema_version": "1.7.0-alpha.1",
        "monitors": [
            {"name": "probe", "type": "field_time", "file": "probe.bin",
             "dtype": "float32", "shape": [3, 2],
             "dims": ["sample", "component"], "components": ["Ex", "Ez"],
             "sample_steps": [1, 2, 3], "dt_s": DT},
            {"name": "slab", "type": "field_dft", "file": "slab.bin",
             "dtype": "float32", "shape": [2, 1, 1, 1, 1, 2],
             "dims": ["freq", "component", "z", "y", "x", "complex"],
             "components": ["Ex"], "freqs_hz": [1.0e14, 2.0e14],
             "origin_cells": [1, 1, 0], "dt_s": DT},
            {"name": "flux", "type": "flux", "file": "flux.bin",
             "dtype": "float32", "shape": [2], "dims": ["freq"], "axis": "z",
             "freqs_hz": [1.0e14, 2.0e14], "dt_s": DT},
        ],
        "run": {"n_steps": 50, "steps_run": 30, "dt_s": DT, "wall_seconds": 0.1,
                "mcells_per_s": 12.0, "aborted": False, "abort_reason": "",
                "shut_off": True},
        "grid": {"shape": [NX, NY, NZ], "dl_um": DL_UM,
                 "size_um": [NX * DL_UM, NY * DL_UM, NZ * DL_UM]},
        "provenance": {"solver_version": "0.0.1", "device_name": "cpu_ref"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


class TestRoundTrip:
    def test_default_dest_and_existence(self, full_dir):
        out = convert_to_hdf5(full_dir)
        assert out == full_dir / "simulation.h5"
        assert out.is_file()

    def test_each_monitor_loads_identically(self, full_dir):
        h5 = convert_to_hdf5(full_dir, full_dir / "sim.h5")
        from_dir = SimulationData(full_dir)
        from_h5 = SimulationData(h5)
        assert from_h5.monitor_names == from_dir.monitor_names
        for name in from_dir.monitor_names:
            xr.testing.assert_identical(from_h5[name], from_dir[name])

    def test_complex_dft_survives(self, full_dir):
        h5 = convert_to_hdf5(full_dir, full_dir / "sim.h5")
        da = SimulationData(h5)["slab"]
        assert da.dtype == np.complex64
        # raw [0,1,2,3,4,5,6,7] -> complex (0+1j),(2+3j),(4+5j),(6+7j) before
        # the section-12 normalization; identity vs the dir load already pins
        # the values, so here just confirm the complex reconstruction ran.
        assert da.dims == ("f", "component", "z", "y", "x")
        assert da.sizes["f"] == 2

    def test_run_metadata_preserved(self, full_dir):
        h5 = convert_to_hdf5(full_dir, full_dir / "sim.h5")
        d = SimulationData(h5)
        assert d.dt_s == DT
        assert d.aborted is False
        assert d.shut_off is True
        assert d.steps_run == 30
        assert d.run["n_steps"] == 50
        assert d.provenance["device_name"] == "cpu_ref"


class TestAutoDetect:
    def test_loads_h5_file_path(self, full_dir):
        h5 = convert_to_hdf5(full_dir, full_dir / "sim.h5")
        assert "probe" in SimulationData(h5)

    def test_loads_directory_holding_only_h5(self, tmp_path, full_dir):
        # A directory with the .h5 but no manifest.json still resolves.
        h5 = convert_to_hdf5(full_dir, tmp_path / "only" / "simulation.h5")
        assert "flux" in SimulationData(h5.parent)


class TestErrors:
    def test_convert_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert_to_hdf5(tmp_path)

    def test_convert_rejects_traversal_before_creating_destination(self, full_dir):
        # Make the traversal target real: without preflight validation the
        # converter would read this parent-directory blob into the HDF5 file.
        raw_dir = full_dir / "raw"
        raw_dir.mkdir()
        manifest = json.loads((full_dir / "manifest.json").read_text())
        manifest["monitors"] = [manifest["monitors"][0]]
        manifest["monitors"][0]["file"] = "../probe.bin"
        (raw_dir / "manifest.json").write_text(json.dumps(manifest))
        dest = raw_dir / "packed" / "simulation.h5"

        with pytest.raises(ValueError, match="unsafe result filename"):
            convert_to_hdf5(raw_dir, dest)

        assert not dest.parent.exists()

    def test_convert_rejects_hdf5_path_name_before_creating_destination(
            self, full_dir):
        manifest_path = full_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["monitors"][0]["name"] = "nested/probe"
        manifest_path.write_text(json.dumps(manifest))
        dest = full_dir / "packed" / "simulation.h5"

        with pytest.raises(ValueError, match="unsafe name"):
            convert_to_hdf5(full_dir, dest)

        assert not dest.parent.exists()

    def test_convert_rejects_manifest_as_destination_without_replacing_it(
            self, full_dir):
        manifest_path = full_dir / "manifest.json"
        original = manifest_path.read_bytes()

        with pytest.raises(ValueError, match="aliases the manifest source"):
            convert_to_hdf5(full_dir, manifest_path)

        assert manifest_path.read_bytes() == original

    def test_convert_rejects_monitor_blob_as_destination_without_replacing_it(
            self, full_dir):
        blob_path = full_dir / "probe.bin"
        original = blob_path.read_bytes()

        with pytest.raises(ValueError, match="aliases the monitor 'probe' source"):
            convert_to_hdf5(full_dir, blob_path)

        assert blob_path.read_bytes() == original

    def test_convert_preflights_every_blob_before_creating_destination(
            self, full_dir):
        (full_dir / "slab.bin").unlink()
        dest = full_dir / "packed" / "simulation.h5"

        with pytest.raises(FileNotFoundError, match="result file not found"):
            convert_to_hdf5(full_dir, dest)

        assert not dest.parent.exists()

    def test_convert_rejects_symlink_blob_before_creating_destination(
            self, full_dir):
        outside = full_dir.parent / f"{full_dir.name}-outside.bin"
        outside.write_bytes((full_dir / "probe.bin").read_bytes())
        (full_dir / "probe.bin").unlink()
        try:
            (full_dir / "probe.bin").symlink_to(outside)
        except OSError:
            pytest.skip("symlinks are unavailable")
        dest = full_dir / "packed" / "simulation.h5"

        with pytest.raises(ValueError, match="regular non-symlink"):
            convert_to_hdf5(full_dir, dest)

        assert not dest.parent.exists()

    def test_bad_blob_size_does_not_replace_existing_destination(self, full_dir):
        np.zeros(3, dtype="<f4").tofile(full_dir / "slab.bin")
        dest = full_dir / "simulation.h5"
        original = b"existing destination"
        dest.write_bytes(original)

        with pytest.raises(ValueError, match="manifest shape requires 16"):
            convert_to_hdf5(full_dir, dest)

        assert dest.read_bytes() == original

    def test_write_failure_keeps_existing_destination_and_removes_temp(
            self, full_dir, monkeypatch):
        import photonhub.hdf5 as hdf5_module

        real_fromfile = hdf5_module.np.fromfile
        calls = 0

        def fail_second_read(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected read failure")
            return real_fromfile(*args, **kwargs)

        monkeypatch.setattr(hdf5_module.np, "fromfile", fail_second_read)
        dest = full_dir / "simulation.h5"
        original = b"existing destination"
        dest.write_bytes(original)

        with pytest.raises(RuntimeError, match="injected read failure"):
            convert_to_hdf5(full_dir, dest)

        assert dest.read_bytes() == original
        assert list(full_dir.glob(".photonhub-h5-*.tmp")) == []

    def test_temp_descriptor_is_closed_when_fdopen_fails(
            self, full_dir, monkeypatch):
        import photonhub.hdf5 as hdf5_module

        real_fdopen = hdf5_module.os.fdopen
        real_mkstemp = hdf5_module.tempfile.mkstemp
        created = {}

        def recording_mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            created.update(fd=fd, path=name)
            return fd, name

        def fail_output_fdopen(fd, mode="r", *args, **kwargs):
            if mode == "w+b":
                raise RuntimeError("injected fdopen failure")
            return real_fdopen(fd, mode, *args, **kwargs)

        monkeypatch.setattr(hdf5_module.tempfile, "mkstemp", recording_mkstemp)
        monkeypatch.setattr(hdf5_module.os, "fdopen", fail_output_fdopen)
        dest = full_dir / "simulation.h5"
        original = b"existing destination"
        dest.write_bytes(original)

        with pytest.raises(RuntimeError, match="injected fdopen failure"):
            convert_to_hdf5(full_dir, dest)

        assert dest.read_bytes() == original
        assert not os.path.exists(created["path"])
        with pytest.raises(OSError):
            os.fstat(created["fd"])

    def test_non_photonhub_h5_raises(self, tmp_path):
        import h5py
        bad = tmp_path / "bad.h5"
        with h5py.File(bad, "w") as f:
            f.create_dataset("x", data=np.zeros(3))
        with pytest.raises(ValueError, match="not a PhotonHub HDF5 file"):
            SimulationData(bad)

    def test_unknown_format_raises(self, tmp_path):
        import h5py
        bad = tmp_path / "future.h5"
        with h5py.File(bad, "w") as f:
            f.attrs["format"] = "photonhub-hdf5-99"
            f.attrs["manifest_json"] = "{}"
        with pytest.raises(ValueError, match="unsupported HDF5 format"):
            SimulationData(bad)


def _real_solver():
    import photonhub as ph
    try:
        return ph.find_solver()
    except ph.SolverRunError:
        return None


@pytest.mark.skipif(_real_solver() is None,
                    reason="no phsolver binary found (build the engine first)")
def test_integration_real_solver_roundtrip(tmp_path):
    # Real engine output (time + complex DFT + flux) packs to HDF5 and reloads
    # bit-identically — the end-to-end contract, not just hand-built manifests.
    import photonhub as ph

    sim = ph.Simulation(
        size_um=(0.4, 0.4, 1.2),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=80),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
        pml_num_layers=6,
        sources=[ph.PointDipole(
            center_um=(0.2, 0.2, 0.6), polarization="Ex",
            source_time=ph.GaussianPulse(freq0_hz=2e14, fwidth_hz=1e14))],
        monitors=[
            ph.FieldTimeMonitor(name="probe", center_um=(0.2, 0.2, 0.7),
                                fields=["Ex"]),
            ph.FieldDftMonitor(name="slab", center_um=(0.2, 0.2, 0.7),
                               size_um=(0.4, 0.4, 0.0), fields=["Ex", "Hy"],
                               freqs_hz=[2e14]),
            ph.FluxMonitor(name="flux", axis="z", position_um=0.7,
                           freqs_hz=[2e14])],
    )
    data = ph.run_local(sim, output_dir=tmp_path / "out", timeout=300)
    h5 = convert_to_hdf5(tmp_path / "out")
    from_h5 = SimulationData(h5)
    assert from_h5.monitor_names == data.monitor_names
    for name in data.monitor_names:
        xr.testing.assert_identical(from_h5[name], data[name])


@pytest.mark.skipif(_real_solver() is None,
                    reason="no phsolver binary found (build the engine first)")
def test_integration_graded_grid_coords_survive_hdf5(tmp_path):
    # Graded axes ride the manifest as coords_um (NUMERICS.md section 15.10) —
    # a distinct coordinate-reconstruction path from uniform i*dl — and must
    # come back identically from the raw directory and the packed HDF5.
    import photonhub as ph

    z = (0.0, 0.05, 0.1, 0.15, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.65,
         0.7, 0.75, 0.8)
    sim = ph.Simulation(
        size_um=(0.4, 0.4, 0.8),
        grid=ph.GradedGridSpec(dl_um=0.05,
                               coords=ph.GradedAxisCoords(z=z)),
        run=ph.RunSpec(n_steps=20),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
        pml_num_layers=4,
        sources=[ph.PointDipole(
            center_um=(0.2, 0.2, 0.4), polarization="Ex",
            source_time=ph.GaussianPulse(freq0_hz=2e14, fwidth_hz=1e14))],
        monitors=[ph.FieldSnapshotMonitor(name="snap", fields=["Ex"],
                                          interval_steps=20)],
    )
    data = ph.run_local(sim, output_dir=tmp_path / "out", timeout=300)
    got_z = np.asarray(data["snap"].coords["z"].values)
    np.testing.assert_allclose(got_z[:len(z)], np.asarray(z))

    from_h5 = SimulationData(convert_to_hdf5(tmp_path / "out"))
    xr.testing.assert_identical(from_h5["snap"], data["snap"])


def test_aborted_run_h5_warns_on_load(tmp_path):
    # Conversion preserves the aborted flag; loading the .h5 warns just like
    # loading the raw directory would.
    np.arange(4, dtype="<f4").tofile(tmp_path / "probe.bin")
    manifest = {
        "manifest_version": "1",
        "monitors": [{"name": "probe", "type": "field_time", "file": "probe.bin",
                      "dtype": "float32", "shape": [2, 2],
                      "dims": ["sample", "component"], "components": ["Ex", "Ez"],
                      "sample_steps": [1, 2], "dt_s": DT}],
        "run": {"n_steps": 10, "dt_s": DT, "wall_seconds": 0.1,
                "mcells_per_s": 1.0, "aborted": True,
                "abort_reason": "non_finite_energy"},
        "grid": {"shape": [NX, NY, NZ], "dl_um": DL_UM},
        "provenance": {"solver_version": "t"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    h5 = convert_to_hdf5(tmp_path)
    with pytest.warns(UserWarning, match="ABORTED"):
        d = SimulationData(h5)
    assert d.aborted is True
    assert d.abort_reason == "non_finite_energy"
