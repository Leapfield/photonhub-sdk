"""SimulationData against a hand-built manifest + binary fixture."""

import json

import numpy as np
import pytest

from photonhub.data import SimulationData

DT = 9.53e-17
DL_UM = 0.25
NX, NY, NZ = 4, 3, 2


# Fixture mirrors EXACTLY the keys engine/src/io/output.cpp emits — no
# aliases for never-shipped manifest drafts (type: field_time/field_snapshot,
# leading dim "sample", dt under run.dt_s and per-monitor dt_s).
@pytest.fixture
def out_dir(tmp_path):
    probe = np.arange(6, dtype="<f4").reshape(3, 2)          # 3 samples x (Ex,Ez)
    snap = np.arange(2 * NZ * NY * NX, dtype="<f4")          # 1 sample x (Ez,Hx)
    probe.tofile(tmp_path / "probe.bin")
    snap.tofile(tmp_path / "final.bin")
    manifest = {
        "manifest_version": "1",
        "schema_version": "1.0.0-alpha.1",
        "monitors": [
            {
                "name": "probe", "type": "field_time", "file": "probe.bin",
                "dtype": "float32", "shape": [3, 2],
                "dims": ["sample", "component"], "components": ["Ex", "Ez"],
                "sample_steps": [1, 2, 3], "dt_s": DT,
            },
            {
                "name": "final", "type": "field_snapshot", "file": "final.bin",
                "dtype": "float32", "shape": [1, 2, NZ, NY, NX],
                "dims": ["sample", "component", "z", "y", "x"],
                "components": ["Ez", "Hx"], "sample_steps": [10], "dt_s": DT,
            },
        ],
        "run": {"n_steps": 10, "dt_s": DT, "wall_seconds": 0.1,
                "mcells_per_s": 12.0, "aborted": False, "abort_reason": ""},
        "grid": {"shape": [NX, NY, NZ], "dl_um": DL_UM,
                 "size_um": [NX * DL_UM, NY * DL_UM, NZ * DL_UM]},
        "provenance": {"solver_version": "0.0.1", "device_name": "cpu_ref"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_monitor_listing(out_dir):
    data = SimulationData(out_dir)
    assert data.monitor_names == ["probe", "final"]
    assert "probe" in data and "nope" not in data
    assert len(data) == 2
    assert data.dt_s == DT


def test_time_series_dataarray(out_dir):
    da = SimulationData(out_dir)["probe"]
    assert da.dims == ("t", "component")
    assert da.shape == (3, 2)
    assert da.dtype == np.float32
    np.testing.assert_allclose(da.coords["t"].values, np.array([1, 2, 3]) * DT)
    assert list(da.coords["component"].values) == ["Ex", "Ez"]
    # x-fastest packing: data[sample * ncomp + c]
    assert float(da.sel(component="Ez").isel(t=1)) == 3.0
    assert da.attrs["kind"] == "time_series"
    assert da.attrs["provenance"]["device_name"] == "cpu_ref"


def test_snapshot_dataarray(out_dir):
    da = SimulationData(out_dir)["final"]
    assert da.dims == ("t", "component", "z", "y", "x")
    assert da.shape == (1, 2, NZ, NY, NX)
    np.testing.assert_allclose(da.coords["x"].values, np.arange(NX) * DL_UM)
    np.testing.assert_allclose(da.coords["y"].values, np.arange(NY) * DL_UM)
    np.testing.assert_allclose(da.coords["z"].values, np.arange(NZ) * DL_UM)
    np.testing.assert_allclose(da.coords["t"].values, [10 * DT])
    # binary order [sample][component][k][j][i], x fastest
    i, j, k, c = 2, 1, 1, 1
    expected = ((c * NZ + k) * NY + j) * NX + i
    assert float(da.isel(t=0, component=c, z=k, y=j, x=i)) == expected


def test_lazy_load_and_cache(out_dir):
    data = SimulationData(out_dir)
    assert data["probe"] is data["probe"]


def test_accepts_manifest_path_directly(out_dir):
    data = SimulationData(out_dir / "manifest.json")
    assert data.monitor_names == ["probe", "final"]


def test_unknown_monitor_raises_keyerror(out_dir):
    with pytest.raises(KeyError, match="available"):
        SimulationData(out_dir)["nope"]


def test_size_mismatch_raises(out_dir):
    np.zeros(5, dtype="<f4").tofile(out_dir / "probe.bin")
    with pytest.raises(ValueError, match="float32 values"):
        SimulationData(out_dir)["probe"]


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SimulationData(tmp_path / "empty")


def test_manifest_rejects_traversal_filename_before_read(out_dir):
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][0]["file"] = "../outside.bin"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe result filename"):
        SimulationData(out_dir)


def test_manifest_rejects_duplicate_monitor_names(out_dir):
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][1]["name"] = manifest["monitors"][0]["name"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate monitor name"):
        SimulationData(out_dir)


def test_manifest_rejects_ascii_case_alias_monitor_names(out_dir):
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][1]["name"] = "Probe"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate monitor name"):
        SimulationData(out_dir)


@pytest.mark.parametrize("name", ["aux", "con.txt", "bad:name", ".hidden", "µ"])
def test_manifest_rejects_nonportable_monitor_names(out_dir, name):
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][0]["name"] = name
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe name"):
        SimulationData(out_dir)


@pytest.mark.parametrize("filename", ["aux.bin", "bad:name.bin", ".hidden"])
def test_manifest_rejects_nonportable_result_filenames(out_dir, filename):
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][0]["file"] = filename
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe result filename"):
        SimulationData(out_dir)


def test_manifest_rejects_ascii_case_alias_result_filenames(out_dir):
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["monitors"][1]["file"] = "PROBE.bin"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="alias result filename"):
        SimulationData(out_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest["grid"].update({
            "coords_um": {
                "x": [0.0, 0.25, "bad", 0.75],
                "y": [0.0, 0.25, 0.5],
                "z": [0.0, 0.25],
            }}), "must contain numbers"),
        (lambda manifest: manifest["grid"].update({"shape": [5, NY, NZ]}),
         "does not match grid shape"),
        (lambda manifest: manifest["monitors"][1].update(
            {"shape": [1, 2, NZ, NY, 10**100]}), "invalid shape"),
    ],
)
def test_manifest_rejects_unreadable_snapshot_coordinates_and_shapes(
        out_dir, mutation, message):
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        SimulationData(out_dir)


def test_raw_monitor_read_rejects_symlink(out_dir):
    outside = out_dir.parent / "outside.bin"
    outside.write_bytes((out_dir / "probe.bin").read_bytes())
    (out_dir / "probe.bin").unlink()
    (out_dir / "probe.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="missing or unsafe"):
        SimulationData(out_dir)["probe"]


def test_manifest_root_and_monitor_table_must_have_expected_types(tmp_path):
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        SimulationData(tmp_path)

    (tmp_path / "manifest.json").write_text(
        json.dumps({"monitors": {"probe": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        SimulationData(tmp_path)


def test_run_block_and_healthy_flags(out_dir):
    data = SimulationData(out_dir)
    assert data.aborted is False
    assert data.abort_reason is None
    assert data.run["n_steps"] == 10
    assert data.run["wall_seconds"] == 0.1
    # NUMERICS.md section 7: healthy run, and an older manifest without the
    # steps_run key surfaces None (not a crash).
    assert data.shut_off is False
    assert data.steps_run is None


def test_shut_off_run_surfaces_steps_run(tmp_path):
    # A clean auto-shutoff finish (NOT an abort): shut_off true, steps_run < the
    # planned n_steps, no warning.
    np.arange(4, dtype="<f4").tofile(tmp_path / "probe.bin")
    manifest = {
        "manifest_version": "1",
        "monitors": [{
            "name": "probe", "type": "field_time", "file": "probe.bin",
            "dtype": "float32", "shape": [2, 2],
            "dims": ["sample", "component"], "components": ["Ex", "Ez"],
            "sample_steps": [20, 40], "dt_s": DT,
        }],
        "run": {"n_steps": 1000, "steps_run": 460, "dt_s": DT,
                "wall_seconds": 0.2, "mcells_per_s": 12.0, "aborted": False,
                "abort_reason": "", "shut_off": True},
        "grid": {"shape": [NX, NY, NZ], "dl_um": 0.05},
        "provenance": {"solver_version": "test"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    data = SimulationData(tmp_path)
    assert data.shut_off is True
    assert data.aborted is False
    assert data.steps_run == 460
    assert data.run["n_steps"] == 1000  # planned (unchanged)


def test_aborted_run_warns_and_surfaces_reason(tmp_path):
    # Aborted run per NUMERICS.md section 7: write_outputs runs BEFORE the
    # abort check in phsolver, so a complete manifest with truncated data
    # lands on disk. Loading must warn, not silently present partial fields.
    probe = np.arange(4, dtype="<f4")  # truncated: 2 samples x 2 components
    probe.tofile(tmp_path / "probe.bin")
    (tmp_path / "final.bin").write_bytes(b"")  # zero-sample snapshot
    manifest = {
        "manifest_version": "1",
        "monitors": [
            {
                "name": "probe", "type": "field_time", "file": "probe.bin",
                "dtype": "float32", "shape": [2, 2],
                "dims": ["sample", "component"], "components": ["Ex", "Ez"],
                "sample_steps": [50, 100], "dt_s": DT,
            },
            {
                # interval-0 snapshot, run aborted before the final step:
                # zero samples, empty .bin (engine writes shape[0] = 0).
                "name": "final", "type": "field_snapshot", "file": "final.bin",
                "dtype": "float32", "shape": [0, 2, NZ, NY, NX],
                "dims": ["sample", "component", "z", "y", "x"],
                "components": ["Ez", "Hx"], "sample_steps": [], "dt_s": DT,
            },
        ],
        "run": {"n_steps": 200, "dt_s": DT, "wall_seconds": 0.05,
                "mcells_per_s": 1.0, "aborted": True,
                "abort_reason": "non_finite_energy"},
        "grid": {"shape": [NX, NY, NZ], "dl_um": DL_UM},
        "provenance": {"solver_version": "0.0.1", "device_name": "cpu_ref"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.warns(UserWarning, match="non_finite_energy"):
        data = SimulationData(tmp_path)
    assert data.aborted is True
    assert data.abort_reason == "non_finite_energy"

    probe_da = data["probe"]
    assert probe_da.attrs["aborted"] is True
    assert probe_da.attrs["abort_reason"] == "non_finite_energy"
    assert probe_da.shape == (2, 2)

    # Zero-sample snapshot loads as an empty, well-shaped array.
    final_da = data["final"]
    assert final_da.shape == (0, 2, NZ, NY, NX)
    assert final_da.sizes["t"] == 0


def test_unsupported_manifest_version_rejected(out_dir):
    manifest = json.loads((out_dir / "manifest.json").read_text())
    manifest["manifest_version"] = "2"
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest_version"):
        SimulationData(out_dir)


def test_manifest_without_run_block_loads_as_healthy(out_dir):
    # Older draft manifests may lack the run block entirely; per-monitor dt_s
    # still provides the time base and the run is treated as healthy.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    del manifest["run"]
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    data = SimulationData(out_dir)
    assert data.aborted is False
    assert data.abort_reason is None
    assert data.dt_s == DT


# --- Phase 1a-1 frequency-domain monitors (NUMERICS.md section 12) ----------

# Tiny DFT region: 2 freqs x 2 components x (z=1, y=2, x=2) cells.
DFT_FREQS = [1.784e14, 1.934e14]
DFT_NF, DFT_NC, DFT_NZ, DFT_NY, DFT_NX = 2, 2, 1, 2, 2


@pytest.fixture
def freq_out_dir(tmp_path):
    """Hand-built manifest + binaries mirroring EXACTLY what output.cpp emits
    for field_dft and flux entries (dims ['freq', ..., 'complex'] / ['freq'],
    freqs_hz instead of sample_steps)."""
    n_dft = DFT_NF * DFT_NC * DFT_NZ * DFT_NY * DFT_NX
    # Sequential floats: pair p holds (2p, 2p+1) -> complex 2p + (2p+1)j.
    np.arange(2 * n_dft, dtype="<f4").tofile(tmp_path / "slab.bin")
    flux = np.array([0.5, -0.25, 0.125], dtype="<f4")
    flux.tofile(tmp_path / "reflection.bin")
    manifest = {
        "manifest_version": "1",
        "schema_version": "1.1.0-alpha.1",
        "monitors": [
            {
                "name": "slab", "type": "field_dft", "file": "slab.bin",
                "dtype": "float32",
                "shape": [DFT_NF, DFT_NC, DFT_NZ, DFT_NY, DFT_NX, 2],
                "dims": ["freq", "component", "z", "y", "x", "complex"],
                "components": ["Ex", "Hy"], "freqs_hz": DFT_FREQS,
                "origin_cells": [1, 2, 3], "dt_s": DT,
            },
            {
                "name": "reflection", "type": "flux", "file": "reflection.bin",
                "dtype": "float32", "shape": [3], "dims": ["freq"],
                "axis": "z", "freqs_hz": [1.7e14, 1.9e14, 2.1e14], "dt_s": DT,
            },
        ],
        "run": {"n_steps": 10, "dt_s": DT, "wall_seconds": 0.1,
                "mcells_per_s": 12.0, "aborted": False, "abort_reason": ""},
        # Large enough to contain the DFT fixture's non-zero origin_cells.
        "grid": {"shape": [12, 12, 12], "dl_um": DL_UM,
                 "size_um": [12 * DL_UM, 12 * DL_UM, 12 * DL_UM]},
        "provenance": {"solver_version": "0.0.1", "device_name": "cpu_ref"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


class TestFieldDft:
    def test_complex64_reconstruction(self, freq_out_dir):
        da = SimulationData(freq_out_dir)["slab"]
        assert da.dims == ("f", "component", "z", "y", "x")
        assert da.shape == (DFT_NF, DFT_NC, DFT_NZ, DFT_NY, DFT_NX)
        assert da.dtype == np.complex64
        # Binary order [freq][component][k][j][i][re,im], x fastest.
        f, c, k, j, i = 1, 0, 0, 1, 1
        p = (((f * DFT_NC + c) * DFT_NZ + k) * DFT_NY + j) * DFT_NX + i
        assert complex(da.isel(f=f, component=c, z=k, y=j, x=i)) == complex(
            2 * p, 2 * p + 1)

    def test_coords_and_attrs(self, freq_out_dir):
        da = SimulationData(freq_out_dir)["slab"]
        np.testing.assert_allclose(da.coords["f"].values, DFT_FREQS)
        assert list(da.coords["component"].values) == ["Ex", "Hy"]
        # origin_cells [i0, j0, k0] = [1, 2, 3] offsets the micron coords.
        np.testing.assert_allclose(da.coords["x"].values,
                                   (1 + np.arange(DFT_NX)) * DL_UM)
        np.testing.assert_allclose(da.coords["y"].values,
                                   (2 + np.arange(DFT_NY)) * DL_UM)
        np.testing.assert_allclose(da.coords["z"].values,
                                   (3 + np.arange(DFT_NZ)) * DL_UM)
        assert "A0*S(f)" in da.attrs["normalization"]
        assert da.attrs["freqs_hz"] == DFT_FREQS

    def test_graded_coords_um_used_for_spatial_axes(self, freq_out_dir):
        # NUMERICS.md §15.10: a graded manifest carries per-axis coords_um;
        # the DFT spatial coordinates come from those arrays (offset by
        # origin_cells), NOT from i*dl_um.
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        grid = manifest["grid"]
        # Nonlinear (graded) node coordinates; long enough to slice at the
        # fixture's origin_cells offsets (this fixture exercises the offset
        # arithmetic, not grid-consistency).
        cx = list(np.cumsum([0.0] + [DL_UM * (1.05 ** k) for k in range(8)]))
        cy = list(np.cumsum([0.0] + [DL_UM * (1.07 ** k) for k in range(8)]))
        cz = list(np.cumsum([0.0] + [DL_UM * (1.09 ** k) for k in range(8)]))
        grid["coords_um"] = {"x": cx, "y": cy, "z": cz}
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        da = SimulationData(freq_out_dir)["slab"]
        # origin_cells [i0, j0, k0] = [1, 2, 3] slices into the coord arrays.
        np.testing.assert_allclose(da.coords["x"].values, cx[1:1 + DFT_NX])
        np.testing.assert_allclose(da.coords["y"].values, cy[2:2 + DFT_NY])
        np.testing.assert_allclose(da.coords["z"].values, cz[3:3 + DFT_NZ])

    def test_interval_space_strides_uniform_coords(self, freq_out_dir):
        # §12 interval_space: a decimated DFT region reads back at strided
        # coordinates (origin + i*stride)*dl, x/y/z = strides 2/3/4 here.
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        manifest["monitors"][0]["interval_space"] = [2, 3, 4]  # (x, y, z)
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        da = SimulationData(freq_out_dir)["slab"]
        np.testing.assert_allclose(da.coords["x"].values,
                                   (1 + np.arange(DFT_NX) * 2) * DL_UM)
        np.testing.assert_allclose(da.coords["y"].values,
                                   (2 + np.arange(DFT_NY) * 3) * DL_UM)
        np.testing.assert_allclose(da.coords["z"].values,
                                   (3 + np.arange(DFT_NZ) * 4) * DL_UM)

    def test_interval_space_strides_graded_coords(self, freq_out_dir):
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        cx = list(np.cumsum([0.0] + [DL_UM * (1.05 ** k) for k in range(12)]))
        cy = list(np.cumsum([0.0] + [DL_UM * (1.07 ** k) for k in range(12)]))
        cz = list(np.cumsum([0.0] + [DL_UM * (1.09 ** k) for k in range(12)]))
        manifest["grid"]["coords_um"] = {"x": cx, "y": cy, "z": cz}
        manifest["monitors"][0]["interval_space"] = [2, 3, 1]
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        da = SimulationData(freq_out_dir)["slab"]
        np.testing.assert_allclose(da.coords["x"].values,
                                   cx[1:1 + DFT_NX * 2:2])
        np.testing.assert_allclose(da.coords["y"].values,
                                   cy[2:2 + DFT_NY * 3:3])
        np.testing.assert_allclose(da.coords["z"].values, cz[3:3 + DFT_NZ:1])

    def test_re_im_dim_alias_rejected(self, freq_out_dir):
        # data.py accepts exactly the dim names output.cpp emits — the
        # "re_im" draft alias was never shipped and is not accepted (one
        # contract, no drift).
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        manifest["monitors"][0]["dims"][-1] = "re_im"
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="dims"):
            SimulationData(freq_out_dir)["slab"]

    def test_missing_origin_defaults_to_region_local_coords(self, freq_out_dir):
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        del manifest["monitors"][0]["origin_cells"]
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        da = SimulationData(freq_out_dir)["slab"]
        np.testing.assert_allclose(da.coords["x"].values,
                                   np.arange(DFT_NX) * DL_UM)

    def test_missing_freqs_rejected(self, freq_out_dir):
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        del manifest["monitors"][0]["freqs_hz"]
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="freqs_hz"):
            SimulationData(freq_out_dir)["slab"]

    def test_bad_trailing_dim_rejected(self, freq_out_dir):
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        manifest["monitors"][0]["dims"][-1] = "parts"
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="dims"):
            SimulationData(freq_out_dir)["slab"]

    def test_freq_count_mismatch_rejected(self, freq_out_dir):
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        manifest["monitors"][0]["freqs_hz"] = [1.9e14]  # shape says 2
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="inconsistent"):
            SimulationData(freq_out_dir)["slab"]


class TestFlux:
    def test_loads_per_frequency_power(self, freq_out_dir):
        da = SimulationData(freq_out_dir)["reflection"]
        assert da.dims == ("f",)
        assert da.dtype == np.float32
        np.testing.assert_allclose(da.coords["f"].values,
                                   [1.7e14, 1.9e14, 2.1e14])
        np.testing.assert_allclose(da.values, [0.5, -0.25, 0.125])

    def test_attrs_carry_normalization_and_axis(self, freq_out_dir):
        da = SimulationData(freq_out_dir)["reflection"]
        assert "1/|A0*S(f)|^2" in da.attrs["normalization"]
        assert "NOT absolute watts" in da.attrs["normalization"]
        assert da.attrs["axis"] == "z"

    def test_missing_axis_rejected(self, freq_out_dir):
        # output.cpp emits "axis" unconditionally (and throws on a spec
        # lookup miss); a flux entry without it has an unrecoverable sign
        # convention, so the reader treats it as required.
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        entry = next(m for m in manifest["monitors"]
                     if m["name"] == "reflection")
        del entry["axis"]
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="axis"):
            SimulationData(freq_out_dir)["reflection"]

    def test_size_mismatch_rejected(self, freq_out_dir):
        np.zeros(2, dtype="<f4").tofile(freq_out_dir / "reflection.bin")
        with pytest.raises(ValueError, match="float32 values"):
            SimulationData(freq_out_dir)["reflection"]

    def test_empty_freqs_rejected(self, freq_out_dir):
        manifest = json.loads((freq_out_dir / "manifest.json").read_text())
        manifest["monitors"][1]["freqs_hz"] = []
        (freq_out_dir / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="freqs_hz"):
            SimulationData(freq_out_dir)["reflection"]
