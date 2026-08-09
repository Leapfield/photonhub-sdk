"""Desktop workbench HTTP/data correctness contracts.

These tests cover the seam the React app consumes.  The established viz tests
exercise matplotlib/Jupyter presentation; they do not catch stale result
sessions, structured authoring failures, or a wrongly inferred DFT plane.
"""

import json
import os
import threading
import time
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from photonhub.runners.phsolver import SolverRunError, run_phsolver
from photonhub.data import SimulationData
from photonhub.viz import service
from photonhub.viz.ledger import RunLedger


class _PlaneData:
    """Small in-memory SimulationData stand-in for service slicing tests."""

    def __init__(self):
        self.manifest = {
            "monitors": [{
                "name": "port", "type": "field_dft",
                "dims": ["f", "component", "z", "y", "x"],
                "shape": [1, 1, 3, 4, 1], "components": ["Ey"],
            }]
        }
        self.da = xr.DataArray(
            np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4, 1),
            dims=("f", "component", "z", "y", "x"),
            coords={
                "f": [193.4e12], "component": ["Ey"],
                "z": [0.1, 0.2, 0.3], "y": [0.4, 0.5, 0.6, 0.7], "x": [1.25],
            },
        )

    def __getitem__(self, name):
        assert name == "port"
        return self.da


def test_singleton_normal_dft_plane_is_sliced_on_that_axis():
    data = _PlaneData()
    meta = service.meta(data, "port")
    assert meta["volumetric"] is False
    assert meta["plane"] == {"axis": "x", "pos": 1.25}
    plane, values, row, col, resolved = service.slice_plane(
        data, "port", field="Ey", val="real")
    assert (row, col) == ("z", "y")
    assert values.shape == (3, 4)
    assert plane.shape == (3, 4)
    assert resolved["cut"] == {"axis": "x", "value_um": 1.25}


def test_rank1_profile_and_rank0_field_spectrum_are_not_heatmaps():
    class RankData:
        manifest = {"monitors": [
            {"name": "line", "type": "field_dft",
             "dims": ["freq", "component", "z", "y", "x", "complex"],
             "shape": [2, 1, 1, 4, 1, 2], "components": ["Ey"]},
            {"name": "point", "type": "field_dft",
             "dims": ["freq", "component", "z", "y", "x", "complex"],
             "shape": [2, 1, 1, 1, 1, 2], "components": ["Ey"]},
        ]}

        def __init__(self):
            coords = {"f": [190e12, 200e12], "component": ["Ey"],
                      "z": [0.2], "x": [0.3]}
            self.arrays = {
                "line": xr.DataArray(
                    np.arange(8, dtype=np.float32).reshape(2, 1, 1, 4, 1).astype(np.complex64),
                    dims=("f", "component", "z", "y", "x"),
                    coords={**coords, "y": [0.0, 0.1, 0.2, 0.3]}),
                "point": xr.DataArray(
                    np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64).reshape(2, 1, 1, 1, 1),
                    dims=("f", "component", "z", "y", "x"),
                    coords={**coords, "y": [0.1]}),
            }

        def __getitem__(self, name):
            return self.arrays[name]

    data = RankData()
    catalog = service.monitor_catalog(data)
    assert [item["kind"] for item in catalog] == ["profile", "field_spectrum"]
    line = service.line_profile_values(data, "line", field="Ey", val="real")
    assert line["axis"] == "y"
    assert len(line["value"]) == 4
    spectrum = service.field_spectrum_values(data, "point", field="Ey", val="abs")
    assert spectrum["value"] == pytest.approx([5.0, np.sqrt(5.0)])


def _client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app
    return TestClient(create_app())


def test_packaged_cpu_example_is_fresnel_slab():
    """The CPU-runnable packaged example: the gallery's Fresnel-slab scene,
    small enough for seconds-scale local runs, listed before the GPU scene."""
    resource = resources.files("photonhub.viz").joinpath(
        "examples", "fresnel_slab.sim.json")
    assert resource.is_file()

    packaged = json.loads(resource.read_text(encoding="utf-8"))
    sim, starter = service.example_sim("fresnel-slab-tmm-cpu")
    assert sim.to_wire_dict() == packaged

    estimate = sim.cost_estimate()
    assert estimate.cells_per_axis == (4, 4, 320)
    assert estimate.num_cells < 10_000            # stays a seconds-scale run
    assert (sim.boundaries.x, sim.boundaries.y, sim.boundaries.z) == (
        "periodic", "periodic", "pml")
    assert [s.geometry.type for s in sim.structures] == ["box"]
    assert sim.structures[0].medium.permittivity == pytest.approx(4.0)
    assert [src.type for src in sim.sources] == ["plane_wave"]
    assert [m.name for m in sim.monitors] == ["up", "down"]
    assert all(m.type == "flux" and len(m.freqs_hz) == 11
               for m in sim.monitors)

    assert starter["id"] == "fresnel-slab-tmm-cpu"
    assert starter["profile"].startswith("CPU example")
    assert "0.6446" in starter["reference"]       # the analytic TMM anchor
    starter["title"] = "changed"
    assert service.example_starters()[0]["title"] == (
        "Fresnel slab vs analytic theory")


def test_packaged_gpu_example_is_mode_converter():
    resource = resources.files("photonhub.viz").joinpath(
        "examples", "mode_converter.sim.json")
    assert resource.is_file()

    packaged = json.loads(resource.read_text(encoding="utf-8"))
    sim, starter = service.example_sim("gds-mode-converter-matched-res10")
    assert sim.to_wire_dict() == packaged

    assert [s.geometry.type for s in sim.structures].count("box") == 4
    assert [s.geometry.type for s in sim.structures].count("polyslab") == 8
    assert len(sim.structures) == 12

    assert len(sim.sources) == 1
    source = sim.sources[0]
    assert source.type == "mode_source"
    assert (source.axis, source.direction) == ("x", "+")
    assert (source.polarization, source.minor_polarization) == ("Ey", "Ez")
    assert source.profile_minor is not None
    assert source.profile_h is not None
    assert source.mode_solve is not None
    assert source.mode_solve.mode_index == 0
    assert source.mode_solve.input_sha256 == service.mode_source_input_sha256(
        sim, 0)
    assert service.mode_source_statuses(sim)[0]["status"] == "fresh"
    assert source.profile_h_minor is not None
    assert {
        len(source.profile), len(source.profile_minor),
        len(source.profile_h), len(source.profile_h_minor),
    } == {source.nu * source.nv}

    assert sim.grid.type == "uniform"
    assert sim.grid.dl_um == pytest.approx(1.55 / (3.4738 * 10))
    assert sim.cost_estimate().cells_per_axis == (1638, 239, 69)
    assert sim.background.permittivity == pytest.approx(1.444 ** 2)
    assert sim.subpixel is True
    assert sim.subpixel_method == "tensor"
    assert (sim.boundaries.x, sim.boundaries.y, sim.boundaries.z) == (
        "pml", "pml", "pml")

    expected_monitors = ["o1", "o2", "o3", "o4", "field_z0"]
    assert [m.name for m in sim.monitors] == expected_monitors
    assert all(m.type == "field_dft" for m in sim.monitors)
    assert all(len(m.freqs_hz) == 101 for m in sim.monitors[:-1])
    wavelengths_nm = [299_792_458.0 / f * 1e9 for f in sim.monitors[0].freqs_hz]
    assert min(wavelengths_nm) == pytest.approx(1500.0)
    assert max(wavelengths_nm) == pytest.approx(1600.0)
    ports = [m.mode_port for m in sim.monitors[:-1]]
    assert all(port is not None for port in ports)
    assert [port.out_direction for port in ports] == ["-", "-", "+", "+"]
    assert [port.source_index for port in ports] == [0, None, None, None]
    assert [port.center_um for port in ports] == pytest.approx([
        (8.893436697564628, 1.5354366975646268),
        (2.135436697564628, 1.5354366975646268),
        (2.135436697564628, 1.5354366975646268),
        (8.893436697564628, 1.5354366975646268),
    ])
    assert [
        [(mode.polarization, mode.mode_index) for mode in port.modes]
        for port in ports
    ] == [[("TE", 0)], [("TE", 0)], [("TE", 0), ("TE", 1)], [("TE", 0)]]

    assert starter == {
        "id": "gds-mode-converter-matched-res10",
        "title": "TE0→TE1 GDS mode converter",
        "profile": "GPU example · matched res10",
        "description": (
            "Exact gdsfactory generic-PDK mode-converter geometry: a 0.5 µm TE0 "
            "input couples across a 0.15 µm gap into the TE1 mode of a 1.2 µm bus."
        ),
        "provenance": (
            "JPPhotonics/fdtd-pipeline@622e0a9 · Liu & Poon, arXiv:2506.16665"
        ),
        "fidelity_note": (
            "Uses the benchmark's matched res10 materials, uniform grid, 1500–1600 nm "
            "sweep, true-H guided-mode profile, and port planes. To keep the editor "
            "interactive it uses one auxiliary ModeSource instead of thousands of "
            "equivalence-current dipoles. Four physical modal ports are ready in "
            "Design and Results; o3 resolves both TE0 and TE1 from one raw DFT plane."
        ),
        "reference": (
            "Matched res25 headline at 1550 nm: PhotonHub 45.74% vs Tidy3D 45.69% "
            "TE0→TE1 conversion (+0.05 percentage points)."
        ),
    }
    starter["title"] = "changed"
    assert service.example_starters()[1]["title"] == "TE0→TE1 GDS mode converter"


def test_packaged_mode_converter_o3_recipe_resolves_two_guided_te_modes():
    from photonhub.plugins.yee_mode import solve_yee_port_mode_bank

    sim = service.example_sim("gds-mode-converter-matched-res10")[0]
    monitor = sim.monitors[2]
    port = monitor.mode_port
    assert port is not None
    frequency = min(
        monitor.freqs_hz,
        key=lambda value: abs(299_792_458.0 / value * 1e6 - 1.55),
    )
    bank = solve_yee_port_mode_bank(
        sim,
        "x",
        monitor.center_um[0],
        (frequency,),
        modes=tuple(
            (mode.polarization, mode.mode_index) for mode in port.modes
        ),
        h_center_um=port.center_um[0],
        v_center_um=port.center_um[1],
        half_w_um=port.size_um[0] / 2,
        half_v_um=port.size_um[1] / 2,
        dl_um=port.dl_um,
        supersample=port.supersample,
        num_modes=port.num_modes,
    )
    modes = bank[frequency]
    assert modes[("TE", 0)].n_eff > modes[("TE", 1)].n_eff > 1.444


def test_packaged_mode_converter_recipe_reproduces_committed_mode_profile():
    sim = service.example_sim("gds-mode-converter-matched-res10")[0]
    original = sim.sources[0]
    recipe = original.mode_solve
    assert recipe is not None
    settings = {
        "axis": original.axis,
        "position_um": original.position_um,
        "direction": original.direction,
        "polarization": recipe.polarization,
        "mode_index": recipe.mode_index,
        "wavelength_um": recipe.wavelength_um,
        "center_um": recipe.center_um,
        "size_um": recipe.size_um,
        "dl_um": recipe.dl_um,
        "supersample": recipe.supersample,
        "num_modes": recipe.num_modes,
        "num_freqs": recipe.num_freqs,
    }

    resolved, summary = service.solve_mode_source(sim, 0, settings)
    actual = resolved.sources[0]
    assert summary["n_eff"] == pytest.approx(
        2.4163079263991527, abs=1e-12)
    assert summary["profile_shape"] == [69, 239]
    assert summary["mode_window_shape"] == [33, 39]
    assert actual.amplitude == pytest.approx(1.0)
    for field in ("profile", "profile_minor", "profile_h", "profile_h_minor"):
        before = np.asarray(getattr(original, field), dtype=float)
        after = np.asarray(getattr(actual, field), dtype=float)
        assert np.max(np.abs(after - before)) < 5e-7
    assert service.mode_source_statuses(resolved)[0]["status"] == "fresh"


def test_default_workbench_starter_is_cpu_friendly():
    sim = service.default_sim()
    estimate = sim.cost_estimate()
    assert estimate.cells_per_axis == (80, 40, 24)
    assert estimate.num_steps == 1000
    assert estimate.num_cells < 100_000
    assert [monitor.name for monitor in sim.monitors] == [
        "field", "output_flux", "probe",
    ]
    starter = service.default_starter()
    assert starter["id"] == "soi-waveguide-cpu-quickstart"
    assert "CPU" in starter["title"]
    starter["title"] = "changed"
    assert service.default_starter()["title"] == "SOI waveguide CPU quickstart"
    assert [item["id"] for item in service.example_starters()] == [
        "fresnel-slab-tmm-cpu",
        "gds-mode-converter-matched-res10",
    ]


def test_workbench_lists_and_loads_packaged_gpu_example():
    with _client() as client:
        listed = client.get("/api/workspace/examples")
        assert listed.status_code == 200
        assert listed.json() == {"examples": service.example_starters()}

        loaded = client.post("/api/workspace/example", json={
            "id": "gds-mode-converter-matched-res10",
        })
        assert loaded.status_code == 200
        payload = loaded.json()
        assert payload["dirty"] is True
        assert payload["path"] == ""
        assert payload["starter"] == next(
            s for s in service.example_starters()
            if s["id"] == "gds-mode-converter-matched-res10")
        assert payload["estimate"]["cells_per_axis"] == [1638, 239, 69]
        assert payload["spec"]["sources"][0]["type"] == "mode_source"

        cpu = client.post("/api/workspace/example", json={
            "id": "fresnel-slab-tmm-cpu",
        })
        assert cpu.status_code == 200
        cpu_payload = cpu.json()
        assert cpu_payload["starter"] == next(
            s for s in service.example_starters()
            if s["id"] == "fresnel-slab-tmm-cpu")
        assert cpu_payload["estimate"]["cells_per_axis"] == [4, 4, 320]
        assert cpu_payload["spec"]["sources"][0]["type"] == "plane_wave"

        missing = client.post("/api/workspace/example", json={"id": "missing"})
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "example_not_found"

        invalid = client.post("/api/workspace/example", json={"id": ""})
        assert invalid.status_code == 422


def test_packaged_gpu_example_preserves_canonical_gds_polygons():
    pytest.importorskip("gdstk")
    from photonhub import GdsLayer, Medium, import_gds, read_gds_cell_names

    root = Path(__file__).resolve().parents[2]
    gds = (root / "benchmarks" / "gds" / "test_cases" / "mode_converter" /
           "gds" / "mode_converter.gds")
    cell = "mode_converter_gap0p15_length20"
    assert cell in read_gds_cell_names(gds)
    imported = import_gds(
        gds,
        [GdsLayer(
            layer=(1, 0), medium=Medium(permittivity=3.4738 ** 2),
            zmin_um=0.0, thickness_um=0.22,
        )],
        cell_name=cell,
    )
    starter = [
        s for s in service.example_sim("gds-mode-converter-matched-res10")[0].structures
        if s.geometry.type == "polyslab"
    ]
    assert len(starter) == len(imported) == 8

    xy_shift = (np.asarray(starter[0].geometry.vertices_um)
                - np.asarray(imported[0].geometry.vertices_um))[0]
    z_shift = starter[0].geometry.slab_bounds_um[0]
    for actual, original in zip(starter, imported):
        actual_vertices = np.asarray(actual.geometry.vertices_um)
        original_vertices = np.asarray(original.geometry.vertices_um)
        assert actual_vertices.shape == original_vertices.shape
        assert np.allclose(actual_vertices - original_vertices, xy_shift)
        assert np.asarray(actual.geometry.slab_bounds_um) == pytest.approx(
            np.asarray(original.geometry.slab_bounds_um) + z_shift)
        assert actual.medium.permittivity == pytest.approx(original.medium.permittivity)


def test_packaged_gpu_example_scene_includes_mode_source_plane():
    sim = service.example_sim("gds-mode-converter-matched-res10")[0]
    figure = sim.plot_3d()

    # 12 structures + one source plane + five monitor handles/planes + domain.
    assert len(figure.data) == 19
    trace_names = [trace.name for trace in figure.data]
    assert trace_names.index("domain") < trace_names.index("source0")
    assert trace_names.index("domain") < trace_names.index("monitor:o3")
    source = next(trace for trace in figure.data if trace.name == "source0")
    assert source.type == "mesh3d"
    assert source.visible in (None, True)
    assert 0.0 < source.opacity <= 1.0
    assert min(source.x) < sim.sources[0].position_um < max(source.x)
    output = next(trace for trace in figure.data if trace.name == "monitor:o3")
    assert max(output.x) - min(output.x) > 0.5
    assert output.meta["photonhub"] == {"kind": "port", "id": "o3", "index": 2}
    assert output.uid.startswith("photonhub-port-")
    field_z0 = [trace for trace in figure.data if trace.name == "monitor:field_z0"]
    assert [trace.type for trace in field_z0] == ["scatter3d"]
    assert field_z0[0].mode == "lines+markers"
    assert field_z0[0].showlegend is False
    assert figure.layout.scene.camera.eye.x > 1.25


def test_packaged_gpu_example_scene_preserves_concave_bend_caps():
    sim = service.example_sim("gds-mode-converter-matched-res10")[0]
    traces = {trace.name: trace for trace in sim.plot_3d().data}

    def signed_twice_area(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    def polygon_contains(point, polygon):
        x, y = point
        inside = False
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            if (a[1] > y) == (b[1] > y):
                continue
            crossing_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < crossing_x:
                inside = not inside
        return inside

    # These four exact GDS polygons are the sampled Euler bends. Each is
    # concave, so a vertex-0 triangle fan visibly bridges its inner curvature.
    for structure_index in (8, 9, 10, 11):
        geometry = sim.structures[structure_index].geometry
        polygon = [tuple(vertex) for vertex in geometry.vertices_um]
        assert len(polygon) == 118

        mesh = traces[f"structure{structure_index}"]
        n = len(polygon)
        rendered_ring = np.column_stack((mesh.x[:n], mesh.y[:n]))
        assert rendered_ring == pytest.approx(np.asarray(polygon))
        assert len(mesh.x) == 2 * n

        bottom = [
            (a, b, c) for a, b, c in zip(mesh.i, mesh.j, mesh.k)
            if max(a, b, c) < n
        ]
        polygon_area = 0.5 * abs(sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(polygon, polygon[1:] + polygon[:1])
        ))
        cap_area = 0.5 * sum(
            abs(signed_twice_area(polygon[a], polygon[b], polygon[c]))
            for a, b, c in bottom
        )
        assert cap_area == pytest.approx(polygon_area, rel=1e-10)

        for a, b, c in bottom:
            centroid = tuple(
                (polygon[a][coordinate] + polygon[b][coordinate]
                 + polygon[c][coordinate]) / 3.0
                for coordinate in (0, 1)
            )
            assert polygon_contains(centroid, polygon)


def test_packaged_gpu_example_box_and_polyslab_tops_share_outward_normal():
    sim = service.example_sim("gds-mode-converter-matched-res10")[0]
    traces = {trace.name: trace for trace in sim.plot_3d().data}

    for structure_index, structure in enumerate(sim.structures):
        mesh = traces[f"structure{structure_index}"]
        points = np.column_stack((mesh.x, mesh.y, mesh.z)).astype(float)
        triangles = np.column_stack((mesh.i, mesh.j, mesh.k)).astype(int)
        normals = np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        )

        if structure.geometry.type == "box":
            top_z = max(mesh.z)
            top = np.all(
                np.isclose(points[triangles, 2], top_z), axis=1)
        else:
            assert structure.geometry.type == "polyslab"
            assert structure.geometry.axis == "z"
            n = len(structure.geometry.vertices_um)
            top = np.all(triangles >= n, axis=1)

        assert np.any(top)
        assert np.all(normals[top, 2] > 1e-12)
        assert np.allclose(normals[top, :2], 0.0, atol=1e-12)


def test_packaged_gpu_example_structures_share_solid_material_style():
    sim = service.example_sim("gds-mode-converter-matched-res10")[0]
    figure = sim.plot_3d()
    structures = [
        trace for trace in figure.data
        if trace.name.startswith("structure")
    ]
    assert len(structures) == 12

    expected_lighting = {
        "ambient": 0.95,
        "diffuse": 0.15,
        "specular": 0.0,
        "fresnel": 0.0,
        "roughness": 1.0,
    }
    assert {trace.color for trace in structures} == {"rgb(253,231,36)"}
    assert {trace.opacity for trace in structures} == {1.0}
    assert all(
        trace.lighting.to_plotly_json() == expected_lighting
        for trace in structures
    )

    overlays = [
        trace for trace in figure.data
        if trace.type == "mesh3d" and not trace.name.startswith("structure")
    ]
    assert overlays
    assert all(trace.lighting.to_plotly_json() == {} for trace in overlays)


def test_workbench_new_validate_and_structured_error():
    with _client() as client:
        created = client.post("/api/workspace/new", json={})
        assert created.status_code == 200
        payload = created.json()
        assert payload["dirty"] is True
        assert payload["starter"] == service.default_starter()
        assert payload["estimate"]["cells_per_axis"] == [80, 40, 24]
        assert payload["spec"]["monitors"][1]["name"] == "output_flux"

        spec = payload["spec"]
        spec["run"] = {"n_steps": 0}
        invalid = client.post("/api/workspace/validate", json={"spec": spec})
        assert invalid.status_code == 422
        detail = invalid.json()["detail"]
        assert detail["issues"][0]["loc"] == ["run", "n_steps"]

        preflight = client.post("/api/workspace/preflight", json={"spec": spec})
        assert preflight.status_code == 422
        assert preflight.json()["detail"]["issues"][0]["loc"] == ["run", "n_steps"]


def test_workbench_save_is_canonical_and_reopens(tmp_path):
    with _client() as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        target = tmp_path / "device.sim.json"
        saved = client.post(
            "/api/workspace/save", json={"spec": spec, "path": str(target)})
        assert saved.status_code == 200
        assert saved.json()["dirty"] is False
        assert saved.json()["path"] == str(target.resolve())
        assert json.loads(target.read_text())["schema_version"] == spec["schema_version"]
        reopened = client.post("/api/preview", json={"path": str(target)})
        assert reopened.status_code == 200
        assert reopened.json()["spec"] == saved.json()["spec"]


def test_workbench_save_as_never_clobbers_without_explicit_force(tmp_path):
    with _client() as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        target = tmp_path / "existing.sim.json"
        sentinel = b"owned by another application\n"
        target.write_bytes(sentinel)

        refused = client.post("/api/workspace/save", json={
            "spec": spec, "path": str(target),
        })
        assert refused.status_code == 409
        assert refused.json()["detail"] == {
            "code": "target_exists",
            "message": (
                "existing.sim.json already exists; confirm overwrite or choose "
                "a different Save As path"
            ),
            "path": str(target.resolve()),
        }
        assert target.read_bytes() == sentinel

        overwritten = client.post("/api/workspace/save", json={
            "spec": spec, "path": str(target), "force": True,
        })
        assert overwritten.status_code == 200
        assert overwritten.json()["path"] == str(target.resolve())
        assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == spec["schema_version"]


def test_workbench_save_rejects_external_change_without_explicit_force(tmp_path):
    with _client() as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        target = tmp_path / "device.sim.json"
        saved = client.post("/api/workspace/save", json={
            "spec": spec, "path": str(target),
        }).json()
        assert saved["file_identity"]["sha256"]

        local = json.loads(json.dumps(spec))
        local["run"]["shutoff"] = 2.0e-5
        external = json.loads(json.dumps(spec))
        external["run"]["shutoff"] = 3.0e-5
        external_sim, _ = service.parse_sim_spec(external)
        target.write_text(external_sim.to_wire_json() + "\n", encoding="utf-8")
        # Preserve the visible stat precondition. The differing SHA-256 must
        # still catch a same-size replacement whose mtime was deliberately reset.
        assert target.stat().st_size == saved["file_identity"]["size"]
        os.utime(target, ns=(
            target.stat().st_atime_ns,
            saved["file_identity"]["mtime_ns"],
        ))

        refused = client.post("/api/workspace/save", json={
            "spec": local,
            "path": str(target),
            "expected_identity": saved["file_identity"],
        })
        assert refused.status_code == 409
        assert refused.json()["detail"]["code"] == "external_change"
        assert json.loads(target.read_text())["run"]["shutoff"] == 3.0e-5
        conflicted = client.get("/api/workspace").json()
        assert conflicted["external_change"] is True

        forced = client.post("/api/workspace/save", json={
            "spec": local,
            "path": str(target),
            "expected_identity": saved["file_identity"],
            "force": True,
        })
        assert forced.status_code == 200
        assert forced.json()["external_change"] is False
        assert json.loads(target.read_text())["run"]["shutoff"] == 2.0e-5


def test_immediate_quit_flush_recovers_exact_latest_edit_after_restart(tmp_path):
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    run_root = tmp_path / "runs"
    with TestClient(create_app(run_root=run_root)) as first:
        spec = first.post("/api/workspace/new", json={}).json()["spec"]
        spec["run"]["shutoff"] = 4.0e-5
        spec["run"]["courant"] = 0.731
        preserved = first.post(
            "/api/workspace/recovery-preserve", json={"spec": spec})
        assert preserved.status_code == 200
        assert preserved.json()["dirty"] is True
        flushed_spec = preserved.json()["spec"]
        # Exiting the first server immediately after this acknowledged response
        # models Electron's close barrier: no debounce or later request exists.

    assert (run_root / "workspace-recovery.json").is_file()
    with TestClient(create_app(run_root=run_root)) as restarted:
        restored = restarted.get("/api/workspace")
        assert restored.status_code == 200
        assert restored.json()["dirty"] is True
        assert restored.json()["spec"] == flushed_spec


def test_recovery_preserve_keeps_clean_saved_workspace_clean(tmp_path):
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    run_root = tmp_path / "runs"
    target = tmp_path / "saved.sim.json"
    with TestClient(create_app(run_root=run_root)) as first:
        spec = first.post("/api/workspace/new", json={}).json()["spec"]
        saved = first.post("/api/workspace/save", json={
            "spec": spec,
            "path": str(target),
        })
        assert saved.status_code == 200
        saved_workspace = saved.json()
        assert saved_workspace["dirty"] is False

        preserved = first.post(
            "/api/workspace/recovery-preserve", json={"spec": spec})
        assert preserved.status_code == 200
        payload = preserved.json()
        assert payload["dirty"] is False
        assert payload["path"] == saved_workspace["path"]
        assert payload["file_identity"] == saved_workspace["file_identity"]
        assert payload["mtime"] == saved_workspace["mtime"]
        assert payload["spec"] == saved_workspace["spec"]

    with TestClient(create_app(run_root=run_root)) as restarted:
        restored = restarted.get("/api/workspace")
        assert restored.status_code == 200
        payload = restored.json()
        assert payload["dirty"] is False
        assert payload["path"] == saved_workspace["path"]
        assert payload["file_identity"] == saved_workspace["file_identity"]
        assert payload["spec"] == saved_workspace["spec"]


def test_recovery_preserve_marks_newer_renderer_snapshot_dirty(tmp_path):
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    run_root = tmp_path / "runs"
    target = tmp_path / "saved.sim.json"
    with TestClient(create_app(run_root=run_root)) as first:
        spec = first.post("/api/workspace/new", json={}).json()["spec"]
        saved = first.post("/api/workspace/save", json={
            "spec": spec,
            "path": str(target),
        }).json()
        newer = json.loads(json.dumps(spec))
        newer["run"]["courant"] = 0.731

        preserved = first.post(
            "/api/workspace/recovery-preserve", json={"spec": newer})
        assert preserved.status_code == 200
        payload = preserved.json()
        assert payload["dirty"] is True
        assert payload["path"] == saved["path"]
        assert payload["file_identity"] == saved["file_identity"]
        assert payload["spec"]["run"]["courant"] == pytest.approx(0.731)

    with TestClient(create_app(run_root=run_root)) as restarted:
        restored = restarted.get("/api/workspace")
        assert restored.status_code == 200
        payload = restored.json()
        assert payload["dirty"] is True
        assert payload["path"] == saved["path"]
        assert payload["file_identity"] == saved["file_identity"]
        assert payload["spec"]["run"]["courant"] == pytest.approx(0.731)


def test_recovery_preserve_404_does_not_manufacture_workspace(tmp_path):
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    run_root = tmp_path / "runs"
    with TestClient(create_app(run_root=run_root)) as client:
        response = client.post("/api/workspace/recovery-preserve", json={
            "spec": service.default_sim().to_wire_dict(),
        })
        assert response.status_code == 404
        assert client.get("/api/workspace").status_code == 404
        assert not (run_root / "workspace-recovery.json").exists()


def test_failed_run_setup_can_be_recovered_without_a_result(tmp_path):
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    run_root = tmp_path / "runs"
    ledger = RunLedger(run_root)
    canonical = service.default_sim().to_wire_json() + "\n"
    record = ledger.create_request(
        run_id=None,
        canonical_spec=canonical,
        device="cpu",
        timeout_s=None,
        solver={"available": True, "info": {}, "capabilities": {}},
        estimate={"num_cells": 1},
    )
    ledger.seal(record["run_id"], "failed", error={
        "type": "RuntimeError", "message": "intentional failure",
    })

    with TestClient(create_app(run_root=run_root)) as client:
        recovered = client.post(
            f"/api/runs/{record['run_id']}/workspace", json={})
        assert recovered.status_code == 200
        payload = recovered.json()
        assert payload["dirty"] is True
        assert payload["path"] == ""
        assert payload["spec"] == json.loads(canonical)
        assert "failed run" in payload["warnings"][0]


def test_desktop_health_is_authenticated_without_exposing_launch_secret(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    run_root = tmp_path / "runs"
    capability = "expected-launch"
    with TestClient(create_app(
            run_root=run_root, launch_token=capability)) as client:
        denied = client.get("/api/health")
        response = client.get("/api/health", headers={
            "X-PhotonHub-Launch-Capability": capability,
        })
        health = response.json()
    assert denied.status_code == 401
    assert response.status_code == 200
    assert health["ok"] is True
    assert "launch_token" not in health
    assert "launch_capability" not in health
    assert "run_root" not in health


def test_workbench_auto_grid_compiles_to_canonical_coordinates():
    with _client() as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        settings = {
            "wavelength_nm": 1550,
            "steps_per_wvl": 12,
            "max_grading": 1.3,
            "axes": "x",
            "min_nodes": 4,
            "refine_pad_um": 0,
            "dl_min_um": 0,
            "refine_regions": [["x", 0.9, 1.1, 0.025]],
            "snap_interfaces": True,
            "feature_ceil": True,
        }
        response = client.post(
            "/api/workspace/auto-grid", json={"spec": spec, "settings": settings})
        assert response.status_code == 200
        payload = response.json()
        assert payload["dirty"] is True
        assert payload["spec"]["grid"]["type"] == "graded"
        assert set(payload["spec"]["grid"]["coords"]) == {"x"}
        assert len(payload["spec"]["grid"]["coords"]["x"]) >= 4
        assert payload["estimate"]["cells_per_axis"][0] > 4

        invalid = client.post("/api/workspace/auto-grid", json={
            "spec": spec,
            "settings": {**settings, "refine_regions": [["x", 2, 1, .1]]},
        })
        assert invalid.status_code == 422
        assert "requires hi > lo" in str(invalid.json()["detail"])

        wrong_type = client.post(
            "/api/workspace/auto-grid", json={"spec": spec, "settings": []})
        assert wrong_type.status_code == 422
        assert "must be an object" in str(wrong_type.json()["detail"])


def _small_mode_source_sim(*, amplitude=2.5):
    """CPU-sized canonical scene for Workbench mode-solve contract tests."""
    from photonhub import GaussianPulse, ModeSource
    from photonhub.components.grid import realized_cells

    sim = service.default_sim()
    nu = realized_cells(sim.size_um[1], sim.grid.dl_um)
    nv = realized_cells(sim.size_um[2], sim.grid.dl_um)
    source = ModeSource(
        axis="x", direction="+", position_um=1.0,
        polarization="Ey", amplitude=amplitude, n_eff=2.1,
        nu=nu, nv=nv, profile=(0.0,) * (nu * nv),
        source_time=GaussianPulse(
            freq0_hz=193e12, fwidth_hz=10e12, offset=6.0, phase=0.25),
    )
    return sim._validated_copy({"sources": (source,)})


def _with_fresh_mode_solve(sim):
    from photonhub import ModeSolveProvenance, ModeSource

    source = sim.sources[0]
    recipe_data = {
        "solver": "yee", "polarization": "TE", "mode_index": 0,
        "wavelength_um": 299_792_458.0 / source.source_time.freq0_hz * 1e6,
        "center_um": (1.0, 0.6), "size_um": (1.2, 0.6),
        "dl_um": 0.1, "supersample": 8,
        "num_modes": 6, "num_freqs": 1,
    }
    provenance = ModeSolveProvenance(
        **recipe_data,
        input_sha256=service.mode_source_input_sha256(sim, 0, recipe_data),
    )
    data = source.model_dump(mode="python")
    data["mode_solve"] = provenance
    solved = ModeSource.model_validate(data)
    return sim._validated_copy({"sources": (solved,)})


def _install_fake_mode_solver(monkeypatch, sim, calls):
    from photonhub import ModeSource
    from photonhub.components.grid import realized_cells
    from photonhub.plugins import mode_devices, yee_mode

    fake_mode = SimpleNamespace(
        n_eff=2.4321, te_fraction=0.91, shape=(7, 13))

    def solve_single(*args, **kwargs):
        calls["single"] = (args, kwargs)
        return fake_mode

    def solve_bank(*args, **kwargs):
        calls["bank"] = (args, kwargs)
        return {float(freq): fake_mode for freq in args[3]}

    def launch(*args, **kwargs):
        calls["launch"] = (args, kwargs)
        launch_sim = args[0]
        transverse = {
            "x": (1, 2), "y": (2, 0), "z": (0, 1),
        }[kwargs["axis"]]
        nu = realized_cells(
            launch_sim.size_um[transverse[0]], launch_sim.grid.dl_um)
        nv = realized_cells(
            launch_sim.size_um[transverse[1]], launch_sim.grid.dl_um)
        n = nu * nv
        return [ModeSource(
            axis=kwargs["axis"], direction=kwargs["direction"],
            position_um=kwargs["position_um"], polarization="Ey",
            amplitude=1.0, n_eff=fake_mode.n_eff,
            nu=nu, nv=nv, profile=(1.0,) * n,
            minor_polarization="Ez", profile_minor=(0.1,) * n,
            profile_h=(0.9,) * n, profile_h_minor=(0.08,) * n,
            source_time=kwargs["source_time"],
        )]

    monkeypatch.setattr(yee_mode, "solve_yee_mode", solve_single)
    monkeypatch.setattr(yee_mode, "solve_yee_mode_bank", solve_bank)
    monkeypatch.setattr(mode_devices, "mode_launch", launch)
    return fake_mode


def test_mode_source_solve_compiles_controls_and_preserves_explicit_amplitude(
        monkeypatch):
    sim = _small_mode_source_sim()
    calls = {}
    fake_mode = _install_fake_mode_solver(monkeypatch, sim, calls)
    settings = {
        "axis": "x", "position_um": 1.1, "direction": "-",
        "polarization": "TE", "mode_index": 1,
        "wavelength_um": 1.31, "center_um": [1.0, 0.6],
        "size_um": [1.2, 0.6], "dl_um": 0.1,
        "supersample": 3, "num_modes": 5, "num_freqs": 1,
    }

    updated, summary = service.solve_mode_source(sim, 0, settings)
    source = updated.sources[0]
    expected_carrier = 299_792_458.0 / (1.31e-6)

    assert source.type == "mode_source"
    assert source.axis == "x"
    assert source.position_um == pytest.approx(1.1)
    assert source.direction == "-"
    assert source.amplitude == pytest.approx(2.5)
    assert source.source_time.freq0_hz == pytest.approx(expected_carrier)
    assert source.source_time.fwidth_hz == pytest.approx(10e12)
    assert source.source_time.offset == pytest.approx(6.0)
    assert source.source_time.phase == pytest.approx(0.25)
    assert source.profile_minor is not None
    assert source.profile_h is not None
    assert source.mode_solve is not None
    assert source.mode_solve.mode_index == 1
    assert source.mode_solve.input_sha256 == service.mode_source_input_sha256(
        updated, 0)
    assert service.mode_source_statuses(updated)[0]["status"] == "fresh"

    single_args, single_kwargs = calls["single"]
    assert single_args[1:6] == ("x", 1.1, 1.31, "TE", 1)
    assert single_kwargs == {
        "h_center_um": 1.0, "v_center_um": 0.6,
        "half_w_um": 0.6, "half_v_um": 0.3,
        "dl_um": 0.1, "supersample": 3, "num_modes": 5,
    }
    launch_kwargs = calls["launch"][1]
    assert launch_kwargs["launch"] == "aux"
    assert launch_kwargs["power_watts"] == pytest.approx(1.0)
    assert launch_kwargs["center_um"] == (1.0, 0.6)
    assert launch_kwargs["thickness_axis"] == "z"
    assert launch_kwargs["modes_by_freq"] is None

    assert summary == {
        "source_index": 0,
        "solver": "yee",
        "polarization": "TE",
        "mode_index": 1,
        "wavelength_um": 1.31,
        "n_eff": fake_mode.n_eff,
        "te_fraction": fake_mode.te_fraction,
        "profile_shape": [source.nv, source.nu],
        "mode_window_shape": [7, 13],
        "frequency_samples": 1,
        "frequency_samples_hz": [pytest.approx(expected_carrier)],
        "has_minor": True,
        "has_true_h": True,
    }


def test_mode_source_resolve_atomically_rebinds_or_unlinks_driven_port(
        monkeypatch):
    import photonhub as ph

    sim = _with_fresh_mode_solve(_small_mode_source_sim())
    frequency = sim.sources[0].source_time.freq0_hz
    monitor = ph.FieldDftMonitor(
        name="input", center_um=(1.5, 1.0, 0.6), size_um=(0.0, 2.0, 1.2),
        fields=("Ey", "Ez", "Hy", "Hz"), freqs_hz=(frequency,),
        mode_port=ph.ModePort(
            out_direction="-", center_um=(1.0, 0.6), size_um=(1.2, 0.4),
            dl_um=0.1, num_modes=6,
            modes=(ph.PortMode(polarization="TE", mode_index=0),),
            source_index=0, thickness_axis="z",
        ),
    )
    sim = sim._validated_copy({"monitors": (monitor,)})
    _install_fake_mode_solver(monkeypatch, sim, {})
    settings = {
        "axis": "x", "position_um": 1.1, "direction": "+",
        "polarization": "TE", "mode_index": 1,
        "wavelength_um": 1.31, "center_um": [1.0, 0.6],
        "size_um": [1.2, 0.4], "dl_um": 0.1,
        "supersample": 3, "num_modes": 6, "num_freqs": 1,
    }

    rebound, summary = service.solve_mode_source(sim, 0, settings)
    port = rebound.monitors[0].mode_port
    assert port.source_index == 0
    assert port.out_direction == "-"
    assert ("TE", 1) in {
        (mode.polarization, mode.mode_index) for mode in port.modes}
    assert summary["rebound_ports"] == ["input"]

    moved, summary = service.solve_mode_source(
        rebound, 0, {**settings, "position_um": 1.8})
    assert moved.monitors[0].mode_port.source_index is None
    assert summary["unlinked_ports"] == ["input"]


def test_mode_source_rebind_uses_port_physical_polarization(monkeypatch):
    import photonhub as ph

    sim = _with_fresh_mode_solve(_small_mode_source_sim())
    frequency = sim.sources[0].source_time.freq0_hz
    monitor = ph.FieldDftMonitor(
        name="input", center_um=(1.5, 1.0, 0.6), size_um=(0.0, 2.0, 1.2),
        fields=("Ey", "Ez", "Hy", "Hz"), freqs_hz=(frequency,),
        mode_port=ph.ModePort(
            out_direction="-", center_um=(1.0, 0.6), size_um=(1.2, 0.4),
            dl_um=0.1, num_modes=6,
            # x-normal natural TE is E_y. With y declared as thickness this is
            # the port's physical TM family.
            modes=(ph.PortMode(polarization="TM", mode_index=0),),
            source_index=0, thickness_axis="y",
        ),
    )
    sim = sim._validated_copy({"monitors": (monitor,)})
    _install_fake_mode_solver(monkeypatch, sim, {})

    rebound, summary = service.solve_mode_source(sim, 0, {
        "axis": "x", "position_um": 1.1, "direction": "+",
        "polarization": "TE", "mode_index": 1,
        "wavelength_um": 1.31, "center_um": [1.0, 0.6],
        "size_um": [1.2, 0.4], "dl_um": 0.1,
        "supersample": 3, "num_modes": 6, "num_freqs": 1,
    })

    modes = {
        (mode.polarization, mode.mode_index)
        for mode in rebound.monitors[0].mode_port.modes
    }
    assert ("TM", 1) in modes
    assert ("TE", 1) not in modes
    assert summary["rebound_ports"] == ["input"]


def test_mode_source_resolve_expands_trials_for_a_new_mode_family(monkeypatch):
    import photonhub as ph

    sim = _with_fresh_mode_solve(_small_mode_source_sim())
    frequency = sim.sources[0].source_time.freq0_hz
    monitor = ph.FieldDftMonitor(
        name="input", center_um=(1.5, 1.0, 0.6), size_um=(0.0, 2.0, 1.2),
        fields=("Ey", "Ez", "Hy", "Hz"), freqs_hz=(frequency,),
        mode_port=ph.ModePort(
            out_direction="-", center_um=(1.0, 0.6), size_um=(1.2, 0.4),
            dl_um=0.1, num_modes=1,
            modes=(ph.PortMode(polarization="TE", mode_index=0),),
            source_index=0, thickness_axis="z",
        ),
    )
    sim = sim._validated_copy({"monitors": (monitor,)})
    _install_fake_mode_solver(monkeypatch, sim, {})

    rebound, summary = service.solve_mode_source(sim, 0, {
        "axis": "x", "position_um": 1.1, "direction": "+",
        "polarization": "TM", "mode_index": 0,
        "wavelength_um": 1.31, "center_um": [1.0, 0.6],
        "size_um": [1.2, 0.4], "dl_um": 0.1,
        "supersample": 3, "num_modes": 1, "num_freqs": 1,
    })

    port = rebound.monitors[0].mode_port
    assert port.num_modes == 2
    assert {
        (mode.polarization, mode.mode_index) for mode in port.modes
    } == {("TE", 0), ("TM", 0)}
    assert summary["rebound_ports"] == ["input"]


def test_mode_source_solve_builds_broadband_bank_from_pulse(monkeypatch):
    sim = _small_mode_source_sim()
    calls = {}
    _install_fake_mode_solver(monkeypatch, sim, calls)

    updated, summary = service.solve_mode_source(sim, 0, {
        "wavelength_um": 1.55,
        "center_um": [1.0, 0.6], "size_um": [1.2, 0.6],
        "num_freqs": 3,
    })

    carrier = 299_792_458.0 / 1.55e-6
    expected = [carrier - 20e12, carrier, carrier + 20e12]
    bank_args = calls["bank"][0]
    assert bank_args[3] == pytest.approx(expected)
    assert calls["launch"][1]["modes_by_freq"].keys() == pytest.approx(expected)
    assert summary["frequency_samples"] == 3
    assert summary["frequency_samples_hz"] == pytest.approx(expected)
    assert updated.sources[0].source_time.freq0_hz == pytest.approx(carrier)
    assert updated.sources[0].amplitude == pytest.approx(2.5)


def test_mode_source_broadband_rejects_zero_or_unresolvable_bandwidth(
        monkeypatch):
    sim = _small_mode_source_sim()
    source = sim.sources[0]
    zero_pulse = source.source_time.model_copy(update={"fwidth_hz": 0.0})
    invalid_source = source.model_copy(update={"source_time": zero_pulse})
    invalid_sim = sim.model_copy(update={"sources": (invalid_source,)})
    with pytest.raises(service.ModeSourceSolveError, match="bandwidth") as zero:
        service.solve_mode_source(invalid_sim, 0, {
            "center_um": [1.0, 0.6], "size_um": [1.2, 0.6],
            "num_freqs": 3,
        })
    assert zero.value.field == "num_freqs"

    tiny_pulse = source.source_time.model_copy(update={"fwidth_hz": 1e-9})
    tiny_source = source.model_copy(update={"source_time": tiny_pulse})
    tiny_sim = sim.model_copy(update={"sources": (tiny_source,)})
    with pytest.raises(service.ModeSourceSolveError, match="too narrow") as narrow:
        service.solve_mode_source(tiny_sim, 0, {
            "center_um": [1.0, 0.6], "size_um": [1.2, 0.6],
            "num_freqs": 3,
        })
    assert narrow.value.field == "num_freqs"


def test_mode_source_append_publishes_only_the_solved_source(
        monkeypatch):
    sim = service.default_sim()
    calls = {}
    _install_fake_mode_solver(monkeypatch, sim, calls)
    settings = {
        "axis": "x", "position_um": 1.0, "direction": "+",
        "polarization": "TE", "mode_index": 0,
        "wavelength_um": 1.55, "center_um": [1.0, 0.6],
        "size_um": [1.2, 0.6], "dl_um": 0.1,
        "supersample": 2, "num_modes": 6, "num_freqs": 1,
    }
    seed = {
        "amplitude": 1.75,
        "source_time": {
            "type": "gaussian_pulse", "freq0_hz": 190e12,
            "fwidth_hz": 12e12, "offset": 5.5, "phase": 0.1,
        },
    }

    updated, summary = service.append_mode_source(sim, settings, seed)
    assert len(sim.sources) == 1
    assert sim.sources[0].type == "point_dipole"
    assert len(updated.sources) == 2
    solved = updated.sources[1]
    assert solved.type == "mode_source"
    assert solved.nu * solved.nv > 1
    assert solved.amplitude == pytest.approx(1.75)
    assert solved.mode_solve is not None
    assert summary["source_index"] == 1
    assert service.mode_source_statuses(updated) == [{
        "source_index": 1,
        "status": "fresh",
        "message": "Solved profile matches the current geometry and grid.",
        "expected_sha256": solved.mode_solve.input_sha256,
        "actual_sha256": solved.mode_solve.input_sha256,
    }]

    with _client() as client:
        response = client.post("/api/workspace/mode-source/solve", json={
            "spec": sim.to_wire_dict(), "source_index": None,
            "append": True, "settings": settings, "seed": seed,
        })
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["spec"]["sources"]) == 2
    assert payload["spec"]["sources"][1]["nu"] > 1
    assert payload["spec"]["sources"][1]["mode_solve"]["solver"] == "yee"
    assert payload["mode_source_summary"]["source_index"] == 1


def test_mode_source_solve_rejects_wrong_source_and_unsafe_placement():
    with pytest.raises(service.ModeSourceSolveError, match="not a mode_source") as wrong:
        service.solve_mode_source(service.default_sim(), 0, {})
    assert wrong.value.field == "source_index"

    sim = _small_mode_source_sim()
    with pytest.raises(service.ModeSourceSolveError, match="edge guard") as plane:
        service.solve_mode_source(sim, 0, {
            "position_um": 0.05,
            "center_um": [1.0, 0.6], "size_um": [1.2, 0.6],
        })
    assert plane.value.field == "position_um"

    with pytest.raises(service.ModeSourceSolveError, match="outside") as window:
        service.solve_mode_source(sim, 0, {
            "center_um": [0.1, 0.6], "size_um": [1.2, 0.6],
        })
    assert window.value.field == "center_um"


def test_mode_source_provenance_detects_physics_changes_not_card_rename():
    from photonhub import Medium

    legacy = _small_mode_source_sim()
    assert service.mode_source_statuses(legacy)[0]["status"] == "legacy"
    service.assert_no_stale_mode_sources(legacy)

    fresh = _with_fresh_mode_solve(legacy)
    status = service.mode_source_statuses(fresh)[0]
    assert status["status"] == "fresh"
    assert status["expected_sha256"] == status["actual_sha256"]

    renamed_structure = fresh.structures[0].model_copy(
        update={"name": "friendly_core"})
    renamed = fresh._validated_copy({"structures": (renamed_structure,)})
    assert service.mode_source_statuses(renamed)[0]["status"] == "fresh"

    changed_structure = fresh.structures[0].model_copy(
        update={"medium": Medium(permittivity=11.5)})
    changed = fresh._validated_copy({"structures": (changed_structure,)})
    stale = service.mode_source_statuses(changed)[0]
    assert stale["status"] == "stale"
    assert stale["expected_sha256"] != stale["actual_sha256"]
    with pytest.raises(service.StaleModeSourceError, match="re-solve") as error:
        service.assert_no_stale_mode_sources(changed)
    assert error.value.statuses == [stale]

    source = fresh.sources[0]
    shifted_pulse = source.source_time.model_copy(
        update={"freq0_hz": source.source_time.freq0_hz * 1.01})
    shifted_source = source.model_copy(update={"source_time": shifted_pulse})
    shifted = fresh._validated_copy({"sources": (shifted_source,)})
    assert service.mode_source_statuses(shifted)[0]["status"] == "stale"

    reversed_source = source.model_copy(update={"direction": "-"})
    reversed_sim = fresh._validated_copy({"sources": (reversed_source,)})
    assert service.mode_source_statuses(reversed_sim)[0]["status"] == "stale"


def test_mode_source_freshness_hash_does_not_serialize_profile_arrays(
        monkeypatch):
    fresh = _with_fresh_mode_solve(_small_mode_source_sim())
    expected = fresh.sources[0].mode_solve.input_sha256

    def forbidden(*_args, **_kwargs):
        raise AssertionError("freshness hashing must not serialize full Simulation")

    monkeypatch.setattr(type(fresh), "to_wire_dict", forbidden)
    assert service.mode_source_input_sha256(fresh, 0) == expected
    assert service.mode_source_statuses(fresh)[0]["status"] == "fresh"


def test_stale_mode_source_is_reported_and_blocked_before_execution():
    from photonhub import Medium

    fresh = _with_fresh_mode_solve(_small_mode_source_sim())
    changed_structure = fresh.structures[0].model_copy(
        update={"medium": Medium(permittivity=11.25)})
    stale = fresh._validated_copy({"structures": (changed_structure,)})
    stale_spec = stale.to_wire_dict()

    with _client() as client:
        workspace = client.post(
            "/api/workspace/validate", json={"spec": stale_spec})
        assert workspace.status_code == 200
        status = workspace.json()["mode_source_statuses"][0]
        assert status["status"] == "stale"

        for path, body in (
            ("/api/workspace/preflight", {"spec": stale_spec}),
            ("/api/run", {"spec": stale_spec, "device": "cpu"}),
            ("/api/cloud/preflight", {
                "spec": stale_spec, "device": "gpu", "max_usd": 1.0}),
            ("/api/cloud/run", {
                "spec": stale_spec, "preflight_token": "unused"}),
        ):
            blocked = client.post(path, json=body)
            assert blocked.status_code == 409, (path, blocked.text)
            detail = blocked.json()["detail"]
            assert detail["code"] == "stale_mode_source"
            assert detail["mode_source_statuses"][0]["status"] == "stale"

        # Stale rejection happens before any durable/local or paid cloud job
        # mutation, so both execution surfaces remain idle.
        assert client.get("/api/run/status").json()["status"] == "idle"


def test_incomplete_modal_ports_are_blocked_before_local_or_paid_execution():
    resource = resources.files("photonhub.viz").joinpath(
        "examples", "mode_converter.sim.json")
    spec = json.loads(resource.read_text(encoding="utf-8"))
    for monitor in spec["monitors"]:
        if monitor.get("mode_port"):
            monitor["mode_port"]["source_index"] = None

    with _client() as client:
        # Incremental authoring remains schema-valid, but every execution
        # surface fails before launching a solver or requesting a paid quote.
        assert client.post(
            "/api/workspace/validate", json={"spec": spec}).status_code == 200
        for path, body in (
            ("/api/workspace/preflight", {"spec": spec}),
            ("/api/run", {"spec": spec, "device": "cpu"}),
            ("/api/cloud/preflight", {
                "spec": spec, "device": "gpu", "max_usd": 1.0}),
        ):
            blocked = client.post(path, json=body)
            assert blocked.status_code == 422, (path, blocked.text)
            detail = blocked.json()["detail"]
            assert detail["code"] == "modal_ports_not_ready"
            assert "exactly one source-linked" in detail["message"]


def test_modal_preflight_explains_first_source_normalization_contract():
    resource = resources.files("photonhub.viz").joinpath(
        "examples", "mode_converter.sim.json")
    spec = json.loads(resource.read_text(encoding="utf-8"))
    driven = json.loads(json.dumps(spec["sources"][0]))
    inactive_first = json.loads(json.dumps(driven))
    inactive_first["amplitude"] = 0.0
    spec["sources"] = [inactive_first, driven]
    for monitor in spec["monitors"]:
        if monitor.get("mode_port", {}).get("source_index") == 0:
            monitor["mode_port"]["source_index"] = 1

    with _client() as client:
        assert client.post(
            "/api/workspace/validate", json={"spec": spec}).status_code == 200
        blocked = client.post(
            "/api/workspace/preflight", json={"spec": spec})

    assert blocked.status_code == 422
    detail = blocked.json()["detail"]
    assert detail["code"] == "modal_ports_not_ready"
    assert "source_index 0" in detail["message"]
    assert "first wire-order source" in detail["message"]


def test_workbench_mode_source_solve_endpoint_returns_summary(monkeypatch):
    sim = _small_mode_source_sim()
    expected_summary = {
        "source_index": 0, "solver": "yee", "n_eff": 2.4,
    }

    def fake_solve(actual, source_index, settings):
        assert actual.to_wire_dict() == sim.to_wire_dict()
        assert source_index == 0
        assert settings == {"wavelength_um": 1.55}
        return actual, expected_summary

    monkeypatch.setattr(service, "solve_mode_source", fake_solve)
    with _client() as client:
        response = client.post("/api/workspace/mode-source/solve", json={
            "spec": sim.to_wire_dict(), "source_index": 0,
            "settings": {"wavelength_um": 1.55},
        })
        assert response.status_code == 200
        payload = response.json()
        assert payload["dirty"] is True
        assert payload["spec"] == sim.to_wire_dict()
        assert payload["mode_source_summary"] == expected_summary

    def invalid(*_args, **_kwargs):
        raise service.ModeSourceSolveError("size_um", "reduce the mode window")

    monkeypatch.setattr(service, "solve_mode_source", invalid)
    with _client() as client:
        response = client.post("/api/workspace/mode-source/solve", json={
            "spec": sim.to_wire_dict(), "source_index": 0, "settings": {},
        })
    assert response.status_code == 422
    assert response.json()["detail"]["issues"][0]["loc"] == [
        "mode_source", "size_um"]


def test_workspace_validation_is_last_request_wins(monkeypatch):
    with _client() as client:
        base = client.post("/api/workspace/new", json={}).json()["spec"]
        base_x = float(base["size_um"][0])
        slow_x = base_x + 1.0
        fast_x = base_x + 2.0
        slow = json.loads(json.dumps(base))
        fast = json.loads(json.dumps(base))
        slow["size_um"][0] = slow_x
        fast["size_um"][0] = fast_x
        entered = threading.Event()
        release = threading.Event()
        original = service.parse_sim_spec

        def delayed_parse(spec):
            if spec.get("size_um", [None])[0] == slow_x:
                entered.set()
                assert release.wait(2)
            return original(spec)

        monkeypatch.setattr(service, "parse_sim_spec", delayed_parse)
        responses = {}
        thread = threading.Thread(target=lambda: responses.setdefault(
            "slow", client.post("/api/workspace/validate", json={"spec": slow})))
        thread.start()
        assert entered.wait(2)
        newer = client.post("/api/workspace/validate", json={"spec": fast})
        assert newer.status_code == 200
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert responses["slow"].status_code == 409
        stored = client.get("/api/workspace").json()
        assert stored["spec"]["size_um"][0] == pytest.approx(fast_x)


def test_workspace_new_reserves_arrival_order_before_default_creation(monkeypatch):
    base = service.default_sim().to_wire_dict()
    newer_x = float(base["size_um"][0]) + 2.0
    newer = json.loads(json.dumps(base))
    newer["size_um"][0] = newer_x
    entered = threading.Event()
    release = threading.Event()
    original = service.default_sim

    def delayed_default():
        entered.set()
        assert release.wait(2)
        return original()

    monkeypatch.setattr(service, "default_sim", delayed_default)
    with _client() as client:
        responses = {}
        thread = threading.Thread(target=lambda: responses.setdefault(
            "new", client.post("/api/workspace/new", json={})))
        thread.start()
        assert entered.wait(2)
        validated = client.post(
            "/api/workspace/validate", json={"spec": newer})
        assert validated.status_code == 200
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert responses["new"].status_code == 409
        assert client.get("/api/workspace").json()["spec"]["size_um"][0] == \
            pytest.approx(newer_x)


def test_workspace_from_result_reserves_arrival_order(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from photonhub.viz import server

    (tmp_path / "manifest.json").write_text(json.dumps({
        "manifest_version": "1", "run": {}, "grid": {},
        "provenance": {}, "monitors": [],
    }))
    app = server.create_app(tmp_path)
    base = service.default_sim().to_wire_dict()
    newer_x = float(base["size_um"][0]) + 2.0
    newer = json.loads(json.dumps(base))
    newer["size_um"][0] = newer_x
    entered = threading.Event()
    release = threading.Event()

    def delayed_sim_for(_data):
        entered.set()
        assert release.wait(2)
        return service.default_sim()

    monkeypatch.setattr(service, "sim_for", delayed_sim_for)
    monkeypatch.setattr(
        service, "geometry_status", lambda _data: {"status": "matched"})
    with TestClient(app) as client:
        responses = {}
        thread = threading.Thread(target=lambda: responses.setdefault(
            "from_result", client.post("/api/workspace/from-result", json={})))
        thread.start()
        assert entered.wait(2)
        validated = client.post(
            "/api/workspace/validate", json={"spec": newer})
        assert validated.status_code == 200
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert responses["from_result"].status_code == 409
        assert client.get("/api/workspace").json()["spec"]["size_um"][0] == \
            pytest.approx(newer_x)


def test_stale_save_does_not_overwrite_disk_or_newer_workspace(tmp_path, monkeypatch):
    with _client() as client:
        base = client.post("/api/workspace/new", json={}).json()["spec"]
        target = tmp_path / "race.sim.json"
        initial = client.post(
            "/api/workspace/save", json={"spec": base, "path": str(target)})
        assert initial.status_code == 200
        slow = json.loads(json.dumps(base))
        newer = json.loads(json.dumps(base))
        base_x = float(base["size_um"][0])
        slow_x = base_x + 1.0
        newer_x = base_x + 2.0
        slow["size_um"][0] = slow_x
        newer["size_um"][0] = newer_x
        entered = threading.Event()
        release = threading.Event()
        original = service.parse_sim_spec

        def delayed_parse(spec):
            if spec.get("size_um", [None])[0] == slow_x:
                entered.set()
                assert release.wait(2)
            return original(spec)

        monkeypatch.setattr(service, "parse_sim_spec", delayed_parse)
        responses = {}
        thread = threading.Thread(target=lambda: responses.setdefault(
            "save", client.post("/api/workspace/save", json={
                "spec": slow, "path": str(target),
            })))
        thread.start()
        assert entered.wait(2)
        validated = client.post(
            "/api/workspace/validate", json={"spec": newer})
        assert validated.status_code == 200
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert responses["save"].status_code == 409
        assert json.loads(target.read_text())["size_um"][0] == base["size_um"][0]
        assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
        stored = client.get("/api/workspace").json()
        assert stored["spec"]["size_um"][0] == pytest.approx(newer_x)
        assert stored["dirty"] is True


def test_repeated_missing_file_poll_does_not_supersede_newer_edit(
        tmp_path, monkeypatch):
    with _client() as client:
        base = client.post("/api/workspace/new", json={}).json()["spec"]
        target = tmp_path / "watched.sim.json"
        assert client.post("/api/workspace/save", json={
            "spec": base, "path": str(target),
        }).status_code == 200
        dirty = json.loads(json.dumps(base))
        base_x = float(base["size_um"][0])
        dirty_x = base_x + 1.0
        dirty["size_um"][0] = dirty_x
        assert client.post("/api/workspace/validate", json={
            "spec": dirty,
        }).status_code == 200
        target.unlink()
        missing = client.get("/api/workspace")
        assert missing.status_code == 200
        assert missing.json()["external_change"] is True
        assert missing.json()["error"]
        missing_error = missing.json()["error"]

        newer = json.loads(json.dumps(base))
        newer_x = base_x + 2.0
        newer["size_um"][0] = newer_x
        entered = threading.Event()
        release = threading.Event()
        original = service.parse_sim_spec

        def delayed_parse(spec):
            if spec.get("size_um", [None])[0] == newer_x:
                entered.set()
                assert release.wait(2)
            return original(spec)

        monkeypatch.setattr(service, "parse_sim_spec", delayed_parse)
        responses = {}
        thread = threading.Thread(target=lambda: responses.setdefault(
            "validate", client.post(
                "/api/workspace/validate", json={"spec": newer})))
        thread.start()
        assert entered.wait(2)
        for _ in range(3):
            repeated = client.get("/api/workspace")
            assert repeated.status_code == 200
            assert repeated.json()["error"] == missing_error
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert responses["validate"].status_code == 200
        stored = client.get("/api/workspace").json()
        assert stored["spec"]["size_um"][0] == pytest.approx(newer_x)
        assert stored["external_change"] is True
        assert stored["error"]
        assert stored["dirty"] is True


def test_slow_file_watch_observation_cannot_overwrite_newer_gui_edit(
        tmp_path, monkeypatch):
    with _client() as client:
        base = client.post("/api/workspace/new", json={}).json()["spec"]
        target = tmp_path / "slow-watch.sim.json"
        saved = client.post("/api/workspace/save", json={
            "spec": base, "path": str(target),
        }).json()
        base_x = float(base["size_um"][0])
        external_x = base_x + 1.0
        external = json.loads(json.dumps(base))
        external["size_um"][0] = external_x
        target.write_text(json.dumps(external))
        os.utime(target, (saved["mtime"] + 2, saved["mtime"] + 2))

        entered = threading.Event()
        release = threading.Event()
        original_stat = Path.stat

        def delayed_stat(path, *args, **kwargs):
            if str(path) == str(target):
                entered.set()
                assert release.wait(2)
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", delayed_stat)
        responses = {}
        thread = threading.Thread(target=lambda: responses.setdefault(
            "poll", client.get("/api/workspace")))
        thread.start()
        assert entered.wait(2)
        newer_x = base_x + 2.0
        newer = json.loads(json.dumps(base))
        newer["size_um"][0] = newer_x
        validated = client.post(
            "/api/workspace/validate", json={"spec": newer})
        assert validated.status_code == 200
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert responses["poll"].status_code == 200
        assert responses["poll"].json()["spec"]["size_um"][0] == \
            pytest.approx(newer_x)
        assert client.get("/api/workspace").json()["spec"]["size_um"][0] == \
            pytest.approx(newer_x)


def test_opening_same_result_again_advances_result_id(tmp_path):
    manifest = {
        "manifest_version": "1.0", "schema_version": "1.0.0",
        "run": {}, "grid": {}, "provenance": {}, "monitors": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with _client() as client:
        first = client.post("/api/open", json={"dir": str(tmp_path)}).json()
        second = client.post("/api/open", json={"dir": str(tmp_path)}).json()
        assert first["result_id"]
        assert second["result_id"]
        assert first["result_id"] != second["result_id"]
        stale = client.get("/api/scene", params={"rev": first["result_id"]})
        assert stale.status_code == 409
        assert "stale result revision" in stale.json()["detail"]


def test_result_open_is_last_request_wins_by_arrival(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from photonhub.viz import server

    first_dir = tmp_path / "slow"
    second_dir = tmp_path / "newer"
    for directory in (first_dir, second_dir):
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({
            "manifest_version": "1", "run": {}, "grid": {},
            "provenance": {}, "monitors": [],
        }))
    entered = threading.Event()
    release = threading.Event()
    original = service.load_result

    def delayed_load(path):
        if Path(path).resolve() == first_dir.resolve():
            entered.set()
            assert release.wait(2)
        return original(path)

    monkeypatch.setattr(service, "load_result", delayed_load)
    with TestClient(server.create_app()) as client:
        responses = {}
        thread = threading.Thread(target=lambda: responses.setdefault(
            "slow", client.post("/api/open", json={"dir": str(first_dir)})))
        thread.start()
        assert entered.wait(2)
        newer = client.post("/api/open", json={"dir": str(second_dir)})
        assert newer.status_code == 200
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert responses["slow"].status_code == 409
        final = client.get("/api/session").json()
        assert final["result_id"] == newer.json()["result_id"]
        assert Path(final["output_dir"]) == second_dir


def test_result_bundle_and_revision_publish_as_one_snapshot(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from photonhub.viz import server

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    for directory in (first_dir, second_dir):
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({
            "manifest_version": "1", "run": {}, "grid": {},
            "provenance": {}, "monitors": [],
        }))

    app = server.create_app(first_dir)
    entered = threading.Event()
    release = threading.Event()
    original_uuid4 = server.uuid.uuid4

    def delayed_uuid4():
        entered.set()
        assert release.wait(2)
        return original_uuid4()

    monkeypatch.setattr(server.uuid, "uuid4", delayed_uuid4)
    monkeypatch.setattr(
        service, "meta",
        lambda data, _name: {"bundle": Path(data.output_dir).name},
    )
    with TestClient(app) as client:
        old_revision = client.get("/api/session").json()["result_id"]
        opened = {}
        thread = threading.Thread(
            target=lambda: opened.setdefault(
                "response", client.post("/api/open", json={"dir": str(second_dir)})))
        thread.start()
        assert entered.wait(2)
        # While the new bundle is still being prepared, the old revision must
        # continue to address the old data; it may never see a mixed pair.
        concurrent = client.get("/api/monitor/anything/meta", params={"rev": old_revision})
        assert concurrent.status_code == 200
        assert concurrent.json() == {"bundle": "first"}
        release.set()
        thread.join(timeout=2)
        assert opened["response"].status_code == 200


def test_completed_run_is_published_with_session_and_finish_time(monkeypatch):
    from photonhub.runners import local

    with _client() as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        # This fake solver exercises publication ordering only and emits no
        # monitor artifacts, so its immutable request must say the same.
        spec["monitors"] = []

        def fake_run(sim, *, output_dir=None, log_file=None, **_kwargs):
            import hashlib

            output_dir = Path(output_dir)
            canonical = sim.to_wire_json() + "\n"
            (output_dir / "sim.json").write_text(canonical)
            Path(log_file).write_text(json.dumps({"event": "done"}) + "\n")
            (output_dir / "manifest.json").write_text(json.dumps({
                "manifest_version": "1", "schema_version": "1.0.0",
                "run": {"n_steps": 1, "steps_run": 1, "dt_s": 1e-16,
                        "aborted": False, "abort_reason": ""},
                "grid": {"shape": [1, 1, 1], "dl_um": 0.1},
                "provenance": {
                    "input_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                },
                "monitors": [],
            }) + "\n")
            return SimpleNamespace(output_dir=output_dir)

        monkeypatch.setattr(local, "run_local", fake_run)

        def delayed_session(data, result_id, run_id=None):
            time.sleep(0.25)
            return {"result_id": result_id, "run_id": run_id,
                    "output_dir": str(data.output_dir)}

        monkeypatch.setattr(service, "session", delayed_session)
        started = client.post("/api/run", json={"spec": spec, "device": "cpu"})
        assert started.status_code == 200

        deadline = time.monotonic() + 2
        terminal = None
        while time.monotonic() < deadline:
            polled = client.get("/api/run/status").json()
            if polled["status"] not in {"queued", "running"}:
                terminal = polled
                break
            time.sleep(0.01)
        assert terminal is not None
        assert terminal["status"] == "completed"
        assert terminal["session"]["result_id"]
        assert terminal["finished_at"] is not None


def test_app_shutdown_cancels_active_run_and_waits_for_worker(monkeypatch):
    from fastapi.testclient import TestClient
    from photonhub.runners import local
    from photonhub.viz.server import create_app

    started = threading.Event()
    cancelled = threading.Event()

    def blocking_run(*_args, cancel_event=None, **_kwargs):
        started.set()
        assert cancel_event is not None and cancel_event.wait(2)
        cancelled.set()
        raise SolverRunError("cancelled by app shutdown")

    monkeypatch.setattr(local, "run_local", blocking_run)
    with TestClient(create_app()) as client:
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        response = client.post("/api/run", json={"spec": spec, "device": "cpu"})
        assert response.status_code == 200
        assert started.wait(2)
    assert cancelled.is_set()


def test_nonfinite_aborted_field_is_serialized_as_gaps(tmp_path):
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    raw = np.asarray([1, 0, np.nan, 0, np.inf, 0, 4, 0], dtype="<f4")
    raw.tofile(tmp_path / "partial.bin")
    manifest = {
        "manifest_version": "1", "schema_version": "1.0.0",
        "run": {"dt_s": 1e-16, "aborted": True,
                "abort_reason": "non_finite_energy"},
        "grid": {"shape": [2, 2, 1], "dl_um": 0.1,
                 "size_um": [0.2, 0.2, 0.1]},
        "provenance": {},
        "monitors": [{
            "name": "partial", "type": "field_dft", "file": "partial.bin",
            "dtype": "float32", "shape": [1, 1, 1, 2, 2, 2],
            "dims": ["freq", "component", "z", "y", "x", "complex"],
            "components": ["Ex"], "freqs_hz": [193.4e12],
        }],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.warns(UserWarning, match="ABORTED"):
        app = create_app(tmp_path)
    with TestClient(app) as client:
        revision = client.get("/api/session").json()["result_id"]
        field = client.get("/api/monitor/partial/field", params={
            "field": "Ex", "val": "real", "rev": revision,
        })
        assert field.status_code == 200
        assert field.json()["data"][0]["z"] == [[1.0, None], [None, 4.0]]
        stats = client.get("/api/monitor/partial/stats", params={
            "field": "Ex", "val": "real", "rev": revision,
        })
        assert stats.status_code == 200
        assert stats.json()["nonfinite_count"] == 2
        assert stats.json()["sample_sum_squares"] == pytest.approx(17.0)


def test_geometry_overlay_fails_closed_on_input_hash_mismatch(tmp_path):
    sim_text = service.default_sim().to_wire_json() + "\n"
    (tmp_path / "sim.json").write_text(sim_text)
    manifest = {
        "manifest_version": "1.0", "schema_version": "1.0.0", "run": {},
        "grid": {}, "provenance": {"input_sha256": "0" * 64}, "monitors": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    payload = service.session(service.load_result(tmp_path), result_id="result")
    assert payload["geometry"]["status"] == "mismatch"
    assert payload["has_scene"] is False
    assert service.sim_for(service.load_result(tmp_path)) is None


def test_raw_result_arrays_are_disk_backed(tmp_path):
    np.arange(8, dtype="<f4").tofile(tmp_path / "probe.bin")
    manifest = {
        "manifest_version": "1", "monitors": [{
            "name": "probe", "type": "field_time", "file": "probe.bin",
            "dtype": "float32", "shape": [4, 2],
            "dims": ["sample", "component"], "components": ["Ex", "Ey"],
            "sample_steps": [1, 2, 3, 4],
        }],
        "run": {"dt_s": 1e-16}, "grid": {"dl_um": 0.1}, "provenance": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = SimulationData(tmp_path)
    assert isinstance(result["probe"].data, np.memmap)
    assert result["probe"].values[-1, -1] == 7


def test_run_phsolver_cancel_event_terminates_child(tmp_path):
    import sys

    marker = tmp_path / "survived.txt"
    code = (
        "import json,time,pathlib; "
        "print(json.dumps({'event':'start'}),flush=True); "
        "time.sleep(10); pathlib.Path(r'%s').write_text('orphan')" % marker
    )
    cancel = threading.Event()
    timer = threading.Timer(0.15, cancel.set)
    timer.start()
    started = time.monotonic()
    with pytest.raises(SolverRunError, match="cancelled"):
        run_phsolver([sys.executable, "-c", code], cancel_event=cancel)
    timer.cancel()
    assert time.monotonic() - started < 3.0
    time.sleep(0.1)
    assert not marker.exists()


def test_run_phsolver_cancel_terminates_descendant_process_tree(tmp_path):
    import sys

    marker = tmp_path / "descendant-survived.txt"
    descendant = (
        "import pathlib,time; time.sleep(0.8); "
        f"pathlib.Path({str(marker)!r}).write_text('orphan')"
    )
    parent = (
        "import json,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}]); "
        "print(json.dumps({'event':'start'}),flush=True); time.sleep(10)"
    )
    cancel = threading.Event()
    timer = threading.Timer(0.15, cancel.set)
    timer.start()
    try:
        with pytest.raises(SolverRunError, match="cancelled"):
            run_phsolver([sys.executable, "-c", parent], cancel_event=cancel)
    finally:
        timer.cancel()
    time.sleep(1.0)
    assert not marker.exists()


def test_windows_solver_tree_uses_kill_on_close_job(monkeypatch):
    from photonhub.runners import phsolver

    class FakeProcess:
        pid = 1234

        def poll(self):
            return None

    class FakeJob:
        def __init__(self, proc):
            assert proc.pid == 1234
            self.closed = False

        def close(self):
            self.closed = True

    fake_job = FakeJob(FakeProcess())
    monkeypatch.setattr(phsolver.os, "name", "nt")
    monkeypatch.setattr(phsolver, "_WindowsJob", lambda _proc: fake_job)

    owner = phsolver._ProcessTreeOwner(FakeProcess())
    assert phsolver._process_group_popen_kwargs()["creationflags"] != 0
    owner.stop(force=True)
    assert fake_job.closed is True


def test_run_phsolver_reaps_unused_cancel_waiter():
    import sys

    before = {
        thread.ident for thread in threading.enumerate()
        if thread.name == "photonhub-cancel-waiter" and thread.is_alive()
    }
    event = run_phsolver([
        sys.executable, "-c",
        "import json; print(json.dumps({'event': 'done'}), flush=True)",
    ], cancel_event=threading.Event())
    assert event["event"] == "done"
    leaked = [
        thread for thread in threading.enumerate()
        if thread.name == "photonhub-cancel-waiter" and thread.is_alive()
        and thread.ident not in before
    ]
    assert leaked == []
