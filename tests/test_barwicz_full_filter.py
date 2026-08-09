"""Engine-free checks for the prepared full Barwicz three-ring GPU scene."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from photonhub.components import Box, Cylinder

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "benchmarks" / "resonators"))
_PLAN = (
    _REPO / "benchmarks" / "resonators" / "results" / "barwicz_microring"
    / "barwicz_full_filter_run_plan.json"
)

import barwicz_full_filter as full  # noqa: E402
import barwicz_microring as br  # noqa: E402


@pytest.fixture(scope="module")
def prepared():
    # A small monitor-frequency set keeps the test quick; geometry, launch mode,
    # port planes, and source construction are identical to the production plan.
    return full.build_full_filter(
        dl_um=0.10,
        run_time_ps=1.0,
        num_freqs=5,
        mode_supersample=2,
    )


def test_full_filter_reconstructed_figure2_geometry_and_stack(prepared):
    sim, meta = prepared.simulation, prepared.metadata
    by_role = dict(zip(meta["geometry"]["structure_roles"], sim.structures))
    assert len(sim.structures) == 11
    assert meta["geometry"]["ring_count"] == 3
    assert meta["geometry"]["bus_count"] == 2

    rings = [by_role[f"ring_{i}_sin"].geometry for i in range(3)]
    for ring in rings:
        assert isinstance(ring, Cylinder)
        assert ring.radius_um == pytest.approx(7.300)
        assert ring.inner_radius_um == pytest.approx(6.250)
        assert ring.length_um == pytest.approx(0.314)
        assert ring.center_um[2] == pytest.approx(2.687)
    d = 14.868 / math.sqrt(2.0)
    for first, second in zip(rings, rings[1:]):
        assert second.center_um[0] - first.center_um[0] == pytest.approx(d)
        assert second.center_um[1] - first.center_um[1] == pytest.approx(d)
        assert math.dist(first.center_um[:2], second.center_um[:2]) == pytest.approx(
            14.868
        )

    left = by_role["left_bus_sin"].geometry
    right = by_role["right_bus_sin"].geometry
    assert isinstance(left, Box) and isinstance(right, Box)
    assert left.size_um == pytest.approx((1.050, sim.size_um[1], 0.314))
    assert right.size_um == pytest.approx((1.050, sim.size_um[1], 0.314))
    left_gap = rings[0].center_um[0] - rings[0].radius_um - (
        left.center_um[0] + 0.5 * left.size_um[0]
    )
    right_gap = right.center_um[0] - 0.5 * right.size_um[0] - (
        rings[2].center_um[0] + rings[2].radius_um
    )
    assert left_gap == pytest.approx(0.060)
    assert right_gap == pytest.approx(0.060)

    base = by_role["recessed_sio2_substrate"].geometry
    assert base.size_um[2] == pytest.approx(2.404)
    for i in range(3):
        support = by_role[f"ring_{i}_sio2_pedestal"].geometry
        assert support.length_um == pytest.approx(0.126)
        assert support.center_um[2] == pytest.approx(2.467)
    for name in ("left_bus", "right_bus"):
        support = by_role[f"{name}_sio2_pedestal"].geometry
        assert support.size_um[2] == pytest.approx(0.126)
        assert support.center_um[2] == pytest.approx(2.467)


def test_full_filter_mode_launch_ports_and_field_profile(prepared):
    sim, meta = prepared.simulation, prepared.metadata
    assert sim.symmetry == (0, 0, 0)
    assert (sim.boundaries.x, sim.boundaries.y, sim.boundaries.z) == (
        "pml", "pml", "pml"
    )
    assert sim.subpixel is True and sim.subpixel_method == "contour"
    assert sim.pml_num_layers == 12
    assert sim.pml_num_layers * sim.grid.dl_um == pytest.approx(1.2)

    # A full-vector equivalence-current sheet has electric and magnetic currents
    # and is solved from the actual asymmetric TE0 bus cross-section.
    assert len(sim.sources) > 100
    assert {source.type for source in sim.sources} == {"point_dipole"}
    assert {source.polarization[0] for source in sim.sources} == {"E", "H"}
    assert meta["source"]["polarization"] == "TE0"
    assert meta["source"]["mode_te_fraction"] > 0.90
    assert meta["source"]["frequency_dependent_profile"] is False

    expected_direction = {
        "input": "+", "through": "+", "drop": "-", "add": "+"
    }
    assert meta["monitors"]["port_directions"] == expected_direction
    by_name = {monitor.name: monitor for monitor in sim.monitors}
    for port in expected_direction:
        monitor = by_name[f"port_{port}"]
        assert monitor.type == "field_dft"
        assert monitor.fields == ("Ez", "Ex", "Hz", "Hx")
        assert monitor.size_um[1] == pytest.approx(0.0)
        assert monitor.size_um[0] < 5.0
        assert monitor.size_um[2] < 5.0
    assert by_name["port_input"].center_um[1] < meta["geometry"][
        "ring_centers_xy_um"
    ][0][1]
    assert by_name["port_drop"].center_um[1] == pytest.approx(
        by_name["port_input"].center_um[1]
    )
    assert by_name["port_through"].center_um[1] > meta["geometry"][
        "ring_centers_xy_um"
    ][-1][1]

    field = by_name["field_xy"]
    assert field.fields == ("Ex", "Ey", "Ez")
    assert field.size_um[2] == pytest.approx(0.0)
    assert field.interval_space == (2, 2, 1)
    assert len([m for m in sim.monitors if m.type == "field_time"]) == 3

    # Every source cell and every observation window must remain in the
    # non-absorbing interior.  The bus/substrate structures intentionally cross
    # PML, but launch/readout support must not.
    pml = sim.pml_num_layers * sim.grid.dl_um
    interior_hi = tuple(size - pml for size in sim.size_um)
    for source in sim.sources:
        for axis, value in enumerate(source.center_um):
            assert pml <= value <= interior_hi[axis]
    for monitor in sim.monitors:
        extent = getattr(monitor, "size_um", (0.0, 0.0, 0.0))
        for axis, (value, width) in enumerate(zip(monitor.center_um, extent)):
            assert value - 0.5 * width >= pml
            assert value + 0.5 * width <= interior_hi[axis]


def test_full_filter_wire_roundtrip_and_honest_claim_boundary(prepared):
    sim, meta = prepared.simulation, prepared.metadata
    wire = sim.to_wire_json()
    restored = full.ph.Simulation.from_wire_json(wire)
    assert json.loads(restored.to_wire_json()) == json.loads(wire)
    assert meta["variant"] == "full_three_ring_whole_filter_cross_check"
    assert meta["paper"]["figure_2_geometry_reconstruction"] is True
    assert meta["paper"]["figure_3_whole_filter_fdtd"] is False
    assert meta["claims"]["paper_method_claim"] is False
    assert meta["claims"]["paper_identical_claim"] is False
    assert meta["claims"]["physics_result_claim"] is False
    assert meta["validation"]["gpu_execution"] is False
    assert meta["stack"]["silicon_substrate_included"] is False
    assert br.CASES == ("isolated", "ring_bus", "ring_pair")


def test_full_filter_pml_and_run_plan_contracts():
    with pytest.raises(ValueError, match="divide the fixed 1.2-um PML"):
        full.build_full_filter(dl_um=0.07, run_time_ps=1.0, num_freqs=3)
    with pytest.raises(ValueError, match="lower z-PML"):
        full.build_full_filter(
            dl_um=0.10,
            run_time_ps=1.0,
            num_freqs=3,
            port_window_half_height_um=2.0,
        )

    args = SimpleNamespace(
        pilot_dl=0.10,
        pilot_time_ps=60.0,
        spatial_dl=[0.10, 0.08, 0.06],
        spatial_time_ps=120.0,
        time_dl=0.08,
        time_ps=[60.0, 120.0, 240.0],
    )
    rows = full._run_requests(args)
    assert len(rows) == 6
    duplicate = [
        row for row in rows
        if row["dl_um"] == pytest.approx(0.08)
        and row["run_time_ps"] == pytest.approx(120.0)
    ]
    assert duplicate == [{
        "dl_um": 0.08,
        "run_time_ps": 120.0,
        "roles": ["spatial_ladder", "time_ladder"],
    }]


def test_committed_full_filter_plan_is_validated_but_not_run():
    plan = json.loads(_PLAN.read_text(encoding="utf-8"))
    assert plan["variant"] == "full_three_ring_whole_filter_cross_check"
    assert plan["status"] == "prepared_not_run"
    assert plan["paper_method_claim"] is False
    assert len(plan["runs"]) == 6
    assert plan["validation"] == {
        "wire_roundtrip": True,
        "phsolver_validate": True,
        "gpu_execution": False,
        "spatial_convergence": False,
        "time_window_convergence": False,
    }
    for row in plan["runs"]:
        assert row["wire_roundtrip"] is True
        assert row["phsolver_validation"]["ok"] is True
        assert row["phsolver_validation"]["grid_shape"] == row["cells_per_axis"]
        assert row["phsolver_validation"]["n_steps"] == row["num_steps"]
        assert row["spec_written"] is False
