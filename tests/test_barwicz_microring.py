"""Engine-free tests for the Barwicz et al. (2004) microring runner."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from photonhub.components import Box, Cylinder, Medium
from photonhub.replicate import PaperSpec, build_geometry

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "benchmarks" / "resonators"))

import barwicz_microring as br  # noqa: E402

_SPEC = (
    _REPO / "benchmarks" / "replicate" / "specs"
    / "barwicz_microring_filter_2004.yaml"
)
_RESULTS = _REPO / "benchmarks" / "resonators" / "results" / "barwicz_microring"
_PUBLIC_MIRROR_GPU = _RESULTS / "barwicz_public_mirror_mi300x_60ps.json"


def test_barwicz_spec_and_registered_three_ring_geometry():
    spec = PaperSpec.from_yaml(_SPEC)
    assert spec.name == "barwicz_microring_filter_2004"
    assert spec.device.kind == "barwicz_three_ring_add_drop"
    assert spec.stack.core.thickness_um == pytest.approx(0.314)
    assert spec.stack.core.material == "n=2.217"
    assert spec.optical.polarization == "TE"

    component = build_geometry(
        spec.device.kind,
        spec.device.params,
        medium=Medium(permittivity=2.217**2),
        thickness_um=spec.stack.core.thickness_um,
    )
    rings = [s.geometry for s in component.structures
             if isinstance(s.geometry, Cylinder)]
    buses = [s.geometry for s in component.structures
             if isinstance(s.geometry, Box)]
    assert len(rings) == 3
    assert len(buses) == 2
    assert {port.name for port in component.ports} == {
        "bus_in", "through", "drop", "add",
    }
    assert all(ring.radius_um == pytest.approx(7.3) for ring in rings)
    assert all(ring.inner_radius_um == pytest.approx(6.25) for ring in rings)
    assert math.dist(rings[0].center_um[:2], rings[1].center_um[:2]) == pytest.approx(
        14.868
    )
    left_gap = rings[0].center_um[0] - rings[0].radius_um - (
        buses[0].center_um[0] + 0.5 * buses[0].size_um[0]
    )
    right_gap = buses[1].center_um[0] - 0.5 * buses[1].size_um[0] - (
        rings[2].center_um[0] + rings[2].radius_um
    )
    assert left_gap == pytest.approx(0.060)
    assert right_gap == pytest.approx(0.060)


def test_paper_defaults_keep_measurement_and_inference_provenance():
    p = br.BARWICZ_2004_PARAMETERS
    assert p["sin_thickness_um"]["value"] == pytest.approx(0.314)
    assert p["sin_index_1550"]["value"] == pytest.approx(2.217)
    assert p["sio2_thickness_um"]["value"] == pytest.approx(2.530)
    assert p["sio2_index_1550"]["value"] == pytest.approx(1.455)
    assert p["etch_depth_um"]["value"] == pytest.approx(0.440)
    assert p["ring_outer_radius_um"]["value"] == pytest.approx(7.300)
    assert p["ring_bus_gap_um"]["value"] == pytest.approx(0.060)
    assert p["ring_ring_gap_um"]["value"] == pytest.approx(0.268)
    assert tuple(p["bus_widths_um"]["value"]) == br.BUS_WIDTH_SWEEP_UM
    assert p["ring_width_um"]["value"] == pytest.approx(1.050)
    assert p["ring_width_um"]["inferred"] is True
    assert p["chain_angle_deg"]["inferred"] is True
    assert p["substrate_wafer"]["value"] == "silicon"
    assert all("provenance" in record and "inferred" in record
               for record in p.values())


def test_exact_derived_dimensions_and_ring_centers():
    assert br.oxide_overetch_um() == pytest.approx(0.126)
    assert br.ring_centerline_radius_um() == pytest.approx(6.775)
    assert br.ring_center_separation_um() == pytest.approx(14.868)
    pair = br.series_ring_centers(2)
    assert math.dist(pair[0][:2], pair[1][:2]) == pytest.approx(14.868)
    three = br.series_ring_centers(3)
    assert math.dist(three[0][:2], three[1][:2]) == pytest.approx(14.868)
    assert math.dist(three[1][:2], three[2][:2]) == pytest.approx(14.868)


def test_ring_pair_annuli_and_oxide_pedestals_are_exact():
    sim, meta = br.build_simulation("ring_pair", dl_um=0.10, n_steps=20)
    by_role = dict(zip(meta["structure_roles"], sim.structures))
    assert meta["oxide_overetch_um"] == pytest.approx(0.126)
    assert meta["oxide_recessed_surface_z_um"] == pytest.approx(2.404)
    for i in (0, 1):
        core = by_role[f"ring_{i}_sin"].geometry
        support = by_role[f"ring_{i}_sio2_pedestal"].geometry
        assert core.radius_um == pytest.approx(7.300)
        assert core.inner_radius_um == pytest.approx(6.250)
        assert core.length_um == pytest.approx(0.314)
        assert support.radius_um == pytest.approx(core.radius_um)
        assert support.inner_radius_um == pytest.approx(core.inner_radius_um)
        assert support.length_um == pytest.approx(0.126)
    c0 = by_role["ring_0_sin"].geometry.center_um
    c1 = by_role["ring_1_sin"].geometry.center_um
    assert math.dist(c0[:2], c1[:2]) == pytest.approx(14.868)


def test_ring_bus_gap_and_bus_width_are_geometric_not_grid_rounded():
    width = 0.650
    sim, meta = br.build_simulation(
        "ring_bus", dl_um=0.08, n_steps=20, bus_width_um=width)
    by_role = dict(zip(meta["structure_roles"], sim.structures))
    ring = by_role["ring_0_sin"].geometry
    bus = by_role["bus_sin"].geometry
    edge_gap = ring.center_um[0] - ring.radius_um - (
        bus.center_um[0] + 0.5 * bus.size_um[0])
    assert edge_gap == pytest.approx(0.060)
    assert bus.size_um[0] == pytest.approx(width)
    assert by_role["bus_sio2_pedestal"].geometry.size_um[2] == pytest.approx(0.126)


@pytest.mark.parametrize("case,n_rings", [
    ("isolated", 1), ("ring_bus", 1), ("ring_pair", 2),
])
def test_all_cases_build_valid_asymmetric_3d_wires(case, n_rings):
    sim, meta = br.build_simulation(case, dl_um=0.10, n_steps=50)
    wire = sim.to_wire_dict()
    json.dumps(wire)
    assert meta["asymmetric_3d"] is True
    assert len(meta["ring_centers_um"]) == n_rings
    assert sim.symmetry == (0, 0, 0)
    assert (sim.boundaries.x, sim.boundaries.y, sim.boundaries.z) == (
        "pml", "pml", "pml")
    assert sim.subpixel is True and sim.subpixel_method == "contour"
    assert sim.background.permittivity == pytest.approx(1.0)
    assert len(sim.sources) == 1
    assert len(sim.monitors) == n_rings
    assert all(n >= 4 for n in meta["n_cells"])
    assert meta["mcells"] > 0.0
    assert meta["core_to_top_pml_clearance_um"] >= 0.700 - 1e-12
    assert meta["bottom_pml_to_recessed_surface_clearance_um"] > 0.0
    assert meta["silicon_substrate_included"] is False
    assert meta["analysis_monitor_name"] == "ringdown_0"
    assert meta["analysis_fields"] == ["Ey"]
    assert sim.sources[0].polarization == "Ex"
    assert sim.monitors[0].fields == ("Ex", "Ey", "Ez")
    ring0 = np.asarray(meta["ring_centers_um"][0])
    source = np.asarray(sim.sources[0].center_um)
    probe = np.asarray(sim.monitors[0].center_um)
    assert source[0] - ring0[0] == pytest.approx(
        meta["ring_centerline_radius_um"]
    )
    assert source[1] == pytest.approx(ring0[1])
    assert probe[0] == pytest.approx(ring0[0])
    assert ring0[1] - probe[1] == pytest.approx(
        meta["ring_centerline_radius_um"]
    )


@pytest.mark.parametrize("case,n_rings", [
    ("isolated", 1), ("ring_bus", 1), ("ring_pair", 2),
])
def test_public_mirror_cells_have_fixed_half_domain_stack_and_probes(
    case, n_rings,
):
    sim, meta = br.build_public_mirror_simulation(
        case, dl_um=0.10, n_steps=50,
    )
    wire = sim.to_wire_dict()
    json.dumps(wire)

    assert sim.size_um == pytest.approx((10.3, 35.5, 5.0))
    assert meta["full_domain_equivalent_size_um"] == pytest.approx(
        (20.6, 35.5, 5.0)
    )
    assert sim.symmetry == (1, 0, 0)
    assert wire["symmetry"] == [1, 0, 0]
    assert meta["mirror_plane"] == {
        "axis": "x",
        "face": "minimum",
        "coordinate_um": 0.0,
        "boundary": "PMC",
        "tangential_e_parity": "even",
    }
    assert meta["full_asymmetric_z_stack"] is True
    assert meta["domain_fraction"] == pytest.approx(0.5)
    assert meta["independent_public_input_variant"] is True
    assert meta["parity_claim"] is False
    assert meta["convergence_claim"] is False
    assert "not geometry parity" in meta["variant_scope"]

    assert sim.pml_num_layers == 12
    assert meta["pml_realized_physical_thickness_um"] == pytest.approx(1.2)
    assert sim.pml_kappa_max == pytest.approx(5.0)
    assert sim.pml_sigma_max == pytest.approx(1.5)
    assert sim.pml_alpha_max > 0.24
    assert wire["pml_kappa_max"] == pytest.approx(5.0)
    # The wire omits schema-default 1.5; the engine resolves the same value.
    assert wire.get("pml_sigma_max", 1.5) == pytest.approx(1.5)
    assert wire["pml_alpha_max"] == pytest.approx(sim.pml_alpha_max)

    by_role = dict(zip(meta["structure_roles"], sim.structures))
    base = by_role["recessed_sio2_substrate"].geometry
    assert base.center_um == pytest.approx((5.15, 17.75, 1.202))
    assert base.size_um == pytest.approx((10.3, 35.5, 2.404))
    assert len(meta["ring_centers_um"]) == n_rings
    assert meta["ring_centers_um"][0] == pytest.approx(
        (0.0, 10.3, 2.687)
    )
    for i in range(n_rings):
        core = by_role[f"ring_{i}_sin"].geometry
        support = by_role[f"ring_{i}_sio2_pedestal"].geometry
        assert core.radius_um == pytest.approx(7.3)
        assert core.inner_radius_um == pytest.approx(6.25)
        assert core.length_um == pytest.approx(0.314)
        assert support.length_um == pytest.approx(0.126)
    if case == "ring_pair":
        assert meta["ring_centers_um"][1] == pytest.approx(
            (0.0, 25.168, 2.687)
        )

    assert sim.sources[0].polarization == "Ey"
    assert sim.sources[0].center_um == pytest.approx((0.0, 3.525, 2.687))
    assert sim.monitors[0].name == "ringdown_0"
    assert sim.monitors[0].center_um == pytest.approx(sim.sources[0].center_um)
    assert all(m.fields == ("Ey",) for m in sim.monitors)
    assert all(m.interval_steps == 10 for m in sim.monitors)
    if case == "ring_pair":
        assert sim.monitors[1].name == "ringdown_1"
        assert sim.monitors[1].center_um == pytest.approx(
            (0.0, 31.943, 2.687)
        )


def test_public_mirror_ring_bus_is_horizontal_above_lower_ring():
    width = 0.650
    sim, meta = br.build_public_mirror_simulation(
        "ring_bus", dl_um=0.08, n_steps=20, bus_width_um=width,
    )
    by_role = dict(zip(meta["structure_roles"], sim.structures))
    ring = by_role["ring_0_sin"].geometry
    bus = by_role["bus_sin"].geometry
    support = by_role["bus_sio2_pedestal"].geometry
    assert bus.center_um[0] == pytest.approx(5.15)
    assert bus.size_um == pytest.approx((10.3, width, 0.314))
    assert support.size_um == pytest.approx((10.3, width, 0.126))
    edge_gap = bus.center_um[1] - 0.5 * bus.size_um[1] - (
        ring.center_um[1] + ring.radius_um
    )
    assert edge_gap == pytest.approx(0.060)
    assert meta["bus_center_y_um"] == pytest.approx(bus.center_um[1])
    assert meta["local_cell_orientation"] == "horizontal bus above lower ring"


def test_public_mirror_requires_exact_physical_pml_and_keeps_primary_default():
    with pytest.raises(ValueError, match="divide the fixed 1.2-um PML"):
        br.build_public_mirror_simulation("isolated", dl_um=0.07, n_steps=20)

    primary, primary_meta = br.build_simulation(
        "isolated", dl_um=0.10, n_steps=20,
    )
    assert primary.symmetry == (0, 0, 0)
    assert primary.size_um != br.PUBLIC_MIRROR_HALF_SIZE_UM
    assert "independent_public_input_variant" not in primary_meta


def test_coarse_smoke_grid_grows_top_air_to_keep_sources_out_of_pml():
    sim, meta = br.build_simulation(
        "isolated",
        dl_um=0.15,
        n_steps=20,
        pml_num_layers=8,
        radiation_buffer_um=0.30,
        top_air_um=0.50,
    )
    assert meta["requested_top_air_um"] == pytest.approx(0.50)
    assert meta["realized_top_air_um"] == pytest.approx(1.50)
    assert meta["core_to_top_pml_clearance_um"] == pytest.approx(0.30)
    source_z = sim.sources[0].center_um[2]
    assert source_z < meta["top_pml_start_z_um"]
    assert all(monitor.center_um[2] < meta["top_pml_start_z_um"]
               for monitor in sim.monitors)


def test_builder_rejects_lower_pml_that_reaches_patterned_stack():
    with pytest.raises(ValueError, match="z-PML reaches"):
        br.build_simulation(
            "isolated", dl_um=0.20, n_steps=20, pml_num_layers=13,
        )


def test_ring_zero_has_identical_grid_phase_across_local_cases():
    built = {
        case: br.build_simulation(case, dl_um=0.08, n_steps=20)[1]
        for case in br.CASES
    }
    reference_center = built["isolated"]["ring_centers_um"][0]
    reference_phase = built["isolated"]["ring0_grid_phase_xy"]
    for case in br.CASES:
        assert built[case]["ring_centers_um"][0] == pytest.approx(reference_center)
        assert built[case]["ring0_grid_phase_xy"] == pytest.approx(reference_phase)


def test_constant_runtime_step_scaling_and_figure3_wavelength_alignment():
    assert br.INDEX_REFERENCE_WAVELENGTH_UM == pytest.approx(1.550)
    assert br.DEFAULT_WAVELENGTH_UM == pytest.approx(1.5688)
    coarse_steps = br.steps_for_run_time(0.08, 120.0)
    fine_steps = br.steps_for_run_time(0.04, 120.0)
    assert fine_steps == pytest.approx(2 * coarse_steps, abs=1)
    for dl, steps in ((0.08, coarse_steps), (0.04, fine_steps)):
        duration_ps = steps * br.approximate_dt_s(dl) * 1e12
        assert 120.0 <= duration_ps < 120.0 + br.approximate_dt_s(dl) * 1e12
    _, meta = br.build_simulation("isolated", dl_um=0.08, n_steps=20)
    assert meta["analysis_wavelength_um"] == pytest.approx(1.5688)
    assert meta["material_index_reference_wavelength_um"] == pytest.approx(1.550)


def _probe(signal, dt=1.0):
    t = np.arange(len(signal), dtype=float) * dt
    return xr.DataArray(
        np.asarray(signal)[:, None], dims=("t", "component"),
        coords={"t": t, "component": ["Ey"]},
    )


def test_synthetic_simulationdata_resonances_and_spacing_table():
    t = np.arange(5000, dtype=float)
    modes = [(0.120, 0.0012, 1.0), (0.190, 0.0007, 0.65)]
    signal = sum(a * np.exp(-alpha * t) * np.cos(2 * np.pi * f * t)
                 for f, alpha, a in modes)
    found = br.extract_resonances(
        {"ringdown_0": _probe(signal)}, "ringdown_0",
        freq_window_hz=(0.08, 0.23), drop_fraction=0.0,
        init_num_freqs=90, min_q=1.0,
    )
    table = br.resonance_table(found)
    assert len(table) >= 2
    for f, alpha, _ in modes:
        row = min(table, key=lambda r: abs(r["frequency_hz"] - f))
        assert row["frequency_hz"] == pytest.approx(f, rel=2e-3)
        assert row["q"] == pytest.approx(np.pi * f / alpha, rel=0.03)
    assert any(row["spacing_to_next_hz"] is not None for row in table)


def test_extract_resonances_physical_gate_overrides_trace_fraction(monkeypatch):
    captured_times = []

    class FakeFinder:
        def __init__(self, **kwargs):
            pass

        def run(self, sim_data, monitor_names, fields=None):
            captured_times.append(np.asarray(
                sim_data[monitor_names[0]].coords["t"].values
            ))
            return xr.Dataset(
                data_vars={
                    "Q": ("freq", [1000.0]),
                    "decay": ("freq", [-1.0]),
                    "amplitude": ("freq", [1.0]),
                },
                coords={"freq": [0.15]},
            )

    monkeypatch.setattr(br, "ResonanceFinder", FakeFinder)
    monkeypatch.setattr(
        br, "select_resonances", lambda resonances, **kwargs: resonances
    )
    times = np.arange(10, dtype=float) * 1e-12
    probe = xr.DataArray(
        np.ones((times.size, 1)), dims=("t", "component"),
        coords={"t": times, "component": ["Ey"]},
    )

    br.extract_resonances(
        {"ringdown_0": probe}, "ringdown_0",
        freq_window_hz=(0.1, 0.2), drop_fraction=0.9,
        gate_start_s=3.5e-12,
    )

    assert len(captured_times) == 1
    assert captured_times[0][0] == pytest.approx(4.0e-12)
    assert captured_times[0][-1] == pytest.approx(9.0e-12)


@pytest.mark.parametrize("gate_start_s", [-1.0, math.nan, math.inf])
def test_extract_resonances_rejects_invalid_physical_gate(gate_start_s):
    with pytest.raises(ValueError, match="gate_start_s"):
        br.extract_resonances(
            {"ringdown_0": _probe(np.ones(20))}, "ringdown_0",
            freq_window_hz=(0.1, 0.2), gate_start_s=gate_start_s,
        )


def test_extract_resonances_trims_phsolver_forced_final_sample():
    regular_t = np.arange(5000, dtype=float)
    t = np.r_[regular_t, regular_t[-1] + 0.25]
    signal = np.exp(-0.0008 * t) * np.cos(2 * np.pi * 0.155 * t)
    probe = xr.DataArray(
        signal[:, None], dims=("t", "component"),
        coords={"t": t, "component": ["Ey"]},
        attrs={"sample_steps": list(range(1, len(t) + 1))},
    )

    found = br.extract_resonances(
        {"ringdown_0": probe}, "ringdown_0",
        freq_window_hz=(0.12, 0.19), fields=("Ey",),
        drop_fraction=0.0, init_num_freqs=60, min_q=1.0,
    )

    table = br.resonance_table(found)
    strongest = max(table, key=lambda row: row["amplitude"])
    assert strongest["frequency_hz"] == pytest.approx(0.155, rel=2e-3)
    assert probe.sizes["t"] == len(t)  # preprocessing does not mutate input


def test_forced_final_sample_trim_keeps_entire_regular_prefix():
    sample_steps = [4, 8, 12, 16, 19]
    t = np.asarray(sample_steps, dtype=float) * 0.25
    probe = xr.DataArray(
        np.arange(len(t)), dims=("t",), coords={"t": t},
        attrs={"sample_steps": sample_steps},
    )

    sanitized = br._trim_forced_final_monitor_sample(probe)

    np.testing.assert_array_equal(sanitized.values, probe.values[:-1])
    np.testing.assert_array_equal(sanitized.coords["t"], probe.coords["t"][:-1])
    assert sanitized.attrs["sample_steps"] == sample_steps[:-1]


@pytest.mark.parametrize(
    "t",
    [
        np.arange(8, dtype=float),
        np.array([0.0, 1.0, 2.0, 2.5, 4.0, 5.0]),
        np.array([0.0, 1.0, 2.0, 4.0]),
    ],
    ids=("uniform", "interior-irregularity", "long-terminal-gap"),
)
def test_forced_final_sample_trim_preserves_nonmatching_traces(t):
    probe = xr.DataArray(np.arange(t.size), dims=("t",), coords={"t": t})
    sanitized = br._trim_forced_final_monitor_sample(probe)
    xr.testing.assert_identical(sanitized, probe)


def test_opposite_phase_probes_cannot_cancel_a_ring_pair_pole():
    t = np.arange(5000, dtype=float)
    signal = np.exp(-0.0008 * t) * np.cos(2 * np.pi * 0.155 * t)
    found = br.extract_resonances(
        {"ringdown_0": _probe(signal), "ringdown_1": _probe(-signal)},
        ("ringdown_0", "ringdown_1"),
        freq_window_hz=(0.12, 0.19),
        fields=("Ey",),
        drop_fraction=0.0,
        init_num_freqs=60,
        min_q=1.0,
    )
    table = br.resonance_table(found)
    assert table
    strongest = max(table, key=lambda row: row["amplitude"])
    assert strongest["frequency_hz"] == pytest.approx(0.155, rel=2e-3)


def test_local_cell_shifts_form_filter_cifs_and_split_proxies():
    f0 = 193.4e12

    def row(freq, q, amplitude=1.0):
        return {"frequency_hz": freq, "q": q, "amplitude": amplitude}

    result = br.analyze_coupling_cases({
        "isolated": [row(f0, 30_000.0)],
        "ring_bus": [row(f0 + 7e9, 10_000.0)],
        # Mean ring-neighbour shift is +50 GHz; half splitting is 30 GHz.
        "ring_pair": [row(f0 + 20e9, 20_000.0), row(f0 + 80e9, 20_000.0)],
    })
    assert result["ring_bus_self_shift_ghz"] == pytest.approx(7.0)
    assert result["ring_pair_mean_self_shift_ghz"] == pytest.approx(50.0)
    assert result["ring_pair_half_splitting_ghz"] == pytest.approx(30.0)
    assert result["predicted_center_minus_outer_cifs_ghz"] == pytest.approx(43.0)
    assert result["predicted_filter_cifs_magnitude_ghz"] == pytest.approx(43.0)
    assert result["predicted_cifs_magnitude_error_ghz"] == pytest.approx(0.0)
    assert result["ring_bus_external_q_proxy"] == pytest.approx(15_000.0)


def test_coupling_run_rows_can_lock_to_figure3_longitudinal_family():
    figure3 = br.FIGURE3_CENTER_FREQUENCY_HZ
    strongest_family = figure3 - 1.0e12

    def row(freq, amplitude):
        return {"frequency_hz": freq, "q": 20_000.0,
                "amplitude": amplitude}

    rows = [
        {
            "case": "isolated", "dl_um": 0.1,
            "resonances": [
                row(strongest_family, 10.0), row(figure3 + 2e9, 1.0),
            ],
        },
        {
            "case": "ring_bus", "dl_um": 0.1,
            "resonances": [
                row(strongest_family + 5e9, 10.0),
                row(figure3 + 9e9, 1.0),
            ],
        },
        {
            "case": "ring_pair", "dl_um": 0.1,
            "resonances": [
                row(strongest_family - 20e9, 10.0),
                row(strongest_family + 40e9, 10.0),
                row(figure3 - 18e9, 1.0), row(figure3 + 42e9, 1.0),
            ],
        },
    ]

    default = br.analyze_coupling_run_rows(rows)[0]
    locked = br.analyze_coupling_run_rows(
        rows, reference_frequency_hz=figure3
    )[0]

    assert default["reference_frequency_hz"] == pytest.approx(strongest_family)
    assert locked["reference_frequency_hz"] == pytest.approx(figure3 + 2e9)
    assert locked["ring_bus_frequency_hz"] == pytest.approx(figure3 + 9e9)
    assert locked["ring_pair_low_frequency_hz"] == pytest.approx(
        figure3 - 18e9
    )
    assert locked["ring_pair_high_frequency_hz"] == pytest.approx(
        figure3 + 42e9
    )


def test_lossless_three_ring_cmt_conserves_energy():
    f0 = 193.4e12
    f = np.linspace(f0 - 150e9, f0 + 150e9, 601)
    result = br.cmt_spectrum(
        f, resonance_hz=f0, q_outer=math.inf, q_center=math.inf,
        inter_ring_coupling_ghz=(19.0, 27.0),
        external_coupling_ghz=(16.0, 11.0),
    )
    total = result["through_power"] + result["drop_power"]
    assert np.max(np.abs(total - 1.0)) < 2e-10
    assert np.all(result["absorbed_power"] < 2e-10)
    assert result["metadata"]["figure_3_whole_filter_fdtd"] is False
    assert "not_paper" in result["metadata"]["coupling_provenance"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inter_ring_coupling_ghz": math.nan},
        {"external_coupling_ghz": 0.0},
        {"q_outer": -math.inf},
        {"center_detuning_ghz": math.nan},
    ],
)
def test_cmt_rejects_nonfinite_or_degenerate_parameters(kwargs):
    with pytest.raises(ValueError):
        br.cmt_spectrum([193.3e12, 193.4e12, 193.5e12], **kwargs)


def test_spectral_metrics_extract_loss_bandwidth_rejection_and_fsr():
    f = np.linspace(190e12, 196e12, 12001)
    resonances = np.array([191.5e12, 194.5e12])
    half_width = 20e9
    drop = sum(0.5 / (1.0 + ((f - fr) / half_width) ** 2)
               for fr in resonances)
    through = np.clip(1.0 - drop, 0.0, None)
    metrics = br.spectral_metrics(f, drop, through)
    expected_bw = 2.0 * half_width * math.sqrt(10.0**0.1 - 1.0) / 1e9
    assert metrics["drop_loss_db"] == pytest.approx(3.01, abs=0.02)
    assert metrics["bandwidth_1db_ghz"] == pytest.approx(expected_bw, rel=0.03)
    assert metrics["inband_through_rejection_db"] == pytest.approx(3.01, abs=0.03)
    assert metrics["fsr_ghz"] == pytest.approx(3000.0, rel=2e-3)
    assert metrics["resonance_count"] == 2


def test_cmt_paper_calibration_matches_targets_with_physical_q():
    calibrated = br.calibrate_cmt_to_paper()
    params = calibrated["fitted_parameters"]
    metrics = calibrated["metrics"]

    assert calibrated["least_squares"]["success"] is True
    assert "unpublished_couplings_and_phases" in calibrated["label"]
    assert "not an independent prediction" in calibrated["interpretation"]
    assert params["external_rate_definition"] == "(gamma_e / 2 pi) / 1e9"
    assert params["external_q_equivalent"] == pytest.approx(
        params["resonance_hz"]
        / (2.0 * params["external_coupling_ghz"] * 1e9)
    )
    assert params["inter_ring_coupling_ghz"] > 0.0
    assert params["external_coupling_ghz"] > 0.0
    assert 5_000.0 < params["q_outer"] < 30_000.0
    assert 5_000.0 < params["q_center"] < 30_000.0
    assert metrics["drop_loss_db"] == pytest.approx(3.0, abs=0.01)
    assert metrics["bandwidth_1db_ghz"] == pytest.approx(88.0, abs=0.05)
    assert metrics["inband_through_rejection_db"] == pytest.approx(
        7.5, abs=0.01
    )
    assert "worst through-port" in metrics["inband_rejection_definition"]
    assert "reconstruction convention" in (
        metrics["inband_rejection_definition_provenance"]
    )
    assert len(calibrated["response"]["frequency_hz"]) == 6001
    assert len(calibrated["normalized_residuals"]) == len(
        calibrated["residual_labels"]
    )


def test_asymmetric_bend_cross_section_and_pure_conversions():
    eps, meta = br.build_bend_cross_section(dl_um=0.025)
    assert eps.shape == tuple(meta["shape"])
    regions = meta["regions"]
    assert regions["bulk_sio2_y_max_um"] == pytest.approx(-0.126)
    assert regions["sio2_pedestal_y_bounds_um"] == pytest.approx([-0.126, 0.0])
    assert regions["sin_y_bounds_um"] == pytest.approx([0.0, 0.314])
    x, y = meta["x_um"], meta["y_um"]
    ix0 = int(np.argmin(abs(x)))
    ix_air = int(np.argmin(abs(x - 1.0)))
    iy_bulk = int(np.argmin(abs(y + 0.5)))
    iy_ped = int(np.argmin(abs(y + 0.06)))
    iy_core = int(np.argmin(abs(y - 0.15)))
    iy_air = int(np.argmin(abs(y - 0.7)))
    assert eps[iy_bulk, ix_air] == pytest.approx(1.455**2)
    assert eps[iy_ped, ix0] == pytest.approx(1.455**2)
    assert eps[iy_ped, ix_air] == pytest.approx(1.0)
    assert eps[iy_core, ix0] == pytest.approx(2.217**2)
    assert eps[iy_air, ix0] == pytest.approx(1.0)

    loss = 100.0
    expected_quarter = loss * (math.pi * 6.775 / 2.0) / 1e4
    assert br.bend_loss_db_per_90deg(loss, 6.775) == pytest.approx(expected_quarter)
    assert math.isinf(br.radiation_q_from_loss(0.0, 2.2))
    assert 20.0 < br.fsr_from_group_index_nm(2.2) < 30.0


def test_bend_mode_filter_tracks_sin_te0_and_exposes_window_loss_sensitivity():
    results = [
        br.solve_bend_modes(
            dl_um=0.05, num_modes=2, window_width_um=5.0,
            window_height_um=height,
        )
        for height in (3.5, 4.5)
    ]
    fundamentals = [
        next(mode for mode in result["modes"] if mode["loss_reference_mode"])
        for result in results
    ]
    assert all(mode["te_fraction"] >= 0.90 for mode in fundamentals)
    assert all(
        mode["core_electric_energy_fraction"] >= 0.30
        for mode in fundamentals
    )
    assert max(mode["n_eff"] for mode in fundamentals) - min(
        mode["n_eff"] for mode in fundamentals
    ) < 0.003
    assert all(22.0 < mode["fsr_nm"] < 26.0 for mode in fundamentals)
    # The real index is stable while the tiny imaginary part is not.  This is
    # why the tutorial labels Q as provisional pending window/PML convergence.
    q_values = [mode["q_rad"] for mode in fundamentals]
    assert abs(q_values[1] / q_values[0] - 1.0) > 0.05
    assert all(
        mode["q_rad"] is None
        for result in results
        for mode in result["modes"]
        if not mode["loss_reference_mode"]
    )


def test_committed_public_mirror_mi300x_baseline_is_self_consistent():
    evidence = json.loads(_PUBLIC_MIRROR_GPU.read_text())

    assert evidence["schema_version"] == 1
    assert evidence["benchmark"] == (
        "barwicz_2004_microring_public_mirror_gpu"
    )
    assert evidence["status"] == (
        "single_grid_single_time_window_baseline_not_converged"
    )
    assert "not converged" in evidence["interpretation"].lower()

    input_provenance = evidence["input_provenance"]
    assert input_provenance["scope"] == "public_paper_values_only"
    assert input_provenance["private_source_exported"] is False
    assert input_provenance["nominal_bus_width_um"] == pytest.approx(1.050)

    variant = evidence["simulation_variant"]
    assert variant["name"] == (
        "independent_public_input_mirror_reduced_local_cell"
    )
    assert variant["dimensionality"] == "3D"
    assert variant["symmetry"] == [1, 0, 0]
    assert variant["domain_fraction"] == pytest.approx(0.5)
    assert variant["full_asymmetric_z_stack"] is True
    assert variant["primary_geometry_parity_claim"] is False

    environment = evidence["execution_environment"]
    assert "MI300X" in environment["device_name"]
    assert environment["gpu_arch"] == "gfx942"
    assert environment["backend"].lower() == "hip"
    assert evidence["run_time_ps"] == pytest.approx(60.0)

    rows = evidence["rows"]
    assert len(rows) == len(br.CASES)
    assert {row["case"] for row in rows} == set(br.CASES)
    assert {float(row["dl_um"]) for row in rows} == {0.10}
    assert len({int(row["steps_run"]) for row in rows}) == 1
    assert next(iter({int(row["steps_run"]) for row in rows})) > 0
    for row in rows:
        assert "MI300X" in row["device"]
        assert row["meta"]["symmetry"] == [1, 0, 0]
        assert row["meta"]["domain_fraction"] == pytest.approx(0.5)
        assert row["meta"]["full_asymmetric_z_stack"] is True
        assert row["resonances"]
        for resonance in row["resonances"]:
            for key in (
                "frequency_hz", "wavelength_nm", "q", "decay_per_s",
                "amplitude",
            ):
                assert math.isfinite(float(resonance[key]))
            assert resonance["frequency_hz"] > 0.0
            assert resonance["wavelength_nm"] > 0.0
            assert resonance["q"] > 0.0
            if "error" in resonance:
                assert math.isfinite(float(resonance["error"]))

    validation = evidence["validation"]
    for key in (
        "spatial_convergence",
        "time_window_convergence",
        "primary_geometry_parity",
        "mode_identity_validation",
    ):
        assert validation[key] is False

    assert isinstance(evidence["fit_window_robustness"], dict)
    recomputed = br.analyze_coupling_run_rows(
        rows, reference_frequency_hz=br.FIGURE3_CENTER_FREQUENCY_HZ,
    )
    committed = evidence["coupling_analysis"]
    assert len(recomputed) == len(committed) == 1
    for key, expected in recomputed[0].items():
        assert key in committed[0]
        actual = committed[0][key]
        if isinstance(expected, (int, float)):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected
    # Deliberately no closeness assertion against the paper's 43 GHz CIFS:
    # one grid and one time window cannot validate that physical comparison.


def test_cli_all_dry_run_and_missing_solver_are_explicit(tmp_path, monkeypatch):
    output = tmp_path / "dry.json"
    assert br.main([
        "--case", "all", "--dl", "0.10", "0.08",
        "--run-time-ps", "12", "--dry-run", "--out", str(output),
    ]) == 0
    payload = json.loads(output.read_text())
    assert payload["case"] == "all"
    assert len(payload["cases"]) == len(br.CASES)
    assert all(
        case["time_sampling_policy"]
        == "constant approximate runtime 12 ps"
        for case in payload["cases"]
    )

    monkeypatch.setattr(br, "find_solver", lambda: None)
    with pytest.raises(SystemExit) as error:
        br.main(["--case", "isolated", "--dl", "0.10"])
    assert error.value.code == 2

    with pytest.raises(SystemExit) as error:
        br.main(["--self-test", "--dry-run"])
    assert error.value.code == 2


def test_cli_public_mirror_is_explicit_and_serializes_all_cells(tmp_path):
    output = tmp_path / "public-mirror.json"
    assert br.main([
        "--public-mirror", "--case", "all", "--dl", "0.10",
        "--steps", "20", "--dry-run", "--out", str(output),
    ]) == 0
    payload = json.loads(output.read_text())
    assert payload["case"] == "all"
    assert len(payload["cases"]) == len(br.CASES)
    for case_payload in payload["cases"]:
        assert case_payload["variant"] == (
            "independent_public_input_mirror_reduced_local_cell"
        )
        assert case_payload["parity_claim"] is False
        assert case_payload["convergence_claim"] is False
        row = case_payload["rows"][0]
        assert row["wire_valid"] is True
        assert row["meta"]["size_um"] == pytest.approx((10.3, 35.5, 5.0))
        assert row["meta"]["symmetry"] == [1, 0, 0]


def test_run_one_public_mirror_uses_fixed_physical_gate(monkeypatch):
    class FakeData(dict):
        steps_run = 100
        provenance = {"device_name": "fake-gpu", "backend": "test"}

    found = xr.Dataset(
        data_vars={
            "Q": ("freq", [1000.0]),
            "decay": ("freq", [-1.0]),
            "amplitude": ("freq", [1.0]),
        },
        coords={"freq": [br.FIGURE3_CENTER_FREQUENCY_HZ]},
    )
    calls = []

    def fake_extract(*args, **kwargs):
        calls.append(kwargs)
        return found

    monkeypatch.setattr(
        br.ph, "run_local", lambda simulation, device: FakeData()
    )
    monkeypatch.setattr(br, "extract_resonances", fake_extract)

    public = br.run_one(
        "isolated", 0.10, 100, "gpu", public_mirror=True
    )
    primary = br.run_one(
        "isolated", 0.10, 100, "gpu", public_mirror=False
    )

    assert calls[0]["gate_start_s"] == pytest.approx(5.0e-12)
    assert calls[0]["drop_fraction"] == pytest.approx(0.15)
    assert public["meta"]["analysis_gate"] == {
        "mode": "physical_start_time",
        "gate_start_s": pytest.approx(5.0e-12),
        "gate_start_ps": pytest.approx(5.0),
        "drop_fraction": None,
    }
    assert calls[1]["gate_start_s"] is None
    assert calls[1]["drop_fraction"] == pytest.approx(0.15)
    assert primary["meta"]["analysis_gate"] == {
        "mode": "fraction_of_trace",
        "gate_start_s": None,
        "drop_fraction": pytest.approx(0.15),
    }


def test_engine_smoke_executes_monitors_without_resonance_claim(
    tmp_path, monkeypatch,
):
    class FakeData(dict):
        steps_run = 100
        provenance = {"device_name": "fake-gpu", "backend": "test"}

    fake_data = FakeData({"ringdown_0": object()})
    monkeypatch.setattr(br.ph, "run_local", lambda simulation, device: fake_data)
    monkeypatch.setattr(br, "find_solver", lambda: Path("/fake/phsolver"))

    row = br.run_engine_smoke("isolated", 0.10, 100, "gpu")
    assert row["steps_run"] == 100
    assert row["monitor_payloads"] == ["ringdown_0"]
    assert row["device"] == "fake-gpu"
    assert row["physics_claim"] is False
    assert "no resonance" in row["interpretation"]

    output = tmp_path / "smoke.json"
    assert br.main([
        "--engine-smoke", "--case", "isolated", "--dl", "0.10",
        "--steps", "100", "--device", "gpu", "--out", str(output),
    ]) == 0
    payload = json.loads(output.read_text())
    assert payload["mode"] == "engine_smoke"
    assert payload["physics_claim"] is False
    assert payload["rows"][0]["monitor_payloads"] == ["ringdown_0"]
