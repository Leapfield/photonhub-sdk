"""Workbench modal-port service and HTTP contracts."""

import importlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from photonhub.components.monitors import FieldDftMonitor, ModePort, PortMode
from photonhub.components.source_time import GaussianPulse
from photonhub.components.sources import ModeSolveProvenance, ModeSource
from photonhub.viz import service
from photonhub.viz.ledger import RunLedger


FREQS = (190.0e12, 200.0e12)
FIELDS_X = ("Ey", "Ez", "Hy", "Hz")


def _source() -> ModeSource:
    provenance = ModeSolveProvenance(
        polarization="TE",
        mode_index=0,
        wavelength_um=1.55,
        center_um=(1.0, 0.5),
        size_um=(2.0, 1.0),
        dl_um=0.05,
        supersample=4,
        input_sha256="0" * 64,
    )
    return ModeSource(
        axis="x",
        direction="+",
        position_um=0.25,
        polarization="Ey",
        n_eff=2.5,
        nu=1,
        nv=1,
        profile=(1.0,),
        source_time=GaussianPulse(freq0_hz=195.0e12, fwidth_hz=10.0e12),
        mode_solve=provenance,
    )


def _port_monitor(
    name: str,
    position_um: float,
    out_direction: str,
    modes,
    *,
    source_index=None,
) -> FieldDftMonitor:
    return FieldDftMonitor(
        name=name,
        center_um=(position_um, 1.0, 0.5),
        size_um=(0.0, 2.0, 1.0),
        fields=FIELDS_X,
        freqs_hz=FREQS,
        mode_port=ModePort(
            out_direction=out_direction,
            center_um=(1.0, 0.5),
            size_um=(2.0, 1.0),
            dl_um=0.05,
            supersample=4,
            modes=tuple(PortMode(polarization=pol, mode_index=index)
                        for pol, index in modes),
            source_index=source_index,
            thickness_axis="z",
        ),
    )


def test_modal_results_build_virtual_channels_and_match_ui_contract(monkeypatch):
    monitors = (
        _port_monitor("input", 0.5, "-", (("TE", 0),), source_index=0),
        _port_monitor("output", 3.5, "+", (("TE", 0), ("TE", 1))),
    )
    sim = SimpleNamespace(
        monitors=monitors,
        sources=(_source(),),
        symmetry=(0, 0, 0),
    )
    data = SimpleNamespace(manifest={
        "monitors": [
            {"name": "input", "file": "input.bin",
             "freqs_hz": list(FREQS)},
            {"name": "output", "file": "output.bin",
             "freqs_hz": list(FREQS)},
        ],
    })
    monkeypatch.setattr(service, "sim_for", lambda _data: sim)
    monkeypatch.setattr(service, "mode_source_statuses", lambda _sim: [{
        "source_index": 0,
        "status": "fresh",
    }])

    solve_calls = []

    def fake_solve(_sim, axis, position_um, freqs_hz, *, modes, **settings):
        solve_calls.append((axis, position_um, tuple(freqs_hz), tuple(modes), settings))
        return {
            float(frequency): {
                mode: SimpleNamespace(
                    polarization=mode[0],
                    mode_index=mode[1],
                    n_eff=2.5 - 0.1 * mode[1],
                )
                for mode in modes
            }
            for frequency in freqs_hz
        }

    yee_module = importlib.import_module("photonhub.plugins.yee_mode")
    grid_module = importlib.import_module("photonhub.components.grid")
    smatrix_module = importlib.import_module("photonhub.plugins.smatrix")
    monkeypatch.setattr(yee_module, "solve_yee_port_mode_bank", fake_solve)
    monkeypatch.setattr(
        grid_module,
        "snap_mixed_plane",
        lambda _sim, _axis_index, position_um: (position_um, 0.04),
    )

    captured = {}

    def fake_smatrix(ports, driven, seen_data):
        captured.update(ports=list(ports), driven=driven, data=seen_data)
        values = {
            "port0:mode0": 0.1 + 0.2j,
            "port1:mode0": 0.8 + 0.0j,
            "port1:mode1": 0.0 + 0.5j,
        }
        return {
            (port.name, driven): {
                frequency: values[port.name] for frequency in FREQS
            }
            for port in ports
        }

    monkeypatch.setattr(smatrix_module, "smatrix", fake_smatrix)

    response = service.modal_port_results(data)

    assert [call[:4] for call in solve_calls] == [
        ("x", 0.5, FREQS, (("TE", 0),)),
        ("x", 3.5, FREQS, (("TE", 0), ("TE", 1))),
    ]
    assert captured["data"] is data
    assert captured["driven"] == "port0:mode0"
    assert [port.name for port in captured["ports"]] == [
        "port0:mode0", "port1:mode0", "port1:mode1",
    ]
    assert [port.monitor.name for port in captured["ports"]] == [
        "input", "output", "output",
    ]
    assert all(port.monitor.center_um == (1.0, 0.5)
               for port in captured["ports"])
    assert all(port.monitor.dl_um == pytest.approx(0.04)
               for port in captured["ports"])

    assert response["driven_port"] == "input"
    assert response["driven_mode"] == {"polarization": "TE", "mode_index": 0}
    assert response["reference_plane"] == (
        "recorded monitor planes; no phase de-embedding")
    assert [port["name"] for port in response["ports"]] == ["input", "output"]
    assert response["ports"][0]["out_direction"] == "-"
    assert response["ports"][1]["out_direction"] == "+"

    output_modes = response["ports"][1]["modes"]
    assert [(sample["polarization"], sample["mode_index"])
            for sample in output_modes] == [
        ("TE", 0), ("TE", 0), ("TE", 1), ("TE", 1),
    ]
    assert [sample["wavelength_nm"] for sample in output_modes[:2]] == sorted(
        299_792_458.0 / frequency * 1e9 for frequency in FREQS)
    assert output_modes[0]["power"] == pytest.approx(0.64)
    assert output_modes[0]["db"] == pytest.approx(10.0 * math.log10(0.64))
    assert output_modes[2]["power"] == pytest.approx(0.25)
    assert output_modes[2]["phase_deg"] == pytest.approx(90.0)
    assert (output_modes[2]["s_re"], output_modes[2]["s_im"]) == (0.0, 0.5)

    assert service.modal_port_monitor_names(data) == ["input", "output"]
    expected_summaries = [
        {
            "name": "input",
            "monitor_name": "input",
            "axis": "x",
            "position_um": 0.5,
            "out_direction": "-",
            "polarization": "TE",
            "mode_indices": [0],
            "frequency_count": 2,
            "modes": [{"polarization": "TE", "mode_index": 0}],
            "source_index": 0,
        },
        {
            "name": "output",
            "monitor_name": "output",
            "axis": "x",
            "position_um": 3.5,
            "out_direction": "+",
            "polarization": "TE",
            "mode_indices": [0, 1],
            "frequency_count": 2,
            "modes": [
                {"polarization": "TE", "mode_index": 0},
                {"polarization": "TE", "mode_index": 1},
            ],
            "source_index": None,
        },
    ]
    assert service.modal_port_summaries_from_sim(sim) == expected_summaries

    session_data = SimpleNamespace(
        output_dir=Path("/modal-result"),
        manifest={"run": {}, "grid": {}, "provenance": {}, "monitors": []},
        aborted=False,
        abort_reason=None,
    )
    monkeypatch.setattr(
        service, "geometry_status", lambda _data: {"status": "matched"})
    assert service.session(session_data, result_id="revision")["ports"] == \
        expected_summaries


def test_modal_results_reject_mismatched_frequency_grids_before_solving(
    monkeypatch,
):
    input_monitor = _port_monitor(
        "input", 0.5, "-", (("TE", 0),), source_index=0)
    output_monitor = _port_monitor(
        "output", 3.5, "+", (("TE", 0),)).model_copy(
            update={"freqs_hz": (FREQS[0],)})
    sim = SimpleNamespace(
        monitors=(input_monitor, output_monitor),
        sources=(_source(),),
        symmetry=(0, 0, 0),
    )
    data = SimpleNamespace(manifest={
        "monitors": [{"name": "input"}, {"name": "output"}],
    })
    monkeypatch.setattr(service, "sim_for", lambda _data: sim)

    yee_module = importlib.import_module("photonhub.plugins.yee_mode")
    solve = pytest.fail
    monkeypatch.setattr(yee_module, "solve_yee_port_mode_bank", solve)

    with pytest.raises(ValueError, match="exact frequency grid.*output"):
        service.modal_port_results(data)


def test_modal_results_reject_artifact_frequency_grid_mismatched_to_sim(
    monkeypatch,
):
    monitors = (
        _port_monitor("input", 0.5, "-", (("TE", 0),), source_index=0),
        _port_monitor("output", 3.5, "+", (("TE", 0),)),
    )
    sim = SimpleNamespace(
        monitors=monitors,
        sources=(_source(),),
        symmetry=(0, 0, 0),
    )
    data = SimpleNamespace(manifest={"monitors": [
        {"name": "input", "freqs_hz": list(FREQS)},
        {"name": "output", "freqs_hz": [FREQS[0]]},
    ]})
    monkeypatch.setattr(service, "sim_for", lambda _data: sim)

    yee_module = importlib.import_module("photonhub.plugins.yee_mode")
    monkeypatch.setattr(
        yee_module, "solve_yee_port_mode_bank",
        lambda *_args, **_kwargs: pytest.fail(
            "artifact mismatch must fail before solving"),
    )

    with pytest.raises(
            ValueError, match="artifact frequency grids.*output"):
        service.modal_port_results(data)


def test_modal_readiness_rejects_additional_active_sources_and_wrong_side():
    monitor = _port_monitor(
        "input", 0.5, "-", (("TE", 0),), source_index=0)
    source = _source()
    second = source.model_copy(update={"position_um": 0.1})
    multiple = SimpleNamespace(
        monitors=(monitor,), sources=(source, second), symmetry=(0, 0, 0))
    with pytest.raises(ValueError, match="only active excitation"):
        service.assert_modal_ports_ready(multiple)

    behind = monitor.model_copy(update={"center_um": (0.1, 1.0, 0.5)})
    wrong_side = SimpleNamespace(
        monitors=(behind,), sources=(source,), symmetry=(0, 0, 0))
    with pytest.raises(ValueError, match="must lie downstream"):
        service.assert_modal_ports_ready(wrong_side)


def test_modal_readiness_requires_engine_normalization_source_zero():
    source = _source()
    inactive_first = source.model_copy(update={"amplitude": 0.0})
    monitor = _port_monitor(
        "input", 0.5, "-", (("TE", 0),), source_index=1)
    sim = SimpleNamespace(
        monitors=(monitor,),
        sources=(inactive_first, source),
        symmetry=(0, 0, 0),
    )

    with pytest.raises(
            ValueError, match="source_index 0.*first wire-order source"):
        service.assert_modal_ports_ready(sim)


def test_modal_readiness_caps_highest_index_auto_trials(monkeypatch):
    monitor = _port_monitor(
        "input", 0.5, "-", (("TE", 0), ("TE", 31)), source_index=0)
    sim = SimpleNamespace(
        monitors=(monitor,),
        sources=(_source(),),
        symmetry=(0, 0, 0),
        grid=object(),
    )
    grid_module = importlib.import_module("photonhub.components.grid")
    yee_module = importlib.import_module("photonhub.plugins.yee_mode")
    monkeypatch.setattr(
        grid_module,
        "snap_mixed_plane",
        lambda _sim, _axis_index, position_um: (position_um, 0.0),
    )
    # Seventeen synthetic solve cells make 32 trials feasible but would reject
    # the former uncapped automatic count of 34.
    monkeypatch.setattr(
        yee_module,
        "window_nodes",
        lambda *_args, **_kwargs: (
            np.arange(17), None, None, np.arange(1), None, None,
        ),
    )

    service.assert_modal_ports_ready(sim)


def test_y_normal_port_keeps_natural_solve_axes_and_right_handed_overlap(
    monkeypatch,
):
    monitor = FieldDftMonitor(
        name="input",
        center_um=(1.0, 0.5, 2.0),
        size_um=(2.0, 0.0, 4.0),
        fields=("Ez", "Ex", "Hz", "Hx"),
        freqs_hz=FREQS,
        mode_port=ModePort(
            out_direction="-",
            center_um=(1.0, 2.0),  # natural x,z editor order
            size_um=(2.0, 4.0),
            dl_um=0.05,
            # The solved source's natural TE family is physical TM when the
            # port declares natural-horizontal x as its thickness axis.
            modes=(PortMode(polarization="TM", mode_index=0),),
            source_index=0,
            thickness_axis="x",
        ),
    )
    source = _source().model_copy(update={
        "axis": "y", "direction": "+", "polarization": "Ex",
    })
    sim = SimpleNamespace(
        monitors=(monitor,), sources=(source,), symmetry=(0, 0, 0))
    data = SimpleNamespace(manifest={"monitors": [{
        "name": "input", "freqs_hz": list(FREQS),
    }]})
    monkeypatch.setattr(service, "sim_for", lambda _data: sim)
    monkeypatch.setattr(service, "mode_source_statuses", lambda _sim: [{
        "source_index": 0, "status": "fresh",
    }])

    captured = {}

    def fake_solve(_sim, axis, position_um, freqs_hz, *, modes, **settings):
        captured["solve"] = (axis, position_um, tuple(freqs_hz), modes, settings)
        return {
            float(frequency): {
                ("TM", 0): SimpleNamespace(polarization="TE", n_eff=2.5),
            }
            for frequency in freqs_hz
        }

    yee_module = importlib.import_module("photonhub.plugins.yee_mode")
    grid_module = importlib.import_module("photonhub.components.grid")
    smatrix_module = importlib.import_module("photonhub.plugins.smatrix")
    monkeypatch.setattr(yee_module, "solve_yee_port_mode_bank", fake_solve)
    monkeypatch.setattr(
        grid_module, "snap_mixed_plane",
        lambda _sim, _axis_index, position_um: (position_um + 0.0125, 0.04),
    )

    def fake_smatrix(ports, driven, _data):
        captured["monitor"] = ports[0].monitor
        return {(ports[0].name, driven): {frequency: 1.0 for frequency in FREQS}}

    monkeypatch.setattr(smatrix_module, "smatrix", fake_smatrix)
    response = service.modal_port_results(data)

    axis, position, frequencies, _, settings = captured["solve"]
    assert (axis, position, frequencies) == ("y", 0.5125, FREQS)
    assert settings["h_center_um"] == 1.0  # x in natural editor order
    assert settings["v_center_um"] == 2.0  # z in natural editor order
    assert settings["thickness_axis"] == "x"
    assert captured["solve"][3] == [("TM", 0)]
    assert response["driven_mode"] == {
        "polarization": "TM", "mode_index": 0,
    }
    assert response["ports"][0]["position_um"] == pytest.approx(0.5125)
    runtime = captured["monitor"]
    assert runtime.center_um == (2.0, 1.0)  # z,x right-handed overlap order
    # The physical thickness axis selects the TE/TM solve family, but the Yee
    # VectorMode itself remains rastered in natural (x,z) editor order.  The
    # overlap therefore maps its mode-y coordinate onto natural vertical z.
    assert runtime.thickness_axis == "z"


def test_y_normal_nondefault_thickness_projects_natural_asymmetric_vector_mode(
    monkeypatch,
):
    """A physical-axis TE/TM swap must not rotate the solved Yee raster."""
    from photonhub.plugins._constants import ETA0
    from photonhub.plugins.mode_overlap import vector_modal_fields
    from photonhub.plugins.vector_modes import VectorMode

    frequency = FREQS[0]
    wavelength_um = 299_792_458.0 / frequency * 1e6
    x_relative = (np.arange(23) - 11) * 0.08
    z_relative = (np.arange(15) - 7) * 0.05
    xx, zz = np.meshgrid(x_relative, z_relative, indexing="xy")
    # Deliberately off-centre and anisotropic: transposing/rotating this raster
    # cannot accidentally give the same overlap.
    profile = np.exp(
        -((xx - 0.19) ** 2 / 0.22 + (zz + 0.09) ** 2 / 0.025)
    ).astype(complex)
    ex = profile
    ey = (0.13 + 0.08j) * (1.0 + 0.4 * xx) * profile
    norm = np.sqrt(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
    ex /= norm
    ey /= norm
    n_eff = 2.3
    mode = VectorMode(
        n_eff=n_eff,
        n_group=None,
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex),
        hx=-(n_eff / ETA0) * ey,
        hy=(n_eff / ETA0) * ex,
        hz=np.zeros_like(ex),
        wavelength_um=wavelength_um,
        dl_x_um=0.08,
        dl_y_um=0.05,
        yee_staggered=True,
    )

    def port(name, position_um, out_direction, source_index=None):
        return FieldDftMonitor(
            name=name,
            center_um=(1.0, position_um, 2.0),
            size_um=(2.0, 0.0, 1.0),
            fields=("Ez", "Ex", "Hz", "Hx"),
            freqs_hz=(frequency,),
            mode_port=ModePort(
                out_direction=out_direction,
                center_um=(1.0, 2.0),  # natural editor axes: x,z
                size_um=(2.0, 1.0),
                dl_um=0.05,
                supersample=4,
                modes=(PortMode(polarization="TM", mode_index=0),),
                source_index=source_index,
                # Non-default physical slab axis swaps physical TM to the
                # solver's natural-horizontal TE family.
                thickness_axis="x",
            ),
        )

    monitors = (
        port("input", 0.5, "-", source_index=0),
        port("output", 3.5, "+"),
    )
    provenance = _source().mode_solve.model_copy(update={
        "center_um": (1.0, 2.0),
        "size_um": (2.0, 1.0),
    })
    source = _source().model_copy(update={
        "axis": "y",
        "direction": "+",
        "polarization": "Ex",
        "mode_solve": provenance,
    })
    sim = SimpleNamespace(
        monitors=monitors, sources=(source,), symmetry=(0, 0, 0))

    x_coords = x_relative + 1.0
    z_coords = z_relative + 2.0

    def plane(position_um, amplitude):
        # solve_yee_port_mode_bank's VectorMode axes are always natural
        # horizontal x / vertical z, regardless of the physical thickness axis.
        fields = vector_modal_fields(
            mode,
            z_coords,
            x_coords,
            axis="y",
            direction="+",
            center_um=(2.0, 1.0),  # overlap order for y normal: z,x
            thickness_axis="z",
        )
        # vector_modal_fields is [x,z] for a y-normal plane; SimulationData's
        # canonical spatial order is [z,y,x].
        components = [
            amplitude * fields[key].T
            for key in ("e1", "e2", "h1", "h2")
        ]
        values = np.stack(components)[None, :, :, None, :]
        return xr.DataArray(
            values,
            dims=("f", "component", "z", "y", "x"),
            coords={
                "f": [frequency],
                "component": ["Ez", "Ex", "Hz", "Hx"],
                "z": z_coords,
                "y": [position_um],
                "x": x_coords,
            },
        )

    class SyntheticData(dict):
        pass

    transmission = 0.47 - 0.21j
    data = SyntheticData(
        input=plane(0.5, 1.0 + 0.0j),
        output=plane(3.5, transmission),
    )
    data.manifest = {"monitors": [
        {"name": "input", "freqs_hz": [frequency]},
        {"name": "output", "freqs_hz": [frequency]},
    ]}
    monkeypatch.setattr(service, "sim_for", lambda _data: sim)
    monkeypatch.setattr(service, "mode_source_statuses", lambda _sim: [{
        "source_index": 0, "status": "fresh",
    }])

    yee_module = importlib.import_module("photonhub.plugins.yee_mode")
    grid_module = importlib.import_module("photonhub.components.grid")
    monkeypatch.setattr(
        yee_module,
        "solve_yee_port_mode_bank",
        lambda _sim, _axis, _position, freqs_hz, *, modes, **_settings: {
            float(item): {key: mode for key in modes} for item in freqs_hz
        },
    )
    monkeypatch.setattr(
        grid_module,
        "snap_mixed_plane",
        lambda _sim, _axis_index, position_um: (position_um, 0.0),
    )

    response = service.modal_port_results(data)
    transmitted = response["ports"][1]["modes"][0]
    assert response["driven_mode"] == {
        "polarization": "TM", "mode_index": 0,
    }
    assert transmitted["power"] == pytest.approx(
        abs(transmission) ** 2, rel=2e-3)
    assert transmitted["s_re"] == pytest.approx(transmission.real, rel=2e-3)
    assert transmitted["s_im"] == pytest.approx(transmission.imag, rel=2e-3)


def test_modal_results_projects_synthetic_fields_without_mocking_smatrix(
    monkeypatch,
):
    """Compact end-to-end pin for service -> ModeMonitor -> SPort overlap."""
    from photonhub.plugins._constants import ETA0
    from photonhub.plugins.mode_overlap import vector_modal_fields
    from photonhub.plugins.vector_modes import VectorMode

    frequency = FREQS[0]
    wavelength_um = 299_792_458.0 / frequency * 1e6
    grid = np.arange(21) - 10
    zz, yy = np.meshgrid(grid, grid, indexing="ij")
    profile = np.exp(-(yy * yy + zz * zz) / 18.0).astype(complex)
    profile /= np.sqrt(np.sum(np.abs(profile) ** 2))
    ex = profile
    ey = 0.2 * profile
    n_eff = 2.2
    mode = VectorMode(
        n_eff=n_eff,
        n_group=None,
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex),
        hx=-(n_eff / ETA0) * ey,
        hy=(n_eff / ETA0) * ex,
        hz=np.zeros_like(ex),
        wavelength_um=wavelength_um,
        dl_x_um=0.05,
        dl_y_um=0.05,
        yee_staggered=True,
    )
    monitors = (
        _port_monitor("input", 0.5, "-", (("TE", 0),), source_index=0)
        .model_copy(update={"freqs_hz": (frequency,)}),
        _port_monitor("output", 3.5, "+", (("TE", 0),))
        .model_copy(update={"freqs_hz": (frequency,)}),
    )
    sim = SimpleNamespace(
        monitors=monitors, sources=(_source(),), symmetry=(0, 0, 0))

    t1 = (np.arange(21) - 10) * 0.05 + 1.0  # y
    t2 = (np.arange(21) - 10) * 0.05 + 0.5  # z

    def plane(position, amplitude):
        fields = vector_modal_fields(
            mode, t1, t2, axis="x", direction="+",
            center_um=(1.0, 0.5), thickness_axis="z",
        )
        components = [
            amplitude * fields[key] for key in ("e1", "e2", "h1", "h2")
        ]
        values = np.stack(components)[None, :, :, :, None]
        return xr.DataArray(
            values,
            dims=("f", "component", "z", "y", "x"),
            coords={
                "f": [frequency], "component": list(FIELDS_X),
                "z": t2, "y": t1, "x": [position],
            },
        )

    class SyntheticData(dict):
        pass

    data = SyntheticData(
        input=plane(0.5, 1.0 + 0.0j),
        output=plane(3.5, 0.6 - 0.2j),
    )
    data.manifest = {
        "monitors": [
            {"name": "input", "freqs_hz": [frequency]},
            {"name": "output", "freqs_hz": [frequency]},
        ],
    }
    monkeypatch.setattr(service, "sim_for", lambda _data: sim)
    monkeypatch.setattr(service, "mode_source_statuses", lambda _sim: [{
        "source_index": 0, "status": "fresh",
    }])

    yee_module = importlib.import_module("photonhub.plugins.yee_mode")
    grid_module = importlib.import_module("photonhub.components.grid")
    monkeypatch.setattr(
        yee_module,
        "solve_yee_port_mode_bank",
        lambda _sim, _axis, _position, freqs_hz, *, modes, **_settings: {
            float(item): {key: mode for key in modes} for item in freqs_hz
        },
    )
    monkeypatch.setattr(
        grid_module, "snap_mixed_plane",
        lambda _sim, _axis_index, position_um: (position_um, 0.0),
    )

    response = service.modal_port_results(data)
    reflected = response["ports"][0]["modes"][0]
    transmitted = response["ports"][1]["modes"][0]
    assert reflected["power"] == pytest.approx(0.0, abs=1e-12)
    assert transmitted["power"] == pytest.approx(
        abs(0.6 - 0.2j) ** 2, rel=2e-3)
    assert transmitted["s_re"] == pytest.approx(0.6, rel=2e-3)
    assert transmitted["s_im"] == pytest.approx(-0.2, rel=2e-3)


def test_modal_endpoint_honors_revision_and_caches_decomposition(
    tmp_path, monkeypatch,
):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    assert fastapi is not None
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    (tmp_path / "manifest.json").write_text(json.dumps({
        "manifest_version": "1",
        "run": {},
        "grid": {},
        "provenance": {},
        "monitors": [],
    }), encoding="utf-8")
    (tmp_path / "sim.json").write_text("{}", encoding="utf-8")
    calls = []
    expected = {
        "driven_port": "input",
        "driven_mode": {"polarization": "TE", "mode_index": 0},
        "normalization": "test normalization",
        "reference_plane": "test plane",
        "ports": [],
    }
    monkeypatch.setattr(
        service, "modal_port_monitor_names", lambda _data: ["input", "output"])

    def fake_results(data):
        calls.append(data)
        return expected

    monkeypatch.setattr(service, "modal_port_results", fake_results)
    with TestClient(create_app(tmp_path)) as client:
        session_payload = client.get("/api/session").json()
        revision = session_payload["result_id"]
        first = client.get("/api/ports/modal", params={"rev": revision})
        second = client.get("/api/ports/modal", params={"rev": revision})
        stale = client.get("/api/ports/modal", params={"rev": "stale"})

    assert first.status_code == 200
    assert first.json() == expected
    assert second.status_code == 200
    assert second.json() == expected
    assert "ports" not in session_payload
    assert len(calls) == 1
    assert stale.status_code == 409
    assert "stale result revision" in stale.json()["detail"]


def _write_external_flux_bundle(path: Path, value: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.asarray([value], dtype="<f4").tofile(path / "port.bin")
    (path / "manifest.json").write_text(json.dumps({
        "manifest_version": "1",
        "run": {"dt_s": 1.0e-16},
        "grid": {"dl_um": 0.1},
        "provenance": {},
        "monitors": [{
            "name": "port",
            "type": "flux",
            "file": "port.bin",
            "dtype": "float32",
            "shape": [1],
            "dims": ["freq"],
            "components": [],
            "freqs_hz": [1.0],
            "axis": "x",
        }],
    }), encoding="utf-8")


def test_modal_endpoint_reloads_atomically_replaced_external_monitor(
    tmp_path, monkeypatch,
):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    _write_external_flux_bundle(tmp_path, 1.0)
    (tmp_path / "sim.json").write_text("{}", encoding="utf-8")
    calls = 0
    monkeypatch.setattr(
        service, "modal_port_monitor_names", lambda _data: ["port"])

    def fake_results(data):
        nonlocal calls
        calls += 1
        return {"value": float(data["port"].values[0])}

    monkeypatch.setattr(service, "modal_port_results", fake_results)
    with TestClient(create_app(tmp_path)) as client:
        revision = client.get("/api/session").json()["result_id"]
        assert client.get(
            "/api/ports/modal", params={"rev": revision}).json()["value"] == 1.0
        assert client.get(
            "/api/ports/modal", params={"rev": revision}).json()["value"] == 1.0
        assert calls == 1

        replacement = tmp_path / "port.next"
        np.asarray([2.0], dtype="<f4").tofile(replacement)
        os.replace(replacement, tmp_path / "port.bin")

        refreshed = client.get(
            "/api/ports/modal", params={"rev": revision})

    assert refreshed.status_code == 200
    assert refreshed.json()["value"] == 2.0
    assert calls == 2


def test_modal_endpoint_reloads_replaced_sim_with_preserved_mtime(
    tmp_path, monkeypatch,
):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    _write_external_flux_bundle(tmp_path, 1.0)
    sim_a = service.default_sim()
    spec_b = sim_a.to_wire_dict()
    spec_b["size_um"][0] = 5.0
    sim_b, _warnings = service.parse_sim_spec(spec_b)
    sim_path = tmp_path / "sim.json"
    sim_path.write_text(sim_a.to_wire_json(), encoding="utf-8")
    calls = 0
    monkeypatch.setattr(
        service, "modal_port_monitor_names", lambda _data: ["port"])

    def fake_results(data):
        nonlocal calls
        calls += 1
        return {"size_x": float(service.sim_for(data).size_um[0])}

    monkeypatch.setattr(service, "modal_port_results", fake_results)
    cache_key = str(sim_path.resolve())
    service._SIM_CACHE.pop(cache_key, None)
    try:
        with TestClient(create_app(tmp_path)) as client:
            revision = client.get("/api/session").json()["result_id"]
            assert client.get(
                "/api/ports/modal",
                params={"rev": revision},
            ).json()["size_x"] == 4.0
            assert client.get(
                "/api/ports/modal",
                params={"rev": revision},
            ).json()["size_x"] == 4.0
            assert calls == 1

            original = sim_path.stat()
            replacement = tmp_path / "sim.next"
            replacement.write_text(sim_b.to_wire_json(), encoding="utf-8")
            os.replace(replacement, sim_path)
            os.utime(
                sim_path,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )
            assert sim_path.stat().st_mtime_ns == original.st_mtime_ns

            refreshed = client.get(
                "/api/ports/modal", params={"rev": revision})
    finally:
        service._SIM_CACHE.pop(cache_key, None)

    assert refreshed.status_code == 200
    assert refreshed.json()["size_x"] == 5.0
    assert calls == 2


def test_modal_endpoint_tracks_hdf5_container_replacement(
    tmp_path, monkeypatch,
):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("h5py")
    from fastapi.testclient import TestClient
    from photonhub import convert_to_hdf5
    from photonhub.viz.server import create_app

    raw = tmp_path / "raw"
    bundle = tmp_path / "bundle"
    _write_external_flux_bundle(raw, 1.0)
    bundle.mkdir()
    h5_path = convert_to_hdf5(raw, bundle / "simulation.h5")
    (bundle / "sim.json").write_text("{}", encoding="utf-8")
    calls = 0
    monkeypatch.setattr(
        service, "modal_port_monitor_names", lambda _data: ["port"])

    def fake_results(data):
        nonlocal calls
        calls += 1
        return {"value": float(data["port"].values[0])}

    monkeypatch.setattr(service, "modal_port_results", fake_results)
    with TestClient(create_app(h5_path)) as client:
        revision = client.get("/api/session").json()["result_id"]
        assert client.get(
            "/api/ports/modal", params={"rev": revision}).json()["value"] == 1.0
        assert client.get(
            "/api/ports/modal", params={"rev": revision}).json()["value"] == 1.0
        assert calls == 1

        np.asarray([3.0], dtype="<f4").tofile(raw / "port.bin")
        replacement = convert_to_hdf5(raw, bundle / "simulation.next.h5")
        os.replace(replacement, h5_path)
        refreshed = client.get(
            "/api/ports/modal", params={"rev": revision})

    assert refreshed.status_code == 200
    assert refreshed.json()["value"] == 3.0
    assert calls == 2


def test_modal_endpoint_single_flights_concurrent_requests_and_releases_lock(
    tmp_path, monkeypatch,
):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    (tmp_path / "manifest.json").write_text(json.dumps({
        "manifest_version": "1", "run": {}, "grid": {}, "provenance": {},
        "monitors": [],
    }), encoding="utf-8")
    (tmp_path / "sim.json").write_text("{}", encoding="utf-8")
    expected = {
        "driven_port": "input",
        "driven_mode": {"polarization": "TE", "mode_index": 0},
        "normalization": "test normalization",
        "reference_plane": "test plane",
        "ports": [],
    }
    calls = 0
    calls_lock = threading.Lock()
    monkeypatch.setattr(
        service, "modal_port_monitor_names", lambda _data: ["input"])

    def fake_results(_data):
        nonlocal calls
        with calls_lock:
            calls += 1
        # Long enough for both HTTP worker threads to overlap at the cache
        # boundary; the server lock must keep the expensive render single-shot.
        time.sleep(0.05)
        return expected

    monkeypatch.setattr(service, "modal_port_results", fake_results)
    with TestClient(create_app(tmp_path)) as client:
        revision = client.get("/api/session").json()["result_id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(
                lambda _index: client.get(
                    "/api/ports/modal", params={"rev": revision}),
                range(2),
            ))
        assert [response.status_code for response in responses] == [200, 200]
        assert calls == 1

    attempts = 0

    def fail_once(_data):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("synthetic modal failure")
        return expected

    monkeypatch.setattr(service, "modal_port_results", fail_once)
    with TestClient(create_app(tmp_path)) as client:
        revision = client.get("/api/session").json()["result_id"]
        assert client.get(
            "/api/ports/modal", params={"rev": revision}).status_code == 400
        assert client.get(
            "/api/ports/modal", params={"rev": revision}).status_code == 200
    assert attempts == 2


def test_cached_modal_endpoint_still_verifies_every_sealed_port_artifact(
    tmp_path, monkeypatch,
):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    spec = service.default_sim().to_wire_dict()
    spec["monitors"] = [{
        "type": "flux",
        "name": "port",
        "axis": "x",
        "position_um": 3.5,
        "freqs_hz": list(FREQS),
    }]
    sim, _ = service.parse_sim_spec(spec)
    canonical = sim.to_wire_json() + "\n"
    run_root = tmp_path / "runs"
    ledger = RunLedger(run_root)
    record = ledger.create_request(
        run_id=None,
        canonical_spec=canonical,
        device="cpu",
        timeout_s=None,
        solver={"available": True, "info": {}, "capabilities": {}},
        estimate={"num_cells": 8},
    )
    ledger.append_event(record["run_id"], "running")
    output = Path(record["output_dir"])
    (output / "sim.json").write_text(canonical, encoding="utf-8")
    (output / "solver-events.jsonl").write_text(
        json.dumps({"event": "done"}) + "\n", encoding="utf-8")
    np.asarray((0.4, 0.5), dtype="<f4").tofile(output / "port.bin")
    (output / "manifest.json").write_text(json.dumps({
        "manifest_version": "1",
        "schema_version": "1.0.0",
        "run": {
            "n_steps": 100,
            "steps_run": 90,
            "dt_s": 1.0e-16,
            "wall_seconds": 1.0,
            "aborted": False,
            "abort_reason": "",
        },
        "grid": {
            "shape": [2, 2, 2],
            "dl_um": 0.1,
            "size_um": [0.2, 0.2, 0.2],
        },
        "provenance": {"input_sha256": record["spec_sha256"]},
        "monitors": [{
            "name": "port",
            "type": "flux",
            "file": "port.bin",
            "dtype": "float32",
            "shape": [2],
            "dims": ["freq"],
            "components": [],
            "freqs_hz": list(FREQS),
            "axis": "x",
            "dt_s": 1.0e-16,
        }],
    }) + "\n", encoding="utf-8")
    ledger.seal(record["run_id"], "completed")

    expected = {
        "driven_port": "port",
        "driven_mode": {"polarization": "TE", "mode_index": 0},
        "normalization": "test normalization",
        "reference_plane": "test plane",
        "ports": [],
    }
    calls = []
    monkeypatch.setattr(
        service, "modal_port_monitor_names", lambda _data: ["port"])

    def fake_results(data):
        calls.append(data)
        return expected

    monkeypatch.setattr(service, "modal_port_results", fake_results)
    with TestClient(create_app(run_root=run_root)) as client:
        opened = client.post(f"/api/runs/{record['run_id']}/open", json={})
        assert opened.status_code == 200
        revision = opened.json()["result_id"]
        first = client.get("/api/ports/modal", params={"rev": revision})
        assert first.status_code == 200
        assert first.json() == expected
        assert len(calls) == 1

        # A cache hit must remain inside the ledger boundary. Rewriting the
        # blob with the same byte count cannot expose the prior cached payload.
        np.asarray((0.6, 0.7), dtype="<f4").tofile(output / "port.bin")
        rejected = client.get("/api/ports/modal", params={"rev": revision})

    assert rejected.status_code == 409
    assert "integrity" in rejected.json()["detail"]
    assert len(calls) == 1
