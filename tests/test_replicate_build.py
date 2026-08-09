"""Structural tests for spec -> Simulation assembly (no engine run): valid
Simulation, faithful source/monitor placement, arm auto-extension, cost."""

from pathlib import Path

import pytest

from photonhub.components.simulation import Simulation
from photonhub.replicate import PaperSpec
from photonhub.replicate.build import build_simulation

_REPO = Path(__file__).resolve().parents[2]
_SPEC = _REPO / "benchmarks" / "replicate" / "specs" / "chandran_cosine_crossing.yaml"


@pytest.fixture(scope="module")
def built():
    spec = PaperSpec.from_yaml(_SPEC)
    return build_simulation(spec, cells_per_wavelength=15)


def test_build_produces_valid_simulation(built):
    sim = built.sim
    assert isinstance(sim, Simulation)
    # eq-current (Huygens) launch: a per-cell dipole sheet, not a single source
    assert built.meta["launch"] == "eq_current_yee"
    assert len(sim.sources) > 1
    assert len(sim.sources) == built.meta["n_launch_sources"]
    assert len(sim.monitors) == 4           # in + through + 2 cross
    assert sim.boundaries.x == sim.boundaries.y == sim.boundaries.z == "pml"
    # curved cosine walls -> contour subpixel on (repo default exact-fill KFJ)
    assert sim.subpixel is True
    assert sim.subpixel_method == "contour"


def test_grid_from_cells_per_wavelength(built):
    # dl = lambda_c / (cpw * n_core)
    dl = built.meta["dl_um"]
    expected = built.meta["wavelength_c_um"] / (15 * built.meta["n_core"])
    assert dl == pytest.approx(expected)


def test_source_injects_inward(built):
    # the launch sheet lives on the input arm (x-), one clearance inside the PML,
    # in the -x half of the domain
    xs = [s.center_um[0] for s in built.sim.sources]
    src_plane = sum(xs) / len(xs)
    assert src_plane < built.meta["sim_arm_um"]                       # -x half
    assert src_plane == pytest.approx(built.meta["pml_um"] + 0.4, abs=3 * built.meta["dl_um"])
    # the input monitor reads the injected (inward = +x) power
    assert built.in_monitor.direction == "+"


def test_monitor_directions_face_outward(built):
    assert built.in_monitor.direction == "+"           # forward into device
    assert built.out_monitors["through"].direction == "+"
    assert built.out_monitors["y-"].direction == "-"   # -y arm exits toward -y
    assert built.out_monitors["y+"].direction == "+"


def test_arm_auto_extended_for_monitor_room():
    # a spec whose arm is too short must be extended so a monitor fits in the
    # straight stub inside the PML
    spec = PaperSpec.from_yaml(_SPEC)
    short = spec.device.params | {"arm_length_um": 0.1}
    from dataclasses import replace
    from photonhub.replicate.spec import Device
    spec2 = replace(spec, device=replace(spec.device, params=short))
    built = build_simulation(spec2, cells_per_wavelength=12)
    inner = 0.5 * short["junction_width_um"] + short["taper_length_um"]
    assert built.meta["sim_arm_um"] > inner + built.meta["pml_um"]


def test_field_slice_monitor_added():
    spec = PaperSpec.from_yaml(_SPEC)
    b = build_simulation(spec, cells_per_wavelength=10, field_slice=True)
    names = [m.name for m in b.sim.monitors]
    assert "field_xy" in names
    assert b.meta["field_slice_name"] == "field_xy"
    assert b.meta["field_slice_freq_hz"] is not None
    # default is lean (no field slice)
    assert build_simulation(spec, cells_per_wavelength=10).meta["field_slice_name"] is None


def test_field_intensity_extraction():
    import numpy as np
    xr = pytest.importorskip("xarray")
    from photonhub.replicate.build import BuiltSim
    arr = np.zeros((1, 3, 1, 4, 5), dtype=complex)
    arr[0, 0, 0, 2, 3] = 2 + 0j          # Ex peak at (y=2, x=3)
    arr[0, 1, 0, 1, 1] = 1 + 0j          # a weaker Ey elsewhere
    da = xr.DataArray(arr, dims=("f", "component", "z", "y", "x"),
                      coords={"component": ["Ex", "Ey", "Ez"], "x": [0, 1, 2, 3, 4],
                              "y": [0, 1, 2, 3], "z": [0.5], "f": [2e14]})
    stub = BuiltSim(sim=None, in_monitor=None, out_monitors={}, freqs_hz=(2e14,),
                    meta={"field_slice_name": "field_xy"})
    x, y, e2 = stub.field_intensity({"field_xy": da})
    assert e2.shape == (4, 5)            # [y, x]
    assert e2.max() == pytest.approx(1.0)          # normalized to peak
    assert e2[2, 3] == pytest.approx(1.0)          # peak at (y=2, x=3)
    # no field slice -> None
    assert BuiltSim(sim=None, in_monitor=None, out_monitors={}, freqs_hz=(), meta={}).field_intensity({}) is None


def test_cost_is_estimable(built):
    ce = built.sim.cost_estimate()
    assert ce.num_cells > 0
    # low resolution -> a cheap run
    assert ce.usd < 5.0


def test_metrics_db_math():
    # exercise the dB conversion on a hand-built transmissions dict
    import math

    from photonhub.replicate.build import BuiltSim

    class _Stub(BuiltSim):
        def transmissions(self, data):
            return {"through": {1.0: 0.95}, "y-": {1.0: 1e-3}, "y+": {1.0: 2e-3}}

    stub = _Stub(sim=None, in_monitor=None, out_monitors={"through": None, "y-": None, "y+": None},
                 freqs_hz=(1.0,), meta={"through_role": "through"})
    m = stub.metrics_db(data=None)
    assert m["insertion_loss_db"][0] == pytest.approx(-10 * math.log10(0.95), abs=1e-9)
    assert m["crosstalk_db"]["y-"][0] == pytest.approx(10 * math.log10(1e-3), abs=1e-9)
    assert "through" not in m["crosstalk_db"]
