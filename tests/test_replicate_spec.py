"""Tests for the paper-replication intake: PaperSpec loading/validation and the
geometry registry dispatch. No engine run."""

from pathlib import Path

import pytest

from photonhub.components.structures import Medium, PolySlab, Box
from photonhub.replicate import PaperSpec, SpecError, build_geometry

_REPO = Path(__file__).resolve().parents[2]
_SPEC = _REPO / "benchmarks" / "replicate" / "specs" / "chandran_cosine_crossing.yaml"


def test_chandran_spec_loads():
    spec = PaperSpec.from_yaml(_SPEC)
    assert spec.name == "chandran_cosine_crossing"
    assert spec.source.doi == "10.1364/OL.402446"
    assert spec.source.matched_sim == "tidy3d:WaveguideCrossing"
    assert spec.device.kind == "cosine_taper_crossing"
    # O-band, TE0
    assert spec.optical.band_um == (1.26, 1.36)
    assert spec.optical.center_um == pytest.approx(1.31)
    assert spec.optical.polarization == "TE"
    assert spec.optical.mode_index == 0
    # stack core is the 161 nm layer
    assert spec.stack.core.thickness_um == pytest.approx(0.161)
    assert spec.stack.core.material == "cSi"
    assert spec.stack.clad_material == "SiO2"
    # port roles
    assert spec.ports.input == "x-"
    assert spec.ports.through == "x+"
    assert set(spec.ports.cross) == {"y-", "y+"}


def test_reference_units_are_parsed():
    spec = PaperSpec.from_yaml(_SPEC)
    quantities = {(r.quantity, r.port): r for r in spec.references}
    il = quantities[("insertion_loss", "x+")]
    assert il.units == "dB"
    assert il.paper_value == pytest.approx(0.216)
    assert il.curve is not None and len(il.curve) == 11   # digitized Fig 2(b)
    xt = quantities[("crosstalk", "y-")]
    assert xt.units == "dB"
    assert xt.paper_value == pytest.approx(-30.0)


def test_reference_units_required():
    d = {
        "name": "x",
        "source": {"citation": "c"},
        "device": {"kind": "cosine_taper_crossing", "params": {}},
        "stack": {"layers": [{"name": "core", "material": "cSi", "zmin_um": 0.0, "thickness_um": 0.161}], "clad_material": "SiO2"},
        "optical": {"band_nm": [1260, 1360], "polarization": "TE0"},
        "ports": {"input": "x-", "through": "x+"},
        "references": [{"quantity": "insertion_loss", "port": "x+"}],  # no units
    }
    with pytest.raises(SpecError, match="units"):
        PaperSpec.from_dict(d)


def test_reference_port_must_be_a_role():
    spec_dict = {
        "name": "x",
        "source": {"citation": "c"},
        "device": {"kind": "cosine_taper_crossing", "params": {}},
        "stack": {"layers": [{"name": "core", "material": "cSi", "zmin_um": 0.0, "thickness_um": 0.161}], "clad_material": "SiO2"},
        "optical": {"band_nm": [1260, 1360], "polarization": "TE0"},
        "ports": {"input": "x-", "through": "x+"},
        "references": [{"quantity": "insertion_loss", "units": "dB", "port": "z9"}],
    }
    with pytest.raises(SpecError, match="role ports"):
        PaperSpec.from_dict(spec_dict)


def test_build_geometry_from_spec():
    spec = PaperSpec.from_yaml(_SPEC)
    medium = Medium(permittivity=3.5**2)
    comp = build_geometry(
        spec.device.kind,
        spec.device.params,
        medium=medium,
        thickness_um=spec.stack.core.thickness_um,
    )
    polys = [s for s in comp.structures if isinstance(s.geometry, PolySlab)]
    boxes = [s for s in comp.structures if isinstance(s.geometry, Box)]
    assert len(polys) == 4          # four cosine tapers
    assert len(boxes) == 1 + 4      # junction + routing stubs
    assert {p.name for p in comp.ports} == {"x-", "x+", "y-", "y+"}
    # every emitted structure carries the resolved core thickness
    for s in comp.structures:
        lo, hi = (s.geometry.slab_bounds_um if isinstance(s.geometry, PolySlab)
                  else (s.geometry.center_um[2] - s.geometry.size_um[2] / 2,
                        s.geometry.center_um[2] + s.geometry.size_um[2] / 2))
        assert (hi - lo) == pytest.approx(0.161)


def test_unknown_kind_raises():
    with pytest.raises(KeyError, match="unknown device kind"):
        build_geometry("no_such_device", {}, medium=Medium(permittivity=1.0), thickness_um=0.2)
