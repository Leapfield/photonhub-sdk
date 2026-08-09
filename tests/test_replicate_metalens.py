"""Tests for the metasurface meta-atom replication seam: the metalens pillar
geometry builder and the metasurface unit-cell spec extension (reserved
"unit_cell" port, transmission_phase quantity, portless device). No engine run.

Device: Liang et al., Nanomaterials 8, 288 (2018) — TiO2 nanopillar metalens.
"""

from pathlib import Path

import pytest

from photonhub.components.structures import Cylinder, Medium
from photonhub.replicate import PaperSpec, SpecError, build_geometry

_REPO = Path(__file__).resolve().parents[2]
_SPEC = _REPO / "benchmarks" / "replicate" / "specs" / "liang_tio2_metalens_meta_atom.yaml"


def test_builder_emits_one_pillar_no_ports():
    comp = build_geometry(
        "metalens_meta_atom", dict(radius_um=0.100),
        medium=Medium(permittivity=2.58 ** 2), thickness_um=0.488,
        center_um=(0.159, 0.159, 0.444),
    )
    assert len(comp.structures) == 1
    assert comp.ports == ()                      # a metasurface has no waveguide ports
    pil = comp.structures[0]
    assert isinstance(pil.geometry, Cylinder)
    assert pil.geometry.radius_um == pytest.approx(0.100)
    assert pil.geometry.length_um == pytest.approx(0.488)   # = pillar height
    assert pil.geometry.axis == "z"
    assert pil.medium.permittivity == pytest.approx(2.58 ** 2)


def test_spec_loads_portless_with_unit_cell_fom():
    spec = PaperSpec.from_yaml(_SPEC)
    assert spec.device.kind == "metalens_meta_atom"
    # ports omitted -> defaults, and nothing references them
    assert (spec.ports.input, spec.ports.through) == ("in", "out")
    by_q = {r.quantity: r for r in spec.references}
    assert by_q["transmission_phase"].units == "radian"
    assert by_q["transmission_phase"].port == "unit_cell"
    assert by_q["transmission_phase"].bound is True        # >= 2*pi coverage
    assert by_q["transmission"].port == "unit_cell"        # unit-cell transmission, not a waveguide port
    # the design carries a radius sweep for the phase library
    assert spec.device.params["radius_sweep_nm"][0] == 10
    assert spec.stack.core.thickness_um == pytest.approx(0.488)


def test_phase_quantity_needs_unit_cell_port():
    base = dict(
        name="x", source={"citation": "c"},
        device={"kind": "metalens_meta_atom", "params": {}},
        stack={"layers": [{"name": "pillar", "material": "n=2.58",
                           "zmin_um": 0.0, "thickness_um": 0.488}],
               "clad_material": "n=1.0", "box_material": "n=1.45"},
        optical={"band_nm": [560, 800], "center_nm": 633, "polarization": "TE"},
    )
    # transmission_phase on a physical port is rejected
    bad = dict(base, ports={"input": "in", "through": "out"},
               references=[{"quantity": "transmission_phase", "units": "radian",
                            "port": "out"}])
    with pytest.raises(SpecError, match="reserved port"):
        PaperSpec.from_dict(bad)
    # a cavity quantity on the metasurface's unit_cell port is rejected
    wrong = dict(base, references=[{"quantity": "quality_factor",
                                    "units": "dimensionless", "port": "unit_cell"}])
    with pytest.raises(SpecError, match="not valid on the reserved port"):
        PaperSpec.from_dict(wrong)


def test_unit_cell_transmission_phase_ok():
    d = dict(
        name="x", source={"citation": "c"},
        device={"kind": "metalens_meta_atom", "params": {}},
        stack={"layers": [{"name": "pillar", "material": "n=2.58",
                           "zmin_um": 0.0, "thickness_um": 0.488}],
               "clad_material": "n=1.0", "box_material": "n=1.45"},
        optical={"band_nm": [560, 800], "center_nm": 633, "polarization": "TE"},
        # ports omitted entirely -> portless device validates
        references=[
            {"quantity": "transmission_phase", "units": "radian", "port": "unit_cell",
             "paper_value": 6.2832, "bound": True},
            {"quantity": "transmission", "units": "linear", "port": "unit_cell",
             "paper_value": 0.84},
        ],
    )
    spec = PaperSpec.from_dict(d)   # must not raise
    assert len(spec.references) == 2
