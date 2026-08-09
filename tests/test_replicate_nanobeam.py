"""Tests for the nanobeam-cavity replication seam: the deterministic hole layout,
the registered geometry builder, and the cavity-FOM spec extension (reserved
"cavity" port). No engine run.

Device: Quan, Deotare & Loncar, APL 96, 203102 (2010) (arXiv:1002.1319).
"""

import math
from pathlib import Path

import pytest

from photonhub.components.structures import Box, Cylinder, Medium
from photonhub.replicate import PaperSpec, SpecError, build_geometry
from photonhub.replicate.geometry import nanobeam_hole_layout

_REPO = Path(__file__).resolve().parents[2]
_SPEC = _REPO / "benchmarks" / "replicate" / "specs" / "quan_nanobeam_cavity.yaml"

A_UM = 0.305
W_UM = 0.700


def test_layout_is_symmetric_and_ff_matched():
    holes, span = nanobeam_hole_layout(
        period_um=A_UM, beam_width_um=W_UM, n_segments_per_side=40)
    # symmetric about x=0: every +x hole has a mirror -x hole of equal radius
    pos = sorted((x, r) for x, r in holes if x > 0)
    neg = sorted((-x, r) for x, r in holes if x < 0)
    assert pos == neg
    # mirror span = 2 * N * a  (zero cavity length)
    assert span == pytest.approx(2 * 40 * A_UM)
    # innermost segment has FF = 0.15 -> r = sqrt(FF*a*w/pi)
    r_in = max(r for _, r in holes)
    ff_in = math.pi * r_in ** 2 / (A_UM * W_UM)
    assert ff_in == pytest.approx(0.15, abs=1e-6)
    # the FF=0 outermost segment contributes no hole -> 39 holes per side, not 40
    assert len(holes) == 2 * 39


def test_layout_cavity_length_shifts_innermost_holes():
    a, L = A_UM, 0.5
    holes, _ = nanobeam_hole_layout(
        period_um=a, beam_width_um=W_UM, n_segments_per_side=5,
        cavity_length_um=L)
    innermost = min(abs(x) for x, _ in holes)
    assert innermost == pytest.approx(L / 2 + 0.5 * a)  # L=0 would give a/2


def test_builder_emits_beam_holes_and_ports():
    comp = build_geometry(
        "nanobeam_modulated_cavity",
        dict(period_um=A_UM, beam_width_um=W_UM, n_segments_per_side=40,
             feed_stub_um=1.0, clad_index=1.45),
        medium=Medium(permittivity=3.46 ** 2), thickness_um=0.220,
        center_um=(10.0, 3.0, 1.5),
    )
    boxes = [s for s in comp.structures if isinstance(s.geometry, Box)]
    cyls = [s for s in comp.structures if isinstance(s.geometry, Cylinder)]
    assert len(boxes) == 1                 # the Si beam
    assert len(cyls) == 2 * 39             # silica holes (FF=0 segment has none)
    assert {p.name for p in comp.ports} == {"x-", "x+"}
    # every hole carries the cladding index (silica-backfilled), the beam the core
    assert boxes[0].medium.permittivity == pytest.approx(3.46 ** 2)
    for c in cyls:
        assert c.medium.permittivity == pytest.approx(1.45 ** 2)
    # beam height == resolved core thickness; holes punch the full height
    assert boxes[0].geometry.size_um[2] == pytest.approx(0.220)
    assert all(c.geometry.length_um == pytest.approx(0.220) for c in cyls)


def test_spec_loads_with_cavity_fom():
    spec = PaperSpec.from_yaml(_SPEC)
    assert spec.device.kind == "nanobeam_modulated_cavity"
    by_q = {r.quantity: r for r in spec.references}
    assert by_q["quality_factor"].paper_value == pytest.approx(1.2e6)
    assert by_q["quality_factor"].units == "dimensionless"
    assert by_q["quality_factor"].port == "cavity"
    assert by_q["resonance_wavelength"].units == "nm"
    assert by_q["transmission"].port == "x+"  # a real port quantity


def test_cavity_quantity_requires_reserved_port():
    d = {
        "name": "x", "source": {"citation": "c"},
        "device": {"kind": "nanobeam_modulated_cavity", "params": {}},
        "stack": {"layers": [{"name": "core", "material": "n=3.46",
                              "zmin_um": 0.0, "thickness_um": 0.22}],
                  "clad_material": "n=1.45"},
        "optical": {"band_nm": [1500, 1600], "polarization": "TE0"},
        "ports": {"input": "x-", "through": "x+"},
        # a cavity quantity on a physical port must be rejected
        "references": [{"quantity": "quality_factor", "units": "dimensionless",
                        "port": "x+"}],
    }
    with pytest.raises(SpecError, match="reserved port"):
        PaperSpec.from_dict(d)
