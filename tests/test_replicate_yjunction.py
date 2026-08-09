"""Geometry regression for the Lin & Shi arbitrary-ratio Y-junction.

The paper defines r1/r2 and R1/R2 on the OUTER boundaries of a widened solid
region.  The original reproduction accidentally treated each pair as the two
arcs of one output-arm centreline and did not use the published width ``w``.
These tests pin the boundary construction so that failure cannot recur while
still producing a plausible-looking 50:50 splitter.
"""

import math
from pathlib import Path

import pytest

from photonhub.components.structures import Medium
from photonhub.replicate import PaperSpec
from photonhub.replicate.geometry import build_geometry


_REPO = Path(__file__).resolve().parents[2]
_SPEC = _REPO / "benchmarks" / "replicate" / "specs" / "yjunction_lin_2019.yaml"


def _component(r1=1.70):
    spec = PaperSpec.from_yaml(_SPEC)
    params = spec.device.params | {"r_top1_um": r1}
    return build_geometry(
        "y_branch", params, medium=Medium(permittivity=12.0), thickness_um=0.220
    )


def _xy(structure):
    return list(structure.geometry.vertices_um)


def _contains(points, expected, tol=1e-9):
    return any(math.dist(point, expected) <= tol for point in points)


def test_published_width_and_output_gap_are_load_bearing():
    comp = _component()
    top, bottom = map(_xy, comp.structures[:2])

    assert max(y for _, y in top) == pytest.approx(+0.70)
    assert min(y for _, y in bottom) == pytest.approx(-0.70)

    # Outer output edges are ±(w0 + D/2) = ±0.60; inner edges are ±D/2 = ±0.10.
    assert _contains(top, (1.16, 0.60))
    assert _contains(bottom, (1.16, -0.60))
    assert top[-3] == pytest.approx((1.16, +0.10))
    assert bottom[-3] == pytest.approx((1.16, -0.10))


def test_symmetric_boundary_is_mirror_symmetric():
    comp = _component(1.70)
    top, bottom = map(_xy, comp.structures[:2])
    # Same sampling and a sign mirror for r1=R1, r2=R2, l=0.
    assert len(top) == len(bottom)
    for (xt, yt), (xb, yb) in zip(top, bottom):
        assert xt == pytest.approx(xb, abs=1e-12)
        assert yt == pytest.approx(-yb, abs=1e-12)


def test_small_r1_delays_only_the_top_boundary_expansion():
    comp = _component(0.22)
    top, bottom = map(_xy, comp.structures[:2])
    # vertex 0 is the region entrance; vertex 1 is the end of the lead-in.
    # A small r1 leaves a long top input-width section, while the fixed bottom
    # R1 boundary expands essentially from the entrance.  This is the paper's
    # single-knob asymmetry and is what the old centreline build erased.
    assert top[0] == pytest.approx((-1.16, +0.25))
    assert bottom[0] == pytest.approx((-1.16, -0.25))
    assert top[1][1] == pytest.approx(+0.25)
    assert top[1][0] > 0.2
    assert bottom[1][1] == pytest.approx(-0.25)
    assert bottom[1][0] < -1.1


def test_branch_point_is_sharp_and_derived_from_r2_tangent():
    comp = _component()
    top, bottom = map(_xy, comp.structures[:2])
    # b = (w - w0 - D/2)/2 from the paper.  The branch point aligns with the
    # tangent between the two equal-R2 arcs, one R2*sin(theta) before x=L.
    b = (0.70 - 0.50 - 0.20 / 2.0) / 2.0
    half_final_s = 1.0 * math.sin(math.acos((1.0 - b) / 1.0))
    expected_x = 2.32 / 2.0 - half_final_s
    assert top[-2] == pytest.approx((expected_x, 0.0), abs=1e-12)
    assert bottom[-2] == pytest.approx((expected_x, 0.0), abs=1e-12)


def test_asymmetric_branch_point_tracks_derived_top_r2():
    comp = _component(0.22)
    top, bottom = map(_xy, comp.structures[:2])
    w, w0, gap, r1, R1, R2 = 0.70, 0.50, 0.20, 0.22, 1.70, 1.00
    a = (w - w0 / 2.0) / 4.0
    b = (w - w0 - gap / 2.0) / 2.0
    r2 = (r1 - a) / (R1 - a) * (R2 - b) + b
    half_final_s = r2 * math.sin(math.acos((r2 - b) / r2))
    expected_x = 2.32 / 2.0 - half_final_s
    assert top[-2] == pytest.approx((expected_x, 0.0), abs=1e-12)
    assert bottom[-2] == pytest.approx((expected_x, 0.0), abs=1e-12)


def test_paper_spec_has_no_fitted_tip_parameters():
    spec = PaperSpec.from_yaml(_SPEC)
    assert spec.device.params["solid_junction"] is True
    assert "branch_tip_frac" not in spec.device.params
    assert "tip_gap_um" not in spec.device.params


def test_impossible_paper_radius_is_rejected():
    with pytest.raises(ValueError, match="first paper S segment cannot span"):
        _component(0.10)
