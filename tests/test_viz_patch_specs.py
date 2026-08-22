"""Cut-plane patch specs (§5): parallel cuts must trace the true section.

A plane cut PARALLEL to a PolySlab's extrusion axis used to draw the polygon's
bounding rectangle — for a curved taper the overlay claimed the guide was the
polygon's full transverse hull (7× too wide in the mode-converter bundle).
These tests pin the exact-interval behavior instead.
"""

import math

from photonhub.viz import _geometry as geom
from photonhub.viz.service import _spec_loops


SLAB = (0.0, 0.22)


def _rects(spec):
    assert spec is not None and spec[0] == "rects"
    return list(spec[1])


def test_parallel_cut_uses_true_interval_not_bounding_box():
    # Right triangle in (x, y), extruded along z: at x=2 the section is
    # y in [0, 2], while the bounding box would claim [0, 4].
    tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
    rects = _rects(geom.polyslab_section("z", tri, SLAB, "x", 2.0))
    assert len(rects) == 1
    (y0, z0, w, h) = rects[0]
    assert math.isclose(y0, 0.0) and math.isclose(y0 + w, 2.0)
    assert math.isclose(z0, SLAB[0]) and math.isclose(h, SLAB[1] - SLAB[0])


def test_parallel_cut_through_concave_polygon_yields_two_rects():
    # A "U" opening upward: a horizontal line through the arms crosses two
    # disjoint spans — one rectangle per arm, not their hull.
    u_shape = [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (3.0, 4.0),
               (3.0, 1.0), (2.0, 1.0), (2.0, 4.0), (0.0, 4.0)]
    rects = _rects(geom.polyslab_section("z", u_shape, SLAB, "y", 2.0))
    spans = sorted((r[0], r[0] + r[2]) for r in rects)
    assert len(spans) == 2
    assert math.isclose(spans[0][0], 0.0) and math.isclose(spans[0][1], 2.0)
    assert math.isclose(spans[1][0], 3.0) and math.isclose(spans[1][1], 5.0)


def test_parallel_cut_missing_the_polygon_is_none():
    tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
    assert geom.polyslab_section("z", tri, SLAB, "x", 5.0) is None
    assert geom.polyslab_section("z", tri, SLAB, "x", -1.0) is None


def test_perpendicular_cut_still_returns_the_exact_polygon():
    tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
    spec = geom.polyslab_section("z", tri, SLAB, "z", 0.1)
    assert spec is not None and spec[0] == "polygon"
    assert list(spec[1]) == tri


def test_cylinder_parallel_cut_is_the_chord_not_the_diameter():
    # z-axis cylinder, r=2 at (u, v)=(5, 5): at x=6 the chord half-width is
    # sqrt(4-1)=sqrt(3), not the radius.
    spec = geom.cylinder_section("z", (5.0, 5.0, 0.11), 2.0, 0.0, 0.22,
                                 0.0, 2.0 * math.pi, "x", 6.0)
    assert spec is not None and spec[0] == "rect"
    (y0, _z0, w, _h) = spec[1]
    half = math.sqrt(3.0)
    assert math.isclose(y0, 5.0 - half, rel_tol=1e-12)
    assert math.isclose(y0 + w, 5.0 + half, rel_tol=1e-12)
    assert geom.cylinder_section("z", (5.0, 5.0, 0.11), 2.0, 0.0, 0.22,
                                 0.0, 2.0 * math.pi, "x", 7.5) is None


def test_spec_loops_expands_rects_to_one_loop_per_interval():
    loops = _spec_loops(("rects", ((0.0, 0.0, 2.0, 0.22), (3.0, 0.0, 2.0, 0.22))))
    assert len(loops) == 2
    assert loops[0][0] == (0.0, 0.0) and loops[1][0] == (3.0, 0.0)
