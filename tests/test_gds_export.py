"""Round-trip tests for export_gds (the reciprocal of import_gds). Geometry
equality is defined on the extruded shape (vertices/bounds/medium), not the
Python class — a Box exports to a rectangle that re-imports as a 4-vertex
PolySlab, a Cylinder to a faceted polygon."""

import math

import pytest

gdstk = pytest.importorskip("gdstk")

from photonhub.components.structures import Box, Cylinder, Medium, PolySlab, Structure
from photonhub.gds import GdsLayer, export_gds, import_gds
from photonhub.library import cosine_taper_crossing


SI = Medium(permittivity=3.5**2)
OX = Medium(permittivity=1.44**2)


def _bbox(vertices):
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def test_box_roundtrips(tmp_path):
    box = Structure(
        geometry=Box(center_um=(2.0, 1.0, 0.08), size_um=(1.0, 0.5, 0.16)),
        medium=SI,
    )
    path = tmp_path / "box.gds"
    layers = export_gds([box], path)
    assert len(layers) == 1
    assert layers[0].slab_bounds_um == pytest.approx((0.0, 0.16))
    back = import_gds(path, layers)
    assert len(back) == 1
    poly = back[0].geometry
    assert isinstance(poly, PolySlab)
    # same footprint bbox and slab
    bb = _bbox(poly.vertices_um)
    assert bb == pytest.approx((1.5, 0.75, 2.5, 1.25))
    assert poly.slab_bounds_um == pytest.approx((0.0, 0.16))
    assert back[0].medium.permittivity == pytest.approx(SI.permittivity)


def test_polyslab_vertices_preserved(tmp_path):
    verts = ((0.0, 0.0), (2.0, 0.0), (2.0, 0.4), (1.0, 0.6), (0.0, 0.4))
    ps = Structure(
        geometry=PolySlab(axis="z", vertices_um=verts, slab_bounds_um=(0.0, 0.22)),
        medium=SI,
    )
    path = tmp_path / "poly.gds"
    layers = export_gds([ps], path)
    back = import_gds(path, layers)
    got = back[0].geometry.vertices_um
    # same set of vertices (winding/rotation may differ; compare as sets rounded)
    a = {(round(x, 6), round(y, 6)) for x, y in verts}
    b = {(round(x, 6), round(y, 6)) for x, y in got}
    assert a == b


def test_cylinder_ring_roundtrips_area(tmp_path):
    ring = Structure(
        geometry=Cylinder(axis="z", center_um=(3.0, 3.0, 0.1), radius_um=2.0,
                          inner_radius_um=1.5, length_um=0.2),
        medium=SI,
    )
    path = tmp_path / "ring.gds"
    layers = export_gds([ring], path, cylinder_tolerance_um=2e-3)
    back = import_gds(path, layers)
    # area of the faceted annulus ~ pi(R^2 - r^2); one polygon, not fractured
    assert len(back) >= 1
    from photonhub.gds import _signed_area
    total = sum(abs(_signed_area(s.geometry.vertices_um)) for s in back)
    assert total == pytest.approx(math.pi * (2.0**2 - 1.5**2), rel=0.02)


def test_two_media_get_two_layers(tmp_path):
    core = Structure(geometry=Box(center_um=(1, 1, 0.08), size_um=(0.5, 0.5, 0.16)), medium=SI)
    clad = Structure(geometry=Box(center_um=(1, 1, 0.08), size_um=(2.0, 2.0, 0.16)), medium=OX)
    path = tmp_path / "two.gds"
    layers = export_gds([core, clad], path)
    assert len(layers) == 2
    assert {l.layer for l in layers} == {(1, 0), (2, 0)}
    media = {round(l.medium.permittivity, 4) for l in layers}
    assert media == {round(SI.permittivity, 4), round(OX.permittivity, 4)}


def test_explicit_layers_pin_numbers(tmp_path):
    box = Structure(geometry=Box(center_um=(1, 1, 0.08), size_um=(0.5, 0.5, 0.16)), medium=SI)
    layers = [GdsLayer(layer=(7, 3), medium=SI, zmin_um=0.0, thickness_um=0.16)]
    path = tmp_path / "pinned.gds"
    out = export_gds([box], path, layers=layers)
    assert out[0].layer == (7, 3)
    back = import_gds(path, layers)
    assert len(back) == 1


def test_crossing_component_exports(tmp_path):
    comp = cosine_taper_crossing(
        wg_width_um=0.35, junction_width_um=0.875, peak_width_um=1.0,
        taper_length_um=1.91, arm_length_um=3.0, thickness_um=0.161,
        n_points=24, medium=SI,
    )
    path = tmp_path / "crossing.gds"
    layers = export_gds(comp, path)  # accepts a Component
    assert len(layers) == 1          # one layer (single medium + slab)
    back = import_gds(path, layers)
    # 9 structures (4 tapers + junction + 4 stubs) -> 9 polygons
    assert len(back) == 9


def test_sphere_rejected(tmp_path):
    from photonhub.components.structures import Sphere
    s = Structure(geometry=Sphere(center_um=(1, 1, 1), radius_um=0.5), medium=SI)
    with pytest.raises(ValueError, match="cannot represent"):
        export_gds([s], tmp_path / "x.gds")
