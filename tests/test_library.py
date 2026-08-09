"""Geometry-level tests for the parametric PIC component library
(``photonhub.library``). No engine run — we assert only that each builder
emits the expected primitive geometry and the right ports."""

import math

import pytest

from photonhub.components.structures import Box, Cylinder, Medium, PolySlab
from photonhub.library import (
    SILICON,
    Component,
    Port,
    bend,
    coupler,
    crossing,
    ring,
    straight,
)


def _widths_of_box(box: Box):
    return box.size_um


# ---------------------------------------------------------------------------
# straight
# ---------------------------------------------------------------------------


def test_straight_box_dimensions_and_ports():
    c = straight(length_um=10.0, width_um=0.5, thickness_um=0.22, center_um=(1, 2, 3))
    assert isinstance(c, Component)
    assert len(c.structures) == 1
    geo = c.structures[0].geometry
    assert isinstance(geo, Box)
    # axis=x prop, y width, z thickness
    assert geo.size_um == (10.0, 0.5, 0.22)
    assert geo.center_um == (1, 2, 3)
    # default material is silicon
    assert c.structures[0].medium == SILICON
    assert c.structures[0].medium.permittivity == pytest.approx(12.25)

    assert len(c.ports) == 2
    pin, pout = c.ports
    assert {p.name for p in c.ports} == {"in", "out"}
    assert pin.axis == "x" and pout.axis == "x"
    assert pin.width_um == 0.5
    # ports at the two ends along x: center.x +/- length/2
    xs = sorted(p.center_um[0] for p in c.ports)
    assert xs == pytest.approx([1 - 5.0, 1 + 5.0])
    # transverse coords untouched
    for p in c.ports:
        assert p.center_um[1] == 2 and p.center_um[2] == 3


def test_straight_defaults():
    c = straight(length_um=4.0)
    geo = c.structures[0].geometry
    # default width 0.45, thickness 0.22
    assert geo.size_um == (4.0, 0.45, 0.22)


def test_straight_alternate_axis():
    c = straight(length_um=6.0, axis="y", thickness_axis="z")
    geo = c.structures[0].geometry
    # prop=y, thickness=z, width=x
    assert geo.size_um[1] == 6.0  # length along y
    assert geo.size_um[2] == 0.22  # thickness along z
    for p in c.ports:
        assert p.axis == "y"


# ---------------------------------------------------------------------------
# bend
# ---------------------------------------------------------------------------


def test_bend_is_quarter_annulus():
    R, w, t = 5.0, 0.5, 0.22
    c = bend(radius_um=R, width_um=w, thickness_um=t)
    assert len(c.structures) == 1
    geo = c.structures[0].geometry
    assert isinstance(geo, Cylinder)
    # ~pi/2 sweep
    sweep = geo.angle_stop - geo.angle_start
    assert sweep == pytest.approx(math.pi / 2.0)
    # inner/outer straddle radius +/- width/2
    assert geo.inner_radius_um == pytest.approx(R - w / 2.0)
    assert geo.radius_um == pytest.approx(R + w / 2.0)
    assert geo.inner_radius_um > 0
    assert geo.length_um == pytest.approx(t)
    # extruded along thickness axis (default z)
    assert geo.axis == "z"

    assert len(c.ports) == 2
    # the two arc ends propagate along the two in-plane axes
    assert {p.axis for p in c.ports} == {"x", "y"}
    for p in c.ports:
        assert p.width_um == w


def test_bend_port_positions_on_centerline():
    R = 4.0
    c = bend(radius_um=R, center_um=(0, 0, 0))
    # one port at (R,0,*) propagating x, one at (0,R,*) propagating y
    by_axis = {p.axis: p for p in c.ports}
    assert by_axis["x"].center_um[0] == pytest.approx(R)
    assert by_axis["x"].center_um[1] == pytest.approx(0.0)
    assert by_axis["y"].center_um[1] == pytest.approx(R)
    assert by_axis["y"].center_um[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# taper
# ---------------------------------------------------------------------------


def test_taper_polyslab_widths():
    c = taper_call()
    geo = c.structures[0].geometry
    assert isinstance(geo, PolySlab)
    assert len(geo.vertices_um) == 4
    assert geo.axis == "z"
    # the two ends (min-x and max-x corners) span width1 and width2
    w1_end = _width_at_x(geo, x_target=min(v_x(geo)))
    w2_end = _width_at_x(geo, x_target=max(v_x(geo)))
    assert w1_end == pytest.approx(0.4)
    assert w2_end == pytest.approx(1.0)
    # ports carry the local widths
    by_name = {p.name: p for p in c.ports}
    assert by_name["in"].width_um == pytest.approx(0.4)
    assert by_name["out"].width_um == pytest.approx(1.0)
    assert by_name["in"].axis == "x" and by_name["out"].axis == "x"


def taper_call():
    from photonhub.library import taper

    return taper(length_um=8.0, width1_um=0.4, width2_um=1.0, center_um=(0, 0, 0))


def v_x(geo):
    # x is the lower-indexed transverse axis for an axis="z" polyslab → u
    return [u for (u, v) in geo.vertices_um]


def _width_at_x(geo, x_target):
    ys = [v for (u, v) in geo.vertices_um if u == pytest.approx(x_target)]
    return max(ys) - min(ys)


def test_taper_slab_bounds():
    c = taper_call()
    geo = c.structures[0].geometry
    lo, hi = geo.slab_bounds_um
    assert hi - lo == pytest.approx(0.22)


# ---------------------------------------------------------------------------
# crossing
# ---------------------------------------------------------------------------


def test_crossing_two_perpendicular_guides():
    c = crossing(width_um=0.5, arm_length_um=3.0)
    assert len(c.structures) == 2
    g0, g1 = (s.geometry for s in c.structures)
    assert isinstance(g0, Box) and isinstance(g1, Box)
    # one arm long along x, the other long along y
    assert g0.size_um[0] == pytest.approx(3.0)  # x-arm
    assert g0.size_um[1] == pytest.approx(0.5)
    assert g1.size_um[1] == pytest.approx(3.0)  # y-arm
    assert g1.size_um[0] == pytest.approx(0.5)
    # 4 ports, two per in-plane axis
    assert len(c.ports) == 4
    axes = [p.axis for p in c.ports]
    assert axes.count("x") == 2 and axes.count("y") == 2


def test_crossing_port_positions():
    c = crossing(arm_length_um=2.0, center_um=(0, 0, 0))
    xs = sorted(p.center_um[0] for p in c.ports if p.axis == "x")
    assert xs == pytest.approx([-1.0, 1.0])
    ys = sorted(p.center_um[1] for p in c.ports if p.axis == "y")
    assert ys == pytest.approx([-1.0, 1.0])


# ---------------------------------------------------------------------------
# coupler
# ---------------------------------------------------------------------------


def test_coupler_two_parallel_guides_at_gap():
    w, gap = 0.5, 0.2
    c = coupler(length_um=10.0, width_um=w, gap_um=gap, center_um=(0, 0, 0))
    assert len(c.structures) == 2
    g0, g1 = (s.geometry for s in c.structures)
    assert isinstance(g0, Box) and isinstance(g1, Box)
    # both long along x with width along y
    for g in (g0, g1):
        assert g.size_um[0] == pytest.approx(10.0)
        assert g.size_um[1] == pytest.approx(w)
    # edge-to-edge gap along y == gap_um
    y_centers = sorted(g.center_um[1] for g in (g0, g1))
    center_to_center = y_centers[1] - y_centers[0]
    edge_to_edge = center_to_center - w
    assert edge_to_edge == pytest.approx(gap)
    # 4 ports
    assert len(c.ports) == 4
    assert all(p.axis == "x" for p in c.ports)
    assert {p.name for p in c.ports} == {"top_in", "top_out", "bot_in", "bot_out"}


# ---------------------------------------------------------------------------
# ring
# ---------------------------------------------------------------------------


def test_ring_full_sweep_annulus_plus_bus():
    R, w, gap = 5.0, 0.45, 0.2
    c = ring(radius_um=R, width_um=w, gap_um=gap, center_um=(0, 0, 0))
    assert len(c.structures) == 2
    geos = [s.geometry for s in c.structures]
    cyl = next(g for g in geos if isinstance(g, Cylinder))
    bus = next(g for g in geos if isinstance(g, Box))

    # full 2*pi sweep, annulus with inner > 0
    assert cyl.angle_stop - cyl.angle_start == pytest.approx(2.0 * math.pi)
    assert cyl.inner_radius_um == pytest.approx(R - w / 2.0)
    assert cyl.radius_um == pytest.approx(R + w / 2.0)
    assert cyl.inner_radius_um > 0
    assert cyl.axis == "z"

    # bus offset so its inner edge is gap_um from ring outer edge
    outer = R + w / 2.0
    bus_y = bus.center_um[1]
    bus_inner_edge = bus_y - bus.size_um[1] / 2.0
    assert bus_inner_edge - outer == pytest.approx(gap)

    # 2 ports on the bus
    assert len(c.ports) == 2
    assert {p.name for p in c.ports} == {"in", "through"}
    assert all(p.axis == "x" for p in c.ports)


def test_ring_custom_bus_width():
    c = ring(radius_um=4.0, width_um=0.4, bus_width_um=0.6, gap_um=0.15)
    bus = next(s.geometry for s in c.structures if isinstance(s.geometry, Box))
    assert bus.size_um[1] == pytest.approx(0.6)
    for p in c.ports:
        assert p.width_um == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# dataclasses & shared behaviour
# ---------------------------------------------------------------------------


def test_port_and_component_are_frozen():
    p = Port("in", (0, 0, 0), "x", 0.5)
    with pytest.raises(Exception):
        p.name = "out"
    c = straight(length_um=1.0)
    with pytest.raises(Exception):
        c.structures = ()


def test_custom_medium_propagates():
    m = Medium(permittivity=4.0)
    c = straight(length_um=2.0, medium=m)
    assert c.structures[0].medium == m
