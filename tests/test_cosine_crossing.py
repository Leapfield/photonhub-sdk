"""Geometry-level tests for the cosine ("beam shaping") taper and the
cosine-taper waveguide crossing (``photonhub.library``). No engine run — we
assert only the emitted primitive geometry and ports, the way the other
library builders are tested. The crossing reproduces the topology of Chandran
et al., Opt. Lett. 45, 6230 (2020)."""

import math

import pytest

from photonhub.components.structures import Box, PolySlab
from photonhub.library import cosine_taper, cosine_taper_crossing


def _full_widths_by_prop(poly: PolySlab, prop_index: int, width_index: int):
    """Group vertices by their propagation coordinate and return
    ``{prop: full_width}`` from the min/max transverse coordinate there."""
    groups: dict[float, list[float]] = {}
    for vertex in poly.vertices_um:
        prop = round(vertex[prop_index], 6)
        groups.setdefault(prop, []).append(vertex[width_index])
    return {p: max(w) - min(w) for p, w in groups.items()}


def test_cosine_taper_endpoints_and_convexity():
    length, w1, w2 = 1.9, 0.4, 1.0
    n = 32
    comp = cosine_taper(length, w1, w2, n_points=n)
    (struct,) = comp.structures
    poly = struct.geometry
    assert isinstance(poly, PolySlab)
    # two edges of n points each
    assert len(poly.vertices_um) == 2 * n
    # vertices live in (x, y); axis="x" prop, width along y
    widths = _full_widths_by_prop(poly, prop_index=0, width_index=1)
    xs = sorted(widths)
    assert widths[xs[0]] == pytest.approx(w1, abs=1e-9)   # narrow end
    assert widths[xs[-1]] == pytest.approx(w2, abs=1e-9)  # wide (aperture) end
    # monotonic widening from narrow to wide
    seq = [widths[x] for x in xs]
    assert all(b >= a - 1e-12 for a, b in zip(seq, seq[1:]))
    # convex ("bulging"): every interior width sits at/above the straight-taper
    # chord between the two ends
    for x in xs:
        frac = (x - xs[0]) / (xs[-1] - xs[0])
        chord = w1 + frac * (w2 - w1)
        assert widths[x] >= chord - 1e-9
    # zero-slope (flat) aperture: last step is smaller than the first step
    assert (seq[-1] - seq[-2]) < (seq[1] - seq[0])
    # ports carry the local widths at the two ends
    names = {p.name: p for p in comp.ports}
    assert names["in"].width_um == pytest.approx(w1)
    assert names["out"].width_um == pytest.approx(w2)
    assert names["in"].center_um[0] == pytest.approx(-length / 2)
    assert names["out"].center_um[0] == pytest.approx(+length / 2)


def test_cosine_taper_lens_peak_is_interior():
    # W_in=0.35 -> W_out=0.875 with peak W_m=1.0: the peak width is reached
    # INSIDE the taper (the convex lens), not at an end
    comp = cosine_taper(1.91, 0.35, 0.875, peak_width_um=1.0, n_points=41)
    poly = comp.structures[0].geometry
    widths = _full_widths_by_prop(poly, prop_index=0, width_index=1)
    xs = sorted(widths)
    assert widths[xs[0]] == pytest.approx(0.35, abs=1e-6)
    assert widths[xs[-1]] == pytest.approx(0.875, abs=1e-6)
    peak_x = max(xs, key=lambda x: widths[x])
    # discrete samples approach the continuous peak W_m=1.0 from below
    assert 0.99 <= widths[peak_x] <= 1.0 + 1e-9            # W_m
    assert xs[0] < peak_x < xs[-1]                          # interior => lens


def test_cosine_taper_peak_below_ends_rejected():
    with pytest.raises(ValueError, match="peak_width_um"):
        cosine_taper(1.0, 0.4, 0.9, peak_width_um=0.5)


def test_cosine_taper_equal_widths_is_straight():
    comp = cosine_taper(2.0, 0.5, 0.5, n_points=16)
    poly = comp.structures[0].geometry
    widths = _full_widths_by_prop(poly, prop_index=0, width_index=1)
    assert all(w == pytest.approx(0.5, abs=1e-9) for w in widths.values())


def test_cosine_taper_slab_bounds():
    comp = cosine_taper(1.0, 0.4, 0.9, thickness_um=0.161, center_um=(0, 0, 0.2))
    lo, hi = comp.structures[0].geometry.slab_bounds_um
    assert (hi - lo) == pytest.approx(0.161)
    assert (lo + hi) / 2 == pytest.approx(0.2)


def test_crossing_structure_and_ports():
    comp = cosine_taper_crossing(
        wg_width_um=0.35,
        junction_width_um=0.875,
        peak_width_um=1.0,
        taper_length_um=1.91,
        arm_length_um=3.0,
        thickness_um=0.161,
        n_points=24,
    )
    boxes = [s for s in comp.structures if isinstance(s.geometry, Box)]
    polys = [s for s in comp.structures if isinstance(s.geometry, PolySlab)]
    assert len(polys) == 4          # one cosine taper per arm
    assert len(boxes) == 1 + 4      # central junction + 4 routing stubs
    # central junction is the W_out-width square
    central = min(boxes, key=lambda s: abs(s.geometry.center_um[0]) + abs(s.geometry.center_um[1]))
    assert central.geometry.size_um[0] == pytest.approx(0.875)
    assert central.geometry.size_um[1] == pytest.approx(0.875)
    assert central.geometry.size_um[2] == pytest.approx(0.161)
    # four ports, one per arm end, at +/-arm_length on both in-plane axes
    ports = {p.name: p for p in comp.ports}
    assert set(ports) == {"x-", "x+", "y-", "y+"}
    assert ports["x+"].center_um[0] == pytest.approx(3.0)
    assert ports["x-"].center_um[0] == pytest.approx(-3.0)
    assert ports["y+"].center_um[1] == pytest.approx(3.0)
    assert ports["y-"].center_um[1] == pytest.approx(-3.0)
    for p in comp.ports:
        assert p.width_um == pytest.approx(0.35)   # W_in routing width


def test_crossing_no_stub_when_arm_is_shaped_region():
    # arm_length exactly the shaped region => no routing stubs
    comp = cosine_taper_crossing(
        wg_width_um=0.35,
        junction_width_um=0.875,
        peak_width_um=1.0,
        taper_length_um=1.91,
        arm_length_um=0.4375 + 1.91,  # junction/2 + taper_length
    )
    boxes = [s for s in comp.structures if isinstance(s.geometry, Box)]
    assert len(boxes) == 1  # only the central junction


def test_crossing_rejects_short_arm():
    with pytest.raises(ValueError, match="arm_length_um"):
        cosine_taper_crossing(
            wg_width_um=0.35,
            junction_width_um=0.875,
            peak_width_um=1.0,
            taper_length_um=1.91,
            arm_length_um=1.0,  # < junction/2 + taper_length = 2.35
        )
