"""Axis-aligned bounding boxes for the geometry primitives.

Used by the material-aware boundary selection (which structures reach into the
PML/absorber region on a given face — :meth:`Simulation.with_auto_boundaries`).
Every returned box CONTAINS the geometry, so a "does it cross the boundary"
test is conservative: it can over-report a crossing (slanted Polygon), never
miss one. An annular sector's box is tight: a 90-degree bend spans its own
quadrant, not the full circle.
"""

import math
from typing import Tuple

from .structures import Box, Cylinder, Polygon, Sphere

Interval = Tuple[float, float]
Bounds = Tuple[Interval, Interval, Interval]


def geometry_bounds_um(geom) -> Bounds:
    """The ``((xlo, xhi), (ylo, yhi), (zlo, zhi))`` axis-aligned bounding box of
    a geometry primitive, in microns. Outer (containing) bound for curved /
    slanted shapes."""
    if isinstance(geom, Box):
        c, s = geom.center_um, geom.size_um
        return tuple((c[a] - s[a] / 2.0, c[a] + s[a] / 2.0) for a in range(3))

    if isinstance(geom, Sphere):
        c, r = geom.center_um, geom.radius_um
        return tuple((c[a] - r, c[a] + r) for a in range(3))

    if isinstance(geom, Cylinder):
        a = "xyz".index(geom.axis)
        c, r = geom.center_um, geom.radius_um
        half = geom.length_um / 2.0
        tu, tv = [ax for ax in range(3) if ax != a]     # transverse axes in index order, angles by atan2(v, u)
        a0, a1 = float(geom.angle_start_rad), float(geom.angle_stop_rad)
        out = [None, None, None]
        out[a] = (c[a] - half, c[a] + half)
        if a1 - a0 >= 2.0 * math.pi - 1e-12:
            # a full ring: the outer radius bounds both transverse axes
            out[tu], out[tv] = (c[tu] - r, c[tu] + r), (c[tv] - r, c[tv] + r)
            return tuple(out)
        # a sector: the arc's end points at both radii, plus every axis crossing
        # the outer arc sweeps through (the tight box a 90-degree bend needs)
        us, vs = [], []
        for radius in (float(geom.inner_radius_um), r):
            for t in (a0, a1):
                us.append(radius * math.cos(t))
                vs.append(radius * math.sin(t))
        k = math.ceil(a0 / (math.pi / 2))
        while k * math.pi / 2 <= a1 + 1e-12:
            t = k * math.pi / 2
            us.append(r * math.cos(t))
            vs.append(r * math.sin(t))
            k += 1
        out[tu] = (c[tu] + min(us), c[tu] + max(us))
        out[tv] = (c[tv] + min(vs), c[tv] + max(vs))
        return tuple(out)

    if isinstance(geom, Polygon):
        a = "xyz".index(geom.axis)
        lo, hi = geom.slab_bounds_um
        # Transverse axes in index order: vertices are (u, v) with u the
        # lower-indexed and v the higher-indexed transverse axis (structures.py).
        tu, tv = [ax for ax in range(3) if ax != a]
        us = [v[0] for v in geom.vertices_um]
        vs = [v[1] for v in geom.vertices_um]
        # Slanted walls dilate the cross-section away from the reference plane by
        # up to |tan(angle)| * slab_thickness; pad both transverse extents
        # outward so the box still contains the widest section (conservative).
        pad = abs(math.tan(geom.sidewall_angle)) * (hi - lo)
        out = [None, None, None]
        out[a] = (lo, hi)
        out[tu] = (min(us) - pad, max(us) + pad)
        out[tv] = (min(vs) - pad, max(vs) + pad)
        return tuple(out)

    raise TypeError(f"geometry_bounds_um: unknown geometry {type(geom).__name__}")
