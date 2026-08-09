"""Axis-aligned bounding boxes for the geometry primitives.

Used by the material-aware boundary selection (which structures reach into the
PML/absorber region on a given face — :meth:`Simulation.with_auto_boundaries`).
Every returned box CONTAINS the geometry, so a "does it cross the boundary"
test is conservative: it can over-report a crossing (slanted PolySlab, annular
Cylinder), never miss one.
"""

import math
from typing import Tuple

from .structures import Box, Cylinder, PolySlab, Sphere

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
        out = []
        for ax in range(3):
            if ax == a:
                out.append((c[ax] - half, c[ax] + half))
            else:
                # Outer radius bounds the annular sector on both transverse axes.
                out.append((c[ax] - r, c[ax] + r))
        return tuple(out)

    if isinstance(geom, PolySlab):
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
