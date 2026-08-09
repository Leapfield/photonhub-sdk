"""``plot_3d()`` — interactive 3D geometry as a plotly figure (design §3, §6).

Structures become ``Mesh3d`` (boxes -> 8 verts / 12 tris; spheres ->
parametric mesh), sources ``Scatter3d`` markers / translucent planes, monitors
point markers / translucent boxes / planes, plus a wireframe box of the physical
region. By default the PML is not drawn: the wireframe sits at the inner (non-PML)
boundary and geometry is clipped to it; ``show_pml=True`` restores the full
domain wireframe with translucent PML shells and unclipped geometry.

plotly is the optional ``photonhub[viz]`` extra and is imported LAZILY: when it
is absent, :func:`plot_3d` raises ``ImportError`` with the exact
``pip install photonhub[viz]`` hint (design §8).
"""

import hashlib
import math

from . import _geometry as geom
from . import _style

_PLOTLY_HINT = (
    "plot_3d requires plotly (the optional 3D viz extra). Install it with:\n"
    "    pip install photonhub[viz]"
)

# Plotly's default diffuse lighting exaggerates differences between a box's
# broad faces and a densely faceted bend. Keep material color dominant while
# retaining a small amount of directional depth. This contract is applied only
# to solid simulation structures, not translucent authoring overlays.
_SOLID_MATERIAL_LIGHTING = {
    "ambient": 0.95,
    "diffuse": 0.15,
    "specular": 0.0,
    "fresnel": 0.0,
    "roughness": 1.0,
}


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without plotly
        raise ImportError(_PLOTLY_HINT) from exc
    return go


def _eps_color(permittivity, vmin, vmax):
    r, g, b, _ = _style.eps_facecolor(permittivity, vmin, vmax)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def _box_mesh(go, center, size, color, name, opacity=1.0, lighting=None):
    cx, cy, cz = center
    hx, hy, hz = (s / 2.0 for s in size)
    xs = [cx - hx, cx + hx, cx + hx, cx - hx, cx - hx, cx + hx, cx + hx, cx - hx]
    ys = [cy - hy, cy - hy, cy + hy, cy + hy, cy - hy, cy - hy, cy + hy, cy + hy]
    zs = [cz - hz, cz - hz, cz - hz, cz - hz, cz + hz, cz + hz, cz + hz, cz + hz]
    # 12 triangles (two per face), standard unit-cube triangulation.
    i = [0, 0, 0, 0, 4, 4, 6, 6, 1, 1, 2, 2]
    j = [1, 2, 4, 3, 5, 6, 5, 7, 5, 6, 6, 7]
    k = [2, 3, 5, 7, 6, 7, 1, 3, 6, 2, 7, 3]
    kw = {"lighting": lighting} if lighting is not None else {}
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color,
                     opacity=opacity, name=name, showscale=False,
                     flatshading=True, **kw)


def _sphere_mesh(go, center, radius, color, name, n=16, bounds=None,
                 lighting=None):
    cx, cy, cz = center
    us = [math.pi * a / n for a in range(n + 1)]          # polar [0, pi]
    vs = [2 * math.pi * b / n for b in range(n + 1)]      # azimuth [0, 2pi]
    xs, ys, zs = [], [], []
    for u in us:
        for v in vs:
            xs.append(cx + radius * math.sin(u) * math.cos(v))
            ys.append(cy + radius * math.sin(u) * math.sin(v))
            zs.append(cz + radius * math.cos(u))
    if bounds is not None:
        # Ball ∩ box is convex, so clamping the surface samples into the box
        # and taking the convex hull (alphahull=0) approximates the clip.
        xs = [min(max(x, bounds[0][0]), bounds[0][1]) for x in xs]
        ys = [min(max(y, bounds[1][0]), bounds[1][1]) for y in ys]
        zs = [min(max(z, bounds[2][0]), bounds[2][1]) for z in zs]
    kw = {"lighting": lighting} if lighting is not None else {}
    return go.Mesh3d(x=xs, y=ys, z=zs, alphahull=0, color=color, opacity=1.0,
                     name=name, showscale=False, flatshading=True, **kw)


def _sphere_intersects(center, radius, bounds):
    """True iff the ball reaches into the axis-aligned box ``bounds``."""
    d2 = 0.0
    for ci, (lo, hi) in zip(center, bounds):
        nearest = min(max(ci, lo), hi)
        d2 += (nearest - ci) ** 2
    return d2 <= radius ** 2


def _clip_box(center, size, bounds):
    """Intersect an axis-aligned box with ``bounds``; ``None`` when empty."""
    c, s = [], []
    for i in range(3):
        lo = max(center[i] - size[i] / 2.0, bounds[i][0])
        hi = min(center[i] + size[i] / 2.0, bounds[i][1])
        if hi - lo <= 0:
            return None
        c.append(0.5 * (lo + hi))
        s.append(hi - lo)
    return c, s


def _clip_polygon_uv(verts, u_bounds, v_bounds):
    """Sutherland–Hodgman clip of a transverse polygon to the axis-aligned
    rectangle ``u_bounds`` x ``v_bounds``. Returns the (possibly empty) clipped
    vertex list."""
    pts = [(p[0], p[1]) for p in verts]
    planes = ((0, u_bounds[0], 1), (0, u_bounds[1], -1),
              (1, v_bounds[0], 1), (1, v_bounds[1], -1))
    for ci, bound, sign in planes:
        if not pts:
            return []
        nxt = []
        n = len(pts)
        for idx in range(n):
            a, b = pts[idx], pts[(idx + 1) % n]
            ina = sign * (a[ci] - bound) >= 0
            inb = sign * (b[ci] - bound) >= 0
            if ina != inb:
                t = (bound - a[ci]) / (b[ci] - a[ci])
                cross = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if ina:
                nxt.append(a)
                if not inb:
                    nxt.append(cross)
            elif inb:
                nxt.append(cross)
        pts = nxt
    return pts


# Axis-letter -> (transverse u index, transverse v index, axial index). Mirrors
# the (u, v) transverse convention in viz/_geometry.in_plane_axes.
_AXIS_FRAME = {
    "x": (1, 2, 0),  # u=y, v=z, axial=x
    "y": (0, 2, 1),  # u=x, v=z, axial=y
    "z": (0, 1, 2),  # u=x, v=y, axial=z
}
# Sign of ``cross(e_u, e_v)`` along the positive axial direction. The y frame
# is left-handed because x cross z points toward -y.
_AXIS_FRAME_HANDEDNESS = {"x": 1.0, "y": -1.0, "z": 1.0}


def _embed_uv(u, v, w, u_i, v_i, a_i):
    """Place a transverse ``(u, v)`` point at axial coordinate ``w`` into a 3D
    ``(x, y, z)`` tuple using the axis frame ``(u_i, v_i, a_i)``."""
    p = [0.0, 0.0, 0.0]
    p[u_i] = u
    p[v_i] = v
    p[a_i] = w
    return p[0], p[1], p[2]


def _cross_2d(a, b, c):
    """Signed twice-area of triangle ``a, b, c`` in polygon coordinates."""
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))


def _polygon_area2(vertices):
    """Signed twice-area of an open polygon ring."""
    return sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(vertices, vertices[1:] + vertices[:1])
    )


def _point_in_triangle(point, a, b, c, orientation, eps, *, strict=False):
    """Return whether ``point`` lies in the oriented triangle ``a, b, c``.

    Boundary points normally count as inside so an ear diagonal cannot skip a
    polygon vertex. ``strict=True`` is a numerical fallback for weakly simple
    rings whose duplicate/collinear boundary samples would otherwise block
    every remaining ear.
    """
    sides = (
        orientation * _cross_2d(a, b, point),
        orientation * _cross_2d(b, c, point),
        orientation * _cross_2d(c, a, point),
    )
    return min(sides) > eps if strict else min(sides) >= -eps


def _triangulate_polygon(vertices):
    """Triangulate a simple polygon ring without changing its sampled outline.

    Ear clipping is deliberately used instead of a vertex-0 fan: a fan covers
    the notch of a concave bend whenever vertex 0 cannot see the whole polygon.
    Returned indices refer to the original ring. Straight-through collinear
    vertices stay available to the side walls but need not appear in a cap
    triangle, so dense GDS sampling remains untouched.
    """
    n = len(vertices)
    if n < 3:
        raise ValueError("a polygon cap needs at least three vertices")

    area2 = _polygon_area2(vertices)
    u_values = [point[0] for point in vertices]
    v_values = [point[1] for point in vertices]
    span = max(
        max(u_values) - min(u_values),
        max(v_values) - min(v_values),
        1.0,
    )
    eps = 1e-12 * span * span
    if abs(area2) <= eps:
        raise ValueError("cannot triangulate a zero-area polygon cap")
    orientation = 1.0 if area2 > 0.0 else -1.0

    active = list(range(n))
    # Remove only redundant, straight-through vertices from the triangulation
    # ring. They remain in ``vertices`` and therefore in the rendered sidewall.
    changed = True
    while changed and len(active) > 3:
        changed = False
        for position, current in enumerate(active):
            previous = active[position - 1]
            following = active[(position + 1) % len(active)]
            a, b, c = (vertices[previous], vertices[current],
                       vertices[following])
            if abs(_cross_2d(a, b, c)) > eps:
                continue
            # b must lie between a and c. A collinear reversal is not a
            # redundant sample and must remain for validation below.
            if ((b[0] - a[0]) * (b[0] - c[0])
                    + (b[1] - a[1]) * (b[1] - c[1])) <= eps:
                del active[position]
                changed = True
                break

    triangles = []
    while len(active) > 3:
        ear = None
        # Boundary-inclusive containment is the normal/simple-ring path.
        # Strict containment is a safe second pass for numerically duplicated
        # samples; the final area check still prevents an over-covered cap.
        for strict in (False, True):
            for position, current in enumerate(active):
                previous = active[position - 1]
                following = active[(position + 1) % len(active)]
                a, b, c = (vertices[previous], vertices[current],
                           vertices[following])
                if orientation * _cross_2d(a, b, c) <= eps:
                    continue
                if any(
                    _point_in_triangle(
                        vertices[index], a, b, c, orientation, eps,
                        strict=strict,
                    )
                    for index in active
                    if index not in (previous, current, following)
                ):
                    continue
                ear = (position, (previous, current, following))
                break
            if ear is not None:
                break
        if ear is None:
            raise ValueError(
                "cannot triangulate polygon cap; vertices must form a simple ring"
            )
        position, triangle = ear
        triangles.append(triangle)
        del active[position]

    if orientation * _cross_2d(
            vertices[active[0]], vertices[active[1]],
            vertices[active[2]]) > eps:
        triangles.append(tuple(active))

    cap_area = sum(
        abs(_cross_2d(vertices[a], vertices[b], vertices[c]))
        for a, b, c in triangles
    )
    if not math.isclose(cap_area, abs(area2), rel_tol=1e-9,
                        abs_tol=eps * max(1, n)):
        raise ValueError(
            "polygon cap triangulation did not preserve the polygon area"
        )
    return triangles


def _prism_mesh(go, axis, vertices_uv, axial_lo, axial_hi, color, name,
                opacity=1.0, cap_triangles=None, lighting=None):
    """A polygon (``vertices_uv`` in transverse (u, v)) extruded along ``axis``
    between ``axial_lo`` and ``axial_hi`` as a closed ``Mesh3d`` prism.

    ``cap_triangles`` may provide a truthful triangulation for a concave ring;
    otherwise the legacy fan is used by parametric cylinder polygons. Sidewall
    slant is approximated as vertical (``sidewall_angle`` ignored), documented
    at the call site. Triangle indices are wound outward without reordering the
    sampled polygon vertices, for either ring orientation and every axis."""
    n = len(vertices_uv)
    u_i, v_i, a_i = _AXIS_FRAME[axis]
    xs, ys, zs = [], [], []
    for (u, v) in vertices_uv:                       # bottom ring [0, n)
        x, y, z = _embed_uv(u, v, axial_lo, u_i, v_i, a_i)
        xs.append(x); ys.append(y); zs.append(z)
    for (u, v) in vertices_uv:                       # top ring [n, 2n)
        x, y, z = _embed_uv(u, v, axial_hi, u_i, v_i, a_i)
        xs.append(x); ys.append(y); zs.append(z)
    i, j, k = [], [], []
    # A forward ring triangle points toward ``frame_handedness * ring_sign``
    # along the axial direction. Use it for the top only when that sign is
    # positive; the bottom must always point the other way.
    forward_is_top = (
        _AXIS_FRAME_HANDEDNESS[axis] * _polygon_area2(vertices_uv) > 0.0
    )
    if cap_triangles is None:
        cap_triangles = [(0, t, t + 1) for t in range(1, n - 1)]
    for a, b, c in cap_triangles:
        if forward_is_top:
            i.append(a); j.append(c); k.append(b)
            i.append(n + a); j.append(n + b); k.append(n + c)
        else:
            i.append(a); j.append(b); k.append(c)
            i.append(n + a); j.append(n + c); k.append(n + b)
    # Walls: two triangles per edge connecting bottom ring to top ring.
    for e in range(n):
        b0, b1 = e, (e + 1) % n
        t0, t1 = n + e, n + (e + 1) % n
        if forward_is_top:
            i.append(b0); j.append(b1); k.append(t1)
            i.append(b0); j.append(t1); k.append(t0)
        else:
            i.append(b0); j.append(t1); k.append(b1)
            i.append(b0); j.append(t0); k.append(t1)
    kw = {"lighting": lighting} if lighting is not None else {}
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color,
                     opacity=opacity, name=name, showscale=False,
                     flatshading=True, **kw)


def _clipped_prism(go, axis, verts, lo, hi, color, name, bounds,
                   *, triangulate_caps=False, lighting=None):
    """A prism clipped to ``bounds`` (axial extent + transverse polygon);
    ``None`` when nothing remains. ``bounds=None`` skips clipping."""
    if bounds is not None:
        u_i, v_i, a_i = _AXIS_FRAME[axis]
        lo = max(lo, bounds[a_i][0])
        hi = min(hi, bounds[a_i][1])
        verts = _clip_polygon_uv(verts, bounds[u_i], bounds[v_i])
        if hi <= lo or len(verts) < 3:
            return None
    cap_triangles = _triangulate_polygon(verts) if triangulate_caps else None
    return _prism_mesh(
        go, axis, verts, lo, hi, color, name,
        cap_triangles=cap_triangles, lighting=lighting,
    )


def _polyslab_mesh(go, g, color, name, bounds=None, lighting=None):
    """PolySlab -> extruded-polygon prism. ``slab_bounds_um`` are the axial
    (lo, hi); the polygon lives in the two transverse axes. Vertical walls
    (sidewall_angle approximated as 0)."""
    lo, hi = g.slab_bounds_um
    return _clipped_prism(go, g.axis, g.vertices_um, lo, hi, color, name,
                          bounds, triangulate_caps=True, lighting=lighting)


def _cylinder_mesh(go, g, color, name, n=48, bounds=None, lighting=None):
    """Cylinder (solid disk / ring / annular sector) -> a parametric tube
    ``Mesh3d``. The arc is sampled into an (annular) polygon cross-section that
    is then extruded along ``axis`` by :func:`_prism_mesh`. A solid disk
    (``inner_radius_um == 0``) reduces to the outer-arc polygon closed through
    the centre; the curved wall is faceted into ``n`` segments per full turn."""
    a_i = "xyz".index(g.axis)
    u_i, v_i, _ = _AXIS_FRAME[g.axis]
    cu = g.center_um[u_i]
    cv = g.center_um[v_i]
    half = g.length_um / 2.0
    lo = g.center_um[a_i] - half
    hi = g.center_um[a_i] + half
    verts = geom._arc_polygon(cu, cv, g.radius_um, g.inner_radius_um,
                              g.angle_start, g.angle_stop, n=n)
    return _clipped_prism(
        go, g.axis, verts, lo, hi, color, name, bounds,
        lighting=lighting,
    )


def _domain_wireframe(go, bounds):
    """12 edges of the box spanned by per-axis ``(lo, hi)`` ``bounds`` as one
    Scatter3d."""
    (x0, x1), (y0, y1), (z0, z1) = bounds
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [c[a][0], c[b][0], None]
        ys += [c[a][1], c[b][1], None]
        zs += [c[a][2], c[b][2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color="#444", width=2), name="domain",
                        showlegend=True)


def _tag_monitor_trace(trace, monitor, index, uid_suffix=""):
    """Attach the stable, machine-readable identity used by interactive UIs.

    ``name`` remains the human-readable / backwards-compatible legend label;
    callers should prefer ``meta.photonhub`` when resolving a clicked trace.
    Monitor names are unique within a Simulation, so a digest of the name also
    makes a stable Plotly ``uid`` when monitors are reordered.  The digest is
    intentionally CSS-safe: Plotly interpolates ``uid`` into selectors during
    cleanup, so punctuation from a human-facing monitor name cannot be used
    directly here.
    """
    kind = "port" if getattr(monitor, "mode_port", None) is not None else "monitor"
    trace.meta = {
        "photonhub": {
            "kind": kind,
            "id": monitor.name,
            "index": index,
        },
    }
    name_digest = hashlib.sha256(monitor.name.encode("utf-8")).hexdigest()[:16]
    suffix = (hashlib.sha256(uid_suffix.encode("utf-8")).hexdigest()[:8]
              if uid_suffix else "")
    trace.uid = f"photonhub-{kind}-{name_digest}{suffix}"
    return trace


def _interior_bounds_um(sim):
    """Per-axis ``(lo, hi)`` of the physical interior — the realized domain
    minus the PML slabs. An axis whose PML halves meet or overlap (possible in
    tiny schematic scenes) falls back to the full axis extent."""
    from ._style import _axis_spacings_um

    realized = sim._realized_um()
    layers = sim.pml_num_layers
    bounds = []
    for axis_i, letter in enumerate("xyz"):
        lo, hi = 0.0, realized[axis_i]
        if getattr(sim.boundaries, letter) == "pml":
            lo_dl, hi_dl, length = _axis_spacings_um(sim, axis_i)
            inner_lo, inner_hi = layers * lo_dl, length - layers * hi_dl
            if inner_hi > inner_lo:
                lo, hi = inner_lo, inner_hi
        bounds.append((lo, hi))
    return bounds


def _pml_shells(go, sim):
    """Translucent slab shells on each non-periodic (PML) boundary face."""
    from ._style import _axis_spacings_um

    shells = []
    realized = sim._realized_um()
    layers = sim.pml_num_layers
    for axis_i, letter in enumerate("xyz"):
        if getattr(sim.boundaries, letter) != "pml":
            continue
        lo_dl, hi_dl, length = _axis_spacings_um(sim, axis_i)
        lo_t = layers * lo_dl
        hi_t = layers * hi_dl
        for face_lo, face_hi, tag in (
            (0.0, lo_t, "low"), (length - hi_t, length, "high"),
        ):
            size = list(realized)
            center = [r / 2.0 for r in realized]
            size[axis_i] = face_hi - face_lo
            center[axis_i] = 0.5 * (face_lo + face_hi)
            shells.append(_box_mesh(go, center, size, _style.PML_COLOR,
                                   f"PML {letter}-{tag}", opacity=0.12))
    return shells


def plot_3d(sim, show_pml=False, **kw):
    """Build and return a plotly ``Figure`` of the 3D scene geometry. Raises
    ``ImportError`` (with the install hint) when plotly is not installed.

    By default the PML is not drawn: the wireframe marks the inner (non-PML)
    boundary and structures / planes are clipped to it. ``show_pml=True``
    restores the full-domain wireframe, translucent PML shells, and unclipped
    geometry."""
    go = _require_plotly()
    fig = go.Figure()

    realized = sim._realized_um()
    bounds = ([(0.0, r) for r in realized] if show_pml
              else _interior_bounds_um(sim))
    clip = None if show_pml else bounds

    eps_vals = ([sim.background.permittivity]
                + [s.medium.permittivity for s in sim.structures])
    vmin, vmax = _style.eps_norm(eps_vals)

    # Structures (clipped to the physical interior unless show_pml).
    for n, structure in enumerate(sim.structures):
        g = structure.geometry
        color = _eps_color(structure.medium.permittivity, vmin, vmax)
        gtype = getattr(g, "type", None)
        if gtype == "box":
            center, size = g.center_um, g.size_um
            if clip is not None:
                clipped = _clip_box(center, size, clip)
                if clipped is None:
                    continue
                center, size = clipped
            fig.add_trace(_box_mesh(
                go, center, size, color, f"structure{n}",
                lighting=_SOLID_MATERIAL_LIGHTING,
            ))
        elif gtype == "sphere":
            if clip is not None and not _sphere_intersects(
                    g.center_um, g.radius_um, clip):
                continue
            fig.add_trace(_sphere_mesh(go, g.center_um, g.radius_um, color,
                                      f"structure{n}", bounds=clip,
                                      lighting=_SOLID_MATERIAL_LIGHTING))
        elif gtype == "polyslab":
            mesh = _polyslab_mesh(
                go, g, color, f"structure{n}", bounds=clip,
                lighting=_SOLID_MATERIAL_LIGHTING,
            )
            if mesh is not None:
                fig.add_trace(mesh)
        elif gtype == "cylinder":
            mesh = _cylinder_mesh(
                go, g, color, f"structure{n}", bounds=clip,
                lighting=_SOLID_MATERIAL_LIGHTING,
            )
            if mesh is not None:
                fig.add_trace(mesh)

    # Put non-interactive boundary guides behind authoring handles in Plotly's
    # trace pick order. A monitor near the domain wall (the normal placement for
    # an output port) must win a click over the projected wireframe.
    fig.add_trace(_domain_wireframe(go, bounds))
    if show_pml:
        for shell in _pml_shells(go, sim):
            fig.add_trace(shell)

    # Sources.
    for n, s in enumerate(sim.sources):
        stype = getattr(s, "type", None)
        if stype == "point_dipole":
            cx, cy, cz = s.center_um
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz], mode="markers",
                marker=dict(size=6, color=_style.SOURCE_COLOR),
                name=f"source{n}"))
        elif stype in {"plane_wave", "mode_source"}:
            # Both sources inject on a plane normal to ``axis``. ModeSource's
            # solved transverse profile lives in its wire arrays, but it has no
            # separate authoring-space window; the full-domain plane is the
            # honest scene marker and keeps guided launches discoverable in the
            # same way as plane waves.
            fig.add_trace(_plane_mesh(go, s.axis, s.position_um, bounds,
                                     _style.SOURCE_COLOR, f"source{n}"))

    # Monitors. Render broad overview planes first so smaller, more precise
    # port/probe handles sit later in Plotly's pick order and win clicks where
    # the planes overlap (for example field_z0 behind a modal output port).
    def monitor_pick_footprint(item):
        _, monitor = item
        monitor_type = getattr(monitor, "type", None)
        if monitor_type == "field_time":
            return 0.0
        if monitor_type == "field_dft":
            sizes = sorted((abs(float(v)) for v in monitor.size_um), reverse=True)
            return sizes[0] * sizes[1]
        if monitor_type == "flux":
            normal = "xyz".index(monitor.axis)
            transverse_spans = [
                hi - lo for axis, (lo, hi) in enumerate(bounds)
                if axis != normal
            ]
            return transverse_spans[0] * transverse_spans[1]
        return 0.0

    monitor_items = sorted(
        enumerate(sim.monitors), key=lambda item: -monitor_pick_footprint(item))
    domain_spans = [hi - lo for lo, hi in bounds]
    largest_domain_face = max(
        domain_spans[0] * domain_spans[1],
        domain_spans[0] * domain_spans[2],
        domain_spans[1] * domain_spans[2],
    )
    for n, m in monitor_items:
        mtype = getattr(m, "type", None)
        if mtype == "field_time":
            cx, cy, cz = m.center_um
            # A point probe commonly sits inside an opaque structure.  Keep the
            # physical point, and add a short leader to a visible handle on the
            # top of the plotted domain so it remains discoverable/clickable.
            y_lo, y_hi = bounds[1]
            z_lo, z_hi = bounds[2]
            # Keep the handle just inside the scene clip volume; points exactly
            # on Plotly's 3D boundary can be clipped by the WebGL depth pass.
            handle_y = y_hi - 0.08 * (y_hi - y_lo)
            handle_z = z_hi - 0.08 * (z_hi - z_lo)
            has_leader = not (math.isclose(cy, handle_y) and math.isclose(cz, handle_z))
            ys = [cy, handle_y] if has_leader else [cy]
            zs = [cz, handle_z] if has_leader else [cz]
            trace = go.Scatter3d(
                x=[cx] * len(zs), y=ys, z=zs,
                mode="markers" if len(zs) == 1 else "lines+markers",
                marker=dict(
                    size=7,
                    color=_style.MONITOR_COLOR,
                    symbol="diamond",
                    line=dict(color="#ffffff", width=1),
                ),
                line=dict(color=_style.MONITOR_COLOR, width=3, dash="dot"),
                name=f"monitor:{m.name}",
            )
            fig.add_trace(_tag_monitor_trace(trace, m, n))
        elif mtype == "field_dft":
            # A mathematically zero-thickness monitor is difficult to pick in
            # a perspective WebGL scene. Give only its zero axes a 1% domain
            # click volume: still visually planar, but forgiving enough for a
            # normal pointer or touchpad interaction.
            size = [
                max(d, 0.01 * (bounds[i][1] - bounds[i][0]), 1e-6)
                for i, d in enumerate(m.size_um)
            ]
            center = m.center_um
            if clip is not None:
                clipped = _clip_box(center, size, clip)
                if clipped is None:
                    continue
                center, size = clipped
            is_overview_plane = (
                monitor_pick_footprint((n, m)) >= 0.25 * largest_domain_face)
            if not is_overview_plane:
                trace = _box_mesh(
                    go, center, size, _style.MONITOR_COLOR,
                    f"monitor:{m.name}", opacity=0.2)
                fig.add_trace(_tag_monitor_trace(trace, m, n))
            if is_overview_plane:
                # A full-domain sampling plane would intercept nearly every
                # scene click. Represent it with a dedicated leader/diamond;
                # the editor still exposes its exact region numerically.
                normal_axes = [
                    axis for axis, extent in enumerate(m.size_um)
                    if math.isclose(extent, 0.0)
                ]
                handle_axis = normal_axes[0] if normal_axes else 2
                handle = list(m.center_um)
                lo, hi = bounds[handle_axis]
                handle[handle_axis] = hi - 0.08 * (hi - lo)
                coordinates = [list(m.center_um), handle]
                handle_trace = go.Scatter3d(
                    x=[point[0] for point in coordinates],
                    y=[point[1] for point in coordinates],
                    z=[point[2] for point in coordinates],
                    mode="lines+markers",
                    marker=dict(
                        size=7,
                        color=_style.MONITOR_COLOR,
                        symbol="diamond",
                        line=dict(color="#ffffff", width=1),
                    ),
                    line=dict(color=_style.MONITOR_COLOR, width=3, dash="dot"),
                    name=f"monitor:{m.name}",
                    showlegend=False,
                )
                fig.add_trace(_tag_monitor_trace(
                    handle_trace, m, n, uid_suffix="handle"))
        elif mtype == "flux":
            trace = _plane_mesh(go, m.axis, m.position_um, bounds,
                                _style.MONITOR_COLOR, f"monitor:{m.name}")
            fig.add_trace(_tag_monitor_trace(trace, m, n))

    spans = sorted((hi - lo for lo, hi in bounds), reverse=True)
    elongation = spans[0] / max(spans[1], 1e-12)
    # Plotly's fixed default eye (1.25, 1.25, 1.25) can sit effectively inside
    # a long photonic circuit when aspectmode="data", cropping most of the
    # device on first paint. Pull back only as much as the two largest domain
    # spans require; ordinary near-square scenes retain the familiar camera.
    eye_distance = min(8.0, max(1.25, 0.65 * elongation))
    fig.update_layout(
        scene=dict(
            xaxis_title="x (µm)", yaxis_title="y (µm)", zaxis_title="z (µm)",
            aspectmode="data",
            camera=dict(eye=dict(
                x=eye_distance, y=eye_distance, z=eye_distance,
            )),
        ),
        # No title: the pane's own "3D view" tab already names this figure,
        # and the reclaimed strip goes to the scene.
        margin=dict(l=0, r=0, t=8, b=0),
    )
    return fig


def _plane_mesh(go, axis, position, bounds, color, name):
    """A translucent plane perpendicular to ``axis`` at ``position``, spanning
    the per-axis ``(lo, hi)`` ``bounds`` (PlaneWave source / FluxMonitor)."""
    ai = "xyz".index(axis)
    size = [hi - lo for lo, hi in bounds]
    center = [0.5 * (lo + hi) for lo, hi in bounds]
    size[ai] = max(size[ai] * 1e-3, 1e-6)  # near-zero thickness
    center[ai] = position
    return _box_mesh(go, center, size, color, name, opacity=0.2)
