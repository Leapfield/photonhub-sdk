"""``plot()``, analytic 2D cross-section of a :class:`Simulation` on a cut
plane.

Each Box/Sphere is intersected with the plane (the §5 cut-plane geometry) and
drawn as a matplotlib patch (Rectangle/Circle), facecolor mapped from
``medium.permittivity`` via the shared ε colormap over a background fill. The
shared overlay glyphs (sources, monitors, PML) and a compact legend are added.
No grid, no engine, exact analytic shapes, instant.
"""

import warnings

from matplotlib.patches import Rectangle

from . import _geometry as geom
from . import _style
from .._compat import caller_stacklevel


def plot(sim, x=None, y=None, z=None, *, ax=None, legend=True, grid=False,
         unfold=True, **kw):
    """Draw the analytic scene cross-section on the selected cut plane and
    return the matplotlib ``Axes``.

    Exactly one of x/y/z (microns) selects the constant-coordinate cut plane;
    not exactly one -> ValueError. A cut outside the realized domain warns and
    omits structure cross-sections and the grid. The background, applicable
    overlays, symmetry markers, and domain limits are retained.
    ``grid=True`` overlays the realized Yee
    cell edges (the mesh-resolution sanity check). ``unfold`` (the default)
    mirrors the half domain back across each in-plane §20 symmetry plane, so
    the picture is the whole device; ``unfold=False`` draws the reduced domain
    the solver actually steps. Extra ``**kw`` is forwarded to each structure
    patch."""
    import matplotlib.pyplot as plt

    axis, value = geom.select_plane(x, y, z)
    if ax is None:
        _, ax = plt.subplots()
    before = set(ax.get_children())

    realized = sim._realized_um()
    a = geom.axis_index(axis)
    # The cut is given in the user's frame; the scene is drawn, reflected and
    # clipped in the corner frame, then moved into the user's (spec §4.4).
    origin = geom.frame_origin(sim)
    oh, ov = geom.plane_offsets(origin, axis)
    value_c = value - origin[a]
    h_ax, v_ax = geom.in_plane_axes(axis)
    h_i = "xyz".index(h_ax)
    v_i = "xyz".index(v_ax)

    mirror_h, mirror_v = _style.unfold_in_plane(sim, axis, unfold)
    copies = _style.mirror_copies(mirror_h, mirror_v)

    # Background fill across the whole domain (the §6 background ε fill).
    eps_vals = ([sim.background.permittivity]
                + [s.medium.permittivity for s in sim.structures])
    vmin, vmax = _style.eps_norm(eps_vals)
    bg_color = _style.eps_facecolor(sim.background.permittivity, vmin, vmax)
    lo_h = -realized[h_i] if mirror_h else 0.0
    lo_v = -realized[v_i] if mirror_v else 0.0
    ax.add_patch(Rectangle((lo_h, lo_v), realized[h_i] - lo_h,
                           realized[v_i] - lo_v,
                           facecolor=bg_color, edgecolor="none", zorder=0))

    out_of_domain = not (0.0 <= value_c <= realized[a])
    if out_of_domain:
        warnings.warn(
            f"{axis}={value} um is outside the realized domain "
            f"[{origin[a]:.6g}, {origin[a] + realized[a]:.6g}] um on that axis; only the background and "
            "out-of-plane overlays are drawn",
            UserWarning, stacklevel=caller_stacklevel())

    drew_structure = False
    if not out_of_domain:
        for structure in sim.structures:
            spec = geom.structure_patch_spec(structure.geometry, axis, value_c)
            if spec is None:
                continue  # structure does not intersect the plane (design §9)
            color = _style.eps_facecolor(structure.medium.permittivity,
                                         vmin, vmax)
            for sign_h, sign_v in copies:
                kind, params = geom.reflect_spec(*spec, sign_h, sign_v)
                patch_kw = dict(kw)
                clip = _quadrant_clip(ax, realized, h_i, v_i, sign_h, sign_v,
                                      mirror_h, mirror_v)
                _add_filled_patch(ax, kind, params, color, clip=clip,
                                  **patch_kw)
            drew_structure = True

    if grid and not out_of_domain:
        _style.draw_grid(ax, sim, axis, unfold=unfold)

    drew = _style.draw_overlays(ax, sim, axis, value_c, unfold=unfold)
    drew_symmetry = _style.draw_symmetry_planes(ax, sim, axis, mirror_h,
                                                mirror_v)

    _style.translate_artists(ax, before, oh, ov)
    ax.set_xlim(lo_h + oh, realized[h_i] + oh)
    ax.set_ylim(lo_v + ov, realized[v_i] + ov)
    stretch = _style.set_view_aspect(ax, realized[h_i] - lo_h, realized[v_i] - lo_v)
    ax.set_xlabel(f"{h_ax} (µm)")
    ax.set_ylabel(f"{v_ax} (µm)")
    _style.set_titles(ax, f"scene at {_style.cut_label(axis, value)}",
                      _style.join_notes(_style.mesh_summary(sim),
                                        _style.stretch_note(v_ax, stretch)))

    if legend:
        _style.add_legend(ax, source=drew["source"], monitor=drew["monitor"],
                          pml=drew["pml"], structure=drew_structure,
                          symmetry=drew_symmetry,
                          source_kinds=drew["source_kinds"],
                          monitor_kinds=drew["monitor_kinds"],
                          mode_monitor=drew["mode_monitor"],
                          flux_monitor=drew["flux_monitor"],
                          loc=_legend_corner(sim, axis, value_c, lo_h, lo_v,
                                             realized[h_i], realized[v_i]))
    return ax


def _legend_corner(sim, axis, value, lo_h, lo_v, hi_h, hi_v) -> str:
    """Pick the legend corner from a coarse occupancy sample of the view, so
    the key does not sit on the device. A 32 by 32 mesh is a few thousand
    point tests, nothing next to drawing the scene."""
    import numpy as np

    from .eps import eps_at_points

    n = 32
    h = lo_h + (np.arange(n) + 0.5) / n * (hi_h - lo_h)
    v = lo_v + (np.arange(n) + 0.5) / n * (hi_v - lo_v)
    HH, VV = np.meshgrid(np.abs(h), np.abs(v))   # the mirror image is the same
    eps = eps_at_points(sim, axis, HH, VV, float(value))
    return _style.legend_corner(eps != float(sim.background.permittivity))


def _quadrant_clip(ax, realized, h_i, v_i, sign_h, sign_v, mirror_h,
                   mirror_v):
    """Clip rectangle for one unfolded copy, in data coordinates, or ``None``
    when nothing is unfolded.

    The analytic geometry is written in the FULL device's coordinates, so a
    structure centred on the mirror plane already reaches into the half that
    is not simulated. Without a clip each copy would paint over the other's
    quadrant, and a scene the solver would reject (geometry that is not
    actually mirror-symmetric) would still be drawn as if it were. Clipping
    each copy to its own quadrant shows exactly the half that runs, plus its
    mirror."""
    if not (mirror_h or mirror_v):
        return None
    x0 = 0.0 if sign_h == 1 else -realized[h_i]
    y0 = 0.0 if sign_v == 1 else -realized[v_i]
    if not mirror_h:
        x0, wide = -realized[h_i], 2.0 * realized[h_i]
    else:
        wide = realized[h_i]
    if not mirror_v:
        y0, tall = -realized[v_i], 2.0 * realized[v_i]
    else:
        tall = realized[v_i]
    return Rectangle((x0, y0), wide, tall, transform=ax.transData)


def _add_filled_patch(ax, kind, params, color, clip=None, **kw):
    """Add one ε-colored, edged structure patch for a §5 cut-plane spec.
    Delegates the kind dispatch to :func:`_style.add_structure_patch`, which
    also applies the unfold clip after attaching the patch."""
    style = dict(facecolor=color, edgecolor=_style.STRUCTURE_EDGE,
                 linewidth=0.8, zorder=2, **kw)
    _style.add_structure_patch(ax, kind, params, style=style, clip=clip)
