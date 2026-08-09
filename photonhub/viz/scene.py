"""``plot()`` — analytic 2D cross-section of a :class:`Simulation` on a cut
plane (design §3, §6).

Each Box/Sphere is intersected with the plane (the §5 cut-plane geometry) and
drawn as a matplotlib patch (Rectangle/Circle), facecolor mapped from
``medium.permittivity`` via the shared ε colormap over a background fill. The
shared overlay glyphs (sources, monitors, PML) and a compact legend are added.
No grid, no engine — exact analytic shapes, instant.
"""

import warnings

from matplotlib.patches import Rectangle

from . import _geometry as geom
from . import _style


def plot(sim, x=None, y=None, z=None, *, ax=None, legend=True, grid=False, **kw):
    """Draw the analytic scene cross-section on the selected cut plane and
    return the matplotlib ``Axes``.

    Exactly one of x/y/z (microns) selects the constant-coordinate cut plane;
    not exactly one -> ValueError. A cut outside the domain yields an empty
    Axes plus a warning (design §9). ``grid=True`` overlays the realized Yee
    cell edges (the mesh-resolution sanity check). Extra ``**kw`` is forwarded
    to each structure patch."""
    import matplotlib.pyplot as plt

    axis, value = geom.select_plane(x, y, z)
    if ax is None:
        _, ax = plt.subplots()

    realized = sim._realized_um()
    a = geom.axis_index(axis)
    h_ax, v_ax = geom.in_plane_axes(axis)
    h_i = "xyz".index(h_ax)
    v_i = "xyz".index(v_ax)

    # Background fill across the whole domain (the §6 background ε fill).
    eps_vals = ([sim.background.permittivity]
                + [s.medium.permittivity for s in sim.structures])
    vmin, vmax = _style.eps_norm(eps_vals)
    bg_color = _style.eps_facecolor(sim.background.permittivity, vmin, vmax)
    ax.add_patch(Rectangle((0.0, 0.0), realized[h_i], realized[v_i],
                           facecolor=bg_color, edgecolor="none", zorder=0))

    out_of_domain = not (0.0 <= value <= realized[a])
    if out_of_domain:
        warnings.warn(
            f"{axis}={value} um is outside the realized domain "
            f"[0, {realized[a]:.6g}] um on that axis; only the background and "
            "out-of-plane overlays are drawn",
            UserWarning, stacklevel=2)

    drew_structure = False
    if not out_of_domain:
        for structure in sim.structures:
            spec = geom.structure_patch_spec(structure.geometry, axis, value)
            if spec is None:
                continue  # structure does not intersect the plane (design §9)
            color = _style.eps_facecolor(structure.medium.permittivity,
                                         vmin, vmax)
            kind, params = spec
            _add_filled_patch(ax, kind, params, color, **kw)
            drew_structure = True

    if grid and not out_of_domain:
        _style.draw_grid(ax, sim, axis)

    drew = _style.draw_overlays(ax, sim, axis, value)

    ax.set_xlim(0.0, realized[h_i])
    ax.set_ylim(0.0, realized[v_i])
    ax.set_aspect("equal")
    ax.set_xlabel(f"{h_ax} (µm)")
    ax.set_ylabel(f"{v_ax} (µm)")
    ax.set_title(f"scene  ({axis}-cut)")

    if legend:
        _style.add_legend(ax, source=drew["source"], monitor=drew["monitor"],
                          pml=drew["pml"], structure=drew_structure)
    return ax


def _add_filled_patch(ax, kind, params, color, **kw):
    """Add one ε-colored, edged structure patch for a §5 cut-plane spec.
    Delegates the kind dispatch to :func:`_style.add_structure_patch` (shared
    with the field-overlay outlines)."""
    style = dict(facecolor=color, edgecolor=_style.STRUCTURE_EDGE,
                 linewidth=0.8, zorder=2, **kw)
    _style.add_structure_patch(ax, kind, params, style=style)
