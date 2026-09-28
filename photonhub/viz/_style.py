"""Shared styling and overlay glyphs: the ε colormap, field-colormap /
normalization selection, overlay glyph colors + drawing, PML bands, and the
compact legend (design §6, §7).

Kept free of plotting-method orchestration so both the 2D views and the 3D
builder map the same permittivity to the same color and draw the same glyphs.
"""

import bisect
from typing import Dict, Iterable, Optional, Tuple

from ..components.grid import (graded_primary_spacings, realized_cells,
                               sim_axis_min_cells)
from . import _geometry as geom

_AXES = "xyz"

# Overlay glyph colors (design §6 — matching the approved mockup's legend).
SOURCE_COLOR = "#ff7f5c"      # coral
MONITOR_COLOR = "#f0a800"     # amber: field (profile) and flux monitors
MODE_MONITOR_COLOR = "#1a9850"  # green: a port's mode monitor
MODE_MONITOR_STYLE = "-."     # dash-dot, apart from the dashed field monitor
PML_COLOR = "#7a7a7a"         # neutral grey
SYMMETRY_COLOR = "#6a6a6a"    # the §20 mirror plane, when a view unfolds it
STRUCTURE_EDGE = "#222222"    # structure outline on the light epsilon views

# Structure outline over a FIELD heatmap. The epsilon views draw on a
# light Blues fill, where near-black reads as ink on paper; a field heatmap
# does not. "magma" is black at zero and "RdBu_r" is deep blue at one end, so
# the same near-black edge disappears over exactly the dark background the
# device sits in. White reads on both, and a soft dark halo keeps it legible
# where the map itself goes light (magma's yellow core, RdBu_r's white zero).
STRUCTURE_EDGE_FIELD = "#ffffff"
STRUCTURE_EDGE_FIELD_HALO = "#00000073"
STRUCTURE_EDGE_FIELD_LW = 1.1

# Primary-grid overlay (the ``grid=True`` mesh sanity-check): thin, light lines
# above the structures/heatmap but below the source/monitor glyphs (zorder 4-5).
GRID_COLOR = "#5a5a5a"
GRID_LW = 0.4
GRID_ALPHA = 0.35
GRID_Z = 2.5

# ε heatmap / structure-fill colormap. Sequential, light->dark with ε.
# Blues, not viridis: the ε views are structure drawings (a few discrete
# material levels), so low ε ≈ paper and high ε = ink. The same constant
# colors the 3D scene meshes via eps_facecolor, keeping "material X looks
# like Y" identical across the 2D cut, the CLI renders, and the 3D view —
# and it frees yellow/orange/cyan to mean source, monitor, and selection.
EPS_CMAP = "Blues"

# Field colormaps by kind (design §7).
_SIGNED_CMAP = "RdBu_r"       # diverging, centered on 0 (real/imag time-domain)
_MAGNITUDE_CMAP = "magma"     # sequential (abs / E / intensity / H)
_PHASE_CMAP = "twilight"      # cyclic, [-pi, pi]


def eps_norm(eps_values: Iterable[float]) -> Tuple[float, float]:
    """(vmin, vmax) for the ε colormap over a set of permittivities. Always a
    finite, non-degenerate range so a single-material scene still renders with
    contrast (pad a flat range)."""
    vals = [float(v) for v in eps_values]
    if not vals:
        return (1.0, 2.0)
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        # Flat ε: pad so structures are visibly distinct from background 1.0.
        return (min(1.0, lo) - 0.01, hi + 1.0)
    return (lo, hi)


def eps_facecolor(permittivity: float, vmin: float, vmax: float):
    """RGBA for a permittivity under the shared ε colormap and (vmin, vmax)."""
    import matplotlib as mpl
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=vmin, vmax=vmax)
    return mpl.colormaps[EPS_CMAP](norm(float(permittivity)))


def field_cmap_and_norm(field: str, val: str, data, cmap: Optional[str] = None):
    """Pick the colormap and a matplotlib ``Normalize`` for a field slice
    (design §7).

    - ``val`` "real"/"imag" of a signed field, or a real (time-domain) field ->
      diverging ``RdBu_r`` centered on 0 (symmetric vmin/vmax).
    - magnitudes (``val`` "abs", or derived "E"/"intensity"/"H") -> sequential
      ``magma`` from 0.
    - ``val`` "phase" -> cyclic ``twilight`` over [-pi, pi].

    ``data`` is the real-valued 2D numpy array already extracted for display.
    ``cmap=`` overrides the colormap only (the normalization still follows the
    kind). Returns ``(cmap_name, Normalize)``."""
    import math

    import numpy as np
    from matplotlib.colors import Normalize

    is_magnitude = field in ("E", "intensity", "H") or val == "abs"
    is_phase = val == "phase"

    if is_phase:
        chosen = cmap or _PHASE_CMAP
        return chosen, Normalize(vmin=-math.pi, vmax=math.pi)

    finite = data[np.isfinite(data)] if data.size else data
    if finite.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanmax(np.abs(finite))) or 1.0

    if is_magnitude:
        chosen = cmap or _MAGNITUDE_CMAP
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmin = min(vmin, 0.0) if vmin < 0 else 0.0
        return chosen, Normalize(vmin=vmin, vmax=vmax)

    # Signed real/imag: symmetric diverging map centered on zero.
    chosen = cmap or _SIGNED_CMAP
    return chosen, Normalize(vmin=-vmax, vmax=vmax)


def field_colorbar_label(field: str, val: str, attrs: dict) -> str:
    """Colorbar label from the field/``val`` and the DataArray's attrs (units /
    normalization), design §7."""
    if field in ("E", "H"):
        base = f"|{field}|"
    elif field == "intensity":
        base = "|E|^2"
    else:
        base = field
    label = base if field in ("E", "H", "intensity") else f"{base} ({val})"
    norm = attrs.get("normalization")
    if norm:
        # Keep the legend compact: a short tag, not the full sentence.
        label += " [normalized]"
    return label


def field_outline_style(**overrides) -> dict:
    """Matplotlib style for a structure outline drawn OVER a field heatmap:
    a white edge on a soft dark halo, so the boundary reads over both ends of
    the field colormap. Shared by the overlay and its legend swatch so the two
    cannot drift apart."""
    from matplotlib import patheffects

    style = dict(
        fill=False,
        edgecolor=STRUCTURE_EDGE_FIELD,
        linewidth=STRUCTURE_EDGE_FIELD_LW,
        zorder=3,
        path_effects=[patheffects.withStroke(
            linewidth=STRUCTURE_EDGE_FIELD_LW + 1.4,
            foreground=STRUCTURE_EDGE_FIELD_HALO)],
    )
    style.update(overrides)
    return style


def add_legend(ax, *, source: bool, monitor: bool, pml: bool, structure: bool,
               over_field: bool = False, symmetry: bool = False,
               loc: str = "best", source_kinds=None, monitor_kinds=None,
               mode_monitor: bool = False, flux_monitor: bool = False):
    """A compact legend identifying whichever of source / monitor / PML /
    structure / mirror plane are present (design §6). No-op when nothing is
    present. ``over_field`` draws the structure swatch the way the field view
    outlines a structure — white on a dark halo — so the key matches the
    picture.

    ``source_kinds`` / ``monitor_kinds`` are the glyph kinds
    :func:`draw_overlays` reports, and the swatch is of the same kind: a
    launch plane is a line with its arrowhead, a lone dipole a dot, a monitor
    plane a dashed line, a time probe a hollow square. A key that shows a dot
    beside "source" while the picture shows a line is a key to some other
    picture."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    handler_map = {}
    if structure:
        if over_field:
            handles.append(Line2D(
                [0], [0], color=STRUCTURE_EDGE_FIELD, label="material",
                linewidth=STRUCTURE_EDGE_FIELD_LW,
                path_effects=field_outline_style()["path_effects"]))
        else:
            handles.append(Patch(facecolor="#7fa8c9", edgecolor=STRUCTURE_EDGE,
                                 label="structure"))
    if source:
        kinds = set(source_kinds) if source_kinds else {"line"}
        planar = bool(kinds & {"line", "rect"})
        if planar:
            if "arrow" in kinds:
                handle, handler = _arrow_swatch(SOURCE_COLOR, "source")
                handler_map[handle] = handler
                handles.append(handle)
            else:
                handles.append(Line2D([0], [0], color=SOURCE_COLOR,
                                      linestyle="-", label="source"))
        if "point" in kinds:
            handles.append(Line2D(
                [0], [0], marker="o", color=SOURCE_COLOR, markersize=7,
                markeredgecolor="white", linestyle="none",
                label="point source" if planar else "source"))
    # A port's mode monitor (green dash-dot) and the field monitors (amber
    # dashed) have their own keys; with a flux plane among the amber ones the
    # key reads "monitor".
    if mode_monitor:
        handles.append(Line2D([0], [0], color=MODE_MONITOR_COLOR,
                              linestyle=MODE_MONITOR_STYLE, linewidth=1.6,
                              label="mode monitor"))
    if monitor:
        kinds = set(monitor_kinds) if monitor_kinds else {"line"}
        planar = bool(kinds & {"line", "rect"})
        if planar:
            handles.append(Line2D([0], [0], color=MONITOR_COLOR, linestyle="--",
                                  label="monitor" if flux_monitor else "field monitor"))
        if "point" in kinds:
            handles.append(Line2D(
                [0], [0], marker="s", color=MONITOR_COLOR, markersize=6,
                fillstyle="none", linestyle="none",
                label="probe" if planar else "monitor"))
    if pml:
        handles.append(Patch(facecolor=PML_COLOR, alpha=0.25, label="PML"))
    if symmetry:
        handles.append(Line2D([0], [0], color=SYMMETRY_COLOR,
                              linestyle=(0, (6, 4)), label="mirror plane"))
    if handles:
        # The caller picks the corner from the picture (legend_corner); a
        # fixed corner sat on the crossing's vertical arm in every render.
        if loc == LEGEND_OUTSIDE:
            # A long thin plane has no corner to spare: the key goes under the
            # picture in one row, where it covers nothing.
            # The pad is in font sizes, not axes fractions: a band a fraction
            # of an inch tall would otherwise put the key back on its own
            # x axis label.
            ax.legend(handles=handles, loc="upper center", ncol=len(handles),
                      bbox_to_anchor=(0.5, 0.0), bbox_transform=ax.transAxes,
                      fontsize="small", framealpha=0.9,
                      borderaxespad=LEGEND_BELOW_PAD,
                      handler_map=handler_map or None)
        else:
            ax.legend(handles=handles, loc=loc, fontsize="small",
                      framealpha=0.9, handler_map=handler_map or None)


def _arrow_swatch(color: str, label: str):
    """A legend handle drawn as a line with an arrowhead, the way a launch
    plane is drawn, plus the handler that renders it in the key."""
    from matplotlib.legend_handler import HandlerPatch
    from matplotlib.patches import FancyArrowPatch

    handle = FancyArrowPatch((0.0, 0.0), (1.0, 0.0), color=color, label=label,
                             linewidth=1.5)

    def make(legend, orig_handle, xdescent, ydescent, width, height,
             fontsize):
        # matplotlib calls this with keyword arguments of exactly these names.
        y = ydescent + height / 2.0
        return FancyArrowPatch((xdescent, y), (xdescent + width, y),
                               arrowstyle="-|>", mutation_scale=9,
                               color=color, linewidth=1.5, shrinkA=0,
                               shrinkB=0)

    return handle, HandlerPatch(patch_func=make)


# Text beside a monitor or source glyph, and its white halo over dark fill.
GLYPH_LABEL_SIZE = 7.5
# The settings subtitle under a view's title, findable by id.
SUBTITLE_GID = "photonhub-subtitle"
# A plane up to this many times longer than it is wide is drawn in true
# proportion. Past it the short axis is stretched, by at most _MAX_STRETCH, so
# a long device is not a hairline and a 1 um pillar never reads as a bar five
# times its height. A plane past both limits is drawn at the capped stretch in
# a flatter box, and the subtitle says how much it is stretched.
_MAX_TRUE_RATIO = 3.0
_MAX_STRETCH = 3.0
# The shortest colour bar, in inches, that keeps its ticks and label legible.
_COLORBAR_MIN_IN = 1.8
# A plane this much taller than it is wide is drawn with its long axis
# horizontal instead (viz._geometry.displayed_transposed). Well clear of 1 so
# a nearly square plane keeps the ascending axis order readers expect, and
# below the ratio at which a device stops being recognisable on a page: a
# grating coupler is 32 um of propagation across 6 um of stack.
_TRANSPOSE_RATIO = 2.0
# The figure a view sizes for itself when it was not handed an Axes.
_FIGURE_WIDTH_IN = 7.2
_FIGURE_HEIGHT_RANGE_IN = (3.3, 8.0)
_FIGURE_CHROME_IN = 1.35          # axis labels, title and the colorbar's share
# A plane wider than this has no corner a key can sit in without covering the
# device, so the key goes below the picture instead.
_LEGEND_OUTSIDE_RATIO = 3.0
LEGEND_OUTSIDE = "outside"
# How far below the axes the outside key sits, in font sizes (absolute), so a
# band a fraction of an inch tall still clears its own x axis label.
LEGEND_BELOW_PAD = 4.6
# The index axis on the bar's left needs room its default pad does not give.
EPS_COLORBAR_PAD = 0.12


def sharp_inline_figures(enable: Optional[bool] = None) -> bool:
    """Show notebook figures at twice the pixel density ("retina" PNG).

    Jupyter's inline backend renders a figure as a PNG at the figure's own
    100 dpi, which a high-density screen, and the HTML page made from the
    notebook, blur. Retina output renders the same figure at 200 dpi and
    shows it at the same size, so every built-in plot is sharp at no change
    in layout. Called once when ``photonhub.viz`` is imported; it does
    nothing outside an IPython kernel, and ``PHOTONHUB_SHARP_FIGURES=0`` (or
    ``enable=False``) leaves the notebook's own setting alone. Returns whether
    the retina format is now selected."""
    import os
    import sys

    if enable is None:
        enable = os.environ.get("PHOTONHUB_SHARP_FIGURES", "1") != "0"
    if not enable or "IPython" not in sys.modules:
        return False
    try:
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import set_matplotlib_formats
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None or not hasattr(shell, "kernel"):
        return False
    set_matplotlib_formats("retina")
    return True


def cut_label(axis: str, value: float) -> str:
    """``z = 1.84 µm``: the cut plane, for a title."""
    return f"{axis} = {float(value):.3g} µm"


def mesh_summary(sim) -> str:
    """The realized cell size in nm: ``dl 50 nm`` on a uniform grid, ``dl 31
    to 76 nm`` on a graded one. A figure that states its mesh can be read
    without the notebook that made it."""
    lo, hi = [], []
    for i in range(3):
        q = sim._axis_coords_um(i)
        if q is None:
            lo.append(float(sim.grid.dl_um))
            hi.append(float(sim.grid.dl_um))
        else:
            dq = graded_primary_spacings(q)
            lo.append(float(min(dq)))
            hi.append(float(max(dq)))
    lo_nm, hi_nm = min(lo) * 1e3, max(hi) * 1e3
    if hi_nm - lo_nm < 0.5:
        return f"dl {lo_nm:.0f} nm"
    return f"dl {lo_nm:.0f} to {hi_nm:.0f} nm"


def view_stretch(span_h: float, span_v: float) -> float:
    """How many times the vertical axis is stretched against the horizontal
    one: 1 (true proportion) for a plane up to ``_MAX_TRUE_RATIO`` times
    longer than it is tall, then just enough to keep the plot box at that
    ratio, never more than ``_MAX_STRETCH``."""
    if span_h <= 0 or span_v <= 0:
        return 1.0
    ratio = span_h / span_v
    if ratio <= _MAX_TRUE_RATIO:
        return 1.0
    return min(ratio / _MAX_TRUE_RATIO, _MAX_STRETCH)


def stretch_note(v_axis: str, stretch: float) -> Optional[str]:
    """``z stretched 3×`` for the subtitle of a view drawn out of proportion,
    ``None`` for one in proportion."""
    if stretch <= 1.0 + 1e-9:
        return None
    return f"{v_axis} stretched {stretch:.2g}×"


def join_notes(*notes) -> Optional[str]:
    """The subtitle's parts that are set, comma-separated."""
    parts = [n for n in notes if n]
    return ", ".join(parts) if parts else None


def set_view_aspect(ax, span_h: float, span_v: float) -> float:
    """Draw the plane in true proportion or, when it is long and thin, with
    its short axis stretched by :func:`view_stretch` (a fixed data aspect, so
    the stretch is the same wherever the figure lands). Returns the stretch,
    for the subtitle (:func:`stretch_note`)."""
    stretch = view_stretch(span_h, span_v)
    ax.set_aspect(stretch)
    return stretch


def prefer_long_axis_horizontal(span_h: float, span_v: float) -> bool:
    """Whether a plane this shape reads better with its axes swapped.

    A figure is wider than it is tall, and so is the page it lands on, so a
    plane whose vertical axis is much the longer one is drawn as a sliver: a
    grating coupler cut across its layer stack is 32 µm of propagation over
    6 µm of stack, and at true proportions its 24 periods disappear. Turning
    it a quarter turn costs nothing (the axes are labelled either way) and
    gives back the side view the device is always drawn in."""
    return span_h > 0 and span_v > 0 and span_v / span_h > _TRANSPOSE_RATIO


def size_figure_for_plane(ax, span_h: float, span_v: float) -> None:
    """Shape a figure we created ourselves to the plane it has to show.

    Matplotlib's default figure is close to square, so at equal aspect a long
    thin plane is drawn as a thin band across the middle with most of the
    figure left blank, small enough that the device is hard to read. A plane
    past the stretch ratio is not drawn in proportion at all
    (:func:`set_view_aspect`), so its true ratio would only flatten the plot
    box and shrink every feature in it. The plot box takes the plane's shape
    after :func:`view_stretch`. Callers use this only on a figure they made
    themselves, so an Axes handed in by a caller keeps whatever figure it came
    on."""
    fig = ax.figure
    if span_h <= 0 or span_v <= 0:
        return
    lo, hi = _FIGURE_HEIGHT_RANGE_IN
    plot_width = _FIGURE_WIDTH_IN - _FIGURE_CHROME_IN
    ratio = span_v * view_stretch(span_h, span_v) / span_h
    height = plot_width * ratio + _FIGURE_CHROME_IN
    fig.set_size_inches(_FIGURE_WIDTH_IN, min(max(height, lo), hi))
    fit_colorbars_to_plot(ax)


def fit_colorbars_to_plot(ax) -> None:
    """Give the plot's colour bars the height of its box. A fixed aspect
    shrinks the box inside the space matplotlib set aside for it, and the bar
    keeps the full height, so on a long flat plane it towers over the
    picture. A bar never gets shorter than ``_COLORBAR_MIN_IN``, so its ticks
    and label stay legible; a shorter box gets a bar centred on it. Placed
    once, on a figure the view made for itself."""
    ax.apply_aspect()
    box = ax.get_position()
    height = max(box.height, _COLORBAR_MIN_IN / ax.figure.get_size_inches()[1])
    y0 = box.y0 + 0.5 * (box.height - height)
    for artist in ax.collections + ax.images:
        bar = getattr(artist, "colorbar", None)
        if bar is None:
            continue
        pos = bar.ax.get_position()
        bar.ax.set_position([pos.x0, y0, pos.width, height])


def set_titles(ax, title: str, note: Optional[str] = None) -> None:
    """The figure's own caption: ``title`` on the top line and ``note`` (the
    mesh, the sample kind) as a small grey subtitle under it, both
    left-aligned. A subtitle rather than a second title on the same line, so
    a long quantity never runs into its settings and a reader can skip
    them."""
    if not note:
        ax.set_title(title, loc="left")
        return
    ax.set_title(title, loc="left", pad=17)
    ax.annotate(note, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 3),
                textcoords="offset points", ha="left", va="bottom",
                fontsize="small", color="0.4", annotation_clip=False,
                gid=SUBTITLE_GID)


def legend_goes_outside(span_h: float, span_v: float) -> bool:
    """Whether a plane this shape leaves no corner for its key. A key is a
    fixed size in points, so on a band far wider than it is tall every corner
    sits on the device however empty the mesh samples say that corner is."""
    return span_h > 0 and span_v > 0 and span_h / span_v > _LEGEND_OUTSIDE_RATIO


def legend_corner(occupied, span_h: float = 0.0, span_v: float = 0.0) -> str:
    """The legend location whose corner covers the least structure, from a
    boolean occupancy mesh of the view (row 0 at the bottom, as the ε
    sample is laid out). matplotlib's ``loc="best"`` scores bounding boxes,
    so a diagonal taper or a heatmap defeats it; the sampled mesh is the picture
    itself.

    With the plane's spans given, a plane far wider than it is tall returns
    :data:`LEGEND_OUTSIDE` instead: a key is a fixed size in points, so on a
    band a few hundred points high no corner holds it, whatever the mesh samples
    say."""
    import numpy as np

    if legend_goes_outside(span_h, span_v):
        return LEGEND_OUTSIDE
    occ = np.asarray(occupied, dtype=bool)
    if occ.size == 0:
        return "upper right"
    n_v, n_h = occ.shape
    dv, dh = max(1, int(round(0.32 * n_v))), max(1, int(round(0.32 * n_h)))
    corners = {
        "upper right": occ[n_v - dv:, n_h - dh:],
        "upper left": occ[n_v - dv:, :dh],
        "lower right": occ[:dv, n_h - dh:],
        "lower left": occ[:dv, :dh],
    }
    # Ties go to the conventional corner, in this order.
    return min(corners, key=lambda k: (float(corners[k].mean()),
                                       list(corners).index(k)))


def material_marks(permittivities, vmin: float, vmax: float):
    """The distinct permittivities of a scene, for marking on the colorbar.
    Values closer than 3 percent of the range share one mark, and more than
    six materials is a sweep, not a stack of interfaces, so nothing is
    marked."""
    vals = sorted({round(float(v), 9) for v in permittivities})
    if len(vals) < 2 or len(vals) > 6:
        return []
    gap = 0.03 * max(vmax - vmin, 1e-12)
    merged = [[vals[0]]]
    for v in vals[1:]:
        if v - merged[-1][-1] < gap:
            merged[-1].append(v)
        else:
            merged.append([v])
    return [sum(g) / len(g) for g in merged]


def ticks_with_ends(lo: float, hi: float, nbins: int = 6):
    """``(ticks, labels)`` for an axis that must always label its ends:
    matplotlib's round interior ticks, minus any within 6 percent of an end
    (it would sit on the end label), plus the exact ``lo`` and ``hi`` to
    three significant figures."""
    from matplotlib.ticker import MaxNLocator

    span = hi - lo
    if not span > 0.0:
        return [lo], [f"{lo:.3g}"]
    margin = 0.06 * span
    inner = [float(t) for t in MaxNLocator(nbins=nbins).tick_values(lo, hi)
             if lo + margin < t < hi - margin]
    ticks = [lo] + inner + [hi]
    labels = [f"{lo:.3g}"] + [f"{t:g}" for t in inner] + [f"{hi:.3g}"]
    return ticks, labels


def finish_eps_colorbar(cbar, permittivities, vmin: float, vmax: float,
                        label: str) -> None:
    """The permittivity colorbar the way a photonics reader wants to read
    it: ε ticks and label on one side, the refractive index ``n = sqrt(ε)``
    on the other, and a short mark across the bar at each material.

    The index axis is a secondary axis on the bar's own axis with the
    square-root map, so its ticks land at round values of n rather than at
    the ε ticks re-labelled. It is skipped when the range reaches below zero
    (a metal), where the map has no real value. The material marks are lines
    across the bar rather than markers beside it, so they cannot collide
    with either set of tick labels; white on a dark halo reads at both ends
    of the map."""
    import numpy as np
    from matplotlib import patheffects

    cbar.set_label(label)
    # Both scales always carry their exact ends: the reader gets the true
    # minimum and maximum, not just the nearest round tick.
    ticks, labels = ticks_with_ends(vmin, vmax)
    cbar.set_ticks(ticks, labels=labels)
    if vmin > 0.0:
        index_axis = cbar.ax.secondary_yaxis(
            "left", functions=(np.sqrt, np.square))
        index_axis.set_ylabel("refractive index n")
        index_axis.tick_params(labelsize="small")
        n_ticks, n_labels = ticks_with_ends(float(np.sqrt(vmin)),
                                            float(np.sqrt(vmax)))
        index_axis.set_yticks(n_ticks, labels=n_labels)
    for v in material_marks(permittivities, vmin, vmax):
        cbar.ax.axhline(v, color="white", linewidth=1.2, zorder=5,
                        path_effects=[patheffects.withStroke(
                            linewidth=2.6, foreground="0.25")])


def glyph_label(ax, text: str, xy, *, color: str, offset=(3, 3), ha="left",
                va="bottom", xycoords="data") -> None:
    """A small name beside a glyph, haloed so it reads over the structure
    fill. Three amber dashed lines with no names cannot be told apart."""
    from matplotlib import patheffects

    ax.annotate(text, xy=xy, xycoords=xycoords, xytext=offset,
                textcoords="offset points", ha=ha, va=va,
                fontsize=GLYPH_LABEL_SIZE, color=color, zorder=7,
                path_effects=[patheffects.withStroke(linewidth=2.2,
                                                     foreground="white")])


def add_structure_patch(ax, kind, params, *, style, clip=None) -> list:
    """Add one structure patch for a §5 cut-plane spec
    (``rect``/``circle``/``polygon``/``annulus``) with the given matplotlib
    ``style`` dict, and return the patches added. Shared by the filled scene
    view (:func:`scene.plot`) and the 3D/CLI builders so a new geometry kind is
    added in exactly one place.

    ``clip`` is applied AFTER the patch is attached: ``Axes.add_patch`` resets
    an artist's clip to the axes box, so a clip passed at construction is
    silently dropped."""
    from matplotlib.patches import Annulus, Circle, Polygon, Rectangle

    added = []
    if kind == "rect":
        x0, y0, w, h = params
        added.append(ax.add_patch(Rectangle((x0, y0), w, h, **style)))
    elif kind == "rects":
        for x0, y0, w, h in params:
            added.append(ax.add_patch(Rectangle((x0, y0), w, h, **style)))
    elif kind == "circle":
        cx, cy, r = params
        added.append(ax.add_patch(Circle((cx, cy), r, **style)))
    elif kind == "polygon":
        added.append(ax.add_patch(Polygon(params, closed=True, **style)))
    elif kind == "annulus":
        cx, cy, r_outer, r_inner = params
        added.append(ax.add_patch(
            Annulus((cx, cy), r_outer, width=r_outer - r_inner, **style)))
    if clip is not None:
        for patch in added:
            patch.set_clip_path(clip)
    return added


# --------------------------------------------------------------------------- #
# PML geometry (shared by the 2D views and the 3D builder).
# --------------------------------------------------------------------------- #

def _axis_spacings_um(sim, axis_index: int):
    """(low-edge spacing, high-edge spacing, realized length) in microns for an
    axis. The PML band is ``pml_num_layers`` cells thick translated to µm via
    the boundary's LOCAL spacing — the first/last primary spacing for a graded
    axis, ``dl`` for a uniform one (design §6)."""
    dl = sim.grid.dl_um
    q = sim._axis_coords_um(axis_index)
    if q is None:
        n = realized_cells(sim.size_um[axis_index], dl,
                           sim_axis_min_cells(sim, axis_index))
        return dl, dl, n * dl
    dq = graded_primary_spacings(q)
    realized = q[-1] + dq[-1]
    return dq[0], dq[-1], realized


def pml_bands(sim, axis: str):
    """The PML shaded bands for a 2D cut on ``axis``: a list of
    ``(in_plane_axis, lo_um, hi_um)`` spans, one per in-plane axis whose
    boundary kind is 'pml', covering ``pml_num_layers`` cells at each face.
    Bands on the cut axis itself are not drawable in a 2D slice and are
    omitted.

    A §20 symmetry axis carries NO low-face band: the absorber there is
    one-sided (§20.3), the min face being a mirror rather than an exit. The
    same rule governs ``Simulation._pml_bounds``, so the picture and the
    scene checks agree on where the absorber is."""
    bands = []
    layers = sim.pml_num_layers
    boundaries = sim.boundaries
    for letter in geom.in_plane_axes(axis):
        if getattr(boundaries, letter) != "pml":
            continue
        i = _AXES.index(letter)
        lo_dl, hi_dl, realized = _axis_spacings_um(sim, i)
        if sim.symmetry[i] == 0:
            bands.append((letter, 0.0, layers * lo_dl))              # low face
        bands.append((letter, realized - layers * hi_dl, realized))  # high face
    return bands


def unfold_in_plane(sim, axis: str, unfold: bool = True):
    """``(mirror_h, mirror_v)`` — which of the cut's in-plane axes carry a §20
    symmetry plane that a view should unfold.

    A symmetry plane means the run solved HALF the device and the other half
    was never stepped. Left folded, the picture is half a device with the
    interesting boundary at the frame edge, which is not what anyone drawing
    the scene wants to see. A plane on the CUT axis itself is not in the
    picture and mirrors the same plane onto itself, so it is ignored."""
    if not unfold:
        return (False, False)
    h_ax, v_ax = geom.in_plane_axes(axis)
    return (sim.symmetry[_AXES.index(h_ax)] != 0,
            sim.symmetry[_AXES.index(v_ax)] != 0)


def mirror_copies(mirror_h: bool, mirror_v: bool):
    """The ``(sign_h, sign_v)`` copies a view draws: the scene itself, then one
    reflection per unfolded axis, and the diagonal when both are unfolded (a
    quarter domain unfolds into four quadrants)."""
    copies = [(1, 1)]
    if mirror_h:
        copies.append((-1, 1))
    if mirror_v:
        copies.append((1, -1))
    if mirror_h and mirror_v:
        copies.append((-1, -1))
    return copies


def draw_symmetry_planes(ax, sim, axis: str, mirror_h: bool,
                         mirror_v: bool) -> bool:
    """Mark each unfolded §20 mirror at coordinate 0 with a thin line. Half
    the picture was mirrored, not simulated, and the reader is entitled to see
    where that starts. Returns whether anything was drawn."""
    drew = False
    for is_h in (True, False):
        if not (mirror_h if is_h else mirror_v):
            continue
        (ax.axvline if is_h else ax.axhline)(
            0.0, color=SYMMETRY_COLOR, linestyle=(0, (6, 4)), linewidth=1.0,
            zorder=6)
        drew = True
    return drew


def translate_artists(ax, before, dh: float, dv: float) -> None:
    """Move every artist added to ``ax`` since ``before`` (a set of its
    children) by ``(dh, dv)`` in data units. The views sample, reflect and
    clip the scene in the wire's corner frame, where the mirror planes sit at
    0, and show it in the user's frame (design spec §4.4): a data-space
    artist gets a translated transform, a blended transform keeps its
    axes-fraction side, an annotation moves its anchors, and a clip path moves
    with its patch."""
    if not (dh or dv):
        return
    from matplotlib.patches import Patch
    from matplotlib.text import Annotation
    from matplotlib.transforms import (Affine2D, BlendedGenericTransform, Transform,
                                       blended_transform_factory)

    data = ax.transData

    def current(art):
        # a patch's get_transform() folds its own shape transform in front of
        # the data transform; the data transform is the one to move
        return art.get_data_transform() if isinstance(art, Patch) else art.get_transform()

    def moved(t):
        if t is data:
            return Affine2D().translate(dh, dv) + data
        if isinstance(t, BlendedGenericTransform):
            x = (Affine2D().translate(dh, 0.0) + data) if t._x is data else t._x
            y = (Affine2D().translate(0.0, dv) + data) if t._y is data else t._y
            if x is not t._x or y is not t._y:
                return blended_transform_factory(x, y)
        return None

    for art in ax.get_children():
        if art in before:
            continue
        if isinstance(art, Annotation):
            if art.xycoords == "data" or isinstance(art.xycoords, Transform):
                art.xy = (art.xy[0] + dh, art.xy[1] + dv)
            if art.anncoords == "data" or isinstance(art.anncoords, Transform):
                art.xyann = (art.xyann[0] + dh, art.xyann[1] + dv)
            continue
        t = moved(current(art))
        if t is not None:
            art.set_transform(t)
        clip = getattr(art.get_clip_path(), "_patch", None)   # a TransformedPatchPath's patch
        if clip is not None:
            tc = moved(current(clip))
            if tc is not None:
                clip.set_transform(tc)


def has_pml(sim) -> bool:
    """True iff any axis boundary is a PML (design §6 legend gate)."""
    b = sim.boundaries
    return any(getattr(b, a) == "pml" for a in "xyz")


# --------------------------------------------------------------------------- #
# Overlay drawing — sources, monitors, PML — on a 2D Axes (design §6).
# --------------------------------------------------------------------------- #

def draw_grid(ax, sim, axis: str, unfold: bool = True) -> None:
    """Overlay the realized primary-grid cell edges on a 2D cut so mesh
    resolution can be eyeballed against the geometry (the ``grid=True`` flag on
    ``plot`` / ``plot_index``). Uses the SAME node coordinates the solver meshes —
    a uniform ``n*dl`` ladder or the graded cell edges — so the spacing shown is
    exactly what will run. Lines span the realized domain."""
    from .eps import axis_nodes_um  # lazy: eps imports _style (avoid a cycle)

    h_letter, v_letter = geom.in_plane_axes(axis)
    h_i = _AXES.index(h_letter)
    v_i = _AXES.index(v_letter)
    h_nodes = axis_nodes_um(sim, h_i)
    v_nodes = axis_nodes_um(sim, v_i)
    realized = sim._realized_um()
    mirror_h, mirror_v = unfold_in_plane(sim, axis, unfold)
    # An unfolded axis carries the mirrored ladder too, and the lines spanning
    # the other axis reach across the whole unfolded extent.
    h_lines = list(h_nodes) + ([-n for n in h_nodes] if mirror_h else [])
    v_lines = list(v_nodes) + ([-n for n in v_nodes] if mirror_v else [])
    v_lo = -realized[v_i] if mirror_v else 0.0
    h_lo = -realized[h_i] if mirror_h else 0.0
    ax.vlines(h_lines, v_lo, realized[v_i], colors=GRID_COLOR,
              linewidth=GRID_LW, alpha=GRID_ALPHA, zorder=GRID_Z)
    ax.hlines(v_lines, h_lo, realized[h_i], colors=GRID_COLOR,
              linewidth=GRID_LW, alpha=GRID_ALPHA, zorder=GRID_Z)


def axis_tolerance_um(sim, axis_index: int) -> float:
    """Coarsest primary spacing on an axis (µm) — the distance below which two
    coordinates anywhere on that axis may name the same grid plane. Uniform:
    ``dl_um``. Graded: the LARGEST realized spacing, since ``dl_um`` on a
    :class:`GradedMesh` is the BASE spacing and a coarse region is wider than
    it."""
    q = sim._axis_coords_um(axis_index)
    if q is None:
        return float(sim.grid.dl_um)
    return float(max(graded_primary_spacings(q)))


def local_spacing_um(sim, axis_index: int, value: float) -> float:
    """Primary spacing (µm) of the cell that CONTAINS ``value`` on an axis.
    Uniform: ``dl_um``. Graded: the local cell's own width, not the axis-wide
    coarsest — "within half a cell of the cut plane" is a statement about the
    cut's own cell, and a graded axis is an order of magnitude finer through a
    waveguide core than out in the cladding."""
    q = sim._axis_coords_um(axis_index)
    if q is None:
        return float(sim.grid.dl_um)
    dq = graded_primary_spacings(q)
    i = bisect.bisect_right([float(v) for v in q], float(value)) - 1
    return float(dq[min(max(i, 0), len(dq) - 1)])


def _plane_cluster_ids(values, tol: float):
    """Cluster 1-D coordinates into runs separated by more than ``tol``, and
    return one cluster id per input value — or ``None`` when any cluster is
    itself WIDER than ``tol``.

    ``None`` is the useful half: it says the feature SPREADS along this axis
    instead of sitting on a few planes, which is how a sheet's transverse axes
    are told from its normal."""
    if not values:
        return None
    order = sorted(range(len(values)), key=lambda i: values[i])
    ids = [0] * len(values)
    cid = 0
    start = prev = values[order[0]]
    for k, i in enumerate(order):
        v = values[i]
        if k and v - prev > tol:
            cid += 1
            start = v
        if v - start > tol:
            return None      # this axis is a spread, not a stack of planes
        ids[i] = cid
        prev = v
    return ids


# A group this size or larger is a stamped sheet, not hand-placed dipoles.
_SHEET_MIN_DIPOLES = 4


def dipole_features(sim):
    """Resolve the scene's ``PointDipole`` sources into the features a reader
    means to see: the SHEETS they were stamped on, plus any genuinely isolated
    probe dipole.

    The default mode launch (``mode_launch``) is an equivalence-current Huygens
    sheet — hundreds to thousands of per-cell dipoles tiling one plane. Drawing
    a marker per dipole paints a blob of overlapping dots where the picture
    should carry a single line, so the dipoles are grouped back into the plane
    they came from and that plane is drawn once.

    Grouping keys on every axis whose coordinates collapse onto a few planes
    (:func:`_plane_cluster_ids`) and ignores the axes the dipoles spread along.
    A sheet's normal qualifies (one plane per sheet, at most the E/H half-cell
    pair); its transverse axes do not. Two sheets stay two groups because they
    land in different clusters on the axis that separates them.

    Returns a list of ``("point", center3)`` and ``("sheet", (center3, size3,
    normal_axis, direction))`` entries; a sheet's ``size3`` is 0 on every axis
    thinner than one cell, so a plane is a plane and a one-cell-wide sheet is
    a line. ``normal_axis``/``direction`` ("+"/"-") are the launch axis and
    sense read off the sheet itself (see :func:`sheet_direction`), or ``None``
    when the sheet does not carry the electric/magnetic pair that encodes
    them."""
    dipoles = [s for s in sim.sources
               if getattr(s, "type", None) == "point_dipole"]
    if not dipoles:
        return []
    pts = [tuple(float(c) for c in d.center_um) for d in dipoles]
    n = len(pts)
    # A cell and a half, not one cell: a sheet's transverse rows sit exactly
    # one cell apart on a uniform grid, and float rounding puts some of those
    # gaps a hair over dl, which split a sheet into one-cell strips. The
    # planes that must stay apart (the E and H sheets, half a cell; two
    # launches, many cells) are nowhere near the boundary.
    tol = [1.5 * axis_tolerance_um(sim, a) for a in range(3)]

    keys = [[] for _ in range(n)]
    planar_axes = 0
    for a in range(3):
        ids = _plane_cluster_ids([p[a] for p in pts], tol[a])
        if ids is None:
            continue
        planar_axes += 1
        for i in range(n):
            keys[i].append(ids[i])

    if planar_axes == 0:
        # A cloud with no planar axis: nothing to summarize, draw the points.
        return [("point", pt) for pt in pts]

    buckets: Dict[tuple, list] = {}
    for i in range(n):
        buckets.setdefault(tuple(keys[i]), []).append(i)

    out = []
    for members in buckets.values():
        if len(members) < _SHEET_MIN_DIPOLES:
            out.extend(("point", pts[i]) for i in members)
            continue
        center, size = [], []
        for a in range(3):
            lo = min(pts[i][a] for i in members)
            hi = max(pts[i][a] for i in members)
            center.append(0.5 * (lo + hi))
            size.append(0.0 if hi - lo <= tol[a] else hi - lo)
        # The launch axis is one the sheet is thin along. Its transverse axes
        # are not candidates: on a sheet cut by a mirror plane the electric and
        # magnetic dipoles cover the half-sheet unevenly, and their centroids
        # sit further apart across the sheet than the half cell along it.
        normal, direction = sheet_direction(
            [pts[i] for i in members],
            [dipoles[i].polarization for i in members],
            axes=[a for a in range(3) if size[a] == 0.0])
        out.append(("sheet", (tuple(center), tuple(size), normal, direction)))
    return out


def sheet_direction(points, polarizations, axes=None):
    """``(normal_axis, "+"/"-")`` of an equivalence-current sheet, or
    ``(None, None)``.

    A ``PointDipole`` carries no direction, but the sheet does: the builder
    (``analysis.eq_current_source``) stamps the magnetic ``M = -n x E`` sheet
    on the H nodes half a cell UPSTREAM of the electric ``J = n x H`` sheet
    for a "+" launch and half a cell downstream for "-". So the launch sense
    is the sign of (mean electric-dipole position minus mean magnetic-dipole
    position) along the sheet's normal, and the normal is the axis on which
    the two planes are offset at all. A sheet of one dipole kind only (a
    hand-made current sheet) has no such pair and reports ``None``.

    ``axes`` limits the normal to those axis indices (the ones the sheet is
    thin along); by default any axis qualifies."""
    e = [p for p, pol in zip(points, polarizations) if pol.startswith("E")]
    h = [p for p, pol in zip(points, polarizations) if pol.startswith("H")]
    if not e or not h:
        return None, None
    delta = [sum(p[a] for p in e) / len(e) - sum(p[a] for p in h) / len(h)
             for a in range(3)]
    candidates = list(range(3)) if axes is None else list(axes)
    if not candidates:
        return None, None
    a = max(candidates, key=lambda i: abs(delta[i]))
    if abs(delta[a]) <= 1e-9:
        return None, None
    return _AXES[a], ("+" if delta[a] > 0 else "-")


def direction_arrow(ax, xy, transform, dh: int, dv: int, color: str) -> None:
    """An arrow whose TAIL sits on ``xy`` (in ``transform``) and whose head is
    16 points away along the in-plane direction ``(dh, dv)`` (one of them
    zero): "the light leaves this line that way". Points rather than data
    units, so the arrow is the same size on a 3 µm cavity and a 60 µm
    taper."""
    from matplotlib.transforms import offset_copy

    head_coords = offset_copy(transform, fig=ax.figure, x=16 * dh, y=16 * dv,
                              units="points")
    ax.annotate("", xy=xy, xycoords=head_coords, xytext=xy,
                textcoords=transform, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6,
                                mutation_scale=13, shrinkA=0, shrinkB=0))


def _flux_window_rect(monitor, axis: str):
    """A windowed :class:`PowerMonitor` as a ``(center3, size3)`` box, or
    ``None`` for the default full-plane monitor.

    The window's ``center_um``/``size_um`` are ``(u, v)`` pairs in the plane's
    CYCLIC transverse order ``u = (axis+1) % 3, v = (axis+2) % 3`` — not the
    natural in-plane order the rest of this module uses — so they are mapped
    back to absolute x/y/z here."""
    if getattr(monitor, "center_um", None) is None:
        return None
    a = geom.axis_index(axis)
    center = [0.0, 0.0, 0.0]
    size = [0.0, 0.0, 0.0]
    center[a] = float(monitor.position_um)
    for k, i in enumerate(((a + 1) % 3, (a + 2) % 3)):
        center[i] = float(monitor.center_um[k])
        size[i] = float(monitor.size_um[k])
    return tuple(center), tuple(size)


def draw_overlays(ax, sim, axis: str, value: float,
                  unfold: bool = True) -> Dict[str, bool]:
    """Draw source / monitor / PML overlays for a cut plane onto ``ax``.
    Returns which kinds were actually drawn (for the legend gate). Reuses the
    §5 cut-plane geometry so glyphs match across all views.

    Every feature that is a PLANE in the scene is drawn as a line (or a
    rectangle, on the cut it faces) rather than a marker: a mode-launch sheet,
    a plane wave, a §18 mode source, a monitor plane. A marker is reserved for
    a feature that really is a point — a lone dipole, a time probe.

    ``unfold`` mirrors every glyph across each in-plane §20 symmetry plane, so
    the overlays land on the unfolded device rather than on half of it."""
    h_ax, v_ax = geom.in_plane_axes(axis)
    half_cell = 0.5 * local_spacing_um(sim, geom.axis_index(axis), value)
    copies = mirror_copies(*unfold_in_plane(sim, axis, unfold))

    drew = {"source": False, "monitor": False, "pml": False,
            "source_kinds": set(), "monitor_kinds": set(),
            "mode_monitor": False, "flux_monitor": False}

    # PML bands first so structures/overlays draw on top.
    for letter, lo, hi in pml_bands(sim, axis):
        for sign in _axis_signs(copies, letter == h_ax):
            span = (ax.axvspan if letter == h_ax else ax.axhspan)
            band = (lo, hi) if sign == 1 else (-hi, -lo)
            span(band[0], band[1], color=PML_COLOR, alpha=0.18, zorder=0)
        drew["pml"] = True

    # Sources. Dipoles first, resolved into the planes they were stamped on.
    for kind, payload in dipole_features(sim):
        if kind == "point":
            for sh, sv in copies:
                pt = geom.point_in_plane(_reflect_center(payload, axis, sh, sv),
                                         axis, value, half_cell)
                if pt is not None:
                    ax.plot(pt[0], pt[1], marker="o", color=SOURCE_COLOR,
                            markersize=8, markeredgecolor="white",
                            linestyle="none", zorder=5)
                    drew["source"] = True
                    drew["source_kinds"].add("point")
            continue
        center, size, normal, direction = payload
        # The launch sense reads as an in-plane arrow only when the sheet is
        # seen edge-on, i.e. its normal lies in the cut; face-on, the light
        # leaves the page and there is nothing to point along.
        arrow = None
        if direction and normal in (h_ax, v_ax):
            sign = 1 if direction == "+" else -1
            arrow = (sign, 0) if normal == h_ax else (0, sign)
        kinds = _draw_box_copies(
            ax, center, size, axis, value, half_cell, copies, SOURCE_COLOR,
            "-", arrow=arrow,
            plane_tol=(axis_tolerance_um(sim, _AXES.index(h_ax)),
                       axis_tolerance_um(sim, _AXES.index(v_ax))))
        if kinds:
            drew["source"] = True
            drew["source_kinds"] |= kinds

    for src in sim.sources:
        stype = getattr(src, "type", None)
        if stype in ("plane_wave", "mode_source"):
            # Both are full injection PLANES named by (axis, position), and
            # both know which way they launch.
            line = geom.axis_line_in_plane(src.axis, src.position_um, axis)
            if line is not None:
                direction = getattr(src, "direction", None)
                _draw_axis_line_copies(ax, h_ax, line, SOURCE_COLOR, "-",
                                       copies, direction=direction)
                drew["source"] = True
                drew["source_kinds"].add("line")
                if direction in ("+", "-"):
                    drew["source_kinds"].add("arrow")

    # Monitors. A port's readout plane is its mode monitor: drawn over the
    # window its mode was solved on, named after the port.
    port_planes = _port_planes(sim)
    for m in sim.monitors:
        mtype = getattr(m, "type", None)
        name = getattr(m, "name", None)
        if mtype == "field_dft" and name in port_planes:
            label, box = port_planes[name]
            if box is None:
                center, size = m.center_um, m.size_um
            else:
                normal = _plane_axis(m)
                center = tuple(m.center_um[i] if i == normal else c
                               for i, c in enumerate(box[0]))
                size = box[1]
            if _draw_box_copies(ax, center, size, axis, value, half_cell, copies,
                                MODE_MONITOR_COLOR, MODE_MONITOR_STYLE, name=label):
                drew["mode_monitor"] = True
        elif mtype == "field_dft":
            kinds = _draw_box_copies(ax, m.center_um, m.size_um, axis, value,
                                     half_cell, copies, MONITOR_COLOR, "--",
                                     name=name)
            if kinds:
                drew["monitor"] = True
                drew["monitor_kinds"] |= kinds
        elif mtype == "flux":
            drew["flux_monitor"] = True
            window = _flux_window_rect(m, m.axis)
            if window is not None:
                # A sub-region window covers part of the plane, not all of it.
                kinds = _draw_box_copies(ax, window[0], window[1], axis, value,
                                         half_cell, copies, MONITOR_COLOR,
                                         "--", name=name)
                if kinds:
                    drew["monitor"] = True
                    drew["monitor_kinds"] |= kinds
                continue
            line = geom.axis_line_in_plane(m.axis, m.position_um, axis)
            if line is not None:
                _draw_axis_line_copies(ax, h_ax, line, MONITOR_COLOR, "--",
                                       copies, name=name)
                drew["monitor"] = True
                drew["monitor_kinds"].add("line")
        elif mtype in ("field_time", "field_snapshot"):
            center = getattr(m, "center_um", None)
            if center is None:
                continue
            for sh, sv in copies:
                pt = geom.point_in_plane(_reflect_center(center, axis, sh, sv),
                                         axis, value, half_cell)
                if pt is not None:
                    ax.plot(pt[0], pt[1], marker="s", color=MONITOR_COLOR,
                            markersize=6, fillstyle="none", linestyle="none",
                            zorder=5)
                    if name and (sh, sv) == (1, 1):
                        glyph_label(ax, name, pt, color=MONITOR_COLOR,
                                    offset=(5, 0), ha="left", va="center")
                    drew["monitor"] = True
                    drew["monitor_kinds"].add("point")
    return drew


def _plane_axis(monitor) -> int:
    """The axis a plane monitor is normal to: the one its size is zero along."""
    return min(range(3), key=lambda i: abs(float(monitor.size_um[i])))


def _port_planes(sim):
    """``{readout monitor name: (port name, window box or None)}`` for the
    ports of a declarative simulation. The box is ``(center3, size3)`` in the
    solver's frame: the window the port's mode was solved on, or ``None`` when
    it is not known (a simulation loaded from a file), in which case the whole
    plane is drawn."""
    rec = getattr(sim, "_declarative", None)
    if rec is None:
        return {}
    windows = getattr(rec, "port_windows_um", None) or {}
    out = {}
    for port in getattr(rec, "ports", ()) or ():
        box = None
        win = windows.get(port.name)
        if win is not None:
            a = _AXES.index(port.axis)
            h, v = (i for i in range(3) if i != a)
            size = [0.0, 0.0, 0.0]
            size[h], size[v] = 2.0 * float(win[0]), 2.0 * float(win[1])
            box = (tuple(float(c) for c in port.center_um), tuple(size))
        out[port.monitor_name] = (port.name, box)
    return out


def _axis_signs(copies, is_horizontal: bool):
    """The distinct reflection signs the copies apply to one in-plane axis."""
    seen = []
    for sh, sv in copies:
        sign = sh if is_horizontal else sv
        if sign not in seen:
            seen.append(sign)
    return seen


def _reflect_center(center_um, axis: str, sign_h: int, sign_v: int):
    """A 3-D feature centre with its two IN-PLANE components reflected. The
    cut-axis component is untouched: the mirror is in the picture, not
    through it."""
    if sign_h == 1 and sign_v == 1:
        return center_um
    out = list(center_um)
    h_ax, v_ax = geom.in_plane_axes(axis)
    out[_AXES.index(h_ax)] *= sign_h
    out[_AXES.index(v_ax)] *= sign_v
    return tuple(out)


def _draw_box_copies(ax, center_um, size_um, axis: str, value: float,
                     half_cell: float, copies, color: str, style: str,
                     name: Optional[str] = None, arrow=None,
                     plane_tol=(0.0, 0.0)) -> bool:
    """Draw one overlay box and each of its unfolded mirror images. The name
    and the direction arrow go on the scene's own copy only; its mirror is
    the same feature. Returns the kinds of glyph drawn (``"rect"``,
    ``"line"``, ``"point"``, plus ``"arrow"`` when a direction was shown), so
    the legend can show a swatch of the same kind; empty when the box misses
    the cut."""
    kinds = set()
    for sh, sv in copies:
        rect = _feature_rect(_reflect_center(center_um, axis, sh, sv), size_um,
                             axis, value, half_cell)
        if rect is not None:
            primary = (sh, sv) == (1, 1)
            kinds |= _draw_region(
                ax, rect, color, style, name=name if primary else None,
                arrow=(_unfolded_arrow(rect, arrow, copies, plane_tol)
                       if primary and arrow else None))
    return kinds


def _unfolded_arrow(rect, arrow, copies, plane_tol):
    """The arrow spec, re-anchored when the sheet continues across a mirror.
    A launch sheet clipped at a §20 plane is half a sheet; its mirror is the
    other half, and the arrow belongs at the centre of the whole, which is
    the plane itself, not the centre of the simulated half. The sheet's first
    dipole row sits up to a cell off the plane (Yee stagger), so "touches the
    plane" is judged to within one cell of that axis."""
    x0, y0, w, h = rect
    tol_h, tol_v = plane_tol
    mirrored_h = any(sh == -1 for sh, _ in copies)
    mirrored_v = any(sv == -1 for _, sv in copies)
    anchor = None
    if w == 0.0 and h > 0.0 and mirrored_v and abs(y0) <= tol_v:
        anchor = ("v", 0.0)
    elif h == 0.0 and w > 0.0 and mirrored_h and abs(x0) <= tol_h:
        anchor = ("h", 0.0)
    return (arrow[0], arrow[1], anchor)


def _draw_axis_line_copies(ax, h_ax: str, line: Tuple[str, float], color: str,
                           style: str, copies, name: Optional[str] = None,
                           direction: Optional[str] = None) -> None:
    """Draw a full-span axis line and its unfolded mirror image. The name and
    the direction arrow go on the scene's own copy only."""
    orientation_axis, pos = line
    for sign in _axis_signs(copies, orientation_axis == h_ax):
        primary = sign == 1
        _draw_axis_line(ax, h_ax, (orientation_axis, sign * pos), color, style,
                        name=name if primary else None,
                        direction=direction if primary else None)


def _feature_rect(center_um, size_um, axis: str, value: float,
                  half_cell: float):
    """Cut-plane rectangle of an overlay box. A box that is FLAT on the cut
    axis is matched within half a cell: it carries no thickness to intersect,
    and the §12 quarter-cell auto-snap moves a monitor plane off whatever round
    coordinate the cut was asked for, so exact containment would drop the
    monitor out of the very view meant to place it."""
    a = geom.axis_index(axis)
    tol = half_cell if float(size_um[a]) == 0.0 else 0.0
    return geom.box_rectangle(center_um, size_um, axis, value, tol=tol)


def _draw_axis_line(ax, h_ax: str, line: Tuple[str, float], color: str,
                    style: str, name: Optional[str] = None,
                    direction: Optional[str] = None) -> None:
    """A full-span line for an in-plane axis feature ``(orientation_axis,
    position)``. If the feature's axis is the horizontal axis, the line is
    vertical at that position; otherwise horizontal. ``name`` is written at
    the line's far end; ``direction`` ("+"/"-") puts an arrow at its middle
    pointing along the feature's own axis, the way a plane wave travels."""
    orientation_axis, pos = line
    vertical = orientation_axis == h_ax
    if vertical:
        ax.axvline(pos, color=color, linestyle=style, linewidth=1.5, zorder=4)
        if name:
            glyph_label(ax, name, (pos, 0.985), color=color, offset=(3, 0),
                        ha="left", va="top", xycoords=("data", "axes fraction"))
    else:
        ax.axhline(pos, color=color, linestyle=style, linewidth=1.5, zorder=4)
        if name:
            glyph_label(ax, name, (0.99, pos), color=color, offset=(0, 3),
                        ha="right", va="bottom",
                        xycoords=("axes fraction", "data"))
    if direction in ("+", "-"):
        sign = 1 if direction == "+" else -1
        # Tail on the line at mid-span, head along the axis the feature
        # propagates on. The blended transform keeps the mid-span at the
        # axes' middle whatever limits the caller sets afterwards.
        if vertical:
            direction_arrow(ax, (pos, 0.5), ax.get_xaxis_transform(), sign, 0,
                            color)
        else:
            direction_arrow(ax, (0.5, pos), ax.get_yaxis_transform(), 0, sign,
                            color)


def _arrow_anchor(arrow, along: str, default: float) -> float:
    """The along-line coordinate for an arrow: the re-anchored one when the
    spec carries it for this axis, else the line's own midpoint."""
    if len(arrow) > 2 and arrow[2] is not None and arrow[2][0] == along:
        return arrow[2][1]
    return default


def _draw_region(ax, rect: Tuple[float, float, float, float], color: str,
                 style: str, name: Optional[str] = None, arrow=None) -> set:
    """Outline a cut-plane region: a rectangle when it has area, the LINE it
    collapses to when one in-plane extent is zero, a marker only when both
    are. ``name`` is written beside it; ``arrow`` ``(dh, dv)`` puts a
    direction arrow at a line's midpoint. Returns the glyph kinds drawn."""
    from matplotlib.patches import Rectangle

    x0, y0, w, h = rect
    if w == 0.0 or h == 0.0:
        if w == 0.0 and h == 0.0:
            ax.plot(x0, y0, marker="s", color=color, markersize=6,
                    fillstyle="none", linestyle="none", zorder=4)
            if name:
                glyph_label(ax, name, (x0, y0), color=color, offset=(5, 0),
                            ha="left", va="center")
            return {"point"}
        if w == 0.0:
            ax.plot([x0, x0], [y0, y0 + h], color=color, linestyle=style,
                    linewidth=1.5, zorder=4)
            if name:
                glyph_label(ax, name, (x0, y0 + h), color=color,
                            offset=(3, 0), ha="left", va="top")
            if arrow:
                mid = _arrow_anchor(arrow, "v", y0 + h / 2.0)
                direction_arrow(ax, (x0, mid), ax.transData, arrow[0],
                                arrow[1], color)
        else:
            ax.plot([x0, x0 + w], [y0, y0], color=color, linestyle=style,
                    linewidth=1.5, zorder=4)
            if name:
                glyph_label(ax, name, (x0 + w, y0), color=color,
                            offset=(0, 3), ha="right", va="bottom")
            if arrow:
                mid = _arrow_anchor(arrow, "h", x0 + w / 2.0)
                direction_arrow(ax, (mid, y0), ax.transData, arrow[0],
                                arrow[1], color)
        return {"line", "arrow"} if arrow else {"line"}
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color,
                           linestyle=style, linewidth=1.5, zorder=4))
    if name:
        glyph_label(ax, name, (x0, y0 + h), color=color, offset=(3, -3),
                    ha="left", va="top")
    return {"rect"}

