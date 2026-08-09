"""Shared styling and overlay glyphs: the ε colormap, field-colormap /
normalization selection, overlay glyph colors + drawing, PML bands, and the
compact legend (design §6, §7).

Kept free of plotting-method orchestration so both the 2D views and the 3D
builder map the same permittivity to the same color and draw the same glyphs.
"""

from typing import Dict, Iterable, Optional, Tuple

from ..components.grid import graded_primary_spacings, realized_cells
from . import _geometry as geom

_AXES = "xyz"

# Overlay glyph colors (design §6 — matching the approved mockup's legend).
SOURCE_COLOR = "#ff7f5c"      # coral
MONITOR_COLOR = "#f0a800"     # amber
PML_COLOR = "#7a7a7a"         # neutral grey
STRUCTURE_EDGE = "#222222"    # structure outline over field data

# Primary-grid overlay (the ``grid=True`` mesh sanity-check): thin, light lines
# above the structures/heatmap but below the source/monitor glyphs (zorder 4-5).
GRID_COLOR = "#5a5a5a"
GRID_LW = 0.4
GRID_ALPHA = 0.35
GRID_Z = 2.5

# ε heatmap / structure-fill colormap. Sequential, light->dark with ε.
EPS_CMAP = "viridis"

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


def add_legend(ax, *, source: bool, monitor: bool, pml: bool, structure: bool):
    """A compact legend identifying whichever of source / monitor / PML /
    structure are present (design §6). No-op when nothing is present."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    if structure:
        handles.append(Patch(facecolor="#7fa8c9", edgecolor=STRUCTURE_EDGE,
                             label="structure"))
    if source:
        handles.append(Line2D([0], [0], marker="o", color=SOURCE_COLOR,
                             linestyle="-", label="source"))
    if monitor:
        handles.append(Line2D([0], [0], color=MONITOR_COLOR, linestyle="--",
                             label="monitor"))
    if pml:
        handles.append(Patch(facecolor=PML_COLOR, alpha=0.25, label="PML"))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize="small",
                  framealpha=0.9)


def add_structure_patch(ax, kind, params, *, style) -> None:
    """Add one structure patch for a §5 cut-plane spec
    (``rect``/``circle``/``polygon``/``annulus``) with the given matplotlib
    ``style`` dict. Shared by the filled scene view (:func:`scene.plot`) and the
    field-overlay outlines (:func:`field.plot_field`) so a new geometry kind is
    added in exactly one place."""
    from matplotlib.patches import Annulus, Circle, Polygon, Rectangle

    if kind == "rect":
        x0, y0, w, h = params
        ax.add_patch(Rectangle((x0, y0), w, h, **style))
    elif kind == "circle":
        cx, cy, r = params
        ax.add_patch(Circle((cx, cy), r, **style))
    elif kind == "polygon":
        ax.add_patch(Polygon(params, closed=True, **style))
    elif kind == "annulus":
        cx, cy, r_outer, r_inner = params
        ax.add_patch(Annulus((cx, cy), r_outer, width=r_outer - r_inner,
                             **style))


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
        n = realized_cells(sim.size_um[axis_index], dl)
        return dl, dl, n * dl
    dq = graded_primary_spacings(q)
    realized = q[-1] + dq[-1]
    return dq[0], dq[-1], realized


def pml_bands(sim, axis: str):
    """The PML shaded bands for a 2D cut on ``axis``: a list of
    ``(in_plane_axis, lo_um, hi_um)`` spans, one per in-plane axis whose
    boundary kind is 'pml', covering ``pml_num_layers`` cells at each face.
    Bands on the cut axis itself are not drawable in a 2D slice and are
    omitted."""
    bands = []
    layers = sim.pml_num_layers
    boundaries = sim.boundaries
    for letter in geom.in_plane_axes(axis):
        if getattr(boundaries, letter) != "pml":
            continue
        i = _AXES.index(letter)
        lo_dl, hi_dl, realized = _axis_spacings_um(sim, i)
        bands.append((letter, 0.0, layers * lo_dl))                 # low face
        bands.append((letter, realized - layers * hi_dl, realized))  # high face
    return bands


def has_pml(sim) -> bool:
    """True iff any axis boundary is a PML (design §6 legend gate)."""
    b = sim.boundaries
    return any(getattr(b, a) == "pml" for a in "xyz")


# --------------------------------------------------------------------------- #
# Overlay drawing — sources, monitors, PML — on a 2D Axes (design §6).
# --------------------------------------------------------------------------- #

def draw_grid(ax, sim, axis: str) -> None:
    """Overlay the realized primary-grid cell edges on a 2D cut so mesh
    resolution can be eyeballed against the geometry (the ``grid=True`` flag on
    ``plot`` / ``plot_eps``). Uses the SAME node coordinates the solver meshes —
    a uniform ``n*dl`` ladder or the graded cell edges — so the spacing shown is
    exactly what will run. Lines span the realized domain."""
    from .eps import axis_nodes_um  # lazy: eps imports _style (avoid a cycle)

    h_letter, v_letter = geom.in_plane_axes(axis)
    h_i = _AXES.index(h_letter)
    v_i = _AXES.index(v_letter)
    h_nodes = axis_nodes_um(sim, h_i)
    v_nodes = axis_nodes_um(sim, v_i)
    realized = sim._realized_um()
    ax.vlines(h_nodes, 0.0, realized[v_i], colors=GRID_COLOR, linewidth=GRID_LW,
              alpha=GRID_ALPHA, zorder=GRID_Z)
    ax.hlines(v_nodes, 0.0, realized[h_i], colors=GRID_COLOR, linewidth=GRID_LW,
              alpha=GRID_ALPHA, zorder=GRID_Z)


def draw_overlays(ax, sim, axis: str, value: float) -> Dict[str, bool]:
    """Draw source / monitor / PML overlays for a cut plane onto ``ax``.
    Returns which kinds were actually drawn (for the legend gate). Reuses the
    §5 cut-plane geometry so glyphs match across all views."""
    h_ax, v_ax = geom.in_plane_axes(axis)
    half_cell = 0.5 * sim.grid.dl_um  # point-feature in-plane tolerance

    drew = {"source": False, "monitor": False, "pml": False}

    # PML bands first so structures/overlays draw on top.
    for letter, lo, hi in pml_bands(sim, axis):
        if letter == h_ax:
            ax.axvspan(lo, hi, color=PML_COLOR, alpha=0.18, zorder=0)
        else:
            ax.axhspan(lo, hi, color=PML_COLOR, alpha=0.18, zorder=0)
        drew["pml"] = True

    # Sources.
    for s in sim.sources:
        if getattr(s, "type", None) == "point_dipole":
            pt = geom.point_in_plane(s.center_um, axis, value, half_cell)
            if pt is not None:
                ax.plot(pt[0], pt[1], marker="o", color=SOURCE_COLOR,
                        markersize=8, markeredgecolor="white",
                        linestyle="none", zorder=5)
                drew["source"] = True
        elif getattr(s, "type", None) == "plane_wave":
            line = geom.axis_line_in_plane(s.axis, s.position_um, axis)
            if line is not None:
                _draw_axis_line(ax, h_ax, line, SOURCE_COLOR, "-")
                drew["source"] = True

    # Monitors.
    for m in sim.monitors:
        mtype = getattr(m, "type", None)
        if mtype == "field_dft":
            rect = geom.box_rectangle(m.center_um, m.size_um, axis, value)
            if rect is not None:
                _draw_dashed_rect(ax, rect, MONITOR_COLOR)
                drew["monitor"] = True
        elif mtype == "flux":
            line = geom.axis_line_in_plane(m.axis, m.position_um, axis)
            if line is not None:
                _draw_axis_line(ax, h_ax, line, MONITOR_COLOR, "--")
                drew["monitor"] = True
        elif mtype in ("field_time", "field_snapshot"):
            center = getattr(m, "center_um", None)
            if center is not None:
                pt = geom.point_in_plane(center, axis, value, half_cell)
                if pt is not None:
                    ax.plot(pt[0], pt[1], marker="s", color=MONITOR_COLOR,
                            markersize=6, fillstyle="none", linestyle="none",
                            zorder=5)
                    drew["monitor"] = True
    return drew


def _draw_axis_line(ax, h_ax: str, line: Tuple[str, float], color: str,
                    style: str) -> None:
    """A full-span line for an in-plane axis feature ``(orientation_axis,
    position)``. If the feature's axis is the horizontal axis, the line is
    vertical at that position; otherwise horizontal."""
    orientation_axis, pos = line
    if orientation_axis == h_ax:
        ax.axvline(pos, color=color, linestyle=style, linewidth=1.5, zorder=4)
    else:
        ax.axhline(pos, color=color, linestyle=style, linewidth=1.5, zorder=4)


def _draw_dashed_rect(ax, rect: Tuple[float, float, float, float],
                      color: str) -> None:
    from matplotlib.patches import Rectangle

    x0, y0, w, h = rect
    if w == 0.0 or h == 0.0:
        # Plane/line region: draw it as a span line rather than a zero-area box.
        if w == 0.0 and h == 0.0:
            ax.plot(x0, y0, marker="s", color=color, markersize=6,
                    fillstyle="none", linestyle="none", zorder=4)
        elif w == 0.0:
            ax.plot([x0, x0], [y0, y0 + h], color=color, linestyle="--",
                    linewidth=1.5, zorder=4)
        else:
            ax.plot([x0, x0 + w], [y0, y0], color=color, linestyle="--",
                    linewidth=1.5, zorder=4)
        return
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color,
                           linestyle="--", linewidth=1.5, zorder=4))
