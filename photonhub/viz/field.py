"""``plot_field()``, a field-component heatmap on a 2D slice of a monitor's
DataArray.

Consumes the ``xarray.DataArray`` that ``RunResult[monitor]`` returns
(already in µm coordinates). Supports the raw components Ex..Hz plus derived
``"E"`` (vector magnitude), ``"intensity"`` (|E|²) and ``"H"``; ``freq=`` is
required for a multi-frequency DFT monitor; ``val`` in real/imag/abs/phase
selects what to show for complex data. Colormap/normalization follow §7 and a
colorbar is labeled from the DataArray's attrs. ``structures=True`` overlays
the MATERIAL BOUNDARY on the cut plane, a contour of the same ε sample
``plot_index`` draws, so several bodies of one material read as one silhouette, WHITE on a soft dark halo (:func:`_style.field_outline_style`) so it reads
over both the dark and the bright end of every field colormap.
"""

import warnings
from typing import Optional

import numpy as np

from ..constants import c0
from . import _geometry as geom
from . import _style

_E_COMPONENTS = ("Ex", "Ey", "Ez")
_H_COMPONENTS = ("Hx", "Hy", "Hz")
_SPATIAL = ("x", "y", "z")

# Emitted once if outlines are requested without geometry to draw them from.
_NO_GEOMETRY_NOTED = {"flag": False}


def _isel_nearest(array, dim: str, value: float):
    coords = np.asarray(array.coords[dim].values, dtype=float)
    if coords.size == 0 or not np.all(np.isfinite(coords)):
        raise ValueError(f"coordinate {dim!r} has no finite recorded samples")
    return array.isel({dim: int(np.argmin(np.abs(coords - float(value))))})


def _available_components(da) -> list:
    if "component" in da.coords:
        return [str(c) for c in da.coords["component"].values]
    return []


def _component_array(da, field: str, freq, val: str, time=None):
    """Reduce the DataArray to a real 2D-ready numpy array for ``field``,
    resolving the frequency/time selection, the derived magnitudes, and the
    complex ``val`` projection. Returns ``(values_da, used_val)`` where
    ``values_da`` is a DataArray still carrying its spatial coords."""
    available = _available_components(da)

    # Resolve the sample/frequency/time selection down to a single slice.
    da = _select_sample(da, freq, time)

    if field in ("E", "H", "intensity"):
        comps = _E_COMPONENTS if field in ("E", "intensity") else _H_COMPONENTS
        missing = [c for c in comps if c not in available]
        if missing:
            raise ValueError(
                f"derived field {field!r} needs all of {list(comps)} in monitor "
                f"{da.name!r}; missing {missing} (available: {available}). "
                "Record the full vector to plot a magnitude."
            )
        stack = [da.sel(component=c) for c in comps]
        sq = sum(np.abs(s) ** 2 for s in stack)
        values = sq if field == "intensity" else np.sqrt(sq)
        return values, "abs"

    if field not in available:
        raise KeyError(
            f"field {field!r} not in monitor {da.name!r}; available "
            f"components: {available} (or derived 'E'/'intensity'/'H')"
        )
    comp = da.sel(component=field)

    if np.iscomplexobj(comp.values):
        if val == "real":
            return comp.real, val
        if val == "imag":
            return comp.imag, val
        if val == "abs":
            return np.abs(comp), val
        if val == "phase":
            return _phase(comp), val
        raise ValueError(
            f"val must be one of 'real', 'imag', 'abs', 'phase'; got {val!r}"
        )
    # Real (time-domain) field: val is ignored (design §9).
    return comp, "real"


def _phase(comp):
    out = comp.copy(data=np.angle(comp.values))
    return out


def _select_sample(da, freq, time=None):
    """Collapse the frequency ('f') or time ('t') dim to a single slice.
    Multi-frequency DFT with no ``freq=`` -> ValueError listing the freqs
    (design §9); a single value selects implicitly. ``time=`` (seconds) picks a
    recorded sample on time data, nearest; without it, time data defaults to the
    last frame (the snapshot use)."""
    if "f" in da.dims:
        freqs = [float(v) for v in da.coords["f"].values]
        if freq is None:
            if len(freqs) == 1:
                return da.isel(f=0)
            raise ValueError(
                f"monitor {da.name!r} carries multiple frequencies {freqs} Hz; "
                "pass freq= to choose one"
            )
        return _isel_nearest(da, "f", freq)
    if "t" in da.dims:
        if time is None:
            return da.isel(t=-1)   # default: last recorded sample (snapshot)
        return _isel_nearest(da, "t", time)
    return da


def _reduce_to_plane(values, x, y, z):
    """Reduce a spatial DataArray to a 2D (vertical, horizontal) array on the
    requested plane. If exactly one of x/y/z is given, slice that axis; if the
    monitor is already planar (a spatial dim of length 1), use its own plane.
    Returns ``(h_coord, v_coord, arr2d, axis_letter)``."""
    spatial = [d for d in values.dims if d in _SPATIAL]

    given = [(ax, v) for ax, v in (("x", x), ("y", y), ("z", z)) if v is not None]

    if len(given) == 1:
        axis, val = given[0]
        if axis in values.dims:
            values = _isel_nearest(values, axis, val)
        # else: that axis is already absent (monitor is planar there) -> nothing
        # to slice; fall through.
    elif len(given) == 0:
        # No explicit plane: the monitor must be intrinsically planar.
        thin = [d for d in spatial if values.sizes[d] == 1]
        if not thin:
            raise ValueError(
                "monitor is volumetric; pass exactly one of x=, y=, z= (µm) to "
                "choose the slice plane"
            )
        axis = thin[0]
        values = values.isel({axis: 0})
    else:
        raise ValueError(
            "pass at most one of x=, y=, z= (the slice plane, in microns)"
        )

    remaining = [d for d in values.dims if d in _SPATIAL]
    # Drop any length-1 spatial dims that are not the slice axis.
    for d in list(remaining):
        if values.sizes[d] == 1:
            values = values.isel({d: 0})
    remaining = [d for d in values.dims if d in _SPATIAL]
    if len(remaining) != 2:
        raise ValueError(
            f"after slicing, {len(remaining)} spatial dims remain ({remaining}); "
            "the monitor data is not reducible to a 2D plane"
        )
    # Orient as (vertical, horizontal) by the canonical in-plane order.
    # remaining dims are a subset of x/y/z; pick the natural (h, v) pair order.
    h_letter, v_letter = _orient(remaining)
    arr = values.transpose(v_letter, h_letter)
    return (values.coords[h_letter].values, values.coords[v_letter].values,
            arr, (h_letter, v_letter))


def _orient(remaining):
    """(horizontal, vertical) order for the two surviving spatial dims, from
    the §5 in-plane axis convention, so it follows a
    :func:`~photonhub.viz._geometry.displayed_transposed` block the way every
    other helper does."""
    s = set(remaining)
    missing = [a for a in "xyz" if a not in s]
    if len(missing) != 1:
        # Fallback (shouldn't happen): keep given order.
        return remaining[0], remaining[1]
    return geom.in_plane_axes(missing[0])


def plot_field(data, monitor, field="Ex", x=None, y=None, z=None, *,
               freq=None, time=None, val="real", structures=True,
               simulation=None, ax=None, cmap=None, legend=True, unfold=True,
               scale="max", db_floor=-40.0, **kw):
    """Heatmap of a field component on a 2D slice of ``data[monitor]``.

    ``data`` is a :class:`RunResult`; ``monitor`` is its key. ``freq=``
    picks a frequency on a DFT monitor; ``time=`` (seconds) picks a recorded
    sample on a time/snapshot monitor (default: the last frame). See the module
    docstring and design §3 for the rest.

    ``scale`` sets the colour scale. ``"max"`` (the default) divides by the
    slice's own maximum, so the colorbar runs 0 to 1 (or -1 to 1 for a signed
    part) instead of showing a per-unit-source number like ``1.7e-5`` that no
    reader can place. ``"db"`` shows a magnitude in decibels relative to that
    maximum down to ``db_floor``, which is how a feature a hundred times
    weaker than the guide (a crossing's crosstalk arm) becomes visible at
    all; it needs a magnitude (``field="E"``, ``"H"``, ``"intensity"`` or
    ``val="abs"``). ``"raw"`` keeps the recorded values.

    ``unfold`` (the default) mirrors the recorded half back across each
    in-plane §20 symmetry plane, with the component's own parity about that
    plane, so the picture is the whole device's field. It needs a
    ``simulation`` to know the symmetry, the same object the structure
    outlines need. ``unfold=False`` shows only the half that was stepped.
    Returns the matplotlib ``Axes``."""
    da = data[monitor]  # KeyError (with available list) for an unknown monitor.

    values, used_val = _component_array(da, field, freq, val, time)
    h_coord, v_coord, _, _ = _reduce_to_plane(values, x, y, z)
    # Which way round the recorded plane reads best, from its own extent, and
    # then true of every helper below (design §5). Reducing again inside the
    # block is a relabelled view of the same array, not a second slice.
    transposed = _style.prefer_long_axis_horizontal(
        float(np.ptp(h_coord)), float(np.ptp(v_coord)))
    if transposed:
        with geom.displayed_transposed():
            return _plot_field_on(ax, data, da, monitor, field, x, y, z,
                                  values, used_val, scale, db_floor, simulation,
                                  cmap, legend, unfold, structures, **kw)
    return _plot_field_on(ax, data, da, monitor, field, x, y, z, values,
                          used_val, scale, db_floor, simulation, cmap, legend,
                          unfold, structures, **kw)


def _plot_field_on(ax, data, da, monitor, field, x, y, z, values, used_val,
                   scale, db_floor, simulation, cmap, legend, unfold,
                   structures, **kw):
    """:func:`plot_field`'s body, with the plane's display order already
    settled so every helper it calls agrees on it."""
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    h_coord, v_coord, arr2d, (h_letter, v_letter) = _reduce_to_plane(
        values, x, y, z)

    arr = np.asarray(arr2d.values, dtype=np.float64)
    arr, cb_label, db_norm = _apply_scale(arr, field, used_val, scale,
                                          db_floor, dict(da.attrs))

    if ax is None:
        _, ax = plt.subplots()

    sim = simulation if simulation is not None else _sim_from_manifest(data)
    slice_axis, slice_val = _slice_axis_value(da, x, y, z, h_letter, v_letter,
                                              values)
    mirror_h, mirror_v = (False, False)
    if sim is not None and unfold and slice_axis is not None:
        mirror_h, mirror_v = _style.unfold_in_plane(sim, slice_axis, unfold)
    # A simulation that folded a declared symmetry plane (design spec §4.5)
    # hands out the plane already whole through its RunResult: the outlines
    # and the mirror marks still unfold, the recorded data must not.
    fold = getattr(sim, "_fold", None) if sim is not None else None
    data_mirror = (False, False) if (fold is not None and monitor in fold.unfolded_monitors) else (mirror_h, mirror_v)

    cmap_name, norm = _style.field_cmap_and_norm(field, used_val, arr, cmap)
    if db_norm is not None:
        norm = db_norm
    h_all, v_all = [h_coord], [v_coord]
    mesh = None
    for sign_h, sign_v in _style.mirror_copies(*data_mirror):
        hc, vc, block = _mirror_block(
            h_coord, v_coord, arr, sign_h, sign_v, field, used_val,
            (h_letter, v_letter), sim)
        mesh = ax.pcolormesh(hc, vc, block, cmap=cmap_name, norm=norm,
                             shading="nearest", **kw)
        h_all.append(hc)
        v_all.append(vc)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label(cb_label)

    # Structure outlines (design §3): reuse the §5 geometry; need a Simulation.
    # The field's coordinates are the user's; the scene is sampled in the
    # corner frame and its outlines moved into the field's frame (spec §4.4).
    before = set(ax.get_children())
    origin = geom.frame_origin(sim) if sim is not None else (0.0, 0.0, 0.0)
    drew_structure = False
    if structures:
        if sim is None:
            if not _NO_GEOMETRY_NOTED["flag"]:
                warnings.warn(
                    "plot_field(structures=True) has no geometry to outline; "
                    "pass simulation= to overlay structure outlines",
                    UserWarning, stacklevel=2)
                _NO_GEOMETRY_NOTED["flag"] = True
        elif slice_axis is not None:
            value_c = slice_val - origin[geom.axis_index(slice_axis)]
            drew_structure = _draw_outlines(ax, sim, slice_axis, value_c,
                                            mirror_h, mirror_v)
    drew_symmetry = False
    if slice_axis is not None and (mirror_h or mirror_v):
        drew_symmetry = _style.draw_symmetry_planes(ax, sim, slice_axis,
                                                    mirror_h, mirror_v)
    if sim is not None and slice_axis is not None:
        _style.translate_artists(ax, before, *geom.plane_offsets(origin, slice_axis))

    # Crop to the recorded slice, unfolded copies included. Outlines of
    # structures that extend past the monitor plane would otherwise stretch
    # the frame around empty space; the field's own extent is the picture.
    h_lo = min(float(np.min(c)) for c in h_all)
    h_hi = max(float(np.max(c)) for c in h_all)
    v_lo = min(float(np.min(c)) for c in v_all)
    v_hi = max(float(np.max(c)) for c in v_all)
    ax.set_xlim(h_lo, h_hi)
    ax.set_ylim(v_lo, v_hi)

    stretch = _style.set_view_aspect(ax, h_hi - h_lo, v_hi - v_lo)
    ax.set_xlabel(f"{h_letter} (µm)")
    ax.set_ylabel(f"{v_letter} (µm)")
    # The figure's own caption: what, at which wavelength or time, on which
    # plane; and the mesh it was run on when the scene is known.
    where = _sample_label(values)
    title = f"{monitor}: {_quantity_label(field, used_val)}"
    if where:
        title += f", {where}"
    if slice_axis is not None:
        title += f", {_style.cut_label(slice_axis, slice_val)}"
    _style.set_titles(ax, title, _style.join_notes(
        _style.mesh_summary(sim) if sim is not None else None,
        _style.stretch_note(v_letter, stretch)))
    if legend and (drew_structure or drew_symmetry):
        _style.add_legend(ax, source=False, monitor=False, pml=False,
                          structure=drew_structure, over_field=True,
                          symmetry=drew_symmetry,
                          # A long thin plane has no corner to spare; anything
                          # else keeps matplotlib's own choice, as before.
                          loc=(_style.LEGEND_OUTSIDE
                               if _style.legend_goes_outside(h_hi - h_lo, v_hi - v_lo)
                               else "best"))
    if owns_figure:
        _style.size_figure_for_plane(ax, h_hi - h_lo, v_hi - v_lo)
    return ax


def _quantity_label(field: str, used_val: str) -> str:
    """``|E|``, ``|E|²``, ``|H|``, or ``Ex (real)``."""
    if field in ("E", "H"):
        return f"|{field}|"
    if field == "intensity":
        return "|E|²"
    return f"{field} ({used_val})"


def _sample_label(values) -> str:
    """``λ = 1550 nm`` or ``t = 12.3 fs`` from the scalar coordinate the
    sample selection left behind, or empty for data with neither."""
    for dim, fmt in (("f", lambda f: f"λ = {c0 / f * 1e9:.4g} nm"),
                     ("t", lambda t: f"t = {t * 1e15:.3g} fs")):
        if dim in values.coords and values.coords[dim].ndim == 0:
            return fmt(float(values.coords[dim].values))
    return ""


_SCALES = ("max", "db", "raw")


def _apply_scale(arr, field: str, used_val: str, scale: str, db_floor: float,
                 attrs: dict):
    """Rescale the displayed slice and name the colorbar. Returns
    ``(arr, label, norm_override)``; the override is set only for decibels,
    whose range is fixed at ``[db_floor, 0]`` rather than taken from the
    data."""
    from matplotlib.colors import Normalize

    if scale not in _SCALES:
        raise ValueError(f"scale must be one of {_SCALES}, got {scale!r}")
    base = _quantity_label(field, used_val)
    magnitude = field in ("E", "H", "intensity") or used_val == "abs"

    if scale == "raw" or used_val == "phase":
        return arr, _style.field_colorbar_label(field, used_val, attrs), None

    finite = arr[np.isfinite(arr)]
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    if peak <= 0.0:
        return arr, _style.field_colorbar_label(field, used_val, attrs), None

    if scale == "max":
        return arr / peak, f"{base} / max", None

    if not magnitude:
        raise ValueError(
            "scale='db' needs a magnitude: field='E', 'H', 'intensity', or "
            f"val='abs'; got field={field!r} with val={used_val!r}")
    # Intensity is already a power; an amplitude squares on the way to dB.
    factor = 10.0 if field == "intensity" else 20.0
    with np.errstate(divide="ignore"):
        db = factor * np.log10(np.maximum(arr, 0.0) / peak)
    db = np.clip(np.where(np.isfinite(db), db, db_floor), db_floor, 0.0)
    return db, f"{base} (dB re max)", Normalize(vmin=db_floor, vmax=0.0)


def component_parity(component: str, symmetry_axis: str, symmetry: int) -> int:
    """Parity (+1 even, -1 odd) of one field component about a NUMERICS §20
    mirror plane normal to ``symmetry_axis``.

    An odd (``-1``, PEC) plane pins tangential E to zero on the plane, so
    tangential E is odd and normal E even; H is the dual, normal odd and
    tangential even. An even (``+1``, PMC) plane is the electric dual of that,
    which negates all four. So the whole table is one base sign flipped twice::

        base   = +1 for the component along the plane normal, -1 otherwise
        H flips it, and an even plane flips it again.
    """
    parity = 1 if component[-1] == symmetry_axis else -1
    if component[0] == "H":
        parity = -parity
    if symmetry > 0:
        parity = -parity
    return parity


def _mirror_sign(field, used_val, letter, sim):
    """How the displayed quantity transforms across the §20 plane normal to
    ``letter``: ``+1`` keep, ``-1`` negate, ``"phase"`` shift by pi.

    A magnitude is even by construction. A signed component carries its own
    parity, and a PHASE cannot be negated: a sign flip on a phasor is a pi
    turn, so the mirrored phase is the wrapped ``angle + pi``."""
    if field in ("E", "H", "intensity") or used_val == "abs":
        return 1
    parity = component_parity(field, letter, sim.symmetry["xyz".index(letter)])
    if parity == 1:
        return 1
    return "phase" if used_val == "phase" else -1


def _apply_sign(block, sign):
    if sign == 1:
        return block
    if sign == "phase":
        # angle(-z) = angle(z) + pi, wrapped back onto [-pi, pi].
        return (block + 2.0 * np.pi) % (2.0 * np.pi) - np.pi
    return sign * block


def _mirror_block(h_coord, v_coord, arr, sign_h, sign_v, field, used_val,
                  letters, sim):
    """One unfolded copy of the recorded plane: coordinates negated (and
    re-sorted, so they stay monotonic) and the samples reflected with the
    component's parity."""
    h_letter, v_letter = letters
    hc, vc, block = h_coord, v_coord, arr
    if sign_h == -1:
        hc, block = _reflect_axis(
            hc, block, 1, _mirror_sign(field, used_val, h_letter, sim))
    if sign_v == -1:
        vc, block = _reflect_axis(
            vc, block, 0, _mirror_sign(field, used_val, v_letter, sim))
    return hc, vc, block


def _reflect_axis(coord, block, block_axis: int, sign):
    """Negate one coordinate axis of a recorded plane and reflect the samples
    with it.

    A sample sitting ON the mirror (coordinate 0) is its own image, and
    ``shading="nearest"`` centres a whole cell on it, so keeping it would
    paint the mirrored copy's cell over the original's at the plane. It is
    dropped from the mirrored block instead."""
    coord = np.asarray(coord, dtype=np.float64)
    flipped = np.flip(coord) * -1.0
    block = _apply_sign(np.flip(block, axis=block_axis), sign)
    if coord.size > 1 and abs(coord[0]) < 0.5 * abs(coord[1] - coord[0]):
        flipped = flipped[:-1]
        block = block[:, :-1] if block_axis == 1 else block[:-1, :]
    return flipped, block


def _slice_axis_value(da, x, y, z, h_letter, v_letter, values):
    """The (axis, value) of the cut plane for the structure overlay: the
    in-plane axes are h_letter/v_letter, so the slice axis is the remaining
    one. Its value is the explicit x/y/z if given, else the monitor's own
    fixed-plane coordinate."""
    slice_axis = next(a for a in "xyz" if a not in (h_letter, v_letter))
    explicit = {"x": x, "y": y, "z": z}[slice_axis]
    if explicit is not None:
        return slice_axis, float(explicit)
    # Use the monitor's own fixed-plane coordinate if it carries one.
    if slice_axis in da.coords and da.coords[slice_axis].size >= 1:
        return slice_axis, float(np.asarray(da.coords[slice_axis].values).flat[0])
    return None, None


# A scene with more distinct permittivities than this is a material sweep or
# a custom-data medium, not a stack of interfaces worth drawing one by one.
_MAX_OUTLINE_LEVELS = 8


def _draw_outlines(ax, sim, axis, value, mirror_h=False,
                   mirror_v=False) -> bool:
    """Outline the MATERIAL BOUNDARY on the cut plane, as a contour of the
    same ε sample :func:`plot_index` draws.

    Not one outline per structure: a device is routinely assembled from
    several bodies of one material (this crossing is four tapers plus four
    arms), and outlining each of them draws every seam where two of them
    overlap. What a field plot is asking about is where the material ENDS, so
    the boundary is taken from the sampled ε, where coincident bodies of one
    permittivity have already merged into one region.

    White on a soft dark halo, not the ε views' near-black: the field maps are
    dark where the device usually sits ("magma" is black at zero, "RdBu_r"
    deep blue), so a dark outline vanishes into exactly the background it is
    meant to separate the material from."""
    from .eps import axis_cell_centers_um, sample_eps_plane

    levels = _material_levels(sim)
    if not levels:
        return False
    realized = sim._realized_um()
    if not (0.0 <= value <= realized[geom.axis_index(axis)]):
        return False

    # Follow what the run rasterizes: on a smoothed scene the §16 average
    # places the boundary inside the cell that straddles it, so the contour
    # stops following the staircase the hard sample would show.
    h_nodes, v_nodes, eps = sample_eps_plane(sim, axis, value,
                                             subpixel=bool(sim.subpixel))
    h_centers = axis_cell_centers_um(h_nodes)
    v_centers = axis_cell_centers_um(v_nodes)
    style = _style.field_outline_style()

    drew = False
    for sign_h, sign_v in _style.mirror_copies(mirror_h, mirror_v):
        hc, block = ((h_centers, eps) if sign_h == 1
                     else (-h_centers[::-1], eps[:, ::-1]))
        vc, block = ((v_centers, block) if sign_v == 1
                     else (-v_centers[::-1], block[::-1, :]))
        contours = ax.contour(hc, vc, block, levels=levels,
                              colors=style["edgecolor"],
                              linewidths=style["linewidth"],
                              zorder=style["zorder"])
        contours.set_path_effects(style["path_effects"])
        drew = True
    return drew


def _material_levels(sim):
    """Contour levels that separate the scene's permittivities: one midway
    between each neighbouring pair actually present. An empty list means there
    is no interface to draw (a single-material scene), or too many to be an
    outline."""
    values = sorted({round(float(sim.background.permittivity), 9)}
                    | {round(float(s.medium.permittivity), 9)
                       for s in sim.structures})
    if len(values) < 2 or len(values) > _MAX_OUTLINE_LEVELS + 1:
        return []
    return [0.5 * (lo + hi) for lo, hi in zip(values, values[1:])]


def _sim_from_manifest(data) -> Optional[object]:
    """Reconstruct a Simulation from the output manifest if it carries the
    input structures. Today's manifest (data.py) does not persist the structure
    list, so this returns None, the §12 forward-compat seam for self-describing
    results. Kept as a single lookup point so persisting structures later only
    needs a change here."""
    manifest = getattr(data, "manifest", {}) or {}
    spec = manifest.get("simulation") or manifest.get("input_spec")
    if not spec:
        return None
    try:
        from ..components.simulation import Simulation
        return Simulation.model_validate(spec)
    except Exception:
        return None
