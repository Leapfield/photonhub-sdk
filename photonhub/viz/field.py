"""``plot_field()`` — a field-component heatmap on a 2D slice of a monitor's
DataArray (design §3, §7, §9).

Consumes the ``xarray.DataArray`` that ``SimulationData[monitor]`` returns
(already in µm coordinates). Supports the raw components Ex..Hz plus derived
``"E"`` (vector magnitude), ``"intensity"`` (|E|²) and ``"H"``; ``freq=`` is
required for a multi-frequency DFT monitor; ``val`` in real/imag/abs/phase
selects what to show for complex data. Colormap/normalization follow §7 and a
colorbar is labeled from the DataArray's attrs. ``structures=True`` overlays
structure outlines using the §5 cut-plane geometry.
"""

import warnings
from typing import Optional

import numpy as np

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
    """Canonical (horizontal, vertical) order for the two surviving spatial
    dims, matching the §5 in-plane axis convention."""
    s = set(remaining)
    if s == {"x", "y"}:
        return "x", "y"
    if s == {"y", "z"}:
        return "y", "z"
    if s == {"x", "z"}:
        return "x", "z"
    # Fallback (shouldn't happen): keep given order.
    return remaining[0], remaining[1]


def plot_field(data, monitor, field="Ex", x=None, y=None, z=None, *,
               freq=None, time=None, val="real", structures=True,
               simulation=None, ax=None, cmap=None, legend=True, **kw):
    """Heatmap of a field component on a 2D slice of ``data[monitor]``.

    ``data`` is a :class:`SimulationData`; ``monitor`` is its key. ``freq=``
    picks a frequency on a DFT monitor; ``time=`` (seconds) picks a recorded
    sample on a time/snapshot monitor (default: the last frame). See the module
    docstring and design §3 for the rest. Returns the matplotlib ``Axes``."""
    import matplotlib.pyplot as plt

    da = data[monitor]  # KeyError (with available list) for an unknown monitor.

    values, used_val = _component_array(da, field, freq, val, time)
    h_coord, v_coord, arr2d, (h_letter, v_letter) = _reduce_to_plane(
        values, x, y, z)

    arr = np.asarray(arr2d.values, dtype=np.float64)

    if ax is None:
        _, ax = plt.subplots()

    cmap_name, norm = _style.field_cmap_and_norm(field, used_val, arr, cmap)
    mesh = ax.pcolormesh(h_coord, v_coord, arr, cmap=cmap_name, norm=norm,
                         shading="nearest", **kw)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label(_style.field_colorbar_label(field, used_val, dict(da.attrs)))

    # Structure outlines (design §3): reuse the §5 geometry; need a Simulation.
    drew_structure = False
    if structures:
        slice_axis, slice_val = _slice_axis_value(da, x, y, z, h_letter,
                                                  v_letter, values)
        sim = simulation if simulation is not None else _sim_from_manifest(data)
        if sim is None:
            if not _NO_GEOMETRY_NOTED["flag"]:
                warnings.warn(
                    "plot_field(structures=True) has no geometry to outline; "
                    "pass simulation= to overlay structure outlines",
                    UserWarning, stacklevel=2)
                _NO_GEOMETRY_NOTED["flag"] = True
        elif slice_axis is not None:
            drew_structure = _draw_outlines(ax, sim, slice_axis, slice_val)

    ax.set_aspect("equal")
    ax.set_xlabel(f"{h_letter} (µm)")
    ax.set_ylabel(f"{v_letter} (µm)")
    ax.set_title(f"{monitor}: {field} ({used_val})")
    if legend and drew_structure:
        _style.add_legend(ax, source=False, monitor=False, pml=False,
                          structure=True)
    return ax


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


def _draw_outlines(ax, sim, axis, value) -> bool:
    style = dict(fill=False, edgecolor=_style.STRUCTURE_EDGE, linewidth=1.0,
                 zorder=3)
    drew = False
    for structure in sim.structures:
        spec = geom.structure_patch_spec(structure.geometry, axis, value)
        if spec is None:
            continue
        kind, params = spec
        _style.add_structure_patch(ax, kind, params, style=style)
        drew = True
    return drew


def _sim_from_manifest(data) -> Optional[object]:
    """Reconstruct a Simulation from the output manifest if it carries the
    input structures. Today's manifest (data.py) does not persist the structure
    list, so this returns None — the §12 forward-compat seam for self-describing
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
