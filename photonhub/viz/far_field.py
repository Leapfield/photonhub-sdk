"""``plot_far_field_cut()``, a polar cut through a far-field pattern, and
``plot_far_field_map()``, the pattern over direction cosines (or over the
plane of a Cartesian grid).

Both read a :class:`photonhub.analysis.FarField`'s labelled
:attr:`~photonhub.analysis.FarField.dataset` (what
:func:`photonhub.analysis.far_field` returns), so they are module-level
helpers like :func:`plot_mode`: a far field comes from post-processing, not
from a model. They return the matplotlib ``Axes`` and never call
``plt.show()``.
"""

import numpy as np

from ..constants import c0
from . import _style

_QUANTITIES = ("intensity", "directivity")
_SCALES = ("raw", "max", "db")
_UNITS = {"intensity": "W/sr", "directivity": ""}


def _quantity(pattern, quantity: str, freq_hz):
    """``(dataset, the quantity at one frequency, its frequency)``."""
    if quantity not in _QUANTITIES:
        raise ValueError(f"quantity must be one of {_QUANTITIES}, got {quantity!r}")
    ds = pattern.dataset
    if quantity not in ds:
        raise ValueError(
            "the directivity needs a meshed angle grid (theta_rad and phi_rad, mesh=True); "
            "plot the intensity instead")
    da = ds[quantity]
    if freq_hz is None:
        if da.sizes["f"] != 1:
            raise ValueError(
                f"this projection has {da.sizes['f']} frequencies: pass freq_hz= to pick one")
        da = da.isel(f=0)
    else:
        # by position: the projected frequencies may be unsorted or repeated
        f = np.asarray(ds["f"].values, dtype=np.float64)
        da = da.isel(f=int(np.argmin(np.abs(f - float(freq_hz)))))
    return ds, da, float(da.coords["f"].values)


def _scaled(values: np.ndarray, quantity: str, scale: str, db_floor: float):
    """``(values to draw, the axis label's unit, the floor or None)``."""
    if scale not in _SCALES:
        raise ValueError(f"scale must be one of {_SCALES}, got {scale!r}")
    v = np.asarray(values, dtype=np.float64)
    peak = float(np.nanmax(v)) if np.any(np.isfinite(v)) else 0.0
    if scale == "raw":
        return v, _UNITS[quantity], None
    if peak <= 0.0:
        raise ValueError(
            f"the {quantity} is zero everywhere in what is drawn (an empty half space, "
            "or a plane's dark side): nothing to scale; draw the other half space or "
            "scale='raw'")
    if scale == "max":
        return v / peak, "relative to its maximum", None
    ref = 1.0 if quantity == "directivity" else peak
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(v / ref)
    floor = float(np.nanmax(db)) + float(db_floor)
    return np.maximum(db, floor), ("dBi" if quantity == "directivity" else "dB"), floor


def _title(quantity: str, freq_hz: float) -> str:
    wl = c0 / freq_hz * 1e6 if freq_hz > 0 else float("inf")
    return f"{quantity}, λ = {wl:.4g} µm"


def plot_far_field_cut(pattern, *, phi_rad: float = 0.0, quantity: str = "directivity",
                       freq_hz=None, scale: str = "raw", db_floor: float = -40.0, ax=None,
                       **kw):
    """A polar cut through a far-field pattern on a meshed angle grid: the
    ``quantity`` against the polar angle ``theta`` in the plane of azimuth
    ``phi_rad`` (the grid's azimuth nearest it), drawn on a polar ``Axes``
    with ``theta = 0`` (``+z``) at the top and ``theta`` growing clockwise.
    Where the grid also holds the opposite azimuth ``phi_rad + pi``, that
    half of the plane is drawn on the left, so a full grid gives the whole
    great circle. Returns the ``Axes``.

    ``quantity`` is ``"directivity"`` (the default) or ``"intensity"``
    (W/sr). ``scale``: ``"raw"`` (the values), ``"max"`` (divided by the
    cut's maximum) or ``"db"`` (decibels: the directivity in dBi, the
    intensity relative to the cut's maximum, both drawn down to
    ``db_floor`` below the cut's peak). ``freq_hz`` picks the frequency
    nearest it (required when the projection has several; the projected
    frequencies may be in any order). ``ax=`` draws into an existing polar
    ``Axes``; extra ``**kw`` pass through to ``plot``."""
    import matplotlib.pyplot as plt

    ds, da, f = _quantity(pattern, quantity, freq_hz)
    if ds.attrs.get("grid") != "angle" or "theta" not in da.dims or "phi" not in da.dims:
        raise ValueError(
            "plot_far_field_cut needs a meshed angle grid (theta_rad and phi_rad, mesh=True)")
    phis = np.asarray(ds["phi"].values, dtype=np.float64)
    theta = np.asarray(ds["theta"].values, dtype=np.float64)
    wrap = np.mod(phis - float(phi_rad) + np.pi, 2.0 * np.pi) - np.pi
    i0 = int(np.argmin(np.abs(wrap)))
    opposite = np.flatnonzero(np.abs(np.mod(phis - phis[i0], 2.0 * np.pi) - np.pi) < 1e-9)
    near = np.asarray(da.isel(phi=i0).values, dtype=np.float64)
    angles, values = theta, near
    if opposite.size:
        far = np.asarray(da.isel(phi=int(opposite[0])).values, dtype=np.float64)
        # a grid that does not reach theta = 0 (a cone) leaves a gap there
        gap = [np.nan] if theta[0] > 1e-12 else []
        angles = np.concatenate([-theta[::-1], gap, theta])
        values = np.concatenate([far[::-1], gap, near])
    shown, unit, floor = _scaled(values, quantity, scale, db_floor)

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    elif getattr(ax, "name", None) != "polar":
        raise ValueError("plot_far_field_cut draws on a polar Axes: "
                         "plt.subplots(subplot_kw={'projection': 'polar'})")
    ax.plot(angles, shown, **kw)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if floor is not None:
        ax.set_rmin(floor)
    label = f"φ = {np.degrees(phis[i0]):.4g}°"
    if opposite.size:
        label += f" (right), {np.degrees(phis[int(opposite[0])]):.4g}° (left)"
    ax.set_title(f"{_title(quantity, f)}, {label}" + (f" [{unit}]" if unit else ""))
    return ax


def plot_far_field_map(pattern, *, quantity: str = "intensity", freq_hz=None,
                       scale: str = "max", db_floor: float = -40.0, half_space: str = "+",
                       ax=None, cmap=None, **kw):
    """A map of a far-field pattern on a meshed grid. Returns the ``Axes``.

    * A k-space grid: over its direction cosines ``(ux, uy)``, with the
      unit circle (the edge of the propagating directions) drawn.
    * An angle grid: the half space ``half_space`` (``"+"``, ``theta <=
      pi/2``, or ``"-"``) over its direction cosines along x and y,
      ``(sin(theta) cos(phi), sin(theta) sin(phi))``, the view down the z
      axis.
    * A Cartesian grid: over the plane's two axes (um).

    ``quantity`` is ``"intensity"`` (W/sr, the default) or
    ``"directivity"`` (a meshed angle grid). ``scale``: ``"max"`` (the
    default, divided by the map's maximum), ``"db"`` (decibels, the
    directivity in dBi and the intensity relative to its maximum, down to
    ``db_floor`` below the peak) or ``"raw"``. ``freq_hz`` picks the
    frequency nearest it (required when there are several). ``cmap=``
    overrides the sequential colormap; ``ax=`` draws into an existing
    ``Axes``; extra ``**kw`` pass through to ``pcolormesh`` (``vmin``,
    ``vmax`` or ``norm`` replace the default color range)."""
    import matplotlib.pyplot as plt

    ds, da, f = _quantity(pattern, quantity, freq_hz)
    grid = ds.attrs.get("grid")
    if "point" in da.dims:
        raise ValueError("plot_far_field_map needs a meshed grid (mesh=True)")
    if half_space not in ("+", "-"):
        raise ValueError(f"half_space must be '+' or '-', got {half_space!r}")
    if ax is None:
        _, ax = plt.subplots()
    shading = "nearest"
    if grid == "kspace":
        x = np.asarray(ds["ux"].values, dtype=np.float64)
        y = np.asarray(ds["uy"].values, dtype=np.float64)
        values = np.asarray(da.transpose("uy", "ux").values, dtype=np.float64)
        p, q = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[ds.attrs["plane_axis"]]
        xlabel, ylabel = f"u_{p} (direction cosine)", f"u_{q} (direction cosine)"
    elif grid == "angle":
        theta = np.asarray(ds["theta"].values, dtype=np.float64)
        phi = np.asarray(ds["phi"].values, dtype=np.float64)
        keep = theta <= np.pi / 2 + 1e-12 if half_space == "+" else theta >= np.pi / 2 - 1e-12
        if not np.any(keep):
            raise ValueError(f"the angle grid has no direction in the {half_space}z half space")
        vals = np.asarray(da.transpose("theta", "phi").values, dtype=np.float64)[keep]
        th = theta[keep]
        # close the azimuth when the grid covers the whole circle without repeating 2 pi
        if phi.size > 1 and abs(phi[0] + 2.0 * np.pi - phi[-1] - (phi[1] - phi[0])) < 1e-9:
            phi = np.append(phi, phi[0] + 2.0 * np.pi)
            vals = np.concatenate([vals, vals[:, :1]], axis=1)
        TH, PH = np.meshgrid(th, phi, indexing="ij")
        x, y = np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH)
        values = vals
        shading = "gouraud"
        xlabel, ylabel = "u_x (direction cosine)", "u_y (direction cosine)"
    elif grid == "cartesian":
        p, q = [d for d in da.dims]
        x = np.asarray(ds[p].values, dtype=np.float64)
        y = np.asarray(ds[q].values, dtype=np.float64)
        values = np.asarray(da.transpose(q, p).values, dtype=np.float64)
        xlabel, ylabel = f"{p} (µm)", f"{q} (µm)"
    else:
        raise ValueError(f"unknown far-field grid {grid!r}")
    shown, unit, floor = _scaled(values, quantity, scale, db_floor)
    finite = shown[np.isfinite(shown)]
    vmin = floor if floor is not None else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if "norm" not in kw:
        kw.setdefault("vmin", vmin)
        kw.setdefault("vmax", vmax)
    mesh = ax.pcolormesh(x, y, shown, cmap=cmap or _style._MAGNITUDE_CMAP, shading=shading,
                         **kw)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label(f"{quantity}" + (f" ({unit})" if unit else ""))
    if grid in ("kspace", "angle"):
        t = np.linspace(0.0, 2.0 * np.pi, 361)
        ax.plot(np.cos(t), np.sin(t), color="#ffffff", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    side = f", {half_space}z half space" if grid == "angle" else ""
    ax.set_title(_title(quantity, f) + side)
    return ax
