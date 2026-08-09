"""``plot_mode()`` — the transverse field of an FDE eigenmode as a heatmap.

Renders a :class:`photonhub.plugins.modes.Mode` (the dominant transverse
component ``Ex``/``Ey`` of a guided mode) on its real-space µm cross-section.
The field is signed, so it uses the same diverging, zero-centered colormap the
field views use for a real/imag slice (design §7). The title carries the
component, the modal index ``n_eff``, and the free-space wavelength.

Consumes the mode's :meth:`~photonhub.plugins.modes.Mode.field_dataarray`
(an :class:`xarray.DataArray` already in µm x/y coords with ``n_eff`` /
``polarization`` / ``wavelength_um`` attrs), so it stays decoupled from the
solver internals. Returns the matplotlib ``Axes``; never calls ``plt.show()``.
"""

import numpy as np

from . import _style


def plot_mode(mode, *, ax=None, cmap=None, legend=False, **kw):
    """Heatmap of an FDE :class:`~photonhub.plugins.modes.Mode`'s transverse
    field on its µm cross-section. Returns the matplotlib ``Axes``.

    ``mode`` is a ``Mode`` (carrying ``.field``, ``.n_eff``, ``.polarization``,
    ``.wavelength_um`` and ``.field_dataarray()``). ``cmap=`` overrides the
    diverging colormap (the symmetric zero-centered normalization is kept).
    ``ax=`` draws into an existing Axes. Extra ``**kw`` pass through to
    ``pcolormesh``."""
    import matplotlib.pyplot as plt

    da = mode.field_dataarray()
    arr = np.asarray(da.values, dtype=np.float64)
    xs = np.asarray(da.coords["x"].values, dtype=np.float64)
    ys = np.asarray(da.coords["y"].values, dtype=np.float64)
    component = str(da.attrs.get("component", da.name or "E"))

    if ax is None:
        _, ax = plt.subplots()

    # Signed field -> the §7 diverging map, symmetric about 0 (reuse the field
    # colormap/normalization selection so a mode and a field slice match).
    cmap_name, norm = _style.field_cmap_and_norm(component, "real", arr, cmap)
    mesh = ax.pcolormesh(xs, ys, arr, cmap=cmap_name, norm=norm,
                         shading="nearest", **kw)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label(f"{component} (mode field)")

    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(
        f"{component} mode  (n_eff={float(mode.n_eff):.4g}, "
        f"λ={float(mode.wavelength_um):.4g} µm)"
    )
    return ax


def plot_overlap(mode_a, mode_b, *, axes=None, center_a=(0.0, 0.0),
                 center_b=(0.0, 0.0), direction="+", cmap=None):
    """Three-panel view of a mode⇄mode overlap (:func:`photonhub.plugins.mode_overlap`):
    ``|E|`` of each mode on the shared grid and the **overlap integrand**
    ``Re(E_A*·E_B)`` — so you can *see* where the two profiles match (the integrand
    is positive where they reinforce, negative/zero where they mismatch). The
    suptitle carries the power coupling, the field overlap and the mismatch loss.

    ``mode_a``/``mode_b`` are any of scalar :class:`~photonhub.plugins.modes.Mode`,
    :class:`~photonhub.plugins.vector_modes.VectorMode`, or
    :func:`~photonhub.plugins.mode_overlap.gaussian_mode`. ``center_b`` shifts mode B
    (lateral misalignment), ``direction="-"`` uses mode B's backward partner — both
    forwarded to the overlap so the panels match the reported numbers. Returns the
    three matplotlib ``Axes``; never calls ``plt.show()``."""
    import matplotlib.pyplot as plt

    from ..plugins.mode_overlap import (
        mode_overlap, _union_grid, _mode_plane_fields)

    r = mode_overlap(mode_a, mode_b, center_a=center_a, center_b=center_b,
                     direction=direction)
    c1, c2 = _union_grid(mode_a, mode_b, center_a, center_b)
    ma = _mode_plane_fields(mode_a, c1, c2, axis="z", center_um=center_a,
                            thickness_axis=None)
    mb = _mode_plane_fields(mode_b, c1, c2, axis="z", center_um=center_b,
                            thickness_axis=None, direction=direction)

    def _emag(m):
        e = np.sqrt(np.abs(m["e1"]) ** 2 + np.abs(m["e2"]) ** 2)
        peak = e.max()
        return e / peak if peak > 0 else e         # normalize shape, not scale
    ea, eb = _emag(ma), _emag(mb)
    integrand = np.real(np.conj(ma["e1"]) * mb["e1"]
                        + np.conj(ma["e2"]) * mb["e2"])

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    if len(axes) != 3:
        raise ValueError("axes must be a sequence of 3 matplotlib Axes")

    # |E| panels: positive magnitude, sequential map, peak-normalized.
    for ax, data, title in ((axes[0], ea, "|E| mode A"),
                            (axes[1], eb, "|E| mode B")):
        mesh = ax.pcolormesh(c1, c2, data, cmap=(cmap or "magma"),
                             vmin=0.0, vmax=1.0, shading="nearest")
        ax.figure.colorbar(mesh, ax=ax)
        ax.set_title(title)

    # Overlap integrand: signed → the §7 diverging map, symmetric about 0.
    cmap_name, norm = _style.field_cmap_and_norm("Ex", "real", integrand, None)
    mesh = axes[2].pcolormesh(c1, c2, integrand, cmap=cmap_name, norm=norm,
                              shading="nearest")
    axes[2].figure.colorbar(mesh, ax=axes[2])
    axes[2].set_title("overlap integrand  Re(E$_A^*$·E$_B$)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    axes[0].figure.suptitle(
        f"mode overlap — power {r.power:.4f}  ·  field {r.field:.4f}  ·  "
        f"mismatch {r.mismatch_db:.2f} dB  ({r.method})")
    return axes
