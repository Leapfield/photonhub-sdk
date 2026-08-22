"""Import source — launch a user-supplied field profile as a Huygens sheet.

The PhotonHub analogue of Lumerical's *Import source* and Tidy3D's
``CustomFieldSource``: any transverse field map — from another solver, an
analytic model, a measurement, or a previous run — becomes an excitation by
resampling it onto the simulation's Yee injection plane and stamping the
per-cell equivalence-current sheet
(:func:`~photonhub.plugins.eq_current_source.equivalence_current_source`,
``J = n̂ × H``, ``M = -n̂ × E``), the same machinery behind
:func:`~photonhub.plugins.mode_devices.mode_launch` and
:func:`~photonhub.plugins.gaussian_beam.gaussian_beam_source`.

Two layers, mirroring the Gaussian-beam plugin:

* :func:`import_field` — resample the user arrays onto the plane's true Yee
  sample locations and package them as a ``yee_staggered``
  :class:`~photonhub.plugins.vector_modes.VectorMode` (also usable as a
  monitor/overlap reference);
* :func:`import_source` — that plus the sheet, returning the ``PointDipole``
  list for ``Simulation.sources``.

Field conventions
=================
Arrays are indexed ``[iv, ih]`` over the plane's natural in-plane axes
``(h, v)`` (x-cut -> (y, z), y-cut -> (x, z), z-cut -> (x, y)), on the
caller's own rectilinear grid ``coords_h_um`` x ``coords_v_um`` measured
RELATIVE to ``center_um`` (so a profile exported centred on 0 drops in
unchanged). Values are complex phasors in the engine's ``e^{-i w t}``
convention; points outside the supplied grid are taken as 0.

``H`` is optional: when only ``e_h``/``e_v`` are given, the paired magnetic
field is filled in the forward quasi-plane-wave limit
``(h_h, h_v) = (n / eta0) (-e_v, +e_h)`` — exact for a normal-incidence plane
wave, the scalar limit otherwise (same approximation the scalar mode stack
uses). Supply all four components for tilted/structured fields where the
true E/H relation matters; ``direction`` is applied by the sheet either way.

The transverse-E pair is jointly L2-normalized (all components scaled
together, keeping E/H a consistent Huygens pair); absolute launch strength is
set by ``power_watts`` at :func:`import_source` (``None`` keeps the imported
units).

What's not handled
==================
* **Broadband profile banks** — one profile, phased at the pulse centre. For
  a per-frequency profile bank call ``equivalence_current_source`` directly
  with ``modes_by_freq={freq: import_field(...)}``.
* **Recorded-monitor import** — a ``field_dft`` plane stores its four
  tangential components on ONE base-index coordinate set while each
  component physically sits half a cell off it; feeding those arrays here
  treats them as co-located (a <= half-cell registration blur). A
  stagger-exact ``from_monitor`` importer is a follow-up.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..components.sources import PointDipole
from ..viz import _geometry as _geom
from .gaussian_beam import _plane_grids, _resolve_index, _resolve_wavelength
from .mode_overlap import ETA0
from .vector_modes import VectorMode
from .yee_mode import _window_center_offset

__all__ = ["import_field", "import_source"]

_AXES = ("x", "y", "z")


def _as_field(name: str, arr, nv: int, nh: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.complex128)
    if a.shape != (nv, nh):
        raise ValueError(
            f"{name} has shape {a.shape}, expected (len(coords_v_um), "
            f"len(coords_h_um)) = ({nv}, {nh}) — arrays are indexed [iv, ih]")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _as_coords(name: str, arr) -> np.ndarray:
    c = np.asarray(arr, dtype=np.float64)
    if c.ndim != 1 or c.size < 2:
        raise ValueError(f"{name} must be a 1-D coordinate array with >= 2 "
                         f"entries, got shape {c.shape}")
    if not np.all(np.diff(c) > 0):
        raise ValueError(f"{name} must be strictly ascending")
    return c


def _interp(coords_v, coords_h, field):
    """Bilinear complex interpolation, 0 outside the supplied grid."""
    re = RegularGridInterpolator((coords_v, coords_h), field.real,
                                 bounds_error=False, fill_value=0.0)
    im = RegularGridInterpolator((coords_v, coords_h), field.imag,
                                 bounds_error=False, fill_value=0.0)

    def at(H, V):
        pts = np.stack([V.ravel(), H.ravel()], axis=-1)
        return (re(pts) + 1j * im(pts)).reshape(H.shape)

    return at


def _clip_window(sim, axis, h_c, v_c, half_w, half_v):
    h_letter, v_letter = _geom.in_plane_axes(axis)
    size = sim.size_um

    def clip(half, c, letter):
        L = float(size[_AXES.index(letter)])
        return min(float(half), max(c, L - c))

    half_w = clip(half_w, h_c, h_letter)
    half_v = clip(half_v, v_c, v_letter)
    if not (half_w > 0.0 and half_v > 0.0):
        raise ValueError("the import window half-extents must be > 0")
    return half_w, half_v


def import_field(
    sim,
    *,
    axis: str,
    e_h,
    e_v,
    coords_h_um,
    coords_v_um,
    h_h=None,
    h_v=None,
    wavelength_um: Optional[float] = None,
    freq_hz: Optional[float] = None,
    source_time=None,
    n: Optional[float] = None,
    n_eff: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    half_w_um: Optional[float] = None,
    half_v_um: Optional[float] = None,
) -> VectorMode:
    """Resample a user field map onto the ``axis``-normal Yee plane.

    Parameters
    ----------
    sim:
        The simulation whose grid/size/§20 symmetry the profile is sampled on
        (a placeholder shell with the same grid is fine).
    axis:
        The injection plane's normal, ``"x"``/``"y"``/``"z"``.
    e_h, e_v:
        Complex transverse E components along the plane's (h, v) in-plane
        axes, shape ``(len(coords_v_um), len(coords_h_um))``.
    coords_h_um, coords_v_um:
        Strictly-ascending sample coordinates of the arrays, RELATIVE to
        ``center_um``.
    h_h, h_v:
        Optional transverse H (same shape/grid). Give both or neither;
        omitted -> forward quasi-plane-wave fill (module docstring).
    wavelength_um, freq_hz, source_time:
        The frequency the profile is phased at (at most one of the first
        two; else ``source_time.freq0_hz``).
    n:
        Index of the launch medium. Default
        ``sqrt(sim.background.permittivity)``.
    n_eff:
        Effective longitudinal phase index used by the sheet's half-cell
        straddle (a beam tilted by theta has ``n cos(theta)``). Default: ``n``.
    center_um:
        Transverse placement of the profile origin as ``(h, v)``; default the
        domain centre (0 on a §20-folded axis).
    half_w_um, half_v_um:
        Half-extents of the sampled window; default the data's own extent,
        clipped to the domain.

    Returns
    -------
    VectorMode
        ``yee_staggered``, six components on the window (``e_a = h_a = 0``),
        transverse-E jointly L2-normalized, ready for
        :func:`~photonhub.plugins.eq_current_source.equivalence_current_source`,
        :func:`~photonhub.plugins.mode_devices.mode_monitor`, or overlaps.
    """
    if axis not in _AXES:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if (h_h is None) != (h_v is None):
        raise ValueError("give both h_h and h_v, or neither")
    dl = getattr(sim.grid, "dl_um", None)
    if not dl:
        raise ValueError("import_field needs the grid's base dl_um")
    dl = float(dl)

    ch = _as_coords("coords_h_um", coords_h_um)
    cv = _as_coords("coords_v_um", coords_v_um)
    nv, nh = cv.size, ch.size
    E1 = _as_field("e_h", e_h, nv, nh)
    E2 = _as_field("e_v", e_v, nv, nh)
    have_h = h_h is not None
    if have_h:
        H1 = _as_field("h_h", h_h, nv, nh)
        H2 = _as_field("h_v", h_v, nv, nh)

    lam_um = _resolve_wavelength(wavelength_um, freq_hz, source_time)
    n_bg = _resolve_index(sim, n)
    n_eff = float(n_eff) if n_eff is not None else n_bg
    if not n_eff > 0.0:
        raise ValueError(f"n_eff must be > 0, got {n_eff}")

    h_letter, v_letter = _geom.in_plane_axes(axis)
    size = sim.size_um
    if center_um is None:
        h_c = float(size[_AXES.index(h_letter)]) / 2.0
        v_c = float(size[_AXES.index(v_letter)]) / 2.0
    else:
        h_c, v_c = float(center_um[0]), float(center_um[1])

    if half_w_um is None:
        half_w_um = max(abs(float(ch[0])), abs(float(ch[-1])))
    if half_v_um is None:
        half_v_um = max(abs(float(cv[0])), abs(float(cv[-1])))
    half_w, half_v = _clip_window(sim, axis, h_c, v_c, half_w_um, half_v_um)

    h_node, v_node, (h_dq, v_dq), grids = _plane_grids(
        sim, axis, h_center=h_c, v_center=v_c, half_w=half_w, half_v=half_v,
        dl=dl)

    def rel(grid):
        H, V = grid
        return H - h_c, V - v_c

    at_e1 = _interp(cv, ch, E1)
    at_e2 = _interp(cv, ch, E2)
    # E_h, H_v live at (h+1/2, v); E_v, H_h at (h, v+1/2) — _plane_grids
    ex = at_e1(*rel(grids["mid_node"]))
    ey = at_e2(*rel(grids["node_mid"]))
    if have_h:
        at_h1 = _interp(cv, ch, H1)
        at_h2 = _interp(cv, ch, H2)
        hx = at_h1(*rel(grids["node_mid"]))
        hy = at_h2(*rel(grids["mid_node"]))
    else:
        # forward quasi-plane-wave fill: (h_h, h_v) = (n/eta0)(-e_v, +e_h),
        # sampled at H's OWN locations (which its E partner shares in-plane).
        y0 = n_bg / ETA0
        hx = -y0 * at_e2(*rel(grids["node_mid"]))
        hy = +y0 * at_e1(*rel(grids["mid_node"]))
    zeros = np.zeros_like(ex)

    norm = math.sqrt(float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2)))
    if not norm > 0.0:
        raise ValueError(
            "the imported field is identically zero on the sampled window — "
            "check center_um / coords against the domain")
    ex, ey, hx, hy = (f / norm for f in (ex, ey, hx, hy))

    nvw, nhw = ex.shape
    graded = h_dq is not None or v_dq is not None
    if graded:
        off = (0.5 * float(h_node[0] + h_node[-1]) - h_c,
               0.5 * float(v_node[0] + v_node[-1]) - v_c)
    else:
        off = _window_center_offset(float(h_node[0]), float(v_node[0]),
                                    nhw, nvw, dl, h_c, v_c)
    return VectorMode(
        n_eff=n_eff,
        n_group=None,
        ex=ex, ey=ey, ez=zeros.copy(), hx=hx, hy=hy, hz=zeros.copy(),
        wavelength_um=lam_um,
        dl_x_um=dl,
        dl_y_um=dl,
        center_offset_um=off,
        yee_staggered=True,
        x_coords_um=(h_node - h_c) if graded else None,
        y_coords_um=(v_node - v_c) if graded else None,
    )


def import_source(
    sim,
    *,
    axis: str,
    position_um: float,
    source_time,
    e_h,
    e_v,
    coords_h_um,
    coords_v_um,
    h_h=None,
    h_v=None,
    direction: str = "+",
    power_watts: Optional[float] = 1.0,
    n: Optional[float] = None,
    n_eff: Optional[float] = None,
    wavelength_um: Optional[float] = None,
    freq_hz: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    half_w_um: Optional[float] = None,
    half_v_um: Optional[float] = None,
    amplitude_threshold: float = 1e-6,
) -> List[PointDipole]:
    """Launch an imported field map — the one-call custom excitation.

    :func:`import_field` (whose parameters this shares) plus the Huygens
    sheet. ``power_watts`` (default 1 W) normalizes the launched power on the
    engine's discrete Poynting quadrature exactly like the mode and beam
    launches (into the modelled half domain under §20 symmetry); ``None``
    keeps the imported amplitude scale. Returns the ``PointDipole`` list for
    ``Simulation.sources``.
    """
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    if power_watts is not None and not power_watts > 0.0:
        raise ValueError(f"power_watts must be > 0 or None, got {power_watts}")
    from .eq_current_source import equivalence_current_source

    field = import_field(
        sim, axis=axis, e_h=e_h, e_v=e_v, coords_h_um=coords_h_um,
        coords_v_um=coords_v_um, h_h=h_h, h_v=h_v,
        wavelength_um=wavelength_um, freq_hz=freq_hz,
        source_time=source_time, n=n, n_eff=n_eff, center_um=center_um,
        half_w_um=half_w_um, half_v_um=half_v_um)

    # Re-derive the window exactly as import_field resolved it, so the sheet
    # registers on the same ladder (same pattern as gaussian_beam_source).
    h_letter, v_letter = _geom.in_plane_axes(axis)
    size = sim.size_um
    if center_um is None:
        h_c = float(size[_AXES.index(h_letter)]) / 2.0
        v_c = float(size[_AXES.index(v_letter)]) / 2.0
    else:
        h_c, v_c = float(center_um[0]), float(center_um[1])
    ch = _as_coords("coords_h_um", coords_h_um)
    cv = _as_coords("coords_v_um", coords_v_um)
    hw = (half_w_um if half_w_um is not None
          else max(abs(float(ch[0])), abs(float(ch[-1]))))
    hv = (half_v_um if half_v_um is not None
          else max(abs(float(cv[0])), abs(float(cv[-1]))))
    hw, hv = _clip_window(sim, axis, h_c, v_c, hw, hv)

    return equivalence_current_source(
        sim, field, axis=axis, position_um=position_um,
        source_time=source_time, direction=direction,
        h_center_um=h_c, v_center_um=v_c, half_w_um=hw, half_v_um=hv,
        power_watts=(None if power_watts is None else float(power_watts)),
        amplitude_threshold=float(amplitude_threshold))
