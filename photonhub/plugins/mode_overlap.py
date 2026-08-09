"""Directional-power mode-overlap: a recorded field plane -> mode-resolved T.

This is the Phase-2 Track-B *mode-monitor transmission* post-processor. Given a
field plane recorded by an FDTD run (the tangential ``E`` and ``H`` DataArrays
on a plane whose normal is the waveguide's propagation axis) and a frozen FDE
:class:`~photonhub.plugins.modes.Mode`, it computes the **forward (or backward)
power transmission** ``T(f)`` into that single mode. There is **NO S-matrix**
here — this is a one-mode-at-a-time projection.

Physics / method
================
**Directional power overlap.** For a monitor plane with outward normal ``n_hat``
along the propagation axis, the complex modal-amplitude coefficient of the
simulated field on the mode is (``e^{-i omega t}`` time convention)

    a_pm = (1/4) * integral_A [ E_sim x h_mode*  +  e_mode* x H_sim ] . n_hat dA

and the power transmission into the (normalized) mode is

    T = |a_pm|^2 / P_mode^2 ,
    P_mode = (1/2) * integral_A Re( e_mode x h_mode* ) . n_hat dA .

NOTE ON THE DENOMINATOR (deviation from the handoff brief): the brief wrote
``T = |a_pm|^2 / P_mode``, but with the ``1/4`` overlap coefficient above the
*self*-overlap evaluates to ``a_pm = P_mode`` exactly (substitute
``E_sim=e_mode, H_sim=h_mode``: both cross terms equal ``2*P_mode * (1/4)``).
``|a_pm|^2 / P_mode`` would then give ``P_mode`` rather than the required
``T=1``. The self-consistent power ratio is ``T = |a_pm|^2 / P_mode^2`` — i.e.
``a_pm`` is the *unnormalized* coefficient and the normalized modal amplitude is
``a_pm / P_mode``. We implement that (so self-overlap == 1 exactly); see the
test suite which pins it.

Carrying *both* the simulated ``E`` and ``H`` is what separates forward from
backward power: a clean single-mode field travelling along ``+n_hat`` reads
``T_forward ~= 1`` and ``T_backward ~= 0``; reverse the field's propagation and
the two swap. ``direction="-"`` selects the backward mode by flipping the modal
transverse ``H`` (``h_mode -> -h_mode``), equivalently picking ``a_minus``.

**Scalar-limit H reconstruction (APPROXIMATION).** The frozen FDE solver returns
only a *scalar* transverse ``E`` profile (the major component ``Ex`` for TE,
``Ey`` for TM) and a real ``n_eff`` — it carries no ``H`` and no minor-component
``E``. We therefore reconstruct the modal transverse ``H`` from the scalar mode
in the **quasi-TEM / weakly-guided limit**:

    e_mode  = major transverse E unit vector * scalar_profile   (minor E := 0),
    h_mode  = (n_eff / eta0) * ( z_hat x e_mode ) ,

with ``eta0`` the vacuum wave impedance and ``z_hat`` the propagation axis. This
is *exact* in the weakly-guided limit and *approximate* for high-contrast SOI
(it drops the longitudinal ``E_z``/``H_z`` and the minor transverse components).
That error is accepted for the MVP and is pinned later by a Tier-2b leakage
gate. With this reconstruction, ``e_mode x h_mode*`` is purely along ``n_hat``
and ``P_mode = (n_eff / (2 eta0)) * integral |profile|^2 dA``.

**Area element.** ``dA`` is taken from the plane's *real* transverse coordinate
spacings (centered-difference cell widths), so graded / non-uniform meshes are
handled correctly — no uniform-spacing assumption.

**Scope.** Fundamental mode, looped over the monitor's frequencies. By default
one scalar mode profile (+ its ``n_eff``) is used for every frequency (the frozen
mode); pass ``modes_by_freq`` to project each frequency onto its OWN solved mode
(profile + ``n_eff``), matching Tidy3D's per-frequency ``ModeMonitor`` and
recovering the waveguide dispersion the frozen mode drops. A scalar per-frequency
``n_eff`` override is also accepted.

Mode ⇄ mode overlap (no FDTD run)
=================================
:func:`mode_overlap` is the **mode-to-mode** companion: it takes two *modes* (a
scalar :class:`~photonhub.plugins.modes.Mode`, a full-vector
:class:`~photonhub.plugins.vector_modes.VectorMode`, or an analytic
:func:`gaussian_mode`) — not a recorded plane — resamples both onto a common
transverse grid, and returns their coupling efficiency. It answers "how much of
mode A couples into mode B": a waveguide TE0 into a lensed-fibre / free-space
Gaussian (fibre-to-chip coupling efficiency), the fundamental of one guide into
that of another (a butt-joint / taper-step mismatch loss), or any two solved
cross-sections. It reports BOTH the impedance-aware power coupling (the two-term
``E``×``H`` Poynting overlap, the same physics as the field-to-mode kernel above)
and the simpler Hermitian field overlap — see :class:`ModeOverlap`.

Dependency-light: numpy + the xarray DataArrays the rest of PhotonHub already
produces. No matplotlib, no engine calls.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from ._constants import C0, ETA0  # noqa: F401  (ETA0 re-exported via __all__)
from .modes import Mode

__all__ = [
    "ETA0",
    "ModeBank",
    "ModeOverlap",
    "mode_amplitude",
    "mode_transmission",
    "mode_decomposition",
    "mode_overlap",
    "mode_overlap_matrix",
    "gaussian_mode",
    "resample_profile",
    "modal_fields",
    "vector_modal_fields",
]

#: A multi-mode bank handed to :func:`mode_decomposition`. Either the
#: per-frequency form ``{freq_hz: {mode_index: Mode}}`` (each plane frequency is
#: projected onto the mode solved AT that frequency — the accurate, dispersive
#: case) or the frozen form ``{mode_index: Mode}`` (one mode per index, projected
#: onto every plane frequency). ``Mode`` here is a scalar :class:`Mode` or a
#: full-vector ``VectorMode`` (the overlap kernel handles both).
ModeBank = Union[Mapping[float, Mapping[int, Any]], Mapping[int, Any]]

_QUANTITIES = ("transmission", "power", "amplitude")

# ETA0 (vacuum wave impedance, ohms) and C0 (free-space speed of light, m/s —
# maps a monitor frequency to a wavelength for the longitudinal Yee de-stagger
# phase beta = 2*pi*n_eff/lambda) come from the shared plugins._constants
# (values matching the engine); ETA0 stays re-exported here via __all__.

Axis = Literal["x", "y", "z"]
Direction = Literal["+", "-"]

# For a propagation axis, the (transverse_axis_1, transverse_axis_2) such that
# axis_1 x axis_2 = +propagation_axis (right-handed). z_hat x t1 = t2.
_TRANSVERSE: Dict[str, Tuple[str, str]] = {
    "x": ("y", "z"),
    "y": ("z", "x"),
    "z": ("x", "y"),
}


def _cell_widths(coords: np.ndarray) -> np.ndarray:
    """Per-sample cell widths for a 1-D set of (possibly non-uniform) sample
    coordinates, via centered differences with half-cells at the ends. The sum
    equals the span plus one mean end-cell, i.e. a midpoint quadrature weight.

    For a single sample (a degenerate 1-cell transverse extent) the width is 1.0
    so the "integral" reduces to that sample's value (a line/point monitor)."""
    c = np.asarray(coords, dtype=np.float64)
    n = c.size
    if n == 1:
        return np.array([1.0])
    edges = np.empty(n + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return np.abs(np.diff(edges))


def _colocate_to_node(a: np.ndarray, axis: int) -> np.ndarray:
    """Average a +½-cell Yee-staggered field component onto the cell NODE along
    ``axis`` (``node[j] = ½(a[j-1] + a[j])``; ``a[-1] ≡ 0`` since a guided mode is
    ~0 at the transverse boundary).

    The engine's DFT monitor emits each component at its own Yee node in
    *cell-index* space (``grid.h`` ``yee_offset``: E_t1 is +½ in t1, E_t2 +½ in
    t2, H_t1 +½ in t2, H_t2 +½ in t1), so the recorded E and H tangential
    components are physically staggered by half a cell. Combining them in the
    overlap cross-products without first interpolating each to a COMMON point is a
    FIRST-ORDER error; co-locating restores SECOND-ORDER accuracy (Oskooi &
    Johnson, *Comp. Phys. Comm.* 181, 687 (2010); MEEP issues #1470/#1773). This
    is what Lumerical (monitor spatial-interpolation, default "nearest mesh cell")
    and Tidy3D (``ModeMonitor(colocate=True)``, the default) do before the
    two-term mode overlap. The collocated FDE mode needs no shift."""
    prev = np.roll(a, 1, axis=axis)
    idx = [slice(None)] * a.ndim
    idx[axis] = 0
    prev[tuple(idx)] = 0.0
    return 0.5 * (prev + a)


def resample_profile(
    field: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
) -> np.ndarray:
    """Separable bilinear resample of ``field[iy, ix]`` (defined on the centered
    1-D grids ``src_x``/``src_y``) onto the destination coordinates
    ``dst_x``/``dst_y``, zero-filled outside the source window.

    Generalizes ``benchmarks/waveguide/run_waveguide.py:_resample`` — numpy-only
    (two passes of :func:`numpy.interp`, x then y). Returns a ``(dst_y.size,
    dst_x.size)`` array indexed ``[iy, ix]``."""
    field = np.asarray(field, dtype=np.float64)
    src_x = np.asarray(src_x, dtype=np.float64)
    src_y = np.asarray(src_y, dtype=np.float64)
    dst_x = np.asarray(dst_x, dtype=np.float64)
    dst_y = np.asarray(dst_y, dtype=np.float64)

    tmp = np.empty((field.shape[0], dst_x.size))
    for j in range(field.shape[0]):
        tmp[j] = np.interp(dst_x, src_x, field[j], left=0.0, right=0.0)
    out = np.empty((dst_y.size, dst_x.size))
    for i in range(dst_x.size):
        out[:, i] = np.interp(dst_y, src_y, tmp[:, i], left=0.0, right=0.0)
    return out


def _resample_cubic(
    field: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
) -> Optional[np.ndarray]:
    """Cubic (bicubic spline) analogue of :func:`resample_profile`, zero-filled
    outside the source window. Returns ``None`` if scipy is unavailable or the
    source has too few samples for the requested order, so the caller can fall
    back to the bilinear path.

    Cubic interpolation of a smooth (well-resolved) mode profile reduces the
    cross-grid resampling error by ~10³–10⁴× vs bilinear (a guided/Gaussian mode is
    smooth at the grid scale), which is why the mode⇄mode overlap uses it when the
    two modes live on different grids — see :func:`mode_overlap`'s ``interp``."""
    try:
        from scipy.interpolate import RectBivariateSpline
    except Exception:  # pragma: no cover - scipy missing ⇒ caller uses bilinear
        return None
    field = np.asarray(field, dtype=np.float64)
    sx = np.asarray(src_x, dtype=np.float64)
    sy = np.asarray(src_y, dtype=np.float64)
    dx = np.asarray(dst_x, dtype=np.float64)
    dy = np.asarray(dst_y, dtype=np.float64)
    # spline degree per axis (RectBivariateSpline needs k < n along that axis).
    kx = min(3, sy.size - 1)        # rows (y)
    ky = min(3, sx.size - 1)        # cols (x)
    if kx < 1 or ky < 1:
        return None
    spline = RectBivariateSpline(sy, sx, field, kx=kx, ky=ky)
    out = spline(dy, dx)            # [i_dy, i_dx]
    # Zero outside the source window (match resample_profile's left/right=0).
    out[(dy < sy[0]) | (dy > sy[-1]), :] = 0.0
    out[:, (dx < sx[0]) | (dx > sx[-1])] = 0.0
    return out


def _resample_real(
    field: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:
    """Resample a real ``field[iy, ix]`` onto ``(dst_x, dst_y)``. ``order=1`` is the
    bilinear :func:`resample_profile` (unchanged — the field-to-mode path); higher
    ``order`` uses the bicubic :func:`_resample_cubic` (the mode⇄mode path), with a
    fast exact return when the destination grid coincides with the source (so a
    same-grid overlap does no interpolation at all) and a bilinear fallback when
    scipy is absent."""
    if order < 3:
        return resample_profile(field, src_x, src_y, dst_x, dst_y)
    sx = np.asarray(src_x, dtype=np.float64)
    sy = np.asarray(src_y, dtype=np.float64)
    dx = np.asarray(dst_x, dtype=np.float64)
    dy = np.asarray(dst_y, dtype=np.float64)
    if (dx.shape == sx.shape and dy.shape == sy.shape
            and np.array_equal(dx, sx) and np.array_equal(dy, sy)):
        return np.asarray(field, dtype=np.float64).copy()   # coincident ⇒ exact
    cubic = _resample_cubic(field, sx, sy, dx, dy)
    return cubic if cubic is not None else resample_profile(field, sx, sy, dx, dy)


def modal_fields(
    mode: Mode,
    t1_um: np.ndarray,
    t2_um: np.ndarray,
    *,
    axis: Axis,
    direction: Direction = "+",
    n_eff: Optional[float] = None,
    center_um: Tuple[float, float] = (0.0, 0.0),
    thickness_axis: Optional[Axis] = None,
    interp_order: int = 1,
) -> Dict[str, np.ndarray]:
    """Assemble the scalar-limit modal transverse fields on a monitor plane.

    The mode's scalar profile is resampled onto the plane's transverse grid
    ``(t1_um, t2_um)`` (the two in-plane axes for ``axis``, in their natural
    Yee order — see :func:`mode_transmission`). The major transverse ``E`` carries
    the whole profile, the minor transverse ``E`` is zero (scalar limit), and the
    transverse ``H`` is ``(n_eff/eta0) * (z_hat x e_mode)``; ``direction="-"``
    flips ``H`` to select the backward mode.

    Parameters
    ----------
    mode:
        The frozen FDE :class:`~photonhub.plugins.modes.Mode`. Its ``.field`` is
        the major transverse-E component (``Ex`` for TE, ``Ey`` for TM).
    t1_um, t2_um:
        The plane's two transverse coordinate axes (microns), in the order
        ``_TRANSVERSE[axis]`` (so ``t1 x t2 = +axis``).
    axis:
        Propagation axis ``"x"``/``"y"``/``"z"``.
    direction:
        ``"+"`` forward (default) or ``"-"`` backward.
    n_eff:
        Optional override for the modal index used in the ``H`` reconstruction
        (per-frequency dispersion). Defaults to ``mode.n_eff``.
    center_um:
        ``(t1, t2)`` location of the waveguide axis in the plane's coordinate
        frame (microns). The mode profile (centered at its own origin) is shifted
        here before resampling. Defaults to the plane origin.
    thickness_axis:
        The simulation axis along the guide's slab thickness (the mode's HEIGHT
        / ``dl_y`` direction); must be one of the two transverse axes for
        ``axis``. The mode's WIDTH (``dl_x``) is mapped to the OTHER transverse
        axis. ``None`` (default) keeps the legacy mapping ``width->t1,
        height->t2`` — correct only when the thickness lies on the second
        transverse axis (e.g. x-propagation with a z-normal slab). For
        y-propagation of a z-normal slab the thickness is the FIRST transverse
        axis, so pass ``thickness_axis="z"`` to orient the mode correctly (else
        the profile comes out rotated 90 degrees).

    Returns
    -------
    dict
        Keys ``"e1"``, ``"e2"`` (transverse-E components along ``t1``/``t2``),
        ``"h1"``, ``"h2"`` (transverse-H), each a ``(t2.size, t1.size)`` array.
        The major-E component is whichever of ``t1``/``t2`` is the mode's major
        axis; the other E component is all zeros.
    """
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    a1, a2 = _TRANSVERSE[axis]  # the two transverse axis NAMES (a1 x a2 = +axis)
    if thickness_axis is None:
        thickness_axis = a2  # legacy: slab thickness on the 2nd transverse axis
    if thickness_axis not in (a1, a2):
        raise ValueError(
            f"thickness_axis {thickness_axis!r} must be a transverse axis "
            f"({a1!r} or {a2!r}) for propagation axis {axis!r}")
    # Physically the mode's WIDTH (dl_x) lies on the in-plane transverse axis and
    # its HEIGHT (dl_y) on the slab-normal (thickness) axis. Map width -> the
    # non-thickness axis, height -> the thickness axis (NOT the fixed x->t1,
    # y->t2, which is right only when the thickness happens to be a2).
    width_axis = a1 if thickness_axis == a2 else a2

    neff = float(mode.n_eff if n_eff is None else n_eff)

    # Mode's own centered real-space coords (microns), matching field_dataarray.
    # Graded-window modes carry their true node ladders — prefer them.
    ny, nx = mode.field.shape
    mode_xc = getattr(mode, "x_coords_um", None)
    if mode_xc is not None:
        w_coords = np.asarray(mode_xc, dtype=np.float64)
        h_coords = np.asarray(mode.y_coords_um, dtype=np.float64)
    else:
        w_coords = (np.arange(nx) - (nx - 1) / 2.0) * mode.dl_x_um  # width
        h_coords = (np.arange(ny) - (ny - 1) / 2.0) * mode.dl_y_um  # height
    t1c = np.asarray(t1_um, dtype=np.float64)
    t2c = np.asarray(t2_um, dtype=np.float64)

    if width_axis == a1:  # width -> t1, height -> t2 (legacy orientation)
        wc = w_coords + center_um[0]
        hc = h_coords + center_um[1]
        profile = _resample_real(mode.field, wc, hc, t1c, t2c,
                                 order=interp_order)  # [i_t2, i_t1]
    else:  # width -> t2, height -> t1 (e.g. y-propagation, thickness on a1)
        wc = w_coords + center_um[1]
        hc = h_coords + center_um[0]
        # width(mode-x)->t2, height(mode-y)->t1; transpose to [i_t2, i_t1].
        profile = _resample_real(mode.field, wc, hc, t2c, t1c,
                                 order=interp_order).T

    # Major transverse-E axis: TE's major (mode Ex) lies along the WIDTH axis,
    # TM's major (mode Ey) along the HEIGHT = thickness axis. In the scalar limit
    # the minor transverse E is zero.
    major_axis = width_axis if mode.polarization != "TM" else thickness_axis
    major_is_t1 = major_axis == a1
    e1 = profile if major_is_t1 else np.zeros_like(profile)
    e2 = np.zeros_like(profile) if major_is_t1 else profile

    # h = (n_eff/eta0) * (z_hat x e), z_hat = +axis. With e = (e1, e2) in the
    # right-handed (t1, t2) frame: z_hat x (e1 t1_hat + e2 t2_hat)
    #   = e1 (z_hat x t1_hat) + e2 (z_hat x t2_hat) = e1 t2_hat - e2 t1_hat.
    sign = 1.0 if direction == "+" else -1.0
    scale = sign * neff / ETA0
    h1 = -scale * e2
    h2 = scale * e1
    return {"e1": e1, "e2": e2, "h1": h1, "h2": h2}


def _resample_complex(
    field: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:
    """Like :func:`resample_profile` but for a complex ``field`` — real and
    imaginary parts are resampled independently (the interpolation is linear in the
    data, so this preserves the per-point complex value exactly on the source grid
    and interpolates the relative phase between components). ``order`` selects
    bilinear (1) or bicubic (3); see :func:`_resample_real`."""
    field = np.asarray(field)
    re = _resample_real(field.real, src_x, src_y, dst_x, dst_y, order=order)
    if np.iscomplexobj(field) and np.any(field.imag):
        im = _resample_real(field.imag, src_x, src_y, dst_x, dst_y, order=order)
        return re + 1j * im
    return re.astype(np.complex128)


def vector_modal_fields(
    mode,
    t1_um: np.ndarray,
    t2_um: np.ndarray,
    *,
    axis: Axis,
    direction: Direction = "+",
    center_um: Tuple[float, float] = (0.0, 0.0),
    thickness_axis: Optional[Axis] = None,
    interp_order: int = 1,
) -> Dict[str, np.ndarray]:
    """Assemble the FULL-VECTOR transverse fields of a
    :class:`~photonhub.plugins.vector_modes.VectorMode` on a monitor/injection
    plane — the full-vector analogue of :func:`modal_fields`.

    Unlike :func:`modal_fields` (which carries one scalar profile in the major-E
    component and reconstructs ``H`` in the scalar limit), this resamples the
    mode's *actual* transverse-E pair ``(ex, ey)`` AND transverse-H pair
    ``(hx, hy)`` onto the plane, preserving their true component RATIO and
    relative phase. The mode's own width/height axes are mapped to the plane's
    ``(t1, t2)`` exactly as :func:`modal_fields` does (via ``thickness_axis``),
    and ``direction="-"`` flips ``H`` to select the backward mode.

    Parameters mirror :func:`modal_fields`. ``mode`` is a ``VectorMode`` (the
    six complex ``(ny, nx)`` component arrays ``ex, ey, ez, hx, hy, hz`` indexed
    ``[iy, ix]``). Returns a dict with ``"e1"``, ``"e2"``, ``"h1"``, ``"h2"``
    (transverse components along ``t1``/``t2``), each a ``(t2.size, t1.size)``
    complex array — same layout as :func:`modal_fields`.
    """
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    a1, a2 = _TRANSVERSE[axis]
    if thickness_axis is None:
        thickness_axis = a2  # legacy: slab thickness on the 2nd transverse axis
    if thickness_axis not in (a1, a2):
        raise ValueError(
            f"thickness_axis {thickness_axis!r} must be a transverse axis "
            f"({a1!r} or {a2!r}) for propagation axis {axis!r}")
    width_axis = a1 if thickness_axis == a2 else a2

    # The mode's own centered real-space coords (microns) — width along mode-x
    # (dl_x_um, carried by ex) and height along mode-y (dl_y_um, carried by ey).
    # A mode carrying window-placement metadata (``center_offset_um``: the
    # actual raster-window center minus the requested center — set by the
    # grid-snapped cross-section solvers) is shifted by it, so it lands where
    # its eps raster truly was instead of assuming a centered array (the snap
    # displaces the window by up to ~a cell).
    # A GRADED-window mode carries its true node ladders (x_coords_um /
    # y_coords_um, centre-relative — §15 nonuniform solves); prefer them over
    # reconstructing a uniform ladder from the scalar pitch, which would
    # misplace every node of a nonuniform raster.
    mode_xc = getattr(mode, "x_coords_um", None)
    ny, nx = mode.ex.shape
    if mode_xc is not None:
        w_coords = np.asarray(mode_xc, dtype=np.float64)
        h_coords = np.asarray(mode.y_coords_um, dtype=np.float64)
    else:
        off_w, off_h = getattr(mode, "center_offset_um", None) or (0.0, 0.0)
        w_coords = (np.arange(nx) - (nx - 1) / 2.0) * mode.dl_x_um + off_w
        h_coords = (np.arange(ny) - (ny - 1) / 2.0) * mode.dl_y_um + off_h
    t1c = np.asarray(t1_um, dtype=np.float64)
    t2c = np.asarray(t2_um, dtype=np.float64)

    def to_plane(mode_field: np.ndarray) -> np.ndarray:
        """Resample a mode-frame [iy, ix] field onto the plane [i_t2, i_t1]."""
        if width_axis == a1:  # mode-x -> t1, mode-y -> t2 (legacy)
            wc = w_coords + center_um[0]
            hc = h_coords + center_um[1]
            return _resample_complex(mode_field, wc, hc, t1c, t2c,
                                     order=interp_order)
        # mode-x -> t2, mode-y -> t1; resample then transpose to [i_t2, i_t1].
        wc = w_coords + center_um[1]
        hc = h_coords + center_um[0]
        return _resample_complex(mode_field, wc, hc, t2c, t1c,
                                 order=interp_order).T

    # The mode's x-field (ex/hx) lies along the WIDTH axis, the y-field (ey/hy)
    # along the HEIGHT (= thickness) axis. Route each to t1/t2 accordingly.
    ex_p, ey_p = to_plane(mode.ex), to_plane(mode.ey)
    hx_p, hy_p = to_plane(mode.hx), to_plane(mode.hy)
    if width_axis == a1:  # width(mode-x) -> t1, height(mode-y) -> t2
        e1, e2, h1, h2 = ex_p, ey_p, hx_p, hy_p
    else:                 # width(mode-x) -> t2, height(mode-y) -> t1
        # This (t1, t2) <- (mode-y, mode-x) assignment REFLECTS the mode frame
        # (swaps two axes), flipping its handedness. The mode's forward
        # H = (n/η)·ζ̂×E is tied to the mode-frame handedness, so leaving H
        # unnegated makes the assembled plane Poynting e1·h2*−e2·h1* come out
        # along −axis: the returned "forward" (direction="+") mode is actually the
        # BACKWARD traveler (verified: P·n̂ < 0 on any swap-branch plane, e.g. the
        # standard axis="y"/thickness_axis="z" z-slab port). Negate the transverse
        # H to restore ĥ-consistency so direction="+" carries +axis power.
        e1, e2, h1, h2 = ey_p, ex_p, -hy_p, -hx_p

    if direction == "-":  # backward mode: H -> -H
        h1, h2 = -h1, -h2
    return {"e1": e1, "e2": e2, "h1": h1, "h2": h2}


def _plane_component(
    fields: Mapping[str, xr.DataArray],
    name: str,
    freq_hz: Optional[float],
    t1: str,
    t2: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull a tangential component DataArray (e.g. ``"Ey"``), drop the singleton
    normal axis and any freq/component dims, and return it as a 2-D complex
    ``[i_t2, i_t1]`` array plus its ``(t1_coords, t2_coords)`` in microns."""
    da = fields[name]
    if "component" in da.dims:
        da = da.sel(component=name) if name in list(da.coords.get("component", [])) \
            else da.squeeze("component", drop=True)
    if "f" in da.dims:
        da = da.sel(f=freq_hz, method="nearest") if freq_hz is not None \
            else da.isel(f=0)
    # Drop the (singleton) normal axis and any other length-1 dims, keeping t1/t2.
    da = da.squeeze(drop=True)
    if set(da.dims) != {t1, t2}:
        raise ValueError(
            f"component {name!r}: after reduction dims are {tuple(da.dims)}, "
            f"expected the two transverse axes {{{t1!r}, {t2!r}}}")
    # Orient as [i_t2, i_t1] so it matches modal_fields' (t2.size, t1.size).
    da = da.transpose(t2, t1)
    vals = np.asarray(da.values)
    c1 = np.asarray(da.coords[t1].values, dtype=np.float64)
    c2 = np.asarray(da.coords[t2].values, dtype=np.float64)
    return vals, c1, c2


def _overlap_terms(
    sim_plane_fields: Mapping[str, xr.DataArray],
    mode: Mode,
    *,
    axis: Axis,
    direction: Direction = "+",
    n_eff: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[Axis] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
    colocate: bool = True,
    destagger_dl: Optional[float] = None,
    fold_low: Tuple[bool, bool] = (False, False),
) -> Dict[float, Tuple[complex, float]]:
    """Per-frequency directional-power overlap terms ``{f: (a_pm, P_mode)}`` for a
    recorded plane projected onto ``mode`` — the shared kernel behind
    :func:`mode_amplitude` (``c = a_pm/P_mode``), :func:`mode_transmission`
    (``|c|² = |a_pm|²/P_mode²``) and the power readout (``|a_pm|²/P_mode``).
    ``a_pm`` is the unnormalized complex coefficient, ``P_mode`` the mode's own
    power on the plane. See :func:`mode_transmission` for the argument schema.

    ``destagger_dl`` (the grid spacing along the propagation/normal axis, microns)
    enables the **longitudinal Yee de-stagger** — see :func:`mode_transmission`.

    ``fold_low`` = (t1 folded, t2 folded): the in-plane axis' MIN face is a §20
    symmetry plane (PMC/PEC fold) the plane's coordinate ladder starts ON. The
    half-domain quadrature then weights an ON-PLANE (node-registered) sample row
    by half a cell — its cell covers only ``[0, dl/2]`` of the modeled half —
    where :func:`_cell_widths`' end rule would extend it ``dl/2`` into the
    MIRROR half and over-count fold-antinode modes (an even mode's ~+dl/(2 w_eff)
    power inflation; a fold-node mode is untouched — the parity-asymmetric bias
    that under-read T on folded readings). Products whose components sit ½-cell
    off the fold axis (grid.h Yee offsets) have no on-plane row and keep full
    weights, so the halving is applied per product term. Default (False, False)
    keeps every unfolded reading on the exact historical code path."""
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    t1, t2 = _TRANSVERSE[axis]
    e1_name, e2_name = f"E{t1}", f"E{t2}"
    h1_name, h2_name = f"H{t1}", f"H{t2}"
    for key in (e1_name, e2_name, h1_name, h2_name):
        if key not in sim_plane_fields:
            raise ValueError(
                f"sim_plane_fields missing {key!r}; for axis={axis!r} need the "
                f"tangential components {e1_name!r},{e2_name!r},{h1_name!r},"
                f"{h2_name!r}")

    # Determine the set of frequencies from the first E component.
    ref = sim_plane_fields[e1_name]
    if "f" in getattr(ref, "dims", ()):  # DataArray with a freq axis
        freqs = [float(f) for f in np.asarray(ref.coords["f"].values)]
    else:
        freqs = [None]  # plane carries a single (frequencyless) snapshot

    # Grid-consistency: a Yee-staggered reference mode (``solve_yee_mode``) carries
    # the SAME intra-cell Yee offsets as the recorded FDTD field, so its component
    # arrays already register 1:1 with the raw sim components at their native Yee
    # locations. Co-locating the sim field to the node would then DOUBLE the
    # registration — a ½-cell transverse mismatch against the un-shifted mode. For
    # such a mode the transverse co-location is skipped and the overlap is done
    # natively (each component × the same-named sim component at its Yee point);
    # the orthogonal LONGITUDINAL (propagation-axis) de-stagger below is unchanged
    # and still applied. A node-collocated FLM mode (default solver) keeps
    # ``yee_staggered`` False, so the sim field is co-located to match it as before.
    # The flag is resolved PER FREQUENCY from the mode actually projected at that
    # frequency, so a mixed ``modes_by_freq`` bank (Yee + FLM entries) co-locates
    # exactly the frequencies whose reference mode needs it — one bank entry must
    # not decide for all the others.
    out: Dict[float, Tuple[complex, float]] = {}
    for f in freqs:
        Es1, c1, c2 = _plane_component(sim_plane_fields, e1_name, f, t1, t2)
        Es2, _, _ = _plane_component(sim_plane_fields, e2_name, f, t1, t2)
        Hs1, _, _ = _plane_component(sim_plane_fields, h1_name, f, t1, t2)
        Hs2, _, _ = _plane_component(sim_plane_fields, h2_name, f, t1, t2)

        use_mode, use_neff = mode, n_eff
        if modes_by_freq and f is not None:
            # per-λ: project this frequency onto its OWN solved mode (profile +
            # n_eff), matching Tidy3D's per-frequency ModeMonitor decomposition.
            key = min(modes_by_freq, key=lambda k: abs(k - f))
            use_mode, use_neff = modes_by_freq[key], None
        yee_mode = getattr(use_mode, "yee_staggered", False)

        if colocate and not yee_mode:
            # Yee co-location (§ _colocate_to_node): shift each staggered sim
            # component to the cell node so the two-term overlap is 2nd-order.
            # Arrays are [i_t2, i_t1] (t1 = last axis, t2 = axis 0); offsets from
            # grid.h yee_offset: E_t1 +½t1, E_t2 +½t2, H_t1 +½t2, H_t2 +½t1.
            Es1 = _colocate_to_node(Es1, -1)
            Es2 = _colocate_to_node(Es2, 0)
            Hs1 = _colocate_to_node(Hs1, 0)
            Hs2 = _colocate_to_node(Hs2, -1)

        # Area element from the plane's real (possibly graded) coord spacings.
        w1 = _cell_widths(c1)            # along t1
        w2 = _cell_widths(c2)            # along t2
        # §20 folded-domain quadrature (see the docstring): halve the first
        # sample's weight on a folded in-plane axis whose ladder starts ON the
        # symmetry plane — for NODE-registered products only. Guard on the
        # ladder actually reaching the plane (a shrunk/offset window whose
        # first node sits a cell up needs no correction).
        def _fold_halved(w: np.ndarray, c: np.ndarray, folded: bool) -> np.ndarray:
            if not folded or w.size < 2 or not (
                    abs(c[0]) <= 0.25 * abs(c[1] - c[0])):
                return w
            wh = w.copy()
            wh[0] *= 0.5
            return wh
        w1n = _fold_halved(w1, c1, fold_low[0])  # node-registered along t1
        w2n = _fold_halved(w2, c2, fold_low[1])  # node-registered along t2
        if w1n is w1 and w2n is w2:
            # No fold correction engaged: single shared area element, and the
            # combined-integrand path below is BIT-identical to the historical
            # readout.
            dA_1 = dA_2 = np.outer(w2, w1)   # [i_t2, i_t1], matches field arrays
        elif yee_mode:
            # Native Yee registration (grid.h yee_offset): the E_t1·H_t2-type
            # products sit +½ ALONG t1 / node along t2 (no sample on a t1 fold);
            # the E_t2·H_t1-type products sit node along t1 / +½ along t2.
            dA_1 = np.outer(w2n, w1)         # E_t1 x H_t2 products
            dA_2 = np.outer(w2, w1n)         # E_t2 x H_t1 products
        else:
            # Node-registered path: either the colocation above shifted every
            # component to the cell node, or (colocate=False) the caller supplied
            # already-aligned fields — both product families then carry an
            # on-plane row on a folded axis.
            dA_1 = dA_2 = np.outer(w2n, w1n)

        cen = center_um
        if cen is None:
            cen = (float(np.mean(c1)), float(np.mean(c2)))
        # n_eff for the de-stagger phase: the override if given, else the mode's.
        neff_ds = float(use_neff) if use_neff is not None \
            else float(getattr(use_mode, "n_eff", 0.0))
        if hasattr(use_mode, "hx"):
            # Full-vector mode: project with the mode's TRUE transverse H, not the
            # scalar-limit (n_eff/eta0)·(z_hat x e). This is the grid-consistent
            # "smooth readout" path — see benchmarks/tidy3d/SMOOTH_CONVERGENCE_PLAN.md
            # (issue #34). n_eff is intrinsic to the vector mode, so use_neff is
            # not applicable here.
            m = vector_modal_fields(use_mode, c1, c2, axis=axis,
                                    direction=direction, center_um=cen,
                                    thickness_axis=thickness_axis)
        else:
            m = modal_fields(use_mode, c1, c2, axis=axis, direction=direction,
                             n_eff=use_neff, center_um=cen,
                             thickness_axis=thickness_axis)
        e1, e2, h1, h2 = m["e1"], m["e2"], m["h1"], m["h2"]

        # n_hat-component of a cross product of transverse vectors
        # (a1, a2) x (b1, b2) = (a1 b2 - a2 b1) n_hat.
        # a_pm = (1/4) integral [ E_sim x h_mode* + e_mode* x H_sim ] . n_hat dA
        if dA_1 is dA_2:
            dA = dA_1
            cross_eh = (Es1 * np.conj(h2) - Es2 * np.conj(h1))  # E_sim x h*
            cross_he = (np.conj(e1) * Hs2 - np.conj(e2) * Hs1)  # e* x H_sim
            I_Eh = np.sum(cross_eh * dA)                        # integral E x h*
            I_eH = np.sum(cross_he * dA)                        # integral e* x H

            # P_mode = (1/2) integral Re( e_mode x h_mode* ) . n_hat dA.
            p_density = np.real(e1 * np.conj(h2) - e2 * np.conj(h1))
            p_mode = 0.5 * np.sum(p_density * dA)
        else:
            # Folded per-term quadrature: the two product families carry
            # different fold-row registrations, so each takes its own dA.
            I_Eh = (np.sum(Es1 * np.conj(h2) * dA_1)
                    - np.sum(Es2 * np.conj(h1) * dA_2))
            I_eH = (np.sum(np.conj(e1) * Hs2 * dA_1)
                    - np.sum(np.conj(e2) * Hs1 * dA_2))
            p_mode = 0.5 * (np.sum(np.real(e1 * np.conj(h2)) * dA_1)
                            - np.sum(np.real(e2 * np.conj(h1)) * dA_2))
        a_pm = 0.25 * (I_Eh + I_eH)

        if p_mode == 0.0:
            raise ValueError(
                "P_mode is zero — the resampled mode has no power on this plane "
                "(check the mode window vs the plane extent and center_um).")

        if destagger_dl and f is not None:
            # Longitudinal Yee DE-STAGGER. The engine records E at the cell node
            # but H half a cell along the PROPAGATION axis (the monitor normal), so
            # the two-term overlap carries a phase phi = beta*dl/2 (beta =
            # 2*pi*n_eff/lambda). That phase both under-reads a clean co-propagating
            # mode by cos(phi/2) AND mixes a fraction sin(phi/2) of the COUNTER-
            # propagating wave into the reading — the standing-wave "ripple" at a
            # plane in front of a reflecting junction (the transverse colocation
            # above does NOT fix this; it is the normal-axis stagger). With the
            # mode self-norm N = 2*P_mode, the two recorded overlaps are
            #   I_Eh/N = a + b ,   I_eH/N = a e^{i phi} - b e^{-i phi}
            # (a = co-, b = counter-propagating amplitude); solving the 2x2 for the
            # clean co-propagating amplitude gives
            #   a = ( I_eH/N + (I_Eh/N) e^{-i phi} ) / (2 cos phi) .
            # A clean forward wave then reads a exactly (self-overlap 1 preserved)
            # and a pure reflection reads ~0 forward. Default OFF (synthetic,
            # already-colocated test fields have no such stagger).
            lam_um = C0 / f * 1e6
            phi = (2.0 * np.pi * neff_ds / lam_um) * (0.5 * destagger_dl)
            if direction == "-":
                phi = -phi
            if abs(np.cos(phi)) < 0.2:
                # The 2x2 solve divides by 2*cos(phi): at phi -> pi/2 (i.e.
                # dl -> lambda/(2 n_eff), the modal Nyquist grid) the forward/
                # backward system is singular and the "correction" would just
                # amplify noise. Refuse loudly rather than return garbage.
                raise ValueError(
                    f"longitudinal de-stagger is singular here: phi = "
                    f"beta*dl/2 = {phi:.3f} rad (|cos phi| = "
                    f"{abs(np.cos(phi)):.3f} < 0.2) at f={f:.4g} Hz, "
                    f"n_eff={neff_ds:.4g}, dl={destagger_dl:.4g} um — the grid "
                    "is near lambda/(2 n_eff) per cell. Refine the grid, or "
                    "pass destagger_dl=None to read without the correction.")
            N = 2.0 * p_mode
            a_ds = ((I_eH / N) + (I_Eh / N) * np.exp(-1j * phi)) / (2.0 * np.cos(phi))
            a_pm = a_ds * p_mode  # |a_pm|^2/P_mode^2 = |a_ds|^2 (T) downstream
        elif destagger_dl and f is None:
            warnings.warn(
                "destagger_dl was given but this plane carries no frequency "
                "axis (needed to form beta = 2*pi*n_eff/lambda) — the "
                "longitudinal de-stagger is being SKIPPED for this reading.",
                UserWarning, stacklevel=2)

        out[f if f is not None else 0.0] = (complex(a_pm), float(p_mode))
    return out


def mode_amplitude(
    sim_plane_fields: Mapping[str, xr.DataArray],
    mode: Mode,
    *,
    axis: Axis,
    direction: Direction = "+",
    n_eff: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[Axis] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
    colocate: bool = True,
    destagger_dl: Optional[float] = None,
    fold_low: Tuple[bool, bool] = (False, False),
) -> Dict[float, complex]:
    """Mode-resolved **complex** modal amplitude ``c(f)`` of a recorded plane.

    This is the COMPLEX coefficient that :func:`mode_transmission` squares to a
    power. Per frequency on the plane it computes the directional-power overlap

        a_pm   = (1/4) integral [ E_sim x h_mode* + e_mode* x H_sim ] . n_hat dA
        P_mode = (1/2) integral Re( e_mode x h_mode* ) . n_hat dA
        c      = a_pm / P_mode

    in the ``e^{-i omega t}`` convention, with the scalar-limit modal ``H``
    (see the module docstring). The normalization by ``P_mode`` makes a clean
    single-mode forward self-overlap read ``c == 1`` exactly (so ``|c|^2 == T``,
    the power transmission). Crucially ``c`` retains the **phase** of the modal
    projection — in the engine's ``e^{-i omega t}`` convention a forward wave's
    phasor advances as ``e^{+i beta z}``, so ``c`` picks up ``e^{+i beta L}``
    along a straight guide — which is exactly what an S-matrix assembler needs
    (``S_ij = b_i / a_j``).

    The amplitude is *directional*: ``direction="+"`` projects onto the forward
    mode, ``direction="-"`` onto the backward one. A pure forward wave reads a
    near-unit forward ``c`` and a near-zero backward ``c``, and vice versa — this
    is what separates incident (forward) from scattered (backward) at a port.

    Parameters mirror :func:`mode_transmission`; see it for the
    ``sim_plane_fields`` schema and the per-argument documentation.

    Returns
    -------
    dict[float, complex]
        ``{freq_hz: c}`` for every frequency on the plane (frequencyless planes
        key on ``0.0``). ``c`` is the normalized complex modal amplitude.
    """
    return {
        f: complex(a_pm / p_mode)
        for f, (a_pm, p_mode) in _overlap_terms(
            sim_plane_fields, mode, axis=axis, direction=direction, n_eff=n_eff,
            center_um=center_um, thickness_axis=thickness_axis,
            modes_by_freq=modes_by_freq, colocate=colocate,
            destagger_dl=destagger_dl, fold_low=fold_low).items()
    }


def mode_transmission(
    sim_plane_fields: Mapping[str, xr.DataArray],
    mode: Mode,
    *,
    axis: Axis,
    direction: Direction = "+",
    n_eff: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[Axis] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
    power: bool = False,
    colocate: bool = True,
    destagger_dl: Optional[float] = None,
    fold_low: Tuple[bool, bool] = (False, False),
) -> Dict[float, float]:
    """Mode-resolved power transmission ``T(f)`` of a recorded plane onto ``mode``.

    Computes, per frequency on the plane, the directional-power overlap

        a_pm   = (1/4) integral [ E_sim x h_mode* + e_mode* x H_sim ] . n_hat dA
        T      = |a_pm|^2 / P_mode^2 ,           (power=False, default)
        P_mode = (1/2) integral Re( e_mode x h_mode* ) . n_hat dA

    in the ``e^{-i omega t}`` convention, with the scalar-limit modal ``H``
    (see the module docstring; the ``P_mode^2`` denominator — not ``P_mode`` —
    is the squared NORMALISED amplitude ``|c|^2``, so a clean single-mode
    self-overlap reads ``T == 1``).
    ``direction="+"`` returns forward T, ``direction="-"`` backward T.

    ``power=True`` instead returns the actual modal **power**
    ``|a_pm|^2 / |P_mode|`` (= ``|c|^2 * |P_mode|``, always ``>= 0`` — the
    magnitude of ``P_mode`` so a backward ``direction="-"`` reading, whose
    signed flux through the +n_hat plane is negative, is still a power). Use
    this when ratioing two planes whose modes may DIFFER (e.g. a w1→w2 taper):
    ``P_out / P_in`` is then the true power transmission. The bare ``|c|^2`` (power=False) drops each port's ``P_mode``,
    so its ratio is only correct when both ports carry the SAME mode (it cancels);
    for unequal-width ports it is wrong (the historical taper-parity bug).

    This is exactly ``|c|^2`` of the complex amplitude from
    :func:`mode_amplitude` — use that function when you need the phase (e.g. for
    an S-matrix). Behaviour here is unchanged (back-compatible).

    Parameters
    ----------
    sim_plane_fields:
        Mapping from component name to its plane DataArray, supplying the two
        tangential ``E`` and two tangential ``H`` components for ``axis``:

        * ``axis="z"`` -> keys ``"Ex"``, ``"Ey"``, ``"Hx"``, ``"Hy"``;
        * ``axis="x"`` -> keys ``"Ey"``, ``"Ez"``, ``"Hy"``, ``"Hz"``;
        * ``axis="y"`` -> keys ``"Ez"``, ``"Ex"``, ``"Hz"``, ``"Hx"``.

        Each is an xarray ``DataArray`` in µm coords (a single-plane ``field_dft``
        slice: dims like ``('f','component','z','y','x')`` with a singleton normal
        axis are reduced automatically; a plain 2-D ``(t2, t1)`` DataArray also
        works). The two transverse axes are ``_TRANSVERSE[axis]``.
    mode:
        The frozen FDE :class:`~photonhub.plugins.modes.Mode` to project onto.
    axis:
        Propagation axis / plane normal, ``"x"``/``"y"``/``"z"``.
    direction:
        ``"+"`` forward (default) or ``"-"`` backward.
    n_eff:
        Optional override for the modal index in the ``H`` reconstruction.
    center_um:
        ``(t1, t2)`` location of the waveguide axis in the plane's coordinate
        frame (microns). If ``None`` (default) the plane's transverse coordinate
        midpoints are used, i.e. the mode is centered on the monitor.
    thickness_axis:
        Simulation axis along the guide's slab thickness; forwarded to
        :func:`modal_fields` to orient the mode (pass the slab normal, e.g.
        ``"z"``, for any non-x propagation — see that function). ``None`` keeps
        the legacy thickness-on-second-transverse-axis mapping.
    modes_by_freq:
        Optional ``{freq_hz: Mode}`` map. When given, each plane frequency is
        projected onto the mode whose key is nearest that frequency (using that
        mode's own profile *and* ``n_eff``), instead of the single frozen
        ``mode`` — the per-λ mode solve. ``mode`` is still required (used as the
        fallback for any frequencyless plane).
    destagger_dl:
        Grid spacing (microns) along the propagation / monitor-normal axis. When
        given, applies the **longitudinal Yee de-stagger**: the engine records
        ``E`` at the cell node but ``H`` half a cell along the normal, so the
        two-term overlap carries a phase ``phi = beta*dl/2`` (``beta =
        2*pi*n_eff/lambda``) that under-reads a clean mode by ``cos(phi/2)`` and
        leaks ``sin(phi/2)`` of the COUNTER-propagating wave into the reading — a
        ~1% standing-wave ripple at a normalization plane in front of a reflecting
        junction (the transverse :func:`_colocate_to_node` does NOT fix this).
        The correction solves the 2x2 forward/backward system for the clean
        co-propagating amplitude (a clean forward wave still reads ``T=1``; a pure
        reflection reads ``~0`` forward). ``None`` (default) = off, so synthetic
        already-co-located fields and the legacy readout are unchanged. Pass the
        run's uniform grid ``dl`` (e.g. ``scene.dl_um``).

    Returns
    -------
    dict[float, float]
        ``{freq_hz: T}`` for every frequency present on the plane (real, >= 0;
        ``~1`` for a clean single-mode forward field, ``~0`` for the opposite
        direction).
    """
    terms = _overlap_terms(
        sim_plane_fields, mode, axis=axis, direction=direction, n_eff=n_eff,
        center_um=center_um, thickness_axis=thickness_axis,
        modes_by_freq=modes_by_freq, colocate=colocate, destagger_dl=destagger_dl,
        fold_low=fold_low,
    )
    if power:
        # |P_mode|, not the signed P_mode: a backward mode (direction="-")
        # carries a NEGATIVE signed flux through the +n_hat plane, and dividing
        # by it would return a negative "power" (making reflection ratios
        # negative). The modal power carried in the mode's own propagation
        # sense is |a_pm|^2/|P_mode| >= 0 — the same |P| normalization
        # mode_overlap applies to its backward operand.
        return {f: float(np.abs(a_pm) ** 2 / abs(p_mode))
                for f, (a_pm, p_mode) in terms.items()}
    return {f: float(np.abs(a_pm) ** 2 / p_mode ** 2)
            for f, (a_pm, p_mode) in terms.items()}


def _is_mode(obj: Any) -> bool:
    """A mode-like object (scalar :class:`Mode` or full-vector ``VectorMode``):
    carries a transverse-E profile (``.field`` for scalar, ``.ex`` for vector)."""
    return hasattr(obj, "field") or hasattr(obj, "ex")


def _term_to_quantity(a_pm: complex, p_mode: float, quantity: str):
    """Map one ``(a_pm, P_mode)`` overlap term to the requested readout."""
    if quantity == "amplitude":
        return complex(a_pm / p_mode)
    if quantity == "power":
        # |P_mode| so a backward (direction="-") reading is a non-negative
        # power — see mode_transmission's power branch.
        return float(np.abs(a_pm) ** 2 / abs(p_mode))
    # "transmission": squared normalised amplitude |c|^2 (self-overlap == 1).
    return float(np.abs(a_pm) ** 2 / p_mode ** 2)


def mode_decomposition(
    sim_plane_fields: Mapping[str, xr.DataArray],
    mode_bank: ModeBank,
    *,
    axis: Axis,
    direction: Direction = "+",
    quantity: str = "transmission",
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[Axis] = None,
    colocate: bool = True,
    destagger_dl: Optional[float] = None,
    fold_low: Tuple[bool, bool] = (False, False),
) -> Dict[int, Dict[float, Any]]:
    """Multi-mode modal decomposition of a recorded plane — project it onto EACH
    mode in ``mode_bank`` and return the per-mode result keyed ``{mode_index:
    {freq_hz: value}}``.

    This is the multi-mode / multi-frequency generalization of
    :func:`mode_transmission` and :func:`mode_amplitude`, which project onto a
    SINGLE mode at a time. It is the PhotonHub analogue of Tidy3D's
    ``ModeMonitor(mode_spec=ModeSpec(num_modes=N), freqs=[...])``: the recorded
    field on a port plane is decomposed into the guided-mode basis ``[0..N-1]``,
    so you can read how much power leaves in each mode (and at each frequency),
    separate the fundamental from higher-order content, and check that the modal
    powers sum to (≤) the raw flux. Each mode index is an INDEPENDENT directional
    projection (the modes need not be mutually orthogonal under this discretized
    overlap; for a well-resolved guided basis they are, to the readout floor).

    Parameters
    ----------
    sim_plane_fields:
        The four tangential plane components, exactly as :func:`mode_transmission`
        takes them (see that function for the per-axis key schema).
    mode_bank:
        The modes to project onto, in one of two forms (:data:`ModeBank`):

        * **per-frequency** ``{freq_hz: {mode_index: Mode}}`` — each plane
          frequency is projected onto the mode of that index solved AT that
          frequency (true ``H`` + ``n_eff(λ)``), via the same nearest-frequency
          lookup as :func:`mode_transmission`'s ``modes_by_freq``. This is the
          dispersive, accurate case; build it with
          :func:`~photonhub.plugins.mode_devices.solve_mode_bank`. The bank must be
          **rectangular** — the SAME mode indices at every frequency; a ragged
          bank raises (else the nearest-frequency lookup would silently fabricate
          a reading at a frequency missing that index).
        * **frozen** ``{mode_index: Mode}`` — one mode per index, projected onto
          every plane frequency (the band-centre modes).

        Modes may be scalar :class:`Mode` or full-vector ``VectorMode`` (the
        overlap kernel uses the mode's true transverse ``H`` when present, else
        the scalar-limit reconstruction — same rule as the single-mode path).
    direction:
        ``"+"`` forward (default) or ``"-"`` backward — applied to every index.
    quantity:
        ``"transmission"`` (default) → ``|c|² = |a_pm|²/P_mode²`` (real,
        self-overlap 1); ``"power"`` → ``|a_pm|²/P_mode`` (real modal power, the
        quantity to ratio across unequal-mode ports); ``"amplitude"`` → the
        complex normalised amplitude ``c = a_pm/P_mode`` (carries phase, for an
        S-matrix / multimode-port assembler).
    center_um, thickness_axis, colocate:
        Forwarded unchanged to the per-mode overlap (see
        :func:`mode_transmission`).

    Returns
    -------
    dict[int, dict[float, value]]
        ``{mode_index: {freq_hz: value}}`` with mode indices in ascending order
        and ``value`` a float (``"transmission"``/``"power"``) or complex
        (``"amplitude"``). Frequencyless planes key the inner dict on ``0.0``.
    """
    if quantity not in _QUANTITIES:
        raise ValueError(
            f"quantity must be one of {_QUANTITIES}, got {quantity!r}")
    if not mode_bank:
        raise ValueError("mode_bank is empty — nothing to decompose onto")

    first = next(iter(mode_bank.values()))
    out: Dict[int, Dict[float, Any]] = {}

    if isinstance(first, Mapping):
        # per-frequency form: {freq_hz: {mode_index: Mode}}.
        index_sets = []
        for f, inner in mode_bank.items():
            if not isinstance(inner, Mapping):
                raise ValueError(
                    "mode_bank mixes per-frequency ({freq: {idx: Mode}}) and "
                    "frozen ({idx: Mode}) forms; use one consistently")
            for v in inner.values():
                if not _is_mode(v):
                    raise ValueError(
                        "mode_bank inner values must be modes "
                        f"({{freq: {{idx: Mode}}}}); got a {type(v).__name__} "
                        f"at f={f}")
            index_sets.append(frozenset(int(i) for i in inner))
        indices = sorted(set().union(*index_sets))
        if not indices:
            raise ValueError(
                "mode_bank has frequency entries but no mode indices — every "
                "inner {idx: Mode} map is empty")
        # Require a RECTANGULAR bank: the SAME indices at every frequency. A
        # ragged bank (an index present at only some freqs) would make the
        # per-index nearest-frequency lookup silently fabricate a reading at a
        # frequency the caller never supplied that mode for (an unphysical value).
        if any(s != index_sets[0] for s in index_sets):
            raise ValueError(
                f"mode_bank is ragged: every frequency must carry the SAME mode "
                f"indices {indices} (build it with solve_mode_bank)")
        for idx in indices:
            mbf = {float(f): inner[idx] for f, inner in mode_bank.items()}
            fallback = next(iter(mbf.values()))  # used only for a frequencyless plane
            terms = _overlap_terms(
                sim_plane_fields, fallback, axis=axis, direction=direction,
                center_um=center_um, thickness_axis=thickness_axis,
                modes_by_freq=mbf, colocate=colocate, destagger_dl=destagger_dl,
                fold_low=fold_low)
            out[idx] = {f: _term_to_quantity(a, p, quantity)
                        for f, (a, p) in terms.items()}
        return out

    if not _is_mode(first):
        raise ValueError(
            "mode_bank values must be either per-frequency maps "
            "({freq: {idx: Mode}}) or modes ({idx: Mode}); "
            f"got a {type(first).__name__}")
    for v in mode_bank.values():
        if not _is_mode(v):
            raise ValueError(
                "mode_bank mixes frozen ({idx: Mode}) and per-frequency "
                "({freq: {idx: Mode}}) forms; use one consistently")
    # frozen form: {mode_index: Mode} projected onto every plane frequency.
    for idx in sorted(int(i) for i in mode_bank):
        terms = _overlap_terms(
            sim_plane_fields, mode_bank[idx], axis=axis, direction=direction,
            center_um=center_um, thickness_axis=thickness_axis, colocate=colocate,
            destagger_dl=destagger_dl, fold_low=fold_low)
        out[idx] = {f: _term_to_quantity(a, p, quantity)
                    for f, (a, p) in terms.items()}
    return out


# ---------------------------------------------------------------------------
# Mode ⇄ mode overlap (coupling efficiency between two cross-section modes).
#
# Unlike :func:`mode_transmission` (which projects a *recorded FDTD field plane*
# onto one frozen mode), this projects one MODE onto another — e.g. how much
# power a waveguide's TE0 couples into a different waveguide's TE0, or into a
# lensed-fibre / free-space Gaussian beam (the fibre-to-chip coupling efficiency,
# or a w1→w2 taper's intrinsic mode-mismatch loss). Either operand may be a
# scalar :class:`~photonhub.plugins.modes.Mode`, a full-vector
# :class:`~photonhub.plugins.vector_modes.VectorMode`, or a :func:`gaussian_mode`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeOverlap:
    """Result of a mode⇄mode overlap (:func:`mode_overlap`).

    Carries BOTH common definitions of "mode overlap" so the caller picks the one
    their convention wants — Lumerical reports the same pair ("power coupling" and
    "overlap"):

    Attributes
    ----------
    power:
        **Power-coupling efficiency** ``∈ [0, 1]`` — the headline number, the
        fraction of mode_a's power that couples into mode_b across a butt joint.
        The impedance-aware two-term (E *and* H) Poynting coupling, with the form
        chosen (see :attr:`method`) to be the *most accurate available* for the
        operands:

        * **both full-vector** (``method="snyder_love"``) — the rigorous Snyder &
          Love / Tidy3D / Lumerical power coupling

              A = ∫ (E₁ × H₂*) · n̂ dA ,   B = ∫ (E₂ × H₁*) · n̂ dA ,
              P_i = ½ Re ∫ (E_i × H_i*) · n̂ dA ,
              power = | ¼ (conj(A) + B) |² / (P₁ P₂)
                    = | ¼ ∫ (E₁* × H₂ + E₂ × H₁*) · n̂ dA |² / (P₁ P₂) .

          This is the physically exact power transfer and reproduces Tidy3D's
          ``ModeData.dot`` to the colocation floor (~1e-4), carrying the genuine
          ``n_eff`` (impedance) mismatch through each mode's true ``H``.

        * **either operand scalar / Gaussian** (``method="geomean"``) — the bounded,
          argument-order-independent geometric mean ``|A| |B| / (4 P₁ P₂)``. A scalar
          mode's ``H`` is the ``(n_eff/η₀)·(ẑ×E)`` reconstruction, which makes the
          Snyder–Love form **over-count by the inverse-Fresnel factor**
          ``(n_a+n_b)²/(4 n_a n_b)`` and breach 1 across an index step; the geometric
          mean cancels it and equals :attr:`field` exactly for a scalar pair.

        ``power`` is ``1`` for two identical co-propagating modes and ``0`` for
        power-orthogonal modes (e.g. TE0 vs TE1). It is bounded by 1 to within
        discretization error; two *dissimilar* full-vector modes can overshoot 1 by
        a small margin (``≲ 1e-3`` on a shared grid — Tidy3D's ``dot`` does the same,
        since distinct modes of *different* guides are not a single orthonormal
        basis). Take ``min(power, 1)`` for a hard efficiency, or read :attr:`field`
        for the rigorously bounded overlap.
    field:
        **Field overlap** ``|F|² ∈ [0, 1]`` — the simpler, impedance-blind
        Hermitian transverse-E correlation

            F = ∫ E₁* · E₂ dA / sqrt( ∫|E₁|² dA · ∫|E₂|² dA ) ,

        the same quantity :func:`~photonhub.plugins.mode_tracking.transverse_overlap`
        reports. For two full-vector modes of differing impedance this differs from
        :attr:`power`; for scalar/Gaussian modes the two coincide.
    coupling:
        **Complex power-coupling amplitude** — ``|coupling|² == power`` and
        ``arg(coupling)`` is the relative modal phase, taken from the Snyder & Love
        coefficient ``a₁₂ = ¼∫(E₁* × H₂ + E₂ × H₁*) · n̂ dA``. For two full-vector
        modes this is exactly the complex coupling coefficient ``a₁₂/√(P₁P₂)`` an
        **S-matrix** assembler needs (``Sᵢⱼ = bᵢ/aⱼ``): it carries the ``e^{-iβL}``
        phase the mode accumulates, so a coherent multimode port sums correctly. (For
        a scalar/Gaussian operand the magnitude follows the bounded :attr:`power` and
        the phase the overlap sign.) Distinct from :attr:`amplitude`, which is the
        impedance-blind FIELD overlap.
    amplitude:
        The complex field overlap ``F`` above — ``|amplitude|² == field`` (NOT
        ``power``), and its phase ``arg(F)`` is the relative modal phase of the
        E-fields. For the power-coupling phase prefer :attr:`coupling`; for the
        field-shape sign use this.
    power_a, power_b:
        The two modes' own powers ``P₁``, ``P₂`` on the evaluation grid (SI watts
        for a VectorMode; ``(n_eff/2η₀)∫|E|²`` for a scalar/Gaussian mode). Carried
        for diagnostics.
    method:
        Which power-coupling form :attr:`power` used: ``"snyder_love"`` (the exact
        two-term coupling — both operands full-vector, or the contra-propagating
        ``direction="-"`` fallback for scalar/Gaussian operands, where the
        direction-blind geometric mean would wrongly read ``1`` for a mode
        against its own backward partner) or ``"geomean"`` (the bounded
        geometric mean, at least one scalar/Gaussian operand, co-propagating).
        See :attr:`power`.
    """

    power: float
    field: float
    amplitude: complex
    power_a: float
    power_b: float
    method: str = "geomean"
    coupling: complex = 0.0 + 0.0j

    @property
    def mismatch_db(self) -> float:
        """Mode-mismatch insertion loss in dB, ``-10 log10(power)`` (``>= 0``;
        ``0`` for a perfect match, ``+inf`` for orthogonal modes). The loss a
        butt-coupling / fibre-chip / taper junction incurs purely from the modal
        profile mismatch (no propagation/material loss)."""
        p = float(self.power)
        return float("inf") if p <= 0.0 else float(-10.0 * np.log10(p))

    def __float__(self) -> float:
        """The headline power-coupling efficiency, so ``float(mode_overlap(a, b))``
        is the coupling number."""
        return float(self.power)


def _mode_plane_fields(
    mode: Any,
    c1: np.ndarray,
    c2: np.ndarray,
    *,
    axis: Axis,
    center_um: Tuple[float, float],
    thickness_axis: Optional[Axis],
    interp_order: int = 3,
    direction: Direction = "+",
) -> Dict[str, np.ndarray]:
    """Transverse ``(e1, e2, h1, h2)`` of one mode on the plane grid ``(c1, c2)`` —
    the full-vector path for a ``VectorMode`` (its true H), the scalar-limit
    reconstruction for a scalar :class:`Mode` / :func:`gaussian_mode`. Mirrors the
    dispatch in :func:`_overlap_terms`. ``direction="-"`` builds the backward
    (contra-propagating) mode by flipping its transverse ``H``. ``interp_order`` is
    the cross-grid resample order (3 = bicubic, the mode⇄mode default; 1 =
    bilinear)."""
    if hasattr(mode, "ex"):  # VectorMode — carries its own (ex, ey, hx, hy)
        return vector_modal_fields(
            mode, c1, c2, axis=axis, direction=direction, center_um=center_um,
            thickness_axis=thickness_axis, interp_order=interp_order)
    return modal_fields(
        mode, c1, c2, axis=axis, direction=direction, center_um=center_um,
        thickness_axis=thickness_axis, interp_order=interp_order)


def _union_grid(
    mode_a: Any,
    mode_b: Any,
    center_a: Tuple[float, float],
    center_b: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """A common transverse ``(x_um, y_um)`` grid (for ``axis="z"``) covering BOTH
    modes' windows at the finer of their two spacings, so neither profile is
    truncated. For two modes already on the same grid (same shape + ``dl``,
    centered) this reproduces their native coordinates exactly, so an identical
    pair self-overlaps to ``1`` with no resample error."""
    def wh(m: Any, c: Tuple[float, float]):
        # A GRADED-window mode carries its true node ladders (centre-relative);
        # its array pitch is NOT the scalar dl_x_um, so — exactly as
        # vector_modal_fields does — prefer the ladders. Reconstructing a uniform
        # ladder from dl_x_um would mis-size the window ((nx-1)*dl_base ≠ the true
        # graded span) and the resample would zero-fill (clip) the mode tails out
        # of the overlap. The ladders already encode placement (drop the offset).
        mxc = getattr(m, "x_coords_um", None)
        if mxc is not None:
            x = np.asarray(mxc, dtype=np.float64) + c[0]
            y = np.asarray(m.y_coords_um, dtype=np.float64) + c[1]
            dx = float(np.min(np.diff(x))) if x.size > 1 else float(m.dl_x_um)
            dy = float(np.min(np.diff(y))) if y.size > 1 else float(m.dl_y_um)
            return x, y, dx, dy
        ny, nx = (m.ex.shape if hasattr(m, "ex") else m.field.shape)
        # Include any window-placement offset (see vector_modal_fields) so the
        # union grid covers the mode where it will actually be placed.
        ow, oh = getattr(m, "center_offset_um", None) or (0.0, 0.0)
        x = (np.arange(nx) - (nx - 1) / 2.0) * m.dl_x_um + c[0] + ow
        y = (np.arange(ny) - (ny - 1) / 2.0) * m.dl_y_um + c[1] + oh
        return x, y, float(m.dl_x_um), float(m.dl_y_um)

    xa, ya, dxa, dya = wh(mode_a, center_a)
    xb, yb, dxb, dyb = wh(mode_b, center_b)
    dx, dy = min(dxa, dxb), min(dya, dyb)
    x0, x1 = min(xa[0], xb[0]), max(xa[-1], xb[-1])
    y0, y1 = min(ya[0], yb[0]), max(ya[-1], yb[-1])
    nX = int(round((x1 - x0) / dx)) + 1
    nY = int(round((y1 - y0) / dy)) + 1
    return x0 + np.arange(nX) * dx, y0 + np.arange(nY) * dy


def mode_overlap(
    mode_a: Any,
    mode_b: Any,
    *,
    axis: Axis = "z",
    grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    center_a: Tuple[float, float] = (0.0, 0.0),
    center_b: Tuple[float, float] = (0.0, 0.0),
    thickness_axis: Optional[Axis] = None,
    interp: Literal["cubic", "linear"] = "cubic",
    direction: Direction = "+",
) -> ModeOverlap:
    """Overlap / coupling efficiency between two **modes** ``mode_a`` and
    ``mode_b`` — the mode⇄mode companion to the field-plane :func:`mode_transmission`.

    Each operand is a frozen FDE mode (scalar :class:`~photonhub.plugins.modes.Mode`
    or full-vector :class:`~photonhub.plugins.vector_modes.VectorMode`) or an analytic
    :func:`gaussian_mode`. Both are resampled onto a **common transverse grid**, so
    they need not share resolution, window, or even a polarization branch — the
    overlap measures how much of one mode's power couples into the other. Typical
    uses: a waveguide TE0 onto a lensed-fibre / free-space **Gaussian** (the
    fibre-to-chip coupling efficiency), the fundamental of a width-``w1`` guide onto
    that of a width-``w2`` guide (a butt-joint / taper-step mismatch loss), or any
    two solved cross-section modes.

    ``mode_a`` is taken as forward; ``direction`` sets ``mode_b``'s propagation
    (``"+"`` co-propagating, the default; ``"-"`` contra-propagating / backward).

    Parameters
    ----------
    mode_a, mode_b:
        The two modes. Either may be scalar, full-vector, or a Gaussian — mixed is
        fine (the kernel uses each mode's true ``H`` when it has one, else the
        scalar-limit ``H = (n_eff/η₀)(ẑ × e)``; see the module docstring).
    axis:
        Propagation axis / cross-section normal, default ``"z"`` (the natural frame
        for an FDE cross-section: transverse plane ``x``–``y``). For any other axis
        you must pass ``grid`` explicitly (the auto-grid assumes the ``z`` mapping
        width→x, height→y).
    grid:
        Optional ``(x_um, y_um)`` 1-D coordinate arrays (microns) of the evaluation
        plane. ``None`` (default) auto-builds the union of both modes' windows at
        the finer spacing (:func:`_union_grid`) so neither is clipped — for two
        modes already on the same grid this is their native grid (exact self-overlap).
    center_a, center_b:
        ``(t1, t2)`` offset (microns) of each mode's axis in the plane frame —
        use ``center_b`` to model a **lateral misalignment** (e.g. a fibre offset
        from the waveguide), which lowers the coupling.
    thickness_axis:
        Slab-normal axis, forwarded to the per-mode field assembly to orient the
        profile (see :func:`modal_fields`); ``None`` keeps the default ``z``-frame
        mapping.
    interp:
        Cross-grid resample order when the two modes live on **different** grids:
        ``"cubic"`` (default, bicubic — ~10³–10⁴× lower resampling error for a
        smooth, well-resolved mode) or ``"linear"`` (bilinear). Modes already on a
        common grid are not resampled at all, so this has no effect there. Cubic
        needs scipy; without it the kernel transparently falls back to bilinear.
    direction:
        Propagation direction of ``mode_b`` relative to the forward ``mode_a``:
        ``"+"`` (default) co-propagating — the transmission / butt-joint coupling;
        ``"-"`` contra-propagating — ``mode_b``'s transverse ``H`` is flipped to the
        backward mode (for contra-directional couplers, Bragg back-coupling, or a
        forward-vs-backward orthogonality check). **Note:** a forward mode and the
        *same* mode's backward partner are power-orthogonal, so their overlap is
        ``~0``; the reflection *amplitude* at a real junction comes from mode-matching
        the boundary conditions (the :mod:`~photonhub.plugins.eme` interface S-matrix),
        not from this single-plane overlap. With a scalar / Gaussian operand the
        bounded geometric-mean power form is direction-blind, so ``"-"`` falls
        back to the two-term ``|a12|^2`` (``snyder_love``) form — which does
        carry the forward/backward cancellation and reads the physical ``~0`` —
        and emits a ``UserWarning`` (see :attr:`ModeOverlap.method`).

    Returns
    -------
    ModeOverlap
        ``.power`` is the headline impedance-aware power-coupling efficiency
        (``∈ [0, 1]``, ``1`` for identical modes); ``.field`` the simpler E-only
        field overlap; ``.coupling`` the complex power-coupling coefficient
        (``|coupling|² == power``, ``arg`` = the S-matrix modal phase);
        ``.amplitude`` the complex field overlap; ``.mismatch_db`` the mismatch
        insertion loss. See :class:`ModeOverlap`.
    """
    if not (_is_mode(mode_a) and _is_mode(mode_b)):
        raise ValueError(
            "mode_overlap takes two modes (scalar Mode, VectorMode, or "
            "gaussian_mode); got "
            f"{type(mode_a).__name__} and {type(mode_b).__name__}")
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if interp not in ("cubic", "linear"):
        raise ValueError(f"interp must be 'cubic' or 'linear', got {interp!r}")
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    order = 3 if interp == "cubic" else 1

    if grid is None:
        if axis != "z":
            raise ValueError(
                "auto grid is only built for axis='z'; pass grid=(x_um, y_um) "
                f"explicitly for axis={axis!r}")
        c1, c2 = _union_grid(mode_a, mode_b, center_a, center_b)
    else:
        c1 = np.asarray(grid[0], dtype=np.float64)
        c2 = np.asarray(grid[1], dtype=np.float64)
        if c1.ndim != 1 or c2.ndim != 1:
            raise ValueError("grid must be two 1-D coordinate arrays (x_um, y_um)")

    ma = _mode_plane_fields(mode_a, c1, c2, axis=axis, center_um=center_a,
                            thickness_axis=thickness_axis, interp_order=order)
    mb = _mode_plane_fields(mode_b, c1, c2, axis=axis, center_um=center_b,
                            thickness_axis=thickness_axis, interp_order=order,
                            direction=direction)
    e1a, e2a, h1a, h2a = ma["e1"], ma["e2"], ma["h1"], ma["h2"]
    e1b, e2b, h1b, h2b = mb["e1"], mb["e2"], mb["h1"], mb["h2"]

    w1 = _cell_widths(c1)                       # along t1
    w2 = _cell_widths(c2)                       # along t2
    dA = np.outer(w2, w1)                        # [i_t2, i_t1]

    # Power coupling — the two-term (E and H) Poynting overlap. n̂-component of a
    # transverse cross product (a1,a2)×(b1,b2) = a1 b2 − a2 b1.
    A = np.sum((e1a * np.conj(h2b) - e2a * np.conj(h1b)) * dA)   # ∫ E_a × H_b*
    B = np.sum((e1b * np.conj(h2a) - e2b * np.conj(h1a)) * dA)   # ∫ E_b × H_a*
    # Mode powers. mode_a is forward (P_a > 0); a backward mode_b carries power in
    # −n̂ so its signed flux is negative — normalize by the MAGNITUDE |P| (the
    # power the mode carries in its own propagation sense), which leaves the
    # co-propagating case unchanged and makes the contra-propagating case well-posed.
    p_a = 0.5 * abs(float(np.real(np.sum(
        (e1a * np.conj(h2a) - e2a * np.conj(h1a)) * dA))))        # ½|Re ∫ E_a × H_a*|
    p_b = 0.5 * abs(float(np.real(np.sum(
        (e1b * np.conj(h2b) - e2b * np.conj(h1b)) * dA))))        # ½|Re ∫ E_b × H_b*|
    if p_a == 0.0 or p_b == 0.0:
        raise ValueError(
            "a mode has zero power on the evaluation grid "
            f"(P_a={p_a:.3e}, P_b={p_b:.3e}); check that the grid covers each mode.")

    # Headline power coupling. When BOTH modes carry a true vector H (a VectorMode
    # / FDTD-derived mode), use the rigorous Snyder & Love / Tidy3D / Lumerical form
    #   power = |¼∫(E_a*×H_b + E_b×H_a*)|² / (P_a P_b) = |¼(conj(A)+B)|² / (P_a P_b),
    # the physically exact butt-joint power transfer — it matches Tidy3D's
    # ``ModeData.dot`` to the colocation floor (~1e-4; see
    # benchmarks/tidy3d/mode_overlap_parity.py), versus the ~1e-3 the geometric-mean
    # form costs for dissimilar modes. When EITHER operand is scalar / Gaussian, its
    # (n_eff/η₀)·(ẑ×E) reconstruction of H makes that two-term metric non-positive
    # across an index step (it over-counts by the inverse-Fresnel factor
    # (n_a+n_b)²/(4 n_a n_b) and breaches 1), so there we fall back to the bounded,
    # argument-order-independent geometric mean |A||B|/(4 P_a P_b), which for a
    # scalar pair equals the field overlap exactly.
    a12 = 0.25 * (np.conj(A) + B)                # ¼∫(E_a*×H_b + E_b×H_a*)
    if hasattr(mode_a, "hx") and hasattr(mode_b, "hx"):
        power = float(np.abs(a12) ** 2 / (p_a * p_b))
        method = "snyder_love"
    elif direction == "-":
        # The geometric mean |A||B| is DIRECTION-BLIND: flipping mode_b's H
        # flips A's sign but not |A|, so a scalar mode would "overlap" its own
        # backward partner at power == 1 instead of the physical ~0. Only the
        # two-term Snyder & Love combination ¼(conj(A)+B) carries the
        # forward/backward cancellation, so the contra-propagating scalar /
        # Gaussian case falls back to it — with a warning, because the
        # scalar-limit H makes that form over-count co-directional coupling
        # across an index step by the inverse-Fresnel factor (the reason the
        # geomean exists); for the near-orthogonal backward reading that bias
        # is immaterial.
        warnings.warn(
            "mode_overlap(direction='-') with a scalar/Gaussian operand: the "
            "bounded geometric-mean form is direction-blind, falling back to "
            "the two-term |a12|^2 (snyder_love) form for the backward "
            "coupling (correct ~0 orthogonality; may over-count across an "
            "index step).",
            UserWarning, stacklevel=2)
        power = float(np.abs(a12) ** 2 / (p_a * p_b))
        method = "snyder_love"
    else:
        power = float(np.abs(A) * np.abs(B) / (4.0 * p_a * p_b))
        method = "geomean"
    # Complex power-coupling coefficient: |coupling|² == power, arg = relative modal
    # phase (= a₁₂'s phase). For the snyder_love path this equals a₁₂/√(P_a P_b).
    phase = float(np.angle(a12)) if a12 != 0.0 else 0.0
    coupling = complex(np.sqrt(power) * np.exp(1j * phase))

    # Field overlap — the impedance-blind Hermitian transverse-E correlation.
    num = np.sum((np.conj(e1a) * e1b + np.conj(e2a) * e2b) * dA)
    na = np.sum((np.abs(e1a) ** 2 + np.abs(e2a) ** 2) * dA)
    nb = np.sum((np.abs(e1b) ** 2 + np.abs(e2b) ** 2) * dA)
    amp = complex(num / np.sqrt(na * nb)) if na > 0.0 and nb > 0.0 else 0.0 + 0.0j
    field = float(np.abs(amp) ** 2)

    return ModeOverlap(
        power=power, field=field, amplitude=amp,
        power_a=float(p_a), power_b=float(p_b), method=method, coupling=coupling)


_MATRIX_QUANTITIES = ("coupling", "power", "field", "amplitude")


def mode_overlap_matrix(
    modes_a: "Sequence[Any]",
    modes_b: "Sequence[Any]",
    *,
    quantity: str = "coupling",
    **kwargs: Any,
) -> np.ndarray:
    """Pairwise mode⇄mode overlap **matrix** between two sequences of modes — the
    multimode-port companion to :func:`mode_overlap`.

    Entry ``M[i, j]`` is :func:`mode_overlap` of ``modes_a[i]`` with ``modes_b[j]``.
    With the default ``quantity="coupling"`` it is the **complex** coupling
    coefficient (``ModeOverlap.coupling``), i.e. an **S-matrix block** between the
    two mode bases: ``|M[i, j]|²`` is the power-coupling matrix and ``arg`` carries
    the modal phase, so a coherent multimode port sums correctly. Use it to assemble
    a junction/coupler S-matrix (``modes_a`` = the input port basis, ``modes_b`` =
    the output port basis), or with ``modes_a == modes_b`` to check a mode set's
    self-orthogonality (≈ identity).

    Parameters
    ----------
    modes_a, modes_b:
        Sequences of modes (scalar :class:`~photonhub.plugins.modes.Mode`,
        :class:`~photonhub.plugins.vector_modes.VectorMode`, or :func:`gaussian_mode`).
        The matrix is ``(len(modes_a), len(modes_b))``.
    quantity:
        Which :class:`ModeOverlap` field fills the matrix: ``"coupling"`` (default,
        complex S-matrix amplitude), ``"power"`` (real efficiency ``∈ [0, 1]``),
        ``"field"`` (real field overlap), or ``"amplitude"`` (complex field overlap).
    **kwargs:
        Forwarded verbatim to :func:`mode_overlap` for every pair (``axis``,
        ``grid``, ``center_a``, ``center_b``, ``thickness_axis``, ``interp``,
        ``direction``) — e.g. ``direction="-"`` for a contra-directional block.

    Returns
    -------
    numpy.ndarray
        ``(len(modes_a), len(modes_b))`` — complex for ``"coupling"``/``"amplitude"``,
        real for ``"power"``/``"field"``.

    .. warning::
        A MIXED list (some full-vector ``VectorMode``, some scalar/Gaussian
        operands) silently mixes normalization conventions across the entries:
        vector⇄vector pairs use the exact two-term ``snyder_love`` power form
        while any pair with a scalar operand uses the bounded ``geomean`` —
        the two differ by up to the inverse-Fresnel factor across an index
        step, so entries of one matrix are not mutually comparable. A
        ``UserWarning`` is emitted for a mixed list; prefer a homogeneous
        basis (solve everything full-vector).
    """
    if quantity not in _MATRIX_QUANTITIES:
        raise ValueError(
            f"quantity must be one of {_MATRIX_QUANTITIES}, got {quantity!r}")
    rows = list(modes_a)
    cols = list(modes_b)
    kinds = {hasattr(m, "hx") for m in rows + cols}
    if len(kinds) > 1:
        warnings.warn(
            "mode_overlap_matrix over a MIXED basis (full-vector and "
            "scalar/Gaussian operands): vector-vector entries use the "
            "snyder_love power form, scalar-involving entries the geomean — "
            "the matrix mixes normalization conventions and its entries are "
            "not mutually comparable.", UserWarning, stacklevel=2)
    dtype = complex if quantity in ("coupling", "amplitude") else float
    out = np.zeros((len(rows), len(cols)), dtype=dtype)
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            out[i, j] = getattr(mode_overlap(a, b, **kwargs), quantity)
    return out


def gaussian_mode(
    *,
    wavelength_um: float,
    dl_um: float,
    mfd_um: Optional[Union[float, Tuple[float, float]]] = None,
    waist_um: Optional[Union[float, Tuple[float, float]]] = None,
    n: float = 1.0,
    polarization: Literal["TE", "TM"] = "TE",
    window_um: Optional[Union[float, Tuple[float, float]]] = None,
) -> Mode:
    """An analytic **fundamental-Gaussian** beam (a lensed-fibre / free-space
    LP₀₁ / TEM₀₀ spot at its waist) wrapped as a scalar
    :class:`~photonhub.plugins.modes.Mode`, so it drops straight into
    :func:`mode_overlap` for a fibre-to-chip coupling efficiency.

    The transverse field at the waist is the real Gaussian

        E(x, y) = exp( -(x/w₀ₓ)² - (y/w₀ᵧ)² )

    (flat phase at the waist), linearly polarized along ``x`` (``"TE"``, an
    ``Ex``-major mode) or ``y`` (``"TM"``). Its modal index is the background index
    ``n`` (so the scalar-limit ``H = (n/η₀)(ẑ × E)`` carries the right impedance for
    the medium the beam lives in — air ``n=1``, or an index-matched cladding).

    Parameters
    ----------
    wavelength_um:
        Free-space wavelength (microns) — stored on the mode for bookkeeping; the
        waist sizing is wavelength-independent (you specify the spot size directly).
    dl_um:
        Grid spacing of the returned profile (microns). Make it ``<=`` the partner
        mode's spacing so the overlap grid resolves both.
    mfd_um:
        **Mode-field diameter** (microns) — the 1/e² *intensity* diameter, the
        spec fibre vendors quote (e.g. SMF-28 ≈ 10.4 µm at 1550 nm). Scalar (round
        beam) or ``(MFDx, MFDy)`` (elliptical, e.g. a lensed fibre). Relates to the
        field 1/e radius as ``w₀ = MFD/2``. Provide exactly one of ``mfd_um`` /
        ``waist_um``.
    waist_um:
        **Field 1/e radius** ``w₀`` (microns), scalar or ``(w₀ₓ, w₀ᵧ)`` — the other
        way to size the beam (``w₀ = MFD/2``).
    n:
        Background refractive index the beam propagates in (default ``1.0``, air),
        used as the mode's ``n_eff`` for the ``H`` reconstruction / impedance.
    polarization:
        ``"TE"`` (``Ex``-major, default) or ``"TM"`` (``Ey``-major). Match the
        partner waveguide mode's dominant polarization for a meaningful coupling.
    window_um:
        Total grid extent (microns), scalar or ``(Wx, Wy)``. Defaults to ``3×`` the
        MFD on each axis (half-window ``3 w₀``, i.e. the field has decayed to
        ``e⁻⁹``), enough that the truncation does not bias the overlap.

    Returns
    -------
    Mode
        A scalar :class:`~photonhub.plugins.modes.Mode` whose ``.field`` is the
        L2-normalized Gaussian, ``.n_eff == n``, on a centered square grid.
    """
    if (mfd_um is None) == (waist_um is None):
        raise ValueError("provide exactly one of mfd_um or waist_um")
    if not (dl_um > 0.0):
        raise ValueError("dl_um must be > 0")
    if not (wavelength_um > 0.0):
        raise ValueError("wavelength_um must be > 0")
    if polarization not in ("TE", "TM"):
        raise ValueError(f"polarization must be 'TE' or 'TM', got {polarization!r}")

    def _pair(v: Union[float, Tuple[float, float]]) -> Tuple[float, float]:
        if isinstance(v, (tuple, list)):
            if len(v) != 2:
                raise ValueError(f"expected a scalar or a 2-tuple, got {v!r}")
            return float(v[0]), float(v[1])
        return float(v), float(v)

    if waist_um is not None:
        w0x, w0y = _pair(waist_um)
    else:
        mfx, mfy = _pair(mfd_um)
        w0x, w0y = 0.5 * mfx, 0.5 * mfy
    if not (w0x > 0.0 and w0y > 0.0):
        raise ValueError("waist / mode-field diameter must be > 0")

    if window_um is not None:
        wx, wy = _pair(window_um)
    else:
        wx, wy = 6.0 * w0x, 6.0 * w0y      # half-window = 3 w₀ ⇒ field ~ e⁻⁹
    nx = _gauss_odd(max(3, int(round(wx / dl_um))))
    ny = _gauss_odd(max(3, int(round(wy / dl_um))))

    xs = (np.arange(nx) - (nx - 1) / 2.0) * dl_um
    ys = (np.arange(ny) - (ny - 1) / 2.0) * dl_um
    X, Y = np.meshgrid(xs, ys)             # [iy, ix]
    field = np.exp(-(X / w0x) ** 2 - (Y / w0y) ** 2)
    norm = np.sqrt(np.sum(field ** 2))
    if norm > 0.0:
        field = field / norm

    return Mode(
        n_eff=float(n),
        field=field,
        wavelength_um=float(wavelength_um),
        polarization=polarization,
        dl_x_um=float(dl_um),
        dl_y_um=float(dl_um),
    )


def _gauss_odd(v: int) -> int:
    """Smallest odd integer ``>= v`` so a Gaussian's peak sits on a cell center
    (a centered profile, matching the FDE mode rasterizers)."""
    v = int(v)
    return v if v % 2 == 1 else v + 1
