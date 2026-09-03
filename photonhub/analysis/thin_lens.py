"""High-NA vectorial focused beam (Richards–Wolf) — the metalens-grade
excitation the paraxial :mod:`gaussian_beam` cannot provide.

An aplanatic-lens focus is the angular-spectrum superposition of plane waves
over the pupil cone (Richards & Wolf 1959): each pupil ray ``(theta, phi)``
carries the incident linear polarization rotated into its meridional frame
(the vectorial effect that creates the longitudinal ``E_z`` and the
polarization-elongated spot at high NA), the aplanatic apodization
``sqrt(cos theta)``, and an optional user ``pupil(theta, phi)`` amplitude
(complex; vortex plates, annuli, aberrations).

Evaluation is EXACT and fast: the focal field is a 2-D Fourier transform of
the weighted pupil, so the plane fields are computed by FFT on a k-grid
matched to the simulation's own ``dl`` — and each Yee component's half-cell
stagger is applied as an exact spectral phase ``e^{i k delta}`` before the
transform (no interpolation, no paraxial approximation anywhere). E and H
are built per ray (``h = n/eta0 * k_hat x e``), so the Huygens pair is
consistent at any NA.

Two layers, mirroring :mod:`gaussian_beam`:

* :func:`thin_lens_beam` — the sampled plane as a ``yee_staggered``
  :class:`~photonhub.analysis.vector_modes.VectorMode` (transverse-E pair
  jointly L2-normalized; also a monitor/overlap reference);
* :func:`thin_lens_source` — that plus the per-cell equivalence-current
  sheet, returning the ``PointDipole`` list for ``Simulation.sources``.

Conventions: engine ``e^{-i omega t}`` phasors, forward propagation
``e^{+i k_n a}`` toward ``+axis`` (``direction`` handled by the sheet);
``focus_distance_um >= 0`` puts the focus that far DOWNSTREAM of the
injection plane (0 = focus on the plane). ``n_eff`` for the sheet's
half-cell straddle is the power-weighted mean ``n <cos theta>`` of the cone.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import numpy as np

from ..components.sources import PointDipole
from ..viz import _geometry as _geom
from .gaussian_beam import _plane_grids, _resolve_index, _resolve_wavelength
from .mode_overlap import ETA0
from .vector_modes import VectorMode
from .yee_mode import _window_center_offset

__all__ = ["thin_lens_beam", "thin_lens_source"]

_AXES = ("x", "y", "z")


def _resolve_pol_angle(axis: str, polarization, pol_angle) -> float:
    if polarization is not None and pol_angle is not None:
        raise ValueError("give polarization OR pol_angle, not both")
    if pol_angle is not None:
        return float(pol_angle)
    h_letter, v_letter = _geom.in_plane_axes(axis)
    if polarization is None:
        return 0.0
    p = str(polarization)
    if p.startswith("E"):
        p = p[1:].lower()
    if p == h_letter:
        return 0.0
    if p == v_letter:
        return 0.5 * math.pi
    raise ValueError(
        f"polarization must name an in-plane E component of a {axis}-cut "
        f"(E{h_letter} or E{v_letter}), got {polarization!r}")


def thin_lens_beam(
    sim,
    *,
    axis: str,
    na: float,
    wavelength_um: Optional[float] = None,
    freq_hz: Optional[float] = None,
    source_time=None,
    n: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    polarization: Optional[str] = None,
    pol_angle: Optional[float] = None,
    focus_distance_um: float = 0.0,
    pupil=None,
    half_w_um: Optional[float] = None,
    half_v_um: Optional[float] = None,
    window_airy_units: float = 12.0,
) -> VectorMode:
    """The Richards–Wolf focused beam on ``sim``'s ``axis``-normal Yee plane.

    Parameters
    ----------
    sim:
        Simulation whose grid/size/§20 symmetry the beam is sampled on (a
        placeholder shell is fine).
    axis:
        Propagation axis ('x' | 'y' | 'z') — the injection plane's normal.
    na:
        Numerical aperture ``n sin(theta_max)`` of the focusing cone, in the
        LAUNCH medium (``0 < na < n``).
    wavelength_um, freq_hz, source_time:
        Frequency the beam is built at (at most one of the first two; else
        ``source_time.freq0_hz``).
    n:
        Launch-medium index. Default ``sqrt(sim.background.permittivity)``.
    center_um:
        Transverse focus centre ``(h, v)`` in the in-plane-axis order;
        default the domain centre.
    polarization, pol_angle:
        Incident linear polarization before the lens (component name or
        angle from the first in-plane axis). Default: along the first
        in-plane axis.
    focus_distance_um:
        Distance from the injection plane to the focus, along propagation
        (>= 0; 0 = focus on the plane).
    pupil:
        Optional complex pupil function ``pupil(theta, phi) -> complex``
        (vectorized over numpy arrays), multiplied onto the aplanatic
        ``sqrt(cos theta)``. Default: uniform (clipped Airy-type focus).
    half_w_um, half_v_um, window_airy_units:
        Sampled-window half-extents. Default: ``window_airy_units`` (12)
        Airy radii ``0.61 lambda / NA`` — wide enough that the discarded
        tail carries ~<1e-3 of the power for a uniform pupil — widened by
        the defocus cone ``focus_distance * tan(theta_max)``, clipped to
        the domain.

    Returns
    -------
    VectorMode
        ``yee_staggered``; six components sampled at their true in-plane Yee
        locations; ``n_eff = n <cos theta>`` (power-weighted).
    """
    if axis not in _AXES:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    dl = getattr(sim.grid, "dl_um", None)
    if not dl:
        raise ValueError("thin_lens_beam needs the grid's base dl_um")
    dl = float(dl)

    lam_um = _resolve_wavelength(wavelength_um, freq_hz, source_time)
    n_bg = _resolve_index(sim, n)
    if not 0.0 < float(na) < n_bg:
        raise ValueError(
            f"na must satisfy 0 < na < n (= {n_bg:.4g}), got {na}")
    na = float(na)
    if focus_distance_um < 0.0:
        raise ValueError("focus_distance_um must be >= 0 (downstream focus)")
    pol = _resolve_pol_angle(axis, polarization, pol_angle)

    k = 2.0 * math.pi * n_bg / lam_um          # rad/um in the medium
    sin_max = na / n_bg
    theta_max = math.asin(min(sin_max, 1.0))

    # --- window -------------------------------------------------------------
    h_letter, v_letter = _geom.in_plane_axes(axis)
    size = sim.size_um
    if center_um is None:
        h_c = float(size[_AXES.index(h_letter)]) / 2.0
        v_c = float(size[_AXES.index(v_letter)]) / 2.0
    else:
        h_c, v_c = float(center_um[0]), float(center_um[1])
    airy_um = 0.61 * lam_um / na
    spread = float(window_airy_units) * airy_um + \
        focus_distance_um * math.tan(theta_max)
    if half_w_um is None:
        half_w_um = spread
    if half_v_um is None:
        half_v_um = spread

    def clip(half, c, letter):
        L = float(size[_AXES.index(letter)])
        return min(float(half), max(c, L - c))

    half_w = clip(half_w_um, h_c, h_letter)
    half_v = clip(half_v_um, v_c, v_letter)
    if not (half_w > 0.0 and half_v > 0.0):
        raise ValueError("the beam window half-extents must be > 0")

    h_node, v_node, (h_dq, v_dq), grids = _plane_grids(
        sim, axis, h_center=h_c, v_center=v_c, half_w=half_w, half_v=half_v,
        dl=dl)
    if h_dq is not None or v_dq is not None:
        raise ValueError(
            "thin_lens_beam requires a uniform grid over the window (the "
            "FFT evaluation is dl-matched); graded windows are not supported")
    nh = h_node.size
    nv = v_node.size

    # --- FFT k-grid matched to dl -------------------------------------------
    # Choose N so the FFT grid holds the window with >= 2x margin (the FFT is
    # periodic; the margin keeps the wrapped tail negligible) and resolves the
    # pupil disk finely (>= ~64 samples across the cone diameter).
    kt_max = k * sin_max
    N = 1 << max(int(math.ceil(math.log2(max(nh, nv) * 2))), 8)
    while (2.0 * kt_max) / (2.0 * math.pi / (N * dl)) < 64.0:
        N *= 2
    if N > 8192:
        N = 8192
        warnings.warn(
            "thin_lens_beam FFT grid capped at 8192^2; extremely low NA at "
            "coarse dl may sample the pupil disk sparsely", stacklevel=2)
    if max(nh, nv) > N:
        raise ValueError(
            f"beam window ({nh} x {nv} cells) exceeds the {N}^2 FFT "
            "evaluation grid — shrink half_w_um/half_v_um (or refine less)")
    dk = 2.0 * math.pi / (N * dl)
    kx = np.fft.fftfreq(N, d=1.0 / (N * dk))   # rad/um, FFT order
    KX, KY = np.meshgrid(kx, kx, indexing="ij")  # [ih, iv] frame: u=h, v=v
    KT2 = KX * KX + KY * KY
    inside = KT2 <= (kt_max * kt_max) * (1.0 + 1e-12)
    KT = np.sqrt(np.where(inside, KT2, 0.0))
    with np.errstate(invalid="ignore"):
        SIN_T = np.where(inside, KT / k, 0.0)
    COS_T = np.sqrt(np.clip(1.0 - SIN_T * SIN_T, 0.0, 1.0))
    PHI = np.arctan2(KY, KX)

    # pupil weight: aplanatic sqrt(cos) x user pupil, Jacobian 1/cos (d2k),
    # zero outside the cone. (Absolute scale is irrelevant — L2-normalized.)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(inside, 1.0 / np.sqrt(np.maximum(COS_T, 1e-9)), 0.0)
    if pupil is not None:
        THETA = np.arcsin(np.clip(SIN_T, 0.0, 1.0))
        W = W * np.where(inside, np.asarray(pupil(THETA, PHI),
                                            dtype=np.complex128), 0.0)

    # Richards–Wolf polarization rotation of the incident linear e0:
    #   s_hat = (-sin phi, cos phi, 0);   p_out = (cos t cos phi,
    #   cos t sin phi, -sin t);  e = (e0.s) s_hat + (e0.p_in) p_out
    cph, sph = np.cos(PHI), np.sin(PHI)
    e0s = math.sin(pol) * cph - math.cos(pol) * sph      # e0 . s_hat
    e0p = math.cos(pol) * cph + math.sin(pol) * sph      # e0 . p_in
    EH = W * (-e0s * sph + e0p * COS_T * cph)
    EV = W * (e0s * cph + e0p * COS_T * sph)
    EZ = W * (-e0p * SIN_T)
    # h = (n/eta0) k_hat x e per ray
    KHX, KHY, KHZ = SIN_T * cph, SIN_T * sph, COS_T
    y0 = n_bg / ETA0
    HH = y0 * (KHY * EZ - KHZ * EV)
    HV = y0 * (KHZ * EH - KHX * EZ)
    HZ = y0 * (KHX * EV - KHY * EH)

    # defocus: propagate from the focus (z = 0) back to the injection plane
    # z = -focus_distance (fields there converge toward the focus)
    zoff = -float(focus_distance_um)
    prop = np.exp(1j * (k * COS_T) * zoff)
    EH = EH * prop
    EV = EV * prop
    EZ = EZ * prop
    HH = HH * prop
    HV = HV * prop
    HZ = HZ * prop

    # --- evaluate on the Yee grids via stagger-phased inverse FFTs ----------
    # FFT grid x_i = i*dl (period N*dl). Anchor: the window node ladder h_node
    # is absolute; the beam centre (h_c, v_c) is the k-space origin. A sample
    # at absolute h = h_node[0] + i*dl + delta*dl maps to the FFT output at
    # index i with the spectral phase e^{i k (x0 + delta*dl)} folded in, where
    # x0 = h_node[0] - h_c (window origin relative to the beam centre).
    x0 = float(h_node[0]) - h_c
    v0 = float(v_node[0]) - v_c

    def sample(spec, dh, dv):
        ph = np.exp(1j * (KX * (x0 + dh * dl) + KY * (v0 + dv * dl)))
        # sum_k S e^{i k . r_i}: ifft2 with the N^2 scale folded out (the
        # absolute scale is normalized away below anyway)
        grid = np.fft.ifft2(spec * ph) * (N * N)
        return np.ascontiguousarray(grid[:nh, :nv].T)  # -> [iv, ih]

    ex = sample(EH, 0.5, 0.0)
    ey = sample(EV, 0.0, 0.5)
    ez = sample(EZ, 0.0, 0.0)
    hx = sample(HH, 0.0, 0.5)
    hy = sample(HV, 0.5, 0.0)
    hz = sample(HZ, 0.5, 0.5)

    norm = math.sqrt(float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2)))
    if not norm > 0.0:
        raise ValueError("the focused beam is identically zero on the window")
    ex, ey, ez, hx, hy, hz = (f / norm for f in (ex, ey, ez, hx, hy, hz))

    # power-weighted mean cos(theta) for the sheet's half-cell straddle
    p_w = np.abs(EH) ** 2 + np.abs(EV) ** 2 + np.abs(EZ) ** 2
    denom = float(np.sum(p_w))
    mean_cos = float(np.sum(p_w * COS_T) / denom) if denom > 0 else 1.0

    off = _window_center_offset(float(h_node[0]), float(v_node[0]),
                                nh, nv, dl, h_c, v_c)
    return VectorMode(
        n_eff=n_bg * mean_cos,
        n_group=None,
        ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz,
        wavelength_um=lam_um,
        dl_x_um=dl,
        dl_y_um=dl,
        center_offset_um=off,
        yee_staggered=True,
        x_coords_um=None,
        y_coords_um=None,
    )


def thin_lens_source(
    sim,
    *,
    axis: str,
    position_um: float,
    source_time,
    na: float,
    direction: str = "+",
    power_watts: float = 1.0,
    n: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    polarization: Optional[str] = None,
    pol_angle: Optional[float] = None,
    focus_distance_um: float = 0.0,
    pupil=None,
    wavelength_um: Optional[float] = None,
    freq_hz: Optional[float] = None,
    half_w_um: Optional[float] = None,
    half_v_um: Optional[float] = None,
    window_airy_units: float = 12.0,
    amplitude_threshold: float = 1e-6,
) -> List[PointDipole]:
    """Launch a high-NA focused beam — :func:`thin_lens_beam` plus the
    per-cell equivalence-current Huygens sheet. ``power_watts`` (default 1 W)
    normalizes the launched power on the engine's discrete Poynting
    quadrature like every other launch; parameters match
    :func:`thin_lens_beam` / :func:`~photonhub.analysis.gaussian_beam.gaussian_beam_source`."""
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    if not power_watts > 0.0:
        raise ValueError(f"power_watts must be > 0, got {power_watts}")
    from .eq_current_source import equivalence_current_source

    beam = thin_lens_beam(
        sim, axis=axis, na=na, wavelength_um=wavelength_um, freq_hz=freq_hz,
        source_time=source_time, n=n, center_um=center_um,
        polarization=polarization, pol_angle=pol_angle,
        focus_distance_um=focus_distance_um, pupil=pupil,
        half_w_um=half_w_um, half_v_um=half_v_um,
        window_airy_units=window_airy_units)

    # window as resolved by the beam (mirror of its own derivation)
    h_letter, v_letter = _geom.in_plane_axes(axis)
    size = sim.size_um
    if center_um is None:
        h_c = float(size[_AXES.index(h_letter)]) / 2.0
        v_c = float(size[_AXES.index(v_letter)]) / 2.0
    else:
        h_c, v_c = float(center_um[0]), float(center_um[1])
    lam_um = _resolve_wavelength(wavelength_um, freq_hz, source_time)
    n_bg = _resolve_index(sim, n)
    theta_max = math.asin(min(float(na) / n_bg, 1.0))
    spread = float(window_airy_units) * (0.61 * lam_um / float(na)) + \
        float(focus_distance_um) * math.tan(theta_max)
    hw = float(half_w_um) if half_w_um is not None else spread
    hv = float(half_v_um) if half_v_um is not None else spread

    def clip(half, c, letter):
        L = float(size[_AXES.index(letter)])
        return min(float(half), max(c, L - c))

    return equivalence_current_source(
        sim, beam, axis=axis, position_um=position_um,
        source_time=source_time, direction=direction,
        h_center_um=h_c, v_center_um=v_c,
        half_w_um=clip(hw, h_c, h_letter), half_v_um=clip(hv, v_c, v_letter),
        power_watts=float(power_watts),
        amplitude_threshold=float(amplitude_threshold))
