"""Gaussian-beam excitation source — a free-space / lensed-fibre beam launched
as a per-cell equivalence-current (Huygens) sheet.

This is the excitation twin of :func:`~photonhub.plugins.mode_overlap.gaussian_mode`
(which builds an *analysis-side* Gaussian for a coupling-efficiency overlap).
Where that one is a scalar profile on its own grid, :func:`gaussian_beam` builds
the **full-vector paraxial beam on the simulation's own Yee-staggered injection
plane** — E and H sampled at their true intra-cell locations, with the beam's
complex phase — so it drops straight into
:func:`~photonhub.plugins.eq_current_source.equivalence_current_source` and
launches one-sided (forward only) exactly like a solved waveguide mode does.

Why the Huygens sheet and not a §18 :class:`~photonhub.components.sources.ModeSource`:
a Gaussian beam is only *real* (flat-phase) at its waist and at normal incidence.
Move the waist off the injection plane, or tilt the beam, and the transverse
profile picks up the wavefront-curvature, Gouy and transverse-k phases — which
the §18 wire (real signed ``profile``) cannot carry. The eq-current sheet stamps
one :class:`~photonhub.components.sources.PointDipole` per cell per component with
its own amplitude AND phase, so an arbitrary complex profile is exact on the
existing wire and engine — no schema change, CPU and GPU alike.

The beam
--------
Fundamental (TEM₀₀) Gaussian, generally elliptical. Per transverse axis *j*, with
field 1/e radius ``w0ⱼ`` at the waist and Rayleigh range ``zRⱼ = π n w0ⱼ²/λ``::

    wⱼ(z)   = w0ⱼ √(1 + (z/zRⱼ)²)          spot growth
    1/Rⱼ(z) = z / (z² + zRⱼ²)               wavefront curvature (0 at the waist)
    ψⱼ(z)   = atan(z / zRⱼ)                 Gouy phase

    E(ρ₁, ρ₂) = √(w0₁w0₂ / w₁w₂) · exp(-ρ₁²/w₁² - ρ₂²/w₂²)
                · exp(-i[ k·(r·k̂) + k(ρ₁²/2R₁ + ρ₂²/2R₂) - (ψ₁+ψ₂)/2 ])

evaluated at the beam-frame coordinates of each Yee point on the injection
plane, with ``k = 2πn/λ``. The paired magnetic field is the exact plane-wave
pairing about the beam axis, ``H = (n/η₀) k̂ × E``, which is the correct paired
H to paraxial order (the neglected term is O(1/(k w₀)²) — 3e-4 at the NA ≈ 0.06
of a lensed-fibre facet). Because E and H are supplied as a consistent
Huygens pair, the sheet radiates FORWARD only; the backward residual is the
paraxial error, not a launch artifact.

**Phasor sign.** The ``exp(-i…)`` above is not a typo. The equivalence-current
builder drives every dipole as ``cos(ωt + arg A)``, so the field it realizes is
``Re{A e^{+iωt}}`` and a forward-travelling wave carries ``e^{-i k·r}`` — the
opposite sign to the ``e^{-iωt}`` textbook Gaussian. The distinction is invisible
for a lossless guided mode (real profile, conjugation is a no-op), which is why
nothing upstream had to pin it down; for a beam it decides whether an offset
waist focuses or defocuses and which way a tilt steers. Verified on the engine —
see ``test_gaussian_beam.py``.

Note that RECORDED ``field_dft`` phasors run the other way (a forward wave there
is ``e^{+ikz}``), so comparing this beam against a recorded plane — an
angular-spectrum check, a hand-rolled overlap — needs one conjugation.
:func:`~photonhub.plugins.mode_overlap.mode_overlap` and the mode-monitor readout
already handle their own conventions; this only bites hand-written analysis.

Off-normal injection tilts the whole beam frame: ``β = k cos θ`` (carried as the
mode's ``n_eff = n cos θ``, which is what phases the sheet's half-cell straddle)
and the transverse ``k`` shows up as the ``e^{-i k (r·k̂)}`` ramp above, whose
angular-spectrum centroid is exactly ``k sin θ``. The beam's elliptical axes and
its transverse coordinates are measured in the plane perpendicular to ``k̂``, not
on the (tilted) injection plane, so a tilted beam is sampled correctly rather
than merely phase-ramped.

.. note::
   A beam is only as paraxial as ``w₀/λ`` makes it. At ``w₀ ≈ 0.8 λ`` the
   textbook ``w(z)`` under-predicts the real (exact-diffraction) spot by ~7% one
   Rayleigh range out, and a tilted beam's amplitude centroid walks at
   ``⟨kₓ/k_z⟩``, noticeably faster than ``tan θ``. Both are properties of a
   tightly-focused Gaussian, not of this launch — compare against exact
   angular-spectrum propagation, not against the paraxial formulas, when
   validating at small ``w₀/λ``.

Usage
-----
::

    from photonhub.plugins import gaussian_beam_source

    sim = sim.model_copy(update={"sources": gaussian_beam_source(
        shell, axis="x", position_um=2.0, source_time=pulse,
        mfd_um=10.4,                  # SMF-28 at 1550 nm
        polarization="Ez", n=1.45, power_watts=1.0)})

:func:`gaussian_beam` alone returns the beam as a
:class:`~photonhub.plugins.vector_modes.VectorMode`, which is also what you want
as the *reference* mode of a :func:`~photonhub.plugins.mode_devices.mode_monitor`
for a chip-to-fibre coupling readout.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..components import PointDipole
from ..viz import _geometry as _geom
from ._constants import C0, ETA0
from .vector_modes import VectorMode
from .yee_mode import _window_center_offset, window_nodes

__all__ = ["gaussian_beam", "gaussian_beam_source"]

_AXES = "xyz"


# --------------------------------------------------------------------------- #
# Parameter resolution
# --------------------------------------------------------------------------- #
def _pair(v: Union[float, Sequence[float]], what: str) -> Tuple[float, float]:
    """A scalar (round beam) or a 2-sequence (elliptical) as ``(along h, along v)``."""
    if isinstance(v, (tuple, list, np.ndarray)):
        if len(v) != 2:
            raise ValueError(f"{what} must be a scalar or a 2-tuple, got {v!r}")
        return float(v[0]), float(v[1])
    return float(v), float(v)


def _resolve_waist(waist_um, mfd_um) -> Tuple[float, float]:
    """``(w0h, w0v)`` field 1/e radii from exactly one of the two spellings.

    ``mfd_um`` is the mode-field DIAMETER — the 1/e² *intensity* diameter fibre
    vendors quote (SMF-28 ≈ 10.4 µm at 1550 nm) — and ``w0 = MFD/2``, the same
    relation :func:`~photonhub.plugins.mode_overlap.gaussian_mode` uses."""
    if (waist_um is None) == (mfd_um is None):
        raise ValueError("provide exactly one of waist_um (field 1/e radius) "
                         "or mfd_um (1/e^2 intensity mode-field diameter)")
    if waist_um is not None:
        w0h, w0v = _pair(waist_um, "waist_um")
    else:
        mh, mv = _pair(mfd_um, "mfd_um")
        w0h, w0v = 0.5 * mh, 0.5 * mv
    if not (w0h > 0.0 and w0v > 0.0):
        raise ValueError("the beam waist / mode-field diameter must be > 0")
    return w0h, w0v


def _resolve_wavelength(wavelength_um, freq_hz, source_time) -> float:
    """Free-space wavelength (µm) the beam is built at: an explicit
    ``wavelength_um``/``freq_hz``, else the pulse centre ``source_time.freq0_hz``."""
    given = [x is not None for x in (wavelength_um, freq_hz)]
    if sum(given) > 1:
        raise ValueError("pass at most one of wavelength_um / freq_hz")
    if wavelength_um is not None:
        lam = float(wavelength_um)
    elif freq_hz is not None:
        lam = C0 / float(freq_hz) * 1e6
    elif source_time is not None:
        lam = C0 / float(source_time.freq0_hz) * 1e6
    else:
        raise ValueError(
            "the beam needs a wavelength: pass wavelength_um or freq_hz")
    if not lam > 0.0:
        raise ValueError(f"wavelength must be > 0, got {lam} um")
    return lam


def _resolve_pol_angle(axis: str, polarization, pol_angle) -> float:
    """The linear-polarization angle (radians) in the transverse plane, measured
    from the FIRST in-plane axis toward the second. ``polarization`` names an
    in-plane E component (``"Ez"``, or bare ``"z"``) as the readable spelling of
    the two axis-aligned cases."""
    if polarization is not None and pol_angle is not None:
        raise ValueError("pass at most one of polarization / pol_angle")
    if pol_angle is not None:
        return float(pol_angle)
    if polarization is None:
        return 0.0                      # E along the first in-plane axis
    p = str(polarization)
    letter = p[1:] if p[:1] in ("E", "e") and len(p) == 2 else p
    letter = letter.lower()
    h_letter, v_letter = _geom.in_plane_axes(axis)
    if letter == h_letter:
        return 0.0
    if letter == v_letter:
        return 0.5 * math.pi
    raise ValueError(
        f"polarization {polarization!r} is not tangential to the {axis}-normal "
        f"injection plane; use E{h_letter} or E{v_letter} (or pol_angle for a "
        "rotated linear polarization)")


def _resolve_index(sim, n) -> float:
    """The refractive index the beam propagates in: an explicit ``n``, else
    ``sqrt(eps_r)`` of the simulation background — the medium a beam launched in
    an unpatterned region lives in."""
    if n is None:
        bg = getattr(sim, "background", None)
        eps = getattr(bg, "permittivity", None)
        n = 1.0 if eps is None else math.sqrt(float(eps))
    n = float(n)
    if not n > 0.0:
        raise ValueError(f"the background index n must be > 0, got {n}")
    return n


# --------------------------------------------------------------------------- #
# Beam frame + analytic field
# --------------------------------------------------------------------------- #
def _beam_frame(theta: float, phi: float):
    """``(k̂, b̂₁, b̂₂)`` in the right-handed in-plane frame ``(ĥ, v̂, â)``.

    ``k̂`` tilts off the plane normal ``â`` by polar angle ``theta`` toward
    azimuth ``phi`` (measured from ``ĥ``). ``b̂₁`` is ``ĥ`` projected
    perpendicular to ``k̂`` (so it degenerates to ``ĥ`` at normal incidence) and
    ``b̂₂ = k̂ × b̂₁`` — at ``theta = 0`` that is exactly ``â × ĥ = v̂``, so the
    elliptical waist axes ``(w0h, w0v)`` keep their plain meaning."""
    st, ct = math.sin(theta), math.cos(theta)
    k = np.array([st * math.cos(phi), st * math.sin(phi), ct], dtype=float)
    k /= np.linalg.norm(k)
    b1 = np.array([1.0, 0.0, 0.0]) - k[0] * k          # ĥ - (ĥ·k̂)k̂
    if np.linalg.norm(b1) < 1e-9:                       # k̂ ∥ ĥ (grazing) — use v̂
        b1 = np.array([0.0, 1.0, 0.0]) - k[1] * k
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(k, b1)
    return k, b1, b2


def _axis_beam(zp: np.ndarray, w0: float, zR: float):
    """``(w(z), 1/R(z), ψ(z))`` for one transverse axis. ``1/R`` is written as
    ``z/(z²+zR²)`` rather than ``1/(z(1+(zR/z)²))`` so the waist plane (``z=0``,
    flat phase) is exact instead of a 0/0."""
    t = zp / zR
    return (w0 * np.sqrt(1.0 + t * t),
            zp / (zp * zp + zR * zR),
            np.arctan2(zp, zR))


def _beam_at(dh: np.ndarray, dv: np.ndarray, *, k_hat, b1, b2, w0h, w0v,
             lam_um, n, waist_distance_um):
    """The complex scalar beam envelope at in-plane offsets ``(dh, dv)`` from the
    beam centre, phase-referenced so the beam centre is real-positive.

    ``dh``/``dv`` are offsets ON the injection plane; the beam-frame longitudinal
    and transverse coordinates are their projections onto ``k̂``/``b̂₁``/``b̂₂``,
    which is what makes a TILTED beam sampled correctly (its footprint on the
    plane is the true oblique section, not a phase-ramped normal-incidence spot).

    Returned in the SHEET's phasor convention (see the module docstring): the
    equivalence-current builder drives each dipole as ``cos(ωt + arg A)``, i.e.
    the realized field is ``Re{A e^{+iωt}}``, so a forward-propagating field
    carries ``e^{-i k·r}`` — every phase term below is NEGATED relative to the
    ``e^{-iωt}`` textbook form. Get this backwards and the beam still launches
    forward and still carries the right power, but it defocuses where it should
    focus and steers the wrong way; it is verified on the engine in
    ``test_gaussian_beam.py`` (an offset waist must converge, a tilt must walk
    toward ``angle_phi``)."""
    k = 2.0 * math.pi * n / lam_um                       # 1/um
    zR1 = math.pi * n * w0h * w0h / lam_um
    zR2 = math.pi * n * w0v * w0v / lam_um
    d = float(waist_distance_um)

    # r·k̂, r·b̂₁, r·b̂₂ for r = dh ĥ + dv v̂ (the plane has zero â-offset).
    zp = dh * k_hat[0] + dv * k_hat[1] + d
    r1 = dh * b1[0] + dv * b1[1]
    r2 = dh * b2[0] + dv * b2[1]

    w1, invR1, psi1 = _axis_beam(zp, w0h, zR1)
    w2, invR2, psi2 = _axis_beam(zp, w0v, zR2)
    _, _, psi1_d = _axis_beam(np.asarray(float(d)), w0h, zR1)
    _, _, psi2_d = _axis_beam(np.asarray(float(d)), w0v, zR2)

    amp = np.sqrt((w0h * w0v) / (w1 * w2)) * np.exp(-(r1 / w1) ** 2
                                                    - (r2 / w2) ** 2)
    phase = (k * (zp - d)
             + 0.5 * k * (invR1 * r1 * r1 + invR2 * r2 * r2)
             - 0.5 * (psi1 + psi2) + 0.5 * float(psi1_d + psi2_d))
    return amp * np.exp(-1j * phase)


# --------------------------------------------------------------------------- #
# Window resolution
# --------------------------------------------------------------------------- #
def _spot_on_plane(w0: float, lam_um: float, n: float, d: float) -> float:
    """The field 1/e radius the beam actually has AT the injection plane —
    what the window has to cover, which is bigger than ``w0`` for an offset
    waist."""
    zR = math.pi * n * w0 * w0 / lam_um
    return w0 * math.sqrt(1.0 + (d / zR) ** 2)


def _resolve_window(sim, axis, center_um, half_w_um, half_v_um, *, w0h, w0v,
                    lam_um, n, waist_distance_um, angle_theta, window_sigmas):
    """``(h_center, v_center, half_w, half_v)`` for the Huygens window.

    The default half-extent is ``window_sigmas`` × the beam's 1/e field radius
    ON the plane (default 3 ⇒ the field is down to ``e⁻⁹`` ≈ 1.2e-4 at the edge,
    so the truncated power is ~1e-8 of the launch), widened by ``1/cos θ`` for a
    tilted beam's oblique footprint and clipped to the domain so an
    over-generous request cannot blow up the sheet."""
    h_letter, v_letter = _geom.in_plane_axes(axis)
    size = sim.size_um
    if center_um is None:
        h_c = float(size[_AXES.index(h_letter)]) / 2.0
        v_c = float(size[_AXES.index(v_letter)]) / 2.0
    else:
        h_c, v_c = float(center_um[0]), float(center_um[1])

    stretch = 1.0 / max(math.cos(float(angle_theta)), 0.1)
    if half_w_um is None:
        half_w_um = (window_sigmas * stretch
                     * _spot_on_plane(w0h, lam_um, n, waist_distance_um))
    if half_v_um is None:
        half_v_um = (window_sigmas * stretch
                     * _spot_on_plane(w0v, lam_um, n, waist_distance_um))
    # Clip to the domain: the largest half-extent that still lands inside
    # [0, size] on the far side of the beam centre (a §20-folded axis has its
    # centre at 0, so the clip is the whole modelled half).
    def _clip(half, c, letter):
        L = float(size[_AXES.index(letter)])
        return min(float(half), max(c, L - c))

    half_w = _clip(half_w_um, h_c, h_letter)
    half_v = _clip(half_v_um, v_c, v_letter)
    if not (half_w > 0.0 and half_v > 0.0):
        raise ValueError("the beam window half-extents must be > 0")
    return h_c, v_c, half_w, half_v


def _plane_grids(sim, axis, *, h_center, v_center, half_w, half_v, dl):
    """The four Yee sampling grids of the injection-plane window.

    Registration comes from :func:`~photonhub.plugins.yee_mode.window_nodes` — the
    SAME ladder the eigensolve and the equivalence-current sheet use — so the
    analytic beam lands on exactly the cells the sheet stamps (and is clipped at
    a §20 symmetry plane the same way). The engine's in-plane Yee offsets for a
    cut normal to ``axis`` are

        E_h, H_v  at (h+½, v)      E_v, H_h  at (h, v+½)
        E_a       at (h,   v)      H_a       at (h+½, v+½)

    (the transverse E and its PAIRED H share an in-plane location and differ
    only by the half-cell straddle along the propagation axis, which the sheet
    builder supplies).  Returns ``(h_nodes, v_nodes, (h_dq, v_dq), grids)``,
    where a non-``None`` ``dq`` marks a GRADED window axis and ``grids`` holds
    the four ``(H, V)`` meshgrid pairs indexed ``[iv, ih]``."""
    h_nodes, h_dq, _h_bc, v_nodes, v_dq, _v_bc = window_nodes(
        sim, axis, h_center=h_center, half_w=half_w, v_center=v_center,
        half_v=half_v, dl=dl)
    h_node = np.asarray(h_nodes, dtype=float)
    v_node = np.asarray(v_nodes, dtype=float)
    h_mid = h_node + 0.5 * (np.asarray(h_dq, dtype=float)
                            if h_dq is not None else dl)
    v_mid = v_node + 0.5 * (np.asarray(v_dq, dtype=float)
                            if v_dq is not None else dl)

    def mesh(hs, vs):
        return np.meshgrid(hs, vs, indexing="xy")        # -> [iv, ih]

    return h_node, v_node, (h_dq, v_dq), {
        "mid_node": mesh(h_mid, v_node),    # E_h / H_v
        "node_mid": mesh(h_node, v_mid),    # E_v / H_h
        "node_node": mesh(h_node, v_node),  # E_a
        "mid_mid": mesh(h_mid, v_mid),      # H_a
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def gaussian_beam(
    sim,
    *,
    axis: str,
    waist_um: Optional[Union[float, Sequence[float]]] = None,
    mfd_um: Optional[Union[float, Sequence[float]]] = None,
    wavelength_um: Optional[float] = None,
    freq_hz: Optional[float] = None,
    source_time=None,
    center_um: Optional[Tuple[float, float]] = None,
    n: Optional[float] = None,
    polarization: Optional[str] = None,
    pol_angle: Optional[float] = None,
    waist_distance_um: float = 0.0,
    angle_theta: float = 0.0,
    angle_phi: float = 0.0,
    direction: str = "+",
    half_w_um: Optional[float] = None,
    half_v_um: Optional[float] = None,
    window_sigmas: float = 3.0,
) -> VectorMode:
    """The analytic fundamental-Gaussian beam on ``sim``'s ``axis``-normal Yee
    plane, as a full-vector :class:`~photonhub.plugins.vector_modes.VectorMode`.

    Launch it with :func:`gaussian_beam_source` (which is this call plus the
    Huygens sheet in one step); use it directly when you want the beam object
    itself — e.g. as the reference of a
    :func:`~photonhub.plugins.mode_devices.mode_monitor` for a chip-to-fibre
    coupling readout, or of :func:`~photonhub.plugins.mode_overlap.mode_overlap`.

    Parameters
    ----------
    sim:
        The simulation whose grid, size, and §20 symmetry the beam is sampled
        on. A cheap placeholder shell (same grid/size/symmetry) is fine.
    axis:
        Propagation axis, ``"x"``/``"y"``/``"z"`` — the injection plane's normal.
    waist_um, mfd_um:
        Beam size, exactly one of: ``waist_um`` = the field 1/e radius ``w₀``;
        ``mfd_um`` = the 1/e² intensity mode-field DIAMETER vendors quote
        (``w₀ = MFD/2``). Scalar for a round beam, ``(along h, along v)`` for an
        elliptical one (a lensed fibre), where ``(h, v)`` are the two in-plane
        axes in ascending order — ``(y, z)`` for an x-cut, ``(x, z)`` for a
        y-cut, ``(x, y)`` for a z-cut.
    wavelength_um, freq_hz, source_time:
        The frequency the beam is built at, at most one of the first two; with
        neither, it is taken from ``source_time.freq0_hz`` (the pulse centre).
        Only the phase terms are wavelength-dependent — at the waist, at normal
        incidence, the Gaussian's SHAPE is wavelength-independent.
    center_um:
        Transverse beam centre as ``(h, v)`` in the in-plane-axis order above.
        Default: the domain centre. Under a §20 symmetry plane the folded axis'
        centre is ``0`` (the plane sits on the domain min face).
    n:
        Refractive index of the medium the beam propagates in. Default:
        ``sqrt(sim.background.permittivity)`` — right for a beam launched in an
        unpatterned background (air ``n=1``, an oxide cladding ``n≈1.45``). Pass
        it explicitly if the launch plane sits in a different homogeneous medium.
    polarization, pol_angle:
        Linear polarization, at most one of: ``polarization`` names an in-plane
        E component (``"Ez"``, or bare ``"z"``); ``pol_angle`` is the angle in
        radians from the first in-plane axis toward the second. Default: E along
        the first in-plane axis.
    waist_distance_um:
        Signed distance from the beam WAIST to the injection plane, along the
        propagation direction. ``0`` (default) launches the beam at its waist
        (flat phase). **Positive** puts the waist BEHIND the plane, so the beam
        is already diverging when injected; **negative** puts it ahead, so the
        beam converges to its waist ``|waist_distance_um|`` into the domain.
    angle_theta, angle_phi:
        Off-normal injection (radians). ``angle_theta`` tilts the beam off the
        propagation direction; ``angle_phi`` is the azimuth of that tilt in the
        transverse plane, measured from the first in-plane axis. Both default to
        0 (normal incidence). The tilt is applied about the propagation
        direction implied by ``direction``, so ``angle_theta`` always means "off
        the launch direction".
    direction:
        ``"+"`` (default) launches toward increasing ``axis``, ``"-"`` toward
        decreasing.
    half_w_um, half_v_um, window_sigmas:
        Half-extents of the sampled window. Default: ``window_sigmas`` (3) times
        the beam's 1/e field radius ON the plane, i.e. the field is down to
        ``e⁻⁹`` at the window edge, so truncation costs ~1e-8 of the power.
        Widen it (or set the halves explicitly) if you deliberately clip the
        beam; the window is clipped to the domain and to any symmetry plane.

    Returns
    -------
    VectorMode
        ``yee_staggered``, with the six field components sampled at their true
        in-plane Yee locations over the window, the transverse-E pair jointly
        L2-normalized (all six scaled together, so ``E``/``H`` stay a consistent
        Huygens pair), ``n_eff = n cos(angle_theta)`` (the phase constant along
        ``axis``, which is what phases the launch), and ``center_offset_um``
        recording the window's grid snap.

    Notes
    -----
    The profile is generally COMPLEX. Launch it through :func:`gaussian_beam_source`
    (or :func:`~photonhub.plugins.eq_current_source.equivalence_current_source`),
    which carries per-cell phase. The §18 aux-line path
    (:func:`~photonhub.plugins.mode_devices.mode_source`) keeps only the real part
    and would silently mis-launch anything but an at-waist, normal-incidence beam.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    dl = getattr(sim.grid, "dl_um", None)
    if not dl:
        raise ValueError("gaussian_beam needs the grid's base dl_um")
    dl = float(dl)

    w0h, w0v = _resolve_waist(waist_um, mfd_um)
    lam_um = _resolve_wavelength(wavelength_um, freq_hz, source_time)
    n_bg = _resolve_index(sim, n)
    pol = _resolve_pol_angle(axis, polarization, pol_angle)
    theta = float(angle_theta)
    # `direction='-'` is realized by the sheet flipping H (Poynting reversal),
    # which flips the FULL k vector — including its transverse part. Pre-rotating
    # the azimuth by pi keeps `angle_phi` meaning the same thing (the azimuth of
    # the tilt about the actual launch direction) for either direction.
    phi = float(angle_phi) + (0.0 if direction == "+" else math.pi)
    if not -0.5 * math.pi < theta < 0.5 * math.pi:
        raise ValueError(
            f"angle_theta must be within (-pi/2, pi/2) of the launch direction, "
            f"got {theta}: a beam at or past grazing does not cross the plane")

    h_c, v_c, half_w, half_v = _resolve_window(
        sim, axis, center_um, half_w_um, half_v_um, w0h=w0h, w0v=w0v,
        lam_um=lam_um, n=n_bg, waist_distance_um=waist_distance_um,
        angle_theta=theta, window_sigmas=window_sigmas)
    h_node, v_node, (h_dq, v_dq), grids = _plane_grids(
        sim, axis, h_center=h_c, v_center=v_c, half_w=half_w, half_v=half_v,
        dl=dl)

    k_hat, b1, b2 = _beam_frame(theta, phi)
    e_hat = math.cos(pol) * b1 + math.sin(pol) * b2
    h_hat = np.cross(k_hat, e_hat)                   # = cos(pol) b̂₂ - sin(pol) b̂₁

    def envelope(grid):
        H, V = grid
        return _beam_at(H - h_c, V - v_c, k_hat=k_hat, b1=b1, b2=b2, w0h=w0h,
                        w0v=w0v, lam_um=lam_um, n=n_bg,
                        waist_distance_um=waist_distance_um)

    a_mid_node = envelope(grids["mid_node"])         # E_h, H_v live here
    a_node_mid = envelope(grids["node_mid"])         # E_v, H_h live here
    y0 = n_bg / ETA0                                 # scalar-limit admittance [S]
    ex = a_mid_node * e_hat[0]
    ey = a_node_mid * e_hat[1]
    ez = envelope(grids["node_node"]) * e_hat[2]
    hx = a_node_mid * h_hat[0] * y0
    hy = a_mid_node * h_hat[1] * y0
    hz = envelope(grids["mid_mid"]) * h_hat[2] * y0

    # Joint L2 normalization of the transverse-E pair (the VectorMode contract),
    # applied to ALL six components so E and H remain the same Huygens pair —
    # the absolute scale is set later by `power_watts` anyway.
    norm = math.sqrt(float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2)))
    if not norm > 0.0:
        raise ValueError(
            "the Gaussian beam is identically zero on the injection plane — "
            "check center_um against the domain (and, under a symmetry plane, "
            "that the beam centre sits ON the plane at coordinate 0)")
    ex, ey, ez, hx, hy, hz = (f / norm for f in (ex, ey, ez, hx, hy, hz))

    nv, nh = ex.shape
    graded = h_dq is not None or v_dq is not None
    # Window placement metadata, the SAME two forms yee_mode._window_placement
    # records: the uniform ladder's centre from its pitch, a graded ladder's from
    # its own first/last node (whose midpoint is not lo + (n-1)dl/2).
    if graded:
        off = (0.5 * float(h_node[0] + h_node[-1]) - h_c,
               0.5 * float(v_node[0] + v_node[-1]) - v_c)
    else:
        off = _window_center_offset(float(h_node[0]), float(v_node[0]), nh, nv,
                                    dl, h_c, v_c)
    return VectorMode(
        n_eff=n_bg * math.cos(theta),
        n_group=None,
        ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz,
        wavelength_um=lam_um,
        dl_x_um=dl,
        dl_y_um=dl,
        center_offset_um=off,
        yee_staggered=True,
        x_coords_um=(h_node - h_c) if graded else None,
        y_coords_um=(v_node - v_c) if graded else None,
    )


def gaussian_beam_source(
    sim,
    *,
    axis: str,
    position_um: float,
    source_time,
    waist_um: Optional[Union[float, Sequence[float]]] = None,
    mfd_um: Optional[Union[float, Sequence[float]]] = None,
    direction: str = "+",
    power_watts: float = 1.0,
    center_um: Optional[Tuple[float, float]] = None,
    n: Optional[float] = None,
    polarization: Optional[str] = None,
    pol_angle: Optional[float] = None,
    waist_distance_um: float = 0.0,
    angle_theta: float = 0.0,
    angle_phi: float = 0.0,
    wavelength_um: Optional[float] = None,
    freq_hz: Optional[float] = None,
    freqs_hz: Optional[Sequence[float]] = None,
    half_w_um: Optional[float] = None,
    half_v_um: Optional[float] = None,
    window_sigmas: float = 3.0,
    amplitude_threshold: float = 1e-6,
) -> List[PointDipole]:
    """Launch a Gaussian beam — the one-call excitation source. Returns the LIST
    of :class:`~photonhub.components.sources.PointDipole` to put in
    ``Simulation.sources``.

    The beam (:func:`gaussian_beam`, whose parameters this shares) is injected as
    a per-cell equivalence-current Huygens sheet
    (:func:`~photonhub.plugins.eq_current_source.equivalence_current_source`):
    ``J = n̂ × H`` on the E plane at ``position_um`` and ``M = -n̂ × E`` on the H
    nodes half a cell upstream, each dipole carrying the beam's own complex
    amplitude and phase. That is what lets an offset waist or an off-normal beam
    be exact rather than approximated, and it makes the launch one-sided
    (forward) with no TF/SF plane to keep clear of structures.

    ``power_watts`` (default 1 W) is the beam power through the injection plane,
    normalized on the engine's own discrete Poynting quadrature — so a
    transmission monitor reads an absolute fraction of the launch. Under a §20
    symmetry plane it is the power into the MODELED (half/quarter) domain, the
    same convention every other launch here uses; transmission ratios are
    normalization-invariant either way.

    Extra parameters beyond :func:`gaussian_beam`
    ---------------------------------------------
    position_um:
        Where the injection plane sits along ``axis``. Keep it clear of the PML.
    source_time:
        The shared :class:`~photonhub.components.source_time.GaussianPulse`; its
        ``phase`` is overridden per dipole (that is where the beam profile's
        phase goes), and its ``freq0_hz`` sets the beam's frequency unless
        ``wavelength_um``/``freq_hz`` says otherwise.
    freqs_hz:
        Optional broadband launch (the ``num_freqs`` analogue): build one beam —
        and one dipole sheet — per frequency, driven by partition-of-unity
        windowed carriers that sum back to the source pulse. Worth it only when
        the beam's profile actually moves across the band, i.e. an offset waist,
        an off-normal beam, or a dispersive ``n``; at the waist at normal
        incidence the Gaussian's shape is wavelength-independent and the extra
        sheets buy nothing. ``None`` (default) or a single entry launches the
        single band-centre beam.
    amplitude_threshold:
        Dipoles below this fraction of the peak are dropped (default 1e-6) —
        the Gaussian's far tail, which is why the sheet stays a few 10 k dipoles
        rather than the whole plane.
    """
    if not power_watts > 0.0:
        raise ValueError(f"power_watts must be > 0, got {power_watts}")
    from .eq_current_source import equivalence_current_source

    # Resolve the window ONCE, here, and hand the resolved extents to every beam
    # we build: the band-centre beam, each broadband sheet, and the sheet builder
    # itself must all derive their node ladder from the SAME (centre, half)
    # arguments, or the dipoles land on a differently-snapped grid than the
    # profile they carry. Passing the halves through is idempotent (they are
    # already domain-clipped).
    w0h, w0v = _resolve_waist(waist_um, mfd_um)
    lam_um = _resolve_wavelength(wavelength_um, freq_hz, source_time)
    n_bg = _resolve_index(sim, n)
    # The sheet phases its half-cell straddle at the PULSE centre, so a beam
    # frozen at a materially different wavelength is launched slightly detuned
    # (and, more to the point, profiled for light the pulse is not centred on).
    # A broadband `freqs_hz` bank is exempt: each sheet is phased at its own
    # frequency by the builder.
    lam_pulse = C0 / float(source_time.freq0_hz) * 1e6
    if freqs_hz is None and abs(lam_um - lam_pulse) > 0.02 * lam_pulse:
        warnings.warn(
            f"the beam is built at {lam_um:.4g} um but the pulse is centred at "
            f"{lam_pulse:.4g} um: the Huygens sheet phases its half-cell straddle "
            "at the PULSE centre, so the launch is detuned from the beam. Drop "
            "wavelength_um/freq_hz to follow the pulse, or pass freqs_hz for a "
            "genuinely broadband launch.", UserWarning, stacklevel=2)
    h_c, v_c, half_w, half_v = _resolve_window(
        sim, axis, center_um, half_w_um, half_v_um, w0h=w0h, w0v=w0v,
        lam_um=lam_um, n=n_bg, waist_distance_um=waist_distance_um,
        angle_theta=angle_theta, window_sigmas=window_sigmas)

    beam_kwargs = dict(
        axis=axis, waist_um=waist_um, mfd_um=mfd_um,
        center_um=(h_c, v_c), n=n_bg, polarization=polarization,
        pol_angle=pol_angle, waist_distance_um=waist_distance_um,
        angle_theta=angle_theta, angle_phi=angle_phi, direction=direction,
        half_w_um=half_w, half_v_um=half_v,
    )
    beam = gaussian_beam(sim, wavelength_um=lam_um, **beam_kwargs)

    bank = None
    if freqs_hz is not None and len(list(freqs_hz)) >= 2:
        bank = {float(f): gaussian_beam(sim, freq_hz=float(f), **beam_kwargs)
                for f in freqs_hz}

    return equivalence_current_source(
        sim, beam, axis=axis, position_um=position_um,
        source_time=source_time, direction=direction,
        h_center_um=h_c, v_center_um=v_c, half_w_um=half_w, half_v_um=half_v,
        power_watts=float(power_watts),
        amplitude_threshold=float(amplitude_threshold),
        modes_by_freq=bank)
