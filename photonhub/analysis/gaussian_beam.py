"""Gaussian-beam excitation source, a free-space / lensed-fibre beam launched
as a per-cell source-current (Huygens) source plane.

This is the excitation twin of :func:`~photonhub.analysis.mode_overlap.gaussian_mode`
(which builds an *analysis-side* Gaussian for a coupling-efficiency overlap).
Where that one is a scalar profile on its own grid, :func:`gaussian_beam` builds
the **full-vector paraxial beam on the simulation's own Yee-staggered injection
plane**, E and H sampled at their true intra-cell locations, with the beam's
complex phase, so it drops straight into
:func:`~photonhub.analysis.eq_current_source.equivalence_current_source` and
launches one-sided (forward only) exactly like a solved waveguide mode does.

Why the Huygens source plane and not a §18 :class:`~photonhub.components.sources.ModeSource`:
a Gaussian beam is only *real* (flat-phase) at its waist and at normal incidence.
Move the waist off the injection plane, or tilt the beam, and the transverse
profile picks up the wavefront-curvature, Gouy and transverse-k phases, which
the §18 wire (real signed ``profile``) cannot carry. The eq-current source plane stamps
one :class:`~photonhub.components.sources.PointDipole` per cell per component with
its own amplitude AND phase, so an arbitrary complex profile is exact on the
existing wire and engine, no schema change, CPU and GPU alike.

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
H to paraxial order (the neglected term is O(1/(k w₀)²), 3e-4 at the NA ≈ 0.06
of a lensed-fibre facet). Because E and H are supplied as a consistent
Huygens pair, the source plane radiates FORWARD only; the backward residual is the
paraxial error, not a launch artifact.

**Phasor sign.** The ``exp(-i…)`` above is the source plane's
convention: the builder drives every dipole as ``cos(ωt + arg A)``, so the
field it realizes is ``Re{A e^{+iωt}}`` and a forward-travelling wave carries
``e^{-i k·r}``. RECORDED ``field_dft`` phasors run the other way (``e^{-iωt}``:
a forward wave there is ``e^{+i k·r}``). :func:`gaussian_beam` returns the beam
in the RECORDED convention, the conjugate of the formula above, so it compares
directly with monitor data: as the reference of a
:func:`~photonhub.analysis.mode_devices.mode_monitor`, in
:func:`~photonhub.analysis.mode_overlap.mode_overlap`, or against a recorded
plane by hand. :func:`gaussian_beam_source` conjugates it back for the source plane.
The distinction is invisible for a real profile (a beam at its waist at normal
incidence); for a tilted or off-waist beam it decides which way a tilt steers
and whether an offset waist focuses. Verified on the engine; see
``test_phasor_convention.py`` and
``test_gaussian_beam.py::test_offset_waist_focuses_inside_the_domain``.

Off-normal injection tilts the whole beam frame: ``β = k cos θ`` (carried as the
mode's ``n_eff = n cos θ``, which is what phases the source plane's half-cell straddle)
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
   tightly-focused Gaussian, not of this launch, compare against exact
   angular-spectrum propagation, not against the paraxial formulas, when
   validating at small ``w₀/λ``.

Usage
-----
::

    from photonhub.analysis import gaussian_beam_source

    sim = sim.model_copy(update={"sources": gaussian_beam_source(
        geometry_sim, axis="x", position_um=2.0, source_time=pulse,
        mfd_um=10.4,                  # SMF-28 at 1550 nm
        polarization="Ez", n=1.45, power_watts=1.0)})

:func:`gaussian_beam` alone returns the beam as a
:class:`~photonhub.analysis.vector_modes.VectorMode` in the recorded
``e^{-iωt}`` convention, which is also what you want as the *reference* mode of
a :func:`~photonhub.analysis.mode_devices.mode_monitor` for a chip-to-fibre
coupling readout.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..components import PointDipole
from ..viz import _geometry as _geom
from ._constants import C0, ETA0
from .vector_modes import VectorMode
from .yee_mode import _window_center_offset, window_nodes
from .._compat import caller_stacklevel, legacy_keywords

__all__ = ["conjugate_fields", "gaussian_beam", "gaussian_beam_source", "scalar_beam"]

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

    ``mfd_um`` is the mode-field DIAMETER, the 1/e² *intensity* diameter fibre
    vendors quote (SMF-28 ≈ 10.4 µm at 1550 nm), and ``w0 = MFD/2``, the same
    relation :func:`~photonhub.analysis.mode_overlap.gaussian_mode` uses."""
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


def _resolve_pol_angle(axis: str, polarization, pol_angle_rad) -> float:
    """The linear-polarization angle (radians) in the transverse plane, measured
    from the FIRST in-plane axis toward the second. ``polarization`` names an
    in-plane E component (``"Ez"``, or bare ``"z"``) as the readable spelling of
    the two axis-aligned cases."""
    if polarization is not None and pol_angle_rad is not None:
        raise ValueError("pass at most one of polarization / pol_angle_rad")
    if pol_angle_rad is not None:
        return float(pol_angle_rad)
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
        f"injection plane; use E{h_letter} or E{v_letter} (or pol_angle_rad for a "
        "rotated linear polarization)")


def _resolve_index(sim, n) -> float:
    """The refractive index the beam propagates in: an explicit ``n``, else
    ``sqrt(eps_r)`` of the simulation background, the medium a beam launched in
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
    ``b̂₂ = k̂ × b̂₁``, at ``theta = 0`` that is exactly ``â × ĥ = v̂``, so the
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
    carries ``e^{-i k·r}``, every phase term below is NEGATED relative to the
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
    """The field 1/e radius the beam actually has AT the injection plane ,
    what the window has to cover, which is bigger than ``w0`` for an offset
    waist."""
    zR = math.pi * n * w0 * w0 / lam_um
    return w0 * math.sqrt(1.0 + (d / zR) ** 2)


def _default_center(sim, letter: str) -> float:
    """The default transverse centre of a launch on axis ``letter``: the domain
    centre, or ``0`` on a §20-folded axis, whose mirror plane sits on the
    domain's min face and is the device's centre."""
    a = _AXES.index(letter)
    sym = getattr(sim, "symmetry", None)
    if sym is not None and sym[a] != 0:
        return 0.0
    return float(sim.size_um[a]) / 2.0


def _modeled_watts(sim, axis: str, h_c: float, v_c: float, power_watts):
    """The power a sheet launch centred at ``(h_c, v_c)`` puts into the part
    the simulation models: ``power_watts`` is the whole, unfolded device's
    (NUMERICS §20.8), so a launch centred on k §20 symmetry planes carries
    ``power_watts / 2^k`` here, the same rule as a port launch."""
    if power_watts is None:
        return None
    from .mode_devices import _on_plane_factor
    h_letter, v_letter = _geom.in_plane_axes(axis)
    return float(power_watts) / _on_plane_factor(sim, ((h_letter, h_c), (v_letter, v_c)))


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
        h_c, v_c = _default_center(sim, h_letter), _default_center(sim, v_letter)
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

    Registration comes from :func:`~photonhub.analysis.yee_mode.window_nodes`, the
    SAME ladder the eigensolve and the equivalence-current sheet use, so the
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
@legacy_keywords(wavelength_um="wlen_um", pol_angle="pol_angle_rad", angle_theta="angle_theta_rad", angle_phi="angle_phi_rad")
def gaussian_beam(
    sim,
    *,
    axis: str,
    waist_um: Optional[Union[float, Sequence[float]]] = None,
    mfd_um: Optional[Union[float, Sequence[float]]] = None,
    wlen_um: Optional[float] = None,
    freq_hz: Optional[float] = None,
    source_time=None,
    center_um: Optional[Tuple[float, float]] = None,
    n: Optional[float] = None,
    polarization: Optional[str] = None,
    pol_angle_rad: Optional[float] = None,
    waist_distance_um: float = 0.0,
    angle_theta_rad: float = 0.0,
    angle_phi_rad: float = 0.0,
    direction: str = "+",
    half_w_um: Optional[float] = None,
    half_v_um: Optional[float] = None,
    window_sigmas: float = 3.0,
) -> VectorMode:
    """The analytic fundamental-Gaussian beam on ``sim``'s ``axis``-normal Yee
    plane, as a full-vector :class:`~photonhub.analysis.vector_modes.VectorMode`.

    Launch it with :func:`gaussian_beam_source` (which is this call plus the
    Huygens source plane in one step); use it directly when you want the beam object
    itself, e.g. as the reference of a
    :func:`~photonhub.analysis.mode_devices.mode_monitor` for a chip-to-fibre
    coupling readout, or of :func:`~photonhub.analysis.mode_overlap.mode_overlap`.

    Parameters
    ----------
    sim:
        The simulation whose grid, size, and §20 symmetry the beam is sampled
        on. A cheap placeholder geometry-only simulation (same grid/size/symmetry) is fine.
    axis:
        Propagation axis, ``"x"``/``"y"``/``"z"``, the injection plane's normal.
    waist_um, mfd_um:
        Beam size, exactly one of: ``waist_um`` = the field 1/e radius ``w₀``;
        ``mfd_um`` = the 1/e² intensity mode-field DIAMETER vendors quote
        (``w₀ = MFD/2``). Scalar for a round beam, ``(along h, along v)`` for an
        elliptical one (a lensed fibre), where ``(h, v)`` are the two in-plane
        axes in ascending order, ``(y, z)`` for an x-cut, ``(x, z)`` for a
        y-cut, ``(x, y)`` for a z-cut.
    wlen_um, freq_hz, source_time:
        The frequency the beam is built at, at most one of the first two; with
        neither, it is taken from ``source_time.freq0_hz`` (the pulse centre).
        Only the phase terms are wavelength-dependent, at the waist, at normal
        incidence, the Gaussian's SHAPE is wavelength-independent.
    center_um:
        Transverse beam centre as ``(h, v)`` in the in-plane-axis order above.
        Default: the domain centre. Under a §20 symmetry plane the folded axis'
        centre is ``0`` (the plane sits on the domain min face).
    n:
        Refractive index of the medium the beam propagates in. Default:
        ``sqrt(sim.background.permittivity)``, right for a beam launched in an
        unpatterned background (air ``n=1``, an oxide cladding ``n≈1.45``). Pass
        it explicitly if the launch plane sits in a different homogeneous medium.
    polarization, pol_angle_rad:
        Linear polarization, at most one of: ``polarization`` names an in-plane
        E component (``"Ez"``, or bare ``"z"``); ``pol_angle_rad`` is the angle in
        radians from the first in-plane axis toward the second. Default: E along
        the first in-plane axis.
    waist_distance_um:
        Signed distance from the beam WAIST to the injection plane, along the
        propagation direction. ``0`` (default) launches the beam at its waist
        (flat phase). **Positive** puts the waist BEHIND the plane, so the beam
        is already diverging when injected; **negative** puts it ahead, so the
        beam converges to its waist ``|waist_distance_um|`` into the domain.
    angle_theta_rad, angle_phi_rad:
        Off-normal injection (radians). ``angle_theta_rad`` tilts the beam off the
        propagation direction; ``angle_phi_rad`` is the azimuth of that tilt in the
        transverse plane, measured from the first in-plane axis. Both default to
        0 (normal incidence). The tilt is applied about the propagation
        direction implied by ``direction``, so ``angle_theta_rad`` always means "off
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
        In the RECORDED ``e^{-iωt}`` phasor convention of monitor data (a beam
        tilted toward +h carries ``e^{+i k_h h}``), so it is directly an
        overlap or mode-monitor reference; :func:`gaussian_beam_source`
        conjugates it for the source plane. ``yee_staggered``, with the six field
        components sampled at their true
        in-plane Yee locations over the window, the transverse-E pair jointly
        L2-normalized (all six scaled together, so ``E``/``H`` stay a consistent
        Huygens pair), ``n_eff = n cos(angle_theta_rad)`` (the phase constant along
        ``axis``, which is what phases the launch), and ``center_offset_um``
        recording the window's grid snap.

    Notes
    -----
    The profile is generally COMPLEX. Launch it through :func:`gaussian_beam_source`,
    which conjugates it for the source plane and carries per-cell phase. To hand it to
    :func:`~photonhub.analysis.eq_current_source.equivalence_current_source`
    directly (a per-frequency mode mapping, say), pass ``conjugate_fields(beam)``
    (``from photonhub.analysis.gaussian_beam import conjugate_fields``): the
    source plane stamps ``e^{+iωt}`` phasors. The §18 aux-line path
    (:func:`~photonhub.analysis.mode_devices.mode_source`) keeps only the real part
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
    lam_um = _resolve_wavelength(wlen_um, freq_hz, source_time)
    n_bg = _resolve_index(sim, n)
    pol = _resolve_pol_angle(axis, polarization, pol_angle_rad)
    theta = float(angle_theta_rad)
    # `direction='-'` is realized by the sheet flipping H (Poynting reversal),
    # which flips the FULL k vector — including its transverse part. Pre-rotating
    # the azimuth by pi keeps `angle_phi_rad` meaning the same thing (the azimuth of
    # the tilt about the actual launch direction) for either direction.
    phi = float(angle_phi_rad) + (0.0 if direction == "+" else math.pi)
    if not -0.5 * math.pi < theta < 0.5 * math.pi:
        raise ValueError(
            f"angle_theta_rad must be within (-pi/2, pi/2) of the launch direction, "
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

    def envelope(dh, dv):
        return _beam_at(dh, dv, k_hat=k_hat, b1=b1, b2=b2, w0h=w0h,
                        w0v=w0v, lam_um=lam_um, n=n_bg,
                        waist_distance_um=waist_distance_um)

    sheet = _assemble_beam(envelope, h_node=h_node, v_node=v_node, h_dq=h_dq, v_dq=v_dq, grids=grids,
                           h_c=h_c, v_c=v_c, dl=dl, n=n_bg, lam_um=lam_um, k_hat=k_hat, e_hat=e_hat,
                           n_eff=n_bg * math.cos(theta),
                           empty="the Gaussian beam is identically zero on the injection plane: "
                                 "check center_um against the domain (and, under a symmetry plane, "
                                 "that the beam centre sits ON the plane at coordinate 0)")
    return conjugate_fields(sheet)        # sheet e^{+iωt} -> recorded e^{-iωt}


def conjugate_fields(mode: VectorMode) -> VectorMode:
    """The mode with all six field components conjugated.

    This is the switch between the recorded ``e^{-iωt}`` phasor convention of
    :func:`gaussian_beam`, :func:`~photonhub.analysis.import_source.import_field`
    and :func:`~photonhub.analysis.thin_lens.thin_lens_beam` modes (and of
    monitor data) and the ``e^{+iωt}`` one
    :func:`~photonhub.analysis.eq_current_source.equivalence_current_source`
    stamps. Pass a beam or imported mode through it before handing it to that
    source builder directly; the ``*_source`` launchers already do. It is exact,
    so converting twice returns the original bits. Import it as
    ``from photonhub.analysis.gaussian_beam import conjugate_fields``: in
    ``photonhub.analysis``, ``gaussian_beam`` is the function, not this
    module.

    Only for beam and imported-field modes, whose ``n_eff`` is real: it
    leaves ``n_eff``, ``k_eff`` and every other field alone, so it is not a
    convention switch for a complex (lossy, bend or PML) solver mode."""
    return replace(mode, **{c: np.conj(getattr(mode, c))
                            for c in ("ex", "ey", "ez", "hx", "hy", "hz")})


def _assemble_beam(envelope, *, h_node, v_node, h_dq, v_dq, grids, h_c, v_c, dl, n, lam_um, k_hat, e_hat,
                   n_eff, empty):
    """The six Yee-staggered components of a scalar beam ``envelope(dh, dv)``
    polarized along ``e_hat`` and travelling along ``k_hat``, as a
    :class:`VectorMode`: E on its sublattices, the paired H from the scalar-limit
    admittance ``n / η₀``, the transverse-E pair jointly L2-normalized."""
    h_hat = np.cross(k_hat, e_hat)

    def sample(grid):
        H, V = grid
        return envelope(H - h_c, V - v_c)

    a_mid_node = sample(grids["mid_node"])           # E_h, H_v live here
    a_node_mid = sample(grids["node_mid"])           # E_v, H_h live here
    y0 = n / ETA0                                    # scalar-limit admittance [S]
    ex = a_mid_node * e_hat[0]
    ey = a_node_mid * e_hat[1]
    ez = sample(grids["node_node"]) * e_hat[2]
    hx = a_node_mid * h_hat[0] * y0
    hy = a_mid_node * h_hat[1] * y0
    hz = sample(grids["mid_mid"]) * h_hat[2] * y0

    # Joint L2 normalization of the transverse-E pair (the VectorMode contract),
    # applied to ALL six components so E and H remain the same Huygens pair —
    # the absolute scale is set later by `power_watts` anyway.
    norm = math.sqrt(float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2)))
    if not norm > 0.0:
        raise ValueError(empty)
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
        n_eff=n_eff,
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


def scalar_beam(
    sim,
    *,
    axis: str,
    profile,
    half_w_um: float,
    half_v_um: float,
    wlen_um: Optional[float] = None,
    freq_hz: Optional[float] = None,
    source_time=None,
    center_um: Optional[Tuple[float, float]] = None,
    n: Optional[float] = None,
    polarization: Optional[str] = None,
    pol_angle_rad: Optional[float] = None,
    direction: str = "+",
) -> VectorMode:
    """A beam with any scalar transverse profile on ``sim``'s ``axis``-normal
    Yee plane, as a :class:`~photonhub.analysis.vector_modes.VectorMode` for
    :func:`~photonhub.analysis.eq_current_source.equivalence_current_source`.

    ``profile(dh, dv)`` returns the complex scalar field ON the injection plane
    at in-plane offsets ``(dh, dv)`` from ``center_um`` (arrays of one shape, in
    µm), in the source plane's ``e^{+iωt}`` phasor convention (a flat phase for a beam
    at its waist; a field that converges into the domain has already been
    propagated back to the plane by the caller). The fibre mode a chip is
    coupled from, a top-hat, a measured near field: anything the Gaussian of
    :func:`gaussian_beam` is not. The beam is linearly polarized at normal
    incidence and the paired H comes from the scalar-limit admittance ``n/η₀``,
    exact for a plane wave and right to ``(λ/πw)²`` for a beam ``w`` wide.

    The returned mode stays in the source plane's ``e^{+iωt}`` convention, because it
    is built to be handed to
    :func:`~photonhub.analysis.eq_current_source.equivalence_current_source`
    directly. That is the opposite of :func:`gaussian_beam`, whose mode is in
    the recorded ``e^{-iωt}`` convention of monitor data: conjugate this one's
    fields before using it as an overlap or mode-monitor reference.

    ``half_w_um``/``half_v_um`` are the window's half-extents and have no
    default: the profile's reach is the caller's to know. The window is clipped
    to the domain and to a symmetry plane as the Gaussian's is. The other
    arguments are as for :func:`gaussian_beam`.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    dl = getattr(sim.grid, "dl_um", None)
    if not dl:
        raise ValueError("scalar_beam needs the grid's base dl_um")
    dl = float(dl)
    lam_um = _resolve_wavelength(wlen_um, freq_hz, source_time)
    n_bg = _resolve_index(sim, n)
    pol = _resolve_pol_angle(axis, polarization, pol_angle_rad)
    half_w, half_v = float(half_w_um), float(half_v_um)
    if not (half_w > 0.0 and half_v > 0.0):
        raise ValueError(f"half_w_um and half_v_um must be positive, got {half_w_um!r}, {half_v_um!r}")
    # The window resolver only needs the halves it is given; the spot size it
    # would derive a default from is not used.
    h_c, v_c, half_w, half_v = _resolve_window(
        sim, axis, center_um, half_w, half_v, w0h=half_w, w0v=half_v,
        lam_um=lam_um, n=n_bg, waist_distance_um=0.0, angle_theta=0.0, window_sigmas=1.0)
    h_node, v_node, (h_dq, v_dq), grids = _plane_grids(
        sim, axis, h_center=h_c, v_center=v_c, half_w=half_w, half_v=half_v, dl=dl)
    k_hat, b1, b2 = _beam_frame(0.0, 0.0 if direction == "+" else math.pi)
    e_hat = math.cos(pol) * b1 + math.sin(pol) * b2

    def envelope(dh, dv):
        out = np.asarray(profile(np.asarray(dh, dtype=float), np.asarray(dv, dtype=float)), dtype=complex)
        if out.shape != np.shape(dh):
            raise ValueError(f"profile returned shape {out.shape} for offsets of shape {np.shape(dh)}")
        return out

    return _assemble_beam(envelope, h_node=h_node, v_node=v_node, h_dq=h_dq, v_dq=v_dq, grids=grids,
                          h_c=h_c, v_c=v_c, dl=dl, n=n_bg, lam_um=lam_um, k_hat=k_hat, e_hat=e_hat,
                          n_eff=n_bg,
                          empty="the profile is identically zero on the injection plane — check center_um "
                                "and the window against the domain")


@legacy_keywords(wavelength_um="wlen_um", pol_angle="pol_angle_rad", angle_theta="angle_theta_rad", angle_phi="angle_phi_rad")
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
    pol_angle_rad: Optional[float] = None,
    waist_distance_um: float = 0.0,
    angle_theta_rad: float = 0.0,
    angle_phi_rad: float = 0.0,
    wlen_um: Optional[float] = None,
    freq_hz: Optional[float] = None,
    freqs_hz: Optional[Sequence[float]] = None,
    half_w_um: Optional[float] = None,
    half_v_um: Optional[float] = None,
    window_sigmas: float = 3.0,
    amplitude_threshold: float = 1e-6,
) -> List[PointDipole]:
    """Launch a Gaussian beam. Return a list
    of :class:`~photonhub.components.sources.PointDipole` sources to put in
    ``Simulation.sources``.

    The beam (:func:`gaussian_beam`, whose parameters this shares) is injected as
    a plane of phased dipoles
    (:func:`~photonhub.analysis.eq_current_source.equivalence_current_source`):
    ``J = n̂ × H`` on the E plane at ``position_um`` and ``M = -n̂ × E`` on the H
    nodes half a cell upstream, each dipole carrying the beam's own complex
    amplitude and phase. That is what lets an offset waist or an off-normal beam
    be exact rather than approximated, and it makes the launch one-sided
    (forward) with no TF/SF plane to keep clear of structures.

    ``power_watts`` (default 1 W) is the beam power through the injection plane,
    normalized on the engine's own discrete Poynting quadrature over the
    dipoles actually stamped, so a full-plane ``PowerMonitor`` below the
    source plane reads it back in watts and a
    transmission monitor reads an absolute fraction of the launch. Under §20
    symmetry planes it is the power of the whole, unfolded device. A beam
    centered on k planes puts ``power_watts / 2^k`` into the modeled part.
    A full-plane ``PowerMonitor`` reports ``power_watts`` (NUMERICS §20.8).
    On a one-cell periodic
    (quasi-2-D) axis it is the power through that one cell's width;
    transmission ratios are normalization-invariant either way.

    Extra parameters beyond :func:`gaussian_beam`
    ---------------------------------------------
    position_um:
        Where the injection plane sits along ``axis``. Keep it clear of the PML.
    source_time:
        The shared :class:`~photonhub.components.source_time.GaussianPulse`; its
        ``phase`` is overridden per dipole (that is where the beam profile's
        phase goes), and its ``freq0_hz`` sets the beam's frequency unless
        ``wlen_um``/``freq_hz`` says otherwise.
    freqs_hz:
        Optional broadband launch: build one beam and one dipole source plane
        per frequency. They are driven by partition-of-unity
        windowed carriers that sum back to the source pulse. Worth it only when
        the beam's profile actually moves across the band, i.e. an offset waist,
        an off-normal beam, or a dispersive ``n``; at the waist at normal
        incidence the Gaussian's shape is wavelength-independent and the extra
        source planes buy nothing. ``None`` (default) or a single entry launches the
        single band-centre beam.
    amplitude_threshold:
        Drop dipoles below this fraction of the peak (default 1e-6).
        This removes the Gaussian's far tail from the source plane.
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
    lam_um = _resolve_wavelength(wlen_um, freq_hz, source_time)
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
            "wlen_um/freq_hz to follow the pulse, or pass freqs_hz for a "
            "genuinely broadband launch.", UserWarning, stacklevel=caller_stacklevel())
    h_c, v_c, half_w, half_v = _resolve_window(
        sim, axis, center_um, half_w_um, half_v_um, w0h=w0h, w0v=w0v,
        lam_um=lam_um, n=n_bg, waist_distance_um=waist_distance_um,
        angle_theta=angle_theta_rad, window_sigmas=window_sigmas)

    beam_kwargs = dict(
        axis=axis, waist_um=waist_um, mfd_um=mfd_um,
        center_um=(h_c, v_c), n=n_bg, polarization=polarization,
        pol_angle_rad=pol_angle_rad, waist_distance_um=waist_distance_um,
        angle_theta_rad=angle_theta_rad, angle_phi_rad=angle_phi_rad, direction=direction,
        half_w_um=half_w, half_v_um=half_v,
    )
    # gaussian_beam returns the recorded e^{-iωt} convention; the sheet stamps
    # e^{+iωt}, so each beam is conjugated back (exactly) before it is stamped
    beam = conjugate_fields(gaussian_beam(sim, wlen_um=lam_um, **beam_kwargs))

    bank = None
    if freqs_hz is not None and len(list(freqs_hz)) >= 2:
        bank = {float(f): conjugate_fields(gaussian_beam(sim, freq_hz=float(f), **beam_kwargs))
                for f in freqs_hz}

    return equivalence_current_source(
        sim, beam, axis=axis, position_um=position_um,
        source_time=source_time, direction=direction,
        h_center_um=h_c, v_center_um=v_c, half_w_um=half_w, half_v_um=half_v,
        power_watts=_modeled_watts(sim, axis, h_c, v_c, power_watts),
        amplitude_threshold=float(amplitude_threshold),
        modes_by_freq=bank)
