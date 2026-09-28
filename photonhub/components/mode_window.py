"""The mode window of a waveguide port, sized from the guide alone.

A port reads a guided mode inside a window around the guide. At the -30 dB
bound the mode's intensity |E|^2 at every window edge is 30 dB below its peak
(1e-3 of the peak intensity). Any guided mode decays into a cladding of index
``n_clad`` at least as fast as ``gamma = k0 * sqrt(n_eff**2 - n_clad**2)``, so
the bound puts the window edge

    ln(1000) / (2 * gamma)

beyond each core face. The default pad is 1.2 times that bound (decision
0017). With the mode's solved ``n_eff`` the truncation then
costs a transmission reading under 0.001 dB per port and the edges sit 42 to
51 dB below the peak (silicon TE0 and TM0, silicon nitride TE0, 1550 nm).
With the effective-index estimate a TM mode's ``n_eff`` comes out about 3 %
high, so its pad is a little short: about 0.002 dB per port, edges from
-39 dB.

``n_eff`` comes from the effective-index method, two slab solves with no 2D
mode solve: the slab of the guide's thickness gives ``n_slab``, and the slab of
the guide's width, with ``n_slab`` as its core, gives ``n_eff``. For TE the
thickness slab is solved TE and the width slab TM; for TM the other way round.
The one ``n_eff`` sets the pad in both directions.
"""

from __future__ import annotations

import math
import re
import warnings

from ..constants import c0

__all__ = ["MODE_WINDOW_FACTOR", "mode_window_um"]

_MODE = re.compile(r"^(TE|TM)(\d+)$")
# The bound: |E|^2 at the window edge this many dB below the mode's peak.
_EDGE_DB = 30.0
#: The default pad as a multiple of the -30 dB bound.
MODE_WINDOW_FACTOR = 1.2
# Pads are rounded up to this step, which covers most of the effective-index
# method's n_eff error for TE; a TM estimate can still leave the pad short.
_STEP_UM = 0.05


def _index(material, wlen_um: float) -> float:
    """A refractive index given as a number, a library material or a
    :class:`~photonhub.Medium` (its real permittivity at ``wlen_um``)."""
    n = getattr(material, "n", None)
    if callable(n):
        return float(n(wlen_um))
    at_hz = getattr(material, "permittivity_at_hz", None)
    if callable(at_hz):
        return math.sqrt(max(float(at_hz(c0 / (wlen_um * 1e-6))), 1.0))
    return float(material)


def _slab_neff(thickness_um: float, n_core: float, n_low: float, n_high: float,
               wlen_um: float, polarization: str, order: int) -> float:
    """The guided index of mode ``order`` of a slab of ``n_core`` between
    claddings ``n_low`` and ``n_high``, from the slab's transverse resonance
    condition, solved by bisection. Raises when the mode is cut off."""
    k0 = 2.0 * math.pi / wlen_um
    n_clad = max(n_low, n_high)
    te = polarization == "TE"

    def mismatch(n: float) -> float:
        kx = k0 * math.sqrt(max(n_core ** 2 - n ** 2, 0.0))
        g_low = k0 * math.sqrt(max(n ** 2 - n_low ** 2, 0.0))
        g_high = k0 * math.sqrt(max(n ** 2 - n_high ** 2, 0.0))
        r_low = 1.0 if te else (n_core / n_low) ** 2
        r_high = 1.0 if te else (n_core / n_high) ** 2
        return (kx * thickness_um - math.atan2(r_low * g_low, kx)
                - math.atan2(r_high * g_high, kx) - order * math.pi)

    lo, hi = n_clad, n_core
    if mismatch(lo + 1e-12 * n_core) <= 0.0:
        raise ValueError(
            f"a {thickness_um:g} um slab of index {n_core:g} in {n_clad:g} guides no "
            f"{polarization}{order} mode at {wlen_um:g} um")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mismatch(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _effective_index(width_um: float, thickness_um: float, n_core: float, n_clad: float,
                     n_clad_top: float, wlen_um: float, family: str, order: int) -> float:
    """The effective-index method: thickness slab first, then width slab."""
    first, second = ("TE", "TM") if family == "TE" else ("TM", "TE")
    n_slab = _slab_neff(thickness_um, n_core, n_clad, n_clad_top, wlen_um, first, 0)
    side = max(n_clad, n_clad_top)
    return _slab_neff(width_um, n_slab, side, side, wlen_um, second, order)


def _pad_um(n_eff: float, n_clad: float, wlen_um: float, factor: float = 1.0) -> float:
    """``factor`` times the -30 dB bound on the pad beyond a core face, before
    rounding."""
    gamma = 2.0 * math.pi / wlen_um * math.sqrt(n_eff ** 2 - n_clad ** 2)
    return factor * math.log(10.0 ** (_EDGE_DB / 10.0)) / (2.0 * gamma)


def _round_up(value_um: float) -> float:
    return round(math.ceil(value_um / _STEP_UM - 1e-9) * _STEP_UM, 10)


def mode_window_um(width_um: float, thickness_um: float, n_core, n_clad, wlen_um: float,
                   mode: str = "TE0", n_clad_top=None, *, factor: float = MODE_WINDOW_FACTOR,
                   n_eff: float | None = None) -> tuple[float, float]:
    """The mode window of a port on a rectangular guide, as the two half-extents
    ``(half_width_um, half_thickness_um)`` that :class:`~photonhub.Port`'s
    ``window_um`` takes.

    Each half-extent is the core's half-size plus a cladding pad of ``factor``
    times the -30 dB bound, the distance at which the mode's intensity has
    fallen 30 dB below its peak:
    ``pad = factor * ln(1000) / (2 k0 sqrt(n_eff^2 - n_clad^2))``, rounded up
    to the next 0.05 um. With a solved ``n_eff``, the default ``factor=1.2``
    keeps the truncation error of a port reading under 0.001 dB;
    ``factor=1.0`` is the bound itself (about 0.001 to 0.003 dB per port).

    ``n_eff`` is, unless given, the effective-index estimate from two slab
    solves (the guide's thickness, then its width) in the polarization of
    ``mode``; the mode's order counts across the width. Pass a solved
    ``n_eff`` when one is at hand: the estimate can overshoot (a silicon
    strip's TM0 by about 3 %), which shortens the pad by about 7 % and leaves
    about 0.002 dB per port. The same ``n_eff`` sizes both directions. A mode
    near cut-off has a pad of several wavelengths; that draws a warning.

    ``wlen_um`` is the longest wavelength the port reads: the mode spreads as
    the wavelength grows, so the window that holds it there holds it across
    the band. ``n_core`` and ``n_clad`` are refractive indices or library
    materials (``ph.materials.Si``), read at ``wlen_um``. ``n_clad_top`` is
    the cladding above the core when it differs from the one below (a strip in
    air on oxide). The window is symmetric about the guide, so each direction
    takes the larger pad of its sides, the one in the higher cladding index.

    For a multimode port, pass the highest-order mode it reads. In a coupler,
    place the port where the guides are more than two pads apart, or give a
    window that encloses both guides.

    >>> mode_window_um(0.5, 0.22, 3.476, 1.444, 1.55)
    (0.8, 0.66)
    """
    wlen = float(wlen_um)
    width, thickness = float(width_um), float(thickness_um)
    if not (wlen > 0.0 and width > 0.0 and thickness > 0.0):
        raise ValueError("width_um, thickness_um and wlen_um must be > 0")
    if not float(factor) >= 1.0:
        raise ValueError(f"factor must be at least 1 (the -30 dB bound), got {factor!r}")
    m = _MODE.match(str(mode).upper())
    if not m:
        raise ValueError(f"mode must read like 'TE0' or 'TM1', got {mode!r}")
    family, order = m.group(1), int(m.group(2))
    core = _index(n_core, wlen)
    below = _index(n_clad, wlen)
    above = below if n_clad_top is None else _index(n_clad_top, wlen)
    if not core > max(below, above) >= 1.0:
        raise ValueError(
            f"n_core ({core:g}) must exceed both claddings ({below:g}, {above:g}), "
            "and a cladding index is at least 1")
    if n_eff is None:
        try:
            n_eff = _effective_index(width, thickness, core, below, above, wlen, family, order)
        except ValueError as exc:
            raise ValueError(f"a {width:g} x {thickness:g} um guide of index {core:g} guides no {family}{order} "
                             f"mode at {wlen:g} um (effective-index estimate: {exc})") from None
    elif not max(below, above) < float(n_eff) < core:
        raise ValueError(f"n_eff ({n_eff:g}) must lie between the cladding and the core index")
    pad = _round_up(_pad_um(float(n_eff), max(below, above), wlen, float(factor)))
    if pad > 3.0 * wlen:
        warnings.warn(f"the {family}{order} mode is close to cut-off (n_eff {float(n_eff):.4g} against a cladding "
                      f"of {max(below, above):.4g}): its window pad is {pad:g} um, {pad / wlen:.1f} wavelengths",
                      UserWarning, stacklevel=2)
    return round(0.5 * width + pad, 10), round(0.5 * thickness + pad, 10)
