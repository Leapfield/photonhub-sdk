"""Finite-difference eigenmode (FDE) solver for straight waveguides — CPU only.

This is the Phase-1 mode solver: it computes the guided eigenmodes of a *2-D
dielectric cross-section* that is **invariant along the propagation axis**
(a straight waveguide). It promotes the proven numpy reference solver in
``benchmarks/waveguide/mode_solver.py`` — which cross-validated a silicon strip
waveguide against the FDTD engine (FDTD n_eff = 2.700 vs mode-solver 2.718,
0.6 %; TE0 |Ex| profile overlap 0.98) — into a documented, frozen public API.

Physics / method
================
For a z-invariant cross-section eps(x, y), guided modes have the separable form

    E(x, y, z, t) = e(x, y) * exp(-i (omega t - beta z))
                  = e(x, y) * exp(-i omega t) * exp(+i beta z),

(the engine's ``e^{-i omega t}`` time convention — a forward wave's phasor
advances as ``e^{+i beta z}``; see ``mode_overlap.py``), with propagation
constant ``beta`` and modal index ``n_eff = beta / k0``
(``k0 = 2*pi/lambda``). The dominant transverse field component obeys a
2-D Helmholtz eigenproblem ``A e = beta^2 e``, discretized with second-order
finite differences and Dirichlet walls (the mode decays into the cladding, so
the computational window must pad the core with enough cladding for the field
to die off before the boundary). Two operators are provided:

* ``"scalar"``  : d2/dx2 + d2/dy2 + k0^2 eps
                  — overestimates n_eff for high-contrast SOI.
* ``"semivec"`` : d/dx[(1/eps) d/dx (eps .)] + d2/dy2 + k0^2 eps
                  — semi-vectorial quasi-TE, ``Ex``-major; the flux-conservative
                  x-stencil keeps Dx = eps*Ex continuous across vertical
                  interfaces, which is what makes the high-contrast n_eff right.
                  This is the default and the validated operator.

The largest eigenvalues ``beta^2`` are the best-confined (highest-index) modes;
the fundamental is the global maximum. Grids are kept modest (a dense operator
is O(N^2) memory, O(N^3) to diagonalize) — a ~60x60 window is plenty for a
single-mode strip.

SCOPE — STRAIGHT WAVEGUIDES ONLY (bent-mode exclusion)
======================================================
**This solver models only straight (translationally-invariant) waveguides.**
Bent / curved waveguides are *out of scope* for Phase 1 and are not supported.
Physically, a bend of radius ``R`` introduces an effective index gradient
``n_eff(x) ~= n_eff(0) * (1 + x / R)`` across the cross-section (the conformal /
equivalent-straight-waveguide transform): the mode shifts toward the outer wall
and acquires radiation (bending) loss. None of that is modeled here — the
operator above assumes a flat cross-section with no curvature term.

Error bound if you misuse this solver on a bend: ignoring curvature, the
*relative* n_eff error scales like the ratio of the mode's lateral extent
``w`` to the bend radius, roughly

    |Delta n_eff| / n_eff  ~  (w / R)         (first order),

and the bending loss (entirely unmodeled) grows ~exp(-R) for tight bends. As a
rule of thumb the straight-waveguide n_eff is good to well under 1 % only while
``R >> w`` (e.g. ``R`` of tens of microns for a sub-micron strip); for the
micron-scale radii used in ring resonators it is *not* trustworthy. Use a
dedicated bent-mode solver there. :meth:`ModeSolver.solve` will refuse to run
if a nonzero ``bend_radius_um`` is supplied (see that method).

Public API (FROZEN — Phase 1)
=============================
Pinned by ``tests/test_modesolver.py``; treat as a stable contract.

* ``ModeSolver(eps, dl_x_um, dl_y_um, wavelength_um)``
      Construct from a raw permittivity cross-section ``eps`` (a real 2-D numpy
      array indexed ``[iy, ix]``, i.e. row = y, col = x) and the *uniform*
      transverse grid spacings + free-space wavelength (microns).
* ``ModeSolver.from_rectangular_core(...)`` (classmethod)
      Convenience builder that rasterizes a centered rectangular core in a
      uniform cladding onto a square grid (mirrors the validated benchmark).
* ``ModeSolver.solve(num_modes=1, polarization="TE", n_guess=None) -> tuple[Mode, ...]``
      Returns the ``num_modes`` best-confined guided modes, highest n_eff first.
* ``Mode`` (frozen dataclass): ``.n_eff: float``, ``.field: np.ndarray``
      (the dominant-component profile, shape ``(ny, nx)``, L2-normalized),
      ``.wavelength_um``, ``.polarization``, plus helpers ``.field_dataarray()``
      (xarray, real-space x/y coords in microns) and
      ``.core_fraction(...)`` (confinement in a bounding box).

CPU / numpy only — no scipy, no GPU. Dispersive / magnetic / anisotropic media
and PML-backed leaky modes are out of scope for Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import xarray as xr

from ._constants import C0 as _C0

__all__ = ["Mode", "ModeSolver"]

# Polarization selector for the semi-vectorial operator. "TE" is the validated
# Ex-major quasi-TE branch (flux-conservative x-derivative); "TM" swaps the
# flux-conservative direction to y (Ey-major quasi-TM); "scalar" drops the
# vectorial correction entirely (over-estimates high-contrast n_eff).
PolarizationName = Literal["TE", "TM", "scalar"]


def _odd(n: int) -> int:
    """Smallest odd integer ``>= n`` (cell-count helper so a cell center sits on
    the cross-section origin — see :meth:`ModeSolver.from_rectangular_core`)."""
    return n if n % 2 == 1 else n + 1


@dataclass(frozen=True)
class Mode:
    """One guided eigenmode of a straight-waveguide cross-section.

    Attributes
    ----------
    n_eff:
        Modal effective index ``beta / k0`` (dimensionless, real). For a
        properly confined guided mode this lies between the cladding and core
        refractive indices.
    field:
        The dominant transverse field component on the cross-section, a real
        ``float64`` array of shape ``(ny, nx)`` indexed ``[iy, ix]`` (row = y,
        column = x), L2-normalized (``sum(field**2) == 1``) and sign-fixed so
        the lobe is positive. This is ``Ex`` for ``"TE"``/``"scalar"`` and
        ``Ey`` for ``"TM"``.
    wavelength_um:
        Free-space wavelength the mode was solved at (microns).
    polarization:
        Which operator branch produced this mode (``"TE"``/``"TM"``/``"scalar"``).
    dl_x_um, dl_y_um:
        Transverse grid spacings (microns), carried for real-space helpers.
    """

    n_eff: float
    field: np.ndarray
    wavelength_um: float
    polarization: PolarizationName
    dl_x_um: float
    dl_y_um: float

    @property
    def shape(self) -> Tuple[int, int]:
        """``(ny, nx)`` of the field profile."""
        return tuple(self.field.shape)  # type: ignore[return-value]

    def field_dataarray(self) -> xr.DataArray:
        """The field profile as an :class:`xarray.DataArray` with real-space
        ``x``/``y`` coordinates in microns (origin at the cross-section center,
        matching :meth:`ModeSolver.from_rectangular_core`). Dims ``("y", "x")``."""
        ny, nx = self.field.shape
        xs = (np.arange(nx) - (nx - 1) / 2.0) * self.dl_x_um
        ys = (np.arange(ny) - (ny - 1) / 2.0) * self.dl_y_um
        comp = "Ey" if self.polarization == "TM" else "Ex"
        return xr.DataArray(
            self.field,
            dims=("y", "x"),
            coords={"y": ys, "x": xs},
            name=comp,
            attrs={
                "n_eff": self.n_eff,
                "wavelength_um": self.wavelength_um,
                "polarization": self.polarization,
                "component": comp,
            },
        )

    def core_fraction(self, core_w_um: float, core_h_um: float) -> float:
        """Fraction of ``|field|^2`` inside a centered ``core_w_um x core_h_um``
        bounding box — a simple confinement metric (1.0 = fully confined). The
        box is centered on the cross-section, matching
        :meth:`ModeSolver.from_rectangular_core`."""
        ny, nx = self.field.shape
        xs = np.abs(np.arange(nx) - (nx - 1) / 2.0) * self.dl_x_um
        ys = np.abs(np.arange(ny) - (ny - 1) / 2.0) * self.dl_y_um
        inside = (ys[:, None] <= core_h_um / 2.0 + 1e-12) & \
                 (xs[None, :] <= core_w_um / 2.0 + 1e-12)
        p = self.field ** 2
        return float(p[inside].sum() / p.sum())


class ModeSolver:
    """Finite-difference eigenmode solver for a straight waveguide cross-section.

    Construct from a raw permittivity cross-section and the transverse grid, then
    call :meth:`solve`. See the module docstring for the physics, the operator
    definitions, and — importantly — the **straight-waveguide-only** scope with
    the bent-mode error bound.

    Parameters
    ----------
    eps:
        Real relative-permittivity cross-section, a 2-D array indexed
        ``[iy, ix]`` (row = y, column = x). Must be finite and ``>= 1``
        everywhere. Not copied defensively beyond an ``asarray(float)``; do not
        mutate it after construction.
    dl_x_um, dl_y_um:
        Uniform transverse grid spacings along x and y (microns, ``> 0``).
        (Non-uniform transverse meshing is out of scope for Phase 1.)
    wavelength_um:
        Free-space wavelength (microns, ``> 0``). Use
        ``wavelength_um = c0 / freq_hz * 1e6`` to go from frequency.
    """

    #: Speed of light in vacuum (m/s) — the one physical constant, matching
    #: the benchmark reference solver (shared plugins._constants value).
    C0: float = _C0

    #: Hard cap on N = nx*ny for the dense O(N^2)-memory / O(N^3)-time solve.
    #: A 90x90 window is N=8100 -> a ~0.5 GB dense operator, already generous
    #: for a single-mode strip; larger grids should coarsen ``dl`` or shrink the
    #: window. Exceeding this raises in :meth:`solve` rather than thrashing.
    MAX_UNKNOWNS: int = 8100

    def __init__(
        self,
        eps: np.ndarray,
        dl_x_um: float,
        dl_y_um: float,
        wavelength_um: float,
    ) -> None:
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.ndim != 2:
            raise ValueError(
                f"eps must be a 2-D [iy, ix] array, got ndim={eps_arr.ndim}")
        if eps_arr.shape[0] < 3 or eps_arr.shape[1] < 3:
            raise ValueError(
                f"eps grid too small {eps_arr.shape}; need at least 3x3 so a "
                "second-order stencil has interior points")
        if not np.all(np.isfinite(eps_arr)):
            raise ValueError("eps contains non-finite values (NaN/Inf)")
        if np.any(eps_arr < 1.0):
            raise ValueError("eps must be >= 1 everywhere (passive dielectric)")
        if not (dl_x_um > 0 and dl_y_um > 0):
            raise ValueError("dl_x_um and dl_y_um must be > 0")
        if not (wavelength_um > 0):
            raise ValueError("wavelength_um must be > 0")

        self.eps: np.ndarray = eps_arr
        self.dl_x_um: float = float(dl_x_um)
        self.dl_y_um: float = float(dl_y_um)
        self.wavelength_um: float = float(wavelength_um)

    def at_wavelength(self, wavelength_um: float) -> "ModeSolver":
        """A sibling solver on the SAME cross-section (``eps``, ``dl``) at a new
        free-space wavelength. Use this to re-solve a mode across a frequency
        band — ``wavelength_um = C0 / freq_hz * 1e6`` — without re-rasterizing
        the geometry (the basis for the broadband ``num_freqs`` mode solves on
        both the source and monitor sides). ``eps`` is shared by reference (the
        operator never mutates it)."""
        return ModeSolver(self.eps, self.dl_x_um, self.dl_y_um, wavelength_um)

    # -- convenience cross-section builder ---------------------------------

    @classmethod
    def from_rectangular_core(
        cls,
        *,
        wavelength_um: float,
        dl_um: float,
        core_w_um: float,
        core_h_um: float,
        n_core: float,
        n_clad: float,
        window_w_um: Optional[float] = None,
        window_h_um: Optional[float] = None,
        clad_pad_um: float = 0.5,
    ) -> "ModeSolver":
        """Build a solver for a centered rectangular core in a uniform cladding.

        Rasterizes the canonical strip-waveguide cross-section (the validated
        SOI case) onto a *square* uniform grid of spacing ``dl_um``. This keeps
        the plugin decoupled from the :mod:`photonhub.components` spec models —
        feed it plain numbers (or read them off a ``Box`` + two ``Medium``
        permittivities yourself and call the raw constructor).

        Parameters
        ----------
        wavelength_um:
            Free-space wavelength (microns).
        dl_um:
            Uniform grid spacing for both x and y (microns).
        core_w_um, core_h_um:
            Core full width (x) and height (y) in microns.
        n_core, n_clad:
            Core and cladding refractive indices (``eps = n**2``).
        window_w_um, window_h_um:
            Total computational window extents (microns). If omitted, default to
            the core plus ``clad_pad_um`` of cladding on *each* side so the field
            decays before the Dirichlet wall. (Padding is additive, not a core
            multiple, to keep the dense operator a tractable size.)
        clad_pad_um:
            Per-side cladding padding (microns) used only when a window extent is
            omitted. ~0.5 um is several evanescent decay lengths for a typical
            SOI strip; enlarge it for weakly-guided (low-contrast) modes.
        """
        if window_w_um is None:
            window_w_um = core_w_um + 2.0 * clad_pad_um
        if window_h_um is None:
            window_h_um = core_h_um + 2.0 * clad_pad_um
        # Force an ODD cell count on each axis so a cell CENTER lands on the
        # cross-section origin (x = y = 0). With the core centered there, a core
        # of an integer number of cells then lands symmetrically on whole cells;
        # an even count would straddle the core edges with the boundary, smearing
        # the high-contrast interface and depressing n_eff. (The validated
        # benchmark used 61x61 for exactly this reason.)
        nx = _odd(max(3, int(round(window_w_um / dl_um))))
        ny = _odd(max(3, int(round(window_h_um / dl_um))))
        eps = cls._rasterize_rect(
            nx, ny, dl_um, dl_um, core_w_um, core_h_um,
            float(n_core) ** 2, float(n_clad) ** 2)
        return cls(eps, dl_um, dl_um, wavelength_um)

    @staticmethod
    def _rasterize_rect(
        nx: int, ny: int, dl_x_um: float, dl_y_um: float,
        core_w_um: float, core_h_um: float,
        eps_core: float, eps_clad: float,
    ) -> np.ndarray:
        """Centered rectangular core on a uniform grid. Returns ``eps[iy, ix]``.
        (Same rasterization as ``benchmarks/waveguide/mode_solver.build_eps``.)"""
        eps = np.full((ny, nx), eps_clad, dtype=float)
        xs = (np.arange(nx) - (nx - 1) / 2.0) * dl_x_um
        ys = (np.arange(ny) - (ny - 1) / 2.0) * dl_y_um
        X, Y = np.meshgrid(xs, ys)
        core = (np.abs(X) <= core_w_um / 2.0 + 1e-9) & \
               (np.abs(Y) <= core_h_um / 2.0 + 1e-9)
        eps[core] = eps_core
        return eps

    # -- the eigenproblem ---------------------------------------------------

    def _operator(self, k0: float, polarization: PolarizationName) -> np.ndarray:
        """Assemble the dense ``(N, N)`` transverse operator, ``N = nx*ny``.

        Generalizes ``benchmarks/waveguide/mode_solver._operator`` to anisotropic
        grid spacing (separate ``dl_x``, ``dl_y``) and a selectable flux-
        conservative direction. The ``"TE"`` (Ex-major) branch with
        ``dl_x == dl_y`` reproduces the benchmark operator exactly.
        """
        e = self.eps
        ny, nx = e.shape
        n = nx * ny
        a = np.zeros((n, n))
        hx = self.dl_x_um * 1e-6
        hy = self.dl_y_um * 1e-6
        ihx2 = 1.0 / (hx * hx)
        ihy2 = 1.0 / (hy * hy)

        # Flux-conservative axis: x for quasi-TE (Dx = eps*Ex continuous across
        # vertical walls), y for quasi-TM. "scalar" uses plain stencils on both.
        flux_x = polarization == "TE"
        flux_y = polarization == "TM"

        for j in range(ny):
            for i in range(nx):
                p = j * nx + i
                diag = 0.0

                # --- y-derivative ---
                if not flux_y:
                    diag += -2.0 * ihy2
                    if j > 0:
                        a[p, p - nx] += ihy2
                    if j < ny - 1:
                        a[p, p + nx] += ihy2
                else:
                    if j < ny - 1:
                        g_hi = 2.0 / (e[j, i] + e[j + 1, i])
                        a[p, p + nx] += ihy2 * g_hi * e[j + 1, i]
                        diag += -ihy2 * g_hi * e[j, i]
                    if j > 0:
                        g_lo = 2.0 / (e[j - 1, i] + e[j, i])
                        a[p, p - nx] += ihy2 * g_lo * e[j - 1, i]
                        diag += -ihy2 * g_lo * e[j, i]

                # --- x-derivative ---
                if not flux_x:
                    diag += -2.0 * ihx2
                    if i > 0:
                        a[p, p - 1] += ihx2
                    if i < nx - 1:
                        a[p, p + 1] += ihx2
                else:
                    # flux_{i+1/2} = 2/(e_i+e_{i+1}) * (e_{i+1}u_{i+1}-e_i u_i)/h
                    if i < nx - 1:
                        g_hi = 2.0 / (e[j, i] + e[j, i + 1])
                        a[p, p + 1] += ihx2 * g_hi * e[j, i + 1]
                        diag += -ihx2 * g_hi * e[j, i]
                    if i > 0:
                        g_lo = 2.0 / (e[j, i - 1] + e[j, i])
                        a[p, p - 1] += ihx2 * g_lo * e[j, i - 1]
                        diag += -ihx2 * g_lo * e[j, i]

                a[p, p] += diag + k0 * k0 * e[j, i]
        return a

    def solve(
        self,
        num_modes: int = 1,
        polarization: PolarizationName = "TE",
        n_guess: Optional[float] = None,
        bend_radius_um: Optional[float] = None,
    ) -> Tuple[Mode, ...]:
        """Compute the ``num_modes`` best-confined guided modes.

        Parameters
        ----------
        num_modes:
            Number of modes to return, ordered by descending ``n_eff`` (the
            first is the fundamental). Must be ``>= 1``.
        polarization:
            ``"TE"`` (default, validated quasi-TE Ex-major), ``"TM"``
            (quasi-TM Ey-major), or ``"scalar"`` (no vectorial correction).
        n_guess:
            Optional starting index hint. Modes are filtered to ``n_eff`` below
            ``sqrt(max(eps))`` (no mode can be more confined than the core)
            regardless; ``n_guess`` is currently advisory and does not change
            the returned set (the dense solve finds the true spectrum). Kept in
            the signature so a future inverse-iteration fast path can use it
            without an API break.
        bend_radius_um:
            Must be ``None`` (or non-finite/<=0 is rejected). **Bent waveguides
            are out of scope** — see the module docstring's error bound. A
            finite positive value raises ``NotImplementedError`` rather than
            silently returning a wrong (curvature-free) result.

        Returns
        -------
        tuple[Mode, ...]
            Up to ``num_modes`` :class:`Mode` objects, highest ``n_eff`` first.
            Fewer are returned if the window supports fewer guided modes.
        """
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        if polarization not in ("TE", "TM", "scalar"):
            raise ValueError(
                f"polarization must be 'TE', 'TM', or 'scalar', got "
                f"{polarization!r}")
        if bend_radius_um is not None:
            # Straight-waveguide-only: refuse rather than mislead. The conformal
            # curvature term is not in the operator (module docstring).
            raise NotImplementedError(
                "ModeSolver models STRAIGHT waveguides only; bent/curved modes "
                "(bend_radius_um set) are out of scope for Phase 1. The "
                "straight-waveguide n_eff error grows like (mode width / R) and "
                "ignores bending loss entirely — use a dedicated bent-mode "
                "solver. See photonhub.plugins.modes module docstring.")

        ny, nx = self.eps.shape
        n_unknowns = nx * ny
        if n_unknowns > self.MAX_UNKNOWNS:
            raise ValueError(
                f"cross-section is {nx}x{ny} = {n_unknowns} unknowns, exceeding "
                f"the dense-solver cap MAX_UNKNOWNS={self.MAX_UNKNOWNS}. Coarsen "
                "the grid spacing or shrink the computational window (a "
                "single-mode strip needs only a modest window).")

        k0 = 2.0 * np.pi / (self.wavelength_um * 1e-6)
        a = self._operator(k0, polarization)

        # Dense eigendecomposition: grids are modest by design. The operator is
        # real but non-symmetric (the flux-conservative branch), so use the
        # general eig; physical guided modes have real beta^2, so we keep the
        # (near-)real eigenpairs and discard spurious complex ones.
        vals, vecs = np.linalg.eig(a)

        n_max = float(np.sqrt(self.eps.max()))
        n_min = float(np.sqrt(self.eps.min()))
        # A guided mode satisfies n_clad < n_eff < n_core, i.e.
        # (k0 n_min)^2 < beta^2 <= (k0 n_max)^2. Filter to that band and to
        # essentially-real eigenvalues, then sort by descending beta^2.
        beta2 = vals.real
        imag_ok = np.abs(vals.imag) <= 1e-6 * (np.abs(vals.real) + 1.0)
        upper = (k0 * n_max) ** 2 * (1.0 + 1e-9)
        lower = (k0 * n_min) ** 2
        keep = imag_ok & (beta2 <= upper) & (beta2 > lower)
        order = np.argsort(beta2[np.where(keep)[0]])[::-1]
        idx = np.where(keep)[0][order]

        modes: list[Mode] = []
        for col in idx[:num_modes]:
            b2 = float(beta2[col])
            if b2 <= 0:
                continue
            n_eff = float(np.sqrt(b2) / k0)
            prof = vecs[:, col].real.reshape(ny, nx)
            # Sign-fix so the dominant lobe is positive, then L2-normalize.
            if prof.sum() < 0:
                prof = -prof
            norm = np.sqrt(np.sum(prof ** 2))
            if norm > 0:
                prof = prof / norm
            modes.append(Mode(
                n_eff=n_eff,
                field=prof,
                wavelength_um=self.wavelength_um,
                polarization=polarization,
                dl_x_um=self.dl_x_um,
                dl_y_um=self.dl_y_um,
            ))
        return tuple(modes)
