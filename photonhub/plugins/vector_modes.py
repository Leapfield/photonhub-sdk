"""Full-vectorial finite-difference eigenmode (FDE) solver — CPU only.

This is the **full-vector** companion to the frozen semi-vectorial
:mod:`photonhub.plugins.modes`. Where that solver carries a single dominant
transverse field component (Ex-major quasi-TE / Ey-major quasi-TM) and drops the
operator that couples the two transverse fields, this one solves the **coupled
transverse-magnetic-field eigenproblem** — keeping the vectorial coupling — so
it returns *real* hybrid/TM effective indices and **all six** field components.
It models a straight (z-invariant) waveguide cross-section like the semi-vec, and
additionally — via an opt-in curvature term + tangential PML (see *Bent
waveguides* below) — **bent** waveguides with radiation (bend) loss, returning a
**complex** ``n_eff``. Anisotropy (diagonal ε) is supported; dispersion and a
fully anisotropic (off-diagonal) ε are out of scope.

Physics / method
================
For a z-invariant cross-section with a **diagonal permittivity tensor**
``ε(x, y) = diag(εxx, εyy, εzz)``, guided modes have the separable form
``H(x, y, z, t) = h(x, y) * exp(-i (omega t - beta z))`` — the engine's
``e^{-i omega t}`` time convention, forward phasor ``e^{+i beta z}`` (see
``mode_overlap.py``; a note on the longitudinal components' phase under this
convention lives in the field-reconstruction docstring) — with propagation
constant ``beta`` and modal index ``n_eff = beta / k0`` (``k0 = 2*pi/lambda``). The
transverse magnetic field ``[Hx; Hy]`` satisfies the **Fallahkhair–Li–Murphy**
full-vector finite-difference eigenproblem (A.B. Fallahkhair, K.S. Li, T.E.
Murphy, *"Vector Finite Difference Modesolver for Anisotropic Dielectric
Waveguides"*, J. Lightwave Technol. **26**(11), 1423–1431, 2008), implemented
here directly from the paper's Appendix eqs (21)–(37) for the diagonal-ε case
(off-diagonal εxy=εyx=0):

    [Pxx Pxy] [Hx]      [Hx]
    [Pyx Pyy] [Hy] = b^2 [Hy] ,     b^2 = (n_eff * k0)^2 .

Each ``P`` block is a second-order finite-difference stencil built from ``eps``
and its interface averages: the scheme enforces the dielectric-interface
continuity conditions (tangential-H / normal-D matching) at the half-cell faces,
so the index jumps at high-contrast walls are handled correctly. The
**off-diagonal** ``Pxy``/``Pyx`` blocks are the vectorial coupling the semi-vec
omits; they vanish where ``eps`` is locally uniform, which is why the semi-vec is
a good approximation in the low-contrast limit and degrades for high-contrast
SOI / near-square cores.

**Why the H-field (not E):** ``Hx``, ``Hy`` are continuous across dielectric
interfaces, so the H-field operator is the better-conditioned, standard choice;
the E-field formulation handles the interface discontinuities worse.

The operator is assembled as a :class:`scipy.sparse` matrix (a compact stencil
per row → ``O(N)`` nonzeros) and the eigenpairs nearest a target index come from
:func:`scipy.sparse.linalg.eigs` in **shift-invert** mode
(``sigma = (n_guess*k0)^2``, ``which='LM'``) — far cheaper than the semi-vec's
dense ``O(N^3)`` path, so the ``2N`` problem stays tractable on a generous
window. Returned eigenpairs are filtered to genuinely guided
(``n_clad < n_eff < n_core``, near-real positive ``b^2``) and sorted by
descending ``n_eff``.

Field reconstruction (all six components)
=========================================
From the transverse magnetic field ``(Hx, Hy)`` and ``beta``:

    Hz = (i / beta) (dHx/dx + dHy/dy)          (from div H = 0),
    E  = (1 / (i omega eps)) (curl H)          (Ampere, source-free),

with ``omega = k0 c0``. All six components are returned as complex ``(ny, nx)``
arrays indexed ``[iy, ix]`` (row = y, col = x). The modal power is the real
z-Poynting flux ``(1/2) integral Re(E x H*) . z_hat dA``.

Boundary conditions
====================
The transverse window is padded with cladding (the guided mode decays before the
wall). The default walls are **PEC** electric walls (tangential E = 0): for the
transverse H this is a homogeneous-Neumann ghost on the tangential-H component
and a homogeneous-Dirichlet ghost on the normal-H component.
:meth:`from_rectangular_core` exposes ``x_symmetry="pmc"`` for a *magnetic*
x-wall, which makes the lowest mode exactly x-uniform (``kx = 0``) — the 1-D slab
limit a 2-D solver must reproduce for the analytic-slab validation.

Bent waveguides (complex n_eff = bend loss)
===========================================
``solve(bend_radius_um=R)`` solves the **bent**-waveguide mode of radius ``R``
(microns) by the **physical cylindrical** treatment. An azimuthal mode evolves as
``exp(-i ν φ)`` with ``ν = n_eff·k0·R`` (Tidy3D's convention, ``n_eff`` referenced
at ``R``, so ``R → ∞`` recovers the straight ``n_eff``); substituting into the
Helmholtz equation turns the constant longitudinal ``β²`` into the radius-dependent
``ν²/r²``, so with ``β₀ = n_eff·k0`` the transverse-H eigenproblem becomes the
**generalized** problem ``A h = β₀² B h`` where ``A`` is the *ungraded* (physical-ε)
straight FLM operator and ``B = diag((R/r)²)``, ``r = R + x`` the absolute radius at
the radial offset ``x`` (+x outward from the bend center). This ``(R/r)²``
"centrifugal" weight is the full non-perturbative curvature effect — it replaces the
older scalar Heiblum–Harris conformal-index map ``ε → ε·exp(2x/R)``, which only
reproduced the bend shift to first order in ``1/R`` and was ~12× too weak at tight
radii vs Tidy3D's ``ModeSolver`` (which is mathematically equivalent: a
transformation-optics radial Jacobian on ε *and* μ — see
:meth:`_centrifugal_weight`). The reported ``Re(n_eff)`` is the **highest in-band**
eigenvalue of a clean (PML-free) generalized solve — the physical, outer-shifted
bend index, matching Tidy3D's "sort by descending neff". A bend radiates, so the
**loss** ``mode.k_eff`` / ``mode.loss_db_per_cm`` comes from a second generalized
solve **with** a tangential PML (complex coordinate stretch ``s(x) = 1 + iσ/k0`` on
the in-plane edges, made by turning the FLM half-cell x-spacings complex): it takes
the radiating **core-confined** mode's ``Im(β)``, which is monotone in ``R`` and
guarded by a **passivity clamp** (a lossless bend cannot amplify). The straight path
(``bend_radius_um=None``, default) makes ``B`` the identity and keeps every quantity
real — bit-for-bit the original lossless operator.

.. note::
   The leaky bend ``n_eff(R)`` (real part) and loss are *window-dependent* for a
   tight, lossy bend (the outward-radiating mode samples the finite window/PML) —
   the same caveat Tidy3D carries. The match to Tidy3D's ``ModeSolver`` is at the
   *identical* cross-section + window (``benchmarks/tidy3d/bent_modes/spec.py``,
   5×3 µm). Widen the window ~linearly with ``R`` for a converged result.

Group index
===========
``n_g = n_eff - lambda * dn_eff/dlambda = c / v_g`` by **central difference**
over two extra solves at ``lambda (1 +/- delta)`` (``delta ~ 1e-3``); the same
mode is tracked across the three solves by maximum transverse-field overlap.
PhotonHub's media are non-dispersive (``eps`` is λ-independent), so this is pure
*waveguide* dispersion. Opt-in (``solve(..., group_index=True)``) — it triples
the solve cost.

Public API
==========
* ``VectorModeSolver(eps, dl_x_um, dl_y_um, wavelength_um)`` — same raw-eps
  validation as :class:`photonhub.plugins.modes.ModeSolver`.
* ``VectorModeSolver.from_rectangular_core(...)`` — the centered-rectangular-core
  rasterizer (mirrors the semi-vec), plus ``x_symmetry`` for the slab limit.
* ``VectorModeSolver.solve(num_modes=1, n_guess=None, group_index=False,
  bend_radius_um=None, num_pml=0, pml_strength=30.0)`` — the best-confined guided
  modes, highest ``Re(n_eff)`` first. **No polarization argument**: the
  full-vector solve finds every mode; polarization is a property of the result.
  ``bend_radius_um`` switches on the bent/leaky (complex-``n_eff``) solve.
* ``VectorMode`` (frozen dataclass): ``.n_eff`` (real part), ``.k_eff`` (imaginary
  modal index / loss, 0 for a straight mode), ``.bend_radius_um``, ``.n_group``,
  the six complex component arrays ``ex, ey, ez, hx, hy, hz``, ``.wavelength_um``,
  ``.dl_x_um``/``.dl_y_um``; derived ``.te_fraction`` / ``.polarization`` /
  ``.n_eff_complex`` / ``.loss_db_per_cm`` and the helpers
  ``.field_dataarray(component=...)`` (xarray, real-space µm coords) and
  ``.core_fraction(...)``.

CPU only. Requires :mod:`scipy` (sparse assembly + shift-invert eigensolve);
:mod:`numpy` + :mod:`xarray` as for the rest of the plugins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from ._constants import EPS0, C0 as _C0

try:  # scipy is a hard dependency of the full-vector solver (sparse eigensolve)
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
except ImportError as exc:  # pragma: no cover - exercised only without scipy
    raise ImportError(
        "photonhub.plugins.vector_modes requires scipy (sparse matrices + "
        "shift-invert eigensolve). Install scipy >= 1.10 to use VectorModeSolver."
    ) from exc

__all__ = ["VectorMode", "VectorModeSolver"]

#: The six field component names a :class:`VectorMode` carries.
ComponentName = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

#: Boundary symmetry on the x-axis walls. ``"none"`` is the default PEC electric
#: wall; ``"pmc"`` is a magnetic wall that makes the lowest mode x-uniform
#: (kx = 0) — used to recover the 1-D slab limit in the 2-D solver.
XSymmetry = Literal["none", "pmc"]

#: Cross-section rasterization for :meth:`VectorModeSolver.from_rectangular_core`.
#: ``"staircase"`` hard-samples ε (no smoothing); ``"volume"`` area-averages each
#: boundary cell (arithmetic / second-order on the tangential field); ``"tensor"``
#: applies the Kottke–Farjadpour–Johnson subpixel **tensor** (harmonic mean for
#: the interface-normal component, arithmetic for the tangential) so the *normal*
#: field also converges at second order — the default.
SubpixelMethod = Literal["staircase", "volume", "tensor"]


# ---------------------------------------------------------------------------
# Anisotropic (diagonal-ε) Fallahkhair–Li–Murphy stencil coefficients.
#
# Implemented directly from the paper (A.B. Fallahkhair, K.S. Li, T.E. Murphy,
# "Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides",
# J. Lightwave Technol. 26(11):1423, 2008), Appendix eqs (21)–(37), with the
# Fig. 1 quadrant numbering 1=NW, 2=SW, 3=SE, 4=NE and distances n,s,e,w =
# |P→N|,|P→S|,|P→E|,|P→W|. ``a_xx`` (21–27) and ``a_xy`` (28–34) are explicit;
# ``a_yy``/``a_yx`` follow from the paper's transformation (36): x↔y, n,N↔e,E,
# s,S↔w,W, ε³↔ε¹, v21→h23, v34→h14 — applied by remapping inputs/labels in
# :func:`_yblock`, so the validated x-block code is reused verbatim. Each quadrant
# qi carries a diagonal tensor ``(exx, eyy, ezz)``; off-diagonal εxy=εyx=0, so the
# pure-εyx terms drop and the surviving anisotropy enters through εyy/εzz (and
# εxx/εzz after the transform). In the isotropic limit exx=eyy=ezz this reduces
# to the standard isotropic full-vector operator (the paper's refs [16],[17]).
# ---------------------------------------------------------------------------
def _axx_coeffs(q1, q2, q3, q4, n, s, e, w, k):
    """Pxx block (Hx←Hx), eqs (21)–(27) reduced to diagonal ε (no corners)."""
    (_, yy1, zz1) = q1
    (_, yy2, zz2) = q2
    (_, yy3, zz3) = q3
    (_, yy4, zz4) = q4
    ew = e + w
    v21 = n * yy2 + s * yy1            # eq (35)
    v34 = n * yy3 + s * yy4
    aN = (yy3 / zz4) * 2.0 * e * yy4 / (v34 * n * ew) \
        + (yy2 / zz1) * 2.0 * w * yy1 / (v21 * n * ew)         # eq (21)
    aS = (yy4 / zz3) * 2.0 * e * yy3 / (v34 * s * ew) \
        + (yy1 / zz2) * 2.0 * w * yy2 / (v21 * s * ew)         # eq (22)
    aE = 2.0 / (e * ew)                                        # eq (23)
    aW = 2.0 / (w * ew)                                        # eq (24)
    z = np.zeros_like(yy1)                                     # eqs (25),(26) → 0
    aP = -(aN + aS + aE + aW) \
        + k * k * (n + s) / ew * (yy4 * yy3 * e / v34
                                  + yy1 * yy2 * w / v21)        # eq (27)
    return dict(P=aP, N=aN, S=aS, E=aE, W=aW,
                NE=z, NW=z.copy(), SE=z.copy(), SW=z.copy())


def _axy_coeffs(q1, q2, q3, q4, n, s, e, w, k):
    """Pxy block (Hx←Hy), eqs (28)–(34) for diagonal ε. Corners (32),(33) are
    nonzero wherever εyy≠εzz — the genuine anisotropic vectorial coupling."""
    (_, yy1, zz1) = q1
    (_, yy2, zz2) = q2
    (_, yy3, zz3) = q3
    (_, yy4, zz4) = q4
    ew = e + w
    v21 = n * yy2 + s * yy1
    v34 = n * yy3 + s * yy4
    aN = (s * yy2 * yy4 / (v21 * v34) - s * yy1 * yy3 / (v21 * v34)
          + yy3 * yy4 / (zz4 * v34) - yy2 * yy1 / (zz1 * v21)) / ew   # eq (28)
    aS = (n * yy2 * yy4 / (v21 * v34) - n * yy1 * yy3 / (v21 * v34)
          + yy1 * yy2 / (zz2 * v21) - yy4 * yy3 / (zz3 * v34)) / ew   # eq (29)
    # eq (30): a_xy^E = first-fraction − (2/(e·ew²))·[bracket]; εyx terms drop.
    fracE = (yy4 * (1.0 - yy3 / zz3) - yy3 * (1.0 - yy4 / zz4)) / (v34 * ew)
    brE = (w * w * yy1 * yy2 / v21 * (1.0 / zz1 - 1.0 / zz2)
           + e * w * yy3 * yy4 / v34 * (1.0 / zz4 - 1.0 / zz3))
    aE = fracE - 2.0 / (e * ew * ew) * brE
    # eq (31): a_xy^W (mirror of E).
    fracW = (yy2 * (1.0 - yy1 / zz1) - yy1 * (1.0 - yy2 / zz2)) / (v21 * ew)
    brW = (e * e * yy4 * yy3 / v34 * (1.0 / zz3 - 1.0 / zz4)
           + e * w * yy2 * yy1 / v21 * (1.0 / zz2 - 1.0 / zz1))
    aW = fracW - 2.0 / (w * ew * ew) * brW
    aNE = yy3 * (1.0 - yy4 / zz4) / (v34 * ew)                  # eq (32)
    aSE = yy4 * (yy3 / zz3 - 1.0) / (v34 * ew)
    aNW = yy2 * (yy1 / zz1 - 1.0) / (v21 * ew)                  # eq (33)
    aSW = yy1 * (1.0 - yy2 / zz2) / (v21 * ew)
    aP = -(aN + aS + aE + aW + aNE + aSE + aNW + aSW)           # eq (34)
    return dict(P=aP, N=aN, S=aS, E=aE, W=aW,
                NE=aNE, NW=aNW, SE=aSE, SW=aSW)


def _swapxy(q):
    (xx, yy, zz) = q
    return (yy, xx, zz)


#: Output-label remap for the transformation (36): N↔E, S↔W, NW↔SE, NE/SW fixed.
_TLABEL = dict(P="P", N="E", E="N", S="W", W="S",
               NE="NE", SW="SW", NW="SE", SE="NW")


def _yblock(xcoeff_fn, q1, q2, q3, q4, n, s, e, w, k):
    """a_yy = T[a_xx], a_yx = T[a_xy]: apply transformation (36) to the inputs
    (x↔y via :func:`_swapxy`, ε¹↔ε³, n↔e, s↔w) and relabel the outputs."""
    q1t, q2t, q3t, q4t = _swapxy(q3), _swapxy(q2), _swapxy(q1), _swapxy(q4)
    ct = xcoeff_fn(q1t, q2t, q3t, q4t, e, w, n, s, k)
    return {d: ct[_TLABEL[d]] for d in ct}


def _odd(n: int) -> int:
    """Smallest odd integer ``>= n`` (cell-count helper so a cell center sits on
    the cross-section origin — mirrors :mod:`photonhub.plugins.modes`)."""
    return n if n % 2 == 1 else n + 1


@dataclass(frozen=True)
class VectorMode:
    """One guided eigenmode of a straight waveguide, **full-vector** (all six
    field components).

    Attributes
    ----------
    n_eff:
        Modal effective index ``beta / k0`` (dimensionless, real). For a properly
        confined guided mode this lies between the cladding and core indices.
    n_group:
        Group index ``n_g = n_eff - lambda dn_eff/dlambda`` (waveguide
        dispersion), or ``None`` if not requested (``solve(group_index=False)``).
    ex, ey, ez, hx, hy, hz:
        The six field components on the cross-section, complex ``(ny, nx)`` arrays
        indexed ``[iy, ix]`` (row = y, column = x). The transverse pair
        ``(ex, ey)`` is jointly L2-normalized (``sum |ex|^2 + |ey|^2 == 1``) and
        phase-fixed so the dominant transverse-E component is real-positive at its
        peak; the other components are scaled consistently with the same
        eigenvector and ``beta``.
    wavelength_um:
        Free-space wavelength the mode was solved at (microns).
    dl_x_um, dl_y_um:
        Transverse grid spacings (microns), carried for the real-space helpers.
    center_offset_um:
        Optional ``(x, y)`` placement metadata (microns, mode frame: x = width,
        y = height): the ACTUAL center of the solver's sample window minus the
        center the caller REQUESTED. The cross-section rasterizers deliberately
        snap their eps window to the simulation grid (so dielectric walls land
        at the same sub-cell positions the FDTD saw), which displaces the array
        center from the requested center by up to ~a cell; consumers that
        reconstruct real-space coordinates as ``(i - (n-1)/2)*dl + center``
        (``vector_modal_fields`` and everything built on it) shift by this
        offset so the mode lands where its raster truly was. ``None`` (default,
        e.g. a hand-built or idealized-rectangle mode) means "centered as
        requested" — the legacy behavior.
    """

    n_eff: float
    n_group: Optional[float]
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray
    wavelength_um: float
    dl_x_um: float
    dl_y_um: float
    k_eff: float = 0.0
    bend_radius_um: Optional[float] = None
    center_offset_um: Optional[Tuple[float, float]] = None
    #: True when this mode was solved on the engine's OWN Yee staggering
    #: (``solve_yee_mode``): its per-component arrays carry the SAME intra-cell
    #: Yee offsets as the recorded FDTD field. A mode-overlap readout of such a
    #: mode is grid-consistent WITHOUT transverse co-location — each component
    #: multiplies the same-named sim component at its native Yee location (only
    #: the longitudinal, propagation-axis de-stagger is still needed). A
    #: node-collocated FLM mode (the default solver) leaves this False, and the
    #: readout co-locates the sim field to the node to match it. See
    #: mode_overlap._overlap_terms.
    yee_staggered: bool = False
    #: Optional per-axis node coordinates of the mode's raster, RELATIVE to the
    #: requested mode centre (mode-x = width axis, len == ex.shape[1]; mode-y =
    #: height, len == ex.shape[0]). Set by GRADED-window Yee solves (§15
    #: nonuniform grids), where the array pitch is NOT the scalar
    #: ``dl_x_um``/``dl_y_um`` — coordinate-aware consumers
    #: (``vector_modal_fields``) must prefer these over reconstructing a
    #: uniform ladder. ``None`` (uniform solves) keeps the legacy pitch path
    #: bit-identically.
    x_coords_um: Optional[np.ndarray] = None
    y_coords_um: Optional[np.ndarray] = None
    #: Optional solve provenance, attached by the cross-section dispatcher
    #: (``solve_mode_on_cross_section``): the exact solve kwargs MINUS the
    #: wavelength, so a consumer can re-solve the SAME mode identity (same
    #: window/pol/index/solver) at other frequencies —
    #: :class:`~photonhub.plugins.mode_devices.ModeMonitor` uses it to build its
    #: automatic per-frequency readout bank. ``None`` for hand-built solver
    #: modes (no auto-bank) and for dispatcher solves given an ``eps_of_medium``
    #: override (id-keyed, not replayable).
    solve_params: Optional[Mapping[str, Any]] = None

    @property
    def shape(self) -> Tuple[int, int]:
        """``(ny, nx)`` of the field arrays."""
        return tuple(self.ex.shape)  # type: ignore[return-value]

    @property
    def n_eff_complex(self) -> complex:
        """Complex modal index ``Re(n_eff) + i Im(n_eff)``. The imaginary part
        :attr:`k_eff` is the modal attenuation: ``k_eff > 0`` is LOSS — under
        the engine's ``e^{-i(omega t - beta z)}`` convention a forward mode
        goes as ``e^{+i beta z}`` with ``beta = k0*(n_eff + i*k_eff)``, so
        ``k_eff > 0`` gives ``|field| ~ e^{-k0*k_eff*z}``, decay along +z
        (this is exactly how :attr:`loss_db_per_cm` and the EME propagation
        factor consume it). Zero for a straight, lossless mode
        (``bend_radius_um is None``)."""
        return complex(self.n_eff, self.k_eff)

    @property
    def loss_db_per_cm(self) -> float:
        """Propagation/bend loss in **dB/cm** from the imaginary modal index.

        ``alpha [1/m] = 2 k0 k_eff`` (field ~ ``exp(-alpha/2 z)`` ⇒ power ~
        ``exp(-alpha z)``), and ``loss[dB/m] = 10 alpha / ln 10``; converted to
        dB/cm. For a bent mode this is the **radiation (bend) loss**; ``0`` for a
        straight lossless mode. A small *negative* value can appear for a
        nominally lossless mode from finite-PML residue — the bend solve filters
        spurious-gain (strongly negative) eigenpairs (see
        :meth:`VectorModeSolver.solve`)."""
        k0 = 2.0 * np.pi / (self.wavelength_um * 1e-6)
        alpha = 2.0 * k0 * self.k_eff           # power attenuation [1/m]
        loss_db_per_m = 10.0 * alpha / np.log(10.0)
        return float(loss_db_per_m / 100.0)

    @property
    def te_fraction(self) -> float:
        """Transverse-E TE fraction ``∫|Ex|² / (∫|Ex|² + ∫|Ey|²)`` — the
        polarization purity. ``~1`` for a clean TE (Ex-major) mode, ``~0`` for a
        clean TM (Ey-major) mode, ``~0.5`` for a strongly hybrid mode."""
        px = float(np.sum(np.abs(self.ex) ** 2))
        py = float(np.sum(np.abs(self.ey) ** 2))
        denom = px + py
        if denom <= 0.0:
            return 0.0
        return px / denom

    @property
    def polarization(self) -> str:
        """``"TE"`` if :attr:`te_fraction` ``>= 0.5`` (Ex-major), else ``"TM"``
        (Ey-major). A label for the dominant transverse-E component; strongly
        hybrid modes near 0.5 are classified by the majority component."""
        return "TE" if self.te_fraction >= 0.5 else "TM"

    def _component(self, component: str) -> Tuple[np.ndarray, str]:
        if not isinstance(component, str) or len(component) != 2:
            raise ValueError(
                f"component must be one of Ex/Ey/Ez/Hx/Hy/Hz, got {component!r}")
        key = component[0].upper() + component[1].lower()
        table = {
            "Ex": self.ex, "Ey": self.ey, "Ez": self.ez,
            "Hx": self.hx, "Hy": self.hy, "Hz": self.hz,
        }
        if key not in table:
            raise ValueError(
                f"component must be one of Ex/Ey/Ez/Hx/Hy/Hz, got {component!r}")
        return table[key], key

    def _node_coords_um(self) -> Tuple[np.ndarray, np.ndarray]:
        """Real-space ``(xs, ys)`` node coordinates in microns, RELATIVE to the
        requested mode centre — the SAME reconstruction
        :func:`~photonhub.plugins.mode_overlap.vector_modal_fields` uses, so this
        class reports the coordinates its own overlap consumers place it at.

        Prefers the carried graded node ladders (:attr:`x_coords_um` /
        :attr:`y_coords_um`, whose pitch is NOT the scalar ``dl_x_um``); else the
        uniform ladder PLUS :attr:`center_offset_um`, the window-placement shift a
        grid-snapped cross-section solve applies (up to ~a cell). Ignoring either
        mis-reports every coordinate by that shift."""
        ny, nx = self.ex.shape
        if self.x_coords_um is not None:
            return (np.asarray(self.x_coords_um, dtype=np.float64),
                    np.asarray(self.y_coords_um, dtype=np.float64))
        off_w, off_h = self.center_offset_um or (0.0, 0.0)
        xs = (np.arange(nx) - (nx - 1) / 2.0) * self.dl_x_um + off_w
        ys = (np.arange(ny) - (ny - 1) / 2.0) * self.dl_y_um + off_h
        return xs, ys

    def field_dataarray(self, component: str = "Ex") -> xr.DataArray:
        """One field ``component`` as an :class:`xarray.DataArray` with real-space
        ``x``/``y`` coordinates in microns (origin at the REQUESTED cross-section
        centre; a grid-snapped solve's :attr:`center_offset_um` and a graded solve's
        node ladders are honoured — see :meth:`_node_coords_um`). Dims
        ``("y", "x")``.

        ``component`` is one of ``"Ex"``, ``"Ey"``, ``"Ez"``, ``"Hx"``, ``"Hy"``,
        ``"Hz"`` (default ``"Ex"``). The data is complex; ``plot_mode`` and other
        consumers take ``.real`` / ``np.abs`` as needed."""
        arr, key = self._component(component)
        xs, ys = self._node_coords_um()
        attrs = {
            "n_eff": self.n_eff,
            "wavelength_um": self.wavelength_um,
            "polarization": self.polarization,
            "te_fraction": self.te_fraction,
            "component": key,
        }
        if self.n_group is not None:
            attrs["n_group"] = self.n_group
        if self.bend_radius_um is not None:
            attrs["bend_radius_um"] = self.bend_radius_um
            attrs["k_eff"] = self.k_eff
            attrs["loss_db_per_cm"] = self.loss_db_per_cm
        return xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": ys, "x": xs},
            name=key,
            attrs=attrs,
        )

    def core_fraction(self, core_w_um: float, core_h_um: float) -> float:
        """Fraction of the transverse-E energy ``|Ex|² + |Ey|²`` inside a centered
        ``core_w_um x core_h_um`` bounding box — a confinement metric (1.0 = fully
        confined). The box is centered on the REQUESTED cross-section centre,
        matching :meth:`VectorModeSolver.from_rectangular_core`; the mode's own
        node coordinates honour a grid-snapped solve's :attr:`center_offset_um` and
        a graded solve's ladders (see :meth:`_node_coords_um`), so the box lands on
        the real core instead of being displaced by the window snap."""
        xs_c, ys_c = self._node_coords_um()
        xs = np.abs(xs_c)
        ys = np.abs(ys_c)
        inside = (ys[:, None] <= core_h_um / 2.0 + 1e-12) & \
                 (xs[None, :] <= core_w_um / 2.0 + 1e-12)
        p = np.abs(self.ex) ** 2 + np.abs(self.ey) ** 2
        total = float(p.sum())
        if total <= 0.0:
            return 0.0
        return float(p[inside].sum() / total)

    def modal_power(self) -> float:
        """Time-averaged modal power (watts) carried across the cross-section by
        this full-vector mode — the **physical** z-Poynting flux

            P = (1/2) integral Re( E x H* ) . z_hat dA
              = (1/2) integral Re( Ex Hy* - Ey Hx* ) dA ,

        evaluated on the mode's own transverse grid (``dA = dl_x * dl_y`` in m²,
        midpoint quadrature). The six components are in SI units (E in V/m, H in
        A/m), so the integral is in watts. Used by the §18 mode-source builder to
        scale the injected profiles to **1 W** (``profile /= sqrt(P)``), and by
        any consumer that wants the absolute (not just relative) modal power.

        For the dominant-forward (``+z``) guided mode this is positive; a sign
        flip indicates a predominantly backward eigenpair.
        """
        dx = self.dl_x_um * 1e-6
        dy = self.dl_y_um * 1e-6
        sz = np.real(self.ex * np.conj(self.hy) - self.ey * np.conj(self.hx))
        return 0.5 * float(np.sum(sz)) * dx * dy


class VectorModeSolver:
    """Full-vectorial finite-difference eigenmode solver for a straight-waveguide
    cross-section.

    Construct from a raw permittivity cross-section and the transverse grid, then
    call :meth:`solve`. See the module docstring for the physics (the
    Fallahkhair–Li–Murphy transverse-H operator), the boundary conditions, and
    the **straight-waveguide-only** scope.

    Parameters
    ----------
    eps:
        Real relative-permittivity cross-section, a 2-D array indexed
        ``[iy, ix]`` (row = y, column = x). Must be finite and ``>= 1``
        everywhere. Stored as ``asarray(float)``; do not mutate after
        construction.
    dl_x_um, dl_y_um:
        Uniform transverse grid spacings along x and y (microns, ``> 0``).
    wavelength_um:
        Free-space wavelength (microns, ``> 0``).
    x_symmetry:
        ``"none"`` (default) for electric (PEC) walls on every edge, or ``"pmc"``
        for a magnetic wall on the x-axis edges. On a y-only (x-invariant) ε
        profile each wall gives an x-uniform (``kx = 0``) slab fundamental, but of
        a DIFFERENT polarization — the operator is in H, so the electric wall
        (``"none"``) yields the slab **TE0** (Eₓ-major) and the magnetic wall
        (``"pmc"``) the slab **TM0** (E_y-major). Choose the wall to match the
        polarization you want (the wrong one returns a too-low n_eff; see
        :meth:`from_rectangular_core`).
    """

    #: Speed of light in vacuum (m/s), matching :mod:`photonhub.plugins.modes`.
    C0: float = _C0  # speed of light (m/s), shared plugins._constants value

    #: Hard cap on ``N = nx*ny`` (the eigenproblem is ``2N``). The sparse
    #: shift-invert solve is far lighter than the semi-vec's dense path, so the
    #: cap is generous; a single-mode strip needs only a modest window.
    MAX_UNKNOWNS: int = 40000

    def __init__(
        self,
        eps: np.ndarray,
        dl_x_um: float,
        dl_y_um: float,
        wavelength_um: float,
        x_symmetry: XSymmetry = "none",
        *,
        eps_tensor: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
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
        if x_symmetry not in ("none", "pmc"):
            raise ValueError(
                f"x_symmetry must be 'none' or 'pmc', got {x_symmetry!r}")

        self.eps: np.ndarray = eps_arr
        self.dl_x_um: float = float(dl_x_um)
        self.dl_y_um: float = float(dl_y_um)
        self.wavelength_um: float = float(wavelength_um)
        self.x_symmetry: XSymmetry = x_symmetry

        # Diagonal permittivity tensor (εxx, εyy, εzz) the operator + field
        # reconstruction use. Scalar ε ⇒ all three equal ``eps`` (the isotropic
        # full-vector operator). A KFJ subpixel raster supplies an anisotropic
        # tensor whose normal component carries the harmonic mean (see
        # ``subpixel_method="tensor"`` in :meth:`from_rectangular_core`).
        if eps_tensor is None:
            self._exx = eps_arr
            self._eyy = eps_arr
            self._ezz = eps_arr
            self._is_tensor = False
        else:
            comps = tuple(np.asarray(c, dtype=float) for c in eps_tensor)
            if len(comps) != 3 or any(c.shape != eps_arr.shape for c in comps):
                raise ValueError(
                    "eps_tensor must be three arrays (εxx, εyy, εzz) matching "
                    f"eps shape {eps_arr.shape}")
            if any(not np.all(np.isfinite(c)) or np.any(c < 1.0) for c in comps):
                raise ValueError("eps_tensor components must be finite and >= 1")
            self._exx, self._eyy, self._ezz = comps
            self._is_tensor = True

    def at_wavelength(self, wavelength_um: float) -> "VectorModeSolver":
        """A sibling solver on the SAME cross-section (``eps`` / tensor, ``dl``,
        symmetry) at a new free-space wavelength — for re-solving a full-vector
        mode across a frequency band (``wavelength_um = C0 / freq_hz * 1e6``)
        without re-rasterizing. Mirrors :meth:`ModeSolver.at_wavelength`; the
        permittivity (and any KFJ tensor) is shared by reference."""
        eps_tensor = (
            (self._exx, self._eyy, self._ezz) if self._is_tensor else None
        )
        return VectorModeSolver(
            self.eps, self.dl_x_um, self.dl_y_um, wavelength_um,
            self.x_symmetry, eps_tensor=eps_tensor,
        )

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
        x_symmetry: XSymmetry = "none",
        x_min_symmetry: Optional[str] = None,
        subpixel: bool = True,
        subpixel_method: SubpixelMethod = "tensor",
    ) -> "VectorModeSolver":
        """Build a solver for a centered rectangular core in a uniform cladding.

        Rasterizes the canonical strip-waveguide cross-section onto a *square*
        uniform grid of spacing ``dl_um`` (the same rasterization as
        :meth:`photonhub.plugins.modes.ModeSolver.from_rectangular_core`, so the
        full-vector and semi-vec solvers are drop-in comparable).

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
            decays before the wall.
        clad_pad_um:
            Per-side cladding padding (microns) used only when a window extent is
            omitted.
        x_symmetry:
            x-wall boundary, ``"none"`` (default, electric / PEC walls) or
            ``"pmc"`` (magnetic x-wall). With ``core_w_um == window_w_um`` this
            builds a y-only slab cross-section whose fundamental is exactly
            x-uniform (kx = 0) — the 1-D slab limit. **The wall selects the
            polarization**, because the operator is in H: an electric wall
            (``"none"``) makes the major-H component Neumann (and Hₓ=0), so its
            x-uniform fundamental is the slab **TE0** (Eₓ-major); a magnetic wall
            (``"pmc"``) flips the parity, so its x-uniform fundamental is the slab
            **TM0** (E_y-major). Picking the *wrong* wall for the polarization you
            want does not error — it returns a mode of that polarization forced to
            zero at the walls (a spurious kx ≈ π/window_w), whose n_eff is too low.
            So: ``"none"`` for a TE slab, ``"pmc"`` for a TM slab (see
            ``validation/test_tier2a_vector_modesolver.py``).
        x_min_symmetry:
            ``None`` (default, full cross-section), ``"pec"``, or ``"pmc"`` to
            exploit a mirror symmetry about the width center (NUMERICS.md §20):
            the solver keeps only the **right half** (x ≥ centre), so the
            eigenproblem is half the size. The right half shares its nodes with
            the full centered grid. ``"pec"`` makes the x-min wall an odd /
            electric plane (reconstructs the full **even** mode exactly — the
            right choice for a TE-like width-even fundamental such as the taper's
            TE0); ``"pmc"`` makes it an even / magnetic plane (the complementary
            parity, selecting width-odd modes). The far wall sits in the cladding
            where the field has decayed, so its BC is immaterial.
        subpixel:
            ``False`` hard-samples ε (staircase). ``True`` (default) smooths the
            high-contrast walls; the kind is set by ``subpixel_method``.
        subpixel_method:
            ``"tensor"`` (default) applies the **Kottke–Farjadpour–Johnson**
            subpixel tensor — the interface-normal ε-component gets the harmonic
            mean and the tangential ones the arithmetic mean — so both the normal
            and tangential fields converge second-order (the FDTD engine's §16
            smoothing, here in the diagonal-tensor FDE operator). ``"volume"`` uses
            the scalar area-average (second-order on the tangential field only;
            the normal-field error stays first-order at high contrast). Ignored
            when ``subpixel=False``.
        """
        if window_w_um is None:
            window_w_um = core_w_um + 2.0 * clad_pad_um
        if window_h_um is None:
            window_h_um = core_h_um + 2.0 * clad_pad_um
        # Odd cell count -> a cell CENTER on the origin, so a centered integer-cell
        # core lands symmetrically on whole cells (matches the semi-vec).
        nx = _odd(max(3, int(round(window_w_um / dl_um))))
        ny = _odd(max(3, int(round(window_h_um / dl_um))))
        ec, ecl = float(n_core) ** 2, float(n_clad) ** 2
        # A core as wide/tall as the window is a *slab* in that axis (the
        # documented ``core_w==window_w`` x-uniform case). Extend it past the
        # outer cell edges so every cell is fully core (fill = 1) — otherwise the
        # boundary-coincident edge cells are half-filled, which the subpixel tensor
        # would read as a spurious wall interface and break the invariance.
        cw, ch = core_w_um, core_h_um
        if core_w_um >= window_w_um - 1e-12:
            cw = (nx + 2) * dl_um
        if core_h_um >= window_h_um - 1e-12:
            ch = (ny + 2) * dl_um
        if x_min_symmetry is not None and x_min_symmetry not in ("pec", "pmc"):
            raise ValueError(
                "x_min_symmetry must be None, 'pec' (odd / electric), or 'pmc' "
                f"(even / magnetic), got {x_min_symmetry!r}")
        # §20: keep the RIGHT half [centre .. +x edge]; the x-min wall is the
        # symmetry plane. nx is odd so the centre node is index (nx-1)//2 and the
        # slice shares its nodes with the full grid. The wall fold sign comes from
        # x_symmetry: "none" (-1) = PEC plane (reconstructs the full EVEN mode,
        # e.g. TE0); "pmc" (+1) = magnetic plane (the complementary parity). The
        # far (east) wall sits in the cladding (field ≈ 0), so its matching BC is
        # immaterial — both x-walls take the plane's sign. The half has an EVEN
        # node count, fine for the operator (>= 3 nodes).
        half = x_min_symmetry is not None
        c0 = (nx - 1) // 2 if half else 0
        if half:
            x_symmetry = "pmc" if x_min_symmetry == "pmc" else "none"

        if subpixel and subpixel_method == "tensor":
            eps, tensor = cls._kfj_tensor_rect(
                nx, ny, dl_um, dl_um, cw, ch, ec, ecl)
            if half:
                eps = eps[:, c0:]
                tensor = tuple(t[:, c0:] for t in tensor)
            return cls(eps, dl_um, dl_um, wavelength_um,
                       x_symmetry=x_symmetry, eps_tensor=tensor)
        eps = cls._rasterize_rect(
            nx, ny, dl_um, dl_um, cw, ch, ec, ecl, subpixel=subpixel)
        if half:
            eps = eps[:, c0:]
        return cls(eps, dl_um, dl_um, wavelength_um, x_symmetry=x_symmetry)

    @staticmethod
    def _rasterize_rect(
        nx: int, ny: int, dl_x_um: float, dl_y_um: float,
        core_w_um: float, core_h_um: float,
        eps_core: float, eps_clad: float,
        subpixel: bool = True,
    ) -> np.ndarray:
        """Centered rectangular core on a uniform grid. Returns ``eps[iy, ix]``.

        With ``subpixel=False`` this hard-samples (binary core/clad at the cell
        center, the semi-vec convention). With ``subpixel=True`` (default) each
        cell carries the **volume-fraction-averaged** permittivity of the core box
        it overlaps — `eps = f*eps_core + (1-f)*eps_clad`, `f` the EXACT separable
        axis-aligned fill fraction (box-cap KFJ §16.2). This removes the staircase
        at the high-contrast walls, so n_eff converges smoothly (the same
        accuracy-vs-resolution win the FDTD engine gets from §16 subpixel)."""
        xs = (np.arange(nx) - (nx - 1) / 2.0) * dl_x_um
        ys = (np.arange(ny) - (ny - 1) / 2.0) * dl_y_um
        if not subpixel:
            X, Y = np.meshgrid(xs, ys)
            core = (np.abs(X) <= core_w_um / 2.0 + 1e-9) & \
                   (np.abs(Y) <= core_h_um / 2.0 + 1e-9)
            eps = np.full((ny, nx), eps_clad, dtype=float)
            eps[core] = eps_core
            return eps

        # Per-axis 1-D overlap fraction of each cell with the core half-extent.
        def frac(centers, dl, half):
            lo = centers - dl / 2.0
            hi = centers + dl / 2.0
            ov = np.minimum(hi, half) - np.maximum(lo, -half)
            return np.clip(ov, 0.0, dl) / dl

        fx = frac(xs, dl_x_um, core_w_um / 2.0)   # (nx,)
        fy = frac(ys, dl_y_um, core_h_um / 2.0)   # (ny,)
        f = np.outer(fy, fx)                       # (ny, nx) separable fill
        return f * eps_core + (1.0 - f) * eps_clad

    @staticmethod
    def _kfj_tensor_rect(
        nx: int, ny: int, dl_x_um: float, dl_y_um: float,
        core_w_um: float, core_h_um: float,
        eps_core: float, eps_clad: float,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Centered rectangular core with the **Kottke–Farjadpour–Johnson**
        subpixel tensor. Returns ``(eps_scalar, (εxx, εyy, εzz))`` indexed
        ``[iy, ix]``.

        Each cell's fill fraction ``f`` (the exact separable axis-aligned overlap)
        and the interface normal ``n̂ = ∇f/|∇f|`` give a diagonal effective tensor:
        the normal component carries the harmonic mean ``ε⊥ = 1/(f/ε_in +
        (1-f)/ε_out)`` (the correct average for the discontinuous normal **D**),
        the tangential ones the arithmetic mean ``ε∥ = f·ε_in + (1-f)·ε_out``,
        combined as ``ε_aa = ε∥ + (ε⊥-ε∥)·n̂_a²`` (KFJ; the FDTD engine's §16.x
        smoothing). ``z`` is always tangential to a z-invariant cross-section, so
        ``εzz = ε∥``. Interior/exterior cells have ``f ∈ {0,1}`` (``ε⊥ = ε∥``), so
        the tensor is isotropic there. ``eps_scalar`` is the volume average ``ε∥``
        (the representative scalar for ``self.eps`` / ``n_guess``)."""
        xs = (np.arange(nx) - (nx - 1) / 2.0) * dl_x_um
        ys = (np.arange(ny) - (ny - 1) / 2.0) * dl_y_um

        def frac(centers, dl, half):
            lo = centers - dl / 2.0
            hi = centers + dl / 2.0
            return np.clip(np.minimum(hi, half) - np.maximum(lo, -half), 0.0,
                           dl) / dl

        fx = frac(xs, dl_x_um, core_w_um / 2.0)
        fy = frac(ys, dl_y_um, core_h_um / 2.0)
        f = np.outer(fy, fx)
        epar = f * eps_core + (1.0 - f) * eps_clad                  # arithmetic
        eperp = 1.0 / (f / eps_core + (1.0 - f) / eps_clad)         # harmonic
        # Interface normal from the fill-fraction gradient (cells with |∇f|≈0 are
        # uniform: n̂ irrelevant since ε⊥=ε∥ there).
        gy, gx = np.gradient(f, dl_y_um, dl_x_um)
        gmag = np.hypot(gx, gy)
        safe = np.where(gmag > 1e-12, gmag, 1.0)
        nxh = np.where(gmag > 1e-12, gx / safe, 0.0)
        nyh = np.where(gmag > 1e-12, gy / safe, 0.0)
        d = eperp - epar
        exx = epar + d * nxh ** 2
        eyy = epar + d * nyh ** 2
        ezz = epar
        return epar, (exx, eyy, ezz)

    # -- bent-waveguide curvature + tangential PML helpers -----------------

    def _x_centers_um(self, nx: int) -> np.ndarray:
        """In-plane x cell-center coordinates (µm), origin at the cross-section
        center, +x outward from the bend center — the radial offset the
        curvature map and PML profile are functions of."""
        return (np.arange(nx) - (nx - 1) / 2.0) * self.dl_x_um

    def _centrifugal_weight(
        self, nx: int, bend_radius_um: Optional[float]) -> np.ndarray:
        """Per-x-column **centrifugal** eigenvalue weight ``(R/r)² = (R/(R+x))²``
        for a bend of radius ``R`` (``r = R + x`` the absolute radius at the radial
        offset ``x``; ``x`` is the in-plane offset from the cross-section center,
        +x outward from the bend center).

        This is the *physical cylindrical* bend treatment (it replaces the older
        scalar Heiblum–Harris conformal-index map ``ε → ε·exp(2x/R)``, which only
        reproduced the bend shift to first order in ``1/R`` and was ~12× too weak
        at tight radii vs Tidy3D's ``ModeSolver``). An azimuthal mode of a bend
        evolves as ``exp(-i ν φ)``; Tidy3D's reported index is ``ν = n_eff·k0·R``
        (referenced at ``R``, so ``R→∞`` recovers the straight ``n_eff``).
        Substituting ``exp(-iνφ)`` into the Helmholtz equation, the **longitudinal**
        term is ``ν²/r²`` instead of a constant ``β²``: with ``β₀ = n_eff·k0 = ν/R``
        the transverse wave equation reads

            ∇ₜ² h + k0² ε(x,y) h = β₀² (R/r)² h ,

        i.e. a **generalized eigenproblem** ``A h = β₀² B h`` with the diagonal
        mass matrix ``B = diag((R/r)²)`` and ``A`` the *ungraded* (physical-ε)
        straight transverse-H operator. The ``(R/r)²`` "centrifugal" weight is the
        full non-perturbative curvature effect — the dominant term at tight bends,
        where Tidy3D's ``n_eff(R)`` rises far faster than ``1/R``.

        Returns the per-column ``(R/r)²`` (length ``nx``); all-ones for the
        straight path (``bend_radius_um is None``), so ``B`` is the identity and
        the eigenproblem collapses **bit-for-bit** to the straight one.
        ``bend_radius_um`` may be negative (bend center on the +x side)."""
        if bend_radius_um is None:
            return np.ones(nx)
        if bend_radius_um == 0.0:
            raise ValueError("bend_radius_um must be nonzero (use None for straight)")
        R = float(bend_radius_um)
        x = self._x_centers_um(nx)
        return (R / (R + x)) ** 2

    def _mass_matrix(
        self, nx: int, ny: int, bend_radius_um: Optional[float],
        *, complex_dtype: bool) -> "_sp.spmatrix":
        """Diagonal RHS mass matrix ``B`` for the bent generalized eigenproblem
        ``A h = β₀² B h``. ``B = diag((R/r)²)`` replicated over both the ``Hx`` and
        ``Hy`` field blocks (operator ordering ``(ix, iy)``, ``p = ix*ny + iy``).
        For the straight path (``bend_radius_um is None``) this is the identity, so
        the generalized solve reduces to the ordinary one and the straight result
        is unchanged. See :meth:`_centrifugal_weight` for the physics."""
        w = self._centrifugal_weight(nx, bend_radius_um)        # (nx,) = (R/r)²
        diag = np.repeat(w, ny)                                 # (N,) (ix,iy) order
        diag = np.concatenate([diag, diag])                     # Hx then Hy blocks
        if complex_dtype:
            diag = diag.astype(complex)
        return _sp.diags(diag)

    def _pml_stretch(
        self, nx: int, pml_cells: int, strength: float, k0: float,
    ) -> np.ndarray:
        """Complex coordinate-stretch factor ``s(x) = 1 + i σ(x)/k0`` per x-column
        for a tangential PML of ``pml_cells`` layers on **both** x-edges.

        Standard polynomial UPML conductivity profile ``σ(ξ) = σ_max ξ^m`` over the
        normalized PML depth ``ξ∈[0,1]`` (``m=3``); the complex stretch
        ``1 + iσ/k0`` enters the FLM half-cell spacings (so ``∂x → ∂x/s``),
        attenuating outgoing radiation and turning the eigenvalue complex. The
        ``+i`` sign is set so that a radiating (lossy) mode gets
        ``Im(n_eff) > 0`` — positive ``k_eff`` = loss, the sign
        :attr:`VectorMode.loss_db_per_cm` and the EME attenuation consume;
        the opposite sign would label loss as gain. The PML sits
        *outside* the guided core (the window is cladding-padded), so it does not
        perturb the bound mode while absorbing the leaky/radiated tail. Returns
        all-ones (real, no stretch) when ``pml_cells == 0`` — the straight,
        lossless path."""
        s = np.ones(nx, dtype=complex)
        if pml_cells <= 0:
            return s
        pml_cells = min(pml_cells, nx // 2)
        # Normalized depth into each PML (0 at inner edge → 1 at the wall).
        depth = (np.arange(pml_cells) + 1.0) / pml_cells
        sigma = strength * depth ** 3                       # σ/k0 (dimensionless)
        s[nx - pml_cells:] = 1.0 + 1j * sigma               # +x (outer) wall
        s[:pml_cells] = 1.0 + 1j * sigma[::-1]              # -x (inner) wall
        return s

    def _core_confinement(
        self, vecs: np.ndarray, nx: int, ny: int,
    ) -> np.ndarray:
        """Core-energy fraction of each eigenvector column (for bent-mode
        selection). The "core" is the high-ε region (``ε`` above the midpoint
        between the window's min and max), a geometry-free proxy that works for
        the raw-eps solver. Returns a length-``vecs.shape[1]`` array in ``[0, 1]``;
        a well-confined guided mode is ``~0.5–0.9``, a PML-localized or radiation
        eigenpair ``~0`` — so ranking by this cleanly rejects the spurious modes a
        leaky/PML spectrum is full of.

        The operator's H-field is flattened in (ix, iy) order (``p = ix*ny+iy``),
        with both ``Hx`` (first ``N``) and ``Hy`` (second ``N``) blocks."""
        N = nx * ny
        eps_t = self.eps.T                                  # (ix, iy)
        thresh = 0.5 * (float(eps_t.min()) + float(eps_t.max()))
        core = (eps_t.ravel() > thresh)
        if not core.any():                                  # uniform ε: no core
            return np.ones(vecs.shape[1])
        hx = vecs[:N, :]
        hy = vecs[N:, :]
        power = np.abs(hx) ** 2 + np.abs(hy) ** 2           # (N, ncols)
        total = power.sum(axis=0)
        total = np.where(total > 0.0, total, 1.0)
        return power[core, :].sum(axis=0) / total

    # -- the Fallahkhair-Li-Murphy transverse-H operator -------------------

    def _build_operator(
        self,
        k0: float,
        bend_radius_um: Optional[float] = None,
        pml_cells: int = 0,
        pml_strength: float = 5.0,
    ) -> "_sp.csr_matrix":
        """Assemble the sparse ``(2N, 2N)`` full-vector transverse-H operator
        ``[[Pxx, Pxy], [Pyx, Pyy]]`` for a **diagonal permittivity tensor**
        ``(εxx, εyy, εzz)``, with rows/cols ordered ``[Hx (N), Hy (N)]``.

        ``bend_radius_um`` is accepted for signature stability but **does not**
        scale ε here: the bend curvature is carried by the RHS centrifugal mass
        matrix ``B = diag((R/r)²)`` in :meth:`_solve_at` (the physical cylindrical
        treatment, :meth:`_centrifugal_weight`), so ``A`` is the *ungraded*
        physical-ε operator. A tangential PML (``pml_cells > 0``) is layered on
        **without touching the FLM stencil derivation** as complex-stretched
        half-cell x-spacings — they flow through the algebraic coefficient
        expressions, so the eigenvalue ``β²`` becomes complex (its imaginary part
        is the radiation/bend loss). With ``pml_cells == 0`` every quantity is real
        and the operator is **bit-for-bit** the straight, lossless operator.

        Direct implementation of the Fallahkhair–Li–Murphy anisotropic stencil
        (A.B. Fallahkhair, K.S. Li, T.E. Murphy, JLT 26(11):1423, 2008), Appendix
        eqs (21)–(37): ``Pxx``/``Pxy`` from :func:`_axx_coeffs`/:func:`_axy_coeffs`,
        ``Pyy``/``Pyx`` from the paper's x↔y transformation (36) via
        :func:`_yblock`. Scalar ε (``εxx=εyy=εzz``) recovers the isotropic
        full-vector operator (the paper's refs [16],[17]); a KFJ subpixel tensor
        feeds the interface-normal harmonic mean through εyy/εzz so the *normal*
        field also converges second-order.

        Index convention: built in **(ix, iy)** order, ``p = ix*ny + iy`` (x outer,
        y inner), on each ε component transposed to ``[ix, iy]``. The four ε
        quadrants around a node (paper Fig. 1: 1=NW, 2=SW, 3=SE, 4=NE) are read
        from a one-ring edge-replication pad (a cladding ghost ring — exact to the
        mode's accuracy since the window is cladding-padded). The diagonal
        ``Pxx``/``Pyy`` blocks are 5-point; the off-diagonal ``Pxy``/``Pyx`` carry
        the vectorial coupling, including the 4 anisotropic corner couplings (eqs
        32,33) that are nonzero wherever εyy≠εzz (an interface cell). The
        eigenvalue is ``b² = (n_eff k0)²``.

        Boundary conditions: symmetric/antisymmetric wall folding of the N/S/E/W
        coefficients — the default (PEC-like) electric walls use ``Hx`` tangential
        → symmetric on the y-walls / antisymmetric on the x-walls and vice-versa
        for ``Hy``; ``x_symmetry == "pmc"`` swaps the x-wall parity so the lowest
        mode is x-uniform (``kx = 0``) — the 1-D slab limit. The corner couplings
        vanish in the cladding ring at the walls, so they need no folding.
        """
        # Work in (ix, iy) ordering: transpose each ε component to [ix, iy].
        nx, ny = self.eps.T.shape
        N = nx * ny
        k = k0
        # The straight, lossless path keeps every quantity REAL (no PML stretch) so
        # the operator is **bit-for-bit** the original straight operator. Curvature
        # is now carried entirely by the RHS mass matrix ``B = diag((R/r)²)`` in
        # :meth:`_solve_at` (the physical cylindrical/centrifugal treatment, see
        # :meth:`_centrifugal_weight`), NOT by scaling ε here — so the bent operator
        # ``A`` is the *ungraded* (physical-ε) straight operator, optionally with a
        # complex tangential PML stretch (``pml_cells > 0``) that absorbs the bend
        # radiation and makes the eigenvalue complex (the loss).
        leaky = pml_cells > 0
        # Uniform half-cell spacings (meters) as length-N arrays, so the published
        # coefficient expressions evaluate verbatim per node.
        dx = self.dl_x_um * 1e-6
        dy = self.dl_y_um * 1e-6
        if leaky:
            sx = self._pml_stretch(nx, pml_cells, pml_strength, k0)  # (nx,) cplx
            # Half-cell distance to the EAST/WEST face carries the average stretch
            # of the two cells the face separates (e at ix uses
            # (sx[ix]+sx[ix+1])/2). The tangential PML stretches only x.
            sx_e = 0.5 * (sx + np.r_[sx[1:], sx[-1]])               # face to x+
            sx_w = 0.5 * (sx + np.r_[sx[0], sx[:-1]])               # face to x-
            n = np.full(N, dy, dtype=complex)            # north  (y+)
            s = np.full(N, dy, dtype=complex)            # south  (y-)
            e = (dx * np.repeat(sx_e, ny)).astype(complex)   # east  (x+)
            w = (dx * np.repeat(sx_w, ny)).astype(complex)   # west  (x-)
        else:
            n = np.full(N, dy)   # north  (y+)
            s = np.full(N, dy)   # south  (y-)
            e = np.full(N, dx)   # east   (x+)
            w = np.full(N, dx)   # west   (x-)

        # Four quadrant samples around each node, for each tensor component. The
        # node sits at the shared corner of a 2×2 ε block; with ε at nodes we read
        # that block via a one-ring edge-replication pad. Paper quadrants
        # 1=NW,2=SW,3=SE,4=NE map to the (x,y+),(x,y),(x+,y),(x+,y+) samples (the
        # numbering for which the isotropic reduction matches the standard
        # operator — validated against the analytic slab).
        def corners(a):
            P = np.empty((nx + 1, ny + 1), dtype=a.dtype)
            P[:-1, :-1] = a
            P[-1, :-1] = a[-1, :]
            P[:-1, -1] = a[:, -1]
            P[-1, -1] = a[-1, -1]
            return (P[:-1, 1:].ravel(), P[:-1, :-1].ravel(),
                    P[1:, :-1].ravel(), P[1:, 1:].ravel())

        # ε in (ix, iy) — the *physical* (ungraded) permittivity tensor. Curvature
        # is in the RHS mass matrix B (see _solve_at), not an ε scaling.
        exx = np.ascontiguousarray(self._exx.T)
        eyy = np.ascontiguousarray(self._eyy.T)
        ezz = np.ascontiguousarray(self._ezz.T)
        xx1, xx2, xx3, xx4 = corners(exx)
        yy1, yy2, yy3, yy4 = corners(eyy)
        zz1, zz2, zz3, zz4 = corners(ezz)
        q1 = (xx1, yy1, zz1)
        q2 = (xx2, yy2, zz2)
        q3 = (xx3, yy3, zz3)
        q4 = (xx4, yy4, zz4)

        Pxx = _axx_coeffs(q1, q2, q3, q4, n, s, e, w, k)
        Pxy = _axy_coeffs(q1, q2, q3, q4, n, s, e, w, k)
        Pyy = _yblock(_axx_coeffs, q1, q2, q3, q4, n, s, e, w, k)
        Pyx = _yblock(_axy_coeffs, q1, q2, q3, q4, n, s, e, w, k)

        # ----- boundary folding (N/S/E/W; corners vanish in the cladding ring) --
        ii = np.arange(N).reshape(nx, ny)
        sgn_x = +1.0 if self.x_symmetry == "pmc" else -1.0
        sgn_y = +1.0
        # Hx (Pxx,Pyx) symmetric on the same-parity wall, Hy (Pyy,Pxy) the mirror.
        for (wall, dirA, dirB, sgn) in (
                (ii[:, -1], "N", "S", sgn_y),    # NORTH wall: fold N onto S
                (ii[:, 0], "S", "N", sgn_y),     # SOUTH wall: fold S onto N
                (ii[-1, :], "E", "W", sgn_x),    # EAST  wall: fold E onto W
                (ii[0, :], "W", "E", sgn_x)):    # WEST  wall: fold W onto E
            Pxx[dirB][wall] += sgn * Pxx[dirA][wall]
            Pyx[dirB][wall] += sgn * Pyx[dirA][wall]
            Pyy[dirB][wall] -= sgn * Pyy[dirA][wall]
            Pxy[dirB][wall] -= sgn * Pxy[dirA][wall]

        # ----- 9-point sparse assembly -----
        # neighbor (row-subset, col-subset) index pairs in the (ix, iy) grid.
        nbr = {
            "E": (ii[:-1, :], ii[1:, :]),
            "W": (ii[1:, :], ii[:-1, :]),
            "N": (ii[:, :-1], ii[:, 1:]),
            "S": (ii[:, 1:], ii[:, :-1]),
            "NE": (ii[:-1, :-1], ii[1:, 1:]),
            "NW": (ii[1:, :-1], ii[:-1, 1:]),
            "SE": (ii[:-1, 1:], ii[1:, :-1]),
            "SW": (ii[1:, 1:], ii[:-1, :-1]),
        }
        iall = ii.ravel()

        def scatter(block, roff, coff):
            I = [iall + roff]
            J = [iall + coff]
            V = [block["P"][iall]]
            for key, (rows, cols) in nbr.items():
                r = rows.ravel()
                I.append(r + roff)
                J.append(cols.ravel() + coff)
                V.append(block[key][r])
            return I, J, V

        I, J, V = [], [], []
        for blk, roff, coff in ((Pxx, 0, 0), (Pxy, 0, N),
                                (Pyx, N, 0), (Pyy, N, N)):
            bi, bj, bv = scatter(blk, roff, coff)
            I += bi
            J += bj
            V += bv
        A = _sp.coo_matrix(
            (np.concatenate(V), (np.concatenate(I), np.concatenate(J))),
            shape=(2 * N, 2 * N)).tocsr()
        # Operator is built in (ix, iy) ordering; the solve unflattens with that
        # same convention (see _solve_at).
        return A

    # -- the solve ----------------------------------------------------------

    def solve(
        self,
        num_modes: int = 1,
        n_guess: Optional[float] = None,
        group_index: bool = False,
        bend_radius_um: Optional[float] = None,
        num_pml: int = 0,
        pml_strength: float = 30.0,
    ) -> Tuple[VectorMode, ...]:
        """Compute the ``num_modes`` best-confined guided modes (full-vector).

        Parameters
        ----------
        num_modes:
            Number of modes to return, ordered by descending ``Re(n_eff)`` (the
            first is the fundamental). Must be ``>= 1``.
        n_guess:
            Shift-invert target index. The solve returns the modes whose ``n_eff``
            is nearest ``n_guess``; defaults to ``sqrt(max(eps))`` (the
            best-confined, near the core index). A few extra eigenpairs are
            requested internally and filtered to the guided band.
        group_index:
            If ``True``, also compute each returned mode's group index ``n_g`` by
            central difference over two extra solves at ``lambda(1 +/- 1e-3)``
            (the same mode is tracked by maximum transverse-field overlap). Triples
            the cost. Default ``False`` (``mode.n_group is None``).
        bend_radius_um:
            If set (nonzero), solve the **bent**-waveguide mode of this radius
            (microns) via the physical cylindrical/centrifugal generalized
            eigenproblem ``A h = β₀² B h``, ``B = diag((R/r)²)`` (see
            :meth:`_centrifugal_weight`); the reported ``n_eff`` follows Tidy3D's
            convention (field ~ ``exp(i n k0 R φ)``, so ``R→∞`` recovers the
            straight ``n_eff``). The bend radiates, so a second PML solve makes the
            eigenvalue **complex** — ``mode.k_eff`` (imaginary index) and
            ``mode.loss_db_per_cm`` carry the bend loss. The loss solve needs a
            tangential PML to absorb the radiation; if ``num_pml`` is left at 0 a
            default PML is used. ``None`` (default) is the straight, lossless solve
            (``k_eff == 0``).
        num_pml:
            Number of complex-coordinate-stretched **PML** cells on each in-plane
            (x) window edge (:meth:`_pml_stretch`) so leaky/radiated fields are
            absorbed and the eigenvalue becomes complex. ``0`` (default) on the
            straight path keeps the operator real and lossless; a bend solve with
            ``num_pml == 0`` falls back to a sensible default thickness.
        pml_strength:
            PML conductivity scale ``σ_max/k0`` (dimensionless); the polynomial
            profile peaks at the wall. Default ``30.0`` — strong enough to absorb
            the bend radiation in a handful of cells.

        Returns
        -------
        tuple[VectorMode, ...]
            Up to ``num_modes`` :class:`VectorMode` objects, highest ``Re(n_eff)``
            first. Fewer are returned if the window supports fewer guided modes.

        Notes
        -----
        **Large-radius spurious-gain trap.** For a wide bend the field is nearly
        bound and its evanescent tail barely reaches the PML; finite PML absorption
        of that tail can flip the loss sign (nonphysical *gain*, ``Im(n_eff) < 0``)
        or spawn PML-localized spurious modes — the documented Tidy3D caveat
        (window/PML size must grow ~linearly with R). The bend solve guards this in
        two layers: (i) the physical mode is selected by **core confinement**
        (:meth:`_core_confinement`), not by ``n_eff`` proximity, so the
        strongly-spurious PML/radiation eigenpairs (which carry the large nonphysical
        ±gain) are never chosen; (ii) the surviving near-floor residue of a
        barely-leaky wide bend is **clamped to lossless** (a passive dielectric bend
        cannot amplify), so the wide-bend limit reports ~0 loss rather than a
        spurious negative. Quantitatively trustworthy bend loss therefore needs the
        bend to be tight enough that the loss is well above this floor (use a wide
        window so the PML sits well outside the mode)."""
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        if bend_radius_um is not None and bend_radius_um == 0.0:
            raise ValueError(
                "bend_radius_um must be nonzero (use None for a straight guide)")
        if num_pml < 0:
            raise ValueError("num_pml must be >= 0")
        ny, nx = self.eps.shape
        if nx * ny > self.MAX_UNKNOWNS:
            raise ValueError(
                f"cross-section is {nx}x{ny} = {nx * ny} unknowns (2N = "
                f"{2 * nx * ny} eigenproblem), exceeding MAX_UNKNOWNS="
                f"{self.MAX_UNKNOWNS}. Coarsen the grid or shrink the window.")

        # A bend radiates; it needs a tangential PML to absorb the leakage (else
        # the radiated field reflects off the PEC wall). Default to ~1/4 of the
        # x-window so the PML sits in the cladding outside the core.
        if bend_radius_um is not None and num_pml == 0:
            num_pml = max(6, nx // 4)

        # Anchor the shift-invert target on the **straight** fundamental (one cheap
        # real straight solve): the bent n_eff starts there at large R and shifts
        # UP as R tightens, and the bend solve sweeps a few shift anchors above it
        # to land on the outer-shifted physical branch (see _solve_at).
        if bend_radius_um is not None and n_guess is None:
            straight = self._solve_at(self.wavelength_um, 1, None)
            if straight:
                n_guess = straight[0].n_eff

        modes = self._solve_at(self.wavelength_um, num_modes, n_guess,
                               bend_radius_um, num_pml, pml_strength)
        if not group_index:
            return modes

        # Group index via central difference. Re-solve at lambda(1 +/- delta) and
        # match each base mode to its perturbed partner by max transverse-field
        # overlap (the spectrum can reorder slightly near-degenerate).
        delta = 1e-3
        lam = self.wavelength_um
        nm = max(num_modes + 2, len(modes) + 2)
        modes_lo = self._solve_at(lam * (1.0 - delta), nm, n_guess,
                                  bend_radius_um, num_pml, pml_strength)
        modes_hi = self._solve_at(lam * (1.0 + delta), nm, n_guess,
                                  bend_radius_um, num_pml, pml_strength)

        out: list[VectorMode] = []
        for m in modes:
            m_lo = self._match_mode(m, modes_lo)
            m_hi = self._match_mode(m, modes_hi)
            if m_lo is None or m_hi is None:
                out.append(m)  # could not track -> leave n_group None
                continue
            # dn_eff/dlambda via central difference (lambda step = lam*delta).
            dneff_dlam = (m_hi.n_eff - m_lo.n_eff) / (2.0 * lam * delta)
            n_g = m.n_eff - lam * dneff_dlam
            out.append(VectorMode(
                n_eff=m.n_eff, n_group=float(n_g),
                ex=m.ex, ey=m.ey, ez=m.ez, hx=m.hx, hy=m.hy, hz=m.hz,
                wavelength_um=m.wavelength_um,
                dl_x_um=m.dl_x_um, dl_y_um=m.dl_y_um,
                k_eff=m.k_eff, bend_radius_um=m.bend_radius_um,
            ))
        return tuple(out)

    # -- internals ----------------------------------------------------------

    def _solve_at(
        self,
        wavelength_um: float,
        num_modes: int,
        n_guess: Optional[float],
        bend_radius_um: Optional[float] = None,
        num_pml: int = 0,
        pml_strength: float = 5.0,
    ) -> Tuple[VectorMode, ...]:
        """Single-wavelength eigensolve + filter + reconstruct. Returns guided
        :class:`VectorMode`s sorted by descending ``Re(n_eff)`` (no group index).

        Straight, lossless path (``bend_radius_um is None and num_pml == 0``):
        ordinary real eigenproblem (``B`` = identity), near-real ``β²`` band
        filter, ``k_eff == 0`` — **bit-for-bit** the original solver. Bent path:
        the **generalized** eigenproblem ``A h = β₀² B h`` with the centrifugal
        mass matrix ``B = diag((R/r)²)`` (:meth:`_mass_matrix`) — the physical
        cylindrical treatment (see :meth:`_centrifugal_weight`). The reported
        ``Re(n_eff)`` is the physical bend index (Tidy3D's ``exp(-iνφ)``,
        ``ν = n_eff k0 R``); when a PML is present the **loss** (``k_eff``) comes
        from the radiating confined mode (a second PML eigensolve)."""
        eps = self.eps
        ny, nx = eps.shape
        N = nx * ny
        k0 = 2.0 * np.pi / (wavelength_um * 1e-6)
        n_max = float(np.sqrt(eps.max()))
        n_min = float(np.sqrt(eps.min()))
        is_bend = bend_radius_um is not None
        if n_guess is None:
            n_guess = n_max

        # Guided band on Re(β²). The bend pushes Re(n_eff) WELL above the straight
        # n_max at tight radii (Tidy3D: n_eff(R=5) ~ 1.99 for an n_max=2 core), so a
        # generous upper margin is needed; the straight path keeps the tight band.
        margin = 1.0 if not is_bend else 1.5
        upper = (k0 * n_max * margin) ** 2
        lower = (k0 * n_min) ** 2

        if not is_bend:
            # ---- straight (+ optional standalone PML) : ordinary eigenproblem --
            sigma = complex((n_guess * k0) ** 2) if num_pml > 0 \
                else float((n_guess * k0) ** 2)
            A = self._build_operator(k0, None, num_pml, pml_strength)
            extra = 16 if num_pml > 0 else 6
            k_req = max(1, min(2 * N - 2, max(num_modes + extra, 2 * num_modes)))
            try:
                vals, vecs = _spla.eigs(A, k=k_req, sigma=sigma, which="LM")
            except _spla.ArpackNoConvergence as exc:
                vals, vecs = exc.eigenvalues, exc.eigenvectors
                if vals.size == 0:
                    return tuple()
            beta_c = np.sqrt(vals.astype(complex))
            beta_c = np.where(beta_c.real < 0.0, -beta_c, beta_c)
            beta2_re = vals.real
            if num_pml > 0:
                conf = self._core_confinement(vecs, nx, ny)
                keep = (beta2_re > lower) & (beta2_re <= upper) & (conf > 0.25)
                idx = np.where(keep)[0]
                if idx.size == 0:
                    idx = np.where((beta2_re > lower) & (beta2_re <= upper))[0]
                order = idx[np.lexsort((beta2_re[idx], conf[idx]))[::-1]]
            else:
                imag_ok = np.abs(vals.imag) <= 1e-4 * (np.abs(vals.real) + 1.0)
                keep = imag_ok & (beta2_re > lower) & (beta2_re <= upper)
                idx = np.where(keep)[0]
                order = idx[np.argsort(beta2_re[idx])[::-1]]
            out: list[VectorMode] = []
            for col in order[:num_modes]:
                if beta2_re[col] <= 0.0:
                    continue
                beta = complex(beta_c[col]) if num_pml > 0 else float(beta_c[col].real)
                if num_pml > 0 and beta.imag < 0.0:
                    beta = complex(beta.real, 0.0)
                vec = vecs[:, col]
                hx = vec[:N].reshape(nx, ny).T.astype(np.complex128)
                hy = vec[N:].reshape(nx, ny).T.astype(np.complex128)
                out.append(self._reconstruct(hx, hy, beta, k0, wavelength_um, None))
            return tuple(out)

        # ---- bent path: generalized eigenproblem A h = β₀² B h ----------------
        # (1) n_eff (and the fields) come from a PML-FREE generalized solve, where
        #     the physical bend mode is the **highest in-band Re(n_eff)** — the
        #     centrifugal weight makes the outer-shifted mode the top guided one,
        #     matching Tidy3D's "sort by descending neff". (No PML here keeps the
        #     real-index spectrum clean: PML adds dense leaky/edge eigenpairs that
        #     muddy the highest-n selection.)
        # (2) the loss (k_eff) comes from a SEPARATE PML solve, taking the radiating
        #     confined mode's Im — robust + monotone in R, where (1) is purely real.
        B = self._mass_matrix(nx, ny, bend_radius_um, complex_dtype=False)
        extra = 20
        k_req = max(1, min(2 * N - 2, max(num_modes + extra, 2 * num_modes)))
        A0 = self._build_operator(k0, None, 0, 0.0)            # ungraded, real

        cols: list[tuple[float, np.ndarray]] = []
        # A couple of shift anchors so ARPACK lands on the (outer-shifted) physical
        # branch as well as near the straight index; merge the in-band guided modes.
        anchors = (n_guess, n_guess + 0.3 * (n_max - n_min),
                   min(n_max * (margin - 1e-3), n_guess + 0.6 * (n_max - n_min)))
        seen: set[int] = set()
        for a in anchors:
            try:
                vals, vecs = _spla.eigs(A0, k=k_req, M=B,
                                        sigma=float((a * k0) ** 2), which="LM")
            except _spla.ArpackNoConvergence as exc:
                vals, vecs = exc.eigenvalues, exc.eigenvectors
                if vals.size == 0:
                    continue
            except Exception:
                continue
            beta2_re = vals.real
            imag_ok = np.abs(vals.imag) <= 1e-3 * (np.abs(vals.real) + 1.0)
            keep = imag_ok & (beta2_re > lower) & (beta2_re <= upper)
            for col in np.where(keep)[0]:
                if beta2_re[col] <= 0.0:
                    continue
                neff = float(np.sqrt(beta2_re[col]) / k0)
                key = int(round(neff * 1e6))
                if key in seen:
                    continue
                seen.add(key)
                cols.append((neff, vecs[:, col].copy()))
        if not cols:
            return tuple()
        cols.sort(key=lambda t: -t[0])      # descending Re(n_eff)

        # Loss per (sign-anchored) confined PML mode — one extra eigensolve.
        loss_keff = self._bend_loss_keff(
            k0, bend_radius_um, num_pml, pml_strength, n_guess, lower, upper,
            nx, ny, N)

        out = []
        for neff, vec in cols[:num_modes]:
            beta = complex(neff * k0, loss_keff * k0)
            hx = vec[:N].reshape(nx, ny).T.astype(np.complex128)
            hy = vec[N:].reshape(nx, ny).T.astype(np.complex128)
            out.append(self._reconstruct(hx, hy, beta, k0, wavelength_um,
                                         bend_radius_um))
        return tuple(out)

    def _bend_loss_keff(
        self, k0: float, bend_radius_um: float, num_pml: int,
        pml_strength: float, n_guess: float, lower: float, upper: float,
        nx: int, ny: int, N: int) -> float:
        """Imaginary modal index ``k_eff`` (bend/radiation loss) for the bend.

        Solves the centrifugal generalized eigenproblem **with** the tangential
        PML (so the radiated tail is absorbed and the eigenvalue goes complex) and
        takes the **most core-confined** in-band mode's ``Im(β)/k0``. The confined
        radiating mode gives a stable, monotone-in-``R`` loss (the PML-free n_eff
        solve is purely real and carries no loss). A passivity clamp (a lossless
        dielectric bend cannot amplify) zeroes any near-floor numerical gain. Loss
        magnitude is only trustworthy when the bend is tight enough to radiate well
        above the PML floor and the window holds the mode (see :meth:`solve`)."""
        if num_pml <= 0:
            return 0.0
        Bc = self._mass_matrix(nx, ny, bend_radius_um, complex_dtype=True)
        A = self._build_operator(k0, None, num_pml, pml_strength)   # complex (PML)
        k_req = max(1, min(2 * N - 2, 24))
        try:
            vals, vecs = _spla.eigs(A, k=k_req, M=Bc,
                                    sigma=complex((n_guess * k0) ** 2), which="LM")
        except _spla.ArpackNoConvergence as exc:
            vals, vecs = exc.eigenvalues, exc.eigenvectors
            if vals.size == 0:
                return 0.0
        except Exception:
            return 0.0
        beta_c = np.sqrt(vals.astype(complex))
        beta_c = np.where(beta_c.real < 0.0, -beta_c, beta_c)
        beta2_re = vals.real
        conf = self._core_confinement(vecs, nx, ny)
        keep = (beta2_re > lower) & (beta2_re <= upper) & (conf > 0.25)
        idx = np.where(keep)[0]
        if idx.size == 0:
            idx = np.where((beta2_re > lower) & (beta2_re <= upper))[0]
        if idx.size == 0:
            return 0.0
        best = idx[int(np.argmax(conf[idx]))]
        keff = float(beta_c[best].imag / k0)
        return keff if keff > 0.0 else 0.0     # passivity clamp

    def _reconstruct(
        self,
        hx: np.ndarray,
        hy: np.ndarray,
        beta: complex,
        k0: float,
        wavelength_um: float,
        bend_radius_um: Optional[float] = None,
    ) -> VectorMode:
        """Reconstruct all six complex field components from ``(Hx, Hy, beta)``.

        ``beta`` may be **complex** for a bent/leaky mode (its imaginary part is
        the modal attenuation); the curl/divergence algebra below already uses
        ``1j*beta`` so it carries through, and the returned :class:`VectorMode`
        records ``k_eff = Im(beta)/k0`` and ``bend_radius_um``.

        Ampère + ``div H = 0``. NOTE ON THE PHASOR SIGN: the algebra below is
        written with ``d/dz -> -i beta`` — the ``exp(+i(omega t - beta z))``
        phasor, the CONJUGATE of the engine's ``e^{-i omega t}`` convention the
        rest of the stack (and this module's headline) uses. For the lossless
        straight modes returned here the transverse fields are real, so both
        conventions coincide there; the LONGITUDINAL components ``Ez``/``Hz``
        (purely imaginary) come out as the complex conjugates of the
        engine-convention fields (their imaginary parts carry the opposite
        sign). Transverse-only consumers (mode overlap, EME, launch profiles)
        are unaffected; anything consuming Ez/Hz phases against engine DFT
        data must conjugate. With ``omega = k0 c0``:

            Hz = (-i / beta) (dHx/dx + dHy/dy),                 (div H = 0)
            (curl H)_x = dHz/dy + i beta Hy,
            (curl H)_y = -i beta Hx - dHz/dx,
            (curl H)_z = dHy/dx - dHx/dy,
            E = (1 / (i omega eps0 eps_eff)) curl H.

        The permittivity used per E-component is **tangential-averaged**: ``Ex``
        is continuous across the y/z interfaces (tangential there) and jumps only
        across x, so it is divided by ``eps`` smoothed along y; ``Ey`` by ``eps``
        smoothed along x; ``Ez`` (everywhere tangential to the cross-section
        interfaces) by ``eps`` smoothed in both. This is the standard
        tangential-field permittivity averaging — it honours the actual
        field-continuity conditions so the dominant transverse E stays confined
        in the core (a *pointwise* ``curl/eps`` instead spuriously amplifies the
        tangential component on the low-index side of every interface). The sharp
        normal-E discontinuity is preserved (no averaging across the normal
        direction). The transverse-E pair is L2-normalized and phase-fixed on the
        dominant component's magnitude peak."""
        ny, nx = hx.shape
        dx = self.dl_x_um * 1e-6
        dy = self.dl_y_um * 1e-6
        eps = self.eps
        omega = k0 * self.C0
        # F/m = (mu0 c0^2)^-1, CODATA 2018 (matches engine kEps0 in types.h).
        eps0 = EPS0

        def ddx(f: np.ndarray) -> np.ndarray:
            g = np.zeros_like(f)
            g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
            g[:, 0] = (f[:, 1] - f[:, 0]) / dx
            g[:, -1] = (f[:, -1] - f[:, -2]) / dx
            return g

        def ddy(f: np.ndarray) -> np.ndarray:
            g = np.zeros_like(f)
            g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dy)
            g[0, :] = (f[1, :] - f[0, :]) / dy
            g[-1, :] = (f[-1, :] - f[-2, :]) / dy
            return g

        # Hz from div(H)=0: dHx/dx + dHy/dy - i*beta*Hz = 0 -> Hz = (-i/beta)(...).
        hz = (-1j / beta) * (ddx(hx) + ddy(hy))

        curl_x = ddy(hz) + 1j * beta * hy
        curl_y = -1j * beta * hx - ddx(hz)
        curl_z = ddx(hy) - ddy(hx)

        # Tangential permittivity averaging (one 3-point pass per tangential axis):
        # Ex tangential in y -> smooth eps in y; Ey tangential in x -> smooth in x;
        # Ez tangential in x and y -> smooth in both. Keeps the normal-direction
        # interface sharp so the physical normal-E jump survives.
        def smooth_y(e: np.ndarray) -> np.ndarray:
            g = e.astype(float).copy()
            g[1:-1, :] = (e[:-2, :] + e[1:-1, :] + e[2:, :]) / 3.0
            return g

        def smooth_x(e: np.ndarray) -> np.ndarray:
            g = e.astype(float).copy()
            g[:, 1:-1] = (e[:, :-2] + e[:, 1:-1] + e[:, 2:]) / 3.0
            return g

        eps_x = smooth_y(eps)            # for Ex
        eps_y = smooth_x(eps)            # for Ey
        eps_z = smooth_x(smooth_y(eps))  # for Ez
        pref = 1.0 / (1j * omega * eps0)
        ex = pref * curl_x / eps_x
        ey = pref * curl_y / eps_y
        ez = pref * curl_z / eps_z

        # Joint transverse-E L2 normalization + global phase fix so the dominant
        # transverse-E component is real-positive at its magnitude peak.
        norm = np.sqrt(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
        if norm > 0:
            scale = 1.0 / norm
        else:
            scale = 1.0
        # Pick the dominant transverse-E component for the phase reference.
        if np.sum(np.abs(ex) ** 2) >= np.sum(np.abs(ey) ** 2):
            ref = ex
        else:
            ref = ey
        peak = np.unravel_index(np.argmax(np.abs(ref)), ref.shape)
        phase = ref[peak]
        if abs(phase) > 0:
            phase_fix = np.conj(phase) / abs(phase)
        else:
            phase_fix = 1.0 + 0.0j
        g = scale * phase_fix

        neff_c = complex(beta) / k0
        return VectorMode(
            n_eff=float(neff_c.real),
            n_group=None,
            ex=ex * g, ey=ey * g, ez=ez * g,
            hx=hx * g, hy=hy * g, hz=hz * g,
            wavelength_um=float(wavelength_um),
            dl_x_um=self.dl_x_um, dl_y_um=self.dl_y_um,
            k_eff=float(neff_c.imag),
            bend_radius_um=(None if bend_radius_um is None
                            else float(bend_radius_um)),
        )

    @staticmethod
    def _match_mode(
        target: VectorMode,
        candidates: Tuple[VectorMode, ...],
    ) -> Optional[VectorMode]:
        """Best transverse-field overlap match of ``target`` within ``candidates``
        (for group-index mode tracking across the λ-perturbed solves). Returns
        ``None`` if no candidate has the same grid shape."""
        best = None
        best_score = -1.0
        tx = target.ex.ravel()
        ty = target.ey.ravel()
        tn = np.sqrt(np.vdot(tx, tx).real + np.vdot(ty, ty).real)
        for c in candidates:
            if c.ex.shape != target.ex.shape:
                continue
            cx = c.ex.ravel()
            cy = c.ey.ravel()
            cn = np.sqrt(np.vdot(cx, cx).real + np.vdot(cy, cy).real)
            if tn <= 0 or cn <= 0:
                continue
            ov = abs(np.vdot(tx, cx) + np.vdot(ty, cy)) / (tn * cn)
            if ov > best_score:
                best_score = ov
                best = c
        return best
