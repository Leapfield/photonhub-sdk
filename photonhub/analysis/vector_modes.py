"""Full-vectorial finite-difference eigenmode (FDE) solver, CPU only.

This is the **full-vector** companion to the frozen semi-vectorial
:mod:`photonhub.analysis.modes`. Where that solver carries a single dominant
transverse field component (Ex-major quasi-TE / Ey-major quasi-TM) and drops the
operator that couples the two transverse fields, this one solves the **coupled
transverse-magnetic-field eigenproblem**, keeping the vectorial coupling, so
it returns *real* hybrid/TM effective indices and **all six** field components.
It models a straight (z-invariant) waveguide cross-section like the semi-vec, and
additionally, via an opt-in curvature term + tangential PML (see *Bent
waveguides* below), **bent** waveguides with radiation (bend) loss, returning a
real ``n_eff`` plus positive attenuation index ``k_eff`` (the derived
``n_eff_complex = n_eff + i*k_eff`` is positive-loss metadata). Anisotropy
(diagonal ε) is supported; dispersion and a fully anisotropic (off-diagonal) ε
are out of scope.

Physics / method
================
For a z-invariant cross-section with a **diagonal permittivity tensor**
``ε(x, y) = diag(εxx, εyy, εzz)``, the eigensolver uses the engine convention
``exp(-i*omega*t + i*beta_store*z)``.  Its positive-loss metadata is
``beta_store = k0*(n_eff + i*k_eff)``; ``k_eff > 0`` therefore decays along +z.
Returned :class:`VectorMode` arrays instead use EME's conjugate convention
``exp(+i*omega*t - i*beta_eme*z)``, with
``beta_eme = beta_store.conjugate()``.  Lossless transverse fields coincide
between the two conventions; complex PML/bend fields, their beta, and their
contour are conjugated together before return.  The transverse magnetic field
``[Hx; Hy]`` satisfies the **Fallahkhair–Li–Murphy**
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
(``sigma = (n_guess*k0)^2``, ``which='LM'``), far cheaper than the semi-vec's
dense ``O(N^3)`` path, so the ``2N`` problem stays tractable on a generous
window. :meth:`solve` filters layer-free straight eigenpairs to the guided band.
With absorbing layers it also checks interior field energy and, for straight
guides, the real-index response to layer strength. Modes are ordered by
descending real ``n_eff``. :meth:`solve_eme_basis` instead targets guided, box/PML
radiation, and negative-``beta^2`` evanescent parts of one common operator.
The continuum path is experimental: its analytic operator regressions pass, but
device-level radiation/PML boundary-region convergence has not yet been demonstrated.

Field reconstruction (all six components)
=========================================
From the returned transverse magnetic field ``(Hx, Hy)`` and ``beta_eme``:

    Hz = (-i / beta_eme) (dHx/dx + dHy/dy)     (from div H = 0),
    E  = (1 / (i omega eps)) (curl H)          (Ampere, source-free),

with ``omega = k0 c0``. All six components are returned as complex ``(ny, nx)``
arrays indexed ``[iy, ix]`` (row = y, col = x). The modal power is the real
z-Poynting flux ``(1/2) integral Re(E x H*) . z_hat dA``.  The public same-shape
arrays use a documented component-aware co-location approximation; native FLM
places H on vertices and E/ε at cell centres, so field-level tensor accuracy is
not claimed to be identical to a fully staggered FLM reconstruction.

Boundary conditions
====================
The transverse window is padded with cladding (the guided mode decays before the
wall). The default walls are **PEC** electric walls (tangential E = 0): for the
transverse H this is a homogeneous-Neumann ghost on the tangential-H component
and a homogeneous-Dirichlet ghost on the normal-H component.
:meth:`from_rectangular_core` exposes ``x_symmetry="pmc"`` for a *magnetic*
x-wall, which makes the lowest mode exactly x-uniform (``kx = 0``), the 1-D slab
limit a 2-D solver must reproduce for the analytic-slab validation.

``x_min_symmetry="pec"|"pmc"`` makes the x-min wall a **mirror plane** through
the centre of ε column 0. The H nodes sit on cell corners,
half a cell from that plane, so the ghost column is the image of node 0
(``H[-1] = ±H[0]``), not of node 1 as at the other walls. The half window then
carries the full operator restricted to one parity. Its modes take the plane as
the x origin: ``center_offset_um`` puts column ``j`` at ``x = j*dl_x``, the full
solve's coordinate for the same column. ``core_fraction`` lays the half out as
the full solve's arrays hold it, so it describes the full mode; the other
integrals, including overlaps, cover the stored half.

Bent waveguides (real phase index + loss metadata)
==================================================
``solve(bend_radius_um=R)`` solves the **bent**-waveguide mode of radius ``R``
(microns) by the **physical cylindrical** treatment. In the engine/eigensolver
convention an azimuthal mode evolves as ``exp(+i ν_store φ)``; returned EME
arrays use ``exp(-i ν_eme φ)`` with ``ν_eme = ν_store.conjugate()``. Here
``Re(ν_store) = n_eff·k0·R`` (``n_eff`` referenced at ``R``, so ``R → ∞``
recovers the straight ``n_eff``). Substituting into the
Helmholtz equation turns the constant longitudinal ``β²`` into the radius-dependent
``ν²/r²``, so with ``β₀ = n_eff·k0`` the transverse-H eigenproblem becomes the
**generalized** problem ``A h = β₀² B h`` where ``A`` is the *ungraded* (physical-ε)
straight FLM operator and ``B = diag((R/r)²)``, ``r = R + x`` the absolute radius at
the radial offset ``x`` (+x outward from the bend center). This ``(R/r)²``
"centrifugal" weight is the full non-perturbative curvature effect, it replaces the
older scalar Heiblum–Harris conformal-index map ``ε → ε·exp(2x/R)``, which only
reproduced the bend shift to first order in ``1/R`` and was ~12× too weak at tight
radii vs an independent full-vector bend solve (which is mathematically equivalent: a
transformation-optics radial Jacobian on ε *and* μ, see
:meth:`_centrifugal_weight`). A bend radiates, so its solve uses a tangential
PML (complex coordinate stretch ``s(x) = 1 + iσ/k0`` on the in-plane edges).
Eigenpairs localized in those layers are rejected by the fraction of their
magnetic-field energy outside the layers. A bend refuses a radius whose
magnitude is less than half the full window width plus one x cell. Its
candidate pairs are assigned one to one to the straight modes in the same
window and layers, by interior field overlap times the square root of the
interior energy share. Unassigned pairs are discarded. The remaining leaky
modes are ordered by descending real ``n_eff``; each index, field and
attenuation comes from the same eigenpair. A passivity clamp removes numerical
gain at the PML floor. ``loss_unresolved`` flags a caustic too close to the
outer layer or a loss change above 10% at half layer strength. The straight path
(``bend_radius_um=None``, default) makes ``B`` the identity and keeps every quantity
real, bit-for-bit the original lossless operator.

.. note::
   The leaky bend ``n_eff(R)`` (real part) and loss are *window-dependent* for a
   tight, lossy bend (the outward-radiating mode samples the finite window/PML), a caveat any bend solver carries. The match to an independent full-vector bend solve is at the
   *identical* cross-section + window (a fixed bend-mode study,
   5×3 µm). Widen the window ~linearly with ``R`` for a converged result.

Yee-lattice method (opt-in)
===========================
``method="yee"`` replaces the transverse-H operator by the transverse-E
finite-difference operator on a staggered Yee lattice (Z. Zhu and T. G. Brown,
Opt. Express 10, 853 (2002)), the discretization the FDTD engine steps. Each
ε cell centre is a node carrying ``Ez``; ``Ex`` sits on the centre of the
cell's +x face and ``Ey`` on the centre of its +y face, and each component sees
its own permittivity sample. On a high-contrast rectangular core it converges
faster under mesh refinement than the default method, which loses accuracy at
the core's corners. The returned fields are co-located on the cell centres,
like the default method's. It covers straight solves only; see
:class:`VectorModeSolver` for its walls, mirror plane and limits.

Group index
===========
``n_g = n_eff - lambda * dn_eff/dlambda = c / v_g`` by **central difference**
over two extra solves at ``lambda (1 +/- delta)`` (``delta ~ 1e-3``); the same
mode is tracked across the three solves by maximum transverse-field overlap.
PhotonHub's media are non-dispersive (``eps`` is λ-independent), so this is pure
*waveguide* dispersion. Opt-in (``solve(..., group_index=True)``), it triples
the solve cost.

Public API
==========
* ``VectorModeSolver(eps, dl_x_um, dl_y_um, wavelength_um)``, same raw-eps
  validation as :class:`photonhub.analysis.modes.ModeSolver`.
* ``VectorModeSolver.from_rectangular_core(...)``, the centered-rectangular-core
  mesh sampler (mirrors the semi-vec), plus ``x_symmetry`` for the slab limit.
* ``VectorModeSolver.solve(num_modes=1, n_guess=None, group_index=False,
  bend_radius_um=None, num_pml=0, pml_strength=30.0)``, guided
  modes ordered by descending ``Re(n_eff)``. **No polarization argument**: the
  full-vector solve finds every mode; polarization is a property of the result.
  ``bend_radius_um`` switches on the bent/leaky solve.
* ``method="yee"`` on the constructor and on ``from_rectangular_core`` selects
  the Yee-lattice operator for straight solves.
* ``VectorModeSolver.solve_eme_basis(num_guided=1, num_radiation=0,
  num_evanescent=0, ...)``, a common-operator EME basis. Continuum requests
  use closed-box spectral seeds followed by four-sided complex-coordinate PML
  continuation; ``pml_cells_xy=(0, 0)`` selects an analytic hard-wall box.
  Radiation/evanescent requests are experimental and are not yet validated for
  quantitative device radiation loss.
* ``VectorMode`` (frozen dataclass): ``.n_eff`` (real phase index), ``.k_eff``
  (nonnegative attenuation metadata, 0 for a straight lossless mode),
  ``.window_limited`` (straight partner extends into the absorbing layers),
  ``.loss_unresolved`` (bend loss needs a convergence check),
  ``.bend_radius_um``, ``.n_group``,
  the six complex component arrays ``ex, ey, ez, hx, hy, hz``, ``.wavelength_um``,
  ``.dl_x_um``/``.dl_y_um``; derived ``.te_fraction`` / ``.polarization`` /
  ``.n_eff_complex`` (positive-loss engine/eigensolver metadata) /
  ``.loss_db_per_cm`` and the helpers
  ``.field_dataarray(component=...)`` (xarray, real-space µm coords) and
  ``.core_fraction(...)``. EME bases additionally set ``.mode_type``,
  ``.overlap_weights``, ``.physical_mask``, ``.pml_cells_xy``, and
  ``.eigen_residual``. Half-window solves set ``.x_min_symmetry`` and
  ``.center_offset_um``.

CPU only. Requires :mod:`scipy` (sparse assembly + shift-invert eigensolve);
:mod:`numpy` + :mod:`xarray` as for the rest of the plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import xarray as xr

from ._constants import EPS0, C0 as _C0
from .._compat import caller_stacklevel, legacy_keywords
import warnings

try:  # scipy is a hard dependency of the full-vector solver (sparse eigensolve)
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
except ImportError as exc:  # pragma: no cover - exercised only without scipy
    raise ImportError(
        "photonhub.analysis.vector_modes requires scipy (sparse matrices + "
        "shift-invert eigensolve). Install scipy >= 1.10 to use VectorModeSolver."
    ) from exc

__all__ = ["VectorMode", "VectorModeSolver"]

# The straight threshold includes leaky rib/TM and weak low-contrast modes;
# strength stability separates their window-filling box partners. Bend modes
# can move with absorber strength, so they use projection onto straight modes.
_STRAIGHT_MIN_INTERIOR_SHARE = 0.80
_STRAIGHT_SKIP_HALF_SHARE = 0.995
_STRAIGHT_WINDOW_LIMIT_SHARE = 0.95
_STRAIGHT_MAX_REAL_INDEX_MOVE = 1e-3
_HALF_PARTNER_MIN_OVERLAP = 0.90
_DUPLICATE_MIN_OVERLAP = 0.90
# A shift-invert candidate joins an eigenvalue cluster only when it adds a
# direction at least this large (relative norm of its remainder after removing
# the cluster's span). Two copies of one eigenvector found from different
# shifts differ by about residual / relative gap: in a 0.37 x 0.22 um Si
# taper box the copy from a distant shift (residual 5e-8, nearest eigenvalue
# 8e-4 away) sat 1.8e-6 off the accurate one, which a 1e-6 threshold kept as
# a second "direction", making the reaction Gram singular. Genuinely new
# directions of a degenerate eigenspace arrive with remainders of 1e-3 and
# more (2.8e-3 in a square-PML box), so 1e-4 separates the two cases.
_EIGENSPACE_NEW_DIRECTION_MIN = 1e-4
_BEND_MIN_INTERIOR_SHARE = 0.50
_BEND_MIN_STRAIGHT_PROJECTION = 0.40
_BEND_MAX_HALF_LOSS_CHANGE = 0.10
_BEND_MIN_CAUSTIC_MARGIN_AIRY = 0.75

#: The six field component names a :class:`VectorMode` carries.
ComponentName = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

#: Boundary symmetry on the x-axis walls. ``"none"`` is the default PEC electric
#: wall; ``"pmc"`` is a magnetic wall that makes the lowest mode x-uniform
#: (kx = 0) — used to recover the 1-D slab limit in the 2-D solver.
XSymmetry = Literal["none", "pmc"]

#: Parity of the x-min mirror plane of a half-window solve.
#: ``"pec"`` is odd / electric (Hx odd, Hy even); ``"pmc"`` is even / magnetic.
MirrorSymmetry = Literal["pec", "pmc"]

#: Cross-section rasterization for :meth:`VectorModeSolver.from_rectangular_core`.
#: ``"staircase"`` hard-samples ε (no smoothing); ``"volume"`` area-averages each
#: boundary cell (arithmetic / second-order on the tangential field); ``"tensor"``
#: applies a diagonalized Kottke–Farjadpour–Johnson subpixel tensor (harmonic
#: interface-normal and arithmetic tangential means). It is exact on the
#: axis-aligned face cells; the diagonal-only solver omits the off-diagonal
#: xy term of partially filled corner cells — the default.
SubpixelMethod = Literal["staircase", "volume", "tensor"]

#: Cross-section discretization of :class:`VectorModeSolver`. ``"flm"`` (the
#: default) is the Fallahkhair-Li-Murphy transverse-H operator; ``"yee"`` is
#: the transverse-E operator on the staggered Yee lattice, straight solves only.
ModeMethod = Literal["flm", "yee"]


class _YeeLattice(NamedTuple):
    """Permittivity on the solve lattice of ``method="yee"``, ``[iv, ih]``.

    The raster's own nodes plus, on a walled axis, one leading wall line that
    the returned arrays drop (``lead_x``/``lead_y``). ``h_bc``/``v_bc`` are the
    low-edge conditions of the lattice operator (``"pec"`` wall, a mirror
    parity, or ``"periodic"``)."""

    exx: np.ndarray
    eyy: np.ndarray
    ezz: np.ndarray
    h_bc: str
    v_bc: str
    lead_x: int
    lead_y: int


#: Physical role of a mode in an EME basis.  ``solve()`` continues to return
#: only ``"guided"`` modes; :meth:`VectorModeSolver.solve_eme_basis` additionally
#: returns PML-discretised radiation and below-cutoff evanescent modes.
ModeType = Literal["guided", "radiation", "evanescent"]


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
    nonzero wherever εyy≠εzz, the genuine anisotropic vectorial coupling."""
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
    the cross-section origin, mirrors :mod:`photonhub.analysis.modes`)."""
    return n if n % 2 == 1 else n + 1


def _detuned_eigenvalue_shift(beta_squared: complex, k0: float) -> complex:
    """Move an exact Ritz-value target off the singular shift-invert pole."""
    return complex(beta_squared + 1e-8 * k0 * k0)


def _deterministic_arpack_start(
    size: int, stream: int = 0, *, complex_dtype: bool = False
) -> np.ndarray:
    """Return a reproducible, non-pathological ARPACK starting vector.

    Recent SciPy releases seed ``eigs(rng=None)`` from operating-system entropy,
    so setting NumPy's legacy global seed does not make a solve reproducible.
    Supplying ``v0`` avoids that hidden entropy source while retaining distinct
    starts for a multi-shift search.  The stream is local to one deterministic
    solve sequence; it is deliberately independent of process-global RNG state.
    """
    seed = np.random.SeedSequence((0x50484F54, int(size), int(stream)))
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(size)
    if complex_dtype:
        values = values + 1j * rng.standard_normal(size)
    return values / np.linalg.norm(values)


@dataclass(frozen=True)
class VectorMode:
    """One full-vector waveguide mode (guided, radiation, evanescent, or bent)
    carrying all six field components.

    Attributes
    ----------
    n_eff:
        Real phase index (dimensionless). For a properly confined guided mode
        this lies between the cladding and core indices.
    k_eff:
        Nonnegative passive attenuation metadata. The
        engine/eigensolver beta is ``beta_store = k0*(n_eff + i*k_eff)``;
        ``k_eff > 0`` decays under its
        ``exp(-i*omega*t + i*beta_store*z)`` convention.  Returned field arrays
        use the conjugate EME beta ``beta_store.conjugate()``.
    n_group:
        Group index ``n_g = n_eff - lambda dn_eff/dlambda`` (waveguide
        dispersion), or ``None`` if not requested (``solve(group_index=False)``).
    window_limited:
        For a straight mode, true when less than 95% of transverse magnetic
        energy lies outside the absorbing layers. A bend inherits this flag
        from its straight partner. Widen the window before using its index.
    loss_unresolved:
        True for a bend when the radiation caustic is too close to the outer
        layer or its attenuation changes by over 10% at half layer strength.
        Widen the window or strengthen the layers before using its loss.
    ex, ey, ez, hx, hy, hz:
        The six field components on the cross-section, complex ``(ny, nx)`` arrays
        indexed ``[iy, ix]`` (row = y, column = x). The transverse pair
        ``(ex, ey)`` is jointly L2-normalized (``sum |ex|^2 + |ey|^2 == 1``) and
        phase-fixed so the dominant transverse-E component is real-positive at its
        peak; the other components share that scale.  All returned arrays use
        ``exp(+i*omega*t - i*beta_eme*z)`` with
        ``beta_eme = k0*n_eff_complex.conjugate()``.  For complex PML/bend modes
        this is intentionally the conjugate of the positive-loss metadata.
    wavelength_um:
        Free-space wavelength the mode was solved at (microns).
    dl_x_um, dl_y_um:
        Transverse grid spacings (microns), carried for the real-space helpers.
    center_offset_um:
        Optional ``(x, y)`` placement metadata (microns, mode frame: x = width,
        y = height): the ACTUAL center of the solver's sample window minus the
        center the caller REQUESTED. The cross-section mesh samplers deliberately
        snap their eps window to the simulation grid (so dielectric walls land
        at the same sub-cell positions the FDTD saw), which displaces the array
        center from the requested center by up to ~a cell; consumers that
        reconstruct real-space coordinates as ``(i - (n-1)/2)*dl + center``
        (``vector_modal_fields`` and everything built on it) shift by this
        offset so the mode lands where its window truly was. A half-window
        solve (:attr:`x_min_symmetry`) sets ``((nx-1)/2*dl_x_um, 0)``: its
        window starts on the symmetry plane, which is the requested centre, so
        column ``j`` lands at ``x = j*dl_x_um``. A further placement shift adds
        to that offset rather than replacing it. ``None`` (default, e.g. a
        hand-built or idealized-rectangle mode) means "centered as requested",
        the legacy behavior.
    x_min_symmetry:
        ``"pec"`` or ``"pmc"`` for a mode solved on a half window (the parity of
        its symmetry plane), else ``None``. The arrays hold
        the columns from the plane to the +x edge. :meth:`core_fraction`
        describes the full mode. The arrays themselves, :attr:`te_fraction`,
        :meth:`modal_power` and the overlap helpers describe the stored half
        window (with :attr:`x_min_plane_on_node`, column 0 straddles the plane
        and the first two count it at half weight, so :attr:`te_fraction`
        equals the full window's and :meth:`modal_power` is half of it): an
        overlap with a full-width partner, such as a full-window
        mode, a Gaussian beam or a recorded plane, returns about half the full
        coupling. Solve the full window for coupling to such a partner.
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
    #: mode_overlap._overlap_terms. ``VectorModeSolver(method="yee")`` modes
    #: leave it False: the method co-locates its six components on the ε cell
    #: centres before returning, because its lattice is the cross-section's
    #: own, not a simulation's grid.
    yee_staggered: bool = False
    #: Optional per-axis node coordinates of the mode's raster, RELATIVE to the
    #: requested mode centre (mode-x = width axis, len == ex.shape[1]; mode-y =
    #: height, len == ex.shape[0]). Set by GRADED-window Yee solves
    #: (nonuniform grids), where the array pitch is NOT the scalar
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
    #: :class:`~photonhub.analysis.mode_devices.ModeMonitor` uses it to build its
    #: automatic per-frequency readout bank. ``None`` for hand-built solver
    #: modes (no auto-bank) and for dispatcher solves given an ``eps_of_medium``
    #: override (id-keyed, not replayable).
    solve_params: Optional[Mapping[str, Any]] = None
    #: Role in an EME expansion.  Kept as a trailing default so existing
    #: hand-built ``VectorMode`` objects remain source compatible.
    mode_type: ModeType = "guided"
    #: Complex-coordinate Jacobian ``s_x(x) s_y(y)`` used by Lorentz/reaction
    #: overlaps for a PML-discretised modal basis. ``None`` means ordinary real
    #: quadrature.  This is deliberately separate from physical power
    #: integration, which never uses a complex contour weight.
    overlap_weights: Optional[np.ndarray] = None
    #: Boolean mask selecting the non-PML physical part of the cross-section.
    #: Hermitian Poynting-flux observables use this mask; reaction overlaps use
    #: :attr:`overlap_weights`.
    physical_mask: Optional[np.ndarray] = None
    #: ``(x, y)`` PML cell counts used for this modal solve.
    pml_cells_xy: Tuple[int, int] = (0, 0)
    #: Relative eigenpair residual ``||A h - beta^2 h||`` for spectrum/PML
    #: solves. ``None`` for legacy and hand-built modes.
    eigen_residual: Optional[float] = None
    #: Explicit provenance gate for native-staggered EME matching.  Generic
    #: ``yee_staggered`` modes include symmetry-reduced and graded solves whose
    #: reaction mass is not the full uniform box EME assumes.  This flag stays
    #: False by default and is set True only after
    #: :func:`solve_yee_eme_basis` validates the complete basis reaction Gram.
    yee_eme_compatible: bool = False
    #: Propagation axis and absolute transverse node origin for a validated
    #: Yee EME basis.  ``center_offset_um`` records only the placement
    #: relative to a requested centre, so it cannot distinguish two otherwise
    #: identical windows translated in the simulation.  These fields are set
    #: together with ``yee_eme_compatible`` and checked at every interface.
    yee_eme_axis: Optional[str] = None
    yee_eme_origin_um: Optional[Tuple[float, float]] = None
    #: Parity of the mirror plane of a half-window solve, or ``None`` (see the
    #: class docstring). A trailing default, like the fields above, so
    #: hand-built modes stay source compatible.
    x_min_symmetry: Optional[MirrorSymmetry] = None
    #: True if the straight mode or bend's straight partner has >5% of H energy in layers.
    window_limited: bool = False
    #: True when the bend attenuation has not passed the caustic and absorber checks.
    loss_unresolved: bool = False
    #: True for a half-window mode whose mirror plane passes through its
    #: column 0 (``VectorModeSolver(method="yee")``), so column ``j`` images
    #: to ``-j`` and column 0 straddles the plane: :meth:`core_fraction`
    #: rebuilds the full mode that way, and :attr:`te_fraction` and
    #: :meth:`modal_power` count column 0 at half weight. False for the
    #: default method's half windows, whose stored column ``j`` images to
    #: ``-(j + 1)``. Kept last so positional construction of the earlier
    #: fields is unaffected.
    x_min_plane_on_node: bool = False

    @property
    def shape(self) -> Tuple[int, int]:
        """``(ny, nx)`` of the field arrays."""
        return tuple(self.ex.shape)  # type: ignore[return-value]

    @property
    def n_eff_complex(self) -> complex:
        """Positive-loss complex-index metadata for the engine/eigensolver.

        ``beta_store = k0*(n_eff + i*k_eff)`` decays in
        ``exp(-i*omega*t + i*beta_store*z)`` when ``k_eff > 0``.  The returned
        field arrays use the conjugate EME phasor, so their curl/propagation
        constant is ``beta_eme = beta_store.conjugate()`` and EME propagates by
        ``exp(-i*beta_eme*L)``.  Do not insert this property directly into
        ``exp(-i*k0*n*L)``; that would turn passive loss into growth."""
        return complex(self.n_eff, self.k_eff)

    @property
    def loss_db_per_cm(self) -> float:
        """Propagation/bend loss in **dB/cm** from the imaginary modal index.

        ``alpha [1/m] = 2 k0 k_eff`` (field ~ ``exp(-alpha/2 z)`` ⇒ power ~
        ``exp(-alpha z)``), and ``loss[dB/m] = 10 alpha / ln 10``; converted to
        dB/cm. For a bent mode this is the **radiation (bend) loss**; ``0`` for a
        straight lossless mode. Solver-produced passive modes have
        ``k_eff >= 0``. A negative user-supplied value denotes gain metadata; it
        is not the negative imaginary part of a passive EME field beta."""
        k0 = 2.0 * np.pi / (self.wavelength_um * 1e-6)
        alpha = 2.0 * k0 * self.k_eff           # power attenuation [1/m]
        loss_db_per_m = 10.0 * alpha / np.log(10.0)
        return float(loss_db_per_m / 100.0)

    def _half_window_column_weights(self) -> np.ndarray:
        """Column weights ``(1, nx)`` of a half window whose mirror plane
        passes through column 0: 1/2 for column 0, 1 elsewhere."""
        weight = np.ones((1, np.asarray(self.ex).shape[1]))
        weight[0, 0] = 0.5
        return weight

    @property
    def te_fraction(self) -> float:
        """Transverse-E TE fraction ``∫|Ex|² / (∫|Ex|² + ∫|Ey|²)``, the
        polarization purity. ``~1`` for a clean TE (Ex-major) mode, ``~0`` for a
        clean TM (Ey-major) mode, ``~0.5`` for a strongly hybrid mode.

        On a half window whose mirror plane passes through column 0
        (:attr:`x_min_plane_on_node`), that column's cell straddles the plane,
        so it counts at half weight and the fraction equals the full
        window's."""
        if self.x_min_symmetry is not None and self.x_min_plane_on_node:
            weight = self._half_window_column_weights()
            px = float(np.sum(np.abs(self.ex) ** 2 * weight))
            py = float(np.sum(np.abs(self.ey) ** 2 * weight))
        else:
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
        requested mode centre: the placement
        :func:`~photonhub.analysis.mode_overlap.vector_modal_fields` uses, so this
        class reports the coordinates its own overlap consumers place it at.

        Prefers the carried graded node ladders (:attr:`x_coords_um` /
        :attr:`y_coords_um`, whose pitch is NOT the scalar ``dl_x_um``); else the
        uniform ladder PLUS :attr:`center_offset_um`, the window-placement shift a
        grid-snapped cross-section solve applies (up to ~a cell) and a half-window
        solve sets to half its width. Ignoring either mis-reports every
        coordinate by that shift."""
        ny, nx = self.ex.shape
        if self.x_coords_um is not None:
            return (np.asarray(self.x_coords_um, dtype=np.float64),
                    np.asarray(self.y_coords_um, dtype=np.float64))
        off_w, off_h = self.center_offset_um or (0.0, 0.0)
        if self.x_min_symmetry is not None:
            # half window: take the half width back out of the offset
            # first, so column j lands on exactly j*dl_x, the value the full
            # solve reports for that column. The centred ladder plus the offset
            # agrees only to rounding, which misaligns the two DataArrays.
            xs = (np.arange(nx) * self.dl_x_um
                  + (off_w - 0.5 * (nx - 1) * self.dl_x_um))
        else:
            xs = (np.arange(nx) - (nx - 1) / 2.0) * self.dl_x_um + off_w
        ys = (np.arange(ny) - (ny - 1) / 2.0) * self.dl_y_um + off_h
        return xs, ys

    def field_dataarray(self, component: str = "Ex") -> xr.DataArray:
        """One field ``component`` as an :class:`xarray.DataArray` with real-space
        ``x``/``y`` coordinates in microns (origin at the REQUESTED cross-section
        centre; a grid-snapped or half-window solve's :attr:`center_offset_um` and a
        graded solve's node ladders are honoured, see :meth:`_node_coords_um`). Dims
        ``("y", "x")``. A half-window mode's ``x`` starts at 0 on its symmetry
        plane and equals the full solve's ``x`` from the centre column on.

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
        ``core_w_um x core_h_um`` bounding box, a confinement metric (1.0 = fully
        confined). The box is centered on the REQUESTED cross-section centre,
        matching :meth:`VectorModeSolver.from_rectangular_core`; the mode's own
        node coordinates honour a grid-snapped or half-window solve's
        :attr:`center_offset_um` and a graded solve's ladders (see
        :meth:`_node_coords_um`), so the box lands on the real core instead of
        being displaced by the window placement.

        On a half-window mode (:attr:`x_min_symmetry` set) the fraction is the
        full mode's, laid out as the full solve's arrays hold it. Those arrays
        report each sample half a cell before its node, so the mirror image of
        stored column ``j`` sits at ``x = -(j+1)*dl_x_um`` from the plane, one
        column further out than the column itself. Every stored column but the
        last, on the far wall, therefore counts twice, and the box is tested at
        both positions. A full solve's H arrays are exact mirror images in this
        layout. Its E arrays are not beside a vertical core wall, where the two
        samples of a pair divide by the permittivity of different cells, so the
        result can differ from the full solve's own value by O(dl) there.

        A ``method="yee"`` half mode (:attr:`x_min_plane_on_node`) has its
        plane on node 0 and every component co-located on the nodes, so its
        column 0 is its own image and stored column ``j`` images to
        ``x = -j*dl_x_um``. Rebuilt that way, the fraction equals the full
        Yee solve's to eigensolver precision."""
        xs_c, ys_c = self._node_coords_um()
        if self.x_min_symmetry is not None and self.x_min_plane_on_node:
            # Plane on node 0: column j pairs with -j, column 0 its own image.
            p = np.abs(self.ex) ** 2 + np.abs(self.ey) ** 2
            nx = p.shape[1]
            p = np.concatenate([p[:, :0:-1], p], axis=1)
            xs_c = np.concatenate(
                [xs_c[0] - np.arange(nx - 1, 0, -1) * self.dl_x_um, xs_c])
            inside = ((np.abs(ys_c)[:, None] <= core_h_um / 2.0 + 1e-12) &
                      (np.abs(xs_c)[None, :] <= core_w_um / 2.0 + 1e-12))
            total = float(p.sum())
            return float(p[inside].sum() / total) if total > 0.0 else 0.0
        p = np.abs(self.ex) ** 2 + np.abs(self.ey) ** 2
        if self.x_min_symmetry is not None:
            # Half-window solve: the full solve pairs column c0 + j with c0 - 1 - j, so the
            # image of half column j is reported at xs_c[0] - (j + 1)*dl_x and
            # the far-wall column has none. Rebuild that layout.
            nx = p.shape[1]
            p = np.concatenate([p[:, -2::-1], p], axis=1)
            xs_c = np.concatenate(
                [xs_c[0] - np.arange(nx - 1, 0, -1) * self.dl_x_um, xs_c])
        xs = np.abs(xs_c)
        ys = np.abs(ys_c)
        inside = (ys[:, None] <= core_h_um / 2.0 + 1e-12) & \
                 (xs[None, :] <= core_w_um / 2.0 + 1e-12)
        total = float(p.sum())
        if total <= 0.0:
            return 0.0
        return float(p[inside].sum() / total)

    def modal_power(self) -> float:
        """Time-averaged modal power (watts) carried across the cross-section by
        this full-vector mode, the **physical** z-Poynting flux

            P = (1/2) integral Re( E x H* ) . z_hat dA
              = (1/2) integral Re( Ex Hy* - Ey Hx* ) dA ,

        evaluated on the mode's own transverse grid (``dA = dl_x * dl_y`` in m²,
        midpoint quadrature). For a PML-discretized EME mode,
        :attr:`physical_mask` excludes the artificial complex-contour cells from
        this physical observable. The six components are in SI units (E in V/m,
        H in A/m), so the integral is in watts. Used by the mode-source
        builder to scale the injected profiles to **1 W**
        (``profile /= sqrt(P)``), and by any consumer that wants the absolute
        (not just relative) modal power.

        For the dominant-forward (``+z``) guided mode this is positive; a sign
        flip indicates a predominantly backward eigenpair.

        On a half window whose mirror plane passes through column 0
        (:attr:`x_min_plane_on_node`), that column counts at half weight, so
        the result is the power through the stored half window, half the full
        window's.
        """
        dx = self.dl_x_um * 1e-6
        dy = self.dl_y_um * 1e-6
        sz = np.real(self.ex * np.conj(self.hy) - self.ey * np.conj(self.hx))
        if self.x_min_symmetry is not None and self.x_min_plane_on_node:
            sz = sz * self._half_window_column_weights()
        if self.physical_mask is not None:
            mask = np.asarray(self.physical_mask, dtype=bool)
            if mask.shape != sz.shape:
                raise ValueError(
                    "physical_mask shape must match the modal field grid"
                )
            sz = sz[mask]
        return 0.5 * float(np.sum(sz)) * dx * dy



def _merge_eigen_candidate(
    candidates: list, val: complex, vec: np.ndarray, residual: float,
    k0: float,
) -> None:
    """Add one shift-invert eigenpair to ``candidates`` (a list of
    ``(eigenvalue, vector, residual)``) unless it repeats a direction already
    held.

    Repeated searches return arbitrary rotations of a degenerate eigenspace,
    and pairwise collinearity is not enough: three rotations of a
    two-dimensional eigenspace are pairwise distinct but jointly rank
    deficient. Within an eigenvalue cluster a vector is kept only when its
    remainder after the cluster's span exceeds ``_EIGENSPACE_NEW_DIRECTION_MIN``.
    A vector that instead copies one held member is kept in that member's
    place when its residual is lower, so the more accurate copy survives
    whichever shift found it first."""
    same = [
        i for i, (old_val, _, _) in enumerate(candidates)
        if abs(val - old_val) <= 1e-7 * max(abs(val), abs(old_val), k0 * k0)
    ]
    vec_norm = float(np.linalg.norm(vec))
    if not same or vec_norm == 0.0:
        candidates.append((val, vec, residual))
        return
    cluster = np.column_stack([candidates[i][1] for i in same])
    q, _ = np.linalg.qr(cluster, mode="reduced")
    remainder = vec - q @ (q.conj().T @ vec)
    if float(np.linalg.norm(remainder)) / vec_norm > _EIGENSPACE_NEW_DIRECTION_MIN:
        candidates.append((val, vec, residual))
        return
    unit = vec / vec_norm
    for i in same:
        old = candidates[i][1]
        old_unit = old / np.linalg.norm(old)
        off = unit - old_unit * np.vdot(old_unit, unit)
        if (float(np.linalg.norm(off)) <= _EIGENSPACE_NEW_DIRECTION_MIN
                and residual < candidates[i][2]):
            candidates[i] = (val, vec, residual)
            return

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
        a DIFFERENT polarization, the operator is in H, so the electric wall
        (``"none"``) yields the slab **TE0** (Eₓ-major) and the magnetic wall
        (``"pmc"``) the slab **TM0** (E_y-major). Choose the wall to match the
        polarization you want (the wrong one returns a too-low n_eff; see
        :meth:`from_rectangular_core`). With ``x_min_symmetry`` set it governs
        the x-max wall only.
    x_min_symmetry:
        ``None`` (default) or a symmetry plane on the x-min edge: ``"pec"``
        (odd / electric: Hₓ odd, H_y even, so Eₓ even) or ``"pmc"`` (even /
        magnetic: the complementary parity). The plane passes through the centre
        of ε column 0. So ``eps`` is the half of a mirror-symmetric cross-section
        that keeps the column straddling the plane. The plane
        is the x origin of the returned modes: column ``j`` reports
        ``x = j*dl_x_um`` through :attr:`VectorMode.center_offset_um`. A caller
        whose plane sits elsewhere in its own frame centres the mode on the
        plane when placing it, as with any solve. Overlaps of such a mode cover
        the half window only (see :attr:`VectorMode.x_min_symmetry`). Bent
        solves and an x PML are rejected: a bend has no mirror symmetry, and
        the PML would cover the plane. With ``method="yee"`` the plane is the
        node line through the centre of ε column 0 and the parities keep the
        same families: ``"pec"`` pins ``Ey`` and ``Ez`` to zero on the plane
        (``Ex`` even; TE0), ``"pmc"`` keeps the complementary family (TM0,
        TE1).
    method:
        The cross-section discretization. ``"flm"`` (default) is the
        transverse-H operator described above. ``"yee"`` solves the
        transverse electric field on a staggered Yee lattice (Z. Zhu and
        T. G. Brown, Opt. Express 10, 853 (2002)). The centre of ε cell
        ``[iy, ix]`` is lattice node ``(ix, iy)`` and carries ``Ez``; ``Ex``
        sits on the centre of the cell's +x face and ``Ey`` on the centre of
        its +y face, and each component sees its own permittivity sample.
        Prefer it for high-contrast cores with corners, such as a silicon
        strip, where it converges faster under mesh refinement than the
        default, which loses accuracy at the corners. Its walls are electric
        (PEC) and lie on node lines half a cell outside the ε grid on every
        side, so opposite walls are mirror images of each other; a
        half-window solve (``x_min_symmetry``) therefore reproduces the full
        solve's index of that parity to eigensolver precision. Returned modes
        hold one entry per ε cell for every component, co-located on the cell
        centres: a component half a cell off takes the mean of its two
        neighbours (``Hz`` of its four). That mean is second order where the
        field is smooth and first order across a material face, where the
        normal E component jumps; it leaves ``n_eff`` and ``n_group``
        unchanged and moves :attr:`VectorMode.te_fraction` at first order
        (2.1e-4 for a 500 x 220 nm Si strip's TE0 at 5 nm cells). They carry
        ``yee_staggered=False`` like the default method's, so every consumer
        reads them as node-collocated fields and none mistakes them for modes
        solved on a simulation's own grid. Limits: straight solves
        only (``bend_radius_um``, ``num_pml`` and :meth:`solve_eme_basis`
        raise ``NotImplementedError``; they are planned), square cells
        (``dl_x_um == dl_y_um``) and ``x_symmetry="none"``. From a
        cell-centred ``eps`` (or ``eps_tensor``) the staggered values are
        derived from the cells: εzz is the cell's value and εxx at a face the
        harmonic mean of the two cells sharing that x face, εyy likewise across
        a y face (the outermost faces take their one cell's value). This
        re-averages an already averaged grid and is less accurate than
        sampling the geometry at each component, which
        :meth:`from_rectangular_core` does and ``yee_eps_tensor`` accepts.
    yee_eps_tensor:
        Optional ``(εxx, εyy, εzz)`` sampled at the Yee positions of each
        cell, each shaped like ``eps``: εxx at the centre of the cell's +x
        face, εyy at the centre of its +y face, εzz at its centre. Only with
        ``method="yee"``; it replaces the values derived from ``eps``.
    yee_eps_outer:
        Optional ``(εxx, εyy)`` for the faces that are no cell's +x or +y
        face: εxx on the -x face of column 0 (shape ``(ny,)``) and εyy on the
        -y face of row 0 (shape ``(nx,)``). Both lie on the low edge of the
        ``eps`` grid, half a cell inside the low walls; a mirror plane or a
        periodic axis does not use the x one. Only with ``yee_eps_tensor``.
        Without it those
        faces repeat the sample of the same cell's +x (+y) face, which is
        exact only when the material is uniform across both cells, so pass
        it when geometry touches the window's low x or y edge.
    """

    #: Speed of light in vacuum (m/s), matching :mod:`photonhub.analysis.modes`.
    C0: float = _C0  # speed of light (m/s), shared analysis._constants value

    #: Hard cap on ``N = nx*ny`` (the eigenproblem is ``2N``). The sparse
    #: shift-invert solve is far lighter than the semi-vec's dense path, so the
    #: cap is generous; a single-mode strip needs only a modest window.
    MAX_UNKNOWNS: int = 40000

    @legacy_keywords(wavelength_um="wlen_um")
    def __init__(
        self,
        eps: np.ndarray,
        dl_x_um: float,
        dl_y_um: float,
        wlen_um: float,
        x_symmetry: XSymmetry = "none",
        *,
        eps_tensor: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        x_min_symmetry: Optional[MirrorSymmetry] = None,
        method: ModeMethod = "flm",
        yee_eps_tensor: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        yee_eps_outer: Optional[Tuple[np.ndarray, np.ndarray]] = None,
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
        if not (wlen_um > 0):
            raise ValueError("wlen_um must be > 0")
        if x_symmetry not in ("none", "pmc"):
            raise ValueError(
                f"x_symmetry must be 'none' or 'pmc', got {x_symmetry!r}")
        if x_min_symmetry not in (None, "pec", "pmc"):
            raise ValueError(
                "x_min_symmetry must be None, 'pec' (odd / electric), or 'pmc' "
                f"(even / magnetic), got {x_min_symmetry!r}")

        self.eps: np.ndarray = eps_arr
        self.dl_x_um: float = float(dl_x_um)
        self.dl_y_um: float = float(dl_y_um)
        self.wavelength_um: float = float(wlen_um)
        self.x_symmetry: XSymmetry = x_symmetry
        self.x_min_symmetry: Optional[MirrorSymmetry] = x_min_symmetry

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

        if method not in ("flm", "yee"):
            raise ValueError(
                f"method must be 'flm' (default) or 'yee', got {method!r}")
        self.method: ModeMethod = method
        # method="yee": permittivity at the Yee positions of each cell (εxx on
        # its +x face, εyy on its +y face, εzz at its centre), the -x face of
        # column 0 and the -y face of row 0, and which axes close periodically
        # (a slab spanning the window, set by from_rectangular_core).
        self._yee_samples: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._yee_outer: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._yee_periodic: Tuple[bool, bool] = (False, False)
        if method == "flm":
            if yee_eps_tensor is not None or yee_eps_outer is not None:
                raise ValueError(
                    "yee_eps_tensor and yee_eps_outer are used only with "
                    "method='yee'")
            return
        if self.dl_x_um != self.dl_y_um:
            raise ValueError(
                "method='yee' needs square cells (dl_x_um == dl_y_um), got "
                f"{self.dl_x_um!r} and {self.dl_y_um!r}")
        if x_symmetry != "none":
            raise ValueError(
                "method='yee' has electric walls only (x_symmetry='none'); a "
                "core as wide as the window in from_rectangular_core is solved "
                "as an x-invariant slab, which returns both slab polarizations")
        if yee_eps_tensor is None:
            if yee_eps_outer is not None:
                raise ValueError("yee_eps_outer needs yee_eps_tensor")
            self._yee_samples, self._yee_outer = self._yee_samples_from_cells()
            return
        comps = tuple(np.asarray(c, dtype=float) for c in yee_eps_tensor)
        if len(comps) != 3 or any(c.shape != eps_arr.shape for c in comps):
            raise ValueError(
                "yee_eps_tensor must be three arrays (εxx, εyy, εzz) matching "
                f"eps shape {eps_arr.shape}")
        if any(not np.all(np.isfinite(c)) or np.any(c < 1.0) for c in comps):
            raise ValueError("yee_eps_tensor components must be finite and >= 1")
        self._yee_samples = comps
        if yee_eps_outer is None:
            self._yee_outer = (comps[0][:, 0].copy(), comps[1][0, :].copy())
            return
        outer = tuple(np.asarray(c, dtype=float) for c in yee_eps_outer)
        if (len(outer) != 2 or outer[0].shape != (eps_arr.shape[0],)
                or outer[1].shape != (eps_arr.shape[1],)):
            raise ValueError(
                "yee_eps_outer must be (εxx on the -x face of column 0, shape "
                f"({eps_arr.shape[0]},), εyy on the -y face of row 0, shape "
                f"({eps_arr.shape[1]},))")
        if any(not np.all(np.isfinite(c)) or np.any(c < 1.0) for c in outer):
            raise ValueError("yee_eps_outer samples must be finite and >= 1")
        self._yee_outer = outer

    def _yee_samples_from_cells(self):
        """Yee-position permittivity derived from the cell-centred tensor:
        εzz the cell value, εxx on the +x face the harmonic mean of the two
        cells sharing it, εyy likewise on the +y face. The faces outside the
        raster (+x of the last column, +y of the last row, and the -x / -y
        faces of the first) take their one cell's value."""
        def face_mean(a, axis):
            following = np.concatenate(
                (np.delete(a, 0, axis=axis), np.take(a, [-1], axis=axis)),
                axis=axis)
            # Equal neighbours keep their value bit for bit, so a mirror
            # symmetric raster gives a mirror symmetric lattice.
            return np.where(a == following, a,
                            2.0 / (1.0 / a + 1.0 / following))

        samples = (face_mean(self._exx, 1), face_mean(self._eyy, 0), self._ezz)
        return samples, (self._exx[:, 0].copy(), self._eyy[0, :].copy())

    def at_wlen(self, wlen_um: float) -> "VectorModeSolver":
        """A sibling solver on the SAME cross-section (``eps`` / tensor, ``dl``,
        symmetry) at a new free-space wavelength, for re-solving a full-vector
        mode across a frequency band (``wavelength_um = C0 / freq_hz * 1e6``)
        without repeating mesh sampling. Mirrors :meth:`ModeSolver.at_wavelength`; the
        permittivity (and any KFJ tensor) is shared by reference."""
        eps_tensor = (
            (self._exx, self._eyy, self._ezz) if self._is_tensor else None
        )
        if self.method == "yee":
            sibling = VectorModeSolver(
                self.eps, self.dl_x_um, self.dl_y_um, wlen_um,
                self.x_symmetry, eps_tensor=eps_tensor,
                x_min_symmetry=self.x_min_symmetry, method="yee",
                yee_eps_tensor=self._yee_samples, yee_eps_outer=self._yee_outer,
            )
            sibling._yee_periodic = self._yee_periodic
            return sibling
        return VectorModeSolver(
            self.eps, self.dl_x_um, self.dl_y_um, wlen_um,
            self.x_symmetry, eps_tensor=eps_tensor,
            x_min_symmetry=self.x_min_symmetry,
        )

    def at_wavelength(self, wavelength_um: float) -> "VectorModeSolver":
        """Deprecated spelling of :meth:`at_wlen`."""
        warnings.warn(
            "VectorModeSolver.at_wavelength() was renamed to at_wlen(); the old name "
            "will be removed in a future release", DeprecationWarning, stacklevel=caller_stacklevel())
        return self.at_wlen(wavelength_um)

    # -- convenience cross-section builder ---------------------------------

    @classmethod
    @legacy_keywords(wavelength_um="wlen_um")
    def from_rectangular_core(
        cls,
        *,
        wlen_um: float,
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
        method: ModeMethod = "flm",
    ) -> "VectorModeSolver":
        """Build a solver for a centered rectangular core in a uniform cladding.

        Samples the canonical strip-waveguide cross-section onto a *square*
        uniform grid of spacing ``dl_um`` (the same mesh sampling as
        :meth:`photonhub.analysis.modes.ModeSolver.from_rectangular_core`, so the
        full-vector and semi-vec solvers are drop-in comparable).

        Parameters
        ----------
        wlen_um:
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
            x-uniform (kx = 0), the 1-D slab limit. **The wall selects the
            polarization**, because the operator is in H: an electric wall
            (``"none"``) makes the major-H component Neumann (and Hₓ=0), so its
            x-uniform fundamental is the slab **TE0** (Eₓ-major); a magnetic wall
            (``"pmc"``) flips the parity, so its x-uniform fundamental is the slab
            **TM0** (E_y-major). Picking the *wrong* wall for the polarization you
            want does not error, it returns a mode of that polarization forced to
            zero at the walls (a spurious kx ≈ π/window_w), whose n_eff is too low.
            So: ``"none"`` for a TE slab, ``"pmc"`` for a TM slab.
        x_min_symmetry:
            ``None`` (default, full cross-section), ``"pec"``, or ``"pmc"`` to
            exploit a mirror symmetry about the width center.
            The solver keeps the ε columns from the centre column, the one
            straddling the plane, to the +x edge. The eigenproblem is about half
            the size, and every kept H node is a node of the full grid.
            ``"pec"`` makes the x-min wall an odd / electric plane: it keeps the
            modes whose Eₓ is even and E_y odd in x, such as TE0. ``"pmc"`` makes
            it an even / magnetic plane with the complementary parity, such as
            TM0 and the width-odd TE1. The half solve reproduces the full solve's
            modes of that parity at the same ``dl``, to eigensolver precision:
            equal ``n_eff``, and field arrays equal to the full ones from the
            centre column on, up to normalization. Its modes report ``x`` from
            the plane, the core centre, so column ``j`` sits at ``x = j*dl_um``
            as in the full solve. Their overlaps cover the half window only, so
            an overlap with a full-width partner (a fibre mode, say) returns
            about half the full coupling; see :attr:`VectorMode.x_min_symmetry`.
            ``x_symmetry`` still sets the far (x-max) wall, so the default
            matches the full solve.
        subpixel:
            ``False`` hard-samples ε (staircase). ``True`` (default) smooths the
            high-contrast walls; the kind is set by ``subpixel_method``.
        subpixel_method:
            ``"tensor"`` (default) applies a diagonalized
            **Kottke–Farjadpour–Johnson** subpixel tensor, the interface-normal
            ε-component gets the harmonic mean and tangential components the
            arithmetic mean. This is exact on the rectangle's axis-aligned face
            cells; at a partially filled corner the diagonal-only FLM solver
            omits KFJ's off-diagonal xy term. ``"volume"`` uses the scalar
            area-average. Ignored when ``subpixel=False``.
        method:
            ``"flm"`` (default) or ``"yee"``, the discretization described on
            :class:`VectorModeSolver`. With ``"yee"`` each field component's
            permittivity is sampled from the exact rectangle over a cell
            centred on that component: εxx at ``Ex`` is the arithmetic mean
            along y of the harmonic means along x, εyy at ``Ey`` the same with
            x and y exchanged, and εzz at ``Ez`` the area mean. On a cell cut
            by one core face this is the diagonal of the Kottke, Farjadpour and
            Johnson subpixel tensor (Phys. Rev. E 77, 036611 (2008)); a corner
            cell takes the same nested mean and, as in the default, no
            off-diagonal term. ``subpixel_method="volume"`` takes the area mean
            for every component and ``subpixel=False`` the value at each
            component's position. The core centre is a lattice node, the
            centre of the middle ε cell, where the default method centres it,
            so both methods solve the same geometry; the electric walls sit
            half a cell outside the default method's window on every side.
            ``x_min_symmetry`` puts the mirror plane on that centre node line,
            and the half solve reproduces the full solve's index of its parity
            to eigensolver precision. A core as wide (tall) as the window is an
            x-invariant (y-invariant) slab: that axis closes periodically
            across the window instead of meeting walls, so the slab modes of
            both polarizations are returned uniform along it, without
            ``x_symmetry`` (``x_symmetry="pmc"`` is rejected).
            ``x_min_symmetry`` is refused on an x-invariant slab only; a
            y-invariant slab keeps the x mirror plane.
        """
        if method not in ("flm", "yee"):
            raise ValueError(
                f"method must be 'flm' (default) or 'yee', got {method!r}")
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
        # Half-window solve: keep the ε columns [centre .. +x edge]. nx is odd, so column
        # (nx-1)//2 is centred on the plane x = 0; H node j of the half is node
        # (nx-1)//2 + j of the full grid, at x = (j + 1/2)·dl, and both solves
        # report it at its ε column centre x = j·dl. The solver folds the x-min
        # wall about the plane itself (half a cell before node 0); x_symmetry
        # keeps setting the far wall, as in the full solve.
        half = x_min_symmetry is not None
        c0 = (nx - 1) // 2 if half else 0

        if method == "yee":
            if x_symmetry != "none":
                raise ValueError(
                    "method='yee' has electric walls only (x_symmetry='none'); "
                    "a core as wide as the window is solved as an x-invariant "
                    "slab, which returns both slab polarizations")
            slab_x = core_w_um >= window_w_um - 1e-12
            slab_y = core_h_um >= window_h_um - 1e-12
            if half and slab_x:
                raise ValueError(
                    "x_min_symmetry needs a core narrower than the window; a "
                    "core as wide as the window is an x-invariant slab, which "
                    "method='yee' solves without a mirror plane")
            eps = cls._rasterize_rect(
                nx, ny, dl_um, dl_um, cw, ch, ec, ecl, subpixel=subpixel)
            samples, outer = cls._yee_rect_samples(
                nx, ny, dl_um, cw, ch, ec, ecl, subpixel=subpixel,
                subpixel_method=subpixel_method)
            if half:
                eps = eps[:, c0:]
                samples = tuple(s[:, c0:] for s in samples)
                outer = (outer[0], outer[1][c0:])
            solver = cls(eps, dl_um, dl_um, wlen_um,
                         x_min_symmetry=x_min_symmetry, method="yee",
                         yee_eps_tensor=samples, yee_eps_outer=outer)
            solver._yee_periodic = (slab_x, slab_y)
            return solver

        if subpixel and subpixel_method == "tensor":
            eps, tensor = cls._kfj_tensor_rect(
                nx, ny, dl_um, dl_um, cw, ch, ec, ecl)
            if half:
                eps = eps[:, c0:]
                tensor = tuple(t[:, c0:] for t in tensor)
            return cls(eps, dl_um, dl_um, wlen_um,
                       x_symmetry=x_symmetry, eps_tensor=tensor,
                       x_min_symmetry=x_min_symmetry)
        eps = cls._rasterize_rect(
            nx, ny, dl_um, dl_um, cw, ch, ec, ecl, subpixel=subpixel)
        if half:
            eps = eps[:, c0:]
        return cls(eps, dl_um, dl_um, wlen_um, x_symmetry=x_symmetry,
                   x_min_symmetry=x_min_symmetry)

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
        it overlaps, `eps = f*eps_core + (1-f)*eps_clad`, `f` the EXACT separable
        axis-aligned fill fraction (box-cap KFJ). This removes the staircase
        at the high-contrast walls, so n_eff converges smoothly (the same
        accuracy-vs-resolution win the FDTD engine gets from subpixel)."""
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
        """Centered rectangular core with a diagonalized
        **Kottke–Farjadpour–Johnson** subpixel tensor. Returns
        ``(eps_scalar, (εxx, εyy, εzz))`` indexed ``[iy, ix]``.

        Each cell's fill fraction ``f`` (the exact separable axis-aligned overlap)
        and the interface normal ``n̂ = ∇f/|∇f|`` give a diagonal effective tensor:
        the normal component carries the harmonic mean ``ε⊥ = 1/(f/ε_in +
        (1-f)/ε_out)`` (the correct average for the discontinuous normal **D**),
        the tangential ones the arithmetic mean ``ε∥ = f·ε_in + (1-f)·ε_out``,
        combined as ``ε_aa = ε∥ + (ε⊥-ε∥)·n̂_a²`` (KFJ; the FDTD engine's subpixel
        smoothing). ``z`` is always tangential to a z-invariant cross-section, so
        ``εzz = ε∥``. Interior/exterior cells have ``f ∈ {0,1}`` (``ε⊥ = ε∥``), so
        the tensor is isotropic there. On face cells the interface normal is
        axis-aligned and this diagonal form is exact. At partially filled corner
        cells the full KFJ transform also has
        ``εxy=(ε⊥-ε∥) n_x n_y``; the diagonal-only FLM surface omits that
        small-measure term. ``eps_scalar`` is the volume average ``ε∥`` (the
        representative scalar for ``self.eps`` / ``n_guess``)."""
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

    @staticmethod
    def _yee_rect_samples(
        nx: int, ny: int, dl_um: float,
        core_w_um: float, core_h_um: float,
        eps_core: float, eps_clad: float,
        *, subpixel: bool = True, subpixel_method: SubpixelMethod = "tensor",
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray]]:
        """Permittivity of a centred rectangle at the Yee positions of each
        cell of the ``ny x nx`` raster, for ``method="yee"``.

        Cell ``[iy, ix]`` is centred at ``((ix - (nx-1)/2) dl, (iy - (ny-1)/2)
        dl)``, the default method's raster. ``Ez`` sits there, ``Ex`` on the
        cell's +x face and ``Ey`` on its +y face. Each value is the mean over
        a ``dl x dl`` cell centred on its component, from the exact
        separable overlap fractions ``fx``, ``fy`` of that cell with the core:
        ``εxx = fy·H(fx) + (1 - fy)·ε_clad`` with ``H(f) = 1/(f/ε_core +
        (1 - f)/ε_clad)`` (harmonic along x, arithmetic along y), ``εyy`` the
        same with x and y exchanged, ``εzz = f·ε_core + (1 - f)·ε_clad`` with
        ``f = fx·fy``. ``subpixel_method != "tensor"`` takes that area mean
        for all three and ``subpixel=False`` the value at each component's
        position. Returns ``((εxx, εyy, εzz), (εxx on the -x face of column
        0, εyy on the -y face of row 0))``. Positions are integer or
        half-integer multiples of ``dl``, so mirror pairs are exact negatives
        and a symmetric core gives a lattice symmetric to the last bit."""
        ix = np.arange(nx) - (nx - 1) / 2.0
        iy = np.arange(ny) - (ny - 1) / 2.0
        x_node, x_face = ix * dl_um, (ix + 0.5) * dl_um
        y_node, y_face = iy * dl_um, (iy + 0.5) * dl_um
        x_out = np.array([(ix[0] - 0.5) * dl_um])     # -x face of column 0
        y_out = np.array([(iy[0] - 0.5) * dl_um])     # -y face of row 0
        hw, hh = core_w_um / 2.0, core_h_um / 2.0

        if not subpixel:
            def sample(xs, ys, _component):
                core = ((np.abs(ys)[:, None] <= hh + 1e-9) &
                        (np.abs(xs)[None, :] <= hw + 1e-9))
                return np.where(core, eps_core, eps_clad)
        else:
            def frac(centers, half):
                lo = centers - dl_um / 2.0
                hi = centers + dl_um / 2.0
                return np.clip(np.minimum(hi, half) - np.maximum(lo, -half),
                               0.0, dl_um) / dl_um

            def harmonic(f):
                return 1.0 / (f / eps_core + (1.0 - f) / eps_clad)

            def sample(xs, ys, component):
                fx = frac(xs, hw)[None, :]
                fy = frac(ys, hh)[:, None]
                if subpixel_method == "tensor" and component == "xx":
                    return fy * harmonic(fx) + (1.0 - fy) * eps_clad
                if subpixel_method == "tensor" and component == "yy":
                    return fx * harmonic(fy) + (1.0 - fx) * eps_clad
                f = fx * fy
                return f * eps_core + (1.0 - f) * eps_clad

        samples = (sample(x_face, y_node, "xx"), sample(x_node, y_face, "yy"),
                   sample(x_node, y_node, "zz"))
        outer = (sample(x_out, y_node, "xx")[:, 0],
                 sample(x_node, y_out, "yy")[0, :])
        return samples, outer

    def _yee_lattice(self) -> _YeeLattice:
        """The solve lattice of ``method="yee"``.

        A walled axis gains one leading node line, half a cell outside the
        raster: its low wall (``"pec"``). The operator's high wall is the node
        line one past the last, half a cell outside the raster on the other
        side, so the two walls are mirror images. Along x a mirror plane
        (``x_min_symmetry``) replaces the leading line: node 0 is the centre
        of ε column 0, on the plane. A periodic axis has no wall line."""
        exx, eyy, ezz = self._yee_samples
        lo_exx, lo_eyy = self._yee_outer
        per_x, per_y = self._yee_periodic
        lead_y = 0 if per_y else 1
        lead_x = 0 if (per_x or self.x_min_symmetry is not None) else 1
        if lead_y:
            # Wall row: εyy at the -y face of row 0; εxx and εzz there act on
            # values the wall pins to zero, so they repeat row 0.
            exx = np.vstack((exx[:1], exx))
            eyy = np.vstack((lo_eyy[None, :], eyy))
            ezz = np.vstack((ezz[:1], ezz))
            lo_exx = np.concatenate((lo_exx[:1], lo_exx))
        if lead_x:
            exx = np.hstack((lo_exx[:, None], exx))
            eyy = np.hstack((eyy[:, :1], eyy))
            ezz = np.hstack((ezz[:, :1], ezz))
        h_bc = "periodic" if per_x else (self.x_min_symmetry or "pec")
        v_bc = "periodic" if per_y else "pec"
        return _YeeLattice(exx, eyy, ezz, h_bc, v_bc, lead_x, lead_y)

    # -- bent-waveguide curvature + tangential PML helpers -----------------

    def _x_centers_um(self, nx: int) -> np.ndarray:
        """In-plane x cell-center coordinates (µm), origin at the cross-section
        center, +x outward from the bend center, the radial offset the
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
        at tight radii vs an independent bend solve). In the engine convention an
        azimuthal mode evolves as ``exp(+i ν_store φ)``; returned arrays use the
        conjugate EME phasor. A reported phase index gives
        ``Re(ν_store) = n_eff·k0·R`` (referenced at ``R``, so ``R→∞`` recovers the
        straight ``n_eff``). The squared generalized eigenproblem is insensitive
        to that phasor sign: its **longitudinal** term is ``ν²/r²`` instead of a
        constant ``β²``. With ``β₀ = n_eff·k0 = Re(ν_store)/R`` the transverse
        wave equation reads

            ∇ₜ² h + k0² ε(x,y) h = β₀² (R/r)² h ,

        i.e. a **generalized eigenproblem** ``A h = β₀² B h`` with the diagonal
        mass matrix ``B = diag((R/r)²)`` and ``A`` the *ungraded* (physical-ε)
        straight transverse-H operator. The ``(R/r)²`` "centrifugal" weight is the
        full non-perturbative curvature effect, the dominant term at tight bends,
        where the reported ``n_eff(R)`` rises far faster than ``1/R``.

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
        if np.any((R + x) * R <= 0):
            raise ValueError("bend radius R places a cell at or across the bend axis")
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
        """Complex coordinate-stretch factor ``s = 1 + i σ/k0`` along one axis
        for a PML of ``pml_cells`` layers on both ends.

        Standard polynomial UPML conductivity profile ``σ(ξ) = σ_max ξ^m`` over the
        normalized PML depth ``ξ∈[0,1]`` (``m=3``); the complex stretch
        ``1 + iσ/k0`` enters the FLM half-cell spacings (so ``∂x → ∂x/s``),
        attenuating outgoing radiation and turning the eigenvalue complex. The
        ``+i`` sign belongs to the engine/operator convention and gives a
        radiating mode positive ``k_eff`` metadata. EME return paths conjugate
        the stretch, eigenfield, and beta together, while retaining that
        positive-loss metadata for propagation. The PML sits
        *outside* the guided core (the window is cladding-padded), so it does not
        perturb the bound mode while absorbing the leaky/radiated tail. Returns
        all-ones (real, no stretch) when ``pml_cells == 0``, the straight,
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

    @staticmethod
    def _pml_strength_for_imaginary_thickness(
        pml_cells: int, dl_um: float, imaginary_thickness_um: float,
    ) -> float:
        """Convert an integrated imaginary contour thickness to peak strength.

        The stretch is ``s = 1 + i strength * depth**3``.  Describing a PML by
        ``sum(Im(s) * dl)`` makes grid and PML-cell convergence meaningful,
        unlike holding the peak profile value fixed while changing the mesh.
        """
        if pml_cells <= 0:
            return 0.0
        depth = (np.arange(pml_cells, dtype=float) + 1.0) / pml_cells
        denom = float(dl_um * np.sum(depth ** 3))
        return float(imaginary_thickness_um / denom) if denom > 0.0 else 0.0

    def _core_confinement(
        self, vecs: np.ndarray, nx: int, ny: int,
    ) -> np.ndarray:
        """High-permittivity energy fraction of each eigenvector column.

        This is a diagnostic, not a selection rule: a rib's slab can cross the
        absorbing layers. The "core" is the high-ε region (``ε`` above the midpoint
        between the window's min and max), a geometry-free proxy that works for
        the raw-eps solver. Returns a length-``vecs.shape[1]`` array in ``[0, 1]``.
        Its value depends on the geometry and does not establish whether an
        eigenpair is localized in an absorbing layer.

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

    @staticmethod
    def _interior_energy(vecs: np.ndarray, nx: int, ny: int,
                         num_pml: int) -> np.ndarray:
        """Fraction of transverse H energy outside both absorbing x layers.

        This uses layer *coordinates*, never a permittivity-defined core: a
        rib's slab can extend into the layers and must not promote a layer mode.
        Eigenvectors are flattened in (ix, iy) order in two H blocks.
        """
        n = nx * ny
        power = (np.abs(vecs[:n]) ** 2 + np.abs(vecs[n:]) ** 2)
        power = power.reshape(nx, ny, -1)
        total = power.sum(axis=(0, 1))
        return np.divide(
            power[num_pml:nx - num_pml].sum(axis=(0, 1)), total,
            out=np.zeros_like(total), where=total > 0,
        )

    @staticmethod
    def _field_overlap(first: np.ndarray, second: np.ndarray) -> float:
        norm = np.linalg.norm(first) * np.linalg.norm(second)
        return float(abs(np.vdot(first, second)) / norm) if norm else 0.0

    def _layered_straight_pairs(
        self, wavelength_um: float, num_modes: int, n_guess: Optional[float],
        num_pml: int, pml_strength: float,
        rejected: Optional[list[tuple[float, float, Optional[float]]]] = None,
    ) -> list[tuple[complex, np.ndarray, float, float]]:
        """Select straight PML pairs, retaining the raw H vector for bends."""
        ny, nx = self.eps.shape
        k0 = 2.0 * np.pi / (wavelength_um * 1e-6)
        n_min = float(np.sqrt(self.eps.min()))
        n_max = float(np.sqrt(self.eps.max()))
        # A core-index shift misses weak tips entirely. Use the layer-free
        # fundamental at this wavelength as the default search anchor.
        if n_guess is None:
            reference = self._solve_at(wavelength_um, 1, None)
            if not reference:
                return []
            n_guess = reference[0].n_eff
        sigma = complex((n_guess * k0) ** 2)
        A = self._build_operator(k0, None, num_pml, pml_strength)
        k_req = max(1, min(2 * nx * ny - 2, max(32, 2 * num_modes)))
        try:
            vals, vecs = _spla.eigs(
                A, k=k_req, sigma=sigma, which="LM",
                v0=_deterministic_arpack_start(A.shape[0], complex_dtype=True),
            )
        except _spla.ArpackNoConvergence as exc:
            vals, vecs = exc.eigenvalues, exc.eigenvectors
            if vals.size == 0:
                return []
        except Exception:
            return []
        indices = np.sqrt(vals.astype(complex)) / k0
        indices = np.where(indices.real < 0, -indices, indices)
        shares = self._interior_energy(vecs, nx, ny, num_pml)
        eligible = np.where(
            (vals.real > (k0 * n_min) ** 2) &
            (vals.real <= (k0 * n_max) ** 2) &
            (shares >= _STRAIGHT_MIN_INTERIOR_SHARE)
        )[0]
        if not len(eligible):
            return []
        movement = np.zeros(len(vals))
        if np.any(shares[eligible] < _STRAIGHT_SKIP_HALF_SHARE):
            half_A = self._build_operator(k0, None, num_pml, 0.5 * pml_strength)
            try:
                half_vals, half_vecs = _spla.eigs(
                    half_A, k=k_req, sigma=sigma, which="LM",
                    v0=_deterministic_arpack_start(
                        half_A.shape[0], complex_dtype=True),
                )
            except _spla.ArpackNoConvergence as exc:
                half_vals, half_vecs = exc.eigenvalues, exc.eigenvectors
            except Exception:
                half_vals, half_vecs = np.empty(0), np.empty((A.shape[0], 0))
            half_indices = np.sqrt(half_vals.astype(complex)) / k0
            half_indices = np.where(half_indices.real < 0, -half_indices,
                                    half_indices)
            checked = []
            for col in eligible:
                overlaps = [self._field_overlap(vecs[:, col], half_vecs[:, j])
                            for j in range(half_vecs.shape[1])]
                if not overlaps:
                    if shares[col] >= _STRAIGHT_SKIP_HALF_SHARE:
                        checked.append(col)
                    elif rejected is not None:
                        rejected.append((float(indices[col].real), float(shares[col]), None))
                    continue
                partner = int(np.argmax(overlaps))
                movement[col] = abs(indices[col].real - half_indices[partner].real)
                if (overlaps[partner] >= _HALF_PARTNER_MIN_OVERLAP and
                        movement[col] <= _STRAIGHT_MAX_REAL_INDEX_MOVE):
                    checked.append(col)
                elif rejected is not None:
                    rejected.append((float(indices[col].real), float(shares[col]),
                                     float(movement[col])))
            eligible = np.asarray(checked, dtype=int)
        # Near-identical H fields can yield two different eigenvalues in a
        # small window. Retain the field least sensitive to layer strength.
        selected: list[int] = []
        for col in sorted(eligible, key=lambda j: (movement[j], -indices[j].real)):
            if all(self._field_overlap(vecs[:, col], vecs[:, old]) <=
                   _DUPLICATE_MIN_OVERLAP for old in selected):
                selected.append(int(col))
        selected.sort(key=lambda j: -indices[j].real)
        return [(complex(indices[j] * k0), vecs[:, j].copy(),
                 float(shares[j]), float(movement[j])) for j in selected]

    # -- the Fallahkhair-Li-Murphy transverse-H operator -------------------

    def _build_operator(
        self,
        k0: float,
        bend_radius_um: Optional[float] = None,
        pml_cells: int = 0,
        pml_strength: float = 5.0,
        pml_cells_y: int = 0,
        pml_strength_y: Optional[float] = None,
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
        half-cell x-spacings. ``pml_cells_y`` adds the y contour used by an EME
        radiation basis; the legacy bend path remains x-only. The stretches flow
        through the algebraic coefficient expressions, so ``β²`` becomes complex.
        With both PML counts zero every quantity is real and the operator is
        **bit-for-bit** the straight, lossless operator.

        Direct implementation of the Fallahkhair–Li–Murphy anisotropic stencil
        (A.B. Fallahkhair, K.S. Li, T.E. Murphy, JLT 26(11):1423, 2008), Appendix
        eqs (21)–(37): ``Pxx``/``Pxy`` from :func:`_axx_coeffs`/:func:`_axy_coeffs`,
        ``Pyy``/``Pyx`` from the paper's x↔y transformation (36) via
        :func:`_yblock`. Scalar ε (``εxx=εyy=εzz``) recovers the isotropic
        full-vector operator (the paper's refs [16],[17]); the diagonalized KFJ
        tensor feeds the interface-normal harmonic mean through εyy/εzz.

        Index convention: built in **(ix, iy)** order, ``p = ix*ny + iy`` (x outer,
        y inner), on each ε component transposed to ``[ix, iy]``. The four ε
        quadrants around a node (paper Fig. 1: 1=NW, 2=SW, 3=SE, 4=NE) are read
        from a one-ring edge-replication pad (a cladding ghost ring, exact to the
        mode's accuracy since the window is cladding-padded). The diagonal
        ``Pxx``/``Pyy`` blocks are 5-point; the off-diagonal ``Pxy``/``Pyx`` carry
        the vectorial coupling, including the 4 anisotropic corner couplings (eqs
        32,33) that are nonzero wherever εyy≠εzz (an interface cell). The
        eigenvalue is ``b² = (n_eff k0)²``.

        Boundary conditions: symmetric/antisymmetric wall folding of the N/S/E/W
        coefficients, the default (PEC-like) electric walls use ``Hx`` tangential
        → symmetric on the y-walls / antisymmetric on the x-walls and vice-versa
        for ``Hy``; ``x_symmetry == "pmc"`` swaps the x-wall parity so the lowest
        mode is x-uniform (``kx = 0``), the 1-D slab limit. The corner couplings
        vanish in the cladding ring at the walls, so they need no folding.
        These edge folds mirror about the wall node (``H[-1] = ±H[+1]``). An
        ``x_min_symmetry`` plane instead mirrors about the centre of ε column 0,
        half a cell before node 0 (``H[-1] = ±H[0]``), and folds the corner
        couplings as well.
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
        leaky = pml_cells > 0 or pml_cells_y > 0
        # Uniform half-cell spacings (meters) as length-N arrays, so the published
        # coefficient expressions evaluate verbatim per node.
        dx = self.dl_x_um * 1e-6
        dy = self.dl_y_um * 1e-6
        if leaky:
            sx = self._pml_stretch(nx, pml_cells, pml_strength, k0)  # (nx,) cplx
            sy = self._pml_stretch(
                ny,
                pml_cells_y,
                pml_strength if pml_strength_y is None else pml_strength_y,
                k0,
            )
            # Half-cell distance to the EAST/WEST face carries the average stretch
            # of the two cells the face separates. North/south use the analogous
            # y-contour distance.
            sx_e = 0.5 * (sx + np.r_[sx[1:], sx[-1]])               # face to x+
            sx_w = 0.5 * (sx + np.r_[sx[0], sx[:-1]])               # face to x-
            sy_n = 0.5 * (sy + np.r_[sy[1:], sy[-1]])               # face to y+
            sy_s = 0.5 * (sy + np.r_[sy[0], sy[:-1]])               # face to y-
            n = (dy * np.tile(sy_n, nx)).astype(complex)    # north (y+)
            s = (dy * np.tile(sy_s, nx)).astype(complex)    # south (y-)
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
        folds = [
            (ii[:, -1], "N", "S", sgn_y),    # NORTH wall: fold N onto S
            (ii[:, 0], "S", "N", sgn_y),     # SOUTH wall: fold S onto N
            (ii[-1, :], "E", "W", sgn_x),    # EAST  wall: fold E onto W
        ]
        if self.x_min_symmetry is None:
            folds.append((ii[0, :], "W", "E", sgn_x))   # WEST wall: W onto E
        for (wall, dirA, dirB, sgn) in folds:
            Pxx[dirB][wall] += sgn * Pxx[dirA][wall]
            Pyx[dirB][wall] += sgn * Pyx[dirA][wall]
            Pyy[dirB][wall] -= sgn * Pyy[dirA][wall]
            Pxy[dirB][wall] -= sgn * Pxy[dirA][wall]
        if self.x_min_symmetry is not None:
            # mirror plane through the centre of ε column 0. Node ix sits at
            # the corner x = (ix + 1/2)·dl, so the plane lies half a cell before
            # node 0 and the ghost column -1 is the image of column 0 itself:
            # H[-1] = ±H[0], not the ±H[+1] of the edge-node folds above. The
            # W / NW / SW couplings therefore land on P / N / S. Unlike a
            # cladding wall, the plane crosses the core, where the Pxy/Pyx corner
            # couplings are nonzero, so they fold too. This runs after the N/S
            # folds, so a corner node drops the same out-of-range couplings the
            # full grid drops. "pec" = Hx odd, Hy even; "pmc" the reverse.
            sgn_m = +1.0 if self.x_min_symmetry == "pmc" else -1.0
            wall = ii[0, :]
            for blk, sgn in ((Pxx, sgn_m), (Pyx, sgn_m),
                             (Pyy, -sgn_m), (Pxy, -sgn_m)):
                blk["P"][wall] += sgn * blk["W"][wall]
                blk["N"][wall] += sgn * blk["NW"][wall]
                blk["S"][wall] += sgn * blk["SW"][wall]

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
        """Compute up to ``num_modes`` guided modes (full-vector).

        Parameters
        ----------
        num_modes:
            Number of modes to return, ordered by descending ``Re(n_eff)`` (the
            first is the fundamental). Must be ``>= 1``.
        n_guess:
            Shift-invert target index. The solve returns the modes whose ``n_eff``
            is nearest ``n_guess``. Layer-free straight solves default to
            ``sqrt(max(eps))``; layered solves anchor on the layer-free
            fundamental so weak guides remain in the search.
        group_index:
            If ``True``, also compute each returned mode's group index ``n_g`` by
            central difference over two extra solves at ``lambda(1 +/- 1e-3)``
            (the same mode is tracked by maximum transverse-field overlap). Triples
            the cost. Default ``False`` (``mode.n_group is None``).
        bend_radius_um:
            If set (nonzero), solve the **bent**-waveguide mode of this radius
            (microns) via the physical cylindrical/centrifugal generalized
            eigenproblem ``A h = β₀² B h``, ``B = diag((R/r)²)`` (see
            :meth:`_centrifugal_weight`); the reported ``n_eff`` follows the
            engine convention (field ~ ``exp(+i n k0 R φ)``, so ``R→∞`` recovers
            the straight ``n_eff``). Index, field and attenuation come from
            one eigenpair of the solve with absorbing layers. The returned
            arrays are conjugated into the EME phasor. If ``num_pml`` is
            left at 0 a default PML at strength 30 is added with a warning.
            A radius with magnitude below half the full x window width plus
            one x mesh cell is refused before the eigensolve.
            ``None`` is the
            straight, lossless solve (``k_eff == 0``).
        num_pml:
            Number of complex-coordinate-stretched **PML** cells on each in-plane
            (x) window edge (:meth:`_pml_stretch`) so leaky/radiated fields are
            absorbed and the eigenvalue becomes complex. ``0`` (default) on the
            straight path keeps the operator real and lossless; a bend solve with
            ``num_pml == 0`` adds a default thickness and warns.
        pml_strength:
            PML conductivity scale ``σ_max/k0`` (dimensionless); the polynomial
            profile peaks at the wall. Default ``30.0``, strong enough to absorb
            the bend radiation in a handful of cells.

        Returns
        -------
        tuple[VectorMode, ...]
            Up to ``num_modes`` :class:`VectorMode` objects, highest ``Re(n_eff)``
            first. Fewer are returned if the window supports fewer guided modes.
            Any complex mode follows the split convention documented on
            :attr:`VectorMode.n_eff_complex`: positive-loss metadata remains in
            ``n_eff + i*k_eff``, while the field arrays use its EME conjugate.

        Notes
        -----
        Finite windows and absorbing layers affect leaky bend loss. The solver
        rejects layer-localized eigenpairs by their energy outside the layers.
        Straight guides check the real-index response to half layer strength;
        bend pairs are assigned one to one to the selected straight modes by
        interior overlap times the square root of interior energy share.
        ``loss_unresolved`` flags a bend whose caustic margin is under 0.75
        Airy lengths or whose loss changes by over 10% at half layer strength.
        Negative attenuation is clamped to zero; it can reach a few hundred
        dB/cm on a narrow window, so check a zero bend loss on a wider window
        like any other. Widen the window and vary layer thickness when
        quantitative loss matters.

        With ``method="yee"`` the solve is the straight transverse-E
        eigenproblem on the Yee lattice: ``n_guess`` sets its shift
        (default the largest sampled index), ``group_index`` uses the same
        central difference with the permittivity samples unchanged, and
        ``bend_radius_um`` or ``num_pml > 0`` raise ``NotImplementedError``
        (planned). Modes above the smallest sampled index are returned, with
        ``k_eff == 0`` and their fields co-located on the cell centres."""
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        if self.method == "yee":
            if bend_radius_um is not None:
                raise NotImplementedError(
                    "method='yee' solves straight cross-sections only; bent "
                    "solves on the Yee lattice are planned. Use the default "
                    "method='flm' for a bend.")
            if num_pml != 0:
                raise NotImplementedError(
                    "method='yee' has no absorbing x layers yet (planned); "
                    "solve with num_pml=0, or use the default method='flm' "
                    "for a layered straight solve.")
            modes = self._solve_yee_at(self.wavelength_um, num_modes, n_guess)
            if not group_index:
                return modes
            nm = max(num_modes + 2, len(modes) + 2)
            return self._with_group_index(
                modes, lambda wl: self._solve_yee_at(wl, nm, n_guess))
        if bend_radius_um is not None and bend_radius_um == 0.0:
            raise ValueError(
                "bend_radius_um must be nonzero (use None for a straight guide)")
        if num_pml < 0:
            raise ValueError("num_pml must be >= 0")
        if self.x_min_symmetry is not None and (
                bend_radius_um is not None or num_pml > 0):
            raise ValueError(
                "x_min_symmetry supports only the straight PML-free solve: a "
                "bend is not mirror symmetric about the width centre, and the "
                "x PML would cover the mirror plane")
        ny, nx = self.eps.shape
        if (bend_radius_um is not None and
                abs(bend_radius_um) < nx * self.dl_x_um / 2 + self.dl_x_um):
            raise ValueError(
                f"bend radius |R|={abs(bend_radius_um):g} um must be at least "
                f"W/2 + dl={nx * self.dl_x_um / 2 + self.dl_x_um:g} um "
                f"(W/2={nx * self.dl_x_um / 2:g} um, "
                f"dl={self.dl_x_um:g} um)")
        if nx * ny > self.MAX_UNKNOWNS:
            raise ValueError(
                f"cross-section is {nx}x{ny} = {nx * ny} unknowns (2N = "
                f"{2 * nx * ny} eigenproblem), exceeding MAX_UNKNOWNS="
                f"{self.MAX_UNKNOWNS}. Coarsen the grid or shrink the window.")

        # A bend radiates; it needs a tangential PML to absorb the leakage (else
        # the radiated field reflects off the PEC wall). Default to ~1/4 of the
        # x-window so the PML sits in the cladding outside the core.
        auto_added_layers = bend_radius_um is not None and num_pml == 0
        if auto_added_layers:
            num_pml = max(6, nx // 4)
            pml_strength = 30.0

        modes = self._solve_at(self.wavelength_um, num_modes, n_guess,
                               bend_radius_um, num_pml, pml_strength,
                               warn_on_shortage=True,
                               auto_added_layers=auto_added_layers)
        if not group_index:
            return modes

        # Preserve the layer-free straight side spectrum exactly; layered
        # solves track only the modes the caller requested.
        nm = (max(num_modes + 2, len(modes) + 2)
              if bend_radius_um is None and num_pml == 0 else num_modes)
        # The side solves feed only n_eff to the match below, so they skip
        # the bend's half-strength loss check, whose flag nothing reads.
        return self._with_group_index(
            modes, lambda wl: self._solve_at(wl, nm, n_guess, bend_radius_um,
                                             num_pml, pml_strength,
                                             check_bend_loss=False))

    def _with_group_index(
        self,
        modes: Tuple[VectorMode, ...],
        side_solve: Callable[[float], Tuple[VectorMode, ...]],
    ) -> Tuple[VectorMode, ...]:
        """Attach ``n_group`` by central difference over ``side_solve`` at
        ``lambda (1 -/+ delta)``, ``delta = 1e-3``.

        Re-solve at lambda(1 +/- delta) and match each base mode to its
        perturbed partner by max transverse-field overlap (the spectrum can
        reorder slightly near-degenerate). A mode with no partner keeps
        ``n_group=None``."""
        delta = 1e-3
        lam = self.wavelength_um
        modes_lo = side_solve(lam * (1.0 - delta))
        modes_hi = side_solve(lam * (1.0 + delta))

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
            # replace, not a rebuild: a rebuild dropped a half-window mode's
            # center_offset_um and x_min_symmetry.
            out.append(replace(m, n_group=float(n_g)))
        return tuple(out)

    def solve_eme_basis(
        self,
        *,
        num_guided: int = 1,
        num_radiation: int = 0,
        num_evanescent: int = 0,
        n_guess: Optional[float] = None,
        pml_cells_xy: Optional[Tuple[int, int]] = None,
        pml_imaginary_thickness_um: Tuple[float, float] = (0.10, 0.10),
        residual_tolerance: float = 1e-7,
    ) -> Tuple[VectorMode, ...]:
        """Return a common-operator EME basis containing guided, radiation and
        evanescent modes.

        .. warning::
           Radiation/evanescent and PML requests are experimental. Analytic
           spectrum/Maxwell checks pass, but the current high-contrast device
           basis does not stabilize under nested radiation-mode-group refinement.
           Require independent mesh, window, PML, basis, and FDTD convergence
           before interpreting a result as quantitative radiation loss.

        Unlike :meth:`solve`, which deliberately filters to bound guided modes,
        this method samples the complex spectrum at guided, radiation and
        negative-``beta^2`` shifts.  When continuum modes are requested it uses
        a four-sided complex-coordinate PML, so the open radiation continuum is
        represented by a finite set of passive box/PML modes.  All returned modes
        come from that *same* operator and therefore retain diagonal propagation.
        Before return, each PML eigenfield, beta, and complex contour is conjugated
        together into the EME convention. ``mode.n_eff_complex`` deliberately
        retains the positive-loss engine eigenvalue as metadata. Thus an
        evanescent channel has approximately ``n_eff=0, k_eff>0``,
        ``beta_eme=-i*k0*k_eff``, and the EME propagation factor
        ``exp(-i*beta_eme*L)=exp(-k0*k_eff*L)`` decays.

        ``pml_cells_xy=(0, 0)`` is an explicit hard-wall box-spectrum mode useful
        for analytic verification.  Otherwise the default PML occupies roughly
        one eighth of each transverse window.  Its strength is parameterised by
        integrated imaginary contour thickness, not a grid-dependent peak. The
        current classifier requires one homogeneous scalar cladding surrounding
        all four PML strips; asymmetric air/substrate exteriors need a
        multi-exterior light-line treatment and are rejected.

        Not available with ``method="yee"`` yet (planned): it raises
        ``NotImplementedError``.
        """
        if self.method == "yee":
            raise NotImplementedError(
                "solve_eme_basis is not available with method='yee' yet "
                "(planned); build the EME basis with the default "
                "method='flm'.")
        counts = (int(num_guided), int(num_radiation), int(num_evanescent))
        if any(n < 0 for n in counts) or sum(counts) < 1:
            raise ValueError(
                "num_guided/num_radiation/num_evanescent must be >= 0 and "
                "at least one mode must be requested"
            )
        if residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be > 0")
        if len(pml_imaginary_thickness_um) != 2 or any(
            x < 0.0 or not np.isfinite(x) for x in pml_imaginary_thickness_um
        ):
            raise ValueError(
                "pml_imaginary_thickness_um must be two finite values >= 0"
            )
        if num_radiation == 0 and num_evanescent == 0:
            return self.solve(num_modes=max(1, num_guided), n_guess=n_guess)[
                :num_guided
            ]

        ny, nx = self.eps.shape
        N = nx * ny
        if N > self.MAX_UNKNOWNS:
            raise ValueError(
                f"cross-section has {N} cells, exceeding MAX_UNKNOWNS="
                f"{self.MAX_UNKNOWNS}"
            )
        if pml_cells_xy is None:
            pml_x = max(3, nx // 8)
            pml_y = max(3, ny // 8)
        else:
            if len(pml_cells_xy) != 2:
                raise ValueError("pml_cells_xy must be an (x, y) pair")
            pml_x, pml_y = (int(pml_cells_xy[0]), int(pml_cells_xy[1]))
        if pml_x < 0 or pml_y < 0:
            raise ValueError("PML cell counts must be >= 0")
        if self.x_min_symmetry is not None and pml_x > 0:
            raise ValueError(
                "x_min_symmetry: the x PML would cover the mirror plane; use "
                "the hard-wall box pml_cells_xy=(0, 0)")
        if 2 * pml_x >= nx - 2 or 2 * pml_y >= ny - 2:
            raise ValueError(
                f"PML ({pml_x}, {pml_y}) leaves too little physical window "
                f"inside the {nx}x{ny} grid"
            )
        if (pml_x == 0) != (pml_y == 0):
            raise ValueError(
                "an EME radiation basis needs either four-sided PML or the "
                "explicit hard-wall box pml_cells_xy=(0, 0)"
            )
        if (pml_x > 0 or pml_y > 0) and any(
            value <= 0.0 for value in pml_imaginary_thickness_um
        ):
            raise ValueError(
                "a nonzero PML requires positive imaginary thickness on both "
                "axes; use pml_cells_xy=(0, 0) for a hard-wall box"
            )

        # A PML must lie entirely in homogeneous cladding.  Letting a core or a
        # material interface enter the complex contour invalidates both the
        # coordinate transform and its overlap quadrature.
        if pml_x > 0 or pml_y > 0:
            mask = np.zeros((ny, nx), dtype=bool)
            if pml_x > 0:
                mask[:, :pml_x] = True
                mask[:, -pml_x:] = True
            if pml_y > 0:
                mask[:pml_y, :] = True
                mask[-pml_y:, :] = True
            edge = self.eps[mask]
            clad = float(self.eps.min())
            if edge.size == 0 or not np.allclose(edge, clad, rtol=1e-8, atol=1e-10):
                raise ValueError(
                    "PML overlaps non-cladding material; enlarge the transverse "
                    "window or reduce pml_cells_xy"
                )

        wavelength_um = self.wavelength_um
        k0 = 2.0 * np.pi / (wavelength_um * 1e-6)
        n_max = float(np.sqrt(self.eps.max()))
        n_min = float(np.sqrt(self.eps.min()))
        if n_guess is None:
            n_guess = n_max
        strength_x = self._pml_strength_for_imaginary_thickness(
            pml_x, self.dl_x_um, float(pml_imaginary_thickness_um[0])
        )
        strength_y = self._pml_strength_for_imaginary_thickness(
            pml_y, self.dl_y_um, float(pml_imaginary_thickness_um[1])
        )
        closed_seeds: Optional[Tuple[VectorMode, ...]] = None
        if pml_x > 0 or pml_y > 0:
            # A blind search of a PML spectrum is dominated by contour-localised
            # branches.  First solve the corresponding closed box, then target
            # the open/PML operator at those physical eigenvalues and match on
            # the non-PML interior.
            closed_seeds = self.solve_eme_basis(
                num_guided=num_guided,
                num_radiation=num_radiation,
                num_evanescent=num_evanescent,
                n_guess=n_guess,
                pml_cells_xy=(0, 0),
                pml_imaginary_thickness_um=(0.0, 0.0),
                residual_tolerance=residual_tolerance,
            )
        A = self._build_operator(
            k0,
            None,
            pml_x,
            strength_x,
            pml_y,
            strength_y,
        )

        # Multiple shifts are essential: one high-index solve cannot see enough
        # of the radiation continuum, while a negative beta^2 shift is required
        # to retrieve below-cutoff evanescent channels.
        anchors: list[complex] = []
        if closed_seeds is not None:
            anchors.extend(
                complex((k0 * m.n_eff_complex) ** 2) for m in closed_seeds
            )
        else:
            if num_guided:
                guided_seeds = self.solve(
                    num_modes=num_guided, n_guess=n_guess
                )
                # ``guided_seeds`` came from this exact hard-wall operator.  Using
                # one of its eigenvalues verbatim as a shift makes ``A-sigma*I``
                # singular; depending on the SuperLU pivot path, ARPACK then emits
                # an exact-shift warning or fails before returning any vectors.
                # Move only the *search target* by a deterministic, mesh-scale-
                # negligible amount.  The returned Ritz values/eigenvectors remain
                # those of ``A``; this is not an eigenvalue perturbation.
                anchors.extend(
                    _detuned_eigenvalue_shift((m.n_eff * k0) ** 2, k0)
                    for m in guided_seeds
                )
            if num_radiation:
                for frac in np.linspace(
                    0.18, 0.94, min(5, max(2, num_radiation))
                ):
                    anchors.append(complex((float(frac) * n_min * k0) ** 2))
            if num_evanescent:
                for alpha in np.geomspace(
                    0.15 * n_min,
                    1.5 * max(n_min, n_max),
                    min(5, max(2, num_evanescent)),
                ):
                    anchors.append(complex(-(float(alpha) * k0) ** 2))

        per_shift = max(8, 2 * max(counts) + 4)
        k_req = max(1, min(2 * N - 2, per_shift))
        candidates: list[tuple[complex, np.ndarray, float]] = []
        for shift_index, sigma in enumerate(anchors):
            try:
                vals, vecs = _spla.eigs(
                    A,
                    k=k_req,
                    sigma=sigma,
                    which="LM",
                    v0=_deterministic_arpack_start(
                        A.shape[0],
                        shift_index,
                        complex_dtype=np.issubdtype(A.dtype, np.complexfloating),
                    ),
                )
            except _spla.ArpackNoConvergence as exc:
                vals, vecs = exc.eigenvalues, exc.eigenvectors
                if vals.size == 0:
                    continue
            for col, val in enumerate(vals):
                vec = np.asarray(vecs[:, col], dtype=complex)
                av = A @ vec
                denom = np.linalg.norm(av) + abs(val) * np.linalg.norm(vec)
                residual = float(np.linalg.norm(av - val * vec) / max(denom, 1e-300))
                if not np.isfinite(residual) or residual > residual_tolerance:
                    continue
                _merge_eigen_candidate(candidates, complex(val), vec, residual,
                                       k0)

        guided: list[tuple[float, float, VectorMode]] = []
        radiation: list[tuple[float, float, VectorMode]] = []
        evanescent: list[tuple[float, float, VectorMode]] = []
        physical = np.ones((ny, nx), dtype=bool)
        if pml_x > 0:
            physical[:, :pml_x] = False
            physical[:, -pml_x:] = False
        if pml_y > 0:
            physical[:pml_y, :] = False
            physical[-pml_y:, :] = False

        for val, vec, residual in candidates:
            beta = complex(np.sqrt(val))
            # Causal forward branch: positive phase for propagating modes and
            # positive decay for evanescent/complex modes.  Never convert a gain
            # branch into a loss branch by clamping Im(beta).
            if val.real < 0.0:
                if beta.imag < 0.0:
                    beta = -beta
            elif beta.real < 0.0:
                beta = -beta
            if beta.imag < 0.0:
                # A well-bound guided seed can pick up a tiny negative PML
                # residue from the finite contour/discretisation.  Project only
                # that near-real branch back to the passive axis; a material
                # negative-imaginary branch is rejected, never relabelled loss.
                looks_bound = beta.real / k0 > n_min * (1.0 + 1e-6)
                near_axis = abs(beta.imag) <= 1e-3 * max(abs(beta.real), k0)
                if looks_bound and near_axis:
                    beta = complex(beta.real, 0.0)
                else:
                    continue
            if abs(beta) < 1e-12 * k0:
                continue

            hx = vec[:N].reshape(nx, ny).T.astype(np.complex128)
            hy = vec[N:].reshape(nx, ny).T.astype(np.complex128)
            h_energy = np.abs(hx) ** 2 + np.abs(hy) ** 2
            physical_fraction = float(
                h_energy[physical].sum() / max(float(h_energy.sum()), 1e-300)
            )
            if (pml_x > 0 or pml_y > 0) and physical_fraction < 0.03:
                continue  # PML-localised numerical branch

            neff_r = float(beta.real / k0)
            if val.real < 0.0 or abs(neff_r) < 1e-7:
                kind: ModeType = "evanescent"
            elif neff_r <= n_min * (1.0 + 1e-6):
                kind = "radiation"
            else:
                kind = "guided"
            # The sparse operator/PML uses the engine convention
            # exp(-i*omega*t + i*beta*z). EME uses its complex conjugate
            # exp(+i*omega*t - i*beta* z). Convert the eigenfield, propagation
            # constant, and complex contour together before applying the EME
            # curl reconstruction; retain positive k_eff as storage metadata.
            mode = self._reconstruct(
                hx.conj(),
                hy.conj(),
                beta.conjugate(),
                k0,
                wavelength_um,
                None,
                mode_type=kind,
                pml_cells_xy=(pml_x, pml_y),
                pml_strength_xy=(-strength_x, -strength_y),
                eigen_residual=residual,
            )
            mode = replace(
                mode,
                n_eff=float(beta.real / k0),
                k_eff=float(beta.imag / k0),
            )
            key = (neff_r, float(beta.imag / k0), mode)
            if kind == "guided":
                guided.append(key)
            elif kind == "radiation":
                radiation.append(key)
            else:
                evanescent.append(key)

        guided.sort(key=lambda x: (-x[0], x[1]))
        radiation.sort(key=lambda x: (-x[0], x[1]))
        evanescent.sort(key=lambda x: (x[1], -x[0]))

        if closed_seeds is None:
            selected_guided = [m for _, _, m in guided[:num_guided]]
            selected_radiation = [m for _, _, m in radiation[:num_radiation]]
            selected_evanescent = [
                m for _, _, m in evanescent[:num_evanescent]
            ]
        else:
            # PML beta is complex and can cross the closed-box light-line
            # classifier.  Classification therefore belongs to the real
            # closed-box seed, not to the PML eigenvalue.  Match each degenerate
            # seed *subspace* against one global candidate pool and inherit its
            # role.  A vector-by-vector greedy match is not rotation invariant:
            # ARPACK may return arbitrary bases for the same degenerate
            # eigenspace, allowing the first match to strand its rotated partner
            # and substitute an unrelated branch for the second.
            available = [
                m for _, _, m in (*guided, *radiation, *evanescent)
            ]
            selected_by_type: dict[ModeType, list[VectorMode]] = {
                "guided": [],
                "radiation": [],
                "evanescent": [],
            }
            interior = physical

            seed_clusters: list[list[VectorMode]] = []
            for seed in closed_seeds:
                for cluster in seed_clusters:
                    reference = cluster[0]
                    same_type = seed.mode_type == reference.mode_type
                    beta_close = abs(
                        seed.n_eff_complex - reference.n_eff_complex
                    ) <= 1e-6 * max(
                        abs(seed.n_eff_complex),
                        abs(reference.n_eff_complex),
                        1.0,
                    )
                    if same_type and beta_close:
                        cluster.append(seed)
                        break
                else:
                    seed_clusters.append([seed])

            def interior_e(mode: VectorMode) -> np.ndarray:
                return np.concatenate(
                    [mode.ex[interior], mode.ey[interior]]
                )

            for seed_cluster in seed_clusters:
                kind = seed_cluster[0].mode_type
                multiplicity = len(seed_cluster)
                seed_matrix = np.column_stack(
                    [interior_e(seed) for seed in seed_cluster]
                )
                u_seed, singular_values, _ = np.linalg.svd(
                    seed_matrix, full_matrices=False
                )
                if (
                    singular_values.size < multiplicity
                    or singular_values[0] <= 0.0
                    or np.count_nonzero(
                        singular_values > 1e-8 * singular_values[0]
                    ) < multiplicity
                ):
                    raise RuntimeError(
                        "closed-box EME seed cluster is rank deficient; "
                        "increase the window or change the spectral targets"
                    )
                seed_subspace = u_seed[:, :multiplicity]
                beta_center = sum(
                    seed.n_eff_complex for seed in seed_cluster
                ) / multiplicity

                scored: list[tuple[float, float, int]] = []
                for i, candidate in enumerate(available):
                    candidate_e = interior_e(candidate)
                    candidate_norm = float(np.linalg.norm(candidate_e))
                    if candidate_norm <= 0.0:
                        continue
                    # Squared projection onto the entire seed subspace is
                    # invariant to rotations within a degenerate eigenbasis.
                    projection = float(
                        np.linalg.norm(
                            seed_subspace.conj().T @ candidate_e
                        ) ** 2
                        / candidate_norm**2
                    )
                    beta_gap = abs(
                        candidate.n_eff_complex - beta_center
                    ) / max(abs(beta_center), 1.0)
                    scored.append(
                        (projection - 0.05 * beta_gap, projection, i)
                    )
                scored.sort(reverse=True)
                chosen = scored[:multiplicity]
                if (
                    len(chosen) < multiplicity
                    or min(item[1] for item in chosen) < 0.10
                ):
                    raise RuntimeError(
                        "PML/closed-box modal-subspace correspondence is "
                        "ambiguous; enlarge the physical window or weaken the "
                        "complex contour"
                    )

                # Selection happened for the cluster as a whole.  Reorder only
                # within that fixed set for deterministic seed association; this
                # cannot eject a correct rotated partner.
                chosen_indices = [item[2] for item in chosen]
                chosen_modes = [available[i] for i in chosen_indices]
                ordered: list[VectorMode] = []
                remaining = list(chosen_modes)
                for seed in seed_cluster:
                    seed_e = interior_e(seed)
                    seed_norm = float(np.linalg.norm(seed_e))
                    best_i = max(
                        range(len(remaining)),
                        key=lambda i: (
                            abs(np.vdot(seed_e, interior_e(remaining[i])))
                            / max(
                                seed_norm
                                * float(np.linalg.norm(interior_e(remaining[i]))),
                                1e-300,
                            )
                            - 0.05
                            * abs(
                                remaining[i].n_eff_complex
                                - seed.n_eff_complex
                            )
                            / max(abs(seed.n_eff_complex), 1.0)
                        ),
                    )
                    ordered.append(remaining.pop(best_i))
                for i in sorted(chosen_indices, reverse=True):
                    available.pop(i)
                selected_by_type[kind].extend(
                    replace(candidate, mode_type=kind)
                    for candidate in ordered
                )
            selected_guided = selected_by_type["guided"]
            selected_radiation = selected_by_type["radiation"]
            selected_evanescent = selected_by_type["evanescent"]
        picked = selected_guided + selected_radiation + selected_evanescent
        got = (
            len(selected_guided),
            len(selected_radiation),
            len(selected_evanescent),
        )
        if got != counts:
            raise RuntimeError(
                "EME basis spectrum incomplete: requested "
                f"guided/radiation/evanescent={counts}, found {got}. Increase "
                "the window, relax residual_tolerance, or change the PML contour."
            )
        return tuple(picked)

    # -- internals ----------------------------------------------------------

    def _solve_yee_at(
        self,
        wavelength_um: float,
        num_modes: int,
        n_guess: Optional[float],
    ) -> Tuple[VectorMode, ...]:
        """Straight ``method="yee"`` solve at one wavelength: the guided
        modes (``n_eff`` above the smallest sampled index and real to 1e-6),
        highest ``n_eff`` first, at most ``num_modes``."""
        from .yee_mode import _solve_yee_eig

        lat = self._yee_lattice()
        nv, nh = lat.exx.shape
        n = nh * nv
        if n > self.MAX_UNKNOWNS:
            raise ValueError(
                f"Yee lattice is {nh}x{nv} = {n} nodes (2N = {2 * n} "
                f"eigenproblem), exceeding MAX_UNKNOWNS={self.MAX_UNKNOWNS}. "
                "Coarsen the grid or shrink the window.")
        every = np.concatenate((lat.exx.ravel(), lat.eyy.ravel(),
                                lat.ezz.ravel()))
        n_min = float(np.sqrt(every.min()))
        n_max = float(np.sqrt(every.max()))

        def flat(a: np.ndarray) -> np.ndarray:      # [iv, ih] -> [ih*nv + iv]
            return np.ascontiguousarray(a.T).ravel()

        trial = max(1, min(2 * n - 2, max(num_modes + 6, 2 * num_modes)))
        raw = _solve_yee_eig(
            flat(lat.exx), flat(lat.eyy), flat(lat.ezz), nh, nv,
            wavelength_um, self.dl_x_um, trial, h_min_bc=lat.h_bc,
            v_min_bc=lat.v_bc, min_neff=n_min, n_guess=n_guess)
        out: list[VectorMode] = []
        for mode in raw:
            if mode.n_eff > n_max or abs(mode.k_eff) > 1e-6 * mode.n_eff:
                continue
            out.append(self._yee_public_mode(mode, lat))
            if len(out) == num_modes:
                break
        return tuple(out)

    def _yee_public_mode(self, mode: VectorMode, lat: _YeeLattice) -> VectorMode:
        """Co-locate the six components on the lattice nodes (the ε cell
        centres), drop the wall lines, so one entry per ε cell remains, then
        restore the joint transverse-E normalization and the phase fix.

        A component half a cell off its node along an axis takes the mean of
        its two neighbours there (second order for a smooth field, first
        order across a material face where the normal E component jumps): ``Ex``
        and ``Hy`` along x, ``Ey`` and ``Hx`` along y, ``Hz`` along both;
        ``Ez`` already sits on the node. The value before the first node is
        the lattice's own: beside a wall the leading line's, across a mirror
        plane the image (sign +1 for ``"pec"``, whose ``Ex``, ``Hy`` and ``Hz``
        are even, -1 for ``"pmc"``), across a periodic axis the last one. The
        returned mode is therefore node-collocated like the default method's
        (``yee_staggered=False``), and every consumer reads it as one. A half
        window carries the default method's placement, column ``j`` at
        ``x = j*dl_x_um`` from the plane; its plane is node 0
        (``x_min_plane_on_node=True``)."""
        raw = {name: np.asarray(getattr(mode, name))
               for name in ("ex", "ey", "ez", "hx", "hy", "hz")}
        image = 1.0 if self.x_min_symmetry == "pec" else -1.0

        def before(a: np.ndarray, axis: int, periodic: bool,
                   mirrored: bool) -> np.ndarray:
            first = np.take(a, [0], axis=axis)
            if periodic:
                first = np.take(a, [-1], axis=axis)
            elif mirrored:
                first = image * first
            # Beside a wall the first entry belongs to the dropped wall line.
            rest = np.take(a, np.arange(a.shape[axis] - 1), axis=axis)
            return np.concatenate((first, rest), axis=axis)

        def to_node_x(a: np.ndarray) -> np.ndarray:
            return 0.5 * (before(a, 1, lat.h_bc == "periodic",
                                 lat.lead_x == 0 and lat.h_bc in ("pec", "pmc"))
                          + a)

        def to_node_y(a: np.ndarray) -> np.ndarray:
            return 0.5 * (before(a, 0, lat.v_bc == "periodic", False) + a)

        nodes = {"ex": to_node_x(raw["ex"]), "ey": to_node_y(raw["ey"]),
                 "ez": raw["ez"], "hx": to_node_y(raw["hx"]),
                 "hy": to_node_x(raw["hy"]), "hz": to_node_y(to_node_x(raw["hz"]))}
        cut = (slice(lat.lead_y, None), slice(lat.lead_x, None))
        fields = {name: value[cut] for name, value in nodes.items()}
        px = float(np.sum(np.abs(fields["ex"]) ** 2))
        py = float(np.sum(np.abs(fields["ey"]) ** 2))
        norm = float(np.sqrt(px + py))
        ref = fields["ex"] if px >= py else fields["ey"]
        peak = ref.flat[int(np.argmax(np.abs(ref)))]
        phase = np.conj(peak) / abs(peak) if abs(peak) > 0 else 1.0 + 0.0j
        scale = (1.0 / norm if norm > 0 else 1.0) * phase
        nx = fields["ex"].shape[1]
        center = ((0.5 * (nx - 1) * self.dl_x_um, 0.0)
                  if self.x_min_symmetry is not None else None)
        return replace(
            mode, k_eff=0.0, dl_x_um=self.dl_x_um, dl_y_um=self.dl_y_um,
            center_offset_um=center, x_min_symmetry=self.x_min_symmetry,
            yee_staggered=False,
            x_min_plane_on_node=self.x_min_symmetry is not None,
            **{name: value * scale for name, value in fields.items()})

    def _solve_at(
        self,
        wavelength_um: float,
        num_modes: int,
        n_guess: Optional[float],
        bend_radius_um: Optional[float] = None,
        num_pml: int = 0,
        pml_strength: float = 5.0,
        warn_on_shortage: bool = False,
        auto_added_layers: bool = False,
        check_bend_loss: bool = True,
    ) -> Tuple[VectorMode, ...]:
        """Single-wavelength eigensolve + filter + reconstruct. Returns guided
        :class:`VectorMode`s sorted by descending ``Re(n_eff)`` (no group index).

        Straight, lossless path (``bend_radius_um is None and num_pml == 0``):
        ordinary real eigenproblem (``B`` = identity), near-real ``β²`` band
        filter, ``k_eff == 0``, **bit-for-bit** the original solver. Bent path:
        the **generalized** eigenproblem ``A h = β₀² B h`` with the centrifugal
        mass matrix ``B = diag((R/r)²)`` (:meth:`_mass_matrix`), the physical
        cylindrical treatment (see :meth:`_centrifugal_weight`). The reported
        real ``n_eff`` and attenuation of a bend come from the same PML
        eigenpair as its field. Returned fields use the conjugate EME phasor."""
        eps = self.eps
        ny, nx = eps.shape
        N = nx * ny
        k0 = 2.0 * np.pi / (wavelength_um * 1e-6)
        n_max = float(np.sqrt(eps.max()))
        n_min = float(np.sqrt(eps.min()))
        is_bend = bend_radius_um is not None
        if (is_bend and
                abs(bend_radius_um) < nx * self.dl_x_um / 2 + self.dl_x_um):
            raise ValueError(
                f"bend radius |R|={abs(bend_radius_um):g} um must be at least "
                f"W/2 + dl={nx * self.dl_x_um / 2 + self.dl_x_um:g} um "
                f"(W/2={nx * self.dl_x_um / 2:g} um, "
                f"dl={self.dl_x_um:g} um)")
        if n_guess is None and num_pml == 0:
            n_guess = n_max

        # Guided band on Re(β²). Curvature can push a bend above the straight
        # core index; retain a generous upper bound for the leaky search.
        margin = 1.0 if not is_bend else 1.5
        upper = (k0 * n_max * margin) ** 2
        lower = (k0 * n_min) ** 2

        if not is_bend:
            # ---- straight (+ optional standalone PML) : ordinary eigenproblem --
            if num_pml > 0:
                rejected: list[tuple[float, float, Optional[float]]] = []
                pairs = self._layered_straight_pairs(
                    wavelength_um, num_modes, n_guess, num_pml, pml_strength,
                    rejected)
                out: list[VectorMode] = []
                for beta, vec, share, _ in pairs[:num_modes]:
                    beta = complex(beta.real, max(0.0, beta.imag))
                    hx = vec[:N].reshape(nx, ny).T.astype(np.complex128)
                    hy = vec[N:].reshape(nx, ny).T.astype(np.complex128)
                    mode = self._reconstruct(
                        hx.conj(), hy.conj(), beta.conjugate(), k0,
                        wavelength_um, None, pml_cells_xy=(num_pml, 0),
                        pml_strength_xy=(-pml_strength, 0.0),
                    )
                    out.append(replace(
                        mode, n_eff=float(beta.real / k0),
                        k_eff=float(beta.imag / k0),
                        window_limited=share < _STRAIGHT_WINDOW_LIMIT_SHARE,
                    ))
                if warn_on_shortage:
                    notices = []
                    if rejected:
                        details = ", ".join(
                            f"n={index:.6f}, share={share:.3f}, "
                            f"half-strength move={move:.2e}"
                            if move is not None else
                            f"n={index:.6f}, share={share:.3f}, "
                            "half-strength move=unavailable"
                            for index, share, move in rejected)
                        notices.append(
                            "Layer-dependent pairs rejected (" + details + "). "
                            "Widen the window, keeping at least eight layer cells; "
                            "thicker layers in the same window do not help.")
                    if any(mode.window_limited for mode in out):
                        notices.append(
                            "Mode is window-limited (less than 95% of field "
                            "energy is outside the absorbing layers); widen "
                            "the window for a converged result.")
                    if len(out) < num_modes:
                        quantity = ("No guided mode was" if not out else
                                    f"Only {len(out)} guided mode" +
                                    (" was" if len(out) == 1 else "s were"))
                        notices.append(
                            f"{quantity} found inside the window; widen the "
                            "window or adjust the absorbing layers.")
                    if notices:
                        warnings.warn(" ".join(notices), RuntimeWarning,
                                      stacklevel=caller_stacklevel())
                return tuple(out)
            # Keep the original layer-free eigensolve and ordering unchanged.
            sigma = float((n_guess * k0) ** 2)
            A = self._build_operator(k0, None, num_pml, pml_strength)
            extra = 6
            requested = max(num_modes + extra, 2 * num_modes)
            k_req = max(1, min(2 * N - 2, requested))
            try:
                vals, vecs = _spla.eigs(
                    A,
                    k=k_req,
                    sigma=sigma,
                    which="LM",
                    v0=_deterministic_arpack_start(
                        A.shape[0],
                        complex_dtype=np.issubdtype(A.dtype, np.complexfloating),
                    ),
                )
            except _spla.ArpackNoConvergence as exc:
                vals, vecs = exc.eigenvalues, exc.eigenvectors
                if vals.size == 0:
                    return tuple()
            beta_c = np.sqrt(vals.astype(complex))
            beta_c = np.where(beta_c.real < 0.0, -beta_c, beta_c)
            beta2_re = vals.real
            imag_ok = np.abs(vals.imag) <= 1e-4 * (np.abs(vals.real) + 1.0)
            keep = imag_ok & (beta2_re > lower) & (beta2_re <= upper)
            idx = np.where(keep)[0]
            order = idx[np.argsort(beta2_re[idx])[::-1]]
            out: list[VectorMode] = []
            for col in order[:num_modes]:
                if beta2_re[col] <= 0.0:
                    continue
                beta = float(beta_c[col].real)
                vec = vecs[:, col]
                hx = vec[:N].reshape(nx, ny).T.astype(np.complex128)
                hy = vec[N:].reshape(nx, ny).T.astype(np.complex128)
                mode = self._reconstruct(
                    hx, hy, beta, k0, wavelength_um, None
                )
                out.append(mode)
            return tuple(out)

        # Select the straight guide at this same window and contour. Every
        # selected pair is an anchor and can receive at most one bend partner.
        rejected = []
        straight = self._layered_straight_pairs(
            wavelength_um, max(2, num_modes), n_guess if num_pml else None,
            num_pml, pml_strength, rejected)
        if not straight:
            if warn_on_shortage:
                details = ", ".join(
                    f"n={index:.6f}, share={share:.3f}, "
                    f"half-strength move={move:.2e}"
                    if move is not None else
                    f"n={index:.6f}, share={share:.3f}, "
                    "half-strength move=unavailable"
                    for index, share, move in rejected)
                prefix = ("A bend radiates, so absorbing layers were added "
                          "automatically at strength 30. "
                          if auto_added_layers else "")
                warnings.warn(
                    prefix + "No straight guided mode was found inside the window; "
                    + (f"layer-dependent pairs rejected ({details}); " if details else "")
                    + "widen the window, keeping at least eight layer cells; "
                    "thicker layers in the same window do not help. "
                    f"Bent mode search found 0 of {num_modes} guided modes.",
                    RuntimeWarning, stacklevel=caller_stacklevel(),
                )
            return ()
        interior = np.zeros((nx, ny), dtype=bool)
        interior[num_pml:nx - num_pml] = True
        mask = np.tile(interior.ravel(), 2)
        straight_fields = [pair[1][mask] for pair in straight]
        straight_norms = [np.linalg.norm(field) for field in straight_fields]
        Bc = self._mass_matrix(nx, ny, bend_radius_um, complex_dtype=True)
        A = self._build_operator(k0, None, num_pml, pml_strength)
        candidates: list[tuple[complex, np.ndarray, float]] = []
        # Search the same 32-pair spectrum at each straight anchor.
        for anchor, _, _, _ in straight:
            sigma = complex((anchor.real) ** 2)
            k_req = max(1, min(2 * N - 2, 32))
            try:
                vals, vecs = _spla.eigs(
                    A, k=k_req, M=Bc, sigma=sigma, which="LM",
                    v0=_deterministic_arpack_start(
                        A.shape[0], complex_dtype=True),
                )
            except _spla.ArpackNoConvergence as exc:
                vals, vecs = exc.eigenvalues, exc.eigenvectors
                if vals.size == 0:
                    continue
            except Exception:
                continue
            shares = self._interior_energy(vecs, nx, ny, num_pml)
            indices = np.sqrt(vals.astype(complex)) / k0
            indices = np.where(indices.real < 0, -indices, indices)
            for col in range(len(vals)):
                if not ((n_min < indices[col].real <= 1.5 * n_max) and
                        shares[col] >= _BEND_MIN_INTERIOR_SHARE):
                    continue
                if any(abs(indices[col] - old[0] / k0) < 1e-9 and
                       self._field_overlap(vecs[:, col], old[1]) > 0.99
                       for old in candidates):
                    continue
                candidates.append((complex(indices[col] * k0),
                                   vecs[:, col].copy(), float(shares[col])))
        scores = []
        for j, (_, vec, share) in enumerate(candidates):
            field = vec[mask]
            norm = np.linalg.norm(field)
            for i, straight_field in enumerate(straight_fields):
                denominator = straight_norms[i] * norm
                projection = (float(abs(np.vdot(straight_field, field)) /
                                    denominator) if denominator else 0.0)
                if projection >= _BEND_MIN_STRAIGHT_PROJECTION:
                    scores.append((projection * np.sqrt(share), i, j))
        used_straight: set[int] = set()
        used_candidates: set[int] = set()
        assigned: list[tuple[complex, np.ndarray, float, int]] = []
        for _, i, j in sorted(scores, reverse=True):
            if i in used_straight or j in used_candidates:
                continue
            used_straight.add(i)
            used_candidates.add(j)
            beta, vec, share = candidates[j]
            assigned.append((beta, vec, share, i))
        assigned.sort(key=lambda item: -item[0].real)
        out: list[VectorMode] = []
        unresolved_reasons: list[str] = []
        outer_col = nx - num_pml if bend_radius_um > 0 else num_pml - 1
        n_out = float(np.sqrt(np.max(eps[:, outer_col])))
        layer_start_um = (nx / 2 - num_pml) * self.dl_x_um
        airy_um = (abs(bend_radius_um) /
                   (2 * (2 * np.pi / wavelength_um * n_out) ** 2)) ** (1 / 3)
        half_A = None
        for mode_number, (beta, vec, share, partner) in enumerate(assigned[:num_modes], 1):
            # PML floor noise can put a passive pair just below zero loss.
            beta = complex(beta.real, max(0.0, beta.imag))
            hx = vec[:N].reshape(nx, ny).T.astype(np.complex128)
            hy = vec[N:].reshape(nx, ny).T.astype(np.complex128)
            mode = self._reconstruct(
                hx.conj(), hy.conj(), beta.conjugate(), k0,
                wavelength_um, bend_radius_um,
                pml_cells_xy=(num_pml, 0),
                pml_strength_xy=(-pml_strength, 0.0),
            )
            caustic_um = abs(bend_radius_um) * (beta.real / k0 / n_out - 1)
            caustic_unresolved = (layer_start_um - caustic_um <
                                  _BEND_MIN_CAUSTIC_MARGIN_AIRY * airy_um)
            # Check attenuation at half strength close to this selected pair.
            # Group-index side solves skip it: only their n_eff is used.
            absorber_unresolved = False
            if check_bend_loss:
                if half_A is None:
                    half_A = self._build_operator(
                        k0, None, num_pml, 0.5 * pml_strength)
                absorber_unresolved = True
                try:
                    half_vals, half_vecs = _spla.eigs(
                        half_A, k=max(1, min(2 * N - 2, 32)), M=Bc,
                        sigma=complex(beta.real ** 2), which="LM",
                        v0=_deterministic_arpack_start(
                            half_A.shape[0], complex_dtype=True),
                    )
                except _spla.ArpackNoConvergence as exc:
                    half_vals, half_vecs = exc.eigenvalues, exc.eigenvectors
                except Exception:
                    half_vals, half_vecs = np.empty(0), np.empty((A.shape[0], 0))
                if half_vecs.shape[1]:
                    overlap = [self._field_overlap(vec, half_vecs[:, j])
                               for j in range(half_vecs.shape[1])]
                    half_beta = np.sqrt(
                        complex(half_vals[int(np.argmax(overlap))]))
                    if half_beta.real < 0:
                        half_beta = -half_beta
                    full_loss = max(0.0, beta.imag)
                    half_loss = max(0.0, half_beta.imag)
                    absorber_unresolved = (
                        abs(half_loss - full_loss) >
                        _BEND_MAX_HALF_LOSS_CHANGE * full_loss)
            loss_unresolved = caustic_unresolved or absorber_unresolved
            if loss_unresolved:
                reasons = []
                if caustic_unresolved:
                    required_width_um = 2 * (
                        num_pml * self.dl_x_um + caustic_um +
                        _BEND_MIN_CAUSTIC_MARGIN_AIRY * airy_um)
                    # The radius refusal caps the window at 2 (|R| - dl).
                    max_width_um = 2 * (abs(bend_radius_um) - self.dl_x_um)
                    if required_width_um > max_width_um:
                        reasons.append(
                            f"caustic: x_L >= x_c + 0.75 l needs a window "
                            f"of {required_width_um:.3g} um, wider than the "
                            f"{max_width_um:.3g} um this radius allows "
                            f"(|R| >= W/2 + dl); this loss cannot be resolved "
                            f"at this radius with {num_pml} layer cells")
                    else:
                        reasons.append(
                            f"caustic: widen the window to at least "
                            f"{required_width_um:.3g} um so x_L >= x_c + 0.75 l")
                if absorber_unresolved:
                    reasons.append("absorber: raise strength or layer count")
                unresolved_reasons.append(
                    f"mode {mode_number} (n={beta.real / k0:.6f}): " +
                    "; ".join(reasons))
            out.append(replace(
                mode, n_eff=float(beta.real / k0),
                k_eff=float(beta.imag / k0),
                window_limited=straight[partner][2] < _STRAIGHT_WINDOW_LIMIT_SHARE,
                loss_unresolved=loss_unresolved,
            ))
        if warn_on_shortage:
            notices = []
            if auto_added_layers:
                notices.append(
                    "A bend radiates, so absorbing layers were added "
                    "automatically at strength 30. Check window convergence "
                    "for bend loss.")
            if unresolved_reasons:
                notices.append("Bend loss unresolved for " + " | ".join(
                    unresolved_reasons) + ".")
            if any(mode.window_limited for mode in out):
                notices.append("A straight partner is window-limited; widen the window.")
            if rejected:
                details = ", ".join(
                    f"n={index:.6f}, share={share:.3f}, "
                    f"half-strength move={move:.2e}"
                    if move is not None else
                    f"n={index:.6f}, share={share:.3f}, "
                    "half-strength move=unavailable"
                    for index, share, move in rejected)
                notices.append(
                    f"Layer-dependent straight pairs rejected ({details}); "
                    "widen the window, keeping at least eight layer cells; "
                    "thicker layers in the same window do not help.")
            if len(out) < num_modes:
                notices.append(
                    f"Bent mode search found {len(out)} of {num_modes} guided "
                    "modes near the straight guide's modes. Widen the window "
                    "or adjust the absorbing layers.")
            if notices:
                warnings.warn(" ".join(notices), RuntimeWarning,
                              stacklevel=caller_stacklevel())
        return tuple(out)

    def _reconstruct(
        self,
        hx: np.ndarray,
        hy: np.ndarray,
        beta: complex,
        k0: float,
        wavelength_um: float,
        bend_radius_um: Optional[float] = None,
        *,
        mode_type: ModeType = "guided",
        pml_cells_xy: Tuple[int, int] = (0, 0),
        pml_strength_xy: Tuple[float, float] = (0.0, 0.0),
        eigen_residual: Optional[float] = None,
    ) -> VectorMode:
        """Reconstruct all six complex field components from ``(Hx, Hy, beta)``.

        ``beta`` is always the returned-field/EME constant:
        ``beta_eme = beta_store.conjugate()`` for a complex passive mode.
        Consequently its imaginary part is negative for passive loss. Complex
        callers conjugate the engine eigenfield, beta, and PML contour together,
        then restore positive ``n_eff``/``k_eff`` engine metadata on the public
        mode after reconstruction.

        Ampère + ``div H = 0`` use ``d/dz -> -i beta`` under
        ``exp(+i*omega*t - i*beta*z)``. For lossless straight modes the transverse
        fields coincide with the engine convention; the purely imaginary
        longitudinal components carry the conjugate sign. Anything comparing
        ``Ez/Hz`` phase directly against engine DFT data must conjugate. With
        ``omega = k0 c0``:

            Hz = (-i / beta) (dHx/dx + dHy/dy),                 (div H = 0)
            (curl H)_x = dHz/dy + i beta Hy,
            (curl H)_y = -i beta Hx - dHz/dx,
            (curl H)_z = dHy/dx - dHx/dy,
            E = (1 / (i omega eps0 eps_eff)) curl H.

        This legacy public surface collocates H and E onto one same-size node
        array, whereas the native FLM lattice places H on cell vertices and
        ``D/E/ε`` at cell centres.  Until that staggered registration is exposed
        here, reconstruction uses a component-aware co-location bridge:
        ``Ex`` smooths ``εxx`` along y, ``Ey`` smooths ``εyy`` along x, and
        ``Ez`` smooths ``εzz`` along both.  This preserves the KFJ normal versus
        tangential tensor distinction and is materially closer to the reference
        cell-centred FLM fields than either scalar-ε division or direct tensor
        division at the wrong lattice points.  It is an approximation, not a
        replacement for native vertex-to-cell FLM reconstruction.  The
        transverse-E pair is L2-normalized and phase-fixed on the dominant
        component's magnitude peak."""
        ny, nx = hx.shape
        dx = self.dl_x_um * 1e-6
        dy = self.dl_y_um * 1e-6
        eps = self.eps
        omega = k0 * self.C0
        # F/m = (mu0 c0^2)^-1, CODATA 2018 (matches the engine's kEps0).
        eps0 = EPS0

        pml_x, pml_y = (int(pml_cells_xy[0]), int(pml_cells_xy[1]))
        strength_x, strength_y = (
            float(pml_strength_xy[0]), float(pml_strength_xy[1])
        )
        sx = self._pml_stretch(nx, pml_x, strength_x, k0)
        sy = self._pml_stretch(ny, pml_y, strength_y, k0)

        # x-min mirror plane: the ghost column -1 is the image of column 0
        # (f[-1] = parity * f[0]). "pec" makes Hx odd and Hy, Hz even; the ghost
        # ε column -1 is column 1, since ε column 0 is centred on the plane.
        mirror = self.x_min_symmetry is not None
        parity_hx = +1.0 if self.x_min_symmetry == "pmc" else -1.0
        parity_ht = -parity_hx                  # Hy and Hz

        def derivative(
            f: np.ndarray, spacing: float, stretch: np.ndarray, axis: int,
            lo_parity: Optional[float] = None,
        ) -> np.ndarray:
            """Differentiate on the same complex coordinate contour as the
            eigen-operator.  Face distances are the arithmetic stretch average;
            a centred denominator is the sum of its two adjacent face distances.
            ``lo_parity`` replaces the one-sided x-min difference by the centred
            one against the mirror ghost ``lo_parity * f[:, 0]``.
            """
            face = 0.5 * (stretch[:-1] + stretch[1:]) * spacing
            g = np.zeros_like(f, dtype=complex)
            if axis == 1:
                g[:, 1:-1] = (
                    (f[:, 2:] - f[:, :-2])
                    / (face[:-1] + face[1:])[None, :]
                )
                if lo_parity is None:
                    g[:, 0] = (f[:, 1] - f[:, 0]) / face[0]
                else:
                    g[:, 0] = ((f[:, 1] - lo_parity * f[:, 0])
                               / (stretch[0] * spacing + face[0]))
                g[:, -1] = (f[:, -1] - f[:, -2]) / face[-1]
            else:
                g[1:-1, :] = (
                    (f[2:, :] - f[:-2, :])
                    / (face[:-1] + face[1:])[:, None]
                )
                g[0, :] = (f[1, :] - f[0, :]) / face[0]
                g[-1, :] = (f[-1, :] - f[-2, :]) / face[-1]
            return g

        def ddx(f: np.ndarray, parity: float) -> np.ndarray:
            return derivative(f, dx, sx, axis=1,
                              lo_parity=parity if mirror else None)

        def ddy(f: np.ndarray) -> np.ndarray:
            return derivative(f, dy, sy, axis=0)

        # Hz from div(H)=0: dHx/dx + dHy/dy - i*beta*Hz = 0 -> Hz = (-i/beta)(...).
        hz = (-1j / beta) * (ddx(hx, parity_hx) + ddy(hy))

        curl_x = ddy(hz) + 1j * beta * hy
        curl_y = -1j * beta * hx - ddx(hz, parity_ht)
        curl_z = ddx(hy, parity_ht) - ddy(hx)

        # Same-grid compatibility bridge for the native FLM vertex/cell
        # staggering (one 3-point pass per tangential axis).  Preserve each
        # tensor component instead of collapsing the anisotropic KFJ boundary
        # cell back to representative scalar epsilon.
        def smooth_y(e: np.ndarray) -> np.ndarray:
            g = e.astype(float).copy()
            g[1:-1, :] = (e[:-2, :] + e[1:-1, :] + e[2:, :]) / 3.0
            return g

        def smooth_x(e: np.ndarray) -> np.ndarray:
            g = e.astype(float).copy()
            g[:, 1:-1] = (e[:, :-2] + e[:, 1:-1] + e[:, 2:]) / 3.0
            if mirror:
                g[:, 0] = (e[:, 1] + e[:, 0] + e[:, 1]) / 3.0
            return g

        eps_x = smooth_y(self._exx)            # for Ex
        eps_y = smooth_x(self._eyy)            # for Ey
        eps_z = smooth_x(smooth_y(self._ezz))  # for Ez
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
        overlap_weights: Optional[np.ndarray]
        physical_mask: Optional[np.ndarray]
        if pml_x > 0 or pml_y > 0:
            # Analytic-coordinate representation: reaction integrals follow the
            # same complex contour dxtilde*dytilde.  Physical Poynting observables
            # instead exclude the artificial PML cells.
            overlap_weights = sy[:, None] * sx[None, :]
            physical_mask = np.ones((ny, nx), dtype=bool)
            if pml_x > 0:
                physical_mask[:, :pml_x] = False
                physical_mask[:, -pml_x:] = False
            if pml_y > 0:
                physical_mask[:pml_y, :] = False
                physical_mask[-pml_y:, :] = False
        else:
            overlap_weights = None
            physical_mask = None

        # placement: the plane is the x origin of a half window. Its array
        # centre sits (nx-1)/2 columns past the plane, so the centred ladder
        # plus this offset reports column j at x = j·dl_x, the coordinate the
        # full solve reports for the same column.
        center_offset_um = (
            (0.5 * (nx - 1) * self.dl_x_um, 0.0) if mirror else None)

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
            center_offset_um=center_offset_um,
            mode_type=mode_type,
            overlap_weights=overlap_weights,
            physical_mask=physical_mask,
            pml_cells_xy=(pml_x, pml_y),
            eigen_residual=eigen_residual,
            x_min_symmetry=self.x_min_symmetry,
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
