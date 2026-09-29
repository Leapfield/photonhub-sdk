"""Yee-grid FDFD waveguide mode solver, the **engine-consistent** discrete mode.

ph's Fallahkhair-Li-Murphy solver (`vector_modes.py`) puts the transverse H at grid
nodes and eps at the four surrounding quadrants, a different discretization from
the engine's FDTD (Yee-staggered E/H, subpixel eps placed per E-component). The
mode it finds therefore is NOT the one the grid propagates, so injecting it radiates
the difference (~2.4% near-source shedding).

This module solves the mode on the **engine's own Yee discretization**:
  * fields on the standard Yee locations (Ex@(i+1/2,j), Ey@(i,j+1/2), Ez@(i,j)),
  * forward/backward staggered curls matching engine/src/kernels/update_body.h,
  * the diagonal KFJ subpixel eps sampled PER-COMPONENT at its own Yee location
    (eps_xx at Ex, eps_yy at Ey, eps_zz at Ez), exactly as the engine samples on the mesh
    (engine/src/cpu_ref/reference_solver.cpp sample_voxel comp_axis).

The eigenproblem is the canonical transverse-E full-vector FDFD (diagonal eps,
mu=1): ``mat @ [Ex;Ey] = -n_eff^2 [Ex;Ey]`` with the block operator built from the
forward/backward derivative matrices (standard formulation; here wired to the
engine's curls + staggered eps). The launched mode is then the FDTD discrete mode,
so a TF/SF injection of it is clean (this is how a mode solve is matched to an FDTD grid).

KNOWN COMPROMISE, real-space field consumers still COLLOCATE the staggered
components. The returned :class:`VectorMode` carries one field array per
component on a single index grid; ``vector_modal_fields`` and its downstream
real-space consumers assign ALL components the same node coordinates
``lo + i*dl`` (+ the carried ``center_offset_um``), discarding the intra-cell
Yee stagger this solver faithfully used (Ex at +1/2 in h, Ey at +1/2 in v,
Ez at the node), a per-component error of up to half a cell. EME reaction
matching is the deliberate exception: ``Ex``/``Hy`` and ``Ey``/``Hx`` are
multiplied directly at their shared native Yee locations. Do not assume other
downstream array consumers preserve those offsets.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional, Tuple

import numpy as np

from ..components.monitors import (
    mode_port_solver_polarization,
    mode_port_trial_modes,
)
from ..viz import _geometry as _geom
from ._constants import C0, MU0
from .kfj_smoothing import _paint_hard
from .vector_modes import VectorMode, _deterministic_arpack_start
from .._compat import caller_stacklevel, legacy_keywords


# --------------------------------------------------------------------------- #
# Yee forward/backward derivative matrices (match engine update_body.h).
# --------------------------------------------------------------------------- #
def dual_spacings(dq: np.ndarray, periodic: bool = False) -> np.ndarray:
    """Dual-grid steps for the backward (H-curl) derivatives from the primal
    steps ``dq`` (§15.2): interior dual width = midpoint average of the two
    adjacent primal cells; the first entry keeps the first primal width (the
    standard convention, ``dl_b[0] = dl_f[0]``, so the
    §20 face rules read the face row over one whole cell). Uniform input
    reproduces the constant spacing exactly. ``periodic`` closes the ladder
    across the seam instead (§15.2: the first dual width straddles the last
    and the first primal cell); the engine admits only seam-symmetric graded
    periodic axes (``dq[0] == dq[-1]``), where the two closures coincide."""
    dq = np.asarray(dq, dtype=float)
    dual = np.empty_like(dq)
    dual[0] = 0.5 * (dq[-1] + dq[0]) if periodic else dq[0]
    dual[1:] = 0.5 * (dq[:-1] + dq[1:])
    return dual


def _dmats(nx: int, ny: int, dl: float, h_min_bc=None, v_min_bc=None,
           dq_h=None, dq_v=None):
    """Forward/backward x,y difference matrices on the nx*ny Yee grid (row-major
    [ix*ny + iy]). ``dxf`` maps a node field to the +1/2 face (forward, = engine
    H-curl-of-E direction); ``dxb`` is its backward adjoint (= engine
    E-curl-of-H direction).

    Spacing: with ``dq_h``/``dq_v`` None (uniform), every row is scaled by the
    single ``1/dl``, byte-identical to the historical operator. A GRADED axis
    passes its PRIMAL spacing vector ``dq`` (length n, §15.1 replicate-last):
    forward rows divide by the primal widths (node i -> i+1 distance), backward
    rows by the DUAL widths (:func:`dual_spacings`), the standard nonuniform
    Yee FDFD (Zhu & Brown; the same construction an open mode solver
    uses, ``diags(1/dls) @ D``). The scalar path is kept verbatim rather than
    expressed as a constant vector so uniform grids stay bit-identical
    (reciprocal-multiply vs divide differ in ULPs).

    Low-edge boundary (``h_min_bc``/``v_min_bc``): ``None`` keeps the legacy
    implicit ghost, the half-located quantities that ``bwd`` acts on (eps*E_n
    and the tangential H) are taken as 0 half a cell outside, a MAGNETIC (PMC)
    mirror OUTSIDE the window; immaterial when the edge sits in decayed
    cladding. ``"pmc"`` / ``"pec"`` instead put the engine's §20 symmetry plane
    exactly ON the low node line: PMC takes the odd ghost X[-1] = -X[0], so
    bwd's first row reads 2*X[0]/dl, the engine's §20.4 backward-read rule;
    PEC takes the even ghost, zeroing that row (the caller must ALSO pin the
    on-plane tangential-E DOFs, see :func:`_solve_yee_eig`). The high edge
    stays the implicit node-line PEC in those cases.

    ``"periodic"`` closes BOTH edges of that axis across the seam (the k = 0
    Bloch wrap of a plain periodic axis, NUMERICS §1/§15.2): ``fwd``'s last
    row reads node 0 as its neighbour and ``bwd``'s first row reads node
    n-1. On a ONE-cell axis both operators are identically zero, the field
    is its own periodic image, so the transverse derivative along that axis
    vanishes and the eigenproblem collapses to the slab (quasi-2D) problem
    the engine actually steps."""
    import scipy.sparse as sp

    def fwd(n, bc, dq):
        if bc == "periodic":
            if n == 1:
                m = sp.csr_matrix((1, 1))           # X[1] := X[0]: zero
            else:
                m = sp.diags([-1.0, 1.0, 1.0], [0, 1, -(n - 1)],
                             shape=(n, n), format="csr")
        else:
            # PEC at the high edge is already implicit in the diags
            # construction: its last row is just -1 on the diagonal (the +1
            # superdiagonal entry falls outside the matrix), i.e. the field
            # is taken as 0 outside the window — verified identical to the
            # explicit lil-matrix edge assignment this replaces.
            m = sp.diags([-1.0, 1.0], [0, 1], shape=(n, n), format="csr")
        if dq is None:
            return m / dl
        return sp.diags(1.0 / np.asarray(dq, dtype=float)) @ m

    def bwd(n, bc, dq):
        if bc == "periodic":
            if n == 1:
                m = sp.csr_matrix((1, 1))           # X[-1] := X[0]: zero
            else:
                m = sp.diags([np.ones(n), -np.ones(n - 1), [-1.0]],
                             [0, -1, n - 1], format="csr")
            if dq is None:
                return m / dl
            return sp.diags(1.0 / dual_spacings(dq, periodic=True)) @ m
        d0 = np.ones(n)
        if bc == "pmc":
            d0[0] = 2.0        # odd ghost: (X[0] - (-X[0]))/dl (§20.4)
        elif bc == "pec":
            d0[0] = 0.0        # even ghost: (X[0] - X[0])/dl
        m = sp.diags([d0, -np.ones(n - 1)], [0, -1], format="csr")
        if dq is None:
            return m / dl
        return sp.diags(1.0 / dual_spacings(dq)) @ m

    Ix, Iy = sp.eye(nx), sp.eye(ny)
    dxf = sp.kron(fwd(nx, h_min_bc, dq_h), Iy, format="csr")
    dxb = sp.kron(bwd(nx, h_min_bc, dq_h), Iy, format="csr")
    dyf = sp.kron(Ix, fwd(ny, v_min_bc, dq_v), format="csr")
    dyb = sp.kron(Ix, bwd(ny, v_min_bc, dq_v), format="csr")
    return dxf, dxb, dyf, dyb


def min_face_symmetry_bcs(sim, axis):
    """The §20 symmetry parity of each in-plane axis' MIN face for a
    cross-section normal to ``axis``, as ``(h_bc, v_bc)`` in :func:`_dmats`
    terms: ``"pec"`` (-1, odd/electric), ``"pmc"`` (+1, even/magnetic), or
    ``None``. Read from ``Simulation.symmetry``; objects without the field
    (duck-typed sims) get ``(None, None)``. NOTE: this is the plane's
    EXISTENCE, whether it applies to a given mode window also requires the
    window's low edge to sit ON the plane (see :func:`window_min_face_bcs`)."""
    sym = getattr(sim, "symmetry", None) or (0, 0, 0)
    h_letter, v_letter = _geom.in_plane_axes(axis)
    m = {-1: "pec", 0: None, 1: "pmc"}
    return (m[int(sym["xyz".index(h_letter)])],
            m[int(sym["xyz".index(v_letter)])])


def window_min_face_bcs(sim, axis, *, h_center, half_w, v_center, half_v, dl):
    """The shared window-registration + symmetry rule for every consumer of a
    cross-section window (the eps sampler, the eigensolve, and the
    source plane MUST agree bit-for-bit on this). Returns
    ``(h_lo, v_lo, h_bc, v_bc)``:

    - ``lo`` = the grid-snapped window origin ``floor((center-half)/dl)*dl``,
      CLIPPED to 0 on any in-plane axis carrying a §20 symmetry plane (the
      plane sits on the domain min face at coordinate 0; the below-plane half
      of a requested window is the mirror image the boundary supplies, it
      must not be solved or stamped).
    - ``bc`` = the axis' symmetry parity when the (clipped) window edge lands
      exactly ON the plane, else ``None`` (an interior window keeps the legacy
      far-from-edge behavior; its tails must not reach the plane, the same
      immaterial-wall assumption every window edge already makes)."""
    h_bc, v_bc = min_face_symmetry_bcs(sim, axis)
    h_lo = float(np.floor((h_center - half_w) / dl) * dl)
    v_lo = float(np.floor((v_center - half_v) / dl) * dl)
    if h_bc is not None and h_lo < 0.0:
        h_lo = 0.0
    if v_bc is not None and v_lo < 0.0:
        v_lo = 0.0
    return (h_lo, v_lo,
            h_bc if h_lo == 0.0 else None,
            v_bc if v_lo == 0.0 else None)


def is_plain_periodic_axis(sim, axis_letter: str) -> bool:
    """True when ``sim``'s ``axis_letter`` WRAPS: a ``"periodic"`` boundary
    with no §20 symmetry plane on it (NUMERICS §1, the axis that may be a
    single cell). Its two faces are one seam, so a mode window that reaches a
    face has no wall there, see :func:`_periodic_window`. Duck-typed sims
    without ``boundaries`` (viz stubs) are never periodic. ``"bloch"`` is NOT
    plain periodic: its wrap carries a phase the k = 0 closure would fold
    incorrectly, so a Bloch axis keeps the wall window."""
    # Not grid.sim_axis_min_cells(...) == 1: that floor reads an unparseable
    # symmetry entry as no plane, and a window must not wrap on that guess.
    bounds = getattr(sim, "boundaries", None)
    kind = getattr(bounds, axis_letter, None) if bounds is not None else None
    if kind != "periodic":
        return False
    sym = getattr(sim, "symmetry", None) or (0, 0, 0)
    try:
        return int(sym["xyz".index(axis_letter)]) == 0
    except (TypeError, ValueError, IndexError):
        return False


def _periodic_window(sim, axis_letter, dl, q):
    """The whole-period ladder ``(nodes, dq, "periodic")`` for a mode window
    on a plain periodic axis.

    The period IS the cross-section's extent on that axis: the engine wraps
    the field across the seam (k = 0), so the solve must too, or the
    eigenproblem sees a fictitious PEC/PMC wall where the structure continues.
    The requested ``(center, half)`` therefore do not size the window on such
    an axis; the ladder is the axis' OWN cell ladder, on a uniform axis
    ``arange(n)*dl_axis`` with ``dq=None`` (the legacy scalar fast path) when
    the caller's ``dl`` is that spacing, else the explicit spacing vector so
    every consumer registers on the true grid; on a graded axis the stored
    coordinates with their §15.1 primal widths (the engine admits only
    seam-symmetric graded periodic axes, so the periodic dual closure equals
    the replicate one). On a ONE-cell axis (the quasi-2D reduction) the ladder
    is the single node ``[0]``: the transverse derivative along that axis is
    then identically zero (:func:`_dmats`) and the solve returns the slab
    modes of the cross-section, the modes the engine actually propagates.
    Before this closure a one-cell axis was solved on a three-node wall window
    painted with the structure's true (few-cell) width in a background sea,
    which reports a barely guided ``n_eff`` just above the cladding and no TE
    family at all."""
    i = "xyz".index(axis_letter)
    if q is None:
        from ..components.grid import realized_cells

        dl_axis = float(sim.grid.dl_um)
        n = realized_cells(float(sim.size_um[i]), dl_axis, min_cells=1)
        nodes = np.arange(n) * dl_axis
        dq = None if abs(float(dl) - dl_axis) <= 1e-9 * dl_axis \
            else np.full(n, dl_axis)
        return nodes, dq, "periodic"
    from ..components.grid import graded_primary_spacings

    nodes = np.asarray(q, dtype=float)
    dq = np.asarray(graded_primary_spacings(tuple(nodes)), dtype=float)
    return nodes, dq, "periodic"


def _axis_window_nodes(sim, axis_letter, center, half, dl, bc):
    """Window node ladder for ONE in-plane axis: ``(nodes, dq, bc_eff)``.

    ``nodes`` are the sim's PRIMARY node coordinates covering
    ``[center-half, center+half]`` (a node at each cell's low edge; the §15.1
    replicate-last primal widths in ``dq``), ``bc_eff`` the §20 parity when the
    window's first node sits ON the min-face plane, or ``"periodic"`` on a
    plain periodic axis (the ladder is then the whole period whatever the
    requested window, :func:`_periodic_window`). A UNIFORM axis reproduces
    :func:`window_min_face_bcs`' floor-snap ladder with the IDENTICAL floats
    (nodes = h_lo + arange(n)*dl) and returns ``dq=None``, the marker every
    downstream consumer uses to take its legacy scalar-dl fast path, keeping
    uniform grids bit-identical."""
    q = sim._axis_coords_um("xyz".index(axis_letter)) \
        if hasattr(sim, "_axis_coords_um") else None
    if is_plain_periodic_axis(sim, axis_letter):
        return _periodic_window(sim, axis_letter, dl, q)
    if q is None:                                   # uniform axis — legacy snap
        lo = float(np.floor((center - half) / dl) * dl)
        if bc is not None and lo < 0.0:
            lo = 0.0
        n = max(3, int(np.ceil((center + half - lo) / dl)))
        return lo + np.arange(n) * dl, None, (bc if lo == 0.0 else None)
    from ..components.grid import graded_primary_spacings

    q = np.asarray(q, dtype=float)
    dq_full = np.asarray(graded_primary_spacings(tuple(q)), dtype=float)
    # floor-snap analogue: first node <= (center-half); cell coverage analogue:
    # last node whose CELL reaches (center+half).
    i_lo = int(np.searchsorted(q, center - half, side="right") - 1)
    if i_lo < 0 or (bc is not None and center - half < 0.0):
        i_lo = 0
    i_hi = int(np.searchsorted(q + dq_full, center + half, side="left"))
    i_hi = min(max(i_hi, i_lo + 2), len(q) - 1)     # >= 3 nodes, in-domain
    nodes = q[i_lo:i_hi + 1]
    dq = dq_full[i_lo:i_hi + 1]
    return nodes, dq, (bc if (i_lo == 0 and nodes[0] == 0.0) else None)


def window_nodes(sim, axis, *, h_center, half_w, v_center, half_v, dl):
    """The graded-aware form of :func:`window_min_face_bcs`: per-axis node
    ladders for a cross-section window normal to ``axis``. Returns
    ``(h_nodes, h_dq, h_bc, v_nodes, v_dq, v_bc)`` where ``dq`` is the primal
    spacing vector for a GRADED axis and ``None`` for a uniform one (the
    legacy-fast-path marker), and ``bc`` is ``"pec"``/``"pmc"`` for a §20
    fold on the axis' min face, ``"periodic"`` on a plain periodic axis (the
    ladder is then the whole period, :func:`_periodic_window`; a one-cell
    quasi-2D axis is one node), else ``None``. Every consumer of the window, the eps sampler,
    the eigensolve, and the equivalence-current sheet, must derive its
    registration from THIS ladder so they agree bit-for-bit."""
    h_bc0, v_bc0 = min_face_symmetry_bcs(sim, axis)
    h_letter, v_letter = _geom.in_plane_axes(axis)
    h_nodes, h_dq, h_bc = _axis_window_nodes(sim, h_letter, h_center, half_w,
                                             dl, h_bc0)
    v_nodes, v_dq, v_bc = _axis_window_nodes(sim, v_letter, v_center, half_v,
                                             dl, v_bc0)
    return h_nodes, h_dq, h_bc, v_nodes, v_dq, v_bc


# --------------------------------------------------------------------------- #
# Per-component (staggered) diagonal-KFJ eps on the Yee grid.
# --------------------------------------------------------------------------- #
def _fine_centers(h0, v0, nh, nv, dl, off_h, off_v, ss):
    """The supersampled fill-fraction grid centers for the nh*nv Yee-offset grid
    whose node (ih,iv) sits at (h0+(ih+off_h)*dl, v0+(iv+off_v)*dl)."""
    # The engine evaluates a uniform-grid point from its GLOBAL integer index,
    # ``(i + offset) * dl``.  Reassociating this as ``h0 + i * dl`` makes the
    # last bit depend on the selected window origin and can flip exact closed-
    # face Box membership when an otherwise identical window is padded.  The
    # origins are grid-snapped, so recover their integer indices and retain the
    # engine arithmetic for every sub-sample.
    h_index0 = int(np.rint(h0 / dl))
    v_index0 = int(np.rint(v0 / dl))
    sub = (np.arange(ss) - (ss - 1) / 2.0) / ss
    fine_h = (
        h_index0 + np.arange(nh)[:, None] + off_h + sub[None, :]
    ) * dl
    fine_v = (
        v_index0 + np.arange(nv)[:, None] + off_v + sub[None, :]
    ) * dl
    fine_h = fine_h.ravel()
    fine_v = fine_v.ravel()
    return fine_h, fine_v


def _kfj_reduce(eps_fine, nh, nv, ss, dl, pts=None, periods=(None, None)):
    """Diagonal-KFJ (eps_par, eps_xx_along_h, eps_yy_along_v) reduction of a
    supersampled hard-paint ``eps_fine`` [v*ss, h*ss]. Returns arrays [iv, ih].
    ``pts`` = optional ``(h_pts, v_pts)`` Yee point coordinates for a GRADED
    window (the interface-normal gradient then uses the true nonuniform
    spacings); ``None`` keeps the scalar-``dl`` gradient bit-identically.
    ``periods`` = ``(h_period, v_period)``: the period length of an axis the
    window wraps (``_WindowGeom.h_period``), ``None`` on a walled or folded
    axis. A wrapped axis takes the central difference across its seam too."""
    blk = eps_fine.reshape(nv, ss, nh, ss)
    # For isotropic constituents sharing one interface normal the Kottke/KFJ
    # construction gives EXACTLY eps_par = <eps> and eps_perp = <1/eps>^-1 for ANY
    # number of media; the (emax, emin, f) two-phase form is only the N=2 case and
    # mis-assigns every INTERMEDIATE medium to emin (at an air / BOX / core triple
    # junction the BOX sub-cells were counted as air). Average all sub-samples.
    epar = blk.mean(axis=(1, 3))                              # arithmetic, ε‖
    eperp = 1.0 / np.mean(1.0 / blk, axis=(1, 3))             # harmonic, ε⊥
    # Interface normal from the gradient of the CONTINUOUS mean-eps field, NOT
    # of ``f``: f is the fill of the per-cell brightest material, which is == 1
    # in BOTH bulk media, so a wall contained in a single cell column reads
    # 1, f_wall, 1 and the central difference at the wall cell VANISHES ->
    # n_hat^2 = 0 -> every component silently got the ARITHMETIC average (the
    # normal component must be harmonic). A wall straddling two columns DID get
    # a normal, so the defect was registration-dependent (mode-profile errors
    # that walk with dl). epar is monotone across any two-phase wall, so its
    # gradient always points along the true normal; in uniform cells d == 0
    # makes the zero-gradient fallback irrelevant.
    # Per axis, so a ONE-sample axis (the one-cell periodic axis of a
    # quasi-2D window) contributes no normal component: the structure is
    # uniform across that cell by construction, so its gradient is zero (numpy
    # refuses a gradient on fewer than two samples). Two-plus samples take
    # the identical central-difference numpy evaluates for the joint call.
    # On a wrapped axis the first and last samples are neighbours across the
    # seam: one wrapped sample is padded on each side and only the original
    # entries are kept, so the seam cells take the same central difference
    # as the interior (numpy's one-sided edge difference there moved n_eff
    # with the placement of a wall in the seam cell) and every interior entry
    # is bit-identical to the unpadded call.
    def _grad(axis, coord, period):
        n = epar.shape[axis]
        if n < 2:
            return np.zeros_like(epar)
        if period is None:
            return np.gradient(epar, coord, axis=axis)
        ext = np.concatenate((np.take(epar, [n - 1], axis=axis), epar,
                              np.take(epar, [0], axis=axis)), axis=axis)
        if np.ndim(coord) == 0:
            c = coord
        else:
            c = np.asarray(coord, dtype=float)
            c = np.concatenate(([c[-1] - period], c, [c[0] + period]))
        return np.take(np.gradient(ext, c, axis=axis), np.arange(1, n + 1), axis=axis)
    gy = _grad(0, dl if pts is None else pts[1], periods[1])
    gx = _grad(1, dl if pts is None else pts[0], periods[0])
    gmag = np.hypot(gx, gy); safe = np.where(gmag > 1e-12, gmag, 1.0)
    nh2 = np.where(gmag > 1e-12, (gx / safe) ** 2, 0.0)
    nv2 = np.where(gmag > 1e-12, (gy / safe) ** 2, 0.0)
    d = eperp - epar
    return epar, epar + d * nh2, epar + d * nv2     # (eps_par, eps_along_h, eps_along_v)


def _kfj_at_offset(sim, axis, plane_value_um, h0, v0, nh, nv, dl, off_h, off_v,
                   ss, eps_of):
    """Diagonal-KFJ (eps_par, eps_xx_along_h, eps_yy_along_v) sampled on the nh*nv
    grid whose node (ih,iv) sits at (h0+(ih+off_h)*dl, v0+(iv+off_v)*dl). Returns
    arrays shaped [iv, ih]. One-shot form of the paint+reduce pair
    (:func:`staggered_eps_sampler` is the geometry-cached, per-frequency form)."""
    fine_h, fine_v = _fine_centers(h0, v0, nh, nv, dl, off_h, off_v, ss)
    eps_fine = _paint_hard(sim, axis, plane_value_um, fine_h, fine_v, eps_of)
    return _kfj_reduce(eps_fine, nh, nv, ss, dl)


class _WindowGeom:
    """The resolved cross-section window: node ladders, primal spacings
    (``None`` on a uniform axis, the legacy-fast-path marker), §20 BCs, and
    the per-offset Yee point coordinates. ``graded`` is True when either
    in-plane axis actually grades."""

    def __init__(self, h_nodes, h_dq, h_bc, v_nodes, v_dq, v_bc, dl):
        self.h_nodes, self.h_dq, self.h_bc = h_nodes, h_dq, h_bc
        self.v_nodes, self.v_dq, self.v_bc = v_nodes, v_dq, v_bc
        self.dl = dl
        self.nh, self.nv = len(h_nodes), len(v_nodes)
        self.h_lo, self.v_lo = float(h_nodes[0]), float(v_nodes[0])
        self.graded = h_dq is not None or v_dq is not None
        # The period length of a wrapped ("periodic") axis: its closing node
        # plus the closing cell (§15.1 replicate-last on a graded ladder);
        # None on a walled or folded axis.
        self.h_period = self._period(h_nodes, h_dq, h_bc)
        self.v_period = self._period(v_nodes, v_dq, v_bc)

    def _period(self, nodes, dq, bc):
        if bc != "periodic":
            return None
        return float(nodes[-1] + (dq[-1] if dq is not None else self.dl))

    def pts(self, axis_hv, offset):
        """Yee point coordinates along one window axis ('h'|'v') at Yee offset
        0 (node) or 0.5 (mid-cell)."""
        nodes = self.h_nodes if axis_hv == "h" else self.v_nodes
        dq = self.h_dq if axis_hv == "h" else self.v_dq
        if offset == 0.0:
            return nodes
        if dq is None:
            return nodes + 0.5 * self.dl
        return nodes + 0.5 * dq


def _fine_centers_graded(nodes, dq, dl, offset, ss):
    """Per-cell supersample points along one graded-aware axis. Offset 0.5
    samples each PRIMAL cell [q_i, q_i+dq_i] (the cell whose centre is the Yee
    point); offset 0 samples the DUAL cell [q_i - dq_{i-1}/2, q_i + dq_i/2]
    (replicate-first at the edge). Returns flat [i*ss + k] points. (The
    uniform path keeps :func:`_fine_centers` verbatim, algebraically equal
    but float-op-order different, and uniform must stay bit-identical.)"""
    frac = (np.arange(ss) + 0.5) / ss
    if dq is None:
        dq = np.full(len(nodes), dl)
    if offset == 0.5:
        lo, w = nodes, dq
    else:
        dqm = np.concatenate(([dq[0]], dq[:-1]))
        lo, w = nodes - 0.5 * dqm, 0.5 * (dqm + dq)
    return (lo[:, None] + frac[None, :] * w[:, None]).ravel()


def staggered_eps_sampler(sim, axis, plane_value_um, *, h_center, v_center,
                          half_w, half_v, dl, supersample=8, eps_of_medium=None):
    """Frequency-parameterized Yee-staggered eps sampler. Samples the window on the mesh
    GEOMETRY once (a structure-index paint per Yee offset) and returns
    ``(sample, geom)`` where ``sample(freq_hz)`` maps material values at that
    frequency onto the cached geometry and returns the flat
    ``(eps_xx, eps_yy, eps_zz)``, and ``geom`` is the :class:`_WindowGeom`
    (node ladders + spacings + §20 BCs) every downstream consumer must derive
    its registration from. Graded in-plane axes are supported natively: each
    Yee point's sampling cell is its own primal/dual cell (§15), and the KFJ
    interface normal uses the true point coordinates. Uniform axes keep the
    legacy fine-center path bit-identically.

    Material values: an ``eps_of_medium`` entry wins at EVERY frequency (an
    explicit anchor is frozen by design; the freeze is PER medium, other,
    un-overridden dispersive media keep their per-frequency anchoring);
    otherwise :meth:`Medium.permittivity_at_hz` at ``sample``'s ``freq_hz`` ,
    for a Lorentz medium the band value, NOT the eps_inf that bare
    ``permittivity`` is; ``sample(None)`` falls back to bare ``permittivity``
    (legacy). When no un-overridden dispersive medium is present the first
    result is cached and the (large) geometry samples released, callers just
    call ``sample(f)`` per frequency and the sampler decides what repeats."""
    from .kfj_smoothing import (_any_dispersive, _default_eps_of, _eps_lut,
                                _paint_indices)

    geom = _WindowGeom(*window_nodes(
        sim, axis, h_center=h_center, half_w=half_w,
        v_center=v_center, half_v=half_v, dl=dl), dl=dl)
    nh, nv = geom.nh, geom.nv
    ss = int(supersample)
    # Geometry paint per Yee offset: Ex at (h+1/2), Ey at (v+1/2), node.
    idx_maps, grads = [], []
    for off_h, off_v in ((0.5, 0.0), (0.0, 0.5), (0.0, 0.0)):
        if not geom.graded:
            # legacy fine centers — bit-identical on uniform grids
            fh, fv = _fine_centers(geom.h_lo, geom.v_lo, nh, nv, dl,
                                   off_h, off_v, ss)
        else:
            fh = _fine_centers_graded(geom.h_nodes, geom.h_dq, dl, off_h, ss)
            fv = _fine_centers_graded(geom.v_nodes, geom.v_dq, dl, off_v, ss)
        # A wrapped axis paints its sub-samples modulo the period: the dual
        # cell of node 0 (the Ey and node offsets) reaches half a cell below
        # the seam, where the structure is the top half of the LAST cell, not
        # background (unwrapped, a one-cell Si slab read as half air at Ey and
        # Ez, which cost the TM family its guided mode).
        if geom.h_period is not None:
            fh = np.mod(fh, geom.h_period)
        if geom.v_period is not None:
            fv = np.mod(fv, geom.v_period)
        idx_maps.append(_paint_indices(sim, axis, plane_value_um, fh, fv))
        # gradient coordinates = the Yee point positions (None -> scalar dl)
        grads.append(None if not geom.graded else
                     (geom.pts("h", off_h), geom.pts("v", off_v)))
    frozen = not _any_dispersive(sim, eps_of_medium)
    state = {}

    def flat(a):  # [iv, ih] -> flat [ih*nv + iv]
        return a.T.ravel()

    def sample(freq_hz):
        if "cached" in state:
            return state["cached"]
        lut = _eps_lut(sim, _default_eps_of(eps_of_medium, freq_hz))
        # eps_xx: tensor h-component at Ex; eps_yy: v-component at Ey;
        # eps_zz: eps_par (propagation tangential) at the node.
        periods = (geom.h_period, geom.v_period)
        _, exx, _ = _kfj_reduce(lut[idx_maps[0] + 1], nh, nv, ss, dl, grads[0], periods)
        _, _, eyy = _kfj_reduce(lut[idx_maps[1] + 1], nh, nv, ss, dl, grads[1], periods)
        ezz, _, _ = _kfj_reduce(lut[idx_maps[2] + 1], nh, nv, ss, dl, grads[2], periods)
        result = flat(exx), flat(eyy), flat(ezz)
        if frozen:
            # eps is frequency-independent: keep the three flat vectors,
            # release the supersampled geometry rasters.
            state["cached"] = result
            idx_maps.clear()
        return result

    return sample, geom


def sample_staggered_eps(sim, axis, plane_value_um, *, h_center, v_center,
                         half_w, half_v, dl, supersample=8, eps_of_medium=None,
                         freq_hz=None):
    """Diagonal subpixel eps at the Yee E-component locations, snapped to the sim
    grid. Returns ``(eps_xx, eps_yy, eps_zz, nh, nv, h_lo, v_lo)``: the three
    eps components each a flat [ih*nv+iv] vector of length nh*nv (eps_xx at the
    Ex location (+1/2 in h), eps_yy at Ey (+1/2 in v), eps_zz at the node), the
    grid extents, and the snapped window origin (microns), node (ih, iv) sits
    at ``(h_lo + ih*dl, v_lo + iv*dl)`` on a uniform grid (on a graded one the
    nodes are the sim's own ladder; use :func:`staggered_eps_sampler` for the
    full geometry). h = mode-x (width), v = mode-y (height). ``freq_hz``
    anchors dispersive media at that frequency (see
    :func:`staggered_eps_sampler`); ``None`` keeps the legacy bare
    ``permittivity`` (= eps_inf for a Lorentz medium, wrong for a dispersive
    solve, so pass the solve frequency)."""
    sample, geom = staggered_eps_sampler(
        sim, axis, plane_value_um, h_center=h_center, v_center=v_center,
        half_w=half_w, half_v=half_v, dl=dl, supersample=supersample,
        eps_of_medium=eps_of_medium)
    exx, eyy, ezz = sample(freq_hz)
    return exx, eyy, ezz, geom.nh, geom.nv, geom.h_lo, geom.v_lo


# --------------------------------------------------------------------------- #
# The eigenproblem (canonical transverse-E FDFD, diagonal eps, mu=1).
# --------------------------------------------------------------------------- #
def _solve_yee_eig(exx, eyy, ezz, nh, nv, wavelength_um: float, dl_um: float,
                   nmodes: int, center_offset=None, h_min_bc=None,
                   v_min_bc=None, dq_h=None, dq_v=None,
                   x_coords_um=None, y_coords_um=None, min_neff: float = 1.0):
    """Solve the discrete-Yee eigenproblem at ``wavelength_um`` for pre-sampled,
    Yee-staggered diagonal permittivity arrays over a window, returning
    :class:`VectorMode`\\ s above ``min_neff`` in descending real ``n_eff``.
    The legacy default ``min_neff=1`` retains guided modes; the experimental
    hard-wall EME caller lowers it to retain propagating box-radiation modes.
    Pure-imaginary-beta roots remain excluded because reconstruction divides by
    real beta. Each mode carries its ordinary right-eigenpair residual. Factored
    out of :func:`solve_yee_mode`
    so a per-frequency bank (:func:`solve_yee_mode_bank`) can sample the ε ONCE and
    re-solve per λ (only ``k0`` changes), the Yee analogue of
    :meth:`VectorModeSolver.at_wavelength`. ``center_offset`` is the window
    placement metadata computed by the caller from ``sample_staggered_eps``'s
    snapped origin (see :func:`_window_center_offset`), carried on every
    returned mode.

    ``h_min_bc``/``v_min_bc`` = ``"periodic"`` closes that axis across its
    seam (k = 0 wrap, see :func:`_dmats`; a one-cell axis drops out of the
    operator entirely). ``"pec"``/``"pmc"`` put an engine §20 symmetry plane
    ON the window's low node line (see :func:`_dmats`): the restricted half of the matching-
    parity full-window eigenmode satisfies the half problem EXACTLY on the
    lattice, so a half-window solve reproduces the full mode's n_eff to
    eigensolver precision. ``"pec"`` (-1, odd) additionally pins the on-plane
    tangential-E DOFs (E_v on an h-min plane, E_h on a v-min plane), the
    engine pins those same nodes every step; the mode's values there are 0 by
    parity. With a plane active, the spectrum contains ONLY the matching-
    parity family (mode_index counts within it).

    ``dq_h``/``dq_v`` = optional PRIMAL spacing vectors for GRADED window axes
    (:func:`_dmats` then builds the nonuniform Yee operators; ``None`` keeps
    the uniform scalar path bit-identically). ``x_coords_um``/``y_coords_um``
    = the node ladders RELATIVE to the requested mode centre, carried on the
    returned :class:`VectorMode` so consumers place a graded-solved mode on
    its true nonuniform mesh (uniform callers may pass ``None``, consumers
    then reconstruct coords from ``dl_x_um`` as before)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    N = nh * nv
    k0 = 2.0 * np.pi / wavelength_um            # 1/um (consistent with dl in um)
    dl = dl_um
    bcs = dict(h_min_bc=h_min_bc, v_min_bc=v_min_bc,
               dq_h=dq_h, dq_v=dq_v)
    dxf, dxb, dyf, dyb = (m / k0 for m in _dmats(nh, nv, dl, **bcs))
    inv_ezz = sp.spdiags(1.0 / ezz, 0, N, N)
    I = sp.eye(N)
    p_mu = sp.bmat([[None, I], [-I, None]], format="csr")
    p_partial = sp.bmat([[-dxf @ inv_ezz @ dyb, dxf @ inv_ezz @ dxb],
                         [-dyf @ inv_ezz @ dyb, dyf @ inv_ezz @ dxb]], format="csr")
    q_ep = sp.bmat([[None, sp.spdiags(eyy, 0, N, N)],
                    [-sp.spdiags(exx, 0, N, N), None]], format="csr")
    q_partial = sp.bmat([[-dxb @ dyf, dxb @ dxf],
                         [-dyb @ dyf, dyb @ dxf]], format="csr")
    qmat = q_ep + q_partial
    mat = (p_mu @ qmat + p_partial @ q_ep).tocsc()

    if h_min_bc == "pec" or v_min_bc == "pec":
        # Pin the tangential-E DOFs ON the plane (odd -> identically 0), like
        # the engine's §4/§20 face pinning: zero their rows AND columns. The
        # decoupled DOFs get eigenvalue 0 == n_eff^2 = 0, far outside the
        # guided shift-invert window, so they never surface as modes.
        mask = np.ones(2 * N)
        ii = np.arange(N).reshape(nh, nv)
        if h_min_bc == "pec":
            mask[N + ii[0, :]] = 0.0     # E_v (ey block) on the h=lo line
        if v_min_bc == "pec":
            mask[ii[:, 0]] = 0.0         # E_h (ex block) on the v=lo line
        P = sp.spdiags(mask, 0, 2 * N, 2 * N)
        mat = (P @ mat @ P).tocsc()

    # KFJ samples epsilon independently at the three E-component locations.
    # A sub-cell wall can therefore leave (say) exx harmonic-averaged while
    # eyy/ezz retain the full core epsilon.  Shifting from max(exx) alone can
    # target the radiation spectrum and omit every guided root.  The scalar-
    # dielectric upper bound is the maximum over all component samples.
    n_core = float(
        np.sqrt(
            max(
                np.max(np.asarray(exx).real),
                np.max(np.asarray(eyy).real),
                np.max(np.asarray(ezz).real),
            )
        )
    )
    if nmodes >= 2 * N - 1:
        raise ValueError(
            f"trial mode count {nmodes} is too large for a {nh} x {nv} "
            f"cross-section ({2 * N} transverse unknowns); reduce num_modes "
            "or enlarge the solve window")
    vals, vecs = spl.eigs(
        mat,
        k=nmodes,
        sigma=-(n_core ** 2),
        which="LM",
        v0=_deterministic_arpack_start(
            mat.shape[0],
            complex_dtype=np.issubdtype(mat.dtype, np.complexfloating),
        ),
    )
    neff = np.sqrt(-vals)                        # eigenvalue = -n_eff^2
    order = np.argsort(-neff.real)
    vals, neff, vecs = vals[order], neff[order], vecs[:, order]

    # raw (per-um) and per-meter derivative ops for field reconstruction
    Dxf_u, Dxb_u, Dyf_u, Dyb_u = _dmats(nh, nv, dl, **bcs)     # per um
    bcs_m = dict(h_min_bc=h_min_bc, v_min_bc=v_min_bc,
                 dq_h=None if dq_h is None else np.asarray(dq_h) * 1e-6,
                 dq_v=None if dq_v is None else np.asarray(dq_v) * 1e-6)
    Dxf_m, _, Dyf_m, _ = _dmats(nh, nv, dl * 1e-6, **bcs_m)   # per m
    diag_exx = sp.spdiags(exx, 0, N, N); diag_eyy = sp.spdiags(eyy, 0, N, N)
    inv_e = sp.spdiags(1.0 / ezz, 0, N, N)
    omega = C0 * k0 * 1e6                                     # rad/s (k0 in 1/um)
    pref = 1j / (omega * MU0)                                 # H = (i/(w*mu0)) curl E
    modes = []
    for m in range(len(order)):
        ne = complex(neff[m])
        if ne.real <= min_neff:
            continue
        eigvec = np.asarray(vecs[:, m], dtype=complex)
        av = mat @ eigvec
        residual_denominator = (
            np.linalg.norm(av)
            + abs(complex(vals[m])) * np.linalg.norm(eigvec)
        )
        eigen_residual = float(
            np.linalg.norm(av - complex(vals[m]) * eigvec)
            / max(float(residual_denominator), 1e-300)
        )
        ex = vecs[:N, m]; ey = vecs[N:, m]
        beta = ne.real * k0                                  # per um
        beta_m = beta * 1e6                                  # per m
        # Ez from div(eps E)=0:  i*beta*eps_zz*Ez = d/dx(eps_xx Ex)+d/dy(eps_yy Ey)
        ezc = (inv_e @ (Dxb_u @ (diag_exx @ ex) + Dyb_u @ (diag_eyy @ ey))) / (1j * beta)
        hx = pref * (Dyf_m @ ezc + 1j * beta_m * ey)
        hy = pref * (-1j * beta_m * ex - Dxf_m @ ezc)
        hz = pref * (Dxf_m @ ey - Dyf_m @ ex)
        def grid(v):  # flat [ih*nv+iv] -> [iy=v, ix=h]
            return v.reshape(nh, nv).T
        exg, eyg, ezg = grid(ex), grid(ey), grid(ezc)
        hxg, hyg, hzg = grid(hx), grid(hy), grid(hz)
        # Restore VectorMode's declared invariant (vector_modes.py): the
        # transverse-E pair jointly L2-normalized and phase-fixed so the
        # dominant transverse-E component is real-positive at its magnitude
        # peak. eigs returns an arbitrary eigenvector scale/phase; consumers
        # renormalize powers anyway, but the invariant keeps sign-sensitive
        # paths (profile sign alignment, phase pins) deterministic.
        norm = float(np.sqrt(np.sum(np.abs(exg) ** 2 + np.abs(eyg) ** 2)))
        ref = exg if np.sum(np.abs(exg) ** 2) >= np.sum(np.abs(eyg) ** 2) else eyg
        peak = ref.flat[int(np.argmax(np.abs(ref)))]
        phase_fix = np.conj(peak) / abs(peak) if abs(peak) > 0 else 1.0 + 0.0j
        g = (1.0 / norm if norm > 0 else 1.0) * phase_fix
        vm = VectorMode(n_eff=ne.real, n_group=None,
                        ex=exg * g, ey=eyg * g, ez=ezg * g,
                        hx=hxg * g, hy=hyg * g, hz=hzg * g,
                        wavelength_um=wavelength_um, dl_x_um=dl, dl_y_um=dl,
                        k_eff=ne.imag, center_offset_um=center_offset,
                        yee_staggered=True,   # solved on the engine's Yee grid
                        x_coords_um=x_coords_um, y_coords_um=y_coords_um,
                        eigen_residual=eigen_residual)
        modes.append(vm)
    return modes


def _window_center_offset(h_lo, v_lo, nh, nv, dl_um, h_center_um, v_center_um):
    """Window placement metadata: node (ih, iv) sits at (lo + i*dl), so the
    array center, which consumers place at the requested center, actually
    sits at lo + (n-1)/2*dl. Carrying the difference lets vector_modal_fields
    put the mode back where its sampled window truly was (the grid snap displaces it by
    up to ~a cell). The remaining per-component ±dl/2 Yee stagger is a
    separate, documented compromise (consumers collocate, module docstring)."""
    return (h_lo + 0.5 * (nh - 1) * dl_um - h_center_um,
            v_lo + 0.5 * (nv - 1) * dl_um - v_center_um)


def _window_placement(geom: "_WindowGeom", h_center_um, v_center_um):
    """(center_offset, x_coords_um, y_coords_um) for a solved window. Uniform
    windows keep the legacy scalar offset (identical floats) and carry NO
    coords, consumers reconstruct the mesh from ``dl_x_um`` exactly as
    before. Graded windows additionally carry the node ladders RELATIVE to the
    requested centre, which coordinate-aware consumers must prefer."""
    if not geom.graded:
        off = _window_center_offset(geom.h_lo, geom.v_lo, geom.nh, geom.nv,
                                    geom.dl, h_center_um, v_center_um)
        return off, None, None
    off = (0.5 * (geom.h_nodes[0] + geom.h_nodes[-1]) - h_center_um,
           0.5 * (geom.v_nodes[0] + geom.v_nodes[-1]) - v_center_um)
    return (off, geom.h_nodes - h_center_um, geom.v_nodes - v_center_um)


#: Largest field amplitude a solved mode may carry on an ARTIFICIAL window
#: face, as a fraction of its own peak, before :func:`solve_yee_mode` warns
#: that the window is too tight. Every window truncates the mode's evanescent
#: tail against a hard wall; the number below is where that truncation stops
#: being negligible. Calibrated on a 500 x 220 nm Si-in-SiO2 strip at 37 nm
#: cells, full domain and z-folded: 0.05 flags every window whose n_eff is off
#: by >= 2e-3 and stays quiet at <= 1.4e-3, and leaves the shipped example
#: notebooks silent. Pinned by test_yee_symmetry.py.
#:
#: The threshold tracks n_eff, which is the WEAKER of the two things a window
#: controls. n_eff is an eigenvalue and converges second order in the wall
#: perturbation; the PROFILE converges first order, so a window can hold n_eff
#: to 1e-6 and still be visibly one-sided. On a symmetric cross-section the
#: mode's own mirror mismatch runs ~2.7e-3 at half_w_um 0.85 (67 nm cells) and
#: only reaches 1e-9 near 2.6 um, while n_eff is settled by 0.85. A readout
#: that depends on the profile's symmetry, such as two arms of a symmetric
#: splitter, therefore needs a wider window than this warning demands.
_WINDOW_EDGE_TOL = 0.05


def _window_edge_amplitude(mode, h_bc, v_bc) -> float:
    """``max|E|`` on the window's ARTIFICIAL faces over the mode's peak ``|E|``.

    A §20 symmetry face is not artificial, it is the mode's own mirror plane
    and routinely carries the field PEAK (a fold-antinode mode reads 1.0
    there), so a face whose ``bc`` is set is excluded; a ``"periodic"``
    axis has no artificial face at all (both edges are the one seam the
    field wraps across). Arrays are ``[iy=v, ix=h]``: row 0 / -1 are the
    v-min / v-max faces, column 0 / -1 the h-min / h-max."""
    e = np.sqrt(np.abs(np.asarray(mode.ex)) ** 2
                + np.abs(np.asarray(mode.ey)) ** 2
                + np.abs(np.asarray(mode.ez)) ** 2)
    peak = float(np.max(e))
    if not peak > 0.0:
        return 0.0
    faces = []
    if v_bc != "periodic":
        faces.append(float(np.max(e[-1, :])))
        if v_bc is None:
            faces.append(float(np.max(e[0, :])))
    if h_bc != "periodic":
        faces.append(float(np.max(e[:, -1])))
        if h_bc is None:
            faces.append(float(np.max(e[:, 0])))
    return max(faces) / peak if faces else 0.0


def _warn_tight_window(mode, geom: "_WindowGeom", axis, plane_value_um):
    """Warn when the solved mode is still large on a hard window wall.

    The wall is not neutral: the window's low faces carry an implicit MAGNETIC
    wall half a cell out (``_dmats``' ``bwd`` zero ghost) and its high faces an
    implicit ELECTRIC wall one cell out (``fwd``'s), and the two bias n_eff in
    OPPOSITE directions. On an unfolded window the two partially cancel, so a
    tight window still reads plausibly; a §20 fold replaces the low wall with
    the exact mirror and removes that cancellation, so the SAME ``half_v_um``
    that looked converged unfolded can be an order of magnitude worse folded.
    Neither case is visible in the returned n_eff, hence the warning."""
    edge = _window_edge_amplitude(mode, geom.h_bc, geom.v_bc)
    if edge <= _WINDOW_EDGE_TOL:
        return
    folded = [name for name, bc in (("h", geom.h_bc), ("v", geom.v_bc))
              if bc in ("pec", "pmc")]
    note = (" The window's " + "/".join(folded) + "-min face sits on a "
            "symmetry plane. The mirror there replaces the wall whose error "
            "partly cancels the opposite wall's on a full window, so a "
            "half-domain window needs a LARGER half-width than the same guide "
            "on a full domain." if folded else "")
    import warnings

    warnings.warn(
        f"mode window at {axis}={plane_value_um:.3f} is tight: the mode still "
        f"carries {edge:.1%} of its peak |E| on a hard window wall (limit "
        f"{_WINDOW_EDGE_TOL:.0%}), so the wall, not the guide, is setting its "
        f"n_eff and profile. Enlarge half_w_um/half_v_um until the reading "
        f"stops moving.{note}",
        UserWarning, stacklevel=caller_stacklevel())


def _pick_yee(modes, pol, mode_index, axis, plane_value_um):
    """Return the ``mode_index``-th ``pol``-polarized mode (n_eff-descending)."""
    cands = [m for m in modes if m.polarization == pol]
    if mode_index >= len(cands):
        raise RuntimeError(
            f"yee mode solve at {axis}={plane_value_um:.3f}: requested {pol}{mode_index} "
            f"but found {len(cands)} {pol} mode(s); "
            f"n_eff={[round(m.n_eff, 4) for m in modes]}, "
            f"te_frac={[round(m.te_fraction, 3) for m in modes]}")
    return cands[mode_index]


def _homogeneous_yee_exterior(exx, eyy, ezz, geom: "_WindowGeom") -> float:
    """Return the scalar relative permittivity on the hard-wall box exterior.

    The classifier used by :func:`solve_yee_eme_basis` needs one unambiguous
    exterior light line. Check every Yee E-component on all four outer faces.
    Requiring all samples to agree catches a core, substrate, or material
    junction that reaches the box wall. Symmetry-reduced boxes are rejected by
    the public caller before this helper is reached.
    """

    def grid(values):
        raw = np.asarray(values)
        if np.iscomplexobj(raw) and np.any(np.abs(raw.imag) > 0.0):
            raise ValueError(
                "solve_yee_eme_basis requires real, lossless sampled "
                "permittivity; complex epsilon is not supported"
            )
        return np.asarray(raw.real, dtype=float).reshape(geom.nh, geom.nv).T

    edge_values = []
    for component in (grid(exx), grid(eyy), grid(ezz)):
        edge_values.append(component[0, :])
        edge_values.append(component[-1, :])
        edge_values.append(component[:, 0])
        edge_values.append(component[:, -1])
    edge = np.concatenate([np.ravel(values) for values in edge_values])
    exterior_eps = float(edge[0])
    if (
        not np.isfinite(edge).all()
        or exterior_eps <= 0.0
        or not np.allclose(edge, exterior_eps, rtol=1e-8, atol=1e-10)
    ):
        edge_min = float(np.nanmin(edge)) if edge.size else float("nan")
        edge_max = float(np.nanmax(edge)) if edge.size else float("nan")
        raise ValueError(
            "solve_yee_eme_basis requires one homogeneous scalar exterior "
            "around the hard-wall window so the guided/radiation light line "
            f"is defined; sampled exterior permittivity spans "
            f"[{edge_min:.6g}, {edge_max:.6g}]. Enlarge the transverse "
            "window beyond every material interface."
        )
    return exterior_eps


def _stable_positive_yee_modes(modes, residual_tolerance: float):
    """Near-real, finite, positive-beta hard-wall modes with good Ritz pairs."""
    stable = []
    for mode in modes:
        scale = max(abs(float(mode.n_eff)), 1.0)
        finite_fields = all(
            np.isfinite(field).all()
            for field in (mode.ex, mode.ey, mode.ez, mode.hx, mode.hy, mode.hz)
        )
        residual = mode.eigen_residual
        if (
            mode.n_eff > 1e-6
            and np.isfinite(mode.n_eff)
            and np.isfinite(mode.k_eff)
            and abs(mode.k_eff) <= 1e-8 * scale
            and residual is not None
            and np.isfinite(residual)
            and residual <= residual_tolerance
            and finite_fields
        ):
            # A real hard-wall operator has real beta.  Remove only accepted
            # roundoff-scale Im(n_eff); retaining it as signed gain/loss metadata
            # would make otherwise lossless EME propagation non-passive.
            stable.append(replace(mode, k_eff=0.0))
    return stable


def _yee_window_structure_indices(
    sim, axis, plane_value_um, geom: "_WindowGeom", supersample: int
):
    """Structures sampled by this exact uniform Yee/KFJ window."""
    from .kfj_smoothing import _paint_indices

    active = set()
    for off_h, off_v in ((0.5, 0.0), (0.0, 0.5), (0.0, 0.0)):
        fine_h, fine_v = _fine_centers(
            geom.h_lo,
            geom.v_lo,
            geom.nh,
            geom.nv,
            geom.dl,
            off_h,
            off_v,
            supersample,
        )
        painted = _paint_indices(
            sim, axis, plane_value_um, fine_h, fine_v
        )
        active.update(int(i) for i in np.unique(painted) if i >= 0)
    return active


def _reject_lossy_yee_eme_materials(
    sim, eps_of_medium, active_indices
) -> None:
    """Reject unsupported media sampled by this real hard-wall window."""
    active_media_ids = {
        id(sim.structures[index].medium) for index in active_indices
    }
    if eps_of_medium is not None:
        for medium_id, value in eps_of_medium.items():
            if medium_id not in active_media_ids:
                continue
            anchored = complex(value)
            if not np.isfinite(anchored.real) or not np.isfinite(anchored.imag):
                raise ValueError("eps_of_medium values must be finite")
            if anchored.imag != 0.0:
                raise ValueError(
                    "solve_yee_eme_basis requires real, lossless epsilon; "
                    "complex eps_of_medium anchors are not supported"
                )
    for index in active_indices:
        structure = sim.structures[index]
        medium = structure.medium
        if (
            getattr(medium, "is_anisotropic", False)
            or getattr(medium, "is_custom", False)
            or bool(getattr(medium, "pec", False))
        ):
            raise ValueError(
                "solve_yee_eme_basis supports only finite scalar dielectric "
                "media; anisotropic, custom-data, and PEC media are not "
                "supported"
            )
        lossy = float(getattr(medium, "conductivity_s_per_m", 0.0)) > 0.0
        poles = tuple(getattr(medium, "all_lorentz_poles", lambda: ())())
        poles += tuple(getattr(medium, "drude", None) or ())
        lossy = lossy or any(float(p.linewidth_hz) > 0.0 for p in poles)
        if lossy:
            raise ValueError(
                "solve_yee_eme_basis requires a real, lossless cross-section; "
                "conductive or damped-dispersive media are not supported"
            )


def _validate_yee_reaction_basis(
    modes, dl_um: float, reaction_tolerance: float
) -> None:
    """Reject a numerically valid right spectrum that is unsafe for EME.

    A tiny-beta pair can have an excellent ordinary eigen residual while its
    self reaction tends to zero and the reaction Gram becomes singular.  EME
    matching uses that Gram, so validate the actual Yee reaction traces before
    exposing the basis rather than allowing a later interface solve to amplify
    the exceptional shell catastrophically.
    """
    from .eme import _basis_trace

    if any(mode.overlap_weights is not None for mode in modes):
        raise RuntimeError(
            "solve_yee_eme_basis supports only a real hard-wall contour; "
            "complex overlap_weights/PML modes are not accepted"
        )
    try:
        trace = _basis_trace(modes, dl_um, dl_um, rcond=1e-10)
    except ValueError as exc:
        raise RuntimeError(
            "Yee EME basis is reaction-unstable: "
            f"{exc}. Request fewer modes or enlarge the transverse window."
        ) from exc

    condition_limit = 1.0 / reaction_tolerance
    unstable = (
        not np.isfinite(trace.gram_condition)
        or trace.gram_condition > condition_limit
        or not np.isfinite(trace.gram_asymmetry)
        or trace.gram_asymmetry > reaction_tolerance
        or not np.isfinite(trace.reaction_orthogonality_error)
        or trace.reaction_orthogonality_error > reaction_tolerance
    )
    if unstable:
        raise RuntimeError(
            "Yee EME basis is reaction-unstable "
            f"(Gram condition={trace.gram_condition:.3g}, "
            f"asymmetry={trace.gram_asymmetry:.3g}, nondegenerate "
            f"off-diagonal={trace.reaction_orthogonality_error:.3g}; "
            f"limits={condition_limit:.3g}, {reaction_tolerance:.3g}, "
            f"{reaction_tolerance:.3g}). Request fewer modes or enlarge the "
            "transverse window."
        )


@legacy_keywords(wavelength_um="wlen_um")
def solve_yee_mode(sim, axis: str, plane_value_um: float, wlen_um: float,
                   pol: str, mode_index: int, *, h_center_um: float,
                   v_center_um: float, half_w_um: float, half_v_um: float,
                   dl_um: float, supersample: int = 8, num_modes: Optional[int] = None,
                   eps_of_medium: Optional[Mapping[int, float]] = None) -> VectorMode:
    """Solve the engine-consistent Yee-grid discrete eigenmode and return it as a
    :class:`VectorMode` (mode-frame [iy=height, ix=width]). Dispersive media are
    anchored at the solve frequency (:meth:`Medium.permittivity_at_hz`), matching
    the eps the engine's ADE realizes there, NOT the eps_inf that bare
    ``permittivity`` carries; ``eps_of_medium`` overrides per medium.

    §20 symmetry planes are honored AUTOMATICALLY: when ``sim.symmetry`` puts a
    plane on an in-plane axis' min face and the window reaches it, the window
    is clipped at the plane and the matching parity BC applied (PEC -1 / PMC
    +1, :func:`window_min_face_bcs`), the half-window mode is the engine's
    half-domain field, and ``mode_index`` counts within the matching-parity
    family only.

    A plain PERIODIC in-plane axis is honored the same way: the window is the
    whole period with the k = 0 periodic closure instead of walls, whatever
    half-extent was asked for (:func:`_periodic_window`). On a one-cell axis (the
    quasi-2D reduction) the window is that single cell and the solve returns
    the cross-section's SLAB modes, so a ``Port`` on a quasi-2D device
    resolves. There ``"TE"`` is the family with E along the periodic axis
    (``te_fraction`` ~1)."""
    sample, geom = staggered_eps_sampler(
        sim, axis, plane_value_um, h_center=h_center_um, v_center=v_center_um,
        half_w=half_w_um, half_v=half_v_um, dl=dl_um, supersample=supersample,
        eps_of_medium=eps_of_medium)
    exx, eyy, ezz = sample(C0 / (wlen_um * 1e-6))
    off, xc, yc = _window_placement(geom, h_center_um, v_center_um)
    nmodes = num_modes or max(6, mode_index + 3)
    modes = _solve_yee_eig(exx, eyy, ezz, geom.nh, geom.nv, wlen_um,
                           dl_um, nmodes, center_offset=off,
                           h_min_bc=geom.h_bc, v_min_bc=geom.v_bc,
                           dq_h=geom.h_dq, dq_v=geom.v_dq,
                           x_coords_um=xc, y_coords_um=yc)
    mode = _pick_yee(modes, pol, mode_index, axis, plane_value_um)
    _warn_tight_window(mode, geom, axis, plane_value_um)
    return mode


@legacy_keywords(wavelength_um="wlen_um")
def solve_yee_eme_basis(
    sim,
    axis: str,
    plane_value_um: float,
    wlen_um: float,
    num_modes: Optional[int] = None,
    *,
    h_center_um: float,
    v_center_um: float,
    half_w_um: float,
    half_v_um: float,
    dl_um: float,
    supersample: int = 8,
    eps_of_medium: Optional[Mapping[int, float]] = None,
    residual_tolerance: float = 1e-7,
    reaction_tolerance: float = 1e-7,
    neff_cutoff: Optional[float] = None,
    max_modes: int = 64,
) -> Tuple[VectorMode, ...]:
    """Return an experimental propagating Yee hard-wall basis for EME.

    The solve uses the same per-component KFJ permittivity and staggered curl
    operator as :func:`solve_yee_mode`, but retains both bound modes and the
    positive-beta hard-wall discretization of the radiation continuum.  Modes
    are ordered by decreasing real ``n_eff`` and labelled ``"guided"`` above
    the homogeneous exterior light line, otherwise ``"radiation"``.  Every
    returned mode is lossless, ``yee_staggered=True``, and safe under the
    unconjugated reaction product used by EME.

    .. warning::
       This is a propagating-only basis: it excludes ``beta≈0`` and
       pure-imaginary-beta roots because the current Yee field reconstruction
       divides by real beta.  It therefore contains guided and box-radiation
       modes, not the evanescent channels required for a complete EME basis.
       Passing the reaction guard is a numerical precondition, not evidence of
       quantitative device-radiation accuracy.

    This first public continuum path deliberately supports only a uniform
    transverse lattice and a homogeneous scalar exterior.  It has no modal PML:
    the requested window is the discretization box, so radiation modes are box
    modes whose quantitative use requires independent window and complete-mode-group
    convergence sweeps.  Holding ``num_modes`` fixed while changing the window
    is not a valid continuum control because it changes the represented beta
    band and modal density.  For a window sweep, omit ``num_modes`` and pass
    ``neff_cutoff``: the solver returns every complete propagating mode group above
    that spectral cutoff, subject to the explicit ``max_modes`` safety cap.
    """
    count_request = num_modes is not None
    cutoff_request = neff_cutoff is not None
    if count_request == cutoff_request:
        raise ValueError(
            "provide exactly one of num_modes (fixed-count solve) or "
            "neff_cutoff (complete-shell spectral solve)"
        )
    requested = None
    if count_request:
        try:
            count_is_finite = bool(np.isfinite(num_modes))
        except (TypeError, ValueError):
            count_is_finite = False
        if (
            isinstance(num_modes, (bool, np.bool_))
            or not count_is_finite
            or int(num_modes) != num_modes
        ):
            raise ValueError(
                f"num_modes must be an integer >= 1, got {num_modes!r}"
            )
        requested = int(num_modes)
        if requested < 1:
            raise ValueError(f"num_modes must be >= 1, got {requested}")
    else:
        if not np.isfinite(neff_cutoff) or float(neff_cutoff) <= 1e-6:
            raise ValueError("neff_cutoff must be finite and > 1e-6")
        neff_cutoff = float(neff_cutoff)
        try:
            cap_is_finite = bool(np.isfinite(max_modes))
        except (TypeError, ValueError):
            cap_is_finite = False
        if (
            isinstance(max_modes, (bool, np.bool_))
            or not cap_is_finite
            or int(max_modes) != max_modes
            or int(max_modes) < 1
        ):
            raise ValueError("max_modes must be an integer >= 1")
        max_modes = int(max_modes)
    if not np.isfinite(wlen_um) or wlen_um <= 0.0:
        raise ValueError(f"wlen_um must be finite and > 0, got {wlen_um}")
    if not np.isfinite(dl_um) or dl_um <= 0.0:
        raise ValueError(f"dl_um must be finite and > 0, got {dl_um}")
    if not np.isfinite(plane_value_um):
        raise ValueError(
            f"plane_value_um must be finite, got {plane_value_um}"
        )
    for name, value in (
        ("h_center_um", h_center_um),
        ("v_center_um", v_center_um),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    for name, value in (
        ("half_w_um", half_w_um),
        ("half_v_um", half_v_um),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    if (
        isinstance(supersample, (bool, np.bool_))
        or int(supersample) != supersample
        or int(supersample) < 1
    ):
        raise ValueError("supersample must be an integer >= 1")
    supersample = int(supersample)
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and > 0")
    if (
        not np.isfinite(reaction_tolerance)
        or reaction_tolerance <= 0.0
        or reaction_tolerance >= 1.0
    ):
        raise ValueError("reaction_tolerance must be finite and between 0 and 1")

    sample, geom = staggered_eps_sampler(
        sim,
        axis,
        plane_value_um,
        h_center=h_center_um,
        v_center=v_center_um,
        half_w=half_w_um,
        half_v=half_v_um,
        dl=dl_um,
        supersample=supersample,
        eps_of_medium=eps_of_medium,
    )
    if "periodic" in (geom.h_bc, geom.v_bc):
        raise ValueError(
            "solve_yee_eme_basis requires a full transverse hard-wall box; a "
            "plain periodic transverse axis is solved as its whole period with "
            "the periodic closure, which EME interface matching does not "
            "support (give that axis a non-periodic boundary)"
        )
    if geom.graded:
        raise ValueError(
            "solve_yee_eme_basis requires a uniform transverse grid; graded "
            "Yee coordinates are not supported by EME interface matching"
        )
    if geom.h_bc is not None or geom.v_bc is not None:
        raise ValueError(
            "solve_yee_eme_basis requires a full transverse hard-wall box; "
            "symmetry-reduced Yee windows need reaction-mass multiplicity "
            "weights that are not implemented"
        )
    sim_dl = float(getattr(sim.grid, "dl_um", dl_um))
    if not np.isclose(dl_um, sim_dl, rtol=1e-12, atol=1e-15):
        raise ValueError(
            "solve_yee_eme_basis must use the Simulation's uniform Yee "
            f"spacing ({sim_dl:.9g} um), got dl_um={dl_um:.9g}"
        )
    active_indices = _yee_window_structure_indices(
        sim, axis, plane_value_um, geom, supersample
    )
    _reject_lossy_yee_eme_materials(
        sim, eps_of_medium, active_indices
    )

    frequency_hz = C0 / (wlen_um * 1e-6)
    exx, eyy, ezz = sample(frequency_hz)
    exterior_eps = _homogeneous_yee_exterior(exx, eyy, ezz, geom)
    exterior_index = float(np.sqrt(exterior_eps))
    off, xc, yc = _window_placement(geom, h_center_um, v_center_um)

    unknowns = 2 * geom.nh * geom.nv
    max_trial = unknowns - 2
    if requested is not None and requested > max_trial:
        raise ValueError(
            f"num_modes={requested} is too large for the {geom.nh} x "
            f"{geom.nv} Yee box ({unknowns} transverse unknowns); request "
            "fewer modes or enlarge the transverse window"
        )
    common_solve_kwargs = {
        "center_offset": off,
        "h_min_bc": geom.h_bc,
        "v_min_bc": geom.v_bc,
        "x_coords_um": xc,
        "y_coords_um": yc,
        "min_neff": 1e-6,
    }
    stable = []
    if requested is not None:
        # Oversample the Ritz spectrum because positive-beta filtering can
        # discard below-cutoff pairs.  A bounded retry prevents a large
        # impossible request from silently returning a truncated basis.
        trial = min(max_trial, max(6, requested + max(4, requested // 2)))
        for _ in range(3):
            modes = _solve_yee_eig(
                exx, eyy, ezz, geom.nh, geom.nv, wlen_um, dl_um,
                trial, **common_solve_kwargs,
            )
            stable = _stable_positive_yee_modes(modes, residual_tolerance)
            if len(stable) >= requested:
                break
            larger = min(max_trial, max(trial + 4, 2 * trial))
            if larger == trial:
                break
            trial = larger
        if len(stable) < requested:
            raise RuntimeError(
                "Yee EME basis spectrum incomplete: requested "
                f"{requested} positive-beta stable modes but found "
                f"{len(stable)} in the {geom.nh} x {geom.nv} hard-wall box. "
                "Request fewer modes or enlarge the transverse window."
            )

        selected = stable[:requested]
        # Do not expose an arbitrary truncation of an (approximately)
        # degenerate polarization/spatial multiplet.  Such a basis changes
        # under harmless ARPACK rotations.
        if len(stable) > requested:
            edge = selected[-1].n_eff
            following = stable[requested].n_eff
            if abs(edge - following) <= 1e-6 * max(
                abs(edge), abs(following), 1.0
            ):
                raise RuntimeError(
                    f"num_modes={requested} splits a degenerate Yee beta "
                    f"shell near n_eff={edge:.9g}; include the complete "
                    "multiplet or request fewer modes"
                )
    else:
        # A spectral cutoff is a defensible window-sweep control only after the
        # eigensolve has crossed it.  Continue through TWO distinct shells below
        # the threshold: the first below-cutoff shell is then internal to the
        # returned Ritz spectrum rather than a possibly truncated final shell.
        assert neff_cutoff is not None
        lookahead = max(8, max_modes // 4)
        trial_limit = min(max_trial, max_modes + lookahead)
        trial = min(trial_limit, max(8, min(16, max_modes + 2)))
        selected = []
        bracketed = False
        while True:
            modes = _solve_yee_eig(
                exx, eyy, ezz, geom.nh, geom.nv, wlen_um, dl_um,
                trial, **common_solve_kwargs,
            )
            stable = _stable_positive_yee_modes(modes, residual_tolerance)
            shell_tol = 1e-6 * max(abs(neff_cutoff), 1.0)
            if any(abs(mode.n_eff - neff_cutoff) <= shell_tol for mode in stable):
                raise RuntimeError(
                    f"neff_cutoff={neff_cutoff:.9g} intersects a Yee beta "
                    "shell; move the cutoff between adjacent shells"
                )
            raw_above = [mode for mode in modes if mode.n_eff > neff_cutoff]
            selected = [mode for mode in stable if mode.n_eff > neff_cutoff]
            if len(raw_above) != len(selected):
                raise RuntimeError(
                    "Yee EME spectrum contains a non-real, non-finite, or "
                    "high-residual mode above neff_cutoff; move the cutoff up, "
                    "request fewer modes, or enlarge the transverse window"
                )
            if len(selected) > max_modes:
                raise RuntimeError(
                    f"neff_cutoff={neff_cutoff:.9g} selects more than "
                    f"max_modes={max_modes}; raise max_modes or move the "
                    "cutoff up"
                )
            below = [
                mode.n_eff for mode in stable
                if mode.n_eff < neff_cutoff - shell_tol
            ]
            distinct_below = []
            for beta in below:
                if not distinct_below or abs(beta - distinct_below[-1]) > (
                    1e-6 * max(abs(beta), abs(distinct_below[-1]), 1.0)
                ):
                    distinct_below.append(beta)
            if len(distinct_below) >= 2:
                bracketed = True
                break
            if trial == trial_limit:
                break
            trial = min(trial_limit, max(trial + 4, 2 * trial))
        if not selected:
            raise RuntimeError(
                f"neff_cutoff={neff_cutoff:.9g} selects no stable "
                "positive-beta modes in this hard-wall box"
            )
        if not bracketed:
            raise RuntimeError(
                f"could not resolve two complete Yee beta shells below "
                f"neff_cutoff={neff_cutoff:.9g} within max_modes={max_modes}; "
                "raise max_modes, move the cutoff up, or enlarge the "
                "transverse window"
            )

    light_line = exterior_index * (1.0 + 1e-6)
    selected = [
        replace(
            mode,
            mode_type="guided" if mode.n_eff > light_line else "radiation",
        )
        for mode in selected
    ]
    _validate_yee_reaction_basis(selected, dl_um, reaction_tolerance)
    return tuple(
        replace(
            mode,
            yee_eme_compatible=True,
            yee_eme_axis=axis,
            yee_eme_origin_um=(geom.h_lo, geom.v_lo),
        )
        for mode in selected
    )


def solve_yee_mode_bank(sim, axis: str, plane_value_um: float, freqs_hz, pol: str,
                        mode_index: int, *, h_center_um: float, v_center_um: float,
                        half_w_um: float, half_v_um: float, dl_um: float,
                        supersample: int = 8, num_modes: Optional[int] = None,
                        eps_of_medium: Optional[Mapping[int, float]] = None):
    """``{freq_hz: VectorMode}`` per-frequency Yee-grid readout mode mapping, the engine-
    consistent analogue of
    :func:`~photonhub.analysis.kfj_smoothing.mode_bank_on_cross_section` (which uses the
    node-collocated FLM ``VectorModeSolver``). The window geometry is sampled on the mesh
    ONCE; a non-dispersive cross-section shares one Yee-staggered ε for every
    frequency (λ-independent at constant n), while dispersive media are re-anchored
    at each frequency (:meth:`Medium.permittivity_at_hz`), then the discrete-Yee
    eigenproblem is re-solved per frequency, so the readout reference mode matches
    the FDTD field's discretization at every λ, the same discrete operator the
    launch used (:func:`solve_yee_mode`).

    Each solve fixes its mode's phase so the dominant transverse E is real and
    positive at its peak; for a two-lobe mode (TE1, TM1) the winning lobe, and
    so the sign, changes with frequency. The mode mapping is therefore phase-aligned
    along the band from its lowest frequency, whose mode keeps its own phase
    (the reference :func:`solve_yee_port_mode_bank` uses too): the complex
    amplitude read against it, and every S-parameter phase, is continuous
    across the band instead of jumping by pi. Neighbouring picks that overlap
    below 0.5 warn."""
    from .mode_tracking import _follow_bank

    nmodes = num_modes or max(6, mode_index + 3)
    freqs = sorted({float(f) for f in freqs_hz})
    picked = [_pick_yee(modes, pol, mode_index, axis, plane_value_um)
              for _, modes in _yee_bank_frames(
                  sim, axis, plane_value_um, freqs, nmodes=nmodes,
                  h_center_um=h_center_um, v_center_um=v_center_um,
                  half_w_um=half_w_um, half_v_um=half_v_um, dl_um=dl_um,
                  supersample=supersample, eps_of_medium=eps_of_medium)]
    aligned = _follow_bank([[m] for m in picked], [0], freqs, anchor=0, min_overlap=0.5,
                           names={0: f"{str(pol).upper()}{int(mode_index)}"})
    by_f = {f: frame[0] for f, frame in zip(freqs, aligned)}
    return {float(f): by_f[float(f)] for f in freqs_hz}


def _yee_bank_frames(sim, axis, plane_value_um, freqs_hz, *, nmodes,
                     h_center_um, v_center_um, half_w_um, half_v_um, dl_um,
                     supersample, eps_of_medium):
    """The shared per-frequency solve loop of the Yee banks: yields
    ``(freq_hz, guided-modes-descending)`` per bank frequency. Geometry is
    sampled on the mesh once by :func:`staggered_eps_sampler`, which also decides
    whether eps repeats per frequency (un-overridden dispersive media) or is
    computed once and cached (everything else)."""
    sample, geom = staggered_eps_sampler(
        sim, axis, plane_value_um, h_center=h_center_um, v_center=v_center_um,
        half_w=half_w_um, half_v=half_v_um, dl=dl_um, supersample=supersample,
        eps_of_medium=eps_of_medium)
    off, xc, yc = _window_placement(geom, h_center_um, v_center_um)
    for f in freqs_hz:
        ff = float(f)
        exx, eyy, ezz = sample(ff)
        yield ff, _solve_yee_eig(exx, eyy, ezz, geom.nh, geom.nv,
                                 C0 / ff * 1e6, dl_um, nmodes,
                                 center_offset=off,
                                 h_min_bc=geom.h_bc, v_min_bc=geom.v_bc,
                                 dq_h=geom.h_dq, dq_v=geom.v_dq,
                                 x_coords_um=xc, y_coords_um=yc)


def solve_yee_multimode_bank(sim, axis: str, plane_value_um: float, freqs_hz, *,
                             mode_indices=(0,), h_center_um: float,
                             v_center_um: float, half_w_um: float,
                             half_v_um: float, dl_um: float,
                             supersample: int = 8,
                             num_modes: Optional[int] = None,
                             eps_of_medium: Optional[Mapping[int, float]] = None):
    """``{freq_hz: {mode_index: VectorMode}}`` MULTI-mode per-frequency Yee mode mapping, the engine-consistent analogue of
    :func:`~photonhub.analysis.mode_devices.solve_mode_bank` (which needs an FLM/scalar
    solver object), ready for :meth:`ModeMonitor.mode_decomposition`.

    Indexing follows ``solve_mode_bank``'s convention: ``mode_indices`` count the
    guided modes in descending-``n_eff`` order ACROSS polarizations (0 = the
    fundamental), NOT the per-polarization ``(pol, mode_index)`` selection of
    :func:`solve_yee_mode_bank`. Geometry is sampled on the mesh once; dispersive media are
    re-anchored per frequency (:meth:`Medium.permittivity_at_hz`).

    The indices are counted at the band's middle frequency, and each mode is
    followed across the band by field overlap with its neighbouring frequency
    and phase-aligned to it: a mode keeps its index through a crossing in
    ``n_eff`` where the solver's own order swaps. Where two modes mix near a
    crossing (common for TE1 and TM0 on a Yee window), no index names one
    physical mode across it; the mode mapping warns when a link's overlap falls below
    0.9 or a mode's TE fraction changes by more than 0.5 over the band."""
    freqs = [float(f) for f in freqs_hz]
    if not freqs:
        raise ValueError("freqs_hz must be non-empty")
    idxs = sorted({int(i) for i in mode_indices})
    if not idxs:
        raise ValueError("mode_indices must be non-empty")
    if idxs[0] < 0:
        raise ValueError(f"mode_indices must be >= 0, got {idxs[0]}")
    # Ask for a few eigenpairs beyond the highest requested index — the guided
    # filter (n_eff > 1) of _solve_yee_eig may drop some of the k pairs.
    nmodes = max(int(num_modes or 0), idxs[-1] + 3, 6)
    from .mode_tracking import _follow_bank

    band = sorted(set(freqs))
    frames = []
    for f, modes in _yee_bank_frames(
            sim, axis, plane_value_um, band, nmodes=nmodes,
            h_center_um=h_center_um, v_center_um=v_center_um,
            half_w_um=half_w_um, half_v_um=half_v_um, dl_um=dl_um,
            supersample=supersample, eps_of_medium=eps_of_medium):
        if idxs[-1] >= len(modes):
            raise ValueError(
                f"requested mode_index {idxs[-1]} but the Yee solve found only "
                f"{len(modes)} guided mode(s) at {f:.4g} Hz "
                f"({C0 / f * 1e6:.4f} um) — the waveguide may not support it "
                "across the whole band")
        frames.append(modes)
    followed = _follow_bank(frames, idxs, band, what="guided mode")
    by_f = {f: {i: frame[i] for i in idxs} for f, frame in zip(band, followed)}
    return {f: by_f[f] for f in freqs}


def solve_yee_port_mode_bank(sim, axis: str, plane_value_um: float, freqs_hz, *,
                             modes=(("TE", 0),), h_center_um: float,
                             v_center_um: float, half_w_um: float,
                             half_v_um: float, dl_um: float,
                             supersample: int = 8,
                             num_modes: Optional[int] = None,
                             thickness_axis: Optional[str] = None,
                             eps_of_medium: Optional[Mapping[int, float]] = None):
    """Solve several polarization-family modes with one eigensolve per frequency.

    Returns ``{freq_hz: {(polarization, mode_index): VectorMode}}``.  Unlike
    :func:`solve_yee_multimode_bank`, each ``mode_index`` is counted *within*
    its requested TE/TM family.  This is the identity exposed by a modal-port
    editor (``TE0``, ``TE1``, ``TM0``) and the one used by
    :func:`solve_yee_mode_bank` for a single channel.

    The cross-section is sampled on the mesh once and every requested family/index is
    selected from the same guided-mode frame at each frequency.  That avoids
    repeating the sparse Yee eigensolve when one recorded plane is decomposed
    into, for example, both TE0 and TE1.
    """
    freqs = [float(f) for f in freqs_hz]
    if not freqs:
        raise ValueError("freqs_hz must be non-empty")

    natural_axes = _geom.in_plane_axes(axis)
    if thickness_axis is None:
        thickness_axis = natural_axes[1]
    if thickness_axis not in natural_axes:
        raise ValueError(
            f"thickness_axis {thickness_axis!r} must be transverse to {axis!r}")
    requested = []
    seen = set()
    for item in modes:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "modes entries must be (polarization, mode_index) pairs")
        polarization, raw_index = item
        polarization = str(polarization).upper()
        if polarization not in ("TE", "TM"):
            raise ValueError(
                f"mode polarization must be TE or TM, got {polarization!r}")
        if isinstance(raw_index, bool) or not isinstance(raw_index, (int, np.integer)):
            raise ValueError(f"mode index must be an integer, got {raw_index!r}")
        mode_index = int(raw_index)
        if not 0 <= mode_index <= 31:
            raise ValueError(
                f"mode indices must be between 0 and 31, got {mode_index}")
        key = (polarization, mode_index)
        if key not in seen:
            requested.append(key)
            seen.add(key)
    if not requested:
        raise ValueError("modes must be non-empty")

    # Match solve_yee_mode_bank's established search posture.  A caller may
    # deliberately request more trial eigenpairs for a weakly guided/high-order
    # family; an unavailable family/index still fails through _pick_yee with a
    # diagnostic listing the modes that were found. The shared resolver also
    # accounts for indices being family-relative while ``nmodes`` is the total
    # eigensolver frame size.
    nmodes = mode_port_trial_modes(requested, num_modes)

    def solver_family(key):
        return mode_port_solver_polarization(
            key[0], axis, thickness_axis)

    def continuity_score(previous, candidate) -> float:
        if isinstance(previous, VectorMode) and isinstance(candidate, VectorMode):
            overlap = np.vdot(previous.ex, candidate.ex) + np.vdot(
                previous.ey, candidate.ey)
            previous_norm = np.sqrt(
                np.vdot(previous.ex, previous.ex).real
                + np.vdot(previous.ey, previous.ey).real)
            candidate_norm = np.sqrt(
                np.vdot(candidate.ex, candidate.ex).real
                + np.vdot(candidate.ey, candidate.ey).real)
            if previous_norm > 0.0 and candidate_norm > 0.0:
                return float(abs(overlap) / (previous_norm * candidate_norm))
        return 1.0 / (1.0 + abs(
            float(previous.n_eff) - float(candidate.n_eff)))

    def phase_align(previous, candidate):
        if not isinstance(previous, VectorMode) or not isinstance(candidate, VectorMode):
            return candidate
        overlap = np.vdot(previous.ex, candidate.ex) + np.vdot(
            previous.ey, candidate.ey)
        if abs(overlap) == 0.0:
            return candidate
        factor = np.conj(overlap) / abs(overlap)
        return replace(
            candidate,
            ex=candidate.ex * factor, ey=candidate.ey * factor,
            ez=candidate.ez * factor, hx=candidate.hx * factor,
            hy=candidate.hy * factor, hz=candidate.hz * factor,
        )

    out = {}
    previous = None
    for f, frame in _yee_bank_frames(
            sim, axis, plane_value_um, sorted(freqs), nmodes=nmodes,
            h_center_um=h_center_um, v_center_um=v_center_um,
            half_w_um=half_w_um, half_v_um=half_v_um, dl_um=dl_um,
            supersample=supersample, eps_of_medium=eps_of_medium):
        if previous is None:
            selected = {
                key: _pick_yee(
                    frame, solver_family(key), key[1], axis, plane_value_um)
                for key in requested
            }
        else:
            if len(frame) < len(requested):
                raise RuntimeError(
                    f"yee mode solve at {axis}={plane_value_um:.3f} found "
                    f"only {len(frame)} guided modes for {len(requested)} "
                    "tracked modal-port channels")
            from scipy.optimize import linear_sum_assignment

            scores = np.asarray([
                [continuity_score(previous[key], candidate)
                 for candidate in frame]
                for key in requested
            ], dtype=np.float64)
            rows, columns = linear_sum_assignment(-scores)
            assignment = dict(zip(rows.tolist(), columns.tolist()))
            selected = {
                key: phase_align(previous[key], frame[assignment[index]])
                for index, key in enumerate(requested)
            }
        out[f] = selected
        previous = selected
    return out
