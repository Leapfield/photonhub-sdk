"""Yee-grid FDFD waveguide mode solver — the **engine-consistent** discrete mode.

ph's Fallahkhair-Li-Murphy solver (`vector_modes.py`) puts the transverse H at grid
nodes and eps at the four surrounding quadrants — a different discretization from
the engine's FDTD (Yee-staggered E/H, subpixel eps placed per E-component). The
mode it finds therefore is NOT the one the grid propagates, so injecting it radiates
the difference (~2.4% near-source shedding).

This module solves the mode on the **engine's own Yee discretization**:
  * fields on the standard Yee locations (Ex@(i+1/2,j), Ey@(i,j+1/2), Ez@(i,j)),
  * forward/backward staggered curls matching engine/src/kernels/update_body.h,
  * the diagonal KFJ subpixel eps sampled PER-COMPONENT at its own Yee location
    (eps_xx at Ex, eps_yy at Ey, eps_zz at Ez) — exactly as the engine rasterizes
    (engine/src/cpu_ref/reference_solver.cpp sample_voxel comp_axis).

The eigenproblem is the canonical transverse-E full-vector FDFD (diagonal eps,
mu=1): ``mat @ [Ex;Ey] = -n_eff^2 [Ex;Ey]`` with the block operator built from the
forward/backward derivative matrices (standard formulation; here wired to the
engine's curls + staggered eps). The launched mode is then the FDTD discrete mode,
so a TF/SF injection of it is clean (Tidy3D matches its FDTD the same way).

KNOWN COMPROMISE — consumers COLLOCATE the staggered components. The returned
:class:`VectorMode` carries one field array per component on a single index
grid; downstream (``vector_modal_fields`` and everything built on it) assigns
ALL components the same node coordinates ``lo + i*dl`` (+ the carried
``center_offset_um``), discarding the intra-cell Yee stagger this solver
faithfully used (Ex at +1/2 in h, Ey at +1/2 in v, Ez at the node) — a
per-component error of up to half a cell, recorded in the GDS-benchmark
findings. Do not assume the arrays keep their Yee offsets downstream.
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
from .vector_modes import VectorMode


# --------------------------------------------------------------------------- #
# Yee forward/backward derivative matrices (match engine update_body.h).
# --------------------------------------------------------------------------- #
def dual_spacings(dq: np.ndarray) -> np.ndarray:
    """Dual-grid steps for the backward (H-curl) derivatives from the primal
    steps ``dq`` (§15.2): interior dual width = midpoint average of the two
    adjacent primal cells; the first entry keeps the first primal width (the
    same convention Tidy3D's mode solver uses — ``dl_b[0] = dl_f[0]`` — so the
    §20 face rules read the face row over one whole cell). Uniform input
    reproduces the constant spacing exactly."""
    dq = np.asarray(dq, dtype=float)
    dual = np.empty_like(dq)
    dual[0] = dq[0]
    dual[1:] = 0.5 * (dq[:-1] + dq[1:])
    return dual


def _dmats(nx: int, ny: int, dl: float, h_min_bc=None, v_min_bc=None,
           dq_h=None, dq_v=None):
    """Forward/backward x,y difference matrices on the nx*ny Yee grid (row-major
    [ix*ny + iy]). ``dxf`` maps a node field to the +1/2 face (forward, = engine
    H-curl-of-E direction); ``dxb`` is its backward adjoint (= engine
    E-curl-of-H direction).

    Spacing: with ``dq_h``/``dq_v`` None (uniform), every row is scaled by the
    single ``1/dl`` — byte-identical to the historical operator. A GRADED axis
    passes its PRIMAL spacing vector ``dq`` (length n, §15.1 replicate-last):
    forward rows divide by the primal widths (node i -> i+1 distance), backward
    rows by the DUAL widths (:func:`dual_spacings`) — the standard nonuniform
    Yee FDFD (Zhu & Brown; the same construction Tidy3D's open mode solver
    uses, ``diags(1/dls) @ D``). The scalar path is kept verbatim rather than
    expressed as a constant vector so uniform grids stay bit-identical
    (reciprocal-multiply vs divide differ in ULPs).

    Low-edge boundary (``h_min_bc``/``v_min_bc``): ``None`` keeps the legacy
    implicit ghost — the half-located quantities that ``bwd`` acts on (eps*E_n
    and the tangential H) are taken as 0 half a cell outside, a MAGNETIC (PMC)
    mirror OUTSIDE the window; immaterial when the edge sits in decayed
    cladding. ``"pmc"`` / ``"pec"`` instead put the engine's §20 symmetry plane
    exactly ON the low node line: PMC takes the odd ghost X[-1] = -X[0], so
    bwd's first row reads 2*X[0]/dl — the engine's §20.4 backward-read rule;
    PEC takes the even ghost, zeroing that row (the caller must ALSO pin the
    on-plane tangential-E DOFs — see :func:`_solve_yee_eig`). The high edge
    stays the implicit node-line PEC in all cases."""
    import scipy.sparse as sp

    def fwd(n, dq):
        # PEC at the high edge is already implicit in the diags construction:
        # its last row is just -1 on the diagonal (the +1 superdiagonal entry
        # falls outside the matrix), i.e. the field is taken as 0 outside the
        # window — verified identical to the explicit lil-matrix edge
        # assignment this replaces.
        m = sp.diags([-1.0, 1.0], [0, 1], shape=(n, n), format="csr")
        if dq is None:
            return m / dl
        return sp.diags(1.0 / np.asarray(dq, dtype=float)) @ m

    def bwd(n, bc, dq):
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
    dxf = sp.kron(fwd(nx, dq_h), Iy, format="csr")
    dxb = sp.kron(bwd(nx, h_min_bc, dq_h), Iy, format="csr")
    dyf = sp.kron(Ix, fwd(ny, dq_v), format="csr")
    dyb = sp.kron(Ix, bwd(ny, v_min_bc, dq_v), format="csr")
    return dxf, dxb, dyf, dyb


def min_face_symmetry_bcs(sim, axis):
    """The §20 symmetry parity of each in-plane axis' MIN face for a
    cross-section normal to ``axis``, as ``(h_bc, v_bc)`` in :func:`_dmats`
    terms: ``"pec"`` (-1, odd/electric), ``"pmc"`` (+1, even/magnetic), or
    ``None``. Read from ``Simulation.symmetry``; objects without the field
    (duck-typed sims) get ``(None, None)``. NOTE: this is the plane's
    EXISTENCE — whether it applies to a given mode window also requires the
    window's low edge to sit ON the plane (see :func:`window_min_face_bcs`)."""
    sym = getattr(sim, "symmetry", None) or (0, 0, 0)
    h_letter, v_letter = _geom.in_plane_axes(axis)
    m = {-1: "pec", 0: None, 1: "pmc"}
    return (m[int(sym["xyz".index(h_letter)])],
            m[int(sym["xyz".index(v_letter)])])


def window_min_face_bcs(sim, axis, *, h_center, half_w, v_center, half_v, dl):
    """The shared window-registration + symmetry rule for every consumer of a
    cross-section window (the eps sampler, the eigensolve, and the
    equivalence-current sheet MUST agree bit-for-bit on this). Returns
    ``(h_lo, v_lo, h_bc, v_bc)``:

    - ``lo`` = the grid-snapped window origin ``floor((center-half)/dl)*dl``,
      CLIPPED to 0 on any in-plane axis carrying a §20 symmetry plane (the
      plane sits on the domain min face at coordinate 0; the below-plane half
      of a requested window is the mirror image the boundary supplies — it
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


def _axis_window_nodes(sim, axis_letter, center, half, dl, bc):
    """Window node ladder for ONE in-plane axis: ``(nodes, dq, bc_eff)``.

    ``nodes`` are the sim's PRIMARY node coordinates covering
    ``[center-half, center+half]`` (a node at each cell's low edge; the §15.1
    replicate-last primal widths in ``dq``), ``bc_eff`` the §20 parity when the
    window's first node sits ON the min-face plane. A UNIFORM axis reproduces
    :func:`window_min_face_bcs`' floor-snap ladder with the IDENTICAL floats
    (nodes = h_lo + arange(n)*dl) and returns ``dq=None`` — the marker every
    downstream consumer uses to take its legacy scalar-dl fast path, keeping
    uniform grids bit-identical."""
    q = sim._axis_coords_um("xyz".index(axis_letter)) \
        if hasattr(sim, "_axis_coords_um") else None
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
    legacy-fast-path marker). Every consumer of the window — the eps sampler,
    the eigensolve, and the equivalence-current sheet — must derive its
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
    hcen = h0 + (np.arange(nh) + off_h) * dl
    vcen = v0 + (np.arange(nv) + off_v) * dl
    fine_h = (hcen[:, None] + (np.arange(ss) - (ss - 1) / 2.0)[None, :] / ss * dl).ravel()
    fine_v = (vcen[:, None] + (np.arange(ss) - (ss - 1) / 2.0)[None, :] / ss * dl).ravel()
    return fine_h, fine_v


def _kfj_reduce(eps_fine, nh, nv, ss, dl, pts=None):
    """Diagonal-KFJ (eps_par, eps_xx_along_h, eps_yy_along_v) reduction of a
    supersampled hard-paint ``eps_fine`` [v*ss, h*ss]. Returns arrays [iv, ih].
    ``pts`` = optional ``(h_pts, v_pts)`` Yee point coordinates for a GRADED
    window (the interface-normal gradient then uses the true nonuniform
    spacings); ``None`` keeps the scalar-``dl`` gradient bit-identically."""
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
    if pts is None:
        gy, gx = np.gradient(epar, dl, dl)
    else:
        gy, gx = np.gradient(epar, pts[1], pts[0])
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
    (``None`` on a uniform axis — the legacy-fast-path marker), §20 BCs, and
    the per-offset Yee point coordinates. ``graded`` is True when either
    in-plane axis actually grades."""

    def __init__(self, h_nodes, h_dq, h_bc, v_nodes, v_dq, v_bc, dl):
        self.h_nodes, self.h_dq, self.h_bc = h_nodes, h_dq, h_bc
        self.v_nodes, self.v_dq, self.v_bc = v_nodes, v_dq, v_bc
        self.dl = dl
        self.nh, self.nv = len(h_nodes), len(v_nodes)
        self.h_lo, self.v_lo = float(h_nodes[0]), float(v_nodes[0])
        self.graded = h_dq is not None or v_dq is not None

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
    uniform path keeps :func:`_fine_centers` verbatim — algebraically equal
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
    """Frequency-parameterized Yee-staggered eps sampler. Rasterizes the window
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
    explicit anchor is frozen by design; the freeze is PER medium — other,
    un-overridden dispersive media keep their per-frequency anchoring);
    otherwise :meth:`Medium.permittivity_at_hz` at ``sample``'s ``freq_hz`` —
    for a Lorentz medium the band value, NOT the eps_inf that bare
    ``permittivity`` is; ``sample(None)`` falls back to bare ``permittivity``
    (legacy). When no un-overridden dispersive medium is present the first
    result is cached and the (large) geometry rasters released — callers just
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
        _, exx, _ = _kfj_reduce(lut[idx_maps[0] + 1], nh, nv, ss, dl, grads[0])
        _, _, eyy = _kfj_reduce(lut[idx_maps[1] + 1], nh, nv, ss, dl, grads[1])
        ezz, _, _ = _kfj_reduce(lut[idx_maps[2] + 1], nh, nv, ss, dl, grads[2])
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
    grid extents, and the snapped window origin (microns) — node (ih, iv) sits
    at ``(h_lo + ih*dl, v_lo + iv*dl)`` on a uniform grid (on a graded one the
    nodes are the sim's own ladder; use :func:`staggered_eps_sampler` for the
    full geometry). h = mode-x (width), v = mode-y (height). ``freq_hz``
    anchors dispersive media at that frequency (see
    :func:`staggered_eps_sampler`); ``None`` keeps the legacy bare
    ``permittivity`` (= eps_inf for a Lorentz medium — wrong for a dispersive
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
                   x_coords_um=None, y_coords_um=None):
    """Solve the discrete-Yee eigenproblem at ``wavelength_um`` for pre-sampled,
    Yee-staggered diagonal permittivity arrays over a window, returning the guided
    :class:`VectorMode`\\ s (n_eff-descending). Factored out of :func:`solve_yee_mode`
    so a per-frequency bank (:func:`solve_yee_mode_bank`) can sample the ε ONCE and
    re-solve per λ (only ``k0`` changes) — the Yee analogue of
    :meth:`VectorModeSolver.at_wavelength`. ``center_offset`` is the window
    placement metadata computed by the caller from ``sample_staggered_eps``'s
    snapped origin (see :func:`_window_center_offset`), carried on every
    returned mode.

    ``h_min_bc``/``v_min_bc`` put an engine §20 symmetry plane ON the window's
    low node line (see :func:`_dmats`): the restricted half of the matching-
    parity full-window eigenmode satisfies the half problem EXACTLY on the
    lattice, so a half-window solve reproduces the full mode's n_eff to
    eigensolver precision. ``"pec"`` (-1, odd) additionally pins the on-plane
    tangential-E DOFs (E_v on an h-min plane, E_h on a v-min plane) — the
    engine pins those same nodes every step; the mode's values there are 0 by
    parity. With a plane active, the spectrum contains ONLY the matching-
    parity family (mode_index counts within it).

    ``dq_h``/``dq_v`` = optional PRIMAL spacing vectors for GRADED window axes
    (:func:`_dmats` then builds the nonuniform Yee operators; ``None`` keeps
    the uniform scalar path bit-identically). ``x_coords_um``/``y_coords_um``
    = the node ladders RELATIVE to the requested mode centre, carried on the
    returned :class:`VectorMode` so consumers place a graded-solved mode on
    its true nonuniform raster (uniform callers may pass ``None`` — consumers
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

    n_core = float(np.sqrt(np.max(exx.real)))
    if nmodes >= 2 * N - 1:
        raise ValueError(
            f"trial mode count {nmodes} is too large for a {nh} x {nv} "
            f"cross-section ({2 * N} transverse unknowns); reduce num_modes "
            "or enlarge the solve window")
    vals, vecs = spl.eigs(mat, k=nmodes, sigma=-(n_core ** 2), which="LM")
    neff = np.sqrt(-vals)                        # eigenvalue = -n_eff^2
    order = np.argsort(-neff.real)
    neff, vecs = neff[order], vecs[:, order]

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
        if ne.real <= 1.0:
            continue
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
                        x_coords_um=x_coords_um, y_coords_um=y_coords_um)
        modes.append(vm)
    return modes


def _window_center_offset(h_lo, v_lo, nh, nv, dl_um, h_center_um, v_center_um):
    """Window placement metadata: node (ih, iv) sits at (lo + i*dl), so the
    array center — which consumers place at the requested center — actually
    sits at lo + (n-1)/2*dl. Carrying the difference lets vector_modal_fields
    put the mode back where its raster truly was (the grid snap displaces it by
    up to ~a cell). The remaining per-component ±dl/2 Yee stagger is a
    separate, documented compromise (consumers collocate — module docstring)."""
    return (h_lo + 0.5 * (nh - 1) * dl_um - h_center_um,
            v_lo + 0.5 * (nv - 1) * dl_um - v_center_um)


def _window_placement(geom: "_WindowGeom", h_center_um, v_center_um):
    """(center_offset, x_coords_um, y_coords_um) for a solved window. Uniform
    windows keep the legacy scalar offset (identical floats) and carry NO
    coords — consumers reconstruct the raster from ``dl_x_um`` exactly as
    before. Graded windows additionally carry the node ladders RELATIVE to the
    requested centre, which coordinate-aware consumers must prefer."""
    if not geom.graded:
        off = _window_center_offset(geom.h_lo, geom.v_lo, geom.nh, geom.nv,
                                    geom.dl, h_center_um, v_center_um)
        return off, None, None
    off = (0.5 * (geom.h_nodes[0] + geom.h_nodes[-1]) - h_center_um,
           0.5 * (geom.v_nodes[0] + geom.v_nodes[-1]) - v_center_um)
    return (off, geom.h_nodes - h_center_um, geom.v_nodes - v_center_um)


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


def solve_yee_mode(sim, axis: str, plane_value_um: float, wavelength_um: float,
                   pol: str, mode_index: int, *, h_center_um: float,
                   v_center_um: float, half_w_um: float, half_v_um: float,
                   dl_um: float, supersample: int = 8, num_modes: Optional[int] = None,
                   eps_of_medium: Optional[Mapping[int, float]] = None) -> VectorMode:
    """Solve the engine-consistent Yee-grid discrete eigenmode and return it as a
    :class:`VectorMode` (mode-frame [iy=height, ix=width]). Dispersive media are
    anchored at the solve frequency (:meth:`Medium.permittivity_at_hz`), matching
    the eps the engine's ADE realizes there — NOT the eps_inf that bare
    ``permittivity`` carries; ``eps_of_medium`` overrides per medium.

    §20 symmetry planes are honored AUTOMATICALLY: when ``sim.symmetry`` puts a
    plane on an in-plane axis' min face and the window reaches it, the window
    is clipped at the plane and the matching parity BC applied (PEC -1 / PMC
    +1, :func:`window_min_face_bcs`) — the half-window mode is the engine's
    half-domain field, and ``mode_index`` counts within the matching-parity
    family only."""
    sample, geom = staggered_eps_sampler(
        sim, axis, plane_value_um, h_center=h_center_um, v_center=v_center_um,
        half_w=half_w_um, half_v=half_v_um, dl=dl_um, supersample=supersample,
        eps_of_medium=eps_of_medium)
    exx, eyy, ezz = sample(C0 / (wavelength_um * 1e-6))
    off, xc, yc = _window_placement(geom, h_center_um, v_center_um)
    nmodes = num_modes or max(6, mode_index + 3)
    modes = _solve_yee_eig(exx, eyy, ezz, geom.nh, geom.nv, wavelength_um,
                           dl_um, nmodes, center_offset=off,
                           h_min_bc=geom.h_bc, v_min_bc=geom.v_bc,
                           dq_h=geom.h_dq, dq_v=geom.v_dq,
                           x_coords_um=xc, y_coords_um=yc)
    return _pick_yee(modes, pol, mode_index, axis, plane_value_um)


def solve_yee_mode_bank(sim, axis: str, plane_value_um: float, freqs_hz, pol: str,
                        mode_index: int, *, h_center_um: float, v_center_um: float,
                        half_w_um: float, half_v_um: float, dl_um: float,
                        supersample: int = 8, num_modes: Optional[int] = None,
                        eps_of_medium: Optional[Mapping[int, float]] = None):
    """``{freq_hz: VectorMode}`` per-frequency Yee-grid readout bank — the engine-
    consistent analogue of
    :func:`~photonhub.plugins.kfj_smoothing.mode_bank_on_cross_section` (which uses the
    node-collocated FLM ``VectorModeSolver``). The window geometry is rasterized
    ONCE; a non-dispersive cross-section shares one Yee-staggered ε for every
    frequency (λ-independent at constant n), while dispersive media are re-anchored
    per bank frequency (:meth:`Medium.permittivity_at_hz`) — then the discrete-Yee
    eigenproblem is re-solved per frequency, so the readout reference mode matches
    the FDTD field's discretization at every λ — the same discrete operator the
    launch used (:func:`solve_yee_mode`)."""
    nmodes = num_modes or max(6, mode_index + 3)
    out = {}
    for f, modes in _yee_bank_frames(
            sim, axis, plane_value_um, freqs_hz, nmodes=nmodes,
            h_center_um=h_center_um, v_center_um=v_center_um,
            half_w_um=half_w_um, half_v_um=half_v_um, dl_um=dl_um,
            supersample=supersample, eps_of_medium=eps_of_medium):
        out[f] = _pick_yee(modes, pol, mode_index, axis, plane_value_um)
    return out


def _yee_bank_frames(sim, axis, plane_value_um, freqs_hz, *, nmodes,
                     h_center_um, v_center_um, half_w_um, half_v_um, dl_um,
                     supersample, eps_of_medium):
    """The shared per-frequency solve loop of the Yee banks: yields
    ``(freq_hz, guided-modes-descending)`` per bank frequency. Geometry is
    rasterized once by :func:`staggered_eps_sampler`, which also decides
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
    """``{freq_hz: {mode_index: VectorMode}}`` MULTI-mode per-frequency Yee bank —
    the engine-consistent analogue of
    :func:`~photonhub.plugins.mode_devices.solve_mode_bank` (which needs an FLM/scalar
    solver object), ready for :meth:`ModeMonitor.mode_decomposition`.

    Indexing follows ``solve_mode_bank``'s convention: ``mode_indices`` count the
    guided modes in descending-``n_eff`` order ACROSS polarizations (0 = the
    fundamental) — NOT the per-polarization ``(pol, mode_index)`` selection of
    :func:`solve_yee_mode_bank`. Geometry is rasterized once; dispersive media are
    re-anchored per frequency (:meth:`Medium.permittivity_at_hz`)."""
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
    out = {}
    for f, modes in _yee_bank_frames(
            sim, axis, plane_value_um, freqs, nmodes=nmodes,
            h_center_um=h_center_um, v_center_um=v_center_um,
            half_w_um=half_w_um, half_v_um=half_v_um, dl_um=dl_um,
            supersample=supersample, eps_of_medium=eps_of_medium):
        if idxs[-1] >= len(modes):
            raise ValueError(
                f"requested mode_index {idxs[-1]} but the Yee solve found only "
                f"{len(modes)} guided mode(s) at {f:.4g} Hz "
                f"({C0 / f * 1e6:.4f} um) — the waveguide may not support it "
                "across the whole band")
        out[f] = {i: modes[i] for i in idxs}
    return out


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

    The cross-section is rasterized once and every requested family/index is
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
