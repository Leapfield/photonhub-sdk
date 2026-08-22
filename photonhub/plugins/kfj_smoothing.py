"""Diagonal KFJ/Kottke subpixel tensor on the ACTUAL device cross-section.

The benchmark (and any readout that wants a faithful reference mode) used to solve
its launch/readout modes on an *idealized* rectangular core (``_asym_strip_eps``,
8x8 volume average). Tidy3D instead solves its modes on the **real** grid
cross-section with a subpixel tensor. This module closes that gap: it samples the
actual ``Simulation`` geometry on a transverse plane, builds the **diagonal
Kottke-Farjadpour-Johnson** effective-permittivity tensor (the same
``averaged_eps_kfj`` math the FDTD engine uses, mirrored from
``engine/include/phcore/smoothing.h`` and from
:meth:`VectorModeSolver._kfj_tensor_rect`), and hands it to the full-vector FDE
solver.

Conventions (must match :meth:`VectorModeSolver._kfj_tensor_rect` and the overlap):
  * grid orientation ``[iy, ix]`` = ``(row = z/height, col = width)`` — exactly what
    :func:`photonhub.viz.eps.sample_eps_plane` returns (its vertical axis is always z
    for an x- or y-normal cut), i.e. ``_asym_strip_eps``'s convention.
  * ``eps_scalar = eps_par`` (the arithmetic average) — the solver reconstructs E by
    dividing by this scalar, so it must be ε‖, not ε⊥ or a tensor component.
  * tensor in the grid frame: ``exx`` along ix (width), ``eyy`` along iy (height),
    ``ezz = eps_par`` (the propagation axis is tangential to a z-invariant guide).

For isotropic constituents sharing one interface normal the Kottke/KFJ construction
(average tau(eps) over the cell, then invert) reduces EXACTLY to two sub-sample
averages — ``eps_par = <eps>`` (arithmetic) and ``eps_perp = <1/eps>^-1``
(harmonic) — for ANY number of media in the cell, so no foreground/surround
bookkeeping is required for the diagonal path. (This is NOT the engine's pairing:
``averaged_eps_kfj`` blends the topmost-at-centre medium against a
surround-excluding sample, a two-phase form. Do not assume the two agree cell for
cell on a >=3-medium corner.)
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional, Tuple

import numpy as np

from ..viz import _geometry as _geom
from ..viz import eps as _veps
from ._constants import C0

_AXES = "xyz"


def _paint_indices(sim, axis: str, plane_value_um: float,
                   h_centers: np.ndarray, v_centers: np.ndarray) -> np.ndarray:
    """Last-structure-wins GEOMETRY paint on an arbitrary ``(v, h)`` center grid
    (NUMERICS §9): ``idx[iy=v, ix=h]`` with ``-1`` = background and ``i`` =
    ``sim.structures[i]``. Painting indices (not eps values) lets a per-frequency
    consumer rasterize the geometry ONCE and re-map material values per frequency
    (dispersive media, :meth:`Medium.permittivity_at_hz`)."""
    a = _geom.axis_index(axis)
    h_letter, v_letter = _geom.in_plane_axes(axis)
    h_idx, v_idx = _AXES.index(h_letter), _AXES.index(v_letter)
    HH, VV = np.meshgrid(h_centers, v_centers)  # (n_v, n_h)
    idx = np.full(HH.shape, -1, dtype=np.int32)
    for si, s in enumerate(sim.structures):
        g = s.geometry
        gt = getattr(g, "type", None)
        if gt == "box":
            c, sz = g.center_um, g.size_um
            if abs(plane_value_um - c[a]) > sz[a] / 2.0:
                continue
            inside = ((np.abs(HH - c[h_idx]) <= sz[h_idx] / 2.0)
                      & (np.abs(VV - c[v_idx]) <= sz[v_idx] / 2.0))
            idx[inside] = si
        elif gt == "polyslab":
            inside = _veps._polyslab_inside(g, axis, plane_value_um, HH, VV, h_idx, v_idx)
            if inside is not None:
                idx[inside] = si
        elif gt == "cylinder":
            inside = _veps._cylinder_inside(g, axis, plane_value_um, HH, VV, h_idx, v_idx)
            if inside is not None:
                idx[inside] = si
        elif gt == "sphere":
            c, r = g.center_um, g.radius_um
            d_axis = plane_value_um - c[a]
            if abs(d_axis) >= r:
                continue
            inside = ((HH - c[h_idx]) ** 2 + (VV - c[v_idx]) ** 2 + d_axis ** 2) <= r * r
            idx[inside] = si
    return idx


def _paint_hard(sim, axis: str, plane_value_um: float,
                h_centers: np.ndarray, v_centers: np.ndarray,
                eps_of_struct) -> np.ndarray:
    """Last-structure-wins hard permittivity on an arbitrary ``(v, h)`` center grid
    (NUMERICS §9), mirroring :func:`viz.eps.sample_eps_plane` but on a grid we
    choose (so we can supersample). Returns ``eps[iy=v, ix=h]``; ``eps_of_struct``
    maps a structure to its permittivity *value* (so dispersive media can be anchored
    to a band-centre n rather than ``eps_inf``)."""
    idx = _paint_indices(sim, axis, plane_value_um, h_centers, v_centers)
    return _eps_lut(sim, eps_of_struct)[idx + 1]


def _any_dispersive(sim, eps_of_medium=None) -> bool:
    """True when any structure medium NOT covered by the ``eps_of_medium``
    override carries a Lorentz pole, i.e. its eps still depends on frequency
    (the ``Background`` is non-dispersive by schema). The override is
    PER-medium: pinning one medium must not freeze the others at whatever
    frequency a bank happens to evaluate first."""
    return any(
        getattr(s.medium, "is_dispersive", False)
        and not (eps_of_medium is not None and id(s.medium) in eps_of_medium)
        for s in sim.structures)


def _eps_lut(sim, eps_of) -> np.ndarray:
    """The background+structures eps look-up table paired with
    :func:`_paint_indices`' convention (``lut[idx + 1]``, background at 0).
    Rejects non-positive values — a frequency anchor that landed close enough
    to a Lorentz resonance to push Re eps <= 0 would silently break the mode
    eigensolve (sqrt of a negative core eps) far downstream."""
    lut = np.empty(len(sim.structures) + 1, dtype=np.float64)
    lut[0] = float(sim.background.permittivity)
    for si, s in enumerate(sim.structures):
        v = float(eps_of(s))
        if v <= 0.0:
            raise ValueError(
                f"non-positive permittivity {v:.4g} for structure {si} — the "
                "frequency anchor is too close to a Lorentz resonance for a "
                "mode solve (pin the medium via eps_of_medium, or move the "
                "band off the pole)")
        lut[si + 1] = v
    return lut


def _default_eps_of(eps_of_medium, freq_hz):
    """The shared structure→eps evaluator of the cross-section rasterizers: an
    explicit ``eps_of_medium`` entry wins (a deliberate anchor, held at EVERY
    frequency); otherwise the medium's eps AT ``freq_hz``
    (:meth:`Medium.permittivity_at_hz` — for a Lorentz medium the band value,
    NOT the eps_inf that bare ``permittivity`` is); ``freq_hz=None`` falls back
    to bare ``permittivity`` (legacy, correct only for non-dispersive media)."""
    def eps_of(s):
        if eps_of_medium is not None and id(s.medium) in eps_of_medium:
            return eps_of_medium[id(s.medium)]
        if freq_hz is not None:
            return s.medium.permittivity_at_hz(freq_hz)
        return float(s.medium.permittivity)
    return eps_of


def _snap_window(h_lo_um, h_hi_um, v_lo_um, v_hi_um, dl_um):
    """The grid-snapped raster window: origin floored to the sim grid, extent
    ceil'd to cover the requested hi. Returns ``(h_lo, v_lo, nh, nv)`` — the
    single source of truth for the window both the eps raster and the
    center-offset metadata use (cell ``i`` spans ``[lo + i*dl, lo + (i+1)*dl]``)."""
    h_lo_um = np.floor(h_lo_um / dl_um) * dl_um
    v_lo_um = np.floor(v_lo_um / dl_um) * dl_um
    nh = max(3, int(np.ceil((h_hi_um - h_lo_um) / dl_um)))
    nv = max(3, int(np.ceil((v_hi_um - v_lo_um) / dl_um)))
    return float(h_lo_um), float(v_lo_um), nh, nv


def _flm_center_offset(h_center_um, v_center_um, half_w_um, half_v_um, dl_um):
    """``center_offset_um`` for a mode solved on the snapped KFJ window: the
    actual position of the FLM field-array center minus the requested center.

    The FLM operator (:meth:`VectorModeSolver._assemble`) reads the four eps
    quadrants of node ``(ix, iy)`` from raster cells ``{ix, ix+1} x {iy, iy+1}``,
    i.e. the field node sits at the shared CORNER ``(h_lo + (ix+1)*dl,
    v_lo + (iy+1)*dl)`` (verified: the H-field intensity centroid of a strip
    mode lands exactly -0.5 raster-index units from the eps centroid). The
    array center is therefore at ``lo + (n+1)/2*dl``."""
    h_lo, v_lo, nh, nv = _snap_window(
        h_center_um - half_w_um, h_center_um + half_w_um,
        v_center_um - half_v_um, v_center_um + half_v_um, dl_um)
    return (h_lo + 0.5 * (nh + 1) * dl_um - h_center_um,
            v_lo + 0.5 * (nv + 1) * dl_um - v_center_um)


def sample_cross_section_kfj(
    sim, axis: str, plane_value_um: float, *,
    h_lo_um: float, h_hi_um: float, v_lo_um: float, v_hi_um: float,
    dl_um: float, supersample: int = 8,
    eps_of_medium: Optional[Mapping[int, float]] = None,
    freq_hz: Optional[float] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Diagonal KFJ tensor of the real cross-section over a window, on the sim grid.

    Returns ``(eps_scalar = eps_par, (exx, eyy, ezz))`` indexed ``[iy=height, ix=width]``
    — a drop-in for :meth:`VectorModeSolver._kfj_tensor_rect`'s output but built from
    the actual ``sim`` geometry. The window ``[h_lo,h_hi] x [v_lo,v_hi]`` (in the
    cut's (h, v) in-plane axes) is sampled at ``dl_um`` and supersampled ``supersample``x
    to get per-cell fill fractions. Material values: ``eps_of_medium``
    (``{id(medium): eps_value}``) overrides win; otherwise each medium is anchored
    at ``freq_hz`` when given (:meth:`Medium.permittivity_at_hz` — REQUIRED for a
    correct dispersive solve; bare ``permittivity`` is only eps_inf there), else
    bare ``permittivity`` (legacy)."""
    sample = cross_section_kfj_sampler(
        sim, axis, plane_value_um, h_lo_um=h_lo_um, h_hi_um=h_hi_um,
        v_lo_um=v_lo_um, v_hi_um=v_hi_um, dl_um=dl_um,
        supersample=supersample, eps_of_medium=eps_of_medium)
    return sample(freq_hz)


def cross_section_kfj_sampler(
    sim, axis: str, plane_value_um: float, *,
    h_lo_um: float, h_hi_um: float, v_lo_um: float, v_hi_um: float,
    dl_um: float, supersample: int = 8,
    eps_of_medium: Optional[Mapping[int, float]] = None,
):
    """Frequency-parameterized form of :func:`sample_cross_section_kfj`: the
    window geometry is rasterized ONCE (a structure-index paint) and the
    returned ``sample(freq_hz)`` re-maps material values at that frequency and
    reduces to the KFJ tensor — what makes a per-frequency bank over a
    DISPERSIVE cross-section affordable (the supersampled paint is the setup
    hotspot, not the LUT re-map). When no un-overridden dispersive medium is
    present the first result is cached and the geometry raster released."""
    # Snap the window origin to the simulation grid (whose cell centers sit at
    # (k+1/2)*dl from the domain origin 0), so the sampled cross-section's cells
    # COINCIDE with the sim's — the readout/launch mode then sees the dielectric
    # walls at the same sub-cell positions the FDTD field did (grid-consistent,
    # like Tidy3D solving its mode on its own grid). Without this the window has an
    # arbitrary sub-cell offset and the wall placement (hence n_eff) drifts.
    h_lo_um, v_lo_um, nh, nv = _snap_window(
        h_lo_um, h_hi_um, v_lo_um, v_hi_um, dl_um)
    ss = int(supersample)
    fine_dl = dl_um / ss
    hf = h_lo_um + (np.arange(nh * ss) + 0.5) * fine_dl
    vf = v_lo_um + (np.arange(nv * ss) + 0.5) * fine_dl
    state = {"idx": _paint_indices(sim, axis, plane_value_um, hf, vf)}
    frozen = not _any_dispersive(sim, eps_of_medium)

    def sample(freq_hz):
        if "cached" in state:
            return state["cached"]
        lut = _eps_lut(sim, _default_eps_of(eps_of_medium, freq_hz))
        result = _kfj_tensor_reduce(lut[state["idx"] + 1], nh, nv, ss, dl_um)
        if frozen:
            # eps is frequency-independent here: keep the (small) tensor,
            # release the supersampled geometry raster.
            state["cached"] = result
            state["idx"] = None
        return result

    return sample


def _kfj_tensor_reduce(eps_fine, nh, nv, ss, dl_um):
    """The diagonal-KFJ tensor reduction of a supersampled hard paint
    ``eps_fine`` [(nv*ss, nh*ss)] → ``(eps_par, (exx, eyy, ezz))`` [iy, ix]."""
    blk = eps_fine.reshape(nv, ss, nh, ss)

    # KFJ diagonal tensor. For isotropic constituents sharing one interface normal
    # the Kottke/KFJ construction (average tau(eps) over the cell, then invert)
    # gives EXACTLY eps_par = <eps> and eps_perp = <1/eps>^-1 for ANY number of
    # media — the (emax, emin, f) two-phase forms are only the N=2 special case.
    # Reducing a cell to its brightest/darkest pair mis-assigns every INTERMEDIATE
    # medium to emin: at an air / BOX / core triple junction the BOX sub-cells were
    # counted as air. Average over all sub-samples instead. Uniform cells: every
    # sub-sample equal, so epar == eperp == eps and the tensor is isotropic.
    epar = blk.mean(axis=(1, 3))                             # arithmetic, ε‖
    eperp = 1.0 / np.mean(1.0 / blk, axis=(1, 3))            # harmonic, ε⊥
    # Normal from the gradient of the CONTINUOUS epar field, NOT of ``f``: f is
    # the fill of the per-cell brightest material (== 1 in BOTH bulks), so a
    # wall contained in one cell column reads 1, f_wall, 1 and the central
    # difference at the wall cell VANISHES -> n_hat^2 = 0 -> the ARITHMETIC
    # average was silently applied to the NORMAL component too; a two-column
    # wall DID get a normal, so the defect was registration-dependent. epar is
    # monotone across any two-phase wall; in uniform cells d == 0, so the
    # zero-gradient fallback is irrelevant.
    gy, gx = np.gradient(epar, dl_um, dl_um)                 # iy, ix gradient
    gmag = np.hypot(gx, gy)
    safe = np.where(gmag > 1e-12, gmag, 1.0)
    nxh = np.where(gmag > 1e-12, gx / safe, 0.0)             # normal along ix (width)
    nyh = np.where(gmag > 1e-12, gy / safe, 0.0)             # normal along iy (height)
    d = eperp - epar
    exx = epar + d * nxh ** 2                                 # along ix
    eyy = epar + d * nyh ** 2                                 # along iy
    ezz = epar                                                # propagation (tangential)
    return epar, (exx, eyy, ezz)


def solve_mode_on_cross_section(
    sim, axis: str, plane_value_um: float, wavelength_um: float,
    pol: str, mode_index: int, *,
    h_center_um: float, v_center_um: float,
    half_w_um: float, half_v_um: float, dl_um: float,
    supersample: int = 8, x_symmetry: str = "none",
    num_modes: Optional[int] = None,
    eps_of_medium: Optional[Mapping[int, float]] = None,
    use_yee: bool = True,
):
    """Full-vector mode of the ACTUAL cross-section at ``axis = plane_value_um``,
    windowed to ``(h_center +/- half_w, v_center +/- half_v)``. The engine-wide
    default reference-mode builder for both launch and readout.

    ``use_yee`` (default **True**) solves the **engine-consistent discrete-Yee**
    eigenmode (:func:`~photonhub.plugins.yee_mode.solve_yee_mode`): the mode on the FDTD's
    own Yee-staggered grid, so source and readout match the propagated field's
    discretization exactly (the discrete analogue of what Tidy3D does).
    ``use_yee=False`` falls back to the node-collocated Fallahkhair–Li–Murphy
    ``VectorModeSolver`` (the prior default; kept for A/B and backward comparison).
    Both read the real geometry + diagonal KFJ subpixel and return the ``mode_index``-th
    ``pol`` mode (n_eff-descending).

    Symmetry: the Yee path honors ``sim.symmetry`` §20 planes AUTOMATICALLY
    (window clipped at the plane + the matching parity BC; see
    :func:`~photonhub.plugins.yee_mode.window_min_face_bcs`) — nothing to pass.
    ``x_symmetry`` is the FLM path's MANUAL width-axis wall control only
    (``"none"`` = electric walls, ``"pmc"`` = magnetic; it is NOT read from
    the sim and does not affect the Yee path).

    Dispersive media are anchored at the solve frequency automatically
    (:meth:`Medium.permittivity_at_hz`); an explicit ``eps_of_medium`` overrides
    that per medium. Unless ``eps_of_medium`` is given, the returned mode carries
    its ``solve_params`` provenance, which lets :class:`ModeMonitor` re-solve the
    SAME mode identity at every monitor frequency (the automatic per-frequency
    readout bank)."""
    solve_params = None
    if eps_of_medium is None:
        # Provenance for the per-frequency auto-bank: everything but the
        # wavelength (the bank supplies its own frequencies), INCLUDING the
        # simulation the mode was solved on — a replay must extend THIS mode's
        # identity, not re-solve on whatever (possibly different) simulation a
        # monitor was later built from. Deliberately NOT recorded when
        # eps_of_medium is given — the id-keyed override cannot be replayed
        # faithfully, and silently re-solving without it would anchor the bank
        # differently from the mode it extends.
        solve_params = dict(
            sim=sim, axis=axis, plane_value_um=plane_value_um, pol=pol,
            mode_index=mode_index, h_center_um=h_center_um,
            v_center_um=v_center_um, half_w_um=half_w_um, half_v_um=half_v_um,
            dl_um=dl_um, supersample=supersample, x_symmetry=x_symmetry,
            num_modes=num_modes, use_yee=use_yee)
    if use_yee:
        from .yee_mode import solve_yee_mode
        mode = solve_yee_mode(
            sim, axis, plane_value_um, wavelength_um, pol, mode_index,
            h_center_um=h_center_um, v_center_um=v_center_um,
            half_w_um=half_w_um, half_v_um=half_v_um, dl_um=dl_um,
            supersample=supersample, num_modes=num_modes, eps_of_medium=eps_of_medium)
        return replace(mode, solve_params=solve_params) if solve_params else mode

    from .vector_modes import VectorModeSolver

    _warn_flm_symmetry_blind(sim)
    eps_scalar, tensor = sample_cross_section_kfj(
        sim, axis, plane_value_um,
        h_lo_um=h_center_um - half_w_um, h_hi_um=h_center_um + half_w_um,
        v_lo_um=v_center_um - half_v_um, v_hi_um=v_center_um + half_v_um,
        dl_um=dl_um, supersample=supersample, eps_of_medium=eps_of_medium,
        freq_hz=C0 / (wavelength_um * 1e-6))
    solver = VectorModeSolver(eps_scalar, dl_um, dl_um, wavelength_um,
                              x_symmetry, eps_tensor=tensor)
    offset = _flm_center_offset(h_center_um, v_center_um, half_w_um, half_v_um,
                                dl_um)
    mode = _pick(solver, pol, mode_index, num_modes, axis, plane_value_um)
    return replace(mode, center_offset_um=offset, solve_params=solve_params)


def _warn_flm_symmetry_blind(sim):
    """The FLM (use_yee=False) path does NOT honor §20 symmetry planes: the
    eps paint happily rasterizes the mirror geometry below the domain min
    face, so a half-domain sim gets a FULL-cross-section mode that does not
    match the engine's half-domain field. Warn — the Yee default handles it."""
    import warnings

    if any(s != 0 for s in (getattr(sim, "symmetry", None) or (0, 0, 0))):
        warnings.warn(
            "use_yee=False on a simulation with a §20 symmetry plane: the FLM "
            "solver ignores sim.symmetry and solves the FULL (mirror-painted) "
            "cross-section — its mode does not match the engine's half-domain "
            "field. Use the Yee default (use_yee=True), which clips the window "
            "at the plane and applies the parity BC automatically.",
            UserWarning, stacklevel=3)


def _pick(solver, pol, mode_index, num_modes, axis, plane_value_um):
    modes = solver.solve(num_modes=num_modes or max(6, mode_index + 3))
    cands = [m for m in modes if m.polarization == pol]
    if mode_index >= len(cands):
        raise RuntimeError(
            f"cross-section solve at {axis}={plane_value_um:.3f}: requested "
            f"{pol}{mode_index} but found {len(cands)} {pol} mode(s); "
            f"te_fractions={[round(m.te_fraction, 3) for m in modes]}, "
            f"n_eff={[round(float(m.n_eff), 4) for m in modes]}")
    return cands[mode_index]


def mode_bank_on_cross_section(
    sim, axis: str, plane_value_um: float, freqs_hz, pol: str, mode_index: int, *,
    h_center_um: float, v_center_um: float, half_w_um: float, half_v_um: float,
    dl_um: float, supersample: int = 8, x_symmetry: str = "none",
    num_modes: Optional[int] = None, eps_of_medium=None, use_yee: bool = True):
    """``{freq_hz: VectorMode}`` per-frequency readout bank for a port. The window
    geometry is rasterized once; a NON-dispersive cross-section shares one ε for
    every frequency (λ-independent at constant n), while dispersive media are
    re-anchored at each bank frequency (:meth:`Medium.permittivity_at_hz`) so both
    the material AND waveguide dispersion land in the per-λ modes.

    ``use_yee`` (default **True**) uses the **engine-consistent discrete-Yee** solver
    (:func:`~photonhub.plugins.yee_mode.solve_yee_mode_bank`) so the readout reference mode
    matches the FDTD field's discretization at every λ — the same operator the launch
    used. ``use_yee=False`` re-solves via the node-collocated FLM
    :meth:`VectorModeSolver.at_wavelength` (the prior default; kept for A/B)."""
    if use_yee:
        from .yee_mode import solve_yee_mode_bank
        return solve_yee_mode_bank(
            sim, axis, plane_value_um, freqs_hz, pol, mode_index,
            h_center_um=h_center_um, v_center_um=v_center_um,
            half_w_um=half_w_um, half_v_um=half_v_um, dl_um=dl_um,
            supersample=supersample, num_modes=num_modes, eps_of_medium=eps_of_medium)

    from .vector_modes import VectorModeSolver

    _warn_flm_symmetry_blind(sim)
    freqs = list(freqs_hz)
    # Geometry rasterized once; the sampler re-anchors eps per frequency for
    # un-overridden dispersive media and caches the tensor otherwise.
    sample = cross_section_kfj_sampler(
        sim, axis, plane_value_um,
        h_lo_um=h_center_um - half_w_um, h_hi_um=h_center_um + half_w_um,
        v_lo_um=v_center_um - half_v_um, v_hi_um=v_center_um + half_v_um,
        dl_um=dl_um, supersample=supersample, eps_of_medium=eps_of_medium)
    offset = _flm_center_offset(h_center_um, v_center_um, half_w_um, half_v_um,
                                dl_um)
    out = {}
    for f in freqs:
        eps_scalar, tensor = sample(float(f))
        solver = VectorModeSolver(eps_scalar, dl_um, dl_um, C0 / float(f) * 1e6,
                                  x_symmetry, eps_tensor=tensor)
        mode = _pick(solver, pol, mode_index, num_modes, axis, plane_value_um)
        out[float(f)] = replace(mode, center_offset_um=offset)
    return out
