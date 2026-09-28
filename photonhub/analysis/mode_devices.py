"""Mode source & monitor builders, the client bridge from an FDE eigenmode to
the engine's ModeSource (NUMERICS.md §18) and to a mode-resolved transmission
readout.

``mode_source`` resamples a frozen FDE :class:`~photonhub.analysis.modes.Mode`
onto a simulation's transverse grid plane and returns a
:class:`~photonhub.components.sources.ModeSource` the engine injects via TF/SF.
``mode_monitor`` returns a :class:`ModeMonitor`, which carries a 4-tangential
``ProfileMonitor`` to add to the simulation and a ``.transmission(data)``
post-process that overlaps the recorded plane onto the mode (forward/backward
power ``T``) via :func:`photonhub.analysis.mode_overlap.mode_transmission`.

The injection and the overlap share one scalar-limit modal-H convention
(``h ≈ (n_eff/eta0) z_hat x e``), so a clean single-mode straight waveguide
reads ``T ≈ 1`` forward.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from ..viz import _geometry as _geom
from ..components import frame as _frame
from ..components.grid import (graded_primary_spacings, realized_cells,
                               sim_axis_min_cells,
                               snap_mixed_plane)
from ..components.monitors import ProfileMonitor
from ..components.sources import ModeSource
from ..components.source_time import SourceTimeType
# C0 maps a monitor/source frequency (Hz) to the FDE solver's wavelength
# (microns) via wavelength_um = C0 / freq_hz * 1e6 for the broadband
# (num_freqs) mode solves; _TANGENTIAL is the shared per-axis tangential
# component table (the order mode_transmission expects).
from ._constants import _TANGENTIAL, C0
from .mode_overlap import (
    ETA0,
    ModeBank,
    _TRANSVERSE,
    _cell_widths,
    modal_fields,
    mode_decomposition,
    mode_transmission,
    vector_modal_fields,
)
from .modes import Mode

_OPPOSITE = {"+": "-", "-": "+"}

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _axis_cell_centers(simulation, axis_name: str) -> np.ndarray:
    """Transverse cell-center coordinates (microns) along one axis.

    A uniform axis uses ``(i + 0.5)·dl``. A GRADED axis (GradedMesh coords)
    uses the midpoints of its primary-node cells (the §15.2 dual nodes), so the
    mode profile is sampled at the TRUE cell centers, this is what lets a mode
    source / monitor live on a transverse-graded mesh. The §18 auxiliary line
    also supports a graded propagation axis (NUMERICS.md §15.9); this helper
    samples only the two transverse plane axes because the source profile is
    defined on that plane."""
    idx = _AXIS_IDX[axis_name]
    q = simulation._axis_coords_um(idx)
    if q is None:  # uniform axis (UniformMesh, or a non-graded graded axis)
        dl = simulation.grid.dl_um
        size = simulation.size_um[idx]
        n = realized_cells(size, dl, sim_axis_min_cells(simulation, idx))
        return (np.arange(n) + 0.5) * dl
    # Graded axis: cell i spans [q[i], q[i+1]] (q[n] = §15.1 replicate-last
    # closing node), so its center is q[i] + dq[i]/2 with dq the primary
    # spacings (replicate-last for the final cell). Matches the engine's §15.2
    # dual-node convention, so the resampled profile lands on the cells the
    # solver injects into.
    qa = np.asarray(q, dtype=float)
    dq = np.asarray(graded_primary_spacings(tuple(q)), dtype=float)
    return qa + dq / 2.0


def _axis_nodes(simulation, axis_name: str) -> np.ndarray:
    """Transverse primary-node coordinates (microns) along one axis, one per
    cell: ``i·dl`` on a uniform axis, the graded ladder's nodes otherwise. The
    Yee sample positions of the field components that sit ON a node along
    this axis (the partner of :func:`_axis_cell_centers`)."""
    idx = _AXIS_IDX[axis_name]
    q = simulation._axis_coords_um(idx)
    if q is None:
        dl = simulation.grid.dl_um
        n = realized_cells(simulation.size_um[idx], dl, sim_axis_min_cells(simulation, idx))
        return np.arange(n) * dl
    return np.asarray(q, dtype=float)


def _yee_plane_grids(simulation, axis: str):
    """The plane's sample positions for the transverse E component along each
    in-plane axis, as the §18 engine stamps ``profile[cv·nu + cu]``: E along
    ``t1`` (and the H it pairs with) at ``(t1 cell centre, t2 node)``, E along
    ``t2`` at ``(t1 node, t2 cell centre)`` (NUMERICS §18.2). Returns
    ``((u, v) for the t1 component, (u, v) for the t2 component)``."""
    t1, t2 = _TRANSVERSE[axis]
    uc, vc = _axis_cell_centers(simulation, t1), _axis_cell_centers(simulation, t2)
    un, vn = _axis_nodes(simulation, t1), _axis_nodes(simulation, t2)
    return (uc, vn), (un, vc)


def _node_weights(simulation, axis_name: str, nodes: np.ndarray) -> np.ndarray:
    """Quadrature widths (microns) of node-registered samples along one axis:
    the dual cell of each node, the node row on a §20 symmetry plane or a §4
    ``pmc`` lower wall counting half (only half its dual cell is modeled, the
    flux reduction's rule, NUMERICS §12)."""
    w = _cell_widths(nodes).astype(float)
    a = _AXIS_IDX[axis_name]
    sym = getattr(simulation, "symmetry", None)
    bnd = getattr(getattr(simulation, "boundaries", None), axis_name, None)
    if w.size and ((sym is not None and sym[a] != 0) or bnd == "pmc"):
        w[0] *= 0.5
    return w


def _default_center(simulation, axis: str) -> Tuple[float, float]:
    """The transverse domain midpoints (t1, t2), where a centered waveguide
    sits, used as the default mode location."""
    t1, t2 = _TRANSVERSE[axis]
    return (
        simulation.size_um[_AXIS_IDX[t1]] / 2.0,
        simulation.size_um[_AXIS_IDX[t2]] / 2.0,
    )


def _broadband_arrays(modes_by_freq, resample, central_pol, central_major,
                      central_minor, resample_h=None):
    """Pack ``{freq_hz: Mode}`` into the :class:`ModeSource` broadband kwargs
    (``freqs_hz`` / ``n_eff_by_freq`` / ``profiles_by_freq`` [+ minor] [+ true-H]).

    Returns ``{}`` for fewer than two entries, the legacy single-mode launch.
    Each mode is resampled with the SAME ``resample`` callable as the band-centre
    mode (it returns ``(major_flat, minor_flat_or_None, polarization)``). Two
    invariants make the engine's partition-of-unity windowing well-posed:
    (1) the major polarization must not change across the band (same guided
    mode); (2) each profile's arbitrary global eigen-sign is aligned to
    ``central_major`` (the same sign applied to the minor AND to the per-frequency
    true-H profiles, to preserve the component ratio and E–H consistency) so
    adjacent windowed carriers add coherently rather than cancel.

    When ``resample_h`` is given (full-vector source) the mode's TRUE paired-H is
    resampled at EACH frequency (its own n_eff) and shipped as
    ``profiles_h_by_freq`` [+ minor], so every carrier injects the H of the mode
    at that frequency, not the single band-centre H, which is correct only at the
    band centre and radiates a non-decaying residual off centre (§18.3)."""
    if modes_by_freq is None or len(modes_by_freq) < 2:
        return {}
    freqs = sorted(float(f) for f in modes_by_freq)
    has_minor = central_minor is not None
    ship_h = resample_h is not None
    neffs, majors, minors, h_majors, h_minors = [], [], [], [], []
    for f in freqs:
        m = modes_by_freq[f]
        maj, minr, pol = resample(m)
        if pol != central_pol:
            raise ValueError(
                f"the mode's major polarization changes across the band "
                f"({central_pol} -> {pol} at {f:.4g} Hz); a broadband source "
                "needs the SAME mode at every frequency (narrow the band, or "
                "select the matching mode_index in solve_modes_by_freq)"
            )
        sign = -1.0 if float(np.dot(maj, central_major)) < 0.0 else 1.0
        majors.append(tuple(float(v) for v in sign * maj))
        neffs.append(float(m.n_eff))
        if has_minor:
            if minr is None:
                raise ValueError(
                    "the band-centre mode is full-vector but the mode at "
                    f"{f:.4g} Hz has no minor component"
                )
            minors.append(tuple(float(v) for v in sign * minr))
        if ship_h:
            hmaj, hmin = resample_h(m)   # true H at this freq's own n_eff
            h_majors.append(tuple(float(v) for v in sign * hmaj))
            if has_minor:
                h_minors.append(tuple(float(v) for v in sign * hmin))
    out = dict(
        freqs_hz=tuple(freqs),
        n_eff_by_freq=tuple(neffs),
        profiles_by_freq=tuple(majors),
    )
    if has_minor:
        out["profiles_minor_by_freq"] = tuple(minors)
    if ship_h:
        out["profiles_h_by_freq"] = tuple(h_majors)
        if has_minor:
            out["profiles_h_minor_by_freq"] = tuple(h_minors)
    return out


def _is_full_vector(mode) -> bool:
    """A full-vector mode carries the true paired H (``hx``/``hy``), e.g. a
    :class:`~photonhub.analysis.vector_modes.VectorMode` (incl. the engine-consistent
    ``yee_mode`` discrete eigenmode). A scalar :class:`Mode` does not."""
    return getattr(mode, "hx", None) is not None and \
        getattr(mode, "hy", None) is not None


def mode_source(
    simulation,
    mode: Mode,
    *,
    axis: str,
    position_um: float,
    source_time: SourceTimeType,
    direction: str = "+",
    amplitude: float = 1.0,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[str] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
    paired_h: bool = True,
) -> ModeSource:
    """Build a :class:`ModeSource` injecting ``mode`` on the ``axis`` plane at
    ``position_um`` of ``simulation`` (uniform or graded grid).

    .. deprecated::
        The §18 aux-line ModeSource is deprecated in favour of the source current launch: prefer :func:`mode_launch` with a discrete Yee mode
        (:func:`~photonhub.analysis.yee_mode.solve_yee_mode` /
        :func:`~photonhub.analysis.kfj_smoothing.solve_mode_on_cross_section`). The
        Phased-dipole launch works on uniform AND graded grids, supports
        broadband, and sheds less near-source radiation. Full-vector calls here
        delegate to :func:`mode_source_vector` (which emits the deprecation
        warning); §18 is retained for the adjoint and scalar/FLM modes.

    **Full-vector launch is the default (NUMERICS.md §18.2a / launch_fidelity).**
    When ``mode`` is a full-vector mode (it carries the true paired ``H``, e.g. a
    :class:`~photonhub.analysis.vector_modes.VectorMode`, especially the engine's own
    ``yee_mode`` discrete eigenmode) and ``paired_h`` is True (default), this
    delegates to :func:`mode_source_vector` so the source ships the mode's TRUE
    discrete paired-H (``profile_h``). The full-vector launch uses that paired H
    instead of the scalar-impedance-H approximation. The launch is then power-normalized
    (``power_watts = amplitude²``
    so a non-unit ``amplitude`` still scales power as a peak-field would). Pass
    ``paired_h=False`` to force the legacy scalar-limit launch, or pass a scalar
    :class:`Mode` (no H), both give the prior single-component behavior.

    The mode's major transverse-E profile is resampled (peak-normalized) at
    that component's own Yee positions on the plane (NUMERICS §18.2).
    ``amplitude`` is then the peak injected field.
    ``center_um`` places the waveguide in the transverse plane (default: the
    domain center, i.e. a centered guide). ``thickness_axis`` is the simulation
    axis along the guide's slab thickness; pass the slab normal (e.g. ``"z"``)
    for any non-x propagation so the mode is not rotated 90 degrees (see
    :func:`~photonhub.analysis.mode_overlap.modal_fields`). ``None`` keeps the
    legacy thickness-on-second-transverse-axis mapping.

    **Broadband injection (``num_freqs`` analogue, NUMERICS.md §18.3).** Pass
    ``modes_by_freq`` (``{freq_hz: Mode}`` from :func:`solve_modes_by_freq`, the
    same map a :func:`mode_monitor` takes) to inject a FREQUENCY-DEPENDENT
    profile and ``n_eff`` instead of the single frozen ``mode``. Each mode is
    resampled and the engine partition-of-unity-windows them across the band, so
    a wide-band / dispersive launch stays mode-matched at every frequency. The
    positional ``mode`` remains the band-centre representative (and the global
    sign reference the per-frequency profiles are aligned to). With fewer than
    two entries this is a no-op (the single ``mode`` is used)."""
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    # A full-vector mode launches via the vector source. With paired_h=True
    # (default) it ships the mode's TRUE discrete paired-H (profile_h) — the
    # §18.2a discrete full-vector launch; with paired_h=False the true-H profiles
    # are stripped, giving the legacy scalar-impedance-H launch of the same
    # (major+minor) E. A scalar Mode (no H) always falls through to the scalar
    # path below.
    if _is_full_vector(mode):
        src = mode_source_vector(
            simulation, mode, axis=axis, position_um=position_um,
            source_time=source_time, direction=direction,
            power_watts=float(amplitude) ** 2, center_um=center_um,
            thickness_axis=thickness_axis, modes_by_freq=modes_by_freq,
        )
        if not paired_h:
            src = src.model_copy(
                update={"profile_h": None, "profile_h_minor": None})
        return src
    t1_name, t2_name = _TRANSVERSE[axis]
    # each component at its own Yee position on the plane (NUMERICS §18.2)
    grid1, grid2 = _yee_plane_grids(simulation, axis)
    if center_um is None:
        center_um = _default_center(simulation, axis)

    def _resample(m: Mode):
        """Peak-normalized major-E profile (flat C-order) + its polarization,
        resampled onto this plane at that component's Yee positions: the
        shared scalar-source readout."""
        f1 = modal_fields(
            m, *grid1, axis=axis, n_eff=m.n_eff,
            center_um=center_um, thickness_axis=thickness_axis,
        )
        # The major-E component is whichever of e1/e2 modal_fields filled (the
        # other is identically zero); read it back rather than re-deriving.
        if np.any(f1["e1"]):
            profile2d, pol = f1["e1"], "E" + t1_name  # [iv, iu]
        else:
            f2 = modal_fields(
                m, *grid2, axis=axis, n_eff=m.n_eff,
                center_um=center_um, thickness_axis=thickness_axis,
            )
            profile2d, pol = f2["e2"], "E" + t2_name
        peak = float(np.max(np.abs(profile2d)))
        if not peak > 0.0:
            raise ValueError(
                "the resampled mode profile is identically zero on this plane "
                "— check the mode window vs the simulation transverse extent / "
                "center"
            )
        # [iv*nu+iu] = [cv*nu+cu]; no minor in the scalar limit.
        return (profile2d / peak).reshape(-1), None, pol

    profile, _, polarization = _resample(mode)

    bb = _broadband_arrays(
        modes_by_freq, _resample, polarization, profile, central_minor=None,
    )
    return ModeSource(
        axis=axis,
        direction=direction,
        position_um=position_um,
        polarization=polarization,
        amplitude=amplitude,
        n_eff=float(mode.n_eff),
        nu=int(grid1[0].size),
        nv=int(grid1[1].size),
        profile=tuple(float(v) for v in profile),
        source_time=source_time,
        **bb,
    )


def _launch_window_origin(mode, h_center, v_center):
    """The EXACT window origin ``(h_lo, v_lo)`` the mode was solved on,
    recovered from its own recorded placement so the launch registers on the
    solve grid without threading the original window through. Inverts
    :func:`~photonhub.analysis.yee_mode._window_center_offset`
    (``off = lo + 0.5(n-1)dl - center``): ``lo = center + off - 0.5(n-1)dl``,
    all exact grid multiples, passed straight to the sheet builder, so no
    float-boundary-sensitive floor-snap of a reconstructed half-width. Requires
    the mode to carry ``center_offset_um`` (the Yee cross-section solve sets
    it)."""
    off = getattr(mode, "center_offset_um", None)
    if off is None:
        raise ValueError(
            "mode carries no center_offset_um — solve it with "
            "solve_yee_mode / solve_mode_on_cross_section to enable the "
            "equivalence-current launch, or pass launch='aux'")
    dl = float(mode.dl_x_um)
    nv, nh = np.asarray(mode.ex).shape
    return (h_center + float(off[0]) - 0.5 * (nh - 1) * dl,
            v_center + float(off[1]) - 0.5 * (nv - 1) * dl)


def _eq_current_ineligible(simulation, mode, modes_by_freq):
    """Why the equivalence-current launch can't be used for this call, or
    ``None`` if it can. The gating for the ``launch='auto'`` default: the
    per-cell Huygens sheet needs the engine-consistent full-vector Yee mode
    (a single frozen band centre OR a per-frequency broadband bank of them ,
    Stage B: the sheet now carries the band via partition-of-unity windowed
    carriers, so broadband is eligible for eq-current on uniform AND graded
    grids). Graded (§15) grids additionally need the mode's solve provenance
    (``solve_params``) to re-derive the exact window ladder; the per-frequency
    modes must each carry the same provenance/placement (checked on ``mode``,
    the band-centre representative, which shares the window with the bank)."""
    if not (_is_full_vector(mode) and getattr(mode, "yee_staggered", False)):
        return ("the mode is not a discrete full-vector Yee mode (needs true "
                "paired H on the engine grid — use solve_yee_mode / "
                "solve_mode_on_cross_section)")
    if not getattr(simulation.grid, "dl_um", None):
        return "the grid carries no base dl_um"
    graded_mode = getattr(mode, "x_coords_um", None) is not None
    p = getattr(mode, "solve_params", None)
    if graded_mode and not (p and "h_center_um" in p):
        return ("a graded-window mode needs solve provenance (solve_params) "
                "to re-derive its window ladder — solve it via "
                "solve_yee_mode / solve_mode_on_cross_section")
    if not graded_mode and getattr(mode, "center_offset_um", None) is None:
        return "the mode carries no window placement (center_offset_um)"
    return None


def _solved_center_um(mode) -> Optional[Tuple[float, float]]:
    """The transverse ``(h, v)`` centre the mode was actually SOLVED at, from
    its ``solve_params`` provenance, or ``None`` when it carries none.

    Defaulting a launch or a readout to the domain centre silently mis-places
    every off-centre device: the sheet is stamped, and the overlap projected,
    on a cut the mode was never solved on. It does not raise, the run
    completes and the numbers look plausible. A measured coupler read
    through 1.0065 + cross 0.9538 (energy sum 1.96, physically impossible)
    and passed for a working 3 dB splitter; supplying the solve centre gave
    0.4922 / 0.5097, sum 1.0020. Every shipped example is centred, so the
    default was never exercised by the docs.

    ``solve_params`` is recorded by ``solve_mode_on_cross_section`` for every
    mode except an ``eps_of_medium`` override (which cannot be replayed), so
    this resolves for the ordinary path.
    """
    params = getattr(mode, "solve_params", None)
    if not params:
        return None
    try:
        h = params["h_center_um"]
        v = params["v_center_um"]
    except (KeyError, TypeError):
        return None
    if h is None or v is None:
        return None
    return float(h), float(v)


def _solved_center_in_plane_frame(mode, axis: str) -> Optional[Tuple[float, float]]:
    """:func:`_solved_center_um` re-ordered into the readout plane's
    ``(t1, t2) = _TRANSVERSE[axis]`` frame. The solve provenance records the
    centre in the solve window's ``(h, v) = in_plane_axes(axis)`` order; the
    two frames coincide for an x- or z-cut but SWAP for a y-cut (``(x, z)`` vs
    ``(z, x)``), so handing the raw pair to a y-normal monitor placed the mode
    at ``(t1=x_c, t2=z_c)``, off the plane entirely (a zero-power reference
    mode, or a plausible-looking wrong overlap on a taller domain)."""
    solved = _solved_center_um(mode)
    if solved is None:
        return None
    h_name, v_name = _geom.in_plane_axes(axis)
    by_name = {h_name: solved[0], v_name: solved[1]}
    t1, t2 = _TRANSVERSE[axis]
    return (by_name[t1], by_name[t2])


def mode_launch(
    simulation,
    mode,
    *,
    axis: str,
    position_um: float,
    source_time: SourceTimeType,
    direction: str = "+",
    power_watts: float = 1.0,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[str] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
    launch: str = "auto",
) -> list:
    """Build the source list for a guided mode and pass it to ``Simulation.sources``.

    ``launch='auto'`` uses phased ``PointDipole`` sources for a full-vector Yee
    mode with recorded grid placement. This supports graded grids and broadband
    ``modes_by_freq``. Other inputs use :class:`ModeSource`.
    ``launch='eq_current'`` requires the phased-dipole path and raises when the
    input is ineligible. ``launch='aux'`` always uses ``ModeSource``.

    ``center_um`` places the transverse waveguide in the cut's
    ``(horizontal, vertical)`` axis order on the uniform phased-dipole and
    ``ModeSource`` paths. With no explicit center, these paths use the mode's
    recorded solve center, or the domain center when it is unavailable.
    The phased-dipole path for a graded-window mode always uses its recorded
    solve window, including its center and widths. It ignores an explicit
    ``center_um``. Solve the mode at the intended center before launching it.
    ``power_watts`` sets the launched modal power
    by scaling the source plane's dipole amplitudes: the port's ``mode_power`` and a
    ``PowerMonitor`` downstream report that power in watts (``RunResult``
    restores the engine's per-unit-amplitude normalization), while
    ``transmission`` and every other ratio is unchanged by it. It is the
    power of the whole, unfolded device: a launch centred on §20 symmetry
    planes puts ``power_watts / 2`` per plane into the part the simulation
    models. The port and a full-plane ``PowerMonitor`` report ``power_watts``
    for the whole device (NUMERICS §20.8). The source window is recovered
    from the mode's recorded placement, so the launch registers on the solve
    grid.

    The continuous-adjoint pipeline uses :func:`mode_source`, whose gradient
    normalization is tied to that excitation."""
    if launch not in ("auto", "eq_current", "aux"):
        raise ValueError(
            f"launch must be 'auto', 'eq_current', or 'aux', got {launch!r}")
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")

    why = _eq_current_ineligible(simulation, mode, modes_by_freq)
    if launch == "eq_current" and why is not None:
        raise ValueError(f"launch='eq_current' not possible: {why}")
    use_eq = (launch == "eq_current") or (launch == "auto" and why is None)

    # center in the (h, v) = in_plane_axes frame (what the Yee/eq stack uses).
    h_letter, v_letter = _geom.in_plane_axes(axis)
    if center_um is None:
        # Default to where the mode was SOLVED, not the domain centre: an
        # off-centre waveguide launched at the domain centre is silently
        # wrong physics (see _solved_center_um). The domain centre remains
        # the fallback only when the mode carries no provenance to use.
        solved = _solved_center_um(mode)
        if solved is not None:
            h_center, v_center = solved
        else:
            h_center = simulation.size_um[_AXIS_IDX[h_letter]] / 2.0
            v_center = simulation.size_um[_AXIS_IDX[v_letter]] / 2.0
    else:
        h_center, v_center = float(center_um[0]), float(center_um[1])

    def modeled_watts(hc, vc):
        # power_watts is the unfolded device's: a launch centred on k §20
        # symmetry planes puts 1/2^k of it into the modeled part, so the port
        # reads power_watts back through its whole plane (NUMERICS §20.8).
        if power_watts is None:
            return None
        return float(power_watts) / _on_plane_factor(
            simulation, ((h_letter, hc), (v_letter, vc)))

    if use_eq:
        from .eq_current_source import equivalence_current_source

        # Broadband: pass the per-frequency Yee bank so the sheet carries the
        # band via windowed carriers (Stage B). None / single-entry falls to the
        # frozen band-centre `mode` inside the builder — bit-identical.
        bank = modes_by_freq if (modes_by_freq is not None
                                 and len(modes_by_freq) >= 2) else None
        p = getattr(mode, "solve_params", None)
        if getattr(mode, "x_coords_um", None) is not None:
            # Graded-window mode: re-derive the window ladder from the EXACT
            # solve arguments (provenance — guaranteed by the eligibility
            # gate), so the sheet lands on the same graded nodes the solve
            # used. The launch is placed at the mode's own solved window.
            return equivalence_current_source(
                simulation, mode, axis=axis, position_um=position_um,
                source_time=source_time, direction=direction,
                h_center_um=p["h_center_um"], v_center_um=p["v_center_um"],
                half_w_um=p["half_w_um"], half_v_um=p["half_v_um"],
                power_watts=modeled_watts(p["h_center_um"], p["v_center_um"]),
                modes_by_freq=bank)
        origin = _launch_window_origin(mode, h_center, v_center)
        return equivalence_current_source(
            simulation, mode, axis=axis, position_um=position_um,
            source_time=source_time, direction=direction,
            h_center_um=h_center, v_center_um=v_center,
            half_w_um=0.0, half_v_um=0.0,
            power_watts=modeled_watts(h_center, v_center),
            _window_origin=origin, modes_by_freq=bank)

    # §18 fallback. mode_source's center_um is in the _TRANSVERSE (t1, t2)
    # order, which SWAPS vs in_plane_axes for a y-cut — key by axis letter so
    # the reorder is correct for every propagation axis.
    coord = {h_letter: h_center, v_letter: v_center}
    t1, t2 = _TRANSVERSE[axis]
    return [mode_source(
        simulation, mode, axis=axis, position_um=position_um,
        source_time=source_time, direction=direction,
        amplitude=math.sqrt(modeled_watts(h_center, v_center)),
        center_um=(coord[t1], coord[t2]), thickness_axis=thickness_axis,
        modes_by_freq=modes_by_freq)]


def mode_source_vector(
    simulation,
    mode,
    *,
    axis: str,
    position_um: float,
    source_time: SourceTimeType,
    direction: str = "+",
    power_watts: float = 1.0,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[str] = None,
    modes_by_freq: Optional[Mapping[float, object]] = None,
) -> ModeSource:
    """Build a FULL-VECTOR, power-normalized :class:`ModeSource` from a
    ``VectorMode`` (NUMERICS.md §18).

    Where :func:`mode_source` injects the scalar-limit major-E component
    (peak-normalized, ``amplitude`` = peak field), this packs BOTH transverse-E
    components of the full-vector mode and **power-normalizes** the launch to
    ``power_watts`` (default **1 W**). Both transverse-E profiles are resampled,
    each at its own Yee positions on the plane (NUMERICS §18.2), preserving
    their true component ratio (via
    :func:`~photonhub.analysis.mode_overlap.vector_modal_fields`); the minor
    component rides the same guided-mode aux carrier as the major (engine §18.2),
    with its own scalar-limit paired H.

    **1 W normalization (computed here, on the Python side; the engine stays
    power-agnostic).** The engine injects ``E_t = amplitude * profile`` and the
    scalar-limit paired ``H = (n_eff/eta0)(z_hat x E_t)``, so the launched modal
    Poynting flux is

        P_inj = (1/2) integral Re(E x H*) . z_hat dA
              = (n_eff / (2 eta0)) * amplitude^2
                * integral (|profile_major|^2 + |profile_minor|^2) dA .

    We resample the *unnormalized* transverse-E pair, evaluate that integral on
    each sample's area: the cell width along a cell-centered axis and the dual
    width along a node axis, halved on a symmetry plane or pmc wall. Scale
    BOTH packed profiles by
    ``1/sqrt(P_inj_at_unit_scale / power_watts)`` so the injected mode carries
    exactly ``power_watts`` in the engine's own (scalar-H) convention. (The
    field-only L2 normalization the FDE solver applies has arbitrary units, so a
    power normalization here is what makes the launch physically meaningful and
    lets transmission read an absolute fraction.) ``amplitude`` is left at 1.0;
    the whole power scaling lives in the profiles.

    Phase note: for a lossless guided mode both transverse-E components are
    co-real (relative phase 0 or π), so the real signed ``profile``/
    ``profile_minor`` capture the launch exactly; any out-of-phase (quadrature)
    part of the minor-E would need a second carrier and is dropped (a no-op for
    the lossless guided modes this targets).

    Accuracy note (absolute power): the 1 W normalization above integrates the
    SCALAR-LIMIT paired H (``P = n_eff/(2 eta0) * integral |E_t|^2``), while
    the source also ships the mode's TRUE-H profiles (``profile_h`` /
    ``profile_h_minor``) for the engine's injection. Where the true H deviates
    from the scalar limit (high-contrast cores, ~1%), the actually injected
    modal power differs from ``power_watts`` by that correction, transmission
    RATIOS cancel it (both planes read the same launch), only the absolute
    wattage carries the bias. Left as-is pending an engine-side verification
    of the injected-power convention.

    **Broadband injection (``num_freqs`` analogue, NUMERICS.md §18.3).** Pass
    ``modes_by_freq`` (``{freq_hz: VectorMode}`` from :func:`solve_modes_by_freq`
    over a :class:`~photonhub.analysis.vector_modes.VectorModeSolver`) to inject a
    frequency-dependent full-vector profile across the band; each carrier is
    power-normalized to ``power_watts`` and the engine partition-of-unity-windows
    them. The positional ``mode`` stays the band-centre representative and the
    sign reference. Fewer than two entries is a no-op (single ``mode``).
    """
    warnings.warn(
        "mode_source_vector / the §18 aux-line ModeSource is deprecated: prefer "
        "mode_launch(...) with a discrete Yee mode (solve_yee_mode / "
        "solve_mode_on_cross_section), which injects a per-cell equivalence-"
        "current Huygens sheet that works on uniform AND graded grids and now "
        "supports broadband (num_freqs>1). The §18 path is retained only for the "
        "adjoint (its gradient is pinned to it) and scalar/FLM modes.",
        DeprecationWarning, stacklevel=2,
    )
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if not power_watts > 0.0:
        raise ValueError(f"power_watts must be > 0, got {power_watts}")
    t1_name, t2_name = _TRANSVERSE[axis]
    # Each transverse component at its own Yee position on the plane, as the
    # engine stamps it (NUMERICS §18.2): E along t1 and the H it pairs with at
    # (t1 cell centre, t2 node), E along t2 and its H at (t1 node, t2 cell
    # centre). The dA of each: the cell width along a cell-centred axis, the
    # dual width along a node axis, whose row on a §20 plane counts half.
    grid1, grid2 = _yee_plane_grids(simulation, axis)
    if center_um is None:
        center_um = _default_center(simulation, axis)
    dA1_m2 = np.outer(_node_weights(simulation, t2_name, grid1[1]), _cell_widths(grid1[0])) * 1e-12
    dA2_m2 = np.outer(_cell_widths(grid2[1]), _node_weights(simulation, t1_name, grid2[0])) * 1e-12

    def _fields(m):
        """(e1, e2, h1, h2) each at its own Yee positions: e1 with the h2 it
        pairs with, e2 with h1."""
        f1 = vector_modal_fields(m, *grid1, axis=axis, direction=direction,
                                 center_um=center_um, thickness_axis=thickness_axis)
        f2 = vector_modal_fields(m, *grid2, axis=axis, direction=direction,
                                 center_um=center_um, thickness_axis=thickness_axis)
        return f1["e1"], f2["e2"], f2["h1"], f1["h2"]

    def _resample(m):
        """Power-normalized (major, minor) real profiles + major polarization,
        resampled onto this plane, the shared full-vector source readout."""
        e1, e2, _, _ = _fields(m)  # transverse-E along t1, t2 ([iv, iu])
        # The MAJOR transverse axis carries the larger transverse-E energy.
        if float(np.sum(np.abs(e1) ** 2)) >= float(np.sum(np.abs(e2) ** 2)):
            e_major, pol_maj, e_minor = e1, "E" + t1_name, e2
            dA_maj, dA_min = dA1_m2, dA2_m2
        else:
            e_major, pol_maj, e_minor = e2, "E" + t2_name, e1
            dA_maj, dA_min = dA2_m2, dA1_m2
        if not float(np.sum(np.abs(e_major) ** 2)) > 0.0:
            raise ValueError(
                "the resampled mode profile is identically zero on this plane "
                "— check the mode window vs the simulation transverse extent / "
                "center"
            )
        # Real signed profiles (lossless guided mode -> transverse-E co-real;
        # the real part is exact there). Keep the major/minor RATIO.
        maj = np.real(e_major)
        minr = np.real(e_minor)
        # power_watts normalization in the engine's scalar-H convention (see the
        # docstring P_inj derivation), evaluated AT this mode's n_eff.
        p_unit = (float(m.n_eff) / (2.0 * ETA0)) * float(
            np.sum(maj ** 2 * dA_maj) + np.sum(minr ** 2 * dA_min)
        )
        if not p_unit > 0.0:
            raise ValueError(
                "modal power integral is non-positive; cannot normalize")
        scale = float(np.sqrt(power_watts / p_unit))
        # C-order [iv*nu + iu] = [cv*nu + cu]
        return (maj * scale).reshape(-1), (minr * scale).reshape(-1), pol_maj

    maj, minr, pol_major = _resample(mode)
    pol_minor = ("E" + t2_name) if pol_major == "E" + t1_name else ("E" + t1_name)

    def _resample_h(m):
        """True paired-H profiles (E-equivalent units h·η0/n_eff) for mode ``m``,
        evaluated at ITS OWN n_eff, sign-aligned to the E profiles so each reduces
        to +profile in the scalar limit (matching the legacy engine path), the
        deviation IS the true-H correction the engine's E-correction needs to stop
        radiating the scalar-limit-H mismatch (~few %). Called once per band-centre
        and once per broadband carrier (each at its own frequency's n_eff)."""
        c1, c2, g1, g2 = _fields(m)
        # SAME major/minor criterion as _resample above (complex transverse-E
        # energy) — a real-part criterion could route the E and H profiles to
        # opposite axes for a mode with residual imaginary content; for the
        # lossless co-real modes this targets the two coincide.
        major_t1 = (float(np.sum(np.abs(c1) ** 2))
                    >= float(np.sum(np.abs(c2) ** 2)))
        e1, e2, h1, h2 = (np.real(f) for f in (c1, c2, g1, g2))
        e_maj, e_min = (e1, e2) if major_t1 else (e2, e1)
        h_maj, h_min = (h2, h1) if major_t1 else (h1, h2)  # E_t pairs with H of the OTHER axis
        dA_maj, dA_min = (dA1_m2, dA2_m2) if major_t1 else (dA2_m2, dA1_m2)
        p_unit = (float(m.n_eff) / (2.0 * ETA0)) * float(
            np.sum(e_maj ** 2 * dA_maj) + np.sum(e_min ** 2 * dA_min))
        sc = float(np.sqrt(power_watts / p_unit))
        fac = (ETA0 / float(m.n_eff)) * sc
        hmaj, hmin = h_maj * fac, h_min * fac
        if float(np.vdot(e_maj.ravel(), hmaj.ravel())) < 0.0:
            hmaj = -hmaj
        if float(np.vdot(e_min.ravel(), hmin.ravel())) < 0.0:
            hmin = -hmin
        return hmaj.reshape(-1), hmin.reshape(-1)

    h_maj_prof, h_min_prof = _resample_h(mode)

    bb = _broadband_arrays(
        modes_by_freq, _resample, pol_major, maj, central_minor=minr,
        resample_h=_resample_h,
    )
    return ModeSource(
        axis=axis,
        direction=direction,
        position_um=position_um,
        polarization=pol_major,
        amplitude=1.0,  # the power scaling lives entirely in the profiles
        n_eff=float(mode.n_eff),
        nu=int(grid1[0].size),
        nv=int(grid1[1].size),
        profile=tuple(float(v) for v in maj),
        minor_polarization=pol_minor,
        profile_minor=tuple(float(v) for v in minr),
        profile_h=tuple(float(v) for v in h_maj_prof),
        profile_h_minor=tuple(float(v) for v in h_min_prof),
        source_time=source_time,
        **bb,
    )


#: Sentinel for the de-stagger default. When a readout's ``destagger_dl`` is left
#: at this value, de-stagger is applied AUTOMATICALLY using the monitor's grid
#: spacing ``dl_um`` whenever ``colocate=True`` (real Yee-staggered FDTD data),
#: and skipped when ``colocate=False`` (synthetic, already-co-located fields).
#: Pass an explicit ``destagger_dl=None`` to force it off, or a float to override.
_DESTAGGER_AUTO = object()

# A port whose transverse centre lies within this distance of a §20 symmetry
# plane (coordinate 0 of a folded axis) is centred on it (ModeMonitor
# ._fold_power_factor); float noise only, a folded port sits on the plane exactly.
_ON_PLANE_TOL_UM = 1e-6


def _on_plane_factor(simulation, centre) -> float:
    """2 for every §20 symmetry plane a port's transverse centre lies on, the
    ratio of the power through the port's whole plane to the part a folded
    simulation models (NUMERICS §20.8). ``centre`` pairs each in-plane axis
    letter with the centre's coordinate on it."""
    sym = getattr(simulation, "symmetry", None) if simulation is not None else None
    if not sym:
        return 1.0
    factor = 1.0
    for letter, c in centre:
        if sym[_AXIS_IDX[letter]] != 0 and abs(float(c)) <= _ON_PLANE_TOL_UM:
            factor *= 2.0
    return factor


@dataclass(frozen=True)
class ModeMonitor:
    """A mode-resolved transmission monitor: a 4-tangential ``ProfileMonitor``
    (add ``.field_monitor`` to the simulation) plus a ``.transmission(data)``
    post-process that overlaps the recorded plane onto ``mode``."""

    field_monitor: ProfileMonitor
    mode: Mode
    axis: str
    center_um: Optional[Tuple[float, float]] = None
    direction: str = "+"
    thickness_axis: Optional[str] = None
    modes_by_freq: Optional[Mapping[float, Mode]] = None
    #: Optional multi-mode bank ``{freq_hz: {mode_index: Mode}}`` (per-frequency)
    #: or ``{mode_index: Mode}`` (frozen) for :meth:`mode_decomposition`. Build
    #: the per-frequency form with :func:`solve_mode_bank`.
    mode_bank: Optional[ModeBank] = None
    #: Grid spacing (microns) along the propagation/normal axis, captured from the
    #: simulation by :func:`mode_monitor`. Enables the de-stagger by default (see
    #: :data:`_DESTAGGER_AUTO`); ``None`` if the monitor was built without a grid.
    dl_um: Optional[float] = None
    #: The simulation this monitor was built from (:func:`mode_monitor` stores
    #: it). Needed only by the automatic per-frequency reference-mode bank —
    #: ``None`` (a hand-built monitor) disables that and keeps the frozen-mode
    #: readout.
    simulation: Optional[Any] = None
    #: Automatic per-frequency reference modes (the default readout): when True
    #: and no explicit ``modes_by_freq``/bank is supplied, :meth:`mode_power`
    #: re-solves ``mode`` at EVERY monitor frequency through its solve
    #: provenance (``mode.solve_params``, attached by
    #: ``solve_mode_on_cross_section``) and projects each frequency onto its
    #: own-frequency mode — the standard mode-monitor convention. Silently keeps
    #: the frozen band-centre mode when the provenance or ``simulation`` is
    #: missing. Set False for the legacy frozen-mode readout.
    per_freq_modes: bool = True

    @property
    def name(self) -> str:
        """Name of the underlying field monitor used to load this readout."""
        return self.field_monitor.name

    def _auto_modes_by_freq(self) -> Optional[Mapping[float, Mode]]:
        """The automatic per-frequency reference-mode bank (built once, cached
        on the instance). ``None`` when ineligible: :attr:`per_freq_modes` off,
        no :attr:`simulation`, ``mode`` carries no solve provenance, the
        monitor's single frequency IS the mode's own solve frequency (a bank
        would just re-solve the same mode), or the re-solve failed (warned once,
        frozen-mode fallback)."""
        if not self.per_freq_modes or self.simulation is None:
            return None
        params = getattr(self.mode, "solve_params", None)
        if not params:
            return None
        try:
            return getattr(self, "_auto_bank_cache")
        except AttributeError:
            pass
        freqs = [float(f) for f in self.field_monitor.freqs_hz]
        bank: Optional[Mapping[float, Mode]] = None
        lam0 = getattr(self.mode, "wavelength_um", None)
        single_at_centre = (
            len(freqs) == 1 and lam0
            and abs(C0 / freqs[0] * 1e6 - lam0) <= 1e-9 * lam0)
        if freqs and not single_at_centre:
            from .kfj_smoothing import mode_bank_on_cross_section

            p = dict(params)
            # Re-solve on the simulation the mode was SOLVED on (carried in
            # its provenance), not the monitor's: the bank extends the given
            # mode's identity, and the two simulations can legitimately differ
            # (e.g. a reference shell vs the full device).
            bank_sim = p.pop("sim", None) or self.simulation
            try:
                bank = mode_bank_on_cross_section(
                    bank_sim, p.pop("axis"), p.pop("plane_value_um"),
                    freqs, p.pop("pol"), p.pop("mode_index"), **p)
            except Exception as e:  # noqa: BLE001 — the readout must never be
                # worse than the legacy frozen-mode path: ANY re-solve failure
                # (eigensolver non-convergence, LinAlgError, a missing scipy,
                # provenance drift) falls back, loudly and once.
                warnings.warn(
                    f"automatic per-frequency mode bank for monitor "
                    f"{self.name!r} failed ({type(e).__name__}: {e}); falling "
                    "back to the frozen band-centre mode. Pass "
                    "per_freq_modes=False to silence, or an explicit "
                    "modes_by_freq.",
                    UserWarning, stacklevel=3)
                bank = None
        # Cache the outcome (a failed build too — warn once, not per reading).
        object.__setattr__(self, "_auto_bank_cache", bank)
        return bank

    def _resolved_modes_by_freq(self, explicit=None, n_eff=None):
        """The reference-mode map a readout should project onto: an explicit
        per-call map wins, then the stored :attr:`modes_by_freq`, then the
        automatic per-frequency bank. An explicit ``n_eff`` override suppresses
        the AUTO bank only, each bank mode carries its own n_eff, which would
        silently discard the caller's value (explicit maps already had that
        semantics before the auto-bank existed). Every readout path (power,
        amplitude, S-matrix) must resolve through here so they agree on the
        reference modes."""
        mbf = explicit if explicit is not None else self.modes_by_freq
        if mbf is None and n_eff is None:
            mbf = self._auto_modes_by_freq()
        return mbf

    def _fold_low(self) -> Tuple[bool, bool]:
        """(t1, t2) in-plane §20 fold flags for the folded-domain readout
        quadrature (``mode_overlap._overlap_terms``): True where the in-plane
        axis carries a symmetry fold on the monitor's simulation, so a
        node-registered sample row ON the fold plane (the axis MIN face)
        weights half a cell instead of spilling ``dl/2`` into the mirror half ,
        the parity-asymmetric power inflation that under-read T for
        cross-parity port pairs (fold-antinode in, fold-node out). Without a
        stored simulation there is no fold information and no correction."""
        sim = self.simulation
        sym = getattr(sim, "symmetry", None) if sim is not None else None
        if not sym:
            return (False, False)
        t1, t2 = _TRANSVERSE[self.axis]
        return (sym[_AXIS_IDX[t1]] != 0, sym[_AXIS_IDX[t2]] != 0)

    def _fold_power_factor(self) -> float:
        """How much more power crosses this port's whole plane than its folded
        record holds: 2 for every §20 symmetry plane the port is centred on
        (its transverse centre on the plane, which a half domain puts at
        coordinate 0 of that axis). Such a port's mode is solved on the half
        window the fold keeps, so the recorded half carries exactly half of the
        port's power; a port off the plane is recorded whole, and its mirror
        twin is read through it. Without this factor a transmission between a
        port on the plane and one off it (a 1 x 2 splitter's input and an arm,
        a crossing's input and its cross port) reads 2x per such plane."""
        if self.center_um is None:
            return 1.0
        return _on_plane_factor(self.simulation, zip(_TRANSVERSE[self.axis], self.center_um))

    def mode_power(
        self,
        data,
        *,
        direction: Optional[str] = None,
        n_eff: Optional[float] = None,
        modes_by_freq: Optional[Mapping[float, Mode]] = None,
        colocate: bool = True,
        destagger_dl=_DESTAGGER_AUTO,
    ) -> Dict[float, float]:
        """The forward (or backward) modal **power** ``{freq_hz:
        |a_pm|²/P_mode · 1e-12}`` on this plane, the actual power carried by
        ``mode`` through it, in the run's (source-spectrum-normalized) SI flux
        units. The value is **flux-commensurate**: it shares both the §12
        normalization and the SI (m²) area element with a ``PowerMonitor``, so
        ``mode_power / flux`` on one plane is the modal power fraction (~the
        modal confinement, O(1)), on a simulation with symmetry planes too:
        a port centred on a plane and a ``PowerMonitor`` the plane cuts both
        read the whole, unfolded device's power (NUMERICS §20.8); see
        :func:`~photonhub.analysis.mode_overlap.mode_transmission`
        ``power=True`` for the µm²→m² conversion note. This is still NOT a 0–1
        transmission on its own; ratio two planes for that (see
        :func:`transmission`).

        Returns true *power* (``|c|²·P_mode``), not the bare squared amplitude
        ``|c|²``, so that ``P_out / P_in`` is the correct power transmission even
        when the two ports carry **different** modes (e.g. a w1→w2 taper, where the
        per-mode ``P_mode`` differs and must not cancel). With the de-stagger on,
        ``P_mode`` is the mode's power through this plane's own cell (below), so
        two planes in cells of different widths also ratio correctly; only a
        reflection ``-``/``+`` at one plane cancels it exactly. ``data`` is the
        ``RunResult`` from the run; ``data[self.name]`` is the recorded DFT
        plane. Pass ``modes_by_freq`` (``{freq_hz: Mode}``) to project each
        frequency onto its own per-λ mode instead of the frozen ``self.mode``
        (overrides the monitor's stored ``modes_by_freq`` if any).

        **Per-frequency reference modes are the default**: with no explicit
        ``modes_by_freq`` anywhere, a monitor built by :func:`mode_monitor`
        from a dispatcher-solved mode re-solves that mode at every monitor
        frequency automatically (see :attr:`per_freq_modes`), so wide-band
        readings track the modal profile/n_eff drift instead of freezing the
        band-centre mode. Ineligible monitors keep the frozen mode silently.

        **De-stagger is ON by default** (the longitudinal Yee de-stagger; see
        :func:`~photonhub.analysis.mode_overlap.mode_transmission`): when ``colocate``
        is True it uses the monitor's grid ``dl_um`` automatically, matching what
        a colocating mode monitor does when it interpolates the
        staggered Yee components to common coordinates. Pass ``destagger_dl=None``
        to force it off (e.g. for already-co-located synthetic fields), or a float
        to override the spacing. With it on, the power is the flux the Yee
        scheme conserves through the plane, ``|c|^2 * P_mode * cos(beta*dl/2)``,
        so two ports in cells of different widths, or carrying modes of
        different ``n_eff``, ratio to the true transmission (NUMERICS §18.7).

        On a simulation with §20 symmetry planes the value is the power through
        the port's whole plane: a port centred on a plane reads its recorded
        half doubled per plane (NUMERICS §20.8)."""
        if destagger_dl is _DESTAGGER_AUTO:
            destagger_dl = self.dl_um if colocate else None
        da = _frame.wire_array(data, self.name)
        planes: Mapping[str, object] = {
            c: da.sel(component=c) for c in _TANGENTIAL[self.axis]
        }
        mbf = self._resolved_modes_by_freq(modes_by_freq, n_eff=n_eff)
        power = mode_transmission(
            planes,
            self.mode,
            axis=self.axis,
            direction=direction or self.direction,
            n_eff=n_eff,
            center_um=self.center_um,
            thickness_axis=self.thickness_axis,
            modes_by_freq=mbf,
            power=True,
            colocate=colocate,
            destagger_dl=destagger_dl,
            fold_low=self._fold_low(),
        )
        factor = self._fold_power_factor()
        return power if factor == 1.0 else {f: factor * p for f, p in power.items()}

    def mode_decomposition(
        self,
        data,
        *,
        quantity: str = "transmission",
        direction: Optional[str] = None,
        mode_bank: Optional[ModeBank] = None,
        colocate: bool = True,
        destagger_dl=_DESTAGGER_AUTO,
    ) -> Dict[int, Dict[float, Any]]:
        """Decompose the recorded plane onto MULTIPLE modes → ``{mode_index:
        {freq_hz: value}}`` (a mode monitor with ``num_modes``).

        Projects the plane onto every mode in the mode mapping (each index, each
        frequency) instead of the single ``self.mode`` that :meth:`mode_power`
        uses. The mode mapping is ``mode_bank`` if given, else the monitor's stored
        ``self.mode_bank``; it is ``{freq_hz: {mode_index: Mode}}`` (per-frequency,
        dispersive, see :func:`solve_mode_bank`) or ``{mode_index: Mode}``
        (frozen). ``quantity`` selects ``"transmission"`` (``|c|²``, default),
        ``"power"`` (``|a_pm|²/P_mode · 1e-12``, the flux-commensurate per-mode
        power to ratio across ports),
        or ``"amplitude"`` (complex ``c``, for a multimode S-matrix). See
        :func:`~photonhub.analysis.mode_overlap.mode_decomposition`."""
        bank = mode_bank if mode_bank is not None else self.mode_bank
        if not bank:
            raise ValueError(
                "no mode_bank: pass mode_bank=... or build the ModeMonitor with "
                "one (see solve_mode_bank); mode_decomposition needs >1 mode")
        if destagger_dl is _DESTAGGER_AUTO:  # de-stagger ON by default (see mode_power)
            destagger_dl = self.dl_um if colocate else None
        da = _frame.wire_array(data, self.name)
        planes: Mapping[str, object] = {
            c: da.sel(component=c) for c in _TANGENTIAL[self.axis]
        }
        out = mode_decomposition(
            planes,
            bank,
            axis=self.axis,
            direction=direction or self.direction,
            quantity=quantity,
            center_um=self.center_um,
            thickness_axis=self.thickness_axis,
            colocate=colocate,
            destagger_dl=destagger_dl,
            fold_low=self._fold_low(),
        )
        factor = self._fold_power_factor()
        if quantity == "power" and factor != 1.0:    # the whole plane, as mode_power
            out = {m: {f: factor * p for f, p in per.items()} for m, per in out.items()}
        return out


def transmission(
    out_monitor: ModeMonitor,
    in_monitor: ModeMonitor,
    data,
    *,
    direction: Optional[str] = None,
    n_eff: Optional[float] = None,
    colocate: bool = True,
    destagger_dl=_DESTAGGER_AUTO,
) -> Dict[float, float]:
    """Mode-resolved power transmission ``{freq_hz: T}`` from ``in_monitor`` to
    ``out_monitor``, the ratio of modal powers, which cancels the source and
    spectrum normalization so a lossless single-mode straight guide reads
    ``T ≈ 1``. Place ``in_monitor`` just after the source (total-field side) and
    ``out_monitor`` at the device output.

    ``direction=None`` (default) reads each monitor in its OWN stored
    ``direction`` (so e.g. an out-monitor built with ``direction="-"``, a port
    facing the source, is read backward, as placed). An explicit ``"+"``/``"-"``
    overrides BOTH planes with that one direction (it used to be the silent
    default, flipping a ``"-"`` out-monitor to a forward read).

    The longitudinal Yee de-stagger is applied by default (each monitor uses
    its own grid ``dl_um`` when ``colocate=True``), it removes the input-plane
    standing-wave ripple and matches the colocating-readout convention;
    pass ``destagger_dl=None`` to force it off. See
    :meth:`ModeMonitor.mode_power`."""
    p_in = in_monitor.mode_power(data, direction=direction, n_eff=n_eff,
                                 colocate=colocate, destagger_dl=destagger_dl)
    p_out = out_monitor.mode_power(data, direction=direction, n_eff=n_eff,
                                   colocate=colocate, destagger_dl=destagger_dl)
    return {f: p_out[f] / p_in[f] for f in p_out if f in p_in}


def reflection(
    monitor: ModeMonitor,
    in_monitor: ModeMonitor,
    data,
    **kwargs,
) -> Dict[float, float]:
    """Mode-resolved power reflection ``{freq_hz: R}`` at the driven port:
    ``monitor`` read AGAINST its stored ``direction`` (the modal power the
    device sends back toward the source) over ``in_monitor`` read in its own
    direction (the launched power). ``in_monitor`` is the driven port's
    plane, on the device side of the launch and reading in the launch
    direction, as for :func:`transmission`. ``monitor`` must record that same
    plane with the same normal and direction: ``in_monitor`` itself, or a
    monitor of another mode on its recorded plane (the same name) for the
    power reflected into that mode. Any other ``monitor`` raises
    ``ValueError``.

    When ``data`` comes from a simulation that drives a port (``ports=`` with
    ``source=`` naming one), ``in_monitor`` must be that port's monitor,
    reading in the launch direction, or ``ValueError`` is raised: a port that
    is not driven reads the power leaving the device, so read the other way
    it measures the wave arriving from its own boundary (nothing, with no
    source there), not the power reflected into that port. To read the
    reflection at another port, drive that port. When the simulation drives
    no port (built by hand with its own sources and monitors, or loaded from
    the wire), nothing says which monitor is driven or which way the launch
    travels, and neither is checked: pass as ``in_monitor`` the monitor at
    the launch plane (on its device side), reading in the launch direction.
    Any other monitor returns a number that is not the reflection.

    The two readings share one plane, so the source spectrum cancels, and with
    one mode the mode normalization too: ``R + T = 1`` across the band is the
    energy check for a lossless device (a Bragg grating's stopband is read
    this way). Keyword arguments are those of :meth:`ModeMonitor.mode_power`
    except ``direction``, which the reading fixes; passing it raises
    ``TypeError``."""
    if "direction" in kwargs:
        raise TypeError(
            "reflection() takes no direction=: it reads the driven port's plane "
            "against its stored direction for the reflected power and along it for "
            "the launched power. For another reading use "
            "ModeMonitor.mode_power(data, direction=...)")
    if monitor.name != in_monitor.name:
        raise ValueError(
            f"reflection is read at the driven port: monitor {monitor.name!r} does not "
            f"record the plane of in_monitor {in_monitor.name!r}, which reads the "
            "launched power; pass in_monitor as monitor too, or a monitor of another "
            "mode on its recorded plane. A port's plane that reads the power leaving "
            "the device, read the other way, measures the wave arriving from its own "
            "boundary, not a reflection; drive that port to read its reflection")
    if monitor.axis != in_monitor.axis:
        raise ValueError(
            f"monitor and in_monitor both name the recorded plane {monitor.name!r} but "
            f"give it different normals ({monitor.axis!r} and {in_monitor.axis!r}); "
            "pass in_monitor as monitor too")
    if monitor.direction != in_monitor.direction:
        raise ValueError(
            f"monitor {monitor.name!r} reads {monitor.direction!r} but in_monitor "
            f"{in_monitor.name!r} reads {in_monitor.direction!r}: reflection reads one "
            "plane both ways, in in_monitor's direction for the launched power and "
            "against it for the reflected power, so monitor must read in_monitor's "
            "direction; pass in_monitor as monitor too")
    driven, port_monitor = _driven_port_monitor(data)
    if port_monitor is not None and in_monitor.name != port_monitor.name:
        raise ValueError(
            f"reflection is read at the driven port: in_monitor {in_monitor.name!r} is "
            f"not the monitor of the driven port {driven!r}, {port_monitor.name!r}. A "
            "port that is not driven reads the power leaving the device, so read the "
            "other way it measures the wave arriving from its own boundary, not a "
            "reflection; pass the driven port's monitor "
            f"(simulation.port_monitors[{driven!r}]) as in_monitor and monitor, or drive "
            "the other port to read its reflection")
    if port_monitor is not None and in_monitor.direction != port_monitor.direction:
        raise ValueError(
            f"in_monitor {in_monitor.name!r} reads {in_monitor.direction!r}, but the "
            f"driven port {driven!r} launches {port_monitor.direction!r}: the launched "
            "power is read in the launch direction; pass the driven port's monitor "
            f"(simulation.port_monitors[{driven!r}]) as in_monitor and monitor")
    back = monitor.mode_power(data, direction=_OPPOSITE[monitor.direction], **kwargs)
    launched = in_monitor.mode_power(data, direction=in_monitor.direction, **kwargs)
    return {f: back[f] / launched[f] for f in back if f in launched}


def _driven_port_monitor(data) -> Tuple[Optional[str], Optional[ModeMonitor]]:
    """``(port name, ModeMonitor)`` of the port driven by the simulation that
    ``data`` comes from, or ``(None, None)`` when there is no such port: no
    simulation, or one built by hand, loaded from the wire, or beam-driven."""
    try:
        sim = getattr(data, "simulation", None)
    except (OSError, ValueError):   # an unreadable sim.json: the result reads on without it
        return None, None
    driven = getattr(sim, "driven_port", None)
    if driven is None:
        return None, None
    return driven, sim.port_monitors.get(driven)


def _spectrum(values_by_freq: Mapping[float, float], *, name: str, attrs: Dict[str, str]):
    """``{freq_hz: value}`` as an ``xarray.DataArray`` over ``f`` (Hz,
    ascending) with a ``wlen_um`` coordinate (microns)."""
    import xarray as xr

    freqs = np.asarray(sorted(values_by_freq), dtype=np.float64)
    values = np.asarray([values_by_freq[f] for f in freqs], dtype=np.float64)
    return xr.DataArray(
        values, dims=("f",),
        coords={"f": ("f", freqs, {"units": "Hz"}),
                "wlen_um": ("f", C0 / freqs * 1e6, {"units": "um"})},
        attrs=attrs, name=name)


def transmission_spectrum(
    out_monitor: ModeMonitor,
    in_monitor: ModeMonitor,
    data,
    **kwargs,
):
    """:func:`transmission` as a labelled array: an ``xarray.DataArray`` over
    ``f`` (Hz, ascending) carrying a ``wlen_um`` coordinate (microns) and the
    two monitor names in ``attrs``, so a spectrum plots and slices without the
    ``{freq: T}`` bookkeeping. Keyword arguments are those of
    :func:`transmission`."""
    return _spectrum(transmission(out_monitor, in_monitor, data, **kwargs),
                     name=f"T[{out_monitor.name}/{in_monitor.name}]",
                     attrs={"out": out_monitor.name, "in": in_monitor.name,
                            "quantity": "modal power transmission"})


def reflection_spectrum(
    monitor: ModeMonitor,
    in_monitor: ModeMonitor,
    data,
    **kwargs,
):
    """:func:`reflection` as a labelled array, the shape of
    :func:`transmission_spectrum`: the reflection at the driven port,
    ``in_monitor``, read on ``monitor``, which must record the same plane
    (see :func:`reflection` for what is checked, what the caller must pass
    for a simulation that drives no port, and the keywords)."""
    return _spectrum(reflection(monitor, in_monitor, data, **kwargs),
                     name=f"R[{monitor.name}/{in_monitor.name}]",
                     attrs={"port": monitor.name, "in": in_monitor.name,
                            "quantity": "modal power reflection"})


def mode_monitor(
    simulation,
    mode: Mode,
    *,
    axis: str,
    position_um: float,
    freqs_hz,
    name: str,
    direction: str = "+",
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[str] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
    mode_bank: Optional[ModeBank] = None,
    per_freq_modes: bool = True,
) -> ModeMonitor:
    """Build a :class:`ModeMonitor` (a 4-tangential ``ProfileMonitor`` on the
    ``axis`` plane at ``position_um`` + a transmission post-process onto
    ``mode``). Add ``.field_monitor`` to the simulation's monitors, run, then
    call ``.transmission(data)``. ``thickness_axis`` is the slab-normal axis
    (pass e.g. ``"z"`` for non-x propagation so the overlap mode is not rotated
    90 degrees); ``None`` keeps the legacy mapping. Pass ``mode_bank``
    (``{freq_hz: {mode_index: Mode}}``, see :func:`solve_mode_bank` /
    :func:`~photonhub.analysis.yee_mode.solve_yee_multimode_bank`) to enable
    :meth:`ModeMonitor.mode_decomposition` (multi-mode readout). See
    :func:`mode_source`.

    ``per_freq_modes`` (default True): when no ``modes_by_freq`` is given and
    ``mode`` came from ``solve_mode_on_cross_section`` (it carries its solve
    provenance), the monitor re-solves the mode at EVERY ``freqs_hz`` on first
    use and projects each recorded frequency onto its own-frequency mode, the per-λ readout a standard mode monitor performs, now the default here
    too. False keeps the frozen band-centre ``mode`` for all frequencies (the
    legacy readout)."""
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    idx = _AXIS_IDX[axis]
    # A plane carrying mixed Yee offsets (Ex/Ey at integer, Hx/Hy at half-cell
    # along the normal) is rejected unless every component snaps to one cell;
    # placing the plane at the §12 quarter point of ONE cell does that. The
    # snap is graded-aware: a graded normal axis snaps to its LOCAL cell's
    # quarter point and the returned local spacing (NOT the grid's base dl_um,
    # which GradedMesh also carries) feeds the longitudinal de-stagger.
    position_um, dl = snap_mixed_plane(simulation, idx, position_um)
    # The plane spans the REALIZED domain (graded coords — and non-commensurate
    # uniform sims — can realize a hair short of the nominal size; a
    # nominal-size box then fails the engine's in-domain validation). The
    # 1e-6 um shave (a picometre — 4+ orders below any dl, cannot exclude a
    # Yee point) is UNCONDITIONAL: even when the client's realized length
    # equals the nominal size bit-for-bit, the engine's own realized-length
    # arithmetic can land a few float ULPs below it (seen at e.g.
    # dl = 1.55/(3.4738*14) um), and a plane exactly on that edge is rejected.
    extent = list(simulation.size_um)
    realized = getattr(simulation, "_realized_um", None)
    if callable(realized):
        extent = [min(n, r) - 1e-6
                  for n, r in zip(extent, realized())]
    size = list(extent)
    size[idx] = 0.0  # a plane normal to `axis`
    center = [s / 2.0 for s in extent]
    center[idx] = position_um
    fm = ProfileMonitor(
        name=name,
        center_um=tuple(center),
        size_um=tuple(size),
        fields=_TANGENTIAL[axis],
        freqs_hz=tuple(freqs_hz),
    )
    return ModeMonitor(
        field_monitor=fm,
        mode=mode,
        axis=axis,
        # Same default as mode_launch: project the readout where the mode was
        # SOLVED. A launch and a readout that disagree on the transverse
        # centre produce a plausible-looking but wrong transmission, which is
        # exactly how the 1.96 energy sum arose.
        center_um=center_um if center_um is not None
        else _solved_center_in_plane_frame(mode, axis),
        direction=direction,
        thickness_axis=thickness_axis,
        modes_by_freq=modes_by_freq,
        mode_bank=mode_bank,
        # normal-axis spacing AT THE PLANE (local cell width on a graded axis)
        # → enables the de-stagger by default in mode_power/mode_decomposition.
        dl_um=float(dl) if dl else None,
        simulation=simulation,
        per_freq_modes=per_freq_modes,
    )


def solve_modes_by_freq(
    solver: Any,
    freqs_hz: Iterable[float],
    *,
    mode_index: int = 0,
    **solve_kwargs: Any,
) -> Dict[float, Mode]:
    """Solve the FDE eigenmode at each frequency and return ``{freq_hz: Mode}``,
    ready to hand to :func:`mode_monitor` (or :class:`ModeMonitor`) as
    ``modes_by_freq``, the readout-side per-frequency mode basis.

    A single frozen mode is overlapped per frequency by default; with
    ``modes_by_freq`` each recorded DFT frequency is instead projected onto a
    mode solved AT that frequency, which matters when the modal profile / n_eff
    drifts across a wide band (the same motivation as a broadband mode SOURCE,
    see :func:`mode_source`). This helper automates the per-frequency solve that
    fills that map.

    Parameters
    ----------
    solver:
        A :class:`~photonhub.analysis.modes.ModeSolver` or
        :class:`~photonhub.analysis.vector_modes.VectorModeSolver` carrying the
        waveguide cross-section. It is re-solved on the SAME geometry at each
        frequency via ``solver.at_wlen(C0 / f * 1e6)`` (the eps is shared
        by reference), so the cross-section is sampled on the mesh once.
    freqs_hz:
        The monitor frequencies (Hz). Typically the same tuple passed as the
        ``ProfileMonitor.freqs_hz`` / ``mode_monitor(freqs_hz=...)``.
    mode_index:
        Which solved mode to keep (0 = fundamental, the descending-``n_eff``
        order ``solve`` returns), counted at the band's middle frequency. The
        mode is followed across the band by field overlap with its
        neighbouring frequency and phase-aligned to it (the middle
        frequency's mode keeps the solver's own phase), so it keeps its
        identity through a crossing in ``n_eff`` (where the solver's order
        swaps) and its sign does not flip. Where it mixes with another mode
        near a crossing no index names one physical mode; that warns (a link
        overlap below 0.9, or a TE fraction that changes by more than 0.5).
        The branch must support the same mode at every frequency.
    **solve_kwargs:
        Forwarded to ``solver.solve`` (e.g. ``polarization="TM"`` for the
        scalar solver, ``num_modes=...``). ``num_modes`` is bumped to at least
        ``mode_index + 1`` so the requested mode is available, and by two more
        on a band so the mode can be followed past a crossing.

    Returns
    -------
    dict[float, Mode]
        ``{freq_hz: Mode}`` in the input order. Cost: one CPU FDE solve per
        frequency.
    """
    freqs = [float(f) for f in freqs_hz]
    if not freqs:
        raise ValueError("freqs_hz must be non-empty")
    if mode_index < 0:
        raise ValueError(f"mode_index must be >= 0, got {mode_index}")
    num_modes = max(int(solve_kwargs.pop("num_modes", 1)), mode_index + 1)
    bank = _followed_bank(solver, freqs, [mode_index], num_modes, solve_kwargs)
    return {f: bank[f][mode_index] for f in freqs}


def _followed_bank(solver, freqs, idxs, num_modes, solve_kwargs):
    """``{freq_hz: {index: Mode}}``: the modes ``idxs`` of the band's middle
    frequency followed across ``freqs`` by field overlap and phase-aligned
    (:func:`~photonhub.analysis.mode_tracking._follow_bank`). A band solves two
    modes beyond ``num_modes`` so a followed mode that drops in the ``n_eff``
    order is still among the candidates."""
    from .mode_tracking import _follow_bank

    band = sorted(set(freqs))
    extra = 2 if len(band) > 1 else 0
    frames = []
    for f in band:
        if not f > 0.0:
            raise ValueError(f"frequencies must be > 0 Hz, got {f}")
        wavelength_um = C0 / f * 1e6
        modes = solver.at_wlen(wavelength_um).solve(
            num_modes=num_modes + extra, **solve_kwargs
        )
        if idxs[-1] >= len(modes):
            raise ValueError(
                f"requested mode_index {idxs[-1]} but the solver returned only "
                f"{len(modes)} mode(s) at {f:.4g} Hz "
                f"({wavelength_um:.4f} um) — the waveguide may not support it "
                "across the whole band"
            )
        frames.append(list(modes))
    followed = _follow_bank(frames, idxs, band)
    return {f: {i: frame[i] for i in idxs} for f, frame in zip(band, followed)}


def solve_mode_bank(
    solver: Any,
    freqs_hz: Iterable[float],
    *,
    mode_indices: Iterable[int] = (0,),
    **solve_kwargs: Any,
) -> Dict[float, Dict[int, Mode]]:
    """Solve SEVERAL FDE eigenmodes at EACH frequency and return the multi-mode
    mode mapping ``{freq_hz: {mode_index: Mode}}``, ready to hand to :func:`mode_monitor`
    (or :class:`ModeMonitor`) as ``mode_bank`` for :meth:`ModeMonitor.mode_decomposition`.

    This is the multi-mode generalization of :func:`solve_modes_by_freq` (which
    keeps only a single ``mode_index`` per frequency) and the readout-side
    multi-mode, multi-frequency basis: it gives the full guided-mode basis ``mode_indices`` at every
    monitor frequency, so a recorded plane can be decomposed into per-mode powers
    (the fundamental vs higher-order content) with the correct per-(mode, λ)
    profile and ``n_eff``.

    Parameters
    ----------
    solver:
        A :class:`~photonhub.analysis.modes.ModeSolver` or
        :class:`~photonhub.analysis.vector_modes.VectorModeSolver` carrying the
        waveguide cross-section (re-solved per frequency via ``at_wavelength``;
        the eps is shared by reference, so it is sampled on the mesh once).
    freqs_hz:
        The monitor frequencies (Hz), typically the ``ProfileMonitor`` /
        ``mode_monitor`` frequencies.
    mode_indices:
        Which solved modes to keep, in the descending-``n_eff`` order ``solve``
        returns (``0`` = fundamental), counted at the band's middle frequency.
        Each is followed across the band by field overlap with its
        neighbouring frequency and phase-aligned to it (the middle
        frequency's modes keep the solver's own phase), so an index keeps its
        mode through a crossing in ``n_eff`` (where the solver's order swaps).
        A mode that mixes with another near a crossing warns (a link overlap
        below 0.9, or a TE fraction that changes by more than 0.5). Duplicates are collapsed
        and the result is sorted ascending. ``num_modes`` is bumped to at least
        ``max(mode_indices) + 1`` so every requested mode is available, and by
        two more on a band.
    **solve_kwargs:
        Forwarded to ``solver.solve`` (e.g. ``polarization="TE"`` for the scalar
        solver, ``n_guess=...``).

    Returns
    -------
    dict[float, dict[int, Mode]]
        ``{freq_hz: {mode_index: Mode}}`` in the input frequency order, each
        inner dict carrying the requested ``mode_indices`` (ascending). Cost: one
        CPU FDE solve per frequency (each returns all requested modes at once).
    """
    freqs = [float(f) for f in freqs_hz]
    if not freqs:
        raise ValueError("freqs_hz must be non-empty")
    idxs = sorted({int(i) for i in mode_indices})
    if not idxs:
        raise ValueError("mode_indices must be non-empty")
    if idxs[0] < 0:
        raise ValueError(f"mode_indices must be >= 0, got {idxs[0]}")
    num_modes = max(int(solve_kwargs.pop("num_modes", 1)), idxs[-1] + 1)
    bank = _followed_bank(solver, freqs, idxs, num_modes, solve_kwargs)
    return {f: bank[f] for f in freqs}
