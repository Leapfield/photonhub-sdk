"""Mode source & monitor builders — the client bridge from an FDE eigenmode to
the engine's ModeSource (NUMERICS.md §18) and to a mode-resolved transmission
readout.

``mode_source`` resamples a frozen FDE :class:`~photonhub.plugins.modes.Mode`
onto a simulation's transverse grid plane and returns a
:class:`~photonhub.components.sources.ModeSource` the engine injects via TF/SF.
``mode_monitor`` returns a :class:`ModeMonitor`, which carries a 4-tangential
``FieldDftMonitor`` to add to the simulation and a ``.transmission(data)``
post-process that overlaps the recorded plane onto the mode (forward/backward
power ``T``) via :func:`photonhub.plugins.mode_overlap.mode_transmission`.

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
from ..components.grid import (graded_primary_spacings, realized_cells,
                               snap_mixed_plane)
from ..components.monitors import FieldDftMonitor
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

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _axis_cell_centers(simulation, axis_name: str) -> np.ndarray:
    """Transverse cell-center coordinates (microns) along one axis.

    A uniform axis uses ``(i + 0.5)·dl``. A GRADED axis (GradedGridSpec coords)
    uses the midpoints of its primary-node cells (the §15.2 dual nodes), so the
    mode profile is sampled at the TRUE cell centers — this is what lets a mode
    source / monitor live on a transverse-graded mesh. The §18 auxiliary line
    also supports a graded propagation axis (NUMERICS.md §15.9); this helper
    samples only the two transverse plane axes because the source profile is
    defined on that plane."""
    idx = _AXIS_IDX[axis_name]
    q = simulation._axis_coords_um(idx)
    if q is None:  # uniform axis (UniformGridSpec, or a non-graded graded axis)
        dl = simulation.grid.dl_um
        size = simulation.size_um[idx]
        n = realized_cells(size, dl)
        return (np.arange(n) + 0.5) * dl
    # Graded axis: cell i spans [q[i], q[i+1]] (q[n] = §15.1 replicate-last
    # closing node), so its center is q[i] + dq[i]/2 with dq the primary
    # spacings (replicate-last for the final cell). Matches the engine's §15.2
    # dual-node convention, so the resampled profile lands on the cells the
    # solver injects into.
    qa = np.asarray(q, dtype=float)
    dq = np.asarray(graded_primary_spacings(tuple(q)), dtype=float)
    return qa + dq / 2.0


def _default_center(simulation, axis: str) -> Tuple[float, float]:
    """The transverse domain midpoints (t1, t2) — where a centered waveguide
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

    Returns ``{}`` for fewer than two entries — the legacy single-mode launch.
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
    at that frequency — not the single band-centre H, which is correct only at the
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
    """A full-vector mode carries the true paired H (``hx``/``hy``) — e.g. a
    :class:`~photonhub.plugins.vector_modes.VectorMode` (incl. the engine-consistent
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
        The §18 aux-line ModeSource is deprecated in favour of the equivalence-
        current launch: prefer :func:`mode_launch` with a discrete Yee mode
        (:func:`~photonhub.plugins.yee_mode.solve_yee_mode` /
        :func:`~photonhub.plugins.kfj_smoothing.solve_mode_on_cross_section`). The
        Huygens dipole-sheet launch works on uniform AND graded grids, supports
        broadband, and sheds less near-source radiation. Full-vector calls here
        delegate to :func:`mode_source_vector` (which emits the deprecation
        warning); §18 is retained for the adjoint and scalar/FLM modes.

    **Full-vector launch is the default (NUMERICS.md §18.2a / launch_fidelity).**
    When ``mode`` is a full-vector mode (it carries the true paired ``H`` — e.g. a
    :class:`~photonhub.plugins.vector_modes.VectorMode`, especially the engine's own
    ``yee_mode`` discrete eigenmode) and ``paired_h`` is True (default), this
    delegates to :func:`mode_source_vector` so the source ships the mode's TRUE
    discrete paired-H (``profile_h``). That makes the launch the discrete
    full-vector equivalent-current source (J=n×H_true, M=−n×E_true) instead of the
    scalar-impedance-H limit, which cuts the near-source radiation several-fold (a
    controlled study is in ``benchmarks/launch_fidelity/``; Tidy3D/Meep use this
    construction). The launch is then power-normalized (``power_watts = amplitude²``
    so a non-unit ``amplitude`` still scales power as a peak-field would). Pass
    ``paired_h=False`` to force the legacy scalar-limit launch, or pass a scalar
    :class:`Mode` (no H) — both give the prior single-component behavior.

    The mode's major transverse-E profile is resampled (peak-normalized) onto
    the grid's transverse cells; ``amplitude`` is then the peak injected field.
    ``center_um`` places the waveguide in the transverse plane (default: the
    domain center, i.e. a centered guide). ``thickness_axis`` is the simulation
    axis along the guide's slab thickness; pass the slab normal (e.g. ``"z"``)
    for any non-x propagation so the mode is not rotated 90 degrees (see
    :func:`~photonhub.plugins.mode_overlap.modal_fields`). ``None`` keeps the
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
    # path below, byte-identical to before.
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
    u_coords = _axis_cell_centers(simulation, t1_name)
    v_coords = _axis_cell_centers(simulation, t2_name)
    if center_um is None:
        center_um = _default_center(simulation, axis)

    def _resample(m: Mode):
        """Peak-normalized major-E profile (flat C-order) + its polarization,
        resampled onto this plane — the shared scalar-source readout."""
        fields = modal_fields(
            m, u_coords, v_coords, axis=axis, n_eff=m.n_eff,
            center_um=center_um, thickness_axis=thickness_axis,
        )
        # The major-E component is whichever of e1/e2 modal_fields filled (the
        # other is identically zero); read it back rather than re-deriving.
        if np.any(fields["e1"]):
            profile2d, pol = fields["e1"], "E" + t1_name  # [iv, iu]
        else:
            profile2d, pol = fields["e2"], "E" + t2_name
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
        nu=int(u_coords.size),
        nv=int(v_coords.size),
        profile=tuple(float(v) for v in profile),
        source_time=source_time,
        **bb,
    )


def _launch_window_origin(mode, h_center, v_center):
    """The EXACT window origin ``(h_lo, v_lo)`` the mode was solved on,
    recovered from its own recorded placement so the launch registers on the
    solve grid without threading the original window through. Inverts
    :func:`~photonhub.plugins.yee_mode._window_center_offset`
    (``off = lo + 0.5(n-1)dl - center``): ``lo = center + off - 0.5(n-1)dl``,
    all exact grid multiples — passed straight to the sheet builder, so no
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
    (a single frozen band centre OR a per-frequency broadband bank of them —
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
    """The library's DEFAULT mode launch — returns a LIST of sources to drop
    into ``Simulation.sources``.

    ``launch='auto'`` (default) injects ``mode`` as a **per-cell equivalence-
    current Huygens sheet** (``J = n̂×H``, ``M = −n̂×E`` phased ``PointDipole``
    sheets from the discrete Yee mode) whenever that is eligible: a full-vector
    Yee mode with recorded grid placement, including supported broadband banks
    and graded grids. It falls back to the §18 aux-line :class:`ModeSource` for
    scalar/FLM modes and other ineligible inputs. The eq-current launch's
    near-source radiation falls
    with resolution where §18 floors (0.44→0.09% at 20 nm on a straight strip;
    see ``benchmarks/launch_fidelity``), and it is what Tidy3D/Meep do; it also
    honors §20 symmetry planes automatically. Returns a list because the sheet
    is many dipoles while §18 is one component — a single call, either way.

    ``launch='eq_current'`` forces the sheet and RAISES if ineligible;
    ``launch='aux'`` forces §18. ``center_um`` is the transverse waveguide
    location in the cut's ``(horizontal, vertical)`` in-plane-axis order
    (default: the domain centre); it is reordered internally for the §18
    builder's own convention. ``power_watts`` is the launched modal power
    (into the HALF domain when a §20 symmetry plane is present). The window
    the sheet stamps is recovered from the mode's own recorded placement, so
    the launch registers on the solve grid with nothing to thread through.

    NB the continuous-adjoint pipeline deliberately keeps the §18
    :func:`mode_source` (its gradient constant is pinned to that excitation) —
    it does not route through here."""
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
        h_center = simulation.size_um[_AXIS_IDX[h_letter]] / 2.0
        v_center = simulation.size_um[_AXIS_IDX[v_letter]] / 2.0
    else:
        h_center, v_center = float(center_um[0]), float(center_um[1])

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
                power_watts=power_watts, modes_by_freq=bank)
        origin = _launch_window_origin(mode, h_center, v_center)
        return equivalence_current_source(
            simulation, mode, axis=axis, position_um=position_um,
            source_time=source_time, direction=direction,
            h_center_um=h_center, v_center_um=v_center,
            half_w_um=0.0, half_v_um=0.0, power_watts=power_watts,
            _window_origin=origin, modes_by_freq=bank)

    # §18 fallback. mode_source's center_um is in the _TRANSVERSE (t1, t2)
    # order, which SWAPS vs in_plane_axes for a y-cut — key by axis letter so
    # the reorder is correct for every propagation axis.
    coord = {h_letter: h_center, v_letter: v_center}
    t1, t2 = _TRANSVERSE[axis]
    return [mode_source(
        simulation, mode, axis=axis, position_um=position_um,
        source_time=source_time, direction=direction,
        amplitude=math.sqrt(float(power_watts)),
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
    ``power_watts`` (default **1 W**). Both transverse-E profiles are resampled
    onto the grid's transverse cells preserving their true component ratio (via
    :func:`~photonhub.plugins.mode_overlap.vector_modal_fields`); the minor
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
    the plane's real cell areas, and scale BOTH packed profiles by
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
    modal power differs from ``power_watts`` by that correction — transmission
    RATIOS cancel it (both planes read the same launch), only the absolute
    wattage carries the bias. Left as-is pending an engine-side verification
    of the injected-power convention.

    **Broadband injection (``num_freqs`` analogue, NUMERICS.md §18.3).** Pass
    ``modes_by_freq`` (``{freq_hz: VectorMode}`` from :func:`solve_modes_by_freq`
    over a :class:`~photonhub.plugins.vector_modes.VectorModeSolver`) to inject a
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
    u_coords = _axis_cell_centers(simulation, t1_name)
    v_coords = _axis_cell_centers(simulation, t2_name)
    if center_um is None:
        center_um = _default_center(simulation, axis)
    # dA from the plane's real transverse cell widths (uniform here, but use the
    # shared quadrature so this stays correct if the plane ever grades).
    dA_m2 = np.outer(_cell_widths(v_coords), _cell_widths(u_coords)) * 1e-12

    def _resample(m):
        """Power-normalized (major, minor) real profiles + major polarization,
        resampled onto this plane — the shared full-vector source readout."""
        fv = vector_modal_fields(
            m, u_coords, v_coords, axis=axis, direction=direction,
            center_um=center_um, thickness_axis=thickness_axis,
        )
        e1, e2 = fv["e1"], fv["e2"]  # transverse-E along t1, t2 ([iv, iu])
        # The MAJOR transverse axis carries the larger transverse-E energy.
        if float(np.sum(np.abs(e1) ** 2)) >= float(np.sum(np.abs(e2) ** 2)):
            e_major, pol_maj, e_minor = e1, "E" + t1_name, e2
        else:
            e_major, pol_maj, e_minor = e2, "E" + t2_name, e1
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
            np.sum((maj ** 2 + minr ** 2) * dA_m2)
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
        to +profile in the scalar limit (matching the legacy engine path) — the
        deviation IS the true-H correction the engine's E-correction needs to stop
        radiating the scalar-limit-H mismatch (~few %). Called once per band-centre
        and once per broadband carrier (each at its own frequency's n_eff)."""
        fv = vector_modal_fields(m, u_coords, v_coords, axis=axis,
                                 direction=direction, center_um=center_um,
                                 thickness_axis=thickness_axis)
        # SAME major/minor criterion as _resample above (complex transverse-E
        # energy) — a real-part criterion could route the E and H profiles to
        # opposite axes for a mode with residual imaginary content; for the
        # lossless co-real modes this targets the two coincide.
        major_t1 = (float(np.sum(np.abs(fv["e1"]) ** 2))
                    >= float(np.sum(np.abs(fv["e2"]) ** 2)))
        e1, e2, h1, h2 = (np.real(fv[k]) for k in ("e1", "e2", "h1", "h2"))
        e_maj, e_min = (e1, e2) if major_t1 else (e2, e1)
        h_maj, h_min = (h2, h1) if major_t1 else (h1, h2)  # E_t pairs with H of the OTHER axis
        p_unit = (float(m.n_eff) / (2.0 * ETA0)) * float(
            np.sum((e_maj ** 2 + e_min ** 2) * dA_m2))
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
        nu=int(u_coords.size),
        nv=int(v_coords.size),
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


@dataclass(frozen=True)
class ModeMonitor:
    """A mode-resolved transmission monitor: a 4-tangential ``FieldDftMonitor``
    (add ``.field_monitor`` to the simulation) plus a ``.transmission(data)``
    post-process that overlaps the recorded plane onto ``mode``."""

    field_monitor: FieldDftMonitor
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
    #: own-frequency mode — Tidy3D's ``ModeMonitor`` convention. Silently keeps
    #: the frozen band-centre mode when the provenance or ``simulation`` is
    #: missing. Set False for the legacy frozen-mode readout.
    per_freq_modes: bool = True

    @property
    def name(self) -> str:
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
        the AUTO bank only — each bank mode carries its own n_eff, which would
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
        weights half a cell instead of spilling ``dl/2`` into the mirror half —
        the parity-asymmetric power inflation that under-read T for
        cross-parity port pairs (fold-antinode in, fold-node out). Without a
        stored simulation there is no fold information and no correction."""
        sim = self.simulation
        sym = getattr(sim, "symmetry", None) if sim is not None else None
        if not sym:
            return (False, False)
        t1, t2 = _TRANSVERSE[self.axis]
        return (sym[_AXIS_IDX[t1]] != 0, sym[_AXIS_IDX[t2]] != 0)

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
        """The forward (or backward) modal **power** ``{freq_hz: |a_pm|²/P_mode}``
        on this plane — the actual power carried by ``mode`` through it, in the
        run's (source-spectrum-normalized) units. This is NOT a 0–1 transmission
        on its own; ratio two planes for that (see :func:`transmission`).

        Returns true *power* (``|c|²·P_mode``), not the bare squared amplitude
        ``|c|²``, so that ``P_out / P_in`` is the correct power transmission even
        when the two ports carry **different** modes (e.g. a w1→w2 taper, where the
        per-mode ``P_mode`` differs and must not cancel). For same-mode ratios (a
        uniform-width straight, or a reflection ``-``/``+`` at one plane) the
        ``P_mode`` cancels, so those readings are unchanged. ``data`` is the
        ``SimulationData`` from the run; ``data[self.name]`` is the recorded DFT
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
        :func:`~photonhub.plugins.mode_overlap.mode_transmission`): when ``colocate``
        is True it uses the monitor's grid ``dl_um`` automatically, matching what
        Tidy3D's ``ModeMonitor(colocate=True)`` does when it interpolates the
        staggered Yee components to common coordinates. Pass ``destagger_dl=None``
        to force it off (e.g. for already-co-located synthetic fields), or a float
        to override the spacing."""
        if destagger_dl is _DESTAGGER_AUTO:
            destagger_dl = self.dl_um if colocate else None
        da = data[self.name]
        planes: Mapping[str, object] = {
            c: da.sel(component=c) for c in _TANGENTIAL[self.axis]
        }
        mbf = self._resolved_modes_by_freq(modes_by_freq, n_eff=n_eff)
        return mode_transmission(
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
        {freq_hz: value}}`` (Tidy3D ``ModeMonitor`` with ``num_modes``).

        Projects the plane onto every mode in the bank (each index, each
        frequency) instead of the single ``self.mode`` that :meth:`mode_power`
        uses. The bank is ``mode_bank`` if given, else the monitor's stored
        ``self.mode_bank``; it is ``{freq_hz: {mode_index: Mode}}`` (per-frequency,
        dispersive — see :func:`solve_mode_bank`) or ``{mode_index: Mode}``
        (frozen). ``quantity`` selects ``"transmission"`` (``|c|²``, default),
        ``"power"`` (``|a_pm|²/P_mode``, the per-mode power to ratio across ports),
        or ``"amplitude"`` (complex ``c``, for a multimode S-matrix). See
        :func:`~photonhub.plugins.mode_overlap.mode_decomposition`."""
        bank = mode_bank if mode_bank is not None else self.mode_bank
        if not bank:
            raise ValueError(
                "no mode_bank: pass mode_bank=... or build the ModeMonitor with "
                "one (see solve_mode_bank); mode_decomposition needs >1 mode")
        if destagger_dl is _DESTAGGER_AUTO:  # de-stagger ON by default (see mode_power)
            destagger_dl = self.dl_um if colocate else None
        da = data[self.name]
        planes: Mapping[str, object] = {
            c: da.sel(component=c) for c in _TANGENTIAL[self.axis]
        }
        return mode_decomposition(
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
    ``out_monitor`` — the ratio of modal powers, which cancels the source and
    spectrum normalization so a lossless single-mode straight guide reads
    ``T ≈ 1``. Place ``in_monitor`` just after the source (total-field side) and
    ``out_monitor`` at the device output.

    ``direction=None`` (default) reads each monitor in its OWN stored
    ``direction`` (so e.g. an out-monitor built with ``direction="-"`` — a port
    facing the source — is read backward, as placed). An explicit ``"+"``/``"-"``
    overrides BOTH planes with that one direction (it used to be the silent
    default, flipping a ``"-"`` out-monitor to a forward read).

    The longitudinal Yee de-stagger is applied by default (each monitor uses
    its own grid ``dl_um`` when ``colocate=True``) — it removes the input-plane
    standing-wave ripple and matches Tidy3D's ``colocate=True`` convention;
    pass ``destagger_dl=None`` to force it off. See
    :meth:`ModeMonitor.mode_power`."""
    p_in = in_monitor.mode_power(data, direction=direction, n_eff=n_eff,
                                 colocate=colocate, destagger_dl=destagger_dl)
    p_out = out_monitor.mode_power(data, direction=direction, n_eff=n_eff,
                                   colocate=colocate, destagger_dl=destagger_dl)
    return {f: p_out[f] / p_in[f] for f in p_out if f in p_in}


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
    """Build a :class:`ModeMonitor` (a 4-tangential ``FieldDftMonitor`` on the
    ``axis`` plane at ``position_um`` + a transmission post-process onto
    ``mode``). Add ``.field_monitor`` to the simulation's monitors, run, then
    call ``.transmission(data)``. ``thickness_axis`` is the slab-normal axis
    (pass e.g. ``"z"`` for non-x propagation so the overlap mode is not rotated
    90 degrees); ``None`` keeps the legacy mapping. Pass ``mode_bank``
    (``{freq_hz: {mode_index: Mode}}``, see :func:`solve_mode_bank` /
    :func:`~photonhub.plugins.yee_mode.solve_yee_multimode_bank`) to enable
    :meth:`ModeMonitor.mode_decomposition` (multi-mode readout). See
    :func:`mode_source`.

    ``per_freq_modes`` (default True): when no ``modes_by_freq`` is given and
    ``mode`` came from ``solve_mode_on_cross_section`` (it carries its solve
    provenance), the monitor re-solves the mode at EVERY ``freqs_hz`` on first
    use and projects each recorded frequency onto its own-frequency mode —
    the per-λ readout Tidy3D's ``ModeMonitor`` performs, now the default here
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
    # which GradedGridSpec also carries) feeds the longitudinal de-stagger.
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
    fm = FieldDftMonitor(
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
        center_um=center_um,
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
    ``modes_by_freq`` — the readout-side analogue of Tidy3D's ``num_freqs``.

    A single frozen mode is overlapped per frequency by default; with
    ``modes_by_freq`` each recorded DFT frequency is instead projected onto a
    mode solved AT that frequency, which matters when the modal profile / n_eff
    drifts across a wide band (the same motivation as a broadband mode SOURCE,
    see :func:`mode_source`). This helper automates the per-frequency solve that
    fills that map.

    Parameters
    ----------
    solver:
        A :class:`~photonhub.plugins.modes.ModeSolver` or
        :class:`~photonhub.plugins.vector_modes.VectorModeSolver` carrying the
        waveguide cross-section. It is re-solved on the SAME geometry at each
        frequency via ``solver.at_wavelength(C0 / f * 1e6)`` (the eps is shared
        by reference), so the cross-section is rasterized once.
    freqs_hz:
        The monitor frequencies (Hz). Typically the same tuple passed as the
        ``FieldDftMonitor.freqs_hz`` / ``mode_monitor(freqs_hz=...)``.
    mode_index:
        Which solved mode to keep per frequency (0 = fundamental, the
        descending-``n_eff`` order ``solve`` returns). The branch must support
        the same mode at every frequency.
    **solve_kwargs:
        Forwarded to ``solver.solve`` (e.g. ``polarization="TM"`` for the
        scalar solver, ``num_modes=...``). ``num_modes`` is bumped to at least
        ``mode_index + 1`` so the requested mode is available.

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
    out: Dict[float, Mode] = {}
    for f in freqs:
        if not f > 0.0:
            raise ValueError(f"frequencies must be > 0 Hz, got {f}")
        wavelength_um = C0 / f * 1e6
        modes = solver.at_wavelength(wavelength_um).solve(
            num_modes=num_modes, **solve_kwargs
        )
        if mode_index >= len(modes):
            raise ValueError(
                f"requested mode_index {mode_index} but the solver returned only "
                f"{len(modes)} mode(s) at {f:.4g} Hz "
                f"({wavelength_um:.4f} um) — the waveguide may not support it "
                "across the whole band"
            )
        out[f] = modes[mode_index]
    return out


def solve_mode_bank(
    solver: Any,
    freqs_hz: Iterable[float],
    *,
    mode_indices: Iterable[int] = (0,),
    **solve_kwargs: Any,
) -> Dict[float, Dict[int, Mode]]:
    """Solve SEVERAL FDE eigenmodes at EACH frequency and return the multi-mode
    bank ``{freq_hz: {mode_index: Mode}}`` — ready to hand to :func:`mode_monitor`
    (or :class:`ModeMonitor`) as ``mode_bank`` for :meth:`ModeMonitor.mode_decomposition`.

    This is the multi-mode generalization of :func:`solve_modes_by_freq` (which
    keeps only a single ``mode_index`` per frequency) and the readout-side
    analogue of Tidy3D's ``ModeMonitor(mode_spec=ModeSpec(num_modes=N),
    num_freqs=M)``: it gives the full guided-mode basis ``mode_indices`` at every
    monitor frequency, so a recorded plane can be decomposed into per-mode powers
    (the fundamental vs higher-order content) with the correct per-(mode, λ)
    profile and ``n_eff``.

    Parameters
    ----------
    solver:
        A :class:`~photonhub.plugins.modes.ModeSolver` or
        :class:`~photonhub.plugins.vector_modes.VectorModeSolver` carrying the
        waveguide cross-section (re-solved per frequency via ``at_wavelength``;
        the eps is shared by reference, so it is rasterized once).
    freqs_hz:
        The monitor frequencies (Hz) — typically the ``FieldDftMonitor`` /
        ``mode_monitor`` frequencies.
    mode_indices:
        Which solved modes to keep per frequency, in the descending-``n_eff``
        order ``solve`` returns (``0`` = fundamental). Duplicates are collapsed
        and the result is sorted ascending. ``num_modes`` is bumped to at least
        ``max(mode_indices) + 1`` so every requested mode is available.
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
    out: Dict[float, Dict[int, Mode]] = {}
    for f in freqs:
        if not f > 0.0:
            raise ValueError(f"frequencies must be > 0 Hz, got {f}")
        wavelength_um = C0 / f * 1e6
        modes = solver.at_wavelength(wavelength_um).solve(
            num_modes=num_modes, **solve_kwargs
        )
        if idxs[-1] >= len(modes):
            raise ValueError(
                f"requested mode_index {idxs[-1]} but the solver returned only "
                f"{len(modes)} mode(s) at {f:.4g} Hz ({wavelength_um:.4f} um) — "
                "the waveguide may not support it across the whole band"
            )
        out[f] = {i: modes[i] for i in idxs}
    return out
