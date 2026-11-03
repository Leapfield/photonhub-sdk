"""Field monitors: time-domain and frequency-domain.

Monitor names must be filename-safe (the engine writes ``<name>.bin``) and, a constraint JSON Schema cannot express across array items, unique within a
simulation (enforced by ``Simulation``).
"""

import math
from typing import Annotated, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from ..constants import c0
from .base import (
    _MONITOR_NAME_MAX_LENGTH,
    MAX_INT32,
    AxisName,
    DirectionName,
    FieldComponentName,
    FreqHz,
    FrozenModel,
    MonitorName,
    NonNegativeUm,
    PositiveUm,
    Vec3Um,
)


class TimeMonitor(FrozenModel):
    """Scalar time-series probe at the Yee node nearest ``center_um``. Samples
    are raw, non-colocated Yee values; H lags E by dt/2."""

    type: Literal["field_time"] = "field_time"
    name: MonitorName
    center_um: Vec3Um
    fields: Tuple[FieldComponentName, ...] = Field(min_length=1)
    interval_steps: int = Field(default=1, ge=1, le=MAX_INT32)

    @field_validator("fields")
    @classmethod
    def _unique_fields(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("monitor fields must be unique")
        return value


class SnapshotMonitor(FrozenModel):
    """Full-domain dump of selected components. ``interval_steps = 0`` (the
    default) records only the final step."""

    type: Literal["field_snapshot"] = "field_snapshot"
    name: MonitorName
    fields: Tuple[FieldComponentName, ...] = Field(min_length=1)
    interval_steps: int = Field(default=0, ge=0, le=MAX_INT32)

    @field_validator("fields")
    @classmethod
    def _unique_fields(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("monitor fields must be unique")
        return value


class Apodization(FrozenModel):
    """Time window applied to a monitor's running DFT (the standard
    ``ApodizationSpec`` analogue). A Gaussian roll-ON of
    standard deviation ``width_s`` for t < ``start_s``, flat (== 1) on
    ``[start_s, end_s]``, and a Gaussian roll-OFF for t > ``end_s``, used to
    isolate the late-time steady state of a resonant structure (suppressing the
    source-injection transient). ``start_s``/``end_s`` (seconds) are each
    optional, omit a side to leave it ungated; ``width_s`` (seconds) is the
    Gaussian standard deviation of each roll and must be positive."""

    start_s: Optional[float] = Field(default=None, ge=0.0)
    end_s: Optional[float] = Field(default=None, ge=0.0)
    width_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self):
        if (
            self.start_s is not None
            and self.end_s is not None
            and self.end_s < self.start_s
        ):
            raise ValueError(
                f"end_s ({self.end_s}) must be >= start_s ({self.start_s})"
            )
        return self


class PortMode(FrozenModel):
    """One polarization-family channel requested on a :class:`ModePort`."""

    polarization: Literal["TE", "TM"]
    mode_index: int = Field(ge=0, le=31)


MODE_PORT_MAX_TRIAL_MODES = 32

# Free-space speed of light (m/s), identical to the engine's kC0 and to
# source_time.py, so ``wlens_um`` converts bit-comparably with ``for_band``.
_C0_M_PER_S = c0


def _monitor_freqs(wlens_um, freqs_hz, who: str) -> Tuple[float, ...]:
    """The frequencies of a monitor given as ``wlens_um`` (microns, one value
    or several) or ``freqs_hz``, exactly one of them."""
    if (wlens_um is None) == (freqs_hz is None):
        raise ValueError(f"{who}: pass exactly one of wlens_um or freqs_hz")
    if wlens_um is not None:
        wl = (float(wlens_um),) if isinstance(wlens_um, (int, float)) else tuple(float(w) for w in wlens_um)
        if any(w <= 0.0 for w in wl):
            raise ValueError(f"{who}: wlens_um must be positive")
        return tuple(_C0_M_PER_S / (w * 1e-6) for w in wl)
    return tuple(float(f) for f in freqs_hz)


def mode_port_solver_polarization(
    polarization: str,
    normal_axis: str,
    thickness_axis: Optional[str] = None,
) -> str:
    """Map a physical port TE/TM label to the Yee solver's natural family.

    The Yee eigensolver calls electric field along the natural horizontal
    in-plane axis TE.  A modal port instead calls electric field along the
    waveguide width TE, where width is orthogonal to ``thickness_axis``.  The
    two labels therefore swap when thickness occupies the natural horizontal
    axis.  This transform is an involution, so
    :func:`mode_port_physical_polarization` applies the same swap in reverse.
    """
    family = str(polarization).upper()
    if family not in {"TE", "TM"}:
        raise ValueError(
            f"mode polarization must be TE or TM, got {polarization!r}")
    if normal_axis not in {"x", "y", "z"}:
        raise ValueError(
            f"normal_axis must be x, y, or z, got {normal_axis!r}")
    natural_axes = tuple(axis for axis in ("x", "y", "z")
                         if axis != normal_axis)
    resolved_thickness = thickness_axis or natural_axes[1]
    if resolved_thickness not in natural_axes:
        raise ValueError(
            f"thickness_axis {resolved_thickness!r} must be transverse to "
            f"{normal_axis!r}")
    if resolved_thickness == natural_axes[0]:
        return "TM" if family == "TE" else "TE"
    return family


def mode_port_physical_polarization(
    solver_polarization: str,
    normal_axis: str,
    thickness_axis: Optional[str] = None,
) -> str:
    """Map a Yee solver-family label to the port's physical TE/TM label."""
    return mode_port_solver_polarization(
        solver_polarization, normal_axis, thickness_axis)


def mode_port_required_trial_modes(modes) -> int:
    """Minimum total eigensolver trials that can cover ``modes``.

    ``mode_index`` is counted independently inside each TE/TM family, while
    the Yee eigensolver's ``num_modes`` is a total frame size.  Covering TE1
    and TM0 therefore needs at least three trial eigenpairs: two TE ranks and
    one TM rank.
    """
    highest_by_family: dict[str, int] = {}
    for mode in modes:
        if isinstance(mode, PortMode):
            polarization = mode.polarization
            mode_index = int(mode.mode_index)
        else:
            polarization, mode_index = mode
            polarization = str(polarization).upper()
            mode_index = int(mode_index)
        highest_by_family[polarization] = max(
            highest_by_family.get(polarization, -1), mode_index)
    return sum(index + 1 for index in highest_by_family.values())


def mode_port_trial_modes(modes, num_modes: Optional[int]) -> int:
    """Resolve a feasible explicit/automatic total Yee trial count."""
    required = mode_port_required_trial_modes(modes)
    if required > MODE_PORT_MAX_TRIAL_MODES:
        raise ValueError(
            "requested polarization-family mode indices require "
            f"{required} trial modes, exceeding the maximum of "
            f"{MODE_PORT_MAX_TRIAL_MODES}"
        )
    if num_modes is not None:
        if isinstance(num_modes, bool):
            raise ValueError("num_modes must be an integer")
        resolved = int(num_modes)
        if not 1 <= resolved <= MODE_PORT_MAX_TRIAL_MODES:
            raise ValueError(
                "num_modes must be between 1 and "
                f"{MODE_PORT_MAX_TRIAL_MODES}, got {resolved}"
            )
        if resolved < required:
            raise ValueError(
                f"num_modes must be at least {required} to cover the requested "
                "polarization-family mode indices"
            )
        return resolved
    # Preserve the established six-mode search and two-mode headroom, but cap
    # the automatic value instead of making a valid highest-rank request ask
    # the 32-mode solver for 34 eigenpairs.
    return min(
        MODE_PORT_MAX_TRIAL_MODES,
        max(6, required + 2),
    )
class ModePort(FrozenModel):
    """Authoring recipe that turns a DFT field plane into a modal port.

    This block has no native time-stepping semantics. The engine validates and
    ignores it; result post-processing solves the requested Yee modes on the
    saved cross-section and projects this monitor's four tangential fields onto
    them. ``out_direction`` points away from the device. ``source_index`` marks
    the one driven port in the current run, whose incident direction is the
    opposite sign.

    ``center_um`` / ``size_um`` use the natural horizontal/vertical axes of the
    monitor plane (x-normal -> y,z; y-normal -> x,z; z-normal -> x,y).
    ``TE`` means electric field primarily along waveguide width (orthogonal to
    ``thickness_axis``), not necessarily the Yee solver's natural-horizontal
    family.
    ``num_modes`` is the total eigensolver frame size even though each requested
    ``mode_index`` is counted independently within its TE/TM family.
    """

    solver: Literal["yee"] = "yee"
    out_direction: Literal["+", "-"]
    center_um: Tuple[float, float]
    size_um: Tuple[PositiveUm, PositiveUm]
    dl_um: PositiveUm
    supersample: int = Field(default=8, ge=1, le=16)
    num_modes: Optional[int] = Field(
        default=None, ge=1, le=MODE_PORT_MAX_TRIAL_MODES)
    modes: Tuple[PortMode, ...] = Field(min_length=1)
    source_index: Optional[int] = Field(default=None, ge=0)
    thickness_axis: Optional[AxisName] = None

    @field_validator("modes")
    @classmethod
    def _unique_modes(cls, value):
        keys = [(mode.polarization, mode.mode_index) for mode in value]
        if len(set(keys)) != len(keys):
            raise ValueError("port modes must be unique")
        return value

    @model_validator(mode="after")
    def _trial_mode_count_covers_channels(self) -> "ModePort":
        mode_port_trial_modes(self.modes, self.num_modes)
        return self


class ProfileMonitor(FrozenModel):
    """Running-DFT field monitor over a box region:
    fp64 accumulation every step over the full run, raw Yee-located phasors.
    The engine normalizes them by the first wire-order source's ``A0 * S(f)``
    and :class:`~photonhub.RunResult` multiplies ``A0`` back in, so the
    arrays are the continuous-wave phasors of the sources as declared, in
    V/m and A/m (:class:`PowerMonitor` states the convention). ``size_um``
    components may be 0 (plane/line/point regions); the region is snapped per
    component to that component's Yee sublattice, and the engine validator
    REJECTS boxes whose per-component snaps disagree (the output carries one
    shape/origin per monitor). When ``fields`` mixes Yee offsets along an
    axis, a face belongs strictly between an integer cell boundary and the
    next half-cell plane, canonically ``(k + 0.25) * dl``, which every
    component snaps to cell ``k`` with quarter-cell fp margin.

    You do not have to place faces there yourself: building a
    :class:`~photonhub.Simulation` AUTO-SNAPS every box face that would fail
    (or pass only on float rounding luck) that engine check to the nearest
    quarter-cell plane of its local cell, clamped into the grid, and reports
    each adjustment on the ``photonhub.components.simulation`` DEBUG log.
    The policy is deterministic and minimal: faces already strictly inside a
    first half-cell, axes whose listed components share one Yee offset, and
    faces at/beyond the domain edges (where the engine's index clamp makes
    every component agree, full-domain boxes stay full-domain) are left
    byte-identical, and a face and its quarter point snap to the SAME cell,
    so any scene the engine already accepted keeps its exact recorded
    region. A sub-half-cell box straddling a cell boundary cannot be snapped
    and is rejected with guidance at construction. Client-side only: the
    wire schema is unchanged, documents ingested via ``from_wire_json`` /
    ``from_file`` are never adjusted, and
    ``with_auto_mesh``/``with_mesh_overrides`` re-apply the snap against the
    regenerated grid."""

    type: Literal["field_dft"] = "field_dft"
    name: MonitorName
    center_um: Vec3Um
    size_um: Tuple[NonNegativeUm, NonNegativeUm, NonNegativeUm]
    fields: Tuple[FieldComponentName, ...] = Field(min_length=1)
    freqs_hz: Tuple[FreqHz, ...] = Field(min_length=1)
    # Per-axis spatial sampling stride (schema 1.11.0, additive/optional — the
    # interval_space). None (default) records every cell; (sx, sy, sz)
    # decimates the recorded region along each axis (output cell i -> snapped
    # cell + i*stride), cutting field-monitor output for large planes/volumes.
    # Each stride >= 1. Omitted from the wire when unset (older engines/readers
    # round-trip unchanged); the data layer strides the coordinates to match.
    interval_space: Optional[Tuple[int, int, int]] = None
    # Optional time-apodization of the running DFT (schema 1.13.0, additive —
    # an apodization spec). None (default) => no window, omitted from the
    # wire so older engines/readers round-trip unchanged.
    apodization: Optional[Apodization] = None
    # Schema 1.16 authoring metadata. The resolved execution object remains this
    # ordinary DFT monitor; Workbench/result APIs compile the saved recipe into
    # ModeMonitor/SPort post-processing after a run.
    mode_port: Optional[ModePort] = None
    # Client-side authoring marker (never on the wire, never in the schema):
    # ``"interior:<axis>"`` asks the Simulation to fill the in-plane centre and
    # extent of a :meth:`plane` monitor normal to ``<axis>`` with the PML-free
    # interior of the domain at construction
    # (``Simulation._resolve_plane_spans``). None once resolved.
    span: SkipJsonSchema[Optional[Literal["interior:x", "interior:y", "interior:z"]]] = None
    # Client-side authoring marker, like ``span``: ``(i, n)`` asks the
    # Simulation to place this plane at the i-th of ``n`` stations spread
    # evenly along its normal across the interior (:meth:`sections`). None
    # once resolved, and never on the wire.
    station: SkipJsonSchema[Optional[Tuple[int, int]]] = None

    @classmethod
    def plane(
        cls,
        name: str,
        axis: str,
        position_um: float,
        *,
        wlens_um: Optional[Union[float, Sequence[float]]] = None,
        freqs_hz: Optional[Sequence[float]] = None,
        fields: Tuple[str, ...] = ("Ex", "Ey", "Ez"),
        size_um: Optional[Tuple[float, float, float]] = None,
        center_um: Optional[Tuple[float, float, float]] = None,
        **kw,
    ) -> "ProfileMonitor":
        """A zero-thickness field monitor on the plane ``axis = position_um``.

        With no ``size_um`` the plane spans the PML-free interior of the
        domain: the in-plane centre and extent are filled in when the monitor
        joins a :class:`~photonhub.Simulation` (a periodic or PEC axis spans
        fully, a symmetry axis from the mirror to the far PML face). An
        explicit ``size_um`` is a 3-tuple whose component along ``axis`` is
        ignored and needs ``center_um`` alongside. Frequencies come from
        ``wlens_um`` (microns, one value or several) or ``freqs_hz``, exactly
        one of them. Other keywords (``apodization``, ``interval_space``,
        ``mode_port``) pass through."""
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
        freqs = _monitor_freqs(wlens_um, freqs_hz, "ProfileMonitor.plane")
        a = "xyz".index(axis)
        if size_um is None:
            if center_um is not None:
                raise ValueError(
                    "ProfileMonitor.plane: center_um needs size_um alongside; omit both "
                    "for a plane spanning the domain interior")
            center = [0.0, 0.0, 0.0]
            size = [0.0, 0.0, 0.0]
            span = f"interior:{axis}"
        else:
            if center_um is None:
                raise ValueError("ProfileMonitor.plane: size_um needs center_um alongside")
            center = [float(c) for c in center_um]
            size = [float(s) for s in size_um]
            span = None
        center[a] = float(position_um)
        size[a] = 0.0
        return cls(name=name, center_um=tuple(center), size_um=tuple(size),
                   fields=tuple(fields), freqs_hz=freqs, span=span, **kw)

    @classmethod
    def sections(
        cls,
        name: str,
        axis: str,
        n: int = 6,
        *,
        wlens_um: Optional[Union[float, Sequence[float]]] = None,
        freqs_hz: Optional[Sequence[float]] = None,
        fields: Tuple[str, ...] = ("Ex", "Ey", "Ez"),
        **kw,
    ) -> List["ProfileMonitor"]:
        """``n`` cross-section planes normal to ``axis``, spread evenly along
        the domain interior: the planes a long device's featured figure stands
        along it (:func:`photonhub.viz.export_scene` with ``plane=name`` draws
        them together).

        Each is a :meth:`plane` spanning the interior, named ``{name}_0`` to
        ``{name}_{n-1}``. Their positions are filled in when they join a
        :class:`~photonhub.Simulation`: plane ``i`` sits at
        ``lo + (i + 1/2) (hi - lo) / n`` of the PML-free interior along
        ``axis``, so a simulation rebuilt around a longer device moves them
        with it. Frequencies and other keywords are as for :meth:`plane`.
        Returns a list: add it to ``monitors`` with ``*``."""
        if int(n) != n or n < 1:
            raise ValueError(f"ProfileMonitor.sections: n must be a positive integer, got {n!r}")
        n = int(n)
        return [cls.plane(f"{name}_{i}", axis, 0.0, wlens_um=wlens_um, freqs_hz=freqs_hz,
                          fields=fields, **kw).model_copy(update={"station": (i, n)})
                for i in range(n)]

    @field_validator("fields")
    @classmethod
    def _unique_fields(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("monitor fields must be unique")
        return value

    @field_validator("freqs_hz")
    @classmethod
    def _unique_freqs(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("monitor freqs_hz must be unique")
        return value

    @field_validator("interval_space")
    @classmethod
    def _strides_positive(cls, v):
        if v is not None and any(s < 1 for s in v):
            raise ValueError(
                f"interval_space strides must be >= 1 (1 = every cell), got {v}"
            )
        return v


class PowerMonitor(FrozenModel):
    """Poynting-flux monitor perpendicular to ``axis`` at ``position_um``,
    snapped to a plane index ``1 <= kp <= n_axis - 1``. Positive values mean power toward +axis. The value
    :class:`~photonhub.RunResult` returns is the time-averaged power in
    WATTS of a continuous-wave excitation at each frequency by the sources
    as declared: the engine accumulates the flux from phasors normalized by
    the first wire-order source's ``A0*S(f)`` (the response to a
    unit-amplitude drive), and the reader multiplies ``A0^2`` back in (the
    array's ``norm_amplitude`` attr; its ``normalization`` attr says so). A
    unit dipole therefore reads its Hertzian power, and a beam or mode
    launched with ``power_watts=1`` reads 1 W on a full plane below it.
    Only a result whose simulation is unknown (an output directory or HDF5
    bundle without its ``sim.json``) keeps the engine's unit-amplitude
    values. Ratios such as R and T (divide by an empty reference run) are
    amplitude-free either way. The plane average of the staggered H
    components makes this exactly the flux the solver conserves from plane to
    plane on any mesh: the power a port's modal readout reports, and the one
    a port or beam launch built cell by cell (the default) normalizes
    ``power_watts`` to; a paraxial beam and an auxiliary-line
    mode source keep the continuum normalization and read ``cos(k*dl/2)`` of
    it. A
    plane wave of field amplitude ``E0`` in an empty domain carries
    ``cos(k*dl/2)`` of the continuum value ``E0^2*A/(2*eta0)`` (-0.5 % at 31
    cells per wavelength, -2 % at 15), which cancels in ratios. At its pulse
    centre ``f0``, an oblique one
    (:meth:`~photonhub.Simulation.with_oblique_plane_wave`) in a medium of
    index ``n`` carries ``E0^2*A*n*cos(theta)/(2*eta0)`` through a plane
    normal to its axis, with ``k`` becoming ``k*cos(theta)`` in that factor.
    That closed form covers ``f0`` only: off it the constant wavevector
    leaves a forward wave ``E0*(1+r)/2``, with ``r = cos(theta)/cos(theta(f))``
    and ``sin(theta(f)) = f0*sin(theta)/f``, and a backward leak
    ``E0*(1-r)/2``, and the band must stay clear of the cutoff
    ``f0*sin(theta)``. Every reading on a Bloch axis has this same
    normalization. A solver from photonhub 0.1.6 or earlier reads 4 times
    the power, and twice the field of a :class:`ProfileMonitor`, on a Bloch
    axis. Time convention is ``e^{-i omega t}``.

    **Resolving a resonance peak** (Q from a spectrum): the FWHM of a
    quality-factor-``Q`` peak at ``f0`` is ``f0/Q``, so ``freqs_hz`` must be
    spaced much finer than that, ``df << f0/Q``, or the fitted width (and
    hence Q) is dominated by sampling. E.g. a Q ~ 400 cavity peak needs a
    dedicated narrow band around ``f0``, not the source's full bandwidth
    (measured: a full-band 300-point sweep under-read Q by ~13%; a
    peak-centred band recovered it, the ring-down ``ResonanceAnalysis`` route
    avoids the issue entirely).

    By default the monitor integrates the FULL transverse plane. Schema 1.17
    adds an optional sub-region WINDOW: pass BOTH ``center_um`` and
    ``size_um`` as ``(u, v)`` pairs in the plane's CYCLIC transverse order
    ``u = (axis+1) % 3, v = (axis+2) % 3`` (z-normal -> (x, y); x-normal ->
    (y, z); y-normal -> (z, x)). The realized window snaps to whole cells by
    cell-centre membership and must cover at least one cell (the engine
    validates). None/None (default) keeps the legacy full plane and is
    omitted from the wire. A window runs on the CPU, on one GPU and on
    several GPUs (``device="gpu:0,1"`` and the like). A solver from
    photonhub 0.1.6 or earlier refuses a window on several GPUs; run it on
    one GPU there.

    **With symmetry planes** (``symmetry=``) the value is the
    power of the whole device through the plane, what the simulation without
    the planes reads, the convention of ports and of every launch's
    ``power_watts``: the engine integrates the part of the
    plane the simulation models, and the reader multiplies it by 2 for every
    symmetry plane that cuts the monitor plane (one whose normal lies in it,
    reached by the window; the array's ``symmetry_factor`` attr, absent when
    no plane cuts the monitor). A window that stops short of the symmetry
    plane reads its own region and gets no factor, and so does a plane
    parallel to a symmetry plane: a closed box of monitors around a source
    therefore balances only with the image of each such face added.
    ``RunResult.wire(name)`` keeps the modeled part."""

    type: Literal["flux"] = "flux"
    name: MonitorName
    axis: AxisName
    position_um: float
    freqs_hz: Tuple[FreqHz, ...] = Field(min_length=1)
    center_um: Optional[Tuple[float, float]] = None
    size_um: Optional[Tuple[PositiveUm, PositiveUm]] = None

    @field_validator("freqs_hz")
    @classmethod
    def _unique_freqs(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("monitor freqs_hz must be unique")
        return value

    @model_validator(mode="after")
    def _window_both_or_neither(self) -> "PowerMonitor":
        if (self.center_um is None) != (self.size_um is None):
            raise ValueError(
                "flux window needs BOTH center_um and size_um (or neither "
                "for the full plane)")
        return self


#: The suffix of each face monitor a :class:`FarFieldMonitor` records:
#: ``<name>.x_minus`` ... ``<name>.z_plus`` (a monitor name allows letters,
#: digits, ``_``, ``-`` and ``.``).
FAR_FIELD_FACE_SIDES = {-1: "minus", 1: "plus"}


def far_field_face_name(name: str, axis: str, sign: float) -> str:
    """The name of the face monitor normal to ``axis`` with outward normal
    ``sign`` that the :class:`FarFieldMonitor` ``name`` records."""
    return f"{name}.{axis}_{FAR_FIELD_FACE_SIDES[1 if sign > 0 else -1]}"


#: A face of a box as ``FarFieldMonitor.box(exclude=...)`` names it: the
#: sign of its outward normal, then its axis.
FaceSideName = Literal["-x", "+x", "-y", "+y", "-z", "+z"]

#: The largest edge-window fraction: at 0.5 the tapers from the two ends
#: meet in the middle of the plane.
FAR_FIELD_WINDOW_MAX = 0.5


class FarFieldMonitor(FrozenModel):
    """A near-field surface recorded for far-field projection: a box
    (:meth:`box`, closed, or open with faces left out) or a single plane
    (:meth:`plane`).

    Client-side only, like a ``ProfileMonitor.plane`` span: when it joins a
    :class:`~photonhub.Simulation` it is compiled into ordinary
    ``field_dft`` monitors, one per face, named ``<name>.x_minus``,
    ``<name>.x_plus`` and so on, recorded the way
    :func:`~photonhub.analysis.far_field` reads them best: the four
    tangential E and H components, one cell thick along the normal (two
    samples, so E and H can be brought onto one surface), and, for a box,
    every cell, one cell past the box in each face's plane. The wire
    document and the solver see only those monitors. The recipe (the faces,
    their outward normals, the medium the faces sit in, the declared
    geometry, the faces left out, the symmetry planes and the edge window)
    is kept beside the run (``client.json``), so ``far_field(data,
    "<name>", ...)`` projects the surface by its name and into its medium;
    a result reloaded without that record still projects from the face
    names, with a warning (and without an open box's edge or a window).

    On a simulation with symmetry planes only the kept part is recorded: a
    face across a plane starts on it, and the box face on the far side of a
    plane is not recorded (``far_field`` reads it as the mirror image of the
    near one). The box must then reach across each plane symmetrically, or
    start on it.

    Building the simulation refuses, naming the face: a face in a PML or
    absorber layer, a face that crosses a structure or a medium boundary,
    faces in different media, a medium the projection cannot use
    (anisotropic, custom, PEC or lossy), a box on a quasi-2-D axis, a box
    or a plane's extent that does not reach a symmetry plane or crosses it
    off centre, a plane on a symmetry plane or in its mirrored half, and
    leaving out a box face normal to a symmetry plane. Faces left out are
    neither recorded nor checked. It warns when a face is closer than half a
    wavelength (in its medium) to a structure, when a source or a structure
    lies within 2 cells of a face's innermost sample (the readout's error
    grows sharply there; each source measured where the solver runs it, a
    point dipole on its component's nearest grid point, a plane source on
    its grid line), and, on a graded mesh, when a box face is nearer
    than 0.35 wavelengths to a grid line where the cell size changes, along
    its normal, or a source is within 0.1 wavelengths of one: the asymmetry
    the projection reads on a graded mesh is shown to stay within 2 % of the
    peak (measured up to 0.93 %) at 30 background cells per wavelength
    (refinement ratios 2 and 3) only for boxes clear of such changes."""

    type: Literal["far_field"] = "far_field"
    name: MonitorName
    kind: Literal["box", "plane"]
    # box: the box; plane: the in-plane extent (both None: the PML-free interior)
    center_um: Optional[Vec3Um] = None
    size_um: Optional[Tuple[NonNegativeUm, NonNegativeUm, NonNegativeUm]] = None
    # plane: its normal, position along it and the side it radiates into
    axis: Optional[AxisName] = None
    position_um: Optional[float] = None
    direction: DirectionName = "+"
    freqs_hz: Tuple[FreqHz, ...] = Field(min_length=1)
    # plane: the in-plane sampling, points per wavelength in the plane's medium
    pts_per_wlen: Optional[float] = Field(default=None, gt=0)
    apodization: Optional[Apodization] = None
    # box: the faces left out (an open box), by outward normal
    exclude: Tuple[FaceSideName, ...] = ()
    # plane: the edge window, the fraction of the plane's extent tapered at
    # each end along its two in-plane axes (in xyz order)
    window: Optional[Tuple[float, float]] = None

    @classmethod
    def box(
        cls,
        name: str,
        center_um: Tuple[float, float, float],
        size_um: Tuple[float, float, float],
        *,
        wlens_um: Optional[Union[float, Sequence[float]]] = None,
        freqs_hz: Optional[Sequence[float]] = None,
        exclude: Union[str, Sequence[str]] = (),
        **kw,
    ) -> "FarFieldMonitor":
        """A box of six faces around ``center_um``, ``size_um`` wide
        (microns). Each face sits on the grid line nearest its plane. The
        frequencies come from ``wlens_um`` (microns) or ``freqs_hz``, exactly
        one of them; ``apodization`` passes through to the face monitors.
        Box faces record every cell: a stride (``pts_per_wlen``) is refused,
        since a decimated box face reads the power up to 17 % off with the
        source near a corner.

        ``exclude`` leaves faces out, an open box: ``("-z",)`` drops the
        face whose outward normal is ``-z``, for instance where a substrate
        lies below the box (the faces kept must still sit in one medium, so
        the open edge stays above the substrate). A face left out is neither
        recorded nor checked. The faces kept still record one cell past the
        box on the open side too, and ``far_field`` ends them at the plane
        the face left out is declared on (the box's centre plus or minus half
        its size), with the closed box's fourth-order edge rule wherever that
        plane falls between samples, so the open edge stays where it was
        declared on any mesh. The projection is the closed box's less that
        face's own contribution. It no longer sees the light leaving through
        the open side, and what
        reaches the kept faces from there is read as if the box were closed,
        so the result approximates the pattern on the side the box is closed
        on and is not a full-sphere pattern. A window is refused on a box:
        tapering the currents at a box's edges opens the surface the
        projection relies on being closed."""
        freqs = _monitor_freqs(wlens_um, freqs_hz, "FarFieldMonitor.box")
        if isinstance(exclude, str):
            exclude = (exclude,)
        return cls(name=name, kind="box", center_um=tuple(center_um),
                   size_um=tuple(size_um), freqs_hz=freqs, exclude=tuple(exclude), **kw)

    @classmethod
    def plane(
        cls,
        name: str,
        axis: str,
        position_um: float,
        *,
        direction: str = "+",
        wlens_um: Optional[Union[float, Sequence[float]]] = None,
        freqs_hz: Optional[Sequence[float]] = None,
        pts_per_wlen: Optional[float] = None,
        center_um: Optional[Tuple[float, float, float]] = None,
        size_um: Optional[Tuple[float, float, float]] = None,
        window: Optional[Tuple[float, float]] = None,
        **kw,
    ) -> "FarFieldMonitor":
        """A single plane ``axis = position_um`` radiating toward
        ``direction`` (``"+"`` or ``"-"`` along ``axis``), on the grid line
        nearest it. With no ``size_um`` it spans the PML-free interior of
        the domain; an explicit ``size_um`` (its component along ``axis`` is
        ignored) needs ``center_um`` alongside. ``pts_per_wlen`` sets an
        in-plane stride so the plane keeps at least that many samples per
        wavelength in its medium (at the shortest wavelength, across the
        largest cell); without it every cell is recorded.

        ``window=(f1, f2)`` tapers the currents toward the plane's edges
        before they are integrated, off by default: over the outer fraction
        ``f1`` of the plane's declared extent at each end of its first
        in-plane axis, and ``f2`` along its second (the in-plane axes in
        xyz order: x and y for a plane normal to z, y and z for x, x and z
        for y), the currents are scaled by ``sin^2``, rising from 0 at the
        edge to 1. The declared extent is ``size_um``, or the PML-free
        interior, mirrored across a symmetry plane, so a run with symmetry
        planes and the full run taper the same surface. Each fraction is 0 (no taper along that axis) to 0.5
        (the two tapers meet in the middle). An open plane is a truncated
        surface: the field cut off at its edges radiates an edge ripple
        into the pattern, which the window trades for a smoother pattern
        that underweights light leaving near the edges."""
        freqs = _monitor_freqs(wlens_um, freqs_hz, "FarFieldMonitor.plane")
        return cls(name=name, kind="plane", axis=axis, position_um=float(position_um),
                   direction=direction, freqs_hz=freqs, pts_per_wlen=pts_per_wlen,
                   center_um=None if center_um is None else tuple(center_um),
                   size_um=None if size_um is None else tuple(size_um),
                   window=None if window is None else tuple(window), **kw)

    @property
    def faces(self) -> Tuple[Tuple[str, str, float], ...]:
        """``(monitor name, axis, outward sign)`` of each face it declares, in
        the form ``far_field(..., faces=...)`` takes; a box's faces left out
        are not among them. On a simulation with symmetry planes the far-side
        face of each plane is not recorded either:
        ``Simulation.far_field_monitors`` lists the faces a simulation
        records."""
        if self.kind == "plane":
            sign = 1.0 if self.direction == "+" else -1.0
            return ((far_field_face_name(self.name, self.axis, sign), self.axis, sign),)
        return tuple((far_field_face_name(self.name, a, s), a, s)
                     for a in "xyz" for s in (-1.0, 1.0)
                     if f"{'+' if s > 0 else '-'}{a}" not in self.exclude)

    @field_validator("name")
    @classmethod
    def _room_for_the_face_names(cls, value):
        longest = len(far_field_face_name(value, "x", -1.0))
        if longest > _MONITOR_NAME_MAX_LENGTH:
            raise ValueError(
                f"far-field monitor name {value!r} is too long: its face monitors "
                f"(<name>.x_minus) would have {longest} characters, more than "
                f"{_MONITOR_NAME_MAX_LENGTH}")
        return value

    @field_validator("freqs_hz")
    @classmethod
    def _unique_freqs(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("monitor freqs_hz must be unique")
        return value

    @model_validator(mode="after")
    def _kind_fields(self) -> "FarFieldMonitor":
        who = f"FarFieldMonitor {self.name!r}"
        if self.kind == "box":
            if self.center_um is None or self.size_um is None:
                raise ValueError(f"{who}: a box needs center_um and size_um")
            if any(float(s) <= 0.0 for s in self.size_um):
                raise ValueError(f"{who}: a box needs a positive size along every axis, got "
                                 f"{self.size_um}")
            if self.axis is not None or self.position_um is not None or self.direction != "+":
                raise ValueError(f"{who}: axis, position_um and direction belong to a plane")
            if self.pts_per_wlen is not None:
                raise ValueError(
                    f"{who}: the faces of a box are recorded at every cell, so pts_per_wlen "
                    "(an in-plane stride) applies to a single plane only. A strided box face "
                    "misses the near field of a source near it: the radiated power read up "
                    "to 17 % off at a stride of 2 with the source near a corner")
            if self.window is not None:
                raise ValueError(
                    f"{who}: an edge window applies to a single plane only. The faces of a "
                    "box meet at its edges, where the projection needs the full currents of "
                    "both faces: tapering them there opens the surface it relies on being "
                    "closed, so the pattern and the radiated power would both be off")
            if len(set(self.exclude)) != len(self.exclude):
                raise ValueError(f"{who}: exclude names a face twice: {self.exclude}")
            if len(self.exclude) >= 6:
                raise ValueError(f"{who}: exclude leaves out every face of the box")
        else:
            if self.axis is None or self.position_um is None:
                raise ValueError(f"{who}: a plane needs axis and position_um")
            if (self.center_um is None) != (self.size_um is None):
                raise ValueError(f"{who}: size_um needs center_um alongside (omit both for a "
                                 "plane spanning the domain interior)")
            if self.exclude:
                raise ValueError(f"{who}: exclude leaves faces out of a box; a plane has one")
            if self.window is not None and not all(
                    math.isfinite(float(f)) and 0.0 <= float(f) <= FAR_FIELD_WINDOW_MAX
                    for f in self.window):
                raise ValueError(
                    f"{who}: window takes the fraction of the plane's extent tapered at each "
                    f"end along each in-plane axis, 0 to {FAR_FIELD_WINDOW_MAX:g}; got "
                    f"{self.window}")
        return self


MonitorType = Annotated[
    Union[TimeMonitor, SnapshotMonitor, ProfileMonitor, PowerMonitor,
          # client-only: compiled into ProfileMonitors at construction,
          # never on the wire and never in the schema
          SkipJsonSchema[FarFieldMonitor]],
    Field(discriminator="type"),
]
