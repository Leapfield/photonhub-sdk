"""Field monitors (NUMERICS.md sections 6 and 12).

Monitor names must be filename-safe (the engine writes ``<name>.bin``) and, a constraint JSON Schema cannot express across array items, unique within a
simulation (enforced by ``Simulation``).
"""

from typing import Annotated, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from ..constants import c0
from .base import (
    MAX_INT32,
    AxisName,
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
    ``ApodizationSpec`` analogue, NUMERICS.md section 12). A Gaussian roll-ON of
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
    """Running-DFT field monitor over a box region (NUMERICS.md section 12):
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
        if (wlens_um is None) == (freqs_hz is None):
            raise ValueError("ProfileMonitor.plane: pass exactly one of wlens_um or freqs_hz")
        if wlens_um is not None:
            wl = (float(wlens_um),) if isinstance(wlens_um, (int, float)) else tuple(float(w) for w in wlens_um)
            if any(w <= 0.0 for w in wl):
                raise ValueError("ProfileMonitor.plane: wlens_um must be positive")
            freqs = tuple(_C0_M_PER_S / (w * 1e-6) for w in wl)
        else:
            freqs = tuple(float(f) for f in freqs_hz)
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
    snapped to a plane index ``1 <= kp <= n_axis - 1`` (NUMERICS.md
    section 12). Positive values mean power toward +axis. The value
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
    ``power_watts`` to (NUMERICS §18.7); a paraxial beam and an auxiliary-line
    mode source keep the continuum normalization and read ``cos(k*dl/2)`` of
    it. A
    plane wave of field amplitude ``E0`` in an empty domain carries
    ``cos(k*dl/2)`` of the continuum value ``E0^2*A/(2*eta0)`` (-0.5 % at 31
    cells per wavelength, -2 % at 15), which cancels in ratios. Time
    convention is ``e^{-i omega t}``.

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
    omitted from the wire. Not yet supported by the multi-GPU decomposition
    (single GPU / CPU only).

    **With symmetry planes** (``symmetry=``, NUMERICS §20) the value is the
    power of the whole device through the plane, what the simulation without
    the planes reads, the convention of ports and of every launch's
    ``power_watts`` (NUMERICS §20.8): the engine integrates the part of the
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


MonitorType = Annotated[
    Union[TimeMonitor, SnapshotMonitor, ProfileMonitor, PowerMonitor],
    Field(discriminator="type"),
]
