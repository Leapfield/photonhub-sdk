"""Field monitors (NUMERICS.md sections 6 and 12).

Monitor names must be filename-safe (the engine writes ``<name>.bin``) and —
a constraint JSON Schema cannot express across array items — unique within a
simulation (enforced by ``Simulation``).
"""

from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

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


class FieldTimeMonitor(FrozenModel):
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


class FieldSnapshotMonitor(FrozenModel):
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
    """Time window applied to a monitor's running DFT (the Tidy3D
    ``ApodizationSpec`` analogue, NUMERICS.md section 12). A Gaussian roll-ON of
    standard deviation ``width_s`` for t < ``start_s``, flat (== 1) on
    ``[start_s, end_s]``, and a Gaussian roll-OFF for t > ``end_s`` — used to
    isolate the late-time steady state of a resonant structure (suppressing the
    source-injection transient). ``start_s``/``end_s`` (seconds) are each
    optional — omit a side to leave it ungated; ``width_s`` (seconds) is the
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


class FieldDftMonitor(FrozenModel):
    """Running-DFT field monitor over a box region (NUMERICS.md section 12):
    fp64 accumulation every step over the full run, raw Yee-located phasors,
    normalized by the first wire-order source's ``A0 * S(f)``. ``size_um``
    components may be 0 (plane/line/point regions); the region is snapped per
    component to that component's Yee sublattice, and the engine validator
    REJECTS boxes whose per-component snaps disagree (the output carries one
    shape/origin per monitor). When ``fields`` mixes Yee offsets along an
    axis, place that axis' box faces strictly between an integer cell
    boundary and the next half-cell plane — canonically ``(k + 0.25) * dl``,
    which every component snaps to cell ``k`` with quarter-cell fp margin."""

    type: Literal["field_dft"] = "field_dft"
    name: MonitorName
    center_um: Vec3Um
    size_um: Tuple[NonNegativeUm, NonNegativeUm, NonNegativeUm]
    fields: Tuple[FieldComponentName, ...] = Field(min_length=1)
    freqs_hz: Tuple[FreqHz, ...] = Field(min_length=1)
    # Per-axis spatial sampling stride (schema 1.11.0, additive/optional — the
    # Tidy3D interval_space). None (default) records every cell; (sx, sy, sz)
    # decimates the recorded region along each axis (output cell i -> snapped
    # cell + i*stride), cutting field-monitor output for large planes/volumes.
    # Each stride >= 1. Omitted from the wire when unset (older engines/readers
    # round-trip unchanged); the data layer strides the coordinates to match.
    interval_space: Optional[Tuple[int, int, int]] = None
    # Optional time-apodization of the running DFT (schema 1.13.0, additive —
    # the Tidy3D ApodizationSpec). None (default) => no window, omitted from the
    # wire so older engines/readers round-trip unchanged.
    apodization: Optional[Apodization] = None
    # Schema 1.16 authoring metadata. The resolved execution object remains this
    # ordinary DFT monitor; Workbench/result APIs compile the saved recipe into
    # ModeMonitor/SPort post-processing after a run.
    mode_port: Optional[ModePort] = None

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


class FluxMonitor(FrozenModel):
    """Poynting-flux monitor over the full plane perpendicular to ``axis`` at
    ``position_um``, snapped to a plane index ``1 <= kp <= n_axis - 1``
    (NUMERICS.md section 12). Positive values mean power toward +axis; the
    reported power carries the ``1/|A0*S(f)|^2`` normalization of the shared
    phasors, so it is not absolute watts."""

    type: Literal["flux"] = "flux"
    name: MonitorName
    axis: AxisName
    position_um: float
    freqs_hz: Tuple[FreqHz, ...] = Field(min_length=1)

    @field_validator("freqs_hz")
    @classmethod
    def _unique_freqs(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("monitor freqs_hz must be unique")
        return value


MonitorType = Annotated[
    Union[FieldTimeMonitor, FieldSnapshotMonitor, FieldDftMonitor, FluxMonitor],
    Field(discriminator="type"),
]
