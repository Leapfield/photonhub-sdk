"""Current sources (NUMERICS.md sections 5 and 13)."""

from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import (
    AxisName,
    DirectionName,
    FieldComponentName,
    FreqHz,
    FrozenModel,
    PositiveUm,
    Vec3Um,
)
from .source_time import SourceTimeType


class PointDipole(FrozenModel):
    """Soft point current source at the nearest Yee node. An ELECTRIC
    polarization (``Ex``/``Ey``/``Ez``) injects an electric current density
    onto E after the E-update; a MAGNETIC polarization (``Hx``/``Hy``/``Hz``)
    injects a magnetic current density onto H right after the H-update
    (NUMERICS.md sections 5 and 12). ``amplitude`` is the peak current density
    J0 [A/m^2] (electric) or M0 [V/m^2] (magnetic).

    Magnetic dipoles are supported on both the CPU reference solver and the
    GPU (NUMERICS.md sections 5 and 12)."""

    type: Literal["point_dipole"] = "point_dipole"
    center_um: Vec3Um
    polarization: FieldComponentName
    amplitude: float = 1.0
    source_time: SourceTimeType


# Published-schema mirror of the tangential-polarization rule below: for each
# propagation axis, restrict the polarization enum to the two transverse E
# components, so third-party producers cannot emit specs the engine rejects.
_PLANE_WAVE_TANGENTIAL = {
    "allOf": [
        {
            "if": {"properties": {"axis": {"const": axis}}, "required": ["axis"]},
            "then": {
                "properties": {
                    "polarization": {
                        "enum": [c for c in ("Ex", "Ey", "Ez") if c[1] != axis]
                    }
                }
            },
        }
        for axis in ("x", "y", "z")
    ]
}


class PlaneWave(FrozenModel):
    """Normal-incidence plane wave injected on a TF/SF plane perpendicular
    to ``axis`` at ``position_um``, fed by a dispersion-matched 1-D auxiliary
    solver (NUMERICS.md section 13). ``amplitude`` is the peak incident
    E-field E0 in V/m. Both transverse axes must be periodic (enforced by
    ``Simulation``) and the plane must not intersect PML (checked by
    ``phsolver validate``)."""

    model_config = ConfigDict(json_schema_extra=_PLANE_WAVE_TANGENTIAL)

    type: Literal["plane_wave"] = "plane_wave"
    axis: AxisName
    direction: DirectionName
    position_um: float
    polarization: FieldComponentName = Field(
        json_schema_extra={"enum": ["Ex", "Ey", "Ez"]}
    )
    amplitude: float = 1.0
    source_time: SourceTimeType

    @field_validator("polarization")
    @classmethod
    def _electric_only(cls, v: str) -> str:
        if not v.startswith("E"):
            raise ValueError(
                f"plane-wave polarization '{v}' must be an E component "
                "(the tangential E axis); use one of Ex, Ey, Ez"
            )
        return v

    @model_validator(mode="after")
    def _polarization_tangential(self) -> "PlaneWave":
        if self.polarization[1] == self.axis:
            raise ValueError(
                f"plane-wave polarization '{self.polarization}' must be "
                f"tangential to the injection plane: it cannot lie along the "
                f"propagation axis '{self.axis}'"
            )
        if self.source_time.band_freqs_hz is not None:
            raise ValueError(
                "GaussianPulse band_freqs_hz/carrier_index are reserved for "
                "PointDipole equivalence-current sheets; a plane wave uses "
                "its ordinary Gaussian carrier"
            )
        return self


class ModeSolveProvenance(FrozenModel):
    """Reproducible authoring recipe for a solved :class:`ModeSource`.

    This optional block has no native numerical semantics: the engine consumes
    the resolved arrays beside it.  Workbench uses it to restore the compact
    port editor after reopen and to detect when geometry/grid edits have made
    those arrays stale.  ``input_sha256`` is a conservative hash of the
    canonical solve inputs; legacy sources without this block remain valid but
    have unverified profile provenance. ``polarization`` names the Yee solver's
    natural horizontal/vertical family. A linked modal port maps that family to
    physical width/thickness TE/TM using the port's ``thickness_axis``.
    """

    solver: Literal["yee"] = "yee"
    polarization: Literal["TE", "TM"]
    mode_index: int = Field(ge=0, le=31)
    wavelength_um: PositiveUm
    center_um: Tuple[float, float]
    size_um: Tuple[PositiveUm, PositiveUm]
    dl_um: PositiveUm
    supersample: int = Field(ge=1, le=16)
    num_modes: Optional[int] = Field(default=None, ge=1, le=32)
    num_freqs: int = Field(default=1, ge=1, le=11)
    input_sha256: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _mode_count_covers_index(self) -> "ModeSolveProvenance":
        if self.num_modes is not None and self.num_modes <= self.mode_index:
            raise ValueError("num_modes must be greater than mode_index")
        return self


class ModeSource(FrozenModel):
    """Inject a guided mode on a TF/SF plane (NUMERICS.md §18).

    .. deprecated::
        The §18 aux-line ModeSource is deprecated in favour of the equivalence-
        current Huygens launch (per-cell ``PointDipole`` sheets via
        ``mode_launch`` over a discrete Yee mode). Both paths support uniform and
        graded transverse/propagation axes; §18 uses the graded auxiliary-line
        construction in NUMERICS §§15.9 and 18.4. The equivalence-current path is
        preferred for solved Yee modes because it carries their discrete paired-H
        fields and grid provenance. §18 remains for the adjoint (gradient pinned
        to it) and scalar/FLM modes; new solved-mode code should use ``mode_launch``.

    The 1-D
    auxiliary line runs at the mode phase index ``n_eff`` (which gives both the
    modal phase velocity and the scalar-limit modal impedance), and each
    transverse plane point is scaled by ``profile`` — the FDE eigenmode
    resampled onto the grid's transverse plane, row-major ``[v*nu + u]`` with
    ``u``/``v`` the lower/higher-indexed transverse axes. ``polarization`` is
    the major tangential E component. Build via
    ``photonhub.plugins.mode_devices.mode_source`` (it computes ``profile``/
    ``n_eff`` from an FDE mode); hand-construction is rarely needed. Unlike the
    plane wave, the injection plane MAY cut through structures.

    **Full-vector injection (schema 1.8.0, additive/optional).** A full-vector
    FDE mode (``VectorMode``) carries BOTH transverse-E components. When the
    builder ships them, ``minor_polarization`` names the second tangential-E
    component and ``profile_minor`` is its real signed profile (same row-major
    ``[v*nu + u]`` layout, sampled onto the same plane, scaled consistently with
    ``profile`` so the two carry the mode's true component RATIO). For a lossless
    guided mode both transverse-E components are co-real (relative phase 0 or π),
    so a single real signed profile is exact; the engine injects each off the
    same guided-mode aux carrier with its own scalar-limit paired H. When the
    minor fields are omitted (the legacy default) the engine injects the
    scalar-limit major component only — older engines/JSON stay valid."""

    type: Literal["mode_source"] = "mode_source"
    axis: AxisName
    direction: DirectionName
    position_um: float
    polarization: FieldComponentName = Field(
        json_schema_extra={"enum": ["Ex", "Ey", "Ez"]}
    )
    amplitude: float = 1.0
    n_eff: float = Field(ge=1.0)
    nu: int = Field(ge=1)
    nv: int = Field(ge=1)
    profile: Tuple[float, ...] = Field(min_length=1)
    # Full-vector minor transverse-E (schema 1.8.0, additive/optional). Both are
    # present together or both absent; validated below.
    minor_polarization: Optional[FieldComponentName] = Field(
        default=None, json_schema_extra={"enum": ["Ex", "Ey", "Ez"]}
    )
    profile_minor: Optional[Tuple[float, ...]] = Field(default=None, min_length=1)
    # True paired-H profiles (additive/optional). By default the engine's TF/SF
    # E-correction uses the SCALAR-LIMIT incident H = (n_eff/η0)·E (the aux-line
    # impedance), which is not the true mode H for a high-contrast guide and
    # radiates ~few %. When set, `profile_h` (paired with the major E) and
    # `profile_h_minor` (paired with the minor E, present iff `profile_minor`)
    # carry the mode's TRUE transverse H, scaled into "E-equivalent" units
    # (h_w·η0/n_eff) so the same aux carrier reconstructs it. Omitted ⇒ legacy
    # scalar-limit (byte-identical wire).
    profile_h: Optional[Tuple[float, ...]] = Field(default=None, min_length=1)
    profile_h_minor: Optional[Tuple[float, ...]] = Field(default=None, min_length=1)
    # Broadband injection (schema 1.11.0, additive/optional — the Tidy3D
    # `num_freqs` analogue, NUMERICS.md §18.3). When `freqs_hz` is set the engine
    # injects the mode with a FREQUENCY-DEPENDENT transverse profile and phase
    # index, partition-of-unity-windowed across the band from these N >= 2
    # samples (each a mode solved AT that frequency). The scalar `n_eff`/
    # `profile`/`profile_minor` above stay the band-CENTRE representative, so an
    # engine that ignores these arrays injects a sensible single mode (graceful
    # degradation) and the wire is byte-identical to 1.10 when unset. `freqs_hz`
    # is strictly ascending; the per-frequency arrays are parallel to it.
    freqs_hz: Optional[Tuple[FreqHz, ...]] = Field(default=None, min_length=2)
    n_eff_by_freq: Optional[Tuple[float, ...]] = Field(default=None, min_length=2)
    profiles_by_freq: Optional[Tuple[Tuple[float, ...], ...]] = Field(
        default=None, min_length=2
    )
    profiles_minor_by_freq: Optional[Tuple[Tuple[float, ...], ...]] = Field(
        default=None, min_length=2
    )
    # Per-frequency TRUE paired-H (schema 1.13.0). Parallel to `freqs_hz`; the
    # broadband analogue of `profile_h`. When set, the engine injects each carrier
    # with the mode's true H AT that frequency instead of the single band-centre
    # `profile_h` (correct only at the band centre; off-centre it radiates a
    # non-decaying residual). Present iff `profile_h` is; minor follows the minor E.
    profiles_h_by_freq: Optional[Tuple[Tuple[float, ...], ...]] = Field(
        default=None, min_length=2
    )
    profiles_h_minor_by_freq: Optional[Tuple[Tuple[float, ...], ...]] = Field(
        default=None, min_length=2
    )
    source_time: SourceTimeType
    # Additive schema-1.15 authoring provenance. The resolved profile arrays
    # remain the execution contract; phsolver accepts and deliberately ignores
    # this block. Workbench hashes the recipe + canonical modal inputs to stop a
    # geometry/grid edit from silently running an old mode profile.
    mode_solve: Optional[ModeSolveProvenance] = None

    @field_validator("polarization", "minor_polarization")
    @classmethod
    def _electric_only(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith("E"):
            raise ValueError(
                f"mode-source polarization '{v}' must be an E component "
                "(a tangential E axis); use one of Ex, Ey, Ez"
            )
        return v

    @model_validator(mode="after")
    def _check(self) -> "ModeSource":
        if self.source_time.band_freqs_hz is not None:
            raise ValueError(
                "GaussianPulse band_freqs_hz/carrier_index are reserved for "
                "PointDipole equivalence-current sheets; broadband ModeSource "
                "injection uses freqs_hz and the parallel mode-profile arrays"
            )
        if len(self.profile) != self.nu * self.nv:
            raise ValueError(
                f"profile length {len(self.profile)} != nu*nv "
                f"({self.nu}*{self.nv})"
            )
        if self.polarization[1] == self.axis:
            raise ValueError(
                f"mode-source polarization '{self.polarization}' must be "
                f"tangential to the injection plane (axis '{self.axis}')"
            )
        has_minor_pol = self.minor_polarization is not None
        has_minor_prof = self.profile_minor is not None
        if has_minor_pol != has_minor_prof:
            raise ValueError(
                "minor_polarization and profile_minor must be set together "
                "(full-vector injection) or both omitted (scalar limit)"
            )
        if has_minor_pol:
            if self.minor_polarization[1] == self.axis:
                raise ValueError(
                    f"mode-source minor_polarization "
                    f"'{self.minor_polarization}' must be tangential to the "
                    f"injection plane (axis '{self.axis}')"
                )
            if self.minor_polarization == self.polarization:
                raise ValueError(
                    "minor_polarization must differ from the major "
                    f"polarization '{self.polarization}'"
                )
            if len(self.profile_minor) != self.nu * self.nv:
                raise ValueError(
                    f"profile_minor length {len(self.profile_minor)} != nu*nv "
                    f"({self.nu}*{self.nv})"
                )
        # True-H profiles: profile_h pairs the major E; profile_h_minor pairs the
        # minor E and is present iff a minor E is.
        if self.profile_h is not None:
            if len(self.profile_h) != self.nu * self.nv:
                raise ValueError(
                    f"profile_h length {len(self.profile_h)} != nu*nv "
                    f"({self.nu}*{self.nv})"
                )
            has_minor_h = self.profile_h_minor is not None
            if has_minor_h != has_minor_pol:
                raise ValueError(
                    "profile_h_minor must be set iff the minor E is "
                    "(profile_minor) for true-H injection"
                )
            if has_minor_h and len(self.profile_h_minor) != self.nu * self.nv:
                raise ValueError(
                    f"profile_h_minor length {len(self.profile_h_minor)} != "
                    f"nu*nv ({self.nu}*{self.nv})"
                )
        elif self.profile_h_minor is not None:
            raise ValueError("profile_h_minor set without profile_h")
        self._check_broadband(has_minor_pol)
        return self

    def _check_broadband(self, has_minor: bool) -> None:
        """Validate the optional broadband (num_freqs) sample arrays. Either
        none of them are set (the single frozen mode) or `freqs_hz`,
        `n_eff_by_freq`, and `profiles_by_freq` are all set, parallel, and
        consistent with the scalar fields (NUMERICS.md §18.3)."""
        bb = self.freqs_hz is not None
        # All-or-nothing on the core triple; the minor array follows the scalar
        # minor's presence.
        if not bb:
            for name in ("n_eff_by_freq", "profiles_by_freq",
                         "profiles_minor_by_freq", "profiles_h_by_freq",
                         "profiles_h_minor_by_freq"):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"{name} requires freqs_hz to be set (broadband "
                        "injection); set freqs_hz or drop the per-frequency "
                        "arrays"
                    )
            return
        if self.n_eff_by_freq is None or self.profiles_by_freq is None:
            raise ValueError(
                "broadband injection needs freqs_hz, n_eff_by_freq, and "
                "profiles_by_freq set together"
            )
        n = len(self.freqs_hz)
        if len(self.n_eff_by_freq) != n or len(self.profiles_by_freq) != n:
            raise ValueError(
                f"broadband arrays must be parallel to freqs_hz (length {n}): "
                f"n_eff_by_freq={len(self.n_eff_by_freq)}, "
                f"profiles_by_freq={len(self.profiles_by_freq)}"
            )
        if any(b <= a for a, b in zip(self.freqs_hz, self.freqs_hz[1:])):
            raise ValueError("freqs_hz must be strictly ascending")
        if any(ne < 1.0 for ne in self.n_eff_by_freq):
            raise ValueError("every n_eff_by_freq must be >= 1.0")
        expect = self.nu * self.nv
        for i, prof in enumerate(self.profiles_by_freq):
            if len(prof) != expect:
                raise ValueError(
                    f"profiles_by_freq[{i}] length {len(prof)} != nu*nv "
                    f"({self.nu}*{self.nv})"
                )
        if has_minor:
            if self.profiles_minor_by_freq is None:
                raise ValueError(
                    "profiles_minor_by_freq must be set for a full-vector "
                    "broadband source (the scalar profile_minor is present)"
                )
            if len(self.profiles_minor_by_freq) != n:
                raise ValueError(
                    f"profiles_minor_by_freq length "
                    f"{len(self.profiles_minor_by_freq)} != freqs_hz length {n}"
                )
            for i, prof in enumerate(self.profiles_minor_by_freq):
                if len(prof) != expect:
                    raise ValueError(
                        f"profiles_minor_by_freq[{i}] length {len(prof)} != "
                        f"nu*nv ({self.nu}*{self.nv})"
                    )
        elif self.profiles_minor_by_freq is not None:
            raise ValueError(
                "profiles_minor_by_freq set without a scalar profile_minor — "
                "the minor component must be present at the band centre too"
            )
        # Per-frequency true-H (1.13.0): parallel to freqs_hz, paired with the
        # band-centre profile_h; the minor H follows the minor component.
        if self.profiles_h_by_freq is not None:
            if self.profile_h is None:
                raise ValueError(
                    "profiles_h_by_freq set without the band-centre profile_h "
                    "(true-H injection)"
                )
            if len(self.profiles_h_by_freq) != n:
                raise ValueError(
                    f"profiles_h_by_freq length {len(self.profiles_h_by_freq)} "
                    f"!= freqs_hz length {n}"
                )
            for i, prof in enumerate(self.profiles_h_by_freq):
                if len(prof) != expect:
                    raise ValueError(
                        f"profiles_h_by_freq[{i}] length {len(prof)} != nu*nv "
                        f"({self.nu}*{self.nv})"
                    )
            if has_minor:
                if self.profiles_h_minor_by_freq is None:
                    raise ValueError(
                        "profiles_h_minor_by_freq must be set for a full-vector "
                        "broadband true-H source"
                    )
                if len(self.profiles_h_minor_by_freq) != n:
                    raise ValueError(
                        "profiles_h_minor_by_freq length "
                        f"{len(self.profiles_h_minor_by_freq)} != freqs_hz "
                        f"length {n}"
                    )
                for i, prof in enumerate(self.profiles_h_minor_by_freq):
                    if len(prof) != expect:
                        raise ValueError(
                            f"profiles_h_minor_by_freq[{i}] length {len(prof)} "
                            f"!= nu*nv ({self.nu}*{self.nv})"
                        )
            elif self.profiles_h_minor_by_freq is not None:
                raise ValueError(
                    "profiles_h_minor_by_freq set without a minor component"
                )
        elif self.profiles_h_minor_by_freq is not None:
            raise ValueError(
                "profiles_h_minor_by_freq set without profiles_h_by_freq"
            )


SourceType = Annotated[
    Union[PointDipole, PlaneWave, ModeSource], Field(discriminator="type")
]
