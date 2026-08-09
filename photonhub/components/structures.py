"""Geometry/material structures (NUMERICS.md sections 9-10).

``structures`` is an ordered list; materials are rasterized per E component
at that component's own Yee point and the LAST structure containing the
point wins (containment is closed). Geometries may extend beyond the domain
— only the part inside the grid matters — so no domain check applies here.
"""

import math
from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import Field, model_validator

from .base import AxisName, FrozenModel, NonNegativeUm, PositiveUm, Vec3Um


class LorentzPole(FrozenModel):
    """A single Lorentz pole for a frequency-dependent (dispersive) medium
    (NUMERICS.md §19). Under the engine's e^{-i omega t} convention the medium
    permittivity is

        eps(omega) = eps_inf + delta_eps * omega0^2
                     / (omega0^2 - omega^2 - i*gamma*omega)

    where omega0 = 2*pi*resonance_frequency_hz and gamma = 2*pi*linewidth_hz
    (both stored as ordinary Hz on the wire; the engine multiplies by 2*pi).
    ``delta_eps`` is the oscillator strength (the static contribution of this
    pole, eps(0) - eps_inf = delta_eps). The MVP supports exactly ONE pole,
    scalar and isotropic (multi-pole / Drude / anisotropic poles are deferred,
    NUMERICS.md §19 TODO).

    Passivity (Im eps >= 0 for omega > 0) requires delta_eps >= 0 and
    gamma >= 0; the engine validates these. ``gamma = 0`` is a lossless
    (undamped) resonance — allowed, but the timestep must stay clear of the
    resonance for stability (NUMERICS.md §19 omega0*dt/2 < 1)."""

    resonance_frequency_hz: float = Field(gt=0.0)  # f0 = omega0 / 2pi
    delta_eps: float = Field(ge=0.0)               # oscillator strength
    linewidth_hz: float = Field(default=0.0, ge=0.0)  # gamma / 2pi


class Medium(FrozenModel):
    """Isotropic nonmagnetic medium: scalar relative permittivity plus an
    electric conductivity entering the lossy Ca/Cb update (NUMERICS.md
    section 10). ``sigma = 0`` reproduces the Phase-0 update bit-exactly.

    A non-dispersive medium leaves ``lorentz`` unset (None) and the field is
    omitted from the wire entirely (back-compat: schema < 1.9 documents and
    every existing scene round-trip byte-identically). When a single Lorentz
    pole is supplied, ``permittivity`` is the high-frequency limit eps_inf
    (NUMERICS.md §19) and the medium is dispersive (the ADE polarization
    update engages only in cells carrying that pole)."""

    permittivity: float = Field(ge=1.0)
    conductivity_s_per_m: float = Field(default=0.0, ge=0.0)
    # NUMERICS.md §19 — optional single Lorentz pole (additive/back-compat).
    # When set, ``permittivity`` is eps_inf and the cell is dispersive. The MVP
    # is a SINGLE scalar isotropic pole; a list (multi-pole) is a later phase.
    lorentz: Optional[LorentzPole] = None

    def permittivity_at_hz(self, freq_hz: float) -> float:
        """Real relative permittivity ``Re eps(omega)`` at ``freq_hz`` under the
        pole model above — for a non-dispersive medium this is just
        ``permittivity``; with a Lorentz pole, ``permittivity`` alone is only
        the high-frequency limit eps_inf and is the WRONG value to hand a mode
        solver or any other frequency-anchored consumer (a Si pole fit reads
        ~9.6 instead of ~12.1 at 1.55 um). Raises at an exactly undamped
        resonance, where eps diverges."""
        if self.lorentz is None:
            return float(self.permittivity)
        if not freq_hz > 0.0:
            raise ValueError(f"freq_hz must be > 0, got {freq_hz}")
        p = self.lorentz
        w = 2.0 * math.pi * float(freq_hz)
        w0 = 2.0 * math.pi * p.resonance_frequency_hz
        g = 2.0 * math.pi * p.linewidth_hz
        det = w0 * w0 - w * w
        den = det * det + (g * w) ** 2
        if den == 0.0:
            raise ValueError(
                "permittivity_at_hz evaluated exactly ON an undamped Lorentz "
                f"resonance (freq_hz = {freq_hz:g} = resonance, linewidth 0): "
                "eps diverges there")
        return float(self.permittivity + p.delta_eps * w0 * w0 * det / den)


class Box(FrozenModel):
    """Axis-aligned box: full extents ``size_um`` centered on ``center_um``."""

    type: Literal["box"] = "box"
    center_um: Vec3Um
    size_um: Tuple[PositiveUm, PositiveUm, PositiveUm]


class Sphere(FrozenModel):
    type: Literal["sphere"] = "sphere"
    center_um: Vec3Um
    radius_um: PositiveUm


class Cylinder(FrozenModel):
    """Annular sector / solid disk / ring (NUMERICS.md §17). ``axis`` is the
    extrusion (= propagation) axis; the two transverse axes carry the radial
    test. ``inner_radius_um = 0`` is a solid disk; a full 2*pi sweep is a
    ring/cylinder (no angular test). A 90-degree waveguide bend is an annulus
    with a 90-degree sweep. The curved sidewall is exact (faceting-free).
    Angles are in radians, measured by ``atan2(v, u)`` in the transverse
    (u, v) plane. Hard-sampled in Phase 2 (curved subpixel deferred, §16.6)."""

    type: Literal["cylinder"] = "cylinder"
    axis: AxisName
    center_um: Vec3Um
    radius_um: PositiveUm
    inner_radius_um: NonNegativeUm = 0.0
    length_um: PositiveUm
    angle_start: float = 0.0
    angle_stop: float = 2.0 * math.pi

    @model_validator(mode="after")
    def _check(self) -> "Cylinder":
        if not (self.inner_radius_um < self.radius_um):
            raise ValueError(
                f"inner_radius_um ({self.inner_radius_um}) must be < "
                f"radius_um ({self.radius_um})"
            )
        sweep = self.angle_stop - self.angle_start
        if not (0.0 < sweep <= 2.0 * math.pi + 1e-9):
            raise ValueError(
                "angle_stop - angle_start must be in (0, 2*pi], got "
                f"{sweep} (start={self.angle_start}, stop={self.angle_stop})"
            )
        return self


class PolySlab(FrozenModel):
    """Polygon cross-section extruded along ``axis`` with optional slanted
    sidewalls (NUMERICS.md §17). ``vertices_um`` are the ordered (u, v) polygon
    in the two transverse axes (u = lower-indexed, v = higher-indexed),
    counter-clockwise. ``sidewall_angle > 0`` (radians) narrows the
    cross-section toward +axis; the given vertices live at ``reference_plane``.
    Hard-sampled in Phase 2 (curved/polygon subpixel deferred, §16.6)."""

    type: Literal["polyslab"] = "polyslab"
    axis: AxisName
    vertices_um: Tuple[Tuple[float, float], ...] = Field(min_length=3)
    slab_bounds_um: Tuple[float, float]
    sidewall_angle: float = 0.0
    reference_plane: Literal["bottom", "middle", "top"] = "middle"

    @model_validator(mode="after")
    def _check(self) -> "PolySlab":
        lo, hi = self.slab_bounds_um
        if not (hi > lo):
            raise ValueError(f"slab_bounds_um hi ({hi}) must be > lo ({lo})")
        if not (abs(self.sidewall_angle) < math.pi / 2.0):
            raise ValueError(
                f"sidewall_angle ({self.sidewall_angle}) must be in "
                "(-pi/2, pi/2)"
            )
        return self


GeometryType = Annotated[
    Union[Box, Sphere, Cylinder, PolySlab], Field(discriminator="type")
]
StructureName = Annotated[str, Field(min_length=1)]


class Structure(FrozenModel):
    """One geometry filled with one medium; list order is paint order
    (last wins, NUMERICS.md section 9). ``name`` is optional authoring/display
    metadata and never changes rasterization or paint-order semantics."""

    geometry: GeometryType
    medium: Medium
    name: Optional[StructureName] = None
