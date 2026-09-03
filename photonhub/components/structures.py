"""Geometry/material structures (NUMERICS.md sections 9-10).

``structures`` is an ordered list; materials are rasterized per E component
at that component's own Yee point and the LAST structure containing the
point wins (containment is closed). Geometries may extend beyond the domain
— only the part inside the grid matters — so no domain check applies here.
"""

import math
from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

from .base import AxisName, FrozenModel, NonNegativeUm, PositiveUm, Vec3Um


# NUMERICS.md §19 — hard per-medium pole budget (lorentz + poles + drude
# combined); mirrors the engine's kMaxAdePoles.
MAX_ADE_POLES = 6


class LorentzPole(FrozenModel):
    """One Lorentz pole of a frequency-dependent (dispersive) medium
    (NUMERICS.md §19). Under the engine's e^{-i omega t} convention the pole
    contributes

        chi(omega) = delta_eps * omega0^2
                     / (omega0^2 - omega^2 - i*gamma*omega)

    where omega0 = 2*pi*resonance_frequency_hz and gamma = 2*pi*linewidth_hz
    (both stored as ordinary Hz on the wire; the engine multiplies by 2*pi).
    ``delta_eps`` is the oscillator strength (the static contribution of this
    pole, delta of eps(0)). A medium carries up to ``MAX_ADE_POLES`` poles
    combined across ``lorentz``/``poles``/``drude``.

    Passivity (Im eps >= 0 for omega > 0) requires delta_eps >= 0 and
    gamma >= 0; the engine validates these. ``gamma = 0`` is a lossless
    (undamped) resonance — allowed, but the timestep must stay clear of the
    resonance for stability (NUMERICS.md §19 omega0*dt/2 < 1)."""

    resonance_frequency_hz: float = Field(gt=0.0)  # f0 = omega0 / 2pi
    delta_eps: float = Field(ge=0.0)               # oscillator strength
    linewidth_hz: float = Field(default=0.0, ge=0.0)  # gamma / 2pi


class DrudePole(FrozenModel):
    """One Drude (free-carrier) term of a dispersive medium (NUMERICS.md §19)
    — the metal/plasmonics building block. Under e^{-i omega t} it contributes

        chi(omega) = -wp^2 / (omega^2 + i*gamma*omega)

    with wp = 2*pi*plasma_frequency_hz and gamma = 2*pi*linewidth_hz (the
    collision rate; 0 = collisionless plasma). Below the plasma frequency the
    real permittivity goes strongly NEGATIVE — the medium reflects like a
    metal. The engine realizes it as the omega0 = 0 ADE pole with strength
    wp^2 (no resonance-Nyquist bound applies)."""

    plasma_frequency_hz: float = Field(gt=0.0)        # wp / 2pi
    linewidth_hz: float = Field(default=0.0, ge=0.0)  # gamma / 2pi


class PermittivityArray(FrozenModel):
    """Node-based permittivity data grid for a custom medium (NUMERICS.md
    §10.3, schema 1.18). ``shape`` = (nx, ny, nz), each >= 2; ``values`` is
    the flat C-order (x-major) array of nx*ny*nz node values, each >= 1.
    Node [0, 0, 0] sits ON the structure box's low corner and
    [nx-1, ny-1, nz-1] on the high corner; the engine trilinear-interpolates
    at its §9 Yee sample points. Prefer :meth:`Medium.from_eps_array`."""

    shape: Tuple[int, int, int]
    values: Tuple[float, ...] = Field(min_length=8, repr=False)

    @model_validator(mode="after")
    def _consistent(self) -> "PermittivityArray":
        if any(n < 2 for n in self.shape):
            raise ValueError(
                f"permittivity_data shape must be >= 2 per axis (node-based "
                f"trilinear grid), got {self.shape}")
        want = self.shape[0] * self.shape[1] * self.shape[2]
        if want > (1 << 24):
            raise ValueError(
                f"permittivity_data carries {want} nodes — exceeds the "
                "16.7M inline-wire cap")
        if len(self.values) != want:
            raise ValueError(
                f"permittivity_data values length {len(self.values)} != "
                f"shape product {want} (C order, x-major)")
        bad = next((v for v in self.values
                    if not (v >= 1.0) or v != v or v in (float("inf"),)),
                   None)
        if bad is not None:
            raise ValueError(
                f"permittivity_data values must be finite and >= 1, "
                f"found {bad}")
        return self


class Medium(FrozenModel):
    """Isotropic nonmagnetic medium: scalar relative permittivity plus an
    electric conductivity entering the lossy Ca/Cb update (NUMERICS.md
    section 10). ``sigma = 0`` reproduces the Phase-0 update bit-exactly.

    A non-dispersive medium leaves ``lorentz``/``poles``/``drude`` unset and
    the fields are omitted from the wire entirely (back-compat: schema < 1.9
    documents and every existing scene round-trip byte-identically). With any
    poles supplied, ``permittivity`` is the high-frequency limit eps_inf
    (NUMERICS.md §19) and the medium is dispersive (the ADE polarization
    update engages only in cells carrying poles). The engine consumes the
    poles in WIRE ORDER — ``lorentz`` first, then ``poles``, then ``drude`` —
    up to ``MAX_ADE_POLES`` combined."""

    permittivity: float = Field(ge=1.0)
    conductivity_s_per_m: float = Field(default=0.0, ge=0.0)
    # NUMERICS.md §19 — the legacy single Lorentz pole (additive/back-compat;
    # kept alongside the lists so schema 1.9-1.16 documents parse unchanged).
    lorentz: Optional[LorentzPole] = None
    # Schema 1.17 — additional Lorentz poles and Drude terms (additive;
    # None/empty are equivalent and omitted from the wire, so every existing
    # document round-trips byte-identically).
    poles: Optional[Tuple[LorentzPole, ...]] = None
    drude: Optional[Tuple[DrudePole, ...]] = None
    # Schema 1.18 — optional DIAGONAL permittivity tensor (eps_xx, eps_yy,
    # eps_zz), each >= 1: uniaxial/biaxial crystals on the axis-aligned
    # principal frame (LiNbO3: ordinary on two axes, extraordinary on the
    # optic axis). When set, ``permittivity`` is required by the wire but
    # IGNORED (write e.g. the mean). v1 exclusions: no dispersion poles, no
    # pec, and the engine requires subpixel=False (the Simulation default
    # flips off automatically for anisotropic scenes). None omitted from the
    # wire (byte-back-compat).
    permittivity_xyz: Optional[Tuple[float, float, float]] = None
    # Schema 1.17 — perfect electric conductor STRUCTURE material (NUMERICS.md
    # §10.1): every E component whose Yee point falls inside is pinned to 0
    # (staircased hard mirror — the structure analogue of the 'pec' outer
    # boundary). ``permittivity`` is required by the wire but ignored (write 1);
    # conductivity and dispersion poles are contradictions and rejected.
    # None/False are equivalent and omitted from the wire (byte-back-compat).
    pec: Optional[bool] = None
    # Schema 1.18 — optional spatially-varying ISOTROPIC permittivity
    # (NUMERICS.md §10.3): a node-based data grid spanning the structure's
    # Box extent, trilinear-interpolated at the engine's §9 sample points.
    # When set, ``permittivity`` is required by the wire but IGNORED (write
    # e.g. the mean). Build with :meth:`Medium.from_eps_array`. v1
    # exclusions: Box geometry only, no dispersion poles, no pec, no
    # permittivity_xyz, subpixel off (Simulation auto-flips), CPU solver
    # only. None omitted from the wire (byte-back-compat).
    permittivity_data: Optional["PermittivityArray"] = None

    @model_validator(mode="after")
    def _aniso_rules(self) -> "Medium":
        if self.permittivity_xyz is not None:
            if any(not (e >= 1.0) for e in self.permittivity_xyz):
                raise ValueError(
                    "permittivity_xyz components must each be >= 1, got "
                    f"{self.permittivity_xyz}")
            if self.is_dispersive:
                raise ValueError(
                    "anisotropic media cannot carry dispersion poles "
                    "(anisotropic dispersion is deferred)")
            if self.pec:
                raise ValueError("permittivity_xyz contradicts pec")
        return self

    @model_validator(mode="after")
    def _custom_data_rules(self) -> "Medium":
        if self.permittivity_data is not None:
            if self.is_dispersive:
                raise ValueError(
                    "custom media cannot carry dispersion poles (deferred; "
                    "NUMERICS.md §10.3)")
            if self.pec:
                raise ValueError("permittivity_data contradicts pec")
            if self.permittivity_xyz is not None:
                raise ValueError(
                    "permittivity_data contradicts permittivity_xyz "
                    "(anisotropic custom media are deferred)")
        return self

    @property
    def is_anisotropic(self) -> bool:
        return self.permittivity_xyz is not None

    @property
    def is_custom(self) -> bool:
        return self.permittivity_data is not None

    @classmethod
    def from_eps_array(cls, values, *, conductivity_s_per_m: float = 0.0
                       ) -> "Medium":
        """A custom medium from a 3-D array of permittivity node values
        (NUMERICS.md §10.3). ``values`` is any (nx, ny, nz) nested sequence /
        numpy array with nx, ny, nz >= 2 and every value >= 1; node [0,0,0]
        sits on the structure box's low corner, node [-1,-1,-1] on the high
        corner, C order on the wire. ``permittivity`` is set to the mean
        (the wire requires it; the engine ignores it)."""
        import numpy as _np

        arr = _np.asarray(values, dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                f"from_eps_array needs a 3-D (nx, ny, nz) array, got shape "
                f"{arr.shape}")
        return cls(
            permittivity=max(1.0, float(arr.mean())),
            conductivity_s_per_m=conductivity_s_per_m,
            permittivity_data=PermittivityArray(
                shape=tuple(int(n) for n in arr.shape),
                values=tuple(float(v) for v in arr.reshape(-1))),
        )

    @model_validator(mode="after")
    def _pec_excludes_other_response(self) -> "Medium":
        if self.pec is False:
            object.__setattr__(self, "pec", None)  # canonical omitted form
        if self.pec:
            if self.is_dispersive:
                raise ValueError(
                    "a PEC medium cannot carry dispersion poles (it is the "
                    "infinite-conductivity limit already)")
            if self.conductivity_s_per_m != 0.0:
                raise ValueError(
                    "a PEC medium cannot carry a finite conductivity")
        return self

    @model_validator(mode="after")
    def _pole_budget(self) -> "Medium":
        # canonicalize empty lists to None (the omitted-from-wire form)
        if self.poles is not None and len(self.poles) == 0:
            object.__setattr__(self, "poles", None)
        if self.drude is not None and len(self.drude) == 0:
            object.__setattr__(self, "drude", None)
        n = ((1 if self.lorentz is not None else 0)
             + len(self.poles or ()) + len(self.drude or ()))
        if n > MAX_ADE_POLES:
            raise ValueError(
                f"medium carries {n} dispersion poles (lorentz + poles + "
                f"drude combined); the engine supports at most "
                f"{MAX_ADE_POLES} (NUMERICS.md §19)")
        return self

    @property
    def is_dispersive(self) -> bool:
        """True when the medium carries any ADE pole (Lorentz or Drude)."""
        return (self.lorentz is not None or bool(self.poles)
                or bool(self.drude))

    def all_lorentz_poles(self) -> Tuple[LorentzPole, ...]:
        """Every Lorentz pole in wire order (legacy ``lorentz`` first)."""
        head = (self.lorentz,) if self.lorentz is not None else ()
        return head + tuple(self.poles or ())

    def permittivity_at_hz(self, freq_hz: float) -> float:
        """Real relative permittivity ``Re eps(omega)`` at ``freq_hz`` under
        the §19 pole model — for a non-dispersive medium this is just
        ``permittivity``; with poles, ``permittivity`` alone is only the
        high-frequency limit eps_inf and is the WRONG value to hand a mode
        solver or any other frequency-anchored consumer (a Si pole fit reads
        ~9.6 instead of ~12.1 at 1.55 um). Sums every Lorentz and Drude term;
        can go NEGATIVE for a metal below its plasma frequency. Raises at an
        exactly undamped Lorentz resonance, where eps diverges."""
        if not self.is_dispersive:
            return float(self.permittivity)
        if not freq_hz > 0.0:
            raise ValueError(f"freq_hz must be > 0, got {freq_hz}")
        w = 2.0 * math.pi * float(freq_hz)
        eps = float(self.permittivity)
        for p in self.all_lorentz_poles():
            w0 = 2.0 * math.pi * p.resonance_frequency_hz
            g = 2.0 * math.pi * p.linewidth_hz
            det = w0 * w0 - w * w
            den = det * det + (g * w) ** 2
            if den == 0.0:
                raise ValueError(
                    "permittivity_at_hz evaluated exactly ON an undamped "
                    f"Lorentz resonance (freq_hz = {freq_hz:g} = resonance, "
                    "linewidth 0): eps diverges there")
            eps += p.delta_eps * w0 * w0 * det / den
        for d in (self.drude or ()):
            wp = 2.0 * math.pi * d.plasma_frequency_hz
            g = 2.0 * math.pi * d.linewidth_hz
            eps += -(wp * wp) / (w * w + g * g)
        return float(eps)


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


class Polygon(FrozenModel):
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
    def _check(self) -> "Polygon":
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
    Union[Box, Sphere, Cylinder, Polygon], Field(discriminator="type")
]
StructureName = Annotated[str, Field(min_length=1)]


class Structure(FrozenModel):
    """One geometry filled with one medium; list order is paint order
    (last wins, NUMERICS.md section 9). ``name`` is optional authoring/display
    metadata and never changes rasterization or paint-order semantics."""

    geometry: GeometryType
    medium: Medium
    name: Optional[StructureName] = None

    @field_validator("medium", mode="before")
    @classmethod
    def _medium_not_material(cls, v):
        # Beta papercuts (duck-typed: materials.py imports Medium from here,
        # the reverse import would cycle). A fit object (PoleFit/LorentzFit)
        # carries its engine Medium as a property — coerce it losslessly. A
        # Material is a fit FACTORY whose .medium needs arguments — pydantic's
        # generic model_type error gives no way forward, so name the
        # conversion explicitly.
        fitted = getattr(v, "medium", None)
        if isinstance(fitted, Medium) and not isinstance(v, Medium):
            return fitted
        if type(v).__name__ == "Material" and callable(fitted):
            raise ValueError(
                f"got the materials-library entry {getattr(v, 'name', v)!r} "
                "where a Medium is required — pick the fit first: "
                ".medium(wavelength_um=...) for a constant index at one "
                "wavelength, or .medium(band_um=(lo, hi)) for a dispersive "
                "fit over a band (e.g. "
                "ph.materials.Si.medium(band_um=(1.5, 1.6)))")
        return v

    @model_validator(mode="after")
    def _custom_medium_needs_box(self) -> "Structure":
        # NUMERICS.md §10.3: the data grid spans the structure's Box extent.
        if getattr(self.medium, "is_custom", False) and (
                self.geometry.type != "box"):
            raise ValueError(
                "permittivity_data (custom media) requires a Box geometry — "
                f"got '{self.geometry.type}' (NUMERICS.md §10.3)")
        return self
