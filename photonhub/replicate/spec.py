"""``PaperSpec`` — the machine-readable record of a published photonic device,
the front door of the paper-replication workflow.

A ``PaperSpec`` decouples *what a paper reports* (the device parameters, the
material stack, the operating band, the reference figures-of-merit) from *how we
simulate it* (grid resolution, domain sizing, monitor placement — derived later
by :mod:`photonhub.replicate.build`). One YAML file per paper; the fields below are
driven by a real example (Chandran et al., Opt. Lett. 45, 6230 (2020), a
cosine-taper waveguide crossing) rather than invented in the abstract.

    from photonhub.replicate import PaperSpec
    spec = PaperSpec.from_yaml("specs/chandran_cosine_crossing.yaml")

The ``units`` field on every reference is MANDATORY and validated: a paper's
insertion loss / crosstalk are quoted in dB while a raw transmission is linear,
and a silent dB<->linear mix-up is exactly the class of error a replication
workflow exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

__all__ = [
    "PaperSpec",
    "Source",
    "Layer",
    "Stack",
    "Optical",
    "PortRoles",
    "Reference",
    "Device",
    "Convergence",
    "SpecError",
]

# Recognized reference quantities and unit conventions. ``dB`` values are
# 10*log10(power ratio); ``linear`` values are a 0..1 power fraction.
#
# Passive-device quantities attach to a physical port (through/cross). Two other
# device classes describe something that is NOT a waveguide port and so use a
# reserved port name (see :meth:`PaperSpec.validate` and ``_RESERVED_PORTS``):
#   * resonator / cavity quantities (Q, resonance, mode volume) -> port "cavity";
#   * metasurface unit-cell quantities (a periodic meta-atom's plane-wave
#     transmission and transmitted phase) -> port "unit_cell".
# ``Q`` and a normalized mode volume are ``dimensionless``, a resonance is in
# ``nm``, a transmitted phase is in ``radian``/``degree``. Keeping the mandatory
# ``units`` discipline while making a resonator or metasurface paper expressible,
# not just a waveguide transmission device.
_PORT_QUANTITIES = {"insertion_loss", "crosstalk", "transmission", "reflection"}
_CAVITY_QUANTITIES = {"quality_factor", "resonance_wavelength", "mode_volume"}
_METASURFACE_QUANTITIES = {"transmission_phase"}
_QUANTITIES = _PORT_QUANTITIES | _CAVITY_QUANTITIES | _METASURFACE_QUANTITIES
_UNITS = {"dB", "linear", "dimensionless", "nm", "radian", "degree"}
# reserved port name -> the quantities allowed to attach to it
_RESERVED_PORTS = {
    "cavity": _CAVITY_QUANTITIES,
    "unit_cell": _METASURFACE_QUANTITIES | {"transmission", "reflection"},
}
_CAVITY_PORT = "cavity"  # kept for back-compat references
_POLARIZATIONS = {"TE", "TM"}


class SpecError(ValueError):
    """A ``PaperSpec`` is missing a field, self-contradictory, or names an
    unknown quantity/unit/polarization."""


@dataclass(frozen=True)
class Source:
    """Where the device and its reference results come from."""

    citation: str
    doi: str = ""
    arxiv: str = ""
    #: A runnable matched-simulator reference, e.g. ``"tidy3d:WaveguideCrossing"``
    #: (the Tidy3D example that reproduces this paper). Free text; consumed by the
    #: report layer to run a head-to-head comparison when available.
    matched_sim: str = ""

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Source":
        return cls(
            citation=str(d["citation"]),
            doi=str(d.get("doi", "")),
            arxiv=str(d.get("arxiv", "")),
            matched_sim=str(d.get("matched_sim", "")),
        )


@dataclass(frozen=True)
class Layer:
    """One patterned layer of the stack (the device geometry is extruded into
    it). ``material`` is a name resolved at build time — a
    :mod:`photonhub.materials` entry (e.g. ``"cSi"``) or an ``"n=<value>"`` literal
    for a constant index."""

    name: str
    material: str
    zmin_um: float
    thickness_um: float

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Layer":
        return cls(
            name=str(d["name"]),
            material=str(d["material"]),
            zmin_um=float(d["zmin_um"]),
            thickness_um=float(d["thickness_um"]),
        )


@dataclass(frozen=True)
class Stack:
    """The vertical layer stack: the patterned layer(s) plus the surrounding
    cladding/box background. This replaces hardcoded platform constants so a
    paper on any platform (thickness, cladding) is expressible."""

    layers: Tuple[Layer, ...]
    clad_material: str
    box_material: str

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Stack":
        layers = tuple(Layer.from_dict(x) for x in d["layers"])
        if not layers:
            raise SpecError("stack.layers must be non-empty")
        # box defaults to the cladding when a single symmetric background is meant
        clad = str(d["clad_material"])
        return cls(
            layers=layers,
            clad_material=clad,
            box_material=str(d.get("box_material", clad)),
        )

    @property
    def core(self) -> Layer:
        """The primary (thickest) patterned layer — the waveguide core."""
        return max(self.layers, key=lambda l: l.thickness_um)


@dataclass(frozen=True)
class Optical:
    """The operating band and the mode to launch."""

    band_nm: Tuple[float, float]
    center_nm: float
    n_points: int
    polarization: str  # "TE" or "TM"
    mode_index: int = 0
    #: Fit the core material to a dispersive Lorentz pole across the band (to
    #: reproduce a paper's IL(λ) slope), rather than freezing it at band centre.
    dispersive: bool = False

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Optical":
        band = tuple(float(x) for x in d["band_nm"])
        if len(band) != 2:
            raise SpecError("optical.band_nm must be [lo, hi] in nm")
        pol = str(d.get("polarization", "TE")).upper()
        # accept "TE0"/"TM0" style: split trailing mode index
        mode_index = int(d.get("mode_index", 0))
        if pol[:2] in _POLARIZATIONS and pol[2:].isdigit():
            mode_index = int(pol[2:])
            pol = pol[:2]
        return cls(
            band_nm=(band[0], band[1]),
            center_nm=float(d.get("center_nm", 0.5 * (band[0] + band[1]))),
            n_points=int(d.get("n_points", 51)),
            polarization=pol,
            mode_index=mode_index,
            dispersive=bool(d.get("dispersive", False)),
        )

    @property
    def band_um(self) -> Tuple[float, float]:
        return (self.band_nm[0] / 1000.0, self.band_nm[1] / 1000.0)

    @property
    def center_um(self) -> float:
        return self.center_nm / 1000.0


@dataclass(frozen=True)
class PortRoles:
    """Which named device port plays which role in the readout. ``input`` is
    driven; ``through`` is the low-loss output; ``cross`` are the isolated ports
    (crosstalk). Port names must match those the geometry builder emits.

    ``input``/``through`` default to ``"in"``/``"out"`` so a portless device (a
    metasurface unit cell driven by a plane wave, whose readout attaches to the
    reserved ``"unit_cell"`` port) can omit them entirely."""

    input: str = "in"
    through: str = "out"
    cross: Tuple[str, ...] = ()
    reflect: Optional[str] = None  # defaults to the input port read backward

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PortRoles":
        cross = d.get("cross", ())
        if isinstance(cross, str):
            cross = (cross,)
        return cls(
            input=str(d.get("input", "in")),
            through=str(d.get("through", "out")),
            cross=tuple(str(x) for x in cross),
            reflect=None if d.get("reflect") is None else str(d["reflect"]),
        )

    @property
    def all_output(self) -> Tuple[str, ...]:
        return (self.through, *self.cross)


@dataclass(frozen=True)
class Reference:
    """A reported figure-of-merit to compare against. ``units`` is mandatory.

    A reference can carry, in increasing fidelity: a single ``paper_value`` (a
    band-centre number), an optional ``flatness`` (± band, e.g. the paper's
    reported wavelength-flatness), a ``bound`` flag (``paper_value`` is an upper
    or lower limit, not a point — e.g. "crosstalk < −30 dB"), and/or a
    ``curve`` — the digitized paper/reference spectrum as ``(wavelength_nm,
    value)`` pairs, which the report overlays against the ph curve for a true
    curve-vs-curve comparison."""

    quantity: str
    units: str
    port: str
    paper_value: Optional[float] = None
    flatness: Optional[float] = None
    bound: bool = False
    curve: Optional[Tuple[Tuple[float, float], ...]] = None
    label: str = "paper"
    source: str = ""  # e.g. "Fig. 3(b)"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Reference":
        q = str(d["quantity"])
        u = str(d["units"])
        if q not in _QUANTITIES:
            raise SpecError(
                f"reference quantity {q!r} not in {sorted(_QUANTITIES)}"
            )
        if u not in _UNITS:
            raise SpecError(
                f"reference units {u!r} not in {sorted(_UNITS)} — every "
                "reference MUST declare dB or linear (silent mix-ups are the "
                "error class this workflow guards against)"
            )
        curve = d.get("curve")
        if curve is not None:
            curve = tuple((float(w), float(v)) for w, v in curve)
        return cls(
            quantity=q,
            units=u,
            port=str(d["port"]),
            paper_value=None if d.get("paper_value") is None else float(d["paper_value"]),
            flatness=None if d.get("flatness") is None else float(d["flatness"]),
            bound=bool(d.get("bound", False)),
            curve=curve,
            label=str(d.get("label", "paper")),
            source=str(d.get("source", "")),
        )


@dataclass(frozen=True)
class Device:
    """The geometry: a builder ``kind`` (resolved against the geometry registry
    in :mod:`photonhub.replicate.geometry`) plus its parameter dict."""

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Device":
        return cls(kind=str(d["kind"]), params=dict(d.get("params", {})))


@dataclass(frozen=True)
class Convergence:
    """The convergence-gate ladder: escalating cells-per-wavelength (in the core
    medium) until the band-centre metric stops moving within ``tol_pp``."""

    ladder_cpw: Tuple[int, ...] = (15, 20, 25, 30)
    subpixel_method: str = "contour"  # repo default (Tidy3D exact-fill); best for curved walls
    tol_pp: float = 0.3

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Convergence":
        return cls(
            ladder_cpw=tuple(int(x) for x in d.get("ladder_cpw", (15, 20, 25, 30))),
            subpixel_method=str(d.get("subpixel_method", "contour")),
            tol_pp=float(d.get("tol_pp", 0.3)),
        )


@dataclass(frozen=True)
class PaperSpec:
    """The full replication record for one published device."""

    name: str
    source: Source
    device: Device
    stack: Stack
    optical: Optical
    ports: PortRoles
    references: Tuple[Reference, ...] = ()
    convergence: Convergence = field(default_factory=Convergence)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PaperSpec":
        try:
            spec = cls(
                name=str(d["name"]),
                source=Source.from_dict(d["source"]),
                device=Device.from_dict(d["device"]),
                stack=Stack.from_dict(d["stack"]),
                optical=Optical.from_dict(d["optical"]),
                ports=PortRoles.from_dict(d.get("ports", {})),
                references=tuple(Reference.from_dict(x) for x in d.get("references", ())),
                convergence=Convergence.from_dict(d.get("convergence", {})),
            )
        except KeyError as e:
            raise SpecError(f"PaperSpec missing required field: {e}") from None
        spec.validate()
        return spec

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "PaperSpec":
        try:
            import yaml  # lazy: keep the package importable without pyyaml
        except ImportError as exc:  # pragma: no cover
            raise SpecError(
                "PaperSpec.from_yaml needs PyYAML — install `photonhub[replicate]`"
            ) from exc

        text = Path(path).read_text()
        data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise SpecError(f"{path}: top-level YAML must be a mapping")
        return cls.from_dict(data)

    def validate(self) -> None:
        lo, hi = self.optical.band_nm
        if not (0.0 < lo < hi):
            raise SpecError(f"optical.band_nm must be 0 < lo < hi, got {self.optical.band_nm}")
        if not (lo <= self.optical.center_nm <= hi):
            raise SpecError(
                f"optical.center_nm {self.optical.center_nm} outside band {self.optical.band_nm}"
            )
        if self.optical.polarization not in _POLARIZATIONS:
            raise SpecError(
                f"optical.polarization {self.optical.polarization!r} not in {sorted(_POLARIZATIONS)}"
            )
        # Every reference port must be a declared device port role target -- or
        # a reserved port ("cavity"/"unit_cell") whose allowed quantity family the
        # reference's quantity belongs to. A cavity/metasurface-intrinsic quantity
        # on a physical port (or the wrong reserved port) is an error.
        known_ports = {self.ports.input, self.ports.through, *self.ports.cross}
        if self.ports.reflect:
            known_ports.add(self.ports.reflect)
        intrinsic = _CAVITY_QUANTITIES | _METASURFACE_QUANTITIES
        for ref in self.references:
            if ref.port in _RESERVED_PORTS:
                allowed = _RESERVED_PORTS[ref.port]
                if ref.quantity not in allowed:
                    raise SpecError(
                        f"quantity {ref.quantity!r} is not valid on the reserved "
                        f"port {ref.port!r} (allowed: {sorted(allowed)})"
                    )
                continue
            if ref.quantity in intrinsic:
                raise SpecError(
                    f"intrinsic quantity {ref.quantity!r} must use a reserved port "
                    f"{sorted(p for p, q in _RESERVED_PORTS.items() if ref.quantity in q)}, "
                    f"got physical port {ref.port!r}"
                )
            if ref.port not in known_ports:
                raise SpecError(
                    f"reference port {ref.port!r} is not one of the device's "
                    f"role ports {sorted(known_ports)}"
                )
