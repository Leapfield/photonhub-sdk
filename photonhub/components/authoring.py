"""Authoring values that never reach the wire: a port and a declared beam.

A :class:`Port` is the unit of excitation and readout of the setup layer. A :class:`~photonhub.Simulation` given ``ports=`` solves
each port's mode on its own grid at construction and carries the readout
planes; ``source=`` names the driven port, or declares a :class:`GaussianBeam`.
Both are plain frozen dataclasses: the wire document is built from what they
resolve to (``PointDipole`` source planes and ``ProfileMonitor`` planes), so the
schema does not know them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from pydantic import Field, model_validator

from .base import FrozenModel
from .grid import MeshOverride
from .structures import Medium

_AXES = ("x", "y", "z")
_OPPOSITE = {"+": "-", "-": "+"}
_MODE = re.compile(r"^(TE|TM)(\d+)$")


def _hv_axes(axis: str) -> Tuple[int, int]:
    """The natural (horizontal, vertical) in-plane axis indices of a plane
    normal to ``axis``: x-normal (y, z), y-normal (x, z), z-normal (x, y). The
    convention the Yee mode solver and ``ModePort`` use."""
    a = _AXES.index(axis)
    return tuple(i for i in range(3) if i != a)  # type: ignore[return-value]


@dataclass(frozen=True)
class Port:
    """A waveguide port: the plane where a mode is launched or read out.

    ``center_um`` is the port's centre in the user's frame; its component along
    ``axis`` is the readout plane and the other two are the waveguide's
    transverse centre. ``width_um`` and ``thickness_um`` size the default mode
    window, ``width/2 + pad`` by ``thickness/2 + pad``: the pad reaches 20 %
    beyond the distance at which the mode's intensity is 30 dB below its peak,
    at the longest wavelength the port reads, from the mode's effective index
    and the highest cladding index at the core's faces (the rule of
    :func:`photonhub.mode_window_um`, refined by the port's own mode solve;
    NUMERICS §18.8). ``window_um`` gives the two half-extents explicitly. ``mode`` is a family and an index in
    one string (``"TE0"``, ``"TM1"``). ``out_direction`` is the side of the
    plane that faces the domain wall, ``"+"`` or ``"-"``; ``None`` lets the
    simulation infer it from where the port sits relative to the device.
    ``source_offset_um`` is how far behind the plane, toward the wall, a launch
    sits when this port is driven. It is a LENGTH, defaulting to a third of a
    wavelength in the background: what the launch injects that is not the
    guide's discrete mode interferes with it at this port's plane, at a phase
    set by the distance in microns, so a standoff counted in cells would move
    that phase with the mesh and this port's reading would drift with it
    (NUMERICS §18.6). ``medium``
    is the port waveguide's material, used to extend the guide
    through the wall. ``dl_um``, ``supersample`` and ``num_modes`` are the
    mode-solve frame controls.
    """

    name: str
    center_um: Tuple[float, float, float]
    axis: str
    width_um: float
    out_direction: Optional[str] = None
    thickness_um: Optional[float] = None
    thickness_axis: str = "z"
    mode: str = "TE0"
    medium: Optional[Medium] = None
    window_um: Optional[Tuple[float, float]] = None
    source_offset_um: Optional[float] = None
    dl_um: Optional[float] = None
    supersample: int = 8
    num_modes: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name or any(c.isspace() or c in "/\\" for c in self.name):
            raise ValueError(
                f"port name {self.name!r} must be non-empty, with no whitespace or path "
                "separators (the readout monitor is named after it)")
        if self.axis not in _AXES:
            raise ValueError(f"port {self.name!r}: axis must be one of x/y/z, got {self.axis!r}")
        try:
            center = tuple(float(c) for c in self.center_um)
        except (TypeError, ValueError):
            center = ()
        if len(center) != 3:
            raise ValueError(f"port {self.name!r}: center_um must be three numbers (x, y, z)")
        object.__setattr__(self, "center_um", center)
        if not float(self.width_um) > 0.0:
            raise ValueError(f"port {self.name!r}: width_um must be > 0")
        if self.thickness_um is not None and not float(self.thickness_um) > 0.0:
            raise ValueError(f"port {self.name!r}: thickness_um must be > 0")
        if self.thickness_axis not in _AXES:
            raise ValueError(f"port {self.name!r}: thickness_axis must be one of x/y/z")
        if self.thickness_axis == self.axis:
            raise ValueError(
                f"port {self.name!r}: thickness_axis {self.thickness_axis!r} cannot be the "
                f"propagation axis")
        if self.out_direction is not None and self.out_direction not in _OPPOSITE:
            raise ValueError(f"port {self.name!r}: out_direction must be '+', '-' or None")
        m = _MODE.match(str(self.mode).upper())
        if not m:
            raise ValueError(
                f"port {self.name!r}: mode must read like 'TE0' or 'TM1', got {self.mode!r}")
        object.__setattr__(self, "mode", m.group(0))
        if self.window_um is not None:
            try:
                win = tuple(float(w) for w in self.window_um)
            except (TypeError, ValueError):
                win = ()
            if len(win) != 2 or any(not w > 0.0 for w in win):
                raise ValueError(
                    f"port {self.name!r}: window_um must be two positive half-extents "
                    "(half_w_um, half_v_um)")
            object.__setattr__(self, "window_um", win)
        if self.source_offset_um is not None and not float(self.source_offset_um) > 0.0:
            raise ValueError(f"port {self.name!r}: source_offset_um must be > 0")
        if self.dl_um is not None and not float(self.dl_um) > 0.0:
            raise ValueError(f"port {self.name!r}: dl_um must be > 0")
        if not (1 <= int(self.supersample) <= 16):
            raise ValueError(f"port {self.name!r}: supersample must be within 1..16")
        if self.num_modes is not None and int(self.num_modes) < 1:
            raise ValueError(f"port {self.name!r}: num_modes must be >= 1")

    @property
    def monitor_name(self) -> str:
        """The portable filename token that names this port's readout monitor:
        the name itself when it already is one, otherwise ``+`` spelled
        ``_plus`` and any other character outside letters, digits, ``_``,
        ``-``, ``.`` replaced by ``_`` (a library crossing's ``"x+"`` reads out
        as ``"x_plus"``). Results are looked up by the port name, not this."""
        out = self.name.replace("+", "_plus")
        out = "".join(c if (c.isalnum() and c.isascii()) or c in "_-." else "_" for c in out)
        return out.strip(".") or "port"

    @property
    def plane_um(self) -> float:
        """The readout plane: the centre's component along ``axis``."""
        return float(self.center_um[_AXES.index(self.axis)])

    @property
    def in_direction(self) -> Optional[str]:
        """The launch direction when this port is driven, into the device."""
        return None if self.out_direction is None else _OPPOSITE[self.out_direction]

    def family_index(self) -> Tuple[str, int]:
        """``("TE", 0)`` for ``mode="TE0"``."""
        m = _MODE.match(self.mode)
        assert m is not None
        return m.group(1), int(m.group(2))

    def hv_center_um(self) -> Tuple[float, float]:
        """The transverse centre in the plane's natural (horizontal, vertical)
        order, the order the Yee mode solver takes."""
        h, v = _hv_axes(self.axis)
        return float(self.center_um[h]), float(self.center_um[v])

    def with_out_direction(self, out_direction: str) -> "Port":
        """A copy with ``out_direction`` set (the simulation infers it when None)."""
        from dataclasses import replace
        return replace(self, out_direction=out_direction)


@dataclass(frozen=True)
class GaussianBeam:
    """A declared Gaussian-beam excitation on the plane ``axis = position_um``.

    Mirrors :func:`photonhub.analysis.gaussian_beam_source`, which builds the
    beam's exact fields cell by cell on the plane when the beam joins a
    :class:`~photonhub.Simulation` as ``source=``: accurate for a tightly
    focused or tilted beam, and a pair of dipoles per cell of the window. The
    wavelength and the pulse come from the simulation's ``wlens_um``.
    ``paraxial=True`` launches a beam at normal incidence, with its waist on the
    plane, as its Gaussian profile sampled over the plane: one source with the
    medium's index as its ``n_eff``, right when the waist is many wavelengths
    wide (a fibre beam on a lens tens of microns across, where the cell-by-cell
    launch would be millions of dipoles). ``center_um`` is the beam axis in the user's frame (its
    component along ``axis`` is ignored); ``None`` is the domain centre, and
    on an axis with a symmetry plane the symmetry plane. ``power_watts`` is the
    beam power through its plane in watts, as a :class:`~photonhub.PowerMonitor`
    reads it; with symmetry planes it is the whole device's power, like a port
    launch (NUMERICS §20.8). Exactly
    one of ``waist_um`` (1/e field radius) or ``mfd_um`` (1/e² intensity
    diameter) sizes the spot, each a number or an (h, v) pair. Angles are in
    radians.
    """

    axis: str
    position_um: float
    waist_um: Optional[object] = None
    mfd_um: Optional[object] = None
    direction: str = "+"
    power_watts: float = 1.0
    center_um: Optional[Tuple[float, float, float]] = None
    n: Optional[float] = None
    polarization: Optional[str] = None
    pol_angle_rad: Optional[float] = None
    waist_distance_um: float = 0.0
    angle_theta_rad: float = 0.0
    angle_phi_rad: float = 0.0
    half_w_um: Optional[float] = None
    half_v_um: Optional[float] = None
    window_sigmas: float = 3.0
    amplitude_threshold: float = 1e-6
    paraxial: bool = False

    def __post_init__(self) -> None:
        if self.axis not in _AXES:
            raise ValueError(f"GaussianBeam: axis must be one of x/y/z, got {self.axis!r}")
        if self.paraxial and (float(self.angle_theta_rad) != 0.0 or float(self.waist_distance_um) != 0.0
                              or self.pol_angle_rad is not None):
            raise ValueError("GaussianBeam: paraxial=True launches a flat Gaussian profile; it needs normal "
                             "incidence, the waist on the plane and a plain Ex/Ey/Ez polarization")
        if self.direction not in _OPPOSITE:
            raise ValueError("GaussianBeam: direction must be '+' or '-'")
        if (self.waist_um is None) == (self.mfd_um is None):
            raise ValueError("GaussianBeam: pass exactly one of waist_um or mfd_um")
        if not float(self.power_watts) > 0.0:
            raise ValueError("GaussianBeam: power_watts must be > 0")
        if self.center_um is not None:
            try:
                center = tuple(float(c) for c in self.center_um)
            except (TypeError, ValueError):
                center = ()
            if len(center) != 3:
                raise ValueError("GaussianBeam: center_um must be three numbers (x, y, z)")
            object.__setattr__(self, "center_um", center)

    def hv_center_um(self) -> Optional[Tuple[float, float]]:
        """The beam axis in the plane's natural (horizontal, vertical) order, or
        None for the domain centre."""
        if self.center_um is None:
            return None
        h, v = _hv_axes(self.axis)
        return float(self.center_um[h]), float(self.center_um[v])


class Domain(FrozenModel):
    """Margins for a domain fitted around the device.

    On an axis no port exits through, the PML inner face sits ``clearance_um``
    beyond the structures' bounding box (a float, or a per-axis 3-tuple whose
    ``None`` entries take the default): two thirds of a wavelength in the
    background, ``wlen0_um / (1.5 * n_background)``, the same margin the port
    windows use.
    On an axis a port exits through, the face sits ``port_margin_um`` beyond
    the outermost port plane on that side (the launch plane on the driven
    side, the readout plane otherwise); default ten background cells. A
    periodic axis takes its extent from ``period_um`` (a 3-tuple with ``None``
    for the other axes) or from the bounding box, with no margin and no PML.
    ``extent_um`` (a 3-tuple with ``None`` for the other axes) gives an axis
    its interior extent outright, centred on the structures, in place of the
    bounding box and the clearance: a membrane that runs past every wall, one
    period of a cavity array. An entry may also be a ``(low, high)`` pair, the
    two walls at those coordinates of the user's frame: a substrate that fills
    the domain below a device and air above it, each running into its own
    boundary layers.
    The PML thickness is added outside every open face, the extent rounds to
    the nearest whole background cell (the one cell of a uniform mesh), and
    the simulation records where the wire's corner landed in ``origin_um``. A
    port whose guide stops short of its wall has the guide continued through
    the boundary layers in the port's width, thickness and medium.
    """

    clearance_um: Optional[Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]] = None
    port_margin_um: Optional[float] = Field(default=None, gt=0)
    period_um: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None
    extent_um: Optional[Tuple[Optional[Union[float, Tuple[float, float]]],
                              Optional[Union[float, Tuple[float, float]]],
                              Optional[Union[float, Tuple[float, float]]]]] = None

    @model_validator(mode="after")
    def _positive(self) -> "Domain":
        vals = (self.clearance_um,) if isinstance(self.clearance_um, (int, float)) else (self.clearance_um or ())
        for v in vals:
            if v is not None and not float(v) > 0.0:
                raise ValueError("Domain.clearance_um entries must be > 0 (or None for the default)")
        for v in (self.period_um or ()):
            if v is not None and not float(v) > 0.0:
                raise ValueError("Domain.period_um entries must be > 0 (or None)")
        for v in (self.extent_um or ()):
            if v is None:
                continue
            if isinstance(v, tuple):
                if not float(v[1]) > float(v[0]):
                    raise ValueError("Domain.extent_um (low, high) walls need high > low")
            elif not float(v) > 0.0:
                raise ValueError("Domain.extent_um entries must be > 0, a (low, high) pair, or None")
        return self

    def extent(self, axis_index: int) -> Optional[float]:
        """The explicit interior extent for one axis (the length between two
        given walls too), or ``None``."""
        e = self.extent_um
        if e is None or e[axis_index] is None:
            return None
        v = e[axis_index]
        return float(v[1]) - float(v[0]) if isinstance(v, tuple) else float(v)

    def walls(self, axis_index: int) -> Optional[Tuple[float, float]]:
        """The two walls given outright for one axis, in the user's frame, or
        ``None`` when the axis has none (a plain extent is centred on the
        structures instead)."""
        e = self.extent_um
        if e is None or not isinstance(e[axis_index], tuple):
            return None
        return float(e[axis_index][0]), float(e[axis_index][1])

    def clearance(self, axis_index: int) -> Optional[float]:
        """The explicit clearance for one axis, or ``None`` for the default."""
        c = self.clearance_um
        if c is None:
            return None
        if isinstance(c, (int, float)):
            return float(c)
        return None if c[axis_index] is None else float(c[axis_index])


class Mesh(FrozenModel):
    """A mesh declared by resolution, resolved when it joins a
    :class:`~photonhub.Simulation`.

    Exactly one of ``cells_per_wlen`` (cells per wavelength in each medium,
    counted at the band centre ``wlen0_um``) and ``dl_um`` (an explicit cell
    size, uniform), or both when ``dl_um`` is a per-axis 3-tuple: an axis with
    a number is uniform at that spacing (a lattice-commensurate mesh), an axis
    with ``None`` is graded by ``cells_per_wlen``. ``uniform=True`` with ``cells_per_wlen`` gives a
    :class:`~photonhub.UniformMesh` at ``wlen0_um / (n_max * cells_per_wlen)``,
    ``n_max`` the highest structure index; otherwise :func:`~photonhub.auto_mesh`
    grades the mesh with ``refine`` as its overrides, ``dl_min_um`` as its
    floor and ``max_grading`` as its cell-to-cell ratio. A mesh by
    ``cells_per_wlen`` needs the simulation's ``wlens_um`` (or ``wlen0_um``).
    """

    cells_per_wlen: Optional[float] = Field(default=None, gt=0)
    dl_um: Optional[Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]] = None
    refine: Tuple[MeshOverride, ...] = ()
    dl_min_um: Optional[float] = Field(default=None, gt=0)
    max_grading: float = Field(default=1.4, gt=1.0)
    uniform: bool = False

    @model_validator(mode="after")
    def _one_resolution(self) -> "Mesh":
        if isinstance(self.dl_um, tuple):
            given = [v for v in self.dl_um if v is not None]
            if not given or any(not float(v) > 0.0 for v in given):
                raise ValueError("Mesh.dl_um as a tuple needs at least one positive spacing (None for a graded axis)")
            if len(given) < 3 and self.cells_per_wlen is None:
                raise ValueError("Mesh: an axis with dl_um=None is graded by cells_per_wlen; pass it too")
            if len(given) == 3 and self.cells_per_wlen is not None:
                raise ValueError("Mesh: every axis has its spacing; cells_per_wlen would set none")
            if self.uniform:
                raise ValueError("Mesh: per-axis dl_um already says which axes are uniform; drop uniform=True")
            return self
        if self.dl_um is not None and not float(self.dl_um) > 0.0:
            raise ValueError("Mesh.dl_um must be > 0")
        if (self.cells_per_wlen is None) == (self.dl_um is None):
            raise ValueError("Mesh: pass exactly one of cells_per_wlen or dl_um")
        if self.dl_um is not None and (self.refine or self.dl_min_um is not None):
            raise ValueError("Mesh: refine and dl_min_um go with cells_per_wlen (a graded mesh), not dl_um")
        return self

    def spacing(self, axis_index: int) -> Optional[float]:
        """The explicit spacing for one axis (``None`` for a graded axis, or
        when the mesh is by ``cells_per_wlen`` alone)."""
        d = self.dl_um
        if d is None:
            return None
        if isinstance(d, tuple):
            return None if d[axis_index] is None else float(d[axis_index])
        return float(d)
