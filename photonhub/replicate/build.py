"""Spec -> Simulation: assemble a faithfully-set-up FDTD run from a
:class:`~photonhub.replicate.spec.PaperSpec`.

This is the "metric -> simulation" seam the audit found missing at the library
level: the domain sizing, PML clearance, mode-source launch, and per-port mode
monitors that the legacy benchmark buried inside its one-paper ``_engine`` are
derived here from the spec, for any registered device.

    from photonhub.replicate import PaperSpec
    from photonhub.replicate.build import build_simulation
    spec = PaperSpec.from_yaml("specs/chandran_cosine_crossing.yaml")
    built = build_simulation(spec, cells_per_wavelength=20)
    data = built.run(device="cpu")          # or hand built.sim to the GPU
    metrics = built.metrics_db(data)        # insertion loss + crosstalk in dB

Coordinate model: the geometry builder emits an origin-centered device; here it
is placed at the domain center of a corner-origin ``[0, size]`` box, the routing
arms run out through the PML on every face, and the mode planes sit one clearance
inside the PML.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ..components.medium import Background, Boundaries
from ..components.run import RunSpec
from ..components.simulation import Simulation
from ..components.source_time import GaussianPulse
from ..components.sources import PointDipole
from ..components.structures import Medium, Structure
from ..components.grid import UniformGridSpec, realized_cells
from ..components.monitors import FieldDftMonitor
from .. import materials as _materials
from ..plugins.mode_devices import ModeMonitor, mode_launch, mode_monitor, transmission
from ..plugins.yee_mode import solve_yee_mode
from .geometry import build_geometry
from .spec import PaperSpec

__all__ = ["BuiltSim", "build_simulation"]

_C0 = 2.99792458e8  # m/s, the shared speed-of-light constant

# Faithful-setup clearances (microns), referenced to the PML inner edge.
_SRC_CLEARANCE_UM = 0.4   # source plane -> PML inner edge
_MON_GAP_UM = 0.4         # input monitor downstream of the source
_CLAD_PAD_Z_UM = 0.8      # clear cladding above/below the core (z), each side
_ARM_SAFETY_UM = 0.3      # extra straight stub past the monitor plane
_TRANS_PAD_UM = 0.6       # in-plane transverse clearance: outermost guide edge -> PML


def _shaped_half_extent(params: Mapping[str, object]) -> float:
    """Half-length (from device centre) of the shaped/patterned region along the
    propagation axis — the routing arm must reach past this so a mode plane sits
    in straight single-mode guide. A device kind states it directly via
    ``shaped_half_extent_um``; the crossing default is ``junction_width/2 +
    taper_length`` (its shaped region), kept for back-compatibility."""
    if "shaped_half_extent_um" in params:
        return float(params["shaped_half_extent_um"])
    return 0.5 * float(params["junction_width_um"]) + float(params["taper_length_um"])


def _resolve_index(material: str, wavelength_um: float) -> float:
    """Refractive index of a material name at a wavelength. Accepts a
    :mod:`photonhub.materials` entry (``"cSi"``) or an ``"n=<value>"`` literal."""
    material = material.strip()
    if material.lower().startswith("n="):
        return float(material[2:])
    return float(_materials.get(material).n(wavelength_um))


def _resolve_medium(
    material: str, wavelength_um: float, *, band_um: Optional[Tuple[float, float]] = None
) -> Medium:
    """A :class:`Medium` for a material name. With ``band_um`` set, a DISPERSIVE
    single-pole Lorentz fit over that band (so the index tracks its true
    wavelength dependence across the sweep); otherwise a non-dispersive medium
    frozen at ``wavelength_um``. An ``"n=<value>"`` literal is always
    non-dispersive. A dispersive scene needs a stabilized PML (the dispersive x
    CFS-inert-PML late-time instability); :func:`build_simulation` applies it."""
    material = material.strip()
    if material.lower().startswith("n="):
        return Medium(permittivity=float(material[2:]) ** 2)
    mat = _materials.get(material)
    if band_um is not None:
        return mat.medium(band_um=band_um)
    return mat.medium(wavelength_um=wavelength_um)


def _band_freqs_hz(band_um: Tuple[float, float], n_points: int) -> Tuple[float, ...]:
    """``n_points`` frequencies (Hz) evenly spaced in WAVELENGTH across the band
    (the paper's sampling), returned ascending in frequency."""
    lam = np.linspace(band_um[0], band_um[1], n_points)
    freqs = _C0 / (lam * 1e-6)
    return tuple(sorted(float(f) for f in freqs))


@dataclass(frozen=True)
class BuiltSim:
    """A ready-to-run :class:`Simulation` plus the mode monitors that read it.

    ``in_monitor`` sits just after the source on the input arm; ``out_monitors``
    maps a role (``"through"`` and each cross-port name) to its output monitor.
    ``metrics_db`` ratios each output against the input and converts to dB."""

    sim: Simulation
    in_monitor: ModeMonitor
    out_monitors: Mapping[str, ModeMonitor]
    freqs_hz: Tuple[float, ...]
    meta: Mapping[str, object] = field(default_factory=dict)

    @property
    def wavelengths_um(self) -> Tuple[float, ...]:
        return tuple(_C0 / f * 1e6 for f in self.freqs_hz)

    def transmissions(self, data) -> Dict[str, Dict[float, float]]:
        """``{role: {freq_hz: T}}`` — each output port's modal power transmission
        relative to the input plane."""
        return {
            role: transmission(mon, self.in_monitor, data)
            for role, mon in self.out_monitors.items()
        }

    def reflection(self, data) -> Dict[float, float]:
        """Back-reflection ``{freq_hz: R}`` at the input plane: the input monitor
        read BACKWARD (reflected modal power) over the same plane read FORWARD
        (incident power). The directional modal overlap isolates each sense and
        the per-mode normalization cancels (one plane, one mode), so this is the
        true power reflection — the natural readout for a Bragg/DBR stopband.
        Empty when the build recorded no backward direction."""
        back = self.meta.get("in_backward_dir")
        fwd = self.meta.get("in_forward_dir")
        if not back or not fwd:
            return {}
        p_back = self.in_monitor.mode_power(data, direction=str(back))
        p_fwd = self.in_monitor.mode_power(data, direction=str(fwd))
        return {f: p_back[f] / p_fwd[f]
                for f in p_fwd if f in p_back and p_fwd[f] > 0}

    def metrics_db(self, data) -> Dict[str, object]:
        """Insertion loss (dB, through port) and crosstalk (dB, each cross port)
        as arrays over :attr:`wavelengths_um`, plus the raw transmissions."""
        trans = self.transmissions(data)
        freqs = self.freqs_hz
        out: Dict[str, object] = {
            "wavelengths_um": list(self.wavelengths_um),
            "transmission": {r: [t[f] for f in freqs] for r, t in trans.items()},
        }
        through = self.meta.get("through_role", "through")
        if through in trans:
            out["insertion_loss_db"] = [
                -10.0 * math.log10(max(trans[through][f], 1e-300)) for f in freqs
            ]
        out["crosstalk_db"] = {
            role: [10.0 * math.log10(max(t[f], 1e-300)) for f in freqs]
            for role, t in trans.items()
            if role != through
        }
        refl = self.reflection(data)
        if refl:
            out["reflection"] = [refl.get(f, float("nan")) for f in freqs]
            out["reflection_db"] = [
                10.0 * math.log10(max(refl.get(f, 1e-300), 1e-300)) for f in freqs
            ]
        return out

    def run(self, *, device: Optional[str] = None, **run_kwargs):
        """Run locally / on ``device`` via :func:`photonhub.run_local` and return
        the ``SimulationData``. A thin convenience; hand ``self.sim`` to any
        runner (cloud GPU) instead if you prefer."""
        from ..runners.local import run_local

        return run_local(self.sim, device=device, **run_kwargs)

    def field_intensity(self, data):
        """The in-plane ``|E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2`` on the xy field-slice
        monitor (present only when the sim was built with ``field_slice=True``),
        as ``(x_um, y_um, E2)`` with ``E2`` a 2-D ``[y, x]`` array normalized to
        its peak — or ``None`` if no field slice was recorded."""
        name = self.meta.get("field_slice_name")
        if not name:
            return None
        da = data[name]
        e2 = (np.abs(da) ** 2).sum("component")
        for d in list(e2.dims):        # drop the singleton frequency / normal axis
            if e2.sizes[d] == 1:
                e2 = e2.isel({d: 0})
        sd = list(e2.dims)             # remaining two spatial dims, [y, x] order
        x = np.asarray(e2[sd[-1]].values)
        y = np.asarray(e2[sd[0]].values)
        z = np.asarray(e2.transpose(sd[0], sd[-1]).values, dtype=float)
        peak = float(z.max()) or 1.0
        return x, y, z / peak


def _outward_sign(port_name: str) -> float:
    """+1 if the port faces the +axis end, -1 for the -axis end."""
    return 1.0 if port_name.endswith("+") else -1.0


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _snap_transverse_faces(
    mode_monitor: ModeMonitor, *, dl_um: float, pml_layers: int
) -> ModeMonitor:
    """Pull a mode monitor's TRANSVERSE plane faces inside the PML and land each
    on a ``(k + 1/4)*dl`` position (NUMERICS.md §12).

    A full-domain plane touches the realized boundary (rejected as outside); an
    integer-cell inset lands the faces exactly on a cell boundary, where the
    different Yee components (Ey vs Ez) snap to opposite cells (rejected as an
    ambiguous per-component region). Quarter-cell faces are unambiguous — every
    component snaps to the same cell — and clear the absorbing layers. The mode
    window (centered on the guide) is unchanged; only the plane extent shrinks."""
    fm = mode_monitor.field_monitor
    ai = _AXIS_INDEX[mode_monitor.axis]
    center = list(fm.center_um)
    size = list(fm.size_um)
    for i in range(3):
        if i == ai:
            continue
        n = realized_cells(fm.size_um[i], dl_um)  # transverse extent = domain size
        k_lo = pml_layers + 2
        k_hi = n - (pml_layers + 3)
        lo = (k_lo + 0.25) * dl_um
        hi = (k_hi + 0.25) * dl_um
        center[i] = 0.5 * (lo + hi)
        size[i] = hi - lo
    new_fm = fm.model_copy(update={"center_um": tuple(center), "size_um": tuple(size)})
    return dataclasses.replace(mode_monitor, field_monitor=new_fm)


def build_simulation(
    spec: PaperSpec,
    *,
    cells_per_wavelength: float = 20.0,
    pml_num_layers: int = 12,
    subpixel_method: Optional[str] = None,
    run_periods: float = 60.0,
    shutoff: float = 1e-7,
    dispersive: Optional[bool] = None,
    field_slice: bool = False,
) -> BuiltSim:
    """Assemble a :class:`BuiltSim` for ``spec`` at ``cells_per_wavelength``
    (cells per wavelength IN THE CORE — the convergence-ladder knob).

    ``subpixel_method`` overrides ``spec.convergence.subpixel_method`` (the
    contour operator for the curved walls). ``shutoff`` is set low by default
    because a faithful crosstalk (-30 dB and below) needs the field to decay well
    past the cross-port floor before the run stops.

    ``dispersive`` (defaults to ``spec.optical.dispersive``) fits the CORE
    material to a single-pole Lorentz over the band so the index tracks its true
    wavelength dependence — needed to reproduce a paper's IL(λ) *slope*, not just
    its band-centre level. A dispersive scene is switched to a stabilized CFS-PML
    (the dispersive-pole × CFS-inert-PML late-time instability).
    """
    opt = spec.optical
    lam_c = opt.center_um
    core_layer = spec.stack.core
    thickness = core_layer.thickness_um

    if dispersive is None:
        dispersive = spec.optical.dispersive
    n_core = _resolve_index(core_layer.material, lam_c)
    n_clad = _resolve_index(spec.stack.clad_material, lam_c)
    n_box = _resolve_index(spec.stack.box_material, lam_c)
    # The CORE tracks material dispersion across the band when requested (the
    # dominant λ-dependence); the SiO2 background dispersion is negligible in-band
    # and stays a scalar index.
    core_medium = _resolve_medium(
        core_layer.material, lam_c, band_um=opt.band_um if dispersive else None)

    # Grid from cells-per-wavelength in the core (the highest-index medium).
    dl = lam_c / (cells_per_wavelength * n_core)
    pml_um = pml_num_layers * dl

    # Device parameters. Extend the routing arm so a mode plane fits in straight
    # guide between the shaped region and the PML.
    params = dict(spec.device.params)
    inner = _shaped_half_extent(params)
    needed_arm = inner + pml_um + _SRC_CLEARANCE_UM + _MON_GAP_UM + _ARM_SAFETY_UM
    sim_arm = max(float(params["arm_length_um"]), needed_arm)
    params["arm_length_um"] = sim_arm

    # Domain (corner-origin), sized PER IN-PLANE AXIS from the device's port
    # layout (probed at the origin): an axis carrying a routing port spans
    # arm-tip to arm-tip (the waveguide runs out through the PML on that face);
    # a purely transverse axis needs only the outermost guide edge + a field pad
    # + the PML. This makes the box RECTANGULAR — a long thin grating or a
    # transversely-split Y-branch no longer pays for a square domain. Each extent
    # is snapped to a whole number of cells so ``size_um`` equals the realized
    # ``n*dl`` domain. z is core + cladding pad + PML on each side.
    probe = build_geometry(
        spec.device.kind, params, medium=core_medium,
        thickness_um=thickness, center_um=(0.0, 0.0, 0.0),
    )

    def _half_extent(ai: int) -> float:
        along = max((abs(p.center_um[ai]) for p in probe.ports
                     if _AXIS_INDEX[p.axis] == ai), default=0.0)
        trans = max((abs(p.center_um[ai]) + 0.5 * p.width_um for p in probe.ports
                     if _AXIS_INDEX[p.axis] != ai), default=0.0)
        pad = trans + _TRANS_PAD_UM + pml_um
        return max(along, pad) if along > 0.0 else pad

    size_x = realized_cells(2.0 * _half_extent(0), dl) * dl
    size_y = realized_cells(2.0 * _half_extent(1), dl) * dl
    size_z = realized_cells(thickness + 2.0 * _CLAD_PAD_Z_UM + 2.0 * pml_um, dl) * dl
    cx, cy, cz = size_x / 2.0, size_y / 2.0, size_z / 2.0

    geom = build_geometry(
        spec.device.kind, params, medium=core_medium,
        thickness_um=thickness, center_um=(cx, cy, cz),
    )

    # Optional substrate box when the box index differs from the cladding
    # (asymmetric stack). Symmetric SiO2 clad/box (this device) uses a uniform
    # background and needs none.
    structures = list(geom.structures)
    if abs(n_box - n_clad) > 1e-6:
        structures.insert(0, Structure(
            geometry=_substrate_box(max(size_x, size_y), cz, thickness),
            medium=Medium(permittivity=n_box ** 2),
        ))

    freqs_hz = _band_freqs_hz(opt.band_um, opt.n_points)
    pulse = GaussianPulse.for_band(wavelengths_um=list(opt.band_um), freq0_hz=_C0 / (lam_c * 1e-6))

    # Solve the routing-waveguide TE0 mode once (square-ish core -> same mode on
    # x- and y-normal planes; resampled per plane).
    grid = UniformGridSpec(dl_um=dl)
    boundaries = Boundaries(x="pml", y="pml", z="pml")
    method = subpixel_method or spec.convergence.subpixel_method

    # A shell sim (placeholder dipole) so solve_yee_mode / mode_launch can read
    # the scene's own rasterized eps; the final sim swaps in the launch sheet and
    # the mode monitors. It carries the final subpixel settings so the Yee mode
    # sees the eps the engine will run.
    shell = Simulation(
        size_um=(size_x, size_y, size_z), grid=grid, run=RunSpec(n_steps=1),
        background=Background(permittivity=n_clad ** 2), boundaries=boundaries,
        pml_num_layers=pml_num_layers, subpixel=True, subpixel_method=method,
        structures=tuple(structures),
        sources=(PointDipole(center_um=(cx, cy, cz), polarization="Ex", source_time=pulse),),
    )

    ports = {p.name: p for p in geom.ports}
    center_xyz = (cx, cy, cz)
    _sizes = (size_x, size_y, size_z)

    def _sign(port) -> float:
        """+1 if the port faces the +axis end of its axis, -1 for the -axis end
        (derived from the port's own position, not a name suffix — so a port need
        not be on the domain axis nor encode its sign in its name)."""
        ai = _AXIS_INDEX[port.axis]
        return 1.0 if port.center_um[ai] >= center_xyz[ai] else -1.0

    def plane_pos(port, offset: float) -> float:
        """Along-axis coordinate of a mode plane ``offset`` inside the PML inner
        edge on this port's face."""
        ai = _AXIS_INDEX[port.axis]
        return port.center_um[ai] - _sign(port) * (pml_um + offset)

    # A transversely-OFFSET arm (a Y-branch output) must launch/read on ITS OWN
    # guide, not the domain centre — so the eq-current sheet and the mode-overlap
    # need the port's transverse position. The two APIs disagree on axis order:
    # ``mode_launch`` takes ``center_um`` in in-plane-axes order (x→(y,z), y→(x,z));
    # the mode monitor takes it in _TRANSVERSE order (x→(y,z), y→(z,x)) — they
    # SWAP for a y-cut. For a centred port both equal the domain centre (so the
    # crossing is byte-for-byte unchanged).
    _INPLANE = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}
    _TRANSV = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}

    def _centre(port, order):
        a, b = order[port.axis]
        c = (port.center_um[_AXIS_INDEX[a]], port.center_um[_AXIS_INDEX[b]])
        dflt = (_sizes[_AXIS_INDEX[a]] / 2.0, _sizes[_AXIS_INDEX[b]] / 2.0)
        # A centred port defaults (center_um=None) to the domain centre — return
        # None there so the launch/readout is byte-identical to the pre-offset
        # path (the crossing is unchanged); only an OFFSET arm gets an explicit
        # centre.
        if abs(c[0] - dflt[0]) < 1e-9 and abs(c[1] - dflt[1]) < 1e-9:
            return None
        return c

    def launch_center(port):
        return _centre(port, _INPLANE)

    def monitor_center(port):
        return _centre(port, _TRANSV)

    # Yee-native routing mode: solve_yee_mode reads the scene's OWN staggered eps
    # -> a grid-consistent discrete eigenmode (the construction Tidy3D/Lumerical
    # use). The window is centred on the PORT's transverse position, so a
    # transversely-offset arm (a Y-branch output) reads its own guide, not the
    # domain axis. Cached per (axis, transverse cell) since collinear same-offset
    # ports share a cross-section.
    def yee_window(port) -> dict:
        width_ax = "y" if port.axis == "x" else "x"  # in-plane transverse; z = thickness
        wi = _AXIS_INDEX[width_ax]
        return dict(
            h_center_um=port.center_um[wi],
            v_center_um=cz,
            half_w_um=min(port.width_um / 2 + 0.5, _sizes[wi] / 2 - pml_um - 2 * dl),
            half_v_um=min(thickness / 2 + 0.6, size_z / 2 - pml_um - 2 * dl),
            dl_um=dl,
        )

    yee_cache: Dict[object, object] = {}

    def yee_mode(port, at_plane: float):
        width_ax = "y" if port.axis == "x" else "x"
        key = (port.axis, round(port.center_um[_AXIS_INDEX[width_ax]] / dl))
        if key not in yee_cache:
            yee_cache[key] = solve_yee_mode(
                shell, port.axis, at_plane, lam_c, opt.polarization, opt.mode_index,
                **yee_window(port))
        return yee_cache[key]

    in_port = ports[spec.ports.input]
    in_axis = in_port.axis
    inward_dir = "+" if _sign(in_port) < 0 else "-"
    outward_dir = "-" if inward_dir == "+" else "+"

    # Source: equivalence-current (Huygens) sheet from the Yee mode, injecting
    # INWARD (opposite the input port's outward sign).
    src_pos = plane_pos(in_port, _SRC_CLEARANCE_UM)
    in_mode = yee_mode(in_port, src_pos)
    launch_sources = mode_launch(
        shell, in_mode, axis=in_axis, position_um=src_pos,
        source_time=pulse, direction=inward_dir, power_watts=1.0,
        center_um=launch_center(in_port),
    )

    # Input monitor one gap downstream of the source (total-field side). Read
    # FORWARD (its stored ``inward_dir``) it is the incident power; the SAME plane
    # read BACKWARD (``outward_dir``) is the reflected power (directional modal
    # overlap, P_mode cancels) — reflection needs no extra monitor plane.
    in_mon = mode_monitor(
        shell, in_mode, axis=in_axis,
        position_um=plane_pos(in_port, _SRC_CLEARANCE_UM + _MON_GAP_UM),
        freqs_hz=freqs_hz, name="in", direction=inward_dir,
        center_um=monitor_center(in_port),
    )

    out_monitors: Dict[str, ModeMonitor] = {}
    role_of = {spec.ports.through: "through"}
    for c in spec.ports.cross:
        role_of[c] = c  # cross ports keyed by their own name
    for port_name, role in role_of.items():
        p = ports[port_name]
        pos = plane_pos(p, _SRC_CLEARANCE_UM)
        out_monitors[role] = mode_monitor(
            shell, yee_mode(p, pos), axis=p.axis, position_um=pos,
            freqs_hz=freqs_hz, name=f"out_{role}", direction="+" if _sign(p) > 0 else "-",
            center_um=monitor_center(p),
        )

    # Pull every monitor plane's transverse faces inside the PML, onto (k+1/4)*dl
    # positions so they never touch the boundary or a cell edge (NUMERICS.md §12).
    in_mon = _snap_transverse_faces(in_mon, dl_um=dl, pml_layers=pml_num_layers)
    out_monitors = {
        r: _snap_transverse_faces(m, dl_um=dl, pml_layers=pml_num_layers)
        for r, m in out_monitors.items()
    }

    # Run duration: cover several transits of the domain in the core (the longest
    # in-plane span — the propagation length for a long grating), plus rely on the
    # low shutoff to end early once the field has decayed.
    transit_s = (max(size_x, size_y) * 1e-6 * n_core) / _C0
    run_time_s = run_periods * transit_s

    monitors = tuple(m.field_monitor for m in (in_mon, *out_monitors.values()))
    # Optional xy field slice (for the |E|^2 intensity figure): a band-centre DFT
    # over the whole in-plane at the core mid-height. Its z-plane is snapped to
    # (k+1/4)*dl so every Yee component lands in one cell (NUMERICS.md §12).
    field_slice_name = None
    field_slice_cf = None
    if field_slice:
        field_slice_cf = min(freqs_hz, key=lambda f: abs(f - _C0 / (lam_c * 1e-6)))
        z_plane = (round(cz / dl - 0.25) + 0.25) * dl
        # Pull the in-plane slice faces to (k+1/4)*dl inside the PML — the same
        # §12 snap the mode monitors use. A full-domain plane both pokes past the
        # realized boundary (non-dl-commensurate rectangular box) AND lands on cell
        # edges where Ex/Ey snap to different cells (rejected); quarter-cell faces
        # avoid both and the slice covers the physical (non-PML) region.
        def _slice_span(n_cells: int):
            k_lo, k_hi = pml_num_layers + 2, n_cells - (pml_num_layers + 3)
            lo, hi = (k_lo + 0.25) * dl, (k_hi + 0.25) * dl
            return 0.5 * (lo + hi), hi - lo
        fxc, fxs = _slice_span(realized_cells(size_x, dl))
        fyc, fys = _slice_span(realized_cells(size_y, dl))
        monitors = monitors + (FieldDftMonitor(
            name="field_xy", center_um=(fxc, fyc, z_plane),
            size_um=(fxs, fys, 0.0), fields=("Ex", "Ey", "Ez"),
            freqs_hz=(field_slice_cf,)),)
        field_slice_name = "field_xy"
    sim = Simulation(
        size_um=(size_x, size_y, size_z),
        grid=grid,
        run=RunSpec(run_time_s=run_time_s, shutoff=shutoff),
        background=Background(permittivity=n_clad ** 2),
        boundaries=boundaries,
        pml_num_layers=pml_num_layers,
        subpixel=True,
        subpixel_method=method,
        structures=tuple(structures),
        sources=tuple(launch_sources),
        monitors=monitors,
    )
    # A dispersive (Lorentz-pole) core needs a stabilized CFS-PML or the
    # pole-assisted resonance the plain sigma-graded PML re-amplifies drives a
    # late-time divergence (the cpw=30 case). PR #158 auto-applies the aligned
    # profile at construction — but ONLY when no PML knob is user-set, and we DO
    # pass pml_num_layers (to size the domain around the mode planes), which
    # suppresses it. So apply the same Tidy3D-StablePML-aligned alpha+kappa here,
    # keeping our layer count (with_stabilized_pml's default 40-layer slab would
    # collide with the monitor planes).
    if dispersive:
        sim = sim.with_stabilized_pml(num_layers=pml_num_layers)

    meta = {
        "dl_um": dl,
        "cells_per_wavelength": cells_per_wavelength,
        "n_core": n_core,
        "n_clad": n_clad,
        "n_box": n_box,
        "wavelength_c_um": lam_c,
        "pml_um": pml_um,
        "sim_arm_um": sim_arm,
        "size_um": (size_x, size_y, size_z),
        "n_eff_TE0": float(in_mode.n_eff),
        "in_forward_dir": inward_dir,
        "in_backward_dir": outward_dir,
        "dispersive_core": bool(dispersive),
        "field_slice_name": field_slice_name,
        "field_slice_freq_hz": field_slice_cf,
        "launch": "eq_current_yee",
        "n_launch_sources": len(launch_sources),
        "through_role": "through",
        "subpixel_method": method,
    }
    return BuiltSim(sim=sim, in_monitor=in_mon, out_monitors=out_monitors,
                    freqs_hz=freqs_hz, meta=meta)


def _substrate_box(size_xy: float, cz: float, thickness: float):
    """A box filling everything below the core centerline (for an asymmetric
    box/clad stack)."""
    from ..components.structures import Box

    half_below = cz - thickness / 2.0
    return Box(
        center_um=(size_xy / 2.0, size_xy / 2.0, half_below / 2.0),
        size_um=(4.0 * size_xy, 4.0 * size_xy, max(half_below, 1e-3)),
    )
