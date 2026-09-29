"""Single-frequency discrete-Yee adjoint gradients for inverse design.

The default design uses scalar pixel boxes; trilinear custom media are optional.
Each E-component sample of permittivity has a density derivative. The frequency
derivative uses the leapfrog frequency `(2/dt)*sin(pi*f*dt)` and the
engine's source-normalized DFT fields.
"""

from __future__ import annotations

import warnings
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .components import (
    Box,
    ProfileMonitor,
    Medium,
    PointDipole,
    Simulation,
    Structure,
)
from .data import RunResult
from .components.simulation import realized_cells
from .components.grid import UniformMesh
from .constants import c0 as _C0, eps0 as _EPS0
from .runners import run_local
from .analysis.mode_devices import _TANGENTIAL, mode_monitor, mode_source
from .analysis.mode_overlap import mode_amplitude
from .analysis.mode_overlap import (
    _plane_component, _widths_with_grid, modal_fields, vector_modal_fields,
)
from .analysis.modes import Mode
from .analysis.vector_modes import VectorMode
from ._compat import caller_stacklevel

# Deprecated legacy import names. Gradients derive their coefficients from
# each simulation; these sentinel values are never used by the driver.
BETA: complex = 1.0 + 0.0j
BETA_MODE: complex = 1.0 + 0.0j


def _warn_objective_beta(beta: complex, placeholder: complex) -> None:
    if beta != placeholder:
        warnings.warn(
            "the objective's beta field is deprecated and ignored; the gradient "
            "derives its coefficient from the simulation",
            DeprecationWarning, stacklevel=caller_stacklevel())


def _axis_box(c0: int, c1: int, dl: float, quarter: bool) -> Tuple[float, float]:
    """(center, size) microns for a box spanning integer cells ``[c0, c1)``.

    ``quarter=False`` puts faces on ``k*dl``. Yee nodes on shared faces
    require the engine's inclusive, last-wins ownership rule (§9).
    ``quarter=True`` shifts the faces to ``(k+0.25)*dl`` so a *multi-component*
    DFT monitor snaps Ex/Ey/Ez to the same cells (§12)."""
    off = 0.25 if quarter else 0.0
    lo = (c0 + off) * dl
    hi = (c1 + off) * dl
    return 0.5 * (lo + hi), (hi - lo)


@dataclass(frozen=True)
class DesignRegion:
    """A rectangular topology-optimization region tiled into ``shape`` pixels.

    The region occupies integer Yee-cell ranges ``[i0,i1) x [j0,j1) x [k0,k1)``
    on a uniform grid of pitch ``dl_um``; ``shape = (npx, npy, npz)`` pixels must
    divide each range evenly. Each pixel is a density ``rho in [0, 1]`` mapped to
    a relative permittivity ``eps = eps_min + rho*(eps_max - eps_min)``. The
    controls become quarter-cell-offset pixel boxes by default, so no Yee E
    sample lies on any pixel face. ``box_face_offset_cells=0`` retains integer
    faces and uses exact engine-style face ownership. ``parametrization="trilinear"`` selects one
    trilinear custom medium instead and is CPU only. ``eps_min >= 1``.

    Build with :meth:`on_grid` from physical microns; the raw constructor takes
    cell indices.
    """

    dl_um: float
    cells: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]  # (i0,i1)...
    shape: Tuple[int, int, int]
    eps_min: float = 1.0
    eps_max: float = 12.25  # ~Si at 1.55 um
    parametrization: str = "boxes"
    box_face_offset_cells: float = 0.25
    # background permittivity for the *gap* between region and pixels is the
    # simulation's own background; pixels fully tile the region so every region
    # cell belongs to a pixel.

    def __post_init__(self) -> None:
        for (a0, a1), n, ax in zip(self.cells, self.shape, "xyz"):
            if a1 <= a0:
                raise ValueError(f"DesignRegion {ax} range {(a0, a1)} is empty")
            if n < 1:
                raise ValueError(f"DesignRegion {ax} pixel count {n} < 1")
            if (a1 - a0) % n != 0:
                raise ValueError(
                    f"DesignRegion {ax}: {a1 - a0} cells do not divide evenly "
                    f"into {n} pixels (cells per pixel must be integer)"
                )
        if self.eps_min < 1.0:
            raise ValueError("eps_min must be >= 1 (vacuum); the client forbids "
                             "sub-vacuum permittivity")
        if self.eps_max < self.eps_min:
            raise ValueError("eps_max must be >= eps_min")
        if self.parametrization not in ("boxes", "trilinear"):
            raise ValueError("parametrization must be 'boxes' or 'trilinear'")
        if self.box_face_offset_cells not in (0.0, 0.25):
            raise ValueError("box_face_offset_cells must be 0 or 0.25")
        if self.parametrization == "trilinear" and self.box_face_offset_cells != 0.25:
            raise ValueError("trilinear requires box_face_offset_cells=0.25")

    # -- construction -------------------------------------------------------

    @classmethod
    def on_grid(
        cls,
        *,
        center_um: Tuple[float, float, float],
        size_um: Tuple[float, float, float],
        dl_um: float,
        shape: Tuple[int, int, int],
        eps_min: float = 1.0,
        eps_max: float = 12.25,
        parametrization: str = "boxes",
        box_face_offset_cells: float = 0.25,
    ) -> "DesignRegion":
        """Snap a physical region (centre/size in microns) to integer cells.

        The low corner snaps to ``round((center-size/2)/dl)`` and the span to
        ``round(size/dl)`` cells, then rounded UP so each axis divides into its
        pixel count. :attr:`size_um` and :attr:`center_um` describe the snapped
        integer-cell extent; box faces shift by ``box_face_offset_cells``."""
        cells = []
        for c, s, n in zip(center_um, size_um, shape):
            lo = int(round((c - 0.5 * s) / dl_um))
            span = int(round(s / dl_um))
            span += (-span) % n  # round span up to a multiple of n
            cells.append((lo, lo + span))
        return cls(dl_um=dl_um, cells=tuple(cells), shape=shape,
                   eps_min=eps_min, eps_max=eps_max,
                   parametrization=parametrization,
                   box_face_offset_cells=box_face_offset_cells)

    # -- geometry -----------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Number of density parameters: the product of the pixel-grid dimensions."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def cells_per_pixel(self) -> Tuple[int, int, int]:
        """Number of solver cells along each axis of one design pixel."""
        return tuple((a1 - a0) // n
                     for (a0, a1), n in zip(self.cells, self.shape))  # type: ignore

    @property
    def size_um(self) -> Tuple[float, float, float]:
        """Realized physical extent along each axis, in microns."""
        return tuple((a1 - a0) * self.dl_um for a0, a1 in self.cells)  # type: ignore

    @property
    def center_um(self) -> Tuple[float, float, float]:
        """Center of the realized design region, in microns."""
        return tuple(0.5 * (a0 + a1) * self.dl_um
                     for a0, a1 in self.cells)  # type: ignore

    def eps(self, rho: np.ndarray) -> np.ndarray:
        """Per-pixel relative permittivity for densities ``rho in [0,1]``."""
        return self.eps_min + np.asarray(rho) * (self.eps_max - self.eps_min)

    def structures(self, rho: np.ndarray) -> Tuple[Structure, ...]:
        """Material structures for this region's parametrization."""
        if self.parametrization == "boxes":
            return self.box_structures(rho)
        return self.trilinear_structure(rho)

    def trilinear_structure(self, rho: np.ndarray) -> Tuple[Structure, ...]:
        """One CPU-only trilinear custom-medium box for the density grid.

        The control at node ``(i,j,k)`` is density
        ``rho[min(i,nx-1),min(j,ny-1),min(k,nz-1)]``. The box faces are a
        quarter cell past integer planes. No Yee E node lies on a face.
        """
        rho = np.asarray(rho, dtype=float).reshape(self.shape)
        node = np.indices(tuple(n + 1 for n in self.shape))
        owners = tuple(np.minimum(node[a], self.shape[a] - 1)
                       for a in range(3))
        eps = self.eps(rho)[owners]
        lo = np.array([a for a, _ in self.cells], dtype=float)
        hi = np.array([b for _, b in self.cells], dtype=float)
        low_um = (lo + 0.25) * self.dl_um
        high_um = (hi + 0.25) * self.dl_um
        return (Structure(
            geometry=Box(center_um=tuple((low_um + high_um) / 2),
                         size_um=tuple(high_um - low_um)),
            medium=Medium.from_eps_array(eps)),)

    def box_structures(self, rho: np.ndarray) -> Tuple[Structure, ...]:
        """Pixel boxes with quarter-cell faces unless integer faces are selected."""
        rho = np.asarray(rho, dtype=float).reshape(self.shape)
        eps = self.eps(rho)
        (i0, _), (j0, _), (k0, _) = self.cells
        cpx, cpy, cpz = self.cells_per_pixel
        out: List[Structure] = []
        for ix in range(self.shape[0]):
            for iy in range(self.shape[1]):
                for iz in range(self.shape[2]):
                    cx, sx = _axis_box(i0 + ix * cpx, i0 + (ix + 1) * cpx,
                                       self.dl_um, quarter=self.box_face_offset_cells == 0.25)
                    cy, sy = _axis_box(j0 + iy * cpy, j0 + (iy + 1) * cpy,
                                       self.dl_um, quarter=self.box_face_offset_cells == 0.25)
                    cz, sz = _axis_box(k0 + iz * cpz, k0 + (iz + 1) * cpz,
                                       self.dl_um, quarter=self.box_face_offset_cells == 0.25)
                    out.append(Structure(
                        geometry=Box(center_um=(cx, cy, cz),
                                     size_um=(sx, sy, sz)),
                        medium=Medium(permittivity=float(eps[ix, iy, iz]))))
        return tuple(out)

    def monitor(self, freq_hz: float, name: str = "design_region"
                ) -> ProfileMonitor:
        """The DFT monitor recording all three E-components over the region
        (quarter-cell faces so the components co-snap, §12)."""
        (i0, i1), (j0, j1), (k0, k1) = self.cells
        cx, sx = _axis_box(i0, i1, self.dl_um, quarter=True)
        cy, sy = _axis_box(j0, j1, self.dl_um, quarter=True)
        cz, sz = _axis_box(k0, k1, self.dl_um, quarter=True)
        return ProfileMonitor(
            name=name, center_um=(cx, cy, cz), size_um=(sx, sy, sz),
            fields=("Ex", "Ey", "Ez"), freqs_hz=(freq_hz,))

    # -- cell -> pixel scatter map ------------------------------------------

    def _pixel_index(self, da) -> np.ndarray:
        """For each recorded cell in DFT DataArray ``da`` (dims ..z,y,x), the
        flat pixel index, or -1 if the cell lies outside the tiled region (an
        edge cell the §12 snap recorded just past the high face). Row-major
        ``[ix, iy, iz]`` to match :meth:`structures`."""
        (i0, i1), (j0, j1), (k0, k1) = self.cells
        cpx, cpy, cpz = self.cells_per_pixel
        npx, npy, npz = self.shape

        def axis_pix(coord_um, a0, a1, cp, npix):
            idx = np.round(np.asarray(coord_um) / self.dl_um).astype(int)
            p = (idx - a0) // cp
            p[(idx < a0) | (idx >= a1)] = -1
            return p

        px = axis_pix(da.coords["x"].values, i0, i1, cpx, npx)
        py = axis_pix(da.coords["y"].values, j0, j1, cpy, npy)
        pz = axis_pix(da.coords["z"].values, k0, k1, cpz, npz)
        # outer combine into (z, y, x) grid matching da's spatial dims
        PX = px[None, None, :]
        PY = py[None, :, None]
        PZ = pz[:, None, None]
        flat = (PX * npy + PY) * npz + PZ  # row-major [ix,iy,iz]
        bad = (PX < 0) | (PY < 0) | (PZ < 0)
        flat = np.where(bad, -1, flat)
        return flat  # shape (nz, ny, nx)

    def _box_pixel_index(self, da, component: str) -> np.ndarray:
        """Pixel owner at each component's Yee point, or -1 outside boxes."""
        offsets = {"Ex": (0.5, 0.0, 0.0),
                   "Ey": (0.0, 0.5, 0.0),
                   "Ez": (0.0, 0.0, 0.5)}
        off = offsets[component]
        if self.box_face_offset_cells == 0.0:
            # At an integer face, two inclusive boxes can both contain a Yee
            # point. Compare the same SI coordinates and box arithmetic as
            # the engine, then let the later structure overwrite the owner.
            axes_si = [
                (np.rint(np.asarray(da.coords[key]) / self.dl_um) + off[a])
                * (self.dl_um * 1e-6)
                for a, key in enumerate("xyz")
            ]
            x = axes_si[0][None, None, :]
            y = axes_si[1][None, :, None]
            z = axes_si[2][:, None, None]
            owner = np.full((len(axes_si[2]), len(axes_si[1]),
                             len(axes_si[0])), -1, dtype=int)
            for pi, structure in enumerate(self.box_structures(
                    np.zeros(self.shape))):
                center = np.asarray(structure.geometry.center_um) * 1e-6
                size = np.asarray(structure.geometry.size_um) * 1e-6
                lo, hi = center - size / 2, center + size / 2
                mask = ((x >= lo[0]) & (x <= hi[0]) &
                        (y >= lo[1]) & (y <= hi[1]) &
                        (z >= lo[2]) & (z <= hi[2]))
                owner[mask] = pi
            return owner
        axes = []
        for a, key in enumerate("xyz"):
            index = np.rint(np.asarray(da.coords[key]) / self.dl_um)
            low = self.cells[a][0] + self.box_face_offset_cells
            high = self.cells[a][1] + self.box_face_offset_cells
            point = index + off[a]
            owner = np.floor((point - low) / self.cells_per_pixel[a]).astype(int)
            owner = np.minimum(owner, self.shape[a] - 1)
            owner[(point < low) | (point > high)] = -1
            axes.append(owner)
        px, py, pz = axes
        PX, PY, PZ = px[None, None, :], py[None, :, None], pz[:, None, None]
        flat = (PX * self.shape[1] + PY) * self.shape[2] + PZ
        return np.where((PX < 0) | (PY < 0) | (PZ < 0), -1, flat)

    def _density_weights(self, da, component: str) -> np.ndarray:
        """Exact d(eps at Yee point)/d(rho), without contrast.

        Shape is ``(n_params,nz,ny,nx)``. This mirrors the engine's
        ``eps_data_at`` interpolation and the node-to-control replication in
        :meth:`structures`.
        """
        offsets = {"Ex": (0.5, 0.0, 0.0),
                   "Ey": (0.0, 0.5, 0.0),
                   "Ez": (0.0, 0.0, 0.5)}
        off = np.asarray(offsets[component])
        lo = np.asarray([a for a, _ in self.cells], dtype=float)
        hi = np.asarray([b for _, b in self.cells], dtype=float)
        box_lo = lo + self.box_face_offset_cells
        box_hi = hi + self.box_face_offset_cells
        ns = np.asarray(self.shape) + 1
        shape = (da.sizes["z"], da.sizes["y"], da.sizes["x"])
        out = np.zeros((self.n_params, *shape), dtype=float)
        # DFT coordinates are base-cell positions for every component.
        ix = np.rint(np.asarray(da.coords["x"]) / self.dl_um).astype(int)
        iy = np.rint(np.asarray(da.coords["y"]) / self.dl_um).astype(int)
        iz = np.rint(np.asarray(da.coords["z"]) / self.dl_um).astype(int)
        if self.parametrization == "boxes":
            owner = self._box_pixel_index(da, component)
            for pi in range(self.n_params):
                out[pi] = owner == pi
            return out
        for kz, z in enumerate(iz):
            for jy, y in enumerate(iy):
                for xi, x in enumerate(ix):
                    point = np.asarray((x, y, z), dtype=float) + off
                    if np.any(point < box_lo) or np.any(point > box_hi):
                        continue
                    t = (point - box_lo) / (box_hi - box_lo) * (ns - 1)
                    low = np.minimum(t.astype(int), ns - 2)
                    q = t - low
                    for sx in (0, 1):
                        for sy in (0, 1):
                            for sz in (0, 1):
                                corner = np.asarray((sx, sy, sz))
                                node = low + corner
                                pix = tuple(min(node[a], self.shape[a] - 1)
                                            for a in range(3))
                                flat = np.ravel_multi_index(pix, self.shape)
                                out[flat, kz, jy, xi] += np.prod(
                                    np.where(corner, q, 1.0 - q))
        return out


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointIntensity:
    """Maximize ``|E_comp(probe)|^2`` at a single point and frequency, the
    simplest adjoint objective: its adjoint source is a single point dipole at
    the probe, polarized along ``component``, with post-multiplied coefficient
    ``conj(u)`` (``u`` = the forward phasor at the probe).

    A focusing/concentrator design (push energy to a focal point) is exactly
    this objective.
    """

    probe_um: Tuple[float, float, float]
    freq_hz: float
    component: str = "Ez"
    name: str = "probe"

    #: Deprecated and ignored; the driver derives its coefficient.
    beta: complex = BETA

    def __post_init__(self) -> None:
        _warn_objective_beta(self.beta, BETA)

    def monitor(self) -> ProfileMonitor:
        """Build the single-point DFT monitor for this objective component and frequency."""
        return ProfileMonitor(
            name=self.name, center_um=self.probe_um, size_um=(0.0, 0.0, 0.0),
            fields=(self.component,), freqs_hz=(self.freq_hz,))

    def amplitude(self, data: RunResult) -> complex:
        """The complex forward objective phasor ``u = E_comp(probe)``."""
        da = data[self.name].sel(component=self.component, f=self.freq_hz)
        return complex(np.asarray(da.values).reshape(()).item())

    def value(self, data: RunResult) -> float:
        """Figure of merit ``|u|^2``."""
        return float(abs(self.amplitude(data)) ** 2)

    def adjoint_source(self, forward_sim: Simulation) -> PointDipole:
        """Unit point dipole at the probe, reusing the forward pulse so the §12
        normalization matches (the objective coefficient is applied in
        post-processing, not in the source amplitude)."""
        pulse = forward_sim.sources[0].source_time
        return PointDipole(center_um=self.probe_um, polarization=self.component,
                           amplitude=1.0, source_time=pulse)

    def adjoint_coeff(self, data: RunResult) -> complex:
        """The complex excitation coefficient ``conj(u)`` applied to the
        forward·adjoint field product when assembling the gradient."""
        return complex(np.conjugate(self.amplitude(data)))


@dataclass(frozen=True)
class ModePower:
    """Maximize the power coupled into a guided ``mode`` at an output port.

    ``J = |c|^2``, where ``c`` is the P_mode-normalized complex modal amplitude
    of the recorded plane (``mode_amplitude``). This is THE objective for
    waveguide inverse design: bends, mode converters, (de)multiplexers,
    grating couplers.

    .. warning:: ``J`` is a RELATIVE objective, not a calibrated power
       transmission. ``|c|^2 == T`` would require the launch to read
       ``c_in == 1`` on an input plane, but ``mode_source(power_watts=1)``
       normalizes to SI watts, so ``J`` carries a scene/grid-dependent
       positive scale (measured ``~4/dl_um`` on SOI strip scenes: J = 23.4
       at dl = 0.05 um where T ~ 0.29). Maximizing J still maximizes T;
       optimization and relative comparisons are unaffected, but do NOT
       report J as transmission; use the S-matrix path
       (:func:`photonhub.analysis.smatrix`), which normalizes by the driven
       port's incident amplitude and is calibrated (|S21|^2).

    The adjoint uses electric and magnetic point currents built from the
    transpose of the actual four-field overlap, including transverse Yee
    co-location for scalar modes. The recording monitor excludes transverse PML rows.
    """

    mode: Union[Mode, VectorMode]
    axis: str                 # propagation axis (x/y/z)
    position_um: float        # output plane position along `axis`
    freq_hz: float
    direction: str = "+"      # "+" = power flowing toward +axis is "transmitted"
    name: str = "mode_out"
    center_um: Optional[Tuple[float, float]] = None
    thickness_axis: Optional[str] = None

    #: Deprecated and ignored; the driver derives its coefficient.
    beta: complex = BETA_MODE

    def __post_init__(self) -> None:
        _warn_objective_beta(self.beta, BETA_MODE)

    def _mm(self, simulation: Simulation):
        return mode_monitor(
            simulation, self.mode, axis=self.axis, position_um=self.position_um,
            freqs_hz=[self.freq_hz], name=self.name, direction=self.direction,
            center_um=self.center_um, thickness_axis=self.thickness_axis)

    def monitor(self, simulation: Simulation) -> ProfileMonitor:
        """Four-field output plane with transverse PML rows excluded.

        The finite-aperture overlap differs from the former full-plane
        objective; it keeps the adjoint current cloud in the physical
        interior where the reciprocal Yee operator applies.
        """
        monitor = self._mm(simulation).field_monitor
        if not isinstance(simulation.grid, UniformMesh):
            return monitor
        dl = simulation.grid.dl_um
        center = list(monitor.center_um)
        size = list(monitor.size_um)
        for a, name in enumerate("xyz"):
            if name == self.axis or getattr(simulation.boundaries, name) != "pml":
                continue
            n = round(simulation.size_um[a] / dl)
            layers = simulation.pml_num_layers
            low = (0.0 if simulation.symmetry[a] else (layers + 2.25) * dl)
            high = (n - layers - 2.75) * dl
            if high <= low:
                continue
            center[a] = (low + high) / 2
            size[a] = high - low
        return monitor.model_copy(update={"center_um": tuple(center),
                                          "size_um": tuple(size)})

    def amplitude(self, data: RunResult) -> complex:
        """The complex normalized modal amplitude ``c`` on the output plane.

        NOTE: this reads WITHOUT the longitudinal Yee de-stagger that
        ``ModeMonitor.mode_power`` applies by default (``mode_amplitude``'s
        ``destagger_dl`` is left off). The objective and its adjoint
        coefficient ``conj(c)`` use the SAME reading, so the gradient stays
        self-consistent; only the absolute |c|² differs from ``mode_power``'s
        default reading by the O(beta*dl/2) stagger term."""
        da = data[self.name]
        planes = {c: da.sel(component=c) for c in _TANGENTIAL[self.axis]}
        c = mode_amplitude(planes, self.mode, axis=self.axis,
                           direction=self.direction, center_um=self.center_um,
                           thickness_axis=self.thickness_axis)
        return complex(c[self.freq_hz])

    def value(self, data: RunResult) -> float:
        """Figure of merit ``|c|^2``, relative modal power (not a calibrated
        transmission; see the class warning)."""
        return float(abs(self.amplitude(data)) ** 2)

    def adjoint_source(self, forward_sim: Simulation):
        """Legacy backward mode launch for direct callers.

        Gradient evaluation uses :meth:`transpose_sources` instead.
        """
        back = "-" if self.direction == "+" else "+"
        pulse = forward_sim.sources[0].source_time
        return mode_source(
            forward_sim, self.mode, axis=self.axis, position_um=self.position_um,
            source_time=pulse, direction=back, amplitude=1.0,
            center_um=self.center_um, thickness_axis=self.thickness_axis)

    def adjoint_coeff(self, data: RunResult) -> complex:
        """The complex excitation coefficient ``conj(c)``."""
        return complex(np.conjugate(self.amplitude(data)))

    def transpose_sources(self, simulation: Simulation, data: RunResult):
        """Point-current cloud equal to the transpose of the modal readout.

        Returns ``(sources, coefficient)``. The first electric dipole has
        positive unit amplitude, fixing the engine's DFT normalization.
        """
        if not isinstance(simulation.grid, UniformMesh):
            raise ValueError("ModePower adjoint requires a uniform mesh")
        dl = simulation.grid.dl_um
        t1, t2 = {"x": ("y", "z"), "y": ("z", "x"),
                  "z": ("x", "y")}[self.axis]
        e1, e2, h1, h2 = (f"E{t1}", f"E{t2}", f"H{t1}", f"H{t2}")
        plane = data[self.name]
        arrays = {c: plane.sel(component=c) for c in (e1, e2, h1, h2)}
        _, c1, c2 = _plane_component(arrays, e1, self.freq_hz, t1, t2)
        w1 = _widths_with_grid(c1, self.mode, t1, t1, t2)
        w2 = _widths_with_grid(c2, self.mode, t2, t1, t2)
        area = np.outer(w2, w1)
        cen = self.center_um or (float(np.mean(c1)), float(np.mean(c2)))
        mode_fields = vector_modal_fields if hasattr(self.mode, "hx") else modal_fields
        mode = mode_fields(self.mode, c1, c2, axis=self.axis,
                           direction=self.direction, center_um=cen,
                           thickness_axis=self.thickness_axis)
        me1, me2, mh1, mh2 = (mode[k] for k in ("e1", "e2", "h1", "h2"))
        power = 0.5 * np.sum(np.real(me1 * np.conj(mh2)
                                    - me2 * np.conj(mh1)) * area)
        if power == 0:
            raise ValueError("mode has zero power on the objective plane")
        q = {e1: 0.25 * np.conj(mh2) * area / power,
             e2: -0.25 * np.conj(mh1) * area / power,
             h1: -0.25 * np.conj(me2) * area / power,
             h2: 0.25 * np.conj(me1) * area / power}

        # Transpose of node[j] = (raw[j-1] + raw[j])/2.
        if not getattr(self.mode, "yee_staggered", False):
            for comp, arr in q.items():
                axis = 1 if comp in (e1, h2) else 0
                half = 0.5 * arr
                shifted = np.roll(half, -1, axis=axis)
                edge = [slice(None)] * 2
                edge[axis] = -1
                shifted[tuple(edge)] = 0
                q[comp] = half + shifted

        electric = (e1, e2)
        anchor = max(electric, key=lambda c: np.max(np.abs(q[c])))
        anchor_pos = np.unravel_index(np.argmax(np.abs(q[anchor])), q[anchor].shape)
        scale = abs(q[anchor][anchor_pos])
        if scale == 0:
            raise ValueError("mode has no electric overlap weights")
        rotate = np.exp(-1j * np.angle(q[anchor][anchor_pos]))
        offsets = {"Ex": (0.5, 0, 0), "Ey": (0, 0.5, 0),
                   "Ez": (0, 0, 0.5), "Hx": (0, 0.5, 0.5),
                   "Hy": (0.5, 0, 0.5), "Hz": (0.5, 0.5, 0)}
        omega_dt = 2.0 * np.pi * self.freq_hz * _time_step(simulation)
        pulse = simulation.sources[0].source_time
        sources = []
        order = (anchor, *(c for c in (e1, e2, h1, h2) if c != anchor))
        for comp in order:
            values = q[comp] * rotate / scale
            indices = [(int(anchor_pos[0]), int(anchor_pos[1]))] if comp == anchor else []
            indices.extend((j, i) for j in range(values.shape[0])
                           for i in range(values.shape[1])
                           if comp != anchor or (j, i) != anchor_pos)
            for j, i in indices:
                value = complex(values[j, i])
                if value == 0:
                    continue
                if comp.startswith("H"):
                    # Magnetic injection is half a step late (§5).
                    value *= -np.exp(0.5j * omega_dt)
                loc = {a: float(plane.coords[a].values[0])
                       for a in "xyz"}
                loc[t1] = float(c1[i])
                loc[t2] = float(c2[j])
                point = tuple(loc[a] + offsets[comp][k] * dl
                              for k, a in enumerate("xyz"))
                sources.append(PointDipole(
                    center_um=point, polarization=comp,
                    amplitude=abs(value),
                    source_time=pulse.model_copy(
                        update={"phase": -float(np.angle(value))})))
        # The unit-amplitude, zero-phase anchor is emitted first for §12.
        coefficient = complex(np.conjugate(self.amplitude(data) * rotate) * scale)
        return tuple(sources), coefficient


# ---------------------------------------------------------------------------
# Gradient assembly
# ---------------------------------------------------------------------------

@dataclass
class GradientResult:
    """Objective and gradient for the subpixel-off model.

    ``shutoff`` is shared by the forward and adjoint solves. Their actual
    lengths are ``forward_steps_run`` and ``adjoint_steps_run``; they can differ
    when the caller's shutoff ends either solve early.
    """
    value: float                 # figure of merit J
    grad: np.ndarray             # dJ/drho, shape DesignRegion.shape (flat order)
    forward: RunResult
    adjoint: RunResult
    amplitude: complex           # forward objective phasor u
    forward_steps_run: Optional[int]
    adjoint_steps_run: Optional[int]
    shutoff: float


def _region_field(data: RunResult, region: DesignRegion, freq_hz: float,
                  name: str) -> np.ndarray:
    """Complex (3, nz, ny, nx) array of (Ex,Ey,Ez) over the design monitor."""
    da = data[name].sel(f=freq_hz)
    comps = [np.asarray(da.sel(component=c).values) for c in ("Ex", "Ey", "Ez")]
    return np.stack(comps, axis=0)


def assemble_gradient(
    region: DesignRegion,
    forward: RunResult,
    adjoint: RunResult,
    coeff: complex,
    freq_hz: float,
    *,
    monitor_name: str = "design_region",
    beta: complex = BETA,
) -> np.ndarray:
    """Contract the field product with exact trilinear control weights.

    ``beta`` is the derived physical coefficient supplied by the driver.
    The default is retained solely for old direct assembly callers.
    """
    da = forward[monitor_name].sel(f=freq_hz)
    e_fwd = _region_field(forward, region, freq_hz, monitor_name)
    e_adj = _region_field(adjoint, region, freq_hz, monitor_name)
    pix_prod = np.zeros(region.n_params, dtype=complex)
    for c, comp in enumerate(("Ex", "Ey", "Ez")):
        product = e_fwd[c] * e_adj[c]
        if region.parametrization == "boxes":
            owner = region._box_pixel_index(da, comp).ravel()
            good = owner >= 0
            np.add.at(pix_prod, owner[good], product.ravel()[good])
        else:
            weights = region._density_weights(da, comp)
            pix_prod += np.sum(weights * product[None, ...],
                               axis=(1, 2, 3))
    g = np.real(beta * coeff * pix_prod)
    g *= (region.eps_max - region.eps_min)                # chain rule rho->eps
    return g


def _time_step(simulation: Simulation) -> float:
    """The resolved uniform-grid Courant time step in seconds."""
    if not isinstance(simulation.grid, UniformMesh):
        raise ValueError("inverse-design gradient requires a uniform mesh")
    dl_m = simulation.grid.dl_um * 1e-6
    active = sum(int(realized_cells(length, simulation.grid.dl_um, floor) > 1)
                 for length, floor in zip(simulation.size_um,
                                          simulation._axis_min_cells()))
    return float(simulation.run.courant * dl_m /
                 (_C0 * np.sqrt(max(active, 1))))


def _discrete_omega(simulation: Simulation, freq_hz: float) -> float:
    """The leapfrog frequency corresponding to a DFT frequency."""
    dt = _time_step(simulation)
    return float(2.0 * np.sin(np.pi * freq_hz * dt) / dt)


def _validate_pml_clear(simulation: Simulation, region: DesignRegion,
                        objective: Objective) -> None:
    """Reject E design nodes and modal adjoint currents in CPML slabs."""
    dl = simulation.grid.dl_um
    n_cells = [realized_cells(length, dl, floor)
               for length, floor in zip(simulation.size_um,
                                        simulation._axis_min_cells())]
    face = region.box_face_offset_cells
    e_offsets = ((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5))
    for a, name in enumerate("xyz"):
        if getattr(simulation.boundaries, name) != "pml":
            continue
        left, right = region.cells[a]
        for offsets in e_offsets:
            off = offsets[a]
            first = np.ceil(left + face - off) + off
            last = np.floor(right + face - off) + off
            if first < simulation.pml_num_layers or last > n_cells[a] - simulation.pml_num_layers:
                raise ValueError(f"design E node lies inside the {name} PML slab")
    if not isinstance(objective, ModePower):
        return
    monitors = [m for m in simulation.monitors if m.name == objective.name]
    if not monitors:
        return
    monitor = monitors[0]
    for a, name in enumerate("xyz"):
        if getattr(simulation.boundaries, name) != "pml":
            continue
        low = (monitor.center_um[a] - 0.5 * monitor.size_um[a]) / dl
        high = (monitor.center_um[a] + 0.5 * monitor.size_um[a]) / dl
        # The source cloud adds up to half a cell of Yee displacement.
        if (low < simulation.pml_num_layers + 0.5 or
                high >= n_cells[a] - simulation.pml_num_layers - 0.5):
            raise ValueError(f"ModePower adjoint cloud would enter the {name} PML slab")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

BuildForward = Callable[[np.ndarray], Simulation]
"""User callback: ``rho -> Simulation``. Must place the domain/source and
include BOTH ``region.monitor(freq)`` and ``objective.monitor()`` plus
``region.structures(rho)``."""


Objective = Union[PointIntensity, ModePower]
"""A single-frequency objective with a transpose-matched adjoint source."""


def value_and_gradient(
    build_forward: BuildForward,
    region: DesignRegion,
    objective: Objective,
    rho: np.ndarray,
    *,
    device: Optional[str] = None,
    run_kwargs: Optional[dict] = None,
    beta: Optional[complex] = None,
    monitor_name: str = "design_region",
    _warn_beta: bool = True,
) -> GradientResult:
    """Return the objective and physical ``dJ/drho`` from two solves.

    The returned value and gradient use the caller's simulation with subpixel
    smoothing off, with a warning when the caller's simulation enables it.
    The adjoint run injects the transpose of the objective readout and records
    the E field over the design. ``beta`` is a compatibility override; the
    default is derived from the leapfrog frequency and vacuum permittivity.
    Supplying ``beta`` is deprecated. Both solves use the caller's run settings,
    including the default stop rule. The gradient is exact for the recorded
    window; its error relative to a converged gradient follows the truncation
    of the forward results. With the energy-only stop (``shutoff`` 1e-5) that
    was about 0.2% on a measured point-source scene, and more on resonant
    designs. On a local CPU solver the default stop also waits for an estimate
    of the remaining change in the recorded frequency-domain results
    (``RunSpec.dft_shutoff``) when the solver supports it; the estimate is not
    a guarantee, and GPU runs and older solvers stop on the field energy
    alone. For resonant designs and validation, set ``shutoff=0`` with an
    explicit ``n_steps`` or ``run_time_s`` after checking convergence. A
    simulation fitted with ``domain=`` is refused; give ``size_um`` and
    ``grid``.
    Quarter-cell pixel boxes support CPU and GPU; the optional trilinear
    custom medium supports CPU only.
    """
    run_kwargs = dict(run_kwargs or {})
    run_kwargs.setdefault("quiet", True)
    if device is not None:
        run_kwargs["device"] = device
    if region.parametrization == "trilinear" and str(
            run_kwargs.get("device", "cpu")).startswith("gpu"):
        raise ValueError(
            "trilinear custom-medium parametrization is CPU only; "
            "use parametrization='boxes' for GPU gradients")
    fwd_sim = build_forward(np.asarray(rho, dtype=float))
    if any(fwd_sim.symmetry):
        raise ValueError("symmetric gradients are not validated")
    if any(fwd_sim.origin_um):
        # A domain= fit stores positions in a shifted frame; the region, the
        # objective and its adjoint sources are in the caller's frame.
        raise ValueError(
            "inverse-design gradients need a simulation sized by hand "
            "(size_um and grid); a domain= fit shifts the stored frame")
    if not isinstance(fwd_sim.grid, UniformMesh):
        raise ValueError("inverse-design gradient requires a uniform mesh")
    if region.dl_um != fwd_sim.grid.dl_um:
        raise ValueError("DesignRegion.dl_um must equal the simulation grid dl_um")
    if region.parametrization == "trilinear":
        expected_shape = tuple(n + 1 for n in region.shape)
        expected = region.structures(rho)[0]
        if not fwd_sim.structures or (
                fwd_sim.structures[-1].geometry != expected.geometry or
                fwd_sim.structures[-1].medium.permittivity_data is None or
                fwd_sim.structures[-1].medium.permittivity_data.shape != expected_shape):
            raise ValueError(
                "build_forward must place region.structures(rho) last so the "
                "trilinear density map matches the adjoint derivative")
    elif (len(fwd_sim.structures) < region.n_params or
          any(s.geometry != expected.geometry or
              s.medium.permittivity_data is not None or
              s.medium.permittivity != expected.medium.permittivity
              for s, expected in zip(fwd_sim.structures[-region.n_params:],
                                     region.box_structures(rho)))):
        raise ValueError(
            "build_forward must place region.structures(rho) last so the "
            "pixel boxes match the adjoint derivative")
    _validate_pml_clear(fwd_sim, region, objective)
    if fwd_sim.subpixel:
        warnings.warn("inverse-design objective and gradient use the subpixel-off model",
                      UserWarning, stacklevel=caller_stacklevel())
    if beta is not None and _warn_beta:
        warnings.warn("beta= is deprecated; omit it to use the derived discrete adjoint coefficient",
                      DeprecationWarning, stacklevel=caller_stacklevel())
    shutoff = fwd_sim.run.shutoff
    # Both box indicators and custom-medium weights assume smoothing is off.
    fwd_sim = fwd_sim.model_copy(update={"subpixel": False})
    def solve_pair(output_root: Path) -> tuple[RunResult, RunResult, complex, np.ndarray, float]:
        forward = run_local(fwd_sim, **{**run_kwargs,
                                       "output_dir": output_root / "forward"})
        if forward.aborted:
            raise RuntimeError(f"forward run aborted: {forward.abort_reason}")

        u = objective.amplitude(forward)
        if isinstance(objective, ModePower):
            adj_sources, coeff = objective.transpose_sources(fwd_sim, forward)
        else:
            adj_sources = (objective.adjoint_source(fwd_sim),)
            coeff = objective.adjoint_coeff(forward)
        derived_beta = beta
        if derived_beta is None:
            derived_beta = (-2j * _discrete_omega(fwd_sim, objective.freq_hz)
                            * _EPS0)
            if isinstance(objective, PointIntensity) and objective.component.startswith("H"):
                derived_beta *= -np.exp(1j * np.pi * objective.freq_hz * _time_step(fwd_sim))

        adj_sim = fwd_sim.model_copy(update={
            "sources": adj_sources,
            "monitors": (region.monitor(objective.freq_hz, monitor_name),),
        })
        adjoint = run_local(adj_sim, **{**run_kwargs,
                                       "output_dir": output_root / "adjoint"})
        if adjoint.aborted:
            raise RuntimeError(f"adjoint run aborted: {adjoint.abort_reason}")

        g = assemble_gradient(region, forward, adjoint, coeff, objective.freq_hz,
                              monitor_name=monitor_name, beta=derived_beta)
        value = objective.value(forward)
        # RunResult reads monitors lazily. Cache them before removing temporary
        # solver output, so the returned results remain usable.
        for result in (forward, adjoint):
            for name in result.monitor_names:
                result._raw(name)
        return forward, adjoint, u, g, value

    output_dir = run_kwargs.pop("output_dir", None)
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="photonhub-gradient-") as directory:
            forward, adjoint, u, g, value = solve_pair(Path(directory))
    else:
        forward, adjoint, u, g, value = solve_pair(Path(output_dir))
    return GradientResult(value=value, grad=g,
                          forward=forward, adjoint=adjoint, amplitude=u,
                          forward_steps_run=forward.steps_run,
                          adjoint_steps_run=adjoint.steps_run,
                          shutoff=shutoff)


@dataclass
class OptimizeResult:
    """Optimization history and best value for the subpixel-off model.

    ``history``, ``grads``, and ``best`` all refer to that same model even if
    the caller's constructed simulation enabled subpixel smoothing.
    """
    rho: np.ndarray              # final density grid (region.shape)
    history: List[float]         # objective per iteration
    grads: List[np.ndarray]
    best: GradientResult


def _descend(method, fg, x0, bounds, n_iters, step, maximize):
    """Drive the chosen optimizer on an oracle ``fg(x) -> (J, dJ/dx)`` that does
    its own bookkeeping (history / best / callback). MAXIMIZES ``J`` if
    ``maximize`` else minimizes; ``bounds`` is ``(lo, hi)`` applied to every
    variable, or None.

    Default ``method="lbfgs"`` is SciPy L-BFGS-B, a quasi-Newton method that
    builds curvature from the gradient history and line-searches each step, the
    standard choice for adjoint inverse design. Unlike Adam, L-BFGS USES the
    gradient magnitude. The older one-probe calibration remains for optimizer
    compatibility while the adjoint gradient now has physical scale. It probes
    ``J`` along the first gradient and hands SciPy a ``J(x0)``-normalized pair.
    ``method="adam"`` keeps the scale-free normalized-gradient Adam (``step`` =
    max per-variable change per iteration); only Adam uses ``step``."""
    sign = 1.0 if maximize else -1.0
    x0 = np.asarray(x0, dtype=float).ravel().copy()
    if method == "lbfgs":
        from scipy.optimize import minimize

        bnds = None if bounds is None else [(bounds[0], bounds[1])] * x0.size

        # --- one-time gradient-scale calibration (see docstring) ---
        J0, g0 = fg(x0)
        absJ0 = abs(float(J0)) or 1.0
        cal = 1.0                                     # grad_J = cal * g_adjoint
        gn = float(np.linalg.norm(g0))
        if gn > 0 and np.isfinite(J0):
            xp = x0 + (1e-3 / gn) * g0                # tiny step along the gradient
            if bounds is not None:
                xp = np.clip(xp, bounds[0], bounds[1])
            d = xp - x0
            hh = float(np.linalg.norm(d))
            if hh > 0:
                Jp, _ = fg(xp)
                dderiv_adj = float(g0 @ (d / hh))     # adjoint directional derivative
                if dderiv_adj != 0 and np.isfinite(Jp):
                    cal = ((Jp - J0) / hh) / dderiv_adj
        if not np.isfinite(cal) or cal <= 0.0:
            # A non-positive cal claims the adjoint gradient points downhill
            # along itself, which usually means the FD probe was dominated by
            # solver jitter. Freezing a
            # flipped/garbage scale would corrupt every subsequent step. The
            # calibration is an efficiency aid, not correctness-critical, so
            # fall back to the raw adjoint gradient scale.
            warnings.warn(
                f"gradient-scale calibration probe returned cal={cal:.3g} "
                "(noise-dominated FD probe?); falling back to cal=1.0",
                stacklevel=caller_stacklevel())
            cal = 1.0
        gscale = cal / absJ0                          # makes (f, grad) consistent + O(1)

        def neg(x):
            J, gJ = fg(x)
            return float(-sign * J / absJ0), (-sign * gscale * np.asarray(gJ, float))

        minimize(neg, x0, jac=True, method="L-BFGS-B", bounds=bnds,
                 options=dict(maxiter=int(n_iters), maxfun=int(2 * n_iters + 10),
                              ftol=1e-9, gtol=1e-12, maxls=20))
    elif method == "adam":
        x = x0
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for it in range(1, n_iters + 1):
            _, gJ = fg(x)
            g = sign * np.asarray(gJ, dtype=float)
            gmax = np.max(np.abs(g))
            if gmax > 0:                              # scale-free: direction only
                g = g / gmax
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            x = x + step * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + eps)
            if bounds is not None:
                x = np.clip(x, bounds[0], bounds[1])
    else:
        raise ValueError(f"method must be 'lbfgs' or 'adam', got {method!r}")


def optimize(
    build_forward: BuildForward,
    region: DesignRegion,
    objective: Objective,
    rho0: np.ndarray,
    *,
    n_iters: int = 30,
    method: str = "adam",
    step: float = 0.05,
    bounds: Tuple[float, float] = (0.0, 1.0),
    maximize: bool = True,
    device: Optional[str] = None,
    run_kwargs: Optional[dict] = None,
    beta: Optional[complex] = None,
    param_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    callback: Optional[Callable[[int, GradientResult, np.ndarray], None]] = None,
) -> OptimizeResult:
    """Topology optimization driven by the adjoint gradient.

    ``method`` is ``"adam"`` (default, normalized-gradient Adam) or ``"lbfgs"``
    (SciPy L-BFGS-B); see :func:`_descend`. Adam is the default HERE because for
    high-dimensional topology a scale-free update can be convenient. For a FEW smooth shape
    parameters, prefer :func:`optimize_parametric`, which defaults to L-BFGS-B.
    ``n_iters`` bounds the optimizer iterations. Densities are box-constrained to
    ``bounds``. Maximizes ``objective`` by default; returns the BEST design seen.
    ``step`` is used only by Adam.

    ``device`` (``"cpu"`` / ``"gpu"`` / ``"gpu:N"``) runs every iteration's two
    solves on that backend (forwarded to ``run_local``).

    ``param_map`` (a ``(region.shape) -> (region.shape)`` array map) constrains the
    design: the structure is built from ``param_map(rho)`` and the gradient is
    mapped through it, a proper projected gradient. For a LINEAR, self-adjoint
    projection (e.g. a symmetry-averaging map, which is its own adjoint) this is
    exact; ``best`` and ``rho`` are reported in the mapped (constrained) space.
    Use it to enforce device symmetry or a density filter. Returned values and
    gradients use the subpixel-off model. ``beta=`` is deprecated. Both solves
    retain the caller's run settings and are exact for their recorded window;
    the error against a converged gradient follows the truncation of the
    forward results (see :func:`value_and_gradient`). For resonant designs and
    validation, set ``shutoff=0`` with an explicit ``n_steps`` or
    ``run_time_s`` after checking convergence."""
    if beta is not None:
        warnings.warn("beta= is deprecated; omit it to use the derived discrete adjoint coefficient",
                      DeprecationWarning, stacklevel=caller_stacklevel())
    pm = param_map if param_map is not None else (lambda r: r)
    shape = region.shape
    sign = 1.0 if maximize else -1.0
    lo, hi = bounds

    history: List[float] = []
    grads: List[np.ndarray] = []
    state = {"best": None, "x": np.asarray(rho0, float).reshape(shape).ravel().copy()}

    def fg(x):
        eff = pm(x.reshape(shape)).ravel()            # constrained design
        res = value_and_gradient(build_forward, region, objective, eff,
                                 device=device, run_kwargs=run_kwargs, beta=beta,
                                 _warn_beta=False)
        history.append(res.value)
        grads.append(res.grad)
        if state["best"] is None or (sign * res.value > sign * state["best"].value):
            state["best"] = res
            state["x"] = np.asarray(x, float).copy()  # return the BEST, not the last
        if callback is not None:
            callback(len(history), res, eff.reshape(shape))
        g = pm(res.grad.reshape(shape)).ravel()       # project gradient (self-adjoint pm)
        return res.value, g

    _descend(method, fg, state["x"], (lo, hi), n_iters, step, maximize)
    best = state["best"]
    assert best is not None
    return OptimizeResult(rho=pm(state["x"].reshape(shape)), history=history,
                          grads=grads, best=best)


# ---------------------------------------------------------------------------
# Parameter (shape) optimization: a handful of geometric design variables
# ---------------------------------------------------------------------------

@dataclass
class ParametricResult:
    params: np.ndarray               # optimized parameter vector
    rho: np.ndarray                  # final density grid, expand(params)
    history: List[float]             # objective per iteration
    params_history: List[np.ndarray]
    best: GradientResult


def optimize_parametric(
    build_forward: BuildForward,
    region: DesignRegion,
    objective: Objective,
    p0: np.ndarray,
    expand: Callable[[np.ndarray], np.ndarray],
    *,
    n_iters: int = 30,
    method: str = "lbfgs",
    step: float = 0.02,
    bounds: Optional[Tuple[float, float]] = None,
    maximize: bool = True,
    device: Optional[str] = None,
    run_kwargs: Optional[dict] = None,
    beta: Optional[complex] = None,
    fd_step: float = 1e-3,
    callback: Optional[Callable[[int, GradientResult, np.ndarray], None]] = None,
) -> ParametricResult:
    """Adjoint PARAMETER (shape) optimization: optimize a handful of geometric
    design variables ``p`` instead of a free per-pixel density (topology).

    ``expand(p) -> density grid`` (``region.shape``, values in [0, 1]) is the
    differentiable parameterization (e.g. a taper's control-point widths -> a
    rendered waveguide). Each evaluation runs ONE forward + ONE adjoint solve to
    get ``dJ/drho`` over the pixels, then chains it to the parameters through the
    parameterization Jacobian ``drho/dp``, which is a **cheap central
    finite-difference of the analytic** ``expand`` (no extra FDTD solves):

        dJ/dp_j = sum_i (dJ/drho_i) (drho_i/dp_j),
        drho/dp_j  ≈  (expand(p + h e_j) - expand(p - h e_j)) / 2h   (h = fd_step)

    So the cost is the SAME two solves per evaluation as topology optimization,
    regardless of the pixel count, while optimizing only ``len(p)`` variables.
    Make ``expand`` smooth (a soft/graded boundary over ~1 cell) so ``drho/dp`` is
    well-defined. ``method`` is ``"lbfgs"`` (default, SciPy L-BFGS-B, well suited
    to a few smooth parameters) or ``"adam"``; see :func:`_descend`. ``bounds``
    (lo, hi) box-constrains every parameter. Returns the BEST parameters seen.
    Maximizes ``objective`` by default. Returned values and gradients use the
    subpixel-off model. ``beta=`` is deprecated. Both solves retain the
    caller's run settings and are exact for their recorded window; the error
    against a converged gradient follows the truncation of the forward results
    (see :func:`value_and_gradient`). For resonant designs and validation, set
    ``shutoff=0`` with an explicit ``n_steps`` or ``run_time_s`` after checking
    convergence."""
    if beta is not None:
        warnings.warn("beta= is deprecated; omit it to use the derived discrete adjoint coefficient",
                      DeprecationWarning, stacklevel=caller_stacklevel())
    sign = 1.0 if maximize else -1.0
    history: List[float] = []
    params_history: List[np.ndarray] = []
    state = {"best": None, "p": np.asarray(p0, float).copy()}

    def fg(p):
        p = np.asarray(p, dtype=float)
        rho = np.asarray(expand(p), dtype=float)
        res = value_and_gradient(build_forward, region, objective, rho,
                                 device=device, run_kwargs=run_kwargs, beta=beta,
                                 _warn_beta=False)
        g_rho = res.grad                              # dJ/drho per pixel (flat)
        history.append(res.value)
        params_history.append(p.copy())
        if state["best"] is None or (sign * res.value > sign * state["best"].value):
            state["best"] = res
            state["p"] = p.copy()                     # return the BEST, not the last
        if callback is not None:
            callback(len(history), res, p.copy())

        # parameterization Jacobian via cheap central FD of expand (no solves)
        dJdp = np.zeros_like(p)
        for j in range(p.size):
            pp = p.copy(); pp[j] += fd_step
            pm_ = p.copy(); pm_[j] -= fd_step
            drho = (np.asarray(expand(pp), float).ravel()
                    - np.asarray(expand(pm_), float).ravel()) / (2.0 * fd_step)
            dJdp[j] = float(g_rho @ drho)
        return res.value, dJdp

    _descend(method, fg, np.asarray(p0, float), bounds, n_iters, step, maximize)
    best = state["best"]
    assert best is not None
    return ParametricResult(params=state["p"],
                            rho=np.asarray(expand(state["p"]), float),
                            history=history, params_history=params_history,
                            best=best)
