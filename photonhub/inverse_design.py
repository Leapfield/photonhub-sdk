"""Single-frequency discrete-Yee adjoint gradients for inverse design.

The default design uses scalar pixel boxes; trilinear custom media are optional.
Each E-component sample of permittivity has a density derivative. The frequency
derivative uses the leapfrog frequency `(2/dt)*sin(pi*f*dt)` and the
engine's source-normalized DFT fields.
"""

from __future__ import annotations

import warnings
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse as _sparse

from .components import (
    Box,
    ProfileMonitor,
    Medium,
    ModeSource,
    PointDipole,
    Simulation,
    Structure,
)
from .data import RunResult
from .components.base import MAX_INT32
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
from .analysis.smatrix import SPort
from ._compat import caller_stacklevel
from .polygon_profile import (  # noqa: F401 - re-exported shape tools
    PolygonProfile, curvature_penalty, curvature_radii,
)

# Deprecated legacy import names. Gradients derive their coefficients from
# each simulation; these sentinel values are never used by the driver.
BETA: complex = 1.0 + 0.0j
BETA_MODE: complex = 1.0 + 0.0j


def _adjoint_pulse(simulation: Simulation):
    """The forward run's first pulse, as the plain pulse adjoint sources reuse.

    A dipole of a broadband launch built with ``modes_by_freq`` carries a
    windowed carrier (``band_freqs_hz`` and ``carrier_index``) that drives
    only a slice of the pulse's spectrum, while the engine normalizes every
    frequency-domain result by the full pulse. An adjoint current copied
    from it would carry almost nothing away from its carrier's frequency, so
    the window is dropped; a plain pulse is returned unchanged."""
    pulse = simulation.sources[0].source_time
    if getattr(pulse, "band_freqs_hz", None) is not None:
        pulse = pulse.model_copy(update={"band_freqs_hz": None, "carrier_index": None})
    return pulse


def _warn_objective_beta(beta: complex, placeholder: complex) -> None:
    if beta != placeholder:
        warnings.warn(
            "the objective's beta field is deprecated and ignored; the gradient "
            "derives its coefficient from the simulation",
            DeprecationWarning, stacklevel=caller_stacklevel())


def _axis_box(c0: int, c1: int, dl: float, quarter: bool) -> Tuple[float, float]:
    """(center, size) microns for a box spanning integer cells ``[c0, c1)``.

    ``quarter=False`` puts faces on ``k*dl``. Yee nodes on shared faces
    require the engine's inclusive, last-wins ownership rule.
    ``quarter=True`` shifts the faces to ``(k+0.25)*dl`` so a *multi-component*
    DFT monitor snaps Ex/Ey/Ez to the same cells."""
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

    ``merge_boxes=True`` emits the same pixel model with far fewer structures:
    one box at ``eps(0)`` over the whole region, then one box per run of
    pixels with bitwise equal density along the region's longest axis
    (adjacent equal runs merged), and none for a density-0 pixel. With
    quarter-cell faces no Yee sample lies on any face, so the engine samples
    exactly the permittivity of the one-box-per-pixel layout at every point,
    and the gradient is unchanged. The engine scans every structure at every
    sample, so its setup time grows with the number of structures: on a
    quasi-2D CPU test with 259200 pixel boxes (720 x 360, 0.025 um cells) the
    one-box-per-pixel layout took 483 s to set up. A shape rendering that
    is 0 or 1 over most of the region needs only a few thousand merged
    boxes. Requires ``parametrization="boxes"`` and
    ``box_face_offset_cells=0.25``.

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
    merge_boxes: bool = False
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
        if self.merge_boxes and (self.parametrization != "boxes"
                                 or self.box_face_offset_cells != 0.25):
            raise ValueError("merge_boxes requires parametrization='boxes' and "
                             "box_face_offset_cells=0.25")

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
        merge_boxes: bool = False,
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
                   box_face_offset_cells=box_face_offset_cells,
                   merge_boxes=merge_boxes)

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
        """Pixel boxes with quarter-cell faces unless integer faces are selected.

        With ``merge_boxes`` the merged layout of :meth:`merged_box_structures`."""
        if self.merge_boxes:
            return self.merged_box_structures(rho)
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

    def _pixel_range_box(self, ranges) -> Box:
        """Quarter-cell-faced box over pixel ranges ``[(lo, hi)] * 3``."""
        center, size = [], []
        for a, (p0, p1) in enumerate(ranges):
            per = self.cells_per_pixel[a]
            c, s = _axis_box(self.cells[a][0] + p0 * per, self.cells[a][0] + p1 * per,
                             self.dl_um, quarter=True)
            center.append(c)
            size.append(s)
        return Box(center_um=tuple(center), size_um=tuple(size))

    def merged_box_structures(self, rho: np.ndarray) -> Tuple[Structure, ...]:
        """The ``merge_boxes`` layout: a box at ``eps(0)`` over the region, then
        one box per run of bitwise equal densities along the longest axis,
        equal runs on adjacent lines of the next longest axis merged, and
        nothing for a pixel whose permittivity equals ``eps(0)``.

        Every pixel's permittivity is the same float as in the
        one-box-per-pixel layout, and with quarter-cell faces every Yee
        sample lies inside exactly one of the merged boxes or only in the
        first, so the engine samples the two layouts identically."""
        if self.box_face_offset_cells != 0.25:
            raise ValueError("merged boxes require box_face_offset_cells=0.25")
        rho = np.asarray(rho, dtype=float).reshape(self.shape)
        eps = self.eps(rho)
        background = self.eps(np.zeros(1))[0]
        whole = tuple((0, n) for n in self.shape)
        out: List[Structure] = [Structure(geometry=self._pixel_range_box(whole),
                                          medium=Medium(permittivity=float(background)))]
        run = int(np.argmax(self.shape))
        second, third = sorted((a for a in range(3) if a != run),
                               key=lambda a: -self.shape[a])
        lines = np.transpose(eps, (third, second, run)).reshape(-1, self.shape[run])
        change = np.ones(lines.shape, dtype=bool)
        change[:, 1:] = lines[:, 1:] != lines[:, :-1]
        line, start = np.nonzero(change)
        if line.size == 0:
            return tuple(out)
        stop = np.full(start.shape, self.shape[run])
        same = line[1:] == line[:-1]
        stop[:-1] = np.where(same, start[1:], self.shape[run])
        value = lines[line, start]
        keep = value != background
        line, start, stop, value = line[keep], start[keep], stop[keep], value[keep]
        k3, k2 = np.divmod(line, self.shape[second])
        order = np.lexsort((k2, value, stop, start, k3))
        k3, k2, start, stop, value = (a[order] for a in (k3, k2, start, stop, value))
        fresh = np.ones(k2.shape, dtype=bool)
        fresh[1:] = ((k3[1:] != k3[:-1]) | (start[1:] != start[:-1]) | (stop[1:] != stop[:-1])
                     | (value[1:] != value[:-1]) | (k2[1:] != k2[:-1] + 1))
        heads = np.flatnonzero(fresh)
        tails = np.append(heads[1:], k2.size) - 1
        for h, t in zip(heads, tails):
            ranges = [None, None, None]
            ranges[run] = (int(start[h]), int(stop[h]))
            ranges[second] = (int(k2[h]), int(k2[t]) + 1)
            ranges[third] = (int(k3[h]), int(k3[h]) + 1)
            out.append(Structure(geometry=self._pixel_range_box(ranges),
                                 medium=Medium(permittivity=float(value[h]))))
        return tuple(out)

    def monitor(self, freq_hz, name: str = "design_region", *,
                simulation: Optional[Simulation] = None) -> ProfileMonitor:
        """The DFT monitor recording all three E-components over the region
        (quarter-cell faces so the components co-snap).

        ``freq_hz`` is one frequency, or a sequence or array of frequencies
        for a :class:`BroadbandModeTransmission` objective, which records the
        region at every one of its ``freqs_hz`` in the same run.

        Pass ``simulation`` (any simulation with the run's size, grid and
        boundaries) for a quasi-2D run: on an axis that the simulation
        realizes as one plain periodic cell the monitor becomes the plane at
        a quarter cell, which the engine snaps to that cell for every
        component, instead of a box reaching outside the one-cell domain.
        Other axes are unchanged. :func:`value_and_gradient` passes the
        forward simulation for the adjoint run."""
        (i0, i1), (j0, j1), (k0, k1) = self.cells
        cx, sx = _axis_box(i0, i1, self.dl_um, quarter=True)
        cy, sy = _axis_box(j0, j1, self.dl_um, quarter=True)
        cz, sz = _axis_box(k0, k1, self.dl_um, quarter=True)
        center, size = [cx, cy, cz], [sx, sy, sz]
        if simulation is not None and isinstance(simulation.grid, UniformMesh):
            mins = simulation._axis_min_cells()
            for a in range(3):
                if mins[a] == 1 and realized_cells(simulation.size_um[a], self.dl_um, 1) == 1:
                    center[a], size[a] = 0.25 * self.dl_um, 0.0
        if isinstance(freq_hz, np.ndarray) or hasattr(freq_hz, "__iter__"):
            values = freq_hz if isinstance(freq_hz, np.ndarray) else list(freq_hz)
            freqs = tuple(float(f) for f in np.ravel(np.asarray(values, dtype=float)))
        else:
            freqs = (freq_hz,)
        return ProfileMonitor(
            name=name, center_um=tuple(center), size_um=tuple(size),
            fields=("Ex", "Ey", "Ez"), freqs_hz=freqs)

    # -- cell -> pixel scatter map ------------------------------------------

    def _pixel_index(self, da) -> np.ndarray:
        """For each recorded cell in DFT DataArray ``da`` (dims ..z,y,x), the
        flat pixel index, or -1 if the cell lies outside the tiled region (an
        edge cell the snap recorded just past the high face). Row-major
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
# Soft-boundary shape rendering
# ---------------------------------------------------------------------------

def _pixel_centers_um(region: DesignRegion, axis: int) -> np.ndarray:
    """Centers of the pixel boxes along one axis, face offset included."""
    a0 = region.cells[axis][0]
    per = region.cells_per_pixel[axis]
    k = np.arange(region.shape[axis])
    return (a0 + region.box_face_offset_cells + (k + 0.5) * per) * region.dl_um


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """Quintic step ``6u^5 - 15u^4 + 10u^3`` on ``[0, 1]``, 0 below, 1 above."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)


def _smoothstep_slope(u: np.ndarray) -> np.ndarray:
    """Derivative of :func:`_smoothstep`, ``30u^2(1-u)^2`` inside, 0 outside."""
    inside = (u > 0.0) & (u < 1.0)
    return np.where(inside, 30.0 * u ** 2 * (1.0 - u) ** 2, 0.0)


@dataclass(frozen=True)
class WidthProfile:
    """A guide whose width varies along one axis, rendered with a soft edge.

    :meth:`density` and :meth:`jacobian` take ``widths_um``, where
    ``widths_um[j]`` is the full width at ``positions_um[j]`` along ``axis``.
    Between controls the width is interpolated linearly; before the first
    and after the last control it is held. The guide is centered at
    ``guide_center_um`` along ``width_axis`` and fills the region along the
    third axis. A pixel whose box center lies at ``s`` along ``axis`` and
    ``t`` along ``width_axis`` takes density ``S(u)`` with
    ``u = (w(s)/2 - |t - guide_center_um|)/edge_um + 1/2`` and ``S`` the
    quintic step ``6u^5 - 15u^4 + 10u^3`` on ``[0, 1]``. The edge is
    ``edge_um`` wide and centered on the nominal boundary; outside it the
    density is exactly 0 or 1. ``u`` is linear in the widths and ``S`` has
    continuous first and second derivatives, so the density is twice
    continuously differentiable in the widths (its third derivative jumps
    where ``u`` crosses 0 or 1), and :meth:`jacobian` returns its exact
    derivative for :func:`parametric_value_and_gradient` and
    :func:`optimize_parametric`.

    The edge sets how evenly a width acts on the density. Only pixel centers
    inside the edge band move, so ``d(sum rho)/dw`` follows ``S'`` of those
    centers. With an edge of one pixel pitch along ``width_axis`` exactly
    one center lies in the band: as the half-width moves across one pitch
    that sensitivity swings from 0 to 1.92 times its mean, and it is exactly
    zero whenever the nominal edge lies on a pixel face (the edge pixels sit
    at ``u = 0`` and 1). The width is then dead, and an optimizer meets
    plateaus and strong curvature between them. On a 3D CPU test (0.044 um
    uniform mesh, one cell per pixel, subpixel off, non-dispersive Si in
    SiO2, PML 6 on every axis, 3000 fixed steps, shutoff 0) a one-cell edge
    made ``dT/dt`` of a uniform guide thickness swing from -0.099 per um to
    exactly 0 over one cell of half-thickness; a two-cell edge kept it
    between -0.021 and -0.062. An edge of two pitches holds the sensitivity
    within 0.94 to 1.05 times its mean, and three within 0.99 to 1.01. An
    edge below the pitch can hold no pixel center at all: on a region of two
    cells per pixel a one-cell edge left the density exactly 0 or 1, and the
    gradient exactly zero, over 49% of a two-cell sweep of the half-width.
    The default edge is the larger of two cells and the pitch along
    ``width_axis``, which is two pitches on one cell per pixel. On coarser
    pixels it is one pitch, so pass ``edge_um`` of two pitches for an
    optimization there. A smaller ``edge_um`` than the pitch warns.

    A width whose edge reaches the region's first or last pixel along
    ``width_axis`` is clipped there, and its gradient is lost; :meth:`density`
    warns when either boundary pixel has nonzero density.

    Build the region with :meth:`on_grid` for one cell per pixel along
    ``axis`` and ``width_axis``, or pass any pixel-box region. Use
    :meth:`density` as ``expand`` and :meth:`jacobian` as ``jacobian``.
    Symmetric or composite shapes are built from these two methods with the
    chain rule.
    """

    region: DesignRegion
    axis: str
    width_axis: str
    positions_um: Tuple[float, ...]
    guide_center_um: float
    edge_um: Optional[float] = None

    def __post_init__(self) -> None:
        if (self.axis not in ("x", "y", "z") or self.width_axis not in ("x", "y", "z")
                or self.axis == self.width_axis):
            raise ValueError("axis and width_axis must be two different axes among x, y, z")
        if self.region.parametrization != "boxes":
            raise ValueError("WidthProfile renders pixel boxes; use parametrization='boxes'")
        positions = np.asarray(self.positions_um, dtype=float)
        if (positions.ndim != 1 or positions.size < 1 or not np.all(np.isfinite(positions))
                or np.any(np.diff(positions) <= 0)):
            raise ValueError("positions_um must be finite and strictly increasing")
        object.__setattr__(self, "positions_um", tuple(float(p) for p in positions))
        if not np.isfinite(self.guide_center_um):
            raise ValueError("guide_center_um must be finite")
        if self.edge_um is not None and not (np.isfinite(self.edge_um) and self.edge_um > 0):
            raise ValueError("edge_um must be positive and finite")
        pitch = self._pitch()
        if self.edge_um is not None and self.edge_um < pitch * (1.0 - 1e-9):
            warnings.warn(
                f"WidthProfile edge_um={self.edge_um:g} is below the pixel pitch "
                f"{pitch:g} um along {self.width_axis}: over part of the width range "
                "no pixel center lies in the edge band, so the density is exactly 0 "
                "or 1 there and the width's gradient is exactly zero. Use an edge of "
                "at least one pitch, or leave edge_um unset for the default",
                UserWarning, stacklevel=caller_stacklevel())

    @classmethod
    def on_grid(
        cls,
        *,
        center_um: Tuple[float, float, float],
        size_um: Tuple[float, float, float],
        dl_um: float,
        axis: str,
        width_axis: str,
        positions_um: Tuple[float, ...],
        guide_center_um: float,
        eps_min: float = 1.0,
        eps_max: float = 12.25,
        edge_um: Optional[float] = None,
    ) -> "WidthProfile":
        """Profile on a region of one cell per pixel along ``axis`` and
        ``width_axis`` and one pixel across the third axis.

        The region snaps as :meth:`DesignRegion.on_grid` does."""
        if (axis not in ("x", "y", "z") or width_axis not in ("x", "y", "z")
                or axis == width_axis):
            raise ValueError("axis and width_axis must be two different axes among x, y, z")
        planar = ("xyz".index(axis), "xyz".index(width_axis))
        shape = tuple(max(int(round(s / dl_um)), 1) if a in planar else 1
                      for a, s in enumerate(size_um))
        region = DesignRegion.on_grid(
            center_um=center_um, size_um=size_um, dl_um=dl_um, shape=shape,
            eps_min=eps_min, eps_max=eps_max)
        return cls(region=region, axis=axis, width_axis=width_axis,
                   positions_um=tuple(positions_um),
                   guide_center_um=guide_center_um, edge_um=edge_um)

    @property
    def num_widths(self) -> int:
        """Number of width controls."""
        return len(self.positions_um)

    def _pitch(self) -> float:
        """Pixel pitch along ``width_axis``, in microns."""
        return self.region.cells_per_pixel["xyz".index(self.width_axis)] * self.region.dl_um

    def _edge(self) -> float:
        if self.edge_um is not None:
            return float(self.edge_um)
        return max(2.0 * self.region.dl_um, self._pitch())

    def _sample(self, widths_um) -> Tuple[np.ndarray, np.ndarray]:
        widths = np.asarray(widths_um, dtype=float)
        if widths.shape != (self.num_widths,) or not np.all(np.isfinite(widths)):
            raise ValueError(f"widths_um must be {self.num_widths} finite values")
        s = _pixel_centers_um(self.region, "xyz".index(self.axis))
        t = _pixel_centers_um(self.region, "xyz".index(self.width_axis))
        basis = np.stack([np.interp(s, self.positions_um, e)
                          for e in np.eye(self.num_widths)])
        half = 0.5 * (widths @ basis)
        u = (half[:, None] - np.abs(t[None, :] - self.guide_center_um)) / self._edge() + 0.5
        return u, basis

    def _to_region(self, planar: np.ndarray) -> np.ndarray:
        """Spread an ``(n_axis, n_width, ...)`` array across the third axis
        and reorder it to the region's ``(x, y, z, ...)``."""
        a, b = "xyz".index(self.axis), "xyz".index(self.width_axis)
        c = 3 - a - b
        n_c = self.region.shape[c]
        spread = np.broadcast_to(planar[:, :, None],
                                 planar.shape[:2] + (n_c,) + planar.shape[2:])
        order = list(np.argsort([a, b, c])) + list(range(3, spread.ndim))
        return np.ascontiguousarray(np.transpose(spread, order))

    def density(self, widths_um) -> np.ndarray:
        """Density grid of shape ``region.shape`` for the given widths.

        Warns when the first or last pixel along ``width_axis`` has nonzero
        density: the guide's edge is then cut by the region boundary."""
        u, _ = self._sample(widths_um)
        planar = _smoothstep(u)
        if np.any(planar[:, [0, -1]] > 0):
            warnings.warn(
                f"WidthProfile: the guide reaches the region's boundary pixels along "
                f"{self.width_axis}, so its rendered width is clipped there and the "
                "clipped part has no gradient; enlarge the region along "
                f"{self.width_axis} or bound the widths",
                UserWarning, stacklevel=caller_stacklevel())
        return self._to_region(planar)

    def jacobian(self, widths_um) -> np.ndarray:
        """Exact ``d density / d widths``, shape ``(region.n_params, num_widths)``
        in flat ``density(...).ravel()`` order."""
        u, basis = self._sample(widths_um)
        slope = _smoothstep_slope(u) / (2.0 * self._edge())
        planar = slope[:, :, None] * basis.T[:, None, :]
        return self._to_region(planar).reshape(self.region.n_params, self.num_widths)


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
        """Unit point dipole at the probe, reusing the forward pulse so the normalization matches (the objective coefficient is applied in
        post-processing, not in the source amplitude)."""
        pulse = _adjoint_pulse(forward_sim)
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
    of the recorded plane (``mode_amplitude``). For port-to-port
    transmission, use :class:`ModeTransmission` with suitable port modes.

    .. warning:: ``J`` is a RELATIVE objective, not a calibrated power
       transmission. ``|c|^2 == T`` would require the launch to read
       ``c_in == 1`` on an input plane, but ``mode_source(power_watts=1)``
       normalizes to SI watts, so ``J`` carries a scene/grid-dependent
       positive scale (measured ``~4/dl_um`` on SOI strip scenes: J = 23.4
       at dl = 0.05 um where T ~ 0.29). Maximizing J still maximizes T;
       optimization and relative comparisons are unaffected, but do NOT
       report J as transmission; use the S-matrix path
       (:func:`photonhub.analysis.smatrix`), which normalizes by the driven
       port's incident amplitude. The accuracy of that reading depends on
       the reference mode and the placement of its input plane.

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
        pulse = _adjoint_pulse(forward_sim)
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
        pulse = _adjoint_pulse(simulation)
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
                    # Magnetic injection is half a step late.
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
        # The unit-amplitude, zero-phase anchor is emitted first: the engine
        # normalizes frequency-domain monitors by the first source.
        coefficient = complex(np.conjugate(self.amplitude(data) * rotate) * scale)
        return tuple(sources), coefficient


@dataclass(frozen=True)
class TransmissionPort:
    """One modal port plane for :class:`ModeTransmission`.

    ``out_direction`` points away from the design. The input port's incident
    wave travels in the opposite direction. The plane is snapped by
    :func:`photonhub.analysis.mode_monitor`, and its four-field monitor is
    cropped clear of transverse PML, as for :class:`ModePower`: on each
    transverse PML axis the aperture starts 2.25 cells after the low PML slab
    and ends 2.75 cells before the high one. Both faces must sit at the same
    quarter-cell phase so that all four tangential components snap to the same
    cells, so the two margins differ by half a cell and the aperture center
    lies a quarter cell below the domain center. The overlap therefore differs
    from a full-plane port when the mode's tail reaches the crop; the
    objective warns when more than 1e-3 of the reference mode's sampled power
    lies outside the aperture.
    """

    mode: Union[Mode, VectorMode]
    axis: str
    position_um: float
    out_direction: str
    name: str
    center_um: Optional[Tuple[float, float]] = None
    thickness_axis: Optional[str] = None

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y", "z") or self.out_direction not in ("+", "-"):
            raise ValueError("port axis must be x, y, or z and out_direction must be + or -")

    def sport(self, simulation: Simulation, freq_hz: float) -> SPort:
        """Build the S-matrix port and its PML-clear four-field monitor."""
        mm = mode_monitor(
            simulation, self.mode, axis=self.axis, position_um=self.position_um,
            freqs_hz=[freq_hz], name=self.name, direction=self.out_direction,
            center_um=self.center_um, thickness_axis=self.thickness_axis)
        crop = ModePower(
            mode=self.mode, axis=self.axis, position_um=self.position_um,
            freq_hz=freq_hz, direction=self.out_direction, name=self.name,
            center_um=self.center_um,
            thickness_axis=self.thickness_axis).monitor(simulation)
        return SPort(self.name, replace(mm, field_monitor=crop),
                     out_direction=self.out_direction)


# The settings behind the magnitudes that the reference-mode warnings quote.
_READOUT_TESTS = ("two reflective 3D CPU tests (0.05 and 0.04 um uniform Yee "
                  "mesh, subpixel off, non-dispersive Si in SiO2, PML 6 per axis, "
                  "3000 fixed steps, shutoff 0, input planes 0.75 to 15.75 cells "
                  "before the design)")


def _warn_reference_modes(ports: Tuple[TransmissionPort, ...]) -> None:
    """Warn when a port's reference mode is scalar or was solved smoothed."""
    scalar, smoothed = [], []
    for p in ports:
        supersample = (getattr(p.mode, "solve_params", None) or {}).get("supersample", 1)
        if not getattr(p.mode, "yee_staggered", False):
            scalar.append(p.name)
        elif supersample != 1:
            smoothed.append(f"{p.name} (supersample={supersample})")
    remedy = ("; solve the port modes with solve_mode_on_cross_section(..., "
              "supersample=1). A fixed incident_amplitude removes the "
              "reflection term from the gradient, not this error")
    if scalar:
        warnings.warn(
            f"reference mode on port(s) {', '.join(scalar)} is not "
            "Yee-staggered (scalar or node-collocated), so the incident "
            "reading carries a plane-dependent error that distance from the "
            f"source does not remove: on {_READOUT_TESTS} such modes moved "
            "the same-run incident power by up to 5.7% from a design-free "
            "run, against 0.16% and 0.24% for Yee-staggered modes solved "
            "with supersample=1" + remedy,
            UserWarning, stacklevel=caller_stacklevel())
    if smoothed:
        warnings.warn(
            f"reference mode on port(s) {', '.join(smoothed)} was solved on "
            "a smoothed cross-section but the adjoint runs subpixel off, so "
            "the incident reading carries a plane-dependent error that "
            f"distance from the source does not remove: on {_READOUT_TESTS} "
            "such modes moved the same-run incident power by up to 0.42% "
            "and 1.0% from a design-free run, against 0.16% and 0.24% with "
            "supersample=1" + remedy,
            UserWarning, stacklevel=caller_stacklevel())


def _warn_cropped_ports(warned: set, descriptions, ports, simulation: Simulation) -> None:
    """Warn once per port aperture when the mode reaches past its crop."""
    for desc, port in zip(descriptions, ports):
        aperture = port.monitor.field_monitor
        key = (desc.name, tuple(aperture.center_um), tuple(aperture.size_um),
               getattr(simulation.grid, "dl_um", None))
        if key not in warned:
            warned.add(key)
            fraction = _port_cropped_power_fraction(simulation, port)
            if fraction > 1e-3:
                warnings.warn(
                    f"reference mode on port {desc.name!r} carries "
                    f"{fraction:.3g} of sampled power outside its cropped aperture",
                    UserWarning, stacklevel=caller_stacklevel())


@dataclass(frozen=True)
class ModeTransmission:
    """Single-frequency modal transmission between two finite-aperture ports.

    By default ``T = |b_out|**2 / |a_in|**2`` reads the incident wave
    ``a_in`` and the outgoing wave ``b_out`` from the same forward run. This
    is the S-matrix element ``|S_out,in|**2`` of this objective's own ports,
    which are cropped clear of transverse PML (see :class:`TransmissionPort`);
    a full-plane port on the same run reads a slightly different value. The
    adjoint differentiates both waves exactly. Include :meth:`monitors` in
    ``build_forward``.

    How well ``T`` measures power transmission depends on the reference
    modes. The incident projection also picks up part of the design's
    reflection through any mismatch between the reference mode and the
    solver's own discrete mode, and that leak changes with the plane position
    rather than fading with distance from the design or from the source. Two
    reflective 3D CPU tests measured it: the adjoint validation record (0.05
    um mesh, eight input planes 0.75 to 7.75 cells before the design) and an
    independent review scene (0.04 um mesh, nine planes 0.75 to 15.75 cells
    before the design), both on a uniform Yee mesh with subpixel off,
    non-dispersive Si in SiO2, six PML layers on every axis, 3000 fixed
    steps and shutoff 0. Across the planes the incident power differed from
    a design-free run by up to 5.7% with scalar modes in both tests, 0.42%
    and 1.0% with Yee-staggered modes solved on the smoothed cross-section
    (``solve_mode_on_cross_section`` with its default ``supersample=8``),
    and 0.16% and 0.24% with Yee-staggered modes solved on the staircase
    permittivity of the subpixel-off run (``supersample=1``); the same-run
    ``T`` spread 6.4% and 12.4%, 1.7% and 1.4%, and 0.11% and 0.16%. The
    adjoint runs subpixel off, so solve port modes with ``supersample=1``.
    Constructing the objective warns when a port's mode is not
    Yee-staggered, or records a smoothed solve (``supersample`` other than
    1 in its ``solve_params``).

    The geometry guard requires the input readout rows to lie strictly
    between the source and the design and the output rows to lie beyond the
    design. With staircase Yee modes nothing more was needed in either test:
    the design-free guide read within 0.11% of 1 at every plane from 1.25
    cells past the source. Scalar and smoothed readings carry a
    plane-dependent error, several percent with scalar modes and up to about
    1% with smoothed ones, that distance from the source does not remove: in
    the review scene the scalar design-free guide read 0.985 to 1.036 over
    planes 1.5 to 16.5 cells past the source, beating without settling. The
    remedy is staircase Yee modes (``supersample=1``), not a larger gap.

    Pass ``incident_amplitude`` to hold the denominator fixed:
    ``T = |b_out|**2 / |a_ref|**2``, where ``a_ref`` is
    :meth:`reference_incident_amplitude` of a design-free run with the same
    source, mesh and input port. The adjoint then keeps only the output-port
    term. Prefer it for strongly reflective designs, so that an optimizer
    cannot raise ``T`` by steering the reflection into the incident reading:
    at input planes 0.75 cells and 6.75 or 7.75 cells before the design, the
    same-run incident term was 6.8-22% of the gradient norm with scalar
    modes, 0.7-3.8% with smoothed and 0.37-1.6% with staircase Yee modes
    over the two tests. The fixed amplitude removes that reflection term from
    the gradient, not the plane-dependent error of a scalar or smoothed mode,
    which also enters ``a_ref``. It assumes the source launches the same wave
    with and without the design, which a mode source does up to the
    reflection that the boundary behind the source returns: 0.17% of the
    incident power with six PML layers in the review scene and 0.003% with
    twelve. The same-run ``a`` includes that return and ``a_ref`` cannot, so
    fixed-incident ``T`` carries it (0.27% higher with twelve layers there),
    and it is part of the staircase residual above.
    """

    input_port: TransmissionPort
    output_port: TransmissionPort
    freq_hz: float
    incident_amplitude: Optional[complex] = None
    _port_cache: list = field(default_factory=list, init=False, repr=False, compare=False)
    _crop_warned: set = field(default_factory=set, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.freq_hz) or self.freq_hz <= 0:
            raise ValueError("freq_hz must be positive and finite")
        if (self.input_port.name == self.output_port.name or
                "design_region" in (self.input_port.name, self.output_port.name)):
            raise ValueError("port names must differ and cannot use reserved 'design_region'")
        if self.incident_amplitude is not None:
            a = complex(self.incident_amplitude)
            if not np.isfinite(a) or a == 0:
                raise ValueError("incident_amplitude must be finite and nonzero")
        _warn_reference_modes((self.input_port, self.output_port))

    def ports(self, simulation: Simulation) -> Tuple[SPort, SPort]:
        """Return the exact finite-aperture SPorts used by this objective.

        The ports of the two most recent simulations are cached by identity,
        so repeated readouts of one run do not rebuild the monitors or
        re-solve the automatic per-frequency set of modes. The crop warning fires
        once per port name and aperture (the cropped monitor's center and
        size, and the grid step), so a later simulation with a smaller
        cross-section or more PML layers is checked again."""
        for known, ports in self._port_cache:
            if known is simulation:
                return ports
        ports = (self.input_port.sport(simulation, self.freq_hz),
                 self.output_port.sport(simulation, self.freq_hz))
        self._port_cache.append((simulation, ports))
        del self._port_cache[:-2]
        _warn_cropped_ports(self._crop_warned, (self.input_port, self.output_port),
                            ports, simulation)
        return ports

    def monitors(self, simulation: Simulation) -> Tuple[ProfileMonitor, ProfileMonitor]:
        """The input and output four-field monitors for ``build_forward``."""
        return tuple(p.monitor.field_monitor for p in self.ports(simulation))

    def waves(self, data: RunResult, simulation: Optional[Simulation] = None
              ) -> Tuple[complex, complex]:
        """Return same-run incident input ``a`` and outgoing output ``b``."""
        simulation = simulation if simulation is not None else data.simulation
        pin, pout = self.ports(simulation)
        a = pin.incoming(data)[self.freq_hz]
        b = pout.outgoing(data)[self.freq_hz]
        if a == 0 and self.incident_amplitude is None:
            raise ValueError("incident amplitude at input port is zero")
        return a, b

    def reference_incident_amplitude(
        self, data: RunResult, simulation: Optional[Simulation] = None
    ) -> complex:
        """Read ``a_ref`` from a design-free run with the same input port.

        Run a normalization simulation with the same source, mesh, run
        window, input monitor and input guide, without the design structures,
        and pass the returned complex amplitude as ``incident_amplitude`` to a
        new objective with the same ports.
        """
        simulation = simulation if simulation is not None else data.simulation
        a = complex(self.ports(simulation)[0].incoming(data)[self.freq_hz])
        if not np.isfinite(a) or a == 0:
            raise ValueError("reference incident amplitude must be finite and nonzero")
        return a

    def amplitude(self, data: RunResult, simulation: Optional[Simulation] = None) -> complex:
        """Complex ``b/a``, using the fixed reference when supplied."""
        a, b = self.waves(data, simulation)
        return b / (self.incident_amplitude if self.incident_amplitude is not None else a)

    def value(self, data: RunResult, simulation: Optional[Simulation] = None) -> float:
        """Power ratio of the output wave to the chosen incident reading."""
        return float(abs(self.amplitude(data, simulation)) ** 2)

    def transpose_sources(self, simulation: Simulation, data: RunResult):
        """One point-current cloud for the exact differential of transmission."""
        pin, pout = self.ports(simulation)
        a, b = self.waves(data, simulation)
        denominator = self.incident_amplitude if self.incident_amplitude is not None else a
        power_in = abs(denominator) ** 2
        factors = [(pout, pout.out_direction, np.conj(b) / power_in)]
        if self.incident_amplitude is None:
            factors.insert(0, (pin, pin.in_direction,
                               -abs(b) ** 2 * np.conj(a) / power_in ** 2))
        weights = []
        for port, direction, factor in factors:
            weights.extend(_transmission_port_weights(port, direction, data,
                                                       self.freq_hz, factor))
        return _transmission_cloud(simulation, weights, self.freq_hz)


def _port_cropped_power_fraction(simulation: Simulation, port: SPort) -> float:
    """Estimate modal power omitted by a port's transverse aperture."""
    if not isinstance(simulation.grid, UniformMesh):
        return 0.0
    mon = port.monitor
    axis = mon.axis
    t1, t2 = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}[axis]
    dl = simulation.grid.dl_um
    c1 = np.arange(realized_cells(simulation.size_um["xyz".index(t1)], dl, False)) * dl
    c2 = np.arange(realized_cells(simulation.size_um["xyz".index(t2)], dl, False)) * dl
    cen = mon.center_um or (float(np.mean(c1)), float(np.mean(c2)))
    mode_fields = vector_modal_fields if hasattr(mon.mode, "hx") else modal_fields
    m = mode_fields(mon.mode, c1, c2, axis=axis, direction=port.out_direction,
                    center_um=cen, thickness_axis=mon.thickness_axis)
    density = np.abs(np.real(m["e1"] * np.conj(m["h2"])
                             - m["e2"] * np.conj(m["h1"])))
    total = float(np.sum(density))
    if total == 0:
        return 0.0
    field_mon = mon.field_monitor
    bounds = [(field_mon.center_um["xyz".index(t)] -
               field_mon.size_um["xyz".index(t)] / 2,
               field_mon.center_um["xyz".index(t)] +
               field_mon.size_um["xyz".index(t)] / 2) for t in (t1, t2)]
    inside = ((c1 >= bounds[0][0]) & (c1 <= bounds[0][1]))[None, :] & \
             ((c2 >= bounds[1][0]) & (c2 <= bounds[1][1]))[:, None]
    return float(np.sum(density[~inside]) / total)


def _transmission_port_weights(port: SPort, direction: str, data: RunResult,
                               freq_hz: float, factor: complex):
    """Raw Yee weights for the SPort amplitude times one real-variation factor."""
    mon = port.monitor
    axis = mon.axis
    t1, t2 = {"x": ("y", "z"), "y": ("z", "x"),
              "z": ("x", "y")}[axis]
    e1, e2, h1, h2 = (f"E{t1}", f"E{t2}", f"H{t1}", f"H{t2}")
    plane = data[mon.name]
    arrays = {c: plane.sel(component=c) for c in (e1, e2, h1, h2)}
    _, c1, c2 = _plane_component(arrays, e1, freq_hz, t1, t2)
    bank = mon._resolved_modes_by_freq()
    use_mode = mon.mode if not bank else bank[min(bank, key=lambda k: abs(k - freq_hz))]
    w1 = _widths_with_grid(c1, use_mode, t1, t1, t2)
    w2 = _widths_with_grid(c2, use_mode, t2, t1, t2)
    area = np.outer(w2, w1)
    cen = mon.center_um or (float(np.mean(c1)), float(np.mean(c2)))
    mode_fields = vector_modal_fields if hasattr(use_mode, "hx") else modal_fields
    mode = mode_fields(use_mode, c1, c2, axis=axis, direction=direction,
                       center_um=cen, thickness_axis=mon.thickness_axis)
    me1, me2, mh1, mh2 = (mode[k] for k in ("e1", "e2", "h1", "h2"))
    power = 0.5 * np.sum(np.real(me1 * np.conj(mh2)
                                - me2 * np.conj(mh1)) * area)
    if power == 0:
        raise ValueError(f"mode has zero power on port {port.name!r}")
    phi = (1 if direction == "+" else -1) * (
        np.pi * float(use_mode.n_eff) * mon.dl_um * freq_hz / _C0 * 1e-6)
    cosine = np.cos(phi)
    if abs(cosine) < 0.2:
        raise ValueError("longitudinal de-stagger is singular on a transmission port")
    wave_scale = np.sqrt(abs(power * cosine) * 1e-12)
    prefactor = factor * wave_scale / (4 * power * cosine)
    q = {e1: prefactor * np.exp(-1j * phi) * np.conj(mh2) * area,
         e2: -prefactor * np.exp(-1j * phi) * np.conj(mh1) * area,
         h1: -prefactor * np.conj(me2) * area,
         h2: prefactor * np.conj(me1) * area}
    if not getattr(use_mode, "yee_staggered", False):
        for comp, arr in q.items():
            transverse = 1 if comp in (e1, h2) else 0
            half = 0.5 * arr
            shifted = np.roll(half, -1, axis=transverse)
            edge = [slice(None)] * 2
            edge[transverse] = -1
            shifted[tuple(edge)] = 0
            q[comp] = half + shifted
    offsets = {"Ex": (0.5, 0, 0), "Ey": (0, 0.5, 0),
               "Ez": (0, 0, 0.5), "Hx": (0, 0.5, 0.5),
               "Hy": (0.5, 0, 0.5), "Hz": (0.5, 0.5, 0)}
    out = []
    for comp, values in q.items():
        for j, i in np.ndindex(values.shape):
            value = complex(values[j, i])
            if value == 0:
                continue
            loc = {letter: float(plane.coords[letter].values[0])
                   for letter in "xyz"}
            loc[t1], loc[t2] = float(c1[i]), float(c2[j])
            point = tuple(loc[letter] + offsets[comp][k] * mon.dl_um
                          for k, letter in enumerate("xyz"))
            out.append((comp, point, value))
    return out


def _transmission_cloud(simulation: Simulation, weights, freq_hz: float):
    """Normalize the combined electric/magnetic cloud at one electric anchor."""
    combined = {}
    for comp, point, value in weights:
        key = (comp, tuple(round(p, 12) for p in point))
        combined[key] = combined.get(key, 0j) + value
    electric = [(key, value) for key, value in combined.items()
                if key[0].startswith("E")]
    if not electric:
        raise ValueError("transmission readout has no electric weights")
    anchor, anchor_value = max(electric, key=lambda item: abs(item[1]))
    scale = abs(anchor_value)
    if scale == 0:
        raise ValueError("transmission readout has zero electric weights")
    rotate = np.exp(-1j * np.angle(anchor_value))
    pulse = _adjoint_pulse(simulation)
    omega_dt = 2 * np.pi * freq_hz * _time_step(simulation)
    sources = []
    for key in (anchor, *(k for k in combined if k != anchor)):
        comp, point = key
        value = complex(combined[key] * rotate / scale)
        if value == 0:
            continue
        if comp.startswith("H"):
            value *= -np.exp(0.5j * omega_dt)
        sources.append(PointDipole(
            center_um=point, polarization=comp, amplitude=abs(value),
            source_time=pulse.model_copy(
                update={"phase": -float(np.angle(value))})))
    return tuple(sources), complex(scale * np.conj(rotate))


# ---------------------------------------------------------------------------
# Broadband objective: several frequencies from one forward run
# ---------------------------------------------------------------------------

_REDUCTIONS = ("mean", "sum", "softmin")


def _band_sport(port: TransmissionPort, simulation: Simulation, freqs: Tuple[float, ...],
                modes_by_freq=None) -> SPort:
    """:meth:`TransmissionPort.sport` recording every frequency of ``freqs``."""
    mm = mode_monitor(
        simulation, port.mode, axis=port.axis, position_um=port.position_um,
        freqs_hz=list(freqs), name=port.name, direction=port.out_direction,
        center_um=port.center_um, thickness_axis=port.thickness_axis,
        modes_by_freq=modes_by_freq)
    crop = ModePower(
        mode=port.mode, axis=port.axis, position_um=port.position_um,
        freq_hz=freqs[0], direction=port.out_direction, name=port.name,
        center_um=port.center_um,
        thickness_axis=port.thickness_axis).monitor(simulation)
    crop = crop.model_copy(update={"freqs_hz": tuple(freqs)})
    return SPort(port.name, replace(mm, field_monitor=crop),
                 out_direction=port.out_direction)


def _at_frequencies(values: dict, freqs: Tuple[float, ...]) -> np.ndarray:
    """``{freq: value}`` read at ``freqs`` in order, matching the nearest key."""
    items = list(values.items())
    keys = np.asarray([key for key, _ in items], dtype=float)
    out = np.empty(len(freqs), dtype=complex)
    for k, f in enumerate(freqs):
        nearest = int(np.argmin(np.abs(keys - f)))
        if abs(keys[nearest] - f) > 1e-9 * f:
            raise ValueError(f"port reading has no frequency {f:.9g} Hz")
        out[k] = items[nearest][1]
    return out


@dataclass(frozen=True)
class BroadbandModeTransmission:
    """Modal transmission at several frequencies, reduced to one figure of merit.

    ``T_k = |b_k|**2 / |a_k|**2`` is read at every frequency of ``freqs_hz``
    from ONE forward run: ``b_k`` is the outgoing wave of the output port and
    ``a_k`` the fixed ``incident_amplitudes[k]``, or the same-run incident wave
    at the input port when ``incident_amplitudes`` is None. The ports are the
    finite-aperture :class:`TransmissionPort` planes of
    :class:`ModeTransmission`, recording all frequencies. A port mode solved
    with ``solve_mode_on_cross_section`` carries its solve settings, and
    each frequency is then read against the mode re-solved at that
    frequency; the re-solved modes are kept for later simulations of the
    same objective. Read every ``a_k`` from one design-free broadband run
    with :meth:`reference_incident_amplitudes`.

    The figure of merit is ``F = reduction(T)``:

    - ``"mean"``: ``sum_k w_k T_k / sum_k w_k``, with ``w`` = ``weights``
      (default all 1);
    - ``"sum"``: ``sum_k w_k T_k``;
    - ``"softmin"``: ``-(1/beta) log(sum_k w_k exp(-beta T_k) / sum_k w_k)``
      with ``beta`` = ``softmin_beta``, a smooth minimum that lies between the
      smallest ``T_k`` and the weighted mean and approaches the minimum as
      ``beta`` grows;
    - a callable taking the array ``T`` and returning ``(F, dF/dT)``.

    :func:`value_and_gradient` returns ``F`` and ``dF/drho`` from one forward
    and one adjoint solve whatever the number of frequencies. Its adjoint run
    drives every port sample with several delayed copies of the forward
    pulse whose amplitudes are solved so the summed current has, at each
    ``f_k``, exactly the single-frequency adjoint current scaled by
    ``dF/dT_k``; the adjoint window is longer than the forward one by the
    span of those delays, about ``1/df`` for the smallest frequency spacing
    ``df``. Include :meth:`monitors` and ``region.monitor(freqs_hz)`` in
    ``build_forward``, and give the forward simulation one pulse that covers
    every frequency (each within 3.7 ``fwidth_hz`` of its ``freq0_hz``, where
    its spectrum is at least 1e-3 of the peak).

    Launch each frequency's own mode when ``T_k`` should match one run per
    wavelength: ``mode_launch(..., modes_by_freq=...)`` with a mode solved at
    every frequency. A launch of one profile injects the centre frequency's
    mode at every frequency, while each ``T_k`` is read against the port
    mode of its own frequency. On a quasi-2D CPU taper (0.025 um uniform
    mesh, subpixel off, non-dispersive, 12 PML layers on x and y, 8000 fixed
    steps, shutoff 0, 21 wavelengths from 1.35 to 1.85 um, one pulse centred
    at 1.55 um) the one-run ``T_k`` differed from one run per wavelength by
    up to 1.3% at the ends of the band with the centre mode launched, and by
    at most 5.2e-4 with each frequency's own mode.
    """

    input_port: TransmissionPort
    output_port: TransmissionPort
    freqs_hz: Tuple[float, ...]
    incident_amplitudes: Optional[Tuple[complex, ...]] = None
    reduction: Union[str, Callable[[np.ndarray], Tuple[float, np.ndarray]]] = "mean"
    weights: Optional[Tuple[float, ...]] = None
    softmin_beta: float = 20.0
    _port_cache: list = field(default_factory=list, init=False, repr=False, compare=False)
    _crop_warned: set = field(default_factory=set, init=False, repr=False, compare=False)
    _banks: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        freqs = tuple(float(f) for f in np.atleast_1d(np.asarray(self.freqs_hz, dtype=float)))
        if not freqs or not all(np.isfinite(f) and f > 0 for f in freqs):
            raise ValueError("freqs_hz must be one or more positive finite frequencies")
        if len(set(freqs)) != len(freqs):
            raise ValueError("freqs_hz must not repeat a frequency")
        object.__setattr__(self, "freqs_hz", freqs)
        if (self.input_port.name == self.output_port.name or
                "design_region" in (self.input_port.name, self.output_port.name)):
            raise ValueError("port names must differ and cannot use reserved 'design_region'")
        if self.incident_amplitudes is not None:
            a = np.atleast_1d(np.asarray(self.incident_amplitudes, dtype=complex))
            if a.shape != (len(freqs),) or not np.all(np.isfinite(a)) or np.any(a == 0):
                raise ValueError("incident_amplitudes must hold one finite nonzero "
                                 "amplitude per frequency")
            object.__setattr__(self, "incident_amplitudes", tuple(complex(v) for v in a))
        if not callable(self.reduction) and self.reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS} or a callable")
        if self.weights is not None:
            w = np.atleast_1d(np.asarray(self.weights, dtype=float))
            if (w.shape != (len(freqs),) or not np.all(np.isfinite(w)) or np.any(w < 0)
                    or not np.sum(w) > 0):
                raise ValueError("weights must be one nonnegative finite weight per "
                                 "frequency with a positive sum")
            object.__setattr__(self, "weights", tuple(float(v) for v in w))
        if not (np.isfinite(self.softmin_beta) and self.softmin_beta > 0):
            raise ValueError("softmin_beta must be positive and finite")
        _warn_reference_modes((self.input_port, self.output_port))

    # -- ports and readout --------------------------------------------------

    def ports(self, simulation: Simulation) -> Tuple[SPort, SPort]:
        """The two finite-aperture ports, recording every frequency.

        The ports of the two most recent simulations are cached by identity,
        and each port's per-frequency modes by port name, grid and domain
        size, so readouts of later simulations on the same mesh re-solve no
        mode."""
        for known, ports in self._port_cache:
            if known is simulation:
                return ports
        ports = []
        for desc in (self.input_port, self.output_port):
            key = (desc.name, repr(simulation.grid), tuple(simulation.size_um))
            bank = self._banks.get(key)
            port = _band_sport(desc, simulation, self.freqs_hz, modes_by_freq=bank)
            if key not in self._banks:
                self._banks[key] = port.monitor._resolved_modes_by_freq()
            ports.append(port)
        ports = tuple(ports)
        self._port_cache.append((simulation, ports))
        del self._port_cache[:-2]
        _warn_cropped_ports(self._crop_warned, (self.input_port, self.output_port),
                            ports, simulation)
        return ports

    def monitors(self, simulation: Simulation) -> Tuple[ProfileMonitor, ProfileMonitor]:
        """The input and output four-field monitors for ``build_forward``."""
        return tuple(p.monitor.field_monitor for p in self.ports(simulation))

    def waves(self, data: RunResult, simulation: Optional[Simulation] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Same-run incident waves ``a`` and outgoing waves ``b``, one per frequency."""
        simulation = simulation if simulation is not None else data.simulation
        pin, pout = self.ports(simulation)
        a = _at_frequencies(pin.incoming(data), self.freqs_hz)
        b = _at_frequencies(pout.outgoing(data), self.freqs_hz)
        if self.incident_amplitudes is None and np.any(a == 0):
            raise ValueError("incident amplitude at input port is zero")
        return a, b

    def reference_incident_amplitudes(
        self, data: RunResult, simulation: Optional[Simulation] = None
    ) -> Tuple[complex, ...]:
        """Read every ``a_ref(f_k)`` from one design-free broadband run.

        Run a normalization simulation with the same source, mesh, run
        window, input monitor and input guide, without the design structures,
        and pass the returned amplitudes as ``incident_amplitudes`` to a new
        objective with the same ports and frequencies."""
        simulation = simulation if simulation is not None else data.simulation
        a = _at_frequencies(self.ports(simulation)[0].incoming(data), self.freqs_hz)
        if not np.all(np.isfinite(a)) or np.any(a == 0):
            raise ValueError("reference incident amplitudes must be finite and nonzero")
        return tuple(complex(v) for v in a)

    def _denominators(self, a: np.ndarray) -> np.ndarray:
        if self.incident_amplitudes is None:
            return a
        return np.asarray(self.incident_amplitudes, dtype=complex)

    def amplitudes(self, data: RunResult, simulation: Optional[Simulation] = None
                   ) -> np.ndarray:
        """Complex ``b_k/a_k`` per frequency, using the fixed references when supplied."""
        a, b = self.waves(data, simulation)
        return b / self._denominators(a)

    def transmissions(self, data: RunResult, simulation: Optional[Simulation] = None
                      ) -> np.ndarray:
        """``T_k`` per frequency, in ``freqs_hz`` order."""
        return np.abs(self.amplitudes(data, simulation)) ** 2

    def reduce(self, transmissions) -> Tuple[float, np.ndarray]:
        """``(F, dF/dT)`` of the reduction at the given ``T_k``."""
        t = np.asarray(transmissions, dtype=float)
        if t.shape != (len(self.freqs_hz),):
            raise ValueError("one transmission per frequency is required")
        if callable(self.reduction):
            out = self.reduction(t.copy())
            if not isinstance(out, tuple) or len(out) != 2:
                raise ValueError("a reduction callable must return (F, dF/dT)")
            value = float(out[0])
            slope = np.asarray(out[1], dtype=float)
            if slope.shape != t.shape or not np.isfinite(value) or not np.all(np.isfinite(slope)):
                raise ValueError("a reduction callable must return a finite F and one "
                                 "finite dF/dT per frequency")
            return value, slope
        w = (np.ones_like(t) if self.weights is None
             else np.asarray(self.weights, dtype=float))
        if self.reduction == "mean":
            return float(w @ t / np.sum(w)), w / np.sum(w)
        if self.reduction == "sum":
            return float(w @ t), w.copy()
        z = -self.softmin_beta * t
        used = w > 0
        peak = float(np.max(z[used]))
        e = np.zeros_like(t)
        e[used] = w[used] * np.exp(z[used] - peak)
        total = float(np.sum(e))
        value = -(peak + np.log(total / np.sum(w))) / self.softmin_beta
        return float(value), e / total

    def value(self, data: RunResult, simulation: Optional[Simulation] = None) -> float:
        """The figure of merit ``F``."""
        return self.reduce(self.transmissions(data, simulation))[0]

    def _adjoint_targets(self, simulation: Simulation, data: RunResult):
        """Per port sample, the adjoint current each frequency needs.

        Returns ``(targets, readout)``: ``targets`` maps ``(component, point)``
        to the complex current at every frequency, the single-frequency
        transpose weights of :class:`ModeTransmission` scaled by ``dF/dT_k``,
        with the half-step phase of magnetic injection; ``readout`` holds
        ``a``, ``b``, ``T``, ``F`` and ``dF/dT``."""
        pin, pout = self.ports(simulation)
        a, b = self.waves(data, simulation)
        den = self._denominators(a)
        t = np.abs(b / den) ** 2
        value, slope = self.reduce(t)
        dt = _time_step(simulation)
        n = len(self.freqs_hz)
        targets: dict = {}
        for k, f in enumerate(self.freqs_hz):
            power_in = abs(den[k]) ** 2
            factors = [(pout, pout.out_direction, slope[k] * np.conj(b[k]) / power_in)]
            if self.incident_amplitudes is None:
                factors.insert(0, (pin, pin.in_direction,
                                   -slope[k] * abs(b[k]) ** 2 * np.conj(a[k]) / power_in ** 2))
            magnetic = -np.exp(0.5j * 2 * np.pi * f * dt)
            for port, direction, factor in factors:
                if factor == 0:
                    continue
                for comp, point, weight in _transmission_port_weights(
                        port, direction, data, f, factor):
                    key = (comp, tuple(round(p, 12) for p in point))
                    current = targets.setdefault(key, np.zeros(n, dtype=complex))
                    current[k] += weight * (magnetic if comp.startswith("H") else 1.0)
        readout = dict(a=a, b=b, transmissions=t, value=value, slope=slope)
        return targets, readout


def _pulse_terms(pulse, freqs) -> Tuple[np.ndarray, np.ndarray]:
    """The two terms ``(P, Q)`` of a Gaussian pulse's unit-amplitude spectrum
    at ``freqs``, without its phase: the pulse with phase ``phi`` has the
    spectrum ``exp(-1j phi) P + exp(1j phi) Q``, the spectrum the solver divides
    frequency-domain results by."""
    f = np.asarray(freqs, dtype=float)
    tau = 1.0 / (2.0 * np.pi * pulse.fwidth_hz)
    t0 = pulse.offset * tau
    pref = tau * np.sqrt(2.0 * np.pi) / 2.0 * np.exp(1j * 2 * np.pi * f * t0)
    return (pref * np.exp(-2.0 * np.pi ** 2 * tau ** 2 * (f - pulse.freq0_hz) ** 2),
            pref * np.exp(-2.0 * np.pi ** 2 * tau ** 2 * (f + pulse.freq0_hz) ** 2))


def _pulse_spectrum(pulse, freqs) -> np.ndarray:
    """The engine's analytic unit-amplitude spectrum of a Gaussian pulse,
    phase included, at ``freqs``: the factor the engine divides DFTs by."""
    p_term, q_term = _pulse_terms(pulse, freqs)
    return np.exp(-1j * pulse.phase) * p_term + np.exp(1j * pulse.phase) * q_term


# Taps per frequency, the span grid in units of 1/(smallest spacing), the
# smallest relative spectrum of the forward pulse at a design frequency, and
# the adjoint window extension, as a multiple of the forward window, above
# which the objective warns.
_TAPS_PER_FREQUENCY = 1.5
_TAP_SPANS = np.linspace(0.5, 1.5, 41)
_BAND_FLOOR = 1e-3
_SPAN_WARN_RATIO = 10.0


@dataclass(frozen=True)
class _TapPlan:
    """Delayed copies of one pulse and the solve that sets their amplitudes.

    A tap dipole of amplitude ``A`` and phase ``phi`` has, at ``f > 0``, the
    spectrum ``c P_m(f) + conj(c) Q_m(f)`` with ``c = A exp(-1j phi)``.
    ``solve`` returns, per port sample, the minimum-norm ``c`` whose summed
    spectrum equals the target at every design frequency."""

    freqs: Tuple[float, ...]
    pulses: Tuple[object, ...]       # one plain, zero-phase pulse per tap
    span_s: float
    reference: np.ndarray            # R_k, divided out after the adjoint run
    inverse: np.ndarray              # (2M, 2N) minimum-norm solve
    condition: float

    @property
    def taps(self) -> int:
        """Number of delayed pulses per port sample."""
        return len(self.pulses)

    def solve(self, targets: np.ndarray) -> np.ndarray:
        """Tap coefficients ``c``, shape ``(samples, taps)``, for spectra
        ``targets * reference`` of shape ``(samples, frequencies)``."""
        x = np.asarray(targets, dtype=complex) * self.reference[None, :]
        rhs = np.concatenate([x.real, x.imag], axis=1).T
        sol = self.inverse @ rhs
        m = self.taps
        return (sol[:m] + 1j * sol[m:]).T


def _tap_system(freqs, pulses) -> np.ndarray:
    """Real ``(2N, 2M)`` map from ``(Re c, Im c)`` of the taps to the real and
    imaginary parts of their summed spectrum at the ``N`` frequencies:
    ``c P + conj(c) Q = Re(c) (P + Q) + 1j Im(c) (P - Q)``."""
    terms = [_pulse_terms(p, freqs) for p in pulses]
    p_term = np.stack([t[0] for t in terms], axis=1)
    q_term = np.stack([t[1] for t in terms], axis=1)
    plus, minus = p_term + q_term, p_term - q_term
    return np.block([[plus.real, -minus.imag], [plus.imag, minus.real]])


def _tap_plan(freqs: Tuple[float, ...], pulse) -> _TapPlan:
    """Choose the taps for ``freqs`` from the forward pulse alone.

    ``M = ceil(1.5 N)`` copies of the pulse, delayed evenly over a span
    picked from 0.5 to 1.5 times ``1/df`` (``df`` the smallest frequency
    spacing) to minimize the condition number of the solve; one frequency
    takes the pulse itself."""
    from .components import GaussianPulse

    base = GaussianPulse(freq0_hz=pulse.freq0_hz, fwidth_hz=pulse.fwidth_hz,
                         offset=pulse.offset)
    tau = 1.0 / (2.0 * np.pi * base.fwidth_hz)
    n = len(freqs)

    def taps(span):
        m = 1 if n == 1 else int(np.ceil(_TAPS_PER_FREQUENCY * n))
        delays = np.linspace(0.0, span, m) if m > 1 else np.zeros(1)
        return tuple(base.model_copy(update={"offset": float(base.offset + d / tau)})
                     for d in delays)

    if n == 1:
        spans = (0.0,)
    else:
        spacing = float(np.min(np.diff(np.sort(np.asarray(freqs, dtype=float)))))
        spans = tuple(float(s) / spacing for s in _TAP_SPANS)
    best = None
    for span in spans:
        pulses = taps(span)
        system = _tap_system(freqs, pulses)
        sv = np.linalg.svd(system, compute_uv=False)
        condition = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        if best is None or condition < best[0]:
            best = (condition, span, pulses, system)
    condition, span, pulses, system = best
    if not np.isfinite(condition) or condition > 1e6:
        raise ValueError("the design frequencies cannot be driven independently by "
                         "delayed copies of the forward pulse; check that they are "
                         "distinct and inside the pulse's band")
    middle = base.model_copy(update={"offset": float(base.offset + 0.5 * span / tau)})
    reference = _pulse_spectrum(middle, freqs)
    return _TapPlan(freqs=tuple(freqs), pulses=pulses, span_s=span,
                    reference=reference, inverse=np.linalg.pinv(system),
                    condition=condition)


def _band_pulse(simulation: Simulation, freqs: Tuple[float, ...]):
    """The forward simulation's first pulse, checked to cover ``freqs``."""
    from .components import GaussianPulse

    if not simulation.sources:
        raise ValueError("the forward simulation has no source")
    pulse = getattr(simulation.sources[0], "source_time", None)
    if not isinstance(pulse, GaussianPulse):
        raise ValueError("a broadband gradient needs a Gaussian-pulse forward source")
    floor = np.sqrt(-2.0 * np.log(_BAND_FLOOR)) * pulse.fwidth_hz
    outside = [f for f in freqs if abs(f - pulse.freq0_hz) > floor]
    if outside:
        raise ValueError(
            f"frequencies {[f'{f:.6g}' for f in outside]} Hz lie more than "
            f"{floor / pulse.fwidth_hz:.2f} fwidth from the forward pulse's freq0, "
            "where its spectrum is below 1e-3 of the peak; widen fwidth_hz or "
            "recentre freq0_hz")
    return pulse


def _tap_sources(targets: dict, plan: _TapPlan):
    """The adjoint point currents for per-sample targets, and how to undo
    the engine's normalization.

    Returns ``(sources, scale, first)``: the dipoles (the largest first,
    with unit amplitude), the factor ``g`` they were divided by, and the
    first dipole's pulse, whose spectrum the engine normalizes by."""
    keys = list(targets)
    if not keys:
        raise ValueError("the broadband adjoint has no current: no transmission "
                         "changes F to first order (dF/dT_k or the output amplitude "
                         "is zero at every frequency)")
    coefficients = plan.solve(np.stack([targets[k] for k in keys]))
    magnitude = np.abs(coefficients)
    scale = float(np.max(magnitude))
    if not scale > 0:
        raise ValueError("the broadband adjoint has no current: no transmission "
                         "changes F to first order (dF/dT_k or the output amplitude "
                         "is zero at every frequency)")
    first = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    order = [first] + [(i, m) for i in range(len(keys)) for m in range(plan.taps)
                       if (i, m) != first]
    sources = []
    for i, m in order:
        value = complex(coefficients[i, m] / scale)
        if value == 0:
            continue
        comp, point = keys[i]
        sources.append(PointDipole(
            center_um=point, polarization=comp, amplitude=abs(value),
            source_time=plan.pulses[m].model_copy(
                update={"phase": -float(np.angle(value))})))
    return tuple(sources), scale, sources[0].source_time


# ---------------------------------------------------------------------------
# Gradient assembly
# ---------------------------------------------------------------------------

@dataclass
class GradientResult:
    """Objective and gradient for the subpixel-off model.

    ``shutoff`` is shared by the forward and adjoint solves. Their actual
    lengths are ``forward_steps_run`` and ``adjoint_steps_run``; they can differ
    when the caller's shutoff ends either solve early. ``forward_simulation``
    and ``adjoint_simulation`` are the exact specifications sent to the solver.
    """
    value: float                 # figure of merit J
    grad: np.ndarray             # dJ/drho, shape DesignRegion.shape (flat order)
    forward: RunResult
    adjoint: RunResult
    amplitude: complex           # forward objective phasor u
    forward_steps_run: Optional[int]
    adjoint_steps_run: Optional[int]
    shutoff: float
    forward_simulation: Optional[Simulation] = None
    adjoint_simulation: Optional[Simulation] = None


@dataclass
class BroadbandGradientResult(GradientResult):
    """Objective and gradient of a :class:`BroadbandModeTransmission`.

    ``value`` is ``F`` and ``grad`` is ``dF/drho``, from one forward and one
    adjoint solve. ``amplitude`` holds ``b_k/a_k`` per frequency, as does
    ``amplitudes``. ``transmissions`` are the ``T_k`` and ``dF_dT`` the
    reduction's slopes. ``gradients_by_freq[k]`` is frequency ``k``'s part of
    ``grad``, already weighted by ``dF_dT[k]``; the parts sum to ``grad``.
    The adjoint run drives each port sample with ``taps`` delayed pulses
    spread over ``tap_span_s`` seconds, ``adjoint_dipoles`` dipoles in all,
    and runs ``adjoint_extra_steps`` steps longer than the forward run.
    ``tap_condition`` is the condition number of the tap solve.
    """
    freqs_hz: Tuple[float, ...] = ()
    transmissions: Optional[np.ndarray] = None
    amplitudes: Optional[np.ndarray] = None
    dF_dT: Optional[np.ndarray] = None
    gradients_by_freq: Optional[np.ndarray] = None
    taps: int = 0
    tap_span_s: float = 0.0
    tap_condition: float = float("nan")
    adjoint_extra_steps: int = 0
    adjoint_dipoles: int = 0


@dataclass
class FiniteDifferenceResult:
    """Adjoint and central finite-difference gradients from real solver runs.

    ``indices`` selects flat controls: densities for
    :func:`finite_difference_check`, shape parameters for
    :func:`parametric_finite_difference_check`. ``value`` is the objective at
    the checked point. ``finite_difference_values`` and
    ``finite_difference_steps_run`` hold the plus and minus objective values
    and run lengths in index order. Checking all controls costs two extra
    forward solves per control, in addition to the adjoint pair. For a
    parameter check, ``jacobian`` is the ``drho/dp`` the adjoint gradient was
    chained through.
    """

    indices: Tuple[int, ...]
    h: float
    adjoint_gradient: np.ndarray
    finite_difference_gradient: np.ndarray
    cosine: float
    norm_ratio: float
    relative_l2: float
    forward_steps_run: Optional[int]
    adjoint_steps_run: Optional[int]
    finite_difference_steps_run: Tuple[Tuple[Optional[int], Optional[int]], ...]
    value: Optional[float] = None
    finite_difference_values: Tuple[Tuple[float, float], ...] = ()
    jacobian: Optional[np.ndarray] = None


@dataclass
class ParametricGradientResult:
    """Objective and its gradient with respect to shape parameters.

    ``grad`` is ``dJ/dp = jacobian.T @ density_gradient.grad``. ``jacobian``
    is ``drho/dp`` with shape ``(region.n_params, len(params))`` in flat
    density order, a ``scipy.sparse`` matrix when the caller's
    ``jacobian(p)`` returned one (as :meth:`PolygonProfile.jacobian` does),
    and ``jacobian_method`` says whether the caller supplied
    it or it is a central difference of ``expand``. ``rho`` is
    ``expand(params)``. ``density_gradient`` is the underlying
    :class:`GradientResult`, with the forward and adjoint simulations and
    their run lengths.
    """

    value: float
    grad: np.ndarray
    params: np.ndarray
    rho: np.ndarray
    jacobian: np.ndarray
    jacobian_method: str
    density_gradient: GradientResult


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
    transmission = isinstance(objective, (ModeTransmission, BroadbandModeTransmission))
    if transmission:
        freqs = (objective.freqs_hz if isinstance(objective, BroadbandModeTransmission)
                 else (objective.freq_hz,))
        expected_monitors = objective.monitors(simulation)
        actual = {m.name: m for m in simulation.monitors}
        for expected in expected_monitors:
            if sum(m.name == expected.name for m in simulation.monitors) > 1:
                raise ValueError(
                    f"monitor name {expected.name!r} is used more than once")
            present = actual.get(expected.name)
            if present is None or (present.center_um != expected.center_um or
                                   present.size_um != expected.size_um or
                                   present.fields != expected.fields or
                                   any(f not in present.freqs_hz for f in freqs)):
                raise ValueError(
                    f"build_forward must include objective monitor {expected.name!r}")
        port_monitors = expected_monitors
    elif isinstance(objective, ModePower):
        port_monitors = [m for m in simulation.monitors if m.name == objective.name]
    else:
        return
    for monitor in port_monitors:
        for a, name in enumerate("xyz"):
            if getattr(simulation.boundaries, name) != "pml":
                continue
            low = (monitor.center_um[a] - 0.5 * monitor.size_um[a]) / dl
            high = (monitor.center_um[a] + 0.5 * monitor.size_um[a]) / dl
            # The current cloud adds up to half a cell of Yee displacement.
            if (low < simulation.pml_num_layers + 0.5 or
                    high >= n_cells[a] - simulation.pml_num_layers - 0.5):
                raise ValueError(
                    f"modal adjoint cloud would enter the {name} PML slab")
    if transmission:
        port = objective.input_port
        axis = "xyz".index(port.axis)
        plane = objective.ports(simulation)[0].monitor.field_monitor.center_um[axis]
        direction = "+" if port.out_direction == "-" else "-"
        source_positions = []
        for source in simulation.sources:
            if isinstance(source, ModeSource):
                if source.axis != port.axis or source.direction != direction:
                    raise ValueError(
                        "input port needs sources launched toward the design along its axis")
                source_positions.append(source.position_um)
            elif isinstance(source, PointDipole):
                source_positions.append(source.center_um[axis])
            else:
                raise ValueError(
                    "input port source position is unavailable for this source type")
        edge = (region.cells[axis][0] + region.box_face_offset_cells) * dl \
            if direction == "+" else \
            (region.cells[axis][1] + region.box_face_offset_cells) * dl
        row_clear = (plane + dl / 4 < edge if direction == "+"
                     else plane - dl / 4 > edge)
        if not row_clear or not source_positions or not all(
                (position < plane < edge if direction == "+"
                 else position > plane > edge) for position in source_positions):
            raise ValueError(
                "input port readout rows must lie between its mode source and the design")
        out = objective.output_port
        out_axis = "xyz".index(out.axis)
        out_plane = objective.ports(simulation)[1].monitor.field_monitor.center_um[out_axis]
        low_edge = (region.cells[out_axis][0] + region.box_face_offset_cells) * dl
        high_edge = (region.cells[out_axis][1] + region.box_face_offset_cells) * dl
        if not (out_plane - dl / 4 > high_edge if out.out_direction == "+"
                else out_plane + dl / 4 < low_edge):
            raise ValueError("output port readout rows must lie outside the design")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

BuildForward = Callable[[np.ndarray], Simulation]
"""User callback: ``rho -> Simulation``. Must place the domain/source and
include ``region.monitor(freq)`` (``region.monitor(objective.freqs_hz)`` for a
:class:`BroadbandModeTransmission`), the objective's monitor or monitors, and
``region.structures(rho)``. Use the ``monitors(simulation)`` method of
:class:`ModeTransmission` or :class:`BroadbandModeTransmission` for the input
and output planes."""


Objective = Union[PointIntensity, ModePower, ModeTransmission, BroadbandModeTransmission]
"""An objective with a transpose-matched adjoint source: one of the
single-frequency objectives, or :class:`BroadbandModeTransmission`."""


def _validate_design_structures(simulation: Simulation, region: DesignRegion,
                                rho: Optional[np.ndarray] = None) -> None:
    """Validate trailing design geometry and media for manual and driver paths."""
    if region.parametrization == "trilinear":
        expected_geometry = region.structures(np.zeros(region.shape))[0].geometry
        if not simulation.structures:
            raise ValueError("build_forward must place region.structures(rho) last")
        actual = simulation.structures[-1]
        medium = actual.medium
        data = medium.permittivity_data
        expected_shape = tuple(n + 1 for n in region.shape)
        if actual.geometry != expected_geometry or data is None or data.shape != expected_shape:
            raise ValueError(
                "build_forward must place region.structures(rho) last so the "
                "trilinear density map matches the adjoint derivative")
        values = np.asarray(data.values, dtype=float).reshape(data.shape)
        if (not np.all(np.isfinite(values)) or
                np.any(values < region.eps_min - 1e-10) or
                np.any(values > region.eps_max + 1e-10) or
                medium != Medium.from_eps_array(values) or
                (rho is not None and not np.allclose(
                    values,
                    np.asarray(region.structures(rho)[0].medium.permittivity_data.values,
                               dtype=float).reshape(expected_shape),
                    rtol=0, atol=1e-10))):
            raise ValueError(
                "build_forward must place region.structures(rho) last so the "
                "trilinear density map matches the adjoint derivative")
    elif region.merge_boxes:
        _validate_merged_boxes(simulation, region, rho)
    else:
        expected = region.box_structures(np.zeros(region.shape))
        actuals = simulation.structures[-region.n_params:]
        if len(actuals) != region.n_params:
            raise ValueError(
                "build_forward must place region.structures(rho) last so the "
                "pixel boxes match the adjoint derivative")
        rho_media = region.box_structures(rho) if rho is not None else None
        for i, (actual, sample) in enumerate(zip(actuals, expected)):
            permittivity = actual.medium.permittivity
            if (actual.geometry != sample.geometry or
                    actual.medium != Medium(permittivity=float(permittivity)) or
                    not region.eps_min - 1e-10 <= permittivity <= region.eps_max + 1e-10 or
                    (rho_media is not None and actual.medium != rho_media[i].medium)):
                raise ValueError(
                    "build_forward must place region.structures(rho) last so the "
                    "pixel boxes match the adjoint derivative")


def _validate_merged_boxes(simulation: Simulation, region: DesignRegion,
                           rho: Optional[np.ndarray]) -> None:
    """The ``merge_boxes`` layout must close the structure list.

    With ``rho`` it must equal ``region.structures(rho)``. Without it the
    last box at ``eps(0)`` over the whole region starts the layout, and every
    later structure must be a box on whole pixels inside the region with a
    scalar permittivity in ``[eps_min, eps_max]``; merged boxes never
    overlap, so no two of those may share a pixel."""
    message = ("build_forward must place region.structures(rho) last so the "
               "pixel boxes match the adjoint derivative")
    structures = tuple(simulation.structures)
    if rho is not None:
        expected = region.box_structures(rho)
        if len(structures) < len(expected) or structures[-len(expected):] != expected:
            raise ValueError(message)
        return
    base = region.merged_box_structures(np.zeros(region.shape))[0]
    starts = [i for i, s in enumerate(structures) if s == base]
    if not starts:
        raise ValueError(message)
    covered = np.zeros(region.shape, dtype=np.int64)
    for actual in structures[starts[-1] + 1:]:
        permittivity = actual.medium.permittivity
        if (not isinstance(actual.geometry, Box) or permittivity is None or
                actual.medium != Medium(permittivity=float(permittivity)) or
                not region.eps_min - 1e-10 <= permittivity <= region.eps_max + 1e-10):
            raise ValueError(message)
        span = []
        for a in range(3):
            c, s = actual.geometry.center_um[a], actual.geometry.size_um[a]
            faces = np.array([c - s / 2, c + s / 2]) / region.dl_um
            pixel = (faces - region.cells[a][0] - region.box_face_offset_cells) \
                / region.cells_per_pixel[a]
            if (np.any(np.abs(pixel - np.round(pixel)) > 1e-6) or pixel[0] < -1e-6
                    or pixel[1] > region.shape[a] + 1e-6 or pixel[1] - pixel[0] < 1 - 1e-6):
                raise ValueError(message)
            span.append(slice(*np.round(pixel).astype(int)))
        covered[tuple(span)] += 1
    if covered.max(initial=0) > 1:
        raise ValueError(message)


def _adjoint_simulation_parts(
    forward_simulation: Simulation,
    forward: RunResult,
    region: DesignRegion,
    objective: Objective,
    *,
    monitor_name: str = "design_region",
    _validated: bool = False,
):
    """Shared adjoint specification, readout weight, and physical factor."""
    if isinstance(objective, BroadbandModeTransmission):
        raise ValueError("a broadband objective's adjoint is built by "
                         "value_and_gradient; it has no single-frequency manual path")
    if forward_simulation.subpixel:
        raise ValueError("manual adjoint requires a subpixel-off forward simulation")
    if any(forward_simulation.symmetry) or any(forward_simulation.origin_um):
        raise ValueError("manual adjoint requires an explicit, unsymmetrized domain")
    if not isinstance(forward_simulation.grid, UniformMesh) or (
            region.dl_um != forward_simulation.grid.dl_um):
        raise ValueError("manual adjoint requires a matching uniform mesh")
    if not _validated:
        _validate_design_structures(forward_simulation, region)
        _validate_pml_clear(forward_simulation, region, objective)
    if isinstance(objective, (ModePower, ModeTransmission)):
        adj_sources, coeff = objective.transpose_sources(forward_simulation, forward)
    else:
        adj_sources = (objective.adjoint_source(forward_simulation),)
        coeff = objective.adjoint_coeff(forward)
    adj_sim = forward_simulation.model_copy(update={
        "sources": adj_sources,
        "monitors": (region.monitor(objective.freq_hz, monitor_name,
                                    simulation=forward_simulation),),
    })
    derived = -2j * _discrete_omega(forward_simulation, objective.freq_hz) * _EPS0
    if isinstance(objective, PointIntensity) and objective.component.startswith("H"):
        derived *= -np.exp(1j * np.pi * objective.freq_hz
                           * _time_step(forward_simulation))
    return adj_sim, coeff, derived


def build_adjoint_simulation(
    forward_simulation: Simulation,
    forward: RunResult,
    region: DesignRegion,
    objective: Objective,
    *,
    monitor_name: str = "design_region",
) -> Tuple[Simulation, complex]:
    """Return the runnable adjoint simulation and its gradient coefficient.

    Run the returned simulation with :func:`run_local`, then pass both results
    and the returned complete coefficient to :func:`assemble_gradient`
    with its default ``beta``. The coefficient already includes the full
    discrete physical factor.
    The forward simulation must use subpixel off.
    :func:`value_and_gradient` uses these same pieces automatically.
    """
    sim, readout, physical = _adjoint_simulation_parts(
        forward_simulation, forward, region, objective,
        monitor_name=monitor_name)
    return sim, complex(physical * readout)


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

    With a :class:`BroadbandModeTransmission` the forward run reads every
    ``T_k`` and the design fields at every frequency, and ONE adjoint run
    carries the adjoint currents of all frequencies at once (see that class);
    the result is a :class:`BroadbandGradientResult` whose ``value`` is ``F``
    and ``grad`` is ``dF/drho``. The adjoint run is longer than the forward
    run by the span of its delayed pulses, ``adjoint_extra_steps`` in the
    result; the forward run's other settings, its stop rule included, carry
    over. ``beta`` is refused for it. The broadband path is validated on the
    CPU.
    """
    if isinstance(objective, BroadbandModeTransmission) and beta is not None:
        raise ValueError("beta= is not supported for a broadband objective")
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
    _validate_design_structures(fwd_sim, region, np.asarray(rho, dtype=float))
    _validate_pml_clear(fwd_sim, region, objective)
    if fwd_sim.subpixel:
        warnings.warn("inverse-design objective and gradient use the subpixel-off model",
                      UserWarning, stacklevel=caller_stacklevel())
    if isinstance(objective, BroadbandModeTransmission):
        return _broadband_value_and_gradient(
            fwd_sim.model_copy(update={"subpixel": False}), region, objective,
            run_kwargs, monitor_name)
    if beta is not None and _warn_beta:
        warnings.warn("beta= is deprecated; omit it to use the derived discrete adjoint coefficient",
                      DeprecationWarning, stacklevel=caller_stacklevel())
    shutoff = fwd_sim.run.shutoff
    # Both box indicators and custom-medium weights assume smoothing is off.
    fwd_sim = fwd_sim.model_copy(update={"subpixel": False})
    def solve_pair(output_root: Path):
        forward = run_local(fwd_sim, **{**run_kwargs,
                                       "output_dir": output_root / "forward"})
        if forward.aborted:
            raise RuntimeError(f"forward run aborted: {forward.abort_reason}")

        u = objective.amplitude(forward)
        adj_sim, coeff, derived_beta = _adjoint_simulation_parts(
            fwd_sim, forward, region, objective, monitor_name=monitor_name,
            _validated=True)
        if beta is not None:
            derived_beta = beta

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
        return forward, adjoint, adj_sim, u, g, value

    output_dir = run_kwargs.pop("output_dir", None)
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="photonhub-gradient-") as directory:
            forward, adjoint, adj_sim, u, g, value = solve_pair(Path(directory))
    else:
        forward, adjoint, adj_sim, u, g, value = solve_pair(Path(output_dir))
    return GradientResult(value=value, grad=g,
                          forward=forward, adjoint=adjoint, amplitude=u,
                          forward_simulation=fwd_sim,
                          adjoint_simulation=adj_sim,
                          forward_steps_run=forward.steps_run,
                          adjoint_steps_run=adjoint.steps_run,
                          shutoff=shutoff)


def _check_adjoint_window(run, extra_steps: int, dt: float, freqs: Tuple[float, ...]) -> None:
    """Refuse, before any solve, an adjoint window the engine cannot run, and
    warn when it exceeds the forward window many times over. The extension
    grows as one over the smallest frequency spacing, so nearly equal design
    frequencies ask for a very long adjoint run."""
    if extra_steps == 0:
        return
    forward_steps = (run.n_steps if run.n_steps is not None
                     else int(np.ceil(run.run_time_s / dt - 1e-9)))
    spacing = float(np.min(np.diff(np.sort(np.asarray(freqs, dtype=float)))))
    where = (f"the closest design frequencies are {spacing:.4g} Hz apart, so the "
             f"adjoint run takes {extra_steps} steps more than the forward run's "
             f"{forward_steps}")
    if forward_steps + extra_steps > MAX_INT32:
        raise ValueError(f"{where}, beyond the solver's limit of {MAX_INT32} steps; "
                         "space the design frequencies further apart")
    if extra_steps > _SPAN_WARN_RATIO * forward_steps:
        warnings.warn(f"{where}; its cost grows as one over the smallest frequency "
                      "spacing", UserWarning, stacklevel=caller_stacklevel())


def _extended_run(run, extra_steps: int, dt: float):
    """The forward run settings with ``extra_steps`` more steps of window."""
    if extra_steps == 0:
        return run
    if run.n_steps is not None:
        return run.model_copy(update={"n_steps": run.n_steps + extra_steps})
    return run.model_copy(update={"run_time_s": run.run_time_s + extra_steps * dt})


def _broadband_value_and_gradient(fwd_sim: Simulation, region: DesignRegion,
                                  objective: BroadbandModeTransmission,
                                  run_kwargs: dict, monitor_name: str
                                  ) -> BroadbandGradientResult:
    """One forward and one adjoint solve for a broadband objective.

    ``fwd_sim`` is the validated, subpixel-off forward simulation. The
    adjoint drives every port sample with the tap currents of
    :func:`_tap_sources` and records the forward run's own design monitor;
    each frequency's adjoint field is restored from the engine's
    normalization by ``S_first(f_k) g / R_k`` and contracted with the
    forward field and the leapfrog factor at ``f_k``."""
    freqs = objective.freqs_hz
    design = [m for m in fwd_sim.monitors if m.name == monitor_name]
    if (len(design) != 1 or set(design[0].fields) != {"Ex", "Ey", "Ez"}
            or any(f not in design[0].freqs_hz for f in freqs)):
        raise ValueError(
            f"build_forward must include region.monitor(objective.freqs_hz) named "
            f"{monitor_name!r}, recording Ex, Ey and Ez at every objective frequency")
    plan = _tap_plan(freqs, _band_pulse(fwd_sim, freqs))
    dt = _time_step(fwd_sim)
    extra_steps = int(np.ceil(plan.span_s / dt - 1e-9)) if plan.span_s > 0 else 0
    _check_adjoint_window(fwd_sim.run, extra_steps, dt, freqs)

    def solve_pair(output_root: Path):
        forward = run_local(fwd_sim, **{**run_kwargs,
                                       "output_dir": output_root / "forward"})
        if forward.aborted:
            raise RuntimeError(f"forward run aborted: {forward.abort_reason}")
        targets, readout = objective._adjoint_targets(fwd_sim, forward)
        sources, scale, first = _tap_sources(targets, plan)
        adj_sim = fwd_sim.model_copy(update={
            "sources": sources, "monitors": (design[0],),
            "run": _extended_run(fwd_sim.run, extra_steps, dt)})
        adjoint = run_local(adj_sim, **{**run_kwargs,
                                       "output_dir": output_root / "adjoint"})
        if adjoint.aborted:
            raise RuntimeError(f"adjoint run aborted: {adjoint.abort_reason}")
        normalization = _pulse_spectrum(first, freqs) * scale / plan.reference
        parts = np.stack([
            assemble_gradient(region, forward, adjoint, complex(normalization[k]), f,
                              monitor_name=monitor_name,
                              beta=-2j * _discrete_omega(fwd_sim, f) * _EPS0)
            for k, f in enumerate(freqs)])
        for result in (forward, adjoint):
            for name in result.monitor_names:
                result._raw(name)
        return forward, adjoint, adj_sim, readout, parts, len(sources)

    run_kwargs = dict(run_kwargs)
    output_dir = run_kwargs.pop("output_dir", None)
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="photonhub-gradient-") as directory:
            forward, adjoint, adj_sim, readout, parts, dipoles = solve_pair(Path(directory))
    else:
        forward, adjoint, adj_sim, readout, parts, dipoles = solve_pair(Path(output_dir))
    den = objective._denominators(readout["a"])
    ratio = readout["b"] / den
    return BroadbandGradientResult(
        value=readout["value"], grad=np.sum(parts, axis=0),
        forward=forward, adjoint=adjoint, amplitude=ratio,
        forward_simulation=fwd_sim, adjoint_simulation=adj_sim,
        forward_steps_run=forward.steps_run, adjoint_steps_run=adjoint.steps_run,
        shutoff=fwd_sim.run.shutoff, freqs_hz=freqs,
        transmissions=readout["transmissions"], amplitudes=ratio,
        dF_dT=readout["slope"], gradients_by_freq=parts, taps=plan.taps,
        tap_span_s=plan.span_s, tap_condition=plan.condition,
        adjoint_extra_steps=extra_steps, adjoint_dipoles=dipoles)


def finite_difference_check(
    build_forward: BuildForward,
    region: DesignRegion,
    objective: Objective,
    rho: np.ndarray,
    h: float,
    *,
    indices: Optional[Tuple[int, ...]] = None,
    device: Optional[str] = None,
    run_kwargs: Optional[dict] = None,
) -> FiniteDifferenceResult:
    """Compare the discrete adjoint with central differences of real solves.

    Each checked control takes two additional forward simulations, at
    ``rho[i] + h`` and ``rho[i] - h``. The objective value, every plus and
    minus value and every solve length are recorded. Keep the density
    perturbations inside the physical [0, 1] range.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")
    flat = np.asarray(rho, dtype=float).ravel()
    if flat.size != region.n_params:
        raise ValueError("rho shape does not match DesignRegion")
    selected = tuple(range(flat.size)) if indices is None else tuple(indices)
    if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= flat.size
                                                    for i in selected):
        raise ValueError("indices must be nonempty unique valid flat control indices")
    if any(flat[i] - h < 0 or flat[i] + h > 1 for i in selected):
        raise ValueError("finite-difference perturbation leaves density range [0, 1]")
    options = _fd_options(device, run_kwargs)
    adj = value_and_gradient(build_forward, region, objective,
                             flat.reshape(region.shape), device=device,
                             run_kwargs=options)

    def render(i, sign):
        perturbed = flat.copy()
        perturbed[i] += sign * h
        return perturbed.reshape(region.shape)

    fd, lengths, values = _central_differences(
        build_forward, objective, selected, h, render, options)
    g = np.asarray(adj.grad).ravel()[list(selected)]
    return _fd_result(selected, h, g, fd, lengths, values, adj)


def _fd_options(device: Optional[str], run_kwargs: Optional[dict]) -> dict:
    options = dict(run_kwargs or {})
    options.setdefault("quiet", True)
    options.pop("output_dir", None)
    if device is not None:
        options["device"] = device
    return options


def _central_differences(build_forward, objective, selected, h, render, options):
    """Two forward solves per selected control at ``render(i, +1)`` and
    ``render(i, -1)``; returns the differences, run lengths and values."""
    fd = np.empty(len(selected), dtype=float)
    lengths = []
    values = []
    with tempfile.TemporaryDirectory(prefix="photonhub-fd-") as directory:
        for j, i in enumerate(selected):
            pair = []
            steps = []
            for sign, label in ((1, "plus"), (-1, "minus")):
                sim = build_forward(render(i, sign))
                sim = sim.model_copy(update={"subpixel": False})
                result = run_local(
                    sim, **{**options, "output_dir": Path(directory) / f"{i}-{label}"})
                if result.aborted:
                    raise RuntimeError(f"finite-difference run aborted: {result.abort_reason}")
                pair.append(objective.value(result))
                steps.append(result.steps_run)
            fd[j] = (pair[0] - pair[1]) / (2 * h)
            lengths.append(tuple(steps))
            values.append(tuple(pair))
    return fd, lengths, values


def _fd_result(selected, h, g, fd, lengths, values, adj, jacobian=None):
    gnorm, fnorm = np.linalg.norm(g), np.linalg.norm(fd)
    cosine = float(np.dot(g, fd) / (gnorm * fnorm)) if gnorm and fnorm else float("nan")
    ratio = float(gnorm / fnorm) if fnorm else float("nan")
    rel = float(np.linalg.norm(g - fd) / fnorm) if fnorm else float("nan")
    return FiniteDifferenceResult(
        indices=selected, h=h, adjoint_gradient=g,
        finite_difference_gradient=fd, cosine=cosine,
        norm_ratio=ratio, relative_l2=rel,
        forward_steps_run=adj.forward_steps_run,
        adjoint_steps_run=adj.adjoint_steps_run,
        finite_difference_steps_run=tuple(lengths),
        value=adj.value, finite_difference_values=tuple(values),
        jacobian=jacobian)


# ---------------------------------------------------------------------------
# Shape-parameter gradients
# ---------------------------------------------------------------------------

def _parameter_vector(p) -> np.ndarray:
    params = np.array(p, dtype=float)
    if params.ndim != 1 or params.size == 0 or not np.all(np.isfinite(params)):
        raise ValueError("p must be a nonempty, finite, one-dimensional parameter vector")
    return params


def _render(region: DesignRegion, expand, params: np.ndarray) -> np.ndarray:
    rho = np.asarray(expand(params.copy()), dtype=float)
    if rho.size != region.n_params:
        raise ValueError(f"expand(p) returned {rho.size} densities; the region has "
                         f"{region.n_params}")
    rho = rho.reshape(region.shape)
    if not np.all(np.isfinite(rho)) or np.any(rho < -1e-10) or np.any(rho > 1 + 1e-10):
        raise ValueError("expand(p) must return densities in [0, 1]")
    return rho


def _parameter_jacobian(
    region: DesignRegion,
    expand: Callable[[np.ndarray], np.ndarray],
    p: np.ndarray,
    *,
    jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fd_step: float = 1e-3,
    check: bool = True,
) -> Tuple[np.ndarray, str]:
    """Return ``drho/dp`` with shape ``(region.n_params, len(p))`` and how it
    was obtained. No solver runs.

    With ``jacobian`` the caller's derivative is used; it may return
    ``(region.n_params, len(p))`` or ``region.shape + (len(p),)``. Without
    it, column ``j`` is the central difference
    ``(expand(p + fd_step e_j) - expand(p - fd_step e_j)) / (2 fd_step)``, in
    the parameter's own unit. See :func:`parametric_value_and_gradient` for
    its accuracy, including at kinks of ``expand``, and for the two warnings
    that renders of ``expand`` at smaller steps add; ``check=False`` skips
    those renders.
    """
    params = _parameter_vector(p)
    n = params.size
    if not np.isfinite(fd_step) or fd_step <= 0:
        raise ValueError("fd_step must be positive and finite")
    if jacobian is not None:
        raw = jacobian(params.copy())
        if _sparse.issparse(raw):
            matrix = _sparse.csr_matrix(raw, dtype=float)
            if matrix.shape != (region.n_params, n):
                raise ValueError(f"a sparse jacobian(p) must have shape ({region.n_params}, {n}); "
                                 f"got {matrix.shape}")
            if not np.all(np.isfinite(matrix.data)):
                raise ValueError("parameter Jacobian is not finite")
            if check:
                _spot_check_sparse_jacobian(expand, params, fd_step, matrix)
            return matrix, "analytic, supplied by the caller"
        matrix = np.asarray(raw, dtype=float)
        if matrix.shape == tuple(region.shape) + (n,):
            matrix = matrix.reshape(region.n_params, n)
        if matrix.shape != (region.n_params, n):
            raise ValueError(f"jacobian(p) must have shape ({region.n_params}, {n}) or "
                             f"{tuple(region.shape) + (n,)}; got {matrix.shape}")
        method = "analytic, supplied by the caller"
        if check and np.all(np.isfinite(matrix)):
            _spot_check_jacobian(expand, params, fd_step, matrix)
    else:
        matrix, kinked = _expand_central_difference(expand, params, fd_step, check)
        method = f"central difference of expand, fd_step={fd_step:g}"
        if kinked.size:
            warnings.warn(
                f"the central difference of expand at fd_step={fd_step:g} is not a "
                f"derivative for parameter(s) {kinked.tolist()}: its forward and "
                "backward differences disagree by more than a smooth rendering "
                "allows (O(fd_step)), so expand has a kink (np.clip, np.maximum) "
                "or varies on a scale near fd_step within fd_step of p, and the "
                "column is a blend of two one-sided slopes. Supply jacobian= or a "
                "smooth rendering, or keep p away from the kink",
                UserWarning, stacklevel=caller_stacklevel())
    if not np.all(np.isfinite(matrix)):
        raise ValueError("parameter Jacobian is not finite")
    return matrix, method


# The no-solve Jacobian checks warn above this fraction of the largest
# one-sided slope of ``expand``.
_JACOBIAN_CHECK_TOLERANCE = 1e-2
# Samples per ``fd_step`` along each parameter for the kink check, and how
# many times larger than the slope changes two samples away a change must be
# to count as concentrated at one point.
_KINK_SAMPLES = 8
_KINK_CONCENTRATION = 3.0
# Densities judged together by the kink check; its slope arrays scale with
# this, not with the region.
_KINK_BLOCK = 2048


def _render_flat(expand, q: np.ndarray) -> np.ndarray:
    return np.asarray(expand(q), dtype=float).ravel()


class _ExpandSamples:
    """Renders of ``expand`` along one parameter at a time, and the kink check.

    :meth:`along` renders ``expand`` along parameter ``j`` at multiples of
    ``w = fd_step/8`` out to ``1.25 fd_step``, into one buffer that the next
    call overwrites, and judges them :data:`_KINK_BLOCK` densities at a
    time, so the checks' memory is the ``21`` renders of one column plus the
    matrices the caller keeps, not ``21`` renders of every column.

    A smooth rendering's slope changes gradually between neighbouring
    intervals, even at the cubic onset of a quintic edge, while a kink puts
    its whole slope jump at one sample point. A density is flagged when, at
    a point within ``fd_step`` of ``p``, the slope changes by more than
    :data:`_JACOBIAN_CHECK_TOLERANCE` of the largest slope along any
    parameter and by more than :data:`_KINK_CONCENTRATION` times the changes
    two samples to either side. The forward and backward differences at
    ``fd_step`` then disagree by more than ``O(fd_step)``. The largest slope
    is known only after the last parameter, so each column keeps its
    candidates above the largest slope seen so far, a bar that only rises,
    and :meth:`kinks` judges them against the final one.
    """

    reach = _KINK_SAMPLES + 2

    def __init__(self, expand, params: np.ndarray, fd_step: float):
        self.expand = expand
        self.params = params
        self.w = fd_step / _KINK_SAMPLES
        center = _render_flat(expand, params.copy())
        self.samples = np.empty((2 * self.reach + 1, center.size))
        self.samples[self.reach] = center
        self.scale = 0.0
        self._candidates: List[Tuple[int, np.ndarray, np.ndarray]] = []

    def at(self, k: int) -> np.ndarray:
        """The render at offset ``k w`` along the last parameter rendered."""
        return self.samples[self.reach + k]

    def along(self, j: int, known: Optional[dict] = None) -> None:
        """Render along parameter ``j``, reusing ``known`` renders by offset,
        and judge its kinks."""
        known = known or {}
        for k in range(-self.reach, self.reach + 1):
            if k == 0:
                continue
            if k in known:
                self.samples[self.reach + k] = known[k]
                continue
            q = self.params.copy()
            q[j] += k * self.w
            self.samples[self.reach + k] = _render_flat(self.expand, q)
        for start in range(0, self.samples.shape[1], _KINK_BLOCK):
            self._judge(j, start, self.samples[:, start:start + _KINK_BLOCK])

    def _judge(self, j: int, start: int, block: np.ndarray) -> None:
        slopes = np.diff(block, axis=0) / self.w
        if slopes.size:
            # np.max keeps a NaN slope sticky, so no density is flagged.
            self.scale = float(np.max([self.scale, np.max(np.abs(slopes))]))
        # change[i] is the slope change at offset (i - reach + 1) * w; the
        # points judged are offsets -(n-1)..(n-1), each against i - 2 and i + 2.
        change = np.abs(np.diff(slopes, axis=0))
        middle = change[2:-2]
        neighbours = np.maximum(change[:-4], change[4:])
        concentrated = middle > _KINK_CONCENTRATION * neighbours
        strength = np.max(np.where(concentrated, middle, 0.0), axis=0)
        rows = np.flatnonzero(strength > _JACOBIAN_CHECK_TOLERANCE * self.scale)
        if rows.size:
            self._candidates.append((j, start + rows, strength[rows]))

    def kinks(self, shape: Tuple[int, int]) -> np.ndarray:
        """Kink flags, ``shape`` = (densities, parameters)."""
        flags = np.zeros(shape, dtype=bool)
        bar = _JACOBIAN_CHECK_TOLERANCE * self.scale
        for j, rows, strength in self._candidates:
            flags[rows[strength > bar], j] = True
        return flags


def _expand_central_difference(expand, params: np.ndarray, fd_step: float,
                               check: bool) -> Tuple[np.ndarray, np.ndarray]:
    """The default Jacobian, and the parameters the kink check flags.

    Column ``j`` needs only the renders at ``p +/- fd_step e_j``; one that
    fails fails the call. With ``check`` the other renders of
    :class:`_ExpandSamples` follow under a ``try``, and a failure among them
    skips the kink check rather than failing a call that needs no such
    render."""
    samples = None
    if check:
        try:
            samples = _ExpandSamples(expand, params, fd_step)
        except Exception:  # noqa: BLE001 - the check must not fail a valid call
            samples = None
    n = _KINK_SAMPLES
    matrix = None
    for j in range(params.size):
        ends = {}
        for sign in (1, -1):
            q = params.copy()
            q[j] += sign * fd_step
            ends[sign * n] = _render_flat(expand, q)
        column = (ends[n] - ends[-n]) / (2.0 * fd_step)
        if matrix is None:
            matrix = np.empty((column.size, params.size))
        matrix[:, j] = column
        if samples is not None:
            try:
                samples.along(j, known=ends)
            except Exception:  # noqa: BLE001 - as above
                samples = None
    if samples is None:
        return matrix, np.zeros(0, dtype=int)
    return matrix, np.flatnonzero(np.any(samples.kinks(matrix.shape), axis=0))


def _spot_check_jacobian(expand, params: np.ndarray, fd_step: float,
                         matrix: np.ndarray) -> None:
    """Warn when a caller's Jacobian disagrees with differences of ``expand``.

    The reference is the central difference at ``fd_step/8``, allowed twice
    its change from ``fd_step/4`` as truncation error plus
    :data:`_JACOBIAN_CHECK_TOLERANCE` of the largest slope. Densities flagged
    as kinks are skipped, since a difference there is no reference. No solver
    runs; a failure to render near ``p`` skips the check rather than failing
    a call that needs no such render. Besides the caller's matrix it keeps
    the two difference matrices and one parameter's renders."""
    try:
        samples = _ExpandSamples(expand, params, fd_step)
        fine = np.empty_like(matrix)
        coarse = np.empty_like(matrix)
        for j in range(params.size):
            samples.along(j)
            fine[:, j] = (samples.at(1) - samples.at(-1)) / (2.0 * samples.w)
            coarse[:, j] = (samples.at(2) - samples.at(-2)) / (4.0 * samples.w)
    except Exception:  # noqa: BLE001 - the check must not fail a valid call
        return
    if not np.all(np.isfinite(fine)):
        return
    # max |matrix| without an |matrix| copy; the matrix is finite here.
    scale = max(samples.scale, float(max(np.max(matrix), -np.min(matrix))))
    if scale == 0:
        return
    kink = samples.kinks(matrix.shape)
    wrong, worst = [], 0.0
    for j in range(params.size):
        allowed = (_JACOBIAN_CHECK_TOLERANCE * scale
                   + 2.0 * np.abs(coarse[:, j] - fine[:, j]))
        mismatch = np.where(kink[:, j], 0.0, np.abs(matrix[:, j] - fine[:, j]))
        if np.any(mismatch > allowed):
            wrong.append(j)
        worst = max(worst, float(np.max(mismatch)))
    if wrong:
        warnings.warn(
            f"jacobian(p) disagrees with differences of expand(p) for parameter(s) "
            f"{wrong}, by up to {worst / scale:.2g} of the "
            "largest slope; check its units, sign and flattening order "
            "(region.n_params rows in expand(p).ravel() order) before paying for solves",
            UserWarning, stacklevel=caller_stacklevel())


def _spot_check_sparse_jacobian(expand, params: np.ndarray, fd_step: float,
                                matrix) -> None:
    """:func:`_spot_check_jacobian` for a sparse Jacobian, one column at a time.

    The same reference, allowance and kink exclusion, but no dense
    ``(densities, parameters)`` array: each column is compared as it is
    rendered and only the densities above the rising bar are kept, as the
    kink check keeps its candidates. Besides the caller's matrix it holds
    the renders of one parameter and a few dense columns."""
    try:
        samples = _ExpandSamples(expand, params, fd_step)
        scale = float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
        csc = matrix.tocsc()
        excess = []
        for j in range(params.size):
            samples.along(j)
            fine = (samples.at(1) - samples.at(-1)) / (2.0 * samples.w)
            coarse = (samples.at(2) - samples.at(-2)) / (4.0 * samples.w)
            if not np.all(np.isfinite(fine)):
                return
            column = csc[:, j].toarray().ravel()
            over = np.abs(column - fine) - 2.0 * np.abs(coarse - fine)
            bar = _JACOBIAN_CHECK_TOLERANCE * max(scale, samples.scale)
            rows = np.flatnonzero(over > bar)
            if rows.size:
                excess.append((j, rows, over[rows], np.abs(column - fine)[rows]))
    except Exception:  # noqa: BLE001 - the check must not fail a valid call
        return
    scale = max(scale, samples.scale)
    if scale == 0:
        return
    bar = _JACOBIAN_CHECK_TOLERANCE * scale
    kinked = {}
    for j, rows, strength in samples._candidates:
        kinked.setdefault(j, set()).update(rows[strength > bar].tolist())
    wrong, worst = [], 0.0
    for j, rows, over, mismatch in excess:
        flagged = np.array([r not in kinked.get(j, ()) and o > bar
                            for r, o in zip(rows, over)], dtype=bool)
        if np.any(flagged):
            wrong.append(j)
            worst = max(worst, float(np.max(mismatch[flagged])))
    if wrong:
        warnings.warn(
            f"jacobian(p) disagrees with differences of expand(p) for parameter(s) "
            f"{wrong}, by up to {worst / scale:.2g} of the "
            "largest slope; check its units, sign and flattening order "
            "(region.n_params rows in expand(p).ravel() order) before paying for solves",
            UserWarning, stacklevel=caller_stacklevel())


def _parametric(build_forward, region, objective, p, expand, *, jacobian, fd_step,
                device, run_kwargs, monitor_name, beta=None, check_jacobian=True
                ) -> ParametricGradientResult:
    params = _parameter_vector(p)
    rho = _render(region, expand, params)
    matrix, method = _parameter_jacobian(region, expand, params,
                                        jacobian=jacobian, fd_step=fd_step,
                                        check=check_jacobian)
    density = value_and_gradient(build_forward, region, objective, rho,
                                 device=device, run_kwargs=run_kwargs, beta=beta,
                                 monitor_name=monitor_name, _warn_beta=False)
    grad = np.asarray(matrix.T @ np.asarray(density.grad, dtype=float).ravel(),
                      dtype=float).ravel()
    return ParametricGradientResult(
        value=density.value, grad=grad, params=params, rho=rho,
        jacobian=matrix, jacobian_method=method, density_gradient=density)


def parametric_value_and_gradient(
    build_forward: BuildForward,
    region: DesignRegion,
    objective: Objective,
    p: np.ndarray,
    expand: Callable[[np.ndarray], np.ndarray],
    *,
    jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fd_step: float = 1e-3,
    device: Optional[str] = None,
    run_kwargs: Optional[dict] = None,
    monitor_name: str = "design_region",
    check_jacobian: bool = True,
) -> ParametricGradientResult:
    """Return the objective and ``dJ/dp`` for shape parameters ``p``.

    ``expand(p)`` renders the density grid (``region.shape``, values in
    ``[0, 1]``). One forward and one adjoint solve give ``dJ/drho`` as in
    :func:`value_and_gradient`, and the chain rule gives
    ``dJ/dp = (drho/dp)^T dJ/drho`` without further solves. The cost is two
    solves whatever the number of parameters.

    ``jacobian(p)`` supplies ``drho/dp`` exactly, as
    :meth:`WidthProfile.jacobian` does, with shape
    ``(region.n_params, len(p))`` in flat density order or
    ``region.shape + (len(p),)``. It may be a ``scipy.sparse`` matrix of
    the first shape, as :meth:`PolygonProfile.jacobian` returns; it is then
    chained and returned sparse, never densified. Without it, column ``j`` of ``drho/dp`` is
    ``(expand(p + fd_step e_j) - expand(p - fd_step e_j)) / (2 fd_step)``,
    with ``fd_step`` in the parameter's own unit. Two cases:

    - A smooth ``expand``: the difference is accurate to second order in
      ``fd_step``, and the error scales with ``(fd_step / edge)^2`` for an
      edge of width ``edge``. On the two 3D shape-validation scenes, rendered
      with :class:`WidthProfile` on a 0.044 um mesh with its default
      two-cell edge, the default 1e-3 um step moved ``dJ/dp`` by 1.5e-5 and
      2.3e-4 relative to the exact Jacobian, within the resolution of
      central differences of solves there (1.2e-4 and 6.4e-4).
    - A piecewise-linear ``expand`` (``np.clip``, ``np.interp`` in the
      parameters, ``np.maximum`` of two renders): away from a kink the
      difference is exact up to rounding, but for every density whose kink
      lies within ``fd_step`` of ``p`` it returns a weighted mean of the two
      one-sided slopes (their plain mean at the kink), which is the slope on
      neither side. At the kink the objective has no gradient, and an
      optimizer stepping either way sees a different slope from the one
      reported. Prefer a smooth rendering, keep ``p`` away from kinks, or
      supply ``jacobian``.

    Two checks render ``expand`` again before any solve, at ``p`` and along
    every parameter at multiples of ``fd_step/8`` out to ``1.25 fd_step``,
    and warn without changing the result. Without ``jacobian``, a kink
    within ``fd_step`` of ``p`` makes the forward and backward differences
    disagree by more than the ``O(fd_step)`` a smooth rendering allows; it
    shows as a slope change concentrated at one sample, which a twice
    continuously differentiable ``expand`` cannot produce. The warning names
    the parameters. With ``jacobian``, the supplied matrix is compared with
    the central difference at ``fd_step/8``, away from such kinks and
    allowing for its truncation error, which catches a wrong unit, sign or
    flattening order before the ``2N + 2`` solves of a finite-difference
    check. Both warn above 1% of the largest one-sided slope of ``expand``.
    Neither warns on :class:`WidthProfile` renderings, including widths
    whose edge pixels sit at the ends of their soft edge. The checks render
    one parameter at a time, so their memory stays within a few times the
    Jacobian's; for a sparse ``jacobian`` the comparison also runs one
    column at a time and keeps about 30 renders of ``expand`` whatever the
    number of parameters (about 67 MB and 21 s for the 400 vertex
    displacements of a 736 x 384 pixel :class:`PolygonProfile` on a laptop
    CPU). ``check_jacobian=False`` skips them, and a render that fails
    beyond ``p +/- fd_step`` skips them too.

    Check the result against real solves with
    :func:`parametric_finite_difference_check`. Run settings, the subpixel
    model, the stop rule and the refusals are those of
    :func:`value_and_gradient`.
    """
    return _parametric(build_forward, region, objective, p, expand,
                       jacobian=jacobian, fd_step=fd_step, device=device,
                       run_kwargs=run_kwargs, monitor_name=monitor_name,
                       check_jacobian=check_jacobian)


def parametric_finite_difference_check(
    build_forward: BuildForward,
    region: DesignRegion,
    objective: Objective,
    p: np.ndarray,
    expand: Callable[[np.ndarray], np.ndarray],
    h: float,
    *,
    jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fd_step: float = 1e-3,
    indices: Optional[Tuple[int, ...]] = None,
    device: Optional[str] = None,
    run_kwargs: Optional[dict] = None,
    check_jacobian: bool = True,
) -> FiniteDifferenceResult:
    """Compare the chained parameter gradient with central differences of
    real solves.

    Each checked parameter takes two forward simulations, at ``p[i] + h``
    and ``p[i] - h``, each re-rendered with ``expand``; ``h`` is in the
    parameter's own unit (microns for widths). The adjoint side is
    :func:`parametric_value_and_gradient` with the same ``jacobian``,
    ``fd_step`` and ``check_jacobian``. The objective value, every plus and minus value, every
    solve length and the Jacobian used are recorded. Keep every rendered
    density in ``[0, 1]`` and keep ``p +/- h`` clear of kinks in ``expand``,
    where central differences of the objective are not a derivative.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")
    params = _parameter_vector(p)
    selected = tuple(range(params.size)) if indices is None else tuple(indices)
    if not selected or len(set(selected)) != len(selected) or any(
            i < 0 or i >= params.size for i in selected):
        raise ValueError("indices must be nonempty unique valid parameter indices")
    legs = {}
    for i in selected:
        for sign in (1, -1):
            perturbed = params.copy()
            perturbed[i] += sign * h
            legs[(i, sign)] = _render(region, expand, perturbed)
    options = _fd_options(device, run_kwargs)
    adj = _parametric(build_forward, region, objective, params, expand,
                      jacobian=jacobian, fd_step=fd_step, device=device,
                      run_kwargs=options, monitor_name="design_region",
                      check_jacobian=check_jacobian)
    fd, lengths, values = _central_differences(
        build_forward, objective, selected, h,
        lambda i, sign: legs[(i, sign)], options)
    return _fd_result(selected, h, adj.grad[list(selected)], fd, lengths, values,
                      adj.density_gradient, jacobian=adj.jacobian)


@dataclass
class OptimizeResult:
    """Optimization history and best value for the subpixel-off model.

    ``history``, ``grads``, and ``best`` all refer to that same model even if
    the caller's constructed simulation enabled subpixel smoothing.
    """
    rho: np.ndarray              # best design's density grid (region.shape)
    history: List[float]         # objective per evaluation
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
    gradient magnitude. The adjoint gradient is the exact derivative of the
    evaluated objective, so SciPy gets ``-sign * J / |J0|`` and
    ``-sign * dJ/dx / |J0|`` unchanged otherwise: a consistent pair of order
    one. ``J0`` is SciPy's own first evaluation, at the start, so every
    evaluation is one SciPy asked for and none is spent on scaling.
    ``method="adam"`` keeps the scale-free normalized-gradient Adam (``step`` =
    max per-variable change per iteration); only Adam uses ``step``."""
    sign = 1.0 if maximize else -1.0
    x0 = np.asarray(x0, dtype=float).ravel().copy()
    if method == "lbfgs":
        from scipy.optimize import minimize

        bnds = None if bounds is None else [(bounds[0], bounds[1])] * x0.size
        scale = []                                    # |J0|, set by the first evaluation

        def neg(x):
            J, gJ = fg(x)
            if not scale:                             # SciPy evaluates the start first
                absJ0 = abs(float(J))
                scale.append(absJ0 if np.isfinite(absJ0) and absJ0 > 0 else 1.0)
            return (float(-sign * J / scale[0]),
                    -sign * np.asarray(gJ, dtype=float) / scale[0])

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
    method: str = "lbfgs",
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

    A :class:`BroadbandModeTransmission` objective optimizes its figure of
    merit over all of its frequencies at the same cost per evaluation, one
    forward and one adjoint solve; ``best`` is then a
    :class:`BroadbandGradientResult`.

    ``method`` is ``"lbfgs"`` (default, SciPy L-BFGS-B) or ``"adam"``
    (normalized-gradient Adam); see :func:`_descend`. L-BFGS-B is the default
    here as in :func:`optimize_parametric`: in an optimizer comparison with the
    exact gradient (one 3D CPU run per optimizer on one mesh) it reached the
    higher transmission on a 64-pixel topology crossing and led Adam at equal
    evaluations from the 6th evaluation on. Adam was the default before that
    comparison, for its scale-free update; pass ``method="adam"`` to keep it.
    Every L-BFGS-B evaluation is one SciPy asks for, the first at ``rho0``;
    each iteration takes one or more (its line search), so ``n_iters``
    iterations can take more than ``n_iters`` evaluations. For a FEW smooth
    shape parameters, prefer :func:`optimize_parametric`. ``n_iters`` bounds
    the optimizer iterations. Densities are box-constrained to ``bounds``.
    Maximizes ``objective`` by default; returns the BEST design seen. ``step``
    is used only by Adam.

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
    """Shape-optimization history and the best parameters seen.

    ``history``, ``params_history`` and ``grads`` hold every evaluation in
    order, the first at the starting point; each evaluation is two solves.
    With L-BFGS-B they include its line-search trials, so the last
    evaluation need not be the best. ``grads`` are the
    chained ``dJ/dp`` of each evaluation, before any scaling inside the
    optimizer.
    ``best`` is the density-level :class:`GradientResult` of the best
    evaluation. With a ``penalty``, ``history`` and ``grads`` stay the
    objective's own, ``penalty_history`` holds the penalty of each
    evaluation, and the best evaluation is the one with the best penalized
    merit (objective minus penalty when maximizing, plus when minimizing).
    """
    params: np.ndarray               # optimized parameter vector
    rho: np.ndarray                  # best design's density grid, expand(params)
    history: List[float]             # objective per evaluation
    params_history: List[np.ndarray]
    best: GradientResult
    grads: List[np.ndarray] = field(default_factory=list)
    penalty_history: List[float] = field(default_factory=list)


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
    jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    callback: Optional[Callable[[int, GradientResult, np.ndarray], None]] = None,
    check_jacobian: bool = True,
    penalty: Optional[Callable[[np.ndarray], Tuple[float, np.ndarray]]] = None,
) -> ParametricResult:
    """Adjoint shape optimization of a few geometric parameters ``p``
    instead of a free density per pixel.

    ``expand(p)`` renders the density grid (``region.shape``, values in
    ``[0, 1]``), for example a guide's widths rendered by
    :meth:`WidthProfile.density`. Each evaluation is
    :func:`parametric_value_and_gradient`: one forward and one adjoint solve
    give ``dJ/drho``, and the chain rule through ``drho/dp`` gives
    ``dJ/dp`` without further solves, so the cost per evaluation does not
    grow with the number of pixels or parameters, nor with the number of
    frequencies of a :class:`BroadbandModeTransmission`. ``jacobian(p)`` supplies
    ``drho/dp`` exactly (:meth:`WidthProfile.jacobian`); without it a
    central difference of ``expand`` with step ``fd_step`` is used, which is
    wrong for densities within ``fd_step`` of a kink of ``expand``. Both
    paths run the no-solve checks of :func:`parametric_value_and_gradient`
    on the first evaluation, at ``p0``, and warn on a kink or on a supplied
    Jacobian that disagrees with ``expand``; ``check_jacobian=False`` skips
    them. Check the gradient at a
    representative ``p`` with :func:`parametric_finite_difference_check`.

    ``method`` is ``"lbfgs"`` (default, SciPy L-BFGS-B) or ``"adam"``; see
    :func:`_descend`. Every L-BFGS-B evaluation is one SciPy asks for, the
    first at ``p0``; each iteration takes one or more (its line search).
    ``bounds`` (lo, hi) box-constrains every parameter. Returns the best
    parameters seen. Maximizes ``objective`` by default. Values and
    gradients use the subpixel-off model. ``beta=`` is deprecated. Both
    solves keep the caller's run settings and are exact for their recorded
    window; the error against a converged gradient follows the truncation
    of the forward results (see :func:`value_and_gradient`). For resonant
    designs and validation, set ``shutoff=0`` with an explicit ``n_steps``
    or ``run_time_s`` after checking convergence.

    ``penalty(p)`` returns a value and its gradient with respect to ``p``,
    for example :meth:`PolygonProfile.curvature_penalty`; it needs no
    solve. The optimizer then works on the merit ``J - penalty`` when
    maximizing and ``J + penalty`` when minimizing, so the penalty always
    counts against the design. ``history`` keeps ``J`` and
    ``penalty_history`` the penalty, each per evaluation, and the best
    evaluation is the one with the best merit."""
    if beta is not None:
        warnings.warn("beta= is deprecated; omit it to use the derived discrete adjoint coefficient",
                      DeprecationWarning, stacklevel=caller_stacklevel())
    sign = 1.0 if maximize else -1.0
    history: List[float] = []
    params_history: List[np.ndarray] = []
    grads: List[np.ndarray] = []
    penalties: List[float] = []
    state = {"best": None, "merit": None, "p": np.asarray(p0, float).copy()}

    def fg(p):
        p = np.asarray(p, dtype=float)
        res = _parametric(build_forward, region, objective, p, expand,
                          jacobian=jacobian, fd_step=fd_step, device=device,
                          run_kwargs=run_kwargs, monitor_name="design_region",
                          beta=beta, check_jacobian=check_jacobian and not history)
        merit, merit_grad = res.value, res.grad
        if penalty is not None:
            value, gradient = penalty(p.copy())
            gradient = np.asarray(gradient, dtype=float)
            if not np.isfinite(value) or gradient.shape != p.shape or \
                    not np.all(np.isfinite(gradient)):
                raise ValueError("penalty(p) must return a finite value and a finite "
                                 "gradient shaped like p")
            penalties.append(float(value))
            merit = res.value - sign * float(value)
            merit_grad = res.grad - sign * gradient
        history.append(res.value)
        params_history.append(p.copy())
        grads.append(res.grad.copy())
        if state["best"] is None or (sign * merit > sign * state["merit"]):
            state["best"] = res.density_gradient
            state["merit"] = merit
            state["p"] = p.copy()                     # return the BEST, not the last
        if callback is not None:
            callback(len(history), res.density_gradient, p.copy())
        return merit, merit_grad

    _descend(method, fg, np.asarray(p0, float), bounds, n_iters, step, maximize)
    best = state["best"]
    assert best is not None
    return ParametricResult(params=state["p"],
                            rho=np.asarray(expand(state["p"]), float),
                            history=history, params_history=params_history,
                            best=best, grads=grads, penalty_history=penalties)
