"""Adjoint-method gradients for inverse design (topology optimization).

A *continuous* (frequency-domain) adjoint built entirely on the existing
forward solver — no engine change. The whole point: one gradient over the whole
design region costs **two** simulations (one forward, one adjoint) regardless of
the number of design variables, where a central finite-difference check of the
same gradient costs **2N** simulations. ``benchmarks/adjoint/gradient_check.py``
verifies the two agree.

Physics
-------
Time-harmonic Maxwell in second-order (curl-curl) form, ``e^{-i w t}``::

    A(eps) E = i w mu0 J ,     A = curl curl - k0^2 eps ,   k0 = w/c .

Expose the design as the relative permittivity ``eps_i`` of each design *pixel*
(a small box of Yee cells). For a figure of merit ``J(E)`` read at frequency
``w``, first-order perturbation + the reciprocity ``A = A^T`` of a lossless
reciprocal medium gives the textbook adjoint sensitivity

    dJ/deps_i  =  Re[ beta . conj(u) . sum_{cells in i} sum_c E^c_fwd E^c_adj ]

where

* ``E_fwd`` is the forward field (the DFT monitor over the design region),
* ``E_adj`` is the field of an **adjoint** run whose *unit* source sits at the
  objective monitor, polarized along the objective (here a point dipole),
* ``u`` is the complex forward objective amplitude (e.g. ``E_z`` at the probe),
  and ``conj(u)`` is the objective's adjoint excitation coefficient,
* the sum runs over the Yee cells of pixel ``i`` and the three E-components, each
  sampled at its own Yee node — matching how the engine assigns ``eps`` per
  component (``sample_component_eps``, ``reference_solver.cpp``), and
* ``beta`` is one complex normalization constant.

Only the *direction* of the gradient carries physics; ``beta`` is a
units/normalization factor (it absorbs the section-12 DFT-phasor normalization
and the Gaussian-pulse delay phase) that is **pinned once** against finite
differences (``BETA``, confirmed in ``benchmarks/adjoint/gradient_check.py``). A
step-normalized optimizer (Adam, line search) sees only the gradient direction,
i.e. only ``arg(beta)``; ``|beta|`` rescales every component equally and never
changes a design.

The recorded phasors are normalized by the first source's ``A0 . S(f)``
(``NUMERICS.md`` section 12). Running both the forward and adjoint sources at
unit amplitude with the pulse centred on the objective frequency (``S(f0)=1``)
makes the recorded field the per-unit-drive Green's response, so ``arg(beta)`` is
a property of the discretization and transfers across geometry (the benchmark's
second problem freezes ``beta`` from the first and still agrees).

This is the *continuous* adjoint (it differentiates the frequency-domain Maxwell
operator, not the discrete time-stepper), so it matches a finite-difference
gradient of the FDTD solver up to discretization error — a few percent on a
coarse grid, shrinking as ``dl`` falls.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .components import (
    Box,
    FieldDftMonitor,
    Medium,
    PointDipole,
    Simulation,
    Structure,
)
from .data import SimulationData
from .runners import run_local
from .plugins.mode_devices import _TANGENTIAL, mode_monitor, mode_source
from .plugins.mode_overlap import mode_amplitude
from .plugins.modes import Mode

# Complex adjoint normalization constant, pinned against central finite
# differences (benchmarks/adjoint/gradient_check.py). The structural physics
# (gradient ∝ Re[beta . conj(u) . sum_c E_fwd.E_adj]) is exact; BETA fixes the
# overall complex scale set by the §12 phasor normalization and the pulse delay
# phase. arg(BETA) sets the gradient DIRECTION (what a step-normalized optimizer
# uses); |BETA| is an arbitrary overall scale. Value from the benchmark fit,
# normalized to |BETA|=1 (direction is all that matters downstream):
# arg = -1.440 rad, the e^{-iwt} §12-phasor + Gaussian-pulse-delay phase. The
# source/probe propagation phases cancel, so arg is ~geometry-independent (the
# benchmark's two problems agree to ~0.2 rad -> cos > 0.98 frozen).
BETA: complex = 0.1304 - 0.9915j    # exp(-1.440j); see gradient_check.py

# Same role as BETA but for a MODE-SOURCE adjoint (ModePower): the adjoint
# excitation is a backward guided mode (peak-normalized profile), not a unit
# point dipole, and the objective coefficient is the P_mode-normalized overlap
# c, so the constant differs from BETA. Pinned the same way (mode_power_check.py,
# SOI TE0 transmission): cos(adjoint, finite-diff) = 0.992.
BETA_MODE: complex = -0.1928 + 0.9812j   # exp(1.765j); see mode_power_check.py


def _axis_box(c0: int, c1: int, dl: float, quarter: bool) -> Tuple[float, float]:
    """(center, size) microns for a box spanning integer cells ``[c0, c1)``.

    ``quarter=False`` puts the faces on cell boundaries ``k*dl`` — cell centres
    ``(k+0.5)*dl`` then fall strictly inside exactly one such box, so adjacent
    pixel boxes tile the design region with no last-wins ambiguity (§9).
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
    a relative permittivity ``eps = eps_min + rho*(eps_max - eps_min)`` and
    emitted as one :class:`~photonhub.Structure` box. ``eps_min >= 1`` (the
    client forbids sub-vacuum permittivity).

    Build with :meth:`on_grid` from physical microns; the raw constructor takes
    cell indices.
    """

    dl_um: float
    cells: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]  # (i0,i1)...
    shape: Tuple[int, int, int]
    eps_min: float = 1.0
    eps_max: float = 12.25  # ~Si at 1.55 um
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
    ) -> "DesignRegion":
        """Snap a physical region (centre/size in microns) to integer cells.

        The low corner snaps to ``round((center-size/2)/dl)`` and the span to
        ``round(size/dl)`` cells, then rounded UP so each axis divides into its
        pixel count. Use :attr:`size_um` / :attr:`center_um` afterwards to read
        the realized (snapped) extent."""
        cells = []
        for c, s, n in zip(center_um, size_um, shape):
            lo = int(round((c - 0.5 * s) / dl_um))
            span = int(round(s / dl_um))
            span += (-span) % n  # round span up to a multiple of n
            cells.append((lo, lo + span))
        return cls(dl_um=dl_um, cells=tuple(cells), shape=shape,
                   eps_min=eps_min, eps_max=eps_max)

    # -- geometry -----------------------------------------------------------

    @property
    def n_params(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def cells_per_pixel(self) -> Tuple[int, int, int]:
        return tuple((a1 - a0) // n
                     for (a0, a1), n in zip(self.cells, self.shape))  # type: ignore

    @property
    def size_um(self) -> Tuple[float, float, float]:
        return tuple((a1 - a0) * self.dl_um for a0, a1 in self.cells)  # type: ignore

    @property
    def center_um(self) -> Tuple[float, float, float]:
        return tuple(0.5 * (a0 + a1) * self.dl_um
                     for a0, a1 in self.cells)  # type: ignore

    def eps(self, rho: np.ndarray) -> np.ndarray:
        """Per-pixel relative permittivity for densities ``rho in [0,1]``."""
        return self.eps_min + np.asarray(rho) * (self.eps_max - self.eps_min)

    def structures(self, rho: np.ndarray) -> Tuple[Structure, ...]:
        """One box :class:`~photonhub.Structure` per pixel for the density grid
        ``rho`` (shape ``self.shape`` or flat, row-major ``[ix, iy, iz]``)."""
        rho = np.asarray(rho, dtype=float).reshape(self.shape)
        eps = self.eps(rho)
        (i0, _), (j0, _), (k0, _) = self.cells
        cpx, cpy, cpz = self.cells_per_pixel
        out: List[Structure] = []
        for ix in range(self.shape[0]):
            for iy in range(self.shape[1]):
                for iz in range(self.shape[2]):
                    cx, sx = _axis_box(i0 + ix * cpx, i0 + (ix + 1) * cpx,
                                       self.dl_um, quarter=False)
                    cy, sy = _axis_box(j0 + iy * cpy, j0 + (iy + 1) * cpy,
                                       self.dl_um, quarter=False)
                    cz, sz = _axis_box(k0 + iz * cpz, k0 + (iz + 1) * cpz,
                                       self.dl_um, quarter=False)
                    out.append(Structure(
                        geometry=Box(center_um=(cx, cy, cz),
                                     size_um=(sx, sy, sz)),
                        medium=Medium(permittivity=float(eps[ix, iy, iz]))))
        return tuple(out)

    def monitor(self, freq_hz: float, name: str = "design_region"
                ) -> FieldDftMonitor:
        """The DFT monitor recording all three E-components over the region
        (quarter-cell faces so the components co-snap, §12)."""
        (i0, i1), (j0, j1), (k0, k1) = self.cells
        cx, sx = _axis_box(i0, i1, self.dl_um, quarter=True)
        cy, sy = _axis_box(j0, j1, self.dl_um, quarter=True)
        cz, sz = _axis_box(k0, k1, self.dl_um, quarter=True)
        return FieldDftMonitor(
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


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointIntensity:
    """Maximize ``|E_comp(probe)|^2`` at a single point and frequency — the
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

    #: Normalization constant for this objective's adjoint (unit point dipole).
    beta: complex = BETA

    def monitor(self) -> FieldDftMonitor:
        return FieldDftMonitor(
            name=self.name, center_um=self.probe_um, size_um=(0.0, 0.0, 0.0),
            fields=(self.component,), freqs_hz=(self.freq_hz,))

    def amplitude(self, data: SimulationData) -> complex:
        """The complex forward objective phasor ``u = E_comp(probe)``."""
        da = data[self.name].sel(component=self.component, f=self.freq_hz)
        return complex(np.asarray(da.values).reshape(()).item())

    def value(self, data: SimulationData) -> float:
        """Figure of merit ``|u|^2``."""
        return float(abs(self.amplitude(data)) ** 2)

    def adjoint_source(self, forward_sim: Simulation) -> PointDipole:
        """Unit point dipole at the probe, reusing the forward pulse so the §12
        normalization matches (the objective coefficient is applied in
        post-processing, not in the source amplitude)."""
        pulse = forward_sim.sources[0].source_time
        return PointDipole(center_um=self.probe_um, polarization=self.component,
                           amplitude=1.0, source_time=pulse)

    def adjoint_coeff(self, data: SimulationData) -> complex:
        """The complex excitation coefficient ``conj(u)`` applied to the
        forward·adjoint field product when assembling the gradient."""
        return complex(np.conjugate(self.amplitude(data)))


@dataclass(frozen=True)
class ModePower:
    """Maximize the power coupled into a guided ``mode`` at an output port —
    ``J = |c|^2``, where ``c`` is the P_mode-normalized complex modal amplitude
    of the recorded plane (``mode_amplitude``; ``|c|^2`` is the modal power
    transmission). This is THE objective for waveguide inverse design: bends,
    mode converters, (de)multiplexers, grating couplers.

    By reciprocity the adjoint excitation is the SAME mode launched BACKWARD from
    the output plane (a `ModeSource` with ``direction`` reversed), with the
    post-multiplied coefficient ``conj(c)``. Build the recording monitor with
    :meth:`monitor` (a 4-tangential `FieldDftMonitor` on the output plane) — pass
    a `Simulation` that carries the run's grid (any shell with the right
    ``size_um``/``grid`` works; the domain is fixed across the optimization).
    """

    mode: Mode
    axis: str                 # propagation axis (x/y/z)
    position_um: float        # output plane position along `axis`
    freq_hz: float
    direction: str = "+"      # "+" = power flowing toward +axis is "transmitted"
    name: str = "mode_out"
    center_um: Optional[Tuple[float, float]] = None
    thickness_axis: Optional[str] = None

    #: Normalization constant for this objective's adjoint (backward mode source).
    beta: complex = BETA_MODE

    def _mm(self, simulation: Simulation):
        return mode_monitor(
            simulation, self.mode, axis=self.axis, position_um=self.position_um,
            freqs_hz=[self.freq_hz], name=self.name, direction=self.direction,
            center_um=self.center_um, thickness_axis=self.thickness_axis)

    def monitor(self, simulation: Simulation) -> FieldDftMonitor:
        """The 4-tangential DFT monitor on the output plane (add to the sim's
        monitors). ``simulation`` supplies the grid/size only."""
        return self._mm(simulation).field_monitor

    def amplitude(self, data: SimulationData) -> complex:
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

    def value(self, data: SimulationData) -> float:
        """Figure of merit ``|c|^2`` — the modal power transmission."""
        return float(abs(self.amplitude(data)) ** 2)

    def adjoint_source(self, forward_sim: Simulation):
        """The target mode launched BACKWARD from the output plane (unit
        amplitude, reusing the forward pulse); the ``conj(c)`` coefficient is
        applied in post-processing."""
        back = "-" if self.direction == "+" else "+"
        pulse = forward_sim.sources[0].source_time
        return mode_source(
            forward_sim, self.mode, axis=self.axis, position_um=self.position_um,
            source_time=pulse, direction=back, amplitude=1.0,
            center_um=self.center_um, thickness_axis=self.thickness_axis)

    def adjoint_coeff(self, data: SimulationData) -> complex:
        """The complex excitation coefficient ``conj(c)``."""
        return complex(np.conjugate(self.amplitude(data)))


# ---------------------------------------------------------------------------
# Gradient assembly
# ---------------------------------------------------------------------------

@dataclass
class GradientResult:
    value: float                 # figure of merit J
    grad: np.ndarray             # dJ/drho, shape DesignRegion.shape (flat order)
    forward: SimulationData
    adjoint: SimulationData
    amplitude: complex           # forward objective phasor u


def _region_field(data: SimulationData, region: DesignRegion, freq_hz: float,
                  name: str) -> np.ndarray:
    """Complex (3, nz, ny, nx) array of (Ex,Ey,Ez) over the design monitor."""
    da = data[name].sel(f=freq_hz)
    comps = [np.asarray(da.sel(component=c).values) for c in ("Ex", "Ey", "Ez")]
    return np.stack(comps, axis=0)


def assemble_gradient(
    region: DesignRegion,
    forward: SimulationData,
    adjoint: SimulationData,
    coeff: complex,
    freq_hz: float,
    *,
    monitor_name: str = "design_region",
    beta: complex = BETA,
) -> np.ndarray:
    """dJ/drho per pixel from the forward and adjoint region fields.

    ``coeff`` is the objective's adjoint coefficient (``conj(u)`` for
    :class:`PointIntensity`). Returns a flat array of length
    ``region.n_params`` in row-major ``[ix, iy, iz]`` order.
    """
    da = forward[monitor_name].sel(f=freq_hz)
    pix = region._pixel_index(da).ravel()                 # (Ncells,)
    e_fwd = _region_field(forward, region, freq_hz, monitor_name)
    e_adj = _region_field(adjoint, region, freq_hz, monitor_name)
    # sum_c E_fwd^c . E_adj^c at each cell (NOT conjugated — reciprocity pairs
    # the un-conjugated phasors), summed into pixels, then weighted by the
    # objective coefficient and the complex normalization constant.
    prod = np.sum(e_fwd * e_adj, axis=0).ravel()          # complex (Ncells,)
    n = region.n_params
    pix_prod = np.zeros(n, dtype=complex)
    good = pix >= 0
    np.add.at(pix_prod, pix[good], prod[good])
    g = np.real(beta * coeff * pix_prod)                  # real per pixel
    g *= (region.eps_max - region.eps_min)                # chain rule rho->eps
    return g


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

BuildForward = Callable[[np.ndarray], Simulation]
"""User callback: ``rho -> Simulation``. Must place the domain/source and
include BOTH ``region.monitor(freq)`` and ``objective.monitor()`` plus
``region.structures(rho)``."""


Objective = Union[PointIntensity, ModePower]
"""An adjoint objective: provides ``value``, ``amplitude``, ``adjoint_source``,
``adjoint_coeff``, ``freq_hz`` and a ``beta`` normalization constant."""


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
) -> GradientResult:
    """One adjoint gradient: a forward solve + an adjoint solve (2 total).

    ``build_forward(rho)`` returns the forward :class:`~photonhub.Simulation`.
    The adjoint simulation is derived from it by swapping in the objective's
    unit adjoint source and keeping only the design-region monitor. ``beta``
    defaults to the objective's own normalization constant.

    ``device`` (``"cpu"`` / ``"gpu"`` / ``"gpu:N"``) runs BOTH solves — the
    overwhelming majority of the gradient's compute — on that backend (it is
    forwarded to ``run_local``); it overrides any ``device`` in ``run_kwargs``.
    The host-side gradient assembly (``assemble_gradient``) is a negligible
    NumPy reduction over the design-region phasors and always runs on the CPU."""
    run_kwargs = dict(run_kwargs or {})
    run_kwargs.setdefault("quiet", True)
    if device is not None:
        run_kwargs["device"] = device
    if beta is None:
        beta = objective.beta

    fwd_sim = build_forward(np.asarray(rho, dtype=float))
    # The continuous adjoint gradient is derived for the pixel-eps = rho
    # rasterization WITHOUT subpixel smoothing (chaining the smoothing weights
    # into d(eps)/d(rho) is the deferred curved-adjoint work). Pin subpixel OFF
    # for both the forward and adjoint solves so the gradient stays FD-consistent
    # regardless of the project-wide subpixel default (D2, NUMERICS §16). This is
    # backward-compatible: the pre-D2 default was already off here.
    fwd_sim = fwd_sim.model_copy(update={"subpixel": False})
    forward = run_local(fwd_sim, **run_kwargs)
    if forward.aborted:
        raise RuntimeError(f"forward run aborted: {forward.abort_reason}")

    u = objective.amplitude(forward)
    coeff = objective.adjoint_coeff(forward)

    adj_sim = fwd_sim.model_copy(update={
        "sources": (objective.adjoint_source(fwd_sim),),
        "monitors": (region.monitor(objective.freq_hz, monitor_name),),
    })
    adjoint = run_local(adj_sim, **run_kwargs)
    if adjoint.aborted:
        raise RuntimeError(f"adjoint run aborted: {adjoint.abort_reason}")

    g = assemble_gradient(region, forward, adjoint, coeff, objective.freq_hz,
                          monitor_name=monitor_name, beta=beta)
    return GradientResult(value=objective.value(forward), grad=g,
                          forward=forward, adjoint=adjoint, amplitude=u)


@dataclass
class OptimizeResult:
    rho: np.ndarray              # final density grid (region.shape)
    history: List[float]         # objective per iteration
    grads: List[np.ndarray]
    best: GradientResult


def _descend(method, fg, x0, bounds, n_iters, step, maximize):
    """Drive the chosen optimizer on an oracle ``fg(x) -> (J, dJ/dx)`` that does
    its own bookkeeping (history / best / callback). MAXIMIZES ``J`` if
    ``maximize`` else minimizes; ``bounds`` is ``(lo, hi)`` applied to every
    variable, or None.

    Default ``method="lbfgs"`` is SciPy L-BFGS-B — a quasi-Newton method that
    builds curvature from the gradient history and line-searches each step, the
    standard choice for adjoint inverse design. Unlike Adam, L-BFGS USES the
    gradient magnitude, but the adjoint gradient's magnitude is a normalization
    constant (only ``arg(beta)`` — the direction — is physically pinned; the raw
    magnitude is ~1e-21). So we CALIBRATE that constant once — probe ``J`` along
    the gradient at the start to recover the scale that makes ``grad`` consistent
    with ``J`` (the scale is constant across the design, so one probe suffices) —
    then hand SciPy a consistent, ``J(x0)``-normalized ``(f, grad)``.
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
            # along itself — with the BETA phase pinned, that means the FD
            # probe was noise-dominated (solver jitter ~ (Jp-J0)); freezing a
            # flipped/garbage scale would corrupt every subsequent step. The
            # calibration is an efficiency aid, not correctness-critical, so
            # fall back to the raw adjoint gradient scale.
            warnings.warn(
                f"gradient-scale calibration probe returned cal={cal:.3g} "
                "(noise-dominated FD probe?); falling back to cal=1.0",
                stacklevel=2)
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

    ``method`` is ``"adam"`` (default — normalized-gradient Adam) or ``"lbfgs"``
    (SciPy L-BFGS-B); see :func:`_descend`. Adam is the default HERE because for
    high-dimensional topology the continuous-adjoint gradient is slightly noisy
    (cos ~0.98 vs finite differences), which perturbs L-BFGS's curvature estimates
    — empirically Adam reaches a better design in fewer evaluations
    (``benchmarks/adjoint/optimizer_compare.py``). For a FEW smooth shape
    parameters, prefer :func:`optimize_parametric`, which defaults to L-BFGS-B.
    ``n_iters`` bounds the optimizer iterations. Densities are box-constrained to
    ``bounds``. Maximizes ``objective`` by default; returns the BEST design seen.
    ``step`` is used only by Adam.

    ``device`` (``"cpu"`` / ``"gpu"`` / ``"gpu:N"``) runs every iteration's two
    solves on that backend (forwarded to ``run_local``).

    ``param_map`` (a ``(region.shape) -> (region.shape)`` array map) constrains the
    design: the structure is built from ``param_map(rho)`` and the gradient is
    mapped through it — a proper projected gradient. For a LINEAR, self-adjoint
    projection (e.g. a symmetry-averaging map, which is its own adjoint) this is
    exact; ``best`` and ``rho`` are reported in the mapped (constrained) space.
    Use it to enforce device symmetry or a density filter."""
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
                                 device=device, run_kwargs=run_kwargs, beta=beta)
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
# Parameter (shape) optimization — a handful of geometric design variables
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
    parameterization Jacobian ``drho/dp`` — which is a **cheap central
    finite-difference of the analytic** ``expand`` (no extra FDTD solves):

        dJ/dp_j = sum_i (dJ/drho_i) (drho_i/dp_j),
        drho/dp_j  ≈  (expand(p + h e_j) - expand(p - h e_j)) / 2h   (h = fd_step)

    So the cost is the SAME two solves per evaluation as topology optimization,
    regardless of the pixel count, while optimizing only ``len(p)`` variables.
    Make ``expand`` smooth (a soft/graded boundary over ~1 cell) so ``drho/dp`` is
    well-defined. ``method`` is ``"lbfgs"`` (default, SciPy L-BFGS-B — well suited
    to a few smooth parameters) or ``"adam"``; see :func:`_descend`. ``bounds``
    (lo, hi) box-constrains every parameter. Returns the BEST parameters seen.
    Maximizes ``objective`` by default."""
    sign = 1.0 if maximize else -1.0
    history: List[float] = []
    params_history: List[np.ndarray] = []
    state = {"best": None, "p": np.asarray(p0, float).copy()}

    def fg(p):
        p = np.asarray(p, dtype=float)
        rho = np.asarray(expand(p), dtype=float)
        res = value_and_gradient(build_forward, region, objective, rho,
                                 device=device, run_kwargs=run_kwargs, beta=beta)
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
