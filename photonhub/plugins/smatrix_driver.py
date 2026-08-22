"""One-call multiport S-matrix extraction — plan, run, assemble, export.

This is the automation layer over :mod:`photonhub.plugins.smatrix` (the
per-column assembler): the PhotonHub analogue of Tidy3D's
``ModalComponentModeler`` and Lumerical's S-parameter matrix sweep. You declare
the device once (a :class:`~photonhub.components.simulation.Simulation` whose
sources are placeholders) and the N ports once (plane + outgoing direction +
mode channel); the driver then

1. solves each port's per-frequency Yee mode bank on the ACTUAL simulation
   cross-section (:func:`~photonhub.plugins.yee_mode.solve_yee_mode_bank` — the
   same discrete operator the launch and readout use),
2. builds one :class:`~photonhub.plugins.mode_devices.ModeMonitor` per port and
   one driven simulation per port (the port's mode launched INTO the device
   from just outside the port plane, all port monitors recording),
3. runs them — locally in a :class:`~photonhub.runners.batch.Batch`, on the
   cloud via ``ph.web.Batch``, or through any callable you inject,
4. assembles the columns into the full S-matrix
   (:func:`~photonhub.plugins.smatrix.assemble_smatrix`) with reciprocity /
   passivity checks attached, and
5. exports Touchstone (:func:`write_touchstone`) for circuit tools.

The manual loop this replaces is spelled out in
``examples/notebooks/11_smatrix.ipynb``; every step above remains available
individually — the driver only orchestrates public building blocks, so a
partially-manual workflow (custom banks, extra monitors, special launches) can
drop down a layer at any point.

Port geometry convention
========================
``SMatrixPort.position_um`` is the PORT PLANE (where S is referenced — the
:class:`~photonhub.plugins.smatrix.SPort` monitor plane). The mode source for
that port's drive is placed ``source_offset_um`` OUTSIDE the port plane (toward
the domain wall, in the port's ``out_direction``) and launched INWARD, so the
port monitor sits on the total-field side of its own source and records
incident + reflected when driven — exactly the arrangement
:func:`~photonhub.plugins.smatrix.smatrix` expects for ``S_jj``.

What's not handled (deferred, matching :mod:`.smatrix`)
=======================================================
* **Multimode ports** — one mode channel per port. Decompose a physical plane
  into several channels by declaring one port per (polarization, mode_index)
  at the same plane; each carries its own DFT monitor (duplicated plane
  recording — acceptable for a handful of channels, a shared-monitor
  optimization is a follow-up).
* **De-embedding** — S is referenced to the port planes as placed.
* **Reciprocity shortcuts** — every requested port is driven; use ``drive=`` to
  run a subset and mirror externally if the device is known reciprocal.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import numpy as np

from ..viz import _geometry as _geom
from ..components.source_time import GaussianPulse, SourceTimeType
from .mode_devices import ModeMonitor, mode_launch, mode_monitor
from .smatrix import (SPort, assemble_smatrix, is_passive, is_reciprocal,
                      passivity_violation, reciprocity_error, smatrix)
from .yee_mode import solve_yee_mode_bank

__all__ = [
    "SMatrixPort",
    "SMatrixPlan",
    "SMatrixResult",
    "plan_smatrix",
    "run_smatrix",
    "write_touchstone",
]

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
_SIGN = {"+": 1.0, "-": -1.0}
_OPPOSITE = {"+": "-", "-": "+"}


@dataclass(frozen=True)
class SMatrixPort:
    """One S-matrix port declaration: a plane, an outgoing direction, and a
    single mode channel.

    Parameters
    ----------
    name:
        Port label — the S-matrix index, the monitor name, and the batch entry
        key (so it must be filesystem-safe: letters, digits, ``-_.``).
    axis:
        Propagation axis of the port waveguide ('x' | 'y' | 'z').
    position_um:
        The port plane along ``axis`` — where the port monitor sits and where
        S is referenced.
    out_direction:
        '+' or '-': the direction along ``axis`` that points OUT of the device
        through this port (toward the nearby domain wall). The drive launches
        the opposite way.
    half_w_um / half_v_um:
        Half-extents of the mode-solve window in the plane's natural
        (horizontal, vertical) in-plane axes — same meaning as
        :func:`~photonhub.plugins.yee_mode.solve_yee_mode`. Choose them wide
        enough that the guided mode's evanescent tail dies inside the window.
    polarization / mode_index:
        The mode channel, counted WITHIN the TE/TM family ('TE', 0 = TE0).
    center_um:
        Transverse waveguide location as (horizontal, vertical) in-plane
        coordinates; ``None`` = the domain centre (matching
        :func:`~photonhub.plugins.mode_devices.mode_launch`).
    dl_um:
        Mode-solve transverse step; ``None`` = the simulation grid's ``dl_um``.
    source_offset_um:
        Distance from the port plane to the drive's launch plane, measured
        toward the wall (``out_direction``). ``None`` = ``10 * dl``.
    supersample / num_modes:
        Forwarded to the Yee mode solve (eigensolver frame controls).
    """

    name: str
    axis: str
    position_um: float
    out_direction: str
    half_w_um: float
    half_v_um: float
    polarization: str = "TE"
    mode_index: int = 0
    center_um: Optional[Tuple[float, float]] = None
    dl_um: Optional[float] = None
    source_offset_um: Optional[float] = None
    supersample: int = 8
    num_modes: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name or not all(
                c.isalnum() or c in "-_." for c in self.name):
            raise ValueError(
                f"port name {self.name!r} must be non-empty and contain only "
                "letters, digits, '-', '_', '.' (it becomes a batch/output "
                "directory name)")
        if self.axis not in _AXIS_IDX:
            raise ValueError(f"port {self.name!r}: axis must be one of x/y/z, "
                             f"got {self.axis!r}")
        if self.out_direction not in ("+", "-"):
            raise ValueError(f"port {self.name!r}: out_direction must be '+' "
                             f"or '-', got {self.out_direction!r}")
        pol = str(self.polarization).upper()
        if pol not in ("TE", "TM"):
            raise ValueError(f"port {self.name!r}: polarization must be TE or "
                             f"TM, got {self.polarization!r}")
        object.__setattr__(self, "polarization", pol)
        if self.mode_index < 0:
            raise ValueError(f"port {self.name!r}: mode_index must be >= 0")
        if not (self.half_w_um > 0.0 and self.half_v_um > 0.0):
            raise ValueError(
                f"port {self.name!r}: half_w_um/half_v_um must be > 0")
        if self.source_offset_um is not None and self.source_offset_um <= 0.0:
            raise ValueError(
                f"port {self.name!r}: source_offset_um must be > 0 (the "
                "launch sits strictly outside the port plane)")

    @property
    def in_direction(self) -> str:
        """The launch direction for this port's drive — into the device."""
        return _OPPOSITE[self.out_direction]


def _domain_center_hv(sim, axis: str) -> Tuple[float, float]:
    h_letter, v_letter = _geom.in_plane_axes(axis)
    return (sim.size_um[_AXIS_IDX[h_letter]] / 2.0,
            sim.size_um[_AXIS_IDX[v_letter]] / 2.0)


def _grid_dl(sim) -> float:
    dl = getattr(sim.grid, "dl_um", None)
    if dl is None:
        raise ValueError(
            "the simulation grid carries no dl_um — pass an explicit "
            "SMatrixPort.dl_um per port")
    return float(dl)


def _warn_if_in_absorbing_layers(sim, axis: str, position_um: float,
                                 what: str) -> None:
    """Soft check: a launch/monitor plane inside the PML/absorber slab is a
    physics bug the engine cannot always reject. Thickness is estimated with
    the base ``dl_um`` (approximate on graded axes — hence a warning, not an
    error)."""
    kind = getattr(sim.boundaries, axis, None)
    layers = 0
    if kind == "pml":
        layers = int(getattr(sim, "pml_num_layers", 0))
    elif kind == "absorber":
        layers = int(getattr(sim, "absorber_num_layers", 0))
    if layers <= 0:
        return
    try:
        slab = layers * _grid_dl(sim)
    except ValueError:
        return
    length = sim.size_um[_AXIS_IDX[axis]]
    if position_um < slab or position_um > length - slab:
        warnings.warn(
            f"{what} at {axis}={position_um:.4g} um lies inside the "
            f"~{slab:.4g} um {kind} slab on that wall — move the port plane "
            "or shrink source_offset_um",
            stacklevel=3)


@dataclass(frozen=True)
class SMatrixResult:
    """The assembled S-matrix plus everything needed to audit it.

    ``S`` is the complex :class:`xarray.DataArray` from
    :func:`~photonhub.plugins.smatrix.assemble_smatrix` — dims
    ``(port_out, port_in, f)``, ``|S_ij|^2`` a power ratio. ``data`` maps each
    DRIVEN port name to its run's ``SimulationData``; ``errors`` carries
    per-drive failures when ``allow_partial=True`` let the assembly proceed
    (those columns are NaN)."""

    S: Any
    ports: Tuple[SMatrixPort, ...]
    data: Mapping[str, Any]
    errors: Mapping[str, Exception]
    simulations: Mapping[str, Any]

    def sij(self, port_out: str, port_in: str):
        """``S_ij(f)`` as a 1-D complex xarray slice."""
        return self.S.sel(port_out=port_out, port_in=port_in)

    # -- checks (delegates to plugins.smatrix) -------------------------------
    def reciprocity_error(self) -> float:
        return reciprocity_error(self.S)

    def passivity_violation(self) -> float:
        return passivity_violation(self.S)

    def is_reciprocal(self, *, atol: float = 1e-6) -> bool:
        return is_reciprocal(self.S, atol=atol)

    def is_passive(self, *, atol: float = 1e-6) -> bool:
        return is_passive(self.S, atol=atol)

    def to_touchstone(self, path, **kwargs) -> Path:
        """Write the matrix as a Touchstone v1 ``.sNp`` file — see
        :func:`write_touchstone`."""
        return write_touchstone(self.S, path, **kwargs)

    def summary(self) -> str:
        """Human-readable per-frequency |S|^2 table + the two checks."""
        names = [str(p) for p in self.S.coords["port_in"].values]
        freqs = np.asarray(self.S.coords["f"].values)
        lines = ["S-matrix over ports " + ", ".join(names)]
        for j in names:
            for i in names:
                mag2 = np.abs(np.asarray(self.sij(i, j).values)) ** 2
                vals = "  ".join(f"{m:.5f}" for m in mag2)
                lines.append(f"  |S[{i},{j}]|^2 : {vals}")
        lines.append("  f (Hz)       : " +
                     "  ".join(f"{f:.5g}" for f in freqs))
        lines.append(f"  reciprocity error   max|S_ij - S_ji| : "
                     f"{self.reciprocity_error():.3e}")
        lines.append(f"  passivity violation max eig(S†S) - 1 : "
                     f"{self.passivity_violation():.3e}")
        if self.errors:
            lines.append(f"  FAILED drives: {sorted(self.errors)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SMatrixPlan:
    """A fully-prepared S-matrix extraction: the driven simulations, the port
    readers, and the run method. Built by :func:`plan_smatrix`; inspect
    ``simulations`` / ``sports`` (or ``estimate_cost()``) before committing
    compute, then call :meth:`run`."""

    base: Any
    ports: Tuple[SMatrixPort, ...]
    freqs_hz: Tuple[float, ...]
    source_time: SourceTimeType
    simulations: Mapping[str, Any]     # driven-port name -> Simulation
    sports: Tuple[SPort, ...]          # readers, one per port (all ports)

    def estimate_cost(self, **kwargs):
        """Per-drive :class:`~photonhub.cost.CostEstimate` + batch total USD,
        via :meth:`~photonhub.runners.batch.Batch.estimate_cost`."""
        from ..runners.batch import Batch
        return Batch(dict(self.simulations)).estimate_cost(**kwargs)

    def run(
        self,
        runner: Union[str, Callable[[Dict[str, Any]], Mapping[str, Any]]] = "local",
        *,
        path_dir=None,
        max_workers: int = 1,
        colocate: bool = True,
        allow_partial: bool = False,
        **runner_kwargs: Any,
    ) -> SMatrixResult:
        """Run every driven-port simulation and assemble the S-matrix.

        Parameters
        ----------
        runner:
            ``"local"`` (default) runs through
            :class:`~photonhub.runners.batch.Batch` (``path_dir`` /
            ``max_workers`` / ``solver_path`` / ``timeout`` / ``progress``
            forwarded); ``"web"`` submits via ``ph.web.Batch``
            (``max_workers`` and extra kwargs such as ``device=`` forwarded;
            ``path_dir`` is not a web concept and must be None). Any other
            callable is invoked as ``runner(simulations_dict)`` and must
            return a mapping ``driven-port name -> SimulationData`` (or any
            ``name -> DataArray`` mapping per run) — the injection point for
            custom backends and tests.
        colocate:
            Forwarded to :func:`~photonhub.plugins.smatrix.smatrix` — keep True
            for real (Yee-staggered) engine output.
        allow_partial:
            When some drives fail: False (default) raises with the per-drive
            errors; True assembles the surviving columns (missing entries NaN)
            and records the failures on the result.
        """
        sims = dict(self.simulations)
        errors: Dict[str, Exception] = {}
        if callable(runner):
            results = dict(runner(sims))
        elif runner == "local":
            from ..runners.batch import Batch
            bd = Batch(sims).run(path_dir=path_dir, max_workers=max_workers,
                                 **runner_kwargs)
            results = {name: bd[name] for name in bd.succeeded}
            errors = dict(bd.errors)
        elif runner == "web":
            if path_dir is not None:
                raise ValueError(
                    "path_dir applies to the local runner only; the web "
                    "backend manages its own artifact storage")
            from .. import web
            bd = web.Batch(sims).run(max_workers=max_workers, **runner_kwargs)
            results = {name: bd[name] for name in bd.succeeded}
            errors = dict(bd.errors)
        else:
            raise ValueError(
                f"runner must be 'local', 'web', or a callable, got {runner!r}")

        missing = [n for n in sims if n not in results]
        for n in missing:
            errors.setdefault(n, RuntimeError("runner returned no data"))
        if errors and not allow_partial:
            detail = "; ".join(f"{n}: {e!r}" for n, e in sorted(errors.items()))
            raise RuntimeError(
                f"{len(errors)} of {len(sims)} S-matrix drives failed "
                f"({detail}) — fix the failures or pass allow_partial=True")

        columns = [smatrix(list(self.sports), name, data, colocate=colocate)
                   for name, data in results.items()]
        if not columns:
            raise RuntimeError("every S-matrix drive failed; nothing to "
                               "assemble")
        S = assemble_smatrix(columns,
                             port_order=[p.name for p in self.ports])
        return SMatrixResult(S=S, ports=self.ports, data=results,
                             errors=errors, simulations=sims)


def plan_smatrix(
    simulation,
    ports: Sequence[SMatrixPort],
    *,
    freqs_hz: Sequence[float],
    source_time: Optional[SourceTimeType] = None,
    power_watts: float = 1.0,
    launch: str = "auto",
    drive: Optional[Sequence[str]] = None,
    keep_monitors: bool = False,
) -> SMatrixPlan:
    """Prepare the N driven-port simulations + port readers for a full
    S-matrix, without running anything.

    Parameters
    ----------
    simulation:
        The device: a complete :class:`~photonhub.components.simulation.Simulation`
        (structures, grid, boundaries, run controls). Its ``sources`` are
        placeholders — each driven copy replaces them with that port's mode
        launch — and its ``monitors`` are dropped unless ``keep_monitors``.
    ports:
        The port declarations (unique names).
    freqs_hz:
        Monitor frequencies for every port — the S-matrix's frequency axis.
    source_time:
        Drive pulse; ``None`` auto-tunes a
        :meth:`~photonhub.components.source_time.GaussianPulse.for_band` over
        ``freqs_hz``.
    power_watts / launch:
        Forwarded to :func:`~photonhub.plugins.mode_devices.mode_launch`.
    drive:
        Port names to actually drive (default: all). Undriven ports still get
        monitors in every run (their rows fill; their columns stay NaN).
    keep_monitors:
        Also carry the base simulation's own monitors into every driven copy.

    Returns
    -------
    SMatrixPlan
        With one prepared simulation per driven port and one
        :class:`~photonhub.plugins.smatrix.SPort` reader per port.
    """
    port_list = list(ports)
    if not port_list:
        raise ValueError("ports must be non-empty")
    names = [p.name for p in port_list]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate port name(s): {dup}")

    freqs = tuple(sorted({float(f) for f in freqs_hz}))
    if not freqs:
        raise ValueError("freqs_hz must be non-empty")
    if any(not f > 0.0 for f in freqs):
        raise ValueError("freqs_hz must all be > 0")

    if source_time is None:
        source_time = GaussianPulse.for_band(freqs_hz=freqs)
    f_central = min(freqs, key=lambda f: abs(f - source_time.freq0_hz))

    if drive is None:
        driven_names = list(names)
    else:
        driven_names = [str(d) for d in drive]
        unknown = [d for d in driven_names if d not in names]
        if unknown:
            raise ValueError(f"drive names {unknown} not among ports {names}")
        if len(set(driven_names)) != len(driven_names):
            raise ValueError("drive names must be unique")

    # --- per-port mode banks, monitors, readers -----------------------------
    banks: Dict[str, Mapping[float, Any]] = {}
    centrals: Dict[str, Any] = {}
    monitors: Dict[str, ModeMonitor] = {}
    sports: List[SPort] = []
    for p in port_list:
        dl = float(p.dl_um) if p.dl_um is not None else _grid_dl(simulation)
        h_c, v_c = (p.center_um if p.center_um is not None
                    else _domain_center_hv(simulation, p.axis))
        _warn_if_in_absorbing_layers(simulation, p.axis, p.position_um,
                                     f"port {p.name!r} plane")
        bank = solve_yee_mode_bank(
            simulation, p.axis, p.position_um, freqs, p.polarization,
            p.mode_index, h_center_um=float(h_c), v_center_um=float(v_c),
            half_w_um=float(p.half_w_um), half_v_um=float(p.half_v_um),
            dl_um=dl, supersample=p.supersample, num_modes=p.num_modes)
        banks[p.name] = bank
        centrals[p.name] = bank[f_central]
        mm = mode_monitor(
            simulation, bank[f_central], axis=p.axis,
            position_um=p.position_um, freqs_hz=freqs, name=p.name,
            direction=p.out_direction, modes_by_freq=bank)
        monitors[p.name] = mm
        sports.append(SPort(p.name, mm, out_direction=p.out_direction))

    port_field_monitors = tuple(monitors[n].field_monitor for n in names)
    extra = tuple(simulation.monitors) if keep_monitors else ()

    # --- one driven simulation per port -------------------------------------
    sims: Dict[str, Any] = {}
    for p in port_list:
        if p.name not in driven_names:
            continue
        dl = float(p.dl_um) if p.dl_um is not None else _grid_dl(simulation)
        offset = (float(p.source_offset_um)
                  if p.source_offset_um is not None else 10.0 * dl)
        src_pos = p.position_um + _SIGN[p.out_direction] * offset
        length = simulation.size_um[_AXIS_IDX[p.axis]]
        if not (0.0 < src_pos < length):
            raise ValueError(
                f"port {p.name!r}: launch plane {p.axis}={src_pos:.4g} um "
                f"falls outside the domain (0, {length:.4g}) — reduce "
                "source_offset_um or move the port plane inward")
        _warn_if_in_absorbing_layers(simulation, p.axis, src_pos,
                                     f"port {p.name!r} launch plane")
        sources = mode_launch(
            simulation, centrals[p.name], axis=p.axis, position_um=src_pos,
            source_time=source_time, direction=p.in_direction,
            power_watts=power_watts, modes_by_freq=banks[p.name],
            launch=launch)
        sims[p.name] = simulation.model_copy(update={
            "sources": tuple(sources),
            "monitors": port_field_monitors + extra,
        })

    return SMatrixPlan(base=simulation, ports=tuple(port_list),
                       freqs_hz=freqs, source_time=source_time,
                       simulations=sims, sports=tuple(sports))


def run_smatrix(
    simulation,
    ports: Sequence[SMatrixPort],
    *,
    freqs_hz: Sequence[float],
    source_time: Optional[SourceTimeType] = None,
    power_watts: float = 1.0,
    launch: str = "auto",
    drive: Optional[Sequence[str]] = None,
    keep_monitors: bool = False,
    runner: Union[str, Callable[[Dict[str, Any]], Mapping[str, Any]]] = "local",
    path_dir=None,
    max_workers: int = 1,
    colocate: bool = True,
    allow_partial: bool = False,
    **runner_kwargs: Any,
) -> SMatrixResult:
    """One call: :func:`plan_smatrix` then :meth:`SMatrixPlan.run`. See both
    for the parameters. For anything beyond a straight shot (cost preview,
    inspecting the generated sims, custom per-drive tweaks) build the plan
    explicitly."""
    plan = plan_smatrix(
        simulation, ports, freqs_hz=freqs_hz, source_time=source_time,
        power_watts=power_watts, launch=launch, drive=drive,
        keep_monitors=keep_monitors)
    return plan.run(runner, path_dir=path_dir, max_workers=max_workers,
                    colocate=colocate, allow_partial=allow_partial,
                    **runner_kwargs)


# ----------------------------------------------------------------------------
# Touchstone export
# ----------------------------------------------------------------------------

def write_touchstone(
    S,
    path,
    *,
    z0: float = 50.0,
    comments: Sequence[str] = (),
) -> Path:
    """Write an assembled S-matrix as a Touchstone v1 ``.sNp`` file.

    ``S`` is the :class:`xarray.DataArray` from
    :func:`~photonhub.plugins.smatrix.assemble_smatrix` /
    :attr:`SMatrixResult.S` — dims ``(port_out, port_in, f)`` with matching
    port labels on both axes. Frequencies are written ascending in Hz,
    real/imaginary (``RI``) format, one reference resistance ``z0`` (the
    Touchstone header requires one; it is dimensional bookkeeping only for
    these modal wave amplitudes — power waves are already normalized so
    ``|S_ij|^2`` is a power ratio).

    Layout follows the v1 spec: 1-port and 3+-port matrices in row-major
    (``port_out``-major) order with at most four S-pairs per line and each
    matrix row starting a new line; the 2-port special column-major order
    ``S11 S21 S12 S22`` on one line. NaN entries (an unassembled partial
    matrix) are rejected.

    The file extension is forced to the port count: ``device`` or
    ``device.s3p`` for a 3-port; a MISMATCHED explicit suffix raises (readers
    key the port count on it).
    """
    Sd = S.transpose("port_out", "port_in", "f")
    out_labels = [str(v) for v in Sd.coords["port_out"].values]
    in_labels = [str(v) for v in Sd.coords["port_in"].values]
    if out_labels != in_labels:
        raise ValueError(
            "port_out and port_in labels differ — assemble with a common "
            f"port_order (got {out_labels} vs {in_labels})")
    n = len(out_labels)
    if n == 0:
        raise ValueError("S carries no ports")

    freqs = np.asarray(Sd.coords["f"].values, dtype=np.float64)
    order = np.argsort(freqs)
    freqs = freqs[order]
    arr = np.asarray(Sd.values)[:, :, order]
    if not np.all(np.isfinite(arr)):
        bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
        raise ValueError(
            f"S contains {bad} non-finite entries (undriven ports / failed "
            "drives?) — Touchstone needs the complete matrix")
    if not (z0 > 0.0 and math.isfinite(z0)):
        raise ValueError(f"z0 must be a positive finite resistance, got {z0}")

    path = Path(path)
    expected = f".s{n}p"
    if path.suffix == "":
        path = path.with_suffix(expected)
    elif path.suffix.lower() != expected:
        raise ValueError(
            f"{path.name!r} does not match the {n}-port Touchstone extension "
            f"{expected!r} (readers key the port count on it); drop the "
            "suffix or use the matching one")

    def pair(v: complex) -> str:
        return f"{v.real:.12e} {v.imag:.12e}"

    lines: List[str] = []
    lines.append(f"! {n}-port S-parameters (PhotonHub photonic modal ports)")
    for k, label in enumerate(out_labels, start=1):
        lines.append(f"! port {k} = {label}")
    for c in comments:
        for piece in str(c).splitlines():
            lines.append(f"! {piece}")
    lines.append(f"# Hz S RI R {z0:g}")

    for kf, f in enumerate(freqs):
        Sf = arr[:, :, kf]
        if n == 1:
            lines.append(f"{f:.12e} {pair(Sf[0, 0])}")
        elif n == 2:
            # v1 2-port special order: S11 S21 S12 S22 on one line.
            lines.append(
                f"{f:.12e} {pair(Sf[0, 0])} {pair(Sf[1, 0])} "
                f"{pair(Sf[0, 1])} {pair(Sf[1, 1])}")
        else:
            # Row-major, each matrix row starts a new line, <= 4 pairs/line.
            for i in range(n):
                row = [pair(Sf[i, j]) for j in range(n)]
                for start in range(0, n, 4):
                    chunk = " ".join(row[start:start + 4])
                    if i == 0 and start == 0:
                        lines.append(f"{f:.12e} {chunk}")
                    else:
                        lines.append(f"  {chunk}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path
