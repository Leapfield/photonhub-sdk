"""One-call multiport S-matrix extraction, plan, run, assemble, export.

This is the automation layer over :mod:`photonhub.analysis.smatrix` (the
per-column assembler): a modal multi-port
S-parameter matrix sweep. You declare
the device once (a :class:`~photonhub.components.simulation.Simulation` whose
sources are placeholders) and the N ports once (plane + outgoing direction +
mode channel); the driver then

1. solves each port's per-frequency Yee mode mode mapping on the ACTUAL simulation
   cross-section (:func:`~photonhub.analysis.yee_mode.solve_yee_mode_bank`, the
   same discrete operator the launch and readout use),
2. builds one :class:`~photonhub.analysis.mode_devices.ModeMonitor` per port and
   one driven simulation per port (the port's mode launched INTO the device
   from just outside the port plane, all port monitors recording),
3. runs them, locally in a :class:`~photonhub.runners.batch.Batch`, on the
   cloud via ``ph.cloud.Batch``, or through any callable you inject,
4. assembles the columns into the full S-matrix
   (:func:`~photonhub.analysis.smatrix.assemble_smatrix`) with reciprocity /
   passivity checks attached, and
5. exports Touchstone (:func:`write_touchstone`) for circuit tools.

The manual loop this replaces is spelled out in
``examples/course/11_smatrix.ipynb``; every step above remains available
individually, the driver only orchestrates public building blocks, so a
partially-manual workflow (custom mode mappings, extra monitors, special launches) can
drop down a layer at any point.

Port geometry convention
========================
``SMatrixPort.position_um`` is the PORT PLANE (where S is referenced, the
:class:`~photonhub.analysis.smatrix.SPort` monitor plane). The mode source for
that port's drive is placed ``source_offset_um`` OUTSIDE the port plane (toward
the domain wall, in the port's ``out_direction``) and launched INWARD, so the
port monitor sits on the total-field side of its own source and records
incident + reflected when driven, exactly the arrangement
:func:`~photonhub.analysis.smatrix.smatrix` expects for ``S_jj``.

Port frames
===========
A simulation fitted with ``domain=`` stores its coordinates in the wire frame,
from the fitted box's low corner, which sits at ``origin_um`` in the user's
frame. The mode solve, the monitors and the launch all read the wire frame, so
:func:`plan_smatrix` moves each port there first:

* a :class:`photonhub.Port` is in the user's frame, the same place as in
  ``Simulation(ports=...)``, its outward side and mode window included. The one
  limit is a port in the half a symmetry plane mirrors away, which ``ports=``
  reads through its image and the plan refuses. A port read back from
  ``sim.ports`` of the same simulation is in the wire frame already and stands
  as it is; it stays marked through ``dataclasses.replace`` and a copy of the
  object, so an edit of its centre is in the wire frame too, and keeps its
  ``out_direction``. A new ``ph.Port``, whatever its numbers, carries no mark:
  one built from a held port's fields, or with the origin subtracted by hand,
  is in the user's frame and moves twice; a solved ``n_eff`` below the
  background at the port draws a warning.
* an :class:`SMatrixPort` names its frame in ``frame``. Where ``origin_um`` is
  not zero an unnamed frame raises, because either reading would be wrong for
  someone: earlier versions read the values in the wire frame, which callers
  relied on by subtracting the origin themselves.

The ports on a plan and a result are the ones the plan placed, in the wire
frame. The monitors and readers need no move of their own: the plan stores
wire coordinates and the readers take the recorded planes through
:func:`~photonhub.components.frame.wire_array`.

What's not handled (deferred, matching :mod:`.smatrix`)
=======================================================
* **Multimode ports**, one mode channel per port. Decompose a physical plane
  into several channels by declaring one port per (polarization, mode_index)
  at the same plane; each carries its own DFT monitor (duplicated plane
  recording, acceptable for a handful of channels, a shared-monitor
  optimization is a follow-up).
* **De-embedding**, S is referenced to the port planes as placed.
* **Reciprocity shortcuts**, every requested port is driven; use ``drive=`` to
  run a subset and mirror externally if the device is known reciprocal.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import numpy as np
import xarray as xr

from ..viz import _geometry as _geom
from ..components import frame as _frame
from ..components.source_time import GaussianPulse, SourceTimeType
from .mode_devices import ModeMonitor, mode_launch, mode_monitor
from .mode_overlap import _TRANSVERSE
from .smatrix import (SPort, assemble_smatrix, is_passive, is_reciprocal,
                      passivity_violation, reciprocity_error, smatrix)
from .yee_mode import solve_yee_mode_bank
from ._constants import C0 as _C0
from .._compat import caller_stacklevel

if TYPE_CHECKING:
    from .compact_model import SMatrixModel

__all__ = [
    "SMatrixPort",
    "SMatrixPlan",
    "SMatrixResult",
    "plan_smatrix",
    "read_touchstone",
    "run_smatrix",
    "write_touchstone",
]

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
_SIGN = {"+": 1.0, "-": -1.0}
_OPPOSITE = {"+": "-", "-": "+"}
# How far past a wall a position can sit and still be on it: the tolerance the
# fold itself keeps a port by (Simulation._fold_symmetry).
_ON_A_WALL_UM = 1e-9


@dataclass(frozen=True)
class SMatrixPort:
    """One S-matrix port declaration: a plane, an outgoing direction, and a
    single mode channel.

    Parameters
    ----------
    name:
        Port label, the S-matrix index, the monitor name, and the batch entry
        key (so it must be filesystem-safe: letters, digits, ``-_.``).
    axis:
        Propagation axis of the port waveguide ('x' | 'y' | 'z').
    position_um:
        The port plane along ``axis``, where the port monitor sits and where
        S is referenced, in the frame ``frame`` names.
    out_direction:
        '+' or '-': the direction along ``axis`` that points OUT of the device
        through this port (toward the nearby domain wall). The drive launches
        the opposite way.
    half_w_um / half_v_um:
        Half-extents of the mode-solve window in the plane's natural
        (horizontal, vertical) in-plane axes, same meaning as
        :func:`~photonhub.analysis.yee_mode.solve_yee_mode`. Choose them wide
        enough that the guided mode's evanescent tail dies inside the window.
    polarization / mode_index:
        The mode channel, counted WITHIN the TE/TM family ('TE', 0 = TE0).
    center_um:
        Transverse waveguide location as (horizontal, vertical) in-plane
        coordinates ((y, z) for an x-normal port, (x, z) for y-normal, (x, y)
        for z-normal), in the frame ``frame`` names; ``None`` = the domain
        centre. The port's mode is solved, read and launched at this centre.
    dl_um:
        Mode-solve transverse step; ``None`` = the simulation grid's ``dl_um``.
    source_offset_um:
        Distance from the port plane to the drive's launch plane, measured
        toward the wall (``out_direction``). ``None`` = a third of a
        wavelength in the background, the standoff the declarative ports take.
    supersample / num_modes:
        Forwarded to the Yee mode solve (eigensolver frame controls).
    frame:
        What ``position_um`` and ``center_um`` are measured from on a
        simulation fitted with ``domain=``, whose stored coordinates start at
        ``origin_um``. ``"user"`` is your own frame, the one a
        :class:`photonhub.Port` reads in; the plan takes ``origin_um`` off.
        ``"wire"`` is the fitted box's low corner, the frame of
        ``solve_yee_mode``, ``mode_monitor`` and ``mode_launch``; the values
        stand. ``None``, the default, is for a simulation at the origin
        (0, 0, 0), where the two are one place; on any other,
        :func:`plan_smatrix` raises ``ValueError``.
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
    frame: Optional[str] = None

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
        if self.frame not in (None, "user", "wire"):
            raise ValueError(f"port {self.name!r}: frame must be 'user', "
                             f"'wire' or None, got {self.frame!r}")

    @property
    def in_direction(self) -> str:
        """The launch direction for this port's drive, into the device."""
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


def _user_um(sim, axis: str, position_um: float) -> float:
    """A wire-frame position along ``axis`` in the user's frame, the one a
    message reports it in."""
    return position_um + _frame.frame_origin(sim)[_AXIS_IDX[axis]]


def _refuse_outside_domain(sim, port_name: str, axis: str, position_um: float,
                           what: str) -> None:
    """A port position past the walls would snap onto a wall and read what is
    there, with nothing to stop it when the port is not driven: the likely sign
    of a coordinate in the wrong frame, or of a port in the half a symmetry
    plane mirrors away. A position on a wall stands within rounding: a fold's
    plane is ``0.5 * (lo + hi)``, so a port kept on it can land just below wire
    0 (:data:`_ON_A_WALL_UM`)."""
    a = _AXIS_IDX[axis]
    length = sim.size_um[a]
    if -_ON_A_WALL_UM <= position_um <= length + _ON_A_WALL_UM:
        return
    low, high = _user_um(sim, axis, 0.0), _user_um(sim, axis, length)
    where = (f"port {port_name!r}: {what} {axis}={_user_um(sim, axis, position_um):.4g} um "
             f"lies outside the domain ({low:.4g}, {high:.4g})")
    if position_um < 0.0 and sim.symmetry[a]:
        fold = getattr(sim, "_fold", None)
        advice = ("Simulation(ports=...) reads such a port through its mirror image, but an "
                  "S-matrix port is solved where it stands: use its image in the kept half, or "
                  "drop that symmetry" if fold is not None and a in fold.planes
                  else "This half domain does not hold it: use its image inside the domain")
        raise ValueError(f"{where}, in the half the symmetry plane {axis}={low:.4g} um mirrors away. {advice}")
    raise ValueError(f"{where}; move the port inside it")


def _warn_if_not_guided(sim, port: SMatrixPort, center_hv: Tuple[float, float],
                        mode, freq_hz: float) -> None:
    """Soft check after the solve: a mode below the highest background index
    at the port is not guided there. The background is read at the corners of
    the mode window, above and below any slab beside the guide, so a filled
    cladding or a substrate counts where the simulation's own ``background``
    would not. A leaky mode can be legitimate, so this warns; it is the one
    signal for a ``ph.Port`` in the wrong frame, which carries no mark."""
    from ..components import declarative as _decl

    h_name, v_name = _geom.in_plane_axes(port.axis)
    painter = _decl._Painter(sim)
    point = [0.0, 0.0, 0.0]
    point[_AXIS_IDX[port.axis]] = port.position_um
    highest = 0.0
    for h_sign, v_sign in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)):
        point[_AXIS_IDX[h_name]] = center_hv[0] + h_sign * port.half_w_um
        point[_AXIS_IDX[v_name]] = center_hv[1] + v_sign * port.half_v_um
        highest = max(highest, painter.index(point, freq_hz))
    n_eff = float(np.real(mode.n_eff))
    if n_eff < highest:
        warnings.warn(
            f"port {port.name!r}: its mode solved at n_eff {n_eff:.4g}, below the highest "
            f"background index at the port, {highest:.4g}, so it is not guided there. A leaky "
            "mode can do this, but the port's coordinates may be in the wrong frame: a ph.Port "
            "reads in your frame, so do not subtract origin_um from it",
            stacklevel=caller_stacklevel())


def _warn_if_in_absorbing_layers(sim, axis: str, position_um: float,
                                 what: str) -> None:
    """Soft check: a launch/monitor plane inside the PML/absorber slab is a
    physics bug the engine cannot always reject. The slab is the realized
    outer layer cells on each face (:meth:`Simulation._open_interval_um`): on
    a graded axis those of the medium at that face, which can be finer than
    the base ``dl_um``; a symmetry plane's face has none."""
    kind = getattr(sim.boundaries, axis, None)
    layers = 0
    if kind == "pml":
        layers = int(getattr(sim, "pml_num_layers", 0))
    elif kind == "absorber":
        layers = int(getattr(sim, "absorber_num_layers", 0))
    if layers <= 0:
        return
    length = sim.size_um[_AXIS_IDX[axis]]
    try:
        lo, hi = sim._open_interval_um(_AXIS_IDX[axis])
    except ValueError:
        lo = hi = 0.5 * length                   # the layers fill the axis
    if position_um < lo or position_um > hi:
        slab = lo if position_um < lo else length - hi
        warnings.warn(
            f"{what} at {axis}={_user_um(sim, axis, position_um):.4g} um lies inside the "
            f"~{slab:.4g} um {kind} slab on that wall; move the port plane "
            "or shrink source_offset_um",
            stacklevel=caller_stacklevel())


@dataclass(frozen=True)
class SMatrixResult:
    """The assembled S-matrix plus everything needed to audit it.

    ``S`` is the complex :class:`xarray.DataArray` from
    :func:`~photonhub.analysis.smatrix.assemble_smatrix`, dims
    ``(port_out, port_in, f)``, ``|S_ij|^2`` a power ratio. ``data`` maps each
    DRIVEN port name to its run's ``RunResult``; ``errors`` carries
    per-drive failures when ``allow_partial=True`` let the assembly proceed
    (those columns are NaN).

    On a simulation fitted with ``domain=``, ``ports`` are the ports as placed,
    in the wire frame, from the fitted box's low corner, while the field arrays
    in ``data`` read in your frame, as any ``RunResult`` does."""

    S: Any
    ports: Tuple[SMatrixPort, ...]
    data: Mapping[str, Any]
    errors: Mapping[str, Exception]
    simulations: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def sij(self, port_out: str, port_in: str):
        """``S_ij(f)`` as a 1-D complex xarray slice."""
        return self.S.sel(port_out=port_out, port_in=port_in)

    # -- checks (delegates to analysis.smatrix) -------------------------------
    def reciprocity_error(self) -> float:
        """Maximum finite ``abs(S_ij - S_ji)`` across ports and frequencies.

        Ignore non-finite differences. Return zero when none remain."""
        return reciprocity_error(self.S)

    def passivity_violation(self) -> float:
        """Largest eigenvalue of ``S.conj().T @ S`` minus one, across frequencies.

        Skip matrices with non-finite entries. Return zero when no finite
        frequency matrix is available. A non-positive value passes passivity."""
        return passivity_violation(self.S)

    def is_reciprocal(self, *, atol: float = 1e-6) -> bool:
        """True when :meth:`reciprocity_error` is at most ``atol``."""
        return is_reciprocal(self.S, atol=atol)

    def is_passive(self, *, atol: float = 1e-6) -> bool:
        """True when :meth:`passivity_violation` is at most ``atol``."""
        return is_passive(self.S, atol=atol)

    def to_touchstone(self, path, **kwargs) -> Path:
        """Write the matrix as a Touchstone file, see
        :func:`write_touchstone`."""
        options = dict(kwargs)
        options["metadata"] = {**self.metadata, **(options.get("metadata") or {})}
        return write_touchstone(self.S, path, **options)

    def to_model(self, **kwargs) -> SMatrixModel:
        """Build a checked wavelength-interpolated S-matrix model."""
        from .compact_model import smatrix_model

        options = dict(kwargs)
        options["metadata"] = {**self.metadata, **(options.get("metadata") or {})}
        return smatrix_model(self.S, **options)

    def to_json(self, path, **kwargs) -> Path:
        """Write the S-matrix as a JSON compact model."""
        return self.to_model(**kwargs).to_json(path)

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
    ``simulations`` / ``sports`` (or ``quote()``) before committing
    compute, then call :meth:`run`. ``ports`` are the ports as placed, in the
    wire frame on a simulation fitted with ``domain=``."""

    base: Any
    ports: Tuple[SMatrixPort, ...]
    freqs_hz: Tuple[float, ...]
    source_time: SourceTimeType
    simulations: Mapping[str, Any]     # driven-port name -> Simulation
    sports: Tuple[SPort, ...]          # readers, one per port (all ports)

    def quote(self, **kwargs):
        """Per-drive :class:`~photonhub.cost.CostEstimate` + batch total USD,
        via :meth:`~photonhub.runners.batch.Batch.quote`."""
        from ..runners.batch import Batch
        return Batch(dict(self.simulations)).quote(**kwargs)

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
            forwarded); ``"cloud"`` (or the deprecated ``"web"``) submits via ``ph.cloud.Batch``
            (``max_workers`` and extra kwargs such as ``device=`` and ``solver=`` forwarded,
            with the cloud defaults when omitted: the service's only GPU and
            the installed solver's commit, resolved once for all drives;
            ``path_dir`` is not a web concept and must be None). Any other
            callable is invoked as ``runner(simulations_dict)`` and must
            return a mapping ``driven-port name -> RunResult`` (or any
            ``name -> DataArray`` mapping per run), the injection point for
            custom backends and tests.
        colocate:
            Forwarded to :func:`~photonhub.analysis.smatrix.smatrix`, keep True
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
        elif runner in ("cloud", "web"):
            if path_dir is not None:
                raise ValueError(
                    "path_dir applies to the local runner only; the web "
                    "backend manages its own artifact storage")
            from .. import cloud
            bd = cloud.Batch(sims).run(max_workers=max_workers, **runner_kwargs)
            results = {name: bd[name] for name in bd.succeeded}
            errors = dict(bd.errors)
        else:
            raise ValueError(
                f"runner must be 'local', 'cloud', or a callable, got {runner!r}")

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


def _in_wire_frame(port: SMatrixPort, simulation) -> SMatrixPort:
    """``port`` in the wire frame the mode solve, monitor and launch read: the
    origin taken off a port in the user's frame (see "Port frames" above)."""
    origin = _frame.frame_origin(simulation)
    if not any(origin) or port.frame == "wire":
        return port
    if port.frame is None:
        at = ", ".join(f"{o:.4g}" for o in origin)
        raise ValueError(
            f"port {port.name!r}: this simulation's stored coordinates start at "
            f"origin_um = ({at}), not at your origin, so an SMatrixPort must say "
            "which it is measured from. Pass frame='user' for your own coordinates, "
            "the ones a ph.Port takes, or frame='wire' for coordinates with origin_um "
            "already taken off, the way earlier versions read them")
    h_name, v_name = _geom.in_plane_axes(port.axis)
    center = port.center_um
    if center is not None:
        center = (float(center[0]) - origin[_AXIS_IDX[h_name]],
                  float(center[1]) - origin[_AXIS_IDX[v_name]])
    return replace(port, position_um=float(port.position_um) - origin[_AXIS_IDX[port.axis]],
                   center_um=center, frame="wire")


def _as_smatrix_port(port, simulation, freqs_hz, source_time) -> SMatrixPort:
    """A port as the plan reads it, an :class:`SMatrixPort` in the wire frame.
    A :class:`photonhub.Port` is taken into that frame before anything reads
    it, so that its outward side and mode window resolve as the Simulation's
    own port layer resolves them, with the band centre taken from
    ``source_time`` or the middle of ``freqs_hz``."""
    from ..components.authoring import HeldPort, Port as _Port
    from ..components import declarative as _decl

    if not isinstance(port, _Port):
        return _in_wire_frame(port, simulation)
    # A port read back from a simulation is in the wire frame, and the mark
    # stays on it through an edit, a rename and a copy. Any other is the user's.
    if not isinstance(port, HeldPort):
        port = _frame.translate(port, tuple(-o for o in _frame.frame_origin(simulation)))
    freqs = [float(f) for f in freqs_hz]
    f_c = (float(source_time.freq0_hz) if source_time is not None
           else 0.5 * (min(freqs) + max(freqs)))
    wlen0_um = _C0 / f_c * 1e6
    out_dir = _decl.infer_out_direction(simulation, port)
    n_bg = math.sqrt(float(simulation.background.permittivity))
    # the port-window rule at the longest wavelength read,
    # from the effective-index estimate: no mode solve here
    half_w, half_v = _decl.default_window(port, wlen0_um, n_bg, shell=simulation,
                                          wlen_max_um=_C0 / min(freqs) * 1e6)
    family, index = port.family_index()
    return SMatrixPort(
        port.monitor_name, port.axis, port.plane_um, out_dir, half_w, half_v,
        polarization=family, mode_index=index, center_um=port.hv_center_um(),
        dl_um=port.dl_um, source_offset_um=port.source_offset_um,
        supersample=port.supersample, num_modes=port.num_modes, frame="wire")


def plan_smatrix(
    simulation,
    ports: Sequence[Union[SMatrixPort, Any]],
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
        placeholders, each driven copy replaces them with that port's mode
        launch, and its ``monitors`` are dropped unless ``keep_monitors``.
    ports:
        The port declarations (unique names): :class:`photonhub.Port` values,
        in your frame and at the same place as in ``Simulation(ports=...)``,
        whose outward side and mode window resolve as they do there; or
        :class:`SMatrixPort` values, which name their ``frame`` on a simulation
        fitted with ``domain=``. A port in the half a symmetry plane mirrors
        away raises ``ValueError``, where ``ports=`` reads it through its
        image: use the image, or drop the symmetry. A port read from
        ``sim.ports`` of this same simulation is in the wire frame and stands
        as it is, also after a rename or an edit of its centre, which is in
        that frame too. Such an edit keeps the port's ``out_direction``: moved
        across the device, a port needs ``out_direction=None``. A new
        :class:`photonhub.Port` built from its fields is in your frame.
    freqs_hz:
        Monitor frequencies for every port, the S-matrix's frequency axis.
    source_time:
        Drive pulse; ``None`` auto-tunes a
        :meth:`~photonhub.components.source_time.GaussianPulse.for_band` over
        ``freqs_hz``.
    power_watts / launch:
        Forwarded to :func:`~photonhub.analysis.mode_devices.mode_launch`.
    drive:
        Port names to actually drive (default: all). Undriven ports still get
        monitors in every run (their rows fill; their columns stay NaN).
    keep_monitors:
        Also carry the base simulation's own monitors into every driven copy.

    Returns
    -------
    SMatrixPlan
        With one prepared simulation per driven port and one
        :class:`~photonhub.analysis.smatrix.SPort` reader per port.
    """
    port_list = [_as_smatrix_port(p, simulation, freqs_hz, source_time) for p in ports]
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
    centers: Dict[str, Tuple[float, float]] = {}
    monitors: Dict[str, ModeMonitor] = {}
    sports: List[SPort] = []
    for p in port_list:
        dl = float(p.dl_um) if p.dl_um is not None else _grid_dl(simulation)
        h_c, v_c = (p.center_um if p.center_um is not None
                    else _domain_center_hv(simulation, p.axis))
        centers[p.name] = (float(h_c), float(v_c))
        _refuse_outside_domain(simulation, p.name, p.axis, p.position_um, "plane")
        for letter, centre in zip(_geom.in_plane_axes(p.axis), centers[p.name]):
            if getattr(simulation.boundaries, letter) not in ("periodic", "bloch"):   # a period wraps
                _refuse_outside_domain(simulation, p.name, letter, centre, "centre")
        _warn_if_in_absorbing_layers(simulation, p.axis, p.position_um,
                                     f"port {p.name!r} plane")
        bank = solve_yee_mode_bank(
            simulation, p.axis, p.position_um, freqs, p.polarization,
            p.mode_index, h_center_um=float(h_c), v_center_um=float(v_c),
            half_w_um=float(p.half_w_um), half_v_um=float(p.half_v_um),
            dl_um=dl, supersample=p.supersample, num_modes=p.num_modes)
        banks[p.name] = bank
        centrals[p.name] = bank[f_central]
        _warn_if_not_guided(simulation, p, centers[p.name], bank[f_central], f_central)
        # The readout frame (t1, t2) swaps the solve's (h, v) on a y-normal
        # port. Without a centre the monitor reads at the recorded plane's
        # mean coordinate, the domain centre (the bank's modes carry no solve
        # provenance for mode_monitor to take it from).
        h_name, v_name = _geom.in_plane_axes(p.axis)
        by_axis = {h_name: float(h_c), v_name: float(v_c)}
        mm = mode_monitor(
            simulation, bank[f_central], axis=p.axis,
            position_um=p.position_um, freqs_hz=freqs, name=p.name,
            direction=p.out_direction,
            center_um=tuple(by_axis[t] for t in _TRANSVERSE[p.axis]),
            modes_by_freq=bank)
        monitors[p.name] = mm
        sports.append(SPort(p.name, mm, out_direction=p.out_direction))

    port_field_monitors = tuple(monitors[n].field_monitor for n in names)
    extra = tuple(simulation.monitors) if keep_monitors else ()

    # --- one driven simulation per port -------------------------------------
    # The launch standoff is a LENGTH, never a cell count: the
    # same wavelength-derived default the declarative ports take.
    from ..components import declarative as _decl

    wlen0_um = _C0 / float(source_time.freq0_hz) * 1e6
    n_bg = math.sqrt(float(simulation.background.permittivity))
    sims: Dict[str, Any] = {}
    for p in port_list:
        if p.name not in driven_names:
            continue
        offset = (float(p.source_offset_um) if p.source_offset_um is not None
                  else _decl.default_source_offset_um(wlen0_um, n_bg))
        src_pos = p.position_um + _SIGN[p.out_direction] * offset
        length = simulation.size_um[_AXIS_IDX[p.axis]]
        if not (0.0 < src_pos < length):
            raise ValueError(
                f"port {p.name!r}: launch plane {p.axis}="
                f"{_user_um(simulation, p.axis, src_pos):.4g} um falls outside the "
                f"domain ({_user_um(simulation, p.axis, 0.0):.4g}, "
                f"{_user_um(simulation, p.axis, length):.4g}); reduce "
                "source_offset_um or move the port plane inward")
        _warn_if_in_absorbing_layers(simulation, p.axis, src_pos,
                                     f"port {p.name!r} launch plane")
        # Launch at the port's own centre, where its bank was solved and its
        # monitor reads. The bank's modes carry no solve provenance, so without
        # it mode_launch falls back to the domain centre and an off-centre port
        # (any arm of a 1 x 2 or 2 x 2 device) was driven beside its guide.
        sources = mode_launch(
            simulation, centrals[p.name], axis=p.axis, position_um=src_pos,
            source_time=source_time, direction=p.in_direction,
            power_watts=power_watts, center_um=centers[p.name],
            modes_by_freq=banks[p.name], launch=launch)
        sims[p.name] = simulation.model_copy(update={
            "sources": tuple(sources),
            "monitors": port_field_monitors + extra,
        })

    return SMatrixPlan(base=simulation, ports=tuple(port_list),
                       freqs_hz=freqs, source_time=source_time,
                       simulations=sims, sports=tuple(sports))


def run_smatrix(
    simulation,
    ports: Sequence[Union[SMatrixPort, Any]],
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
    version: int = 1,
    metadata: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
    passivity_atol: float = 1e-3,
    reciprocity_atol: float = 1e-3,
) -> Path:
    """Write an assembled S-matrix as a Touchstone file.

    ``S`` is the :class:`xarray.DataArray` from
    :func:`~photonhub.analysis.smatrix.assemble_smatrix` /
    :attr:`SMatrixResult.S`, dims ``(port_out, port_in, f)`` with matching
    port labels on both axes. Frequencies are written ascending in Hz,
    real/imaginary (``RI``) format, one reference resistance ``z0`` (the
    Touchstone header requires one; it is dimensional bookkeeping only for
    these modal wave amplitudes, power waves are already normalized so
    ``|S_ij|^2`` is a power ratio).

    Version 1 layout follows the v1 spec: 1-port and 3+-port matrices in row-major
    (``port_out``-major) order with at most four S-pairs per line and each
    matrix row starting a new line; the 2-port special column-major order
    ``S11 S21 S12 S22`` on one line. NaN entries (an unassembled partial
    matrix) are rejected.

    Version 1's file extension is forced to the port count: ``device`` or
    ``device.s3p`` for a 3-port; a mismatched suffix raises. Version 2 uses
    ``.ts`` by default and records port count in its header. Failed checks
    warn and are recorded, or raise before writing when ``strict=True``.
    Port labels, comments and metadata must contain ASCII text.
    """
    from .compact_model import PROVENANCE_NOTE, _prepare_smatrix, _writer_checks
    from photonhub import __version__

    if version not in (1, 2):
        raise ValueError("Touchstone version must be 1 or 2")
    out_labels, freqs, arr = _prepare_smatrix(S)
    n = len(out_labels)
    if not (z0 > 0.0 and math.isfinite(z0)):
        raise ValueError(f"z0 must be a positive finite resistance, got {z0}")

    path = Path(path)
    expected = f".s{n}p"
    if path.suffix == "":
        path = path.with_suffix(expected if version == 1 else ".ts")
    elif (path.suffix.lower() != expected if version == 1
          else path.suffix.lower() not in (expected, ".ts")):
        raise ValueError(
            f"{path.name!r} does not match the {n}-port Touchstone extension "
            f"{expected!r} or the version 2 extension '.ts'")

    checks = _writer_checks(arr.transpose(2, 0, 1), strict=strict,
                            passivity_atol=passivity_atol,
                            reciprocity_atol=reciprocity_atol)
    if any(not isinstance(key, str) for key in (metadata or {})):
        raise ValueError("metadata keys must be strings")

    def ascii_text(value: str, source: str) -> str:
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"Touchstone {source} must contain only ASCII text") from exc
        return value

    def pair(v: complex) -> str:
        return f"{v.real:.12e} {v.imag:.12e}"

    lines: List[str] = []
    lines.append(f"! {n}-port S-parameters (PhotonHub photonic modal ports)")
    lines.append(f"! photonhub_version: {__version__}")
    lines.append(f"! checks: {checks.summary()}")
    lines.append(f"! note: {PROVENANCE_NOTE}")
    for k, label in enumerate(out_labels, start=1):
        lines.append(f"! port {k} = {ascii_text(label, f'port label {label!r}')}")
    for key, value in sorted((metadata or {}).items()):
        ascii_text(key, f"metadata entry {key!r}")
        value_text = ascii_text(str(value), f"metadata entry {key!r}")
        for piece in value_text.splitlines() or [""]:
            lines.append(f"! {key}: {piece}")
    for c in comments:
        for piece in ascii_text(str(c), f"comment {c!r}").splitlines():
            lines.append(f"! {piece}")
    if version == 2:
        lines.append("[Version] 2.0")
    lines.append(f"# Hz S RI R {z0:g}")
    if version == 2:
        lines.append(f"[Number of Ports] {n}")
        if n == 2:
            lines.append("[Two-Port Data Order] 12_21")
        lines.extend((f"[Number of Frequencies] {len(freqs)}",
                      "[Matrix Format] Full", "[Network Data]"))

    for kf, f in enumerate(freqs):
        Sf = arr[:, :, kf]
        if version == 2:
            for i in range(n):
                row = " ".join(pair(Sf[i, j]) for j in range(n))
                lines.append((f"{f:.12e} " if i == 0 else "  ") + row)
        elif n == 1:
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

    if version == 2:
        lines.append("[End]")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def read_touchstone(path) -> xr.DataArray:
    """Read a Touchstone v1 or v2 S-parameter file.

    Parameters
    ----------
    path:
        Touchstone ``.sNp`` or ``.ts`` file. RI, MA and DB data and Hz, kHz,
        MHz and GHz frequencies are accepted. Version 2 noise data is ignored.

    Returns
    -------
    xarray.DataArray
        Complex matrix with dimensions ``(port_out, port_in, f)`` and frequency
        coordinates in Hz.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8").splitlines()
    comments: List[str] = []
    labels: Dict[int, str] = {}
    keywords: Dict[str, str] = {}
    tokens: List[str] = []
    option = None
    in_data = False
    network_data_seen = False
    version = 1
    for line in raw:
        body, sep, comment = line.partition("!")
        if sep:
            comment = comment.lstrip()
            comments.append(comment)
            match = re.fullmatch(r"port\s+(\d+)\s*=\s*(.*)", comment)
            if match:
                labels.setdefault(int(match.group(1)), match.group(2))
        body = body.strip()
        if not body:
            continue
        if body.startswith("#"):
            option = body[1:].split()
            continue
        keyword = re.fullmatch(r"\[([^]]+)\]\s*(.*)", body)
        if keyword:
            key = keyword.group(1).lower()
            value = keyword.group(2).strip()
            keywords[key] = value
            if key == "version":
                if value != "2.0":
                    raise ValueError(f"unsupported Touchstone version {value}")
                version = 2
            if key == "network data" and not network_data_seen:
                in_data = True
                network_data_seen = True
            elif network_data_seen:
                in_data = False
            continue
        if version == 1 or in_data:
            tokens.extend(body.split())

    unit, parameter, fmt, z0 = "GHZ", "S", "MA", 50.0
    if option:
        remaining = [v.upper() for v in option]
        if "R" in remaining:
            idx = remaining.index("R")
            if idx + 1 >= len(option):
                raise ValueError("Touchstone R option lacks z0")
            if idx + 2 != len(option):
                raise ValueError(f"unsupported Touchstone option after R z0: {option[idx + 2]}")
            z0 = float(option[idx + 1])
            remaining = remaining[:idx]
        if remaining and remaining[0] in ("HZ", "KHZ", "MHZ", "GHZ"):
            unit = remaining.pop(0)
        if remaining and remaining[0] in ("S", "Y", "Z", "H", "G"):
            parameter = remaining.pop(0)
        if remaining and remaining[0] in ("RI", "MA", "DB"):
            fmt = remaining.pop(0)
        if remaining:
            raise ValueError(f"unsupported Touchstone option {remaining[0]}")
    if parameter != "S":
        raise ValueError(f"Touchstone parameter {parameter} is not S")
    scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}.get(unit)
    if scale is None:
        raise ValueError(f"unsupported Touchstone frequency unit {unit}")
    if fmt not in ("RI", "MA", "DB"):
        raise ValueError(f"unsupported Touchstone data format {fmt}")
    if version == 2:
        try:
            n = int(keywords["number of ports"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Touchstone 2.0 needs [Number of Ports]") from exc
        matrix_format = keywords.get("matrix format", "Full")
        if matrix_format.lower() != "full":
            raise ValueError(f"unsupported [Matrix Format] {matrix_format}")
        two_port_order = keywords.get("two-port data order", "21_12")
        if n == 2 and two_port_order not in ("12_21", "21_12"):
            raise ValueError(f"unsupported [Two-Port Data Order] {two_port_order}")
    else:
        suffix = re.fullmatch(r"\.s(\d+)p", path.suffix.lower())
        if suffix is None:
            raise ValueError("Touchstone v1 needs a .sNp extension")
        n = int(suffix.group(1))
        two_port_order = "21_12"
    if n <= 0:
        raise ValueError("Touchstone port count must be positive")
    stride = 1 + 2 * n * n
    if len(tokens) % stride:
        raise ValueError("Touchstone network data has incomplete frequency records")
    nf = len(tokens) // stride
    if version == 2 and "number of frequencies" in keywords:
        if nf != int(keywords["number of frequencies"]):
            raise ValueError("[Number of Frequencies] differs from network data")
    freqs = np.empty(nf, dtype=np.float64)
    arr = np.empty((n, n, nf), dtype=np.complex128)
    for k in range(nf):
        offset = k * stride
        freqs[k] = float(tokens[offset]) * scale
        pairs = []
        for p in range(n * n):
            a = float(tokens[offset + 1 + 2 * p])
            b = float(tokens[offset + 2 + 2 * p])
            if fmt == "RI":
                value = complex(a, b)
            else:
                magnitude = a if fmt == "MA" else 10 ** (a / 20)
                value = magnitude * np.exp(1j * np.deg2rad(b))
            pairs.append(value)
        if n == 2 and two_port_order == "21_12":
            arr[:, :, k] = np.array(pairs)[[0, 2, 1, 3]].reshape(2, 2)
        else:
            arr[:, :, k] = np.array(pairs).reshape(n, n)
    port_names = ([labels[k] for k in range(1, n + 1)]
                  if all(k in labels for k in range(1, n + 1))
                  else [str(k) for k in range(1, n + 1)])
    attrs: Dict[str, Any] = {"version": version, "z0": z0,
                             "comments": comments}
    check_line = next((c[len("checks: "):] for c in comments
                       if c.startswith("checks: ")), None)
    if check_line is not None:
        pattern = (r"passivity_violation=([+\-\d.eE]+) \(atol ([+\-\d.eE]+), "
                   r"(pass|FAIL)\); reciprocity_error=([+\-\d.eE]+) "
                   r"\(atol ([+\-\d.eE]+), (pass|FAIL)\)")
        match = re.fullmatch(pattern, check_line)
        if match:
            attrs["checks"] = {
                "passivity_violation": float(match.group(1)),
                "passivity_atol": float(match.group(2)),
                "passive": match.group(3) == "pass",
                "reciprocity_error": float(match.group(4)),
                "reciprocity_atol": float(match.group(5)),
                "reciprocal": match.group(6) == "pass",
                "num_freqs": nf,
            }
    return xr.DataArray(arr, dims=("port_out", "port_in", "f"),
                        coords={"port_out": port_names, "port_in": port_names,
                                "f": freqs}, attrs=attrs)
