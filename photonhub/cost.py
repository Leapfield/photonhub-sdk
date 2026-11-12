"""Upfront cost / memory / time estimate for a :class:`Simulation`.

`:meth:`Simulation.cost_estimate` returns a structured planning estimate before
a run. It is pure Python (no solver or GPU required), and every monetary input
is explicit and overridable. It is not a billing quote or published rate card.

The calculation unit is one **Tcell-step** = 1e12 cell-updates =
``num_cells * num_steps / 1e12``. The default planning assumption is
$0.20/Tcell-step, the beta rate, and is overridable.

The cell count, ``dt`` and step count are computed to match the engine's
exactly, the round-half-away-from-zero cell rule
(:func:`photonhub.components.grid.realized_cells`), the uniform Courant
limit and the graded (per-axis minimum spacing) limit, so the dollar
figure never drifts from what ``phsolver`` actually runs. It is exact for a
full-duration run; auto-shutoff can only make a run *cheaper*.

The output byte count is the size of the monitor files the solver writes for
a full-duration run, to the byte: each frequency monitor's box on the grid
nodes the solver records it on, and each time or snapshot monitor's sample
count. An early stop writes no more, so it is an upper bound. The cloud
service sizes the result it accepts from this figure. Resident device memory
is an approximate model: the six fp32 field arrays plus per-cell Ca/Cb
coefficients (the reference solver's bulk allocation) and the frequency
monitors' fp64 accumulators, the otherwise-invisible cost of volume frequency
monitors.
"""

from dataclasses import dataclass
from math import ceil, sqrt
from typing import TYPE_CHECKING, Optional, Tuple

from .components.grid import (
    _YEE_OFFSETS_BY_COMPONENT,
    EngineSnap,
    graded_primary_spacings,
    resolved_cell_counts,
)
from .constants import c0

if TYPE_CHECKING:  # avoid a runtime import cycle (cost <- simulation <- cost)
    from .components.simulation import Simulation

# Speed of light, identical to the engine's kC0
# so the resolved dt matches bit-for-bit-comparable inputs.
_C0 = c0  # m/s, exact

# Planning-cost unit. One Tcell-step = 1e12 cell-updates.
TCELL = 1.0e12
#: Default planning assumption, the beta rate; not a production rate or
#: billing commitment. The service's quote is the price.
DEFAULT_RATE_USD_PER_TCELL_STEP = 0.20

#: Throughput for the WALL-TIME estimate only (dollars do not depend on it).
#: The conservative Phase-1 acceptance FLOOR (>=20 Gcells/s vacuum on the cloud GPU),
#: not the measured untuned 49.6 — a defensible lower bound so the time figure
#: never over-promises. Realistic dielectric/PML scenes run slower; pass a
#: scene/device-specific value to override.
DEFAULT_THROUGHPUT_GCELLS_PER_S = 20.0

# Resident device memory model (per cell, fp32 words): the six Yee field
# arrays (Ex..Hz) plus per-component Ca and Cb coefficient arrays — the
# reference solver's bulk allocation (ca[3]/cb[3] + the six fields). 6 + 6 = 12 words = 48 B/cell. This is the
# general-media upper bound; a tuned vacuum kernel folds scalar coefficients
# and uses less. CPML psi (boundary slabs only) is a small term and excluded.
_RESIDENT_FP32_WORDS_PER_CELL = 12

# z-ghost planes carried by every field array (halo_z = 1).
_HALO_Z = 1
# Pitched x-rows: pitch_x = round_up(nx, 32) (128-B rows).
_PITCH = 32


def _round_up(v: int, multiple: int) -> int:
    return ((v + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class CostEstimate:
    """Pre-run estimate for one :class:`Simulation`. Returned by
    :meth:`Simulation.cost_estimate`; ``str(estimate)`` is a human summary."""

    # Geometry and time — engine-faithful.
    cells_per_axis: Tuple[int, int, int]
    num_cells: int
    num_steps: int
    dt_s: float
    # Planning cost — exact for a full-duration run at the supplied rate.
    tcell_steps: float
    rate_usd_per_tcell_step: float
    usd: float
    # Memory (resident, device) and output (disk / egress), bytes. Approximate.
    device_memory_bytes: int
    output_bytes: int
    # Wall-time at the assumed throughput.
    throughput_gcells_per_s: float
    wall_seconds: float

    def summary(self) -> str:
        """Format the estimated price, grid, memory, output size, and wall time.

        Wall time assumes the full duration and the configured throughput.
        Auto-shutoff may finish sooner. This method makes no service request."""
        nx, ny, nz = self.cells_per_axis
        return (
            f"cost estimate: ${self.usd:,.2f} "
            f"({self.tcell_steps:.4g} Tcell-steps @ "
            f"${self.rate_usd_per_tcell_step:g}/Tcell-step)\n"
            f"  grid     : {nx} x {ny} x {nz} = {self.num_cells:,} cells, "
            f"{self.num_steps:,} steps (dt = {self.dt_s:.4g} s)\n"
            f"  memory   : {_fmt_bytes(self.device_memory_bytes)} resident "
            f"(device), {_fmt_bytes(self.output_bytes)} output\n"
            f"  walltime : ~{_fmt_seconds(self.wall_seconds)} "
            f"@ {self.throughput_gcells_per_s:g} Gcells/s "
            "(full duration; auto-shutoff may finish sooner)"
        )

    def __str__(self) -> str:
        return self.summary()


def _axis_coords_um(grid, axis_index: int) -> Optional[Tuple[float, ...]]:
    """Graded primary-node coordinates (microns) for an axis, or None when
    that axis is uniform (UniformMesh, or a GradedMesh axis omitted
    from ``coords``). Mirrors Simulation._axis_coords_um without importing it."""
    coords = getattr(grid, "coords", None)
    if coords is None:
        return None
    return getattr(coords, "xyz"[axis_index])


def _cells_and_min_spacing_um(sim: "Simulation"):
    """Per-axis (cell count, minimum primary spacing in microns), matching the
    engine: a graded axis has ``len(coords)`` cells (the replicate-last
    closing node derives the final cell), a uniform axis has the round-half-away
    cell count and constant spacing ``dl`` (with the boundary-aware floor ,
    a plain periodic axis may be a single cell)."""
    dl = sim.grid.dl_um
    counts = list(
        resolved_cell_counts(sim.size_um, sim.grid, sim._axis_min_cells())
    )
    min_spacing = []
    for i, length_um in enumerate(sim.size_um):
        q = _axis_coords_um(sim.grid, i)
        if q is None:
            min_spacing.append(dl)
        else:
            min_spacing.append(min(graded_primary_spacings(q)))
    return counts, min_spacing


def _dt_seconds(sim: "Simulation", counts, min_spacing_um) -> float:
    """dt exactly as the engine computes it: the uniform limit
    ``C*dl/(c0*sqrt(D))`` over the D ACTIVE axes (n > 1; a 1-cell plain
    periodic axis carries no curl term and leaves the Courant sum) for a
    UniformMesh, else the graded limit over the active axes' minimum
    spacings (a graded grid whose axes happen to be uniform reduces to the uniform
    form, as the engine's graded Courant limit does)."""
    courant = sim.run.courant
    is_graded = getattr(sim.grid, "coords", None) is not None
    if not is_graded:
        dl_m = sim.grid.dl_um * 1e-6
        active = sum(1 for n in counts if n > 1) or 1
        return courant * dl_m / (_C0 * sqrt(float(active)))
    inv_sq = sum(
        1.0 / (d * 1e-6) ** 2
        for n, d in zip(counts, min_spacing_um)
        if n > 1
    )
    if inv_sq == 0.0:
        inv_sq = 1.0 / (sim.grid.dl_um * 1e-6) ** 2
    return courant / (_C0 * sqrt(inv_sq))


def _num_steps(sim: "Simulation", dt_s: float) -> int:
    """As the engine resolves it: an explicit n_steps wins; otherwise ceil(run_time / dt)."""
    if sim.run.n_steps is not None:
        return int(sim.run.n_steps)
    return int(ceil(sim.run.run_time_s / dt_s))


def _dft_region_cells(monitor, snap: EngineSnap) -> int:
    """Recorded cell count of a ``field_dft`` box as the solver sizes it: both
    corners ``center -+ size/2`` snapped to the nearest grid node of the
    first listed field component (the solver requires every listed component
    to give the same box), ``hi - lo + 1`` per axis, then ``interval_space``
    keeps ``(n - 1) // stride + 1``. The corners are formed in metres from
    the scaled centre and size, in the solver's order: a face on an exact
    tie between two nodes rounds differently when formed in microns first.
    On a graded grid this counts the fine cells the box spans: ``size / dl``
    with the base ``dl`` undercounted a refined axis (3.4x on a 76.5 M-cell
    graded coupler), and the cloud service bounds a run's result by this
    estimate."""
    fields = getattr(monitor, "fields", ()) or ()
    if not fields:
        return 0
    offsets = _YEE_OFFSETS_BY_COMPONENT[fields[0]]
    strides = getattr(monitor, "interval_space", None) or (1, 1, 1)
    out = 1
    for a in range(3):
        c = float(monitor.center_um[a]) * 1e-6
        half = float(monitor.size_um[a]) * 1e-6 / 2.0
        lo = snap.node_m(a, c - half, offsets[a])
        hi = snap.node_m(a, c + half, offsets[a])
        out *= (max(hi - lo, 0)) // strides[a] + 1
    return out


def _time_samples(num_steps: int, interval_steps: int) -> int:
    """Samples a time or snapshot monitor records over ``num_steps`` steps:
    every ``interval_steps``-th step plus the final step when the cadence
    misses it, so ``ceil(num_steps / K)``; ``K = 0`` (a snapshot's default)
    records the final step only. Exact for a full-duration run; a run that
    stops early records no more."""
    if interval_steps == 0:
        return 1
    return max(1, -(-num_steps // interval_steps))


def _monitor_bytes(sim: "Simulation", num_steps: int, cells_per_axis):
    """(resident DFT/snapshot device bytes, total output bytes) over all
    monitors. Output bytes are the monitor files the solver writes for a
    full-duration run, to the byte (an early stop writes no more); the DFT
    resident term is fp64 accumulators (the plan's
    fp64 monitor accumulation), the dominant hidden cost of volume monitors."""
    domain_cells = cells_per_axis[0] * cells_per_axis[1] * cells_per_axis[2]
    snap = None  # built on the first field_dft monitor
    resident = 0
    output = 0
    for m in sim.monitors:
        kind = getattr(m, "type", None)
        if kind is None:
            # A plugin WRAPPER (ModeMonitor and friends) carries the real
            # monitor on .field_monitor; Simulation.model_copy(update=...)
            # does not re-validate, so a wrapper passed by mistake reaches
            # here and used to die on a bare AttributeError far from the
            # cause. Say what actually happened.
            hint = (" — add its .field_monitor to the simulation"
                    if hasattr(m, "field_monitor") else "")
            raise TypeError(
                f"monitor {getattr(m, 'name', m)!r} is a "
                f"{type(m).__name__}, not a wire monitor{hint}")
        ncomp = len(getattr(m, "fields", ()) or ())
        if kind == "field_time":
            n_samp = _time_samples(num_steps, m.interval_steps)
            output += n_samp * ncomp * 4
        elif kind == "field_snapshot":
            n_samp = _time_samples(num_steps, m.interval_steps)
            frame = ncomp * domain_cells * 4
            output += n_samp * frame
            resident += frame  # one staging frame on the device
        elif kind == "field_dft":
            nfreq = len(m.freqs_hz)
            if snap is None:
                snap = EngineSnap(sim)
            region = _dft_region_cells(m, snap)
            output += nfreq * ncomp * region * 2 * 4   # [re, im] float32
            resident += nfreq * ncomp * region * 2 * 8  # fp64 accumulators
        elif kind == "flux":
            output += len(m.freqs_hz) * 4
    return resident, output


def quote(
    sim: "Simulation",
    *,
    rate_usd_per_tcell_step: float = DEFAULT_RATE_USD_PER_TCELL_STEP,
    throughput_gcells_per_s: float = DEFAULT_THROUGHPUT_GCELLS_PER_S,
) -> CostEstimate:
    """Estimate dollars, device memory, output size and wall-time for ``sim``.

    See the module docstring for the model. ``rate_usd_per_tcell_step`` is an
    explicit planning assumption; ``throughput`` affects only the wall-time
    estimate, never the dollar figure."""
    if rate_usd_per_tcell_step < 0:
        raise ValueError("rate_usd_per_tcell_step must be non-negative")
    if throughput_gcells_per_s <= 0:
        raise ValueError("throughput_gcells_per_s must be positive")

    counts, min_spacing = _cells_and_min_spacing_um(sim)
    num_cells = counts[0] * counts[1] * counts[2]
    dt_s = _dt_seconds(sim, counts, min_spacing)
    num_steps = _num_steps(sim, dt_s)

    cell_steps = num_cells * num_steps
    tcell_steps = cell_steps / TCELL
    usd = tcell_steps * rate_usd_per_tcell_step

    padded_cells = (_round_up(counts[0], _PITCH) * counts[1]
                    * (counts[2] + 2 * _HALO_Z))
    field_coeff_bytes = padded_cells * _RESIDENT_FP32_WORDS_PER_CELL * 4
    mon_resident, output_bytes = _monitor_bytes(sim, num_steps, counts)
    device_memory_bytes = field_coeff_bytes + mon_resident

    wall_seconds = cell_steps / (throughput_gcells_per_s * 1e9)

    return CostEstimate(
        cells_per_axis=(counts[0], counts[1], counts[2]),
        num_cells=num_cells,
        num_steps=num_steps,
        dt_s=dt_s,
        tcell_steps=tcell_steps,
        rate_usd_per_tcell_step=rate_usd_per_tcell_step,
        usd=usd,
        device_memory_bytes=device_memory_bytes,
        output_bytes=output_bytes,
        throughput_gcells_per_s=throughput_gcells_per_s,
        wall_seconds=wall_seconds,
    )


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if x < 1024.0 or unit == "GiB":
            return f"{x:.1f} {unit}" if unit != "B" else f"{int(x)} B"
        x /= 1024.0
    return f"{x:.1f} GiB"


def _fmt_seconds(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.0f} us"
    if s < 1.0:
        return f"{s * 1e3:.0f} ms"
    if s < 60.0:
        return f"{s:.1f} s"
    if s < 3600.0:
        return f"{s / 60.0:.1f} min"
    return f"{s / 3600.0:.2f} h"
