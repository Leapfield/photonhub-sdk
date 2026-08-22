"""Upfront cost / memory / time estimate for a :class:`Simulation`.

`:meth:`Simulation.cost_estimate` returns a structured planning estimate before
a run. It is pure Python (no solver or GPU required), and every monetary input
is explicit and overridable. It is not a billing quote or published rate card.

The calculation unit is one **Tcell-step** = 1e12 cell-updates =
``num_cells * num_steps / 1e12``. The default planning assumption is
$0.40/Tcell-step and is overridable.

The cell count, ``dt`` and step count are computed to match the engine's
``resolve.cpp`` exactly — the round-half-away-from-zero cell rule
(:func:`photonhub.components.grid.realized_cells`), the §2 uniform Courant
limit and the §15.5 graded (per-axis minimum spacing) limit — so the dollar
figure never drifts from what ``phsolver`` actually runs. It is exact for a
full-duration run; auto-shutoff (NUMERICS.md §7) can only make a run *cheaper*.

Memory and output byte counts are faithful but approximate and each carries a
documented model: resident device memory is dominated by the six fp32 field
arrays plus per-cell Ca/Cb coefficients (the reference solver's bulk
allocation); the DFT-monitor terms surface the otherwise-invisible cost of
volume frequency monitors that the plan calls out explicitly.
"""

from dataclasses import dataclass
from math import ceil, sqrt
from typing import TYPE_CHECKING, Optional, Tuple

from .components.grid import graded_primary_spacings, resolved_cell_counts

if TYPE_CHECKING:  # avoid a runtime import cycle (cost <- simulation <- cost)
    from .components.simulation import Simulation

# Speed of light, identical to the engine's kC0 (engine/include/phcore/types.h)
# so the resolved dt matches bit-for-bit-comparable inputs.
_C0 = 2.99792458e8  # m/s, exact

# Planning-cost unit. One Tcell-step = 1e12 cell-updates.
TCELL = 1.0e12
#: Default planning assumption; not a production rate or billing commitment.
DEFAULT_RATE_USD_PER_TCELL_STEP = 0.40

#: Throughput for the WALL-TIME estimate only (dollars do not depend on it).
#: The conservative Phase-1 acceptance FLOOR (>=20 Gcells/s vacuum on MI300X),
#: not the measured untuned 49.6 — a defensible lower bound so the time figure
#: never over-promises. Realistic dielectric/PML scenes run slower; pass a
#: scene/device-specific value to override.
DEFAULT_THROUGHPUT_GCELLS_PER_S = 20.0

# Resident device memory model (per cell, fp32 words): the six Yee field
# arrays (Ex..Hz) plus per-component Ca and Cb coefficient arrays — the
# reference solver's bulk allocation (engine/src/cpu_ref/reference_solver.cpp:
# ca[3]/cb[3] + the six fields). 6 + 6 = 12 words = 48 B/cell. This is the
# general-media upper bound; a tuned vacuum kernel folds scalar coefficients
# and uses less. CPML psi (boundary slabs only) is a small term and excluded.
_RESIDENT_FP32_WORDS_PER_CELL = 12

# z-ghost planes carried by every field array (NUMERICS.md §1.2, halo_z = 1).
_HALO_Z = 1
# Pitched x-rows: pitch_x = round_up(nx, 32) (fields.h round_up, 128-B rows).
_PITCH = 32


def _round_up(v: int, multiple: int) -> int:
    return ((v + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class CostEstimate:
    """Pre-run estimate for one :class:`Simulation`. Returned by
    :meth:`Simulation.cost_estimate`; ``str(estimate)`` is a human summary."""

    # Geometry and time — engine-faithful (resolve.cpp).
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
    that axis is uniform (UniformGridSpec, or a GradedGridSpec axis omitted
    from ``coords``). Mirrors Simulation._axis_coords_um without importing it."""
    coords = getattr(grid, "coords", None)
    if coords is None:
        return None
    return getattr(coords, "xyz"[axis_index])


def _cells_and_min_spacing_um(sim: "Simulation"):
    """Per-axis (cell count, minimum primary spacing in microns), matching the
    engine: a graded axis has ``len(coords)`` cells (the §15.1 replicate-last
    closing node derives the final cell), a uniform axis has the round-half-away
    cell count and constant spacing ``dl``."""
    dl = sim.grid.dl_um
    counts = list(resolved_cell_counts(sim.size_um, sim.grid))
    min_spacing = []
    for i, length_um in enumerate(sim.size_um):
        q = _axis_coords_um(sim.grid, i)
        if q is None:
            min_spacing.append(dl)
        else:
            min_spacing.append(min(graded_primary_spacings(q)))
    return counts, min_spacing


def _dt_seconds(sim: "Simulation", min_spacing_um) -> float:
    """dt exactly as resolve.cpp computes it: the §2 uniform limit
    ``C*dl/(c0*sqrt(3))`` for a UniformGridSpec, else the §15.5 graded limit
    over per-axis minimum spacings (a graded grid whose axes happen to be
    uniform reduces to the §2 form, as the engine's graded_courant_dt does)."""
    courant = sim.run.courant
    is_graded = getattr(sim.grid, "coords", None) is not None
    if not is_graded:
        dl_m = sim.grid.dl_um * 1e-6
        return courant * dl_m / (_C0 * sqrt(3.0))
    inv_sq = sum(1.0 / (d * 1e-6) ** 2 for d in min_spacing_um)
    return courant / (_C0 * sqrt(inv_sq))


def _num_steps(sim: "Simulation", dt_s: float) -> int:
    """resolve.cpp: an explicit n_steps wins; otherwise ceil(run_time / dt)."""
    if sim.run.n_steps is not None:
        return int(sim.run.n_steps)
    return int(ceil(sim.run.run_time_s / dt_s))


def _region_cells(size_um: Tuple[float, float, float], dl_um: float,
                  cells_per_axis, interval_space=None) -> int:
    """Approximate cell count of a DFT box: ``max(1, round(size/dl))`` per axis
    (1 for a 0-size plane/line/point axis), capped at the domain. Uniform-dl
    approximation; exact snapping is the engine's (NUMERICS.md §12).
    ``interval_space`` (§12 spatial decimation) reduces each axis to
    ``ceil(n/stride)``, mirroring reference_solver.cpp's strided extents."""
    strides = interval_space or (1, 1, 1)
    out = 1
    for a in range(3):
        n = max(1, int(round(size_um[a] / dl_um)))
        n = min(n, cells_per_axis[a])
        out *= (n - 1) // strides[a] + 1
    return out


def _monitor_bytes(sim: "Simulation", num_steps: int, cells_per_axis):
    """(resident DFT/snapshot device bytes, total output bytes) over all
    monitors. Output bytes are the on-disk float32 sizes from the manifest
    shapes (data.py); the DFT resident term is fp64 accumulators (the plan's
    fp64 monitor accumulation), the dominant hidden cost of volume monitors."""
    dl = sim.grid.dl_um
    domain_cells = cells_per_axis[0] * cells_per_axis[1] * cells_per_axis[2]
    resident = 0
    output = 0
    for m in sim.monitors:
        ncomp = len(getattr(m, "fields", ()) or ())
        if m.type == "field_time":
            n_samp = max(1, num_steps // m.interval_steps)
            output += n_samp * ncomp * 4
        elif m.type == "field_snapshot":
            n_samp = 1 if m.interval_steps == 0 else max(1, num_steps // m.interval_steps)
            frame = ncomp * domain_cells * 4
            output += n_samp * frame
            resident += frame  # one staging frame on the device
        elif m.type == "field_dft":
            nfreq = len(m.freqs_hz)
            region = _region_cells(m.size_um, dl, cells_per_axis,
                                   getattr(m, "interval_space", None))
            output += nfreq * ncomp * region * 2 * 4   # [re, im] float32
            resident += nfreq * ncomp * region * 2 * 8  # fp64 accumulators
        elif m.type == "flux":
            output += len(m.freqs_hz) * 4
    return resident, output


def estimate_cost(
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
    dt_s = _dt_seconds(sim, min_spacing)
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
