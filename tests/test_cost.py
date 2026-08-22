"""Pre-run cost / memory / time estimate (photonhub.cost).

The dollar figure is the contract ("estimate in dollars before you press
run"), so these pin it exactly against the engine-faithful cell/dt/step math;
memory and output bytes are checked for the right model and monotonicity, not
to the byte.
"""

import math

import pytest

import photonhub as ph
from photonhub.components.grid import realized_cells
from photonhub.cost import (
    DEFAULT_RATE_USD_PER_TCELL_STEP,
    DEFAULT_THROUGHPUT_GCELLS_PER_S,
    estimate_cost,
)

from .helpers import make_sim

_C0 = 2.99792458e8


def _uniform_dt(dl_um: float, courant: float = 0.99) -> float:
    return courant * dl_um * 1e-6 / (_C0 * math.sqrt(3.0))


class TestBillingExact:
    def test_tcell_steps_and_dollars_are_exact(self):
        # 40 x 40 x 80 = 128k cells, 2000 steps -> 2.56e8 cell-steps.
        sim = make_sim(
            size_um=(2.0, 2.0, 4.0),
            grid=ph.UniformGridSpec(dl_um=0.05),
            run=ph.RunSpec(n_steps=2000),
            monitors=[],
        )
        est = sim.cost_estimate()
        assert est.cells_per_axis == (40, 40, 80)
        assert est.num_cells == 128_000
        assert est.num_steps == 2000
        assert est.tcell_steps == pytest.approx(128_000 * 2000 / 1e12)
        assert est.usd == pytest.approx(
            est.tcell_steps * DEFAULT_RATE_USD_PER_TCELL_STEP)

    def test_dollars_scale_linearly_with_rate(self):
        sim = make_sim(monitors=[])
        base = sim.cost_estimate(rate_usd_per_tcell_step=0.30).usd
        hi = sim.cost_estimate(rate_usd_per_tcell_step=0.50).usd
        assert hi == pytest.approx(base * 0.50 / 0.30)

    def test_dollars_independent_of_throughput(self):
        sim = make_sim(monitors=[])
        a = sim.cost_estimate(throughput_gcells_per_s=5.0)
        b = sim.cost_estimate(throughput_gcells_per_s=50.0)
        assert a.usd == b.usd
        # but wall-time scales inversely with throughput
        assert a.wall_seconds == pytest.approx(b.wall_seconds * 10.0)


class TestEngineFaithfulResolve:
    def test_dt_matches_uniform_courant_formula(self):
        sim = make_sim(grid=ph.UniformGridSpec(dl_um=0.04),
                       run=ph.RunSpec(n_steps=10, courant=0.5), monitors=[])
        assert sim.cost_estimate().dt_s == pytest.approx(_uniform_dt(0.04, 0.5))

    def test_n_steps_from_run_time_is_ceil(self):
        dl = 0.05
        dt = _uniform_dt(dl)
        run_time = 3.5 * dt   # not an integer number of steps
        sim = make_sim(grid=ph.UniformGridSpec(dl_um=dl),
                       run=ph.RunSpec(run_time_s=run_time), monitors=[])
        assert sim.cost_estimate().num_steps == math.ceil(run_time / dt) == 4

    def test_cell_count_uses_round_half_away(self):
        # L/dl = 4.5 -> round-half-away gives 5 (banker's would give 4).
        dl = 0.1
        sim = make_sim(size_um=(0.45, 0.45, 0.45),
                       grid=ph.UniformGridSpec(dl_um=dl),
                       run=ph.RunSpec(n_steps=1),
                       sources=[ph.PointDipole(
                           center_um=(0.2, 0.2, 0.2), polarization="Ez",
                           source_time=ph.GaussianPulse(
                               freq0_hz=1.934e14, fwidth_hz=4e13))],
                       monitors=[])
        n = realized_cells(0.45, dl)
        assert n == 5
        assert sim.cost_estimate().cells_per_axis == (n, n, n)

    def test_cost_estimate_defends_against_unvalidated_oversized_copy(self):
        # model_copy(update=...) is intentionally unvalidated in pydantic.
        # Cost estimation still applies the engine envelope before doing its
        # memory/output arithmetic.
        sim = make_sim(monitors=[]).model_copy(update={
            "size_um": (1291.0, 1291.0, 1291.0),
            "grid": ph.UniformGridSpec(dl_um=1.0),
        })
        with pytest.raises(ValueError, match="beta maximum.*logical cells"):
            estimate_cost(sim)


class TestGraded:
    def _graded_sim(self, n_nodes=80, dl=0.05):
        coords = tuple(dl * i for i in range(n_nodes))
        return make_sim(
            size_um=(2.0, 2.0, dl * n_nodes),
            grid=ph.GradedGridSpec(
                dl_um=dl, coords=ph.GradedAxisCoords(z=coords)),
            run=ph.RunSpec(n_steps=100),
            sources=[ph.PointDipole(
                center_um=(1.0, 1.0, 1.0), polarization="Ez",
                source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4e13))],
            monitors=[])

    def test_graded_axis_cell_count_is_len_coords(self):
        est = self._graded_sim(n_nodes=80).cost_estimate()
        assert est.cells_per_axis[2] == 80

    def test_uniform_spacing_graded_reduces_to_uniform_dt(self):
        # A graded grid whose listed axis is uniformly spaced must give the
        # same dt as the §2 uniform limit (graded_courant_dt reduction).
        est = self._graded_sim(dl=0.05).cost_estimate()
        assert est.dt_s == pytest.approx(_uniform_dt(0.05))

    def test_finer_local_spacing_shrinks_dt(self):
        # A refined patch (smaller min spacing) must reduce the stable dt.
        dl = 0.05
        coords = [0.0]
        while coords[-1] < 4.0 - 1e-9:
            step = 0.01 if 1.0 < coords[-1] < 1.5 else dl  # fine patch
            coords.append(round(coords[-1] + step, 6))
        sim = make_sim(
            size_um=(2.0, 2.0, coords[-1]),
            grid=ph.GradedGridSpec(
                dl_um=dl, coords=ph.GradedAxisCoords(z=tuple(coords))),
            run=ph.RunSpec(n_steps=10),
            sources=[ph.PointDipole(
                center_um=(1.0, 1.0, 1.0), polarization="Ez",
                source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4e13))],
            monitors=[])
        assert sim.cost_estimate().dt_s < _uniform_dt(dl)


class TestMemoryAndOutput:
    def test_memory_grows_with_grid(self):
        coarse = make_sim(grid=ph.UniformGridSpec(dl_um=0.1),
                          run=ph.RunSpec(n_steps=1), monitors=[]).cost_estimate()
        fine = make_sim(grid=ph.UniformGridSpec(dl_um=0.025),
                        run=ph.RunSpec(n_steps=1), monitors=[]).cost_estimate()
        assert fine.device_memory_bytes > coarse.device_memory_bytes

    def test_dft_monitor_adds_output_and_resident_bytes(self):
        base = make_sim(monitors=[]).cost_estimate()
        dft = make_sim(monitors=[ph.FieldDftMonitor(
            name="slab", center_um=(0.1, 0.1, 0.1), size_um=(0.2, 0.2, 0.0),
            fields=["Ex", "Hy"], freqs_hz=[1.934e14, 2.0e14])]).cost_estimate()
        assert dft.output_bytes > base.output_bytes
        assert dft.device_memory_bytes > base.device_memory_bytes

    def test_more_snapshot_samples_grow_output(self):
        few = make_sim(monitors=[ph.FieldSnapshotMonitor(
            name="s", fields=["Ez"], interval_steps=0)],
            run=ph.RunSpec(n_steps=1000)).cost_estimate()
        many = make_sim(monitors=[ph.FieldSnapshotMonitor(
            name="s", fields=["Ez"], interval_steps=10)],
            run=ph.RunSpec(n_steps=1000)).cost_estimate()
        assert many.output_bytes > few.output_bytes


class TestApiSurface:
    def test_defaults_come_from_module_constants(self):
        sim = make_sim(monitors=[])
        est = sim.cost_estimate()
        assert est.rate_usd_per_tcell_step == DEFAULT_RATE_USD_PER_TCELL_STEP
        assert est.throughput_gcells_per_s == DEFAULT_THROUGHPUT_GCELLS_PER_S

    def test_function_and_method_agree(self):
        sim = make_sim(monitors=[])
        assert estimate_cost(sim).usd == sim.cost_estimate().usd

    def test_summary_is_human_readable(self):
        s = str(make_sim(monitors=[]).cost_estimate())
        assert "$" in s and "cells" in s and "Gcells/s" in s

    @pytest.mark.parametrize("rate", [-0.01, -1.0])
    def test_negative_rate_rejected(self, rate):
        with pytest.raises(ValueError):
            make_sim(monitors=[]).cost_estimate(rate_usd_per_tcell_step=rate)

    @pytest.mark.parametrize("tput", [0.0, -5.0])
    def test_nonpositive_throughput_rejected(self, tput):
        with pytest.raises(ValueError):
            make_sim(monitors=[]).cost_estimate(throughput_gcells_per_s=tput)
