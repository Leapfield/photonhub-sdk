#!/usr/bin/env python3
"""Installed-package smoke test that does not require phsolver."""

import simupod as ph


f0 = 1.934e14
sim = ph.Simulation(
    size_um=(3.0, 3.0, 3.0),
    grid=ph.UniformGridSpec(dl_um=0.06),
    run=ph.RunSpec(n_steps=20),
    sources=[
        ph.PointDipole(
            center_um=(1.5, 1.5, 1.5),
            polarization="Ez",
            source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.5 * f0),
        )
    ],
)
assert ph.Simulation.from_wire_json(sim.to_wire_json()) == sim
assert ph.estimate_cost(sim).num_cells > 0
print(f"simupod {ph.__version__} / schema {ph.SCHEMA_VERSION} smoke passed")
