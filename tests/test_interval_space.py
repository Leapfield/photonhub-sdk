"""FieldDftMonitor.interval_space — per-axis spatial sampling stride (the
Tidy3D interval_space; NUMERICS.md §12, schema 1.11.0). Additive/optional:
omitted from the wire when unset (older engines/readers round-trip unchanged),
strides per axis when set. The engine sampling is verified on the GPU box; the
coordinate-striding readback is covered in test_data.py."""

import pytest
from pydantic import ValidationError

import photonhub as ph
from photonhub.components.monitors import FieldDftMonitor

F0 = 1.934e14


def _mon(**kw):
    base = dict(
        name="m", center_um=(1.0, 1.0, 1.0), size_um=(2.0, 2.0, 0.0),
        fields=("Ex", "Hy"), freqs_hz=(F0,),
    )
    base.update(kw)
    return FieldDftMonitor(**base)


def test_interval_space_defaults_none():
    assert _mon().interval_space is None


def test_interval_space_accepts_strides():
    assert _mon(interval_space=(8, 2, 2)).interval_space == (8, 2, 2)


def test_interval_space_rejects_below_one():
    with pytest.raises(ValidationError, match=">= 1"):
        _mon(interval_space=(8, 0, 2))


def test_interval_space_omitted_from_wire_when_unset():
    js = _mon().model_dump(by_alias=True, exclude_none=True)
    assert "interval_space" not in js


def test_interval_space_carried_when_set():
    js = _mon(interval_space=(8, 2, 2)).model_dump(mode="json", by_alias=True,
                                                   exclude_none=True)
    assert js["interval_space"] == [8, 2, 2]


def test_interval_space_roundtrips_in_simulation_wire():
    sim = ph.Simulation(
        size_um=(2.0, 2.0, 2.0),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=10),
        sources=[ph.PointDipole(center_um=(1.0, 1.0, 1.0),
                                polarization="Ey",
                                source_time=ph.GaussianPulse(freq0_hz=F0,
                                                             fwidth_hz=2e13))],
        monitors=[_mon(interval_space=(4, 2, 1))],
    )
    restored = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert restored.monitors[0].interval_space == (4, 2, 1)
    # Unset monitor stays unset across the wire (byte-clean back-compat).
    sim2 = sim.model_copy(update={"monitors": (_mon(),)})
    assert ph.Simulation.from_wire_json(
        sim2.to_wire_json()).monitors[0].interval_space is None
