"""S-matrix driver — orchestration, assembly plumbing, and Touchstone format.

No engine and no overlap physics here: the per-column amplitude bookkeeping is
pinned by ``test_smatrix.py`` on synthetic planes, and the end-to-end physics
gate (straight guide |S21|^2 ~ 1, reciprocity) lives in
``validation/test_tier2b_smatrix_driver.py`` behind the engine gate. These
tests pin what the DRIVER adds on top:

* port declaration validation,
* the generated driven simulations (sources placed outside the port plane,
  launched inward; every port monitored; drive subsets; base monitors
  dropped/kept),
* run orchestration (runner injection, partial-failure semantics, column ->
  matrix assembly with a stable port order),
* the Touchstone v1 writer (2-port column-major special case, n-port row-major
  wrapping, suffix enforcement, NaN rejection).
"""

import math

import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins import (SMatrixPort, assemble_smatrix, plan_smatrix,
                               run_smatrix, write_touchstone)
from photonhub.plugins import smatrix_driver as drv

C0 = 2.99792458e8
WL_UM = 1.55
F0 = C0 / (WL_UM * 1e-6)
FREQS = (0.98 * F0, 1.02 * F0)

N_SI, N_SIO2 = 3.48, 1.444
DL = 0.05
SX, SY, SZ = 1.6, 1.2, 3.0
XC, YC = SX / 2, SY / 2
Z_P1, Z_P2 = 0.9, 2.1
PML_LAYERS = 8            # 0.4 um slab at DL

WINDOW = dict(half_w_um=0.55, half_v_um=0.45)


def _base_sim(**overrides):
    si = ph.Medium(permittivity=N_SI**2)
    kwargs = dict(
        size_um=(SX, SY, SZ),
        grid=ph.UniformGridSpec(dl_um=DL),
        run=ph.RunSpec(n_steps=4000),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=PML_LAYERS,
        background=ph.Background(permittivity=N_SIO2**2),
        structures=[
            ph.Structure(
                geometry=ph.Box(center_um=(XC, YC, SZ / 2),
                                size_um=(0.5, 0.3, SZ)),
                medium=si),
        ],
        # Placeholder — every driven copy replaces it.
        sources=[ph.PointDipole(
            center_um=(XC, YC, SZ / 2), polarization="Ex",
            source_time=ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0))],
        monitors=[ph.FieldTimeMonitor(
            name="base_probe", center_um=(XC, YC, SZ / 2), fields=["Ex"])],
    )
    kwargs.update(overrides)
    return ph.Simulation(**kwargs)


def _ports(**overrides):
    common = {**WINDOW, "source_offset_um": 0.3, **overrides}
    return [
        SMatrixPort("P1", "z", Z_P1, "-", **common),
        SMatrixPort("P2", "z", Z_P2, "+", **common),
    ]


@pytest.fixture(scope="module")
def two_port_plan():
    sim = _base_sim()
    return plan_smatrix(sim, _ports(), freqs_hz=FREQS)


# ---------------------------------------------------------------------------
# Port declaration validation
# ---------------------------------------------------------------------------

def test_port_rejects_bad_fields():
    ok = dict(half_w_um=0.5, half_v_um=0.4)
    with pytest.raises(ValueError, match="axis"):
        SMatrixPort("P", "w", 1.0, "+", **ok)
    with pytest.raises(ValueError, match="out_direction"):
        SMatrixPort("P", "z", 1.0, "out", **ok)
    with pytest.raises(ValueError, match="polarization"):
        SMatrixPort("P", "z", 1.0, "+", polarization="TEM", **ok)
    with pytest.raises(ValueError, match="mode_index"):
        SMatrixPort("P", "z", 1.0, "+", mode_index=-1, **ok)
    with pytest.raises(ValueError, match="half_w_um"):
        SMatrixPort("P", "z", 1.0, "+", half_w_um=0.0, half_v_um=0.4)
    with pytest.raises(ValueError, match="source_offset_um"):
        SMatrixPort("P", "z", 1.0, "+", source_offset_um=0.0, **ok)
    # batch/output-directory safety
    with pytest.raises(ValueError, match="name"):
        SMatrixPort("a/b", "z", 1.0, "+", **ok)
    # normalization + derived direction
    p = SMatrixPort("P", "z", 1.0, "-", polarization="te", **ok)
    assert p.polarization == "TE"
    assert p.in_direction == "+"


def test_plan_rejects_duplicate_and_unknown_ports():
    sim = _base_sim()
    dup = [SMatrixPort("P", "z", Z_P1, "-", **WINDOW),
           SMatrixPort("P", "z", Z_P2, "+", **WINDOW)]
    with pytest.raises(ValueError, match="duplicate"):
        plan_smatrix(sim, dup, freqs_hz=FREQS)
    with pytest.raises(ValueError, match="drive names"):
        plan_smatrix(sim, _ports(), freqs_hz=FREQS, drive=["nope"])
    with pytest.raises(ValueError, match="non-empty"):
        plan_smatrix(sim, [], freqs_hz=FREQS)
    with pytest.raises(ValueError, match="freqs_hz"):
        plan_smatrix(sim, _ports(), freqs_hz=[])


# ---------------------------------------------------------------------------
# Generated simulations
# ---------------------------------------------------------------------------

def _source_positions_um(sim, axis_idx=2):
    return sorted({s.center_um[axis_idx] for s in sim.sources})


def test_plan_builds_one_driven_sim_per_port(two_port_plan):
    plan = two_port_plan
    assert sorted(plan.simulations) == ["P1", "P2"]
    assert [p.name for p in plan.sports] == ["P1", "P2"]
    assert plan.freqs_hz == tuple(sorted(FREQS))

    for name, sim in plan.simulations.items():
        # every port plane monitored in every drive, base monitors dropped
        assert [m.name for m in sim.monitors] == ["P1", "P2"]
        for m in sim.monitors:
            assert m.freqs_hz == plan.freqs_hz
        # the placeholder dipole is gone, replaced by the launch
        assert len(sim.sources) > 1

    # P1's out direction is '-', so its launch sits at z = Z_P1 - 0.3 and
    # fires '+' (inward); P2 mirrors. The eq-current sheet staggers J/M a
    # half-cell along z, so allow one cell of slack around the launch plane.
    z1 = _source_positions_um(plan.simulations["P1"])
    assert all(abs(z - (Z_P1 - 0.3)) <= DL for z in z1)
    z2 = _source_positions_um(plan.simulations["P2"])
    assert all(abs(z - (Z_P2 + 0.3)) <= DL for z in z2)

    # readers agree with the declaration
    assert plan.sports[0].out_direction == "-"
    assert plan.sports[0].in_direction == "+"
    assert plan.sports[1].out_direction == "+"


def test_plan_drive_subset_and_keep_monitors():
    sim = _base_sim()
    plan = plan_smatrix(sim, _ports(), freqs_hz=FREQS, drive=["P2"],
                        keep_monitors=True)
    assert sorted(plan.simulations) == ["P2"]
    # both ports still monitored, plus the base sim's own probe
    names = [m.name for m in plan.simulations["P2"].monitors]
    assert names == ["P1", "P2", "base_probe"]


def test_plan_warns_when_launch_lands_in_pml():
    sim = _base_sim()
    # offset pushes the P1 launch to z = 0.9 - 0.6 = 0.3 < 0.4 (the PML slab)
    ports = _ports(source_offset_um=0.6)
    with pytest.warns(UserWarning, match="pml slab"):
        plan_smatrix(sim, ports, freqs_hz=FREQS)


def test_plan_rejects_launch_outside_domain():
    sim = _base_sim()
    ports = [SMatrixPort("P1", "z", Z_P1, "-", source_offset_um=1.0, **WINDOW),
             SMatrixPort("P2", "z", Z_P2, "+", source_offset_um=1.0, **WINDOW)]
    with pytest.raises(ValueError, match="outside the domain"):
        plan_smatrix(sim, ports, freqs_hz=FREQS)


def test_default_source_time_covers_band(two_port_plan):
    st = two_port_plan.source_time
    lo, hi = min(FREQS), max(FREQS)
    assert lo <= st.freq0_hz <= hi
    assert st.fwidth_hz > 0


# ---------------------------------------------------------------------------
# Run orchestration (runner injection; no engine)
# ---------------------------------------------------------------------------

def _stub_columns(monkeypatch, table):
    """Replace the per-run column extraction with a lookup: driven name ->
    {(out, in): {f: S}}."""
    def fake_column(sports, driven, data, colocate=True):
        assert colocate is False  # the tests pass colocate=False through
        return table[driven]
    monkeypatch.setattr(drv, "smatrix", fake_column)


def _column_table():
    s21 = {f: 0.9 * np.exp(1j * 0.3) for f in sorted(FREQS)}
    s12 = {f: 0.9 * np.exp(1j * 0.3) for f in sorted(FREQS)}
    s11 = {f: 0.1 + 0j for f in sorted(FREQS)}
    s22 = {f: 0.1 + 0j for f in sorted(FREQS)}
    return {
        "P1": {("P1", "P1"): s11, ("P2", "P1"): s21},
        "P2": {("P1", "P2"): s12, ("P2", "P2"): s22},
    }


def test_run_assembles_full_matrix(two_port_plan, monkeypatch):
    plan = two_port_plan
    _stub_columns(monkeypatch, _column_table())

    seen = {}

    def runner(sims):
        seen.update(sims)
        return {name: {"marker": name} for name in sims}

    result = plan.run(runner, colocate=False)
    assert sorted(seen) == ["P1", "P2"]
    assert seen == dict(plan.simulations)

    assert list(result.S.coords["port_out"].values) == ["P1", "P2"]
    s21 = np.asarray(result.sij("P2", "P1").values)
    assert np.allclose(np.abs(s21) ** 2, 0.81)
    assert result.is_reciprocal(atol=1e-9)
    assert result.is_passive(atol=1e-9)
    assert result.reciprocity_error() < 1e-12
    assert not result.errors
    text = result.summary()
    assert "|S[P2,P1]|^2" in text and "reciprocity" in text


def test_run_partial_failure_semantics(two_port_plan, monkeypatch):
    plan = two_port_plan
    _stub_columns(monkeypatch, _column_table())

    def runner(sims):
        return {"P1": {"marker": "P1"}}  # P2 never produced data

    with pytest.raises(RuntimeError, match="P2"):
        plan.run(runner, colocate=False)

    result = plan.run(runner, colocate=False, allow_partial=True)
    assert sorted(result.errors) == ["P2"]
    # the undriven column is NaN, the driven one intact
    assert np.isnan(result.sij("P1", "P2").values).all()
    assert np.allclose(np.abs(result.sij("P2", "P1").values) ** 2, 0.81)
    with pytest.raises(ValueError, match="non-finite"):
        result.to_touchstone("partial")


def test_run_rejects_bad_runner_and_web_path_dir(two_port_plan):
    with pytest.raises(ValueError, match="runner"):
        two_port_plan.run("bogus")
    with pytest.raises(ValueError, match="path_dir"):
        two_port_plan.run("web", path_dir="somewhere")


def test_run_smatrix_one_call(monkeypatch):
    monkeypatch.setattr(drv, "smatrix",
                        lambda sports, driven, data, colocate=True:
                        _column_table()[driven])
    result = run_smatrix(
        _base_sim(), _ports(), freqs_hz=FREQS,
        runner=lambda sims: {n: {"marker": n} for n in sims})
    assert np.allclose(np.abs(result.sij("P2", "P1").values) ** 2, 0.81)


# ---------------------------------------------------------------------------
# Touchstone writer
# ---------------------------------------------------------------------------

def _matrix(n, freqs, fill):
    """Assemble an n-port S DataArray with S_ij(f) = fill(i, j, f)."""
    cols = []
    for j in range(n):
        col = {}
        for i in range(n):
            col[(f"P{i+1}", f"P{j+1}")] = {
                f: fill(i, j, f) for f in freqs}
        cols.append(col)
    return assemble_smatrix(cols, port_order=[f"P{k+1}" for k in range(n)])


def _read_touchstone(path):
    lines = [ln for ln in path.read_text().splitlines()
             if ln.strip() and not ln.startswith("!")]
    assert lines[0].startswith("# Hz S RI R ")
    return lines[0], [ln for ln in lines[1:]]


def test_touchstone_two_port_column_major(tmp_path):
    freqs = [1.0e14, 2.0e14]
    S = _matrix(2, freqs, lambda i, j, f: complex(i + 1, j + 1) * f / 1e14)
    path = write_touchstone(S, tmp_path / "dev")
    assert path.name == "dev.s2p"

    header, data = _read_touchstone(path)
    assert header.split()[-1] == "50"
    assert len(data) == 2  # one line per frequency
    row = data[0].split()
    assert float(row[0]) == pytest.approx(1.0e14)
    # v1 2-port order: f S11 S21 S12 S22 -> S21 occupies tokens 3,4
    vals = [complex(float(row[k]), float(row[k + 1]))
            for k in range(1, 9, 2)]
    assert vals[0] == pytest.approx(complex(1, 1))   # S11
    assert vals[1] == pytest.approx(complex(2, 1))   # S21 (out=2, in=1)
    assert vals[2] == pytest.approx(complex(1, 2))   # S12
    assert vals[3] == pytest.approx(complex(2, 2))   # S22


def test_touchstone_one_port_and_sorting(tmp_path):
    # unsorted input frequencies come out ascending
    freqs = [2.0e14, 1.0e14]
    S = _matrix(1, freqs, lambda i, j, f: complex(f / 1e14, 0))
    path = write_touchstone(S, tmp_path / "refl", z0=25.0)
    assert path.name == "refl.s1p"
    header, data = _read_touchstone(path)
    assert header.split()[-1] == "25"
    assert [float(ln.split()[0]) for ln in data] == [1.0e14, 2.0e14]
    assert float(data[0].split()[1]) == pytest.approx(1.0)


def test_touchstone_five_port_row_major_wrapping(tmp_path):
    freqs = [1.0e14]
    S = _matrix(5, freqs, lambda i, j, f: complex(i + 1, j + 1))
    path = write_touchstone(S, tmp_path / "star")
    assert path.name == "star.s5p"
    header, data = _read_touchstone(path)
    # 5 ports -> each row wraps to 4 + 1 pairs = 2 lines; 5 rows -> 10 lines
    assert len(data) == 10
    first = data[0].split()
    assert float(first[0]) == pytest.approx(1.0e14)
    assert len(first) == 1 + 8            # freq + 4 pairs
    assert len(data[1].split()) == 2      # the wrapped 5th pair of row 1
    # row-major: line 3 starts row 2 (S21 = 2 + 1j), no freq prefix
    row2 = data[2].split()
    assert len(row2) == 8
    assert complex(float(row2[0]), float(row2[1])) == pytest.approx(
        complex(2, 1))


def test_touchstone_suffix_and_label_guards(tmp_path):
    S2 = _matrix(2, [1.0e14], lambda i, j, f: 0.5 + 0j)
    with pytest.raises(ValueError, match="extension"):
        write_touchstone(S2, tmp_path / "dev.s3p")
    with pytest.raises(ValueError, match="z0"):
        write_touchstone(S2, tmp_path / "dev", z0=0.0)
    # mismatched port labels between the two axes are rejected
    import xarray as xr
    bad = xr.DataArray(
        np.zeros((1, 1, 1), dtype=complex),
        dims=("port_out", "port_in", "f"),
        coords={"port_out": ["A"], "port_in": ["B"], "f": [1.0e14]})
    with pytest.raises(ValueError, match="labels differ"):
        write_touchstone(bad, tmp_path / "bad")
