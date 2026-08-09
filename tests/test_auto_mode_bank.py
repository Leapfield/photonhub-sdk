"""ModeMonitor's automatic per-frequency reference-mode bank.

The library default readout used to project EVERY monitor frequency onto one
frozen band-centre mode; the GDS benchmark showed the per-frequency Yee bank
(each recorded frequency onto a mode solved AT that frequency) is the faithful
readout, so it is now the default: a monitor built by ``mode_monitor`` from a
dispatcher-solved mode (``mode.solve_params`` provenance) re-solves the mode at
every monitor frequency on first use. Pinned here: the auto-bank triggers only
when eligible, is built exactly once (cached), loses to every explicit choice,
and falls back to the frozen mode (with one warning) when the re-solve fails.
"""
import warnings

import numpy as np
import pytest
import xarray as xr

import photonhub as ph
import photonhub.plugins.kfj_smoothing as kfj
from photonhub.plugins import mode_monitor, solve_mode_on_cross_section
from photonhub.plugins.mode_overlap import vector_modal_fields

C0 = 2.99792458e8
LAM = 1.55
F0 = C0 / (LAM * 1e-6)
DL = 0.05
HC, VC = 0.9, 0.7
WIN = dict(h_center_um=HC, v_center_um=VC, half_w_um=0.75, half_v_um=0.6,
           dl_um=DL)


def _strip():
    L = (1.8, 1.4, 1.0)
    core = ph.Structure(
        geometry=ph.Box(center_um=(L[0] / 2, L[1] / 2, L[2] / 2),
                        size_um=(0.5, 0.22, L[2] * 2)),
        medium=ph.Medium(permittivity=3.4738 ** 2))
    return ph.Simulation(
        size_um=L, grid=ph.UniformGridSpec(dl_um=DL), run={"n_steps": 100},
        background=ph.Background(permittivity=1.444 ** 2), structures=(core,),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"), pml_num_layers=10,
        sources=[ph.PointDipole(center_um=(L[0] / 2, L[1] / 2, 0.5),
                                polarization="Ex",
                                source_time=ph.GaussianPulse(
                                    freq0_hz=F0, fwidth_hz=0.1 * F0))])


FREQS = (0.97 * F0, F0, 1.03 * F0)


def _synthetic_data(mode, name, freqs=FREQS):
    """DFT-shaped data whose every frequency records ``mode``'s own fields."""
    ny, nx = mode.ex.shape
    x = HC + (np.arange(nx) - (nx - 1) / 2.0) * mode.dl_x_um
    y = VC + (np.arange(ny) - (ny - 1) / 2.0) * mode.dl_y_um
    m = vector_modal_fields(mode, x, y, axis="z", direction="+",
                            center_um=(HC, VC))
    per_comp = []
    for comp, arr in (("Ex", m["e1"]), ("Ey", m["e2"]),
                      ("Hx", m["h1"]), ("Hy", m["h2"])):
        a = np.asarray(arr, dtype=np.complex128)
        a = np.broadcast_to(a, (len(freqs), 1, 1) + a.shape).copy()
        per_comp.append(xr.DataArray(
            a, dims=("f", "component", "z", "y", "x"),
            coords={"f": list(freqs), "component": [comp], "z": [0.5],
                    "y": y, "x": x}))
    return {name: xr.concat(per_comp, dim="component")}


@pytest.fixture()
def counted_bank_builder(monkeypatch):
    calls = []
    orig = kfj.mode_bank_on_cross_section

    def counting(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)

    monkeypatch.setattr(kfj, "mode_bank_on_cross_section", counting)
    return calls


def _monitor(sim, mode, **kwargs):
    return mode_monitor(sim, mode, axis="z", position_um=0.5, freqs_hz=FREQS,
                        name="port", **kwargs)


def test_auto_bank_builds_once_and_projects_per_frequency(counted_bank_builder):
    sim = _strip()
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0, **WIN)
    mon = _monitor(sim, mode)
    data = _synthetic_data(mode, mon.name)
    p1 = mon.mode_power(data, destagger_dl=None)  # synthetic plane: no stagger
    p2 = mon.mode_power(data, destagger_dl=None)
    assert len(counted_bank_builder) == 1        # built once, cached
    assert sorted(p1) == sorted(float(f) for f in FREQS)
    assert p1 == p2
    # mode_power returns POWER (P_mode-scaled), so compare against the frozen
    # readout of the same plane: at F0 the bank re-solves the SAME mode as the
    # frozen one (identical reading); off-centre the plane is projected onto
    # the f-solved mode instead, so the reading must move by the (real)
    # profile/n_eff drift the frozen readout cannot see.
    frozen = _monitor(sim, mode, per_freq_modes=False).mode_power(
        data, destagger_dl=None)
    assert p1[F0] == pytest.approx(frozen[F0], rel=1e-6)
    for f in (float(FREQS[0]), float(FREQS[-1])):
        assert p1[f] == pytest.approx(frozen[f], rel=0.2)   # same mode family
        assert abs(p1[f] - frozen[f]) / frozen[f] > 1e-5    # but not frozen


def test_frozen_mode_when_disabled_reads_every_frequency_identically(
        counted_bank_builder):
    sim = _strip()
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0, **WIN)
    mon = _monitor(sim, mode, per_freq_modes=False)
    data = _synthetic_data(mode, mon.name)
    frozen = mon.mode_power(data, destagger_dl=None)
    assert counted_bank_builder == []            # legacy path: no re-solve
    # a plane that IS the frozen mode self-overlaps identically at every
    # frequency — exactly the frequency-blindness the auto-bank removes.
    for f in FREQS:
        assert frozen[float(f)] == pytest.approx(frozen[F0], rel=1e-9)


def test_explicit_modes_by_freq_wins_over_the_auto_bank(counted_bank_builder):
    sim = _strip()
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0, **WIN)
    mon = _monitor(sim, mode)
    data = _synthetic_data(mode, mon.name)
    explicit = {float(f): mode for f in FREQS}
    p = mon.mode_power(data, modes_by_freq=explicit, destagger_dl=None)
    assert counted_bank_builder == []
    # an explicit frozen-per-freq map reproduces the frozen readout exactly
    frozen = _monitor(sim, mode, per_freq_modes=False).mode_power(
        data, destagger_dl=None)
    for f in FREQS:
        assert p[float(f)] == pytest.approx(frozen[float(f)], rel=1e-12)


def test_explicit_n_eff_suppresses_the_auto_bank(counted_bank_builder):
    # An n_eff= override pins the de-stagger/reconstruction index; the auto
    # bank would silently replace it with each bank mode's own n_eff, so it
    # must stand down (regression: the override was being discarded).
    sim = _strip()
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0, **WIN)
    mon = _monitor(sim, mode)
    data = _synthetic_data(mode, mon.name)
    p = mon.mode_power(data, n_eff=float(mode.n_eff), destagger_dl=None)
    assert counted_bank_builder == []
    assert sorted(p) == sorted(float(f) for f in FREQS)


def test_auto_bank_replays_on_the_modes_own_simulation(counted_bank_builder):
    # The bank extends the GIVEN mode's identity: it must re-solve on the
    # simulation the mode was solved on (carried in its provenance), not on
    # whatever simulation the monitor was later built from.
    sim_solve = _strip()
    mode = solve_mode_on_cross_section(sim_solve, "z", 0.5, LAM, "TE", 0,
                                       **WIN)
    sim_monitor = _strip()                       # a different (equal) object
    assert sim_monitor is not sim_solve
    mon = _monitor(sim_monitor, mode)
    mon.mode_power(_synthetic_data(mode, mon.name), destagger_dl=None)
    (args, _kwargs) = counted_bank_builder[0]
    assert args[0] is sim_solve


def test_no_provenance_means_no_auto_bank(counted_bank_builder):
    import dataclasses

    sim = _strip()
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0, **WIN)
    bare = dataclasses.replace(mode, solve_params=None)
    mon = _monitor(sim, bare)
    mon.mode_power(_synthetic_data(bare, mon.name), destagger_dl=None)
    assert counted_bank_builder == []


def test_single_frequency_at_the_solve_wavelength_skips_the_bank(
        counted_bank_builder):
    sim = _strip()
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0, **WIN)
    mon = mode_monitor(sim, mode, axis="z", position_um=0.5, freqs_hz=(F0,),
                       name="port")
    mon.mode_power(_synthetic_data(mode, mon.name, freqs=(F0,)),
                   destagger_dl=None)
    assert counted_bank_builder == []            # a 1-entry bank == the mode


def test_failed_bank_warns_once_and_falls_back_to_the_frozen_mode(monkeypatch):
    sim = _strip()
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0, **WIN)

    def boom(*a, **k):
        raise RuntimeError("band edge past cutoff")

    monkeypatch.setattr(kfj, "mode_bank_on_cross_section", boom)
    mon = _monitor(sim, mode)
    data = _synthetic_data(mode, mon.name)
    with pytest.warns(UserWarning, match="falling back to the frozen"):
        p = mon.mode_power(data, destagger_dl=None)
    frozen = _monitor(sim, mode, per_freq_modes=False).mode_power(
        data, destagger_dl=None)
    for f in FREQS:                              # frozen-mode fallback
        assert p[float(f)] == pytest.approx(frozen[float(f)], rel=1e-12)
    with warnings.catch_warnings():              # cached: warn ONCE, not per read
        warnings.simplefilter("error")
        assert mon.mode_power(data, destagger_dl=None) == p
