"""mode_launch — the library's default launch: eq-current when eligible, §18
otherwise. Returns a LIST of sources (the sheet is many dipoles, §18 is one),
picks the mechanism from the mode + grid + bandwidth, and recovers the sheet's
window from the mode's OWN recorded placement so it registers on the solve grid
without re-threading the window.
"""
import math

import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins import (equivalence_current_source, mode_launch,
                             mode_source, solve_yee_mode)
from photonhub.plugins.modes import ModeSolver
from photonhub.runners.phsolver import find_solver

C0 = 2.99792458e8
LAM = 1.55
F0 = C0 / (LAM * 1e-6)
DL = 0.03
CW, CH = 0.5, 0.22


def _scene(sym=(0, 0, 0), Lx=4.0, x0=2.0, lz=6.0):
    core = ph.Structure(
        geometry=ph.Box(center_um=(x0, 1.4, lz / 2), size_um=(CW, CH, lz * 2)),
        medium=ph.Medium(permittivity=3.4738 ** 2))
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0)
    sim = ph.Simulation(
        size_um=(Lx, 2.8, lz), grid=ph.UniformGridSpec(dl_um=DL),
        run={"n_steps": 100}, background=ph.Background(permittivity=1.444 ** 2),
        structures=(core,), boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=10, symmetry=sym,
        sources=[ph.PointDipole(center_um=(Lx / 2, 1.4, 1.0),
                                polarization="Ex", source_time=pulse)])
    return sim, pulse


def _same_dipole(a, b):
    """Physically identical PointDipole: positions snap to the same node
    (float-tolerant), amplitude+phase exact."""
    return (a.polarization == b.polarization
            and np.allclose(a.center_um, b.center_um, atol=1e-12)
            and a.amplitude == pytest.approx(b.amplitude, abs=1e-12)
            and a.source_time.phase == pytest.approx(b.source_time.phase,
                                                     abs=1e-12))


def _yee_mode(sim, center, **win):
    return solve_yee_mode(sim, "z", 1.2, LAM, "TE", 0, dl_um=DL,
                          h_center_um=center[0], v_center_um=center[1], **win)


@pytest.mark.parametrize("sym,Lx,x0,ctr", [
    ((0, 0, 0), 4.0, 2.0, (2.0, 1.4)),      # full domain
    ((-1, 0, 0), 2.0, 0.0, (0.0, 1.4)),     # x-PEC half domain
    ((1, 0, 0), 2.0, 0.0, (0.0, 1.4)),      # x-PMC half domain
])
def test_auto_reproduces_the_manual_eq_current_launch(sym, Lx, x0, ctr):
    sim, pulse = _scene(sym=sym, Lx=Lx, x0=x0)
    win = dict(half_w_um=0.75, half_v_um=0.55)
    mode = _yee_mode(sim, ctr, **win)
    auto = mode_launch(sim, mode, axis="z", position_um=1.2,
                       source_time=pulse, center_um=ctr)
    manual = equivalence_current_source(
        sim, mode, axis="z", position_um=1.2, source_time=pulse,
        h_center_um=ctr[0], v_center_um=ctr[1], power_watts=1.0, **win)
    assert all(d.type == "point_dipole" for d in auto)
    assert len(auto) == len(manual) and len(auto) > 100
    assert all(_same_dipole(a, m) for a, m in zip(auto, manual))
    # under symmetry the sheet must stay in-domain (x0=0 plane kept, no x<0)
    if sym[0]:
        assert all(d.center_um[0] >= 0.0 for d in auto)


def test_origin_from_mode_is_grid_exact_and_matches_the_solve():
    # the launch recovers the mode's OWN origin (an exact grid multiple),
    # NOT a re-floored half-width: passed straight to the sheet, it lands on
    # the same nodes the solve used regardless of float-boundary snapping.
    from photonhub.plugins.mode_devices import _launch_window_origin
    from photonhub.plugins.yee_mode import window_min_face_bcs
    sim, _ = _scene()
    for hw, hv in ((0.75, 0.55), (0.75, 0.56), (0.72, 0.6)):
        mode = _yee_mode(sim, (2.0, 1.4), half_w_um=hw, half_v_um=hv)
        h_lo, v_lo = _launch_window_origin(mode, 2.0, 1.4)
        # grid-exact
        assert h_lo == pytest.approx(round(h_lo / DL) * DL, abs=1e-12)
        assert v_lo == pytest.approx(round(v_lo / DL) * DL, abs=1e-12)
        # == the origin the solve actually used (whatever the floor gave)
        sh, sv, _, _ = window_min_face_bcs(
            sim, "z", h_center=2.0, half_w=hw, v_center=1.4, half_v=hv, dl=DL)
        assert (h_lo, v_lo) == pytest.approx((sh, sv), abs=1e-12)


def test_auto_falls_back_to_aux_for_scalar_and_eq_current_for_broadband_yee():
    sim, pulse = _scene()
    # scalar FLM mode -> no true-H Yee mode -> §18
    scalar = ModeSolver.from_rectangular_core(
        wavelength_um=LAM, dl_um=DL, core_w_um=CW, core_h_um=CH,
        n_core=3.4738, n_clad=1.444).solve()[0]
    fb = mode_launch(sim, scalar, axis="z", position_um=1.2,
                     source_time=pulse, center_um=(2.0, 1.4))
    assert [s.type for s in fb] == ["mode_source"]
    # explicit eq_current on a scalar mode must RAISE (not silently fall back)
    with pytest.raises(ValueError, match="not a discrete full-vector Yee"):
        mode_launch(sim, scalar, axis="z", position_um=1.2, source_time=pulse,
                    center_um=(2.0, 1.4), launch="eq_current")
    # Stage B: a BROADBAND Yee mode is now eq-current-eligible (windowed-carrier
    # dipole sheets), NOT §18 — the broadband kill-switch is gone.
    mode = _yee_mode(sim, (2.0, 1.4), half_w_um=0.75, half_v_um=0.55)
    bb = {F0: mode, F0 * 1.01: mode}
    fb2 = mode_launch(sim, mode, axis="z", position_um=1.2, source_time=pulse,
                      center_um=(2.0, 1.4), modes_by_freq=bb)
    assert {s.type for s in fb2} == {"point_dipole"}
    # every dipole carries the 2-freq windowed carrier band (carrier_index 0/1)
    bands = {tuple(s.source_time.band_freqs_hz) for s in fb2}
    assert bands == {(F0, F0 * 1.01)}
    assert {s.source_time.carrier_index for s in fb2} == {0, 1}


def test_aux_matches_mode_source_including_y_axis_center_swap():
    # §18 fallback must reorder center_um from the (h,v)=in_plane_axes frame
    # into mode_source's _TRANSVERSE frame — which SWAPS for a y-cut.
    sim, pulse = _scene()
    mode = _yee_mode(sim, (2.0, 1.4), half_w_um=0.75, half_v_um=0.55)
    for axis, ctr in (("z", (2.0, 1.4)), ("y", (2.0, 3.0))):
        got = mode_launch(sim, mode, axis=axis, position_um=1.2,
                          source_time=pulse, center_um=ctr, launch="aux")[0]
        # in_plane_axes order -> _TRANSVERSE order
        from photonhub.viz._geometry import in_plane_axes
        from photonhub.plugins.mode_overlap import _TRANSVERSE
        hl, vl = in_plane_axes(axis)
        coord = {hl: ctr[0], vl: ctr[1]}
        t1, t2 = _TRANSVERSE[axis]
        ref = mode_source(sim, mode, axis=axis, position_um=1.2,
                          source_time=pulse, amplitude=1.0,
                          center_um=(coord[t1], coord[t2]))
        # ModeSource stores the RESAMPLED profile, not a center — an identical
        # profile means the (reordered) center placed the guide the same way.
        assert got.type == ref.type == "mode_source"
        assert got.profile == ref.profile
        assert got.position_um == ref.position_um


def test_bad_launch_arg_rejected():
    sim, pulse = _scene()
    mode = _yee_mode(sim, (2.0, 1.4), half_w_um=0.75, half_v_um=0.55)
    with pytest.raises(ValueError, match="launch must be"):
        mode_launch(sim, mode, axis="z", position_um=1.2, source_time=pulse,
                    center_um=(2.0, 1.4), launch="bogus")


@pytest.mark.skipif(find_solver() is None, reason="needs a phsolver binary")
def test_auto_launch_runs_and_conserves_like_the_manual_launch():
    from photonhub.plugins.mode_devices import ModeMonitor

    sim, pulse = _scene(lz=6.0)
    sim = sim.model_copy(update={"run": ph.RunSpec(n_steps=6000)})
    win = dict(half_w_um=0.75, half_v_um=0.55)
    mode = _yee_mode(sim, (2.0, 1.4), **win)
    dips = mode_launch(sim, mode, axis="z", position_um=1.2,
                       source_time=pulse, center_um=(2.0, 1.4))
    nx, ny = round(4.0 / DL), round(2.8 / DL)
    cux = 0.5 * (1.25 * DL + (nx - 1 + 0.25) * DL)
    sux = (nx - 1 + 0.25) * DL - 1.25 * DL
    cvy = 0.5 * (1.25 * DL + (ny - 1 + 0.25) * DL)
    svy = (ny - 1 + 0.25) * DL - 1.25 * DL
    fms = {nm: ph.FieldDftMonitor(
        name=nm, center_um=(cux, cvy, (round(z / DL - 0.25) + 0.25) * DL),
        size_um=(sux, svy, 0.0), fields=("Ex", "Ey", "Hx", "Hy"),
        freqs_hz=(F0,)) for nm, z in (("in", 1.8), ("out", 4.5))}
    sim_run = sim.model_copy(update=dict(sources=tuple(dips),
                                         monitors=tuple(fms.values())))
    data = ph.run_local(sim_run, solver_path=find_solver(), quiet=True,
                        timeout=1800)
    p = {nm: ModeMonitor(field_monitor=fms[nm], mode=mode, axis="z",
                         center_um=(2.0, 1.4), direction="+", dl_um=DL
                         ).mode_power(data)[F0] for nm in ("in", "out")}
    # straight guide: near-unit transmission, clean launch
    assert p["out"] / p["in"] == pytest.approx(1.0, abs=0.02)
