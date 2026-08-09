"""Equivalence-current (phased-dipole Huygens) mode source: construction-level
invariants + an engine-gated launch-quality check."""
import math

import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins import equivalence_current_source, solve_yee_mode
from photonhub.runners.phsolver import find_solver

C0 = 2.99792458e8
LAM = 1.55
F0 = C0 / (LAM * 1e-6)
DL = 0.03
CW, CH = 0.45, 0.22


def _strip(dl=DL, lz=6.0):
    L = (round(1.8 / dl) * dl, round(1.4 / dl) * dl, round(lz / dl) * dl)
    core = ph.Structure(
        geometry=ph.Box(center_um=(L[0] / 2, L[1] / 2, L[2] / 2), size_um=(CW, CH, L[2] * 2)),
        medium=ph.Medium(permittivity=3.4738 ** 2))
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0)
    sim = ph.Simulation(
        size_um=L, grid=ph.UniformGridSpec(dl_um=dl), run={"n_steps": 4500},
        background=ph.Background(permittivity=1.444 ** 2), structures=(core,),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"), pml_num_layers=10,
        sources=[ph.PointDipole(center_um=(L[0] / 2, L[1] / 2, 1.2), polarization="Ex",
                                source_time=pulse)])
    return sim, L, pulse


def _mode_and_window(sim, L, dl=DL):
    win = dict(h_center_um=L[0] / 2, v_center_um=L[1] / 2,
               half_w_um=min(CW / 2 + 0.5, L[0] / 2 - 2 * dl),
               half_v_um=min(CH / 2 + 0.5, L[1] / 2 - 2 * dl))
    mode = solve_yee_mode(sim, "z", 1.2, LAM, "TE", 0, dl_um=dl, **win)
    return mode, win


def test_sheets_land_on_yee_points_of_the_two_planes():
    sim, L, pulse = _strip()
    mode, win = _mode_and_window(sim, L)
    dips = equivalence_current_source(sim, mode, axis="z", position_um=1.2,
                                      source_time=pulse, direction="+", **win)
    assert len(dips) > 100
    k0 = round(1.2 / DL)
    for d in dips:
        x, y, z = d.center_um
        if d.polarization.startswith("H"):          # M sheet: H nodes of cell k0-1
            assert z == pytest.approx((k0 - 0.5) * DL, abs=1e-9)
        else:                                       # J sheet: E plane k0
            assert z == pytest.approx(k0 * DL, abs=1e-9)
        # transverse Yee offsets: each coordinate is an integer or half-integer cell
        for c in (x, y):
            frac = (c / DL) % 1.0
            assert min(frac, abs(frac - 0.5), abs(frac - 1.0)) < 1e-6
        # snapping identity: the engine's round(pos/dl - offset) must be exact
        assert 0 < x < L[0] and 0 < y < L[1]


def test_polarization_pairing_and_both_current_types_present():
    sim, L, pulse = _strip()
    mode, win = _mode_and_window(sim, L)
    dips = equivalence_current_source(sim, mode, axis="z", position_um=1.2,
                                      source_time=pulse, direction="+", **win)
    pols = {d.polarization for d in dips}
    # TE strip mode: dominant Ex -> M_y (Hy) + J_x (Ex) sheets, plus the minor pair
    assert "Hy" in pols and "Ex" in pols
    n_e = sum(d.polarization.startswith("E") for d in dips)
    n_h = sum(d.polarization.startswith("H") for d in dips)
    assert n_e > 0 and n_h > 0


def test_j_sheet_carries_half_cell_plus_half_step_phase():
    sim, L, pulse = _strip()
    mode, win = _mode_and_window(sim, L)
    dips = equivalence_current_source(sim, mode, axis="z", position_um=1.2,
                                      source_time=pulse, direction="+", **win)
    dt = 0.99 * (DL * 1e-6) / (C0 * math.sqrt(3.0))
    beta = mode.n_eff * 2 * math.pi / LAM
    ph_j = beta * DL / 2 + 2 * math.pi * F0 * dt / 2
    # strongest J_x (Ex) dipole vs strongest M_y (Hy) dipole: for the co-real
    # guided mode (E real, H real) the phase difference is pi+ph_j - pi = ph_j
    jx = max((d for d in dips if d.polarization == "Ex"), key=lambda d: d.amplitude)
    my = max((d for d in dips if d.polarization == "Hy"), key=lambda d: d.amplitude)
    dphi = (jx.source_time.phase - my.source_time.phase) % (2 * math.pi)
    assert min(dphi - ph_j % (2 * math.pi),
               abs(dphi - ph_j % (2 * math.pi) - 2 * math.pi)) == pytest.approx(0.0, abs=1e-6) \
        or dphi == pytest.approx(ph_j % (2 * math.pi), abs=1e-6)


def test_half_step_phase_honors_the_simulations_courant():
    # RunSpec.courant is user-settable; the J-sheet's half-step temporal phase
    # e^{i w dt/2} must use the run's ACTUAL dt, not a hardcoded 0.99 (which
    # silently de-tunes the launch for any other courant).
    sim, L, pulse = _strip()
    slow = ph.Simulation(**{**{k: getattr(sim, k) for k in (
        "size_um", "grid", "background", "structures", "boundaries",
        "pml_num_layers", "sources")}, "run": {"n_steps": 4500, "courant": 0.5}})
    mode, win = _mode_and_window(slow, L)
    dips = equivalence_current_source(slow, mode, axis="z", position_um=1.2,
                                      source_time=pulse, direction="+", **win)
    dt = 0.5 * (DL * 1e-6) / (C0 * math.sqrt(3.0))
    beta = mode.n_eff * 2 * math.pi / LAM
    ph_j = (beta * DL / 2 + 2 * math.pi * F0 * dt / 2) % (2 * math.pi)
    jx = max((d for d in dips if d.polarization == "Ex"), key=lambda d: d.amplitude)
    my = max((d for d in dips if d.polarization == "Hy"), key=lambda d: d.amplitude)
    dphi = (jx.source_time.phase - my.source_time.phase) % (2 * math.pi)
    assert dphi == pytest.approx(ph_j, abs=1e-6)
    # and it must DIFFER from the hardcoded-0.99 phase by the dt gap
    dt99 = 0.99 * (DL * 1e-6) / (C0 * math.sqrt(3.0))
    ph_j99 = (beta * DL / 2 + 2 * math.pi * F0 * dt99 / 2) % (2 * math.pi)
    assert abs(dphi - ph_j99) > 1e-4


def test_power_scaling_scales_amplitudes():
    sim, L, pulse = _strip()
    mode, win = _mode_and_window(sim, L)
    d1 = equivalence_current_source(sim, mode, axis="z", position_um=1.2,
                                    source_time=pulse, power_watts=1.0, **win)
    d4 = equivalence_current_source(sim, mode, axis="z", position_um=1.2,
                                    source_time=pulse, power_watts=4.0, **win)
    a1 = max(d.amplitude for d in d1)
    a4 = max(d.amplitude for d in d4)
    assert a4 / a1 == pytest.approx(2.0, rel=1e-9)   # power ∝ amplitude²


@pytest.mark.skipif(find_solver() is None, reason="needs a phsolver binary")
@pytest.mark.parametrize("direction", ["+", "-"])
def test_launch_is_unidirectional_and_low_loss(direction):
    """End-to-end on the engine: >99% of the flux goes the requested way and the
    0.5->2.5 um flux loss beats the sec-18 floor (<0.4% at this coarse dl)."""
    from photonhub.plugins.mode_devices import ModeMonitor

    sim0, L, pulse = _strip(lz=6.0)
    mode, win = _mode_and_window(sim0, L)
    z0 = 3.0 if direction == "+" else 3.0
    dips = equivalence_current_source(sim0, mode, axis="z", position_um=z0,
                                      source_time=pulse, direction=direction, **win)
    sgn = 1 if direction == "+" else -1
    # §12: transverse faces at (k+1/4)*dl so every Yee component snaps to one cell
    nx, ny = round(L[0] / DL), round(L[1] / DL)
    lo_x, hi_x = 1.25 * DL, (nx - 1 + 0.25) * DL
    lo_y, hi_y = 1.25 * DL, (ny - 1 + 0.25) * DL
    cux, sux = 0.5 * (lo_x + hi_x), hi_x - lo_x
    cvy, svy = 0.5 * (lo_y + hi_y), hi_y - lo_y
    mons, mm = [], {}
    for nm, d in (("n", 0.5), ("f", 2.5), ("b", -0.9)):
        zp = (round((z0 + sgn * d) / DL - 0.25) + 0.25) * DL
        fm = ph.FieldDftMonitor(name=nm, center_um=(cux, cvy, zp),
                                size_um=(sux, svy, 0.0),
                                fields=("Ex", "Ey", "Hx", "Hy"), freqs_hz=(F0,))
        mons.append(fm)
        mm[nm] = fm
    sim = sim0.model_copy(update=dict(sources=tuple(dips), monitors=tuple(mons)))
    data = ph.run_local(sim, solver_path=find_solver(), quiet=True, timeout=1800)

    from photonhub.plugins.mode_overlap import _TRANSVERSE, _cell_widths

    def flux(name):
        da = data[name]
        def comp(n):
            arr = da.sel(component=n)
            if "f" in arr.dims:
                arr = arr.isel(f=0)
            return arr.squeeze(drop=True).transpose("y", "x")
        E1, E2, H1, H2 = comp("Ex"), comp("Ey"), comp("Hx"), comp("Hy")
        cx = np.asarray(E1.coords["x"].values, float)
        cy = np.asarray(E1.coords["y"].values, float)
        dA = np.outer(_cell_widths(cy), _cell_widths(cx))
        s = 0.5 * np.real(E1.values * np.conj(H2.values) - E2.values * np.conj(H1.values))
        return float(np.sum(s * dA))

    f_n, f_f, f_b = sgn * flux("n"), sgn * flux("f"), -sgn * flux("b")
    assert f_n > 0 and f_f > 0
    assert f_b / f_n < 0.01                      # <1% backward
    assert abs(1 - f_f / f_n) < 0.004            # flux loss beats the sec-18 floor


@pytest.mark.skipif(find_solver() is None, reason="needs a phsolver binary")
def test_launch_is_forward_on_a_y_normal_plane():
    """Regression for the left-handed-cut launch bug: in_plane_axes('y') = (x, z)
    is LEFT-handed (x̂×ẑ = −ŷ), so the M=−n̂×E sheet was sign-flipped relative to J
    and a +y eq-current launch put ~98% of the flux toward −y. The z/x cuts are
    right-handed and were always correct (covered above); y was untested. Assert a
    +y launch is downstream-dominant, mirroring the z case."""
    from photonhub.viz._geometry import in_plane_axes

    dl = 0.04
    long_um = 6.0
    tw = round(1.8 / dl) * dl
    L = (tw, round(long_um / dl) * dl, tw)       # long axis = y
    cen = tuple(v / 2 for v in L)
    core = ph.Structure(
        geometry=ph.Box(center_um=cen, size_um=(CW, L[1] * 2, CH)),  # CW in x, CH in z
        medium=ph.Medium(permittivity=3.4738 ** 2))
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0)
    sim = ph.Simulation(
        size_um=L, grid=ph.UniformGridSpec(dl_um=dl), run={"n_steps": 4000},
        background=ph.Background(permittivity=1.444 ** 2), structures=(core,),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"), pml_num_layers=10,
        sources=[ph.PointDipole(center_um=cen, polarization="Ex", source_time=pulse)])
    h, v = in_plane_axes("y")                    # ('x', 'z')
    hi, vi = {"x": 0, "y": 1, "z": 2}[h], {"x": 0, "y": 1, "z": 2}[v]
    win = dict(h_center_um=cen[hi], v_center_um=cen[vi],
               half_w_um=min(CW / 2 + 0.5, L[hi] / 2 - 2 * dl),
               half_v_um=min(CH / 2 + 0.5, L[vi] / 2 - 2 * dl))
    pos = 3.0
    mode = solve_yee_mode(sim, "y", pos, LAM, "TE", 0, dl_um=dl, **win)
    dips = equivalence_current_source(sim, mode, axis="y", position_um=pos,
                                      source_time=pulse, direction="+", **win)

    def yflux_monitor(name, off):
        yc = (round((pos + off) / dl - 0.25) + 0.25) * dl
        return ph.FieldDftMonitor(
            name=name, center_um=(cen[0], yc, cen[2]),
            size_um=(L[0] - 3 * dl, 0.0, L[2] - 3 * dl),
            fields=("Ex", "Ez", "Hx", "Hz"), freqs_hz=(F0,))

    sim2 = sim.model_copy(update=dict(
        sources=tuple(dips),
        monitors=(yflux_monitor("up", -0.9), yflux_monitor("dn", 2.0))))
    data = ph.run_local(sim2, solver_path=find_solver(), quiet=True, timeout=1800)

    from photonhub.plugins.mode_overlap import _cell_widths

    def yflux(name):                             # S_y = 0.5 Re(Ez Hx* − Ex Hz*)
        da = data[name]
        def comp(letter):
            arr = da.sel(component=letter)
            if "f" in arr.dims:
                arr = arr.isel(f=0)
            return arr.squeeze(drop=True).transpose("z", "x")
        Ex, Ez, Hx, Hz = comp("Ex"), comp("Ez"), comp("Hx"), comp("Hz")
        cx = np.asarray(Ex.coords["x"].values, float)
        cz = np.asarray(Ex.coords["z"].values, float)
        dA = np.outer(_cell_widths(cz), _cell_widths(cx))
        s = 0.5 * np.real(Ez.values * np.conj(Hx.values) - Ex.values * np.conj(Hz.values))
        return float(np.sum(s * dA))

    f_up, f_dn = yflux("up"), yflux("dn")
    assert f_dn > 0                              # forward (+y) power is positive
    assert abs(f_up) / abs(f_dn) < 0.02          # <2% backward (was ~50x, launched −y)
