"""§20 symmetry planes in the Yee cross-section mode solve + eq-current launch.

The discrete mirror is EXACT on the lattice: the matching-parity half of a
full-window Yee eigenmode satisfies the half-window problem with the §20 BC
(PMC: bwd-row 2/dl; PEC: bwd-row 0 + on-plane tangential-E pinned) exactly, so
half==full holds to eigensolver precision — asserted sharply here, both
parities, both in-plane axes. The engine-gated test then closes the loop:
a half-domain eq-current launch + modal readout reproduces the full-domain
transmission (the engine's §20 boundary supplies the mirror images at 1x
amplitude — no half-weighting anywhere, matching engine/tests/test_symmetry).
"""
import math

import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins import equivalence_current_source, solve_yee_mode
from photonhub.plugins.yee_mode import solve_yee_mode_bank, window_min_face_bcs
from photonhub.runners.phsolver import find_solver

C0 = 2.99792458e8
LAM = 1.55
F0 = C0 / (LAM * 1e-6)
DL = 0.05
CW, CH = 0.5, 0.22
EPS_CORE = 3.4738 ** 2
EPS_CLAD = 1.444 ** 2


def _sim(L, core_center, symmetry=(0, 0, 0), dl=DL, lz=1.0, n_steps=100):
    core = ph.Structure(
        geometry=ph.Box(center_um=(core_center[0], core_center[1], L[2] / 2),
                        size_um=(CW, CH, L[2] * 4)),
        medium=ph.Medium(permittivity=EPS_CORE))
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0)
    return ph.Simulation(
        size_um=L, grid=ph.UniformGridSpec(dl_um=dl), run={"n_steps": n_steps},
        background=ph.Background(permittivity=EPS_CLAD), structures=(core,),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"), pml_num_layers=10,
        symmetry=symmetry,
        sources=[ph.PointDipole(center_um=(L[0] / 2, L[1] / 2, L[2] / 2),
                                polarization="Ex", source_time=pulse)])


# Full geometry: core centered at (2.0, 1.4) in a (4.0, 2.8) cross-section.
XC, YC = 2.0, 1.4
W, V = 1.6, 1.2      # window half-extents (node multiples of DL)


def _full_sim(**kw):
    return _sim((4.0, 2.8, 1.0), (XC, YC), **kw)


def _align(a, b):
    """Cancel the arbitrary eigenvector norm/phase between two field slices:
    scale ``b`` so both equal 1 at ``a``'s magnitude peak."""
    i = int(np.argmax(np.abs(a)))
    return a / a.flat[i], b / b.flat[i]


def _compare_half_to_full(half, full, ih0, dominant):
    """half mode arrays == the upper-half slice of the full mode's, after
    norm/phase alignment on the dominant component."""
    nhh = half.ex.shape[1]
    ref_h = getattr(half, dominant)
    ref_f = getattr(full, dominant)[:, ih0:ih0 + nhh]
    ra, rb = _align(ref_f, ref_h)
    scale = float(np.max(np.abs(ra)))
    for comp in ("ex", "ey", "ez", "hx", "hy"):
        a = getattr(full, comp)[:, ih0:ih0 + nhh]
        b = getattr(half, comp)
        # apply the SAME alignment factors as the dominant component
        a = a / getattr(full, dominant)[:, ih0:ih0 + nhh].flat[
            int(np.argmax(np.abs(getattr(full, dominant)[:, ih0:ih0 + nhh])))]
        b = b / getattr(half, dominant).flat[
            int(np.argmax(np.abs(getattr(half, dominant))))]
        np.testing.assert_allclose(b, a, rtol=2e-5, atol=5e-6 * scale,
                                   err_msg=comp)


# Per-polarization windows: half==full is exact up to the FULL solve's
# residual tails at ITS low edge (the half solve's plane BC is exact; the
# full window's low edge sits one snap-cell farther with the legacy phantom
# wall) — TM0's fatter tails (kappa ~4.5/um vs TE0's ~7.8/um) need a wider
# window to push that below the 1e-8 gate. The v-edges are common to both
# solves in the x-plane tests (and vice versa) and cancel exactly.
_WIN = {"TE": (1.6, 1.2), "TM": (2.4, 2.2)}


@pytest.mark.parametrize("parity,pol", [(-1, "TE"), (1, "TM")])
def test_x_plane_half_equals_full_exactly(parity, pol):
    # x-min symmetry plane through the core centre: TE0 is the PEC(-1) family
    # (normal Ex even, tangential Ey/Ez odd), TM0 the PMC(+1) family.
    w, v = _WIN[pol]
    full = solve_yee_mode(_full_sim(), "z", 0.5, LAM, pol, 0, dl_um=DL,
                          h_center_um=XC, v_center_um=YC,
                          half_w_um=w, half_v_um=v)
    hsim = _sim((2.0, 2.8, 1.0), (0.0, YC), symmetry=(parity, 0, 0))
    half = solve_yee_mode(hsim, "z", 0.5, LAM, pol, 0, dl_um=DL,
                          h_center_um=0.0, v_center_um=YC,
                          half_w_um=w, half_v_um=v)
    assert half.n_eff == pytest.approx(full.n_eff, abs=1e-8)
    # window bookkeeping: the half window was clipped to start ON the plane.
    # The full window origin uses the code's own floor-snap (float floor can
    # land one cell below the nominal XC-w — recompute, don't hand-derive).
    h_lo_full = float(np.floor((XC - w) / DL) * DL)
    ih0 = round((XC - h_lo_full) / DL)         # full-grid index of the plane
    assert half.ex.shape[1] == round(w / DL)
    _compare_half_to_full(half, full, ih0, "ex" if pol == "TE" else "ey")
    if parity == -1:
        # PEC pins the on-plane tangential E (and the derived on-plane Ez);
        # the eigensolver returns machine-zero, not literal 0, in masked DOFs.
        assert np.max(np.abs(half.ey[:, 0])) < 1e-12
        assert np.max(np.abs(half.ez[:, 0])) < 1e-12


@pytest.mark.parametrize("parity,pol", [(1, "TE"), (-1, "TM")])
def test_y_plane_half_equals_full_exactly(parity, pol):
    # y-min (horizontal mid-) plane: TE0 is the PMC(+1) family there
    # (tangential Ex even), TM0 the PEC(-1) family (normal Ey even).
    w, v = _WIN[pol]
    full = solve_yee_mode(_full_sim(), "z", 0.5, LAM, pol, 0, dl_um=DL,
                          h_center_um=XC, v_center_um=YC,
                          half_w_um=w, half_v_um=v)
    hsim = _sim((4.0, 1.4, 1.0), (XC, 0.0), symmetry=(0, parity, 0))
    half = solve_yee_mode(hsim, "z", 0.5, LAM, pol, 0, dl_um=DL,
                          h_center_um=XC, v_center_um=0.0,
                          half_w_um=w, half_v_um=v)
    assert half.n_eff == pytest.approx(full.n_eff, abs=1e-8)
    assert half.ex.shape[0] == round(v / DL)   # v-window clipped at the plane
    if parity == -1:
        assert np.max(np.abs(half.ex[0, :])) < 1e-12


def test_window_rule_clips_only_symmetry_axes():
    hsim = _sim((2.0, 2.8, 1.0), (0.0, YC), symmetry=(-1, 0, 0))
    h_lo, v_lo, h_bc, v_bc = window_min_face_bcs(
        hsim, "z", h_center=0.0, half_w=W, v_center=YC, half_v=V, dl=DL)
    assert (h_lo, h_bc) == (0.0, "pec")
    # the v origin keeps the plain grid floor-snap (float floor may land one
    # cell below the nominal YC-V) — compare against the formula, not the value
    assert v_lo == float(np.floor((YC - V) / DL) * DL) and v_bc is None
    # interior window on the symmetry axis: plane out of reach, BC off
    h_lo2, _, h_bc2, _ = window_min_face_bcs(
        hsim, "z", h_center=0.8, half_w=0.4, v_center=YC, half_v=V, dl=DL)
    assert h_lo2 > 0.0 and h_bc2 is None
    # no symmetry: nothing clips even for a plane-crossing window
    fsim = _full_sim()
    h_lo3, _, h_bc3, _ = window_min_face_bcs(
        fsim, "z", h_center=0.0, half_w=W, v_center=YC, half_v=V, dl=DL)
    assert h_lo3 == pytest.approx(-W) and h_bc3 is None


def test_bank_carries_the_symmetry_bcs():
    hsim = _sim((2.0, 2.8, 1.0), (0.0, YC), symmetry=(1, 0, 0))
    win = dict(h_center_um=0.0, v_center_um=YC, half_w_um=W, half_v_um=V,
               dl_um=DL)
    mode = solve_yee_mode(hsim, "z", 0.5, LAM, "TM", 0, **win)
    bank = solve_yee_mode_bank(hsim, "z", 0.5, [F0], "TM", 0, **win)
    assert bank[F0].n_eff == pytest.approx(mode.n_eff, abs=1e-12)


def test_eq_current_keeps_on_plane_dipoles_under_pmc():
    # TM0 under x-PMC peaks ON the plane in its even components — the on-plane
    # dipole row is the self-mirror row the engine expects at 1x amplitude
    # (regression: the 0.25*dl interior margin used to silently drop it).
    hsim = _sim((2.0, 2.8, 1.0), (0.0, YC), symmetry=(1, 0, 0), lz=1.0)
    pulse = hsim.sources[0].source_time
    win = dict(h_center_um=0.0, v_center_um=YC, half_w_um=W, half_v_um=V)
    mode = solve_yee_mode(hsim, "z", 0.5, LAM, "TM", 0, dl_um=DL, **win)
    dips = equivalence_current_source(hsim, mode, axis="z", position_um=0.5,
                                      source_time=pulse, direction="+", **win)
    on_plane = [d for d in dips if d.center_um[0] == 0.0]
    assert on_plane, "on-plane self-mirror dipoles must be kept under PMC"
    assert all(d.center_um[0] >= 0.0 for d in dips)
    # the on-plane row carries only the plane-registered species
    kinds = {d.polarization for d in on_plane}
    assert kinds <= {"Hx", "Ey"}


def test_eq_current_on_plane_row_vanishes_under_pec():
    # TE0 under x-PEC: every on-plane species carries a parity-pinned (exact
    # 0) mode value, so the threshold removes the whole row — nothing for the
    # engine to silently drop or freeze.
    hsim = _sim((2.0, 2.8, 1.0), (0.0, YC), symmetry=(-1, 0, 0), lz=1.0)
    pulse = hsim.sources[0].source_time
    win = dict(h_center_um=0.0, v_center_um=YC, half_w_um=W, half_v_um=V)
    mode = solve_yee_mode(hsim, "z", 0.5, LAM, "TE", 0, dl_um=DL, **win)
    dips = equivalence_current_source(hsim, mode, axis="z", position_um=0.5,
                                      source_time=pulse, direction="+", **win)
    assert dips
    assert not [d for d in dips if d.center_um[0] == 0.0]
    # and the no-symmetry margin behavior is unchanged
    fsim = _full_sim()
    fwin = dict(h_center_um=XC, v_center_um=YC, half_w_um=W, half_v_um=V)
    fmode = solve_yee_mode(fsim, "z", 0.5, LAM, "TE", 0, dl_um=DL, **fwin)
    fdips = equivalence_current_source(fsim, fmode, axis="z", position_um=0.5,
                                       source_time=fsim.sources[0].source_time,
                                       direction="+", **fwin)
    assert min(d.center_um[0] for d in fdips) > 0.25 * DL


# --------------------------------------------------------------------------- #
# Engine-gated: half-domain launch + readout reproduces the full-domain T.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(find_solver() is None, reason="needs a phsolver binary")
def test_half_domain_transmission_equals_full():
    from photonhub.plugins.mode_devices import ModeMonitor

    dl, lz = 0.03, 6.0
    xc, yc = 0.9, 0.7
    z_src, z_in, z_out = 1.2, 1.8, 4.8

    def scene(Lx, x0, symmetry):
        L = (Lx, 1.4, lz)
        core = ph.Structure(
            geometry=ph.Box(center_um=(x0, yc, lz / 2), size_um=(CW, CH, lz * 2)),
            medium=ph.Medium(permittivity=EPS_CORE))
        # x-symmetric perturbation: a wider Si stub (mini-MMI) => T < 1
        stub = ph.Structure(
            geometry=ph.Box(center_um=(x0, yc, 3.2), size_um=(1.1, CH, 0.5)),
            medium=ph.Medium(permittivity=EPS_CORE))
        pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0)
        sim0 = ph.Simulation(
            size_um=L, grid=ph.UniformGridSpec(dl_um=dl), run={"n_steps": 5000},
            background=ph.Background(permittivity=EPS_CLAD),
            structures=(core, stub),
            boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
            pml_num_layers=10, symmetry=symmetry,
            sources=[ph.PointDipole(center_um=(Lx / 2, yc, 1.0),
                                    polarization="Ex", source_time=pulse)])
        # identical PHYSICAL windows: the half window [0, 0.75] is exactly the
        # upper half of the full one [xc-0.75, xc+0.75]
        win = dict(h_center_um=x0, v_center_um=yc, half_w_um=0.75,
                   half_v_um=0.55)
        mode = solve_yee_mode(sim0, "z", z_src, LAM, "TE", 0, dl_um=dl, **win)
        dips = equivalence_current_source(sim0, mode, axis="z",
                                          position_um=z_src, source_time=pulse,
                                          direction="+", **win)
        # §12-snapped transverse faces spanning the interior. At the symmetry
        # plane the face drops to k=0 (0.25*dl): the one-cell interior margin
        # is PML clearance, and the plane carries no PML — clipping the
        # plane-adjacent columns out of the overlap would bias the readout
        # (the on-plane nodes themselves are parity-zero and stay excluded).
        nx, ny = round(L[0] / dl), round(L[1] / dl)
        lo_x = 0.25 * dl if x0 == 0.0 else 1.25 * dl
        hi_x = (nx - 1 + 0.25) * dl
        lo_y, hi_y = 1.25 * dl, (ny - 1 + 0.25) * dl
        cux, sux = 0.5 * (lo_x + hi_x), hi_x - lo_x
        cvy, svy = 0.5 * (lo_y + hi_y), hi_y - lo_y
        mons = {}
        for nm, zp in (("in", z_in), ("out", z_out)):
            zq = (round(zp / dl - 0.25) + 0.25) * dl
            mons[nm] = ph.FieldDftMonitor(
                name=nm, center_um=(cux, cvy, zq), size_um=(sux, svy, 0.0),
                fields=("Ex", "Ey", "Hx", "Hy"), freqs_hz=(F0,))
        sim = sim0.model_copy(update=dict(
            sources=tuple(dips), monitors=tuple(mons.values())))
        return sim, mode, mons

    def transmission(sim, mode, mons, center):
        data = ph.run_local(sim, solver_path=find_solver(), quiet=True,
                            timeout=1800)
        p = {}
        for nm in ("in", "out"):
            mm = ModeMonitor(field_monitor=mons[nm], mode=mode, axis="z",
                             center_um=center, direction="+", dl_um=sim.grid.dl_um)
            p[nm] = mm.mode_power(data)[F0]
        return p["out"] / p["in"]

    t_full = transmission(*scene(1.8, xc, (0, 0, 0)), center=(xc, yc))
    t_half = transmission(*scene(0.9, 0.0, (-1, 0, 0)), center=(0.0, yc))
    assert 0.05 < t_full < 0.999           # the stub actually perturbs
    assert t_half == pytest.approx(t_full, rel=2e-3)


def test_flm_path_warns_on_symmetric_sims():
    # use_yee=False cannot honor the plane (it paints the mirror geometry) —
    # it must say so instead of silently solving the wrong cross-section.
    from photonhub.plugins import mode_bank_on_cross_section, solve_mode_on_cross_section

    hsim = _sim((2.0, 2.8, 1.0), (0.0, YC), symmetry=(-1, 0, 0))
    win = dict(h_center_um=0.0, v_center_um=YC, half_w_um=0.8, half_v_um=0.8,
               dl_um=DL)
    with pytest.warns(UserWarning, match="FLM solver ignores sim.symmetry"):
        solve_mode_on_cross_section(hsim, "z", 0.5, LAM, "TE", 0,
                                    use_yee=False, **win)
    with pytest.warns(UserWarning, match="FLM solver ignores sim.symmetry"):
        mode_bank_on_cross_section(hsim, "z", 0.5, [F0], "TE", 0,
                                   use_yee=False, **win)
