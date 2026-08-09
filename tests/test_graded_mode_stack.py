"""Graded (§15 nonuniform / auto-mesh) support in the default mode stack:
Yee cross-section solve, per-frequency banks, equivalence-current launch, and
monitor readout — Path B of the launch/readout default program.

The approach is GRADED-NATIVE (the same construction Tidy3D's open mode solver
uses): forward derivatives divided by the primal cell widths, backward by the
dual (midpoint) widths, eps sampled over each Yee point's own primal/dual cell,
dipoles stamped on the sim's true node ladder with local-width amplitudes.
No resampling anywhere — the launch/readout registration stays 1:1.

Invariants pinned here:
  * uniform grids reduce BIT-IDENTICALLY (scalar fast paths);
  * a DEGENERATE graded grid (uniform coords listed explicitly) matches the
    uniform solve to machine precision — the vector operators are exactly the
    scalar ones at constant spacing;
  * graded->uniform convergence with resolution;
  * the engine-gated end-to-end: automesh launch+readout reproduces the
    fine-uniform transmission.
"""
import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins import (equivalence_current_source, mode_launch,
                             mode_monitor, solve_mode_on_cross_section,
                             solve_yee_mode)
from photonhub.components.grid import auto_grid, snap_mixed_plane
from photonhub.runners.phsolver import find_solver

C0 = 2.99792458e8
LAM = 1.55
F0 = C0 / (LAM * 1e-6)
CW, CH = 0.5, 0.22
NCORE, NCLAD = 3.4738, 1.444
L = (4.0, 3.0, 2.0)
XC, YC = 2.0, 1.5


def _sim(grid, L=L, core_center=(XC, YC)):
    core = ph.Structure(
        geometry=ph.Box(center_um=(core_center[0], core_center[1], L[2] / 2),
                        size_um=(CW, CH, L[2] * 4)),
        medium=ph.Medium(permittivity=NCORE ** 2))
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0)
    return ph.Simulation(
        size_um=L, grid=grid, run={"n_steps": 100},
        background=ph.Background(permittivity=NCLAD ** 2), structures=(core,),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=10,
        sources=[ph.PointDipole(center_um=(L[0] / 2, L[1] / 2, L[2] / 2),
                                polarization="Ex", source_time=pulse)])


WIN = dict(h_center_um=XC, v_center_um=YC, half_w_um=1.2, half_v_um=1.0)


def test_degenerate_graded_matches_uniform_to_machine_precision():
    # A graded grid whose coords ARE a uniform ladder must reproduce the
    # uniform solve exactly (up to operator-assembly ULPs): the vector-spacing
    # path is the scalar operator at constant dq.
    dl = LAM / (NCORE * 25)
    nx, ny = round(L[0] / dl), round(L[1] / dl)
    deg = ph.GradedGridSpec(dl_um=dl, coords={
        "x": tuple(float(i * dl) for i in range(nx)),
        "y": tuple(float(i * dl) for i in range(ny))})
    mu = solve_yee_mode(_sim(ph.UniformGridSpec(dl_um=dl)), "z", 0.5, LAM,
                        "TE", 0, dl_um=dl, **WIN)
    md = solve_yee_mode(_sim(deg), "z", 0.5, LAM, "TE", 0, dl_um=dl, **WIN)
    assert md.n_eff == pytest.approx(mu.n_eff, abs=1e-11)
    assert md.x_coords_um is not None          # graded path engaged
    assert mu.x_coords_um is None              # uniform path carries no coords


def test_graded_solve_converges_to_uniform_with_resolution():
    diffs = []
    for spw in (25, 60):
        dlf = LAM / (NCORE * spw)
        u = solve_yee_mode(_sim(ph.UniformGridSpec(dl_um=dlf)), "z", 0.5, LAM,
                           "TE", 0, dl_um=dlf, **WIN)
        g = auto_grid(size_um=L, wavelength_um=LAM,
                      structures=_sim(ph.UniformGridSpec(dl_um=dlf)).structures,
                      background_index=NCLAD, steps_per_wvl=float(spw))
        gm = solve_yee_mode(_sim(g), "z", 0.5, LAM, "TE", 0, dl_um=g.dl_um,
                            **WIN)
        diffs.append(abs(u.n_eff - gm.n_eff))
    assert diffs[0] < 2e-2                     # sane at 25/λ
    assert diffs[1] < 0.4 * diffs[0]           # and shrinking with resolution


def test_graded_mode_carries_true_node_ladder():
    g = auto_grid(size_um=L, wavelength_um=LAM,
                  structures=_sim(ph.UniformGridSpec(dl_um=0.05)).structures,
                  background_index=NCLAD, steps_per_wvl=25.0)
    m = solve_yee_mode(_sim(g), "z", 0.5, LAM, "TE", 0, dl_um=g.dl_um, **WIN)
    assert m.x_coords_um is not None and m.y_coords_um is not None
    assert len(m.x_coords_um) == m.ex.shape[1]
    assert len(m.y_coords_um) == m.ex.shape[0]
    dq = np.diff(m.x_coords_um)
    assert dq.max() > 1.5 * dq.min()           # genuinely nonuniform window
    # ladders are the sim's own nodes (center-relative): every absolute node
    # coordinate must be a member of the grid's x coords
    qx = np.asarray(g.coords.x, dtype=float)
    absn = np.asarray(m.x_coords_um, dtype=float) + XC
    assert all(np.min(np.abs(qx - a)) < 1e-9 for a in absn)


def test_eq_current_stamps_on_graded_nodes_with_local_widths():
    g = auto_grid(size_um=L, wavelength_um=LAM,
                  structures=_sim(ph.UniformGridSpec(dl_um=0.05)).structures,
                  background_index=NCLAD, steps_per_wvl=25.0)
    sim = _sim(g)
    pulse = sim.sources[0].source_time
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0,
                                       dl_um=g.dl_um, **WIN)
    dips = mode_launch(sim, mode, axis="z", position_um=0.5,
                       source_time=pulse, center_um=(XC, YC))
    assert len(dips) > 100
    assert {d.type for d in dips} == {"point_dipole"}
    # transverse coordinates: every x either a grid node or a mid-cell point
    qx = np.asarray(g.coords.x, dtype=float)
    from photonhub.components.grid import graded_primary_spacings
    dqx = np.asarray(graded_primary_spacings(tuple(qx)), dtype=float)
    mids = qx + 0.5 * dqx
    ok_x = np.concatenate([qx, mids])
    xs = np.array(sorted({d.center_um[0] for d in dips}))
    assert all(np.min(np.abs(ok_x - x)) < 1e-9 for x in xs)
    # amplitudes use LOCAL propagation-axis widths: J and M sheets differ when
    # the plane sits in a graded-z region — at minimum both sheets exist
    kinds = {d.polarization[0] for d in dips}
    assert kinds == {"E", "H"}


def test_monitor_snap_uses_local_spacing_on_graded_axis():
    g = auto_grid(size_um=L, wavelength_um=LAM,
                  structures=_sim(ph.UniformGridSpec(dl_um=0.05)).structures,
                  background_index=NCLAD, steps_per_wvl=25.0)
    sim = _sim(g)
    # z grades (structure spans z); the snapped plane must sit on a LOCAL
    # quarter point with the LOCAL width, not the base dl_um
    pos, dl_local = snap_mixed_plane(sim, 2, 0.5)
    qz = np.asarray(g.coords.z, dtype=float)
    from photonhub.components.grid import graded_primary_spacings
    dqz = np.asarray(graded_primary_spacings(tuple(qz)), dtype=float)
    quarters = qz + 0.25 * dqz
    k = int(np.argmin(np.abs(quarters - 0.5)))
    assert pos == pytest.approx(quarters[k], abs=1e-12)
    assert dl_local == pytest.approx(dqz[k], abs=1e-12)
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0,
                                       dl_um=g.dl_um, **WIN)
    mm = mode_monitor(sim, mode, axis="z", position_um=0.5, freqs_hz=(F0,),
                      name="m", direction="+", center_um=(XC, YC),
                      thickness_axis="y")
    assert mm.dl_um == pytest.approx(dl_local, abs=1e-12)


def test_uniform_snap_behavior_unchanged():
    dl = 0.05
    sim = _sim(ph.UniformGridSpec(dl_um=dl))
    pos, w = snap_mixed_plane(sim, 2, 0.5)
    k = round(0.5 / dl - 0.25)
    assert pos == (k + 0.25) * dl and w == dl


# --------------------------------------------------------------------------- #
# Engine-gated: the full graded pipeline reproduces the fine-uniform T.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(find_solver() is None, reason="needs a phsolver binary")
def test_automesh_transmission_matches_fine_uniform():
    spw = 18.0
    dlf = LAM / (NCORE * spw)
    # the uniform reference domain must be dl-commensurate (the established
    # scene convention — non-commensurate uniform domains fail the engine's
    # realized-domain checks and are out of scope here)
    Ls = tuple(round(l / dlf) * dlf for l in (2.6, 2.0, 4.5))
    core = ph.Structure(
        geometry=ph.Box(center_um=(1.3, 1.0, Ls[2] / 2),
                        size_um=(CW, CH, Ls[2] * 4)),
        medium=ph.Medium(permittivity=NCORE ** 2))
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0)

    def scene(grid, dl_solve):
        sim0 = ph.Simulation(
            size_um=Ls, grid=grid, run={"n_steps": 5000},
            background=ph.Background(permittivity=NCLAD ** 2),
            structures=(core,),
            boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
            pml_num_layers=10,
            sources=[ph.PointDipole(center_um=(1.3, 1.0, 0.8),
                                    polarization="Ex", source_time=pulse)])
        win = dict(h_center_um=1.3, v_center_um=1.0, half_w_um=0.8,
                   half_v_um=0.6)
        mode = solve_mode_on_cross_section(sim0, "z", 1.0, LAM, "TE", 0,
                                           dl_um=dl_solve, **win)
        srcs = mode_launch(sim0, mode, axis="z", position_um=1.0,
                           source_time=pulse, center_um=(1.3, 1.0))
        mons = {nm: mode_monitor(sim0, mode, axis="z", position_um=zp,
                                 freqs_hz=(F0,), name=nm, direction="+",
                                 center_um=(1.3, 1.0), thickness_axis="y")
                for nm, zp in (("in", 1.5), ("out", 3.4))}
        sim = sim0.model_copy(update=dict(
            sources=tuple(srcs),
            monitors=tuple(m.field_monitor for m in mons.values())))
        data = ph.run_local(sim, solver_path=find_solver(), quiet=True,
                            timeout=1800)
        p = {nm: m.mode_power(data)[F0] for nm, m in mons.items()}
        return p["out"] / p["in"], srcs

    g = auto_grid(size_um=Ls, wavelength_um=LAM, structures=(core,),
                  background_index=NCLAD, steps_per_wvl=spw)
    t_graded, srcs = scene(g, g.dl_um)
    assert {s.type for s in srcs} == {"point_dipole"}   # graded eq-current
    t_uniform, _ = scene(ph.UniformGridSpec(dl_um=dlf), dlf)
    # straight guide: both ~1; graded must reproduce the uniform answer
    assert t_uniform == pytest.approx(1.0, abs=0.03)
    assert t_graded == pytest.approx(t_uniform, abs=5e-3)
