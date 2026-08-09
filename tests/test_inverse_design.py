"""Adjoint inverse-design: geometry/mapping/assembly math (solver-free) plus
gradient-vs-finite-difference and optimization integration tests (skipped when
no phsolver binary is built)."""

import numpy as np
import pytest
import xarray as xr

import photonhub as ph
from photonhub import inverse_design as idz
from photonhub.runners.local import find_solver

C0 = 2.99792458e8


def _real_solver():
    try:
        return find_solver()
    except ph.SolverRunError:
        return None


SKIP_NO_SOLVER = pytest.mark.skipif(
    _real_solver() is None,
    reason="no phsolver binary found (build the engine first)")


# ---------------------------------------------------------------------------
# DesignRegion geometry (no solver)
# ---------------------------------------------------------------------------

def test_on_grid_snaps_and_divides():
    r = ph.DesignRegion.on_grid(center_um=(1.0, 1.0, 0.4), size_um=(0.45, 0.45, 0.24),
                                dl_um=0.04, shape=(4, 4, 1))
    # span rounds UP to a multiple of the pixel count on each axis
    for (a0, a1), n in zip(r.cells, r.shape):
        assert (a1 - a0) % n == 0
    assert r.n_params == 16
    assert r.cells_per_pixel == (3, 3, 6)         # 12/4, 12/4, 6/1
    assert np.allclose(r.size_um, (0.48, 0.48, 0.24))


def test_rejects_nondividing_shape():
    with pytest.raises(ValueError, match="divide evenly"):
        ph.DesignRegion(dl_um=0.04, cells=((0, 7), (0, 4), (0, 4)), shape=(2, 2, 2))


def test_rejects_subvacuum_eps():
    with pytest.raises(ValueError, match="eps_min"):
        ph.DesignRegion(dl_um=0.04, cells=((0, 4), (0, 4), (0, 4)), shape=(2, 2, 2),
                        eps_min=0.5)


def test_eps_mapping_is_linear():
    r = ph.DesignRegion(dl_um=0.04, cells=((0, 4), (0, 4), (0, 2)), shape=(2, 2, 1),
                        eps_min=1.0, eps_max=11.0)
    assert r.eps(np.zeros(r.shape)).max() == pytest.approx(1.0)
    assert r.eps(np.ones(r.shape)).min() == pytest.approx(11.0)
    assert float(r.eps(np.full(r.shape, 0.5)).flat[0]) == pytest.approx(6.0)


def test_structures_count_eps_and_tiling():
    r = ph.DesignRegion(dl_um=0.05, cells=((4, 10), (4, 10), (2, 5)), shape=(2, 2, 1),
                        eps_min=1.0, eps_max=9.0)
    rho = np.array([[[0.0], [0.25]], [[0.5], [1.0]]])   # shape (2,2,1)
    structs = r.structures(rho)
    assert len(structs) == r.n_params
    eps = sorted(s.medium.permittivity for s in structs)
    assert eps == pytest.approx([1.0, 3.0, 5.0, 9.0])   # 1 + rho*8
    # boxes tile without overlap: every box face sits on an integer cell line
    for s in structs:
        for c, sz in zip(s.geometry.center_um, s.geometry.size_um):
            lo, hi = c - sz / 2, c + sz / 2
            assert lo / r.dl_um == pytest.approx(round(lo / r.dl_um))
            assert hi / r.dl_um == pytest.approx(round(hi / r.dl_um))


def test_monitor_is_multicomponent_quarter_cell():
    r = ph.DesignRegion(dl_um=0.04, cells=((10, 16), (10, 16), (4, 8)), shape=(3, 3, 2))
    m = r.monitor(3e14)
    assert m.fields == ("Ex", "Ey", "Ez")
    # quarter-cell faces => (low face)/dl has fractional part 0.25
    for c, sz in zip(m.center_um, m.size_um):
        lo = (c - sz / 2) / r.dl_um
        assert lo - np.floor(lo) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# cell -> pixel scatter map (no solver): build a coord-bearing DataArray
# ---------------------------------------------------------------------------

def _dft_like(region, freq=3e14):
    (i0, i1), (j0, j1), (k0, k1) = region.cells
    dl = region.dl_um
    x = np.arange(i0, i1) * dl
    y = np.arange(j0, j1) * dl
    z = np.arange(k0, k1) * dl
    data = np.zeros((1, 3, z.size, y.size, x.size), dtype=complex)
    return xr.DataArray(
        data, dims=("f", "component", "z", "y", "x"),
        coords={"f": [freq], "component": ["Ex", "Ey", "Ez"], "z": z, "y": y, "x": x})


def test_pixel_index_assigns_cells():
    r = ph.DesignRegion(dl_um=0.05, cells=((10, 16), (10, 16), (0, 3)), shape=(3, 3, 1))
    da = _dft_like(r).sel(f=3e14)
    pix = r._pixel_index(da)                # (nz, ny, nx)
    assert pix.shape == (3, 6, 6)
    # x cells 10..15 -> px 0,0,1,1,2,2 (cells/pixel = 2)
    # flat pixel index is row-major [ix, iy, iz]; iz=0 always here
    # bottom-left cell (x=10,y=10) -> pixel (0,0,0) -> flat 0
    assert pix[0, 0, 0] == 0
    # top-right cell (x=15,y=15) -> pixel (2,2,0) -> flat (2*3+2)*1 = 8
    assert pix[0, -1, -1] == 8
    assert set(np.unique(pix)) == set(range(9))    # all 9 pixels covered, no -1


def test_pixel_index_marks_out_of_region():
    r = ph.DesignRegion(dl_um=0.05, cells=((10, 14), (10, 14), (0, 2)), shape=(2, 2, 1))
    da = _dft_like(r).sel(f=3e14)
    # append a stray cell one past the high x face
    x2 = np.append(da.coords["x"].values, 14 * 0.05)
    da2 = da.reindex(x=x2)
    pix = r._pixel_index(da2)
    assert (pix[..., -1] == -1).all()              # stray column unmapped


# ---------------------------------------------------------------------------
# gradient assembly (no solver): synthetic uniform fields
# ---------------------------------------------------------------------------

def test_assemble_gradient_uniform_fields():
    r = ph.DesignRegion(dl_um=0.05, cells=((0, 4), (0, 4), (0, 6)), shape=(2, 2, 1),
                        eps_min=1.0, eps_max=3.0)
    da = _dft_like(r)
    da = da + (1.0 + 0.0j)                          # all components/cells = 1
    fwd = {"design_region": da}
    adj = {"design_region": da}
    # prod = sum_c 1*1 = 3 per cell; cells/pixel = 2*2*6 = 24 -> pix_prod = 72
    g = idz.assemble_gradient(r, fwd, adj, coeff=1.0 + 0j, freq_hz=3e14)
    assert g.shape == (r.n_params,)
    expected = np.real(idz.BETA * 72) * (r.eps_max - r.eps_min)
    assert np.allclose(g, expected)                # uniform field -> uniform grad


def test_assemble_gradient_coeff_is_linear():
    r = ph.DesignRegion(dl_um=0.05, cells=((0, 4), (0, 4), (0, 2)), shape=(2, 2, 1))
    da = _dft_like(r) + (1.0 + 0.0j)
    d = {"design_region": da}
    g1 = idz.assemble_gradient(r, d, d, coeff=1.0 + 0j, freq_hz=3e14)
    g2 = idz.assemble_gradient(r, d, d, coeff=2.0 + 0j, freq_hz=3e14)
    assert np.allclose(g2, 2.0 * g1)


# ---------------------------------------------------------------------------
# objective (no solver)
# ---------------------------------------------------------------------------

def test_point_intensity_objective_pieces():
    obj = ph.PointIntensity(probe_um=(1.5, 1.0, 0.4), freq_hz=3e14, component="Ez")
    mon = obj.monitor()
    assert mon.fields == ("Ez",) and mon.size_um == (0.0, 0.0, 0.0)

    pulse = ph.GaussianPulse(freq0_hz=3e14, fwidth_hz=4e13)
    src = ph.PointDipole(center_um=(0.5, 1.0, 0.4), polarization="Ez", source_time=pulse)
    sim = ph.Simulation(size_um=(2, 2, 0.8), grid=ph.UniformGridSpec(dl_um=0.05),
                        run={"n_steps": 10}, sources=[src],
                        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
                        pml_num_layers=6)
    a = obj.adjoint_source(sim)
    assert isinstance(a, ph.PointDipole)
    assert a.center_um == obj.probe_um and a.polarization == "Ez"
    assert a.amplitude == 1.0 and a.source_time is pulse   # reuse fwd pulse


# ---------------------------------------------------------------------------
# integration: gradient vs finite differences, and optimization (need solver)
# ---------------------------------------------------------------------------

def _tiny_problem(dl=0.05, shape=(2, 2, 1), eps_max=6.0):
    f0 = C0 / (1.0e-6)
    LX, LY, LZ = 1.6, 1.6, 0.8
    zc = LZ / 2
    region = ph.DesignRegion.on_grid(center_um=(LX / 2, LY / 2, zc),
                                     size_um=(0.4, 0.4, 0.2), dl_um=dl, shape=shape,
                                     eps_min=1.0, eps_max=eps_max)
    obj = ph.PointIntensity(probe_um=(LX / 2 + 0.45, LY / 2, zc), freq_hz=f0,
                            component="Ez")
    src = ph.PointDipole(center_um=(0.35, LY / 2, zc), polarization="Ez",
                         source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.15 * f0))

    def build(rho):
        return ph.Simulation(
            size_um=(LX, LY, LZ), grid=ph.UniformGridSpec(dl_um=dl),
            run={"n_steps": 2500}, background=ph.Background(permittivity=1.0),
            structures=region.structures(rho),
            boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
            pml_num_layers=6, sources=[src],
            monitors=(region.monitor(f0), obj.monitor()))

    rng = np.random.default_rng(3)
    rho = rng.uniform(0.3, 0.7, size=region.shape)
    return build, region, obj, rho


@SKIP_NO_SOLVER
def test_adjoint_gradient_matches_finite_difference():
    build, region, obj, rho = _tiny_problem()
    res = ph.value_and_gradient(build, region, obj, rho)
    g_adj = res.grad

    # central FD on every pixel (2N solves)
    flat = rho.ravel().copy()
    g_fd = np.zeros_like(flat)
    delta = 0.03
    for i in range(flat.size):
        rp = flat.copy(); rp[i] += delta
        rm = flat.copy(); rm[i] -= delta
        Jp = obj.value(ph.run_local(build(rp.reshape(region.shape)), quiet=True))
        Jm = obj.value(ph.run_local(build(rm.reshape(region.shape)), quiet=True))
        g_fd[i] = (Jp - Jm) / (2 * delta)

    cos = float(g_adj @ g_fd / (np.linalg.norm(g_adj) * np.linalg.norm(g_fd)))
    assert cos > 0.9, f"adjoint/FD cosine {cos:.3f} too low"


@SKIP_NO_SOLVER
def test_optimize_increases_objective():
    build, region, obj, rho = _tiny_problem()
    res = ph.optimize(build, region, obj, rho, n_iters=6,
                      run_kwargs=dict(quiet=True))
    # L-BFGS line-searches, so the LAST eval need not be the best; the optimizer
    # improved the objective over the start.
    assert res.best.value > res.history[0]
    assert len(res.best.grad) == region.n_params


# ---------------------------------------------------------------------------
# ModePower objective (mode solve is pure-Python; FDTD pieces are solver-gated)
# ---------------------------------------------------------------------------

from photonhub.plugins import ModeSolver, mode_source  # noqa: E402

F0_WG = 1.934e14   # ~1.55 um
N_CORE, N_CLAD = 3.5, 1.444


def _te0_mode(dl=0.05):
    return ModeSolver.from_rectangular_core(
        wavelength_um=2.99792458e8 / F0_WG * 1e6, dl_um=dl, core_w_um=0.45,
        core_h_um=0.22, n_core=N_CORE, n_clad=N_CLAD).solve(
            num_modes=1, polarization="TE", n_guess=3.0)[0]


def _wg_shell(LX=1.4, LY=1.4, LZ=2.2, dl=0.05):
    pulse = ph.GaussianPulse(freq0_hz=F0_WG, fwidth_hz=0.1 * F0_WG)
    return ph.Simulation(
        size_um=(LX, LY, LZ), grid=ph.UniformGridSpec(dl_um=dl),
        run={"n_steps": 1}, background=ph.Background(permittivity=N_CLAD ** 2),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"), pml_num_layers=6,
        sources=[ph.PointDipole(center_um=(LX / 2, LY / 2, 0.4),
                                polarization="Ex", source_time=pulse)])


def test_modepower_construction_and_beta():
    obj = ph.ModePower(mode=_te0_mode(), axis="z", position_um=1.6,
                       freq_hz=F0_WG, direction="+", name="out")
    assert obj.beta == idz.BETA_MODE
    assert obj.beta != idz.BETA          # distinct from the point-dipole constant


def test_modepower_monitor_is_four_tangential():
    obj = ph.ModePower(mode=_te0_mode(), axis="z", position_um=1.6, freq_hz=F0_WG,
                       name="out")
    mon = obj.monitor(_wg_shell())
    assert set(mon.fields) == {"Ex", "Ey", "Hx", "Hy"}   # _TANGENTIAL["z"]
    assert mon.name == "out"
    assert mon.size_um[2] == 0.0                          # a z-plane


def test_modepower_adjoint_source_is_backward_mode():
    shell = _wg_shell()
    obj = ph.ModePower(mode=_te0_mode(), axis="z", position_um=1.6, freq_hz=F0_WG,
                       direction="+", name="out")
    a = obj.adjoint_source(shell)
    assert isinstance(a, ph.ModeSource)
    assert a.direction == "-"             # forward "+" -> adjoint launches "-"
    assert a.amplitude == 1.0
    assert a.source_time is shell.sources[0].source_time   # reuse fwd pulse


def _wg_problem(shape=(3, 2, 1), dl=0.05):
    LX = LY = 1.4
    LZ = 2.2
    cx = cy = LX / 2
    mode = _te0_mode(dl)
    pulse = ph.GaussianPulse(freq0_hz=F0_WG, fwidth_hz=0.1 * F0_WG)
    region = ph.DesignRegion.on_grid(
        center_um=(cx, cy, LZ / 2), size_um=(0.5, 0.3, 0.5), dl_um=dl,
        shape=shape, eps_min=N_CLAD ** 2, eps_max=N_CORE ** 2)
    obj = ph.ModePower(mode=mode, axis="z", position_um=1.75, freq_hz=F0_WG,
                       direction="+", name="out")
    base = dict(size_um=(LX, LY, LZ), grid=ph.UniformGridSpec(dl_um=dl),
                run={"n_steps": 2200},
                background=ph.Background(permittivity=N_CLAD ** 2),
                boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
                pml_num_layers=6)
    shell = ph.Simulation(**base, sources=[ph.PointDipole(
        center_um=(cx, cy, 0.45), polarization="Ex", source_time=pulse)])
    src = mode_source(shell, mode, axis="z", position_um=0.45,
                      source_time=pulse, direction="+")
    out_mon = obj.monitor(shell)
    core = ph.Structure(geometry=ph.Box(center_um=(cx, cy, LZ / 2),
                                        size_um=(0.45, 0.22, LZ * 2)),
                        medium=ph.Medium(permittivity=N_CORE ** 2))

    def build(rho):
        return ph.Simulation(**base, sources=[src],
                             structures=(core,) + region.structures(rho),
                             monitors=(region.monitor(F0_WG), out_mon))

    rho = np.random.default_rng(1).uniform(0.4, 0.9, size=region.shape)
    return build, region, obj, rho


@SKIP_NO_SOLVER
def test_modepower_gradient_matches_finite_difference():
    build, region, obj, rho = _wg_problem()
    res = ph.value_and_gradient(build, region, obj, rho)
    flat = rho.ravel().copy()
    g_fd = np.zeros_like(flat)
    delta = 0.03
    for i in range(flat.size):
        rp = flat.copy(); rp[i] += delta
        rm = flat.copy(); rm[i] -= delta
        Jp = obj.value(ph.run_local(build(rp.reshape(region.shape)), quiet=True))
        Jm = obj.value(ph.run_local(build(rm.reshape(region.shape)), quiet=True))
        g_fd[i] = (Jp - Jm) / (2 * delta)
    cos = float(res.grad @ g_fd / (np.linalg.norm(res.grad) * np.linalg.norm(g_fd)))
    # coarse 6-pixel guard; the rigorous validation (benchmarks/adjoint/
    # mode_power_check.py) reaches cos 0.99 at the pinning grid.
    assert cos > 0.85, f"ModePower adjoint/FD cosine {cos:.3f} too low"


# ---------------------------------------------------------------------------
# device routing (the two solves run on the chosen backend)
# ---------------------------------------------------------------------------

@SKIP_NO_SOLVER
def test_device_param_threads_to_solves():
    build, region, obj, rho = _tiny_problem()
    res_cpu = ph.value_and_gradient(build, region, obj, rho, device="cpu")
    assert res_cpu.grad.shape == (region.n_params,)
    assert np.all(np.isfinite(res_cpu.grad))
    # explicit device="cpu" must reproduce the default backend exactly
    res_default = ph.value_and_gradient(build, region, obj, rho)
    assert np.allclose(res_cpu.grad, res_default.grad)
    # an explicit device also overrides one inside run_kwargs
    res_kw = ph.value_and_gradient(build, region, obj, rho, device="cpu",
                                   run_kwargs=dict(device="gpu"))
    assert np.allclose(res_kw.grad, res_default.grad)


# ---------------------------------------------------------------------------
# param_map: a projected gradient that constrains the design (e.g. symmetry)
# ---------------------------------------------------------------------------

@SKIP_NO_SOLVER
def test_optimize_param_map_constrains_design():
    build, region, obj, rho = _tiny_problem(shape=(2, 2, 1))

    def x_mirror(r):                       # average a pixel with its x-flip
        return 0.5 * (r + r[::-1, :, :])

    res = ph.optimize(build, region, obj, rho, n_iters=4, step=0.1,
                      param_map=x_mirror, run_kwargs=dict(quiet=True))
    # the returned design lives in the constrained (mirror-symmetric) subspace
    assert np.allclose(res.rho, res.rho[::-1, :, :])


# ---------------------------------------------------------------------------
# optimize_parametric: adjoint gradient chained to a few shape parameters
# ---------------------------------------------------------------------------

@SKIP_NO_SOLVER
def test_optimize_parametric_improves_objective():
    build, region, obj, _ = _tiny_problem(shape=(4, 4, 1))

    half = region.shape[0] // 2

    def expand(p):                          # 2 params: left-half / right-half density
        g = np.empty(region.shape)
        g[:half] = np.clip(p[0], 0.0, 1.0)
        g[half:] = np.clip(p[1], 0.0, 1.0)
        return g

    res = ph.optimize_parametric(build, region, obj, np.array([0.4, 0.6]),
                                 expand, n_iters=5, bounds=(0.0, 1.0),
                                 run_kwargs=dict(quiet=True))
    assert res.params.shape == (2,)
    # the chained parameter gradient improves the objective ...
    assert res.best.value >= res.history[0]
    # ... and the returned density is exactly expand(optimized params)
    assert np.allclose(res.rho, expand(res.params))
