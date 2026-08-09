"""Regression gate: PhotonHub ``mode_overlap`` ≡ Tidy3D ``ModeData.dot``.

Locks in the formula parity established in ``benchmarks/tidy3d/mode_overlap_parity.py``
as a committed test so a future change to the overlap kernel can't silently drift
away from the reference. SKIPPED when tidy3d is not installed (it lives in the dev
venv, not the runtime deps).

Method — the tightest possible comparison: solve real waveguide modes with Tidy3D's
LOCAL mode solver (no cloud / no API key), wrap each mode's own tangential fields as
a PhotonHub ``VectorMode``, and feed them to ``mode_overlap`` on the SAME grid. Any
disagreement with Tidy3D's ``dot`` is then purely the overlap formula + quadrature
(the modes are byte-identical), which our Snyder & Love power coupling reproduces to
the colocation floor (~1e-4).
"""

import numpy as np
import pytest

td = pytest.importorskip("tidy3d")
from tidy3d.plugins.mode import ModeSolver as TDMS  # noqa: E402

from photonhub.plugins import mode_overlap  # noqa: E402
from photonhub.plugins.vector_modes import VectorMode  # noqa: E402

WL_UM = 1.31
F0 = td.C_0 / WL_UM
N_SI, N_SIO2 = 3.5, 1.444
DL = 0.025
WIN = (2.0, 1.5)


def _td_solve(core_w_um, num_modes=1):
    core = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(core_w_um, 0.22, td.inf)),
        medium=td.Medium(permittivity=N_SI ** 2))
    sim = td.Simulation(
        size=(WIN[0], WIN[1], 1.0), structures=[core],
        medium=td.Medium(permittivity=N_SIO2 ** 2),
        grid_spec=td.GridSpec.uniform(dl=DL),
        boundary_spec=td.BoundarySpec.all_sides(td.Periodic()),
        sources=[], monitors=[], run_time=1e-13)
    return TDMS(
        simulation=sim, plane=td.Box(center=(0, 0, 0), size=(WIN[0], WIN[1], 0)),
        mode_spec=td.ModeSpec(num_modes=num_modes, target_neff=N_SI),
        freqs=[F0]).solve()


def _to_vmode(data, mi=0):
    def g(name):
        da = getattr(data, name).isel(f=0, mode_index=mi).squeeze(drop=True)
        return np.asarray(da.transpose("y", "x").values)
    x = np.asarray(data.Ex.coords["x"].values)
    y = np.asarray(data.Ex.coords["y"].values)
    neff = float(np.asarray(data.n_eff.isel(mode_index=mi).values).ravel()[0])
    vm = VectorMode(
        n_eff=neff, n_group=None, ex=g("Ex"), ey=g("Ey"), ez=g("Ez"),
        hx=g("Hx"), hy=g("Hy"), hz=g("Hz"),
        wavelength_um=WL_UM, dl_x_um=float(x[1] - x[0]), dl_y_um=float(y[1] - y[0]))
    return vm, x, y


def _td_coupling(data_a, data_b):
    return float(np.abs(np.asarray(data_a.dot(data_b).values).ravel()[0]) ** 2)


@pytest.fixture(scope="module")
def td_modes():
    """Tidy3D-solved TE0 of two strips (0.45, 0.55 µm) on the shared 2.0×1.5 grid."""
    return {w: _td_solve(w) for w in (0.45, 0.55)}


def test_power_matches_tidy3d_dot_on_identical_fields(td_modes):
    """ph.mode_overlap(.power) reproduces Tidy3D |dot|² to the colocation floor when
    fed Tidy3D's own mode fields (so only the formula+quadrature is under test)."""
    va, x, y = _to_vmode(td_modes[0.45])
    vb, _, _ = _to_vmode(td_modes[0.55])
    r = mode_overlap(va, vb, grid=(x, y))
    td_c = _td_coupling(td_modes[0.45], td_modes[0.55])
    assert r.method == "snyder_love"
    assert abs(r.power - td_c) < 5e-4
    # the complex coupling magnitude is consistent with the power efficiency
    assert abs(r.coupling) ** 2 == pytest.approx(r.power, abs=1e-9)


def test_self_overlap_matches_tidy3d(td_modes):
    """Both tools normalize a mode's self-overlap to exactly 1."""
    va, x, y = _to_vmode(td_modes[0.45])
    assert mode_overlap(va, va, grid=(x, y)).power == pytest.approx(1.0, abs=1e-6)
    assert _td_coupling(td_modes[0.45], td_modes[0.45]) == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_modes_match_tidy3d():
    """TE0/TE1 of one multimode guide: both tools read ~0 mutual coupling."""
    data = _td_solve(1.2, num_modes=2)
    v0, x, y = _to_vmode(data, mi=0)
    v1, _, _ = _to_vmode(data, mi=1)
    ph = mode_overlap(v0, v1, grid=(x, y)).power
    M = np.abs(np.asarray(data.outer_dot(data).values)) ** 2
    td_c = M.reshape(M.shape[-2], M.shape[-1])[0, 1]
    assert ph < 5e-3
    assert td_c < 5e-3
