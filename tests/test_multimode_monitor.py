"""Multi-mode / multi-frequency mode monitor — physics + API pins.

Covers the multi-mode generalization of the single-mode readout:

* :func:`photonhub.plugins.mode_devices.solve_mode_bank` — solve N modes at M
  frequencies into ``{freq_hz: {mode_index: Mode}}`` (the readout-side analogue
  of Tidy3D ``ModeMonitor(ModeSpec(num_modes=N), num_freqs=M)``);
* :func:`photonhub.plugins.mode_overlap.mode_decomposition` — project a recorded
  plane onto EACH mode in the bank → ``{mode_index: {freq_hz: value}}``;
* :meth:`photonhub.plugins.mode_devices.ModeMonitor.mode_decomposition`.

The synthetic planes are built with the SAME scalar-limit reconstruction the
overlap uses (E = e_mode, H = ±h_mode), so a clean single-mode plane reads
forward T==1 in its own index and ~0 in the others (modal orthogonality), and a
scaled superposition splits power across indices by the squared coefficients.
Co-location is OFF here (the analytic E/H are already at the same points), as in
``test_mode_overlap``.
"""

import functools
import math

import numpy as np
import pytest
import xarray as xr

import photonhub as ph
from photonhub.plugins import (
    Mode,
    ModeSolver,
    VectorModeSolver,
    mode_decomposition as _mode_decomposition,
    mode_monitor,
    mode_transmission as _mode_transmission,
    solve_mode_bank,
)
from photonhub.plugins.mode_devices import C0
from photonhub.plugins.mode_overlap import modal_fields, vector_modal_fields

# Synthetic, already-collocated analytic fields -> co-location off (see module doc).
mode_decomposition = functools.partial(_mode_decomposition, colocate=False)
mode_transmission = functools.partial(_mode_transmission, colocate=False)

WL_UM = 1.31
DL_UM = 0.04
CORE_W_UM, CORE_H_UM = 1.0, 0.22  # wide strip: 4 guided TE modes @1.31 um
N_SI, N_SIO2 = 3.5, 1.444
F0 = C0 / (WL_UM * 1e-6)
# A small O-band-ish band for the per-frequency / dispersion tests.
WAVELENGTHS = (1.28, 1.31, 1.34)
FREQS = tuple(C0 / (wl * 1e-6) for wl in WAVELENGTHS)


def _solver(wavelength_um=WL_UM):
    return ModeSolver.from_rectangular_core(
        wavelength_um=wavelength_um, dl_um=DL_UM,
        core_w_um=CORE_W_UM, core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2)


def _vsolver(wavelength_um=WL_UM):
    return VectorModeSolver.from_rectangular_core(
        wavelength_um=wavelength_um, dl_um=DL_UM,
        core_w_um=CORE_W_UM, core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2)


@pytest.fixture(scope="module")
def modes():
    """The first three guided TE modes of the wide strip (descending n_eff)."""
    return _solver().solve(num_modes=3, polarization="TE")


def _plane_axes(mode, *, pad_cells=8, dl_um=DL_UM):
    ny, nx = mode.field.shape
    nX, nY = nx + 2 * pad_cells, ny + 2 * pad_cells
    x = (np.arange(nX) - (nX - 1) / 2.0) * dl_um
    y = (np.arange(nY) - (nY - 1) / 2.0) * dl_um
    return x, y


def _superpose_plane(terms, x, y, *, direction="+", as_dft=False, freqs_hz=(F0,),
                     vector=False):
    """Assemble the four tangential plane components for a sum of modal fields:
    ``terms`` is a list of ``(mode, coeff)``; the plane carries
    ``sum(coeff * field(mode))`` travelling in ``direction``. With ``as_dft`` the
    SAME field is broadcast over every frequency in ``freqs_hz`` (dims
    (f, component, z, y, x)); otherwise a plain 2-D (y, x) DataArray per freq=F0.
    """
    e1 = e2 = h1 = h2 = 0.0
    for mode, coeff in terms:
        if vector:
            m = vector_modal_fields(mode, x, y, axis="z", direction=direction,
                                    center_um=(0.0, 0.0))
        else:
            m = modal_fields(mode, x, y, axis="z", direction=direction,
                             center_um=(0.0, 0.0))
        e1 = e1 + coeff * m["e1"]
        e2 = e2 + coeff * m["e2"]
        h1 = h1 + coeff * m["h1"]
        h2 = h2 + coeff * m["h2"]
    comps = {"Ex": e1, "Ey": e2, "Hx": h1, "Hy": h2}
    out = {}
    for name, arr in comps.items():
        arr = np.asarray(arr, dtype=np.complex128)
        if as_dft:
            stack = np.broadcast_to(arr[None, None, None, :, :],
                                    (len(freqs_hz), 1, 1) + arr.shape)
            da = xr.DataArray(
                stack, dims=("f", "component", "z", "y", "x"),
                coords={"f": list(freqs_hz), "component": [name], "z": [0.0],
                        "y": y, "x": x})
        else:
            da = xr.DataArray(arr, dims=("y", "x"), coords={"y": y, "x": x})
        out[name] = da
    return out


# --------------------------------------------------------------------------
# solve_mode_bank
# --------------------------------------------------------------------------

def test_solve_mode_bank_structure():
    s = _solver()
    bank = solve_mode_bank(s, FREQS, mode_indices=(0, 1, 2),
                           polarization="TE", n_guess=2.8)
    assert tuple(bank.keys()) == FREQS                # freq keys, input order
    for f, inner in bank.items():
        assert sorted(inner) == [0, 1, 2]             # ascending mode indices
        assert all(isinstance(m, Mode) for m in inner.values())
        # each mode solved AT its own frequency, and ordered by descending n_eff.
        assert all(m.wavelength_um == pytest.approx(C0 / f * 1e6)
                   for m in inner.values())
        assert inner[0].n_eff > inner[1].n_eff > inner[2].n_eff


def test_solve_mode_bank_default_fundamental_only():
    bank = solve_mode_bank(_solver(), (F0,), polarization="TE", n_guess=2.8)
    assert list(bank[F0]) == [0]


def test_solve_mode_bank_collapses_and_sorts_indices():
    bank = solve_mode_bank(_solver(), (F0,), mode_indices=(2, 0, 2),
                           polarization="TE", n_guess=2.8)
    assert sorted(bank[F0]) == [0, 2]                 # dedup + sort, num_modes bumped


def test_solve_mode_bank_dispersion_per_index():
    bank = solve_mode_bank(_solver(), FREQS, mode_indices=(0, 1),
                           polarization="TE", n_guess=2.8)
    for idx in (0, 1):
        neff = [bank[f][idx].n_eff for f in FREQS]    # FREQS ascending
        assert neff[0] > neff[-1]                      # higher freq more confined
        assert max(neff) - min(neff) > 1e-3            # a resolvable drift


def test_solve_mode_bank_rejects_bad_inputs():
    s = _solver()
    with pytest.raises(ValueError, match="non-empty"):
        solve_mode_bank(s, [], mode_indices=(0,))
    with pytest.raises(ValueError, match="mode_indices must be non-empty"):
        solve_mode_bank(s, (F0,), mode_indices=())
    with pytest.raises(ValueError, match="mode_indices must be >= 0"):
        solve_mode_bank(s, (F0,), mode_indices=(-1,))
    with pytest.raises(ValueError, match="> 0 Hz"):
        solve_mode_bank(s, (-1.0,), mode_indices=(0,))
    with pytest.raises(ValueError, match="returned only"):
        # 99 is far beyond the guided count.
        solve_mode_bank(s, (F0,), mode_indices=(99,), polarization="TE")


def test_solve_mode_bank_vector_solver():
    bank = solve_mode_bank(_vsolver(), FREQS, mode_indices=(0, 1), n_guess=2.8)
    assert tuple(bank.keys()) == FREQS
    assert all(math.isfinite(bank[f][i].n_eff)
               for f in FREQS for i in (0, 1))


# --------------------------------------------------------------------------
# mode_decomposition — orthogonality, superposition, directionality
# --------------------------------------------------------------------------

def test_decomposition_orthogonality(modes):
    """A plane that IS forward mode 0 reads T~=1 in index 0, ~=0 in 1 and 2."""
    m0, m1, m2 = modes
    x, y = _plane_axes(m0)
    plane = _superpose_plane([(m0, 1.0)], x, y, direction="+")
    bank = {0: m0, 1: m1, 2: m2}                       # frozen form
    T = mode_decomposition(plane, bank, axis="z", direction="+")
    assert sorted(T) == [0, 1, 2]
    assert T[0][0.0] == pytest.approx(1.0, rel=2e-3)
    assert abs(T[1][0.0]) < 1e-3
    assert abs(T[2][0.0]) < 1e-3


def test_decomposition_superposition_splits_power(modes):
    """Plane = a*mode0 + b*mode1 -> T[0]~=a^2, T[1]~=b^2 (orthogonal modes)."""
    m0, m1, m2 = modes
    a, b = 0.6, 0.8
    x, y = _plane_axes(m0)
    plane = _superpose_plane([(m0, a), (m1, b)], x, y, direction="+")
    T = mode_decomposition(plane, {0: m0, 1: m1, 2: m2}, axis="z", direction="+")
    assert T[0][0.0] == pytest.approx(a ** 2, rel=3e-3)
    assert T[1][0.0] == pytest.approx(b ** 2, rel=3e-3)
    assert abs(T[2][0.0]) < 2e-3


def test_decomposition_power_and_amplitude_consistent(modes):
    """quantity='power' == |c|^2 * P_mode, 'amplitude' is complex with
    |amp|^2 == transmission."""
    m0, m1, _ = modes
    a, b = 0.6, 0.8
    x, y = _plane_axes(m0)
    plane = _superpose_plane([(m0, a), (m1, b)], x, y, direction="+")
    bank = {0: m0, 1: m1}
    T = mode_decomposition(plane, bank, axis="z", quantity="transmission")
    P = mode_decomposition(plane, bank, axis="z", quantity="power")
    A = mode_decomposition(plane, bank, axis="z", quantity="amplitude")
    for idx in (0, 1):
        amp = A[idx][0.0]
        assert isinstance(amp, complex)
        assert abs(amp) ** 2 == pytest.approx(T[idx][0.0], rel=1e-9)
        # power / transmission == P_mode > 0 (same P_mode cancels into |c|^2).
        assert P[idx][0.0] / T[idx][0.0] > 0.0


def test_decomposition_directional(modes):
    """Forward mode 0 reads ~1 forward and ~0 backward in index 0."""
    m0, m1, _ = modes
    x, y = _plane_axes(m0)
    plane = _superpose_plane([(m0, 1.0)], x, y, direction="+")
    bank = {0: m0, 1: m1}
    fwd = mode_decomposition(plane, bank, axis="z", direction="+")
    bwd = mode_decomposition(plane, bank, axis="z", direction="-")
    assert fwd[0][0.0] == pytest.approx(1.0, rel=2e-3)
    assert abs(bwd[0][0.0]) < 1e-3


def test_decomposition_matches_single_mode_transmission(modes):
    """Index 0 of a multi-mode decomposition == the single-mode mode_transmission."""
    m0, m1, _ = modes
    x, y = _plane_axes(m0)
    plane = _superpose_plane([(m0, 0.7), (m1, 0.5)], x, y, direction="+")
    T_multi = mode_decomposition(plane, {0: m0, 1: m1}, axis="z")
    T_single = mode_transmission(plane, m0, axis="z")
    assert T_multi[0][0.0] == pytest.approx(T_single[0.0], rel=1e-9)


def test_decomposition_per_frequency_bank(modes):
    """A per-frequency bank {f: {idx: Mode}} on a DFT plane projects each freq onto
    its own mode; a clean forward mode-0 plane reads ~1 in index 0 at every freq."""
    s = _solver()
    bank = solve_mode_bank(s, FREQS, mode_indices=(0, 1),
                           polarization="TE", n_guess=2.8)
    # Build the recorded plane from the BAND-CENTRE mode-0 (a single guided wave).
    m0 = bank[FREQS[1]][0]
    x, y = _plane_axes(m0)
    plane = _superpose_plane([(m0, 1.0)], x, y, as_dft=True, freqs_hz=FREQS)
    T = mode_decomposition(plane, bank, axis="z", direction="+")
    assert sorted(T) == [0, 1]
    for f in FREQS:
        assert T[0][f] == pytest.approx(1.0, rel=1e-2)  # ~1 in fundamental
        assert abs(T[1][f]) < 5e-3                        # ~0 in TE1


def test_decomposition_rejects_bad_inputs(modes):
    m0, m1, _ = modes
    x, y = _plane_axes(m0)
    plane = _superpose_plane([(m0, 1.0)], x, y)
    with pytest.raises(ValueError, match="quantity must be one of"):
        mode_decomposition(plane, {0: m0}, axis="z", quantity="bogus")
    with pytest.raises(ValueError, match="empty"):
        mode_decomposition(plane, {}, axis="z")
    with pytest.raises(ValueError, match="mixes frozen .* per-frequency"):
        # mixing a Mode value with a per-frequency map value
        mode_decomposition(plane, {0: m0, 1: {0: m1}}, axis="z")


def test_decomposition_rejects_ragged_and_empty_banks(modes):
    """A per-frequency bank must be RECTANGULAR (same indices at every freq), and
    inner maps must hold modes — else the per-index nearest-frequency lookup would
    silently fabricate a reading at a frequency the caller never supplied."""
    m0, m1, _ = modes
    x, y = _plane_axes(m0)
    f0, f1 = FREQS[0], FREQS[1]
    plane = _superpose_plane([(m0, 1.0)], x, y, as_dft=True, freqs_hz=(f0, f1))
    # ragged: index 1 present only at f0
    with pytest.raises(ValueError, match="ragged"):
        mode_decomposition(plane, {f0: {0: m0, 1: m1}, f1: {0: m0}}, axis="z")
    # all inner maps empty
    with pytest.raises(ValueError, match="no mode indices"):
        mode_decomposition(plane, {f0: {}, f1: {}}, axis="z")
    # per-frequency form with a non-mode leaf value
    with pytest.raises(ValueError, match="inner values must be modes"):
        mode_decomposition(plane, {f0: {0: 123}, f1: {0: 123}}, axis="z")
    # per-frequency-first mixing (Mapping then Mode)
    with pytest.raises(ValueError, match="mixes per-frequency .* frozen"):
        mode_decomposition(plane, {f0: {0: m0}, f1: m1}, axis="z")


def test_decomposition_amplitude_recovers_phase(modes):
    """quantity='amplitude' carries the modal PHASE: a plane that is mode0 with a
    complex prefactor e^{iφ} reads c ≈ e^{iφ} in index 0."""
    m0, m1, _ = modes
    x, y = _plane_axes(m0)
    phi = 0.7  # radians
    plane = _superpose_plane([(m0, np.exp(1j * phi))], x, y, direction="+")
    A = mode_decomposition(plane, {0: m0, 1: m1}, axis="z", quantity="amplitude")
    c0 = A[0][0.0]
    assert abs(c0) == pytest.approx(1.0, rel=2e-3)
    assert np.angle(c0) == pytest.approx(phi, abs=2e-3)
    assert abs(A[1][0.0]) < 1e-3


def test_decomposition_power_retains_pmode_weighting(modes):
    """'transmission' (|c|^2) cancels each port's P_mode, but 'power'
    (|a_pm|^2/P_mode) keeps it. So two DIFFERENT modes carrying the SAME modal
    amplitude read EQUAL transmission but UNEQUAL power (P_mode0 != P_mode1)."""
    m0, m1, _ = modes
    x, y = _plane_axes(m0)
    a = 0.5
    plane0 = _superpose_plane([(m0, a)], x, y, direction="+")
    plane1 = _superpose_plane([(m1, a)], x, y, direction="+")
    T0 = mode_decomposition(plane0, {0: m0, 1: m1}, axis="z",
                            quantity="transmission")
    T1 = mode_decomposition(plane1, {0: m0, 1: m1}, axis="z",
                            quantity="transmission")
    P0 = mode_decomposition(plane0, {0: m0, 1: m1}, axis="z", quantity="power")
    P1 = mode_decomposition(plane1, {0: m0, 1: m1}, axis="z", quantity="power")
    # both modes carry |c|^2 == a^2 (P_mode cancels in transmission)
    assert T0[0][0.0] == pytest.approx(a ** 2, rel=3e-3)
    assert T1[1][0.0] == pytest.approx(a ** 2, rel=3e-3)
    # but the POWER differs because P_mode0 != P_mode1 (different mode order)
    pm0 = P0[0][0.0] / T0[0][0.0]   # == P_mode0
    pm1 = P1[1][0.0] / T1[1][0.0]   # == P_mode1
    assert abs(pm0 - pm1) / max(pm0, pm1) > 0.02   # a clear, real difference


def test_decomposition_perfreq_uses_dispersive_mode(modes):
    """The per-frequency bank actually projects each freq onto its OWN solved
    mode (dispersive n_eff), distinct from a single frozen mode."""
    s = _solver()
    bank = solve_mode_bank(s, FREQS, mode_indices=(0,), polarization="TE",
                           n_guess=2.8)
    # n_eff must differ across the band (the reason per-freq exists).
    neffs = [bank[f][0].n_eff for f in FREQS]
    assert max(neffs) - min(neffs) > 1e-3
    # Build the recorded plane from the band-EDGE mode (f0), broadcast to all freqs.
    medge = bank[FREQS[0]][0]
    x, y = _plane_axes(medge)
    plane = _superpose_plane([(medge, 1.0)], x, y, as_dft=True, freqs_hz=FREQS)
    T = mode_decomposition(plane, bank, axis="z", direction="+")
    # The band-edge freq projects onto its own mode -> ~1; the per-freq path is
    # exercised at every freq and stays physical (<= ~1).
    assert T[0][FREQS[0]] == pytest.approx(1.0, rel=5e-3)
    assert all(0.9 < T[0][f] <= 1.02 for f in FREQS)


def test_decomposition_vector_modes_orthogonality():
    """Full-vector modes (true transverse H) also decompose orthogonally."""
    vmodes = _vsolver().solve(num_modes=3, n_guess=2.8)
    v0, v1, v2 = vmodes[0], vmodes[1], vmodes[2]
    x, y = _plane_axes_vec(v0)
    plane = _superpose_plane([(v0, 1.0)], x, y, direction="+", vector=True)
    T = mode_decomposition(plane, {0: v0, 1: v1, 2: v2}, axis="z", direction="+")
    assert T[0][0.0] == pytest.approx(1.0, rel=5e-3)
    assert abs(T[1][0.0]) < 1e-2
    assert abs(T[2][0.0]) < 1e-2


def _plane_axes_vec(vmode, *, pad_cells=8, dl_um=DL_UM):
    ny, nx = vmode.ex.shape
    nX, nY = nx + 2 * pad_cells, ny + 2 * pad_cells
    x = (np.arange(nX) - (nX - 1) / 2.0) * dl_um
    y = (np.arange(nY) - (nY - 1) / 2.0) * dl_um
    return x, y


# --------------------------------------------------------------------------
# ModeMonitor.mode_decomposition integration
# --------------------------------------------------------------------------

def _monitor_data(plane_comps):
    """Stack per-component 2-D DataArrays into one (component, y, x) DataArray
    keyed by a monitor name, mimicking SimulationData[name]."""
    comps = ["Ex", "Ey", "Hx", "Hy"]
    da = xr.concat([plane_comps[c] for c in comps], dim="component")
    da = da.assign_coords(component=comps)
    return {"port": da}


def _monitor_data_dft(plane_comps):
    """Stack per-component DFT-shaped (f, component, z, y, x) DataArrays into one
    (f, component, z, y, x) DataArray keyed 'port' — mimics a multi-freq plane."""
    comps = ["Ex", "Ey", "Hx", "Hy"]
    da = xr.concat([plane_comps[c] for c in comps], dim="component")
    da = da.assign_coords(component=comps)
    return {"port": da}


def _sim_with_port():
    base = dict(
        size_um=(2.0, 1.6, 4.0),
        grid=ph.UniformGridSpec(dl_um=DL_UM),
        run=ph.RunSpec(n_steps=10),
        background=ph.Background(permittivity=N_SIO2 ** 2),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
    )
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=2e13)
    return ph.Simulation(**base, sources=[
        ph.PointDipole(center_um=(1.0, 0.8, 0.5), polarization="Ey",
                       source_time=pulse)])


def test_mode_monitor_perfreq_bank_and_overrides(modes):
    """ModeMonitor.mode_decomposition routes a stored per-frequency bank, and the
    direction= / quantity= / mode_bank= call overrides all take effect."""
    s = _solver()
    bank = solve_mode_bank(s, FREQS, mode_indices=(0, 1), polarization="TE",
                           n_guess=2.8)
    medge = bank[FREQS[0]][0]
    x, y = _plane_axes(medge)
    plane = _superpose_plane([(medge, 1.0)], x, y, direction="+",
                             as_dft=True, freqs_hz=FREQS)
    data = _monitor_data_dft(plane)
    sim = _sim_with_port()
    mm = mode_monitor(sim, medge, axis="z", position_um=2.0, freqs_hz=FREQS,
                      name="port", center_um=(0.0, 0.0), mode_bank=bank)

    # stored bank, forward: index 0 ~1 at the band-edge freq, backward ~0.
    Tf = mm.mode_decomposition(data, quantity="transmission", colocate=False)
    assert sorted(Tf) == [0, 1]
    assert Tf[0][FREQS[0]] == pytest.approx(1.0, rel=5e-3)
    Tb = mm.mode_decomposition(data, direction="-", colocate=False)
    assert abs(Tb[0][FREQS[0]]) < 5e-3                      # direction override
    # quantity override returns complex amplitudes
    A = mm.mode_decomposition(data, quantity="amplitude", colocate=False)
    assert isinstance(A[0][FREQS[0]], complex)
    # mode_bank= call arg overrides the stored bank (frozen single-index here)
    Tov = mm.mode_decomposition(data, mode_bank={0: medge}, colocate=False)
    assert list(Tov) == [0]


def test_mode_monitor_mode_decomposition_method(modes):
    m0, m1, m2 = modes
    x, y = _plane_axes(m0)
    plane = _superpose_plane([(m0, 0.6), (m1, 0.8)], x, y, direction="+")
    data = _monitor_data(plane)

    base = dict(
        size_um=(2.0, 1.6, 4.0),
        grid=ph.UniformGridSpec(dl_um=DL_UM),
        run=ph.RunSpec(n_steps=10),
        background=ph.Background(permittivity=N_SIO2 ** 2),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
    )
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=2e13)
    sim = ph.Simulation(
        **base,
        sources=[ph.PointDipole(center_um=(1.0, 0.8, 0.5),
                                polarization="Ey", source_time=pulse)],
    )
    mm = mode_monitor(sim, m0, axis="z", position_um=2.0, freqs_hz=(F0,),
                      name="port", center_um=(0.0, 0.0),
                      mode_bank={0: m0, 1: m1, 2: m2})
    assert mm.mode_bank is not None
    # synthetic analytic fields are already co-located -> colocate off (module doc)
    T = mm.mode_decomposition(data, quantity="transmission", colocate=False)
    assert T[0][0.0] == pytest.approx(0.6 ** 2, rel=3e-3)
    assert T[1][0.0] == pytest.approx(0.8 ** 2, rel=3e-3)
    assert abs(T[2][0.0]) < 2e-3


def test_mode_monitor_mode_decomposition_requires_bank(modes):
    m0, _, _ = modes
    x, y = _plane_axes(m0)
    data = _monitor_data(_superpose_plane([(m0, 1.0)], x, y))
    base = dict(
        size_um=(2.0, 1.6, 4.0),
        grid=ph.UniformGridSpec(dl_um=DL_UM),
        run=ph.RunSpec(n_steps=10),
        background=ph.Background(permittivity=N_SIO2 ** 2),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
    )
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=2e13)
    sim = ph.Simulation(**base, sources=[
        ph.PointDipole(center_um=(1.0, 0.8, 0.5), polarization="Ey",
                       source_time=pulse)])
    mm = mode_monitor(sim, m0, axis="z", position_um=2.0, freqs_hz=(F0,),
                      name="port", center_um=(0.0, 0.0))
    with pytest.raises(ValueError, match="no mode_bank"):
        mm.mode_decomposition(data)


def _mm_and_staggered_data(mode, *, a=1.0, b=0.3j):
    """A ModeMonitor (built from a real sim, so it captures dl_um) + a synthetic
    DFT plane carrying the engine's LONGITUDINAL Yee stagger: H = (a e^{iφ} −
    b e^{-iφ})·h, E = (a+b)·e (φ = β·dl/2). Returns (mm, data, dl)."""
    x, y = _plane_axes(mode)
    dl = DL_UM
    base = dict(size_um=(2.0, 1.6, 4.0), grid=ph.UniformGridSpec(dl_um=dl),
                run=ph.RunSpec(n_steps=10),
                background=ph.Background(permittivity=N_SIO2 ** 2),
                boundaries=ph.Boundaries(x="pml", y="pml", z="pml"))
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=2e13)
    sim = ph.Simulation(**base, sources=[ph.PointDipole(
        center_um=(1.0, 0.8, 0.5), polarization="Ey", source_time=pulse)])
    mm = mode_monitor(sim, mode, axis="z", position_um=2.0, freqs_hz=(F0,),
                      name="port", center_um=(0.0, 0.0))
    phi = 2.0 * np.pi * mode.n_eff / WL_UM * (0.5 * dl)
    md = modal_fields(mode, x, y, axis="z", direction="+")
    hf = a * np.exp(1j * phi) - b * np.exp(-1j * phi)
    comps = {"Ex": (a + b) * md["e1"], "Ey": (a + b) * md["e2"],
             "Hx": hf * md["h1"], "Hy": hf * md["h2"]}
    arr = np.stack([np.asarray(comps[c], complex) for c in
                    ("Ex", "Ey", "Hx", "Hy")], axis=0)[None]  # (f, component, y, x)
    da = xr.DataArray(arr, dims=("f", "component", "y", "x"),
                      coords={"f": [F0], "component": ["Ex", "Ey", "Hx", "Hy"],
                              "y": y, "x": x})
    return mm, {"port": da}, dl


def test_mode_monitor_captures_grid_dl():
    """mode_monitor() captures the propagation-axis grid spacing into dl_um, which
    is what enables the de-stagger by default."""
    mm, _, dl = _mm_and_staggered_data(_solver().solve(num_modes=1,
                                                        polarization="TE")[0])
    assert mm.dl_um == pytest.approx(dl)


def test_destagger_is_default_on_via_mode_monitor():
    """ModeMonitor.mode_power de-staggers BY DEFAULT (colocate=True): the default
    equals an explicit destagger_dl, differs from destagger_dl=None (proving it is
    applied), and is OFF when colocate=False (synthetic opt-out)."""
    m0 = _solver().solve(num_modes=1, polarization="TE")[0]
    mm, data, dl = _mm_and_staggered_data(m0)
    t_default = mm.mode_power(data)[F0]                      # auto -> de-stagger ON
    t_off = mm.mode_power(data, destagger_dl=None)[F0]       # forced off (colocate on)
    t_explicit = mm.mode_power(data, destagger_dl=dl)[F0]    # explicit spacing
    assert t_default == pytest.approx(t_explicit, rel=1e-9)  # default uses dl_um
    assert abs(t_default - t_off) > 1e-3 * t_off            # de-stagger really applied
    # colocate=False -> de-stagger auto-OFF (tied to colocate): equals the explicit
    # off at the SAME colocate setting (comparing to the colocate=True off would
    # differ by the transverse co-location, not the de-stagger).
    t_nocolo_auto = mm.mode_power(data, colocate=False)[F0]
    t_nocolo_off = mm.mode_power(data, colocate=False, destagger_dl=None)[F0]
    assert t_nocolo_auto == pytest.approx(t_nocolo_off, rel=1e-9)
