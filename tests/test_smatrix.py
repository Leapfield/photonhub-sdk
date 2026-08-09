"""Multiport S-matrix assembler — physics pins on SYNTHETIC recorded planes.

No FDTD run: each port's recorded DFT plane is built analytically from the frozen
FDE mode (via ``modal_fields``), shaped exactly like a real ``ModeMonitor`` DFT
slice (a single DataArray with dims ``('f','component','z','y','x')`` carrying the
four tangential components stacked on the ``component`` axis — what
``data[monitor.name]`` returns and what ``SPort._planes`` / ``ModeMonitor`` slice
with ``.sel(component=...)``).

We synthesize the field a known forward/backward wave puts on each plane and pin:

* a 2-port straight guide -> ``S21 ≈ e^{-i beta L}``, ``|S21| ≈ 1``, ``S11 ≈ 0``,
  ``|S21|^2 + |S11|^2 ≈ 1`` (energy);
* reciprocity ``S_ij = S_ji`` for a reciprocal device;
* passivity ``max eig(S† S) ≤ 1 + eps``;
* a symmetric 1->2 splitter (``|S21|=|S31|=1/sqrt(2)``), reciprocal & passive.

These pin the assembler's amplitude bookkeeping, directional incident/scattered
split, and normalization — not the scalar-H overlap physics (a downstream gate),
since the planes are built with the same reconstruction the overlap uses.
"""

import functools

import numpy as np
import pytest
import xarray as xr

from photonhub.plugins import Mode, ModeSolver
from photonhub.plugins.mode_devices import ModeMonitor
from photonhub.plugins.mode_overlap import modal_fields
from photonhub.plugins.smatrix import (
    SPort,
    assemble_smatrix,
    assert_passive,
    assert_reciprocal,
    is_passive,
    is_reciprocal,
    passivity_violation,
    reciprocity_error,
    smatrix as _smatrix,
)

# These tests feed SYNTHETIC, already-collocated analytic planes (E and H built at
# the SAME grid points), so the Yee co-location — correct only for the engine's
# STAGGERED DFT output — must be OFF here (otherwise it perturbs the exact
# zero/orthogonality relations the tests pin). Real runs default colocate=True.
smatrix = functools.partial(_smatrix, colocate=False)

# --- SOI strip @ 1310 nm (mirror test_mode_overlap) -------------------------
WL_UM = 1.31
DL_UM = 0.025
CORE_W_UM, CORE_H_UM = 0.45, 0.22
N_SI, N_SIO2 = 3.5, 1.444
C0 = 2.99792458e8
F0 = C0 / (WL_UM * 1e-6)
_TANGENTIAL_Z = ("Ex", "Ey", "Hx", "Hy")


@pytest.fixture(scope="module")
def te0() -> Mode:
    solver = ModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM,
        core_w_um=CORE_W_UM, core_h_um=CORE_H_UM,
        n_core=N_SI, n_clad=N_SIO2)
    return solver.solve(num_modes=1, polarization="TE")[0]


def _plane_axes(mode: Mode, *, pad_cells: int = 6, dl_um: float = DL_UM):
    ny, nx = mode.field.shape
    nX = nx + 2 * pad_cells
    nY = ny + 2 * pad_cells
    x = (np.arange(nX) - (nX - 1) / 2.0) * dl_um
    y = (np.arange(nY) - (nY - 1) / 2.0) * dl_um
    return x, y


def _stacked_plane(mode: Mode, x, y, waves, *, freqs=(F0,)):
    """Build one z-normal DFT-shaped DataArray (dims f/component/z/y/x) carrying
    the SUPERPOSITION of ``waves`` on the plane.

    ``waves`` is a list of ``(amplitude, direction)``: each contributes
    ``amplitude * (forward/backward mode fields)`` so the plane is the sum of an
    incident + a scattered wave (or just one). ``amplitude`` may be complex
    (carries the propagation phase). The four tangential components are stacked on
    the ``component`` axis in the order ``Ex, Ey, Hx, Hy`` and the same field is
    repeated for every frequency in ``freqs`` (single-freq tests pass one)."""
    comp_arrays = {c: np.zeros((y.size, x.size), dtype=np.complex128)
                   for c in _TANGENTIAL_Z}
    for amp, direction in waves:
        m = modal_fields(mode, x, y, axis="z", direction=direction,
                         center_um=(0.0, 0.0))
        comp_arrays["Ex"] += amp * m["e1"]
        comp_arrays["Ey"] += amp * m["e2"]
        comp_arrays["Hx"] += amp * m["h1"]
        comp_arrays["Hy"] += amp * m["h2"]

    # Stack (component, y, x) then broadcast over freqs -> (f, component, z, y, x).
    stack = np.stack([comp_arrays[c] for c in _TANGENTIAL_Z])  # [comp, y, x]
    data = np.broadcast_to(
        stack[None, :, None, :, :],
        (len(freqs), len(_TANGENTIAL_Z), 1, y.size, x.size),
    ).copy()
    return xr.DataArray(
        data,
        dims=("f", "component", "z", "y", "x"),
        coords={"f": list(freqs), "component": list(_TANGENTIAL_Z),
                "z": [0.0], "y": y, "x": x},
        name="port",
    )


def _monitor(mode: Mode, name: str, *, out_direction: str) -> ModeMonitor:
    """A bare ModeMonitor for a z-normal port (no Simulation needed — only its
    name/mode/axis are used by the overlap)."""
    from photonhub.components.monitors import FieldDftMonitor
    fm = FieldDftMonitor(
        name=name, center_um=(0.0, 0.0, 0.0), size_um=(2.0, 2.0, 0.0),
        fields=_TANGENTIAL_Z, freqs_hz=(F0,))
    return ModeMonitor(field_monitor=fm, mode=mode, axis="z",
                       center_um=(0.0, 0.0), direction=out_direction)


class _Data(dict):
    """Minimal SimulationData stand-in: ``data[name]`` -> the port's DataArray."""


# ---------------------------------------------------------------------------
# 2-port straight waveguide
# ---------------------------------------------------------------------------

def _two_port(mode: Mode):
    """Port 1 (out_direction '-', the input/west side) and port 2 (out_direction
    '+', the output/east side) of a straight guide along +z."""
    x, y = _plane_axes(mode)
    p1 = SPort("p1", _monitor(mode, "p1", out_direction="-"))
    p2 = SPort("p2", _monitor(mode, "p2", out_direction="+"))
    return x, y, p1, p2


def test_straight_waveguide_S21_unit_with_phase(te0: Mode):
    """Drive port 1: a lossless straight guide passes the mode to port 2 with the
    propagation phase. S21 ≈ e^{-i beta L}, |S21| ≈ 1, S11 ≈ 0."""
    x, y, p1, p2 = _two_port(te0)
    L_um = 4.0
    beta = 2 * np.pi * te0.n_eff / WL_UM            # rad/um
    phase = np.exp(-1j * beta * L_um)

    # Driven port 1: incident forward wave (amp 1, into device = +z), no reflection.
    plane1 = _stacked_plane(te0, x, y, [(1.0 + 0j, "+")])
    # Port 2: transmitted forward wave at amplitude e^{-i beta L} leaving via +z.
    plane2 = _stacked_plane(te0, x, y, [(phase, "+")])
    data = _Data(p1=plane1, p2=plane2)

    col = smatrix([p1, p2], "p1", data)
    S21 = col[("p2", "p1")][F0]
    S11 = col[("p1", "p1")][F0]

    assert abs(S21) == pytest.approx(1.0, rel=2e-3)
    assert S21 == pytest.approx(phase, rel=2e-3)
    assert abs(S11) == pytest.approx(0.0, abs=1e-6)
    # Energy: a lossless 2-port driven at 1 has |S21|^2 + |S11|^2 ≈ 1.
    assert abs(S21) ** 2 + abs(S11) ** 2 == pytest.approx(1.0, rel=2e-3)


def test_straight_waveguide_with_reflection_energy(te0: Mode):
    """Add a partial reflection at port 1: a Fresnel-like split with
    |S11|^2 + |S21|^2 ≈ 1 (energy conserved, lossless)."""
    x, y, p1, p2 = _two_port(te0)
    r = 0.3 + 0.1j                 # reflection coefficient at port 1
    t = np.sqrt(1.0 - abs(r) ** 2)  # lossless transmission magnitude
    # Driven port 1 plane = incident (into +z) + reflected (back out, -z).
    plane1 = _stacked_plane(te0, x, y, [(1.0 + 0j, "+"), (r, "-")])
    plane2 = _stacked_plane(te0, x, y, [(t + 0j, "+")])
    data = _Data(p1=plane1, p2=plane2)

    col = smatrix([p1, p2], "p1", data)
    S11 = col[("p1", "p1")][F0]
    S21 = col[("p2", "p1")][F0]
    assert S11 == pytest.approx(r, rel=2e-3)
    assert abs(S21) == pytest.approx(t, rel=2e-3)
    assert abs(S11) ** 2 + abs(S21) ** 2 == pytest.approx(1.0, rel=2e-3)


def test_full_two_port_matrix_reciprocal_passive(te0: Mode):
    """Assemble both columns (drive p1, then p2) of a symmetric, lossless
    straight guide and check the assembled 2x2 S is reciprocal + passive."""
    x, y, p1, p2 = _two_port(te0)
    beta = 2 * np.pi * te0.n_eff / WL_UM
    phase = np.exp(-1j * beta * 4.0)

    # Drive p1 -> light exits p2 with `phase`, nothing reflects.
    d1 = _Data(p1=_stacked_plane(te0, x, y, [(1.0 + 0j, "+")]),
               p2=_stacked_plane(te0, x, y, [(phase, "+")]))
    # Drive p2 (from the +z/east side, incident travels -z into the device) ->
    # light exits p1 with the SAME phase; reciprocal.
    d2 = _Data(p2=_stacked_plane(te0, x, y, [(1.0 + 0j, "-")]),
               p1=_stacked_plane(te0, x, y, [(phase, "-")]))

    col1 = smatrix([p1, p2], "p1", d1)
    col2 = smatrix([p1, p2], "p2", d2)
    S = assemble_smatrix([col1, col2])

    assert set(S.coords["port_out"].values) == {"p1", "p2"}
    assert S.sel(port_out="p2", port_in="p1").item() == pytest.approx(phase, rel=2e-3)
    assert S.sel(port_out="p1", port_in="p2").item() == pytest.approx(phase, rel=2e-3)
    assert reciprocity_error(S) < 5e-3
    assert is_reciprocal(S, atol=5e-3)
    assert passivity_violation(S) < 5e-3
    assert is_passive(S, atol=5e-3)
    assert_reciprocal(S, atol=5e-3)
    assert_passive(S, atol=5e-3)


# ---------------------------------------------------------------------------
# Unequal-mode ports (taper-like): power normalization + reciprocity
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def te0_wide() -> Mode:
    """TE0 of a wider (0.80 µm) strip — a DIFFERENT mode (different n_eff and
    P_mode) than ``te0``, for the unequal-port normalization regression."""
    solver = ModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM,
        core_w_um=0.80, core_h_um=CORE_H_UM,
        n_core=N_SI, n_clad=N_SIO2)
    return solver.solve(num_modes=1, polarization="TE")[0]


def _mode_plane_power(mode: Mode, x, y) -> float:
    """P_mode of ``mode`` on the (x, y) plane — the power its unit-amplitude
    field carries (via the same overlap kernel the assembler uses)."""
    from photonhub.plugins.mode_overlap import _overlap_terms
    plane = _stacked_plane(mode, x, y, [(1.0 + 0j, "+")])
    planes = {c: plane.sel(component=c) for c in _TANGENTIAL_Z}
    ((_, p_mode),) = [v for v in _overlap_terms(
        planes, mode, axis="z", direction="+", center_um=(0.0, 0.0),
        colocate=False).values()]
    return float(p_mode)


def test_unequal_mode_ports_power_ratio_and_reciprocity(te0: Mode, te0_wide: Mode):
    """Regression: with ports carrying DIFFERENT modes (a w1→w2 taper), |S21|^2
    must be the TRUE power ratio and S must stay reciprocal.

    A lossless full-power transition is synthesized: the output plane carries
    the wide mode at the amplitude that makes its modal power EQUAL the input
    power. The pre-fix ``c = a_pm/P_mode`` normalization biased |S21|^2 by
    P_mode1/P_mode2 (≈ n_eff1/n_eff2) and gave S21/S12 = P1/P2 ≠ 1."""
    # A common plane covering both modes.
    x, y = _plane_axes(te0_wide)
    p1_mon = _monitor(te0, "p1", out_direction="-")
    p2_mon = _monitor(te0_wide, "p2", out_direction="+")
    p1 = SPort("p1", p1_mon)
    p2 = SPort("p2", p2_mon)

    P1 = _mode_plane_power(te0, x, y)
    P2 = _mode_plane_power(te0_wide, x, y)
    assert abs(P1 - P2) / P1 > 0.02, "modes chosen so P_mode really differs"

    # Lossless, reflectionless full-power transfer: |amp_out|^2 * P2 == 1^2 * P1.
    amp_out = np.sqrt(P1 / P2)
    d1 = _Data(p1=_stacked_plane(te0, x, y, [(1.0 + 0j, "+")]),
               p2=_stacked_plane(te0_wide, x, y, [(amp_out + 0j, "+")]))
    col1 = smatrix([p1, p2], "p1", d1)
    S21 = col1[("p2", "p1")][F0]
    assert abs(S21) ** 2 == pytest.approx(1.0, rel=2e-3)   # true power ratio
    assert abs(col1[("p1", "p1")][F0]) == pytest.approx(0.0, abs=1e-6)

    # Reverse drive (wide side, incident travelling -z): reciprocity S12 == S21.
    amp_back = np.sqrt(P2 / P1)
    d2 = _Data(p2=_stacked_plane(te0_wide, x, y, [(1.0 + 0j, "-")]),
               p1=_stacked_plane(te0, x, y, [(amp_back + 0j, "-")]))
    col2 = smatrix([p1, p2], "p2", d2)
    S = assemble_smatrix([col1, col2])
    assert reciprocity_error(S) < 5e-3
    assert passivity_violation(S) < 5e-3


# ---------------------------------------------------------------------------
# Multi-frequency
# ---------------------------------------------------------------------------

def test_multifrequency_column(te0: Mode):
    """A two-frequency run returns S per frequency; the transmitted amplitude at
    f1 is half -> |S21(f1)| = 0.5."""
    x, y, p1, p2 = _two_port(te0)
    f0, f1 = F0, F0 * 1.02
    # Build per-freq planes: incident unit at both freqs; transmit 1.0 @ f0,
    # 0.5 @ f1. Stack two single-freq planes by hand.
    inc = _stacked_plane(te0, x, y, [(1.0 + 0j, "+")], freqs=(f0, f1))
    tr0 = _stacked_plane(te0, x, y, [(1.0 + 0j, "+")], freqs=(f0,))
    tr1 = _stacked_plane(te0, x, y, [(0.5 + 0j, "+")], freqs=(f1,))
    tr = xr.concat([tr0, tr1], dim="f").assign_coords(f=[f0, f1])
    data = _Data(p1=inc, p2=tr)

    col = smatrix([p1, p2], "p1", data)
    s21 = col[("p2", "p1")]
    assert abs(s21[f0]) == pytest.approx(1.0, rel=2e-3)
    assert abs(s21[f1]) == pytest.approx(0.5, rel=2e-3)


# ---------------------------------------------------------------------------
# Symmetric 1->2 splitter (3-port)
# ---------------------------------------------------------------------------

def test_symmetric_splitter_reciprocal_passive(te0: Mode):
    """A lossless symmetric 1->2 splitter: drive p1 -> half the power to each of
    p2/p3 (|S21|=|S31|=1/sqrt(2)), no reflection. Assemble the full reciprocal
    3x3 (using the analytic ideal splitter S for the other two columns) and check
    reciprocity + passivity."""
    x, y = _plane_axes(te0)
    p1 = SPort("p1", _monitor(te0, "p1", out_direction="-"))  # input, west
    p2 = SPort("p2", _monitor(te0, "p2", out_direction="+"))  # output arm A
    p3 = SPort("p3", _monitor(te0, "p3", out_direction="+"))  # output arm B
    ports = [p1, p2, p3]
    s = 1.0 / np.sqrt(2.0)

    # Drive p1: incident into +z at p1, half-amplitude transmitted out of p2/p3.
    data1 = _Data(
        p1=_stacked_plane(te0, x, y, [(1.0 + 0j, "+")]),  # incident only, no refl
        p2=_stacked_plane(te0, x, y, [(s + 0j, "+")]),
        p3=_stacked_plane(te0, x, y, [(s + 0j, "+")]),
    )
    col1 = smatrix(ports, "p1", data1)
    assert abs(col1[("p1", "p1")][F0]) == pytest.approx(0.0, abs=1e-6)  # S11
    assert abs(col1[("p2", "p1")][F0]) == pytest.approx(s, rel=2e-3)    # S21
    assert abs(col1[("p3", "p1")][F0]) == pytest.approx(s, rel=2e-3)    # S31

    # For the full matrix, supply the reciprocal columns for driving p2 and p3.
    # Ideal lossless 3-port splitter with the input split equally and the two
    # outputs isolated/matched: S = [[0, s, s],[s, 0, 0],[s, 0, 0]] (this is the
    # standard non-resonant power splitter; it is reciprocal and passive though
    # not unitary — half the back-driven power radiates, |col|^2 = 0.5 ≤ 1).
    data2 = _Data(
        p2=_stacked_plane(te0, x, y, [(1.0 + 0j, "-")]),  # drive p2 from east
        p1=_stacked_plane(te0, x, y, [(s + 0j, "-")]),    # exits p1 (west)
        p3=_stacked_plane(te0, x, y, []),                 # isolated
    )
    data3 = _Data(
        p3=_stacked_plane(te0, x, y, [(1.0 + 0j, "-")]),
        p1=_stacked_plane(te0, x, y, [(s + 0j, "-")]),
        p2=_stacked_plane(te0, x, y, []),
    )
    col2 = smatrix(ports, "p2", data2)
    col3 = smatrix(ports, "p3", data3)
    S = assemble_smatrix([col1, col2, col3], port_order=["p1", "p2", "p3"])

    # Reciprocity: S21 == S12, S31 == S13.
    assert reciprocity_error(S) < 5e-3
    assert_reciprocal(S, atol=5e-3)
    # Passivity: the splitter does not amplify (largest singular value ≤ 1).
    assert passivity_violation(S) < 5e-3
    assert_passive(S, atol=5e-3)


# ---------------------------------------------------------------------------
# Helper-level checks: reciprocity & passivity flag real violations
# ---------------------------------------------------------------------------

def test_reciprocity_helper_detects_asymmetry():
    S = xr.DataArray(
        np.array([[[0.0], [0.9]], [[0.1], [0.0]]], dtype=np.complex128),
        dims=("port_out", "port_in", "f"),
        coords={"port_out": ["a", "b"], "port_in": ["a", "b"], "f": [F0]})
    assert reciprocity_error(S) == pytest.approx(0.8)
    assert not is_reciprocal(S, atol=1e-3)
    with pytest.raises(AssertionError):
        assert_reciprocal(S, atol=1e-3)


def test_passivity_helper_detects_gain():
    # An amplifying 1-port: |S11| = 1.5 -> eig(S† S) = 2.25 > 1.
    S = xr.DataArray(
        np.array([[[1.5 + 0j]]]), dims=("port_out", "port_in", "f"),
        coords={"port_out": ["a"], "port_in": ["a"], "f": [F0]})
    assert passivity_violation(S) == pytest.approx(1.25, rel=1e-9)
    assert not is_passive(S, atol=1e-3)
    with pytest.raises(AssertionError):
        assert_passive(S, atol=1e-3)


def test_passivity_unitary_matrix_is_passive():
    # A unitary 2x2 (lossless reciprocal) has eig(S† S) == 1 exactly.
    theta = 0.7
    U = np.array([[np.cos(theta), 1j * np.sin(theta)],
                  [1j * np.sin(theta), np.cos(theta)]], dtype=np.complex128)
    S = xr.DataArray(U[:, :, None], dims=("port_out", "port_in", "f"),
                     coords={"port_out": ["a", "b"], "port_in": ["a", "b"],
                             "f": [F0]})
    assert passivity_violation(S) == pytest.approx(0.0, abs=1e-12)
    assert is_passive(S)
    assert reciprocity_error(S) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# API / validation
# ---------------------------------------------------------------------------

def test_smatrix_ndarray_input_to_helpers():
    """The reciprocity/passivity helpers also accept a bare (n,n) or (n_f,n,n)
    ndarray, not just the xarray."""
    U = np.eye(2, dtype=np.complex128)
    assert reciprocity_error(U) == pytest.approx(0.0)
    assert passivity_violation(U) == pytest.approx(0.0, abs=1e-12)
    assert passivity_violation(U[None, :, :]) == pytest.approx(0.0, abs=1e-12)


def test_duplicate_port_names_raise(te0: Mode):
    x, y = _plane_axes(te0)
    p = SPort("p1", _monitor(te0, "p1", out_direction="-"))
    data = _Data(p1=_stacked_plane(te0, x, y, [(1.0 + 0j, "+")]))
    with pytest.raises(ValueError, match="duplicate"):
        smatrix([p, p], "p1", data)


def test_unknown_driven_port_raises(te0: Mode):
    x, y, p1, p2 = _two_port(te0)
    data = _Data(p1=_stacked_plane(te0, x, y, [(1.0 + 0j, "+")]),
                 p2=_stacked_plane(te0, x, y, [(1.0 + 0j, "+")]))
    with pytest.raises(ValueError, match="not among ports"):
        smatrix([p1, p2], "pX", data)


def test_zero_incident_amplitude_raises(te0: Mode):
    """Driving a port whose recorded plane has no incoming wave -> a_j == 0 ->
    cannot normalize."""
    x, y, p1, p2 = _two_port(te0)
    # p1 plane carries ONLY an outgoing (-z) wave: its incoming (+z) projection ~0.
    data = _Data(p1=_stacked_plane(te0, x, y, [(1.0 + 0j, "-")]),
                 p2=_stacked_plane(te0, x, y, [(1.0 + 0j, "+")]))
    with pytest.raises(ValueError, match="incident amplitude"):
        smatrix([p1, p2], "p1", data)


def test_out_direction_defaults_to_monitor_direction(te0: Mode):
    mon = _monitor(te0, "p", out_direction="+")
    port = SPort("p", mon)
    assert port.out_direction == "+"
    assert port.in_direction == "-"


def test_bad_out_direction_raises(te0: Mode):
    mon = _monitor(te0, "p", out_direction="+")
    with pytest.raises(ValueError, match="out_direction"):
        SPort("p", mon, out_direction="?")


def test_sport_amplitude_uses_folded_quadrature_like_mode_power(te0: Mode):
    """F09 regression: on a simulation with a §20 symmetry plane, SPort._amplitude
    must apply the folded-domain quadrature (fold_low) EXACTLY as
    ModeMonitor.mode_power / mode_decomposition do — half-weighting the sample row
    that sits ON the symmetry plane. Here the monitor's t2 (y) axis is folded and
    the plane is a half-domain ladder starting at y=0, so the on-plane row's weight
    matters. |SPort.outgoing|² must equal ModeMonitor.mode_power; before the fix
    SPort ignored fold_low and the two readouts diverged by the on-plane row's
    (fold-antinode) power."""
    from photonhub.components.monitors import FieldDftMonitor

    class _SymSim:
        symmetry = (0, 1, 0)          # y (= t2 for a z-normal plane) symmetry fold

    ny, nx = te0.field.shape
    x = (np.arange(nx + 12) - (nx + 11) / 2.0) * DL_UM
    y = np.arange(ny // 2 + 8) * DL_UM        # HALF-domain: first sample ON the plane
    fm = FieldDftMonitor(name="p", center_um=(0.0, 0.0, 0.0),
                         size_um=(2.0, 2.0, 0.0), fields=_TANGENTIAL_Z, freqs_hz=(F0,))
    mon = ModeMonitor(field_monitor=fm, mode=te0, axis="z", center_um=(0.0, 0.0),
                      direction="+", simulation=_SymSim(), per_freq_modes=False)
    assert mon._fold_low() == (False, True)   # t2 (y) is folded

    plane = _stacked_plane(te0, x, y, [(1.0, "+")])
    data = _Data({"p": plane})
    b = SPort("p", mon).outgoing(data, colocate=False)[F0]
    p_power = mon.mode_power(data, direction="+", colocate=False)[F0]
    assert abs(b) ** 2 == pytest.approx(p_power, rel=1e-9)
