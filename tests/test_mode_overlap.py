"""Directional-power mode-overlap (mode-monitor transmission) — physics pins.

Validates ``photonhub.plugins.mode_overlap.mode_transmission`` on SYNTHETIC
field planes (no FDTD run): a plane that *is* the mode -> forward T==1, the
opposite-direction plane -> forward T==0 / backward T==1, an orthogonal profile
-> T<1, amplitude scaling -> T scales by |c|^2, and a graded (non-uniform)
transverse mesh still gives T==1. Mode profiles come from the frozen FDE
``ModeSolver.from_rectangular_core`` (a small SOI strip, mirroring the
benchmark) and a hand-built gaussian wrapped as a ``Mode``.

The plane fields are assembled with the SAME scalar-limit reconstruction the
module uses (E = e_mode, H = +/- h_mode for a forward/backward wave), so these
tests pin the *overlap algebra and normalization*, not the scalar-H physics
approximation itself (that is a downstream Tier-2b gate).
"""

import numpy as np
import functools

import pytest
import xarray as xr

from photonhub.plugins import Mode, ModeSolver
from photonhub.plugins.mode_overlap import (
    ETA0,
    _cell_widths,
    _colocate_to_node,
    mode_transmission as _mode_transmission,
    modal_fields,
    resample_profile,
    vector_modal_fields,
)
from photonhub.plugins.vector_modes import VectorMode

# These tests feed SYNTHETIC, already-collocated analytic fields (E and H built
# at the SAME grid points), so the Yee co-location — which averages adjacent
# cells and is correct only for the engine's STAGGERED DFT output — does not
# apply; default it OFF here. The co-location path is covered by
# test_colocate_to_node_* below and the straight-guide floor benchmark.
mode_transmission = functools.partial(_mode_transmission, colocate=False)

# --- SOI strip @ 1310 nm (mirror the benchmark / test_modesolver) -----------
WL_UM = 1.31
DL_UM = 0.025
CORE_W_UM, CORE_H_UM = 0.45, 0.22
N_SI, N_SIO2 = 3.5, 1.444
F0 = 2.99792458e8 / (WL_UM * 1e-6)


@pytest.fixture(scope="module")
def te0() -> Mode:
    solver = ModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM,
        core_w_um=CORE_W_UM, core_h_um=CORE_H_UM,
        n_core=N_SI, n_clad=N_SIO2)
    return solver.solve(num_modes=1, polarization="TE")[0]


@pytest.fixture(scope="module")
def te0_wide() -> Mode:
    """TE0 of a wider (0.80 µm) strip — a DIFFERENT P_mode than ``te0``, for the
    unequal-width power-transmission regression below."""
    solver = ModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM,
        core_w_um=0.80, core_h_um=CORE_H_UM,
        n_core=N_SI, n_clad=N_SIO2)
    return solver.solve(num_modes=1, polarization="TE")[0]


def _plane_axes(mode: Mode, *, pad_cells: int = 6, dl_um: float = DL_UM):
    """A z-normal monitor plane's transverse (x, y) coordinates (microns,
    centered on the mode), a bit wider than the mode window."""
    ny, nx = mode.field.shape
    nX = nx + 2 * pad_cells
    nY = ny + 2 * pad_cells
    x = (np.arange(nX) - (nX - 1) / 2.0) * dl_um
    y = (np.arange(nY) - (nY - 1) / 2.0) * dl_um
    return x, y


def _build_plane(mode: Mode, x, y, *, direction="+", n_eff=None, scale=1.0,
                 freq_hz=F0, as_dft=False, profile_override=None):
    """Assemble the four tangential plane DataArrays (Ex, Ey, Hx, Hy) for a
    z-propagating wave that is `scale` times the mode travelling in `direction`.

    If `as_dft`, each component is shaped like a real single-plane field_dft
    slice with dims ('f','component','z','y','x'); otherwise a plain 2-D
    (y, x) DataArray. `profile_override` swaps the transverse E shape (for the
    orthogonality test) while keeping the same H reconstruction."""
    m = modal_fields(mode, x, y, axis="z", direction=direction, n_eff=n_eff,
                     center_um=(0.0, 0.0))
    e1, e2, h1, h2 = m["e1"], m["e2"], m["h1"], m["h2"]
    if profile_override is not None:
        # Replace the major-E shape (and rebuild H from it) with a custom one.
        neff = float(mode.n_eff if n_eff is None else n_eff)
        sgn = 1.0 if direction == "+" else -1.0
        e1 = profile_override
        e2 = np.zeros_like(e1)
        h1 = np.zeros_like(e1)
        h2 = sgn * (neff / ETA0) * e1
    comps = {"Ex": scale * e1, "Ey": scale * e2,
             "Hx": scale * h1, "Hy": scale * h2}

    out = {}
    for name, arr in comps.items():
        arr = np.asarray(arr, dtype=np.complex128)
        if as_dft:
            # (f, component, z, y, x) with singleton f/component/z.
            da = xr.DataArray(
                arr[None, None, None, :, :],
                dims=("f", "component", "z", "y", "x"),
                coords={"f": [freq_hz], "component": [name], "z": [0.0],
                        "y": y, "x": x})
        else:
            da = xr.DataArray(arr, dims=("y", "x"), coords={"y": y, "x": x})
        out[name] = da
    return out


# --- longitudinal Yee de-stagger (Fix B) ------------------------------------

def _staggered_incident_reflected(mode, x, y, *, a, b, phi, freq_hz=F0):
    """A plane carrying incident `a` + reflected `b` of `mode`, with the
    engine's LONGITUDINAL Yee stagger baked in: H is half a cell along the
    normal from E, so the forward part of H carries e^{+i phi} and the backward
    part e^{-i phi} relative to E (phi = beta*dl/2). E = (a+b)e ;
    H = (a e^{i phi} - b e^{-i phi}) h. DFT-shaped (carries `freq_hz`) so the
    de-stagger can recover phi from lambda = C0/f. Returns the four tangential
    DataArrays."""
    m = modal_fields(mode, x, y, axis="z", direction="+")
    e1, e2, h1, h2 = m["e1"], m["e2"], m["h1"], m["h2"]
    hfac = a * np.exp(1j * phi) - b * np.exp(-1j * phi)
    comps = {"Ex": (a + b) * e1, "Ey": (a + b) * e2,
             "Hx": hfac * h1, "Hy": hfac * h2}
    out = {}
    for n, v in comps.items():
        arr = np.asarray(v, dtype=np.complex128)[None, None, None, :, :]
        out[n] = xr.DataArray(arr, dims=("f", "component", "z", "y", "x"),
                              coords={"f": [freq_hz], "component": [n],
                                      "z": [0.0], "y": y, "x": x})
    return out


def test_destagger_removes_reflection_leak(te0: Mode):
    """The de-stagger recovers the incident amplitude independent of the
    reflection's PHASE (the standing-wave ripple), while the uncorrected reading
    leaks: T0 swings with the reflection phase, T_destagger stays = |a|^2."""
    x, y = _plane_axes(te0)
    dl = DL_UM
    phi = 2.0 * np.pi * te0.n_eff / WL_UM * (0.5 * dl)
    assert phi > 0.05  # a meaningful stagger at this dl
    a, b = 1.0, 0.3
    T0, T1 = [], []
    for psi in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        plane = _staggered_incident_reflected(
            te0, x, y, a=a, b=b * np.exp(1j * psi), phi=phi)
        T0.append(mode_transmission(plane, te0, axis="z", direction="+")[F0])
        T1.append(mode_transmission(plane, te0, axis="z", direction="+",
                                    destagger_dl=dl)[F0])
    T0 = np.array(T0); T1 = np.array(T1)
    # de-staggered: clean incident |a|^2 == 1 at every reflection phase (no ripple)
    assert np.allclose(T1, a ** 2, rtol=2e-3)
    assert (T1.max() - T1.min()) < 1e-3
    # uncorrected: a real, phase-dependent leak (ripple) the fix removes
    assert (T0.max() - T0.min()) > 5e-3


def test_destagger_clean_forward_is_unity(te0: Mode):
    """With no reflection, the de-stagger still reads a clean forward wave as
    T == 1 (it does not bias the self-overlap)."""
    x, y = _plane_axes(te0)
    dl = DL_UM
    phi = 2.0 * np.pi * te0.n_eff / WL_UM * (0.5 * dl)
    plane = _staggered_incident_reflected(te0, x, y, a=1.0, b=0.0, phi=phi)
    T = mode_transmission(plane, te0, axis="z", direction="+", destagger_dl=dl)[F0]
    assert T == pytest.approx(1.0, rel=2e-3)


def test_destagger_near_nyquist_raises(te0: Mode):
    """The de-stagger's 2x2 solve divides by 2*cos(phi); at dl near
    lambda/(2 n_eff) (phi -> pi/2) it is singular and must refuse loudly
    instead of amplifying noise."""
    x, y = _plane_axes(te0)
    dl_nyq = WL_UM / (2.0 * te0.n_eff)   # phi == pi/2 exactly
    plane = _build_plane(te0, x, y, direction="+", as_dft=True)
    with pytest.raises(ValueError, match="singular"):
        mode_transmission(plane, te0, axis="z", direction="+",
                          destagger_dl=dl_nyq)


def test_destagger_frequencyless_plane_warns(te0: Mode):
    """An explicitly requested de-stagger on a plane with NO frequency axis
    (beta cannot be formed) is skipped — but says so instead of silently
    ignoring the argument."""
    x, y = _plane_axes(te0)
    plane = _build_plane(te0, x, y, direction="+")   # plain 2-D, no f dim
    with pytest.warns(UserWarning, match="SKIPPED"):
        T = mode_transmission(plane, te0, axis="z", direction="+",
                              destagger_dl=DL_UM)
    assert next(iter(T.values())) == pytest.approx(1.0, rel=1e-3)


# --- self-overlap = 1 -------------------------------------------------------

def test_self_overlap_is_one(te0: Mode):
    """A plane that IS the forward mode reads forward T ~= 1 (tight)."""
    x, y = _plane_axes(te0)
    plane = _build_plane(te0, x, y, direction="+")
    T = mode_transmission(plane, te0, axis="z", direction="+")
    # A plain 2-D plane carries no frequency axis -> the single key is 0.0.
    assert list(T.keys()) == [0.0]
    (Tval,) = T.values()
    assert Tval == pytest.approx(1.0, rel=1e-3)


def test_self_overlap_dft_shaped_input(te0: Mode):
    """The same self-overlap when the plane is shaped like a real field_dft
    monitor slice (dims f/component/z/y/x) — exercises the dim-reduction path."""
    x, y = _plane_axes(te0)
    plane = _build_plane(te0, x, y, direction="+", as_dft=True)
    T = mode_transmission(plane, te0, axis="z", direction="+")
    (Tval,) = T.values()
    assert Tval == pytest.approx(1.0, rel=1e-3)


# --- power transmission across UNEQUAL-WIDTH ports (taper bug regression) ----

def test_power_transmission_correct_across_unequal_width_ports(te0, te0_wide):
    """Regression for the w1→w2 taper bug: a lossless transition between two
    DIFFERENT-width modes must read T≈1 *by power*. ``mode_transmission(power=True)``
    returns actual modal power ``|a_pm|²/P_mode``, so the ``P_out/P_in`` ratio is
    correct even when the two ports' ``P_mode`` differ; the bare ``|c|²``
    (``power=False``) ratio drops the per-mode power scale and is wrong there.

    The historical bug: ``ModeMonitor.mode_power`` returned ``|c|² = |a_pm|²/P_mode²``,
    so the acceptance taper (0.45→0.80 µm) under-reported by ``P_mode(w1)/P_mode(w2)``
    (~0.91 instead of ~1.0)."""
    x1, y1 = _plane_axes(te0)
    x2, y2 = _plane_axes(te0_wide)

    def tx(plane, mode, **kw):
        return next(iter(mode_transmission(
            plane, mode, axis="z", direction="+", **kw).values()))

    # P_mode of each port = power of its unit-amplitude mode (scale=1, power=True).
    p1 = tx(_build_plane(te0, x1, y1), te0, power=True)
    p2 = tx(_build_plane(te0_wide, x2, y2), te0_wide, power=True)
    assert p1 > 0 and p2 > 0
    assert abs(p1 - p2) / p1 > 0.05, "widths chosen so P_mode really differs"

    # A LOSSLESS transition: input (w1, amp 1) and output (w2) carrying EQUAL power.
    a_out = (p1 / p2) ** 0.5
    p_in = _build_plane(te0, x1, y1, scale=1.0)
    p_out = _build_plane(te0_wide, x2, y2, scale=a_out)

    T_power = tx(p_out, te0_wide, power=True) / tx(p_in, te0, power=True)
    T_legacy = tx(p_out, te0_wide) / tx(p_in, te0)         # bare |c|² ratio
    assert T_power == pytest.approx(1.0, rel=1e-3)          # FIXED: power conserved
    assert abs(T_legacy - 1.0) > 0.05                       # legacy ratio is wrong here


# --- directionality ---------------------------------------------------------

def test_backward_wave_forward_T_zero_backward_T_one(te0: Mode):
    """A purely BACKWARD mode: forward T ~= 0, backward T ~= 1."""
    x, y = _plane_axes(te0)
    plane = _build_plane(te0, x, y, direction="-")
    Tf = next(iter(mode_transmission(plane, te0, axis="z",
                                     direction="+").values()))
    Tb = next(iter(mode_transmission(plane, te0, axis="z",
                                     direction="-").values()))
    assert Tf == pytest.approx(0.0, abs=1e-9)
    assert Tb == pytest.approx(1.0, rel=1e-3)


def test_forward_wave_backward_T_zero(te0: Mode):
    """Conversely, a forward mode has ~0 backward T."""
    x, y = _plane_axes(te0)
    plane = _build_plane(te0, x, y, direction="+")
    Tb = next(iter(mode_transmission(plane, te0, axis="z",
                                     direction="-").values()))
    assert Tb == pytest.approx(0.0, abs=1e-9)


def test_backward_power_is_nonnegative(te0: Mode):
    """Regression: ``power=True`` with ``direction="-"`` is a POWER (>= 0), not
    the signed flux. A clean backward wave used to read a NEGATIVE backward
    "power" (division by the signed P_mode < 0), which made every reflection
    ratio negative and the ``r < 0.1`` reflection assertions vacuous."""
    x, y = _plane_axes(te0)
    plane_b = _build_plane(te0, x, y, direction="-")
    p_bwd = next(iter(mode_transmission(plane_b, te0, axis="z", direction="-",
                                        power=True).values()))
    plane_f = _build_plane(te0, x, y, direction="+")
    p_fwd = next(iter(mode_transmission(plane_f, te0, axis="z", direction="+",
                                        power=True).values()))
    assert p_bwd > 0.0
    # Same mode, same unit amplitude: backward power == forward power.
    assert p_bwd == pytest.approx(p_fwd, rel=1e-9)


# --- orthogonality ----------------------------------------------------------

def test_orthogonal_profile_T_below_one(te0: Mode):
    """A transversely-shifted / different profile gives forward T clearly < 1."""
    x, y = _plane_axes(te0)
    # A profile shifted off the mode center is partially orthogonal to it.
    base = modal_fields(te0, x, y, axis="z", direction="+",
                        center_um=(0.0, 0.0))
    shifted = modal_fields(te0, x, y, axis="z", direction="+",
                           center_um=(0.25, 0.0))  # 250 nm lateral shift
    plane = _build_plane(te0, x, y, direction="+",
                         profile_override=shifted["e1"])
    T = next(iter(mode_transmission(plane, te0, axis="z",
                                    direction="+").values()))
    assert T < 0.85
    # The overlap with itself (unshifted) is still 1, as a control.
    plane0 = _build_plane(te0, x, y, direction="+",
                          profile_override=base["e1"])
    T0 = next(iter(mode_transmission(plane0, te0, axis="z",
                                     direction="+").values()))
    assert T0 == pytest.approx(1.0, rel=1e-3)
    assert T < T0


def test_higher_order_like_profile_is_nearly_orthogonal(te0: Mode):
    """An odd (sign-flipped half) profile is ~orthogonal to the even TE0 -> T~0."""
    x, y = _plane_axes(te0)
    base = modal_fields(te0, x, y, axis="z", direction="+",
                        center_um=(0.0, 0.0))["e1"]
    odd = base * np.sign(x)[None, :]   # antisymmetric in x
    plane = _build_plane(te0, x, y, direction="+", profile_override=odd)
    T = next(iter(mode_transmission(plane, te0, axis="z",
                                    direction="+").values()))
    assert T == pytest.approx(0.0, abs=1e-6)


# --- power normalization ----------------------------------------------------

@pytest.mark.parametrize("c", [0.5, 2.0, 3.0])
def test_amplitude_scaling_is_power_ratio(te0: Mode, c: float):
    """Scaling the simulated field by c scales T by |c|^2 (power ratio)."""
    x, y = _plane_axes(te0)
    plane = _build_plane(te0, x, y, direction="+", scale=c)
    T = next(iter(mode_transmission(plane, te0, axis="z",
                                    direction="+").values()))
    assert T == pytest.approx(c ** 2, rel=1e-3)


def test_complex_amplitude_scaling(te0: Mode):
    """A complex amplitude c (phase + magnitude) scales T by |c|^2."""
    x, y = _plane_axes(te0)
    c = 0.7 * np.exp(1j * 1.1)
    plane = _build_plane(te0, x, y, direction="+", scale=c)
    T = next(iter(mode_transmission(plane, te0, axis="z",
                                    direction="+").values()))
    assert T == pytest.approx(abs(c) ** 2, rel=1e-3)


# --- graded (non-uniform) transverse mesh -----------------------------------

def _stretched_coords(n, dl0):
    """Symmetric, monotonically-coarsening centered coords (geometric ramp from
    the center outward) — a non-uniform transverse mesh."""
    half = [0.0]
    step = dl0
    for _ in range(n // 2):
        half.append(half[-1] + step)
        step *= 1.06
    half = np.array(half)
    c = np.concatenate([-half[::-1], half[1:]])
    return c


def test_graded_plane_self_overlap_is_one(te0: Mode):
    """Self-overlap on a stretched (non-uniform) transverse grid still T ~= 1:
    the area element is taken from the real coord spacings."""
    nx = te0.field.shape[1] + 24
    ny = te0.field.shape[0] + 24
    x = _stretched_coords(nx, DL_UM * 0.6)
    y = _stretched_coords(ny, DL_UM * 0.6)
    plane = _build_plane(te0, x, y, direction="+")
    T = next(iter(mode_transmission(plane, te0, axis="z",
                                    direction="+").values()))
    assert T == pytest.approx(1.0, rel=5e-3)


# --- multi-frequency + per-frequency n_eff ----------------------------------

def test_multifrequency_plane_returns_T_per_freq(te0: Mode):
    """A plane carrying two frequencies returns a T for each."""
    x, y = _plane_axes(te0)
    f0, f1 = F0, F0 * 1.02
    m = modal_fields(te0, x, y, axis="z", direction="+", center_um=(0.0, 0.0))
    comps = {"Ex": m["e1"], "Ey": m["e2"], "Hx": m["h1"], "Hy": m["h2"]}
    plane = {}
    for name, arr in comps.items():
        arr = np.asarray(arr, dtype=np.complex128)
        stacked = np.stack([arr, 0.5 * arr])  # f1 at half amplitude
        plane[name] = xr.DataArray(
            stacked[:, None, None, :, :],
            dims=("f", "component", "z", "y", "x"),
            coords={"f": [f0, f1], "component": [name], "z": [0.0],
                    "y": y, "x": x})
    T = mode_transmission(plane, te0, axis="z", direction="+")
    keys = sorted(T.keys())
    assert keys[0] == pytest.approx(f0)
    assert keys[1] == pytest.approx(f1)
    assert T[keys[0]] == pytest.approx(1.0, rel=1e-3)
    assert T[keys[1]] == pytest.approx(0.25, rel=1e-3)  # |0.5|^2


# --- non-z propagation axis (x-normal plane) --------------------------------

def test_x_normal_plane_self_overlap(te0: Mode):
    """The overlap is axis-agnostic: an x-propagating plane (transverse y, z)
    fed its own mode still reads T ~= 1. Components: Ey,Ez,Hy,Hz."""
    # Reuse the TE0 profile as a generic transverse shape on a (y, z) plane.
    yc = (np.arange(te0.field.shape[1] + 8) -
          (te0.field.shape[1] + 7) / 2.0) * DL_UM
    zc = (np.arange(te0.field.shape[0] + 8) -
          (te0.field.shape[0] + 7) / 2.0) * DL_UM
    m = modal_fields(te0, yc, zc, axis="x", direction="+", center_um=(0.0, 0.0))
    # axis="x" -> transverse (t1,t2)=(y,z); components Ey,Ez,Hy,Hz.
    plane = {
        "Ey": xr.DataArray(m["e1"], dims=("z", "y"), coords={"z": zc, "y": yc}),
        "Ez": xr.DataArray(m["e2"], dims=("z", "y"), coords={"z": zc, "y": yc}),
        "Hy": xr.DataArray(m["h1"], dims=("z", "y"), coords={"z": zc, "y": yc}),
        "Hz": xr.DataArray(m["h2"], dims=("z", "y"), coords={"z": zc, "y": yc}),
    }
    T = next(iter(mode_transmission(plane, te0, axis="x",
                                    direction="+").values()))
    assert T == pytest.approx(1.0, rel=1e-3)


def test_y_normal_plane_orientation_needs_thickness_axis(te0: Mode):
    """A y-propagating plane (transverse (t1,t2)=(z,x)) of a z-normal slab: the
    strip thickness lies on the FIRST transverse axis, so the overlap mode must
    be oriented with ``thickness_axis="z"`` — then a plane that IS the mode reads
    T~=1. The legacy default (thickness on the 2nd transverse axis) rotates the
    non-square mode 90 degrees onto this plane and badly mis-reads it. This pins
    the orientation fix (previously this case read ~0 in the acceptance bend's
    y-input and the crossing's y-arm)."""
    ny, nx = te0.field.shape
    zc = (np.arange(ny + 8) - (ny + 7) / 2.0) * DL_UM  # t1 = z (thickness)
    xc = (np.arange(nx + 8) - (nx + 7) / 2.0) * DL_UM  # t2 = x (width)
    # axis="y" -> (t1,t2)=(z,x); modal_fields returns [i_t2, i_t1] = [i_x, i_z].
    m = modal_fields(te0, zc, xc, axis="y", direction="+", center_um=(0.0, 0.0),
                     thickness_axis="z")
    plane = {
        "Ez": xr.DataArray(m["e1"], dims=("x", "z"), coords={"x": xc, "z": zc}),
        "Ex": xr.DataArray(m["e2"], dims=("x", "z"), coords={"x": xc, "z": zc}),
        "Hz": xr.DataArray(m["h1"], dims=("x", "z"), coords={"x": xc, "z": zc}),
        "Hx": xr.DataArray(m["h2"], dims=("x", "z"), coords={"x": xc, "z": zc}),
    }
    # Correct orientation recovers T == 1.
    T = next(iter(mode_transmission(plane, te0, axis="y", direction="+",
                                    thickness_axis="z").values()))
    assert T == pytest.approx(1.0, rel=1e-3)
    # The legacy mapping (no thickness_axis) rotates the non-square mode -> large
    # mismatch -> T well below 1. This is exactly the bug thickness_axis fixes.
    T_legacy = next(iter(mode_transmission(plane, te0, axis="y",
                                           direction="+").values()))
    assert T_legacy < 0.5


def _forward_vector_mode(nx=9, ny=7, dl=DL_UM, n_eff=2.4):
    """A synthetic VectorMode whose transverse H obeys the FORWARD relation
    hx = -(n_eff/ETA0)*ey, hy = +(n_eff/ETA0)*ex — so its own +propagation
    Poynting (ex*hy* - ey*hx*) is strictly positive."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    ex = np.exp(-(((xx - (nx - 1) / 2) / 2.0) ** 2 + ((yy - (ny - 1) / 2) / 1.2) ** 2))
    ey = 0.3 * ex * (xx - (nx - 1) / 2) / nx        # a nonzero minor-E so the swap matters
    ex = ex.astype(np.complex128); ey = ey.astype(np.complex128)
    s = n_eff / ETA0
    hx, hy = -s * ey, s * ex
    z = np.zeros_like(ex)
    return VectorMode(n_eff=n_eff, n_group=None, ex=ex, ey=ey, ez=z,
                      hx=hx, hy=hy, hz=z, wavelength_um=WL_UM,
                      dl_x_um=dl, dl_y_um=dl)


def _plane_poynting(fv, t1c, t2c):
    dA = np.outer(_cell_widths(t2c), _cell_widths(t1c))
    s = 0.5 * np.real(fv["e1"] * np.conj(fv["h2"]) - fv["e2"] * np.conj(fv["h1"]))
    return float(np.sum(s * dA))


@pytest.mark.parametrize("axis,thickness_axis", [
    ("y", "z"),   # swap branch (t1,t2)=(z,x), thickness on t1 — the z-slab y-port
    ("x", "y"),   # swap branch (t1,t2)=(y,z), thickness on t1
    ("z", "y"),   # non-swap control
    ("y", "x"),   # non-swap control
])
def test_vector_readout_direction_is_consistent(axis, thickness_axis):
    """Regression: on a swap-branch plane (thickness on the FIRST transverse
    axis) the assembled full-vector mode had its transverse H unnegated after a
    handedness-flipping reflection, so a direction="+" readout carried −axis
    Poynting — the returned "forward" mode was the BACKWARD traveler (the crossing
    XT / y-port through-power read the wrong direction). A forward mode must give
    POSITIVE plane Poynting for "+" and NEGATIVE for "-", on every orientation."""
    mode = _forward_vector_mode()
    ny, nx = mode.ex.shape
    t1c = (np.arange(nx + 6) - (nx + 5) / 2.0) * DL_UM
    t2c = (np.arange(ny + 6) - (ny + 5) / 2.0) * DL_UM
    fwd = vector_modal_fields(mode, t1c, t2c, axis=axis, direction="+",
                              center_um=(0.0, 0.0), thickness_axis=thickness_axis)
    bwd = vector_modal_fields(mode, t1c, t2c, axis=axis, direction="-",
                              center_um=(0.0, 0.0), thickness_axis=thickness_axis)
    p_fwd = _plane_poynting(fwd, t1c, t2c)
    p_bwd = _plane_poynting(bwd, t1c, t2c)
    assert p_fwd > 0, f"direction='+' must carry +axis power (got {p_fwd:.3e})"
    assert p_bwd < 0, f"direction='-' must carry -axis power (got {p_bwd:.3e})"


# --- per-frequency n_eff override threads through ---------------------------

def test_n_eff_override_is_consistent(te0: Mode):
    """Passing a custom n_eff for both the plane build and the projection keeps
    self-overlap at 1 (the H scale cancels in the power ratio)."""
    x, y = _plane_axes(te0)
    neff = te0.n_eff * 0.9
    plane = _build_plane(te0, x, y, direction="+", n_eff=neff)
    T = next(iter(mode_transmission(plane, te0, axis="z", direction="+",
                                    n_eff=neff).values()))
    assert T == pytest.approx(1.0, rel=1e-3)


# --- gaussian hand-built mode (independent of the FDE solver) ---------------

def test_handbuilt_gaussian_mode_self_overlap():
    """A gaussian profile wrapped as a Mode self-overlaps to 1 — confirms the
    pipeline needs only the frozen Mode surface, not the eigen-solver."""
    n = 41
    dl = 0.02
    xs = (np.arange(n) - (n - 1) / 2.0) * dl
    X, Y = np.meshgrid(xs, xs)
    prof = np.exp(-(X ** 2 + Y ** 2) / (2 * 0.1 ** 2))
    prof /= np.sqrt(np.sum(prof ** 2))   # L2-normalized, like the FDE Mode
    mode = Mode(n_eff=2.5, field=prof, wavelength_um=1.31,
                polarization="TE", dl_x_um=dl, dl_y_um=dl)
    x, y = _plane_axes(mode, pad_cells=4, dl_um=dl)
    plane = _build_plane(mode, x, y, direction="+")
    T = next(iter(mode_transmission(plane, mode, axis="z",
                                    direction="+").values()))
    assert T == pytest.approx(1.0, rel=1e-3)


# --- input validation -------------------------------------------------------

def test_missing_component_raises(te0: Mode):
    x, y = _plane_axes(te0)
    plane = _build_plane(te0, x, y, direction="+")
    del plane["Hy"]
    with pytest.raises(ValueError):
        mode_transmission(plane, te0, axis="z", direction="+")


def test_bad_axis_and_direction(te0: Mode):
    x, y = _plane_axes(te0)
    plane = _build_plane(te0, x, y, direction="+")
    with pytest.raises(ValueError):
        mode_transmission(plane, te0, axis="w")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        modal_fields(te0, x, y, axis="z", direction="?")  # type: ignore[arg-type]


# --- resample helper sanity -------------------------------------------------

def test_resample_zero_fills_outside_window():
    prof = np.ones((3, 3))
    src = np.array([-1.0, 0.0, 1.0])
    dst = np.array([-5.0, 0.0, 5.0])      # ends are outside the source window
    out = resample_profile(prof, src, src, dst, dst)
    assert out[1, 1] == pytest.approx(1.0)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[2, 2] == pytest.approx(0.0)


# --- full-vector overlap dispatch (issue #34, Stage 1) ----------------------
# When given a VectorMode (true transverse H, not the scalar-limit
# (n_eff/eta0)·z_hat x e), _overlap_terms projects with vector_modal_fields.
# These pin that dispatch on SYNTHETIC planes built from the vector mode itself.

from photonhub.plugins.vector_modes import VectorModeSolver  # noqa: E402
from photonhub.plugins.mode_overlap import vector_modal_fields  # noqa: E402


@pytest.fixture(scope="module")
def te0_vec():
    return VectorModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM, core_w_um=CORE_W_UM,
        core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2,
        subpixel=False).solve(num_modes=1, n_guess=3.0)[0]


@pytest.fixture(scope="module")
def te0_vec_wide():
    return VectorModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM, core_w_um=0.80,
        core_h_um=CORE_H_UM, n_core=N_SI, n_clad=N_SIO2,
        subpixel=False).solve(num_modes=1, n_guess=3.0)[0]


def _vec_plane_axes(vmode, *, pad_cells: int = 6):
    """Transverse (x, y) coords for a z-normal plane centered on a VectorMode."""
    ny, nx = vmode.shape
    nX, nY = nx + 2 * pad_cells, ny + 2 * pad_cells
    x = (np.arange(nX) - (nX - 1) / 2.0) * vmode.dl_x_um
    y = (np.arange(nY) - (nY - 1) / 2.0) * vmode.dl_y_um
    return x, y


def _build_vector_plane(vmode, x, y, *, direction="+", scale=1.0):
    """Assemble (Ex, Ey, Hx, Hy) for a z-normal plane that IS `scale`·(vector
    mode) travelling in `direction` — using the mode's TRUE transverse fields."""
    f = vector_modal_fields(vmode, x, y, axis="z", direction=direction,
                            center_um=(0.0, 0.0))
    comps = {"Ex": f["e1"], "Ey": f["e2"], "Hx": f["h1"], "Hy": f["h2"]}
    return {n: xr.DataArray(scale * arr, dims=("y", "x"),
                            coords={"y": y, "x": x}) for n, arr in comps.items()}


def test_vector_self_overlap_is_one(te0_vec):
    """A plane that IS the forward VECTOR mode reads forward T ~= 1 (dispatch to
    the true-h overlap)."""
    x, y = _vec_plane_axes(te0_vec)
    plane = _build_vector_plane(te0_vec, x, y, direction="+")
    Tval = next(iter(mode_transmission(plane, te0_vec, axis="z",
                                       direction="+").values()))
    assert Tval == pytest.approx(1.0, rel=1e-3)


def test_vector_directionality(te0_vec):
    """Backward vector mode -> forward T ~= 0, backward T ~= 1."""
    x, y = _vec_plane_axes(te0_vec)
    plane = _build_vector_plane(te0_vec, x, y, direction="-")
    Tf = next(iter(mode_transmission(plane, te0_vec, axis="z",
                                     direction="+").values()))
    Tb = next(iter(mode_transmission(plane, te0_vec, axis="z",
                                     direction="-").values()))
    assert Tf == pytest.approx(0.0, abs=1e-6)
    assert Tb == pytest.approx(1.0, rel=1e-3)


def test_vector_amplitude_scaling_is_power_ratio(te0_vec):
    """Scaling the plane by c scales the vector-overlap power by |c|^2."""
    x, y = _vec_plane_axes(te0_vec)
    base = next(iter(mode_transmission(
        _build_vector_plane(te0_vec, x, y), te0_vec, axis="z",
        direction="+", power=True).values()))
    scaled = next(iter(mode_transmission(
        _build_vector_plane(te0_vec, x, y, scale=2.0), te0_vec, axis="z",
        direction="+", power=True).values()))
    assert scaled == pytest.approx(4.0 * base, rel=1e-3)


def test_vector_power_transmission_unequal_width_ports(te0_vec, te0_vec_wide):
    """Lossless w1->w2 transition between two DIFFERENT-width VECTOR modes reads
    T~=1 by power (the cross-mode normalization Stage-2 must preserve)."""
    x1, y1 = _vec_plane_axes(te0_vec)
    x2, y2 = _vec_plane_axes(te0_vec_wide)

    def txp(plane, mode):
        return next(iter(mode_transmission(
            plane, mode, axis="z", direction="+", power=True).values()))

    p1 = txp(_build_vector_plane(te0_vec, x1, y1), te0_vec)
    p2 = txp(_build_vector_plane(te0_vec_wide, x2, y2), te0_vec_wide)
    assert p1 > 0 and p2 > 0
    a_out = (p1 / p2) ** 0.5            # equal power in/out (lossless)
    T = (txp(_build_vector_plane(te0_vec_wide, x2, y2, scale=a_out), te0_vec_wide)
         / txp(_build_vector_plane(te0_vec, x1, y1), te0_vec))
    assert T == pytest.approx(1.0, rel=1e-3)


def test_colocate_to_node_recovers_linear_field():
    """``_colocate_to_node`` averages a +½-cell Yee-staggered component back onto
    the cell node — exact for a linear field (the leading order). A ramp
    ``f(x)=x`` sampled at the half-offsets (``a[j]=j+½``) must recover the integer
    node values ``j`` in the interior, on either array axis."""
    import numpy as np

    n = 8
    a1 = (np.arange(n) + 0.5)[None, :].astype(float)        # stagger on last axis
    node1 = _colocate_to_node(a1, -1)
    assert np.allclose(node1[0, 1:], np.arange(1, n))       # node[j]=½(a[j-1]+a[j])=j
    assert node1[0, 0] == pytest.approx(0.25)               # edge: ½·a[0] (a[-1]≡0)

    a0 = np.repeat((np.arange(n) + 0.5)[:, None], 3, axis=1)  # stagger on axis 0
    node0 = _colocate_to_node(a0, 0)
    assert np.allclose(node0[1:, :], np.repeat(np.arange(1, n)[:, None], 3, axis=1))
