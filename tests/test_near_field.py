"""Near-to-far-field (NTFF) projection — physics pins.

Validates ``photonhub.plugins.near_field.far_field`` on SYNTHETIC near-field
surfaces (no FDTD run): the near fields are built analytically as xarray
DataArrays shaped like a real ``field_dft`` monitor slice (dims
``f/component/z/y/x``), projected, and the far-field pattern is asserted against
its closed form.

Primary analytic case — UNIFORM RECTANGULAR APERTURE
====================================================
A plane ``z = 0`` aperture uniformly illuminated by a (forward) plane wave
``E_x = E0`` over ``|x|<=a/2, |y|<=b/2`` (zero outside), with the paired plane-wave
``H_y = E0/eta0``, radiates into the upper half space with the textbook pattern

    U(theta, phi) ~ [ sinc(k a/2 sin th cos ph) sinc(k b/2 sin th sin ph) ]^2
                    * [ (1 + cos th)/2 ]^2 ,

i.e. the 2-D Fourier transform of the aperture (the array factor) times the
Huygens-source element factor ``(1+cos th)/2`` that the equivalence currents
``J=n_hat x H`` and ``M=-n_hat x E`` together produce. The projector reproduces
this to ~5e-5 (limited only by the hard aperture edge on a finite grid, not by
the projection algebra). This is the canonical NTFF analytic check.

Beam steering (Fourier shift). A linear phase taper ``E_x = e^{i k0 x}`` on the
aperture steers the main lobe to ``sin(theta0) = k0/k``; a ``+x`` phase progression
radiates toward ``+x`` (pins the kernel SIGN / time convention).

Power conservation (the reciprocity/sanity check). For a LARGE aperture
(``a,b >> wavelength``) essentially all the power radiates into the forward
hemisphere, so the NTFF-integrated radiated power approaches the near-field
Poynting flux through the aperture (here matched to ~2%).

Directivity machinery. Fed an EXACT ``sin^2(theta)`` intensity pattern, the
solid-angle quadrature recovers the Hertzian-dipole directivity ``D = 1.5`` and
``P_rad = 8 pi/3`` (validates ``FarField.radiated_power`` / ``.directivity``
independently of the projection).

Closed-surface (6-face box) — SMOKE/SHAPE only
==============================================
A z-oriented Hertzian dipole's exact near fields on a closed box project to the
``sin^2(theta)`` doughnut with the main lobe at the equator and finite positive
radiated power. The 6-face box around a point source's reactive near field is a
known-finicky discretization (the side faces over/under-cancel), so this is a
LOOSE shape/main-lobe sanity check, not a tight accuracy gate — the single-plane
aperture above is the rigorous validation.

All coordinates are in MICRONS (the PhotonHub convention); the plugin converts
to metres internally. Time convention ``e^{-i omega t}`` (matching mode_overlap).
"""

import functools

import numpy as np
import pytest
import xarray as xr

from photonhub.plugins import FarField, far_field as _far_field
from photonhub.plugins.near_field import C0, ETA0, _TANGENTIAL, _TRANSVERSE

# These tests feed SYNTHETIC, already-co-located analytic planes (E and H built
# at the SAME grid points), so the Yee co-location — correct only for the
# engine's STAGGERED DFT output — must be OFF here (same convention as
# test_mode_overlap / test_smatrix). Real runs default colocate=True.
far_field = functools.partial(_far_field, colocate=False)

WL_UM = 1.0
F0 = C0 / (WL_UM * 1e-6)        # Hz
K = 2.0 * np.pi * F0 / C0       # 1/m
K_UM = K * 1e-6                 # 1/micron (phase per micron of travel)


def _sinc(t: np.ndarray) -> np.ndarray:
    """sin(t)/t (numpy's sinc is sin(pi x)/(pi x))."""
    return np.sinc(t / np.pi)


def _aperture_plane(a_um, b_um, dl_um, *, pad=1.4, taper_k_um=0.0,
                    e_component="Ex"):
    """A z=0 uniform rectangular aperture as a DFT-shaped field mapping.

    ``E`` lives in ``e_component`` (``"Ex"`` or ``"Ey"``); the paired plane-wave
    ``H`` is ``(zhat x E)/eta0`` so the wave travels toward ``+z``. A non-zero
    ``taper_k_um`` adds a linear phase ``e^{i taper_k_um x}`` across the aperture
    (beam steering in the x-z plane). Coordinates in microns."""
    half_x, half_y = pad * a_um, pad * b_um
    x = np.arange(-half_x, half_x + dl_um / 2, dl_um)
    y = np.arange(-half_y, half_y + dl_um / 2, dl_um)
    X, Y = np.meshgrid(x, y, indexing="xy")          # [iy, ix]
    mask = (np.abs(X) <= a_um / 2) & (np.abs(Y) <= b_um / 2)
    E = mask.astype(np.complex128)
    if taper_k_um:
        E = E * np.exp(1j * taper_k_um * X)
    H = E / ETA0                                     # |H| = |E|/eta0
    comps = {"Ex": np.zeros_like(E), "Ey": np.zeros_like(E),
             "Hx": np.zeros_like(E), "Hy": np.zeros_like(E)}
    if e_component == "Ex":
        comps["Ex"], comps["Hy"] = E, H              # zhat x (Ex xhat) = Ex yhat
    else:
        comps["Ey"], comps["Hx"] = E, -H             # zhat x (Ey yhat) = -Ey xhat

    out = {}
    for name in _TANGENTIAL["z"]:                    # Ex, Ey, Hx, Hy
        out[name] = xr.DataArray(
            comps[name][None, None, None, :, :],
            dims=("f", "component", "z", "y", "x"),
            coords={"f": [F0], "component": [name], "z": [0.0], "y": y, "x": x})
    # Stack into one component-indexed DataArray, as data[monitor] would be.
    da = xr.concat(list(out.values()), dim="component")
    return {"ap": da}, x, y


# --- primary analytic check: uniform aperture sinc^2 pattern ----------------

@pytest.mark.parametrize("phi", [0.0, np.pi / 2])
def test_uniform_aperture_matches_sinc_pattern(phi):
    """The far field of a uniform rectangular aperture is the (obliquity-weighted)
    sinc^2 Fourier transform of the aperture, in both principal cuts."""
    a, b, dl = 4.0, 6.0, 0.05
    data, _, _ = _aperture_plane(a, b, dl)
    theta = np.linspace(0.0, np.pi / 2 * 0.95, 80)
    ff = far_field(data, "ap", theta=theta, phi=phi, axis="z", sign=1.0)
    U = ff.intensity[0]
    U = U / U.max()

    u = K * (a * 1e-6) * np.sin(theta) * np.cos(phi) / 2.0
    v = K * (b * 1e-6) * np.sin(theta) * np.sin(phi) / 2.0
    obliquity = ((1.0 + np.cos(theta)) / 2.0) ** 2
    ana = (_sinc(u) * _sinc(v)) ** 2 * obliquity
    ana = ana / ana.max()

    assert np.max(np.abs(U - ana)) < 5e-3
    # Main lobe at broadside (theta = 0).
    assert theta[np.argmax(U)] == pytest.approx(0.0, abs=1e-9)


def test_on_axis_absolute_sign_matches_rayleigh_sommerfeld():
    """The ABSOLUTE sign/phase of the returned far field, not just its magnitude.

    Every other test here reads ``intensity``/power/lobe position, all invariant
    under a global sign — which is exactly how a constant pi absolute-phase error
    survived: Balanis' e^{+j omega t} formulas were adapted to e^{-i omega t} by
    flipping the exponential but NOT the leading ``j k``, leaving E_theta/E_phi
    globally negated.

    On-axis (theta=0) a Huygens patch carrying Ex=g with the paired plane-wave
    Hy=g/eta0 must reproduce the Rayleigh-Sommerfeld amplitude in this same
    convention (Goodman): F = -i k / (2 pi) * Int g dS. Derivation: at theta=0,
    theta_hat = x_hat, so L_phi = -Int g dS and N_theta = -(1/eta0) Int g dS, and
    F_theta = pref*(L_phi + eta0*N_theta) = -(i k / 2 pi) Int g dS.
    """
    a, b, dl = 4.0, 4.0, 0.05
    data, x, y = _aperture_plane(a, b, dl)
    ff = far_field(data, "ap", theta=np.array([0.0]), phi=0.0, axis="z")

    # Int g dS over the SAME discrete aperture, in m^2 (uniform grid -> dA = dl^2).
    Ex = data["ap"].sel(component="Ex").isel(f=0).squeeze(drop=True).values
    integral = np.sum(Ex) * (dl * 1e-6) ** 2
    expected = -1j * K / (2.0 * np.pi) * integral

    got = complex(ff.e_theta[0, 0])
    # Ratio == +1 (not -1): pins magnitude AND the absolute sign together.
    ratio = got / expected
    assert ratio.real == pytest.approx(1.0, rel=2e-2), f"got {got}, expected {expected}"
    assert abs(ratio.imag) < 2e-2, f"unexpected phase: ratio={ratio}"


def test_uniform_aperture_first_null_location():
    """The first null of the E-plane (phi=0) pattern sits at the analytic
    ``sin(theta_null) = wavelength / a`` (the sinc's first zero)."""
    a, b, dl = 5.0, 5.0, 0.04
    data, _, _ = _aperture_plane(a, b, dl)
    theta = np.linspace(0.001, 0.6, 600)
    U = far_field(data, "ap", theta=theta, phi=0.0, axis="z").intensity[0]
    # First local minimum away from broadside.
    null_theta = theta[1:-1][(U[1:-1] < U[:-2]) & (U[1:-1] < U[2:])][0]
    assert np.sin(null_theta) == pytest.approx(WL_UM / a, rel=2e-2)


def test_uniform_aperture_tight_against_obliquity_model():
    """Against the full equivalence-current closed form (sinc^2 array factor times
    the ``((1+cos th)/2)^2`` element factor) the agreement is ~5e-5 — limited only
    by the hard aperture edge sampled on a finite grid, NOT by the projection
    algebra. This pins the obliquity model, not just the broadside shape."""
    a, b, dl = 4.0, 6.0, 0.05
    data, _, _ = _aperture_plane(a, b, dl)
    theta = np.linspace(0.0, np.pi / 2 * 0.95, 60)
    U = far_field(data, "ap", theta=theta, phi=0.0, axis="z").intensity[0]
    U = U / U.max()
    u = K * (a * 1e-6) * np.sin(theta) / 2.0
    ana = (_sinc(u)) ** 2 * ((1.0 + np.cos(theta)) / 2.0) ** 2
    ana = ana / ana.max()
    assert np.max(np.abs(U - ana)) < 2e-4


def test_colocate_path_keeps_aperture_pattern():
    """The default colocate=True path (correct for real staggered DFT data):
    on an already-co-located synthetic plane the node-averaging amounts to a
    half-cell translation plus mild edge smoothing, so the normalized intensity
    pattern must stay close to the analytic sinc^2 form (|FT| is translation-
    invariant) — a loose pin that the co-location path is wired correctly."""
    a, b, dl = 4.0, 6.0, 0.05
    data, _, _ = _aperture_plane(a, b, dl)
    theta = np.linspace(0.0, np.pi / 2 * 0.95, 60)
    U = _far_field(data, "ap", theta=theta, phi=0.0, axis="z",
                   colocate=True).intensity[0]
    U = U / U.max()
    u = K * (a * 1e-6) * np.sin(theta) / 2.0
    ana = (_sinc(u)) ** 2 * ((1.0 + np.cos(theta)) / 2.0) ** 2
    ana = ana / ana.max()
    assert np.max(np.abs(U - ana)) < 5e-3


# --- beam steering (Fourier shift -> kernel sign / time convention) ---------

@pytest.mark.parametrize("theta0_deg", [15.0, -25.0])
def test_linear_phase_taper_steers_main_lobe(theta0_deg):
    """A linear aperture phase ``e^{i k0 x}`` steers the main lobe to
    ``sin(theta0) = k0/k`` (toward +x for +k0), pinning the kernel SIGN."""
    a, b, dl = 8.0, 8.0, 0.05
    theta0 = np.deg2rad(theta0_deg)
    k0_um = K_UM * np.sin(theta0)                    # phase per micron
    data, _, _ = _aperture_plane(a, b, dl, taper_k_um=k0_um)
    theta = np.linspace(0.0, np.pi / 2 * 0.9, 240)
    # Signed x-z cut: phi=0 is +x, phi=pi is -x.
    Up = far_field(data, "ap", theta=theta, phi=0.0, axis="z").intensity[0]
    Um = far_field(data, "ap", theta=theta, phi=np.pi, axis="z").intensity[0]
    signed = np.concatenate([-theta[::-1], theta])
    U = np.concatenate([Um[::-1], Up])
    lobe = signed[np.argmax(U)]
    assert np.rad2deg(lobe) == pytest.approx(theta0_deg, abs=1.0)


# --- power conservation (reciprocity / sanity) ------------------------------

def test_radiated_power_matches_near_field_flux_large_aperture():
    """For a LARGE aperture (a,b >> wavelength) nearly all the power radiates
    forward, so the NTFF hemisphere radiated power approaches the near-field
    Poynting flux through the aperture."""
    a, b, dl = 20.0, 20.0, 0.1
    data, x, y = _aperture_plane(a, b, dl)
    da = data["ap"]
    Ex = da.sel(component="Ex").squeeze(drop=True).values
    Hy = da.sel(component="Hy").squeeze(drop=True).values
    dA = (dl * 1e-6) ** 2                             # m^2
    p_near = 0.5 * np.real(np.sum(Ex * np.conj(Hy)) * dA)

    thg = np.linspace(1e-3, np.pi / 2 - 1e-3, 120)
    phg = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
    ff = far_field(data, "ap", theta=thg, phi=phg, axis="z", sign=1.0, mesh=True)
    p_rad = ff.radiated_power()[0]

    assert p_rad > 0.0
    assert p_rad / p_near == pytest.approx(1.0, abs=0.05)


def test_radiated_power_positive_and_finite():
    """A modest aperture still radiates finite, positive power into the
    hemisphere (basic sanity)."""
    data, _, _ = _aperture_plane(3.0, 3.0, 0.05)
    thg = np.linspace(1e-3, np.pi / 2 - 1e-3, 60)
    phg = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    p = far_field(data, "ap", theta=thg, phi=phg, axis="z", mesh=True
                  ).radiated_power()[0]
    assert np.isfinite(p) and p > 0.0


# --- directivity machinery (independent of the projection) ------------------

def test_directivity_quadrature_recovers_dipole_1p5():
    """Fed an EXACT sin(theta) far-field amplitude (intensity sin^2), the
    solid-angle quadrature returns the Hertzian-dipole directivity D=1.5 and the
    analytic radiated power 8 pi/3 — validates radiated_power()/directivity()."""
    th_ax = np.linspace(0.0, np.pi, 200)
    ph_ax = np.linspace(0.0, 2 * np.pi, 60, endpoint=False)
    TH, PH = np.meshgrid(th_ax, ph_ax, indexing="ij")
    theta, phi = TH.reshape(-1), PH.reshape(-1)
    e_theta = (np.sin(theta) * np.sqrt(2.0 * ETA0))[None, :].astype(np.complex128)
    e_phi = np.zeros_like(e_theta)
    ff = FarField(theta=theta, phi=phi, freqs_hz=np.array([F0]),
                  e_theta=e_theta, e_phi=e_phi)
    assert ff.radiated_power()[0] == pytest.approx(8.0 * np.pi / 3.0, rel=1e-3)
    assert np.max(ff.directivity()[0]) == pytest.approx(1.5, rel=2e-3)


# --- closed-surface (6-face box) Hertzian dipole sin^2 (loose shape) --------

def _dipole_fields_um(x_um, y_um, z_um):
    """Exact z-oriented Hertzian-dipole fields (I0 l = 1, e^{-i omega t},
    outgoing e^{+ikr}) sampled at positions given in MICRONS, returned as the
    six Cartesian complex components."""
    xm, ym, zm = x_um * 1e-6, y_um * 1e-6, z_um * 1e-6
    r = np.sqrt(xm ** 2 + ym ** 2 + zm ** 2)
    th = np.arccos(np.clip(zm / r, -1.0, 1.0))
    ph = np.arctan2(ym, xm)
    e = np.exp(1j * K * r)
    Er = ETA0 * np.cos(th) / (2 * np.pi * r ** 2) * (1 + 1 / (1j * K * r)) * e
    Eth = (1j * ETA0 * K * np.sin(th) / (4 * np.pi * r)
           * (1 + 1 / (1j * K * r) - 1 / (K * r) ** 2) * e)
    Hph = 1j * K * np.sin(th) / (4 * np.pi * r) * (1 + 1 / (1j * K * r)) * e
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    return {
        "Ex": Er * st * cp + Eth * ct * cp,
        "Ey": Er * st * sp + Eth * ct * sp,
        "Ez": Er * ct - Eth * st,
        "Hx": -Hph * sp, "Hy": Hph * cp, "Hz": np.zeros_like(Hph),
    }


def _dipole_box(half_um, dl_um):
    """A closed box of side ``2*half_um`` centered on a z-dipole: a ``data`` dict
    of per-face DFT DataArrays + the ``faces`` list ``(name, axis, sign)``."""
    g = np.arange(-half_um, half_um + dl_um / 2, dl_um)
    data, faces = {}, []
    for ax in ("x", "y", "z"):
        t1, t2 = _TRANSVERSE[ax]
        A, B = np.meshgrid(g, g, indexing="ij")      # A along t1, B along t2
        for pos, sign, end in ((-half_um, -1.0, "lo"), (half_um, 1.0, "hi")):
            full = {ax: np.full_like(A, pos), t1: A, t2: B}
            fl = _dipole_fields_um(full["x"], full["y"], full["z"])
            das = []
            for c in _TANGENTIAL[ax]:
                # dims (t2, t1) to match the [i_t2, i_t1] layout the plugin reads.
                da = xr.DataArray(
                    fl[c].T[None, :, :], dims=("f", t2, t1),
                    coords={"f": [F0], t2: g, t1: g}
                ).assign_coords(**{ax: pos, "component": c}).expand_dims("component")
                das.append(da)
            name = f"{ax}_{end}"
            data[name] = xr.concat(das, dim="component")
            faces.append((name, ax, sign))
    return data, faces


def test_dipole_closed_box_sin2_shape():
    """The z-dipole's closed-box projection gives the sin^2(theta) doughnut: a
    null toward the poles, the peak at the equator. LOOSE shape check (the small
    box around a point source's reactive near field is a known-finicky
    discretization)."""
    data, faces = _dipole_box(half_um=0.4, dl_um=WL_UM / 20)
    theta = np.linspace(0.05, np.pi - 0.05, 40)
    U = far_field(data, monitor_name="", theta=theta, phi=0.3,
                  faces=faces).intensity[0]
    U = U / U.max()
    # Main lobe at the equator (theta ~ pi/2), nulls at the poles.
    assert np.rad2deg(theta[np.argmax(U)]) == pytest.approx(90.0, abs=8.0)
    assert U[0] < 0.1 and U[-1] < 0.1                # poles are nulls
    # Overall shape tracks sin^2 to within the closed-box discretization error.
    assert np.max(np.abs(U - np.sin(theta) ** 2)) < 0.12


def test_closed_box_radiated_power_positive():
    """The closed-box projection radiates finite positive power over the full
    sphere (sanity for the multi-face API)."""
    data, faces = _dipole_box(half_um=0.4, dl_um=WL_UM / 20)
    thg = np.linspace(0.02, np.pi - 0.02, 50)
    phg = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    p = far_field(data, "", theta=thg, phi=phg, faces=faces,
                  mesh=True).radiated_power()[0]
    assert np.isfinite(p) and p > 0.0


# --- API / reshape / multi-frequency / inference ----------------------------

def test_axis_inference_from_singleton_dim():
    """``axis`` is inferred from the monitor's singleton spatial dim (z here)."""
    data, _, _ = _aperture_plane(3.0, 3.0, 0.08)
    ff = far_field(data, "ap", theta=np.linspace(0, 1.0, 10), phi=0.0)
    assert ff.e_theta.shape == (1, 10)


def test_reshape_meshed_grid():
    """``FarField.reshape`` recovers the (n_freq, n_theta, n_phi) layout."""
    data, _, _ = _aperture_plane(3.0, 3.0, 0.08)
    th = np.linspace(0.0, 1.0, 7)
    ph = np.linspace(0.0, 2 * np.pi, 5, endpoint=False)
    ff = far_field(data, "ap", theta=th, phi=ph, axis="z", mesh=True)
    g = ff.reshape(7, 5)
    assert g["intensity"].shape == (1, 7, 5)
    assert np.allclose(g["theta"], th)
    assert np.allclose(g["phi"], ph)


def test_paired_directions_mesh_false():
    """``mesh=False`` pairs theta/phi elementwise (an arbitrary direction list)."""
    data, _, _ = _aperture_plane(3.0, 3.0, 0.08)
    th = np.array([0.1, 0.2, 0.3])
    ph = np.array([0.0, 1.0, 2.0])
    ff = far_field(data, "ap", theta=th, phi=ph, axis="z", mesh=False)
    assert ff.e_theta.shape == (1, 3)
    assert np.allclose(ff.theta, th) and np.allclose(ff.phi, ph)


def test_multifrequency_projection():
    """Two frequencies on the monitor -> a far field per frequency."""
    a, b, dl = 3.0, 3.0, 0.06
    half = 1.4 * a
    x = np.arange(-half, half + dl / 2, dl)
    y = x.copy()
    X, Y = np.meshgrid(x, y, indexing="xy")
    mask = (np.abs(X) <= a / 2) & (np.abs(Y) <= b / 2)
    E = mask.astype(np.complex128)
    H = E / ETA0
    f0, f1 = F0, F0 * 1.05
    comps = {"Ex": E, "Ey": np.zeros_like(E), "Hx": np.zeros_like(E), "Hy": H}
    das = []
    for name in _TANGENTIAL["z"]:
        arr = np.stack([comps[name], comps[name]])           # same field both freqs
        das.append(xr.DataArray(
            arr[:, None, :, :], dims=("f", "z", "y", "x"),
            coords={"f": [f0, f1], "z": [0.0], "y": y, "x": x}
        ).assign_coords(component=name).expand_dims("component"))
    data = {"ap": xr.concat(das, dim="component")}
    ff = far_field(data, "ap", theta=np.linspace(0, 1.0, 12), phi=0.0, axis="z")
    assert ff.freqs_hz.shape == (2,)
    assert ff.e_theta.shape == (2, 12)


def test_bad_axis_raises():
    data, _, _ = _aperture_plane(3.0, 3.0, 0.1)
    with pytest.raises(ValueError):
        far_field(data, "ap", theta=0.5, phi=0.0, axis="w")  # type: ignore[arg-type]


def test_mismatched_paired_directions_raise():
    data, _, _ = _aperture_plane(3.0, 3.0, 0.1)
    with pytest.raises(ValueError):
        far_field(data, "ap", theta=[0.1, 0.2], phi=[0.0], axis="z", mesh=False)
