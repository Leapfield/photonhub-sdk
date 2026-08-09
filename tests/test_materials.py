"""Material library (``photonhub.materials``): literature index values, the
single-pole Lorentz band fit (NUMERICS.md §19), Medium construction, and the
user measured-data path. One focused engine integration test is included when
``phsolver`` is available."""

import math
import warnings

import numpy as np
import pytest

import photonhub as ph
from photonhub import materials
from photonhub.materials import MATERIALS, LorentzFit, Material

C0 = 299792458.0
EPS0 = 8.8541878128e-12


# ---------------------------------------------------------------------------
# literature index pins
# ---------------------------------------------------------------------------

# (material, wavelength_um, n, tolerance) — handbook / paper values evaluated
# from the cited dispersion relations (refractiveindex.info transcriptions).
INDEX_PINS = [
    ("cSi", 1.55, 3.4757, 1e-4),  # Li 1993 table point, exact
    ("cSi", 1.31, 3.5003, 5e-4),  # O-band, interpolated between table points
    ("SiO2", 1.55, 1.44402, 2e-4),  # Malitson 1965
    ("SiO2", 0.5876, 1.45846, 2e-4),  # fused silica n_d, canonical
    ("Si3N4", 1.55, 1.9963, 5e-4),  # Luke 2015
    ("GaAs", 1.55, 3.3702, 5e-4),  # Skauli 2003 (older refs quote ~3.374)
    ("InP", 1.55, 3.1649, 5e-4),  # Pettit & Turner
    ("Ge", 4.0, 4.0243, 1e-3),  # Burnett 2016
    ("Sapphire", 0.5893, 1.7681, 1e-3),  # Malitson & Dodge, o-ray
    ("AlN", 1.55, 2.1199, 1e-3),  # Pastrnak & Roskovcova, o-ray
    ("TiO2", 0.5893, 2.6129, 1e-3),  # DeVore 1951, rutile o-ray
    ("LiNbO3_o", 1.55, 2.2111, 5e-4),  # Zelmon 1997
    ("LiNbO3_e", 1.55, 2.1376, 5e-4),
    ("MgF2_o", 0.5876, 1.37774, 2e-4),  # Dodge 1984
    ("MgF2_e", 0.5876, 1.38956, 2e-4),
    ("CaF2", 0.5876, 1.43385, 2e-4),  # Malitson 1963
    ("PMMA", 0.5876, 1.4925, 1e-3),  # Beadie 2015
    ("PMMA", 1.55, 1.4809, 1e-3),
    ("Vacuum", 1.55, 1.0, 0.0),
]


@pytest.mark.parametrize("name,wl,expected,tol", INDEX_PINS)
def test_literature_index_pins(name, wl, expected, tol):
    assert materials.get(name).n(wl) == pytest.approx(expected, abs=max(tol, 1e-12))


def test_registry_entries_are_complete():
    assert len(MATERIALS) == 16
    for name, mat in MATERIALS.items():
        assert mat.name == name
        lo, hi = mat.valid_range_um
        assert 0.0 <= lo < hi
        assert mat.reference
        # every model evaluates to a physical index mid-range (vectorized)
        mid = 0.5 * (lo + min(hi, 20.0))
        n = mat.n(np.asarray([mid, mid * 1.01]))
        assert n.shape == (2,)
        assert np.all(np.isfinite(n)) and np.all(n >= 1.0)


def test_get_unknown_material_lists_choices():
    with pytest.raises(KeyError, match="cSi"):
        materials.get("unobtainium")


def test_out_of_range_raises_with_range_in_message():
    with pytest.raises(ValueError, match=r"\[0.43, 1.53\]"):
        materials.TiO2.n(1.55)  # DeVore validity ends short of the C-band
    with pytest.raises(ValueError):
        materials.Ge.n(1.55)  # Ge is absorbing there; data starts at 2 um
    with pytest.raises(ValueError):
        materials.SiO2.n(np.asarray([1.55, 7.5]))  # any element out -> raise


def test_eps_is_complex_square_of_n_ik():
    wl = np.linspace(1.5, 1.6, 5)
    lossy = Material.from_nk_data(
        "lossy", wavelength_um=wl, n=np.full(5, 2.0), k=np.full(5, 0.1)
    )
    eps = lossy.eps(1.55)
    assert eps == pytest.approx((2.0 + 0.1j) ** 2)
    assert materials.SiO2.eps(1.55).imag == 0.0


# ---------------------------------------------------------------------------
# constant (single-wavelength) Medium
# ---------------------------------------------------------------------------


def test_constant_medium_is_n_squared_without_pole():
    m = materials.SiO2.medium(1.55)
    assert isinstance(m, ph.Medium)
    assert m.lorentz is None
    assert m.conductivity_s_per_m == 0.0
    assert m.permittivity == pytest.approx(materials.SiO2.n(1.55) ** 2)


def test_constant_medium_maps_absorption_to_conductivity():
    wl = np.linspace(1.5, 1.6, 5)
    n0, k0 = 2.0, 1e-4
    lossy = Material.from_nk_data(
        "lossy", wavelength_um=wl, n=np.full(5, n0), k=np.full(5, k0)
    )
    m = lossy.medium(1.55)
    omega = 2.0 * math.pi * C0 / 1.55e-6
    assert m.permittivity == pytest.approx(n0**2 - k0**2)
    assert m.conductivity_s_per_m == pytest.approx(omega * EPS0 * 2 * n0 * k0)


def test_constant_medium_rejects_sub_unity_permittivity():
    wl = np.linspace(1.5, 1.6, 5)
    low = Material.from_nk_data("low", wavelength_um=wl, n=np.full(5, 0.9))
    with pytest.raises(ValueError, match="permittivity >= 1"):
        low.medium(1.55)


def test_medium_requires_exactly_one_of_wavelength_or_band():
    with pytest.raises(ValueError, match="exactly one"):
        materials.SiO2.medium(1.55, band_um=(1.5, 1.6))
    with pytest.raises(ValueError, match="exactly one"):
        materials.SiO2.medium()
    with pytest.raises(ValueError, match="band_um"):
        materials.SiO2.medium(1.55, pole_wavelength_um=0.5)


# ---------------------------------------------------------------------------
# Lorentz single-pole band fit
# ---------------------------------------------------------------------------

TELECOM_BAND = (1.5, 1.6)
# materials whose validity covers the C-band and that carry real dispersion
TELECOM_MATERIALS = [
    "cSi", "SiO2", "Si3N4", "GaAs", "InP",
    "Sapphire", "AlN", "LiNbO3_o", "LiNbO3_e", "MgF2_o", "MgF2_e",
    "CaF2", "PMMA",
]


@pytest.mark.parametrize("name", TELECOM_MATERIALS)
def test_band_fit_matches_literature_curve(name):
    """The single-pole fit reproduces the literature index over a 100 nm
    telecom band to a few 1e-5 — well under the engine's ~1e-3 parity floor."""
    mat = materials.get(name)
    fit = mat.lorentz_fit(TELECOM_BAND)
    assert fit.max_abs_n_error < 2e-4
    # spot-check the claim on a fresh grid (not the fitter's own samples)
    wl = np.linspace(*TELECOM_BAND, 37)
    assert np.max(np.abs(fit.n_model(wl) - mat.n(wl))) < 2e-4


@pytest.mark.parametrize("name", TELECOM_MATERIALS)
def test_band_fit_is_courant_and_ade_safe(name):
    """Pole placement keeps eps_inf away from the Courant edge (the c-Si
    lesson: a near-UV pole drives eps_inf -> 1 and diverges) and satisfies
    the §19.4 ADE bound omega0*dt < 2 at a coarse 6-cells-per-lambda grid."""
    mat = materials.get(name)
    fit = mat.lorentz_fit(TELECOM_BAND)
    n_band_min = float(np.min(mat.n(np.linspace(*TELECOM_BAND, 33))))
    assert math.sqrt(fit.eps_inf) >= 0.7 * n_band_min
    assert fit.eps_inf >= 1.0
    if fit.pole is not None:
        assert fit.pole.delta_eps >= 0.0
        assert fit.pole.linewidth_hz == 0.0
        dl_coarse = 1.55 / (6.0 * mat.n(1.55))  # 6 c/lambda in the material
        assert fit.omega0_dt(dl_coarse) < 2.0
        assert fit.max_dl_um() > dl_coarse


def test_cSi_fit_reproduces_benchmark_hand_placement():
    """Pinning the pole at 0.6 um reproduces the hand-tuned dispersive-Si
    values used by benchmarks/gds (PR #55/#57): eps_inf ~ 9.62, deps ~ 2.11."""
    fit = materials.cSi.lorentz_fit((1.54, 1.56), pole_wavelength_um=0.6)
    assert fit.eps_inf == pytest.approx(9.62, abs=0.05)
    assert fit.delta_eps == pytest.approx(2.10, abs=0.05)
    assert fit.max_abs_n_error < 1e-4


def test_cSi_fit_reproduces_li_slope():
    """The fitted dn/dlambda at 1.55 um matches Li 1993 (~-0.0806/um)."""
    fit = materials.cSi.lorentz_fit(TELECOM_BAND)
    slope = float((fit.n_model(1.56) - fit.n_model(1.54)) / 0.02)
    assert slope == pytest.approx(-0.0806, rel=0.05)


def test_band_fit_medium_wire_fields():
    m = materials.cSi.medium(band_um=TELECOM_BAND)
    assert isinstance(m.lorentz, ph.LorentzPole)
    assert m.lorentz.resonance_frequency_hz > 0
    assert m.lorentz.delta_eps > 0
    assert m.permittivity >= 1.0
    # eps_inf + pole reproduce the band-centre permittivity
    w0 = 2 * math.pi * m.lorentz.resonance_frequency_hz
    w = 2 * math.pi * C0 / 1.55e-6
    eps_at_155 = m.permittivity + m.lorentz.delta_eps * w0**2 / (w0**2 - w**2)
    assert eps_at_155 == pytest.approx(materials.cSi.n(1.55) ** 2, abs=1e-3)


def test_band_fit_pole_stays_outside_band():
    for name in ("cSi", "SiO2"):
        fit = materials.get(name).lorentz_fit(TELECOM_BAND)
        lam0 = fit.pole_wavelength_um
        assert lam0 is not None
        assert lam0 < TELECOM_BAND[0] or lam0 > TELECOM_BAND[1]


def test_pinned_pole_inside_band_rejected():
    with pytest.raises(ValueError, match="inside the band"):
        materials.cSi.lorentz_fit(TELECOM_BAND, pole_wavelength_um=1.55)


def test_band_fit_rejects_band_outside_validity():
    with pytest.raises(ValueError):
        materials.Ge.lorentz_fit((1.5, 1.6))
    with pytest.raises(ValueError, match="lo < hi"):
        materials.SiO2.lorentz_fit((1.6, 1.5))


def test_flat_index_returns_constant_medium_silently():
    wl = np.linspace(1.4, 1.7, 16)
    flat = Material.from_nk_data("flat", wavelength_um=wl, n=np.full(16, 2.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        m = flat.medium(band_um=(1.45, 1.65))
    assert m.lorentz is None
    assert m.permittivity == pytest.approx(4.0)


def test_anomalous_slope_falls_back_to_constant_with_warning():
    # n RISING with wavelength = dn/domega < 0: impossible for a passive
    # lossless pole, so the fit degenerates to the band-mean constant
    wl = np.linspace(1.4, 1.7, 31)
    anom = Material.from_nk_data(
        "anom", wavelength_um=wl, n=1.5 + 0.1 * (wl - 1.4)
    )
    with pytest.warns(UserWarning, match="no feasible"):
        fit = anom.lorentz_fit((1.45, 1.65))
    assert fit.delta_eps == 0.0
    assert fit.resonance_frequency_hz is None
    assert fit.medium.lorentz is None


def test_band_fit_carries_absorption_as_conductivity():
    wl = np.linspace(1.5, 1.6, 11)
    n = 2.0 - 0.05 * (wl - 1.5)  # normal dispersion + uniform loss
    lossy = Material.from_nk_data("lossy", wavelength_um=wl, n=n, k=np.full(11, 1e-4))
    fit = lossy.lorentz_fit((1.5, 1.6))
    assert fit.conductivity_s_per_m > 0.0
    assert fit.medium.conductivity_s_per_m == fit.conductivity_s_per_m


# ---------------------------------------------------------------------------
# simulation round-trip
# ---------------------------------------------------------------------------


def test_dispersive_material_roundtrips_on_the_wire():
    med = materials.cSi.medium(band_um=TELECOM_BAND)
    sim = ph.Simulation(
        size_um=(4, 4, 4),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=10),
        structures=[
            ph.Structure(
                geometry=ph.Box(center_um=(2, 2, 2), size_um=(2, 0.5, 0.22)),
                medium=med,
            )
        ],
        sources=[
            ph.PointDipole(
                center_um=(1, 2, 2),
                polarization="Ez",
                source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=6e12),
            )
        ],
    )
    again = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert again.to_wire_dict() == sim.to_wire_dict()
    pole = again.structures[0].medium.lorentz
    assert pole is not None
    assert pole.resonance_frequency_hz == med.lorentz.resonance_frequency_hz


# ---------------------------------------------------------------------------
# user measured-data path
# ---------------------------------------------------------------------------


def test_from_nk_data_range_is_table_span():
    wl = np.linspace(1.2, 1.8, 13)
    mat = Material.from_nk_data("mine", wavelength_um=wl, n=np.linspace(2.1, 2.0, 13))
    assert mat.valid_range_um == (1.2, 1.8)
    with pytest.raises(ValueError):
        mat.n(1.0)
    # interpolation hits the table nodes exactly
    assert mat.n(1.2) == pytest.approx(2.1)
    assert mat.n(1.8) == pytest.approx(2.0)


def test_tabulated_requires_ascending_and_matching_lengths():
    with pytest.raises(ValueError, match="ascending"):
        Material.from_nk_data("bad", wavelength_um=[1.5, 1.4], n=[2.0, 2.0])
    with pytest.raises(ValueError, match="match"):
        Material.from_nk_data("bad", wavelength_um=[1.4, 1.5], n=[2.0])
    with pytest.raises(ValueError, match=">= 2"):
        Material.from_nk_data("bad", wavelength_um=[1.5], n=[2.0])


def _real_solver():
    try:
        return ph.find_solver()
    except ph.SolverRunError:
        return None


@pytest.mark.skipif(
    _real_solver() is None,
    reason="no phsolver binary found (build the engine first)",
)
def test_integration_fresnel_slab_dispersive_cSi_band_center(tmp_path):
    """Run the library's fitted cSi pole through the real §19 engine path.

    The unit gates above pin the fit across the full band. This integration
    gate uses the band center, where a finite DFT of the lossless
    (zero-linewidth) fitted pole is well conditioned. The damped,
    multi-frequency engine gate is ``validation/test_tier3_lorentz.py``.
    """
    d = 0.5  # slab thickness um
    lams = (1.55,)
    freqs = tuple(C0 / (lam * 1e-6) for lam in lams)

    def build(structures):
        return ph.Simulation(
            size_um=(0.2, 0.2, 6.0),
            grid=ph.UniformGridSpec(dl_um=0.02),
            # A normalized slab/reference ratio requires the same DFT window.
            # Disable auto-shutoff so the empty reference cannot stop early.
            run=ph.RunSpec(run_time_s=3e-13, shutoff=0),
            # subpixel unset => the dispersive default (off) applies.
            boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
            structures=structures,
            sources=[
                ph.PlaneWave(
                    axis="z", direction="+", position_um=1.0, polarization="Ex",
                    source_time=ph.GaussianPulse(
                        freq0_hz=C0 / 1.55e-6, fwidth_hz=3.0e13
                    ),
                )
            ],
            monitors=[
                ph.FluxMonitor(name="T", axis="z", position_um=5.0, freqs_hz=freqs)
            ],
        )

    slab = ph.Structure(
        geometry=ph.Box(center_um=(0.1, 0.1, 3.0), size_um=(10.0, 10.0, d)),
        medium=materials.cSi.medium(band_um=(1.5, 1.6)),
    )
    flux_slab = np.asarray(
        ph.run_local(build([slab]), output_dir=tmp_path / "slab")["T"]
    ).ravel()
    flux_ref = np.asarray(
        ph.run_local(build([]), output_dir=tmp_path / "ref")["T"]
    ).ravel()
    T_fdtd = flux_slab / flux_ref

    for lam, t_num in zip(lams, T_fdtd):
        n = float(materials.cSi.n(lam))
        r2 = ((n - 1.0) / (n + 1.0)) ** 2
        delta = 2.0 * math.pi * n * d / lam
        t = (4.0 * n / (1.0 + n) ** 2) * np.exp(1j * delta) / (
            1.0 - r2 * np.exp(2j * delta)
        )
        t_analytic = abs(t) ** 2
        # a few percent: Yee dispersion at 22 points/lambda-in-Si dominates
        assert t_num == pytest.approx(t_analytic, rel=0.06)


def test_measured_data_end_to_end_fit():
    """A realistic user path: normally-dispersive measured film -> dispersive
    Medium whose fit reproduces the measurement."""
    wl = np.linspace(1.45, 1.65, 21)
    n_meas = 1.98 - 0.03 * (wl - 1.55)  # ~PECVD SiN-like
    film = Material.from_nk_data(
        "PECVD_SiN", wavelength_um=wl, n=n_meas, reference="fab ellipsometry"
    )
    fit = film.lorentz_fit((1.5, 1.6))
    assert fit.pole is not None
    assert fit.max_abs_n_error < 1e-4
    check = np.linspace(1.5, 1.6, 7)
    assert np.max(np.abs(fit.n_model(check) - film.n(check))) < 1e-4
