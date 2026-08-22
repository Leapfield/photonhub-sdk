"""Metal library entries (Au/Ag/Cu Johnson & Christy, Al Rakic) — tabulated
n/k + the pole_fit path that turns them into runnable schema-1.17 media.

The fits here are DETERMINISTIC (pole_fit uses fixed ladders, no RNG), so the
error bounds are stable regression gates, calibrated with ~2x margin over the
measured values (Au 1.4 %, Ag 1.0 %, Cu 0.8 %, Al 0.7 % on these bands).
"""

import numpy as np
import pytest

from photonhub import materials

TELECOM = (1.0, 1.6)


@pytest.mark.parametrize("name", ["Au", "Ag", "Cu", "Al"])
def test_registry_and_metallic_eps(name):
    m = materials.get(name)
    assert m.name == name
    assert "CC0" in m.reference
    eps = m.eps(1.55)
    assert eps.real < -10        # all four are strongly metallic at 1550 nm
    assert eps.imag > 0          # absorbing under e^{-i omega t}


@pytest.mark.parametrize("name,band,n_lorentz,tol", [
    ("Au", TELECOM, 1, 0.03),
    ("Ag", TELECOM, 1, 0.02),
    ("Cu", TELECOM, 1, 0.02),
    ("Al", TELECOM, 2, 0.02),    # 2 poles: the 827 nm interband tail
])
def test_pole_fit_reaches_band_accuracy(name, band, n_lorentz, tol):
    fit = materials.get(name).pole_fit(band_um=band, n_lorentz=n_lorentz,
                                       drude=True)
    assert fit.max_rel_eps_error < tol, fit.max_rel_eps_error
    assert len(fit.drude) == 1
    medium = fit.medium
    assert medium.is_dispersive
    # the medium reproduces the metallic band eps sign
    f_mid = 299792458.0 / (sum(band) / 2 * 1e-6)
    assert medium.permittivity_at_hz(f_mid) < -1.0


def test_au_plasma_frequency_is_physical():
    fit = materials.Au.pole_fit(band_um=TELECOM, n_lorentz=1, drude=True)
    # literature Au Drude plasma frequency ~ 2.18e15 Hz (~9 eV)
    assert fit.drude[0].plasma_frequency_hz == pytest.approx(2.18e15, rel=0.1)


def test_frozen_and_single_pole_paths_reject_metals():
    with pytest.raises(ValueError, match="pole_fit"):
        materials.Au.medium(wavelength_um=1.55)
    with pytest.raises(ValueError, match="pole_fit"):
        materials.Au.medium(band_um=TELECOM)


def test_tabulated_values_spot_check():
    """Johnson & Christy 1972 table values, quoted digits (Au at
    lambda = 0.6595 um: n = 0.14, k = 3.697; Ag at 1.088 um: n = 0.04,
    k = 7.795 — direct rows of the embedded tables, verified against the CC0 source files)."""
    assert materials.Au.n(0.6595) == pytest.approx(0.14, abs=1e-6)
    assert materials.Au.k(0.6595) == pytest.approx(3.697, abs=1e-6)
    assert materials.Ag.n(1.088) == pytest.approx(0.04, abs=1e-6)
    assert materials.Ag.k(1.088) == pytest.approx(7.795, abs=1e-6)
