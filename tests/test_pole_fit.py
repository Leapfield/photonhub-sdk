"""Material.pole_fit — multi-pole (Lorentz + Drude) complex-eps fitting.

Deterministic round-trip pins: data synthesized from a KNOWN pole model must
be recovered essentially exactly (the fit residual is the only free part —
no RNG anywhere, so these are stable). Passivity of the emitted Medium is a
hard invariant.
"""

import math

import numpy as np
import pytest

from photonhub.materials import Material

C0 = 299792458.0


def _synth(lam_um, eps_inf, lorentz=(), drude=()):
    w = 2 * np.pi * C0 / (np.asarray(lam_um) * 1e-6)
    eps = np.full_like(w, eps_inf, dtype=np.complex128)
    for (f0, de, g) in lorentz:
        w0, gg = 2 * np.pi * f0, 2 * np.pi * g
        eps += de * w0**2 / (w0**2 - w**2 - 1j * gg * w)
    for (fp, g) in drude:
        wp, gg = 2 * np.pi * fp, 2 * np.pi * g
        eps += -(wp**2) / (w**2 + 1j * gg * w)
    return eps


def _material(name, lam, eps):
    nk = np.sqrt(eps)
    nk = np.where(nk.imag < 0, -nk, nk)
    return Material.from_nk_data(name, wavelength_um=lam, n=nk.real,
                                 k=nk.imag, reference="synthetic")


def test_two_lorentz_roundtrip():
    lam = np.linspace(1.0, 1.7, 240)
    poles = [(6.0e14, 1.5, 4.0e13), (1.2e15, 0.8, 6.0e13)]
    mat = _material("TwoPole", lam, _synth(lam, 2.1, lorentz=poles))
    fit = mat.pole_fit(band_um=(1.05, 1.65), n_lorentz=2)
    assert fit.max_rel_eps_error < 1e-3
    assert not fit.drude
    m = fit.medium
    assert m.is_dispersive and len(m.poles) >= 1
    for p in m.poles:
        assert p.delta_eps >= 0 and p.linewidth_hz >= 0


def test_drude_metal_roundtrip_and_negative_band():
    lam = np.linspace(1.0, 1.6, 200)
    mat = _material(
        "FakeAu", lam,
        _synth(lam, 1.8, lorentz=[(6.0e14, 1.2, 8.0e13)],
               drude=[(2.1e15, 1.2e13)]))
    fit = mat.pole_fit(band_um=(1.05, 1.55), n_lorentz=1, drude=True)
    assert fit.max_rel_eps_error < 1e-3
    assert len(fit.drude) == 1
    assert fit.drude[0].plasma_frequency_hz == pytest.approx(2.1e15, rel=0.05)
    # the emitted Medium reproduces the metallic (negative) band eps
    eps_mid = fit.medium.permittivity_at_hz(C0 / 1.3e-6)
    assert eps_mid == pytest.approx(float(
        _synth([1.3], 1.8, lorentz=[(6.0e14, 1.2, 8.0e13)],
               drude=[(2.1e15, 1.2e13)]).real[0]), rel=2e-3)
    assert eps_mid < -10


def test_absorbing_dielectric_fits_imaginary_part():
    """Unlike lorentz_fit (lossless pole + band-centre sigma), pole_fit's
    damped poles carry Im eps across the whole band."""
    lam = np.linspace(1.2, 1.5, 150)
    poles = [(2.6e14, 0.9, 3.0e13)]        # resonance just above the band
    mat = _material("Lossy", lam, _synth(lam, 2.25, lorentz=poles))
    fit = mat.pole_fit(band_um=(1.22, 1.48), n_lorentz=1)
    lam_q = np.linspace(1.25, 1.45, 40)
    got = fit.eps_model(lam_q)
    want = _synth(lam_q, 2.25, lorentz=poles)
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-3
    assert np.all(got.imag >= -1e-9)       # passive


def test_stability_helper_and_budget_guards():
    lam = np.linspace(1.0, 1.6, 100)
    mat = _material("D", lam, _synth(lam, 2.0, lorentz=[(6.0e14, 1.0, 5e13)]))
    fit = mat.pole_fit(band_um=(1.05, 1.55), n_lorentz=1)
    # omega0*dt at a photonics grid is far below the §19.4 bound
    assert 0 < fit.omega0_dt(dl_um=0.02) < 0.5
    with pytest.raises(ValueError, match="budget"):
        mat.pole_fit(band_um=(1.05, 1.55), n_lorentz=6, drude=True)
    with pytest.raises(ValueError, match="at least one"):
        mat.pole_fit(band_um=(1.05, 1.55), n_lorentz=0)
    with pytest.raises(ValueError, match="band"):
        mat.pole_fit(band_um=(1.5, 1.0), n_lorentz=1)
