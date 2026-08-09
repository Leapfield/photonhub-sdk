"""CVCS — continuously-varying-cross-section EME via mode interpolation
(``photonhub.plugins.cvcs``).

Three things are validated:

1. **Interpolation fidelity** (the premise): a mode interpolated from a few key
   planes reproduces the *true* solved mode at an intermediate cross-section, and
   the error shrinks as key planes are added.
2. **End-to-end efficiency**: for an adiabatic taper, a CVCS cascade built from a
   few solved key planes (interpolated to a fine sub-slicing) reproduces the full
   ``N``-solve staircase transmission — the ``N/K`` eigensolve saving.
3. **Multimode conversion**: with a fixed basis (no mode dropped), CVCS reproduces
   a multimode staircase's inter-mode conversion (a symmetric taper's TE0->TE2) at
   ``K << N`` solves — not just the adiabatic limit.
"""

from functools import lru_cache

import numpy as np
import pytest

from photonhub.plugins import eme
from photonhub.plugins.cvcs import cvcs_sections, interpolate_mode, interpolate_plane
from photonhub.plugins.mode_tracking import transverse_overlap

WL_UM = 1.31
DL_UM = 0.05
N_CORE, N_CLAD = 3.5, 1.444
CORE_H_UM = 0.22
WIN_W_UM, WIN_H_UM = 2.8, 1.4
W_IN, W_OUT, TAPER_LEN = 0.4, 1.1, 2.0  # adiabatic taper (CVCS's valid regime)


@lru_cache(maxsize=None)
def _modes(core_w_um, n=2):
    return eme.waveguide_section(
        wavelength_um=WL_UM, dl_um=DL_UM, core_w_um=core_w_um, core_h_um=CORE_H_UM,
        n_core=N_CORE, n_clad=N_CLAD, window_w_um=WIN_W_UM, window_h_um=WIN_H_UM,
        num_modes=n, neff_margin=0.05,
    ).modes


def _width(z_frac):
    return W_IN + (W_OUT - W_IN) * z_frac


def _staircase(n_seg):
    secs = [eme.Section(_modes(W_IN), 0.0)]
    for k in range(n_seg):
        secs.append(eme.Section(_modes(_width((k + 0.5) / n_seg)), TAPER_LEN / n_seg))
    secs.append(eme.Section(_modes(W_OUT), 0.0))
    return eme.run_eme(secs)


def _key_planes(k):
    zfs = np.linspace(0.0, 1.0, k)
    z_um = zfs * TAPER_LEN
    return [_modes(_width(zf)) for zf in zfs], z_um


# --- abrupt symmetric multimode taper (real TE0->TE2 conversion) ------------
MM_W_IN, MM_W_OUT, MM_LEN = 0.7, 1.5, 0.6


@lru_cache(maxsize=None)
def _mm_modes(core_w_um):
    return eme.waveguide_section(
        wavelength_um=WL_UM, dl_um=DL_UM, core_w_um=core_w_um, core_h_um=CORE_H_UM,
        n_core=N_CORE, n_clad=N_CLAD, window_w_um=3.4, window_h_um=WIN_H_UM,
        num_modes=3, neff_margin=0.05,
    ).modes


def _mm_width(zf):
    return MM_W_IN + (MM_W_OUT - MM_W_IN) * zf


def _mm_staircase(n_seg):
    secs = [eme.Section(_mm_modes(MM_W_IN), 0.0)]
    for k in range(n_seg):
        secs.append(eme.Section(_mm_modes(_mm_width((k + 0.5) / n_seg)), MM_LEN / n_seg))
    secs.append(eme.Section(_mm_modes(MM_W_OUT), 0.0))
    return eme.run_eme(secs)


def _mm_key_planes(k):
    zfs = np.linspace(0.0, 1.0, k)
    return [_mm_modes(_mm_width(zf)) for zf in zfs], zfs * MM_LEN


# --- interpolate_mode primitive --------------------------------------------


def test_interpolate_mode_endpoints_are_identities():
    a, b = _modes(0.5)[0], _modes(0.9)[0]
    assert interpolate_mode(a, b, 0.0) is a
    assert interpolate_mode(a, b, 1.0) is b


def test_interpolate_mode_normalized_and_neff_between():
    a, b = _modes(0.5)[0], _modes(0.9)[0]
    mid = interpolate_mode(a, b, 0.5)
    l2 = float(np.sum(np.abs(mid.ex) ** 2 + np.abs(mid.ey) ** 2))
    assert l2 == pytest.approx(1.0, abs=1e-9)  # VectorMode transverse-E convention
    assert min(a.n_eff, b.n_eff) < mid.n_eff < max(a.n_eff, b.n_eff)
    assert mid.n_eff == pytest.approx(0.5 * (a.n_eff + b.n_eff))


def test_interpolate_mode_invalid_s_raises():
    a, b = _modes(0.5)[0], _modes(0.9)[0]
    with pytest.raises(ValueError):
        interpolate_mode(a, b, 1.5)


def test_interpolate_mode_spacing_mismatch_raises():
    """Same point count but a different dl is NOT the same grid — the blend
    would silently mix fields at different physical locations."""
    from dataclasses import replace

    a = _modes(0.5)[0]
    b = replace(a, dl_x_um=a.dl_x_um * 2.0)
    with pytest.raises(ValueError, match="spacing"):
        interpolate_mode(a, b, 0.5)


def test_interpolate_plane_count_mismatch_raises():
    with pytest.raises(ValueError):
        interpolate_plane(_modes(0.5)[:1], _modes(0.9)[:2], 0.5)


# --- interpolation fidelity vs the true mode -------------------------------


def test_interpolated_mode_matches_true_mode():
    # midpoint fundamental interpolated from the taper endpoints vs the solved
    # midpoint mode (the fundamental is unambiguously mode 0 at every width).
    a, b = _modes(W_IN)[0], _modes(W_OUT)[0]
    true_mid = _modes(_width(0.5))[0]
    interp = interpolate_mode(a, b, 0.5)
    sim = abs(transverse_overlap([interp], [true_mid])[0, 0])
    assert sim > 0.95  # K=2 endpoints already reproduce the true fundamental well


def test_fidelity_improves_with_more_key_planes():
    probes = [0.35, 0.55, 0.75]  # intermediate z-fractions

    def max_err(k):
        keys, _ = _key_planes(k)
        worst = 0.0
        for zf in probes:
            pos = zf * (k - 1)
            j = min(int(pos), k - 2)
            s = pos - j
            true0 = _modes(_width(zf))[0]  # fundamental only (no tracking needed)
            interp = interpolate_mode(keys[j][0], keys[j + 1][0], s)
            worst = max(worst, 1.0 - abs(transverse_overlap([interp], [true0])[0, 0]))
        return worst

    err_coarse = max_err(2)
    err_fine = max_err(5)
    assert err_fine < 0.6 * err_coarse  # clearly converging toward the true modes


# --- end-to-end: CVCS reproduces the staircase, with K << N solves ---------


def test_cvcs_sections_structure():
    keys, z_um = _key_planes(3)
    secs = cvcs_sections(keys, z_um, n_subslices=10)
    assert len(secs) == 10 + 2  # 2 length-0 ports + interior slices
    assert secs[0].length_um == 0.0 and secs[-1].length_um == 0.0
    # interior slices each carry the per-slice propagation length
    assert all(s.length_um == pytest.approx(TAPER_LEN / 10) for s in secs[1:-1])
    # every section exposes the same (tracked) number of modes
    n_modes = len(secs[0].modes)
    assert n_modes >= 1 and all(len(s.modes) == n_modes for s in secs)
    # the ports are the solved endpoint modes (n_eff preserved)
    assert secs[0].modes[0].n_eff == pytest.approx(keys[0][0].n_eff)
    assert secs[-1].modes[0].n_eff == pytest.approx(keys[-1][0].n_eff)


def test_cvcs_reproduces_staircase_for_adiabatic_taper():
    ref = _staircase(24)
    # The fixed multimode basis carries the (near-cutoff) second mode too, so it
    # needs a few key planes for that mode to correspond across the taper.
    keys, z_um = _key_planes(5)  # 5 solves vs the 24-solve staircase
    cvcs = eme.run_eme(cvcs_sections(keys, z_um, n_subslices=24))
    assert cvcs.transmission == pytest.approx(ref.transmission, abs=2e-3)
    assert cvcs.energy_balance() == pytest.approx(1.0, abs=1e-9)
    # CVCS reflection is at/below the converged staircase (no staircasing artifact)
    assert cvcs.reflection <= ref.reflection + 1e-6


def test_cvcs_converges_with_more_key_planes():
    ref = _staircase(24).transmission
    keys3, z3 = _key_planes(3)
    keys9, z9 = _key_planes(9)
    e3 = abs(eme.run_eme(cvcs_sections(keys3, z3, n_subslices=24)).transmission - ref)
    e9 = abs(eme.run_eme(cvcs_sections(keys9, z9, n_subslices=24)).transmission - ref)
    assert e9 < e3  # more key planes -> closer to the staircase
    assert e9 < 1e-3


# --- multimode conversion (the fixed-basis payoff) -------------------------


def test_cvcs_captures_multimode_conversion():
    # Abrupt symmetric taper: real TE0->TE2 conversion (T00 ~ 0.91 << 1). The
    # fixed basis keeps the (mid-taper-born) TE2, so CVCS reproduces it at K<<N.
    ref = _mm_staircase(48)
    assert ref.transmission < 0.95  # the staircase genuinely converts ~9%
    keys, z_um = _mm_key_planes(5)
    assert all(len(p) == 3 for p in keys)  # fixed 3-mode basis, nothing dropped
    cvcs = eme.run_eme(cvcs_sections(keys, z_um, n_subslices=48))
    assert cvcs.n_modes == 3  # the conversion channel survives
    assert cvcs.transmission == pytest.approx(ref.transmission, abs=1e-2)
    assert cvcs.energy_balance() == pytest.approx(1.0, abs=1e-6)  # unitary
    assert cvcs.transmitted_power() > 0.99  # ~all power forward (tiny reflection)


# --- guards ----------------------------------------------------------------


def test_cvcs_requires_constant_mode_count():
    keys, z_um = _mm_key_planes(3)
    bad = [keys[0][:2], keys[1], keys[2]]  # first plane has fewer modes
    with pytest.raises(ValueError):
        cvcs_sections(bad, z_um, n_subslices=8)


def test_cvcs_sections_needs_two_key_planes():
    keys, z_um = _key_planes(2)
    with pytest.raises(ValueError):
        cvcs_sections(keys[:1], z_um[:1], n_subslices=8)


def test_cvcs_sections_z_must_increase():
    keys, _ = _key_planes(3)
    with pytest.raises(ValueError):
        cvcs_sections(keys, [0.0, 2.0, 1.0], n_subslices=8)


def test_cvcs_sections_z_length_must_match():
    keys, _ = _key_planes(3)
    with pytest.raises(ValueError):
        cvcs_sections(keys, [0.0, 1.0], n_subslices=8)
