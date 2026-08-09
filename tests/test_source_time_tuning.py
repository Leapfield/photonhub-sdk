"""GaussianPulse settling diagnostics and band -> pulse tuning (for_band).

Settling floor is engine ``source_end_time`` (NUMERICS.md §7): the run cannot
auto-shutoff before ``(offset+8)*tau``, ``tau = 1/(2 pi fwidth)``. These tests
pin the client's diagnostics to that closed form and check that ``for_band``
trades band coverage against settling correctly and stays clear of the
over-broadband DC instability.
"""

import math

import pytest

import photonhub as ph

# Spec constant (engine kC0 / grid.py / cost.py); the client must match it.
_C0 = 2.99792458e8


def _tau(fwidth_hz: float) -> float:
    return 1.0 / (2.0 * math.pi * fwidth_hz)


class TestSettlingDiagnostics:
    def test_tau_t0_and_source_end_match_engine_closed_form(self):
        p = ph.GaussianPulse(freq0_hz=2.0e14, fwidth_hz=2.0e13, offset=5.0)
        tau = _tau(2.0e13)
        assert p.tau_s == pytest.approx(tau, rel=1e-15)
        assert p.t0_s == pytest.approx(5.0 * tau, rel=1e-15)
        # engine source_time.h: t0 + 8*tau == (offset + 8)*tau
        assert p.source_end_time_s == pytest.approx((5.0 + 8.0) * tau, rel=1e-15)

    def test_source_end_honours_offset(self):
        tau = _tau(2.0e13)
        p = ph.GaussianPulse(freq0_hz=2.0e14, fwidth_hz=2.0e13, offset=3.0)
        assert p.source_end_time_s == pytest.approx((3.0 + 8.0) * tau, rel=1e-15)

    def test_settling_is_inversely_proportional_to_fwidth(self):
        narrow = ph.GaussianPulse(freq0_hz=2.0e14, fwidth_hz=1.0e13)
        broad = ph.GaussianPulse(freq0_hz=2.0e14, fwidth_hz=2.0e13)
        # doubling fwidth halves the settling floor
        assert broad.source_end_time_s == pytest.approx(
            narrow.source_end_time_s / 2.0, rel=1e-15
        )

    def test_spectral_amplitude_envelope(self):
        f0, fwidth = 2.0e14, 2.0e13
        p = ph.GaussianPulse(freq0_hz=f0, fwidth_hz=fwidth)
        assert p.spectral_amplitude(f0) == pytest.approx(1.0)
        # fwidth IS the spectral std dev: edge at 1 sigma -> exp(-1/2)
        assert p.spectral_amplitude(f0 + fwidth) == pytest.approx(math.exp(-0.5))
        assert p.spectral_amplitude(f0 - fwidth) == pytest.approx(math.exp(-0.5))
        assert p.spectral_amplitude(f0 + 2 * fwidth) == pytest.approx(math.exp(-2.0))

    def test_dc_amplitude_is_the_instability_tell(self):
        # |S(0)|/|S(f0)| = 2|cos(phase)| * exp(-(f0/fw)^2/2): at f=0 BOTH
        # sidebands of the analytic spectrum (engine gaussian_pulse_spectrum)
        # contribute equally — the +f0 envelope alone under-reports DC by
        # 2|cos(phase)| (2x at the default phase=0).
        # freq0/fwidth = 10 sigma to DC -> utterly negligible (stable)
        safe = ph.GaussianPulse(freq0_hz=2.0e14, fwidth_hz=2.0e13)
        assert safe.dc_amplitude == pytest.approx(2.0 * math.exp(-0.5 * 10.0**2))
        assert safe.dc_amplitude < 1e-20
        # freq0/fwidth = 2 sigma to DC -> ~27% on DC (the documented blow-up)
        risky = ph.GaussianPulse(freq0_hz=2.0e14, fwidth_hz=1.0e14)
        assert risky.dc_amplitude == pytest.approx(2.0 * math.exp(-2.0))

    def test_dc_amplitude_matches_engine_spectrum_ratio(self):
        # Pin dc_amplitude to |S(0)|/|S(f0)| of the engine's closed-form
        # two-sideband spectrum (source_time.h gaussian_pulse_spectrum),
        # re-derived here independently, across carrier phases.
        def spectrum_abs(p, f):
            tau = 1.0 / (2.0 * math.pi * p.fwidth_hz)
            dm, dp = f - p.freq0_hz, f + p.freq0_hz
            g = lambda d: math.exp(-2.0 * math.pi**2 * tau**2 * d * d)
            # |e^{-i phase} g(dm) + e^{+i phase} g(dp)| (prefactor/phase ramp
            # cancel in the ratio)
            re = math.cos(p.phase) * (g(dm) + g(dp))
            im = math.sin(p.phase) * (g(dp) - g(dm))
            return math.hypot(re, im)

        # f0/fw = 4: the neglected -f0 sideband AT +f0 is exp(-2*(f0/fw)^2)
        # ~ 1.3e-14 of the denominator, far inside the rel tolerance (the
        # diagnostic's closed form drops it deliberately).
        for phase in (0.0, 0.25, math.pi / 3, 2.0):
            p = ph.GaussianPulse(freq0_hz=2.0e14, fwidth_hz=0.5e14, phase=phase)
            expected = spectrum_abs(p, 0.0) / spectrum_abs(p, p.freq0_hz)
            assert p.dc_amplitude == pytest.approx(expected, rel=1e-10), phase
        # phase = pi/2: the two sidebands cancel exactly at DC.
        quad = ph.GaussianPulse(freq0_hz=2.0e14, fwidth_hz=1.0e14,
                                phase=math.pi / 2)
        assert quad.dc_amplitude == pytest.approx(0.0, abs=1e-16)


class TestForBandFromFreqs:
    def test_centre_and_width_cover_band_at_one_sigma(self):
        p = ph.GaussianPulse.for_band(freqs_hz=[1.9e14, 2.0e14, 2.1e14])
        assert p.freq0_hz == pytest.approx(2.0e14)
        # band_sigmas=1 default -> fwidth == half-band; edges sit at 1 sigma
        assert p.fwidth_hz == pytest.approx(1.0e13)
        assert p.spectral_amplitude(1.9e14) == pytest.approx(math.exp(-0.5))
        assert p.spectral_amplitude(2.1e14) == pytest.approx(math.exp(-0.5))

    def test_only_band_extremes_matter(self):
        sparse = ph.GaussianPulse.for_band(freqs_hz=[1.9e14, 2.1e14])
        dense = ph.GaussianPulse.for_band(
            freqs_hz=[1.9e14, 1.95e14, 2.0e14, 2.05e14, 2.1e14]
        )
        assert sparse.freq0_hz == pytest.approx(dense.freq0_hz)
        assert sparse.fwidth_hz == pytest.approx(dense.fwidth_hz)

    def test_band_sigmas_trades_coverage_for_settling(self):
        band = [1.9e14, 2.1e14]
        wide = ph.GaussianPulse.for_band(freqs_hz=band, band_sigmas=0.5)
        mid = ph.GaussianPulse.for_band(freqs_hz=band, band_sigmas=1.0)
        tight = ph.GaussianPulse.for_band(freqs_hz=band, band_sigmas=2.0)
        # smaller band_sigmas -> broader pulse -> shorter settling
        assert wide.fwidth_hz > mid.fwidth_hz > tight.fwidth_hz
        assert wide.source_end_time_s < mid.source_end_time_s < tight.source_end_time_s
        # a broader pulse keeps the band nearer the spectral peak (flatter
        # coverage); a narrow pulse rolls the edges off
        edge = 2.1e14
        assert (
            wide.spectral_amplitude(edge)
            > mid.spectral_amplitude(edge)
            > tight.spectral_amplitude(edge)
        )
        assert mid.spectral_amplitude(edge) == pytest.approx(math.exp(-0.5))

    def test_explicit_freq0_uses_further_edge(self):
        # off-centre freq0: fwidth covers the FURTHER edge at band_sigmas
        p = ph.GaussianPulse.for_band(
            freqs_hz=[1.9e14, 2.1e14], freq0_hz=1.95e14, band_sigmas=1.0
        )
        assert p.freq0_hz == pytest.approx(1.95e14)
        half = max(1.95e14 - 1.9e14, 2.1e14 - 1.95e14)  # = 1.5e13
        assert p.fwidth_hz == pytest.approx(half)

    def test_offset_and_phase_pass_through(self):
        p = ph.GaussianPulse.for_band(
            freqs_hz=[1.9e14, 2.1e14], offset=4.0, phase=0.25
        )
        assert p.offset == 4.0
        assert p.phase == 0.25

    def test_result_is_a_valid_frozen_wire_model(self):
        p = ph.GaussianPulse.for_band(freqs_hz=[1.9e14, 2.1e14])
        # round-trips through the wire format
        assert ph.GaussianPulse.model_validate(p.model_dump()) == p
        with pytest.raises(Exception):
            p.fwidth_hz = 1.0  # frozen


class TestForBandFromWavelengths:
    def test_wavelengths_equivalent_to_freqs(self):
        lam = [1.5, 1.6]  # microns
        fhi = _C0 / (1.5e-6)
        flo = _C0 / (1.6e-6)
        from_lam = ph.GaussianPulse.for_band(wavelengths_um=lam)
        from_f = ph.GaussianPulse.for_band(freqs_hz=[flo, fhi])
        assert from_lam.freq0_hz == pytest.approx(from_f.freq0_hz)
        assert from_lam.fwidth_hz == pytest.approx(from_f.fwidth_hz)
        assert from_lam.freq0_hz == pytest.approx((flo + fhi) / 2.0)


class TestForBandStability:
    def test_wide_band_clamps_to_dc_safe_fwidth_with_warning(self):
        # an octave-spanning band wants fwidth = half-band = 1e14 (DC at 2 sigma)
        with pytest.warns(UserWarning, match="clamping"):
            p = ph.GaussianPulse.for_band(freqs_hz=[1.0e14, 3.0e14])
        # clamped to freq0 / min_dc_sigmas = 2e14 / 4; DC weight is the
        # two-sideband 2*exp(-8) ~ 6.7e-4 (phase=0), still comfortably small.
        assert p.fwidth_hz == pytest.approx(2.0e14 / 4.0)
        assert p.dc_amplitude == pytest.approx(2.0 * math.exp(-0.5 * 4.0**2))

    def test_min_dc_sigmas_controls_the_cap(self):
        with pytest.warns(UserWarning):
            p = ph.GaussianPulse.for_band(freqs_hz=[1.0e14, 3.0e14], min_dc_sigmas=6.0)
        assert p.fwidth_hz == pytest.approx(2.0e14 / 6.0)

    def test_narrow_band_does_not_warn_or_clamp(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            p = ph.GaussianPulse.for_band(freqs_hz=[1.9e14, 2.1e14])
        assert p.fwidth_hz == pytest.approx(1.0e13)  # not clamped
        assert p.dc_amplitude < 1e-20  # comfortably DC-safe


class TestForBandErrors:
    def test_requires_exactly_one_band_spec(self):
        with pytest.raises(ValueError, match="exactly one"):
            ph.GaussianPulse.for_band()
        with pytest.raises(ValueError, match="exactly one"):
            ph.GaussianPulse.for_band(
                freqs_hz=[1.9e14, 2.1e14], wavelengths_um=[1.5, 1.6]
            )

    def test_degenerate_band_rejected(self):
        with pytest.raises(ValueError, match="non-degenerate"):
            ph.GaussianPulse.for_band(freqs_hz=[2.0e14, 2.0e14])
        with pytest.raises(ValueError, match="non-degenerate"):
            ph.GaussianPulse.for_band(freqs_hz=[2.0e14])

    def test_non_positive_inputs_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ph.GaussianPulse.for_band(freqs_hz=[-1.0, 2.0e14])
        with pytest.raises(ValueError, match="positive"):
            ph.GaussianPulse.for_band(wavelengths_um=[0.0, 1.5])

    def test_bad_band_sigmas_and_min_dc_sigmas_rejected(self):
        with pytest.raises(ValueError, match="band_sigmas"):
            ph.GaussianPulse.for_band(freqs_hz=[1.9e14, 2.1e14], band_sigmas=0.0)
        with pytest.raises(ValueError, match="min_dc_sigmas"):
            ph.GaussianPulse.for_band(freqs_hz=[1.9e14, 2.1e14], min_dc_sigmas=-1.0)

    def test_freq0_outside_band_rejected(self):
        with pytest.raises(ValueError, match="inside the band"):
            ph.GaussianPulse.for_band(freqs_hz=[1.9e14, 2.1e14], freq0_hz=3.0e14)
