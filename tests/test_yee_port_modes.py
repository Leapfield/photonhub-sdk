"""Polarization-family multimode Yee bank contracts for modal ports."""

from types import SimpleNamespace

import numpy as np
import pytest

from photonhub.plugins import yee_mode
from photonhub.plugins.vector_modes import VectorMode


def _mode(polarization: str, n_eff: float):
    return SimpleNamespace(
        polarization=polarization,
        n_eff=n_eff,
        te_fraction=0.9 if polarization == "TE" else 0.1,
    )


def test_port_mode_bank_reuses_one_frame_solve_per_frequency(monkeypatch):
    frequencies = (190.0e12, 200.0e12)
    frames = [
        _mode("TE", 2.8),
        _mode("TM", 2.7),
        _mode("TE", 2.6),
        _mode("TM", 2.5),
    ]
    calls = []

    def fake_frames(sim, axis, plane_value_um, freqs_hz, **settings):
        calls.append((sim, axis, plane_value_um, tuple(freqs_hz), settings))
        for frequency in freqs_hz:
            yield float(frequency), frames

    monkeypatch.setattr(yee_mode, "_yee_bank_frames", fake_frames)
    sim = object()
    bank = yee_mode.solve_yee_port_mode_bank(
        sim,
        "x",
        1.25,
        frequencies,
        modes=(("TE", 0), ("TE", 1), ("TM", 0), ("TE", 0)),
        h_center_um=2.0,
        v_center_um=0.5,
        half_w_um=1.0,
        half_v_um=0.4,
        dl_um=0.05,
    )

    assert len(calls) == 1
    assert calls[0][:4] == (sim, "x", 1.25, frequencies)
    assert calls[0][4]["nmodes"] == 6
    assert list(bank) == list(frequencies)
    for per_frequency in bank.values():
        assert list(per_frequency) == [("TE", 0), ("TE", 1), ("TM", 0)]
        assert per_frequency[("TE", 0)] is frames[0]
        assert per_frequency[("TE", 1)] is frames[2]
        assert per_frequency[("TM", 0)] is frames[1]


def test_port_mode_bank_caps_auto_trials_for_highest_family_index(monkeypatch):
    frames = [_mode("TE", 3.0 - 0.01 * index) for index in range(32)]
    trial_counts = []

    def fake_frames(_sim, _axis, _plane_value_um, freqs_hz, **settings):
        trial_counts.append(settings["nmodes"])
        for frequency in freqs_hz:
            yield float(frequency), frames

    monkeypatch.setattr(yee_mode, "_yee_bank_frames", fake_frames)
    bank = yee_mode.solve_yee_port_mode_bank(
        object(), "x", 1.0, (200.0e12,),
        modes=(("TE", 31),),
        h_center_um=1.0, v_center_um=0.5,
        half_w_um=1.0, half_v_um=0.5, dl_um=0.05,
    )

    assert trial_counts == [32]
    assert bank[200.0e12][("TE", 31)] is frames[-1]


@pytest.mark.parametrize(
    "modes, num_modes, message",
    [
        ((("TE", 0), ("TM", 0)), 1, "at least 2"),
        ((("TE", 31), ("TM", 0)), None,
         "require 33 trial modes.*maximum of 32"),
    ],
)
def test_port_mode_bank_rejects_infeasible_total_trial_counts(
    modes, num_modes, message,
):
    with pytest.raises(ValueError, match=message):
        yee_mode.solve_yee_port_mode_bank(
            object(), "x", 1.0, (200.0e12,),
            modes=modes, num_modes=num_modes,
            h_center_um=1.0, v_center_um=0.5,
            half_w_um=1.0, half_v_um=0.5, dl_um=0.05,
        )


def test_port_mode_bank_interprets_te_relative_to_width_axis(monkeypatch):
    frames = [_mode("TE", 2.8), _mode("TM", 2.7)]

    def fake_frames(_sim, _axis, _plane_value_um, freqs_hz, **_settings):
        for frequency in freqs_hz:
            yield float(frequency), frames

    monkeypatch.setattr(yee_mode, "_yee_bank_frames", fake_frames)
    bank = yee_mode.solve_yee_port_mode_bank(
        object(), "x", 1.0, (200.0e12,),
        modes=(("TE", 0), ("TM", 0)),
        h_center_um=1.0, v_center_um=0.5,
        half_w_um=1.0, half_v_um=0.5, dl_um=0.05,
        # Natural x-normal axes are y,z. Making y the thickness axis means
        # physical TE (E along width z) is the solver's natural TM family.
        thickness_axis="y",
    )

    per_frequency = bank[200.0e12]
    assert per_frequency[("TE", 0)] is frames[1]
    assert per_frequency[("TM", 0)] is frames[0]


def test_port_mode_bank_tracks_channels_and_phase_across_crossing(monkeypatch):
    def vector(ex, n_eff):
        ex = np.asarray([ex], dtype=np.complex128)
        zero = np.zeros_like(ex)
        return VectorMode(
            n_eff=n_eff, n_group=None,
            ex=ex, ey=zero, ez=zero, hx=zero, hy=zero, hz=zero,
            wavelength_um=1.55, dl_x_um=0.1, dl_y_um=0.1,
        )

    a0 = vector([1.0, 0.0], 2.8)
    b0 = vector([0.0, 1.0], 2.7)
    # The n_eff ordering crosses and eigs returns A with the opposite phase.
    b1 = vector([0.0, 1.0], 2.9)
    a1 = vector([-1.0, 0.0], 2.6)

    def fake_frames(_sim, _axis, _plane_value_um, freqs_hz, **_settings):
        yield float(freqs_hz[0]), [a0, b0]
        yield float(freqs_hz[1]), [b1, a1]

    monkeypatch.setattr(yee_mode, "_yee_bank_frames", fake_frames)
    bank = yee_mode.solve_yee_port_mode_bank(
        object(), "x", 1.0, (190.0e12, 200.0e12),
        modes=(("TE", 0), ("TE", 1)),
        h_center_um=1.0, v_center_um=0.5,
        half_w_um=1.0, half_v_um=0.5, dl_um=0.1,
    )

    later = bank[200.0e12]
    assert later[("TE", 0)].n_eff == pytest.approx(2.6)
    assert later[("TE", 1)].n_eff == pytest.approx(2.9)
    assert later[("TE", 0)].ex == pytest.approx(a0.ex)
    assert later[("TE", 1)].ex == pytest.approx(b0.ex)


@pytest.mark.parametrize(
    "modes, message",
    [
        ((), "modes must be non-empty"),
        ((("HYBRID", 0),), "must be TE or TM"),
        ((("TE", -1),), "must be between 0 and 31"),
        ((("TE", True),), "must be an integer"),
        (("TE0",), r"must be \(polarization, mode_index\) pairs"),
    ],
)
def test_port_mode_bank_rejects_ambiguous_channel_ids(modes, message):
    with pytest.raises(ValueError, match=message):
        yee_mode.solve_yee_port_mode_bank(
            object(),
            "x",
            0.0,
            (200.0e12,),
            modes=modes,
            h_center_um=0.0,
            v_center_um=0.0,
            half_w_um=1.0,
            half_v_um=1.0,
            dl_um=0.05,
        )
