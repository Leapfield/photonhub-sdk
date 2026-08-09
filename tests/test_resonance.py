"""Tests for the resonance / Q-factor extraction plugin (FDM harmonic inversion).

Validation ladder (no FDTD engine required -- pure post-processing):
  * synthetic decaying exponentials with analytic (f, Q) ground truth;
  * agreement with an independent matrix-pencil / ESPRIT oracle;
  * algorithm parity against Tidy3D's ResonanceFinder on an identical signal
    (skipped if tidy3d is not importable).
"""

import numpy as np
import pytest
import xarray as xr

from photonhub.plugins import ResonanceFinder, select_resonances
from photonhub.plugins.resonance import _matrix_pencil_poles


# -- synthetic signal helpers ----------------------------------------------


def _decaying_tone(t, freq, alpha, amp=1.0, phase=0.0):
    """One complex decaying exponential a e^{i phi} e^{(-2pi i f - alpha) t}."""
    return amp * np.exp(1j * phase) * np.exp((-2j * np.pi * freq - alpha) * t)


def _two_tone():
    """Two well-separated complex modes; mirrors Tidy3D's docstring example.

    Returns (signal, dt, modes) with modes = [(freq, alpha, Q, amp), ...].
    """
    dt = 1.0
    t = np.arange(8000) * dt
    modes = [
        (0.10, 0.002, np.pi * 0.10 / 0.002, 2.0),
        (0.20, 0.0005, np.pi * 0.20 / 0.0005, 3.0),
    ]
    sig = _decaying_tone(t, 0.10, 0.002, amp=2.0) + _decaying_tone(
        t, 0.20, 0.0005, amp=3.0, phase=np.pi / 2
    )
    return sig, dt, modes


def _match(modes_ds, freq):
    """The row of a resonance Dataset nearest `freq`, as a plain dict."""
    i = int(np.argmin(np.abs(modes_ds.coords["freq"].values - freq)))
    row = modes_ds.isel(freq=i)
    return {k: float(row[k].values) for k in row.data_vars} | {
        "freq": float(row.coords["freq"].values)
    }


# -- Tier 0: synthetic analytic ground truth --------------------------------


def test_two_tone_recovers_freq_and_q():
    sig, dt, modes = _two_tone()
    rf = ResonanceFinder(freq_window=(0.05, 0.25), init_num_freqs=120)
    found = select_resonances(rf.run_raw_signal(sig, dt), min_amplitude=0.1)

    assert found.sizes["freq"] >= 2
    for freq, alpha, q, amp in modes:
        got = _match(found, freq)
        assert abs(got["freq"] - freq) / freq < 1e-4
        assert abs(got["decay"] - alpha) / alpha < 5e-3
        assert abs(got["Q"] - q) / q < 5e-3
        assert abs(got["amplitude"] - amp) / amp < 1e-2


def test_real_valued_single_tone():
    # A real FDTD probe: e^{-alpha t} cos(2 pi f t) = conjugate pair at +/- f.
    dt = 1.0
    t = np.arange(6000) * dt
    freq, alpha = 0.15, 0.001
    sig = np.exp(-alpha * t) * np.cos(2 * np.pi * freq * t)  # real
    assert np.isrealobj(sig)

    rf = ResonanceFinder(freq_window=(0.10, 0.20), init_num_freqs=80)
    found = select_resonances(
        rf.run_raw_signal(sig, dt), freq_window=(0.05, 0.5), min_amplitude=0.05
    )
    got = _match(found, freq)
    assert abs(got["freq"] - freq) / freq < 1e-4
    assert abs(got["Q"] - np.pi * freq / alpha) / (np.pi * freq / alpha) < 1e-2


def test_high_q_undecayed_signal():
    # Q ~ 4.7e4; the signal is nowhere near fully decayed over the window,
    # which is exactly where FDM beats a spectral-FWHM fit.
    dt = 1.0
    t = np.arange(5000) * dt
    freq, alpha = 0.3, 1e-5
    q_true = np.pi * freq / alpha
    sig = _decaying_tone(t, freq, alpha, amp=1.0)
    assert np.exp(-alpha * t[-1]) > 0.9  # < 10% decayed

    rf = ResonanceFinder(freq_window=(0.25, 0.35), init_num_freqs=60)
    found = select_resonances(rf.run_raw_signal(sig, dt), min_amplitude=0.1)
    got = _match(found, freq)
    assert abs(got["freq"] - freq) / freq < 1e-5
    assert abs(got["Q"] - q_true) / q_true < 0.05


# -- independent oracle agreement -------------------------------------------


def test_matrix_pencil_oracle_agrees_with_fdm():
    dt = 1.0
    t = np.arange(2000) * dt
    sig = _decaying_tone(t, 0.10, 0.002, amp=2.0) + _decaying_tone(
        t, 0.20, 0.0005, amp=3.0
    )
    rf = ResonanceFinder(freq_window=(0.05, 0.25), init_num_freqs=80)
    fdm = select_resonances(rf.run_raw_signal(sig, dt), min_amplitude=0.1)

    pen_f, pen_a = _matrix_pencil_poles(sig, dt, n_poles=2)
    # Keep the physical (positive-frequency, decaying) pencil poles.
    keep = (pen_f > 0) & (pen_a > 0)
    pen_f, pen_a = pen_f[keep], pen_a[keep]

    for freq in (0.10, 0.20):
        i = int(np.argmin(np.abs(pen_f - freq)))
        assert abs(pen_f[i] - freq) / freq < 1e-3
        got = _match(fdm, freq)
        # Two independent algorithms agree on the same poles.
        assert abs(got["freq"] - pen_f[i]) / freq < 1e-3
        assert abs(got["decay"] - pen_a[i]) / pen_a[i] < 5e-2


# -- algorithm parity vs the Tidy3D reference (free, local) -----------------


def test_tidy3d_algorithm_parity():
    td = pytest.importorskip("tidy3d")
    from tidy3d.plugins.resonance import ResonanceFinder as TdResonanceFinder

    sig, dt, _ = _two_tone()
    window = (0.05, 0.25)
    nfreqs = 100

    ours = ResonanceFinder(freq_window=window, init_num_freqs=nfreqs).run_raw_signal(
        sig, dt
    )
    theirs = TdResonanceFinder(
        freq_window=window, init_num_freqs=nfreqs
    ).run_raw_signal(signal=sig, time_step=dt)

    # Same FDM formulation + same scipy eig => the dominant modes must agree to
    # near machine precision.
    for freq in (0.10, 0.20):
        a = _match(ours, freq)
        b = _match(theirs, freq)
        assert abs(a["freq"] - b["freq"]) <= 1e-6 * abs(b["freq"])
        assert abs(a["Q"] - b["Q"]) <= 1e-5 * abs(b["Q"])
        assert abs(a["amplitude"] - b["amplitude"]) <= 1e-5 * abs(b["amplitude"])


# -- ingestion from FieldTimeMonitor-shaped xarray --------------------------


def _probe_dataarray(sig, dt, component="Ez"):
    t = np.arange(len(sig)) * dt
    return xr.DataArray(
        np.asarray(sig).reshape(-1, 1),
        dims=("t", "component"),
        coords={"t": t, "component": [component]},
        name="probe",
    )


def test_run_time_series_from_dataarray():
    dt = 1.0
    t = np.arange(6000) * dt
    freq, alpha = 0.15, 0.001
    sig = np.exp(-alpha * t) * np.cos(2 * np.pi * freq * t)
    da = _probe_dataarray(sig, dt)

    rf = ResonanceFinder(freq_window=(0.10, 0.20), init_num_freqs=80)
    found = select_resonances(rf.run_time_series(da), min_amplitude=0.05)
    assert abs(_match(found, freq)["freq"] - freq) / freq < 1e-4


def test_run_from_simdata_mapping():
    # A SimulationData stand-in: any mapping name -> DataArray.
    dt = 1.0
    t = np.arange(6000) * dt
    freq, alpha = 0.15, 0.001
    sig = np.exp(-alpha * t) * np.cos(2 * np.pi * freq * t)
    sim_data = {"probe": _probe_dataarray(sig, dt)}

    rf = ResonanceFinder(freq_window=(0.10, 0.20), init_num_freqs=80)
    found = select_resonances(rf.run(sim_data, "probe"), min_amplitude=0.05)
    assert abs(_match(found, freq)["freq"] - freq) / freq < 1e-4


def test_run_derives_dt_with_interval_steps():
    # interval_steps > 1 => 't' spaced by interval_steps * dt_engine; dt must be
    # read off the coordinate, not assumed to be one step.
    dt_eff = 3.0
    t = np.arange(4000) * dt_eff
    freq, alpha = 0.05, 5e-4
    sig = _decaying_tone(t, freq, alpha, amp=1.0)
    da = _probe_dataarray(sig, dt_eff)

    rf = ResonanceFinder(freq_window=(0.03, 0.07), init_num_freqs=60)
    found = select_resonances(rf.run_time_series(da), min_amplitude=0.1)
    assert abs(_match(found, freq)["freq"] - freq) / freq < 1e-3


# -- selection / filtering --------------------------------------------------


def test_select_resonances_filters_and_ranks():
    sig, dt, _ = _two_tone()
    raw = ResonanceFinder(freq_window=(0.05, 0.25), init_num_freqs=120).run_raw_signal(
        sig, dt
    )
    # Raw output keeps spurious low-amplitude / out-of-window modes.
    physical = select_resonances(
        raw, freq_window=(0.05, 0.25), min_amplitude=0.1, sort_by="Q"
    )
    assert physical.sizes["freq"] <= raw.sizes["freq"]
    assert physical.sizes["freq"] >= 2
    # All survivors are physical and in-window.
    assert np.all(physical["decay"].values > 0)
    f = physical.coords["freq"].values
    assert np.all((f >= 0.05) & (f <= 0.25))
    # Ranked by Q descending.
    q = physical["Q"].values
    assert np.all(np.diff(q) <= 1e-9)


# -- input validation -------------------------------------------------------


def test_freq_window_must_be_ordered():
    with pytest.raises(ValueError, match="f_max >= f_min"):
        ResonanceFinder(freq_window=(0.3, 0.1))


def test_signal_must_be_1d():
    rf = ResonanceFinder(freq_window=(0.1, 0.2))
    with pytest.raises(ValueError, match="1-D"):
        rf.run_raw_signal(np.zeros((10, 2)), 1.0)


def test_signal_too_short():
    rf = ResonanceFinder(freq_window=(0.1, 0.2))
    with pytest.raises(ValueError, match="samples"):
        rf.run_raw_signal(np.ones(4), 1.0)


def test_nyquist_warning():
    rf = ResonanceFinder(freq_window=(0.1, 0.7), init_num_freqs=20)
    sig = _decaying_tone(np.arange(500) * 1.0, 0.1, 0.01)
    with pytest.warns(UserWarning, match="Nyquist"):
        rf.run_raw_signal(sig, 1.0)


def test_nonuniform_time_coordinate_rejected():
    rf = ResonanceFinder(freq_window=(0.1, 0.2))
    t = np.sort(np.r_[np.arange(0, 100), 100.5])  # one irregular step
    sig = _decaying_tone(t, 0.15, 0.001)
    da = xr.DataArray(
        sig.reshape(-1, 1), dims=("t", "component"),
        coords={"t": t, "component": ["Ez"]},
    )
    with pytest.raises(ValueError, match="uniformly spaced"):
        rf.run_time_series(da)


# -- component selection / ingestion guards (review fixes) ------------------


def test_select_components_e_then_h_fallback():
    from photonhub.plugins.resonance import _select_components

    assert _select_components(["Ez", "Hy"], None) == ["Ez"]  # E preferred, never E+H
    assert _select_components(["Ex", "Ey", "Ez"], None) == ["Ex", "Ey", "Ez"]
    assert _select_components(["Hx", "Hy"], None) == ["Hx", "Hy"]  # H only if no E
    assert _select_components(["Ez", "Hy"], ["Hy"]) == ["Hy"]  # explicit override
    assert _select_components([], None) is None


def test_run_and_run_time_series_agree_on_mixed_eh():
    # A monitor recording both E and H must use ONLY E (Tidy3D convention), and
    # run() / run_time_series() must agree (they used to diverge).
    dt = 1.0
    t = np.arange(6000) * dt
    fE, aE = 0.15, 0.001
    ez = np.exp(-aE * t) * np.cos(2 * np.pi * fE * t)
    hy = 0.3 * np.exp(-0.05 * t) * np.cos(2 * np.pi * 0.30 * t)  # different H content
    da = xr.DataArray(
        np.stack([ez, hy], axis=1), dims=("t", "component"),
        coords={"t": t, "component": ["Ez", "Hy"]},
    )
    rf = ResonanceFinder(freq_window=(0.10, 0.20), init_num_freqs=80)
    via_ts = rf.run_time_series(da)
    via_run = rf.run({"probe": da}, "probe")
    xr.testing.assert_allclose(via_ts, via_run)  # both select Ez only -> identical
    got = _match(select_resonances(via_ts, min_amplitude=0.05), fE)
    assert abs(got["freq"] - fE) / fE < 1e-4


def test_snapshot_like_array_rejected():
    t = np.arange(64) * 1.0
    snap = xr.DataArray(
        np.zeros((64, 1, 2, 2, 2)), dims=("t", "component", "z", "y", "x"),
        coords={"t": t, "component": ["Ez"]},
    )
    rf = ResonanceFinder(freq_window=(0.1, 0.2))
    with pytest.raises(ValueError, match="non-time dimensions"):
        rf.run_time_series(snap)


def test_fields_requested_but_no_component_dim_raises():
    t = np.arange(64) * 1.0
    da = xr.DataArray(np.zeros(64), dims=("t",), coords={"t": t})
    rf = ResonanceFinder(freq_window=(0.1, 0.2))
    with pytest.raises(ValueError, match=r"no 'component' dimension"):
        rf.run({"p": da}, "p", fields=["Ez"])


def test_select_resonances_rejects_reversed_window():
    sig, dt, _ = _two_tone()
    raw = ResonanceFinder(freq_window=(0.05, 0.25), init_num_freqs=60).run_raw_signal(sig, dt)
    with pytest.raises(ValueError, match="f_max >= f_min"):
        select_resonances(raw, freq_window=(0.25, 0.05))


def test_nan_q_sorts_last():
    ds = xr.Dataset(
        {
            "decay": ("freq", [1.0, 0.0, 2.0]),
            "Q": ("freq", [100.0, np.nan, 50.0]),  # zero-decay pole -> NaN Q
            "amplitude": ("freq", [1.0, 1.0, 1.0]),
            "phase": ("freq", [0.0, 0.0, 0.0]),
            "error": ("freq", [0.0, 0.0, 0.0]),
        },
        coords={"freq": [0.1, 0.2, 0.3]},
    )
    out = select_resonances(ds, require_decay=False, sort_by="Q")
    qs = out["Q"].values
    assert qs[0] == 100.0  # largest finite Q first, NaN not promoted to the top
    assert np.isnan(qs[-1])  # NaN ranked last
