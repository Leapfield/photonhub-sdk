"""Advisory monitor checks — setup-time findings, result data health, and the
time-probe FFT (spec: docs/superpowers/specs/
2026-07-29-results-viz-monitor-checks-design.md)."""

import json

import numpy as np
import pytest
import xarray as xr

import photonhub as ph
from photonhub.viz import checks, service

_C = 299_792_458.0
_F0 = 193.4e12
_FWIDTH = 20e12


def _pulse():
    return ph.GaussianPulse(freq0_hz=_F0, fwidth_hz=_FWIDTH)


def _sim(monitors=(), *, size_um=(4.0, 2.0, 2.4), n_steps=2000):
    """Small uniform-grid scene: dl 0.05 µm, PML all faces (12 layers =
    0.6 µm band), one in-band dipole at the centre."""
    return ph.Simulation(
        size_um=size_um,
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=n_steps),
        sources=(ph.PointDipole(center_um=(2.0, 1.0, 1.2),
                                polarization="Ez",
                                source_time=_pulse()),),
        monitors=tuple(monitors),
    )


def _ids(report, severity=None):
    return [f["id"] for f in report["findings"]
            if severity is None or f["severity"] == severity]


def _by_id(report, id_):
    return [f for f in report["findings"] if f["id"] == id_]


def test_clean_sim_has_no_findings():
    report = checks.monitor_checks(_sim([
        ph.FluxMonitor(name="out", axis="x", position_um=3.2,
                       freqs_hz=(_F0,)),
        ph.FieldDftMonitor(name="mid", center_um=(2.0, 1.0, 1.2),
                           size_um=(4.0, 2.0, 0.0), fields=("Ex", "Ey", "Ez"),
                           freqs_hz=(_F0,)),
        ph.FieldTimeMonitor(name="probe", center_um=(2.0, 1.0, 1.2),
                            fields=("Ez",), interval_steps=1),
    ]))
    assert report["findings"] == []
    assert report["counts"] == {"error": 0, "warning": 0, "info": 0}
    assert report["monitor_count"] == 3
    assert report["num_steps"] == 2000
    assert report["dt_s"] == pytest.approx(
        0.99 * 0.05e-6 / (_C * np.sqrt(3.0)))
    assert report["output_bytes_total"] > 0


def test_no_monitors_is_an_info_finding():
    report = checks.monitor_checks(_sim([]))
    assert _ids(report) == ["no-monitors"]
    assert report["counts"]["info"] == 1


def test_flux_plane_inside_pml_band_warns():
    report = checks.monitor_checks(_sim([
        ph.FluxMonitor(name="edge", axis="x", position_um=0.3,
                       freqs_hz=(_F0,)),
    ]))
    finding = _by_id(report, "absorbing-band")[0]
    assert finding["severity"] == "warning"
    assert finding["monitor"] == "edge"
    assert finding["detail"]["axis"] == "x"
    assert finding["detail"]["boundary"] == "pml"
    # 12 layers * 0.05 µm on each face of the 4.0 µm axis.
    assert finding["detail"]["interior_um"] == pytest.approx([0.6, 3.4])


def test_dft_plane_and_time_probe_inside_band_warn():
    report = checks.monitor_checks(_sim([
        ph.FieldDftMonitor(name="deep", center_um=(2.0, 1.0, 0.3),
                           size_um=(4.0, 2.0, 0.0), fields=("Ex",),
                           freqs_hz=(_F0,)),
        ph.FieldTimeMonitor(name="corner", center_um=(0.2, 1.0, 1.2),
                            fields=("Ez",), interval_steps=1),
    ]))
    monitors = {f["monitor"]: f for f in _by_id(report, "absorbing-band")}
    assert set(monitors) == {"deep", "corner"}
    assert monitors["deep"]["detail"]["axis"] == "z"
    assert monitors["corner"]["detail"]["axis"] == "x"


def test_modal_port_plane_is_exempt_from_band_advisory():
    # The model hard-validates port planes; the advisory must not duplicate.
    sim = _sim([
        ph.FieldDftMonitor(name="mid", center_um=(2.0, 1.0, 1.2),
                           size_um=(0.0, 1.0, 1.0),
                           fields=("Ey", "Ez", "Hy", "Hz"),
                           freqs_hz=(_F0,),
                           mode_port=ph.ModePort(
                               out_direction="+", center_um=(1.0, 1.2),
                               size_um=(0.8, 0.8), dl_um=0.1,
                               modes=(ph.PortMode(polarization="TE",
                                                  mode_index=0),))),
    ])
    assert _by_id(checks.monitor_checks(sim), "absorbing-band") == []


def test_out_of_domain_flux_plane_is_already_a_hard_model_error():
    # No advisory id exists for this case: the model's own snapping validator
    # rejects it at parse time, so the checks layer can never see one.
    with pytest.raises(Exception, match="outside the interior range"):
        _sim([ph.FluxMonitor(name="lost", axis="x", position_um=5.0,
                             freqs_hz=(_F0,))])


def test_out_of_band_frequencies_warn_with_worst_offender():
    report = checks.monitor_checks(_sim([
        ph.FluxMonitor(name="wide", axis="x", position_um=3.2,
                       freqs_hz=(_F0, 400e12)),
    ]))
    finding = _by_id(report, "out-of-band")[0]
    assert finding["severity"] == "warning"
    assert finding["detail"]["offending"] == 1
    assert finding["detail"]["total"] == 2
    assert finding["detail"]["worst_wavelength_nm"] == pytest.approx(
        _C / 400e12 * 1e9, rel=1e-6)
    expected = float(np.exp(-0.5 * ((400e12 - _F0) / _FWIDTH) ** 2))
    assert finding["detail"]["worst_envelope"] == pytest.approx(expected)


def test_aliased_time_probe_warns_with_alias_free_interval():
    report = checks.monitor_checks(_sim([
        ph.FieldTimeMonitor(name="sparse", center_um=(2.0, 1.0, 1.2),
                            fields=("Ez",), interval_steps=100),
    ]))
    finding = _by_id(report, "aliasing")[0]
    dt = report["dt_s"]
    fmax = _F0 + 3.0 * _FWIDTH
    assert finding["detail"]["required_hz"] == pytest.approx(2.0 * fmax)
    safe = finding["detail"]["max_alias_free_interval"]
    assert safe == int(1.0 / (2.0 * fmax * dt))
    # The suggested interval must itself be alias-free.
    assert 1.0 / (dt * safe) >= 2.0 * fmax


def test_large_snapshot_output_warns_and_total_is_reported():
    report = checks.monitor_checks(_sim([
        ph.FieldSnapshotMonitor(name="movie", fields=("Ex", "Ey", "Ez"),
                                interval_steps=1),
    ]))
    per_monitor = _by_id(report, "output-size")
    assert [f["severity"] for f in per_monitor] == ["warning", "info"]
    # 80*40*48 cells * 3 components * 4 bytes * 2000 frames ≈ 3.4 GiB.
    expected = 80 * 40 * 48 * 3 * 4 * 2000
    assert per_monitor[0]["detail"]["output_bytes"] == expected
    assert report["output_bytes_total"] == expected


def test_apodization_after_budgeted_run_warns():
    dt = checks.monitor_checks(_sim([]))["dt_s"]
    duration = 2000 * dt
    late = ph.FieldDftMonitor(
        name="late", center_um=(2.0, 1.0, 1.2), size_um=(4.0, 2.0, 0.0),
        fields=("Ex",), freqs_hz=(_F0,),
        apodization=ph.Apodization(start_s=duration * 2, width_s=1e-14))
    long_tail = ph.FieldDftMonitor(
        name="tail", center_um=(2.0, 1.0, 1.2), size_um=(4.0, 2.0, 0.0),
        fields=("Ex",), freqs_hz=(_F0,),
        apodization=ph.Apodization(start_s=0.0, end_s=duration * 2,
                                   width_s=1e-14))
    report = checks.monitor_checks(_sim([late, long_tail]))
    findings = {f["monitor"]: f for f in _by_id(report, "apodization-window")}
    assert findings["late"]["severity"] == "warning"
    assert findings["tail"]["severity"] == "info"


# --------------------------------------------------------------------------- #
# result_checks — post-run data health
# --------------------------------------------------------------------------- #


class _HealthData:
    def __init__(self, entries, aborted=False):
        """entries: list of (manifest_entry, array-or-exception)."""
        self.manifest = {"monitors": [entry for entry, _ in entries]}
        self._arrays = {entry["name"]: value for entry, value in entries}
        self.aborted = aborted
        self.abort_reason = "test" if aborted else None

    def __getitem__(self, name):
        value = self._arrays[name]
        if isinstance(value, Exception):
            raise value
        return xr.DataArray(value)


def _entry(name, type_="field_dft"):
    return {"name": name, "type": type_}


def test_result_checks_flags_nonfinite_allzero_and_hot_flux():
    report = checks.result_checks(_HealthData([
        (_entry("diverged"), np.asarray([1.0, np.nan, 2.0, np.inf])),
        (_entry("dead"), np.zeros((4, 4))),
        (_entry("hot", "flux"), np.asarray([0.5, -2.0])),
        (_entry("fine", "flux"), np.asarray([0.25, 0.5])),
    ]))
    assert report["monitors"]["diverged"]["status"] == "error"
    assert report["monitors"]["diverged"]["finite_fraction"] == pytest.approx(0.5)
    assert report["monitors"]["dead"]["status"] == "warning"
    assert report["monitors"]["hot"]["status"] == "warning"
    assert report["monitors"]["fine"] == {
        "status": "ok", "finite_fraction": 1.0, "abs_max": 0.5, "sampled": 2}
    assert sorted(_ids(report)) == ["all-zero", "flux-above-unity",
                                    "non-finite"]
    assert report["counts"] == {"error": 1, "warning": 2, "info": 0}


def test_result_checks_survives_an_unreadable_monitor():
    report = checks.result_checks(_HealthData([
        (_entry("broken"), ValueError("truncated blob")),
        (_entry("fine"), np.ones(3)),
    ]))
    finding = _by_id(report, "read-error")[0]
    assert finding["monitor"] == "broken"
    assert "truncated blob" in finding["message"]
    assert report["monitors"]["fine"]["status"] == "ok"


def test_result_checks_strides_large_monitors():
    big = np.full((64, 64, 64, 8), np.nan)
    report = checks.result_checks(_HealthData([(_entry("big"), big)]))
    sampled = report["monitors"]["big"]["sampled"]
    assert 0 < sampled <= checks._HEALTH_SAMPLE_CAP
    assert sampled < big.size
    assert report["monitors"]["big"]["status"] == "error"


def test_result_checks_reports_empty_output():
    report = checks.result_checks(_HealthData([
        (_entry("empty"), np.empty((0,), dtype=np.float32)),
    ]))
    assert _ids(report) == ["all-zero"]
    assert report["monitors"]["empty"]["sampled"] == 0


# --------------------------------------------------------------------------- #
# timeseries_fft_values
# --------------------------------------------------------------------------- #


class _TimeData:
    manifest = {"monitors": [{
        "name": "probe", "type": "field_time",
        "dims": ["sample", "component"], "shape": [512, 1],
        "components": ["Ez"],
    }]}

    def __init__(self, y, dt=1e-16):
        t = np.arange(len(y)) * dt
        self.da = xr.DataArray(
            np.asarray(y, dtype=np.float32)[:, None],
            dims=("t", "component"),
            coords={"t": t, "component": ["Ez"]})

    def __getitem__(self, name):
        assert name == "probe"
        return self.da


def test_timeseries_fft_peaks_at_the_driven_frequency():
    dt, n, f0 = 1e-16, 512, 200e12
    t = np.arange(n) * dt
    data = _TimeData(np.sin(2 * np.pi * f0 * t), dt)
    s = service.timeseries_fft_values(data, "probe", field="Ez")
    assert s["samples"] == n
    assert s["resolution_hz"] == pytest.approx(1.0 / (n * dt))
    assert s["nyquist_hz"] == pytest.approx(0.5 / dt)
    freq = np.asarray(s["freq_hz"])
    db = np.asarray(s["psd_db"])
    assert freq[0] > 0.0  # DC bin dropped
    peak_hz = freq[int(np.argmax(db))]
    assert peak_hz == pytest.approx(f0, abs=s["resolution_hz"])
    assert db.max() == pytest.approx(0.0)
    assert s["wavelength_nm"][int(np.argmax(db))] == pytest.approx(
        _C / peak_hz * 1e9)


def test_timeseries_fft_figure_opens_zoomed_to_the_significant_band():
    from photonhub.viz import figures

    dt, n, f0 = 1e-16, 512, 200e12
    t = np.arange(n) * dt
    data = _TimeData(np.sin(2 * np.pi * f0 * t), dt)
    figure = figures.timeseries_fft_figure(data, "probe", field="Ez")
    lo_thz, hi_thz = figure["layout"]["xaxis"]["range"]
    assert lo_thz <= f0 / 1e12 <= hi_thz
    # The initial view is the significant band, not the 5000 THz Nyquist span.
    assert hi_thz - lo_thz < 0.5 / dt / 1e12 / 2
    assert lo_thz >= 0.0


def test_timeseries_fft_rejects_short_and_silent_series():
    with pytest.raises(ValueError, match="at least 4"):
        service.timeseries_fft_values(_TimeData([1.0, 2.0, 3.0]), "probe",
                                      field="Ez")
    with pytest.raises(ValueError, match="identically zero"):
        service.timeseries_fft_values(_TimeData(np.ones(16)), "probe",
                                      field="Ez")


# --------------------------------------------------------------------------- #
# HTTP seams
# --------------------------------------------------------------------------- #


def test_monitor_check_endpoint_reports_findings_and_422s(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    sim = _sim([ph.FluxMonitor(name="edge", axis="x", position_um=0.3,
                               freqs_hz=(_F0,))])
    with TestClient(create_app()) as client:
        response = client.post("/api/workspace/monitor-check",
                               json={"spec": sim.to_wire_dict()})
        assert response.status_code == 200
        payload = response.json()
        assert [f["id"] for f in payload["findings"]] == ["absorbing-band"]
        assert payload["counts"]["warning"] == 1
        assert payload["num_steps"] == 2000

        # Advisory endpoint must not mutate the shared workspace state.
        assert client.get("/api/workspace").status_code == 404

        broken = client.post("/api/workspace/monitor-check",
                             json={"spec": {"schema_version": "9.0.0"}})
        assert broken.status_code == 422
        assert "message" in broken.json()["detail"]


def test_result_checks_and_fft_endpoints(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    dt = 1e-16
    n = 1000  # puts f0 exactly on an rfft bin (bin spacing 10 THz)
    f0 = 200e12
    samples = np.sin(2 * np.pi * f0 * np.arange(n) * dt)
    interleaved = np.zeros((n, 1), dtype="<f4")
    interleaved[:, 0] = samples
    interleaved.tofile(tmp_path / "probe.bin")
    np.asarray([0.5, 2.5], dtype="<f4").tofile(tmp_path / "flux.bin")
    manifest = {
        "manifest_version": "1", "schema_version": "1.0.0",
        "run": {"dt_s": dt}, "grid": {"dl_um": 0.05}, "provenance": {},
        "monitors": [
            {"name": "probe", "type": "field_time", "file": "probe.bin",
             "dtype": "float32", "shape": [n, 1],
             "dims": ["sample", "component"], "components": ["Ez"],
             "sample_steps": list(range(1, n + 1))},
            {"name": "flux", "type": "flux", "file": "flux.bin",
             "dtype": "float32", "shape": [2], "axis": "x",
             "dims": ["freq"], "freqs_hz": [190e12, 200e12]},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with TestClient(create_app(tmp_path)) as client:
        revision = client.get("/api/session").json()["result_id"]

        health = client.get("/api/result/checks", params={"rev": revision})
        assert health.status_code == 200
        payload = health.json()
        assert payload["monitors"]["probe"]["status"] == "ok"
        assert payload["monitors"]["flux"]["status"] == "warning"
        assert [f["id"] for f in payload["findings"]] == ["flux-above-unity"]

        stale = client.get("/api/result/checks", params={"rev": "bogus"})
        assert stale.status_code == 409

        fft = client.get("/api/monitor/probe/fft",
                         params={"field": "Ez", "rev": revision})
        assert fft.status_code == 200
        figure = fft.json()
        trace = figure["data"][0]
        top = int(np.argmax(np.asarray(trace["y"], dtype=float)))
        assert trace["x"][top] == pytest.approx(f0 / 1e12, rel=0.02)
        assert figure["layout"]["xaxis"]["title"] == "frequency (THz)"

        missing = client.get("/api/monitor/flux/fft",
                             params={"field": "Ez", "rev": revision})
        assert missing.status_code == 400


def test_result_spec_endpoint_serves_and_gates_the_recorded_input(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import hashlib

    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    sim = _sim([ph.FluxMonitor(name="thru", axis="x", position_um=2.0,
                               freqs_hz=(_F0,))])
    canonical = sim.to_wire_json() + "\n"
    np.asarray([0.5], dtype="<f4").tofile(tmp_path / "flux.bin")

    def manifest(input_sha256):
        provenance = {"input_sha256": input_sha256} if input_sha256 else {}
        return {
            "manifest_version": "1", "schema_version": "1.0.0",
            "run": {"dt_s": 1e-16}, "grid": {"dl_um": 0.05},
            "provenance": provenance,
            "monitors": [{"name": "thru", "type": "flux", "file": "flux.bin",
                          "dtype": "float32", "shape": [1], "axis": "x",
                          "dims": ["freq"], "freqs_hz": [_F0]}],
        }

    (tmp_path / "sim.json").write_text(canonical, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(
        manifest(hashlib.sha256(canonical.encode()).hexdigest())))
    with TestClient(create_app(tmp_path)) as client:
        revision = client.get("/api/session").json()["result_id"]
        served = client.get("/api/result/spec", params={"rev": revision})
        assert served.status_code == 200
        payload = served.json()
        assert payload["available"] is True
        assert payload["geometry_status"] == "matched"
        assert payload["spec"]["monitors"][0]["name"] == "thru"
        # Mode-source provenance rides along, evaluated against the recorded
        # document itself (no mode sources here -> empty, but always present).
        assert payload["mode_source_statuses"] == []

        stale = client.get("/api/result/spec", params={"rev": "bogus"})
        assert stale.status_code == 409

    # Legacy bundle (no recorded hash): served, explicitly unverified.
    (tmp_path / "manifest.json").write_text(json.dumps(manifest(None)))
    with TestClient(create_app(tmp_path)) as client:
        payload = client.get("/api/result/spec").json()
        assert payload["available"] is True
        assert payload["geometry_status"] == "unverified"

    # Tampered sibling: withheld with the reason, never served.
    (tmp_path / "manifest.json").write_text(json.dumps(
        manifest("0" * 64)))
    with TestClient(create_app(tmp_path)) as client:
        payload = client.get("/api/result/spec").json()
        assert payload["available"] is False
        assert payload["geometry_status"] == "mismatch"
        assert "not the input recorded" in payload["reason"]

    # No sibling at all.
    (tmp_path / "sim.json").unlink()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest(None)))
    with TestClient(create_app(tmp_path)) as client:
        payload = client.get("/api/result/spec").json()
        assert payload["available"] is False
        assert payload["geometry_status"] == "missing"
