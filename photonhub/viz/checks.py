"""Advisory monitor checks (spec: docs/superpowers/specs/
2026-07-29-results-viz-monitor-checks-design.md).

Two pure entry points, both returning ``{"findings": [...], "counts": ...}``:

- :func:`monitor_checks` — setup-time advisories over a parsed
  ``Simulation``.  This is the cheap layer UNDER ``phsolver validate``: the
  engine stays authoritative for hard rejects, while these findings flag
  setups that would RUN fine and record silent garbage (a flux plane inside
  the PML band, DFT frequencies the normalizing source barely drives, an
  aliased time probe, an apodization gate that excludes the whole run, a
  multi-GB output budget).
- :func:`result_checks` — post-run data health over an open result bundle
  (non-finite values, all-zero recordings, a source-normalized flux far
  above unity).

Advisory only: nothing here raises for a merely-suspicious setup, and no
finding blocks a run.  Findings are ``{id, severity, monitor, message,
detail}`` with kebab-case ids and severity ``error | warning | info``.
"""

from typing import Any, Optional

import numpy as np

from ..cost import (_cells_and_min_spacing_um, _dt_seconds, _num_steps,
                    _region_cells)

_AXES = "xyz"
_C = 299_792_458.0

# Below this relative spectral envelope of the FIRST source (the recorded
# normalization reference), a normalized DFT/flux value divides by a
# vanishing amplitude and amplifies numerical noise by >= 1/floor.
_ENVELOPE_FLOOR = 1.0e-2
# A source-normalized flux mildly above 1 can be legitimate (resonant
# recycling, finite-band normalization); 1.5 is far outside that.
_FLUX_UNITY_TOLERANCE = 1.5
# Per-monitor on-disk output warning threshold, and the whole-run info line.
_MONITOR_OUTPUT_WARN_BYTES = 1 << 30      # 1 GiB
_TOTAL_OUTPUT_INFO_BYTES = 2 << 30        # 2 GiB
# Aliasing margin: a Gaussian pulse's energy above freq0 + 3*fwidth is
# negligible (99.7 %), so that is the highest frequency a probe must resolve.
_BAND_SIGMAS = 3.0
# result_checks reads at most about this many samples per monitor (strided
# view pages off the memmap, never the whole blob).
_HEALTH_SAMPLE_CAP = 1_000_000


def _finding(id_: str, severity: str, message: str,
             monitor: Optional[str] = None, **detail: Any) -> dict:
    out: dict[str, Any] = {"id": id_, "severity": severity,
                           "monitor": monitor, "message": message}
    if detail:
        out["detail"] = detail
    return out


def _counts(findings: list[dict]) -> dict:
    out = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        out[f["severity"]] = out.get(f["severity"], 0) + 1
    return out


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"


def _monitor_output_bytes(monitor, num_steps: int, dl_um: float,
                          cells_per_axis) -> int:
    """One monitor's on-disk output bytes — the per-monitor split of
    ``cost._monitor_bytes`` (same float32 manifest sizes)."""
    ncomp = len(getattr(monitor, "fields", ()) or ())
    if monitor.type == "field_time":
        return max(1, num_steps // monitor.interval_steps) * ncomp * 4
    if monitor.type == "field_snapshot":
        frames = (1 if monitor.interval_steps == 0
                  else max(1, num_steps // monitor.interval_steps))
        domain_cells = (cells_per_axis[0] * cells_per_axis[1]
                        * cells_per_axis[2])
        return frames * ncomp * domain_cells * 4
    if monitor.type == "field_dft":
        region = _region_cells(monitor.size_um, dl_um, cells_per_axis,
                               getattr(monitor, "interval_space", None))
        return len(monitor.freqs_hz) * ncomp * region * 2 * 4
    if monitor.type == "flux":
        return len(monitor.freqs_hz) * 4
    return 0


def monitor_checks(sim) -> dict:
    """Setup-time advisory findings for every monitor of a ``Simulation``."""
    findings: list[dict] = []
    cells_per_axis, min_spacing_um = _cells_and_min_spacing_um(sim)
    dt_s = _dt_seconds(sim, min_spacing_um)
    num_steps = _num_steps(sim, dt_s)
    reference = sim.sources[0].source_time  # the recorded normalization

    band_cache: dict[int, Optional[tuple]] = {}

    def bounds(axis_index: int):
        """(lo, hi, boundary-kind) outside the absorbing band, or None when
        the model itself cannot produce one (layers consume the axis — the
        engine's validate rejects that scene with its own message)."""
        if axis_index not in band_cache:
            try:
                band_cache[axis_index] = sim._nonabsorbing_bounds_um(axis_index)
            except ValueError:
                band_cache[axis_index] = None
        return band_cache[axis_index]

    if not sim.monitors:
        findings.append(_finding(
            "no-monitors", "info",
            "No monitors: the run records only ledger statistics, no field "
            "or flux data."))

    total_bytes = 0
    for monitor in sim.monitors:
        name = monitor.name

        # --- absorbing-band (warning): the recorded sample plane/point sits
        # inside a PML/absorber band, where fields are attenuated/stretched.
        # (Fully OUT-of-domain planes/centers never reach here — the model's
        # own validators hard-reject them at parse time.)  Modal-port planes
        # are exempt: the model hard-validates those against the band too.
        positions: list[tuple[int, float, str]] = []
        if monitor.type == "flux":
            axis_index = _AXES.index(monitor.axis)
            positions = [(axis_index, float(monitor.position_um),
                          "flux plane")]
        elif (monitor.type == "field_dft"
              and getattr(monitor, "mode_port", None) is None):
            positions = [(i, float(monitor.center_um[i]), "DFT plane")
                         for i, size in enumerate(monitor.size_um)
                         if size == 0.0]
        elif monitor.type == "field_time":
            positions = [(i, float(monitor.center_um[i]), "time probe")
                         for i in range(3)]
        for axis_index, position_um, what in positions:
            interval = bounds(axis_index)
            if interval is None or interval[2] not in ("pml", "absorber"):
                continue
            lo, hi, kind = interval
            if not (lo + 1e-12 < position_um < hi - 1e-12):
                findings.append(_finding(
                    "absorbing-band", "warning",
                    f"The {what} sits inside the {kind} band on "
                    f"'{_AXES[axis_index]}' ({position_um:.6g} µm; "
                    f"nonabsorbing interior is ({lo:.6g}, {hi:.6g}) µm). "
                    "Data recorded there is attenuated by the boundary and "
                    "is not quantitative.",
                    monitor=name, axis=_AXES[axis_index],
                    position_um=position_um, interior_um=[lo, hi],
                    boundary=kind))
                break  # one band finding per monitor is enough

        # --- out-of-band (warning): normalized against the FIRST source's
        # pulse spectrum; a vanishing envelope amplifies noise by 1/envelope.
        freqs = getattr(monitor, "freqs_hz", None)
        if freqs:
            envelopes = [(float(reference.spectral_amplitude(f)), float(f))
                         for f in freqs]
            offenders = [pair for pair in envelopes
                         if pair[0] < _ENVELOPE_FLOOR]
            if offenders:
                worst_env, worst_f = min(offenders)
                findings.append(_finding(
                    "out-of-band", "warning",
                    f"{len(offenders)} of {len(freqs)} recorded frequencies "
                    "fall outside the first source's band (worst: "
                    f"{_C / worst_f * 1e9:.1f} nm at relative spectral "
                    f"amplitude {worst_env:.1e}). Source-normalized values "
                    f"there amplify numerical noise by ≳{1.0 / max(worst_env, 1e-300):.0e}.",
                    monitor=name, offending=len(offenders),
                    total=len(freqs), worst_wavelength_nm=_C / worst_f * 1e9,
                    worst_envelope=worst_env))

        # --- aliasing (warning): a decimated time probe must still resolve
        # the fastest driven oscillation.
        if monitor.type == "field_time" and monitor.interval_steps > 1:
            fmax = max(s.source_time.freq0_hz
                       + _BAND_SIGMAS * s.source_time.fwidth_hz
                       for s in sim.sources)
            sample_hz = 1.0 / (dt_s * monitor.interval_steps)
            if sample_hz < 2.0 * fmax:
                safe = max(1, int(1.0 / (2.0 * fmax * dt_s)))
                findings.append(_finding(
                    "aliasing", "warning",
                    f"Sampling every {monitor.interval_steps} steps "
                    f"({sample_hz:.3e} Hz) undersamples the source band "
                    f"(needs ≥ {2.0 * fmax:.3e} Hz): the recorded trace is "
                    f"aliased. Largest alias-free interval: {safe} steps.",
                    monitor=name, interval_steps=monitor.interval_steps,
                    sample_hz=sample_hz, required_hz=2.0 * fmax,
                    max_alias_free_interval=safe))

        # --- output-size (warning per monitor; info total below).
        nbytes = _monitor_output_bytes(monitor, num_steps, sim.grid.dl_um,
                                       cells_per_axis)
        total_bytes += nbytes
        if nbytes > _MONITOR_OUTPUT_WARN_BYTES:
            findings.append(_finding(
                "output-size", "warning",
                f"Budgeted output is {_fmt_bytes(nbytes)} for this monitor "
                "alone. Consider spatial decimation, fewer frequencies, or a "
                "larger snapshot interval.",
                monitor=name, output_bytes=nbytes))

        # --- apodization-window: the DFT gate versus the budgeted run.
        apodization = getattr(monitor, "apodization", None)
        if apodization is not None:
            duration_s = num_steps * dt_s
            if (apodization.start_s is not None
                    and apodization.start_s >= duration_s):
                findings.append(_finding(
                    "apodization-window", "warning",
                    f"Apodization opens at {apodization.start_s * 1e12:.3g} "
                    f"ps but the budgeted run ends at "
                    f"{duration_s * 1e12:.3g} ps: the DFT accumulates "
                    "(almost) nothing.",
                    monitor=name, start_s=apodization.start_s,
                    run_duration_s=duration_s))
            elif (apodization.end_s is not None
                    and apodization.end_s > duration_s):
                findings.append(_finding(
                    "apodization-window", "info",
                    f"Apodization closes at {apodization.end_s * 1e12:.3g} "
                    f"ps, past the budgeted run end "
                    f"({duration_s * 1e12:.3g} ps); the gate is effectively "
                    "open-ended. (Auto-shutoff can end the run earlier "
                    "still.)",
                    monitor=name, end_s=apodization.end_s,
                    run_duration_s=duration_s))

    if total_bytes > _TOTAL_OUTPUT_INFO_BYTES:
        findings.append(_finding(
            "output-size", "info",
            f"All monitors together budget {_fmt_bytes(total_bytes)} of "
            "output.",
            output_bytes=total_bytes))

    return {
        "findings": findings,
        "counts": _counts(findings),
        "dt_s": dt_s,
        "num_steps": num_steps,
        "output_bytes_total": total_bytes,
        "monitor_count": len(sim.monitors),
    }


def _strided_sample(array) -> np.ndarray:
    """A materialized ≤ ~1M-sample strided view of a (possibly memmap-backed)
    array — reads only the touched pages, never the whole blob."""
    if array.size <= _HEALTH_SAMPLE_CAP:
        return np.asarray(array)
    stride = int(np.ceil((array.size / _HEALTH_SAMPLE_CAP)
                         ** (1.0 / array.ndim)))
    window = tuple(slice(None, None, stride) for _ in range(array.ndim))
    return np.asarray(array[window])


def result_checks(data) -> dict:
    """Post-run data-health findings for every monitor of a result bundle."""
    findings: list[dict] = []
    monitors: dict[str, dict] = {}
    for entry in data.manifest.get("monitors", []):
        name = entry["name"]
        try:
            sample = _strided_sample(data[name].data)
        except Exception as exc:  # one corrupt blob must not hide the rest
            findings.append(_finding(
                "read-error", "error",
                f"Monitor data could not be read: {exc}", monitor=name))
            monitors[name] = {"status": "error", "finite_fraction": None,
                              "abs_max": None, "sampled": 0}
            continue

        status = "ok"
        if sample.size == 0:
            findings.append(_finding(
                "all-zero", "warning",
                "Recorded no samples (empty output) — aborted run or an "
                "interval that never fired.", monitor=name))
            monitors[name] = {"status": "warning", "finite_fraction": None,
                              "abs_max": None, "sampled": 0}
            continue

        finite = np.isfinite(sample)
        finite_fraction = float(np.count_nonzero(finite)) / sample.size
        magnitudes = np.abs(sample[finite])
        abs_max = float(magnitudes.max()) if magnitudes.size else None

        if finite_fraction < 1.0:
            status = "error"
            findings.append(_finding(
                "non-finite", "error",
                f"{(1.0 - finite_fraction):.1%} of sampled values are "
                "NaN/Inf — the run likely diverged or aborted mid-write.",
                monitor=name, finite_fraction=finite_fraction))
        elif abs_max == 0.0:
            status = "warning"
            findings.append(_finding(
                "all-zero", "warning",
                "Every sampled value is exactly zero: the monitor recorded "
                "nothing (probe in dead space, gated out, or before the "
                "source ramp).", monitor=name))

        if (entry.get("type") == "flux" and abs_max is not None
                and abs_max > _FLUX_UNITY_TOLERANCE):
            status = "error" if status == "error" else "warning"
            findings.append(_finding(
                "flux-above-unity", "warning",
                f"Source-normalized flux reaches {abs_max:.3g} (> "
                f"{_FLUX_UNITY_TOLERANCE:g}): non-physical for a passive "
                "scene — usually an under-resolved run or frequencies "
                "outside the normalizing source's band.",
                monitor=name, abs_max=abs_max))

        monitors[name] = {"status": status,
                          "finite_fraction": finite_fraction,
                          "abs_max": abs_max,
                          "sampled": int(sample.size)}

    return {
        "findings": findings,
        "counts": _counts(findings),
        "monitors": monitors,
        "aborted": bool(data.aborted),
    }
