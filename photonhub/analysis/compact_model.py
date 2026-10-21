"""Checked S-matrix data and wavelength-interpolated circuit models."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .._compat import caller_stacklevel
from ._constants import C0
from .smatrix import _passivity_excess, passivity_violation, reciprocity_error

PROVENANCE_NOTE = (
    "S-parameters inherit the convergence of the simulation that produced "
    "them and the mesh it was run on; these checks test consistency, not accuracy."
)


@dataclass(frozen=True)
class SMatrixChecks:
    """Passivity and reciprocity measurements; ``num_freqs`` counts samples."""

    passivity_violation: float
    passivity_atol: float
    passive: bool
    reciprocity_error: float
    reciprocity_atol: float
    reciprocal: bool
    num_freqs: int

    def to_dict(self) -> dict[str, Any]:
        """Return values suitable for JSON serialization."""
        return asdict(self)

    def summary(self) -> str:
        """Summarize the two checks on one line."""
        p = "pass" if self.passive else "FAIL"
        r = "pass" if self.reciprocal else "FAIL"
        return (f"passivity_violation={self.passivity_violation:.3e} "
                f"(atol {self.passivity_atol:g}, {p}); "
                f"reciprocity_error={self.reciprocity_error:.3e} "
                f"(atol {self.reciprocity_atol:g}, {r})")


def _validate_tolerance(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and >= 0") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _prepare_smatrix(S):
    """Return ordered labels, sorted frequencies and a complete complex array."""
    Sd = S.transpose("port_out", "port_in", "f")
    out_labels = [str(v) for v in Sd.coords["port_out"].values]
    in_labels = [str(v) for v in Sd.coords["port_in"].values]
    if out_labels != in_labels:
        raise ValueError("port_out and port_in labels differ: "
                         f"{out_labels} vs {in_labels}")
    if not out_labels:
        raise ValueError("S carries no ports")
    if len(set(out_labels)) != len(out_labels):
        raise ValueError("S port labels must be unique")
    freqs = np.asarray(Sd.coords["f"].values, dtype=np.float64)
    if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError("frequencies must be finite and > 0 Hz")
    order = np.argsort(freqs)
    if not len(freqs):
        raise ValueError("S carries no frequencies")
    arr = np.asarray(Sd.values, dtype=np.complex128)[:, :, order]
    if not np.all(np.isfinite(arr)):
        bad = int(arr.size - np.count_nonzero(np.isfinite(arr)))
        raise ValueError(f"S contains {bad} non-finite entries")
    return tuple(out_labels), freqs[order], arr


def smatrix_checks(S, *, passivity_atol: float = 1e-3,
                   reciprocity_atol: float = 1e-3) -> SMatrixChecks:
    """Measure passivity and reciprocity with nonnegative finite tolerances.

    Parameters
    ----------
    S:
        S-matrix with axes ``(port_out, port_in, f)`` or an array accepted by
        the S-matrix diagnostics.
    passivity_atol, reciprocity_atol:
        Allowed excess for each consistency check.
    """
    p_atol = _validate_tolerance(passivity_atol, "passivity_atol")
    r_atol = _validate_tolerance(reciprocity_atol, "reciprocity_atol")
    if hasattr(S, "dims"):
        num_freqs = int(S.sizes["f"])
    else:
        arr = np.asarray(S)
        num_freqs = 1 if arr.ndim == 2 else int(arr.shape[0])
    p = passivity_violation(S)
    r = reciprocity_error(S)
    return SMatrixChecks(p, p_atol, p <= p_atol, r, r_atol,
                         r <= r_atol, num_freqs)


def _writer_checks(S, *, strict: bool, passivity_atol: float,
                   reciprocity_atol: float) -> SMatrixChecks:
    """Apply the shared writer rule before any output is created."""
    if not np.all(np.isfinite(np.asarray(S))):
        raise ValueError("S contains non-finite entries")
    checks = smatrix_checks(S, passivity_atol=passivity_atol,
                            reciprocity_atol=reciprocity_atol)
    failures = []
    if not checks.passive:
        failures.append(f"passivity_violation={checks.passivity_violation:.3e} "
                        f"exceeds tolerance {checks.passivity_atol:g}")
    if not checks.reciprocal:
        failures.append(f"reciprocity_error={checks.reciprocity_error:.3e} "
                        f"exceeds tolerance {checks.reciprocity_atol:g}")
    if failures:
        message = "S-matrix check failed: " + "; ".join(failures)
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=caller_stacklevel())
    return checks


# These parameters describe diagnostics, not a continuous passivity certificate.
_INTERPOLATION = {
    "method": "polar with checked whole-matrix complex fallback",
    "probe_fractions": [k / 16 for k in range(1, 16)],
    "significant_relative_magnitude": 0.1,
    "outside_range": "error",
}


def _diagnostic_metadata(metadata, diagnostics):
    """Replace generated diagnostics while preserving caller-owned keys."""
    namespace = "photonhub.compact_model"
    extra = dict(metadata)
    for key, value in metadata.items():
        suffix = key.removeprefix(namespace)
        generated_key = key == namespace or (
            key.startswith(namespace) and suffix.replace(".generated", "") == "")
        if generated_key and isinstance(value, Mapping) and (
            {"sample_checks", "interpolant_checks"} <= value.keys()
            or value.get("policy") == (
                "legacy polar everywhere; no interior checks on load")):
            del extra[key]
    key = namespace
    while key in extra:
        key += ".generated"
    extra[key] = diagnostics
    return extra


@dataclass(frozen=True, eq=False)
class SMatrixModel:
    """A sampled S-matrix interpolated in wavelength for circuit use.

    ``s`` uses ``[port_out, port_in, wavelength]`` order. Callable dictionary
    keys use ``(port_from, port_to)`` order.

    Polar (linear magnitude and unwrapped phase) interpolation is the default.
    At build time, an interval switches to whole-matrix complex interpolation
    only if a polar interior probe exceeds the largest sample gain (at least
    zero) plus the passivity tolerance and a roundoff margin. Samples
    and 15 interior points (k/16, k=1..15) are checked. These finite probes
    are diagnostics, not a bound on unsampled polar peaks or an accuracy
    certificate. Saved choices are used on every call without rechecking.

    Any switch warns once per model with the count, worst interval bounds
    and worst significant-entry midpoint amplitude drop in dB versus polar.
    Significant means at least 10% of the larger row/column maximum there.
    Phase steps above pi/2 warn when both endpoint magnitudes exceed 1e-6.
    Sampling more densely resolves rapid phase evolution,
    but cannot remove a zero's phase jump or detect aliased whole turns.
    Plain NumPy calls work in SAX circuits, but not with ``jax.jit``.
    """

    ports: tuple[str, ...]
    wlens_um: np.ndarray
    s: np.ndarray
    checks: SMatrixChecks
    metadata: Mapping[str, Any]
    max_phase_step_rad: float
    complex_intervals: tuple[int, ...] = ()
    interpolation: Mapping[str, Any] = field(default_factory=dict)
    _complex_mask: np.ndarray = field(init=False, repr=False)
    _magnitude: np.ndarray = field(init=False, repr=False)
    _phase: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        wavelengths = np.array(self.wlens_um, dtype=np.float64, copy=True)
        values = np.array(self.s, dtype=np.complex128, copy=True)
        n = len(self.ports)
        if (not n or wavelengths.ndim != 1 or not len(wavelengths)
                or values.shape != (n, n, len(wavelengths))):
            raise ValueError("model wavelengths or S shape is invalid")
        if (not np.all(np.isfinite(wavelengths))
                or np.any(wavelengths <= 0)
                or np.any(np.diff(wavelengths) <= 0)
                or not np.all(np.isfinite(values))):
            raise ValueError("model wavelengths and S must be finite and ordered")
        wavelengths.flags.writeable = False
        values.flags.writeable = False
        if ("photonhub_version" not in self.metadata
                or "note" not in self.metadata):
            raise ValueError("model metadata needs photonhub_version and note")
        intervals = tuple(self.complex_intervals)
        if (any(type(k) is not int or k < 0 or k >= len(wavelengths) - 1
                for k in intervals) or len(set(intervals)) != len(intervals)):
            raise ValueError("invalid complex_intervals")
        # Version 2 is unreleased; discard obsolete diagnostic settings so
        # loading and re-exporting cannot write parameters with no effect.
        config = {key: value for key, value in self.interpolation.items()
                  if key in _INTERPOLATION}
        name = "significant_relative_magnitude"
        if name in config:
            config[name] = _validate_tolerance(config[name], name)
        if "probe_fractions" in config:
            fractions = np.asarray(config["probe_fractions"], dtype=float)
            if (fractions.ndim != 1 or not len(fractions)
                    or not np.all(np.isfinite(fractions))
                    or np.any(fractions <= 0) or np.any(fractions >= 1)
                    or np.any(np.diff(fractions) <= 0)):
                raise ValueError("invalid probe_fractions")
            config["probe_fractions"] = fractions.tolist()
        mask = np.zeros(max(0, len(wavelengths) - 1), dtype=bool)
        mask[list(intervals)] = True
        magnitude = np.ascontiguousarray(np.abs(values))
        phase = np.ascontiguousarray(np.unwrap(np.angle(values), axis=2))
        for cached in (mask, magnitude, phase):
            cached.flags.writeable = False
        object.__setattr__(self, "complex_intervals", intervals)
        object.__setattr__(self, "interpolation", MappingProxyType(config))
        object.__setattr__(self, "_complex_mask", mask)
        object.__setattr__(self, "_magnitude", magnitude)
        object.__setattr__(self, "_phase", phase)
        object.__setattr__(self, "wlens_um", wavelengths)
        object.__setattr__(self, "s", values)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def _interpolate(self, wl) -> np.ndarray:
        query = np.asarray(((self.wlens_um[0] + self.wlens_um[-1]) / 2
                            if wl is None else wl), dtype=np.float64)
        low, high = self.wlens_um[[0, -1]]
        if (not np.all(np.isfinite(query))
                or np.any(query < low - 1e-12 * low)
                or np.any(query > high + 1e-12 * high)):
            raise ValueError(f"wavelength must be in [{low:g}, {high:g}] um")
        query = np.clip(query, low, high)
        if len(self.wlens_um) == 1:
            return np.broadcast_to(self.s[:, :, 0].reshape(
                self.s.shape[:2] + (1,) * query.ndim),
                self.s.shape[:2] + query.shape)
        flat = query.reshape(-1)
        intervals = np.clip(np.searchsorted(self.wlens_um, flat, side="right") - 1,
                            0, len(self.wlens_um) - 2)
        weight = ((flat - self.wlens_um[intervals])
                  / np.diff(self.wlens_um)[intervals])
        # Keep np.interp's exact arithmetic for legacy JSON compatibility.
        # Magnitudes and unwrapped phases are cached, not rebuilt per call.
        out = np.empty(self.s.shape[:2] + (flat.size,), dtype=complex)
        for i in range(len(self.ports)):
            for j in range(len(self.ports)):
                magnitude = np.interp(flat, self.wlens_um, self._magnitude[i, j])
                phase = np.interp(flat, self.wlens_um, self._phase[i, j])
                out[i, j] = magnitude * np.exp(1j * phase)
        use_complex = self._complex_mask[intervals]
        if np.any(use_complex):
            k, t = intervals[use_complex], weight[use_complex]
            out[..., use_complex] = self.s[..., k] * (1 - t) + self.s[..., k + 1] * t
        # Preserve stored values exactly, including the last sample.
        at_left = flat == self.wlens_um[intervals]
        at_right = flat == self.wlens_um[intervals + 1]
        out[..., at_left] = self.s[..., intervals[at_left]]
        out[..., at_right] = self.s[..., intervals[at_right] + 1]
        return out.reshape(self.s.shape[:2] + query.shape)

    def __call__(self, wl=None) -> dict[tuple[str, str], np.ndarray]:
        """Return all ordered circuit port pairs at wavelength ``wl`` in um."""
        values = self._interpolate(wl)
        return {(port_from, port_to): np.asarray(values[i, j])
                for i, port_to in enumerate(self.ports)
                for j, port_from in enumerate(self.ports)}

    def matrix(self, wl=None) -> np.ndarray:
        """Return a dense ``[port_out, port_in]`` matrix at ``wl`` in um."""
        values = self._interpolate(wl)
        if values.ndim == 2:
            return values
        return np.moveaxis(values, (0, 1), (-2, -1))

    def to_json(self, path) -> Path:
        """Write a deterministic JSON model file."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".json")
        obj = {
            "format": "photonhub-smatrix-model",
            "format_version": 2,
            "ports": list(self.ports),
            "wlens_um": self.wlens_um.tolist(),
            "freqs_hz": (C0 / (self.wlens_um * 1e-6)).tolist(),
            "s_real": self.s.real.tolist(),
            "s_imag": self.s.imag.tolist(),
            "index_order": ["port_out", "port_in", "wavelength"],
            "sdict_key_order": ["port_from", "port_to"],
            "interpolation": {**self.interpolation,
                              "complex_intervals": list(self.complex_intervals)},
            "checks": self.checks.to_dict(),
            "max_phase_step_rad": self.max_phase_step_rad,
            "metadata": dict(self.metadata),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1, sort_keys=True,
                                   allow_nan=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path) -> SMatrixModel:
        """Read a PhotonHub S-matrix JSON model."""
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("format") != "photonhub-smatrix-model":
            raise ValueError("unknown S-matrix model format")
        if obj.get("format_version") not in (1, 2):
            raise ValueError(f"unknown format_version: {obj.get('format_version')}")
        if "checks" not in obj:
            raise ValueError("model checks are missing")
        try:
            real = np.asarray(obj["s_real"], dtype=np.float64)
            imag = np.asarray(obj["s_imag"], dtype=np.float64)
            frequencies = np.asarray(obj["freqs_hz"], dtype=np.float64)
            wavelengths = np.asarray(obj["wlens_um"], dtype=np.float64)
            if real.shape != imag.shape:
                raise ValueError("s_real and s_imag shapes differ")
            if frequencies.shape != wavelengths.shape:
                raise ValueError("freqs_hz and wlens_um shapes differ")
            config = dict(obj.get("interpolation", {}))
            metadata = obj["metadata"]
            if obj["format_version"] == 1:
                config.pop("complex_intervals", None)
                intervals = ()
                metadata = _diagnostic_metadata(metadata, {
                    "policy": "legacy polar everywhere; no interior checks on load",
                })
            else:
                intervals = tuple(config.pop("complex_intervals"))
            return cls(tuple(obj["ports"]), wavelengths,
                       real + 1j * imag, SMatrixChecks(**obj["checks"]),
                       metadata, float(obj["max_phase_step_rad"]), intervals, config)
        except (KeyError, TypeError) as exc:
            raise ValueError("model JSON is incomplete") from exc


def smatrix_model(S, *, metadata: Mapping[str, Any] | None = None,
                  strict: bool = False, passivity_atol: float = 1e-3,
                  reciprocity_atol: float = 1e-3) -> SMatrixModel:
    """Build a checked wavelength-interpolated model from S-parameters.

    Parameters
    ----------
    S:
        Complex S-matrix with matching output and input port labels.
    metadata:
        JSON-serializable descriptive values with string keys.
    strict:
        Raise if passivity or reciprocity exceeds its tolerance.
    passivity_atol, reciprocity_atol:
        Nonnegative consistency check tolerances.

    Polar interpolation is the default. An interval switches to whole-matrix
    complex interpolation only when its polar excess exceeds the largest
    sample gain (at least zero) plus the tolerance and a roundoff margin.
    Noisy or uniformly gained samples keep polar and are reported by the
    checks, whose tolerance is unchanged. Construction
    rechecks the selected interpolant at 15 interior points (k/16, k=1..15),
    in addition to the samples. Finite probes can miss unsampled polar peaks;
    they do not certify continuous passivity or accuracy. Strict mode refuses
    failures; non-strict mode warns and records them. Every switch warns with
    its worst significant midpoint loss in dB: passivity is not accuracy.
    See :class:`SMatrixModel` for warning thresholds and sampling limitations.
    Plain NumPy calls work in SAX circuits, but not with ``jax.jit``.
    """
    ports, freqs, arr = _prepare_smatrix(S)
    checks = _writer_checks(arr.transpose(2, 0, 1), strict=strict,
                            passivity_atol=passivity_atol,
                            reciprocity_atol=reciprocity_atol)
    wavelengths = C0 / freqs * 1e6
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("duplicate wavelengths are not allowed")
    arr = arr[:, :, order]
    steps = np.abs(np.diff(np.unwrap(np.angle(arr), axis=2), axis=2))
    valid = ((np.abs(arr[:, :, 1:]) > 1e-6)
             & (np.abs(arr[:, :, :-1]) > 1e-6))
    max_step = float(np.max(steps[valid])) if np.any(valid) else 0.0
    extra = {} if metadata is None else dict(metadata)
    if any(not isinstance(key, str) for key in extra):
        raise ValueError("metadata keys must be strings")
    try:
        json.dumps(extra, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata values must be JSON-serializable") from exc
    from photonhub import __version__

    extra.update(photonhub_version=__version__, note=PROVENANCE_NOTE)
    model = SMatrixModel(ports, wavelengths, arr, checks, extra, max_step,
                         interpolation=_INTERPOLATION)
    config = model.interpolation
    fractions = np.asarray(config["probe_fractions"])
    num_intervals = len(wavelengths) - 1
    sample_excess = _passivity_excess(np.moveaxis(arr, -1, 0))
    reference = max(0., float(sample_excess.max()))
    switch_threshold = reference + checks.passivity_atol
    # Gram products and eigensolves accumulate roundoff with port count and
    # matrix power. 32 eps per port is a numerical tie margin, not a noise
    # allowance (about 1.4e-14 for a near-passive two-port matrix).
    switch_margin = (32 * np.finfo(float).eps * len(ports)
                     * max(1., 1 + reference))
    # Compare with all samples so existing sample noise or uniform gain does
    # not select the fallback. Sample/interior checks retain their tolerance.
    fallback = np.zeros(num_intervals, dtype=bool)
    p = np.empty((num_intervals, len(fractions)))
    r = np.empty_like(p)
    polar_p = np.empty_like(p)
    # Bound temporary memory: at 32 ports this is at most 960 matrices.
    for start in range(0, num_intervals, 64):
        stop = min(start + 64, num_intervals)
        probes = (wavelengths[start:stop, None]
                  + np.diff(wavelengths)[start:stop, None] * fractions)
        matrices = model.matrix(probes)
        polar_p[start:stop] = _passivity_excess(matrices)
        switch = np.max(polar_p[start:stop], axis=1) > (
            switch_threshold + switch_margin)
        fallback[start:stop] = switch
        p[start:stop] = polar_p[start:stop]
        if np.any(switch):
            k = np.arange(start, stop)[switch]
            left = np.moveaxis(arr[..., k], -1, 0)
            right = np.moveaxis(arr[..., k + 1], -1, 0)
            t = ((probes[switch] - wavelengths[k, None])
                 / np.diff(wavelengths)[k, None])[..., None, None]
            matrices[switch] = left[:, None] * (1 - t) + right[:, None] * t
            p[start:stop][switch] = _passivity_excess(matrices[switch])
        r[start:stop] = np.max(np.abs(matrices - matrices.swapaxes(-1, -2)),
                               axis=(-2, -1))

    # Every fallback reports loss, even below the former 5% silent allowance.
    switched = np.flatnonzero(fallback)
    loss_diagnostic = None
    if len(switched):
        midpoint_magnitude = (model._magnitude[..., :-1]
                              + model._magnitude[..., 1:]) / 2
        scale = np.maximum(midpoint_magnitude.max(axis=0, keepdims=True),
                           midpoint_magnitude.max(axis=1, keepdims=True))
        significant = (midpoint_magnitude > 0) & (
            midpoint_magnitude >= config["significant_relative_magnitude"] * scale)
        complex_magnitude = np.abs((arr[..., :-1] + arr[..., 1:]) / 2)
        ratio = np.divide(complex_magnitude, midpoint_magnitude,
                          out=np.ones_like(midpoint_magnitude),
                          where=significant)
        with np.errstate(divide="ignore"):
            drops_db = -20 * np.log10(np.minimum(ratio, 1.))
        per_interval = drops_db.max(axis=(0, 1))
        worst = int(switched[np.argmax(per_interval[switched])])
        drop_db = float(per_interval[worst])
        bounds = wavelengths[worst:worst + 2].tolist()
        # Exact cancellation has infinite dB loss; keep JSON standards-compliant.
        loss_diagnostic = {"value_db": drop_db if math.isfinite(drop_db) else None,
                           "complete_cancellation": not math.isfinite(drop_db),
                           "interval_index": worst, "interval_um": bounds}
        warnings.warn(
            "S-matrix polar interpolation gained power between samples; "
            f"complex fallback switched {len(switched)} interval(s); "
            f"worst midpoint amplitude drop {drop_db:.6g} dB relative to polar "
            f"on a significant entry in interval {worst} "
            f"[{bounds[0]:.12g}, {bounds[1]:.12g}] um",
            UserWarning, stacklevel=caller_stacklevel())

    # Preserve the original phase warning exactly, including near-zero steps.
    if max_step > math.pi / 2:
        warnings.warn(f"max_phase_step_rad={max_step:.3g} exceeds pi/2; "
                      "sampling may be too coarse to interpolate",
                      UserWarning, stacklevel=caller_stacklevel())

    nonpassive = (np.max(p, axis=1) > checks.passivity_atol
                  if num_intervals else fallback)
    undersampled = np.flatnonzero(fallback | nonpassive)
    diagnostics = {"fractions": fractions.tolist(), "num_points": int(p.size),
                   "complex_intervals": switched.tolist(),
                   "worst_midpoint_amplitude_drop": loss_diagnostic,
                   "undersampled_intervals": undersampled.tolist(),
                   "polar_passivity_violation": (
                       float(polar_p.max()) if p.size else None)}
    failures = []
    for name, errors, tolerance in (
        ("passivity_violation", p, checks.passivity_atol),
        ("reciprocity_error", r, checks.reciprocity_atol),
    ):
        if not errors.size:
            continue
        interval, point = np.unravel_index(np.argmax(errors), errors.shape)
        bounds = wavelengths[interval:interval + 2].tolist()
        probe = bounds[0] + (bounds[1] - bounds[0]) * fractions[point]
        worst = float(errors[interval, point])
        diagnostics[name] = {"value": worst, "wlen_um": float(probe),
                             "interval_index": int(interval), "interval_um": bounds}
        if worst > tolerance:
            failures.append(f"{name}={worst:.3e} exceeds tolerance "
                            f"{tolerance:g} in interval {interval} "
                            f"[{bounds[0]:.12g}, {bounds[1]:.12g}] um "
                            f"at {probe:.12g} um")
    if failures:
        message = "S-matrix check failed: interpolant " + "; ".join(failures)
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=caller_stacklevel())
    p_worst = max(checks.passivity_violation, float(p.max(initial=-np.inf)))
    r_worst = max(checks.reciprocity_error, float(r.max(initial=-np.inf)))
    combined = replace(checks, passivity_violation=p_worst,
                       passive=p_worst <= checks.passivity_atol,
                       reciprocity_error=r_worst,
                       reciprocal=r_worst <= checks.reciprocity_atol)
    extra = _diagnostic_metadata(extra, {"sample_checks": checks.to_dict(),
                                        "interpolant_checks": diagnostics})
    return replace(model, checks=combined, metadata=extra,
                   complex_intervals=tuple(np.flatnonzero(fallback).tolist()))
