"""Analytic completion of truncated resonator spectra (Prony / FDM tail).

A high-Q resonator forces a painful trade in FDTD: the spectrum of a
:class:`~photonhub.components.TimeMonitor` signal is the integral
:math:`F(f) = \\int_0^T u(t) e^{2\\pi i f t} dt`, and for a cavity whose energy
decays like :math:`e^{-2\\alpha t}` the integral converges only on the ringdown
timescale :math:`1/\\alpha = Q/(\\pi f_r)` — so the run length scales with Q,
and a :math:`Q = 10^5` cavity needs ~10\\ :sup:`6` steps of ringdown for a
spectrum the physics determined long before.

But the ringdown is not arbitrary data. After the sources switch off, a
resonator's field is a sum of decaying complex exponentials

.. math::

    u(t) = \\mathrm{Re} \\sum_k c_k e^{s_k t}, \\qquad
    s_k = -\\alpha_k - 2\\pi i f_k, \\quad \\alpha_k > 0,

one term per mode. If the poles :math:`s_k` and amplitudes :math:`c_k` are
known, the *remainder* of the spectrum integral has a closed form — so a run
may stop after a few well-resolved ringdown periods and the spectrum can be
completed analytically instead of stepped to convergence.

:class:`SpectrumCompleter` does exactly this, as CPU post-processing on a
recorded time series (the plugin contract: nowhere near the engine or the
wire format):

1. **Poles** come from :class:`~photonhub.plugins.ResonanceAnalysis` (filter
   diagonalization), fitted on the late, source-free part of the record.
2. **Amplitudes** are re-fit by linear least squares of the *real* recorded
   signal against those poles — a deliberately transparent step, so the model
   the acceptance gate validates is bit-for-bit the model the tail uses.
3. **The tail is a geometric series, not an integral.** The engine's §12
   running DFT is a rectangle-rule sum over samples spaced :math:`\\Delta`;
   completing it with the continuous integral would change quadrature mid-
   spectrum. Summing the model over the *same* discrete grid keeps the two
   parts of the answer on one quadrature, so as the record grows the
   completed spectrum converges to what the engine itself would have produced
   (up to fit quality), with no residual quadrature offset:

   .. math::

       \\sum_{m=1}^{\\infty} e^{(s_k + 2\\pi i f)(t_N + m\\Delta)} \\Delta
       = \\Delta e^{(s_k + 2\\pi i f) t_N}
         \\frac{r_k}{1 - r_k}, \\qquad r_k = e^{(s_k + 2\\pi i f)\\Delta},

   absolutely convergent for every decaying pole (:math:`|r_k| < 1`).
4. **A held-out window decides whether to believe any of it.** The model is
   fitted on the last segment of the record only, then extrapolated
   *backward* over an earlier, fully held-out window and compared against
   what was actually recorded there. Backward is the stringent direction —
   amplitudes grow, so an over-fitted decay rate or a missed mode is
   amplified rather than buried. Only if the relative residual on that
   window is below ``residual_tol`` is the tail added; otherwise the
   truncated spectrum is returned unchanged (with ``accepted = 0`` and a
   warning), or an exception is raised under ``strict=True``. Completion is
   an accelerator with a correctness gate, never a guess.

What this is NOT (v1 scope): it completes *point-probe* spectra — Q
extraction, mode lineshapes, local spectra. Completing a plane monitor
(flux/port transmission) needs per-cell amplitudes for every cell of the
plane, which requires late-window plane sampling or engine support; that is
a follow-up, not this module.

Example
-------
>>> import numpy as np
>>> from photonhub.plugins import SpectrumCompleter
>>> dt = 1.0
>>> t = np.arange(3000) * dt
>>> sig = np.real(2.0 * np.exp((-2j*np.pi*0.10 - 1e-3) * t))   # Q ~ 314
>>> sc = SpectrumCompleter(freq_window=(0.05, 0.15), fit_start=500 * dt)
>>> out = sc.complete_raw_signal(sig, dt, freqs=np.linspace(0.09, 0.11, 21))
>>> bool(out.attrs["accepted"])
True

After a real run::

    data = ph.run_local(sim)          # sim has a TimeMonitor "probe"
    sc = SpectrumCompleter(freq_window=(1.8e14, 2.1e14), fit_start=2.0e-13)
    out = sc.complete(data, "probe", freqs=np.linspace(1.9e14, 2.0e14, 201))
    out["spectrum"]                   # completed complex spectrum over 'freq'

The returned spectrum is the raw (un-normalized) DFT integral of the probe
signal, the same quantity a §12 point monitor accumulates before its
``A0·S(f)`` source normalization; divide by your source spectrum if you need
the engine-normalized form.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from .resonance import (
    ResonanceAnalysis,
    _combine_time_series,
    select_resonances,
)

__all__ = ["CompletionRejected", "SpectrumCompleter"]


class CompletionRejected(RuntimeError):
    """The held-out-window gate refused the ringdown model (``strict=True``)."""


class SpectrumCompleter:
    """Complete a truncated resonator spectrum from its fitted ringdown.

    Parameters
    ----------
    freq_window : (float, float)
        Frequency band (Hz) searched for resonant poles — passed to
        :class:`ResonanceAnalysis`. Should generously cover the band of
        ``freqs`` you will complete.
    fit_start : float, optional
        Absolute time (seconds, on the record's own time axis) where the
        pure-ringdown model becomes valid — i.e. after every source has
        switched off. Defaults to the midpoint of the record; setting it
        explicitly to just after your source's end is always better.
    holdout_fraction : float, default 0.25
        Fraction of the fit window ``[fit_start, T]`` held out for the
        acceptance gate. The model is fitted on the LAST
        ``1 - holdout_fraction`` of the window and extrapolated backward
        over the first ``holdout_fraction``, which the fit never saw.
    residual_tol : float, default 0.02
        Acceptance threshold on the held-out window: relative RMS residual
        between the backward-extrapolated model and the recorded signal.
    min_amplitude_rel : float, default 1e-3
        Poles whose FDM amplitude is below this fraction of the strongest
        pole's are dropped before the amplitude re-fit (spurious-mode guard).
    min_decay_resolved : float, default 0.1
        Reject unless every kept pole decays by at least this many decay
        constants across the fit segment (``alpha * span >= this``). The
        tail is an extrapolation of the fitted decay RATE; a record over
        which a mode's amplitude changes by under ~10% has not measured that
        rate, it has assumed it — the limiting case being a pure sinusoid,
        whose fitted ``alpha`` is numerical noise and whose "tail" would be
        a near-divergent spike the holdout gate cannot catch (the model fits
        the held-out window perfectly).
    strict : bool, default False
        If True, a rejected gate raises :class:`CompletionRejected` instead
        of warning and returning the truncated spectrum.
    resonance_analysis : ResonanceAnalysis, optional
        Pre-configured pole finder; by default one is built from
        ``freq_window`` with its default basis size.
    """

    def __init__(
        self,
        freq_window: Tuple[float, float],
        *,
        fit_start: Optional[float] = None,
        holdout_fraction: float = 0.25,
        residual_tol: float = 0.02,
        min_amplitude_rel: float = 1e-3,
        min_decay_resolved: float = 0.1,
        strict: bool = False,
        resonance_analysis: Optional[ResonanceAnalysis] = None,
    ):
        if not (0.0 < holdout_fraction < 0.9):
            raise ValueError(
                f"holdout_fraction must be in (0, 0.9); got {holdout_fraction}"
            )
        if residual_tol <= 0.0:
            raise ValueError(f"residual_tol must be > 0; got {residual_tol}")
        if min_amplitude_rel < 0.0:
            raise ValueError(
                f"min_amplitude_rel must be >= 0; got {min_amplitude_rel}"
            )
        if min_decay_resolved < 0.0:
            raise ValueError(
                f"min_decay_resolved must be >= 0; got {min_decay_resolved}"
            )
        self.freq_window = (float(freq_window[0]), float(freq_window[1]))
        self.fit_start = None if fit_start is None else float(fit_start)
        self.holdout_fraction = float(holdout_fraction)
        self.residual_tol = float(residual_tol)
        self.min_amplitude_rel = float(min_amplitude_rel)
        self.min_decay_resolved = float(min_decay_resolved)
        self.strict = bool(strict)
        self._rf = resonance_analysis or ResonanceAnalysis(freq_window=freq_window)

    # -- public entry points -------------------------------------------------

    def complete(
        self,
        sim_data: Mapping[str, xr.DataArray],
        monitor: str,
        freqs: Sequence[float],
        fields: Optional[Sequence[str]] = None,
    ) -> xr.Dataset:
        """Complete the spectrum of a ``TimeMonitor`` output.

        Parameters mirror :meth:`ResonanceAnalysis.run`; the monitor's own time
        coordinate supplies both the sample times and the spacing, so a
        decimated probe (``interval_steps > 1``) is handled consistently.
        """
        arr = sim_data[monitor]
        signal, dt = _combine_time_series([arr], fields)
        t0 = float(np.asarray(arr.coords["t"].values, dtype=float)[0])
        return self.complete_raw_signal(np.real(signal), dt, freqs, t0=t0)

    def complete_raw_signal(
        self,
        signal: np.ndarray,
        dt: float,
        freqs: Sequence[float],
        *,
        t0: float = 0.0,
    ) -> xr.Dataset:
        """Complete the spectrum of a raw, uniformly sampled REAL signal.

        Parameters
        ----------
        signal : array
            Real samples ``u(t_n)`` at ``t_n = t0 + n*dt``.
        dt : float
            Sample spacing (seconds). For an engine probe this is the
            monitor's recording interval, not necessarily the solver step.
        freqs : sequence of float
            Frequencies (Hz) at which to evaluate the spectrum.
        t0 : float, default 0.0
            Absolute time of the first sample.

        Returns
        -------
        xarray.Dataset
            ``spectrum`` (complex, completed), ``truncated`` (complex, the
            recorded-samples-only DFT), ``tail`` (complex, the analytic
            remainder actually added — zero when rejected), each over
            ``freq``; per-mode diagnostics ``mode_freq`` / ``mode_decay`` /
            ``mode_q`` over ``mode``; and attrs ``accepted`` (1/0),
            ``holdout_residual``, ``residual_tol``, ``fit_start``,
            ``holdout_end``, ``record_end``, ``reject_reason`` ("" when
            accepted).
        """
        u = np.asarray(signal)
        if np.iscomplexobj(u):
            raise ValueError(
                "complete_raw_signal expects the REAL recorded signal; the "
                "analytic-tail bookkeeping is written for real data"
            )
        u = u.astype(float, copy=False)
        if u.ndim != 1:
            raise ValueError(f"signal must be 1-D; got shape {u.shape}")
        if u.size < 16:
            raise ValueError(f"signal too short to fit a ringdown: {u.size}")
        dt = float(dt)
        if dt <= 0:
            raise ValueError(f"dt must be > 0; got {dt}")
        f_arr = np.asarray(freqs, dtype=float)
        if f_arr.ndim != 1 or f_arr.size == 0:
            raise ValueError("freqs must be a non-empty 1-D sequence")

        t = t0 + dt * np.arange(u.size)
        truncated = _rectangle_dft(u, t, dt, f_arr)

        # A signal that has already decayed to numerical dust needs no tail:
        # report success with an identically-zero correction rather than
        # fitting noise. Threshold: the last 5% of the record vs the peak.
        n_tail = max(8, u.size // 20)
        peak = float(np.max(np.abs(u))) if u.size else 0.0
        if peak == 0.0 or float(np.max(np.abs(u[-n_tail:]))) < 1e-9 * peak:
            return self._package(
                f_arr, truncated, np.zeros_like(truncated), None,
                accepted=True, residual=0.0, reason="",
                fit_start=t[0], holdout_end=t[0], record_end=t[-1])

        # -- fit / holdout split ---------------------------------------------
        a = self.fit_start if self.fit_start is not None else t0 + 0.5 * (
            t[-1] - t0)
        if not (t[0] <= a < t[-1]):
            raise ValueError(
                f"fit_start {a} is outside the record [{t[0]}, {t[-1]}]")
        i_a = int(np.searchsorted(t, a))
        n_win = u.size - i_a
        if n_win < 16:
            raise ValueError(
                f"fit window [{a}, {t[-1]}] holds only {n_win} samples; "
                "record longer or move fit_start earlier")
        i_h = i_a + max(4, int(round(self.holdout_fraction * n_win)))
        if u.size - i_h < 8:
            raise ValueError("fit segment after holdout is too short")

        holdout_t, holdout_u = t[i_a:i_h], u[i_a:i_h]
        fit_t, fit_u = t[i_h:], u[i_h:]

        # -- poles (FDM on the fit segment ONLY — the holdout stays unseen) --
        # Windowed to freq_window: FDM deliberately also returns out-of-band
        # and image poles (a real signal has conjugate images at -f); those
        # are basis artifacts to DROP, not evidence about the in-band model.
        # require_decay stays False so a genuinely non-decaying IN-BAND pole
        # is seen and rejected below rather than silently discarded.
        reason = ""
        poles = np.empty(0, dtype=complex)
        try:
            modes = select_resonances(
                self._rf.run_raw_signal(fit_u.astype(complex), dt),
                freq_window=self.freq_window,
                require_decay=False, sort_by="amplitude")
        except (ValueError, np.linalg.LinAlgError) as e:
            # A degenerate fit segment (numerical dust, rank-deficient basis)
            # can break the FDM eigensolve outright. That is a rejection, not
            # a crash: the record cannot support a ringdown model.
            modes = None
            reason = f"pole fit failed on the fit segment: {e}"
        if reason:
            pass
        elif modes.sizes.get("freq", 0) == 0:
            reason = "no resonant poles found in freq_window"
        else:
            amp = modes["amplitude"].values
            keep = amp >= self.min_amplitude_rel * float(np.max(amp))
            mf = modes.coords["freq"].values[keep]
            md = modes["decay"].values[keep]
            span = float(fit_t[-1] - fit_t[0])
            if np.any(md <= 0):
                # A non-decaying pole with real amplitude means the tail sum
                # diverges — the record is simply too short to complete.
                reason = (
                    "non-decaying pole(s) at "
                    f"{mf[md <= 0].tolist()} Hz — the tail does not converge")
            elif np.any(md * span < self.min_decay_resolved):
                # The record never MEASURED this decay rate (amplitude moved
                # by under min_decay_resolved decay constants across the fit
                # segment) — the tail would extrapolate an assumption. The
                # holdout gate cannot catch this: the model fits the held-out
                # window perfectly, the error is entirely beyond the record.
                bad = mf[md * span < self.min_decay_resolved]
                reason = (
                    f"decay not resolved for pole(s) at {bad.tolist()} Hz "
                    f"(alpha*span < {self.min_decay_resolved:g}) — record "
                    "too short to trust an extrapolated tail")
            else:
                poles = -md - 2j * np.pi * mf

        c = None
        residual = np.inf
        if not reason:
            # -- amplitudes: LLS of the real fit segment against the poles --
            c = _lls_amplitudes(fit_u, fit_t, poles, t_ref=fit_t[0])
            # -- gate: extrapolate BACKWARD over the held-out window --------
            pred = _model_eval(holdout_t, poles, c, t_ref=fit_t[0])
            denom = float(np.sqrt(np.mean(holdout_u**2)))
            if denom == 0.0:
                reason = "held-out window is identically zero"
            else:
                residual = float(
                    np.sqrt(np.mean((pred - holdout_u) ** 2)) / denom)
                if residual > self.residual_tol:
                    reason = (
                        f"held-out residual {residual:.3g} exceeds "
                        f"residual_tol {self.residual_tol:g}")

        if reason:
            if self.strict:
                raise CompletionRejected(reason)
            warnings.warn(
                "spectrum completion rejected — returning the truncated "
                f"spectrum unchanged: {reason}", stacklevel=2)
            return self._package(
                f_arr, truncated, np.zeros_like(truncated), None,
                accepted=False, residual=residual, reason=reason,
                fit_start=t[i_a], holdout_end=t[i_h - 1], record_end=t[-1])

        tail = _discrete_tail(poles, c, t_ref=fit_t[0], t_last=t[-1],
                              delta=dt, freqs=f_arr)
        return self._package(
            f_arr, truncated, tail, (poles, c),
            accepted=True, residual=residual, reason="",
            fit_start=t[i_a], holdout_end=t[i_h - 1], record_end=t[-1])

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _package(freqs, truncated, tail, model, *, accepted, residual,
                 reason, fit_start, holdout_end, record_end) -> xr.Dataset:
        data = {
            "spectrum": ("freq", truncated + tail),
            "truncated": ("freq", truncated),
            "tail": ("freq", tail),
        }
        if model is not None:
            poles, c = model
            data |= {
                "mode_freq": ("mode", -np.imag(poles) / (2 * np.pi)),
                "mode_decay": ("mode", -np.real(poles)),
                "mode_q": ("mode",
                           np.pi * np.abs(np.imag(poles) / (2 * np.pi))
                           / -np.real(poles)),
                "mode_amplitude": ("mode", np.abs(c)),
            }
        ds = xr.Dataset(data, coords={"freq": np.asarray(freqs, dtype=float)})
        ds.attrs.update(
            accepted=int(accepted),
            holdout_residual=float(residual),
            fit_start=float(fit_start),
            holdout_end=float(holdout_end),
            record_end=float(record_end),
            reject_reason=reason,
        )
        return ds


# -- numerics ----------------------------------------------------------------


def _rectangle_dft(u: np.ndarray, t: np.ndarray, delta: float,
                   freqs: np.ndarray) -> np.ndarray:
    """§12-style rectangle-rule DFT over the recorded samples.

    ``F(f) = delta * sum_n u(t_n) e^{+2 pi i f t_n}`` — the e^{+i w t} phasor
    convention of the engine's running DFT, evaluated at the record's own
    sample times. Chunked over frequencies to bound the outer-product memory.
    """
    out = np.empty(freqs.size, dtype=complex)
    for lo in range(0, freqs.size, 64):
        f_blk = freqs[lo:lo + 64]
        out[lo:lo + 64] = (np.exp(2j * np.pi * np.outer(f_blk, t)) @ u) * delta
    return out


def _lls_amplitudes(u: np.ndarray, t: np.ndarray, poles: np.ndarray,
                    t_ref: float) -> np.ndarray:
    """Complex amplitudes ``c_k`` minimizing ``|u - Re sum c_k e^{s_k (t-tr)}|``.

    Linear in (Re c, Im c): with ``E = e^{s_k (t - t_ref)}``,
    ``Re[c E] = Re(E) Re(c) - Im(E) Im(c)`` — a real design matrix
    ``[Re E | -Im E]`` solved by ``lstsq``.
    """
    e = np.exp(np.outer(t - t_ref, poles))
    design = np.hstack([np.real(e), -np.imag(e)])
    sol, *_ = np.linalg.lstsq(design, u, rcond=None)
    k = poles.size
    return sol[:k] + 1j * sol[k:]


def _model_eval(t: np.ndarray, poles: np.ndarray, c: np.ndarray,
                t_ref: float) -> np.ndarray:
    """The fitted real ringdown model at times ``t``."""
    return np.real(np.exp(np.outer(t - t_ref, poles)) @ c)


def _discrete_tail(poles: np.ndarray, c: np.ndarray, *, t_ref: float,
                   t_last: float, delta: float,
                   freqs: np.ndarray) -> np.ndarray:
    """Closed form of the DFT sum over every UNRECORDED sample.

    The remainder ``delta * sum_{m>=1} u(t_last + m delta) e^{2 pi i f t}``
    of the rectangle-rule DFT, with ``u`` the fitted model. Writing
    ``Re[z] = (z + conj z)/2`` splits each mode into its two half-amplitude
    complex poles, and each is a geometric series in
    ``r = e^{(s + 2 pi i f) delta}`` with ``|r| < 1`` (every pole entering
    here decays):

        tail(f) = delta * e^{2 pi i f t_last} * sum_k [
                      (c'_k / 2) r_k / (1 - r_k)
                    + (conj(c'_k) / 2) q_k / (1 - q_k) ],

    with ``c'_k = c_k e^{s_k (t_last - t_ref)}`` (the amplitude walked
    forward to the last recorded sample), ``r_k`` from ``s_k`` and ``q_k``
    from ``conj(s_k)``. Matching the engine's quadrature exactly — a sum,
    not the continuous integral — means the completed spectrum converges to
    the engine's own infinite-run value with no quadrature offset.
    """
    c_at_end = c * np.exp(poles * (t_last - t_ref))
    out = np.zeros(freqs.size, dtype=complex)
    for s_k, ck in zip(poles, c_at_end):
        for s, amp in ((s_k, ck / 2.0), (np.conj(s_k), np.conj(ck) / 2.0)):
            r = np.exp((s + 2j * np.pi * freqs) * delta)
            out += amp * r / (1.0 - r)
    return delta * np.exp(2j * np.pi * freqs * t_last) * out
