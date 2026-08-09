"""Resonance / Q-factor extraction from FDTD time signals (harmonic inversion).

A resonant structure (ring, photonic-crystal cavity, Fabry-Perot, ...) excited by
a short pulse rings down as a sum of decaying complex exponentials,

.. math::

    f(t) = \\sum_k a_k e^{i \\phi_k} e^{-2\\pi i f_k t - \\alpha_k t},

one term per resonant mode. :class:`ResonanceFinder` recovers the per-mode
frequency :math:`f_k`, decay rate :math:`\\alpha_k`, quality factor
:math:`Q_k = \\pi |f_k| / \\alpha_k`, amplitude :math:`a_k` and phase
:math:`\\phi_k` from a :class:`~photonhub.components.FieldTimeMonitor` time series.

This is a CPU-only post-processing plugin (the plugin contract in
``photonhub.plugins``): it consumes the raw ``E(t)`` already recorded by a
``FieldTimeMonitor`` and runs nowhere near the engine or the wire format.

Algorithm
---------
The filter-diagonalization method (FDM) / harmonic inversion of

* V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time signals and
  its applications," J. Chem. Phys. **107**, 6756 (1997);
* M. R. Wall and D. Neuhauser, "Extraction, through filter-diagonalization, of
  general quantum eigenvalues ... of a short-time segment of a signal,"
  J. Chem. Phys. **102**, 8001 (1995).

The signal is recast as the autocorrelation of a fictitious evolution operator;
its complex eigenvalues :math:`u_k = e^{(-2\\pi i f_k - \\alpha_k)\\Delta t}` are
the per-step resonance poles. They are found by solving a generalized eigenvalue
problem :math:`U_1 B = u\\, U_0 B` in a small basis seeded on an even grid of
``init_num_freqs`` trial frequencies across ``freq_window`` -- so a *short*,
not-yet-decayed signal still yields sharp resonances (the property a spectral
FWHM fit lacks). This is an independent implementation; it is validated against
Tidy3D's ``ResonanceFinder`` (same algorithm) and an independent matrix-pencil
oracle in the test suite.

Example
-------
>>> import numpy as np
>>> from photonhub.plugins import ResonanceFinder, select_resonances
>>> dt = 1.0
>>> t = np.arange(8000) * dt
>>> sig = 2.0 * np.exp((-2j*np.pi*0.10 - 0.002) * t) \\
...     + 3.0 * np.exp((-2j*np.pi*0.20 - 0.0005) * t)
>>> rf = ResonanceFinder(freq_window=(0.05, 0.25))
>>> modes = select_resonances(rf.run_raw_signal(sig, dt), min_amplitude=0.1)
>>> round(float(modes["Q"].sel(freq=0.10, method="nearest")))  # pi*0.10/0.002 = 157.08
157  # doctest: +SKIP

After a real run::

    data = ph.run_local(sim)                       # sim has a FieldTimeMonitor "probe"
    rf = ResonanceFinder(freq_window=(1.9e14, 2.0e14))
    resonances = rf.run(data, "probe")             # xr.Dataset over 'freq'
    modes = select_resonances(resonances, min_amplitude=1e-3)
"""

from __future__ import annotations

import warnings
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

__all__ = ["ResonanceFinder", "select_resonances"]

_ELECTRIC = ("Ex", "Ey", "Ez")
_MAGNETIC = ("Hx", "Hy", "Hz")
_TIME_STEP_RTOL = 1e-5
_MIN_SAMPLES = 8  # need half_len = n//2 - 2 >= 2 for the U-matrix sums


class ResonanceFinder:
    """Extract resonances (f, decay, Q, amplitude, phase) from a time signal.

    Parameters
    ----------
    freq_window : (float, float)
        ``(f_min, f_max)`` in Hz. The solver is seeded with an even grid of
        trial frequencies in this window; it then iteratively optimizes and
        prunes them. A narrow window around a few resonances is more accurate
        than a broad one. Resonances slightly outside the window may still be
        returned.
    init_num_freqs : int, default 200
        Number of trial frequencies. Larger finds more (and fainter)
        resonances at higher cost; the number returned is typically smaller.
    rcond : float, default 1e-4
        Relative cutoff for the generalized-eigenproblem conditioning: singular
        values of the overlap matrix below ``rcond * max`` are dropped. Closer
        to zero returns more (and more spurious) resonances.

    Notes
    -----
    The input signal must be **uniformly sampled** and should begin *after the
    sources have turned off* -- the source-injection transient is not part of
    the ring-down model and degrades the fit.
    """

    def __init__(
        self,
        freq_window: Tuple[float, float],
        init_num_freqs: int = 200,
        rcond: float = 1e-4,
    ):
        f_min, f_max = float(freq_window[0]), float(freq_window[1])
        if f_max < f_min:
            raise ValueError(
                f"freq_window must be (f_min, f_max) with f_max >= f_min; got "
                f"({f_min}, {f_max})"
            )
        if int(init_num_freqs) < 1:
            raise ValueError(f"init_num_freqs must be >= 1; got {init_num_freqs}")
        if float(rcond) < 0:
            raise ValueError(f"rcond must be >= 0; got {rcond}")
        self.freq_window: Tuple[float, float] = (f_min, f_max)
        self.init_num_freqs: int = int(init_num_freqs)
        self.rcond: float = float(rcond)

    # -- public entry points -----------------------------------------------

    def run_raw_signal(
        self, signal: Sequence[complex], dt: float
    ) -> xr.Dataset:
        """Find resonances in a 1-D time series sampled every ``dt`` seconds.

        Parameters
        ----------
        signal : array-like, shape (n_samples,)
            Real or complex time series (an FDTD field probe is real). Must be
            uniformly sampled and start after the sources are off.
        dt : float
            Sample spacing in seconds.

        Returns
        -------
        xarray.Dataset
            Indexed by ``freq`` (Hz), with data variables ``decay`` (1/s),
            ``Q``, ``amplitude``, ``phase`` (rad) and ``error`` (a rough
            per-mode reliability estimate). Sorted by ``freq``. The raw set may
            include faint or spurious modes -- use :func:`select_resonances` to
            keep the physical ones.
        """
        sig = np.asarray(signal)
        if sig.ndim != 1:
            raise ValueError(f"signal must be 1-D; got shape {sig.shape}")
        if sig.size < _MIN_SAMPLES:
            raise ValueError(
                f"signal has {sig.size} samples; need >= {_MIN_SAMPLES} for the "
                "filter-diagonalization sums"
            )
        sig = sig.astype(complex)
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be a positive finite number; got {dt}")
        if not np.all(np.isfinite(sig)):
            raise ValueError("signal contains non-finite values")

        f_min, f_max = self.freq_window
        nyquist = 0.5 / dt
        if f_max > nyquist:
            warnings.warn(
                f"freq_window max {f_max:g} Hz exceeds the Nyquist frequency "
                f"{nyquist:g} Hz (1/2dt); resonances above Nyquist cannot be "
                "resolved -- sample the monitor more often (smaller "
                "interval_steps) or narrow the window",
                stacklevel=2,
            )

        # Seed trial poles on the unit circle at evenly spaced frequencies.
        n = self.init_num_freqs
        omega_dt = np.linspace(2 * np.pi * f_min * dt, 2 * np.pi * f_max * dt, n)
        eigvals = np.exp(-1j * omega_dt)

        # First refinement, then iterate until the surviving-mode count settles.
        eigvals, amplitudes, errors = _iterate(sig, eigvals, self.rcond)
        prev_count = len(eigvals)
        for _ in range(1 + 2 * n):
            eigvals, amplitudes, errors = _iterate(sig, eigvals, self.rcond)
            if len(eigvals) == prev_count:
                break
            prev_count = len(eigvals)

        return _resonance_dataset(eigvals, amplitudes, errors, dt)

    def run_time_series(self, dataarray: xr.DataArray) -> xr.Dataset:
        """Find resonances in a ``FieldTimeMonitor`` :class:`xarray.DataArray`.

        ``dt`` is derived from the ``t`` coordinate (which must be uniformly
        spaced). If the array has a ``component`` dimension, the electric
        components are summed, falling back to the magnetic ones if no E field
        is present (the Tidy3D convention -- never E and H mixed); pass a
        single-component array to control this exactly.
        """
        signal, dt = _signal_from_dataarray(dataarray, fields=None)
        return self.run_raw_signal(signal, dt)

    def run(
        self,
        sim_data: Mapping[str, xr.DataArray],
        monitors: Union[str, Sequence[str]],
        fields: Optional[Sequence[str]] = None,
    ) -> xr.Dataset:
        """Find resonances in one or more ``FieldTimeMonitor`` outputs.

        Parameters
        ----------
        sim_data : SimulationData or mapping ``name -> DataArray``
            The run output (anything indexable by monitor name).
        monitors : str or sequence of str
            Monitor name(s) to read. Multiple monitors are summed into one
            signal (more independent probes improve conditioning).
        fields : sequence of str, optional
            Field components to use (e.g. ``["Ez"]``). Default: sum the electric
            components present, falling back to the magnetic ones -- the Tidy3D
            convention.

        Returns
        -------
        xarray.Dataset
            Same schema as :meth:`run_raw_signal`.
        """
        names = [monitors] if isinstance(monitors, str) else list(monitors)
        if not names:
            raise ValueError("monitors must name at least one monitor")
        arrays = [sim_data[name] for name in names]
        signal, dt = _combine_time_series(arrays, fields)
        return self.run_raw_signal(signal, dt)


# -- result selection -------------------------------------------------------


def select_resonances(
    resonances: xr.Dataset,
    *,
    freq_window: Optional[Tuple[float, float]] = None,
    min_amplitude: Optional[float] = None,
    min_q: Optional[float] = None,
    max_error: Optional[float] = None,
    require_decay: bool = True,
    sort_by: Optional[str] = "Q",
) -> xr.Dataset:
    """Filter and rank a raw resonance :class:`xarray.Dataset`.

    The filter-diagonalization output deliberately includes faint and spurious
    modes (low amplitude, high ``error``, or non-physical negative decay). This
    keeps the physical ones.

    Parameters
    ----------
    resonances : xarray.Dataset
        Output of :meth:`ResonanceFinder.run_raw_signal` / ``run``.
    freq_window : (float, float), optional
        Keep only resonances with ``f_min <= freq <= f_max``.
    min_amplitude, min_q, max_error : float, optional
        Lower bound on amplitude / Q, upper bound on ``error``.
    require_decay : bool, default True
        Drop non-physical modes with ``decay <= 0`` (growing in time).
    sort_by : str or None, default "Q"
        Data variable to sort by, descending (e.g. ``"Q"`` or ``"amplitude"``).
        ``None`` leaves the ``freq`` ordering.

    Returns
    -------
    xarray.Dataset
        The filtered (and optionally ranked) subset.
    """
    freq = resonances.coords["freq"].values
    mask = np.ones(freq.shape, dtype=bool)
    if freq_window is not None:
        if freq_window[1] < freq_window[0]:
            raise ValueError(
                f"freq_window must be (f_min, f_max) with f_max >= f_min; "
                f"got {tuple(freq_window)}"
            )
        mask &= (freq >= freq_window[0]) & (freq <= freq_window[1])
    if require_decay:
        mask &= resonances["decay"].values > 0
    if min_amplitude is not None:
        mask &= resonances["amplitude"].values >= min_amplitude
    if min_q is not None:
        mask &= resonances["Q"].values >= min_q
    if max_error is not None:
        mask &= resonances["error"].values <= max_error

    out = resonances.isel(freq=np.flatnonzero(mask))
    if sort_by is not None and out.sizes["freq"] > 0:
        if sort_by not in out:
            raise KeyError(
                f"sort_by={sort_by!r} is not a data variable; choose from "
                f"{list(out.data_vars)}"
            )
        vals = out[sort_by].values
        # NaN (e.g. Q of a zero-decay pole) is least trustworthy -> rank last.
        order = np.argsort(np.where(np.isnan(vals), -np.inf, vals))[::-1]
        out = out.isel(freq=order)
    return out


# -- FDM core (one refinement iteration) ------------------------------------


def _iterate(signal: np.ndarray, basis_eigvals: np.ndarray, rcond: float):
    """One filter-diagonalization refinement.

    Builds the U-matrices in the basis seeded at ``basis_eigvals``, solves the
    generalized eigenproblem ``U1 B = u U0 B`` for the refined poles ``u``, then
    their amplitudes and a per-pole error estimate. Returns
    ``(eigvals, amplitudes, errors)``.
    """
    u_matrices = _evaluate_u_matrices(signal, basis_eigvals)
    eigvals, eigvecs = _solve_complex_symmetric_gevp(
        u_matrices[1], u_matrices[0], rcond
    )
    errors = _eigenvalue_errors(eigvals, u_matrices, eigvecs)
    amplitudes = _amplitudes(signal, basis_eigvals, eigvecs)
    return eigvals, amplitudes, errors


def _evaluate_u_matrices(signal: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
    """The three FDM evolution matrices ``U^(0), U^(1), U^(2)`` (shape (3,n,n)).

    Closed-form Mandelshtam-Taylor matrix elements: off-diagonals from the
    geometric-sum identity, diagonals from the triangular-weighted limit.
    Products use the *bilinear* (non-conjugated) inner product intrinsic to FDM.
    """
    n = len(eigvals)
    half = len(signal) // 2 - 2
    if half < 1:
        raise ValueError("signal too short for the FDM evolution matrices")
    # Trial poles projected onto the unit circle; z_pow[j, m] = z_j^{-m}.
    z = eigvals / np.abs(eigvals)
    z_pow = np.exp(1j * np.outer(np.log(z) * 1j, np.arange(2 * half + 1)))

    prefactor = np.zeros((n, n), dtype=complex)
    off_diag = ~np.eye(n, dtype=bool)
    prefactor[off_diag] = np.reciprocal(
        np.subtract.outer(z, z)[off_diag]
    )

    # Triangular weights [1,2,..,half+1,half,..,1] for the diagonal limit.
    tri = np.concatenate((np.arange(1, half + 2), np.arange(half, 0, -1)))

    u_matrices = np.zeros((3, n, n), dtype=complex)
    for p in range(3):
        sp = signal[p:]
        head = z_pow[:, : half + 1] @ sp[: half + 1]
        tail = z_pow[:, :half] @ sp[half + 1 : 2 * half + 1]
        upper = np.outer(z, head) + np.outer(tail, z_pow[:, half])
        u_p = prefactor * upper
        u_p = u_p + u_p.T  # symmetrize (off-diagonals; diagonal is still 0)
        diag = (z_pow[:, : 2 * half + 1] * (tri * sp[: 2 * half + 1])).sum(axis=1)
        np.fill_diagonal(u_p, diag)
        u_matrices[p] = u_p
    return u_matrices


def _solve_complex_symmetric_gevp(
    a_matrix: np.ndarray, b_matrix: np.ndarray, rcond: float
):
    """Solve ``A v = u B v`` for a complex-symmetric pencil (Wall-Neuhauser).

    ``B`` is regularized by dropping eigenvalues below ``rcond * max`` and the
    problem is reduced to a standard one in the surviving subspace. Returns
    ``(eigvals, eigvecs)`` with eigvecs in the original basis.
    """
    import scipy.linalg

    b_vals, b_vecs = scipy.linalg.eig(b_matrix)
    keep = np.abs(b_vals) > rcond * np.amax(np.abs(b_vals))
    b_vals = b_vals[keep]
    b_vecs = _gram_schmidt_bilinear(b_vecs[:, keep])

    change_basis = b_vecs @ np.diag(1.0 / np.sqrt(b_vals))
    reduced = change_basis.T @ a_matrix @ change_basis
    a_vals, a_vecs = scipy.linalg.eig(reduced)
    a_vecs = _gram_schmidt_bilinear(a_vecs)

    return a_vals, change_basis @ a_vecs


def _gram_schmidt_bilinear(a_matrix: np.ndarray) -> np.ndarray:
    """Gram-Schmidt over columns under the bilinear form ``u . v`` (no conjugate).

    FDM's overlap is the complex-symmetric ``sum(u*v)``, not the Hermitian
    ``sum(conj(u)*v)`` -- using the wrong one silently corrupts the amplitudes.
    """
    out = np.zeros(a_matrix.shape, dtype=complex)
    for i in range(out.shape[1]):
        out[:, i] = a_matrix[:, i]
        for j in range(i):
            out[:, i] -= out[:, j] * np.dot(out[:, j], a_matrix[:, i])
        out[:, i] /= np.sqrt(np.dot(out[:, i], out[:, i]))
    return out


def _amplitudes(
    signal: np.ndarray, basis_eigvals: np.ndarray, eigvecs: np.ndarray
) -> np.ndarray:
    """Complex amplitudes ``d_k = a_k e^{i phi_k}`` from the eigenvector overlaps."""
    half = len(signal) // 2 - 2
    z = basis_eigvals / np.abs(basis_eigvals)
    z_pow = np.exp(1j * np.outer(np.log(z) * 1j, np.arange(half + 1)))
    overlap = z_pow @ signal[: half + 1]
    return np.array(
        [np.square(eigvecs[:, k] @ overlap) for k in range(eigvecs.shape[1])],
        dtype=complex,
    )


def _eigenvalue_errors(
    eigvals: np.ndarray, u_matrices: np.ndarray, eigvecs: np.ndarray
) -> np.ndarray:
    """Per-pole reliability estimate ``||(U2 - u^2 U0) v||`` (small => trustworthy)."""
    errors = np.zeros(len(eigvals))
    for k in range(len(eigvals)):
        residual = (u_matrices[2] - (eigvals[k] ** 2) * u_matrices[0]) @ eigvecs[:, k]
        errors[k] = np.linalg.norm(residual)
    return errors


def _resonance_dataset(
    eigvals: np.ndarray,
    amplitudes: np.ndarray,
    errors: np.ndarray,
    dt: float,
) -> xr.Dataset:
    """Convert per-step poles into a physical resonance :class:`xarray.Dataset`.

    For ``u = e^{(-2 pi i f - alpha) dt}``: ``f = Re(i ln u / 2pi)/dt`` and
    ``alpha = -Im(i ln u)/dt``, giving ``Q = pi |f| / alpha``.
    """
    complex_omega = 1j * np.log(eigvals)
    freqs = np.real(complex_omega / (2 * np.pi)) / dt
    decays = -np.imag(complex_omega) / dt
    with np.errstate(divide="ignore", invalid="ignore"):
        q_factors = np.pi * np.abs(freqs) / decays
    coords = {"freq": freqs}
    ds = xr.Dataset(
        {
            "decay": ("freq", decays),
            "Q": ("freq", q_factors),
            "amplitude": ("freq", np.abs(amplitudes)),
            "phase": ("freq", np.angle(amplitudes)),
            "error": ("freq", np.asarray(errors, dtype=float)),
        },
        coords=coords,
    )
    return ds.sortby("freq")


# -- time-series ingestion --------------------------------------------------


def _uniform_dt(t: np.ndarray) -> float:
    """Sample spacing of a uniformly-spaced time coordinate (validated)."""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        raise ValueError("time coordinate needs at least two samples")
    diffs = np.diff(t)
    dt = float(diffs[0])
    if dt <= 0 or not np.allclose(diffs, dt, rtol=_TIME_STEP_RTOL, atol=0):
        raise ValueError(
            "time coordinate must be uniformly spaced to extract resonances"
        )
    return dt


def _components_of(da: xr.DataArray) -> list:
    if "component" in da.dims:
        return [str(c) for c in np.asarray(da.coords["component"].values)]
    return []


def _select_components(present, fields: Optional[Sequence[str]]):
    """Field components to sum: explicit ``fields`` if given, else the electric
    components present, falling back to the magnetic ones (the Tidy3D
    convention) -- never E and H mixed, which is physically meaningless. Returns
    ``None`` when there are no recognized components (a component-less array)."""
    if fields is not None:
        return list(fields)
    present = set(present)
    if present & set(_ELECTRIC):
        return [c for c in _ELECTRIC if c in present]
    if present & set(_MAGNETIC):
        return [c for c in _MAGNETIC if c in present]
    return None


def _signal_from_dataarray(
    da: xr.DataArray, fields: Optional[Sequence[str]]
) -> Tuple[np.ndarray, float]:
    """One signal + dt from a single time-domain DataArray (dims ``('t',)`` or
    ``('t', 'component')``)."""
    if "t" not in da.dims:
        raise ValueError(
            f"DataArray has no time dimension 't' (dims={da.dims}); resonance "
            "extraction needs a FieldTimeMonitor output"
        )
    extra = set(da.dims) - {"t", "component"}
    if extra:
        raise ValueError(
            f"DataArray has non-time dimensions {sorted(extra)}; resonance "
            "extraction needs a point FieldTimeMonitor (dims ('t',) or "
            "('t', 'component')), not a snapshot/DFT monitor"
        )
    dt = _uniform_dt(da.coords["t"].values)
    comps = _components_of(da)
    if not comps:
        if fields is not None:
            raise ValueError(
                f"fields={list(fields)} requested but the array has no "
                "'component' dimension to select from"
            )
        return np.asarray(da.values, dtype=complex).ravel(), dt
    chosen = _select_components(comps, fields)
    total = None
    for comp in chosen or comps:
        if comp not in comps:
            continue
        series = np.asarray(da.sel(component=comp).values, dtype=complex).ravel()
        total = series if total is None else total + series
    if total is None:
        raise ValueError(
            f"none of the requested fields {chosen} are in the monitor "
            f"(available: {comps})"
        )
    return total, dt


def _combine_time_series(
    arrays: Sequence[xr.DataArray], fields: Optional[Sequence[str]]
) -> Tuple[np.ndarray, float]:
    """Sum the chosen components across several monitors into one signal + dt.

    Components are chosen once across all monitors (explicit ``fields`` or the
    electric-then-magnetic fallback) so every monitor contributes consistently
    -- matching :func:`_signal_from_dataarray`."""
    present = set()
    for da in arrays:
        present |= set(_components_of(da))
    chosen = _select_components(present, fields)

    total = None
    dt_ref = None
    length = None
    for da in arrays:
        signal, dt = _signal_from_dataarray(da, chosen)
        if dt_ref is None:
            dt_ref, length = dt, signal.size
        elif abs(dt - dt_ref) > _TIME_STEP_RTOL * dt_ref:
            raise ValueError("monitors have different time steps")
        elif signal.size != length:
            raise ValueError("monitors have different numbers of samples")
        total = signal if total is None else total + signal
    return total, dt_ref


# -- independent oracle (matrix pencil / ESPRIT) ----------------------------


def _matrix_pencil_poles(
    signal: Sequence[complex], dt: float, n_poles: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Frequencies and decays via matrix-pencil / ESPRIT (independent of FDM).

    A Hankel matrix of the signal is SVD-truncated to ``n_poles`` and the
    poles are read off the shift-invariance of the signal subspace. Used only
    as an independent cross-check in the tests -- not a shipped public method.
    Returns ``(freqs, decays)`` with the same sign convention as the FDM path.
    """
    import scipy.linalg

    x = np.asarray(signal, dtype=complex).ravel()
    n = x.size
    rows = n // 2
    ncol = n - rows + 1
    hankel = np.empty((rows, ncol), dtype=complex)
    for i in range(rows):
        hankel[i] = x[i : i + ncol]

    u_left, _, _ = np.linalg.svd(hankel, full_matrices=False)
    p = int(min(n_poles, u_left.shape[1], rows - 1))
    subspace = u_left[:, :p]
    shift = np.linalg.pinv(subspace[:-1]) @ subspace[1:]
    poles = scipy.linalg.eigvals(shift)

    complex_omega = 1j * np.log(poles)
    freqs = np.real(complex_omega / (2 * np.pi)) / dt
    decays = -np.imag(complex_omega) / dt
    return freqs, decays
