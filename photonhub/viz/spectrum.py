"""``plot_spectrum()`` — transmission ``T(λ)`` from the mode-monitor pipeline.

Plots the power-transmission spectrum that
:func:`photonhub.plugins.transmission` /
:meth:`photonhub.plugins.ModeMonitor.mode_power` produce. It accepts either:

- a single ``{freq_hz: T}`` mapping — one trace, or
- a ``{label: {freq_hz: T}}`` mapping — several labelled traces (e.g. a
  coupler's through/cross ports), drawn with a legend.

Frequencies are converted to free-space wavelength (``λ_nm = c / f``,
``c = 2.99792458e8 m/s``) and each trace is plotted T-vs-λ(nm), sorted by
ascending wavelength. ``y`` spans ``[0, ~1.05]`` with a grid. Dependency-light
(numpy + matplotlib); returns the matplotlib ``Axes`` and never calls
``plt.show()``.
"""

import numpy as np

#: Speed of light in vacuum (m/s) — matches the plugins' C0 so λ round-trips.
_C0 = 2.99792458e8


def _is_spectrum(value) -> bool:
    """True for a single ``{freq_hz: T}`` mapping (numeric keys + values),
    False for a ``{label: {...}}`` mapping of named traces."""
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(v, (int, float)) for v in value.values())


def _trace_xy(spectrum):
    """``(wavelength_nm, T)`` arrays for one ``{freq_hz: T}`` mapping, sorted by
    ascending wavelength."""
    freqs = np.asarray(list(spectrum.keys()), dtype=np.float64)
    tvals = np.asarray(list(spectrum.values()), dtype=np.float64)
    lam_nm = _C0 / freqs * 1e9
    order = np.argsort(lam_nm)
    return lam_nm[order], tvals[order]


def plot_spectrum(spectra, *, ax=None, ymax=1.05, **kw):
    """Plot transmission ``T`` versus wavelength (nm).

    ``spectra`` is either a single ``{freq_hz: T}`` mapping (one trace) or a
    ``{label: {freq_hz: T}}`` mapping (one labelled trace per key, with a
    legend). ``ax=`` draws into an existing Axes; ``ymax`` caps the y-axis
    (default ~1.05). Extra ``**kw`` pass through to ``ax.plot``. Returns the
    matplotlib ``Axes``."""
    import matplotlib.pyplot as plt

    if not isinstance(spectra, dict) or not spectra:
        raise ValueError(
            "spectra must be a non-empty {freq_hz: T} dict or a "
            "{label: {freq_hz: T}} mapping"
        )

    if ax is None:
        _, ax = plt.subplots()

    if _is_spectrum(spectra):
        traces = {None: spectra}
    else:
        traces = spectra

    drew_label = False
    for label, spectrum in traces.items():
        if not _is_spectrum(spectrum):
            raise ValueError(
                f"trace {label!r} is not a {{freq_hz: T}} mapping of numbers"
            )
        lam_nm, tvals = _trace_xy(spectrum)
        ax.plot(lam_nm, tvals, label=(str(label) if label is not None else None),
                **kw)
        drew_label = drew_label or label is not None

    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmission T")
    ax.set_ylim(0.0, ymax)
    ax.grid(True, alpha=0.3)
    if drew_label:
        ax.legend(loc="best", fontsize="small", framealpha=0.9)
    return ax
