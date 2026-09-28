"""``plot_spectrum()``, transmission ``T(λ)`` from the mode-monitor pipeline, and ``plot_comparison()``, an observable against the paper's extracted series.

Plots the power-transmission spectrum that
:func:`photonhub.analysis.transmission` /
:meth:`photonhub.analysis.ModeMonitor.mode_power` produce. It accepts either:

- a single ``{freq_hz: T}`` mapping, one trace, or
- a ``{label: {freq_hz: T}}`` mapping, several labelled traces (e.g. a
  coupler's through/cross ports), drawn with a legend.

Frequencies are converted to free-space wavelength (``λ_nm = c / f``,
``c = 2.99792458e8 m/s``) and each trace is plotted T-vs-λ(nm), sorted by
ascending wavelength. ``y`` spans ``[0, ~1.05]`` with a grid. Dependency-light
(numpy + matplotlib); returns the matplotlib ``Axes`` and never calls
``plt.show()``.
"""

import numpy as np

from ..constants import c0

#: Speed of light in vacuum (m/s) — matches the plugins' C0 so λ round-trips.
_C0 = c0


def _is_dataarray_spectrum(value) -> bool:
    """True for a 1-D labelled array over ``f`` (what
    :func:`photonhub.analysis.transmission_spectrum` returns)."""
    return (hasattr(value, "dims") and hasattr(value, "coords")
            and tuple(value.dims) == ("f",) and "f" in value.coords)


def _is_spectrum(value) -> bool:
    """True for a single spectrum: a ``{freq_hz: T}`` mapping (numeric keys +
    values) or a 1-D labelled array over ``f``; False for a ``{label: {...}}``
    mapping of named traces."""
    if _is_dataarray_spectrum(value):
        return True
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(v, (int, float)) for v in value.values())


def _trace_xy(spectrum):
    """``(wavelength_nm, T)`` arrays for one ``{freq_hz: T}`` mapping, sorted by
    ascending wavelength."""
    if _is_dataarray_spectrum(spectrum):
        freqs = np.asarray(spectrum.coords["f"].values, dtype=np.float64)
        tvals = np.asarray(spectrum.values, dtype=np.float64)
    else:
        freqs = np.asarray(list(spectrum.keys()), dtype=np.float64)
        tvals = np.asarray(list(spectrum.values()), dtype=np.float64)
    lam_nm = _C0 / freqs * 1e9
    order = np.argsort(lam_nm)
    return lam_nm[order], tvals[order]


def plot_spectrum(spectra, *, ax=None, ymax=1.05, **kw):
    """Plot transmission ``T`` versus wavelength (nm).

    ``spectra`` is a single spectrum (a ``{freq_hz: T}`` mapping or the
    labelled array :func:`photonhub.analysis.transmission_spectrum` returns)
    for one trace, or a ``{label: spectrum}`` mapping (one labelled trace per
    key, with a legend). ``ax=`` draws into an existing Axes; ``ymax`` caps the y-axis
    (default ~1.05). Extra ``**kw`` pass through to ``ax.plot``. Returns the
    matplotlib ``Axes``."""
    import matplotlib.pyplot as plt

    if not _is_dataarray_spectrum(spectra) and (not isinstance(spectra, dict) or not spectra):
        raise ValueError(
            "spectra must be a non-empty {freq_hz: T} dict, a labelled array over "
            "f, or a {label: spectrum} mapping"
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
                f"trace {label!r} is not a {{freq_hz: T}} mapping of numbers "
                "or a labelled array over f"
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


# Fixed colours, so a series keeps its colour from one figure to the next:
# our series first, then the paper's extracted series, then model curves.
_OURS = ("C0", "C2", "C4", "C5")
_PAPER = ("C1", "C3", "C6", "C7")
_MODEL = ("C8", "C9")
# Stated values: grey lines told apart by their dash pattern, and hollow
# square points at their own x.
_STATED_STYLES = (("--", "0.30"), (":", "0.30"), ("-.", "0.45"), ((0, (6, 2, 1, 2, 1, 2)), "0.45"))
#: Fewer points than this and our series is drawn with markers too: a sweep of
#: discrete runs, not a sampled curve.
MARKER_POINTS = 12


def _xy(x, y, what: str):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError(f"{what}: x and y must be non-empty 1-D arrays of the same length")
    return x, y


def _own_series(x_nm, values, label):
    """``[(label, x, y)]`` for our series: one array, a transmission spectrum,
    or a ``{label: y}`` / ``{label: (x, y)}`` mapping of several."""
    if values is None and _is_dataarray_spectrum(x_nm):
        # a transmission_spectrum(): its wlen_um coordinate is the x axis
        lam = np.asarray(x_nm.coords["wlen_um"].values, dtype=np.float64) * 1e3
        order = np.argsort(lam)
        return [(label, lam[order], np.asarray(x_nm.values, dtype=np.float64)[order])]
    if isinstance(values, dict):
        if not values:
            raise ValueError("values is an empty mapping; give at least one series")
        out = []
        for name, series in values.items():
            if _is_dataarray_spectrum(series):          # a transmission_spectrum() of its own
                lam = np.asarray(series.coords["wlen_um"].values, dtype=np.float64) * 1e3
                order = np.argsort(lam)
                sx, sy = lam[order], np.asarray(series.values, dtype=np.float64)[order]
            elif isinstance(series, tuple) and len(series) == 2:
                sx, sy = series
            else:
                sx, sy = x_nm, series
            out.append((str(name), *_xy(sx, sy, f"series {name!r}")))
        return out
    try:
        return [(label, *_xy(x_nm, values, "x_nm and values"))]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "x_nm and values must be non-empty 1-D arrays of the same length"
        ) from exc


def _entries(value) -> list:
    """One ``(..., label)`` tuple, or a list of them, as a list."""
    if value is None:
        return []
    if (isinstance(value, (tuple, list)) and value and isinstance(value[-1], str)
            and not isinstance(value[0], (tuple, list))):
        return [value]
    return list(value)


def plot_comparison(x_nm, values=None, *, reference=None, reference_scale=1.0,
                    stated=None, model=None, markers=None, ylabel="",
                    xlabel="wavelength (nm)", label="PhotonHub", ax=None, ylim=None,
                    yscale=None, **kw):
    """The result figure of an example: our observable as a line, the paper's
    extracted series as markers, the paper's stated values as grey dashed
    lines or points, and a model as a coloured dashed line.

    ``x_nm`` / ``values`` are our curve, in whatever unit the observable has
    (dB, a ratio, a Q). A spectrum from
    :func:`photonhub.analysis.transmission_spectrum` may be passed alone as
    ``x_nm`` and plots against its own wavelength coordinate. ``values`` may
    also be a mapping ``{label: y}`` (several of our series over ``x_nm``) or
    ``{label: (x, y)}`` (each over its own x) or ``{label: spectrum}``: each
    keeps a fixed colour in the order given, so the same series has the same
    colour in every figure (a single series takes the axes' next colour; a
    line drawn by hand after the mapping form does not advance past it).
    A series of fewer than ``MARKER_POINTS`` points (a sweep of discrete runs)
    is drawn with markers as well; ``markers=True`` or ``False`` forces it.
    ``label`` names a single series.

    ``reference`` is one series or a list of series from a reference file,
    ``{"x": [...], "y": [...], "label": "..."}``, drawn as markers, each ``y``
    multiplied by ``reference_scale`` (``-1`` turns a digitized transmittance
    in dB into an insertion loss). ``stated`` draws the paper's *stated*
    values, the form a page uses when the paper's license does not allow its
    figure to be re-plotted: ``(value, label)`` is a horizontal line (several
    are told apart by their dash pattern), ``(x, y, label)`` a point at its
    own x (a stated Q at a stated radius). Either form, or a list mixing
    them. ``model`` is ``(x, y, label)`` or a list of them: an analytic curve
    as a coloured dashed line.

    ``ylim`` fixes the y-range so the agreement is visible. ``yscale`` sets
    the axis scale, ``"linear"`` or ``"log"`` (the latter for an observable
    that spans decades, a resonator Q); left at ``None`` the axes keep the
    scale they have, so a log ``ax`` passed in, or one a first call set to
    log, stays log. When every x of our series is a whole number (a count of
    segments or periods) the ticks are whole numbers too. The figure has no
    title: the paragraph above it is its caption. Returns the matplotlib
    ``Axes`` and never calls ``plt.show()``."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if yscale is not None and str(yscale) not in ("linear", "log"):
        raise ValueError(f"yscale must be 'linear' or 'log'; got {yscale!r}")
    ours = _own_series(x_nm, values, label)
    if reference is None:
        refs = []
    elif isinstance(reference, dict):
        refs = [reference]
    else:
        refs = list(reference)

    if ax is None:
        _, ax = plt.subplots()
    # The {label: ...} form fixes each series' colour by its place; a single
    # series leaves colours to the axes' cycle, so a caller's next line on the
    # same axes (or a second call with ax=) still gets the next colour.
    fixed = isinstance(values, dict)
    for i, (name, x, y) in enumerate(ours):
        style = {"color": _OURS[i % len(_OURS)]} if fixed else {}
        if markers or (markers is None and x.size < MARKER_POINTS):
            style.update(marker="o", ms=5)
        ax.plot(x, y, "-", label=name, **{**style, **kw})
    for i, ref in enumerate(refs):
        try:
            rx = np.asarray(ref["x"], dtype=np.float64)
            ry = np.asarray(ref["y"], dtype=np.float64) * float(reference_scale)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "each reference series needs numeric 'x' (nm) and 'y' arrays"
            ) from exc
        if rx.shape != ry.shape:
            raise ValueError("a reference series' x and y differ in length")
        ax.plot(rx, ry, "o", ms=3, label=str(ref.get("label", "paper")),
                **({"color": _PAPER[i % len(_PAPER)]} if fixed else {}))
    lines = points = 0
    for entry in _entries(stated):
        if len(entry) == 2:
            value, text = entry
            ls, colour = _STATED_STYLES[lines % len(_STATED_STYLES)]
            ax.axhline(float(value), ls=ls, lw=1.2, color=colour, label=str(text))
            lines += 1
        elif len(entry) == 3:
            px, py, text = entry
            ax.plot([float(px)], [float(py)], "s", ms=7, mfc="none", mew=1.4,
                    color=("0.15", "0.45")[points % 2], label=str(text))
            points += 1
        else:
            raise ValueError("a stated entry is (value, label) or (x, y, label)")
    for i, entry in enumerate(_entries(model)):
        try:
            mx, my, text = entry
        except ValueError as exc:
            raise ValueError("a model entry is (x, y, label)") from exc
        mx, my = _xy(mx, my, f"model {text!r}")
        ax.plot(mx, my, "--", lw=1.4, color=_MODEL[i % len(_MODEL)], label=str(text))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if all(np.all(np.isclose(x, np.round(x), rtol=0.0, atol=1e-9)) for _, x, _ in ours):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if yscale is not None:
        ax.set_yscale(str(yscale))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small", framealpha=0.9)
    return ax
