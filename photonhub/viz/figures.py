"""Presentation layer — plotly figure dicts over the :mod:`service` data core.

The HTTP facade serves these; the MCP/agent path never imports this module. All
heavy slicing/geometry lives in ``service``; here we only shape numbers into
ready-to-render plotly specs.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import _style, service

_C = 299_792_458.0
_MARGIN = {"l": 60, "r": 20, "t": 40, "b": 50}
# Written names for the reduction the UI offers, so a figure title reads the
# same as the control that produced it ("Magnitude", not "abs").
_VAL_LABELS = {"real": "Real", "imag": "Imaginary", "abs": "Magnitude",
               "phase": "Phase, rad"}


def _val_label(val: str) -> str:
    return _VAL_LABELS.get(val, val)


def _field_label(field: str, val: str) -> str:
    """Field with its value flavor — except for the derived magnitude-like
    fields (E, H, intensity), where the forced "(Real)" suffix would only
    mislead: intensity is not the real part of anything."""
    if field in ("E", "H", "intensity"):
        return field
    return f"{field} ({_val_label(val)})"


def _json_safe_numbers(values) -> list:
    """Convert numeric arrays to strict-JSON lists, mapping NaN/inf to null.

    Partial/aborted result bundles are intentionally inspectable and may carry
    non-finite samples.  Starlette's strict JSON encoder rejects NaN, so a
    single diverged cell must not turn the entire viewer endpoint into a 500.
    Plotly renders ``null`` as a gap, which preserves every finite sample.
    """
    a = np.asarray(values)
    if not np.issubdtype(a.dtype, np.number):
        return a.tolist()
    out = a.astype(object)
    out[~np.isfinite(a)] = None
    return out.tolist()


def field_figure(data, monitor: str, *, field: str = "Ex", val: str = "real",
                 freq: Optional[float] = None, time: Optional[float] = None,
                 axis: Optional[str] = None, pos: Optional[float] = None,
                 cmap: Optional[str] = None, structures: bool = True) -> dict:
    """A 2D field cut-plane as a plotly heatmap, with structure outlines overlaid."""
    arr, z, row, col, resolved = service.slice_plane(
        data, monitor, field=field, val=val, freq=freq, time=time, axis=axis, pos=pos)
    z = np.asarray(z, dtype=float)
    # Signed components -> symmetric RdBu about 0. The derived magnitude-like
    # fields (E, H, intensity) ride val="real" but are non-negative — a
    # diverging scale would wash them into uniform gray.
    diverging = val in ("real", "imag") and field not in ("E", "H", "intensity")
    # Defaults follow the design-§7 constants the CLI matplotlib path uses:
    # magma for magnitudes, cyclic twilight for phase, RdBu for signed parts.
    if cmap:
        colorscale = _mpl_colorscale(cmap)
    elif diverging:
        colorscale = "RdBu"
    elif val == "phase":
        colorscale = _mpl_colorscale(_style._PHASE_CMAP)
    else:
        colorscale = _mpl_colorscale(_style._MAGNITUDE_CMAP)
    trace: dict[str, Any] = {
        "type": "heatmap",
        "z": _json_safe_numbers(z),
        "x": _json_safe_numbers(arr.coords[col].values),
        "y": _json_safe_numbers(arr.coords[row].values),
        "colorscale": colorscale,
        # DFT phasors are source-normalized (A0·S(f)); raw time snapshots keep
        # solver field units with the source's arbitrary drive amplitude.
        "colorbar": {"title": f"{_field_label(field, val)}<br>{'source-normalized' if 'freq_hz' in resolved else 'arb. units'}"},
    }
    if diverging:
        finite = np.abs(z[np.isfinite(z)])
        amax = float(finite.max()) if finite.size else 1.0
        trace["zmin"], trace["zmax"] = -(amax or 1.0), (amax or 1.0)

    traces = [trace]
    cut = resolved.get("cut")
    if structures and cut:
        outlines = service.structure_outlines(service.sim_for(data), cut["axis"], cut["value_um"])
        traces += _overlay_traces(outlines, col, row)

    title = f"{monitor} · {_field_label(field, val)}"
    if "freq_hz" in resolved:
        title += f" · {_C / resolved['freq_hz'] * 1e9:.0f} nm"
    if "time_s" in resolved:
        title += f" · t={resolved['time_s'] * 1e15:.0f} fs"
    return {
        "data": traces,
        "layout": {
            "title": title,
            "xaxis": {"title": f"{col} (µm)", "constrain": "domain"},
            "yaxis": {"title": f"{row} (µm)", "scaleanchor": "x", "scaleratio": 1},
            "margin": _MARGIN, "showlegend": False,
        },
    }


def _overlay_traces(outlines: Optional[dict], col: str, row: str) -> list:
    """Map structure-outline point loops into the heatmap's (x=col, y=row) frame."""
    if not outlines or not outlines.get("loops"):
        return []
    h_axis, v_axis = outlines["h_axis"], outlines["v_axis"]

    def xy(a: float, b: float):
        c = {h_axis: a, v_axis: b}
        return c.get(col), c.get(row)

    out = []
    for loop in outlines["loops"]:
        xs, ys = [], []
        for a, b in loop:
            x, y = xy(a, b)
            xs.append(x)
            ys.append(y)
        out.append({
            "type": "scatter", "mode": "lines", "x": xs, "y": ys,
            "line": {"color": "rgba(255,255,255,0.85)", "width": 1.3},
            "hoverinfo": "skip", "showlegend": False,
        })
    return out


def spectrum_figure(data, monitor: str) -> dict:
    """A flux monitor's signed, source-normalized value vs wavelength."""
    s = service.spectrum_values(data, monitor)
    return {
        "data": [{"type": "scatter", "mode": "lines+markers",
                  "x": _json_safe_numbers(s["wavelength_nm"]),
                  "y": _json_safe_numbers(s["value"]), "name": monitor}],
        "layout": {"title": f"{monitor} · signed source-normalized flux",
                   "xaxis": {"title": "wavelength (nm)"},
                   "yaxis": {"title": "signed normalized flux"}, "margin": _MARGIN},
    }


def profile_figure(data, monitor: str, *, field: str = "Ex", val: str = "real",
                   freq: Optional[float] = None, time: Optional[float] = None) -> dict:
    """A rank-1 field profile versus its one varying spatial coordinate."""
    p = service.line_profile_values(
        data, monitor, field=field, val=val, freq=freq, time=time)
    title = f"{monitor} · {_field_label(field, val)}"
    if "freq_hz" in p:
        title += f" · {_C / p['freq_hz'] * 1e9:.3f} nm"
    if "time_s" in p:
        title += f" · {p['time_s'] * 1e15:.3f} fs"
    return {
        "data": [{"type": "scatter", "mode": "lines",
                  "x": _json_safe_numbers(p["coord_um"]),
                  "y": _json_safe_numbers(p["value"]), "name": field}],
        "layout": {"title": title,
                   "xaxis": {"title": f"{p['axis']} (µm)"},
                   "yaxis": {"title": f"{_field_label(field, val)}"}, "margin": _MARGIN},
    }


def field_spectrum_figure(data, monitor: str, *, field: str = "Ex",
                          val: str = "abs") -> dict:
    """A rank-0 DFT field sample versus wavelength."""
    s = service.field_spectrum_values(data, monitor, field=field, val=val)
    return {
        "data": [{"type": "scatter", "mode": "lines+markers",
                  "x": _json_safe_numbers(s["wavelength_nm"]),
                  "y": _json_safe_numbers(s["value"]), "name": field}],
        "layout": {"title": f"{monitor} · {_field_label(field, val)} spectrum",
                   "xaxis": {"title": "wavelength (nm)"},
                   "yaxis": {"title": f"{_field_label(field, val)} · source-normalized"}, "margin": _MARGIN},
    }


def timeseries_figure(data, monitor: str, *, field: str = "Ex", val: str = "real") -> dict:
    """A point/time monitor's component vs time as a plotly line."""
    s = service.timeseries_values(data, monitor, field=field, val=val)
    return {
        "data": [{"type": "scatter", "mode": "lines",
                  "x": _json_safe_numbers([t * 1e15 for t in s["time_s"]]),
                  "y": _json_safe_numbers(s["value"]), "name": field}],
        "layout": {"title": f"{monitor} · {field} ({_val_label(val)})", "xaxis": {"title": "time (fs)"},
                   # Raw Yee samples in solver units with an arbitrary source
                   # drive amplitude — never absolute V/m.
                   "yaxis": {"title": f"{field} ({_val_label(val)}) · arb. units"}, "margin": _MARGIN},
    }


def timeseries_fft_figure(data, monitor: str, *, field: str = "Ex") -> dict:
    """Power spectrum (dB re peak) of a recorded time series vs frequency.

    A frequency axis keeps the uniform rfft bins readable (a wavelength axis
    squashes the optical band against the near-DC bins); hover shows the
    wavelength via customdata.  The initial x-range zooms to the band within
    40 dB of the peak (full-span autoscale crams the optical band against a
    Nyquist axis thousands of THz wide); the full spectrum stays in the trace
    for zoom-out."""
    s = service.timeseries_fft_values(data, monitor, field=field)
    significant = [f for f, db in zip(s["freq_hz"], s["psd_db"]) if db >= -40.0]
    lo, hi = min(significant), max(significant)
    pad = max(0.15 * (hi - lo), 3.0 * s["resolution_hz"])
    x_range = [max(0.0, lo - pad) / 1e12, (hi + pad) / 1e12]
    return {
        "data": [{"type": "scatter", "mode": "lines",
                  "x": _json_safe_numbers([f / 1e12 for f in s["freq_hz"]]),
                  "y": _json_safe_numbers(s["psd_db"]), "name": field,
                  "customdata": _json_safe_numbers(s["wavelength_nm"]),
                  "hovertemplate": ("%{x:.2f} THz · %{customdata:.1f} nm"
                                    "<br>%{y:.1f} dB<extra></extra>")}],
        "layout": {"title": (f"{monitor} · {field} · spectrum of the "
                             "recorded window"),
                   "xaxis": {"title": "frequency (THz)", "range": x_range},
                   "yaxis": {"title": "|FFT|² (dB re peak)"},
                   "margin": _MARGIN},
    }


def scene_figure_from_sim(sim) -> dict:
    """The 3D geometry of a ``Simulation`` spec as a plotly figure dict."""
    return sim.plot_3d().to_plotly_json()


def scene_figure(data) -> dict:
    """The 3D geometry as a plotly figure dict, reconstructed from sim.json."""
    sim = service.sim_for(data)
    if sim is None:
        raise FileNotFoundError("no sim.json next to the result bundle — 3D scene unavailable")
    return scene_figure_from_sim(sim)


_COLORSCALES: dict = {}


def _mpl_colorscale(name: str):
    """A colormap as explicit plotly stops sampled from matplotlib. Plotly.js
    ships a small built-in list whose same-named scales can differ (its
    "Blues" is a legacy gray→indigo ramp, and it has no "Magma"/"Plasma" at
    all), so naming scales would desync the app from the CLI matplotlib
    renders that share ``_style``'s constants. Unknown names pass through
    unchanged for plotly to interpret."""
    if name in _COLORSCALES:
        return _COLORSCALES[name]
    import matplotlib as mpl
    cmap = mpl.colormaps.get(name) or mpl.colormaps.get(name.lower())
    if cmap is None:
        return name
    stops = []
    for i in range(9):
        t = i / 8
        r, g, b, _a = cmap(t)
        stops.append([round(t, 3), f"rgb({round(r * 255)},{round(g * 255)},{round(b * 255)})"])
    _COLORSCALES[name] = stops
    return stops


def _eps_fig(e: dict, axis: str, value: float) -> dict:
    """Shape an ``eps_plane`` result into a plotly heatmap."""
    return {
        "data": [{"type": "heatmap", "z": e["eps"], "x": e["h_um"], "y": e["v_um"],
                  # Structure drawing, not continuous data: a soft single-hue
                  # ramp (low ε ≈ paper, high ε = ink) reads as geometry and
                  # stops competing with the Viridis field maps.
                  "colorscale": _mpl_colorscale(_style.EPS_CMAP),
                  "colorbar": {"title": "ε"}}],
        "layout": {
            "title": f"permittivity · {axis} = {value:.2f} µm",
            "xaxis": {"title": f"{e['h_axis']} (µm)", "constrain": "domain"},
            "yaxis": {"title": f"{e['v_axis']} (µm)", "scaleanchor": "x", "scaleratio": 1},
            "margin": _MARGIN,
        },
    }


def eps_figure(data, axis: str = "z", value: float = 0.0) -> dict:
    """Permittivity cross-section of a result bundle — verify geometry/materials."""
    return _eps_fig(service.eps_plane(data, axis, value), axis, value)


def eps_figure_from_sim(sim, axis: str = "z", value: float = 0.0) -> dict:
    """Permittivity cross-section of a live ``Simulation`` spec (preview)."""
    return _eps_fig(service.eps_plane_sim(sim, axis, value), axis, value)
