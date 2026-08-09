"""``interactive_preview()`` — a Jupyter scrubber for a :class:`Simulation`:
slide the cut plane through the domain and watch the cross-section update live,
with cut-axis / grid / scene-vs-ε toggles. The "see it before you run it" moment,
interactive but fully local (no web app).

ipywidgets is the optional ``photonhub[viz]`` extra and is imported LAZILY, so the
core SDK never depends on it. The frame rendering (:func:`render_slice`) is a
plain matplotlib call — pure and testable headless; ``interactive_preview`` just
wires it to widgets.
"""

_NO_IPYWIDGETS = (
    "interactive_preview requires ipywidgets (the interactive viz extra) and a "
    "Jupyter/IPython environment. Install it with:\n    pip install photonhub[viz]"
)


def _require_ipywidgets():
    try:
        import ipywidgets as widgets  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without ipywidgets
        raise ImportError(_NO_IPYWIDGETS) from exc
    return widgets


def render_slice(sim, axis: str, value: float, *, eps: bool = True,
                 grid: bool = True, ax=None):
    """Render one preview frame: the meshed-ε heatmap (``eps=True``) or the
    analytic scene (``eps=False``) at ``axis=value``, optionally with the grid
    overlay. Returns the matplotlib ``Axes`` (pure — no widgets)."""
    if eps:
        return sim.plot_eps(**{axis: value}, ax=ax, grid=grid)
    return sim.plot(**{axis: value}, ax=ax, grid=grid)


def interactive_preview(sim, *, axis: str = "z", eps: bool = True,
                        grid: bool = True):
    """Build a Jupyter widget that scrubs the cut plane through ``sim``.

    A cut-axis dropdown, a position slider (auto-ranged to the realized domain),
    and ε / grid toggles drive a live cross-section. Returns the ipywidgets
    container, which displays inline as a cell's last expression. Requires the
    ``photonhub[viz]`` extra and a notebook."""
    widgets = _require_ipywidgets()
    import matplotlib.pyplot as plt

    realized = sim._realized_um()

    def _axis_max(a: str) -> float:
        return float(realized["xyz".index(a)])

    m0 = _axis_max(axis)
    axis_dd = widgets.Dropdown(options=["x", "y", "z"], value=axis,
                               description="cut")
    pos = widgets.FloatSlider(min=0.0, max=m0, value=m0 / 2.0, step=m0 / 100.0,
                              description="µm", continuous_update=False,
                              readout_format=".3f")
    eps_cb = widgets.Checkbox(value=eps, description="meshed ε")
    grid_cb = widgets.Checkbox(value=grid, description="grid")
    out = widgets.Output()

    def render(*_):
        from IPython.display import display
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            render_slice(sim, axis_dd.value, pos.value,
                         eps=eps_cb.value, grid=grid_cb.value, ax=ax)
            display(fig)        # inline-backend friendly; no Agg "cannot show" warning
            plt.close(fig)

    def on_axis(_change):
        m = _axis_max(axis_dd.value)
        pos.max = m
        pos.step = m / 100.0
        pos.value = m / 2.0
        render()

    axis_dd.observe(on_axis, names="value")
    for w in (pos, eps_cb, grid_cb):
        w.observe(render, names="value")

    render()
    return widgets.VBox([widgets.HBox([axis_dd, pos]),
                         widgets.HBox([eps_cb, grid_cb]), out])


def render_field_slice(data, monitor, *, field="Ex", val="real", freq=None,
                       time=None, axis=None, value=None, structures=False,
                       simulation=None, ax=None):
    """Render one field-preview frame via :func:`plot_field` — the pure,
    testable core of :func:`interactive_field`. ``axis``/``value`` pick the cut
    plane for a volumetric monitor; ``freq=``/``time=`` pick the DFT frequency /
    time frame. Omit the cut for an already-planar monitor."""
    from .field import plot_field
    cut = {axis: value} if axis is not None and value is not None else {}
    return plot_field(data, monitor, field=field, val=val, freq=freq, time=time,
                      structures=structures, simulation=simulation, ax=ax, **cut)


def _field_component_options(da):
    """Selectable field options for a recorded array: its stored components plus
    the derived magnitudes (``E``/``intensity`` when all of Ex/Ey/Ez are
    present, ``H`` when all of Hx/Hy/Hz are)."""
    comps = ([str(c) for c in da.coords["component"].values]
             if "component" in da.coords else [])
    extra = []
    if {"Ex", "Ey", "Ez"} <= set(comps):
        extra += ["E", "intensity"]
    if {"Hx", "Hy", "Hz"} <= set(comps):
        extra += ["H"]
    return comps + extra


def interactive_field(data, monitor, *, simulation=None):
    """A Jupyter scrubber over a recorded field monitor: field component,
    real/imag/abs/phase, frequency (DFT monitors) or a time slider + Play button
    for field-propagation animation (time/snapshot monitors), and (for a
    volumetric monitor) the cut plane, with optional structure outlines. Returns
    the ipywidgets container. Requires the ``photonhub[viz]`` extra and a notebook."""
    widgets = _require_ipywidgets()
    import matplotlib.pyplot as plt

    da = data[monitor]
    comps = _field_component_options(da) or ["Ex"]
    field_dd = widgets.Dropdown(options=comps, value=comps[0], description="field")
    val_dd = widgets.Dropdown(options=["real", "imag", "abs", "phase"],
                              value="real", description="part")
    rows = [widgets.HBox([field_dd, val_dd])]

    freq_w = None
    if "f" in da.coords and da.sizes.get("f", 1) > 1:
        freqs = [float(f) for f in da.coords["f"].values]
        freq_w = widgets.SelectionSlider(
            options=[(f"{f * 1e-12:.1f} THz", f) for f in freqs], value=freqs[0],
            description="freq")

    # Time/snapshot monitor -> a time slider + a Play button (field-propagation
    # animation). Play scrubs the slider via a jslink on its index.
    time_w = play = None
    if "t" in da.coords and da.sizes.get("t", 1) > 1:
        times = [float(t) for t in da.coords["t"].values]
        time_w = widgets.SelectionSlider(
            options=[(f"{t * 1e15:.1f} fs", t) for t in times], value=times[0],
            description="time")
        play = widgets.Play(min=0, max=len(times) - 1, step=1, interval=200)
        widgets.jslink((play, "value"), (time_w, "index"))

    spatial = [a for a in ("x", "y", "z") if a in da.dims]
    thick = [a for a in spatial if da.sizes[a] > 1]
    singleton = [a for a in spatial if da.sizes[a] == 1]
    volumetric = len(thick) == 3

    def _pos_opts(a):
        return [(f"{float(c):.3f}", float(c)) for c in da.coords[a].values]

    axis_dd = pos = None
    if volumetric:
        axis_dd = widgets.Dropdown(options=thick, value=thick[-1], description="cut")
        pos = widgets.SelectionSlider(options=_pos_opts(axis_dd.value),
                                      description="µm")

    sample_row = [w for w in (freq_w, play, time_w) if w is not None]
    cut_row = [w for w in (axis_dd, pos) if w is not None]
    for r in (sample_row, cut_row):
        if r:
            rows.append(widgets.HBox(r))

    struct_cb = None
    if simulation is not None:
        struct_cb = widgets.Checkbox(value=True, description="structures")
        rows.append(struct_cb)

    out = widgets.Output()

    def render(*_):
        from IPython.display import display
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            if volumetric:
                cut = {"axis": axis_dd.value, "value": pos.value}
            elif singleton:
                cut = {"axis": singleton[0],
                       "value": float(da.coords[singleton[0]].values[0])}
            else:
                cut = {}
            render_field_slice(
                data, monitor, field=field_dd.value, val=val_dd.value,
                freq=(freq_w.value if freq_w is not None else None),
                time=(time_w.value if time_w is not None else None),
                structures=(struct_cb.value if struct_cb is not None else False),
                simulation=simulation, ax=ax, **cut)
            display(fig)
            plt.close(fig)

    def on_axis(_change):
        pos.options = _pos_opts(axis_dd.value)
        render()

    if volumetric:
        axis_dd.observe(on_axis, names="value")
    for w in (field_dd, val_dd, freq_w, time_w, pos, struct_cb):
        if w is not None:
            w.observe(render, names="value")

    render()
    return widgets.VBox(rows + [out])
