"""``plot_source_time()`` — 1D preview of a source's drive: the injected
current ``J(t)`` (Gaussian envelope × carrier) and its spectral envelope.

Faithful to the engine: the time waveform and the spectral envelope are the
exact analytic forms the :class:`~photonhub.components.source_time.GaussianPulse`
model exposes (``tau_s`` / ``t0_s`` / ``source_end_time_s`` /
``spectral_amplitude``), so the preview matches what the solver injects. Returns
the time-domain matplotlib ``Axes`` (sharing a figure with the spectrum panel
unless an explicit ``ax=`` is given). Never calls ``plt.show()``.
"""

import numpy as np

# carrier / envelope / spectrum colors (kept distinct from the ε/field maps).
_J_COLOR = "#1f77b4"
_ENV_COLOR = "#999999"
_SPEC_COLOR = "#d62728"


def plot_source_time(st, *, ax=None, num: int = 512):
    """Plot the source drive ``J(t)`` (with its Gaussian envelope) and, when
    building its own figure, the spectral envelope ``|J(f)|`` alongside.

    ``ax=`` draws only the time-domain trace on the given Axes; otherwise a
    1×2 (time | spectrum) figure is created. Returns the time-domain ``Axes``."""
    import matplotlib.pyplot as plt

    tau, t0 = st.tau_s, st.t0_s
    f0, fw = st.freq0_hz, st.fwidth_hz
    phase = getattr(st, "phase", 0.0)

    t = np.linspace(0.0, st.source_end_time_s, num)
    env = np.exp(-((t - t0) ** 2) / (2.0 * tau * tau))
    j = env * np.cos(2.0 * np.pi * f0 * (t - t0) + phase)

    axf = None
    if ax is None:
        _, (ax, axf) = plt.subplots(1, 2, figsize=(10.0, 3.6))

    t_fs = t * 1e15
    ax.plot(t_fs, j, color=_J_COLOR, lw=1.0, label="J(t)")
    ax.plot(t_fs, env, color=_ENV_COLOR, lw=1.0, ls="--", label="envelope")
    ax.plot(t_fs, -env, color=_ENV_COLOR, lw=1.0, ls="--")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("amplitude (norm.)")
    ax.set_title("source time")
    ax.legend(loc="upper right", fontsize="small")

    if axf is not None:
        f = np.linspace(max(0.0, f0 - 4.0 * fw), f0 + 4.0 * fw, num)
        spec = np.exp(-0.5 * ((f - f0) / fw) ** 2)
        f_thz = f * 1e-12
        axf.plot(f_thz, spec, color=_SPEC_COLOR, lw=1.2)
        axf.axvline(f0 * 1e-12, color=_ENV_COLOR, ls=":", lw=0.8, label="freq0")
        axf.set_xlabel("frequency (THz)")
        axf.set_ylabel("|J(f)| (norm.)")
        axf.set_title("spectrum")
        axf.legend(loc="upper right", fontsize="small")

    return ax
