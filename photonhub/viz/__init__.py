"""Plot PhotonHub simulation geometry, fields, modes, and spectra.

The rendering engine behind ``Simulation.plot``/``plot_index``/``plot_3d`` and
``RunResult.plot_field``. All 2D methods return a matplotlib ``Axes``,
accept ``ax=``, and never call ``plt.show()``; ``plot_3d`` returns a plotly
``Figure`` (the optional ``photonhub[viz]`` extra, lazy-imported).

``plot_mode`` (an FDE mode's transverse field), ``plot_spectrum``
(transmission ``T(λ)`` from the mode-monitor pipeline) and ``plot_comparison``
(an observable against a paper's extracted series) are module-level helpers
only, a ``Mode`` is not a ``Simulation`` and a spectrum comes from
post-processing, so neither maps cleanly onto a model method.

These helpers can render without a graphical application or event loop.
"""

from ._style import sharp_inline_figures
from .eps import (
                          plot_eps,  # deprecated alias
                          plot_index,
)
from .featured import Scene, export_scene
from .field import plot_field
from .interactive import (
                          interactive_field,
                          interactive_preview,
                          render_field_slice,
                          render_slice,
)
from .mode import plot_mode, plot_overlap
from .scene import plot
from .scene3d import plot_3d
from .source import plot_source_time
from .spectrum import plot_comparison, plot_spectrum

sharp_inline_figures()

__all__ = [
                          "Scene",
                          "export_scene",
                          "interactive_field",
                          "interactive_preview",
                          "plot",
                          "plot_3d",
                          "plot_comparison",
                          "plot_field",
                          "plot_index",
                          "plot_mode",
                          "plot_overlap",
                          "plot_source_time",
                          "plot_spectrum",
                          "render_field_slice",
                          "render_slice",
    "sharp_inline_figures",
]
