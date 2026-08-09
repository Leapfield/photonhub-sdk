"""PhotonHub visualization layer (design doc: docs/viz-layer-design.md).

The rendering engine behind ``Simulation.plot``/``plot_eps``/``plot_3d`` and
``SimulationData.plot_field``. All 2D methods return a matplotlib ``Axes``,
accept ``ax=``, and never call ``plt.show()``; ``plot_3d`` returns a plotly
``Figure`` (the optional ``photonhub[viz]`` extra, lazy-imported).

``plot_mode`` (an FDE mode's transverse field) and ``plot_spectrum``
(transmission ``T(λ)`` from the mode-monitor pipeline) are module-level helpers
only — a ``Mode`` is not a ``Simulation`` and a spectrum comes from
post-processing, so neither maps cleanly onto a model method.

No UI/event-loop assumptions live here, so the future standalone viewer and the
Phase-4 GUI can call these headless (design §12).
"""

from .eps import plot_eps
from .field import plot_field
from .interactive import (interactive_field, interactive_preview,
                          render_field_slice, render_slice)
from .mode import plot_mode, plot_overlap
from .scene import plot
from .scene3d import plot_3d
from .source import plot_source_time
from .spectrum import plot_spectrum

__all__ = ["plot", "plot_eps", "plot_field", "plot_3d", "plot_mode",
           "plot_overlap", "plot_source_time", "plot_spectrum",
           "interactive_preview", "render_slice", "interactive_field",
           "render_field_slice"]
