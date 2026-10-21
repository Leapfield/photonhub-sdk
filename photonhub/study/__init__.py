"""Design studies: parameter sweeps, process corners, Monte Carlo and yield.

Describe the parameters, pick a sampler, give a ``build(params) ->
Simulation`` function and, optionally, a ``measure(result) -> {name: float}``
function; :func:`run` executes the unique simulations through the local
:class:`~photonhub.Batch` or, on request, :class:`photonhub.cloud.Batch`, and
returns one table row per sample.

>>> from photonhub import study
>>> width = study.Span("width_um", 0.40, 0.50, sigma=0.01)
>>> samples = study.monte_carlo([width], num_samples=50, seed=1)
"""

from .parameters import Choices, Span
from ._execute import StudyPlan, plan, run
from .sampling import (
    Samples,
    corners,
    grid,
    latin_hypercube,
    monte_carlo,
    points,
)
from .table import YieldEstimate, load, save, yield_fraction

__all__ = [
    "Span",
    "Choices",
    "Samples",
    "grid",
    "points",
    "corners",
    "monte_carlo",
    "latin_hypercube",
    "save",
    "load",
    "yield_fraction",
    "YieldEstimate",
    "StudyPlan",
    "plan",
    "run",
]
