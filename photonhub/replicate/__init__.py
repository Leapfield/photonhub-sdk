"""Paper-replication workflow — take a published photonic device (a
:class:`PaperSpec`) and reproduce it faithfully: build the geometry from
parameters, assemble a converged FDTD simulation, read the figures-of-merit, and
compare them against the paper.

This is the paper-agnostic front end of the agentic designer: the legacy
``benchmarks/gds`` framework was specialized to one paper (hardcoded stack,
monkey-patched global scene builder); this package rebuilds the pieces as
installable, spec-driven modules so any paper is expressible.

    from photonhub.replicate import PaperSpec, build_geometry
    spec = PaperSpec.from_yaml("specs/chandran_cosine_crossing.yaml")
"""

from __future__ import annotations

from . import geometry
from .convergence import (
    ConvergenceReport,
    Rung,
    auto_converge,
    converge_through_transmission,
)
from .driver import ReplicationResult, replicate
from .geometry import REGISTRY, build_geometry, geometry_builder, register
from .notebook import generate_notebook
from .report import build_markdown_report, compare_rows
from .spec import (
    Convergence,
    Device,
    Layer,
    Optical,
    PaperSpec,
    PortRoles,
    Reference,
    Source,
    SpecError,
    Stack,
)

__all__ = [
    "PaperSpec",
    "Source",
    "Layer",
    "Stack",
    "Optical",
    "PortRoles",
    "Reference",
    "Device",
    "Convergence",
    "SpecError",
    "geometry",
    "build_geometry",
    "register",
    "geometry_builder",
    "REGISTRY",
    "auto_converge",
    "converge_through_transmission",
    "ConvergenceReport",
    "Rung",
    "replicate",
    "ReplicationResult",
    "build_markdown_report",
    "compare_rows",
    "generate_notebook",
]
