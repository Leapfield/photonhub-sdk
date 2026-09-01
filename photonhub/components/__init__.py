"""Pydantic models defining the simulation JSON wire format.

These models are the single source of truth for the schema
(schemas/GOVERNANCE.md); ``schemas/simulation_v1.json`` is generated from
them via ``python -m photonhub.schema emit``.
"""

from .base import BoundaryKind, FieldComponentName, FrozenModel
from .grid import (
    GradedAxisCoords,
    GradedGridSpec,
    GridSpecType,
    MeshOverride,
    UniformGridSpec,
    auto_grid,
)
from .medium import Background, Boundaries
from .monitors import (
    Apodization,
    FieldDftMonitor,
    FieldSnapshotMonitor,
    FieldTimeMonitor,
    FluxMonitor,
    ModePort,
    MonitorType,
    PortMode,
)
from .run import RunSpec
from .simulation import SCHEMA_VERSION, Simulation
from .source_time import CW, GaussianPulse, SourceTimeType
from .sources import (
    ModeSolveProvenance,
    ModeSource,
    PlaneWave,
    TfsfBox,
    PointDipole,
    SourceType,
)
from .structures import (
    Box,
    Cylinder,
    GeometryType,
    DrudePole,
    LorentzPole,
    Medium,
    PermittivityData,
    PolySlab,
    Sphere,
    Structure,
)

__all__ = [
    "Background",
    "Boundaries",
    "BoundaryKind",
    "Apodization",
    "Box",
    "Cylinder",
    "FieldComponentName",
    "FieldDftMonitor",
    "FieldSnapshotMonitor",
    "FieldTimeMonitor",
    "FluxMonitor",
    "FrozenModel",
    "CW",
    "GaussianPulse",
    "GeometryType",
    "GradedAxisCoords",
    "GradedGridSpec",
    "MeshOverride",
    "GridSpecType",
    "DrudePole",
    "LorentzPole",
    "Medium",
    "PermittivityData",
    "ModeSolveProvenance",
    "ModePort",
    "ModeSource",
    "MonitorType",
    "PlaneWave",
    "TfsfBox",
    "PointDipole",
    "PortMode",
    "PolySlab",
    "RunSpec",
    "SCHEMA_VERSION",
    "Simulation",
    "SourceTimeType",
    "SourceType",
    "Sphere",
    "Structure",
    "UniformGridSpec",
    "auto_grid",
]
