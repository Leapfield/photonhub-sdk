"""Pydantic models defining the simulation JSON wire format.

These models are the single source of truth for the schema
(schemas/GOVERNANCE.md); ``schemas/simulation_v1.json`` is generated from
them via ``python -m photonhub.schema emit``.
"""

from .base import BoundaryKind, FieldComponentName, FrozenModel
from .grid import (
    GradedMeshAxis,
    GradedMesh,
    MeshType,
    MeshOverride,
    UniformMesh,
    auto_mesh,
)
from .medium import Background, Boundaries
from .monitors import (
    Apodization,
    ProfileMonitor,
    SnapshotMonitor,
    TimeMonitor,
    PowerMonitor,
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
    PermittivityArray,
    Polygon,
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
    "ProfileMonitor",
    "SnapshotMonitor",
    "TimeMonitor",
    "PowerMonitor",
    "FrozenModel",
    "CW",
    "GaussianPulse",
    "GeometryType",
    "GradedMeshAxis",
    "GradedMesh",
    "MeshOverride",
    "MeshType",
    "DrudePole",
    "LorentzPole",
    "Medium",
    "PermittivityArray",
    "ModeSolveProvenance",
    "ModePort",
    "ModeSource",
    "MonitorType",
    "PlaneWave",
    "TfsfBox",
    "PointDipole",
    "PortMode",
    "Polygon",
    "RunSpec",
    "SCHEMA_VERSION",
    "Simulation",
    "SourceTimeType",
    "SourceType",
    "Sphere",
    "Structure",
    "UniformMesh",
    "auto_mesh",
]
