"""PhotonHub Python client — build simulation specs, run the solver, load results.

>>> import photonhub as ph
>>> sim = ph.Simulation(...)
>>> data = ph.run_local(sim)
>>> probe = data["probe"]  # xarray.DataArray
"""

from .components import (
    SCHEMA_VERSION,
    Apodization,
    Background,
    Boundaries,
    Box,
    Cylinder,
    ProfileMonitor,
    SnapshotMonitor,
    TimeMonitor,
    PowerMonitor,
    CW,
    GaussianPulse,
    DrudePole,
    LorentzPole,
    Medium,
    PermittivityArray,
    ModePort,
    ModeSolveProvenance,
    ModeSource,
    PlaneWave,
    TfsfBox,
    PointDipole,
    PortMode,
    Polygon,
    RunSpec,
    Simulation,
    Sphere,
    Structure,
    GradedMeshAxis,
    GradedMesh,
    MeshOverride,
    UniformMesh,
    auto_mesh,
)
from . import library
from . import materials
from .cost import CostEstimate, quote
from .data import RunResult
from .gds import GdsLayer, export_gds, import_gds, read_gds_cell_names
from .hdf5 import convert_to_hdf5
from .runners import (
    Batch,
    BatchResults,
    Job,
    SolverRunError,
    find_solver,
    submit,
    run_local,
)
from . import cloud
from .cloud import gpus  # convenience: ph.gpus() == ph.cloud.gpus() (the GPU menu)
from . import inverse_design
from .inverse_design import (
    DesignRegion,
    GradientResult,
    ModePower,
    OptimizeResult,
    ParametricResult,
    PointIntensity,
    assemble_gradient,
    optimize,
    optimize_parametric,
    value_and_gradient,
)

__version__ = "0.1.2"

__all__ = [
    "Apodization",
    "Background",
    "Batch",
    "BatchResults",
    "Boundaries",
    "Box",
    "CostEstimate",
    "Cylinder",
    "ProfileMonitor",
    "SnapshotMonitor",
    "TimeMonitor",
    "PowerMonitor",
    "CW",
    "GaussianPulse",
    "GdsLayer",
    "import_gds",
    "export_gds",
    "Job",
    "DrudePole",
    "LorentzPole",
    "Medium",
    "PermittivityArray",
    "ModePort",
    "ModeSolveProvenance",
    "ModeSource",
    "library",
    "materials",
    "PlaneWave",
    "TfsfBox",
    "PointDipole",
    "PortMode",
    "Polygon",
    "read_gds_cell_names",
    "RunSpec",
    "SCHEMA_VERSION",
    "Simulation",
    "cloud",
    "RunResult",
    "SolverRunError",
    "Sphere",
    "Structure",
    "GradedMeshAxis",
    "GradedMesh",
    "MeshOverride",
    "UniformMesh",
    "auto_mesh",
    "convert_to_hdf5",
    "quote",
    "find_solver",
    "gpus",
    "submit",
    "run_local",
    # inverse design (adjoint topology optimization)
    "inverse_design",
    "DesignRegion",
    "PointIntensity",
    "ModePower",
    "GradientResult",
    "OptimizeResult",
    "ParametricResult",
    "value_and_gradient",
    "assemble_gradient",
    "optimize",
    "optimize_parametric",
    "__version__",
]


# --- deprecated aliases (2026-09 cross-solver rename; remove in 0.2) ---------
_RENAMED = {
    "web": "cloud",
    "PolySlab": "Polygon",
    "FluxMonitor": "PowerMonitor",
    "FieldTimeMonitor": "TimeMonitor",
    "FieldDftMonitor": "ProfileMonitor",
    "FieldSnapshotMonitor": "SnapshotMonitor",
    "SimulationData": "RunResult",
    "BatchData": "BatchResults",
    "PermittivityData": "PermittivityArray",
    "UniformGridSpec": "UniformMesh",
    "GradedGridSpec": "GradedMesh",
    "GradedAxisCoords": "GradedMeshAxis",
    "auto_grid": "auto_mesh",
    "run_async": "submit",
    "estimate_cost": "quote",
}


def __getattr__(name):
    replacement = _RENAMED.get(name)
    if replacement is not None:
        import warnings

        warnings.warn(
            f"photonhub.{name} was renamed to photonhub.{replacement}; "
            "the old alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[replacement]
    raise AttributeError(f"module 'photonhub' has no attribute {name!r}")
