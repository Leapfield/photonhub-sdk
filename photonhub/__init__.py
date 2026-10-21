"""PhotonHub Python client. Build simulation specs, run the solver, and load results.

>>> import photonhub as ph
>>> sim = ph.Simulation(...)
>>> data = ph.run_local(sim)
>>> probe = data["probe"]  # xarray.DataArray
"""

from .components import (
    Domain,
    GaussianBeam,
    Mesh,
    Port,
    SCHEMA_VERSION,
    mode_window_um,
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
    TFSFBox,
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
    axis_mirror_mismatch,
    graded_primary_spacings,
)
from . import library
from . import materials
from .constants import c0, eps0
from .cost import CostEstimate, quote
from .data import RunResult
from .gds import GDSLayer, export_gds, import_gds, read_gds_cell_names
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
from . import study
from .spec_hash import spec_sha256
from . import perturbation
from .perturbation import SpatialMap, carrier_plasma, perturbed_medium, thermo_optic_index
from .cloud import gpus  # convenience: ph.gpus() == ph.cloud.gpus() (the GPU menu)
from . import inverse_design
from .inverse_design import (
    BroadbandGradientResult,
    BroadbandModeTransmission,
    DesignRegion,
    FiniteDifferenceResult,
    GradientResult,
    ModePower,
    ModeTransmission,
    OptimizeResult,
    ParametricGradientResult,
    ParametricResult,
    PointIntensity,
    PolygonProfile,
    TransmissionPort,
    WidthProfile,
    assemble_gradient,
    build_adjoint_simulation,
    curvature_penalty,
    finite_difference_check,
    optimize,
    optimize_parametric,
    parametric_finite_difference_check,
    parametric_value_and_gradient,
    value_and_gradient,
)

__version__ = "0.1.6"

__all__ = [
    "study",
    "perturbation",
    "SpatialMap",
    "carrier_plasma",
    "perturbed_medium",
    "thermo_optic_index",
    "spec_sha256",
    "c0",
    "eps0",
    "mode_window_um",
    "Apodization",
    "Domain",
    "GaussianBeam",
    "Mesh",
    "Port",
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
    "GDSLayer",
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
    "TFSFBox",
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
    "axis_mirror_mismatch",
    "graded_primary_spacings",
    "convert_to_hdf5",
    "quote",
    "find_solver",
    "gpus",
    "submit",
    "run_local",
    # inverse design (adjoint topology optimization)
    "inverse_design",
    "DesignRegion",
    "WidthProfile",
    "PolygonProfile",
    "curvature_penalty",
    "PointIntensity",
    "ModePower",
    "TransmissionPort",
    "ModeTransmission",
    "BroadbandModeTransmission",
    "GradientResult",
    "BroadbandGradientResult",
    "FiniteDifferenceResult",
    "build_adjoint_simulation",
    "finite_difference_check",
    "OptimizeResult",
    "ParametricGradientResult",
    "ParametricResult",
    "value_and_gradient",
    "parametric_value_and_gradient",
    "parametric_finite_difference_check",
    "assemble_gradient",
    "optimize",
    "optimize_parametric",
    "__version__",
]


# --- deprecated aliases (2026-09 naming convention; remove in 0.2) -----------
_RENAMED = {
    # acronyms in caps
    "GdsLayer": "GDSLayer",
    "TfsfBox": "TFSFBox",
}

# Removed in 0.1.6. These were renamed in 0.1.2 (the cross-solver rename) and
# kept as deprecated aliases until then; each now fails with the current
# spelling.
_REMOVED = {
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
    if name == "web":
        # The shim module, not ``cloud`` itself: only the shim carries the
        # ``Web*`` class aliases. Importing it issues the one
        # DeprecationWarning and binds ``photonhub.web``, so this runs once.
        import importlib

        return importlib.import_module(f"{__name__}.web")
    replacement = _RENAMED.get(name)
    if replacement is not None:
        import warnings

        from ._compat import caller_stacklevel

        warnings.warn(
            f"photonhub.{name} was renamed to photonhub.{replacement}; "
            "the old alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=caller_stacklevel(),
        )
        return globals()[replacement]
    current = _REMOVED.get(name)
    if current is not None:
        raise AttributeError(f"photonhub.{name} was removed in 0.1.6; use photonhub.{current}")
    raise AttributeError(f"module 'photonhub' has no attribute {name!r}")
