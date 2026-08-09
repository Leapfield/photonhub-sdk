"""PhotonHub optional mode, analysis, and workflow helpers.

The eigensolvers and post-processors execute on the host, while source/monitor
builders emit ordinary simulation components that run on the selected engine
backend. Plugins live outside ``components/`` so the frozen wire models stay
pure. Import what you need explicitly::

    from photonhub.plugins import ModeSolver

Phase-1 plugins
---------------
``ModeSolver`` — a finite-difference eigenmode (FDE) solver for the guided
modes of a *straight* dielectric waveguide cross-section (semi-vectorial,
quasi-TE/quasi-TM). CPU/numpy only; see :mod:`photonhub.plugins.modes`.

``VectorModeSolver`` — a *full-vectorial* FDE solver using the
Fallahkhair–Li–Murphy transverse-H operator: all six field components, group
index, and bent/leaky modes with complex ``n_eff``. Host-side; requires scipy.
See :mod:`photonhub.plugins.vector_modes`.

``run_eme`` — a minimal eigenmode-expansion (EME) propagator: staircase a
z-varying device into z-invariant sections, mode-match at the interfaces and
cascade the per-section/-interface scattering matrices (Redheffer star product)
into one device S-matrix. Built on ``VectorModeSolver``; CPU only. See
:mod:`photonhub.plugins.eme`.
"""

from .cvcs import cvcs_sections, interpolate_mode, interpolate_plane
from .eme import (
    EMEResult,
    Section,
    cascade,
    interface_smatrix,
    propagation_smatrix,
    rectangular_base_section,
    run_eme,
    run_eme_band,
    star_product,
    waveguide_section,
)
from .mode_tracking import (
    TrackingResult,
    match_modes,
    reorder_to_tracks,
    track_modes,
    transverse_overlap,
)
from .mode_devices import (
    ModeMonitor,
    mode_monitor,
    mode_launch,
    mode_source,
    mode_source_vector,
    solve_mode_bank,
    solve_modes_by_freq,
    transmission,
)
from .mode_overlap import (
    ModeBank,
    ModeOverlap,
    gaussian_mode,
    mode_amplitude,
    mode_decomposition,
    mode_overlap,
    mode_overlap_matrix,
    mode_transmission,
    vector_modal_fields,
)
from .eq_current_source import equivalence_current_source
from .gaussian_beam import gaussian_beam, gaussian_beam_source
from .kfj_smoothing import (
    mode_bank_on_cross_section,
    sample_cross_section_kfj,
    solve_mode_on_cross_section,
)
from .modes import Mode, ModeSolver
from .near_field import FarField, equivalent_currents, far_field
from .resonance import ResonanceFinder, select_resonances
from .yee_mode import (
    sample_staggered_eps,
    solve_yee_mode,
    solve_yee_mode_bank,
    solve_yee_port_mode_bank,
    window_min_face_bcs,
    solve_yee_multimode_bank,
)
from .smatrix import (
    SPort,
    assemble_smatrix,
    assert_passive,
    assert_reciprocal,
    is_passive,
    is_reciprocal,
    passivity_violation,
    reciprocity_error,
    smatrix,
)
from .vector_modes import VectorMode, VectorModeSolver

__all__ = [
    "EMEResult",
    "FarField",
    "Mode",
    "ModeBank",
    "ModeMonitor",
    "ModeOverlap",
    "ModeSolver",
    "ResonanceFinder",
    "SPort",
    "Section",
    "TrackingResult",
    "VectorMode",
    "VectorModeSolver",
    "assemble_smatrix",
    "assert_passive",
    "assert_reciprocal",
    "cascade",
    "cvcs_sections",
    "equivalence_current_source",
    "equivalent_currents",
    "far_field",
    "gaussian_beam",
    "gaussian_beam_source",
    "gaussian_mode",
    "interface_smatrix",
    "interpolate_mode",
    "interpolate_plane",
    "is_passive",
    "is_reciprocal",
    "match_modes",
    "mode_amplitude",
    "mode_bank_on_cross_section",
    "mode_decomposition",
    "mode_monitor",
    "mode_overlap",
    "mode_overlap_matrix",
    "mode_launch",
    "mode_source",
    "mode_source_vector",
    "mode_transmission",
    "passivity_violation",
    "propagation_smatrix",
    "reciprocity_error",
    "rectangular_base_section",
    "reorder_to_tracks",
    "run_eme",
    "run_eme_band",
    "sample_cross_section_kfj",
    "sample_staggered_eps",
    "select_resonances",
    "smatrix",
    "solve_mode_bank",
    "solve_mode_on_cross_section",
    "solve_modes_by_freq",
    "solve_yee_mode",
    "solve_yee_mode_bank",
    "solve_yee_port_mode_bank",
    "window_min_face_bcs",
    "solve_yee_multimode_bank",
    "star_product",
    "track_modes",
    "transmission",
    "transverse_overlap",
    "vector_modal_fields",
    "waveguide_section",
]
