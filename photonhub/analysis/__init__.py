"""PhotonHub analysis package: mode solvers, EME, S-matrix, far-field, resonance, and beam helpers.

The eigensolvers and post-processors execute on the host, while source/monitor
builders emit ordinary simulation components that run on the selected engine
backend. These helpers live outside ``components/`` so the frozen wire models stay
pure. Import what you need explicitly::

    from photonhub.analysis import ModeSolver

Mode and propagation helpers
----------------------------
``ModeSolver`` — a finite-difference eigenmode (FDE) solver for the guided
modes of a *straight* dielectric waveguide cross-section (semi-vectorial,
quasi-TE/quasi-TM). CPU/numpy only; see :mod:`photonhub.analysis.modes`.

``VectorModeSolver`` — a *full-vectorial* FDE solver using the
Fallahkhair–Li–Murphy transverse-H operator: all six field components, group
index, bent/leaky modes with real ``n_eff`` plus ``k_eff`` attenuation metadata,
and experimental EME bases containing guided, radiation, and evanescent modes.
The continuum/PML path is not yet validated for quantitative device radiation
loss. Host-side; requires scipy. See
:mod:`photonhub.analysis.vector_modes`.

``run_eme`` — a full-vector, bidirectional eigenmode-expansion propagator:
staircase a z-varying device into z-invariant sections, match independent
tangential-E/H equations across unequal modal bases, and cascade the rectangular
scattering matrices with the Redheffer star product. Interface residual,
passivity, reciprocity, and fixed-port stability diagnostics expose truncation
instead of forcing unitarity. Built on ``VectorModeSolver``; CPU only. See
:mod:`photonhub.analysis.eme`.

``solve_yee_eme_basis`` — an experimental, engine-native Yee hard-wall basis
for EME. It returns reaction-validated propagating guided and box-radiation
modes only; evanescent roots/PML and quantitative radiation accuracy remain
unvalidated. Use a spectral ``neff_cutoff`` (not a fixed count) for window
convergence controls. CPU only. See :mod:`photonhub.analysis.yee_mode`.

``cvcs_sections`` — interpolate a few tracked, ordinary guided non-PML key
planes into a dense smooth-section EME model, including each mode's full complex
propagation constant. See :mod:`photonhub.analysis.cvcs`.

``SpectrumCompleter`` — analytic completion of a truncated resonator spectrum:
fit the ringdown's poles (via ``ResonanceAnalysis``), validate the model on a
held-out window, and add the closed-form remainder of the DFT sum, so a high-Q
run can stop after a few resolved ringdown periods instead of stepping the
spectrum to convergence. Point-probe spectra only in v1; CPU only. See
:mod:`photonhub.analysis.spectral_completion`.
"""

from .cvcs import cvcs_sections, interpolate_mode, interpolate_plane
from .diffraction import DiffractionOrders, diffraction_orders
from .eme import (
    EMEConvergenceReport,
    EMEResult,
    InterfaceDiagnostics,
    Section,
    cascade,
    eme_convergence_report,
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
from .import_source import import_field, import_source
from .kfj_smoothing import (
    mode_bank_on_cross_section,
    sample_cross_section_kfj,
    solve_mode_on_cross_section,
)
from .modes import Mode, ModeSolver
from .near_field import FarField, equivalent_currents, far_field
from .propagate import FocalScan, focal_scan, propagate_plane, FocalMetrics, focal_metrics
from .thin_lens import thin_lens_beam, thin_lens_source
from .resonance import ResonanceAnalysis, select_resonances
from .spectral_completion import CompletionRejected, SpectrumCompleter
from .yee_mode import (
    sample_staggered_eps,
    solve_yee_eme_basis,
    solve_yee_mode,
    solve_yee_mode_bank,
    solve_yee_multimode_bank,
    solve_yee_port_mode_bank,
    window_min_face_bcs,
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
from .smatrix_driver import (
    SMatrixPlan,
    SMatrixPort,
    SMatrixResult,
    plan_smatrix,
    run_smatrix,
    write_touchstone,
)
from .vector_modes import VectorMode, VectorModeSolver
from .waveguide import WaveguideModes, rectangular_waveguide

__all__ = [
    "DiffractionOrders",
    "EMEConvergenceReport",
    "EMEResult",
    "FarField",
    "FocalMetrics",
    "FocalScan",
    "InterfaceDiagnostics",
    "Mode",
    "ModeBank",
    "ModeMonitor",
    "ModeOverlap",
    "ModeSolver",
    "ResonanceAnalysis",
    "SMatrixPlan",
    "SMatrixPort",
    "SMatrixResult",
    "SPort",
    "SpectrumCompleter",
    "Section",
    "TrackingResult",
    "VectorMode",
    "VectorModeSolver",
    "WaveguideModes",
    "assemble_smatrix",
    "assert_passive",
    "assert_reciprocal",
    "CompletionRejected",
    "cascade",
    "cvcs_sections",
    "diffraction_orders",
    "eme_convergence_report",
    "equivalence_current_source",
    "equivalent_currents",
    "far_field",
    "focal_metrics",
    "focal_scan",
    "gaussian_beam",
    "gaussian_beam_source",
    "gaussian_mode",
    "import_field",
    "import_source",
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
    "plan_smatrix",
    "propagate_plane",
    "propagation_smatrix",
    "reciprocity_error",
    "rectangular_base_section",
    "rectangular_waveguide",
    "reorder_to_tracks",
    "run_eme",
    "run_eme_band",
    "run_smatrix",
    "sample_cross_section_kfj",
    "sample_staggered_eps",
    "select_resonances",
    "smatrix",
    "solve_mode_bank",
    "solve_mode_on_cross_section",
    "solve_modes_by_freq",
    "solve_yee_eme_basis",
    "solve_yee_mode",
    "solve_yee_mode_bank",
    "solve_yee_multimode_bank",
    "solve_yee_port_mode_bank",
    "star_product",
    "thin_lens_beam",
    "thin_lens_source",
    "track_modes",
    "transmission",
    "transverse_overlap",
    "vector_modal_fields",
    "waveguide_section",
    "window_min_face_bcs",
    "write_touchstone",
]
