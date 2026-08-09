"""Top-level simulation model — the root of the wire format."""

import math
import os
import stat
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

from ..cost import CostEstimate, estimate_cost
from .base import MAX_INT32, FrozenModel, PositiveUm, SubpixelMethodName
from .grid import (
    GridSpecType,
    graded_primary_spacings,
    realized_cells,
    snap_mixed_plane,
    snapped_plane_index,
)
from .medium import Background, Boundaries
from .monitors import (
    FieldDftMonitor,
    FluxMonitor,
    MonitorType,
    mode_port_physical_polarization,
)
from .run import RunSpec
from .sources import ModeSource, PlaneWave, SourceType
from .structures import Structure

SCHEMA_VERSION = "1.16.0-alpha.1"
SUPPORTED_SCHEMA_MAJOR = 1

_AXES = "xyz"

# eta0 = mu0 * c0 — the vacuum wave impedance. Bridges the CPML sigma/alpha peak
# between Tidy3D's dimensionless "units of 2*eps0/dt" convention and the engine's
# S/m: since eps0*c0 = 1/eta0, the peak-conductivity unit 2*eps0/dt reduces to a
# form needing only eta0, the cell spacing, and the Courant number (see
# Simulation._two_eps0_over_dt).
_ETA0 = 1.25663706212e-6 * 2.99792458e8

# Tidy3D StablePML-aligned CPML stabilization (its DefaultStablePMLParameters,
# tidy3d/components/boundary.py): kappa_max = 5.0 and a CFS alpha_max = 0.9, both
# quoted in Tidy3D's 2*eps0/dt units (StablePML also uses 40 layers and
# sigma_max = 1.0). The engine already reads sigma in that convention
# (pml_sigma_max, default 1.5) but alpha in absolute S/m (pml_alpha_max) — which
# is precisely WHY the default alpha (0.24 S/m) is inert — so we convert
# 0.9 * 2*eps0/dt to S/m at the scene's timestep (_two_eps0_over_dt). alpha and
# kappa are the fit-safe stability levers (they never change the slab thickness);
# the layer bump is applied only by the explicit with_stabilized_pml().
_STABLE_PML_LAYERS = 40
_STABLE_PML_KAPPA_MAX = 5.0
_STABLE_PML_ALPHA_SCALE = 0.9
# The AUTO-stabilizer dose (_auto_stabilize_dispersive_pml) is 9x GENTLER than
# with_stabilized_pml's Tidy3D-parity 0.9. Tidy3D treats StablePML as an OPT-IN
# profile paired with 40 layers; auto-applying its alpha to the default 12-layer
# slab de-tunes the PML for PROPAGATING waves: at optical grids omega*eps0 is
# only ~0.02*(2*eps0/dt), so alpha = 0.9*(2*eps0/dt) throttles the CFS
# absorptive term sigma*omega*eps0/(alpha^2 + (omega*eps0)^2) ~40x and the slab
# REFLECTS instead of absorbing (dispersive crossing @25 c/lambda: R_input 0.35
# with const-n IDENTICAL to dispersive — the alpha, not the Lorentz ADE — and
# 5.5x the ring-down; 1-D normal incidence R 0.49). Curing the trapped-resonance
# divergence only needs the CFS crossover alpha/(2*pi*eps0) up at the mode's
# optical frequency — the 2026-07-03 cure measured ~2% of sigma_max, ~30x below
# the Tidy3D value — so 0.1 keeps ~3x that margin (rod probe: stable through
# 294k steps at alpha 3e4-1e5 S/m; CFS-inert diverges @197k) while restoring
# low reflection (crossing R: 3e4 -> 1e-4, 1e5 -> 0.024, vs 0.35 at the 0.9
# dose). Measured 2026-07-17 on MI300X gfx942:
# engine/docs/subpixel-dispersion-instability.md, final section.
_AUTO_PML_ALPHA_SCALE = 0.1
# CFS-inert threshold: an alpha below this fraction of the sigma peak leaves the
# PML's DC pole effectively undamped (the divergence lever). 0.5%.
_CFS_INERT_FRAC = 0.005


class Simulation(FrozenModel):
    """Complete simulation description. Serializes 1:1 to the JSON wire
    format consumed by ``phsolver`` (schemas/GOVERNANCE.md).

    The cross-field validators here are best-effort early feedback mirroring
    the engine's checks where they are cheap and unambiguous; ``phsolver
    validate`` remains authoritative (notably for Yee snapping at exact
    boundaries and for the plane-wave/PML intersection rule)."""

    schema_version: str = SCHEMA_VERSION
    size_um: Tuple[PositiveUm, PositiveUm, PositiveUm]
    grid: GridSpecType
    run: RunSpec
    background: Background = Background()
    # NUMERICS.md section 11: layer count for every "pml" boundary axis. The
    # engine default is 12; an UNSET value is omitted from the wire format
    # (see to_wire_dict) so Phase-0 documents round-trip byte-identically and
    # remain consumable by schema-1.0 parsers that reject unknown keys.
    pml_num_layers: int = Field(default=12, ge=4, le=MAX_INT32)
    # NUMERICS.md §11 CPML profile (Roden–Gedney) tuning knobs. The defaults
    # reproduce the historically-hardcoded profile BIT-FOR-BIT, so an UNSET
    # value is omitted from the wire format (see _wire_exclude) and the engine
    # applies the same constants — documents from earlier minors round-trip
    # byte-identically and stay consumable by parsers that reject unknown keys.
    #   pml_m         polynomial grading order (>= 1)
    #   pml_kappa_max real-stretch peak at the wall (>= 1)
    #   pml_alpha_max CFS frequency-shift peak in S/m (>= 0)
    # The default alpha_max (0.24 S/m) is CFS-INERT: at optical frequencies it is
    # ~1e-5 of the sigma peak, so it costs no in-band reflectionlessness but does
    # NOT damp the DC/late-time pole. The alpha the CFS actually needs is quoted
    # in Tidy3D's dimensionless 2*eps0/dt convention (like pml_sigma_max) — its
    # StablePML uses alpha_max = 0.9 — and this asymmetry (sigma dt-relative,
    # alpha absolute S/m) is why the default alpha reads as inert. Raising kappa_max
    # + alpha_max is the "stabilized" recipe for a grazing/long-run/dispersive
    # scene that diverges: a DISPERSIVE (Lorentz) scene gets kappa 5.0 +
    # alpha 0.9*(2*eps0/dt) applied AUTOMATICALLY at construction
    # (_auto_stabilize_dispersive_pml), and ``with_stabilized_pml`` builds the
    # full Tidy3D-StablePML-aligned copy (also adding the layer bump).
    #   pml_sigma_max peak conductivity in Tidy3D units (2*eps0/dt). The DEFAULT
    #     1.5 matches Tidy3D's default PML sigma_max exactly — a resolution-
    #     consistent, dt-based peak that drains grazing/trapped modes cleanly.
    #     0 = the LEGACY Roden-Gedney dl-heuristic (0.8*(m+1)/(eta0*dl)), which is
    #     ~1.6-2.2x WEAKER (worst on a graded mesh) and reproduces the pre-1.5
    #     coefficient path bit-for-bit — an escape hatch for exact back-compat.
    #     (The raw engine SimulationSpec default is also 1.5; the CpmlProfile
    #     struct primitive stays 0.0, so only a hand-written spec omitting every
    #     field hits legacy.)
    pml_m: float = Field(default=3.0, ge=1.0)
    pml_kappa_max: float = Field(default=3.0, ge=1.0)
    pml_alpha_max: float = Field(default=0.24, ge=0.0)
    pml_sigma_max: float = Field(default=1.5, ge=0.0)
    # NUMERICS.md §21 adiabatic-absorber knobs (apply to every "absorber" axis).
    # The absorber is a graded electric-conductivity ramp, NOT a stretched-
    # coordinate PML — the robustness fallback for the cases that make a PML
    # diverge (a structure crossing the boundary, dispersive/gain media at the
    # edge). It needs more layers than the PML for comparable reflection (40 vs
    # 12) because, being impedance-unmatched, its reflection falls only
    # polynomially with thickness. Additive-optional: an UNSET value is omitted
    # from the wire (see _wire_exclude) so earlier-minor parsers accept the
    # document and golden specs round-trip byte-identically.
    #   absorber_num_layers  slab thickness in cells, both faces (>= 4)
    #   absorber_m           polynomial conductivity grading order (>= 1)
    absorber_num_layers: int = Field(default=40, ge=4, le=MAX_INT32)
    absorber_m: float = Field(default=3.0, ge=1.0)
    # NUMERICS.md §16: volume-fraction subpixel smoothing of the rasterized
    # permittivity. The FIELD default is False (the §9 last-wins point sample,
    # bit-exact with prior schema minors, and the engine/wire default), but
    # CONSTRUCTION auto-enables it: _resolve_subpixel_default turns subpixel ON
    # (method "contour") for a non-dispersive scene — the effective out-of-box
    # default — and leaves it OFF for a dispersive (Lorentz) scene (the divergence
    # guard, see that validator). An UNSET value is omitted from the wire format
    # (see _wire_exclude) so documents stay byte-identical and consumable by
    # parsers from earlier minors that reject unknown keys. Box (exact) and
    # curved (Cylinder/PolySlab/Sphere, supersampled §16.7) interfaces are
    # smoothed on BOTH uniform and graded meshes (CPU, single GPU, and multi-GPU).
    # Only the off-diagonal ``tensor_full`` on a graded mesh remains deferred
    # (§16.6; engine reference_solver.cpp / gpu_solver.hip reject it). Uniform
    # ``tensor_full`` is available on GPU subject to its engine-wide
    # lossless/non-dispersive combination rules.
    subpixel: bool = False
    # NUMERICS.md §16.5/§16.8/§16.11: which smoothing to apply when ``subpixel``
    # is on. Six operators: "volume" (isotropic volume average, bit-identical to
    # schema < 1.7.0); "tensor" (diagonal anisotropic KFJ); "tensor_full" (full
    # off-diagonal KFJ); "contour" (diagonal KFJ fed the exact §16.10 PolySlab
    # fill == Tidy3D's default PolarizedAveraging plus the exact vertical-wall
    # fill — the DEFAULT); and the rigorous contour-path EPs (Mohammadi-Nadgaran-
    # Agio 2005 = Tidy3D's ContourPathAveraging): "contour_diag" (the paper's
    # per-component scalar CP-EP) and "contour_full" (its full off-diagonal Kottke
    # tensor for tilted/curved walls). The FIELD default is "contour" to MATCH
    # the effective construction default: _resolve_subpixel_default auto-enables
    # subpixel+contour on a non-dispersive scene and fills an unset method with
    # "contour" on any explicit subpixel-on, so the declared default and the
    # auto/explicit-on paths agree. contour == tensor == contour_diag on axis-
    # aligned interfaces (they reduce to arithmetic/harmonic) and differ only on
    # tilted/curved cells; contour is the Tidy3D-default match, contour_diag the
    # rigorous CP-EP alternative.
    # Omitted from the wire when unset (see _wire_exclude), so an ingested pre-1.7.0
    # subpixel-on document (no method key) is still run by the engine as ITS default
    # (volume) — the field default is cosmetic on that ingest path.
    subpixel_method: SubpixelMethodName = "contour"
    structures: Tuple[Structure, ...] = ()
    boundaries: Boundaries = Boundaries()
    # NUMERICS.md §20: optional symmetry plane on each axis' MINIMUM face.
    # 0 = none; -1 = odd / electric (a PEC mirror: tangential E pinned, normal E
    # free — the common case for a TE-like mode); +1 = even / magnetic (PMC
    # mirror: tangential E free, the cross-plane H read is the odd-H image).
    # PMC is available on all three axes on both the CPU reference solver and
    # the GPU (z via the negated k=-1 ghost-plane mirror; NUMERICS.md §20.4).
    # When symmetry[a] != 0 the axis is non-periodic and boundaries[a] governs
    # the FAR (max) face only (pml or pec); the PML on that axis is built
    # one-sided so the min/symmetry face reflects. You supply the reduced (half)
    # domain with the structure's mirror plane on that face. Additive-optional:
    # an all-zero symmetry is omitted from the wire (see _wire_exclude), so
    # earlier-minor parsers and golden specs round-trip byte-identically.
    symmetry: Tuple[int, int, int] = (0, 0, 0)
    sources: Tuple[SourceType, ...] = Field(min_length=1)
    monitors: Tuple[MonitorType, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def _supported_major_version(cls, v: str) -> str:
        # Mirrors the engine's schema_major() gate (engine/src/core/
        # resolve.cpp): leading dotted component, digits only, must be the
        # supported major — schemas/GOVERNANCE.md requires an explicit
        # migration for breaking wire changes, and
        # an unsupported spec must fail at construction, not at submission.
        head = v.split(".", 1)[0]
        if not (head.isascii() and head.isdigit()) or int(
                head) != SUPPORTED_SCHEMA_MAJOR:
            raise ValueError(
                f"unsupported schema_version {v!r}: this client supports "
                f"major version {SUPPORTED_SCHEMA_MAJOR} only; no migration "
                "for other major versions is currently provided"
            )
        return v

    @field_validator("monitors")
    @classmethod
    def _unique_monitor_names(cls, v):
        names = [m.name for m in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"monitor names must be unique; duplicates: {dupes}")
        return v

    @field_validator("symmetry")
    @classmethod
    def _symmetry_values(cls, v):
        # NUMERICS.md §20: each axis is -1 (odd/electric), 0 (none), or +1
        # (even/magnetic). CPU and GPU support both signs on every axis.
        for a, s in enumerate(v):
            if s not in (-1, 0, 1):
                raise ValueError(
                    f"symmetry[{a}] ('{_AXES[a]}'): must be -1 (electric/PEC), "
                    f"0 (none), or +1 (magnetic/PMC), got {s}"
                )
        return v

    @model_validator(mode="after")
    def _modal_port_rules(self) -> "Simulation":
        """Validate the authoring contract for modal post-processing.

        ``mode_port`` is deliberately metadata on an ordinary DFT monitor: the
        engine still records raw tangential Yee fields, while Workbench solves
        and projects the requested modes after the run.  These checks keep that
        later projection reproducible and prevent a run from completing with a
        port definition that cannot be evaluated.
        """
        required_fields = {
            0: {"Ey", "Ez", "Hy", "Hz"},
            1: {"Ez", "Ex", "Hz", "Hx"},
            2: {"Ex", "Ey", "Hx", "Hy"},
        }
        realized = self._realized_um()
        port_monitors = [
            monitor for monitor in self.monitors
            if isinstance(monitor, FieldDftMonitor)
            and monitor.mode_port is not None
        ]
        if not port_monitors:
            return self

        driven = []
        for monitor in port_monitors:
            port = monitor.mode_port
            assert port is not None
            zero_axes = [i for i, size in enumerate(monitor.size_um)
                         if size == 0.0]
            if len(zero_axes) != 1:
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port requires a plane "
                    "field_dft monitor with exactly one zero size_um axis"
                )
            normal = zero_axes[0]
            transverse = tuple(i for i in range(3) if i != normal)
            missing = sorted(required_fields[normal] - set(monitor.fields))
            if missing:
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port requires all four "
                    f"tangential fields; missing {missing}"
                )
            if monitor.interval_space is not None and any(
                    stride != 1 for stride in monitor.interval_space):
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port does not support "
                    "spatially decimated field data; use interval_space=(1,1,1) "
                    "or omit it"
                )
            if monitor.apodization is not None:
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port does not support "
                    "time apodization because independently gated DFT fields "
                    "do not preserve modal S-parameter ratios"
                )
            if port.thickness_axis is not None:
                thickness = _AXES.index(port.thickness_axis)
                if thickness not in transverse:
                    raise ValueError(
                        f"monitor '{monitor.name}'.mode_port.thickness_axis "
                        f"cannot equal the plane normal '{_AXES[normal]}'"
                    )

            normal_lo, normal_hi, normal_boundary = (
                self._nonabsorbing_bounds_um(normal))
            requested_normal_position = float(monitor.center_um[normal])
            snapped_normal_position, _ = snap_mixed_plane(
                self, normal, requested_normal_position)
            if (
                normal_boundary in ("pml", "absorber")
                and not (
                    normal_lo + 1e-12
                    < requested_normal_position
                    < normal_hi - 1e-12
                )
            ):
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port plane on "
                    f"'{_AXES[normal]}' requested "
                    f"{requested_normal_position:.6g} um (snaps to "
                    f"{snapped_normal_position:.6g} um) lies inside the "
                    f"{normal_boundary} band; "
                    f"choose a position in the nonabsorbing interval "
                    f"({normal_lo:.6g}, {normal_hi:.6g}) um"
                )
            if (
                normal_boundary in ("pml", "absorber")
                and not (
                    normal_lo + 1e-12
                    < snapped_normal_position
                    < normal_hi - 1e-12
                )
            ):
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port plane on "
                    f"'{_AXES[normal]}' requested "
                    f"{requested_normal_position:.6g} um but snaps to "
                    f"{snapped_normal_position:.6g} um inside the "
                    f"{normal_boundary} band; choose a position whose "
                    "mixed-Yee quarter-cell plane stays in the nonabsorbing "
                    f"interval ({normal_lo:.6g}, {normal_hi:.6g}) um"
                )
            for local, axis in enumerate(transverse):
                half_window = 0.5 * port.size_um[local]
                half_monitor = 0.5 * monitor.size_um[axis]
                offset = abs(port.center_um[local] - monitor.center_um[axis])
                if offset + half_window > half_monitor + 1e-12:
                    raise ValueError(
                        f"monitor '{monitor.name}'.mode_port window on "
                        f"'{_AXES[axis]}' extends outside the recorded DFT plane"
                    )
                lo = port.center_um[local] - half_window
                hi = port.center_um[local] + half_window
                if lo < -1e-12 or hi > realized[axis] + 1e-12:
                    raise ValueError(
                        f"monitor '{monitor.name}'.mode_port window on "
                        f"'{_AXES[axis]}' lies outside the realized domain"
                    )
                interior_lo, interior_hi, boundary = (
                    self._nonabsorbing_bounds_um(axis))
                if (
                    boundary in ("pml", "absorber")
                    and (
                        lo < interior_lo - 1e-12
                        or hi > interior_hi + 1e-12
                    )
                ):
                    raise ValueError(
                        f"monitor '{monitor.name}'.mode_port window on "
                        f"'{_AXES[axis]}' overlaps the {boundary} band; keep "
                        f"the solve window inside the nonabsorbing interval "
                        f"[{interior_lo:.6g}, {interior_hi:.6g}] um"
                    )

            if port.source_index is None:
                continue
            driven.append(monitor.name)
            if port.source_index >= len(self.sources):
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port.source_index "
                    f"{port.source_index} is outside sources"
                )
            source = self.sources[port.source_index]
            if not isinstance(source, ModeSource) or source.mode_solve is None:
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port.source_index must "
                    "reference a ModeSource with mode_solve provenance"
                )
            if source.axis != _AXES[normal]:
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port plane normal "
                    f"'{_AXES[normal]}' does not match its ModeSource axis "
                    f"'{source.axis}'"
                )
            incident_direction = "+" if port.out_direction == "-" else "-"
            if source.direction != incident_direction:
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port out_direction "
                    f"'{port.out_direction}' requires its incident ModeSource "
                    f"direction to be '{incident_direction}'"
                )
            travel = 1.0 if source.direction == "+" else -1.0
            if ((monitor.center_um[normal] - source.position_um) * travel
                    <= 1e-12):
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port must lie downstream "
                    f"of its ModeSource plane along {source.direction}{source.axis}"
                )
            launched = (
                mode_port_physical_polarization(
                    source.mode_solve.polarization,
                    _AXES[normal],
                    port.thickness_axis,
                ),
                source.mode_solve.mode_index,
            )
            requested = {(mode.polarization, mode.mode_index)
                         for mode in port.modes}
            if launched not in requested:
                raise ValueError(
                    f"monitor '{monitor.name}'.mode_port modes must include "
                    f"the launched {launched[0]}{launched[1]} channel"
                )

        if len(driven) > 1:
            raise ValueError(
                "a single simulation run supports at most one source-linked "
                f"driven modal port; found {len(driven)} ({driven})"
            )
        return self

    def _axis_coords_um(self, axis_index: int):
        """The graded coordinate array (microns) for an axis, or None when
        that axis is uniform (UniformGridSpec, or a GradedGridSpec axis not
        listed in ``coords``)."""
        coords = getattr(self.grid, "coords", None)
        if coords is None:
            return None
        return getattr(coords, "xyz"[axis_index])

    def _realized_um(self) -> Tuple[float, float, float]:
        dl = self.grid.dl_um
        out = []
        for i, L in enumerate(self.size_um):
            q = self._axis_coords_um(i)
            if q is None:
                out.append(realized_cells(L, dl) * dl)
            else:
                # NUMERICS.md section 15.1: realized length = closing node
                # q[n-1] + (replicate-last spacing).
                out.append(q[-1] + graded_primary_spacings(q)[-1])
        return tuple(out)

    def _nonabsorbing_bounds_um(
            self, axis_index: int) -> Tuple[float, float, str]:
        """Physical interval outside this axis' PML/absorber cells.

        Boundary layers are counted in realized primary-grid cells, including
        on graded axes. A symmetry plane replaces the minimum-face absorbing
        slab, so only the far-face layers are reserved in that case.
        """
        axis = _AXES[axis_index]
        boundary = getattr(self.boundaries, axis)
        realized = self._realized_um()[axis_index]
        if boundary not in ("pml", "absorber"):
            return 0.0, realized, boundary

        layers = (
            self.pml_num_layers
            if boundary == "pml"
            else self.absorber_num_layers
        )
        lower_layers = 0 if self.symmetry[axis_index] != 0 else layers
        q = self._axis_coords_um(axis_index)
        if q is None:
            cells = realized_cells(self.size_um[axis_index], self.grid.dl_um)
            lower_index = lower_layers
            upper_index = cells - layers

            def coordinate(index):
                return index * self.grid.dl_um
        else:
            values = tuple(q)
            cells = len(values)
            closing = values[-1] + graded_primary_spacings(values)[-1]
            lower_index = lower_layers
            upper_index = cells - layers

            def coordinate(index):
                return closing if index == cells else values[index]

        if lower_index > cells or upper_index < 0 or lower_index >= upper_index:
            raise ValueError(
                f"modal ports require a nonabsorbing interior on axis "
                f"'{axis}', but {layers} {boundary} layers leave none"
            )
        return (
            float(coordinate(lower_index)),
            float(coordinate(upper_index)),
            boundary,
        )

    def _two_eps0_over_dt(self) -> float:
        """The CPML peak-conductivity unit ``2*eps0/dt`` [S/m] at this scene's
        timestep — the scale Tidy3D quotes ``sigma_max`` / ``alpha_max`` in, and
        the bridge between its dimensionless convention and the engine's S/m
        ``pml_alpha_max``.

        Built from the engine's CFL timestep (NUMERICS.md §2; resolve.cpp and
        grid.h ``graded_courant_dt``): ``dt = courant / (c0*sqrt(sum_a 1/dl_a^2))``
        over the per-axis MINIMUM primary spacing, which on a uniform grid is
        ``courant*dl / (c0*sqrt(3))``. Since ``eps0*c0 = 1/eta0`` this reduces to
        ``(2/eta0)*sqrt(sum_a 1/dl_a^2)/courant`` — needing only ``eta0``, the
        cell spacings, and the Courant number, and matching the engine's dt so a
        converted alpha lands exactly on Tidy3D's scale."""
        inv_sq = 0.0
        for a in range(3):
            q = self._axis_coords_um(a)
            dl_um = self.grid.dl_um if q is None \
                else min(graded_primary_spacings(q))
            dl_m = dl_um * 1e-6
            inv_sq += 1.0 / (dl_m * dl_m)
        return (2.0 / _ETA0) * math.sqrt(inv_sq) / self.run.courant

    def _pml_sigma_peak_Sm(self) -> float:
        """The peak CPML conductivity [S/m] the engine will actually use:
        ``pml_sigma_max`` interpreted in Tidy3D's ``2*eps0/dt`` convention when
        it is > 0 (the default 1.5), else the legacy dl-heuristic
        ``0.8*(m+1)/(eta0*dl)`` (see spec.pml_sigma_max, reference_solver.cpp
        ``cpml_coef``). The reference for the CFS-inert test — an alpha far below
        this leaves the DC pole undamped."""
        if self.pml_sigma_max > 0.0:
            return self.pml_sigma_max * self._two_eps0_over_dt()
        return 0.8 * (self.pml_m + 1.0) / (_ETA0 * self.grid.dl_um * 1e-6)

    def _dispersive_boundary_crossings(self) -> Tuple[bool, bool, bool]:
        """Per axis: does a dispersive (Lorentz) structure's bounding box reach
        into that axis' OUTER absorbing band (the PML/absorber layers)? These
        are the structures for which a stretched-coordinate PML can diverge —
        the absorber's reason to exist (NUMERICS.md §21). Conservative: the
        bounding box contains slanted/curved geometry, so a crossing is never
        missed (it can be over-reported).

        The band thickness is referenced to the base spacing ``dl_um`` (the PML
        layer count times one cell); this is the same approximation the engine's
        uniform-grid PML uses and is adequate for an early-feedback verdict."""
        from ._bounds import geometry_bounds_um

        realized = self._realized_um()
        # Outer PML-stretch thickness (microns): the region where a dispersive
        # medium is hostile. ``pml_num_layers`` (not the thicker absorber count) —
        # the decision is "does the medium reach the stretched-coordinate region
        # of the boundary currently in place", which is the PML's depth.
        band = self.pml_num_layers * self.grid.dl_um
        out = [False, False, False]
        for s in self.structures:
            if getattr(s.medium, "lorentz", None) is None:
                continue
            bb = geometry_bounds_um(s.geometry)
            for a in range(3):
                lo, hi = bb[a]
                L = realized[a]
                if lo < band or hi > L - band:
                    out[a] = True
        return tuple(out)

    def _validated_copy(self, update: dict) -> "Simulation":
        """``model_copy(update=)`` with the cross-field model validators
        RE-RUN — the shared backend of every ``with_*`` helper.

        pydantic's ``model_copy`` skips validation entirely, so a helper using
        it alone could hand back a Simulation that direct construction rejects
        (e.g. ``with_absorber`` replacing the periodic transverse boundaries
        required by a §13 plane-wave source), deferring the failure to engine
        submission and defeating the model's cross-field validation contract.

        The returned object is the plain ``model_copy`` result: its
        ``model_fields_set`` (originals plus exactly the updated keys) is what
        ``_wire_exclude`` keys the unset-field omission on, so the wire bytes
        of a valid scene are identical to the pre-validation behavior. The
        validation happens on a THROWAWAY rebuilt from the FULL
        ``model_dump`` — full, not ``exclude_unset=True``, because the unset
        ``type`` discriminators of nested source/monitor/geometry unions would
        be dropped and the throwaway could not re-validate; values, not
        fields_set, are what the raising validators inspect. It is validated
        under the ``wire_ingest`` context so the construction-time
        conveniences (§16 subpixel default resolution, the dispersive-PML
        advisory warning — both non-raising, both already applied/emitted when
        ``self`` was built) do not re-fire on a copy; every RAISING validator
        runs in that context too, so an invalid combination raises here with
        the same error direct construction gives."""
        new = self.model_copy(update=update)
        type(self).model_validate(new.model_dump(mode="python"),
                                  context={"wire_ingest": True})
        return new

    def with_auto_grid(
        self,
        *,
        wavelength_um: Optional[float] = None,
        steps_per_wvl: float = 20.0,
        **auto_grid_kwargs,
    ) -> "Simulation":
        """Return a COPY of this simulation whose ``grid`` is replaced by an
        auto-meshed :class:`GradedGridSpec` derived from this scene — its
        domain ``size_um``, ``structures``, ``background`` index, and (if
        ``wavelength_um`` is omitted) the wavelength inferred from the first
        source. A convenience wrapper over :func:`photonhub.auto_grid`; extra
        keyword arguments (``max_grading``, ``axes``, ``dl_min_um``,
        ``refine_regions``, ...) pass straight through.

        Opt-in only: the default :class:`UniformGridSpec` is unchanged, so no
        existing scene's wire output moves. Use this when you want Tidy3D-style
        per-medium refinement without hand-building coordinate arrays::

            sim = sim.with_auto_grid(steps_per_wvl=20)

        Axes whose boundary is PERIODIC are passed to :func:`auto_grid` as
        ``periodic_axes`` (unless you override it explicitly), so a graded
        periodic axis is generated seam-symmetrically — equal first/last
        primary spacings, the §15.2 closure requirement the engine hard-checks.
        Non-periodic scenes are byte-identical to before.
        """
        from .grid import auto_grid as _auto_grid  # local: avoid import cycle

        bg_index = float(self.background.permittivity) ** 0.5
        src = self.sources[0] if self.sources else None
        if "periodic_axes" not in auto_grid_kwargs:
            kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
            auto_grid_kwargs["periodic_axes"] = "".join(
                _AXES[a] for a in range(3) if kinds[a] == "periodic")
        spec = _auto_grid(
            size_um=tuple(self.size_um),
            wavelength_um=wavelength_um,
            source=None if wavelength_um is not None else src,
            structures=self.structures,
            background_index=bg_index,
            steps_per_wvl=steps_per_wvl,
            **auto_grid_kwargs,
        )
        return self._validated_copy({"grid": spec})

    def with_mesh_overrides(
        self,
        *overrides,
        wavelength_um: Optional[float] = None,
        steps_per_wvl: float = 20.0,
        **auto_grid_kwargs,
    ) -> "Simulation":
        """Return a COPY whose ``grid`` is auto-meshed with one or more
        geometry-based :class:`photonhub.MeshOverride` regions applied (Tidy3D's
        ``MeshOverrideStructure``) — the mesh is forced fine inside each
        override's geometry regardless of the local material, on top of the
        ordinary per-medium refinement.

        A thin wrapper over :meth:`with_auto_grid` that forwards the overrides as
        ``mesh_overrides=``; all other auto-mesh knobs (``max_grading``, ``axes``,
        ``dl_min_um``, ``refine_pad_um``, ...) pass straight through — including
        the periodic-boundary seam handling: axes whose boundary is periodic get
        seam-symmetric coordinates (equal first/last primary spacings, the §15.2
        closure requirement). Opt-in only, like the other ``with_*`` mesh
        helpers — no existing scene's wire output moves unless you call it::

            from photonhub import MeshOverride, Box
            sim = sim.with_mesh_overrides(
                MeshOverride(geometry=Box(center_um=(2, 1, 0.5),
                                          size_um=(0.5, 0.5, 1.0)),
                             dl_um=(0.02, 0.02, None)),
                steps_per_wvl=20)
        """
        return self.with_auto_grid(
            wavelength_um=wavelength_um, steps_per_wvl=steps_per_wvl,
            mesh_overrides=overrides, **auto_grid_kwargs)

    def with_stabilized_pml(
        self,
        *,
        num_layers: int = _STABLE_PML_LAYERS,
        kappa_max: float = _STABLE_PML_KAPPA_MAX,
        alpha_scale: float = _STABLE_PML_ALPHA_SCALE,
    ) -> "Simulation":
        """Return a COPY with the Tidy3D ``StablePML``-aligned CPML profile: more
        layers, a higher real-stretch peak ``kappa_max``, and — the lever the
        default profile keeps inert — a RAISED CFS ``alpha_max`` (NUMERICS.md
        §11). The complex frequency shift is what moves the PML pole off DC and
        cures the late-time / grazing-incidence / dispersive-medium divergences
        that more layers alone do not fix; it costs a few percent of in-band
        absorption, paid back by the extra layers.

        ``alpha_scale`` is quoted in Tidy3D's dimensionless ``2*eps0/dt`` units —
        the SAME convention as ``pml_sigma_max`` — and converted to the engine's
        S/m ``pml_alpha_max`` at this scene's timestep, so the per-step CFS
        damping is mesh-independent and lands on Tidy3D's value (a fixed absolute
        alpha would weaken on finer grids). The defaults reproduce Tidy3D's
        ``StablePML`` (40 layers, ``kappa_max`` 5, ``alpha_max`` 0.9);
        ``pml_sigma_max`` is left at its default (1.5, already in Tidy3D units —
        a slightly stronger peak than StablePML's 1.0, for a lower floor).

        A dispersive (Lorentz) scene gets the alpha+kappa half of this
        AUTOMATICALLY at construction (see ``_auto_stabilize_dispersive_pml``);
        reach for this to ALSO thicken the slab, or on a non-dispersive scene
        that leaks/drifts. Opt-in — no non-dispersive scene's wire output moves
        unless you call it::

            sim = sim.with_stabilized_pml()                 # StablePML-aligned
            sim = sim.with_stabilized_pml(num_layers=60, alpha_scale=1.2)

        (Renamed from the earlier ``with_stable_pml``, which raised only layers
        and kappa — never alpha, so it missed the actual stability lever.)
        """
        return self._validated_copy({
            "pml_num_layers": num_layers,
            "pml_kappa_max": kappa_max,
            # alpha_scale is a fraction of 2*eps0/dt (Tidy3D's alpha_max unit),
            # converted to the engine's S/m field at this scene's dt.
            "pml_alpha_max": alpha_scale * self._two_eps0_over_dt(),
        })

    def with_absorber(self, *, num_layers: int = 40) -> "Simulation":
        """Return a COPY with every face set to the adiabatic absorber
        (NUMERICS.md §21) instead of a PML. Use this when a structure crosses
        the domain boundary or a dispersive/gain medium touches the edge — the
        cases where a stretched-coordinate PML can diverge. The absorber trades
        some reflection (≈ -28 dB at the default 40 layers, vs the PML's -68 dB)
        for robustness; add layers if you need it tighter::

            sim = sim.with_absorber()                 # 40-layer absorber, 6 faces
            sim = sim.with_absorber(num_layers=60)
        """
        return self._validated_copy({
            "boundaries": Boundaries(x="absorber", y="absorber", z="absorber"),
            "absorber_num_layers": num_layers,
        })

    def with_auto_boundaries(self) -> "Simulation":
        """Return a COPY whose OPEN (radiating) boundaries are chosen PER AXIS
        from the materials that reach the domain edge — mirroring Tidy3D's
        material-aware guidance that a stretched-coordinate PML wants a
        non-dispersive medium in its absorbing region:

        * an axis where a **dispersive (Lorentz) medium crosses the boundary**
          gets the adiabatic **absorber** (graded electric conductivity,
          NUMERICS.md §21) — the robust fallback for the regime where a PML can
          diverge (Oskooi & Johnson 2011);
        * every other open axis keeps the (thinner, lower-floor) **PML**.

        ``periodic`` and ``pec`` axes are left untouched: those encode explicit
        physics (a Bloch / transverse-infinite axis, a hard mirror), not an open
        boundary to auto-select. Opt-in — the default boundaries are unchanged,
        so no existing scene's wire output moves unless you call this::

            sim = sim.with_auto_boundaries()   # PML, but absorber where dispersive

        This is the programmatic form of the construction-time warning the same
        crossing raises; call it to act on that advice in one line.

        The open-boundary decision has three cases: a well-behaved scene keeps
        the default PML; a scene that *diverges* without a dispersive edge
        (grazing / long run) wants ``with_stabilized_pml()``; a dispersive
        medium at the wall wants the absorber this method selects.
        """
        crossings = self._dispersive_boundary_crossings()
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        chosen = []
        for a in range(3):
            if kinds[a] in ("periodic", "pec"):
                chosen.append(kinds[a])          # explicit physics — respect it
            elif crossings[a]:
                chosen.append("absorber")        # PML-hostile medium at the wall
            else:
                chosen.append("pml")             # plain open boundary
        return self._validated_copy({
            "boundaries": Boundaries(x=chosen[0], y=chosen[1], z=chosen[2]),
        })

    @model_validator(mode="after")
    def _symmetry_plane_rules(self) -> "Simulation":
        # NUMERICS.md §20.2, mirrored at construction (cheap, unambiguous): a
        # symmetry axis must be non-periodic (boundaries governs the far face),
        # and a symmetry plane is incompatible with a plane-wave source (its
        # TF/SF aux line spans the full transverse plane).
        if not any(s != 0 for s in self.symmetry):
            return self
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        for a, s in enumerate(self.symmetry):
            if s != 0 and kinds[a] == "periodic":
                raise ValueError(
                    f"symmetry[{a}] ('{_AXES[a]}'): a symmetry axis cannot be "
                    f"periodic; set boundaries.{_AXES[a]} to 'pml', 'absorber', "
                    "or 'pec' for the far face (NUMERICS.md §20.2)"
                )
        if any(isinstance(s, PlaneWave) for s in self.sources):
            raise ValueError(
                "a symmetry plane cannot be combined with a plane-wave source "
                "(NUMERICS.md §20.2)"
            )
        return self

    @model_validator(mode="after")
    def _periodic_graded_seam_spacings_match(self) -> "Simulation":
        # NUMERICS.md §15.2, mirrored at construction (cheap, unambiguous —
        # the same 1e-12 relative tolerance as the engine gate in
        # engine/src/core/resolve.cpp): the engine implements the REPLICATE
        # dual-spacing closure at node 0, which on a PERIODIC axis is only
        # correct when the first and last primary spacings match (the
        # periodic-wrap dual length at the seam is their average). phsolver
        # validate hard-rejects the unequal-seam case, so a scene that would
        # die at solver time must fail here, at construction.
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        for a in range(3):
            if kinds[a] != "periodic":
                continue
            q = self._axis_coords_um(a)
            if q is None:  # uniform axis: trivially seam-equal
                continue
            dq_first = q[1] - q[0]
            dq_last = q[-1] - q[-2]
            if abs(dq_first - dq_last) > 1e-12 * max(dq_first, dq_last):
                ax = _AXES[a]
                raise ValueError(
                    f"boundaries.{ax} is periodic but grid.coords.{ax} has "
                    f"unequal first/last primary spacings ({dq_first:.7g} vs "
                    f"{dq_last:.7g} um): the engine's replicate dual-spacing "
                    "closure is only correct at a periodic seam when they "
                    "match, and phsolver validate rejects the scene "
                    "(NUMERICS.md §15.2). Regenerate the mesh seam-"
                    "symmetrically — auto_grid(periodic_axes=...); "
                    "with_auto_grid/with_mesh_overrides pass it from the "
                    "boundaries automatically — or make that axis boundary "
                    "non-periodic (pml/absorber/pec)."
                )
        return self

    @model_validator(mode="after")
    def _centers_inside_realized_domain(self) -> "Simulation":
        # Best-effort early feedback mirroring the engine's domain check
        # (engine/src/core/resolve.cpp): centers must lie inside the REALIZED
        # domain n_axis * dl (NUMERICS.md section 1 — the n >= 4 floor and
        # half-away rounding can make it differ from size_um). The engine
        # computes in meters, so `phsolver validate` remains authoritative at
        # exact boundaries. Structures are exempt: geometry may extend beyond
        # the domain (NUMERICS.md section 9).
        realized = self._realized_um()
        domain = (f"[0, {realized[0]:.9g}] x [0, {realized[1]:.9g}] x "
                  f"[0, {realized[2]:.9g}] um (realized)")

        def check(center, label: str) -> None:
            for c, r in zip(center, realized):
                if not (0.0 <= c <= r):
                    raise ValueError(
                        f"{label}.center_um {tuple(center)} is outside the "
                        f"domain {domain}"
                    )

        for i, s in enumerate(self.sources):
            center = getattr(s, "center_um", None)  # plane waves have none
            if center is not None:
                check(center, f"sources[{i}]")
            elif isinstance(s, PlaneWave):
                axis = _AXES.index(s.axis)
                if not (0.0 <= s.position_um <= realized[axis]):
                    raise ValueError(
                        f"sources[{i}].position_um {s.position_um} ({s.axis} "
                        f"axis) is outside the domain {domain}"
                    )
        for m in self.monitors:
            center = getattr(m, "center_um", None)  # snapshots/flux have none
            if center is not None:
                check(center, f"monitor '{m.name}'")
        return self

    @model_validator(mode="after")
    def _plane_wave_transverse_axes_periodic(self) -> "Simulation":
        # NUMERICS.md section 13 validator, mirrored exactly (no float math,
        # safe to enforce strictly): a normal-incidence plane wave requires
        # both transverse axes periodic.
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        for i, s in enumerate(self.sources):
            if not isinstance(s, PlaneWave):
                continue
            for t, kind in enumerate(kinds):
                if _AXES[t] != s.axis and kind != "periodic":
                    raise ValueError(
                        f"sources[{i}] (plane_wave along {s.axis}): transverse "
                        f"axis '{_AXES[t]}' must be periodic, got '{kind}' "
                        "(NUMERICS.md section 13)"
                    )
        return self

    @model_validator(mode="after")
    def _resolve_subpixel_default(self, info) -> "Simulation":
        # D2 (NUMERICS.md §16): default-ON subpixel smoothing for the common
        # case. When ``subpixel`` is NOT set explicitly, enable the diagonal-KFJ
        # ``tensor`` average (matching Tidy3D's subpixel-on posture, the more
        # accurate out-of-box choice) for a NON-dispersive scene, and fall back
        # to OFF for a dispersive one. (Historical note: the dispersive
        # fallback was added for the "subpixel × Lorentz-ADE" divergence, since
        # root-caused (2026-07-03) to a pole-trapped resonance × CFS-inert PML
        # — smoothing only detunes it, staircased scenes diverge too; see
        # engine/docs/subpixel-dispersion-instability.md. The OFF fallback is
        # KEPT until the dispersive GDS benchmarks are revalidated with
        # with_stabilized_pml() + subpixel on; flipping it changes wire bytes.)
        # The
        # resolved value is marked "set" so it serialises on the wire (the engine
        # field default is off), keeping CPU and GPU runs consistent with this
        # choice. An explicit ``subpixel`` is always respected verbatim; an
        # explicit subpixel-ON dispersive scene only gets a warning, never an
        # override.
        #
        # This is a CONSTRUCTION-time convenience only. When INGESTING an existing
        # wire document (``from_wire_json`` passes context ``wire_ingest``), the
        # absence of ``subpixel`` means the engine default (off) — flipping it
        # would break byte-identical round-trip of older docs — so the resolution
        # is skipped and the field keeps whatever the document stated (or its
        # unset default).
        if (info.context or {}).get("wire_ingest"):
            return self
        dispersive = any(
            getattr(s.medium, "lorentz", None) is not None
            for s in self.structures
        )
        if "subpixel" not in self.model_fields_set:
            if not dispersive:
                # Enable + mark set so it serialises (engine field default = off).
                object.__setattr__(self, "subpixel", True)
                self.__pydantic_fields_set__.add("subpixel")
                if "subpixel_method" not in self.model_fields_set:
                    object.__setattr__(self, "subpixel_method", "contour")
                    self.__pydantic_fields_set__.add("subpixel_method")
            # Dispersive: leave ``subpixel`` at its unset default (off, omitted
            # from the wire = the engine default) — the auto-fallback.
        elif self.subpixel:
            # An explicit ``subpixel=True`` gets the SAME method resolution as the
            # unset auto-default: fill an unset method with "contour_diag" AND mark
            # it set so it serialises. Without this the method stays unset → omitted
            # from the wire → the engine applies ITS default (volume), so the
            # deliberate subpixel-on user would silently get a weaker operator than
            # the auto-default path — two smoothing operators for "the same"
            # subpixel-on scene. Marking it keeps them in lockstep (both contour).
            #
            # This applies to a DISPERSIVE scene too: the method fill is about
            # wire/engine agreement, not about dispersion. Gating it on
            # ``not dispersive`` left an explicit subpixel-on dispersive scene
            # REPORTING subpixel_method="contour" (the field default) while the
            # wire omitted it and the engine silently ran "volume" — an isotropic
            # linear average instead of the diagonal KFJ the model advertises, a
            # first-order operator error on EVERY partially-filled interface cell.
            # The dispersive scene still gets the divergence warning below.
            if "subpixel_method" not in self.model_fields_set:
                object.__setattr__(self, "subpixel_method", "contour")
                self.__pydantic_fields_set__.add("subpixel_method")
        if self.subpixel and dispersive and "subpixel" in self.model_fields_set:
            warnings.warn(
                "subpixel smoothing is enabled on a dispersive (Lorentz) scene. "
                "Dispersive scenes with default-profile PML can diverge "
                "late-time at fine grids — root cause is a pole-assisted "
                "trapped resonance × the CFS-inert PML, NOT the smoothing "
                "itself, but smoothing shifts the resonance and has triggered "
                "it in production scenes "
                "(engine/docs/subpixel-dispersion-instability.md). Use "
                "sim.with_stabilized_pml() with subpixel+dispersive runs.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _warn_dispersive_media_in_pml(self, info) -> "Simulation":
        # Material-aware boundary guidance, mirroring Tidy3D: a stretched-
        # coordinate PML derives its absorbing profile assuming a NON-dispersive
        # medium, so a dispersive (Lorentz) structure extending into the PML can
        # drive a late-time divergence (Oskooi & Johnson, "Distinguishing
        # correct from incorrect PML proposals...", J. Comput. Phys. 2011). The
        # adiabatic absorber (graded electric conductivity, NUMERICS.md §21) is
        # the robust fallback for exactly that regime. WARN — never override —
        # matching Tidy3D's posture, so the wire output is unchanged and the user
        # decides. ``with_auto_boundaries()`` acts on the advice automatically.
        #
        # Skipped on wire ingest (a parsed document is the user's deliberate
        # choice, like the §16 subpixel default), and only for axes whose
        # boundary is actually a PML (an axis already on the absorber is the fix).
        if (info.context or {}).get("wire_ingest") or not self.structures:
            return self
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        crossings = self._dispersive_boundary_crossings()
        hostile = [a for a in range(3) if crossings[a] and kinds[a] == "pml"]
        if hostile:
            axes = ", ".join(_AXES[a] for a in hostile)
            warnings.warn(
                f"a dispersive (Lorentz) medium extends into the PML on axis "
                f"'{axes}': a stretched-coordinate PML assumes a non-dispersive "
                "absorbing region and can diverge there. Switch that boundary to "
                "the adiabatic absorber (NUMERICS.md §21) — "
                "sim.with_auto_boundaries() picks it per axis, or "
                f"boundaries.{_AXES[hostile[0]]}='absorber' / sim.with_absorber().",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _auto_stabilize_dispersive_pml(self, info) -> "Simulation":
        # A dispersive (Lorentz) scene with ANY PML face and the default
        # (CFS-inert) profile can self-oscillate even when every dispersive
        # structure is fully INTERIOR: an undamped pole raises a high-Q trapped
        # resonance, and where its frequency lands near an evanescent/grazing
        # window of the boundary (e.g. just below a transverse-harmonic cutoff),
        # the over-unity evanescent reflection of the plain sigma-graded layer
        # feeds it back — slow exponential late-time growth, cured by the CFS
        # frequency shift. Measured 2026-07-03 (rod 20 cells from the wall,
        # growth e-fold ~20k steps, stable with the raised-alpha profile;
        # staircased controls diverge at 4 of 6 radii, so this is NOT gated on
        # subpixel): engine/docs/subpixel-dispersion-instability.md.
        #
        # Mirroring _resolve_subpixel_default: when the user has NOT tuned any
        # PML knob, AUTO-APPLY the CFS stabilization — kappa_max 5.0 and the
        # GENTLE alpha_max 0.1*(2*eps0/dt) (_AUTO_PML_ALPHA_SCALE; the Tidy3D-
        # parity 0.9 dose reflects ~35% of a propagating guided mode at the
        # default 12 layers), the two levers that never change the slab
        # thickness (so they can never over-thicken a small domain, unlike the
        # layer bump — engine resolve.cpp rejects 2*num_layers >= n_cells). Layer count and sigma_max stay at their
        # defaults; with_stabilized_pml() adds the layer bump for a tighter
        # floor. The two fields are marked "set" so they serialise on the wire
        # (the engine field default is CFS-inert), keeping CPU and GPU runs
        # consistent. If the user HAS tuned the PML explicitly we respect it
        # verbatim and fall back to a WARNING when the alpha they chose is still
        # CFS-inert. Skipped on wire ingest (a parsed document is the user's
        # deliberate choice, like the §16 subpixel default) and when no face is
        # a PML.
        if (info.context or {}).get("wire_ingest") or not self.structures:
            return self
        if not any(getattr(s.medium, "lorentz", None) is not None
                   for s in self.structures):
            return self
        if "pml" not in (self.boundaries.x, self.boundaries.y, self.boundaries.z):
            return self
        pml_knobs = ("pml_num_layers", "pml_kappa_max", "pml_alpha_max",
                     "pml_sigma_max")
        if not any(k in self.model_fields_set for k in pml_knobs):
            # Auto-stabilize: raise kappa + the CFS alpha (fit-safe), mark set so
            # they ride the wire. sigma_max (default 1.5, already in Tidy3D's
            # 2*eps0/dt convention) and the 12-layer count are left untouched.
            # The alpha DOSE is _AUTO_PML_ALPHA_SCALE (0.1*2*eps0/dt), NOT
            # with_stabilized_pml's Tidy3D-parity 0.9: at the default 12 layers
            # the 0.9 dose de-tunes the slab for PROPAGATING waves and reflects
            # ~35% of a guided mode crossing the boundary (measured 2026-07-17;
            # see _AUTO_PML_ALPHA_SCALE above), while 0.1 keeps ~3x the measured
            # stabilization threshold with reflection back at the ~1e-3 level.
            object.__setattr__(self, "pml_kappa_max", _STABLE_PML_KAPPA_MAX)
            self.__pydantic_fields_set__.add("pml_kappa_max")
            object.__setattr__(self, "pml_alpha_max",
                               _AUTO_PML_ALPHA_SCALE * self._two_eps0_over_dt())
            self.__pydantic_fields_set__.add("pml_alpha_max")
            return self
        # The user owns the PML profile — respect it verbatim, but warn if the
        # alpha they set is still CFS-inert (< 0.5% of the sigma peak): the
        # divergence lever is unaddressed and the run may drift late-time.
        if self.pml_alpha_max < _CFS_INERT_FRAC * self._pml_sigma_peak_Sm():
            warnings.warn(
                "dispersive (Lorentz) scene with PML faces and an explicitly-set "
                "but CFS-inert PML profile: an undamped pole can trap a high-Q "
                "resonance whose evanescent tail the plain sigma-graded PML "
                "re-amplifies — late-time divergence even with every dispersive "
                "structure far from the wall, and independent of subpixel "
                "smoothing (engine/docs/subpixel-dispersion-instability.md). "
                "Raise pml_alpha_max, or use sim.with_stabilized_pml() for the "
                "Tidy3D-StablePML-aligned profile.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _flux_planes_inside_domain(self) -> "Simulation":
        # Best-effort mirror of the engine's flux-plane bound (NUMERICS.md
        # section 12: snapped plane index 1 <= kp <= n_axis - 1); phsolver
        # remains authoritative at exact half-cell positions.
        dl = self.grid.dl_um
        for m in self.monitors:
            if not isinstance(m, FluxMonitor):
                continue
            axis = _AXES.index(m.axis)
            # Graded axis: the coordinate-based plane snap (NUMERICS.md
            # section 15.6) is the engine's; skip the uniform-dl best-effort
            # check here (phsolver validate remains authoritative).
            if self._axis_coords_um(axis) is not None:
                continue
            n = realized_cells(self.size_um[axis], dl)
            kp = snapped_plane_index(m.position_um, dl)
            if not (1 <= kp <= n - 1):
                raise ValueError(
                    f"monitor '{m.name}': flux plane at position_um "
                    f"{m.position_um} snaps to {m.axis}-plane index {kp}, "
                    f"outside the interior range [1, {n - 1}] "
                    "(NUMERICS.md section 12)"
                )
        return self

    @classmethod
    def from_wire_json(cls, text: Union[str, bytes]) -> "Simulation":
        """Strictly-typed ingestion of wire JSON, matching the engine's
        nlohmann typing exactly: JSON int -> float fields is accepted,
        string -> number and float -> int are rejected. Use this (not lax
        ``model_validate_json``) when consuming sim.json files."""
        # context wire_ingest: do NOT apply the D2 construction-time subpixel
        # default to a parsed document — absent means the engine default (off),
        # so older docs round-trip byte-identically (see _resolve_subpixel_default).
        return cls.model_validate_json(text, strict=True,
                                       context={"wire_ingest": True})

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Simulation":
        """Load a canonical ``*.json`` simulation document from disk.

        File ingestion keeps :meth:`from_wire_json`'s strict JSON typing; it
        does not execute Python or apply the more permissive construction-time
        coercions.  Missing files and filesystem read errors are surfaced
        unchanged so callers can distinguish them from schema validation.
        """

        source = Path(path).expanduser()
        if source.suffix.lower() != ".json":
            raise ValueError("simulation specs must use a .json filename")
        try:
            mode = source.stat().st_mode
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"simulation spec not found: {source}") from exc
        if stat.S_ISDIR(mode):
            raise IsADirectoryError(f"simulation spec is not a file: {source}")
        if not stat.S_ISREG(mode):
            raise ValueError(f"simulation spec is not a regular file: {source}")
        return cls.from_wire_json(source.read_text(encoding="utf-8"))

    def _wire_exclude(self):
        # Omit additive-optional fields that were never explicitly set so
        # older documents round-trip byte-identically and stay consumable by
        # earlier-minor parsers that reject unknown keys; the engine applies
        # the same defaults. pml_num_layers entered the wire in schema 1.1.0
        # (default 12); run.shutoff in 1.3.0 (default 1e-5, NUMERICS.md §7).
        exclude: dict = {}
        if "pml_num_layers" not in self.model_fields_set:
            exclude["pml_num_layers"] = True
        # The §11 CPML profile knobs entered the wire in schema 1.8.0 (defaults
        # m=3 / kappa_max=3 / alpha_max=0.24, bit-identical to the prior
        # hardcoded profile); omit each when unset so earlier-minor parsers
        # accept the document and golden specs round-trip byte-identically.
        for _f in ("pml_m", "pml_kappa_max", "pml_alpha_max", "pml_sigma_max"):
            if _f not in self.model_fields_set:
                exclude[_f] = True
        # The §21 absorber knobs entered the wire in schema 1.12.0 (defaults
        # 40 layers / m=3); omit each when unset so earlier-minor parsers accept
        # the document and golden specs round-trip byte-identically.
        for _f in ("absorber_num_layers", "absorber_m"):
            if _f not in self.model_fields_set:
                exclude[_f] = True
        # subpixel entered the wire in schema 1.4.0 (default false, NUMERICS.md
        # §16); omit it when unset so 1.3-and-earlier parsers still accept the
        # document and golden specs round-trip byte-identically.
        if "subpixel" not in self.model_fields_set:
            exclude["subpixel"] = True
        # subpixel_method entered the wire in schema 1.7.0 (default "volume",
        # NUMERICS.md §16.5); omit when unset so earlier-minor parsers accept the
        # document and golden specs round-trip byte-identically.
        if "subpixel_method" not in self.model_fields_set:
            exclude["subpixel_method"] = True
        # symmetry entered the wire in schema 1.11.0 (NUMERICS.md §20); omit when
        # all-zero (the no-symmetry default) so earlier-minor parsers accept the
        # document and golden specs round-trip byte-identically.
        if self.symmetry == (0, 0, 0):
            exclude["symmetry"] = True
        if "shutoff" not in self.run.model_fields_set:
            exclude["run"] = {"shutoff"}
        return exclude or None

    def to_wire_dict(self) -> dict:
        """Canonical JSON-level dict: defaults materialized, unset optionals
        (the unused run_time_s/n_steps key, an unset pml_num_layers) omitted."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True,
                               exclude=self._wire_exclude())

    def to_wire_json(self, indent: int = 2) -> str:
        return self.model_dump_json(by_alias=True, exclude_none=True,
                                    exclude=self._wire_exclude(), indent=indent)

    def to_file(self, path: Union[str, Path]) -> Path:
        """Atomically save this immutable model as canonical ``*.json``.

        The parent directory must already exist: a typo in a beta user's path
        must not silently create a new directory tree.  The canonical wire
        document is written to a sibling temporary file, flushed to disk, and
        published with :func:`os.replace`, so readers observe either the old
        complete document or the new complete document.  The model itself is
        unchanged and the saved path is returned for convenient logging.
        """

        target = Path(path).expanduser()
        if target.suffix.lower() != ".json":
            raise ValueError("simulation specs must use a .json filename")
        parent = target.parent
        if not parent.exists():
            raise FileNotFoundError(
                f"simulation spec parent directory does not exist: {parent}")
        if not parent.is_dir():
            raise NotADirectoryError(
                f"simulation spec parent is not a directory: {parent}")
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"simulation spec path is a directory: {target}")

        temporary: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(self.to_wire_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target

    # -- Visualization (photonhub.viz; design doc docs/viz-layer-design.md) ---
    # Thin delegations: the rendering logic lives entirely in photonhub.viz so
    # these pydantic models stay clean. Imported lazily so matplotlib is only
    # loaded when a plot is actually requested.

    def plot(self, x=None, y=None, z=None, *, ax=None, legend=True,
             grid=False, **kw):
        """2D analytic cross-section of the scene on a cut plane (exactly one
        of x/y/z, in microns). ``grid=True`` overlays the Yee mesh cell edges
        (the resolution sanity-check). Returns a matplotlib ``Axes``. See
        :func:`photonhub.viz.plot`."""
        from ..viz import plot as _plot
        return _plot(self, x=x, y=y, z=z, ax=ax, legend=legend, grid=grid, **kw)

    def plot_eps(self, x=None, y=None, z=None, *, ax=None, cmap=None,
                 grid=False, **kw):
        """Rasterized permittivity heatmap (the §9 hard sample the solver
        takes) on a cut plane. ``grid=True`` overlays the cell edges. Returns a
        matplotlib ``Axes``. See :func:`photonhub.viz.plot_eps`."""
        from ..viz import plot_eps as _plot_eps
        return _plot_eps(self, x=x, y=y, z=z, ax=ax, cmap=cmap, grid=grid, **kw)

    def plot_3d(self, **kw):
        """Interactive 3D geometry as a plotly ``Figure`` (requires the
        ``photonhub[viz]`` extra). See :func:`photonhub.viz.plot_3d`."""
        from ..viz import plot_3d as _plot_3d
        return _plot_3d(self, **kw)

    def preview(self, **kw):
        """Interactive Jupyter scrubber over the cut plane (slider + axis/grid/ε
        toggles). Requires the ``photonhub[viz]`` extra and a notebook. See
        :func:`photonhub.viz.interactive_preview`."""
        from ..viz import interactive_preview as _preview
        return _preview(self, **kw)

    def cost_estimate(
        self,
        *,
        rate_usd_per_tcell_step: float = ...,
        throughput_gcells_per_s: float = ...,
    ) -> "CostEstimate":
        """Pure-Python dollar / memory / output / wall-time estimate (the
        plan's "estimate in dollars before you press run"). See
        :func:`photonhub.cost.estimate_cost`. Cell count, dt and step count
        match the engine's resolve.cpp, so the dollar figure tracks what
        ``phsolver`` will run; it is exact for a full-duration run (auto-shutoff
        can only make it cheaper)."""
        # Forward only explicitly-passed overrides so the single source of the
        # default rate/throughput stays in photonhub.cost.
        kwargs = {}
        if rate_usd_per_tcell_step is not ...:
            kwargs["rate_usd_per_tcell_step"] = rate_usd_per_tcell_step
        if throughput_gcells_per_s is not ...:
            kwargs["throughput_gcells_per_s"] = throughput_gcells_per_s
        return estimate_cost(self, **kwargs)
