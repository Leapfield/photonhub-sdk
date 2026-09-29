"""Top-level simulation model, the root of the wire format."""

import contextlib
import contextvars
import logging
import math
import os
import stat
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from pydantic import AliasChoices, Field, PrivateAttr, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from ..cost import CostEstimate, quote
from ..constants import c0, eps0, mu0
from .base import (
    MAX_INT32,
    DftPrecisionName,
    FieldPrecisionName,
    FrozenModel,
    PositiveUm,
    SubpixelMethodName,
    _monitor_name_key,
)
from .grid import (
    _structure_index,
    auto_mesh,
    axis_min_cells,
    graded_primary_spacings,
    GradedMesh,
    GradedMeshAxis,
    MeshType,
    quarter_snap_dft_face,
    realized_cells,
    resolved_cell_counts,
    snap_mixed_plane,
    snapped_plane_index,
    UniformMesh,
    yee_axis_offsets,
)
from .medium import Background, Boundaries
from .monitors import (
    ProfileMonitor,
    PowerMonitor,
    TimeMonitor,
    MonitorType,
    mode_port_physical_polarization,
)
from .run import RunSpec
from .sources import ModeSource, PlaneWave, PointDipole, SourceType, TfsfBox
from . import declarative as _decl
from . import frame as _frame
from ._bounds import geometry_bounds_um
from .authoring import Domain, GaussianBeam, Mesh, Port
from .source_time import _C0_M_PER_S, CW
from .structures import Box, MaterialEntry, Medium, Structure, is_material
from .._compat import caller_stacklevel, legacy_keywords

# The run-length cap of a simulation that gives no ``run``: transits of the
# longest domain extent at the highest index. The auto-shutoff ends a run
# well before it (a straight strip decays in 4.7 transits at shutoff 1e-7,
# the Y-junction of notebook 37 in 3.3, phase-5 study); the cap is the
# ceiling the cost estimate quotes and the length a device that never decays
# runs to, with a warning.
DEFAULT_TRANSITS = 40.0


def _medium_poles(medium):
    """Every Lorentz pole on a medium, across both wire spellings.

    ``lorentz`` is the legacy OPTIONAL SINGLE pole (NUMERICS.md section 19.5)
    and ``poles`` is the list form. Wrapping the single one in ``list(...)``
    iterates a pydantic model into (name, value) tuples instead, which is how
    two validators here quietly grew an AttributeError on any scene that used
    the legacy spelling.
    """
    single = getattr(medium, "lorentz", None)
    out = [single] if single is not None else []
    out.extend(getattr(medium, "poles", None) or [])
    return out


def _loaded_document(info) -> bool:
    """True when a validator runs on a document being loaded from the wire
    (context ``wire_ingest``), where the engine is the authority and a
    document that breaks a rule must still load so it can be fixed; False on
    construction and on the validated copy an edit makes (``edit_copy``),
    which must refuse what direct construction refuses."""
    ctx = info.context or {}
    return bool(ctx.get("wire_ingest")) and not ctx.get("edit_copy")


def _ade_nyquist_eps(eps_inf, lorentz, drude, dt) -> float:
    """A dispersive medium's discrete permittivity at the temporal Nyquist of
    step ``dt`` (NUMERICS.md §19.4, the engine's ade_nyquist_eps):
    eps_inf - sum delta_eps*x/(1-x) - sum (wp*dt/2)^2, x = (omega0*dt/2)^2, over
    ``lorentz`` [(omega0, delta_eps)] and ``drude`` [wp] in rad/s; -inf once a
    Lorentz pole reaches its own Nyquist (x >= 1)."""
    e = float(eps_inf)
    for w0, deps in lorentz:
        x = 0.25 * w0 * w0 * dt * dt
        if not x < 1.0:
            return -math.inf
        e -= deps * x / (1.0 - x)
    for wp in drude:
        h = 0.5 * wp * dt
        e -= h * h
    return e


def _ade_max_courant(eps_inf, lorentz, drude, dt, courant) -> float:
    """The largest courant C' <= ``courant`` meeting the §19.4 bound, dt
    scaling with it (the engine's ade_max_courant: 80 bisection steps)."""
    unit = dt / courant
    lo, hi = 0.0, courant
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _ade_nyquist_eps(eps_inf, lorentz, drude, mid * unit) >= mid * mid:
            lo = mid
        else:
            hi = mid
    return lo


def _ade_max_dt(eps_inf, lorentz, drude, dt, courant) -> float:
    """The largest dt' <= ``dt`` meeting the §19.4 bound at the same courant
    (the engine's ade_max_dt)."""
    c2 = courant * courant
    lo, hi = 0.0, dt
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _ade_nyquist_eps(eps_inf, lorentz, drude, mid) >= c2:
            lo = mid
        else:
            hi = mid
    return lo


def _structure_axis_span(geometry, axis_index: int):
    """(lo, hi) extent of one geometry along an axis, or None if not derivable.

    Used only by the quasi-2-D subpixel warning: a conservative None means "do
    not warn about this shape".
    """
    kind = getattr(geometry, "type", None)
    center = getattr(geometry, "center_um", None)
    if kind == "box":
        c = center[axis_index]
        h = 0.5 * geometry.size_um[axis_index]
        return c - h, c + h
    if kind == "sphere":
        c = center[axis_index]
        return c - geometry.radius_um, c + geometry.radius_um
    if kind == "cylinder":
        c = center[axis_index]
        along = _AXES[axis_index] == geometry.axis
        h = 0.5 * geometry.length_um if along else geometry.radius_um
        return c - h, c + h
    if kind == "polyslab" and _AXES[axis_index] == geometry.axis:
        lo, hi = geometry.slab_bounds_um
        return float(lo), float(hi)
    return None

SCHEMA_VERSION = "1.21.0-alpha.1"
SUPPORTED_SCHEMA_MAJOR = 1

_AXES = "xyz"
# The constructor's declaration aliases (Simulation.size_um / .grid), mapped to
# their field names so with_changes() takes the spelling a simulation is written in.
_FIELD_ALIASES = {"domain": "size_um", "mesh": "grid"}
# The fields Simulation.model_copy may still replace as a field copy on a
# resolved simulation: wire content no fit or port solve reads back.
_RAW_COPY_FIELDS = frozenset({"sources", "monitors", "subpixel", "subpixel_method",
                              "field_precision", "dft_precision"})
# The fields the domain/mesh/run fit and the declarative resolution move
# without marking them set: a validated copy passes them explicitly.
_RESOLVED_FIELDS = ("size_um", "grid", "run", "structures", "sources", "monitors",
                    "background", "origin_um")
# The fields whose values are coordinates in the stored (fitted) frame. A
# domain= fit moves that frame off the user's, so an edit of one of them on
# such a simulation cannot be read as the user wrote it.
_FRAME_FIELDS = frozenset({"size_um", "grid", "structures", "sources", "monitors", "origin_um"})


# The fields the declarative resolution reads from the scene it solves the
# ports on (the port windows, cells and modes): sources and monitors a
# model_copy swapped in over the resolved ports stay valid only while none of
# these moves.
_PORT_SOLVE_FIELDS = frozenset({"size_um", "grid", "origin_um", "structures", "background", "boundaries",
                                "symmetry", "bloch_k_per_um", "pml_num_layers", "absorber_num_layers"})


def _tupled(update: dict) -> dict:
    """``update`` with its list or iterator sequence fields as tuples, as
    validation makes them: a list is kept as given by a field copy, and a
    generator would be used up by the first look at it. Anything else (a
    tuple, None, a single model) is left for validation to accept or refuse."""
    return {k: (tuple(v) if k in ("structures", "sources", "monitors") and isinstance(v, (list, Iterator)) else v)
            for k, v in update.items()}


# The simulation an edit (with_changes, model_copy, the with_* helpers) is
# copying, while it constructs the copy: the advice about the caller's own
# inputs, a monitor frequency the pulse barely drives or a wavelength typed as
# a frequency, is not repeated when the original drew the same advice (beta
# review API-06); advice the edit brings in is given, and every refusal runs.
_EDITING: contextvars.ContextVar = contextvars.ContextVar("photonhub_simulation_editing", default=None)


@contextlib.contextmanager
def _editing(original):
    token = _EDITING.set(original)
    try:
        yield
    finally:
        _EDITING.reset(token)


def _advised_before(advice: str, found) -> bool:
    """True inside an edit whose original drew ``advice`` (the name of the
    Simulation method that finds it) for every one of the ``found`` findings."""
    original = _EDITING.get()
    return original is not None and set(found) <= set(getattr(original, advice)())

_LOG = logging.getLogger(__name__)

# eta0 = mu0 * c0 — the vacuum wave impedance. Bridges the CPML sigma/alpha peak
# between the dimensionless "units of 2*eps0/dt" convention and the engine's
# S/m: since eps0*c0 = 1/eta0, the peak-conductivity unit 2*eps0/dt reduces to a
# form needing only eta0, the cell spacing, and the Courant number (see
# Simulation._two_eps0_over_dt).
_ETA0 = mu0 * c0
_EPS0 = eps0
# API-06: a monitor frequency where the normalizing pulse's spectral amplitude
# is below this fraction of its peak reads noise over almost no drive.
_WEAK_SPECTRAL_AMPLITUDE = 1e-3

# Stabilized-CPML profile: kappa_max = 5.0 and a CFS alpha_max = 0.9, both
# quoted in 2*eps0/dt units (that profile also uses 40 layers and
# sigma_max = 1.0). The engine already reads sigma in that convention
# (pml_sigma_max, default 1.5) but alpha in absolute S/m (pml_alpha_max) — which
# is precisely WHY the default alpha (0.24 S/m) is inert — so we convert
# 0.9 * 2*eps0/dt to S/m at the scene's timestep (_two_eps0_over_dt). alpha and
# kappa are the fit-safe stability levers (they never change the slab thickness);
# the layer bump is applied only by the explicit with_stabilized_pml().
_STABLE_PML_LAYERS = 40
_STABLE_PML_KAPPA_MAX = 5.0
_STABLE_PML_ALPHA_SCALE = 0.9
# The AUTO-stabilizer dose (_auto_stabilize_dispersive_pml) is anchored to the
# BAND, not to the timestep: alpha_max = _AUTO_PML_ALPHA_SCALE * eps0 * omega0,
# omega0 = 2*pi times the highest carrier frequency among the sources. Both
# sides of the trade-off are set by the optical frequency, not by dt:
# - the CFS absorptive term sigma*omega*eps0/(alpha^2 + (omega*eps0)^2) loses
#   in-band absorption once alpha passes omega*eps0, so the 12-layer slab
#   REFLECTS (1-D normal incidence, vacuum, kappa 5: alpha 1.0e4 S/m -86 to
#   -89 dB, 1.6e4 -67 to -74 dB, 2.3e4 -53 to -60 dB at dl 0.04 and 0.02 um);
# - the trapped-resonance divergence the dose cures (a dispersive pole-boosted
#   mode fed by the evanescent reflection of a CFS-inert PML) stops once the
#   CFS pole sits at a fraction of the mode's optical frequency (rod probe,
#   engine/docs/subpixel-dispersion-instability.md, 25 c/lambda in Si, mode
#   2.93e14 Hz, source 1.93e14 Hz, 588k steps: alpha 0.24 diverges at ~110k,
#   2e3 grows from ~300k, 3e3 grows faintly from ~440k, 5e3 and above flat).
# 1.0 * eps0*omega0 (1.08e4 S/m at 1550 nm) sits 2x above 5e3 S/m, which is
# flat through 588k steps and rises faintly after ~880k; the dose itself stays
# flat through 1.18M steps (40 ps)
# and reflects -85 to -96 dB at 153-233 THz (a 40 THz pulse at 193.4 THz),
# level with the default profile, at every resolution; below the carrier the
# CFS term bites: -58 dB at half the carrier frequency. The old dose,
# 0.1 * 2*eps0/dt, grew as 1/dl: 2.3e4 S/m at dl 0.04 um (-53 dB), 4.6e4 at
# 0.02 (-36 dB), 9.3e4 at 0.01 (-24 dB) (beta review CORE-03). The probe mode
# sat at 1.52x the carrier and was flat at 0.30 eps0*omega_mode; if the
# threshold scales with the mode frequency, this dose covers trapped modes up
# to about 3x the carrier.
_AUTO_PML_ALPHA_SCALE = 1.0
# The band a scene without a source anchors to: its wlen0_um, else its
# shortest wlens_um, else 1550 nm. Such a scene cannot run, but it can gain a
# source through a copy that does not re-resolve the profile (with_changes,
# with_oblique_plane_wave), so it is stabilized for a nominal band.
_NOMINAL_BAND_WLEN_UM = 1.55
# CFS-inert threshold for an explicitly set profile, in the same eps0*omega0
# units: the rod probe still grows at 0.19 and 0.28 (2e3 and 3e3 S/m) and is
# flat from 0.46 (5e3 S/m) up.
_CFS_INERT_SCALE = 0.4

# with_absorber / with_auto_boundaries size the section 21 absorber
# physically: at least this many wavelengths in the background medium at the
# lowest source frequency (freq0 - fwidth of a pulse, freq0 of a cw), and
# never fewer than the engine's 40 layers. With the ramp strength fixed per
# physical thickness beyond 40 layers (NUMERICS.md section 21), the
# reflection depends on the thickness in wavelengths, not on the mesh (normal
# incidence, vacuum, m 3, a 40 THz pulse at 193.4 THz read at 153-233 THz,
# worst frequency 153 THz): -16 dB at one wavelength of 193.4 THz, -40 dB at
# two (dl 0.08 to 0.01 um), and -52 dB at the two wavelengths of 153.4 THz
# chosen here (-60 dB in SiO2). In a background of index n it cannot go below
# about -139/n dB.
_ABSORBER_WAVELENGTHS = 2.0
_ABSORBER_MIN_LAYERS = 40


# Two media whose indices at the band centre differ by less than this are the
# same material to the port-medium check below. It is the largest index error
# `Material.medium(band_um=...)` accepts before warning that its dispersive fit
# no longer stands for the material, so a library material given as a band fit
# on the structure and by name on the port (2.6e-5 apart for silicon over
# 1.5-1.6 um) never warns, while an index typed by hand against the library's
# (3.48 against silicon's 3.4757, 4.3e-3 apart) does. The printed indices carry
# four decimals, so two indices that fail the check never print alike.
_PORT_MEDIUM_INDEX_TOL = 1e-3


def _warn_port_medium_mismatch(port, containing, wlen0) -> None:
    """Warn when a port's declared ``medium`` is not the material of the
    structure the port sits on.

    ``containing`` lists every structure whose bounds contain the port centre,
    in list order (after a symmetry fold has dropped the mirrored half, so a
    structure is named by its ``name`` or its shape and index, never by a list
    position the caller might not recognise); the host is the LAST
    (NUMERICS.md §9 paint order, last wins), what the scene draws at the port.
    ``Port.medium`` never reaches the mode solve, which reads the scene; it
    sizes the port's cell and is the medium of the guide extension the fit
    paints from one cell inside the port plane out through the wall, when the
    guide stops short of it. A disagreement therefore puts one material at the
    port plane and another in the device behind it.

    Structure order is named only when an EARLIER structure containing the port
    centre has the port's material: a cladding drawn after the guide buries it,
    and reordering is the fix. Otherwise it is a material mismatch. A warning
    rather than an error, because an overlapping cladding, a mode-matching stub
    or a deliberately off-guide probe are all legitimate."""
    dummy = Box(center_um=(0.0, 0.0, 0.0), size_um=(1.0, 1.0, 1.0))
    wlen = float(wlen0) if wlen0 is not None else 1.55
    n_port = _structure_index(Structure(geometry=dummy, medium=port.medium), wlen)

    def label(st, n=None):
        name = getattr(st, "name", None)
        if name:
            return f"structure {name!r}" + (f" (index {n:.4f})" if n is not None else "")
        return f"a {type(st.geometry).__name__}" + (f" of index {n:.4f}" if n is not None else "")

    host = containing[-1]
    n_host = _structure_index(host, wlen)
    if abs(n_port - n_host) < _PORT_MEDIUM_INDEX_TOL:
        return
    matching = [st for st in containing[:-1] if abs(_structure_index(st, wlen) - n_port) < _PORT_MEDIUM_INDEX_TOL]
    head = f"port {port.name!r}: medium= has index {n_port:.4f} at {wlen:g} um"
    if matching:
        message = (
            f"{head}, the index of {label(matching[-1])}, but {label(host, n_host)} is listed after it and "
            "also contains the port centre. List order is paint order (last wins, NUMERICS.md section 9), "
            "so the later structure buries the earlier one at the port, while a guide extension through the "
            "wall, when the fit builds one, is painted in medium=. If the later structure is a cladding, "
            "list it first; otherwise drop medium= to continue the scene's.")
    else:
        message = (
            f"{head}, but {label(host, n_host)}, which the port centre falls in, differs from it by "
            f"{abs(n_port - n_host):.2g}: a material mismatch (two media within {_PORT_MEDIUM_INDEX_TOL:g} "
            "count as one material). A guide extension through the wall, when the fit builds one, is "
            "painted in medium=, so the port plane and the guide behind it would be different materials. "
            "Give the port the structure's medium, or drop medium= to take it.")
    warnings.warn(message, UserWarning, stacklevel=caller_stacklevel())


def _quarter_snapped_dft_monitors(monitors, *, size_um, grid, axis_min_cells=(4, 4, 4)):
    """NUMERICS.md §12 quarter-cell auto-snap over a monitor list.

    Pure function of ``(monitors, size_um, grid, axis_min_cells)``: returns
    ``(new_monitors, notes)`` where ``notes`` carries one line per ADJUSTED
    monitor (empty =
    nothing moved and ``new_monitors`` is the input, element-identical). For
    each :class:`ProfileMonitor` and each axis whose listed components mix
    Yee offsets, both box faces are put through
    :func:`~photonhub.components.grid.quarter_snap_dft_face`: a face whose
    per-component engine snap already agrees, including domain-edge faces
    rescued by the engine's index clamp, is left byte-identical, anything
    else moves to the nearest local ``(k + 1/4)`` quarter-cell plane. A box
    face and its quarter point snap to the SAME cell, so a scene the engine
    already accepted keeps its exact recorded region; only rejected or
    rounding-sensitive placements change at all.

    Raises ``ValueError`` for a sub-half-cell box straddling a cell boundary
    (its two faces would collapse onto one quarter point or invert), and when
    a ``mode_port`` window can no longer fit on its snapped plane.

    ``axis_min_cells`` is the simulation's §1 per-axis cell floor
    (:meth:`Simulation._axis_min_cells`: 1 on a plain periodic axis, 4
    elsewhere). Counting a one-cell quasi-2D axis with the default floor of
    4 snapped a full-extent plane's high face to the quarter point of cell 1,
    beyond the one cell that exists, and the engine rejected every port plane
    on such an axis.
    """
    coords = getattr(grid, "coords", None)
    dl = grid.dl_um
    axis_q = []
    for a, axis in enumerate(_AXES):
        q = getattr(coords, axis) if coords is not None else None
        n = len(q) if q is not None else realized_cells(size_um[a], dl, axis_min_cells[a])
        axis_q.append((q, n))

    out, notes = [], []
    for m in monitors:
        if not isinstance(m, ProfileMonitor):
            out.append(m)
            continue
        lo = [m.center_um[a] - m.size_um[a] / 2.0 for a in range(3)]
        hi = [m.center_um[a] + m.size_um[a] / 2.0 for a in range(3)]
        moved = []  # (axis_index, "lo"/"hi", old_um, new_um)
        for a in range(3):
            offsets = yee_axis_offsets(m.fields, a)
            q, n = axis_q[a]
            for which, faces in (("lo", lo), ("hi", hi)):
                snapped = quarter_snap_dft_face(
                    faces[a], offsets, n_cells=n, dl_um=dl, coords_um=q)
                if snapped is not None and snapped != faces[a]:
                    moved.append((a, which, faces[a], snapped))
                    faces[a] = snapped
            if m.size_um[a] > 0.0 and hi[a] <= lo[a]:
                raise ValueError(
                    f"monitor '{m.name}': the §12 quarter-snap of its "
                    f"{_AXES[a]}-axis faces "
                    f"[{m.center_um[a] - m.size_um[a] / 2.0:.9g}, "
                    f"{m.center_um[a] + m.size_um[a] / 2.0:.9g}] um would "
                    f"collapse or invert the box (snapped [{lo[a]:.9g}, "
                    f"{hi[a]:.9g}] um): a box thinner than half a cell "
                    "straddling a cell boundary cannot satisfy the engine's "
                    "per-component region snap (NUMERICS.md §12). Place both "
                    "faces strictly inside ONE first half-cell (e.g. center "
                    "the box on a (k + 1/4)*dl_um plane), give the axis "
                    "size 0 (a plane — snapped automatically), or widen it "
                    "past a full cell."
                )
        if not moved:
            out.append(m)
            continue

        update = {
            "center_um": tuple(
                0.5 * (lo[a] + hi[a])
                if any(mv[0] == a for mv in moved) else m.center_um[a]
                for a in range(3)),
            "size_um": tuple(
                hi[a] - lo[a]
                if any(mv[0] == a for mv in moved) else m.size_um[a]
                for a in range(3)),
        }

        # A snapped face can shrink the recorded plane from under a mode_port
        # solve window authored flush against the old extents; clamp the
        # window into the new plane (post-processing metadata only — the
        # engine validates and discards it) instead of letting
        # _modal_port_rules reject the auto-snapped scene. Only a window that
        # FIT THE PLANE AS AUTHORED is followed: one that already stuck out is
        # the author's error and stays for _modal_port_rules to reject.
        port = m.mode_port
        zero_axes = [a for a in range(3) if update["size_um"][a] == 0.0]
        if port is not None and len(zero_axes) == 1:
            normal = zero_axes[0]
            transverse = tuple(a for a in range(3) if a != normal)
            w_center, w_size, port_moved = list(port.center_um), list(
                port.size_um), False
            for local, a in enumerate(transverse):
                w_lo = port.center_um[local] - port.size_um[local] / 2.0
                w_hi = port.center_um[local] + port.size_um[local] / 2.0
                orig_lo = m.center_um[a] - m.size_um[a] / 2.0
                orig_hi = m.center_um[a] + m.size_um[a] / 2.0
                if w_lo < orig_lo - 1e-12 or w_hi > orig_hi + 1e-12:
                    continue  # never fit the authored plane — not ours to fix
                new_lo, new_hi = max(w_lo, lo[a]), min(w_hi, hi[a])
                if new_lo > w_lo + 1e-12 or new_hi < w_hi - 1e-12:
                    if not (new_hi - new_lo > 0.0):
                        raise ValueError(
                            f"monitor '{m.name}': the §12 quarter-snap moved "
                            f"its DFT plane off the mode_port window on "
                            f"'{_AXES[a]}' ([{w_lo:.9g}, {w_hi:.9g}] um vs "
                            f"snapped plane [{lo[a]:.9g}, {hi[a]:.9g}] um); "
                            "re-center the window inside the plane."
                        )
                    w_center[local] = 0.5 * (new_lo + new_hi)
                    w_size[local] = new_hi - new_lo
                    port_moved = True
                    moved.append((a, f"mode_port window {'lo' if new_lo != w_lo else 'hi'}",
                                  w_lo if new_lo != w_lo else w_hi,
                                  new_lo if new_lo != w_lo else new_hi))
            if port_moved:
                update["mode_port"] = port.model_copy(update={
                    "center_um": tuple(w_center), "size_um": tuple(w_size)})

        out.append(m.model_copy(update=update))
        details = ", ".join(
            f"{_AXES[a]} {which} {old:.9g} -> {new:.9g} um"
            for a, which, old, new in moved)
        notes.append(
            f"monitor '{m.name}': §12 quarter-snap adjusted {details} "
            "(box faces on/beyond a cell's half-cell plane, or on an interior "
            "cell boundary, make the per-component region snap of mixed-Yee-"
            "offset components disagree or depend on float rounding; each "
            "face was nudged to the nearest (k + 1/4) quarter-cell plane of "
            "its local cell — NUMERICS.md §12)")
    return (tuple(out) if notes else monitors), notes


class Simulation(FrozenModel):
    """Complete simulation description. Serializes 1:1 to the JSON wire
    format consumed by ``phsolver`` (schemas/GOVERNANCE.md).

    The cross-field validators here are best-effort early feedback mirroring
    the engine's checks where they are cheap and unambiguous; ``phsolver
    validate`` remains authoritative (notably for the plane-wave/PML
    intersection rule). One check is resolved rather than mirrored: DFT
    field-monitor box faces are AUTO-SNAPPED at construction to §12
    quarter-cell planes wherever the engine's per-component region snap would
    reject them or pass on float rounding luck (see
    :class:`~photonhub.components.monitors.ProfileMonitor`); ingestion via
    ``from_wire_json``/``from_file`` never adjusts a document."""

    schema_version: str = SCHEMA_VERSION
    # ``size_um`` and ``grid`` are the wire fields. Each also accepts, under the
    # aliases ``domain=`` and ``mesh=``, a client-only declaration (design spec
    # §4.1, §5.1, §5.2) that _fit_domain_mesh_run resolves into the wire value
    # before any other validator runs; the declarations are skipped in the
    # schema and the wire key stays the first alias, so the document and the
    # generated schema are unchanged.
    size_um: Union[Tuple[PositiveUm, PositiveUm, PositiveUm], SkipJsonSchema[Domain]] = Field(
        validation_alias=AliasChoices("size_um", "domain"))
    grid: Union[MeshType, SkipJsonSchema[Mesh]] = Field(
        validation_alias=AliasChoices("grid", "mesh"))
    # Optional in Python: a simulation without ``run`` takes the transit cap
    # (phase-5 plan, refinement 3) and ends at the auto-shutoff in practice.
    # Required on the wire: the schema module keeps ``run`` in the required
    # list and an ingested document without it is rejected below.
    run: RunSpec = Field(default_factory=lambda: RunSpec(transits=DEFAULT_TRANSITS))
    # The user-frame position of the wire's low corner when the domain was
    # fitted around the device (design spec §4.4): the fit translates every
    # positional field by -origin_um into the corner frame the wire and every
    # validator use; results and plots add it back. Client-only; (0, 0, 0) for
    # a domain given by hand or a document loaded from the wire.
    origin_um: SkipJsonSchema[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    # A materials-library entry may stand here too (resolved at construction);
    # anything else is validated as a Background.
    background: Union[Background, MaterialEntry] = Background()
    # NUMERICS.md section 11: layer count for every "pml" boundary axis. The
    # engine default is 12; an UNSET value is omitted from the wire format
    # (see to_wire_dict) so Phase-0 documents round-trip byte-identically and
    # remain consumable by schema-1.0 parsers that reject unknown keys.
    #
    # 4 is a hard FLOOR, not a recommendation. Measured spurious reflection of
    # a normally incident pulse (vs a reflection-free reference domain):
    # 1.9e-05 at 4 layers, 5.5e-09 at 8, 2.2e-10 at 12, 4.0e-11 at 16. The
    # default of 12 is quiet; 8 is the practical minimum for a clean boundary;
    # 4 should be treated as "cheap and lossy".
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
    # in the dimensionless 2*eps0/dt convention (like pml_sigma_max) — its
    # the stabilized profile uses alpha_max = 0.9 — and this asymmetry (sigma dt-relative,
    # alpha absolute S/m) is why the default alpha reads as inert. Raising kappa_max
    # + alpha_max is the "stabilized" recipe for a grazing/long-run/dispersive
    # scene that diverges: a DISPERSIVE (Lorentz) scene gets kappa 5.0 +
    # alpha eps0*omega0 of the highest source carrier applied AUTOMATICALLY at construction
    # (_auto_stabilize_dispersive_pml), and ``with_stabilized_pml`` builds the
    # full stabilized-CPML copy (also adding the layer bump).
    #   pml_sigma_max peak conductivity in 2*eps0/dt units. The DEFAULT
    #     1.5 matches the standard stabilized-profile PML sigma_max exactly — a resolution-
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
    # curved (Cylinder/Polygon/Sphere, supersampled §16.7) interfaces are
    # smoothed on BOTH uniform and graded meshes (CPU, single GPU, and multi-GPU).
    # Only the off-diagonal ``tensor_full`` on a graded mesh remains deferred
    # (§16.6; the engine's reference and GPU solvers reject it). Uniform
    # ``tensor_full`` is available on GPU subject to its engine-wide
    # lossless/non-dispersive combination rules.
    subpixel: bool = False
    # NUMERICS.md §16.5/§16.8/§16.11: which smoothing to apply when ``subpixel``
    # is on. Six operators: "volume" (isotropic volume average, bit-identical to
    # schema < 1.7.0); "tensor" (diagonal anisotropic KFJ); "tensor_full" (full
    # off-diagonal KFJ); "contour" (diagonal KFJ fed the exact §16.10 Polygon
    # fill == standard polarized averaging plus the exact vertical-wall
    # fill — the DEFAULT); and the rigorous contour-path EPs (Mohammadi-Nadgaran-
    # Agio 2005, contour-path averaging): "contour_diag" (the paper's
    # per-component scalar CP-EP) and "contour_full" (its full off-diagonal Kottke
    # tensor for tilted/curved walls). The FIELD default is "contour" to MATCH
    # the effective construction default: _resolve_subpixel_default auto-enables
    # subpixel+contour on a non-dispersive scene and fills an unset method with
    # "contour" on any explicit subpixel-on, so the declared default and the
    # auto/explicit-on paths agree. contour == tensor == contour_diag on axis-
    # aligned interfaces (they reduce to arithmetic/harmonic) and differ only on
    # tilted/curved cells; contour is the standard match, contour_diag the
    # rigorous CP-EP alternative.
    # Omitted from the wire when unset AND smoothing is off (see _wire_exclude);
    # with smoothing on it is always written, so the operator the engine runs is
    # the one on the model, whatever the engine's own absent default.
    subpixel_method: SubpixelMethodName = "contour"
    # Schema 1.19 — NUMERICS.md §23 field STORAGE precision. "fp16" stores the
    # six field arrays as binary16 behind exact power-of-two per-run scales
    # (lambda for E, lambda*2^9 for H); all arithmetic, coefficients, CPML psi,
    # ADE state, and DFT accumulators stay fp32/fp64. Opt-in fast lane for
    # bandwidth-bound GPU runs, validated by the §23 two-tier regime (not the
    # §8 bit-equality gate); fp32 (default) is byte-identical to prior
    # releases. Single-GPU + real-field core only in this release (the engine
    # rejects fp16 with bloch boundaries or multi-GPU). Omitted from the wire
    # when unset (see _wire_exclude) for earlier-minor parser back-compat.
    field_precision: FieldPrecisionName = "fp32"
    # Schema 1.20 — NUMERICS.md §12.6 field_dft ACCUMULATOR storage precision.
    # The running DFT read-modify-writes one accumulator element per (monitored
    # cell, frequency, component) per accumulation step, which is the dominant
    # cost of a volume monitor and the only monitor term that scales with the
    # frequency count. "fp64" (default) is byte-identical to prior releases.
    # "fp32c" keeps a compensated (Kahan-Babuska-Neumaier) fp32 pair: the same
    # 16 B per complex element, so it moves no fewer bytes, but its error stays
    # ~1e-7 relative regardless of step count. "fp32" is a plain fp32 pair —
    # 8 B per element, halving both accumulator traffic and footprint, at a
    # signal-dependent accuracy cost that is worst on a ringdown (~1e-4 over
    # 200k steps; see NUMERICS.md §12.6 for the measured table).
    # Flux monitors always accumulate in fp64, in every mode.
    # Omitted from the wire when unset (see _wire_exclude) for earlier-minor
    # parser back-compat.
    dft_precision: DftPrecisionName = "fp64"
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
    # Schema 1.18 — per-axis Bloch wavevector (rad/um), used only on axes whose
    # boundary kind is "bloch": F(x+L) = F(x) e^{i k L}. None (default) is
    # omitted from the wire (byte-back-compat). Pair with
    # boundaries.<axis> = "bloch"; the engine rejects a nonzero component on a
    # non-bloch axis (silent-ignore trap). CPU solver only in this release.
    bloch_k_per_um: Optional[Tuple[float, float, float]] = None
    # May be EMPTY at the model level: a source-less "shell" Simulation is a
    # legitimate authoring artifact (mode_launch/mode_monitor take one for its
    # grid and structures; previously even the gallery resorted to placeholder
    # dipoles). Running one is still an error — phsolver validate/run rejects
    # an empty array with "sources: at least one source is required".
    sources: Tuple[SourceType, ...] = ()
    monitors: Tuple[MonitorType, ...] = ()
    # --- declarative setup fields (client-only; never on the wire, never in the
    # schema; design spec 2026-09-10 §4.1). Resolved at construction by
    # _resolve_declarative into ordinary sources and monitors, and kept in the
    # private ``_declarative`` so a result can read a port back and a with_*
    # copy can resolve again on its new grid.
    #   wlens_um  readout wavelengths (microns): every port's frequencies and the
    #             pulse; wlen0_um the pulse centre (default the mean of extremes)
    #   ports     Port values: one readout plane each, solved on this grid
    #   source    the driven port (name or Port) or a GaussianBeam
    wlens_um: SkipJsonSchema[Optional[Union[float, Tuple[float, ...]]]] = None
    wlen0_um: SkipJsonSchema[Optional[float]] = None
    ports: SkipJsonSchema[Tuple[Any, ...]] = ()
    source: SkipJsonSchema[Optional[Any]] = None
    _declarative: Optional[_decl.Resolved] = PrivateAttr(default=None)
    # The symmetry fold a fitted simulation applied (design spec §4.5): which
    # axes were folded, the ports read through their images, the monitors
    # returned unfolded. Client state like origin_um; None on a hand-built or
    # ingested scene.
    _fold: Optional[_decl.Fold] = PrivateAttr(default=None)
    # The transit cap a run length was resolved from (None when run_time_s or
    # n_steps was given): run_local warns when the run hits it undecayed.
    _run_transits: Optional[float] = PrivateAttr(default=None)
    # The fields as the user gave them (the Domain, the Mesh, the user-frame
    # structures and ports, the run as written) for a fitted or declarative
    # simulation: with_changes rebuilds from them. None on a hand-built scene.
    _inputs: Optional[dict] = PrivateAttr(default=None)
    # The fields a construction-time resolver filled in and marked set so they
    # ride the wire (the §16 subpixel default, the dispersive-PML CFS profile).
    # They are not the user's: an edit drops them so the resolver decides again
    # for the edited scene, while a value the user gave is kept verbatim.
    _auto_fields: frozenset = PrivateAttr(default=frozenset())
    # True for a document parsed from the wire (from_wire_json, from_file): its
    # edits keep the ingestion contract (no construction-time resolution).
    _wire_ingested: bool = PrivateAttr(default=False)
    # The caller's own monitors as written (after the domain fit's translation,
    # before plane spans resolve and faces quarter-snap): an edit resolves them
    # again for the edited scene, as construction would.
    _user_monitors: Optional[tuple] = PrivateAttr(default=None)
    # Stored-frame fields a with_* helper replaced on a domain= fit: the
    # inputs no longer describe the simulation, so with_changes refuses.
    _frame_edits: tuple = PrivateAttr(default=())

    def _resolve_materials(self) -> None:
        """Replace every materials-library entry among the structures and the
        background by the constant-index medium at the band centre
        (``Material.medium(wlen_um=wlen0)``: absorption to conductivity, as
        that method does), warning when the index moves by more than 0.5 %
        across the band. The dispersive fit stays explicit
        (``.medium(band_um=...)``). Phase-5 plan, refinement 4."""
        entries = [st.medium for st in self.structures if is_material(st.medium)]
        if is_material(self.background):
            entries.append(self.background)
        # the ports name their guide's medium too (the library passes its own through)
        port_values = [pp for pp in self.ports if isinstance(pp, Port)]
        if isinstance(self.source, Port):
            port_values.append(self.source)
        entries.extend(pp.medium for pp in port_values if is_material(pp.medium))
        if not entries:
            return
        names = sorted({str(getattr(m, "name", m)) for m in entries})
        if self.wlens_um is None and self.wlen0_um is None:
            raise ValueError(
                f"the materials-library entries {names} need a wavelength to resolve at: pass wlens_um "
                "(or wlen0_um), or pick the fit yourself with .medium(wlen_um=...) or .medium(band_um=(lo, hi))")
        if self.wlens_um is not None:
            wlens = (float(self.wlens_um),) if isinstance(self.wlens_um, (int, float)) else tuple(float(w) for w in self.wlens_um)
            wlen0 = float(self.wlen0_um) if self.wlen0_um is not None else 0.5 * (min(wlens) + max(wlens))
        else:
            wlen0 = float(self.wlen0_um)
            wlens = (wlen0,)
        lo, hi = min(wlens), max(wlens)
        resolved = {}
        for mat in entries:
            key = id(mat)
            if key in resolved:
                continue
            n0 = float(mat.n(wlen0))
            if hi > lo:
                drift = max(abs(float(mat.n(lo)) - n0), abs(float(mat.n(hi)) - n0)) / n0
                if drift > 0.005:
                    warnings.warn(
                        f"{mat.name}: the index moves {drift:.1%} across {lo:.4g} to {hi:.4g} um; the constant "
                        f"index at {wlen0:.4g} um is used. For dispersion pass "
                        f"ph.materials.{mat.name}.medium(band_um=({lo:.4g}, {hi:.4g})) explicitly.",
                        stacklevel=caller_stacklevel())
            resolved[key] = mat
        structures = tuple(
            st.model_copy(update={"medium": st.medium.medium(wlen_um=wlen0)}) if is_material(st.medium) else st
            for st in self.structures)
        object.__setattr__(self, "structures", structures)
        if is_material(self.background):
            mat = self.background
            if float(mat.k(wlen0)) > 0.0:
                raise ValueError(f"{mat.name} absorbs at {wlen0:.4g} um; a Background carries no conductivity, "
                                 "give ph.Background(permittivity=...)")
            object.__setattr__(self, "background", Background(permittivity=float(mat.n(wlen0)) ** 2))
        import dataclasses as _dc
        ports = tuple(_dc.replace(pp, medium=pp.medium.medium(wlen_um=wlen0))
                      if isinstance(pp, Port) and is_material(pp.medium) else pp for pp in self.ports)
        object.__setattr__(self, "ports", ports)
        if isinstance(self.source, Port) and is_material(self.source.medium):
            object.__setattr__(self, "source", _dc.replace(self.source, medium=self.source.medium.medium(wlen_um=wlen0)))

    def with_changes(self, **fields) -> "Simulation":
        """A copy with ``fields`` replaced and everything re-validated: the
        supported edit of a simulation. A fitted or
        declarative simulation is rebuilt from the fields as the user gave
        them, in the user's frame, so new structures are fitted, folded and
        their ports re-solved like the originals; a hand-built one is the
        validated copy the ``with_*`` helpers use, built as the constructor
        builds it, so the construction-time resolution (the subpixel default,
        plane spans, the §12 quarter-snap, the dispersive-PML profile and their
        warnings) is decided again for the edited scene from the fields as
        written. Either way the result is the simulation the constructor gives
        for the same fields. A simulation loaded with :meth:`from_wire_json`
        is edited as the document it is: nothing is resolved again, except
        that the monitors the edit adds (all of them when it changes the grid
        or the size) are placed on the cells as construction places them, and
        the declarations only the constructor resolves (a ``Domain``, a
        ``Mesh``, a run in transits, ports) are refused. ``domain=`` and ``mesh=`` are
        accepted as the constructor accepts them, for ``size_um`` and
        ``grid``."""
        fields = {_FIELD_ALIASES.get(name, name): value for name, value in fields.items()}
        if self._inputs is not None:
            if self._frame_edits:
                hint = " (for a mesh, mesh= rather than with_auto_mesh)" if "grid" in self._frame_edits else ""
                raise ValueError(
                    f"with_changes cannot rebuild this simulation from the fields as you wrote them: "
                    f"{list(self._frame_edits)} were replaced as stored (by model_copy or a with_* helper), in "
                    "coordinates or content its domain= fit or its ports resolved. Build it again with "
                    f"ph.Simulation(...), declaring that change there{hint}.")
            with _editing(self):
                return type(self)(**{**self._inputs, **fields})
        return self._validated_copy(dict(fields))

    def model_copy(self, *, update=None, deep: bool = False) -> "Simulation":
        """A copy with ``update`` applied. Prefer :meth:`with_changes`, the
        validated edit; this keeps pydantic's signature and what it means for
        each kind of simulation:

        - built by hand with a ``run``, or loaded from the wire: pydantic's
          field copy, which re-validates nothing;
        - one the constructor resolved (``domain=``, ``mesh=``, ports, or a run
          given in transits or not at all): :meth:`with_changes`, so the copy
          is the simulation the constructor builds from the same fields;
        - except on a ``domain=`` fit, where the stored structures, sources,
          monitors, size and grid are coordinates of the fitted box, whose
          origin is its low corner and not the user's: an update of one of
          them is ambiguous and raises ``ValueError`` naming
          :meth:`with_changes`, which takes them as you wrote them. An update
          that mixes one of them with other fields raises too. Renaming the
          structures (the same geometry and media) is allowed, and a later
          rebuild keeps the names. After a
          ``with_*`` helper replaced stored coordinates of such a fit (say
          ``with_auto_mesh``), the other fields are copied as that helper
          copied them, in the stored coordinates.

        On a resolved simulation the ``sources`` and ``monitors`` (as stored)
        and the ``subpixel``, ``subpixel_method``, ``field_precision`` and
        ``dft_precision`` switches are still a field copy: the path the SDK's
        own drivers (the S-matrix plan, the adjoint gradient) take to swap a
        resolved scene's launch and readout. Where the stored coordinates are
        yours (no ``domain=`` fit, no ports) a later :meth:`with_changes` or
        ``with_*`` helper keeps such a swap; on a ``domain=`` fit or a scene
        with ports, whose stored sources and monitors are not the ones you
        wrote, a later :meth:`with_changes` refuses instead of dropping it,
        while a ``with_*`` helper or a ``model_copy`` of other fields keeps the
        swap as stored. On a scene with ports or ``source=`` that edit refuses
        when it changes what the port solves read (the size, grid, structures,
        background, boundaries, symmetry, Bloch vector or layer counts): the
        swapped launch and readout were made for the old ones."""
        if update:
            update = _tupled(dict(update))
            inputs = self._inputs
            keys = set(update)
            if inputs is not None and not keys <= _RAW_COPY_FIELDS and not self._renames(update):
                tied = sorted(keys & _FRAME_FIELDS)
                if tied and isinstance(inputs.get("size_um"), Domain):
                    raise ValueError(
                        f"model_copy(update=...) cannot change {tied} on a simulation fitted with domain=: its "
                        "stored coordinates start at the fitted box's corner, not at your origin, and the copy "
                        "would skip the fit and the port solves. Use "
                        f"sim.with_changes({tied[0]}=...), which rebuilds the simulation from the fields as "
                        "you wrote them, in your coordinates.")
                new = self._validated_copy(update) if self._frame_edits else self.with_changes(**update)
                return super(Simulation, new).model_copy(deep=True) if deep else new
        new = super().model_copy(update=update, deep=deep)
        if update:
            new._auto_fields = self._auto_fields - set(update)    # a value given here is the caller's
            if "monitors" in update:
                new._user_monitors = update["monitors"]
            if self._inputs is not None:
                # a later rebuild (with_changes, a with_* helper) starts from the
                # inputs: the switches are frame-free; swapped sources and
                # monitors are the user's own where the stored frame is theirs,
                # and otherwise make the inputs stale (a rename changes nothing
                # a rebuild reads)
                stored = {k: v for k, v in update.items() if k in _FRAME_FIELDS}
                merged = {**self._inputs, **{k: v for k, v in update.items() if k not in _FRAME_FIELDS}}
                renamed = self._renamed_inputs(update) if self._renames(update) else None
                if stored and not self._generated() and not isinstance(self._inputs.get("size_um"), Domain):
                    merged.update(stored)
                elif renamed is not None:
                    merged["structures"] = renamed
                elif stored:
                    new._frame_edits = tuple(sorted(set(self._frame_edits) | set(stored)))
                new._inputs = merged
        return new

    def _generated(self) -> bool:
        """True when the stored sources and monitors hold what the declarative
        resolution made (the port readout planes, a port or beam launch), so
        they are not the ones the caller wrote. A ``wlens_um=`` alone, with no
        ports and no ``source=``, generates nothing."""
        rec = self._declarative
        return rec is not None and (bool(rec.ports) or self.source is not None)

    def _renamed_inputs(self, update: dict):
        """The recorded input structures with the names a renaming ``update``
        gives the stored ones, or None when they cannot be paired. The stored
        structures are the inputs in order, moved into the corner frame, less
        those a fold dropped (the mirrored half), then the port guide
        extensions: each input pairs with the next stored structure when its
        geometry, moved, is that one's. A library material is resolved in the
        stored one, so the geometry, not the medium, pairs them."""
        given = self._inputs.get("structures")
        if given is None:
            return None
        delta = tuple(-float(o) for o in self.origin_um)
        stored = list(zip(self.structures, update["structures"]))
        out, j = [], 0
        for a in given:
            if not isinstance(a, Structure):
                return None
            if j < len(stored) and _frame.translate(a.geometry, delta) == stored[j][0].geometry:
                out.append(a.model_copy(update={"name": stored[j][1].name}))
                j += 1
            elif self._fold is not None:
                out.append(a)                  # the fold dropped it: a rebuild drops it again
            else:
                return None
        return tuple(out)

    def _renames(self, update: dict) -> bool:
        """True when ``update`` only renames the structures: the same geometry
        and media in the same order, which no fit or port solve reads."""
        new = update.get("structures")
        if set(update) != {"structures"} or not isinstance(new, tuple) or len(new) != len(self.structures):
            return False
        return all(isinstance(a, Structure) and a.geometry == b.geometry and a.medium == b.medium
                   for a, b in zip(new, self.structures))

    def _fit_domain_mesh_run(self) -> None:
        """Resolve ``domain=`` into ``size_um`` and ``origin_um`` (translating
        every positional field into the corner frame), ``mesh=`` into ``grid``
        and ``run.transits`` into ``run_time_s`` (design spec §4.2 steps 3 to 6,
        §4.3). Called from ``model_post_init``, which pydantic runs after the
        field validation and BEFORE the after-validators, so every one of them
        sees an ordinary corner-frame simulation. Nothing to do on a hand-built
        scene."""
        domain = self.size_um if isinstance(self.size_um, Domain) else None
        mesh = self.grid if isinstance(self.grid, Mesh) else None
        transits = self.run.transits
        if transits is None and self.run.run_time_s is None and self.run.n_steps is None:
            transits = DEFAULT_TRANSITS            # a run given for its shutoff alone: the cap
        if domain is None and mesh is None and transits is None:
            return
        fold = None
        bbox = None
        if domain is not None and any(self.symmetry):
            whole = list(self.structures)
            if whole:
                bounds = [geometry_bounds_um(st.geometry) for st in whole]
                bbox = tuple((min(b[a][0] for b in bounds), max(b[a][1] for b in bounds)) for a in range(3))
            fold = self._fold_symmetry(whole)
        wlen0 = None
        if self.wlens_um is not None:
            wlen0 = _decl.band(self.wlens_um, self.wlen0_um)[1]
        elif self.wlen0_um is not None:
            wlen0 = float(self.wlen0_um)
        n_bg = math.sqrt(float(self.background.permittivity))
        structures = list(self.structures)
        continuations = ()      # (port guide extension, its port's axis), set by the domain fit
        n_max = max([_structure_index(st, wlen0 or 1.55) for st in structures] + [n_bg])
        # the boundary-layer cell, in closed form before the mesh exists: the
        # background cell of a graded mesh, the one cell of a uniform mesh
        if mesh is not None:
            if mesh.dl_um is not None and not isinstance(mesh.dl_um, tuple):
                dl_bg = float(mesh.dl_um)
            elif mesh.cells_per_wlen is not None:
                if wlen0 is None:
                    raise ValueError("mesh=Mesh(cells_per_wlen=...) needs wlens_um (or wlen0_um)")
                dl_bg = wlen0 / ((n_max if mesh.uniform else n_bg) * float(mesh.cells_per_wlen))
            else:                                     # every axis has its own spacing
                dl_bg = min(float(v) for v in mesh.dl_um if v is not None)
        else:
            dl_bg = float(self.grid.dl_um)
        # per axis: an axis with its own spacing (a lattice-commensurate mesh) counts in it
        dl_ax = tuple(mesh.spacing(a) if mesh is not None and mesh.spacing(a) is not None else dl_bg for a in range(3))
        if domain is not None:
            # the port's own cell, before the mesh exists: the mesh's cell in the
            # port's medium (the highest index when it names none) for a mesh by
            # cells per wavelength, else the one spacing given
            if mesh is not None and mesh.dl_um is not None and not isinstance(mesh.dl_um, tuple):
                cpw = None
            else:
                cpw = float(mesh.cells_per_wlen) if mesh is not None and mesh.cells_per_wlen is not None else None

            def port_cell(pp) -> float:
                if cpw is None:
                    return min(dl_ax)
                if mesh.uniform:
                    return dl_bg
                n_port = n_max if pp.medium is None else max(_structure_index(
                    Structure(geometry=Box(center_um=(0.0, 0.0, 0.0), size_um=(1.0, 1.0, 1.0)), medium=pp.medium),
                    wlen0 or 1.55), n_bg)
                return wlen0 / (n_port * cpw)

            # the longest wavelength a port reads sizes its window
            wlen_max = (c0 / min(_decl.band(self.wlens_um, self.wlen0_um)[0]) * 1e6
                        if self.wlens_um is not None else wlen0)
            size, origin, extensions = self._fit_box(domain, dl_ax, wlen0, n_bg, fold=fold, port_cell=port_cell,
                                                     bbox=bbox, wlen_max=wlen_max)
            delta = tuple(-o for o in origin)
            object.__setattr__(self, "size_um", size)
            object.__setattr__(self, "origin_um", tuple(float(o) for o in origin))
            object.__setattr__(self, "structures",
                               tuple(_frame.translate(structures, delta)) + tuple(e for e, _ in extensions))
            continuations = tuple((e, along) for e, along in extensions if along is not None)
            object.__setattr__(self, "sources", tuple(_frame.translate(list(self.sources), delta)))
            object.__setattr__(self, "monitors", tuple(_frame.translate(list(self.monitors), delta)))
            object.__setattr__(self, "ports", tuple(_frame.translate(list(self.ports), delta)))
            if isinstance(self.source, (Port, GaussianBeam)):
                object.__setattr__(self, "source", _frame.translate(self.source, delta))
        if mesh is not None:
            periodic = "".join(a for a in _AXES if getattr(self.boundaries, a) in ("periodic", "bloch"))
            # the refine regions are drawn in the user's frame; the mesh is built in the wire's
            shift = tuple(-float(o) for o in self.origin_um)
            refine = tuple(o.model_copy(update={"geometry": _frame.translate(o.geometry, shift)}) for o in mesh.refine)
            # the port guide extensions refine the mesh but are no interface along
            # their port's axis (auto_mesh ``continuations``, NUMERICS §18.7)
            drawn = {id(e) for e, _ in continuations}
            meshed = tuple(st for st in self.structures if id(st) not in drawn)
            if isinstance(mesh.dl_um, tuple):
                # a lattice-commensurate mesh: each axis with a spacing is a uniform
                # ladder at it, the others are graded by cells_per_wlen
                graded_axes = "".join(_AXES[a] for a in range(3) if mesh.spacing(a) is None)
                coords = {}
                if graded_axes:
                    auto = auto_mesh(size_um=self.size_um, wlen_um=wlen0, structures=meshed,
                                     background_index=n_bg, cells_per_wlen=float(mesh.cells_per_wlen),
                                     max_grading=mesh.max_grading, dl_min_um=mesh.dl_min_um,
                                     mesh_overrides=refine, periodic_axes=periodic, axes=graded_axes,
                                     continuations=continuations)
                    for letter in graded_axes:
                        coords[letter] = getattr(auto.coords, letter)
                given = [mesh.spacing(a) for a in range(3) if mesh.spacing(a) is not None]
                base = given[0]
                for a in range(3):
                    d = mesh.spacing(a)
                    if d is not None and d != base:
                        n = realized_cells(self.size_um[a], d, self._axis_min_cells()[a])
                        coords[_AXES[a]] = tuple(round(i * d, 9) for i in range(n))
                if coords:
                    grid = GradedMesh(dl_um=base, coords=GradedMeshAxis(**coords))
                else:
                    grid = UniformMesh(dl_um=base)
            elif mesh.dl_um is not None or mesh.uniform:
                dl = float(mesh.dl_um) if mesh.dl_um is not None else wlen0 / (n_max * float(mesh.cells_per_wlen))
                grid = UniformMesh(dl_um=dl)
            else:
                grid = auto_mesh(size_um=self.size_um, wlen_um=wlen0, structures=meshed,
                                 background_index=n_bg, cells_per_wlen=float(mesh.cells_per_wlen),
                                 max_grading=mesh.max_grading, dl_min_um=mesh.dl_min_um,
                                 mesh_overrides=refine, periodic_axes=periodic,
                                 continuations=continuations)
            object.__setattr__(self, "grid", grid)
        if transits is not None:
            transit_s = max(self.size_um) * 1e-6 * n_max / _C0_M_PER_S
            kept = {k: getattr(self.run, k) for k in self.run.model_fields_set
                    if k not in ("run_time_s", "n_steps", "transits")}
            object.__setattr__(self, "run", RunSpec(run_time_s=transits * transit_s, **kept))
            self._run_transits = float(transits)
        self._fold = fold
        if fold is not None:
            self._refuse_flux_windows_in_the_mirror_cell(fold)

    def _refuse_flux_windows_in_the_mirror_cell(self, fold) -> None:
        """Refuse a PowerMonitor window, on the kept side of an even (PMC)
        symmetry plane, whose low edge lies within the centre of the first cell
        from the plane (NUMERICS §20.8). The engine then holds the node row on
        the plane (``flux_window_range``) and counts it at half weight (§12),
        where the unfolded run counts it whole, so the window has no
        whole-device reading (it read 10 % low at 100 nm cells). On an odd
        (PEC) plane that row carries a pinned tangential E and nothing is
        lost. Runs once the mesh exists; the plane is the wire's coordinate 0."""
        for m in self.monitors:
            if not isinstance(m, PowerMonitor) or m.center_um is None:
                continue
            for i, a in enumerate(_frame._cyclic(m.axis)):
                if a not in fold.planes or self.symmetry[a] != 1:
                    continue
                lo = float(m.center_um[i]) - 0.5 * float(m.size_um[i])
                q = self._axis_coords_um(a)
                dq = float(q[1]) - float(q[0]) if q is not None and len(q) > 1 else float(self.grid.dl_um)
                if 1e-6 < lo <= 0.5 * dq + 1e-9 * dq:
                    raise ValueError(
                        f"monitor {m.name!r}: its window starts {lo:.6g} um from the even (PMC) symmetry "
                        f"plane {_AXES[a]} = {fold.planes[a]:.6g} um, inside the first cell, whose centre is "
                        f"{0.5 * dq:.6g} um from the plane. The window then holds the row on the plane, which "
                        "the simulation counts at half weight and the run without the plane counts whole, so "
                        "it has no whole-device reading. Move the edge more than "
                        f"{0.5 * dq:.6g} um from the plane, or make the window symmetric about it")

    def _fold_symmetry(self, structures) -> "_decl.Fold":
        """Fold a device described whole onto the half domain the wire and the
        engine know (design spec §4.5, phase-4 plan). For every axis with a
        nonzero ``symmetry`` entry the mirror plane is the structures'
        bounding-box centre; the structure set must be mirror-symmetric about
        it. Ports in the mirrored half are dropped and read through their
        images, monitors crossing the plane are clipped to the kept half, and
        anything placed entirely in the mirrored half is an error: the kept
        half is the far side of the plane. Sets ``ports`` and ``monitors``;
        returns the record the results read."""
        tol = 1e-9
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for st in structures:
            b = geometry_bounds_um(st.geometry)
            for a in range(3):
                lo[a], hi[a] = min(lo[a], b[a][0]), max(hi[a], b[a][1])
        planes = {a: 0.5 * (lo[a] + hi[a]) for a in range(3) if self.symmetry[a] != 0}
        for a, plane in planes.items():
            odd = _frame.mirror_image_missing(structures, _AXES[a], plane)
            if odd is not None:
                label = f"structure {structures.index(odd)}" + (f" ({odd.name!r})" if odd.name else "")
                raise ValueError(
                    f"symmetry[{a}] ('{_AXES[a]}'): {label} has no mirror image about {_AXES[a]} = "
                    f"{plane:.6g} um, the device's centre; a symmetry plane needs a mirror-symmetric "
                    "device, or build the half domain by hand with size_um")

        def near(value, a) -> bool:
            return float(value) < planes[a] - tol

        hand = "; build the half domain by hand with size_um to place it there"
        ports = [pp if isinstance(pp, Port) else Port(**pp) for pp in self.ports]
        driven = self.source.name if isinstance(self.source, Port) else (
            self.source if isinstance(self.source, str) else None)
        kept, dropped = [], []
        for pp in ports:
            pa = _AXES.index(pp.axis)
            if pa in planes and abs(float(pp.center_um[pa]) - planes[pa]) <= tol:
                raise ValueError(f"port {pp.name!r}: its plane lies on the symmetry plane {pp.axis} = "
                                 f"{planes[pa]:.6g} um; a port cannot be read on the mirror itself")
            near_axes = [a for a in planes if near(pp.center_um[a], a)]
            (dropped if near_axes else kept).append((pp, near_axes))
        mirrored = {}
        for pp, near_axes in dropped:
            image = pp
            for a in near_axes:
                image = _frame.mirror(image, _AXES[a], planes[a])
            match = next((k for k, _ in kept
                          if k.axis == pp.axis and abs(float(k.width_um) - float(pp.width_um)) <= tol
                          and all(abs(float(x) - float(y)) <= tol for x, y in zip(k.center_um, image.center_um))),
                         None)
            if match is None:
                raise ValueError(f"port {pp.name!r} lies in the mirrored half of the fold and no port sits at "
                                 f"its image {tuple(round(c, 6) for c in image.center_um)}{hand}")
            if pp.name == driven:
                raise ValueError(f"source={pp.name!r} lies in the mirrored half of the fold; drive its image "
                                 f"{match.name!r} or drop the symmetry plane")
            mirrored[pp.name] = match.name
        if isinstance(self.source, GaussianBeam):
            beam = self.source
            ba = _AXES.index(beam.axis)
            if (ba in planes and near(beam.position_um, ba)) or (
                    beam.center_um is not None and any(near(beam.center_um[a], a) for a in planes)):
                raise ValueError(f"the beam source lies in the mirrored half of the fold{hand}")

        monitors, unfolded = [], []
        for m in self.monitors:
            if isinstance(m, ProfileMonitor):
                c, s = list(map(float, m.center_um)), list(map(float, m.size_um))
                normal = _AXES.index(m.span.split(":", 1)[1]) if m.span is not None else None
                for a, plane in planes.items():
                    if normal is not None and a != normal:
                        unfolded.append(m.name)        # the interior span begins on the plane
                        continue
                    mlo, mhi = c[a] - s[a] / 2.0, c[a] + s[a] / 2.0
                    if s[a] == 0.0:
                        if near(c[a], a):
                            raise ValueError(f"monitor {m.name!r} lies in the mirrored half of the fold{hand}")
                        continue
                    if mhi <= plane + tol:
                        raise ValueError(f"monitor {m.name!r} lies in the mirrored half of the fold{hand}")
                    if mlo < plane - tol:
                        c[a], s[a] = 0.5 * (plane + mhi), mhi - plane
                        unfolded.append(m.name)
                m = m.model_copy(update={"center_um": tuple(c), "size_um": tuple(s)})
            elif isinstance(m, PowerMonitor):
                ma = _AXES.index(m.axis)
                for a, plane in planes.items():
                    if a == ma:
                        if near(m.position_um, a):
                            raise ValueError(f"monitor {m.name!r} lies in the mirrored half of the fold{hand}")
                    elif m.center_um is not None:
                        i = _frame._cyclic(m.axis).index(a)
                        c, s = list(map(float, m.center_um)), list(map(float, m.size_um))
                        mlo, mhi = c[i] - s[i] / 2.0, c[i] + s[i] / 2.0
                        # The result reports a window the fold clips as the whole
                        # device's power through it, twice the kept half (NUMERICS
                        # §20.8). That is the unfolded reading only for a window
                        # symmetric about the plane; one wholly on the kept side
                        # reads its own region. A window with an edge on the plane,
                        # or crossing it asymmetrically, has no such reading.
                        edge = 1e-6
                        if abs(mlo - plane) <= edge or abs(mhi - plane) <= edge:
                            raise ValueError(
                                f"monitor {m.name!r}: its window has an edge on the symmetry plane "
                                f"{_AXES[a]} = {plane:.6g} um, so it covers one side of a mirror-symmetric "
                                "field and the result, which reports the whole device's power, has no "
                                "reading for it; make the window symmetric about the plane (it then reads "
                                f"both sides), move it off the plane, or drop the symmetry plane{hand}")
                        if mhi < plane:
                            raise ValueError(f"monitor {m.name!r} lies in the mirrored half of the fold{hand}")
                        if mlo < plane:
                            if abs((plane - mlo) - (mhi - plane)) > edge:
                                raise ValueError(
                                    f"monitor {m.name!r}: its window crosses the symmetry plane "
                                    f"{_AXES[a]} = {plane:.6g} um asymmetrically ({mlo:.6g} to {mhi:.6g} um), "
                                    "and the result, which reports the whole device's power, can only read a "
                                    "window symmetric about the plane; center it on the plane, keep it on one "
                                    f"side, or drop the symmetry plane{hand}")
                            c[i], s[i] = 0.5 * (plane + mhi), mhi - plane
                            m = m.model_copy(update={"center_um": tuple(c), "size_um": tuple(s)})
            elif isinstance(m, TimeMonitor):
                if any(near(m.center_um[a], a) for a in planes):
                    raise ValueError(f"monitor {m.name!r} lies in the mirrored half of the fold{hand}")
            monitors.append(m)
        for src in self.sources:
            if isinstance(src, PointDipole) and any(near(src.center_um[a], a) for a in planes):
                raise ValueError(f"a point dipole at {src.center_um} lies in the mirrored half of the fold{hand}")
            if isinstance(src, TfsfBox) and any(near(src.center_um[a] - src.size_um[a] / 2.0, a) for a in planes):
                raise ValueError(f"the TFSF box crosses the symmetry plane{hand}")
            if isinstance(src, ModeSource) and _AXES.index(src.axis) in planes and near(src.position_um, _AXES.index(src.axis)):
                raise ValueError(f"the mode source at {src.axis} = {src.position_um} lies in the mirrored half of the fold{hand}")
        # a structure entirely in the mirrored half is its kept image's mirror:
        # the engine never sees that half, and the wire stays the half domain's
        kept_structures = []
        for st in structures:
            b = geometry_bounds_um(st.geometry)
            if all(b[a][1] > planes[a] - tol for a in planes):
                kept_structures.append(st)
        object.__setattr__(self, "structures", tuple(kept_structures))
        object.__setattr__(self, "ports", tuple(k for k, _ in kept))
        object.__setattr__(self, "monitors", tuple(monitors))
        return _decl.Fold(planes=planes, mirrored_ports=mirrored, unfolded_monitors=tuple(dict.fromkeys(unfolded)))

    def _fit_box(self, domain: Domain, dl_ax, wlen0, n_bg: float, fold=None, port_cell=None, bbox=None,
                 wlen_max=None):
        """The fitted box in the user's frame: ``(size_um, origin_um,
        extensions)`` per design spec §4.3, where ``extensions`` pairs each port
        guide extension (§4.2 step 6, already in the corner frame) with the
        port's axis, the axis along which it continues its guide, or ``None``
        when it is not its host's medium (no continuation)."""
        structures = list(self.structures)
        if not structures:
            raise ValueError("domain= needs structures (or a device) to fit the box around")
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for st in structures:
            b = geometry_bounds_um(st.geometry)
            for a in range(3):
                lo[a], hi[a] = min(lo[a], b[a][0]), max(hi[a], b[a][1])
        if bbox is not None:                          # the whole device's bounds, before the fold dropped its mirrored half
            lo, hi = [b[0] for b in bbox], [b[1] for b in bbox]
        ports = [pp if isinstance(pp, Port) else Port(**pp) for pp in self.ports]
        driven = self.source.name if isinstance(self.source, Port) else (
            self.source if isinstance(self.source, str) else None)
        if isinstance(self.source, Port) and self.source.name not in {pp.name for pp in ports}:
            ports.append(self.source)
        if isinstance(dl_ax, (int, float)):
            dl_ax = (float(dl_ax),) * 3
        if port_cell is None:
            port_cell = lambda pp: min(dl_ax)  # noqa: E731

        def margin_of(pp) -> float:
            return float(domain.port_margin_um) if domain.port_margin_um is not None else 10.0 * port_cell(pp)
        # Room for each port's default mode window across its guide (NUMERICS
        # §18.8): the rule's window with 20 % more pad, which covers the
        # refinement from the port's own solve, and two cells to the layers
        # (one the window keeps from them, one the box's rounding to whole
        # cells may take). A clearance is a minimum distance to the
        # structures, so the room is added to it; an extent or walls the
        # caller gives are the caller's box, and a window they cut is clipped
        # with a warning.
        need_lo, need_hi = {}, {}
        painter = _decl._Painter(self)
        if wlen0 is not None:
            for pp in ports:
                if pp.window_um is not None or pp.thickness_um is None:
                    continue
                plan = _decl.plan_default_window(self, pp, wlen0, wlen_max or wlen0, n_bg, painter=painter)
                w, t = _decl._port_axes(pp)
                for i, half, core in ((w, plan.half_w_um, 0.5 * float(pp.width_um)),
                                      (t, plan.half_v_um, 0.5 * float(pp.thickness_um))):
                    reach = core + 1.2 * (half - core) + 2.0 * dl_ax[i]
                    c = float(pp.center_um[i])
                    need_lo[i] = min(need_lo.get(i, math.inf), c - reach)
                    need_hi[i] = max(need_hi.get(i, -math.inf), c + reach)
        # user-frame side of each port: relative to the device's own centre
        sides = {}
        for pp in ports:
            a = _AXES.index(pp.axis)
            if pp.out_direction is not None:
                sides[pp.name] = pp.out_direction
            else:
                mid = 0.5 * (lo[a] + hi[a])
                if abs(pp.plane_um - mid) <= 1e-9:
                    raise ValueError(f"port {pp.name!r}: its plane sits on the device centre along "
                                     f"{pp.axis}; pass out_direction='+' or '-'")
                sides[pp.name] = "+" if pp.plane_um > mid else "-"
        size, origin, outer = [0.0] * 3, [0.0] * 3, []
        for a, axis in enumerate(_AXES):
            dl_bg = dl_ax[a]
            kind = getattr(self.boundaries, axis)
            if kind in ("periodic", "bloch"):
                period = domain.period_um[a] if domain.period_um is not None else None
                extent = float(period) if period is not None else hi[a] - lo[a]
                n = max(1, int(round(extent / dl_bg)))
                size[a], origin[a] = n * dl_bg, 0.5 * (lo[a] + hi[a]) - 0.5 * n * dl_bg   # centred on the device
                outer.append((None, None))
                continue
            layers = self.pml_num_layers if kind == "pml" else (self.absorber_num_layers if kind == "absorber" else 0)
            pml = layers * dl_bg
            walls = {}
            for pp in ports:
                if pp.axis != axis:
                    continue
                side = sides[pp.name]
                plane = pp.plane_um
                if pp.name == driven:
                    off = (float(pp.source_offset_um) if pp.source_offset_um is not None
                           else _decl.default_source_offset_um(wlen0, n_bg))
                    plane = plane + (off if side == "+" else -off)
                plane = plane + (margin_of(pp) if side == "+" else -margin_of(pp))
                cur = walls.get(side)
                walls[side] = plane if cur is None else (max(cur, plane) if side == "+" else min(cur, plane))
            given_extent = domain.extent(a)
            if domain.walls(a) is not None:
                # the two walls are the design's, at the coordinates given
                walls = dict(zip("-+", domain.walls(a)))
            elif given_extent is not None:
                # the extent is the design's: centred on the structures, in place
                # of the bounding box, the clearance and the port margins
                mid = 0.5 * (lo[a] + hi[a])
                walls = {"-": mid - 0.5 * given_extent, "+": mid + 0.5 * given_extent}
            if wlen0 is None and (domain.clearance(a) is None) and ("+" not in walls or "-" not in walls):
                raise ValueError("domain= needs wlens_um (or wlen0_um) for the default clearance, or clearance_um")
            clearance = (domain.clearance(a) if domain.clearance(a) is not None
                         else (_decl.default_margin_um(wlen0, n_bg) if wlen0 is not None else 0.0))
            wall_hi = walls["+"] if "+" in walls else hi[a] + clearance
            own = domain.walls(a) is None and given_extent is None
            if own and a in need_hi:
                wall_hi = max(wall_hi, need_hi[a])
            if fold is not None and a in fold.planes:
                # the mirror is the low face: no margin, no boundary layers there,
                # and the rounding excess goes to the far side (§20)
                wall_lo = fold.planes[a]
                extent = (wall_hi + pml) - wall_lo
                n = int(round(extent / dl_bg))
                if n < 4 or not (wall_hi > wall_lo):
                    raise ValueError(f"axis {axis!r}: the folded domain has no interior ({extent:.4g} um for "
                                     f"{layers} boundary layers on the far face)")
                size[a], origin[a] = n * dl_bg, wall_lo
                outer.append((wall_lo, wall_hi + pml))
                continue
            wall_lo = walls["-"] if "-" in walls else lo[a] - clearance
            if own and a in need_lo:
                wall_lo = min(wall_lo, need_lo[a])
            extent = (wall_hi + pml) - (wall_lo - pml)
            n = int(round(extent / dl_bg))
            if n < 4 or not (wall_hi > wall_lo):
                raise ValueError(f"axis {axis!r}: the fitted domain has no interior ({extent:.4g} um for {layers} "
                                 "boundary layers each face)")
            size[a] = n * dl_bg
            origin[a] = (wall_lo - pml) - 0.5 * (size[a] - extent)
            outer.append((wall_lo - pml, wall_hi + pml))
        # port guide extensions, to one cell past the REALIZED outer face (the
        # rounding may have widened the box by up to half a cell each side)
        extensions = []
        for pp in ports:
            a = _AXES.index(pp.axis)
            dl_bg = dl_ax[a]
            side = sides[pp.name]
            if outer[a][0] is None:
                continue
            face = origin[a] + size[a] if side == "+" else origin[a]
            # the structure the port sits on (design spec §5.3): its medium, and how
            # far it already reaches toward the wall. Paint order decides which one
            # that is: the LAST structure in list order containing the port centre
            # wins (NUMERICS.md §9), so the guide under a cladding authored before it
            # is the host, and a guide buried under a cladding authored after it is
            # not (the raster has no guide there to continue).
            containing = []
            for st in structures:
                b = geometry_bounds_um(st.geometry)
                if all(b[i][0] - 1e-9 <= float(pp.center_um[i]) <= b[i][1] + 1e-9 for i in range(3)):
                    containing.append(st)
            host = containing[-1] if containing else None
            if host is not None and pp.medium is not None:
                _warn_port_medium_mismatch(pp, containing, wlen0)
            if host is not None:
                hb = geometry_bounds_um(host.geometry)[a]
                reach = hb[1] if side == "+" else hb[0]
            else:
                reach = pp.plane_um
            if (face - reach if side == "+" else reach - face) <= dl_bg:
                continue                       # the guide already runs through the PML
            medium = pp.medium if pp.medium is not None else (host.medium if host is not None else None)
            if pp.thickness_um is None or medium is None:
                raise ValueError(f"port {pp.name!r}: extending its guide through the wall needs thickness_um "
                                 "on the Port and a medium (the Port's, or a structure the port sits on)")
            end = face + dl_bg if side == "+" else face - dl_bg
            # from one cell inside the guide (a plane on a polygon's end face would
            # otherwise read a bare cross-section) to one cell past the outer face
            start = pp.plane_um - dl_bg if side == "+" else pp.plane_um + dl_bg
            t = _AXES.index(pp.thickness_axis)
            w = [i for i in range(3) if i not in (a, t)][0]
            center = list(pp.center_um)
            center[a] = 0.5 * (start + end)
            ext = [0.0, 0.0, 0.0]
            ext[a], ext[w], ext[t] = abs(end - start), float(pp.width_um), float(pp.thickness_um)
            box = Box(center_um=tuple(float(c) - origin[i] for i, c in enumerate(center)), size_um=tuple(ext))
            # a continuation of its host only when it is the host's medium: otherwise
            # (no host, or a Port medium that differs) its inner face is a real
            # interface and stays a snap target
            along = pp.axis if host is not None and medium == host.medium else None
            extensions.append((Structure(geometry=box, medium=medium), along))
        return tuple(size), tuple(origin), extensions

    def model_post_init(self, __context) -> None:
        """Fit the domain, mesh and run length, then resolve ``ports`` /
        ``source`` / ``wlens_um``, once the fields have validated (pydantic
        runs this before the after-validators, which then check the resolved
        scene): the model at this point is the geometry-only simulation the mode solves need
        (final grid, structures, symmetry, the caller's sources and monitors).
        The launch and the readout planes are then set beside the caller's, and
        the merged wire-level content is validated once more so a plane that
        lands in the PML or a name clash fails here, at construction. Wire
        ingestion carries the ``wire_ingest`` context (as do the throwaway
        validations here and in ``_validated_copy``) and is left alone; a
        fitted simulation refuses ``model_copy`` edits that would need this
        resolution again (see :meth:`model_copy`)."""
        ctx = __context if isinstance(__context, dict) else {}
        if ctx.get("wire_ingest"):
            self._wire_ingested = True
            return
        fitted = (isinstance(self.size_um, Domain) or isinstance(self.grid, Mesh)
                  or self.run.transits is not None or self.run.run_time_s is None and self.run.n_steps is None)
        declared = bool(self.ports) or self.source is not None or self.wlens_um is not None
        if fitted or declared:
            self._inputs = {name: getattr(self, name) for name in self.model_fields_set}
        self._resolve_materials()
        self._fit_domain_mesh_run()
        self._user_monitors = tuple(self.monitors)
        if declared:
            self._apply_declarative(user_sources=tuple(self.sources),
                                    user_monitors=tuple(self.monitors))
        if fitted or declared:
            type(self).model_validate(
                self.model_dump(mode="python", exclude=set(_decl.DECLARATIVE_FIELDS) | {"origin_um"}),
                context={"wire_ingest": True})

    def _apply_declarative(self, *, user_sources, user_monitors) -> None:
        """Solve the ports on ``self`` and set the resolved sources and monitors
        (the readout planes quarter-snapped like any hand-built plane)."""
        sources, monitors, resolved = _decl.resolve(
            self, ports=tuple(self.ports), source=self.source, wlens_um=self.wlens_um,
            wlen0_um=self.wlen0_um, sources=user_sources, monitors=user_monitors)
        snapped, _notes = _quarter_snapped_dft_monitors(
            monitors, size_um=self.size_um, grid=self.grid, axis_min_cells=self._axis_min_cells())
        object.__setattr__(self, "sources", tuple(sources))
        object.__setattr__(self, "monitors", tuple(snapped))
        object.__setattr__(self, "ports", tuple(resolved.ports))
        self._declarative = resolved

    def __eq__(self, other) -> bool:
        """Two simulations are equal when their fields are: the private
        resolution record (solved modes) does not take part."""
        if not isinstance(other, Simulation):
            return NotImplemented
        return self.__dict__ == other.__dict__

    @property
    def port_monitors(self):
        """``{port name: ModeMonitor}`` for a declarative simulation, else ``{}``."""
        return dict(self._declarative.port_monitors) if self._declarative else {}

    @property
    def port_windows_um(self):
        """``{port name: (half_w_um, half_v_um)}``: the mode window each port's
        mode was solved on (a port's ``window_um``, or the default, clipped to
        the domain). Empty for a simulation without ports, and for one loaded
        from a file, whose modes are solved again on first use. A port the
        symmetry fold dropped reads through its mirror image and has no
        entry."""
        rec = self._declarative
        return dict(getattr(rec, "port_windows_um", None) or {}) if rec else {}

    @property
    def driven_port(self) -> Optional[str]:
        """The name of the port ``source=`` drives, if any."""
        return self._declarative.driven if self._declarative else None

    @model_validator(mode="after")
    def _bloch_pairing(self) -> "Simulation":
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        k = self.bloch_k_per_um or (0.0, 0.0, 0.0)
        for a, name in enumerate("xyz"):
            if k[a] != 0.0 and kinds[a] != "bloch":
                raise ValueError(
                    f"bloch_k_per_um[{name}] is nonzero but boundaries.{name} "
                    f"is {kinds[a]!r} — set it to 'bloch' (the value would be "
                    "silently ignored otherwise)")
        for s in self.sources:
            theta = getattr(s, "angle_theta_rad", None)
            if theta:
                ax = "xyz".index(s.axis)
                for a, name in enumerate("xyz"):
                    if a != ax and kinds[a] != "bloch":
                        raise ValueError(
                            f"plane wave with angle_theta_rad != 0 requires "
                            f"transverse boundaries.{name} = 'bloch' with the "
                            "matching bloch_k_per_um — use "
                            "Simulation.with_oblique_plane_wave(...)")
        return self

    @field_validator("background", mode="before")
    @classmethod
    def _background_from_medium(cls, v):
        # Beta papercut: materials.X.medium(...) hands back a Medium, and
        # passing it as the background is the natural move. A plain
        # dielectric Medium carries exactly the information Background holds,
        # so coerce it losslessly; anything with more physics must fail with
        # directions rather than pydantic's generic model_type error.
        fitted = getattr(v, "medium", None)
        if isinstance(fitted, Medium) and not isinstance(v, Medium):
            v = fitted   # PoleFit/LorentzFit: fall through to the Medium rules
        if is_material(v):
            return v   # resolved by the Simulation at its band centre (setup layer phase 5)
        if isinstance(v, Medium):
            if (v.conductivity_s_per_m == 0.0 and not v.lorentz
                    and not v.poles and not v.drude
                    and v.permittivity_xyz is None and not v.pec
                    and v.permittivity_data is None):
                return Background(permittivity=v.permittivity)
            raise ValueError(
                "background must be a homogeneous non-dispersive dielectric "
                "(ph.Background); this Medium carries conductivity, poles, "
                "anisotropy, PEC, or permittivity_data, which the background "
                "cannot express — model that region as a Structure instead")
        return v

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
        by_key: dict[str, list[str]] = {}
        for monitor in v:
            by_key.setdefault(_monitor_name_key(monitor.name), []).append(
                monitor.name
            )
        dupes = sorted(
            {name for names in by_key.values() if len(names) > 1 for name in names}
        )
        if dupes:
            raise ValueError(
                "monitor names must be unique ignoring ASCII case; "
                f"duplicates: {dupes}"
            )
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
    def _resolved_grid_resource_limits(self) -> "Simulation":
        # Cheap mirror of engine/include/phcore/grid.h: reject a typo-sized
        # grid before cost estimation or solver launch. This computes only
        # three integers (or tuple lengths); phsolver validate remains the
        # authoritative resolver for the complete device/grid contract.
        counts = resolved_cell_counts(
            self.size_um, self.grid, self._axis_min_cells()
        )
        # NUMERICS.md section 1 floor warning: an axis whose requested span
        # rounds to fewer cells than the engine will realize gets silently
        # widened server-side (4-cell floor on every non-plain-periodic
        # axis). Surface that here, at authoring time — the classic trap is a
        # quasi-2D request (size = 1*dl) on a non-periodic axis, which
        # quadruples the cell count. A plain periodic axis realizes n = 1
        # exactly, so it never warns.
        dl = self.grid.dl_um
        for axis_index, name in enumerate(_AXES):
            if self._axis_coords_um(axis_index) is not None:
                continue  # graded ladders carry their own >= 4-node contract
            requested = realized_cells(
                self.size_um[axis_index], dl, min_cells=1
            )
            if requested < counts[axis_index]:
                warnings.warn(
                    f"size_um[{axis_index}] ('{name}') = "
                    f"{self.size_um[axis_index]:g} um spans "
                    f"{requested} cell(s) at dl_um = {dl:g}, but the engine "
                    f"realizes {counts[axis_index]} cells: NUMERICS.md "
                    "section 1 floors a "
                    f"'{getattr(self.boundaries, name)}' axis at "
                    f"{counts[axis_index]} cells. Only a plain periodic axis "
                    "(no symmetry plane) may run 1 cell deep (quasi-2D); "
                    "widen the axis or set boundaries."
                    f"{name} = 'periodic' if a quasi-2D reduction was "
                    "intended.",
                    UserWarning,
                    stacklevel=caller_stacklevel(),
                )
        return self

    def _warn_degenerate_axis_subpixel(self) -> "Simulation":
        """Warn when subpixel smoothing will dilute a quasi-2-D structure.

        On a 1-cell plain-periodic axis (the quasi-2-D reduction) the physics
        is invariant along that axis, but the subpixel sampler still smooths
        ALONG it: the Yee voxel of a field component staggered on that axis is
        centred half a cell off the primary node, so a structure sized to the
        single cell fills only HALF of it and its in-plane eps is averaged with
        the background. TM/Ez physics is unaffected (that voxel is aligned);
        in-plane-E (TE) physics is corrupted, a photonic-crystal cavity mode
        can vanish entirely. Until the engine treats a degenerate axis as
        invariant, structures must extend PAST the domain along it.
        """
        if not self.subpixel or not self.structures:
            return self
        counts = resolved_cell_counts(
            self.size_um, self.grid, self._axis_min_cells()
        )
        dl = self.grid.dl_um
        for axis_index, name in enumerate(_AXES):
            if counts[axis_index] != 1:
                continue
            if self._axis_coords_um(axis_index) is not None:
                continue                      # graded axis: not the 1-cell case
            extent = counts[axis_index] * dl
            thin = []
            for structure in self.structures:
                span = _structure_axis_span(structure.geometry, axis_index)
                if span is None:
                    continue
                lo, hi = span
                # NUMERICS.md section 16.13: the engine wraps the smoothing
                # voxel on a degenerate axis, so a structure covering the WHOLE
                # single cell (lo <= 0, hi >= extent) is exact. Only a PARTIAL
                # cover still warns — almost surely an authoring slip in a
                # scene meant to be 2-D (it becomes a uniform in-plane average
                # with the background at the covered fraction).
                if lo > 1e-12 * dl or hi < extent * (1 - 1e-12):
                    thin.append(structure.name or type(structure.geometry).__name__)
            if thin:
                warnings.warn(
                    f"axis '{name}' realizes a single plain-periodic cell "
                    f"(quasi-2D) and {len(thin)} structure(s) cover only part "
                    f"of it ({', '.join(thin[:4])}"
                    f"{', ...' if len(thin) > 4 else ''}): the axis is "
                    "invariant, so a partial cover becomes a uniform average "
                    "with the background at the covered fraction "
                    "(NUMERICS.md section 16.13) — rarely what a 2-D scene "
                    f"intends. Span the full 0..{extent:g} um (or beyond) on "
                    f"'{name}' for solid 2-D geometry.",
                    UserWarning,
                    stacklevel=caller_stacklevel(),
                )
                return self
        return self

    @model_validator(mode="after")
    def _run_required_on_the_wire(self, info) -> "Simulation":
        """``run`` is optional in Python (the transit cap fills it) and required
        on the wire: a document ingested without it is rejected, as the schema
        says."""
        if (info.context or {}).get("wire_ingest") and (
                "run" not in self.model_fields_set
                or (self.run.run_time_s is None and self.run.n_steps is None)):
            raise ValueError("run is required on the wire (run_time_s or n_steps)")
        return self

    @model_validator(mode="after")
    def _resolve_plane_spans(self, info) -> "Simulation":
        """Fill the in-plane centre and extent of every
        :meth:`ProfileMonitor.plane` monitor marked ``span="interior:<axis>"`` with the
        PML-free interior of this domain. Client-side only: the marker never
        reaches the wire (it is None once resolved and excluded when None), and
        an ingested document carries no marker; a monitor added to one by an
        edit (context ``wire_edit``) is still resolved. Runs BEFORE the
        quarter-cell snap so the resolved faces are snapped like any hand-built
        plane."""
        ctx = info.context or {}
        if ctx.get("wire_ingest") and not ctx.get("wire_edit"):
            return self
        if not any(isinstance(m, ProfileMonitor) and (m.span is not None or m.station is not None)
                   for m in self.monitors):
            return self
        lo_hi = [self._open_interval_um(a) for a in range(3)]
        resolved = []
        for m in self.monitors:
            if not (isinstance(m, ProfileMonitor) and (m.span is not None or m.station is not None)):
                resolved.append(m)
                continue
            if m.station is not None:
                # a ProfileMonitor.sections plane: station i of n, spread evenly
                # along its normal across the interior
                i, n = m.station
                along = "xyz".index(m.span.split(":", 1)[1]) if m.span else \
                    min(range(3), key=lambda a: m.size_um[a])
                lo, hi = lo_hi[along]
                center = list(m.center_um)
                center[along] = lo + (i + 0.5) * (hi - lo) / n
                m = m.model_copy(update={"center_um": tuple(center), "station": None})
                if m.span is None:
                    resolved.append(m)
                    continue
            normal = "xyz".index(m.span.split(":", 1)[1])
            center = list(m.center_um)
            size = list(m.size_um)
            for a in range(3):
                if a == normal:
                    continue
                lo, hi = lo_hi[a]
                center[a] = 0.5 * (lo + hi)
                size[a] = hi - lo
            resolved.append(m.model_copy(update={
                "center_um": tuple(center), "size_um": tuple(size), "span": None}))
        object.__setattr__(self, "monitors", tuple(resolved))
        return self

    def _open_interval_um(self, axis_index: int) -> Tuple[float, float]:
        """``(lo, hi)`` of the absorbing-layer-free interior along an axis:
        the PML or absorber slab thickness (its layer count times the local
        cell spacings) is removed from each face that carries one; a periodic,
        Bloch or PEC face contributes nothing; a symmetry plane sits on the
        low face and is kept."""
        kind = getattr(self.boundaries, _AXES[axis_index])
        layers = {"pml": self.pml_num_layers,
                  "absorber": self.absorber_num_layers}.get(kind, 0)
        length = float(self.size_um[axis_index])
        q = self._axis_coords_um(axis_index)
        if q is None:
            thick_lo = thick_hi = layers * float(self.grid.dl_um)
        else:
            dq = graded_primary_spacings(q)
            thick_lo = float(sum(dq[:layers]))
            thick_hi = float(sum(dq[-layers:])) if layers else 0.0
        lo = 0.0 if self.symmetry[axis_index] != 0 else thick_lo
        hi = length - thick_hi
        if not hi > lo:
            raise ValueError(
                f"axis {_AXES[axis_index]!r}: the absorbing layers leave no interior "
                f"(size {length} um, {layers} layers each face)")
        return lo, hi

    @model_validator(mode="after")
    def _dft_regions_quarter_snap(self, info) -> "Simulation":
        # NUMERICS.md §12 ergonomics: auto-snap DFT field-monitor box faces
        # to quarter-cell planes wherever the engine's per-component region
        # snap would reject them or pass on float luck (see
        # _quarter_snapped_dft_monitors / grid.quarter_snap_dft_face). This
        # is a CONSTRUCTION-time convenience, client-side only — the wire
        # schema is untouched and an INGESTED document (from_wire_json /
        # from_file passes context ``wire_ingest``) is NEVER adjusted, so
        # existing sim.json files round-trip byte-identically and phsolver
        # remains authoritative for them. Runs BEFORE _modal_port_rules so
        # the port checks see the final geometry. Adjustments are reported
        # per monitor on this module's DEBUG log.
        # An edit of an ingested document (context ``wire_edit``, the names of
        # the monitors it adds) snaps only those: they were written in Python.
        ctx = info.context or {}
        added = ctx.get("wire_edit") if ctx.get("wire_ingest") else None
        if ctx.get("wire_ingest") and not added:
            return self
        if not any(isinstance(m, ProfileMonitor) for m in self.monitors):
            return self
        snapped, notes = _quarter_snapped_dft_monitors(
            self.monitors, size_um=self.size_um, grid=self.grid, axis_min_cells=self._axis_min_cells())
        if notes:
            if added is not None:
                snapped = tuple(s if m.name in added else m for m, s in zip(self.monitors, snapped))
            object.__setattr__(self, "monitors", snapped)
            for note in notes:
                _LOG.debug("%s", note)
        return self

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
            if isinstance(monitor, ProfileMonitor)
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
        that axis is uniform (UniformMesh, or a GradedMesh axis not
        listed in ``coords``)."""
        coords = getattr(self.grid, "coords", None)
        if coords is None:
            return None
        return getattr(coords, "xyz"[axis_index])

    def _axis_min_cells(self) -> Tuple[int, int, int]:
        """NUMERICS.md section 1 per-axis cell floor for THIS simulation:
        1 on plain periodic axes (no symmetry plane), 4 elsewhere."""
        return tuple(
            axis_min_cells(getattr(self.boundaries, name), self.symmetry[a])
            for a, name in enumerate(_AXES)
        )

    def _realized_um(self) -> Tuple[float, float, float]:
        dl = self.grid.dl_um
        mins = self._axis_min_cells()
        out = []
        for i, L in enumerate(self.size_um):
            q = self._axis_coords_um(i)
            if q is None:
                out.append(realized_cells(L, dl, mins[i]) * dl)
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
                f"no nonabsorbing interior on axis '{axis}': "
                f"{layers} {boundary} layers cover the whole axis"
            )
        return (
            float(coordinate(lower_index)),
            float(coordinate(upper_index)),
            boundary,
        )

    def _two_eps0_over_dt(self) -> float:
        """The CPML peak-conductivity unit ``2*eps0/dt`` [S/m] at this scene's
        timestep, the scale ``sigma_max`` / ``alpha_max`` are quoted in, and
        the bridge between its dimensionless convention and the engine's S/m
        ``pml_alpha_max``.

        Built from the engine's CFL timestep (NUMERICS.md §2; resolve.cpp and
        grid.h ``graded_courant_dt``): ``dt = courant / (c0*sqrt(sum_a 1/dl_a^2))``
        over the per-axis MINIMUM primary spacing of the ACTIVE axes (a 1-cell
        plain-periodic axis contributes no curl term and leaves the sum, the
        section 2 quasi-2D reduction), which on a uniform 3-D grid is
        ``courant*dl / (c0*sqrt(3))``. Since ``eps0*c0 = 1/eta0`` this reduces to
        ``(2/eta0)*sqrt(sum_a 1/dl_a^2)/courant``, needing only ``eta0``, the
        cell spacings, and the Courant number, and matching the engine's dt so a
        converted alpha lands exactly on that scale."""
        mins = self._axis_min_cells()
        inv_sq = 0.0
        for a in range(3):
            q = self._axis_coords_um(a)
            if q is None:
                if realized_cells(self.size_um[a], self.grid.dl_um,
                                  mins[a]) <= 1:
                    continue  # degenerate quasi-2D axis: no curl term
                dl_um = self.grid.dl_um
            else:
                dl_um = min(graded_primary_spacings(q))
            dl_m = dl_um * 1e-6
            inv_sq += 1.0 / (dl_m * dl_m)
        if inv_sq == 0.0:
            dl_m = self.grid.dl_um * 1e-6
            inv_sq = 1.0 / (dl_m * dl_m)
        return (2.0 / _ETA0) * math.sqrt(inv_sq) / self.run.courant

    def _dispersive_boundary_crossings(self) -> Tuple[bool, bool, bool]:
        """Per axis: does a dispersive (Lorentz) structure's bounding box reach
        into that axis' OUTER absorbing band (the PML/absorber layers)? These
        are the structures for which a stretched-coordinate PML can diverge ,
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
            if not s.medium.is_dispersive:
                continue
            bb = geometry_bounds_um(s.geometry)
            for a in range(3):
                lo, hi = bb[a]
                L = realized[a]
                # NUMERICS §20: a symmetry plane replaces the minimum face's
                # absorbing band (geometry past it is the mirror image), so
                # only the far face is stretched there.
                if (self.symmetry[a] == 0 and lo < band) or hi > L - band:
                    out[a] = True
        return tuple(out)

    def _validated_copy(self, update: dict) -> "Simulation":
        """A copy with ``update`` applied, built and validated as the
        constructor builds a simulation: the shared backend of
        :meth:`with_changes` on a hand-built scene and of every ``with_*``
        helper. Values are the stored ones (the corner frame of a fitted scene).

        pydantic's ``model_copy`` skips validation entirely, so a helper using
        it alone could hand back a Simulation that direct construction rejects
        (e.g. ``with_absorber`` replacing the periodic transverse boundaries
        required by a §13 plane-wave source), and it would keep every
        construction-time decision of the original: the §16 subpixel default
        chosen for the old structures, a ``ProfileMonitor.plane`` span marker
        unresolved (a key the engine refuses), the §12 quarter-snap and the
        dispersive-PML CFS profile of the old grid.

        So the copy is constructed from the fields the user set (those a
        resolver filled in are left out, see ``_auto_fields``, so the resolver
        decides again while an explicit choice wins), the resolved scene (size,
        grid, run, structures, sources, background, origin), the monitors as
        the user wrote them (``_user_monitors``: plane spans and faces are
        resolved again for the edited domain and grid, not carried over) and
        the update. Its ``model_fields_set`` is set to what direct construction
        with the same fields gives, which is what ``_wire_exclude`` keys the
        unset-field omission on, so the two wires are byte-identical. A
        declarative scene is constructed from the caller's own sources and
        monitors, its ports are solved again on the copy's grid, and the merged
        content is validated once more, as at construction. The copy is built
        inside :func:`_editing`: the advice on the caller's own inputs (a
        weakly driven monitor frequency, a wavelength typed as a frequency) is
        not repeated when the original drew the same; every refusal runs.

        A scene the constructor resolved (``_inputs``) is rebuilt from its
        inputs with the update, so a helper's result is what construction with
        the helper's fields gives and a later :meth:`with_changes` keeps it.
        The one exception is an update of stored coordinates on a ``domain=``
        fit (``with_auto_mesh``): that is a copy in those coordinates, and the
        inputs are marked stale (``_frame_edits``) so ``with_changes`` refuses
        instead of dropping it. Sources or monitors a ``model_copy`` swapped in
        over a scene's resolved ports are kept as stored, the ports with them,
        and an update of what the port solves read is refused (see
        :meth:`model_copy`).

        A document parsed from the wire keeps the ingestion contract: the copy
        is validated under ``wire_ingest`` (nothing resolved, nothing adjusted)
        except that the monitors the edit adds (all of them when it changes the
        grid or the size) have their plane spans resolved and their faces
        quarter-snapped (context ``wire_edit``), as the engine requires of
        them, and the construction rules a load skips hold (``edit_copy``). A
        declaration only the constructor resolves (a ``Domain``, a ``Mesh``, a
        run in transits, ports) is refused there."""
        with _editing(self):
            return self._edited_copy(_tupled(dict(update)))

    def _edited_copy(self, update: dict) -> "Simulation":
        """:meth:`_validated_copy`'s body, run inside :func:`_editing`."""
        given = dict(update)
        if self._wire_ingested:
            declared = sorted(
                [k for k in update if k in _decl.DECLARATIVE_FIELDS]
                + [k for k in ("size_um", "grid") if isinstance(update.get(k), (Domain, Mesh))]
                + (["run"] if getattr(update.get("run"), "transits", None) is not None else []))
            if declared:
                raise ValueError(
                    f"cannot apply {declared} to a simulation loaded from the wire: domain=, mesh=, a run in "
                    "transits and ports=/source=/wlens_um= are declarations only the constructor resolves, and a "
                    "loaded document is edited as the document it is. Give size_um, grid (a UniformMesh or "
                    "GradedMesh) and run_time_s or n_steps, or build the simulation with ph.Simulation(...).")
            kept = {name: getattr(self, name) for name in self.model_fields_set}
            # (what is no tuple here, validation refuses)
            monitors = update.get("monitors", self.monitors)
            monitors = monitors if isinstance(monitors, tuple) else ()
            if "grid" in update or "size_um" in update:
                added = tuple(getattr(m, "name", None) for m in monitors)      # every face on the new cells
            else:
                added = tuple(getattr(m, "name", None) for m in monitors if m not in self.monitors)
            # edit_copy: the construction rules a load skips hold for the edit
            return type(self).model_validate(
                {**kept, **update}, context={"wire_ingest": True, "wire_edit": added, "edit_copy": True})
        if self._inputs is not None and not self._frame_edits and not (
                set(given) & _FRAME_FIELDS and isinstance(self._inputs.get("size_um"), Domain)):
            return type(self)(**{**self._inputs, **given})
        kwargs = {name: getattr(self, name) for name in self.model_fields_set
                  if name not in self._auto_fields}
        kwargs.update({name: getattr(self, name) for name in _RESOLVED_FIELDS})
        for name in _decl.DECLARATIVE_FIELDS:
            kwargs.pop(name, None)
        # the monitors as written, so their spans and faces resolve for the copy
        kwargs["monitors"] = self._user_monitors if self._user_monitors is not None else self.monitors
        rec = self._declarative
        swapped = sorted(set(self._frame_edits) & {"sources", "monitors"})
        kept_rec = None
        if rec is not None and swapped and self._generated():
            # model_copy swapped the stored launch or readout (the S-matrix
            # plan, the adjoint solve): keep what is stored, ports and all, as
            # long as nothing the port solves read moves
            moved = sorted(k for k in set(given) & (_PORT_SOLVE_FIELDS | set(_decl.DECLARATIVE_FIELDS))
                           if given[k] != getattr(self, k))
            if moved:
                raise ValueError(
                    f"cannot change {moved} on this simulation: its {swapped} were replaced by model_copy over "
                    "the sources and monitors its ports resolved, and a change of what the port solves read "
                    "would leave them solved for the old scene. Make this change first and swap after it, or "
                    "build the simulation again with ph.Simulation(...).")
            kwargs["monitors"] = self.monitors
            kept_rec, rec = rec, None
        elif rec is not None:
            # the caller's own sources and monitors: an explicit update replaces
            # them (dropping the port planes it may carry re-snapped), otherwise
            # what was passed at construction (or swapped in over a record that
            # generates nothing, where the stored sources are the caller's);
            # the ports are solved again below
            generated = {p.monitor_name for p in rec.ports}
            own = rec.user_sources if self._generated() else self.sources
            update["sources"] = tuple(update.pop("sources", own))
            update["monitors"] = tuple(m for m in update.pop("monitors", kwargs["monitors"])
                                       if getattr(m, "name", None) not in generated)
        new = type(self)(**{**kwargs, **update})
        # the fields set as direct construction with the same fields sets them
        new.__pydantic_fields_set__.clear()
        new.__pydantic_fields_set__.update(
            (set(self.model_fields_set) - set(self._auto_fields)) | set(given) | set(new._auto_fields))
        if new._fold is None:
            new._fold = self._fold
        if new._run_transits is None and "run" not in given:
            new._run_transits = self._run_transits
        if self._inputs is not None:                     # stored coordinates of a domain= fit changed
            new._inputs = self._inputs
            new._frame_edits = tuple(sorted(set(self._frame_edits) | (set(given) & _FRAME_FIELDS)))
        if kept_rec is not None:
            for name in _decl.DECLARATIVE_FIELDS:          # the resolved ports, as stored
                object.__setattr__(new, name, getattr(self, name))
            new._declarative = kept_rec
            new._user_monitors = self._user_monitors
        elif rec is not None:
            for name in _decl.DECLARATIVE_FIELDS:          # the resolved ports, in the corner frame
                object.__setattr__(new, name, getattr(self, name))
            new._apply_declarative(user_sources=new.sources, user_monitors=new.monitors)
            # the merged wire-level content, as model_post_init checks it
            type(self).model_validate(
                new.model_dump(mode="python", exclude=set(_decl.DECLARATIVE_FIELDS) | {"origin_um"}),
                context={"wire_ingest": True, "edit_copy": True})
        return new

    @legacy_keywords(wavelength_um="wlen_um", steps_per_wvl="cells_per_wlen")
    def with_auto_mesh(
        self,
        *,
        wlen_um: Optional[float] = None,
        cells_per_wlen: float = 20.0,
        **auto_mesh_kwargs,
    ) -> "Simulation":
        """Return a COPY of this simulation whose ``grid`` is replaced by an
        auto-meshed :class:`GradedMesh` derived from this scene, its
        domain ``size_um``, ``structures``, ``background`` index, and (if
        ``wlen_um`` is omitted) the wavelength inferred from the first
        source. A convenience wrapper over :func:`photonhub.auto_mesh`; extra
        keyword arguments (``max_grading``, ``axes``, ``dl_min_um``,
        ``refine_regions``, ...) pass straight through.

        Opt-in only: the default :class:`UniformMesh` is unchanged, so no
        existing scene's wire output moves. Use this when you want per-medium
        per-medium refinement without hand-building coordinate arrays::

            sim = sim.with_auto_mesh(cells_per_wlen=20)

        Axes whose boundary is PERIODIC are passed to :func:`auto_mesh` as
        ``periodic_axes`` (unless you override it explicitly), so a graded
        periodic axis is generated seam-symmetrically, equal first/last
        primary spacings, the §15.2 closure requirement the engine hard-checks.
        Non-periodic scenes are byte-identical to before.

        An axis whose refinement set is already mirror-symmetric about the
        domain centre gets a MIRROR-SYMMETRIC ladder, so the two halves of a
        symmetric device are on the same discretization and a quantity
        symmetry guarantees (a splitter's arm balance) is not thrown off by a
        grid artifact. Pass ``mirror_axes=""`` to switch that off, or explicit
        letters to force it; :func:`auto_mesh` documents the detection, and
        :func:`photonhub.axis_mirror_mismatch` measures the result.
        """
        from .grid import auto_mesh as _auto_grid  # local: avoid import cycle

        bg_index = float(self.background.permittivity) ** 0.5
        src = self.sources[0] if self.sources else None
        if "periodic_axes" not in auto_mesh_kwargs:
            kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
            auto_mesh_kwargs["periodic_axes"] = "".join(
                _AXES[a] for a in range(3) if kinds[a] == "periodic")
        spec = _auto_grid(
            size_um=tuple(self.size_um),
            wlen_um=wlen_um,
            source=None if wlen_um is not None else src,
            structures=self.structures,
            background_index=bg_index,
            cells_per_wlen=cells_per_wlen,
            **auto_mesh_kwargs,
        )
        update: dict = {"grid": spec}
        # A constructed scene's copy quarter-snaps the monitors as written
        # against the NEW cell ladder (§12). The copy of a document parsed
        # from the wire is validated under ``wire_ingest``, which skips that
        # convenience, so its faces are snapped here: a face snapped for the
        # old uniform grid could land on a graded cell boundary and be
        # rejected by the engine.
        if self._wire_ingested:
            snapped, notes = _quarter_snapped_dft_monitors(
                self.monitors, size_um=self.size_um, grid=spec, axis_min_cells=self._axis_min_cells())
            if notes:
                update["monitors"] = snapped
                for note in notes:
                    _LOG.debug("%s", note)
        return self._validated_copy(update)

    @legacy_keywords(wavelength_um="wlen_um", steps_per_wvl="cells_per_wlen")
    def with_mesh_overrides(
        self,
        *overrides,
        wlen_um: Optional[float] = None,
        cells_per_wlen: float = 20.0,
        **auto_mesh_kwargs,
    ) -> "Simulation":
        """Return a COPY whose ``grid`` is auto-meshed with one or more
        geometry-based :class:`photonhub.MeshOverride` regions applied, the mesh
        is forced fine inside each
        override's geometry regardless of the local material, on top of the
        ordinary per-medium refinement.

        A thin wrapper over :meth:`with_auto_mesh` that forwards the overrides as
        ``mesh_overrides=``; all other auto-mesh knobs (``max_grading``, ``axes``,
        ``dl_min_um``, ``refine_pad_um``, ...) pass straight through, including
        the periodic-boundary seam handling: axes whose boundary is periodic get
        seam-symmetric coordinates (equal first/last primary spacings, the §15.2
        closure requirement). Opt-in only, like the other ``with_*`` mesh
        helpers, no existing scene's wire output moves unless you call it::

            from photonhub import MeshOverride, Box
            sim = sim.with_mesh_overrides(
                MeshOverride(geometry=Box(center_um=(2, 1, 0.5),
                                          size_um=(0.5, 0.5, 1.0)),
                             dl_um=(0.02, 0.02, None)),
                cells_per_wlen=20)
        """
        return self.with_auto_mesh(
            wlen_um=wlen_um, cells_per_wlen=cells_per_wlen,
            mesh_overrides=overrides, **auto_mesh_kwargs)

    def with_stabilized_pml(
        self,
        *,
        num_layers: int = _STABLE_PML_LAYERS,
        kappa_max: float = _STABLE_PML_KAPPA_MAX,
        alpha_scale: float = _STABLE_PML_ALPHA_SCALE,
    ) -> "Simulation":
        """Return a copy with the stabilized CPML profile.

        Increase the layer count, real-stretch peak ``kappa_max``, and complex
        frequency shift ``alpha_max`` (NUMERICS.md §11).
        ``alpha_scale`` uses dimensionless ``2*eps0/dt`` units, the same
        convention as ``pml_sigma_max``. Convert it to ``pml_alpha_max`` in S/m
        at the returned simulation's timestep. A fitted domain may change the
        timestep, so the conversion is recalculated after fitting.

        This helper keeps the timestep-based convention of the stabilized
        profile (40 layers, ``kappa_max`` 5, ``alpha_max`` 0.9).
        ``pml_sigma_max`` stays at its default (1.5 in ``2*eps0/dt`` units).
        The automatic dispersive profile instead anchors alpha to the band
        (NUMERICS.md §11); timestep-scaled alpha reflects more on finer grids.

        Dispersive scenes receive alpha and kappa stabilization at construction.
        Use this helper to also increase their layer count, or to opt in for a
        non-dispersive scene::

            sim = sim.with_stabilized_pml()
            sim = sim.with_stabilized_pml(num_layers=60, alpha_scale=1.2)
        """
        unit = self._two_eps0_over_dt()
        new = self._validated_copy({
            "pml_num_layers": num_layers,
            "pml_kappa_max": kappa_max,
            # alpha_scale is a fraction of 2*eps0/dt (the alpha_max unit),
            # converted to the engine's S/m field at the copy's dt.
            "pml_alpha_max": alpha_scale * unit,
        })
        if new._two_eps0_over_dt() != unit:
            # a domain= fit is fitted again for the new layers and its mesh
            # with it, so the time step can move: quote alpha at the copy's.
            # Nothing the fit, the mesh or the port solves decide reads alpha,
            # so the copy is the construction with that alpha once the value
            # is set; its rules are checked again without a second build.
            alpha = alpha_scale * new._two_eps0_over_dt()
            object.__setattr__(new, "pml_alpha_max", alpha)
            if new._inputs is not None:
                new._inputs = {**new._inputs, "pml_alpha_max": alpha}
            type(self).model_validate(
                new.model_dump(mode="python", exclude=set(_decl.DECLARATIVE_FIELDS) | {"origin_um"}),
                context={"wire_ingest": True, "edit_copy": True})
        return new

    @legacy_keywords(angle_theta="angle_theta_rad", angle_phi="angle_phi_rad")
    def with_oblique_plane_wave(
        self,
        *,
        axis: str,
        direction: str,
        position_um: float,
        polarization: str,
        source_time,
        angle_theta_rad: float,
        angle_phi_rad: float = 0.0,
        n: Optional[float] = None,
        amplitude: float = 1.0,
    ) -> "Simulation":
        """A copy with an OBLIQUE plane wave as the (only) source and the
        transverse axes configured as matching Bloch boundaries (schema 1.18,
        constant-k method; CPU solver only in this release).

        The in-plane Bloch wavevector is derived at the PULSE CENTRE:
        ``k_t = 2 pi n f0 / c * sin(theta)``, split onto the two cyclic
        transverse axes by ``angle_phi_rad`` (measured from the first cyclic
        transverse axis ``(axis+1) % 3``). ``n`` defaults to
        ``sqrt(background.permittivity)``, the index of the medium the wave
        is launched in. NOTE (constant-k): across a broadband pulse the
        physical angle varies with frequency; keep the band narrow when the
        angle matters (``sin theta(f) = f0 sin(theta) / f``). ``position_um``
        is in your coordinates, as the rest of a ``domain=`` fit is written.
        """
        import math as _math

        if not -0.5 * _math.pi < float(angle_theta_rad) < 0.5 * _math.pi:
            raise ValueError("angle_theta_rad must be within (-pi/2, pi/2)")
        c_phi = _math.cos(float(angle_phi_rad))
        s_phi = _math.sin(float(angle_phi_rad))
        # Engine v1 contract (NUMERICS.md §22): the tilt must lie along ONE
        # transverse axis — the single-aux-line s/p decomposition. Mirror the
        # engine's rejection here so it fails at authoring time.
        if min(abs(c_phi), abs(s_phi)) > 1e-9:
            raise ValueError(
                "angle_phi_rad must be a multiple of 90 degrees (pi/2) in this "
                "release: the oblique tilt must lie along a single "
                "transverse axis (NUMERICS.md §22)")
        n_bg = float(n) if n is not None else _math.sqrt(
            float(self.background.permittivity))
        # Engine §22 CFL contract: the incident aux line runs at
        # eps_eff = (n cos theta)^2, phase velocity c/(n cos theta) — FASTER
        # than the 3-D wave — so courant <= sqrt(3) * n * cos(theta).
        s_max = _math.sqrt(3.0) * n_bg * _math.cos(float(angle_theta_rad))
        if float(self.run.courant) > s_max * (1.0 + 1e-9):
            raise ValueError(
                f"run.courant = {self.run.courant} exceeds the oblique "
                f"incident-line stability bound sqrt(3)*n*cos(theta) = "
                f"{s_max:.4f} (NUMERICS.md §22); set run.courant to at most "
                f"{0.95 * s_max:.4f}, e.g. "
                f"sim.with_changes(run=sim.run.model_copy("
                f"update={{'courant': {0.95 * s_max:.3f}}}))")
        f0 = float(source_time.freq0_hz)
        k_t = 2.0 * _math.pi * n_bg * f0 / c0 * _math.sin(
            float(angle_theta_rad)) * 1e-6  # rad/um
        ax = "xyz".index(axis)
        t1, t2 = (ax + 1) % 3, (ax + 2) % 3
        k = [0.0, 0.0, 0.0]
        # Snap the numerically-zero component of the axis-aligned azimuth
        # exactly to 0 (cos(pi/2) ~ 6e-17), matching the engine's snap.
        k[t1] = k_t * c_phi if abs(c_phi) > 0.5 else 0.0
        k[t2] = k_t * s_phi if abs(s_phi) > 0.5 else 0.0
        kinds = {"x": self.boundaries.x, "y": self.boundaries.y,
                 "z": self.boundaries.z}
        for a in (t1, t2):
            kinds["xyz"[a]] = "bloch"
        # stored coordinates start at the fitted box's corner (origin_um)
        position_um = float(position_um) - float(self.origin_um["xyz".index(axis)])
        pw = PlaneWave(
            axis=axis, direction=direction, position_um=position_um,
            polarization=polarization, amplitude=amplitude,
            source_time=source_time, angle_theta_rad=float(angle_theta_rad),
            angle_phi_rad=float(angle_phi_rad))
        return self._validated_copy({
            "sources": (pw,),
            "boundaries": Boundaries(**kinds),
            "bloch_k_per_um": tuple(k),
        })

    def _absorber_layers(self, axes, *, keep_set: bool) -> int:
        """Layers per face for a section 21 absorber on ``axes`` that is at
        least ``_ABSORBER_WAVELENGTHS`` wavelengths thick in the background at
        the lowest source frequency, and at least ``_ABSORBER_MIN_LAYERS``.
        On a graded axis the cells are counted in from each face until they
        span that thickness. The lowest frequency is the lower 1-sigma edge of
        each pulse (floored at half its carrier), each cw carrier, and each
        monitored frequency. With ``keep_set`` a count the scene already sets
        is kept (it may have been sized for this mesh by an earlier call, or
        a builder may have sized the domain around it). Without a source the
        engine's 40 layers stand.
        Slabs that do not fit the domain are not refused here: like any
        run-time rule, :meth:`check_runnable` reports them before a run."""
        if keep_set and "absorber_num_layers" in self.model_fields_set:
            return self.absorber_num_layers
        lows = []
        for s in self.sources:
            st = getattr(s, "source_time", None)
            if st is None:
                continue
            width = getattr(st, "fwidth_hz", None)
            lows.append(max(st.freq0_hz - width, 0.5 * st.freq0_hz) if width else st.freq0_hz)
        if not lows:
            return _ABSORBER_MIN_LAYERS
        # A monitored frequency below the pulse's 1-sigma edge sets it instead.
        for mon in self.monitors:
            freqs = getattr(mon, "freqs_hz", None)
            if freqs:
                lows.append(min(float(f) for f in freqs))
        n_bg = math.sqrt(float(self.background.permittivity))
        wlen_um = _C0_M_PER_S / (min(lows) * n_bg) * 1e6
        need_um = _ABSORBER_WAVELENGTHS * wlen_um
        counts = resolved_cell_counts(self.size_um, self.grid, self._axis_min_cells())
        layers = _ABSORBER_MIN_LAYERS
        for a in axes:
            q = self._axis_coords_um(a)
            spacings = (graded_primary_spacings(q) if q is not None
                        else [self.grid.dl_um] * counts[a])
            for side in (spacings, spacings[::-1]):
                total, cells = 0.0, 0
                for dq in side:
                    if total >= need_um * (1.0 - 1e-9):
                        break
                    total += dq
                    cells += 1
                if total < need_um * (1.0 - 1e-9):
                    cells = counts[a]  # the axis is thinner than the absorber
                layers = max(layers, cells)
        return layers

    def with_absorber(self, *, num_layers: Optional[int] = None) -> "Simulation":
        """Return a COPY with every face set to the adiabatic absorber
        (NUMERICS.md §21) instead of a PML. Use this when a structure crosses
        the domain boundary or a dispersive/gain medium touches the edge, the
        cases where a stretched-coordinate PML can diverge. The absorber trades
        reflection for robustness: its reflection depends on its thickness in
        wavelengths, not on the mesh. By default the slab is two wavelengths
        thick in the background at the lowest source or monitor frequency
        (``freq0_hz - fwidth_hz`` of a pulse), and never thinner than 40
        layers (at 1550 nm: 3.9 um per face in vacuum, 2.7 um in SiO2); at
        normal incidence, for a wave in a vacuum or SiO2 background, that
        reflects about -52 to -60 dB at the band's low edge and less above it
        (a PML: -68 dB and below). The count is recomputed on every call, so
        call it again after changing the mesh; ``num_layers`` sets it instead
        (one wavelength of the carrier reflects about -16 dB)::

            sim = sim.with_absorber()                 # two wavelengths, 6 faces
            sim = sim.with_absorber(num_layers=60)

        A domain too small for the slabs fails :meth:`check_runnable` (and so
        ``run_local``) with the layer count; enlarge it, or pass a smaller
        ``num_layers`` and accept the higher reflection. The count is one for
        all axes, set by the axis that needs the most cells (on a graded mesh,
        the one with the finest face cells). At normal incidence a plane wave
        in a medium of index ``n`` inside the slab is reflected at least about
        ``-139/n`` dB, however thick the slab: -40 dB in bulk silicon; a
        guided mode's bound depends on its group index and field and is
        usually lower (NUMERICS.md section 21).
        """
        if num_layers is None:
            num_layers = self._absorber_layers(range(3), keep_set=False)
        return self._validated_copy({
            "boundaries": Boundaries(x="absorber", y="absorber", z="absorber"),
            "absorber_num_layers": num_layers,
        })

    def with_auto_boundaries(self, *, absorber_num_layers: Optional[int] = None) -> "Simulation":
        """Return a COPY whose OPEN (radiating) boundaries are chosen PER AXIS
        from the materials that reach the domain edge, following standard
        material-aware guidance that a stretched-coordinate PML wants a
        non-dispersive medium in its absorbing region:

        * an axis where a **dispersive (Lorentz) medium crosses the boundary**
          gets the adiabatic **absorber** (graded electric conductivity,
          NUMERICS.md §21), the robust fallback for the regime where a PML can
          diverge (Oskooi & Johnson 2011);
        * every other open axis keeps the (thinner, lower-floor) **PML**.

        ``periodic`` and ``pec`` axes are left untouched: those encode explicit
        physics (a Bloch / transverse-infinite axis, a hard mirror), not an open
        boundary to auto-select. Opt-in, the default boundaries are unchanged,
        so no existing scene's wire output moves unless you call this::

            sim = sim.with_auto_boundaries()   # PML, but absorber where dispersive

        This is the programmatic form of the construction-time warning the same
        crossing raises; call it to act on that advice in one line.

        The open-boundary decision has three cases: a well-behaved scene keeps
        the default PML; a scene that *diverges* without a dispersive edge
        (grazing / long run) wants ``with_stabilized_pml()``; a dispersive
        medium at the wall wants the absorber this method selects.

        The absorber is sized as in :meth:`with_absorber`: two wavelengths in
        the background at the lowest source or monitor frequency, at least 40
        layers; ``absorber_num_layers`` gives the count instead (unused when
        no axis gets the absorber). Unlike :meth:`with_absorber`, a count the
        scene already sets is kept (a builder may have sized the domain
        around it), and it is a count of cells: after changing the mesh, pass
        ``absorber_num_layers`` or call :meth:`with_absorber`. A domain too
        small for the slabs fails :meth:`check_runnable` before a run.
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
        update = {"boundaries": Boundaries(x=chosen[0], y=chosen[1], z=chosen[2])}
        absorber_axes = [a for a in range(3) if chosen[a] == "absorber"]
        if absorber_axes:
            update["absorber_num_layers"] = (
                absorber_num_layers if absorber_num_layers is not None
                else self._absorber_layers(absorber_axes, keep_set=True))
        return self._validated_copy(update)

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
                    "symmetrically — auto_mesh(periodic_axes=...); "
                    "with_auto_mesh/with_mesh_overrides pass it from the "
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

        def check(center, label: str, axis=None) -> None:
            if len(center) == 2:
                # A plane WINDOW centre is a (u, v) pair in the plane's CYCLIC
                # transverse order u = (axis+1) % 3, v = (axis+2) % 3
                # (PowerMonitor docstring; engine resolve.cpp): z-normal ->
                # (x, y), x-normal -> (y, z), y-normal -> (z, x). Compare each
                # entry against ITS axis, not against x then y.
                if axis is None:
                    return  # no plane to map against; phsolver validates
                a = _AXES.index(axis)
                order = ((a + 1) % 3, (a + 2) % 3)
                pairs = list(zip(center, order))
                where = (f" ({_AXES[order[0]]}, {_AXES[order[1]]} for the "
                         f"{axis}-normal plane)")
            else:
                pairs = [(c, i) for i, c in enumerate(center)]
                where = ""
            for c, i in pairs:
                if not (0.0 <= c <= realized[i]):
                    raise ValueError(
                        f"{label}.center_um {tuple(center)}{where} is outside "
                        f"the domain {domain}"
                    )

        for i, s in enumerate(self.sources):
            center = getattr(s, "center_um", None)  # plane waves have none
            if center is not None:
                check(center, f"sources[{i}]", getattr(s, "axis", None))
            elif isinstance(s, PlaneWave):
                axis = _AXES.index(s.axis)
                if not (0.0 <= s.position_um <= realized[axis]):
                    raise ValueError(
                        f"sources[{i}].position_um {s.position_um} ({s.axis} "
                        f"axis) is outside the domain {domain}"
                    )
        for m in self.monitors:
            # Snapshots and full-plane flux monitors have none; a flux WINDOW
            # carries a 2-tuple (u, v) centre in cyclic order.
            center = getattr(m, "center_um", None)
            if center is not None:
                check(center, f"monitor '{m.name}'", getattr(m, "axis", None))
        return self

    @model_validator(mode="after")
    def _plane_wave_transverse_axes_periodic(self) -> "Simulation":
        # NUMERICS.md sections 13 and 22, mirrored exactly (no float math, safe
        # to enforce strictly; engine resolve.cpp): a plane wave requires both
        # transverse axes periodic or bloch (the wrap an oblique wave's phase
        # advance needs, set by with_oblique_plane_wave).
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        for i, s in enumerate(self.sources):
            if not isinstance(s, PlaneWave):
                continue
            for t, kind in enumerate(kinds):
                if _AXES[t] != s.axis and kind not in ("periodic", "bloch"):
                    raise ValueError(
                        f"sources[{i}] (plane_wave along {s.axis}): transverse "
                        f"axis '{_AXES[t]}' must be periodic (or bloch), got "
                        f"'{kind}' (NUMERICS.md sections 13 and 22)"
                    )
        return self

    @model_validator(mode="after")
    def _resolve_subpixel_default(self, info) -> "Simulation":
        # D2 (NUMERICS.md §16): default-ON subpixel smoothing for the common
        # case. When ``subpixel`` is NOT set explicitly, enable the diagonal-KFJ
        # ``tensor`` average (matching the common subpixel-on posture, the more
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
        # choice, and recorded in ``_auto_fields`` so an edit (with_changes,
        # the with_* helpers) resolves it again for the edited scene. An
        # explicit ``subpixel`` is always respected verbatim; an explicit
        # subpixel-ON dispersive scene only gets a warning, never an override.
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
            s.medium.is_dispersive
            for s in self.structures
        )
        # §10.2 / §10.3: anisotropic and custom (data-grid) media follow the
        # dispersive policy — subpixel smoothing of either constituent is
        # deferred, and the engine REJECTS subpixel != off with them.
        anisotropic = any(
            getattr(s.medium, "is_anisotropic", False)
            or getattr(s.medium, "is_custom", False)
            for s in self.structures
        )
        dispersive = dispersive or anisotropic
        auto = set()
        if "subpixel" not in self.model_fields_set:
            if not dispersive:
                # Enable + mark set so it serialises (engine field default = off).
                object.__setattr__(self, "subpixel", True)
                self.__pydantic_fields_set__.add("subpixel")
                auto.add("subpixel")
                if "subpixel_method" not in self.model_fields_set:
                    object.__setattr__(self, "subpixel_method", "contour")
                    self.__pydantic_fields_set__.add("subpixel_method")
                    auto.add("subpixel_method")
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
                auto.add("subpixel_method")
        # an edit leaves these out, so this resolution runs again (_validated_copy)
        self._auto_fields = self._auto_fields | auto
        # the advice below is moot once the user set a CFS-active alpha
        # (with_stabilized_pml, or pml_alpha_max by hand)
        stabilized = ("pml_alpha_max" in self.model_fields_set and "pml_alpha_max" not in self._auto_fields
                      and not self._pml_alpha_is_cfs_inert())
        if self.subpixel and dispersive and "subpixel" in self.model_fields_set and not stabilized:
            warnings.warn(
                "subpixel smoothing is enabled on a dispersive (Lorentz) scene. "
                "Dispersive scenes with default-profile PML can diverge "
                "late-time at fine grids — root cause is a pole-assisted "
                "trapped resonance × the CFS-inert PML, NOT the smoothing "
                "itself, but smoothing shifts the resonance and has triggered "
                "it in production scenes "
                "(engine/docs/subpixel-dispersion-instability.md). Use "
                "sim.with_stabilized_pml() with subpixel+dispersive runs.",
                stacklevel=caller_stacklevel(),
            )
        # the resolved value of ``subpixel`` is only known here, so the
        # quasi-2-D dilution warning runs at the end of this validator
        self._warn_degenerate_axis_subpixel()
        return self

    @model_validator(mode="after")
    def _ade_stability(self, info) -> "Simulation":
        """NUMERICS.md §19.4, mirrored from the engine's validate() (beta review
        CORE-02): every Lorentz pole needs omega0*dt < 2, and then the medium's
        discrete permittivity at the temporal Nyquist must cover the largest
        spatial eigenvalue,

            eps_inf - sum_Lorentz delta_eps*x/(1-x) - sum_Drude (wp*dt/2)^2 >= courant^2,
            x = (omega0*dt/2)^2,

        at the resolved dt. The omega0*dt < 2 rule alone passed specs that are
        certain to diverge: a telecom glass fit at omega0*dt = 1.06 (it was a
        warning here), a Drude metal with eps_inf near 1 at a 20 nm mesh. The
        message quotes the largest courant that holds and the grid refinement
        that would. Skipped on a document being loaded, where the engine is the
        authority.
        (The bound is also in materials.py's fit helpers; one copy should
        serve both.)"""
        if _loaded_document(info):
            return self
        courant = float(self.run.courant)
        dt = 2.0 * _EPS0 / self._two_eps0_over_dt()
        for i, st in enumerate(self.structures):
            medium = st.medium
            single = getattr(medium, "lorentz", None)
            lorentz = []
            for k, pole in enumerate(_medium_poles(medium)):
                w0 = 2.0 * math.pi * pole.resonance_frequency_hz
                field = "lorentz" if (single is not None and k == 0) else \
                    f"poles[{k - (1 if single is not None else 0)}]"
                if not w0 * dt < 2.0:
                    raise ValueError(
                        f"structures[{i}].medium.{field}: omega0*dt must be < 2 for ADE stability "
                        f"(NUMERICS.md §19.4); got omega0*dt = {w0 * dt:.9g}. Refine the grid (a smaller dl "
                        "lowers dt) or move the resonance below the time-step Nyquist.")
                lorentz.append((w0, float(pole.delta_eps)))
            drude = [2.0 * math.pi * p.plasma_frequency_hz for p in (getattr(medium, "drude", None) or ())]
            if not lorentz and not drude:
                continue
            eps_inf = float(medium.permittivity)
            e_nyq = _ade_nyquist_eps(eps_inf, lorentz, drude, dt)
            if e_nyq >= courant * courant:
                continue
            c_max = math.floor(_ade_max_courant(eps_inf, lorentz, drude, dt, courant) * 1000.0) / 1000.0
            dt_max = _ade_max_dt(eps_inf, lorentz, drude, dt, courant)
            finer = math.ceil(dt / dt_max * 100.0) / 100.0
            lower = f"lower run.courant to <= {c_max:.3f}, or " if c_max > 0.0 else ""
            raise ValueError(
                f"structures[{i}].medium: ADE stability (NUMERICS.md §19.4) needs the discrete Nyquist "
                "permittivity eps_inf - sum_Lorentz delta_eps*x/(1-x) - sum_Drude (wp*dt/2)^2 "
                f"(x = (omega0*dt/2)^2) >= courant^2 = {courant * courant:.9g}; got {e_nyq:.9g} at dt = "
                f"{dt:.9g} s: {lower}refine the grid until dt <= {dt_max:.9g} s (about {finer:.2f}x finer at "
                "this courant).")
        return self

    @model_validator(mode="after")
    def _warn_bloch_with_undamped_poles(self, info) -> "Simulation":
        """Bloch wrap x low-loss resonance: warn that stability is Courant-
        NON-monotone (validation/suites/meep FINDINGS.md F6).

        Measured on the Meep material-dispersion scene (two Lorentz poles, a
        4-cell Bloch axis): k = 1.8 (2 pi/a) is stable at courant 0.99 but
        DIVERGES at 0.7, while k = 2.1 diverges at 0.99 and is stable at 0.7 ,
        a resonance between dt, the undamped pole, and the Bloch phase, not a
        CFL margin. Until the NUMERICS 19/22 preflight covers the joint
        (pole, k, dt) spectrum, surface the failure mode and the remedy (a
        Courant RETRY LADDER, not a single lower value) at authoring time.
        Skipped on wire ingest (a parsed document is a deliberate choice).
        """
        if (info.context or {}).get("wire_ingest"):
            return self
        if not self.bloch_k_per_um or not any(self.bloch_k_per_um):
            return self
        low_loss = []
        for structure in self.structures:
            m = structure.medium
            for pole in _medium_poles(m):
                if pole.linewidth_hz < 1e-3 * pole.resonance_frequency_hz:
                    low_loss.append(structure.name or "structure")
                    break
        if low_loss:
            warnings.warn(
                "Bloch boundaries with low-loss Lorentz pole(s) "
                f"({', '.join(low_loss[:3])}"
                f"{', ...' if len(low_loss) > 3 else ''}): stability is "
                "NON-monotone in the Courant number — a (k, dt) combination "
                "can diverge at courant 0.7 yet run at 0.99, and vice versa "
                "(a dt x pole x Bloch-phase resonance, not a CFL margin). If "
                "the run aborts with 'divergence', retry over a Courant "
                "LADDER (e.g. 0.99, 0.7, 0.5, 0.35) instead of assuming "
                "lower is safer.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        return self

    @model_validator(mode="after")
    def _warn_dispersive_media_in_pml(self, info) -> "Simulation":
        # Material-aware boundary guidance: a stretched-
        # coordinate PML derives its absorbing profile assuming a NON-dispersive
        # medium, so a dispersive (Lorentz) structure extending into the PML can
        # drive a late-time divergence (Oskooi & Johnson, "Distinguishing
        # correct from incorrect PML proposals...", J. Comput. Phys. 2011). The
        # adiabatic absorber (graded electric conductivity, NUMERICS.md §21) is
        # the robust fallback for exactly that regime. WARN — never override —
        # matching the conservative posture, so the wire output is unchanged and the user
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
                stacklevel=caller_stacklevel(),
            )
        return self

    def point_sources_in_boundary_layers(self) -> list:
        """Point dipoles whose centre lies inside the PML/absorber band.

        The engine accepts such a source and runs it, but the boundary layers
        absorb it in place, so every recorded spectrum comes out physically
        meaningless and near zero with no diagnostic. Construction does NOT
        warn (geometry-only simulations legitimately carry a
        placeholder dipole that is never run); :func:`photonhub.run_local`
        warns before launching. Uniform grids only (the band is
        ``layers * dl``). On a §20 symmetry axis the band is one-sided: the
        min face is the mirror, not an absorber, so only the far face is
        tested there. Returns ``[(source_index, axis_name, band_um), ...]``
        (empty when every point source is interior).
        """
        dl = getattr(self.grid, "dl_um", None)
        if dl is None or not self.sources:
            return []
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        realized = self._realized_um()
        hits = []
        for i, src in enumerate(self.sources):
            center = getattr(src, "center_um", None)
            if center is None or getattr(src, "type", "") != "point_dipole":
                continue
            for a in range(3):
                if kinds[a] == "pml":
                    band = self.pml_num_layers * dl
                elif kinds[a] == "absorber":
                    band = self.absorber_num_layers * dl
                else:
                    continue
                # NUMERICS.md §20: a symmetry plane replaces the MIN-face
                # absorbing slab (the PML on that axis is built one-sided;
                # same rule as _nonabsorbing_bounds_um's lower_layers), so
                # only the far face can absorb the source there. A folded
                # mode launch legitimately puts its equivalence-current
                # dipoles a fraction of a cell from the mirror.
                lower = 0.0 if self.symmetry[a] != 0 else band
                if center[a] < lower or center[a] > realized[a] - band:
                    hits.append((i, _AXES[a], band))
                    break
        return hits

    def _pml_band_hz(self) -> float:
        """The frequency the dispersive PML profile is anchored to: the
        highest source carrier, else the band the scene declares (wlen0_um,
        then its shortest wlens_um), else 1550 nm."""
        carriers = [s.source_time.freq0_hz for s in self.sources
                    if getattr(s, "source_time", None) is not None]
        if carriers:
            return max(carriers)
        declared = []
        if self.wlen0_um is not None:
            declared = [float(self.wlen0_um)]
        elif self.wlens_um is not None:
            wlens = self.wlens_um if isinstance(self.wlens_um, (tuple, list)) else (self.wlens_um,)
            declared = [min(float(w) for w in wlens)]
        if declared and declared[0] > 0.0:
            return _C0_M_PER_S / (declared[0] * 1e-6)
        # No usable band (none declared, or a non-positive wavelength): the
        # nominal one, never a negative or infinite alpha.
        return _C0_M_PER_S / (_NOMINAL_BAND_WLEN_UM * 1e-6)

    def _pml_alpha_is_cfs_inert(self) -> bool:
        """Whether ``pml_alpha_max`` is below ``_CFS_INERT_SCALE * eps0 *
        omega0`` of the band (:meth:`_pml_band_hz`), too low to cure the
        dispersive late-time divergence: the one test the explicit-profile
        warning and the subpixel advice share."""
        return self.pml_alpha_max < _CFS_INERT_SCALE * 2.0 * math.pi * _EPS0 * self._pml_band_hz()

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
        # band-anchored alpha_max eps0*omega0 (_AUTO_PML_ALPHA_SCALE, omega0
        # from the highest source carrier; the reference-parity 0.9*2*eps0/dt
        # dose reflects ~35% of a propagating guided mode at the default 12
        # layers), the two levers that never change the slab
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
        if not any(s.medium.is_dispersive
                   for s in self.structures):
            return self
        if "pml" not in (self.boundaries.x, self.boundaries.y, self.boundaries.z):
            return self
        pml_knobs = ("pml_num_layers", "pml_kappa_max", "pml_alpha_max",
                     "pml_sigma_max")
        if not any(k in self.model_fields_set for k in pml_knobs):
            # Auto-stabilize: raise kappa + the CFS alpha (fit-safe), mark set so
            # they ride the wire. sigma_max (default 1.5, already in the stabilized profile's
            # 2*eps0/dt convention) and the 12-layer count are left untouched.
            # The alpha DOSE is _AUTO_PML_ALPHA_SCALE * eps0*omega0 at the
            # highest carrier, the same S/m at every resolution (see
            # _AUTO_PML_ALPHA_SCALE above): at the reflection floor of the
            # default profile, 2x above the lowest dose the rod probe holds
            # flat.
            object.__setattr__(self, "pml_kappa_max", _STABLE_PML_KAPPA_MAX)
            self.__pydantic_fields_set__.add("pml_kappa_max")
            object.__setattr__(self, "pml_alpha_max",
                               _AUTO_PML_ALPHA_SCALE * 2.0 * math.pi * _EPS0 * self._pml_band_hz())
            self.__pydantic_fields_set__.add("pml_alpha_max")
            # the alpha is the grid's: an edit leaves both out so they resolve again
            self._auto_fields = self._auto_fields | {"pml_kappa_max", "pml_alpha_max"}
            return self
        # The user owns the PML profile — respect it verbatim, but warn if the
        # alpha they set is still CFS-inert (below _CFS_INERT_SCALE * eps0 *
        # omega0 of the band): the divergence lever is unaddressed and the run
        # may drift late-time.
        if self._pml_alpha_is_cfs_inert():
            warnings.warn(
                "dispersive (Lorentz) scene with PML faces and an explicitly-set "
                "but CFS-inert PML profile: an undamped pole can trap a high-Q "
                "resonance whose evanescent tail the plain sigma-graded PML "
                "re-amplifies — late-time divergence even with every dispersive "
                "structure far from the wall, and independent of subpixel "
                "smoothing (engine/docs/subpixel-dispersion-instability.md). "
                "Raise pml_alpha_max, or use sim.with_stabilized_pml() for the "
                "Stabilized-CPML profile.",
                stacklevel=caller_stacklevel(),
            )
        return self

    @model_validator(mode="after")
    def _flux_planes_inside_domain(self) -> "Simulation":
        # Best-effort mirror of the engine's flux-plane bound (NUMERICS.md
        # section 12: snapped plane index 1 <= kp <= n_axis - 1); phsolver
        # remains authoritative at exact half-cell positions. The engine's
        # second rule, clearing the boundary layers, is a run check
        # (check_runnable): a plotted scene or a shell may carry such a plane.
        dl = self.grid.dl_um
        for m in self.monitors:
            if not isinstance(m, PowerMonitor):
                continue
            axis = _AXES.index(m.axis)
            # Graded axis: the coordinate-based plane snap (NUMERICS.md
            # section 15.6) is the engine's; skip the uniform-dl best-effort
            # check here (phsolver validate remains authoritative).
            if self._axis_coords_um(axis) is not None:
                continue
            n = realized_cells(
                self.size_um[axis], dl, self._axis_min_cells()[axis]
            )
            kp = snapped_plane_index(m.position_um, dl)
            if not (1 <= kp <= n - 1):
                raise ValueError(
                    f"monitor '{m.name}': flux plane at position_um "
                    f"{m.position_um} snaps to {m.axis}-plane index {kp}, "
                    f"outside the interior range [1, {n - 1}] "
                    "(NUMERICS.md section 12)"
                )
        return self

    # ---- the engine's cheap checks (beta review INT-06) ------------------------
    # Each mirrors a phsolver validate() rule the client has everything for (the
    # realized cells, dt, the sources' spectra), so a mistake fails before a
    # quote or the solver. The rules on a scene's own sources, media and
    # smoothing are raised at construction, below. The rules on whether the
    # scene can run at all (the boundary slabs fit, a flux plane clears them,
    # a cw run outlasts its ramp, there is a source) are check_runnable's: a
    # scene built to plot or to serve a mode solve may break them. A document
    # loaded from the wire skips the construction rules: the engine is the
    # authority on an ingested document, and one that breaks a rule must
    # still load so it can be fixed.

    @model_validator(mode="after")
    def _monitor_freqs_in_source_band(self, info) -> "Simulation":
        """engine resolve.cpp (NUMERICS.md §12): the first source is the
        normalization source; every field or flux monitor frequency must lie
        within 12 fwidth of its freq0 (beyond that 1/(A0*S(f)) is not
        representable in float32), or be the carrier of a cw one. Warns (beta
        review API-06) where the pulse's spectral amplitude is below 1e-3 of
        its peak: the normalized spectrum there is noise over almost no drive,
        silently orders of magnitude off."""
        if _loaded_document(info) or not self.sources:
            return self
        pulse = self.sources[0].source_time
        f0 = float(pulse.freq0_hz)
        for m in self.monitors:
            if not isinstance(m, (ProfileMonitor, PowerMonitor)):
                continue
            for k, f in enumerate(m.freqs_hz):
                f = float(f)
                if isinstance(pulse, CW):
                    if abs(f - f0) > 1e-9 * f0:
                        raise ValueError(
                            f"monitor '{m.name}': freqs_hz[{k}] = {f:.9g} Hz is not the carrier {f0:.9g} Hz "
                            "of the cw source that normalizes it (sources[0]); under a cw source every field "
                            "or flux monitor reads the carrier only (NUMERICS.md §5-CW, §12). Set freqs_hz to "
                            "the carrier, or put a GaussianPulse source first.")
                    continue
                fwidth = float(pulse.fwidth_hz)
                sigmas = abs(f - f0) / fwidth
                if sigmas > 12.0:
                    raise ValueError(
                        f"monitor '{m.name}': freqs_hz[{k}] = {f:.6g} Hz is {sigmas:.3g} fwidth from the freq0 "
                        f"{f0:.6g} Hz of the source that normalizes it (sources[0], fwidth {fwidth:.6g} Hz); "
                        "beyond 12 fwidth the 1/(A0*S(f)) normalization is not representable "
                        "(NUMERICS.md §12). Keep monitor frequencies in the pulse's band: "
                        "GaussianPulse.for_band(freqs_hz=...) fits a pulse to them.")
        # the advice is construction's: an edit's copy re-checks the refusal
        # above but repeats the advice only when the edit brings it in
        weak = self._weak_drive()
        if weak and not (info.context or {}).get("wire_ingest") and not _advised_before("_weak_drive", weak):
            amplitude, name, f = min(weak)
            names = sorted({w[1] for w in weak})
            warnings.warn(
                f"monitor(s) {names} read frequencies where the source that normalizes them (sources[0], "
                f"freq0 {f0:.6g} Hz, fwidth {float(pulse.fwidth_hz):.6g} Hz) has under "
                f"{_WEAK_SPECTRAL_AMPLITUDE:g} of its peak spectral amplitude (monitor {name!r} at {f:.6g} Hz: "
                f"{amplitude:.1e}). The spectrum there is noise over almost no drive and can be off by orders "
                "of magnitude. Keep monitor frequencies in the pulse's band: "
                "GaussianPulse.for_band(freqs_hz=...) fits a pulse to them.",
                UserWarning, stacklevel=caller_stacklevel())
        return self

    def _weak_drive(self) -> list:
        """``[(amplitude, monitor name, frequency)]`` for every field or flux
        monitor frequency within the 12-fwidth band where the first source's
        pulse has under ``_WEAK_SPECTRAL_AMPLITUDE`` of its peak spectral
        amplitude (none under a cw source)."""
        if not self.sources or isinstance(self.sources[0].source_time, CW):
            return []
        pulse = self.sources[0].source_time
        f0, fwidth = float(pulse.freq0_hz), float(pulse.fwidth_hz)
        weak = []
        for m in self.monitors:
            if isinstance(m, (ProfileMonitor, PowerMonitor)):
                for f in map(float, m.freqs_hz):
                    amplitude = pulse.spectral_amplitude(f)
                    if abs(f - f0) <= 12.0 * fwidth and amplitude < _WEAK_SPECTRAL_AMPLITUDE:
                        weak.append((amplitude, m.name, f))
        return weak

    def _typed_wavelengths(self) -> list:
        """``[(source index, freq0_hz)]`` for every source whose free-space
        wavelength is over 1000 times the domain."""
        extent_m = max(self._realized_um()) * 1e-6
        return [(i, float(src.source_time.freq0_hz)) for i, src in enumerate(self.sources)
                if _C0_M_PER_S / float(src.source_time.freq0_hz) > 1e3 * extent_m]

    def _typed_frequencies(self) -> list:
        """The ``freq0_hz`` of :meth:`_typed_wavelengths`, which an edit that
        moves the sources compares (their indices shift)."""
        return [f0 for _, f0 in self._typed_wavelengths()]

    @model_validator(mode="after")
    def _warn_non_optical_source(self, info) -> "Simulation":
        """Beta review API-06: a free-space wavelength over 1000 times the
        domain is almost always a wavelength typed where a frequency is
        expected (``freq0_hz=1.55`` for 1.55 um), which runs to zeros. An edit
        repeats it only when the edit brings it in."""
        if (info.context or {}).get("wire_ingest") or not self.sources:
            return self
        extent_m = max(self._realized_um()) * 1e-6
        long = self._typed_wavelengths()
        if long and not _advised_before("_typed_frequencies", [f0 for _, f0 in long]):
            i, f0 = long[0]
            more = f" (and {len(long) - 1} more source(s))" if len(long) > 1 else ""
            warnings.warn(
                f"sources[{i}].source_time.freq0_hz = {f0:.6g} Hz{more} is a free-space wavelength of "
                f"{_C0_M_PER_S / f0:.3g} m, over 1000 times the {extent_m * 1e6:.4g} um domain: a wavelength "
                "typed where a frequency is expected? Frequencies are in Hz; 1.55 um is "
                f"{_C0_M_PER_S / 1.55e-6:.6g} Hz (the speed of light over the wavelength).",
                UserWarning, stacklevel=caller_stacklevel())
        return self

    @model_validator(mode="after")
    def _full_tensor_subpixel_rules(self, info) -> "Simulation":
        """engine resolve.cpp (NUMERICS.md §10.1/§16.6/§16.11): the full
        off-diagonal methods ``tensor_full`` and ``contour_full`` are lossless
        and uniform-grid only. Refused with an absorber axis, a dispersive
        medium, a PEC structure (beta review SUB-01: the tensor overrides the
        conductor's pinned field, so a curved conductor diverges and a box
        leaks into its inside) or a graded axis (INT-07)."""
        if _loaded_document(info) or not self.subpixel:
            return self
        method = self.subpixel_method
        if method not in ("tensor_full", "contour_full"):
            return self
        diag = "tensor" if method == "tensor_full" else "contour_diag"
        use = f"use subpixel_method='{diag}' (the diagonal form)"
        if "absorber" in (self.boundaries.x, self.boundaries.y, self.boundaries.z):
            raise ValueError(
                f"subpixel_method='{method}' with an absorber boundary: the full off-diagonal tensor is "
                "lossless-only and the absorber folds a graded conductivity into the same cells "
                f"(NUMERICS.md §16.6/§16.11); {use}, or PML boundaries.")
        for i, st in enumerate(self.structures):
            if st.medium.is_dispersive:
                raise ValueError(
                    f"subpixel_method='{method}' with the dispersive medium of structures[{i}]: the ADE "
                    "correction pairs with the diagonal update the full tensor overwrites "
                    f"(NUMERICS.md §16.6/§16.11); {use}.")
            if st.medium.pec:
                raise ValueError(
                    f"subpixel_method='{method}' with the PEC structures[{i}]: the full tensor does not honour "
                    "the PEC pin (NUMERICS.md §10.1/§16.6), so a curved conductor diverges and a box leaks "
                    f"field inside it; {use}.")
        graded = [_AXES[a] for a in range(3) if self._axis_coords_um(a)]
        if graded:
            raise ValueError(
                f"subpixel_method='{method}' on a graded mesh (axes {', '.join(graded)}): full-tensor "
                f"smoothing on a graded mesh is deferred (NUMERICS.md §16.6/§16.11); {use}, 'volume', or "
                "subpixel=False.")
        return self

    def check_runnable(self) -> None:
        """Raise ``ValueError`` naming every rule this simulation breaks that
        stops the solver from running it, each with its field and the fix. A
        scene may break these and still serve a plot or a mode solve, so
        construction allows them; :func:`photonhub.run_local` calls this before
        it starts the solver. The rules mirror the solver's own validation:

        - at least one source;
        - on a PML or absorber axis the slabs (two, one on a symmetry axis)
          leave cells between them: slabs * layers < cells (NUMERICS.md
          §11/§21);
        - a flux plane clears the boundary layers of its axis, L+1 <= kp <=
          n-L-1, as the flux reads E at plane kp and H at kp-1 and kp
          (NUMERICS.md §12; uniform axes, the solver judges graded ones);
        - the run outlasts every cw source's ramp (NUMERICS.md §5-CW)."""
        problems = []
        if not self.sources:
            problems.append(
                "sources: at least one source is required to run a simulation; add a source (or ports= "
                "with source=). A simulation without sources serves mode solves and plots.")
        counts = resolved_cell_counts(self.size_um, self.grid, self._axis_min_cells())
        for a, axis in enumerate(_AXES):
            kind = getattr(self.boundaries, axis)
            if kind not in ("pml", "absorber"):
                continue
            field = "pml_num_layers" if kind == "pml" else "absorber_num_layers"
            layers = getattr(self, field)
            slabs = 1 if self.symmetry[a] != 0 else 2
            if slabs * layers >= counts[a]:
                problems.append(
                    f"boundaries.{axis}: {slabs} * {field} ({slabs * layers}) must be < the {counts[a]} "
                    f"cells of axis {axis!r} (size_um[{a}] = {self.size_um[a]:g} um): the {kind} slabs "
                    f"overlap (NUMERICS.md §11/§21). Enlarge size_um[{a}], refine the mesh or lower {field}.")
        dl = self.grid.dl_um
        for m in self.monitors:
            if not isinstance(m, PowerMonitor):
                continue
            a = _AXES.index(m.axis)
            kind = getattr(self.boundaries, m.axis)
            layers = {"pml": self.pml_num_layers, "absorber": self.absorber_num_layers}.get(kind, 0)
            if not layers or self._axis_coords_um(a) is not None:
                continue
            n = realized_cells(self.size_um[a], dl, self._axis_min_cells()[a])
            kp = snapped_plane_index(m.position_um, dl)
            if not (layers + 1 <= kp <= n - layers - 1):
                label = "PML" if kind == "pml" else "absorber"
                where = (f"a position_um between {(layers + 1) * dl:g} and {(n - layers - 1) * dl:g} um"
                         if n - layers - 1 >= layers + 1 else "no position on this axis; enlarge it")
                problems.append(
                    f"monitor '{m.name}': flux plane at position_um {m.position_um} snaps to {m.axis}-plane "
                    f"index {kp}, inside the {layers}-layer {label} on axis {m.axis!r}, where the boundary "
                    f"attenuates the flux (NUMERICS.md §12). It needs {layers + 1} <= kp <= {n - layers - 1}: "
                    f"{where}.")
        ramps = [(i, src.source_time) for i, src in enumerate(self.sources) if isinstance(src.source_time, CW)]
        if ramps and (self.run.n_steps is not None or self.run.run_time_s is not None):
            total = (self.run.n_steps * (2.0 * _EPS0 / self._two_eps0_over_dt()) if self.run.n_steps is not None
                     else float(self.run.run_time_s))
            for i, pulse in ramps:
                ramp = float(pulse.ramp_cycles) / float(pulse.freq0_hz)
                if total < ramp:
                    problems.append(
                        f"sources[{i}].source_time: the run ({total:.4g} s) is shorter than the cw ramp "
                        f"({ramp:.4g} s = ramp_cycles / freq0_hz), so the drive never reaches full amplitude "
                        "(NUMERICS.md §5-CW). Lengthen the run (run_time_s or n_steps) or lower ramp_cycles.")
        if problems:
            raise ValueError("this simulation cannot run:\n" + "\n".join(problems))

    @classmethod
    def from_wire_json(cls, text: Union[str, bytes]) -> "Simulation":
        """Strictly-typed ingestion of wire JSON, matching the engine's
        nlohmann typing exactly: JSON int -> float fields is accepted,
        string -> number and float -> int are rejected. Use this (not lax
        ``model_validate_json``) when consuming sim.json files."""
        # Engine parity: nlohmann skips a UTF-8 BOM, so a hand-edited
        # (Windows-authored) sim.json the engine runs must load here too.
        if isinstance(text, bytes):
            if text.startswith(b"\xef\xbb\xbf"):
                text = text[3:]
        elif text.startswith("\ufeff"):
            text = text[1:]
        # context wire_ingest: do NOT apply the D2 construction-time subpixel
        # default to a parsed document — absent means the engine default (off),
        # so older docs round-trip byte-identically (see _resolve_subpixel_default),
        # except that a subpixel-on document without a method gains the key
        # (see _wire_exclude).
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
        # the same defaults. (subpixel_method is the exception, below.) pml_num_layers entered the wire in schema 1.1.0
        # (default 12); run.shutoff in 1.3.0 (default 1e-5, NUMERICS.md §7).
        exclude: dict = {}
        for _f in _decl.DECLARATIVE_FIELDS:
            exclude[_f] = True     # client-only setup fields (never on the wire)
        exclude["origin_um"] = True
        run_exclude = {"transits"}
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
        # field_precision entered the wire in schema 1.19.0 (default "fp32",
        # NUMERICS.md §23); omit it when unset so earlier-minor parsers accept
        # the document and golden specs round-trip byte-identically.
        if "field_precision" not in self.model_fields_set:
            exclude["field_precision"] = True
        # dft_precision entered the wire in schema 1.20.0 (default "fp64",
        # NUMERICS.md §12.6); omit it when unset so earlier-minor parsers accept
        # the document and golden specs round-trip byte-identically.
        if "dft_precision" not in self.model_fields_set:
            exclude["dft_precision"] = True
        # subpixel_method entered the wire in schema 1.7.0 (NUMERICS.md §16.5);
        # omit it when unset and smoothing is off, so earlier-minor parsers
        # accept the document and golden specs round-trip byte-identically.
        # With smoothing on it is always written: the engine's absent default
        # has differed from the model's (volume against contour), so an
        # omitted key ran an operator other than the one the model reports
        # (beta review INT-01, M8).
        if "subpixel_method" not in self.model_fields_set and not self.subpixel:
            exclude["subpixel_method"] = True
        # symmetry entered the wire in schema 1.11.0 (NUMERICS.md §20); omit when
        # all-zero (the no-symmetry default) so earlier-minor parsers accept the
        # document and golden specs round-trip byte-identically.
        if self.symmetry == (0, 0, 0):
            exclude["symmetry"] = True
        if "shutoff" not in self.run.model_fields_set:
            run_exclude.add("shutoff")
        # A zero guard means the legacy energy-only rule and is omitted so
        # older GPU/cloud parsers receive the same wire as before this key.
        if not self.run.dft_shutoff:
            run_exclude.add("dft_shutoff")
        exclude["run"] = run_exclude
        return exclude or None

    def to_wire_dict(self) -> dict:
        """Canonical JSON-level dict: defaults materialized, unset optionals
        (the unused run_time_s/n_steps key, an unset pml_num_layers) omitted."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True,
                               exclude=self._wire_exclude())

    def to_wire_json(self, indent: int = 2) -> str:
        """Serialize the resolved solver document as JSON with ``indent`` spacing.

        Use the same aliases and exclusions as :meth:`to_wire_dict`. Client
        authoring state, including declared ports and wavelengths, is omitted.
        This is the solver contract, not a complete authoring-state round trip."""
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
             grid=False, unfold=True, **kw):
        """2D analytic cross-section of the scene on a cut plane (exactly one
        of x/y/z, in microns). ``grid=True`` overlays the Yee mesh cell edges
        (the resolution sanity-check). ``unfold`` (the default) mirrors a
        NUMERICS §20 half domain back into the whole device; pass
        ``unfold=False`` for the reduced domain the solver steps. Returns a
        matplotlib ``Axes``. See :func:`photonhub.viz.plot`."""
        from ..viz import plot as _plot
        return _plot(self, x=x, y=y, z=z, ax=ax, legend=legend, grid=grid,
                     unfold=unfold, **kw)

    def plot_eps(self, *args, **kwargs):
        """Deprecated alias for :meth:`plot_index` (renamed 2026-09)."""
        import warnings

        warnings.warn(
            "Simulation.plot_eps was renamed to plot_index; the old name will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=caller_stacklevel(),
        )
        return self.plot_index(*args, **kwargs)

    def plot_index(self, x=None, y=None, z=None, *, ax=None, cmap=None,
                 grid=False, unfold=True, **kw):
        """Draw a heatmap of permittivity sampled on the mesh on a cut plane.

        By default, follow ``Simulation.subpixel``: use the §16 volume-fraction
        average when smoothing is enabled and the §9 hard point sample otherwise.
        ``grid=True`` overlays the cell edges.
        ``unfold`` (the default) mirrors a NUMERICS §20 half domain back into
        the whole device; pass ``unfold=False`` for the reduced domain the
        solver steps. Returns a matplotlib ``Axes``. See
        :func:`photonhub.viz.plot_index`."""
        from ..viz import plot_index as _plot_eps
        return _plot_eps(self, x=x, y=y, z=z, ax=ax, cmap=cmap, grid=grid,
                         unfold=unfold, **kw)

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
        :func:`photonhub.cost.quote`. Cell count, dt and step count
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
        return quote(self, **kwargs)
