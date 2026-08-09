"""Grid specifications (NUMERICS.md section 1, section 15)."""

import math
from typing import (
    TYPE_CHECKING,
    Annotated,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydantic import Field, model_validator

from .base import FrozenModel, PositiveUm
from .structures import GeometryType

if TYPE_CHECKING:  # avoid an import cycle (structures imports nothing from here,
    # but simulation imports grid, and we only need these for type hints).
    from .structures import Structure

# NUMERICS.md section 15.10: the per-axis max/min primary-spacing guard. One
# pinned numeric threshold so the CPU golden, GPU, and this validator agree on
# exactly which graded specs are legal (a divergent accept/reject set would
# itself be an equivalence failure).
GRADED_RATIO_GUARD = 10.0


def realized_cells(length_um: float, dl_um: float) -> int:
    """NUMERICS.md section 1 cell-count rule, shared by every client-side
    consumer: ``n = max(4, round(L/dl))`` with round-half-AWAY-FROM-ZERO —
    NOT Python's built-in banker's rounding, which would disagree with the
    engine's ``std::llround`` at exact halves."""
    x = length_um / dl_um
    n = math.floor(x)
    if x - n >= 0.5:
        n += 1
    return max(4, n)


def snapped_plane_index(position_um: float, dl_um: float) -> int:
    """Nearest grid-plane index ``round(position / dl)``, implemented as
    round-half-UP (floor + 0.5 test). The engine snaps with ``std::llround``
    (half away from zero), which differs only at NEGATIVE exact halves
    (-2.5 -> here -2, llround -3) — positions there are already outside the
    domain and both roundings land at ``kp < 1``, which the caller rejects,
    so the divergence is unobservable. Used by the best-effort flux-plane
    bounds check; ``phsolver validate`` remains authoritative at exact
    halves."""
    x = position_um / dl_um
    n = math.floor(x)
    if x - n >= 0.5:
        n += 1
    return n


def graded_primary_spacings(coords_um: Tuple[float, ...]) -> list[float]:
    """NUMERICS.md section 15.1/15.2 primary cell spacings of a graded axis:
    dq[i] = q[i+1]-q[i] for the interior, with the last cell replicating the
    final spacing (the section-15.1 replicate-last closing node). Length n."""
    n = len(coords_um)
    dq = [coords_um[i + 1] - coords_um[i] for i in range(n - 1)]
    dq.append(dq[-1])  # replicate-last closing
    return dq


def snap_mixed_plane(sim, axis_index: int, position_um: float):
    """Snap a mixed-Yee-offset plane (mode source/monitor, NUMERICS section 12)
    on ANY axis and return ``(snapped_position_um, local_dl_um)``.

    Such a plane must sit at the section-12 quarter point of ONE cell so every
    tangential Yee component (E at the node, H half a cell in) records inside
    the same cell. On a uniform axis that is the historical
    ``(k + 0.25) * dl`` with ``local_dl = dl``. On a GRADED axis it is
    ``q_k + 0.25 * dq_k`` of the cell whose quarter point lies nearest the
    request, and ``local_dl = dq_k`` — the LOCAL spacing, which is also the
    E-to-H plane distance the longitudinal mode-readout de-stagger needs
    (using the grid's base ``dl_um`` there is silently wrong: GradedGridSpec
    carries a ``dl_um`` too, so a truthy check does not detect grading)."""
    q = sim._axis_coords_um(axis_index) if hasattr(sim, "_axis_coords_um") \
        else None
    dl = getattr(sim.grid, "dl_um", None)
    if q is None:                                   # uniform axis — legacy snap
        if not dl:
            return float(position_um), None
        k = round(position_um / dl - 0.25)
        return (k + 0.25) * dl, float(dl)
    dq = graded_primary_spacings(tuple(q))
    quarters = [qi + 0.25 * di for qi, di in zip(q, dq)]
    import bisect

    j = bisect.bisect_left(quarters, position_um)
    best = min((abs(quarters[k] - position_um), k)
               for k in (max(j - 1, 0), min(j, len(quarters) - 1)))[1]
    return float(quarters[best]), float(dq[best])


class UniformGridSpec(FrozenModel):
    """Uniform Cartesian grid, single spacing for all axes.
    n_axis = max(4, round(L_axis / dl)), round half away from zero."""

    type: Literal["uniform"] = "uniform"
    dl_um: float = Field(gt=0)


class GradedAxisCoords(FrozenModel):
    """Per-axis primary-node coordinate arrays (microns) for the graded axes.
    An omitted axis is uniform at the parent ``dl_um``. At least one must be
    present (otherwise use :class:`UniformGridSpec`)."""

    x: Optional[Tuple[float, ...]] = None
    y: Optional[Tuple[float, ...]] = None
    z: Optional[Tuple[float, ...]] = None


class GradedGridSpec(FrozenModel):
    """Per-axis nonuniform (graded) grid — NUMERICS.md section 15. Each listed
    axis carries absolute primary-node coordinates in microns (strictly
    increasing, coords[0] == 0, >= 4 entries); the realized domain length and
    last cell width follow the section-15.1 replicate-last rule. Any axis NOT
    listed in ``coords`` is uniform at ``dl_um`` (identical to
    :class:`UniformGridSpec`). Users supply coordinates manually, or generate
    them from a physical target with the client-side :func:`auto_grid` resolver
    (which returns exactly this model — no new wire member)."""

    type: Literal["graded"] = "graded"
    dl_um: float = Field(gt=0)  # base spacing for any axis not in coords.
    coords: GradedAxisCoords

    @model_validator(mode="after")
    def _check_coords(self) -> "GradedGridSpec":
        present = {a: getattr(self.coords, a)
                   for a in "xyz" if getattr(self.coords, a) is not None}
        if not present:
            raise ValueError(
                "GradedGridSpec.coords lists no axis; use UniformGridSpec for a "
                "fully uniform grid")
        for axis, q in present.items():
            if len(q) < 4:
                raise ValueError(
                    f"grid.coords.{axis}: need >= 4 primary nodes, got {len(q)}")
            if q[0] != 0.0:
                raise ValueError(
                    f"grid.coords.{axis}: must start at 0 (the section-15.1 "
                    f"origin), got {q[0]}")
            if any(q[i + 1] <= q[i] for i in range(len(q) - 1)):
                raise ValueError(
                    f"grid.coords.{axis}: must be strictly increasing")
            dq = graded_primary_spacings(q)
            ratio = max(dq) / min(dq)
            if ratio > GRADED_RATIO_GUARD:
                raise ValueError(
                    f"grid.coords.{axis}: max/min primary-spacing ratio "
                    f"{ratio:.3g} exceeds the {GRADED_RATIO_GUARD:g} guard "
                    "(NUMERICS.md section 15.10)")
        return self


GridSpecType = Annotated[
    Union[UniformGridSpec, GradedGridSpec], Field(discriminator="type")]


# --------------------------------------------------------------------------- #
# Auto-mesh (Track E): a CLIENT-SIDE resolver from a physical target to a valid
# GradedGridSpec. This is a PURE FUNCTION of its inputs (no schema change, no
# iteration/optimizer state) — the roadmap's mesh-freeze requirement ("auto-
# meshing must not make optimization objectives discontinuous between adjoint
# iterations") is met by determinism: identical inputs => byte-identical coords.
# The engine still consumes the ordinary GradedGridSpec it produces (§15.10); we
# add no new wire member, so there is nothing for phsolver to learn.
#
# OPT-IN by design. ``auto_grid`` (and ``Simulation.with_auto_grid``) are never
# the DEFAULT grid: a fresh Simulation still uses whatever GridSpec the caller
# passes (typically UniformGridSpec). This is deliberate — making auto_grid the
# default would change the byte-output (the realized coordinate arrays, hence the
# wire JSON and every golden) of EVERY existing scene, which is golden churn and
# a product decision, not a numerics one.
#
# FOLLOW-UP DECISION (not done in this pass): flipping the default GridSpec to an
# auto-meshed one (Tidy3D ships AutoGrid as the default). If/when taken, it must
# (1) regenerate every golden/example wire file in the same commit (the
# coordinate arrays move), (2) bump the schema minor and gate it behind a
# capability/version so older parsers are not silently handed graded specs, and
# (3) pin a default steps_per_wvl + max_grading + dl_min that reproduce today's
# accuracy on the §14 gates. Until then: opt in explicitly per scene.
# --------------------------------------------------------------------------- #

# Coordinates are rounded to this many decimal microns before being returned, so
# the output is a deterministic, platform-independent (no last-bit fp wobble)
# byte-identical array for identical inputs — the core of mesh-freeze. 1e-7 um =
# 0.1 pm is far finer than any physical mesh, matching the round used in the
# hand-built benchmark mesh (benchmarks/waveguide/waveguide.py).
_AUTO_COORD_DECIMALS = 7

# The grading ratio the roadmap asks for is the CELL-TO-CELL growth limit
# (1.2-1.4). It is intentionally well below GRADED_RATIO_GUARD = 10 (the
# max/min GLOBAL guard the GradedGridSpec validator enforces); keeping the local
# growth small is what keeps the global ratio under the guard too.
_DEFAULT_MAX_GRADING = 1.4

# Speed of light (m/s) — used only to convert a source frequency to a free-space
# wavelength when ``auto_grid`` infers the wavelength from a source.
_C0_M_PER_S = 2.99792458e8


def _wavelength_from_source(source) -> float:
    """Free-space wavelength (microns) implied by a source's time profile —
    ``c / freq0_hz`` of its :class:`GaussianPulse`. Used by ``auto_grid`` when
    no explicit ``wavelength_um`` is given (Tidy3D AutoGrid infers its target
    frequency from the source the same way). Raises if the source has no
    readable ``source_time.freq0_hz``."""
    st = getattr(source, "source_time", None)
    f0 = getattr(st, "freq0_hz", None)
    if f0 is None or f0 <= 0.0:
        raise ValueError(
            "cannot infer wavelength: source has no positive "
            "source_time.freq0_hz; pass wavelength_um explicitly")
    return _C0_M_PER_S / f0 * 1e6  # m -> um


def _geometry_axis_span(geom, axis: int) -> Optional[Tuple[float, float]]:
    """The axis-`axis` bounding span ``[lo, hi]`` (microns, unclamped) that a
    geometry occupies, or ``None`` for an unrecognised type. Every shape is
    bounded by its enclosing box — a safe OVER-estimate of where the fine mesh
    is needed, so the resolver never UNDER-refines around a structure.

    Curved / extruded shapes (Cylinder, PolySlab) report their *bounding box*
    here: the curvature is exactly where subpixel pays off, so the auto-grid
    must still pack fine cells around them even though the closed-form interface
    is faceting-free downstream. The transverse extent of a partial-angle
    cylinder sector is bounded by the full disk; a slanted PolySlab by its
    reference-plane polygon bbox DILATED by the sidewall widening at the
    furthest face (see the polyslab branch — the reference bbox alone is only
    an over-estimate for ``reference_plane="bottom"`` with a positive angle),
    keeping the mesh conservatively fine."""
    gtype = getattr(geom, "type", None)
    if gtype == "box":
        c = geom.center_um[axis]
        half = geom.size_um[axis] / 2.0
        return (c - half, c + half)
    if gtype == "sphere":
        c = geom.center_um[axis]
        return (c - geom.radius_um, c + geom.radius_um)
    if gtype == "cylinder":
        # ``axis`` (the AxisName) is the extrusion axis: that world axis carries
        # the length; the two transverse world axes each carry the radius (the
        # full disk bbox bounds any angular sector — a safe over-estimate).
        ext_axis = "xyz".index(geom.axis)
        c = geom.center_um[axis]
        if axis == ext_axis:
            half = geom.length_um / 2.0
        else:
            half = geom.radius_um
        return (c - half, c + half)
    if gtype == "polyslab":
        # Extruded along ``geom.axis`` between ``slab_bounds_um``; the polygon
        # ``vertices_um`` live in the two transverse axes (u = lower-indexed,
        # v = higher-indexed). A nonzero sidewall_angle narrows the section
        # toward +axis, i.e. the section is WIDEST at the bottom face for
        # angle > 0 and at the top face for angle < 0 — so the reference-plane
        # vertex bbox under-covers the true transverse extent whenever the
        # widest face is not the reference plane ("middle"/"top" with a
        # positive angle, "bottom"/"middle" with a negative one). Pad the bbox
        # by |tan(angle)| times the distance from the reference plane to that
        # widest face (each edge offsets by exactly that along its outward
        # normal, so the bbox grows by exactly the pad) — the same conservative
        # dilation _bounds.py applies; refinement/snap targets then cover the
        # whole structure.
        ext_axis = "xyz".index(geom.axis)
        lo_b, hi_b = geom.slab_bounds_um
        if axis == ext_axis:
            return (lo_b, hi_b)
        transverse = [a for a in range(3) if a != ext_axis]
        # vertices are ordered (u, v) = (transverse[0], transverse[1]).
        comp = 0 if axis == transverse[0] else 1
        coords = [v[comp] for v in geom.vertices_um]
        lo, hi = min(coords), max(coords)
        angle = geom.sidewall_angle
        if angle != 0.0:
            thick = hi_b - lo_b
            # Distance from the reference plane DOWN to the bottom face.
            dist_down = {"bottom": 0.0, "middle": thick / 2.0,
                         "top": thick}[geom.reference_plane]
            far = dist_down if angle > 0.0 else thick - dist_down
            pad = math.tan(abs(angle)) * far
            lo, hi = lo - pad, hi + pad
        return (lo, hi)
    return None  # unknown geometry: ignore rather than guess


def _structure_index(structure: "Structure", wavelength_um: float) -> float:
    """The refractive index that drives refinement for a structure — the
    IN-BAND index at the auto-mesh target wavelength, not the raw
    ``permittivity`` field. For a non-dispersive medium these coincide
    (n = sqrt(eps_r)); for a dispersive (Lorentz) medium ``permittivity`` is
    the high-frequency limit eps_inf (NUMERICS.md §19), which UNDERSTATES the
    in-band index — a Si fit with eps_inf ~ 8 would mesh at n ~ 2.8 instead of
    ~3.48, ~20% under-refinement, breaking the module's never-UNDER-refine
    contract. So we evaluate the §19 single-pole model at
    omega = 2*pi*c/wavelength (:meth:`Medium.permittivity_at_hz` — the ONE
    shared Re-eps evaluator, also what the mode solvers anchor to) and take
    n = sqrt(Re eps), floored at 1 (anomalous dispersion can push Re eps below
    1 near resonance; the mesh target need not chase that below the vacuum
    index)."""
    medium = structure.medium
    try:
        eps = float(medium.permittivity_at_hz(
            _C0_M_PER_S / (wavelength_um * 1e-6)))
    except ValueError:
        # exactly ON an undamped resonance eps diverges; meshing at eps_inf is
        # the sane (pre-existing) fallback for that measure-zero case.
        eps = float(medium.permittivity)
    return math.sqrt(max(eps, 1.0))


def _structure_index_intervals(
    structure: "Structure", axis: int, domain_um: float, wavelength_um: float
) -> Optional[Tuple[float, float, float]]:
    """The axis-`axis` span [lo, hi] (microns, clamped to [0, domain]) that a
    structure occupies, paired with its refractive index (the in-band
    :func:`_structure_index` at the target wavelength), or ``None`` if the
    structure does not intersect the domain on this axis. Boxes give an exact
    span; Sphere / Cylinder / PolySlab are bounded by their enclosing box (a
    safe OVER-estimate of where the fine mesh is needed — never under-refines),
    so CURVED structures — where subpixel matters most — are refined too."""
    geom = structure.geometry
    n = _structure_index(structure, wavelength_um)
    span = _geometry_axis_span(geom, axis)
    if span is None:  # unknown geometry (caller validated the wire types)
        return None
    lo, hi = span
    # Clamp to the domain; reject if it misses the axis extent entirely.
    lo_c, hi_c = max(0.0, lo), min(domain_um, hi)
    if hi_c <= 0.0 or lo_c >= domain_um:
        return None
    return (lo_c, hi_c, n)


def _axis_target_field(
    domain_um: float,
    wavelength_um: float,
    steps_per_wvl: float,
    n_background: float,
    intervals: Sequence[Tuple[float, float, float]],
    refine_pad_um: float,
    dl_min_um: Optional[float] = None,
    enforced: Sequence[Tuple[float, float, float]] = (),
    feature_ceil: bool = True,
):
    """Build the per-axis piecewise target-spacing field dl_target(x) as a sorted
    list of (boundary_position, dl_target_to_the_RIGHT). The local cell size in a
    medium of index n is dl = lambda / (n * steps_per_wvl) — finer in higher
    index, exactly the physics of resolving a wavelength that is shorter by 1/n
    in that medium. Each structure interval is widened by ``refine_pad_um`` on
    both sides so the fine mesh BRACKETS the boundary (the evanescent field and
    the boundary itself want resolution), reproducing the hand-built waveguide
    mesh which kept dl_min out to core_half + pad.

    ``feature_ceil`` (default True) quantizes each structure interval's target so
    an INTEGER number of cells spans the feature at a size <= the raw target:
    ``eff = width / ceil(width / dl_target)`` where ``width`` is the structure's
    own extent on this axis. This is Tidy3D's GradedMesher convention
    (``num = ceil(len/dl)``) and guarantees the realized in-material resolution
    is AT LEAST the requested steps-per-wavelength (never coarser) — where the
    bare marcher could land a finite feature just under the target (a partial
    last cell absorbed by the global close). ``width`` is the UNPADDED span
    because the interfaces snap to the structure edges; the finer ``eff`` is then
    applied across the padded bracket. A feature that exactly fills its target
    (or a full-domain medium) is unchanged (ceil is a no-op there). False
    recovers the exact-target marcher (realized steps ~= requested).

    ``enforced`` carries explicit ``(lo, hi, dl)`` override regions (an
    enforced-refinement box the caller wants meshed at a fixed ``dl`` regardless
    of the local material — Tidy3D's MeshOverrideStructure); these are NOT padded
    (the caller sized them) and compete with the material targets, finest wins.
    ``dl_min_um`` is an absolute lower bound: no segment's target may fall below
    it, so a high-index structure or an over-fine override cannot blow up the
    cell count past the requested floor (Tidy3D AutoGrid ``dl_min``)."""

    def dl_of_index(n: float) -> float:
        return wavelength_um / (n * steps_per_wvl)

    floor = dl_min_um if (dl_min_um is not None and dl_min_um > 0.0) else 0.0

    def clamp(dl: float) -> float:
        return max(dl, floor)

    def ceil_fit(width: float, dl: float) -> float:
        """dl shrunk so an INTEGER number of cells spans `width` (Tidy3D's
        num = ceil(width/dl); eff = width/num <= dl). No-op when off, or for a
        degenerate span. The 1e-6 relative tolerance keeps a width that already
        holds a whole number of cells EXACTLY at dl (no 1-ULP perturbation of
        the marched profile, which the snap is chaotically sensitive to)."""
        if not feature_ceil or width <= 0.0 or dl <= 0.0:
            return dl
        ratio = width / dl
        num = math.ceil(ratio - 1e-6 * max(ratio, 1.0))  # exact fit stays exact
        eff = width / max(num, 1)
        return dl if abs(eff - dl) <= 1e-12 * dl else eff  # bit-stable no-op

    # Sample the requested-target at any x by taking the SMALLEST dl_target of
    # every interval covering x (background otherwise) — the finest medium wins,
    # which is the conservative choice at overlaps.
    bg_dl = clamp(dl_of_index(n_background))
    # Collect breakpoints: padded interval edges plus 0 and domain.
    bps = {0.0, domain_um}
    padded = []
    for lo, hi, n in intervals:
        plo = max(0.0, lo - refine_pad_um)
        phi = min(domain_um, hi + refine_pad_um)
        # ceil-fit the RAW target to the UNPADDED feature width (the span that
        # snaps between the two interfaces), then bracket that eff over the pad.
        eff = ceil_fit(hi - lo, clamp(dl_of_index(n)))
        padded.append((plo, phi, clamp(eff)))
        bps.add(plo)
        bps.add(phi)
    # Enforced override regions: explicit dl, no pad (the caller sized them), but
    # still clamped to the dl_min floor and to the domain. NOT ceil-fit — the
    # override dl is the caller's exact request and its edges carry a hard
    # snap contract; feature-ceil is a material-refinement convention only.
    for lo, hi, dl in enforced:
        elo = max(0.0, lo)
        ehi = min(domain_um, hi)
        if ehi <= elo:
            continue
        padded.append((elo, ehi, clamp(dl)))
        bps.add(elo)
        bps.add(ehi)
    edges = sorted(b for b in bps if 0.0 <= b <= domain_um)

    # For each segment [edges[k], edges[k+1]) pick the finest covering dl.
    field = []  # (segment_start, dl_target)
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        mid = 0.5 * (a + b)
        dl = bg_dl
        for plo, phi, d in padded:
            if plo <= mid < phi and d < dl:
                dl = d
        field.append((a, dl))
    return field


def _target_at(field: Sequence[Tuple[float, float]], x: float) -> float:
    """dl_target(x) from the piecewise field (right-continuous; last value held
    past the final breakpoint). Linear scan — fields have O(#structures) pieces."""
    dl = field[0][1]
    for start, value in field:
        if x >= start:
            dl = value
        else:
            break
    return dl


def _march_axis_coords(
    domain_um: float,
    field: Sequence[Tuple[float, float]],
    max_grading: float,
    min_nodes: int,
    seam_symmetric: bool = False,
) -> Tuple[float, ...]:
    """Integrate the target field into primary-node coordinates (§15.1): start at
    0 and grow cells toward the per-position target spacing, with cell-to-cell
    growth bounded by ``max_grading`` in BOTH directions so the mesh refines
    symmetrically in/around a structure and coarsens smoothly away from it.

    ``seam_symmetric`` additionally grades BOTH walls down to the finer of the
    two seam-adjacent cells so the first and last primary spacings come out
    EXACTLY equal — the §15.2 requirement for a PERIODIC axis, whose
    replicate dual-spacing closure is only correct when the seam spacings
    match (the engine hard-rejects the unequal case).

    Algorithm (a pure function of its inputs — the mesh-freeze contract):

    1. Build the per-position target dl(x) on a fixed FINE sampling grid (step =
       the finest target / 4), so the profile is independent of any prior march.
    2. Smooth that profile with the grading limit in both directions (a cell may
       differ from each neighbour-sample by at most ``max_grading``), giving a
       graded-feasible spacing PROFILE that is fine in the structure and ramps up
       symmetrically — the bracketing the §15.12 gate wants.
    3. Place nodes by integrating dl(x): from each node, the next spacing is the
       smoothed profile sampled at the node, additionally clamped to
       ``prev * max_grading`` (forward grading) so the realized cell-to-cell
       ratio is guaranteed <= max_grading regardless of sampling.
    4. Rescale all cells by one common factor to close on ``domain`` exactly.
       A uniform scale preserves every cell-to-cell ratio, so the grading bound
       (and hence the §15.10 max/min guard) survives the close — and there is no
       tiny "runt" final cell to blow up the global ratio."""
    finest = min(v for _, v in field)
    # Fine sampling grid for the profile (deterministic count).
    h = finest / 4.0
    n_samp = max(min_nodes, int(math.ceil(domain_um / h)) + 1)
    xs = [domain_um * k / (n_samp - 1) for k in range(n_samp)]
    prof = [_target_at(field, x) for x in xs]

    # Smooth the profile with the grading limit in both directions. Iterating to
    # a fixed point is O(n_samp) per sweep; a handful of sweeps converge because
    # each sweep can only lower a value toward a neighbour-bounded ceiling.
    g_samp = max_grading ** (h / max(finest, 1e-30))  # per-sample growth budget
    # Cap the per-sample budget at max_grading: samples are finer than a cell, so
    # neighbouring SAMPLES may differ by at most max_grading (a cell spans >= ~4
    # samples and accumulates more, but the forward clamp in step (3) is the
    # binding guarantee; here we just want a smooth, fine-bracketed profile).
    g_samp = min(g_samp, max_grading)
    for _ in range(n_samp):
        changed = False
        for i in range(1, n_samp):
            cap = prof[i - 1] * g_samp
            if prof[i] > cap:
                prof[i] = cap
                changed = True
        for i in range(n_samp - 2, -1, -1):
            cap = prof[i + 1] * g_samp
            if prof[i] > cap:
                prof[i] = cap
                changed = True
        if not changed:
            break

    def prof_at(x: float) -> float:
        # Nearest-sample lookup on the uniform sampling grid (deterministic).
        k = int(round(x / h)) if h > 0 else 0
        k = min(max(k, 0), n_samp - 1)
        return prof[k]

    # Integrate the smoothed profile into cell spacings, with a hard forward
    # grading clamp so the REALIZED cell-to-cell ratio obeys max_grading exactly.
    finest_prof = min(prof)
    max_cells = max(min_nodes, int(domain_um / finest_prof) + 8) * 4 + 64
    spacings: list[float] = []
    pos = 0.0
    while pos < domain_um - 1e-12 and len(spacings) < max_cells:
        step = prof_at(pos)
        if spacings:
            step = min(step, spacings[-1] * max_grading)
        spacings.append(step)
        pos += step
    if not spacings:
        spacings = [domain_um]

    # Backward grading clamp on the REALIZED spacings: a cell may be at most its
    # right neighbour times max_grading. The forward pass already bounded growth
    # (cell <= left * max_grading); this bounds SHRINK (cell <= right *
    # max_grading) so the mesh decelerates into a downstream fine region within
    # the ratio instead of stepping across it abruptly. Together the two passes
    # guarantee EVERY cell-to-cell ratio (both directions) is <= max_grading.
    for i in range(len(spacings) - 2, -1, -1):
        cap = spacings[i + 1] * max_grading
        if spacings[i] > cap:
            spacings[i] = cap

    # The §15.1 REALIZED length is the closing node q[n] = q[n-1] + dq[n-1],
    # i.e. the stored coords plus ONE replicated final cell. So the stored cells
    # must sum to domain_um leaving the last cell IMPLICIT: we make the last two
    # cells equal (so replicate-last reproduces exactly the intended final cell)
    # and scale the whole profile so the realized length is domain_um.
    if len(spacings) >= 2:
        spacings[-1] = spacings[-2]  # plateau edge so replicate-last is exact
    if seam_symmetric and len(spacings) >= 2:
        # §15.2 periodic-seam symmetry: grade both walls down to the finer of
        # the two seam cells with the g-graded envelope t*g^k. The envelope is
        # g-Lipschitz in log — exactly the bound the two clamp passes above
        # established for the spacings — and min() of two such sequences stays
        # g-Lipschitz, so every cell-to-cell ratio remains <= max_grading.
        # Capping only ever SHRINKS a cell, so structure resolution and the
        # dl_min floor are untouched. The early break is sound because growth
        # is also bounded by max_grading: once the envelope clears one cell it
        # clears every later one. The rescale below multiplies every cell by
        # one common factor, so the exact seam equality installed here (first
        # cell == last stored cell == replicated closing cell) survives it.
        t = min(spacings[0], spacings[-1])
        cap = t
        for i in range(len(spacings)):  # head wall
            if cap >= spacings[i]:
                break
            spacings[i] = cap
            cap *= max_grading
        cap = t
        for i in range(len(spacings) - 2, -1, -1):  # tail wall
            if cap >= spacings[i]:
                break
            spacings[i] = cap
            cap *= max_grading
        spacings[-1] = spacings[-2]  # re-plateau: both closing cells == t
    # Scale so the sum of all cells (the realized length) == domain_um.
    total = sum(spacings)
    scale = domain_um / total
    spacings = [s * scale for s in spacings]

    # Stored primary nodes drop the final cell (it is the §15.1 replicate-last
    # implicit cell): q = [0, s0, s0+s1, ... , sum(s[:-1])]. Then
    # realized = q[-1] + (q[-1]-q[-2]) = sum(s[:-1]) + s[-2] == sum(s) == domain.
    coords = [0.0]
    acc = 0.0
    for s in spacings[:-1]:
        acc += s
        coords.append(acc)
    if len(coords) < 2:  # degenerate single-cell march: fall back to a node pair
        coords = [0.0, domain_um - spacings[-1]]

    # Enforce the >= min_nodes floor (§15.10 needs >= 4 primary nodes): if a very
    # coarse target produced too few, uniformly subdivide into min_nodes cells of
    # width domain/min_nodes and store the first min_nodes nodes (the last is the
    # replicate-last implicit cell), so the realized length stays == domain.
    # Pure & deterministic.
    if len(coords) < min_nodes:
        cell = domain_um / min_nodes
        coords = [cell * k for k in range(min_nodes)]
    return tuple(coords)


def _interface_targets(
    intervals: Sequence[Tuple[float, float, float]],
    enforced: Sequence[Tuple[float, float, float]],
    domain_um: float,
) -> list[float]:
    """The set of axis coordinates a primary node should land on — every
    structure interface (the clamped ``lo``/``hi`` of each material span) plus
    each enforced-override-region edge — that lies STRICTLY interior to
    ``(0, domain)``. The domain edges 0 and ``domain`` are already exact nodes
    (the §15.1 origin and the closing node) so they are never snap targets.

    Returned sorted and de-duplicated at the coordinate-quantization scale, so
    the target list is a deterministic pure function of the geometry (mesh-
    freeze): identical structures -> identical targets regardless of list order.
    Two interfaces closer than the quantization collapse to one (snapping a node
    onto each separately would otherwise demand a sub-quantum cell)."""
    raw: list[float] = []
    for lo, hi, _n in intervals:
        raw.append(lo)
        raw.append(hi)
    for lo, hi, _dl in enforced:
        raw.append(max(0.0, lo))
        raw.append(min(domain_um, hi))
    tol = 10.0 ** (-_AUTO_COORD_DECIMALS)
    out: list[float] = []
    for t in sorted(raw):
        if t <= tol or t >= domain_um - tol:
            continue  # 0 and domain are already exact nodes; skip edge targets
        if out and t - out[-1] <= tol:
            continue  # collapse duplicates / sub-quantum-close interfaces
        out.append(t)
    return out


def _snap_axis_coords(
    coords: Tuple[float, ...],
    targets: Sequence[float],
    domain_um: float,
    max_grading: float,
    dl_min_um: Optional[float],
    preserve_first_spacing: bool = False,
) -> Tuple[float, ...]:
    """Nudge the marched primary nodes so a node coincides with each structure
    interface in ``targets`` (Tidy3D AutoGrid grid-line snapping), WITHOUT
    breaking the graded mesh's invariants.

    Method — a monotone PIECEWISE-LINEAR remap. For each target we pick the
    nearest existing interior node as its *anchor* (one anchor per target,
    assigned left-to-right so anchors stay strictly increasing and never
    collide). The anchors and the fixed breakpoints — node 0 at the origin plus
    the LAST TWO nodes, pinned at their original positions — define segments;
    between consecutive breakpoints every node is mapped by the one affine
    function that carries the segment's old endpoints onto its new endpoints.
    An affine map multiplies every cell in the segment by a single constant
    slope, so cell-to-cell ratios are PRESERVED EXACTLY inside a segment; the
    ratio only changes at an anchor, where it scales by the ratio of the two
    adjoining slopes. Interfaces fall in the FINE (near-uniform, ratio ~1)
    region the marcher already bracketed, so each anchor only needs a small
    move and the slopes stay close to 1.

    Invariant preservation:

    * Determinism — a pure function of (coords, sorted targets): mesh-freeze.
    * Realized length — the §15.1 realized domain length is
      ``q[n-1] + (q[n-1] - q[n-2])`` (replicate-last closing cell), so it
      depends on the LAST TWO nodes. Neither is ever an anchor and both are
      pinned as fixed remap endpoints, preserving the marcher's exact closure
      bit-for-bit; ``grading_ok`` additionally re-checks the realized length
      (1e-9 relative) before accepting any candidate, so a future edit that
      broke the pinning would fail closed (unsnapped), never emit a
      wrong-length mesh.
    * Grading / dl_min — a candidate remap is accepted only if EVERY
      cell-to-cell ratio (both directions) stays within ``max_grading`` and no
      cell falls below the ``dl_min_um`` floor. There is no post-remap clamp
      pass: a violating candidate is rejected wholesale and the greedy retries
      with the worst-offending target dropped, until a subset fits or none is
      left (the input is then returned unchanged). Snapping is best-effort and
      never trumps the invariants.
    * Seam (``preserve_first_spacing``) — on a §15.2 periodic axis the marcher
      made the FIRST cell exactly equal to the last (stored + replicated)
      cell, and the remap must not disturb it: node 1 is then pinned like the
      last two (never an anchor, a fixed breakpoint), so the head segment's
      affine slope starts at node 1 and the seam pair stays bit-identical.

    If ``targets`` is empty the input is returned unchanged."""
    if not targets:
        return coords
    nodes = list(coords)
    n = len(nodes)
    if n < 4:  # too few interior nodes to move without disturbing the floor
        return coords
    tol = 10.0 ** (-_AUTO_COORD_DECIMALS)
    floor = dl_min_um if (dl_min_um is not None and dl_min_um > 0.0) else 0.0
    # §15.1 realized length of the INPUT mesh — the closure the marcher already
    # established. Any accepted candidate must preserve it (see grading_ok).
    realized_in = coords[n - 1] + (coords[n - 1] - coords[n - 2])

    # Assign each target to a distinct anchor index in 1..n-3, nearest free
    # node, left to right. Index 0 is the fixed origin, and the LAST TWO nodes
    # are never anchors: the §15.1 realized length is
    # q[n-1] + (q[n-1] - q[n-2]) (replicate-last closing cell), so moving
    # EITHER of them would change the realized domain length. On a seam-
    # preserving (periodic) axis node 1 is pinned too — moving it would change
    # the first primary spacing and break the §15.2 seam equality.
    first_free = 2 if preserve_first_spacing else 1
    used: set[int] = set()
    # (anchor_index, target_position) pairs, kept sorted by anchor index.
    assigned: list[Tuple[int, float]] = []
    for t in targets:
        best = None
        best_d = None
        for i in range(first_free, n - 2):
            if i in used:
                continue
            d = abs(nodes[i] - t)
            if best_d is None or d < best_d - tol:
                best_d, best = d, i
        if best is None:
            break
        used.add(best)
        assigned.append((best, t))
    assigned.sort(key=lambda p: p[0])

    # Targets may have collided onto the same nearest node region; ensure the
    # assigned (anchor, target) pairs are BOTH strictly increasing so the remap
    # stays monotone. Drop any pair that would invert the order, and any whose
    # target reaches the pinned node n-2 (it cannot be honored without moving
    # a pinned node — the realized-length invariant wins over that snap).
    mono: list[Tuple[int, float]] = []
    for idx, t in assigned:
        if mono and (idx <= mono[-1][0] or t <= mono[-1][1] + tol):
            continue
        if t >= coords[n - 2] - tol:
            continue
        if preserve_first_spacing and t <= coords[1] + tol:
            continue  # cannot be honored without moving the pinned node 1
        mono.append((idx, t))
    assigned = mono
    if not assigned:
        return coords

    def remap(anchor_pairs: list[Tuple[int, float]]) -> list[float]:
        """Piecewise-linear map of the original ``coords`` onto new positions so
        each anchor index lands exactly on its target. Breakpoints are
        (0 -> 0), each (anchor_index -> target), and the PINNED last two nodes
        (n-2 -> coords[n-2], n-1 -> coords[n-1]) so the segment between the
        last anchor and node n-2 rescales while the §15.1 closing pair — hence
        the realized domain length — stays bit-identical. A seam-preserving
        axis pins node 1 as a breakpoint too (the §15.2 first spacing)."""
        head_idx = [0, 1] if preserve_first_spacing else [0]
        head_pos = [0.0, coords[1]] if preserve_first_spacing else [0.0]
        bp_idx = head_idx + [i for i, _ in anchor_pairs] + [n - 2, n - 1]
        bp_pos = (head_pos + [t for _, t in anchor_pairs]
                  + [coords[n - 2], coords[n - 1]])
        out = [0.0] * n
        seg = 0
        for i in range(n):
            while seg < len(bp_idx) - 2 and i > bp_idx[seg + 1]:
                seg += 1
            i0, i1 = bp_idx[seg], bp_idx[seg + 1]
            p0, p1 = bp_pos[seg], bp_pos[seg + 1]
            old0, old1 = coords[i0], coords[i1]
            if old1 - old0 <= 0.0 or i1 == i0:
                out[i] = p0
            else:
                frac = (coords[i] - old0) / (old1 - old0)
                out[i] = p0 + frac * (p1 - p0)
        # Exactly pin the anchors and fixed nodes (kill fp drift -> mesh-freeze
        # and the exact §15.1 closure / §15.2 seam).
        out[0] = 0.0
        if preserve_first_spacing:
            out[1] = coords[1]
        out[n - 2] = coords[n - 2]
        out[n - 1] = coords[n - 1]
        for i, t in anchor_pairs:
            out[i] = t
        return out

    def grading_ok(seq: list[float]) -> bool:
        # Cell-to-cell ratio (both directions) and the dl_min floor, with the
        # same small quantization slack the tests allow.
        dq = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
        if any(d <= 0.0 for d in dq):
            return False
        if floor > 0.0 and min(dq) < floor - tol:
            return False
        for a, b in zip(dq[:-1], dq[1:]):
            if max(a / b, b / a) > max_grading + 1e-6:
                return False
        # §15.1 closure: the realized domain length must be preserved. The
        # pinned endpoints make this automatic; the re-check keeps the greedy
        # fail-closed if the pinning is ever broken by a future edit.
        realized = seq[-1] + (seq[-1] - seq[-2])
        if abs(realized - realized_in) > 1e-9 * max(abs(realized_in), 1.0):
            return False
        return True

    # Greedily snap as many interfaces as stay within the invariants: try the
    # full anchor set, and if it violates grading/floor, drop the worst-offending
    # target and retry. Deterministic (targets are sorted; we drop by index).
    current = assigned
    while current:
        candidate = remap(current)
        if grading_ok(candidate):
            return tuple(candidate)
        # Find the anchor whose move most stretches a neighbouring cell ratio and
        # drop it; recompute. (Deterministic: ties broken by lowest index.)
        worst = None
        worst_ratio = -1.0
        dq = [candidate[i + 1] - candidate[i] for i in range(len(candidate) - 1)]
        anchor_set = {i for i, _ in current}
        for k, (i, _t) in enumerate(current):
            # ratios of the two cells adjoining anchor index i
            local = 1.0
            if 0 < i - 1 < len(dq) and dq[i - 1] > 0 and dq[i] > 0:
                local = max(dq[i - 1] / dq[i], dq[i] / dq[i - 1])
            sub_floor = (floor > 0.0 and
                         ((i - 1 < len(dq) and dq[i - 1] < floor - tol) or
                          (i < len(dq) and dq[i] < floor - tol)))
            score = local + (1e6 if sub_floor else 0.0)
            if score > worst_ratio + tol:
                worst_ratio, worst = score, k
        if worst is None:
            break
        current = current[:worst] + current[worst + 1:]
    return coords


class MeshOverride(FrozenModel):
    """A geometry-based mesh-refinement override — Tidy3D's
    ``MeshOverrideStructure``. Inside ``geometry`` the auto-mesh is forced to the
    target spacing ``dl_um`` REGARDLESS of the local material: a coarse
    background region can be meshed fine, or a structure meshed finer than its
    refractive index alone would call for (a feature edge, a resonant gap, a
    field hotspot an adjoint run wants resolved). It acts through the geometry's
    per-axis bounding span, so a :class:`Box` gives an exact axis-aligned region
    and a :class:`Sphere` / :class:`Cylinder` / :class:`PolySlab` act through
    their enclosing box (a safe OVER-refinement, never under).

    Consumed by :func:`auto_grid` (``mesh_overrides=``) and
    :meth:`photonhub.Simulation.with_mesh_overrides`. Like every auto_grid input it
    is a pure value, so identical overrides produce byte-identical coordinates —
    the mesh stays frozen across adjoint iterations (the §15.10 mesh-freeze
    contract).

    ``dl_um`` is either a single spacing applied on every axis, or a per-axis
    ``(dx, dy, dz)`` tuple in which a ``None`` entry leaves that axis ungoverned
    by this override (Tidy3D's per-component ``dl``). For example
    ``dl_um=(0.02, 0.02, None)`` refines x and y tightly around a waveguide while
    leaving the propagation axis at the background spacing."""

    geometry: GeometryType
    dl_um: Union[
        PositiveUm,
        Tuple[Optional[float], Optional[float], Optional[float]],
    ]

    @model_validator(mode="after")
    def _check_dl(self) -> "MeshOverride":
        if isinstance(self.dl_um, tuple):
            present = [d for d in self.dl_um if d is not None]
            if not present:
                raise ValueError(
                    "MeshOverride.dl_um: at least one axis spacing must be set "
                    "(all three entries are None)")
            for d in present:
                if not (d > 0.0):
                    raise ValueError(
                        f"MeshOverride.dl_um entries must be > 0 (or None for an "
                        f"ungoverned axis), got {d}")
        return self

    def axis_regions(self) -> list[Tuple[str, float, float, float]]:
        """Project this override onto each axis as an enforced-refinement region
        ``(axis_letter, lo_um, hi_um, dl_um)`` — exactly the form
        :func:`auto_grid` consumes via ``refine_regions``. An axis whose target
        ``dl`` is ``None``, or one the geometry does not bound, yields nothing.
        The spans are unclamped (auto_grid clamps them to the domain), so an
        override partly outside the domain still refines the in-domain part."""
        out: list[Tuple[str, float, float, float]] = []
        for a, letter in enumerate("xyz"):
            dl = self.dl_um[a] if isinstance(self.dl_um, tuple) else self.dl_um
            if dl is None:
                continue
            span = _geometry_axis_span(self.geometry, a)
            if span is None:
                continue
            lo, hi = span
            out.append((letter, float(lo), float(hi), float(dl)))
        return out


def auto_grid(
    *,
    size_um: Tuple[float, float, float],
    wavelength_um: Optional[float] = None,
    structures: Iterable["Structure"] = (),
    background_index: float = 1.0,
    steps_per_wvl: float = 20.0,
    max_grading: float = _DEFAULT_MAX_GRADING,
    axes: str = "xyz",
    refine_pad_um: Optional[float] = None,
    min_nodes: int = 4,
    source=None,
    dl_min_um: Optional[float] = None,
    refine_regions: Iterable[Tuple[str, float, float, float]] = (),
    mesh_overrides: Iterable["MeshOverride"] = (),
    snap_interfaces: bool = True,
    periodic_axes: str = "",
    feature_ceil: bool = True,
) -> GradedGridSpec:
    """Auto-mesh resolver (Track E): generate a VALID :class:`GradedGridSpec`
    from a physical target — minimum steps-per-wavelength per medium, a maximum
    cell-to-cell grading ratio, the wavelength, the domain size, and the scene's
    structures (to find the high-index regions / material boundaries where the
    mesh must be fine). Refinement concentrates cells in and around high-index
    structures (local dl = lambda / (n * steps_per_wvl)) and coarsens smoothly
    outward within ``max_grading``.

    This is a PURE FUNCTION (the mesh-freeze contract): identical inputs produce
    BYTE-IDENTICAL coordinate arrays, with no dependence on optimizer/iteration
    state, so an adjoint loop that calls it every iteration sees a continuous,
    non-jittering mesh. No schema change — the result is the ordinary graded spec
    the engine already consumes (NUMERICS.md section 15.10).

    .. warning:: ACCURACY CAVEAT — only structures whose in-band index EXCEEDS
       ``background_index`` drive the mesh. A structure at or BELOW the
       background index (an air trench or under-etched gap in oxide, a
       low-index filler) gets NO refinement and NO interface snapping: its
       material boundary can fall mid-cell at the coarse background spacing.
       This is by design (refinement follows the shorter in-medium wavelength,
       and it keeps "no refining structures" meshes exactly uniform), but if a
       low-index feature's interface is what you are resolving — a narrow
       trench, a slot waveguide's gap — cover it explicitly with
       ``refine_regions=`` / ``mesh_overrides=``, which snap and refine
       regardless of index.

    Parameters
    ----------
    size_um : domain extents (Lx, Ly, Lz), microns.
    wavelength_um : free-space wavelength of interest (use the SHORTEST in a
        band so every frequency is resolved). Pass ``c / freq`` to drive from a
        frequency. If omitted, it is INFERRED from ``source`` (``c /
        source.source_time.freq0_hz``) — Tidy3D AutoGrid drives its target
        frequency from the source the same way. Exactly one of ``wavelength_um``
        / ``source`` must be supplied.
    structures : the simulation's structures. Each is sampled per axis; its
        IN-BAND index sets the local cell size inside (and just around) it —
        n = sqrt(permittivity) for a non-dispersive medium, and for a
        dispersive (Lorentz) one n = sqrt(Re eps(omega)) evaluated at this
        wavelength via the §19 pole model (meshing at eps_inf alone would
        under-refine; see :func:`_structure_index`).
        Box / Sphere / Cylinder / PolySlab are all supported (curved
        and extruded shapes are bounded by their enclosing box — refined, never
        under-refined). Geometries may extend beyond the domain — only the
        in-domain part drives the mesh (NUMERICS.md section 9).
    background_index : refractive index of the background medium
        (= sqrt(background.permittivity)); sets the coarse, far-from-structure
        cell size.
    steps_per_wvl : minimum cells per wavelength IN EACH MEDIUM. A finer target
        (larger value) yields more cells / a smaller minimum spacing.
    max_grading : maximum cell-to-cell spacing growth ratio (the roadmap's
        1.2-1.4). Must be > 1 and <= GRADED_RATIO_GUARD; the GLOBAL max/min
        ratio of the final spec is separately bounded by GRADED_RATIO_GUARD,
        which a small ``max_grading`` keeps well clear of.
    axes : which axes to grade (e.g. ``"xy"`` to leave z uniform). Axes omitted
        here are left out of ``coords`` and stay uniform at the returned
        ``dl_um`` (the background spacing).
    refine_pad_um : how far the fine mesh extends past a structure boundary
        (microns). Defaults to one background cell — enough to bracket the
        boundary and the near evanescent field.
    min_nodes : floor on primary nodes per graded axis (>= 4 per section 15.10).
    source : optional source object to infer ``wavelength_um`` from when it is
        not given (reads ``source.source_time.freq0_hz``).
    dl_min_um : absolute LOWER bound (microns) on the minimum cell spacing —
        Tidy3D AutoGrid's ``dl_min``. No medium target or override may push a
        cell below this, so a very high-index inclusion (or an over-fine
        override) cannot explode the cell count. Must be > 0 if given.
    refine_regions : explicit enforced-refinement boxes, each
        ``(axis_letter, lo_um, hi_um, dl_um)``: force the mesh to ``dl_um`` over
        ``[lo, hi]`` on that axis regardless of the local material (the
        axis-interval form of a mesh override). The finest of {material target,
        override} wins at every point. ``dl_um`` is still clamped to
        ``dl_min_um``.
    mesh_overrides : geometry-based :class:`MeshOverride` objects (Tidy3D's
        ``MeshOverrideStructure``). Each is projected onto every axis it governs
        (its per-axis bounding span at the override's ``dl_um``) and merged with
        ``refine_regions`` — the convenient front-end when the refinement region
        is a real scene geometry rather than hand-written axis intervals.
    snap_interfaces : when True (default), nudge the generated nodes so a primary
        grid line lands EXACTLY on each structure interface coordinate (every
        in-domain box face / curved-shape bbox edge / polyslab boundary) and each
        refine-region edge — Tidy3D AutoGrid grid-line snapping, so a material
        boundary never falls mid-cell. Snapping is a monotone piecewise-linear
        remap that PRESERVES the grading-ratio and ``dl_min`` invariants (a target
        that cannot be reconciled with ``max_grading`` / the floor is abandoned
        rather than emitting an out-of-spec mesh) and stays deterministic
        (mesh-freeze). Set False to recover the pre-snap node positions.
    periodic_axes : axes (e.g. ``"xy"``) whose simulation boundary is PERIODIC.
        A graded periodic axis must close with EQUAL first and last primary
        spacings — the engine's §15.2 replicate dual-spacing closure is only
        correct at the seam when they match, and ``phsolver validate``
        hard-rejects the unequal case — so each listed axis is generated
        seam-symmetrically: both walls grade down to the finer of the two
        seam cells (never coarsening a structure, at most a few extra cells)
        and the seam pair is kept exactly equal through snapping and the
        coordinate quantization. Axes listed here but not in ``axes`` stay
        uniform (trivially seam-equal). :meth:`Simulation.with_auto_grid` /
        ``with_mesh_overrides`` fill this in from ``sim.boundaries``
        automatically; only direct ``auto_grid`` callers pass it by hand.
    feature_ceil : when True (default), quantize each structure's target cell so
        an INTEGER number of cells spans the feature at a size <= the requested
        ``lambda/(n*steps_per_wvl)`` — ``eff = width / ceil(width/dl)``, Tidy3D
        AutoGrid's convention. This guarantees the realized in-material
        resolution is AT LEAST ``steps_per_wvl`` (never coarser), matching
        Tidy3D and removing the run-to-run variance of where the plain marcher
        lands a partial last cell. A feature that already holds a whole number
        of target cells (or a full-domain medium) is unchanged. False recovers
        the exact-target marcher (realized steps ~= requested, occasionally a
        hair under for a finite feature). Costs a few extra cells per graded
        axis; the accuracy is never lower.

    Returns
    -------
    GradedGridSpec whose listed axes carry the generated coordinate arrays and
    whose ``dl_um`` is the background spacing (used for any non-graded axis).
    The returned spec is run through GradedGridSpec validation here, so the
    GRADED_RATIO_GUARD and the section-15.1 invariants (coords[0]=0, strictly
    increasing, >= 4 nodes) are guaranteed before it is handed back.

    Raises
    ------
    ValueError on invalid targets (no/non-positive wavelength, both/neither of
    wavelength & source, non-positive steps, grading <= 1, bad axis letters,
    non-positive ``dl_min_um``, malformed ``refine_regions``) or if a generated
    array somehow fails GradedGridSpec validation.
    """
    if wavelength_um is not None and source is not None:
        raise ValueError(
            "pass exactly one of wavelength_um / source, not both")
    if wavelength_um is None:
        if source is None:
            raise ValueError(
                "wavelength_um is required (or pass source= to infer it)")
        wavelength_um = _wavelength_from_source(source)
    if wavelength_um <= 0:
        raise ValueError(f"wavelength_um must be > 0, got {wavelength_um}")
    if dl_min_um is not None and dl_min_um <= 0.0:
        raise ValueError(f"dl_min_um must be > 0 if given, got {dl_min_um}")
    if steps_per_wvl <= 0:
        raise ValueError(f"steps_per_wvl must be > 0, got {steps_per_wvl}")
    if not (max_grading > 1.0):
        raise ValueError(
            f"max_grading must be > 1 (cell-to-cell growth), got {max_grading}")
    if max_grading > GRADED_RATIO_GUARD:
        # A per-cell growth above the global guard cannot satisfy the spec for
        # any non-trivial coarsening — refuse early with a clear message.
        raise ValueError(
            f"max_grading {max_grading} exceeds GRADED_RATIO_GUARD "
            f"{GRADED_RATIO_GUARD:g}; pick a grading ratio <= the guard")
    if background_index < 1.0:
        raise ValueError(
            f"background_index must be >= 1, got {background_index}")
    bad = [a for a in axes if a not in "xyz"]
    if bad or len(set(axes)) != len(axes):
        raise ValueError(f"axes must be a subset of 'xyz' with no repeats, "
                         f"got {axes!r}")
    bad = [a for a in periodic_axes if a not in "xyz"]
    if bad or len(set(periodic_axes)) != len(periodic_axes):
        raise ValueError(f"periodic_axes must be a subset of 'xyz' with no "
                         f"repeats, got {periodic_axes!r}")
    if min_nodes < 4:
        raise ValueError(f"min_nodes must be >= 4 (section 15.10), got "
                         f"{min_nodes}")

    structures = list(structures)
    # Geometry-based mesh overrides project onto the same axis-interval form as
    # refine_regions; merge them and run the one validation/grouping path below.
    all_regions = list(refine_regions)
    for ov in mesh_overrides:
        all_regions.extend(ov.axis_regions())
    # Group enforced-refinement overrides per axis, validating each tuple.
    overrides: dict[str, list] = {a: [] for a in "xyz"}
    for region in all_regions:
        try:
            ax_letter, lo, hi, dl = region
        except (TypeError, ValueError):
            raise ValueError(
                "each refine_regions entry must be "
                "(axis_letter, lo_um, hi_um, dl_um), got "
                f"{region!r}")
        if ax_letter not in "xyz":
            raise ValueError(
                f"refine_regions axis must be one of 'xyz', got {ax_letter!r}")
        if not (hi > lo):
            raise ValueError(
                f"refine_regions region needs hi > lo, got lo={lo} hi={hi}")
        if not (dl > 0.0):
            raise ValueError(
                f"refine_regions dl_um must be > 0, got {dl}")
        overrides[ax_letter].append((float(lo), float(hi), float(dl)))

    # The base/background spacing — also the dl_um for any non-graded axis. It is
    # itself clamped to the dl_min floor so a coarse background still honours it.
    bg_dl = wavelength_um / (background_index * steps_per_wvl)
    if dl_min_um is not None:
        bg_dl = max(bg_dl, dl_min_um)
    pad = refine_pad_um if refine_pad_um is not None else bg_dl

    coords_kwargs: dict = {}
    for axis_letter in "xyz":
        if axis_letter not in axes:
            continue
        axis = "xyz".index(axis_letter)
        domain = float(size_um[axis])
        intervals = []
        for s in structures:
            iv = _structure_index_intervals(s, axis, domain, wavelength_um)
            # Only intervals from media FINER than background drive refinement;
            # a structure at or below background index needs no extra mesh.
            if iv is not None and iv[2] > background_index:
                intervals.append(iv)
        field = _axis_target_field(
            domain_um=domain, wavelength_um=wavelength_um,
            steps_per_wvl=steps_per_wvl, n_background=background_index,
            intervals=intervals, refine_pad_um=pad,
            dl_min_um=dl_min_um, enforced=overrides[axis_letter],
            feature_ceil=feature_ceil)
        seam = axis_letter in periodic_axes
        raw = _march_axis_coords(domain, field, max_grading, min_nodes,
                                 seam_symmetric=seam)
        if snap_interfaces:
            # Snap to the interfaces that actually drive a fine mesh (the
            # refining structures' spans) and each override-region edge. A
            # structure at/below background index gets no refinement, so its
            # boundary is left alone — the "no refinement -> uniform" contract
            # holds and we only pin grid lines where the fine cells already are.
            targets = _interface_targets(
                intervals, overrides[axis_letter], domain)
            raw = _snap_axis_coords(
                raw, targets, domain, max_grading, dl_min_um,
                preserve_first_spacing=seam)
        # §15.1 exact-closure invariant: the realized length (closing node +
        # replicate-last cell) equals the requested domain length. The marcher
        # closes exactly (one common rescale) and snapping pins the last two
        # nodes, so a violation here is an internal bug, not a user error.
        # Checked BEFORE the coordinate quantization below, whose ±1e-7 um
        # rounding is the only remaining (documented) wobble.
        realized = raw[-1] + (raw[-1] - raw[-2])
        assert abs(realized - domain) <= 1e-9 * max(domain, 1.0), (
            f"auto_grid internal error: axis '{axis_letter}' realized length "
            f"{realized!r} != requested {domain!r}")
        rounded = [round(float(c), _AUTO_COORD_DECIMALS) for c in raw]
        if seam:
            # The per-coordinate quantization just above moves each node by up
            # to half a 1e-7 um quantum INDEPENDENTLY, so the first and last
            # primary spacings — exactly equal in ``raw`` — can come out
            # differing by up to a quantum, far beyond the engine's 1e-12
            # relative seam gate. Re-derive the last two nodes from the
            # already-quantized first spacing so the wire-format seam pair is
            # equal to double precision AND the §15.1 closure is exact:
            # q[-1] + (q[-1] - q[-2]) == realized. The two nodes move by at
            # most a few quanta (0.1 pm), negligible against any physical
            # cell, and the construction stays a pure function of the inputs
            # (mesh-freeze).
            s0 = rounded[1] - rounded[0]
            rounded[-2] = realized - 2.0 * s0
            rounded[-1] = realized - s0
        coords_kwargs[axis_letter] = tuple(rounded)

    spec = GradedGridSpec(
        dl_um=round(float(bg_dl), _AUTO_COORD_DECIMALS),
        coords=GradedAxisCoords(**coords_kwargs))
    return spec
