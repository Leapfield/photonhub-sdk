"""Geometry registry — maps a :class:`~photonhub.replicate.spec.Device` ``kind``
to a parametric builder that emits a :class:`~photonhub.library.Component` (the
device geometry at the origin, with its ports).

This is the "geometry regeneration from parameters" seam: most papers publish a
parameter table and figures, not a GDS, so the device shape is reconstructed
analytically from the spec. Register a new ``kind`` with :func:`register` (or the
``@geometry_builder`` decorator) to teach the workflow a new device family.

A builder has signature ``builder(params, *, medium, thickness_um) -> Component``.
Builders are position-agnostic (origin-centered); :mod:`photonhub.replicate.build`
translates the component into the corner-origin simulation domain.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Tuple

from ..components.structures import Box, Cylinder, Medium, Structure
from ..library import Component, Port, bragg_grating, cosine_taper_crossing, y_branch

__all__ = [
    "register",
    "geometry_builder",
    "build_geometry",
    "REGISTRY",
    "barwicz_ring_centers",
    "nanobeam_hole_layout",
]

Builder = Callable[..., Component]

REGISTRY: Dict[str, Builder] = {}


def register(kind: str, builder: Builder) -> None:
    """Register a geometry ``builder`` under ``kind`` (the spec's
    ``device.kind``)."""
    REGISTRY[kind] = builder


def geometry_builder(kind: str) -> Callable[[Builder], Builder]:
    """Decorator form of :func:`register`."""

    def deco(fn: Builder) -> Builder:
        register(kind, fn)
        return fn

    return deco


def build_geometry(
    kind: str,
    params: Mapping[str, Any],
    *,
    medium: Medium,
    thickness_um: float,
    center_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Component:
    """Dispatch to the registered builder for ``kind`` and return its
    :class:`Component`, placed with its center at ``center_um`` (origin by
    default)."""
    try:
        builder = REGISTRY[kind]
    except KeyError:
        raise KeyError(
            f"unknown device kind {kind!r}; registered: {sorted(REGISTRY)}"
        ) from None
    return builder(params, medium=medium, thickness_um=thickness_um, center_um=center_um)


def barwicz_ring_centers(
    *,
    outer_radius_um: float,
    ring_ring_gap_um: float,
    chain_angle_deg: float = 45.0,
    center_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[float, float, float], ...]:
    """Centers of the three series-coupled rings in Barwicz et al. (2004).

    Figure 2 places the equal-radius rings on a diagonal chain.  The paper gives
    the *outer* radius and the edge-to-edge ring gap, so the centre spacing is
    exactly ``2 * outer_radius_um + ring_ring_gap_um``.  The 45-degree chain
    orientation is read from the figure rather than tabulated; callers may
    override it explicitly when studying that reconstruction uncertainty.
    """
    r = float(outer_radius_um)
    gap = float(ring_ring_gap_um)
    if r <= 0.0:
        raise ValueError("outer_radius_um must be positive")
    if gap < 0.0:
        raise ValueError("ring_ring_gap_um must be non-negative")
    theta = math.radians(float(chain_angle_deg))
    ux, uy = math.cos(theta), math.sin(theta)
    spacing = 2.0 * r + gap
    cx, cy, cz = center_um
    return tuple(
        (cx + k * spacing * ux, cy + k * spacing * uy, cz)
        for k in (-1.0, 0.0, 1.0)
    )


@geometry_builder("barwicz_three_ring_add_drop")
def _barwicz_three_ring_add_drop(
    params: Mapping[str, Any],
    *,
    medium: Medium,
    thickness_um: float,
    center_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Component:
    """Three-ring SiN add/drop filter reconstructed from Barwicz et al.

    Published dimensions are ``outer_radius_um``, ``ring_bus_gap_um`` and
    ``ring_ring_gap_um``.  ``ring_width_um`` defaults to the paper's 1.05 um
    width for "most waveguides"; the exact ring and bus widths used in Fig. 3
    were not separately tabulated, so ``bus_width_um`` is deliberately exposed
    for the paper's 0.65/0.85/1.05 um ambiguity sweep.

    The two buses run along y.  Their end ports are named ``bus_in``/``through``
    on the left bus and ``drop``/``add`` on the right bus, matching Fig. 2.
    This builder emits the patterned SiN only; the asymmetric air/SiO2 stack and
    measured 126 nm oxide over-etch pedestal are assembled by the dedicated
    resonator runner.
    """
    outer = float(params["outer_radius_um"])
    ring_w = float(params.get("ring_width_um", 1.050))
    bus_w = float(params.get("bus_width_um", ring_w))
    gap_bus = float(params["ring_bus_gap_um"])
    gap_ring = float(params["ring_ring_gap_um"])
    angle = float(params.get("chain_angle_deg", 45.0))
    if ring_w <= 0.0 or ring_w >= outer:
        raise ValueError("ring_width_um must satisfy 0 < width < outer radius")
    if bus_w <= 0.0:
        raise ValueError("bus_width_um must be positive")
    if gap_bus < 0.0:
        raise ValueError("ring_bus_gap_um must be non-negative")

    centers = barwicz_ring_centers(
        outer_radius_um=outer,
        ring_ring_gap_um=gap_ring,
        chain_angle_deg=angle,
        center_um=center_um,
    )
    rings = tuple(
        Structure(
            geometry=Cylinder(
                axis="z",
                center_um=c,
                radius_um=outer,
                inner_radius_um=outer - ring_w,
                length_um=thickness_um,
            ),
            medium=medium,
        )
        for c in centers
    )

    left_x = centers[0][0] - outer - gap_bus - 0.5 * bus_w
    right_x = centers[-1][0] + outer + gap_bus + 0.5 * bus_w
    default_half_y = abs(centers[-1][1] - center_um[1]) + outer + 2.0
    bus_length = float(params.get("bus_length_um", 2.0 * default_half_y))
    if bus_length <= 0.0:
        raise ValueError("bus_length_um must be positive")
    _, cy, cz = center_um
    buses = (
        Structure(
            geometry=Box(
                center_um=(left_x, cy, cz),
                size_um=(bus_w, bus_length, thickness_um),
            ),
            medium=medium,
        ),
        Structure(
            geometry=Box(
                center_um=(right_x, cy, cz),
                size_um=(bus_w, bus_length, thickness_um),
            ),
            medium=medium,
        ),
    )
    half = 0.5 * bus_length
    ports = (
        Port("bus_in", (left_x, cy - half, cz), "y", bus_w),
        Port("through", (left_x, cy + half, cz), "y", bus_w),
        Port("drop", (right_x, cy - half, cz), "y", bus_w),
        Port("add", (right_x, cy + half, cz), "y", bus_w),
    )
    return Component(structures=rings + buses, ports=ports)


@geometry_builder("cosine_taper_crossing")
def _cosine_taper_crossing(
    params: Mapping[str, Any],
    *,
    medium: Medium,
    thickness_um: float,
    center_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Component:
    """Beam-shaped ultra-compact waveguide crossing (Chandran et al., Opt. Lett.
    45, 6230 (2020)). Params: ``wg_width_um`` (W_in), ``junction_width_um``
    (W_out), ``peak_width_um`` (W_m), ``taper_length_um`` (L_t),
    ``arm_length_um`` (optional ``n_points``)."""
    return cosine_taper_crossing(
        wg_width_um=float(params["wg_width_um"]),
        junction_width_um=float(params["junction_width_um"]),
        peak_width_um=float(params["peak_width_um"]),
        taper_length_um=float(params["taper_length_um"]),
        arm_length_um=float(params["arm_length_um"]),
        n_points=int(params.get("n_points", 48)),
        thickness_um=thickness_um,
        medium=medium,
        center_um=center_um,
    )


def nanobeam_hole_layout(
    *,
    period_um: float,
    beam_width_um: float,
    n_segments_per_side: int,
    ff_center: float = 0.15,
    ff_end: float = 0.0,
    cavity_length_um: float = 0.0,
) -> Tuple[List[Tuple[float, float]], float]:
    """Deterministic "modulated Bragg mirror" hole layout (Quan, Deotare & Loncar,
    *Appl. Phys. Lett.* **96**, 203102 (2010), arXiv:1002.1319).

    The period ``a`` is held CONSTANT; the filling fraction ``FF = hole_area /
    (a*w)`` decreases LINEARLY from ``ff_center`` at the innermost segment to
    ``ff_end`` at the outermost, on each side, symmetric about ``x = 0``. A linear
    FF ramp gives a linear rise in mirror strength away from the centre, hence a
    Gaussian field envelope — the recipe for a radiation-loss-minimised ultra-high
    Q. ``cavity_length_um`` (``L`` in the paper; the certified design uses ``L=0``)
    is the extra dielectric spacer inserted at the centre; the two innermost holes
    sit ``a`` apart across the centre when ``L=0``.

    Holes are reconstructed as circles area-matched to ``FF`` (the paper specifies
    FF as an area ratio, not a hole shape): ``r = sqrt(FF * a * w / pi)``. A
    segment with ``FF=0`` contributes no hole (it is plain feeding waveguide).

    Returns ``(holes, mirror_span_um)`` where ``holes`` is a list of
    ``(x_offset_um, radius_um)`` relative to the cavity centre and ``mirror_span``
    is the full length of the modulated-mirror region (feeding-waveguide stubs are
    added by the caller)."""
    a = float(period_um)
    w = float(beam_width_um)
    n = int(n_segments_per_side)
    if n < 1:
        raise ValueError(f"n_segments_per_side must be >= 1; got {n}")
    if a <= 0 or w <= 0:
        raise ValueError("period_um and beam_width_um must be positive")
    cell_area = a * w
    half_gap = 0.5 * float(cavity_length_um)
    holes: List[Tuple[float, float]] = []
    for m in range(1, n + 1):
        frac = 0.0 if n == 1 else (m - 1) / (n - 1)
        ff = ff_center + (ff_end - ff_center) * frac
        radius = math.sqrt(max(ff, 0.0) * cell_area / math.pi)
        x = half_gap + (m - 0.5) * a
        if radius > 0.0:
            holes.append((+x, radius))
            holes.append((-x, radius))
    mirror_span = 2.0 * (half_gap + n * a)
    return holes, mirror_span


@geometry_builder("nanobeam_modulated_cavity")
def _nanobeam_modulated_cavity(
    params: Mapping[str, Any],
    *,
    medium: Medium,
    thickness_um: float,
    center_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Component:
    """1D photonic-crystal nanobeam cavity with a modulated Bragg mirror (Quan,
    Deotare & Loncar 2010). A ``Si`` beam (``medium``) of width ``beam_width_um``
    and height ``thickness_um`` running along x, with silica-backfilled holes
    (index ``clad_index``, default 1.45) punched on a constant-period lattice
    whose filling fraction ramps ``ff_center`` -> ``ff_end`` per
    :func:`nanobeam_hole_layout`. Feeding-waveguide stubs of ``feed_stub_um``
    extend the beam past the outermost segment. Ports ``x-`` / ``x+`` attach to
    those feed stubs."""
    a = float(params["period_um"])
    w = float(params["beam_width_um"])
    n = int(params["n_segments_per_side"])
    ff_center = float(params.get("ff_center", 0.15))
    ff_end = float(params.get("ff_end", 0.0))
    cavity_length_um = float(params.get("cavity_length_um", 0.0))
    feed_stub_um = float(params.get("feed_stub_um", 5.0 * a))
    clad_index = float(params.get("clad_index", 1.45))

    holes, mirror_span = nanobeam_hole_layout(
        period_um=a,
        beam_width_um=w,
        n_segments_per_side=n,
        ff_center=ff_center,
        ff_end=ff_end,
        cavity_length_um=cavity_length_um,
    )
    beam_len = mirror_span + 2.0 * feed_stub_um
    cx, cy, cz = center_um
    beam = Structure(
        geometry=Box(center_um=(cx, cy, cz), size_um=(beam_len, w, thickness_um)),
        medium=medium,
    )
    hole_medium = Medium(permittivity=clad_index ** 2)
    hole_structs = tuple(
        Structure(
            geometry=Cylinder(
                axis="z", center_um=(cx + hx, cy, cz),
                radius_um=r, length_um=thickness_um,
            ),
            medium=hole_medium,
        )
        for hx, r in holes
    )
    half = beam_len / 2.0
    ports = (
        Port("x-", (cx - half, cy, cz), "x", w),
        Port("x+", (cx + half, cy, cz), "x", w),
    )
    return Component(structures=(beam,) + hole_structs, ports=ports)


@geometry_builder("metalens_meta_atom")
def _metalens_meta_atom(
    params: Mapping[str, Any],
    *,
    medium: Medium,
    thickness_um: float,
    center_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Component:
    """One meta-atom of a dielectric metasurface: a circular high-index nanopillar
    (``medium``, e.g. TiO2) of radius ``radius_um`` and height ``thickness_um``,
    standing on the substrate top, centred on a square lattice cell (Liang et al.,
    *Nanomaterials* 8, 288 (2018)). The pillar's ``center_um[2]`` should be the
    pillar mid-height (substrate_top + H/2). The substrate/air stack and the
    periodic lattice are supplied by the driver, not this builder — it emits only
    the pillar. A metasurface unit cell has no waveguide ports (the readout is a
    plane-wave transmission/phase), so ``ports`` is empty."""
    radius_um = float(params["radius_um"])
    cx, cy, cz = center_um
    pillar = Structure(
        geometry=Cylinder(axis="z", center_um=(cx, cy, cz),
                          radius_um=radius_um, length_um=thickness_um),
        medium=medium,
    )
    return Component(structures=(pillar,), ports=())


@geometry_builder("bragg_grating")
def _bragg_grating(
    params: Mapping[str, Any],
    *,
    medium: Medium,
    thickness_um: float,
    center_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Component:
    """Sidewall-corrugated waveguide Bragg grating (Wang et al., Opt. Express 20,
    15547 (2012)). Params: ``wg_width_um``, ``corrugation_um`` (per-side tooth
    amplitude), ``period_um``, ``n_periods``, ``arm_length_um``; optional
    ``duty``."""
    return bragg_grating(
        wg_width_um=float(params["wg_width_um"]),
        corrugation_um=float(params["corrugation_um"]),
        period_um=float(params["period_um"]),
        n_periods=int(params["n_periods"]),
        arm_length_um=float(params["arm_length_um"]),
        duty=float(params.get("duty", 0.5)),
        thickness_um=thickness_um,
        medium=medium,
        center_um=center_um,
    )


@geometry_builder("y_branch")
def _y_branch(
    params: Mapping[str, Any],
    *,
    medium: Medium,
    thickness_um: float,
    center_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Component:
    """Asymmetric S-curve Y-junction splitter with an arbitrary power-splitting
    ratio (Lin & Shi, Opt. Express 27, 14338 (2019)). The swept knob is ``r1``
    (``r_top1_um``); the paired top arc ``r2`` and the top straight section ``l``
    are DERIVED here from the paper's Eq. (2) and Eq. (3):

        a = (w - w0/2)/4 ,  b = (w - w0 - D/2)/2
        r2 = (r1 - a)/(R1 - a) * (R2 - b) + b
        l  = R2*sin(acos((R2-b)/R2)) - r2*sin(acos((r2-b)/r2))

    Params: ``wg_width_um`` (w0), ``splitting_length_um`` (L),
    ``region_halfwidth_um`` (w), ``gap_um`` (D), ``r_top1_um`` (r1),
    ``r_bottom1_um`` (R1), ``r_bottom2_um`` (R2), ``arm_length_um``; optional
    ``n_points``, ``solid_junction`` (the paper's widened boundary-defined
    junction), and an explicit ``r_top2_um`` override (else Eq. 2). Generic
    callers may override ``branch_tip_frac`` / ``tip_gap_um``; the paper spec
    leaves both unset so the branch point is derived from the top r2 curve and
    remains sharp as published."""
    import math

    w0 = float(params["wg_width_um"])
    w = float(params["region_halfwidth_um"])
    D = float(params["gap_um"])
    r1 = float(params["r_top1_um"])
    R1 = float(params["r_bottom1_um"])
    R2 = float(params["r_bottom2_um"])
    a = (w - w0 / 2.0) / 4.0
    b = (w - w0 - D / 2.0) / 2.0
    r2 = float(params.get("r_top2_um", (r1 - a) / (R1 - a) * (R2 - b) + b))
    # Eq. (3): straight section on the top arm (0 in the symmetric r1=R1 limit).
    l_top = (R2 * math.sin(math.acos((R2 - b) / R2))
             - r2 * math.sin(math.acos((r2 - b) / r2)))
    return y_branch(
        wg_width_um=w0,
        splitting_length_um=float(params["splitting_length_um"]),
        region_halfwidth_um=w,
        gap_um=D,
        arm_length_um=float(params["arm_length_um"]),
        top_radii=(r1, r2),
        bot_radii=(R1, R2),
        top_straight_um=max(0.0, l_top),
        bot_straight_um=0.0,
        output_offset_final_um=float(params.get("output_offset_final_um", (w0 + D) / 2.0)),
        fanout_length_um=float(params.get("fanout_length_um", 0.0)),
        solid_junction=bool(params.get("solid_junction", False)),
        branch_tip_frac=(float(params["branch_tip_frac"])
                         if "branch_tip_frac" in params else None),
        tip_gap_um=float(params.get("tip_gap_um", 0.0)),
        n_points=int(params.get("n_points", 60)),
        thickness_um=thickness_um,
        medium=medium,
        center_um=center_um,
    )
