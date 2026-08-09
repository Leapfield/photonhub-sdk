"""Auto-mesh resolver (Track E) — ``ph.auto_grid``.

The resolver is a CLIENT-SIDE pure function turning a physical target
(steps-per-wavelength per medium + a grading ratio + wavelength + domain +
structures) into a valid GradedGridSpec (NUMERICS.md section 15). These tests
pin the acceptance contract:

  * the output passes GradedGridSpec validation (incl. GRADED_RATIO_GUARD and
    the section-15.1 invariants: coords[0]=0, strictly increasing, >= 4 nodes);
  * a finer target -> more cells and a smaller minimum spacing;
  * refinement actually concentrates cells in / around high-index structures;
  * MESH-FREEZE: identical inputs -> byte-identical coords (determinism), the
    property that keeps an adjoint objective continuous between iterations;
  * a realistic case: refine around a high-index silicon waveguide core.
"""

import math

import pytest

import photonhub as ph
from photonhub import auto_grid
from photonhub.components.grid import (
    GRADED_RATIO_GUARD,
    GradedGridSpec,
    graded_primary_spacings,
)

C0 = 2.99792458e8


def _box(center, size, eps):
    return ph.Structure(
        geometry=ph.Box(center_um=center, size_um=size),
        medium=ph.Medium(permittivity=eps))


def _axis_spacings(spec: GradedGridSpec, axis: str):
    q = getattr(spec.coords, axis)
    return graded_primary_spacings(q)


def _max_cell_to_cell_ratio(spacings):
    """Largest adjacent cell-to-cell growth ratio (both directions)."""
    r = 1.0
    for a, b in zip(spacings[:-1], spacings[1:]):
        r = max(r, a / b, b / a)
    return r


# --------------------------------------------------------------------------- #
# Validity: the resolver returns a spec that already passed GradedGridSpec's
# own validators (the call below would have raised otherwise), and we re-assert
# the section-15.1 invariants explicitly.
# --------------------------------------------------------------------------- #

def test_returns_valid_graded_spec():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                     structures=[core], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="xy")
    assert isinstance(spec, GradedGridSpec)
    for axis in "xy":
        q = getattr(spec.coords, axis)
        assert q[0] == 0.0
        assert len(q) >= 4
        assert all(q[i + 1] > q[i] for i in range(len(q) - 1))
    # z was not requested -> not graded.
    assert spec.coords.z is None


def test_grading_ratio_respected_and_under_guard():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    for grading in (1.2, 1.3, 1.4):
        spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                         structures=[core], background_index=1.444,
                         steps_per_wvl=18.0, max_grading=grading, axes="xy")
        for axis in "xy":
            sp = _axis_spacings(spec, axis)
            ratio = _max_cell_to_cell_ratio(sp)
            # Cell-to-cell growth honors the requested grading (small rounding
            # slack from the 1e-7 um coordinate quantization).
            assert ratio <= grading + 1e-3, (axis, grading, ratio)
            # And the GLOBAL max/min guard the spec enforces holds with margin.
            assert max(sp) / min(sp) <= GRADED_RATIO_GUARD


# --------------------------------------------------------------------------- #
# Finer target -> more cells and a smaller minimum spacing.
# --------------------------------------------------------------------------- #

def test_finer_target_gives_more_cells_and_smaller_min_spacing():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              structures=[core], background_index=1.444,
              max_grading=1.3, axes="xy")
    coarse = auto_grid(steps_per_wvl=10.0, **kw)
    fine = auto_grid(steps_per_wvl=30.0, **kw)
    for axis in "xy":
        qc, qf = getattr(coarse.coords, axis), getattr(fine.coords, axis)
        assert len(qf) > len(qc), axis
        assert min(_axis_spacings(fine, axis)) < min(_axis_spacings(coarse, axis))


# --------------------------------------------------------------------------- #
# Refinement concentrates cells IN / AROUND the high-index structure.
# --------------------------------------------------------------------------- #

def test_cells_concentrate_in_high_index_core():
    # 0.45um silicon (n=3.5) core centered in a 2um cladding (n=1.444) domain.
    core = _box((1.0, 0.5, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    spec = auto_grid(size_um=(2.0, 1.0, 4.0), wavelength_um=1.31,
                     structures=[core], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x")
    qx = spec.coords.x
    # Count nodes inside the core span [1.0 - 0.225, 1.0 + 0.225] vs an
    # equal-width window in the cladding far from the core.
    lo, hi = 1.0 - 0.225, 1.0 + 0.225
    in_core = sum(1 for c in qx if lo <= c <= hi)
    # A cladding window of the same 0.45um width near the edge.
    edge_lo, edge_hi = 0.0, 0.45
    in_edge = sum(1 for c in qx if edge_lo <= c <= edge_hi)
    assert in_core > in_edge, (in_core, in_edge)
    # The finest cell sits inside the core (local dl ~ lambda/(n*steps)).
    sp = _axis_spacings(spec, "x")
    finest_idx = min(range(len(sp)), key=lambda i: sp[i])
    cell_center = 0.5 * (qx[finest_idx] + qx[finest_idx + 1])
    assert lo - 0.1 <= cell_center <= hi + 0.1, cell_center
    # Local dl in the core ~ 1.31/(3.5*20) = 0.0187 um; background ~ 0.045 um.
    expected_core_dl = 1.31 / (3.5 * 20.0)
    assert min(sp) <= expected_core_dl * 1.2


def _incore_min_spacing(spec, axis, lo, hi):
    """Smallest cell whose CENTER lies in [lo, hi] on `axis` (true in-material
    resolution, not a straddling interface/cladding cell)."""
    q = getattr(spec.coords, axis)
    sp = graded_primary_spacings(q)
    incore = [sp[i] for i in range(len(q) - 1)
              if lo <= 0.5 * (q[i] + q[i + 1]) <= hi]
    return min(incore)


@pytest.mark.parametrize("width_um,n", [
    (0.5, 3.4738),      # 500nm core:  500/17.85 = 28.01 -> ceil 29
    (0.18, 3.4738),     # 180nm strip: 180/17.85 = 10.08 -> ceil 11
    (0.22, 3.4738),     # 220nm height:220/17.85 = 12.32 -> ceil 13
])
def test_feature_ceil_guarantees_at_least_requested_resolution(width_um, n):
    # feature_ceil=True (default, Tidy3D's convention): a finite Si feature is
    # meshed with ceil(width/target) cells, so the realized in-material dl is
    # AT MOST lambda/(n*steps) — never coarser than the requested steps/wvl,
    # where the bare marcher can land a finite feature just UNDER the target.
    lam, steps = 1.55, 25.0
    target = lam / (n * steps)
    core = _box((3.0, 2.0, 1.0), (width_um, 0.22, 10.0), eps=n ** 2)
    dom = (6.0, 4.0, 2.0)
    lo, hi = 3.0 - width_um / 2, 3.0 + width_um / 2
    kw = dict(size_um=dom, wavelength_um=lam, structures=[core],
              background_index=1.444, steps_per_wvl=steps, max_grading=1.4,
              axes="x")
    dl_ceil = _incore_min_spacing(auto_grid(**kw, feature_ceil=True), "x", lo, hi)
    dl_march = _incore_min_spacing(auto_grid(**kw, feature_ceil=False), "x", lo, hi)
    # ceil guarantees realized dl <= target (steps >= requested), never coarser
    # than the marcher; allow a hair of coordinate-quantization slack (1e-7 um).
    assert dl_ceil <= target + 1e-7, (dl_ceil, target)
    assert dl_ceil <= dl_march + 1e-7, (dl_ceil, dl_march)
    # the ceil cell equals width / ceil(width/target) exactly (Tidy3D's number)
    num = math.ceil(width_um / target - 1e-6)
    assert dl_ceil == pytest.approx(width_um / num, abs=2e-7)


def test_feature_ceil_strictly_finer_when_marcher_underresolves():
    # A 500nm Si core: 500/17.85 = 28.01, so the marcher lands 28 cells (dl
    # slightly OVER target, steps < 25) while ceil forces 29 (dl <= target).
    lam, steps, n = 1.55, 25.0, 3.4738
    target = lam / (n * steps)
    core = _box((3.0, 2.0, 1.0), (0.5, 0.22, 10.0), eps=n ** 2)
    kw = dict(size_um=(6.0, 4.0, 2.0), wavelength_um=lam, structures=[core],
              background_index=1.444, steps_per_wvl=steps, max_grading=1.4,
              axes="x")
    dl_ceil = _incore_min_spacing(auto_grid(**kw, feature_ceil=True), "x", 2.75, 3.25)
    dl_march = _incore_min_spacing(auto_grid(**kw, feature_ceil=False), "x", 2.75, 3.25)
    assert dl_march > target        # marcher under-resolves (steps < 25)
    assert dl_ceil <= target        # ceil restores steps >= 25
    assert dl_ceil < dl_march


def test_feature_ceil_noop_when_width_is_a_whole_number_of_cells():
    # A feature (or full-domain medium) whose width already holds an integer
    # number of target cells is bit-identical with ceil on vs off.
    lam, steps, n = 1.55, 25.0, 2.0
    target = lam / (n * steps)          # 31.0 nm
    width = 40 * target                 # exactly 40 cells wide
    core = _box((3.0, 2.0, 1.0), (width, 0.22, 10.0), eps=n ** 2)
    kw = dict(size_um=(6.0, 4.0, 2.0), wavelength_um=lam, structures=[core],
              background_index=1.444, steps_per_wvl=steps, max_grading=1.4,
              axes="x")
    a = auto_grid(**kw, feature_ceil=True).coords.x
    b = auto_grid(**kw, feature_ceil=False).coords.x
    assert a == b


def test_no_high_index_structure_gives_essentially_uniform():
    # Background-only domain (or structures at/below background index): no
    # refinement, so the mesh is ~uniform at the background spacing.
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    # Uniform up to the 1e-7 um coordinate quantization (no refinement at all).
    assert _max_cell_to_cell_ratio(sp) == pytest.approx(1.0, abs=1e-4)
    # spacing ~ lambda/(n*steps) = 1/20 = 0.05 um.
    assert min(sp) == pytest.approx(0.05, rel=0.05)


def test_low_index_structure_does_not_refine():
    # A structure with index <= background must not pull in fine cells.
    lowq = _box((1.0, 1.0, 1.0), (0.5, 0.5, 0.5), eps=1.0)  # n=1 < bg 1.444
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[lowq], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    # Uniform up to the 1e-7 um coordinate quantization (no refinement at all).
    assert _max_cell_to_cell_ratio(sp) == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# MESH-FREEZE: determinism. Identical inputs -> byte-identical coords. This is
# the property that keeps an adjoint objective continuous across iterations.
# --------------------------------------------------------------------------- #

def test_mesh_freeze_byte_identical_for_identical_inputs():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              structures=[core], background_index=1.444,
              steps_per_wvl=22.0, max_grading=1.27, axes="xy")
    a = auto_grid(**kw)
    b = auto_grid(**kw)
    # Exact equality of the coordinate tuples (and the JSON wire form).
    assert a.coords.x == b.coords.x
    assert a.coords.y == b.coords.y
    assert a.dl_um == b.dl_um
    assert a.model_dump_json() == b.model_dump_json()


def test_mesh_freeze_independent_of_structure_list_order():
    # The mesh is a pure function of the SET of (span, index) constraints, so
    # reordering structures (an optimizer might) must not move a single node.
    s1 = _box((0.6, 1.0, 1.0), (0.3, 0.3, 4.0), eps=12.25)
    s2 = _box((1.4, 1.0, 1.0), (0.3, 0.3, 4.0), eps=6.0)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              background_index=1.444, steps_per_wvl=18.0,
              max_grading=1.3, axes="x")
    a = auto_grid(structures=[s1, s2], **kw)
    b = auto_grid(structures=[s2, s1], **kw)
    assert a.coords.x == b.coords.x


def test_mesh_freeze_small_input_change_small_output_change():
    # Determinism + continuity: a tiny wavelength nudge must NOT produce a wild
    # mesh jump (the discontinuity the roadmap warns about). We just assert the
    # node count is stable and the coords move only slightly.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), structures=[core],
              background_index=1.444, steps_per_wvl=20.0,
              max_grading=1.3, axes="x")
    a = auto_grid(wavelength_um=1.310, **kw)
    b = auto_grid(wavelength_um=1.311, **kw)
    assert len(a.coords.x) == len(b.coords.x)
    drift = max(abs(p - q) for p, q in zip(a.coords.x, b.coords.x))
    assert drift < 0.01  # < one fine cell


# --------------------------------------------------------------------------- #
# Realistic case: SOI strip waveguide cross-section (matches the hand-built
# mesh in benchmarks/waveguide/waveguide.py — refine the core, coarsen cladding).
# --------------------------------------------------------------------------- #

def test_realistic_soi_waveguide_cross_section():
    LX, LY, LZ = 1.6, 1.4, 5.0
    core = _box((LX / 2, LY / 2, LZ / 2), (0.45, 0.22, LZ * 2), eps=3.5 ** 2)
    spec = auto_grid(
        size_um=(LX, LY, LZ), wavelength_um=1.31, structures=[core],
        background_index=1.444, steps_per_wvl=20.0, max_grading=1.4, axes="xy")
    # Drop it straight into a Simulation to prove the produced spec is usable.
    src = ph.PointDipole(
        center_um=(LX / 2, LY / 2, 0.7), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=C0 / 1.31e-6,
                                      fwidth_hz=0.12 * C0 / 1.31e-6))
    sim = ph.Simulation(
        size_um=(LX, LY, LZ), grid=spec, run={"n_steps": 100},
        background=ph.Background(permittivity=1.444 ** 2), structures=(core,),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=10, sources=[src])
    # Graded x and y realized lengths come from the closing-node rule, not L.
    rx, ry, _ = sim._realized_um()
    assert math.isclose(rx, LX, abs_tol=0.02)
    assert math.isclose(ry, LY, abs_tol=0.02)
    # The core (~0.22um tall) is resolved by several fine cells in y.
    spy = _axis_spacings(spec, "y")
    fine_in_core = [d for q, d in zip(spec.coords.y, spy)
                    if LY / 2 - 0.11 <= q <= LY / 2 + 0.11]
    assert len(fine_in_core) >= 4
    # Cladding cells (near the y edge) are coarser than core cells.
    assert max(spy) > min(spy) * 1.5
    # Round-trips through the wire format unchanged.
    sim2 = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert sim2.grid.coords.y == spec.coords.y


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kw,match", [
    (dict(wavelength_um=0.0), "wavelength_um"),
    (dict(steps_per_wvl=0.0), "steps_per_wvl"),
    (dict(max_grading=1.0), "max_grading must be > 1"),
    (dict(max_grading=11.0), "exceeds GRADED_RATIO_GUARD"),
    (dict(background_index=0.5), "background_index"),
    (dict(axes="xq"), "subset of 'xyz'"),
    (dict(axes="xx"), "no repeats"),
    (dict(min_nodes=3), "min_nodes"),
])
def test_invalid_inputs_rejected(kw, match):
    base = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                background_index=1.0, steps_per_wvl=20.0, max_grading=1.3)
    base.update(kw)
    with pytest.raises(ValueError, match=match):
        auto_grid(**base)


def test_min_nodes_floor_for_tiny_domain():
    # A domain so small the target produces < 4 cells still yields >= 4 nodes
    # (section 15.10), uniformly subdivided.
    spec = auto_grid(size_um=(0.05, 0.05, 0.05), wavelength_um=1.31,
                     structures=[], background_index=1.0,
                     steps_per_wvl=4.0, max_grading=1.3, axes="x")
    assert len(spec.coords.x) >= 4
    assert spec.coords.x[0] == 0.0


def test_sphere_geometry_supported():
    # A high-index sphere refines around its bounding box on each axis.
    sph = ph.Structure(
        geometry=ph.Sphere(center_um=(1.0, 1.0, 1.0), radius_um=0.3),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[sph], background_index=1.0,
                     steps_per_wvl=16.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    finest_idx = min(range(len(sp)), key=lambda i: sp[i])
    qx = spec.coords.x
    cell_center = 0.5 * (qx[finest_idx] + qx[finest_idx + 1])
    assert 0.6 <= cell_center <= 1.4  # within the sphere's bounding box


# --------------------------------------------------------------------------- #
# Curved / extruded geometries: Cylinder and PolySlab now drive refinement
# (previously returned None and were silently ignored — exactly the curved
# structures where subpixel matters most). They are bounded by their enclosing
# box: a safe over-estimate of where the fine mesh is needed.
# --------------------------------------------------------------------------- #

def test_cylinder_refines_along_extrusion_and_transverse():
    # A z-extruded high-index disk (radius 0.4 um, length 1.0 um) centered in a
    # 2um cube. Its bbox is [0.6,1.4] in x,y (radial) and [0.5,1.5] in z (length).
    cyl = ph.Structure(
        geometry=ph.Cylinder(axis="z", center_um=(1.0, 1.0, 1.0),
                             radius_um=0.4, length_um=1.0),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[cyl], background_index=1.0,
                     steps_per_wvl=18.0, max_grading=1.3, axes="xyz")
    # The finest cell on a radial axis sits inside the disk radius bbox.
    for axis, (lo, hi) in (("x", (0.6, 1.4)), ("y", (0.6, 1.4)),
                           ("z", (0.5, 1.5))):
        sp = _axis_spacings(spec, axis)
        q = getattr(spec.coords, axis)
        fi = min(range(len(sp)), key=lambda i: sp[i])
        cell_center = 0.5 * (q[fi] + q[fi + 1])
        assert lo - 0.15 <= cell_center <= hi + 0.15, (axis, cell_center)
        # local dl ~ lambda/(n*steps) ~ 1.31/(3.5*18) = 0.0208 um
        assert min(sp) <= 1.31 / (3.5 * 18.0) * 1.25, (axis, min(sp))


def test_cylinder_partial_sector_uses_full_disk_bbox():
    # A 90-degree sector still refines over the FULL disk bbox (over-estimate is
    # safe — never under-refines). Just assert refinement happened transversely.
    sec = ph.Structure(
        geometry=ph.Cylinder(axis="z", center_um=(1.0, 1.0, 1.0),
                             radius_um=0.4, length_um=2.0,
                             angle_start=0.0, angle_stop=math.pi / 2),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[sec], background_index=1.0,
                     steps_per_wvl=16.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    assert _max_cell_to_cell_ratio(sp) > 1.05  # the mesh is graded, not uniform


def test_polyslab_refines_in_cross_section_and_along_extrusion():
    # A z-extruded triangle (vertices in x,y) with slab_bounds in z. The polygon
    # bbox is [0.6,1.4]x[0.7,1.3]; the slab runs z in [0.4,0.9].
    verts = ((0.6, 0.7), (1.4, 0.7), (1.0, 1.3))
    poly = ph.Structure(
        geometry=ph.PolySlab(axis="z", vertices_um=verts,
                            slab_bounds_um=(0.4, 0.9)),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[poly], background_index=1.0,
                     steps_per_wvl=18.0, max_grading=1.3, axes="xyz")
    for axis, (lo, hi) in (("x", (0.6, 1.4)), ("y", (0.7, 1.3)),
                           ("z", (0.4, 0.9))):
        sp = _axis_spacings(spec, axis)
        q = getattr(spec.coords, axis)
        fi = min(range(len(sp)), key=lambda i: sp[i])
        cell_center = 0.5 * (q[fi] + q[fi + 1])
        assert lo - 0.15 <= cell_center <= hi + 0.15, (axis, cell_center)


def test_polyslab_slanted_sidewall_refines_reference_plane_bbox():
    # reference_plane="bottom" with a POSITIVE angle is the one combination
    # where the section only narrows away from the reference plane, so the
    # reference-plane vertex bbox is exact (no dilation). Refinement still
    # concentrates around the (widest) cross-section.
    verts = ((0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5))
    poly = ph.Structure(
        geometry=ph.PolySlab(axis="z", vertices_um=verts,
                            slab_bounds_um=(0.4, 1.6), sidewall_angle=0.3,
                            reference_plane="bottom"),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[poly], background_index=1.0,
                     steps_per_wvl=16.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    fi = min(range(len(sp)), key=lambda i: sp[i])
    qx = spec.coords.x
    cell_center = 0.5 * (qx[fi] + qx[fi + 1])
    assert 0.5 - 0.15 <= cell_center <= 1.5 + 0.15, cell_center


def test_polyslab_slanted_span_covers_dilated_extent():
    # Regression (K3): for reference_plane "middle"/"top" (or a negative
    # angle) a slanted PolySlab is WIDER than the reference-plane vertex bbox
    # by up to |tan(angle)| * thickness; the old span used the raw bbox and
    # under-covered the structure (refinement intervals and snap targets
    # missed the widened edge). The dilation must match _bounds.py's
    # conservative posture: distance from the reference plane to the WIDEST
    # face (bottom for angle > 0, top for angle < 0).
    from photonhub.components.grid import _geometry_axis_span

    verts = ((0.7, 0.7), (1.3, 0.7), (1.3, 1.3), (0.7, 1.3))
    thick, angle = 1.2, 0.3
    t = math.tan(angle)

    def span(ref, ang, axis=0):
        g = ph.PolySlab(axis="z", vertices_um=verts,
                        slab_bounds_um=(0.2, 0.2 + thick),
                        sidewall_angle=ang, reference_plane=ref)
        return _geometry_axis_span(g, axis)

    # bottom + positive angle: narrows toward +z everywhere -> no dilation.
    assert span("bottom", angle) == pytest.approx((0.7, 1.3))
    # middle: widest at the bottom face, half a thickness from the reference.
    assert span("middle", angle) == pytest.approx(
        (0.7 - t * thick / 2, 1.3 + t * thick / 2))
    # top: widest at the bottom face, a full thickness from the reference.
    assert span("top", angle) == pytest.approx(
        (0.7 - t * thick, 1.3 + t * thick))
    # Negative angle mirrors (widest at the TOP face).
    assert span("top", -angle) == pytest.approx((0.7, 1.3))
    assert span("bottom", -angle) == pytest.approx(
        (0.7 - t * thick, 1.3 + t * thick))
    assert span("middle", -angle) == pytest.approx(
        (0.7 - t * thick / 2, 1.3 + t * thick / 2))
    # The extrusion axis is unaffected (slab bounds are exact).
    assert span("middle", angle, axis=2) == pytest.approx((0.2, 0.2 + thick))


def test_polyslab_middle_reference_snaps_widened_edges():
    # End-to-end K3: with reference_plane="middle" and a nonzero angle, the
    # snap targets sit on the DILATED extent — the old code left the widened
    # edges (0.8 - pad, 1.2 + pad) unrefined and unsnapped.
    thick, angle = 0.6, 0.35
    pad = math.tan(angle) * thick / 2.0
    verts = ((0.8, 0.8), (1.2, 0.8), (1.2, 1.2), (0.8, 1.2))
    poly = ph.Structure(
        geometry=ph.PolySlab(axis="z", vertices_um=verts,
                             slab_bounds_um=(0.7, 0.7 + thick),
                             sidewall_angle=angle, reference_plane="middle"),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[poly], background_index=1.0,
                     steps_per_wvl=18.0, max_grading=1.3, axes="x")
    for e in (0.8 - pad, 1.2 + pad):
        assert _nearest_node_dist(spec.coords.x, e) <= _SNAP_TOL, e


def test_curved_and_box_refine_consistently():
    # A Cylinder with the same bbox as a Box must refine the same axis span (the
    # bbox-based span is geometry-agnostic). Cell counts should be comparable.
    box = _box((1.0, 1.0, 1.0), (0.8, 0.8, 0.8), eps=12.25)
    cyl = ph.Structure(
        geometry=ph.Cylinder(axis="z", center_um=(1.0, 1.0, 1.0),
                             radius_um=0.4, length_um=0.8),
        medium=ph.Medium(permittivity=12.25))
    kw = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
              background_index=1.0, steps_per_wvl=18.0,
              max_grading=1.3, axes="x")
    sbox = auto_grid(structures=[box], **kw)
    scyl = auto_grid(structures=[cyl], **kw)
    # Same x-bbox [0.6,1.4] -> identical coordinate arrays.
    assert sbox.coords.x == scyl.coords.x


def test_dispersive_structure_refines_at_in_band_index_not_eps_inf():
    # A dispersive (Lorentz) medium's `permittivity` is eps_inf (NUMERICS.md
    # §19), which understates the in-band index. Regression (K2): the resolver
    # used to mesh a dispersive Si fit (eps_inf=8 -> n~2.83) ~20% coarser than
    # its true in-band n~3.48. It must now evaluate Re eps(omega) at the target
    # wavelength: strictly finer than the same-eps_inf non-dispersive box, and
    # matching the in-band dl target.
    lam = 1.55
    pole = ph.LorentzPole(resonance_frequency_hz=C0 / 0.6e-6,  # 600 nm
                          delta_eps=3.5)
    geo = ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.45, 0.22, 4.0))
    disp = ph.Structure(geometry=geo,
                        medium=ph.Medium(permittivity=8.0, lorentz=pole))
    nondisp = ph.Structure(geometry=geo, medium=ph.Medium(permittivity=8.0))
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=lam,
              background_index=1.444, steps_per_wvl=20.0, max_grading=1.3,
              axes="x")
    fine = auto_grid(structures=[disp], **kw)
    coarse = auto_grid(structures=[nondisp], **kw)
    assert min(_axis_spacings(fine, "x")) < min(_axis_spacings(coarse, "x"))
    assert len(fine.coords.x) > len(coarse.coords.x)
    # In-band eps = eps_inf + de*f0^2/(f0^2 - f^2) (gamma=0) ~ 12.12, n ~ 3.48;
    # local target dl = lam/(n*steps).
    f, f0 = C0 / (lam * 1e-6), C0 / 0.6e-6
    n_band = math.sqrt(8.0 + 3.5 * f0 ** 2 / (f0 ** 2 - f ** 2))
    expected_dl = lam / (n_band * 20.0)
    assert min(_axis_spacings(fine, "x")) <= expected_dl * 1.2
    assert min(_axis_spacings(coarse, "x")) > expected_dl * 1.1  # eps_inf mesh


# --------------------------------------------------------------------------- #
# Tidy3D-parity hardening: wavelength inference from a source, dl_min floor,
# enforced-refinement override regions.
# --------------------------------------------------------------------------- #

def test_wavelength_inferred_from_source_matches_explicit():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    lam_um = 1.31
    f0 = C0 / (lam_um * 1e-6)
    src = ph.PointDipole(
        center_um=(1.0, 1.0, 0.5), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.1 * f0))
    kw = dict(size_um=(2.0, 2.0, 4.0), structures=[core],
              background_index=1.444, steps_per_wvl=20.0,
              max_grading=1.3, axes="xy")
    explicit = auto_grid(wavelength_um=lam_um, **kw)
    inferred = auto_grid(source=src, **kw)
    # The inferred wavelength is c/f0 (== lam_um here), so coords match closely.
    assert inferred.coords.x == explicit.coords.x
    assert inferred.coords.y == explicit.coords.y


def test_wavelength_source_mutual_exclusion_and_required():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    f0 = C0 / 1.31e-6
    src = ph.PointDipole(
        center_um=(1.0, 1.0, 0.5), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.1 * f0))
    base = dict(size_um=(2.0, 2.0, 4.0), structures=[core],
                background_index=1.444, axes="x")
    # both -> error
    with pytest.raises(ValueError, match="exactly one"):
        auto_grid(wavelength_um=1.31, source=src, **base)
    # neither -> error
    with pytest.raises(ValueError, match="wavelength_um is required"):
        auto_grid(**base)


def test_dl_min_floor_caps_refinement():
    # A very high-index inclusion would normally pull dl ~ lambda/(n*steps); the
    # dl_min floor caps the minimum spacing so the cell count cannot explode.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=49.0)  # n=7
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              structures=[core], background_index=1.444,
              steps_per_wvl=20.0, max_grading=1.3, axes="x")
    free = auto_grid(**kw)
    floor = 0.03
    capped = auto_grid(dl_min_um=floor, **kw)
    # No realized cell falls below the floor (within coordinate quantization).
    sp = _axis_spacings(capped, "x")
    assert min(sp) >= floor - 1e-3, min(sp)
    # And the floor genuinely bites: fewer cells than the unfloored mesh.
    assert len(capped.coords.x) < len(free.coords.x)


def test_dl_min_must_be_positive():
    with pytest.raises(ValueError, match="dl_min_um must be > 0"):
        auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                  background_index=1.0, dl_min_um=0.0, axes="x")


def test_refine_region_override_forces_fine_mesh_in_empty_space():
    # No structures, but an override forces fine cells over [0.8, 1.2] in x.
    fine_dl = 0.01
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x",
                     refine_regions=[("x", 0.8, 1.2, fine_dl)])
    sp = _axis_spacings(spec, "x")
    q = spec.coords.x
    fi = min(range(len(sp)), key=lambda i: sp[i])
    cell_center = 0.5 * (q[fi] + q[fi + 1])
    assert 0.8 - 0.1 <= cell_center <= 1.2 + 0.1, cell_center
    # The override pulled cells well below the background spacing (0.05 um).
    assert min(sp) <= fine_dl * 1.3


def test_refine_region_respects_dl_min_floor():
    # An over-fine override is still clamped to the dl_min floor.
    floor = 0.02
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x",
                     dl_min_um=floor,
                     refine_regions=[("x", 0.8, 1.2, 0.002)])
    sp = _axis_spacings(spec, "x")
    assert min(sp) >= floor - 1e-3, min(sp)


@pytest.mark.parametrize("region,match", [
    (("q", 0.8, 1.2, 0.01), "axis must be one of"),
    (("x", 1.2, 0.8, 0.01), "hi > lo"),
    (("x", 0.8, 1.2, 0.0), "dl_um must be > 0"),
    (("x", 0.8, 1.2), r"\(axis_letter, lo_um, hi_um, dl_um\)"),
])
def test_refine_region_validation(region, match):
    with pytest.raises(ValueError, match=match):
        auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                  background_index=1.0, axes="x", refine_regions=[region])


def test_refine_region_mesh_freeze_deterministic():
    # Overrides are part of the pure-function inputs: same inputs -> same coords.
    kw = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
              background_index=1.0, steps_per_wvl=20.0, max_grading=1.3,
              axes="x", refine_regions=[("x", 0.8, 1.2, 0.01)])
    a = auto_grid(**kw)
    b = auto_grid(**kw)
    assert a.coords.x == b.coords.x


# --------------------------------------------------------------------------- #
# Simulation.with_auto_grid convenience: opt-in, derives the mesh from the
# scene (size / structures / background / source wavelength). The default
# UniformGridSpec is unchanged, so no existing scene's wire output moves.
# --------------------------------------------------------------------------- #

def test_simulation_with_auto_grid_convenience():
    from photonhub.components.grid import GradedGridSpec as _GGS

    LX, LY, LZ = 2.0, 2.0, 4.0
    core = _box((LX / 2, LY / 2, LZ / 2), (0.45, 0.22, LZ * 2), eps=12.25)
    f0 = C0 / 1.31e-6
    src = ph.PointDipole(
        center_um=(LX / 2, LY / 2, 0.7), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.1 * f0))
    sim = ph.Simulation(
        size_um=(LX, LY, LZ), grid=ph.UniformGridSpec(dl_um=0.05),
        run={"n_steps": 100}, background=ph.Background(permittivity=1.444 ** 2),
        structures=(core,), boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=10, sources=[src])
    # Default scene uses the uniform grid (no churn).
    assert isinstance(sim.grid, ph.UniformGridSpec)

    auto = sim.with_auto_grid(steps_per_wvl=20.0, max_grading=1.3, axes="xy")
    assert isinstance(auto.grid, GradedGridSpec)
    # Original is untouched (frozen copy semantics).
    assert isinstance(sim.grid, ph.UniformGridSpec)
    # Wavelength was inferred from the source -> matches the explicit call.
    explicit = auto_grid(size_um=(LX, LY, LZ), wavelength_um=1.31,
                         structures=[core], background_index=1.444,
                         steps_per_wvl=20.0, max_grading=1.3, axes="xy")
    assert auto.grid.coords.x == explicit.coords.x
    assert auto.grid.coords.y == explicit.coords.y
    # The resulting sim round-trips through the wire format.
    sim2 = ph.Simulation.from_wire_json(auto.to_wire_json())
    assert sim2.grid.coords.x == auto.grid.coords.x


def test_with_auto_grid_plane_wave_builds_supported_scene():
    # Graded TF/SF sources now run on transverse-graded and
    # propagation-graded axes; the helper must preserve the source.
    f0 = C0 / 1.55e-6
    pw = ph.PlaneWave(
        axis="z", direction="+", position_um=0.5, polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.1 * f0))
    core = _box((1.0, 1.0, 2.0), (0.45, 0.22, 4.0), eps=12.25)
    sim = ph.Simulation(
        size_um=(2.0, 2.0, 4.0), grid=ph.UniformGridSpec(dl_um=0.05),
        run={"n_steps": 100}, structures=(core,),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
        sources=[pw])
    auto = sim.with_auto_grid(steps_per_wvl=15.0, max_grading=1.3, axes="x")
    assert isinstance(auto.grid, GradedGridSpec)
    assert auto.sources == sim.sources
    # Direct construction of the same combination is valid too.
    spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.55,
                     structures=[core], steps_per_wvl=15.0, max_grading=1.3,
                     axes="x", periodic_axes="x")
    direct = ph.Simulation(
        size_um=(2.0, 2.0, 4.0), grid=spec, run={"n_steps": 100},
        structures=(core,),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
        sources=[pw])
    assert isinstance(direct.grid, GradedGridSpec)


# --------------------------------------------------------------------------- #
# Interface grid-line SNAPPING (Tidy3D AutoGrid parity, gap #10a). A primary
# node must land EXACTLY on each refining-structure interface / override edge,
# WITHOUT breaking the grading-ratio or dl_min invariants, and deterministically
# (mesh-freeze). snap_interfaces defaults to True.
# --------------------------------------------------------------------------- #

_SNAP_TOL = 1e-6  # exact-snap tolerance (one coordinate quantum is 1e-7 um)


def _nearest_node_dist(q, target):
    return min(abs(c - target) for c in q)


def test_snap_box_faces_land_on_nodes():
    # SOI core faces: x at 1.0 +/- 0.225 -> {0.775, 1.225}; y at 1.0 +/- 0.11.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                     structures=[core], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="xy")
    for axis, faces in (("x", (0.775, 1.225)), ("y", (0.89, 1.11))):
        q = getattr(spec.coords, axis)
        for f in faces:
            assert _nearest_node_dist(q, f) <= _SNAP_TOL, (axis, f)


def test_snap_off_recovers_pre_snap_and_interface_falls_mid_cell():
    # With snapping OFF the interface generally does NOT land on a node — this is
    # the gap snapping closes. The snapped spec differs from the unsnapped one.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31, structures=[core],
              background_index=1.444, steps_per_wvl=20.0, max_grading=1.3,
              axes="x")
    on = auto_grid(snap_interfaces=True, **kw)
    off = auto_grid(snap_interfaces=False, **kw)
    assert on.coords.x != off.coords.x  # snapping moved nodes
    # Pre-snap: at least one face is mid-cell (not on a node).
    assert max(_nearest_node_dist(off.coords.x, f)
               for f in (0.775, 1.225)) > _SNAP_TOL
    # Post-snap: both faces are on nodes.
    for f in (0.775, 1.225):
        assert _nearest_node_dist(on.coords.x, f) <= _SNAP_TOL


def test_snap_preserves_grading_ratio():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    for grading in (1.2, 1.3, 1.4):
        spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                         structures=[core], background_index=1.444,
                         steps_per_wvl=18.0, max_grading=grading, axes="xy")
        for axis in "xy":
            sp = _axis_spacings(spec, axis)
            assert _max_cell_to_cell_ratio(sp) <= grading + 1e-3, (axis, grading)
            assert max(sp) / min(sp) <= GRADED_RATIO_GUARD


def test_snap_respects_dl_min_floor():
    # Snapping must never push a cell below the dl_min floor (it abandons a
    # target it cannot reconcile rather than emitting a sub-floor cell).
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=49.0)  # n=7
    floor = 0.03
    spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                     structures=[core], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x",
                     dl_min_um=floor)
    sp = _axis_spacings(spec, "x")
    assert min(sp) >= floor - 1e-3, min(sp)


def test_snap_deterministic_mesh_freeze():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31, structures=[core],
              background_index=1.444, steps_per_wvl=22.0, max_grading=1.3,
              axes="xy")
    a = auto_grid(**kw)
    b = auto_grid(**kw)
    assert a.coords.x == b.coords.x
    assert a.coords.y == b.coords.y
    assert a.model_dump_json() == b.model_dump_json()


def test_snap_independent_of_structure_order():
    # Snap targets are a pure function of the SET of interfaces, so reordering
    # the structure list must not move a single node.
    s1 = _box((0.6, 1.0, 1.0), (0.3, 0.3, 4.0), eps=12.25)
    s2 = _box((1.4, 1.0, 1.0), (0.3, 0.3, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              background_index=1.444, steps_per_wvl=18.0, max_grading=1.3,
              axes="x")
    a = auto_grid(structures=[s1, s2], **kw)
    b = auto_grid(structures=[s2, s1], **kw)
    assert a.coords.x == b.coords.x


def test_snap_cylinder_bbox_edges():
    # A z-extruded disk: x/y radial bbox {0.6, 1.4}; z length bbox {0.5, 1.5}.
    cyl = ph.Structure(
        geometry=ph.Cylinder(axis="z", center_um=(1.0, 1.0, 1.0),
                             radius_um=0.4, length_um=1.0),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[cyl], background_index=1.0,
                     steps_per_wvl=18.0, max_grading=1.3, axes="xyz")
    for axis, edges in (("x", (0.6, 1.4)), ("y", (0.6, 1.4)),
                        ("z", (0.5, 1.5))):
        q = getattr(spec.coords, axis)
        for e in edges:
            assert _nearest_node_dist(q, e) <= _SNAP_TOL, (axis, e)
        assert _max_cell_to_cell_ratio(_axis_spacings(spec, axis)) <= 1.3 + 1e-3


def test_snap_polyslab_boundaries():
    # Triangle bbox x {0.6, 1.4}, y {0.7, 1.3}; slab z {0.4, 0.9}.
    verts = ((0.6, 0.7), (1.4, 0.7), (1.0, 1.3))
    poly = ph.Structure(
        geometry=ph.PolySlab(axis="z", vertices_um=verts,
                            slab_bounds_um=(0.4, 0.9)),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[poly], background_index=1.0,
                     steps_per_wvl=18.0, max_grading=1.3, axes="xyz")
    for axis, edges in (("x", (0.6, 1.4)), ("y", (0.7, 1.3)),
                        ("z", (0.4, 0.9))):
        q = getattr(spec.coords, axis)
        for e in edges:
            assert _nearest_node_dist(q, e) <= _SNAP_TOL, (axis, e)


def test_snap_override_region_edges():
    # An override-region edge is an interface too — snap a node onto it. Use a
    # grading wide enough that the coarsening shoulder has room for both edges.
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.4, axes="x",
                     refine_regions=[("x", 0.8, 1.2, 0.01)])
    q = spec.coords.x
    for e in (0.8, 1.2):
        assert _nearest_node_dist(q, e) <= _SNAP_TOL, e


def test_snap_best_effort_never_violates_grading():
    # When an interface CANNOT be reconciled with the grading guard (a coarsening
    # shoulder already saturated at max_grading), snapping abandons that target
    # rather than emitting an out-of-spec mesh. The result is still valid.
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x",
                     refine_regions=[("x", 0.8, 1.2, 0.01)])
    sp = _axis_spacings(spec, "x")
    # Grading invariant holds regardless of how many targets were snappable.
    assert _max_cell_to_cell_ratio(sp) <= 1.3 + 1e-3
    # The closer edge still snaps even when the farther one is abandoned.
    assert _nearest_node_dist(spec.coords.x, 0.8) <= _SNAP_TOL


def test_snap_off_matches_legacy_for_uniform_cases():
    # snap_interfaces has no effect when there are no refining interfaces (the
    # "no refinement -> uniform" cases): toggling it is a no-op there.
    kw = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0, structures=[],
              background_index=1.0, steps_per_wvl=20.0, max_grading=1.3,
              axes="x")
    on = auto_grid(snap_interfaces=True, **kw)
    off = auto_grid(snap_interfaces=False, **kw)
    assert on.coords.x == off.coords.x


def test_snap_near_edge_interface_preserves_realized_length():
    # Regression (K1): a box edge at x=9.95 in a 10um domain used to get its
    # snap target anchored onto one of the LAST TWO nodes; the §15.1 realized
    # length is q[-1] + (q[-1] - q[-2]) (replicate-last), so moving either
    # changed the realized domain (10.0 -> 9.9929 / 10.0054 depending on the
    # march). The last two nodes are now pinned: the realized length equals the
    # request to within the documented 1e-7 um coordinate quantization (the
    # 1e-9-relative closure is asserted internally BEFORE quantization).
    box = _box((9.0, 5.0, 5.0), (1.9, 1.0, 1.0), eps=12.25)
    spec = auto_grid(size_um=(10.0, 10.0, 10.0), wavelength_um=1.55,
                     structures=[box], axes="x")
    q = spec.coords.x
    realized = q[-1] + (q[-1] - q[-2])
    assert realized == pytest.approx(10.0, abs=5e-7)
    # The interior interface (x=8.05) still snaps; the near-edge one (x=9.95)
    # is best-effort — it may be abandoned when it cannot be reconciled with
    # the grading + exact-closure invariants, but must never stretch the
    # domain. Grading holds throughout.
    assert _nearest_node_dist(q, 8.05) <= _SNAP_TOL
    sp = _axis_spacings(spec, "x")
    assert _max_cell_to_cell_ratio(sp) <= 1.4 + 1e-3


def test_snap_realized_domain_length_unchanged():
    # Snapping only moves INTERIOR nodes; the §15.1 closing node (hence the
    # realized domain length) is untouched, so the snapped and unsnapped specs
    # realize the same length.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31, structures=[core],
              background_index=1.444, steps_per_wvl=20.0, max_grading=1.3,
              axes="x")
    on = auto_grid(snap_interfaces=True, **kw)
    off = auto_grid(snap_interfaces=False, **kw)
    realized = lambda q: q[-1] + (q[-1] - q[-2])
    assert math.isclose(realized(on.coords.x), realized(off.coords.x),
                        abs_tol=1e-3)


# --------------------------------------------------------------------------- #
# Geometry-based MeshOverride (Tidy3D MeshOverrideStructure parity). An override
# is projected onto each axis it governs (the geometry's per-axis bounding span
# at the override dl) and merged into the same enforced-refinement machinery as
# refine_regions — so the existing guarantees (dl_min floor, snapping, grading,
# mesh-freeze) hold for it too. These pin the projection + integration contract.
# --------------------------------------------------------------------------- #

def test_mesh_override_box_forces_fine_mesh_in_empty_space():
    # No structures: a Box override pulls fine cells over its x-span [0.8, 1.2].
    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.4, 0.4, 0.4)),
        dl_um=0.01)
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0, structures=[],
                     background_index=1.0, steps_per_wvl=20.0, max_grading=1.3,
                     axes="x", mesh_overrides=[ov])
    sp = _axis_spacings(spec, "x")
    q = spec.coords.x
    fi = min(range(len(sp)), key=lambda i: sp[i])
    cell_center = 0.5 * (q[fi] + q[fi + 1])
    assert 0.8 - 0.1 <= cell_center <= 1.2 + 0.1, cell_center
    assert min(sp) <= 0.01 * 1.3  # well below the 0.05 background spacing


def test_mesh_override_matches_equivalent_refine_regions():
    # A Box override is exactly the axis-interval refine_regions it projects to,
    # so the two routes must produce byte-identical coordinate arrays.
    box = ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.4, 0.6, 0.8))
    ov = ph.MeshOverride(geometry=box, dl_um=0.02)
    kw = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0, structures=[],
              background_index=1.0, steps_per_wvl=20.0, max_grading=1.3,
              axes="xyz")
    via_override = auto_grid(mesh_overrides=[ov], **kw)
    # Box spans: x 1+/-0.2 -> [0.8,1.2]; y 1+/-0.3 -> [0.7,1.3]; z 1+/-0.4.
    via_regions = auto_grid(refine_regions=[
        ("x", 0.8, 1.2, 0.02), ("y", 0.7, 1.3, 0.02), ("z", 0.6, 1.4, 0.02),
    ], **kw)
    assert via_override.coords.x == via_regions.coords.x
    assert via_override.coords.y == via_regions.coords.y
    assert via_override.coords.z == via_regions.coords.z


def test_mesh_override_per_axis_dl_with_none_skips_axis():
    # (dx, dy, None): refine x and y, leave z ungoverned by the override.
    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.4, 0.4, 0.4)),
        dl_um=(0.02, 0.02, None))
    assert ov.axis_regions() == [("x", 0.8, 1.2, 0.02), ("y", 0.8, 1.2, 0.02)]
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0, structures=[],
                     background_index=1.0, steps_per_wvl=20.0, max_grading=1.3,
                     axes="xyz", mesh_overrides=[ov])
    # x and y are pulled fine; z (no override, no structure, n=1 background) is
    # left at the coarse background spacing everywhere.
    assert min(_axis_spacings(spec, "x")) <= 0.02 * 1.3
    assert min(_axis_spacings(spec, "y")) <= 0.02 * 1.3
    assert min(_axis_spacings(spec, "z")) > 0.04  # ~ background (lambda/20)


def test_mesh_override_respects_dl_min_floor():
    # An over-fine override is still clamped to the dl_min floor (no explosion).
    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.4, 0.4, 0.4)),
        dl_um=0.002)
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0, structures=[],
                     background_index=1.0, steps_per_wvl=20.0, max_grading=1.3,
                     axes="x", dl_min_um=0.02, mesh_overrides=[ov])
    assert min(_axis_spacings(spec, "x")) >= 0.02 - 1e-3


def test_mesh_override_curved_geometry_uses_bounding_box():
    # A Cylinder acts through its enclosing box: along its extrusion (z) axis the
    # span is the length; the two transverse axes carry the radius.
    cyl = ph.Cylinder(axis="z", center_um=(1.0, 1.0, 1.0), radius_um=0.3,
                      length_um=0.5)
    ov = ph.MeshOverride(geometry=cyl, dl_um=0.02)
    regs = dict((r[0], r) for r in ov.axis_regions())
    assert regs["x"][1:3] == (0.7, 1.3)  # 1 +/- radius
    assert regs["y"][1:3] == (0.7, 1.3)
    assert regs["z"][1:3] == (0.75, 1.25)  # 1 +/- length/2


def test_mesh_override_mesh_freeze_deterministic():
    # Overrides are pure-function inputs: identical -> byte-identical coords.
    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.4, 0.4, 0.4)),
        dl_um=0.01)
    kw = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0, structures=[],
              background_index=1.0, steps_per_wvl=20.0, max_grading=1.3,
              axes="x", mesh_overrides=[ov])
    assert auto_grid(**kw).coords.x == auto_grid(**kw).coords.x


def test_mesh_override_snaps_box_edges_onto_nodes():
    # The override region edges are interfaces too: a node lands on each (the
    # same grid-line snapping refine_regions edges get). A grading wide enough
    # that the coarsening shoulder has room for both edges (as in the
    # refine_regions snap test).
    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.4, 0.4, 0.4)),
        dl_um=0.01)
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0, structures=[],
                     background_index=1.0, steps_per_wvl=20.0, max_grading=1.4,
                     axes="x", mesh_overrides=[ov])
    for e in (0.8, 1.2):
        assert _nearest_node_dist(spec.coords.x, e) <= _SNAP_TOL, e


@pytest.mark.parametrize("dl", [(None, None, None), (0.0, None, None),
                                (-0.01, None, None)])
def test_mesh_override_dl_validation(dl):
    with pytest.raises(ValueError):
        ph.MeshOverride(
            geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.4, 0.4, 0.4)),
            dl_um=dl)


def test_simulation_with_mesh_overrides_convenience():
    from photonhub.components.grid import GradedGridSpec as _GGS

    LX, LY, LZ = 2.0, 2.0, 4.0
    f0 = C0 / 1.31e-6
    src = ph.PointDipole(
        center_um=(LX / 2, LY / 2, 0.7), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.1 * f0))
    sim = ph.Simulation(
        size_um=(LX, LY, LZ), grid=ph.UniformGridSpec(dl_um=0.05),
        run={"n_steps": 100}, boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=10, sources=[src])
    assert isinstance(sim.grid, ph.UniformGridSpec)  # default unchanged

    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(LX / 2, LY / 2, LZ / 2),
                        size_um=(0.5, 0.5, 1.0)),
        dl_um=(0.02, 0.02, None))
    meshed = sim.with_mesh_overrides(ov, steps_per_wvl=20.0, max_grading=1.3,
                                     axes="xy")
    assert isinstance(meshed.grid, _GGS)
    assert isinstance(sim.grid, ph.UniformGridSpec)  # original untouched
    # Equivalent to calling with_auto_grid with the projected mesh_overrides.
    direct = sim.with_auto_grid(steps_per_wvl=20.0, max_grading=1.3, axes="xy",
                                mesh_overrides=[ov])
    assert meshed.grid.coords.x == direct.grid.coords.x
    # Round-trips through the wire format (it is an ordinary GradedGridSpec).
    sim2 = ph.Simulation.from_wire_json(meshed.to_wire_json())
    assert sim2.grid.coords.x == meshed.grid.coords.x


# --------------------------------------------------------------------------- #
# Periodic-seam symmetry (NUMERICS.md §15.2). The engine's replicate dual-
# spacing closure is only correct on a PERIODIC axis when the first and last
# primary spacings match — phsolver validate (and the Simulation validator
# mirroring it) hard-rejects the unequal case — so axes listed in
# ``periodic_axes`` are generated seam-symmetrically: both walls grade down to
# the finer of the two seam cells, exactly equal through snap + quantization.
# --------------------------------------------------------------------------- #

def _seam_pair(q):
    """(first, last) primary spacings — the §15.2 pair the engine compares."""
    return q[1] - q[0], q[-1] - q[-2]


def _centered_override_kw(**extra):
    # The integration-test scene shape: empty 2um cube, centered 0.6um override
    # at 0.04um, lambda=1.55, 12 steps/wvl -> coarse ~0.129um walls.
    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.6, 0.6, 0.6)),
        dl_um=0.04)
    kw = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.55, structures=[],
              background_index=1.0, steps_per_wvl=12.0, max_grading=1.3,
              axes="xyz", mesh_overrides=[ov])
    kw.update(extra)
    return kw


def test_periodic_axes_seam_spacings_exactly_equal():
    spec = auto_grid(**_centered_override_kw(periodic_axes="xyz"))
    for axis in "xyz":
        q = getattr(spec.coords, axis)
        first, last = _seam_pair(q)
        # The engine gate is 1e-12 RELATIVE; the generator lands well inside.
        assert abs(first - last) <= 1e-12 * max(first, last), (axis, first, last)
        # Replicate-last closure: realized length == requested domain.
        assert q[-1] + (q[-1] - q[-2]) == pytest.approx(2.0, abs=1e-9)
        # Still refined in the override, still graded within the ratio.
        sp = _axis_spacings(spec, axis)
        assert min(sp) <= 0.04 * 1.3
        assert _max_cell_to_cell_ratio(sp) <= 1.3 + 1e-3


def test_periodic_axes_wall_structure_grades_far_wall_down():
    # Structure at the LEFT wall only: under periodicity its fine mesh abuts
    # the seam, so the RIGHT wall must grade down to the same fine spacing
    # (t = min of the two seam cells) — never the fine wall coarsening up.
    core = _box((0.1, 1.0, 1.0), (0.2, 0.4, 0.4), eps=12.25)  # x in [0, 0.2]
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[core], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x",
                     periodic_axes="x")
    q = spec.coords.x
    first, last = _seam_pair(q)
    assert abs(first - last) <= 1e-12 * max(first, last)
    # Both seam cells are FINE (~lambda/(3.5*20) ~ 0.019), far below the
    # 0.0655 background: the far wall came DOWN to the structure's spacing.
    assert last <= 0.03
    sp = _axis_spacings(spec, "x")
    assert _max_cell_to_cell_ratio(sp) <= 1.3 + 1e-3


def test_periodic_axes_empty_is_default_and_only_flagged_axes_change():
    # periodic_axes="" is byte-identical to omitting it (no behavior change
    # for existing scenes), and flagging one axis leaves the others untouched.
    base = auto_grid(**_centered_override_kw())
    empty = auto_grid(**_centered_override_kw(periodic_axes=""))
    only_x = auto_grid(**_centered_override_kw(periodic_axes="x"))
    for axis in "xyz":
        assert getattr(empty.coords, axis) == getattr(base.coords, axis)
    assert only_x.coords.y == base.coords.y
    assert only_x.coords.z == base.coords.z
    # And x actually changed (the base scene's seam was unequal).
    b_first, b_last = _seam_pair(base.coords.x)
    assert abs(b_first - b_last) > 1e-12 * max(b_first, b_last)
    s_first, s_last = _seam_pair(only_x.coords.x)
    assert abs(s_first - s_last) <= 1e-12 * max(s_first, s_last)


def test_periodic_axes_mesh_freeze_and_uniform_axis_noop():
    # Deterministic (mesh-freeze), and a periodic axis that is NOT graded
    # (not in ``axes``) is simply left uniform.
    kw = _centered_override_kw(axes="xy", periodic_axes="xyz")
    a, b = auto_grid(**kw), auto_grid(**kw)
    assert a.coords.x == b.coords.x and a.coords.y == b.coords.y
    assert a.coords.z is None


def test_periodic_axes_snap_still_lands_override_edges():
    # Seam symmetry pins node 1 during snapping; interfaces must still snap.
    # Same scene as test_mesh_override_snaps_box_edges_onto_nodes (whose
    # shoulder is wide enough to honor BOTH edges), plus periodic_axes.
    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.4, 0.4, 0.4)),
        dl_um=0.01)
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0, structures=[],
                     background_index=1.0, steps_per_wvl=20.0, max_grading=1.4,
                     axes="x", mesh_overrides=[ov], periodic_axes="x")
    q = spec.coords.x
    for e in (0.8, 1.2):
        assert _nearest_node_dist(q, e) <= _SNAP_TOL, e
    first, last = _seam_pair(q)
    assert abs(first - last) <= 1e-12 * max(first, last)


def test_periodic_axes_validation():
    for bad in ("w", "xx", "xq"):
        with pytest.raises(ValueError, match="periodic_axes"):
            auto_grid(**_centered_override_kw(periodic_axes=bad))


def test_with_mesh_overrides_periodic_boundaries_construct_and_seal_seam():
    # The Simulation wrappers pass periodic_axes from the boundaries: a fully
    # periodic scene (the mesh-override integration-test scene) constructs —
    # i.e. passes the §15.2 Simulation seam validator — with equal seams on
    # every graded axis.
    f0 = 1.934e14  # ~1.55 um
    sim = ph.Simulation(
        size_um=(2.0, 2.0, 2.0), grid=ph.UniformGridSpec(dl_um=0.1),
        run={"n_steps": 40},
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="periodic"),
        sources=[ph.PointDipole(
            center_um=(1.0, 1.0, 1.0), polarization="Ez",
            source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=4.0e13))],
    )
    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.6, 0.6, 0.6)),
        dl_um=0.04)
    meshed = sim.with_mesh_overrides(ov, steps_per_wvl=12.0, max_grading=1.3)
    for axis in "xyz":
        q = getattr(meshed.grid.coords, axis)
        first, last = _seam_pair(q)
        assert abs(first - last) <= 1e-12 * max(first, last), axis
    # A PML scene through the same wrapper stays byte-identical to the plain
    # auto_grid result (periodic_axes resolves to "" — no behavior change).
    pml = ph.Simulation(
        size_um=(2.0, 2.0, 2.0), grid=ph.UniformGridSpec(dl_um=0.1),
        run={"n_steps": 40},  # default boundaries: pml on all six faces
        sources=[ph.PointDipole(
            center_um=(1.0, 1.0, 1.0), polarization="Ez",
            source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=4.0e13))],
    )
    # wavelength_um pinned so the comparison is byte-for-byte against the
    # plain auto_grid call (the wrapper would otherwise infer c/freq0 =
    # 1.55001 um from the source and every coordinate shifts a hair).
    meshed_pml = pml.with_mesh_overrides(ov, wavelength_um=1.55,
                                         steps_per_wvl=12.0, max_grading=1.3)
    plain = auto_grid(**_centered_override_kw())
    for axis in "xyz":
        assert getattr(meshed_pml.grid.coords, axis) == \
            getattr(plain.coords, axis)


def test_with_mesh_overrides_explicit_periodic_axes_override_wins():
    # An explicit periodic_axes= through the wrapper beats the boundary-derived
    # default ("" here disables seam symmetry -> construction now fails the
    # §15.2 Simulation validator, proving the override reached auto_grid AND
    # that the validator catches a hand-disabled seam at construction).
    f0 = 1.934e14
    sim = ph.Simulation(
        size_um=(2.0, 2.0, 2.0), grid=ph.UniformGridSpec(dl_um=0.1),
        run={"n_steps": 40},
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="periodic"),
        sources=[ph.PointDipole(
            center_um=(1.0, 1.0, 1.0), polarization="Ez",
            source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=4.0e13))],
    )
    ov = ph.MeshOverride(
        geometry=ph.Box(center_um=(1.0, 1.0, 1.0), size_um=(0.6, 0.6, 0.6)),
        dl_um=0.04)
    with pytest.raises(ValueError,
                       match="unequal first/last primary spacings"):
        sim.with_mesh_overrides(ov, steps_per_wvl=12.0, max_grading=1.3,
                                periodic_axes="")
