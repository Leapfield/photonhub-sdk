"""Material-aware boundary selection (mirrors Tidy3D): a dispersive medium
crossing a PML face warns and is auto-switched to the adiabatic absorber, while
a plain dielectric keeps the PML. Geometry bounding boxes + the warning + the
``with_auto_boundaries`` helper."""

import math
import warnings

import pytest

import photonhub as ph
from photonhub.components._bounds import geometry_bounds_um


def _disp_medium(eps=2.25):
    return ph.Medium(
        permittivity=eps,
        lorentz=ph.LorentzPole(
            resonance_frequency_hz=5.0e14, delta_eps=1.0, linewidth_hz=1.0e13
        ),
    )


def _plain_medium(eps=2.25):
    return ph.Medium(permittivity=eps)


def _sim(structures, *, boundaries=None, size_um=(4.0, 4.0, 4.0), **overrides):
    return ph.Simulation(
        size_um=size_um,
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        boundaries=boundaries or ph.Boundaries(),  # default = pml on all faces
        structures=structures,
        sources=[
            ph.PointDipole(
                center_um=(2.0, 2.0, 2.0),
                polarization="Ez",
                source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4.0e13),
            )
        ],
        monitors=[ph.FieldSnapshotMonitor(name="final", fields=["Ez"])],
        **overrides,  # e.g. explicit pml_num_layers / pml_alpha_max
    )


# --- geometry bounding boxes ------------------------------------------------

def test_box_bounds():
    bb = geometry_bounds_um(ph.Box(center_um=(1.0, 2.0, 3.0), size_um=(2.0, 4.0, 6.0)))
    assert bb == ((0.0, 2.0), (0.0, 4.0), (0.0, 6.0))


def test_sphere_bounds():
    bb = geometry_bounds_um(ph.Sphere(center_um=(1.0, 1.0, 1.0), radius_um=0.5))
    assert bb == ((0.5, 1.5), (0.5, 1.5), (0.5, 1.5))


def test_cylinder_bounds_axial_vs_transverse():
    cyl = ph.Cylinder(
        axis="z", center_um=(1.0, 1.0, 2.0), radius_um=0.5, length_um=3.0
    )
    bb = geometry_bounds_um(cyl)
    assert bb[0] == (0.5, 1.5)            # transverse: center +/- radius
    assert bb[1] == (0.5, 1.5)
    assert bb[2] == (0.5, 3.5)            # axial: center +/- length/2


def test_polyslab_bounds_pad_for_sidewall():
    # A straight-wall slab: transverse box is the vertex hull, axial is the slab.
    straight = ph.PolySlab(
        axis="z",
        vertices_um=((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
        slab_bounds_um=(0.0, 1.0),
    )
    bb = geometry_bounds_um(straight)
    assert bb[0] == (0.0, 2.0)
    assert bb[1] == (0.0, 1.0)
    assert bb[2] == (0.0, 1.0)
    # A slanted slab dilates outward by |tan(angle)| * thickness on both
    # transverse axes (conservative outer bound).
    slant = straight.model_copy(update={"sidewall_angle": 0.2})
    pad = math.tan(0.2) * 1.0
    bbs = geometry_bounds_um(slant)
    assert bbs[0] == pytest.approx((0.0 - pad, 2.0 + pad))
    assert bbs[1] == pytest.approx((0.0 - pad, 1.0 + pad))
    assert bbs[2] == (0.0, 1.0)          # axial unchanged


# --- the construction-time warning ------------------------------------------

def test_dispersive_crossing_pml_warns():
    # A dispersive bar spanning the full z extent runs into the z PML.
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    with pytest.warns(UserWarning, match="dispersive .* extends into the PML"):
        sim = _sim([bar])
    # The warning is advisory; the gentle auto-alpha still applies on a crossing
    # scene. 0.1*(2*eps0/dt) — NOT the Tidy3D-parity 0.9, which at the default
    # 12 layers de-tunes the PML for the propagating guided mode exiting the
    # face and reflects ~35% of it (measured 2026-07-17, const-n identical to
    # dispersive => the alpha, not the ADE); 0.1 keeps the trapped-resonance
    # cure (~3x the measured threshold) with reflection back at ~1e-3.
    assert sim.pml_alpha_max == pytest.approx(0.1 * sim._two_eps0_over_dt())
    assert sim.pml_kappa_max == 5.0


def test_plain_dielectric_crossing_pml_does_not_warn():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_plain_medium(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning would fail
        _sim([bar])


def test_dispersive_fully_interior_auto_stabilizes():
    # A small dispersive cube far from every face does NOT trip the crossing
    # warning; on the default (CFS-inert) PML profile it now AUTO-STABILIZES
    # silently — kappa 5.0 + the GENTLE alpha 0.1*(2*eps0/dt), applied at
    # construction (the 2026-07-03 ladder measured divergence with the
    # dispersive structure 20 cells from the wall, subpixel OFF; the 2026-07-17
    # dose measurement lowered alpha 0.9 -> 0.1 after the Tidy3D-parity value
    # was found to reflect ~35% of a propagating guided mode:
    # engine/docs/subpixel-dispersion-instability.md).
    cube = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.2, 0.2, 0.2)),
        medium=_disp_medium(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # auto-apply is silent
        sim = _sim([cube])
    assert sim.pml_kappa_max == 5.0
    assert sim.pml_alpha_max == pytest.approx(0.1 * sim._two_eps0_over_dt())
    # Fit-safe levers only: the slab thickness and sigma peak are untouched, so
    # auto-stabilization can never over-thicken a small domain.
    assert sim.pml_num_layers == 12
    assert sim.pml_sigma_max == 1.5
    # The raised knobs ride the wire (the engine field default is CFS-inert);
    # the untouched layer count stays omitted.
    wire = sim.to_wire_dict()
    assert wire["pml_kappa_max"] == 5.0
    assert "pml_alpha_max" in wire
    assert "pml_num_layers" not in wire


def test_dispersive_interior_stabilized_pml_stays_silent():
    # The base dispersive scene already auto-stabilizes silently; layering the
    # explicit with_stabilized_pml() on top (the full StablePML profile, which
    # ALSO bumps the layer count) is likewise silent — its alpha is not inert.
    cube = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.2, 0.2, 0.2)),
        medium=_disp_medium(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim = _sim([cube])
        stable = sim.with_stabilized_pml()
    assert stable.pml_num_layers == 40         # the layer bump the auto path skips


def test_dispersive_explicit_pml_tuning_leaves_it_and_warns_if_inert():
    # If the user has explicitly tuned ANY PML knob, auto-stabilization steps
    # aside (they own the profile) — but a still-CFS-inert alpha gets the
    # advisory warning so the divergence lever is not silently unaddressed.
    cube = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.2, 0.2, 0.2)),
        medium=_disp_medium(),
    )
    with pytest.warns(UserWarning, match="CFS-inert"):
        sim = _sim([cube], pml_num_layers=16)
    assert sim.pml_num_layers == 16
    assert sim.pml_alpha_max == 0.24           # left at the inert default
    assert sim.pml_kappa_max == 3.0            # NOT auto-raised


def test_dispersive_explicit_high_alpha_no_warn_no_override():
    # An explicitly-raised alpha is respected verbatim and silences the advisory.
    cube = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.2, 0.2, 0.2)),
        medium=_disp_medium(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim = _sim([cube], pml_alpha_max=5.0e5)
    assert sim.pml_alpha_max == 5.0e5


def test_from_wire_does_not_auto_stabilize_dispersive_default_pml():
    # A saved dispersive document on the default (inert) PML is the user's
    # deliberate choice — ingesting it must NOT auto-raise alpha/kappa (that
    # would break byte-identical round-trip) and must not warn.
    import json

    cube_geo = ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.2, 0.2, 0.2))
    # A non-dispersive twin serialises with the bare default PML (no alpha/kappa
    # keys); transplant the REAL serialized dispersive medium into it to forge a
    # dispersive doc that still carries the bare default PML.
    twin = _sim([ph.Structure(geometry=cube_geo, medium=_plain_medium())])
    d = twin.to_wire_dict()
    assert "pml_alpha_max" not in d and "pml_kappa_max" not in d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp = _sim([ph.Structure(geometry=cube_geo, medium=_disp_medium())])
    d["structures"][0]["medium"] = disp.to_wire_dict()["structures"][0]["medium"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = ph.Simulation.from_wire_json(json.dumps(d))
    assert back.pml_alpha_max == 0.24 and back.pml_kappa_max == 3.0


def test_dispersive_interior_no_pml_faces_does_not_warn():
    # All-periodic boundaries: no PML anywhere -> no CFS warning.
    cube = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.2, 0.2, 0.2)),
        medium=_disp_medium(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _sim([cube], boundaries=ph.Boundaries(x="periodic", y="periodic",
                                              z="periodic"))


def test_plain_dielectric_interior_default_pml_does_not_warn():
    cube = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.2, 0.2, 0.2)),
        medium=_plain_medium(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _sim([cube])


def test_dispersive_crossing_absorber_axis_auto_stabilizes_silently():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    # z already on the absorber -> no CROSSING warning for any axis (x/y PML are
    # not crossed by this z-running bar). The dispersive scene + surviving x/y
    # PML faces auto-stabilize silently -> no warning of any kind.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim = _sim([bar], boundaries=ph.Boundaries(x="pml", y="pml", z="absorber"))
    msgs = [str(w.message) for w in rec]
    assert not any("extends into the PML" in m for m in msgs)
    assert not any("CFS-inert" in m for m in msgs)
    assert sim.pml_kappa_max == 5.0
    assert sim.pml_alpha_max == pytest.approx(0.1 * sim._two_eps0_over_dt())


def test_warning_only_for_the_crossed_axis():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    with pytest.warns(UserWarning) as rec:
        _sim([bar])
    crossing = [str(w.message) for w in rec if "extends into the PML" in
                str(w.message)]
    # Only the crossed axis (z) is named; x/y are not flagged.
    assert len(crossing) == 1 and "on axis 'z':" in crossing[0]


# --- with_auto_boundaries ---------------------------------------------------

def test_auto_boundaries_picks_absorber_only_on_crossed_axis():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    with pytest.warns(UserWarning):
        sim = _sim([bar])
    auto = sim.with_auto_boundaries()
    assert auto.boundaries.z == "absorber"   # dispersive bar crosses z
    assert auto.boundaries.x == "pml"         # bar does not reach x/y faces
    assert auto.boundaries.y == "pml"


def test_auto_boundaries_plain_dielectric_stays_pml():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_plain_medium(),
    )
    auto = _sim([bar]).with_auto_boundaries()
    assert (auto.boundaries.x, auto.boundaries.y, auto.boundaries.z) == (
        "pml", "pml", "pml")


def test_auto_boundaries_preserves_periodic_and_pec():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    # x periodic, y pec are explicit physics -> untouched even though the auto
    # rule only ever considers open axes; z (pml) flips to absorber.
    with pytest.warns(UserWarning):
        sim = _sim([bar], boundaries=ph.Boundaries(x="periodic", y="pec", z="pml"))
    auto = sim.with_auto_boundaries()
    assert auto.boundaries.x == "periodic"
    assert auto.boundaries.y == "pec"
    assert auto.boundaries.z == "absorber"


def test_auto_boundaries_does_not_change_wire_for_plain_scene():
    # No dispersive crossing -> identical boundaries -> byte-identical wire.
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_plain_medium(),
    )
    sim = _sim([bar])
    assert sim.with_auto_boundaries().to_wire_json() == sim.to_wire_json()


def test_from_wire_does_not_warn_on_dispersive_crossing():
    # Ingesting a saved document is the user's deliberate choice (like the §16
    # subpixel default) -> the construction-time advice is skipped.
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    with pytest.warns(UserWarning):
        sim = _sim([bar])
    wire = sim.to_wire_json()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = ph.Simulation.from_wire_json(wire)
    assert back.boundaries.z == "pml"
