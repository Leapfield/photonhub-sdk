"""Subpixel-smoothing flag (NUMERICS.md §16, schema 1.4.0): additive-optional
on the wire — omitted when unset (byte-identical documents), present when set,
and strictly round-tripping. The engine's smoothing math is covered by the
C++ test_subpixel.cpp; here we only pin the client/wire surface.
"""

import json

import photonhub as ph

from .helpers import make_sim


def test_default_is_tensor_on_for_nondispersive():
    # D2 (NUMERICS.md §16/§16.11): a non-dispersive sim with no explicit subpixel
    # choice defaults to subpixel-ON with "contour" (diagonal KFJ fed the exact
    # §16.10 PolySlab fill == Tidy3D's default PolarizedAveraging; reduces to the
    # diagonal tensor on axis-aligned interfaces), serialized explicitly on the wire.
    sim = make_sim()
    assert sim.subpixel is True
    assert sim.subpixel_method == "contour"
    wire = sim.to_wire_dict()
    assert wire["subpixel"] is True
    assert wire["subpixel_method"] == "contour"


def test_default_falls_back_to_off_for_dispersive():
    # D2 auto-fallback: a dispersive (Lorentz) scene defaults to subpixel OFF
    # (the subpixel × ADE late-time instability) when not set explicitly.
    pole = ph.LorentzPole(resonance_frequency_hz=2.0e14, delta_eps=1.5)
    box = ph.Structure(
        geometry=ph.Box(center_um=(0.1, 0.1, 0.1), size_um=(0.1, 0.1, 0.1)),
        medium=ph.Medium(permittivity=2.0, lorentz=pole),
    )
    sim = make_sim(structures=[box])
    assert sim.subpixel is False
    assert "subpixel" not in sim.to_wire_dict()


def test_explicit_subpixel_on_dispersive_still_states_its_method_on_the_wire():
    # Regression: an EXPLICIT subpixel=True on a dispersive (Lorentz) scene used
    # to skip the method fill (it was gated on `not dispersive`), so the model
    # REPORTED subpixel_method="contour" (the field default) while the wire OMITTED
    # it and the engine silently applied ITS default, "volume" — an isotropic
    # linear average instead of the diagonal KFJ advertised, a first-order operator
    # error on every partially-filled interface cell. What the model reports and
    # what the engine runs must agree; the divergence warning is separate.
    import pytest

    pole = ph.LorentzPole(resonance_frequency_hz=2.0e14, delta_eps=1.5)
    box = ph.Structure(
        geometry=ph.Box(center_um=(0.1, 0.1, 0.1), size_um=(0.1, 0.1, 0.1)),
        medium=ph.Medium(permittivity=2.0, lorentz=pole),
    )
    with pytest.warns(UserWarning, match="dispersive"):
        sim = make_sim(structures=[box], subpixel=True)
    assert sim.subpixel is True
    assert sim.subpixel_method == "contour"
    wire = sim.to_wire_dict()
    assert wire["subpixel"] is True
    assert wire["subpixel_method"] == "contour"  # was: omitted -> engine ran "volume"


def test_explicit_subpixel_false_is_respected_when_unset_default_would_enable():
    # An explicit choice always wins over the D2 default.
    sim = make_sim(subpixel=False)
    assert sim.subpixel is False
    assert sim.to_wire_dict()["subpixel"] is False


def test_wire_ingestion_does_not_apply_the_construction_default():
    # Round-trip fidelity: a document that OMITS subpixel means the engine
    # default (off); from_wire_json must NOT flip it to the D2 construction
    # default, or older docs would stop round-tripping byte-identically.
    doc = json.loads(make_sim(subpixel=False).to_wire_json())
    doc.pop("subpixel", None)
    doc.pop("subpixel_method", None)
    back = ph.Simulation.from_wire_json(json.dumps(doc))
    assert back.subpixel is False
    assert "subpixel" not in back.to_wire_dict()


def test_set_true_appears_on_the_wire():
    sim = make_sim(subpixel=True)
    assert sim.subpixel is True
    wire = sim.to_wire_dict()
    assert wire["subpixel"] is True


def test_set_false_explicitly_still_round_trips():
    # Explicitly set (even to the default) -> in model_fields_set -> serialized,
    # so an explicit choice survives the wire.
    sim = make_sim(subpixel=False)
    wire = json.loads(sim.to_wire_json())
    assert wire["subpixel"] is False
    back = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert back.subpixel is False


def test_from_wire_parses_subpixel():
    sim = make_sim(subpixel=True)
    back = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert back.subpixel is True
    assert back.schema_version == ph.SCHEMA_VERSION


def test_wire_ingestion_is_strict_about_bool():
    # from_wire_json uses strict typing to match the engine's nlohmann parse
    # (spec_io.cpp requires a JSON boolean), so a string/number in the wire is
    # rejected — even though lax construction would coerce "yes"/1 to True.
    import pytest
    from pydantic import ValidationError

    base = json.loads(make_sim().to_wire_json())
    for bad in ("true", 1):
        doc = dict(base, subpixel=bad)
        with pytest.raises(ValidationError):
            ph.Simulation.from_wire_json(json.dumps(doc))


# --- subpixel_method selector (schema 1.7.0, NUMERICS.md §16.5) ---------------


def test_subpixel_method_default_is_tensor_when_auto_enabled():
    # D2/§16.11: when the construction default enables subpixel (non-dispersive),
    # the method it selects is "contour" (Tidy3D's default PolarizedAveraging fed
    # the exact §16.10 fill; not the isotropic volume, not the CP-EP contour_diag).
    sim = make_sim()
    assert sim.subpixel_method == "contour"
    assert sim.to_wire_dict()["subpixel_method"] == "contour"


def test_explicit_subpixel_true_fills_contour_like_the_auto_default():
    # An explicit subpixel=True with the method UNSET must resolve to the SAME
    # operator as the auto-default (contour) — the two subpixel-on paths stay
    # in lockstep, not tensor-vs-contour.
    sim = make_sim(subpixel=True)
    assert sim.subpixel_method == "contour"
    assert sim.to_wire_dict()["subpixel_method"] == "contour"


def test_subpixel_method_tensor_appears_on_the_wire():
    sim = make_sim(subpixel=True, subpixel_method="tensor")
    assert sim.subpixel_method == "tensor"
    wire = sim.to_wire_dict()
    assert wire["subpixel_method"] == "tensor"


def test_subpixel_method_round_trips_from_wire():
    sim = make_sim(subpixel=True, subpixel_method="tensor")
    back = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert back.subpixel_method == "tensor"


def test_subpixel_method_rejects_unknown_value():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_sim(subpixel=True, subpixel_method="bogus")
