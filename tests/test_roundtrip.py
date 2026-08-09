"""Golden-corpus round trip (schemas/GOVERNANCE.md rule 4) and engine-typing
parity of the wire-JSON ingestion path."""

import json

import pytest
from pydantic import ValidationError

import photonhub as ph


def test_example_parses_and_roundtrips(example_spec_path):
    text = example_spec_path.read_text(encoding="utf-8")
    sim = ph.Simulation.from_wire_json(text)

    assert sim.grid.dl_um == 0.05
    assert sim.run.run_time_s == 8.0e-14
    assert sim.sources[0].polarization == "Ez"
    assert sim.sources[0].source_time.freq0_hz == 1.934e14
    assert [m.name for m in sim.monitors] == ["probe", "final_fields"]

    # Re-serialization must reproduce the canonical wire dict exactly.
    assert sim.to_wire_dict() == json.loads(text)


def test_wire_json_text_roundtrips(example_spec_path):
    sim = ph.Simulation.from_wire_json(example_spec_path.read_text())
    again = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert again == sim
    assert again.to_wire_dict() == sim.to_wire_dict()


def test_fresnel_example_parses_and_roundtrips(fresnel_spec_path):
    """Golden-corpus round trip of the Phase 1a-1 wire format (plane wave,
    structures, PML, DFT/flux monitors) to canonical dict equality."""
    text = fresnel_spec_path.read_text(encoding="utf-8")
    sim = ph.Simulation.from_wire_json(text)

    # Backward-compat: a 1.1.0 golden document still parses under the 1.2.0
    # client and its version is preserved on round-trip (GOVERNANCE rule 2/4).
    assert sim.schema_version == "1.1.0-alpha.1"
    assert sim.pml_num_layers == 12
    assert sim.boundaries.z == "pml"

    (slab,) = sim.structures
    assert isinstance(slab.geometry, ph.Box)
    assert slab.geometry.size_um == (10.0, 10.0, 0.25)
    assert slab.medium.permittivity == 12.0
    assert slab.medium.conductivity_s_per_m == 0.0

    (src,) = sim.sources
    assert isinstance(src, ph.PlaneWave)
    assert (src.axis, src.direction, src.polarization) == ("z", "+", "Ex")
    assert src.position_um == 0.8
    assert src.amplitude == 1.0

    assert [m.name for m in sim.monitors] == [
        "reflection", "transmission", "slab_fields"]
    assert isinstance(sim.monitors[0], ph.FluxMonitor)
    assert sim.monitors[0].freqs_hz[2] == 1.934e14
    assert isinstance(sim.monitors[2], ph.FieldDftMonitor)
    assert sim.monitors[2].fields == ("Ex", "Hy")

    # Re-serialization must reproduce the canonical wire dict exactly
    # (pml_num_layers was explicitly present, so it stays on the wire).
    assert sim.to_wire_dict() == json.loads(text)


def test_fresnel_wire_json_text_roundtrips(fresnel_spec_path):
    sim = ph.Simulation.from_wire_json(fresnel_spec_path.read_text())
    again = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert again == sim
    assert again.to_wire_dict() == sim.to_wire_dict()


def test_unused_run_key_is_omitted_on_the_wire(tiny_sim):
    wire = tiny_sim.to_wire_dict()
    assert wire["run"] == {"n_steps": 5, "courant": 0.99}
    assert "run_time_s" not in wire["run"]
    assert wire["structures"] == []


def test_unset_pml_num_layers_is_omitted_on_the_wire(tiny_sim):
    # Phase-0 documents must round-trip byte-identically and stay consumable
    # by schema-1.0 parsers that reject unknown keys.
    assert "pml_num_layers" not in tiny_sim.to_wire_dict()

    from .helpers import make_sim
    explicit = make_sim(pml_num_layers=12)
    assert explicit.to_wire_dict()["pml_num_layers"] == 12


def test_unset_cpml_profile_is_omitted_on_the_wire(tiny_sim):
    # The §11 CPML profile knobs (schema 1.8.0) are additive-optional and their
    # defaults reproduce the prior hardcoded constants bit-for-bit; an unset
    # value is omitted so earlier-minor documents round-trip byte-identically.
    wire = tiny_sim.to_wire_dict()
    for f in ("pml_m", "pml_kappa_max", "pml_alpha_max"):
        assert f not in wire

    from .helpers import make_sim
    explicit = make_sim(pml_m=4.0, pml_kappa_max=7.0, pml_alpha_max=0.05)
    ewire = explicit.to_wire_dict()
    assert ewire["pml_m"] == 4.0
    assert ewire["pml_kappa_max"] == 7.0
    assert ewire["pml_alpha_max"] == 0.05


def test_with_stabilized_pml_raises_layers_kappa_and_alpha(tiny_sim):
    stable = tiny_sim.with_stabilized_pml()
    # Defaults reproduce Tidy3D's StablePML (40 layers, kappa 5, alpha 0.9).
    assert stable.pml_num_layers == 40
    assert stable.pml_kappa_max == 5.0
    # The lever the old with_stable_pml missed: alpha is RAISED, quoted in
    # Tidy3D's 2*eps0/dt units (alpha_scale 0.9) and converted to S/m at this
    # scene's dt — far above the 0.24 S/m default nudge — so it now rides the wire.
    two_eps0_over_dt = tiny_sim._two_eps0_over_dt()
    assert stable.pml_alpha_max == pytest.approx(0.9 * two_eps0_over_dt)
    assert stable.pml_alpha_max > 100.0  # >> the 0.24 default
    swire = stable.to_wire_dict()
    assert swire["pml_num_layers"] == 40
    assert swire["pml_kappa_max"] == 5.0
    assert swire["pml_alpha_max"] == pytest.approx(0.9 * two_eps0_over_dt)
    assert "pml_m" not in swire
    # Opt-in only: the original (non-dispersive) sim is unchanged.
    assert tiny_sim.pml_num_layers == 12 and tiny_sim.pml_alpha_max == 0.24

    custom = tiny_sim.with_stabilized_pml(num_layers=32, kappa_max=8.0,
                                          alpha_scale=0.04)
    assert custom.pml_num_layers == 32 and custom.pml_kappa_max == 8.0
    assert custom.pml_alpha_max == pytest.approx(0.04 * two_eps0_over_dt)


def test_unset_shutoff_is_omitted_on_the_wire(tiny_sim):
    # run.shutoff (schema 1.3.0, NUMERICS.md section 7) is additive-optional:
    # an unset value is omitted so pre-1.3 documents round-trip unchanged and
    # the engine applies its own default (1e-5).
    assert "shutoff" not in tiny_sim.to_wire_dict()["run"]

    from .helpers import make_sim
    explicit = make_sim(run=ph.RunSpec(n_steps=5, shutoff=0.0))
    assert explicit.to_wire_dict()["run"]["shutoff"] == 0.0


def _mutate(text: str, old: str, new: str) -> str:
    assert old in text, f"fixture drift: {old!r} not in the golden example"
    return text.replace(old, new)


class TestWireTypingParity:
    """from_wire_json must reject exactly the type coercions phsolver's
    nlohmann parser rejects (str -> number, float -> int) while accepting
    what it accepts (JSON int for float fields)."""

    @pytest.fixture
    def text(self, example_spec_path):
        return example_spec_path.read_text(encoding="utf-8")

    def test_json_int_for_float_field_accepted(self, text):
        # nlohmann's is_number accepts integers for double fields.
        sim = ph.Simulation.from_wire_json(
            _mutate(text, '"size_um": [4.0, 4.0, 4.0]', '"size_um": [4, 4, 4]'))
        assert sim.size_um == (4.0, 4.0, 4.0)

    def test_string_number_rejected(self, text):
        with pytest.raises(ValidationError):
            ph.Simulation.from_wire_json(
                _mutate(text, '"courant": 0.99', '"courant": "0.99"'))

    def test_string_center_component_rejected(self, text):
        with pytest.raises(ValidationError):
            ph.Simulation.from_wire_json(
                _mutate(text, '"center_um": [2.0, 2.0, 2.0]',
                        '"center_um": ["2.0", 2.0, 2.0]'))

    def test_float_for_int_field_rejected(self, text):
        with pytest.raises(ValidationError):
            ph.Simulation.from_wire_json(
                _mutate(text, '"interval_steps": 1', '"interval_steps": 1.0'))

    def test_string_int_rejected(self, text):
        mutated = _mutate(text, '"run_time_s": 8.0e-14', '"n_steps": "10"')
        with pytest.raises(ValidationError):
            ph.Simulation.from_wire_json(mutated)

    def test_n_steps_beyond_int32_rejected(self, text):
        mutated = _mutate(text, '"run_time_s": 8.0e-14',
                          f'"n_steps": {2**31}')
        with pytest.raises(ValidationError):
            ph.Simulation.from_wire_json(mutated)

    def test_explicit_null_run_key_treated_as_absent(self, text):
        # Engine parity (spec_io.cpp): an explicit null counts as "not
        # given" for the exactly-one-of rule.
        sim = ph.Simulation.from_wire_json(
            _mutate(text, '"run_time_s": 8.0e-14',
                    '"run_time_s": null, "n_steps": 10'))
        assert sim.run.n_steps == 10
        assert sim.run.run_time_s is None
