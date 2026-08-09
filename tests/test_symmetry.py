"""Symmetry plane API — NUMERICS.md §20 (schema 1.11.0).

The `symmetry` field places a mirror plane on an axis' MINIMUM face: -1 odd /
electric (PEC), 0 none, +1 even / magnetic (PMC; all three axes). Covers the wire
round-trip (additive-optional: all-zero omitted), and the §20.2/§20.5
construction-time rejections that mirror the engine's validate().
"""

import pytest

import photonhub as ph
from .helpers import make_pw_sim, make_sim


def _sym_sim(symmetry=(0, -1, 0),
            boundaries=ph.Boundaries(x="periodic", y="pml", z="pec"),
            **kw):
    """A small valid odd-symmetry scene: y is the symmetry axis (PML far face),
    a point dipole drives it. Overridable for the rejection cases."""
    return make_sim(size_um=(0.3, 0.6, 0.3), boundaries=boundaries,
                    symmetry=symmetry, monitors=(), **kw)


class TestSymmetryWire:
    def test_default_is_none_and_omitted_from_wire(self):
        sim = make_sim()
        assert sim.symmetry == (0, 0, 0)
        assert "symmetry" not in sim.to_wire_dict()

    def test_odd_plane_serializes_as_array_and_round_trips(self):
        sim = _sym_sim(symmetry=(0, -1, 0))
        wire = sim.to_wire_dict()
        assert wire["symmetry"] == [0, -1, 0]
        assert wire["schema_version"] == ph.SCHEMA_VERSION
        back = ph.Simulation.from_wire_json(sim.to_wire_json())
        assert back.symmetry == (0, -1, 0)

    def test_explicit_zero_symmetry_still_omitted(self):
        # An explicitly-set all-zero symmetry is a no-op ⇒ omitted (back-compat).
        sim = _sym_sim(symmetry=(0, 0, 0),
                       boundaries=ph.Boundaries(x="periodic", y="periodic",
                                                z="periodic"))
        assert "symmetry" not in sim.to_wire_dict()


class TestSymmetryAccepts:
    def test_pml_far_face(self):
        _sym_sim(boundaries=ph.Boundaries(x="periodic", y="pml", z="pec"))

    def test_pec_far_face(self):
        # A PEC far face is allowed (§20.2) — the symmetry plane is the min face.
        _sym_sim(boundaries=ph.Boundaries(x="periodic", y="pec", z="pec"))

    def test_two_symmetry_planes(self):
        # Quarter domain: odd planes on x-min and y-min, PML far faces.
        _sym_sim(symmetry=(-1, -1, 0),
                 boundaries=ph.Boundaries(x="pml", y="pml", z="pec"))

    def test_even_pmc_on_x_and_y(self):
        # +1 (even / magnetic, PMC) is supported on x and y (§20.4).
        _sym_sim(symmetry=(0, 1, 0))
        _sym_sim(symmetry=(1, 0, 0),
                 boundaries=ph.Boundaries(x="pml", y="periodic", z="pec"))
        _sym_sim(symmetry=(1, 1, 0),
                 boundaries=ph.Boundaries(x="pml", y="pml", z="pec"))

    def test_even_pmc_on_z_accepted(self):
        # z-axis PMC is now supported on the CPU reference solver (§20.4) via
        # the negated k=-1 ghost-plane mirror; construction must succeed.
        sim = _sym_sim(symmetry=(0, 0, 1),
                       boundaries=ph.Boundaries(x="periodic", y="periodic",
                                                z="pml"))
        assert sim.symmetry == (0, 0, 1)
        assert sim.to_wire_dict()["symmetry"] == [0, 0, 1]


class TestSymmetryRejects:

    def test_out_of_range_value(self):
        with pytest.raises(Exception) as ei:
            _sym_sim(symmetry=(0, -2, 0))
        assert "symmetry" in str(ei.value)

    def test_periodic_axis(self):
        with pytest.raises(Exception) as ei:
            _sym_sim(boundaries=ph.Boundaries(x="periodic", y="periodic",
                                              z="pec"))
        assert "periodic" in str(ei.value)

    def test_plane_wave_incompatible(self):
        # A plane wave in z; a symmetry plane on z. Transverse x/y periodic so
        # the only violation is the §20.2 symmetry-vs-plane-wave rule.
        with pytest.raises(Exception) as ei:
            make_pw_sim(boundaries=ph.Boundaries(x="periodic", y="periodic",
                                                 z="pml"),
                        symmetry=(0, 0, -1))
        assert "plane-wave" in str(ei.value) or "plane wave" in str(ei.value)
