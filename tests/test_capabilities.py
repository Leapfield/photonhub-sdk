"""The client's pinned ``--capabilities`` view stays aligned with the engine."""

import pytest

import photonhub as ph
from photonhub import capabilities as caps
from photonhub.components.grid import GradedAxisCoords, GradedGridSpec

from .helpers import make_pw_sim, make_sim


def _graded_z(n=28, dl0=0.025, ratio=1.04):
    """A seam-symmetric graded coordinate array (origin 0), as §15.1/§15.2."""
    q, z = [0.0], 0.0
    for k in range(1, n):
        ramp = min(k - 1, n - 1 - k)
        z += dl0 * (ratio ** ramp)
        q.append(round(z, 7))
    return tuple(q)


def test_graded_propagation_axis_plane_wave_builds():
    sim = make_pw_sim(
        size_um=(0.2, 0.2, 1.0),
        grid=GradedGridSpec(dl_um=0.05, coords=GradedAxisCoords(z=_graded_z())),
    )
    assert isinstance(sim.grid, GradedGridSpec)


def test_transverse_only_graded_plane_wave_builds():
    sim = make_pw_sim(
        size_um=(1.0, 0.2, 1.0),
        grid=GradedGridSpec(dl_um=0.05, coords=GradedAxisCoords(x=_graded_z())),
    )
    assert isinstance(sim.grid, GradedGridSpec)


def test_uniform_plane_wave_still_builds():
    # Regression: the gate must not reject a perfectly valid uniform plane wave.
    sim = make_pw_sim()
    assert any(isinstance(s, ph.PlaneWave) for s in sim.sources)


def test_graded_grid_plus_point_dipole_still_builds():
    # Only plane waves are gated on a graded grid; a dipole launch is fine
    # (the silicon-waveguide path depends on exactly this).
    sim = make_sim(
        size_um=(0.2, 0.2, 1.0),
        grid=GradedGridSpec(dl_um=0.05, coords=GradedAxisCoords(z=_graded_z())),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
        pml_num_layers=8,
        monitors=[],
    )
    assert isinstance(sim.grid, GradedGridSpec)


# --- the live engine manifest (the "wired to --capabilities" link) -------- #

def test_engine_manifest_matches_the_pinned_set():
    drift = caps.engine_feature_drift()
    if drift is None:
        pytest.skip("no phsolver binary available to read --capabilities")
    assert drift == set(), (
        "phsolver --capabilities drifted from "
        "capabilities.ENGINE_ADVERTISED_FEATURES; update the pin (and the "
        f"client gates) in lockstep. Symmetric difference: {sorted(drift)}")


def test_engine_capabilities_reports_schema_major():
    info = caps.engine_capabilities()
    if info is None:
        pytest.skip("no phsolver binary available")
    assert info.get("schema_major") == caps.SCHEMA_MAJOR


def test_modal_port_is_an_explicit_negotiated_capability():
    assert "mode_port" in caps.ENGINE_ADVERTISED_FEATURES
    info = caps.engine_capabilities()
    if info is None:
        pytest.skip("no phsolver binary available")
    assert "mode_port" in info.get("features", [])
