"""FDE mode-solver plugin — SOI strip validation + frozen public API surface.

Validates ``photonhub.plugins.ModeSolver`` against the known silicon-on-insulator
strip waveguide (the same case the benchmark FDTD run cross-checked): a
450 nm x 220 nm Si core (n = 3.5) in SiO2 (n = 1.444) at lambda = 1310 nm has a
fundamental quasi-TE (TE0) mode with n_eff ~= 2.7 (the semi-vectorial solver's own
value is ~2.72). Also pins the public API so the "frozen API" contract is
enforced by CI, and checks that bent waveguides are explicitly refused.
"""

import inspect

import numpy as np
import pytest
import xarray as xr

from photonhub.plugins import Mode, ModeSolver

# --- SOI strip @ 1310 nm (the validated cross-check case) -------------------
WL_UM = 1.31
DL_UM = 0.025
CORE_W_UM, CORE_H_UM = 0.45, 0.22
N_SI, N_SIO2 = 3.5, 1.444


@pytest.fixture(scope="module")
def soi_solver() -> ModeSolver:
    return ModeSolver.from_rectangular_core(
        wavelength_um=WL_UM, dl_um=DL_UM,
        core_w_um=CORE_W_UM, core_h_um=CORE_H_UM,
        n_core=N_SI, n_clad=N_SIO2)


@pytest.fixture(scope="module")
def soi_te0(soi_solver: ModeSolver) -> Mode:
    return soi_solver.solve(num_modes=1, polarization="TE")[0]


# --- physics validation -----------------------------------------------------

def test_soi_te0_neff_matches_reference(soi_te0: Mode):
    """Fundamental quasi-TE index ~= 2.7 (mode-solver self-value ~2.72; the
    FDTD cross-check gave 2.700). Tolerance covers both."""
    assert soi_te0.n_eff == pytest.approx(2.72, abs=0.05)
    # Must be a genuinely guided mode: cladding index < n_eff < core index.
    assert N_SIO2 < soi_te0.n_eff < N_SI


def test_soi_te0_profile_is_confined(soi_te0: Mode):
    """The TE0 field is a single positive lobe well confined to the core."""
    field = soi_te0.field
    assert field.shape == soi_te0.shape
    # L2-normalized, sign-fixed positive lobe.
    assert float((field ** 2).sum()) == pytest.approx(1.0, abs=1e-9)
    assert field.max() > 0
    # Most of the power sits in the core bounding box (benchmark: ~0.8).
    conf = soi_te0.core_fraction(CORE_W_UM, CORE_H_UM)
    assert conf > 0.6
    # The peak is near the cross-section center (the core), not at a wall.
    iy, ix = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    ny, nx = field.shape
    assert abs(iy - (ny - 1) / 2) < ny * 0.15
    assert abs(ix - (nx - 1) / 2) < nx * 0.15


def test_raw_eps_constructor_matches_builder(soi_solver: ModeSolver,
                                             soi_te0: Mode):
    """The raw-eps constructor and the rasterizing builder agree on n_eff for
    an identical cross-section (the builder is just sugar). Reuses the cached
    builder solver/mode to avoid a redundant dense solve."""
    raw = ModeSolver(soi_solver.eps, soi_solver.dl_x_um,
                     soi_solver.dl_y_um, soi_solver.wavelength_um)
    assert raw.solve()[0].n_eff == pytest.approx(soi_te0.n_eff, abs=1e-9)


def test_multiple_modes_are_ordered_by_descending_neff(soi_solver: ModeSolver):
    modes = soi_solver.solve(num_modes=3, polarization="TE")
    assert len(modes) >= 1
    neffs = [m.n_eff for m in modes]
    assert neffs == sorted(neffs, reverse=True)
    # Fundamental dominates.
    assert neffs[0] == pytest.approx(2.72, abs=0.05)


def test_scalar_overestimates_and_tm_underconfines(soi_solver: ModeSolver,
                                                   soi_te0: Mode):
    """Sanity of the operator branches: scalar (no vectorial correction)
    over-estimates the high-contrast index vs the semi-vectorial TE branch."""
    scalar = soi_solver.solve(polarization="scalar")[0]
    assert scalar.n_eff > soi_te0.n_eff


def test_field_dataarray_export(soi_te0: Mode):
    da = soi_te0.field_dataarray()
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("y", "x")
    assert da.name == "Ex"  # TE -> Ex-major
    assert set(("n_eff", "wavelength_um", "polarization")) <= set(da.attrs)
    # Real-space coords in microns, centered on the origin.
    assert float(da.coords["x"].values.mean()) == pytest.approx(0.0, abs=1e-9)
    assert float(da.coords["y"].values.mean()) == pytest.approx(0.0, abs=1e-9)


# --- straight-waveguide-only / bent-mode exclusion (roadmap requirement) -----

def test_bent_waveguide_is_refused(soi_solver: ModeSolver):
    """Bent modes are out of scope; a bend radius must raise, not silently
    return a curvature-free (wrong) result."""
    with pytest.raises(NotImplementedError):
        soi_solver.solve(bend_radius_um=5.0)


def test_bent_mode_exclusion_is_documented():
    """The roadmap requires the straight-only scope to be documented in the
    module/class docstring."""
    import photonhub.plugins.modes as modes_mod
    mod_doc = (modes_mod.__doc__ or "").lower()
    cls_doc = (ModeSolver.__doc__ or "").lower()
    assert "straight" in mod_doc
    assert "bent" in mod_doc or "bend" in mod_doc
    # Error bound for misuse on a bend is spelled out.
    assert "error bound" in mod_doc or "n_eff error" in mod_doc
    assert "straight" in cls_doc


# --- input validation -------------------------------------------------------

def test_constructor_rejects_bad_eps():
    with pytest.raises(ValueError):
        ModeSolver(np.ones(10), DL_UM, DL_UM, WL_UM)            # not 2-D
    with pytest.raises(ValueError):
        ModeSolver(np.full((4, 4), 0.5), DL_UM, DL_UM, WL_UM)   # eps < 1
    with pytest.raises(ValueError):
        ModeSolver(np.ones((4, 4)), 0.0, DL_UM, WL_UM)          # dl <= 0
    bad = np.ones((4, 4)); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        ModeSolver(bad, DL_UM, DL_UM, WL_UM)                    # non-finite


def test_solve_rejects_bad_args(soi_solver: ModeSolver):
    with pytest.raises(ValueError):
        soi_solver.solve(num_modes=0)
    with pytest.raises(ValueError):
        soi_solver.solve(polarization="bogus")  # type: ignore[arg-type]


def test_oversized_grid_is_capped():
    """A grid beyond the dense-solver cap fails fast with a clear error rather
    than exhausting memory."""
    n = int(np.sqrt(ModeSolver.MAX_UNKNOWNS)) + 20
    eps = np.full((n, n), N_SIO2 ** 2)
    ms = ModeSolver(eps, DL_UM, DL_UM, WL_UM)
    with pytest.raises(ValueError):
        ms.solve()


# --- FROZEN PUBLIC API SURFACE ----------------------------------------------
# These tests pin the Phase-1 contract: changing a signature or dropping an
# attribute here is a deliberate API break, not an accident.

def test_frozen_modesolver_constructor_signature():
    params = list(inspect.signature(ModeSolver.__init__).parameters)
    assert params == ["self", "eps", "dl_x_um", "dl_y_um", "wavelength_um"]


def test_frozen_from_rectangular_core_signature():
    sig = inspect.signature(ModeSolver.from_rectangular_core)
    params = list(sig.parameters)
    assert params == [
        "wavelength_um", "dl_um", "core_w_um", "core_h_um",
        "n_core", "n_clad", "window_w_um", "window_h_um", "clad_pad_um",
    ]
    # window/pad are optional; the rest are keyword-only required.
    assert sig.parameters["window_w_um"].default is None
    assert sig.parameters["window_h_um"].default is None


def test_frozen_solve_signature():
    sig = inspect.signature(ModeSolver.solve)
    params = list(sig.parameters)
    assert params == [
        "self", "num_modes", "polarization", "n_guess", "bend_radius_um"]
    assert sig.parameters["num_modes"].default == 1
    assert sig.parameters["polarization"].default == "TE"
    assert sig.parameters["n_guess"].default is None
    assert sig.parameters["bend_radius_um"].default is None


def test_frozen_mode_attributes(soi_te0: Mode):
    # Public Mode surface the rest of the system relies on.
    for attr in ("n_eff", "field", "wavelength_um", "polarization",
                 "dl_x_um", "dl_y_um", "shape"):
        assert hasattr(soi_te0, attr)
    assert isinstance(soi_te0.n_eff, float)
    assert isinstance(soi_te0.field, np.ndarray)
    assert callable(soi_te0.field_dataarray)
    assert callable(soi_te0.core_fraction)
    # Frozen dataclass: attributes are immutable.
    with pytest.raises(Exception):
        soi_te0.n_eff = 1.0  # type: ignore[misc]


def test_frozen_top_level_exports():
    import photonhub.plugins as plugins
    # The FDE solver (Mode/ModeSolver) is the frozen Phase-1 surface; Phase-2
    # added the mode source/monitor/overlap builders alongside it.
    assert {"Mode", "ModeSolver"} <= set(plugins.__all__)
    from photonhub.plugins import Mode as M, ModeSolver as MS  # noqa: F401
    assert {
        "ModeMonitor",
        "mode_monitor",
        "mode_source",
        "mode_transmission",
        "transmission",
    } <= set(plugins.__all__)
