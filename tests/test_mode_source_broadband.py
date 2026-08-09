"""Broadband mode source — the Tidy3D ``num_freqs`` analogue (NUMERICS.md
§18.3). The engine injects a frequency-dependent modal profile/n_eff windowed
across the band; here we cover the Python contract:

* the additive ``ModeSource`` broadband fields and their validators;
* the wire format (omitted when unset → byte-identical to 1.10; carried when set);
* the ``mode_source`` / ``mode_source_vector`` builders packing a
  ``{freq_hz: Mode}`` map (sign-aligned, ascending, parallel arrays).

Pure Python (no engine) — the time-domain injection is verified on the GPU box."""

import numpy as np
import pytest
from pydantic import ValidationError

import photonhub as ph
from photonhub.components.sources import ModeSource
from photonhub.plugins import (
    ModeSolver,
    VectorModeSolver,
    mode_source,
    mode_source_vector,
    solve_modes_by_freq,
)
from photonhub.plugins.mode_devices import C0

DL = 0.04
N_CORE = 3.5
N_CLAD = 1.444
WLS = (1.26, 1.31, 1.36)
FREQS = tuple(C0 / (w * 1e-6) for w in WLS)  # descending (shorter wl = higher f)
FSORT = tuple(sorted(FREQS))  # ascending — the order the builder/engine use


def _shell():
    pulse = ph.GaussianPulse(freq0_hz=FREQS[1], fwidth_hz=4e13)
    sim = ph.Simulation(
        size_um=(1.6, 1.6, 4.0),
        grid=ph.UniformGridSpec(dl_um=DL),
        run=ph.RunSpec(n_steps=10),
        background=ph.Background(permittivity=N_CLAD**2),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        sources=[ph.PointDipole(center_um=(0.8, 0.8, 0.5),
                                polarization="Ey", source_time=pulse)],
    )
    return sim, pulse


def _scalar_modes():
    s = ModeSolver.from_rectangular_core(
        wavelength_um=1.31, dl_um=DL, core_w_um=0.45, core_h_um=0.22,
        n_core=N_CORE, n_clad=N_CLAD)
    return solve_modes_by_freq(s, FREQS, polarization="TE", n_guess=3.0)


# --------------------------------------------------------------------------
# Model: validators
# --------------------------------------------------------------------------

def _base_kwargs(n):
    prof = tuple(0.0 for _ in range(n))
    return dict(
        axis="z", direction="+", position_um=1.0, polarization="Ex",
        n_eff=2.5, nu=n, nv=1, profile=prof,
        source_time=ph.GaussianPulse(freq0_hz=2e14, fwidth_hz=2e13),
    )


def test_model_broadband_valid():
    n = 4
    src = ModeSource(
        **_base_kwargs(n),
        freqs_hz=(1.9e14, 2.0e14, 2.1e14),
        n_eff_by_freq=(2.4, 2.5, 2.6),
        profiles_by_freq=tuple(tuple(float(i) for _ in range(n)) for i in range(3)),
    )
    assert len(src.freqs_hz) == 3


def test_model_rejects_arrays_without_freqs():
    with pytest.raises(ValidationError, match="requires freqs_hz"):
        ModeSource(**_base_kwargs(4), n_eff_by_freq=(2.4, 2.5))


def test_model_rejects_nonparallel_arrays():
    n = 4
    with pytest.raises(ValidationError, match="parallel to freqs_hz"):
        ModeSource(
            **_base_kwargs(n),
            freqs_hz=(1.9e14, 2.0e14, 2.1e14),
            n_eff_by_freq=(2.4, 2.5),  # length 2 != 3
            profiles_by_freq=tuple(tuple(0.0 for _ in range(n)) for _ in range(3)),
        )


def test_model_rejects_non_ascending_freqs():
    n = 4
    with pytest.raises(ValidationError, match="ascending"):
        ModeSource(
            **_base_kwargs(n),
            freqs_hz=(2.1e14, 2.0e14, 1.9e14),
            n_eff_by_freq=(2.4, 2.5, 2.6),
            profiles_by_freq=tuple(tuple(0.0 for _ in range(n)) for _ in range(3)),
        )


def test_model_rejects_bad_profile_length():
    n = 4
    with pytest.raises(ValidationError, match="profiles_by_freq.*!= nu"):
        ModeSource(
            **_base_kwargs(n),
            freqs_hz=(1.9e14, 2.0e14),
            n_eff_by_freq=(2.4, 2.5),
            profiles_by_freq=((0.0,) * n, (0.0,) * (n - 1)),  # wrong length
        )


def test_model_rejects_neff_below_one():
    n = 2
    with pytest.raises(ValidationError, match="n_eff_by_freq must be >= 1"):
        ModeSource(
            **_base_kwargs(n),
            freqs_hz=(1.9e14, 2.0e14),
            n_eff_by_freq=(0.9, 2.5),
            profiles_by_freq=((0.0,) * n, (0.0,) * n),
        )


def test_model_rejects_minor_array_without_scalar_minor():
    n = 2
    with pytest.raises(ValidationError, match="without a scalar profile_minor"):
        ModeSource(
            **_base_kwargs(n),
            freqs_hz=(1.9e14, 2.0e14),
            n_eff_by_freq=(2.4, 2.5),
            profiles_by_freq=((0.0,) * n, (0.0,) * n),
            profiles_minor_by_freq=((0.0,) * n, (0.0,) * n),
        )


# --------------------------------------------------------------------------
# Wire format
# --------------------------------------------------------------------------

def test_wire_omits_broadband_when_unset():
    src = ModeSource(**_base_kwargs(4))
    js = src.model_dump(by_alias=True, exclude_none=True)
    for k in ("freqs_hz", "n_eff_by_freq", "profiles_by_freq",
              "profiles_minor_by_freq"):
        assert k not in js


def test_wire_roundtrips_broadband():
    sim, pulse = _shell()
    mbf = _scalar_modes()
    src = mode_source(sim, mbf[FREQS[1]], axis="z", position_um=1.0,
                      source_time=pulse, modes_by_freq=mbf)
    sim2 = sim.model_copy(update={"sources": (src,)})
    restored = ph.Simulation.from_wire_json(sim2.to_wire_json())
    rsrc = restored.sources[0]
    assert rsrc.freqs_hz == src.freqs_hz
    assert rsrc.n_eff_by_freq == pytest.approx(src.n_eff_by_freq)
    assert np.allclose(np.array(rsrc.profiles_by_freq),
                       np.array(src.profiles_by_freq))


# --------------------------------------------------------------------------
# Plugin builders
# --------------------------------------------------------------------------

def test_scalar_builder_packs_dispersive_band():
    sim, pulse = _shell()
    mbf = _scalar_modes()
    src = mode_source(sim, mbf[FREQS[1]], axis="z", position_um=1.0,
                      source_time=pulse, modes_by_freq=mbf)
    assert src.freqs_hz == FSORT  # ascending
    # n_eff disperses across the band (the reason num_freqs exists): higher
    # frequency (last, ascending) is the more-confined, higher-n_eff end.
    assert src.n_eff_by_freq[-1] > src.n_eff_by_freq[0]
    # every per-frequency profile is sign-aligned to the centre (positive corr).
    cen = np.array(src.profile)
    for p in src.profiles_by_freq:
        p = np.array(p)
        assert float(np.dot(p, cen)) > 0.0


def test_scalar_builder_single_mode_is_legacy_noop():
    sim, pulse = _shell()
    mbf = _scalar_modes()
    src = mode_source(sim, mbf[FREQS[1]], axis="z", position_um=1.0,
                      source_time=pulse, modes_by_freq={FREQS[1]: mbf[FREQS[1]]})
    assert src.freqs_hz is None  # <2 entries → single frozen mode


def test_vector_builder_packs_band_with_minor():
    sim, pulse = _shell()
    vs = VectorModeSolver.from_rectangular_core(
        wavelength_um=1.31, dl_um=DL, core_w_um=0.45, core_h_um=0.22,
        n_core=N_CORE, n_clad=N_CLAD)
    mbf = solve_modes_by_freq(vs, FREQS, n_guess=3.0)
    src = mode_source_vector(sim, mbf[FREQS[1]], axis="z", position_um=1.0,
                             source_time=pulse, modes_by_freq=mbf)
    assert src.freqs_hz == FSORT
    assert src.profiles_minor_by_freq is not None
    assert len(src.profiles_minor_by_freq) == len(FREQS)
    assert all(len(p) == src.nu * src.nv for p in src.profiles_minor_by_freq)
