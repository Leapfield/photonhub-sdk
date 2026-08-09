"""Dispersive eps anchoring of the cross-section mode solvers.

A Lorentz medium's ``permittivity`` field is only the high-frequency limit
eps_inf (NUMERICS §19); a mode solved from it sees the WRONG index (benchmark
Si: n_eff 2.06 instead of 2.45 at 1.55 um). The cross-section solvers therefore
anchor every medium at the solve frequency via ``Medium.permittivity_at_hz``
(and per-frequency banks re-anchor at EACH bank frequency), unless an explicit
``eps_of_medium`` override freezes it. Pinned here by equivalence: a dispersive
solve must match the same geometry rebuilt with the non-dispersive eps(f).
"""
import math
import warnings

import pytest

import photonhub as ph
from photonhub.plugins import (
    mode_bank_on_cross_section,
    solve_mode_on_cross_section,
    solve_yee_multimode_bank,
)

C0 = 2.99792458e8
LAM = 1.55
F0 = C0 / (LAM * 1e-6)
DL = 0.05

# A visibly dispersive single-pole fit (delta-eps 2.5, pole above the band):
# eps(F0) ~ 12.27 vs eps_inf 9.619 — the two answers are far apart, so an
# anchor regression cannot hide inside solver tolerance.
POLE = ph.LorentzPole(resonance_frequency_hz=8.0e14, delta_eps=2.5)
EPS_INF = 9.619


def _strip(medium):
    L = (1.8, 1.4, 1.0)
    core = ph.Structure(
        geometry=ph.Box(center_um=(L[0] / 2, L[1] / 2, L[2] / 2),
                        size_um=(0.5, 0.22, L[2] * 2)),
        medium=medium)
    with warnings.catch_warnings():
        # a dispersive medium touching the (test-sized) PML trips the
        # stabilized-PML advisories — irrelevant to the eps anchoring under test
        warnings.simplefilter("ignore", UserWarning)
        return ph.Simulation(
            size_um=L, grid=ph.UniformGridSpec(dl_um=DL), run={"n_steps": 100},
            background=ph.Background(permittivity=1.444 ** 2), structures=(core,),
            boundaries=ph.Boundaries(x="pml", y="pml", z="pml"), pml_num_layers=10,
            sources=[ph.PointDipole(center_um=(L[0] / 2, L[1] / 2, 0.5),
                                    polarization="Ex",
                                    source_time=ph.GaussianPulse(
                                        freq0_hz=F0, fwidth_hz=0.1 * F0))])


WIN = dict(h_center_um=0.9, v_center_um=0.7, half_w_um=0.75, half_v_um=0.6,
           dl_um=DL)


# --- Medium.permittivity_at_hz ------------------------------------------------


def test_permittivity_at_hz_matches_the_documented_pole_model():
    m = ph.Medium(permittivity=EPS_INF, lorentz=POLE)
    w = 2 * math.pi * F0
    w0 = 2 * math.pi * POLE.resonance_frequency_hz
    expected = EPS_INF + POLE.delta_eps * w0 ** 2 / (w0 ** 2 - w ** 2)
    assert m.permittivity_at_hz(F0) == pytest.approx(expected, rel=1e-14)
    # static limit: eps(0+) -> eps_inf + delta_eps
    assert m.permittivity_at_hz(1.0) == pytest.approx(EPS_INF + POLE.delta_eps,
                                                      rel=1e-9)


def test_permittivity_at_hz_is_identity_for_nondispersive():
    m = ph.Medium(permittivity=4.0)
    assert m.permittivity_at_hz(F0) == 4.0
    assert m.permittivity_at_hz(1.0) == 4.0


def test_permittivity_at_hz_damped_pole_takes_the_real_part():
    damped = ph.LorentzPole(resonance_frequency_hz=8.0e14, delta_eps=2.5,
                            linewidth_hz=1.0e13)
    m = ph.Medium(permittivity=EPS_INF, lorentz=damped)
    w = 2 * math.pi * F0
    w0 = 2 * math.pi * damped.resonance_frequency_hz
    g = 2 * math.pi * damped.linewidth_hz
    eps = EPS_INF + damped.delta_eps * w0 ** 2 / (w0 ** 2 - w ** 2 - 1j * g * w)
    assert m.permittivity_at_hz(F0) == pytest.approx(eps.real, rel=1e-14)


def test_permittivity_at_hz_raises_on_undamped_resonance():
    m = ph.Medium(permittivity=EPS_INF, lorentz=POLE)
    with pytest.raises(ValueError, match="resonance"):
        m.permittivity_at_hz(POLE.resonance_frequency_hz)
    with pytest.raises(ValueError):
        m.permittivity_at_hz(0.0)


# --- solve anchoring (Yee + FLM) ----------------------------------------------


@pytest.mark.parametrize("use_yee", [True, False])
def test_dispersive_solve_anchors_at_the_band_eps(use_yee):
    m_disp = ph.Medium(permittivity=EPS_INF, lorentz=POLE)
    eps_f0 = m_disp.permittivity_at_hz(F0)
    assert eps_f0 > EPS_INF + 2.0  # the anchor moves eps far from eps_inf
    md = solve_mode_on_cross_section(_strip(m_disp), "z", 0.5, LAM, "TE", 0,
                                     use_yee=use_yee, **WIN)
    me = solve_mode_on_cross_section(_strip(ph.Medium(permittivity=eps_f0)),
                                     "z", 0.5, LAM, "TE", 0, use_yee=use_yee,
                                     **WIN)
    # identical geometry + identical anchored eps -> identical discrete solve
    assert md.n_eff == pytest.approx(me.n_eff, abs=1e-10)


@pytest.mark.parametrize("use_yee", [True, False])
def test_bank_reanchors_eps_at_every_frequency(use_yee):
    m_disp = ph.Medium(permittivity=EPS_INF, lorentz=POLE)
    sim = _strip(m_disp)
    freqs = [0.95 * F0, F0, 1.05 * F0]
    bank = mode_bank_on_cross_section(sim, "z", 0.5, freqs, "TE", 0,
                                      use_yee=use_yee, **WIN)
    # each bank entry must equal a fresh solve at THAT frequency's eps(f) —
    # a frozen-eps bank (the old defect) misses these references by ~1e-3 in
    # n_eff (eps(f) spans ~0.03 across this band), seven orders above the
    # tolerance.
    for f in freqs:
        eps_f = m_disp.permittivity_at_hz(f)
        ref = solve_mode_on_cross_section(
            _strip(ph.Medium(permittivity=eps_f)), "z", 0.5, C0 / f * 1e6,
            "TE", 0, use_yee=use_yee, **WIN)
        assert bank[float(f)].n_eff == pytest.approx(ref.n_eff, abs=1e-10)
    assert m_disp.permittivity_at_hz(freqs[-1]) - \
        m_disp.permittivity_at_hz(freqs[0]) > 0.02  # the refs really differ


def test_eps_of_medium_override_stays_frozen_at_every_frequency():
    m_disp = ph.Medium(permittivity=EPS_INF, lorentz=POLE)
    sim = _strip(m_disp)
    anchor = 12.0
    freqs = [0.95 * F0, 1.05 * F0]
    medium = sim.structures[0].medium
    bank = mode_bank_on_cross_section(sim, "z", 0.5, freqs, "TE", 0,
                                      eps_of_medium={id(medium): anchor}, **WIN)
    for f in freqs:
        ref = solve_mode_on_cross_section(
            _strip(ph.Medium(permittivity=anchor)), "z", 0.5, C0 / f * 1e6,
            "TE", 0, **WIN)
        assert bank[float(f)].n_eff == pytest.approx(ref.n_eff, abs=1e-10)


def test_nondispersive_solves_are_unchanged_by_the_anchor():
    # freq-anchoring a constant-eps medium is the identity — the pre-fix and
    # post-fix operators must agree exactly (no benchmark drift).
    sim = _strip(ph.Medium(permittivity=3.4738 ** 2))
    from photonhub.plugins.yee_mode import sample_staggered_eps
    legacy = sample_staggered_eps(sim, "z", 0.5, h_center=0.9, v_center=0.7,
                                  half_w=0.75, half_v=0.6, dl=DL)
    anchored = sample_staggered_eps(sim, "z", 0.5, h_center=0.9, v_center=0.7,
                                    half_w=0.75, half_v=0.6, dl=DL, freq_hz=F0)
    for a, b in zip(legacy[:3], anchored[:3]):
        assert (a == b).all()


# --- provenance (feeds the ModeMonitor auto-bank) ------------------------------


def test_dispatcher_mode_carries_solve_provenance():
    sim = _strip(ph.Medium(permittivity=3.4738 ** 2))
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0, **WIN)
    p = mode.solve_params
    assert p is not None
    assert p["sim"] is sim  # a replay must extend THIS mode's geometry
    assert p["axis"] == "z" and p["plane_value_um"] == 0.5
    assert p["pol"] == "TE" and p["mode_index"] == 0
    assert p["dl_um"] == DL and p["use_yee"] is True
    assert "wavelength_um" not in p  # the bank supplies its own frequencies


def test_partial_override_keeps_unoverridden_media_per_frequency():
    # Two dispersive media, ONE pinned via eps_of_medium: the pin must freeze
    # only that medium — the other keeps its per-frequency anchor, so the bank
    # cannot depend on the ORDER of freqs_hz (regression: the old gate treated
    # any override as fully frozen and anchored everything at freqs[0]).
    m_core = ph.Medium(permittivity=EPS_INF, lorentz=POLE)
    m_slab = ph.Medium(permittivity=4.0, lorentz=ph.LorentzPole(
        resonance_frequency_hz=6.0e14, delta_eps=1.0))

    def two_layer(core, slab):
        L = (1.8, 1.4, 1.0)
        core_s = ph.Structure(
            geometry=ph.Box(center_um=(L[0] / 2, L[1] / 2, L[2] / 2),
                            size_um=(0.5, 0.22, L[2] * 2)),
            medium=core)
        slab_s = ph.Structure(
            geometry=ph.Box(center_um=(L[0] / 2, L[1] / 2 - 0.16, L[2] / 2),
                            size_um=(1.6, 0.10, L[2] * 2)),
            medium=slab)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return ph.Simulation(
                size_um=L, grid=ph.UniformGridSpec(dl_um=DL),
                run={"n_steps": 100},
                background=ph.Background(permittivity=1.444 ** 2),
                structures=(core_s, slab_s),
                boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
                pml_num_layers=10,
                sources=[ph.PointDipole(center_um=(L[0] / 2, L[1] / 2, 0.5),
                                        polarization="Ex",
                                        source_time=ph.GaussianPulse(
                                            freq0_hz=F0, fwidth_hz=0.1 * F0))])

    sim = two_layer(m_core, m_slab)
    pin = {id(sim.structures[1].medium): 4.2}      # pin the slab only
    freqs = [0.95 * F0, 1.05 * F0]
    fwd = mode_bank_on_cross_section(sim, "z", 0.5, freqs, "TE", 0,
                                     eps_of_medium=pin, **WIN)
    rev = mode_bank_on_cross_section(sim, "z", 0.5, list(reversed(freqs)),
                                     "TE", 0, eps_of_medium=pin, **WIN)
    for f in freqs:
        assert fwd[float(f)].n_eff == pytest.approx(rev[float(f)].n_eff,
                                                    abs=1e-12)
        # the un-pinned core really is anchored at eps(f), the pin at 4.2
        ref = solve_mode_on_cross_section(
            two_layer(ph.Medium(permittivity=m_core.permittivity_at_hz(f)),
                      ph.Medium(permittivity=4.2)),
            "z", 0.5, C0 / f * 1e6, "TE", 0, **WIN)
        assert fwd[float(f)].n_eff == pytest.approx(ref.n_eff, abs=1e-10)


def test_anchor_rejects_nonpositive_eps():
    # A strong pole BELOW the band drives Re eps negative in-band; feeding
    # that to the mode eigensolve would produce NaN modes far downstream, so
    # the anchor must refuse loudly at the rasterization step.
    pole = ph.LorentzPole(resonance_frequency_hz=0.8 * F0, delta_eps=3.0)
    m = ph.Medium(permittivity=1.5, lorentz=pole)
    assert m.permittivity_at_hz(F0) < 0.0
    with pytest.raises(ValueError, match="non-positive"):
        solve_mode_on_cross_section(_strip(m), "z", 0.5, LAM, "TE", 0, **WIN)


def test_eps_override_suppresses_provenance():
    # an id-keyed override cannot be replayed by a later re-solve — no
    # provenance, so no auto-bank silently anchored differently.
    sim = _strip(ph.Medium(permittivity=EPS_INF, lorentz=POLE))
    medium = sim.structures[0].medium
    mode = solve_mode_on_cross_section(sim, "z", 0.5, LAM, "TE", 0,
                                       eps_of_medium={id(medium): 12.0}, **WIN)
    assert mode.solve_params is None


# --- multimode Yee bank ---------------------------------------------------------


def test_yee_multimode_bank_is_rectangular_and_neff_descending():
    sim = _strip(ph.Medium(permittivity=3.4738 ** 2))
    freqs = [0.98 * F0, 1.02 * F0]
    bank = solve_yee_multimode_bank(sim, "z", 0.5, freqs, mode_indices=(0, 1),
                                    **WIN)
    assert sorted(bank.keys()) == sorted(float(f) for f in freqs)
    for f, per_idx in bank.items():
        assert sorted(per_idx.keys()) == [0, 1]
        assert per_idx[0].n_eff > per_idx[1].n_eff  # descending across pols
        assert all(m.yee_staggered for m in per_idx.values())


def test_yee_multimode_bank_rejects_unsupported_index():
    sim = _strip(ph.Medium(permittivity=3.4738 ** 2))
    with pytest.raises(ValueError, match="guided mode"):
        solve_yee_multimode_bank(sim, "z", 0.5, [F0], mode_indices=(0, 25),
                                 **WIN)
