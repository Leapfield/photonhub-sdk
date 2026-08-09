"""Cross-section mode solvers (kfj_smoothing FLM + yee_mode FDFD) — window
placement pins.

Both solvers deliberately SNAP their eps-raster window to the simulation grid
(grid-consistent dielectric walls), which displaces the sampled window from the
requested center by up to ~a cell. Every consumer reconstructs coordinates as a
CENTERED array (``(i - (n-1)/2)*dl + center``), so without placement metadata
the mode landed up to ~1.4 cells off the guide axis — and the launch (yee) and
readout (kfj) banks were displaced relative to EACH OTHER by ~1 cell. These
tests pin the ``VectorMode.center_offset_um`` metadata that carries the true
placement:

* the reconstructed (offset-carried) field centroid coincides with the true
  strip center to a fraction of a cell, while the as-placed (offset-ignored)
  reconstruction is off by ~half a cell or more;
* the yee-launch ⇄ kfj-readout mode pair on the SAME off-grid cross-section
  overlaps far better with the offsets carried;
* the yee window covers the requested extent (its ``nh`` used to under-cover
  the high side by up to one cell).

These are the first tests of ``plugins/yee_mode.py`` / ``plugins/kfj_smoothing``
— grids are kept small so the whole file runs in well under a second.
"""

from dataclasses import replace

import numpy as np
import pytest

import photonhub as ph
from photonhub.plugins.kfj_smoothing import solve_mode_on_cross_section
from photonhub.plugins.mode_overlap import mode_overlap
from photonhub.plugins.yee_mode import sample_staggered_eps, solve_yee_mode

DL = 0.05
N_CORE, N_CLAD = 3.5, 1.444
WL_UM = 1.55
# A deliberately NON-grid-aligned strip center (fractional-cell offsets differ
# per axis so both offsets are exercised).
FRAC = 0.6
CX = 2.0 + FRAC * DL
CY = 2.0 + 0.5 * FRAC * DL
HALF_W, HALF_V = 0.6, 0.5


@pytest.fixture(scope="module")
def sim() -> ph.Simulation:
    """A straight SOI strip along z whose cross-section center is off-grid."""
    core = ph.Structure(
        geometry=ph.Box(center_um=(CX, CY, 2.0), size_um=(0.45, 0.22, 8.0)),
        medium=ph.Medium(permittivity=N_CORE ** 2))
    pulse = ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4e13)
    return ph.Simulation(
        size_um=(4.0, 4.0, 4.0), grid=ph.UniformGridSpec(dl_um=DL),
        run=ph.RunSpec(n_steps=100),
        background=ph.Background(permittivity=N_CLAD ** 2),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        structures=[core],
        sources=[ph.PointDipole(center_um=(2.0, 2.0, 2.0), polarization="Ex",
                                source_time=pulse)])


@pytest.fixture(scope="module")
def kfj_mode(sim):
    # use_yee=False: these tests pin the FLM (node-collocated) path's window
    # placement and the CROSS-family launch/readout displacement — the library
    # default (use_yee=True) routes through solve_yee_mode, where launch and
    # readout share one placement convention and the displacement cancels.
    return solve_mode_on_cross_section(
        sim, "z", 2.0, WL_UM, "TE", 0, h_center_um=CX, v_center_um=CY,
        half_w_um=HALF_W, half_v_um=HALF_V, dl_um=DL, use_yee=False)


@pytest.fixture(scope="module")
def yee_mode(sim):
    return solve_yee_mode(
        sim, "z", 2.0, WL_UM, "TE", 0, h_center_um=CX, v_center_um=CY,
        half_w_um=HALF_W, half_v_um=HALF_V, dl_um=DL)


def _centroid_err_um(mode, weight):
    """Intensity-centroid position of ``weight`` in the consumer-reconstructed
    mode frame (centered coords + carried offset), relative to the requested
    center — i.e. the placement error vs the true (symmetric) strip axis."""
    ny, nx = weight.shape
    off = mode.center_offset_um or (0.0, 0.0)
    xs = (np.arange(nx) - (nx - 1) / 2.0) * mode.dl_x_um + off[0]
    ys = (np.arange(ny) - (ny - 1) / 2.0) * mode.dl_y_um + off[1]
    t = weight.sum()
    return (float((weight.sum(axis=0) * xs).sum() / t),
            float((weight.sum(axis=1) * ys).sum() / t))


def test_kfj_mode_carries_true_window_center(kfj_mode):
    """With the offset carried, the FLM mode's H centroid sits on the strip
    axis to a small fraction of a cell (the H eigenvector is the clean position
    reference); ignoring the offset (the old centered-array assumption) leaves
    it ~0.3–0.7 cells off."""
    assert kfj_mode.center_offset_um is not None
    w = np.abs(kfj_mode.hx) ** 2 + np.abs(kfj_mode.hy) ** 2
    ex, ey = _centroid_err_um(kfj_mode, w)
    assert abs(ex) < DL / 2 and abs(ey) < DL / 2
    ex0, ey0 = _centroid_err_um(replace(kfj_mode, center_offset_um=None), w)
    assert np.hypot(ex, ey) < np.hypot(ex0, ey0)
    assert np.hypot(ex0, ey0) > 0.25 * DL  # the old placement really was off


def test_yee_mode_carries_true_window_center(yee_mode):
    """Same pin for the Yee-grid solver, using Ez (the node-collocated
    component — Ex/Ey carry the documented ±dl/2 Yee stagger)."""
    assert yee_mode.center_offset_um is not None
    w = np.abs(yee_mode.ez) ** 2
    ex, ey = _centroid_err_um(yee_mode, w)
    assert abs(ex) < DL / 2 and abs(ey) < DL / 2
    ex0, ey0 = _centroid_err_um(replace(yee_mode, center_offset_um=None), w)
    assert np.hypot(ex, ey) < np.hypot(ex0, ey0)
    assert np.hypot(ex0, ey0) > 0.25 * DL


def test_launch_readout_offset_improves_overlap(yee_mode, kfj_mode):
    """Launch ⇄ readout self-consistency: the yee-launch mode and the FLM
    readout-bank mode of the SAME off-grid cross-section must overlap nearly
    perfectly once each is placed where its raster truly was. As-placed
    (centered-array assumption) the two banks are displaced ~1 cell relative to
    each other and the overlap drops by several percent. Pin the improvement,
    not exact values (the residual is discretization- and stagger-limited)."""
    p_with = mode_overlap(yee_mode, kfj_mode).power
    p_without = mode_overlap(
        replace(yee_mode, center_offset_um=None),
        replace(kfj_mode, center_offset_um=None)).power
    assert p_with > 0.95
    assert p_with > p_without + 0.02


def test_yee_mode_satisfies_vectormode_invariant(yee_mode):
    """solve_yee_mode restores the VectorMode-declared invariant: transverse-E
    jointly L2-normalized, dominant transverse-E real-positive at its peak
    (eigs returns an arbitrary eigenvector scale/phase)."""
    l2 = float(np.sum(np.abs(yee_mode.ex) ** 2 + np.abs(yee_mode.ey) ** 2))
    assert l2 == pytest.approx(1.0, abs=1e-9)
    ref = yee_mode.ex  # TE mode: Ex-major
    peak = ref.flat[int(np.argmax(np.abs(ref)))]
    assert peak.real > 0
    assert abs(peak.imag) < 1e-9 * abs(peak)


def test_yee_window_covers_requested_extent(sim):
    """Regression for the nh under-coverage: ``ceil(2*half_w/dl)`` missed the
    high side by up to one cell whenever the grid snap moved the window origin
    down by more than the ceil residual (as it does for this FRAC). The window
    [lo, lo + n*dl] must cover [center - half, center + half] on both axes."""
    *_, nh, nv, h_lo, v_lo = sample_staggered_eps(
        sim, "z", 2.0, h_center=CX, v_center=CY, half_w=HALF_W, half_v=HALF_V,
        dl=DL)
    assert h_lo <= CX - HALF_W + 1e-12
    assert v_lo <= CY - HALF_V + 1e-12
    assert h_lo + nh * DL >= CX + HALF_W - 1e-12
    assert v_lo + nv * DL >= CY + HALF_V - 1e-12
