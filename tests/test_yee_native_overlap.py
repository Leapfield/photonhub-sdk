"""Grid-consistent readout of a Yee-staggered reference mode.

A mode solved on the engine's own Yee grid (``solve_yee_mode`` sets
``VectorMode.yee_staggered=True``) registers 1:1 with the recorded FDTD field, so
the mode-overlap must NOT transversely co-locate the sim field to the node —
doing so introduces a ½-cell mismatch against the un-shifted mode. This pins:

  * a Yee mode self-overlapping its OWN field reads exactly 1.0 (co-location
    skipped), whereas forcing the co-location (an FLM/node mode) degrades it;
  * a node-collocated (FLM) mode is unaffected — it still co-locates.
"""
import dataclasses

import pytest

import numpy as np
import xarray as xr

from photonhub.plugins.mode_overlap import mode_transmission, vector_modal_fields
from photonhub.plugins.vector_modes import VectorMode

F0 = 2.99792458e8 / (1.55e-6)
ETA0 = 376.730313668
DL = 0.05


def _gaussian_vector_mode(yee: bool, ny=21, nx=21, dl=DL, neff=2.0):
    yy, xx = np.meshgrid(np.arange(ny) - (ny - 1) / 2.0,
                         np.arange(nx) - (nx - 1) / 2.0, indexing="ij")
    g = np.exp(-(xx * xx + yy * yy) / (2.0 * 3.0 ** 2)).astype(complex)
    ex, ey = g.copy(), 0.3 * g                       # a hybrid transverse profile
    nrm = np.sqrt(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
    ex, ey = ex / nrm, ey / nrm
    hx, hy = -(neff / ETA0) * ey, (neff / ETA0) * ex  # scalar-limit paired H
    z = np.zeros_like(ex)
    return VectorMode(n_eff=neff, n_group=None, ex=ex, ey=ey, ez=z,
                      hx=hx, hy=hy, hz=z, wavelength_um=1.55,
                      dl_x_um=dl, dl_y_um=dl, yee_staggered=yee)


def _plane_from_mode(mode):
    """A z-normal DFT-shaped plane whose four tangential components ARE the mode's
    own fields (a self-consistent 'recorded' field identical to the mode)."""
    nx = mode.ex.shape[1]
    ny = mode.ex.shape[0]
    x = (np.arange(nx) - (nx - 1) / 2.0) * mode.dl_x_um
    y = (np.arange(ny) - (ny - 1) / 2.0) * mode.dl_y_um
    m = vector_modal_fields(mode, x, y, axis="z", direction="+",
                            center_um=(0.0, 0.0))
    comps = {"Ex": m["e1"], "Ey": m["e2"], "Hx": m["h1"], "Hy": m["h2"]}
    out = {}
    for name, arr in comps.items():
        a = np.asarray(arr, dtype=np.complex128)[None, None, None, :, :]
        out[name] = xr.DataArray(a, dims=("f", "component", "z", "y", "x"),
                                 coords={"f": [F0], "component": [name],
                                         "z": [0.0], "y": y, "x": x})
    return out


def _two_freq_plane(mode_f1, mode_f2, f1, f2):
    """A plane whose f1 recording is mode_f1's own fields and f2 mode_f2's."""
    p1, p2 = _plane_from_mode(mode_f1), _plane_from_mode(mode_f2)
    out = {}
    for name in p1:
        a = np.concatenate([p1[name].values, p2[name].values], axis=0)
        out[name] = xr.DataArray(
            a, dims=("f", "component", "z", "y", "x"),
            coords={"f": [f1, f2], "component": [name],
                    "z": p1[name].coords["z"].values,
                    "y": p1[name].coords["y"].values,
                    "x": p1[name].coords["x"].values})
    return out


def test_mixed_bank_gates_colocation_per_frequency():
    # A modes_by_freq bank mixing a Yee mode (f1) and a node/FLM mode (f2) must
    # co-locate ONLY the f2 reading: one Yee entry must not turn co-location
    # off for the node-mode frequencies (a silent half-cell error), nor the
    # node entry force it back on for the Yee ones.
    yee = _gaussian_vector_mode(yee=True)
    node = _gaussian_vector_mode(yee=False)
    F1, F2 = F0, 1.05 * F0
    plane = _two_freq_plane(yee, node, F1, F2)

    # NB: plugins/__init__ re-exports a FUNCTION named mode_overlap that
    # shadows the submodule attribute, so resolve the module itself.
    import importlib
    mo = importlib.import_module("photonhub.plugins.mode_overlap")
    calls = []
    orig = mo._colocate_to_node

    def counting(arr, ax):
        calls.append(ax)
        return orig(arr, ax)

    mo._colocate_to_node = counting
    try:
        T = mode_transmission(plane, yee, axis="z", direction="+",
                              modes_by_freq={F1: yee, F2: node}, colocate=True)
    finally:
        mo._colocate_to_node = orig
    # exactly the four tangential components of the f2 reading were shifted
    assert len(calls) == 4
    # the Yee frequency still reads its own field natively (self-overlap 1);
    # the node frequency matches a standalone node-mode reading exactly
    assert T[F1] == pytest.approx(1.0, abs=1e-9)
    T2 = mode_transmission(_plane_from_mode(node), node, axis="z",
                           direction="+", colocate=True)[F0]
    assert T[F2] == pytest.approx(T2, abs=1e-12)


def test_yee_mode_self_overlap_is_exactly_unity_without_colocation():
    mode = _gaussian_vector_mode(yee=True)
    plane = _plane_from_mode(mode)
    # colocate=True is REQUESTED, but a Yee mode makes the overlap skip the
    # transverse co-location -> sim == mode exactly -> self-overlap == 1.
    T = mode_transmission(plane, mode, axis="z", direction="+", colocate=True)[F0]
    assert T == pytest.approx(1.0, abs=1e-9)


def test_colocation_would_degrade_the_yee_self_overlap():
    # Same field/profile, but flagged as a node (FLM) mode -> the sim field IS
    # co-located to the node while the mode is not, so the self-overlap drops
    # below unity. This is the ½-cell registration error the Yee path avoids.
    node_mode = _gaussian_vector_mode(yee=False)
    plane = _plane_from_mode(node_mode)
    T_coloc = mode_transmission(plane, node_mode, axis="z", direction="+",
                                colocate=True)[F0]
    yee_mode = dataclasses.replace(node_mode, yee_staggered=True)
    T_native = mode_transmission(plane, yee_mode, axis="z", direction="+",
                                 colocate=True)[F0]
    assert T_native == pytest.approx(1.0, abs=1e-9)
    assert T_coloc < T_native - 1e-4        # co-location measurably degrades it
    assert abs(1.0 - T_native) < abs(1.0 - T_coloc)   # native is closer to unity


def test_flm_node_mode_still_colocates():
    # A node-collocated FLM mode (yee_staggered False) must be UNCHANGED: with
    # colocate=False vs True the results differ (co-location is applied), i.e. the
    # gate only exempts Yee modes.
    node_mode = _gaussian_vector_mode(yee=False)
    plane = _plane_from_mode(node_mode)
    T_on = mode_transmission(plane, node_mode, axis="z", direction="+",
                             colocate=True)[F0]
    T_off = mode_transmission(plane, node_mode, axis="z", direction="+",
                              colocate=False)[F0]
    assert T_on != pytest.approx(T_off, abs=1e-6)   # colocation still active


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
