"""Grating diffraction-order decomposition of a periodic DFT plane.

The PhotonHub analogue of Tidy3D's ``DiffractionMonitor`` and Lumerical's
grating projections, as pure post-processing: a full-plane
``FieldDftMonitor`` recording the four tangential components over a
transverse-PERIODIC unit cell is decomposed into its discrete plane-wave
(grating) orders — complex s/p amplitudes per order and direction, per-order
power, propagation angles, and the propagating/evanescent mask.

Physics
=======
On a plane normal to ``axis`` with both in-plane axes periodic (periods
``L1, L2``), the transverse field is exactly a sum of Floquet orders
``(m1, m2)`` with in-plane wavevectors ``k1 = 2 pi m1 / L1``,
``k2 = 2 pi m2 / L2`` (normal incidence / zero-phase periodic boundaries —
the only kind the engine has; the oblique generalization adds the Bloch
k-offset here when Bloch boundaries land). Each order with
``|k_t|^2 < (n w / c)^2`` is a propagating plane wave with
``k_n = sqrt(k^2 - |k_t|^2)``; the rest are evanescent.

Per order the four recorded tangential components over-determine the four
physical unknowns — the complex s- and p-polarized amplitudes travelling
``+axis`` and ``-axis`` — so the decomposition solves them exactly:

* FFT over the periodic cell gives per-order coefficients; each Yee
  component's half-cell in-plane offset is removed by its exact phase
  ``e^{-i k dl / 2}`` (no interpolation, no high-order damping),
* the s/p basis per order: ``s_hat = n_hat x kt_hat``,
  ``p`` the in-plane-of-incidence polarization (for the ``k_t = 0`` order the
  convention is ``p`` along u1, ``s`` along u2, the right-handed cyclic
  in-plane pair of ``axis``),
* the +/- split uses E (recorded on the plane) against H (recorded a half
  cell off-plane): with ``phi = k_n d_axis / 2`` the recorded pair maps to
  the directional amplitudes through an exact 2x2 solve — the half-cell
  H offset is part of the model, not an error term. In the engine's
  ``e^{-i w t}`` phasor convention "forward" means ``e^{+i k_n a}`` along
  ``+axis``.

Powers are ``P = (A/2) (n/eta0) Re(k_n/k) (|Es|^2 + |Ep|^2)`` per order and
direction — the same source-spectrum-normalized units as the recorded
phasors, so RATIOS (diffraction efficiencies against a reference run's
incident power, or order-vs-order splits) are the meaningful outputs.
``net_power()`` (= forward minus backward totals) is the physical net power
through the plane; counter-propagating interference carries no net Poynting,
so the sum is position-independent. ``staggered_plane_power()`` is the naive
half-cell-staggered real-space Poynting integral of the same plane — per
order it equals the net power times ``cos(Re(k_n) d_axis / 2)``, an
under-count at coarse normal sampling — exposed as an independent
bookkeeping cross-check, not a physics reference.

What's not handled
==================
* **Oblique incidence** — zero-phase periodic boundaries only (the engine has
  no Bloch phase yet); the order ladder is un-shifted.
* **Symmetry-folded planes** — a §20 symmetry plane on an in-plane axis
  folds the recorded half-domain; rebuild is not implemented (raises).
* **Decimated planes** — ``interval_space`` strides > 1 on an in-plane axis
  alias high orders (raises).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from ..components.grid import snap_mixed_plane
from ._constants import _TANGENTIAL, C0
from .mode_overlap import ETA0

__all__ = ["DiffractionOrders", "diffraction_orders"]

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
# Right-handed cyclic in-plane pair (u1, u2) per normal axis — the same order
# _TANGENTIAL lists the components in, so (u1, u2, axis) is right-handed and
# s_hat = n_hat x kt_hat = (-c2, +c1) holds without orientation fix-ups.
_CYCLIC = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}

# |k_n| below this fraction of k counts as grazing (Wood-anomaly window):
# the direction split divides by k_n and loses meaning there.
_GRAZING_REL = 1e-6


@dataclass(frozen=True)
class DiffractionOrders:
    """Result of :func:`diffraction_orders`.

    Order arrays are indexed ``(f, m1, m2)`` with ``orders1`` / ``orders2``
    the integer order coordinates (ascending, 0-centred). Amplitudes are the
    complex FULL E-field amplitudes of each order's s/p wave (engine phasor
    units); they are NaN for evanescent/grazing orders, where a directional
    plane-wave amplitude is not defined. Powers are 0 there.
    """

    axis: str
    in_plane_axes: Tuple[str, str]
    freqs_hz: np.ndarray          # (nf,)
    orders1: np.ndarray           # (n1,) int
    orders2: np.ndarray           # (n2,) int
    k1: np.ndarray                # (n1,) rad/um
    k2: np.ndarray                # (n2,) rad/um
    kn: np.ndarray                # (nf, n1, n2) complex rad/um (+Re/+Im root)
    propagating: np.ndarray       # (nf, n1, n2) bool
    amp_s_forward: np.ndarray     # (nf, n1, n2) complex
    amp_p_forward: np.ndarray
    amp_s_backward: np.ndarray
    amp_p_backward: np.ndarray
    power_forward: np.ndarray     # (nf, n1, n2) float
    power_backward: np.ndarray
    n_medium: float
    area_um2: float
    _raw_plane_power: np.ndarray  # (nf,)

    # -- accessors -----------------------------------------------------------
    def _order_index(self, m1: int, m2: int) -> Tuple[int, int]:
        i1 = np.where(self.orders1 == m1)[0]
        i2 = np.where(self.orders2 == m2)[0]
        if not i1.size or not i2.size:
            raise KeyError(
                f"order ({m1}, {m2}) outside the resolvable set "
                f"[{self.orders1.min()}..{self.orders1.max()}] x "
                f"[{self.orders2.min()}..{self.orders2.max()}]")
        return int(i1[0]), int(i2[0])

    def order(self, m1: int, m2: int) -> Dict[str, np.ndarray]:
        """Everything about one order, per frequency."""
        i1, i2 = self._order_index(m1, m2)
        return {
            "kn": self.kn[:, i1, i2],
            "propagating": self.propagating[:, i1, i2],
            "theta_rad": self.theta_rad[:, i1, i2],
            "amp_s_forward": self.amp_s_forward[:, i1, i2],
            "amp_p_forward": self.amp_p_forward[:, i1, i2],
            "amp_s_backward": self.amp_s_backward[:, i1, i2],
            "amp_p_backward": self.amp_p_backward[:, i1, i2],
            "power_forward": self.power_forward[:, i1, i2],
            "power_backward": self.power_backward[:, i1, i2],
        }

    @property
    def theta_rad(self) -> np.ndarray:
        """Polar diffraction angle from the +axis normal, per (f, m1, m2);
        NaN for non-propagating orders."""
        k = (2.0 * math.pi * self.n_medium / C0 * 1e-6
             * self.freqs_hz)[:, None, None]
        kt = np.sqrt(self.k1[None, :, None] ** 2 + self.k2[None, None, :] ** 2)
        ratio = np.broadcast_to(kt, self.kn.shape) / k
        theta = np.where(self.propagating, np.arcsin(np.clip(ratio, 0, 1)),
                         np.nan)
        return theta

    @property
    def phi_rad(self) -> np.ndarray:
        """Azimuth of the order's in-plane wavevector in the (u1, u2) frame,
        per (m1, m2) (frequency-independent)."""
        return np.arctan2(self.k2[None, :] * np.ones_like(self.k1)[:, None],
                          self.k1[:, None] * np.ones_like(self.k2)[None, :])

    def total_power(self, direction: str = "+") -> np.ndarray:
        """Sum of propagating order powers per frequency."""
        if direction == "+":
            return self.power_forward.sum(axis=(1, 2))
        if direction == "-":
            return self.power_backward.sum(axis=(1, 2))
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")

    def net_power(self) -> np.ndarray:
        """Physical net power through the plane toward +axis per frequency:
        ``total_power('+') - total_power('-')`` (counter-propagating
        interference carries no net Poynting; evanescent-pair tunneling is
        excluded with the evanescent mask)."""
        return self.total_power("+") - self.total_power("-")

    def staggered_plane_power(self) -> np.ndarray:
        """The naive half-cell-staggered Poynting integral of the recorded
        plane, computed directly in real space (E on-plane against H a half
        cell off-plane). Per order it equals the directional net power times
        ``cos(Re(k_n) d_axis / 2)`` — an UNDER-count at coarse normal
        sampling — so it serves as an independent bookkeeping cross-check
        against the order decomposition, not as the physics reference
        (:meth:`net_power` is that)."""
        return self._raw_plane_power.copy()


def _resolve_monitor(simulation, monitor):
    if hasattr(monitor, "name") and hasattr(monitor, "freqs_hz"):
        return monitor
    name = str(monitor)
    for m in getattr(simulation, "monitors", ()):
        if getattr(m, "name", None) == name:
            return m
    raise ValueError(f"monitor {name!r} not found on the simulation")


def _uniform_spacing(coords: np.ndarray, letter: str) -> float:
    if coords.size < 2:
        raise ValueError(
            f"in-plane axis {letter!r} carries {coords.size} sample(s) — a "
            "periodic order decomposition needs the full sampled period "
            "(>= 2 cells)")
    d = np.diff(coords)
    if not np.allclose(d, d[0], rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"in-plane axis {letter!r} is not uniformly sampled — graded "
            "meshes on a periodic axis cannot be Fourier-decomposed")
    return float(d[0])


def _infer_n_medium(simulation, axis: str, position_um: float) -> float:
    """The plane must lie in a homogeneous region for plane-wave orders to be
    well-defined; sample the analytic eps on the cut and require uniformity."""
    from ..viz.eps import sample_eps_plane
    _, _, eps2d = sample_eps_plane(simulation, axis, position_um)
    lo, hi = float(np.min(eps2d)), float(np.max(eps2d))
    if hi - lo > 1e-9 * max(1.0, hi):
        raise ValueError(
            f"the monitor plane at {axis}={position_um:.4g} um crosses "
            f"structures (eps spans [{lo:.4g}, {hi:.4g}]) — grating orders "
            "are defined in a homogeneous region; move the plane or pass "
            "n_medium explicitly")
    return math.sqrt(lo)


def diffraction_orders(
    simulation,
    data,
    monitor,
    *,
    n_medium: Optional[float] = None,
) -> DiffractionOrders:
    """Decompose a recorded periodic DFT plane into grating orders.

    Parameters
    ----------
    simulation:
        The run's Simulation (grid spacing, boundaries, symmetry and — when
        ``n_medium`` is not given — the analytic cross-section all come from
        it).
    data:
        The run's ``SimulationData`` (or any ``name -> DataArray`` mapping).
    monitor:
        The full-plane ``FieldDftMonitor`` (or its name) recording all four
        tangential components of the plane normal to its zero-size axis.
    n_medium:
        Refractive index of the homogeneous region containing the plane.
        Default: sampled from the simulation cross-section (raises if the
        plane crosses structures).

    Returns
    -------
    DiffractionOrders
    """
    mon = _resolve_monitor(simulation, monitor)
    size = mon.size_um
    zero_axes = [i for i, s in enumerate(size) if s == 0.0]
    if len(zero_axes) != 1:
        raise ValueError(
            f"monitor {mon.name!r} must be a plane (exactly one zero size "
            f"component), got size {size}")
    axis = "xyz"[zero_axes[0]]
    u1, u2 = _CYCLIC[axis]

    # --- physics preconditions ---------------------------------------------
    for letter in (u1, u2):
        kind = getattr(simulation.boundaries, letter, None)
        if kind != "periodic":
            raise ValueError(
                f"in-plane axis {letter!r} has boundary {kind!r} — grating "
                "orders need periodic boundaries on both in-plane axes")
        sym = simulation.symmetry["xyz".index(letter)] if getattr(
            simulation, "symmetry", None) else 0
        if sym:
            raise ValueError(
                f"symmetry on in-plane axis {letter!r} folds the recorded "
                "plane; rebuild of folded planes is not implemented")
    ivs = getattr(mon, "interval_space", None)
    if ivs is not None:
        for letter in (u1, u2):
            if ivs["xyz".index(letter)] != 1:
                raise ValueError(
                    f"monitor {mon.name!r} decimates in-plane axis "
                    f"{letter!r} (interval_space {ivs}) — that aliases high "
                    "orders; record the full plane")

    comps = _TANGENTIAL[axis]      # (E_u1, E_u2, H_u1, H_u2), cyclic order
    needed = set(comps)
    have = set(getattr(mon, "fields", ()))
    if not needed <= have:
        raise ValueError(
            f"monitor {mon.name!r} must record all four tangential "
            f"components {sorted(needed)}; it has {sorted(have)}")

    da = data[mon.name]
    freqs = np.asarray(da.coords["f"].values, dtype=np.float64)
    p1 = np.asarray(da.coords[u1].values, dtype=np.float64)
    p2 = np.asarray(da.coords[u2].values, dtype=np.float64)
    d1 = _uniform_spacing(p1, u1)
    d2 = _uniform_spacing(p2, u2)
    n1, n2 = p1.size, p2.size
    L1, L2 = n1 * d1, n2 * d2
    area = L1 * L2

    pos_a = float(np.asarray(da.coords[axis].values).reshape(-1)[0])
    # local normal-axis spacing at the plane (graded-aware), for the exact
    # half-cell H referral
    _, d_a = snap_mixed_plane(simulation, _AXIS_IDX[axis], pos_a)
    d_a = float(d_a) if d_a else 0.0

    # (f, n1, n2) arrays per component, dims resolved BY NAME
    def plane(comp: str) -> np.ndarray:
        arr = da.sel(component=comp).transpose("f", axis, u2, u1)
        vals = np.asarray(arr.values)
        if vals.shape[1] != 1:
            raise ValueError(
                f"monitor {mon.name!r} is {vals.shape[1]} cells thick along "
                f"{axis!r} — expected a single plane")
        return np.transpose(vals[:, 0, :, :], (0, 2, 1))  # (f, n1, n2)

    e1_r, e2_r, h1_r, h2_r = (plane(c) for c in comps)

    # --- raw staggered Poynting integral (bookkeeping cross-check) -----------
    # The Yee pairing E_u1 H_u2* / E_u2 H_u1* is in-plane co-located; the
    # normal half-cell H offset stays IN this integral (see
    # staggered_plane_power's cos(k_n d_a / 2) relation to the order powers).
    raw_power = 0.5 * np.sum(
        np.real(e1_r * np.conj(h2_r) - e2_r * np.conj(h1_r)),
        axis=(1, 2)) * d1 * d2

    # --- per-order coefficients with exact stagger referral ------------------
    # FFT convention: coeff[m] = (1/N) sum_i f[i] e^{-2 pi i m i / N}; the
    # sample at index i sits at p[0] + (i + delta) d, so referring the
    # coefficient to absolute coordinates multiplies by
    # e^{-i k_m (p0 + delta d)}.
    m1 = np.fft.fftfreq(n1, d=1.0 / n1)  # integers 0,1,...,-1
    m2 = np.fft.fftfreq(n2, d=1.0 / n2)
    k1 = 2.0 * math.pi * m1 / L1
    k2 = 2.0 * math.pi * m2 / L2

    def orders_of(arr: np.ndarray, delta1: float, delta2: float) -> np.ndarray:
        c = np.fft.fft2(arr, axes=(1, 2)) / (n1 * n2)
        ph1 = np.exp(-1j * k1 * (p1[0] + delta1 * d1))[None, :, None]
        ph2 = np.exp(-1j * k2 * (p2[0] + delta2 * d2))[None, None, :]
        return c * ph1 * ph2

    # Yee offsets in the (u1, u2) frame: E_c half-staggered along c; H_c
    # half-staggered along BOTH axes perpendicular to c (one of which is the
    # normal, handled separately via phi).
    e1 = orders_of(e1_r, 0.5, 0.0)
    e2 = orders_of(e2_r, 0.0, 0.5)
    h1 = orders_of(h1_r, 0.0, 0.5)
    h2 = orders_of(h2_r, 0.5, 0.0)

    # --- per-order geometry ---------------------------------------------------
    if n_medium is None:
        n_medium = _infer_n_medium(simulation, axis, pos_a)
    n_medium = float(n_medium)
    if not n_medium >= 1.0:
        raise ValueError(f"n_medium must be >= 1, got {n_medium}")

    k = (2.0 * math.pi * n_medium / C0 * 1e-6 * freqs)  # (nf,) rad/um
    K1 = k1[None, :, None]
    K2 = k2[None, None, :]
    kt2 = K1 ** 2 + K2 ** 2
    kn = np.sqrt((k[:, None, None] ** 2 - kt2).astype(np.complex128))
    # principal root: +Re for propagating, +Im (decay) for evanescent
    kn = np.where(kn.real < 0, -kn, kn)
    kn = np.where((kn.real == 0) & (kn.imag < 0), np.conj(kn), kn)
    propagating = kn.real > _GRAZING_REL * k[:, None, None]
    grazing = propagating & (np.abs(kn) < 1e-3 * k[:, None, None])
    if np.any(grazing):
        warnings.warn(
            "grazing (Wood-anomaly) orders present — their directional "
            "amplitudes are ill-conditioned", stacklevel=2)

    kt = np.sqrt(kt2)
    with np.errstate(invalid="ignore", divide="ignore"):
        c1 = np.where(kt > 0, K1 / np.where(kt > 0, kt, 1.0), 1.0)
        c2 = np.where(kt > 0, K2 / np.where(kt > 0, kt, 1.0), 0.0)
    c1 = np.broadcast_to(c1, kn.shape)
    c2 = np.broadcast_to(c2, kn.shape)

    # s/p projections of the recorded per-order fields
    Es_rec = -c2 * e1 + c1 * e2          # s_hat . E
    Ep_rec = c1 * e1 + c2 * e2           # kt_hat . E
    Hs_rec = -c2 * h1 + c1 * h2          # s_hat . H (recorded off-plane)
    Hp_rec = c1 * h1 + c2 * h2           # kt_hat . H (recorded off-plane)

    phi = 0.5 * kn * d_a                 # complex-safe half-cell H phase
    cosphi = np.cos(phi)
    eip = np.exp(1j * phi)
    eim = np.exp(-1j * phi)

    with np.errstate(invalid="ignore", divide="ignore"):
        kr = np.where(np.abs(kn) > 0, k[:, None, None] / kn, np.inf)
        # s-pol: a = Es_rec, B = -(eta0/n)(k/kn) (kt_hat . H_rec)
        B = -(ETA0 / n_medium) * kr * Hp_rec
        Es_f = (Es_rec * eim + B) / (2.0 * cosphi)
        Es_b = (Es_rec * eip - B) / (2.0 * cosphi)
        # p-pol (H along s_hat): B' = (n/eta0)(k/kn)... in H units:
        # Hp+/- from a' = Hs_rec, b' = kt_hat . E, B' = (omega eps / kn) b'
        #   = (n k)/(eta0 kn) b'; report as full-E amplitude Ep = (eta0/n) Hp.
        Bp = (n_medium / ETA0) * kr * Ep_rec
        Hp_f = (Hs_rec + Bp * eim) / (2.0 * cosphi)
        Hp_b = (Hs_rec - Bp * eip) / (2.0 * cosphi)
    Ep_f = (ETA0 / n_medium) * Hp_f
    Ep_b = (ETA0 / n_medium) * Hp_b

    valid = propagating & ~grazing
    nanc = np.complex128(np.nan + 1j * np.nan)
    Es_f = np.where(valid, Es_f, nanc)
    Es_b = np.where(valid, Es_b, nanc)
    Ep_f = np.where(valid, Ep_f, nanc)
    Ep_b = np.where(valid, Ep_b, nanc)

    # order powers (0 for evanescent/grazing)
    pref = 0.5 * area * (n_medium / ETA0) * np.where(
        valid, kn.real / k[:, None, None], 0.0)
    p_fwd = pref * (np.where(valid, np.abs(Es_f), 0.0) ** 2
                    + np.where(valid, np.abs(Ep_f), 0.0) ** 2)
    p_bwd = pref * (np.where(valid, np.abs(Es_b), 0.0) ** 2
                    + np.where(valid, np.abs(Ep_b), 0.0) ** 2)

    # ascending-order (fftshifted) presentation
    s1 = np.argsort(m1)
    s2 = np.argsort(m2)

    def shift(a: np.ndarray) -> np.ndarray:
        return a[:, s1, :][:, :, s2]

    return DiffractionOrders(
        axis=axis,
        in_plane_axes=(u1, u2),
        freqs_hz=freqs,
        orders1=m1[s1].astype(int),
        orders2=m2[s2].astype(int),
        k1=k1[s1],
        k2=k2[s2],
        kn=shift(kn),
        propagating=shift(valid),
        amp_s_forward=shift(Es_f),
        amp_p_forward=shift(Ep_f),
        amp_s_backward=shift(Es_b),
        amp_p_backward=shift(Ep_b),
        power_forward=shift(p_fwd),
        power_backward=shift(p_bwd),
        n_medium=n_medium,
        area_um2=area,
        _raw_plane_power=raw_power,
    )
