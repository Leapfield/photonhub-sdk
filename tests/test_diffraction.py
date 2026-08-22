"""Grating-order decomposition — exact pins on SYNTHETIC periodic planes.

No FDTD run: each test stamps an analytic superposition of Floquet orders on
the Yee-staggered sample points of a periodic plane (with the exact in-plane
half-cell offsets and the half-cell normal offset of H), shaped like a real
``FieldDftMonitor`` DataArray, and requires :func:`diffraction_orders` to
recover it exactly:

* injected (order, polarization, direction) amplitudes to ~1e-10, every other
  propagating order ~0;
* forward/backward separation with both directions present at once;
* evanescent orders masked (NaN amplitude, zero power, no real plane power);
* energy closure: ``total_power('+') - total_power('-') == plane_power()``;
* diffraction angles; the physics-precondition guards (non-periodic
  boundaries, plane through a structure, missing components, decimation,
  in-plane symmetry).

The synthesis uses the SAME conventions the module documents (engine
``e^{-i w t}`` phasors, forward = ``e^{+i k_n a}``, s_hat = n_hat x kt_hat,
p amplitudes as full-E magnitudes), derived independently from the plane-wave
relations — a sign error on either side breaks the exact-recovery pins.
"""

import math

import numpy as np
import pytest
import xarray as xr

import photonhub as ph
from photonhub.plugins import diffraction_orders
from photonhub.plugins.mode_overlap import ETA0

C0 = 2.99792458e8

# Periodic cell: 8 x 6 cells of 0.05 um -> L1 = 0.4, L2 = 0.3 um.
DL = 0.05
N1, N2 = 8, 6
L1, L2 = N1 * DL, N2 * DL
SZ = 0.4
Z0 = 0.2                       # base Yee plane coordinate (index 4)
N_MED = 1.5
WL_UM = 0.35                   # k = 26.93 rad/um: m1,m2 = +/-1 propagate,
F0 = C0 / (WL_UM * 1e-6)       # |m1| >= 2 evanescent


def _sim(structures=(), background_eps=N_MED**2, boundaries=None,
         monitor_kwargs=None):
    mk = dict(
        name="orders",
        center_um=(L1 / 2, L2 / 2, Z0),
        size_um=(L1, L2, 0.0),
        fields=("Ex", "Ey", "Hx", "Hy"),
        freqs_hz=(F0,),
    )
    mk.update(monitor_kwargs or {})
    return ph.Simulation(
        size_um=(L1, L2, SZ),
        grid=ph.UniformGridSpec(dl_um=DL),
        run=ph.RunSpec(n_steps=10),
        boundaries=boundaries or ph.Boundaries(x="periodic", y="periodic",
                                               z="pml"),
        background=ph.Background(permittivity=background_eps),
        structures=list(structures),
        sources=[ph.PointDipole(
            center_um=(L1 / 2, L2 / 2, Z0), polarization="Ex",
            source_time=ph.GaussianPulse(freq0_hz=F0, fwidth_hz=0.1 * F0))],
        monitors=[ph.FieldDftMonitor(**mk)],
    )


def _k(f=F0, n=N_MED):
    return 2.0 * math.pi * n / C0 * 1e-6 * f


def _stamp(waves, f=F0, n=N_MED, d_a=DL):
    """Analytic staggered plane for a superposition of orders.

    ``waves``: iterable of (m1, m2, pol, direction, amplitude) with pol in
    {'s','p'} (amplitude = full-E complex amplitude) and direction '+'/'-'.
    Returns the four (N2, N1)-shaped arrays keyed 'Ex','Ey','Hx','Hy'
    (z-normal plane -> u1=x, u2=y).
    """
    x = DL * np.arange(N1)
    y = DL * np.arange(N2)
    # staggered sample coordinates per component
    XE1, YE1 = np.meshgrid(x + DL / 2, y)        # Ex at (i+1/2, j)
    XE2, YE2 = np.meshgrid(x, y + DL / 2)        # Ey at (i, j+1/2)
    XH1, YH1 = np.meshgrid(x, y + DL / 2)        # Hx at (i, j+1/2) [+ z off]
    XH2, YH2 = np.meshgrid(x + DL / 2, y)        # Hy at (i+1/2, j) [+ z off]

    k = _k(f, n)
    out = {c: np.zeros((N2, N1), dtype=np.complex128)
           for c in ("Ex", "Ey", "Hx", "Hy")}
    for m1, m2, pol, direction, amp in waves:
        k1 = 2.0 * math.pi * m1 / L1
        k2 = 2.0 * math.pi * m2 / L2
        kt = math.hypot(k1, k2)
        kn = np.sqrt(complex(k * k - kt * kt))
        if kn.real < 0:
            kn = -kn
        sgn = +1.0 if direction == "+" else -1.0
        if kt > 0:
            c1, c2 = k1 / kt, k2 / kt
        else:
            c1, c2 = 1.0, 0.0
        s1, s2 = -c2, c1                          # s_hat = n_hat x kt_hat
        hoff = np.exp(1j * sgn * kn * d_a / 2.0)  # H recorded off-plane

        def ph_at(X, Y):
            return np.exp(1j * (k1 * X + k2 * Y))

        if pol == "s":
            # E = A s_hat ; H_t = -sgn * A (n/eta0)(kn/k) kt_hat
            ht = -sgn * amp * (n / ETA0) * (kn / k)
            out["Ex"] += amp * s1 * ph_at(XE1, YE1)
            out["Ey"] += amp * s2 * ph_at(XE2, YE2)
            out["Hx"] += ht * c1 * hoff * ph_at(XH1, YH1)
            out["Hy"] += ht * c2 * hoff * ph_at(XH2, YH2)
        elif pol == "p":
            # E_t = sgn * A (kn/k) kt_hat ; H_t = A (n/eta0) s_hat
            et = sgn * amp * (kn / k)
            hp = amp * (n / ETA0)
            out["Ex"] += et * c1 * ph_at(XE1, YE1)
            out["Ey"] += et * c2 * ph_at(XE2, YE2)
            out["Hx"] += hp * s1 * hoff * ph_at(XH1, YH1)
            out["Hy"] += hp * s2 * hoff * ph_at(XH2, YH2)
        else:
            raise ValueError(pol)
    return out


def _data(planes, f=F0):
    comps = ("Ex", "Ey", "Hx", "Hy")
    arr = np.stack([planes[c] for c in comps])   # (component, y, x)
    arr = arr[None, :, None, :, :]               # (f, component, z, y, x)
    da = xr.DataArray(
        arr,
        dims=("f", "component", "z", "y", "x"),
        coords={"f": [f], "component": list(comps), "z": [Z0],
                "y": DL * np.arange(N2), "x": DL * np.arange(N1)},
    )
    return {"orders": da}


def _amps(result, m1, m2):
    o = result.order(m1, m2)
    return (o["amp_s_forward"][0], o["amp_p_forward"][0],
            o["amp_s_backward"][0], o["amp_p_backward"][0])


# ---------------------------------------------------------------------------

def test_single_forward_s_order_recovered_exactly():
    amp = 0.8 * np.exp(1j * 0.7)
    res = diffraction_orders(_sim(), _data(_stamp([(1, 0, "s", "+", amp)])),
                             "orders")
    s_f, p_f, s_b, p_b = _amps(res, 1, 0)
    assert s_f == pytest.approx(amp, rel=1e-10)
    assert abs(p_f) < 1e-10 and abs(s_b) < 1e-10 and abs(p_b) < 1e-10
    # every other propagating order is empty
    for m1 in res.orders1:
        for m2 in res.orders2:
            if (m1, m2) == (1, 0):
                continue
            o = res.order(int(m1), int(m2))
            if o["propagating"][0]:
                for key in ("amp_s_forward", "amp_p_forward",
                            "amp_s_backward", "amp_p_backward"):
                    assert abs(o[key][0]) < 1e-10, (m1, m2, key)
    # angle of the (1,0) order
    k = _k()
    assert res.order(1, 0)["theta_rad"][0] == pytest.approx(
        math.asin(2 * math.pi / L1 / k), rel=1e-12)


def test_mixed_orders_directions_and_polarizations():
    waves = [
        (0, 0, "s", "+", 1.0 + 0.0j),
        (0, 0, "p", "+", 0.25j),
        (1, 0, "p", "+", 0.5 * np.exp(1j * 1.1)),
        (0, -1, "s", "-", 0.4 * np.exp(-1j * 0.3)),
        (-1, 1, "p", "-", 0.3 + 0.2j),
    ]
    res = diffraction_orders(_sim(), _data(_stamp(waves)), "orders")

    s_f, p_f, s_b, p_b = _amps(res, 0, 0)
    assert s_f == pytest.approx(1.0 + 0j, abs=1e-10)
    assert p_f == pytest.approx(0.25j, abs=1e-10)
    assert abs(s_b) < 1e-10 and abs(p_b) < 1e-10

    s_f, p_f, s_b, p_b = _amps(res, 1, 0)
    assert p_f == pytest.approx(0.5 * np.exp(1j * 1.1), rel=1e-10)
    assert abs(s_f) < 1e-10 and abs(s_b) < 1e-10 and abs(p_b) < 1e-10

    s_f, p_f, s_b, p_b = _amps(res, 0, -1)
    assert s_b == pytest.approx(0.4 * np.exp(-1j * 0.3), rel=1e-10)
    assert abs(s_f) < 1e-10 and abs(p_f) < 1e-10 and abs(p_b) < 1e-10

    s_f, p_f, s_b, p_b = _amps(res, -1, 1)
    assert p_b == pytest.approx(0.3 + 0.2j, rel=1e-10)


def test_counterpropagating_same_order_split():
    a_f, a_b = 0.9 * np.exp(1j * 0.2), 0.35 * np.exp(-1j * 1.4)
    waves = [(1, 0, "s", "+", a_f), (1, 0, "s", "-", a_b)]
    res = diffraction_orders(_sim(), _data(_stamp(waves)), "orders")
    s_f, p_f, s_b, p_b = _amps(res, 1, 0)
    assert s_f == pytest.approx(a_f, rel=1e-10)
    assert s_b == pytest.approx(a_b, rel=1e-10)
    assert abs(p_f) < 1e-10 and abs(p_b) < 1e-10


def test_energy_closure_and_power_values():
    waves = [
        (0, 0, "s", "+", 1.0),
        (1, 0, "p", "+", 0.5),
        (0, 1, "s", "-", 0.4),
        (2, 0, "s", "+", 0.7),      # evanescent: no real power
    ]
    res = diffraction_orders(_sim(), _data(_stamp(waves)), "orders")

    k = _k()
    area = L1 * L2

    def expected_power(m1, m2, amp):
        kt = math.hypot(2 * math.pi * m1 / L1, 2 * math.pi * m2 / L2)
        kn = math.sqrt(k * k - kt * kt)
        return 0.5 * area * (N_MED / ETA0) * (kn / k) * abs(amp) ** 2

    assert res.order(0, 0)["power_forward"][0] == pytest.approx(
        expected_power(0, 0, 1.0), rel=1e-10)
    assert res.order(1, 0)["power_forward"][0] == pytest.approx(
        expected_power(1, 0, 0.5), rel=1e-10)
    assert res.order(0, 1)["power_backward"][0] == pytest.approx(
        expected_power(0, 1, 0.4), rel=1e-10)

    # evanescent order: masked amplitude, zero power
    o = res.order(2, 0)
    assert not o["propagating"][0]
    assert np.isnan(o["amp_s_forward"][0].real)
    assert o["power_forward"][0] == 0.0

    # net physical power is the directional difference
    assert res.net_power()[0] == pytest.approx(
        expected_power(0, 0, 1.0) + expected_power(1, 0, 0.5)
        - expected_power(0, 1, 0.4), rel=1e-10)

    # independent bookkeeping cross-check: the raw half-cell-staggered
    # real-space integral equals the per-order net weighted by
    # cos(k_n d_a / 2) — the module's documented stagger relation. Computed
    # here from the injected waves, entirely outside the module's FFT path.
    def cosw(m1, m2):
        kt = math.hypot(2 * math.pi * m1 / L1, 2 * math.pi * m2 / L2)
        kn = math.sqrt(k * k - kt * kt)
        return math.cos(kn * DL / 2.0)

    expected_raw = (expected_power(0, 0, 1.0) * cosw(0, 0)
                    + expected_power(1, 0, 0.5) * cosw(1, 0)
                    - expected_power(0, 1, 0.4) * cosw(0, 1))
    assert res.staggered_plane_power()[0] == pytest.approx(
        expected_raw, rel=1e-9)


def test_guards():
    # non-periodic in-plane boundary
    sim_pml = _sim(boundaries=ph.Boundaries(x="pml", y="periodic", z="pml"))
    with pytest.raises(ValueError, match="periodic"):
        diffraction_orders(sim_pml, _data(_stamp([])), "orders")

    # plane crossing a structure -> n inference refuses
    bar = ph.Structure(
        geometry=ph.Box(center_um=(L1 / 2, L2 / 2, Z0),
                        size_um=(L1 / 2, L2, 0.2)),
        medium=ph.Medium(permittivity=4.0))
    sim_bar = _sim(structures=[bar])
    with pytest.raises(ValueError, match="crosses"):
        diffraction_orders(sim_bar, _data(_stamp([])), "orders")
    # ... but an explicit n_medium overrides
    res = diffraction_orders(sim_bar, _data(_stamp([(0, 0, "s", "+", 1.0)])),
                             "orders", n_medium=N_MED)
    assert res.n_medium == N_MED

    # missing tangential components
    sim_partial = _sim(monitor_kwargs=dict(fields=("Ex", "Ey")))
    with pytest.raises(ValueError, match="tangential"):
        diffraction_orders(sim_partial, _data(_stamp([])), "orders")

    # in-plane decimation
    sim_dec = _sim(monitor_kwargs=dict(interval_space=(2, 1, 1)))
    with pytest.raises(ValueError, match="interval_space"):
        diffraction_orders(sim_dec, _data(_stamp([])), "orders")

    # unknown monitor name
    with pytest.raises(ValueError, match="not found"):
        diffraction_orders(_sim(), _data(_stamp([])), "nope")

    # unknown order lookup
    res = diffraction_orders(_sim(), _data(_stamp([])), "orders")
    with pytest.raises(KeyError):
        res.order(50, 0)


def test_symmetry_fold_rejected():
    sim = _sim()
    sim = sim.model_copy(update={"symmetry": (0, 1, 0)})
    with pytest.raises(ValueError, match="symmetry"):
        diffraction_orders(sim, _data(_stamp([])), "orders")
