"""Angular-spectrum propagation of a recorded DFT plane — the metalens
focal-spot workflow.

A ``field_dft`` plane recorded on the TRANSMISSION side of a device (in a
homogeneous region) determines the field everywhere downstream: decompose the
tangential E into plane waves by FFT, advance each by ``e^{i k_n dz}``
(evanescent components decay), and reconstruct. This is EXACT for a
homogeneous half-space — no paraxial or far-field approximation — which is
precisely the metalens question: where does the transmitted field focus, how
tight is the spot, what is the peak intensity?

* :func:`propagate_plane` — the complex transverse E (plus the
  divergence-completed ``E_n``) at one or many ``dz`` offsets;
* :func:`focal_scan` — sweep ``dz``, return the on-axis / peak intensity
  curve, the focal plane, and spot metrics (peak z, FWHM along both axes).

Assumptions: uniform grid over the plane; the field has decayed at the
window edge (the FFT is periodic — the plane is zero-padded 2x to keep
wrap-around negligible; a plane clipped mid-beam will show artifacts);
homogeneous (non-dispersive at each frequency) medium of index ``n`` filling
the region swept by ``dz``. Yee half-cell offsets of the two tangential E
components are removed spectrally (exact), so the returned fields are
node-collocated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ._constants import _TANGENTIAL, C0
from .diffraction import _CYCLIC, _infer_n_medium, _resolve_monitor

__all__ = ["propagate_plane", "focal_scan", "FocalScan", "focal_metrics", "FocalMetrics"]

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _plane_arrays(simulation, data, monitor):
    """(e1, e2, coords1, coords2, freqs, axis, (u1, u2)) with the tangential
    E arrays shaped (nf, n1, n2) — mirrors analysis.diffraction's extraction."""
    mon = _resolve_monitor(simulation, monitor)
    size = mon.size_um
    zero_axes = [i for i, s in enumerate(size) if s == 0.0]
    if len(zero_axes) != 1:
        raise ValueError(
            f"monitor {mon.name!r} must be a plane (exactly one zero size)")
    axis = "xyz"[zero_axes[0]]
    u1, u2 = _CYCLIC[axis]
    comps = _TANGENTIAL[axis]
    have = set(getattr(mon, "fields", ()))
    if not {comps[0], comps[1]} <= have:
        raise ValueError(
            f"monitor {mon.name!r} must record the tangential E pair "
            f"{comps[0]}/{comps[1]}")
    da = data[mon.name]
    freqs = np.asarray(da.coords["f"].values, dtype=np.float64)
    p1 = np.asarray(da.coords[u1].values, dtype=np.float64)
    p2 = np.asarray(da.coords[u2].values, dtype=np.float64)
    for c, nm in ((p1, u1), (p2, u2)):
        d = np.diff(c)
        if c.size < 2 or not np.allclose(d, d[0], rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"in-plane axis {nm!r} must be uniformly sampled")

    def plane(comp):
        arr = da.sel(component=comp).transpose("f", axis, u2, u1)
        vals = np.asarray(arr.values)
        return np.transpose(vals[:, 0, :, :], (0, 2, 1))  # (f, n1, n2)

    pos_a = float(np.asarray(da.coords[axis].values).reshape(-1)[0])
    return (plane(comps[0]), plane(comps[1]), p1, p2, freqs, axis, (u1, u2),
            mon, pos_a)


def propagate_plane(
    simulation,
    data,
    monitor,
    dz_um,
    *,
    n_medium: Optional[float] = None,
    direction: str = "+",
) -> Dict[str, np.ndarray]:
    """Reconstruct the node-collocated complex E at offset(s) ``dz_um``
    downstream (``direction`` '+' = toward +axis) of a recorded plane.

    Returns ``{"e1", "e2", "en", "coords1_um", "coords2_um", "dz_um",
    "freqs_hz"}`` — each field shaped ``(n_dz, nf, n1, n2)`` with ``e1/e2``
    the in-plane components along the plane's cyclic transverse axes and
    ``en`` the normal component completed from ``div E = 0``.
    """
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    (e1_r, e2_r, p1, p2, freqs, axis, (u1, u2), mon,
     pos_a) = _plane_arrays(simulation, data, monitor)
    if n_medium is None:
        n_medium = _infer_n_medium(simulation, axis, pos_a)
    n_medium = float(n_medium)

    d1 = float(p1[1] - p1[0])
    d2 = float(p2[1] - p2[0])
    n1, n2 = p1.size, p2.size
    # zero-pad 2x against the FFT's periodic wrap
    N1, N2 = 2 * n1, 2 * n2
    k1 = 2.0 * math.pi * np.fft.fftfreq(N1, d=d1)
    k2 = 2.0 * math.pi * np.fft.fftfreq(N2, d=d2)
    K1 = k1[:, None]
    K2 = k2[None, :]

    dz = np.atleast_1d(np.asarray(dz_um, dtype=np.float64))
    sgn = 1.0 if direction == "+" else -1.0

    out1 = np.empty((dz.size, freqs.size, n1, n2), dtype=np.complex128)
    out2 = np.empty_like(out1)
    outn = np.empty_like(out1)
    for fi, f in enumerate(freqs):
        k = 2.0 * math.pi * n_medium * f / C0 * 1e-6  # rad/um
        kn2 = k * k - (K1 * K1 + K2 * K2)
        kn = np.sqrt(kn2.astype(np.complex128))
        kn = np.where(kn.real < 0, -kn, kn)
        kn = np.where((kn.real == 0) & (kn.imag < 0), np.conj(kn), kn)

        buf1 = np.zeros((N1, N2), dtype=np.complex128)
        buf2 = np.zeros_like(buf1)
        buf1[:n1, :n2] = e1_r[fi]
        buf2[:n1, :n2] = e2_r[fi]
        S1 = np.fft.fft2(buf1)
        S2 = np.fft.fft2(buf2)
        # exact Yee de-stagger: E_u1 recorded at +d1/2 along u1, E_u2 at
        # +d2/2 along u2 — refer both to the base node
        S1 = S1 * np.exp(-1j * K1 * (d1 / 2.0))
        S2 = S2 * np.exp(-1j * K2 * (d2 / 2.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            Sn = np.where(np.abs(kn) > 1e-12 * k,
                          -(K1 * S1 + K2 * S2) / (sgn * kn), 0.0)
        for zi, z in enumerate(dz):
            # propagating: phase advance along the travel direction;
            # evanescent: ALWAYS decay with |dz| (the inverse — exponential
            # re-amplification — is ill-posed and would explode numerical
            # noise; this is the standard band-limited angular spectrum)
            ph = np.exp(1j * sgn * kn.real * z) * \
                np.exp(-np.abs(kn.imag) * abs(z))
            out1[zi, fi] = np.fft.ifft2(S1 * ph)[:n1, :n2]
            out2[zi, fi] = np.fft.ifft2(S2 * ph)[:n1, :n2]
            outn[zi, fi] = np.fft.ifft2(Sn * ph)[:n1, :n2]

    return {
        "e1": out1, "e2": out2, "en": outn,
        "coords1_um": p1.copy(), "coords2_um": p2.copy(),
        "dz_um": dz.copy(), "freqs_hz": freqs.copy(),
        "axes": (u1, u2), "n_medium": n_medium,
    }


@dataclass(frozen=True)
class FocalScan:
    """Result of :func:`focal_scan`: intensity through a z-sweep of one
    recorded frequency. ``intensity`` is ``|E|^2`` (all three components)
    shaped ``(n_dz, n1, n2)``."""

    dz_um: np.ndarray
    coords1_um: np.ndarray
    coords2_um: np.ndarray
    intensity: np.ndarray
    freq_hz: float
    n_medium: float

    @property
    def peak_intensity_per_z(self) -> np.ndarray:
        return self.intensity.reshape(self.dz_um.size, -1).max(axis=1)

    @property
    def peak_dz_um(self) -> float:
        return float(self.dz_um[int(np.argmax(self.peak_intensity_per_z))])

    def focal_plane(self) -> np.ndarray:
        """Intensity map at the peak plane, shape (n1, n2)."""
        return self.intensity[int(np.argmax(self.peak_intensity_per_z))]

    def fwhm_um(self) -> Tuple[float, float]:
        """FWHM of the focal-plane spot along the two in-plane axes, through
        the peak pixel (linear interpolation between samples)."""
        plane = self.focal_plane()
        i1, i2 = np.unravel_index(int(np.argmax(plane)), plane.shape)

        def fwhm(profile, coords):
            half = float(profile.max()) / 2.0
            above = profile >= half
            idx = np.where(above)[0]
            if idx.size == 0:
                return math.nan
            lo, hi = idx[0], idx[-1]

            def edge(a, b):
                if a < 0 or b >= profile.size or profile[b] == profile[a]:
                    return coords[max(min(b, profile.size - 1), 0)]
                t = (half - profile[a]) / (profile[b] - profile[a])
                return coords[a] + t * (coords[b] - coords[a])

            left = edge(lo - 1, lo) if lo > 0 else coords[0]
            right = edge(hi + 1, hi) if hi < profile.size - 1 else coords[-1]
            return float(abs(right - left))

        return (fwhm(plane[:, i2], self.coords1_um),
                fwhm(plane[i1, :], self.coords2_um))


def focal_scan(
    simulation,
    data,
    monitor,
    *,
    dz_um,
    freq_hz: Optional[float] = None,
    n_medium: Optional[float] = None,
    direction: str = "+",
) -> FocalScan:
    """Sweep ``dz_um`` downstream of a recorded plane and return the
    intensity volume + spot metrics for ONE recorded frequency (``freq_hz``
    defaults to the monitor's single frequency; required when it records
    several)."""
    (e1_r, _e2, _p1, _p2, freqs, _axis, _axes, mon,
     _pos) = _plane_arrays(simulation, data, monitor)
    if freq_hz is None:
        if freqs.size != 1:
            raise ValueError(
                f"monitor {mon.name!r} records {freqs.size} frequencies — "
                "pass freq_hz")
        freq_hz = float(freqs[0])
    fi = int(np.argmin(np.abs(freqs - float(freq_hz))))
    if not np.isclose(freqs[fi], float(freq_hz), rtol=1e-9):
        raise ValueError(
            f"freq_hz {freq_hz:g} not among the recorded frequencies")

    res = propagate_plane(simulation, data, monitor, dz_um,
                          n_medium=n_medium, direction=direction)
    sub = slice(fi, fi + 1)
    inten = (np.abs(res["e1"][:, sub]) ** 2 + np.abs(res["e2"][:, sub]) ** 2
             + np.abs(res["en"][:, sub]) ** 2)[:, 0]
    return FocalScan(
        dz_um=res["dz_um"], coords1_um=res["coords1_um"],
        coords2_um=res["coords2_um"], intensity=inten,
        freq_hz=float(freqs[fi]), n_medium=res["n_medium"])


# ---------------------------------------------------------------------------
# Focal-spot metrics with the Poynting flux: the lens-example readout
# ---------------------------------------------------------------------------
@dataclass
class FocalMetrics:
    """Result of :func:`focal_metrics`. Powers are in the recorded phasors'
    own units (E^2 / (2 eta0) times um^2), the same convention as
    :func:`~photonhub.analysis.diffraction.diffraction_orders`, so ratios
    against a forward power read the same way are the physical numbers."""

    z_peak_um: float                 # plane of peak S_z, downstream of the recorded plane
    peak_um: tuple[float, float]     # in-plane position of the peak
    fwhm_um: float                   # 2-D Gaussian fit to S_z at the plane of focus (mean of the two axes)
    fwhm_x_um: float
    fwhm_y_um: float
    fwhm_method: str                 # "gaussian" or "linear" (fit failed)
    efficiency: float                # power through the aperture of radius aperture_fwhm x FWHM / incident_power
    aperture_radius_um: float
    transmission: float | None    # power through the disc at the recorded plane / incident_power (if a disc was given)
    incident_power: float
    power_aperture: float
    power_forward_total: float       # total forward power of the recorded plane (all propagating orders)
    intensity: np.ndarray            # S_z at the plane of focus, shape (n1, n2)
    coords1_um: np.ndarray
    coords2_um: np.ndarray
    cut1: np.ndarray                 # S_z through the peak along axis 1, normalized to the peak
    cut2: np.ndarray
    scan_dz_um: np.ndarray
    scan_peak: np.ndarray            # peak S_z per scanned plane, normalized

    @property
    def focal_cut(self):
        """(offset_um, normalized S_z) along the first in-plane axis through the peak."""
        return self.coords1_um - self.peak_um[0], self.cut1


class _Spectrum:
    """Angular spectrum of ONE recorded frequency of a periodic plane (period =
    the plane), with the Yee stagger removed and H rebuilt per plane-wave order
    from Maxwell's equations, so S_z downstream is exact for forward waves."""

    def __init__(self, e1, e2, p1, p2, k):
        self.n1, self.n2 = e1.shape
        self.d1 = float(p1[1] - p1[0])
        self.d2 = float(p2[1] - p2[0])
        self.area = self.n1 * self.d1 * self.n2 * self.d2
        self.k = float(k)
        self.k1 = 2.0 * math.pi * np.fft.fftfreq(self.n1, d=self.d1)
        self.k2 = 2.0 * math.pi * np.fft.fftfreq(self.n2, d=self.d2)
        K1, K2 = self.k1[:, None], self.k2[None, :]
        self.E1 = np.fft.fft2(e1) * np.exp(-1j * K1 * self.d1 / 2)   # E_u1 sits half a cell along u1
        self.E2 = np.fft.fft2(e2) * np.exp(-1j * K2 * self.d2 / 2)
        kz = np.sqrt((self.k ** 2 - K1 ** 2 - K2 ** 2).astype(np.complex128))
        kz = np.where(kz.real < 0, -kz, kz)
        kz = np.where((kz.real == 0) & (kz.imag < 0), np.conj(kz), kz)
        self.kz = kz
        with np.errstate(invalid="ignore", divide="ignore"):
            self.En = np.where(np.abs(kz) > 1e-12 * self.k, -(K1 * self.E1 + K2 * self.E2) / kz, 0.0)

    def forward_power(self, n):
        N = self.n1 * self.n2
        e2 = (np.abs(self.E1) ** 2 + np.abs(self.E2) ** 2 + np.abs(self.En) ** 2) / N ** 2
        return float(self.area * n / (2 * ETA0) * np.sum(np.where(self.kz.real > 0, (self.kz.real / self.k) * e2, 0.0)))

    def sz_at(self, dz, n):
        prop = np.exp(1j * self.kz * dz)
        E1, E2, En = self.E1 * prop, self.E2 * prop, self.En * prop
        K1, K2 = self.k1[:, None], self.k2[None, :]
        c = n / (self.k * ETA0)
        H1 = c * (K2 * En - self.kz * E2)
        H2 = c * (self.kz * E1 - K1 * En)
        e1, e2, h1, h2 = (np.fft.ifft2(a) for a in (E1, E2, H1, H2))
        return 0.5 * np.real(e1 * np.conj(h2) - e2 * np.conj(h1))


ETA0 = 376.730313668


def _gaussian_fwhm(sz, p1, p2, half_win_um=2.5):
    i0, j0 = np.unravel_index(int(np.argmax(sz)), sz.shape)
    x0, y0 = float(p1[i0]), float(p2[j0])

    def linear(profile, coords):
        half = float(profile.max()) / 2.0
        idx = np.where(profile >= half)[0]
        lo, hi = idx[0], idx[-1]

        def edge(a, b):
            if a < 0 or b >= profile.size or profile[b] == profile[a]:
                return coords[max(min(b, profile.size - 1), 0)]
            t = (half - profile[a]) / (profile[b] - profile[a])
            return coords[a] + t * (coords[b] - coords[a])
        return float(abs(edge(hi + 1, hi) - edge(lo - 1, lo)))

    fx_lin, fy_lin = linear(sz[:, j0], p1), linear(sz[i0, :], p2)
    try:
        from scipy.optimize import curve_fit
        mx = np.abs(p1 - x0) <= half_win_um
        my = np.abs(p2 - y0) <= half_win_um
        X, Y = np.meshgrid(p1[mx], p2[my], indexing="ij")
        Z = sz[np.ix_(mx, my)]

        def g(xy, A, xc, yc, sx, sy):
            xx, yy = xy
            return A * np.exp(-((xx - xc) ** 2) / (2 * sx ** 2) - ((yy - yc) ** 2) / (2 * sy ** 2))

        popt, _ = curve_fit(g, (X.ravel(), Y.ravel()), Z.ravel(),
                            p0=(float(Z.max()), x0, y0, fx_lin / 2.3548, fy_lin / 2.3548), maxfev=20000)
        c = 2.0 * math.sqrt(2.0 * math.log(2.0))
        return (float(popt[1]), float(popt[2]), c * abs(float(popt[3])), c * abs(float(popt[4])),
                "gaussian", i0, j0)
    except (ImportError, RuntimeError, ValueError, TypeError):  # no scipy, or no spot to fit: the linear crossings
        return x0, y0, fx_lin, fy_lin, "linear", i0, j0


def focal_metrics(
    simulation,
    data,
    monitor,
    *,
    dz_um,
    incident_power: float,
    freq_hz: float | None = None,
    n_medium: float | None = None,
    aperture_fwhm: float = 3.0,
    disc_radius_um: float | None = None,
    center_um: tuple[float, float] | None = None,
    refine_um: float = 0.5,
    refine_steps: int = 41,
) -> FocalMetrics:
    """Focal-spot metrics of a lens from its recorded transmitted plane.

    The plane (a full-period ``ProfileMonitor`` with the tangential E pair)
    is decomposed into plane waves, each propagated ``dz`` downstream with
    its paired H, and the Poynting flux ``S_z`` is rebuilt: the plane of
    focus is the plane of peak ``S_z`` over ``dz_um`` (then refined within
    ``refine_um``), the FWHM is a 2-D Gaussian fit to ``S_z`` there, and the
    focusing efficiency is the power through a circular aperture of radius
    ``aperture_fwhm`` x FWHM around the peak over ``incident_power`` — the
    definitions used by the metalens literature. ``transmission`` is the
    power through a disc of ``disc_radius_um`` about ``center_um`` at the
    recorded plane over ``incident_power`` (the lens's own transmission).
    ``incident_power`` must be read in the same units, e.g. the forward power
    of an input plane from :func:`~photonhub.analysis.diffraction_orders`."""
    (e1_r, e2_r, p1, p2, freqs, axis, (_u1, _u2), mon,
     pos_a) = _plane_arrays(simulation, data, monitor)
    if freq_hz is None:
        if freqs.size != 1:
            raise ValueError(f"monitor {mon.name!r} records {freqs.size} frequencies — pass freq_hz")
        freq_hz = float(freqs[0])
    fi = int(np.argmin(np.abs(freqs - float(freq_hz))))
    if n_medium is None:
        n_medium = _infer_n_medium(simulation, axis, pos_a)
    n = float(n_medium)
    k = 2.0 * math.pi * n * freqs[fi] / C0 * 1e-6
    spec = _Spectrum(e1_r[fi], e2_r[fi], p1, p2, k)
    if center_um is None:
        center_um = (float(p1.mean()), float(p2.mean()))
    P1, P2 = np.meshgrid(p1, p2, indexing="ij")
    dA = spec.d1 * spec.d2

    dz = np.atleast_1d(np.asarray(dz_um, dtype=np.float64))
    peaks = np.array([float(spec.sz_at(float(z), n).max()) for z in dz])
    zc = float(dz[int(np.argmax(peaks))])
    if refine_um > 0 and refine_steps > 1:
        fine = np.linspace(max(0.0, zc - refine_um), zc + refine_um, int(refine_steps))
        pf = np.array([float(spec.sz_at(float(z), n).max()) for z in fine])
        zc = float(fine[int(np.argmax(pf))])
        dz_all, pk_all = np.concatenate([dz, fine]), np.concatenate([peaks, pf])
        order = np.argsort(dz_all)
        dz_all, pk_all = dz_all[order], pk_all[order]
    else:
        dz_all, pk_all = dz, peaks
    sz = spec.sz_at(zc, n)
    xc, yc, fx, fy, method, i0, j0 = _gaussian_fwhm(sz, p1, p2)
    fwhm = 0.5 * (fx + fy)
    r_ap = aperture_fwhm * fwhm
    P_ap = float(np.sum(sz[(P1 - xc) ** 2 + (P2 - yc) ** 2 <= r_ap ** 2]) * dA)
    T = None
    if disc_radius_um is not None:
        sz0 = spec.sz_at(0.0, n)
        P_disc = float(np.sum(sz0[(P1 - center_um[0]) ** 2 + (P2 - center_um[1]) ** 2 <= disc_radius_um ** 2]) * dA)
        T = P_disc / float(incident_power)
    return FocalMetrics(
        z_peak_um=zc, peak_um=(xc, yc), fwhm_um=fwhm, fwhm_x_um=fx, fwhm_y_um=fy, fwhm_method=method,
        efficiency=P_ap / float(incident_power), aperture_radius_um=r_ap, transmission=T,
        incident_power=float(incident_power), power_aperture=P_ap,
        power_forward_total=spec.forward_power(n), intensity=sz, coords1_um=p1, coords2_um=p2,
        cut1=sz[:, j0] / float(sz.max()), cut2=sz[i0, :] / float(sz.max()),
        scan_dz_um=dz_all, scan_peak=pk_all / float(pk_all.max()))
