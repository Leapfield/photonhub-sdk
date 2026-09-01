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

__all__ = ["propagate_plane", "focal_scan", "FocalScan"]

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _plane_arrays(simulation, data, monitor):
    """(e1, e2, coords1, coords2, freqs, axis, (u1, u2)) with the tangential
    E arrays shaped (nf, n1, n2) — mirrors plugins.diffraction's extraction."""
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
