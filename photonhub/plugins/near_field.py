"""Near-to-far-field (NTFF) projection — a recorded near-field surface -> the
far-zone radiation pattern, via the surface-equivalence theorem.

This is a pure-host post-processor (no engine / C++ change). Given the
frequency-domain tangential ``E`` and ``H`` recorded by a ``FieldDftMonitor``
on a plane (or the six faces of a box), it computes the far field
``E_theta(theta, phi, f)``, ``E_phi(theta, phi, f)``, the radiation intensity
``U = r^2 |E|^2 / (2 eta0)`` and the (optionally) normalized directivity, in
arbitrary directions ``(theta, phi)``. Tidy3D's analogue is
``FieldProjectionMonitor``; Lumerical's is the far-field projection.

Physics / method
================
**Surface equivalence theorem.** On a closed surface with outward unit normal
``n_hat`` carrying the simulated near fields ``(E, H)``, the equivalent surface
currents that reproduce the *exterior* field are

    J = n_hat x H            (electric current)
    M = -n_hat x E           (magnetic current)

For an OPEN surface (a single plane just above a device) the same currents are
used; the projection is then a Kirchhoff/equivalence approximation of the
upper-half-space radiation (exact when the plane captures essentially all of
the outgoing field, i.e. the standard "monitor box lid" use). Both faces of a
closed box are summed with their respective outward normals.

**Radiation vectors.** With observation unit vector

    r_hat = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)) ,

the electric and magnetic radiation vectors are the surface Fourier transforms
of the currents,

    N(theta, phi) = integral_S J(r') exp(-i k r_hat . r') dS'
    L(theta, phi) = integral_S M(r') exp(-i k r_hat . r') dS'

with free-space wavenumber ``k = 2 pi f / c0`` (the medium above the device is
assumed vacuum / index 1). The ``-i k`` kernel sign goes with PhotonHub's
``e^{-i omega t}`` time convention (matching ``mode_overlap.py``), for which an
outgoing wave is ``e^{+i k r}/r``: the far-field expansion ``|r - r'| ~ r -
r_hat . r'`` makes ``e^{i k |r-r'|} ~ e^{i k r} e^{-i k r_hat . r'}``. (Balanis
derives these with ``e^{+j omega t}`` / ``e^{-j k r}`` and so writes ``e^{+j k
r_hat.r'}``; flipping the time convention flips the kernel sign.) A ``+x``-tilted
aperture field then steers the beam toward ``+x``, as it must.

**Far-zone fields.** Projecting the radiation vectors onto the spherical
``(theta_hat, phi_hat)`` basis,

    N_theta =  N_x cos(theta) cos(phi) + N_y cos(theta) sin(phi) - N_z sin(theta)
    N_phi   = -N_x sin(phi)            + N_y cos(phi)
    (idem for L)

the far-zone transverse E components are (e.g. Balanis, *Antenna Theory*,
eq. 12-10, adapted to ``e^{-i omega t}``). Balanis writes them with ``e^{+j omega
t}`` / ``e^{-j k r}``; the conversion is ``j -> -i``, which flips BOTH the
exponential and the leading ``j k``, giving

    E_theta = +(i k e^{i k r}) / (4 pi r) * ( L_phi   + eta0 * N_theta )
    E_phi   = -(i k e^{i k r}) / (4 pi r) * ( L_theta - eta0 * N_phi   )

with ``eta0`` the vacuum wave impedance. The ``e^{i k r}/r`` spherical-spreading
factor is *omitted* from the returned ``E_theta``/``E_phi``: we return the
direction-dependent **far-field amplitude** ``F = r e^{-i k r} E`` (units
V, i.e. V/m times metres), from which the radiation intensity is

    U(theta, phi) = r^2 |E|^2 / (2 eta0) = (|F_theta|^2 + |F_phi|^2) / (2 eta0) .

The radiated power ``P_rad = integral U dOmega`` and the directivity
``D = 4 pi U / P_rad`` then follow; the pattern SHAPE and ``D`` are independent
of the dropped prefactor.

**Area element.** ``dS`` is taken from each face's *real* transverse coordinate
spacings (centered-difference cell widths, reusing
:func:`~photonhub.plugins.mode_overlap._cell_widths`), so graded / non-uniform
meshes integrate correctly — no uniform-spacing assumption.

**Yee staggering.** The engine's DFT monitor records each tangential component
at its own Yee node, so the raw plane's E and H are staggered by half a cell
both IN the plane and (for H) half a cell ALONG the normal. Combining them into
``J``/``M`` without interpolation is a first-order-in-``dl`` error that
degrades the J/M cancellation (a spurious backward lobe, and an absolute-phase
error). ``colocate=True`` (the default, mirroring the mode-overlap readout)
fixes the TRANSVERSE part by averaging each component to the in-plane cell node
(:func:`~photonhub.plugins.mode_overlap._colocate_to_node`); pass
``colocate=False`` for synthetic, already-co-located fields. The NORMAL-axis
half-cell E/H stagger is NOT corrected — the recorded plane carries a single
normal coordinate and no grid spacing, so the required ``e^{i k cos(theta)
dl/2}``-type phase is not formable here; a remaining O(k dl/2) phase error on
the ``N`` vs ``L`` balance persists (cf. the longitudinal de-stagger
``mode_overlap.mode_transmission(destagger_dl=...)`` applies for modal
readouts).

Coordinates / units. Monitor coordinates are in **microns** (the PhotonHub
convention); they are converted to metres internally so ``k r_hat . r'`` is
dimensionless and the SI ``eta0`` is consistent. Frequencies are in Hz.

Dependency-light: numpy + the xarray DataArrays the rest of PhotonHub already
produces. No matplotlib, no engine calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from ._constants import _TANGENTIAL, C0, ETA0  # noqa: F401  (C0/ETA0 re-exported)
from .mode_overlap import _TRANSVERSE, _cell_widths, _colocate_to_node

__all__ = [
    "C0",
    "ETA0",
    "FarField",
    "far_field",
    "equivalent_currents",
]

Axis = Literal["x", "y", "z"]

#: Direction-chunk budget for the radiation integral: the dense kernel matrix is
#: [chunk_dirs, n_surface_points]; chunk_dirs is chosen so chunk_dirs*n_pts stays
#: near this many complex entries (~16 MB at 1e6), bounding peak memory for a
#: large monitor projected onto many far-field directions.
_DIR_CHUNK: int = 1_000_000


@dataclass(frozen=True)
class FarField:
    """The projected far field on a direction grid, per frequency.

    Attributes
    ----------
    theta, phi:
        1-D arrays of the polar / azimuthal angles (radians) the field was
        sampled at. The pattern arrays are indexed ``[i_freq, i_dir]`` over a
        FLAT list of ``(theta, phi)`` directions; ``theta``/``phi`` are that
        flat list (so ``theta[j], phi[j]`` is direction ``j``). Use
        :meth:`reshape` if you projected onto a meshed ``(n_theta, n_phi)`` grid.
    freqs_hz:
        1-D array of frequencies (Hz).
    e_theta, e_phi:
        Complex far-field AMPLITUDES ``F = r e^{-i k r} E`` (the spherical
        spreading factor ``e^{i k r}/r`` removed), shape ``(n_freq, n_dir)``.
    """

    theta: np.ndarray
    phi: np.ndarray
    freqs_hz: np.ndarray
    e_theta: np.ndarray
    e_phi: np.ndarray

    @property
    def intensity(self) -> np.ndarray:
        """Radiation intensity ``U = (|F_theta|^2 + |F_phi|^2)/(2 eta0)``
        (W/sr), shape ``(n_freq, n_dir)``."""
        return (np.abs(self.e_theta) ** 2 + np.abs(self.e_phi) ** 2) / (2.0 * ETA0)

    def radiated_power(self) -> np.ndarray:
        """Total radiated power ``P_rad = integral U dOmega`` per frequency
        (shape ``(n_freq,)``), integrated over the sampled directions with a
        ``sin(theta)`` solid-angle weight via :func:`_solid_angle_weights`.

        This is only meaningful when the directions tile a contiguous angular
        region (e.g. a full sphere or a hemisphere mesh); for a sparse cut it is
        a crude quadrature. Used as the directivity normalizer."""
        w = _solid_angle_weights(self.theta, self.phi)
        return np.sum(self.intensity * w[None, :], axis=1)

    def directivity(self) -> np.ndarray:
        """Directivity ``D = 4 pi U / P_rad`` (dimensionless), shape
        ``(n_freq, n_dir)`` — the radiation pattern normalized so its
        solid-angle average is 1 (for an isotropic radiator D==1 everywhere)."""
        p = self.radiated_power()
        p = np.where(p > 0, p, np.nan)
        return 4.0 * np.pi * self.intensity / p[:, None]

    def reshape(self, n_theta: int, n_phi: int) -> Dict[str, np.ndarray]:
        """Reshape the flat per-direction arrays back to a meshed
        ``(n_freq, n_theta, n_phi)`` grid (the layout produced when ``theta``
        and ``phi`` were passed as 1-D axes and meshed by :func:`far_field`).
        Returns a dict with ``e_theta``, ``e_phi``, ``intensity``,
        ``directivity``, ``theta`` (``n_theta,``) and ``phi`` (``n_phi,``)."""
        nf = self.freqs_hz.size
        th = self.theta.reshape(n_theta, n_phi)[:, 0]
        ph = self.phi.reshape(n_theta, n_phi)[0, :]
        return {
            "e_theta": self.e_theta.reshape(nf, n_theta, n_phi),
            "e_phi": self.e_phi.reshape(nf, n_theta, n_phi),
            "intensity": self.intensity.reshape(nf, n_theta, n_phi),
            "directivity": self.directivity().reshape(nf, n_theta, n_phi),
            "theta": th,
            "phi": ph,
        }


def _solid_angle_weights(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Per-direction solid-angle quadrature weights ``dOmega = sin(theta)
    dtheta dphi`` for a flat list of directions that came from meshing two 1-D
    axes (``theta`` varying slowest). Recovers the 1-D axes from the flat list
    via :func:`numpy.unique` and builds centered-difference cell widths in each
    (reusing :func:`~photonhub.plugins.mode_overlap._cell_widths`)."""
    th_ax = np.unique(np.round(theta, 12))
    ph_ax = np.unique(np.round(phi, 12))
    dth = _cell_widths(th_ax)
    dph = _cell_widths(ph_ax) if ph_ax.size > 1 else np.array([2.0 * np.pi])
    # Map each flat direction's (theta, phi) to its axis cell widths.
    th_w = {round(float(t), 12): float(w) for t, w in zip(th_ax, dth)}
    ph_w = {round(float(p), 12): float(w) for p, w in zip(ph_ax, dph)}
    weights = np.empty(theta.size)
    for j in range(theta.size):
        t = float(theta[j])
        weights[j] = (np.sin(t) * th_w[round(t, 12)]
                      * ph_w[round(float(phi[j]), 12)])
    return weights


def _directions(
    theta: Union[float, Sequence[float], np.ndarray],
    phi: Union[float, Sequence[float], np.ndarray],
    mesh: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """Normalize the (theta, phi) request into flat 1-D direction arrays.

    If ``mesh`` (default), ``theta`` and ``phi`` are treated as 1-D axes and
    meshed (``theta`` slowest) into ``n_theta*n_phi`` directions; the returned
    shape tuple ``(n_theta, n_phi)`` enables :meth:`FarField.reshape`. If not
    ``mesh``, ``theta`` and ``phi`` must be the same length and are paired
    elementwise (an arbitrary list of directions)."""
    th = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    ph = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    if mesh:
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        return TH.reshape(-1), PH.reshape(-1), (th.size, ph.size)
    if th.size != ph.size:
        raise ValueError(
            f"with mesh=False, theta and phi must be the same length; got "
            f"{th.size} and {ph.size}")
    return th, ph, None


def _plane_component_3d(
    da: xr.DataArray,
    name: str,
    freq_hz: float,
    t1: str,
    t2: str,
    normal: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pull one tangential component off a DFT plane DataArray and return it as a
    2-D complex ``[i_t2, i_t1]`` array, its ``(t1, t2)`` coordinate axes (µm) and
    the plane's normal coordinate (µm).

    Mirrors ``mode_overlap._plane_component`` but also recovers the normal
    position so the radiation kernel ``exp(i k r_hat . r')`` carries the real
    3-D location of the face."""
    if "component" in getattr(da, "dims", ()):
        comp_coords = list(da.coords.get("component", []))
        da = da.sel(component=name) if name in comp_coords \
            else da.squeeze("component", drop=True)
    if "f" in getattr(da, "dims", ()):
        da = da.sel(f=freq_hz, method="nearest")
    # Recover the normal coordinate before squeezing it away.
    normal_um = 0.0
    if normal in getattr(da, "coords", {}):
        nc = np.asarray(da.coords[normal].values, dtype=np.float64).ravel()
        if nc.size >= 1:
            normal_um = float(nc[0])
    da = da.squeeze(drop=True)
    if set(da.dims) != {t1, t2}:
        raise ValueError(
            f"component {name!r}: after reduction dims are {tuple(da.dims)}, "
            f"expected the two transverse axes {{{t1!r}, {t2!r}}}")
    da = da.transpose(t2, t1)
    vals = np.asarray(da.values, dtype=np.complex128)
    c1 = np.asarray(da.coords[t1].values, dtype=np.float64)
    c2 = np.asarray(da.coords[t2].values, dtype=np.float64)
    return vals, c1, c2, normal_um


def equivalent_currents(
    plane_fields: Mapping[str, xr.DataArray],
    *,
    axis: Axis,
    freq_hz: float,
    sign: float = 1.0,
    colocate: bool = True,
) -> Dict[str, np.ndarray]:
    """Surface equivalent currents ``J = n_hat x H``, ``M = -n_hat x E`` on a
    single plane normal to ``axis``, at one frequency.

    ``sign`` is the orientation of the outward normal relative to ``+axis``
    (``+1`` for an outward normal along ``+axis``, ``-1`` along ``-axis``).
    ``colocate`` (default on) averages each Yee-staggered component to the
    in-plane cell node before forming ``J``/``M`` — correct for the engine's
    staggered DFT output; pass ``False`` for synthetic already-co-located
    fields (see the module docstring's "Yee staggering").
    Returns a dict with the three Cartesian components of ``J`` and ``M`` (keys
    ``Jx,Jy,Jz,Mx,My,Mz``, each a ``[i_t2, i_t1]`` complex array) plus the
    plane geometry ``t1``,``t2`` (axis names), ``c1``,``c2`` (their coords, µm),
    ``normal_um`` (the normal position, µm) and ``normal_axis`` (``axis``)."""
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    t1, t2 = _TRANSVERSE[axis]
    e1n, e2n, h1n, h2n = _TANGENTIAL[axis]
    for key in (e1n, e2n, h1n, h2n):
        if key not in plane_fields:
            raise ValueError(
                f"plane_fields missing {key!r}; for axis={axis!r} need the "
                f"tangential components {e1n!r},{e2n!r},{h1n!r},{h2n!r}")

    E1, c1, c2, n_um = _plane_component_3d(plane_fields[e1n], e1n, freq_hz, t1, t2, axis)
    E2, _, _, _ = _plane_component_3d(plane_fields[e2n], e2n, freq_hz, t1, t2, axis)
    H1, _, _, _ = _plane_component_3d(plane_fields[h1n], h1n, freq_hz, t1, t2, axis)
    H2, _, _, _ = _plane_component_3d(plane_fields[h2n], h2n, freq_hz, t1, t2, axis)

    if colocate:
        # Same offsets as mode_overlap._overlap_terms (grid.h yee_offset):
        # arrays are [i_t2, i_t1]; E_t1 is +1/2 in t1 (last axis), E_t2 +1/2 in
        # t2 (axis 0), H_t1 +1/2 in t2, H_t2 +1/2 in t1.
        E1 = _colocate_to_node(E1, -1)
        E2 = _colocate_to_node(E2, 0)
        H1 = _colocate_to_node(H1, 0)
        H2 = _colocate_to_node(H2, -1)

    # n_hat = sign * axis_hat. For a transverse pair (a1, a2) along (t1, t2):
    #   n_hat x (a1 t1_hat + a2 t2_hat) = sign * (a1 (axis x t1) + a2 (axis x t2))
    #   and since t1 x t2 = +axis (right-handed), axis x t1 = t2, axis x t2 = -t1.
    # => n_hat x A_t = sign * (-a2 t1_hat + a1 t2_hat).
    # J = n_hat x H ; M = -(n_hat x E).
    J = {t1: sign * (-H2), t2: sign * (H1)}
    M = {t1: -sign * (-E2), t2: -sign * (E1)}

    out: Dict[str, np.ndarray] = {}
    for comp in ("x", "y", "z"):
        out[f"J{comp}"] = J.get(comp, np.zeros_like(E1))
        out[f"M{comp}"] = M.get(comp, np.zeros_like(E1))
    out["t1"] = t1  # type: ignore[assignment]
    out["t2"] = t2  # type: ignore[assignment]
    out["c1"] = c1
    out["c2"] = c2
    out["normal_um"] = n_um  # type: ignore[assignment]
    out["normal_axis"] = axis  # type: ignore[assignment]
    return out


def _project_one_face(
    currents: Mapping[str, np.ndarray],
    *,
    k: float,
    rx: np.ndarray,
    ry: np.ndarray,
    rz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Radiation-vector contributions ``(N_x,N_y,N_z, L_x,L_y,L_z)`` of one face
    for a set of direction cosines ``(rx,ry,rz)`` (each a flat ``(n_dir,)``
    array). ``k`` is the free-space wavenumber (1/m).

    Computes ``N = integral J exp(-i k r_hat.r') dS`` (and ``L`` from ``M``) —
    the MINUS-sign kernel of the ``e^{-i omega t}`` convention, matching the
    code below and the module docstring — by forming the kernel on the face's
    grid (coords in µm -> m) and summing with the centered-difference
    cell-area quadrature."""
    t1 = currents["t1"]
    t2 = currents["t2"]
    normal_axis = currents["normal_axis"]
    c1 = np.asarray(currents["c1"], dtype=np.float64) * 1e-6  # µm -> m
    c2 = np.asarray(currents["c2"], dtype=np.float64) * 1e-6
    n_m = float(currents["normal_um"]) * 1e-6

    dA = np.outer(_cell_widths(c2), _cell_widths(c1))  # [i_t2, i_t1] m^2

    # Cartesian position of every surface point: t1, t2 in-plane, normal fixed.
    pos = {t1: c1[None, :] * np.ones((c2.size, 1)),       # [i_t2, i_t1]
           t2: c2[:, None] * np.ones((1, c1.size)),
           normal_axis: np.full((c2.size, c1.size), n_m)}
    X = pos["x"]
    Y = pos["y"]
    Z = pos["z"]

    # Direction cosines per direction; kernel exp(-i k r_hat . r').
    # In the e^{-i omega t} convention the outgoing wave is e^{+i k r}, so the
    # far-field expansion |r - r'| ~ r - r_hat . r' gives
    #   e^{i k |r-r'|} ~ e^{i k r} * e^{-i k r_hat . r'},
    # i.e. the surface-FT kernel carries a MINUS sign here (a +kx0 aperture tilt
    # then steers the beam toward +x, as a +x-travelling wave must). Build
    # via the flat spatial axis. The kernel matrix is [n_dir, n_pts]; for a large
    # monitor x many directions that can be huge, so chunk over directions to keep
    # the working set bounded (see _DIR_CHUNK).
    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    Zf = Z.reshape(-1)
    wdA = dA.reshape(-1)  # [n_pts]
    JM = [currents[c].reshape(-1) * wdA for c in
          ("Jx", "Jy", "Jz", "Mx", "My", "Mz")]

    n_dir = rx.size
    n_pts = Xf.size
    out = [np.empty(n_dir, dtype=np.complex128) for _ in range(6)]
    chunk = max(1, int(_DIR_CHUNK // max(n_pts, 1)))
    for lo in range(0, n_dir, chunk):
        hi = min(lo + chunk, n_dir)
        phase = k * (np.outer(rx[lo:hi], Xf)
                     + np.outer(ry[lo:hi], Yf)
                     + np.outer(rz[lo:hi], Zf))
        kernel = np.exp(-1j * phase)  # [chunk, n_pts], kernel = exp(-i k r_hat.r')
        for i, src in enumerate(JM):
            out[i][lo:hi] = kernel @ src
    return tuple(out)  # type: ignore[return-value]


def far_field(
    data,
    monitor_name: str,
    *,
    theta: Union[float, Sequence[float], np.ndarray],
    phi: Union[float, Sequence[float], np.ndarray] = 0.0,
    axis: Optional[Axis] = None,
    sign: float = 1.0,
    freqs_hz: Optional[Sequence[float]] = None,
    mesh: bool = True,
    faces: Optional[Sequence[Tuple[str, str, float]]] = None,
    colocate: bool = True,
) -> FarField:
    """Project a recorded near-field surface onto the far field.

    Single-plane (default) or closed-box projection of the tangential ``E``/``H``
    recorded by a ``FieldDftMonitor`` (``data[monitor_name]``), via the
    surface-equivalence currents ``J = n_hat x H``, ``M = -n_hat x E`` and the
    radiation integral (see the module docstring). Returns a :class:`FarField`
    with ``E_theta``/``E_phi`` (far-field amplitudes), from which intensity,
    radiated power and directivity follow.

    Parameters
    ----------
    data:
        A :class:`~photonhub.data.SimulationData` (or any mapping) such that
        ``data[monitor_name]`` is the recorded DFT field DataArray, dims
        ``('f','component','z','y','x')`` (a single-plane slice has a singleton
        normal axis). For a closed box pass per-face monitors via ``faces``.
    monitor_name:
        Key of the near-field monitor in ``data`` (single-plane mode).
    theta, phi:
        Far-field directions (radians). With ``mesh=True`` (default) these are
        1-D axes meshed into a ``(n_theta, n_phi)`` grid (``theta`` slowest);
        with ``mesh=False`` they are paired elementwise. ``theta`` is the polar
        angle from ``+z``, ``phi`` the azimuth from ``+x`` in the x-y plane.
    axis:
        Plane normal ``"x"``/``"y"``/``"z"`` (single-plane mode). If ``None``,
        inferred from the monitor's singleton spatial dimension.
    sign:
        Orientation of the outward normal: ``+1`` along ``+axis`` (default,
        i.e. radiating toward larger ``axis``), ``-1`` along ``-axis``.
    freqs_hz:
        Frequencies to project (Hz). Defaults to every frequency on the monitor.
    mesh:
        Whether ``theta``/``phi`` are meshed (default) or paired.
    faces:
        Optional closed-surface (multi-face) projection: a sequence of
        ``(monitor_name, axis, sign)`` triples, one per box face, summed with
        each face's own outward normal. When given, ``monitor_name``/``axis``/
        ``sign`` are ignored. Each named monitor must be in ``data``.
    colocate:
        Average each Yee-staggered tangential component to the in-plane cell
        node before forming the equivalence currents (default ``True`` —
        correct for the engine's staggered DFT output, mirroring the mode
        readout's convention). Pass ``False`` for synthetic already-co-located
        fields. See the module docstring's "Yee staggering" for what remains
        uncorrected (the normal-axis E/H half-cell stagger).

    Returns
    -------
    FarField
        ``e_theta``/``e_phi`` arrays of shape ``(n_freq, n_dir)`` plus the flat
        direction and frequency axes; see :class:`FarField` and its
        :meth:`~FarField.reshape`.
    """
    # Assemble the list of (DataArray, axis, sign) faces.
    if faces is not None:
        face_specs = [(data[name], ax, float(sg)) for name, ax, sg in faces]
    else:
        da = data[monitor_name]
        if axis is None:
            axis = _infer_normal_axis(da)
        face_specs = [(da, axis, float(sign))]
    for _, ax, _ in face_specs:
        if ax not in _TRANSVERSE:
            raise ValueError(f"axis must be one of x/y/z, got {ax!r}")

    # Frequencies: default to the monitor's own.
    ref_da = face_specs[0][0]
    if freqs_hz is None:
        if "f" in getattr(ref_da, "dims", ()):
            freqs = [float(f) for f in np.asarray(ref_da.coords["f"].values)]
        else:
            freqs = [0.0]
    else:
        freqs = [float(f) for f in np.atleast_1d(freqs_hz)]

    th, ph, _ = _directions(theta, phi, mesh)
    rx = np.sin(th) * np.cos(ph)
    ry = np.sin(th) * np.sin(ph)
    rz = np.cos(th)

    n_freq = len(freqs)
    n_dir = th.size
    e_theta = np.zeros((n_freq, n_dir), dtype=np.complex128)
    e_phi = np.zeros((n_freq, n_dir), dtype=np.complex128)

    cos_t, sin_t = np.cos(th), np.sin(th)
    cos_p, sin_p = np.cos(ph), np.sin(ph)

    for fi, f in enumerate(freqs):
        k = 2.0 * np.pi * f / C0  # free-space wavenumber (1/m)
        Ntot = [np.zeros(n_dir, dtype=np.complex128) for _ in range(6)]
        for da, ax, sg in face_specs:
            currents = equivalent_currents(
                _as_component_mapping(da, ax), axis=ax, freq_hz=f, sign=sg,
                colocate=colocate)
            contribs = _project_one_face(currents, k=k, rx=rx, ry=ry, rz=rz)
            for i in range(6):
                Ntot[i] = Ntot[i] + contribs[i]
        Nx, Ny, Nz, Lx, Ly, Lz = Ntot

        # Spherical projection of the radiation vectors.
        N_theta = Nx * cos_t * cos_p + Ny * cos_t * sin_p - Nz * sin_t
        N_phi = -Nx * sin_p + Ny * cos_p
        L_theta = Lx * cos_t * cos_p + Ly * cos_t * sin_p - Lz * sin_t
        L_phi = -Lx * sin_p + Ly * cos_p

        # Far-zone E (spreading factor e^{ikr}/r dropped -> far-field amplitude).
        # Prefactor i k / (4 pi); e^{-i omega t} convention. Balanis derives these
        # with e^{+j omega t} / e^{-j k r}; converting to e^{-i omega t} is j -> -i,
        # which flips BOTH the exponential (e^{-jkr} -> e^{+ikr}) AND the leading
        # jk -> -ik. Keeping Balanis' outer signs alongside e^{+ikr} left a global
        # -1 (a constant pi absolute-phase error). On-axis check for a Huygens
        # patch (Ex=g, Hy=g/eta0): these give F_theta = -(ik/2pi) Int g dS, matching
        # the Rayleigh-Sommerfeld reference F = -ik/(2pi) Int E dS in this same
        # convention.
        pref = 1j * k / (4.0 * np.pi)
        e_theta[fi] = pref * (L_phi + ETA0 * N_theta)
        e_phi[fi] = -pref * (L_theta - ETA0 * N_phi)

    return FarField(
        theta=th, phi=ph, freqs_hz=np.asarray(freqs, dtype=np.float64),
        e_theta=e_theta, e_phi=e_phi,
    )


def _infer_normal_axis(da: xr.DataArray) -> str:
    """Infer the plane normal from a DFT field DataArray: the spatial axis whose
    coordinate has length 1 (the singleton, swept-out normal)."""
    singles = [ax for ax in ("x", "y", "z")
               if ax in da.dims and da.sizes.get(ax, 0) == 1]
    if len(singles) == 1:
        return singles[0]
    raise ValueError(
        "could not infer the plane normal axis from the monitor (expected "
        f"exactly one singleton spatial dim; dims={tuple(da.dims)}, "
        f"sizes={{ {', '.join(f'{a}:{da.sizes.get(a)}' for a in ('x','y','z') if a in da.dims)} }}). "
        "Pass axis= explicitly.")


def _as_component_mapping(da: xr.DataArray, axis: str) -> Dict[str, xr.DataArray]:
    """Slice the four tangential components for ``axis`` out of a DFT field
    DataArray with a ``component`` dim, into the ``{name: DataArray}`` mapping
    :func:`equivalent_currents` expects. If ``da`` has no ``component`` dim it is
    assumed already pre-sliced and returned per-name unchanged."""
    names = _TANGENTIAL[axis]
    if "component" in getattr(da, "dims", ()):
        return {n: da.sel(component=n) for n in names}
    return {n: da for n in names}
