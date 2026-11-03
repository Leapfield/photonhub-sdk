"""The labelled form of a :class:`~photonhub.analysis.FarField` and the
arithmetic of its result methods (gain, cross section, polarization bases,
power density, Cartesian fields), kept apart from the projection itself.

Nothing here computes a projected value: the labelled arrays are read-only
views of the arrays :func:`~photonhub.analysis.far_field` returned, reshaped
onto the grid's axes, so every number in ``FarField.dataset`` is bit for bit
the one in the flat layout.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from ._constants import C0, ETA0

#: The in-plane axes of a plane normal to each axis, in xyz order (the order
#: of ``ux``, ``uy`` and ``x_um``, ``y_um``).
NATURAL: Dict[str, Tuple[str, str]] = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}

_FIELD_UNITS = {"e_theta": "V", "e_phi": "V", "e_r": "V",
                "h_theta": "A", "h_phi": "A", "h_r": "A"}
_FIELD_NAMES = {"e_theta": "far-field amplitude r exp(-ikr) E_theta",
                "e_phi": "far-field amplitude r exp(-ikr) E_phi",
                "e_r": "r exp(-ikr) E_r", "h_r": "r exp(-ikr) H_r",
                "h_theta": "r exp(-ikr) H_theta", "h_phi": "r exp(-ikr) H_phi"}


def _frozen(a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """``a`` reshaped to ``shape`` as a read-only view (no copy, no arithmetic)."""
    v = np.asarray(a).reshape(shape)
    if v.base is None:
        v = v.view()
    v.flags.writeable = False
    return v


def grid_dims(ff) -> Tuple[str, ...]:
    """The dims of one frequency's values: ``("theta", "phi")``, ``("ux",
    "uy")``, the two in-plane axes of a Cartesian grid, or ``("point",)``
    for paired points (``mesh=False``)."""
    if ff.shape is None:
        return ("point",)
    if ff.grid == "angle":
        return ("theta", "phi")
    if ff.grid == "kspace":
        return ("ux", "uy")
    return NATURAL[ff.plane_axis]


def _one_distance(r: np.ndarray) -> Optional[float]:
    """The single distance every point with one shares, else ``None``."""
    live = r[np.isfinite(r)]
    if live.size and np.all(live == live[0]):
        return float(live[0])
    return None


def labelled(ff) -> xr.Dataset:
    """The :class:`xarray.Dataset` behind ``FarField.dataset`` (see there)."""
    f = np.asarray(ff.freqs_hz, dtype=np.float64)
    nf = f.size
    dims = grid_dims(ff)
    n_dir = int(np.size(ff.theta))
    gshape = (n_dir,) if ff.shape is None else tuple(int(n) for n in ff.shape)
    if int(np.prod(gshape)) != n_dir:
        raise ValueError(f"FarField.shape {ff.shape} does not hold its {n_dir} points")
    full = (nf,) + gshape

    n = np.broadcast_to(np.asarray(ff.n_medium, dtype=np.float64), (nf,))
    with np.errstate(divide="ignore"):
        wlen = C0 / f * 1e6
    coords: Dict[str, object] = {
        "f": ("f", f, {"units": "Hz", "long_name": "frequency"}),
        "wlen_um": ("f", wlen, {"units": "um", "long_name": "vacuum wavelength"}),
        "n_medium": ("f", np.array(n), {"long_name": "index of the medium projected into"}),
    }
    angle_attrs = {"units": "rad"}
    theta = np.asarray(ff.theta, dtype=np.float64)
    phi = np.asarray(ff.phi, dtype=np.float64)
    if ff.shape is None:
        coords["theta"] = ("point", _frozen(theta, gshape), dict(angle_attrs, long_name="polar angle from +z"))
        coords["phi"] = ("point", _frozen(phi, gshape), dict(angle_attrs, long_name="azimuth from +x"))
        if ff.grid == "kspace":
            p, q = NATURAL[ff.plane_axis]
            coords["ux"] = ("point", _frozen(ff.ux, gshape),
                            {"long_name": f"direction cosine along {p}", "axis": p})
            coords["uy"] = ("point", _frozen(ff.uy, gshape),
                            {"long_name": f"direction cosine along {q}", "axis": q})
        elif ff.grid == "cartesian":
            p, q = NATURAL[ff.plane_axis]
            coords[p] = ("point", _frozen(ff.x_um, gshape), {"units": "um", "axis": p})
            coords[q] = ("point", _frozen(ff.y_um, gshape), {"units": "um", "axis": q})
    elif ff.grid == "angle":
        coords["theta"] = ("theta", theta.reshape(gshape)[:, 0].copy(),
                           dict(angle_attrs, long_name="polar angle from +z"))
        coords["phi"] = ("phi", phi.reshape(gshape)[0, :].copy(),
                         dict(angle_attrs, long_name="azimuth from +x"))
    else:
        one, two = (ff.ux, ff.uy) if ff.grid == "kspace" else (ff.x_um, ff.y_um)
        p, q = NATURAL[ff.plane_axis]
        for dim, values, axis in ((dims[0], np.asarray(one).reshape(gshape)[:, 0], p),
                                  (dims[1], np.asarray(two).reshape(gshape)[0, :], q)):
            attrs = ({"long_name": f"direction cosine along {axis}", "axis": axis}
                     if ff.grid == "kspace" else {"units": "um", "axis": axis})
            coords[dim] = (dim, values.copy(), attrs)
        coords["theta"] = (dims, _frozen(theta, gshape), dict(angle_attrs, long_name="polar angle from +z"))
        coords["phi"] = (dims, _frozen(phi, gshape), dict(angle_attrs, long_name="azimuth from +x"))
    if ff.grid == "kspace":
        coords["propagating"] = (dims, _frozen(ff.propagating, gshape),
                                 {"long_name": "the point has a direction (ux^2 + uy^2 <= 1)"})
    if ff.grid == "cartesian" and ff.plane_um is not None:
        coords[ff.plane_axis] = ((), float(ff.plane_um), {"units": "um"})
    if ff.r_um is not None:
        r = np.asarray(ff.r_um, dtype=np.float64)
        one = None if ff.grid == "cartesian" else _one_distance(r)
        if one is not None:
            coords["r_um"] = ((), one, {"units": "um", "long_name": "distance from origin_um"})
        else:
            coords["r_um"] = (dims, _frozen(r, gshape),
                              {"units": "um", "long_name": "distance from origin_um"})

    data_vars: Dict[str, object] = {}
    for name in ("e_theta", "e_phi", "e_r", "h_r", "h_theta", "h_phi"):
        value = getattr(ff, name)
        if value is not None:
            data_vars[name] = (("f",) + dims, _frozen(value, full),
                               {"units": _FIELD_UNITS[name], "long_name": _FIELD_NAMES[name]})
    data_vars["intensity"] = (("f",) + dims, _frozen(ff.intensity, full),
                              {"units": "W/sr", "long_name": "radiation intensity"})
    if ff.grid == "angle" and ff.shape is not None:
        data_vars["radiated_power"] = ("f", _frozen(ff.radiated_power(), (nf,)),
                                       {"units": "W", "long_name": "power over the angle grid"})
        data_vars["directivity"] = (("f",) + dims, _frozen(ff.directivity(), full),
                                    {"units": "1", "long_name": "directivity 4 pi U / P"})

    attrs: Dict[str, object] = {
        "grid": ff.grid,
        "projection": "exact" if ff.exact else "far-field approximation",
        "time_convention": "exp(-i omega t)",
    }
    for key in ("plane_axis", "direction"):
        if getattr(ff, key) is not None:
            attrs[key] = getattr(ff, key)
    if ff.origin_um is not None:
        attrs["origin_um"] = [float(v) for v in ff.origin_um]
    if ff.incident_e_v_per_m is not None:
        attrs["incident_e_v_per_m"] = float(ff.incident_e_v_per_m)
    if ff.phase_origin_um is not None:
        attrs["phase_origin_um"] = [float(v) for v in ff.phase_origin_um]
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def per_frequency(value, freqs_hz: np.ndarray, what: str, unit: str) -> np.ndarray:
    """A positive quantity per projected frequency, ``(n_freq,)``: a number,
    one value per frequency, or an :class:`xarray.DataArray` over ``f`` (a
    ``PowerMonitor``'s spectrum, for instance) read at each projected
    frequency."""
    f = np.asarray(freqs_hz, dtype=np.float64)
    if isinstance(value, xr.DataArray):
        da = value
        if "f" in da.dims:
            other = [d for d in da.dims if d != "f"]
            if other:
                raise ValueError(
                    f"{what} must be one value per frequency, but the DataArray has the "
                    f"dims {da.dims}; select or sum over {other} first")
            have = np.asarray(da.coords["f"].values, dtype=np.float64) if "f" in da.coords \
                else None
            if have is None:
                raise ValueError(f"{what}: the DataArray's f dim has no coordinate")
            out = np.empty(f.size)
            values = np.asarray(da.values)
            for i, fr in enumerate(f):
                hit = np.flatnonzero(np.isclose(have, fr, rtol=1e-9, atol=0.0))
                if hit.size == 0:
                    raise ValueError(
                        f"{what} has no value at {fr:.10g} Hz, a frequency of the projection")
                out[i] = _real(values[hit[0]], what)
            arr = out
        elif da.ndim == 0:
            arr = np.full(f.size, _real(da.values, what))
        else:
            raise ValueError(f"{what}: a DataArray needs an f dim (one value per frequency)")
    else:
        a = np.asarray(value)
        if np.iscomplexobj(a):
            raise ValueError(f"{what} must be real ({unit}), got a complex value")
        a = a.astype(np.float64)
        if a.ndim == 0:
            arr = np.full(f.size, float(a))
        elif a.shape == (f.size,):
            arr = a.copy()
        else:
            raise ValueError(
                f"{what} must be one number or one per projected frequency ({f.size}), "
                f"got shape {a.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{what} must be positive and finite ({unit}), got {arr}")
    return arr


def _real(v, what: str) -> float:
    v = np.asarray(v)
    if np.iscomplexobj(v):
        if np.imag(v) != 0.0:
            raise ValueError(f"{what} must be real, got {v!r}")
        v = np.real(v)
    return float(v)


def impedance(ff) -> xr.DataArray:
    """``eta = eta0 / n_medium`` over ``f``."""
    n = np.broadcast_to(np.asarray(ff.n_medium, dtype=np.float64), (np.size(ff.freqs_hz),))
    return xr.DataArray(ETA0 / n, dims="f")


def parse_boresight(value: str) -> Tuple[str, float]:
    if not (isinstance(value, str) and len(value) == 2 and value[0] in "+-"
            and value[1] in "xyz"):
        raise ValueError(f"boresight must be '+x', '-x', '+y', '-y', '+z' or '-z', got {value!r}")
    return value[1], 1.0 if value[0] == "+" else -1.0


def unit(axis: str, sign: float = 1.0) -> np.ndarray:
    v = np.zeros(3)
    v["xyz".index(axis)] = sign
    return v


def spherical_basis(theta: np.ndarray, phi: np.ndarray):
    """``r_hat``, ``theta_hat``, ``phi_hat`` as ``(..., 3)`` arrays."""
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    r = np.stack([st * cp, st * sp, ct], -1)
    t = np.stack([ct * cp, ct * sp, -st], -1)
    p = np.stack([-sp, cp, np.zeros_like(st)], -1)
    return r, t, p


def ludwig3_vectors(r_hat: np.ndarray, boresight: np.ndarray, reference: np.ndarray):
    """The co- and cross-polar unit vectors of Ludwig's third definition
    about ``boresight`` with ``reference`` the co-polar direction there:

        i_co    = x' - (r . x') (r + b) / (1 + r . b)
        i_cross = y' - (r . y') (r + b) / (1 + r . b),   y' = b x x'

    the ``cos(phi') theta_hat' - sin(phi') phi_hat'`` and ``sin(phi')
    theta_hat' + cos(phi') phi_hat'`` of a spherical system whose pole is
    ``b`` and whose azimuth runs from ``x'``. The definition has no limit at
    ``r = -b``; numerical stability also masks its antipodal neighborhood
    ``1 + r . b <= 1e-12`` with ``nan``."""
    y_ref = np.cross(boresight, reference)
    rb = r_hat @ boresight
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = 1.0 / (1.0 + rb)
    scale = np.where(1.0 + rb > 1e-12, scale, np.nan)
    s = (r_hat + boresight)
    co = reference - ((r_hat @ reference) * scale)[..., None] * s
    cross = y_ref - ((r_hat @ y_ref) * scale)[..., None] * s
    return co, cross


def incident_amplitude(data) -> Optional[float]:
    """The amplitude (V/m) of the plane wave a run driven by one TF/SF box
    launches: the box's declared ``amplitude``, which a run whose simulation
    is known multiplies back into its fields. ``None`` for anything else (a
    plain mapping of arrays, a result whose simulation is unknown, several
    sources or another kind)."""
    known = getattr(data, "_known_simulation", None)
    if not callable(known):
        return None
    try:
        sim = known()
    except (AttributeError, OSError, ValueError):
        return None
    sources = tuple(getattr(sim, "sources", None) or ())
    if len(sources) != 1 or getattr(sources[0], "type", None) != "tfsf_box":
        return None
    return abs(float(sources[0].amplitude))
