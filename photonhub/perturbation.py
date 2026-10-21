"""Index perturbations from an imported temperature or carrier-density map.

The map is an input: a heat or device (TCAD) tool, or the user's own formula,
computes it on a rectilinear grid in the user's frame. :func:`perturbed_medium`
turns it into a custom :class:`~photonhub.Medium` whose permittivity (and,
for free carriers, conductivity) is given at the map's nodes; the engine
places the nodes by their coordinates and interpolates trilinearly at each
field sample point (wire schema 1.22). PhotonHub solves neither the heat nor
the carrier transport.

Physics, applied at each node:

- thermo-optic, first order in temperature:
  ``n = n0 + dn_dT * (T - T_ref)``;
- carrier plasma dispersion in silicon at 1550 nm, from the excess electron
  and hole densities in cm^-3, with the published fits in
  :data:`CARRIER_MODELS`: ``n += dn(Ne, Nh)`` and an intensity absorption
  ``alpha(Ne, Nh)`` in 1/cm, carried as the conductivity
  ``sigma = eps0 * c * n * alpha`` (``alpha`` in 1/m), which reproduces
  ``alpha`` exactly at every frequency in the small-loss limit
  (``Im eps = sigma / (omega eps0) = 2 n kappa``, ``alpha = 2 k0 kappa``).

The node permittivity is ``n**2``; the ``-kappa**2`` term of the real part is
dropped. It is 3e-8 at 1e18 cm^-3 of each carrier, 2.5e-4 at 5e19 and 1.2e-3
at 1e20, where it shifts the index by under 0.2 % of the carrier index
change. A fixed conductivity gives the same absorption at every wavelength
(``alpha = sigma / (eps0 * c * n)``), while the fits' absorption grows
roughly as lambda**2, so the two agree only near the fit's wavelength. So
carrier maps need ``wavelength_um`` and refuse one outside 1.50 to 1.60 um.
Only that argument is checked, not the band the simulation runs (see
:func:`perturbed_medium`).

Limits: CPU solver only (GPU solvers refuse custom media); subpixel smoothing
is off for custom media, so interfaces are staircased, and a phase shift
taken as the difference of two runs on one mesh removes most of that error.
A port inside a perturbed region solves its mode on the perturbed
cross-section, real part only: the port mode is lossless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .components.structures import Medium, PermittivityArray
from .constants import c0 as _C0, eps0 as _EPS0

__all__ = [
    "CARRIER_MODELS",
    "CarrierModel",
    "SpatialMap",
    "carrier_plasma",
    "perturbed_medium",
    "thermo_optic_index",
]

_CARRIER_BAND_UM = (1.50, 1.60)
# an axis given one coordinate is invariant; the engine needs two nodes, and
# with equal values their placement is immaterial (constant extrapolation)
_INVARIANT_HALF_SPAN_UM = 0.5


@dataclass(frozen=True)
class SpatialMap:
    """A scalar map on a rectilinear grid in the user's frame (microns).

    ``values`` has shape ``(len(x_um), len(y_um), len(z_um))``. Each
    coordinate axis is strictly increasing; an axis given ONE coordinate
    means the map does not vary along it (a 2-D cross-section of a straight
    heater, say). Beyond the outermost node on an axis the map keeps that
    node's value."""

    x_um: Tuple[float, ...]
    y_um: Tuple[float, ...]
    z_um: Tuple[float, ...]
    values: np.ndarray

    def __post_init__(self):
        axes = []
        for name in ("x_um", "y_um", "z_um"):
            c = tuple(float(v) for v in np.atleast_1d(np.asarray(getattr(self, name), dtype=float)))
            if not c:
                raise ValueError(f"SpatialMap.{name} is empty")
            if not all(math.isfinite(v) for v in c):
                raise ValueError(f"SpatialMap.{name} must be finite")
            if any(not (b > a) for a, b in zip(c, c[1:])):
                raise ValueError(f"SpatialMap.{name} must be strictly increasing")
            object.__setattr__(self, name, c)
            axes.append(len(c))
        vals = np.asarray(self.values, dtype=float)
        if vals.shape != tuple(axes):
            raise ValueError(
                f"SpatialMap.values has shape {vals.shape}; the coordinates say {tuple(axes)}")
        if not np.all(np.isfinite(vals)):
            raise ValueError("SpatialMap.values must be finite")
        vals = vals.copy()
        vals.setflags(write=False)
        object.__setattr__(self, "values", vals)

    @property
    def coords_um(self) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        """The node coordinates ``(x_um, y_um, z_um)``."""
        return (self.x_um, self.y_um, self.z_um)

    def same_grid(self, other: "SpatialMap") -> bool:
        """True when ``other`` has exactly the same node coordinates."""
        return self.coords_um == other.coords_um

    @classmethod
    def from_function(cls, fn: Callable, x_um: Sequence[float], y_um: Sequence[float],
                      z_um: Sequence[float]) -> "SpatialMap":
        """A map sampled from ``fn(x, y, z)`` (vectorized, microns) on the
        grid ``x_um`` x ``y_um`` x ``z_um``."""
        X, Y, Z = np.meshgrid(np.atleast_1d(np.asarray(x_um, dtype=float)),
                              np.atleast_1d(np.asarray(y_um, dtype=float)),
                              np.atleast_1d(np.asarray(z_um, dtype=float)), indexing="ij")
        return cls(x_um=tuple(X[:, 0, 0]), y_um=tuple(Y[0, :, 0]), z_um=tuple(Z[0, 0, :]),
                   values=np.broadcast_to(np.asarray(fn(X, Y, Z), dtype=float), X.shape))


@dataclass(frozen=True)
class CarrierModel:
    """A published carrier-plasma fit for silicon: ``dn = -(a_e Ne^p_e +
    a_h Nh^p_h)`` and ``dalpha = b_e Ne^q_e + b_h Nh^q_h`` (1/cm), densities
    in cm^-3, valid at ``wavelength_um``."""

    name: str
    wavelength_um: float
    a_e: float
    p_e: float
    a_h: float
    p_h: float
    b_e: float
    q_e: float
    b_h: float
    q_h: float
    reference: str


#: The silicon carrier models :func:`carrier_plasma` knows, keyed by name.
CARRIER_MODELS: Dict[str, CarrierModel] = {
    m.name: m for m in (
        CarrierModel(
            name="nedeljkovic2011_1550", wavelength_um=1.55,
            a_e=5.4e-22, p_e=1.011, a_h=1.53e-18, p_h=0.838,
            b_e=8.88e-21, q_e=1.167, b_h=5.84e-20, q_h=1.109,
            reference=("M. Nedeljkovic, R. Soref, G. Z. Mashanovich, Free-carrier electrorefraction "
                       "and electroabsorption modulation predictions for silicon over the 1-14 um "
                       "infrared wavelength range, IEEE Photonics Journal 3(6), 1171-1180 (2011), "
                       "doi:10.1109/JPHOT.2011.2171930: the model of eqs. (4) and (5), p. 1175 "
                       "(eq. (5) written -dn = p dNe^q + r dNh^s), with the coefficients of Table 1, "
                       "row 1.55 um, p. 1178; fitted over 3.2e17 to 4e19 cm^-3 electrons and 5e17 to "
                       "1e20 cm^-3 holes. Also quoted in arXiv:1504.03064, eqs. (3a) and (3b)")),
        CarrierModel(
            name="soref_bennett1987_1550", wavelength_um=1.55,
            a_e=8.8e-22, p_e=1.0, a_h=8.5e-18, p_h=0.8,
            b_e=8.5e-18, q_e=1.0, b_h=6.0e-18, q_h=1.0,
            reference=("R. A. Soref, B. R. Bennett, Electrooptical effects in silicon, IEEE J. "
                       "Quantum Electron. 23(1), 123-129 (1987), doi:10.1109/JQE.1987.1073206. The "
                       "coefficients are the commonly quoted 1.55 um expressions for this "
                       "reference, not yet checked against the paper itself; computed from them, "
                       "1e18 holes per cm^3 give dn = -2.135e-3")),
    )
}


def thermo_optic_index(n0, dn_dT: float, temperature_k, reference_temperature_k: float = 300.0):
    """``n0 + dn_dT * (T - T_ref)``, first order in temperature (kelvin)."""
    return np.asarray(n0, dtype=float) + float(dn_dT) * (
        np.asarray(temperature_k, dtype=float) - float(reference_temperature_k))


def carrier_plasma(electrons_cm3, holes_cm3, model: str = "nedeljkovic2011_1550"):
    """``(dn, dalpha_per_cm)`` of silicon for excess electron and hole
    densities in cm^-3 (each >= 0), from the named fit in
    :data:`CARRIER_MODELS`."""
    if model not in CARRIER_MODELS:
        raise ValueError(f"unknown carrier model {model!r}; known: {sorted(CARRIER_MODELS)}")
    m = CARRIER_MODELS[model]
    ne = np.asarray(electrons_cm3, dtype=float)
    nh = np.asarray(holes_cm3, dtype=float)
    if np.any(~np.isfinite(ne)) or np.any(~np.isfinite(nh)) or np.any(ne < 0) or np.any(nh < 0):
        raise ValueError("carrier densities must be finite and >= 0 (cm^-3, in excess of the base medium)")
    dn = -(m.a_e * ne ** m.p_e + m.a_h * nh ** m.p_h)
    dalpha = m.b_e * ne ** m.q_e + m.b_h * nh ** m.q_h
    return dn, dalpha


def _base_medium(base, wavelength_um: float) -> Medium:
    if isinstance(base, Medium):
        medium = base
    elif callable(getattr(base, "medium", None)):          # a materials-library entry
        medium = base.medium(wlen_um=wavelength_um)
    else:
        raise TypeError("base must be a ph.Medium or a ph.materials entry")
    if medium.pec or medium.is_dispersive or medium.permittivity_xyz is not None \
            or medium.permittivity_data is not None:
        raise ValueError("perturbed_medium needs a non-dispersive, isotropic, non-PEC base medium "
                         "without its own permittivity data")
    return medium


def _engine_grid(maps):
    """Coordinates and node arrays for the engine: an invariant (one-node)
    axis becomes two nodes carrying the same value."""
    coords = list(maps[0].coords_um)
    reps = [1, 1, 1]
    for a in range(3):
        if len(coords[a]) == 1:
            c = coords[a][0]
            coords[a] = (c - _INVARIANT_HALF_SPAN_UM, c + _INVARIANT_HALF_SPAN_UM)
            reps[a] = 2
    return tuple(coords), lambda v: np.tile(np.asarray(v, dtype=float), reps)


def perturbed_medium(base, *, temperature: Optional[SpatialMap] = None, dn_dT: Optional[float] = None,
                     reference_temperature_k: float = 300.0,
                     electrons_cm3: Optional[SpatialMap] = None, holes_cm3: Optional[SpatialMap] = None,
                     carrier_model: str = "nedeljkovic2011_1550",
                     wavelength_um: Optional[float] = None) -> Medium:
    """A custom medium for ``base`` perturbed by an imported temperature map
    (kelvin, with ``dn_dT`` per kelvin) and/or excess carrier-density maps
    (cm^-3, silicon fits in :data:`CARRIER_MODELS`), on the maps' grid.

    The maps share one grid, drawn in the user's frame; the simulation moves
    the nodes with its structures. The returned medium's scalar
    ``permittivity`` is the base value, its node values carry the
    perturbation (the port mode solver paints them too, real part only, so a
    port inside the perturbed region solves the perturbed, lossless mode),
    and a base conductivity plus the
    free-carrier absorption become a per-node conductivity grid. Put it on
    any geometry: the geometry decides where the medium is, the map what it
    is there, and beyond the map's outermost node the edge value holds.

    ``wavelength_um`` is required with carrier maps and must lie in 1.50 to
    1.60 um: the carrier models are 1550 nm fits. Give the wavelength the
    simulation runs at. Only this argument is checked, not the run's own band,
    so a medium built for 1.55 um and run at 1.31 um silently uses the 1550 nm
    values. Within the allowed range the 1550 nm values can be off by up to
    about 8 % in the index change and up to about 14 % in the absorption
    change at 1.50 and 1.60 um (interpolated between the 1.3, 1.55 and 2.0 um
    rows of Table 1 in Nedeljkovic 2011, which has no rows at 1.50 or
    1.60 um). The default fit (``nedeljkovic2011_1550``) was made over 3.2e17
    to 4e19 cm^-3 electrons and 5e17 to 1e20 cm^-3 holes; densities outside
    that span extrapolate it. Without carriers, ``wavelength_um`` (1.55 when
    omitted) only resolves a materials-library ``base``."""
    maps = [m for m in (temperature, electrons_cm3, holes_cm3) if m is not None]
    if not maps:
        raise ValueError("give a temperature map, a carrier-density map, or both")
    if any(not isinstance(m, SpatialMap) for m in maps):
        raise TypeError("maps must be ph.SpatialMap")
    if any(not maps[0].same_grid(m) for m in maps[1:]):
        raise ValueError("the temperature and carrier maps must share one grid; resample them first")
    if (temperature is None) != (dn_dT is None):
        raise ValueError("a temperature map needs dn_dT (per kelvin), and dn_dT needs a temperature map")
    carriers = electrons_cm3 is not None or holes_cm3 is not None
    if carriers and wavelength_um is None:
        raise ValueError(
            "carrier maps need wavelength_um, the wavelength the simulation runs at: the carrier "
            f"models are 1550 nm fits, allowed from {_CARRIER_BAND_UM[0]} to {_CARRIER_BAND_UM[1]} um")
    if wavelength_um is None:
        wavelength_um = 1.55
    if carriers and not (_CARRIER_BAND_UM[0] <= float(wavelength_um) <= _CARRIER_BAND_UM[1]):
        raise ValueError(
            f"the carrier models are 1550 nm fits; wavelength_um={wavelength_um} lies outside "
            f"{_CARRIER_BAND_UM[0]} to {_CARRIER_BAND_UM[1]} um, where the conductivity form of "
            "their absorption drifts")
    medium = _base_medium(base, float(wavelength_um))
    shape = maps[0].values.shape
    n0 = math.sqrt(float(medium.permittivity))
    n = np.full(shape, n0)
    if temperature is not None:
        n = thermo_optic_index(n, float(dn_dT), temperature.values, reference_temperature_k)
    sigma = None
    if carriers:
        zero = np.zeros(shape)
        dn, dalpha = carrier_plasma(zero if electrons_cm3 is None else electrons_cm3.values,
                                    zero if holes_cm3 is None else holes_cm3.values, carrier_model)
        n = n + dn
        sigma = _EPS0 * _C0 * n * (dalpha * 100.0) + float(medium.conductivity_s_per_m)
    coords, expand = _engine_grid(maps)
    eps = expand(n * n)
    data = PermittivityArray(
        shape=eps.shape, values=tuple(float(v) for v in eps.reshape(-1)), coords_um=coords,
        conductivity_s_per_m=None if sigma is None else tuple(float(v) for v in expand(sigma).reshape(-1)))
    return Medium(permittivity=float(medium.permittivity),
                  conductivity_s_per_m=0.0 if sigma is not None else float(medium.conductivity_s_per_m),
                  permittivity_data=data)


