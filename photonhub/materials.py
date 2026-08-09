"""Optical material library — literature dispersion models that emit
:class:`~photonhub.Medium` objects (NUMERICS.md §19; the "material library"
deferred there).

Each :class:`Material` bundles a refractive-index model taken verbatim from
the published literature (Sellmeier / polynomial coefficients or tabulated
measurement data, with the source citation and its stated validity range) and
converts it to what the engine can run:

- ``mat.medium(wavelength_um=1.55)`` — a NON-dispersive ``Medium`` frozen at
  one wavelength: ``permittivity = n^2 - k^2`` and any absorption mapped to
  the Ohmic conductivity (exact at that frequency).
- ``mat.medium(band_um=(1.5, 1.6))`` — a dispersive ``Medium`` carrying the
  engine's SINGLE scalar Lorentz pole (§19), least-squares fitted to the
  literature curve over the band. The fit follows the Courant-safe recipe
  proven for c-Si in ``benchmarks/gds`` (PR #55/#57): the pole is placed near
  the band — NOT at the material's physical UV resonance — so ``eps_inf``
  stays well above 1 (a near-UV pole drives ``eps_inf -> 1``, which sits on
  the Courant edge and diverges). Use :meth:`Material.lorentz_fit` for the
  fit diagnostics (band error, pole placement, ADE-stability margin).

>>> import photonhub as ph
>>> from photonhub.materials import cSi, SiO2
>>> core = ph.Structure(geometry=box, medium=cSi.medium(band_um=(1.5, 1.6)))
>>> n_clad = SiO2.n(1.55)                      # 1.4440

Bring your own measured (ellipsometer) data — the right path for
deposition-dependent films (PECVD SiN, a-Si:H, ...):

>>> mySiN = Material.from_nk_data("my_SiN", wavelength_um=wl, n=n, k=k,
...                               reference="in-house ellipsometry 2026-05")

Built-in materials (see ``MATERIALS``): the isotropic scalar engine cannot
carry birefringence, so uniaxial crystals are split into ``_o`` / ``_e``
entries (use the ray your polarization sees). Metals are NOT included — they
need a Drude pole, which the single-Lorentz engine does not have yet
(NUMERICS.md §19 deferred list) and the wire bound ``permittivity >= 1``
excludes a negative-eps* constant-index stand-in.

All data files were taken from the refractiveindex.info database (public
domain, CC0 1.0), which transcribes the cited papers; coefficients are
reproduced verbatim and the original papers are cited on each entry.

Accuracy note (dispersive media): a dispersive structure makes ``Simulation``
default subpixel smoothing OFF, and correctly so — subpixel smoothing x the
Lorentz-ADE update is a LIVE late-time instability at fine grids
(engine/docs/subpixel-dispersion-instability.md; reconfirmed 2026-07-02 on a
current build: a 25 c/lambda dispersive-Si scene diverges with Volume
smoothing, and reducing the Courant number does not reliably cure it). So the
way to sharpen a dispersive interface is grid alignment / a finer uniform grid
(which resolves the geometry without smoothing), NOT ``subpixel=True``. Turning
subpixel on for a dispersive scene is only safe for short, non-resonant runs
(the instability needs a long ringdown to develop) and is not recommended.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .components.structures import LorentzPole, Medium

__all__ = [
    "Material",
    "LorentzFit",
    "Sellmeier",
    "Polynomial",
    "TabulatedNK",
    "ConstantIndex",
    "MATERIALS",
    "get",
    # built-in materials (registry attributes)
    "Vacuum",
    "cSi",
    "SiO2",
    "Si3N4",
    "GaAs",
    "InP",
    "Ge",
    "Sapphire",
    "AlN",
    "TiO2",
    "LiNbO3_o",
    "LiNbO3_e",
    "MgF2_o",
    "MgF2_e",
    "CaF2",
    "PMMA",
]

_C0_M_PER_S = 299792458.0
_EPS0_F_PER_M = 8.8541878128e-12


def _freq_hz(wavelength_um: np.ndarray) -> np.ndarray:
    """Vacuum frequency (Hz) of a vacuum wavelength (µm)."""
    return _C0_M_PER_S / (np.asarray(wavelength_um, dtype=float) * 1e-6)


# ---------------------------------------------------------------------------
# index models (real-index dispersion relations; k separate)
# ---------------------------------------------------------------------------


class _IndexModel:
    """A refractive-index dispersion relation n(lambda), k(lambda)."""

    def n2(self, wavelength_um: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def k(self, wavelength_um: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(wavelength_um, dtype=float))


@dataclass(frozen=True)
class Sellmeier(_IndexModel):
    """``n^2 = a0 + sum B_i * lam^2 / (lam^2 - C_i)`` with ``lam`` in µm and
    ``C_i`` in µm^2 (refractiveindex.info "formula 1" terms enter with
    ``C_i = coeff^2``, "formula 2" terms verbatim)."""

    a0: float
    terms: Tuple[Tuple[float, float], ...]

    def n2(self, wavelength_um):
        lam2 = np.asarray(wavelength_um, dtype=float) ** 2
        out = np.full_like(lam2, self.a0)
        for b, c in self.terms:
            out = out + b * lam2 / (lam2 - c)
        return out


@dataclass(frozen=True)
class Polynomial(_IndexModel):
    """``n^2 = sum c_i * lam^p_i`` (refractiveindex.info "formula 3")."""

    terms: Tuple[Tuple[float, float], ...]  # (coefficient, power)

    def n2(self, wavelength_um):
        lam = np.asarray(wavelength_um, dtype=float)
        out = np.zeros_like(lam)
        for c, p in self.terms:
            out = out + c * lam**p
        return out


@dataclass(frozen=True)
class ConstantIndex(_IndexModel):
    """Wavelength-independent ``n`` (and optional ``k``)."""

    n_const: float
    k_const: float = 0.0

    def n2(self, wavelength_um):
        lam = np.asarray(wavelength_um, dtype=float)
        return np.full_like(lam, self.n_const**2)

    def k(self, wavelength_um):
        lam = np.asarray(wavelength_um, dtype=float)
        return np.full_like(lam, self.k_const)


@dataclass(frozen=True)
class TabulatedNK(_IndexModel):
    """Measured (lambda, n[, k]) samples, linearly interpolated. Wavelengths
    must be strictly ascending; queries outside the table raise via the
    owning :class:`Material`'s range check."""

    wavelength_um: Tuple[float, ...]
    n: Tuple[float, ...]
    k_table: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        wl = np.asarray(self.wavelength_um, dtype=float)
        if wl.ndim != 1 or wl.size < 2:
            raise ValueError("TabulatedNK needs >= 2 wavelength samples")
        if not np.all(np.diff(wl) > 0):
            raise ValueError("TabulatedNK wavelengths must be strictly ascending")
        if len(self.n) != wl.size:
            raise ValueError("TabulatedNK n must match wavelength_um length")
        if self.k_table is not None and len(self.k_table) != wl.size:
            raise ValueError("TabulatedNK k must match wavelength_um length")

    def n2(self, wavelength_um):
        lam = np.asarray(wavelength_um, dtype=float)
        return np.interp(lam, self.wavelength_um, self.n) ** 2

    def k(self, wavelength_um):
        lam = np.asarray(wavelength_um, dtype=float)
        if self.k_table is None:
            return np.zeros_like(lam)
        return np.interp(lam, self.wavelength_um, self.k_table)


# ---------------------------------------------------------------------------
# Lorentz single-pole band fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LorentzFit:
    """A single-pole Lorentz fit of a material over a wavelength band, ready
    to become a dispersive :class:`Medium` (``.medium``).

    ``eps(omega) = eps_inf + delta_eps * omega0^2 / (omega0^2 - omega^2)``
    (lossless pole; any tabulated absorption is carried separately as the
    band-centre Ohmic ``conductivity_s_per_m``). A degenerate fit
    (``delta_eps == 0``, e.g. anomalous band slope no passive lossless pole
    can produce, or negligible dispersion) yields a constant-index medium
    with ``lorentz=None``.

    The fit is an IN-BAND surrogate: the pole placement is numerical, not the
    material's physical resonance, so the model has no meaning outside
    ``band_um`` (weakly-dispersive materials often get a far-IR pole whose
    static ``eps(0) = eps_inf + delta_eps`` is wildly unphysical — irrelevant
    for a source whose spectrum lives in the band).
    """

    material_name: str
    band_um: Tuple[float, float]
    eps_inf: float
    delta_eps: float
    resonance_frequency_hz: Optional[float]  # None for the degenerate fit
    conductivity_s_per_m: float
    max_abs_n_error: float  # max |n_fit - n_literature| over the band

    @property
    def pole_wavelength_um(self) -> Optional[float]:
        if self.resonance_frequency_hz is None:
            return None
        return _C0_M_PER_S / self.resonance_frequency_hz * 1e6

    @property
    def pole(self) -> Optional[LorentzPole]:
        if self.resonance_frequency_hz is None or self.delta_eps == 0.0:
            return None
        return LorentzPole(
            resonance_frequency_hz=self.resonance_frequency_hz,
            delta_eps=self.delta_eps,
            linewidth_hz=0.0,
        )

    @property
    def medium(self) -> Medium:
        return Medium(
            permittivity=self.eps_inf,
            conductivity_s_per_m=self.conductivity_s_per_m,
            lorentz=self.pole,
        )

    def eps_model(self, wavelength_um) -> np.ndarray:
        """The fitted real permittivity at ``wavelength_um`` (µm)."""
        lam = np.asarray(wavelength_um, dtype=float)
        out = np.full_like(lam, self.eps_inf)
        if self.pole is not None:
            w0 = 2.0 * math.pi * self.resonance_frequency_hz
            w = 2.0 * math.pi * _freq_hz(lam)
            out = out + self.delta_eps * w0**2 / (w0**2 - w**2)
        return out

    def n_model(self, wavelength_um) -> np.ndarray:
        """The fitted real index at ``wavelength_um`` (µm)."""
        return np.sqrt(self.eps_model(wavelength_um))

    def omega0_dt(self, dl_um: float, courant: float = 0.99) -> float:
        """The ADE stability product ``omega0 * dt`` at grid spacing ``dl_um``
        (µm) with the engine's 3-D Courant timestep
        ``dt = courant * dl / (c0 * sqrt(3))``. The engine's ``validate()``
        rejects a pole with ``omega0 * dt >= 2`` (NUMERICS.md §19.4); keep a
        healthy margin below that."""
        if self.resonance_frequency_hz is None:
            return 0.0
        # lazy import: plugins.__init__ is heavy and materials must stay
        # importable standalone; _constants itself imports nothing.
        from .plugins._constants import engine_dt_s

        return 2.0 * math.pi * self.resonance_frequency_hz \
            * engine_dt_s(dl_um, courant)

    def max_dl_um(self, courant: float = 0.99) -> float:
        """The coarsest grid spacing (µm) that keeps ``omega0 * dt < 2``
        (§19.4). Finer grids only shrink ``omega0 * dt``, so any ``dl`` below
        this is ADE-safe."""
        if self.resonance_frequency_hz is None:
            return math.inf
        w0 = 2.0 * math.pi * self.resonance_frequency_hz
        return (2.0 / w0) * _C0_M_PER_S * math.sqrt(3.0) / courant * 1e6


def _lstsq_pole(
    omega: np.ndarray, eps_target: np.ndarray, omega0: float
) -> Tuple[float, float, float]:
    """Least-squares (eps_inf, delta_eps) for a fixed pole; returns
    (eps_inf, delta_eps, max |n_fit - n_target|)."""
    x = omega0**2 / (omega0**2 - omega**2)
    design = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(design, eps_target, rcond=None)
    eps_inf, delta_eps = float(coef[0]), float(coef[1])
    eps_fit = eps_inf + delta_eps * x
    if np.any(eps_fit <= 0.0):
        return eps_inf, delta_eps, math.inf
    err = float(np.max(np.abs(np.sqrt(eps_fit) - np.sqrt(eps_target))))
    return eps_inf, delta_eps, err


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    """A named optical material: a literature dispersion model plus its
    citation and validity range, convertible to engine ``Medium`` objects."""

    name: str
    model: _IndexModel
    valid_range_um: Tuple[float, float]
    reference: str
    comments: str = ""

    # -- index queries ------------------------------------------------------

    def _check_range(self, wavelength_um) -> np.ndarray:
        lam = np.asarray(wavelength_um, dtype=float)
        lo, hi = self.valid_range_um
        if np.any(lam < lo) or np.any(lam > hi):
            raise ValueError(
                f"{self.name}: wavelength {wavelength_um} um outside the "
                f"validity range [{lo}, {hi}] um of {self.reference.splitlines()[0]!r}"
            )
        return lam

    def n(self, wavelength_um) -> Union[float, np.ndarray]:
        """Real refractive index at ``wavelength_um`` (µm; scalar or array)."""
        lam = self._check_range(wavelength_um)
        out = np.sqrt(self.model.n2(lam))
        return float(out) if out.ndim == 0 else out

    def k(self, wavelength_um) -> Union[float, np.ndarray]:
        """Extinction coefficient at ``wavelength_um`` (µm)."""
        lam = self._check_range(wavelength_um)
        out = self.model.k(lam)
        return float(out) if out.ndim == 0 else out

    def eps(self, wavelength_um) -> Union[complex, np.ndarray]:
        """Complex relative permittivity ``(n + ik)^2`` under the engine's
        ``e^{-i omega t}`` convention (``Im eps >= 0`` is absorbing)."""
        lam = self._check_range(wavelength_um)
        nk = np.sqrt(self.model.n2(lam)) + 1j * self.model.k(lam)
        out = nk**2
        return complex(out) if out.ndim == 0 else out

    # -- Medium builders ----------------------------------------------------

    def medium(
        self,
        wavelength_um: Optional[float] = None,
        *,
        band_um: Optional[Tuple[float, float]] = None,
        **fit_kwargs,
    ) -> Medium:
        """An engine ``Medium`` for this material.

        Exactly one of ``wavelength_um`` (non-dispersive, frozen at that
        wavelength — permittivity ``n^2 - k^2``, absorption mapped to Ohmic
        conductivity) or ``band_um`` (dispersive single-pole Lorentz fit over
        the band, see :meth:`lorentz_fit`; extra keyword arguments are
        forwarded to it) must be given.
        """
        if (wavelength_um is None) == (band_um is None):
            raise ValueError(
                "give exactly one of wavelength_um= (constant medium) or "
                "band_um= (dispersive single-pole fit)"
            )
        if wavelength_um is not None:
            if fit_kwargs:
                raise ValueError(
                    "fit options are only meaningful with band_um="
                )
            lam = float(wavelength_um)
            self._check_range(lam)
            n_v = float(np.sqrt(self.model.n2(np.asarray(lam))))
            k_v = float(self.model.k(np.asarray(lam)))
            eps_real = n_v**2 - k_v**2
            if eps_real < 1.0:
                raise ValueError(
                    f"{self.name}: Re(eps) = {eps_real:.4f} at {lam} um is "
                    "below the engine's permittivity >= 1 bound"
                )
            sigma = 2.0 * math.pi * float(_freq_hz(lam)) * _EPS0_F_PER_M * (
                2.0 * n_v * k_v
            )
            return Medium(permittivity=eps_real, conductivity_s_per_m=sigma)
        fit = self.lorentz_fit(band_um, **fit_kwargs)
        if fit.max_abs_n_error > 1e-3:
            warnings.warn(
                f"{self.name}: single-pole Lorentz fit over {band_um} um has "
                f"max index error {fit.max_abs_n_error:.2e} (> 1e-3); "
                "consider a narrower band",
                stacklevel=2,
            )
        return fit.medium

    def lorentz_fit(
        self,
        band_um: Tuple[float, float],
        *,
        pole_wavelength_um: Optional[float] = None,
        num_samples: int = 128,
        min_n_inf_ratio: float = 0.7,
    ) -> LorentzFit:
        """Fit the engine's single lossless Lorentz pole (+ ``eps_inf``) to
        this material's real permittivity over ``band_um``.

        For each candidate pole placement the model is linear in
        ``(eps_inf, delta_eps)`` and solved by least squares; the placement
        is scanned on both sides of the band (a pole INSIDE the band is a
        singularity) and the best feasible fit wins, with near-ties broken
        toward the longest pole wavelength (lowest ``omega0`` — the largest
        ADE stability margin, §19.4). Feasibility enforces the wire bound
        ``eps_inf >= 1``, passivity ``delta_eps >= 0``, and
        ``n_inf >= min_n_inf_ratio * min(n_band)`` — the Courant-edge guard
        learned from the c-Si benchmark work (an unconstrained near-UV pole
        drives ``eps_inf`` toward 1 and the run diverges).

        ``pole_wavelength_um`` pins the pole instead of scanning (the c-Si
        benchmark's hand placement at 0.6 µm is reproduced this way).

        Absorption is NOT fitted by the pole: any tabulated ``k`` becomes the
        band-centre Ohmic conductivity on the resulting medium.

        If no feasible pole exists (e.g. the band slope is anomalous, which a
        passive lossless pole cannot produce), a degenerate constant fit is
        returned (``delta_eps = 0``; the medium carries no pole) with a
        warning.
        """
        lo, hi = float(band_um[0]), float(band_um[1])
        if not (0.0 < lo < hi):
            raise ValueError(f"band_um must be (lo, hi) with 0 < lo < hi, got {band_um}")
        self._check_range(np.asarray([lo, hi]))

        # sample uniformly in omega (the fit variable)
        w_edges = 2.0 * math.pi * _freq_hz(np.asarray([hi, lo]))
        w = np.linspace(w_edges[0], w_edges[1], num_samples)
        lam = 2.0 * math.pi * _C0_M_PER_S / w * 1e6
        n_v = np.sqrt(self.model.n2(lam))
        k_v = self.model.k(lam)
        eps_target = n_v**2 - k_v**2
        n_target = np.sqrt(eps_target)
        n_min = float(np.min(n_target))

        lam_c = 0.5 * (lo + hi)
        n_c = float(np.sqrt(self.model.n2(np.asarray(lam_c))))
        k_c = float(self.model.k(np.asarray(lam_c)))
        sigma = 2.0 * math.pi * float(_freq_hz(lam_c)) * _EPS0_F_PER_M * (
            2.0 * n_c * k_c
        )

        def feasible(eps_inf: float, delta_eps: float) -> bool:
            return (
                delta_eps >= 0.0
                and eps_inf >= 1.0
                and math.sqrt(max(eps_inf, 0.0)) >= min_n_inf_ratio * n_min
            )

        eps_mean = float(np.mean(eps_target))
        const_err = float(np.max(np.abs(np.sqrt(eps_mean) - n_target)))
        degenerate = LorentzFit(
            material_name=self.name,
            band_um=(lo, hi),
            eps_inf=eps_mean,
            delta_eps=0.0,
            resonance_frequency_hz=None,
            conductivity_s_per_m=sigma,
            max_abs_n_error=const_err,
        )

        def fit_at(lam0: float):
            w0 = 2.0 * math.pi * float(_freq_hz(lam0))
            eps_inf, delta_eps, err = _lstsq_pole(w, eps_target, w0)
            return eps_inf, delta_eps, err, w0

        if pole_wavelength_um is not None:
            lam0 = float(pole_wavelength_um)
            if lo <= lam0 <= hi:
                raise ValueError(
                    f"pole_wavelength_um={lam0} lies inside the band {band_um}"
                )
            eps_inf, delta_eps, err, w0 = fit_at(lam0)
            if not feasible(eps_inf, delta_eps) or not math.isfinite(err):
                raise ValueError(
                    f"{self.name}: no feasible single-pole fit with the pole "
                    f"pinned at {lam0} um over {band_um} um "
                    f"(eps_inf={eps_inf:.4f}, delta_eps={delta_eps:.4f})"
                )
            return LorentzFit(
                material_name=self.name,
                band_um=(lo, hi),
                eps_inf=eps_inf,
                delta_eps=delta_eps,
                resonance_frequency_hz=w0 / (2.0 * math.pi),
                conductivity_s_per_m=sigma,
                max_abs_n_error=err,
            )

        # negligible dispersion over the band: a pole would only cost engine
        # ADE state — return the constant medium outright
        if const_err <= 1e-6:
            return degenerate

        # scan pole placements on both sides of the band (log-spaced), then
        # zoom once around the coarse winner
        candidates = np.concatenate(
            [
                np.geomspace(0.05 * lo, 0.80 * lo, 60),
                np.geomspace(1.25 * hi, 40.0 * hi, 40),
            ]
        )
        results = []
        for lam0 in candidates:
            eps_inf, delta_eps, err, w0 = fit_at(float(lam0))
            if feasible(eps_inf, delta_eps) and math.isfinite(err):
                results.append((err, float(lam0), eps_inf, delta_eps, w0))
        if results:
            zoom_center = min(results)[1]
            for lam0 in np.geomspace(0.8 * zoom_center, 1.25 * zoom_center, 40):
                lam0 = float(lam0)
                if lo <= lam0 <= hi or not (
                    lam0 < 0.80 * lo or lam0 > 1.25 * hi
                ):
                    continue
                eps_inf, delta_eps, err, w0 = fit_at(lam0)
                if feasible(eps_inf, delta_eps) and math.isfinite(err):
                    results.append((err, lam0, eps_inf, delta_eps, w0))
            best_err = min(r[0] for r in results)
            # near-tie break: among fits within 20% (+1e-9 absolute floor) of
            # the best error, prefer the LONGEST pole wavelength (most
            # ADE-stability headroom at coarse grids)
            tol = best_err * 1.2 + 1e-9
            err, lam0, eps_inf, delta_eps, w0 = max(
                (r for r in results if r[0] <= tol), key=lambda r: r[1]
            )
            return LorentzFit(
                material_name=self.name,
                band_um=(lo, hi),
                eps_inf=eps_inf,
                delta_eps=delta_eps,
                resonance_frequency_hz=w0 / (2.0 * math.pi),
                conductivity_s_per_m=sigma,
                max_abs_n_error=err,
            )

        # no feasible pole (anomalous slope, ...) -> constant medium
        warnings.warn(
            f"{self.name}: no feasible passive single-pole Lorentz fit over "
            f"{band_um} um (anomalous band slope?); returning a constant-index "
            f"medium (max index error {const_err:.2e})",
            stacklevel=3,
        )
        return degenerate

    # -- user data ----------------------------------------------------------

    @classmethod
    def from_nk_data(
        cls,
        name: str,
        *,
        wavelength_um: Sequence[float],
        n: Sequence[float],
        k: Optional[Sequence[float]] = None,
        reference: str = "user-supplied measurement data",
        comments: str = "",
    ) -> "Material":
        """Wrap measured ``(lambda, n[, k])`` samples (e.g. ellipsometry) as
        a :class:`Material`; validity range = the table's span."""
        model = TabulatedNK(
            wavelength_um=tuple(float(x) for x in wavelength_um),
            n=tuple(float(x) for x in n),
            k_table=None if k is None else tuple(float(x) for x in k),
        )
        return cls(
            name=name,
            model=model,
            valid_range_um=(model.wavelength_um[0], model.wavelength_um[-1]),
            reference=reference,
            comments=comments,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lo, hi = self.valid_range_um
        return (
            f"Material({self.name!r}, {lo}-{hi} um, "
            f"{self.reference.splitlines()[0]!r})"
        )


# ---------------------------------------------------------------------------
# built-in materials
# ---------------------------------------------------------------------------
# Coefficients reproduced verbatim from the cited papers via the
# refractiveindex.info database (CC0 1.0). "formula 1" Sellmeier terms enter
# as (B, C^2); "formula 2" terms as (B, C) — C already in µm².

Vacuum = Material(
    name="Vacuum",
    model=ConstantIndex(n_const=1.0),
    valid_range_um=(0.0, math.inf),
    reference="exact",
    comments="n = 1; dry air differs by ~3e-4 (use vacuum for air cladding)",
)

# H. H. Li, J. Phys. Chem. Ref. Data 9, 561-658 (1993 review), 293 K —
# tabulated n, transparent range (the same source the gds benchmark's
# dispersive-Si pole is anchored to).
cSi = Material(
    name="cSi",
    model=TabulatedNK(
        wavelength_um=(
            1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38,
            1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.80, 1.90, 2.00,
            2.25, 2.50, 2.75, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00,
            10.0, 11.0, 12.0, 13.0, 14.0,
        ),
        n=(
            3.5167, 3.5133, 3.5102, 3.5072, 3.5043, 3.5016, 3.4990, 3.4965,
            3.4941, 3.4918, 3.4896, 3.4845, 3.4799, 3.4757, 3.4719, 3.4684,
            3.4653, 3.4597, 3.4550, 3.4510, 3.4431, 3.4375, 3.4334, 3.4302,
            3.4229, 3.4195, 3.4177, 3.4165, 3.4158, 3.4153, 3.4150, 3.4147,
            3.4145, 3.4144, 3.4142,
        ),
    ),
    valid_range_um=(1.2, 14.0),
    reference=(
        "H. H. Li, J. Phys. Chem. Ref. Data 9, 561-658 (1993), 293 K, "
        "doi:10.1063/1.555624"
    ),
    comments="crystalline silicon, transparent range (tabulated n, k ~ 0)",
)

SiO2 = Material(
    name="SiO2",
    model=Sellmeier(
        a0=1.0,
        terms=(
            (0.6961663, 0.0684043**2),
            (0.4079426, 0.1162414**2),
            (0.8974794, 9.896161**2),
        ),
    ),
    valid_range_um=(0.21, 6.7),
    reference=(
        "I. H. Malitson, J. Opt. Soc. Am. 55, 1205-1208 (1965), "
        "doi:10.1364/JOSA.55.001205 (IR validity to 6.7 um per Tan 1998)"
    ),
    comments="fused silica, 20 C",
)

Si3N4 = Material(
    name="Si3N4",
    model=Sellmeier(
        a0=1.0,
        terms=(
            (3.0249, 0.1353406**2),
            (40314.0, 1239.842**2),
        ),
    ),
    valid_range_um=(0.310, 5.504),
    reference=(
        "K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson, "
        "Opt. Lett. 40, 4823-4826 (2015), doi:10.1364/OL.40.004823"
    ),
    comments="stoichiometric LPCVD Si3N4 film; PECVD SiN varies — use "
    "Material.from_nk_data with your own ellipsometry",
)

GaAs = Material(
    name="GaAs",
    model=Sellmeier(
        a0=5.372514,
        terms=(
            (5.466742, 0.4431307**2),
            (0.02429960, 0.8746453**2),
            (1.957522, 36.9166**2),
        ),
    ),
    valid_range_um=(0.97, 17.0),
    reference=(
        "T. Skauli et al., J. Appl. Phys. 94, 6447-6455 (2003), "
        "doi:10.1063/1.1621740"
    ),
    comments="22 C",
)

InP = Material(
    name="InP",
    model=Sellmeier(
        a0=7.255,
        terms=(
            (2.316, 0.6263**2),
            (2.765, 32.935**2),
        ),
    ),
    valid_range_um=(0.95, 10.0),
    reference=(
        "G. D. Pettit and W. J. Turner, J. Appl. Phys. 36, 2081 (1965), "
        "doi:10.1063/1.1714410 (Sellmeier form per Handbook of Optics 2nd ed.)"
    ),
    comments="room temperature",
)

Ge = Material(
    name="Ge",
    model=Sellmeier(
        a0=1.0,
        terms=(
            (0.4886331, 1.393959),
            (14.5142535, 0.1626427),
            (0.0091224, 752.190),
        ),
    ),
    valid_range_um=(2.0, 14.0),
    reference=(
        "J. H. Burnett et al., Proc. SPIE 9974, 99740X (2016), "
        "doi:10.1117/12.2237978"
    ),
    comments="295 K; Ge is absorbing below ~1.9 um (indirect edge) — "
    "mid-IR use only",
)

Sapphire = Material(
    name="Sapphire",
    model=Sellmeier(
        a0=1.0,
        terms=(
            (1.4313493, 0.0726631**2),
            (0.65054713, 0.1193242**2),
            (5.3414021, 18.028251**2),
        ),
    ),
    valid_range_um=(0.20, 5.0),
    reference=(
        "I. H. Malitson and M. J. Dodge, J. Opt. Soc. Am. 62, 1405 (1972), "
        "ordinary ray"
    ),
    comments="single-crystal Al2O3, ordinary ray; ALD/amorphous alumina "
    "films have lower n — use Material.from_nk_data",
)

AlN = Material(
    name="AlN",
    model=Sellmeier(
        a0=3.1399,
        terms=(
            (1.3786, 0.1715**2),
            (3.861, 15.03**2),
        ),
    ),
    valid_range_um=(0.22, 5.0),
    reference=(
        "J. Pastrnak and L. Roskovcova, Phys. Stat. Sol. 14, K5-K8 (1966), "
        "doi:10.1002/pssb.19660140127, ordinary ray"
    ),
    comments="single-crystal AlN, ordinary ray",
)

# DeVore's n^2 = 5.913 + 0.2441/(lam^2 - 0.0803) recast exactly into the
# Sellmeier form B*lam^2/(lam^2 - C) with B = 0.2441/0.0803, so that
# a0 + B = 5.913.
_TIO2_B = 0.2441 / 0.0803
TiO2 = Material(
    name="TiO2",
    model=Sellmeier(a0=5.913 - _TIO2_B, terms=((_TIO2_B, 0.0803),)),
    valid_range_um=(0.43, 1.53),
    reference=(
        "J. R. DeVore, J. Opt. Soc. Am. 41, 416-419 (1951), "
        "doi:10.1364/JOSA.41.000416, ordinary ray"
    ),
    comments="rutile single crystal, ordinary ray; validity ends at 1.53 um "
    "(short of the C-band); amorphous/anatase films differ",
)

LiNbO3_o = Material(
    name="LiNbO3_o",
    model=Sellmeier(
        a0=1.0,
        terms=(
            (2.6734, 0.01764),
            (1.2290, 0.05914),
            (12.614, 474.60),
        ),
    ),
    valid_range_um=(0.4, 5.0),
    reference=(
        "D. E. Zelmon, D. L. Small, D. Jundt, J. Opt. Soc. Am. B 14, "
        "3319-3322 (1997), doi:10.1364/JOSAB.14.003319, ordinary ray"
    ),
    comments="congruent LiNbO3, ordinary ray",
)

LiNbO3_e = Material(
    name="LiNbO3_e",
    model=Sellmeier(
        a0=1.0,
        terms=(
            (2.9804, 0.02047),
            (0.5981, 0.0666),
            (8.9543, 416.08),
        ),
    ),
    valid_range_um=(0.4, 5.0),
    reference=(
        "D. E. Zelmon, D. L. Small, D. Jundt, J. Opt. Soc. Am. B 14, "
        "3319-3322 (1997), doi:10.1364/JOSAB.14.003319, extraordinary ray"
    ),
    comments="congruent LiNbO3, extraordinary ray",
)

MgF2_o = Material(
    name="MgF2_o",
    model=Sellmeier(
        a0=1.0,
        terms=(
            (0.48755108, 0.04338408**2),
            (0.39875031, 0.09461442**2),
            (2.3120353, 23.793604**2),
        ),
    ),
    valid_range_um=(0.2, 7.0),
    reference=(
        "M. J. Dodge, Appl. Opt. 23, 1980-1985 (1984), "
        "doi:10.1364/AO.23.001980, ordinary ray"
    ),
    comments="single-crystal MgF2, ordinary ray",
)

MgF2_e = Material(
    name="MgF2_e",
    model=Sellmeier(
        a0=1.0,
        terms=(
            (0.41344023, 0.03684262**2),
            (0.50497499, 0.09076162**2),
            (2.4904862, 23.771995**2),
        ),
    ),
    valid_range_um=(0.2, 7.0),
    reference=(
        "M. J. Dodge, Appl. Opt. 23, 1980-1985 (1984), "
        "doi:10.1364/AO.23.001980, extraordinary ray"
    ),
    comments="single-crystal MgF2, extraordinary ray",
)

CaF2 = Material(
    name="CaF2",
    model=Sellmeier(
        a0=1.0,
        terms=(
            (0.5675888, 0.050263605**2),
            (0.4710914, 0.1003909**2),
            (3.8484723, 34.649040**2),
        ),
    ),
    valid_range_um=(0.23, 9.7),
    reference=(
        "I. H. Malitson, Appl. Opt. 2, 1103-1107 (1963), "
        "doi:10.1364/AO.2.001103"
    ),
    comments="",
)

PMMA = Material(
    name="PMMA",
    model=Polynomial(
        terms=(
            (2.1778, 0.0),
            (6.1209e-3, 2.0),
            (-1.5004e-3, 4.0),
            (2.3678e-2, -2.0),
            (-4.2137e-3, -4.0),
            (7.3417e-4, -6.0),
            (-4.5042e-5, -8.0),
        ),
    ),
    valid_range_um=(0.42, 1.62),
    reference=(
        "G. Beadie, M. Brindza, R. A. Flynn, A. Rosenberg, J. S. Shirk, "
        "Appl. Opt. 54, F139-F143 (2015), doi:10.1364/AO.54.00F139"
    ),
    comments="poly(methyl methacrylate)",
)


MATERIALS = {
    m.name: m
    for m in (
        Vacuum,
        cSi,
        SiO2,
        Si3N4,
        GaAs,
        InP,
        Ge,
        Sapphire,
        AlN,
        TiO2,
        LiNbO3_o,
        LiNbO3_e,
        MgF2_o,
        MgF2_e,
        CaF2,
        PMMA,
    )
}


def get(name: str) -> Material:
    """Look up a built-in material by name (see ``MATERIALS`` for the list)."""
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(
            f"unknown material {name!r}; available: {sorted(MATERIALS)}"
        ) from None
