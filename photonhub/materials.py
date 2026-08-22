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
entries (use the ray your polarization sees). METALS (Au/Ag/Cu Johnson &
Christy, Al Rakic) are included as tabulated n/k and are DISPERSIVE-ONLY:
their optical/IR Re eps is negative, which no frozen permittivity >= 1
medium can carry — build their media through the multi-pole fitter,

>>> gold = ph.materials.Au.pole_fit(band_um=(1.0, 1.6), n_lorentz=1,
...                                 drude=True).medium

(medium(wavelength_um=...)/medium(band_um=...) raise on a metallic
band and point here).

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

from .components.structures import (MAX_ADE_POLES, DrudePole,
                                    LorentzPole, Medium)

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
    "Au",
    "Ag",
    "Cu",
    "Al",
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


@dataclass(frozen=True)
class PoleFit:
    """A multi-pole (Lorentz + optional Drude) fit of a material band —
    the schema-1.17 companion of :class:`LorentzFit`, fitting the COMPLEX
    permittivity (real dispersion AND absorption ride the poles' damping,
    no band-centre Ohmic sigma needed).

    ``max_abs_eps_error`` / ``max_rel_eps_error`` are the band maxima of
    ``|eps_fit - eps_target|`` (absolute / relative to ``|eps_target|``)."""

    material_name: str
    band_um: Tuple[float, float]
    eps_inf: float
    lorentz: Tuple[LorentzPole, ...]
    drude: Tuple[DrudePole, ...]
    max_abs_eps_error: float
    max_rel_eps_error: float

    @property
    def medium(self) -> Medium:
        """The dispersive engine :class:`Medium` (poles in fit order)."""
        return Medium(permittivity=self.eps_inf, poles=self.lorentz,
                      drude=self.drude)

    def eps_model(self, wavelength_um) -> np.ndarray:
        """Complex model permittivity at ``wavelength_um`` (µm)."""
        lam = np.asarray(wavelength_um, dtype=float)
        w = 2.0 * math.pi * _C0_M_PER_S / (lam * 1e-6)
        eps = np.full_like(w, self.eps_inf, dtype=np.complex128)
        for p in self.lorentz:
            w0 = 2.0 * math.pi * p.resonance_frequency_hz
            g = 2.0 * math.pi * p.linewidth_hz
            eps += p.delta_eps * w0 * w0 / (w0 * w0 - w * w - 1j * g * w)
        for d in self.drude:
            wp = 2.0 * math.pi * d.plasma_frequency_hz
            g = 2.0 * math.pi * d.linewidth_hz
            eps += -(wp * wp) / (w * w + 1j * g * w)
        return eps

    def omega0_dt(self, dl_um: float, courant: float = 0.99) -> float:
        """The largest ``omega0 * dt`` over the fitted Lorentz poles at grid
        spacing ``dl_um`` — must stay < 2 for ADE stability (§19.4; Drude
        poles impose no resonance bound)."""
        if not self.lorentz:
            return 0.0
        dt = courant * (dl_um * 1e-6) / (_C0_M_PER_S * math.sqrt(3.0))
        f_max = max(p.resonance_frequency_hz for p in self.lorentz)
        return 2.0 * math.pi * f_max * dt


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
                    "below the engine's permittivity >= 1 bound — a metal "
                    "needs a dispersive medium: use "
                    "pole_fit(band_um=..., drude=True)"
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
        if float(np.min(eps_target)) < 1.0:
            raise ValueError(
                f"{self.name}: Re eps drops to "
                f"{float(np.min(eps_target)):.3g} in {tuple(band_um)} "
                "um — a metallic/near-metallic band cannot be represented by "
                "the single lossless-Lorentz fit; use "
                "pole_fit(band_um=..., drude=True) instead")
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

    def pole_fit(
        self,
        band_um: Tuple[float, float],
        *,
        n_lorentz: int = 2,
        drude: bool = False,
        num_samples: int = 128,
    ) -> PoleFit:
        """Fit the material's COMPLEX permittivity over ``band_um`` with
        ``n_lorentz`` Lorentz poles plus (optionally) one Drude term — the
        schema-1.17 multi-pole fit for metals and wideband dielectrics.

        Where :meth:`lorentz_fit` fits ``Re eps`` with ONE lossless pole and
        moves absorption into a band-centre Ohmic sigma, this fits real AND
        imaginary parts together with damped poles (passivity enforced through
        the §19.1 bounds: ``delta_eps >= 0``, ``gamma >= 0``, ``wp > 0``).
        Deterministic multi-start least squares (log-spaced pole ladders on
        both sides of the band; no RNG). Raises when the request exceeds the
        engine's pole budget or scipy is unavailable.

        Returns a :class:`PoleFit`; use ``fit.medium`` in structures, check
        ``fit.max_rel_eps_error`` and ``fit.omega0_dt(dl_um)`` before running.
        """
        from scipy.optimize import least_squares

        if n_lorentz < 0 or (n_lorentz == 0 and not drude):
            raise ValueError("need at least one pole (n_lorentz >= 1 or drude)")
        n_total = n_lorentz + (1 if drude else 0)
        if n_total > MAX_ADE_POLES:
            raise ValueError(
                f"{n_total} poles exceed the engine budget of "
                f"{MAX_ADE_POLES} (NUMERICS.md §19)")
        lo, hi = float(band_um[0]), float(band_um[1])
        if not (0.0 < lo < hi):
            raise ValueError(f"invalid band_um {band_um}")

        lam = np.linspace(lo, hi, int(num_samples))
        w = 2.0 * math.pi * _C0_M_PER_S / (lam * 1e-6)
        eps_t = np.asarray(self.eps(lam), dtype=np.complex128)
        scale = float(np.max(np.abs(eps_t)))
        f_lo, f_hi = float(w.min() / (2 * math.pi)), float(w.max() / (2 * math.pi))

        # parameter vector: [eps_inf, (f0, de, g) * n_lorentz, (fp, g)?]
        def unpack(x):
            eps_inf = x[0]
            lor = [(x[1 + 3 * j], x[2 + 3 * j], x[3 + 3 * j])
                   for j in range(n_lorentz)]
            dr = (x[1 + 3 * n_lorentz], x[2 + 3 * n_lorentz]) if drude else None
            return eps_inf, lor, dr

        def model(x):
            eps_inf, lor, dr = unpack(x)
            eps = np.full_like(w, eps_inf, dtype=np.complex128)
            for (f0, de, g) in lor:
                w0 = 2.0 * math.pi * f0
                gg = 2.0 * math.pi * g
                eps = eps + de * w0 * w0 / (w0 * w0 - w * w - 1j * gg * w)
            if dr is not None:
                wp = 2.0 * math.pi * dr[0]
                gg = 2.0 * math.pi * dr[1]
                eps = eps + -(wp * wp) / (w * w + 1j * gg * w)
            return eps

        def resid(x):
            d = (model(x) - eps_t) / scale
            return np.concatenate([d.real, d.imag])

        # bounds: passivity + poles anywhere within a decade of the band
        lo_b = [1.0] + [f_lo / 10.0, 0.0, 0.0] * n_lorentz
        hi_b = [np.inf] + [f_hi * 10.0, np.inf, f_hi * 10.0] * n_lorentz
        if drude:
            # metal plasma frequencies sit far above an IR band (Au: ~2.2e15
            # Hz vs f_hi ~ 2.9e14 at 1.05 um) — give fp two decades of room
            lo_b += [f_lo / 10.0, 0.0]
            hi_b += [f_hi * 100.0, f_hi * 10.0]

        eps_inf0 = float(np.clip(np.min(eps_t.real), 1.0, None)) if             float(np.min(eps_t.real)) > 1.0 else 1.0

        # deterministic multi-start: pole ladders below/above/straddling
        ladders = []
        for kind in ("below", "above", "straddle"):
            f0s = []
            for j in range(max(n_lorentz, 1)):
                frac = (j + 1) / (max(n_lorentz, 1) + 1)
                if kind == "below":
                    f0s.append(f_lo * (0.2 + 0.6 * frac))
                elif kind == "above":
                    f0s.append(f_hi * (1.5 + 3.0 * frac))
                else:
                    f0s.append(f_lo + frac * (f_hi - f_lo) * 2.0)
            ladders.append(f0s)

        best = None
        last_exc = None
        for f0s in ladders:
            x0 = [eps_inf0]
            for j in range(n_lorentz):
                x0 += [f0s[j], max(1.0, float(np.ptp(eps_t.real))),
                       0.05 * f0s[j]]
            if drude:
                # seed from the negative band eps if present (Drude limit:
                # -eps ~ (wp/w)^2), clamped inside the bounds
                neg = max(0.0, -float(np.min(eps_t.real)))
                fp0 = min(f_hi * math.sqrt(max(neg + 1.0, 1.0)),
                          f_hi * 90.0)
                x0 += [fp0, 0.05 * fp0]
            try:
                sol = least_squares(resid, x0, bounds=(lo_b, hi_b),
                                    max_nfev=4000)
            except ValueError as exc:
                last_exc = exc
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        if best is None:
            raise ValueError(
                f"{self.name}: pole_fit failed to converge on "
                f"[{lo}, {hi}] um — try different n_lorentz/drude "
                f"(last solver error: {last_exc})")

        eps_inf, lor, dr = unpack(best.x)
        lorentz = tuple(
            LorentzPole(resonance_frequency_hz=f0, delta_eps=de,
                        linewidth_hz=g)
            for (f0, de, g) in lor if de > 1e-12)
        drude_poles = ()
        if dr is not None and dr[0] > 0.0:
            drude_poles = (DrudePole(plasma_frequency_hz=dr[0],
                                     linewidth_hz=dr[1]),)
        fit = PoleFit(
            material_name=self.name,
            band_um=(lo, hi),
            eps_inf=float(max(eps_inf, 1.0)),
            lorentz=lorentz,
            drude=drude_poles,
            max_abs_eps_error=0.0,
            max_rel_eps_error=0.0,
        )
        err = np.abs(fit.eps_model(lam) - eps_t)
        object.__setattr__(fit, "max_abs_eps_error", float(np.max(err)))
        object.__setattr__(
            fit, "max_rel_eps_error",
            float(np.max(err / np.maximum(np.abs(eps_t), 1e-12))))
        return fit

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


# ---------------------------------------------------------------------------
# metals (schema 1.17 multi-pole/Drude engine): tabulated n/k, DISPERSIVE-ONLY
# ---------------------------------------------------------------------------

Au = Material(
    name="Au",
    valid_range_um=(0.1879, 1.937),
    reference=("P. B. Johnson and R. W. Christy, 'Optical constants of the "
               "noble metals,' Phys. Rev. B 6, 4370-4379 (1972). Tabulated "
               'n/k via refractiveindex.info (CC0 1.0).'),
    comments=('gold (evaporated film); METAL - negative Re eps in the '
              'optical/IR: build media with pole_fit(band_um=..., '
              'drude=True), not medium(...).'),
    model=TabulatedNK(
        wavelength_um=(0.1879, 0.1916, 0.1953, 0.1993, 0.2033, 0.2073, 0.2119, 0.2164, 0.2214,
        0.2262, 0.2313, 0.2371, 0.2426, 0.249, 0.2551, 0.2616, 0.2689, 0.2761,
        0.2844, 0.2924, 0.3009, 0.3107, 0.3204, 0.3315, 0.3425, 0.3542, 0.3679,
        0.3815, 0.3974, 0.4133, 0.4305, 0.4509, 0.4714, 0.4959, 0.5209, 0.5486,
        0.5821, 0.6168, 0.6595, 0.7045, 0.756, 0.8211, 0.892, 0.984, 1.088,
        1.216, 1.393, 1.61, 1.937),
        n=(1.28, 1.32, 1.34, 1.33, 1.33, 1.3, 1.3, 1.3, 1.3, 1.31, 1.3, 1.32,
        1.32, 1.33, 1.33, 1.35, 1.38, 1.43, 1.47, 1.49, 1.53, 1.53, 1.54, 1.48,
        1.48, 1.5, 1.48, 1.46, 1.47, 1.46, 1.45, 1.38, 1.31, 1.04, 0.62, 0.43,
        0.29, 0.21, 0.14, 0.13, 0.14, 0.16, 0.17, 0.22, 0.27, 0.35, 0.43, 0.56,
        0.92),
        k_table=(1.188, 1.203, 1.226, 1.251, 1.277, 1.304, 1.35, 1.387, 1.427, 1.46,
        1.497, 1.536, 1.577, 1.631, 1.688, 1.749, 1.803, 1.847, 1.869, 1.878,
        1.889, 1.893, 1.898, 1.883, 1.871, 1.866, 1.895, 1.933, 1.952, 1.958,
        1.948, 1.914, 1.849, 1.833, 2.081, 2.455, 2.863, 3.272, 3.697, 4.103,
        4.542, 5.083, 5.663, 6.35, 7.15, 8.145, 9.519, 11.21, 13.78),
    ),
)

Ag = Material(
    name="Ag",
    valid_range_um=(0.1879, 1.937),
    reference=("P. B. Johnson and R. W. Christy, 'Optical constants of the "
               "noble metals,' Phys. Rev. B 6, 4370-4379 (1972). Tabulated "
               'n/k via refractiveindex.info (CC0 1.0).'),
    comments=('silver (evaporated film); METAL - negative Re eps in the '
              'optical/IR: build media with pole_fit(band_um=..., '
              'drude=True), not medium(...).'),
    model=TabulatedNK(
        wavelength_um=(0.1879, 0.1916, 0.1953, 0.1993, 0.2033, 0.2073, 0.2119, 0.2164, 0.2214,
        0.2262, 0.2313, 0.2371, 0.2426, 0.249, 0.2551, 0.2616, 0.2689, 0.2761,
        0.2844, 0.2924, 0.3009, 0.3107, 0.3204, 0.3315, 0.3425, 0.3542, 0.3679,
        0.3815, 0.3974, 0.4133, 0.4305, 0.4509, 0.4714, 0.4959, 0.5209, 0.5486,
        0.5821, 0.6168, 0.6595, 0.7045, 0.756, 0.8211, 0.892, 0.984, 1.088,
        1.216, 1.393, 1.61, 1.937),
        n=(1.07, 1.1, 1.12, 1.14, 1.15, 1.18, 1.2, 1.22, 1.25, 1.26, 1.28, 1.28,
        1.3, 1.31, 1.33, 1.35, 1.38, 1.41, 1.41, 1.39, 1.34, 1.13, 0.81, 0.17,
        0.14, 0.1, 0.07, 0.05, 0.05, 0.05, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06,
        0.05, 0.06, 0.05, 0.04, 0.03, 0.04, 0.04, 0.04, 0.04, 0.09, 0.13, 0.15,
        0.24),
        k_table=(1.212, 1.232, 1.255, 1.277, 1.296, 1.312, 1.325, 1.336, 1.342, 1.344,
        1.357, 1.367, 1.378, 1.389, 1.393, 1.387, 1.372, 1.331, 1.264, 1.161,
        0.964, 0.616, 0.392, 0.829, 1.142, 1.419, 1.657, 1.864, 2.07, 2.275,
        2.462, 2.657, 2.869, 3.093, 3.324, 3.586, 3.858, 4.152, 4.483, 4.838,
        5.242, 5.727, 6.312, 6.992, 7.795, 8.828, 10.1, 11.85, 14.08),
    ),
)

Cu = Material(
    name="Cu",
    valid_range_um=(0.1879, 1.937),
    reference=("P. B. Johnson and R. W. Christy, 'Optical constants of the "
               "noble metals,' Phys. Rev. B 6, 4370-4379 (1972). Tabulated "
               'n/k via refractiveindex.info (CC0 1.0).'),
    comments=('copper (evaporated film); METAL - negative Re eps in the '
              'optical/IR: build media with pole_fit(band_um=..., '
              'drude=True), not medium(...).'),
    model=TabulatedNK(
        wavelength_um=(0.1879, 0.1916, 0.1953, 0.1993, 0.2033, 0.2073, 0.2119, 0.2164, 0.2214,
        0.2262, 0.2313, 0.2371, 0.2426, 0.249, 0.2551, 0.2616, 0.2689, 0.2761,
        0.2844, 0.2924, 0.3009, 0.3107, 0.3204, 0.3315, 0.3425, 0.3542, 0.3679,
        0.3815, 0.3974, 0.4133, 0.4305, 0.4509, 0.4714, 0.4959, 0.5209, 0.5486,
        0.5821, 0.6168, 0.6595, 0.7045, 0.756, 0.8211, 0.892, 0.984, 1.088,
        1.216, 1.393, 1.61, 1.937),
        n=(0.94, 0.95, 0.97, 0.98, 0.99, 1.01, 1.04, 1.08, 1.13, 1.18, 1.23, 1.28,
        1.34, 1.37, 1.41, 1.41, 1.45, 1.46, 1.45, 1.42, 1.4, 1.38, 1.38, 1.34,
        1.36, 1.37, 1.36, 1.33, 1.32, 1.28, 1.25, 1.24, 1.25, 1.22, 1.18, 1.02,
        0.7, 0.3, 0.22, 0.21, 0.24, 0.26, 0.3, 0.32, 0.36, 0.48, 0.6, 0.76,
        1.09),
        k_table=(1.337, 1.388, 1.44, 1.493, 1.55, 1.599, 1.651, 1.699, 1.737, 1.768,
        1.792, 1.802, 1.799, 1.783, 1.741, 1.691, 1.668, 1.646, 1.633, 1.633,
        1.679, 1.729, 1.783, 1.821, 1.864, 1.916, 1.975, 2.045, 2.116, 2.207,
        2.305, 2.397, 2.483, 2.564, 2.608, 2.577, 2.704, 3.205, 3.747, 4.205,
        4.665, 5.18, 5.768, 6.421, 7.217, 8.245, 9.439, 11.12, 13.43),
    ),
)

Al = Material(
    name="Al",
    valid_range_um=(0.10332, 8.8561),
    reference=("A. D. Rakic, 'Algorithm for the determination of intrinsic "
               'optical constants of metal films: application to '
               "aluminum,' Appl. Opt. 34, 4755-4767 (1995). Tabulated n/k "
               'via refractiveindex.info (CC0 1.0); rows within 0.1-10 um '
               'kept.'),
    comments=('aluminum (intrinsic film; keep fit bands on ONE side of the '
              '827 nm interband peak, e.g. 1.0-1.6 or 2-5 um); METAL - negative Re eps in the '
              'optical/IR: build media with pole_fit(band_um=..., '
              'drude=True), not medium(...).'),
    model=TabulatedNK(
        wavelength_um=(0.10332, 0.11271, 0.12399, 0.13776, 0.15498, 0.17712, 0.20664, 0.24797,
        0.30996, 0.32628, 0.36466, 0.41328, 0.4428, 0.47687, 0.5166, 0.56357,
        0.61993, 0.65225, 0.68881, 0.72932, 0.77491, 0.79478, 0.81569, 0.83774,
        0.88561, 0.91166, 0.93928, 0.96863, 0.99988, 1.0332, 1.1271, 1.2399,
        1.3776, 1.5498, 1.7712, 2.0664, 2.4797, 2.7552, 3.0996, 3.2628, 3.444,
        3.6466, 3.8745, 4.1328, 4.428, 4.7687, 5.166, 5.6357, 6.1993, 6.8881,
        7.7491, 8.8561),
        n=(0.035753, 0.038468, 0.046304, 0.057167, 0.072505, 0.094236, 0.12677,
        0.18137, 0.28003, 0.31474, 0.39877, 0.52135, 0.6079, 0.7278, 0.8734,
        1.0728, 1.366, 1.5724, 1.8301, 2.1606, 2.6154, 2.7675, 2.7668, 2.6945,
        2.2802, 1.9739, 1.6784, 1.4867, 1.4359, 1.3998, 1.3281, 1.3157, 1.3899,
        1.5782, 1.9205, 2.4738, 3.3372, 3.938, 4.7097, 5.0735, 5.4903, 5.9564,
        6.4808, 7.0796, 7.7757, 8.5881, 9.558, 10.742, 12.195, 14.088, 16.755,
        20.837),
        k_table=(0.77163, 0.95677, 1.1555, 1.3775, 1.6366, 1.9519, 2.3563, 2.9029,
        3.7081, 3.9165, 4.3957, 5.0008, 5.3676, 5.7781, 6.2418, 6.7839, 7.4052,
        7.7354, 8.0601, 8.3565, 8.4914, 8.3866, 8.2573, 8.1878, 8.1134, 8.3058,
        8.597, 9.0655, 9.4939, 9.8914, 10.969, 12.245, 13.784, 15.656, 17.991,
        20.982, 25.004, 27.58, 30.737, 32.183, 33.814, 35.608, 37.595, 39.826,
        42.367, 45.257, 48.593, 52.518, 57.156, 62.841, 69.857, 78.274),
    ),
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
        Au,
        Ag,
        Cu,
        Al,
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
