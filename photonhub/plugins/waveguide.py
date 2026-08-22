"""One-call strip / rib waveguide mode analysis.

The PhotonHub analogue of Tidy3D's ``waveguide.RectangularDielectric`` and the
"draw the cross-section, press solve" FDE workflow: declare the geometry (and
optionally the materials), get the solved mode set with the numbers designers
actually quote — ``n_eff``, TE fraction, group index, bend loss — without
hand-assembling a :class:`~photonhub.plugins.vector_modes.VectorModeSolver`.

Two extras beyond a bare solver call:

* **Materials, not just indices.** ``core`` / ``clad`` accept
  :mod:`photonhub.materials` entries (or their names), evaluated at the
  requested wavelength. With materials, ``group_index=True`` includes
  MATERIAL dispersion: the cross-section is re-solved at ``lambda +/- delta``
  with each material's own ``n(lambda)``, so ``n_g`` carries both waveguide
  and material dispersion (plain ``n_core``/``n_clad`` floats give waveguide
  dispersion only, like ``VectorModeSolver.solve(group_index=True)``).
  Material absorption (``k``) is ignored in the mode solve.
* **Rib cross-sections.** ``slab_h_um > 0`` builds a rib: a slab of that
  thickness extending across the window with the core rising out of it.
  Rasterized with exact per-cell area fractions (scalar volume subpixel);
  the pure strip path keeps ``from_rectangular_core``'s KFJ tensor subpixel.

Geometry convention (rib): the stack occupies ``[0, core_h_um]`` vertically —
slab ``[0, slab_h_um]`` across the full width, core
``[-core_w_um/2, +core_w_um/2] x [0, core_h_um]`` — and is centred in the
window. ``slab_h_um = 0`` is the strip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .vector_modes import VectorMode, VectorModeSolver

__all__ = ["WaveguideModes", "rectangular_waveguide"]


def _resolve_n(material, wavelength_um: float, label: str) -> float:
    """A refractive index from a float, a materials-library entry, or its
    name (lazy import keeps the plugin layer independent of the library)."""
    if isinstance(material, (int, float)):
        n = float(material)
        if not n >= 1.0:
            raise ValueError(f"{label} index must be >= 1, got {n}")
        return n
    from .. import materials as _mat
    entry = _mat.get(material) if isinstance(material, str) else material
    medium = entry.medium(wavelength_um=wavelength_um)
    return math.sqrt(float(medium.permittivity))


def _rect_overlap_1d(lo: float, hi: float, a: float, b: float) -> float:
    return max(0.0, min(hi, b) - max(lo, a))


def _rib_eps(
    *, wavelength_um, dl_um, core_w_um, core_h_um, slab_h_um,
    n_core, n_clad, window_w_um, window_h_um,
) -> np.ndarray:
    """Exact area-fraction raster of the rib union (core rect + slab rect)."""
    nx = max(3, int(round(window_w_um / dl_um)))
    ny = max(3, int(round(window_h_um / dl_um)))
    nx += 1 - nx % 2
    ny += 1 - ny % 2
    ec, ecl = n_core**2, n_clad**2
    # window centred on the stack: x in [-W/2, W/2], y in
    # [core_h/2 - H/2 .. core_h/2 + H/2] shifted so the stack [0, core_h]
    # sits centred.
    x0 = -nx * dl_um / 2.0
    y0 = core_h_um / 2.0 - ny * dl_um / 2.0
    eps = np.empty((ny, nx), dtype=float)
    xw = (-core_w_um / 2.0, core_w_um / 2.0)
    for iy in range(ny):
        ylo, yhi = y0 + iy * dl_um, y0 + (iy + 1) * dl_um
        f_core_y = _rect_overlap_1d(ylo, yhi, 0.0, core_h_um) / dl_um
        f_slab_y = _rect_overlap_1d(ylo, yhi, 0.0, slab_h_um) / dl_um
        for ix in range(nx):
            xlo, xhi = x0 + ix * dl_um, x0 + (ix + 1) * dl_um
            f_core_x = _rect_overlap_1d(xlo, xhi, *xw) / dl_um
            # union fill: slab everywhere + core, minus the double-counted
            # core∩slab region (same x extent as the core)
            fill = (f_core_x * f_core_y + f_slab_y
                    - f_core_x * f_slab_y)
            eps[iy, ix] = ecl + min(max(fill, 0.0), 1.0) * (ec - ecl)
    return eps


@dataclass(frozen=True)
class WaveguideModes:
    """Solved modes of :func:`rectangular_waveguide`, sorted by descending
    ``n_eff``. ``n_group`` / ``loss_db_per_cm`` are None when not requested /
    not applicable."""

    modes: Tuple[VectorMode, ...]
    wavelength_um: float
    n_core: float
    n_clad: float
    solver: VectorModeSolver

    @property
    def n_eff(self) -> np.ndarray:
        return np.asarray([m.n_eff for m in self.modes])

    @property
    def te_fractions(self) -> np.ndarray:
        return np.asarray([m.te_fraction for m in self.modes])

    @property
    def polarizations(self) -> List[str]:
        return [m.polarization for m in self.modes]

    @property
    def n_group(self) -> Optional[np.ndarray]:
        vals = [getattr(m, "n_group", None) for m in self.modes]
        if all(v is None for v in vals):
            return None
        return np.asarray([np.nan if v is None else float(v) for v in vals])

    @property
    def loss_db_per_cm(self) -> Optional[np.ndarray]:
        vals = [float(m.loss_db_per_cm) for m in self.modes]
        if all(v == 0.0 for v in vals):
            return None
        return np.asarray(vals)

    def fundamental(self, polarization: str = "TE") -> VectorMode:
        """The highest-``n_eff`` mode of the requested TE/TM family (by
        ``te_fraction`` majority)."""
        pol = polarization.upper()
        if pol not in ("TE", "TM"):
            raise ValueError(f"polarization must be TE or TM, got {pol!r}")
        for m in self.modes:                      # descending n_eff already
            if m.polarization == pol:
                return m
        raise ValueError(
            f"no {pol} mode among the {len(self.modes)} solved "
            f"(TE fractions {np.round(self.te_fractions, 3)}); raise "
            "num_modes or check the cross-section")

    def summary(self) -> str:
        lines = [f"modes @ {self.wavelength_um:.4g} um  "
                 f"(n_core {self.n_core:.4g}, n_clad {self.n_clad:.4g})"]
        ng = self.n_group
        loss = self.loss_db_per_cm
        for i, m in enumerate(self.modes):
            extra = ""
            if ng is not None and not np.isnan(ng[i]):
                extra += f"  n_g {ng[i]:.4f}"
            if loss is not None and not np.isnan(loss[i]):
                extra += f"  loss {loss[i]:.3g} dB/cm"
            lines.append(
                f"  #{i} {m.polarization}  n_eff {np.real(m.n_eff):.4f}  "
                f"te_frac {m.te_fraction:.3f}{extra}")
        return "\n".join(lines)


def rectangular_waveguide(
    *,
    wavelength_um: float,
    core_w_um: float,
    core_h_um: float,
    core: Union[float, str, object],
    clad: Union[float, str, object],
    slab_h_um: float = 0.0,
    dl_um: Optional[float] = None,
    num_modes: int = 2,
    clad_pad_um: Optional[float] = None,
    window_w_um: Optional[float] = None,
    window_h_um: Optional[float] = None,
    group_index: bool = False,
    group_index_step: float = 0.005,
    bend_radius_um: Optional[float] = None,
    num_pml: int = 0,
    pml_strength: float = 1.0,
) -> WaveguideModes:
    """Solve the guided modes of a strip or rib waveguide in one call.

    Parameters
    ----------
    wavelength_um:
        Free-space wavelength.
    core_w_um, core_h_um:
        Core width and total height (rib: measured from the slab bottom).
    core, clad:
        Refractive indices as floats, or :mod:`photonhub.materials` entries /
        names (``"cSi"``, ``"SiO2"``) evaluated at ``wavelength_um``.
    slab_h_um:
        Rib slab thickness (0 = strip).
    dl_um:
        Transverse grid step; default ``min(core_w, core_h) / 24``, capped at
        ``wavelength / (12 * n_core)``.
    num_modes:
        Eigenpairs to return.
    clad_pad_um / window_w_um / window_h_um:
        Window sizing: explicit extents win; otherwise the core plus
        ``clad_pad_um`` (default ``max(0.5, wavelength/2)``) per side.
    group_index:
        Also compute ``n_g`` per mode. With material ``core``/``clad`` this
        re-solves at ``wavelength * (1 +/- group_index_step)`` with each
        material's own ``n(lambda)`` — waveguide AND material dispersion;
        with float indices it defers to the solver's built-in
        (waveguide-dispersion-only) central difference.
    bend_radius_um, num_pml, pml_strength:
        Bend analysis passthrough to ``VectorModeSolver.solve`` (complex
        ``n_eff``; add PML for the radiation loss, see the solver docs).

    Returns
    -------
    WaveguideModes
    """
    if not wavelength_um > 0:
        raise ValueError("wavelength_um must be > 0")
    if not (core_w_um > 0 and core_h_um > 0):
        raise ValueError("core_w_um and core_h_um must be > 0")
    if slab_h_um < 0 or slab_h_um >= core_h_um:
        raise ValueError(
            f"slab_h_um must be in [0, core_h_um), got {slab_h_um}")

    n_core = _resolve_n(core, wavelength_um, "core")
    n_clad = _resolve_n(clad, wavelength_um, "clad")
    if n_core <= n_clad:
        raise ValueError(
            f"core index {n_core:.4g} must exceed cladding {n_clad:.4g}")

    if dl_um is None:
        dl_um = min(min(core_w_um, core_h_um) / 24.0,
                    wavelength_um / (12.0 * n_core))
    if clad_pad_um is None:
        clad_pad_um = max(0.5, wavelength_um / 2.0)
    if window_w_um is None:
        window_w_um = core_w_um + 2.0 * clad_pad_um
    if window_h_um is None:
        window_h_um = core_h_um + 2.0 * clad_pad_um

    material_aware = not (isinstance(core, (int, float))
                          and isinstance(clad, (int, float)))

    def build(lam: float) -> VectorModeSolver:
        nc = _resolve_n(core, lam, "core")
        ncl = _resolve_n(clad, lam, "clad")
        if slab_h_um > 0.0:
            eps = _rib_eps(
                wavelength_um=lam, dl_um=dl_um, core_w_um=core_w_um,
                core_h_um=core_h_um, slab_h_um=slab_h_um, n_core=nc,
                n_clad=ncl, window_w_um=window_w_um, window_h_um=window_h_um)
            return VectorModeSolver(eps, dl_um, dl_um, lam)
        return VectorModeSolver.from_rectangular_core(
            wavelength_um=lam, dl_um=dl_um, core_w_um=core_w_um,
            core_h_um=core_h_um, n_core=nc, n_clad=ncl,
            window_w_um=window_w_um, window_h_um=window_h_um)

    solver = build(wavelength_um)
    solve_kwargs = dict(num_modes=num_modes)
    if bend_radius_um is not None:
        solve_kwargs.update(bend_radius_um=bend_radius_um, num_pml=num_pml,
                            pml_strength=pml_strength)

    want_builtin_ng = group_index and not material_aware
    modes = solver.solve(group_index=want_builtin_ng, **solve_kwargs)

    if group_index and material_aware:
        # material-aware n_g: central difference with the material's own
        # n(lambda) at each side, modes matched to the centre solve by
        # transverse-field overlap.
        dlam = wavelength_um * float(group_index_step)
        sides = []
        for lam in (wavelength_um - dlam, wavelength_um + dlam):
            side_modes = build(lam).solve(num_modes=max(num_modes + 2, 4))
            sides.append(side_modes)

        def match(ref: VectorMode, candidates: Sequence[VectorMode]):
            best, best_ov = None, -1.0
            for c in candidates:
                num = abs(np.vdot(ref.ex, c.ex) + np.vdot(ref.ey, c.ey))
                den = math.sqrt(
                    float(np.sum(np.abs(ref.ex)**2 + np.abs(ref.ey)**2))
                    * float(np.sum(np.abs(c.ex)**2 + np.abs(c.ey)**2)))
                ov = num / den if den > 0 else 0.0
                if ov > best_ov:
                    best, best_ov = c, ov
            if best is None or best_ov < 0.8:
                raise ValueError(
                    "group-index mode tracking lost a mode across the "
                    f"wavelength step (best overlap {best_ov:.2f}); reduce "
                    "group_index_step or raise num_modes")
            return best

        tracked = []
        for m in modes:
            lo = match(m, sides[0])
            hi = match(m, sides[1])
            dn = (np.real(hi.n_eff) - np.real(lo.n_eff)) / (2.0 * dlam)
            ng = float(np.real(m.n_eff) - wavelength_um * dn)
            tracked.append(_with_ng(m, ng))
        modes = tracked

    return WaveguideModes(
        modes=tuple(modes), wavelength_um=float(wavelength_um),
        n_core=n_core, n_clad=n_clad, solver=solver)


def _with_ng(mode: VectorMode, ng: float) -> VectorMode:
    """A copy of a frozen VectorMode carrying ``n_group`` (dataclass-replace
    style, tolerant of the class's exact construction)."""
    import dataclasses
    if dataclasses.is_dataclass(mode):
        return dataclasses.replace(mode, n_group=ng)
    d = dict(mode.__dict__)
    d["n_group"] = ng
    return type(mode)(**d)
