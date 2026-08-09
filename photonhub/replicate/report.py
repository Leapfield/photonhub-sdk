"""Comparison report — ph vs the paper. Turns the extracted figures-of-merit,
the convergence evidence, and the spec's reference values into a per-device
markdown document (and an optional spectrum figure) for human handoff.

The reference values keep their declared units (dB vs linear — the spec enforces
that), so a paper's insertion loss / crosstalk in dB are compared against ph's dB
directly. Digitized paper values are approximate (typically a few points read off
a figure); the deviation column carries that caveat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .spec import PaperSpec

__all__ = ["ComparisonRow", "compare_rows", "build_markdown_report",
           "plot_spectrum_png", "plot_field_intensity_png"]


@dataclass(frozen=True)
class ComparisonRow:
    quantity: str
    port: str
    units: str
    paper_value: Optional[float]
    measured: Optional[float]
    source: str = ""

    @property
    def delta(self) -> Optional[float]:
        if self.paper_value is None or self.measured is None:
            return None
        return self.measured - self.paper_value


def _band_centre_index(wavelengths_um, center_um: float) -> int:
    return min(range(len(wavelengths_um)), key=lambda i: abs(wavelengths_um[i] - center_um))


def compare_rows(spec: PaperSpec, metrics_db: dict) -> List[ComparisonRow]:
    """One row per spec reference: paper value vs ph's band-centre measurement."""
    lams = metrics_db["wavelengths_um"]
    ic = _band_centre_index(lams, spec.optical.center_um)
    il = metrics_db.get("insertion_loss_db")
    xt = metrics_db.get("crosstalk_db", {})
    trans = metrics_db.get("transmission", {})
    refl = metrics_db.get("reflection")
    refl_db = metrics_db.get("reflection_db")
    through_role = "through"

    rows: List[ComparisonRow] = []
    for ref in spec.references:
        measured: Optional[float] = None
        if ref.quantity == "insertion_loss" and il is not None:
            measured = il[ic]
        elif ref.quantity == "crosstalk":
            # the cross-port metric is keyed by the port name
            role = ref.port
            if role in xt:
                measured = xt[role][ic]
        elif ref.quantity == "transmission":
            role = through_role if ref.port == spec.ports.through else ref.port
            if role in trans:
                measured = trans[role][ic]
        elif ref.quantity == "reflection":
            # back-reflection at the input plane, in the reference's own units
            series = refl_db if ref.units == "dB" else refl
            if series is not None:
                measured = series[ic]
        rows.append(ComparisonRow(
            quantity=ref.quantity, port=ref.port, units=ref.units,
            paper_value=ref.paper_value, measured=measured, source=ref.source,
        ))
    return rows


def _overlay_reference(ax, ref, band_um) -> None:
    """Overlay one paper reference on an axis, at the highest fidelity it carries:
    a digitized ``curve`` (dashed line + markers) > a ``flatness`` band (shaded
    span + centreline) > a scalar ``paper_value`` (a dashed line, annotated as a
    bound when ``ref.bound``)."""
    unit = " " + ref.units if ref.units == "dB" else ""
    if ref.curve:
        xs = [w / 1000.0 for w, _ in ref.curve]
        ys = [v for _, v in ref.curve]
        ax.plot(xs, ys, "k--s", ms=3, lw=1, label=f"{ref.label} (digitized)")
        return
    if ref.paper_value is None:
        return
    pv = ref.paper_value
    if ref.flatness is not None:
        ax.fill_between(band_um, pv - ref.flatness, pv + ref.flatness,
                        color="k", alpha=0.12, lw=0)
        ax.axhline(pv, ls="--", color="k", lw=1,
                   label=f"{ref.label} {pv:g}±{ref.flatness:g}{unit}")
    else:
        tag = "≤ " if ref.bound else ""
        ax.axhline(pv, ls="--", color="k", lw=1,
                   label=f"{ref.label} {tag}{pv:g}{unit}")


def plot_spectrum_png(spec: PaperSpec, metrics_db: dict, path) -> Optional[str]:
    """Save a through/crosstalk-vs-wavelength figure (matplotlib) to ``path``,
    overlaying every paper reference (curve / flatness band / bound) against the
    ph curve. Returns the path, or None if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    lams = metrics_db["wavelengths_um"]
    band = spec.optical.band_um

    # A Y-junction's second output is another intentional signal port, not
    # crosstalk. Plot both modal transmissions and derive the two paper-relevant
    # splitter observables rather than labeling the bottom arm as leakage.
    if spec.device.kind == "y_branch":
        trans = metrics_db.get("transmission", {})
        top = trans.get("through")
        bottom_role = spec.ports.cross[0] if spec.ports.cross else None
        bottom = trans.get(bottom_role) if bottom_role else None
        if top is not None and bottom is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
            ax1.plot(lams, top, "-o", ms=3, label=f"ph {spec.ports.through}")
            ax1.plot(lams, bottom, "-o", ms=3, label=f"ph {bottom_role}")
            for r in (r for r in spec.references if r.quantity == "transmission"):
                _overlay_reference(ax1, r, band)
            psr = [10.0 * math.log10(max(t / max(b, 1e-300), 1e-300))
                   for t, b in zip(top, bottom)]
            excess = [-10.0 * math.log10(max(t + b, 1e-300))
                      for t, b in zip(top, bottom)]
            ax2.plot(lams, psr, "-o", ms=3, label="PSR")
            ax2.plot(lams, excess, "-o", ms=3, label="excess loss")
            ax1.set_xlabel("wavelength (µm)"); ax1.set_ylabel("modal transmission")
            ax1.set_title("power in the two output modes"); ax1.legend(fontsize=8)
            ax2.set_xlabel("wavelength (µm)"); ax2.set_ylabel("dB")
            ax2.set_title("split balance and excess loss"); ax2.legend(fontsize=8)
            fig.suptitle(spec.name); fig.tight_layout(); fig.savefig(path, dpi=120)
            plt.close(fig)
            return str(path)

    # Reflective device (a Bragg/DBR): the story is the stopband — plot reflection
    # R(λ) and through-transmission T(λ), not insertion-loss/crosstalk.
    # ``BuiltSim`` may record backward input power for diagnostics on any
    # waveguide device. Only make reflection the primary spectrum when the paper
    # actually declares a reflection target (for example a Bragg grating). A
    # Y-junction still needs its two output transmissions even if diagnostic R is
    # present in the metrics payload.
    reflective_target = any(r.quantity == "reflection" for r in spec.references)
    if reflective_target and "reflection" in metrics_db:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
        ax1.plot(lams, metrics_db["reflection"], "-o", ms=3, color="C3", label="ph R")
        for r in (r for r in spec.references if r.quantity == "reflection"):
            _overlay_reference(ax1, r, band)
        ax1.set_xlabel("wavelength (µm)"); ax1.set_ylabel("reflection R")
        ax1.set_title("back-reflection (stopband)"); ax1.set_ylim(bottom=0)
        ax1.legend(fontsize=8)
        thru = metrics_db.get("transmission", {}).get("through")
        if thru is not None:
            ax2.plot(lams, thru, "-o", ms=3, color="C0", label="ph T")
        ax2.set_xlabel("wavelength (µm)"); ax2.set_ylabel("transmission T")
        ax2.set_title("through-transmission (notch)"); ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=8)
        fig.suptitle(spec.name); fig.tight_layout(); fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    if "insertion_loss_db" in metrics_db:
        ax1.plot(lams, metrics_db["insertion_loss_db"], "-o", ms=3, label="ph")
    for r in (r for r in spec.references if r.quantity == "insertion_loss"):
        _overlay_reference(ax1, r, band)
    ax1.set_xlabel("wavelength (µm)")
    ax1.set_ylabel("insertion loss (dB)")
    ax1.set_title("through-port insertion loss")
    ax1.legend(fontsize=8)

    for role, series in metrics_db.get("crosstalk_db", {}).items():
        ax2.plot(lams, series, "-o", ms=3, label=f"ph {role}")
    seen_bounds = set()
    for r in (r for r in spec.references if r.quantity == "crosstalk"):
        # a shared crosstalk bound (same value on y-/y+) only needs one overlay
        key = (r.paper_value, bool(r.curve))
        if r.curve is None and key in seen_bounds:
            continue
        seen_bounds.add(key)
        _overlay_reference(ax2, r, band)
    ax2.set_xlabel("wavelength (µm)")
    ax2.set_ylabel("crosstalk (dB)")
    ax2.set_title("cross-port crosstalk")
    ax2.legend(fontsize=8)

    fig.suptitle(spec.name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_field_intensity_png(spec: PaperSpec, x, y, e2, path, *, wavelength_um=None) -> Optional[str]:
    """Save the in-plane ``|E|^2`` intensity heatmap (from a field-slice run) to
    ``path``. ``x``/``y`` are the µm coordinate arrays and ``e2`` the ``[y, x]``
    normalized intensity. Returns the path, or None if matplotlib is
    unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    lam = f"{wavelength_um:.3g} µm" if wavelength_um else f"{spec.optical.center_nm:g} nm"
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    pm = ax.pcolormesh(x, y, e2, shading="auto", cmap="inferno")
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"|E|² intensity ({lam}) — {spec.name}")
    fig.colorbar(pm, label="|E|² (normalized)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def build_markdown_report(
    spec: PaperSpec,
    metrics_db: dict,
    *,
    convergence=None,
    meta: Optional[dict] = None,
    figure_path: Optional[str] = None,
    geometry_path: Optional[str] = None,
    field_path: Optional[str] = None,
    provenance: Optional[dict] = None,
) -> str:
    """Assemble the per-device markdown replication report."""
    lams = metrics_db["wavelengths_um"]
    ic = _band_centre_index(lams, spec.optical.center_um)
    lines: List[str] = []
    src = spec.source
    lines.append(f"# Replication: {spec.name}\n")
    cite = src.citation
    if src.doi:
        cite += f" — doi:[{src.doi}](https://doi.org/{src.doi})"
    lines.append(f"**Paper:** {cite}\n")
    if src.matched_sim:
        lines.append(f"**Matched simulator reference:** `{src.matched_sim}`\n")

    lines.append("## Device\n")
    core = spec.stack.core
    lines.append(
        f"- kind: `{spec.device.kind}`  \n"
        f"- band: {spec.optical.band_nm[0]:g}–{spec.optical.band_nm[1]:g} nm "
        f"(centre {spec.optical.center_nm:g} nm), {spec.optical.polarization}"
        f"{spec.optical.mode_index}  \n"
        f"- stack: {core.thickness_um*1000:g} nm {core.material} core, "
        f"{spec.stack.clad_material} clad  \n"
    )
    params = ", ".join(f"{k}={v}" for k, v in spec.device.params.items())
    lines.append(f"- parameters: {params}\n")

    lines.append("## Measured vs paper (band centre)\n")
    lines.append("| quantity | port | units | paper | ph | Δ (ph−paper) | source |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for row in compare_rows(spec, metrics_db):
        pv = "—" if row.paper_value is None else f"{row.paper_value:.3g}"
        mv = "—" if row.measured is None else f"{row.measured:.3g}"
        dv = "—" if row.delta is None else f"{row.delta:+.3g}"
        lines.append(
            f"| {row.quantity} | `{row.port}` | {row.units} | {pv} | {mv} | {dv} | {row.source} |"
        )
    lines.append("")
    lines.append(
        "> Paper values digitized from the published figures/abstract are "
        "approximate; compare the band-centre level and trend, not per-point. "
        "Simulated (idealized) crosstalk is typically far below the measured "
        "value, which is fabrication/measurement-limited.\n"
    )

    lines.append("## Convergence\n")
    if convergence is not None:
        verdict = "✅ converged" if convergence.converged else "⚠️ NOT converged"
        lines.append(f"{verdict} ({convergence.stop_reason}).\n")
        lines.append("```")
        lines.append(convergence.summary())
        lines.append("```\n")
    else:
        lines.append("_single-resolution run; no convergence ladder._\n")

    lines.append("## Setup\n")
    if meta:
        lines.append(
            f"- grid: dl={meta.get('dl_um', float('nan')):.4f} µm "
            f"({meta.get('cells_per_wavelength', '?')} cells/λ in core), "
            f"subpixel `{meta.get('subpixel_method', '?')}`\n"
            f"- domain: {tuple(round(x, 2) for x in meta.get('size_um', ()))} µm, "
            f"n_core={meta.get('n_core', float('nan')):.4f}, "
            f"n_clad={meta.get('n_clad', float('nan')):.4f}, "
            f"n_eff(TE0)={meta.get('n_eff_TE0', float('nan')):.4f}\n"
        )
    if provenance:
        prov = ", ".join(f"{k}={v}" for k, v in provenance.items())
        lines.append(f"- provenance: {prov}\n")

    if geometry_path or field_path:
        lines.append("## Geometry & field\n")
        if geometry_path:
            lines.append(f"Permittivity ε(x, y):\n\n![geometry]({geometry_path})\n")
        if field_path:
            if spec.device.kind == "cosine_taper_crossing":
                field_caption = (
                    "light stays collimated through the junction; the dark "
                    "cross-arms are the low crosstalk"
                )
            elif spec.device.kind == "y_branch":
                field_caption = (
                    "the input mode expands through the solid junction and divides "
                    "between the two output arms"
                )
            elif spec.device.kind == "bragg_grating":
                field_caption = "the stopband field forms a standing wave and decays into the grating"
            else:
                field_caption = "normalized in-plane field intensity"
            lines.append(
                f"|E|² intensity — {field_caption}:\n\n"
                f"![field intensity]({field_path})\n"
            )

    if figure_path:
        lines.append("## Spectra (ph vs paper)\n")
        lines.append(f"![spectra]({figure_path})\n")

    return "\n".join(lines)
