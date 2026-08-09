"""``replicate()`` — the linear orchestrator that takes a paper (a
:class:`PaperSpec`) to a reproduced design and a comparison bundle.

    spec -> build -> (converge over a dl ladder) -> extract metrics
         -> {sim.json, layout.gds, metrics.json, convergence.json, report.md,
             notebook.ipynb, figures}

It ties together the intake, geometry regeneration, spec->simulation build, the
convergence gate, the figure-of-merit extraction, GDS export, and the comparison
report into one call that writes a self-contained artifacts directory.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..gds import export_gds
from .build import BuiltSim, build_simulation
from .convergence import ConvergenceReport, converge_through_transmission
from .geometry import build_geometry
from .report import build_markdown_report, plot_spectrum_png
from .spec import PaperSpec

__all__ = ["ReplicationResult", "replicate"]


@dataclass
class ReplicationResult:
    """What :func:`replicate` produced (also written to ``outdir``)."""

    spec: PaperSpec
    outdir: Path
    built: BuiltSim
    metrics_db: dict
    convergence: Optional[ConvergenceReport]
    report_md: str
    artifacts: dict = field(default_factory=dict)


def _default_runner(device: Optional[str]) -> Callable:
    from ..runners.local import run_local

    def run(sim):
        return run_local(sim, device=device, quiet=True)

    return run


def replicate(
    spec: "PaperSpec | str | Path",
    *,
    outdir: "str | Path",
    run: Optional[Callable] = None,
    device: Optional[str] = None,
    converge: bool = True,
    cells_per_wavelength: Optional[float] = None,
    build_kwargs: Optional[dict] = None,
    budget_usd: Optional[float] = None,
    write_gds: bool = True,
    plot: bool = True,
    save_notebook: bool = True,
    provenance: Optional[dict] = None,
) -> ReplicationResult:
    """Reproduce ``spec`` and write the artifacts bundle to ``outdir``.

    ``run`` is the run function applied to each ``Simulation`` (defaults to
    :func:`photonhub.run_local` on ``device``). With ``converge=True`` a resolution
    ladder is run (the finest rung's data is the reported result and the
    convergence report is stamped in); with ``converge=False`` a single run at
    ``cells_per_wavelength`` (or the ladder's finest) is done.

    With ``save_notebook=True`` (the default), the bundle also contains a
    runnable ``notebook.ipynb`` tutorial. It teaches the device physics,
    rebuilds the PhotonHub setup from ``spec.json``, analyzes the saved results,
    and provides an opt-in solver rerun cell.
    """
    if not isinstance(spec, PaperSpec):
        spec = PaperSpec.from_yaml(spec)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runner = run if run is not None else _default_runner(device)
    bkw = dict(build_kwargs or {})

    # Capture each rung's (built, data) so the finest gives full metrics without
    # a redundant extra run.
    captured: dict = {}

    # A reflective device (a Bragg/DBR stopband) must converge on its band-centre
    # REFLECTION — through-transmission there is a near-zero notch (ill-conditioned
    # as a convergence metric). Everything else converges on through-transmission.
    reflective = any(r.quantity == "reflection" for r in spec.references)
    metric_name = "peak_reflection" if reflective else "through_transmission"

    def make_run(cpw):
        # capture the xy field slice so the reported (finest) run yields the
        # |E|^2 intensity figure without a second run
        built = build_simulation(spec, cells_per_wavelength=cpw, field_slice=True, **bkw)
        captured[cpw] = {"built": built}

        def extract(data):
            captured[cpw]["data"] = data
            if reflective:
                # PEAK reflection over the band — robust to the stopband λ drifting
                # a few nm between resolutions (converging on a fixed-λ point would
                # then wander off the peak).
                refl = built.reflection(data)
                return max(refl.values()) if refl else 0.0
            ic = min(range(len(built.freqs_hz)),
                     key=lambda i: abs(built.wavelengths_um[i] - spec.optical.center_um))
            return built.transmissions(data)["through"][built.freqs_hz[ic]]

        return built.sim, extract

    convergence: Optional[ConvergenceReport] = None
    if converge:
        from .convergence import auto_converge

        convergence = auto_converge(
            make_run, runner,
            ladder=spec.convergence.ladder_cpw,
            tol=spec.convergence.tol_pp / 100.0,
            metric_name=metric_name,
            budget_usd=budget_usd,
        )
        finest_cpw = convergence.ladder[-1].cells_per_wavelength
    else:
        finest_cpw = cells_per_wavelength or max(spec.convergence.ladder_cpw)
        sim, extract = make_run(finest_cpw)
        extract(runner(sim))

    built = captured[finest_cpw]["built"]
    data = captured[finest_cpw]["data"]
    metrics_db = built.metrics_db(data)

    # --- write artifacts ---
    artifacts: dict = {}

    sim_json = outdir / "sim.json"
    sim_json.write_text(built.sim.to_wire_json())
    artifacts["sim_json"] = str(sim_json)

    metrics_json = outdir / "metrics.json"
    ic = min(range(len(built.freqs_hz)),
             key=lambda i: abs(built.wavelengths_um[i] - spec.optical.center_um))
    metrics_json.write_text(json.dumps({
        "band_centre_um": spec.optical.center_um,
        "band_centre_index": ic,
        "metrics": metrics_db,
        "meta": {k: (list(v) if isinstance(v, tuple) else v) for k, v in built.meta.items()},
    }, indent=2))
    artifacts["metrics_json"] = str(metrics_json)

    # Write the machine-readable paper record before the tutorial: notebook
    # generation reads this portable JSON form and must be a guaranteed workflow
    # artifact, not a best-effort side effect hidden behind an optional package.
    spec_out = outdir / "spec.json"
    spec_out.write_text(json.dumps(_spec_to_dict(spec), indent=2))
    artifacts["spec_json"] = str(spec_out)

    if convergence is not None:
        conv_json = outdir / "convergence.json"
        conv_json.write_text(json.dumps(convergence.to_dict(), indent=2))
        artifacts["convergence_json"] = str(conv_json)

    if write_gds:
        gds_path = outdir / "layout.gds"
        # export the design shape at the origin (original spec parameters)
        from ..components.structures import Medium
        geom = build_geometry(
            spec.device.kind, spec.device.params,
            medium=Medium(permittivity=built.meta["n_core"] ** 2),
            thickness_um=spec.stack.core.thickness_um,
        )
        export_gds(geom, gds_path)
        artifacts["gds"] = str(gds_path)

    figure_rel = None
    geometry_rel = None
    field_rel = None
    if plot:
        # (1) ph-vs-paper transmission comparison
        fig = plot_spectrum_png(spec, metrics_db, outdir / "spectra.png")
        if fig:
            figure_rel = "spectra.png"
            artifacts["spectra_png"] = fig
        # (2) permittivity cross-section
        if _geometry_png(built, outdir / "geometry.png", artifacts):
            geometry_rel = "geometry.png"
        # (3) |E|^2 intensity from the field slice captured in the reported run
        fi = built.field_intensity(data)
        if fi is not None:
            from .report import plot_field_intensity_png

            lam = None
            cf = built.meta.get("field_slice_freq_hz")
            if cf:
                lam = 2.99792458e8 / cf * 1e6
            out = plot_field_intensity_png(spec, *fi, outdir / "field_intensity.png",
                                           wavelength_um=lam)
            if out:
                field_rel = "field_intensity.png"
                artifacts["field_intensity_png"] = out

    report_md = build_markdown_report(
        spec, metrics_db, convergence=convergence, meta=built.meta,
        figure_path=figure_rel, geometry_path=geometry_rel, field_path=field_rel,
        provenance=provenance,
    )
    report_path = outdir / "report.md"
    report_path.write_text(report_md)
    artifacts["report_md"] = str(report_path)

    # A runnable physics + reproduction tutorial over the complete bundle. The
    # expensive solver cell is opt-in, so executing the tutorial still uses the
    # already-converged evidence by default.
    if save_notebook:
        from .notebook import generate_notebook

        artifacts["notebook"] = str(generate_notebook(outdir))

    return ReplicationResult(
        spec=spec, outdir=outdir, built=built, metrics_db=metrics_db,
        convergence=convergence, report_md=report_md, artifacts=artifacts,
    )


def _geometry_png(built: BuiltSim, path, artifacts: dict):
    """Top-down permittivity render of the built scene, if matplotlib is
    available. Returns the path on success, else None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        cz = built.meta["size_um"][2] / 2.0
        ax = built.sim.plot_eps(z=cz)
        ax.set_title(f"Permittivity ε(x, y) — {built.meta.get('subpixel_method', '')} (z = core centre)")
        ax.figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
        artifacts["geometry_png"] = str(path)
        return str(path)
    except Exception:
        return None


def _spec_to_dict(spec: PaperSpec) -> dict:
    from dataclasses import asdict

    return asdict(spec)
