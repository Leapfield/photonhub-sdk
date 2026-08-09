"""Tests for the comparison report (synthetic metrics, no engine)."""

from pathlib import Path

import pytest

from photonhub.replicate import PaperSpec, build_markdown_report, compare_rows
from photonhub.replicate.convergence import ConvergenceReport, Rung

_REPO = Path(__file__).resolve().parents[2]
_SPEC = _REPO / "benchmarks" / "replicate" / "specs" / "chandran_cosine_crossing.yaml"


def _synthetic_metrics():
    lams = [1.26, 1.31, 1.36]
    return {
        "wavelengths_um": lams,
        "insertion_loss_db": [0.30, 0.24, 0.31],
        "transmission": {"through": [0.933, 0.946, 0.931], "y-": [1e-6, 2e-6, 1.5e-6], "y+": [1.1e-6, 2.1e-6, 1.6e-6]},
        "crosstalk_db": {"y-": [-60.0, -57.0, -58.0], "y+": [-59.6, -56.8, -57.9]},
    }


def test_compare_rows_pull_band_centre():
    spec = PaperSpec.from_yaml(_SPEC)
    rows = compare_rows(spec, _synthetic_metrics())
    by = {(r.quantity, r.port): r for r in rows}
    il = by[("insertion_loss", "x+")]
    assert il.units == "dB"
    assert il.paper_value == pytest.approx(0.216)
    assert il.measured == pytest.approx(0.24)   # band centre (1.31)
    assert il.delta == pytest.approx(0.024, abs=1e-9)
    xt = by[("crosstalk", "y-")]
    assert xt.measured == pytest.approx(-57.0)   # band centre
    assert xt.paper_value == pytest.approx(-30.0)


def test_markdown_report_contains_key_sections():
    spec = PaperSpec.from_yaml(_SPEC)
    conv = ConvergenceReport(
        converged=True, metric_name="through_transmission",
        ladder=(Rung(15, 0.025, 0.941, 1.0), Rung(20, 0.019, 0.946, 2.0)),
        tol=0.003, patience=2, stop_reason="tol", total_cost_usd=3.0,
    )
    md = build_markdown_report(
        spec, _synthetic_metrics(), convergence=conv,
        meta={"dl_um": 0.019, "cells_per_wavelength": 20, "n_core": 3.5,
              "n_clad": 1.447, "n_eff_TE0": 2.28, "size_um": (7.6, 7.6, 2.4),
              "subpixel_method": "contour_diag"},
        provenance={"git_sha": "abc123", "device": "cpu"},
    )
    assert "# Replication: chandran_cosine_crossing" in md
    assert "doi:" in md
    assert "Measured vs paper" in md
    assert "| insertion_loss |" in md
    assert "converged" in md
    assert "git_sha=abc123" in md


def test_report_renders_all_three_figures():
    spec = PaperSpec.from_yaml(_SPEC)
    md = build_markdown_report(spec, _synthetic_metrics(),
                               figure_path="spectra.png", geometry_path="geometry.png",
                               field_path="field_intensity.png")
    assert "## Geometry & field" in md
    assert "![geometry](geometry.png)" in md
    assert "![field intensity](field_intensity.png)" in md
    assert "## Spectra (ph vs paper)" in md
    assert "![spectra](spectra.png)" in md


def test_yjunction_report_describes_a_split_not_cross_arms():
    spec = PaperSpec.from_yaml(
        _REPO / "benchmarks" / "replicate" / "specs" / "yjunction_lin_2019.yaml"
    )
    metrics = {
        "wavelengths_um": [1.55],
        "transmission": {"through": [0.48], "o_bot": [0.48]},
        "insertion_loss_db": [3.18],
        "crosstalk_db": {"o_bot": [-3.18]},
    }
    md = build_markdown_report(spec, metrics, field_path="field_intensity.png")
    assert "divides between the two output arms" in md
    assert "dark cross-arms" not in md


def test_yjunction_spectrum_uses_split_outputs_even_with_reflection_diagnostic(
    tmp_path, monkeypatch
):
    """Backward input power is diagnostic on a splitter, not its primary plot."""
    from matplotlib.axes import Axes
    from photonhub.replicate.report import plot_spectrum_png

    spec = PaperSpec.from_yaml(
        _REPO / "benchmarks" / "replicate" / "specs" / "yjunction_lin_2019.yaml"
    )
    metrics = {
        "wavelengths_um": [1.50, 1.55, 1.60],
        "transmission": {
            "through": [0.48, 0.486, 0.49],
            "o_bot": [0.48, 0.485, 0.49],
        },
        "insertion_loss_db": [3.19, 3.13, 3.10],
        "crosstalk_db": {"o_bot": [-3.19, -3.14, -3.10]},
        "reflection": [5e-4, 6e-4, 5e-4],
    }
    titles = []
    original = Axes.set_title

    def record_title(self, label, *args, **kwargs):
        titles.append(label)
        return original(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", record_title)
    out = plot_spectrum_png(spec, metrics, tmp_path / "split.png")
    assert out is None or (tmp_path / "split.png").is_file()
    if out is not None:
        assert "power in the two output modes" in titles
        assert "back-reflection (stopband)" not in titles


def test_plot_field_intensity_smoke(tmp_path):
    import numpy as np
    from photonhub.replicate.report import plot_field_intensity_png
    spec = PaperSpec.from_yaml(_SPEC)
    x = np.linspace(0, 5, 20); y = np.linspace(0, 5, 18)
    e2 = np.random.rand(18, 20)
    out = plot_field_intensity_png(spec, x, y, e2, tmp_path / "f.png", wavelength_um=1.31)
    assert out is None or (tmp_path / "f.png").is_file()


def test_markdown_report_handles_no_convergence():
    spec = PaperSpec.from_yaml(_SPEC)
    md = build_markdown_report(spec, _synthetic_metrics(), convergence=None)
    assert "single-resolution run" in md


def test_reference_carries_flatness_and_bound():
    spec = PaperSpec.from_yaml(_SPEC)
    il = next(r for r in spec.references if r.quantity == "insertion_loss")
    assert il.flatness == pytest.approx(0.032)
    xt = next(r for r in spec.references if r.quantity == "crosstalk")
    assert xt.bound is True


def test_reference_curve_parses():
    from photonhub.replicate.spec import Reference
    r = Reference.from_dict({
        "quantity": "insertion_loss", "units": "dB", "port": "x+",
        "curve": [[1260, 0.20], [1310, 0.22], [1360, 0.21]], "label": "tidy3d",
    })
    assert r.curve == ((1260.0, 0.20), (1310.0, 0.22), (1360.0, 0.21))
    assert r.label == "tidy3d"


def test_spectrum_png_with_band_and_curve(tmp_path):
    # smoke: the overlay path (band + bound + curve) renders without error
    from photonhub.replicate.report import plot_spectrum_png
    from dataclasses import replace
    from photonhub.replicate.spec import Reference
    spec = PaperSpec.from_yaml(_SPEC)
    curve_ref = Reference(quantity="insertion_loss", units="dB", port="x+",
                          curve=((1260, 0.2), (1310, 0.22), (1360, 0.21)), label="tidy3d")
    spec = replace(spec, references=(*spec.references, curve_ref))
    out = plot_spectrum_png(spec, _synthetic_metrics(), tmp_path / "s.png")
    # matplotlib may be unavailable in some envs -> None is acceptable
    assert out is None or (tmp_path / "s.png").is_file()
