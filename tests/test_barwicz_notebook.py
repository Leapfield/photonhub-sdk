"""Structural checks for the concise Barwicz microring tutorial bundle."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = (
    ROOT / "benchmarks" / "resonators" / "results" / "barwicz_microring"
)
NOTEBOOK = ARTIFACTS / "notebook.ipynb"
GENERATOR = ROOT / "examples" / "notebooks" / "_build_barwicz_microring.py"
SAVED_FIGURES = (
    "geometry.png",
    "source_mode_profile.png",
    "local_cell_vs_paper.png",
    "cmt_preview.png",
)


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def test_barwicz_notebook_is_short_portable_beginner_tutorial():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = "\n".join(_source(cell) for cell in cells)
    code_source = "\n".join(
        _source(cell) for cell in cells if cell["cell_type"] == "code"
    )

    assert 12 <= len(cells) <= 16
    assert len(code_source.splitlines()) < 140
    assert "04ox-barwiczpop-microrings-fab-analysis.pdf" in source
    assert "## Key physics" in source
    assert "## Simulation parameters" in source
    assert "## Create the simulation" in source
    assert "## Run the simulation" in source
    assert "## Source profile (whole-device field pending)" in source
    assert "## Key results" in source
    assert "## Convergence" not in source

    order = [
        source.index("## Key physics"),
        source.index("## Simulation parameters"),
        source.index("## Create the simulation"),
        source.index("## Run the simulation"),
        source.index("## Source profile (whole-device field pending)"),
        source.index("## Key results"),
    ]
    assert order == sorted(order)

    assert 'ARTIFACTS = Path(".").resolve()' in source
    assert "barwicz_full_filter" in source
    assert "build_full_filter" in source
    assert "cost_estimate" in source
    assert "RUN_SIMULATION = False" in source
    assert "run_local" in source and 'device="gpu"' in source
    assert "port_transmissions" in source
    assert "plot_field" in source
    assert "source_mode_profile.png" in source
    assert "local_cell_vs_paper.png" in source
    assert "calibrated to paper targets; it is not FDTD" in source
    assert "field_intensity.png" in source and "spectra.png" in source

    # Expert cache, bend, and ladder plumbing belongs in the runbook.
    assert "RUN_GPU_SMOKE" not in source
    assert "BARWICZ_FDTD_CACHE" not in source
    assert "solve_bend_modes" not in source
    assert "width_sensitivity" not in source
    assert "strongest_resonance" not in source

    serialized = json.dumps(notebook)
    assert "/Users/" not in serialized
    assert "/workspace/" not in serialized
    assert not any(
        output.get("output_type") == "error"
        for cell in cells
        for output in cell.get("outputs", [])
    )
    assert all(not cell.get("outputs") for cell in cells if cell["cell_type"] == "code")
    assert notebook["metadata"]["photonhub"] == {
        "artifact_bundle": ".",
        "device_kind": "barwicz_three_ring_add_drop",
        "gpu_default": False,
        "tutorial": True,
    }


def test_barwicz_notebook_generator_matches_merged_tutorial_contract():
    source = GENERATOR.read_text(encoding="utf-8")
    assert "/Users/" not in source
    assert "/workspace/" not in source
    assert "import nbformat" not in source
    assert "RUN_SIMULATION = False" in source
    assert '"tutorial": True' in source
    assert '"artifact_bundle": "."' in source
    assert "## Convergence" not in source


def test_barwicz_generator_matches_committed_notebook(tmp_path):
    spec = importlib.util.spec_from_file_location("build_barwicz_notebook", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generated_path = module.build_notebook(tmp_path / "notebook.ipynb")
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    committed = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert generated["cells"] == committed["cells"]
    assert generated["metadata"] == committed["metadata"]


def test_barwicz_saved_figures_are_real_png_files():
    for name in SAVED_FIGURES:
        path = ARTIFACTS / name
        assert path.is_file(), name
        assert path.stat().st_size > 10_000, name
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), name
