"""The replication workflow saves a valid, runnable tutorial notebook over a
(possibly minimal) artifacts bundle. Executing the solver cell is opt-in, so
here we validate structure and tutorial content without running FDTD."""

import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

from photonhub.replicate.notebook import generate_notebook


def _minimal_bundle(d: Path):
    (d / "spec.json").write_text(json.dumps({
        "name": "demo",
        "source": {"citation": "c", "doi": "", "arxiv": "", "matched_sim": ""},
        "device": {"kind": "cosine_taper_crossing", "params": {}},
        "stack": {"layers": [{"name": "core", "material": "cSi", "zmin_um": 0.0, "thickness_um": 0.161}], "clad_material": "SiO2", "box_material": "SiO2"},
        "optical": {"band_nm": [1260, 1360], "center_nm": 1310, "n_points": 3, "polarization": "TE", "mode_index": 0},
        "ports": {"input": "x-", "through": "x+", "cross": ["y-", "y+"], "reflect": None},
        "references": [],
        "convergence": {"ladder_cpw": [10], "subpixel_method": "contour_diag", "tol_pp": 0.3},
    }))
    (d / "metrics.json").write_text(json.dumps({
        "metrics": {"wavelengths_um": [1.31], "insertion_loss_db": [0.3], "transmission": {"through": [0.93]}, "crosstalk_db": {}},
        "meta": {"dl_um": 0.025, "cells_per_wavelength": 10},
    }))


def test_generate_notebook_is_valid(tmp_path):
    _minimal_bundle(tmp_path)
    path = generate_notebook(tmp_path)
    assert path.name == "notebook.ipynb"
    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)  # raises if malformed
    # It is a portable tutorial, not an absolute-path artifact viewer.
    kinds = {c.cell_type for c in nb.cells}
    assert {"markdown", "code"} <= kinds
    src = "\n".join(c.source for c in nb.cells)
    assert "Reproducing `demo` with PhotonHub" in src
    assert "Key physics" in src
    assert 'ARTIFACTS = Path(".").resolve()' in src
    assert str(tmp_path) not in src
    assert "build_simulation(spec" in src
    assert "RUN_SIMULATION = False" in src
    assert "compare_rows" in src
    assert "plot_eps" in src
    assert "## Convergence" in src
    assert "Learning objectives" not in src
    assert "hashlib" not in src
    assert len(nb.cells) <= 18
    assert nb.metadata["photonhub"]["tutorial"] is True


def test_yjunction_notebook_teaches_psr_and_hotspot_interpretation(tmp_path):
    _minimal_bundle(tmp_path)
    spec = json.loads((tmp_path / "spec.json").read_text())
    spec["name"] = "lin_shi_demo"
    spec["device"] = {"kind": "y_branch", "params": {"r_top1_um": 1.7}}
    (tmp_path / "spec.json").write_text(json.dumps(spec))

    path = generate_notebook(tmp_path)
    nb = nbformat.read(str(path), as_version=4)
    src = "\n".join(c.source for c in nb.cells)
    assert "input TE0 mode expands" in src
    assert "\\mathrm{PSR}=10\\log_{10}" in src
    assert "bright spot at the branch point" in src
    assert "integrated modal power" in src
    assert "psr_sweep.json" in src
    assert "RUN_PSR_POINT" not in src
    assert len(nb.cells) <= 20
