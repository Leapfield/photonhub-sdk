"""End-to-end replication integration: build -> real solver run -> metrics ->
artifacts bundle. Skipped unless a solver binary is discoverable AND
PHOTONHUB_RUN_INTEGRATION=1 (it does a real FDTD run — seconds, not milliseconds).

    PHOTONHUB_SOLVER=/path/to/phsolver PHOTONHUB_RUN_INTEGRATION=1 pytest -k integration
"""

import math
import os
from pathlib import Path

import pytest

from photonhub.runners import find_solver

_REPO = Path(__file__).resolve().parents[2]
_SPEC = _REPO / "benchmarks" / "replicate" / "specs" / "chandran_cosine_crossing.yaml"

_run_it = os.environ.get("PHOTONHUB_RUN_INTEGRATION") == "1" and find_solver() is not None
pytestmark = pytest.mark.skipif(
    not _run_it, reason="set PHOTONHUB_RUN_INTEGRATION=1 and a solver to run"
)


def test_single_run_metrics_are_physical(tmp_path):
    from photonhub.replicate import PaperSpec
    from photonhub.replicate.build import build_simulation

    spec = PaperSpec.from_yaml(_SPEC)
    built = build_simulation(spec, cells_per_wavelength=10, run_periods=6, shutoff=1e-3)
    data = built.run(device="cpu", quiet=True)
    m = built.metrics_db(data)
    lams = m["wavelengths_um"]
    ic = min(range(len(lams)), key=lambda i: abs(lams[i] - 1.31))
    through = m["transmission"]["through"][ic]
    # a low-loss crossing: most power goes through; crosstalk is far below it
    assert 0.5 < through <= 1.05
    for role in ("y-", "y+"):
        assert m["transmission"][role][ic] < 0.05
    assert math.isfinite(m["insertion_loss_db"][ic])


def test_replicate_writes_full_bundle(tmp_path):
    from dataclasses import replace

    from photonhub.replicate import PaperSpec, replicate

    spec = PaperSpec.from_yaml(_SPEC)
    # single coarse resolution, fast
    spec = replace(spec, convergence=replace(spec.convergence, ladder_cpw=(10,)))
    res = replicate(
        spec, outdir=tmp_path / "chandran", device="cpu", converge=False,
        cells_per_wavelength=10, build_kwargs={"run_periods": 6, "shutoff": 1e-3},
    )
    out = Path(res.outdir)
    for name in (
        "sim.json", "metrics.json", "layout.gds", "report.md", "spec.json",
        "notebook.ipynb",
    ):
        assert (out / name).is_file(), name
    assert Path(res.artifacts["notebook"]) == out / "notebook.ipynb"
    assert "# Replication:" in res.report_md
    # the exported GDS re-imports
    from photonhub.gds import import_gds, read_gds_cell_names
    assert read_gds_cell_names(out / "layout.gds")
