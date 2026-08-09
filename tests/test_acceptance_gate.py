"""Acceptance accuracy-gate unit tests (benchmarks/acceptance/run_acceptance.py).

The numeric pass/fail comparator ``compare_to_reference`` is pure python/numpy
(no engine), so it is unit-tested HERE in the standard suite. It is the
acceptance harness's accuracy gate: with ``--reference`` + ``--tolerance`` a
device-accuracy regression makes the run exit non-zero instead of being a
silent visual overlay. These tests pin:

  * an in-tolerance run passes; an out-of-tolerance run fails;
  * the reference is interpolated onto the current wavelength grid (mismatched
    sampling is fine);
  * ports / wavelengths absent from the reference are SKIPPED, not failed
    (a partial reference still gates the ports it covers);
  * the gate is opt-in — without a reference there is nothing to gate.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_RA_PATH = REPO_ROOT / "benchmarks" / "acceptance" / "run_acceptance.py"


def _load_run_acceptance():
    if not _RA_PATH.is_file():
        pytest.skip(f"run_acceptance.py not found at {_RA_PATH}")
    # Make the acceptance dir importable (it imports photonhub.* at module load).
    sys.path.insert(0, str(_RA_PATH.parent))
    spec = importlib.util.spec_from_file_location("run_acceptance", _RA_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass introspection (which reads
    # sys.modules[cls.__module__]) resolves during load.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ra = _load_run_acceptance()


def _result(name, label, lam_um, T):
    """A run-result dict shaped like run_scene() returns."""
    return {"name": name, "outputs": {label: (list(lam_um), list(T))}}


def _reference(name, label, lam_um, T):
    """A reference dict shaped like to_json() writes (lam in nm)."""
    return {"components": {name: {"outputs": {
        label: {"lam_nm": [l * 1000 for l in lam_um], "T": list(T)}}}}}


def test_in_tolerance_passes():
    lam = [1.30, 1.31, 1.32]
    results = [_result("straight", "out", lam, [0.99, 0.98, 0.99])]
    ref = _reference("straight", "out", lam, [0.991, 0.979, 0.992])
    passed, rows = ra.compare_to_reference(results, ref, tolerance=0.01)
    assert passed
    row = next(r for r in rows if r.get("port") == "out")
    assert row["pass"] and row["max_abs_err"] <= 0.01


def test_out_of_tolerance_fails():
    lam = [1.30, 1.31, 1.32]
    results = [_result("straight", "out", lam, [0.99, 0.50, 0.99])]  # big dip
    ref = _reference("straight", "out", lam, [0.99, 0.98, 0.99])
    passed, rows = ra.compare_to_reference(results, ref, tolerance=0.02)
    assert not passed
    row = next(r for r in rows if r.get("port") == "out")
    assert not row["pass"] and row["max_abs_err"] > 0.02


def test_reference_interpolated_onto_current_grid():
    # Reference sampled COARSER than the current grid; comparator interpolates.
    cur_lam = [1.30, 1.305, 1.31, 1.315, 1.32]
    results = [_result("taper", "out", cur_lam, [0.80, 0.85, 0.90, 0.85, 0.80])]
    ref_lam = [1.30, 1.32]
    ref = _reference("taper", "out", ref_lam, [0.80, 0.80])  # linear ref 0.80
    passed, rows = ra.compare_to_reference(results, ref, tolerance=0.02)
    # Mid-band current rises to 0.90 vs interpolated ref 0.80 -> |Δ|=0.10 > tol.
    assert not passed
    row = next(r for r in rows if r.get("port") == "out")
    assert row["n_points"] == len(cur_lam)
    assert row["max_abs_err"] == pytest.approx(0.10, abs=1e-9)


def test_missing_port_is_skipped_not_failed():
    lam = [1.30, 1.31, 1.32]
    results = [_result("crossing", "cross", lam, [0.01, 0.01, 0.01])]
    # Reference only covers "through", not "cross".
    ref = _reference("crossing", "through", lam, [0.95, 0.95, 0.95])
    passed, rows = ra.compare_to_reference(results, ref, tolerance=0.001)
    assert passed  # nothing comparable -> not a failure
    row = next(r for r in rows if r.get("port") == "cross")
    assert "no reference" in row["status"]
    assert "max_abs_err" not in row


def test_non_overlapping_wavelengths_skipped():
    results = [_result("ring", "through", [1.50, 1.51], [0.9, 0.9])]
    ref = _reference("ring", "through", [1.30, 1.31], [0.9, 0.9])  # disjoint band
    passed, rows = ra.compare_to_reference(results, ref, tolerance=0.001)
    assert passed
    row = next(r for r in rows if r.get("port") == "through")
    assert "no overlapping wavelengths" in row["status"]


def test_partial_reference_gates_covered_ports():
    lam = [1.30, 1.31, 1.32]
    results = [
        _result("crossing", "through", lam, [0.5, 0.5, 0.5]),  # FAILS vs 0.95
    ]
    results[0]["outputs"]["cross"] = (lam, [0.01, 0.01, 0.01])  # no ref -> skip
    ref = _reference("crossing", "through", lam, [0.95, 0.95, 0.95])
    passed, rows = ra.compare_to_reference(results, ref, tolerance=0.02)
    assert not passed  # the covered "through" port is out of tolerance
    thru = next(r for r in rows if r.get("port") == "through")
    cross = next(r for r in rows if r.get("port") == "cross")
    assert not thru["pass"]
    assert "no reference" in cross["status"]
