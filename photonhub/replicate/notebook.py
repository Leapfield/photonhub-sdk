"""Generate a concise, runnable tutorial for a paper-replication bundle.

The notebook follows a short, easy-to-understand sequence: key physics,
parameters, simulation setup, run, field plot, and the main paper results.  It
uses the Jupyter v4 JSON schema directly, so generation has no ``nbformat``
runtime dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

__all__ = ["generate_notebook"]


_SETUP = r'''
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Keep the notebook beside the replication artifacts so it remains portable.
ARTIFACTS = Path(".").resolve()
for name in ("spec.json", "metrics.json"):
    if not (ARTIFACTS / name).is_file():
        raise FileNotFoundError(f"Missing {name}; run this notebook from its artifact directory.")

# Import PhotonHub from the checkout, or use an installed photonhub package.
repo = ARTIFACTS
while repo != repo.parent and not (repo / "photonhub" / "photonhub").is_dir():
    repo = repo.parent
if (repo / "photonhub" / "photonhub").is_dir():
    sys.path.insert(0, str(repo / "photonhub"))

from photonhub.replicate import PaperSpec, compare_rows
from photonhub.replicate.build import build_simulation

spec = PaperSpec.from_dict(json.loads((ARTIFACTS / "spec.json").read_text()))
saved = json.loads((ARTIFACTS / "metrics.json").read_text())
saved_metrics = saved["metrics"]
meta = saved.get("meta", {})
conv_path = ARTIFACTS / "convergence.json"
convergence = json.loads(conv_path.read_text()) if conv_path.exists() else None
'''


_PARAMETERS = r'''
print(spec.source.citation)
if spec.source.doi:
    print("doi:", spec.source.doi)
print(
    f"\n{spec.optical.polarization}{spec.optical.mode_index}, "
    f"{spec.optical.band_nm[0]:g}-{spec.optical.band_nm[1]:g} nm "
    f"(center {spec.optical.center_nm:g} nm)"
)
print(f"core: {spec.stack.core.material}, {spec.stack.core.thickness_um:g} um")
print("\nGeometry parameters")
print(json.dumps(dict(spec.device.params), indent=2))
'''


_BUILD = r'''
# Use the resolution of the saved converged result.
CPW = int(round(float(meta.get("cells_per_wavelength", max(spec.convergence.ladder_cpw)))))
built = build_simulation(spec, cells_per_wavelength=CPW, field_slice=True)

print(f"grid: {CPW} cells/wavelength, dl={built.meta['dl_um']:.5f} um")
print("domain (um):", tuple(round(v, 3) for v in built.meta["size_um"]))
print("output monitors:", list(built.out_monitors))
'''


_GEOMETRY = r'''
z = built.meta["size_um"][2] / 2
ax = built.sim.plot_eps(z=z)
ax.set_title(f"{spec.name}: simulation geometry")
plt.show()
'''


_RUN = r'''
# Change to True when phsolver is available. Use "gpu" for a production run.
RUN_SIMULATION = False
DEVICE = "gpu"

data = None
if RUN_SIMULATION:
    data = built.run(device=DEVICE, quiet=False)
    metrics = built.metrics_db(data)
else:
    metrics = saved_metrics
    print("Using the saved converged result. Set RUN_SIMULATION=True to rerun FDTD.")
'''


_FIELD = r'''
if data is not None and built.field_intensity(data) is not None:
    x, y, intensity = built.field_intensity(data)
    plt.figure(figsize=(10, 3.5))
    plt.pcolormesh(x, y, intensity, shading="auto", cmap="inferno")
    plt.xlabel("x (um)"); plt.ylabel("y (um)")
    plt.colorbar(label="normalized |E|^2")
    plt.tight_layout(); plt.show()
else:
    field_path = ARTIFACTS / "field_intensity.png"
    if field_path.is_file():
        plt.figure(figsize=(10, 4))
        plt.imshow(plt.imread(field_path))
        plt.axis("off"); plt.tight_layout(); plt.show()
    else:
        print("No saved field image is available.")
'''


_COMPARE = r'''

print("\nPhotonHub vs paper at band center")
for row in compare_rows(spec, metrics):
    measured = "-" if row.measured is None else f"{row.measured:.5g}"
    paper = "-" if row.paper_value is None else f"{row.paper_value:.5g}"
    delta = "-" if row.delta is None else f"{row.delta:+.5g}"
    print(
        f"{row.quantity:>18}  {row.port:<8}  "
        f"paper={paper:>8}  PhotonHub={measured:>8}  delta={delta:>8} {row.units}"
    )
'''


_Y_RESULTS = r'''
lams = np.asarray(metrics["wavelengths_um"])
ic = int(np.argmin(np.abs(lams - spec.optical.center_um)))
top = np.asarray(metrics["transmission"]["through"])
bottom = np.asarray(metrics["transmission"][spec.ports.cross[0]])
psr = 10 * np.log10(top / bottom)
excess = -10 * np.log10(top + bottom)

print(f"lambda = {lams[ic]:.4f} um")
print(f"T_top = {top[ic]:.6f}, T_bottom = {bottom[ic]:.6f}")
print(f"PSR = {psr[ic]:+.4f} dB, excess loss = {excess[ic]:.4f} dB")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(lams, top, label=spec.ports.through)
ax[0].plot(lams, bottom, label=spec.ports.cross[0])
ax[0].set(xlabel="wavelength (um)", ylabel="modal transmission")
ax[0].legend(); ax[0].grid(alpha=0.2)
ax[1].plot(lams, psr, label="PSR")
ax[1].plot(lams, excess, label="excess loss")
ax[1].set(xlabel="wavelength (um)", ylabel="dB")
ax[1].legend(); ax[1].grid(alpha=0.2)
fig.tight_layout(); plt.show()
'''


_BRAGG_RESULTS = r'''
lams = np.asarray(metrics["wavelengths_um"])
reflection = np.asarray(metrics["reflection"])
transmission = np.asarray(metrics["transmission"]["through"])
peak = int(np.argmax(reflection))

print(f"peak reflection = {reflection[peak]:.5f} at {lams[peak]:.4f} um")
print(f"through transmission = {transmission[peak]:.5f}")

plt.figure(figsize=(6, 3.5))
plt.plot(lams, reflection, label="reflection")
plt.plot(lams, transmission, label="transmission")
plt.xlabel("wavelength (um)"); plt.ylabel("modal power")
plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.show()
'''


_PORT_RESULTS = r'''
lams = np.asarray(metrics["wavelengths_um"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
if "insertion_loss_db" in metrics:
    ax[0].plot(lams, metrics["insertion_loss_db"])
ax[0].set(xlabel="wavelength (um)", ylabel="insertion loss (dB)")
for role, values in metrics.get("crosstalk_db", {}).items():
    ax[1].plot(lams, values, label=role)
ax[1].set(xlabel="wavelength (um)", ylabel="crosstalk (dB)")
if metrics.get("crosstalk_db"):
    ax[1].legend()
for a in ax:
    a.grid(alpha=0.2)
fig.tight_layout(); plt.show()
'''


_Y_SWEEP = r'''
sweep_path = ARTIFACTS / "psr_sweep.json"
if sweep_path.is_file():
    sweep = json.loads(sweep_path.read_text())
    r1 = [row["r1"] for row in sweep]
    paper = [row["PSR_paper"] for row in sweep]
    photonhub = [row["PSR_ph"] for row in sweep]
    error = np.asarray(photonhub) - np.asarray(paper)

    plt.figure(figsize=(6, 3.5))
    plt.plot(r1, paper, "k--o", label="paper Fig. 3(a)")
    plt.plot(r1, photonhub, "-o", label="PhotonHub")
    plt.xlabel("r1 (um)"); plt.ylabel("PSR (dB)")
    plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.show()
    print(f"maximum |PSR error| = {np.max(np.abs(error)):.3f} dB")
else:
    print("No psr_sweep.json was saved for this bundle.")
'''


_CONVERGENCE = r'''
if convergence is None:
    print("Single-resolution result; no convergence ladder was saved.")
else:
    print(
        "converged =", convergence["converged"],
        "| last drift =", f"{convergence['drift_successive']:.6g}",
        "| tolerance =", convergence["tol"],
    )
    cpw = [row["cells_per_wavelength"] for row in convergence["ladder"]]
    values = [row["metric"] for row in convergence["ladder"]]
    for row in convergence["ladder"]:
        print(
            f"{row['cells_per_wavelength']:g} cells/lambda: "
            f"metric={row['metric']:.8f}, dl={row['dl_um']:.5f} um"
        )
    plt.figure(figsize=(5, 3.2))
    plt.plot(cpw, values, "-o")
    plt.xlabel("cells per wavelength"); plt.ylabel(convergence["metric_name"])
    plt.grid(alpha=0.2); plt.tight_layout(); plt.show()
'''


def _physics(kind: str) -> str:
    if kind == "y_branch":
        return r"""## Key physics

The input TE0 mode expands into the widened junction and divides between two
output modes. In the symmetric limit the powers are equal; changing the upper
S-curve radius `r1` controls
\(\mathrm{PSR}=10\log_{10}(T_\mathrm{top}/T_\mathrm{bottom})\).
Smooth boundaries keep \(-10\log_{10}(T_\mathrm{top}+T_\mathrm{bottom})\) small.

A bright spot at the branch point can depend on local grid sampling. The paper
result is determined from integrated modal power and its convergence, not the
brightest field pixel."""
    if kind == "bragg_grating":
        return r"""## Key physics

Each sidewall corrugation weakly couples the forward mode to the backward mode.
Near \(\lambda_B\approx2n_\mathrm{eff}\Lambda\), reflections from successive
periods add coherently, producing a reflection stopband and transmission notch."""
    if kind == "cosine_taper_crossing":
        return """## Key physics

The cosine tapers expand and collimate the guided mode through the crossing.
The key observables are through-port insertion loss and cross-port crosstalk."""
    return """## Key physics

The field plot explains how the device operates; the integrated paper metric
and its grid convergence determine whether the result is reproduced."""


def _result_code(kind: str) -> str:
    if kind == "y_branch":
        return _Y_RESULTS + _COMPARE
    if kind == "bragg_grating":
        return _BRAGG_RESULTS + _COMPARE
    return _PORT_RESULTS + _COMPARE


def _cell(cell_type: str, source: str, cell_id: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": source.strip("\n"),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _load_spec(outdir: Path) -> Mapping[str, Any]:
    path = outdir / "spec.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is required to generate the tutorial")
    data = json.loads(path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return data


def generate_notebook(outdir: "str | Path") -> Path:
    """Write a concise, portable ``notebook.ipynb`` into ``outdir``."""

    outdir = Path(outdir).resolve()
    data = _load_spec(outdir)
    name = str(data.get("name", outdir.name))
    source = data.get("source", {})
    citation = str(source.get("citation", "paper citation unavailable"))
    doi = str(source.get("doi", ""))
    kind = str(data.get("device", {}).get("kind", "unknown"))
    doi_link = f" — [doi:{doi}](https://doi.org/{doi})" if doi else ""

    cells = [
        _cell(
            "markdown",
            f"""# Reproducing `{name}` with PhotonHub

{citation}{doi_link}

This example builds the paper geometry, runs an FDTD simulation, and compares
the main result with the published value. A saved converged run is loaded by
default so the notebook is quick to execute.""",
            "intro",
        ),
        _cell("markdown", _physics(kind), "physics"),
        _cell("markdown", "## Setup", "setup-heading"),
        _cell("code", _SETUP, "setup"),
        _cell("markdown", "## Simulation parameters", "parameters-heading"),
        _cell("code", _PARAMETERS, "parameters"),
        _cell(
            "markdown",
            """## Create the simulation

`build_simulation` creates the geometry, material stack, grid, mode source,
absorbing boundaries, and mode monitors from the paper specification.""",
            "create-heading",
        ),
        _cell("code", _BUILD, "build"),
        _cell("code", _GEOMETRY, "geometry"),
        _cell(
            "markdown",
            """## Run the simulation

Leave `RUN_SIMULATION=False` to use the saved converged result, or enable it on
a machine with the PhotonHub solver.""",
            "run-heading",
        ),
        _cell("code", _RUN, "run"),
        _cell("markdown", "## Field plot", "field-heading"),
        _cell("code", _FIELD, "field"),
        _cell("markdown", "## Results", "results-heading"),
        _cell("code", _result_code(kind), "results"),
    ]

    if kind == "y_branch":
        cells.extend([
            _cell(
                "markdown",
                """## Arbitrary splitting ratio

The paper's main result varies `r1` to tune the output power ratio. The saved
sweep below compares PhotonHub with the digitized paper curve.""",
                "sweep-heading",
            ),
            _cell("code", _Y_SWEEP, "sweep"),
        ])

    cells.extend([
        _cell("markdown", "## Convergence", "convergence-heading"),
        _cell("code", _CONVERGENCE, "convergence"),
        _cell(
            "markdown",
            """The field profile gives the physical picture; the modal result
and its resolution convergence provide the quantitative paper comparison.""",
            "summary",
        ),
    ])

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "photonhub": {
                "artifact_bundle": ".",
                "device_kind": kind,
                "tutorial": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path = outdir / "notebook.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    return path
