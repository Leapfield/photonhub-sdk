---
name: photonhub
description: Use this skill when building, checking, running, or reading PhotonHub FDTD simulations through its Python SDK and local MCP tools.
---

# PhotonHub simulations

PhotonHub is 3D FDTD for photonics.
The Python SDK builds scenes and reads results.
The MCP server checks wire specs, estimates runs, runs locally, and gives compact result views.
Data and compute stay on this machine when using the MCP tools.

## Author in Python

Write a small Python script with `import photonhub as ph`.
Construct a `ph.Simulation` and keep the script as the source of the scene.
Save `sim.to_wire_json()` to a `.json` file for the MCP tools.
Do not hand-write a large wire JSON document.
Do not change the spec to hide an issue.

## Check before running

1. Call `capabilities` first.
2. Read `solver.found`, `devices`, `cpu_only_features`, and `safety`.
3. Use `get_schema` when a wire field or definition is unclear.
4. Call `validate_spec` with the wire JSON string or file path.
5. Fix issues with `error` severity in the Python scene.
6. Save a new `sim.to_wire_json()` and call `validate_spec` again.
7. Continue until `ok` is true.
8. Check `authoritative`: without the solver, the engine stage cannot run.
9. Call `estimate` before any run.

A validation issue has `id`, `severity`, `path`, `monitor`, `message`, `fix`,
and `detail`, which is null when empty.
Use `fix` as the next action.
`path` points into the wire document, such as `sources[0].center_um`.
In validation, an `error` makes `ok` false; correct it before running.
Warnings and info are advice.
The stable issue catalog is the `photonhub://error-codes` resource.
If a tool fails, its error text ends with JSON containing `id`, `severity`,
`message`, and `fix`.
`validate_spec` returns `input_sha256` and `input_sha256_basis`. The basis is
`canonical` for CPU checks with `engine=False`, when no solver is available
for a CPU check, or when a CPU capability probe fails. It is `local-execution`
for GPU checks and for `engine=True` with a working solver.
For the same local scene, device, and solver capabilities, the local execution
digest matches `result_summary.input_sha256` from the result manifest's
`provenance.input_sha256`. Neither tool emits `spec_sha256`.

## Run locally

Use `estimate` to review the cell count, memory, output size, and CPU time range.
Keep runs small on a laptop CPU.
`run_simulation` runs the solver on this machine only.
Choose `device: "cpu"` for the local CPU path.
The solver must be installed for a run.
Python checks and estimates work without the solver. `capabilities` starts
`phsolver info` and `phsolver --capabilities`. `validate_spec(engine=True)`
starts `phsolver --capabilities` when the spec needs it, then `phsolver validate`.
`validate_spec(engine=False)` starts no solver process. `run_simulation` starts
a local solver run and may query `--capabilities` as part of the run.

Never start a cloud run or spend money without the user's explicit approval.
No MCP tool submits a cloud job, uploads a file, or calls the network.
Cloud GPU runs are available through the Python SDK's `ph.cloud` after the
user approves a quote.

## Read a result

Call `result_summary` first for compact metadata and issues.
Do not load whole arrays to answer a question about a result.
Call `convergence_report` before saying a result is converged.
If it says `found: false`, say there is no saved convergence evidence.
`result_summary` reports recorded facts and advisories, not a validity verdict.

Use the older inspection tools when you need more detail:

- `open_result` loads a local result bundle as the current result.
- `describe_result` describes its run, grid, and provenance.
- `list_monitors` lists its monitors and axes.
- `field_stats` gives statistics for a field cut plane.
- `get_spectrum` gives the spectrum from a supported monitor.

`result_summary` and `convergence_report` can read a supplied result directory
without replacing the current result.
Pass `data_checks=False` to `result_summary` to skip checks that read monitor
data. `convergence_report` reads only `convergence.json` in that result
directory; save a `ConvergenceReport` object's `to_dict()` there to attach one.

## Far-field surfaces and projection

Declare a `ph.FarFieldMonitor` before running. Keep every recorded face in
one homogeneous, isotropic, lossless medium and outside the boundary layers.
For a source enclosed in one medium, prefer a closed box:

```python
import photonhub as ph

ff = ph.FarFieldMonitor.box(
    "ff", center_um=(3, 3, 3), size_um=(2, 2, 2), wlens_um=1.55,
)
```

For a device above a substrate, record separate planes in air and inside
the substrate. These example coordinates assume a domain from 0 to 6 um,
a substrate below z = 2 um and a source near z = 3 um. Include `up` and
`down` in `ph.Simulation(monitors=[...])`, instead of a box that crosses
the interface:

```python
up = ph.FarFieldMonitor.plane(
    "up", axis="z", position_um=4.0, direction="+", wlens_um=1.55,
    center_um=(3, 3, 4), size_um=(4, 4, 0), window=(0.2, 0.2),
)
down = ph.FarFieldMonitor.plane(
    "down", axis="z", position_um=1.0, direction="-", wlens_um=1.55,
    center_um=(3, 3, 1), size_um=(4, 4, 0), window=(0.2, 0.2),
)
```

After the run, project the recorded `data`. Angles are radians: theta is
from +z, phi from +x. `n_medium` is the refractive index, not permittivity;
the saved recipe infers it when omitted. It does not transmit a field
through an interface: each plane projects into its own medium.

```python
import numpy as np
from photonhub.analysis import far_field

phi = np.linspace(0, 2 * np.pi, 72, endpoint=False)
theta = np.linspace(0, np.deg2rad(30), 31)
air = far_field(data, "up", theta_rad=theta, phi_rad=phi, n_medium=1.0)
sub = far_field(data, "down", theta_rad=np.pi - theta[::-1],
                phi_rad=phi, n_medium=1.45)
co_cross = air.polarization("ludwig3", boresight="+z", reference="x")
sub_co_cross = sub.polarization("ludwig3", boresight="-z", reference="x")
circular = air.polarization("circular")
air_cone_w = air.dataset["radiated_power"]
substrate_cone_w = sub.dataset["radiated_power"]
```

These grids integrate cones about +z and -z, not total radiated power.
Use `dataset["intensity"]` (W/sr), `e_theta` and `e_phi` (V); results are
labelled by `f`, `theta` and `phi`. Ludwig-3 gives `co` and `cross`;
circular gives `right` and `left` with IEEE handedness for `exp(-i omega t)`.
On an angle grid, Ludwig-3 defaults to boresight `"+z"` even for a downward
plane, so it is undefined (`nan`) at theta = pi. Use `boresight="-z"` for
the downward cone above. In general, Ludwig-3 is undefined opposite the
chosen boresight.
Power uses the solid-angle weight, so do not sum intensity samples directly.
Refine the angular grid when checking a cone integral. Directivity on a
partial grid is normalized to that grid's power; gain needs measured input
power via `air.gain(input_power_w=...)`.

An open surface truncates the field. Check plane size, edge window and
position, especially above a substrate; a higher index has no measured
accuracy bound here. `n_medium` and `exact=True` do not remove truncation
or add a layered-medium model. Preserve the full result directory and its
`client.json` so projection retains the surface recipe and medium.
See the [far-field guide](https://leapfield.ai/docs/guides/far-field/)
for the measured errors and their settings.

## Cloud runs

The same simulation, including `FarFieldMonitor` declarations, can run
through `ph.cloud`; projection and polarization remain local postprocessing.
No local solver is needed when an approved cloud solver commit is supplied
explicitly. Configure the account as the
[cloud guide](https://leapfield.ai/docs/guides/cloud/) describes, select a
device as `"gpu:" + entry["id"]` from `ph.cloud.gpus()`, and pass it and the approved
solver commit to `ph.cloud.preflight(sim, device=..., solver=..., max_usd=...)`.
Preflight quotes without submitting; save its result as `pre`. Show the quote
and obtain explicit spending approval, then bind the accepted quote:

```python
data = ph.cloud.run(sim, quote_id=pre.quote_id, device=pre.device,
                    solver=pre.solver)
```

Keep `max_usd` on preflight; passing it alongside `quote_id` raises
`ValueError`. The call returns a `RunResult` for
the same analysis calls above. A wait timeout leaves the service job running;
resume it or cancel it explicitly rather than submitting a duplicate.

## Resonances from ring-down

Record a `ph.TimeMonitor`, such as `probe` below, in the simulation before
running. Fit only after all sources turn off. With the run's `sim`, the
analysis chooses that start automatically:

```python
from photonhub.analysis import ResonanceAnalysis, select_resonances

probe = ph.TimeMonitor(name="probe", center_um=(3.2, 3, 3), fields=["Ex"])
# Include probe in sim.monitors before producing data.
fit = ResonanceAnalysis(freq_window=(1.8e14, 2.1e14))
modes = fit.run(data, "probe", fields=["Ex"], simulation=sim)
selected = select_resonances(modes, freq_window=(1.8e14, 2.1e14))
```

The result is labelled by `freq` (Hz), with `decay`, `Q`, `amplitude`,
`phase` and `error`. A fit can include spurious modes. Check a longer
ring-down, later fit starts (`t_start` in seconds), mesh refinement and
individual probes; their sum can cancel a mode. Keep monitor sampling
uniform and its Nyquist frequency above the fit window. A fitted Q alone
is not convergence evidence.

## Report numbers with their settings

Report a physics number with:

- Mesh `dl_um` or cells per wavelength.
- Subpixel state and method.
- Dispersion state.
- Boundary layers.
- Convergence evidence from `convergence_report` when claiming convergence.

Do not call a result converged without a report showing it.
If the report is absent or does not show convergence, state that limit plainly.

## Tool map

The checking tools are `get_schema`, `capabilities`, `validate_spec`, and
`estimate`.
The compact result tools are `result_summary` and `convergence_report`.
The existing result tools are `open_result`, `describe_result`,
`list_monitors`, `field_stats`, and `get_spectrum`.
`run_simulation` is the only MCP tool that runs a simulation, locally.

## Documentation

Start at [PhotonHub docs](https://leapfield.ai/docs/).
For a machine-readable page list, see [llms.txt](https://leapfield.ai/docs/llms.txt).
