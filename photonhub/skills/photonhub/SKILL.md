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
