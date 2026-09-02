# Changelog — `photonhub` Python SDK

All notable changes to the published SDK. Dates are release dates on PyPI.
The desktop application and the solver engine share this version number.

## Unreleased

- **Public API renamed** to cross-solver vocabulary. The wire schema is
  unchanged, and every old name remains importable as a deprecated alias that
  emits a `DeprecationWarning` (removal planned for 0.2). Renames, as
  old (now new): `PolySlab` (now `Polygon`), `FluxMonitor` (now
  `PowerMonitor`), `FieldTimeMonitor` (now `TimeMonitor`), `FieldDftMonitor`
  (now `ProfileMonitor`), `FieldSnapshotMonitor` (now `SnapshotMonitor`),
  `SimulationData` (now `RunResult`), `BatchData` (now `BatchResults`),
  `run_async` (now `submit`), `web.run_quoted_async` (now
  `web.submit_quoted`), `estimate_cost` (now `quote`),
  `plugins.ResonanceFinder` (now `ResonanceAnalysis`), `PermittivityData`
  (now `PermittivityArray`), `UniformGridSpec` (now `UniformMesh`),
  `GradedGridSpec` (now `GradedMesh`), `GradedAxisCoords` (now
  `GradedMeshAxis`), `auto_grid` (now `auto_mesh`), and
  `Simulation.with_auto_grid` (now `with_auto_mesh`).
- Actionable errors when a `Material`/`Medium` is used as the background;
  `sources=()` shell simulations accepted at the model level (the engine still
  requires a source to run); HDF5 export extra; `run_async`/`Batch.run` accept
  `device=`; `Job.cancel()`; a warn-only advisory when a broadband dipole pulse
  keeps the auto-shutoff decay flat.
- Docstrings state that the `ModePower` objective and adjoint gradient
  magnitudes are relative (uncalibrated scale); calibration is pending.
- Public docstrings no longer cite private benchmark files or a comparative
  vendor number.

## 0.1.1 — 2026-09-01

- Public-surface text scrub: physics is described without reference to other
  vendors' products; provenance strings and the shipped example README updated.
- Test suite removed from the sdist; repository `.gitignore` no longer leaks
  into the sdist.
- Documentation site self-hosts its fonts.

## 0.1.0 — 2026-08-31

- First PyPI release of the client: pydantic v2 simulation model (schema
  1.20.0-alpha.1), local runner (`run_local`, `Batch`, `run_async`), cloud
  client (`photonhub.web` with `run_quoted`), cost estimator, GDS import, FDE
  mode solver, S-matrix, near-to-far-field, EME, resonance extraction, adjoint
  inverse design, materials library, visualization layer.
