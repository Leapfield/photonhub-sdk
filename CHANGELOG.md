# Changelog — `photonhub` Python SDK

### Unreleased

- `photonhub.analysis.focal_metrics` / `FocalMetrics`: focal-spot readout of a recorded
  transmitted plane with the Poynting flux (plane-wave decomposition with the paired H):
  plane of peak flux, 2-D-Gaussian FWHM, transmission through a disc and focusing efficiency
  through an aperture of N × FWHM over an incident power read in the same units
  (e.g. the forward power of an input plane from `diffraction_orders`). Used by the
  metasurface-lens example.
- `photonhub.cloud`: request bodies are sent as compact JSON (no separator spaces), about a fifth
  smaller for a large mode-source profile.
- `photonhub.viz.export_scene`: a periodic in-plane axis is treated as a quasi-2D column
  (widened to `periodic_extent_um` and tiled) only when the domain is thinner than that
  extent; a wide periodic-padded domain (a lens with periodic transverse boundaries) is
  shown whole. Air/vacuum structures that carve the background (air above a substrate)
  are no longer drawn as bodies. New `propagate_um=` (and `propagate_extent_um=`) draws the
  recorded plane's field reconstructed that far downstream (`propagate_plane`) as a plane
  floating at that height — a lens's focal spot above the device.

All notable changes to the published SDK. Dates are release dates on PyPI.
The desktop application and the solver engine share this version number.

## Unreleased

- `photonhub.viz.export_scene` unfolds a §20 symmetry plane: a half-domain
  run with a mirror on an in-plane minimum face is exported as the whole
  device — interior and structure outlines mirrored about the face, the
  field's |E| mirrored as-is and the dominant component's phasor with its
  parity (PEC: normal component even, tangential odd; PMC the reverse).
- `photonhub.viz.plot_comparison(..., stated=(value, label))` draws a paper's
  *stated* number as a dashed line — the form an example uses when the
  article's license does not allow its extracted curve to be re-plotted on
  the public site.
- **Fixed** `plot_field` (and `RunResult.plot_field`) framing: structure
  outlines that extend past the recorded slice — an arm running through the
  wall, or the unsimulated half beyond a symmetry plane — no longer stretch
  the axes; the frame is the slice's own extent.
- **Fixed** a false-positive "lies inside the boundary layers" warning from
  `run_local` on axes carrying a symmetry plane. The PML on such an axis is
  one-sided — the min face is the mirror, not an absorber — so
  equivalence-current mode-launch dipoles a fraction of a cell from the plane
  are interior and no longer reported.
  `Simulation.point_sources_in_boundary_layers` now tests only the far face
  on a symmetry axis; the far face and the other axes' faces are unchanged.
- `photonhub.viz.plot_comparison(x_nm, values, reference=..., ylabel=..., ylim=...)`: the example notebooks' result figure — an observable against wavelength as a line with a paper's digitized series (from `examples/notebooks/refs/`) as markers; `reference_scale` flips a transmittance in dB into a loss.

## 0.1.2 — 2026-09-02

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
- **Module and helper names renamed** in the same spirit, with the same
  deprecated-alias policy: the cloud client `photonhub.web` (now
  `photonhub.cloud`, so `ph.cloud.run(sim)`; `WebConfig`/`WebError`/
  `WebJobTimeout` now `CloudConfig`/`CloudError`/`CloudJobTimeout`), the
  analysis package `photonhub.plugins` (now `photonhub.analysis`; old
  submodule paths such as `photonhub.plugins.resonance` keep importing),
  `Simulation.plot_eps` and `viz.plot_eps` (now `plot_index`), the built-in
  material `materials.cSi` (now `materials.Si`; `materials.get("cSi")` still
  resolves), and the S-matrix driver's `runner="web"` (now `runner="cloud"`).
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
