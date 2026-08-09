# photonhub

Python client for the PhotonHub FDTD solver. The pydantic models in
`photonhub.components` are the single source of truth for the simulation
JSON wire format ([`../schemas/GOVERNANCE.md`](../schemas/GOVERNANCE.md));
[`../schemas/simulation_v1.json`](../schemas/simulation_v1.json) is generated
from them via `python -m photonhub.schema emit`.

```python
import photonhub as ph

sim = ph.Simulation(
    size_um=(4.0, 4.0, 4.0),
    grid=ph.UniformGridSpec(dl_um=0.05),
    run=ph.RunSpec(run_time_s=8.0e-14),
    sources=[
        ph.PointDipole(
            center_um=(2.0, 2.0, 2.0),
            polarization="Ez",
            source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4.0e13),
        )
    ],
    monitors=[
        ph.FieldTimeMonitor(name="probe", center_um=(3.0, 2.0, 2.0), fields=["Ez"]),
        ph.FieldSnapshotMonitor(name="final", fields=["Ex", "Ey", "Ez"]),
    ],
)

data = ph.run_local(sim)        # finds phsolver, runs it, parses outputs
probe = data["probe"]           # xarray.DataArray, dims ('t', 'component')
```

## See your simulation before you run it

Every `Simulation` plots itself — geometry, sources, monitors, PML, and the
**permittivity the solver actually meshes** — with matplotlib (built in; plotly
for 3D). `grid=True` overlays the Yee cell edges so you can check resolution:

```python
sim.plot(z=0.11)                  # scene cross-section
sim.plot_eps(z=0.11, grid=True)   # rasterized ε + the Yee mesh overlay
sim.plot_3d()                     # interactive 3D  (pip install photonhub[viz])
```

![A microring + bus waveguide — left: sim.plot (scene); right: sim.plot_eps(grid=True), the meshed permittivity the solver samples](docs/img/sim_showcase.png)

## What you can do today

Shipped surface as of schema **v1.16.0-alpha.1** (validated dispersive solver
core — single-pole Lorentz ADE, with recorded MI300X CPU↔GPU equivalence under
the tolerances in [`../engine/NUMERICS.md`](../engine/NUMERICS.md) §8 — plus full-vector mode
injection, GDS import (`ph.import_gds`), adjoint gradients, and the silicon-PIC
MVP — see [`../docs/quickstart.md`](../docs/quickstart.md) and the notebook
gallery under [`../examples/notebooks/`](../examples/notebooks/)):

- **Run a simulation:** `ph.run_local(sim)` (subprocess + file protocol;
  `device="cpu"|"gpu"|"gpu:N"`), `ph.run_async(sim)` and `ph.Batch(...)` for many
  sims in flight.
- **Know the cost first:** `ph.cost_estimate(sim)` returns a dollar estimate
  from an overridable Tcell-step planning rate *before* you press run; the
  default is an estimator input, not a production billing commitment.
- **Geometry:** `Box`, `Sphere`, `Cylinder` (full, or an annular sector / ring
  via `inner_radius_um` + `angle_start` / `angle_stop`), and `PolySlab` (an
  extruded polygon — e.g. a taper).
- **Sources:** `PointDipole`, `PlaneWave` (normal-incidence TF/SF), and the
  recommended `mode_launch` waveguide-mode builder, all driven by a
  `GaussianPulse`. The `ModeSource` wire type remains for legacy scalar and
  continuous-adjoint compatibility.
- **Monitors:** `FieldTimeMonitor`, `FieldSnapshotMonitor`, `FieldDftMonitor`,
  and `FluxMonitor` — fp64 DFT field and flux power. A DFT plane may carry the
  optional `ModePort` authoring recipe; the engine strictly validates and then
  ignores that metadata while recording the same raw fields, and result
  post-processing performs the requested modal projection. Workbench keeps one
  shared wavelength sweep across ports; port planes/windows stay outside
  PML/absorber cells; and preflight requires one downstream driven port linked
  to the first wire-order normalization source with no other active excitation,
  so an unevaluable S-column is rejected before the time-domain run.
- **Component library:** `ph.library.straight / bend / taper / crossing /
  coupler / ring` — each returns a `Component` (structures + ports) in ~one line.
- **Material library:** `ph.materials` — 16 literature materials (cSi, SiO2,
  Si3N4, GaAs, InP, Ge, LiNbO3, sapphire, AlN, TiO2, MgF2, CaF2, PMMA, ...)
  with cited dispersion data and validity ranges. `cSi.medium(1.55)` freezes
  the index at one wavelength; `cSi.medium(band_um=(1.5, 1.6))` least-squares
  fits the engine's single Lorentz pole over the band (Courant-safe pole
  placement, ≤1e-4 index error over 100 nm). Bring measured ellipsometry data
  via `materials.Material.from_nk_data(...)`.
- **Mode-resolved transmission:** the recommended `photonhub.plugins` pipeline —
  `solve_yee_mode` → `mode_launch` → `mode_monitor` →
  `transmission(out, in, data)` — returns `{freq_hz: T}`. Use
  `solve_yee_mode_bank` / `modes_by_freq` for a broadband launch and readout;
  the legacy `ModeSolver` / `mode_source` path remains for scalar and adjoint
  compatibility. Full complex multiport S-matrices:
  `plugins.smatrix` (`SPort` + `assemble_smatrix`, one run per driven port —
  the analogue of Tidy3D's ComponentModeler).
- **Meshing:** `UniformGridSpec`, or `GradedGridSpec` + `auto_grid` for a
  cells-per-λ graded mesh; `MeshOverride` (Tidy3D `MeshOverrideStructure`) forces
  a target spacing inside a geometry regardless of material —
  `sim.with_mesh_overrides(MeshOverride(geometry=Box(...), dl_um=(0.02,0.02,None)))`.
  Graded meshes run on CPU, single GPU, and multi-GPU with PML/absorber,
  dispersive media, DFT/flux monitors, and plane-wave/mode sources. The one
  unsupported combination is graded + `subpixel_method="tensor_full"`; see the
  current status in
  [`../engine/docs/multi-gpu-decomposition.md`](../engine/docs/multi-gpu-decomposition.md).
- **Mode solving:** the FDE plugins provide the lightweight semi-vectorial
  `ModeSolver` and the full-vector `VectorModeSolver`, including bent/leaky
  modes with complex `n_eff` and bend-loss readout.
- **Visualization:** `Simulation.plot` / `plot_eps` (draws Box / Sphere /
  Cylinder / PolySlab; `grid=True` overlays the Yee mesh) / `plot_3d`,
  `SimulationData.plot_field`, and the module-level `photonhub.viz.plot_mode`
  (FDE mode heatmap) and `photonhub.viz.plot_spectrum` (`T` vs λ).
- **Export:** HDF5 converter for the parsed results.
- **Correctness aids:** pydantic models reject structural errors at
  construction, the capability manifest catches client/engine drift, and
  `phsolver validate` is authoritative for grid/device constraints. Subpixel
  smoothing supports Box / Sphere / Cylinder / PolySlab; non-dispersive scenes
  default to contour smoothing, while dispersive scenes default off unless the
  user opts in.

### Limits (today)

This is a validated **dispersive** solver core plus the silicon-PIC MVP:

- Dispersion is a **single Lorentz pole** per medium (ADE, included in the
  recorded MI300X equivalence suite under `engine/NUMERICS.md` §8 tolerances;
  numerical definition in §19) on top of relative permittivity + Ohmic conductivity;
  multi-pole / Drude (metals) / anisotropic poles are still open. Explicit
  subpixel + Lorentz runs are supported and equivalence-tested, but the client
  keeps their construction default off and warns callers to use the stabilized
  PML profile.
- S-matrix assembly (`plugins.smatrix`) costs **one simulation per driven
  port** — there is no single-run multiport solve.

Net: shapes go in and **mode-resolved transmission** comes out — solve the
eigenmode, inject it with `mode_launch`, ratio two `mode_monitor` planes, and plot
`T(λ)`; or assemble the full S-matrix with `plugins.smatrix`.

## Learn more

- [`../docs/quickstart.md`](../docs/quickstart.md) — end-to-end first run.
- [`../examples/README.md`](../examples/README.md) — examples index and
  **twelve-notebook** gallery.
