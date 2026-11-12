# Changelog: `photonhub` Python SDK

All notable changes to the published SDK. Dates are release dates on PyPI.
The desktop application and the solver engine share this version number.

## Unreleased

- **Output size estimates now equal the solver's output.**
  `cost_estimate().output_bytes` and the monitor output-size check now give
  the bytes the solver writes for a full-duration run (an early stop writes
  no more). A frequency monitor counts the grid nodes its box spans, as the
  solver records them; the estimate used the box size over the base cell
  size, which undercounts a refined graded axis: the broadband directional
  coupler example estimated 83.7 MB for a 288.4 MB result, and its cloud
  runs failed, because the service sizes the result it accepts from this
  estimate. Time and snapshot monitors now count the final step the solver
  also records when the sampling interval does not divide the step count.
  Estimates can go down on graded axes coarser than `dl_um`; they are exact
  there too. The frequency monitors' share of the device-memory estimate
  follows the same count. The cloud service computes the estimate with its
  own copy of the SDK, so its result limit changes only after the service is
  redeployed with this release.

## 0.1.7 (2026-11-02)

Far-field recording and analysis lead this release: declare a surface, project
onto angle, direction-cosine or Cartesian grids, and read labelled results.
The SDK also corrects fitted-domain and port handling, restores saved media,
and resolves cloud GPU selections. The wire schema remains 1.22.0-alpha.1.

Solver changes require a solver built from this release. With a 0.1.6 solver,
SDK-side far-field analysis and setup changes are available, but the old Bloch
frequency-domain scale and multi-GPU restrictions remain. An explicit cloud
`solver=` selects that build independently of the SDK; the service must allow
it. See the entries below before comparing results across the upgrade.

### Added

- **Far-field setup and self-run guidance.** The far-field guide now puts
  substrate limits first, explains downward polarization and solid-angle
  integration and subpixel settings. The bundled skill explains local solver
  installation and cloud submissions using accepted quotes.
  A repository self-run workflow uses invented inputs and retains its accepted
  preflight quote when submitting under a spend cap. Its local CPU check and
  fake-service tests do not establish deployed cloud execution or convergence.
- **Ludwig-3 polarization warns about its antipodal mask.** Requested
  directions in `1 + r . b <= 1e-12` emit a `UserWarning` naming the boresight
  at the caller's line, including calls through SDK wrappers.
  Co/cross amplitudes remain NaN there because the basis is undefined opposite
  the boresight. On an angle grid, choose `boresight="-z"` for downward
  emission, or partition a full sphere and choose a boresight for each part.
  Integrate squared amplitudes with solid-angle weights for power shares.
- **Public SDK artifacts exclude the server executor.** Wheel and sdist now
  follow the public mirror's client-only boundary. The private solver image
  adds the executor explicitly from its own source tree. The package metadata
  comment and the quasi-2-D far-field error also use plain punctuation.
- **`FarFieldMonitor`: declare the surface a far-field projection reads.**
  `FarFieldMonitor.box(name, center_um=, size_um=, wlens_um=, exclude=)`
  and `FarFieldMonitor.plane(name, axis=, position_um=, direction=,
  wlens_um=, pts_per_wlen=, window=)` go in `Simulation(monitors=...)` and
  are compiled there into ordinary `field_dft` monitors, one per face, named `<name>.x_minus`,
  `<name>.x_plus` ... `<name>.z_plus` (monitor names allow no `:` or `+`):
  the four tangential components, one cell thick, every cell and one cell
  past the box on a box, and an in-plane stride on a plane that keeps
  `pts_per_wlen` samples per wavelength in its medium. Nothing changes on
  the wire. The recipe (faces, outward normals, the medium they sit in, and
  where they apply the symmetry planes, faces left out and window) is
  written beside the run in `client.json`, and `far_field(data, "<name>",
  ...)` projects the surface by its name into that medium (bit for bit
  the face list with that `n_medium`); a result reloaded without the record
  projects from the face names, with a warning. `FarFieldMonitor.faces`
  lists a declaration's faces as `(monitor name, axis, outward sign)`, and
  `Simulation.far_field_monitors` maps each surface a simulation records
  to its faces. A reloaded result keeps its recipe through an edit of the
  run, sources, monitors or precisions; an edit of the background,
  structures, boundaries, layers or grid keeps the face monitors but drops
  the recipe, with a warning, since its medium and crossing checks were
  made against them. Building the simulation refuses, naming the face, a
  face in a PML or absorber layer, across a structure or a medium
  boundary, faces in different media, a medium the projection cannot use
  (anisotropic, custom, PEC or lossy), a box on a quasi-2-D axis, a stride
  or a window on a box, a box or a plane's extent that does not reach a
  symmetry plane or crosses it off centre, a plane on a symmetry plane,
  and leaving out a face normal to one. It warns when a face is closer
  than half a wavelength to a structure, when a source or a structure lies
  within 2 cells of a face's innermost sample, and, on a graded mesh, when
  a box face is nearer than 0.35 wavelengths (in its medium) to a grid
  line where the cell size changes, along its normal from both recorded
  samples, or a source is nearer than 0.1 wavelengths to one (a point
  source along each axis, a launch along its normal): the range the
  graded-mesh accuracy below is shown for. The source
  checks see every source the simulation runs, the launches that `ports=`
  and `source=` add included. A grid line counts as a change of cell size
  when the cells either side differ by more than 1 %, so a grading gentler
  than that per cell never warns, however far it stretches the cells.
  With symmetry planes it records the kept part (a face across a plane
  from the plane on, no far-side face) and the recipe keeps the planes:
  by name it projects bit for bit as the same faces read as a face list.
  On FDTD dipole boxes (vacuum, 16 cells per wavelength, three cases) a
  closed one differs from the full run's box with its far-side faces the
  mirror of the near ones by 0.16 to 0.30 % of the peak intensity (0.026
  to 0.042 % at 32 cells), and from the same declaration on the full run,
  whose far-side faces sit a cell nearer the planes, by 0.15 to 0.43 %
  (0.04 to 0.10 % at 32). `FarFieldMonitor.box(..., exclude=("-z",))`
  leaves faces out, an open box: those faces are neither recorded nor
  checked, the faces kept keep every check, and `far_field(data, name)`
  ends them at the plane the face left out is declared on, so the open
  edge does not move with the mesh. `FarFieldMonitor.plane(...,
  window=(f1, f2))`, off by default, tapers the currents (sin² over the
  outer fraction of the plane's declared extent at each end, along its two
  in-plane axes), so a run with symmetry planes and the full run window
  the same surface. An open surface is a truncated one and reads the
  pattern well only where the field it cuts off is small. For a point
  dipole in a medium of index 1.45 (20 and 30 cells per wavelength), a
  plane 0.62 µm away and 2.41 µm in half width misreads it by 36 to 37 % of
  the peak within 30° of its normal (7.5 to 7.8 % with a window of 0.4),
  and a box 0.62 µm in half width with a face left out by about 10.1 to
  10.2 % within 30° of its closed side (on exact fields 10.1 to 10.2 % from
  20 to 60 cells, against about 10.0 % for the box as declared; a small
  residual, first order in the cell, comes from where the faces kept are
  read, at most about 0.7 points within 60° at 20 cells). Over a
  substrate, against the closed form of a dipole above a half space, the
  open box reads 13.5 to 13.7 % within 30°. The power within a cone reads
  within 2 % for an open box in one medium, but over the substrate the
  open box's is off by +3.5 to +3.9 % within 30° and -10.8 % within 80°,
  and a windowed plane's by +17 % within 30°. The far-field guide and the
  known limitations give when an open surface holds. **Open surfaces over
  substrates can have substantial pattern and cone-power error; higher-index
  substrate accuracy has not been established.** See the
  [far-field limits](https://leapfield.ai/docs/guides/far-field/#when-an-open-surface-holds)
  for the full measurement settings and limitations.
- **`far_field` reads k-space and Cartesian grids, and the field at a
  finite distance.** Besides directions (`theta_rad`, `phi_rad`),
  `far_field(data, name, ux=, uy=)` reads direction cosines about
  `plane_axis` (a single plane's own normal by default) in the half space
  `direction`, and `far_field(..., plane_axis=, plane_um=, x_um=, y_um=)`
  reads points on a plane, each in the direction it lies in from
  `origin_um` (the surface's centre by default; pass it to compare runs),
  with a warning for points nearer than `2D²/λ` (`D` twice the surface's
  farthest reach from `origin_um`). A k-space point with `ux² + uy² > 1` has no
  direction: its fields are `nan` and its intensity 0, as
  `diffraction_orders` reports evanescent orders. On a plane over one
  period of a periodic structure the k-space intensity at `u = mλ/(nL)`
  gives grating order `m`'s power, `U (λ/n)² / (A cos θ)` in W, which
  `diffraction_orders` reports times the period's area in µm² (1e12 times
  more), with its `(k1, k2)` along its `in_plane_axes` (z and x on a plane
  normal to y, where `ux`, `uy` run along x and z). On an FDTD grating (11
  orders, 20 and 30 cells per wavelength, at most 100 periods with the
  default shutoff) the two agree within 1.1e-4 and 3.8e-5, set by the ±2
  orders at 72° still settling (7.1e-5 and 1.4e-5 from 200 periods), the
  directions exact. `exact=True` evaluates the field the surface's
  currents radiate at a finite distance with the full Green's function of
  the medium, on a Cartesian grid or on a sphere of radius `r_um` about
  `origin_um`; its amplitudes tend to the
  far-field ones as `1/r` (fitted order −1.000 on a dipole box, 5 to
  320 µm), and on the dipole's exact fields it reads the dipole's own
  field to the readout's error at every distance (0.57 % and 0.13 % of the
  peak at 20 and 40 cells per wavelength). For a meshed grid on a plane
  parallel to one evenly sampled recorded plane it uses the plane's angular
  spectrum, elsewhere a direct sum over the recorded samples in bounded
  blocks; the two agree within 1e-5 on an exact beam that has decayed at
  its plane's edges, and by about the cut's own error on a plane that cuts
  the field. Near the surface the direct sum loses accuracy: 3 to 6 % of
  the peak one sample spacing out at 20 cells per wavelength, and over a
  box's edges more on finer meshes (up to 21 % at 80 cells); it warns
  within two spacings or a tenth of a wavelength. A point inside a box or
  a Cartesian point behind a plane is refused; directions of an angle or
  k-space grid that end at or behind a plane read `nan`, with a warning,
  and a grid with none in front of it is refused. On FDTD the exact
  projection of a `FarFieldMonitor` box around a dipole reads the field
  the solver recorded on a plane 0.9 µm beyond the box to 1.1 %, 0.51 %
  and 0.29 % of the peak at 20, 30 and 40 cells per wavelength.
  `FarField` keeps its flat layout and gains the grid's kind and shape
  (`reshape()` needs no sizes), its coordinates (`ux`, `uy`, `x_um`,
  `y_um`, `plane_axis`, `plane_um`, `direction`), `r_um`, `origin_um`,
  `exact`, an exact projection's radial and magnetic components (`e_r`,
  `h_r`, `h_theta`, `h_phi`), `propagating` and `cartesian_fields()`;
  `radiated_power()` and `directivity()` need an angle grid. `theta_rad` is
  no longer required, since one of the three grids is.
- **`FarField` results are labelled `xarray` data, with gain, cross
  section, polarization and plots.** `FarField.dataset` is the projection
  as an `xarray.Dataset` labelled by frequency and the grid's own axes:
  `(f, theta, phi)` on an angle grid (radians), `(f, ux, uy)` on a k-space
  grid, the plane's two axes in µm on a Cartesian grid (`(f, x, y)` for a
  plane normal to z), `(f, point)` for paired points, with `wlen_um` and
  `n_medium` along `f`. It holds `e_theta`, `e_phi` (and an exact
  projection's `e_r`, `h_r`, `h_theta`, `h_phi`), `intensity` and, on a
  meshed angle grid, `radiated_power` and `directivity`: the projection's
  own arrays, bit for bit. New methods return labelled results:
  `gain(input_power_w)` (`4πU / P_in`, the source's power given as a
  number, per frequency, or a `PowerMonitor` spectrum; required, since a
  source's power depends on its surroundings), `cross_section()`
  (`dσ/dΩ = U / I_inc` in µm²/sr; a run driven by one `TFSFBox` gives the
  incident intensity `|E0|²/(2η)` itself, otherwise pass it),
  `polarization("ludwig3")` (co- and cross-polar amplitudes, Ludwig's third
  definition about a boresight: by default `+z` with reference x on an
  angle grid, and the grid's own axis and side on a k-space or Cartesian
  grid) and `polarization("circular")` (right- and left-hand, IEEE
  handedness), `renormalize(distance_um)` (a copy read at that distance
  from `phase_origin_um`, the point the amplitudes' phase refers to, which
  `far_field` now records), `power_density()` (W/m²) and `cartesian()`
  (`Ex` ... `Hz`). `photonhub.viz.plot_far_field_cut` draws a polar cut and
  `plot_far_field_map` a map over direction cosines or a plane. On an FDTD
  dipole in vacuum (uniform mesh of 20, 30 and 40 cells per wavelength,
  no structures, no dispersion, PML one wavelength thick) the
  directivity's peak reads 1.5156, 1.5067 and 1.5037 and the gain from
  the flux of a `PowerMonitor` box centred on it 1.5166, 1.5070 and
  1.5038 (off centre the box's total is off by an amount first order in
  the cell times the offset: centre it, or check it against a larger box).
  On a sphere of index 1.5 and radius 0.6 µm at λ = 1 µm (uniform mesh,
  the default subpixel smoothing, no dispersion, PML 0.5 µm) the
  differential cross section reads within 0.80 % of the Mie series' peak
  at 25 cells per wavelength (0.68 % at 30); on a sphere of index 2 and
  radius 1 µm, beside a sharp resonance there (a 0.5 % lower index moves
  the series by 37 % of its peak), it needs a long run to settle (2.3 to 2.7 %
  at 25 cells after 100 to 300 periods); settled, it reads 3.8, 1.95 and 1.4 %
  at 20, 25 and 30 cells per wavelength, and the run's other four
  frequencies, each beside a mode the grid shifts, read 4.6 to 19.7 %.
- **`colocated_fields`: a recorded plane with E and H on the same
  points.** The solver records each field component at its own Yee point,
  while `data[name]` labels them all with the cell's base node, so a
  product such as `E x H*` read at the labels is wrong at first order.
  `photonhub.analysis.colocated_fields(data, name)` reads a
  `ProfileMonitor` plane, a `FarFieldMonitor` face (`"<name>.z_plus"`
  ...) or a `FarFieldMonitor` plane by its name and returns an
  `xarray.Dataset` labelled `(f, a, b)` (the plane's axes in xyz order, in
  µm in the frame of `data[name]`, `wlen_um` beside `f`) holding the
  recorded `Ex` ... `Hz` in V/m and A/m on the plane's grid nodes. A
  component half a cell off the node is interpolated linearly onto it
  from its two samples either side (by the cells' widths on a graded
  mesh, across the stride on a strided plane, wrapped across the period on
  a periodic or Bloch axis the plane spans at every cell), after `far_field`'s
  placement of the samples (the run's grid lines, a symmetry-reduced
  plane unfolded at each component's own Yee point). A plane recorded one
  cell thick (from the quarter point of a cell to that of the next; every
  `FarFieldMonitor` face) puts E and H on one surface, its second sample's
  E plane, where a `PowerMonitor` reads H; with one sample H stays half a
  cell above E, as recorded, with a warning when the plane holds both. A
  far-field face carries the surface's index as `n_medium` when its recipe
  is known; a surface or box face read without one warns. On a uniform
  plane recorded at every cell with one sample, the colocated fields
  through `far_field`'s quadrature give its radiation integrals times the
  interpolation's `cos(k r_a d / 2)` exactly where the field has died out
  at the plane's edges, and on a wrapped plane at its diffraction orders
  only; strided planes keep the total current with another factor,
  graded ones agree to second order, and a plane recorded one cell thick
  differs from `far_field`'s reading of it by about `(k dl)^2 / 16`
  (0.62 % at 20 cells per wavelength). On a `FarFieldMonitor` box face the
  fields are, bit for bit, `far_field`'s H-onto-E pass there. On an FDTD
  dipole in vacuum (uniform mesh, PML 0.5 µm, no structures, no
  dispersion) the flux of a plane recorded one cell thick that runs into
  the PML reads 0.46 %, 0.21 % and 0.12 % below a `PowerMonitor` on the
  same plane at 20, 30 and 40 cells per wavelength (second order);
  recorded with one sample, 2.0 %, 1.3 % and 0.91 % (first order, from
  where the plane meets the PML: each point of `E x H*` is then first
  order off). A plane cut in the open loses about one cell's strip at its
  edges (1.8 %, 1.1 % and 0.78 % cut 1 µm from the dipole axis); one
  across a periodic cell wraps and reads at second order (0.62 %, 0.28 %
  and 0.16 %). Record a plane one cell thick for any product of E and H.
  The reader's warning about the staggered components of `field_dft`
  arrays now points to it.
- **A packaged skill imports a simulation from another FDTD tool.**
  `photonhub-mcp --print-import-skill` prints it, and the
  `photonhub://skill/import/SKILL.md` resource serves it. An agent reads the
  source as text, writes a PhotonHub script, checks it, and lists what maps,
  what maps in part, and what has no equivalent in an import report.
- **`far_field` takes its angles as `theta_rad` and `phi_rad`.** `theta`
  and `phi` keep working as the earlier spellings of the same arguments.
- **`far_field(..., yee=..., dl_um=...)`.** `yee="exact"`, the default,
  reads each field component where the solver sampled it (see "Changed");
  `yee="legacy"` keeps the earlier readout for one release and gives a
  `FutureWarning`. `dl_um` is the uniform cell size the readout measures
  the sampling offsets in. By default it comes from the run's manifest, or
  from the two samples of a face recorded one cell thick; a value that
  disagrees with either, or with a face's in-plane sample spacing, raises
  `ValueError`, and so does one given with a graded run (its grid lines
  place the samples) or with planes sampled unevenly.
- **`method="yee"` solves bent waveguides.**
  `VectorModeSolver(..., method="yee").solve(bend_radius_um=..., num_pml=...,
  pml_strength=...)` now solves bent waveguides and straight guides with
  absorbing x layers, with the default method's arguments, defaults, checks,
  mode-selection rules, warnings and bend metadata (`k_eff`,
  `loss_db_per_cm`, `loss_unresolved`, `window_limited`, `pml_cells_xy`,
  `physical_mask`). The bend is mapped onto a straight guide by
  transformation optics: permittivity and permeability are scaled by the
  radial factor `(R + x)/R` at each field component's own lattice position,
  and the layers are a complex coordinate stretch folded into the same
  tensors, with the radial factor continued to the stretched radius in the
  outer layer. On two bent slabs with exact solutions of the 2-D dispersion
  relation (a 1 µm core of index 1.7 in 1.6 at R = 50 µm and 1.3 µm
  wavelength, in a 7.2 µm window; a 2 µm core of index 1.7 between 1.6
  inside and 1.55 outside at R = 49 µm and 1.55 µm, in an 11.9 µm window;
  the y axis closed periodically over three cells; exact per-component
  subpixel sampling, 0.48 µm layers at strength 30, no dispersion) the
  index, the loss and the group index converge at second order in the cell
  size over 40 or 20 nm down to 5 nm. At 5 nm the index differs from the
  exact roots by at most 6.5e-7, a 190 dB/cm loss by 0.0161 dB/cm and a
  0.455 dB/cm loss by 1.42e-5 dB/cm. The returned fields are the physical
  ones (`ez` and `hz` are the azimuthal components), co-located on the cell
  centres. A new keyword-only `yee_periodic=(periodic_x, periodic_y)` on the
  constructor closes an axis periodically, as `from_rectangular_core` already
  does for slabs, so a bent slab with different claddings on its two sides
  can be built from an array; arrays that vary along a periodic axis raise
  a `ValueError`. The default method and the straight Yee solve
  are unchanged bit for bit.

  A separate comparison, with no convergence order claimed: on a Si 500 x
  220 nm strip (index 3.4757) in oxide (1.444) bent to R = 3 µm at 1.55 µm
  (5.8 x 1.5 µm window, 0.24 µm layers at strength 30, exact subpixel
  sampling, no dispersion), the TE0 bend shift against the straight guide
  in the same window reads 0.0039991, 0.0038990 and 0.0039465 at 40, 20 and
  10 nm cells, against 0.0039450 ± 5.8e-5 from an independent reference
  solver. The core faces sit at a different place on the grid at each of
  these three cell sizes, so the sequence shows agreement within the
  reference's error bar, not an order of convergence. A 5 nm level, run in
  the validation suite's mode library with the faces on node lines as at
  10 nm, reads 0.0039346 ± 1.2e-5, 1.0e-5 below the reference. Bends of +R
  and -R agree to 3e-15.
- **A windowed `PowerMonitor` runs on several GPUs.** A power monitor with
  `center_um` and `size_um` used to stop a run on several GPUs
  (`device="gpu:all"` or `"gpu:0,1"`) with an error; only one GPU or the
  CPU could run it. The solver now splits the window across the GPUs the
  way it splits a full plane. The GPU test suite passed on one GPU on
  2026-10-22, 79 of 79 tests. Its new test splits the grid on that one GPU.
  On a uniform mesh it uses 2, 3 and 4 parts. On a graded mesh and with a
  symmetry plane it uses 2 and 4 parts. Every windowed reading matched the
  unsplit run, bit for bit. A later two-card hardware run on 2026-10-28
  compared windowed monitor outputs with one card. That historical check
  does not qualify a new release image. No reading on one GPU or the CPU
  changes. A solver released as 0.1.6
  or earlier still refuses the window on several GPUs.
- **Runs on several GPUs can time each GPU.** Set
  `PHOTONHUB_DEVICE_TIMING=K` in the solver's environment to time every K-th
  step on each GPU: how long each one ran its updates, copied grid planes
  from its neighbours, waited for the host and waited for the slowest GPU.
  Progress events then carry `device_time` and the `done` event
  `device_time_s`. Timing is off by default. It only reads GPU timers, so
  outputs are the same bit for bit with it on or off. A six-card hardware
  run on 2026-10-31 measured timing and its overhead. Its timing reader still
  has uncertainty, overhead-accounting and counterfactual-savings limitations;
  that historical measurement does not qualify a new release image or a new
  speed claim. Each progress event of a
  run on several GPUs now also gives each GPU's memory in use (`cards`).
- **`plot_comparison` draws discrete results as points.**
  `points=("FDTD, TE0", "FDTD, TE2")` draws the named series of `values`
  as markers with no joining line, such as a few runs of one method beside
  another method's sampled curve. Each keeps the colour of its place, and
  its legend entry shows the marker only. A single series is named by
  `label`; a name that is not one of the series raises `ValueError`. Left
  unset, every figure draws as before.
- **The far-field projection can radiate into a dielectric, not only into
  vacuum.** `far_field(..., n_medium=1.45)` projects a plane recorded inside
  a silica substrate into that substrate; `n_medium="auto"` reads the index
  from `simulation=` (per frequency in a dispersive medium) and refuses a
  lossy medium. The medium's wavenumber and wave impedance enter the
  pattern, the intensity, the radiated power and the directivity, and
  `FarField.n_medium` records the index. `n_medium` is an index, not a
  permittivity. The default stays vacuum, and its numbers do not change:
  with `simulation=`, a plane or closed box in a medium of another index, or
  a box whose faces sit in different media, only gives a warning. A box
  across an interface is refused when an index is given or read. A layered
  half-space, such as cladding over a substrate, is still outside the model.
  Checked against the exact field of a dipole in a dielectric on a closed
  box and on a plane, and against a point dipole run in a dielectric
  background.

### Changed

- **Deprecated: the flat layout of `FarField`.** `e_theta`, `e_phi`,
  `theta`, `phi`, `intensity`, `radiated_power()`, `directivity()`,
  `reshape()`, `cartesian_fields()` and the per-point `ux`, `uy`, `x_um`,
  `y_um`, `r_um`, `e_r`, `h_r`, `h_theta`, `h_phi` and `propagating` (arrays
  indexed `[frequency, point]` over a flat list of points) keep working,
  with the same values, for one release; reading one gives a
  `FutureWarning` that names its replacement in `FarField.dataset` (or
  `cartesian()`).
- **Numbers move: `far_field` reads each field component where the solver
  sampled it.** The solver stores E and H half a cell apart, along each
  face's normal as well as in its plane. The projection used to average
  each component to the cell node and leave the half cell along the normal,
  so a box gave a pattern with a front/back asymmetry first order in the
  cell size: 0.28 of the peak for a point dipole in vacuum at 25 cells per
  wavelength. It now reads each component at its own point. A single plane
  needs one sample along its normal and may keep an `interval_space`
  stride. **Record each box face one cell thick** (size `dl` along its
  normal, centred at `(k + 3/4) dl`) and one cell past the box on every side
  in its plane; `far_field` then brings E and H onto one surface. The same
  dipole now reads 0.1 % of the peak in front/back asymmetry with it half a
  cell from the box centre, up to about 0.80 % with it near a box corner
  (0.10
  to 0.12 µm from each of the three faces there) and up to about 1.06 %
  near a box edge, two faces near and the third coordinate centred
  (uniform mesh, no structures, no dispersion, 25 PML layers, a box 1.2
  wavelengths wide). Wherever the build does not warn the bound is 2 % of
  the peak at 25 cells per wavelength, and the error falls about 4x per
  halving of the cell. In course notebook 12 the equatorial
  ripple of a dipole's pattern falls from 25 % to 0.4 % at 50 nm cells. In a
  dielectric the resolution counts per medium wavelength, and at the same
  count the asymmetry is larger than in vacuum (the solver's time step
  follows the vacuum limit). Use at least 30 cells per medium wavelength,
  where the bound is also 2 %: a dipole near a box edge reads up to
  0.96 %, 1.07 % and 1.11 % at index 1.5, 2.5 and 3.5, and the worst
  of 22 other placements 0.70 %, 0.81 % and 0.84 % with the dipole three
  cells from a face (0.96 %, 1.11 % and 1.15 % at 25 cells); for
  a source fixed in space it falls as the square of the cell (fitted order
  2.01 to 2.04 with the dipole on grid nodes along y and z, 0.2 medium
  wavelengths from the faces it approaches, and along x on its Ex Yee
  point, half a cell from the node). Box faces recorded with
  `interval_space` (when the cell size is known: a run's result, or
  `dl_um=`), faces with more than two samples along the normal (a volume
  passed as a plane included), and
  boxes whose faces stop short of one cell past the box are refused, naming
  the face; this includes a box with a face left out. Graded meshes and
  symmetry-reduced runs take this readout too (next entry). Box faces
  recorded with one sample, and plain arrays whose cell size or grid lines
  cannot be told (pass `dl_um=`, or the `RunResult`), fall back to the
  earlier readout with a warning (a `FutureWarning` for the box faces).
  `far_field(..., yee="legacy")` keeps the earlier readout for one
  release.
  `colocate=False` still means synthetic fields that share their sample
  points; a box of such fields now takes fourth-order weights at its edges,
  which moves its pattern by up to about 0.3 % of the peak (a dipole box at
  25 cells per wavelength). Each face is integrated with factored phase
  terms: on a 12-core laptop a single 300 x 300 plane projects onto 3,312
  directions about 100x faster (0.08 s against 7.5 s), and a box of six
  99 x 99 faces about 8x faster (0.36 s against 2.8 s; each box face is
  projected twice). With `simulation=`, the medium a face recorded one
  cell thick radiates into is sampled half a cell beyond its outer sample.
- **Numbers move: `far_field` reads graded meshes and symmetry-reduced
  runs the same way.** Both used the earlier readout, whose front/back
  asymmetry on a box is first order in the cell size. On a graded mesh
  each sample is now placed from the run's grid lines (pass the
  `RunResult`: its manifest lists them), a component half a cell off the
  node at the midpoint of its cell; H is brought onto the E plane with the
  two cells' widths as weights, and the box-edge rule integrates the cubic
  through the samples' true positions. An open plane whose samples are
  unevenly spaced, `colocate=False` included, takes the same cubic instead
  of the centred cell widths (a synthetic plane with warped coordinates
  moves by 4.3e-5 of the peak). On a uniform mesh every number is
  unchanged, bit for bit. A vacuum dipole box with a cube 0.4 wavelengths
  wide around the dipole meshed twice as fine (`MeshOverride`, grading by
  up to 1.4 per cell; no structures, no dispersion, PML one wavelength
  thick) closes on the near-field flux within 0.18 %; its front/back
  asymmetry is up to 1.79 % of the peak at 25 cells per wavelength in the
  background, 1.25 % at 30 and 0.92 % at 35 (worst of 33 placements, the
  dipole 0.1 µm from a face, the cube's edge just outside it), falling as
  the square of the cell. The solver's time step follows the finest cell,
  so the background runs at a lower Courant number and its dispersion
  error grows, as in a dielectric (a uniform mesh at that time step reads
  1.05 % for the same placement, 0.67 % at its own). The rest comes from
  the edge of the refined region lying just outside a box face, since the
  projection assumes one homogeneous medium outside the box: with the
  refinement ending inside the box the same placement reads 1.17 %; with
  it 0.19 µm past the faces the worst of 33 placements reads 1.68 %, and
  with it carried past them, 0.35 µm to the end of its grading (0.30 µm to
  the nearest change of cell size), 0.87 %. The range the accuracy is
  shown for: every box face at least 0.35 wavelengths from the nearest
  grid line where the cell size changes, along its normal, on either side
  (a refined region's edge counts, not only the end of its grading),
  every source at least 0.1 wavelengths from one (a point source along
  each axis, a launch along its normal), and at least 30 background cells
  per wavelength. There, for refinements 2 and
  3 times as fine as the background, the front/back asymmetry of a vacuum
  dipole box stays within 2 % at 30 background cells wherever the build
  does not warn, with the power within 0.24 % of the near-field flux. It
  reads up to 0.90 % (ratio 2) and 0.93 % (ratio 3) with the refinement
  inside the box and a dipole near a box edge (0.67 % and 0.71 % at a box
  corner, 0.11 to 0.13 µm from its three faces), and 0.13 % (ratio 2) and
  0.055 % (ratio 3) with it carried past every face; for a
  source fixed in space it falls at about second order (fitted order 2.06
  and 1.98 with the refinement inside the box). Higher ratios are not
  covered. Outside that range no bound is given: a source on or just past
  the edge of a refined region radiates an asymmetric field by itself, up
  to 1.085 % at ratio 3 and 0.92 % at ratio 2 at 30 background cells. A
  symmetry-reduced run is mirrored at
  each component's own point with the sign each plane gives it, and a box
  face normal to a plane gains its mirror image; record the box on the kept
  side only. The planes are read from the run, so `simulation=` is no
  longer needed, and one that disagrees with the run raises `ValueError`.
  Against the full run's box, the pass that brings E onto the H plane
  agrees to the solver's float32 precision; the pass that brings H onto
  the E plane reads the image face at its outer sample, where the full run
  reads its far-side face at the inner one, so the mean of the two differs
  by the readout's second-order error: about 0.1 to 0.4 % of the peak
  intensity on FDTD dipole boxes at 16 cells per wavelength (three cases,
  two symmetries, uniform and graded meshes), falling at about second
  order (0.04 to 0.06 % at 32). Neither the reduced nor the full run is
  systematically closer to the exact pattern. A symmetry-reduced
  box that would fall back to the earlier readout (faces with one normal
  sample, or plain arrays of a graded mesh) is refused, naming the faces:
  that readout cannot add the mirror image, and the box stayed open on the
  far side. Notebook 38 (symmetry-reduced, graded)
  now uses this readout: the share of the escaping light within 10
  degrees of the normal moves from 0.4813 to 0.4812 (48 % as printed, read
  on one mesh and not shown converged).
- **The GPU DFT kernel does less index arithmetic.** On every GPU run, one
  GPU or several, the kernel that accumulates DFT and flux monitors now
  locates its work once per tile of 2048 cells instead of once per cell. It
  only changes which thread updates which element. On one GPU model a test
  scene's outputs were the same bit for bit as before, and its one-GPU
  stepping speed did not change measurably.
- **Runs on several GPUs are built to waste less of each card.** Time
  probes are gathered on the cards and read back once per 256 samples
  instead of one blocking copy per probe per sample, and the update launches
  alternate between cards. Each change only reorders work. On one GPU model
  a test scene's outputs were the same bit for bit as before, and the same
  across one, two and six GPUs; on a second GPU model the GPU test suite
  passed on one GPU. The speed gain on a public scene is not yet measured.
  Such a run now reports, in a
  `multi_gpu_setup` event, which card holds which part of the grid and how
  each pair of neighbouring cards is linked and copies (directly or through
  the host), and its `done` event gives the host time per phase of the step.
- **Runs on several GPUs split the grid by work, by default.** Such a run
  now splits the grid along z by the work each part carries (PML layers,
  materials, DFT and flux planes) instead of into equal plane counts.
  Outputs stay the same bit for bit across GPU counts and across the two
  splits. With auto-shutoff on, the energy check adds up its parts in a
  different order, so a run can stop one progress interval earlier or later
  than under the old default. This is one measurement on one GPU model: a
  synthetic photonic-crystal slab on six GPUs stepped 5.8 % faster with the
  new split (66.02 against 62.38 Gcells/s, setup time excluded, one run per
  split). No other GPU model or GPU count has been measured, and other scenes
  may gain less. A part can now hold many more planes than an equal share, so
  a run that fit its cards with equal parts can need more memory on one card.
  To get equal plane counts back, for example when a run runs out of GPU
  memory, set `PHOTONHUB_Z_PARTITION=equal` in the solver's environment;
  `PHOTONHUB_Z_PARTITION=cost` names the new default. The `partition` key of
  the run's `multi_gpu_setup` event names the split used.
- **Cloud GPU calls name the GPU and the solver build themselves.** The
  managed service runs only a curated `gpu:<id>` and a solver commit it
  approved, and never picks either, so the documented short forms such as
  `ph.cloud.run_quoted(sim)` stopped at the quote with HTTP 400. Now a bare
  `device="gpu"`, every cloud call's default, becomes the GPU `ph.gpus()`
  lists when it lists exactly one; and `solver=None` on a GPU becomes the
  `git_sha` of the locally installed solver (the one `ph.find_solver()` and a
  local run use), so a local CPU run and a cloud GPU run name the same build.
  The pair is resolved once per call, or once per `ph.cloud.Batch`, S-matrix
  run, study or layout-component helper call, and the quote and the
  submission carry the same values. Explicit `device="gpu:<id>"` and
  `solver=` are never changed; `device="cpu"` keeps `solver=None`.
  `ph.cloud.preflight` returns the resolved `device` and `solver`. A cloud
  study records the resolved build and resumes with it when `solver` is left
  out, so collecting and adding jobs needs no local solver. **Behaviour
  change** for callers who left `device="gpu"`: with several GPUs on the menu
  the call now stops before any quote with a `CloudError` that lists the ids;
  with an empty or unreachable menu, read in one attempt with a 10 s
  timeout, `"gpu"` is sent as before. On a GPU without `solver=`, the call
  now needs a local release solver, which reports its full 40-character
  build id. A solver built from source reports a 12-character id the cloud
  cannot match, a build from modified sources `<sha>-dirty`, and the
  Workbench's solver no id at all, since it does not answer `phsolver info`
  on its own; with any of these, or with none installed, the call stops
  before any request, names what the solver reported, and says to pass
  `solver=` with the full commit or install a release solver. Explicit
  forms, the only ones 0.1.6 and earlier accept, keep working.
- **`cost_estimate()` plans at $0.20 per Tcell-step.** The default
  `rate_usd_per_tcell_step` was $0.40, so a default estimate now reads half
  of what it did. It stays a planning figure: pass your own rate, and the
  service's quote is the price. Thirteen course notebooks still print the
  old rate in their recorded outputs until they are next executed.

### Breaking

- **`SMatrixPort` names its frame on a `domain=` scene.** New `frame="user"`
  for your own coordinates or `frame="wire"` for coordinates with `origin_um`
  already taken off, which is how earlier versions read them. Where
  `origin_um` is not zero, `plan_smatrix` and `run_smatrix` now raise
  `ValueError` for a port without one, because the two look alike and either
  would run. If you subtracted the origin yourself, add `frame="wire"` and
  your numbers stay as they were. A hand-built scene, at the origin, needs no
  change.
- **Stop subtracting `origin_um` from a `ph.Port` (#631).** `plan_smatrix`
  and `run_smatrix` now read a `ph.Port` in your own coordinates, so a port
  you moved by hand moves twice: remove the subtraction. Such a port looks
  like any other, so nothing refuses it. `plan_smatrix` warns when a port's
  solved `n_eff` is below the highest background index at that port, and
  refuses one past a wall. A port you build new from a read-back one is the
  same case: `ph.Port(held.name, held.center_um, ...)` and
  `ph.Port(**dataclasses.asdict(held))` are new ports in your own coordinates,
  and move twice. A port read back from `sim.ports` belongs to that
  simulation and is in its stored frame. `dataclasses.replace` and
  `copy.deepcopy` keep it so: it stands as it is, an edit of its centre is in
  that frame too, and it keeps its `out_direction`, so a port moved across the
  device needs `out_direction=None`. On another simulation its numbers stand
  as they are, which is the right place only if both share `origin_um`.

### Fixed

- **Notebook cancellation preserves the private channel.** Interrupting a
  cell during a partial reply read or request write now completes cancellation
  without losing message framing. The worker and its saved variables remain
  usable for the next cell.
- **Numbers move: frequency-domain readings on a Bloch axis (#632).** An
  oblique plane wave from `Simulation.with_oblique_plane_wave` read 4 times
  the power of a normal plane wave of the same amplitude, with no warning.
  The source was right: the field in time carried the declared amplitude, and
  the solver normalized the frequency-domain readings of a Bloch run by the
  wrong pulse spectrum. Every run with a Bloch axis had this scale, with or
  without a wavevector, whatever its source: `PowerMonitor` read 4 times the
  power, `ProfileMonitor` twice the field, and so did everything built on
  them (`diffraction_orders` amplitudes and powers, `mode_amplitude`,
  `mode_transmission`, `far_field` fields and intensity, `colocated_fields`),
  and a Gaussian beam launch read 4 times the power of
  the same launch on periodic axes. `value_and_gradient` on a Bloch axis moves
  too. With an absolute objective such as `PointIntensity`, `J` moves by 1/4
  and the gradient by 1/8. With a ratio objective (`ModeTransmission`,
  `BroadbandModeTransmission`), `J` stays and the gradient moves by 1/2. In
  both, the old gradient was twice too large for its `J`. At zero wavevector
  `J` and the gradient now agree with the periodic run to 2e-6 on a waveguide
  scene that rings down. If the pulse reaches a lattice cutoff and the run does
  not ring down, they differ by what is left at the end of the run, and neither
  gradient can be trusted. A Bloch run now reads the scale of any other run:
  at the pulse centre, a plane wave of amplitude `E0` carries
  `E0^2*A*n*cos(theta)/(2*eta0)`. The power over that value, on an empty
  column at 37 cells per wavelength, went from 3.986 to 0.9964 at 0.5 degrees
  and to 0.9965 at 10 degrees, those being the sampling factor `PowerMonitor`
  states. That closed form covers the pulse centre only. Off it, the constant
  wavevector leaves a forward wave `E0*(1+r)/2`, with
  `r = cos(theta)/cos(theta(f))`, and a backward leak `E0*(1-r)/2`, so
  absolute readings there do not follow it: at 30 degrees, 160 and 240 THz on
  a 200 THz pulse read 1.112 and 0.953 of it at `theta(f)`, and at 45 degrees
  1.577 (s-polarized) and 0.879. Keep the band clear of the cutoff
  `f0*sin(theta)`: near it the reading at the pulse centre swings with the run
  length, by about 1 % at 60 degrees with a 20 THz pulse, with a size and sign
  that depend on where the source and monitor sit. Ratios inside one run or
  between two Bloch runs do not move, apart from the gradient of a ratio
  objective above: `R`, `T`,
  order efficiencies and port-to-port S-parameters read the same, and so do
  time monitors, snapshots and the resonance frequencies and Q read from them.
  Remove any factor of 2 or 4 you applied by hand, and re-run anything that
  compares an absolute Bloch reading with a saved one. That includes a local
  study directory: resumed across the upgrade, it reuses every finished record,
  so one table mixes the two scales with no warning. Start a new directory.
  Runs without a Bloch axis do not change, and GPU runs refuse Bloch axes.
  With a solver from 0.1.6 or earlier, the old scale remains.
- **Breaking: `extent_um` on a periodic or Bloch axis raises, and a `domain=`
  cell that differs from the one described now says so (#633).** `extent_um`
  on such an axis was dropped with no warning (walls at -1 and +1 um round a
  0.4 um device gave a 0.4 um cell); it now raises `ValueError` and points to
  `period_um`. A `domain=` scene with a `period_um` that is not a whole
  number of cells kept a rounded cell with no warning: 0.265 um became
  0.256 um on a graded mesh, 3.4 % off. The cell keeps that size, so no scene
  that ran moves its grid or input hash, and a warning now gives the period
  asked, the cell it has and a spacing that keeps the period
  (`ph.Mesh(dl_um=(...))` takes one per axis). A Bloch axis has at least four
  cells, so a shorter period gets the warning too. A scene the solver refused
  could not run for anyone, so its grid may move: that is the fix. Copying the
  spacing now builds the scene, keeps the period and stops the warning, on a
  Bloch axis too. A spacing of its own makes the mesh graded, so a scene that
  needs a uniform mesh takes one spacing on every axis instead: with a TF/SF
  box the solver refuses a per-axis spacing, and
  `subpixel_method="tensor_full"` or `"contour_full"` raises at the build. The
  axis the solver cannot take as a coordinate ladder sets the mesh's base
  cell: a Bloch axis, an axis under four cells, or an axis whose ladder misses
  the seam, which rounding alone can do from about 2,500 cells, more often the
  longer the axis. Only one spacing can be the base cell, so the warning tells
  an axis under four cells beside one of the others to take four, tells two
  Bloch axes to share one cell that divides both periods, and says so when no
  cell near that size does. Two long axes, or a long axis beside a Bloch axis,
  with different spacings can still be refused: the build raises and gives
  that axis's first and last spacing, and a warning says so first when a
  period is off whole cells. A per-axis spacing such as `0.265 / 13` on a
  periodic axis also builds now, where it could raise on the seam spacings.
  The cell is centred on the structures' bounding box, so a shape drawn off
  centre moves the walls: a shape that then crosses a wall, which cuts it and
  does not wrap it, now gets a warning. A `size_um=` scene on a uniform axis
  still rounds with no warning and `sim.size_um` prints the request;
  `sim.cost_estimate().cells_per_axis` shows the cells used (#642).
- **Breaking: a `DesignRegion` that misses a component the scene names on a
  one-cell periodic axis is refused (#634).** A gradient runs with subpixel
  off and a box does not wrap, so with `cells=(0, 1)` on such an axis the
  default quarter-cell faces, 0.25 to 1.25 cells, hold the half-cell sample of
  the E component along the axis and miss the sample at 0 of the two across
  it. In a quasi-1D scene (a 4 um run of 0.02 um cells, 12 PML layers, one
  0.3 um film of eps 4 at 200 THz) no component saw the region, so the value
  equalled the empty scene and the gradient was exactly 0. `value_and_gradient`
  and `build_adjoint_simulation` now raise `ValueError` before any solve when
  the objective's component misses, or every source's does. They warn when
  only some sources miss, and when they cannot tell, as with a mode source or
  a magnetic dipole. A TM scene, with `Ez` alone on a one-cell z axis, sees
  these faces and runs unchanged. The message leads with `cells=(-1, 1)`,
  which holds both samples in every parametrization and leaves such a TM
  scene's result as it was. `box_face_offset_cells=0` holds them too, but it
  moves the faces on every axis, which changes the design: on a TM scene it
  changed the result and the gradient. `validate_spec` still passes such a
  scene: it sees only the boxes, which are valid, and not the region.
- **`to_file` keeps `dft_shutoff=0`; other saved forms still lose it (part
  of #635).** A saved scene lost an explicit zero, so `from_file` gave
  `None`, which a local CPU run reads as the 1e-4 guard: the saved scene and
  the scene in memory followed different stop rules. `to_file` now writes
  the zero, and a script that reloads the file with `from_file` keeps it.
  It is still lost everywhere else. Text from `to_wire_json()` reloads as
  `None`, because records store digests of those bytes, and so does a run
  folder's `sim.json`. The Workbench loses it even from a file that holds
  it: the Run dialog applies the 1e-4 guard and Save removes the value
  (#528, open). Run such a scene from a script. Nothing else moves:
  `to_wire_json`, `to_wire_dict`, `spec_sha256`, a run's `sim.json` and what
  goes to a GPU or the cloud all leave the zero out as before, and every
  other scene's file is byte-identical, golden specs included. No schema
  change: `run.dft_shutoff` has allowed 0 since 1.21.
- **`FarFieldMonitor`'s source warnings measure where the solver runs each
  source.** They measured where it was declared. The solver runs a point
  dipole on the nearest grid point of its field component, an exact tie
  rounded half away from zero, so a dipole declared on the grid node 2
  cells from a face's innermost sample, along its polarization, could run
  half a cell nearer, 1.5 cells from it, with no warning; and one declared
  a little under 2 cells away could be warned about while it ran 2 cells
  away. The warning about a source within 2 cells of a face's innermost
  sample, and on a graded mesh the one about a source within 0.1
  wavelengths of a change of cell size, now take each dipole where the
  solver runs it (the indices `phsolver validate` reports), a plane wave or
  mode source on the grid line it is injected on with the row of its
  correction half a cell before it, and each face of a TF/SF box on its grid
  line with that row half a cell outside the box. A message names where a
  dipole runs when that differs from where it was declared. Rebuild a
  simulation to see whether a placement you kept is now warned about.
- **Numbers move: `plan_smatrix` and `run_smatrix` put a `ph.Port` where
  `Simulation(ports=...)` puts it (#631).** On a `domain=` scene they read the
  port's coordinates from the fitted box's corner, not from your origin, so
  the port landed `origin_um` away from where you drew it, and its mode, its
  outward side and its window all came from that wrong place, with no error.
  Anything run this way on a `domain=` scene is worth repeating.
  `plan_component_smatrix`, `run_component_smatrix` and a call with `sim.ports`
  were right and stay as they were: those ports are held with the origin
  already taken off. The one limit on "the same place" is a port in the half
  a symmetry plane mirrors away, which `ports=` reads through its image and
  `plan_smatrix` refuses.
- **Breaking: a port outside the domain is refused (#631).** A port that was
  not driven could sit past a wall: its plane snapped onto the wall and read
  an unguided mode, with only a warning that called it inside the PML.
  `plan_smatrix` now raises `ValueError` for a plane past a wall, on any axis,
  and for a transverse centre past a wall, naming the symmetry plane when the
  port is in the half it mirrors away. On a periodic or Bloch axis a
  transverse centre wraps and is not checked, but a plane past the wall is
  refused: the plan used to pass it and the solver then refused the run. A
  position on a wall within rounding stands, so a port on a symmetry plane
  away from zero is kept. The messages quote your frame, not the fitted box's.
- **A strided plane and a closed box project without the old averaging
  error (#575).** On a plane recorded with `interval_space`, `far_field`
  averaged samples a stride apart: on an exact Gaussian beam at 20 cells per
  wavelength its intensity error grew from 0.6 % of the peak (every cell)
  to 1.8 % (every fifth cell). It now reads 6e-5 or less at every stride.
  On a closed box the averaging took the sample before each face's first one
  as zero, which read the radiated power up to 1.8 % low (from -1.8 to
  +0.1 % in the dielectric below, -1.4 to -0.5 % in vacuum); a point dipole
  box recorded one cell thick now closes on the near-field flux within
  0.07 % in a dielectric of index 2.57 and 0.03 % in vacuum (box
  half-widths of 12 to 24 cells, 20 to 40 cells per wavelength in the
  dielectric, the same cells in vacuum; uniform mesh, no structures, no
  dispersion, 12 PML layers).
- **`FarField.radiated_power()` and `directivity()` no longer raise
  `KeyError` on some angle grids.** Ordinary full-sphere and hemisphere
  grids were among them, such as theta `np.linspace(0, np.pi, 113)`: the
  solid-angle weights rounded the angles one way to build each axis and
  another way to look them up. An affected grid raised rather than returned
  a number, so nothing needs re-running. Grids that already worked give the
  same results. The one exception is a list of directions with two angles
  within about 1e-12 rad of each other, where one could take its
  neighbour's cell; each now takes its own.
- **Progress events of a run on several GPUs carry the throughput.** They
  reported 0 Mcells/s and no time remaining; they now fill the instantaneous
  and average rates and the estimate the way a run on one GPU does.
- **Numbers move: a simulation file reads back whatever its media are.**
  `Simulation.from_wire_json` and `Simulation.from_file` refused a `Medium`
  with poles, a Drude term, an anisotropic `permittivity_xyz` or a
  `permittivity_data` grid, and a plain `Medium` hid it. `RunResult(path)`
  lost the simulation of such a result, and `run_local` warned that it could
  not write the client state. Both now work, and the wire is unchanged.
  `RunResult(path)` on such a result now reads absolute units where it read
  unit amplitude: a flux moves by the first source's amplitude squared, and
  by 2 for each symmetry plane that cuts it, and a frequency-domain field by
  the amplitude.
- **Breaking: `RunResult` refuses a simulation that is not the one the
  result ran (#596).** `RunResult(path, simulation=sim)` reads with `sim`:
  its grid and frame place the fields, its ports give the modes and its first
  source the amplitude. A `sim` that differed from the run gave wrong numbers and no
  warning, and a script rebuilt on a newer SDK can build a different mesh
  from the same text. `sim` is now compared with the `sim.json` saved with
  the result, or the copy in an HDF5 bundle, leaving out the `run` block,
  which a reading does not use, and with launch amplitudes compared to 0.1 %,
  since a rebuilt launch can move that far. A bundle with no `sim.json`, as
  some cloud backends return, is compared with the spec this client stored
  when it submitted the job, provided that spec is the input the manifest
  recorded. That spec lives in the local cache, so a read on a machine
  without that cache compares only the grid the manifest recorded, and a
  difference elsewhere goes unseen there. A difference raises
  `SimulationMismatchError`, a `ValueError` that names it.
  `ph.cloud.resume(job_id, simulation=sim)` refuses the same way and keeps
  the cached result. `RunResult(path)` reads a result saved by 0.1.6
  correctly with the saved simulation, and with its ports when a client
  record was kept: `run_local` writes one beside the result, and `ph.cloud`
  keeps one in the local cache of the machine that submitted the job. With
  no record, such as a job read on another machine, the ports do not come
  back, and the refusal says so: read them on the SDK that ran the job. Read
  ports through the result (`data.transmission`,
  `data.simulation.port_monitors`): the `port_monitors` of a simulation you
  rebuilt are not checked, and on another grid they read the saved fields
  wrongly. `far_field(data, simulation=...)` compares only the symmetry
  planes with the run, so with `n_medium="auto"` the medium comes unchecked
  from the simulation you pass: pass `data.simulation`. Port modes
  are not compared, so reading another mode from the same
  fields is accepted; a default window follows the mode, so give `window_um`
  to keep the same plane.
- **A refused cloud request says why.** `str(ph.cloud.CloudError)` read only
  the route and status, such as `POST /v1/estimate -> HTTP 400`, while the
  service's reason sat in `.body`. The message now ends with that reason:
  `POST /v1/estimate -> HTTP 400: the managed GPU solver commit must be
  supplied explicitly`. A structured refusal reads as a sentence, for example
  `one managed GPU job per account can be active (1 of 1); wait for it to
  finish` for the 429 concurrency limit, and a nested
  `{"error": {"code", "message"}}` body as its message. An HTML error page
  reads as its text without its style and script, and an object store's XML
  error as its code and message. The reason is one line of at most 300
  characters without control characters (terminal escapes, bidirectional
  overrides); API keys, bearer and signed tokens, signed-URL query strings
  (also JSON-escaped) and signature parameters are redacted, in time linear
  in the body's size. `.status_code`, `.body` and `.job_id` are unchanged.
  Code that matched the exact old message text may need updating.

- **Fitted domains on a graded mesh are smaller where a high-index medium
  crosses a boundary: numbers move.** With `domain=` and
  `Mesh(cells_per_wlen=...)`, the room for each face's `pml_num_layers` or
  `absorber_num_layers` was counted in background cells, but the mesh fills
  it with the cells of the medium that crosses the face. Where a substrate,
  a cladding or a port's guide ran through a face, the domain held extra
  interior medium between the wall and the layers: a 0.22 um silicon slab
  on a silicon substrate in air, 14 cells per wavelength, had 139 cells
  under 40 absorber layers instead of 40. Each face now reserves its layers
  in the cells the mesh puts there, so such scenes need fewer cells (that
  slab: 250 z cells, now 152; a silicon strip in oxide with ports along z
  and 12 PML layers, 20 cells per wavelength: 294, now 260), and a run
  given in `transits` is shorter. The PML or absorber now begins at the
  walls you asked for, nearer the device than before (beside a port,
  `port_margin_um` beyond its plane: by default ten cells of the port's
  medium), so readings of such scenes may move a little. The fit checks the
  built mesh and fits a face again where the layers would begin inside the
  wall by more than the box's rounding and 3 % of their depth; if that box
  does not build, the last one that did stands and a warning names the face.
  A medium or a `Mesh(refine=...)` region that ends near the outer face is
  now left out of the box or taken into it by enough of its cells to keep
  the mesh's spacing-ratio check in nearly every case (the mesher's last
  stretch of the background cells can still tip a region near the check's
  limit over it), at most 64 and never past its far end; a sliver of it
  inside the box could fail that check. A refine region whose cells are 9.82
  to 10 times finer than the background cell (the check's limit is 10) is
  taken in by 64 of its cells, and a warning names it. Creating the
  simulation can still fail the check and raise a ValueError, sometimes
  where 0.1.6 built it: over many scenes this happens less often than in
  0.1.6 up to about 9.95 times finer, and more often closer to the limit.
  Make the region's cells coarser, or end it further from the boundary. A
  `dl_min_um` above the background cell now
  widens the box, so the layers hold that many cells of the floor (they
  held fewer and reached inside the walls). A port's guide that ends inside
  the layers, short of the outer face by more than one of their cells, is
  now continued to the face. The S-matrix driver's warning about a port
  plane inside the layers measures their actual depth. Uniform meshes, and
  the axes a mesh gives a spacing (`Mesh(dl_um=...)`), keep their wire byte
  for byte. A result saved by an earlier SDK must be read with the
  simulation saved beside it: one rebuilt with this SDK can have another
  grid, and `RunResult` refuses it.

## 0.1.6 (2026-10-21)

This release adds broadband and modal inverse design with checked gradients,
shape optimization of guide widths and polygon vertices, an opt-in mode solve
on the Yee lattice, design studies, gdsfactory import, compact-model export,
index maps from temperature or carrier density, more local MCP tools and the
GDS layout reader in the packaged Workbench build. Setup is much faster on
scenes with many structures. It also moves numbers, among them bend and
rotated ring ports, port modes on custom media, modes solved with absorbing
layers, TF/SF box spectra and Bloch magnetic launches. Re-run anything you
compare against a 0.1.5 result.
Entries marked **numbers move** change a reading; several under Breaking do
too. Breaking entries need a change in your code: `optimize` defaults to
L-BFGS-B, a source inside the boundary layers is refused and fourteen names
deprecated in 0.1.2 are removed, among others. The wire schema moves to
`1.22.0-alpha.1`.

### Added

- Inverse design adds `BroadbandModeTransmission`: modal transmission
  `T_k` at several frequencies read from one forward run, with port modes
  re-solved at each frequency and the input port's reading in the same run
  as the incident amplitude, or fixed incident amplitudes read from one
  design-free broadband run (`reference_incident_amplitudes`), reduced to
  one figure of merit (the mean, a weighted sum, a smooth minimum or a
  callable returning the value and its slopes). `value_and_gradient`,
  `parametric_value_and_gradient`, `finite_difference_check`, `optimize`
  and `optimize_parametric` take it at one forward and one adjoint solve
  per evaluation whatever the number of frequencies, and return
  `BroadbandGradientResult`. The adjoint run drives each port sample with
  delayed copies of the forward pulse whose amplitudes make the summed
  current, at every frequency, exactly the single-frequency adjoint current
  scaled by `dF/dT_k`; it runs longer than the forward run by the span of
  those delays, about `1/df` for the smallest frequency spacing `df`. No
  solver or file-format change. `DesignRegion.monitor` takes a sequence of
  frequencies, and a `simulation` for quasi-2D runs (one periodic cell),
  where it records the quarter-cell plane. On a quasi-2D CPU taper (21
  wavelengths from 1.35 to 1.85 um, 0.025 um uniform mesh, 8000 fixed
  steps) and a 3D CPU strip guide (5 wavelengths from 1.45 to 1.65 um, 0.04
  um uniform mesh, 3000 fixed steps), both subpixel off, non-dispersive, PML
  on every open axis and shutoff 0, the broadband gradient of the mean
  matched the sum of the single-frequency gradients to 3.1e-7 and 3.0e-7
  relative L2 and central differences of real solves to 1.6e-3 and 1.5e-4
  (8.8e-5 and 3.3e-6 against their Richardson extrapolation). Launch each
  frequency's own mode (`mode_launch(..., modes_by_freq=...)`) when `T_k`
  should match one run per wavelength: on the quasi-2D taper it did to
  5.2e-4, against 1.3% with the 1.55 um mode launched at every wavelength.
- Inverse design now offers `ModeTransmission`, the same-run S-matrix
  element `|S_out,in|^2` of two modal ports whose four-field monitors are
  cropped clear of transverse PML. Its exact discrete gradient uses one
  forward and one adjoint solve, including ports on different axes or with
  different modes. The incident reading picks up part of a design's
  reflection when the reference mode differs from the solver's discrete
  mode, with an error that changes with the input plane and does not fade
  with distance from the source: on two reflective 3D CPU tests (0.05 and
  0.04 um uniform mesh, subpixel off, non-dispersive Si in SiO2, six PML
  layers on every axis, 3000 fixed steps, shutoff 0, eight and nine input
  planes) the incident power differed from a design-free run by up to 5.7%
  with scalar modes, 0.42% and 1.0% with Yee-staggered modes solved on the
  smoothed cross-section (the `supersample=8` default) and 0.16% and 0.24%
  with Yee-staggered modes solved with `supersample=1`, and the reported
  transmission moved by 6.4% and 12.4%, 1.7% and 1.4%, and 0.11% and 0.16%
  across the planes. Constructing the objective warns when a port mode is
  not Yee-staggered or was solved on a smoothed cross-section, and building
  its ports warns when more than 1e-3 of a mode's power lies outside the
  crop. An optional `incident_amplitude`, read with
  `reference_incident_amplitude` from a design-free run, holds the
  denominator fixed and drops the input term, the reflection seen through
  the incident reading, from the gradient; it does not remove the
  plane-dependent error of a scalar or smoothed mode. `GradientResult` includes the forward and adjoint
  simulations; `build_adjoint_simulation` returns the runnable adjoint and
  its coefficient after the same checks as `value_and_gradient`;
  `finite_difference_check` compares the gradient with central differences
  of real solves and records every value and solve length. `optimize` and
  `optimize_parametric` accept the objective. `ModePower` remains a relative
  modal-power objective.
- Shape optimization has a checked gradient. `parametric_value_and_gradient`
  returns the objective, `dJ/dp` for a few shape parameters, the Jacobian
  `drho/dp` it was chained through and the underlying density gradient with
  its forward and adjoint simulations; it still costs one forward and one
  adjoint solve. `parametric_finite_difference_check` compares it with
  central differences of real solves, re-rendering the shape for every
  perturbed solve.
  `WidthProfile` renders a guide's widths along one axis with a soft edge and
  gives the exact Jacobian, which `optimize_parametric(jacobian=...)` now
  accepts; `ParametricResult.grads` records every `dJ/dp`. The edge defaults
  to the larger of two cells and the pixel pitch across the width. With an
  edge of one pitch a width's gradient is exactly zero whenever the edge lies
  on a pixel face and swings from 0 to about 1.9 times its mean across each
  pitch; two pitches hold it within 0.94 to 1.05 of the mean. An `edge_um`
  below the pitch, which leaves a width dead over part of its range, warns,
  and so does a width clipped by the region boundary. Without a Jacobian,
  `optimize_parametric` keeps the central difference of `expand`, which is
  wrong for any density within `fd_step` of a kink of `expand` (`np.clip`,
  `np.maximum`): there it returns a blend of the two one-sided slopes. Both
  paths now check `expand` before any solve: a kink within `fd_step` of the
  parameters warns, and so does a supplied Jacobian that disagrees with
  differences of `expand`. The checks render one parameter at a time, so
  their memory stays within a few times the Jacobian's; `optimize_parametric`
  runs them on its first evaluation only, and `check_jacobian=False` skips
  them.
- Shape optimization can move the vertices of a polygon. `PolygonProfile`
  renders a polygon, optionally mirrored about an axis, onto a
  `DesignRegion` as the exact fraction of each pixel it covers, the
  continuous material averaging of Michaels and Yablonovitch (Opt. Express
  26, 31717, 2018), so the permittivity moves continuously with every
  vertex; its parameters are vertex displacements in microns, and
  `jacobian` returns the exact derivative as a sparse matrix. The polygon
  must stay simple: `density` and `jacobian` raise, naming the edges, when
  two non-adjacent edges touch or cross, and `check_simple=False` skips
  that test.
  `parametric_value_and_gradient` and `optimize_parametric` chain a sparse
  Jacobian without densifying it, and their no-solve Jacobian check runs
  one column at a time for it. `curvature_penalty` is that paper's logistic
  radius-of-curvature penalty with its exact gradient. Its radius assumes
  the two neighbours of a vertex are roughly equally far away, so a sharp
  corner next to a distant fixed vertex reads as a gentle bend;
  `PolygonProfile.pad_ends` adds fixed vertices along the end edges of a
  mirrored chain at a given spacing, so the penalty sees the corners where
  a taper meets its fixed guides;
  `optimize_parametric(penalty=...)` subtracts a penalty from the objective
  (adds it when minimizing) and records it per evaluation in
  `ParametricResult.penalty_history`. `DesignRegion(merge_boxes=True)` emits
  the same pixel model as a few merged boxes: the engine checks every
  structure at every sample, and on a quasi-2D CPU test (0.025 um cells)
  a 720 x 360 pixel region took 483 s to set up as one box per pixel.
  `DesignRegion.monitor(simulation=...)` places the design monitor on a
  quasi-2D run's single periodic cell, which the adjoint run now does too,
  so quasi-2D gradients run; 3D specifications are unchanged. On that
  paper's 2D taper (quasi-2D CPU run, 0.025 um uniform mesh, subpixel off,
  non-dispersive, PML on x and y, 12000 fixed steps, shutoff 0, 400 vertex
  displacements) the chained gradient matches central differences of real
  solves on 20 interior displacements, chosen where the render is smooth
  over the stencil, at relative L2 6.1e-4, limited by the differences
  themselves at 2.0e-3. An independent check on 16 further displacements,
  including vertices next to both ends, gives 1.1e-3.
- `VectorModeSolver(..., method="yee")` and
  `VectorModeSolver.from_rectangular_core(..., method="yee")` add an opt-in
  straight cross-section solve on the staggered Yee lattice that the FDTD
  engine steps (Z. Zhu and T. G. Brown, Opt. Express 10, 853 (2002)). Each
  field component sees its own permittivity sample: `from_rectangular_core`
  averages the exact rectangle over a cell centred on each component, and
  `yee_eps_tensor` accepts samples at those positions. The default
  `method="flm"` is unchanged and its results are identical. On a Si 500 x
  220 nm strip (index 3.4757) in oxide (1.444) at 1.55 µm, with a 2.5 x 2.4 µm
  window, electric walls, subpixel tensor sampling, no dispersion and half
  windows, the Yee method at 4 nm cells reads TE0 +1.2e-4 and TM0 -7.6e-5
  from an independent reference solver whose own error bars are ±1.9e-4 and
  ±1.4e-4, where the default method reads +1.9e-3 and +4.5e-3. Its TE0
  converges at second order in the cell size, its TM0 at an observed order of
  1.2 to 1.9 depending on where the core faces fall. A half-window solve
  (`x_min_symmetry`) reproduces the full window's index to 1e-14. The
  returned fields are co-located on the cell centres, like the default
  method's (`yee_staggered=False`), so mode readouts, mode sources and the
  adjoint treat them as they treat default-method modes, and the source
  launch reserved for modes solved on the simulation's own mesh refuses them.
  The co-location leaves `n_eff` and `n_group` unchanged and moves
  `te_fraction` at first order (2.1e-4 for that strip's TE0 at 5 nm cells). A
  half-window mode sets the new `VectorMode.x_min_plane_on_node`; its
  `te_fraction` then equals the full window's and its `modal_power` is half
  of it. `yee_eps_outer` passes the permittivity of the two low-edge faces
  that are no cell's +x or +y face. Bends, absorbing x layers and
  `solve_eme_basis` raise `NotImplementedError` with `method="yee"`; they are
  planned.
- EME adds an interface model for truncated bases:
  `run_eme(..., interface="projection")`, also on `run_eme_band` and as
  `interface_smatrix(..., method="projection")`. Each incidence is matched
  only on the far side's modes, so power the far basis cannot hold is lost as
  radiation instead of being returned to the represented modes. On a
  0.5 x 0.22 um strip (n 3.5 in 1.444) moved sideways by 0.15 um at 1.31 um,
  one guided mode per side on a 0.05 um mesh in a 2.0 x 1.3 um window, the
  default interface reads T = 0.938 and the projection 0.604, the squared
  mode overlap (0.6045). With a complete basis both models give the exact
  interface. The projection drops radiation that would return to the guide
  later, so it reads low at a sharp curvature reversal; use a complete basis
  there. `EMEResult.interface` and `InterfaceDiagnostics.method` record the
  model, and `eme_convergence_report` refuses to compare results built with
  different models. The default is unchanged.
- **Schema 1.22.0-alpha.1** (was 1.21.0-alpha.1). Index change from an
  imported temperature or carrier-density map, CPU solver only. New
  `ph.SpatialMap` (a scalar map on a rectilinear grid in the user's frame;
  an axis given one coordinate is invariant) and `ph.perturbed_medium(base,
  temperature=..., dn_dT=..., electrons_cm3=..., holes_cm3=...)`, which
  returns a custom medium on the map's nodes: thermo-optic
  `n0 + dn/dT (T - T_ref)`, and silicon carrier plasma dispersion and
  absorption at 1550 nm from Nedeljkovic, Soref and Mashanovich (2011,
  default; eqs. (4) and (5), Table 1, 1.55 um row) or the commonly quoted
  1.55 um expressions for Soref and Bennett (1987), with the absorption
  carried as a per-node conductivity. Carrier maps need `wavelength_um`, the
  run's wavelength, within 1.50 to 1.60 um; the run's own band is not
  checked. The wire gains two optional keys on
  `permittivity_data`: `coords_um` places the nodes by their own
  coordinates on any geometry (the edge value holds beyond the outermost
  node), and `conductivity_s_per_m` gives a per-node conductivity. Absent,
  the 1.18 box-spanning grid is unchanged; every SDK wire carries the new
  version string. GPU solvers refuse custom media; subpixel smoothing stays
  off with them. A port may sit inside a heated region (see "Port modes on
  custom media" under Changed). A new guide, "Temperature and carrier
  maps", covers the workflow and its limits. No heat or charge solver.
- `ph.study` adds design studies with parameters, grid, explicit points,
  corners, Monte Carlo and Latin hypercube samplers, local and cloud execution
  through existing batches, deterministic job names, spec-hash deduplication,
  resume, cloud per-job and study spend ceilings, an xarray results table with
  save and load, and a yield helper that reports its sample counts.
  `ph.spec_sha256` returns the digest of a canonical wire document.
  `ph.Batch.run(on_complete=...)` and
  `ph.cloud.Batch.run(max_usd_total=..., on_submit=..., on_complete=...)` add
  completion, spend and submission hooks.
- Cloud batches accept `solver=` on `ph.cloud.Batch.run` and
  `ph.study.run(cloud=True)`, forwarding the selected build on every quote
  and submission, including through `SMatrixPlan` and the layout helper.
- Cloud studies record the selected solver and refuse resume with a different
  solver, so one study directory cannot silently mix builds.
- New optional extra `photonhub[gdsfactory]` (gdsfactory 9.x) and module
  `photonhub.integrations.gdsfactory`. `import_component` turns a gdsfactory
  `Component` and its layer stack, or a list of `GDSLayer`, into extruded
  structures, ports and a background, with a material override map.
  `ImportedComponent.simulation` fits a scene around it and `plot` previews
  it. `plan_component_smatrix` and `run_component_smatrix` plan and run its
  S-matrix study, locally by default. The module imports without gdsfactory.
- Compact-model export. `write_touchstone` writes Touchstone 2.0 with
  `version=2`, and `read_touchstone` reads versions 1 and 2. `smatrix_model`
  builds an `SMatrixModel`: a callable that interpolates the S-matrix in
  wavelength and returns the port-pair dictionary a SAX circuit takes, saved
  and loaded as JSON. `SMatrixResult` gains `metadata`, `to_model` and
  `to_json`. Polar interpolation is the default; an interval switches to
  whole-matrix complex interpolation only when polar interpolation gains more
  than `passivity_atol` above the largest gain at any sample in the model,
  plus a roundoff margin; the sample reference is at least zero.
  Noisy or uniformly gained samples keep polar unless polar interpolation
  still exceeds that threshold, as it can near a zero; gain below it is kept
  and reported by the checks, whose tolerance is unchanged. One bad
  sample raises the threshold for the whole model, so reported interior gain
  can reach the largest sample gain plus the tolerance. Models check samples and
  15 interior points per interval, then
  recheck the selected interpolant. Non-strict mode warns on failed checks;
  `strict=True` refuses non-passive or non-reciprocal samples or intervals.
  Every switch warns once per model with the interval count, worst interval
  index and bounds, and worst significant midpoint amplitude drop in dB
  relative to polar; entries below 10% of their row or column maximum do not
  judge sag. The pi/2 coarse-sampling warning is never suppressed.
  Stored choices avoid repeated selection during calls. Model files use
  format version 2; a version-1 file from a pre-release build loads with
  polar evaluation everywhere. Rebuilding metadata refreshes the SDK version
  and provenance note and replaces stale generated diagnostics. Denser sampling
  helps resolve rapid phase changes but cannot remove zero-crossing phase
  jumps or certify behavior between finite probes. The checks test
  consistency, not accuracy: exported S-parameters inherit the convergence of
  the run and the mesh behind them.
- The local MCP server adds `get_schema`, `capabilities`, `validate_spec`,
  `estimate`, `result_summary`, and `convergence_report`, plus four resources
  for the schema, capabilities, stable issue codes, and agent skill.
  `photonhub.error_codes` holds the issue-code registry, and
  `photonhub.agent_tools` exposes the same checks without the MCP extra.
  `photonhub-mcp --print-skill` prints the packaged skill. The MCP guide and
  `llms.txt` explain the workflow. No MCP tool submits a cloud job or spends
  money. Failures from the six existing MCP tools now return a JSON payload
  with a stable id, severity, message, and fix; the original message text is
  retained inside the payload.
- A long engine setup now reports progress on CPU and single-GPU runs (not
  yet on multi-GPU runs). Once setup has run 15 s, the solver
  writes `setup_progress` events (stage, units done and total, percent,
  estimated time left) to its event stream and `--log-file`, and `run_local`
  draws a setup status line. Readers that switch on `"event"` skip the new kind.
- **The packaged Workbench build includes the GDS layout reader.** Import
  GDS needs the reader to list a file's cells and layers and to turn the
  layers you pick into editable structures; without it, both steps stop with
  a missing-reader error. The desktop build now bundles the reader and stops
  if the packaged app lacks it. This has not been checked on a packaged
  build. Workbench imports keep to the size and nesting limits of
  `import_gds` (see "`import_gds` limits flattened layouts" under Breaking)
  and report a structured error above them. Only polygons become structures:
  a waveguide drawn as a GDS path is not imported.

### Changed

- Engine setup is much faster on scenes with many structures. Each material
  sample now checks only the structures near it, so setup grows with the
  structure count instead of its square. The material grid is bit for bit the
  same. On a laptop, a metasurface lens with 890 structures set up in 9.5 s
  instead of 538 s, and 7,201 pixel boxes with subpixel off in 0.04 s instead
  of 50 s.
- `optimize` and `optimize_parametric` with L-BFGS-B no longer spend two
  evaluations (four solves) before the optimizer starts. A one-step probe
  used to rescale the gradient, and SciPy then evaluated the start again.
  The adjoint gradient is the exact derivative of the evaluated objective,
  so SciPy now receives `J` and `dJ/dx` divided by `|J|` at the start,
  which is SciPy's own first evaluation. `history` starts there and holds
  only evaluations SciPy asked for, including its line-search trials. The
  path an optimization takes changes with the scale, so a re-run of an
  earlier L-BFGS-B design ends at a different point.
- **Numbers move: bend port readings.** `library.bend` keeps each port at the
  same arc endpoint, but its mode plane now crosses the guide end. Before, each
  plane cut along the guide. Mode launches and results read from these ports
  therefore change. With a slab normal of `x` (both ports) or `y` (the
  `x`-normal port), a port's thickness lies on the first of its plane's two
  axes in x, y, z order, so its mode family is still named in the port
  plane's frame and `TE0` there is not the slab's `TE0`; this separate known
  defect remains.
- **Numbers move: rotated ring port readings.** For `library.ring` with a slab
  normal of `x` or `y`, both bus ports now use that normal for their mode
  windows and guide extensions. Before, they used `z`; results read from those
  ports can change. The default `z` orientation is unchanged. On a slab normal
  other than `z`, the port's mode family is still named in the port plane's
  frame, so `TE0` there is not the slab's `TE0`; this separate known defect
  remains.
- **Port modes on custom media: numbers move.** Port modes are solved on
  the node values of every custom medium (`permittivity_data`: perturbed
  media, `Medium.from_eps_array` grids and trilinear design regions alike),
  where they used the medium's scalar `permittivity`, so the port modes,
  and the readings normalized by them, of an existing scene with a
  `Medium.from_eps_array` grid or a design region at a port move. Ordinary
  media are unchanged. Inside a doped region the port mode is that of the
  index-shifted core and stays lossless; a transmission read at such a port
  is not yet checked against a reference run.
- **An `eps_of_medium` key that matches no medium now warns.** Every
  `photonhub.analysis` function that takes `eps_of_medium`
  (`solve_mode_on_cross_section`, `mode_bank_on_cross_section`,
  `solve_yee_eme_basis` and the others) ignored such a key without a word;
  each now issues a `UserWarning` that counts the unmatched keys. A key on
  `ph.materials.Si` itself is the common case. A key on a structure outside
  the mode window stays silent. Results do not change.
- Desktop Workbench notebook cells now execute in a separate local process
  without the Workbench cloud key or launch credential, including in sessions
  with Cloud GPU access. A cloud call in a cell needs a key supplied explicitly
  by the user; Workbench Cloud GPU runs remain in the Run dialog. A programmatic
  visualization server still requires `notebook_enabled=True` for cells. Its
  cells also move to the separate worker and no longer inherit the server
  process's `PHOTONHUB_API_KEY`. Cells can access the same user's files and
  may inspect the app's local server process where the operating system permits it.
  Output from programs a cell starts and raw writes to standard streams are
  not shown in the notebook. `input()` is unavailable. A direct SDK local run
  from a cell in the installed app points to `wb.run()` for local execution.
  Long cell output is truncated with a note, while the cell keeps the status
  its code produced.
- Desktop Workbench now requires a native confirmation showing the actual
  quote, device, chosen maximum spend, cell and step counts, and short
  simulation ID before each paid Cloud GPU submission. Declining the
  confirmation sends no run request and requires a new quote.
- The one-call waveguide size error now reports its grid and resolved `dl_um`
  spacing, and names only window settings that can reduce that grid.
- **Message wording changes; no numbers move.** Solver validation errors and
  warnings, and the SDK's docstrings, errors and warnings, no longer cite
  internal documents or engine source files. Each message states its rule
  directly. Code that matches on message text may need updating: for example,
  the ADE stability error now reads "ADE stability" where it cited a section
  number. The `section` of each entry in `photonhub.capabilities.CPU_ONLY_FEATURES`
  is now a plain feature name, such as "Bloch boundaries".

### Breaking

- **`optimize` defaults to L-BFGS-B.** Topology optimization now uses
  `method="lbfgs"` by default, as `optimize_parametric` already did, so
  callers that relied on the default get L-BFGS-B instead of Adam. Pass
  `method="adam"` to keep the old behaviour. `step` applies only to Adam: a
  call that sets `step` without `method` now ignores it. An L-BFGS-B run of
  `n_iters` iterations can take more than `n_iters` evaluations, two solves
  each, because a line search may try several points (at most
  `2 * n_iters + 10`). In an optimizer comparison with the exact gradient
  (3D CPU runs, one run per optimizer and problem on one mesh each), L-BFGS-B's
  best transmission led Adam's at equal evaluations from the 6th evaluation on for a 64-pixel
  topology crossing and from the 8th for an 8-width shape crossing. The
  shipped topology examples (`inverse_design_crossing.py`,
  `inverse_design_focusing.py` and `inverse_design_waveguide.py`) pin
  `method="adam"`, the optimizer their recorded results used.
- **A source inside the boundary layers is refused before the run.** A point
  dipole inside the PML or absorber layers ran with a warning, once per
  dipole, although the boundary absorbs the part of the source inside it, so
  the launch is not the one asked for and the recorded spectra cannot be
  trusted. `Simulation.check_runnable()` now refuses such a run, so
  `run_local`, the cloud quote, run, submit and batch entry points, the cloud
  worker and the Workbench's engine preflight all stop it (that preflight now
  applies every `check_runnable()` rule before the solver's own check, as Run
  already did). The error is one
  line per logical source: the thousands of dipoles of one mode launch, port
  or Gaussian beam are one entry naming its index range and plane, each face
  it reaches, how deep, and the move that clears it (positions in your frame,
  also for a domain fitted with `domain=`). Construction still accepts the
  scene, and a symmetry plane is a mirror, not a layer. Graded axes are now
  checked too, their layers counted in realized cells. Every sample counts,
  also a window tail far below the peak: a mode window that reaches the
  layers must shrink (`ph.mode_window_um`) or the domain must grow.
  `run_local(..., allow_sources_in_boundary=True)` runs such a scene anyway
  and warns once per source instead: the spectra are physically meaningless,
  so it is for boundary diagnostics only. The keyword is not on the wire, so
  the cloud refuses these runs. New `Simulation.sources_in_boundary_layers()`
  returns the per-source messages.
- **A Gaussian beam's default window stops at the boundary layers.** With
  `half_w_um` or `half_v_um` left unset, `gaussian_beam_source` (and a
  `GaussianBeam` source) now leaves out the window's tail samples inside the
  PML or absorber layers, as it already left out those beyond the domain, and
  normalizes the power over the samples it places. Such a beam was
  refused by the rule above otherwise. A default window whose samples in the
  layers carry more than 5 % of the peak amplitude is refused instead, since
  the clip would cut the beam itself. An explicit window is never clipped.
  None of the gallery or course beams had samples in the layers, so none of
  their numbers moves.
- **Fourteen names deprecated in 0.1.2 are removed.** Each was renamed in the
  0.1.2 naming pass and has warned since; the 0.1.2 notes planned the removal
  for 0.2, and it comes earlier. Reading one as an attribute (`ph.PolySlab`)
  raises `AttributeError` with the current spelling, "photonhub.PolySlab was
  removed in 0.1.6; use photonhub.Polygon". Python turns that into a plain
  `ImportError: cannot import name 'PolySlab'` for `from photonhub import
  PolySlab`; the table below is the guide there. The wire schema, its type
  names and every keyword argument are unchanged, so saved specs load as
  before. Replace:

  | Removed | Use |
  |---|---|
  | `ph.PolySlab` | `ph.Polygon` |
  | `ph.FluxMonitor` | `ph.PowerMonitor` |
  | `ph.FieldTimeMonitor` | `ph.TimeMonitor` |
  | `ph.FieldDftMonitor` | `ph.ProfileMonitor` |
  | `ph.FieldSnapshotMonitor` | `ph.SnapshotMonitor` |
  | `ph.SimulationData` | `ph.RunResult` |
  | `ph.BatchData` | `ph.BatchResults` |
  | `ph.PermittivityData` | `ph.PermittivityArray` |
  | `ph.UniformGridSpec` | `ph.UniformMesh` |
  | `ph.GradedGridSpec` | `ph.GradedMesh` |
  | `ph.GradedAxisCoords` | `ph.GradedMeshAxis` |
  | `ph.auto_grid` | `ph.auto_mesh` |
  | `ph.run_async`, `ph.cloud.run_async` (and `ph.web.run_async`) | `ph.submit`, `ph.cloud.submit` |
  | `ph.estimate_cost` | `ph.quote` |

  Still deprecated and still working: the `ph.web` module path (use
  `ph.cloud`), the `photonhub.plugins` module path (use `photonhub.analysis`),
  `ph.cloud.run_quoted_async` (use `submit_quoted`),
  `photonhub.analysis.resonance.ResonanceFinder` (use `ResonanceAnalysis`),
  `ph.materials.cSi` (use `Si`), and `ph.GdsLayer` and `ph.TfsfBox` (use
  `GDSLayer` and `TFSFBox`).
- **Modes solved with absorbing layers now follow the waveguide branch.**
  `VectorModeSolver.solve(num_pml=...)` and
  `rectangular_waveguide(bend_radius_um=...)` can return a different mode set,
  order, phase index, group index and loss. Straight layered modes must have
  enough field energy outside the layers, stable real index as layer strength
  halves, and no steadier duplicate field; they are ordered by real index.
  For a 1.2 × 0.8 µm guide of index 2.0 in 1.444 cladding at 1.55 µm,
  with a 0.08 µm tensor mesh, 3.2 × 2.0 µm window and seven x layers at
  strength 20, the first index changes from 1.778446 to 1.801463.
  A guided mode whose real index depends on the layers can be dropped with
  one warning that names its index, interior energy share and movement at
  half strength. Widen the window; thicker layers in the same window do
  not help. A rib slab pair at 2.107872 with share 0.995 still passes the
  straight rule on a 5 × 1.6 µm window: 1.55 µm wavelength, 0.02 µm
  area-fraction mesh, 0.5 × 0.22 µm core on a 0.09 µm slab, indices
  3.4764/1.444, eight x layers at strength 30. This is a known limit.
  Bends refuse `|R| < W/2 + dl`, where `W` is the full width including
  layers and `dl` is one x mesh cell. A bend pair is assigned to at most
  one selected straight mode in the same window and layers, ranked by
  interior field overlap times the square root of its interior energy
  share. Its index, field and per-mode loss come from that one leaky
  eigenpair. For a 0.7 × 0.4 µm guide of index 2.0 in 1.444 cladding at
  1.55 µm, with a 0.025 µm tensor mesh, 5 × 3 µm window and 50 x layers
  at strength 30, the TE index at R=20 µm changes from 1.576325 to
  1.576060, and the TM mode now reports its own loss instead of zero
  (about 768 dB/cm on that window, flagged `loss_unresolved`; 545 and
  593 dB/cm on 7 and 8 µm windows with 60 layers). At R=12 µm, the TE index changes from 1.643338 to
  1.579983 and its group index from about 1.748 to 1.938; at R=15, 13,
  11, 9 and 5 µm, the previously recorded TE indices change by about
  -0.018, -0.046, -0.084, -0.141 and -0.390. Bend loss still requires
  window and layer convergence checks. Bend indices also still carry a
  first-order mesh error, present before this change: on a mirror-symmetric
  guide, +R and -R differ by about n dl/R, and +R sits about n dl/(2R)
  below their mean (1.576060 against a mean of 1.577023 at R=20 µm on the
  0.025 µm mesh above). Its cause is not yet settled. Do not cite a bend
  index until that is resolved. A bend requested without layers
  now auto-adds them at strength 30 and warns; the
  `rectangular_waveguide` bend path no longer inherits its strength-1
  default for those auto-added layers. For straight guides, less than 95%
  of transverse magnetic-field energy outside the layers sets
  `window_limited=True`; a bend inherits that flag from its straight
  partner. The new `loss_unresolved` flag marks bend loss when the radiation
  caustic is too close to the layers or loss changes by more than 10% when
  layer strength halves. One warning per solve explains affected modes and
  any shortage. A recorded microring bend solve, at 1.5688 µm wavelength,
  R=6.775 µm and 0.035 µm mesh with its default window and layers,
  requests 16 modes and now returns two with a shortage warning; the mode
  it selects remains the same. EME refuses a junction between straight and bend
  sections with different absorbing-layer contours: give both sections the
  same layers and strength for a common reaction-overlap contour. Its
  nonfinite-loss guard and the waveguide summary's unavailable-loss label
  remain defensive for caller-provided modes. Straight solves without
  absorbing layers are unchanged.
- **Beam amplitude cutoffs now retain paired source currents.** Gaussian-beam,
  imported-field and thin-lens launches at every `amplitude_threshold` are
  affected, as are all mode-port launches: declared ports, S-matrix runs,
  replicate builds and Workbench builds. Electric source currents were dropped
  far more readily than magnetic ones, causing backward radiation and less
  forward power than requested. Each field is now compared with its own peak,
  paired currents are retained together, and power is normalized over the
  retained samples. At a 1% cutoff, a 1.55 um vacuum beam on a uniform 0.1 um
  mesh with four PML cells has its backward/forward power ratio change from
  1.01 to 0.036. In the recorded edge-coupler launches at a 1e-4 cutoff,
  coupling changes by at most 0.08 percentage points on uniform 150 nm meshes
  and up to about 0.17 percentage points for a 12 x 12 um window; those runs use
  contour subpixel averaging, nondispersive materials and 12 PML cells. At the
  default 1e-6 cutoff, dipole counts grow and wire bytes and recorded wire
  hashes move, while added currents carry at most 2.8e-7 of an electric
  source's squared amplitude. Recorded readings move by at most 3.8e-7 dB on
  the tightest record and 1.8e-3 dB on a -65 dB crosstalk reading. Invalid
  cutoffs outside the finite range [0, 1), all-zero fields and launches with
  only one of the electric or magnetic fields are now refused. Currents with
  exactly zero amplitude are omitted.
- **Programmatic visualization servers disable notebook execution by default.**
  `create_app()` and `serve_viz()` now require `notebook_enabled=True` to run
  notebook cells because other users on the same computer may reach a local
  server and run code. Opt in only when everyone with access to that computer
  is trusted. Desktop Workbench executes cells in the separate worker after
  local authorization, regardless of Cloud GPU access.
- **Helpers on a restored result check whether the saved setup can be kept.**
  `with_auto_boundaries()` and `with_stabilized_pml()` keep the saved coordinates,
  symmetry and ports when their edits leave the boundary layer thickness
  unchanged. They now refuse changes to a boundary kind or layer count that
  need the original domain fit, as does `with_absorber()` when it replaces PML.
  `with_auto_mesh()`, `with_mesh_overrides()` and `with_oblique_plane_wave()`
  also refuse edits to a restored result. Rebuild from the original inputs or
  apply the helper before running.
- **Explicit GDS layers must match shape sidewalls.** `export_gds` now refuses
  an explicit stack when a slanted polygon's sidewall angle or reference plane
  does not match a layer on its slab. A flat or vertical shape offered only a
  slanted layer is also refused. Such calls could export before but would
  re-import as different solids. Add a `GDSLayer` with the shape's matching
  sidewall angle and reference plane, then pass that stack to `import_gds` for
  the round trip.
- **`import_gds` limits flattened layouts.** With `flatten=True`, it now
  refuses a cell exceeding 100,000 polygons or 1,000,000 polygon vertices
  after reference and array expansion, or a hierarchy deeper than 256 levels.
  Labels, paths and polygon properties are discarded before counting or
  flattening because only polygons become structures. For a trusted larger
  polygon layout, raise `max_flattened_elements=` and
  `max_flattened_vertices=`. Workbench reports a structured error. Undefined
  cell references remain skipped.
- **Replication budgets can refuse the first run.** With a finite budget,
  `replicate` raises `ValueError` when the first run does not fit or has no
  usable planning estimate. The convergence loop can return a result with no
  completed resolutions. A simulation with no cost estimate is refused under
  a budget. An admitted run counts toward the budget even if it fails or is
  cancelled; the limit applies to the SDK's estimate, not the provider's bill.
  A NaN, negative, or non-number budget now raises `ValueError` before any run;
  the single-run path previously ignored its budget. New result information
  includes the `estimate_unavailable` stop reason, `budget_detail` and
  `budget_cause` text, `ReplicationResult.spent_usd`, and the `spent_usd`
  amount attached to an exception raised after a run starts.
- **Port medium with a box-spanning grid behind a fitted port: breaking.**
  With `domain=ph.Domain()`, a Port's own `medium` that is a box-spanning
  grid (`Medium.from_eps_array`) other than its guide's now raises when the
  fit extends that port's guide through the wall: the grid has no box of
  its own to span there. Such a scene built before, with the grid spread
  over the extension's short box. Give the Port a medium without a grid,
  drop `medium=` to take the guide's, or place the grid with `coords_um`.

### Fixed

- **TF/SF box spectra: numbers move.** A run with a TF/SF box whose band is
  above the first source's band, and a frequency-domain field or flux
  monitor, now samples fast enough to prevent the box's band aliasing into
  monitored frequencies. In the confirmed case (a 20 THz dipole, a
  200.86 THz box and a 20 THz monitor in a periodic vacuum cell on a uniform
  0.1 um mesh, run until both pulses had passed), the sampling interval
  tightens from 29 steps to 2. The monitored reading was 289 times the
  correct value and now agrees with an every-step reference to within 3e-6.
- **Bloch magnetic launches: numbers move.** A run with a Bloch boundary at
  a nonzero wavevector is affected when it launches magnetic point dipoles,
  alone, in a Gaussian-beam launch or in a mode launch made of phased point
  dipoles. The dipoles
  drove both the requested wave and its mirror image in the wavevector.
  For a magnetic dipole alone, frequency-domain field amplitudes now double
  and power quadruples. For a one-way launch with electric and magnetic
  dipoles, the forward amplitude rises by about a third (measured 1.33
  times), while the backward wave falls from 0.34 of the forward amplitude
  to the level of an ordinary non-Bloch run (0.046 in the measured launch). Resonance frequencies
  and quality factors read from ring-down remain effectively unchanged: in
  the recorded cases, the differences were at most 2e-7 THz and 0.013 %.
  Runs at zero wavevector or without magnetic dipoles do not change. GPU
  runs do not change because these features are not available there.
- **Domain fit, box-spanning custom media at a port: numbers move.** Fixed:
  with `domain=ph.Domain()`, the guide extension the fit adds behind a port
  whose guide carries a box-spanning custom medium
  (`Medium.from_eps_array`, schema 1.18) spread the whole grid over the
  extension's own short box, so the port plane read the far end of the
  profile. The extension now places the host's grid by the host box's node
  coordinates (schema 1.22 `coords_um`): it reads what the host reads at
  the port plane and the host's end-face values beyond it, so such a
  scene's port modes and readings move. Scenes without custom media keep
  their wire, apart from the version string, and their port modes.
- **Domain fit, `eps_of_medium` anchors at a port plane: numbers move.**
  Fixed: with `domain=ph.Domain()`, an `eps_of_medium` anchor keyed on a
  guide's medium missed the guide extension the fit adds behind a port
  wherever that extension carried a medium object of its own: a
  box-spanning grid (see the entry above), or a Port medium given as its
  own object and equal to the guide's, such as a materials-library entry.
  The anchor, keyed on the guide's medium as you passed it or as
  `sim.structures` holds it, now holds on the extensions too, and so at the
  port plane, also on copies of the simulation. A materials-library entry
  such as `ph.materials.Si` is not a key: the simulation holds a Medium
  made from it, so key the Medium `sim.structures` holds. A key on a
  structure's own medium still wins. Anchored solves at such a port plane
  move: a 500 x 220 nm strip of library silicon in silica, anchored to
  n = 3.48 by a key on the Medium `sim.structures` holds, read a TE0 n_eff
  of 2.447974 at 1.55 um at the plane of a port given
  `medium=ph.materials.Si` and now reads 2.452536, what the anchored guide
  reads mid-guide on the same mesh (the fitted mesh at 12 cells per
  wavelength, a 2.0 x 1.4 um mode window, default smoothing). Port modes
  and wires do not change.
- Fixed: the single-frequency adjoint copied the carrier window of a
  broadband launch built with `modes_by_freq` into its currents, so its
  gradient at a frequency away from the first carrier was wrong (cosine
  -0.30 against central differences of real solves on a 3D CPU test, 0.05
  um uniform mesh, subpixel off, non-dispersive, six PML layers per axis,
  2200 fixed steps, shutoff 0). Adjoint currents now use the plain pulse;
  results with a plain-pulse launch are unchanged.
- Fixed: `VectorModeSolver.solve_eme_basis` could keep one box mode twice
  when two shift-invert searches found it (the copies 1.8e-6 apart in a
  0.37 x 0.22 um strip in a 5.15 x 6.22 um closed box, 30 box modes), and
  every interface with that basis then failed with "rank-deficient reaction
  Gram". Copies within 1e-4 now merge, keeping the one with the lower
  residual.
- Fixed: cloud batches refresh stale quotes before submitting queued jobs
  and check refreshed prices against the per-job and remaining total spend
  ceilings. Ambiguous submissions keep their accepted price reserved.
- `ph.Batch.run` now stops starting queued simulations when the call is
  interrupted (Ctrl-C) or an error escapes it; simulations already running
  finish first. Before, the whole queue still ran.
- A last-value display that raises now reports a cell error and keeps the
  notebook's variables available for the next cell.
- Workbench closes uploaded layout files before reading them, so imports can
  open on Windows, and removes the temporary file after either outcome.
- `materials.Si` now cites H. H. Li, J. Phys. Chem. Ref. Data 9, 561-658
  (1980); it gave the year as 1993. The index is unchanged.
- **Workbench checks its local server before it opens.** A private startup
  check confirms that the local server belongs to this app launch before
  Workbench sends it access credentials. Another program listening on the
  same port cannot receive those credentials through the startup check.
- **Automatic boundaries keep explicit choices.** `with_auto_boundaries()`
  leaves PMC and Bloch boundaries in place and changes only open axes.
- **Graded boundary checks use the cells at each face.** A coarse edge can now
  trigger the dispersive PML warning at construction and select the absorber
  with `with_auto_boundaries()`. A fine edge that previously warned can now
  keep PML without that warning.
- **GDS export preserves polygon sidewalls on round trip.** Automatic layers
  carry each slanted polygon's sidewall angle and reference plane, assigning
  separate layers when those extrusion rules differ. Flat and vertical shapes
  keep the same layer numbers, returned stack and GDS file as before when
  exported automatically or against vertical layers.
- **Tall cuts show Polygon and Cylinder material in the right place.**
  `plot_index` and the material contours of `plot_field` could place Polygon
  material or Cylinder wedges on the wrong physical axes when a tall view
  rotated the display. They now sample the shapes in their physical positions.
- **3D scenes show the solver's interior.** `plot_3d` and the Workbench 3D
  view could hide structures beside a symmetry plane and draw a PML layer on
  that plane. They now show the one-sided PML there and skip shapes that touch
  the interior only at an edge. On graded grids, both the 2D PML bands and 3D
  interior now follow the realized mesh cells. Uniform grids without symmetry
  keep their previous bounds, to rounding.
- **Workbench Undo keeps later edits.** Undo after deleting a structure,
  source, port or monitor restores the item without reverting later edits.
  Undo is withdrawn when later edits make the restore unsafe.
- **`photonhub-mcp` parses its command line.** `--help` printed nothing and
  was taken as a result path; it now prints usage and exits 0, also without
  the `mcp` extra. An unknown option exits 2, and a result directory given on
  the command line or in `PHOTONHUB_RESULT_DIR` that does not open exits 2
  with the reason instead of starting the server without it.
- **A mode launch is one entry in the 3D scene.** `plot_3d`, and the
  Workbench 3D view built on it, drew each point dipole of a mode launch or a
  Gaussian beam as its own trace, so one launch added thousands of legend
  entries. Eight or more consecutive point dipoles lying in one plane (within
  one grid cell) are now one marker trace named for the range, such as
  "sources 0-5879 (5880 point dipoles)". A few hand-placed dipoles keep one
  entry each.
- **Workbench flux charts state one unit, the right one.** A flux monitor
  reads power in watts for the sources as declared whenever the result knows
  its simulation, as `PowerMonitor` documents. The spectrum chart's title said
  watts while its y-axis, its banner ("not absolute watts") and its CSV header
  said normalized flux. The spectrum payload (`viz.service.spectrum_values`,
  the MCP `get_spectrum` tool and the Workbench spectrum figure) now carries
  `unit`: `"W"`, or `"per unit source amplitude"` for a result without its
  simulation file. Title, axis, banner and CSV header all follow it, in the
  single-run and the compare view; the compare view warns when the two runs
  differ.
- **An engine preflight clears when the scene is replaced.** A passed
  preflight stayed green after New, a packaged example, Open or an import
  swapped in another scene. It now clears whenever the incoming scene differs
  (key order aside), and an answer that arrives after a replacement counts
  only for the scene it checked.
- **GPU runs with the full smoothing methods: a dipole on a structure's
  boundary injects the right amount. GPU numbers move; the CPU solver is
  unchanged.** With `subpixel_method="tensor_full"` or `"contour_full"`, an
  electric point dipole whose node lies in a cell on a tilted or curved
  boundary, or on the edge or corner of a structure, straight waveguides
  included, now injects with the same coefficient the field update uses in
  that cell, on single- and multi-GPU runs, as the CPU solver always did.
  Before, the GPU used the diagonal smoothed coefficient there, so the source
  was off by up to several times at a single node, in either direction.
  Launches built from dipoles were affected on the GPU wherever their plane
  crosses such a cell, which a launch across a strip or rib waveguide does at
  its corners: a driven port or `mode_launch` on the dipole path
  (`launch="eq_current"`, which the default `launch="auto"` picks for a
  full-vector Yee mode with recorded placement), `gaussian_beam_source`,
  `import_source`, `thin_lens_source` and `equivalence_current_source`. GPU
  results of these scenes move toward the CPU's. Only `tensor_full` and
  `contour_full` were affected: the default `contour` never was, nor were the
  other diagonal methods. Dipoles away from those cells, magnetic dipoles,
  plane waves, `mode_source`, `mode_launch(..., launch="aux")` and an `auto`
  launch that falls back to `ModeSource` keep their GPU results.

## 0.1.5 (2026-09-28)

This release moves numbers. Inverse-design gradients are now exact and
physically scaled, so their size changes by a scene-dependent factor, and
design pixels sit a quarter cell off the mesh. The auxiliary mode launch of a
mode from `solve_yee_mode` or `solve_mode_on_cross_section` now delivers the
power asked for, so absolute powers from it fall by 12 to 15 % (ratios are
unchanged). Local CPU runs guarded by
`dft_shutoff` can stop earlier. Entries that move a reading say **numbers
move**; entries that need a change in your code say **breaking**. The wire
schema stays at `1.21.0-alpha.1`.

### Changed

- **Inverse-design gradients are exact and physically scaled.**
  `value_and_gradient`, `optimize` and `optimize_parametric` now return
  `dJ/drho` from one forward and one adjoint solve with every constant derived
  from the simulation: the leapfrog frequency `(2/dt) sin(pi f dt)`, vacuum
  permittivity and the solver's source-normalized frequency-domain fields. The
  fitted `BETA` and `BETA_MODE` phase constants are gone. Before, the gradient
  pointed the right way only to a cosine of 0.987 to 0.992 against central
  finite differences, and its size was off by a scene-dependent factor of about
  1e2 to 4e4. On four 3D CPU scenes (uniform mesh at 0.04 or 0.05 um, subpixel
  off, no dispersion, 6 PML layers per axis, `shutoff=0` with a fixed window)
  the cosine is 0.99999999 or better and the norm ratio within 2e-4 of 1.
- **Numbers move: design pixels move by a quarter cell.** A `DesignRegion`'s
  pixel boxes are shifted by 0.25 mesh cell on each axis by default, so no Yee
  E sample lies on a pixel face and each E component takes its permittivity
  from one pixel. Existing designs rendered from the same densities give a
  slightly different structure. `box_face_offset_cells=0` restores faces on
  cell boundaries, with the engine's last-wins ownership of face nodes.
- **Numbers move: the `ModePower` objective.** Its output monitor leaves out
  the transverse PML rows, so the objective value differs from the former
  full-plane overlap where the boundary rows held field. Its adjoint source is
  the transpose of the four-field overlap readout, a cloud of electric and
  magnetic point currents, and Yee-staggered vector modes are accepted.
- **Breaking: `GradientResult` has three new required fields**,
  `forward_steps_run`, `adjoint_steps_run` and `shutoff`: the actual length of
  each solve and the energy shutoff they used. Code that constructs a
  `GradientResult` itself must pass them.
- **Deprecated: `beta=`** on `value_and_gradient`, `optimize` and
  `optimize_parametric` still works and emits a `DeprecationWarning`; omit it
  to use the derived coefficient. `BETA` and `BETA_MODE` remain importable as
  unused placeholders equal to 1. The `beta` field of `PointIntensity` and
  `ModePower` is ignored and warns when set. `assemble_gradient` is the
  low-level step and takes the coefficient as `beta`; its default of 1 gives
  an unscaled gradient.
- **Gradient solves keep the caller's run settings,** including the default
  stop rule. The gradient is exact for the recorded window;
  against a converged gradient its error follows the truncation of the forward
  results. With the energy-only stop at `shutoff` 1e-5 that was about 0.2 % on
  a measured point-source scene, and more on resonant designs. Local CPU runs
  on a solver that supports it also wait for an estimate of the remaining
  change in the recorded results (`run.dft_shutoff`), which is not a
  guarantee; GPU runs stop on the field energy alone. For resonant designs
  and validation, set `shutoff=0` with an explicit `n_steps` or `run_time_s`
  after checking convergence.
- **Objective and gradient use the subpixel-off model.** `value_and_gradient`
  turns subpixel smoothing off for both solves and warns once per call when
  the caller's simulation had it on.
- **Checks before any solve.** Design E nodes or `ModePower` adjoint currents
  inside the PML, a graded mesh, a region pitch different from the mesh,
  symmetry planes, a simulation fitted with `domain=` (its stored frame is
  shifted; give `size_um` and `grid`), and a `build_forward` whose last
  structures are not the region's own pixel boxes at the supplied density now
  raise `ValueError`.
- **Added: an opt-in trilinear custom-medium parametrization**
  (`DesignRegion(parametrization="trilinear")`), CPU only; `device="gpu"` with
  it raises before a solve.
- Temporary solver outputs of a gradient call are removed after the call; the
  returned results keep their monitor data.

### Fixed

- **Numbers move: the `dft_shutoff` stop settles a slab resonator sooner.** A
  local CPU run with the default `dft_shutoff` (`1e-4`) estimates each
  recorded frequency's remaining change from windows of past changes. A decay
  that falls in steps, such as a packet bouncing in a slab, fitted only at a
  few window widths, so the zero-index pillar array ran on toward its 10 ps
  cap long after the field had decayed. The windows now cover every doubling
  from 16 to 8,192 progress blocks. Such runs stop earlier; on a
  12-cells-per-pitch copy of that example the stop moved from step 51,300 to
  41,700 with a recorded error of 1.3e-6 of the band peak, 0.013 of the
  tolerance. Other guarded runs can also stop earlier; their recorded spectra
  usually move well within the requested tolerance, but a fast decay into a
  weak slow tail in one frequency can stop early with a larger error.
- The cap warning of a guarded run now says how many monitor frequencies were
  still settling and names the furthest one, and it points at your own code
  rather than at the SDK. The manifest's `run.dft_hold` and the solver's
  `done` event carry `unsettled_bins` and `included_bins`.
- **`ph.web.WebError`, `WebConfig` and `WebJobTimeout` work in a fresh
  interpreter.** The deprecated `ph.web` alias returned `photonhub.cloud`,
  which has none of the three names, so they resolved only after an explicit
  `import photonhub.web`. `ph.web` now returns that module, with one
  `DeprecationWarning`, and it still answers every other name
  `photonhub.cloud` has.
- **Warnings point at your own line.** A warning raised inside model
  validation (the `GaussianPulse.for_band` clamp of an octave band, the
  dispersive-in-PML advisory) or reached through another SDK function (the
  run-cap warning, readout advisories) named a file inside the installed
  package, so an executed notebook printed that file's absolute path. SDK
  warnings now name the first line of your code on the stack, looking
  through the SDK, pydantic and the import machinery, also for an install
  reached through a symlink. When no line of yours is on the stack (a
  display hook, a job's background thread), they name the SDK function that
  was called. Deprecation warnings that now name your script's line are
  shown by Python's default filters where they were hidden before; the
  aux-line deprecation warns only when you call `mode_source_vector` or
  `mode_source` with a full-vector mode yourself, not when `mode_launch`
  uses that path.
- **A cloud submission always returns its job.** Storing the submitted spec
  beside the cache runs after the paid submission; an unexpected error there
  could escape before the call returned the job, leaving a billing job with
  no handle. It now warns with the job id instead. `ph.cloud.run_quoted` and
  `submit_quoted` submit the same wire document they quoted.
- **Fixed** the auxiliary mode launch delivering 14 to 17 % more power than
  asked for a mode from `solve_yee_mode` or `solve_mode_on_cross_section`.
  `mode_launch(..., launch="aux")`, `mode_source` and `mode_source_vector`
  with such a mode, and the Workbench mode source, scaled the profile on the
  power a scalar-limit magnetic field would carry, while the solver injects
  the mode's true magnetic field. For the TE0 of a 0.5 x 0.22 um silicon
  strip at 1550 nm, a 1 W launch read 1.136, 1.157, 1.164 and 1.170 W at its
  port at 50, 40, 31 and 25 nm cells, growing instead of converging. It now
  reads 1.000 W to within 0.1 % at all four. The same launch also read such
  a mode half a cell off the guide; folded on the strip's two mirrors it
  carried 2.0 % less than without them, and now the same to 0.01 %. With
  `paired_h=False` the port read 0.966 W at 50 nm and now reads within 0.3 %
  of 1 W at every mesh; a broadband source with `paired_h=False` no longer
  keeps the per-frequency true magnetic field it was meant to drop.
  Transmission, S-parameters and every other ratio are unchanged; absolute
  powers from these launches fall by 12 to 15 %. The equivalence-current
  launch (the default for such a mode) and a mode from `VectorModeSolver` are
  not affected. A Workbench mode source
  solved before this release shows as stale and needs solving again before a
  run; results recorded with one stay readable. The packaged mode-converter
  Workbench example is re-solved: its launch power falls by 13.7 % and its
  profile moves by up to 14 % of its peak pointwise (39 % for the minor
  component, the half-cell shift).
- **Fixed** `mode_launch` ignoring `power_watts` for a scalar `Mode` (a mode
  with no magnetic field, from `ModeSolver`). It passed the square root of
  the watts as the peak field, so a 1 W launch of the strip above carried
  2.6e-16 W. The profile is now scaled to the watts; the port reads 1.032 W,
  the 3 % a scalar-limit mode leaves. A broadband scalar launch is scaled at
  its centre frequency.

### Documentation

- **The examples gallery is searchable.** The documentation site's gallery
  filters by words and by topics; every word and every selected topic must
  match, and the choice stays in the page address. Cards stay readable
  without JavaScript. A feed of the examples is published at
  `/docs/assets/examples-feed.json`. Cards no longer carry a status badge;
  each example page keeps its own convergence statement. The "Waveguides"
  and "Mesh studies" topics are gone; the crossing, the optimal bend and the
  valley transport page carry a new "Bends and crossings" topic.
- The valley transport page says the source sits beyond the PML.
- **Eight gallery notebooks stop on field decay alone.** Notebooks 33, 36,
  37, 40, 41, 45, 46 and 49 set `dft_shutoff=0`. Five were re-run on the CPU
  solver at 0.1.4; 36, 45 and 49 were replayed from GPU runs, which stop on
  field decay alone. Every printed number is unchanged.
- **Known limitations.** Both pages list the limits of anisotropic media and
  say that the 0.1.4 GPU solver image passed the GPU equivalence tests. The
  Workbench's page now also lists the refusals of `tensor_full` and
  `contour_full` (graded mesh, PEC, dispersive media, absorber boundaries)
  and a scene the spectrum stop rule can hold to its time cap.
- Course notebooks 13, 14 and 15 have titles that say what each teaches: the
  GDS-to-port-numbers recipe, the split ratio of a narrow-gap coupler, and
  self-imaging in an MMI.
- A plane wave needs periodic or Bloch transverse axes, and an oblique one
  needs Bloch axes; the quickstart said periodic only.
- **Numbers move: example 39 reads the whole TE3 channel.** The mode and
  polarization multiplexer page quotes the loss of the TE3 channel, with
  bends measured on the paper's figure: 0.407 dB at 1550 nm at 16 cells per
  wavelength (0.1.4 printed 0.107 dB for the TE3 target mode). A mesh ladder
  to 32 cells reads 0.344 dB, not shown converged; the paper gives about
  0.4 dB. Graded mesh, contour subpixel smoothing, constant indices, 12 PML
  layers, one mirror plane.
- Examples 35 and 43 quote their mesh ladders, extended on the 0.1.4 solver,
  each "not shown converged"; their printed numbers do not move.
- The course notebooks name no other solver, use no em-dashes, and print
  warnings without a temporary file path. Course 13's check-yourself number
  follows its re-executed output (about 0.005, was about 0.001).
- The known-limitations pages describe the `dft_shutoff` stop as it now is,
  and drop the warning-path item this release fixes. The `PlaneWave` and
  `uniaxial_medium` docstrings say what the solver enforces, and
  `mode_launch`'s says the inverse-design adjoint is exact.

### Repository

- The GPU runner (`scripts/gpu_run.py`) tries three rental providers in a
  fixed order and moves on only when one refuses for capacity and nothing was
  created. `scripts/gpu_promotion_evidence.py` writes the record a stable GPU
  image tag needs before it moves.
- `scripts/package-headless-solver.sh --closed-test` builds a headless
  archive for a closed test under the draft legal texts,
  stamped closed test in its name, README, notice and `provenance.json`.

## 0.1.4 (2026-09-27)

This release moves numbers. Several fixes change what a port, a flux plane or
a boundary reads, most on coarse meshes and on simulations with symmetry
planes. Re-run anything you compare against a 0.1.3 result. Entries that move
a reading say **numbers move** and give the size of the move. Entries that
need a change in your code say **breaking**. Ports, `ph.Domain`, `ph.Mesh`
and `ph.GaussianBeam` are new in this release; the fixes to them matter only
if you used a pre-release revision from the source mirror. For what is still
open, see
[Known limitations](https://leapfield.ai/docs/guides/known-limitations/).

### Highlights

- **Ports.** `ph.Port` names a plane, a guide and a mode.
  `Simulation(ports=..., source="in", wlens_um=...)` solves each port's mode,
  launches the driven one and records every port.
  `RunResult.transmission(name)` and `RunResult.reflection()` read them back as
  spectra.
- **Setup by declaration.** `ph.Domain` fits the box around your structures,
  `ph.Mesh` builds the uniform or graded mesh and `RunSpec(transits=...)` sets
  the duration. Coordinates stay in the frame you drew in. `symmetry=` reduces
  a symmetric device to the half the solver steps, and every view and readout
  shows the whole device.
- **A stop rule that watches the spectrum.** Local CPU runs with frequency
  monitors now also wait for each recorded frequency to settle
  (`run.dft_shutoff`, default `1e-4`), not only for the field energy to decay.
- **Port power equals the solver's flux.** The modal power of a port, and the
  power a port launch delivers, now match what a flux plane reads. A
  `power_watts=1` port launch reads 1.000 W on its own port. **Numbers move**
  where ports differ in cell width or in mode.
- **Whole-device power with symmetry planes.** Every power the SDK reports or
  takes is the power of the device without the planes. **Breaking; numbers
  move** by 2 per symmetry plane for flux planes and beams.
- **Watts, not per-unit-source values.** `PowerMonitor` reads watts and field
  monitors read V/m for the sources as declared, for local runs, reloaded
  results and resumed cloud jobs alike.
- **Port mode windows follow a rule.** The default window is sized from where
  the mode has fallen 30 dB below its peak. `ph.mode_window_um` gives the same
  rule by hand. **Numbers move** on default windows.
- **Cleaner subpixel smoothing.** Polygons now smooth like boxes, touching
  pieces of one material smooth as one body, and `contour_diag` stays within
  the permittivities of its materials. **Numbers move** for polygon devices.
- **Boundaries that hold under refinement.** The automatic PML of a dispersive
  scene and the adiabatic absorber no longer reflect more as the mesh gets
  finer. **Numbers move** for dispersive scenes on fine meshes and for every
  scene built with `with_absorber()` or `with_auto_boundaries()`, whose slab is
  now two wavelengths thick.
- **Spend-safe cloud calls.** `ph.cloud.run` and `submit` quote first and
  submit only under `max_usd` (default $5). `Job.cancel()` now stops the
  service job. **Breaking.**
- **Results reopen as they were.** A result reloaded from disk, from an HDF5
  bundle or from the cloud reads in the same frame, with the same ports and in
  the same watts as the live one.
- **Solver install command.** `photonhub install-solver` checks and unpacks a
  solver archive and records it, with no environment variable to set.
- **Errors before the run.** `Simulation` and the new `check_runnable()`
  refuse what the solver would refuse, and GPU runs refuse CPU-only features
  before anything starts or bills.
- **Figures that explain themselves.** 2D views carry titles, mesh subtitles,
  monitor names and source arrows, show the whole device across symmetry
  planes and draw long devices
  readably. `plot_comparison` draws a whole result figure. Notebook figures are
  sharp.

### Added

- **Ports as the unit of excitation and readout.** `ph.Port(name, center_um,
  axis, width_um, thickness_um=..., mode="TE0", ...)` names a plane, a guide
  and a mode. `Simulation(ports=..., source="in", wlens_um=...)` solves each
  port's mode on its own grid at construction, launches the driven port's mode
  behind its plane (`source_offset_um`) and records the modal power at every
  port. `RunResult.transmission(name)` reads a port back as a spectrum over `f`
  with a `wlen_um` coordinate, and `RunResult.port_names` lists the ports. A
  declared `ph.GaussianBeam(...)` as `source=` builds the beam launch the same
  way. The wire document and the schema are unchanged: a declarative
  simulation writes, byte for byte, the wire the hand-built `mode_launch` and
  `mode_monitor` pipeline writes. `plan_smatrix` and `run_smatrix` accept
  `Port` values beside `SMatrixPort`. `library.Port` is this class, and every
  library builder records the thickness, the medium and the slab axis on its
  ports. The library crossing's `x+` port reads out as monitor `x_plus`.
- **`RunResult.reflection(port=None)`**: the modal power reflected at the
  driven port, as a spectrum over `f` with `wlen_um`. `R + T = 1` across the
  band is the energy check for a lossless device. Naming a port that is not
  driven raises `ValueError`; drive that port to read its reflection.
  `analysis.reflection` and `analysis.reflection_spectrum` are the hand-built
  forms. They raise `ValueError` when `monitor` and `in_monitor` differ in
  name, normal axis or direction, and, on a simulation that drives a port,
  when `in_monitor` is not that port's monitor. On a simulation that drives no
  port, pass as `in_monitor` the monitor at the launch plane, reading in the
  launch direction. Notebook 49 reads a Bragg grating's stopband with it.
- **The domain, the mesh and the run length by declaration.**
  - `Simulation(domain=ph.Domain(...))` fits the box around the structures in
    the frame they were drawn in: a clearance beyond the geometry on closed
    sides, a margin beyond the outermost launch or readout plane on the sides
    a port leaves through (default ten cells of the port's own cell), the
    boundary layers outside, and the extent rounded to whole cells. The
    default clearance is two thirds of a wavelength in the background; a port
    also gets room for its mode window across its guide. An annular sector
    counts with its own arc's bounding box (a 90-degree bend spans a quadrant,
    not the full circle), so the box around a bend is tight.
  - A port whose guide stops short of the wall gets it extended through the
    boundary layers in its own width, thickness and medium, from the structure
    the port sits on. The extension starts one cell inside the guide, so a
    port plane on a polygon's end face never reads a bare cross-section. A
    `Port` without `medium=` takes the medium of that structure.
  - `origin_um` records where the wire's low corner sits in your frame. Every
    coordinate you write, plot (`plot_index`, `plot_field`, `plot`,
    `export_scene`) or read back from a `RunResult` is in your frame; the wire
    document is unchanged.
  - `ph.Domain(extent_um=...)` gives an axis its interior extent outright,
    centred on the structures, in place of the bounding box and the clearance
    (a membrane that runs past every wall, one period of a cavity array). An
    entry may be a `(low, high)` pair: the two walls at those coordinates of
    your frame, for a substrate below a device and air above it.
  - `mesh=ph.Mesh(cells_per_wlen=..., uniform=False, refine=(...),
    dl_min_um=..., max_grading=...)` builds the graded mesh (`auto_mesh`) or,
    with `uniform=True`, the uniform one, counted at the band centre in the
    highest index. `Mesh(dl_um=...)` gives the spacing directly, as one number
    or a per-axis 3-tuple in which `None` means graded. `refine=` regions are
    in your frame. The boundary layers count each axis in its own cell.
  - `RunSpec(transits=...)` gives the duration as passes of the longest extent
    at the highest index. `run` is optional: without one, or with
    `RunSpec(shutoff=...)` alone, a simulation runs to a cap of 40 transits and
    ends at the auto-shutoff in practice. `run_local` warns when the cap is
    reached with the field still above the shutoff. The wire keeps `run`
    required.
  - A `Structure` or the `background` may carry a materials-library entry
    (`medium=ph.materials.Si`, `background=ph.materials.SiO2`). The simulation
    takes the constant index at the band centre and warns when the index moves
    by more than 0.5 % across `wlens_um`. The dispersive fit stays explicit
    (`Material.medium(band_um=...)`).
  - `size_um`, `grid` and `run_time_s` keep working for a scene built by hand.
  - `ph.library.crossing`, `cosine_taper_crossing`, `spline_taper_crossing`,
    `bragg_grating` and `y_branch` take `lead_um`, the straight guide beyond
    the device body, in place of `arm_length_um`.
- **Symmetry planes by declaration.** On a simulation fitted with `domain=`,
  `symmetry=` describes the whole device's mirror plane through the
  structures' centre, and the simulation reduces it to the half domain the
  solver steps. The structures must be mirror symmetric about the plane (the
  first one without an image is named); a structure wholly in the mirrored
  half is dropped, its image kept. Ports in the mirrored half are read through
  their images (`data.transmission(name)` carries `attrs["mirror_of"]`).
  Monitors crossing the plane are clipped to the kept half and read back whole.
  `plot_index`, `plot`, `plot_field` and `export_scene` show the whole device,
  and `RunResult.wire(name)` returns the recorded half as the solver wrote it.
  A hand-built `size_um` with `symmetry=` keeps its meaning and is never
  reduced.
- **`run.dft_shutoff`**, a second stop condition for local CPU runs with
  field or flux monitors. After the field energy has decayed below
  `run.shutoff`, the run also waits until each recorded frequency's estimated
  remaining change falls below `dft_shutoff` (default `1e-4` on a solver that
  supports it). The estimate assumes one geometric decay per frequency.
  `RunResult.stop_reason` reports `dft_decay_estimate` for a stop that passed
  it.
  - Validation: single-mode sweeps up to Q 50,000 and fixed-mesh CPU cavities
    up to Q about 18,100 met the requested tolerance. It is not a guarantee.
    Two beating modes, interfering modes in one frequency bin and late
    arrivals can still stop early with substantial truncation; one reviewed
    coupled-cavity case had an error 242 times its `1e-4` tolerance. At an
    unlucky beat phase the later stop can be about twice as far off as the
    energy-only stop.
  - Cost: measured single-mode stops took about 12 to 24 amplitude decay
    times. The default cap is 40 transits, and a slower resonance reaches it
    with a warning that names the monitor and frequency holding the run, its
    estimated tail and threshold, and whether the energy rule passed. A
    two-source run took 8,000,000 steps, 14.7 times its energy-only stop, and
    one cavity at `1e-4` went from 2,256,600 to 4,096,700 steps with no
    accuracy gain. An explicit `n_steps` cap also warns when the estimate has
    not settled.
  - Frequencies more than about 5.26 source widths from every source carrier,
    where the drive is below `1e-6` of every carrier, are left out, with a
    warning naming each one.
  - To trade accuracy for time, raise `transits` or `run_time_s`, or set
    `dft_shutoff=1e-3`, or `0` for the energy rule alone.
  - An early CPU stop includes the actual final monitor sample.
  - GPU and cloud runs, older solvers and raw wire files sent to the solver
    keep the energy-only rule, with a warning when you asked for a CPU
    tolerance. The SDK writes the key only for a supporting local CPU solver
    with frequency monitors; an absent key means the energy rule, as the
    published schema says.
- **`ph.mode_window_um(width_um, thickness_um, n_core, n_clad, wlen_um,
  mode="TE0", n_clad_top=None, *, factor=1.2, n_eff=None)`** returns the
  `(half_w, half_v)` mode window of the port-window rule, for
  `Port(window_um=...)`. The pad beyond each core face is `factor` times the
  distance at which the mode's intensity is 30 dB below its peak,
  `factor * ln(1000) / (2 k0 sqrt(n_eff^2 - n_clad^2))`, rounded up to
  0.05 um. With a solved `n_eff`, the default 1.2 keeps the truncation error
  of a port reading under 0.001 dB. Without one, `n_eff` comes from the
  effective-index method (two slab solves), which overshoots a TM mode's
  `n_eff` by about 3 % and leaves about 0.002 dB. `n_core` and `n_clad` may be
  library materials. Pass the longest wavelength the port reads. Measured in
  the SDK's mode solver:

  | Guide | Mode | Window (um) | Field at the edges | Readout error per port |
  |---|---|---|---|---|
  | 500 x 220 nm silicon strip | TE0 | (0.80, 0.66) | -47 dB | 0.0002 dB |
  | 500 x 220 nm silicon strip | TM0 | (1.15, 1.01) | -39 dB or lower | 0.002 dB |
  | 1000 x 400 nm nitride strip | TE0 | (1.8, 1.5) | -43 dB or lower | 0.0005 dB |

- **`Simulation.port_windows_um`**: the mode window each port was solved on.
- **`Simulation.check_runnable()`**, which `run_local` and every cloud call run
  before the solver starts. It checks for at least one source; boundary slabs
  that fit their axis (two slabs, or one on a symmetry axis, fewer than the
  axis's cells); a `PowerMonitor` plane clear of the boundary layers; and a
  `CW` run longer than its ramp. A scene built only to plot, or for a mode
  solve, may break these and still constructs; such a scene needs no source.
- **A `photonhub` command that installs the solver.** `photonhub
  install-solver --url <link> --sha256 <digest>` (or `--archive <file>`)
  checks the digest, unpacks into `~/.cache/photonhub/solver`
  (`$PHOTONHUB_SOLVER_HOME` moves it) and records the binary. `--sha256` takes
  the digest or the line of the `.sha256` file. Downloads stay on https
  through every redirect, and a link that needs a bearer credential reads it
  from `$PHOTONHUB_SOLVER_TOKEN`, never from the command line. `link-solver
  <path>` records a solver unpacked elsewhere, `which-solver` shows which
  binary the client will run and why (`--info` adds its build description and
  warns when the solver home is writable by other users), and
  `uninstall-solver` forgets the record and keeps the files.
- **`ph.ProfileMonitor.plane(name, axis, position_um, wlens_um=...)`**: a
  zero-thickness field monitor on a plane that, with no size given, spans the
  PML-free interior of the domain.
- **`ph.ProfileMonitor.sections(name, axis, n=6, wlens_um=...)`**: `n`
  cross-section planes normal to `axis`, named `{name}_0` to `{name}_{n-1}`,
  spread evenly along the PML-free interior (plane `i` at
  `lo + (i + 1/2) (hi - lo) / n`), so they move with a longer device.
  `export_scene(..., plane=name)` draws the group together, each on its own
  brightness scale.
- **`RunResult.simulation`**: the runners hand the `Simulation` to the result,
  and a result opened from a directory loads the `sim.json` beside it, so
  `plot_field` and `preview` draw structure outlines without `simulation=`.
  Every frequency-domain array carries a `wlen_um` coordinate (microns) beside
  `f`.
- **`analysis.transmission_spectrum`**, the labelled-array form of
  `transmission` (an `xarray.DataArray` over `f` with `wlen_um`).
  `plot_spectrum` and `plot_comparison` accept it. `transmission` still
  returns `{freq_hz: T}`.
- **`ph.GaussianBeam(paraxial=True)`** launches a beam at normal incidence,
  waist on the plane, as its Gaussian profile sampled over the plane: one
  source in the medium's index that reads `power_watts` on a `PowerMonitor`.
  Use it for a beam many wavelengths wide, where the default cell-by-cell
  launch places millions of dipoles.
- **`ph.c0`** (the speed of light, m/s) and **`ph.eps0`** (the vacuum
  permittivity, F/m), from the new `photonhub.constants` module. They are the
  solver's own values: `eps0 = 1 / (mu0 * c0**2)` with the CODATA 2018 `mu0`.
  The SDK now takes its constants from there. `photonhub.materials` and
  `Simulation` used the CODATA literal 8.8541878128e-12 before; the solver's
  value differs by 4e-14 relative, so a conductivity derived from an
  extinction coefficient, and the automatic PML `pml_alpha_max` of a
  dispersive scene, move in the 14th digit. The wire bytes of such scenes, and
  any digest of them, change; the physics does not.
- **`ResonanceAnalysis.run(t_start=..., simulation=...)`** and the same on
  `run_time_series`. `t_start` fits only samples at `t >= t_start`;
  `simulation` alone starts the fit where its last source ends. A fit that
  starts before the last source ends now warns, using the simulation a
  `RunResult` carries when none is passed: fitted from `t = 0`, the strongest
  pole of a pulsed resonator can be the pulse itself.
- **`ph.cloud.resume(job_id, simulation=sim)`** for a job submitted from
  another machine, through a cleared cache, or by a submission whose response
  was lost. It restores the absolute scale and the frame of that job.
- **`conjugate_fields(mode)`** (`from photonhub.analysis.gaussian_beam import
  conjugate_fields`) converts a beam or imported mode to the convention
  `equivalence_current_source` stamps, for a hand-built launch.
- **`photonhub.axis_mirror_mismatch(coords_um, mirror_um)`**: how far a graded
  axis is from its own mirror image about a plane, in microns. Compare it with
  `min(photonhub.graded_primary_spacings(coords_um))` to read it in cells.
  `graded_primary_spacings` is now exported from the package root.
- **`auto_mesh(mirror_axes=...)`** chooses which axes get a mirror-symmetric
  mesh (see Changed), and **`auto_mesh(continuations=...)`** names structures
  that continue another of the same medium along one axis, so their faces
  along it get no mesh node.
- **`PoleFit.max_dl_um`**, like `LorentzFit.max_dl_um`: the coarsest mesh at
  which the fit is stable (`active_axes=2` for quasi-2D).
- **`photonhub.capabilities.CPU_ONLY_FEATURES`**, the features the GPU solver
  does not run in this release (see Fixed), `dft_shutoff` among them.
- **`plot_index(subpixel=..., supersample=...)`** forces the smoothed or the
  hard view and buys a finer fill fraction; **`plot_field(scale="db",
  db_floor=...)`** shows a magnitude in decibels below the slice maximum, and
  `scale="raw"` keeps the recorded values.
- **`export_scene` options.** `plane=[...]` draws several named planes in one
  scene, each on its own brightness scale (one plan view inside each layer of
  a stacked device; the viewer draws such a device translucent).
  `phase_component=` names the recorded component that animates the figure:
  record `Hz` on the plane of a bend or ring and pass
  `phase_component="Hz"`, since a rotation within the plane leaves Hz
  unchanged. `phase_component=("Ey", "Ez")` animates a pair along the major
  axis of the ellipse they trace, for a device that turns the polarization.
  `front_spacing=8` draws the wave fronts that many times further apart on a
  device a hundred wavelengths long, and says so in the caption. A scene
  exported without these options is byte-identical. The still figure is |E|
  either way, and a monitor that also records H no longer adds H to it.
- **Warnings.** At construction: a monitor frequency where the first source's
  spectrum is below 1e-3 of its peak (the normalized spectrum is noise there),
  and a `freq0_hz` whose free-space wavelength is over 1000 times the domain
  (`freq0_hz=1.55` ran to zeros). In the mode solver: a mode still above 5 % of
  its peak on an artificial window wall, excluding a face on a symmetry plane.
  That check tracks `n_eff`, which converges faster than the mode profile; a
  readout that depends on the profile's symmetry, such as a splitter's arm
  balance, needs a wider window than the warning asks for.

### Changed

- **Schema 1.21.0-alpha.1** (was 1.20.0-alpha.1 in 0.1.3). It adds the
  optional `run.dft_shutoff` key, absent or `null` by default; every 1.20
  document stays valid. Other descriptions in the schema were reworded.
- **The default mode window of a `ph.Port` follows the port-window rule.**
  **Numbers move.** The pad beyond each core face is 1.2 times the distance at
  which the mode's intensity is 30 dB below its peak, at the longest
  wavelength the port reads, rounded up to 0.05 um; it was two thirds of a
  wavelength in the background. The cladding is read at the core's faces (a
  sloped sidewall included) and `n_eff` starts from an effective-index
  estimate for a strip, a rib or a quasi-2D slab, which the port's own mode
  solve refines, solving again when it needs a wider window. A cross-section
  the estimate cannot take starts from the old margin. `window_um` stays the
  explicit override. `ph.mode_window_um` gives the rule from the estimate
  alone, with the larger cladding's pad on both sides, so the default can
  differ from it by a 0.05 um step or two. `plan_smatrix` sizes a `Port` by
  the estimate alone. `domain=` leaves room for each port's window across its
  guide; an extent or walls you give are kept, and a window they cut draws one
  warning, as does another guide or layer inside a port's window. What moves:
  silicon strip TE0 ports shrink (0.72 um of pad to 0.55 to 0.60 um in oxide at
  1550 nm); silicon TM0 and silicon nitride ports grow (to about 1 and
  1.4 um), and the fitted domain grows with them. The old pad was too small for
  nitride (its vertical edge sat at -26 dB) and for silicon TM0 (-31 dB).
  Gallery notebooks with default windows (33, 35, 37, 41, 43 and 49) read
  differently when re-run.
- **Breaking; numbers move: one power convention for simulations with
  symmetry planes.** Every power the SDK reports or takes is the whole
  device's, what the same simulation without the planes reads.
  - A `PowerMonitor` read from a result (`data[name]`) is the modeled part
    times 2 for every symmetry plane that cuts the monitor plane (a full
    plane, or a window symmetric about the plane). `.attrs["symmetry_factor"]`
    holds that factor, and `data.wire(name)` still returns the modeled part.
    Before, a flux read 1/2 (one plane) or 1/4 (two planes) of the run without
    the planes. A window wholly on the kept side reads its own region; a plane
    parallel to a symmetry plane reads its own plane, so a closed box of
    monitors balances only with the images of those faces added.
  - On a `domain=` fit, a window with an edge on the symmetry plane, one
    crossing it asymmetrically, and one on the kept side of an even (PMC)
    plane whose edge lies within the first cell's centre are refused. The
    whole device's reading of such a window is not a multiple of anything the
    half records (the last read 10 % low for a 0.7 um waist beam in oxide at
    100 nm cells). A copy re-meshed after the fit that brings such a window
    back raises when the flux is read.
  - Launches follow the same rule. `mode_launch`, which launches declared
    ports and the S-matrix drives, now puts `power_watts / 2` into the modeled
    part per plane the launch is centred on, so a 1 W port launch reads 1 W at
    its own port; the modeled part (`data.wire(name)`) of such a launch reads
    half of what 0.1.3 gave, per plane. `gaussian_beam_source`,
    `ph.GaussianBeam`, `import_source` and `thin_lens_source` now do the same;
    they used to put all of `power_watts` into the half, so the device carried
    2x (4x) the power asked for and every port below such a beam read 2x (4x).
    A `paraxial=True` beam already carried the whole device's power.
    `equivalence_current_source` and `mode_source`, called directly, still
    take the power into the modeled part.
  - Measured on a 0.7 um waist beam in oxide (n 1.44, constant), uniform
    100 nm cells, subpixel averaging on, 8 PML layers, auto-shutoff 1e-6: a
    1 W beam reads 1.0033 W on a full plane 0.2 um downstream with one
    symmetry plane, two or none, the same to 1e-4. On a symmetric 1 x 2
    silicon splitter (graded mesh, 8 PML layers, auto-shutoff 1e-4), the
    input flux reads within 0.7 % of the run without planes at 6 cells per
    wavelength and within 0.35 % at 10; the splitter is not mesh converged,
    and the residual is the same for one plane and two.
  - What moves in the gallery: the edge coupler's coupling ratio (beam and
    output port both on its symmetry plane) returns to the scale of its
    published 88.8 % (TE) and 88.5 % (TM); re-run with every readout and
    mirror fix in this release, it reads 87.0 % (TE) and 90.7 % (TM) at
    100 nm cells. The zero-index pillar array's field maps come out
    1/sqrt(2) lower in absolute value; its index and decay do not move.
- **Breaking: `ph.cloud.run` and `ph.cloud.submit` are spend-safe by
  default.** Without a `quote_id` they quote first and submit, bound to that
  quote, only when it is within the new `max_usd` (default $5) and the
  account's available balance, after the grid check `preflight` applies.
  Nothing is submitted otherwise. `max_usd=None` opts out to an unquoted
  submission. A `quote_id` you already accepted is bound as given, so the
  Workbench and flows that preflight first are unchanged; `max_usd` beside a
  `quote_id` is refused. A script that relied on the old behaviour for a job
  over $5 now gets a `CloudError` naming the quote and the ceiling; pass a
  larger `max_usd`.
- **Breaking: every cloud call defaults to `device="gpu"`.** `run`, `submit`,
  `estimate` and `Batch.run` used to send no device. A quoted submission needs
  one, so `ph.cloud.run(sim, device=None)` now raises `ValueError` unless it
  also passes `max_usd=None` or a `quote_id`.
- **The cloud `timeout=` keyword is now `wait_timeout_s=`** on `run`,
  `submit`, `run_quoted`, `submit_quoted`, `resume` and `Batch.run`. It only
  ever bounded how long the client waits: when it runs out, `CloudJobTimeout`
  is raised and the service job keeps running, and billing. `timeout=` still
  works and emits a `FutureWarning`.
- **`ph.cloud.estimate` raises `CloudError`** for a quote made for a different
  grid (see Fixed) instead of returning its price.
- **Breaking; numbers move: one phasor convention for beam and imported-field
  modes.**
  `gaussian_beam()`, `import_field()` and `thin_lens_beam()` return their
  `VectorMode` in the convention of monitor data (a beam tilted toward +x
  carries `e^{+i kx x}`), so each is directly a `mode_monitor` or
  `mode_overlap` reference. `gaussian_beam()` output is the conjugate of what
  it returned before: code that conjugated it by hand for an overlap must
  stop, and code that passed it straight to `equivalence_current_source` must
  pass `conjugate_fields` of it. `gaussian_beam_source` and `ph.GaussianBeam`
  launches are unchanged to the bit. Mode-solver modes and `scalar_beam()`
  keep their convention. In the test case (vacuum, dl 0.05 um, 10 PML layers,
  w0 0.8 um, a 1 W beam read three cells past the launch plane by a mode
  monitor referenced to `gaussian_beam`) the readback was 0.714 W at 10
  degrees and 0.279 W at 20 degrees, and is now 1.019 W and 1.033 W. Two
  launches fixed with it:
  - `import_source` launched the conjugate of the field it was given, so a
    tilt steered the mirror way and a converging field diverged. A Gaussian
    (1/e radius 0.45 um) tilted 20 degrees toward +x, recorded 0.8 um
    downstream (vacuum, dl 0.05 um, 10 PML layers, a 3 um wide box), sat at
    -0.17 um and now sits at +0.12 um.
  - `thin_lens_source` launched the time reverse of its beam, which diverged
    from a virtual focus behind the launch plane. An NA 0.8 beam focused 1 um
    past the plane, read back three cells downstream in the same box, gave
    0.495 W of 1 W and now gives 0.989 W.
- **Numbers move: `run_eme(enforce_passivity=)` and
  `interface_smatrix(enforce_passivity=)` default to `False`** (was `True`);
  `run_eme_band` gains the keyword. The projection pulled the fundamental
  channel down: on an adiabatic 0.5 to 0.8 um taper (2 um long, dl 0.04 um,
  1.55 um, 48 slabs) TE0 `|S21|^2` read 0.9989 with `num_modes=2` and 0.9956
  with `num_modes=4`, against 1.0000 and 1.0002 raw, and more modes or slabs
  did not remove the bias. `run_eme` now warns when a raw cascade of guided
  ports has `energy_balance(0)` above 1 + 1e-3. Pass `enforce_passivity=True`
  for the former behaviour. Course notebook 11 reads 0.99888 instead of
  0.99850.
- **Numbers move: `ResonanceAnalysis(rcond=)` defaults to `1e-8`** (was
  `1e-4`). Every FDTD probe is real, so each mode has a mirror pole at `-f`,
  outside the fitted window, which at `1e-4` leaked into the in-window pole.
  A float32 ring-down of one Q 1.8e6 mode read Q off by -19 % to +36 % at
  5.5 ps, erratically with the record length, and a mode 1e-3 as strong as a
  neighbour five linewidths away was never returned. At `1e-8` the same cases
  read within 1.1e-6 and 1e-3. On a 10 ps FDTD ring-down of the notebook 44
  cavity, Q moves from +2.0 % to -2e-5 of a 150 ps reference. On a record
  dominated by noise (1e-2 of the peak or more) the lower cutoff returns more
  spurious poles; pass a larger `rcond` there, or `1e-4` for the old
  behaviour. `SpectrumCompleter` keeps `1e-4`.
- **`select_resonances(sort_by=)` defaults to `"amplitude"`** (was `"Q"`), so
  the strongest mode comes first: a pole fitted to noise can carry a higher Q
  than any physical mode.
- **`ResonanceAnalysis.run` raises `ValueError`** when summed monitors start at
  different times, and `freq_window` with `f_min == f_max` raises at
  construction instead of inside the fit.
- **The modal readout checks its `modes_by_freq` keys.** A plane frequency
  with no key within 1e-6 (relative) warns, since it is read against a mode
  solved at another frequency, and one whose nearest key is more than 50 %
  away raises. A set keyed in microns read `T` 1.00 instead of 1.10, and one
  solved 200 nm off the band 0.84 instead of 0.96, silently.
- **Numbers move: the solver reads a simulation file's missing keys the way
  the published schema documents them.** A file without `boundaries`, or
  without one axis of it, runs `pml` on that axis (it ran `periodic`); a file
  with `"subpixel": true` and no `subpixel_method` runs `contour` (it ran the
  first-order `volume` average). Files written by the SDK always carry
  `boundaries`; a hand-written or re-serialized file, and a simulation edited
  with `with_changes(subpixel=True)`, now run what the schema promises. A file
  that relied on periodic walls without saying so is rejected when an axis is
  too small for a PML (a one-cell quasi-2D axis, a small unit cell), with a
  message to set that axis to `periodic`; a plane-wave file is rejected at any
  size. A source or monitor within the PML thickness of a face in such a file
  now sits inside the PML. The solver's `start` event, `phsolver validate` and
  the result manifest state the boundary and symmetry of every axis and the
  subpixel method that ran. The GPU image and the cloud service keep the old
  reading until they are rebuilt from this release.
- **`Simulation` refuses what the solver refuses**, at construction and in
  every `with_changes` and `with_*` copy, each message naming the field and
  the fix:
  - a field or flux monitor frequency more than 12 `fwidth` from the first
    source's `freq0`, which normalizes the spectra, or, under a `CW` first
    source, any frequency other than its carrier;
  - `subpixel_method="tensor_full"` or `"contour_full"` with a PEC structure,
    a graded mesh, an absorber boundary or a dispersive medium;
  - a dispersive medium unstable at the run's time step: every Lorentz pole
    needs `omega0*dt < 2`, and the medium needs
    `eps_inf - sum delta_eps*x/(1-x) - sum (wp*dt/2)^2 >= courant^2` with
    `x = (omega0*dt/2)^2`. The message gives the largest stable `run.courant`
    and the refinement that would do instead. This replaces the warning at
    `omega0*dt > 0.8`, which a Drude pole never triggered: the library
    `Al.pole_fit(band_um=(1.0, 1.6), drude=True)` at dl 20 nm passed and then
    diverged, and now fails up front with courant 0.809. The telecom
    dielectric `lorentz_fit` media clear the bound with a wide margin; a
    `pole_fit` of a dielectric that lands at `eps_inf` near 1 (Si over 1.5 to
    1.6 um with one Lorentz pole) is now rejected at photonics meshes, where
    it diverged before.

  `phsolver validate` rejects the same three, so validate and run agree. A
  document loaded with `from_wire_json` or `from_file` is not checked at load,
  so it can be loaded and fixed; an edit of it is refused until the edit fixes
  the rule.
- **Breaking: `find_solver` looks for the recorded solver.** The order is now
  `solver_path=`, `$PHOTONHUB_SOLVER`, the solver recorded by `photonhub
  install-solver`, `PATH`, then the in-repository build, so a recorded install
  beats a `phsolver` on `PATH`.
- **`Simulation.model_copy(update=...)` on a resolved simulation** (`domain=`,
  `mesh=`, ports, or a run given in transits or not at all) is now
  `with_changes`, the simulation the constructor builds from the same fields.
  On a `domain=` fit, an update of the stored structures, sources, monitors,
  size or grid raises `ValueError` naming `with_changes`, since those are
  coordinates of the fitted box: the copy had skipped the fit, so a structure
  landed at the box corner, inside the PML, with no warning. Renaming the
  structures, or replacing the stored `sources` or `monitors` or the
  `subpixel`, `subpixel_method`, `field_precision` and `dft_precision`
  switches, is still a field copy. Where the stored sources and monitors are
  yours (no `domain=` fit, no ports or `source=`), a later `with_changes` or
  `with_*` helper keeps such a swap, for example after
  `gaussian_beam_source`; otherwise `with_changes` refuses rather than drops
  it, and on a scene with ports or `source=` an edit that changes what the
  port solves read is refused. On a simulation built by hand with a `run`, or
  loaded from the wire, `model_copy` is unchanged.
- **Numbers move: the absorber ramp beyond 40 layers.** Up to 40 layers
  nothing changes. Beyond, the solver holds the ramp's total strength at its
  40-layer value, so a slab of a given thickness is the same ramp on any mesh.
  The fixed strength bounds what the slab can absorb: a plane wave at normal
  incidence in a medium of index n is reflected at least about -139/n dB
  (-96 dB in SiO2, -40 dB in bulk silicon; a guided mode's bound is usually
  lower). In a high-index medium an explicit count well above 40 therefore
  reflects more than before. The GPU image and the cloud engine keep the old
  ramp until they are rebuilt from this release.
- **Numbers move: the default graded mesh of a symmetric device is mirror
  symmetric.** `auto_mesh` builds a mirror-symmetric mesh on any axis whose
  refinement set is mirror symmetric about the domain centre;
  `mirror_axes=None` (the default) detects such axes, `""` turns it off and
  letters such as `"y"` force it. Cells used to be placed from one end of the
  axis, so the two halves of a symmetric device sat differently in their
  cells: on the notebook 37 Y-junction at 16 cells per wavelength the mesh's
  mirror mismatch was 1.08 of the finest cell, and the arms' reference modes
  differed by 3.7e-3 in `n_eff` and their transmissions by 5.1e-3, against
  2e-4 on a uniform mesh. The grading ratio, the `dl_min` floor and interface
  snapping still hold. When an axis is symmetric but its mesh cannot be,
  `auto_mesh` warns. Pass `mirror_axes=""` for the previous mesh.
- **Numbers move: `period_um` centres the device.** A given `period_um` wider
  than the structures now centres the device in the period; it used to start
  the period at the structures' low edge.
- **Names.** The naming convention is written down and the public surface
  follows it; every older spelling is still accepted. `wlen` abbreviates
  wavelength, parallel to `freq`: `wlen_um`, `wlens_um` and `wlen0_um`
  (`GaussianPulse.for_band(wlens_um=..., wlen0_um=...)`,
  `solve_mode_on_cross_section(..., wlen_um)`, `ph.materials.Si.n(wlen_um=...)`,
  the beam, lens, EME and mode-solver builders). `auto_mesh`, `with_auto_mesh`
  and `with_mesh_overrides` take `cells_per_wlen` for `steps_per_wvl`;
  `RunSpec` accepts `num_steps` for `n_steps`; `Cylinder.angle_start_rad` and
  `angle_stop_rad` and `PlaneWave.angle_theta_rad` and `angle_phi_rad` name the
  radian fields (the wire keys are unchanged); `ModeSolver.at_wlen` replaces
  `at_wavelength`; `GDSLayer`, `GDSImportError` and `TFSFBox` are the class
  names (`TfsfBox` stays the class `__name__` because it keys the schema).
  Old keyword spellings are accepted silently in this release; old class,
  attribute and method names warn.
- **2D views describe themselves.** Each carries a title with the quantity,
  the wavelength or time and the cut plane, and a subtitle with the mesh.
  Monitors are named on the scene and every planar source carries a direction
  arrow, read off the launch itself. The permittivity colour bar carries the
  refractive index on its other side, labels its exact ends and marks each
  material. The legend moves to the corner with the least device under it,
  and its swatch is the glyph drawn (a launch plane as a line with its
  arrowhead, a lone dipole as a dot).
- **`plot_index` and `plot` draw a port's mode monitor apart from field
  monitors**: green and dash-dot, spanning the window its mode was solved on,
  labelled with the port's name (`x+`, not `x_plus`). Field monitors stay
  amber and dashed. A `PowerMonitor` with a sub-region window is drawn as that
  window, and a `ModeSource` plane is drawn like a plane wave.
- **2D views show the whole device across symmetry planes by default.**
  `Simulation.plot`, `Simulation.plot_index` and `RunResult.plot_field` take
  `unfold=True`, which mirrors the reduced domain back into the whole device;
  `unfold=False` shows the domain the solver steps. A field is mirrored with
  each component's parity about the plane: an odd plane makes tangential E odd
  and normal E even, with H the dual, and an even plane the reverse. A
  magnitude mirrors unchanged and a phase shifts by pi. Each mirror plane is
  marked with a dashed line and a
  legend entry.
- **`plot_index` and the material boundary on `plot_field` show what the run
  discretizes.** Both follow `Simulation.subpixel`: the hard point sample when
  the run does not smooth, the volume-fraction average when it does. The
  averaged value is the isotropic one; under the tensor methods the component
  along the interface normal is lower, so the view is the tangential half.
  Conductivity is not smoothed, as in the solver. `photonhub.analysis` and the
  Workbench service still read the hard sample.
- **`plot_field` draws the material boundary** as one white contour with a
  dark halo of the same permittivity sample `plot_index` shows, so bodies of
  one material read as one silhouette. It replaced a near-black outline per
  structure, which vanished into dark colour maps and drew every seam.
- **`plot_field` divides by the slice maximum by default**, so the colour bar
  runs 0 to 1 instead of showing a per-unit-source number like `1.7e-5`.
- **Long and thin planes.** `plot_index`, `plot` and `plot_field` keep true
  proportions up to 3:1; past that, the short axis is stretched just enough to
  hold the plot box at 3:1, never more than 3x, and the subtitle names the
  stretch ("z stretched 3x"). Before, a plane more than 8:1 filled a fixed
  box, and a 45 by 4 um plane was stretched about 5x. A plane more than twice
  as tall as wide is drawn with its long axis horizontal, and a figure these
  views create is sized to the plane. The key moves below the picture on a
  plane too wide to hold it in a corner, and the colour bar matches the plot
  box's height, or 1.8 in, whichever is taller.
- **`plot_comparison` draws a whole result figure** without matplotlib calls in
  the notebook.
  - `values` may be `{label: y}`, `{label: (x, y)}` or `{label: spectrum}`:
    several of our series, each in the fixed colour of its place, so a series
    keeps its colour from one figure to the next. A single series still takes
    the axes' next colour.
  - A series of fewer than 12 points is drawn with markers; `markers=` forces
    either way.
  - `stated=(x, y, label)` is a stated value at its own x, drawn as a hollow
    square; several stated lines are told apart by their dash pattern.
  - `model=(x, y, label)` draws an analytic curve as a coloured dashed line.
  - `yscale="log"` or `"linear"` sets the y axis; left unset, the axes keep
    their scale. Any other value raises `ValueError`.
  - A count on the x axis (segments, periods) gets whole-number ticks.
- **Notebook figures render sharp.** Importing `photonhub.viz` inside a Jupyter
  kernel selects retina PNG output. Set `PHOTONHUB_SHARP_FIGURES=0` to keep
  the notebook's own format; `viz.sharp_inline_figures()` does it by hand.
- **`export_scene` draws the simulation's own ports** when `port_in` or
  `port_through` is named and `ports=` is not, including a port a symmetry
  plane dropped, each facing its wall. A call that names no port draws none.
- **The featured figure of a long device.** A device more than six times
  longer than its larger other extent is drawn with its long axis compressed,
  by the smallest of 1.5, 2, 3, 4, 5, 6, 8, 10, 15 or 20 that brings it to
  5:1 or below, with a scale bar along each axis and the factor in the
  caption. A figure of cross-sections is drawn as a cutaway. `export_scene`
  caps each axis of a plane at `max_samples` on its own, so the short axis of
  a long plane keeps full resolution.

### Fixed

- **Numbers move: the modal power of a port over-read the flux** by
  `sec(beta*dl/2)`, where `dl` is the port's own cell along the guide and
  `beta` its mode's propagation constant. The modal power is now the flux the
  solver carries through the port plane: `mode_power` is
  `|c|^2 * P_mode * cos(beta*dl/2)`. The factor cancelled in `transmission()`
  when both ports carried the same mode in cells of one width. It did not
  cancel between ports in cells of different widths (a graded propagation
  axis) or with modes of different effective index (a polarization converter,
  a silicon to nitride transition, a taper). A lossless straight guide on the
  graded mesh read an excess loss of +0.089 dB at 8 cells per wavelength and
  +0.023 dB at 15. The normalized amplitude `c` and `|c|^2` are unchanged, and
  `mode_power / flux` on one plane of a clean guide now reads the modal
  fraction, 0.997, where it read 1.027. Reflections read at one port in one
  mode are unchanged.
- **Numbers move: the absolute power of a cell-by-cell launch** (the default for ports,
  `run_smatrix` and `GaussianBeam`). Such a launch set `power_watts` from the
  continuous Poynting integral of its profile, while the solver carries that
  integral times `cos(beta*dl/2)` of the launch's cell, so with the modal power
  above a `power_watts=1` launch would read 0.963 W at 8 cells per wavelength
  and 0.994 W at 20. It now reads 1.000 W on its port and on a `PowerMonitor`
  below it. Against 0.1.3 a port's own `mode_power` reads about the same, and a
  `PowerMonitor` below a launch reads up to 4 % higher at 8 cells per
  wavelength. Ratios do not change. A paraxial beam,
  `mode_launch(launch="aux")` and a scalar mode's launch keep the continuous
  normalization and so read `cos(beta*dl/2)` of their watt on the flux; the
  `launch="aux"` path also delivers too much power for a separate reason (see
  the mode source entry below). A lossless taper between a 0.45 um and a 0.9 um silicon strip,
  driven from both ends, read `|S21|^2 = 1.0058` and `S21/S12 = 1.0073` before
  these two fixes and reads 0.9996 and 1.0011 at 1550 nm now.
- **Numbers move: the modal power of a port centred on a symmetry plane.** Such
  a port's mode is solved on the modeled half, so its plane records half the
  port's power per plane it sits on, and `mode_power` returned that half. A
  transmission between a port on the plane and one off it read 2x per plane:
  an arm of a symmetric 1 x 2 splitter with its input on the mirror read 0.49
  instead of 0.24 at 1550 nm, and a crossing's cross port read 3 dB high.
  `mode_power`, `mode_decomposition(quantity="power")` and the S-matrix now
  report the port's whole plane. Ratios between ports on the same planes are
  unchanged.
- **Numbers move: `plan_smatrix` and `run_smatrix` launched and read an
  off-centre port at the domain centre**, beside the guide, while solving its
  mode at its own centre. Every multi-port S-matrix with a port off the centre
  was wrong (any arm of a 1 x 2 or 2 x 2 device), whether declared with
  `SMatrixPort` or `ph.Port`. Each port is now launched and read at its own
  centre. Centred ports move by about 1e-4 in `|S|^2`, since their readout now
  sits on the grid the mode was solved on. Ports on `ph.Simulation(ports=...)`
  read through `RunResult` were not affected.
- **Numbers move: the launch standoff of the driven port was counted in cells**
  (ten background cells), so it shrank as the mesh was refined, and the phase
  at which the launch's small non-modal remainder beat with the mode at the
  readout moved with the mesh. On a lossless 400 x 220 nm silicon strip in
  oxide, `transmission()` read -0.0000259 dB at 12 cells per wavelength rising
  steadily to +0.0001591 dB at 32, a loss that grew under refinement.
  `source_offset_um` now defaults to a third of a wavelength in the background,
  and the same ladder reads -0.0000259, -0.0000104 and -0.0000142 dB at 12, 20
  and 28. The readout still oscillates with the standoff itself, by about
  1.4e-4 dB over roughly one vacuum wavelength, so an observable at that level
  wants its own straight-guide control on the same mesh and standoff. A port
  that sets `source_offset_um` is unchanged. The new standoff also re-fits the
  domain: the grid moves under the device by 0.02 to 0.31 cell along the
  propagation axis in the gallery notebooks, and notebook 41's 12-cell bend
  losses move from 0.0029 to 0.0025 dB (optimal) and 0.0140 to 0.0149 dB
  (circular).
- **Numbers move: the mode source placed each field component half a cell off
  its grid position** along one axis, so the launched mode sat half a cell off
  the guide. On a simulation with an even (magnetic) symmetry plane the source
  also lost the row on the plane: a TM strip mode through `mode_launch(...,
  launch="aux")` or the Workbench mode source, at 50 nm cells, carried 11.8 %
  less power than without the plane (0.1 % now), and a `paraxial=True` beam at
  100 nm cells 8.0 % less (0.07 % now). Without symmetry planes the launch is
  a little cleaner: on a 0.5 x 0.22 um silicon strip at 50 nm cells, the power
  sent backward fell from 1.3 % to 1.0 % and the fundamental mode's share of
  the forward power rose by 0.1 to 0.3 %. The TE launch's absolute power on
  this path (the Workbench mode source included), already about 12 % over the
  requested watt, rose by 1.6 %; that excess is still open (see Known
  limitations). A `paraxial=True` beam's power moves by -0.2 % at 50 nm and
  -0.75 % at 100 nm. A Workbench mode source solved
  before this release shows as stale and needs solving again; results recorded
  with one stay readable. The packaged mode-converter Workbench example is
  re-solved (its power moved by 0.05 %).
- **Numbers move: a `PowerMonitor` crossing an even (PMC) symmetry plane read
  high.** The solver gave the row of cells on the mirror full weight, though
  only half of each lies in the modeled half; it now counts that row half, as
  the modal readout does. A `pmc` wall gets the same halving. Measured before
  the fix: +17.4 % for the TE0 mode of a 0.5 x 0.22 um silicon strip in oxide
  on a slab-normal mirror (constant indices, dl 0.037 um, subpixel averaging
  on) and +4.0 % for a 4 um Gaussian beam on a PMC mirror (dl 0.10 um), both
  on a uniform mesh with 12 PML layers and auto-shutoff 1e-7. The over-read is
  about dl/2 over the field's effective half-width, so it changed the shape of
  a convergence ladder, not only its level. Odd (PEC) mirrors were exact.
- **Numbers move: the automatic PML of a dispersive scene reflected more as the
  mesh was refined.** On a scene with a dispersive structure, a PML face and no
  PML settings of its own, `Simulation` raises `pml_kappa_max` to 5 and sets
  `pml_alpha_max` to cure a late-time divergence. That alpha grew as the cells
  shrank: at normal incidence in vacuum (12 layers, read at 153 to 233 THz
  around a 193.4 THz source) the PML reflected -53 dB at a 40 nm mesh, -36 dB
  at 20 nm and -24 dB at 10 nm. It is now `eps0 * 2*pi*f0` at the highest
  source carrier (1.08e4 S/m at 1550 nm), the same on every mesh, and reflects
  -85 to -90 dB at all three, level with the default profile; at half the
  carrier it reflects -58 dB. A dispersive scene without a source is
  stabilized for its declared wavelengths, or 1550 nm. The wire value of
  `pml_alpha_max` changes.
- **Numbers move: the adiabatic absorber reflected more as the mesh was
  refined.** It was a fixed 40 layers, so a finer mesh made it thinner and
  steeper: at normal incidence in vacuum (read at 153 to 233 THz around a
  193.4 THz source) it reflected -16 dB at a 40 nm mesh and -8 dB at 20 nm.
  `with_absorber()` and `with_auto_boundaries()` now make the slab two
  wavelengths thick in the background at the lowest source or monitor
  frequency, at least 40 layers: -52 dB at both meshes, -60 dB in SiO2
  (uniform 40 and 20 nm cells, subpixel off, SiO2 n 1.444, periodic x and y,
  z absorber with ramp order 3, pulse width 40 THz, auto-shutoff off; two
  meshes, not a full convergence study). At 1550 nm that is 3.9 um per face in
  vacuum and 2.7 um in SiO2 on every absorber axis, so enlarge the domain to
  keep the same interior. `with_absorber()` recomputes the count on every call;
  `with_auto_boundaries()` keeps a count the scene sets, and a scene that sets
  `"absorber"` boundaries itself, and Workbench documents, keep theirs.
  `check_runnable` reports a small domain that no longer fits; pass
  `num_layers` (or `absorber_num_layers=`) for a thinner slab.
- **Numbers move: plane waves, TF/SF boxes and mode sources injected a rippled pulse when the
  PML settings were raised**, including the automatic PML of dispersive scenes:
  the source's internal incident line shared the simulation's PML settings and
  re-injected an echo from its far end. On a uniform 20 nm mesh the downstream
  flux deviated by up to 2.7 % in vacuum and 8.2 % for a mode source at an
  effective index of 2.4, growing as the mesh was refined. The incident line
  now always uses the default profile. Scenes on the default PML settings are
  unchanged bit for bit; one on other settings, including the legacy
  `pml_sigma_max=0`, moves (about 1e-4 in the flux of the Fresnel slab example
  on `pml_sigma_max=0`).
- **Numbers move: four cases where subpixel smoothing gave one shape different
  permittivities depending on how it was built**, in the default `contour`
  method and in `contour_diag` and `contour_full`.
  - A `Polygon` took its interface direction at corners and along its top and
    bottom faces from a different estimate than a `Box`, so a rectangle drawn
    either way differed along those edges by up to 2 in permittivity. On a
    400 x 220 nm silicon strip at 12 cells per wavelength, a polygon guide
    whose port extension overlaps it by half a cell read a transmission of
    1.000002; it now reads 0.999978 with a modal reflection of 1.0e-6, the same
    as the guide drawn through the walls. A polygon membrane ending on the
    walls no longer puts anisotropic cells, which a PML amplifies, into its
    boundary layers.
  - Polygons of one material that touch or overlap (a taper and its guide, a
    junction's arms and stem, a rib drawn as a strip over a slab) now smooth as
    their union, as `Box` pieces did.
  - `contour_diag` reads that union at a seam and reads a `Box`'s edges
    exactly, so on a flat face it matches `contour`. The polygon guide above
    read a modal reflection of 5.7e-5 under `contour_diag` and now 2.8e-6.
  - A trench or hole inside a slab, drawn as several polygons, smooths like
    the same shape drawn as one.

  Scenes with polygons also set up 2.5 to 4 times faster (gallery examples 37
  and 33). The Lin and Shi Y-junction at 12 cells per wavelength reads 0.04
  percentage points less in each arm, toward its value at 16 and 20, and its
  reflection falls from 5.0e-4 to 3.3e-4.
- **Numbers move: a readout floor at every port whose guide the domain fit
  continues through the boundary layers.** Smoothing treated the seam between
  the guide and its continuation, one material, as an interface on the guide's
  walls, so one column of the guide's perimeter scattered. On a lossless
  400 x 220 nm silicon strip at 12 cells per wavelength the port read a modal
  reflection of 6e-5 and a transmission of 1.0002 (0.9992 on a longer guide);
  it now reads 1.0e-6 and 0.99998, the same as a guide drawn through the walls.
  Same-material boxes, and a box with one polygon, that touch or overlap now
  smooth as their union. Gallery examples whose guides end at a port plane move
  at the 1e-4 to 1e-3 level in transmission.
- **Numbers move: the graded mesh at the ports of a fitted simulation** placed
  a node on the continuation's inner face as well as on the guide's end face,
  forcing a pair of oversized cells at every port (two 67 nm cells in a
  55.6 nm lattice at 8 cells per wavelength) that reflected about 4e-4 of the
  power per port and cost up to 0.005 dB. A straight guide on the graded mesh
  now reads within 4e-4 dB of lossless at 8 cells per wavelength, as on a
  uniform axis. This changes the mesh of every fitted simulation with ports on
  a graded axis, and its numbers by up to that mesh's discretization error.
- **Numbers move: `contour_diag` gave some cells a permittivity outside the
  range of the materials in the cell**, where a cell centre sits in a sliver
  thinner than the cell (the tip of a gap or a wedge, the top edge of a curved
  sidewall), where an interface passes exactly through the centre, and on
  spheres, cylinders and sloped sidewalls. It read 36.5 at the tip of a wedge
  gap in silicon of 12.08 on a uniform 37 nm mesh, and 0.68 on a silicon
  sphere in oxide of 2.085 whose radius spans 9.4 cells. Those cells now take
  the `contour` value, so every `contour_diag` cell lies within its materials.
  On the GDS benchmark crossing, coupler and MMI (uniform, 25 cells per
  wavelength in silicon, dispersive silicon, 12 PML layers), 120, 339 and 406
  values move by at most 2.6. On the Lin and Shi Y-junction (uniform, 12 cells
  per wavelength, no dispersion, 12 PML layers) each arm moves by less than
  2e-5 and the reflection by at most 2e-6.
- **Numbers move: a `ph.Port` on a periodic axis.** On a one-cell periodic
  (quasi-2D) axis the mode solver walled a window of at least three cells,
  found one barely guided mode and no TE family, and `ph.Simulation` raised
  `requested TE0 but found 0 TE mode(s)`. The mode solve now takes every plain
  periodic transverse axis (no symmetry plane; a Bloch axis keeps its walls)
  as the solver steps it, the whole period closed on itself, so a port's
  window spans the whole period whatever its `width_um` or `window_um`, and a
  one-cell axis returns the cross-section's slab modes. On a period of several
  cells, a side wall in the first or last cell moved `n_eff` by up to 0.16
  (TE0) and 0.12 (TM0) against the same structure shifted by whole cells; the
  placements now agree to rounding (a 500 x 220 nm silicon strip in oxide on a
  30-cell period of 40 nm cells, 1.55 um). A Gaussian beam across a plain
  periodic axis now also places sources on the seam grid line and on the last
  cell of a period whose `size_um` rounds up; an Ey beam on a quasi-2D axis
  launched nothing. On gallery notebook 45 (24 cells per 0.733 um pitch) the
  beam's columns across the pitch go from 23 to 24 and its peak amplitude is
  scaled by 0.978945, sqrt(23/24) for a flat profile. `solve_yee_eme_basis`
  refuses a plain periodic axis by name, and `RunResult.transmission()`
  documents how to normalize a beam-driven run.
- **Numbers move: absolute power in watts.** `PowerMonitor` reads watts and field monitors
  read V/m for the sources as declared; the solver writes values per unit
  amplitude of the first source, and `RunResult` multiplies that back in
  (`A0` on phasors, `A0^2` on flux; the `normalization` and `norm_amplitude`
  attributes say so). A Gaussian beam launched with `power_watts=1` used to
  read `1/A0^2` of a watt (about 1e-16 W in 3D); it now reads 1.00 W on a full
  plane below it. On a one-cell periodic axis the beam normalization also
  counted two padding columns it never stamps, so the beam carried one third
  of the requested power. `convert_to_hdf5` carries `sim.json` so an HDF5
  result reads the same watts. Ratios are unchanged.
- **Reloaded and resumed results read like the live one.** The wire document
  does not carry the user frame of a `domain=` fit, a symmetry plane's mirrored
  half, or the declared ports and wavelengths, so `RunResult(dir)`, an HDF5
  bundle, an executor bundle and `ph.cloud.resume(job_id)` came back shifted
  by `origin_um`, cut in half at a symmetry plane, and with `transmission()`
  raising "declares no ports"; a script that sliced the live result
  (`.sel(x=0.0)`) read another plane after a reload. `run_local` now writes
  `client.json` beside `sim.json`, `convert_to_hdf5` and
  `photonhub.executor` (given `client_state=`) pack it, and the cloud client
  keeps it with its result cache. Ports' modes are solved again on the
  reloaded grid the first time a port is read, so a reloaded `transmission()`
  equals the live one. A record that belongs to another simulation warns and
  restores nothing, and `run_local` removes a leftover record from a reused
  directory. A simulation restored this way refuses `with_changes`;
  rebuild it from your script. `RunResult(path, client_state=False)` reads the
  solver's corner frame. A result written before this release reads as before.
- **Numbers move: cloud jobs read in watts however they are collected.** `ph.cloud.resume(
  job_id)` and the Workbench, opening a job from its download cache, showed the
  solver's per-unit-amplitude values, low by `A0` on phasors and `A0^2` on
  flux, for the same job `ph.cloud.run` returns in watts. The gap was not a
  constant: moving a 2 um beam by half a cell (50 nm mesh, 1.55 um) moved an
  unrestored flux reading by 20 %. The client now stores the wire spec of
  every job it submits and recovers it, used only when it matches the input
  hash the job recorded. A result downloaded before this release reads as it
  did; its arrays name their convention in the `normalization` attribute.
  Ratios such as transmission, R and T were right either way.
- **Numbers move: `analysis.diffraction_orders` sampled the medium `origin_um`
  away from the plane** on every `domain=` simulation. A plane in air over a
  glass substrate (1.55 um, uniform 50 nm mesh, normal-incidence order) read
  `n_medium` 1.45, a forward order power 4.2 % high and a spurious backward
  wave of 3.5 %; it now reads 1.0 and the exact powers. The error depends on
  the case (+20 % at 0.35 um). `propagate_plane`, `focal_scan` and
  `focal_metrics` share the fix.
- **Numbers move: the default `n_medium` of `diffraction_orders`,
  `propagate_plane`, `focal_scan` and `focal_metrics` was `eps_inf`** for a
  plane in a dispersive medium: a `ph.materials.Si.medium(band_um=(1.5, 1.6))`
  fit read n 2.90 instead of 3.48 at 1.55 um. It is now `sqrt(Re eps(f))` at
  each recorded frequency, so `DiffractionOrders.n_medium` and
  `propagate_plane(...)["n_medium"]` are per-frequency arrays for a dispersive
  medium. `n_medium=` also accepts one value per frequency, and any positive
  index.
- **Numbers move: `propagate_plane` and `focal_scan` with `direction="-"`
  propagated upstream.** `direction` is the way the recorded field travels and
  `dz_um > 0` is always downstream; pass a negative `dz_um` to look upstream.
  `export_scene(propagate_um=)` with a negative value now does that, and on a
  fitted run it crops the propagated plane around the shown region (it kept
  one quadrant). `focal_metrics` no longer passes on a scipy covariance warning
  about a value it discards.
- **Numbers move: per-frequency mode sets changed a mode's sign or identity
  across a band.** A two-lobe mode (TE1, TM1) flipped by pi where its stronger
  lobe changed (between 1590 and 1600 nm on a 0.67 um strip), so complex
  S-parameters of such a port jumped by pi within the band. The set is now
  phase-continuous, referenced to the band's lowest frequency, and warns when
  neighbouring frequencies overlap below 0.5. `solve_modes_by_freq`,
  `solve_mode_bank` and `solve_yee_multimode_bank` counted the k-th mode by
  `n_eff` at each frequency, so past a crossing an index named another mode (a
  pure TE1 field read `|c| = 1.0` up to 1600 nm and 0.04 beyond). Modes are now
  counted at the band's middle frequency and followed by field overlap; where
  two modes mix (TE1 and TM0 often do) the set warns. On a band, two extra
  modes are solved per frequency. Complex amplitudes and phases of two-lobe
  modes, and per-index readings past a crossing, change; powers and `|S|^2` of
  the modes ports select do not.
- **EME ports whose fundamental is a degenerate TE/TM pair** (a square or round
  core). The pair comes back as 45-degree mixtures, so `transmission_of("TE")`
  read 4e-8 for a square-core step whose transmission is 1, and
  `transmission` warned of a conversion that was not there. Now
  `transmission_of(<polarization>)` raises there, `transmission` warns that its
  reading depends on the mixing, and both point to `transmitted_power(0)`. The
  pair is detected at any angle with `num_modes >= 2`.
- **Numbers move: the half-window mode solve.**
  `VectorModeSolver.from_rectangular_core` with `x_min_symmetry="pec"` or
  `"pmc"` put its plane half a cell off the width centre, one cell too narrow:
  on a 0.2 x 0.5 um core of index 1.97 in 1.44 at 1.0 um, 40 nm mesh, the half
  solve read `n_eff` 0.023 (TE) and 0.033 (TM) low. It now matches the full
  solve to 1e-10. The half-window mode also reported x centred on the core
  instead of starting at its plane, so `field_dataarray`, `core_fraction` and
  the overlap helpers put it half a window too far toward -x
  (`core_fraction` read 0.0 against 0.72 on a 0.45 x 0.22 um core); it now
  reports the full solve's x, carries its plane in `VectorMode.x_min_symmetry`
  and reads 0.750 there. `te_fraction`, `modal_power` and overlaps cover the
  stored half only, so solve the full window for coupling to a full-width
  field. `x_symmetry` sets only the far wall, and a bend or an x PML on a half
  window raises.
- **Numbers move: the guide extension of a fitted port under a cladding
  block.** The fit took the structure a port sits on to be the first in list
  order containing its centre, while the solver paints the last, so with the
  cladding listed before the guide the fit read the cladding: a nitride upper
  cladding over silicon made each port solve a cladding mode (`n_eff` 1.87
  instead of about 2.5 for a 0.54 um silicon strip). The port now sits on the
  last structure containing its centre. A `Port(medium=...)` whose index at the
  band centre differs from that structure's by 1e-3 or more now warns.
- **Numbers move: a beam launched with no `center_um` on a simulation with
  symmetry planes** landed in the middle of the modeled half. `gaussian_beam`,
  `gaussian_beam_source`, both `ph.GaussianBeam` launches, `import_field`,
  `import_source`, `thin_lens_beam` and `thin_lens_source` now default to the
  symmetry plane, as documented.
- **GPU runs that could only fail.** Bloch boundaries (and so an oblique plane
  wave), a `TFSFBox` source, `permittivity_data` media, `pmc` outer boundaries
  and a `CW` source run on the CPU solver only. The solver rejected them after
  the run had started, on the cloud after a quote was bound and a GPU worker
  provisioned. `run_local` with a GPU device, `ph.cloud.estimate`,
  `preflight`, the submit calls and a cloud `Batch` now raise `SolverRunError`
  first, and nothing is submitted. Run these with `device="cpu"`. On the cloud
  an unset `device` is refused too. The `dft_shutoff` stop rule is also
  CPU-only, but it does not refuse a run: on a GPU or the cloud the SDK drops
  the key with a warning and the run stops on the energy rule.
- **Numbers move on GPU: differences from the CPU solver** (compile-checked;
  the hardware equivalence run is pending). With `device="gpu:all"` on a graded mesh, a
  `pec=True` structure behaved as a dielectric. With `gpu:all` and subpixel
  averaging, the row of a structure ending at an even symmetry plane was
  averaged with the background beyond it. On any GPU, a point dipole on such a
  plane inside such a structure had the wrong strength, a dipole on or inside
  a PEC structure radiated or drove its own node, and a mode-source or
  plane-wave plane crossing a PEC structure wrote into the conductor.
- **Cloud calls quoted and submitted simulations the solver refuses.**
  `ph.cloud.estimate`, `preflight`, `run`, `submit`, `Batch.run` and the
  Workbench quote now call `check_runnable()` first, and the cloud worker
  checks again before it starts.
- **A cloud quote for a different grid was accepted.** `preflight` (and
  through it `run_quoted` and `submit_quoted`), `Batch.run` and `estimate`
  compare the quote's cell counts and time step, when the service reports
  them, with `Simulation.cost_estimate()`, and raise `CloudError` before
  anything is submitted on a mismatch (a service older than the client quoted
  a one-cell periodic axis at four cells). A quote without those fields warns
  and still submits. Given a `quote_id`, `run` and `submit` cannot check the
  grid themselves.
- **A cloud `Job.cancel()` did nothing.** It now asks the service to cancel
  the job and returns the reply. When the job is cancelled the local wait ends
  with `SolverRunError`; a job the service had already finished is still
  downloaded; a refusal raises `CloudError` and the job keeps running. It still
  reaches the service after a wait deadline has passed, asks only once, and
  does nothing once the result was returned.
- **`Simulation.with_changes` and the `with_*` helpers kept the decisions made
  for the original.** A material edit kept the old subpixel default, a plane
  added by an edit kept a client-only key the solver refuses, a plane kept the
  old PML-free interior after the PML grew, a monitor box was not snapped to
  the new grid, and a dispersive scene on a new grid kept the old PML alpha.
  The copy is now built as the constructor builds it, from the monitors as you
  wrote them: `sim.with_changes(x=v).to_wire_json()` equals
  `Simulation(..., x=v).to_wire_json()`. A `with_*` helper on a resolved
  simulation rebuilds it, so on a `domain=` fit the box is fitted again for new
  boundary layers (`with_absorber()` and `with_stabilized_pml()` could leave
  no interior). `with_changes(mesh=...)` and `with_changes(domain=...)` now work
  on fitted and hand-built simulations alike. A document loaded with
  `from_wire_json` keeps the loading contract: nothing is resolved again,
  except that an added monitor is placed and snapped and a grid or size edit
  places every monitor on the new cells. On such a document a `ph.Domain` or
  `ph.Mesh` declaration, a run in transits and `ports=`, `source=` or
  `wlens_um=` raise, since only the constructor resolves those (a size tuple, a
  `UniformMesh` or a `GradedMesh` works). `with_oblique_plane_wave` validates
  its result, accepts `bloch` transverse axes and takes `position_um` in your
  frame on a `domain=` fit. The subpixel-divergence warning no longer fires
  once you set a CFS-active `pml_alpha_max` yourself.
- **Numbers move: a subpixel-on simulation whose `subpixel_method` was never
  set** (a loaded
  document, the Workbench toggle, `with_changes(subpixel=True)`) ran the
  solver's `volume` method while the model reported `contour`: the flux of a
  permittivity-4 sphere differed by up to 4.6 % (uniform, 31 cells per
  wavelength, 12 PML layers, auto-shutoff on; one mesh). With smoothing on, the
  method is now always written.
- **A structure `medium` or the `background` given as a dict, number or string
  was accepted unchecked** (`{"permitivity": 4.0}` loaded and was quoted).
  Both now validate as a `Medium`, a `Background` or a materials-library
  entry.
- **The realized-domain check on a `PowerMonitor` window centre** compared the
  window against the domain's x and y ranges for every plane axis; a y-normal
  window is `(z, x)` and an x-normal one `(y, z)`. A quasi-2D y-normal window
  at the domain centre was wrongly rejected, and a window outside its own axis
  wrongly accepted.
- **`ResonanceAnalysis` and `SpectrumCompleter` raised** `time coordinate must
  be uniformly spaced` on a `TimeMonitor` whose run length is not a multiple
  of its `interval_steps`, because a monitor also records the final step. That
  trailing sample is now dropped with a warning; the resonances are unchanged.
- **Figures.** The source arrow of a mode launch cut by a symmetry plane could
  point across the guide; it points along it (the launch itself was right).
  The scene views drew an absorber band on a symmetry face, which, mirrored,
  ran through the device. The default mode launch was drawn as a blob of dots; it
  is drawn as the plane it covers. A monitor plane could vanish from the cut
  meant to show it; it now matches within half a cell. The featured-figure
  animation of a long device ran backwards once its plane was averaged to 256
  samples an axis (a 59 um transition stepped 183 degrees a sample); an axis
  the wave travels along now keeps at most a quarter wave per sample.

### Repository

- The local gates got stricter. `scripts/regression_gate.py` treats a skipped
  or xfailed test as no evidence, and can build or take a solver for the base
  commit. `pytest validation` runs the offline tests of the replication records.
  With `PHOTONHUB_REQUIRE_SOURCE_MATCH=1`, a test run whose solver is not a
  build of the checked-out revision stops at the start instead of skipping
  every solver test, a `<sha>-dirty` build no longer counts, and a recorded
  install is used only when it matches the source. The energy-closure and
  diffraction-closure gates now sit at 6 to 10 times their measured error, and
  the mirror-symmetry gate at 60 and 400 times.
- Replication records whose drivers use default port windows
  (`wang2022_swg_wdm_demux`, `xu2008_microring_1p5um`, parts of
  `huang2014_interlayer_transition` and `bahadori2019_optimal_bend`) read
  differently when re-run; their committed results are unchanged until then.

## 0.1.3 (2026-09-08)

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
  floating at that height, a lens's focal spot above the device.
- `photonhub.viz.export_scene` unfolds a symmetry plane: a half-domain
  run with a mirror on an in-plane minimum face is exported as the whole
  device: interior and structure outlines mirrored about the face, the
  field's |E| mirrored as-is and the dominant component's phasor with its
  parity (PEC: normal component even, tangential odd; PMC the reverse).
- `photonhub.viz.plot_comparison(..., stated=(value, label))` draws a paper's
  *stated* number as a dashed line, the form an example uses when the
  article's license does not allow its extracted curve to be re-plotted on
  the public site.
- **Fixed** `plot_field` (and `RunResult.plot_field`) framing: structure
  outlines that extend past the recorded slice (an arm running through the
  wall, or the unsimulated half beyond a symmetry plane) no longer stretch
  the axes; the frame is the slice's own extent.
- **Fixed** a false-positive "lies inside the boundary layers" warning from
  `run_local` on axes carrying a symmetry plane. The PML on such an axis is
  one-sided: the min face is the mirror, not an absorber, so
  equivalence-current mode-launch dipoles a fraction of a cell from the plane
  are interior and no longer reported.
  `Simulation.point_sources_in_boundary_layers` now tests only the far face
  on a symmetry axis; the far face and the other axes' faces are unchanged.
- `photonhub.viz.plot_comparison(x_nm, values, reference=..., ylabel=..., ylim=...)`: the example notebooks' result figure: an observable against wavelength as a line with a paper's digitized series (from `examples/notebooks/refs/`) as markers; `reference_scale` flips a transmittance in dB into a loss.
- Public text no longer names hardware vendors, competing tools or the
  suppliers behind the cloud service: the package, its README, the landing
  page and the documentation site say "GPU". `scripts/check_repo.py` keeps
  it that way. No behaviour changes.

## 0.1.2 (2026-09-02)

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

## 0.1.1 (2026-09-01)

- Public-surface text scrub: physics is described without reference to other
  vendors' products; provenance strings and the shipped example README updated.
- Test suite removed from the sdist; repository `.gitignore` no longer leaks
  into the sdist.
- Documentation site self-hosts its fonts.

## 0.1.0 (2026-08-31)

- First PyPI release of the client: pydantic v2 simulation model (schema
  1.20.0-alpha.1), local runner (`run_local`, `Batch`, `run_async`), cloud
  client (`photonhub.web` with `run_quoted`), cost estimator, GDS import, FDE
  mode solver, S-matrix, near-to-far-field, EME, resonance extraction, adjoint
  inverse design, materials library, visualization layer.
