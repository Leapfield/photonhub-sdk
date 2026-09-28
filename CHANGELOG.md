# Changelog: `photonhub` Python SDK

All notable changes to the published SDK. Dates are release dates on PyPI.
The desktop application and the solver engine share this version number.

## Unreleased

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
- `photonhub.viz.export_scene` unfolds a §20 symmetry plane: a half-domain
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
