# Workbench starter resources

`mode_converter.sim.json` is generated from the repository's canonical GDS test
case at `benchmarks/gds/test_cases/mode_converter` and its `matched_res10`
profile. The source layout is the gdsfactory generic-PDK
`mode_converter_gap0p15_length20` cell from
`JPPhotonics/fdtd-pipeline@622e0a9`; the committed GDS SHA-256 is
`d5b29cbc93757def824c730d988fbb33e8fe7a8be9f9095f45abd822b5c95784`.

The converged matched-res25 execution input is not suitable for structured GUI
editing: its equivalence-current launch expands to nearly 30,000 point sources.
The starter therefore sets `framework._engine.USE_EQ_SOURCE = False` before
building the matched-res10 scene. This retains the exact GDS, materials, grid,
band, true-H mode profile, and monitor planes in one auxiliary `ModeSource`.
It is an interactive smoke case, not the converged benchmark headline. The
four physical DFT planes are persisted as modal ports. The `o3` plane requests
both TE0 and TE1, so Workbench can report conversion without duplicating raw
field data or relying on a monitor's name to imply a mode.

Regenerate from a clean Python process at the repository root:

```python
from pathlib import Path

from benchmarks.gds.framework import _engine
from benchmarks.gds.framework.profile import load_profile
from benchmarks.gds.framework.scene import build_ph_scene
from benchmarks.gds.framework.spec import load_spec
from photonhub import (
    FieldDftMonitor, ModePort, ModeSolveProvenance, ModeSource, PortMode,
    SCHEMA_VERSION,
)
from photonhub.viz.service import mode_source_input_sha256

_engine.USE_EQ_SOURCE = False
spec = load_spec(Path("benchmarks/gds/test_cases/mode_converter"))
scene = build_ph_scene(spec, load_profile("matched_res10"))
sim = scene.sim._validated_copy({"schema_version": SCHEMA_VERSION})
recipe_data = {
    "solver": "yee", "polarization": "TE", "mode_index": 0,
    "wavelength_um": 1.55,
    "center_um": (8.893436697564628, 1.5354366975646268),
    "size_um": (1.70, 1.42),
    "dl_um": 0.04461972479705222,
    "supersample": 8, "num_modes": 6, "num_freqs": 1,
}
recipe = ModeSolveProvenance(
    **recipe_data,
    input_sha256=mode_source_input_sha256(sim, 0, recipe_data),
)
source = ModeSource.model_validate({
    **sim.sources[0].model_dump(mode="python"), "mode_solve": recipe,
})
port_defs = {
    "in": ("o1", "-", (8.893436697564628, 1.5354366975646268),
           (1.70, 1.42), (PortMode(polarization="TE", mode_index=0),), 0),
    "o2": ("o2", "-", (2.135436697564628, 1.5354366975646268),
           (2.40, 1.42), (PortMode(polarization="TE", mode_index=0),), None),
    "o3_TE0": ("o3", "+", (2.135436697564628, 1.5354366975646268),
               (2.40, 1.42), (
                   PortMode(polarization="TE", mode_index=0),
                   PortMode(polarization="TE", mode_index=1),
               ), None),
    "o4": ("o4", "+", (8.893436697564628, 1.5354366975646268),
           (1.70, 1.42), (PortMode(polarization="TE", mode_index=0),), None),
}
monitors = []
for monitor in sim.monitors:
    if monitor.name == "o3_TE1":
        continue
    if monitor.name not in port_defs:
        monitors.append(monitor)
        continue
    name, outward, center, size, modes, source_index = port_defs[monitor.name]
    monitors.append(FieldDftMonitor.model_validate({
        **monitor.model_dump(mode="python"),
        "name": name,
        "mode_port": ModePort(
            out_direction=outward, center_um=center, size_um=size,
            dl_um=recipe.dl_um, supersample=8, num_modes=6, modes=modes,
            source_index=source_index, thickness_axis="z",
        ),
    }))
sim = sim._validated_copy({"sources": (source,), "monitors": tuple(monitors)})
print(sim.to_wire_json())
```
