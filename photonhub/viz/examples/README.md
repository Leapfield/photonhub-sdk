# Workbench starter resources

`mode_converter.sim.json` is a TE0-to-TE1 mode converter: the gdsfactory
generic-PDK `mode_converter_gap0p15_length20` cell from
`JPPhotonics/fdtd-pipeline@622e0a9`
([arXiv:2506.16665](https://arxiv.org/abs/2506.16665)); the GDS SHA-256 is
`d5b29cbc93757def824c730d988fbb33e8fe7a8be9f9095f45abd822b5c95784`.

It is an interactive starter at 10 cells per wavelength, not a converged
result. It keeps the exact GDS, materials, grid, band, true-H mode profile and
monitor planes, and launches through one auxiliary `ModeSource`, so the scene
stays small enough to edit in the Workbench (the equivalence-current launch
would expand to nearly 30,000 point sources). The four physical DFT planes are
modal ports. The `o3` plane requests both TE0 and TE1, so the Workbench reports
the conversion without duplicating raw field data or relying on a monitor's
name to imply a mode.
