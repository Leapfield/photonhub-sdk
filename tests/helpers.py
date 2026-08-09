"""Shared test builders that are safe to import outside pytest collection."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
EXAMPLE_SPEC = REPO_ROOT / "examples" / "dipole_vacuum.json"
FRESNEL_SPEC = REPO_ROOT / "examples" / "fresnel_slab.json"


def make_sim(**overrides):
    import photonhub as ph

    kwargs = dict(
        size_um=(0.2, 0.2, 0.2),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        # This is a 4-cell-per-axis domain; the schema-1.12 default (PML on all
        # faces) cannot fit a 12-layer slab, so pin periodic explicitly. (Old
        # default was periodic too, so make_sim's wire output is unchanged.)
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="periodic"),
        sources=[
            ph.PointDipole(
                center_um=(0.1, 0.1, 0.1),
                polarization="Ez",
                source_time=ph.GaussianPulse(
                    freq0_hz=1.934e14, fwidth_hz=4.0e13
                ),
            )
        ],
        monitors=[
            ph.FieldTimeMonitor(
                name="probe", center_um=(0.15, 0.1, 0.1), fields=["Ez"]
            ),
            ph.FieldSnapshotMonitor(name="final", fields=["Ez", "Hx"]),
        ],
    )
    kwargs.update(overrides)
    return ph.Simulation(**kwargs)


def make_pw_sim(**overrides):
    """Build a tiny plane-wave simulation for schema and capability tests."""
    import photonhub as ph

    kwargs = dict(
        size_um=(0.2, 0.2, 0.4),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
        sources=[
            ph.PlaneWave(
                axis="z",
                direction="+",
                position_um=0.1,
                polarization="Ex",
                source_time=ph.GaussianPulse(
                    freq0_hz=1.934e14, fwidth_hz=3.0e13
                ),
            )
        ],
        monitors=[],
    )
    kwargs.update(overrides)
    return ph.Simulation(**kwargs)
