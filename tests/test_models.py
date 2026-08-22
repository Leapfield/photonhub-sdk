"""Model validation: happy and sad paths for the wire-format models."""

import pytest
from pydantic import ValidationError

import photonhub as ph
from photonhub.components.grid import snap_mixed_plane

from .helpers import make_pw_sim, make_sim


F0 = 1.934e14


def _modal_source():
    recipe = ph.ModeSolveProvenance(
        polarization="TE", mode_index=0, wavelength_um=1.55,
        center_um=(0.5, 0.5), size_um=(0.6, 0.6), dl_um=0.1,
        supersample=8, num_modes=2, num_freqs=1,
        input_sha256="a" * 64,
    )
    return ph.ModeSource(
        axis="x", direction="+", position_um=0.25,
        polarization="Ey", n_eff=2.4, nu=1, nv=1, profile=(1.0,),
        source_time=ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13),
        mode_solve=recipe,
    )


def _modal_monitor(name="input", *, source_index=0):
    return ph.FieldDftMonitor(
        name=name, center_um=(0.75, 0.5, 0.5), size_um=(0.0, 0.8, 0.8),
        fields=("Ey", "Ez", "Hy", "Hz"), freqs_hz=(F0,),
        mode_port=ph.ModePort(
            out_direction="-", center_um=(0.5, 0.5), size_um=(0.6, 0.6),
            dl_um=0.1, num_modes=2,
            modes=(ph.PortMode(polarization="TE", mode_index=0),),
            source_index=source_index, thickness_axis="z",
        ),
    )


def _modal_sim(*, monitors=None, sources=None):
    return ph.Simulation(
        size_um=(1.0, 1.0, 1.0), grid=ph.UniformGridSpec(dl_um=0.1),
        run=ph.RunSpec(n_steps=5),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="periodic"),
        sources=tuple(sources) if sources is not None else (_modal_source(),),
        monitors=tuple(monitors) if monitors is not None else (_modal_monitor(),),
    )


class TestDefaults:
    def test_tiny_sim_builds_with_documented_defaults(self, tiny_sim):
        assert tiny_sim.schema_version == ph.SCHEMA_VERSION
        assert tiny_sim.run.courant == 0.99
        assert tiny_sim.run.shutoff == 1.0e-5   # NUMERICS.md section 7
        assert tiny_sim.background.permittivity == 1.0
        # The default boundary flipped to PML on all faces in schema 1.12
        # (tiny_sim pins periodic explicitly because its 4-cell domain can't
        # host a PML slab — see conftest.make_sim).
        assert ph.Boundaries().x == "pml"
        assert tiny_sim.structures == ()
        assert tiny_sim.pml_num_layers == 12
        # §11 CPML profile knobs default to the prior hardcoded constants.
        assert tiny_sim.pml_m == 3.0
        assert tiny_sim.pml_kappa_max == 3.0
        assert tiny_sim.pml_alpha_max == 0.24
        # §21 adiabatic absorber knobs.
        assert tiny_sim.absorber_num_layers == 40
        assert tiny_sim.absorber_m == 3.0
        src = tiny_sim.sources[0]
        assert src.amplitude == 1.0
        assert src.source_time.offset == 5.0
        assert src.source_time.phase == 0.0
        assert tiny_sim.monitors[0].interval_steps == 1   # field_time default
        assert tiny_sim.monitors[1].interval_steps == 0   # field_snapshot default

    def test_models_are_frozen(self, tiny_sim):
        with pytest.raises(ValidationError):
            tiny_sim.schema_version = "2.0.0"
        with pytest.raises(ValidationError):
            tiny_sim.run.courant = 0.5


class TestRunSpec:
    @pytest.mark.parametrize("courant", [0.0, -0.5, 1.0, 1.5])
    def test_courant_out_of_range_rejected(self, courant):
        with pytest.raises(ValidationError):
            ph.RunSpec(n_steps=10, courant=courant)

    @pytest.mark.parametrize("courant", [1e-6, 0.5, 0.99, 0.9999])
    def test_courant_in_range_accepted(self, courant):
        assert ph.RunSpec(n_steps=10, courant=courant).courant == courant

    def test_both_duration_keys_rejected(self):
        with pytest.raises(ValidationError, match="exactly one"):
            ph.RunSpec(run_time_s=1e-13, n_steps=100)

    def test_neither_duration_key_rejected(self):
        with pytest.raises(ValidationError, match="exactly one"):
            ph.RunSpec()

    def test_nonpositive_durations_rejected(self):
        with pytest.raises(ValidationError):
            ph.RunSpec(run_time_s=0.0)
        with pytest.raises(ValidationError):
            ph.RunSpec(n_steps=0)

    def test_shutoff_default_is_tidy3d_parity(self):
        assert ph.RunSpec(n_steps=10).shutoff == 1.0e-5

    @pytest.mark.parametrize("shutoff", [-1e-6, -0.1, 1.0, 1.5])
    def test_shutoff_out_of_range_rejected(self, shutoff):
        # NUMERICS.md section 7: the energy-decay fraction is in [0, 1).
        with pytest.raises(ValidationError):
            ph.RunSpec(n_steps=10, shutoff=shutoff)

    @pytest.mark.parametrize("shutoff", [0.0, 1e-6, 1e-5, 0.5, 0.999])
    def test_shutoff_in_range_accepted(self, shutoff):
        assert ph.RunSpec(n_steps=10, shutoff=shutoff).shutoff == shutoff


class TestStrictness:
    def test_unknown_key_rejected_at_top_level(self):
        with pytest.raises(ValidationError):
            make_sim(not_a_real_key=1)

    def test_unknown_key_rejected_in_nested_model(self):
        with pytest.raises(ValidationError):
            ph.GaussianPulse(freq0_hz=1e14, fwidth_hz=1e13, bogus=2)

    def test_unknown_discriminator_value_rejected(self):
        with pytest.raises(ValidationError):
            make_sim(grid={"type": "nonuniform", "dl_um": 0.05})


class TestSources:
    @pytest.mark.parametrize("pol", ["Hx", "Hy", "Hz"])
    def test_magnetic_polarization_accepted(self, pol):
        # Magnetic (Hx/Hy/Hz) point dipoles are supported on the CPU reference
        # solver (NUMERICS.md section 5); the polarization passes to the wire.
        d = ph.PointDipole(
            center_um=(0, 0, 0),
            polarization=pol,
            source_time=ph.GaussianPulse(freq0_hz=1e14, fwidth_hz=1e13),
        )
        assert d.polarization == pol
        wire = d.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert wire["polarization"] == pol

    def test_nonsense_polarization_rejected(self):
        with pytest.raises(ValidationError):
            ph.PointDipole(
                center_um=(0, 0, 0),
                polarization="Qx",
                source_time=ph.GaussianPulse(freq0_hz=1e14, fwidth_hz=1e13),
            )

    def test_empty_sources_rejected(self):
        with pytest.raises(ValidationError):
            make_sim(sources=[])

    @pytest.mark.parametrize("bad", [{"freq0_hz": 0.0, "fwidth_hz": 1e13},
                                     {"freq0_hz": 1e14, "fwidth_hz": -1.0},
                                     {"freq0_hz": 1e14, "fwidth_hz": 1e13, "offset": -0.1}])
    def test_gaussian_pulse_bounds(self, bad):
        with pytest.raises(ValidationError):
            ph.GaussianPulse(**bad)

    def test_windowed_carrier_is_valid_for_point_dipole(self):
        pulse = ph.GaussianPulse(
            freq0_hz=1.5e14, fwidth_hz=2e13,
            band_freqs_hz=(1.4e14, 1.6e14), carrier_index=0)
        source = ph.PointDipole(
            center_um=(0, 0, 0), polarization="Ex", source_time=pulse)
        assert source.source_time.carrier_index == 0

    @pytest.mark.parametrize("non_finite", [float("inf"), float("nan")])
    def test_windowed_carrier_rejects_non_finite_band_frequency(
            self, non_finite):
        with pytest.raises(ValidationError):
            ph.GaussianPulse(
                freq0_hz=1.5e14, fwidth_hz=2e13,
                band_freqs_hz=(1.4e14, non_finite), carrier_index=0)

    def test_windowed_carrier_is_rejected_for_plane_wave(self):
        pulse = ph.GaussianPulse(
            freq0_hz=1.5e14, fwidth_hz=2e13,
            band_freqs_hz=(1.4e14, 1.6e14), carrier_index=0)
        with pytest.raises(ValidationError, match="reserved for PointDipole"):
            ph.PlaneWave(
                axis="z", direction="+", position_um=0.1,
                polarization="Ex", source_time=pulse)

    def test_windowed_carrier_is_rejected_for_mode_source(self):
        pulse = ph.GaussianPulse(
            freq0_hz=1.5e14, fwidth_hz=2e13,
            band_freqs_hz=(1.4e14, 1.6e14), carrier_index=0)
        with pytest.raises(ValidationError, match="reserved for PointDipole"):
            ph.ModeSource(
                axis="z", direction="+", position_um=0.1,
                polarization="Ex", n_eff=1.5, nu=1, nv=1,
                profile=(1.0,), source_time=pulse)

    def test_mode_source_solve_provenance_round_trips(self):
        provenance = ph.ModeSolveProvenance(
            polarization="TE", mode_index=0, wavelength_um=1.55,
            center_um=(0.2, 0.3), size_um=(0.4, 0.5), dl_um=0.05,
            supersample=8, num_modes=6, num_freqs=1,
            input_sha256="a" * 64,
        )
        source = ph.ModeSource(
            axis="z", direction="+", position_um=0.1,
            polarization="Ex", n_eff=1.5, nu=1, nv=1,
            profile=(1.0,), source_time=ph.GaussianPulse(
                freq0_hz=1e14, fwidth_hz=1e13),
            mode_solve=provenance,
        )
        wire = source.model_dump(mode="json", exclude_none=True)
        assert wire["mode_solve"]["solver"] == "yee"
        assert wire["mode_solve"]["input_sha256"] == "a" * 64
        assert ph.ModeSource.model_validate(wire) == source

    @pytest.mark.parametrize("update,match", [
        ({"input_sha256": "A" * 64}, "input_sha256"),
        ({"mode_index": 32}, "mode_index"),
        ({"num_modes": 1, "mode_index": 1}, "greater than mode_index"),
        ({"size_um": (0.4, 0.0)}, "size_um"),
    ])
    def test_mode_source_solve_provenance_rejects_invalid_values(
            self, update, match):
        values = {
            "polarization": "TE", "mode_index": 0,
            "wavelength_um": 1.55, "center_um": (0.2, 0.3),
            "size_um": (0.4, 0.5), "dl_um": 0.05,
            "supersample": 8, "num_modes": 6, "num_freqs": 1,
            "input_sha256": "a" * 64,
        }
        with pytest.raises(ValidationError, match=match):
            ph.ModeSolveProvenance(**{**values, **update})


class TestMonitors:
    def test_empty_fields_rejected(self):
        with pytest.raises(ValidationError):
            ph.FieldTimeMonitor(name="p", center_um=(0, 0, 0), fields=[])
        with pytest.raises(ValidationError):
            ph.FieldSnapshotMonitor(name="s", fields=[])

    def test_field_time_interval_must_be_positive(self):
        with pytest.raises(ValidationError):
            ph.FieldTimeMonitor(name="p", center_um=(0, 0, 0), fields=["Ez"],
                                interval_steps=0)

    def test_snapshot_interval_zero_means_final_only(self):
        m = ph.FieldSnapshotMonitor(name="s", fields=["Ez"])
        assert m.interval_steps == 0

    def test_duplicate_monitor_names_rejected(self):
        with pytest.raises(ValidationError, match="unique"):
            make_sim(monitors=[
                ph.FieldSnapshotMonitor(name="dup", fields=["Ez"]),
                ph.FieldTimeMonitor(name="dup", center_um=(0, 0, 0), fields=["Ex"]),
            ])

    def test_ascii_case_only_monitor_names_rejected(self):
        with pytest.raises(ValidationError, match="ignoring ASCII case"):
            make_sim(monitors=[
                ph.FieldSnapshotMonitor(name="Probe", fields=["Ez"]),
                ph.FieldTimeMonitor(
                    name="probe", center_um=(0, 0, 0), fields=["Ex"]
                ),
            ])

    def test_mode_port_defaults_and_wire_roundtrip(self):
        port = ph.ModePort(
            out_direction="+", center_um=(0.2, 0.3), size_um=(0.4, 0.5),
            dl_um=0.05, modes=[ph.PortMode(polarization="TE", mode_index=0)],
        )
        assert port.solver == "yee"
        assert port.supersample == 8
        assert port.num_modes is None
        assert port.source_index is None
        monitor = ph.FieldDftMonitor(
            name="port", center_um=(0.5, 0.5, 0.5),
            size_um=(0.0, 0.8, 0.8), fields=("Ey", "Ez", "Hy", "Hz"),
            freqs_hz=(F0,), mode_port=port,
        )
        wire = monitor.model_dump(mode="json", exclude_none=True)
        assert wire["mode_port"]["solver"] == "yee"
        assert wire["mode_port"]["modes"] == [
            {"polarization": "TE", "mode_index": 0}]
        assert ph.FieldDftMonitor.model_validate(wire) == monitor

    def test_mode_port_auto_trials_cover_highest_single_family_index(self):
        port = ph.ModePort(
            out_direction="+", center_um=(0.2, 0.3), size_um=(0.4, 0.5),
            dl_um=0.05,
            modes=[ph.PortMode(polarization="TE", mode_index=31)],
        )
        assert port.num_modes is None

    @pytest.mark.parametrize("update,match", [
        ({"modes": []}, "modes"),
        ({"modes": [
            {"polarization": "TE", "mode_index": 0},
            {"polarization": "TE", "mode_index": 0},
        ]}, "unique"),
        ({"modes": [{"polarization": "TE", "mode_index": 32}]},
         "mode_index"),
        ({"modes": [{"polarization": "TE", "mode_index": 1}],
          "num_modes": 1}, "at least 2"),
        ({"modes": [
            {"polarization": "TE", "mode_index": 0},
            {"polarization": "TM", "mode_index": 0},
        ], "num_modes": 1}, "at least 2"),
        ({"modes": [
            {"polarization": "TE", "mode_index": 31},
            {"polarization": "TM", "mode_index": 0},
        ]}, "require 33 trial modes.*maximum of 32"),
        ({"size_um": (0.0, 0.5)}, "size_um"),
        ({"dl_um": 0.0}, "dl_um"),
        ({"supersample": 17}, "supersample"),
        ({"source_index": -1}, "source_index"),
    ])
    def test_mode_port_rejects_invalid_contract_values(self, update, match):
        values = {
            "out_direction": "+", "center_um": (0.2, 0.3),
            "size_um": (0.4, 0.5), "dl_um": 0.05,
            "modes": [{"polarization": "TE", "mode_index": 0}],
        }
        with pytest.raises(ValidationError, match=match):
            ph.ModePort(**{**values, **update})


class TestModalPortSimulationRules:
    def test_valid_driven_port_round_trips_in_simulation_wire(self):
        sim = _modal_sim()
        wire = sim.to_wire_json()
        assert '"mode_port"' in wire
        assert ph.Simulation.from_wire_json(wire) == sim

    @pytest.mark.parametrize("update,match", [
        ({"size_um": (0.8, 0.8, 0.8)}, "exactly one zero"),
        ({"fields": ("Ey", "Ez", "Hy")}, "missing.*Hz"),
        ({"interval_space": (1, 2, 1)}, "does not support.*decimated"),
        ({"apodization": ph.Apodization(width_s=1e-13)},
         "does not support time apodization"),
    ])
    def test_port_requires_dense_four_field_plane(self, update, match):
        monitor = _modal_monitor().model_copy(update=update)
        with pytest.raises(ValidationError, match=match):
            _modal_sim(monitors=[monitor])

    def test_port_window_must_fit_recorded_plane_and_domain(self):
        monitor = _modal_monitor()
        port = monitor.mode_port.model_copy(update={"center_um": (0.9, 0.5)})
        outside = monitor.model_copy(update={"mode_port": port})
        with pytest.raises(ValidationError, match="extends outside"):
            _modal_sim(monitors=[outside])

    @pytest.mark.parametrize("boundary", ["pml", "absorber"])
    def test_port_plane_and_solve_window_exclude_absorbing_layers(
            self, boundary):
        source = _modal_source().model_copy(update={"position_um": 0.5})
        monitor = _modal_monitor().model_copy(update={
            "center_um": (1.0, 1.0, 1.0),
            "size_um": (0.0, 2.0, 2.0),
        })
        port = monitor.mode_port.model_copy(update={
            "center_um": (1.0, 1.0),
            "size_um": (0.6, 0.6),
        })
        monitor = monitor.model_copy(update={"mode_port": port})
        kwargs = {
            "size_um": (2.0, 2.0, 2.0),
            "grid": ph.UniformGridSpec(dl_um=0.1),
            "run": ph.RunSpec(n_steps=5),
            "boundaries": ph.Boundaries(
                x=boundary, y=boundary, z=boundary),
            "pml_num_layers": 4,
            "absorber_num_layers": 4,
            "sources": (source,),
        }
        assert ph.Simulation(monitors=(monitor,), **kwargs)

        safe_upper_edge = monitor.model_copy(update={
            "center_um": (1.55, 1.0, 1.0),
        })
        safe = ph.Simulation(monitors=(safe_upper_edge,), **kwargs)
        assert snap_mixed_plane(safe, 0, 1.55)[0] == pytest.approx(1.525)

        snaps_into_band = monitor.model_copy(update={
            # Nominally below the 1.6 um nonabsorbing edge, but the nearest
            # mixed-Yee quarter plane is 1.625 um in the first absorbing cell.
            "center_um": (1.59, 1.0, 1.0),
        })
        with pytest.raises(
                ValidationError,
                match=rf"requested 1.59 um but snaps to 1.625 um inside the "
                      rf"{boundary} band"):
            ph.Simulation(monitors=(snaps_into_band,), **kwargs)

        inside_band = monitor.model_copy(update={
            "center_um": (1.7, 1.0, 1.0),
        })
        with pytest.raises(
                ValidationError, match=rf"plane on 'x'.*{boundary} band"):
            ph.Simulation(monitors=(inside_band,), **kwargs)

        overlapping_port = port.model_copy(update={
            "center_um": (1.5, 1.0),
            "size_um": (0.4, 0.6),
        })
        overlapping = monitor.model_copy(update={
            "mode_port": overlapping_port,
        })
        with pytest.raises(
                ValidationError, match=rf"window on 'y'.*{boundary} band"):
            ph.Simulation(monitors=(overlapping,), **kwargs)

    @pytest.mark.parametrize("boundary", ["pml", "absorber"])
    def test_port_plane_graded_snap_excludes_upper_absorbing_cell(
            self, boundary):
        # Twenty primary cells: ten 0.08 um cells followed by ten 0.12 um
        # cells. Four upper absorbing cells start at q[16] = 1.52 um.
        x_coords = tuple(
            [0.08 * index for index in range(11)]
            + [0.8 + 0.12 * index for index in range(1, 10)]
        )
        grid = ph.GradedGridSpec(
            dl_um=0.1,
            coords=ph.GradedAxisCoords(x=x_coords),
        )
        source = _modal_source().model_copy(update={"position_um": 0.5})
        monitor = _modal_monitor().model_copy(update={
            # Requested 1.51 is inside the nominal nonabsorbing interval, but
            # the nearest local quarter plane is 1.55 in the first upper layer.
            "center_um": (1.51, 1.0, 1.0),
            "size_um": (0.0, 2.0, 2.0),
        })
        port = monitor.mode_port.model_copy(update={
            "center_um": (1.0, 1.0),
            "size_um": (0.6, 0.6),
        })
        monitor = monitor.model_copy(update={"mode_port": port})
        kwargs = {
            "size_um": (2.0, 2.0, 2.0),
            "grid": grid,
            "run": ph.RunSpec(n_steps=5),
            "boundaries": ph.Boundaries(
                x=boundary, y="periodic", z="periodic"),
            "pml_num_layers": 4,
            "absorber_num_layers": 4,
            "sources": (source,),
        }

        with pytest.raises(
                ValidationError,
                match=rf"requested 1.51 um but snaps to 1.55 um inside the "
                      rf"{boundary} band"):
            ph.Simulation(monitors=(monitor,), **kwargs)

    def test_port_thickness_axis_cannot_be_plane_normal(self):
        monitor = _modal_monitor()
        port = monitor.mode_port.model_copy(update={"thickness_axis": "x"})
        with pytest.raises(ValidationError, match="cannot equal the plane normal"):
            _modal_sim(monitors=[monitor.model_copy(update={"mode_port": port})])

    def test_driven_port_source_must_be_reproducible_and_directional(self):
        point = ph.PointDipole(
            center_um=(0.25, 0.5, 0.5), polarization="Ey",
            source_time=ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13),
        )
        with pytest.raises(ValidationError, match="ModeSource with mode_solve"):
            _modal_sim(sources=[point])

        wrong_direction = _modal_source().model_copy(update={"direction": "-"})
        with pytest.raises(ValidationError, match="requires.*direction.*'\\+'"):
            _modal_sim(sources=[wrong_direction])

        wrong_side = _modal_source().model_copy(update={"position_um": 0.9})
        with pytest.raises(ValidationError, match="must lie downstream"):
            _modal_sim(sources=[wrong_side])

    def test_driven_port_must_include_launched_channel(self):
        monitor = _modal_monitor()
        port = monitor.mode_port.model_copy(update={
            "modes": (ph.PortMode(polarization="TM", mode_index=0),),
        })
        with pytest.raises(ValidationError, match="include the launched TE0"):
            _modal_sim(monitors=[monitor.model_copy(update={"mode_port": port})])

    def test_driven_channel_is_relative_to_port_thickness_axis(self):
        monitor = _modal_monitor()
        port = monitor.mode_port.model_copy(update={
            # x-normal natural solve axes are y,z. The saved source's natural
            # TE family is E_y, which is physical TM when y is slab thickness.
            "thickness_axis": "y",
            "modes": (ph.PortMode(polarization="TM", mode_index=0),),
        })
        physical = monitor.model_copy(update={"mode_port": port})
        assert _modal_sim(monitors=[physical]).monitors[0] == physical

        mislabeled = port.model_copy(update={
            "modes": (ph.PortMode(polarization="TE", mode_index=0),),
        })
        with pytest.raises(ValidationError, match="include the launched TM0"):
            _modal_sim(monitors=[
                monitor.model_copy(update={"mode_port": mislabeled}),
            ])

    def test_only_one_port_is_source_linked_in_a_run(self):
        first = _modal_monitor()
        second_port = first.mode_port.model_copy(update={
            "source_index": None, "out_direction": "+",
        })
        second = first.model_copy(update={
            "name": "output", "center_um": (0.85, 0.5, 0.5),
            "mode_port": second_port,
        })
        driven_second = second.model_copy(update={
            "mode_port": first.mode_port,
        })
        with pytest.raises(ValidationError, match="at most one.*driven"):
            _modal_sim(monitors=[first, driven_second])


class TestSimulation:
    @pytest.mark.parametrize("size", [(0.0, 1.0, 1.0), (1.0, -2.0, 1.0)])
    def test_nonpositive_size_rejected(self, size):
        with pytest.raises(ValidationError):
            make_sim(size_um=size)

    def test_invalid_boundary_kind_rejected(self):
        with pytest.raises(ValidationError):
            ph.Boundaries(x="mur")

    @pytest.mark.parametrize("kind", ["periodic", "pec", "pml"])
    def test_all_boundary_kinds_accepted(self, kind):
        assert ph.Boundaries(z=kind).z == kind

    def test_pml_num_layers_bounds(self):
        assert make_sim(pml_num_layers=4).pml_num_layers == 4
        with pytest.raises(ValidationError):
            make_sim(pml_num_layers=3)
        with pytest.raises(ValidationError):
            make_sim(pml_num_layers=2**31)

    def test_cpml_profile_bounds(self):
        # §11 ranges: m >= 1, kappa_max >= 1, alpha_max >= 0 (mirror the
        # engine validate() in resolve.cpp).
        assert make_sim(pml_m=1.0, pml_kappa_max=1.0, pml_alpha_max=0.0)
        with pytest.raises(ValidationError):
            make_sim(pml_m=0.5)
        with pytest.raises(ValidationError):
            make_sim(pml_kappa_max=0.5)
        with pytest.raises(ValidationError):
            make_sim(pml_alpha_max=-0.1)


class TestWithHelpersRevalidate:
    """Regression (K5): the with_* helpers used model_copy(update=), which
    skips the cross-field model validators — an invalid combination (graded
    grid + plane wave, absorber on a plane wave's transverse axes) sailed
    through and failed only at engine submission, defeating capabilities.py's
    fail-at-construction contract. They must now raise the same error direct
    construction gives, while leaving VALID copies byte-identical on the wire
    (the model_fields_set semantics _wire_exclude keys on are preserved)."""

    def test_with_absorber_plane_wave_raises_like_direct_construction(self):
        # §13: a plane wave's transverse axes must stay periodic; with_absorber
        # flips all six faces, so it must fail at the helper call.
        sim = make_pw_sim()
        with pytest.raises(ValidationError, match="must be periodic"):
            sim.with_absorber()
        # Same error the same field values raise at direct construction.
        with pytest.raises(ValidationError, match="must be periodic"):
            make_pw_sim(boundaries=ph.Boundaries(
                x="absorber", y="absorber", z="absorber"))

    def test_with_absorber_valid_scene_byte_identical_to_plain_copy(self):
        sim = make_sim()
        new = sim.with_absorber(num_layers=48)
        old = sim.model_copy(update={
            "boundaries": ph.Boundaries(x="absorber", y="absorber",
                                        z="absorber"),
            "absorber_num_layers": 48,
        })
        # The validated helper returns the plain model_copy result: wire bytes
        # and fields_set (what _wire_exclude keys unset-field omission on) are
        # unchanged for a valid scene.
        assert new.to_wire_json() == old.to_wire_json()
        assert new.model_fields_set == old.model_fields_set
        # And the emitted document round-trips through strict wire ingestion.
        again = ph.Simulation.from_wire_json(new.to_wire_json())
        assert again.to_wire_json() == new.to_wire_json()

    def test_with_stabilized_pml_valid_scene_unchanged(self):
        # A valid stabilized-PML copy is untouched by the re-validation
        # (values, fields_set, wire bytes all match the plain copy path).
        sim = make_pw_sim()
        new = sim.with_stabilized_pml(num_layers=20)
        assert new.pml_num_layers == 20
        assert new.pml_kappa_max == 5.0
        old = sim.model_copy(update={
            "pml_num_layers": new.pml_num_layers,
            "pml_kappa_max": new.pml_kappa_max,
            "pml_alpha_max": new.pml_alpha_max,
        })
        assert new.to_wire_json() == old.to_wire_json()
        assert new.model_fields_set == old.model_fields_set


class TestSchemaVersionGate:
    """Engine parity: resolve.cpp accepts schema major 1 only; the client
    must reject other majors at construction, not at submission."""

    def test_default_version_accepted(self, tiny_sim):
        assert tiny_sim.schema_version == ph.SCHEMA_VERSION

    @pytest.mark.parametrize("version", ["1.0.0", "1.2.3-rc.1", "1"])
    def test_same_major_versions_accepted(self, version):
        assert make_sim(schema_version=version).schema_version == version

    @pytest.mark.parametrize("version", ["2.0.0", "0.9.0", "banana", "", "x.1"])
    def test_other_majors_and_garbage_rejected(self, version):
        with pytest.raises(ValidationError, match="major version 1"):
            make_sim(schema_version=version)


class TestMonitorNameRules:
    """Engine parity: resolve.cpp check_name rejects names unusable as
    filenames (the engine writes <name>.bin)."""

    @pytest.mark.parametrize(
        "name",
        [
            "a/b", "a\\b", "a\x00b", "has space", "bad:name", "µprobe",
            ".", "..", ".hidden", "trailing.", "CON", "con.txt", "Lpt9.log",
        ],
    )
    def test_path_unsafe_names_rejected(self, name):
        with pytest.raises(ValidationError, match="filename"):
            ph.FieldSnapshotMonitor(name=name, fields=["Ez"])
        with pytest.raises(ValidationError, match="filename"):
            ph.FieldTimeMonitor(name=name, center_um=(0, 0, 0), fields=["Ez"])

    def test_name_length_keeps_bin_filename_within_portable_limit(self):
        assert ph.FieldSnapshotMonitor(
            name="x" * 251, fields=["Ez"]
        ).name == "x" * 251
        with pytest.raises(ValidationError, match="251"):
            ph.FieldSnapshotMonitor(name="x" * 252, fields=["Ez"])

    @pytest.mark.parametrize(
        "name", ["probe", "final_fields", "a.b", "_tmp", "dash-name", "COM0"]
    )
    def test_filename_safe_names_accepted(self, name):
        assert ph.FieldSnapshotMonitor(name=name, fields=["Ez"]).name == name


class TestResolvedGridResourceEnvelope:
    """Cheap client mirror of the engine's beta field-layout limits."""

    def test_large_grid_inside_both_limits_is_accepted(self):
        sim = make_sim(
            size_um=(1280.0, 1280.0, 1280.0),
            grid=ph.UniformGridSpec(dl_um=1.0),
            monitors=[],
        )
        assert sim.cost_estimate().cells_per_axis == (1280, 1280, 1280)

    def test_logical_cell_limit_is_enforced_at_construction(self):
        with pytest.raises(ValidationError, match="beta maximum.*logical cells"):
            make_sim(
                size_um=(1291.0, 1291.0, 1291.0),
                grid=ph.UniformGridSpec(dl_um=1.0),
                monitors=[],
            )

    def test_padded_halo_limit_is_enforced_at_construction(self):
        # Logical cells fit by 15; the 32-cell x pitch and two z halos do not.
        with pytest.raises(ValidationError, match="x-padded grid.*z halos"):
            make_sim(
                size_um=(134217727.0, 4.0, 4.0),
                grid=ph.UniformGridSpec(dl_um=1.0),
                monitors=[],
            )

    def test_extreme_finite_ratio_is_an_actionable_validation_error(self):
        with pytest.raises(ValidationError, match="axis cell count.*maximum"):
            make_sim(
                size_um=(1.0, 1.0, 1.0),
                grid=ph.UniformGridSpec(dl_um=1e-320),
                monitors=[],
            )


class TestNonFiniteRejection:
    """Engine parity: NaN/Inf serialize to JSON null (rejected by phsolver)
    or are raw-literal parse errors, so they must fail at construction on
    EVERY float field — including gt/ge-constrained ones that +inf passes."""

    NAN = float("nan")
    INF = float("inf")

    @pytest.mark.parametrize("value", [NAN, INF, -INF])
    def test_unconstrained_floats_reject_nonfinite(self, value):
        with pytest.raises(ValidationError):
            ph.PointDipole(center_um=(0, 0, 0), polarization="Ez",
                           amplitude=value,
                           source_time=ph.GaussianPulse(freq0_hz=1e14,
                                                        fwidth_hz=1e13))
        with pytest.raises(ValidationError):
            ph.GaussianPulse(freq0_hz=1e14, fwidth_hz=1e13, phase=value)

    @pytest.mark.parametrize("value", [NAN, INF])
    def test_constrained_floats_reject_nonfinite(self, value):
        with pytest.raises(ValidationError):
            ph.GaussianPulse(freq0_hz=value, fwidth_hz=1e13)
        with pytest.raises(ValidationError):
            ph.UniformGridSpec(dl_um=value)
        with pytest.raises(ValidationError):
            ph.RunSpec(run_time_s=value)

    @pytest.mark.parametrize("value", [NAN, INF, -INF])
    def test_tuple_elements_reject_nonfinite(self, value):
        with pytest.raises(ValidationError):
            make_sim(size_um=(value, 1.0, 1.0))
        with pytest.raises(ValidationError):
            ph.FieldTimeMonitor(name="p", center_um=(value, 0, 0),
                                fields=["Ez"])

    def test_nan_literal_in_wire_json_rejected(self, tiny_sim):
        text = tiny_sim.to_wire_json().replace('"amplitude": 1.0',
                                               '"amplitude": NaN')
        with pytest.raises(ValidationError):
            ph.Simulation.model_validate_json(text)


class TestInt32Bounds:
    """Engine parity: phsolver's as_int rejects values outside int32."""

    def test_n_steps_beyond_int32_rejected(self):
        with pytest.raises(ValidationError):
            ph.RunSpec(n_steps=2**31)
        assert ph.RunSpec(n_steps=2**31 - 1).n_steps == 2**31 - 1

    def test_interval_steps_beyond_int32_rejected(self):
        with pytest.raises(ValidationError):
            ph.FieldTimeMonitor(name="p", center_um=(0, 0, 0), fields=["Ez"],
                                interval_steps=2**31)
        with pytest.raises(ValidationError):
            ph.FieldSnapshotMonitor(name="s", fields=["Ez"],
                                    interval_steps=2**31)


class TestRealizedDomainCheck:
    """Engine parity (best-effort early feedback; phsolver validate remains
    authoritative): centers must lie inside the REALIZED domain n*dl, which
    the n >= 4 floor can make LARGER than size_um."""

    def test_source_outside_domain_rejected(self):
        with pytest.raises(ValidationError, match="outside the domain"):
            make_sim(sources=[ph.PointDipole(
                center_um=(5.0, 0.1, 0.1), polarization="Ez",
                source_time=ph.GaussianPulse(freq0_hz=1e14, fwidth_hz=1e13))])

    def test_monitor_outside_domain_rejected(self):
        with pytest.raises(ValidationError, match="outside the domain"):
            make_sim(monitors=[ph.FieldTimeMonitor(
                name="far", center_um=(0.1, 0.1, 9.0), fields=["Ez"])])

    def test_center_inside_realized_but_outside_requested_accepted(self):
        # size 0.1 um at dl 0.05 -> n = max(4, 2) = 4 cells -> realized
        # 0.2 um: a center at 0.15 um is outside the REQUESTED size but
        # inside the realized domain, exactly like the engine accepts it.
        sim = make_sim(size_um=(0.1, 0.2, 0.2),
                       sources=[ph.PointDipole(
                           center_um=(0.15, 0.1, 0.1), polarization="Ez",
                           source_time=ph.GaussianPulse(freq0_hz=1e14,
                                                        fwidth_hz=1e13))],
                       monitors=[])
        assert sim.size_um[0] == 0.1

    def test_snapshot_monitors_have_no_center_to_check(self):
        sim = make_sim(monitors=[ph.FieldSnapshotMonitor(name="s",
                                                         fields=["Ez"])])
        assert sim.monitors[0].name == "s"


# --- Phase 1a-1 models (NUMERICS.md sections 9-13) -------------------------


def _pulse():
    return ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=3.0e13)


class TestMedium:
    def test_conductivity_defaults_to_lossless(self):
        m = ph.Medium(permittivity=12.0)
        assert m.conductivity_s_per_m == 0.0

    def test_permittivity_below_one_rejected(self):
        with pytest.raises(ValidationError):
            ph.Medium(permittivity=0.99)

    def test_negative_conductivity_rejected(self):
        with pytest.raises(ValidationError):
            ph.Medium(permittivity=1.0, conductivity_s_per_m=-1e-6)

    def test_lossy_medium_accepted(self):
        m = ph.Medium(permittivity=2.25, conductivity_s_per_m=10.0)
        assert m.conductivity_s_per_m == 10.0

    # NUMERICS.md §19 single-pole Lorentz dispersion.
    def test_medium_defaults_to_non_dispersive(self):
        m = ph.Medium(permittivity=4.0)
        assert m.lorentz is None

    def test_lorentz_pole_accepted(self):
        m = ph.Medium(
            permittivity=2.25,
            lorentz=ph.LorentzPole(resonance_frequency_hz=2.2e14,
                                   delta_eps=1.5, linewidth_hz=1.5e13))
        assert m.lorentz is not None
        assert m.lorentz.resonance_frequency_hz == 2.2e14
        assert m.lorentz.delta_eps == 1.5

    def test_lorentz_linewidth_defaults_to_zero(self):
        p = ph.LorentzPole(resonance_frequency_hz=2.2e14, delta_eps=1.5)
        assert p.linewidth_hz == 0.0

    def test_lorentz_nonpositive_resonance_rejected(self):
        with pytest.raises(ValidationError):
            ph.LorentzPole(resonance_frequency_hz=0.0, delta_eps=1.0)

    def test_lorentz_negative_delta_eps_rejected(self):
        with pytest.raises(ValidationError):
            ph.LorentzPole(resonance_frequency_hz=2.2e14, delta_eps=-1.0)

    def test_lorentz_negative_linewidth_rejected(self):
        with pytest.raises(ValidationError):
            ph.LorentzPole(resonance_frequency_hz=2.2e14, delta_eps=1.0,
                           linewidth_hz=-1.0)

    def test_non_dispersive_medium_omits_lorentz_from_wire(self):
        # Back-compat: a non-dispersive medium serializes byte-identically to
        # schema < 1.9 (no "lorentz" key).
        m = ph.Medium(permittivity=4.0)
        assert "lorentz" not in m.model_dump(exclude_none=True)

    def test_dispersive_medium_round_trips_through_wire(self):
        med = ph.Medium(
            permittivity=2.25,
            lorentz=ph.LorentzPole(resonance_frequency_hz=2.2e14,
                                   delta_eps=1.5, linewidth_hz=1.5e13))
        sim = ph.Simulation(
            size_um=(0.4, 0.4, 0.8),
            grid=ph.UniformGridSpec(dl_um=0.05),
            run=ph.RunSpec(n_steps=10),
            boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
            structures=(ph.Structure(
                geometry=ph.Box(center_um=(0.2, 0.2, 0.4),
                                size_um=(0.4, 0.4, 0.2)),
                medium=med),),
            sources=(ph.PlaneWave(
                axis="z", direction="+", position_um=0.1, polarization="Ex",
                source_time=ph.GaussianPulse(freq0_hz=2e14, fwidth_hz=3e13)),),
        )
        back = ph.Simulation.from_wire_json(sim.to_wire_json())
        lz = back.structures[0].medium.lorentz
        assert lz is not None
        assert lz.resonance_frequency_hz == 2.2e14
        assert lz.delta_eps == 1.5
        assert lz.linewidth_hz == 1.5e13


class TestGeometry:
    @pytest.mark.parametrize("size", [(0.0, 1.0, 1.0), (1.0, -1.0, 1.0)])
    def test_box_nonpositive_size_rejected(self, size):
        with pytest.raises(ValidationError):
            ph.Box(center_um=(0, 0, 0), size_um=size)

    @pytest.mark.parametrize("radius", [0.0, -0.5])
    def test_sphere_nonpositive_radius_rejected(self, radius):
        with pytest.raises(ValidationError):
            ph.Sphere(center_um=(0, 0, 0), radius_um=radius)

    def test_unknown_geometry_type_rejected(self):
        with pytest.raises(ValidationError):
            ph.Structure(
                geometry={"type": "cylinder", "center_um": [0, 0, 0],
                          "radius_um": 1.0},
                medium=ph.Medium(permittivity=2.0),
            )

    def test_discriminated_union_dispatches_on_type(self):
        s = ph.Structure(
            geometry={"type": "sphere", "center_um": [1.0, 1.0, 1.0],
                      "radius_um": 0.5},
            medium={"permittivity": 12.0},
        )
        assert isinstance(s.geometry, ph.Sphere)


class TestStructures:
    @staticmethod
    def _box(**overrides):
        kwargs = {
            "geometry": ph.Box(center_um=(0.1, 0.1, 0.1),
                               size_um=(0.1, 0.1, 0.1)),
            "medium": ph.Medium(permittivity=12.0),
        }
        kwargs.update(overrides)
        return ph.Structure(**kwargs)

    def test_optional_name_round_trips_and_unnamed_is_omitted(self):
        unnamed = make_sim(structures=[self._box()])
        assert unnamed.structures[0].name is None
        assert "name" not in unnamed.to_wire_dict()["structures"][0]

        named = make_sim(structures=[self._box(name="Core waveguide")])
        wire = named.to_wire_dict()
        assert wire["structures"][0]["name"] == "Core waveguide"
        back = ph.Simulation.from_wire_json(named.to_wire_json())
        assert back.structures[0].name == "Core waveguide"
        assert back.to_wire_dict() == wire

    def test_name_must_be_nonempty_when_present(self):
        with pytest.raises(ValidationError, match="at least 1 character"):
            self._box(name="")

    def test_simulation_accepts_ordered_structures(self):
        slab = ph.Structure(
            geometry=ph.Box(center_um=(0.1, 0.1, 0.1),
                            size_um=(10.0, 10.0, 0.05)),
            medium=ph.Medium(permittivity=12.0),
        )
        ball = ph.Structure(
            geometry=ph.Sphere(center_um=(0.1, 0.1, 0.1), radius_um=0.05),
            medium=ph.Medium(permittivity=2.0, conductivity_s_per_m=5.0),
        )
        sim = make_sim(structures=[slab, ball])
        # Order is paint order (last wins, NUMERICS.md section 9).
        assert sim.structures == (slab, ball)

    def test_geometry_may_extend_beyond_domain(self):
        # NUMERICS.md section 9: only the part inside the grid matters, so
        # there is deliberately NO domain containment check for structures.
        sim = make_sim(structures=[ph.Structure(
            geometry=ph.Box(center_um=(50.0, 0.1, 0.1),
                            size_um=(200.0, 200.0, 200.0)),
            medium=ph.Medium(permittivity=4.0))])
        assert len(sim.structures) == 1


class TestPlaneWave:
    @pytest.mark.parametrize("axis,pol", [("x", "Ex"), ("y", "Ey"), ("z", "Ez")])
    def test_longitudinal_polarization_rejected(self, axis, pol):
        with pytest.raises(ValidationError, match="tangential"):
            ph.PlaneWave(axis=axis, direction="+", position_um=0.1,
                         polarization=pol, source_time=_pulse())

    @pytest.mark.parametrize("axis,pol", [("x", "Ey"), ("x", "Ez"),
                                          ("y", "Ex"), ("y", "Ez"),
                                          ("z", "Ex"), ("z", "Ey")])
    def test_tangential_polarization_accepted(self, axis, pol):
        pw = ph.PlaneWave(axis=axis, direction="-", position_um=0.1,
                          polarization=pol, source_time=_pulse())
        assert pw.amplitude == 1.0  # documented default

    @pytest.mark.parametrize("pol", ["Hx", "Hy", "Hz", "Qx"])
    def test_non_electric_polarization_rejected(self, pol):
        with pytest.raises(ValidationError):
            ph.PlaneWave(axis="z", direction="+", position_um=0.1,
                         polarization=pol, source_time=_pulse())

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValidationError):
            ph.PlaneWave(axis="z", direction="up", position_um=0.1,
                         polarization="Ex", source_time=_pulse())

    def test_happy_path_simulation(self):
        sim = make_pw_sim()
        assert sim.sources[0].type == "plane_wave"
        assert sim.boundaries.z == "pml"

    @pytest.mark.parametrize("kind", ["pml", "pec"])
    def test_nonperiodic_transverse_axis_rejected(self, kind):
        # NUMERICS.md section 13: BOTH transverse axes must be periodic —
        # notably PML on a transverse axis is rejected, not just PEC.
        with pytest.raises(ValidationError, match="periodic"):
            make_pw_sim(boundaries=ph.Boundaries(x=kind, y="periodic", z="pml"))
        with pytest.raises(ValidationError, match="periodic"):
            make_pw_sim(boundaries=ph.Boundaries(x="periodic", y=kind, z="pml"))

    def test_periodic_propagation_axis_accepted(self):
        # Only the TRANSVERSE axes are constrained by the model layer; the
        # plane/PML intersection rule is phsolver's (it needs Yee snapping).
        # (Transverse axes must be periodic for a plane wave; the propagation
        # axis z being periodic is the case under test.)
        sim = make_pw_sim(
            boundaries=ph.Boundaries(x="periodic", y="periodic", z="periodic"))
        assert sim.boundaries.z == "periodic"

    def test_position_outside_domain_rejected(self):
        with pytest.raises(ValidationError, match="outside the domain"):
            make_pw_sim(sources=[ph.PlaneWave(
                axis="z", direction="+", position_um=5.0,
                polarization="Ex", source_time=_pulse())])


class TestFieldDftMonitor:
    def kwargs(self, **over):
        kw = dict(name="dft", center_um=(0.1, 0.1, 0.1),
                  size_um=(0.1, 0.1, 0.0), fields=["Ex", "Hy"],
                  freqs_hz=[1.9e14, 2.0e14])
        kw.update(over)
        return kw

    def test_happy_path_zero_size_plane_region(self):
        m = ph.FieldDftMonitor(**self.kwargs())
        assert m.size_um[2] == 0.0  # plane regions are legal

    def test_empty_fields_rejected(self):
        with pytest.raises(ValidationError):
            ph.FieldDftMonitor(**self.kwargs(fields=[]))

    def test_empty_freqs_rejected(self):
        with pytest.raises(ValidationError):
            ph.FieldDftMonitor(**self.kwargs(freqs_hz=[]))

    @pytest.mark.parametrize("freqs", [[0.0], [1.9e14, -1.0]])
    def test_nonpositive_freqs_rejected(self, freqs):
        with pytest.raises(ValidationError):
            ph.FieldDftMonitor(**self.kwargs(freqs_hz=freqs))

    def test_negative_size_rejected(self):
        with pytest.raises(ValidationError):
            ph.FieldDftMonitor(**self.kwargs(size_um=(0.1, -0.1, 0.1)))

    @pytest.mark.parametrize("name", ["a/b", "a\x00b", "aux.results"])
    def test_path_unsafe_names_rejected(self, name):
        with pytest.raises(ValidationError, match="filename"):
            ph.FieldDftMonitor(**self.kwargs(name=name))

    def test_center_outside_domain_rejected_in_simulation(self):
        with pytest.raises(ValidationError, match="outside the domain"):
            make_sim(monitors=[ph.FieldDftMonitor(
                **self.kwargs(center_um=(0.1, 0.1, 9.0)))])

    def test_apodization_absent_by_default_and_omitted_from_wire(self):
        m = ph.FieldDftMonitor(**self.kwargs())
        assert m.apodization is None
        wire = m.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert "apodization" not in wire  # back-compat: older engines unaffected

    def test_apodization_round_trips_through_wire(self):
        ap = ph.Apodization(start_s=3.0e-12, end_s=4.0e-12, width_s=1.0e-12)
        m = ph.FieldDftMonitor(**self.kwargs(apodization=ap))
        wire = m.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert wire["apodization"] == {
            "start_s": 3.0e-12, "end_s": 4.0e-12, "width_s": 1.0e-12}

    def test_apodization_one_sided_omits_unset_bound(self):
        # Only a roll-on: end_s stays unset and drops out of the wire.
        m = ph.FieldDftMonitor(
            **self.kwargs(apodization=ph.Apodization(start_s=3e-12, width_s=1e-12)))
        wire = m.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert wire["apodization"] == {"start_s": 3e-12, "width_s": 1e-12}

    def test_apodization_requires_positive_width(self):
        with pytest.raises(ValidationError):
            ph.Apodization(start_s=3e-12, width_s=0.0)

    def test_apodization_rejects_end_before_start(self):
        with pytest.raises(ValidationError, match="start_s"):
            ph.Apodization(start_s=4e-12, end_s=3e-12, width_s=1e-12)


class TestFluxMonitor:
    def kwargs(self, **over):
        kw = dict(name="flux", axis="z", position_um=0.1,
                  freqs_hz=[1.9e14, 2.0e14])
        kw.update(over)
        return kw

    def test_happy_path(self):
        m = ph.FluxMonitor(**self.kwargs())
        assert m.axis == "z"

    def test_empty_freqs_rejected(self):
        with pytest.raises(ValidationError):
            ph.FluxMonitor(**self.kwargs(freqs_hz=[]))

    @pytest.mark.parametrize("freqs", [[0.0], [-2.0e14]])
    def test_nonpositive_freqs_rejected(self, freqs):
        with pytest.raises(ValidationError):
            ph.FluxMonitor(**self.kwargs(freqs_hz=freqs))

    def test_invalid_axis_rejected(self):
        with pytest.raises(ValidationError):
            ph.FluxMonitor(**self.kwargs(axis="w"))

    @pytest.mark.parametrize("name", ["a\\b", "a\x00b", "PRN"])
    def test_path_unsafe_names_rejected(self, name):
        with pytest.raises(ValidationError, match="filename"):
            ph.FluxMonitor(**self.kwargs(name=name))

    def test_interior_plane_accepted(self):
        # make_sim domain: 0.2 um at dl 0.05 -> n = 4; interior planes
        # kp in [1, 3] (NUMERICS.md section 12).
        sim = make_sim(monitors=[ph.FluxMonitor(**self.kwargs())])
        assert sim.monitors[0].name == "flux"

    @pytest.mark.parametrize("pos", [0.0, 0.01, 0.2, 0.19])
    def test_boundary_plane_rejected(self, pos):
        # 0.0/0.01 snap to kp = 0; 0.2/0.19 snap to kp = 4 = n: both walls.
        with pytest.raises(ValidationError, match="flux plane"):
            make_sim(monitors=[ph.FluxMonitor(**self.kwargs(position_um=pos))])
