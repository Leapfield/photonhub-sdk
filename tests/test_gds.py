"""GDS import (:mod:`photonhub.gds`): polygons -> extruded PolySlab structures.

Each test authors a tiny GDSII file in a tmp dir with ``gdstk`` (the same reader
:func:`import_gds` uses), so no external fixtures are needed. Skipped wholesale
when ``gdstk`` is not installed."""

import math

import pytest

import photonhub as ph
from photonhub.gds import GdsLayer, import_gds, read_gds_cell_names

gdstk = pytest.importorskip("gdstk")

SI = ph.Medium(permittivity=3.478 ** 2)
OX = ph.Medium(permittivity=1.444 ** 2)


def _strip(**kw):
    d = dict(layer=(1, 0), medium=SI, zmin_um=0.0, thickness_um=0.22)
    d.update(kw)
    return GdsLayer(**d)


def _signed_area2(verts):
    a = 0.0
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a


def _write(path, cells):
    """Write a list of gdstk.Cell to ``path`` as a GDS library."""
    lib = gdstk.Library()
    for cell in cells:
        lib.add(cell)
    lib.write_gds(str(path))
    return str(path)


def _rect_cell(name, x0, y0, x1, y1, layer=1, datatype=0):
    cell = gdstk.Cell(name)
    cell.add(gdstk.rectangle((x0, y0), (x1, y1), layer=layer, datatype=datatype))
    return cell


# --------------------------------------------------------------------------- #
# Basic extrusion
# --------------------------------------------------------------------------- #


class TestBasicImport:
    def test_single_polygon_to_polyslab(self, tmp_path):
        gds = _write(tmp_path / "a.gds", [_rect_cell("top", 0.0, 0.0, 2.0, 0.5)])
        structures = import_gds(gds, [_strip()])
        assert len(structures) == 1
        geom = structures[0].geometry
        assert isinstance(geom, ph.PolySlab)
        assert geom.axis == "z"
        assert geom.slab_bounds_um == (0.0, 0.22)
        # the four rectangle corners, CCW, in (x, y)
        assert len(geom.vertices_um) == 4
        xs = sorted({v[0] for v in geom.vertices_um})
        ys = sorted({v[1] for v in geom.vertices_um})
        assert xs == [0.0, 2.0] and ys == [0.0, 0.5]
        # medium carried through
        assert structures[0].medium.permittivity == pytest.approx(3.478 ** 2)

    def test_vertices_are_ccw(self, tmp_path):
        gds = _write(tmp_path / "a.gds", [_rect_cell("top", -1.0, -0.3, 4.0, 0.3)])
        (structure,) = import_gds(gds, [_strip()])
        assert _signed_area2(structure.geometry.vertices_um) > 0.0

    def test_clockwise_polygon_normalized_to_ccw(self, tmp_path):
        # author an explicitly CLOCKWISE quad
        cw = gdstk.Polygon(
            [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], layer=1, datatype=0
        )
        cell = gdstk.Cell("top")
        cell.add(cw)
        assert _signed_area2([tuple(p) for p in cw.points]) < 0.0  # CW input
        gds = _write(tmp_path / "cw.gds", [cell])
        (structure,) = import_gds(gds, [_strip()])
        assert _signed_area2(structure.geometry.vertices_um) > 0.0  # CCW output

    def test_layer_datatype_filtering(self, tmp_path):
        cell = gdstk.Cell("top")
        cell.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))   # wanted
        cell.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=7))   # other dtype
        cell.add(gdstk.rectangle((0, 0), (1, 1), layer=99, datatype=0))  # other layer
        gds = _write(tmp_path / "m.gds", [cell])
        structures = import_gds(gds, [_strip()])
        assert len(structures) == 1  # only (1, 0)

    def test_absent_layer_yields_nothing(self, tmp_path):
        gds = _write(tmp_path / "a.gds", [_rect_cell("top", 0, 0, 1, 1, layer=1)])
        assert import_gds(gds, [_strip(layer=(5, 0))]) == ()


# --------------------------------------------------------------------------- #
# Multi-layer stack (rib waveguide: strip + thinner slab)
# --------------------------------------------------------------------------- #


class TestLayerStack:
    def test_two_layers_distinct_z(self, tmp_path):
        cell = gdstk.Cell("rib")
        cell.add(gdstk.rectangle((0, -0.25), (3, 0.25), layer=1, datatype=0))  # strip
        cell.add(gdstk.rectangle((0, -1.0), (3, 1.0), layer=2, datatype=0))    # slab
        gds = _write(tmp_path / "rib.gds", [cell])
        structures = import_gds(
            gds,
            [
                _strip(),  # (1,0) z 0..0.22
                GdsLayer(layer=(2, 0), medium=SI, zmin_um=0.0, thickness_um=0.15),
            ],
        )
        bounds = sorted(s.geometry.slab_bounds_um for s in structures)
        assert bounds == [(0.0, 0.15), (0.0, 0.22)]

    def test_layer_order_preserved(self, tmp_path):
        cell = gdstk.Cell("rib")
        cell.add(gdstk.rectangle((0, 0), (1, 1), layer=2, datatype=0))
        cell.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
        gds = _write(tmp_path / "rib.gds", [cell])
        slab = GdsLayer(layer=(2, 0), medium=OX, zmin_um=0.0, thickness_um=0.15)
        structures = import_gds(gds, [_strip(), slab])  # (1,0) first by spec order
        # first structure is the strip layer (spec order), even though the slab
        # polygon comes first in the file
        assert structures[0].geometry.slab_bounds_um == (0.0, 0.22)
        assert structures[1].geometry.slab_bounds_um == (0.0, 0.15)


# --------------------------------------------------------------------------- #
# Hierarchy / cell selection
# --------------------------------------------------------------------------- #


class TestHierarchy:
    def _hier_gds(self, tmp_path):
        leaf = _rect_cell("leaf", 0.0, -0.25, 2.0, 0.25, layer=1)
        top = gdstk.Cell("top")
        ref0 = gdstk.Reference(leaf, origin=(0.0, 0.0))
        ref1 = gdstk.Reference(leaf, origin=(5.0, 0.0))
        top.add(ref0, ref1)
        return _write(tmp_path / "h.gds", [top, leaf])

    def test_flatten_resolves_references(self, tmp_path):
        gds = self._hier_gds(tmp_path)
        structures = import_gds(gds, [_strip()])  # flatten=True default
        assert len(structures) == 2  # both instances
        x_centers = sorted(
            sum(v[0] for v in s.geometry.vertices_um) / 4.0 for s in structures
        )
        assert x_centers[0] == pytest.approx(1.0)
        assert x_centers[1] == pytest.approx(6.0)  # shifted instance

    def test_no_flatten_skips_references(self, tmp_path):
        gds = self._hier_gds(tmp_path)
        # the top cell has only references, no own polygons
        assert import_gds(gds, [_strip()], flatten=False) == ()

    def test_flatten_does_not_mutate_library_cell(self, tmp_path):
        """Importing twice must give the same count (no cumulative flatten)."""
        gds = self._hier_gds(tmp_path)
        a = import_gds(gds, [_strip()])
        b = import_gds(gds, [_strip()])
        assert len(a) == len(b) == 2

    def test_cell_name_selection(self, tmp_path):
        leaf = _rect_cell("leaf", 0.0, 0.0, 1.0, 1.0, layer=1)
        other = _rect_cell("other", 0.0, 0.0, 9.0, 9.0, layer=1)
        gds = _write(tmp_path / "two.gds", [leaf, other])
        (structure,) = import_gds(gds, [_strip()], cell_name="leaf")
        xs = sorted({v[0] for v in structure.geometry.vertices_um})
        assert xs == [0.0, 1.0]

    def test_missing_cell_name_raises(self, tmp_path):
        gds = _write(tmp_path / "a.gds", [_rect_cell("top", 0, 0, 1, 1, layer=1)])
        with pytest.raises(ValueError, match="not found"):
            import_gds(gds, [_strip()], cell_name="nope")

    def test_multiple_top_cells_requires_name(self, tmp_path):
        a = _rect_cell("a", 0, 0, 1, 1, layer=1)
        b = _rect_cell("b", 0, 0, 1, 1, layer=1)
        gds = _write(tmp_path / "ab.gds", [a, b])  # both top-level, no refs
        with pytest.raises(ValueError, match="multiple top-level"):
            import_gds(gds, [_strip()])


# --------------------------------------------------------------------------- #
# Cleaning / guards
# --------------------------------------------------------------------------- #


class TestCleaning:
    def test_min_area_filter_drops_slivers(self, tmp_path):
        cell = gdstk.Cell("top")
        cell.add(gdstk.rectangle((0, 0), (2, 2), layer=1, datatype=0))       # area 4
        cell.add(gdstk.rectangle((0, 0), (0.01, 0.01), layer=1, datatype=0))  # 1e-4
        gds = _write(tmp_path / "s.gds", [cell])
        assert len(import_gds(gds, [_strip()])) == 2  # both kept by default
        assert len(import_gds(gds, [_strip()], min_area_um2=0.1)) == 1  # sliver gone

    def test_axis_argument_sets_extrusion_axis(self, tmp_path):
        gds = _write(tmp_path / "a.gds", [_rect_cell("top", 0, 0, 2, 1, layer=1)])
        (structure,) = import_gds(gds, [_strip()], axis="x")
        assert structure.geometry.axis == "x"

    def test_bad_axis_raises(self, tmp_path):
        gds = _write(tmp_path / "a.gds", [_rect_cell("top", 0, 0, 1, 1, layer=1)])
        with pytest.raises(ValueError, match="axis must be"):
            import_gds(gds, [_strip()], axis="w")

    def test_no_layers_raises(self, tmp_path):
        gds = _write(tmp_path / "a.gds", [_rect_cell("top", 0, 0, 1, 1, layer=1)])
        with pytest.raises(ValueError, match="at least one"):
            import_gds(gds, [])

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            import_gds("/no/such/file.gds", [_strip()])

    def test_non_micron_user_unit_converted_to_um(self, tmp_path):
        # Regression (K4): a bare gdstk.read_gds returns coordinates in the
        # file's OWN user unit — an nm-authored file came back 1000x too large.
        # import_gds must convert to microns (unit=1e-6) regardless of how the
        # file was authored.
        nm_lib = gdstk.Library(name="nmlib", unit=1e-9, precision=1e-12)
        cell = nm_lib.new_cell("top")
        # 2000 x 500 in nm-units = 2.0 x 0.5 um physically.
        cell.add(gdstk.rectangle((0, 0), (2000, 500), layer=1, datatype=0))
        gds = str(tmp_path / "nm.gds")
        nm_lib.write_gds(gds)
        # The raw reader really does hand back nm-unit coordinates (the trap).
        raw = gdstk.read_gds(gds).top_level()[0].polygons[0].points
        assert raw.max() == pytest.approx(2000.0)
        (structure,) = import_gds(gds, [_strip()])
        xs = sorted({v[0] for v in structure.geometry.vertices_um})
        ys = sorted({v[1] for v in structure.geometry.vertices_um})
        assert xs == pytest.approx([0.0, 2.0])
        assert ys == pytest.approx([0.0, 0.5])

    def test_micron_user_unit_unchanged(self, tmp_path):
        # The unit conversion is a no-op for the common um-authored file: the
        # same rectangle round-trips with identical coordinates.
        gds = _write(tmp_path / "um.gds", [_rect_cell("top", 0.0, 0.0, 2.0, 0.5)])
        (structure,) = import_gds(gds, [_strip()])
        xs = sorted({v[0] for v in structure.geometry.vertices_um})
        ys = sorted({v[1] for v in structure.geometry.vertices_um})
        assert xs == [0.0, 2.0] and ys == [0.0, 0.5]

    def test_bad_thickness_raises(self):
        with pytest.raises(ValueError, match="thickness_um"):
            GdsLayer(layer=(1, 0), medium=SI, zmin_um=0.0, thickness_um=0.0)


# --------------------------------------------------------------------------- #
# Discovery helper + end-to-end into a Simulation
# --------------------------------------------------------------------------- #


class TestIntegration:
    def test_read_cell_names(self, tmp_path):
        leaf = _rect_cell("leaf", 0, 0, 1, 1, layer=1)
        top = gdstk.Cell("top")
        top.add(gdstk.Reference(leaf))
        gds = _write(tmp_path / "h.gds", [top, leaf])
        names = read_gds_cell_names(gds)
        assert set(names) == {"top", "leaf"}
        assert names[0] == "top"  # top-level first

    def test_imported_structures_build_simulation(self, tmp_path):
        gds = _write(tmp_path / "wg.gds", [_rect_cell("top", -2, -0.25, 2, 0.25, layer=1)])
        structures = import_gds(gds, [_strip()])
        sim = ph.Simulation(
            size_um=(4.0, 2.0, 2.0),
            grid=ph.UniformGridSpec(dl_um=0.05),
            run=ph.RunSpec(n_steps=5),
            background=ph.Background(permittivity=1.444 ** 2),
            boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
            structures=structures,
            sources=[
                ph.PointDipole(
                    center_um=(0.0, 1.0, 1.0),
                    polarization="Ey",
                    source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4e13),
                )
            ],
        )
        # round-trips through the wire format
        restored = ph.Simulation.model_validate_json(sim.to_wire_json())
        assert len(restored.structures) == 1
        assert isinstance(restored.structures[0].geometry, ph.PolySlab)
