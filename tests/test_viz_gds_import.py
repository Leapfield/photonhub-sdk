"""Workbench GDS import seam — /api/gds/inspect + /api/gds/import contracts.

The GDS benchmark builds devices with :func:`photonhub.gds.import_gds` in Python;
these endpoints expose the same converter to the desktop Structures editor so a
layout can be imported without hand-typing polygon point lists. Tests cover the
catalog (cells/layers/bbox), the wire-dict conversion the renderer appends
through its ordinary mutate+validate path, and the structured error contract
the dialog renders inline.
"""

import base64

import pytest

from photonhub.viz import service


def _client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app
    return TestClient(create_app())


def _write_demo_gds(path, *, extra_top=False):
    """A two-layer demo layout: (1,0) two rectangles, (2,0) one triangle.

    The (1,0) content lives partly in a referenced child cell so the flatten
    path is exercised; ``extra_top`` adds a second, disjoint top-level cell to
    model multi-top foundry files.
    """
    gdstk = pytest.importorskip("gdstk")
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0.0, 0.0), (2.0, 0.5), layer=1, datatype=0))
    top = lib.new_cell("top")
    top.add(gdstk.rectangle((-3.0, -1.0), (-1.0, -0.5), layer=1, datatype=0))
    # clockwise triangle: import must normalize winding to CCW
    top.add(gdstk.Polygon([(0.0, 2.0), (1.0, 3.0), (2.0, 2.0)],
                          layer=2, datatype=0))
    top.add(gdstk.Reference(child, origin=(1.0, 1.0)))
    if extra_top:
        other = lib.new_cell("alt_top")
        other.add(gdstk.rectangle((5.0, 5.0), (6.0, 6.0), layer=7, datatype=1))
    out = path / ("demo_multi.gds" if extra_top else "demo.gds")
    lib.write_gds(str(out))
    return out


def _b64(path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _si_layer(layer=(1, 0), **overrides):
    spec = {"layer": list(layer), "zmin_um": 0.49, "thickness_um": 0.22,
            "permittivity": 3.476 ** 2}
    spec.update(overrides)
    return spec


def test_gds_inspect_catalogs_cells_layers_and_bbox(tmp_path):
    gds_file = _write_demo_gds(tmp_path)
    with _client() as client:
        by_path = client.post("/api/gds/inspect", json={"path": str(gds_file)})
        assert by_path.status_code == 200
        info = by_path.json()
        # flatten resolves the referenced child; only `top` is top-level and
        # therefore listed first.
        assert info["cells"][0] == "top"
        assert info["cell"] == "top"
        by_layer = {tuple(entry["layer"]): entry for entry in info["layers"]}
        assert set(by_layer) == {(1, 0), (2, 0)}
        assert by_layer[(1, 0)]["polygons"] == 2  # own rect + referenced rect
        assert by_layer[(2, 0)]["polygons"] == 1
        assert by_layer[(1, 0)]["bbox_um"] == [[-3.0, -1.0], [3.0, 1.5]]
        assert info["bbox_um"] == [[-3.0, -1.0], [3.0, 3.0]]

        # the Electron bridge path: identical result from a base64 upload
        by_upload = client.post(
            "/api/gds/inspect", json={"content_base64": _b64(gds_file)})
        assert by_upload.status_code == 200
        assert by_upload.json() == info


def test_gds_inspect_defaults_to_first_top_cell_when_ambiguous(tmp_path):
    gds_file = _write_demo_gds(tmp_path, extra_top=True)
    with _client() as client:
        info = client.post(
            "/api/gds/inspect", json={"path": str(gds_file)}).json()
        # multi-top files must stay inspectable — the dialog needs this listing
        # to offer the cell choice at all.
        assert info["cell"] == "top"
        assert set(info["cells"]) >= {"top", "alt_top"}

        alt = client.post("/api/gds/inspect", json={
            "path": str(gds_file), "cell_name": "alt_top"}).json()
        assert alt["cell"] == "alt_top"
        assert [tuple(entry["layer"]) for entry in alt["layers"]] == [(7, 1)]


def test_gds_import_returns_named_offset_wire_structures(tmp_path):
    gds_file = _write_demo_gds(tmp_path)
    with _client() as client:
        response = client.post("/api/gds/import", json={
            "content_base64": _b64(gds_file),
            "layers": [
                _si_layer(),
                _si_layer(layer=(2, 0), permittivity=2.0 ** 2,
                          zmin_um=0.71, thickness_um=0.15),
            ],
            "offset_um": [10.0, 20.0],
            "name_prefix": "demo",
        })
        assert response.status_code == 200
        payload = response.json()
        assert payload["count"] == 3
        assert payload["per_layer"] == [
            {"layer": [1, 0], "count": 2}, {"layer": [2, 0], "count": 1}]

        names = [s["name"] for s in payload["structures"]]
        assert names == ["demo_L1_0_p1", "demo_L1_0_p2", "demo_L2_0"]
        for structure in payload["structures"]:
            assert structure["geometry"]["type"] == "polyslab"
            assert structure["geometry"]["axis"] == "z"

        triangle = payload["structures"][2]
        assert triangle["geometry"]["slab_bounds_um"] == [0.71, pytest.approx(0.86)]
        assert triangle["medium"]["permittivity"] == pytest.approx(4.0)
        # drawing-plane offset applied, winding normalized to CCW
        vertices = triangle["geometry"]["vertices_um"]
        assert sorted(vertices) == [[10.0, 22.0], [11.0, 23.0], [12.0, 22.0]]
        area2 = sum(
            vertices[i][0] * vertices[(i + 1) % 3][1]
            - vertices[(i + 1) % 3][0] * vertices[i][1]
            for i in range(3))
        assert area2 > 0

        # the renderer's append path: imported structures validate as-is
        spec = client.post("/api/workspace/new", json={}).json()["spec"]
        spec["structures"] = (
            list(spec.get("structures") or []) + payload["structures"])
        validated = client.post("/api/workspace/validate", json={"spec": spec})
        assert validated.status_code == 200
        echoed = validated.json()["spec"]["structures"]
        assert [s.get("name") for s in echoed[-3:]] == names


def test_gds_import_min_area_drops_slivers(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("top")
    cell.add(gdstk.rectangle((0, 0), (2.0, 1.0), layer=1, datatype=0))
    cell.add(gdstk.rectangle((3.0, 0), (3.001, 1.0), layer=1, datatype=0))
    gds_file = tmp_path / "sliver.gds"
    lib.write_gds(str(gds_file))
    with _client() as client:
        full = client.post("/api/gds/import", json={
            "path": str(gds_file), "layers": [_si_layer()]}).json()
        assert full["count"] == 2
        filtered = client.post("/api/gds/import", json={
            "path": str(gds_file), "layers": [_si_layer()],
            "min_area_um2": 0.01}).json()
        assert filtered["count"] == 1


def test_gds_error_contract(tmp_path):
    gds_file = _write_demo_gds(tmp_path)
    content = _b64(gds_file)
    junk = base64.b64encode(b"not a gds stream").decode()
    with _client() as client:
        def import_error(payload, status, code=None):
            response = client.post("/api/gds/import", json=payload)
            assert response.status_code == status
            detail = response.json()["detail"]
            if code is not None:
                assert detail["code"] == code
            return detail

        import_error({"content_base64": content, "layers": []},
                     422, "gds_invalid")
        import_error({"content_base64": content, "cell_name": "nope",
                      "layers": [_si_layer()]}, 422, "gds_cell_not_found")
        import_error({"content_base64": content,
                      "layers": [_si_layer(layer=(9, 9))]}, 422, "gds_empty")
        import_error({"content_base64": content,
                      "layers": [_si_layer(thickness_um=-1.0)]},
                     422, "gds_invalid")
        import_error({"content_base64": content,
                      "layers": [_si_layer(permittivity=0.5)]},
                     422, "gds_invalid")
        import_error({"content_base64": junk, "layers": [_si_layer()]},
                     422, "gds_invalid")
        import_error({"path": str(tmp_path / "missing.gds"),
                      "layers": [_si_layer()]}, 404, "gds_not_found")

        no_source = client.post("/api/gds/import", json={
            "layers": [_si_layer()]})
        assert no_source.status_code == 422
        bad_b64 = client.post("/api/gds/inspect", json={"content_base64": "!!"})
        assert bad_b64.status_code == 422


def test_gds_upload_size_limit(tmp_path, monkeypatch):
    gds_file = _write_demo_gds(tmp_path)
    monkeypatch.setattr(service, "GDS_MAX_BYTES", 16)
    with _client() as client:
        upload = client.post(
            "/api/gds/inspect", json={"content_base64": _b64(gds_file)})
        assert upload.status_code == 413
        assert upload.json()["detail"]["code"] == "gds_too_large"
        local = client.post(
            "/api/gds/inspect", json={"path": str(gds_file)})
        assert local.status_code == 413
