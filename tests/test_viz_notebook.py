"""Workbench notebook kernel + HTTP endpoint contracts.

The notebook executes user Python against the same authoritative workspace and
run machinery the GUI uses.  These tests pin the seam the React panel consumes:
cell CRUD, queued execution with captured outputs, the ``wb`` workspace bridge
(including the workspace revision the 3D preview refresh keys on), interrupt,
restart, and cross-restart persistence of cell sources.
"""

import time

import pytest


def _client(run_root=None):
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app
    return TestClient(create_app(run_root=run_root))


def _wait_idle(client, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get("/api/notebook").json()
        if snap["kernel"]["status"] == "idle":
            return snap
        time.sleep(0.02)
    raise AssertionError("notebook kernel never returned to idle")


def _cell(snap, cell_id):
    return next(cell for cell in snap["cells"] if cell["id"] == cell_id)


def _run(client, cell_id, code=None, timeout=30.0):
    payload = {} if code is None else {"code": code}
    response = client.post(f"/api/notebook/cells/{cell_id}/run", json=payload)
    assert response.status_code == 200, response.text
    return _cell(_wait_idle(client, timeout), cell_id)


def test_fresh_notebook_seeds_the_three_cell_tour_and_idle_kernel():
    client = _client()
    snap = client.get("/api/notebook").json()
    assert snap["kernel"] == {
        "status": "idle", "running_cell": None, "queue": [],
        "execution_count": 0,
    }
    assert snap["workspace_rev"] == 0
    # A runnable mini-tour: cheat-sheet/inspect, edit-and-apply, run-and-plot.
    assert len(snap["cells"]) == 3
    for cell in snap["cells"]:
        assert cell["status"] == "idle"
        assert cell["execution_count"] is None
    assert "wb.spec" in snap["cells"][0]["code"]
    assert "wb.apply(spec)" in snap["cells"][1]["code"]
    assert "wb.run()" in snap["cells"][2]["code"]
    # every seeded cell must at least be valid python
    for cell in snap["cells"]:
        compile(cell["code"], "<seed>", "exec")


def test_cell_execution_captures_stdout_result_and_error_traceback():
    client = _client()
    first = client.get("/api/notebook").json()["cells"][0]["id"]

    cell = _run(client, first, "print('hello'); 40 + 2")
    assert cell["status"] == "ok"
    assert cell["execution_count"] == 1
    assert cell["elapsed_s"] is not None
    assert {"type": "stream", "name": "stdout", "text": "hello\n"} in cell["outputs"]
    assert {"type": "result", "text": "42"} in cell["outputs"]

    cell = _run(client, first, "x = [1]\nx[5]")
    assert cell["status"] == "error"
    error = next(o for o in cell["outputs"] if o["type"] == "error")
    assert error["ename"] == "IndexError"
    assert "<cell 2>" in error["traceback"]
    # kernel frames (server internals) never appear in user tracebacks
    assert "notebook.py" not in error["traceback"]


def test_namespace_persists_across_cells_and_run_all_runs_in_order():
    client = _client()
    snap = client.get("/api/notebook").json()
    first = snap["cells"][0]["id"]
    client.post(f"/api/notebook/cells/{first}", json={"code": "value = 21"})
    second = client.post(
        "/api/notebook/cells", json={"code": "value * 2"}).json()["cells"][-1]["id"]
    response = client.post("/api/notebook/run-all")
    assert response.status_code == 200
    snap = _wait_idle(client)
    assert _cell(snap, first)["status"] == "ok"
    assert {"type": "result", "text": "42"} in _cell(snap, second)["outputs"]


def test_cell_crud_add_move_delete_and_unknown_cell_is_404():
    client = _client()
    seeded = [c["id"] for c in client.get("/api/notebook").json()["cells"]]
    first = seeded[0]
    snap = client.post("/api/notebook/cells",
                       json={"after_id": first, "code": "b"}).json()
    added = snap["cells"][1]["id"]
    assert [c["id"] for c in snap["cells"]] == [first, added, *seeded[1:]]

    snap = client.post(f"/api/notebook/cells/{added}/move", json={"index": 0}).json()
    assert [c["id"] for c in snap["cells"]] == [added, first, *seeded[1:]]

    snap = client.post(f"/api/notebook/cells/{first}/delete").json()
    assert [c["id"] for c in snap["cells"]] == [added, *seeded[1:]]
    for cell_id in seeded[1:]:
        snap = client.post(f"/api/notebook/cells/{cell_id}/delete").json()
    # deleting the last cell leaves one fresh empty cell
    snap = client.post(f"/api/notebook/cells/{added}/delete").json()
    assert len(snap["cells"]) == 1
    assert snap["cells"][0]["code"] == ""

    assert client.post("/api/notebook/cells/nope/run", json={}).status_code == 404
    assert client.post("/api/notebook/cells/nope", json={"code": ""}).status_code == 404


def test_wb_spec_requires_an_open_workspace_then_reads_the_live_draft():
    client = _client()
    first = client.get("/api/notebook").json()["cells"][0]["id"]
    cell = _run(client, first, "wb.spec")
    assert cell["status"] == "error"
    error = next(o for o in cell["outputs"] if o["type"] == "error")
    assert error["ename"] == "LookupError"
    assert "no simulation workspace open" in error["evalue"]

    client.post("/api/workspace/new", json={})
    cell = _run(client, first, "wb.spec['size_um']")
    assert cell["status"] == "ok"
    result = next(o for o in cell["outputs"] if o["type"] == "result")
    assert result["text"] == "[4.0, 2.0, 1.2]"


def test_wb_apply_publishes_the_workspace_and_bumps_workspace_rev():
    client = _client()
    client.post("/api/workspace/new", json={})
    first = client.get("/api/notebook").json()["cells"][0]["id"]
    code = ("spec = wb.spec\n"
            "spec['size_um'] = [6.0, 2.0, 1.2]\n"
            "wb.apply(spec)\n")
    cell = _run(client, first, code)
    assert cell["status"] == "ok"
    snap = client.get("/api/notebook").json()
    assert snap["workspace_rev"] == 1

    workspace = client.get("/api/workspace").json()
    assert workspace["spec"]["size_um"] == [6.0, 2.0, 1.2]
    assert workspace["dirty"] is True

    # an invalid spec fails with the authoritative pydantic message and does
    # not advance the revision or clobber the workspace
    cell = _run(client, first, "wb.apply({'schema_version': 'nope'})")
    assert cell["status"] == "error"
    snap = client.get("/api/notebook").json()
    assert snap["workspace_rev"] == 1
    assert client.get("/api/workspace").json()["spec"]["size_um"] == [6.0, 2.0, 1.2]


def test_seeded_apply_tour_cell_runs_verbatim_against_the_starter():
    """The shipped edit-and-apply tour cell must execute as seeded against the
    ``New`` starter — the guard that the seed and the starter spec stay in
    step (e.g. the starter keeps ``run.n_steps``)."""
    client = _client()
    client.post("/api/workspace/new", json={})
    cells = client.get("/api/notebook").json()["cells"]
    cell = _run(client, cells[1]["id"])          # no code override: as seeded
    assert cell["status"] == "ok", cell["outputs"]
    assert any(o.get("text", "").startswith("applied: n_steps -> 1200")
               for o in cell["outputs"] if o["type"] == "stream")
    workspace = client.get("/api/workspace").json()
    assert workspace["spec"]["run"]["n_steps"] == 1200


def test_wb_show_emits_plotly_figures_and_matplotlib_is_auto_captured():
    client = _client()
    first = client.get("/api/notebook").json()["cells"][0]["id"]
    pytest.importorskip("matplotlib")
    code = (
        "wb.show({'data': [{'x': [1, 2], 'y': [3, 4], 'type': 'scatter'}],"
        " 'layout': {'title': 'T'}})\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([0, 1], [1, 0])\n"
    )
    cell = _run(client, first, code, timeout=90.0)
    assert cell["status"] == "ok", cell["outputs"]
    types = [o["type"] for o in cell["outputs"]]
    assert "figure" in types
    assert "image" in types
    figure = next(o for o in cell["outputs"] if o["type"] == "figure")["figure"]
    assert figure["data"][0]["x"] == [1, 2]
    image = next(o for o in cell["outputs"] if o["type"] == "image")
    assert image["mime"] == "image/png"
    assert len(image["data"]) > 100


def test_interrupt_stops_a_running_cell_and_clears_the_queue():
    client = _client()
    snap = client.get("/api/notebook").json()
    first = snap["cells"][0]["id"]
    queued = client.post("/api/notebook/cells", json={"code": "later = 1"}).json()["cells"][-1]["id"]
    client.post(f"/api/notebook/cells/{first}/run",
                json={"code": "import time\nwhile True: time.sleep(0.02)"})
    client.post(f"/api/notebook/cells/{queued}/run", json={})
    deadline = time.time() + 10
    while time.time() < deadline:
        snap = client.get("/api/notebook").json()
        if snap["kernel"]["running_cell"] == first:
            break
        time.sleep(0.02)
    assert snap["kernel"]["running_cell"] == first
    client.post("/api/notebook/interrupt")
    snap = _wait_idle(client, timeout=15.0)
    assert _cell(snap, first)["status"] == "error"
    error = next(o for o in _cell(snap, first)["outputs"] if o["type"] == "error")
    assert error["ename"] == "KeyboardInterrupt"
    # the queued cell was cleared, not run
    assert _cell(snap, queued)["status"] == "idle"
    assert _cell(snap, queued)["outputs"] == []


def test_restart_gives_a_fresh_namespace_and_keeps_cell_sources():
    client = _client()
    first = client.get("/api/notebook").json()["cells"][0]["id"]
    assert _run(client, first, "leaked = 123")["status"] == "ok"
    snap = client.post("/api/notebook/restart", json={}).json()
    assert snap["kernel"]["running_cell"] is None
    assert _cell(snap, first)["code"] == "leaked = 123"
    cell = _run(client, first, "leaked")
    assert cell["status"] == "error"
    assert any(o["type"] == "error" and o["ename"] == "NameError"
               for o in cell["outputs"])

    snap = client.post("/api/notebook/restart", json={"clear_outputs": True}).json()
    cell = _cell(snap, first)
    assert cell["outputs"] == []
    assert cell["status"] == "idle"
    assert cell["execution_count"] is None


def test_cell_sources_survive_a_sidecar_restart(tmp_path):
    run_root = tmp_path / "runs"
    with _client(run_root=run_root) as first_app:
        seeded = [c["id"]
                  for c in first_app.get("/api/notebook").json()["cells"]]
        first = seeded[0]
        for surplus in seeded[1:]:
            first_app.post(f"/api/notebook/cells/{surplus}/delete")
        first_app.post(f"/api/notebook/cells/{first}", json={"code": "kept = 7"})
        added = first_app.post(
            "/api/notebook/cells", json={"code": "kept + 1"}).json()["cells"][-1]["id"]
        assert _run(first_app, first)["status"] == "ok"

    with _client(run_root=run_root) as restarted:
        snap = restarted.get("/api/notebook").json()
        assert [c["id"] for c in snap["cells"]] == [first, added]
        assert [c["code"] for c in snap["cells"]] == ["kept = 7", "kept + 1"]
        # outputs are session-scoped; sources and counts survive
        assert all(c["outputs"] == [] for c in snap["cells"])
        assert snap["cells"][0]["execution_count"] == 1
        # the restored kernel is executable and its namespace is fresh
        cell = _run(restarted, added)
        assert cell["status"] == "error"
        assert any(o["type"] == "error" and o["ename"] == "NameError"
                   for o in cell["outputs"])


def test_notebook_endpoints_reject_malformed_payloads():
    client = _client()
    first = client.get("/api/notebook").json()["cells"][0]["id"]
    assert client.post("/api/notebook/cells",
                       json={"code": 5}).status_code == 422
    assert client.post(f"/api/notebook/cells/{first}",
                       json={"code": None}).status_code == 422
    assert client.post(f"/api/notebook/cells/{first}/move",
                       json={"index": "top"}).status_code == 422
    assert client.post(f"/api/notebook/cells/{first}/run",
                       json={"code": 5}).status_code == 422


def test_notebook_api_is_behind_the_desktop_capability_gate(tmp_path):
    from fastapi.testclient import TestClient
    from photonhub.viz.server import create_app

    capability = "ab" * 32
    app = create_app(run_root=tmp_path / "runs", launch_token=capability)
    with TestClient(app) as client:
        assert client.get("/api/notebook").status_code == 401
        authed = client.get(
            "/api/notebook",
            headers={"x-photonhub-launch-capability": capability})
        assert authed.status_code == 200
        assert "cells" in authed.json()
