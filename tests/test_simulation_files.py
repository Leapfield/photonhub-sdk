import json

import pytest

import photonhub as ph
from photonhub.components import simulation as simulation_module


def test_simulation_file_roundtrip_is_canonical_and_preserves_model(tiny_sim, tmp_path):
    before = tiny_sim
    target = tmp_path / "case.sim.json"

    saved = tiny_sim.to_file(target)

    assert saved == target
    assert tiny_sim == before
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(target.read_text(encoding="utf-8")) == tiny_sim.to_wire_dict()
    assert ph.Simulation.from_file(target) == tiny_sim


@pytest.mark.parametrize("name", ["simulation", "simulation.yaml", "simulation.json.tmp"])
def test_simulation_file_helpers_require_json_suffix(tiny_sim, tmp_path, name):
    target = tmp_path / name
    with pytest.raises(ValueError, match=r"\.json filename"):
        tiny_sim.to_file(target)
    with pytest.raises(ValueError, match=r"\.json filename"):
        ph.Simulation.from_file(target)


def test_simulation_to_file_requires_existing_parent(tiny_sim, tmp_path):
    target = tmp_path / "missing" / "case.sim.json"
    with pytest.raises(FileNotFoundError, match="parent directory"):
        tiny_sim.to_file(target)
    assert not target.parent.exists()


def test_simulation_file_helpers_reject_directory_target(tiny_sim, tmp_path):
    target = tmp_path / "case.json"
    target.mkdir()
    with pytest.raises(IsADirectoryError):
        tiny_sim.to_file(target)
    with pytest.raises(IsADirectoryError):
        ph.Simulation.from_file(target)


def test_simulation_from_file_preserves_missing_file_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="simulation spec not found"):
        ph.Simulation.from_file(tmp_path / "missing.sim.json")


def test_simulation_to_file_replace_failure_keeps_old_file_and_cleans_temp(
    tiny_sim, tmp_path, monkeypatch
):
    target = tmp_path / "case.sim.json"
    target.write_text("old-complete-document\n", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(simulation_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        tiny_sim.to_file(target)

    assert target.read_text(encoding="utf-8") == "old-complete-document\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
