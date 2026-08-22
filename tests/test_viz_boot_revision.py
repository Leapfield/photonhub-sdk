"""A restarted sidecar over the same unchanged bundle must keep its result
revision (a sidecar crash must not strand the desktop app on 409 stale-revision
errors), while a bundle that actually changed still invalidates readers."""

from pathlib import Path

from photonhub.viz.server import _boot_result_revision


class _FakeData:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir


def test_same_bundle_yields_the_same_revision(tmp_path: Path):
    (tmp_path / "manifest.json").write_text('{"a": 1}')
    (tmp_path / "sim.json").write_text('{"b": 2}')
    first = _boot_result_revision(_FakeData(tmp_path))
    second = _boot_result_revision(_FakeData(tmp_path))
    assert first and first == second


def test_changed_bundle_changes_the_revision(tmp_path: Path):
    (tmp_path / "manifest.json").write_text('{"a": 1}')
    before = _boot_result_revision(_FakeData(tmp_path))
    (tmp_path / "manifest.json").write_text('{"a": 2}')
    assert _boot_result_revision(_FakeData(tmp_path)) != before


def test_no_bundle_has_no_revision():
    assert _boot_result_revision(None) is None
