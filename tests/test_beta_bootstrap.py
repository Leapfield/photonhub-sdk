import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import photonhub as ph

# CI deliberately runs this suite from ``photonhub/``. The beta bootstrap and
# smoke entrypoints live at the repository root, so make that source boundary
# explicit instead of depending on the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import beta_cpu_smoke, bootstrap_beta


def test_bootstrap_paths_default_under_repo(tmp_path):
    paths = bootstrap_beta.resolve_paths(tmp_path)
    assert paths.repo == tmp_path.resolve()
    assert paths.venv == (tmp_path / ".venv").resolve()
    assert paths.build == (tmp_path / "build").resolve()
    assert paths.solver == (tmp_path / "build" / "phsolver").resolve()


def test_bootstrap_paths_resolve_relative_overrides_under_repo(tmp_path):
    paths = bootstrap_beta.resolve_paths(tmp_path, "envs/beta", "artifacts/cpu")
    assert paths.venv == (tmp_path / "envs" / "beta").resolve()
    assert paths.build == (tmp_path / "artifacts" / "cpu").resolve()


def test_workbench_command_carries_custom_runtime_paths(tmp_path):
    defaults = bootstrap_beta.resolve_paths(tmp_path)
    default_command = bootstrap_beta.workbench_command(defaults)
    assert "PHOTONHUB_DEV_AUTH_BYPASS=1" in default_command
    assert f"PHOTONHUB_PY={defaults.venv_python}" in default_command
    assert f"PHOTONHUB_SOLVER={defaults.solver}" in default_command
    assert default_command.endswith("./desktop/dev.sh")
    custom = bootstrap_beta.resolve_paths(tmp_path, "envs/beta", "artifacts/cpu")
    command = bootstrap_beta.workbench_command(custom)
    assert f"PHOTONHUB_PY={custom.venv_python}" in command
    assert f"PHOTONHUB_SOLVER={custom.solver}" in command
    assert command.endswith("./desktop/dev.sh")


def test_bootstrap_help_defines_skip_smoke_as_both_lifecycle_checks():
    help_text = bootstrap_beta._parser().format_help()
    assert "CPU and Workbench lifecycle smokes" in help_text


def test_desktop_install_runs_electron_postinstall_when_probe_fails(
    tmp_path, monkeypatch
):
    paths = bootstrap_beta.resolve_paths(tmp_path)
    for project in (tmp_path / "desktop" / "ui", tmp_path / "desktop" / "electron"):
        project.mkdir(parents=True)
        (project / "package-lock.json").write_text("{}")
    installer = (
        tmp_path
        / "desktop"
        / "electron"
        / "node_modules"
        / "electron"
        / "install.js"
    )
    installer.parent.mkdir(parents=True)
    installer.write_text("// fixture")
    calls = []
    probe_count = 0

    def fake_run(command, **_kwargs):
        nonlocal probe_count
        calls.append([str(value) for value in command])
        if "const fs=require('fs')" in " ".join(calls[-1]):
            probe_count += 1
            if probe_count == 1:
                raise bootstrap_beta.BootstrapError("missing Electron runtime")

    monkeypatch.setattr(bootstrap_beta, "run", fake_run)
    bootstrap_beta._install_desktop(paths, Path("node"), Path("npm"))

    assert ["node", str(installer)] in calls
    assert probe_count == 2


def test_desktop_check_runs_both_production_builds(tmp_path, monkeypatch):
    paths = bootstrap_beta.resolve_paths(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(
            ([str(value) for value in command], Path(kwargs["cwd"]).resolve())
        )

    monkeypatch.setattr(bootstrap_beta, "run", fake_run)
    bootstrap_beta._build_desktop(paths, Path("npm"))

    assert calls == [
        (["npm", "run", "build"], (tmp_path / "desktop" / "ui").resolve()),
        (
            ["npm", "run", "build"],
            (tmp_path / "desktop" / "electron").resolve(),
        ),
    ]


def test_source_env_puts_checkout_before_stale_pythonpath(tmp_path, monkeypatch):
    paths = bootstrap_beta.resolve_paths(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/tmp/stale-checkout")

    env = bootstrap_beta._source_env(paths)

    assert env["PYTHONPATH"].split(bootstrap_beta.os.pathsep) == [
        str((tmp_path / "photonhub").resolve()),
        "/tmp/stale-checkout",
    ]


def test_solver_check_rejects_unknown_provenance(tmp_path, monkeypatch):
    paths = bootstrap_beta.resolve_paths(tmp_path)
    paths.solver.parent.mkdir(parents=True)
    paths.solver.write_text("fixture")
    paths.solver.chmod(0o755)

    monkeypatch.setattr(
        bootstrap_beta,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout='{"gpu": false, "git_sha": "unknown"}', stderr=""
        ),
    )

    with pytest.raises(bootstrap_beta.BootstrapError, match="provenance is unknown"):
        bootstrap_beta._check_solver_source(paths)


def test_bootstrap_installs_venv_cmake_when_host_is_too_old(tmp_path, monkeypatch):
    paths = bootstrap_beta.resolve_paths(tmp_path)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append([str(part) for part in command])
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(bootstrap_beta, "run", fake_run)
    monkeypatch.setattr(
        bootstrap_beta, "_which",
        lambda name: Path("/usr/bin/cmake") if name == "cmake" else None,
    )
    monkeypatch.setattr(bootstrap_beta, "_cmake_version", lambda _path: (3, 20))

    bootstrap_beta._install_python(paths)

    assert any(command[-1] == "cmake>=3.24" for command in commands)


def test_desktop_prerequisite_rejects_unsupported_node(monkeypatch):
    tools = {
        "c++": Path("/usr/bin/c++"),
        "node": Path("/usr/bin/node"),
        "npm": Path("/usr/bin/npm"),
    }
    monkeypatch.setattr(bootstrap_beta, "_which", tools.get)
    monkeypatch.setattr(bootstrap_beta, "_node_version", lambda _path: (17, 9))

    with pytest.raises(bootstrap_beta.BootstrapError, match="Node.js 22"):
        bootstrap_beta._require_host_prerequisites(True)


def test_desktop_prerequisite_rejects_non_lts_node(monkeypatch):
    tools = {
        "c++": Path("/usr/bin/c++"),
        "node": Path("/usr/bin/node"),
        "npm": Path("/usr/bin/npm"),
    }
    monkeypatch.setattr(bootstrap_beta, "_which", tools.get)
    monkeypatch.setattr(bootstrap_beta, "_node_version", lambda _path: (26, 4))
    monkeypatch.setattr(bootstrap_beta, "_node_lts_name", lambda _path: "")

    with pytest.raises(bootstrap_beta.BootstrapError, match="non-LTS 26.4"):
        bootstrap_beta._require_host_prerequisites(True)


def test_smoke_sim_is_tiny_and_wire_roundtrips():
    sim = beta_cpu_smoke.build_smoke_sim()
    assert sim.cost_estimate().cells_per_axis == (8, 8, 8)
    assert sim.run.n_steps == 24
    assert sim.boundaries == ph.Boundaries(x="periodic", y="periodic", z="periodic")
    assert ph.Simulation.from_wire_json(sim.to_wire_json()) == sim


def test_smoke_refuses_nonempty_output(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    (output / "owned.txt").write_text("do not overwrite")
    with pytest.raises(RuntimeError, match="refusing to reuse non-empty"):
        beta_cpu_smoke.run_smoke(Path(sys.executable), output)


def test_beta_cpu_smoke_with_current_solver(tmp_path):
    solver = ph.find_solver()
    if solver is None:
        pytest.skip("needs a current phsolver build")
    summary = beta_cpu_smoke.run_smoke(solver, tmp_path / "result")
    assert summary["ok"] is True
    assert summary["backend"] == "cpu"
    assert summary["grid"] == [8, 8, 8]
    assert summary["steps_run"] == 24
    assert summary["bundle_reopen"] is True
    assert summary["geometry_integrity"] == "matched"
