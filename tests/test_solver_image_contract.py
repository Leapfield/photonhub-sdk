from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "solver.Dockerfile"
WORKFLOW = ROOT / ".github" / "workflows" / "docker-publish.yml"
DOCKERIGNORE = ROOT / ".dockerignore"


def _instructions(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_amd_release_image_is_explicitly_mi300x_only_and_gate_ready():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG AMD_GFX_ARCHS=gfx942" in dockerfile
    assert '-DPHCORE_GFX_ARCHS="${AMD_GFX_ARCHS}"' in dockerfile
    assert 'test "${AMD_GFX_ARCHS}" = "gfx942"' in dockerfile
    assert "-DBUILD_TESTING=ON" in dockerfile
    assert "--target phsolver phcore_tests" in dockerfile
    assert "COPY --from=build /build/phcore_tests /usr/local/bin/phcore_tests" in dockerfile
    assert "COPY docker/run-gpu-equivalence.sh" in dockerfile
    dockerignore = DOCKERIGNORE.read_text(encoding="utf-8")
    assert "!docker/run-gpu-equivalence.sh" in dockerignore
    gate = (ROOT / "docker" / "run-gpu-equivalence.sh").read_text(
        encoding="utf-8",
    )
    assert "--gtest_filter='GpuEquivalence.*' --gtest_list_tests" in gate
    assert (
        'exec "$test_bin" --gtest_filter=\'GpuEquivalence.*\' '
        "--gtest_color=yes"
    ) in gate
    assert 'exec "$test_bin" --gtest_color=yes' not in gate


def test_solver_image_has_full_source_identity_and_no_entrypoint():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    instructions = _instructions(dockerfile)

    assert 'test "${#SOURCE_SHA}" -eq 40' in dockerfile
    assert 'org.opencontainers.image.revision="${SOURCE_SHA}"' in dockerfile
    assert not any(re.match(r"^ENTRYPOINT\b", line, re.IGNORECASE)
                   for line in instructions)


def test_publish_workflow_pins_amd_rocm_and_checks_runpod_contract():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    rocm_digest = (
        "rocm/dev-ubuntu-24.04@sha256:"
        "749f9ee120c739682cc2e1553e62632c2676f98bc49e4d8133f380e0af682bcc"
    )

    assert workflow.count(rocm_digest) == 2
    assert "options: [amd, nvidia]" in workflow
    assert "default: amd" in workflow
    assert "AMD_GFX_ARCHS=gfx942" in workflow
    assert "org.opencontainers.image.revision" in workflow
    assert ".Config.Entrypoint" in workflow
    assert "--gtest_filter='GpuEquivalence.*' --gtest_list_tests" in workflow
    assert "Publish tested digest under source-qualified tag" in workflow
    assert 'docker buildx imagetools create --tag "$source" "$SOURCE_IMAGE"' in workflow
    assert "Publish NVIDIA stable tag after hosted packaging gate" in workflow
    assert "if: inputs.gpu_platform == 'nvidia'" in workflow
    assert 'stable="$IMAGE_REPOSITORY:cuda"' in workflow
    assert "Record AMD candidate awaiting the MI300X hardware gate" in workflow
    assert 'stable="$IMAGE_REPOSITORY:rocm"' not in workflow


def test_bare_solver_image_defaults_to_the_mi300x_rocm_candidate():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    rocm_digest = (
        "rocm/dev-ubuntu-24.04@sha256:"
        "749f9ee120c739682cc2e1553e62632c2676f98bc49e4d8133f380e0af682bcc"
    )

    assert dockerfile.count(f"ARG BUILD_BASE={rocm_digest}") == 1
    assert dockerfile.count(f"ARG RUNTIME_BASE={rocm_digest}") == 1
    assert "ARG GPU_PLATFORM=amd" in dockerfile
