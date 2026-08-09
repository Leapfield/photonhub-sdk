import copy
import hashlib
import json

import pytest

from photonhub.release import (
    APP_ID,
    PRODUCT_NAME,
    ReleaseManifestError,
    build_release_manifest,
    canonical_json_bytes,
    collect_artifacts,
    desktop_release_schema,
    load_release_manifest,
    promote_release_manifest,
    rebind_release_manifest,
    sha256_file,
    timestamp_from_epoch,
    validate_distributable_beta_api_origin,
    validate_release_manifest,
    verify_release_artifacts,
    write_release_manifest,
)


def _manifest(
    tmp_path, *, status="candidate",
    beta_api_url=None,
):
    if beta_api_url is None:
        beta_api_url = (
            "https://identity.photonhub.dev"
            if status == "promoted"
            else "https://identity.example.test"
        )
    solver = tmp_path / "solver" / "phsolver"
    solver.parent.mkdir()
    solver.write_bytes(b"solver")
    sidecar = tmp_path / "sidecar" / "photonhub-serve-viz"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"sidecar")
    notice = tmp_path / "licenses" / "THIRD-PARTY-NOTICES.md"
    notice.parent.mkdir()
    notice.write_text("notices", encoding="utf-8")
    artifacts = collect_artifacts(tmp_path)
    git_sha = "1" * 40 if status == "promoted" else "123456789abc"
    return build_release_manifest(
        status=status,
        product_version="0.0.1",
        source_git_sha=git_sha,
        input_schema_version="1.15.0-alpha.1",
        result_schema_version="1",
        physics_release_id="physics-0.0.1-beta.1",
        validation_profile_id="cpu-beta-2026-07-22",
        created_at="2026-07-22T00:00:00Z",
        platform="macos",
        arch="arm64",
        compiler="AppleClang-17.0.0",
        openmp_runtime="llvm-libomp-20.1.8",
        solver_relative_path="solver/phsolver",
        solver_info={
            "name": "phsolver", "version": "0.0.1", "git_sha": git_sha,
            "schema_major": 1, "gpu": False,
            "workbench_authorization_required": True,
        },
        solver_capabilities={"schema_major": 1, "features": ["uniform_grid"]},
        solver_sha256=sha256_file(solver),
        sidecar_relative_path="sidecar/photonhub-serve-viz",
        sidecar_sha256=sha256_file(sidecar),
        artifacts=artifacts,
        community_eula_version="community-beta-2026-07-22",
        privacy_version="desktop-beta-2026-07-22",
        third_party_notices_sha256=sha256_file(notice),
        beta_api_url=beta_api_url,
    )


def test_manifest_roundtrip_and_artifact_verification(tmp_path):
    manifest = _manifest(tmp_path)
    target = tmp_path / "release.json"
    write_release_manifest(target, manifest)
    loaded = load_release_manifest(target, artifact_root=tmp_path)
    assert loaded == manifest
    assert loaded["product"] == {
        "name": PRODUCT_NAME, "version": "0.0.1", "app_id": APP_ID,
    }
    assert loaded["services"] == {
        "beta_api_url": "https://identity.example.test",
    }


def test_artifact_mutation_fails_closed(tmp_path):
    manifest = _manifest(tmp_path)
    (tmp_path / "solver" / "phsolver").write_bytes(b"changed")
    with pytest.raises(ReleaseManifestError, match="size changed|hash changed"):
        verify_release_artifacts(manifest, tmp_path)


def test_rebind_refreshes_signed_bytes_and_packaged_inventory(tmp_path):
    manifest = _manifest(tmp_path)
    original_build_id = manifest["build"]["build_id"]
    solver = tmp_path / manifest["solver"]["relative_path"]
    solver.write_bytes(b"solver-with-native-signature")
    packaged_app = tmp_path / "app.asar"
    packaged_app.write_bytes(b"renderer")
    (tmp_path / "release.json").write_text("stale", encoding="utf-8")

    rebound = rebind_release_manifest(manifest, tmp_path)

    assert rebound["solver"]["sha256"] == sha256_file(solver)
    assert rebound["build"]["build_id"] != original_build_id
    assert "app.asar" in {item["path"] for item in rebound["artifacts"]}
    assert "release.json" not in {item["path"] for item in rebound["artifacts"]}
    verify_release_artifacts(rebound, tmp_path)


def test_rebind_rejects_promoted_manifest(tmp_path):
    manifest = _manifest(tmp_path, status="promoted")

    with pytest.raises(ReleaseManifestError, match="only candidate"):
        rebind_release_manifest(manifest, tmp_path)


def test_promotion_changes_only_status_for_exact_approved_identity(tmp_path):
    manifest = _manifest(
        tmp_path,
        beta_api_url="https://identity.photonhub.dev",
    )
    manifest["source"]["git_sha"] = "1" * 40
    manifest["solver"]["git_sha"] = "1" * 40
    manifest["legal"]["community_eula_version"] = "community-beta-2026-07-22"
    manifest["legal"]["privacy_version"] = "desktop-beta-2026-07-22"
    identity = {
        "product": manifest["product"], "source": manifest["source"],
        "contracts": manifest["contracts"], "physics": manifest["physics"],
        "services": manifest["services"],
        "build": {
            "platform": manifest["build"]["platform"],
            "arch": manifest["build"]["arch"],
            "compiler": manifest["build"]["compiler"],
            "openmp_runtime": manifest["build"]["openmp_runtime"],
        },
        "solver_sha256": manifest["solver"]["sha256"],
        "sidecar_sha256": manifest["sidecar"]["sha256"],
        "legal": manifest["legal"],
        "artifacts": manifest["artifacts"],
    }
    manifest["build"]["build_id"] = hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()

    promoted = promote_release_manifest(
        manifest,
        source_git_sha="1" * 40,
        beta_api_url="https://identity.photonhub.dev",
        physics_release_id="physics-0.0.1-beta.1",
        validation_profile_id="cpu-beta-2026-07-22",
        community_eula_version="community-beta-2026-07-22",
        privacy_version="desktop-beta-2026-07-22",
    )

    expected = copy.deepcopy(manifest)
    expected["status"] = "promoted"
    assert promoted == expected
    assert promoted["build"]["build_id"] == manifest["build"]["build_id"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_git_sha", "2" * 40, "do not match"),
        ("source_git_sha", "1" * 12, "40-character"),
        ("beta_api_url", "https://other.photonhub.dev", "do not match"),
        ("physics_release_id", "physics-other", "do not match"),
        ("validation_profile_id", "validation-other", "do not match"),
        ("community_eula_version", "other-approved-version", "do not match"),
        ("privacy_version", "other-approved-version", "do not match"),
    ],
)
def test_promotion_rejects_identity_reinterpretation(tmp_path, field, value, match):
    manifest = _manifest(
        tmp_path,
        beta_api_url="https://identity.photonhub.dev",
    )
    manifest["source"]["git_sha"] = "1" * 40
    manifest["solver"]["git_sha"] = "1" * 40
    manifest["legal"]["community_eula_version"] = "community-beta-2026-07-22"
    manifest["legal"]["privacy_version"] = "desktop-beta-2026-07-22"
    identity = {
        "product": manifest["product"], "source": manifest["source"],
        "contracts": manifest["contracts"], "physics": manifest["physics"],
        "services": manifest["services"],
        "build": {
            "platform": manifest["build"]["platform"],
            "arch": manifest["build"]["arch"],
            "compiler": manifest["build"]["compiler"],
            "openmp_runtime": manifest["build"]["openmp_runtime"],
        },
        "solver_sha256": manifest["solver"]["sha256"],
        "sidecar_sha256": manifest["sidecar"]["sha256"],
        "legal": manifest["legal"],
        "artifacts": manifest["artifacts"],
    }
    manifest["build"]["build_id"] = hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()
    inputs = {
        "source_git_sha": "1" * 40,
        "beta_api_url": "https://identity.photonhub.dev",
        "physics_release_id": "physics-0.0.1-beta.1",
        "validation_profile_id": "cpu-beta-2026-07-22",
        "community_eula_version": "community-beta-2026-07-22",
        "privacy_version": "desktop-beta-2026-07-22",
    }
    inputs[field] = value

    with pytest.raises(ReleaseManifestError, match=match):
        promote_release_manifest(manifest, **inputs)


def test_promoted_manifest_rejects_draft_identity(tmp_path):
    manifest = _manifest(tmp_path, status="promoted")
    manifest["physics"]["validation_profile_id"] = "pending"
    # build ID is bound to the validation profile; refresh it to reach the
    # promotion-specific check instead of failing on identity tampering first.
    identity = {
        "product": manifest["product"], "source": manifest["source"],
        "contracts": manifest["contracts"], "physics": manifest["physics"],
        "services": manifest["services"],
        "build": {
            "platform": manifest["build"]["platform"],
            "arch": manifest["build"]["arch"],
            "compiler": manifest["build"]["compiler"],
            "openmp_runtime": manifest["build"]["openmp_runtime"],
        },
        "solver_sha256": manifest["solver"]["sha256"],
        "sidecar_sha256": manifest["sidecar"]["sha256"],
        "legal": manifest["legal"],
        "artifacts": manifest["artifacts"],
    }
    manifest["build"]["build_id"] = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    with pytest.raises(ReleaseManifestError, match="unqualified"):
        validate_release_manifest(manifest)


def test_manifest_rejects_gpu_solver_and_unbound_sidecar(tmp_path):
    manifest = _manifest(tmp_path)
    gpu = copy.deepcopy(manifest)
    gpu["solver"]["gpu"] = True
    with pytest.raises(ReleaseManifestError, match="CPU-only"):
        validate_release_manifest(gpu)
    unbound = copy.deepcopy(manifest)
    unbound["artifacts"] = [
        item for item in unbound["artifacts"]
        if item["path"] != unbound["sidecar"]["relative_path"]
    ]
    with pytest.raises(ReleaseManifestError, match="not bound"):
        validate_release_manifest(unbound)


def test_beta_api_origin_changes_build_identity(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _manifest(
        first_root, beta_api_url="https://identity.example.test",
    )
    second = _manifest(
        second_root, beta_api_url="https://other-identity.example.test",
    )
    assert first["build"]["build_id"] != second["build"]["build_id"]


def test_candidate_accepts_explicit_loopback_beta_api_origin(tmp_path):
    manifest = _manifest(
        tmp_path, beta_api_url="http://127.0.0.1:8000",
    )
    assert manifest["services"]["beta_api_url"] == "http://127.0.0.1:8000"


def test_distributable_beta_api_origin_accepts_canonical_public_dns_origin():
    assert validate_distributable_beta_api_origin(
        "https://identity.photonhub.dev",
    ) == "https://identity.photonhub.dev"
    assert validate_distributable_beta_api_origin(
        "https://identity.photonhub.dev:8443",
    ) == "https://identity.photonhub.dev:8443"


@pytest.mark.parametrize(
    "beta_api_url",
    [
        "https://identity.example.test",
        "https://identity%2Eexample%2Etest",
        "https://identity。example。test",
        "https://127.0.0.1",
        "https://127%2E0%2E0%2E1",
        "https://127.1",
        "https://2130706433",
        "https://0x7f000001",
        "https://identity.photonhub.dev:443",
        "https://IDENTITY.photonhub.dev",
    ],
)
def test_distributable_beta_api_origin_rejects_reserved_or_noncanonical_host(
    beta_api_url,
):
    with pytest.raises(ReleaseManifestError, match="signed or promoted"):
        validate_distributable_beta_api_origin(beta_api_url)


def test_promoted_manifest_rejects_reserved_beta_api_origin(tmp_path):
    with pytest.raises(ReleaseManifestError, match="signed or promoted"):
        _manifest(
            tmp_path,
            status="promoted",
            beta_api_url="https://identity.example.test",
        )


@pytest.mark.parametrize(
    ("status", "beta_api_url", "match"),
    [
        ("candidate", "http://identity.example.test", "must use HTTPS"),
        ("promoted", "http://localhost:8000", "must use HTTPS"),
        ("candidate", "https://user@identity.example.test", "bare origin"),
        ("candidate", "https://identity.example.test/v1", "bare origin"),
        ("candidate", "https://identity.example.test?tenant=beta", "bare origin"),
    ],
)
def test_manifest_rejects_unsafe_beta_api_origin(
    tmp_path, status, beta_api_url, match,
):
    with pytest.raises(ReleaseManifestError, match=match):
        _manifest(tmp_path, status=status, beta_api_url=beta_api_url)


def test_collect_artifacts_rejects_symlink(tmp_path):
    original = tmp_path / "real"
    original.write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(original)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ReleaseManifestError, match="not a regular file"):
        collect_artifacts(tmp_path)


def test_reproducible_timestamp(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    assert timestamp_from_epoch() == "1970-01-01T00:00:00Z"
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-an-integer")
    with pytest.raises(ReleaseManifestError, match="integer"):
        timestamp_from_epoch()


def test_committed_schema_matches_generator():
    schema_path = __import__("pathlib").Path(__file__).resolve().parents[2] / "schemas" / "desktop_release_v1.json"
    assert json.loads(schema_path.read_text(encoding="utf-8")) == desktop_release_schema()
