from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlencode, urlsplit
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "publish_desktop_beta_gcs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "_test_publish_desktop_beta_gcs",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


publish = _load_script()


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def _authorization_proof(
    *,
    state: str,
    run_id: int,
    workflow_name: str,
    workflow_path: str,
    head_sha: str,
    head_branch: str,
    environment: str,
    only_branch_policy: str,
    jobs: dict[str, str],
) -> bytes:
    completed = state in {"completed", "completed_job"}
    return _canonical({
        "schema": "photonhub.github-protected-environment-approval.v1",
        "verified": True,
        "authorization_state": state,
        "repository": "Leapfield/PhotonHub",
        "repository_id": 55,
        "repository_node_id": "R_repo55",
        "run": {
            "id": run_id,
            "attempt": 1,
            "workflow_name": workflow_name,
            "workflow_path": workflow_path,
            "event": "workflow_dispatch",
            "head_sha": head_sha,
            "head_branch": head_branch,
            "status": "completed" if state == "completed" else "in_progress",
            "conclusion": "success" if state == "completed" else None,
            "actor_id": 7,
            "triggering_actor_id": 8,
        },
        "protected_environment": {
            "id": run_id + 10,
            "name": environment,
            "created_at": "2026-07-20T09:00:00Z",
            "updated_at": "2026-07-22T09:59:00Z",
            "can_admins_bypass": False,
            "prevent_self_review": True,
            "required_reviewers": [{"type": "User", "id": 90}],
            "protected_branches": False,
            "custom_branch_policies": True,
            "only_branch_policy": only_branch_policy,
        },
        "approval_history": [{
            "state": "approved",
            "environment_id": run_id + 10,
            "environment_name": environment,
            "reviewer_user_id": 92,
            "run_actor_id": 7,
            "triggering_actor_id": 8,
            "approved_environments": [
                {"id": run_id + 10, "name": environment}
            ],
            "occurrences": 1,
        }],
        "jobs": [
            {
                "binding": binding,
                "id": run_id * 10 + index,
                "name": name,
                "run_id": run_id,
                "run_attempt": 1,
                "check_run_id": run_id * 100 + index,
                "check_run_node_id": f"CR_{run_id}_{index}",
                "started_at": f"2026-07-22T10:0{index}:00Z",
                "completed_at": (
                    f"2026-07-22T10:1{index}:00Z" if completed else None
                ),
                "status": "completed" if completed else "in_progress",
                "conclusion": "success" if completed else None,
                "deployment": {
                    "id": run_id * 1000 + index,
                    "node_id": f"D_{run_id}_{index}",
                    "workflow_job_id": run_id * 10 + index,
                    "check_run_id": run_id * 100 + index,
                    "repository_id": 55,
                    "environment": environment,
                    "commit_sha": head_sha,
                    "ref": head_branch,
                    "deployment_state": "ACTIVE",
                    "latest_status":
                        "SUCCESS" if completed else "IN_PROGRESS",
                    "latest_status_node_id": f"DS_{run_id}_{index}",
                    "latest_status_created_at":
                        f"2026-07-22T10:0{index}:30Z",
                    "latest_status_updated_at":
                        f"2026-07-22T10:0{index}:45Z",
                    "authorization_status_id":
                        run_id * 10000 + index,
                    "authorization_status_node_id":
                        f"DS_{run_id}_{index}",
                    "authorization_status":
                        "success" if completed else "in_progress",
                    "authorization_status_created_at":
                        f"2026-07-22T10:0{index}:30Z",
                    "authorization_status_updated_at":
                        f"2026-07-22T10:0{index}:45Z",
                },
            }
            for index, (binding, name) in enumerate(sorted(jobs.items()), 1)
        ],
    })


def _mi300x_gate(source_sha: str) -> dict[str, object]:
    return {
        "github_repository": "Leapfield/PhotonHub",
        "workflow_path": ".github/workflows/mi300x-hardware-gate.yml",
        "workflow_sha": source_sha,
        "run_id": "808",
        "run_attempt": 1,
        "artifact_id": "909",
        "artifact_digest": f"sha256:{'4' * 64}",
        "report_sha256": "5" * 64,
        "solver_source_sha": source_sha,
        "solver_image":
            f"ghcr.io/leapfield/photonhub-solver@sha256:{'3' * 64}",
    }


def _mi300x_gate_verification(source_sha: str) -> dict[str, object]:
    return {
        "schema": "photonhub.mi300x.gate-verification.v3",
        "verified": True,
        "staging_identity_sha256": "7" * 64,
        "manifest_sha256": "e" * 64,
        **_mi300x_gate(source_sha),
        "lease_sha256": "8" * 64,
        "source_provenance": {
            "default_branch": "main",
            "workflow_path":
                ".github/workflows/mi300x-hardware-gate.yml",
            "workflow_blob_sha": "9" * 40,
            "workflow_sha256": "b" * 64,
            "workflow_matches_default": True,
            "gate_script_path": "scripts/runpod_mi300x_gate.py",
            "gate_script_blob_sha": "a" * 40,
            "gate_script_sha256": "c" * 64,
            "gate_script_matches_default": True,
        },
        "workflow_job": {
            "id": "1001",
            "name": "Exact-digest MI300X equivalence",
            "run_id": "808",
            "run_attempt": 1,
            "check_run_id": "1003",
            "started_at": "2026-07-22T09:59:00Z",
            "completed_at": "2026-07-22T10:03:00Z",
            "completed": True,
        },
        "protected_environment": {
            "id": 1004,
            "name": "mi300x-hardware-gate",
            "created_at": "2026-07-20T09:00:00Z",
            "updated_at": "2026-07-22T09:58:00Z",
            "can_admins_bypass": False,
            "prevent_self_review": True,
            "required_reviewer_count": 1,
            "protected_branches": False,
            "custom_branch_policies": True,
            "only_branch": "main",
        },
        "run_approval": {
            "state": "approved",
            "environment_id": 1004,
            "environment_name": "mi300x-hardware-gate",
            "reviewer_user_id": 42,
            "run_actor_id": 7,
            "triggering_actor_id": 8,
        },
        "deployment": {
            "id": "1002",
            "workflow_job_id": "1001",
            "check_run_id": "1003",
            "environment": "mi300x-hardware-gate",
            "commit_sha": source_sha,
            "state": "ACTIVE",
            "latest_status": "SUCCESS",
        },
        "provider_create_contract": {
            "mutation": "podFindAndDeployOnDemand",
            "deadman_field": "terminateAfter",
            "terminate_after": "2026-07-22T10:30:00.000Z",
            "deadman_create_acknowledged": True,
            "price_ceiling_field": "deployCost",
            "requested_hourly_usd": "2.50",
            "price_ceiling_create_acknowledged": True,
        },
        "cleanup_confirmed": True,
    }


def _staging_deployment_verification() -> dict[str, object]:
    return {
        "schema": "photonhub.staging-deployment-verification.v2",
        "verified": True,
        "manifest_sha256": "e" * 64,
        "staging_identity_sha256": "7" * 64,
        "project_sha256": hashlib.sha256(
            b"beta-staging-project"
        ).hexdigest(),
        "region": "us-central1",
        "profile": "mi300x",
        "release_mode": "acceptance-bootstrap",
        "plan_sha256": "6" * 64,
        "bootstrap_receipt_sha256": "d" * 64,
        "admission_enable_receipt_sha256": "f" * 64,
    }


def _beta_e2e(
    *,
    source_sha: str,
    artifact_digest: str,
    installer_sha256: str,
    platform: str,
    arch: str,
) -> dict[str, object]:
    quote = 2_000_000
    actual = 1_250_000
    available = 5_000_000 - actual
    digest = "c" * 64
    gate_verification = _mi300x_gate_verification(source_sha)
    staging_verification = _staging_deployment_verification()
    return {
        "schema": 1,
        "bindings": {
            "desktop_repository": "Leapfield/PhotonHub",
            "desktop_source_git_sha": source_sha,
            "candidate_artifact_digest": artifact_digest,
            "installer_sha256": installer_sha256,
            "platform": platform,
            "arch": arch,
            "cloud_repository": "Leapfield/photonhub-cloud-beta-auth",
            "cloud_source_git_sha": "d" * 40,
            "staging_manifest_sha256": "e" * 64,
            "release_mode": "acceptance-bootstrap",
            "staging_deployment_verification": staging_verification,
            "staging_deployment_verification_sha256": "6" * 64,
            "staging_deployment_verification_projection_sha256":
                _digest(_canonical(staging_verification)),
            "beta_api_url": "https://identity.photonhub.dev",
            "api_image":
                f"ghcr.io/leapfield/photonhub-cloud-api@sha256:{'1' * 64}",
            "worker_image":
                f"ghcr.io/leapfield/photonhub-cloud-worker@sha256:{'2' * 64}",
            "solver_git_sha": source_sha,
            "solver_image":
                f"ghcr.io/leapfield/photonhub-solver@sha256:{'3' * 64}",
            "mi300x_hardware_gate": _mi300x_gate(source_sha),
            "mi300x_gate_verification_sha256":
                _digest(_canonical(gate_verification)),
        },
        "identity": {
            "invitation_request_sha256": digest,
            "account_sha256": digest,
            "cloud_gpu": True,
            "initial_available_micros": 5_000_000,
            "initial_reserved_micros": 0,
        },
        "gpu": {
            "menu_devices": ["gpu:mi300x"],
            "device": "gpu:mi300x",
            "vendor": "AMD",
            "arch": "gfx942",
            "provider_gpu_type": "AMD Instinct MI300X OAM",
            "provider_cloud_type": "SECURE",
            "quote_sha256": digest,
            "job_sha256": digest,
            "quote_micros": quote,
            "actual_micros": actual,
            "refunded_micros": quote - actual,
            "final_available_micros": available,
            "final_reserved_micros": 0,
            "active_managed_gpu_jobs": 0,
        },
        "ledger": {
            "topup_entry_micros": 5_000_000,
            "reservation_entry_micros": -quote,
            "refund_entry_micros": quote,
            "settlement_entry_micros": -actual,
            "entry_sum_micros": available,
        },
        "archive": {
            "cpu_simulation_sha256": digest,
            "cpu_result_sha256": digest,
            "cloud_result_sha256": digest,
            "cloud_reopened_sha256": digest,
        },
        "recovery": {
            "ambiguous_name_sha256": digest,
            "ambiguous_job_sha256": digest,
            "cleanup_job_sha256": digest,
            "provider_resource_sha256": digest,
            "staging_prefix_sha256": digest,
            "matched_jobs": 1,
            "duplicate_jobs": 0,
            "reservation_winners": 1,
            "cleanup_pending_observed": True,
            "cleanup_retry_completed": True,
            "balance_before_micros": available,
            "balance_after_micros": available,
            "reserved_after_micros": 0,
            "active_managed_gpu_jobs_after": 0,
            "provider_absent": True,
            "staging_absent": True,
        },
        "audit_evidence_sha256": digest,
    }


class DesktopBetaGcsPublishTests(unittest.TestCase):
    version = "1.2.3-beta.4"
    source_sha = "a" * 40

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "release"
        self.root.mkdir()
        self._create_valid_release()

    def _release_manifest(
        self,
        *,
        platform: str,
        arch: str,
        validation_profile_id: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        solver = b"signed-cpu-solver"
        sidecar = b"signed-sidecar"
        candidate = publish.release.build_release_manifest(
            status="candidate",
            product_version=self.version,
            source_git_sha=self.source_sha,
            input_schema_version="simulation-v1",
            result_schema_version="result-v1",
            physics_release_id="physics-release-2026-07",
            validation_profile_id=validation_profile_id,
            created_at="2026-07-22T00:00:00Z",
            platform=platform,
            arch=arch,
            compiler=f"approved-{platform}-compiler-1",
            openmp_runtime=f"approved-{platform}-openmp-1",
            solver_relative_path=(
                "solver/phsolver.exe"
                if platform == "windows"
                else "solver/phsolver"
            ),
            solver_info={
                "name": "phsolver",
                "version": "1.0.0",
                "git_sha": self.source_sha,
                "schema_major": 1,
                "gpu": False,
                "workbench_authorization_required": True,
            },
            solver_capabilities={"fdtd": True},
            solver_sha256=_digest(solver),
            sidecar_relative_path=(
                "sidecar/photonhub-serve-viz.exe"
                if platform == "windows"
                else "sidecar/photonhub-serve-viz"
            ),
            sidecar_sha256=_digest(sidecar),
            artifacts=[
                {
                    "path": (
                        "solver/phsolver.exe"
                        if platform == "windows"
                        else "solver/phsolver"
                    ),
                    "bytes": len(solver),
                    "sha256": _digest(solver),
                },
                {
                    "path": (
                        "sidecar/photonhub-serve-viz.exe"
                        if platform == "windows"
                        else "sidecar/photonhub-serve-viz"
                    ),
                    "bytes": len(sidecar),
                    "sha256": _digest(sidecar),
                },
            ],
            community_eula_version="2026-07-approved",
            privacy_version="2026-07-approved",
            third_party_notices_sha256=_digest(b"notices"),
            beta_api_url="https://identity.photonhub.dev",
        )
        promoted = publish.release.promote_release_manifest(
            candidate,
            source_git_sha=self.source_sha,
            beta_api_url="https://identity.photonhub.dev",
            physics_release_id="physics-release-2026-07",
            validation_profile_id=validation_profile_id,
            community_eula_version="2026-07-approved",
            privacy_version="2026-07-approved",
        )
        return candidate, promoted

    def _create_promotion_archive(
        self,
        *,
        target_name: str,
    ) -> tuple[Path, str, bytes]:
        target = publish.TARGETS[target_name]
        validation_profile = f"{target_name}-installed-validation-2026-07"
        candidate_release, promoted_release = self._release_manifest(
            platform=target["platform"],
            arch=target["arch"],
            validation_profile_id=validation_profile,
        )
        candidate_release_bytes = _canonical(candidate_release)
        promoted_release_bytes = _canonical(promoted_release)
        native_verification = _canonical({
            "ok": True,
            "target": target_name,
        })
        installer_content = f"signed-installer:{target_name}".encode("ascii")
        build_id = promoted_release["build"]["build_id"]
        installer_name = (
            f"PhotonHub-Workbench-v{self.version}-{target_name}-"
            f"{self.source_sha[:12]}-{build_id[:12]}"
            f"{target['installer_suffix']}"
        )
        candidate_sums = (
            f"{_digest(candidate_release_bytes)}  "
            "evidence/installed-release.json\n"
        ).encode("ascii")

        files: dict[str, bytes] = {
            installer_name: installer_content,
            "candidate-SHA256SUMS": candidate_sums,
            "release.json": promoted_release_bytes,
            "evidence/installed-release.json": candidate_release_bytes,
            "evidence/promotion-native-verification.json":
                native_verification,
        }
        clean_machine = {
            "schema": 1,
            "ok": True,
            "attestation_environment":
                publish.CLEAN_MACHINE_ENVIRONMENT,
            "attestor_identity": "qa-attestor-1",
            "repository": "Leapfield/PhotonHub",
            "candidate_run_id": 201,
            "candidate_run_attempt": 1,
            "candidate_artifact_name": f"candidate-{target_name}",
            "candidate_artifact_id": 101,
            "candidate_artifact_digest": f"sha256:{'b' * 64}",
            "platform": target["platform"],
            "arch": target["arch"],
            "installer_sha256": _digest(installer_content),
            "installed_release_sha256":
                _digest(candidate_release_bytes),
            "build_id": build_id,
            "source_git_sha": self.source_sha,
            "beta_api_url": "https://identity.photonhub.dev",
            "physics_release_id": "physics-release-2026-07",
            "validation_profile_id": validation_profile,
            "beta_e2e": _beta_e2e(
                source_sha=self.source_sha,
                artifact_digest=f"sha256:{'b' * 64}",
                installer_sha256=_digest(installer_content),
                platform=target["platform"],
                arch=target["arch"],
            ),
            "mi300x_gate_verification":
                _mi300x_gate_verification(self.source_sha),
            **{
                key: True
                for key in publish.CLEAN_MACHINE_TRUE_FIELDS
                if key != "ok"
            },
        }
        clean_machine_bytes = _canonical(clean_machine)
        files[
            "evidence/promotion-clean-machine-attestation.json"
        ] = clean_machine_bytes
        signing_authorization = _authorization_proof(
            state="completed",
            run_id=201,
            workflow_name="desktop-native-candidate",
            workflow_path=".github/workflows/desktop-native-candidate.yml",
            head_sha=self.source_sha,
            head_branch=f"desktop-candidate-v{self.version}",
            environment="desktop-native-signing",
            only_branch_policy="desktop-candidate-v*",
            jobs={
                "macos": "macOS ARM64 native candidate",
                "windows": "Windows x64 native candidate",
            },
        )
        clean_authorization = _authorization_proof(
            state="completed",
            run_id=401,
            workflow_name="desktop-clean-machine-acceptance",
            workflow_path=(
                ".github/workflows/desktop-clean-machine-acceptance.yml"
            ),
            head_sha=self.source_sha,
            head_branch=f"desktop-candidate-v{self.version}",
            environment=publish.CLEAN_MACHINE_ENVIRONMENT,
            only_branch_policy="desktop-candidate-v*",
            jobs={
                "macos": "Clean-machine acceptance macOS arm64",
                "windows": "Clean-machine acceptance Windows x64",
            },
        )
        promotion_authorization = _authorization_proof(
            state="completed_job",
            run_id=301,
            workflow_name="desktop-native-promote",
            workflow_path=".github/workflows/desktop-native-promote.yml",
            head_sha="f" * 40,
            head_branch="main",
            environment="desktop-native-promotion",
            only_branch_policy="main",
            jobs={"authorization": "Authorize exact protected signing run"},
        )
        files[
            "evidence/signing-environment-authorization.json"
        ] = signing_authorization
        files[
            "evidence/clean-machine-environment-authorization.json"
        ] = clean_authorization
        files[
            "evidence/promotion-environment-authorization.json"
        ] = promotion_authorization
        standalone_content = {
            "COMMUNITY-BETA-EULA.md": b"approved eula\n",
            "COMMUNITY-BETA-EULA.txt": b"approved eula\n",
            "PRIVACY.md": b"approved privacy\n",
            "desktop-beta-quickstart.md": b"quickstart\n",
            "desktop-beta-known-limitations.md": b"limits\n",
            "desktop-beta-support.md": b"support@example.org\n",
            "desktop-beta-upgrade-and-rollback.md": b"rollback\n",
        }
        for standalone_name, inner_path in (
            publish.STANDALONE_INNER_PATHS.items()
        ):
            files[inner_path] = standalone_content[standalone_name]

        promotion = {
            "schema": 1,
            "status": "promoted",
            "product": promoted_release["product"],
            "target": {
                "platform": target["platform"],
                "arch": target["arch"],
            },
            "source_git_sha": self.source_sha,
            "build_id": build_id,
            "physics": promoted_release["physics"],
            "installer": {
                "candidate_path": f"candidate{target['installer_suffix']}",
                "published_path": installer_name,
                "bytes": len(installer_content),
                "sha256": _digest(installer_content),
                "bytes_unchanged": True,
                "native_payload_rebuilt": False,
                "embedded_manifest_status": "candidate",
            },
            "distribution_manifest": {
                "path": "release.json",
                "status": "promoted",
                "sha256": _digest(promoted_release_bytes),
            },
            "candidate": {
                "repository": "Leapfield/PhotonHub",
                "artifact_name": f"candidate-{target_name}",
                "artifact_id": 101,
                "artifact_digest": f"sha256:{'b' * 64}",
                "run_id": 201,
                "run_attempt": 1,
                "sha256sums_sha256": _digest(candidate_sums),
                "installed_release_sha256":
                    _digest(candidate_release_bytes),
            },
            "promotion": {
                "run_id": 301,
                "run_attempt": 1,
                "workflow_sha": "f" * 40,
                "head_branch": "main",
                "signing_environment": "desktop-native-signing",
                "signer_identity": "APPROVEDSIGNER",
                "protected_signing_authorization_required": True,
                "native_verification_sha256":
                    _digest(native_verification),
                "signing_authorization_sha256":
                    _digest(signing_authorization),
                "clean_machine_authorization_sha256":
                    _digest(clean_authorization),
                "promotion_authorization_sha256":
                    _digest(promotion_authorization),
                "clean_machine_environment":
                    publish.CLEAN_MACHINE_ENVIRONMENT,
                "clean_machine_attestor_identity": "qa-attestor-1",
                "clean_machine_attestation_sha256":
                    _digest(clean_machine_bytes),
                "clean_machine_run_id": 401,
                "clean_machine_run_attempt": 1,
                "clean_machine_artifact_name":
                    f"desktop-clean-machine-{target_name}",
                "clean_machine_artifact_id": 501,
                "clean_machine_artifact_digest":
                    f"sha256:{'d' * 64}",
                "mi300x_hardware_gate": {
                    **clean_machine["beta_e2e"]["bindings"][
                        "mi300x_hardware_gate"
                    ],
                    "gate_verification_sha256":
                    clean_machine["beta_e2e"]["bindings"][
                        "mi300x_gate_verification_sha256"
                    ],
                },
                "staging_deployment_verification": {
                    **clean_machine["beta_e2e"]["bindings"][
                        "staging_deployment_verification"
                    ],
                    "receipt_sha256":
                    clean_machine["beta_e2e"]["bindings"][
                        "staging_deployment_verification_sha256"
                    ],
                    "projection_sha256":
                    clean_machine["beta_e2e"]["bindings"][
                        "staging_deployment_verification_projection_sha256"
                    ],
                },
            },
            "release_authorization": {
                "signing": json.loads(signing_authorization),
                "clean_machine": json.loads(clean_authorization),
                "promotion": json.loads(promotion_authorization),
            },
            "legal": {
                "community_eula_version": "2026-07-approved",
                "community_eula_sha256":
                    _digest(standalone_content["COMMUNITY-BETA-EULA.md"]),
                "privacy_version": "2026-07-approved",
                "privacy_notice_sha256":
                    _digest(standalone_content["PRIVACY.md"]),
            },
            "documentation": {
                "quickstart_sha256":
                    _digest(standalone_content["desktop-beta-quickstart.md"]),
                "known_limitations_sha256": _digest(
                    standalone_content["desktop-beta-known-limitations.md"]
                ),
                "support_document_sha256":
                    _digest(standalone_content["desktop-beta-support.md"]),
                "upgrade_rollback_sha256": _digest(
                    standalone_content[
                        "desktop-beta-upgrade-and-rollback.md"
                    ]
                ),
            },
            "artwork": {
                "source_path": f"assets/{target_name}.icon",
                "sha256": "c" * 64,
                "source_bytes": 4,
            },
            "services": promoted_release["services"],
            "native_verification": {
                "ok": True,
                "target": target_name,
            },
            "clean_machine_attestation": clean_machine,
        }
        files["promotion.json"] = _canonical(promotion)
        files["SHA256SUMS"] = "".join(
            f"{_digest(files[name])}  {name}\n"
            for name in sorted(files)
        ).encode("ascii")

        archive_name = (
            f"PhotonHub-Workbench-v{self.version}-{target_name}-"
            f"{self.source_sha[:12]}-promotion.zip"
        )
        archive_path = self.root / archive_name
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name in sorted(files):
                archive.writestr(name, files[name])
        return archive_path, installer_name, installer_content

    def _write_root_sums(self) -> None:
        paths = sorted(
            path for path in self.root.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (self.root / "SHA256SUMS").write_text(
            "".join(
                f"{publish.sha256_path(path)}  {path.name}\n"
                for path in paths
            ),
            encoding="ascii",
        )

    def _create_valid_release(self) -> None:
        standalone_content: dict[str, bytes] | None = None
        for target_name in publish.TARGETS:
            _archive, installer_name, installer_content = (
                self._create_promotion_archive(target_name=target_name)
            )
            (self.root / installer_name).write_bytes(installer_content)
            if standalone_content is None:
                standalone_content = {
                    "COMMUNITY-BETA-EULA.md": b"approved eula\n",
                    "COMMUNITY-BETA-EULA.txt": b"approved eula\n",
                    "PRIVACY.md": b"approved privacy\n",
                    "desktop-beta-quickstart.md": b"quickstart\n",
                    "desktop-beta-known-limitations.md": b"limits\n",
                    "desktop-beta-support.md": b"support@example.org\n",
                    "desktop-beta-upgrade-and-rollback.md": b"rollback\n",
                }
        assert standalone_content is not None
        base = f"PhotonHub-Workbench-v{self.version}"
        for name, content in standalone_content.items():
            (self.root / f"{base}-{name}").write_bytes(content)
        self._write_root_sums()

    def test_verifies_exact_release_and_uses_create_only_resumable_uploads(
        self,
    ) -> None:
        verified = publish.verify_release_root(
            self.root,
            version=self.version,
            source_git_sha=self.source_sha,
        )
        bucket = "photonhub-desktop-beta-downloads"
        plan = publish.build_upload_plan(
            verified,
            bucket_name=bucket,
        )
        self.assertEqual(
            verified.prefix,
            f"desktop/v{self.version}/{self.source_sha}/",
        )
        self.assertEqual(plan[-1].name, "SHA256SUMS")
        expected_gate_verification = _mi300x_gate_verification(
            self.source_sha
        )
        for archive in verified.archives.values():
            self.assertEqual(
                archive.promotion["clean_machine_attestation"][
                    "mi300x_gate_verification"
                ],
                expected_gate_verification,
            )
        for item in plan:
            self.assertIn(verified.prefix, item.uri)
            self.assertNotIn("/latest/", item.uri)

        access_token = "token-value-that-is-long-enough"
        calls: list[dict[str, object]] = []
        pending: dict[str, object] = {}

        def requester(method, url, headers, body):
            captured_body = body if isinstance(body, bytes) else body.read()
            calls.append({
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": captured_body,
            })
            if method == "POST":
                parsed = urlsplit(url)
                query = parse_qs(parsed.query)
                self.assertEqual(query["uploadType"], ["resumable"])
                self.assertEqual(query["ifGenerationMatch"], ["0"])
                object_name = query["name"][0]
                pending["name"] = object_name
                pending["bytes"] = int(headers["X-Upload-Content-Length"])
                location_query = urlencode({
                    "uploadType": "resumable",
                    "upload_id": f"session-{len(calls)}",
                    "name": object_name,
                    "ifGenerationMatch": "0",
                })
                return publish.HttpResponse(
                    200,
                    {
                        "Location": (
                            "https://storage.googleapis.com/upload/storage/v1/"
                            f"b/{bucket}/o?{location_query}"
                        )
                    },
                    b"",
                )
            self.assertEqual(method, "PUT")
            self.assertEqual(len(captured_body), pending["bytes"])
            return publish.HttpResponse(
                200,
                {},
                json.dumps({
                    "bucket": bucket,
                    "name": pending["name"],
                    "generation": "7001",
                    "size": str(pending["bytes"]),
                }).encode("utf-8"),
            )

        publish.execute_uploads(
            verified,
            plan,
            bucket_name=bucket,
            access_token=access_token,
            requester=requester,
        )
        self.assertEqual(len(calls), len(plan) * 2)
        posts = [call for call in calls if call["method"] == "POST"]
        puts = [call for call in calls if call["method"] == "PUT"]
        self.assertEqual(len(posts), len(plan))
        self.assertEqual(len(puts), len(plan))
        self.assertTrue(
            all(
                call["headers"]["Authorization"]
                == f"Bearer {access_token}"
                for call in posts
            )
        )
        self.assertTrue(
            all("Authorization" not in call["headers"] for call in puts)
        )
        self.assertFalse(
            any(
                term in call["url"]
                for call in calls
                for term in ("list", "delete", "get-iam-policy")
            )
        )
        report = publish.publication_report(
            verified,
            bucket_name=bucket,
            plan=plan,
            executed=False,
            authentication=publish.github_wif_identity_evidence(
                role="publisher",
                authenticated=False,
                provider=(
                    "projects/123456789/locations/global/"
                    "workloadIdentityPools/github/providers/photonhub"
                ),
                service_account=(
                    "desktop-publisher@photonhub-beta."
                    "iam.gserviceaccount.com"
                ),
                repository="Leapfield/PhotonHub",
                repository_id="987654321",
                ref="refs/heads/main",
                workflow_ref=(
                    "Leapfield/PhotonHub/.github/workflows/"
                    "desktop-beta-distribute.yml@refs/heads/main"
                ),
                environment=publish.PUBLISH_ENVIRONMENT,
            ),
        )
        self.assertTrue(report["dry_run"])
        self.assertFalse(report["signed_urls_created"])
        self.assertFalse(report["bucket_or_iam_changed"])
        self.assertNotIn(access_token, json.dumps(report))
        self.assertNotIn("session-", json.dumps(report))
        self.assertFalse(report["authentication"]["authenticated"])
        self.assertEqual(
            report["authentication"]["wif_environment_principal_set"],
            "principalSet://iam.googleapis.com/projects/123456789/"
            "locations/global/workloadIdentityPools/github/"
            "attribute.environment/desktop-beta-distribution-publish",
        )

    def test_resumable_upload_rejects_redirects_untrusted_hosts_and_ambiguity(
        self,
    ) -> None:
        verified = publish.verify_release_root(
            self.root,
            version=self.version,
            source_git_sha=self.source_sha,
        )
        bucket = "photonhub-desktop-beta-downloads"
        marker = publish.build_upload_plan(
            verified,
            bucket_name=bucket,
        )[-1:]
        token = "secret-token-value-that-must-never-leak"

        redirect_calls = 0

        def redirect_requester(method, url, headers, body):
            nonlocal redirect_calls
            redirect_calls += 1
            return publish.HttpResponse(
                302,
                {"Location": "https://storage.googleapis.com/redirect"},
                b"",
            )

        with self.assertRaisesRegex(
            publish.DistributionError,
            "initiation failed",
        ):
            publish.execute_uploads(
                verified,
                marker,
                bucket_name=bucket,
                access_token=token,
                requester=redirect_requester,
            )
        self.assertEqual(redirect_calls, 1)

        host_calls = 0

        def untrusted_host_requester(method, url, headers, body):
            nonlocal host_calls
            host_calls += 1
            return publish.HttpResponse(
                200,
                {
                    "Location": (
                        "https://attacker.invalid/upload/storage/v1/b/"
                        f"{bucket}/o?uploadType=resumable&upload_id=stolen"
                    )
                },
                b"",
            )

        with self.assertRaisesRegex(
            publish.DistributionError,
            "safe session URI",
        ):
            publish.execute_uploads(
                verified,
                marker,
                bucket_name=bucket,
                access_token=token,
                requester=untrusted_host_requester,
            )
        self.assertEqual(host_calls, 1)

        ambiguous_calls = 0

        def ambiguous_requester(method, url, headers, body):
            nonlocal ambiguous_calls
            ambiguous_calls += 1
            if method == "POST":
                object_name = marker[0].object_name
                query = urlencode(
                    {
                        "uploadType": "resumable",
                        "upload_id": "ambiguous-session",
                        "name": object_name,
                    }
                )
                return publish.HttpResponse(
                    200,
                    {
                        "Location": (
                            "https://storage.googleapis.com/upload/storage/v1/"
                            f"b/{bucket}/o?{query}"
                        )
                    },
                    b"",
                )
            raise OSError(f"network failed with {token}")

        with self.assertRaises(publish.DistributionError) as raised:
            publish.execute_uploads(
                verified,
                marker,
                bucket_name=bucket,
                access_token=token,
                requester=ambiguous_requester,
            )
        self.assertEqual(ambiguous_calls, 2)
        self.assertNotIn(token, str(raised.exception))
        self.assertIn("must not be retried", str(raised.exception))

    def test_adc_token_is_captured_without_entering_argv_or_errors(self) -> None:
        token = "captured-token-value-that-is-long-enough"
        calls: list[tuple[list[str], dict[str, object]]] = []

        def runner(argv, **kwargs):
            calls.append((list(argv), kwargs))
            return SimpleNamespace(stdout=f"{token}\n")

        self.assertEqual(
            publish.obtain_adc_access_token(runner=runner),
            token,
        )
        self.assertEqual(
            calls,
            [(
                [
                    "gcloud",
                    "--quiet",
                    "auth",
                    "application-default",
                    "print-access-token",
                ],
                {
                    "check": True,
                    "shell": False,
                    "capture_output": True,
                    "text": True,
                },
            )],
        )
        self.assertNotIn(token, json.dumps(calls))

        def failed_runner(argv, **kwargs):
            raise subprocess.CalledProcessError(
                1,
                argv,
                output=token,
                stderr=token,
            )

        with self.assertRaises(publish.DistributionError) as raised:
            publish.obtain_adc_access_token(runner=failed_runner)
        self.assertNotIn(token, str(raised.exception))

    def test_stdlib_https_request_streams_file_in_bounded_chunks(self) -> None:
        content = b"a" * (publish.UPLOAD_STREAM_CHUNK_BYTES + 7)
        path = Path(self.temporary.name) / "stream.bin"
        path.write_bytes(content)
        instances = []

        class Response:
            status = 200

            @staticmethod
            def getheaders():
                return [("Content-Type", "application/json")]

            @staticmethod
            def read(_limit):
                return b"{}"

        class Connection:
            def __init__(self, host, *, port, timeout):
                self.host = host
                self.port = port
                self.timeout = timeout
                self.request = None
                self.headers = []
                self.chunks = []
                self.closed = False
                instances.append(self)

            def putrequest(self, method, target, *, skip_accept_encoding):
                self.request = (method, target, skip_accept_encoding)

            def putheader(self, key, value):
                self.headers.append((key, value))

            def endheaders(self):
                return None

            def send(self, chunk):
                self.chunks.append(chunk)

            def getresponse(self):
                return Response()

            def close(self):
                self.closed = True

        with mock.patch.object(
            publish.http.client,
            "HTTPSConnection",
            Connection,
        ):
            with path.open("rb") as stream:
                response = publish._https_request(
                    "PUT",
                    "https://storage.googleapis.com/upload/session?upload_id=1",
                    {
                        "Content-Length": str(len(content)),
                        "Content-Type": "application/octet-stream",
                    },
                    stream,
                )
        self.assertEqual(response.status, 200)
        self.assertEqual(len(instances), 1)
        connection = instances[0]
        self.assertEqual(connection.host, "storage.googleapis.com")
        self.assertEqual(
            connection.request,
            ("PUT", "/upload/session?upload_id=1", True),
        )
        self.assertEqual(
            [len(chunk) for chunk in connection.chunks],
            [publish.UPLOAD_STREAM_CHUNK_BYTES, 7],
        )
        self.assertEqual(b"".join(connection.chunks), content)
        self.assertTrue(connection.closed)

    def test_rejects_unexpected_top_level_asset(self) -> None:
        (self.root / "latest.dmg").write_bytes(b"do not publish")
        with self.assertRaisesRegex(
            publish.DistributionError,
            "exact promoted asset set",
        ):
            publish.verify_release_root(
                self.root,
                version=self.version,
                source_git_sha=self.source_sha,
            )

    def test_rejects_symlink_even_when_named_as_an_asset(self) -> None:
        target = next(self.root.glob("*.dmg"))
        target.unlink()
        try:
            os.symlink(self.root / "SHA256SUMS", target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable")
        with self.assertRaisesRegex(
            publish.DistributionError,
            "symlink or non-file",
        ):
            publish.verify_release_root(
                self.root,
                version=self.version,
                source_git_sha=self.source_sha,
            )

    def test_rejects_checksum_tampering(self) -> None:
        quickstart = (
            self.root
            / f"PhotonHub-Workbench-v{self.version}-"
            "desktop-beta-quickstart.md"
        )
        quickstart.write_bytes(b"tampered\n")
        with self.assertRaisesRegex(
            publish.DistributionError,
            "checksum mismatch",
        ):
            publish.verify_release_root(
                self.root,
                version=self.version,
                source_git_sha=self.source_sha,
            )

    def test_rejects_a_self_consistent_failed_clean_machine_attestation(
        self,
    ) -> None:
        archive_path = next(
            self.root.glob("*-macos-arm64-*-promotion.zip")
        )
        with zipfile.ZipFile(archive_path) as archive:
            files = {
                info.filename: archive.read(info)
                for info in archive.infolist()
                if not info.is_dir()
            }
        clean_path = "evidence/promotion-clean-machine-attestation.json"
        clean = json.loads(files[clean_path])
        clean["registration_completed"] = False
        files[clean_path] = _canonical(clean)
        promotion = json.loads(files["promotion.json"])
        promotion["clean_machine_attestation"] = clean
        promotion["promotion"]["clean_machine_attestation_sha256"] = (
            _digest(files[clean_path])
        )
        files["promotion.json"] = _canonical(promotion)
        files["SHA256SUMS"] = "".join(
            f"{_digest(files[name])}  {name}\n"
            for name in sorted(files)
            if name != "SHA256SUMS"
        ).encode("ascii")
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name in sorted(files):
                archive.writestr(name, files[name])
        self._write_root_sums()

        with self.assertRaisesRegex(
            publish.DistributionError,
            "clean-machine",
        ):
            publish.verify_release_root(
                self.root,
                version=self.version,
                source_git_sha=self.source_sha,
            )

    def test_rejects_cross_platform_mi300x_gate_provenance_drift(
        self,
    ) -> None:
        archive_path = next(
            self.root.glob("*-windows-x64-*-promotion.zip")
        )
        with zipfile.ZipFile(archive_path) as archive:
            files = {
                info.filename: archive.read(info)
                for info in archive.infolist()
                if not info.is_dir()
            }
        clean_path = "evidence/promotion-clean-machine-attestation.json"
        clean = json.loads(files[clean_path])
        clean["beta_e2e"]["bindings"]["mi300x_hardware_gate"][
            "run_id"
        ] = "809"
        clean["mi300x_gate_verification"]["run_id"] = "809"
        clean["beta_e2e"]["bindings"][
            "mi300x_gate_verification_sha256"
        ] = _digest(_canonical(clean["mi300x_gate_verification"]))
        files[clean_path] = _canonical(clean)
        promotion = json.loads(files["promotion.json"])
        promotion["clean_machine_attestation"] = clean
        promotion["promotion"]["clean_machine_attestation_sha256"] = (
            _digest(files[clean_path])
        )
        promotion["promotion"]["mi300x_hardware_gate"] = {
            **clean["beta_e2e"]["bindings"]["mi300x_hardware_gate"],
            "gate_verification_sha256":
                clean["beta_e2e"]["bindings"][
                    "mi300x_gate_verification_sha256"
                ],
        }
        files["promotion.json"] = _canonical(promotion)
        files["SHA256SUMS"] = "".join(
            f"{_digest(files[name])}  {name}\n"
            for name in sorted(files)
            if name != "SHA256SUMS"
        ).encode("ascii")
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name in sorted(files):
                archive.writestr(name, files[name])
        self._write_root_sums()

        with self.assertRaisesRegex(
            publish.DistributionError,
            "one exact MI300X hardware-gate qualification",
        ):
            publish.verify_release_root(
                self.root,
                version=self.version,
                source_git_sha=self.source_sha,
            )

    def test_rejects_cross_platform_staging_deployment_drift(
        self,
    ) -> None:
        archive_path = next(
            self.root.glob("*-windows-x64-*-promotion.zip")
        )
        with zipfile.ZipFile(archive_path) as archive:
            files = {
                info.filename: archive.read(info)
                for info in archive.infolist()
                if not info.is_dir()
            }
        clean_path = "evidence/promotion-clean-machine-attestation.json"
        clean = json.loads(files[clean_path])
        bindings = clean["beta_e2e"]["bindings"]
        projection = bindings["staging_deployment_verification"]
        projection["plan_sha256"] = "9" * 64
        bindings["staging_deployment_verification_sha256"] = "a" * 64
        bindings[
            "staging_deployment_verification_projection_sha256"
        ] = _digest(_canonical(projection))
        files[clean_path] = _canonical(clean)
        promotion = json.loads(files["promotion.json"])
        promotion["clean_machine_attestation"] = clean
        promotion["promotion"]["clean_machine_attestation_sha256"] = (
            _digest(files[clean_path])
        )
        promotion["promotion"]["staging_deployment_verification"] = {
            **projection,
            "receipt_sha256":
                bindings["staging_deployment_verification_sha256"],
            "projection_sha256":
                bindings[
                    "staging_deployment_verification_projection_sha256"
                ],
        }
        files["promotion.json"] = _canonical(promotion)
        files["SHA256SUMS"] = "".join(
            f"{_digest(files[name])}  {name}\n"
            for name in sorted(files)
            if name != "SHA256SUMS"
        ).encode("ascii")
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name in sorted(files):
                archive.writestr(name, files[name])
        self._write_root_sums()

        with self.assertRaisesRegex(
            publish.DistributionError,
            "one exact acceptance-bootstrap staging deployment receipt",
        ):
            publish.verify_release_root(
                self.root,
                version=self.version,
                source_git_sha=self.source_sha,
            )

    def test_rejects_cross_platform_release_authorization_drift(
        self,
    ) -> None:
        archive_path = next(
            self.root.glob("*-windows-x64-*-promotion.zip")
        )
        with zipfile.ZipFile(archive_path) as archive:
            files = {
                info.filename: archive.read(info)
                for info in archive.infolist()
                if not info.is_dir()
            }
        proof_path = "evidence/signing-environment-authorization.json"
        proof = json.loads(files[proof_path])
        proof["approval_history"][0]["reviewer_user_id"] = 93
        files[proof_path] = _canonical(proof)
        promotion = json.loads(files["promotion.json"])
        promotion["release_authorization"]["signing"] = proof
        promotion["promotion"]["signing_authorization_sha256"] = _digest(
            files[proof_path]
        )
        files["promotion.json"] = _canonical(promotion)
        files["SHA256SUMS"] = "".join(
            f"{_digest(files[name])}  {name}\n"
            for name in sorted(files)
            if name != "SHA256SUMS"
        ).encode("ascii")
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name in sorted(files):
                archive.writestr(name, files[name])
        self._write_root_sums()

        with self.assertRaisesRegex(
            publish.DistributionError,
            "same canonical protected-environment authorization",
        ):
            publish.verify_release_root(
                self.root,
                version=self.version,
                source_git_sha=self.source_sha,
            )

    def test_rejects_gate_verification_receipt_hash_tampering(self) -> None:
        archive_path = next(
            self.root.glob("*-macos-arm64-*-promotion.zip")
        )
        with zipfile.ZipFile(archive_path) as archive:
            files = {
                info.filename: archive.read(info)
                for info in archive.infolist()
                if not info.is_dir()
            }
        clean_path = "evidence/promotion-clean-machine-attestation.json"
        clean = json.loads(files[clean_path])
        clean["mi300x_gate_verification"]["cleanup_confirmed"] = False
        files[clean_path] = _canonical(clean)
        promotion = json.loads(files["promotion.json"])
        promotion["clean_machine_attestation"] = clean
        promotion["promotion"]["clean_machine_attestation_sha256"] = (
            _digest(files[clean_path])
        )
        files["promotion.json"] = _canonical(promotion)
        files["SHA256SUMS"] = "".join(
            f"{_digest(files[name])}  {name}\n"
            for name in sorted(files)
            if name != "SHA256SUMS"
        ).encode("ascii")
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name in sorted(files):
                archive.writestr(name, files[name])
        self._write_root_sums()

        with self.assertRaisesRegex(
            publish.DistributionError,
            "gate verification SHA-256 does not match",
        ):
            publish.verify_release_root(
                self.root,
                version=self.version,
                source_git_sha=self.source_sha,
            )

    def test_rejects_zip_symlink_member(self) -> None:
        archive_path = Path(self.temporary.name) / "symlink.zip"
        info = zipfile.ZipInfo("evidence/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(info, "target")
        with self.assertRaisesRegex(
            publish.DistributionError,
            "not a regular file",
        ):
            publish._read_and_hash_archive(archive_path)

    def test_bucket_requires_exact_dedicated_confirmation(self) -> None:
        self.assertEqual(
            publish.validate_distribution_bucket(
                "gs://photonhub-desktop-beta-downloads",
                confirmation="photonhub-desktop-beta-downloads",
            ),
            "photonhub-desktop-beta-downloads",
        )
        for uri, confirmation in (
            (
                "gs://photonhub-beta-staging-results",
                "photonhub-beta-staging-results",
            ),
            (
                "gs://photonhub-desktop-beta-downloads/path",
                "photonhub-desktop-beta-downloads",
            ),
            (
                "gs://photonhub-desktop-beta-downloads",
                "another-bucket",
            ),
        ):
            with self.subTest(uri=uri):
                with self.assertRaises(publish.DistributionError):
                    publish.validate_distribution_bucket(
                        uri,
                        confirmation=confirmation,
                    )

    def test_bucket_metadata_requires_exact_v1_controls(self) -> None:
        provider = (
            "projects/123456789/locations/global/workloadIdentityPools/"
            "github/providers/photonhub"
        )
        metadata = {
            "name": "photonhub-desktop-beta-downloads",
            "projectNumber": "123456789",
            "metageneration": "17",
            "iamConfiguration": {
                "uniformBucketLevelAccess": {"enabled": True},
                "publicAccessPrevention": "enforced",
            },
            "versioning": {"enabled": True},
            "retentionPolicy": {
                "retentionPeriod": "7776000",
                "isLocked": False,
            },
        }
        calls: list[tuple[list[str], dict[str, object]]] = []

        def runner(argv, **kwargs):
            calls.append((list(argv), kwargs))
            return SimpleNamespace(stdout=json.dumps(metadata))

        contract = publish.verify_distribution_bucket_contract(
            bucket_name="photonhub-desktop-beta-downloads",
            expected_wif_provider=provider,
            runner=runner,
        )
        self.assertEqual(contract["bucket_metageneration"], 17)
        self.assertEqual(
            contract["retention_policy"],
            {"retention_period_seconds": 7_776_000, "is_locked": False},
        )
        self.assertRegex(contract["contract_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            calls[0][0],
            [
                "gcloud",
                "--quiet",
                "storage",
                "buckets",
                "describe",
                "gs://photonhub-desktop-beta-downloads",
                "--raw",
                "--format=json",
            ],
        )
        self.assertEqual(
            calls[0][1],
            {
                "check": True,
                "shell": False,
                "capture_output": True,
                "text": True,
            },
        )

        omitted_unlocked = json.loads(json.dumps(metadata))
        del omitted_unlocked["retentionPolicy"]["isLocked"]
        omitted_contract = publish.validate_distribution_bucket_metadata(
            omitted_unlocked,
            bucket_name="photonhub-desktop-beta-downloads",
            expected_project_number="123456789",
        )
        self.assertEqual(
            omitted_contract["retention_policy"],
            {"retention_period_seconds": 7_776_000, "is_locked": False},
        )

        mutations = {
            "wrong project": ("projectNumber", "222222222"),
            "versioning disabled": ("versioning", {"enabled": False}),
            "retention locked": (
                "retentionPolicy",
                {"retentionPeriod": "7776000", "isLocked": True},
            ),
            "retention lock is not boolean": (
                "retentionPolicy",
                {"retentionPeriod": "7776000", "isLocked": "false"},
            ),
            "retention wrong": (
                "retentionPolicy",
                {"retentionPeriod": "7775999", "isLocked": False},
            ),
            "PAP inherited": (
                "iamConfiguration",
                {
                    "uniformBucketLevelAccess": {"enabled": True},
                    "publicAccessPrevention": "inherited",
                },
            ),
            "UBLA disabled": (
                "iamConfiguration",
                {
                    "uniformBucketLevelAccess": {"enabled": False},
                    "publicAccessPrevention": "enforced",
                },
            ),
        }
        for label, (field, value) in mutations.items():
            with self.subTest(label=label):
                drifted = json.loads(json.dumps(metadata))
                drifted[field] = value
                with self.assertRaisesRegex(
                    publish.DistributionError,
                    "v1 contract",
                ):
                    publish.validate_distribution_bucket_metadata(
                        drifted,
                        bucket_name="photonhub-desktop-beta-downloads",
                        expected_project_number="123456789",
                    )

    def test_wif_identity_evidence_and_distinct_accounts_fail_closed(
        self,
    ) -> None:
        provider = (
            "projects/123456789/locations/global/workloadIdentityPools/"
            "github/providers/photonhub"
        )
        publisher_account = (
            "desktop-publisher@photonhub-beta.iam.gserviceaccount.com"
        )
        verifier_account = (
            "desktop-verifier@photonhub-beta.iam.gserviceaccount.com"
        )
        identity = publish.github_wif_identity_evidence(
            role="verifier",
            authenticated=True,
            provider=provider,
            service_account=verifier_account,
            repository="Leapfield/PhotonHub",
            repository_id="987654321",
            ref="refs/heads/main",
            workflow_ref=(
                "Leapfield/PhotonHub/.github/workflows/"
                "desktop-beta-distribute.yml@refs/heads/main"
            ),
            environment=publish.VERIFY_ENVIRONMENT,
        )
        self.assertEqual(identity["github_repository_id"], "987654321")
        self.assertEqual(
            identity["wif_repository_principal_set"],
            "principalSet://iam.googleapis.com/projects/123456789/"
            "locations/global/workloadIdentityPools/github/"
            "attribute.repository_id/987654321",
        )
        self.assertEqual(
            publish.require_distinct_distribution_service_accounts(
                publisher_account,
                verifier_account,
            ),
            (publisher_account, verifier_account),
        )
        for left, right in (
            (publisher_account, publisher_account),
            ("", verifier_account),
        ):
            with self.subTest(left=left, right=right):
                with self.assertRaises(publish.DistributionError):
                    publish.require_distinct_distribution_service_accounts(
                        left,
                        right,
                    )
        with self.assertRaisesRegex(
            publish.DistributionError,
            "workflow ref",
        ):
            publish.github_wif_identity_evidence(
                role="publisher",
                authenticated=True,
                provider=provider,
                service_account=publisher_account,
                repository="Leapfield/PhotonHub",
                repository_id="987654321",
                ref="refs/heads/main",
                workflow_ref=(
                    "Leapfield/PhotonHub/.github/workflows/"
                    "other.yml@refs/heads/main"
                ),
                environment=publish.PUBLISH_ENVIRONMENT,
            )

    def test_execution_accepts_external_account_and_rejects_static_key(
        self,
    ) -> None:
        provider = (
            "projects/1/locations/global/workloadIdentityPools/"
            "pool/providers/github"
        )
        service_account = (
            "desktop-publisher@photonhub-beta.iam.gserviceaccount.com"
        )
        wif = Path(self.temporary.name) / "wif.json"
        wif.write_text(json.dumps({
            "type": "external_account",
            "audience": f"//iam.googleapis.com/{provider}",
            "subject_token_type":
                "urn:ietf:params:oauth:token-type:jwt",
            "token_url": "https://sts.googleapis.com/v1/token",
            "service_account_impersonation_url": (
                "https://iamcredentials.googleapis.com/v1/projects/-/"
                f"serviceAccounts/{service_account}:generateAccessToken"
            ),
            "credential_source": {
                "url": (
                    "https://pipelines.actions.githubusercontent.com/"
                    "oidc/token?api-version=2.0"
                ),
                "headers": {"Authorization": "Bearer ephemeral"},
                "format": {
                    "type": "json",
                    "subject_token_field_name": "value",
                },
            },
        }))
        environment = {
            "GOOGLE_APPLICATION_CREDENTIALS": str(wif),
            "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": str(wif),
            "GOOGLE_GHA_CREDS_PATH": str(wif),
        }
        result = publish.require_external_account_wif(
            environment,
            expected_provider=provider,
            expected_service_account=service_account,
        )
        self.assertEqual(result, wif.resolve())

        service_account_key = (
            Path(self.temporary.name) / "service-account.json"
        )
        service_account_key.write_text(json.dumps({
            "type": "service_account",
            "private_key": "forbidden",
        }))
        with self.assertRaisesRegex(
            publish.DistributionError,
            "exact GitHub OIDC",
        ):
            publish.require_external_account_wif({
                "GOOGLE_APPLICATION_CREDENTIALS": str(service_account_key),
                "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE":
                    str(service_account_key),
                "GOOGLE_GHA_CREDS_PATH": str(service_account_key),
            }, expected_provider=provider, expected_service_account=service_account)
        with self.assertRaisesRegex(
            publish.DistributionError,
            "static credential environment",
        ):
            publish.require_external_account_wif({
                **environment,
                "GOOGLE_CLOUD_KEYFILE_JSON": "forbidden",
            }, expected_provider=provider, expected_service_account=service_account)
        with self.assertRaisesRegex(
            publish.DistributionError,
            "credential override",
        ):
            publish.require_external_account_wif({
                **environment,
                "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE":
                    str(service_account_key),
            }, expected_provider=provider, expected_service_account=service_account)

        credential = json.loads(wif.read_text())
        cases = {
            "wrong provider": {
                **credential,
                "audience": (
                    "//iam.googleapis.com/projects/2/locations/global/"
                    "workloadIdentityPools/pool/providers/github"
                ),
            },
            "wrong service account": {
                **credential,
                "service_account_impersonation_url": (
                    "https://iamcredentials.googleapis.com/v1/projects/-/"
                    "serviceAccounts/other-publisher@photonhub-beta."
                    "iam.gserviceaccount.com:generateAccessToken"
                ),
            },
            "wrong token type": {
                **credential,
                "subject_token_type":
                    "urn:ietf:params:aws:token-type:aws4_request",
            },
            "executable source": {
                **credential,
                "credential_source": {
                    "executable": {"command": "/tmp/forbidden"},
                },
            },
        }
        for label, value in cases.items():
            with self.subTest(label=label):
                wif.write_text(json.dumps(value))
                with self.assertRaisesRegex(
                    publish.DistributionError,
                    "exact GitHub OIDC",
                ):
                    publish.require_external_account_wif(
                        environment,
                        expected_provider=provider,
                        expected_service_account=service_account,
                    )
        wif.write_text(json.dumps(credential))
        missing_github_path = dict(environment)
        missing_github_path.pop("GOOGLE_GHA_CREDS_PATH")
        with self.assertRaisesRegex(
            publish.DistributionError,
            "GOOGLE_GHA_CREDS_PATH",
        ):
            publish.require_external_account_wif(
                missing_github_path,
                expected_provider=provider,
                expected_service_account=service_account,
            )


if __name__ == "__main__":
    unittest.main()
