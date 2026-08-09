from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FETCH_PATH = REPO_ROOT / "scripts" / "fetch_desktop_beta_release.py"
VERIFY_PATH = REPO_ROOT / "scripts" / "verify_desktop_beta_gcs.py"
BUCKET_CHECK_PATH = (
    REPO_ROOT / "scripts" / "check_desktop_beta_gcs_bucket.py"
)
EXTRACT_PATH = (
    REPO_ROOT / "scripts" / "extract_desktop_publication_artifact.py"
)
EVIDENCE_PATH = (
    REPO_ROOT / "scripts" / "prepare_desktop_distribution_evidence.py"
)
WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "desktop-beta-distribute.yml"
)


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fetch = _load_script(FETCH_PATH, "_test_fetch_desktop_beta_release")
verify = _load_script(VERIFY_PATH, "_test_verify_desktop_beta_gcs")
bucket_check = _load_script(
    BUCKET_CHECK_PATH,
    "_test_check_desktop_beta_gcs_bucket",
)
extract = _load_script(EXTRACT_PATH, "_test_extract_desktop_publication_artifact")
evidence = _load_script(EVIDENCE_PATH, "_test_prepare_desktop_distribution_evidence")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _active_authorization(
    *,
    repository: str,
    repository_id: int,
    run_id: int,
    head_sha: str,
    head_branch: str,
    environment: str,
    binding: str,
    job_name: str,
) -> bytes:
    return (
        json.dumps(
            {
                "schema":
                    "photonhub.github-protected-environment-approval.v1",
                "verified": True,
                "authorization_state": "active",
                "repository": repository,
                "repository_id": repository_id,
                "repository_node_id": f"R_{repository_id}",
                "run": {
                    "id": run_id,
                    "attempt": 1,
                    "workflow_name": "desktop-beta-distribute",
                    "workflow_path":
                        ".github/workflows/desktop-beta-distribute.yml",
                    "event": "workflow_dispatch",
                    "head_sha": head_sha,
                    "head_branch": head_branch,
                    "status": "in_progress",
                    "conclusion": None,
                    "actor_id": 7,
                    "triggering_actor_id": 8,
                },
                "protected_environment": {
                    "id": run_id + (10 if binding == "preflight" else 20),
                    "name": environment,
                    "created_at": "2026-07-20T09:00:00Z",
                    "updated_at": "2026-07-22T09:59:00Z",
                    "can_admins_bypass": False,
                    "prevent_self_review": True,
                    "required_reviewers": [{"type": "User", "id": 90}],
                    "protected_branches": False,
                    "custom_branch_policies": True,
                    "only_branch_policy": head_branch,
                },
                "approval_history": [{
                    "state": "approved",
                    "environment_id":
                        run_id + (10 if binding == "preflight" else 20),
                    "environment_name": environment,
                    "reviewer_user_id": 92,
                    "run_actor_id": 7,
                    "triggering_actor_id": 8,
                    "approved_environments": [{
                        "id":
                            run_id
                            + (10 if binding == "preflight" else 20),
                        "name": environment,
                    }],
                    "occurrences": 1,
                }],
                "jobs": [{
                    "binding": binding,
                    "id": run_id * 10 + (1 if binding == "preflight" else 2),
                    "name": job_name,
                    "run_id": run_id,
                    "run_attempt": 1,
                    "check_run_id":
                        run_id * 100 + (1 if binding == "preflight" else 2),
                    "check_run_node_id":
                        f"CR_{run_id}_{binding}",
                    "started_at": "2026-07-22T10:00:00Z",
                    "completed_at": None,
                    "status": "in_progress",
                    "conclusion": None,
                    "deployment": {
                        "id":
                            run_id * 1000
                            + (1 if binding == "preflight" else 2),
                        "node_id": f"D_{run_id}_{binding}",
                        "workflow_job_id":
                            run_id * 10
                            + (1 if binding == "preflight" else 2),
                        "check_run_id":
                            run_id * 100
                            + (1 if binding == "preflight" else 2),
                        "repository_id": repository_id,
                        "environment": environment,
                        "commit_sha": head_sha,
                        "ref": head_branch,
                        "deployment_state": "IN_PROGRESS",
                        "latest_status": "IN_PROGRESS",
                        "latest_status_node_id":
                            f"DS_{run_id}_{binding}",
                        "latest_status_created_at":
                            "2026-07-22T10:00:10Z",
                        "latest_status_updated_at":
                            "2026-07-22T10:00:20Z",
                        "authorization_status_id":
                            run_id * 10000
                            + (1 if binding == "preflight" else 2),
                        "authorization_status_node_id":
                            f"DS_{run_id}_{binding}",
                        "authorization_status": "in_progress",
                        "authorization_status_created_at":
                            "2026-07-22T10:00:10Z",
                        "authorization_status_updated_at":
                            "2026-07-22T10:00:20Z",
                    },
                }],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


class DesktopBetaReleaseFetchTests(unittest.TestCase):
    version = "1.2.3-beta.4"
    source_sha = "a" * 40
    repository = "Leapfield/PhotonHub"
    release_id = 3001

    def setUp(self) -> None:
        self.contents = self._asset_contents()
        self.release = {
            "id": self.release_id,
            "tag_name": f"desktop-v{self.version}",
            "name": f"PhotonHub Workbench v{self.version}",
            "immutable": True,
            "prerelease": True,
            "draft": False,
            "published_at": "2026-07-22T20:00:00Z",
            "assets": [
                {
                    "id": 4000 + index,
                    "name": name,
                    "size": len(content),
                    "state": "uploaded",
                    "digest": f"sha256:{_sha256(content)}",
                    "browser_download_url":
                        f"https://example.invalid/forbidden/{name}",
                }
                for index, (name, content) in enumerate(
                    sorted(self.contents.items()),
                    start=1,
                )
            ],
        }
        self.reference = {
            "ref": f"refs/tags/desktop-v{self.version}",
            "object": {"type": "commit", "sha": self.source_sha},
        }

    def _asset_contents(self) -> dict[str, bytes]:
        prefix = f"PhotonHub-Workbench-v{self.version}"
        names = {
            "SHA256SUMS",
            f"{prefix}-macos-arm64-{self.source_sha[:12]}-promotion.zip",
            f"{prefix}-windows-x64-{self.source_sha[:12]}-promotion.zip",
            (
                f"{prefix}-macos-arm64-{self.source_sha[:12]}-"
                f"{'b' * 12}.dmg"
            ),
            (
                f"{prefix}-windows-x64-{self.source_sha[:12]}-"
                f"{'c' * 12}.exe"
            ),
            *fetch.publisher._expected_standalone_names(self.version),
        }
        self.assertEqual(len(names), fetch.EXPECTED_ASSET_COUNT)
        return {
            name: f"exact:{name}\n".encode("ascii")
            for name in names
        }

    def _control(self):
        return fetch.validate_release_control(
            self.release,
            self.reference,
            repository=self.repository,
            release_id=self.release_id,
            version=self.version,
            source_git_sha=self.source_sha,
            publication_run_id=5001,
            publication_run_attempt=1,
        )

    def test_accepts_exact_immutable_release_and_filters_control_record(
        self,
    ) -> None:
        control = self._control()
        report = fetch.control_report(control)
        self.assertEqual(report["asset_count"], fetch.EXPECTED_ASSET_COUNT)
        self.assertEqual(report["tag"], f"desktop-v{self.version}")
        self.assertTrue(report["immutable"])
        self.assertNotIn("browser_download_url", json.dumps(report))
        self.assertNotIn("example.invalid", json.dumps(report))
        self.assertEqual(
            [item["name"] for item in report["assets"]],
            sorted(self.contents),
        )

    def test_fetches_metadata_and_downloads_every_asset_by_numeric_id(
        self,
    ) -> None:
        calls: list[list[str]] = []
        by_id = {
            item["id"]: self.contents[item["name"]]
            for item in self.release["assets"]
        }

        def runner(argv, **kwargs):
            calls.append(list(argv))
            endpoint = argv[-1]
            if endpoint == (
                f"repos/{self.repository}/releases/{self.release_id}"
            ):
                return SimpleNamespace(stdout=json.dumps(self.release))
            if endpoint == (
                f"repos/{self.repository}/git/ref/tags/"
                f"desktop-v{self.version}"
            ):
                return SimpleNamespace(stdout=json.dumps(self.reference))
            match = re.fullmatch(
                rf"repos/{re.escape(self.repository)}/releases/assets/([0-9]+)",
                endpoint,
            )
            if match is None:
                raise AssertionError(argv)
            kwargs["stdout"].write(by_id[int(match.group(1))])
            return SimpleNamespace(stdout=None)

        control = fetch.fetch_release_control(
            gh="gh",
            repository=self.repository,
            release_id=self.release_id,
            version=self.version,
            source_git_sha=self.source_sha,
            publication_run_id=5001,
            publication_run_attempt=1,
            runner=runner,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release-assets"
            root = fetch.download_release_assets(
                control,
                output_root=output,
                gh="gh",
                runner=runner,
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in root.iterdir()},
                self.contents,
            )
        download_calls = [
            argv for argv in calls
            if "Accept: application/octet-stream" in argv
        ]
        self.assertEqual(len(download_calls), fetch.EXPECTED_ASSET_COUNT)
        self.assertTrue(
            all(
                re.fullmatch(
                    rf"repos/{re.escape(self.repository)}/releases/assets/[0-9]+",
                    argv[-1],
                )
                for argv in download_calls
            )
        )
        self.assertNotIn("browser_download_url", json.dumps(calls))

    def test_rejects_mutable_release_tag_mismatch_and_missing_digest(
        self,
    ) -> None:
        cases = []
        mutable = json.loads(json.dumps(self.release))
        mutable["immutable"] = False
        cases.append((mutable, self.reference, "immutable"))
        wrong_ref = json.loads(json.dumps(self.reference))
        wrong_ref["object"]["sha"] = "d" * 40
        cases.append((self.release, wrong_ref, "lightweight ref"))
        no_digest = json.loads(json.dumps(self.release))
        no_digest["assets"][0]["digest"] = None
        cases.append((no_digest, self.reference, "server SHA-256"))
        extra = json.loads(json.dumps(self.release))
        extra["assets"].append({
            "id": 9999,
            "name": "latest.dmg",
            "size": 1,
            "state": "uploaded",
            "digest": f"sha256:{'e' * 64}",
        })
        cases.append((extra, self.reference, "exactly 12"))

        for release, reference, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    fetch.ReleaseFetchError,
                    message,
                ):
                    fetch.validate_release_control(
                        release,
                        reference,
                        repository=self.repository,
                        release_id=self.release_id,
                        version=self.version,
                        source_git_sha=self.source_sha,
                        publication_run_id=5001,
                        publication_run_attempt=1,
                    )

    def test_rejects_downloaded_byte_tampering_and_existing_output(
        self,
    ) -> None:
        control = self._control()

        def tampered_runner(argv, **kwargs):
            kwargs["stdout"].write(b"tampered")
            return SimpleNamespace(stdout=None)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release-assets"
            with self.assertRaisesRegex(
                fetch.ReleaseFetchError,
                "downloaded bytes",
            ):
                fetch.download_release_assets(
                    control,
                    output_root=output,
                    gh="gh",
                    runner=tampered_runner,
                )
            self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaisesRegex(
                fetch.ReleaseFetchError,
                "must not already exist",
            ):
                fetch.download_release_assets(
                    control,
                    output_root=output,
                    gh="gh",
                    runner=tampered_runner,
                )


class DesktopBetaGcsReadbackTests(unittest.TestCase):
    version = "1.2.3-beta.4"
    source_sha = "a" * 40
    repository = "Leapfield/PhotonHub"
    release_id = 3001
    bucket = "photonhub-desktop-beta-downloads"
    repository_id = "987654321"
    ref = "refs/heads/main"
    workflow_ref = (
        "Leapfield/PhotonHub/.github/workflows/"
        "desktop-beta-distribute.yml@refs/heads/main"
    )
    workflow_sha = "f" * 40
    provider = (
        "projects/123456789/locations/global/workloadIdentityPools/"
        "github/providers/photonhub"
    )
    publisher_service_account = (
        "desktop-publisher@photonhub-beta.iam.gserviceaccount.com"
    )
    verifier_service_account = (
        "desktop-verifier@photonhub-beta.iam.gserviceaccount.com"
    )

    def setUp(self) -> None:
        prefix = f"PhotonHub-Workbench-v{self.version}"
        names = {
            "SHA256SUMS",
            f"{prefix}-macos-arm64-{self.source_sha[:12]}-promotion.zip",
            f"{prefix}-windows-x64-{self.source_sha[:12]}-promotion.zip",
            (
                f"{prefix}-macos-arm64-{self.source_sha[:12]}-"
                f"{'b' * 12}.dmg"
            ),
            (
                f"{prefix}-windows-x64-{self.source_sha[:12]}-"
                f"{'c' * 12}.exe"
            ),
            *fetch.publisher._expected_standalone_names(self.version),
        }
        self.contents = {
            name: f"exact:{name}\n".encode("ascii")
            for name in names
        }
        assets = [
            {
                "asset_id": 4000 + index,
                "name": name,
                "bytes": len(self.contents[name]),
                "sha256": _sha256(self.contents[name]),
            }
            for index, name in enumerate(sorted(names), start=1)
        ]
        self.control = {
            "schema": 1,
            "ok": True,
            "control_record": "private-github-release",
            "repository": self.repository,
            "release_id": self.release_id,
            "tag": f"desktop-v{self.version}",
            "immutable": True,
            "prerelease": True,
            "draft": False,
            "published_at": "2026-07-22T20:00:00Z",
            "version": self.version,
            "source_git_sha": self.source_sha,
            "publication_run_id": 5001,
            "publication_run_attempt": 1,
            "asset_count": len(assets),
            "total_bytes": sum(item["bytes"] for item in assets),
            "assets": assets,
        }
        by_name = {item["name"]: item for item in assets}
        ordered_names = sorted(name for name in names if name != "SHA256SUMS")
        ordered_names.append("SHA256SUMS")
        immutable_prefix = (
            f"desktop/v{self.version}/{self.source_sha}/"
        )
        self.publisher_authentication = (
            fetch.publisher.github_wif_identity_evidence(
                role="publisher",
                authenticated=True,
                provider=self.provider,
                service_account=self.publisher_service_account,
                repository=self.repository,
                repository_id=self.repository_id,
                ref=self.ref,
                workflow_ref=self.workflow_ref,
                environment=fetch.publisher.PUBLISH_ENVIRONMENT,
            )
        )
        self.verifier_authentication = (
            fetch.publisher.github_wif_identity_evidence(
                role="verifier",
                authenticated=True,
                provider=self.provider,
                service_account=self.verifier_service_account,
                repository=self.repository,
                repository_id=self.repository_id,
                ref=self.ref,
                workflow_ref=self.workflow_ref,
                environment=fetch.publisher.VERIFY_ENVIRONMENT,
            )
        )
        self.bucket_contract = {
            "schema": 1,
            "contract": "photonhub.desktop-distribution-bucket.v1",
            "bucket": f"gs://{self.bucket}",
            "project_number": "123456789",
            "bucket_metageneration": 17,
            "uniform_bucket_level_access": True,
            "public_access_prevention": "enforced",
            "versioning_enabled": True,
            "retention_policy": {
                "retention_period_seconds": 7_776_000,
                "is_locked": False,
            },
            "read_only_check": True,
            "bucket_or_iam_changed": False,
            "contract_sha256": "3" * 64,
        }
        self.publication = {
            "schema": 1,
            "ok": True,
            "executed": True,
            "dry_run": False,
            "version": self.version,
            "source_git_sha": self.source_sha,
            "bucket": f"gs://{self.bucket}",
            "immutable_prefix": immutable_prefix,
            "create_only": True,
            "latest_alias_created": False,
            "signed_urls_created": False,
            "bucket_or_iam_changed": False,
            "control_record": "private-github-release",
            "authentication": self.publisher_authentication,
            "objects": [
                {
                    "name": name,
                    "bytes": by_name[name]["bytes"],
                    "sha256": by_name[name]["sha256"],
                    "uri": f"gs://{self.bucket}/{immutable_prefix}{name}",
                    "completion_marker": name == "SHA256SUMS",
                }
                for name in ordered_names
            ],
        }

    def test_binds_control_and_reads_every_exact_positive_generation(
        self,
    ) -> None:
        objects, identity = verify.validate_control_and_publication(
            self.control,
            self.publication,
            repository=self.repository,
            release_id=self.release_id,
            version=self.version,
            source_git_sha=self.source_sha,
            bucket_name=self.bucket,
            expected_publication_run_id=5001,
            expected_publication_run_attempt=1,
            expected_publisher_authentication=
                self.publisher_authentication,
        )
        self.assertEqual(identity, "5001:1")
        calls: list[list[str]] = []

        def runner(argv, **kwargs):
            calls.append(list(argv))
            if argv[3:5] == ["objects", "describe"]:
                uri = argv[5]
                name = uri.rsplit("/", 1)[-1]
                object_name = uri.removeprefix(f"gs://{self.bucket}/")
                return SimpleNamespace(stdout=json.dumps({
                    "bucket": self.bucket,
                    "name": object_name,
                    "generation": "7001",
                    "size": str(len(self.contents[name])),
                }))
            if argv[3] == "cp":
                generation_uri = argv[5]
                self.assertTrue(generation_uri.endswith("#7001"))
                name = generation_uri.rsplit("/", 1)[-1].split("#", 1)[0]
                Path(argv[6]).write_bytes(self.contents[name])
                return SimpleNamespace(stdout="")
            raise AssertionError(argv)

        remote = verify.verify_remote_objects(
            objects,
            bucket_name=self.bucket,
            runner=runner,
        )
        self.assertEqual(len(remote), verify.EXPECTED_ASSET_COUNT)
        self.assertTrue(all(item["generation"] == 7001 for item in remote))
        self.assertTrue(
            all(item["generation_bound_read"] is True for item in remote)
        )
        self.assertTrue(
            all("--do-not-decompress" in argv for argv in calls if "cp" in argv)
        )
        self.assertFalse(any("list" in argv for argv in calls))
        self.assertFalse(any("sign-url" in argv for argv in calls))

    def test_rejects_tampered_publication_and_nonpositive_generation(
        self,
    ) -> None:
        tampered = json.loads(json.dumps(self.publication))
        tampered["objects"][0]["sha256"] = "e" * 64
        with self.assertRaisesRegex(
            verify.VerificationError,
            "does not match GitHub control",
        ):
            verify.validate_control_and_publication(
                self.control,
                tampered,
                repository=self.repository,
                release_id=self.release_id,
                version=self.version,
                source_git_sha=self.source_sha,
                bucket_name=self.bucket,
                expected_publication_run_id=5001,
                expected_publication_run_attempt=1,
                expected_publisher_authentication=
                    self.publisher_authentication,
            )

        objects, _ = verify.validate_control_and_publication(
            self.control,
            self.publication,
            repository=self.repository,
            release_id=self.release_id,
            version=self.version,
            source_git_sha=self.source_sha,
            bucket_name=self.bucket,
            expected_publication_run_id=5001,
            expected_publication_run_attempt=1,
            expected_publisher_authentication=
                self.publisher_authentication,
        )

        def runner(argv, **kwargs):
            if argv[3:5] == ["objects", "describe"]:
                uri = argv[5]
                name = uri.rsplit("/", 1)[-1]
                return SimpleNamespace(stdout=json.dumps({
                    "bucket": self.bucket,
                    "name": uri.removeprefix(f"gs://{self.bucket}/"),
                    "generation": "0",
                    "size": str(len(self.contents[name])),
                }))
            raise AssertionError(argv)

        with self.assertRaisesRegex(
            verify.VerificationError,
            "positive integer",
        ):
            verify.verify_remote_objects(
                objects,
                bucket_name=self.bucket,
                runner=runner,
            )

    def test_verification_report_is_create_only_and_contains_no_credentials(
        self,
    ) -> None:
        self.assertIsNotNone(verify.ARTIFACT_DIGEST.fullmatch("f" * 64))
        self.assertIsNone(
            verify.ARTIFACT_DIGEST.fullmatch(f"sha256:{'f' * 64}")
        )
        objects, identity = verify.validate_control_and_publication(
            self.control,
            self.publication,
            repository=self.repository,
            release_id=self.release_id,
            version=self.version,
            source_git_sha=self.source_sha,
            bucket_name=self.bucket,
            expected_publication_run_id=5001,
            expected_publication_run_attempt=1,
            expected_publisher_authentication=
                self.publisher_authentication,
        )
        remote = [
            {
                "release_asset_id": item["asset_id"],
                "name": item["name"],
                "uri": item["uri"],
                "generation": 7001,
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "generation_bound_read": True,
            }
            for item in objects
        ]
        report = verify.verification_report(
            repository=self.repository,
            release_id=self.release_id,
            version=self.version,
            source_git_sha=self.source_sha,
            bucket_name=self.bucket,
            publication_identity=identity,
            verification_run_id=5001,
            verification_run_attempt=2,
            publication_artifact_id=9001,
            publication_artifact_digest=f"sha256:{'f' * 64}",
            release_control_sha256="1" * 64,
            publication_report_sha256="2" * 64,
            publisher_authentication=self.publisher_authentication,
            verifier_authentication=self.verifier_authentication,
            preflight_bucket_contract_sha256="3" * 64,
            preflight_bucket_metageneration=17,
            bucket_contract=self.bucket_contract,
            objects=remote,
        )
        self.assertTrue(report["completion_marker_verified"])
        serialized = json.dumps(report)
        for forbidden in (
            "private_key",
            "access_token",
            "delivery_grant",
            "X-Goog-Signature",
        ):
            self.assertNotIn(forbidden, serialized)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "verification.json"
            verify.write_json_create_only(output, report)
            self.assertEqual(json.loads(output.read_text()), report)
            with self.assertRaisesRegex(
                verify.VerificationError,
                "must not already exist",
            ):
                verify.write_json_create_only(output, report)

    def test_postflight_bucket_contract_must_match_preflight(self) -> None:
        verify.require_unchanged_bucket_contract(
            self.bucket_contract,
            preflight_sha256="3" * 64,
            preflight_metageneration=17,
        )
        for digest, metageneration in (
            ("4" * 64, 17),
            ("3" * 64, 18),
        ):
            with self.subTest(
                digest=digest,
                metageneration=metageneration,
            ):
                with self.assertRaisesRegex(
                    verify.VerificationError,
                    "changed after preflight",
                ):
                    verify.require_unchanged_bucket_contract(
                        self.bucket_contract,
                        preflight_sha256=digest,
                        preflight_metageneration=metageneration,
                    )

    def test_rejects_publication_run_identity_mismatch(self) -> None:
        with self.assertRaisesRegex(
            verify.VerificationError,
            "trusted current workflow",
        ):
            verify.validate_control_and_publication(
                self.control,
                self.publication,
                repository=self.repository,
                release_id=self.release_id,
                version=self.version,
                source_git_sha=self.source_sha,
                bucket_name=self.bucket,
                expected_publication_run_id=5002,
                expected_publication_run_attempt=1,
                expected_publisher_authentication=
                    self.publisher_authentication,
            )

    def test_artifact_digest_and_exact_safe_extraction_are_enforced(self) -> None:
        contents = {
            name: json.dumps({"name": name}).encode("utf-8")
            for name in extract.EXPECTED_FILES
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "artifact.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, content in sorted(contents.items()):
                    archive.writestr(name, content)
            digest = _sha256(archive_path.read_bytes())
            output = root / "out"
            extract.extract_exact_artifact(
                archive_path,
                expected_sha256=digest,
                output_root=output,
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.iterdir()},
                contents,
            )
            with self.assertRaisesRegex(
                extract.ArtifactError,
                "recorded digest",
            ):
                extract.extract_exact_artifact(
                    archive_path,
                    expected_sha256="f" * 64,
                    output_root=root / "wrong-digest",
                )

            unsafe_path = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe_path, "w") as archive:
                for name, content in sorted(contents.items()):
                    archive.writestr(name, content)
                archive.writestr("../credential.json", b"secret")
            with self.assertRaisesRegex(
                extract.ArtifactError,
                "wrong entry count",
            ):
                extract.extract_exact_artifact(
                    unsafe_path,
                    expected_sha256=_sha256(unsafe_path.read_bytes()),
                    output_root=root / "unsafe-out",
                )

    def test_failure_evidence_is_allowlisted_schema_checked_and_secret_free(
        self,
    ) -> None:
        dry_run = json.loads(json.dumps(self.publication))
        dry_run["executed"] = False
        dry_run["dry_run"] = True
        dry_run["authentication"]["authenticated"] = False
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "release-control.json").write_text(json.dumps(self.control))
            (raw / "publication-dry-run.json").write_text(json.dumps(dry_run))
            (raw / "publication-executed.json").write_text(json.dumps({
                "authorization": "Bearer should-not-survive",
            }))
            (raw / "gha-creds-secret.json").write_text("credential")
            output = root / "safe"
            evidence.prepare_evidence(
                source_root=raw,
                output_root=output,
                mode="failure",
                repository=self.repository,
                release_id=self.release_id,
                version=self.version,
                source_git_sha=self.source_sha,
                bucket=self.bucket,
                publication_run_id=5001,
                publication_run_attempt=1,
                expected_wif_provider=self.provider,
                expected_publisher_service_account=
                    self.publisher_service_account,
                repository_id=self.repository_id,
                ref=self.ref,
                workflow_ref=self.workflow_ref,
                workflow_sha=self.workflow_sha,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "release-control.json",
                    "publication-dry-run.json",
                    "failure-summary.json",
                },
            )
            serialized = b"".join(
                path.read_bytes() for path in sorted(output.iterdir())
            )
            self.assertNotIn(b"Bearer ", serialized)
            self.assertNotIn(b"gha-creds-", serialized)
            summary = json.loads((output / "failure-summary.json").read_text())
            self.assertFalse(summary["authorizes_download"])
            self.assertFalse(summary["credentials_included"])
            self.assertEqual(summary["invalid_allowed_file_count"], 1)
            self.assertEqual(summary["unexpected_entry_count"], 1)

    def test_success_evidence_requires_exact_matching_plans(self) -> None:
        dry_run = json.loads(json.dumps(self.publication))
        dry_run["executed"] = False
        dry_run["dry_run"] = True
        dry_run["authentication"]["authenticated"] = False
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "release-control.json").write_text(json.dumps(self.control))
            (raw / "publication-dry-run.json").write_text(json.dumps(dry_run))
            (raw / "publication-executed.json").write_text(
                json.dumps(self.publication)
            )
            (raw / "preflight-active-authorization.json").write_bytes(
                _active_authorization(
                    repository=self.repository,
                    repository_id=int(self.repository_id),
                    run_id=5001,
                    head_sha="f" * 40,
                    head_branch="main",
                    environment=
                        "desktop-beta-distribution-verification",
                    binding="preflight",
                    job_name=(
                        "Read-only verify distribution bucket controls "
                        "before publication"
                    ),
                )
            )
            (raw / "publication-active-authorization.json").write_bytes(
                _active_authorization(
                    repository=self.repository,
                    repository_id=int(self.repository_id),
                    run_id=5001,
                    head_sha="f" * 40,
                    head_branch="main",
                    environment="desktop-beta-distribution-publish",
                    binding="publication",
                    job_name=(
                        "Create-only publish exact immutable desktop release"
                    ),
                )
            )
            output = root / "safe"
            evidence.prepare_evidence(
                source_root=raw,
                output_root=output,
                mode="success",
                repository=self.repository,
                release_id=self.release_id,
                version=self.version,
                source_git_sha=self.source_sha,
                bucket=self.bucket,
                publication_run_id=5001,
                publication_run_attempt=1,
                expected_wif_provider=self.provider,
                expected_publisher_service_account=
                    self.publisher_service_account,
                repository_id=self.repository_id,
                ref=self.ref,
                workflow_ref=self.workflow_ref,
                workflow_sha=self.workflow_sha,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                evidence.SAFE_FILES,
            )

            mismatched = json.loads(json.dumps(self.publication))
            mismatched["objects"][0]["bytes"] += 1
            (raw / "publication-executed.json").write_text(
                json.dumps(mismatched)
            )
            with self.assertRaisesRegex(
                evidence.EvidenceError,
                "plans differ",
            ):
                evidence.prepare_evidence(
                    source_root=raw,
                    output_root=root / "mismatch",
                    mode="success",
                    repository=self.repository,
                    release_id=self.release_id,
                    version=self.version,
                    source_git_sha=self.source_sha,
                    bucket=self.bucket,
                    publication_run_id=5001,
                    publication_run_attempt=1,
                    expected_wif_provider=self.provider,
                    expected_publisher_service_account=
                        self.publisher_service_account,
                    repository_id=self.repository_id,
                    ref=self.ref,
                    workflow_ref=self.workflow_ref,
                    workflow_sha=self.workflow_sha,
                )


class DesktopBetaDistributionWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_workflow_has_only_manual_dispatch_and_strict_job_permissions(
        self,
    ) -> None:
        workflow = yaml.load(self.text, Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow["on"]), {"workflow_dispatch"})
        self.assertEqual(workflow["permissions"], {})
        jobs = workflow["jobs"]
        self.assertEqual(
            set(jobs),
            {
                "bucket-preflight",
                "publish",
                "verify",
                "authorization-audit",
            },
        )
        self.assertEqual(
            jobs["bucket-preflight"]["permissions"],
            {
                "actions": "read",
                "checks": "read",
                "contents": "read",
                "deployments": "read",
                "id-token": "write",
            },
        )
        self.assertEqual(
            jobs["publish"]["permissions"],
            {
                "actions": "read",
                "checks": "read",
                "contents": "read",
                "deployments": "read",
                "id-token": "write",
            },
        )
        self.assertEqual(
            jobs["verify"]["permissions"],
            {
                "actions": "read",
                "checks": "read",
                "contents": "read",
                "deployments": "read",
                "id-token": "write",
            },
        )
        self.assertEqual(
            jobs["authorization-audit"]["permissions"],
            {
                "actions": "read",
                "checks": "read",
                "contents": "read",
                "deployments": "read",
            },
        )
        self.assertNotIn("environment", jobs["authorization-audit"])
        self.assertEqual(jobs["publish"]["runs-on"], "ubuntu-24.04")
        self.assertEqual(jobs["verify"]["runs-on"], "ubuntu-24.04")
        self.assertEqual(
            jobs["bucket-preflight"]["environment"],
            "desktop-beta-distribution-verification",
        )
        self.assertEqual(
            jobs["publish"]["environment"],
            "desktop-beta-distribution-publish",
        )
        self.assertEqual(
            jobs["verify"]["environment"],
            "desktop-beta-distribution-verification",
        )
        self.assertEqual(jobs["publish"]["needs"], ["bucket-preflight"])
        self.assertEqual(
            jobs["verify"]["needs"],
            ["bucket-preflight", "publish"],
        )
        self.assertEqual(
            jobs["authorization-audit"]["needs"],
            ["bucket-preflight", "publish", "verify"],
        )

    def test_actions_are_full_sha_pinned_and_unsafe_capabilities_are_absent(
        self,
    ) -> None:
        action_refs = re.findall(
            r"^\s*uses:\s+[^@\s]+@([^\s#]+)",
            self.text,
            re.MULTILINE,
        )
        self.assertTrue(action_refs)
        self.assertTrue(
            all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)
        )
        for forbidden in (
            "secrets.",
            "contents: write",
            "credentials_json:",
            "storage cp",
            "buckets update",
            "buckets delete",
            "buckets add-iam-policy-binding",
            "buckets set-iam-policy",
            "buckets get-iam-policy",
            "add-iam-policy-binding",
            "sign-url",
            "browser_download_url",
            "gh release ",
        ):
            self.assertNotIn(forbidden, self.text)
        for required in (
            'test "$GITHUB_REF_TYPE" = branch',
            'test "$GITHUB_REF" = "refs/heads/${TRUSTED_DEFAULT_BRANCH}"',
            'test "$GITHUB_SHA" = "$TRUSTED_WORKFLOW_SHA"',
            "github.workflow_sha",
            "github.workflow_ref",
            "github.repository_id",
            "path: trusted-source",
            "fetch_desktop_beta_release.py",
            "--release-id \"$RELEASE_ID\"",
            "--execute",
            "--expected-wif-provider",
            "--expected-service-account",
            "--expected-publisher-service-account",
            "--repository-id",
            "--workflow-ref",
            "--workflow-sha",
            "--preflight-bucket-contract-sha256",
            "--preflight-bucket-metageneration",
            "check_desktop_beta_gcs_bucket.py",
            "prepare_desktop_distribution_evidence.py",
            "extract_desktop_publication_artifact.py",
            "artifact-ids: ${{ needs.publish.outputs.publication_artifact_id }}",
            "merge-multiple: true",
            "actions/artifacts/$PUBLICATION_ARTIFACT_ID/zip",
            "--publication-run-id \"$GITHUB_RUN_ID\"",
            "--publication-run-attempt \"$GITHUB_RUN_ATTEMPT\"",
            "desktop-beta-distribution-publish",
            "desktop-beta-distribution-verification",
            "PHOTONHUB_DESKTOP_DISTRIBUTION_BUCKET",
            "PHOTONHUB_DESKTOP_PUBLISHER_SERVICE_ACCOUNT",
            "PHOTONHUB_DESKTOP_VERIFIER_SERVICE_ACCOUNT",
            "PHOTONHUB_DESKTOP_PUBLICATION_WIF_PROVIDER",
            'test "$VERIFIER_SERVICE_ACCOUNT" != "$PUBLISHER_SERVICE_ACCOUNT"',
            "bucket_contract_sha256",
            "bucket_metageneration",
            '[[ "$PUBLICATION_ARTIFACT_DIGEST" =~ ^[0-9a-f]{64}$ ]]',
            "desktop-beta-publication-v${{ inputs.release_version }}-${{ inputs.source_git_sha }}",
            "desktop-beta-preflight-authorization-v${{ inputs.release_version }}-${{ inputs.source_git_sha }}",
            "desktop-beta-publication-authorization-v${{ inputs.release_version }}-${{ inputs.source_git_sha }}",
            "desktop-beta-readback-authorization-v${{ inputs.release_version }}-${{ inputs.source_git_sha }}",
            "desktop-beta-publication-incomplete-${{ github.run_id }}-${{ github.run_attempt }}",
            "desktop-beta-verification-v${{ inputs.release_version }}-${{ inputs.source_git_sha }}",
            "retention-days: 90",
            "python-version: '3.13.14'",
            "version: '576.0.0'",
            "persist-credentials: false",
        ):
            self.assertIn(required, self.text)
        self.assertNotIn("ref: ${{ inputs.source_git_sha }}", self.text)
        self.assertNotIn("path: evidence/publication\n", self.text)
        self.assertNotIn("PHOTONHUB_DESKTOP_GCS_", self.text)

    def test_each_wif_phase_retains_active_proof_before_authentication(
        self,
    ) -> None:
        jobs = yaml.load(self.text, Loader=yaml.BaseLoader)["jobs"]
        expected = {
            "bucket-preflight": (
                "Retain preflight authorization before WIF",
                "Authenticate verifier through Google Workload Identity Federation",
            ),
            "publish": (
                "Retain publication authorization before WIF",
                "Authenticate publisher through Google Workload Identity Federation",
            ),
            "verify": (
                "Retain read-back authorization before WIF",
                "Authenticate verifier through Google Workload Identity Federation",
            ),
        }
        for job_name, (retain_name, auth_name) in expected.items():
            steps = jobs[job_name]["steps"]
            names = [step["name"] for step in steps]
            retain_index = names.index(retain_name)
            auth_index = names.index(auth_name)
            self.assertLess(retain_index, auth_index)
            retain = steps[retain_index]
            self.assertTrue(retain["uses"].startswith("actions/upload-artifact@"))
            self.assertEqual(
                retain["with"]["path"],
                "${{ steps.environment-authorization.outputs.proof_path }}",
            )
    def test_every_workflow_cli_invocation_supplies_required_parser_options(
        self,
    ) -> None:
        workflow = yaml.load(self.text, Loader=yaml.BaseLoader)
        blocks = [
            step["run"]
            for job in workflow["jobs"].values()
            for step in job["steps"]
            if "run" in step
        ]

        def required_options(parser) -> set[str]:
            return {
                action.option_strings[0]
                for action in parser._actions
                if action.required and action.option_strings
            }

        scripts_and_parsers = (
            (
                "check_desktop_beta_gcs_bucket.py",
                bucket_check._parser(),
                1,
            ),
            (
                "publish_desktop_beta_gcs.py",
                fetch.publisher._parser(),
                2,
            ),
            (
                "prepare_desktop_distribution_evidence.py",
                evidence._parser(),
                2,
            ),
            (
                "verify_desktop_beta_gcs.py",
                verify._parser(),
                1,
            ),
        )
        for script, parser, count in scripts_and_parsers:
            matching = [block for block in blocks if script in block]
            self.assertEqual(len(matching), count, script)
            required = required_options(parser)
            for block in matching:
                supplied = set(
                    re.findall(r"(?m)^\s+(--[a-z0-9-]+)(?:\s|$)", block)
                )
                self.assertEqual(
                    required - supplied,
                    set(),
                    f"{script} invocation omits required parser options",
                )

    def test_yaml_bash_and_any_embedded_python_are_syntactically_valid(
        self,
    ) -> None:
        workflow = yaml.load(self.text, Loader=yaml.BaseLoader)
        bash_blocks: list[str] = []
        for job in workflow["jobs"].values():
            for step in job["steps"]:
                if "run" not in step:
                    continue
                self.assertEqual(step.get("shell"), "bash")
                bash_blocks.append(step["run"])
        self.assertGreaterEqual(len(bash_blocks), 6)
        for index, block in enumerate(bash_blocks):
            result = subprocess.run(
                ["bash", "-n"],
                input=block,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"bash block {index} failed syntax check: {result.stderr}",
            )
            for python_block in re.findall(
                r"<<'PY'\n(.*?)\n\s*PY(?:\n|$)",
                block,
                re.DOTALL,
            ):
                compile(python_block, f"workflow-block-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
