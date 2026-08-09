"""PhotonHub Workbench desktop release-manifest contract.

The desktop installer is a composition of independently built native artifacts:
Electron, a frozen Python sidecar, and ``phsolver``.  This module gives that
composition one fail-closed, machine-readable identity.  It intentionally uses
only the Python standard library so the build pipeline and the frozen sidecar
can both validate a manifest without adding another runtime dependency.

The manifest is evidence about *which* artifacts were assembled.  A
``status=promoted`` manifest additionally names the validation campaign that
qualified them; it is not itself proof that the campaign ran.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit


MANIFEST_VERSION = "1"
PRODUCT_NAME = "PhotonHub Workbench"
APP_ID = "com.photonhub.workbench"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CANDIDATE_LOOPBACK_API = re.compile(
    r"^http://(?:localhost|127\.0\.0\.1|\[::1\])(?::[1-9][0-9]{0,4})?$"
)
_DNS_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_DISTRIBUTABLE_BETA_API = re.compile(
    r"^https://"
    rf"(?:{_DNS_LABEL}\.)+"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?::[1-9][0-9]{0,4})?$"
)
_RESERVED_DISTRIBUTABLE_BETA_API = re.compile(
    rf"^https://(?:"
    rf"(?:{_DNS_LABEL}\.)*(?:localhost|local|test|invalid|example)"
    rf"|(?:{_DNS_LABEL}\.)*example\.(?:com|net|org)"
    rf")(?::[1-9][0-9]{{0,4}})?$"
)
_UNQUALIFIED = (
    "candidate",
    "draft",
    "internal",
    "pending",
    "unqualified",
    "unknown",
    "unset",
)


class ReleaseManifestError(ValueError):
    """A desktop release manifest or its bound artifacts are invalid."""


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseManifestError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseManifestError(f"{name} must be a non-empty string")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReleaseManifestError(
            f"{name} keys do not match the contract; missing={missing}, extra={extra}"
        )


def _relative_path(value: Any, name: str) -> str:
    text = _text(value, name).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ReleaseManifestError(f"{name} must be a normalized relative path")
    if re.match(r"^[A-Za-z]:", text):
        raise ReleaseManifestError(f"{name} must not contain a Windows drive")
    return str(path)


def _sha(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _SHA256.fullmatch(text):
        raise ReleaseManifestError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _qualified(value: str) -> bool:
    lowered = value.lower()
    return not any(marker in lowered for marker in _UNQUALIFIED)


def _beta_api_origin(value: Any, status: str) -> str:
    """Validate the fixed identity-service origin bound into an installer.

    Credentials are sent to this origin, so the manifest accepts a bare HTTPS
    origin only. Candidate builds may explicitly bind one of the three literal
    loopback origins over HTTP for local integration testing; promoted builds
    never may.
    """

    text = _text(value, "services.beta_api_url")
    if text != text.strip() or any(character.isspace() for character in text):
        raise ReleaseManifestError(
            "services.beta_api_url must not contain whitespace"
        )
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise ReleaseManifestError(
            "services.beta_api_url must be a valid HTTPS origin"
        ) from exc
    if parsed.scheme not in ("https", "http") or not text.startswith(
        f"{parsed.scheme}://"
    ):
        raise ReleaseManifestError(
            "services.beta_api_url must be a valid HTTPS origin"
        )
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
        or parsed.path
        or "?" in text
        or "#" in text
    ):
        raise ReleaseManifestError(
            "services.beta_api_url must be a bare origin without credentials, "
            "a path, query, or fragment"
        )
    if port is not None and not 1 <= port <= 65535:
        raise ReleaseManifestError(
            "services.beta_api_url has an invalid port"
        )
    if parsed.scheme == "http":
        if status != "candidate" or not _CANDIDATE_LOOPBACK_API.fullmatch(text):
            raise ReleaseManifestError(
                "services.beta_api_url must use HTTPS; HTTP is allowed only "
                "for an explicit loopback candidate origin"
            )
    if status == "promoted":
        _require_distributable_beta_api_origin(text)
    return text


def _require_distributable_beta_api_origin(text: str) -> None:
    """Reject ambiguous, local, and reserved origins before distribution.

    Chromium uses WHATWG URL parsing, which canonicalizes percent-encoded or
    legacy numeric host spellings such as ``127%2E0%2E0%2E1`` and
    ``2130706433`` to ``127.0.0.1``.  Python's ``urlsplit`` deliberately does
    not perform those transformations.  Signed/distributed installers therefore
    accept only one conservative canonical ASCII DNS spelling and never an IP
    literal.  This keeps the manifest string, build-time policy check, and the
    Electron runtime origin identical.
    """

    parsed = urlsplit(text)
    host = parsed.hostname or ""
    port = parsed.port
    if not _DISTRIBUTABLE_BETA_API.fullmatch(text):
        raise ReleaseManifestError(
            "services.beta_api_url for a signed or promoted release must be a "
            "canonical HTTPS DNS origin"
        )
    canonical = f"https://{host}"
    if port is not None and port != 443:
        canonical += f":{port}"
    if text != canonical:
        raise ReleaseManifestError(
            "services.beta_api_url for a signed or promoted release must use "
            "its canonical HTTPS origin spelling"
        )
    if _RESERVED_DISTRIBUTABLE_BETA_API.fullmatch(text):
        raise ReleaseManifestError(
            "services.beta_api_url for a signed or promoted release must not "
            "use a local, test, or reserved DNS name"
        )


def validate_distributable_beta_api_origin(value: Any) -> str:
    """Validate the identity origin allowed in a signed beta installer."""

    text = _beta_api_origin(value, "candidate")
    _require_distributable_beta_api_origin(text)
    return text


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical bytes used for hashes and manifest writes."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_from_epoch(epoch: int | str | None = None) -> str:
    """Return a reproducible RFC3339 timestamp when SOURCE_DATE_EPOCH is set."""

    raw = os.environ.get("SOURCE_DATE_EPOCH") if epoch is None else epoch
    if raw is None:
        moment = datetime.now(timezone.utc)
    else:
        try:
            moment = datetime.fromtimestamp(int(raw), tz=timezone.utc)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReleaseManifestError("SOURCE_DATE_EPOCH must be an integer") from exc
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def collect_artifacts(root: str | Path, *, exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Hash every regular file below ``root`` in deterministic path order.

    Symlinks are rejected: installers should carry the bytes that were audited,
    not links whose target can change between staging and packaging.
    """

    base = Path(root).resolve()
    if not base.is_dir():
        raise ReleaseManifestError(f"artifact root is not a directory: {base}")
    excluded = {_relative_path(item, "exclude path") for item in exclude}
    records: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(base).as_posix()
        if relative in excluded or path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise ReleaseManifestError(f"staged artifact is not a regular file: {relative}")
        records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not records:
        raise ReleaseManifestError("artifact root contains no files")
    return records


def _build_id(payload: Mapping[str, Any]) -> str:
    identity = {
        "product": payload["product"],
        "source": payload["source"],
        "contracts": payload["contracts"],
        "physics": payload["physics"],
        "services": payload["services"],
        "build": {
            "platform": payload["build"]["platform"],
            "arch": payload["build"]["arch"],
            "compiler": payload["build"]["compiler"],
            "openmp_runtime": payload["build"]["openmp_runtime"],
        },
        "solver_sha256": payload["solver"]["sha256"],
        "sidecar_sha256": payload["sidecar"]["sha256"],
        "legal": payload["legal"],
        # release.json is excluded from this inventory, so binding the complete
        # artifact list is non-circular. UI, Electron, examples, notices, and
        # every native library must all change the installed build identity.
        "artifacts": payload["artifacts"],
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def build_release_manifest(
    *,
    status: str,
    product_version: str,
    source_git_sha: str,
    input_schema_version: str,
    result_schema_version: str,
    physics_release_id: str,
    validation_profile_id: str,
    created_at: str,
    platform: str,
    arch: str,
    compiler: str,
    openmp_runtime: str,
    solver_relative_path: str,
    solver_info: Mapping[str, Any],
    solver_capabilities: Mapping[str, Any],
    solver_sha256: str,
    sidecar_relative_path: str,
    sidecar_sha256: str,
    artifacts: Iterable[Mapping[str, Any]],
    community_eula_version: str,
    privacy_version: str,
    third_party_notices_sha256: str,
    beta_api_url: str,
) -> dict[str, Any]:
    """Build and validate one manifest from already inspected artifacts."""

    capabilities = json.loads(canonical_json_bytes(dict(solver_capabilities)))
    artifact_records = [dict(item) for item in artifacts]
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "status": status,
        "product": {
            "name": PRODUCT_NAME,
            "version": product_version,
            "app_id": APP_ID,
        },
        "source": {"git_sha": source_git_sha},
        "contracts": {
            "input_schema_version": input_schema_version,
            "result_schema_version": result_schema_version,
        },
        "physics": {
            "physics_release_id": physics_release_id,
            "validation_profile_id": validation_profile_id,
        },
        "services": {
            "beta_api_url": _beta_api_origin(beta_api_url, status),
        },
        "build": {
            "build_id": "0" * 64,
            "created_at": created_at,
            "platform": platform,
            "arch": arch,
            "compiler": compiler,
            "openmp_runtime": openmp_runtime,
        },
        "solver": {
            "relative_path": solver_relative_path,
            "name": solver_info.get("name"),
            "version": solver_info.get("version"),
            "git_sha": solver_info.get("git_sha"),
            "schema_major": solver_info.get("schema_major"),
            "gpu": solver_info.get("gpu"),
            "workbench_authorization_required": solver_info.get(
                "workbench_authorization_required"
            ),
            "sha256": solver_sha256,
            "capabilities": capabilities,
            "capabilities_sha256": hashlib.sha256(canonical_json_bytes(capabilities)).hexdigest(),
        },
        "sidecar": {
            "relative_path": sidecar_relative_path,
            "version": product_version,
            "sha256": sidecar_sha256,
        },
        "legal": {
            "community_eula_version": community_eula_version,
            "privacy_version": privacy_version,
            "third_party_notices_sha256": third_party_notices_sha256,
        },
        "artifacts": artifact_records,
    }
    manifest["build"]["build_id"] = _build_id(manifest)
    validate_release_manifest(manifest)
    return manifest


def validate_release_manifest(manifest: Any, *, require_promoted: bool = False) -> None:
    """Validate structure, identities, hashes, and promotion invariants."""

    root = _object(manifest, "release manifest")
    _exact_keys(root, {
        "manifest_version", "status", "product", "source", "contracts",
        "physics", "services", "build", "solver", "sidecar", "legal",
        "artifacts",
    }, "release manifest")
    if root["manifest_version"] != MANIFEST_VERSION:
        raise ReleaseManifestError(
            f"unsupported desktop manifest_version {root['manifest_version']!r}"
        )
    status = root["status"]
    if status not in ("candidate", "promoted"):
        raise ReleaseManifestError("status must be 'candidate' or 'promoted'")
    if require_promoted and status != "promoted":
        raise ReleaseManifestError("a promoted desktop release is required")

    product = _object(root["product"], "product")
    _exact_keys(product, {"name", "version", "app_id"}, "product")
    if product["name"] != PRODUCT_NAME or product["app_id"] != APP_ID:
        raise ReleaseManifestError("product name/app_id do not match PhotonHub Workbench")
    _text(product["version"], "product.version")

    source = _object(root["source"], "source")
    _exact_keys(source, {"git_sha"}, "source")
    source_sha = _text(source["git_sha"], "source.git_sha")

    contracts = _object(root["contracts"], "contracts")
    _exact_keys(contracts, {"input_schema_version", "result_schema_version"}, "contracts")
    _text(contracts["input_schema_version"], "contracts.input_schema_version")
    _text(contracts["result_schema_version"], "contracts.result_schema_version")

    physics = _object(root["physics"], "physics")
    _exact_keys(physics, {"physics_release_id", "validation_profile_id"}, "physics")
    physics_id = _text(physics["physics_release_id"], "physics.physics_release_id")
    validation_id = _text(physics["validation_profile_id"], "physics.validation_profile_id")

    services = _object(root["services"], "services")
    _exact_keys(services, {"beta_api_url"}, "services")
    _beta_api_origin(services["beta_api_url"], status)

    build = _object(root["build"], "build")
    _exact_keys(build, {
        "build_id", "created_at", "platform", "arch", "compiler", "openmp_runtime",
    }, "build")
    _sha(build["build_id"], "build.build_id")
    _text(build["created_at"], "build.created_at")
    if build["platform"] not in ("macos", "windows", "linux"):
        raise ReleaseManifestError("build.platform is unsupported")
    if build["arch"] not in ("arm64", "x64"):
        raise ReleaseManifestError("build.arch is unsupported")
    compiler = _text(build["compiler"], "build.compiler")
    openmp = _text(build["openmp_runtime"], "build.openmp_runtime")

    solver = _object(root["solver"], "solver")
    _exact_keys(solver, {
        "relative_path", "name", "version", "git_sha", "schema_major", "gpu",
        "workbench_authorization_required", "sha256", "capabilities",
        "capabilities_sha256",
    }, "solver")
    solver_path = _relative_path(solver["relative_path"], "solver.relative_path")
    if solver["name"] != "phsolver":
        raise ReleaseManifestError("solver.name must be 'phsolver'")
    _text(solver["version"], "solver.version")
    solver_git_sha = _text(solver["git_sha"], "solver.git_sha")
    if not isinstance(solver["schema_major"], int) or solver["schema_major"] < 1:
        raise ReleaseManifestError("solver.schema_major must be a positive integer")
    if solver["gpu"] is not False:
        raise ReleaseManifestError("the desktop beta must bundle a CPU-only solver")
    if solver["workbench_authorization_required"] is not True:
        raise ReleaseManifestError(
            "the desktop beta solver must require Workbench launch authorization"
        )
    _sha(solver["sha256"], "solver.sha256")
    capabilities = _object(solver["capabilities"], "solver.capabilities")
    expected_capabilities_hash = hashlib.sha256(canonical_json_bytes(capabilities)).hexdigest()
    if solver["capabilities_sha256"] != expected_capabilities_hash:
        raise ReleaseManifestError("solver capabilities hash does not match its payload")

    sidecar = _object(root["sidecar"], "sidecar")
    _exact_keys(sidecar, {"relative_path", "version", "sha256"}, "sidecar")
    sidecar_path = _relative_path(sidecar["relative_path"], "sidecar.relative_path")
    _text(sidecar["version"], "sidecar.version")
    _sha(sidecar["sha256"], "sidecar.sha256")

    legal = _object(root["legal"], "legal")
    _exact_keys(legal, {
        "community_eula_version", "privacy_version", "third_party_notices_sha256",
    }, "legal")
    eula_version = _text(legal["community_eula_version"], "legal.community_eula_version")
    privacy_version = _text(legal["privacy_version"], "legal.privacy_version")
    _sha(legal["third_party_notices_sha256"], "legal.third_party_notices_sha256")

    artifact_list = root["artifacts"]
    if not isinstance(artifact_list, list) or not artifact_list:
        raise ReleaseManifestError("artifacts must be a non-empty array")
    by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(artifact_list):
        artifact = _object(raw, f"artifacts[{index}]")
        _exact_keys(artifact, {"path", "bytes", "sha256"}, f"artifacts[{index}]")
        path = _relative_path(artifact["path"], f"artifacts[{index}].path")
        if path in by_path:
            raise ReleaseManifestError(f"duplicate artifact path: {path}")
        if not isinstance(artifact["bytes"], int) or artifact["bytes"] < 0:
            raise ReleaseManifestError(f"artifacts[{index}].bytes must be non-negative")
        _sha(artifact["sha256"], f"artifacts[{index}].sha256")
        by_path[path] = artifact
    for path, expected in (
        (solver_path, solver["sha256"]), (sidecar_path, sidecar["sha256"]),
    ):
        if path not in by_path or by_path[path]["sha256"] != expected:
            raise ReleaseManifestError(f"{path} is not bound by the artifact inventory")

    expected_build_id = _build_id(root)
    if build["build_id"] != expected_build_id:
        raise ReleaseManifestError("build.build_id does not match release identities")

    if status == "promoted":
        promoted_values = {
            "source.git_sha": source_sha,
            "solver.git_sha": solver_git_sha,
            "physics.physics_release_id": physics_id,
            "physics.validation_profile_id": validation_id,
            "build.compiler": compiler,
            "build.openmp_runtime": openmp,
            "legal.community_eula_version": eula_version,
            "legal.privacy_version": privacy_version,
        }
        for name, value in promoted_values.items():
            if not _qualified(value):
                raise ReleaseManifestError(f"promoted release has unqualified {name}: {value!r}")
        if (
            not _FULL_GIT_SHA.fullmatch(source_sha)
            or not _FULL_GIT_SHA.fullmatch(solver_git_sha)
        ):
            raise ReleaseManifestError(
                "promoted release requires exact lowercase 40-character source "
                "and solver Git SHAs"
            )


def verify_release_artifacts(manifest: Any, artifact_root: str | Path) -> None:
    """Re-hash every manifest artifact below ``artifact_root``."""

    validate_release_manifest(manifest)
    base = Path(artifact_root).resolve()
    for artifact in manifest["artifacts"]:
        relative = _relative_path(artifact["path"], "artifact.path")
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ReleaseManifestError(f"artifact escaped release root: {relative}") from exc
        if path.is_symlink() or not path.is_file():
            raise ReleaseManifestError(f"release artifact is missing or not regular: {relative}")
        if path.stat().st_size != artifact["bytes"]:
            raise ReleaseManifestError(f"release artifact size changed: {relative}")
        if sha256_file(path) != artifact["sha256"]:
            raise ReleaseManifestError(f"release artifact hash changed: {relative}")


def rebind_release_manifest(
    manifest: Any,
    artifact_root: str | Path,
    *,
    manifest_relative_path: str = "release.json",
) -> dict[str, Any]:
    """Bind an existing release identity to the bytes in a packaged resource tree.

    Native signing mutates PE and Mach-O files after shared staging.  Installer
    hooks call this function only after those mutations, preserving the audited
    product/source/physics/legal identities while refreshing the artifact
    inventory, executable hashes, capabilities hash, and derived build ID.

    This function does not promote a candidate.  Promoted manifests are
    immutable: signing and packaging must finish while the release is still a
    candidate, then promotion must reference those exact post-sign bytes.
    """

    validate_release_manifest(manifest)
    if manifest["status"] != "candidate":
        raise ReleaseManifestError(
            "only candidate release manifests may be rebound; promoted manifests are immutable"
        )
    rebound = json.loads(canonical_json_bytes(manifest))
    base = Path(artifact_root).resolve()
    if not base.is_dir():
        raise ReleaseManifestError(f"artifact root is not a directory: {base}")

    manifest_path = _relative_path(manifest_relative_path, "manifest relative path")
    solver_path = base / _relative_path(
        rebound["solver"]["relative_path"], "solver.relative_path"
    )
    sidecar_path = base / _relative_path(
        rebound["sidecar"]["relative_path"], "sidecar.relative_path"
    )
    notices_path = base / "licenses" / "THIRD-PARTY-NOTICES.md"
    for name, path in (
        ("solver", solver_path),
        ("sidecar", sidecar_path),
        ("third-party notices", notices_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise ReleaseManifestError(f"{name} is missing or not a regular file: {path}")

    rebound["solver"]["sha256"] = sha256_file(solver_path)
    rebound["solver"]["capabilities_sha256"] = hashlib.sha256(
        canonical_json_bytes(rebound["solver"]["capabilities"])
    ).hexdigest()
    rebound["sidecar"]["sha256"] = sha256_file(sidecar_path)
    rebound["legal"]["third_party_notices_sha256"] = sha256_file(notices_path)
    rebound["artifacts"] = collect_artifacts(base, exclude=(manifest_path,))
    rebound["build"]["build_id"] = _build_id(rebound)
    validate_release_manifest(rebound)
    verify_release_artifacts(rebound, base)
    return rebound


def promote_release_manifest(
    manifest: Any,
    *,
    source_git_sha: str,
    beta_api_url: str,
    physics_release_id: str,
    validation_profile_id: str,
    community_eula_version: str,
    privacy_version: str,
) -> dict[str, Any]:
    """Qualify an immutable candidate identity without changing payload bytes.

    Native candidates are rebound before their final platform signatures are
    applied.  Promotion therefore may change only the qualification status.
    Every distribution-critical identity is supplied again by the release
    owner and must already match the signed candidate exactly.  This prevents a
    detached promotion record from claiming a different source revision,
    identity origin, or legal approval than the installer actually embeds.
    """

    validate_release_manifest(manifest)
    if manifest["status"] != "candidate":
        raise ReleaseManifestError(
            "only an immutable candidate release manifest may be promoted"
        )
    if not _FULL_GIT_SHA.fullmatch(source_git_sha):
        raise ReleaseManifestError(
            "promotion requires an exact lowercase 40-character source Git SHA"
        )
    expected = {
        "source.git_sha": (
            manifest["source"]["git_sha"],
            source_git_sha,
        ),
        "solver.git_sha": (
            manifest["solver"]["git_sha"],
            source_git_sha,
        ),
        "services.beta_api_url": (
            manifest["services"]["beta_api_url"],
            validate_distributable_beta_api_origin(beta_api_url),
        ),
        "physics.physics_release_id": (
            manifest["physics"]["physics_release_id"],
            _text(physics_release_id, "physics_release_id"),
        ),
        "physics.validation_profile_id": (
            manifest["physics"]["validation_profile_id"],
            _text(validation_profile_id, "validation_profile_id"),
        ),
        "legal.community_eula_version": (
            manifest["legal"]["community_eula_version"],
            _text(community_eula_version, "community_eula_version"),
        ),
        "legal.privacy_version": (
            manifest["legal"]["privacy_version"],
            _text(privacy_version, "privacy_version"),
        ),
    }
    mismatches = [
        f"{name}: candidate={actual!r}, approved={approved!r}"
        for name, (actual, approved) in expected.items()
        if actual != approved
    ]
    if mismatches:
        raise ReleaseManifestError(
            "promotion inputs do not match the immutable candidate identity; "
            + "; ".join(mismatches)
        )

    promoted = json.loads(canonical_json_bytes(manifest))
    promoted["status"] = "promoted"
    validate_release_manifest(promoted, require_promoted=True)
    if promoted["build"]["build_id"] != manifest["build"]["build_id"]:
        raise ReleaseManifestError(
            "promotion unexpectedly changed the native payload build identity"
        )
    return promoted


def load_release_manifest(
    path: str | Path, *, artifact_root: str | Path | None = None,
    require_promoted: bool = False,
) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(f"could not read desktop release manifest: {exc}") from exc
    validate_release_manifest(manifest, require_promoted=require_promoted)
    if artifact_root is not None:
        verify_release_artifacts(manifest, artifact_root)
    return manifest


def write_release_manifest(path: str | Path, manifest: Any) -> None:
    """Atomically write a validated canonical release manifest."""

    validate_release_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(canonical_json_bytes(manifest) + b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def desktop_release_schema() -> dict[str, Any]:
    """Return the JSON Schema distributed for external manifest consumers."""

    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    text = {"type": "string", "minLength": 1}
    artifact_path = {"type": "string", "minLength": 1}
    https_origin = {"type": "string", "pattern": r"^https://[^/?#\s@]+$"}
    distributable_origin = {
        "type": "string",
        "allOf": [
            {"pattern": _DISTRIBUTABLE_BETA_API.pattern},
            {"not": {"pattern": _RESERVED_DISTRIBUTABLE_BETA_API.pattern}},
            {"not": {"pattern": r":443$"}},
        ],
    }
    candidate_loopback_origin = {
        "type": "string",
        "pattern": _CANDIDATE_LOOPBACK_API.pattern,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://photonhub.dev/schemas/desktop_release_v1.json",
        "title": "PhotonHub Workbench desktop release manifest v1",
        "type": "object",
        "additionalProperties": False,
        "allOf": [{
            "if": {
                "properties": {"status": {"const": "promoted"}},
                "required": ["status"],
            },
            "then": {
                "properties": {
                    "services": {
                        "properties": {"beta_api_url": distributable_origin},
                    },
                },
            },
        }],
        "required": [
            "manifest_version", "status", "product", "source", "contracts",
            "physics", "services", "build", "solver", "sidecar", "legal",
            "artifacts",
        ],
        "properties": {
            "manifest_version": {"const": MANIFEST_VERSION},
            "status": {"enum": ["candidate", "promoted"]},
            "product": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "version", "app_id"],
                "properties": {
                    "name": {"const": PRODUCT_NAME}, "version": text,
                    "app_id": {"const": APP_ID},
                },
            },
            "source": {
                "type": "object", "additionalProperties": False,
                "required": ["git_sha"], "properties": {"git_sha": text},
            },
            "contracts": {
                "type": "object", "additionalProperties": False,
                "required": ["input_schema_version", "result_schema_version"],
                "properties": {"input_schema_version": text, "result_schema_version": text},
            },
            "physics": {
                "type": "object", "additionalProperties": False,
                "required": ["physics_release_id", "validation_profile_id"],
                "properties": {"physics_release_id": text, "validation_profile_id": text},
            },
            "services": {
                "type": "object", "additionalProperties": False,
                "required": ["beta_api_url"],
                "properties": {
                    "beta_api_url": {
                        "anyOf": [https_origin, candidate_loopback_origin],
                    },
                },
            },
            "build": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "build_id", "created_at", "platform", "arch", "compiler", "openmp_runtime",
                ],
                "properties": {
                    "build_id": digest, "created_at": text,
                    "platform": {"enum": ["macos", "windows", "linux"]},
                    "arch": {"enum": ["arm64", "x64"]},
                    "compiler": text, "openmp_runtime": text,
                },
            },
            "solver": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "relative_path", "name", "version", "git_sha", "schema_major", "gpu",
                    "workbench_authorization_required", "sha256", "capabilities",
                    "capabilities_sha256",
                ],
                "properties": {
                    "relative_path": artifact_path, "name": {"const": "phsolver"},
                    "version": text, "git_sha": text,
                    "schema_major": {"type": "integer", "minimum": 1},
                    "gpu": {"const": False},
                    "workbench_authorization_required": {"const": True},
                    "sha256": digest,
                    "capabilities": {"type": "object"}, "capabilities_sha256": digest,
                },
            },
            "sidecar": {
                "type": "object", "additionalProperties": False,
                "required": ["relative_path", "version", "sha256"],
                "properties": {"relative_path": artifact_path, "version": text, "sha256": digest},
            },
            "legal": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "community_eula_version", "privacy_version", "third_party_notices_sha256",
                ],
                "properties": {
                    "community_eula_version": text, "privacy_version": text,
                    "third_party_notices_sha256": digest,
                },
            },
            "artifacts": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["path", "bytes", "sha256"],
                    "properties": {
                        "path": artifact_path,
                        "bytes": {"type": "integer", "minimum": 0},
                        "sha256": digest,
                    },
                },
            },
        },
    }
