#!/usr/bin/env python3
"""Fail-closed inspection for public simupod release archives."""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import csv
from email.parser import BytesParser
from email.policy import default
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tarfile
import tomllib
import zipfile
import zlib

from package_version import package_version


ROOT = Path(__file__).resolve().parents[2]
LICENSE_SOURCE = ROOT / "LICENSE"
SDIST_SOURCE = ROOT
PACKAGE_SOURCE = ROOT / "simupod"
WHEEL_DIST_INFO_FILES = frozenset({
    "METADATA",
    "WHEEL",
    "entry_points.txt",
    "licenses/LICENSE",
    "RECORD",
})
SDIST_IMAGE = "docs/img/sim_showcase.png"
PACKAGE_JSON = "simupod/schemas/simulation_v1.json"
ARCHIVE_TIMESTAMP = 1_580_601_600
GZIP_HEADER = b"\x1f\x8b\x08\x00" + struct.pack("<I", ARCHIVE_TIMESTAMP) + b"\x02\xff"
ENTRY_POINTS = (
    b"[console_scripts]\n"
    b"simupod-mcp = simupod.cli:mcp\n"
    b"simupod-serve-viz = simupod.cli:serve_viz\n"
)


def safe_path(raw_name: str, archive: str) -> PurePosixPath:
    if not raw_name or "\\" in raw_name or "\0" in raw_name:
        raise ValueError(f"{archive}: unsafe archive path: {raw_name!r}")
    trimmed = raw_name[:-1] if raw_name.endswith("/") else raw_name
    raw_parts = trimmed.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{archive}: unsafe archive path: {raw_name}")
    path = PurePosixPath(*raw_parts)
    if path.is_absolute():
        raise ValueError(f"{archive}: unsafe archive path: {raw_name}")
    if any(part.lower() == "engine" for part in path.parts):
        raise ValueError(f"{archive}: private solver directory present: {raw_name}")
    return path


def require_unique(names: list[str], archive: str) -> None:
    if len(names) != len(set(names)):
        raise ValueError(f"{archive}: duplicate archive members")


def _zip_container_checks(
    wheel: Path, bundle: zipfile.ZipFile, members: list[zipfile.ZipInfo]
) -> None:
    data = wheel.read_bytes()
    if bundle.comment:
        raise ValueError(f"{wheel.name}: ZIP archive comment is not allowed")
    if len(data) < 22 or data[-22:-18] != b"PK\x05\x06":
        raise ValueError(f"{wheel.name}: trailing data or missing ZIP end record")
    if members and members[0].header_offset != 0:
        raise ValueError(f"{wheel.name}: prefixed ZIP payload is not allowed")
    (
        signature,
        disk_number,
        directory_disk,
        disk_entries,
        total_entries,
        directory_size,
        directory_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", data, len(data) - 22)
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or directory_disk != 0
        or disk_entries != len(members)
        or total_entries != len(members)
        or comment_size != 0
        or directory_offset != bundle.start_dir
        or directory_offset + directory_size != len(data) - 22
    ):
        raise ValueError(f"{wheel.name}: malformed or multi-disk ZIP container")

    local_cursor = 0
    local_times: dict[str, tuple[int, int]] = {}
    for member in members:
        if member.extra or member.comment:
            raise ValueError(
                f"{wheel.name}: ZIP member extra data or comment: {member.filename}"
            )
        if member.header_offset != local_cursor:
            raise ValueError(f"{wheel.name}: prefixed, overlapping, or gapped ZIP data")
        try:
            encoded_name = member.filename.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{wheel.name}: non-ASCII archive member name") from exc
        try:
            (
                local_signature,
                extract_version,
                flags,
                compression,
                dos_time,
                dos_date,
                crc,
                compressed_size,
                file_size,
                name_size,
                extra_size,
            ) = struct.unpack_from("<4s5H3L2H", data, local_cursor)
        except struct.error as exc:
            raise ValueError(f"{wheel.name}: truncated local ZIP header") from exc
        name_start = local_cursor + 30
        data_start = name_start + name_size + extra_size
        data_end = data_start + compressed_size
        if (
            local_signature != b"PK\x03\x04"
            or extract_version != member.extract_version
            or flags != member.flag_bits
            or compression != member.compress_type
            or crc != member.CRC
            or compressed_size != member.compress_size
            or file_size != member.file_size
            or data[name_start : name_start + name_size] != encoded_name
            or extra_size != 0
            or data_end > directory_offset
        ):
            raise ValueError(f"{wheel.name}: malformed local ZIP member: {member.filename}")
        compressed = data[data_start:data_end]
        if member.compress_type == zipfile.ZIP_DEFLATED:
            decoder = zlib.decompressobj(-zlib.MAX_WBITS)
            decoded = decoder.decompress(compressed) + decoder.flush()
            if (
                not decoder.eof
                or decoder.unused_data
                or decoder.unconsumed_tail
                or decoded != bundle.read(member)
            ):
                raise ValueError(
                    f"{wheel.name}: malformed compressed member: {member.filename}"
                )
        elif member.compress_type == zipfile.ZIP_STORED:
            if compressed != bundle.read(member):
                raise ValueError(f"{wheel.name}: malformed stored member: {member.filename}")
        else:
            raise ValueError(f"{wheel.name}: unsupported ZIP compression method")
        local_times[member.filename] = (dos_time, dos_date)
        local_cursor = data_end
    if local_cursor != directory_offset:
        raise ValueError(f"{wheel.name}: unregistered data before ZIP directory")

    central_cursor = directory_offset
    for member in members:
        try:
            central = struct.unpack_from("<4s6H3L5H2L", data, central_cursor)
        except struct.error as exc:
            raise ValueError(f"{wheel.name}: truncated central ZIP directory") from exc
        (
            central_signature,
            create_version,
            extract_version,
            flags,
            compression,
            dos_time,
            dos_date,
            crc,
            compressed_size,
            file_size,
            name_size,
            extra_size,
            member_comment_size,
            member_disk,
            internal_attr,
            external_attr,
            local_offset,
        ) = central
        name_start = central_cursor + 46
        try:
            encoded_name = member.filename.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{wheel.name}: non-ASCII archive member name") from exc
        if (
            central_signature != b"PK\x01\x02"
            or create_version != (member.create_system << 8) | member.create_version
            or extract_version != member.extract_version
            or flags != member.flag_bits
            or compression != member.compress_type
            or (dos_time, dos_date) != local_times[member.filename]
            or crc != member.CRC
            or compressed_size != member.compress_size
            or file_size != member.file_size
            or data[name_start : name_start + name_size] != encoded_name
            or extra_size != 0
            or member_comment_size != 0
            or member_disk != member.volume
            or internal_attr != member.internal_attr
            or external_attr != member.external_attr
            or local_offset != member.header_offset
        ):
            raise ValueError(
                f"{wheel.name}: malformed central ZIP member: {member.filename}"
            )
        central_cursor = name_start + name_size
    if central_cursor != len(data) - 22:
        raise ValueError(f"{wheel.name}: unregistered data in ZIP directory")


def _gzip_tar_stream(sdist: Path) -> bytes:
    compressed = sdist.read_bytes()
    if len(compressed) < 18 or compressed[:10] != GZIP_HEADER:
        raise ValueError(f"{sdist.name}: unexpected gzip container metadata")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        tar_data = decoder.decompress(compressed) + decoder.flush()
    except zlib.error as exc:
        raise ValueError(f"{sdist.name}: invalid gzip stream") from exc
    if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError(f"{sdist.name}: trailing data or multiple gzip streams")
    return tar_data


def valid_python(data: bytes, name: str, archive: str) -> None:
    try:
        ast.parse(data.decode("utf-8"), filename=name)
    except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
        raise ValueError(f"{archive}: invalid Python source {name}: {exc}") from exc


def valid_json(data: bytes, name: str, archive: str) -> None:
    try:
        json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{archive}: invalid JSON {name}: {exc}") from exc


def text(data: bytes, name: str, archive: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{archive}: non-text payload in {name}") from exc


def allowed_package_file(name: str) -> bool:
    if not name.startswith("simupod/"):
        return False
    return name.endswith(".py") or name in {"simupod/py.typed", PACKAGE_JSON}


def validate_package_file(data: bytes, name: str, archive: str) -> None:
    if name.endswith(".py"):
        valid_python(data, name, archive)
    elif name == PACKAGE_JSON:
        valid_json(data, name, archive)
    else:
        text(data, name, archive)


def metadata_checks(metadata, archive: str, version: str) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    expected = {
        "Name": "simupod",
        "Version": version,
        "Summary": project["description"],
        "License-Expression": "Apache-2.0",
        "Requires-Python": ">=3.10",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{archive}: {key} is {metadata.get(key)!r}, expected {value!r}"
            )
    expected_urls = {
        f"{label}, {url}" for label, url in project.get("urls", {}).items()
    }
    if set(metadata.get_all("Project-URL", [])) != expected_urls:
        raise ValueError(f"{archive}: project URLs differ from pyproject.toml")
    expected_extras = set(project.get("optional-dependencies", {}))
    if set(metadata.get_all("Provides-Extra", [])) != expected_extras:
        raise ValueError(f"{archive}: provided extras differ from pyproject.toml")
    expected_requirements = set(project.get("dependencies", []))
    for extra, requirements in project.get("optional-dependencies", {}).items():
        expected_requirements.update(
            f"{requirement}; extra == '{extra}'" for requirement in requirements
        )
    if set(metadata.get_all("Requires-Dist", [])) != expected_requirements:
        raise ValueError(f"{archive}: dependencies differ from pyproject.toml")

    expected_authors = ", ".join(
        author["name"] for author in project.get("authors", []) if "name" in author
    )
    if metadata.get("Author") != expected_authors:
        raise ValueError(f"{archive}: author differs from pyproject.toml")
    if set(filter(None, (metadata.get("Keywords") or "").split(","))) != set(
        project.get("keywords", [])
    ):
        raise ValueError(f"{archive}: keywords differ from pyproject.toml")
    if set(metadata.get_all("Classifier", [])) != set(project.get("classifiers", [])):
        raise ValueError(f"{archive}: classifiers differ from pyproject.toml")
    if metadata.get("Description-Content-Type") != "text/markdown":
        raise ValueError(f"{archive}: description content type is not text/markdown")
    if metadata.get_all("License-File", []) != ["LICENSE"]:
        raise ValueError(f"{archive}: license-file metadata differs from the SDK")
    if metadata.get("Metadata-Version") != "2.4":
        raise ValueError(f"{archive}: unexpected core metadata version")

    expected_header_counts = Counter(
        {
            "metadata-version": 1,
            "name": 1,
            "version": 1,
            "summary": 1,
            "project-url": len(expected_urls),
            "author": 1,
            "license-expression": 1,
            "license-file": 1,
            "keywords": 1,
            "classifier": len(project.get("classifiers", [])),
            "requires-python": 1,
            "requires-dist": len(expected_requirements),
            "provides-extra": len(expected_extras),
            "description-content-type": 1,
        }
    )
    actual_header_counts = Counter(key.lower() for key in metadata.keys())
    if actual_header_counts != expected_header_counts:
        raise ValueError(f"{archive}: metadata header names or counts are unexpected")
    description = metadata.get_payload()
    if not isinstance(description, str) or description.encode("utf-8") != (
        ROOT / "README.md"
    ).read_bytes():
        raise ValueError(f"{archive}: package description differs from reviewed README")


def expected_package_files() -> dict[str, bytes]:
    expected: dict[str, bytes] = {}
    for source in PACKAGE_SOURCE.rglob("*"):
        if not source.is_file():
            continue
        name = "simupod/" + source.relative_to(PACKAGE_SOURCE).as_posix()
        if allowed_package_file(name):
            expected[name] = source.read_bytes()
    return expected


def expected_sdist_files() -> dict[str, bytes]:
    expected = expected_package_files()
    for name in (
        "LICENSE",
        "PHOTONHUB_SOURCE_COMMIT",
        "README.md",
        "pyproject.toml",
        SDIST_IMAGE,
    ):
        source = ROOT / name
        expected[name] = source.read_bytes()
    for source in (ROOT / "tests").rglob("*.py"):
        expected["tests/" + source.relative_to(ROOT / "tests").as_posix()] = (
            source.read_bytes()
        )
    return expected


def optional_sdist_files() -> dict[str, bytes]:
    ignore = ROOT / ".gitignore"
    return {".gitignore": ignore.read_bytes()} if ignore.is_file() else {}


def inspect_wheel(wheel: Path, version: str) -> None:
    expected_name = f"simupod-{version}-py3-none-any.whl"
    if wheel.name != expected_name:
        raise ValueError(f"{wheel.name}: expected pure-Python wheel {expected_name}")
    dist_info = f"simupod-{version}.dist-info"
    expected_package = expected_package_files()

    with zipfile.ZipFile(wheel) as bundle:
        members = bundle.infolist()
        _zip_container_checks(wheel, bundle, members)
        names = [member.filename.rstrip("/") for member in members]
        require_unique(names, wheel.name)
        files: set[str] = set()

        for member, normalized in zip(members, names):
            path = safe_path(normalized, wheel.name)
            if member.is_dir():
                raise ValueError(f"{wheel.name}: explicit directory member {member.filename}")
            mode = (member.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if stat.S_ISLNK(mode) or file_type not in (0, stat.S_IFREG):
                raise ValueError(f"{wheel.name}: non-regular member {member.filename}")
            if mode & 0o111:
                raise ValueError(f"{wheel.name}: executable member {member.filename}")

            name = str(path)
            files.add(name)
            if allowed_package_file(name):
                validate_package_file(bundle.read(member), name, wheel.name)
            elif name.startswith(dist_info + "/") and name.removeprefix(
                dist_info + "/"
            ) in WHEEL_DIST_INFO_FILES:
                pass
            else:
                raise ValueError(f"{wheel.name}: unexpected wheel content: {name}")

        required = {
            "simupod/__about__.py",
            "simupod/cli.py",
            "simupod/py.typed",
            PACKAGE_JSON,
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/entry_points.txt",
            f"{dist_info}/licenses/LICENSE",
            f"{dist_info}/RECORD",
        }
        if missing := required - files:
            raise ValueError(f"{wheel.name}: missing {sorted(missing)}")
        package_names = {name for name in files if name.startswith("simupod/")}
        if package_names != set(expected_package):
            raise ValueError(f"{wheel.name}: package files differ from reviewed source")
        for name, expected_data in expected_package.items():
            if bundle.read(name) != expected_data:
                raise ValueError(f"{wheel.name}: {name} differs from reviewed source")

        metadata = BytesParser(policy=default).parsebytes(
            bundle.read(f"{dist_info}/METADATA")
        )
        metadata_checks(metadata, wheel.name, version)
        if bundle.read(f"{dist_info}/licenses/LICENSE") != LICENSE_SOURCE.read_bytes():
            raise ValueError(f"{wheel.name}: packaged license differs from SDK LICENSE")

        wheel_metadata = BytesParser(policy=default).parsebytes(
            bundle.read(f"{dist_info}/WHEEL")
        )
        if Counter(key.lower() for key in wheel_metadata.keys()) != Counter(
            {
                "wheel-version": 1,
                "generator": 1,
                "root-is-purelib": 1,
                "tag": 1,
            }
        ):
            raise ValueError(f"{wheel.name}: unexpected WHEEL metadata")
        if (
            wheel_metadata.get("Wheel-Version") != "1.0"
            or not re.fullmatch(
                r"hatchling [0-9]+(?:\.[0-9]+){2}",
                wheel_metadata.get("Generator", ""),
            )
            or wheel_metadata.get("Root-Is-Purelib") != "true"
            or wheel_metadata.get("Tag") != "py3-none-any"
            or wheel_metadata.get_payload() != ""
        ):
            raise ValueError(f"{wheel.name}: WHEEL metadata differs from policy")

        if bundle.read(f"{dist_info}/entry_points.txt") != ENTRY_POINTS:
            raise ValueError(f"{wheel.name}: unexpected console entry points")

        record_name = f"{dist_info}/RECORD"
        record = text(bundle.read(record_name), "RECORD", wheel.name)
        record_rows = [row for row in csv.reader(io.StringIO(record)) if row]
        if any(len(row) != 3 for row in record_rows):
            raise ValueError(f"{wheel.name}: malformed RECORD row")
        record_by_name = {row[0]: row[1:] for row in record_rows}
        if len(record_by_name) != len(record_rows) or set(record_by_name) != files:
            raise ValueError(f"{wheel.name}: RECORD filenames do not match contents")
        for name, (record_hash, record_size) in record_by_name.items():
            if name == record_name:
                if record_hash or record_size:
                    raise ValueError(f"{wheel.name}: RECORD must not hash itself")
                continue
            data = bundle.read(name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            if record_hash != "sha256=" + digest.decode("ascii"):
                raise ValueError(f"{wheel.name}: bad RECORD hash for {name}")
            if record_size != str(len(data)):
                raise ValueError(f"{wheel.name}: bad RECORD size for {name}")


def inspect_sdist(sdist: Path, version: str) -> None:
    expected_name = f"simupod-{version}.tar.gz"
    if sdist.name != expected_name:
        raise ValueError(f"{sdist.name}: expected source archive {expected_name}")
    prefix = f"simupod-{version}"
    expected_source = expected_sdist_files()
    optional_source = optional_sdist_files()
    tar_data = _gzip_tar_stream(sdist)

    with tarfile.open(sdist, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name.rstrip("/") for member in members]
        require_unique(names, sdist.name)
        files: set[str] = set()
        archive_cursor = 0

        for member, normalized in zip(members, names):
            path = safe_path(normalized, sdist.name)
            if not path.parts or path.parts[0] != prefix:
                raise ValueError(f"{sdist.name}: content outside {prefix}/: {member.name}")
            if member.isdir():
                raise ValueError(f"{sdist.name}: explicit directory member {member.name}")
            if not member.isfile():
                raise ValueError(f"{sdist.name}: link or special member: {member.name}")
            if member.pax_headers:
                raise ValueError(f"{sdist.name}: unexpected PAX metadata: {member.name}")
            if member.mode & 0o111:
                raise ValueError(f"{sdist.name}: executable member: {member.name}")
            if (
                member.uid != 0
                or member.gid != 0
                or member.uname
                or member.gname
                or member.devmajor != 0
                or member.devminor != 0
                or member.mtime != ARCHIVE_TIMESTAMP
                or member.mode != 0o644
                or member.linkname
            ):
                raise ValueError(f"{sdist.name}: unexpected tar member metadata")
            if member.offset != archive_cursor or member.offset_data != archive_cursor + 512:
                raise ValueError(f"{sdist.name}: prefixed, overlapping, or gapped tar data")
            try:
                expected_header = member.tobuf(
                    format=tarfile.USTAR_FORMAT,
                    encoding="utf-8",
                    errors="strict",
                )
            except (UnicodeError, ValueError) as exc:
                raise ValueError(f"{sdist.name}: non-canonical tar header") from exc
            if tar_data[member.offset : member.offset_data] != expected_header:
                raise ValueError(f"{sdist.name}: non-canonical tar header")

            relative = str(PurePosixPath(*path.parts[1:]))
            files.add(relative)
            if (
                relative != "PKG-INFO"
                and relative not in expected_source
                and relative not in optional_source
            ):
                raise ValueError(f"{sdist.name}: unexpected sdist content: {relative}")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"{sdist.name}: cannot read {member.name}")
            data = extracted.read()
            data_end = member.offset_data + member.size
            padded_end = ((data_end + 511) // 512) * 512
            if (
                tar_data[member.offset_data:data_end] != data
                or any(tar_data[data_end:padded_end])
            ):
                raise ValueError(f"{sdist.name}: unregistered tar member data")
            archive_cursor = padded_end
            if relative in expected_source and data != expected_source[relative]:
                raise ValueError(f"{sdist.name}: {relative} differs from reviewed source")
            if relative in optional_source and data != optional_source[relative]:
                raise ValueError(f"{sdist.name}: {relative} differs from reviewed source")
            if allowed_package_file(relative):
                validate_package_file(data, relative, sdist.name)
            elif relative.startswith("tests/"):
                valid_python(data, relative, sdist.name)
            elif relative == SDIST_IMAGE:
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(f"{sdist.name}: invalid PNG {relative}")
            else:
                text(data, relative, sdist.name)

        allowed_names = set(expected_source) | {"PKG-INFO"} | (
            set(optional_source) & files
        )
        if files != allowed_names:
            raise ValueError(f"{sdist.name}: file set differs from reviewed source")
        tar_tail = tar_data[archive_cursor:]
        if len(tar_tail) < 1024 or len(tar_tail) % 512 or any(tar_tail):
            raise ValueError(f"{sdist.name}: trailing or malformed tar data")

        pkg_info = bundle.extractfile(bundle.getmember(f"{prefix}/PKG-INFO"))
        if pkg_info is None:
            raise ValueError(f"{sdist.name}: cannot read PKG-INFO")
        metadata_checks(
            BytesParser(policy=default).parsebytes(pkg_info.read()),
            sdist.name,
            version,
        )
        license_stream = bundle.extractfile(bundle.getmember(f"{prefix}/LICENSE"))
        if license_stream is None or license_stream.read() != LICENSE_SOURCE.read_bytes():
            raise ValueError(f"{sdist.name}: packaged license differs from SDK LICENSE")
        provenance = text(
            bundle.extractfile(
                bundle.getmember(f"{prefix}/PHOTONHUB_SOURCE_COMMIT")
            ).read(),
            "PHOTONHUB_SOURCE_COMMIT",
            sdist.name,
        ).strip()
        if len(provenance) != 40 or any(char not in "0123456789abcdef" for char in provenance):
            raise ValueError(f"{sdist.name}: invalid PhotonHub source provenance")


def distribution_artifacts(dist: Path, version: str) -> tuple[Path, Path]:
    wheels = sorted(dist.glob(f"simupod-{version}-*.whl"))
    sdists = sorted(dist.glob(f"simupod-{version}.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one simupod wheel and one source archive")
    entries = list(dist.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("distribution directory contains a link or non-file entry")
    if set(entries) != {wheels[0], sdists[0]}:
        raise ValueError("distribution directory must contain only the verified artifacts")
    return wheels[0], sdists[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    version = package_version()
    wheel, sdist = distribution_artifacts(args.dist, version)
    inspect_wheel(wheel, version)
    inspect_sdist(sdist, version)
    print(f"verified release artifacts for simupod {version}")
    for artifact in (wheel, sdist):
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        print(f"sha256:{digest}  {artifact.name}")


if __name__ == "__main__":
    main()
