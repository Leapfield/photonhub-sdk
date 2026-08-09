"""The result-bundle wire format: a gzip tar of the flat files in a run's output
directory (``manifest.json`` + monitor ``*.bin``). One definition of pack +
safe-extract, shared by the cloud executor (:mod:`photonhub.executor`, which
produces a bundle) and the cloud result cache (:mod:`photonhub.web.cache`, which
consumes one), so the two ends of the format can never drift apart.

Members are flat (arcnamed by filename, no parent dirs) so a bundle extracts
straight into a directory :class:`~photonhub.SimulationData` reads.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Optional, Union


DEFAULT_MAX_COMPRESSED_BYTES = 4 * 1024**3
DEFAULT_MAX_EXPANDED_BYTES = 16 * 1024**3
DEFAULT_MAX_MEMBERS = 100_000
COMPLETION_MARKER = ".photonhub-complete"
_COPY_CHUNK = 1024 * 1024
# With at most 100k physical records, even 50k maximum-length flat filenames
# need under 32 MiB of GNU/PAX extension blocks. This remains a finite parsing
# ceiling while allowing the configured member limit to be meaningful.
_MAX_EXTENSION_BYTES = 32 * 1024 * 1024


class BundleError(ValueError):
    """A malformed, unsafe, truncated, or resource-excessive result bundle."""


class _BoundedTarInfo(tarfile.TarInfo):
    """TarInfo that accounts for headers hidden by :mod:`tarfile`.

    PAX and GNU long-name records are consumed internally before iteration
    yields a file, so counting yielded members alone does not bound their
    decompressed allocation. Track those records on the owning TarFile and
    reject sparse encodings, which are outside PhotonHub's flat regular-file
    wire format and have additional expansion semantics.
    """

    def _proc_member(self, tar):
        if self.type == tarfile.GNUTYPE_SPARSE:
            raise BundleError("sparse files are not allowed in result bundles")
        extension_types = (
            tarfile.GNUTYPE_LONGNAME,
            tarfile.GNUTYPE_LONGLINK,
            tarfile.XHDTYPE,
            tarfile.XGLTYPE,
            tarfile.SOLARIS_XHDTYPE,
        )
        if self.type in extension_types:
            if self.size < 0:
                raise BundleError("invalid metadata size in result bundle")
            padded = self._block(self.size)
            total = getattr(tar, "_photonhub_extension_bytes", 0) + padded
            count = getattr(tar, "_photonhub_extension_count", 0) + 1
            byte_limit = min(
                _MAX_EXTENSION_BYTES,
                getattr(tar, "_photonhub_max_expanded_bytes",
                        self._photonhub_initial_max_expanded_bytes),
            )
            member_limit = getattr(
                tar, "_photonhub_max_members",
                self._photonhub_initial_max_members)
            if total > byte_limit:
                raise BundleError(
                    "result bundle metadata exceeds the expanded-byte limit")
            if count > member_limit:
                raise BundleError(
                    f"result bundle exceeds the {member_limit} member limit")
            tar._photonhub_extension_bytes = total
            tar._photonhub_extension_count = count
        return super()._proc_member(tar)

    def _proc_sparse(self, tar):
        raise BundleError("sparse files are not allowed in result bundles")

    def _proc_gnusparse_00(self, next, raw_headers):
        raise BundleError("sparse files are not allowed in result bundles")

    def _proc_gnusparse_01(self, next, pax_headers):
        raise BundleError("sparse files are not allowed in result bundles")

    def _proc_gnusparse_10(self, next, pax_headers, tar):
        raise BundleError("sparse files are not allowed in result bundles")


def _bounded_tarinfo(max_expanded_bytes: int, max_members: int):
    """Create a configured class because TarFile parses its first header in
    ``__init__``, before callers can attach per-archive limits to the instance.
    """
    return type(
        "_ConfiguredBoundedTarInfo",
        (_BoundedTarInfo,),
        {
            "_photonhub_initial_max_expanded_bytes": max_expanded_bytes,
            "_photonhub_initial_max_members": max_members,
        },
    )


def pack_bundle(src_dir: Union[str, Path],
                dest: Union[str, Path, None] = None) -> Optional[bytes]:
    """gzip-tar the files directly in ``src_dir`` (flat arcnames). With ``dest``
    (a file path) the archive streams to disk and ``None`` is returned — the
    cheap path for the large bundles a worker writes; with ``dest=None`` the
    archive is built in memory and returned as bytes (for an inline response)."""
    src_dir = Path(src_dir)
    # The consumer accepts regular flat files only. Do not let a symlink in an
    # output directory produce a bundle that our own extractor rejects (or
    # accidentally package a file outside the run directory).
    members = [
        p for p in sorted(src_dir.iterdir())
        if (p.is_file() and not p.is_symlink()
            and p.name != COMPLETION_MARKER)
    ]

    def _write(**open_kw) -> None:
        with tarfile.open(
                mode="w:gz", format=tarfile.PAX_FORMAT, **open_kw) as tar:
            for p in members:
                # Construct deterministic minimal metadata. tar.add() copies a
                # fractional filesystem mtime, causing PAX_FORMAT to emit a
                # hidden extension record for every ordinary file; enough
                # files would then consume the reader's metadata budget even
                # though this producer created the archive itself.
                info = tarfile.TarInfo(p.name)
                info.size = p.stat().st_size
                info.mode = 0o600
                info.mtime = 0
                with p.open("rb") as source:
                    tar.addfile(info, source)

    if dest is not None:
        _write(name=str(dest))
        return None
    buf = io.BytesIO()
    _write(fileobj=buf)
    return buf.getvalue()


def _prepare_destination(dest: Union[str, Path]) -> Path:
    dest = Path(dest)
    if dest.is_symlink():
        raise BundleError(f"unsafe result destination: {str(dest)!r}")
    dest.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or not dest.is_dir():
        raise BundleError(f"unsafe result destination: {str(dest)!r}")
    return dest


def _extract_open_tar(tar: tarfile.TarFile, dest: Path, *,
                      max_members: int, max_expanded_bytes: int) -> None:
    seen: set[str] = set()
    expanded = 0
    for member in tar:  # streaming iteration: never build an unbounded member list
        extension_count = getattr(tar, "_photonhub_extension_count", 0)
        extension_bytes = getattr(tar, "_photonhub_extension_bytes", 0)
        if len(seen) + extension_count >= max_members:
            raise BundleError(
                f"result bundle exceeds the {max_members} member limit")
        if (not member.isfile() or not member.name
                or member.name in (".", "..", COMPLETION_MARKER)
                or "/" in member.name or "\\" in member.name
                or "\x00" in member.name
                or member.name in seen):
            raise BundleError(
                f"unsafe path in result bundle: {member.name!r}")
        if (member.size < 0
                or member.size > max_expanded_bytes - expanded - extension_bytes):
            raise BundleError(
                "result bundle exceeds the expanded-byte limit "
                f"({max_expanded_bytes} bytes)")
        seen.add(member.name)
        expanded += member.size
        source = tar.extractfile(member)
        if source is None:
            raise BundleError(
                f"unreadable file in result bundle: {member.name!r}")
        try:
            # Exclusive creation also refuses a pre-existing symlink at the
            # destination name instead of following it outside ``dest``.
            with source, (dest / member.name).open("xb") as target:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(_COPY_CHUNK, remaining))
                    if not chunk:
                        raise BundleError(
                            f"truncated file in result bundle: {member.name!r}")
                    target.write(chunk)
                    remaining -= len(chunk)
        except FileExistsError as exc:
            raise BundleError(
                f"unsafe existing result path: {member.name!r}") from exc


def _positive_limit(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def extract_bundle_file(
    archive: Union[str, Path],
    dest: Union[str, Path],
    *,
    max_compressed_bytes: int = DEFAULT_MAX_COMPRESSED_BYTES,
    max_expanded_bytes: int = DEFAULT_MAX_EXPANDED_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> None:
    """Stream-extract a gzip tar from disk with finite resource ceilings."""
    max_compressed_bytes = _positive_limit(
        max_compressed_bytes, "max_compressed_bytes")
    max_expanded_bytes = _positive_limit(
        max_expanded_bytes, "max_expanded_bytes")
    max_members = _positive_limit(max_members, "max_members")
    archive = Path(archive)
    try:
        size = archive.stat().st_size
        if size > max_compressed_bytes:
            raise BundleError(
                "result bundle exceeds the compressed-byte limit "
                f"({max_compressed_bytes} bytes)")
        out = _prepare_destination(dest)
        with tarfile.open(
                name=archive, mode="r|gz",
                tarinfo=_bounded_tarinfo(
                    max_expanded_bytes, max_members)) as tar:
            tar._photonhub_max_expanded_bytes = max_expanded_bytes
            tar._photonhub_max_members = max_members
            _extract_open_tar(
                tar, out, max_members=max_members,
                max_expanded_bytes=max_expanded_bytes)
    except BundleError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise BundleError(f"invalid result bundle: {exc}") from exc


def extract_bundle(
    data: bytes,
    dest: Union[str, Path],
    *,
    max_compressed_bytes: int = DEFAULT_MAX_COMPRESSED_BYTES,
    max_expanded_bytes: int = DEFAULT_MAX_EXPANDED_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> None:
    """Extract an in-memory flat result bundle into ``dest``.

    The wire format contains regular files only. Reject directories, links,
    device nodes, and nested/path-like member names instead of relying on
    :meth:`tarfile.TarFile.extractall`, whose safety behavior differs across
    supported Python versions. The web cache uses :func:`extract_bundle_file`
    so large downloads remain streaming rather than occupying RAM twice.
    """
    max_compressed_bytes = _positive_limit(
        max_compressed_bytes, "max_compressed_bytes")
    max_expanded_bytes = _positive_limit(
        max_expanded_bytes, "max_expanded_bytes")
    max_members = _positive_limit(max_members, "max_members")
    if not isinstance(data, bytes):
        raise TypeError("result bundle data must be bytes")
    if len(data) > max_compressed_bytes:
        raise BundleError(
            "result bundle exceeds the compressed-byte limit "
            f"({max_compressed_bytes} bytes)")
    try:
        out = _prepare_destination(dest)
        with tarfile.open(
                fileobj=io.BytesIO(data), mode="r|gz",
                tarinfo=_bounded_tarinfo(
                    max_expanded_bytes, max_members)) as tar:
            tar._photonhub_max_expanded_bytes = max_expanded_bytes
            tar._photonhub_max_members = max_members
            _extract_open_tar(
                tar, out, max_members=max_members,
                max_expanded_bytes=max_expanded_bytes)
    except BundleError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise BundleError(f"invalid result bundle: {exc}") from exc
