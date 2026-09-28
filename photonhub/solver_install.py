"""Install, record and locate a managed ``phsolver``.

The engine is a separate, invitation-gated download: `pip install photonhub`
gives you the client, and one command gives you the solver it drives. This
module is what that command is made of, and it deliberately uses the standard
library only, because the machines that need it most are headless: an SSH
session, a build server, a cluster node with no browser and no package manager.

Two ways in, and both end at the same place:

* ``install_archive`` / ``install_url`` unpack a release archive into a managed
  directory under :func:`solver_home` and record it.
* ``link_solver`` records a solver the user already has, unpacked wherever they
  like, without copying it.

"Record" means a small JSON pointer file rather than a symlink: it works the
same on Windows, it can point outside the managed root, and it carries the
provenance the archive shipped with, so ``which-solver`` can explain where a
binary came from. :func:`installed_solver` reads it, and ``find_solver``
consults that between ``$PHOTONHUB_SOLVER`` and ``PATH``. An explicit install
beats a stray binary on the path, and the environment variable still beats
everything.
"""

from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Optional, Union

from ._env import env

__all__ = [
    "SolverInstallError",
    "solver_home",
    "installed_solver",
    "installed_record",
    "stale_record",
    "writable_by_others",
    "install_archive",
    "install_url",
    "link_solver",
    "uninstall",
]

# Ceilings for a hostile or truncated archive. The real payload is under a
# megabyte; these are three orders of magnitude of headroom, not a fit.
_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
_MAX_EXPANDED_BYTES = 512 * 1024 * 1024
_MAX_MEMBERS = 512
_COPY_CHUNK = 1 << 20

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
# One directory-name component taken from provenance.json (version, git_sha).
# Letters, digits and ``. _ + -`` only, so it can never carry a separator, a
# drive or a root; ``.`` and ``..`` are refused separately.
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._+-]+")


class SolverInstallError(RuntimeError):
    """A solver could not be downloaded, verified, unpacked or recorded."""


def solver_home() -> Path:
    """Directory holding managed solver installs and the pointer file.

    ``$PHOTONHUB_SOLVER_HOME`` overrides it, for a per-project install or a
    home directory on another disk; a relative value is taken from the current
    directory and made absolute. The directory, and the pointer file in it,
    must be writable only by you: whoever can write to them decides which
    binary your client runs, so do not point this at a group-writable shared
    directory. This is not enforced; ``photonhub which-solver`` warns when it
    finds otherwise (:func:`writable_by_others`).
    """
    override = env("SOLVER_HOME")
    if override:
        return Path(os.path.abspath(Path(override).expanduser()))
    return Path.home() / ".cache" / "photonhub" / "solver"


def writable_by_others() -> list:
    """The solver home and pointer file, if either is group- or world-writable.

    POSIX only; on Windows access control lists decide this and the list is
    always empty. Missing paths are skipped.
    """
    if os.name != "posix":
        return []
    found = []
    for path in (solver_home(), _pointer_path()):
        try:
            mode = path.stat().st_mode
        except OSError:
            continue
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            found.append(path)
    return found


def _pointer_path() -> Path:
    return solver_home() / "current.json"


def _read_pointer() -> Optional[dict]:
    """The pointer file's content, whether or not its binary still exists."""
    try:
        record = json.loads(_pointer_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    return record


def installed_record() -> Optional[dict]:
    """The recorded install, or ``None`` when nothing is recorded.

    A pointer to a binary that has since been deleted reads as nothing
    recorded, so a stale pointer degrades to "not installed" rather than to an
    error in the middle of someone's run. :func:`stale_record` reports it.
    """
    record = _read_pointer()
    if record is None or not _is_executable(Path(record["path"])):
        return None
    return record


def stale_record() -> Optional[dict]:
    """The recorded install when its binary is gone or not executable.

    ``None`` when nothing is recorded or the record is usable.

    ``find_solver`` skips such a record and carries on down its order; this is
    how ``which-solver`` can say so instead of leaving the user to wonder why a
    different binary was chosen.
    """
    record = _read_pointer()
    if record is None or _is_executable(Path(record["path"])):
        return None
    return record


def installed_solver() -> Optional[Path]:
    """Path to the recorded solver, or ``None``."""
    record = installed_record()
    return Path(record["path"]) if record else None


def _is_executable(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sha256(text: str) -> str:
    """The digest in ``text``: a bare digest or a line of a ``.sha256`` file.

    ``shasum -a 256`` writes ``<digest>  <file name>``, so the first field is
    taken and compared case-insensitively. Anything that is not 64 hexadecimal
    characters is refused here, before a download or an unpack, rather than
    reported later as a mismatch that reads like tampering.
    """
    fields = str(text).split()
    digest = fields[0].lower() if fields else ""
    if not _SHA256_HEX.fullmatch(digest):
        raise SolverInstallError(
            "the digest given is not a SHA-256 digest: expected 64 hexadecimal "
            "characters, the first field of the archive's .sha256 file")
    return digest


def _within(root: Union[str, Path], path: Union[str, Path]) -> bool:
    """Whether ``path`` is ``root`` or lies under it, compared lexically."""
    root = os.path.abspath(root)
    path = os.path.abspath(path)
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:  # different drives on Windows
        return False


def _safe_members(tar: tarfile.TarFile):
    """Yield the members worth extracting, refusing anything unsafe.

    Regular files and directories only. A symlink, hard link, device or fifo is
    refused outright rather than skipped, because an archive containing one is
    not an archive we produced and the difference matters more than the
    convenience of continuing.
    """
    seen = 0
    expanded = 0
    for member in tar:
        seen += 1
        if seen > _MAX_MEMBERS:
            raise SolverInstallError(
                f"archive exceeds the {_MAX_MEMBERS} member limit")
        name = member.name
        if not name or "\x00" in name or "\\" in name:
            raise SolverInstallError(f"unsafe path in archive: {name!r}")
        parts = Path(name).parts
        if (name.startswith("/") or ".." in parts
                or any(part.startswith("/") for part in parts)):
            raise SolverInstallError(f"unsafe path in archive: {name!r}")
        if not (member.isfile() or member.isdir()):
            raise SolverInstallError(
                f"archive contains a non-regular entry: {name!r}")
        if member.isfile():
            if member.size < 0 or member.size > _MAX_EXPANDED_BYTES - expanded:
                raise SolverInstallError(
                    "archive exceeds the expanded-byte limit "
                    f"({_MAX_EXPANDED_BYTES} bytes)")
            expanded += member.size
        yield member


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in _safe_members(tar):
                target = dest / member.name
                # Belt and braces: the member names were validated above, and
                # the joined path is checked against the destination as well.
                if not _within(dest, target):
                    raise SolverInstallError(
                        f"archive escapes the destination: {member.name!r}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise SolverInstallError(
                        f"unreadable file in archive: {member.name!r}")
                try:
                    with source, target.open("xb") as handle:
                        shutil.copyfileobj(source, handle, _COPY_CHUNK)
                except FileExistsError as exc:
                    # Exclusive creation also refuses a pre-existing symlink at
                    # the destination instead of following it out of the tree.
                    raise SolverInstallError(
                        f"archive names {member.name!r} more than once") from exc
                # Carry the executable bit and nothing else from the archive.
                mode = 0o755 if member.mode & stat.S_IXUSR else 0o644
                target.chmod(mode)
    except (tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile) as exc:
        raise SolverInstallError(
            "not a readable solver archive. The download may be truncated, or "
            "it is not a .tar.gz at all (an expired link often returns a web "
            f"page instead): {exc}") from exc


def _payload_solver(root: Path) -> Path:
    """The solver inside an extracted payload, whatever the archive nested it in."""
    direct = root / "solver" / "phsolver"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob("phsolver"))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise SolverInstallError(
            "the archive does not contain solver/phsolver; this does not look "
            "like a PhotonHub solver archive")
    return matches[0]


def _verify_provenance(payload_root: Path) -> Optional[dict]:
    """Check the unpacked files against the archive's own provenance file.

    This proves the archive is internally consistent, not that it came from us;
    ``expected_sha256`` on the archive is what does that. It is still worth
    doing, because it catches a truncated download that happens to unpack.
    Every file the provenance lists must be present, inside the archive, and
    match its digest.
    """
    provenance_path = payload_root / "provenance.json"
    if not provenance_path.is_file():
        return None
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except RecursionError as exc:
        raise SolverInstallError(
            "provenance.json is nested too deeply to read") from exc
    except (OSError, ValueError) as exc:
        raise SolverInstallError(
            f"provenance.json is not readable JSON: {exc}") from exc
    if not isinstance(provenance, dict):
        raise SolverInstallError("provenance.json is not a JSON object")
    digests = provenance.get("sha256")
    if digests is None:
        return provenance
    if not isinstance(digests, dict):
        raise SolverInstallError(
            "provenance.json: sha256 is not a table of file digests")
    for relative, expected in digests.items():
        candidate = payload_root / str(relative)
        if Path(str(relative)).is_absolute() or not _within(payload_root, candidate):
            raise SolverInstallError(
                f"provenance.json lists a path outside the archive: {relative!r}")
        if not candidate.is_file():
            raise SolverInstallError(
                f"{relative} is listed in provenance.json but missing from the "
                "archive")
        expected_text = str(expected).strip().lower()
        actual = _sha256_file(candidate)
        if actual != expected_text:
            raise SolverInstallError(
                f"{relative} does not match the digest recorded in "
                f"provenance.json (expected {expected_text}, got {actual})")
    return provenance


def _install_name(provenance: Optional[dict]) -> str:
    """The ``versions/`` directory name, from provenance the archive carries.

    The archive is not trusted yet at this point, so each component must be a
    plain name: see ``_SAFE_COMPONENT``.
    """
    if not provenance:
        return "unversioned"
    parts = []
    for key in ("version", "git_sha"):
        value = provenance.get(key)
        if not value:
            continue
        text = str(value)
        if not _SAFE_COMPONENT.fullmatch(text) or text in (".", ".."):
            raise SolverInstallError(
                f"provenance.json {key} {text!r} cannot name an install "
                "directory; this is not an archive we produced")
        parts.append(text)
    return "-".join(parts) or "unversioned"


def _record(solver: Path, *, source: str, provenance: Optional[dict],
            checked: Optional[list] = None) -> dict:
    home = solver_home()
    home.mkdir(parents=True, exist_ok=True)
    record = {
        "path": str(solver),
        "source": source,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if checked is not None:
        record["checked"] = list(checked)
    if provenance:
        for key in ("version", "git_sha", "platform", "macos_floor"):
            if key in provenance:
                record[key] = provenance[key]
    pointer = _pointer_path()
    tmp = pointer.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(pointer)
    return record


def install_archive(archive: Union[str, Path], *,
                    expected_sha256: Optional[str] = None,
                    force: bool = False) -> dict:
    """Unpack a solver archive into the managed directory and record it.

    ``expected_sha256`` is the digest published with the archive, bare or as
    the line of its ``.sha256`` file. Pass it whenever you have it: it is the
    only check that says these bytes are the bytes we released, as opposed to
    bytes that merely unpack. The returned record's ``checked`` lists what was
    verified: ``"sha256"`` and, when the archive carries one,
    ``"provenance"``.
    """
    archive = Path(archive).expanduser()
    return _install(archive, expected_sha256=expected_sha256, force=force,
                    source=str(archive))


def _install(archive: Path, *, expected_sha256: Optional[str], force: bool,
             source: str) -> dict:
    expected = _parse_sha256(expected_sha256) if expected_sha256 else None
    if not archive.is_file():
        raise SolverInstallError(f"archive not found: {archive}")
    checked = []
    if expected:
        actual = _sha256_file(archive)
        if actual != expected:
            raise SolverInstallError(
                "archive digest mismatch: expected "
                f"{expected}, got {actual}. Do not use this download.")
        checked.append("sha256")

    home = solver_home()
    home.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="install-", dir=str(home)))
    payload = staging / "payload"
    try:
        _extract(archive, payload)
        solver = _payload_solver(payload)
        provenance = _verify_provenance(payload)
        if provenance is not None:
            checked.append("provenance")
        if not _is_executable(solver):
            solver.chmod(0o755)

        versions = home / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        destination = versions / _install_name(provenance)
        # The name was validated above; this checks the resolved result too,
        # before anything is deleted or moved: it must be a direct child of
        # versions/, so a symlink planted there cannot redirect the rmtree.
        real_versions = os.path.realpath(versions)
        real_destination = os.path.realpath(destination)
        if (not _within(real_versions, real_destination)
                or os.path.dirname(real_destination) != real_versions):
            raise SolverInstallError(
                f"refusing to install outside {versions}: {destination}")
        if os.path.lexists(destination):
            if not force:
                raise SolverInstallError(
                    f"{destination} already exists; pass force=True to replace it")
            if destination.is_symlink() or not destination.is_dir():
                # Not something this module made; replacing it is not ours to do.
                raise SolverInstallError(
                    f"{destination} is a symlink or a file, not an install "
                    "directory, so it is not replaced even with force; remove "
                    "it by hand if it should go")
            shutil.rmtree(destination)
        relative = solver.relative_to(payload)
        payload.rename(destination)
        installed = destination / relative
        return _record(installed, source=source, provenance=provenance,
                       checked=checked)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to another https URL.

    The plaintext check on the first URL is not a transport guarantee on its
    own: urllib follows https to http redirects by default.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        scheme = urllib.parse.urlsplit(newurl).scheme.lower()
        if scheme != "https":
            raise SolverInstallError(
                f"refusing a redirect to a non-https URL ({scheme or 'no'} "
                "scheme); the solver is only fetched over https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _public_url(url: str) -> str:
    """``url`` without credentials, query or fragment, fit to be recorded.

    An issued link usually carries its signature in the query string.
    """
    parts = urllib.parse.urlsplit(url)
    host = parts.netloc.rpartition("@")[2]
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path, "", ""))


def _download(request: urllib.request.Request, destination: Path,
              timeout: float) -> None:
    opener = urllib.request.build_opener(_HttpsOnlyRedirects)
    try:
        with opener.open(request, timeout=timeout) as response, \
                destination.open("wb") as handle:
            total = 0
            while True:
                chunk = response.read(_COPY_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    raise SolverInstallError(
                        f"download exceeds the {_MAX_DOWNLOAD_BYTES}-byte limit")
                handle.write(chunk)
    except urllib.error.HTTPError as exc:
        detail = {
            401: "the link or token was rejected",
            403: "the link has expired or is not valid for this object",
            404: "no archive at that link",
        }.get(exc.code, exc.reason)
        raise SolverInstallError(f"download failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise SolverInstallError(f"download failed: {exc.reason}") from exc
    except ValueError as exc:
        # http.client refuses a URL it cannot send: non-ASCII characters
        # (UnicodeEncodeError) or a malformed host or port (InvalidURL).
        raise SolverInstallError(
            "not a usable download link (it must be plain ASCII, with any "
            f"other characters percent-encoded): {exc}") from exc
    except (TimeoutError, http.client.HTTPException, OSError) as exc:
        raise SolverInstallError(
            f"download failed: {exc or type(exc).__name__}") from exc


def install_url(url: str, *, expected_sha256: Optional[str] = None,
                token: Optional[str] = None, force: bool = False,
                timeout: float = 60.0) -> dict:
    """Download a solver archive and install it.

    ``url`` is normally a short-lived, object-specific link issued for one
    recipient. ``token``, when given, is sent as a bearer credential instead,
    for an endpoint that authenticates the caller and then redirects; it
    defaults to ``$PHOTONHUB_SOLVER_TOKEN``. The token is never forwarded
    across a redirect, and every redirect must stay on https. The record's
    ``source`` is the URL without its query string.
    """
    if urllib.parse.urlsplit(url).scheme.lower() not in ("https", "file"):
        raise SolverInstallError(
            "refusing to download over a plaintext transport; use https")
    if expected_sha256:
        expected_sha256 = _parse_sha256(expected_sha256)  # before downloading
    if token is None:
        token = env("SOLVER_TOKEN")
    request = urllib.request.Request(url)
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    home = solver_home()
    home.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="download-", suffix=".tar.gz", dir=str(home))
    os.close(descriptor)
    downloaded = Path(name)
    try:
        _download(request, downloaded, timeout)
        return _install(downloaded, expected_sha256=expected_sha256,
                        force=force, source=_public_url(url))
    finally:
        downloaded.unlink(missing_ok=True)


def link_solver(path: Union[str, Path]) -> dict:
    """Record a solver the user already has, without copying it.

    ``path`` is the binary, or a directory holding it as ``solver/phsolver``
    (an unpacked archive) or ``phsolver``. Only those two places are looked
    at, so pointing at a large directory by mistake does not walk it.
    """
    solver = Path(path).expanduser()
    if solver.is_dir():
        candidates = (solver / "solver" / "phsolver", solver / "phsolver")
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise SolverInstallError(
                f"no solver in {solver}: looked for "
                f"{candidates[0]} and {candidates[1]}")
        solver = found
    if not solver.is_file():
        raise SolverInstallError(f"not a file: {solver}")
    if not _is_executable(solver):
        raise SolverInstallError(
            f"not executable: {solver}. Run `chmod +x {solver}` first.")
    provenance = None
    for parent in (solver.parent, solver.parent.parent):
        candidate = parent / "provenance.json"
        if candidate.is_file():
            try:
                provenance = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError):
                provenance = None
            if not isinstance(provenance, dict):
                provenance = None
            break
    return _record(solver.resolve(), source="link", provenance=provenance)


def uninstall() -> bool:
    """Forget the recorded solver. Returns whether there was one."""
    pointer = _pointer_path()
    if not pointer.exists():
        return False
    pointer.unlink()
    return True
