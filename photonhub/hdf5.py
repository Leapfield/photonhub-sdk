"""Convert a ``phsolver`` raw-output directory to a single HDF5 file.

Phase-1a HDF5 migration (master plan): the engine emits raw little-endian
float32 ``.bin`` monitors plus ``manifest.json`` (photonhub.data); this
packs that directory into one ``.h5`` so Phase-0/1a golden outputs survive
into HDF5 without an engine rebuild. The engine-native HighFive writer is
deferred to the Linux/ROCm box where libhdf5 is a package install.

Layout — deliberately "the output directory in one file":

    /                       attrs: format, manifest_version, manifest_json
    /monitors/<name>        dataset: the monitor's raw float32 array (== .bin),
                            flat, little-endian; gzip-compressed when non-empty

Everything else (run/grid/provenance metadata, monitor shapes/dims/coords,
the section-12 complex64 reconstruction and normalization) is carried by the
embedded ``manifest_json`` and rebuilt by :class:`photonhub.data.SimulationData`
using the exact same code path as the raw directory — so an HDF5 load is
bit-identical to a raw-directory load, by construction.

    from photonhub import convert_to_hdf5
    h5 = convert_to_hdf5("out/")            # -> out/simulation.h5
    data = SimulationData(h5)               # same DataArrays as SimulationData("out/")
"""

import errno
import json
import os
import stat
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Union

import numpy as np

from .data import (
    validate_monitor_manifest_entry,
    validate_result_manifest_contract,
)

#: HDF5 container contract version (distinct from the manifest_version it
#: carries). Readers gate on the leading "photonhub-hdf5-1".
H5_FORMAT = "photonhub-hdf5-1"


def _manifest_path(src: Path) -> Path:
    return src if src.name == "manifest.json" else src / "manifest.json"


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths name the same existing or lexical location."""
    try:
        if left.resolve(strict=False) == right.resolve(strict=False):
            return True
    except OSError:
        if os.path.normcase(os.path.abspath(left)) == os.path.normcase(
                os.path.abspath(right)):
            return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _open_monitor_blob(path: Path, name: str, expected: int):
    """Open and pin one regular, non-symlink monitor blob for conversion."""
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"monitor {name!r}: result file not found: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(
            f"monitor {name!r}: result file must be a regular non-symlink "
            f"file: {path}")

    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NONBLOCK", 0))
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"monitor {name!r}: result file became a symlink: {path}") from exc
        raise
    try:
        opened = os.fstat(fd)
        after = path.lstat()
        signature = _stat_signature(opened)
        if (not stat.S_ISREG(opened.st_mode)
                or _stat_signature(before) != signature
                or _stat_signature(after) != signature):
            raise ValueError(
                f"monitor {name!r}: result file changed while opening: {path}")
        expected_bytes = expected * np.dtype("<f4").itemsize
        if opened.st_size != expected_bytes:
            raise ValueError(
                f"monitor {name!r}: result file holds {opened.st_size} bytes, "
                f"manifest shape requires {expected_bytes}")
        return os.fdopen(fd, "rb", buffering=0), signature
    except BaseException as exc:
        try:
            os.close(fd)
        except OSError as cleanup_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"could not close monitor descriptor for {path}: "
                    f"{cleanup_exc}")
        raise


def _read_monitor_blob(source, signature, path: Path, entry: dict,
                       expected: int) -> np.ndarray:
    """Read a pinned blob and reject mutation during conversion."""
    name = entry["name"]
    source.seek(0)
    raw = np.fromfile(source, dtype="<f4")
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise ValueError(
            f"monitor {name!r}: result file disappeared while reading: {path}") \
            from exc
    if (_stat_signature(os.fstat(source.fileno())) != signature
            or _stat_signature(after_path) != signature
            or raw.size != expected):
        raise ValueError(
            f"monitor {name!r}: result file changed while reading: {path}")
    return raw


def convert_to_hdf5(src: Union[str, Path],
                    dest: Union[str, Path, None] = None) -> Path:
    """Pack the raw-output directory ``src`` (or its ``manifest.json``) into a
    single HDF5 file. ``dest`` defaults to ``<src>/simulation.h5``. Returns the
    written path. Requires ``h5py``."""
    import h5py

    src = Path(src)
    manifest_path = _manifest_path(src)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.json found at: {src}")
    out_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monitors = validate_result_manifest_contract(manifest, raw_files=True)

    dest = Path(dest) if dest is not None else out_dir / "simulation.h5"
    source_paths = [("manifest", manifest_path)] + [
        (f"monitor {entry['name']!r}", out_dir / entry["file"])
        for entry in monitors
    ]
    for label, source_path in source_paths:
        if _paths_alias(dest, source_path):
            raise ValueError(
                f"HDF5 destination aliases the {label} source: {dest}")

    with ExitStack() as sources:
        opened = []
        for entry in monitors:
            expected = validate_monitor_manifest_entry(entry, manifest)
            path = out_dir / entry["file"]
            source, signature = _open_monitor_blob(
                path, entry["name"], expected)
            opened.append((entry, sources.enter_context(source), signature,
                           path, expected))

        # Only create output state after every source has passed preflight.
        # A temporary in the destination directory makes the final replacement
        # atomic on every supported local filesystem.
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".photonhub-h5-", suffix=".tmp", dir=dest.parent)
        temp_path = Path(temp_name)
        try:
            temp_file = os.fdopen(fd, "w+b")
        except BaseException as exc:
            try:
                os.close(fd)
            except OSError as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        f"could not close temporary HDF5 descriptor: "
                        f"{cleanup_exc}")
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        f"could not remove temporary HDF5 file {temp_path}: "
                        f"{cleanup_exc}")
            raise
        try:
            with temp_file:
                with h5py.File(temp_file, "w") as f:
                    f.attrs["format"] = H5_FORMAT
                    f.attrs["manifest_version"] = str(
                        manifest.get("manifest_version", "1"))
                    f.attrs["manifest_json"] = json.dumps(manifest)
                    monitor_group = f.create_group("monitors")
                    for entry, source, signature, path, expected in opened:
                        name = entry["name"]
                        raw = _read_monitor_blob(
                            source, signature, path, entry, expected)
                        if raw.size:
                            monitor_group.create_dataset(
                                name, data=raw, compression="gzip")
                        else:
                            # A 0-sample monitor (e.g. an aborted-run snapshot):
                            # a chunked, compressed empty dataset is illegal.
                            monitor_group.create_dataset(name, data=raw)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, dest)
        except BaseException as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        f"could not remove temporary HDF5 file {temp_path}: "
                        f"{cleanup_exc}")
            raise
    return dest


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m photonhub.hdf5",
        description="Pack a phsolver raw-output directory into one HDF5 file.")
    parser.add_argument("src", type=Path,
                        help="output directory (or its manifest.json)")
    parser.add_argument("dest", type=Path, nargs="?", default=None,
                        help="output .h5 path (default <src>/simulation.h5)")
    args = parser.parse_args(argv)
    out = convert_to_hdf5(args.src, args.dest)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
