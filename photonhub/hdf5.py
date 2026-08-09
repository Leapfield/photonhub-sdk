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

import json
from pathlib import Path
from typing import Union

import numpy as np

#: HDF5 container contract version (distinct from the manifest_version it
#: carries). Readers gate on the leading "photonhub-hdf5-1".
H5_FORMAT = "photonhub-hdf5-1"


def _manifest_path(src: Path) -> Path:
    return src if src.name == "manifest.json" else src / "manifest.json"


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

    dest = Path(dest) if dest is not None else out_dir / "simulation.h5"
    dest.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dest, "w") as f:
        f.attrs["format"] = H5_FORMAT
        f.attrs["manifest_version"] = str(manifest.get("manifest_version", "1"))
        f.attrs["manifest_json"] = json.dumps(manifest)
        monitors = f.create_group("monitors")
        for entry in manifest.get("monitors", []):
            name = entry["name"]  # validated filename-safe, so h5-path safe
            raw = np.fromfile(out_dir / entry["file"], dtype="<f4")
            if raw.size:
                monitors.create_dataset(name, data=raw, compression="gzip")
            else:
                # A 0-sample monitor (e.g. an aborted-run snapshot): a chunked,
                # compressed empty dataset is illegal, so store it plain.
                monitors.create_dataset(name, data=raw)
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
