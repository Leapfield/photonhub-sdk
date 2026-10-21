"""Digest a Simulation's canonical wire document."""

import hashlib

from .components import Simulation


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def spec_sha256(sim) -> str:
    """Return the SHA-256 digest of the canonical wire document.

    The digest covers ``sim.to_wire_json(indent=0)``. A cloud service that
    stores this document unchanged records the same input digest. A submission
    that omits an unsupported optional field, such as ``run.dft_shutoff`` on
    the GPU, stores a different document. Identical wire inputs have identical
    digests. Client-only
    authoring state, including declared ports and frame origin, is not part of
    the wire document and is not hashed. ``sim`` may be a Simulation or its
    wire JSON as text or bytes.
    """
    if isinstance(sim, (str, bytes)):
        sim = Simulation.from_wire_json(sim)
    if not isinstance(sim, Simulation):
        raise TypeError("spec_sha256 expects a Simulation or wire JSON text or bytes")
    return _sha256_hex(sim.to_wire_json(indent=0).encode("utf-8"))
