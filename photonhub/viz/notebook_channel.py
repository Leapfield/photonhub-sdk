"""Bounded, private wire format for the notebook child process."""

from __future__ import annotations

import json
import math
import struct

MAX_FRAME_BYTES = 32 * 1024 * 1024


class ChannelError(Exception):
    pass


def read_frame(stream) -> dict:
    header = _read_exact(stream, 4)
    if len(header) != 4:
        raise ChannelError("notebook channel closed")
    size = struct.unpack("!I", header)[0]
    if size == 0 or size > MAX_FRAME_BYTES:
        raise ChannelError("invalid notebook frame size")
    data = _read_exact(stream, size)
    if len(data) != size:
        raise ChannelError("incomplete notebook frame")
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate notebook message key")
            result[key] = item
        return result

    def invalid_constant(_value):
        raise ValueError("invalid notebook number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("invalid notebook number")
        return number

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                           parse_constant=invalid_constant, parse_float=finite_float)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ChannelError("invalid notebook frame JSON") from exc
    if not isinstance(value, dict):
        raise ChannelError("notebook frame must be an object")
    return value


def write_frame(stream, value: dict) -> None:
    try:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ChannelError("notebook message is not JSON") from exc
    if not data or len(data) > MAX_FRAME_BYTES:
        raise ChannelError("notebook message exceeds the frame limit")
    frame = memoryview(struct.pack("!I", len(data)) + data)
    while frame:
        written = stream.write(frame)
        if written is None or written <= 0:
            raise ChannelError("notebook channel closed")
        frame = frame[written:]
    stream.flush()


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def worker_environment(source: dict[str, str]) -> dict[str, str]:
    """Build from benign launch essentials, never from a scrubbed copy."""
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL",
               "LC_CTYPE", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT",
               "WINDIR", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
               "MPLCONFIGDIR", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
               "USERNAME", "COMSPEC", "PATHEXT"}
    result = {name: value for name, value in source.items() if name.upper() in allowed}
    result["MPLBACKEND"] = "Agg"
    return result
