"""Shared model base for the simulation wire format.

Field names ARE the wire format (schemas/GOVERNANCE.md): lengths in microns
(``*_um``), times in seconds (``*_s``), frequencies in Hz (``*_hz``). Models
are frozen and reject unknown keys so typos fail at construction, never
silently.
"""

from typing import Annotated, Literal, Tuple

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

FieldComponentName = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
BoundaryKind = Literal["periodic", "pec", "pml", "absorber"]
SubpixelMethodName = Literal[
    "volume", "tensor", "tensor_full", "contour", "contour_diag", "contour_full"
]
AxisName = Literal["x", "y", "z"]
DirectionName = Literal["+", "-"]

PositiveUm = Annotated[float, Field(gt=0)]
NonNegativeUm = Annotated[float, Field(ge=0)]
Vec3Um = Tuple[float, float, float]

# Frequency lists for the DFT monitors (NUMERICS.md section 12): non-empty,
# every entry strictly positive.
FreqHz = Annotated[float, Field(gt=0)]

_MONITOR_NAME_MAX_LENGTH = 251  # leaves room for the engine's ``.bin`` suffix
_MONITOR_NAME_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)
_WINDOWS_RESERVED_STEMS = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})
_ASCII_LOWER = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
)


def _monitor_name_key(name: str) -> str:
    """Filesystem-independent key used for duplicate detection."""
    return name.translate(_ASCII_LOWER)


def _is_portable_filename_token(name: str, *, max_length: int) -> bool:
    """Whether ``name`` is one portable, case-stable filename component."""
    stem = _monitor_name_key(name.split(".", 1)[0])
    return (
        1 <= len(name) <= max_length
        and all(ch in _MONITOR_NAME_CHARS for ch in name)
        and not name.startswith(".")
        and not name.endswith(".")
        and stem not in _WINDOWS_RESERVED_STEMS
    )


def _filename_safe(name: str) -> str:
    """Engine parity (engine/src/core/resolve.cpp check_name): the engine
    writes ``<name>.bin``. Restrict names to a portable ASCII token so output
    paths have the same meaning on Linux, macOS, and Windows, including their
    case and device-name rules."""
    if not _is_portable_filename_token(
        name, max_length=_MONITOR_NAME_MAX_LENGTH
    ):
        raise ValueError(
            f"monitor name {name!r} must be a portable filename token: "
            "1-251 ASCII letters, digits, '_', '-', or '.', with no leading "
            "or trailing '.', and no reserved Windows device basename"
        )
    return name


# JSON Schema 2020-12 pattern equivalent of the validator above. pydantic's
# Rust regex engine rejects lookahead, so the ECMA pattern is published via
# json_schema_extra while enforcement lives in the AfterValidator.
MonitorName = Annotated[
    str,
    Field(
        min_length=1,
        max_length=_MONITOR_NAME_MAX_LENGTH,
        json_schema_extra={
            "pattern": (
                r"^(?!\.)(?!.*\.$)"
                r"(?!(?:[Cc][Oo][Nn]|[Pp][Rr][Nn]|[Aa][Uu][Xx]|"
                r"[Nn][Uu][Ll]|[Cc][Oo][Mm][1-9]|[Ll][Pp][Tt][1-9])"
                r"(?:\.|$))[A-Za-z0-9_.-]+$"
            )
        },
    ),
    AfterValidator(_filename_safe),
]

# Engine parity: phsolver's as_int rejects integers outside int32
# (engine/src/io/spec_io.cpp), so step counts/intervals are bounded here too.
MAX_INT32 = 2**31 - 1


class FrozenModel(BaseModel):
    # allow_inf_nan=False: the engine rejects non-finite numbers everywhere
    # (NaN/Inf serialize to JSON null, which phsolver refuses; raw NaN
    # literals are an nlohmann parse error), so they must fail at model
    # construction on every float field — including gt/ge-constrained ones,
    # which +inf would otherwise pass.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
