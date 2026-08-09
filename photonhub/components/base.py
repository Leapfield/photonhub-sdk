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


def _filename_safe(name: str) -> str:
    """Engine parity (engine/src/core/resolve.cpp check_name): the engine
    writes ``<name>.bin``, so names with path separators or the literal
    ``.``/``..`` are rejected there — fail at construction instead of at
    submission."""
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(
            f"monitor name {name!r} must be usable as a filename "
            "(no '/' or '\\', not '.' or '..')"
        )
    return name


# JSON Schema 2020-12 pattern equivalent of the validator above. pydantic's
# Rust regex engine rejects lookahead, so the ECMA pattern is published via
# json_schema_extra while enforcement lives in the AfterValidator.
MonitorName = Annotated[
    str,
    Field(min_length=1, json_schema_extra={"pattern": "^(?!\\.{1,2}$)[^/\\\\]+$"}),
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
