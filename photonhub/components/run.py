"""Run controls (NUMERICS.md section 2)."""

from typing import Optional

from pydantic import ConfigDict, Field, model_validator

from .base import MAX_INT32, FrozenModel

# Exactly-one-of, expressed for third-party schema consumers with the same
# null-as-absent semantics both the pydantic runtime and the engine apply
# (an explicit JSON null counts as "not given"): each oneOf branch requires
# one key present with its non-null type while the other key, if present at
# all, must be null.
_RUN_ONE_OF = {
    "oneOf": [
        {
            "required": ["run_time_s"],
            "properties": {
                "run_time_s": {"type": "number"},
                "n_steps": {"type": "null"},
            },
        },
        {
            "required": ["n_steps"],
            "properties": {
                "n_steps": {"type": "integer"},
                "run_time_s": {"type": "null"},
            },
        },
    ]
}


class RunSpec(FrozenModel):
    """Exactly one of ``run_time_s`` / ``n_steps``. dt = courant * dl /
    (c0 * sqrt(3)); courant = 1.0 sits on the 3-D stability limit and is
    rejected."""

    model_config = ConfigDict(json_schema_extra=_RUN_ONE_OF)

    run_time_s: Optional[float] = Field(default=None, gt=0)
    # le bound: the engine's as_int rejects n_steps beyond int32 at parse
    # time, so larger values must fail at construction, not at submission.
    n_steps: Optional[int] = Field(default=None, ge=1, le=MAX_INT32)
    courant: float = Field(default=0.99, gt=0, le=0.9999)
    # NUMERICS.md section 7 auto-shutoff (run-until-field-decay): the run may
    # finish before run_time_s/n_steps once the field energy decays below this
    # fraction of its peak (after the sources stop). 0 disables; default 1e-5
    # (Tidy3D parity). The engine's resolve.cpp validate() is authoritative.
    shutoff: float = Field(default=1.0e-5, ge=0, lt=1)

    @model_validator(mode="after")
    def _exactly_one_duration(self) -> "RunSpec":
        if (self.run_time_s is None) == (self.n_steps is None):
            raise ValueError("exactly one of 'run_time_s' or 'n_steps' must be set")
        return self
