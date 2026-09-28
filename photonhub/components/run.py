"""Run controls (NUMERICS.md section 2)."""

from typing import Optional

from pydantic import AliasChoices, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

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
    """Choose at most one duration: ``run_time_s``, ``n_steps``, or ``transits``.

    With none supplied, ``Simulation`` uses a default cap of 40 transits.
    One transit is ``max(size_um) * 1e-6 * n_max / c0`` seconds: travel across
    the longest resolved domain extent at speed ``c0 / n_max``. ``n_max`` is
    the maximum of the background index and the resolver's structure-index
    estimates at the reference wavelength (1.55 um when none is supplied).
    ``Simulation`` resolves ``transits`` to ``run_time_s`` before serialization.
    The resolved solver document requires exactly one of ``run_time_s`` and
    ``n_steps`` and never includes ``transits``. Auto-shutoff may end a run
    before this duration cap.

    On a uniform 3D grid, ``dt = courant * dl / (c0 * sqrt(3))``.
    ``courant = 1.0`` is on the stability limit and is rejected."""

    model_config = ConfigDict(json_schema_extra=_RUN_ONE_OF)

    run_time_s: Optional[float] = Field(default=None, gt=0)
    # Client-only third way to give the duration (design spec §4.1): the run
    # length in transits of the longest domain axis at the highest structure
    # index. The Simulation resolves it to run_time_s at construction; it never
    # reaches the wire or the schema.
    transits: SkipJsonSchema[Optional[float]] = Field(default=None, gt=0)
    # le bound: the engine's as_int rejects n_steps beyond int32 at parse
    # time, so larger values must fail at construction, not at submission.
    # ``num_steps`` is the current spelling (CONTRIBUTING.md, Names); ``n_steps``
    # stays the attribute and the wire key, listed first so the generated schema
    # keeps it.
    n_steps: Optional[int] = Field(
        default=None, ge=1, le=MAX_INT32,
        validation_alias=AliasChoices("n_steps", "num_steps"))
    courant: float = Field(default=0.99, gt=0, le=0.9999)
    # NUMERICS.md section 7 auto-shutoff (run-until-field-decay): the run may
    # finish before run_time_s/n_steps once the field energy decays below this
    # fraction of its peak (after the sources stop). 0 disables; default 1e-5
    # (standard behavior). The engine's resolve.cpp validate() is authoritative.
    shutoff: float = Field(default=1.0e-5, ge=0, lt=1)
    # Positive values guard frequency-domain results on capable CPU solvers.
    # None is target-aware: local CPU injects 1e-4 when supported; saved specs,
    # GPU and cloud retain the legacy energy-only wire.
    dft_shutoff: Optional[float] = Field(
        default=None, ge=0, lt=1,
        description=("Relative threshold for an estimate of remaining change "
                     "against each monitor's frequency-band peak on local CPU DFT and flux "
                     "monitors. None uses 1e-4 on a capable CPU solver; "
                     "zero disables the guard. Unsupported targets omit it "
                     "with a warning and use energy-only auto-shutoff. "
                     "The estimate assumes one geometric decay per frequency "
                     "and is not a guarantee for beating, several interfering "
                     "modes in one bin, or late arrivals."))
    """Optional monitor-band decay estimate threshold for capable CPU runs."""

    @model_validator(mode="after")
    def _exactly_one_duration(self) -> "RunSpec":
        # At most one duration. None at all is the client-side "take the
        # Simulation's transit cap" (setup layer phase 5): the Simulation fills
        # transits in and rejects a wire document whose run has no duration.
        given = [k for k in ("run_time_s", "n_steps", "transits") if getattr(self, k) is not None]
        if len(given) > 1:
            raise ValueError(
                f"exactly one of 'run_time_s', 'n_steps' or 'transits' must be set, got {given}")
        if self.dft_shutoff is not None and self.dft_shutoff > 0 and self.shutoff == 0:
            raise ValueError("positive dft_shutoff requires shutoff > 0")
        return self
