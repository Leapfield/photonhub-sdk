"""Background medium and boundary conditions (NUMERICS.md sections 3-4)."""

from pydantic import Field

from .base import BoundaryKind, FrozenModel


class Background(FrozenModel):
    """Homogeneous, nonmagnetic background; scalar relative permittivity."""

    permittivity: float = Field(default=1.0, ge=1.0)


class Boundaries(FrozenModel):
    """Per axis, both faces share one condition.

    Default is ``pml`` on all three axes: an open (radiating) domain is the
    common case, so a Simulation that does not set ``boundaries`` is absorbing
    on every face. Set an axis to ``periodic`` for a transversely-infinite /
    Bloch problem (and for the transverse axes of a plane-wave source), ``pec``
    for a hard mirror, or ``absorber`` for the adiabatic-absorber fallback when
    a structure crosses the boundary and the PML would diverge (NUMERICS.md
    §11/§21).
    """

    x: BoundaryKind = "pml"
    y: BoundaryKind = "pml"
    z: BoundaryKind = "pml"
