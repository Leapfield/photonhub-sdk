"""Physical constants, in SI units, shared by the SDK and the notebooks.

``ph.c0`` and ``ph.eps0`` are the values the solver itself uses
(``engine/include/phcore/types.h``): the speed of light is exact SI, the
vacuum permeability is CODATA 2018, and the vacuum permittivity follows from
them as ``1 / (mu0 * c0**2)``. Every module of the SDK takes its constants
from here, so a wavelength converted in a notebook, in the SDK and in the
engine lands on the same number.
"""

from __future__ import annotations

__all__ = ["c0", "eps0", "mu0"]

c0: float = 299792458.0
"""Speed of light in vacuum (m/s), exact SI (the solver's ``kC0``)."""

mu0: float = 1.25663706212e-6
"""Vacuum permeability (H/m), CODATA 2018 (the solver's ``kMu0``); not the
pre-2019 ``4e-7 * pi``, which differs by about 5e-10 relative."""

eps0: float = 1.0 / (mu0 * c0 * c0)
"""Vacuum permittivity (F/m), ``1 / (mu0 * c0**2)`` (the solver's ``kEps0``).
Equal to the CODATA 2018 value 8.8541878128e-12 to its eleven stated digits."""
