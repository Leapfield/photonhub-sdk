"""Shared physical constants (and the Yee tangential-component table) for the
plugin stack.

The ENGINE is the source of truth: the values here match
``engine/include/phcore/types.h`` (``kC0`` exact SI, ``kMu0`` CODATA 2018,
``kEps0 = 1/(kMu0 kC0^2)``). Every plugin imports from this module instead of
re-declaring its own copies, so the constants cannot drift between plugins (or
against the engine) again — ``yee_mode`` once carried the pre-2019
``mu0 = 4e-7*pi`` next to the CODATA ``ETA0``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

#: Free-space speed of light (m/s) — exact SI (engine ``kC0``).
C0: float = 2.99792458e8


def engine_dt_s(dl_um: float, courant: float = 0.99) -> float:
    """The engine's timestep (seconds): ``dt = courant * dl / (C0 * sqrt(3))``
    — the 3-D Courant formula of ``engine/src/core/resolve.cpp`` (NUMERICS §2).
    ``courant`` is ``RunSpec.courant`` (user-settable, default 0.99); client
    code that phases anything against the engine clock must call THIS with the
    simulation's actual courant rather than re-deriving the formula, or a
    non-default courant silently de-tunes it."""
    return float(courant) * (float(dl_um) * 1e-6) / (C0 * math.sqrt(3.0))

#: Vacuum permeability (H/m) — CODATA 2018 (engine ``kMu0``). NOT the pre-2019
#: exact ``4e-7*pi`` (which differs by ~5e-10 relative).
MU0: float = 1.25663706212e-6

#: Vacuum permittivity (F/m) = ``1/(MU0 * C0^2)`` (engine ``kEps0``).
EPS0: float = 1.0 / (MU0 * C0 * C0)

#: Vacuum wave impedance (ohms), ``eta0 = sqrt(MU0/EPS0) = MU0 * C0`` — the
#: literal equals that product to the displayed digits.
ETA0: float = 376.730313668

#: The four tangential field components (2 E, 2 H) recorded on a plane normal
#: to ``axis``, in the order ``(E_t1, E_t2, H_t1, H_t2)`` where ``(t1, t2)``
#: are the two in-plane axes with ``t1 x t2 = +axis`` (right-handed) — the
#: order the mode-overlap kernel expects.
_TANGENTIAL: Dict[str, Tuple[str, str, str, str]] = {
    "x": ("Ey", "Ez", "Hy", "Hz"),
    "y": ("Ez", "Ex", "Hz", "Hx"),
    "z": ("Ex", "Ey", "Hx", "Hy"),
}
