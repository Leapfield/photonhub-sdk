"""Keyword-spelling compatibility (CONTRIBUTING.md, "Names").

The public vocabulary abbreviates wavelength as ``wlen`` (``wlen_um``,
``wlens_um``, ``cells_per_wlen``), parallel to ``freq``. Functions that were
published under an older spelling keep accepting it through
:func:`legacy_keywords`, so existing scripts run unchanged. Whether the old
spelling warns is a package-wide switch: off in this release, turned on once
the notebooks and the docs use the new spellings (design spec
``docs/superpowers/specs/2026-09-10-setup-api-simplification-design.md`` §6).
"""

from __future__ import annotations

import functools
import warnings
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

# Flip to True to warn on every legacy spelling (phase 3 of the setup layer).
WARN_LEGACY_KEYWORDS = False


def caller_stacklevel() -> int:
    """``stacklevel`` for a warning that should point at the first frame
    OUTSIDE this package. Library functions reach a warning through wrappers
    of varying depth (a ``with_*`` method calling :func:`auto_mesh`, the
    :func:`legacy_keywords` wrapper around any public function), so a fixed
    level lands in library code for some of them."""
    import inspect
    import os

    root = os.path.dirname(os.path.abspath(__file__))
    frame = inspect.currentframe()
    level = 0
    while frame is not None:
        path = os.path.abspath(frame.f_code.co_filename)
        if level and not path.startswith(root + os.sep):
            return level
        frame, level = frame.f_back, level + 1
    return 3


def legacy_keywords(**old_to_new: str) -> Callable[[F], F]:
    """Accept ``old`` keyword spellings for parameters now named ``new``.

    ``@legacy_keywords(wavelength_um="wlen_um")`` moves a ``wavelength_um=``
    argument onto ``wlen_um`` before the call. Giving both spellings raises
    ``TypeError`` naming them. With :data:`WARN_LEGACY_KEYWORDS` set, the old
    spelling emits a ``DeprecationWarning`` pointing at the new one. The
    wrapper keeps the function's signature (``functools.wraps``), so
    ``inspect.signature`` and the API reference show the current spelling.
    Place it innermost, below ``@classmethod`` or ``@staticmethod``.
    """

    def decorate(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for old, new in old_to_new.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(
                            f"{fn.__qualname__}() got both {old!r} and {new!r}; "
                            f"{new!r} is the current spelling of that argument")
                    if WARN_LEGACY_KEYWORDS:
                        warnings.warn(
                            f"{fn.__qualname__}(): {old!r} was renamed to {new!r}; "
                            "the old spelling will be removed in a future release",
                            DeprecationWarning, stacklevel=2)
                    kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        wrapper.__legacy_keywords__ = dict(old_to_new)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate
