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


# Packages a warning looks through on its way to the user's line: pydantic
# (``BaseModel.__init__`` runs the validators and ``model_post_init``), the
# import machinery (a warning at a module's import), ``runpy`` (``python -m``)
# and the stdlib wrappers that add a frame (``contextlib``, and ``functools``
# for ``cached_property``).
_TRANSPARENT_PACKAGES = frozenset(
    {"pydantic", "pydantic_core", "importlib", "runpy", "contextlib", "functools"})
# Packages that call into this one with no user frame in between (a display
# hook rendering a result, the kernel's event loop): reaching one means the
# stack holds no user line, so the warning points at this package's outermost
# frame, its public entry point, instead of at the host's internals.
_HOST_PACKAGES = frozenset({"IPython", "ipykernel", "asyncio", "threading", "concurrent"})
_ROOTS: tuple = ()


def _package_roots() -> tuple:
    """This package's directory as imported and as resolved (``realpath``),
    each ending in the separator. They differ when the package is reached
    through a symlink, as an editable or linked install can be."""
    global _ROOTS
    if not _ROOTS:
        import os

        here = os.path.dirname(os.path.abspath(__file__))
        _ROOTS = tuple({os.path.join(here, ""), os.path.join(os.path.realpath(here), "")})
    return _ROOTS


def _in_package(frame) -> bool:
    name = frame.f_globals.get("__name__") or ""
    if name == "photonhub" or name.startswith("photonhub."):
        return True
    import os

    path = frame.f_code.co_filename
    if path.startswith("<"):
        return False
    roots = _package_roots()
    return (os.path.abspath(path).startswith(roots)
            or os.path.realpath(path).startswith(roots))


def _is_import_internal(frame) -> bool:
    """The rule ``warnings.warn`` itself uses to leave a frame out of its
    ``stacklevel`` count (CPython ``warnings._is_internal_filename``)."""
    path = frame.f_code.co_filename
    return "importlib" in path and "_bootstrap" in path


def caller_stacklevel() -> int:
    """``stacklevel`` for a warning that should point at the user's own line.

    Call it inside the ``warnings.warn(...)`` call itself, as
    ``stacklevel=caller_stacklevel()``: level 1 is the function that calls
    ``warnings.warn``. Library functions reach a warning through wrappers of
    varying depth (a ``with_*`` method calling :func:`auto_mesh`, the
    :func:`legacy_keywords` wrapper, pydantic's validation of a model built
    inside the package), so a fixed level lands in library code for some of
    them and a published notebook then prints an absolute path inside the
    installed package.

    The walk skips this package (by module name, and by path compared both
    as imported and resolved, so a symlinked install still counts as this
    package) and :data:`_TRANSPARENT_PACKAGES`, and counts frames the way
    ``warnings.warn`` does (import-machinery frames do not count). When it
    reaches a host in :data:`_HOST_PACKAGES`, or the end of the stack, before
    any user frame, it returns the level of this package's outermost frame.
    """
    import sys

    frame = sys._getframe(1)
    # warnings.warn hides nothing when the frame calling it is itself internal
    skip_internal = not _is_import_internal(frame)
    level = outermost = 1
    while True:
        frame = frame.f_back
        if frame is None:
            return outermost
        if skip_internal and _is_import_internal(frame):
            continue          # invisible to warnings.warn's own count
        level += 1
        if _in_package(frame):
            outermost = level
            continue
        top = (frame.f_globals.get("__name__") or "").partition(".")[0]
        if top in _TRANSPARENT_PACKAGES:
            continue
        if top in _HOST_PACKAGES:
            return outermost
        return level


def called_from_package() -> bool:
    """Whether the function that calls this was itself called from inside
    this package rather than by the user. A deprecated public function that
    the package still uses internally warns only on the user's direct calls;
    otherwise the warning names the user's line for a call that never named
    the deprecated function. The wrappers this module defines (such as
    :func:`legacy_keywords`) are looked through."""
    import sys

    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back
    return frame is not None and _in_package(frame)


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
                            DeprecationWarning, stacklevel=caller_stacklevel())
                    kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        wrapper.__legacy_keywords__ = dict(old_to_new)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate
