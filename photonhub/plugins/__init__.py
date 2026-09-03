"""Deprecated alias for :mod:`photonhub.analysis` (renamed 2026-09; removed in 0.2)."""

import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "photonhub.plugins was renamed to photonhub.analysis; the old module path "
    "will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from .. import analysis as _analysis  # noqa: E402
from ..analysis import *  # noqa: E402,F401,F403

__all__ = list(_analysis.__all__)

for _m in _pkgutil.iter_modules(_analysis.__path__):
    _sys.modules[f"{__name__}.{_m.name}"] = _importlib.import_module(
        f"{_analysis.__name__}.{_m.name}"
    )
