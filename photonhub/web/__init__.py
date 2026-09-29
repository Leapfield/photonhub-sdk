"""Deprecated alias for :mod:`photonhub.cloud` (renamed 2026-09; removed in 0.2)."""

import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys
import warnings as _warnings

from .._compat import caller_stacklevel as _caller_stacklevel

_warnings.warn(
    "photonhub.web was renamed to photonhub.cloud; the old module path will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=_caller_stacklevel(),
)

from .. import cloud as _cloud  # noqa: E402
from ..cloud import *  # noqa: E402,F401,F403

__all__ = list(_cloud.__all__) + ["WebConfig", "WebError", "WebJobTimeout"]
WebConfig = _cloud.CloudConfig
WebError = _cloud.CloudError
WebJobTimeout = _cloud.CloudJobTimeout


def __getattr__(name):
    # Everything else ph.web answered when it was photonhub.cloud itself: the
    # submodules and cloud's own renamed names (run_async, with its warning).
    return getattr(_cloud, name)


for _m in _pkgutil.iter_modules(_cloud.__path__):
    _sys.modules[f"{__name__}.{_m.name}"] = _importlib.import_module(
        f"{_cloud.__name__}.{_m.name}"
    )
