"""The study results table: one row per sample, as an :class:`xarray.Dataset`.

:func:`build_table` assembles it from the samples and their outcomes,
:func:`save` and :func:`load` keep it as deterministic JSON, and
:func:`yield_fraction` counts the samples that meet pass criteria.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from ._outcome import RESERVED, SampleOutcome
from .parameters import Choices
from .sampling import Samples

FORMAT = "photonhub.study_table/1"

# The table's own string columns, in the order they are added.
_STRING_COLUMNS = ("status", "error", "job", "job_id", "spec_sha256",
                   "solver_version", "backend")


def _photonhub_version() -> str:
    from .. import __version__

    return __version__


def _parameter_column(param, values):
    if isinstance(param, Choices):
        family = param.family
        if family == "str":
            return np.array(values, dtype=str)
        if family == "bool":
            return np.array(values, dtype=bool)
        if all(isinstance(v, numbers.Integral) for v in param.values):
            return np.array(values, dtype=np.int64)
        return np.array(values, dtype=np.float64)
    if param.integer:
        return np.array(values, dtype=np.int64)
    return np.array(values, dtype=np.float64)


def _measured_value(name: str, value: object, row: int) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"measured {name!r} of row {row} must be a real number, got {value!r}")
    return float(value)


def build_table(samples: Samples,
                outcomes: Sequence[SampleOutcome]) -> xr.Dataset:
    """The results table for ``samples`` and one :class:`SampleOutcome` per
    row (same order).

    Dimension ``sample`` is the row position, with coordinate
    ``sample_name``. Data variables: each parameter (attrs ``role``,
    ``units``, ``kind``), each measured scalar sorted by name (float64, NaN
    where a row has none; attrs ``role``), then ``status``, ``error``,
    ``job``, ``job_id``, ``spec_sha256``, ``solver_version``, ``backend``
    and ``quote_usd`` (NaN where none)."""
    if not isinstance(samples, Samples):
        raise TypeError(f"samples must be Samples, got {type(samples).__name__}")
    outcomes = list(outcomes)
    if len(outcomes) != len(samples):
        raise ValueError(
            f"{len(outcomes)} outcomes for {len(samples)} sample rows")
    for o in outcomes:
        if not isinstance(o, SampleOutcome):
            raise TypeError(
                f"outcomes must be SampleOutcome, got {type(o).__name__}")
    param_names = set(samples.names)
    measured_names = sorted({k for o in outcomes for k in o.measured})
    for name in measured_names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"a measured name must be a non-empty str, got {name!r}")
        if name in RESERVED:
            raise ValueError(
                f"measured name {name!r} is reserved for a results-table column")
        if name in param_names:
            raise ValueError(f"measured name {name!r} is also a parameter name")

    n = len(samples)
    dims = ("sample",)
    data_vars = {}
    for j, param in enumerate(samples.parameters):
        column = _parameter_column(param, [row[j] for row in samples.rows])
        data_vars[param.name] = xr.Variable(
            dims, column,
            attrs={"role": "parameter", "units": param.unit, "kind": param.kind})
    for name in measured_names:
        column = np.full(n, np.nan, dtype=np.float64)
        for i, o in enumerate(outcomes):
            if name in o.measured:
                column[i] = _measured_value(name, o.measured[name], i)
        data_vars[name] = xr.Variable(dims, column, attrs={"role": "measured"})
    for name in _STRING_COLUMNS:
        data_vars[name] = xr.Variable(
            dims, np.array([getattr(o, name) for o in outcomes], dtype=str))
    data_vars["quote_usd"] = xr.Variable(
        dims, np.array([np.nan if o.quote_usd is None else o.quote_usd
                        for o in outcomes], dtype=np.float64),
        attrs={"units": "USD"})
    coords = {
        "sample": np.arange(n, dtype=np.int64),
        "sample_name": ("sample", np.array(samples.sample_names, dtype=str)),
    }
    attrs = {
        "format": FORMAT,
        "method": samples.method,
        "seed": -1 if samples.seed is None else int(samples.seed),
        "photonhub_version": _photonhub_version(),
        "num_samples": n,
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


_DTYPES = {"float64": np.float64, "int64": np.int64, "bool": bool, "str": str}


def _dtype_name(array: np.ndarray, label: str) -> str:
    kind = array.dtype.kind
    if kind == "f":
        return "float64"
    if kind in "iu":
        return "int64"
    if kind == "b":
        return "bool"
    if kind == "U":
        return "str"
    raise TypeError(f"{label} has dtype {array.dtype}, which a study table cannot hold")


def _json_values(array: np.ndarray, dtype: str) -> list:
    if dtype == "float64":
        return [None if math.isnan(v) else float(v) for v in array.tolist()]
    if dtype == "int64":
        return [int(v) for v in array.tolist()]
    if dtype == "bool":
        return [bool(v) for v in array.tolist()]
    return [str(v) for v in array.tolist()]


def _from_json_values(values: list, dtype: str, label: str) -> np.ndarray:
    if dtype not in _DTYPES:
        raise ValueError(f"{label} has unknown dtype {dtype!r}")
    if dtype == "float64":
        return np.array([np.nan if v is None else v for v in values],
                        dtype=np.float64)
    return np.array(values, dtype=_DTYPES[dtype])


def _check_attrs(attrs: Mapping, label: str) -> dict:
    out = {}
    for key, value in attrs.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} attr names must be str, got {key!r}")
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} attr {key!r} must be finite")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(
                f"{label} attr {key!r} must be a str or a number, got {value!r}")
        out[key] = value
    return out


def save(table: xr.Dataset, file: Union[str, Path]) -> Path:
    """Write ``table`` (a :func:`build_table` dataset) to ``file`` as
    deterministic JSON and return the path. NaN is written as ``null``; dtypes
    and attrs are recorded so :func:`load` gives an identical dataset. The
    write is atomic (a sibling temporary file, then ``os.replace``)."""
    if not isinstance(table, xr.Dataset) or table.attrs.get("format") != FORMAT:
        raise ValueError(f"save takes a {FORMAT} dataset from a study run")
    if tuple(table.dims) != ("sample",):
        raise ValueError("a study table has the single dimension 'sample'")
    coords = {}
    for name in ("sample", "sample_name"):
        array = np.asarray(table.coords[name].values)
        dtype = _dtype_name(array, f"coordinate {name!r}")
        coords[name] = {"dtype": dtype, "values": _json_values(array, dtype)}
    variables = []
    for name, var in table.data_vars.items():
        if var.dims != ("sample",):
            raise ValueError(f"variable {name!r} must have the dimension 'sample'")
        array = np.asarray(var.values)
        dtype = _dtype_name(array, f"variable {name!r}")
        variables.append({
            "name": str(name), "dtype": dtype,
            "attrs": _check_attrs(var.attrs, f"variable {name!r}"),
            "values": _json_values(array, dtype),
        })
    document = {
        "format": FORMAT,
        "attrs": _check_attrs(table.attrs, "table"),
        "coords": coords,
        "variables": variables,
    }
    text = json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n"
    target = Path(file)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                               dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def load(file: Union[str, Path]) -> xr.Dataset:
    """Read a table written by :func:`save`. A directory reads its
    ``table.json`` (a study directory)."""
    source = Path(file)
    if source.is_dir():
        source = source / "table.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ValueError(f"{source} is not a {FORMAT} document")
    dims = ("sample",)
    data_vars = {}
    for entry in document["variables"]:
        name = entry["name"]
        data_vars[name] = xr.Variable(
            dims, _from_json_values(entry["values"], entry["dtype"],
                                    f"variable {name!r}"),
            attrs=dict(entry.get("attrs") or {}))
    raw = document["coords"]
    coords = {
        "sample": _from_json_values(raw["sample"]["values"],
                                    raw["sample"]["dtype"], "coordinate 'sample'"),
        "sample_name": ("sample", _from_json_values(
            raw["sample_name"]["values"], raw["sample_name"]["dtype"],
            "coordinate 'sample_name'")),
    }
    return xr.Dataset(data_vars, coords=coords,
                      attrs=dict(document.get("attrs") or {}))


@dataclass(frozen=True)
class YieldEstimate:
    """The share of samples meeting pass criteria, with the counts behind it.

    ``fraction`` is ``num_pass / num_evaluated`` (NaN when nothing was
    evaluated); ``num_samples`` is every row, ``num_incomplete`` the rows that
    did not succeed, and ``incomplete`` says whether those were excluded or
    counted as failing. ``interval_95`` is the Wilson score interval at 95%
    confidence."""

    fraction: float
    num_pass: int
    num_evaluated: int
    num_samples: int
    num_incomplete: int
    incomplete: str
    criteria: Mapping[str, Tuple[Optional[float], Optional[float]]]
    interval_95: Tuple[float, float]

    def __str__(self) -> str:
        low, high = self.interval_95
        treated = ("excluded" if self.incomplete == "exclude"
                   else "counted as failing")
        return (f"yield {self.fraction:.3f}: {self.num_pass} of "
                f"{self.num_evaluated} evaluated samples pass (95% interval "
                f"{low:.3f} to {high:.3f}); {self.num_incomplete} of "
                f"{self.num_samples} samples incomplete, {treated}")


def yield_fraction(table: xr.Dataset,
                   criteria: Mapping[str, Tuple[Optional[float], Optional[float]]],
                   *, incomplete: str = "exclude") -> YieldEstimate:
    """Count the rows of ``table`` whose measured values meet ``criteria``,
    ``{name: (low, high)}`` with inclusive bounds and ``None`` for an open
    side. Rows whose ``status`` is not ``"succeeded"`` (or that lack a value
    a criterion names) are incomplete: ``incomplete="exclude"`` leaves them
    out of the denominator, ``"fail"`` counts them as failing.

    The fraction is a count over the samples given; it is only as good as the
    sampling and the simulations behind it."""
    if not isinstance(table, xr.Dataset) or table.attrs.get("format") != FORMAT:
        raise ValueError(f"yield_fraction takes a {FORMAT} dataset from a study run")
    if incomplete not in ("exclude", "fail"):
        raise ValueError(
            f"incomplete must be 'exclude' or 'fail', got {incomplete!r}")
    if not isinstance(criteria, Mapping) or not criteria:
        raise ValueError("criteria must be a non-empty mapping {name: (low, high)}")
    measured = sorted(str(name) for name, var in table.data_vars.items()
                      if var.attrs.get("role") == "measured")
    bounds = {}
    for name, pair in criteria.items():
        if name not in measured:
            raise ValueError(
                f"criterion {name!r} is not a measured quantity of this table; "
                f"measured: {measured}")
        try:
            low, high = pair
        except (TypeError, ValueError):
            raise ValueError(
                f"criterion {name!r} must be a (low, high) pair") from None
        low = _bound(low, name, "low")
        high = _bound(high, name, "high")
        if low is None and high is None:
            raise ValueError(f"criterion {name!r} needs at least one bound")
        if low is not None and high is not None and low > high:
            raise ValueError(f"criterion {name!r} has low > high")
        bounds[name] = (low, high)
    complete = np.asarray(table["status"].values) == "succeeded"
    passing = complete.copy()
    for name, (low, high) in bounds.items():
        values = np.asarray(table[name].values, dtype=np.float64)
        finite = np.isfinite(values)
        complete &= finite
        ok = finite.copy()
        if low is not None:
            ok &= values >= low
        if high is not None:
            ok &= values <= high
        passing &= ok
    passing &= complete
    num_samples = int(table.sizes["sample"])
    num_complete = int(complete.sum())
    num_pass = int(passing.sum())
    num_evaluated = num_complete if incomplete == "exclude" else num_samples
    fraction = num_pass / num_evaluated if num_evaluated else math.nan
    return YieldEstimate(
        fraction=fraction, num_pass=num_pass, num_evaluated=num_evaluated,
        num_samples=num_samples, num_incomplete=num_samples - num_complete,
        incomplete=incomplete, criteria=MappingProxyType(bounds),
        interval_95=_wilson(num_pass, num_evaluated))


def _bound(value: object, name: str, side: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{side} bound of criterion {name!r} must be a real number or None")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{side} bound of criterion {name!r} must be finite")
    return out


_Z95 = 1.959963984540054


def _wilson(num_pass: int, num: int) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion at 95% confidence."""
    if num == 0:
        return (math.nan, math.nan)
    p = num_pass / num
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / num
    centre = (p + z2 / (2 * num)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / num + z2 / (4 * num * num)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
