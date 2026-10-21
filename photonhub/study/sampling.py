"""Samplers: turn a list of study parameters into :class:`Samples`.

Every sampler returns a frozen :class:`Samples`: the parameters, one row of
values per sample, the method that made them and its settings. The random
samplers take a required integer ``seed`` and are reproducible: the same
parameters, count and seed give the same rows.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .parameters import Choices, Parameter, Span, Value, parameter_from_dict

#: The largest sample set a sampler builds; a larger request is refused
#: rather than allocated.
MAX_SAMPLES = 100_000

#: The methods a :class:`Samples` may record. ``"combined"`` is the sum of
#: two sample sets (``a + b``).
METHODS = ("grid", "points", "corners", "monte_carlo", "latin_hypercube",
           "combined")

_FORMAT = "photonhub.samples/1"


def _sample_name(row: Mapping[str, Value]) -> str:
    """Deterministic, filesystem-safe name of one row of normalized values:
    ``"s_"`` and the first 12 hex digits of the SHA-256 of its canonical
    JSON."""
    text = json.dumps(dict(row), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return "s_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _check_parameters(parameters: Sequence[Parameter]) -> Tuple[Parameter, ...]:
    if isinstance(parameters, (Span, Choices)):
        parameters = (parameters,)
    try:
        params = tuple(parameters)
    except TypeError:
        raise TypeError("parameters must be a sequence of Span or Choices") from None
    if not params:
        raise ValueError("a study needs at least one parameter")
    names = []
    for p in params:
        if not isinstance(p, (Span, Choices)):
            raise TypeError(
                f"parameters must be Span or Choices, got {type(p).__name__}")
        if p.name in names:
            raise ValueError(f"parameter name {p.name!r} appears twice")
        names.append(p.name)
    return params


def _check_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return seed


def _check_points(num: object, label: str) -> int:
    if isinstance(num, bool) or not isinstance(num, int):
        raise TypeError(f"num_points for {label!r} must be an int, got {num!r}")
    if num < 2:
        raise ValueError(f"num_points for {label!r} must be at least 2, got {num}")
    return num


def _check_product(levels, extra: int = 0) -> None:
    count = 1
    for level in levels:
        count *= len(level)
        if count + extra > MAX_SAMPLES:
            raise ValueError(
                f"this sampler would build more than MAX_SAMPLES ({MAX_SAMPLES}) rows")


def _check_count(num: object, label: str = "num_samples") -> int:
    if isinstance(num, bool) or not isinstance(num, int):
        raise TypeError(f"{label} must be an int, got {num!r}")
    if num < 1:
        raise ValueError(f"{label} must be at least 1, got {num}")
    if num > MAX_SAMPLES:
        raise ValueError(f"{label} ({num}) exceeds MAX_SAMPLES ({MAX_SAMPLES})")
    return num


@dataclass(frozen=True)
class Samples:
    """A frozen set of sample points over ``parameters``.

    ``rows`` holds one tuple of values per sample, in parameter order, each
    value normalized by its parameter. Iterating yields one ``{name: value}``
    dict per row, ``len()`` is the row count and ``samples[i]`` is row ``i``
    as a dict. ``method``, ``seed`` and ``options`` record how the rows were
    made. Rows may repeat. ``a + b`` concatenates two sample sets over the
    same parameters.
    """

    parameters: Tuple[Parameter, ...]
    rows: Tuple[Tuple[Value, ...], ...]
    method: str
    seed: Optional[int] = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        params = _check_parameters(self.parameters)
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        seed = None if self.seed is None else _check_seed(self.seed)
        try:
            raw_rows = list(self.rows)
        except TypeError:
            raise TypeError("rows must be a sequence of value tuples") from None
        _check_count(len(raw_rows), "the number of rows")
        rows = []
        for i, row in enumerate(raw_rows):
            if isinstance(row, (str, bytes)) or not hasattr(row, "__len__"):
                raise TypeError(f"row {i} must be a sequence of values")
            if len(row) != len(params):
                raise ValueError(
                    f"row {i} has {len(row)} values for {len(params)} parameters")
            rows.append(tuple(p.normalize(v) for p, v in zip(params, row)))
        options = dict(self.options)
        try:
            json.dumps(options, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"options must be JSON-safe: {exc}") from None
        object.__setattr__(self, "parameters", params)
        object.__setattr__(self, "rows", tuple(rows))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "options", MappingProxyType(options))

    @property
    def names(self) -> Tuple[str, ...]:
        """The parameter names, in order."""
        return tuple(p.name for p in self.parameters)

    @property
    def sample_names(self) -> Tuple[str, ...]:
        """One deterministic name per row, derived from its values (equal
        rows share a name)."""
        return tuple(_sample_name(row) for row in self)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Dict[str, Value]]:
        names = self.names
        for row in self.rows:
            yield dict(zip(names, row))

    def __getitem__(self, index: int) -> Dict[str, Value]:
        return dict(zip(self.names, self.rows[index]))

    def __add__(self, other: "Samples") -> "Samples":
        if not isinstance(other, Samples):
            return NotImplemented
        if other.parameters != self.parameters:
            raise ValueError("only sample sets over the same parameters add")
        return Samples(self.parameters, self.rows + other.rows, "combined",
                       options={"parts": [self.to_dict(rows=False),
                                          other.to_dict(rows=False)]})

    def to_dict(self, *, rows: bool = True) -> dict:
        """JSON-safe description; :meth:`from_dict` reverses it.
        ``rows=False`` leaves the rows out (a summary of how they were made)."""
        out = {"format": _FORMAT,
               "parameters": [p.to_dict() for p in self.parameters],
               "method": self.method, "seed": self.seed,
               "options": dict(self.options)}
        if rows:
            out["rows"] = [list(r) for r in self.rows]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Samples":
        """The :class:`Samples` a :meth:`to_dict` description (with rows)
        records; every row is checked against its parameters again."""
        if not isinstance(data, Mapping) or data.get("format") != _FORMAT:
            raise ValueError(f"not a {_FORMAT} document")
        params = tuple(parameter_from_dict(p) for p in data["parameters"])
        return cls(params, tuple(tuple(r) for r in data["rows"]),
                   data["method"], seed=data.get("seed"),
                   options=data.get("options") or {})


def points(parameters: Sequence[Parameter],
           rows: Sequence[Mapping[str, Value]]) -> Samples:
    """An explicit list of sample points: ``rows`` is a sequence of
    ``{name: value}`` mappings, each naming every parameter exactly once.
    Values are checked against their parameter (inside a span, one of the
    choices)."""
    params = _check_parameters(parameters)
    names = [p.name for p in params]
    table = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {i} must be a mapping of name to value")
        if set(row) != set(names):
            missing = sorted(set(names) - set(row))
            extra = sorted(set(row) - set(names), key=repr)
            raise ValueError(
                f"row {i} must name every parameter exactly; "
                f"missing={missing}, extra={extra}")
        table.append(tuple(row[n] for n in names))
    return Samples(params, tuple(table), "points")


def grid(parameters: Sequence[Parameter], *,
         num_points: Union[int, Mapping[str, int]] = 5) -> Samples:
    """The full factorial grid. Each :class:`Span` takes ``num_points``
    evenly spaced values from ``low`` to ``high`` inclusive (an int, or a
    ``{name: int}`` mapping naming every span; at least 2 each; an integer
    span keeps the distinct rounded values); each :class:`Choices` takes all
    its values. Rows are in row-major order: the last parameter varies
    fastest."""
    params = _check_parameters(parameters)
    spans = [p.name for p in params if isinstance(p, Span)]
    if isinstance(num_points, Mapping):
        if set(num_points) != set(spans):
            missing = sorted(set(spans) - set(num_points))
            extra = sorted(set(num_points) - set(spans), key=repr)
            raise ValueError(
                "num_points must name every Span exactly once; "
                f"missing={missing}, extra={extra}")
        counts = {name: _check_points(num_points[name], name) for name in spans}
    else:
        n = _check_points(num_points, "num_points")
        counts = {name: n for name in spans}
    levels = []
    for p in params:
        if isinstance(p, Choices):
            levels.append(p.values)
        elif p.integer:
            levels.append(tuple(sorted({int(round(v)) for v in np.linspace(
                p.low, p.high, counts[p.name])})))
        else:
            levels.append(tuple(float(v) for v in np.linspace(
                p.low, p.high, counts[p.name])))
    _check_product(levels)
    return Samples(params, tuple(itertools.product(*levels)), "grid",
                   options={"num_points": counts})


def corners(parameters: Sequence[Parameter], *, style: str = "all") -> Samples:
    """Process corners. The first row is every parameter at its nominal.
    ``style="all"`` then adds every combination of each parameter's extremes
    (``low`` and ``high`` of a span, every non-nominal value of a choice), in
    row-major order; ``style="one_at_a_time"`` adds each parameter at each
    extreme in turn with the others at nominal (span: ``low`` then ``high``;
    choice: each non-nominal value in order). A choice with a single value
    stays at it in every row. Rows equal to the nominal row are not
    repeated. ``options`` records ``{"style": style}``."""
    params = _check_parameters(parameters)
    if style not in ("all", "one_at_a_time"):
        raise ValueError(
            f"style must be 'all' or 'one_at_a_time', got {style!r}")
    nominal = tuple(p.nominal for p in params)
    extremes = []
    for p in params:
        if isinstance(p, Span):
            extremes.append((p.low, p.high))
        else:
            extremes.append(tuple(v for v in p.values if v != p.nominal
                                  or type(v) is not type(p.nominal)))
    if style == "all":
        levels = [e if e else (n,) for e, n in zip(extremes, nominal)]
        _check_product(levels, extra=1)
        others = itertools.product(*levels)
    else:
        if 1 + sum(len(e) for e in extremes) > MAX_SAMPLES:
            raise ValueError(f"corners would exceed MAX_SAMPLES ({MAX_SAMPLES})")
        others = (nominal[:j] + (v,) + nominal[j + 1:]
                  for j, e in enumerate(extremes) for v in e)
    rows = [nominal]
    rows.extend(row for row in others if row != nominal)
    return Samples(params, tuple(rows), "corners", options={"style": style})


def monte_carlo(parameters: Sequence[Parameter], *, num_samples: int,
                seed: int) -> Samples:
    """``num_samples`` independent random draws, reproducible from ``seed``.
    Unit-cube points come from ``numpy.random.Generator(PCG64(seed))
    .random((num_samples, num_parameters))``; column ``j`` is mapped through
    parameter ``j``'s ``from_unit``."""
    params = _check_parameters(parameters)
    num_samples = _check_count(num_samples)
    seed = _check_seed(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    unit = rng.random((num_samples, len(params)))
    return Samples(params, _map_unit(params, unit), "monte_carlo", seed=seed,
                   options={"num_samples": num_samples})


def latin_hypercube(parameters: Sequence[Parameter], *, num_samples: int,
                    seed: int) -> Samples:
    """A Latin hypercube of ``num_samples`` points, reproducible from
    ``seed``: for every parameter, each of the ``num_samples`` equal-
    probability strata of ``[0, 1)`` holds exactly one point before the
    parameter's ``from_unit`` maps it. For each parameter in order the
    generator draws a permutation of the strata, then the offsets inside
    them; that order is part of the reproducibility contract."""
    params = _check_parameters(parameters)
    num_samples = _check_count(num_samples)
    seed = _check_seed(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    unit = np.empty((num_samples, len(params)))
    for j in range(len(params)):
        strata = rng.permutation(num_samples)
        offsets = rng.random(num_samples)
        unit[:, j] = (strata + offsets) / num_samples
    # (k + offset) / n can round up to 1.0 in floating point for the top
    # stratum; keep every point inside [0, 1).
    np.minimum(unit, np.nextafter(1.0, 0.0), out=unit)
    return Samples(params, _map_unit(params, unit), "latin_hypercube",
                   seed=seed, options={"num_samples": num_samples})


def _map_unit(params: Tuple[Parameter, ...], unit: np.ndarray) -> tuple:
    return tuple(tuple(p.from_unit(float(u)) for p, u in zip(params, row))
                 for row in unit)
