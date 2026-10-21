"""Study parameters: the named inputs a design study varies.

A parameter is a frozen value with a name, the values it may take and a unit.
:class:`Span` is a numeric interval (float, or integer with ``integer=True``);
:class:`Choices` is a finite list of values. The samplers in
:mod:`photonhub.study.sampling` turn a list of parameters into
:class:`~photonhub.study.Samples`.
"""

from __future__ import annotations

import keyword
import math
import numbers
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from ._outcome import RESERVED

Value = Union[float, int, str, bool]

# Unit inferred from a name's suffix when ``unit`` is not given.
_UNIT_SUFFIXES = (
    ("_um", "um"), ("_nm", "nm"), ("_hz", "Hz"), ("_rad", "rad"),
    ("_deg", "deg"), ("_s", "s"),
)

_MAX_NAME = 64


def _check_name(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"a parameter name must be a non-empty str, got {name!r}")
    if (not name.isidentifier() or keyword.iskeyword(name)
            or name.startswith("_") or len(name) > _MAX_NAME):
        raise ValueError(
            f"parameter name {name!r} must be a Python identifier of at most "
            f"{_MAX_NAME} characters, not a keyword, without a leading '_'")
    if name in RESERVED:
        raise ValueError(
            f"parameter name {name!r} is reserved for a results-table column; "
            f"reserved names: {sorted(RESERVED)}")
    return name


def _resolve_unit(name: str, unit: object) -> str:
    if unit is None:
        for suffix, inferred in _UNIT_SUFFIXES:
            if name.endswith(suffix):
                return inferred
        return ""
    if not isinstance(unit, str):
        raise TypeError(f"unit of {name!r} must be a str, got {unit!r}")
    return unit


def _is_real(value: object) -> bool:
    return (isinstance(value, numbers.Real) and not isinstance(value, bool))


def _finite_float(value: object, label: str) -> float:
    if not _is_real(value):
        raise TypeError(f"{label} must be a real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return out


def _unit(u: object) -> float:
    if not _is_real(u) or not 0.0 <= float(u) < 1.0:
        raise ValueError(f"u must be a real number in [0, 1), got {u!r}")
    return float(u)


def _integral(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be an integer, got {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float) or not as_float.is_integer():
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return int(as_float)


@dataclass(frozen=True)
class Span:
    """A numeric parameter on the closed interval ``[low, high]``.

    ``nominal`` is the design value (default: the midpoint, rounded down for
    an integer span); the corner sampler uses it and the random samplers
    centre a ``sigma`` draw on it. ``sigma`` (float spans only) makes
    :func:`~photonhub.study.monte_carlo` and
    :func:`~photonhub.study.latin_hypercube` draw from a normal distribution
    around ``nominal`` truncated to ``[low, high]``; without it they draw
    uniformly. ``unit`` is display metadata; when omitted it is inferred from
    the name's suffix (``width_um`` gives ``"um"``). ``integer=True`` keeps
    every value an ``int``.
    """

    name: str
    low: float
    high: float
    nominal: Optional[float] = None
    sigma: Optional[float] = None
    unit: Optional[str] = None
    integer: bool = False

    def __post_init__(self):
        name = _check_name(self.name)
        if not isinstance(self.integer, bool):
            raise TypeError(f"integer of {name!r} must be a bool")
        if self.integer:
            low = _integral(self.low, f"low of {name!r}")
            high = _integral(self.high, f"high of {name!r}")
            if self.sigma is not None:
                raise ValueError(
                    f"sigma of {name!r} is for float spans; an integer span "
                    "draws uniformly")
            sigma = None
        else:
            low = _finite_float(self.low, f"low of {name!r}")
            high = _finite_float(self.high, f"high of {name!r}")
            sigma = (None if self.sigma is None
                     else _finite_float(self.sigma, f"sigma of {name!r}"))
            if sigma is not None and sigma <= 0:
                raise ValueError(f"sigma of {name!r} must be positive, got {sigma}")
        if not low < high:
            raise ValueError(
                f"{name!r} needs low < high, got low={low}, high={high}; a "
                "fixed value is Choices(name, [value])")
        if self.nominal is None:
            nominal = (low + high) // 2 if self.integer else 0.5 * (low + high)
        elif self.integer:
            nominal = _integral(self.nominal, f"nominal of {name!r}")
        else:
            nominal = _finite_float(self.nominal, f"nominal of {name!r}")
        if not low <= nominal <= high:
            raise ValueError(
                f"nominal of {name!r} ({nominal}) is outside [{low}, {high}]")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "nominal", nominal)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "unit", _resolve_unit(name, self.unit))

    @property
    def kind(self) -> str:
        """``"span"``."""
        return "span"

    def normalize(self, value: object) -> Union[float, int]:
        """``value`` as this parameter stores it (``float``, or ``int`` for
        an integer span), refused when it is outside ``[low, high]``."""
        label = f"value of {self.name!r}"
        out = (_integral(value, label) if self.integer
               else _finite_float(value, label))
        if not self.low <= out <= self.high:
            raise ValueError(
                f"{label} ({out}) is outside [{self.low}, {self.high}]")
        return out

    def from_unit(self, u: float) -> Union[float, int]:
        """Map ``u`` in ``[0, 1)`` to a value (the inverse CDF of this
        parameter's sampling distribution): uniform on ``[low, high]``, a
        normal around ``nominal`` truncated to ``[low, high]`` when ``sigma``
        is set, or uniform over the integers of an integer span."""
        u = _unit(u)
        if self.integer:
            return self.low + min(math.floor(u * (self.high - self.low + 1)),
                                  self.high - self.low)
        if self.sigma is None:
            value = self.low + u * (self.high - self.low)
        else:
            from scipy.stats import truncnorm

            a = (self.low - self.nominal) / self.sigma
            b = (self.high - self.nominal) / self.sigma
            value = float(truncnorm.ppf(u, a, b, loc=self.nominal,
                                        scale=self.sigma))
        return float(min(max(value, self.low), self.high))

    def to_dict(self) -> dict:
        """JSON-safe description; :meth:`from_dict` reverses it."""
        return {"kind": "span", "name": self.name, "low": self.low,
                "high": self.high, "nominal": self.nominal,
                "sigma": self.sigma, "unit": self.unit,
                "integer": self.integer}

    @classmethod
    def from_dict(cls, data: dict) -> "Span":
        """The :class:`Span` a :meth:`to_dict` description records."""
        if data.get("kind") != "span":
            raise ValueError(f"not a Span description: {data!r}")
        return cls(data["name"], data["low"], data["high"],
                   nominal=data.get("nominal"), sigma=data.get("sigma"),
                   unit=data.get("unit"), integer=bool(data.get("integer", False)))


def _family(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    if _is_real(value):
        return "number"
    raise TypeError(
        f"choice values must be str, bool or real numbers, got {value!r}")


@dataclass(frozen=True)
class Choices:
    """A parameter that takes one of a finite list of ``values``.

    The values share one type family: all real numbers, all ``str`` or all
    ``bool``. ``nominal`` (default: the first value) is the design value the
    corner sampler starts from. The random samplers pick every value with
    equal probability. ``unit`` is display metadata, inferred from the name's
    suffix when omitted.
    """

    name: str
    values: Tuple[Value, ...]
    nominal: Optional[Value] = None
    unit: Optional[str] = None

    def __post_init__(self):
        name = _check_name(self.name)
        if isinstance(self.values, (str, bytes)) or not hasattr(
                self.values, "__iter__"):
            raise TypeError(f"values of {name!r} must be a sequence of values")
        values = list(self.values)
        if not values:
            raise ValueError(f"{name!r} needs at least one value")
        families = {_family(v) for v in values}
        if len(families) != 1:
            raise TypeError(
                f"values of {name!r} mix types {sorted(families)}; use one of "
                "numbers, str or bool")
        family = families.pop()
        if family == "number":
            values = [v if isinstance(v, int) else _finite_float(
                v, f"value of {name!r}") for v in values]
            values = [int(v) if isinstance(v, numbers.Integral) else v
                      for v in values]
        seen = []
        for v in values:
            if any(_same(v, s) for s in seen):
                raise ValueError(f"values of {name!r} repeat {v!r}")
            seen.append(v)
        object.__setattr__(self, "values", tuple(values))
        if self.nominal is None:
            nominal = values[0]
        else:
            nominal = self.normalize(self.nominal)
        object.__setattr__(self, "nominal", nominal)
        object.__setattr__(self, "unit", _resolve_unit(name, self.unit))

    @property
    def kind(self) -> str:
        """``"choices"``."""
        return "choices"

    @property
    def family(self) -> str:
        """``"number"``, ``"str"`` or ``"bool"``."""
        return _family(self.values[0])

    def normalize(self, value: object) -> Value:
        """The stored value equal to ``value``; refused when ``value`` is not
        one of :attr:`values`. A number matches a number of equal value
        (``1`` matches ``1.0``); a ``bool`` never matches a number."""
        for v in self.values:
            if _same(value, v):
                return v
        raise ValueError(
            f"value of {self.name!r} ({value!r}) is not one of {list(self.values)}")

    def from_unit(self, u: float) -> Value:
        """Map ``u`` in ``[0, 1)`` to one of :attr:`values`, each with equal
        probability."""
        u = _unit(u)
        k = len(self.values)
        return self.values[min(int(u * k), k - 1)]

    def to_dict(self) -> dict:
        """JSON-safe description; :meth:`from_dict` reverses it."""
        return {"kind": "choices", "name": self.name,
                "values": list(self.values), "nominal": self.nominal,
                "unit": self.unit}

    @classmethod
    def from_dict(cls, data: dict) -> "Choices":
        """The :class:`Choices` a :meth:`to_dict` description records."""
        if data.get("kind") != "choices":
            raise ValueError(f"not a Choices description: {data!r}")
        return cls(data["name"], tuple(data["values"]),
                   nominal=data.get("nominal"), unit=data.get("unit"))


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if _is_real(a) and _is_real(b):
        return float(a) == float(b)
    return False


Parameter = Union[Span, Choices]


def parameter_from_dict(data: dict) -> Parameter:
    """A :class:`Span` or :class:`Choices` from its ``to_dict`` form."""
    kind = data.get("kind") if isinstance(data, dict) else None
    if kind == "span":
        return Span.from_dict(data)
    if kind == "choices":
        return Choices.from_dict(data)
    raise ValueError(f"not a parameter description: {data!r}")
