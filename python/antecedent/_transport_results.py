"""Immutable typed scientific projections with a transitional mapping interface."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


class _Record(Mapping[str, Any]):
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = _freeze(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._values)


class TransportUncertainty(_Record):
    """Pointwise sampling uncertainty; availability does not imply calibration."""

    @property
    def available(self) -> bool:
        return bool(self._values["available"])

    @property
    def reason(self) -> str | None:
        return self._values.get("reason")

    @property
    def replicate_ids(self) -> tuple[int, ...] | None:
        return self._values.get("replicate_ids")

    @property
    def mean_intervals(self) -> tuple[tuple[str, float, float], ...] | None:
        return self._values.get("mean_intervals")

    @property
    def atom_intervals(self) -> tuple[tuple[float, float], ...] | None:
        return self._values.get("atom_intervals")


class TransportContrast(_Record):
    """Paired mean contrast, with only the inference supported by its parents."""

    @property
    def estimate(self) -> float:
        return float(self._values["estimate"])

    @property
    def interval(self) -> tuple[float, float] | None:
        return self._values.get("interval")

    @property
    def reason(self) -> str | None:
        return self._values.get("reason")

    @property
    def coverage_target(self) -> float | None:
        return self._values.get("coverage_target")


class TransportSupport(_Record):
    """Located empirical support or missing-evidence failure."""

    @property
    def code(self) -> str:
        return str(self._values["code"])

    @property
    def detail(self) -> str:
        return str(self._values["detail"])

    @property
    def variables(self) -> tuple[int, ...]:
        return self._values["variables"]


class TransportGridPoint(_Record):
    """One requested coordinate, including unavailable coordinates."""

    @property
    def at(self) -> Mapping[str, float]:
        return self._values["at"]

    @property
    def status(self) -> str:
        return str(self._values["status"])

    @property
    def means(self) -> Mapping[str, float]:
        return self._values.get("means", MappingProxyType({}))

    @property
    def uncertainty(self) -> TransportUncertainty | None:
        raw = self._values.get("uncertainty")
        return TransportUncertainty(raw) if raw is not None else None

    @property
    def support_failure(self) -> TransportSupport | None:
        raw = self._values.get("factor_diagnostic")
        return TransportSupport(raw) if raw is not None else None
