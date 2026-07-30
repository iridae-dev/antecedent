"""Named predicates and custom target-distribution weights for analyze()."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class PopulationRegistry:
    """Caller bindings for named row predicates and custom target weights.

    Stratification estimators reject ``CustomDistribution``; use IPW/matching
    (``estimator=\"propensity.weighting\"``) for weighted target populations.
    """

    predicates: dict[str, list[int]] = field(default_factory=dict)
    distributions: dict[int, list[float]] = field(default_factory=dict)

    def insert_predicate(self, name: str, rows: Sequence[int]) -> None:
        self.predicates[str(name)] = [int(r) for r in rows]

    def insert_distribution(self, distribution_id: int, weights: Sequence[float]) -> None:
        self.distributions[int(distribution_id)] = [float(w) for w in weights]


class Population:
    """Base class for ``AverageEffect.target_population`` specifications.

    Subclasses are frozen, ``slots=True`` dataclasses; each implements
    ``_wire()`` returning the exact wire dict :func:`coerce_target_population`
    has always produced for that spec kind — the wire shape is frozen, only
    the caller-facing constructor changed from a bare-dict-returning function
    to a type. ``__slots__ = ()`` here so a slotted subclass doesn't pick up
    an unwanted ``__dict__`` from this base.
    """

    __slots__ = ()

    def _wire(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AllRows(Population):
    """All observed rows (``kind="all"``)."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "all"}


@dataclass(frozen=True, slots=True)
class Treated(Population):
    """Rows with observed treatment (``kind="treated"``)."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "treated"}


@dataclass(frozen=True, slots=True)
class Untreated(Population):
    """Rows with observed control (``kind="untreated"``)."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "untreated"}


@dataclass(frozen=True, slots=True)
class Named(Population):
    """A caller-registered named predicate (see ``PopulationRegistry``)."""

    name: str

    def _wire(self) -> dict[str, object]:
        return {"kind": "named", "name": str(self.name)}


@dataclass(frozen=True, slots=True)
class Rows(Population):
    """An explicit row-index set."""

    rows: tuple[int, ...]

    def _wire(self) -> dict[str, object]:
        return {"kind": "rows", "rows": [int(r) for r in self.rows]}


@dataclass(frozen=True, slots=True)
class CustomDistribution(Population):
    """A caller-registered custom weighting (see ``PopulationRegistry``)."""

    distribution_id: int

    def _wire(self) -> dict[str, object]:
        return {"kind": "custom_distribution", "id": int(self.distribution_id)}


def target_all() -> AllRows:
    return AllRows()


def target_treated() -> Treated:
    return Treated()


def target_untreated() -> Untreated:
    return Untreated()


def target_named(name: str) -> Named:
    return Named(str(name))


def target_rows(rows: Sequence[int]) -> Rows:
    return Rows(tuple(int(r) for r in rows))


def target_custom_distribution(distribution_id: int) -> CustomDistribution:
    return CustomDistribution(int(distribution_id))


def coerce_target_population(spec: object) -> dict[str, object] | None:
    """Normalize AverageEffect.target_population / analyze kwargs to a wire dict."""
    if spec is None:
        return None
    if isinstance(spec, Population):
        return spec._wire()
    if isinstance(spec, str):
        key = spec.strip().lower().replace("-", "_")
        if key in {"all", "all_observed", "observed"}:
            return target_all()._wire()
        if key == "treated":
            return target_treated()._wire()
        if key in {"untreated", "control"}:
            return target_untreated()._wire()
        raise ValueError(
            f"unknown target_population string {spec!r}; "
            "use all|treated|untreated or a target_* helper"
        )
    if isinstance(spec, Mapping):
        kind = str(spec.get("kind", "")).lower()
        if kind in {"all", "all_observed"}:
            return target_all()._wire()
        if kind == "treated":
            return target_treated()._wire()
        if kind == "untreated":
            return target_untreated()._wire()
        if kind == "named":
            return target_named(str(spec["name"]))._wire()
        if kind == "rows":
            return target_rows(spec["rows"])._wire()
        if kind in {"custom_distribution", "custom"}:
            return target_custom_distribution(int(spec["id"]))._wire()
        raise ValueError(f"unknown target_population mapping {spec!r}")
    raise TypeError(f"unsupported target_population type: {type(spec)!r}")


def registry_wire(
    registry: PopulationRegistry | None,
) -> tuple[dict[str, list[int]], dict[int, list[float]]]:
    if registry is None:
        return {}, {}
    return dict(registry.predicates), dict(registry.distributions)


__all__ = [
    "AllRows",
    "CustomDistribution",
    "Named",
    "Population",
    "PopulationRegistry",
    "Rows",
    "Treated",
    "Untreated",
    "coerce_target_population",
    "registry_wire",
    "target_all",
    "target_custom_distribution",
    "target_named",
    "target_rows",
    "target_treated",
    "target_untreated",
]
