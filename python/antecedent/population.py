"""Named predicates and custom target-distribution weights for analyze()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


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


def target_all() -> dict[str, str]:
    return {"kind": "all"}


def target_treated() -> dict[str, str]:
    return {"kind": "treated"}


def target_untreated() -> dict[str, str]:
    return {"kind": "untreated"}


def target_named(name: str) -> dict[str, str]:
    return {"kind": "named", "name": str(name)}


def target_rows(rows: Sequence[int]) -> dict[str, object]:
    return {"kind": "rows", "rows": [int(r) for r in rows]}


def target_custom_distribution(distribution_id: int) -> dict[str, object]:
    return {"kind": "custom_distribution", "id": int(distribution_id)}


def coerce_target_population(spec) -> dict[str, object] | None:
    """Normalize AverageEffect.target_population / analyze kwargs to a wire dict."""
    if spec is None:
        return None
    if isinstance(spec, str):
        key = spec.strip().lower().replace("-", "_")
        if key in {"all", "all_observed", "observed"}:
            return target_all()
        if key == "treated":
            return target_treated()
        if key in {"untreated", "control"}:
            return target_untreated()
        raise ValueError(
            f"unknown target_population string {spec!r}; "
            "use all|treated|untreated or a target_* helper"
        )
    if isinstance(spec, Mapping):
        kind = str(spec.get("kind", "")).lower()
        if kind in {"all", "all_observed"}:
            return target_all()
        if kind == "treated":
            return target_treated()
        if kind == "untreated":
            return target_untreated()
        if kind == "named":
            return target_named(str(spec["name"]))
        if kind == "rows":
            return target_rows(spec["rows"])  # type: ignore[arg-type]
        if kind in {"custom_distribution", "custom"}:
            return target_custom_distribution(int(spec["id"]))  # type: ignore[arg-type]
        raise ValueError(f"unknown target_population mapping {spec!r}")
    raise TypeError(f"unsupported target_population type: {type(spec)!r}")


def registry_wire(registry: PopulationRegistry | None) -> tuple[dict[str, list[int]], dict[int, list[float]]]:
    if registry is None:
        return {}, {}
    return dict(registry.predicates), dict(registry.distributions)


__all__ = [
    "PopulationRegistry",
    "coerce_target_population",
    "registry_wire",
    "target_all",
    "target_custom_distribution",
    "target_named",
    "target_rows",
    "target_treated",
    "target_untreated",
]
