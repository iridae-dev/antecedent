"""Exact binary observation recovery: graph-licensed recovery, not MAR/IPCW.

Some binary variables are item-missing. Each partially observed variable ``X``
has an explicit response indicator ``R`` (1 observed, 0 missing) and a proxy
``X*`` with levels ``0``, ``1`` and ``"?"`` (``X* = X`` when ``R = 1``, ``"?"``
otherwise); fully observed variables ``O`` are always measured. The m-graph (a
DAG over all of them) states how missingness arises. The full law ``P(X, O)``
is recovered from one named observed pattern law ``P(R, X*, O)`` exactly when
no response indicator depends on its own variable (no self-censoring edge
``X -> R``): each response's propensity is then a ratio of observed pattern
margins, and the recovered law is the complete-case cell divided by their
product. A self-censoring edge is proved nonrecoverable by an exactly verified
witness (two models that agree on the observed law and differ on the target).

The assumption lives in the graph and is checked: there is no complete-case
fallback and no missing-at-random or inverse-probability-weighting substitution.
Selection nodes, bidirected edges, non-binary variables and response-to-response
edges are refused as unsupported mechanisms. A downstream effect is identified
from the recovered law by ordinary target identification on the m-graph's causal
restriction. Exact laws only; counted laws are refused (``cell_not_licensed``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .._native import (
    consume_observation_recovery_artifact as _consume_observation_recovery_artifact,
)
from .._native import identify_observation_recovery_stage as _identify_observation_recovery
from ..errors import CausalTypeError, CausalValueError
from ..graph import Admg
from ._impl import EvidenceCatalog, _non_negative, _optional_non_negative


@dataclass(frozen=True, slots=True)
class PartiallyObservedVariable:
    """A partially observed binary variable with its response indicator and proxy."""

    variable: str
    response: str
    proxy: str

    def __post_init__(self) -> None:
        names = (self.variable, self.response, self.proxy)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise CausalValueError("variable, response and proxy must be non-empty names")
        if len(set(names)) != 3:
            raise CausalValueError("variable, response and proxy must be three distinct nodes")


@dataclass(frozen=True, slots=True)
class ObservationRecoveryQuery:
    """Recover ``P(X, O)`` of ``population`` from the catalog regime ``observed_regime``."""

    population: str
    observed_regime: str
    partially_observed: Sequence[PartiallyObservedVariable]
    fully_observed: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.population, str) or not self.population.strip():
            raise CausalValueError("population must be a non-empty name")
        if not isinstance(self.observed_regime, str) or not self.observed_regime.strip():
            raise CausalValueError("observed_regime must name a catalog regime")
        partially = tuple(self.partially_observed)
        if not partially or any(not isinstance(p, PartiallyObservedVariable) for p in partially):
            raise CausalTypeError("partially_observed must be PartiallyObservedVariable values")
        fully = tuple(self.fully_observed)
        if any(not isinstance(name, str) or not name.strip() for name in fully):
            raise CausalValueError("fully_observed must be non-empty names")
        object.__setattr__(self, "partially_observed", partially)
        object.__setattr__(self, "fully_observed", fully)


def identify_observation_recovery(
    *,
    graph: Admg,
    query: ObservationRecoveryQuery,
    catalog: EvidenceCatalog,
    effect_outcomes: Sequence[str] | None = None,
    effect_treatments: Sequence[str] | None = None,
    effect_graph: Admg | None = None,
    max_operations: int = 50_000,
    max_depth: int = 32,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> Any:
    """Decide recovery once; the stage's ``outcome`` is ``recovered`` or ``nonrecoverable``.

    ``decision()`` shows the checked formula (each propensity's conditioning and
    every margin bound to the named catalog distribution) or the verified
    witness. A recovered stage offers ``prepare_exact(law, requests)``; the
    decision is never repeated. When ``effect_outcomes`` and
    ``effect_treatments`` are given, the effect is identified from the recovered
    law under the same budget (``effect_graph`` must equal the m-graph's causal
    restriction; by default it is that restriction). Refusals carry
    ``reason_code`` and a ``recovery.*`` detail.
    """
    if not isinstance(graph, Admg):
        raise CausalTypeError("identify_observation_recovery requires graph=Admg(...)")
    if not isinstance(query, ObservationRecoveryQuery):
        raise CausalTypeError("query must be an ObservationRecoveryQuery")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    if effect_graph is not None and not isinstance(effect_graph, Admg):
        raise CausalTypeError("effect_graph must be an Admg")
    if (effect_outcomes is None) != (effect_treatments is None):
        raise CausalValueError("give both effect_outcomes and effect_treatments, or neither")
    return _identify_observation_recovery(
        graph,
        query.population,
        query.observed_regime,
        [(p.variable, p.response, p.proxy) for p in query.partially_observed],
        list(query.fully_observed),
        catalog,
        effect_outcomes=None if effect_outcomes is None else list(effect_outcomes),
        effect_treatments=None if effect_treatments is None else list(effect_treatments),
        effect_graph=effect_graph,
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def consume_observation_recovery_artifact(
    artifact: bytes,
    *,
    max_search_operations: int = 50_000,
    max_search_depth: int = 32,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_requests: int = 64,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> str:
    """Independently verify an exported recovery and recompute it bit for bit.

    The consumer re-decides under the producer's stored limits (refusing limits
    above its own maxima), re-checks the formula, recomputes the recovered law
    and every effect point, and refuses relabelled names or any edited premise.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_observation_recovery_artifact(
        artifact,
        max_search_operations=_non_negative("max_search_operations", max_search_operations),
        max_search_depth=_non_negative("max_search_depth", max_search_depth),
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        max_requests=_non_negative("max_requests", max_requests),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


__all__ = [
    "ObservationRecoveryQuery",
    "PartiallyObservedVariable",
    "consume_observation_recovery_artifact",
    "identify_observation_recovery",
]
