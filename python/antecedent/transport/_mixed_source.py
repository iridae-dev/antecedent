"""Bounded mixed-source proof search over the target population's studies.

Several studies of one population, each measuring a different set of variables
or experimenting on different ones, may together identify an interventional
query that no single study, and no named theorem route, identifies. The
theorem-scoped routes (target-first sID, then the declared z or mz route, then classical
meta-transport when two or more declared sources can each experiment on every
variable) run first under one shared budget; only when none of them identifies the query does
the generic search run, over a frozen rule set (``x9.rules.v1``): marginalize,
condition, product, and Pearl's three do-calculus rules. Every step records its
premises and the study distribution it uses; an independent checker replays each
one. The search is sound and incomplete: running out of rules or budget is
``not_certified`` or ``exhausted``, never a non-identification claim. A joint
that the catalog holds only as separate marginals is never a leaf; the decision
names the exact leaf that would need it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .._native import consume_mixed_source_artifact as _consume_mixed_source_artifact
from .._native import identify_mixed_source_transport_stage as _identify_mixed_source_stage
from ..errors import CausalTypeError, CausalValueError
from ..graph import Admg
from ._impl import EvidenceCatalog, _non_negative, _optional_non_negative
from ._multi_source import ZTransportSource, _names


@dataclass(frozen=True, slots=True)
class MixedSourceQuery:
    """Interventional query ``P(outcomes | do(treatments))`` in the target population.

    The inputs are the target population's available, measured, whole-population
    regimes in the catalog. ``sources`` are optional: they feed only the
    theorem-scoped z (one source), mz (two to four) and meta (two or more
    unrestricted sources) routes that run first.
    """

    target: str
    outcomes: Sequence[str]
    treatments: Sequence[str]
    sources: Sequence[ZTransportSource] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise CausalValueError("target must be a non-empty population name")
        outcomes = _names("outcomes", self.outcomes)
        treatments = _names("treatments", self.treatments)
        if set(outcomes) & set(treatments):
            raise CausalValueError("outcomes and treatments must not overlap")
        sources = tuple(self.sources)
        if any(not isinstance(source, ZTransportSource) for source in sources):
            raise CausalTypeError("sources must be ZTransportSource values")
        if len(sources) > 4:
            raise CausalValueError("a mixed-source query declares at most four sources")
        populations = [source.population for source in sources]
        if len(set(populations)) != len(populations) or self.target in populations:
            raise CausalValueError("source populations must be distinct and differ from the target")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "treatments", treatments)
        object.__setattr__(self, "sources", sources)


def identify_mixed_source_transport(
    *,
    graph: Admg,
    query: MixedSourceQuery,
    catalog: EvidenceCatalog,
    max_operations: int = 20_000,
    max_depth: int = 16,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> Any:
    """Decide a bounded mixed-source query against one catalog of the target's studies.

    The returned stage's ``outcome`` is ``identified``, ``named_route`` (a
    theorem-scoped route identifies it: use that route), ``missing_evidence``
    (the exact joint the catalog holds only as separate marginals),
    ``not_certified`` or ``exhausted``, and ``decision()`` explains it: the
    checked proof steps with their premises and source distributions, a compact
    proof graph, alternative derivations when actually found, the missing leaf,
    the explored frontier, or the limits receipt. Nothing here is a
    non-identification claim. An identified stage offers ``prepare_exact``; the
    decision is never repeated at estimation. Estimation returns exact-law
    points only: ``interval["available"]`` is always ``False``, and counted laws
    (``prepare_empirical``) are refused as ``cell_not_licensed``. Preparing a
    decision that did not identify raises its typed reason
    (``transport_missing_evidence``, ``transport_not_certified``,
    ``transport_budget_cancel`` or ``route_not_supported``).
    """
    if not isinstance(graph, Admg):
        raise CausalTypeError("identify_mixed_source_transport requires graph=Admg(...)")
    if not isinstance(query, MixedSourceQuery):
        raise CausalTypeError("query must be a MixedSourceQuery")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    return _identify_mixed_source_stage(
        graph,
        query.target,
        list(query.outcomes),
        list(query.treatments),
        [
            (
                source.population,
                list(source.controllable),
                dict(source.experiment_assignment),
                list(source.selections),
            )
            for source in query.sources
        ],
        catalog,
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def consume_mixed_source_artifact(
    artifact: bytes,
    *,
    max_search_operations: int = 20_000,
    max_search_depth: int = 16,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> str:
    """Independently verify an exported mixed-source result and recompute its point.

    The consumer re-runs the bounded decision on the stored graph, query and
    catalog under its own search limits, accepts only the identical derivation
    (frozen rule-set version, every step with its premises and source
    distributions), replays every step with the independent proof checker,
    re-binds every source-named leaf and recomputes every point bit for bit. The
    limits, requests and variable names are bound by the premises digest and the
    snapshot ids by the data-identity digest; relabelled names or any other edit
    is refused. Nothing the artifact recorded raises the consumer's limits.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_mixed_source_artifact(
        artifact,
        max_search_operations=_non_negative("max_search_operations", max_search_operations),
        max_search_depth=_non_negative("max_search_depth", max_search_depth),
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        max_support_rows=_optional_non_negative("max_support_rows", max_support_rows),
        max_laws=_optional_non_negative("max_laws", max_laws),
        max_law_cells=_optional_non_negative("max_law_cells", max_law_cells),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


__all__ = [
    "MixedSourceQuery",
    "consume_mixed_source_artifact",
    "identify_mixed_source_transport",
]
