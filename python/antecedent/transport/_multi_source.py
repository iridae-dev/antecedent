"""Bounded multi-source limited-experiment transport (``TR^mz``).

Several source populations, each able to experiment only on its own declared
controllable variables, may together identify a target effect none of them
identifies alone. The search is sound and incomplete within declared bounds:
two to four sources, twelve observed variables, four controllable variables per
source and 64 candidate regimes. A factor is always taken from one source; a
joint over several sources' interventions is never fabricated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .._native import (
    consume_multi_source_z_transport_artifact as _consume_multi_source_z_transport_artifact,
)
from .._native import (
    identify_multi_source_z_transport_stage as _identify_multi_source_z_transport_stage,
)
from ..errors import CausalTypeError, CausalValueError
from ..graph import Admg
from ._impl import EvidenceCatalog, _non_negative, _optional_non_negative


def _names(label: str, values: Sequence[str], *, allow_empty: bool = False) -> tuple[str, ...]:
    names = tuple(values)
    if (not names and not allow_empty) or len(set(names)) != len(names):
        raise CausalValueError(f"{label} must be distinct variable names")
    if any(not isinstance(v, str) or not v.strip() for v in names):
        raise CausalValueError(f"{label} must contain variable names")
    return names


@dataclass(frozen=True, slots=True)
class ZTransportSource:
    """One source population of a multi-source query.

    ``controllable`` is what the source could experiment on (its theoretical
    availability); the regimes it actually supplied belong in the catalog.
    ``selections`` are the mechanisms that may differ from the target.
    ``experiment_assignment`` fixes the level of an exchanged coordinate the
    formula proves irrelevant.
    """

    population: str
    controllable: Sequence[str]
    selections: Sequence[str] = ()
    experiment_assignment: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.population, str) or not self.population.strip():
            raise CausalValueError("source population must be a non-empty name")
        object.__setattr__(self, "controllable", _names("controllable", self.controllable))
        object.__setattr__(
            self, "selections", _names("selections", self.selections, allow_empty=True)
        )
        if not set(self.experiment_assignment).issubset(self.controllable):
            raise CausalValueError("experiment assignments must name controllable variables")
        if any(not math.isfinite(value) for value in self.experiment_assignment.values()):
            raise CausalValueError("experiment assignments must be finite")
        object.__setattr__(
            self, "experiment_assignment", MappingProxyType(dict(self.experiment_assignment))
        )


@dataclass(frozen=True, slots=True)
class MultiSourceZTransportQuery:
    """Target interventional query answered from two to four limited-experiment sources.

    The target supplies observational evidence only. Source order never changes
    the result.
    """

    target: str
    outcomes: Sequence[str]
    treatments: Sequence[str]
    sources: Sequence[ZTransportSource]

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
        if not 2 <= len(sources) <= 4:
            raise CausalValueError(
                "a multi-source query takes two to four sources; use the z-transport route for one"
            )
        populations = [source.population for source in sources]
        if len(set(populations)) != len(populations) or self.target in populations:
            raise CausalValueError("source populations must be distinct and differ from the target")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "treatments", treatments)
        object.__setattr__(self, "sources", sources)


def identify_multi_source_z_transport(
    *,
    graph: Admg,
    query: MultiSourceZTransportQuery,
    catalog: EvidenceCatalog,
    max_operations: int = 4096,
    max_depth: int = 24,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> Any:
    """Decide a bounded multi-source query against one catalog of target and source evidence.

    The returned stage's ``outcome`` is ``identified``, ``proven_non_transportable``,
    ``missing_evidence``, ``not_certified`` or ``exhausted``, and ``decision()``
    explains it: the identifying route and cited regimes, the failing c-component
    and each source's premises, the missing factor, the explored stages, or the
    limits receipt. Only ``proven_non_transportable`` is an impossibility claim.
    An identified stage offers ``prepare_exact`` and ``prepare_empirical``; the
    decision is never repeated at estimation.
    """
    if not isinstance(graph, Admg):
        raise CausalTypeError("identify_multi_source_z_transport requires graph=Admg(...)")
    if not isinstance(query, MultiSourceZTransportQuery):
        raise CausalTypeError("query must be a MultiSourceZTransportQuery")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    return _identify_multi_source_z_transport_stage(
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


def consume_multi_source_z_transport_artifact(
    artifact: bytes,
    *,
    max_search_operations: int = 4096,
    max_search_depth: int = 24,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> str:
    """Independently verify an exported multi-source result and recompute its point.

    The consumer re-runs the bounded decision on the stored graph, query and
    catalog under its own search limits, re-binds every factor to its source,
    recomputes the point and rechecks the interval bookkeeping. Nothing the
    artifact recorded raises the consumer's limits.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_multi_source_z_transport_artifact(
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
    "MultiSourceZTransportQuery",
    "ZTransportSource",
    "consume_multi_source_z_transport_artifact",
    "identify_multi_source_z_transport",
]
