"""Finite, explicitly supplied graph/selection scenarios for one transport question.

Each scenario is a fixed ADMG over the same named variables with its own selection
targets (and optionally a declared weight). Every scenario is identified and bound
to evidence independently; the report keeps every scenario whatever its status,
gives a structural envelope over the identified ones, and, only for declared
weights, a report that never renormalizes over survivors. This is not
equivalence-class (CPDAG/PAG) transport, and no inferential statement across
scenarios is licensed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._native import (
    consume_transport_scenarios_artifact as _consume_transport_scenarios_artifact,
)
from .._native import prepare_transport_scenarios_stage as _prepare_transport_scenarios_stage
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..graph import Admg, Cpdag, Pag
from ._impl import EvidenceCatalog, _non_negative, _optional_non_negative


@dataclass(frozen=True, slots=True)
class TransportScenario:
    """One scenario: a named fixed graph and the mechanisms that may differ."""

    name: str
    graph: Admg
    selections: Sequence[str] = ()
    weight: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CausalValueError("scenario name must be non-empty")
        if isinstance(self.graph, Cpdag | Pag):
            raise CausalUnsupportedError(
                "scenarios.equivalence_class_input: supply an explicit list of fixed "
                "graphs; CPDAG/PAG-native transport is not licensed",
                reason_code="route_not_supported",
            )
        if not isinstance(self.graph, Admg):
            raise CausalTypeError("scenario graph must be an Admg")
        selections = tuple(self.selections)
        if len(set(selections)) != len(selections):
            raise CausalValueError("scenario selections must be distinct")
        object.__setattr__(self, "selections", selections)
        if self.weight is not None and (not math.isfinite(self.weight) or self.weight < 0):
            raise CausalValueError("scenario weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TransportScenarioSet:
    """One to 64 scenarios for one question. Weights are declared for all or none,
    sum to at most one, and the remainder is kept as residual mass."""

    scenarios: Sequence[TransportScenario]

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        if not scenarios or len(scenarios) > 64:
            raise CausalValueError("a scenario set takes one to 64 scenarios")
        if any(not isinstance(s, TransportScenario) for s in scenarios):
            raise CausalTypeError("scenarios must be TransportScenario values")
        names = [s.name for s in scenarios]
        if len(set(names)) != len(names):
            raise CausalValueError("scenario names must be unique")
        weighted = {s.weight is not None for s in scenarios}
        if len(weighted) != 1:
            raise CausalValueError("declare weights for every scenario or none")
        if True in weighted and sum(s.weight or 0.0 for s in scenarios) > 1.0 + 1e-12:
            raise CausalValueError("declared scenario weights must sum to at most one")
        object.__setattr__(self, "scenarios", scenarios)


def prepare_transport_scenarios(
    scenarios: TransportScenarioSet,
    *,
    outcomes: Sequence[str],
    treatments: Sequence[str],
    source: str,
    target: str,
    catalog: EvidenceCatalog,
    laws: Any,
    at: Mapping[str, float],
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_scenarios: int = 64,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> Any:
    """Decide every scenario once and compile each identified one.

    The returned stage's ``estimate()`` reports every scenario with its status
    (``identified``, ``structurally_unidentified``, ``missing_evidence``,
    ``not_certified``, ``unsupported_provider``, ``support_failure`` or
    ``unevaluated``), counts and masses per status, the structural envelope over
    the identified scenarios, and a declared-weight report when weights were
    given. ``max_scenarios`` bounds how many scenarios are decided; the rest are
    reported unevaluated with a receipt. ``aggregate_interval()`` always refuses.
    """
    if not isinstance(scenarios, TransportScenarioSet):
        raise CausalTypeError("scenarios must be a TransportScenarioSet")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    return _prepare_transport_scenarios_stage(
        [(s.name, s.graph, list(s.selections), s.weight) for s in scenarios.scenarios],
        list(outcomes),
        list(treatments),
        source,
        target,
        catalog,
        laws,
        dict(at),
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        max_scenarios=_non_negative("max_scenarios", max_scenarios),
        max_operations=_non_negative("max_operations", max_operations),
        max_evaluation_depth=_non_negative("max_evaluation_depth", max_evaluation_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def consume_transport_scenarios_artifact(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_scenarios: int = 64,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    max_laws: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> str:
    """Independently replay an exported scenario report.

    Every scenario is re-decided under the producer's recorded limits (refused if
    they exceed the consumer's), recompiled and re-evaluated, and the report must
    match exactly, including failed and unevaluated scenarios.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_transport_scenarios_artifact(
        artifact,
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        max_scenarios=_non_negative("max_scenarios", max_scenarios),
        max_operations=_non_negative("max_operations", max_operations),
        max_evaluation_depth=_non_negative("max_evaluation_depth", max_evaluation_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        max_laws=_non_negative("max_laws", max_laws),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


__all__ = [
    "TransportScenario",
    "TransportScenarioSet",
    "consume_transport_scenarios_artifact",
    "prepare_transport_scenarios",
]
