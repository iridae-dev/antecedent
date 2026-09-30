"""Finite, explicitly supplied graph/selection scenarios for one transport question.

Each scenario is a fixed ADMG over the same named variables with its own selection
targets (and optionally a declared weight). The set declares one shared coordinate
schema (variable names, domains and cardinalities, units); a scenario may restate
its own, and any disagreement refuses with ``schema_mismatch``. Every scenario is
identified and bound to evidence independently; the report keeps every scenario
whatever its status, gives a structural envelope over the identified ones, and,
only for declared weights, a report that never renormalizes over survivors. This
is not equivalence-class (CPDAG/PAG) transport, and no inferential statement
across scenarios is licensed.
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
from ._impl import EvidenceCatalog, VariableCoordinate, _non_negative, _optional_non_negative

MAX_SCENARIOS = 64


def _coordinates(coordinates: Sequence[VariableCoordinate]) -> tuple[VariableCoordinate, ...]:
    coordinates = tuple(coordinates)
    if any(not isinstance(c, VariableCoordinate) for c in coordinates):
        raise CausalTypeError("coordinates must be VariableCoordinate values")
    return coordinates


def _wire(
    coordinates: Sequence[VariableCoordinate],
) -> list[tuple[str, str, int | None, str | None]]:
    return [(c.name, c.domain, c.cardinality, c.unit) for c in coordinates]


@dataclass(frozen=True, slots=True)
class TransportScenario:
    """One scenario: a named fixed graph and the mechanisms that may differ.

    ``coordinates``, when given, restates this scenario's coordinate schema; it
    must agree with the set's.
    """

    name: str
    graph: Admg
    selections: Sequence[str] = ()
    weight: float | None = None
    coordinates: Sequence[VariableCoordinate] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CausalValueError(
                "scenarios.duplicate_or_empty_name: scenario name must be non-empty",
                reason_code="invalid_argument",
            )
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
            raise CausalValueError(
                "scenarios.invalid_weights: scenario weight must be finite and non-negative",
                reason_code="invalid_argument",
            )
        if self.coordinates is not None:
            object.__setattr__(self, "coordinates", _coordinates(self.coordinates))


@dataclass(frozen=True, slots=True)
class TransportScenarioSet:
    """One to 64 scenarios for one question over one shared coordinate schema.

    ``coordinates`` declares every variable of the scenario graphs once: its name,
    domain (with cardinality for a categorical domain) and unit. Weights are
    declared for all scenarios or none, sum to at most one, and the remainder is
    kept as residual mass.
    """

    scenarios: Sequence[TransportScenario]
    coordinates: Sequence[VariableCoordinate]

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise CausalValueError(
                "scenarios.empty: a scenario set needs at least one scenario",
                reason_code="invalid_argument",
            )
        if len(scenarios) > MAX_SCENARIOS:
            raise CausalUnsupportedError(
                f"scenarios.count: {len(scenarios)} scenarios exceed the bound of {MAX_SCENARIOS}",
                reason_code="route_not_supported",
            )
        if any(not isinstance(s, TransportScenario) for s in scenarios):
            raise CausalTypeError("scenarios must be TransportScenario values")
        names = [s.name for s in scenarios]
        if len(set(names)) != len(names):
            raise CausalValueError(
                "scenarios.duplicate_or_empty_name: scenario names must be unique",
                reason_code="invalid_argument",
            )
        weighted = {s.weight is not None for s in scenarios}
        if len(weighted) != 1 or (
            True in weighted and sum(s.weight or 0.0 for s in scenarios) > 1.0 + 1e-12
        ):
            raise CausalValueError(
                "scenarios.invalid_weights: declare weights for every scenario or none, "
                "summing to at most one",
                reason_code="invalid_argument",
            )
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "coordinates", _coordinates(self.coordinates))


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
    given. ``max_steps`` and ``max_depth`` are one search budget shared by the
    whole set: each scenario entered costs one operation at depth one, and each
    scenario's search and proof replay charge the same budget, together with
    live bytes against ``memory_bytes`` and observing ``cancel``. When it stops,
    the scenario being decided and every later one are reported
    ``unevaluated`` (detail ``scenarios.unevaluated_budget: search.<stop>``)
    with one receipt.

    ``laws`` are supplied exact laws (``ExactTransportData`` or a sequence), or
    ``StatisticalTransportData`` whose samples are fitted once by the empirical
    plug-in: per-scenario points only, never an interval. Laws, samples and
    ``at`` are checked against the shared coordinate schema (``schema_mismatch``).
    ``aggregate_interval()`` always refuses.
    """
    if not isinstance(scenarios, TransportScenarioSet):
        raise CausalTypeError("scenarios must be a TransportScenarioSet")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    return _prepare_transport_scenarios_stage(
        [
            (
                s.name,
                s.graph,
                list(s.selections),
                s.weight,
                None if s.coordinates is None else _wire(s.coordinates),
            )
            for s in scenarios.scenarios
        ],
        _wire(scenarios.coordinates),
        list(outcomes),
        list(treatments),
        source,
        target,
        catalog,
        laws,
        dict(at),
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
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
    match exactly, including failed and unevaluated scenarios. The variable names
    the report is read with are bound into the artifact's identity.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_transport_scenarios_artifact(
        artifact,
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
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
