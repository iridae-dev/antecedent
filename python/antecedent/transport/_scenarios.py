"""Finite, explicitly supplied graph/selection scenarios for one transport question.

Each scenario is a fixed ADMG over the same named variables with its own selection
targets (and optionally a declared weight). The set declares one shared coordinate
schema (variable names, domains and cardinalities, units); a scenario may restate
its own, and any disagreement refuses with ``schema_mismatch``. Every scenario is
identified and bound to evidence independently; the report keeps every scenario
whatever its status, gives a structural envelope over the identified ones, and,
only for declared weights, a report that never renormalizes over survivors. This
is not equivalence-class (CPDAG/PAG) transport, and no inferential statement
across scenarios is supported.

With ``conditioned_on`` the question is the conditional ``P*(y | do(x), w)`` of
the bounded ADMG conditional transport row: every scenario is decided by that
route on the same shared budget and enters the same envelope. A scenario it does
not certify is ``not_certified`` (its detail names an inspection-only candidate,
never a proof); it is ``structurally_unidentified`` only when the route's witness
stage proves it with an exactly verified two-model witness.

:func:`compile_transport_scenarios` decides the set the same way and binds it into the
compiled ``TransportScenarioSet`` program: :meth:`CompiledTransportScenarios.execute` evaluates
the retained plans through the program and reports them exactly as the prepared stage's
``estimate()`` does, with the program's route and identity.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .. import _retained
from .._artifact_protocol import artifact_consumer, container_bytes
from .._columns import coerce_fields
from .._native import compile_transport_scenarios_stage as _compile_transport_scenarios_stage
from .._native import (
    consume_transport_scenarios_artifact as _consume_transport_scenarios_artifact,
)
from .._native import prepare_transport_scenarios_stage as _prepare_transport_scenarios_stage
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..execution import Execution
from ..graph import Admg, Cpdag, Pag
from ._consumption import TransportConsumption
from ._controls import call_native
from ._impl import EvidenceCatalog, VariableCoordinate, _non_negative
from ._temporal import _wire

MAX_SCENARIOS = 64


def _coordinates(coordinates: Sequence[VariableCoordinate]) -> tuple[VariableCoordinate, ...]:
    coordinates = tuple(coordinates)
    if any(not isinstance(c, VariableCoordinate) for c in coordinates):
        raise CausalTypeError("coordinates must be VariableCoordinate values")
    return coordinates


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
                "graphs; CPDAG/PAG-native transport is not supported",
                reason_code="route_not_supported",
            )
        if not isinstance(self.graph, Admg):
            raise CausalTypeError("scenario graph must be an Admg")
        selections = tuple(self.selections)
        if len(set(selections)) != len(selections):
            raise CausalValueError("scenario selections must be distinct")
        coerce_fields(self, selections=selections)
        if self.weight is not None and (not math.isfinite(self.weight) or self.weight < 0):
            raise CausalValueError(
                "scenarios.invalid_weights: scenario weight must be finite and non-negative",
                reason_code="invalid_argument",
            )
        if self.coordinates is not None:
            coerce_fields(self, coordinates=_coordinates(self.coordinates))


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
        coerce_fields(self, scenarios=scenarios)
        coerce_fields(self, coordinates=_coordinates(self.coordinates))


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
    conditioned_on: Sequence[str] = (),
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    execution: Execution | None = None,
) -> Any:
    """Decide every scenario once and compile each identified one.

    The returned stage's ``estimate()`` reports every scenario with its status
    (``identified``, ``structurally_unidentified``, ``missing_evidence``,
    ``not_certified``, ``unsupported_provider``, ``support_failure`` or
    ``unevaluated``), counts and masses per status, the structural envelope over
    the identified scenarios, and a declared-weight report when weights were
    given. ``max_steps`` and ``max_depth`` are one search budget shared by the
    whole set: each scenario entered costs one operation at depth one, and each
    scenario's search, s-hedge check and proof replay charge the same budget,
    together with live bytes against ``memory_bytes`` and observing ``cancel``.
    Memory is cumulative: a decided scenario keeps holding its engine's peak
    live state, which every later charge sits on top of. When the budget stops,
    the scenario being decided and every later one are reported
    ``unevaluated`` (detail ``scenarios.unevaluated_budget: search.<stop>``)
    with one receipt whose ``explored`` (scenarios fully decided) and
    ``unevaluated`` lists are disjoint.

    ``laws`` are supplied exact laws (``ExactTransportData`` or a sequence), or
    ``StatisticalTransportData`` whose samples are fitted once by the empirical
    plug-in: per-scenario points only (the bootstrap is fixed at zero), never an
    interval; the tables are fitted once for the whole set and each scenario
    compiles against them. Laws, sample rows and interventions, and ``at`` are
    checked against the shared coordinate schema (``schema_mismatch``): a value
    must be a finite number inside its declared domain (a non-finite value
    belongs to no domain, even an unspecified one). Units are compared between
    the schema declarations and the catalog's environments; laws, samples and
    ``at`` carry no units, so a value's unit is never checked. A report
    truncated by an operation, depth or memory bound exports (the limits are
    recorded and the consumer replays the same prefix); one truncated by
    cancellation does not export, because no consumer could reproduce it.
    ``aggregate_interval()`` always refuses.

    A non-empty ``conditioned_on`` asks ``P*(outcomes | do(treatments),
    conditioned_on)`` in every scenario through the bounded ADMG conditional
    route (at most 6 variables, 3 treatments and 3 conditioned variables;
    ``route_not_supported`` / ``admg_transport.bounds_exceeded`` otherwise).
    ``at`` then binds exactly the treatments and the conditioned variables; an
    identified scenario's point is the reduced joint normalized at the
    conditioning levels, and a zero-mass conditioning event is that scenario's
    ``support_failure``. A reduced joint with an s-hedge is ``not_certified``
    with an inspection-only candidate, and ``structurally_unidentified`` only
    when the route's witness stage finds an exactly verified two-model witness.
    Exact laws only: ``StatisticalTransportData`` or counted laws refuse with
    ``not_supported``. Its artifact is format version 2; a classical
    question still exports version 1.
    """
    if not isinstance(scenarios, TransportScenarioSet):
        raise CausalTypeError("scenarios must be a TransportScenarioSet")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    if isinstance(conditioned_on, str):
        raise CausalTypeError("conditioned_on must be a sequence of variable names")
    return call_native(
        execution,
        _prepare_transport_scenarios_stage,
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
        conditioned_on=list(conditioned_on),
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        max_operations=_non_negative("max_operations", max_operations),
        max_evaluation_depth=_non_negative("max_evaluation_depth", max_evaluation_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        cancel="cancel",
        memory="memory_bytes",
    )


class CompiledTransportScenarios:
    """A decided scenario set compiled into its program family.

    ``route`` is the plan route (``TransportScenarioSet``) and ``program_identity`` the identity
    of the lowered program; nothing is evaluated until :meth:`execute` runs. The report is
    per-scenario points: no inferential statement across scenarios is made
    (:meth:`aggregate_interval` always refuses).
    """

    __slots__ = ("_execution", "_native")

    def __init__(self, native: Any, execution: Execution | None) -> None:
        self._native = native
        self._execution = execution

    def __repr__(self) -> str:
        return (
            f"CompiledTransportScenarios(route={self.route!r}, "
            f"identity={self.program_identity[:12]!r})"
        )

    @property
    def route(self) -> str:
        """The plan route of the compiled program."""
        route: str = self._native.route
        return route

    @property
    def program_identity(self) -> str:
        """Identity of the lowered program, as lowercase hex."""
        identity: str = self._native.program_identity
        return identity

    def execute(self) -> dict[str, Any]:
        """Evaluate every retained plan and report every scenario.

        The report is the one ``prepare_transport_scenarios(...).estimate()`` returns for the
        same decided set and laws: every scenario with its status, the counts and masses per
        status, the structural envelope over the identified scenarios and, for declared
        weights, the report that never renormalizes.
        """
        payload = call_native(
            self._execution, self._native.execute, cancel="cancel", memory="memory_bytes"
        )
        report: dict[str, Any] = json.loads(payload)
        return report

    def refresh(self, laws: Any) -> CompiledTransportScenarios:
        """Compile the program again over new laws or samples; the decisions are kept."""
        native = call_native(
            self._execution, self._native.refresh, laws, cancel="cancel", memory="memory_bytes"
        )
        return CompiledTransportScenarios(native, self._execution)

    def plan_summary(self) -> list[tuple[str, str]]:
        """Each scenario's retained plan: ``compiled``, a refusal, or ``not_identified:<status>``."""
        return [(str(name), str(plan)) for name, plan in self._native.plan_summary()]

    def aggregate_interval(self) -> None:
        """Inference across scenarios sharing data is not supported; always refuses."""
        self._native.aggregate_interval()

    def export(self) -> bytes:
        """The last executed report as a framed artifact for ``consume_transport_scenarios_artifact``."""
        return bytes(self._native.export())


def compile_transport_scenarios(
    scenarios: TransportScenarioSet,
    *,
    outcomes: Sequence[str],
    treatments: Sequence[str],
    source: str,
    target: str,
    catalog: EvidenceCatalog,
    laws: Any,
    at: Mapping[str, float],
    conditioned_on: Sequence[str] = (),
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    execution: Execution | None = None,
) -> CompiledTransportScenarios:
    """Decide every scenario once and compile the decided set into its program.

    The arguments, the decisions, the shared budget and every refusal are those of
    :func:`prepare_transport_scenarios`; the decided set, its retained plans and the laws'
    identities are then bound into the ``TransportScenarioSet`` program.
    """
    stage = prepare_transport_scenarios(
        scenarios,
        outcomes=outcomes,
        treatments=treatments,
        source=source,
        target=target,
        catalog=catalog,
        laws=laws,
        at=at,
        conditioned_on=conditioned_on,
        max_steps=max_steps,
        max_depth=max_depth,
        max_operations=max_operations,
        max_evaluation_depth=max_evaluation_depth,
        max_support_rows=max_support_rows,
        execution=execution,
    )
    return CompiledTransportScenarios(_compile_transport_scenarios_stage(stage), execution)


@artifact_consumer(
    proof="TransportScenarioArtifactWire::consume_with_limits: redecides every scenario and compares the complete retained report",
    replayable=True,
    identity=(),
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
    execution: Execution | None = None,
    retained_identity: str | None = None,
) -> TransportConsumption:
    """Independently replay an exported scenario report.

    Every scenario is re-decided under the producer's recorded limits (refused if
    they exceed the consumer's), recompiled and re-evaluated, and the report must
    match exactly, including failed and unevaluated scenarios. The variable names
    the report is read with are bound into the artifact's scientific identity;
    the catalog, laws, provider and sample summaries are bound by a separate
    data-identity digest, so a changed snapshot label is refused as a data
    identity mismatch.
    """
    _retained.admit(
        artifact,
        retained_identity,
        "antecedent.transport._scenarios.consume_transport_scenarios_artifact",
        "antecedent.transport._scenarios.consume_transport_scenarios_artifact(..., retained_identity=antecedent.artifacts.artifact_identity(retained_bytes))",
    )
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return TransportConsumption.from_checked_json(
        "consume_transport_scenarios_artifact",
        call_native(
            execution,
            _consume_transport_scenarios_artifact,
            artifact,
            max_steps=_non_negative("max_steps", max_steps),
            max_depth=_non_negative("max_depth", max_depth),
            max_operations=_non_negative("max_operations", max_operations),
            max_evaluation_depth=_non_negative("max_evaluation_depth", max_evaluation_depth),
            max_support_rows=_non_negative("max_support_rows", max_support_rows),
            max_laws=_non_negative("max_laws", max_laws),
            cancel="cancel",
            memory="memory_bytes",
        ),
    )


__all__ = [
    "CompiledTransportScenarios",
    "TransportScenario",
    "TransportScenarioSet",
    "compile_transport_scenarios",
    "consume_transport_scenarios_artifact",
    "prepare_transport_scenarios",
]
