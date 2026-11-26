"""Joint confidence set for a finite schedule decision across graphs and preferences."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar

from . import _native, _retained
from ._artifact_protocol import (
    ArtifactExpectation,
    ExportedArtifact,
    artifact_export,
    container_bytes,
)
from ._refusal import raise_refusal
from ._unit_support import read_unit_support
from .composition_bundle import ConsumedBundle
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .execution import Execution, native_controls, resolve_execution
from .outcomes import outcome_operation, require_success

__all__ = [
    "CompiledIntegratedScheduleDecision",
    "IntegratedScheduleBundle",
    "IntegratedScheduleDecision",
    "ScheduleCalibrationDesign",
    "ScheduleCalibrationEvidence",
    "ScheduleDecisionDeclaration",
    "ScheduleDecisionRefusal",
    "ScheduleScenarioResult",
    "ScheduleSourceSubject",
    "ScheduleStructure",
    "ScheduleUtilityDifference",
    "compile_integrated_schedule_decision",
    "compile_schedule_decision_bundle",
    "consume_schedule_decision_bundle",
]

ScheduleStructure = Literal["both_actions_causal", "first_action_only"]
ScheduleCalibrationDesign = Literal["example_unique_optima", "example_tied_optima"]


@dataclass(frozen=True)
class ScheduleSourceSubject:
    """One independent source trajectory with its endpoint absent after censoring."""

    baseline: bool
    actions: tuple[bool, bool]
    endpoint: float | None


@dataclass(frozen=True)
class ScheduleDecisionDeclaration:
    """Finite schedules, structural alternatives and preference costs fixed before sampling."""

    schedules: tuple[int, ...]
    structures: tuple[ScheduleStructure, ...]
    preference_costs: tuple[tuple[float, float, float, float], ...]
    calibration_design: ScheduleCalibrationDesign | None = None


@dataclass(frozen=True)
class ScheduleUtilityDifference:
    """Simultaneous limits on a candidate-minus-other utility difference."""

    candidate: int
    other: int
    bounds: tuple[float, float]


@dataclass(frozen=True)
class ScheduleScenarioResult:
    """The optimum confidence set under one structure and preference state."""

    structure: ScheduleStructure
    preference_state: int
    point_utilities: tuple[tuple[int, float], ...]
    compatible_optima: tuple[int, ...]
    differences: tuple[ScheduleUtilityDifference, ...]


class ScheduleDecisionRefusal(StructuredRefusal):
    """An actionable failure of declaration, support, execution, or artifact replay."""


@dataclass(frozen=True)
class ScheduleCalibrationEvidence:
    """Current empirical standing within the declared example designs."""

    status: str
    record_id: str | None
    reason: str | None
    observed: float | None
    source_rows_range: tuple[int | None, int | None]


@dataclass(frozen=True)
class IntegratedScheduleDecision:
    """Optimum set across all declared states and the scope of its joint guarantee."""

    optimum_set: tuple[int, ...]
    scenarios: tuple[ScheduleScenarioResult, ...]
    first_assignment: tuple[float, float]
    second_assignment: tuple[tuple[float, float], tuple[float, float]]
    observation_survival: tuple[tuple[float, ...], tuple[float, ...]]
    target_prevalence: float
    minimum_coverage: float
    coverage_scope: str
    calibration: ScheduleCalibrationEvidence


@dataclass(frozen=True)
class IntegratedScheduleBundle:
    """Portable decision bundle with the identity needed by an independent consumer."""

    data: bytes
    identity: str
    decision: IntegratedScheduleDecision
    plan: dict[str, Any]
    child_program_identities: tuple[str, str, str]

    @artifact_export
    def export(self) -> ExportedArtifact[ConsumedBundle]:
        """Export the bundle for generic node replay using its retained identity."""
        kind = "composition_bundle_v1"
        return ExportedArtifact(self.data, kind, ArtifactExpectation(kind, self.identity))


def _read_decision(encoded: str) -> IntegratedScheduleDecision:
    wire = json.loads(encoded)
    scenarios = tuple(
        ScheduleScenarioResult(
            structure=row["structure"],
            preference_state=row["preference_state"],
            point_utilities=tuple(map(tuple, row["point_utilities"])),
            compatible_optima=tuple(row["compatible_optima"]),
            differences=tuple(
                ScheduleUtilityDifference(
                    item["candidate"],
                    item["other"],
                    tuple(item["bounds"]),
                )
                for item in row["differences"]
            ),
        )
        for row in wire["scenarios"]
    )
    calibration = wire["calibration"]
    low_first, high_first = wire["second_assignment"]
    low_retention, high_retention = wire["observation_survival"]
    return IntegratedScheduleDecision(
        optimum_set=tuple(wire["optimum_set"]),
        scenarios=scenarios,
        first_assignment=tuple(wire["first_assignment"]),
        second_assignment=(tuple(low_first), tuple(high_first)),
        observation_survival=(tuple(low_retention), tuple(high_retention)),
        target_prevalence=wire["target_prevalence"],
        minimum_coverage=wire["minimum_coverage"],
        coverage_scope=wire["coverage_scope"],
        calibration=ScheduleCalibrationEvidence(
            status=calibration["status"],
            record_id=calibration.get("record_id"),
            reason=calibration.get("reason"),
            observed=calibration.get("observed"),
            source_rows_range=(calibration.get("scope_n"), calibration.get("scope_n_max")),
        ),
    )


_T = TypeVar("_T")


def _checked(value: tuple[_T | None, str | None]) -> _T:
    result, refusal = value
    raise_refusal(refusal, ScheduleDecisionRefusal)
    if result is None:  # pragma: no cover - native invariant
        raise CausalValueError("integrated schedule returned neither a result nor a refusal")
    return result


def _controls(policy: Execution) -> dict[str, Any]:
    return native_controls(
        policy, seed="seed", threads="threads", cancel="cancel", memory="memory_bytes"
    )


class CompiledIntegratedScheduleDecision:
    """An inspectable causal plan whose fixed evidence children have not yet run."""

    def __init__(self, handle: _native.IntegratedSchedulePlanHandle, policy: Execution) -> None:
        self._handle = handle
        self._policy = policy

    @property
    def program_identity(self) -> str:
        """Identity of the full schedule decision program."""
        return self._handle.program_identity

    @property
    def child_program_identities(self) -> tuple[str, str, str]:
        """Identities of the source, target and fixed structural/preference child plans."""
        source, target, declaration = self._handle.child_program_identities
        return source, target, declaration

    def plan_view(self) -> dict[str, Any]:
        """Inspect the causal compiler's plan without fitting its inputs."""
        return dict(json.loads(self._handle.plan_view_json()))

    def unit_support(self) -> dict[str, Any] | None:
        """Read the release support verdict for the compiled decision."""
        return read_unit_support(self._handle)

    @outcome_operation
    def execute(self, *, execution: Execution | None = None) -> IntegratedScheduleBundle:
        """Fit the retained evidence and project its joint region through the decision."""
        policy = resolve_execution(execution) if execution is not None else self._policy
        data, identity, encoded = _checked(policy._call(self._handle.execute, **_controls(policy)))
        return IntegratedScheduleBundle(
            data, identity, _read_decision(encoded), self.plan_view(), self.child_program_identities
        )


@outcome_operation
def compile_integrated_schedule_decision(
    source: Sequence[ScheduleSourceSubject],
    target_baseline: Sequence[bool],
    declaration: ScheduleDecisionDeclaration,
    *,
    execution: Execution | None = None,
) -> CompiledIntegratedScheduleDecision:
    """Compile the named source, target and preference programs without fitting them."""
    if not isinstance(declaration, ScheduleDecisionDeclaration):
        raise CausalTypeError("declaration must be a ScheduleDecisionDeclaration")
    if any(not isinstance(row, ScheduleSourceSubject) for row in source):
        raise CausalTypeError("source must contain ScheduleSourceSubject rows")
    policy = resolve_execution(execution)
    handle = _checked(
        policy._call(
            _native.integrated_schedule_decision_compile,
            json.dumps([asdict(subject) for subject in source]),
            list(target_baseline),
            json.dumps(asdict(declaration)),
            **_controls(policy),
        )
    )
    return CompiledIntegratedScheduleDecision(handle, policy)


def compile_schedule_decision_bundle(
    source: Sequence[ScheduleSourceSubject],
    target_baseline: Sequence[bool],
    declaration: ScheduleDecisionDeclaration,
    *,
    execution: Execution | None = None,
) -> IntegratedScheduleBundle:
    """Compile and execute the decision in one call."""
    compiled = require_success(
        compile_integrated_schedule_decision(
            source, target_baseline, declaration, execution=execution
        )
    )
    return require_success(compiled.execute())


def consume_schedule_decision_bundle(
    data: bytes | ExportedArtifact[ConsumedBundle],
    *,
    expected_identity: str | ArtifactExpectation,
    execution: Execution | None = None,
    retained_identity: str | None = None,
) -> IntegratedScheduleDecision:
    """Recompile the schedule program, replay the bundle, and read the typed decision.

    Supply ``retained_identity`` to verify the exact exported bytes before replay.
    """
    payload = container_bytes(data)
    _retained.admit(
        payload,
        retained_identity,
        "antecedent.intervention_schedule_decision.consume_schedule_decision_bundle",
        "antecedent.intervention_schedule_decision.consume_schedule_decision_bundle",
        bound=True,
    )
    expected = expected_identity
    if isinstance(expected, ArtifactExpectation):
        if expected.kind != "composition_bundle_v1" or not isinstance(expected.identity, str):
            raise CausalTypeError("expected_identity must identify a composition bundle")
        expected = expected.identity
    policy = resolve_execution(execution)
    encoded = _checked(
        policy._call(
            _native.integrated_schedule_decision_consume,
            payload,
            expected,
            **_controls(policy),
        )
    )
    return _read_decision(encoded)
