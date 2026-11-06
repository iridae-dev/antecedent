"""Explicit joint sources and the shared compiler's robust terminal."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import _native
from ._artifact_protocol import (
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    container_bytes,
)
from ._freeze import freeze, thaw
from ._scientific import ScientificResult
from .errors import CausalTypeError, CausalValueError
from .execution import Execution
from .outcomes import outcome_operation
from .transport._controls import call_native

if TYPE_CHECKING:
    from .decision_robust import AdmissibleContract, Claim, RobustVerdict


def _study(study: Any) -> Any:
    from .design.ranking import _bound_native

    return _bound_native(study)


def _contract(contract: Any) -> str:
    from .decision_robust import AdmissibleContract

    if not isinstance(contract, AdmissibleContract):
        raise CausalTypeError("contract must be an AdmissibleContract")
    return contract._declaration()


@dataclass(repr=False, frozen=True, slots=True)
class JointScenarioCarrier(ScientificResult):
    """Immutable joint laws, causal binding, support, and declared search standing.

    Equal row indices are explicit joint states. This source is conditional on
    the supplied laws; it does not construct a joint model from marginal bounds.
    """

    _native: Any = field(repr=False, compare=False)

    @property
    def identity(self) -> str:
        return self._native.identity

    @property
    def causal_target(self) -> str:
        return self._native.causal_target

    @property
    def causal_identification(self) -> str:
        return self._native.causal_identification

    def _summary_fields(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "complete": self.complete,
            "standing": "conditional_on_declared_joint_scenarios",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._summary_fields(),
            "causal_target": self.causal_target,
            "causal_identification": self.causal_identification,
            "joint_structure": self.joint_structure,
            "search_stop": self.search_stop,
            "n_scenarios": self.n_scenarios,
        }

    @property
    def complete(self) -> bool:
        return self._native.complete

    @property
    def search_stop(self) -> str | None:
        return self._native.search_stop

    @property
    def joint_structure(self) -> str:
        return self._native.joint_structure

    @property
    def n_scenarios(self) -> int:
        return self._native.n_scenarios

    def export(
        self, *, execution: Execution | None = None
    ) -> ExportedArtifact[JointScenarioCarrier]:
        data = call_native(execution, self._native.export)
        return ExportedArtifact(
            bytes(data),
            "joint_scenarios_v1",
            ArtifactExpectation("joint_scenarios_v1", self.identity),
        )


@dataclass(repr=False, frozen=True, slots=True)
class CompiledRobustResult(ScientificResult, Mapping[str, Any]):
    """A retained readout with per-scenario values, missing mass and conditional scope."""

    _owner: Any = field(repr=False, compare=False)
    payload: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze(self.payload))

    def _summary_fields(self) -> dict[str, Any]:
        return {"complete": self.complete, "standing": self.standing, "native_verified": False}

    def to_dict(self) -> dict[str, Any]:
        return dict(thaw(self.payload))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    @property
    def verdict(self) -> RobustVerdict:
        from .decision_robust import RobustVerdict

        return RobustVerdict._from_wire(self.payload["robust"]["verdict"])

    @property
    def complete(self) -> bool:
        return bool(self.payload["complete"])

    @property
    def standing(self) -> str:
        return str(self.payload["standing"])


def _result(owner: Any) -> CompiledRobustResult:
    return CompiledRobustResult(owner, json.loads(owner.json()))


@dataclass(repr=False, frozen=True, slots=True)
class CompiledRobustDecision(ScientificResult):
    """A compiler-owned terminal over one exact joint source and bound decision.

    Evaluation performs no fit of the anchor study. A missing/unfinished scenario
    forces an incomplete verdict and retains per-scenario values and unresolved mass.
    """

    _native: Any = field(repr=False, compare=False)

    @property
    def program_identity(self) -> str:
        return self._native.program_identity

    @property
    def source_identity(self) -> str:
        return self._native.source_identity

    @property
    def standing(self) -> str:
        return "conditional_on_declared_joint_scenarios"

    def _summary_fields(self) -> dict[str, Any]:
        return {"program_identity": self.program_identity, "standing": self.standing}

    def to_dict(self) -> dict[str, Any]:
        return dict(thaw(self.describe()))

    def describe(self) -> Mapping[str, Any]:
        return freeze(json.loads(self._native.describe()))

    @outcome_operation
    def evaluate(self, *, execution: Execution | None = None) -> CompiledRobustResult:
        return _result(call_native(execution, self._native.evaluate))

    def export(
        self, *, execution: Execution | None = None
    ) -> ExportedArtifact[CompiledRobustResult]:
        data = call_native(execution, self._native.export)
        return ExportedArtifact(
            bytes(data),
            "compiled_robust_decision_v1",
            ArtifactExpectation(
                "compiled_robust_decision_v1", {"expected_program_identity": self.program_identity}
            ),
        )

    def consume(self, artifact: Any, *, execution: Execution | None = None) -> CompiledRobustResult:
        return consume_compiled_robust_decision(artifact, compiled=self, execution=execution)


@outcome_operation
def joint_scenario_carrier(
    study: Any,
    contract: AdmissibleContract,
    claims: Sequence[Claim],
    *,
    joint_structure: str,
    exhaustive: bool,
    search_stop: str | None = None,
    execution: Execution | None = None,
) -> JointScenarioCarrier:
    """Validate explicit finite joint scenarios without fitting the anchor study.

    ``joint_structure`` names the common cross-action state/dependence. Every
    evaluated source must have joint alignment. Exhaustiveness and any search
    stop are retained declarations; completion counts never become probabilities.
    """
    from ._refusal import raise_refusal
    from .decision_robust import DecisionRefusal, _claims_payload

    payload, sources = _claims_payload(claims, "finite_scenarios")
    source, refusal = call_native(
        execution,
        _native.joint_scenario_carrier_for_study,
        _study(study),
        _contract(contract),
        payload,
        sources,
        joint_structure,
        exhaustive,
        search_stop=search_stop,
    )
    raise_refusal(refusal, DecisionRefusal)
    assert source is not None
    return JointScenarioCarrier(source)


@artifact_consumer(
    proof="JointScenarioCarrier::consume validates the full typed joint source and its separately retained content identity",
    status="loaded",
    replayable=False,
    resumable=False,
)
def consume_joint_scenarios(
    artifact: Any, *, expected_identity: str, execution: Execution | None = None
) -> JointScenarioCarrier:
    """Load immutable source data under a separately retained identity; no live fit resumes."""
    source = call_native(
        execution,
        _native.consume_joint_scenario_carrier,
        container_bytes(artifact),
        expected_identity,
    )
    return JointScenarioCarrier(source)


@outcome_operation
def compile_robust_decision(
    study: Any,
    contract: AdmissibleContract,
    source: JointScenarioCarrier,
    *,
    execution: Execution | None = None,
) -> CompiledRobustDecision:
    """Bind the exact source and decision into the shared lowered program.

    Only invariant, maximin and report-only policies are supported. Marginal-only
    envelopes, unsupported anchors and posterior/Bayes interpretations refuse.
    """
    from ._refusal import raise_refusal
    from .decision_robust import DecisionRefusal

    if not isinstance(source, JointScenarioCarrier):
        raise CausalTypeError("source must be an explicit JointScenarioCarrier")
    plan, refusal = call_native(
        execution,
        _native.compile_robust_for_study,
        _study(study),
        _contract(contract),
        source._native,
    )
    raise_refusal(refusal, DecisionRefusal)
    assert plan is not None
    return CompiledRobustDecision(plan)


@artifact_consumer(
    proof="CompiledRobustDecision::consume checks the independently compiled program/source/decision and recomputes the robust terminal",
    replayable=True,
    resumable=False,
    identity=("compiled",),
)
def consume_compiled_robust_decision(
    artifact: Any,
    *,
    compiled: CompiledRobustDecision,
    expected_program_identity: str | None = None,
    execution: Execution | None = None,
) -> CompiledRobustResult:
    """Replay only under an independently retained compatible compiler owner."""
    if not isinstance(compiled, CompiledRobustDecision):
        raise CausalTypeError("compiled must be an independently compiled CompiledRobustDecision")
    if (
        expected_program_identity is not None
        and expected_program_identity != compiled.program_identity
    ):
        raise CausalValueError(
            "expected program identity differs from the independently compiled owner"
        )
    return _result(call_native(execution, compiled._native.consume, container_bytes(artifact)))
