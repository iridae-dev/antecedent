"""Identify terminal benefit among patients whose binary surrogate improves under treatment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar, cast

from . import _native, _retained
from ._artifact_protocol import (
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    container_bytes,
)
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._unit_support import read_unit_support
from ._validate import count
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .execution import Execution, native_controls, resolve_execution
from .joint_distribution import ScientificQuantity
from .outcomes import outcome_operation

__all__ = [
    "CompiledPatientSurrogate",
    "ObservedSurrogatePatient",
    "PatientSurrogateDesign",
    "PatientSurrogateRefusal",
    "PatientSurrogateResult",
    "PopulationJointSurrogateLaw",
    "SurrogatePatientSample",
    "SurrogateThresholdDecision",
    "compile_patient_surrogate",
    "consume_patient_surrogate",
]

Reading = Literal["population_identification", "empirical_plugin"]
ThresholdReading = Literal["all_at_or_above", "all_below", "ambiguous"]
CellMasses = tuple[int, int, int, int]
_T = TypeVar("_T")


class PatientSurrogateRefusal(StructuredRefusal):
    """The requested evidence, identifying premises or downstream reading was refused."""


def _checked(value: tuple[_T | None, str | None]) -> _T:
    result, refusal = value
    raise_refusal(refusal, PatientSurrogateRefusal)
    if result is None:  # pragma: no cover - native contract
        raise CausalValueError("patient surrogate returned neither result nor refusal")
    return result


def _controls(execution: Execution) -> dict[str, Any]:
    return native_controls(
        execution, seed="seed", threads="threads", cancel="cancel", memory="memory_bytes"
    )


@dataclass(frozen=True, slots=True)
class PatientSurrogateDesign:
    """Name both interventions and the jointly observed binary endpoints of a parallel RCT.

    One denotes the favorable state for both endpoints. The outcome's visit follows the
    surrogate's visit. Declare consistency with no interference, positive individual
    randomized exchangeability, complete joint follow-up and monotone surrogate response.
    """

    trial_id: str
    population: str
    surrogate: str
    outcome: str
    surrogate_visit: int
    outcome_visit: int
    control: str
    treatment: str
    consistency_and_no_interference: bool
    randomized_exchangeability: bool
    complete_joint_follow_up: bool
    monotone_surrogate: bool


@dataclass(frozen=True, slots=True)
class PopulationJointSurrogateLaw:
    """Declared exact joint patient-outcome laws, as integer masses within each arm.

    The four entries in each arm correspond to ``(S, Y) = (0,0), (0,1), (1,0), (1,1)``.
    Marginal trial effects alone do not provide this joint law.
    """

    evidence_id: str
    control_cells: CellMasses
    treatment_cells: CellMasses


@dataclass(frozen=True, slots=True)
class ObservedSurrogatePatient:
    """One enrolled subject with assigned arm and both observed binary endpoints."""

    id: str
    treated: bool
    surrogate: bool
    outcome: bool


@dataclass(frozen=True, slots=True)
class SurrogatePatientSample:
    """Complete trial rows giving empirical plug-in bounds with no sampling coverage claim."""

    patients: tuple[ObservedSurrogatePatient, ...]


@dataclass(frozen=True, slots=True)
class SurrogateThresholdDecision:
    """Verified outcome of checking the entire identified set against a probability cutoff."""

    threshold: tuple[int, int]
    reading: ThresholdReading
    source_identity: str
    identity: str


class PatientSurrogateResult(ScientificResult):
    """Verified exact identified interval or empirical plug-in bounds, according to provenance."""

    def __init__(self, handle: _native.PatientSurrogateResultHandle, program_identity: str) -> None:
        self._handle = handle
        self._program_identity = program_identity

    @property
    def program_identity(self) -> str:
        """Identity of the retained causal program."""
        return self._program_identity

    @property
    def identity(self) -> str:
        """Identity of the source evidence, design and verified result."""
        return self._handle.identity

    @property
    def source_identity(self) -> str:
        """Identity of the source evidence and its design before computing bounds."""
        return self._handle.source_identity

    @property
    def reading(self) -> Reading:
        """Population identification or empirical plug-in uncertainty reading."""
        return cast(Reading, self._handle.reading)

    @property
    def bounds(self) -> tuple[float, float]:
        """Lower and upper endpoints on the probability scale."""
        return self._handle.bounds

    @property
    def exact_bounds(self) -> tuple[int, int, int]:
        """Lower numerator, upper numerator and common positive denominator."""
        return self._handle.exact_bounds

    @property
    def improver_probability(self) -> float:
        """Probability of being a surrogate improver under the retained evidence reading."""
        return self._handle.improver_probability

    @property
    def quantity(self) -> ScientificQuantity:
        """The joint causal endpoint, population and conditioning principal stratum."""
        return ScientificQuantity.from_wire(json.loads(self._handle.quantity_json()))

    @outcome_operation
    def point(self) -> float:
        """Read a single probability only when the exact population identified set is a point."""
        return _checked(self._handle.point())

    @outcome_operation
    def decide(self, numerator: int, denominator: int) -> SurrogateThresholdDecision:
        """Compare the full population identified set to the exact fraction supplied."""
        cutoff = (
            count(numerator, "numerator", minimum=0),
            count(denominator, "denominator", minimum=1),
        )
        wire = json.loads(_checked(self._handle.decide(*cutoff)))
        return SurrogateThresholdDecision(
            cutoff, wire["reading"], wire["source_identity"], wire["identity"]
        )


class CompiledPatientSurrogate:
    """An inspectable causal program over named patient-level endpoint evidence."""

    def __init__(self, handle: _native.PatientSurrogatePlanHandle, execution: Execution) -> None:
        self._handle = handle
        self._execution = execution

    @property
    def program_identity(self) -> str:
        """Identity binding the design, patient-level evidence and operation."""
        return self._handle.program_identity

    def plan_view(self) -> dict[str, Any]:
        """Inspect the six compiled layers without running the operation."""
        return dict(json.loads(self._handle.plan_view_json()))

    def unit_support(self) -> dict[str, Any] | None:
        """Read the compiled unit's release support record."""
        return read_unit_support(self._handle)

    @outcome_operation
    def execute(self, *, execution: Execution | None = None) -> PatientSurrogateResult:
        """Compute the bound under the frozen endpoint and evidence reading."""
        policy = resolve_execution(execution) if execution is not None else self._execution
        result = _checked(policy._call(self._handle.execute, **_controls(policy)))
        return PatientSurrogateResult(result, self.program_identity)

    @artifact_export
    def export(
        self, artifact_id: str = "patient-surrogate", *, execution: Execution | None = None
    ) -> ExportedArtifact[PatientSurrogateResult]:
        """Retain the joint evidence and both identities needed to verify its result."""
        policy = resolve_execution(execution) if execution is not None else self._execution
        data, identity = _checked(
            policy._call(self._handle.export, artifact_id, **_controls(policy))
        )
        kind = "patient_surrogate_identification_v1"
        return ExportedArtifact(
            bytes(data),
            kind,
            ArtifactExpectation(
                kind, {"expected_program": self.program_identity, "expected_result": identity}
            ),
        )


@outcome_operation
def compile_patient_surrogate(
    design: PatientSurrogateDesign,
    evidence: PopulationJointSurrogateLaw | SurrogatePatientSample,
    *,
    execution: Execution | None = None,
) -> CompiledPatientSurrogate:
    """Bind a randomized patient-level target and the joint observable endpoint law."""
    if not isinstance(design, PatientSurrogateDesign):
        raise CausalTypeError("design must be a PatientSurrogateDesign")
    if isinstance(evidence, PopulationJointSurrogateLaw):
        kind = "population_joint_law"
    elif isinstance(evidence, SurrogatePatientSample):
        if any(not isinstance(patient, ObservedSurrogatePatient) for patient in evidence.patients):
            raise CausalTypeError("patients must be ObservedSurrogatePatient rows")
        kind = "complete_patient_sample"
    else:
        raise CausalTypeError("joint patient-level surrogate and outcome evidence is required")
    request = json.dumps(
        {"design": asdict(design), "evidence": {"kind": kind, **asdict(evidence)}},
        allow_nan=False,
    )
    policy = resolve_execution(execution)
    handle = _checked(policy._call(_native.patient_surrogate_compile, request, **_controls(policy)))
    return CompiledPatientSurrogate(handle, policy)


@outcome_operation
@artifact_consumer(
    proof="Recompile the retained design and joint patient evidence and recompute both exact bounds.",
    identity=("expected_program", "expected_result"),
    replayable=True,
)
def consume_patient_surrogate(
    artifact: bytes,
    *,
    expected_program: str,
    expected_result: str,
    execution: Execution | None = None,
    retained_identity: str | None = None,
) -> PatientSurrogateResult:
    """Replay a portable patient-surrogate result under independently retained identities.

    Supply ``retained_identity`` to verify the exact exported bytes before replay.
    """
    payload = container_bytes(artifact)
    _retained.admit(
        payload,
        retained_identity,
        "antecedent.patient_surrogate.consume_patient_surrogate",
        "antecedent.patient_surrogate.consume_patient_surrogate",
        bound=True,
    )
    policy = resolve_execution(execution)
    result = _checked(
        policy._call(
            _native.patient_surrogate_consume,
            payload,
            expected_program,
            expected_result,
            **_controls(policy),
        )
    )
    return PatientSurrogateResult(result, expected_program)
