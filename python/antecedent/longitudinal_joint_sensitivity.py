"""Sensitivity to additive selection at two decisions and final-outcome MNAR."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from . import _native
from ._artifact_protocol import (
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    container_bytes,
)
from ._refusal import raise_refusal
from ._unit_support import read_unit_support
from .errors import CausalValueError, StructuredRefusal
from .execution import Execution, native_controls, resolve_execution
from .outcomes import outcome_operation

Choice = Literal["prefer_active", "prefer_control", "undetermined"]
Interval = tuple[float, float]


class LongitudinalSensitivityRefusal(StructuredRefusal):
    """Input, support, numerical or replay refusal of the two-decision procedure."""


@dataclass(frozen=True, slots=True)
class LongitudinalSubject:
    """One independent subject's two binary decisions, histories and final outcome."""

    subject: int
    baseline: int
    treatment: tuple[int, int]
    intermediate: int
    outcome: float | None


@dataclass(frozen=True, slots=True)
class LongitudinalSensitivitySpec:
    """Schedule 11 versus 00 under a three-coordinate, prespecified sensitivity box.

    At each treatment decision, the additive potential-outcome contrast for
    deviators versus followers is the negative of the declared departure for
    schedule 11 and the positive departure for schedule 00. The final outcome
    model uses the same sign convention for missing versus recorded means.
    All histories are recorded and each departure range contains zero.
    """

    population: str
    outcome: str
    unit: str
    assignment_limits: tuple[Interval, Interval]
    mnar_limits: Interval
    outcome_support: Interval | None = None
    replicates: int = 199
    level: float = 0.95
    decision_threshold: float | None = None
    coverage_design: Literal["balanced_binary_bernoulli_v1"] | None = None


@dataclass(frozen=True, slots=True)
class LongitudinalSensitivityCorner:
    """Estimate and simultaneous band at one registered corner of the box."""

    assignment: Interval
    missingness: float
    contrast: float
    sampling_interval: Interval


@dataclass(frozen=True, slots=True)
class LongitudinalSensitivityDecision:
    """Threshold decision over the assumption range and after simultaneous widening."""

    threshold: float
    assumption_choice: Choice
    sampling_choice: Choice


@dataclass(frozen=True, slots=True)
class LongitudinalSensitivityReport:
    """Exact fitted range and nominal simultaneous whole-subject bootstrap band."""

    reference_contrast: float
    corners: tuple[LongitudinalSensitivityCorner, ...]
    assumption_range: Interval
    extrema_intervals: tuple[Interval, Interval]
    sampling_radius: float
    nominal_level: float
    seed: int
    refits: int
    decision: LongitudinalSensitivityDecision | None


def _checked(pair: tuple[Any, str | None]) -> Any:
    result, refusal = pair
    raise_refusal(refusal, LongitudinalSensitivityRefusal)
    if result is None:  # pragma: no cover
        raise CausalValueError("joint sensitivity returned neither a result nor a refusal")
    return result


def _controls(policy: Execution) -> dict[str, Any]:
    return native_controls(
        policy, seed="seed", threads="threads", cancel="cancel", memory="memory_bytes"
    )


def _interval(values: list[float]) -> Interval:
    return values[0], values[1]


class LongitudinalSensitivityResult:
    """Recomputable joint range, sampling band and threshold decision."""

    def __init__(self, handle: _native.LongitudinalJointSensitivityResultHandle) -> None:
        self._handle = handle

    @property
    def identity(self) -> str:
        """Identity of the report and its complete source rows."""
        return self._handle.identity

    @property
    def program_identity(self) -> str:
        """Semantic identity of the compiled analysis."""
        return self._handle.program_identity

    @property
    def calibration_status(self) -> str:
        """Independent calibration status of the simultaneous band."""
        return self._handle.calibration_status

    @property
    def calibration(self) -> dict[str, Any]:
        """Registry match for the full box, with a secondary joint-extrema reading."""
        return cast(dict[str, Any], json.loads(self._handle.calibration_json()))

    @property
    def report(self) -> LongitudinalSensitivityReport:
        """Rebuilt scientific result and robust threshold comparison."""
        report = json.loads(self._handle.report_json())
        decision = report["decision"]
        return LongitudinalSensitivityReport(
            report["reference_contrast"],
            tuple(
                LongitudinalSensitivityCorner(
                    _interval(point["departures"]["assignment"]),
                    point["departures"]["missingness"],
                    point["contrast"],
                    _interval(point["sampling_interval"]),
                )
                for point in report["corners"]
            ),
            _interval(report["assumption_range"]),
            (
                _interval(report["extrema_intervals"][0]),
                _interval(report["extrema_intervals"][1]),
            ),
            report["sampling_radius"],
            report["nominal_level"],
            report["seed"],
            report["refits"],
            None if decision is None else LongitudinalSensitivityDecision(**decision),
        )

    @artifact_export
    def export(self) -> ExportedArtifact[LongitudinalSensitivityResult]:
        """Retain the analysis, source rows and bootstrap result for independent replay."""
        kind = "longitudinal_joint_sensitivity"
        return ExportedArtifact(
            self._handle.export(),
            kind,
            ArtifactExpectation(
                kind, {"expected": self.identity, "expected_program": self.program_identity}
            ),
        )


class CompiledLongitudinalJointSensitivity:
    """Compiled scientific target before fitting either mechanism."""

    def __init__(self, handle: _native.LongitudinalJointSensitivityPlanHandle) -> None:
        self._handle = handle

    @property
    def identity(self) -> str:
        """Identity of the input data, scientific box and analysis declaration."""
        return self._handle.identity

    def plan_view(self) -> dict[str, Any]:
        """Read the lowered causal program and resampling design."""
        return cast(dict[str, Any], json.loads(self._handle.plan_view_json()))

    @property
    def unit_support(self) -> dict[str, Any] | None:
        """The support status of the compiled analysis."""
        return read_unit_support(self._handle)

    @outcome_operation
    def execute(self, *, execution: Execution | None = None) -> LongitudinalSensitivityResult:
        """Fit both mechanisms and resample the same complete subjects jointly."""
        policy = resolve_execution(execution)
        handle = _checked(policy._call(self._handle.execute, **_controls(policy)))
        return LongitudinalSensitivityResult(handle)


@outcome_operation
def compile_longitudinal_joint_sensitivity(
    spec: LongitudinalSensitivitySpec,
    rows: Sequence[LongitudinalSubject],
) -> CompiledLongitudinalJointSensitivity:
    """Compile the declared two-decision contrast using independent subject histories."""
    handle = _checked(
        _native.longitudinal_joint_sensitivity_compile(
            json.dumps(asdict(spec)),
            [(r.subject, r.baseline, r.treatment, r.intermediate, r.outcome) for r in rows],
        )
    )
    return CompiledLongitudinalJointSensitivity(handle)


@outcome_operation
@artifact_consumer(
    proof="Recompile both mechanisms and reproduce the simultaneous band and decision from retained subject trajectories.",
    replayable=True,
    identity=("expected", "expected_program"),
    input="data",
)
def consume_longitudinal_joint_sensitivity(
    data: object,
    *,
    expected: str,
    expected_program: str,
    execution: Execution | None = None,
) -> LongitudinalSensitivityResult:
    """Recompute a portable result from its rows and retained seed."""
    policy = resolve_execution(execution)
    handle = _checked(
        policy._call(
            _native.longitudinal_joint_sensitivity_consume,
            container_bytes(data),
            expected,
            expected_program,
            **_controls(policy),
        )
    )
    return LongitudinalSensitivityResult(handle)


__all__ = [
    "CompiledLongitudinalJointSensitivity",
    "LongitudinalSensitivityCorner",
    "LongitudinalSensitivityDecision",
    "LongitudinalSensitivityRefusal",
    "LongitudinalSensitivityReport",
    "LongitudinalSensitivityResult",
    "LongitudinalSensitivitySpec",
    "LongitudinalSubject",
    "compile_longitudinal_joint_sensitivity",
    "consume_longitudinal_joint_sensitivity",
]
