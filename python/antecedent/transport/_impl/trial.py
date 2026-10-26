from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from ..._artifact_protocol import ExportedArtifact, artifact_export, artifact_from_receipt
from ..._defaults import OMITTED
from ..._scientific import ScientificResult
from ...errors import CausalUnsupportedError, CausalValueError
from ...graph import Admg
from ...learners import LearnerSpec, Logistic, Ridge, _learner_wire

if TYPE_CHECKING:
    from ...estimation import PreparedAnalysis


from .catalog import SelectionDiagram
from .limits import _non_negative, _optional_non_negative


@dataclass(frozen=True, slots=True)
class TrialAipwData:
    """IID trial and target rows with known source randomization probabilities.

    For ``nested_cohort``, target rows are cohort nonparticipants. For
    ``independent_samples``, target rows are a separately sampled representative
    target population. Target outcomes/treatments are ignored; use finite placeholders.
    """

    covariates: Mapping[str, Sequence[float]]
    outcome: Sequence[float]
    treatment: Sequence[bool]
    source: Sequence[bool]
    randomization: Sequence[float]
    sampling: Literal["nested_cohort", "independent_samples"]

    def _buffers(self, names: Sequence[str]) -> dict[str, Any]:
        from .._buffers import trial_buffers

        return trial_buffers(self, names)

    def _json(self, names: Sequence[str]) -> str:
        import json

        unknown = set(self.covariates) - set(names)
        if unknown:
            raise ValueError(f"Unknown baseline covariates: {sorted(unknown)}")
        features = [i for i, name in enumerate(names) if name in self.covariates]
        return json.dumps(
            dict(
                features=features,
                covariates=[list(map(float, self.covariates[names[i]])) for i in features],
                outcome=list(map(float, self.outcome)),
                treatment=list(self.treatment),
                source=list(self.source),
                randomization=list(map(float, self.randomization)),
                sampling=self.sampling,
            ),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class TrialAipw:
    """Nuisance configuration for the certified binary trial contrast."""

    outcome: LearnerSpec = Ridge()
    membership: LearnerSpec = Logistic()
    folds: int = 5


@dataclass(frozen=True, slots=True)
class TrialAipwQuery:
    graph: Admg
    diagram: SelectionDiagram
    treatment: str
    outcome: str


@dataclass(frozen=True, slots=True)
class TransportInference:
    """Joint outer bootstrap and nominal coverage; RNG belongs to Execution.seed."""

    bootstrap: int = OMITTED["transport_bootstrap"]
    coverage_level: float = OMITTED["transport_coverage_level"]

    def __post_init__(self) -> None:
        _non_negative("bootstrap", self.bootstrap)
        if not isinstance(self.coverage_level, (int, float)) or not (
            0.0 < float(self.coverage_level) < 1.0
        ):
            raise CausalValueError("coverage_level must lie strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class TransportControls:
    """Physical resource limits, separate from scientific inference settings."""

    max_operations: int = 10_000_000
    max_depth: int = 256
    max_support_rows: int = 1_000_000
    memory_bytes: int | None = None
    cancel: Any = None

    def __post_init__(self) -> None:
        _non_negative("max_operations", self.max_operations)
        _non_negative("max_depth", self.max_depth)
        _non_negative("max_support_rows", self.max_support_rows)
        _optional_non_negative("memory_bytes", self.memory_bytes)


@dataclass(frozen=True, slots=True)
class TrialNuisanceDiagnostics:
    """Held-out role losses; these do not establish causal identification."""

    membership_logloss: float
    outcome_rmse: tuple[float, float]


@dataclass(frozen=True, slots=True)
class LearnedTrialEstimate(ScientificResult):
    acceptance: Any = field(default=None, init=False, repr=False, compare=False)

    estimate: float
    interval: tuple[float, float] | None
    uncertainty_reason: str | None
    replicates: tuple[tuple[int, float], ...]
    failures: int
    overlap: Mapping[str, Any]
    diagnostics: TrialNuisanceDiagnostics
    _execution: Any = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "interval": list(self.interval) if self.interval is not None else None,
            "uncertainty_reason": self.uncertainty_reason,
            "replicates": [list(row) for row in self.replicates],
            "failures": self.failures,
            "overlap": {k: dict(v) for k, v in self.overlap.items()},
            "diagnostics": {
                "membership_logloss": self.diagnostics.membership_logloss,
                "outcome_rmse": list(self.diagnostics.outcome_rmse),
            },
            "reasoning": self.inspect().to_dict(),
        }

    @artifact_export
    def export(self) -> ExportedArtifact[Any]:
        return artifact_from_receipt(self, bytes(self._execution.export()))

    def inspect(self) -> Any:
        import json

        from ...results._report import InspectionReport

        return InspectionReport(**json.loads(self._execution.inspection_json()))


def _learned_trial(native: Any, payload: str) -> LearnedTrialEstimate:
    import json

    raw = json.loads(payload)
    return LearnedTrialEstimate(
        raw["estimate"],
        tuple(raw["interval"]) if raw["interval"] else None,
        raw["uncertainty_reason"],
        tuple((i, v) for i, v in raw["replicates"]),
        raw["failures"],
        MappingProxyType({k: MappingProxyType(v) for k, v in raw["overlap"].items()}),
        TrialNuisanceDiagnostics(
            raw["diagnostics"]["membership_logloss"], tuple(raw["diagnostics"]["outcome_rmse"])
        ),
        native.freeze(),
    )


def prepare_trial(
    query: TrialAipwQuery,
    data: TrialAipwData,
    *,
    provider: TrialAipw | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
    seed: int = 1,
) -> PreparedAnalysis[LearnedTrialEstimate]:
    """Prepare a checked binary contrast; baseline covariates must match its certificate."""
    import json

    from ... import _native
    from ...estimation import PreparedAnalysis, _Controls

    provider = provider or TrialAipw()
    inference = inference or TransportInference()
    controls = controls or TransportControls()
    if (controls.max_operations, controls.max_depth, controls.max_support_rows) != (
        10_000_000,
        256,
        1_000_000,
    ):
        raise CausalUnsupportedError(
            "Discrete evaluator limits do not apply to trial AIPW; use memory_bytes and cancel",
            reason_code="option_not_applicable",
        )
    options = dict(
        outcome=_learner_wire(provider.outcome),
        membership=_learner_wire(provider.membership),
        folds=provider.folds,
        bootstrap=inference.bootstrap,
        coverage_level=inference.coverage_level,
    )
    native = _native.prepare_learned_trial(
        query.graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        query.treatment,
        query.outcome,
        data,
        json.dumps(options),
        seed=seed,
        memory_bytes=controls.memory_bytes,
        cancel=controls.cancel,
    )
    return PreparedAnalysis(
        native, kind="learned_trial", controls=_Controls(cancel=controls.cancel)
    )
