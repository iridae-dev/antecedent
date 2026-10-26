from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ..._columns import coerce_fields
from ..._defaults import OMITTED
from ..._scientific import ScientificResult
from ..._transport_results import TransportContrast, TransportUncertainty
from ...errors import CausalValueError
from ...learners import LearnerSpec, Logistic, _learner_wire

if TYPE_CHECKING:
    from ...estimation import PreparedAnalysis


from .authority import _no_native_authority
from .catalog import EvidenceCatalog
from .exact import ClassicalTransportIdentification, ExactDiscreteLaw
from .limits import _non_negative, _optional_non_negative


@dataclass(frozen=True, slots=True)
class RegimeSample:
    """Finite categorical sample for one regime and intervention world."""

    population: str
    regime: str
    snapshot_identity: str
    columns: Mapping[str, Sequence[float | None]]
    interventions: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            columns=MappingProxyType(
                {
                    name: tuple(
                        None if value is None or math.isnan(value) else float(value)
                        for value in values
                    )
                    for name, values in self.columns.items()
                }
            ),
        )
        coerce_fields(self, interventions=tuple(tuple(item) for item in self.interventions))


@dataclass(frozen=True, slots=True)
class StatisticalTransportData:
    """Mixed supplied laws and estimated samples for one statistical prepare."""

    samples: tuple[RegimeSample, ...] = ()
    laws: tuple[ExactDiscreteLaw, ...] = ()

    def __post_init__(self) -> None:
        coerce_fields(self, samples=tuple(self.samples))
        coerce_fields(self, laws=tuple(self.laws))


@dataclass(frozen=True, slots=True)
class StatisticalTransportDistribution(ScientificResult):
    """Plug-in target law with licensed or withheld sampling uncertainty."""

    acceptance: Any = field(default=None, init=False, repr=False, compare=False)

    outcomes: tuple[str, ...]
    atoms: tuple[tuple[float, ...], ...]
    probabilities: tuple[float, ...]
    formula: str
    rules: tuple[str, ...]
    uncertainty: TransportUncertainty | None = None
    _execution: Any = field(default=None, repr=False, compare=False)

    def inspect(self) -> Any:
        import json

        from ...results._report import InspectionReport

        if self._execution is None:
            raise _no_native_authority()
        return InspectionReport(**json.loads(self._execution.inspection_json()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcomes": self.outcomes,
            "atoms": self.atoms,
            "probabilities": self.probabilities,
            "formula": self.formula,
            "rules": self.rules,
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty is not None else None,
            "reasoning": self.inspect().to_dict(),
        }

    def __repr__(self) -> str:
        state = "checked" if self._execution is not None else "unavailable"
        available = self._execution is not None and self.inspect().uncertainty.available
        return (
            f"StatisticalTransportDistribution(outcomes={self.outcomes!r}, probabilities={self.probabilities!r}, "
            f"identification={state}, support={state}, uncertainty={'pointwise_bootstrap' if available else 'unavailable'}, assumptions=declared)"
        )

    def mean(self, outcome: str) -> float:
        coordinate = self.outcomes.index(outcome)
        return math.fsum(
            atom[coordinate] * p for atom, p in zip(self.atoms, self.probabilities, strict=True)
        )

    def contrast(
        self, reference: StatisticalTransportDistribution, outcome: str
    ) -> TransportContrast:
        """Difference of means with paired bootstrap draws from compatible native runs."""
        if self._execution is None or reference._execution is None:
            raise _no_native_authority()
        import json

        result = dict(json.loads(self._execution.contrast(reference._execution, outcome)))
        if result["interval"] is not None:
            result["interval"] = tuple(result["interval"])
        return TransportContrast(result)

    @artifact_export
    def export(self) -> ExportedArtifact[Any]:
        if self._execution is None:
            raise _no_native_authority()
        return artifact_from_receipt(self, bytes(self._execution.export()))


@dataclass(frozen=True, slots=True)
class EmpiricalTable:
    """Unsmoothed empirical joint; empty cells retain sampling-zero semantics."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "empirical_table"}


@dataclass(frozen=True, slots=True)
class LearnedCategorical:
    """Coherent finite categorical chain model, with empirical support checks.

    Model probabilities may extrapolate into empty joint cells. Such probabilities
    do not establish positivity; conditionals on unobserved strata still refuse.
    Bootstrap intervals are nominal and uncalibrated, not doubly robust.
    """

    learner: LearnerSpec | str = Logistic()

    def _wire(self) -> dict[str, object]:
        return {"kind": "learned_categorical", "learner": _learner_wire(self.learner)}


@dataclass(frozen=True, slots=True)
class StatisticalTransportQuery:
    """Checked identification plus empirical-table inference settings."""

    identification: ClassicalTransportIdentification
    catalog: EvidenceCatalog
    at: Mapping[str, float]
    bootstrap: int = OMITTED["transport_bootstrap"]
    coverage_level: float = OMITTED["transport_coverage_level"]
    estimator: EmpiricalTable | LearnedCategorical | str = "plugin"

    def __post_init__(self) -> None:
        coerce_fields(self, at=MappingProxyType(dict(self.at)))


def prepare_statistical(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: StatisticalTransportData,
    *,
    at: Mapping[str, float],
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
    bootstrap: int = OMITTED["transport_bootstrap"],
    coverage_level: float = OMITTED["transport_coverage_level"],
    estimator: EmpiricalTable | LearnedCategorical | str = "plugin",
    seed: int = 1,
) -> PreparedAnalysis[StatisticalTransportDistribution]:
    """Prepare the empirical-table modality of the common PreparedAnalysis lifecycle."""
    from ..._native import prepare_statistical_transport
    from ...estimation import PreparedAnalysis, _Controls

    native = prepare_statistical_transport(
        identification._native,
        catalog,
        data,
        dict(at),
        max_operations=max_operations,
        max_depth=max_depth,
        max_support_rows=max_support_rows,
        memory_bytes=memory_bytes,
        cancel=cancel,
        bootstrap=bootstrap,
        coverage_level=coverage_level,
        estimator=estimator,
        seed=seed,
    )
    return PreparedAnalysis(native, kind="statistical_transport", controls=_Controls(cancel=cancel))


def evaluate_statistical_grid(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: StatisticalTransportData,
    *,
    at: Sequence[Mapping[str, float]],
    **kwargs: Any,
) -> tuple[StatisticalTransportDistribution, ...]:
    """Evaluate a grid using shared dataset refits and shared successful replicate IDs.

    Intervals are pointwise. Use ``result.contrast(reference, outcome)`` to keep
    shared-factor covariance when comparing two points from the same execution.
    """
    if not at:
        raise CausalValueError("The treatment grid must not be empty")
    study = prepare_statistical(identification, catalog, data, at=at[0], **kwargs)
    points = study._native.estimate_grid([dict(point) for point in at], cancel=kwargs.get("cancel"))
    return tuple(_statistical_distribution(point, point.last_result()) for point in points)


@artifact_consumer(
    proof="PreparedStudy<StatisticalPreparedState>::consume checks proof, fitted joints and recorded point without re-bootstrap",
    replayable=True,
    identity=(),
    unresolved=(
        "raw source samples are not embedded; bind_snapshot before a new estimator execution",
    ),
)
def consume_statistical(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[StatisticalTransportDistribution]:
    """Verify portable proof and recomputed plug-in point; do not re-bootstrap.

    ``max_support_rows`` bounds the samples a later ``refresh`` or
    ``replace_snapshot`` may bind to the consumed handle.
    """
    artifact = container_bytes(artifact)
    from ... import _native
    from ...estimation import PreparedAnalysis, _Controls

    native = _native.consume_statistical_transport(
        artifact,
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="statistical_transport", controls=_Controls(cancel=cancel))


def _statistical_distribution(native: Any, payload: Any) -> StatisticalTransportDistribution:
    import json

    atoms, probabilities, formula, rules, uncertainty = payload
    parsed = json.loads(uncertainty) if isinstance(uncertainty, str) else uncertainty
    return StatisticalTransportDistribution(
        tuple(native.outcomes),
        tuple(tuple(row) for row in atoms),
        tuple(probabilities),
        formula,
        tuple(rules),
        uncertainty=TransportUncertainty(parsed),
        _execution=native.freeze(),
    )
