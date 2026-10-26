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
from ..._scientific import ScientificResult
from ...errors import CausalValueError
from ...graph import Admg
from .._status import identification_status as _canonical_status

if TYPE_CHECKING:
    from ...estimation import PreparedAnalysis


from .authority import _no_native_authority
from .catalog import EvidenceCatalog, SelectionDiagram
from .limits import _non_negative


@dataclass(frozen=True, slots=True)
class ExactDiscreteLaw:
    """Dense exact law for one concrete intervention world, last axis fastest.

    A law is supplied as exact input; this does not assert that estimates from
    finite samples are error-free. Native validation occurs before evaluation.
    """

    population: str
    regime: str
    axes: tuple[tuple[str, tuple[float, ...]], ...]
    probabilities: tuple[float, ...]
    snapshot_identity: str
    interventions: tuple[tuple[str, float], ...] = ()
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-10
    empirical_counts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # Freeze caller-owned sequences before retaining them as a snapshot.
        coerce_fields(self, axes=tuple((name, tuple(values)) for name, values in self.axes))
        coerce_fields(self, probabilities=tuple(self.probabilities))
        coerce_fields(self, interventions=tuple(tuple(item) for item in self.interventions))
        if self.empirical_counts is not None:
            counts = tuple(self.empirical_counts)
            if len(counts) != len(self.probabilities):
                raise CausalValueError("empirical_counts must have one count per probability")
            for count in counts:
                _non_negative("empirical_counts", count)
            coerce_fields(self, empirical_counts=counts)


@dataclass(frozen=True, slots=True)
class ExactTransportData:
    """Immutable collection of complete exact laws, separate from sampled data."""

    laws: tuple[ExactDiscreteLaw, ...]

    def __post_init__(self) -> None:
        coerce_fields(self, laws=tuple(self.laws))


@dataclass(frozen=True, slots=True)
class ExactTransportDistribution(ScientificResult):
    """Complete target distribution with no sampling uncertainty claim."""

    acceptance: Any = field(default=None, init=False, repr=False, compare=False)

    outcomes: tuple[str, ...]
    atoms: tuple[tuple[float, ...], ...]
    probabilities: tuple[float, ...]
    formula: str
    rules: tuple[str, ...]
    uncertainty: None = field(default=None, init=False)
    _execution: Any = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        state = "checked" if self._execution is not None else "unavailable"
        return (
            f"ExactTransportDistribution(outcomes={self.outcomes!r}, probabilities={self.probabilities!r}, "
            f"identification={state}, support={state}, uncertainty=not_applicable_exact_law, assumptions=declared)"
        )

    def inspect(self) -> Any:
        """Native four-slot reasoning and factor-level support for this execution."""
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
            "reasoning": self.inspect().to_dict(),
        }

    @artifact_export
    def export(self) -> ExportedArtifact[Any]:
        """Export the immutable native execution, independent of edited display fields."""
        if self._execution is None:
            raise _no_native_authority()
        return artifact_from_receipt(self, bytes(self._execution.export()))

    def contrast(self, reference: ExactTransportDistribution, outcome: str) -> float:
        """Difference of means from two complete target laws; no sampling interval."""
        if self.outcomes != reference.outcomes:
            raise CausalValueError("Contrast outcome coordinates must agree")
        return self.mean(outcome) - reference.mean(outcome)

    def mean(self, outcome: str) -> float:
        """Derive a numeric outcome mean from the full evaluated distribution."""
        coordinate = self.outcomes.index(outcome)
        return math.fsum(
            atom[coordinate] * p for atom, p in zip(self.atoms, self.probabilities, strict=True)
        )


@dataclass(frozen=True, slots=True)
class ClassicalTransportIdentification(ScientificResult):
    """Theorem-stage result under all source experiments and target observation.

    Display fields cannot authorize execution. The retained native derivation
    remains authoritative; catalog binding is a separate, incomplete search.
    """

    acceptance: Any = field(default=None, init=False, repr=False, compare=False)

    outcome: str
    formula: str | None
    rules: tuple[str, ...]
    outcomes: tuple[str, ...]
    _native: Any = field(repr=False, compare=False)

    @property
    def identification_status(self) -> str:
        """Canonical identification status (``identified``, ``proven_non_transportable``,
        ``missing_evidence``, ``not_certified`` or ``budget_cancel``); ``outcome`` keeps
        its earlier spelling."""
        return _canonical_status(self.outcome)

    @artifact_export
    def export(self) -> ExportedArtifact[Any]:
        """Export the retained checked proof or conservative/negative certificate."""
        return artifact_from_receipt(self, bytes(self._native.export()))

    def inspect(self) -> Any:
        """Structural certificate and all four reasoning slots; no provider access."""
        import json

        from ...results._report import InspectionReport, SlotModel

        raw = dict(json.loads(self._native.certificate_json()))
        identified = self.outcome == "identified"
        return InspectionReport(
            identification=SlotModel(
                available=identified,
                reason=None if identified else self.outcome,
                summary=self.outcome,
                payload={"outcome": self.outcome, "engine": raw},
            ),
            support=SlotModel(
                available=identified,
                reason=None,
                summary="not_estimated" if identified else "unavailable",
                payload={"engine": raw},
            ),
            uncertainty=SlotModel(
                available=False,
                reason="not_estimated",
                summary="unavailable",
            ),
            assumptions=SlotModel(
                available=True,
                summary="declared",
                payload={"rules": list(self.rules), "engine": raw.get("reasoning", raw)},
            ),
        )


@artifact_consumer(
    proof="TransportCertificateWire checked consumer: proof or conservative negative certificate",
    identity=(),
)
def consume_identification(artifact: bytes, **limits: Any) -> ClassicalTransportIdentification:
    """Independently verify a structural certificate, without rerunning identification."""
    artifact = container_bytes(artifact)
    from ..._native import consume_transport_certificate

    native = consume_transport_certificate(artifact, **limits)
    return ClassicalTransportIdentification(
        native.outcome, native.formula, tuple(native.rules), tuple(native.outcomes), native
    )


def identify_meta(
    graph: Admg,
    catalog: EvidenceCatalog,
    *,
    target: str,
    outcomes: Sequence[str],
    treatments: Sequence[str],
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> ClassicalTransportIdentification:
    """Identify with classical μsID and source-specific catalog selections.

    Completeness concerns full source experimental families; finite catalog
    binding remains a separate bounded search. No provider data are inspected.
    """
    from ..._native import identify_meta_transport_stage

    native = identify_meta_transport_stage(
        graph,
        catalog,
        target,
        list(outcomes),
        list(treatments),
        max_steps=max_steps,
        max_depth=max_depth,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return ClassicalTransportIdentification(
        native.outcome,
        native.formula,
        tuple(native.rules),
        tuple(native.outcomes),
        native,
    )


def identify_classical(
    graph: Admg,
    selection: SelectionDiagram,
    *,
    outcomes: Sequence[str],
    treatments: Sequence[str],
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> ClassicalTransportIdentification:
    """Identify under the classical *complete source experimental family*.

    The source experiments here are a theorem assumption, not a claim that a
    user's finite catalog supplies every experiment. Actual tables are bound
    separately by :func:`evaluate_exact`.
    """
    from ..._native import identify_classical_transport_stage

    native = identify_classical_transport_stage(
        graph,
        list(selection.selections),
        selection.source,
        selection.target,
        list(outcomes),
        list(treatments),
        max_steps=max_steps,
        max_depth=max_depth,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return ClassicalTransportIdentification(
        native.outcome, native.formula, tuple(native.rules), tuple(native.outcomes), native
    )


def evaluate_exact(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    *,
    at: Mapping[str, float],
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> ExactTransportDistribution:
    """Bind one checked derivation and evaluate its full target law.

    Missing catalog factors describe this derivation, not an impossibility
    proof. This stage does not fit models or estimate sampling uncertainty.
    """
    return prepare_exact(
        identification,
        catalog,
        data,
        at=at,
        max_operations=max_operations,
        max_depth=max_depth,
        max_support_rows=max_support_rows,
        memory_bytes=memory_bytes,
        cancel=cancel,
    ).estimate()


def _exact_distribution(native: Any, payload: Any) -> Any:
    atoms, probabilities, formula, rules = payload
    return ExactTransportDistribution(
        tuple(native.outcomes),
        tuple(tuple(row) for row in atoms),
        tuple(probabilities),
        formula,
        tuple(rules),
        _execution=native.freeze(),
    )


def prepare_exact(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    *,
    at: Mapping[str, float],
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[ExactTransportDistribution]:
    """Prepare the exact-law modality of the common PreparedAnalysis lifecycle."""
    from ...estimation import PreparedAnalysis, _Controls

    native = identification._native.prepare_exact(
        catalog,
        data.laws,
        dict(at),
        max_operations=max_operations,
        max_depth=max_depth,
        max_support_rows=max_support_rows,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="exact_transport", controls=_Controls(cancel=cancel))


@artifact_consumer(
    proof="PreparedStudy<ExactPreparedState>::consume verifies proof, identities and exact-law target replay",
    replayable=True,
    resumable=True,
    identity=(),
)
def consume_exact(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[ExactTransportDistribution]:
    """Verify portable proof, bindings, exact claims and all four reasoning slots.

    Uses embedded immutable laws; never fits models or fetches providers.
    """
    artifact = container_bytes(artifact)
    from ... import _native
    from ...estimation import PreparedAnalysis, _Controls

    native = _native.consume_exact_transport(
        artifact,
        max_operations=max_operations,
        max_depth=max_depth,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="exact_transport", controls=_Controls(cancel=cancel))


@dataclass(frozen=True, slots=True)
class ExactTransportQuery:
    """Checked identification, supplied evidence contract, and concrete target."""

    identification: ClassicalTransportIdentification
    catalog: EvidenceCatalog
    at: Mapping[str, float]

    def __post_init__(self) -> None:
        coerce_fields(self, at=MappingProxyType(dict(self.at)))


def inspect_catalog(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> dict[str, Any]:
    """Inspect bounded alternatives and unmet evidence without fetching providers.

    ``exhausted`` concerns configured alternatives, never arbitrary finite-catalog
    completeness. Proposed future experiments cannot satisfy missing factors.
    """
    import json

    from ..._study_lifecycle import replay_declaration

    return dict(
        json.loads(
            identification._native.catalog_search(
                replay_declaration(catalog),
                max_steps=max_steps,
                max_depth=max_depth,
                memory_bytes=memory_bytes,
                cancel=cancel,
            )
        )
    )


def inspect_proof_graph(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
) -> dict[str, Any]:
    """Return checked theorem steps and source-specific required factor leaves.

    Every leaf names its supplying regime or the exact binding failure. Proposed
    studies remain unavailable; use a hypothetical catalog delta for previews.
    """
    import json

    return dict(
        json.loads(
            identification._native.proof_graph_json(
                catalog,
                max_steps=max_steps,
                max_depth=max_depth,
            )
        )
    )
