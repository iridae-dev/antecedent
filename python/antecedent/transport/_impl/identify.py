from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal, TypedDict

import numpy as np

from ..._artifact_protocol import artifact_consumer, container_bytes
from ..._columns import coerce_fields
from ..._native import (
    consume_z_transport_artifact as _consume_z_transport_artifact,
)
from ..._native import (
    consume_z_transport_sensitivity_artifact as _consume_z_transport_sensitivity_artifact,
)
from ..._native import (
    estimate_trial_transport as _estimate_trial_transport,
)
from ..._native import (
    identify_transport as _identify_transport,
)
from ..._native import (
    identify_z_transport_stage as _identify_z_transport_stage,
)
from ..._native import (
    replay_z_transport_proposal as _replay_z_transport_proposal,
)
from ..._native import (
    roundtrip_expr_arena as _roundtrip_expr_arena,
)
from ...errors import CausalTypeError, CausalValueError
from ...execution import Execution
from ...graph import Admg
from ...query import (
    AverageDerivative,
    DirectionalDerivative,
    Elasticity,
    PointDerivative,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
)
from .._consumption import TransportConsumption
from .._controls import call_native
from .._status import identification_status as _canonical_status
from .catalog import (
    EvidenceCatalog,
    SelectionDiagram,
    ZTransportCandidate,
    ZTransportQuery,
)
from .limits import _non_negative, _optional_non_negative


@dataclass(frozen=True, slots=True)
class TransportQuery:
    """Transport a response query under a single-source selection diagram.

    ``query`` is the response requested in the target population, ``diagram``
    the selection diagram's populations and mechanism-selection targets, and
    ``source_experiments`` the variables randomized in the source.

    :func:`antecedent.analyze` estimates the licensed cell (explicit ``Admg``,
    Frequentist, validation ``none``): a mean :class:`antecedent.ResponseCurve`
    transported from trial to target by binary trial-to-target IPW. It reads
    three data columns named by the keyword-only fields: ``trial`` (source-trial
    membership, nonzero for trial rows), ``selection_probability``
    (``P(S=1 | X)`` on every row) and ``treatment_probability``
    (``P(A=1 | X, S=1)`` on trial rows). The selection diagram and these column
    bindings freeze at prepare; a refresh reads the columns from the new data.
    """

    query: object
    diagram: SelectionDiagram
    source_experiments: Sequence[str] = ()
    _: KW_ONLY
    catalog: EvidenceCatalog | None = None
    trial: str | None = None
    selection_probability: str | None = None
    treatment_probability: str | None = None
    kind: Literal["transport"] = field(default="transport", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.query is None:
            raise CausalValueError("query must not be None")
        if len(set(self.source_experiments)) != len(self.source_experiments):
            raise CausalValueError("source_experiments must not contain duplicates")
        if any(not value.strip() for value in self.source_experiments):
            raise CausalValueError("source_experiments must contain variable names")
        if self.catalog is not None:
            derived = list(self.catalog.source_experiment_variables(self.diagram.source))
            explicit = list(self.source_experiments)
            if explicit and set(explicit) != set(derived):
                raise CausalValueError("source_experiments disagree with the evidence catalog")
            if not explicit:
                coerce_fields(self, source_experiments=tuple(derived))
        columns = (self.trial, self.selection_probability, self.treatment_probability)
        if any(column is not None for column in columns):
            if any(column is None for column in columns):
                raise CausalValueError(
                    "trial, selection_probability and treatment_probability name the trial "
                    "columns together; supply all three"
                )
            if any(not str(column).strip() for column in columns):
                raise CausalValueError("trial columns must be non-empty variable names")
            if len(set(columns)) != len(columns):
                raise CausalValueError("trial columns must be distinct")

    @property
    def trial_columns(self) -> tuple[str, str, str] | None:
        """``(trial, selection_probability, treatment_probability)`` when bound."""
        if self.trial is None or self.selection_probability is None:
            return None
        if self.treatment_probability is None:
            return None
        return (self.trial, self.selection_probability, self.treatment_probability)


@dataclass(frozen=True, slots=True)
class PopulationFactor:
    """One population-labelled distribution factor in a transport formula."""

    population: str
    variables: Sequence[str]
    conditioned_on: Sequence[str]
    interventions: Sequence[str]
    regime: int | None = None


@dataclass(frozen=True, slots=True)
class DirectFormula:
    factor: PopulationFactor


@dataclass(frozen=True, slots=True)
class StandardizationFormula:
    over: Sequence[str]
    source_response: PopulationFactor
    target_law: PopulationFactor


@dataclass(frozen=True, slots=True)
class RecursiveFactorizationFormula:
    sum_out: Sequence[str]
    factors: Sequence[PopulationFactor]


TransportFormula = DirectFormula | StandardizationFormula | RecursiveFactorizationFormula


@dataclass(frozen=True, slots=True)
class TransportCertificate:
    rule: str
    selection_targets: Sequence[str]


@dataclass(frozen=True, slots=True)
class NotCertifiedCertificate:
    """Conservative refusal; this is not a general non-transportability claim."""

    reason: str
    witness: Sequence[str]
    message: str


# Historical name denotes an inconclusive result, never an impossibility proof.
NonTransportableCertificate = NotCertifiedCertificate


@dataclass(frozen=True, slots=True)
class MissingEvidenceCertificate:
    """Required available evidence was absent. Distinct from :class:`NotCertifiedCertificate`."""

    reason: str
    missing: Sequence[str]
    message: str


@dataclass(frozen=True, slots=True)
class TransportIdentification:
    formula: TransportFormula | None
    certificate: TransportCertificate | NotCertifiedCertificate | MissingEvidenceCertificate
    outcome: str = "not_certified"
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"operation_ids": ["identify.transport_sid"]}
    )
    pretty: str | None = None
    latex: str | None = None
    leaf_bindings: tuple[tuple[str, int | None], ...] = ()
    expr_root: int | None = None
    expr_wire_json: str | None = None
    # The native `TransportIdentificationResult` this was built from. `estimate_trial_effect`
    # forwards it to the native estimator so the refusal gate keys off the certificate that was
    # actually produced by `identify(...)`, not off this frozen dataclass's own (re-derivable
    # but caller-editable) `transportable` property. Opaque outside this module.
    _native: Any = field(default=None, repr=False, compare=False)

    @property
    def transportable(self) -> bool:
        return self.formula is not None

    @property
    def identification_status(self) -> str:
        """Canonical identification status (``identified``, ``proven_non_transportable``,
        ``missing_evidence``, ``not_certified`` or ``budget_cancel``); ``outcome`` keeps
        its earlier spelling."""
        return _canonical_status(self.outcome)


@dataclass(frozen=True, slots=True)
class OverlapDiagnostic:
    probability_min: float
    probability_max: float
    effective_sample_size: float
    extreme_weight_count: int


@dataclass(frozen=True, slots=True)
class TransportOverlapReport:
    selection: OverlapDiagnostic
    treatment: OverlapDiagnostic


@dataclass(frozen=True, slots=True)
class TrialTransportEstimate:
    # Rule id of the transport certificate that authorized this estimate, so the result
    # carries its own identification provenance rather than relying on the caller to
    # remember which `TransportIdentification` it passed to `estimate_trial_effect`.
    rule: str
    ipw: float
    aipw: float | None
    overlap: TransportOverlapReport
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"operation_ids": ["estimate.trial_to_target"]}
    )


def _response_args(query: object) -> dict[str, Any]:
    supported = (
        ResponseCurve,
        AverageDerivative,
        PointDerivative,
        Elasticity,
        SemiElasticity,
        DirectionalDerivative,
        ResponseJacobian,
    )
    if not isinstance(query, supported):
        raise CausalTypeError("TransportQuery.query must be a response-family query")
    if getattr(query, "observation", None) is not None:
        raise CausalValueError("transport identification currently requires complete observation")
    if getattr(query, "observation_assumptions", ()):
        raise CausalValueError(
            "transport identification does not yet compose observation_assumptions"
        )
    if getattr(query, "target_population", None) is not None:
        raise CausalValueError(
            "TransportQuery owns its source/target populations; its embedded response "
            "must not set target_population"
        )
    at: list[float] | None
    if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
        treatments = list(query.treatments)
        outcomes = list(query.outcomes)
        at = (
            [query.at[name] for name in treatments]
            if isinstance(query.at, Mapping)
            else list(query.at)
        )
    else:
        treatments = [query.treatment]
        outcomes = [query.outcome]
        at = [query.at] if hasattr(query, "at") else None
    direction = None
    if isinstance(query, DirectionalDerivative):
        direction = (
            [query.direction[name] for name in treatments]
            if isinstance(query.direction, Mapping)
            else list(query.direction)
        )
    scale = "identity"
    if isinstance(query, Elasticity):
        scale = "log_log"
    elif isinstance(query, SemiElasticity):
        scale = "log_treatment" if query.log_scale == "treatment" else "log_outcome"
    weighting = getattr(query, "weighting", None) or "observed"
    if not isinstance(weighting, str):
        raise CausalTypeError("AverageDerivative.weighting currently accepts 'observed'")
    return {
        "kind": query.kind,
        "treatments": treatments,
        "outcomes": outcomes,
        "grid": list(query.grid) if isinstance(query, ResponseCurve) else None,
        "at": at,
        "direction": direction,
        "order": getattr(query, "order", 1),
        "scale": scale,
        "weighting": weighting,
    }


def identify(*, graph: Admg, query: TransportQuery) -> TransportIdentification:
    """Identify a single-source graphical transport formula and its certificate."""

    if not isinstance(graph, Admg):
        raise CausalTypeError("transport.identify requires graph=Admg(...)")
    if not isinstance(query, TransportQuery):
        raise CausalTypeError("query must be a TransportQuery")
    raw = _identify_transport(
        graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        list(query.source_experiments),
        catalog=query.catalog,
        **_response_args(query.query),
    )
    if not raw.transportable:
        if raw.reason is None or raw.message is None:
            raise RuntimeError("native transport refusal omitted its certificate")
        if raw.outcome == "missing_evidence":
            return TransportIdentification(
                None,
                MissingEvidenceCertificate(raw.reason, raw.selection_targets, raw.message),
                outcome=raw.outcome,
                _native=raw,
            )
        return TransportIdentification(
            None,
            NotCertifiedCertificate(raw.reason, raw.selection_targets, raw.message),
            outcome=raw.outcome,
            _native=raw,
        )
    factors = [
        PopulationFactor(population, variables, conditioned_on, interventions, regime)
        for population, variables, conditioned_on, interventions, regime in zip(
            raw.factor_populations,
            raw.factor_variables,
            raw.factor_conditioned_on,
            raw.factor_interventions,
            raw.factor_regimes,
            strict=True,
        )
    ]
    if raw.formula_kind == "direct":
        formula: TransportFormula = DirectFormula(factors[0])
    elif raw.formula_kind == "standardize":
        formula = StandardizationFormula(raw.marginalize, factors[0], factors[1])
    elif raw.formula_kind == "recursive_factorization":
        formula = RecursiveFactorizationFormula(raw.marginalize, factors)
    else:  # native result invariant
        raise RuntimeError(f"unknown native transport formula {raw.formula_kind!r}")
    if raw.rule is None:
        raise RuntimeError("native transport formula omitted its certificate rule")
    return TransportIdentification(
        formula,
        TransportCertificate(raw.rule, raw.selection_targets),
        outcome=raw.outcome,
        pretty=raw.expr_pretty,
        latex=raw.expr_latex,
        leaf_bindings=tuple(zip(raw.leaf_populations, raw.leaf_regimes, strict=True)),
        expr_root=raw.expr_root,
        expr_wire_json=raw.expr_wire_json,
        _native=raw,
    )


def identify_z_transport(
    *,
    graph: Admg,
    query: ZTransportQuery,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    execution: Execution | None = None,
) -> Any:
    """Create a native stage for bounded single-source z-transport search.

    The returned stage exposes ``outcome`` and ``reason``. Positive formulas
    are checked before binding evidence; a negative search is not treated as a
    theorem obstruction unless the complete experimental-family premises hold.
    Call ``stage.decide(catalog)`` to check those premises and obtain a
    replayable bounded obstruction or a precise missing-evidence result.
    On an identified result, call ``prepare_exact`` or ``prepare_empirical``
    to obtain a point-only execution. The limits are retained by the stage and
    inherited by every decision, preparation and proposal made from it.
    """
    if not isinstance(graph, Admg):
        raise CausalTypeError("transport.identify_z_transport requires graph=Admg(...)")
    if not isinstance(query, ZTransportQuery):
        raise CausalTypeError("query must be a ZTransportQuery")
    return call_native(
        execution,
        _identify_z_transport_stage,
        graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        list(query.outcomes),
        list(query.treatments),
        list(query.controllable),
        dict(query.experiment_assignment),
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        cancel="cancel",
        memory="memory_bytes",
    )


class _ReplayLimits(TypedDict):
    """Validated replay limits, keeping the mandatory bounds int and the rest optional."""

    max_operations: int
    max_depth: int
    max_support_rows: int | None
    max_laws: int | None
    max_law_cells: int | None


def _consume_limits(
    max_operations: int,
    max_depth: int,
    max_support_rows: int | None,
    max_laws: int | None,
    max_law_cells: int | None,
) -> _ReplayLimits:
    """Validated consumer-side replay limits; ``None`` keeps the native default."""
    return {
        "max_operations": _non_negative("max_operations", max_operations),
        "max_depth": _non_negative("max_depth", max_depth),
        "max_support_rows": _optional_non_negative("max_support_rows", max_support_rows),
        "max_laws": _optional_non_negative("max_laws", max_laws),
        "max_law_cells": _optional_non_negative("max_law_cells", max_law_cells),
    }


@artifact_consumer(
    proof="ZTransportArtifactWire consumer: checks proof and provider bindings then replays the retained point",
    replayable=True,
    identity=(),
)
def consume_z_transport_artifact(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    execution: Execution | None = None,
) -> TransportConsumption:
    """Independently verify and recompute an exported point-only zTR result.

    The replay runs under the consumer's own limits (``max_operations``,
    ``max_depth``, ``max_support_rows``, ``max_laws``, ``max_law_cells``);
    nothing the artifact recorded raises them, and an artifact whose stored
    laws or recorded limits exceed them is refused.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return TransportConsumption.from_checked_json(
        "consume_z_transport_artifact",
        call_native(
            execution,
            _consume_z_transport_artifact,
            artifact,
            **_consume_limits(max_operations, max_depth, max_support_rows, max_laws, max_law_cells),
            cancel="cancel",
            memory="memory_bytes",
        ),
    )


def plan_z_transport_evidence(
    stage: Any,
    catalog: EvidenceCatalog,
    candidates: Sequence[ZTransportCandidate],
    *,
    failure_snapshot: bytes | None = None,
    max_evaluated: int | None = None,
    execution: Execution | None = None,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Assess candidates against a stage's frozen failure and catalog.

    Passing ``failure_snapshot`` reuses an exported snapshot and verifies that
    it exactly matches this stage and catalog before planning. At most
    ``max_evaluated`` candidates are assessed (all of them by default); the
    report's ``candidate_universe_size``, ``search_limit``, ``evaluated``,
    ``unevaluated`` and ``truncated`` keys are the budget receipt.
    """
    import json

    if not hasattr(stage, "plan_evidence"):
        raise CausalTypeError("stage must be returned by identify_z_transport")
    if failure_snapshot is not None and not isinstance(failure_snapshot, bytes):
        raise CausalTypeError("failure_snapshot must be bytes")
    if max_evaluated is not None and _non_negative("max_evaluated", max_evaluated) == 0:
        raise CausalValueError("max_evaluated must be at least one")
    report, proposals = call_native(
        execution,
        stage.plan_evidence,
        catalog,
        list(candidates),
        failure_snapshot,
        max_evaluated=max_evaluated,
        cancel="cancel",
        memory=None,
    )
    return json.loads(report), tuple(proposals)


@artifact_consumer(
    proof="native sensitivity consumer replays the independently checked baseline and recomputes the discrete contamination assumption range",
    replayable=True,
    resumable=False,
    identity=(),
)
def consume_z_transport_sensitivity_artifact(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    execution: Execution | None = None,
) -> TransportConsumption:
    """Independently verify and recompute a portable zTR sensitivity range.

    The embedded baseline replays under the consumer's own limits, as in
    :func:`consume_z_transport_artifact`.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return TransportConsumption.from_checked_mapping(
        "z_transport_sensitivity",
        call_native(
            execution,
            _consume_z_transport_sensitivity_artifact,
            artifact,
            **_consume_limits(max_operations, max_depth, max_support_rows, max_laws, max_law_cells),
            cancel="cancel",
            memory="memory_bytes",
        ),
    )


@artifact_consumer(
    proof="native Z proposal replay re-derives the original failure and verifies the hypothetical proof/catalog binding",
    replayable=True,
    resumable=False,
    identity=(),
    unresolved=(
        "hypothetical proof replay establishes no numerical effect or arriving-data acceptance",
    ),
)
def replay_z_transport_proposal(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    execution: Execution | None = None,
) -> TransportConsumption:
    """Independently replay a portable hypothetical zTR study proposal.

    The frozen failure is re-derived under the caller's identification limits
    before the hypothetical proof is rebound.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes):
        raise CausalTypeError("proposal artifact must be bytes")
    call_native(
        execution,
        _replay_z_transport_proposal,
        artifact,
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        cancel="cancel",
        memory="memory_bytes",
    )
    return TransportConsumption.from_checked_mapping("z_transport_proposal", {})


def reload_lowered_expression(
    identification: TransportIdentification,
) -> tuple[str, str, tuple[tuple[str, int | None], ...], tuple[int, ...]]:
    """Reload a lowered arena from its artifact JSON and return display/bindings."""

    if identification.expr_wire_json is None or identification.expr_root is None:
        raise CausalValueError("identification has no lowered expression to reload")
    pretty, latex, populations, regimes, free = _roundtrip_expr_arena(
        identification.expr_wire_json,
        identification.expr_root,
    )
    return pretty, latex, tuple(zip(populations, regimes, strict=True)), tuple(free)


def reload_lowered_program(identification: TransportIdentification) -> Any:
    """Return a bounded, checked executable program for an identified expression.

    The returned native program owns the validated arena and semantic schema.
    Exact evaluation requires an explicit catalog and provider laws; this
    function does not infer or fetch them.
    """
    if not isinstance(identification, TransportIdentification):
        raise CausalTypeError("reload_lowered_program requires a TransportIdentification")
    if identification._native is None or not identification.transportable:
        raise CausalValueError("identification has no checked expression to reload")
    return identification._native.reload_checked_program()


def restore_lowered_program(identification: TransportIdentification, wire_json: str) -> Any:
    """Independently import and verify a checked program against its identification."""
    if not isinstance(identification, TransportIdentification):
        raise CausalTypeError("restore_lowered_program requires a TransportIdentification")
    if identification._native is None or not identification.transportable:
        raise CausalValueError("identification has no checked expression to reload")
    return identification._native.load_checked_program(wire_json)


def estimate_trial_effect(
    identification: TransportIdentification,
    treatment: Sequence[bool],
    outcome: Sequence[float],
    trial: Sequence[bool],
    selection_probability: Sequence[float],
    treatment_probability: Sequence[float],
    *,
    mu0: Sequence[float] | None = None,
    mu1: Sequence[float] | None = None,
) -> TrialTransportEstimate:
    """Estimate a trial-to-target binary-treatment contrast by IPW and optional AIPW.

    This is an unlicensed utility: it calls the
    trial-to-target estimator directly and returns bare numbers, with no study,
    contract, export or calibration slot. ``antecedent.analyze(data, graph=Admg,
    query=TransportQuery(..., trial=, selection_probability=,
    treatment_probability=))`` is the licensed, study-retaining path.

    ``identification`` must be the result of :func:`identify` for this query.
    Identification and estimation stay separate operations, but the estimator
    now requires the certificate: when ``identification.transportable`` is
    ``False`` (a :class:`NotCertifiedCertificate`, i.e. ``identify``
    returned ``NotCertified``), this raises ``CausalEstimateError`` instead of
    returning a number for a quantity that was never shown to be identified.
    A refusal is a conservative "no rule applies", not a proof of
    non-transportability — see :class:`NotCertifiedCertificate`.

    The positional order is ``(treatment, outcome)``, matching Antecedent's
    scalar causal-query convention.
    """

    if not isinstance(identification, TransportIdentification):
        raise CausalTypeError(
            "estimate_trial_effect requires identification=transport.identify(...)"
        )
    if (mu0 is None) != (mu1 is None):
        raise CausalValueError("mu0 and mu1 must be supplied together")
    raw = _estimate_trial_transport(
        identification._native,
        np.asarray(outcome, dtype=np.float64),
        list(treatment),
        list(trial),
        np.asarray(selection_probability, dtype=np.float64),
        np.asarray(treatment_probability, dtype=np.float64),
        mu0=None if mu0 is None else np.asarray(mu0, dtype=np.float64),
        mu1=None if mu1 is None else np.asarray(mu1, dtype=np.float64),
    )
    return TrialTransportEstimate(
        raw.rule,
        raw.ipw,
        raw.aipw,
        TransportOverlapReport(
            OverlapDiagnostic(
                raw.selection_probability_min,
                raw.selection_probability_max,
                raw.selection_effective_sample_size,
                raw.selection_extreme_weight_count,
            ),
            OverlapDiagnostic(
                raw.treatment_probability_min,
                raw.treatment_probability_max,
                raw.treatment_effective_sample_size,
                raw.treatment_extreme_weight_count,
            ),
        ),
    )
