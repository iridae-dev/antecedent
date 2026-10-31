"""High-level estimation entry points."""

from __future__ import annotations

import importlib
import json
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

import numpy as np

from . import _estimate_helpers
from ._api import describe_refusal
from ._batch import (
    BatchResult,
    BatchRetarget,
    CandidateScreen,
    CellFamily,
    PreparedBatch,
    RetargetClaim,
    RetargetContrast,
    RetargetCost,
    RetargetMember,
    SharedBatchDesign,
    SimultaneousBandMember,
    SimultaneousInterval,
    analyze_many,
    estimate_many,
    prepare_many,
)
from ._coerce import coerce_latency, coerce_query, coerce_refute
from ._columns import (
    AnalysisColumns,
    IdentifierBinding,
    analysis_columns,
    drop_columns,
    encode_identifier,
    encode_weights,
    identifier_strings,
    resolve_estimator_identifiers,
)
from ._data import (
    as_columns,
    ingest_columns,
    project_columns,
    read_columns,
    table_column_names,
    try_as_arrow_c_columns,
)
from ._estimate_helpers import (
    _not_applicable,
    _unwrap_estimator,
)
from ._from_raw import (
    _execution_slots,
    _interval_bindings_from_raw,
    _missing_data_from_raw,
    _wrap_ate,
)
from ._mediation_summary import mediation_effects_summary
from ._native import (
    MediationEffectsSummary,
    _identify_ate,
    _identify_ate_admg,
)
from ._study_lifecycle import guard_design, replay_declaration, retain, retain_preparation
from ._table_file import coerce_table_source
from ._temporal_design import TemporalDesignQuery
from ._treatment_labels import TreatmentLabels
from ._validate import count as _validate_count
from .descriptive import Association
from .discovery import (
    DbnPosterior,
    ExactDagPosterior,
    GraphPosterior,
)
from .errors import (
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from .execution import Execution
from .experiment import (
    ComplierEffect,
    RandomizedEffect,
    SwitchbackEffect,
    TreatmentOnTreated,
)
from .graph import Admg, Dag, TemporalCpdag, TemporalPag
from .ids import Estimator, Identifier, Latency, Refute
from .inference import (
    Bayesian,
    ClassPrior,
    Frequentist,
)
from .interference import (
    InterferenceQuery,
)
from .intervention_programs import PolicyProgramResult, PolicyProgramValue
from .missing import MissingPolicy, coerce_missing, native_declaration
from .policy import (
    ConditionalDoseResponse,
    MultiActionPolicyValue,
    PolicyValue,
)
from .population import coerce_target_population
from .quasi import (
    AugmentedPanelDiD,
    CovariateAdjustedDiD,
    FuzzyRegressionDiscontinuity,
    IvBounds,
    PanelDifferenceInDifferences,
    PeriodHistoryMatch,
    RegressionKink,
    SharpRegressionDiscontinuity,
    StaggeredAdoption,
    StaggeredRollout,
    SyntheticControl,
    SyntheticDifferenceInDifferences,
)
from .query import (
    AnomalyAttribution,
    AverageDerivative,
    AverageEffect,
    ChangeAttribution,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionalDistribution,
    InterventionResponse,
    MediationEffect,
    NestedCounterfactual,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalConditionalPulse,
    TemporalConditionalSustained,
    TemporalMediationEffect,
    TemporalQuery,
)
from .regimes import LongitudinalRegime, SequentialContrast
from .results import (
    AnalysisResult,
    CausalResponseView,
    ConflictSummaryView,
    EffectEnvelope,
    EstimateView,
    IdentificationView,
    InspectionReport,
    MediationView,
    PerformanceView,
    PhysicalPlanView,
    PlanView,
    PosteriorView,
    PredictiveCheckReport,
    PriorSensitivityReport,
    ProgramPlanView,
    ReasoningSlots,
    RefutationReport,
    ResponseEnvelopeView,
    ResponseUncertainty,
    ResponseValidationCheck,
    ResponseValidationView,
    ResponseView,
    SupportDiagnostic,
    SupportReport,
    ValidationView,
)
from .results.response import IntervalInterpretation, SupportStatus, UncertaintyKind
from .survival import (
    CompetingRisksOutcome,
    SurvivalOutcome,
)
from .transport import (
    Transport,
    TransportInference,
)
from .transport._impl import (
    ExactTransportDistribution,
    StatisticalTransportDistribution,
    TransportResponseGrid,
    TransportStage,
)
from .transport._impl.trial import LearnedTrialEstimate, _TransportLimits
from .transport.advanced import (
    ExactTransportQuery,
    StatisticalTransportQuery,
    TransportQuery,
    TransportResponseGridQuery,
    TrialAipwQuery,
)

# Compatibility alias consumed by the internal routing modules.
_static_edges = _estimate_helpers._static_edges

if TYPE_CHECKING:
    from ._prepared_policy import PreparedPolicyAnalysis
    from .preflight import (
        CostEstimate,
        NuisanceFitDiagnostics,
        PreflightReport,
        RankDropPlan,
    )


_RESPONSE_FAMILY = (
    ResponseCurve,
    InterventionResponse,
    AverageDerivative,
    PointDerivative,
    Elasticity,
    SemiElasticity,
    DirectionalDerivative,
    ResponseJacobian,
)


_SAMPLING_DESIGNS = (
    "independent_subject_trajectories",
    "clustered_panel",
    "dependent_time_series",
)


def _bayesian_inference_kwargs(inference: Bayesian) -> dict[str, Any]:
    backend = str(inference.backend).strip().lower()
    if backend == "laplace":
        inference_s = "bayesian"
    elif backend == "conjugate":
        inference_s = "conjugate"
    elif backend == "hmc":
        inference_s = "hmc"
    else:
        raise CausalValueError(
            f"unknown Bayesian backend {inference.backend!r}; use laplace|conjugate|hmc"
        )
    likelihood = str(inference.likelihood).strip().lower()
    if likelihood not in ("gaussian", "logit", "probit", "poisson"):
        raise CausalValueError(
            f"unknown Bayesian likelihood {inference.likelihood!r}; "
            "use gaussian|logit|probit|poisson",
            reason_code="invalid_argument",
        )
    kw: dict[str, Any] = {
        "inference": inference_s,
        "prior_scale": inference.prior_scale,
        "likelihood": likelihood,
    }
    if inference.n_draws_explicit:
        kw["n_draws"] = inference.n_draws
    if inference.draw_controls is not None:
        if inference.draw_controls != "measured":
            raise CausalValueError(
                f"unknown Bayesian draw_controls {inference.draw_controls!r}; use 'measured'",
                reason_code="invalid_argument",
            )
        kw["draw_controls"] = inference.draw_controls
    if inference.predictive_replicates is not None:
        kw["predictive_replicates"] = (
            inference.predictive_replicates.n_sims,
            inference.predictive_replicates.max_rows,
        )
    if inference.white_innovation is not None:
        max_lag, level = inference.white_innovation
        if (
            isinstance(max_lag, bool)
            or not isinstance(max_lag, int)
            or max_lag < 1
            or not np.isfinite(level)
            or not 0.0 < level < 1.0
        ):
            raise CausalValueError(
                "white_innovation is (max_lag >= 1, level in (0, 1))",
                reason_code="invalid_argument",
            )
        kw["white_innovation"] = (max_lag, float(level))
    if inference.design_model is not None:
        kw["design_model"] = inference.design_model._wire()
    prior_from = inference.prior_from
    if prior_from is not None:
        # Local import avoids circular import with priors ↔ estimation.
        from .priors import ComposedPrior

        if isinstance(prior_from, ComposedPrior):
            kw["composed_prior"] = prior_from.to_native_dict()
        else:
            kw["prior_artifact"] = bytes(prior_from)
    if inference.mapping is not None:
        kw["prior_mapping"] = inference.mapping.to_dict()
    return kw


@dataclass(frozen=True)
class IdentifyResult:
    """Identify-only result (no estimate), retaining its certified query."""

    status: str
    method: str
    adjustment_set: list[str]
    query: Any = None


def identify(
    *,
    graph: Dag | Admg | Sequence[tuple[str, str]],
    query: AverageEffect,
    names: Sequence[str] | None = None,
    identifier: str | Identifier | None = None,
) -> IdentifyResult:
    """Identify without estimating.

    Accepts a ``Dag``, an ``Admg``, or an edge list. Pass ``names`` with an edge
    list (variable order); with a typed graph the names come from
    ``graph.nodes()``.

    ``identify(TieredBackground)`` stays Rust-only
    (``identify_tiered`` / ``identify_tiered_joint``). Prepared identification
    already returns the certificate; this entry does not accept a tier rule.
    Pair-family joint cells use :func:`prepare_many` with joint response queries.

    Prefer an ``Admg`` whenever a confounder is unmeasured. A ``Dag`` has no way
    to say a variable cannot be observed, so a latent common cause flattened
    into one is treated as an ordinary adjustable node and the effect is
    reported identified by adjusting on something no study can measure.
    ``Dag.latent_project(observed)`` produces the ``Admg`` for that graph.
    """
    if isinstance(identifier, Identifier):
        identifier = str(identifier)
    if isinstance(graph, Admg):
        status, method, adjustment = _identify_ate_admg(
            list(graph.nodes()),
            graph,
            query.treatment,
            query.outcome,
            identifier=identifier,
        )
        return IdentifyResult(
            status=status, method=method, adjustment_set=list(adjustment), query=query
        )
    if isinstance(graph, Dag):
        node_names = list(graph.nodes())
        edges = list(graph.edges())
    else:
        if names is None:
            raise CausalValueError("identify(edge_list) requires names=")
        node_names = list(names)
        edges = list(graph)
    status, method, adjustment = _identify_ate(
        node_names,
        edges,
        query.treatment,
        query.outcome,
        identifier=identifier,
    )
    return IdentifyResult(
        status=status, method=method, adjustment_set=list(adjustment), query=query
    )


def _response_support_bounds(raw: Any) -> dict[str, tuple[float, float]]:
    """Map native support minima/maxima onto axis names.

    Static curves align one interval per treatment name. Temporal dose × horizon
    surfaces report a multi-axis query region wider than the treatment list.
    """
    minima = list(getattr(raw, "support_minima", ()) or ())
    maxima = list(getattr(raw, "support_maxima", ()) or ())
    treatments = list(getattr(raw, "treatments", ()) or ())
    if len(minima) == len(treatments):
        return {
            name: (lower, upper)
            for name, lower, upper in zip(treatments, minima, maxima, strict=True)
        }
    axis_names = (
        ["dose", "horizon"] if len(minima) == 2 else [f"axis_{i}" for i in range(len(minima))]
    )
    return {
        name: (lower, upper) for name, lower, upper in zip(axis_names, minima, maxima, strict=True)
    }


def _horizon_adjustment_sets(raw: Any) -> tuple[tuple[str, ...], ...] | None:
    sets = tuple(
        tuple(str(name) for name in group)
        for group in (getattr(raw, "horizon_adjustment_sets", ()) or ())
    )
    return sets or None


def _support_point_status(raw: Any) -> tuple[SupportStatus, ...] | None:
    """Native empty vec means a static curve with no per-cell support grid."""
    cells = tuple(getattr(raw, "support_point_status", ()) or ())
    return tuple(cast(SupportStatus, status) for status in cells) if cells else None


def _wrap_prepared_response(
    raw: Any,
    query: ResponseCurve | InterventionResponse | None = None,
    prepared: Any | None = None,
) -> CausalResponseView:
    """Build a :class:`CausalResponseView` from a prepared-response native DTO."""
    execution = None if prepared is None else prepared._native.snapshot()
    slots = _execution_slots(execution)
    from typing import cast

    response = (
        ResponseView(
            treatments=raw.treatments, outcomes=raw.outcomes, points=raw.points, values=raw.values
        )
        if raw.points and raw.values
        else None
    )
    temporal = bool(getattr(query, "is_temporal", False))
    identifier = getattr(raw, "identifier", None)
    validation: ResponseValidationView | None
    if identifier == "generalized.adjustment":
        method = "generalized.adjustment"
        identify_op = "identify.generalized_adjustment"
        validation = None
    elif temporal:
        method = "temporal.backdoor.unfolded"
        identify_op = "identify.temporal_backdoor"
        validation = ResponseValidationView(
            checks=(
                ResponseValidationCheck(
                    id="refute.temporal_response.skipped",
                    status="skipped",
                    statistic=None,
                    threshold=None,
                    detail="scalar ATE refuters are not applicable to a function-valued temporal response",
                ),
            )
        )
    else:
        method = "response.backdoor"
        identify_op = "identify.response"
        validation = None
    # Refuter states that ran on this response (derivative, temporal or class-aware
    # intervention level, all unlicensed) replace the "skipped" placeholder.
    reports = raw.refutation_reports() if hasattr(raw, "refutation_reports") else []
    if reports:
        import math

        scoped_checks = execution is not None and getattr(
            execution, "response_checks_are_scoped", lambda: False
        )()
        validation = ResponseValidationView(
            whole_target=not scoped_checks,
            checks=tuple(
                ResponseValidationCheck(
                    id=name,
                    status="passed" if passed else "failed",
                    statistic=float(refuted) if math.isfinite(refuted) else None,
                    threshold=None,
                    detail=("completion-level check; " if scoped_checks else "")
                    + (failure or ("informative check" if informative else "uninformative check")),
                )
                for name, passed, informative, _original, refuted, failure in reports
            )
        )
    certificate_json = getattr(raw, "certificate_json", None)
    envelope = None
    if getattr(raw, "identified_mass", None) is not None:
        if raw.lower is None or raw.upper is None:
            raise RuntimeError("native structural response omitted its identified envelope")
        envelope = ResponseEnvelopeView(
            treatments=raw.treatments,
            outcomes=raw.outcomes,
            points=raw.points,
            lower=raw.lower,
            upper=raw.upper,
            identified_mass=float(raw.identified_mass),
            unidentified_mass=float(raw.unidentified_mass),
            completion_count=int(raw.completion_count),
            truncated_completions=int(raw.truncated_completions),
            enumeration_capped=bool(raw.enumeration_capped),
            mass_scope=cast(Literal["full_class", "examined_completions"], raw.mass_scope),
            weight_basis=cast(
                Literal[
                    "posterior_probability",
                    "completion_enumeration",
                    "caller_supplied_class_prior",
                ],
                raw.weight_basis,
            ),
            aggregation_policy=cast(
                Literal[
                    "same_estimand_weighted_mean",
                    "identified_set_envelope",
                    "graph_dependent_atoms",
                ]
                | None,
                getattr(raw, "aggregation_policy", None),
            ),
            atom_keys=tuple(raw.atom_keys),
            atom_weights=tuple(raw.atom_weights),
            atom_statuses=tuple(raw.atom_statuses),
            atom_values=tuple(tuple(values) for values in raw.atom_values),
            unevaluable_mass=float(getattr(raw, "unevaluable_mass", None) or 0.0),
            subsampled_out_mass=float(getattr(raw, "subsampled_out_mass", None) or 0.0),
        )
    basis_json = raw.program_basis_json()
    native_basis = {} if basis_json is None else json.loads(basis_json)
    # ResultModel accepts private retained handles outside its public Pydantic fields.
    return CausalResponseView(  # type: ignore[call-arg]
        certificate=json.loads(certificate_json) if certificate_json else None,
        estimand=query,
        response=response,
        estimate=raw.scalar if raw.scalar is not None else raw.matrix,
        uncertainty=ResponseUncertainty(
            kind=cast(UncertaintyKind, raw.uncertainty_kind),
            lower=raw.lower,
            upper=raw.upper,
            level=raw.level,
            standard_error=raw.standard_error,
            replicates=raw.replicates,
            artifact_id=raw.artifact_id,
            interpretation=cast(
                IntervalInterpretation | None, getattr(raw, "interval_interpretation", None)
            ),
        ),
        support=SupportReport(
            status=cast(SupportStatus, raw.support_status),
            query_region=_response_support_bounds(raw),
            diagnostics=[
                SupportDiagnostic(id=identifier, values=values, detail=detail, scope=scope)
                for identifier, values, detail, scope in zip(
                    raw.diagnostic_ids,
                    raw.diagnostic_values,
                    raw.diagnostic_details,
                    raw.diagnostic_scopes,
                    strict=True,
                )
            ],
            warnings=raw.warnings,
            point_status=_support_point_status(raw),
        ),
        identification=IdentificationView(
            status=raw.identification,
            method=method,
            adjustment_set=list(getattr(raw, "adjustment_set", ())),
            assumption_count=len(raw.assumptions),
            derivation_step_count=0,
            horizon_adjustment_sets=_horizon_adjustment_sets(raw),
        ),
        assumptions=raw.assumptions,
        provenance={
            "operation_id": raw.provenance_id,
            "operation_ids": [identify_op, raw.provenance_id],
        },
        envelope=envelope,
        validation=validation,
        evidence_status=getattr(raw, "evidence_status", None),
        allowlist_reason=getattr(raw, "allowlist_reason", None),
        allowlist_parent=getattr(raw, "allowlist_parent", None),
        diagnostics=tuple(getattr(raw, "diagnostics", ()) or ()),
        missing_data=_missing_data_from_raw(raw, prepared),
        interval_bindings=_interval_bindings_from_raw(raw),
        _prepared=prepared,
        _execution=execution,
        _raw=raw,
        reasoning=slots,
        program_id=native_basis.get("program_id") if slots is None else slots.program_id,
        claim_id=None if slots is None else slots.claim_id,
        data_snapshot_id=native_basis.get("snapshot_id")
        if slots is None
        else slots.data_snapshot_id,
    )


def _prepared_columns(data: Any) -> tuple[list[str], list[Any], bool]:
    from ._table_snapshot import Table

    if isinstance(data, Table):
        names, columns = data._native_columns()
        return names, columns, False
    arrow = try_as_arrow_c_columns(data)
    if arrow is not None:
        names, columns = arrow
        return names, columns, True
    names, columns = as_columns(data)
    return names, columns, False


_PreparedQuery = (
    TemporalQuery
    | TemporalConditionalPulse
    | TemporalConditionalSustained
    | AverageEffect
    | ResponseCurve
    | ConditionalEffect
    | PathSpecificEffect
    | InterventionalDistribution
    | InterventionResponse
    | PulseEffect
    | SustainedEffect
    | MediationEffect
    | NestedCounterfactual
    | Counterfactual
    | PointDerivative
    | Elasticity
    | SemiElasticity
    | AverageDerivative
    | DirectionalDerivative
    | ResponseJacobian
    | TemporalMediationEffect
    | TransportQuery
    | Transport
    | ExactTransportQuery
    | StatisticalTransportQuery
    | TransportResponseGridQuery
    | TrialAipwQuery
    | InterferenceQuery
    | RandomizedEffect
    | ComplierEffect
    | TreatmentOnTreated
    | SwitchbackEffect
    | PolicyValue
    | ConditionalDoseResponse
    | MultiActionPolicyValue
    | AnomalyAttribution
    | ChangeAttribution
    | PanelDifferenceInDifferences
    | AugmentedPanelDiD
    | StaggeredAdoption
    | SyntheticControl
    | SyntheticDifferenceInDifferences
    | StaggeredRollout
    | CovariateAdjustedDiD
    | FuzzyRegressionDiscontinuity
    | RegressionKink
    | SharpRegressionDiscontinuity
    | SurvivalOutcome
    | CompetingRisksOutcome
    | LongitudinalRegime
    | SequentialContrast
    | PeriodHistoryMatch
    | IvBounds
    | Association
    | PolicyProgramValue
    | TemporalDesignQuery
)


@dataclass(frozen=True)
class _Controls:
    """Execution controls a study applies to every click unless overridden.

    In-process only: none of them is part of the contract, program, or claim.
    """

    cancel: Any | None = None
    on_progress: Any | None = None
    on_stage: Any | None = None

    def kwargs(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in (
                ("cancel", self.cancel),
                ("on_progress", self.on_progress),
                ("on_stage", self.on_stage),
            )
            if value is not None
        }


_UNSET: Any = object()


def _frame_payload(
    data: Any, columns: AnalysisColumns | None = None
) -> tuple[list[str], list[Any], dict[str, Any] | None]:
    """Columns plus the panel / multi-environment / event frame the natives rebuild.

    ``columns`` are the declared analysis columns of a typed-graph table: only those
    reach the numeric ingest, chosen by name.
    """
    from .data import EventFrame, MultiEnvFrame, PanelFrame

    if isinstance(data, EventFrame):
        return (
            list(data.names),
            list(data.columns),
            {
                "kind": "events",
                "event_times_ns": data._index,
                "align_interval_ns": int(data.align_interval_ns),
            },
        )
    if isinstance(data, PanelFrame):
        return (
            list(data.names),
            [],
            {
                "kind": "panel",
                "unit_columns": [list(cols) for cols in data.unit_columns],
                "unit_ids": [int(i) for i in data.unit_ids],
            },
        )
    if isinstance(data, MultiEnvFrame):
        return (
            list(data.names),
            [],
            {"kind": "multi_env", "env_columns": [list(cols) for cols in data.env_columns]},
        )
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
        # A sequence of tables is multi-environment data (the J-PCMCI+ spelling).
        from ._data import as_multi_env_columns

        names, env_columns = as_multi_env_columns(list(data))
        return names, [], {"kind": "multi_env", "env_columns": env_columns}
    if columns is not None:
        data = project_columns(
            data, columns.names, subset=not columns.ordered, what="graph or query"
        )
    names, arrays = ingest_columns(data)
    return names, arrays, None


def _panel_did_payload(
    data: Any, query: PanelDifferenceInDifferences
) -> tuple[list[str], list[Any], dict[str, Any]]:
    """Keep panel IDs as query metadata and send only numeric outcomes to Rust."""
    from .quasi import _binary

    required: tuple[str, ...] = (query.outcome, query.subject, query.treated, query.post)
    if query.cluster is not None:
        required += (query.cluster,)
    raw = read_columns(data, required, what="panel DiD")
    subjects = identifier_strings(raw[query.subject], f"subject={query.subject!r}")
    clusters = (
        subjects
        if query.cluster is None
        else identifier_strings(raw[query.cluster], f"cluster={query.cluster!r}")
    )
    if any(not value.strip() for value in (*subjects, *clusters)):
        raise CausalValueError("subject and cluster IDs must be non-empty")
    design = {
        "subjects": subjects,
        "clusters": clusters,
        "treated": tuple(_binary(raw[query.treated], query.treated)),
        "post": tuple(_binary(raw[query.post], query.post)),
        "repeated_cross_section": query.sampling == "repeated_cross_section",
    }
    names, columns = ingest_columns({query.outcome: raw[query.outcome]})
    return names, columns, design


def _augmented_panel_did_payload(
    data: Any, query: AugmentedPanelDiD
) -> tuple[list[str], list[Any], dict[str, tuple[Any, ...]]]:
    """Bind subject and cluster ownership while keeping outcome/nuisance columns numeric."""
    from .quasi import _binary

    numeric = (
        query.outcome_pre,
        query.outcome_post,
        query.propensity,
        query.untreated_change_prediction,
    )
    required: tuple[str, ...] = (*numeric, query.subject, query.treated)
    if query.cluster is not None:
        required += (query.cluster,)
    raw = read_columns(data, required, what="augmented panel DiD")
    subjects = tuple(raw[query.subject])
    clusters = subjects if query.cluster is None else tuple(raw[query.cluster])
    if any(not isinstance(value, str) or not value.strip() for value in (*subjects, *clusters)):
        raise CausalValueError("subject and cluster IDs must be non-empty strings")
    if len(set(subjects)) != len(subjects):
        raise CausalValueError("augmented panel DiD requires one unique row per subject")
    design = {
        "subjects": subjects,
        "clusters": clusters,
        "treated": tuple(_binary(raw[query.treated], query.treated)),
    }
    names, columns = ingest_columns({name: raw[name] for name in numeric})
    return names, columns, design


def _staggered_payload(
    data: Any, query: StaggeredAdoption
) -> tuple[list[str], list[Any], dict[str, Any]]:
    from .quasi import _integer_column

    if not query.event_study and (query.target_cohort is None or query.target_period is None):
        raise CausalValueError(
            "analyze with StaggeredAdoption requires target_cohort and target_period"
        )
    required: tuple[str, ...] = (query.outcome, query.subject, query.period, query.cohort)
    if query.cluster is not None:
        required += (query.cluster,)
    raw = read_columns(data, required, what="staggered DiD")
    subjects = identifier_strings(raw[query.subject], f"subject={query.subject!r}")
    clusters = (
        subjects
        if query.cluster is None
        else identifier_strings(raw[query.cluster], f"cluster={query.cluster!r}")
    )
    if any(not value.strip() for value in (*subjects, *clusters)):
        raise CausalValueError("subject and cluster IDs must be non-empty")
    periods = tuple(_integer_column(raw[query.period], "period", minimum=1))
    cohorts = tuple(_integer_column(raw[query.cohort], "cohort", minimum=0))
    design = {
        "subjects": subjects,
        "clusters": clusters,
        "periods": periods,
        "cohorts": cohorts,
        "target_cohort": query.target_cohort,
        "target_period": query.target_period,
    }
    names, columns = ingest_columns({query.outcome: raw[query.outcome]})
    return names, columns, design


def _synthetic_control_payload(
    data: Any, query: SyntheticControl | SyntheticDifferenceInDifferences
) -> tuple[list[str], list[Any], dict[str, Any]]:
    from .quasi import _integer_column

    raw = read_columns(data, (query.outcome, query.unit, query.period), what="synthetic-control")
    units = identifier_strings(raw[query.unit], f"unit={query.unit!r}")
    if any(not value.strip() for value in units):
        raise CausalValueError("unit IDs must be non-empty")
    periods = tuple(_integer_column(raw[query.period], "period", minimum=1))
    names, columns = ingest_columns({query.outcome: raw[query.outcome]})
    return names, columns, {"units": units, "periods": periods}


def _continuous_dose_payload(
    data: Any, query: ConditionalDoseResponse
) -> tuple[list[str], list[Any], dict[str, Any]]:
    required = (query.outcome, query.dose, query.baseline_group, query.dose_density)
    raw = read_columns(data, required, what="continuous-dose")
    groups = tuple(raw[query.baseline_group])
    if any(not isinstance(group, str) or not group for group in groups):
        raise CausalValueError("baseline group labels must be non-empty strings")
    names, columns = ingest_columns(
        {name: raw[name] for name in (query.outcome, query.dose, query.dose_density)}
    )
    return names, columns, {"baseline_groups": groups}


def _longitudinal_regime_payload(
    data: Any, query: LongitudinalRegime
) -> tuple[list[str], list[Any]]:
    """Send the numeric endpoint to Rust; histories remain frozen query metadata."""
    outcome = read_columns(data, (query.outcome,), what="longitudinal outcome")[query.outcome]
    if len(outcome) != len(query.subject_ids):
        raise CausalValueError("longitudinal outcome rows must align with subject histories")
    if query.method in ("sequential_dr", "marginal_structural_model"):
        values = np.asarray(outcome, dtype=np.float64)
        observed = np.asarray(query.outcome_observed, dtype=bool)
        if not np.isfinite(values[observed]).all():
            raise CausalValueError("observed terminal outcomes must be finite")
        outcome = np.where(observed, values, 0.0)
    return ingest_columns({query.outcome: outcome})


def _sequential_contrast_payload(
    data: Any, query: SequentialContrast
) -> tuple[list[str], list[Any], dict[str, Any]]:
    """The numeric columns of the contrast and the fold of every row, frozen at prepare."""
    required = (
        query.outcome,
        query.first_treatment,
        query.second_treatment,
        *query.baseline,
        *query.intermediate,
    )
    raw = read_columns(data, required, what="sequential contrast")
    rows = len(raw[query.outcome])
    if isinstance(query.folds, int):
        fold_of_row = tuple(row % query.folds for row in range(rows))
        fold_count = query.folds
    else:
        fold_of_row = tuple(
            _validate_count(fold, f"folds[{index}]", minimum=0, type_error=CausalValueError)
            for index, fold in enumerate(query.folds)
        )
        if len(fold_of_row) != rows:
            raise CausalValueError("folds must give one fold per row of the data")
        fold_count = max(fold_of_row) + 1
    names, columns = ingest_columns({name: raw[name] for name in required})
    return names, columns, {"fold_of_row": fold_of_row, "fold_count": fold_count}


def _history_match_payload(data: Any, query: PeriodHistoryMatch) -> tuple[list[str], list[Any]]:
    """The numeric columns of the within-period history match."""
    required = [query.outcome, query.treated, *query.history]
    required.extend(name for name in (query.period, query.weight) if name is not None)
    raw = read_columns(data, tuple(required), what="history match")
    return ingest_columns({name: raw[name] for name in required})


def _iv_bounds_payload(data: Any, query: IvBounds) -> tuple[list[str], list[Any]]:
    """The numeric columns of the binary-instrument bounds study."""
    required = [query.outcome, query.treatment, query.instrument]
    if query.weight is not None:
        required.append(query.weight)
    raw = read_columns(data, tuple(required), what="IV bounds")
    return ingest_columns({name: raw[name] for name in required})


def _association_payload(data: Any, query: Association) -> tuple[list[str], list[Any]]:
    """The columns of the noncausal association, in the order the query declares them."""
    required = [query.outcome, *query.predictors, *query.conditioning]
    if query.weight is not None:
        required.append(query.weight)
    raw = read_columns(data, tuple(required), what="association")
    return ingest_columns({name: raw[name] for name in required})


def _inference_wire(inference: Frequentist | Bayesian) -> dict[str, Any]:
    if isinstance(inference, Frequentist):
        return {"mode": "frequentist", "prior_scale": 10.0}
    kw = _bayesian_inference_kwargs(inference)
    if kw.get("prior_mapping") is not None and kw.get("prior_artifact") is None:
        raise CausalUnsupportedError(
            "prior_mapping names which estimand of a prior artifact to read; it has nothing "
            "to map without prior_from=<artifact>",
            reason_code="option_not_applicable",
        )
    wire = {
        "mode": kw.pop("inference"),
        "n_draws": kw.get("n_draws"),
        "prior_scale": kw["prior_scale"],
        "prior_artifact": kw.get("prior_artifact"),
        "prior_mapping": kw.get("prior_mapping"),
        "composed_prior": kw.get("composed_prior"),
        "likelihood": kw["likelihood"],
    }
    if "draw_controls" in kw:
        # Sent only when requested, so routes with a closed inference vocabulary refuse it.
        wire["draw_controls"] = kw["draw_controls"]
    if "white_innovation" in kw:
        wire["white_innovation"] = kw["white_innovation"]
    if "predictive_replicates" in kw:
        wire["predictive_replicates"] = kw["predictive_replicates"]
    if "design_model" in kw:
        wire["design_model"] = kw["design_model"]
    return wire


def _accept_live_discovery(
    data: Any,
    discovery: Any,
    *,
    accept_discovered: bool,
    regimes: Sequence[int] | None,
    seed: int,
    threads: int | None,
    controls: _Controls,
    execution: Execution | None = None,
) -> Any:
    """Run a live discovery config once and accept it through its review gate.

    Cancellation is checked at the discovery boundary and progress reports the
    discovery stage; estimation on the accepted structure honours both fully.
    """
    from .accepted_graph import AcceptedGraph

    token = controls.cancel
    if token is not None and token.is_cancelled():
        from .errors import CausalCancelledError

        raise CausalCancelledError("cancelled before discovery")
    if controls.on_progress is not None:
        controls.on_progress(0.0, "discovery")
    from .discovery import RPCMCI

    # Only RPCMCI labels regimes; prepare refuses `regimes=` for every other config.
    labelled = {"regimes": regimes} if isinstance(discovery, RPCMCI) else {}
    accepted = discovery.accept(
        data,
        execution=execution
        if execution is not None
        else Execution(seed=seed, threads=threads, cancel=token),
        accept_discovered=accept_discovered,
        **labelled,
    )
    if controls.on_progress is not None:
        controls.on_progress(1.0, "discovery")
    if token is not None and token.is_cancelled():
        from .errors import CausalCancelledError

        raise CausalCancelledError("cancelled after discovery")
    return accepted if isinstance(accepted, AcceptedGraph) else AcceptedGraph(accepted)


_GRAPH_POSTERIOR_TYPES = (ExactDagPosterior, DbnPosterior, GraphPosterior)

# Loaded after `_static_edges`, `_RESPONSE_FAMILY`, and `_not_applicable` exist.
# A top-level import would cycle: `_prepare_route` imports those helpers back.
_PrepareRoute = importlib.import_module("._prepare_route", __package__)._PrepareRoute


def _check_response_observation(query: Any, inference: Any) -> None:
    """Observation mechanism and its assumptions on a response.

    A declared target population is licensed by the Rust study builder, which
    refuses it with ``population_not_estimable`` on every response route.
    """
    from .observation import Complete

    if query.observation is not None and not isinstance(query.observation, Complete):
        if isinstance(inference, Bayesian) and not getattr(query, "is_temporal", False):
            raise CausalUnsupportedError("these response cells require complete observations")
        if not getattr(query, "is_temporal", False) and isinstance(query, InterventionResponse):
            raise CausalUnsupportedError("these response cells require complete observations")
    if query.observation_assumptions and (
        query.observation is None or isinstance(query.observation, Complete)
    ):
        raise CausalUnsupportedError(
            "observation_assumptions require an explicit observation mechanism"
        )


_PreparedResult = (
    AnalysisResult
    | CausalResponseView
    | ExactTransportDistribution
    | StatisticalTransportDistribution
    | TransportResponseGrid
    | LearnedTrialEstimate
    | PolicyProgramResult
)

ResultT = TypeVar("ResultT", covariant=True)


class _PreparedEngine(Generic[ResultT]):
    """Compile-once / re-estimate-many handle for licensed analysis cells.

    **Frozen at prepare:** schema (names, types, order); graph, accepted graph,
    or licensed graph-posterior atoms and weights; query identity; identifier;
    estimator and its configuration; validation suite, custom validators and
    replicate / draw budgets; observation / transport / interference
    assumptions; panel, multi-environment, or event data layout.

    **Estimate click:** same-schema data (or the retained data); seeds /
    threads; the study's execution controls (``cancel`` / ``on_progress`` /
    ``on_stage``), which a click may override. Does not re-identify or
    recompile the logical plan.

    **Refute click:** AverageEffect and scalar Dag ``InterventionResponse``.
    :meth:`refute` raises ``CausalUnsupportedError``, prefixed with the
    matrix's own wire id for the cell: ``not_applicable:`` for ResponseCurve
    (function-valued), ``refused:`` for temporal / class-aware
    InterventionResponse (not built; the refusal names the next attempt).
    ConditionalEffect and Frequentist Dag InterventionResponse cheap/full run
    the scalar ATE-shaped suite; Bayesian Dag InterventionResponse cheap/full
    run the Bayesian refuter state (plugin-level suite on the posterior-mean
    level plus predictive checks and, under full, prior sensitivity), which is
    unlicensed.

    **Re-prepare required:** any frozen field change, including schema mismatch.

    Arrow CDI / stream inputs may be borrowed without copying. Keep their
    exported buffers unchanged while this handle retains them, including any
    writable NumPy aliases backing Arrow arrays. To bind other observations, use
    new backing buffers and ``rebind``; ordinary mappings of NumPy
    columns are copied during ingest.

    ``analyze`` is ``prepare(...).estimate()``; its result retains this handle
    as ``result.study``. For streaming append + incremental OLS, use
    :class:`antecedent.CausalState`.
    """

    def __init__(
        self,
        native: Any,
        *,
        kind: Literal[
            "average",
            "response_curve",
            "intervention_response",
            "exact_transport",
            "statistical_transport",
            "transport_grid",
            "learned_trial",
            "z_transport",
            "policy",
        ] = "average",
        query: _PreparedQuery | None = None,
        display_query: Any = None,
        seed: int = 1,
        threads: int | None = None,
        controls: _Controls | None = None,
        deferred_suite: str | None = None,
        snapshot_data: Any = None,
        design_columns: dict[str, Any] | None = None,
        named_design_columns: frozenset[str] = frozenset(),
        column_names: tuple[str, ...] | None = None,
    ) -> None:
        self._native = native
        self._kind = kind
        from ._transport_lifecycle import transport_lifecycle

        self._transport = transport_lifecycle(kind)
        self._query = query
        self._display_query = query if display_query is None else display_query
        # A scalar Dag InterventionResponse runs its refuter suite as the
        # second click of every estimate, so `refute=` is honoured on the
        # prepared route exactly as it is on a one-shot analyze.
        self._deferred_suite = deferred_suite
        self._snapshot_data = snapshot_data
        self._design_columns = design_columns
        # Data columns the query named for its design inputs (see antecedent._columns).
        # They were read and dropped at prepare; a later click drops them again so the
        # same table can be passed back, but never re-reads the frozen design.
        self._named_design_columns = named_design_columns
        # The columns the native study was prepared on, in its order: a later table is
        # reduced to them by name before ingest, so extra columns and a different
        # column order never reach the native side. ``None`` for a study without one
        # table schema (frames, design routes).
        self._column_names = column_names
        self._identifier_bindings: dict[str, IdentifierBinding] = {}
        # Declared treatment labels of a label-coded binary treatment; later tables are
        # encoded by the same declaration before ingest.
        self._treatment_labels: TreatmentLabels | None = None
        self._seed = seed
        self._threads = threads
        self._controls = controls or _Controls()
        self._cancelled = False
        # Set by transport/_day1.py::prepare_transport when identification is
        # deferred (no native execution yet): the frozen inputs needed to
        # re-derive identification (inspect()) or re-bind data (rebind())
        # without one. Absent for every non-transport study.
        self._transport_stage: TransportStage | None = None
        self._last_result: ResultT | None = None
        self._batch_family = False
        self._identification_view: Any = None

    def _frozen(self, execution: Any) -> _PreparedEngine[ResultT]:
        """A handle over one retained execution with this study's seed and controls."""
        frozen: _PreparedEngine[ResultT] = _PreparedEngine(
            execution,
            kind=self._kind,
            query=self._query,
            display_query=self._display_query,
            seed=self._seed,
            threads=self._threads,
            controls=self._controls,
            design_columns=self._design_columns,
            named_design_columns=self._named_design_columns,
            column_names=self._column_names,
        )
        for name in (
            "_batch_family",
            "_identifier_bindings",
            "_treatment_labels",
            "_preparation_request",
            "_named_design_bindings",
            "_declared_query",
            "_resolved_design_query",
        ):
            if hasattr(self, name):
                setattr(frozen, name, getattr(self, name))
        return frozen

    @property
    def participants(self) -> tuple[str, ...] | None:
        """Original participant label of every row of the prepared table.

        ``None`` unless ``participant=`` named a column at prepare. The labels are the
        column's own values (as text), in the table's original row order, frozen at prepare
        and checked against every refreshed table.
        """
        for binding in self._identifier_bindings.values():
            if binding.row_labels:
                return binding.row_labels
        return None

    @property
    def validator_names(self) -> tuple[str, ...]:
        """Attested custom-validator names frozen at prepare (their claim identity)."""
        if self._transport is not None:
            return ()
        return tuple(self._native.validator_names())

    def rebind_validators(self, validators: Mapping[str, Any]) -> None:
        """Re-bind caller custom validators by their exact attested names.

        Validators are in-process callbacks: a claim carries their attested
        results, never the callables. The names must equal
        :attr:`validator_names`; otherwise the rebind is refused.
        """
        if self._transport is not None:
            if validators:
                raise ValueError("Transport studies do not fit or invoke sampled-data validators")
            return
        self._native.rebind_validators(dict(validators))

    @classmethod
    @retain_preparation
    @describe_refusal
    def prepare(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        query: _PreparedQuery,
        graph: Dag | Sequence[tuple[str, str]] | Any | None = None,
        discovery: Any | None = None,
        inference: Frequentist | Bayesian | TransportInference | None = None,
        identifier: str | Identifier | None = None,
        estimator: str | Estimator | Any | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int | None = None,
        latency: Latency | Literal["interactive", "standard", "report"] | None = None,
        class_prior: ClassPrior | None = None,
        max_completions: int | None = None,
        population_registry: Any | None = None,
        cancel: Any | None = None,
        on_progress: Any | None = None,
        on_stage: Any | None = None,
        validators: Sequence[Any] | Mapping[str, Any] | None = None,
        accept_discovered: bool = True,
        regimes: Sequence[int] | None = None,
        provider: Any | None = None,
        controls: _TransportLimits | None = None,
        participant: str | None = None,
        missing: str | MissingPolicy | None = None,
        sampling_design: Literal[
            "independent_subject_trajectories", "clustered_panel", "dependent_time_series"
        ]
        | None = None,
        data_diagnostics: Any = True,
        retain_joint_horizons: bool = False,
        on_refusal: Literal["refuse", "describe"] = "refuse",
        evidence_candidates: Sequence[Any] | None = None,
    ) -> _PreparedEngine[_PreparedResult] | PreparedPolicyAnalysis:
        """Compile a durable plan for a licensed analysis cell.

        Every argument either reaches the compiled study or is refused with a
        reason code; none is dropped. Omitted ``refute`` / ``bootstrap`` /
        ``latency`` reach the Rust study builder as omitted, so its one
        omitted-default table, latency-tier mapping and refute downgrade apply
        (``antecedent._defaults.OMITTED`` reads that table).

        Supports ``AverageEffect``, ``ResponseCurve``, ``ConditionalEffect``,
        ``PathSpecificEffect``, ``InterventionalDistribution``,
        ``InterventionResponse``, ``PulseEffect``, ``SustainedEffect``,
        ``TemporalMediationEffect``, static ``MediationEffect``,
        ``Counterfactual``, and the six Frequentist derivative query types
        on an explicit graph or accepted wrapper.
        ``AverageEffect`` also prepares on a ``Pag`` or bidirected ``Admg``; the
        generalized-adjustment envelope or general-ID result is frozen at
        prepare and reused by every estimate click (``exec.identify.cached``).
        Temporal ``ResponseCurve`` / ``InterventionResponse`` (keyword ``horizons``)
        prepare on a ``TemporalDag`` or lagged edge list; their Frequentist
        ``bootstrap`` is the joint circular-block replicate count behind the
        pointwise and simultaneous bands. An omitted ``bootstrap`` takes the
        ``latency`` tier's replicate count from the omitted-default table
        (``antecedent._native.omitted_defaults()``), as Pulse / Sustained do;
        ``0`` publishes the point surface with no band and an
        ``estimate.temporal_response.band_withheld`` warning. Every other
        prepared response refuses a positive ``bootstrap``. Pulse / Sustained /
        TemporalMediation prepare on a ``TemporalDag`` or lagged edge list; a
        Frequentist ``TemporalMediationEffect`` with ``bootstrap=None`` follows the
        same ``latency`` tier for its shared circular-block replicates.
        ``discovery=ExactDagPosterior()`` / ``DbnPosterior()`` / a constructed
        ``GraphPosterior`` compiles the licensed graph-posterior cells
        (AverageEffect / ConditionalEffect / ResponseCurve / one-coordinate
        InterventionResponse on DAG, CPDAG, PAG, and ADMG atoms; Pulse /
        Sustained / TemporalMediationEffect / temporal ResponseCurve /
        one-coordinate InterventionResponse on a DBN or temporal-class
        posterior). Frequentist DBN mixtures
        take their shared circular-block replicates from the ``latency`` tier
        (or an explicit ``bootstrap``).

        ``graph=`` takes a ``Dag`` / ``Cpdag`` / ``Pag`` / ``Admg`` /
        ``TieredBackground`` / ``TemporalDag`` / ``TemporalCpdag`` /
        ``TemporalPag``, an edge list, or an :class:`antecedent.AcceptedGraph`.
        ``discovery=`` also takes a live configuration (``PC`` / ``GES`` /
        ``LiNGAM`` / ``NOTEARS`` / ``FCI`` / ``RFCI`` / ``PCMCI`` /
        ``PCMCIPlus`` / ``LPCMCI`` / ``JPCMCIPlus`` / ``RPCMCI``), which runs
        once and is accepted through its review gate (``accept_discovered=False``
        raises ``ReviewRequired`` while anything is pending).

        ``data`` may be a table, a series, or an ``antecedent.data`` panel /
        multi-environment / event frame; frames compile on the matching Rust
        data modality for every query it licenses and are refused otherwise.

        ``cancel`` / ``on_progress`` apply to prepare-time identification and to
        every estimate click; ``on_stage`` streams the stages of each click on
        routes that have them. The handle retains all three; a click may
        override them.

        ``PolicyProgramValue(programs, outcome, n_draws=...)`` with ``graph=Dag(...)`` (or an
        edge list) prepares a policy question: the structural model is fitted on ``data`` once,
        the program is compiled and the licence of each route resolved once, and every
        ``estimate`` reruns the policy value without compiling the program again
        (``handle.preparation`` is the receipt). Its result carries the support decision, the
        contract and the provenance record, and equals ``policy_value`` on the same model and
        seed. Estimand-only options (``discovery``, ``estimator``, ``refute``, ...) are refused.

        ``missing=`` declares how missing values are handled (a policy name or an
        :class:`antecedent.missing.MissingPolicy`). Average-effect routes keep their
        complete-case selection by default and report the rows it dropped; randomized,
        panel and temporal routes refuse a missing value until a policy is declared. A
        policy the route cannot execute is refused when the study is built, and every
        result carries ``missing_data`` (policy, assumption and row accounting).

        ``sampling_design=`` declares once how the rows were sampled:
        ``"independent_subject_trajectories"`` (resampled and split by whole subject, never by
        row), ``"clustered_panel"`` (by whole cluster) or ``"dependent_time_series"``
        (contiguous blocks, split with an embargo). Every estimator of the study that
        bootstraps or splits is held to it, and a request the design does not allow, such as a
        row bootstrap under a trajectory design, is refused with a sampling-design error. The
        declaration is part of the study's data identity. A name that is not one of the three
        raises :class:`CausalValueError`. A ``"clustered_panel"`` or ``"dependent_time_series"``
        declaration with the bootstrap off is honoured only where the analytic error reads it: an
        average effect through a regression, weighting or AIPW estimator whose ``se`` is the
        cluster, multiway or panel-HAC error (panel) or Newey-West or panel-HAC error (series).
        Any other estimator or query is refused (``option_not_applicable``), as are matching and
        instrumental estimators, which read no cluster or serial dependence.

        ``retain_joint_horizons=True`` keeps, for a temporal ``ResponseCurve`` or
        ``InterventionResponse``, the replicates that refit every horizon together (the joint
        block-bootstrap replicates under Frequentist inference, which needs ``bootstrap=``, or
        the block-weight posterior draws under Bayesian inference). The result's
        ``horizon_draws(dose_index)`` reads them back, so a quantity built from several horizons
        keeps their dependence. Any other query refuses the option.
        """
        data = coerce_table_source(data)
        if retain_joint_horizons and not (
            isinstance(query, (ResponseCurve, InterventionResponse))
            and getattr(query, "is_temporal", False)
        ):
            raise CausalUnsupportedError(
                "retain_joint_horizons= keeps the replicates coupled across the horizons of a "
                "temporal ResponseCurve or InterventionResponse; this query has no horizons",
                reason_code="option_not_applicable",
            )
        from .transport import ExactTransportData, StatisticalTransportData
        from .transport._day1 import prepare_transport, refuse_transport_only_kwargs
        from .transport._impl import (
            ExactTransportQuery,
            StatisticalTransportQuery,
            TransportResponseGridQuery,
        )

        refuse_transport_only_kwargs(
            query, provider=provider, inference=inference, controls=controls
        )
        if on_refusal not in ("refuse", "describe"):
            raise CausalValueError(
                f"on_refusal={on_refusal!r} is not a refusal handling; use 'refuse' or 'describe'",
                reason_code="invalid_argument",
            )
        if on_refusal != "describe" and evidence_candidates is not None:
            raise CausalUnsupportedError(
                "evidence_candidates= feed the descriptive fallback; pass on_refusal='describe' "
                "to run it or drop the candidates",
                reason_code="option_not_applicable",
            )
        if (on_refusal != "refuse" or evidence_candidates is not None) and (
            isinstance(
                query,
                (
                    PolicyProgramValue,
                    Transport,
                    TransportResponseGridQuery,
                    StatisticalTransportQuery,
                    ExactTransportQuery,
                ),
            )
            or isinstance(data, (ExactTransportData, StatisticalTransportData))
        ):
            raise CausalUnsupportedError(
                "on_refusal='describe' runs a descriptive fallback on a refused tabular study; "
                "this query is bound to its own design or evidence and has no such fallback",
                reason_code="option_not_applicable",
            )
        if isinstance(query, PolicyProgramValue):
            from ._prepared_policy import PreparedPolicyAnalysis

            return PreparedPolicyAnalysis.build(
                data,
                query=query,
                graph=graph,
                seed=seed,
                threads=threads,
                accept_discovered=accept_discovered,
                options={
                    "discovery": discovery,
                    "inference": inference,
                    "identifier": identifier,
                    "estimator": estimator,
                    "estimator_config": estimator_config,
                    "refute": refute,
                    "bootstrap": bootstrap,
                    "latency": latency,
                    "class_prior": class_prior,
                    "max_completions": max_completions,
                    "participant": participant,
                    "missing": missing,
                    "sampling_design": sampling_design,
                    "population_registry": population_registry,
                    "cancel": cancel,
                    "on_progress": on_progress,
                    "on_stage": on_stage,
                    "validators": validators,
                    "regimes": regimes,
                    "provider": provider,
                    "controls": controls,
                },
            )
        if participant is not None and (
            isinstance(
                query,
                (
                    Transport,
                    TransportResponseGridQuery,
                    StatisticalTransportQuery,
                    ExactTransportQuery,
                ),
            )
            or isinstance(data, (ExactTransportData, StatisticalTransportData))
        ):
            raise CausalUnsupportedError(
                "participant= labels the rows of a prepared table; a transport study is bound "
                "to an evidence catalog or distributional providers, which have no table rows "
                "to carry the labels",
                reason_code="option_not_applicable",
            )
        if sampling_design is not None and sampling_design not in _SAMPLING_DESIGNS:
            raise CausalValueError(
                f"sampling_design={sampling_design!r} is not a declared design; use one of "
                f"{', '.join(_SAMPLING_DESIGNS)}",
                reason_code="invalid_argument",
            )
        if sampling_design is not None and (
            isinstance(
                query,
                (
                    Transport,
                    TransportResponseGridQuery,
                    StatisticalTransportQuery,
                    ExactTransportQuery,
                ),
            )
            or isinstance(data, (ExactTransportData, StatisticalTransportData))
        ):
            raise CausalUnsupportedError(
                "sampling_design= declares how table rows were sampled; a transport study is "
                "bound to an evidence catalog or distributional providers, which have no table "
                "rows",
                reason_code="option_not_applicable",
            )
        missing_policy = coerce_missing(missing)
        if missing_policy is not None and (
            isinstance(
                query,
                (
                    Transport,
                    TransportResponseGridQuery,
                    StatisticalTransportQuery,
                    ExactTransportQuery,
                ),
            )
            or isinstance(data, (ExactTransportData, StatisticalTransportData))
        ):
            raise CausalUnsupportedError(
                "missing= declares how rows with missing values are handled; a transport study "
                "is bound to an evidence catalog or distributional providers, which have no "
                "table rows",
                reason_code="option_not_applicable",
            )
        # A design-carrying query may name its row-aligned inputs by data column;
        # resolve those from the raw data (any dtype) and drop them before the numeric
        # ingest. No-op when the caller passed the arrays inline.
        from ._columns import collect_column_names
        from ._columns import resolve_query as _resolve_query_columns

        named_design_columns = frozenset(collect_column_names(query))
        query, data = _resolve_query_columns(query, data)
        if isinstance(query, Transport):
            if not isinstance(graph, Admg):
                raise CausalTypeError("transport.Transport requires graph=Admg(...)")
            if isinstance(inference, (Frequentist, Bayesian)):
                raise CausalUnsupportedError(
                    "transport.Transport uses inference=TransportInference(...); "
                    "Frequentist/Bayesian do not apply",
                    reason_code="option_not_applicable",
                )
            if identifier not in (None, "transport.sid"):
                raise CausalUnsupportedError(
                    "transport.Transport is identified by transport.sid; "
                    "another identifier= does not apply",
                    reason_code="option_not_applicable",
                )
            if (
                any(
                    option is not None
                    for option in (
                        discovery,
                        estimator,
                        estimator_config,
                        refute,
                        bootstrap,
                        threads,
                        latency,
                        class_prior,
                        max_completions,
                        population_registry,
                        on_progress,
                        on_stage,
                        validators,
                        regimes,
                    )
                )
                or not accept_discovered
            ):
                raise CausalUnsupportedError(
                    "transport.Transport uses provider=, inference=TransportInference, "
                    "and query-owned evaluation bounds; physical controls belong to Execution",
                    reason_code="option_not_applicable",
                )
            return prepare_transport(
                data,
                query=query,
                graph=graph,
                provider=provider,
                inference=inference,
                controls=controls,
                cancel=cancel,
                seed=seed,
            )

        if isinstance(
            query, (ExactTransportQuery, StatisticalTransportQuery, TransportResponseGridQuery)
        ) or isinstance(data, (ExactTransportData, StatisticalTransportData)):
            if not isinstance(
                query, (ExactTransportQuery, StatisticalTransportQuery, TransportResponseGridQuery)
            ):
                raise CausalTypeError(
                    "transport providers require a typed exact, statistical, or grid query"
                )
            if (
                any(
                    option is not None
                    for option in (
                        graph,
                        discovery,
                        identifier,
                        estimator,
                        estimator_config,
                        refute,
                        bootstrap,
                        threads,
                        latency,
                        class_prior,
                        max_completions,
                        population_registry,
                        on_progress,
                        on_stage,
                        validators,
                        regimes,
                    )
                )
                or not accept_discovered
            ):
                raise CausalUnsupportedError(
                    "typed transport queries retain their graph, evidence and scientific declaration",
                    reason_code="option_not_applicable",
                )
            if inference is not None and not isinstance(inference, TransportInference):
                raise CausalTypeError("transport inference requires TransportInference")
            from .transport._day1 import _evaluation_limits
            from .transport._impl.prepare import compile_modality

            limits = controls or _TransportLimits(
                **_evaluation_limits(query.evaluation), cancel=cancel
            )
            return compile_modality(
                query, data, provider=provider, inference=inference, controls=limits, seed=seed
            )

        from .population import registry_wire

        display_query = query
        query = coerce_query(query)
        # Label-declared treatment levels are mapped to the arm codes of the numeric
        # declared-level path here, once; the mapping is frozen with the study.
        treatment_labels = TreatmentLabels.of(query, data=data)
        if treatment_labels is not None:
            data = treatment_labels.encode(data)
            query = treatment_labels.native_query(query)
        if isinstance(identifier, Identifier):
            identifier = str(identifier)
        estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
        # A named cluster column is read from the raw data here and bound to its original
        # rows; the estimator aligns it to the rows it analyses.
        estimator_config, identifier_bindings = resolve_estimator_identifiers(
            estimator, estimator_config, data
        )
        participant_binding: IdentifierBinding | None = None
        if participant is not None:
            participant_binding, _ = encode_identifier(
                data, participant, what="participant", keep_rows=True
            )
            identifier_bindings[participant] = participant_binding
            # The labels are bound to their rows; the column is not an analysis variable and
            # must not reach the numeric ingest of a route that reads every column.
            columns = table_column_names(data)
            if columns is not None and participant in columns:
                data = project_columns(data, [c for c in columns if c != participant])
        if latency is not None:
            latency = str(coerce_latency(latency))  # type: ignore[assignment]
        # The native side takes a suite name or False; the parameter keeps its
        # richer public type.
        suite: str | bool | None = None
        if refute is not None:
            coerced = coerce_refute(refute)
            suite = str(coerced) if isinstance(coerced, Refute) else coerced
        if bootstrap is not None and (
            isinstance(bootstrap, bool)
            or not isinstance(bootstrap, numbers.Integral)
            or bootstrap < 0
        ):
            raise CausalValueError("bootstrap must be a non-negative integer or None")
        inference = inference or Frequentist()
        if not isinstance(inference, (Frequentist, Bayesian)):
            raise CausalTypeError("inference must be Frequentist or Bayesian")
        execution_controls = _Controls(cancel=cancel, on_progress=on_progress, on_stage=on_stage)
        if isinstance(query, (ResponseCurve, InterventionResponse)):
            _check_response_observation(query, inference)

        from .accepted_graph import AcceptedGraph
        from .discovery import RPCMCI

        live_discovery = discovery is not None and not isinstance(discovery, _GRAPH_POSTERIOR_TYPES)
        if isinstance(graph, AcceptedGraph) and discovery is not None:
            raise CausalUnsupportedError(
                "graph=AcceptedGraph(...) rejects discovery=; the structure artifact is "
                "already accepted (call rediscover() explicitly to replace it)"
            )
        if graph is not None and discovery is not None:
            raise CausalValueError("prepare rejects both graph= and discovery=")
        if regimes is not None and not isinstance(discovery, RPCMCI):
            raise _not_applicable("regimes", "a discovery other than RPCMCI")
        if not accept_discovered and not live_discovery:
            raise _not_applicable(
                "accept_discovered=False", "a study without live discovery (nothing to review)"
            )
        if live_discovery and latency == "interactive":
            raise CausalUnsupportedError(
                "discovery= is not on the interactive estimate path; run discovery once "
                "(Config.accept(data) -> AcceptedGraph), then prepare(graph=..., "
                "latency='interactive')",
                reason_code="option_not_applicable",
            )
        if live_discovery:
            graph = _accept_live_discovery(
                data,
                discovery,
                accept_discovered=accept_discovered,
                regimes=regimes,
                seed=seed,
                threads=threads,
                controls=execution_controls,
            )
            discovery = None
        structure_accepted = isinstance(graph, AcceptedGraph)
        discovery_algorithm = None
        if structure_accepted:
            accepted_graph = cast(AcceptedGraph, graph)
            discovery_algorithm = accepted_graph.algorithm_id
            if accepted_graph.background_knowledge is not None:
                # The statements the discovery ran under are part of what the claim rests on:
                # the same graph found under other knowledge is another procedure's result.
                discovery_algorithm = (
                    f"{discovery_algorithm}+knowledge:{accepted_graph.background_knowledge}"
                )
            graph = accepted_graph.graph
        if class_prior is not None and (
            not isinstance(graph, (TemporalCpdag, TemporalPag))
            or not isinstance(inference, Bayesian)
        ):
            raise CausalUnsupportedError(
                "class_prior requires Bayesian inference on TemporalCpdag or TemporalPag",
                reason_code="option_not_applicable",
            )
        if max_completions is not None and not isinstance(graph, (TemporalCpdag, TemporalPag)):
            raise _not_applicable("max_completions", "a structure without class completions")

        design_columns = None
        table_names: tuple[str, ...] | None = None
        from ._temporal_design import TEMPORAL_DESIGN_QUERIES, temporal_design_payload

        if isinstance(query, TEMPORAL_DESIGN_QUERIES):
            names, columns, design_columns = temporal_design_payload(data, query)
            frame = None
        elif isinstance(query, (StaggeredRollout, CovariateAdjustedDiD)):
            from ._quasi_rollout import covariate_adjusted_payload, staggered_rollout_payload

            names, columns, design_columns = (
                staggered_rollout_payload(data, query)
                if isinstance(query, StaggeredRollout)
                else covariate_adjusted_payload(data, query)
            )
            frame = None
        elif isinstance(
            query,
            (
                PanelDifferenceInDifferences,
                AugmentedPanelDiD,
                StaggeredAdoption,
                SyntheticControl,
                SyntheticDifferenceInDifferences,
            ),
        ):
            if isinstance(query, AugmentedPanelDiD):
                names, columns, design_columns = _augmented_panel_did_payload(data, query)
            elif isinstance(query, StaggeredAdoption):
                names, columns, design_columns = _staggered_payload(data, query)
            elif isinstance(query, (SyntheticControl, SyntheticDifferenceInDifferences)):
                names, columns, design_columns = _synthetic_control_payload(data, query)
            else:
                names, columns, design_columns = _panel_did_payload(data, query)
            frame = None
        elif isinstance(query, ConditionalDoseResponse):
            names, columns, design_columns = _continuous_dose_payload(data, query)
            frame = None
        elif isinstance(query, LongitudinalRegime):
            names, columns = _longitudinal_regime_payload(data, query)
            frame = None
        elif isinstance(query, SequentialContrast):
            names, columns, design_columns = _sequential_contrast_payload(data, query)
            frame = None
        elif isinstance(query, PeriodHistoryMatch):
            names, columns = _history_match_payload(data, query)
            frame = None
        elif isinstance(query, IvBounds):
            names, columns = _iv_bounds_payload(data, query)
            frame = None
        elif isinstance(query, Association):
            names, columns = _association_payload(data, query)
            frame = None
        else:
            names, columns, frame = _frame_payload(
                data, analysis_columns(query, graph, estimator_config, missing=missing_policy)
            )
            table_names = tuple(names) if frame is None else None
        predicates, distributions = registry_wire(population_registry)
        population = coerce_target_population(getattr(query, "target_population", None))
        options: dict[str, Any] = {
            "refute": suite,
            "bootstrap": bootstrap,
            "latency": latency,
            "validators": validators,
            "target_population": population,
            "population_predicates": predicates or None,
            "population_distributions": distributions or None,
            "cancel": cancel,
            "on_progress": on_progress,
            "inference": _inference_wire(inference),
            "discovery_algorithm": discovery_algorithm,
        }
        from .data_diagnostics import _setting_json

        options["data_diagnostics"] = _setting_json(data_diagnostics)
        if retain_joint_horizons:
            options["retain_joint_horizons"] = True
        if on_refusal == "describe":
            # A study refused for want of identification keeps its refusal; Rust runs the
            # descriptive fallback on that same study and attaches it to the raised refusal.
            from .fallback import fallback_declaration

            options["fallback"] = fallback_declaration(evidence_candidates)
        if missing_policy is not None:
            options["missing"] = native_declaration(missing_policy, names)
        if sampling_design is not None:
            options["sampling_design"] = sampling_design
        if (query_weights := getattr(query, "weights", None)) is not None:
            # A design query that declares sampling weights: one weight per original row,
            # frozen like a cluster column so a refreshed table with other weights is refused.
            declared, weight_binding = encode_weights(data, query_weights, estimator=None)
            options["sampling_weights"] = declared
            if weight_binding is not None:
                identifier_bindings[weight_binding.column] = weight_binding
        if participant_binding is not None:
            if table_names is None:
                raise CausalUnsupportedError(
                    "participant= labels the rows of one prepared table; this route (frames and "
                    "design queries) reads its units from the design instead",
                    reason_code="option_not_applicable",
                )
            options["participants"] = participant_binding.contract_binding()
        route = _PrepareRoute(
            names=names,
            columns=columns,
            frame=frame,
            query=query,
            graph=graph,
            discovery=discovery,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=suite,
            bootstrap=bootstrap,
            threads=threads,
            seed=seed,
            class_prior=class_prior,
            max_completions=max_completions,
            accepted=structure_accepted,
            options=options,
            design_columns=design_columns,
        )
        native, kind = route.compile()
        prepared = cls(
            native,
            kind=kind,
            query=query,
            display_query=display_query,
            seed=seed,
            threads=threads,
            controls=execution_controls,
            deferred_suite=route.deferred_suite,
            snapshot_data=data if route.deferred_suite else None,
            design_columns=design_columns,
            named_design_columns=named_design_columns,
            column_names=table_names,
        )
        prepared._identifier_bindings = identifier_bindings
        prepared._treatment_labels = treatment_labels
        if on_stage is not None and not native.streams_stages():
            raise CausalUnsupportedError(
                "on_stage streams identify, estimate_point, uncertainty and validate for one "
                "identification and one scalar effect whose point is fitted before its "
                "uncertainty; this route fits its point and uncertainty together or mixes "
                "several identifications, so it has no such stages (use on_progress)",
                reason_code="stage_stream_unavailable",
            )
        return cast(_PreparedEngine[_PreparedResult], prepared)

    def export_artifact(
        self,
        *,
        artifact_id: str = "prepared-result",
        payload: Literal["query", "result"] = "result",
    ) -> bytes:
        """Export the last full posterior/response, or its query, without refitting.

        ``payload="query"`` also supports functional path/distribution queries.

        Decode Bayesian scalar results with ``inference.decode_posterior_artifact``;
        decode response results with ``artifacts.loads``. Posterior artifacts retain
        draws, quantity names, identification and backend metadata. Response
        artifacts also retain support and assumptions; retain the analysis result
        separately for posterior assumptions and validation reports.
        """
        if self._transport is not None:
            if payload != "result":
                raise ValueError(
                    "Transport artifacts export a checked execution and its full proof"
                )
            return self._native.export()
        return self._native.export_artifact(artifact_id=artifact_id, payload=payload)

    def export(self, *, artifact_id: str = "analysis-result") -> bytes:
        """Export the last execution as a contracted ``analysis_result``, or refuse
        if no claim was produced."""
        if self._transport is not None:
            # A transport handle retains its previous claim through a cancelled
            # click (every snapshot change is atomic) and refuses on its own when it
            # holds none, so the native export decides.
            return self._native.export()
        if getattr(self, "_cancelled", False):
            raise CausalUnsupportedError(
                "Cancelled estimate produced no claim.",
                reason_code="cancelled_no_claim",
            )
        return self._native.export_contracted_artifact(artifact_id=artifact_id)

    @property
    def structure_source(self) -> str:
        """Support-matrix structure axis frozen at prepare (`explicit` or `accepted`)."""
        return str(self._native.plan_summary().get("structure_source", "explicit"))

    def plan_view(self) -> ProgramPlanView:
        """Every layer of the program this handle executes, read before any estimate.

        The view states the scientific problem, the identification product
        (status, assumptions, estimand and adjustment set, unresolved
        obligations, retained products), the statistical program, the evidence
        and decision program, the physical plan (operators, quoted design width
        or why there is none, byte estimates) and the uncertainty and claim the
        execution must publish, with the lowered program identity and the
        support cell. ``operation`` names the sealed operation preparation
        retained and carries its inspection record. Reading it identifies, fits
        and scans nothing; :meth:`inspect` and :meth:`preflight` are unchanged.
        """
        native = getattr(self, "_native", None)
        transport = getattr(self, "_transport", None)
        if transport is not None or native is None or not hasattr(native, "plan_view_json"):
            raise CausalUnsupportedError(
                "This handle executes no lowered study program, so it has no plan view.",
                reason_code="plan_view_unavailable",
            )
        return ProgramPlanView.from_json(native.plan_view_json())

    def report(self, format: Literal["markdown", "html"] = "markdown") -> str:
        """A readable report of this handle, as Markdown or one self-contained HTML page.

        The report restates the records the handle holds: after an estimate, the
        executed claim (its kind, value, domains, calibration slot and evidence) with
        the contract it ran under; before one, the prepared contract alone. It carries
        the identification, the assumptions and obligations, the diagnostics, the
        uncertainty axes, the support cell, the refusals and limits the operation
        report names, and the plan view of the program.

        The report computes nothing: every number is a value one of those records
        holds (printed to at most twelve significant digits), the wording is fixed
        text chosen by the records' status tags, and the same records render to the
        same bytes. The HTML carries no script, loads no resource and escapes every
        value. Rendering identifies, fits and scans nothing; it neither runs an
        estimate nor changes this handle.
        """
        if format not in ("markdown", "html"):
            raise CausalValueError(f"unknown report format {format!r}; use 'markdown' or 'html'")
        native = getattr(self, "_native", None)
        transport = getattr(self, "_transport", None)
        if transport is not None or native is None or not hasattr(native, "report"):
            raise CausalUnsupportedError(
                "This handle executes no lowered study program, so it has no plan view to "
                "report from.",
                reason_code="plan_view_unavailable",
            )
        return str(native.report(format))

    @property
    def evidence_status(self) -> str | None:
        """`licensed` or `allowed_unlicensed`, or ``None`` if the query is off-axis."""
        raw = self._native.plan_summary().get("evidence_status")
        return str(raw) if raw is not None else None

    @property
    def allowlist_reason(self) -> str | None:
        raw = self._native.plan_summary().get("allowlist_reason")
        return str(raw) if raw is not None else None

    @property
    def allowlist_parent(self) -> str | None:
        raw = self._native.plan_summary().get("allowlist_parent")
        return str(raw) if raw is not None else None

    @property
    def last_result(self) -> ResultT | None:
        """Latest successful execution; failures preserve the previous result."""
        return self._last_result

    @property
    def identification(self) -> Any:
        """The retained identification product; reading it performs no fitting."""
        if self._identification_view is None:
            from .identify import Identification

            if self._transport is not None:
                from .transport._day1 import identification_from_transport

                stage: Any = self._transport_stage or {}
                self._identification_view = identification_from_transport(
                    cast(Any, stage.get("graph")),
                    cast(Any, self._query),
                    catalog=stage.get("catalog"),
                    identified=stage.get("identified"),
                )
            else:
                payload = json.loads(self._native.identification_json())
                self._identification_view = Identification.from_prepared(
                    payload,
                    query=self._display_query,
                    graph=getattr(self, "_preparation_request", {}).get("graph"),
                )
        return self._identification_view

    @describe_refusal
    def reprepare(
        self, data: Any, *, query: Any = _UNSET
    ) -> _PreparedEngine[Any] | PreparedPolicyAnalysis:
        """Resolve the original declarations against a new table into a new study."""
        request = replay_declaration(getattr(self, "_preparation_request", {}))
        if not request:
            raise CausalValueError(
                "this consumed study has no original preparation request; use antecedent.prepare"
            )
        if query is not _UNSET:
            request["query"] = query
        return type(self).prepare(data, **request)

    def inspect(self) -> InspectionReport:
        """Inspect the latest successful execution, or the prepared plan before execution."""
        if self._last_result is not None:
            report = cast(Any, self._last_result).inspect()
            return report.model_copy(update={"stage": "executed"})
        return self.inspect_plan()

    def __repr__(self) -> str:
        return f"PreparedAnalysis(query={type(self._display_query).__name__}, stage={'executed' if self._last_result is not None else 'prepared'})"

    def _repr_html_(self) -> str:
        return self.inspect()._repr_html_()

    def inspect_plan(self) -> InspectionReport:
        """Everything known about this study, including cached identification.

        ``to_dict()`` gives the structured report; its ``contract`` field is the
        study's domain-separated identities and reasoning slots (ADR 0022), and
        its ``calibration`` is unavailable (``not_executed``) until an estimate
        runs. :meth:`preflight` is the cheap structural-only view.
        """
        if self._kind == "z_transport":
            from .transport._day1 import identification_from_transport

            query = self._query
            if not isinstance(query, Transport):
                raise CausalValueError("transport inspect requires a Transport query")
            stage = getattr(self, "_transport_stage", None) or {}
            graph = stage.get("graph")
            if graph is None:
                raise CausalValueError("transport inspect requires a retained graph")
            return identification_from_transport(
                graph,
                query,
                catalog=stage.get("catalog"),
                identified=stage.get("identified"),
            ).inspect()
        if self._transport is not None:
            if getattr(self, "_native", None) is None:
                from .transport._day1 import identification_from_transport

                query = self._query
                if not isinstance(query, Transport):
                    raise CausalValueError("transport inspect requires a Transport query")
                stage = getattr(self, "_transport_stage", None) or {}
                graph = stage.get("graph")
                if graph is None:
                    raise CausalValueError("transport inspect requires a retained graph")
                return identification_from_transport(
                    graph,
                    query,
                    catalog=stage.get("catalog"),
                    identified=stage.get("identified"),
                ).inspect()
            return InspectionReport(**json.loads(self._native.inspection_json()))

        from dataclasses import replace

        from .results._execution import Answer, CalibrationInfo
        from .results._report import as_inspection

        report = as_inspection(
            replace(
                ReasoningSlots.from_contract(dict(self._native.contract())),
                answer=Answer("unavailable", detail="not_executed"),
                calibration=CalibrationInfo(status="unavailable", reason="not_executed"),
            )
        )
        identification = self.identification
        updates: dict[str, Any] = {"stage": "prepared", "details": identification.to_dict()}
        if identification.design is not None:
            updates["identification"] = identification.inspect().identification
        return report.model_copy(update=updates)

    def preflight(self) -> InspectionReport:
        """Cheap structural-only inspection; identification and fitting are not run."""
        from .results._report import as_inspection

        if self._transport is not None:
            return self.inspect()
        return as_inspection(ReasoningSlots.from_contract(self._native.inspect()))

    def diagnose(self, *, seed: int = 1, threads: int | None = None) -> PreflightReport:
        """Fit-free preflight of the retained table and certified adjustment set.

        Reports complete-case and arm counts, missingness, exact duplicate columns, the
        numerical rank of the ``[1 | Z]`` design with the dependent columns named, and
        review flags for near-deterministic relations (never a change to the adjustment
        set). Nothing is fit; :meth:`diagnose_fit` is the check that fits a propensity
        model. Unlike :meth:`preflight`, which inspects the plan's structure, this reads
        the data.
        """
        from .preflight import diagnose_prepared

        return diagnose_prepared(self._native, seed=seed, threads=threads)

    def diagnose_fit(self, *, seed: int = 1, threads: int | None = None) -> NuisanceFitDiagnostics:
        """Fit a diagnostic propensity model and report its score range and arm ESS.

        Not preflight: the fit is part of the evidence. A fit that fails before any score
        exists reports every fitted quantity as absent.
        """
        from .preflight import diagnose_fit_prepared

        return diagnose_fit_prepared(self._native, seed=seed, threads=threads)

    def plan_rank_drop(
        self,
        priority: Sequence[str] | None = None,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> RankDropPlan:
        """Opt-in rank-deficiency drop plan under a declared column priority (a record,
        not an execution). See :func:`antecedent.preflight.plan_rank_drop`."""
        from .preflight import plan_rank_drop_prepared

        return plan_rank_drop_prepared(self._native, priority, seed=seed, threads=threads)

    def estimate_cost(self) -> CostEstimate:
        """Planning-time counts (nuisance fits, folds, bootstrap replicates, design bytes)
        and the active inference default. A planning hint, not a runtime guarantee; no
        seconds are estimated."""
        from .preflight import estimate_cost_prepared

        return estimate_cost_prepared(self._native)

    def preview_transform(self, intent: str) -> dict[str, str]:
        """Pure preview of a transformation of this study; nothing is re-executed.

        ``intent`` is one of ``display_precision``, ``filter_display``,
        ``compatible_data_replace``, ``retarget``, ``filter_population``,
        ``new_conditional_query``, ``change_graph``, ``change_prior``,
        ``change_physical_policy`` or ``average_unweighted_class``. The report
        names the intent, whether it is refused, its obligations, and the frozen
        input identities it would carry under ``input_<domain>`` keys.
        """
        return dict(self._native.preview_transform(intent))

    @property
    def plan(self) -> PhysicalPlanView:
        """Physical-plan summary retained from prepare."""
        raw = self._native.plan_summary()
        return PhysicalPlanView(
            plan_id=str(raw.get("plan_id", "")),
            estimated_peak_memory_bytes=(
                int(raw["estimated_peak_memory_bytes"])
                if "estimated_peak_memory_bytes" in raw
                else None
            ),
            workspace_bytes=(int(raw["workspace_bytes"]) if "workspace_bytes" in raw else None),
            batch_size=int(raw["batch_size"]) if "batch_size" in raw else None,
            worker_threads=int(raw.get("worker_threads", 0)),
            expected_python_crossings=int(raw.get("expected_python_crossings", 0)),
            deterministic_reductions=str(raw.get("deterministic_reductions", "true")).lower()
            in ("1", "true"),
            kernels=raw.get("kernels") or None,
        )

    def _click_controls(self, cancel: Any, on_progress: Any, on_stage: Any) -> dict[str, Any]:
        """The study's retained controls, with any per-click override applied."""
        controls = self._controls
        chosen = _Controls(
            cancel=controls.cancel if cancel is _UNSET else cancel,
            on_progress=controls.on_progress if on_progress is _UNSET else on_progress,
            on_stage=controls.on_stage if on_stage is _UNSET else on_stage,
        )
        return chosen.kwargs()

    def _run_click(self, call: Any) -> Any:
        from .errors import CausalCancelledError

        try:
            raw = call()
        except CausalCancelledError:
            # The handle still holds the previous execution; it must not export as this one.
            self._cancelled = True
            raise
        self._cancelled = False
        return raw

    def _prepared_table(self, data: Any) -> Any:
        """``data`` without the named design columns, reduced to the prepared schema by name."""
        if self._treatment_labels is not None:
            data = self._treatment_labels.encode(data)
        data = drop_columns(data, self._named_design_columns)
        if self._column_names is None:
            return data
        return project_columns(data, self._column_names, what="prepared")

    def _click_payload(self, data: Any) -> tuple[list[str], list[Any], dict[str, Any] | None]:
        guard_design(self, data)
        data = self._prepared_table(data)
        query = self._query
        if isinstance(query, ConditionalDoseResponse):
            names, columns, design = _continuous_dose_payload(data, query)
            if design != self._design_columns:
                raise CausalUnsupportedError(
                    "continuous-dose refresh requires prepared baseline-group row order",
                    reason_code="invalid_argument",
                )
            return names, columns, None
        from ._temporal_design import TEMPORAL_DESIGN_QUERIES, temporal_design_payload

        if isinstance(query, TEMPORAL_DESIGN_QUERIES):
            names, columns, design = temporal_design_payload(data, query)
            if design != self._design_columns:
                raise CausalUnsupportedError(
                    "a temporal-design refresh requires the prepared units, periods and "
                    "recorded assignment law",
                    reason_code="invalid_argument",
                )
            return names, columns, None
        if isinstance(query, (StaggeredRollout, CovariateAdjustedDiD)):
            from ._quasi_rollout import covariate_adjusted_payload, staggered_rollout_payload

            names, columns, design = (
                staggered_rollout_payload(data, query)
                if isinstance(query, StaggeredRollout)
                else covariate_adjusted_payload(data, query)
            )
            if design != self._design_columns:
                raise CausalUnsupportedError(
                    "design refresh requires the prepared unit and period row order",
                    reason_code="invalid_argument",
                )
            return names, columns, None
        if isinstance(
            query,
            (
                PanelDifferenceInDifferences,
                AugmentedPanelDiD,
                StaggeredAdoption,
                SyntheticControl,
                SyntheticDifferenceInDifferences,
            ),
        ):
            names, columns, design = (
                _augmented_panel_did_payload(data, query)
                if isinstance(query, AugmentedPanelDiD)
                else _staggered_payload(data, query)
                if isinstance(query, StaggeredAdoption)
                else _synthetic_control_payload(data, query)
                if isinstance(query, (SyntheticControl, SyntheticDifferenceInDifferences))
                else _panel_did_payload(data, query)
            )
            if design != self._design_columns:
                raise CausalUnsupportedError(
                    "design refresh requires the prepared unit and period row order",
                    reason_code="invalid_argument",
                )
            return names, columns, None
        if isinstance(query, LongitudinalRegime):
            names, columns = _longitudinal_regime_payload(data, query)
            return names, columns, None
        if isinstance(query, SequentialContrast):
            names, columns, design = _sequential_contrast_payload(data, query)
            if design != self._design_columns:
                raise CausalUnsupportedError(
                    "sequential contrast refresh requires the prepared fold assignment",
                    reason_code="invalid_argument",
                )
            return names, columns, None
        if isinstance(query, PeriodHistoryMatch):
            names, columns = _history_match_payload(data, query)
            return names, columns, None
        if isinstance(query, IvBounds):
            names, columns = _iv_bounds_payload(data, query)
            return names, columns, None
        if isinstance(query, Association):
            names, columns = _association_payload(data, query)
            return names, columns, None
        if isinstance(query, ResponseCurve) and query.observation is not None:
            from .observation import Complete, _ensure_latent_schema_column

            if not isinstance(query.observation, Complete):
                names, columns = ingest_columns(data)
                names, columns = _ensure_latent_schema_column(names, columns, query.observation)
                return names, columns, None
        return _frame_payload(data)

    def _wrap(
        self, raw: Any
    ) -> (
        AnalysisResult
        | CausalResponseView
        | StatisticalTransportDistribution
        | TransportResponseGrid
    ):
        if self._kind in ("response_curve", "intervention_response"):
            target = self._query.query if isinstance(self._query, TemporalQuery) else self._query
            query = target if isinstance(target, _RESPONSE_FAMILY) else None
            return _wrap_prepared_response(raw, query=cast(Any, query), prepared=self)
        return _wrap_ate(raw, query=self._display_query, prepared=self)

    def _click(self, data: Any, **kwargs: Any) -> Any:
        data = coerce_table_source(data)
        guard_design(self, data)
        result = self._execute_click(data, **kwargs)
        self._last_result = cast(ResultT, result)
        return result

    def _execute_click(
        self,
        data: Any,
        *,
        seed: int | None,
        threads: int | None,
        controls: dict[str, Any],
    ) -> (
        AnalysisResult
        | CausalResponseView
        | StatisticalTransportDistribution
        | TransportResponseGrid
    ):
        if self._transport is not None:
            if getattr(self, "_native", None) is None:
                from .transport._wrap import unavailable_from_stage

                return unavailable_from_stage(self)
            raw = self._run_click(
                lambda: self._transport.execute(
                    self._native,
                    data,
                    seed=seed,
                    threads=threads,
                    controls=controls,
                )
            )
            if isinstance(self._query, Transport):
                from .transport._wrap import wrap_transport_result

                if self._kind == "z_transport":
                    stage: Mapping[str, Any] = self._transport_stage or {}
                    self._transport_stage = retain(
                        {
                            **stage,
                            "catalog": self._native.catalog,
                            "bound": data if data is not None else stage.get("bound"),
                        }
                    )
                return wrap_transport_result(self, raw)
            return raw
        seed = self._seed if seed is None else seed
        threads = self._threads if threads is None else threads
        response = self._kind in ("response_curve", "intervention_response")
        native = self._native
        if data is None:
            bound = native.estimate_response_bound if response else native.estimate_bound
            raw = self._run_click(lambda: bound(seed=seed, threads=threads, **controls))
            return self._with_deferred_suite(
                self._wrap(raw), self._snapshot_data, seed=seed, threads=threads
            )
        if isinstance(self._query, (PolicyValue, MultiActionPolicyValue)):
            raise CausalUnsupportedError(
                "PolicyValue recommendations, nuisance predictions, and subject ownership are bound to the prepared evaluation rows; prepare a new study for new data",
                reason_code="option_not_applicable",
            )
        names, columns, frame = self._click_payload(data)
        if frame is not None:
            raw = self._run_click(
                lambda: native.estimate_frame(
                    names,
                    columns,
                    frame,
                    response=response,
                    seed=seed,
                    threads=threads,
                    **controls,
                )
            )
            return self._wrap(raw)
        fn = native.estimate_response if response else native.estimate
        raw = self._run_click(lambda: fn(names, columns, seed=seed, threads=threads, **controls))
        return self._with_deferred_suite(self._wrap(raw), data, seed=seed, threads=threads)

    def _with_deferred_suite(
        self, result: Any, data: Any, *, seed: int, threads: int | None
    ) -> Any:
        """Run the study's refuter suite as this click's second click.

        A scalar ``InterventionResponse`` on a Dag is estimated by the response
        executor, which fits its point and uncertainty together and has no
        validation stage of its own; the ATE-shaped suite denotes on it and
        runs against the estimate just produced.
        """
        if self._deferred_suite is None or data is None:
            return result
        from .results._report import copy_model

        refuted = self.refute(data, suite=self._deferred_suite, seed=seed, threads=threads)
        # The suite ran against this estimate, so the click's claim is the
        # refuted one; the estimate itself is the response executor's.
        return copy_model(
            result,
            validation=refuted.validation,
            certificate=refuted.certificate,
            reasoning=refuted.reasoning,
            evidence_status=refuted.evidence_status,
            claim_id=refuted.claim_id,
        )

    def mechanism_sensitivity(
        self,
        *,
        outcome_values: Sequence[float],
        parent_cardinalities: Sequence[int],
        treatment_levels: tuple[int, int],
        max_fraction: float,
        source_kernel_regime: int,
        source_kernel_snapshot: str,
        target_parent_regime: int,
        target_parent_snapshot: str,
        source_kernel: Sequence[tuple[Sequence[int], Sequence[float]]],
        source_parent_law: Sequence[tuple[Sequence[int], float]],
        target_parent_law: Sequence[tuple[int, Sequence[int], float]],
        decision_threshold: float | None = None,
        perturbed_treatment_level: int | None = None,
        perturbed_root_mechanism: str | None = None,
        perturbed_conditional_mechanism: str | None = None,
        cancel: Any = None,
    ) -> dict[str, Any]:
        """Evaluate checked fixed-graph mechanism sensitivity for exact transport."""
        if self._kind != "exact_transport":
            raise CausalUnsupportedError(
                "mechanism_sensitivity applies only to prepared exact transport",
                reason_code="option_not_applicable",
            )
        raw = self._native.mechanism_sensitivity(
            list(outcome_values),
            list(parent_cardinalities),
            treatment_levels,
            max_fraction,
            source_kernel_regime,
            source_kernel_snapshot,
            target_parent_regime,
            target_parent_snapshot,
            [(list(levels), list(probabilities)) for levels, probabilities in source_kernel],
            [(list(levels), probability) for levels, probability in source_parent_law],
            [
                (treatment, list(levels), probability)
                for treatment, levels, probability in target_parent_law
            ],
            decision_threshold=decision_threshold,
            perturbed_treatment_level=perturbed_treatment_level,
            perturbed_root_mechanism=perturbed_root_mechanism,
            perturbed_conditional_mechanism=perturbed_conditional_mechanism,
            cancel=cancel,
        )
        return json.loads(raw)

    @describe_refusal
    def estimate(
        self,
        data: Mapping[str, Any] | Any = None,
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
        on_progress: Any = _UNSET,
        on_stage: Any = _UNSET,
    ) -> ResultT:
        """Re-estimate without recompiling.

        With no ``data`` this re-executes the prepared program on the retained
        data. New ``data`` must match the prepared schema (and, for a panel /
        multi-environment / event study, its frame kind). ``seed`` / ``threads``
        default to the values given at prepare; ``cancel`` / ``on_progress`` /
        ``on_stage`` default to the study's retained controls and may be
        overridden (``None`` turns one off for this click).
        """
        return cast(
            ResultT,
            self._click(
                data,
                seed=seed,
                threads=threads,
                controls=self._click_controls(cancel, on_progress, on_stage),
            ),
        )

    @describe_refusal
    def retarget(
        self,
        weights: Any,
        depends_on: Sequence[str],
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: Any = _UNSET,
    ) -> AnalysisResult:
        """Estimate a declared target population from frozen scores. No refit.

        Requires a prepared AllObserved iid AIPW or cell-AIPW score table.
        Nonempty ``depends_on`` needs a directed graph (DAG or ADMG) so
        descendant closure can be checked. Nonconstant weights require a
        nonempty ``depends_on``.
        """
        if self._transport is not None:
            raise CausalUnsupportedError(
                "This operation is not licensed on the prepared transport handle",
                reason_code="option_not_applicable",
            )
        import numpy as np

        weights_array = np.asarray(weights, dtype=float)
        raw = self._native.retarget(
            weights_array if weights_array.ndim == 1 else weights_array.tolist(),
            list(depends_on),
            seed=seed,
            threads=threads,
            cancel=self._controls.cancel if cancel is _UNSET else cancel,
        )
        return _wrap_ate(raw, prepared=self)

    @describe_refusal
    def reexecute_retarget(
        self,
        artifact: bytes,
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: Any = _UNSET,
    ) -> AnalysisResult:
        """Re-execute the row-weight retarget an exported contract carries.

        The weights travel in the artifact with the identity that binds them to
        one data snapshot and one score table. Re-executing them on a study
        that holds a different snapshot raises ``CausalUnsupportedError`` with
        ``reason_code="row_weights_bound_to_snapshot"`` instead of silently
        reweighting other rows.
        """
        if self._transport is not None:
            raise CausalUnsupportedError(
                "This operation is not licensed on the prepared transport handle",
                reason_code="option_not_applicable",
            )
        raw = self._native.reexecute_retarget(
            bytes(artifact),
            seed=seed,
            threads=threads,
            cancel=self._controls.cancel if cancel is _UNSET else cancel,
        )
        return _wrap_ate(raw, prepared=self)

    def _staleness_payload(self, data: Any) -> tuple[list[str], list[Any]]:
        """Every column of ``data`` (encoded like a click, never projected onto the schema)."""
        if self._transport is not None:
            raise CausalUnsupportedError(
                "This operation is not licensed on the prepared transport handle",
                reason_code="option_not_applicable",
            )
        if self._treatment_labels is not None:
            data = self._treatment_labels.encode(data)
        data = drop_columns(data, self._named_design_columns)
        present = table_column_names(data)
        if self._column_names is not None and present is not None:
            # Prepared columns in prepared order, then any column the plan never saw, so an
            # added or missing column is reported and a mere reordering of the frame is not.
            prepared = [name for name in self._column_names if name in present]
            data = project_columns(
                data, prepared + [name for name in present if name not in prepared], what="new"
            )
        return ingest_columns(data)

    @describe_refusal
    def staleness(
        self,
        data: Mapping[str, Any] | Any,
        *,
        policy: Any = None,
        threads: int | None = None,
    ) -> Any:
        """Compare ``data`` with the data this study retains; a ``StalenessReport``.

        The report names every diagnostic, the dependency keys the findings change, and one
        recommendation (``reuse``, ``refresh``, ``reidentify`` or ``refuse``) with the next
        attempt. The study is not modified. See :mod:`antecedent.drift`.
        """
        from .drift import StalenessReport, _policy_json

        names, columns = self._staleness_payload(data)
        raw = self._native.staleness(
            names, columns, policy_json=_policy_json(policy), threads=threads
        )
        return StalenessReport._from_json(raw)

    @describe_refusal
    def staleness_baseline(self, *, policy: Any = None, threads: int | None = None) -> Any:
        """The sealed staleness baseline of the retained data, assessable without this study.

        Pass it with new data to :func:`antecedent.staleness`.
        """
        from .drift import StalenessBaseline, _policy_json

        if self._transport is not None:
            raise CausalUnsupportedError(
                "This operation is not licensed on the prepared transport handle",
                reason_code="option_not_applicable",
            )
        raw = self._native.export_staleness_baseline(
            policy_json=_policy_json(policy), threads=threads
        )
        return StalenessBaseline(bytes(raw))

    @describe_refusal
    def refute_target(
        self,
        data: Mapping[str, Any] | Any,
        *,
        suite: Literal["cheap", "placebo", "full"] = "full",
        replicates: int = 30,
        alpha: float = 0.05,
        subset_fraction: float = 0.8,
        negative_control_exposure: str | None = None,
        negative_control_outcome: str | None = None,
        mediator: str | None = None,
        mediation_rho_bar: float = 0.1,
        common_cause_strength: float = 0.0,
        prior_scale_factors: Sequence[float] | None = None,
        derivative_step: float | None = None,
        seed: int = 1,
        threads: int | None = None,
    ) -> Any:
        """Refute the target this study reports, on the whole target vector.

        Estimates on ``data`` (independent rows), then runs the target-native suite of the
        result's own target (conditional, derivative, mediation, world-valued or joint, not a
        scalar average effect standing in for it): every refit perturbs the data and re-runs
        this compiled program in full; the random common cause (declared in the graph with
        ``common_cause_strength``), the alternative priors (``prior_scale_factors`` times the
        study's prior scale) and the fitted response curve of a derivative (``derivative_step``)
        are compiled from this study with that one change. Negative controls default to the
        graph's candidates. Refuters the study cannot construct are returned as refusals with
        the next attempt. A ``TargetRefutation``
        (:mod:`antecedent.validation`); calibration is unmeasured.
        """
        from .validation import TargetRefutation

        if data is None:
            raise CausalUnsupportedError(
                "refute_target needs the data the study is refuted on",
                reason_code="invalid_argument",
            )
        names, columns, frame = self._click_payload(coerce_table_source(data))
        if frame is not None:
            raise CausalUnsupportedError(
                "refute_target perturbs independent rows; panel, event and multi-environment "
                "frames are refused (next attempt: the route's own design checks)",
                reason_code="option_not_applicable",
            )
        raw = self._native.refute_target(
            names,
            columns,
            suite=suite,
            replicates=replicates,
            alpha=alpha,
            subset_fraction=subset_fraction,
            negative_control_exposure=negative_control_exposure,
            negative_control_outcome=negative_control_outcome,
            mediator=mediator,
            mediation_rho_bar=mediation_rho_bar,
            common_cause_strength=common_cause_strength,
            prior_scale_factors=None
            if prior_scale_factors is None
            else [float(f) for f in prior_scale_factors],
            derivative_step=derivative_step,
            seed=seed,
            threads=threads,
        )
        return TargetRefutation._from_json(raw)

    @describe_refusal
    def refute(
        self,
        data: Mapping[str, Any] | Any = None,
        *,
        suite: Refute | Literal["placebo", "full", "cheap"] | bool | str = "placebo",
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
    ) -> AnalysisResult:
        """Second-click refute against the last :meth:`estimate`.

        Interactive first clicks typically use ``refute=False`` or ``cheap``;
        call this with ``suite="placebo"`` or ``"full"`` for the deferred suite.
        ``seed`` / ``threads`` / ``cancel`` default to the study's own.
        """
        if self._transport is not None:
            raise CausalUnsupportedError(
                "This operation is not licensed on the prepared transport handle",
                reason_code="option_not_applicable",
            )
        if self._kind == "response_curve":
            raise CausalUnsupportedError(
                "not_applicable: refute is AverageEffect and scalar "
                "Dag InterventionResponse; ResponseCurve cheap/full/placebo do not denote."
            )
        if isinstance(self._query, (TransportQuery, InterferenceQuery)):
            raise CausalUnsupportedError(
                f"{type(self._query).__name__} has no refuter suite: its estimand is "
                "defined by the design (selection diagram and trial probabilities, or the "
                "randomization and network), and the average-effect refuters do not apply",
                reason_code="option_not_applicable",
            )
        if isinstance(self._query, (AnomalyAttribution, ChangeAttribution)):
            raise CausalUnsupportedError(
                f"{type(self._query).__name__} has no refuter suite: GCM attribution "
                "scores are not an average treatment effect, and the average-effect "
                "refuters do not apply",
                reason_code="option_not_applicable",
            )
        if isinstance(suite, Refute):
            suite = str(suite)
        guard_design(self, data)
        if data is not None:
            names, columns, arrow = _prepared_columns(self._prepared_table(data))
        kwargs: dict[str, Any] = dict(
            seed=self._seed if seed is None else seed,
            threads=self._threads if threads is None else threads,
        )
        token = self._controls.cancel if cancel is _UNSET else cancel
        if token is not None:
            kwargs["cancel"] = token
        if data is None:
            raw = self._native.refute_bound(suite, **kwargs)
        else:
            fn = self._native.refute_arrow_c if arrow else self._native.refute
            raw = fn(names, columns, suite, **kwargs)
        result = _wrap_ate(raw, prepared=self)
        self._last_result = cast(ResultT, result)
        return result


__all__ = [
    "AnalysisResult",
    "ConflictSummaryView",
    "EffectEnvelope",
    "EstimateView",
    "MediationView",
    "IdentificationView",
    "IdentifyResult",
    "MediationEffectsSummary",
    "PerformanceView",
    "PhysicalPlanView",
    "PlanView",
    "PosteriorView",
    "ProgramPlanView",
    "PredictiveCheckReport",
    "PreparedBatch",
    "BatchResult",
    "BatchRetarget",
    "RetargetClaim",
    "RetargetContrast",
    "RetargetCost",
    "RetargetMember",
    "SimultaneousBandMember",
    "SimultaneousInterval",
    "SharedBatchDesign",
    "CandidateScreen",
    "CellFamily",
    "PriorSensitivityReport",
    "RefutationReport",
    "ValidationView",
    "analyze_many",
    "estimate_many",
    "prepare_many",
    "identify",
    "mediation_effects_summary",
]
