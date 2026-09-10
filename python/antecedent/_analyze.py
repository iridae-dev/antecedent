"""Analyze entrypoint and private query handlers.

``analyze`` is the public router; branch bodies live here so new query types
extend via a handler rather than growing a monolith.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from .results import CausalResponseView

from ._coerce import coerce_latency, coerce_query, coerce_refute
from ._data import as_columns, as_multi_env_columns, ingest_columns, try_as_arrow_c_columns
from ._native import CausalUnsupportedError
from ._native import (
    analyze as _analyze_temporal,
)
from ._native import (
    analyze_ate as _analyze_ate,
)
from ._native import (
    analyze_ate_admg as _analyze_ate_admg,
)
from ._native import (
    analyze_ate_admg_arrow_c as _analyze_ate_admg_arrow_c,
)
from ._native import (
    analyze_ate_arrow_c as _analyze_ate_arrow_c,
)
from ._native import (
    analyze_ate_cpdag as _analyze_ate_cpdag,
)
from ._native import (
    analyze_ate_cpdag_arrow_c as _analyze_ate_cpdag_arrow_c,
)
from ._native import (
    analyze_ate_discover as _analyze_ate_discover,
)
from ._native import (
    analyze_ate_graph_posterior as _analyze_ate_graph_posterior,
)
from ._native import (
    analyze_ate_pag as _analyze_ate_pag,
)
from ._native import (
    analyze_ate_pag_arrow_c as _analyze_ate_pag_arrow_c,
)
from ._native import (
    analyze_ate_tiered as _analyze_ate_tiered,
)
from ._native import (
    analyze_conditional as _analyze_conditional,
)
from ._native import (
    analyze_distribution as _analyze_distribution,
)
from ._native import (
    analyze_events as _analyze_events,
)
from ._native import analyze_observation_response as _analyze_observation_response
from ._native import (
    analyze_panel as _analyze_panel,
)
from ._native import (
    analyze_panel_discover as _analyze_panel_discover,
)
from ._native import (
    analyze_path_specific as _analyze_path_specific,
)
from ._native import analyze_response as _analyze_response
from ._native import analyze_response_pag as _analyze_response_pag
from ._native import (
    analyze_temporal_cpdag as _analyze_temporal_cpdag,
)
from ._native import (
    analyze_temporal_discover as _analyze_temporal_discover,
)
from ._native import (
    analyze_temporal_graph_posterior as _analyze_temporal_graph_posterior,
)
from ._native import (
    analyze_temporal_mediation as _analyze_temporal_mediation,
)
from ._native import (
    analyze_temporal_pag as _analyze_temporal_pag,
)
from ._native import analyze_temporal_response as _analyze_temporal_response
from .data import EventFrame, MultiEnvFrame, PanelFrame
from .discovery import (
    FCI,
    GES,
    LPCMCI,
    NOTEARS,
    PC,
    PCMCI,
    RFCI,
    RPCMCI,
    CiScreenedPosterior,
    DbnPosterior,
    ExactDagPosterior,
    GraphPosterior,
    JPCMCIPlus,
    LiNGAM,
    OrderMcmc,
    PCMCIPlus,
    StructureMcmc,
)
from .estimation import (
    AnalysisResult,
    _resolve_latency_budget,
)
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag, TieredBackground
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, Frequentist
from .observation import Complete as _ObservationComplete
from .query import (
    AverageDerivative,
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionalDistribution,
    InterventionResponse,
    MediationEffect,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)

_RESPONSE_QUERIES = (
    ResponseCurve,
    AverageDerivative,
    PointDerivative,
    Elasticity,
    SemiElasticity,
    DirectionalDerivative,
    ResponseJacobian,
    InterventionResponse,
)

_STATIC_DISCOVERY = (PC, GES, LiNGAM, NOTEARS, FCI, RFCI)
_GRAPH_POSTERIOR_DISCOVERY = (
    ExactDagPosterior,
    OrderMcmc,
    StructureMcmc,
    CiScreenedPosterior,
)


class EstimatorConfigLike(Protocol):
    """A typed estimator config from :mod:`antecedent.estimators`."""

    @property
    def estimator_id(self) -> str: ...

    def _wire(self) -> dict[str, Any]: ...


_TEMPORAL_DISCOVERY = (PCMCI, PCMCIPlus, LPCMCI, JPCMCIPlus, RPCMCI)


def _staged_prepared_result(
    data: Any,
    query: Any,
    *,
    graph: Any,
    inference: Frequentist | Bayesian,
    refute: bool | str,
    seed: int,
    bootstrap: int | None,
    threads: int,
    structure_accepted: bool = False,
    identifier: str | None = None,
    estimator: str | None = None,
    validators: Sequence[Any] | None = None,
    latency: Latency | None = None,
) -> Any:
    """Run a licensed staged cell through prepare → estimate, not a Frequentist sidecar."""
    if validators is not None:
        raise CausalUnsupportedError("this staged query path does not support validators")
    from .accepted_graph import AcceptedGraph
    from .estimation import PreparedAnalysis

    prepared = PreparedAnalysis.prepare(
        data,
        query=query,
        graph=AcceptedGraph(graph) if structure_accepted else graph,
        inference=inference,
        identifier=identifier,
        estimator=estimator,
        refute=cast("bool | Literal['full', 'placebo', 'none', 'cheap']", refute),
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        latency=latency,
    )
    return prepared.estimate(data, seed=seed, threads=threads)


def handle_conditional(
    data: Any,
    query: ConditionalEffect,
    *,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    refute: bool | str,
    validators: Sequence[Any] | None,
    seed: int,
    bootstrap: int | None,
    threads: int,
    structure_accepted: bool = False,
) -> Any:
    from .estimation import _static_edges, _wrap_ate
    from .query import coerce_outcome_functional

    if discovery is not None:
        raise ValueError("ConditionalEffect does not support discovery=")
    if isinstance(graph, Admg):
        raise CausalUnsupportedError(
            "refused: ConditionalEffect on Admg has no compile arm; "
            "Dag, Cpdag, and Pag are licensed."
        )
    if isinstance(graph, (Cpdag, Pag)) or isinstance(inference, Bayesian):
        return _staged_prepared_result(
            data,
            query,
            graph=graph,
            inference=inference,
            refute=refute,
            validators=validators,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            structure_accepted=structure_accepted,
        )
    names, columns = ingest_columns(data)
    edges = _static_edges(graph)
    raw = _analyze_conditional(
        names,
        columns,
        edges,
        query.treatment,
        query.outcome,
        query.modifier,
        control_level=query.control_level,
        active_level=query.active_level,
        outcome_functional=coerce_outcome_functional(getattr(query, "outcome_functional", None)),
        refute=refute,
        validators=list(validators) if validators is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        accepted=structure_accepted,
    )
    return _wrap_ate(raw, query=query)


def handle_temporal_mediation(
    data: Any,
    query: TemporalMediationEffect,
    *,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    refute: bool | str = False,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import _lagged_edges, _wrap_temporal

    if discovery is not None:
        raise ValueError("TemporalMediationEffect does not support discovery=")
    if isinstance(inference, Bayesian):
        return _staged_prepared_result(
            data,
            query,
            graph=graph,
            inference=inference,
            refute=refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
    names, columns = ingest_columns(data)
    lagged = _lagged_edges(graph)
    raw = _analyze_temporal_mediation(
        names,
        columns,
        lagged,
        query.treatment,
        query.mediator,
        query.outcome,
        contrast=query.contrast,
        control_level=query.control_level,
        active_level=query.active_level,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_temporal(raw)


def _encode_temporal_intervention(spec: Any) -> tuple[str, str, list[float]]:
    """Map a temporal InterventionResponse step to native (variable, kind, params).

    Licensed single-step :class:`~antecedent.intervention.Sequence` unwraps to its
    inner Set/Shift/Soft. Multi-step and nested Sequence refuse closed.
    """
    from . import intervention as intervention_specs

    if isinstance(spec, intervention_specs.Sequence):
        if len(spec.steps) != 1:
            raise CausalUnsupportedError(
                "refused: multi-step Sequence intervention policies are not licensed "
                "for temporal InterventionResponse; use a single-step Sequence or a "
                "bare Set/Shift/Soft"
            )
        inner = spec.steps[0]
        if isinstance(inner, intervention_specs.Sequence):
            raise CausalUnsupportedError(
                "refused: nested Sequence interventions are not licensed for temporal "
                "InterventionResponse"
            )
        return _encode_temporal_intervention(inner)
    if isinstance(spec, intervention_specs.Set):
        return spec.variable, "set", [spec.value]
    if isinstance(spec, intervention_specs.Shift):
        return spec.variable, "shift", [spec.delta]
    if isinstance(spec, intervention_specs.Soft):
        if spec.mechanism == "constant":
            return spec.variable, "soft_constant", list(spec.parameters)
        if spec.mechanism == "additive_shift":
            return spec.variable, "soft_additive_shift", list(spec.parameters)
        raise CausalUnsupportedError(
            f"Soft mechanism {spec.mechanism!r} is not licensed temporally"
        )
    raise TypeError(
        "temporal InterventionResponse supports Set/Shift/Soft and a single-step Sequence"
    )


def handle_response(
    data: Any,
    query: Any,
    *,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    identifier: str | None,
    estimator: str | None,
    estimator_config: Mapping[str, object] | None,
    validators: Sequence[Any] | None,
    refute_requested: bool,
    refute: bool | str,
    bootstrap_requested: bool,
    seed: int,
    threads: int,
    structure_accepted: bool = False,
) -> Any:
    """Identify and estimate a complete-observation continuous response."""
    from .estimation import _response_support_bounds, _static_edges, _support_point_status
    from .results import (
        CausalResponseView,
        IdentificationView,
        ResponseEnvelopeView,
        ResponseUncertainty,
        ResponseView,
        SupportDiagnostic,
        SupportReport,
    )
    from .results.response import SupportStatus, UncertaintyKind

    if discovery is not None:
        if isinstance(discovery, (*_GRAPH_POSTERIOR_DISCOVERY, GraphPosterior)):
            raise CausalUnsupportedError(
                "refused: Graph-posterior response is a contract choice, not typed "
                "impossibility: the ATE envelope (retained unidentified mass) is the "
                "same object a curve arm would use. This cut does not license a "
                "response mixture."
            )
        raise ValueError("response queries do not yet support discovery=")
    if isinstance(inference, Bayesian):
        if not isinstance(query, (ResponseCurve, InterventionResponse)):
            raise CausalUnsupportedError(
                "refused: Licensed derivative cells are Frequentist explicit or accepted "
                "Dag at validation none; Bayesian derivatives remain 1.7 work."
            )
        if bootstrap_requested:
            raise CausalUnsupportedError(
                "Bayesian responses use posterior intervals; bootstrap is unsupported"
            )
        return _staged_prepared_result(
            data,
            query,
            graph=graph,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            validators=validators,
            refute="none" if not refute_requested else refute,
            seed=seed,
            bootstrap=None,
            threads=threads,
            structure_accepted=structure_accepted,
        )
    # Derivative cells (AverageDerivative/DirectionalDerivative/Elasticity/PointDerivative/
    # ResponseJacobian/SemiElasticity) used to be refused right here with a hand-typed
    # literal duplicating parity/support_closed.toml. That duplication was a drift risk:
    # the native `analyze_response` (python/src/response_api.rs) now consults the
    # generated support matrix itself (via `antecedent::support::refuse_if_not_applicable`)
    # before running, and raises the identical "refused: ..." text straight from the TOML,
    # so the hand-rolled check here was redundant and has been removed.
    #
    # Admg keeps a Python-side refuse: `_static_edges` cannot carry an Admg, and
    # there is no curve plug-in for licensed general-ID ATE. Cpdag/Pag MeanCurve
    # and InterventionResponse now take the staged prepare path (same
    # generalized-adjustment envelope as ATE). Derivatives still refuse here
    # because a Pag/Admg never reaches native `_analyze_response`. The Admg
    # literal is pinned against parity/support_closed.toml.
    if isinstance(graph, (Admg, Cpdag, Pag)) and not getattr(query, "is_temporal", False):
        if isinstance(query, (ResponseCurve, InterventionResponse)):
            if isinstance(graph, Admg):
                raise CausalUnsupportedError(
                    "refused: Admg response has no functional plug-in; licensed general-ID "
                    "ATE does not estimate a curve."
                )
            return _staged_prepared_result(
                data,
                query,
                graph=graph,
                inference=inference,
                identifier=identifier,
                estimator=estimator,
                validators=validators,
                refute="none" if not refute_requested else refute,
                seed=seed,
                bootstrap=None,
                threads=threads,
                structure_accepted=structure_accepted,
            )
        raise CausalUnsupportedError("refused: Derivatives require a supplied static Dag.")
    if getattr(query, "is_temporal", False):
        from .estimation import _lagged_edges, _wrap_prepared_response

        if not isinstance(graph, (TemporalDag, list, tuple)):
            raise TypeError(
                "temporal response requires a TemporalDag or lagged edge list; "
                f"got {type(graph).__name__}"
            )
        if identifier not in (None, "temporal.backdoor.unfolded"):
            raise ValueError(
                f"temporal response requires identifier='temporal.backdoor.unfolded'; "
                f"got {identifier!r}"
            )
        if estimator not in (None, "temporal.response.gcomp"):
            raise ValueError(
                f"temporal response requires estimator='temporal.response.gcomp'; got {estimator!r}"
            )
        names, columns = ingest_columns(data)
        lagged = _lagged_edges(graph)
        temporal_treatments: list[str]
        temporal_outcomes: list[str]
        temporal_intervention_kinds: list[str] | None = None
        temporal_intervention_parameters: list[list[float]] | None = None
        temporal_grid: list[float] | None
        if isinstance(query, InterventionResponse):
            from . import intervention as intervention_specs

            supplied = query.intervention
            interventions = (
                list(supplied)
                if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
                else [supplied]
            )
            temporal_treatments = []
            temporal_outcomes = [query.outcome]
            kinds: list[str] = []
            parameters_list: list[list[float]] = []
            for spec in interventions:
                variable, kind, parameters = _encode_temporal_intervention(spec)
                temporal_treatments.append(variable)
                kinds.append(kind)
                parameters_list.append(parameters)
            temporal_intervention_kinds = kinds
            temporal_intervention_parameters = parameters_list
            temporal_grid = None
        else:
            temporal_treatments = [query.treatment]
            temporal_outcomes = [query.outcome]
            temporal_grid = list(query.grid)
        temporal_raw = _analyze_temporal_response(
            names,
            columns,
            lagged,
            query.kind,
            temporal_treatments,
            temporal_outcomes,
            grid=temporal_grid,
            intervention_kinds=temporal_intervention_kinds,
            intervention_parameters=temporal_intervention_parameters,
            horizons=list(query.horizons or ()),
            policy=query.policy,
            treatment_lag=query.treatment_lag,
            max_history_lag=query.max_history_lag,
            seed=seed,
            threads=threads,
            accepted=structure_accepted,
            refute=refute if refute_requested else False,
        )
        return _wrap_prepared_response(temporal_raw, query)
    if isinstance(graph, TieredBackground):
        if not isinstance(query, InterventionResponse):
            raise CausalUnsupportedError(
                "TieredBackground response cells are licensed for joint InterventionResponse"
            )
        return _staged_prepared_result(
            data,
            query,
            graph=graph,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            validators=validators,
            refute="none" if not refute_requested else refute,
            seed=seed,
            bootstrap=None,
            threads=threads,
            structure_accepted=structure_accepted,
        )
    if (
        graph is not None
        and not isinstance(graph, (Dag, Pag))
        and not isinstance(graph, (list, tuple))
    ):
        raise TypeError(
            "response queries require a Dag, Pag, or directed edge list; "
            f"got {type(graph).__name__}"
        )
    expected_estimator = {
        "response_curve": "response.kennedy_dr",
        "point_derivative": "response.kennedy_dr",
        "elasticity": "response.kennedy_dr",
        "semi_elasticity": "response.kennedy_dr",
        "average_derivative": "response.riesz_ade",
        "directional_derivative": "response.gam_derivative",
        "response_jacobian": "response.gam_derivative",
        "intervention_response": "response.intervention_gcomp",
    }.get(query.kind)
    expected_identifier = (
        "generalized.adjustment" if isinstance(graph, Pag) else "response.backdoor"
    )
    if identifier not in (None, expected_identifier):
        raise ValueError(
            f"{query.kind} requires identifier={expected_identifier!r}; got {identifier!r}"
        )
    if estimator not in (None, expected_estimator):
        raise ValueError(
            f"{query.kind} requires estimator={expected_estimator!r}; got {estimator!r}"
        )
    if threads != 1:
        raise ValueError("response queries currently require threads=1")
    response_options: dict[str, object] = {}
    if estimator_config is not None:
        unknown = set(estimator_config) - {
            "bandwidth",
            "simultaneous_replicates",
            "confidence_level",
            "multiplier_seed",
            "export_row_diagnostics",
        }
        if unknown:
            raise ValueError(
                "unknown response estimator_config keys: " + ", ".join(sorted(unknown))
            )
        response_options.update(estimator_config)
        if "simultaneous_replicates" in response_options and "bandwidth" not in response_options:
            raise ValueError(
                "simultaneous response bands require an explicit estimator_config bandwidth"
            )
        # Point derivatives/elasticities require an explicit bandwidth (Silverman's rule is
        # refused for m'/m''). Simultaneous bands remain MeanCurve-only.
        if isinstance(query, (PointDerivative, Elasticity, SemiElasticity)):
            if "simultaneous_replicates" in response_options:
                raise ValueError(
                    "simultaneous response bands currently apply to ResponseCurve only"
                )
            if "bandwidth" not in response_options:
                raise ValueError(
                    "PointDerivative/Elasticity/SemiElasticity require estimator_config bandwidth"
                )
        elif not isinstance(query, ResponseCurve):
            raise ValueError(
                "response estimator_config currently applies to ResponseCurve and point derivatives only"
            )
    if validators is not None:
        raise ValueError("response queries do not accept scalar ATE validators")
    if bootstrap_requested:
        raise ValueError("response queries do not yet expose bootstrap= through analyze()")
    mechanism = getattr(query, "observation", None)
    # Complete() is the documented "outcome is observed directly" spelling and
    # must be treated exactly like no mechanism at all everywhere below --
    # otherwise a query carrying it is misrouted onto the observation-aware
    # path, which does not (and should not) know how to handle it.
    if isinstance(mechanism, _ObservationComplete):
        mechanism = None
    observation_assumptions = tuple(getattr(query, "observation_assumptions", ()))
    if mechanism is None and observation_assumptions:
        raise ValueError("observation_assumptions require an explicit observation mechanism")
    if mechanism is not None and refute_requested:
        raise ValueError(
            "observation-aware curve validation is unavailable because subset refits must "
            "re-estimate both the observation correction and response jointly"
        )
    if getattr(query, "target_population", None) is not None:
        raise ValueError("response queries do not yet support target_population")
    names, columns = ingest_columns(data)
    at: list[float] | None
    if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
        treatments = list(query.treatments)
        outcomes = list(query.outcomes)
        at = (
            [query.at[name] for name in treatments]
            if isinstance(query.at, Mapping)
            else list(query.at)
        )
    elif isinstance(query, InterventionResponse):
        from . import intervention as intervention_specs

        supplied = query.intervention
        interventions = (
            list(supplied)
            if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
            else [supplied]
        )
        if not interventions:
            raise ValueError("InterventionResponse requires at least one intervention")
        treatments = []
        outcomes = [query.outcome]
        intervention_kinds: list[str] = []
        intervention_parameters: list[list[float]] = []
        for spec in interventions:
            if isinstance(spec, intervention_specs.Set):
                kind, parameters = "set", [spec.value]
            elif isinstance(spec, intervention_specs.Shift):
                kind, parameters = "shift", [spec.delta]
            elif isinstance(spec, intervention_specs.Bernoulli):
                kind, parameters = "bernoulli", [spec.p]
            elif isinstance(spec, intervention_specs.Gaussian):
                kind, parameters = "gaussian", [spec.mean, spec.variance]
            elif isinstance(spec, intervention_specs.Categorical):
                kind, parameters = "categorical", list(spec.probabilities)
            elif isinstance(spec, intervention_specs.Soft):
                if not getattr(query, "is_temporal", False):
                    raise CausalUnsupportedError(
                        "Soft interventions require a temporal response cell "
                        "(set horizons=...) and are not estimable by response.intervention_gcomp"
                    )
                if spec.mechanism == "constant":
                    kind, parameters = "soft_constant", list(spec.parameters)
                elif spec.mechanism == "additive_shift":
                    kind, parameters = "soft_additive_shift", list(spec.parameters)
                else:
                    raise CausalUnsupportedError(
                        f"Soft mechanism {spec.mechanism!r} is not licensed for temporal "
                        "InterventionResponse; use constant or additive_shift"
                    )
            elif isinstance(spec, intervention_specs.Sequence):
                raise CausalUnsupportedError(
                    "Sequence interventions are not licensed on static InterventionResponse; "
                    "use a temporal cell (horizons=...) with a single-step Sequence, or pass "
                    "a bare Set/Shift/Soft"
                )
            else:
                raise TypeError(
                    "InterventionResponse.intervention must be an antecedent.intervention "
                    "specification or a sequence of specifications"
                )
            treatments.append(spec.variable)
            intervention_kinds.append(kind)
            intervention_parameters.append(parameters)
        direction = None
        at = None
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
        raise TypeError("AverageDerivative.weighting currently accepts 'observed'")
    raw: Any
    if mechanism is not None:
        if response_options:
            raise ValueError("observation-aware response does not yet compose estimator_config")
        if not isinstance(query, ResponseCurve):
            raise ValueError(
                "observation-aware response execution currently supports MeanCurve only"
            )
        if isinstance(graph, Pag):
            raise ValueError("observation-aware PAG response envelopes are not yet composed")
        from .observation import (
            _assumption_kwargs,
            _ensure_latent_schema_column,
            _mechanism_kwargs,
        )

        if len(observation_assumptions) != 1:
            raise ValueError("observation-aware response requires exactly one explicit assumption")
        names, columns = _ensure_latent_schema_column(names, columns, mechanism)
        edges = _static_edges(graph)
        observation_kwargs = _mechanism_kwargs(mechanism)
        observation_kwargs.update(_assumption_kwargs(observation_assumptions[0]))
        raw = cast(Any, _analyze_observation_response)(
            names,
            columns,
            edges,
            query.treatment,
            query.outcome,
            list(query.grid),
            accepted=structure_accepted,
            **observation_kwargs,
        )
    elif isinstance(graph, Pag):
        if response_options:
            raise ValueError("PAG response envelopes do not yet compose estimator_config")
        if not isinstance(query, ResponseCurve):
            raise ValueError("PAG response envelopes currently support MeanCurve only")
        raw = _analyze_response_pag(
            names,
            columns,
            graph,
            query.treatment,
            query.outcome,
            list(query.grid),
        )
    else:
        edges = _static_edges(graph)
        raw = _analyze_response(
            names,
            columns,
            edges,
            query.kind,
            treatments,
            outcomes,
            grid=list(query.grid) if isinstance(query, ResponseCurve) else None,
            at=at,
            direction=direction,
            order=getattr(query, "order", 1),
            scale=scale,
            weighting=weighting,
            intervention_kinds=(
                intervention_kinds if isinstance(query, InterventionResponse) else None
            ),
            intervention_parameters=(
                intervention_parameters if isinstance(query, InterventionResponse) else None
            ),
            bandwidth=cast(float | None, response_options.get("bandwidth")),
            simultaneous_replicates=cast(
                int | None, response_options.get("simultaneous_replicates")
            ),
            confidence_level=cast(float, response_options.get("confidence_level", 0.95)),
            multiplier_seed=cast(int, response_options.get("multiplier_seed", seed)),
            export_row_diagnostics=bool(response_options.get("export_row_diagnostics", False)),
            accepted=structure_accepted,
            refute=refute if refute_requested else False,
        )
    response = (
        ResponseView(raw.treatments, raw.outcomes, raw.points, raw.values)
        if raw.points and raw.values
        else None
    )
    uncertainty = ResponseUncertainty(
        cast(UncertaintyKind, raw.uncertainty_kind),
        lower=raw.lower,
        upper=raw.upper,
        level=raw.level,
        standard_error=raw.standard_error,
        replicates=raw.replicates,
        artifact_id=raw.artifact_id,
    )
    diagnostics = [
        SupportDiagnostic(identifier, values, detail)
        for identifier, values, detail in zip(
            raw.diagnostic_ids,
            raw.diagnostic_values,
            raw.diagnostic_details,
            strict=True,
        )
    ]
    region = _response_support_bounds(raw)
    envelope = None
    if getattr(raw, "identified_mass", None) is not None:
        if raw.lower is None or raw.upper is None:
            raise RuntimeError("native PAG response omitted its identified envelope")
        if (
            getattr(raw, "unidentified_mass", None) is None
            or getattr(raw, "completion_count", None) is None
            or getattr(raw, "truncated_completions", None) is None
            or getattr(raw, "enumeration_capped", None) is None
            or getattr(raw, "mass_scope", None) is None
        ):
            raise RuntimeError("native PAG response omitted its completion mass metadata")
        envelope = ResponseEnvelopeView(
            raw.treatments,
            raw.outcomes,
            raw.points,
            raw.lower,
            raw.upper,
            raw.identified_mass,
            raw.unidentified_mass,
            raw.completion_count,
            raw.truncated_completions,
            raw.enumeration_capped,
            cast(Literal["full_class", "examined_completions"], raw.mass_scope),
        )
    identification_operation = (
        "identify.generalized_adjustment" if isinstance(graph, Pag) else "identify.response"
    )
    provenance = {
        "operation_id": raw.provenance_id,
        "operation_ids": [identification_operation, raw.provenance_id],
    }
    certificate_json = getattr(raw, "certificate_json", None)
    return CausalResponseView(
        certificate=json.loads(certificate_json) if certificate_json else None,
        estimand=query,
        response=response,
        estimate=raw.scalar if raw.scalar is not None else raw.matrix,
        uncertainty=uncertainty,
        support=SupportReport(
            cast(SupportStatus, raw.support_status),
            region,
            diagnostics,
            raw.warnings,
            _support_point_status(raw),
        ),
        identification=IdentificationView(
            status=raw.identification,
            method=("generalized.adjustment" if isinstance(graph, Pag) else "response.backdoor"),
            adjustment_set=list(getattr(raw, "adjustment_set", ())),
            assumption_count=len(raw.assumptions),
            derivation_step_count=0,
        ),
        assumptions=raw.assumptions,
        provenance=provenance,
        envelope=envelope,
        validation=None,
        evidence_status=getattr(raw, "evidence_status", None),
        allowlist_reason=getattr(raw, "allowlist_reason", None),
        allowlist_parent=getattr(raw, "allowlist_parent", None),
    )


def handle_mediation(
    data: Any,
    query: MediationEffect,
    *,
    graph: Any,
    discovery: Any,
    refute: bool | str,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    del data, query, graph, refute, seed, bootstrap, threads
    if discovery is not None:
        raise CausalUnsupportedError(
            "refused: Static natural mediation is Frequentist; a Bayesian mediation "
            "estimator is 1.7 work."
        )
    raise CausalUnsupportedError("refused: MediationEffect requires a supplied static Dag.")


def handle_counterfactual(
    data: Any,
    query: Counterfactual,
    *,
    graph: Any,
    discovery: Any,
    seed: int,
    threads: int,
) -> Any:
    del data, query, graph, seed, threads
    if discovery is not None:
        raise CausalUnsupportedError(
            "refused: Staged counterfactuals require an explicit Dag; accepted and "
            "graph-posterior structures are refused."
        )
    raise CausalUnsupportedError("refused: Counterfactual requires a supplied static Dag.")


def handle_distribution(
    data: Any,
    query: InterventionalDistribution,
    *,
    graph: Any,
    discovery: Any,
    accept_discovered: bool,
    refute_requested: bool,
    refute: bool | str,
    seed: int,
    threads: int,
) -> Any:
    from .estimation import (
        _static_edges,
        _wrap_ate,
    )

    del accept_discovered
    if discovery is not None:
        raise CausalUnsupportedError(
            "refused: Graph-posterior path and distribution mixtures are not staged. "
            "For a reviewed discovered Dag, pass graph=AcceptedGraph(...)."
        )
    else:
        edges = _static_edges(graph)
    names, columns = ingest_columns(data)
    raw = _analyze_distribution(
        names,
        columns,
        edges,
        query.outcome,
        dict(query.interventions),
        conditioning=list(query.conditioning) or None,
        refute=refute if refute_requested else False,
        seed=seed,
        threads=threads,
    )
    return _wrap_ate(raw, query=query)


def handle_path_specific(
    data: Any,
    query: PathSpecificEffect,
    *,
    graph: Any,
    discovery: Any,
    accept_discovered: bool,
    refute_requested: bool,
    refute: bool | str,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import (
        _static_edges,
        _wrap_ate,
    )

    del accept_discovered
    if discovery is not None:
        raise CausalUnsupportedError(
            "refused: Graph-posterior path and distribution mixtures are not staged. "
            "For a reviewed discovered Dag, pass graph=AcceptedGraph(...)."
        )
    else:
        edges = _static_edges(graph)
    names, columns = ingest_columns(data)
    raw = _analyze_path_specific(
        names,
        columns,
        edges,
        query.treatment,
        query.outcome,
        control_level=query.control_level,
        active_level=query.active_level,
        path_nodes=list(query.path_nodes) if query.path_nodes is not None else None,
        max_paths=query.max_paths,
        max_len=query.max_len,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        refute=refute if refute_requested else False,
    )
    return _wrap_ate(raw, query=query)


def handle_supplied_graph_posterior(
    data: Any,
    query: AverageEffect | PulseEffect | SustainedEffect,
    *,
    discovery: GraphPosterior,
    inference: Frequentist | Bayesian,
    identifier: str | None,
    estimator: str | None,
    estimator_config: Mapping[str, Any] | None,
    validators: Sequence[Any] | None,
    population_registry: Any | None,
    return_posterior_artifact: bool,
    refute: bool | str,
    seed: int,
    bootstrap: int | None,
    threads: int,
    cancel: Any | None,
    on_progress: Any | None,
) -> Any:
    from .estimation import _bayesian_inference_kwargs, _wrap_ate

    if isinstance(query, (PulseEffect, SustainedEffect)) and not isinstance(inference, Bayesian):
        raise TypeError(
            "graph-posterior discovery requires inference=Bayesian(...) for temporal effect mixture"
        )
    if isinstance(query, AverageEffect) and not isinstance(inference, (Frequentist, Bayesian)):
        raise TypeError(
            "graph-posterior AverageEffect requires inference=Frequentist() or Bayesian(...)"
        )
    if not isinstance(query, (AverageEffect, PulseEffect, SustainedEffect)):
        raise TypeError(
            "GraphPosterior is licensed for AverageEffect, PulseEffect, and SustainedEffect"
        )
    # The supplied-posterior path selects its identifier and estimator per
    # atom and runs the frozen atoms as given. Refuse the options it cannot
    # honour rather than dropping them silently, mirroring PreparedAnalysis.
    dropped = [
        name
        for name, value in (
            ("identifier", identifier),
            ("estimator", estimator),
            ("estimator_config", estimator_config),
            ("validators", validators),
            ("population_registry", population_registry),
        )
        if value is not None
    ]
    if return_posterior_artifact:
        dropped.append("return_posterior_artifact")
    if dropped:
        raise CausalUnsupportedError(
            "analyze(discovery=GraphPosterior(...)) selects its identifier and estimator "
            f"per posterior atom and does not support {', '.join(dropped)}; drop them or "
            "run discovery inside analyze(discovery=ExactDagPosterior()/DbnPosterior(...))"
        )
    names, columns = ingest_columns(data)
    bootstrap_n = 0 if bootstrap is None else bootstrap
    if isinstance(inference, Bayesian):
        bayes_kw = _bayesian_inference_kwargs(inference)
        unsupported = sorted(set(bayes_kw) - {"inference", "n_draws", "prior_scale"})
        if unsupported:
            raise CausalUnsupportedError(
                "analyze(discovery=GraphPosterior(...)) does not support Bayesian prior "
                f"transfer or mapping ({', '.join(unsupported)}); use a plain Bayesian(...)"
            )
        common = {
            "inference": bayes_kw["inference"],
            "n_draws": bayes_kw["n_draws"],
            "prior_scale": bayes_kw["prior_scale"],
            "refute": refute,
            "seed": seed,
            "bootstrap": bootstrap_n,
            "threads": threads,
            "cancel": cancel,
            "on_progress": on_progress,
        }
    else:
        common = {
            "inference": "frequentist",
            "refute": refute,
            "seed": seed,
            "bootstrap": bootstrap_n,
            "threads": threads,
            "cancel": cancel,
            "on_progress": on_progress,
        }
    if isinstance(query, AverageEffect):
        return _wrap_ate(
            _analyze_ate_graph_posterior(
                names,
                columns,
                discovery,
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                **common,
            ),
            query=query,
        )
    return _wrap_ate(
        _analyze_temporal_graph_posterior(
            names,
            columns,
            discovery,
            query.treatment,
            query.outcome,
            policy=query.kind,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            **common,
        ),
        query=query,
    )


def handle_static_ate_discover(
    data: Any,
    query: AverageEffect,
    *,
    discovery: Any,
    inference: Frequentist | Bayesian,
    identifier: str | None,
    estimator: str | None,
    refute: bool | str,
    validators: Sequence[Any] | None,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
    estimator_config: Mapping[str, Any] | None,
) -> Any:
    from .estimation import (
        _bayesian_inference_kwargs,
        _discovery_algorithm,
        _wrap_ate,
    )

    if not isinstance(query, AverageEffect):
        raise ValueError(f"discovery={type(discovery).__name__}(...) requires AverageEffect")
    if (
        isinstance(discovery, _GRAPH_POSTERIOR_DISCOVERY)
        and isinstance(inference, Frequentist)
        and not isinstance(query, AverageEffect)
    ):
        raise TypeError(
            "graph-posterior discovery requires inference=Bayesian(...) for effect mixture"
        )
    if not isinstance(discovery, _STATIC_DISCOVERY + _GRAPH_POSTERIOR_DISCOVERY):
        raise TypeError(f"unsupported static discovery: {type(discovery)!r}")
    names, columns = ingest_columns(data)
    cfg = _discovery_algorithm(discovery)
    bayes_kw: dict[str, Any] = {}
    if isinstance(inference, Bayesian):
        bayes_kw = _bayesian_inference_kwargs(inference)
    elif isinstance(inference, Frequentist):
        bayes_kw = {"inference": "frequentist"}
    raw = _analyze_ate_discover(
        names,
        columns,
        query.treatment,
        query.outcome,
        algorithm=cfg["algorithm"],
        alpha=cfg.get("alpha", 0.05),
        fdr=cfg.get("fdr", True),
        max_cond_size=cfg.get("max_cond_size", 2),
        prune_threshold=cfg.get("prune_threshold", 0.0),
        l1=cfg.get("lambda", 0.1),
        threshold=cfg.get("threshold", 0.3),
        standardize=cfg.get("standardize", True),
        accept_discovered=accept_discovered,
        control_level=query.control_level,
        active_level=query.active_level,
        identifier=identifier,
        estimator=estimator,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        ci=cfg.get("ci"),
        n_chains=cfg.get("n_chains", 2),
        n_warmup=cfg.get("n_warmup", 100),
        mcmc_draws=cfg.get("mcmc_draws", 200),
        thin=cfg.get("thin", 1),
        soft_weight=cfg.get("soft_weight", "none"),
        require_diagnostics_gate=cfg.get("require_diagnostics_gate", True),
        estimator_config=dict(estimator_config) if estimator_config is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        **bayes_kw,
    )
    return _wrap_ate(raw, query=query)


def _typed_ate(
    data: Any,
    graph: Any,
    numpy_fn: Callable[..., Any],
    arrow_fn: Callable[..., Any],
    **common: Any,
) -> Any:
    arrow = try_as_arrow_c_columns(data)
    if arrow is not None:
        names, columns = arrow
        return arrow_fn(names, columns, graph, **common)
    names, columns = as_columns(data)
    return numpy_fn(names, columns, graph, **common)


def handle_static_ate(
    data: Any,
    query: AverageEffect,
    *,
    graph: Any,
    inference: Frequentist | Bayesian,
    identifier: str | None,
    estimator: str | None,
    refute: bool | str,
    validators: Sequence[Any] | None,
    seed: int,
    bootstrap: int | None,
    threads: int,
    running_variable: str | None,
    cutoff: float | None,
    bandwidth: float | None,
    estimator_config: Mapping[str, Any] | None,
    population_registry: Any | None,
    latency: str | None,
    cancel: Any | None,
    on_progress: Any | None,
    on_stage: Any | None,
    return_posterior_artifact: bool,
    structure_accepted: bool = False,
) -> Any:
    from .estimation import (
        _bayesian_inference_kwargs,
        _static_edges,
        _wrap_ate,
    )
    from .population import coerce_target_population, registry_wire

    bayes_kw: dict[str, Any] = {}
    if isinstance(inference, Bayesian):
        bayes_kw = _bayesian_inference_kwargs(inference)
    if estimator == "rd.sharp" or any(v is not None for v in (running_variable, cutoff, bandwidth)):
        # The triple may arrive either as loose kwargs or inside `estimator_config`;
        # Rust merges the two, so this gate must look at both or it rejects the
        # typed spelling before it ever reaches the merge.
        cfg = estimator_config or {}
        running_variable = (
            running_variable if running_variable is not None else cfg.get("running_variable")
        )
        cutoff = cutoff if cutoff is not None else cfg.get("cutoff")
        bandwidth = bandwidth if bandwidth is not None else cfg.get("bandwidth")
        if running_variable is None or cutoff is None or bandwidth is None:
            raise ValueError(
                "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth"
            )
        if estimator is None:
            estimator = "rd.sharp"
        if identifier is None:
            identifier = "rd.sharp"
    common = dict(
        treatment=query.treatment,
        outcome=query.outcome,
        control_level=query.control_level,
        active_level=query.active_level,
        identifier=identifier,
        estimator=estimator,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        running_variable=running_variable,
        cutoff=cutoff,
        bandwidth=bandwidth,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        **bayes_kw,
    )
    if estimator_config is not None:
        common["estimator_config"] = dict(estimator_config)
    if return_posterior_artifact:
        common["return_posterior_artifact"] = True
    if latency is not None:
        common["latency"] = latency
    if cancel is not None:
        common["cancel"] = cancel
    if on_progress is not None:
        common["on_progress"] = on_progress
    if on_stage is not None:
        common["on_stage"] = on_stage

    pop = coerce_target_population(getattr(query, "target_population", None))
    from .query import coerce_outcome_functional

    functional = coerce_outcome_functional(getattr(query, "outcome_functional", None))
    preds, dists = registry_wire(population_registry)
    pop_kw: dict[str, Any] = {}
    if pop is not None:
        pop_kw["target_population"] = pop
    if functional is not None:
        pop_kw["outcome_functional"] = functional
    if preds:
        pop_kw["population_predicates"] = preds
    if dists:
        pop_kw["population_distributions"] = dists
    if isinstance(graph, TieredBackground):
        if pop is not None or preds or dists:
            raise CausalUnsupportedError(
                "tiered Python execution does not yet accept target populations"
            )
        names, columns = ingest_columns(data)
        return _wrap_ate(
            _analyze_ate_tiered(
                names,
                columns,
                [list(tier) for tier in graph.tiers],
                str(graph.within_tier),
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                estimator=estimator,
                refute=refute,
                seed=seed,
                bootstrap=bootstrap or 0,
                threads=threads,
                outcome_functional=functional,
            ),
            query=query,
        )
    if pop_kw and isinstance(graph, (Pag, Cpdag, Admg)):
        raise ValueError(
            "target_population / population_registry currently require a Dag "
            "(or edge list); PAG/CPDAG/ADMG analyze paths do not accept them yet"
        )
    if isinstance(graph, Pag):
        return _wrap_ate(
            _typed_ate(data, graph, _analyze_ate_pag, _analyze_ate_pag_arrow_c, **common),
            query=query,
        )
    if isinstance(graph, Cpdag):
        return _wrap_ate(
            _typed_ate(data, graph, _analyze_ate_cpdag, _analyze_ate_cpdag_arrow_c, **common),
            query=query,
        )
    if isinstance(graph, Admg):
        return _wrap_ate(
            _typed_ate(data, graph, _analyze_ate_admg, _analyze_ate_admg_arrow_c, **common),
            query=query,
        )
    edges = _static_edges(graph)
    arrow = try_as_arrow_c_columns(data)
    ate_kwargs = dict(edges=edges, accepted=structure_accepted, **common, **pop_kw)
    use_arrow = arrow is not None and not pop_kw
    if use_arrow:
        assert arrow is not None
        names, columns = arrow
        raw = _analyze_ate_arrow_c(names, columns, **ate_kwargs)
    else:
        names, columns = as_columns(data)
        raw = _analyze_ate(names, columns, **ate_kwargs)
    return _wrap_ate(raw, query=query)


def _analyze_jpcmci_plus_discover(
    names: list[str],
    env_columns: Sequence[Sequence[Any]],
    query: PulseEffect | SustainedEffect,
    *,
    policy: str,
    cfg: dict[str, Any],
    accept_discovered: bool,
    bayes_kw: dict[str, Any],
    seed: int,
    bootstrap: int | None,
    threads: int,
    refute: bool | str = False,
) -> Any:
    """Shared ``_analyze_temporal_discover`` call for J-PCMCI+ multi-environment discovery.

    Used by both ``handle_temporal_pulse``'s ``MultiEnvFrame`` branch (``names``/
    ``env_columns`` come straight off the frame) and ``_handle_series_discover``'s
    ``"jpcmci_plus"`` branch (``names``/``env_columns`` come from
    ``as_multi_env_columns(data)``); those two callers differ only in how they
    obtain ``names``/``env_columns``, so this holds the ~20 kwargs common to both.
    """
    return _analyze_temporal_discover(
        names,
        env_columns[0],
        query.treatment,
        query.outcome,
        algorithm="jpcmci_plus",
        max_lag=cfg["max_lag"],
        alpha=cfg["alpha"],
        max_cond_size=cfg.get("max_cond_size", 2),
        fdr=cfg["fdr"],
        accept_discovered=accept_discovered,
        treatment_lag=query.treatment_lag,
        horizon_steps=query.horizon_steps,
        active_level=query.active_level,
        policy=policy,
        **bayes_kw,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        env_columns=env_columns,
        context_names=cfg["context_names"],
        include_space_dummy=cfg["include_space_dummy"],
        include_time_dummy=cfg["include_time_dummy"],
        space_dummy_ci=cfg["space_dummy_ci"],
        time_dummy_encoding=cfg["time_dummy_encoding"],
        time_dummy_ci=cfg["time_dummy_ci"],
        ci=cfg.get("ci"),
        refute=refute,
    )


def handle_temporal_pulse(
    data: Any,
    query: PulseEffect | SustainedEffect,
    *,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    refute: bool | str,
    validators: Sequence[Any] | None,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
    regimes: Sequence[int] | None,
    structure_accepted: bool = False,
) -> Any:
    from .estimation import (
        _discovery_algorithm,
        _lagged_edges,
        _reject_unsupported_temporal,
        _temporal_inference_kwargs,
        _wrap_temporal,
    )

    policy = query.kind  # "pulse" | "sustained" — matches the native policy string directly
    _reject_unsupported_temporal(inference=inference, refute=refute, validators=validators)
    bayes_kw = _temporal_inference_kwargs(inference)
    if isinstance(data, EventFrame):
        return _handle_event_frame(
            data,
            query,
            policy=policy,
            graph=graph,
            discovery=discovery,
            inference=inference,
            bayes_kw=bayes_kw,
            refute=refute,
            validators=validators,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            regimes=regimes,
        )
    if isinstance(data, PanelFrame):
        return _handle_panel_frame(
            data,
            query,
            policy=policy,
            graph=graph,
            discovery=discovery,
            bayes_kw=bayes_kw,
            refute=refute,
            validators=validators,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
    if isinstance(data, MultiEnvFrame):
        if discovery is None or not isinstance(discovery, JPCMCIPlus):
            raise TypeError("MultiEnvFrame requires discovery=JPCMCIPlus(...)")
        cfg = _discovery_algorithm(discovery)
        raw = _analyze_jpcmci_plus_discover(
            data.names,
            data.env_columns,
            query,
            policy=policy,
            cfg=cfg,
            accept_discovered=accept_discovered,
            bayes_kw=bayes_kw,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            refute=refute,
        )
        return _wrap_temporal(raw)
    if discovery is not None:
        return _handle_series_discover(
            data,
            query,
            policy=policy,
            discovery=discovery,
            inference=inference,
            bayes_kw=bayes_kw,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            regimes=regimes,
            temporal_discovery=_TEMPORAL_DISCOVERY,
            refute=refute,
        )
    names, columns = ingest_columns(data)
    if isinstance(graph, TemporalPag):
        raw = _analyze_temporal_pag(
            names,
            columns,
            graph,
            query.treatment,
            query.outcome,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            accepted=structure_accepted,
        )
        return _wrap_temporal(raw)
    if isinstance(graph, TemporalCpdag):
        raw = _analyze_temporal_cpdag(
            names,
            columns,
            graph,
            query.treatment,
            query.outcome,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            accepted=structure_accepted,
        )
        return _wrap_temporal(raw)
    lagged = _lagged_edges(graph)
    raw = _analyze_temporal(
        names,
        columns,
        lagged,
        query.treatment,
        query.outcome,
        treatment_lag=query.treatment_lag,
        horizon_steps=query.horizon_steps,
        active_level=query.active_level,
        policy=policy,
        **bayes_kw,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_temporal(raw)


def _handle_event_frame(
    data: EventFrame,
    query: PulseEffect | SustainedEffect,
    *,
    policy: str,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    bayes_kw: dict[str, Any],
    refute: bool | str,
    validators: Sequence[Any] | None,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
    regimes: Sequence[int] | None,
) -> Any:
    from .estimation import _discovery_algorithm, _lagged_edges, _wrap_temporal

    if discovery is not None:
        if isinstance(discovery, JPCMCIPlus):
            raise TypeError(
                "EventFrame does not support discovery=JPCMCIPlus(...); "
                "use MultiEnvFrame or PanelFrame for multi-environment discovery"
            )
        if isinstance(discovery, DbnPosterior):
            if not isinstance(inference, Bayesian):
                raise TypeError(
                    "EventFrame discovery=DbnPosterior(...) requires inference=Bayesian(...)"
                )
        elif not isinstance(discovery, (PCMCI, PCMCIPlus, LPCMCI, RPCMCI)):
            raise TypeError(
                f"EventFrame discovery expects PCMCI/PCMCIPlus/LPCMCI/RPCMCI/DbnPosterior, "
                f"got {type(discovery)!r}"
            )
        cfg = _discovery_algorithm(discovery)
        raw = _analyze_events(
            data.names,
            data.columns,
            data.event_times_ns.tolist(),
            data.align_interval_ns,
            [],
            query.treatment,
            query.outcome,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            algorithm=cfg["algorithm"],
            max_lag=cfg.get("max_lag", 1),
            alpha=cfg.get("alpha", 0.05),
            max_cond_size=cfg.get("max_cond_size", 2),
            fdr=cfg.get("fdr", True),
            accept_discovered=accept_discovered,
            regimes=list(regimes) if regimes is not None else None,
            **{
                k: cfg[k]
                for k in ("n_chains", "n_warmup", "mcmc_draws", "force_mcmc", "ci")
                if k in cfg
            },
        )
        return _wrap_temporal(raw)
    lagged = _lagged_edges(graph)
    raw = _analyze_events(
        data.names,
        data.columns,
        data.event_times_ns.tolist(),
        data.align_interval_ns,
        lagged,
        query.treatment,
        query.outcome,
        treatment_lag=query.treatment_lag,
        horizon_steps=query.horizon_steps,
        active_level=query.active_level,
        policy=policy,
        **bayes_kw,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_temporal(raw)


def _handle_panel_frame(
    data: PanelFrame,
    query: PulseEffect | SustainedEffect,
    *,
    policy: str,
    graph: Any,
    discovery: Any,
    bayes_kw: dict[str, Any],
    refute: bool | str,
    validators: Sequence[Any] | None,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import _discovery_algorithm, _lagged_edges, _wrap_temporal

    if discovery is not None:
        if isinstance(discovery, JPCMCIPlus):
            cfg = _discovery_algorithm(discovery)
            raw = _analyze_panel_discover(
                data.names,
                data.unit_columns,
                data.unit_ids,
                query.treatment,
                query.outcome,
                max_lag=cfg["max_lag"],
                alpha=cfg["alpha"],
                max_cond_size=cfg.get("max_cond_size", 2),
                fdr=cfg["fdr"],
                accept_discovered=accept_discovered,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                policy=policy,
                **bayes_kw,
                refute=refute,
                validators=list(validators) if validators is not None else None,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
                context_names=cfg["context_names"],
                include_space_dummy=cfg["include_space_dummy"],
                include_time_dummy=cfg["include_time_dummy"],
                space_dummy_ci=cfg["space_dummy_ci"]
                in ("multivariate", "multivariate_block", "block", True),
                time_dummy_encoding=cfg["time_dummy_encoding"],
                time_dummy_ci=cfg["time_dummy_ci"]
                in ("multivariate", "multivariate_block", "block", True),
                ci=cfg["ci"],
            )
            return _wrap_temporal(raw)
        if isinstance(discovery, (PCMCI, PCMCIPlus, LPCMCI)):
            cfg = _discovery_algorithm(discovery)
            raw = _analyze_panel_discover(
                data.names,
                data.unit_columns,
                data.unit_ids,
                query.treatment,
                query.outcome,
                max_lag=cfg["max_lag"],
                alpha=cfg["alpha"],
                max_cond_size=cfg.get("max_cond_size", 2),
                fdr=cfg["fdr"],
                accept_discovered=accept_discovered,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                policy=policy,
                **bayes_kw,
                refute=refute,
                validators=list(validators) if validators is not None else None,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
                algorithm=cfg["algorithm"],
                ci=cfg["ci"],
            )
            return _wrap_temporal(raw)
        raise TypeError("PanelFrame discovery supports JPCMCIPlus, PCMCI, PCMCIPlus, or LPCMCI")
    lagged = _lagged_edges(graph)
    raw = _analyze_panel(
        data.names,
        data.unit_columns,
        data.unit_ids,
        lagged,
        query.treatment,
        query.outcome,
        treatment_lag=query.treatment_lag,
        horizon_steps=query.horizon_steps,
        active_level=query.active_level,
        policy=policy,
        **bayes_kw,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_temporal(raw)


def _handle_series_discover(
    data: Any,
    query: PulseEffect | SustainedEffect,
    *,
    policy: str,
    discovery: Any,
    inference: Frequentist | Bayesian,
    bayes_kw: dict[str, Any],
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
    regimes: Sequence[int] | None,
    temporal_discovery: tuple[type, ...],
    refute: bool | str,
) -> Any:
    from .estimation import _discovery_algorithm, _wrap_temporal

    if isinstance(discovery, DbnPosterior):
        if not isinstance(inference, Bayesian):
            raise TypeError(
                "discovery=DbnPosterior(...) requires inference=Bayesian(...) "
                "for temporal effect mixture"
            )
        cfg = _discovery_algorithm(discovery)
        names, columns = as_columns(data)
        raw = _analyze_temporal_discover(
            names,
            columns,
            query.treatment,
            query.outcome,
            algorithm="dbn_posterior",
            max_lag=cfg["max_lag"],
            accept_discovered=accept_discovered,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            n_chains=cfg["n_chains"],
            n_warmup=cfg["n_warmup"],
            mcmc_draws=cfg["mcmc_draws"],
            force_mcmc=cfg["force_mcmc"],
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            refute=refute,
        )
        return _wrap_temporal(raw)
    if not isinstance(discovery, temporal_discovery):
        raise TypeError(
            f"temporal discovery expects PCMCI-family or DbnPosterior, got {type(discovery)!r}"
        )
    cfg = _discovery_algorithm(discovery)
    algo = cfg["algorithm"]
    if algo == "jpcmci_plus":
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes, Mapping)):
            raise TypeError(
                "discovery=JPCMCIPlus(...) requires data as a sequence of "
                "environment mappings/DataFrames"
            )
        names, env_columns = as_multi_env_columns(data)
        raw = _analyze_jpcmci_plus_discover(
            names,
            env_columns,
            query,
            policy=policy,
            cfg=cfg,
            accept_discovered=accept_discovered,
            bayes_kw=bayes_kw,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            refute=refute,
        )
        return _wrap_temporal(raw)
    if algo == "rpcmci":
        if regimes is None:
            raise ValueError("discovery=RPCMCI(...) requires regimes=[…] labels")
        names, columns = as_columns(data)
        raw = _analyze_temporal_discover(
            names,
            columns,
            query.treatment,
            query.outcome,
            algorithm=algo,
            max_lag=cfg["max_lag"],
            alpha=cfg["alpha"],
            max_cond_size=cfg.get("max_cond_size", 2),
            fdr=cfg["fdr"],
            accept_discovered=accept_discovered,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            regimes=list(regimes),
            ci=cfg.get("ci"),
            refute=refute,
        )
        return _wrap_temporal(raw)
    names, columns = as_columns(data)
    raw = _analyze_temporal_discover(
        names,
        columns,
        query.treatment,
        query.outcome,
        algorithm=algo,
        max_lag=cfg["max_lag"],
        alpha=cfg["alpha"],
        max_cond_size=cfg.get("max_cond_size", 2),
        fdr=cfg["fdr"],
        accept_discovered=accept_discovered,
        treatment_lag=query.treatment_lag,
        horizon_steps=query.horizon_steps,
        active_level=query.active_level,
        policy=policy,
        **bayes_kw,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        ci=cfg.get("ci"),
        refute=refute,
    )
    return _wrap_temporal(raw)


# Kinds whose routing is a pure function of `query.kind` — the handler never
# depends on `discovery`'s *type*. "average" / "pulse" / "sustained" are
# deliberately NOT here: a static/graph-posterior `discovery=` value changes
# which handler runs (or raises) ahead of the query kind for those three, so
# they stay as explicit sequential checks below, in the original ladder's
# order, to keep that interaction visible rather than hidden in a table.
#
# Each entry pairs a `handle_*` function with the exact subset of the shared
# `kw` dict (built in `analyze()`) that its keyword-only parameters accept —
# the per-kind key set is the only information the old `_dispatch_*` wrappers
# carried, so it is kept explicit here rather than dispatched dynamically.
_KIND_HANDLER_KEYS: dict[str, tuple[Callable[..., Any], tuple[str, ...]]] = {
    **{
        kind: (
            handle_response,
            (
                "graph",
                "discovery",
                "inference",
                "identifier",
                "estimator",
                "estimator_config",
                "validators",
                "refute_requested",
                "refute",
                "bootstrap_requested",
                "seed",
                "threads",
                "structure_accepted",
            ),
        )
        for kind in (
            "response_curve",
            "average_derivative",
            "point_derivative",
            "elasticity",
            "semi_elasticity",
            "directional_derivative",
            "response_jacobian",
            "intervention_response",
        )
    },
    "conditional": (
        handle_conditional,
        (
            "graph",
            "discovery",
            "inference",
            "refute",
            "validators",
            "seed",
            "bootstrap",
            "threads",
            "structure_accepted",
        ),
    ),
    "temporal_mediation": (
        handle_temporal_mediation,
        ("graph", "discovery", "inference", "refute", "seed", "bootstrap", "threads"),
    ),
    "mediation": (
        handle_mediation,
        ("graph", "discovery", "refute", "seed", "bootstrap", "threads"),
    ),
    "counterfactual": (
        handle_counterfactual,
        ("graph", "discovery", "seed", "threads"),
    ),
    "distribution": (
        handle_distribution,
        (
            "graph",
            "discovery",
            "accept_discovered",
            "refute_requested",
            "refute",
            "seed",
            "threads",
        ),
    ),
    "path_specific": (
        handle_path_specific,
        (
            "graph",
            "discovery",
            "accept_discovered",
            "refute_requested",
            "refute",
            "seed",
            "bootstrap",
            "threads",
        ),
    ),
}


def _dispatch_kind(data: Any, query: Any, kind: str, kw: dict[str, Any]) -> Any:
    handler, keys = _KIND_HANDLER_KEYS[kind]
    return handler(data, query, **{key: kw[key] for key in keys})


def analyze(
    data: Mapping[str, Any] | Any | Sequence[Mapping[str, Any] | Any],
    *,
    query: (
        AverageEffect
        | PulseEffect
        | SustainedEffect
        | InterventionalDistribution
        | PathSpecificEffect
        | ConditionalEffect
        | MediationEffect
        | Counterfactual
        | TemporalMediationEffect
        | ResponseCurve
        | AverageDerivative
        | PointDerivative
        | Elasticity
        | SemiElasticity
        | DirectionalDerivative
        | ResponseJacobian
        | InterventionResponse
    ),
    graph: (
        Dag
        | Cpdag
        | Pag
        | Admg
        | TemporalDag
        | TemporalCpdag
        | TemporalPag
        | TieredBackground
        | Sequence[tuple[str, str]]
        | Sequence[tuple[str, int, str, int]]
        | None
    ) = None,
    discovery: Any | None = None,
    inference: Frequentist | Bayesian | None = None,
    identifier: str | Identifier | None = None,
    estimator: str | Estimator | EstimatorConfigLike | None = None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
    validators: Sequence[Any] | None = None,
    accept_discovered: bool = True,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    regimes: Sequence[int] | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    population_registry: Any | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    latency: Latency | Literal["interactive", "standard", "report"] | None = None,
    cancel: Any | None = None,
    on_progress: Any | None = None,
    on_stage: Any | None = None,
    return_posterior_artifact: bool = False,
) -> AnalysisResult | CausalResponseView:
    """Identify then estimate a causal effect.

    Parameters
    ----------
    data:
        Mapping of column name → 1-d float array, a pandas ``DataFrame``,
        Arrow CDI exporters (PyArrow columns / table), or a
        ``antecedent.data`` frame (``EventFrame`` / ``PanelFrame`` / ``MultiEnvFrame``).
        For ``discovery=JPCMCIPlus(...)``, pass a sequence of environment frames
        or a ``MultiEnvFrame``.
    query:
        ``AverageEffect``, ``PulseEffect`` / ``SustainedEffect``,
        ``InterventionalDistribution``, ``PathSpecificEffect``,
        ``MediationEffect``, ``Counterfactual``, ``TemporalMediationEffect``, or a
        response-family query. Response-family queries return
        :class:`antecedent.results.CausalResponseView`; other queries return
        :class:`antecedent.AnalysisResult`.
    graph:
        ``Dag`` / ``Cpdag`` / ``Pag`` / ``Admg`` / ``TemporalDag`` /
        ``TemporalCpdag`` / ``TemporalPag``, or an edge list. Lagged edges
        ``(from, from_lag, to, to_lag)`` are required for temporal queries
        without ``discovery``. Incomplete ``Cpdag`` / ``TemporalCpdag`` /
        ``TemporalPag`` keep their class. Completing those graphs yourself is
        still the ``Dag`` / ``TemporalDag`` cell. ADMGs without bidirected
        edges coerce to DAGs; ADMGs with latents use general ID + functional
        effect.
    discovery:
        Static: ``PC`` / ``GES`` / ``LiNGAM`` / ``NOTEARS`` / ``FCI`` / ``RFCI``.
        Temporal: ``PCMCI`` / ``PCMCIPlus`` / ``LPCMCI`` / ``JPCMCIPlus`` / ``RPCMCI``.
        Graph-posterior cells also accept a constructed ``GraphPosterior``.
        One-shot script convenience — discovery runs at compile time. For
        interactive / spreadsheet estimate clicks, discover once into
        :class:`antecedent.AcceptedGraph` (or hold a reviewed graph) and pass
        ``graph=`` with ``latency="interactive"`` instead. Combining
        ``discovery=`` with ``latency="interactive"`` raises
        :class:`CausalUnsupportedError`.
        Live discovery strategies refuse ``cancel=``, ``on_progress=``, and
        ``on_stage=`` because their native entry points do not yet implement
        those controls end to end. A supplied ``GraphPosterior`` is a replay
        path: it supports cancellation and progress but refuses ``on_stage=``.
        Otherwise, discover first and pass the reviewed graph via ``graph=``
        when execution controls are required.
    latency:
        Optional compute tier (``interactive`` / ``standard`` / ``report``).
        Maps to known-equivalent bootstrap / refute / draws; explicit
        ``bootstrap=`` / ``refute=`` always win. Interactive refuses inline
        ``discovery=`` (artifact-first UX).
    refute:
        ``False`` or a suite name (``"full"`` / ``"placebo"`` / ``"cheap"`` /
        ``"none"``) / :class:`antecedent.Refute` member. Leave unset (``None``)
        to run the default suite — passing the literal ``True`` raises
        ``TypeError`` (it carried no information beyond "unset" and was easy
        to confuse with an explicit choice).
    cancel:
        Optional ``CancellationToken`` from ``antecedent._native``. Refused
        with live discovery strategies; supported on compatible ``graph=``
        and supplied-``GraphPosterior`` paths.
    on_progress:
        Optional ``(fraction: float, stage: str) -> None`` callback. Refused
        with live discovery strategies; supported on compatible ``graph=``
        and supplied-``GraphPosterior`` paths.
    on_stage:
        Optional ``(stage: str, payload: dict) -> None`` progressive stage
        callback (identify → estimate_point → uncertainty → validate). Refused
        with every ``discovery=`` path; supported on compatible ``graph=`` paths.
    return_posterior_artifact:
        When ``True`` and inference is Bayesian, attach full posterior draw
        bytes on ``result.posterior.artifact`` (for download / sequential-prior
        hydrate). Default ``False``: UI summaries only.
    """
    # Single source of truth for "is this a known query type" —
    # see `_coerce.coerce_query`'s docstring. Everything past this point in
    # `analyze()` already implicitly requires a supported query (the
    # `_KIND_HANDLER_KEYS` dispatch plus the AverageEffect / Pulse-or-Sustained
    # ladder below cover the same kinds), so this makes the
    # final `raise TypeError` at the bottom of this function unreachable in
    # practice; it stays as a defensive fallback rather than being deleted.
    coerce_query(query)
    # An AcceptedGraph is already a reviewed discovery artifact. Unwrap it here so
    # every query family gets the same artifact-first estimate path without ever
    # re-entering discovery.
    from .accepted_graph import AcceptedGraph

    structure_accepted = False
    if isinstance(graph, AcceptedGraph):
        if discovery is not None:
            raise CausalUnsupportedError(
                "analyze(graph=AcceptedGraph(...)) rejects discovery=; the structure "
                "artifact is already accepted (call rediscover() explicitly to replace it)"
            )
        structure_accepted = True
        graph = graph.graph
    # Explicit no-op spellings are important to staged APIs, whose historical
    # defaults are ``refute=False`` and ``bootstrap=0``. They must not turn a
    # response estimate into an unsupported refutation/bootstrap request.
    refute_requested = refute not in (None, False, "none")
    bootstrap_requested = bootstrap not in (None, 0)
    if isinstance(identifier, Identifier):
        identifier = str(identifier)
    if not isinstance(estimator, (str, Estimator)) and estimator is not None:
        # A typed config from `antecedent.estimators` carries both the id and the
        # config, so accepting both spellings at once would be ambiguous.
        if estimator_config is not None:
            raise ValueError(
                "estimator= already carries its configuration; do not also pass estimator_config="
            )
        estimator_config = estimator._wire()
        estimator = estimator.estimator_id
    if isinstance(estimator, Estimator):
        estimator = str(estimator)
    if latency is not None:
        latency = coerce_latency(latency)  # type: ignore[assignment]
    # Unset preserves the historical default (native's own default suite) via the
    # same `refute is True` sentinel `_resolve_latency_budget` already keys off.
    # Only a caller-supplied value goes through `coerce_refute` — that is what
    # makes an explicit `refute=True` rejectable without breaking every call that
    # does not pass `refute=`.
    resolved_refute: bool | str = True if refute is None else coerce_refute(refute)
    inference = inference or Frequentist()
    bootstrap, resolved_refute = _resolve_latency_budget(latency, bootstrap, resolved_refute)

    if discovery is not None and latency == "interactive":
        raise CausalUnsupportedError(
            "discovery= is not on the interactive estimate path; "
            "run discovery once (Config.accept(data) -> AcceptedGraph), then "
            "analyze(graph=..., latency='interactive')"
        )

    if discovery is not None:
        controls = (
            (("on_stage", on_stage),)
            if isinstance(discovery, GraphPosterior)
            else (
                ("cancel", cancel),
                ("on_progress", on_progress),
                ("on_stage", on_stage),
            )
        )
        unsupported_controls = [name for name, value in controls if value is not None]
        if unsupported_controls:
            raise CausalUnsupportedError(
                "analyze(discovery=...) does not support execution controls "
                f"{', '.join(unsupported_controls)}; discover first and pass the reviewed "
                "structure via graph= when cancellation or callbacks are required"
            )

    kind = getattr(query, "kind", "")
    # 1.2/1.3 coordinates share the staged Rust execution path, including the
    # frozen structure axis, validation reports and posterior serialization.
    use_prepared = (
        kind
        in {"path_specific", "distribution", "temporal_mediation", "mediation", "counterfactual"}
        or (
            isinstance(inference, Bayesian)
            and kind in {"conditional", "response_curve", "intervention_response"}
        )
        or (kind == "sustained" and getattr(query, "window", None) is not None)
    )
    if use_prepared and discovery is None:
        assert isinstance(
            query,
            (
                MediationEffect,
                Counterfactual,
                ConditionalEffect,
                TemporalMediationEffect,
                PathSpecificEffect,
                InterventionalDistribution,
                ResponseCurve,
                InterventionResponse,
                SustainedEffect,
            ),
        )
        if graph is None:
            raise ValueError("this query requires graph=")
        if isinstance(graph, TieredBackground):
            raise CausalUnsupportedError(
                "staged Bayesian / path-specific prepare does not take TieredBackground; "
                "joint CoDetermined cells use Frequentist cell.aipw"
            )
        from .estimation import PreparedAnalysis

        unsupported = [
            name
            for name, value in (
                ("cancel", cancel),
                ("on_progress", on_progress),
                ("on_stage", on_stage),
                ("validators", validators),
                ("estimator_config", estimator_config),
            )
            if value is not None
        ]
        if unsupported:
            raise CausalUnsupportedError(
                "this staged query path does not support " + ", ".join(unsupported)
            )
        is_response = kind in {"response_curve", "intervention_response"}
        if is_response and bootstrap_requested:
            raise CausalUnsupportedError(
                "Bayesian responses use posterior intervals; bootstrap is unsupported"
            )
        suite = (
            "none"
            if (is_response or kind == "counterfactual") and not refute_requested
            else "cheap"
            if resolved_refute is True
            else resolved_refute
        )
        if kind == "counterfactual" and bootstrap_requested and bootstrap:
            raise CausalUnsupportedError("counterfactual sampling uncertainty is unavailable")
        prepared = PreparedAnalysis.prepare(
            data,
            query=query,
            graph=AcceptedGraph(graph) if structure_accepted else graph,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            refute=suite,
            seed=seed,
            bootstrap=0 if kind == "counterfactual" else bootstrap,
            threads=threads,
            latency=latency,
        )
        result = prepared.estimate(data, seed=seed, threads=threads)
        if return_posterior_artifact:
            from dataclasses import replace

            from .results import AnalysisResult

            if not isinstance(result, AnalysisResult) or result.posterior is None:
                raise CausalUnsupportedError(
                    "return_posterior_artifact requires a scalar posterior; use PreparedAnalysis.export_artifact for responses"
                )
            result = replace(
                result, posterior=replace(result.posterior, artifact=prepared.export_artifact())
            )
        return result

    if kind and kind in _KIND_HANDLER_KEYS:
        return _dispatch_kind(
            data,
            query,
            kind,
            {
                "graph": graph,
                "discovery": discovery,
                "inference": inference,
                "identifier": identifier,
                "estimator": estimator,
                "estimator_config": estimator_config,
                "validators": validators,
                "refute_requested": refute_requested,
                "refute": resolved_refute,
                "bootstrap_requested": bootstrap_requested,
                "accept_discovered": accept_discovered,
                "seed": seed,
                "bootstrap": bootstrap,
                "threads": threads,
                "structure_accepted": structure_accepted,
            },
        )

    # "average" / "pulse" / "sustained" route on `discovery`'s *type*, not just
    # `query.kind` — a static/graph-posterior `discovery=` preempts even a
    # Pulse/SustainedEffect query with `handle_static_ate_discover`'s own
    # "requires AverageEffect" error, ahead of ever reaching the temporal-pulse
    # handler below. This sequence mirrors the original isinstance ladder
    # exactly (including that quirk) rather than keying purely on `kind`.
    if isinstance(discovery, GraphPosterior):
        if not isinstance(query, (AverageEffect, PulseEffect, SustainedEffect)):
            raise TypeError(
                "GraphPosterior is licensed for AverageEffect, PulseEffect, and SustainedEffect"
            )
        return handle_supplied_graph_posterior(
            data,
            query,
            discovery=discovery,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            validators=validators,
            population_registry=population_registry,
            return_posterior_artifact=return_posterior_artifact,
            refute=resolved_refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            cancel=cancel,
            on_progress=on_progress,
        )

    if discovery is not None and isinstance(
        discovery, _STATIC_DISCOVERY + _GRAPH_POSTERIOR_DISCOVERY
    ):
        return handle_static_ate_discover(
            data,
            query,  # type: ignore[arg-type]
            discovery=discovery,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            refute=resolved_refute,
            validators=validators,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            estimator_config=estimator_config,
        )

    if discovery is not None and kind == "average":
        raise ValueError(
            "AverageEffect with discovery= requires a static algorithm "
            "(PC/GES/LiNGAM/NOTEARS/FCI/RFCI); temporal discovery needs "
            "PulseEffect/SustainedEffect"
        )

    if isinstance(query, AverageEffect):
        return handle_static_ate(
            data,
            query,
            graph=graph,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            refute=resolved_refute,
            validators=validators,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            running_variable=running_variable,
            cutoff=cutoff,
            bandwidth=bandwidth,
            estimator_config=estimator_config,
            population_registry=population_registry,
            latency=latency,
            cancel=cancel,
            on_progress=on_progress,
            on_stage=on_stage,
            return_posterior_artifact=return_posterior_artifact,
            structure_accepted=structure_accepted,
        )

    if isinstance(query, (PulseEffect, SustainedEffect)):
        return handle_temporal_pulse(
            data,
            query,
            graph=graph,
            discovery=discovery,
            inference=inference,
            refute=resolved_refute,
            validators=validators,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            structure_accepted=structure_accepted,
            regimes=regimes,
        )

    raise TypeError(f"unsupported query type: {type(query)!r}")
