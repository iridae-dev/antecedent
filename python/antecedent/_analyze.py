"""Analyze entrypoint and private query handlers.

``analyze`` is the public router; branch bodies live here so new query types
extend via a handler rather than growing a monolith.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from .results import CausalResponseView

from ._coerce import coerce_latency, coerce_query, coerce_refute
from ._data import as_columns, as_multi_env_columns, try_as_arrow_c_columns
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
    analyze_ate_arrow_c as _analyze_ate_arrow_c,
)
from ._native import (
    analyze_ate_cpdag as _analyze_ate_cpdag,
)
from ._native import (
    analyze_ate_discover as _analyze_ate_discover,
)
from ._native import (
    analyze_ate_pag as _analyze_ate_pag,
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
    analyze_temporal_discover as _analyze_temporal_discover,
)
from ._native import (
    analyze_temporal_mediation as _analyze_temporal_mediation,
)
from ._native import (
    analyze_temporal_pag as _analyze_temporal_pag,
)
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
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag
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

    if isinstance(inference, Bayesian):
        raise TypeError("ConditionalEffect does not support inference=Bayesian(...)")
    if discovery is not None:
        raise ValueError("ConditionalEffect does not support discovery=")
    names, columns = as_columns(data)
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
        refute=refute,
        validators=list(validators) if validators is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        accepted=structure_accepted,
    )
    return _wrap_ate(raw)


def handle_temporal_mediation(
    data: Any,
    query: TemporalMediationEffect,
    *,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import _lagged_edges, _wrap_temporal

    if isinstance(inference, Bayesian):
        raise TypeError("TemporalMediationEffect does not support inference=Bayesian(...)")
    if discovery is not None:
        raise ValueError("TemporalMediationEffect does not support discovery=")
    names, columns = as_columns(data)
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
    from .estimation import _static_edges
    from .results import (
        CausalResponseView,
        IdentificationView,
        ResponseEnvelopeView,
        ResponseUncertainty,
        ResponseValidationCheck,
        ResponseValidationView,
        ResponseView,
        SupportDiagnostic,
        SupportReport,
    )
    from .results.response import SupportStatus, UncertaintyKind

    if discovery is not None:
        if isinstance(discovery, _GRAPH_POSTERIOR_DISCOVERY):
            raise CausalUnsupportedError(
                "not_applicable: Structural uncertainty around curves is contrast-only; "
                "graph-posterior mixtures do not license a response cell."
            )
        raise ValueError("response queries do not yet support discovery=")
    if isinstance(inference, Bayesian):
        raise TypeError("response queries do not yet support inference=Bayesian(...)")
    # Derivative cells (AverageDerivative/DirectionalDerivative/Elasticity/PointDerivative/
    # ResponseJacobian/SemiElasticity) used to be refused right here with a hand-typed
    # literal duplicating parity/support_closed.toml. That duplication was a drift risk:
    # the native `analyze_response` (python/src/response_api.rs) now consults the
    # generated support matrix itself (via `antecedent::support::refuse_if_not_applicable`)
    # before running, and raises the identical "refused: ..." text straight from the TOML,
    # so the hand-rolled check here was redundant and has been removed.
    #
    # The Pag/Admg/Cpdag check below must stay: unlike the derivative check, it is not
    # just a refusal message, it is the routing gate that decides whether this call
    # reaches `_analyze_response` at all versus `_analyze_response_pag` a few lines down.
    # Both literals are asserted against parity/support_closed.toml in
    # test_response_support_matrix.py so neither can silently drift. `InterventionResponse`
    # gets its own closed-rule text (parity/support_closed.toml's InterventionResponse ×
    # [Cpdag, Admg, Pag] rule) rather than reusing the ResponseCurve one below it, since the
    # ResponseCurve wording would misdescribe an InterventionResponse refusal.
    if isinstance(graph, (Admg, Cpdag, Pag)):
        if isinstance(query, InterventionResponse):
            raise CausalUnsupportedError(
                "refused: InterventionResponse executes only on a supplied static Dag, the "
                "same requirement ResponseCurve is closed on above; Cpdag/Admg/Pag have no "
                "Response compile arm."
            )
        raise CausalUnsupportedError("refused: ResponseCurve is licensed only on a Dag.")
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
    names, columns = as_columns(data)
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
            elif isinstance(spec, (intervention_specs.Soft, intervention_specs.Sequence)):
                raise CausalUnsupportedError(
                    f"{type(spec).__name__} interventions require a structural/temporal "
                    "model and are not estimable by response.intervention_gcomp"
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
    region = {
        name: (lower, upper)
        for name, lower, upper in zip(
            raw.treatments, raw.support_minima, raw.support_maxima, strict=True
        )
    }
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
    validation = None
    if refute_requested and refute is not False:
        checks = [
            ResponseValidationCheck(
                "overlap.support",
                "passed" if raw.support_status == "supported" else "failed",
                None,
                None,
                f"curve support status is {raw.support_status!r}",
            )
        ]
        if isinstance(query, ResponseCurve):
            import numpy as np

            rng = np.random.default_rng(seed)
            subset_curves: list[list[list[float]]] = []
            n_rows = len(columns[0])
            subset_size = max(2, int(0.8 * n_rows))
            for _ in range(10):
                rows = np.sort(rng.choice(n_rows, size=subset_size, replace=False))
                subset_columns = [column[rows] for column in columns]
                if isinstance(graph, Pag):
                    subset_raw = _analyze_response_pag(
                        names,
                        subset_columns,
                        graph,
                        query.treatment,
                        query.outcome,
                        list(query.grid),
                    )
                    # Envelope widths, not completion-conditioned point curves, are the
                    # honest graph-class sensitivity target.
                    assert subset_raw.lower is not None and subset_raw.upper is not None
                    subset_curves.append(subset_raw.lower + subset_raw.upper)
                else:
                    subset_raw = _analyze_response(
                        names,
                        subset_columns,
                        edges,
                        query.kind,
                        treatments,
                        outcomes,
                        grid=list(query.grid),
                        scale=scale,
                        weighting=weighting,
                        accepted=structure_accepted,
                    )
                    subset_curves.append(subset_raw.values)
            baseline_rows = (
                (raw.lower or []) + (raw.upper or []) if isinstance(graph, Pag) else raw.values
            )
            baseline_array = np.asarray(baseline_rows, dtype=float)
            subset_mean = np.asarray(subset_curves, dtype=float).mean(axis=0)
            max_shift = float(np.max(np.abs(subset_mean - baseline_array)))
            checks.append(
                ResponseValidationCheck(
                    "data.subset",
                    "informative",
                    max_shift,
                    None,
                    "maximum absolute shift of the mean curve/envelope across ten "
                    "deterministic 80% row-subset refits; no universal pass threshold is imposed",
                    10,
                )
            )
        checks.append(
            ResponseValidationCheck(
                "scalar_ate_refuters",
                "skipped",
                None,
                None,
                "placebo treatment, dummy outcome, random common cause, and scalar "
                "sensitivity refuters do not define a function-valued curve check",
            )
        )
        validation = ResponseValidationView(checks)
    identification_operation = (
        "identify.generalized_adjustment" if isinstance(graph, Pag) else "identify.response"
    )
    provenance = {
        "operation_id": raw.provenance_id,
        "operation_ids": [identification_operation, raw.provenance_id],
    }
    if validation is not None:
        provenance["validation_operation_ids"] = [
            "validate.overlap",
            "validate.response_data_subset",
        ]
    return CausalResponseView(
        estimand=query,
        response=response,
        estimate=raw.scalar if raw.scalar is not None else raw.matrix,
        uncertainty=uncertainty,
        support=SupportReport(
            cast(SupportStatus, raw.support_status), region, diagnostics, raw.warnings
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
        validation=validation,
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
    del data, query, graph, discovery, refute, seed, bootstrap, threads
    raise CausalUnsupportedError("refused: MediationEffect is not on the staged handle.")


def handle_counterfactual(
    data: Any,
    query: Counterfactual,
    *,
    graph: Any,
    discovery: Any,
    seed: int,
    threads: int,
) -> Any:
    del data, query, graph, discovery, seed, threads
    raise CausalUnsupportedError("refused: Counterfactual is not on the staged handle.")


def handle_distribution(
    data: Any,
    query: InterventionalDistribution,
    *,
    graph: Any,
    discovery: Any,
    accept_discovered: bool,
    seed: int,
    threads: int,
) -> Any:
    from .estimation import (
        _resolve_static_discovery_edges,
        _static_edges,
        _wrap_ate,
    )

    if discovery is not None:
        raise CausalUnsupportedError(
            "refused: Path and distribution queries are licensed only as explicit "
            "Dag cells; accepted and graph-posterior structures are not staged."
        )
    else:
        edges = _static_edges(graph)
    names, columns = as_columns(data)
    raw = _analyze_distribution(
        names,
        columns,
        edges,
        query.outcome,
        dict(query.interventions),
        conditioning=list(query.conditioning) or None,
        seed=seed,
        threads=threads,
    )
    return _wrap_ate(raw)


def handle_path_specific(
    data: Any,
    query: PathSpecificEffect,
    *,
    graph: Any,
    discovery: Any,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import (
        _resolve_static_discovery_edges,
        _static_edges,
        _wrap_ate,
    )

    if discovery is not None:
        raise CausalUnsupportedError(
            "refused: Path and distribution queries are licensed only as explicit "
            "Dag cells; accepted and graph-posterior structures are not staged."
        )
    else:
        edges = _static_edges(graph)
    names, columns = as_columns(data)
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
    )
    return _wrap_ate(raw)


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
    if isinstance(discovery, _GRAPH_POSTERIOR_DISCOVERY) and not isinstance(inference, Bayesian):
        raise TypeError(
            "graph-posterior discovery requires inference=Bayesian(...) for effect mixture"
        )
    if not isinstance(discovery, _STATIC_DISCOVERY + _GRAPH_POSTERIOR_DISCOVERY):
        raise TypeError(f"unsupported static discovery: {type(discovery)!r}")
    names, columns = as_columns(data)
    cfg = _discovery_algorithm(discovery)
    bayes_kw: dict[str, Any] = {}
    if isinstance(inference, Bayesian):
        bayes_kw = _bayesian_inference_kwargs(inference)
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
    return _wrap_ate(raw)


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
    preds, dists = registry_wire(population_registry)
    pop_kw: dict[str, Any] = {}
    if pop is not None:
        pop_kw["target_population"] = pop
    if preds:
        pop_kw["population_predicates"] = preds
    if dists:
        pop_kw["population_distributions"] = dists
    if pop_kw and isinstance(graph, (Pag, Cpdag, Admg)):
        raise ValueError(
            "target_population / population_registry currently require a Dag "
            "(or edge list); PAG/CPDAG/ADMG analyze paths do not accept them yet"
        )
    if isinstance(graph, Pag):
        names, columns = as_columns(data)
        return _wrap_ate(_analyze_ate_pag(names, columns, graph, **common))
    if isinstance(graph, Cpdag):
        names, columns = as_columns(data)
        return _wrap_ate(_analyze_ate_cpdag(names, columns, graph, **common))
    if isinstance(graph, Admg):
        names, columns = as_columns(data)
        return _wrap_ate(_analyze_ate_admg(names, columns, graph, **common))
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
    return _wrap_ate(raw)


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
) -> Any:
    if isinstance(query, SustainedEffect):
        raise CausalUnsupportedError("refused: SustainedEffect is not on the staged handle.")
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
        )
    names, columns = as_columns(data)
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
        )
        return _wrap_temporal(raw)
    if isinstance(graph, TemporalCpdag):
        try:
            graph = graph.try_into_temporal_dag()
        except Exception as exc:  # noqa: BLE001 — surface orientation failures
            raise ValueError(
                "TemporalCpdag has undirected/conflict marks; orient edges "
                "(try_into_temporal_dag) before analyze, or use discovery review"
            ) from exc
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
        ("graph", "discovery", "inference", "seed", "bootstrap", "threads"),
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
        ("graph", "discovery", "accept_discovered", "seed", "threads"),
    ),
    "path_specific": (
        handle_path_specific,
        ("graph", "discovery", "accept_discovered", "seed", "bootstrap", "threads"),
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
        without ``discovery``. Fully oriented CPDAGs run as DAGs; incomplete
        CPDAGs require review. ADMGs without bidirected edges coerce to DAGs;
        ADMGs with latents use general ID + functional effect.
    discovery:
        Static: ``PC`` / ``GES`` / ``LiNGAM`` / ``NOTEARS`` / ``FCI`` / ``RFCI``.
        Temporal: ``PCMCI`` / ``PCMCIPlus`` / ``LPCMCI`` / ``JPCMCIPlus`` / ``RPCMCI``.
        One-shot script convenience — discovery runs at compile time. For
        interactive / spreadsheet estimate clicks, discover once into
        :class:`antecedent.AcceptedGraph` (or hold a reviewed graph) and pass
        ``graph=`` with ``latency="interactive"`` instead. Combining
        ``discovery=`` with ``latency="interactive"`` raises
        :class:`CausalUnsupportedError`.
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
        Optional ``CancellationToken`` from ``antecedent._native``.
    on_progress:
        Optional ``(fraction: float, stage: str) -> None`` callback.
    on_stage:
        Optional ``(stage: str, payload: dict) -> None`` progressive stage
        callback (identify → estimate_point → uncertainty → validate).
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

    kind = getattr(query, "kind", "")
    if structure_accepted and kind in {"path_specific", "distribution"}:
        raise CausalUnsupportedError(
            "refused: Path and distribution queries are licensed only as explicit "
            "Dag cells; accepted and graph-posterior structures are not staged."
        )

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
            regimes=regimes,
        )

    raise TypeError(f"unsupported query type: {type(query)!r}")
