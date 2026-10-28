"""Prepare-route dispatch for :class:`~antecedent.prepared.PreparedAnalysis`.

The route body is unchanged. Helpers that the rest of ``estimation`` still
calls stay in that module and are imported here. ``estimation`` loads this
module only after those helpers exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from ._columns import resolved, resolved_opt, resolved_scalar
from ._native import PreparedAnalysis as _NativePreparedAnalysis
from ._temporal_design import TEMPORAL_DESIGN_QUERIES, temporal_design_keywords
from .descriptive import Association
from .discovery import DbnPosterior, ExactDagPosterior, GraphPosterior
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .estimation import _RESPONSE_FAMILY, _not_applicable, _static_edges
from .experiment import (
    BernoulliAssignment,
    ComplierEffect,
    FactorialRandomization,
    MultiArmExperimentDesign,
    RandomizedEffect,
    StratifiedRandomization,
    SwitchbackEffect,
    TreatmentOnTreated,
)
from .graph import (
    Admg,
    Cpdag,
    Dag,
    Pag,
    TemporalAdmg,
    TemporalCpdag,
    TemporalDag,
    TemporalPag,
    TieredBackground,
)
from .inference import (
    Bayesian,
    ClassPrior,
    Frequentist,
    _class_prior_kwargs,
    _max_completions_kwargs,
)
from .interference import (
    ClusterRandomization,
    CompleteRandomization,
    InterferenceQuery,
)
from .policy import (
    BinaryPolicy,
    ConditionalDoseResponse,
    MultiActionPolicy,
    MultiActionPolicyValue,
    PolicyValue,
    evaluate_multi_action_policy,
    evaluate_policy,
)
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
    FeedbackMechanism,
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
from .survival import CompetingRisksOutcome, SurvivalOutcome
from .transport.advanced import TransportQuery

#: Nuisance and interval options every continuous-response estimator reads.
_RESPONSE_NUISANCE_KEYS = frozenset(
    {
        "bandwidth",
        "confidence_level",
        "folds",
        "nuisance_basis",
        "nuisance_lambda",
        "minimum_local_ess",
    }
)
#: Curve-only band and diagnostic options, on top of the nuisance options.
_RESPONSE_CONFIG_KEYS = _RESPONSE_NUISANCE_KEYS | frozenset(
    {"simultaneous_replicates", "multiplier_seed", "export_row_diagnostics"}
)
_RESPONSE_ESTIMATORS = {
    "response_curve": "response.kennedy_dr",
    "point_derivative": "response.kennedy_dr",
    "elasticity": "response.kennedy_dr",
    "semi_elasticity": "response.kennedy_dr",
    "average_derivative": "response.riesz_ade",
    "directional_derivative": "response.gam_derivative",
    "response_jacobian": "response.gam_derivative",
    "intervention_response": "response.intervention_gcomp",
}


def _check_response_strategy(
    query: Any,
    *,
    graph: Any,
    identifier: str | None,
    estimator: str | None,
    inference: Frequentist | Bayesian | None,
) -> None:
    if (
        isinstance(query, InterventionResponse)
        and estimator == "cell.aipw"
        and not isinstance(inference, Bayesian)
    ):
        # A Dag cell-AIPW response identifies by response backdoor adjustment
        # (the Rust response strategy table); a tiered background routes earlier.
        if identifier not in (None, "response.backdoor"):
            raise ValueError(
                f"{query.kind} requires identifier='response.backdoor'; got {identifier!r}"
            )
        return
    expected_identifier = (
        "generalized.adjustment" if isinstance(graph, (Pag, Cpdag)) else "response.backdoor"
    )
    if identifier not in (None, expected_identifier):
        raise ValueError(
            f"{query.kind} requires identifier={expected_identifier!r}; got {identifier!r}"
        )
    expected = _RESPONSE_ESTIMATORS[query.kind]
    allowed: tuple[str | None, ...] = (None, expected)
    if isinstance(query, InterventionResponse) and isinstance(inference, Bayesian):
        allowed = (None, expected, "response.bayesian")
    if estimator not in allowed:
        raise ValueError(f"{query.kind} requires estimator={expected!r}; got {estimator!r}")


def _parse_response_estimator_config(
    query: Any, estimator_config: Mapping[str, Any] | None
) -> dict[str, Any]:
    if estimator_config is None:
        return {}
    unknown = set(estimator_config) - _RESPONSE_CONFIG_KEYS
    if unknown:
        raise ValueError("unknown response estimator_config keys: " + ", ".join(sorted(unknown)))
    options = dict(estimator_config)
    if "simultaneous_replicates" in options and "bandwidth" not in options:
        raise ValueError(
            "simultaneous response bands require an explicit estimator_config bandwidth"
        )
    if isinstance(query, (PointDerivative, Elasticity, SemiElasticity)):
        if "simultaneous_replicates" in options:
            raise ValueError("simultaneous response bands currently apply to ResponseCurve only")
        if "bandwidth" not in options:
            raise ValueError(
                "PointDerivative/Elasticity/SemiElasticity require estimator_config bandwidth"
            )
        extra = set(options) - _RESPONSE_NUISANCE_KEYS
        if extra:
            raise ValueError(
                "prepared derivatives accept only "
                + ", ".join(sorted(_RESPONSE_NUISANCE_KEYS))
                + " in estimator_config"
            )
    elif isinstance(query, (AverageDerivative, DirectionalDerivative, ResponseJacobian)):
        extra = set(options) - _RESPONSE_NUISANCE_KEYS
        if extra:
            raise ValueError(
                "prepared derivatives accept only "
                + ", ".join(sorted(_RESPONSE_NUISANCE_KEYS))
                + " in estimator_config"
            )
    elif not isinstance(query, ResponseCurve):
        raise ValueError(
            "response estimator_config currently applies to ResponseCurve and point derivatives only"
        )
    return options


def _response_options_wire(options: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """A curve's response options; the simultaneous-band multipliers follow the study seed."""
    return {"multiplier_seed": seed, **options}


def _feedback_tuple(
    feedback: FeedbackMechanism | None,
) -> tuple[int, float, float, float, float, str | None, float] | None:
    """The declared feedback mechanism in the native argument order."""
    if feedback is None:
        return None
    return (
        feedback.delay,
        feedback.level,
        feedback.autoregression,
        feedback.input_gain,
        feedback.innovation_variance,
        feedback.initial_law,
        feedback.initial_state,
    )


def _lagged_edges(
    graph: TemporalDag | Sequence[tuple[str, int, str, int]] | None,
) -> list[tuple[str, int, str, int]]:
    if graph is None:
        raise CausalValueError("graph= lagged edges are required")
    if isinstance(graph, TemporalDag):
        return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph.edges()]
    edges = list(graph)
    for edge in edges:
        if not isinstance(edge, Sequence) or isinstance(edge, str) or len(edge) != 4:
            raise CausalValueError(
                "a temporal query needs lagged edges (source, source_lag, target, target_lag) "
                f"or a TemporalDag; got {edge!r}",
                reason_code="invalid_argument",
            )
    return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in edges]


def _encode_interventions(
    supplied: Any, *, route: str, soft_hint: str
) -> tuple[list[str], list[str], list[list[float]]]:
    from . import intervention as intervention_specs

    interventions = (
        list(supplied)
        if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
        else [supplied]
    )
    if not interventions:
        raise CausalValueError("InterventionResponse requires at least one intervention")
    treatments: list[str] = []
    kinds: list[str] = []
    parameters: list[list[float]] = []
    for spec in interventions:
        if isinstance(spec, intervention_specs.Set):
            kind, params = "set", [spec.value]
        elif isinstance(spec, intervention_specs.Shift):
            kind, params = "shift", [spec.delta]
        elif isinstance(spec, intervention_specs.Bernoulli):
            kind, params = "bernoulli", [spec.p]
        elif isinstance(spec, intervention_specs.Gaussian):
            kind, params = "gaussian", [spec.mean, spec.variance]
        elif isinstance(spec, intervention_specs.Categorical):
            kind, params = "categorical", list(spec.probabilities)
        elif isinstance(spec, (intervention_specs.Soft, intervention_specs.Sequence)):
            raise CausalUnsupportedError(
                f"{type(spec).__name__} interventions {soft_hint} and are not estimable by {route}"
            )
        else:
            raise TypeError(
                "InterventionResponse.intervention must be an antecedent.intervention "
                "specification or a sequence of specifications"
            )
        treatments.append(spec.variable)
        kinds.append(kind)
        parameters.append(params)
    return treatments, kinds, parameters


#: Ordered family-dispatch registry consulted by :meth:`_PrepareRoute.compile`.
#:
#: Each entry is ``(query_types, handler_name, phase)``. ``compile`` walks this
#: single list IN ORDER within each phase and routes the first ``isinstance``
#: match to the named zero-argument ``_PrepareRoute`` handler. The three phases
#: mark exactly where the compound, non-``isinstance`` arms of the chain must
#: run in between (see ``compile``): ``"pre"`` fires before the
#: ``discovery`` / graph-required / temporal / frame arms, ``"post_temporal"``
#: after the temporal return (Transport / Interference / Attribution), and
#: ``"post_admg"`` after the response-bootstrap guard and the ADMG-response
#: refusal. The order and the phase of every row are load-bearing: they
#: reproduce the precedence of the original chain, so a new straightforward
#: family is added by appending one row in its phase, never by editing the
#: compound arms that stay inline.
_COMPILE_ROUTES: tuple[tuple[tuple[type, ...], str, str], ...] = (
    ((TemporalQuery,), "_temporal_query", "pre"),
    ((PolicyValue, MultiActionPolicyValue), "_policy_value", "pre"),
    ((ConditionalDoseResponse,), "_continuous_dose_response", "pre"),
    ((RandomizedEffect,), "_randomized_effect", "pre"),
    ((ComplierEffect, TreatmentOnTreated), "_complier_effect", "pre"),
    ((SwitchbackEffect,), "_switchback_effect", "pre"),
    ((PanelDifferenceInDifferences, StaggeredAdoption, AugmentedPanelDiD), "_panel_did", "pre"),
    ((StaggeredRollout, CovariateAdjustedDiD), "_rollout_did", "pre"),
    ((SyntheticControl, SyntheticDifferenceInDifferences), "_synthetic_control", "pre"),
    ((SharpRegressionDiscontinuity,), "_sharp_rd", "pre"),
    ((FuzzyRegressionDiscontinuity, RegressionKink), "_local_polynomial_ratio", "pre"),
    ((LongitudinalRegime,), "_longitudinal_regime", "pre"),
    ((SequentialContrast,), "_sequential_contrast", "pre"),
    ((PeriodHistoryMatch,), "_history_match", "pre"),
    ((IvBounds,), "_iv_bounds", "pre"),
    ((Association,), "_association", "pre"),
    ((SurvivalOutcome, CompetingRisksOutcome), "_survival", "pre"),
    (TEMPORAL_DESIGN_QUERIES, "_temporal_design", "pre"),
    ((TransportQuery,), "_transport", "post_temporal"),
    ((InterferenceQuery,), "_interference", "post_temporal"),
    ((AnomalyAttribution, ChangeAttribution), "_attribution", "post_temporal"),
    (_RESPONSE_FAMILY, "_response", "post_admg"),
    ((AverageEffect,), "_average", "post_admg"),
    ((ConditionalEffect,), "_conditional", "post_admg"),
)


@dataclass
class _PrepareRoute:
    """One prepare request, routed to the native entry that compiles it."""

    names: list[str]
    columns: list[Any]
    frame: dict[str, Any] | None
    query: Any
    graph: Any
    discovery: Any
    inference: Frequentist | Bayesian
    identifier: str | None
    estimator: str | None
    estimator_config: Mapping[str, Any] | None
    refute: str | bool | None
    bootstrap: int | None
    threads: int | None
    seed: int
    class_prior: ClassPrior | None
    max_completions: int | None
    accepted: bool
    options: dict[str, Any]
    design_columns: dict[str, Any] | None = None
    #: Suite a route defers to the estimate's second click (see `_response`).
    deferred_suite: str | None = None

    # -- shared -----------------------------------------------------------
    def _common(self) -> dict[str, Any]:
        return {"seed": self.seed, "threads": self.threads, "options": self.options}

    def _refuse_estimator_config(self, route: str) -> None:
        if self.estimator_config:
            raise _not_applicable("estimator_config", route)

    def _refuse_ids(self, route: str, bayesian_id: str | None = None) -> None:
        selected = getattr(self.estimator, "value", self.estimator)
        if (
            bayesian_id is not None
            and selected == bayesian_id
            and isinstance(self.inference, Bayesian)
            and self.identifier is None
        ):
            return
        if self.identifier is not None or self.estimator is not None:
            raise CausalUnsupportedError(
                f"{route} selects its identifier and estimator itself; custom "
                "identifier= / estimator= are not supported",
                reason_code="option_not_applicable",
            )

    def _select_ids(self) -> None:
        """Hand the caller's identifier and estimator to the native compiler.

        The routes that call this accept a family of estimators (design-based and Bayesian
        arm contrasts, the temporal linear, link-contrast and lagged-instrument estimators);
        the compiler owns which combinations are valid and refuses the rest by code.
        """
        if self.identifier is not None:
            self.options["identifier"] = getattr(self.identifier, "value", self.identifier)
        if self.estimator is not None:
            self.options["estimator"] = getattr(self.estimator, "value", self.estimator)

    def _explicit_refute(self) -> bool:
        return self.refute is not None and self.refute not in (False, "none")

    def _route(self, phase: str) -> str | None:
        """Return the ``_COMPILE_ROUTES`` handler for ``query`` in ``phase``.

        Walks the single ordered registry and returns the first row whose
        phase matches and whose query types match ``self.query``; ``None`` when
        no simple family in that phase applies and the compound arms take over.
        """
        query = self.query
        for query_types, handler_name, entry_phase in _COMPILE_ROUTES:
            if entry_phase == phase and isinstance(query, query_types):
                return handler_name
        return None

    def compile(self) -> tuple[Any, Literal["average", "response_curve", "intervention_response"]]:
        query = self.query
        handler = self._route("pre")
        if handler is not None:
            return getattr(self, handler)()
        if self.discovery is not None:
            return self._graph_posterior()
        if self.graph is None:
            raise CausalValueError("PreparedAnalysis.prepare requires graph= or discovery=")
        temporal = isinstance(
            query,
            (
                PulseEffect,
                SustainedEffect,
                TemporalConditionalPulse,
                TemporalConditionalSustained,
                TemporalMediationEffect,
            ),
        ) or (
            isinstance(query, (ResponseCurve, InterventionResponse))
            and getattr(query, "is_temporal", False)
        )
        if self.frame is not None and not temporal:
            raise CausalUnsupportedError(
                f"{type(query).__name__} is a static query; panel, multi-environment and "
                "event data are licensed for temporal queries (Pulse / Sustained / "
                "TemporalMediation / temporal response)",
                reason_code="data_modality_not_licensed",
            )
        if temporal:
            return self._temporal()
        handler = self._route("post_temporal")
        if handler is not None:
            return getattr(self, handler)()
        # (RD kwargs already refused for non-AverageEffect queries at the top.)
        if isinstance(query, _RESPONSE_FAMILY):
            # A requested replicate count is refused before the structural
            # refusals: it is the caller's own request, not the cell's shape.
            self._refuse_response_bootstrap()
        handler = self._route("post_admg")
        if handler is not None:
            return getattr(self, handler)()
        if isinstance(query, InterventionalDistribution):
            self._refuse_estimator_config("InterventionalDistribution")
            self._refuse_ids("InterventionalDistribution (general.id + functional.distribution)")
            admg = isinstance(self.graph, Admg)
            native = _NativePreparedAnalysis.prepare_distribution(
                self.names,
                self.columns,
                [] if admg else _static_edges(self.graph),
                query.outcome,
                dict(query.interventions),
                graph=self.graph if admg else None,
                conditioning=list(query.conditioning) or None,
                accepted=self.accepted,
                **self._common(),
            )
            return native, "average"
        edges = _static_edges(self.graph)
        if isinstance(query, (MediationEffect, NestedCounterfactual, Counterfactual)):
            return self._static_kind(edges)
        if isinstance(query, PathSpecificEffect):
            self._refuse_estimator_config("PathSpecificEffect")
            self._refuse_ids("PathSpecificEffect (path_specific.natural + functional.effect)")
            native = _NativePreparedAnalysis.prepare_path_specific(
                self.names,
                self.columns,
                edges,
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                path_nodes=list(query.path_nodes) if query.path_nodes is not None else None,
                max_paths=query.max_paths,
                max_len=query.max_len,
                accepted=self.accepted,
                **self._common(),
            )
            return native, "average"
        raise CausalTypeError(f"unsupported query type: {type(query)!r}")

    def _temporal_query(self) -> tuple[Any, Any]:
        """Bind the original typed target to its existing native temporal family."""
        attachment = self.query
        query = attachment.query
        self._refuse_estimator_config("TemporalQuery (native unfolded temporal estimator)")
        derivative = isinstance(
            query,
            (
                AverageDerivative,
                PointDerivative,
                Elasticity,
                SemiElasticity,
                DirectionalDerivative,
                ResponseJacobian,
            ),
        )
        if derivative:
            from .observation import Complete

            if query.observation is not None and not isinstance(query.observation, Complete):
                raise CausalUnsupportedError(
                    "temporal derivatives require complete observations",
                    reason_code="option_not_applicable",
                )
            if query.observation_assumptions:
                raise CausalUnsupportedError(
                    "temporal derivatives require the observed population",
                    reason_code="option_not_applicable",
                )
            target = _derivative_target_spec(query)
        else:
            target = {
                "kind": query.kind,
                "treatments": [getattr(query, "treatment", "")],
                "outcomes": [query.outcome],
                "at": None,
                "direction": None,
                "order": 1,
                "scale": "identity",
                "weighting": "observed",
            }
        target |= {
            "control_level": getattr(query, "control_level", 0.0),
            "active_level": getattr(query, "active_level", 1.0),
            "mediators": [query.mediator] if isinstance(query, NestedCounterfactual) else [],
            "path_nodes": list(query.path_nodes)
            if getattr(query, "path_nodes", None) is not None
            else None,
            "max_paths": getattr(query, "max_paths", 64),
            "max_len": getattr(query, "max_len", 16),
            "interventions": list(getattr(query, "interventions", {}).items()),
            "conditioning": list(getattr(query, "conditioning", ())),
        }
        discovery = self.discovery
        if discovery is not None and not isinstance(discovery, (DbnPosterior, GraphPosterior)):
            raise CausalUnsupportedError(
                "TemporalQuery graph uncertainty requires a temporal graph posterior",
                reason_code="option_not_applicable",
            )
        native_graph = self.graph
        if native_graph is not None and not isinstance(
            native_graph, (TemporalDag, TemporalCpdag, TemporalPag, TemporalAdmg)
        ):
            native_graph = TemporalDag.from_lagged_edges(self.names, _lagged_edges(native_graph))
        dbn = (
            None
            if not isinstance(discovery, DbnPosterior)
            else {
                "max_lag": discovery.max_lag,
                "force_mcmc": discovery.force_mcmc,
                "n_chains": discovery.n_chains,
                "n_warmup": discovery.n_warmup,
                "n_draws": discovery.n_draws,
            }
        )
        spec = {
            "target": target,
            "temporal": {
                "horizons": list(attachment.horizons),
                "policy": attachment.policy,
                "treatment_lag": attachment.treatment_lag,
                "max_history_lag": attachment.max_history_lag,
                "window": attachment.window,
            },
            "discovery": dbn,
            "posterior": discovery if isinstance(discovery, GraphPosterior) else None,
            "response_options": None,
            **_class_prior_kwargs(self.class_prior),
            **_max_completions_kwargs(self.max_completions),
        }
        from .population import coerce_target_population

        options = dict(self.options)
        population = coerce_target_population(getattr(query, "target_population", None))
        if population is not None:
            options["target_population"] = population
        native = _NativePreparedAnalysis.prepare_temporal_query(
            self.names,
            self.columns,
            native_graph,
            spec,
            options=options,
            seed=self.seed,
            threads=self.threads,
            frame=self.frame,
            accepted=self.accepted,
        )
        return native, "response_curve" if derivative else "average"

    # -- graph posterior ------------------------------------------------------
    def _graph_posterior(self) -> tuple[Any, Any]:
        query, discovery = self.query, self.discovery
        self._refuse_ids("PreparedAnalysis graph-posterior discovery (per posterior atom)")
        if self.frame is not None and self.frame["kind"] != "events":
            raise CausalUnsupportedError(
                "graph-posterior cells are licensed on one table, one series, or one event "
                "stream; a panel or multi-environment mixture has no single atom weighting",
                reason_code="data_modality_not_licensed",
            )
        static = isinstance(discovery, (ExactDagPosterior, GraphPosterior))
        temporal = isinstance(discovery, (DbnPosterior, GraphPosterior))
        shared: dict[str, Any] = dict(self._common())
        if isinstance(discovery, GraphPosterior):
            shared["posterior"] = discovery
        dbn: dict[str, Any] = {"frame": self.frame} if temporal else {}
        dbn |= (
            {}
            if isinstance(discovery, GraphPosterior) or not isinstance(discovery, DbnPosterior)
            else {
                "max_lag": discovery.max_lag,
                "force_mcmc": discovery.force_mcmc,
                "n_chains": discovery.n_chains,
                "n_warmup": discovery.n_warmup,
                "mcmc_draws": discovery.n_draws,
            }
        )
        if static and isinstance(query, AverageEffect):
            if isinstance(query.control_level, str) or isinstance(query.active_level, str):
                raise CausalTypeError(
                    "graph-posterior effects require numeric treatment levels",
                    reason_code="invalid_argument",
                )
            native = _NativePreparedAnalysis.prepare_graph_posterior_ate(
                self.names,
                self.columns,
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                estimator_config=self.estimator_config,
                **shared,
            )
            return native, "average"
        if static and isinstance(query, ResponseCurve) and not query.is_temporal:
            response_options = _parse_response_estimator_config(query, self.estimator_config)
            native = _NativePreparedAnalysis.prepare_graph_posterior_response(
                self.names,
                self.columns,
                query.treatment,
                query.outcome,
                list(query.grid),
                response_options=_response_options_wire(response_options, self.seed)
                if response_options
                else None,
                **shared,
            )
            return native, "response_curve"
        self._refuse_estimator_config(
            "a graph-posterior mixture other than AverageEffect or a static ResponseCurve"
        )
        if static and isinstance(query, ConditionalEffect):
            from .query import coerce_outcome_functional

            native = _NativePreparedAnalysis.prepare_graph_posterior_conditional(
                self.names,
                self.columns,
                query.treatment,
                query.outcome,
                list(query.modifiers),
                control_level=query.control_level,
                active_level=query.active_level,
                outcome_functional=coerce_outcome_functional(
                    getattr(query, "outcome_functional", None)
                ),
                **shared,
            )
            return native, "average"
        if static and isinstance(query, InterventionResponse) and not query.is_temporal:
            treatments, kinds, parameters = _encode_interventions(
                query.intervention,
                route="a graph-posterior response",
                soft_hint="require a structural/temporal model",
            )
            if any(kind not in ("set", "shift") for kind in kinds):
                raise CausalUnsupportedError(
                    "graph-posterior InterventionResponse is licensed for one Set or Shift "
                    "intervention coordinate",
                    reason_code="option_not_applicable",
                )
            native = _NativePreparedAnalysis.prepare_graph_posterior_intervention_response(
                self.names, self.columns, query.outcome, treatments, kinds, parameters, **shared
            )
            return native, "intervention_response"
        if temporal and isinstance(query, (PulseEffect, SustainedEffect)):
            native = _NativePreparedAnalysis.prepare_dbn_posterior_temporal(
                self.names,
                self.columns,
                query.treatment,
                query.outcome,
                policy=query.kind,
                window=getattr(query, "window", None),
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                control_level=query.control_level,
                active_level=query.active_level,
                max_history_lag=query.max_history_lag,
                **dbn,
                **shared,
            )
            return native, "average"
        if temporal and isinstance(query, TemporalMediationEffect):
            native = _NativePreparedAnalysis.prepare_dbn_posterior_mediation(
                self.names,
                self.columns,
                query.treatment,
                query.mediator,
                query.outcome,
                contrast=query.contrast,
                control_level=query.control_level,
                active_level=query.active_level,
                horizons=list(query.horizons or (1,)),
                **dbn,
                **shared,
            )
            return native, "average"
        if (
            temporal
            and isinstance(query, (ResponseCurve, InterventionResponse))
            and query.is_temporal
        ):
            if isinstance(query, ResponseCurve) and self._explicit_refute():
                raise CausalUnsupportedError(
                    "not_applicable: graph-posterior ResponseCurve cheap/full do not denote; "
                    "InterventionResponse cheap/full mix Pulse-native atom reports.",
                    reason_code="refutation_not_applicable",
                )
            from .intervention import encode_temporal_steps

            if isinstance(query, InterventionResponse):
                supplied = query.intervention
                specs = (
                    list(supplied)
                    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
                    else [supplied]
                )
                treatments = []
                kinds = []
                parameters = []
                for spec in specs:
                    for variable, kind, params in encode_temporal_steps(spec):
                        treatments.append(variable)
                        kinds.append(kind)
                        parameters.append(params)
                grid = None
                response_kind: Literal["response_curve", "intervention_response"] = (
                    "intervention_response"
                )
                outcomes = [query.outcome]
            else:
                treatments, kinds, parameters = [query.treatment], None, None  # type: ignore[assignment]
                grid = list(query.grid)
                response_kind = "response_curve"
                outcomes = [query.outcome]
            native = _NativePreparedAnalysis.prepare_dbn_posterior_response(
                self.names,
                self.columns,
                query.kind,
                treatments,
                outcomes,
                grid=grid,
                intervention_kinds=kinds,
                intervention_parameters=parameters,
                horizons=list(query.horizons or ()),
                policy=query.policy,
                treatment_lag=query.treatment_lag,
                max_history_lag=query.max_history_lag,
                **dbn,
                **shared,
            )
            return native, response_kind
        if static and self.frame is None:
            mixed = self._graph_posterior_cross_world(shared)
            if mixed is not None:
                return mixed
        raise CausalUnsupportedError(
            "this query has no graph-posterior mixture. Graph-posterior cells are "
            "AverageEffect / ConditionalEffect / ResponseCurve / InterventionResponse, "
            "MediationEffect / PathSpecificEffect / NestedCounterfactual / "
            "InterventionalDistribution / AnomalyAttribution / ChangeAttribution / "
            "TransportQuery on DAG, CPDAG, PAG and ADMG atoms, and Pulse / Sustained / "
            "TemporalMediationEffect / temporal ResponseCurve / InterventionResponse on DBN and "
            "temporal-class atoms",
            reason_code="option_not_applicable",
        )

    def _graph_posterior_cross_world(self, shared: dict[str, Any]) -> tuple[Any, Any] | None:
        """Mediation, path-restricted, natural direct, distribution, attribution and transport.

        Every atom of the posterior is answered by the explicit executor of its own structure
        and the atoms are mixed by posterior probability; the native compiler owns which
        validation suites and inference modes apply.
        """
        query = self.query
        route = type(query).__name__
        if isinstance(query, TransportQuery):
            from .transport._impl import _response_args

            columns = query.trial_columns
            if columns is None:
                raise CausalValueError(
                    "TransportQuery on analyze reads its trial columns: pass trial=, "
                    "selection_probability= and treatment_probability="
                )
            response = _response_args(query.query)
            native = _NativePreparedAnalysis.prepare_graph_posterior_transport(
                self.names,
                self.columns,
                list(query.diagram.selections),
                query.diagram.source,
                query.diagram.target,
                list(query.source_experiments),
                response["kind"],
                response["treatments"],
                response["outcomes"],
                columns[0],
                columns[1],
                columns[2],
                grid=response["grid"],
                at=response["at"],
                direction=response["direction"],
                order=response["order"],
                scale=response["scale"],
                weighting=response["weighting"],
                **shared,
            )
            return native, "average"
        arguments: dict[str, Any]
        if isinstance(query, MediationEffect):
            arguments = {
                "kind": "mediation",
                "treatment": query.treatment,
                "outcome": query.outcome,
                "mediators": list(query.mediators),
                "contrast": query.contrast,
                "control_level": query.control_level,
                "active_level": query.active_level,
            }
        elif isinstance(query, NestedCounterfactual):
            arguments = {
                "kind": "nested_counterfactual",
                "treatment": query.treatment,
                "outcome": query.outcome,
                "mediators": [query.mediator],
                "control_level": query.control_level,
                "active_level": query.active_level,
            }
        elif isinstance(query, PathSpecificEffect):
            arguments = {
                "kind": "path_specific",
                "treatment": query.treatment,
                "outcome": query.outcome,
                "control_level": query.control_level,
                "active_level": query.active_level,
                "path_nodes": list(query.path_nodes) if query.path_nodes is not None else None,
                "max_paths": query.max_paths,
                "max_len": query.max_len,
            }
        elif isinstance(query, InterventionalDistribution):
            arguments = {
                "kind": "distribution",
                "outcome": query.outcome,
                "interventions": dict(query.interventions),
                "conditioning": list(query.conditioning) or None,
            }
        elif isinstance(query, AnomalyAttribution):
            reference = (
                (float(query.reference.center), float(query.reference.scale))
                if query.reference is not None
                else None
            )
            arguments = {
                "kind": "anomaly_attribution",
                "targets": list(query.targets),
                "max_units": int(query.max_units),
                "reference": reference,
            }
        elif isinstance(query, ChangeAttribution):
            arguments = {
                "kind": "change_attribution",
                "outcome": query.outcome,
                "baseline_start": int(query.baseline_start),
                "baseline_end": int(query.baseline_end),
                "comparison_start": int(query.comparison_start),
                "comparison_end": int(query.comparison_end),
            }
        else:
            return None
        self._refuse_estimator_config(route)
        self._refuse_ids(route)
        if self.bootstrap and isinstance(self.inference, Bayesian):
            raise CausalUnsupportedError(
                f"Bayesian {route} over a graph posterior publishes its Bayesian-bootstrap "
                "posterior; bootstrap= applies to Frequentist inference",
                reason_code="option_not_applicable",
            )
        native = _NativePreparedAnalysis.prepare_graph_posterior_kind(
            self.names, self.columns, arguments.pop("kind"), **arguments, **shared
        )
        return native, "average"

    # -- temporal -------------------------------------------------------------
    def _temporal(self) -> tuple[Any, Any]:
        query, graph = self.query, self.graph
        self._refuse_estimator_config(f"{type(query).__name__} (fixed temporal estimator)")
        # A temporal ADMG travels beside the equivalence classes: the compiler identifies a
        # pulse, sustained or conditional contrast on its unrolled window.
        class_graph = (
            graph if isinstance(graph, (TemporalCpdag, TemporalPag, TemporalAdmg)) else None
        )
        lagged = (
            []
            if class_graph is not None
            else _lagged_edges(cast("TemporalDag | Sequence[tuple[str, int, str, int]]", graph))
        )
        common: dict[str, Any] = {
            **self._common(),
            "accepted": self.accepted,
            "class_graph": class_graph,
            "frame": self.frame,
            **_class_prior_kwargs(self.class_prior),
            **_max_completions_kwargs(self.max_completions),
        }
        if isinstance(
            query,
            (PulseEffect, SustainedEffect, TemporalConditionalPulse, TemporalConditionalSustained),
        ):
            self._select_ids()
            native = _NativePreparedAnalysis.prepare_temporal_effect(
                self.names,
                self.columns,
                lagged,
                query.treatment,
                query.outcome,
                policy=query.kind,
                window=getattr(query, "window", None),
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                control_level=query.control_level,
                active_level=query.active_level,
                max_history_lag=query.max_history_lag,
                modifier=getattr(query, "modifier", None),
                **common,
            )
            return native, "average"
        if isinstance(query, TemporalMediationEffect):
            self._refuse_ids("TemporalMediationEffect (fixed temporal mediation estimator)")
            native = _NativePreparedAnalysis.prepare_temporal_mediation(
                self.names,
                self.columns,
                lagged,
                query.treatment,
                query.mediator,
                query.outcome,
                contrast=query.contrast,
                control_level=query.control_level,
                active_level=query.active_level,
                horizons=list(query.horizons or (1,)),
                **common,
            )
            return native, "average"
        # Temporal ResponseCurve / InterventionResponse.
        if self.identifier not in (None, "temporal.backdoor.unfolded"):
            raise CausalValueError(
                "temporal response requires identifier='temporal.backdoor.unfolded'; "
                f"got {self.identifier!r}"
            )
        expected = (
            "response.temporal.bayesian"
            if isinstance(self.inference, Bayesian)
            else "temporal.response.gcomp"
        )
        if self.estimator not in (None, expected):
            raise CausalValueError(
                f"temporal response requires estimator={expected!r}; got {self.estimator!r}"
            )
        # A temporal intervention level's cheap/full suite is the per-horizon
        # Pulse-witness refuter state (unlicensed); the native study runs it.
        if self._explicit_refute() and not isinstance(self.query, InterventionResponse):
            raise CausalUnsupportedError(
                "not_applicable: PreparedAnalysis temporal ResponseCurve cheap/full does not "
                "denote; cheap and full name the ATE-shaped scalar refuter suite and a "
                "function-valued curve has no such state. Use refute='none'."
            )
        if isinstance(self.inference, Bayesian) and self.bootstrap:
            raise CausalUnsupportedError(
                "Bayesian responses use posterior intervals; bootstrap is unsupported"
            )
        from .intervention import encode_temporal_steps
        from .observation import (
            Complete,
            _ensure_latent_schema_column,
            _temporal_observation_kwargs,
        )

        names, columns = self.names, self.columns
        if getattr(query, "observation", None) is not None and not isinstance(
            query.observation, Complete
        ):
            if self.frame is not None:
                raise CausalUnsupportedError(
                    "an observation mechanism adds a latent schema column to one series; "
                    "panel, multi-environment and event frames carry fixed partitions",
                    reason_code="data_modality_not_licensed",
                )
            names, columns = _ensure_latent_schema_column(names, columns, query.observation)
        observation_kwargs = _temporal_observation_kwargs(query)
        if isinstance(query, InterventionResponse):
            supplied = query.intervention
            specs = (
                list(supplied)
                if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
                else [supplied]
            )
            treatments: list[str] = []
            kinds: list[str] = []
            parameters: list[list[float]] = []
            for spec in specs:
                for variable, kind, params in encode_temporal_steps(spec):
                    treatments.append(variable)
                    kinds.append(kind)
                    parameters.append(params)
            grid = None
            response_kind: Literal["response_curve", "intervention_response"] = (
                "intervention_response"
            )
            outcomes = [query.outcome]
        else:
            treatments, kinds, parameters = [query.treatment], None, None  # type: ignore[assignment]
            grid = list(query.grid)
            response_kind = "response_curve"
            outcomes = [query.outcome]
        native = _NativePreparedAnalysis.prepare_temporal_response(
            names,
            columns,
            lagged,
            query.kind,
            treatments,
            outcomes,
            grid=grid,
            intervention_kinds=kinds,
            intervention_parameters=parameters,
            horizons=list(query.horizons or ()),
            policy=query.policy,
            treatment_lag=query.treatment_lag,
            max_history_lag=query.max_history_lag,
            feedback=_feedback_tuple(query.feedback),
            **common,
            **observation_kwargs,
        )
        return native, response_kind

    # -- static average / conditional ------------------------------------------
    def _lag_zero_average(self, functional: Any) -> tuple[Any, Any]:
        """An AverageEffect on one series under a temporal graph: the lag-0 contrast.

        It is compiled as the pulse at offset 0 read at horizon 1 and keeps the
        AverageEffect support cell, which executes and is unlicensed (``support_status``
        ``"refused"``); the result carries ``estimate.temporal_class.lag_zero_average_effect``.
        """
        query, graph = self.query, self.graph
        if functional is not None:
            raise CausalUnsupportedError(
                "a lag-0 AverageEffect on a temporal graph is the mean contrast. Next attempt: "
                "a temporal ResponseCurve for another functional",
                reason_code="route_not_supported",
            )
        if getattr(query, "target_population", None) is not None:
            raise CausalUnsupportedError(
                "a lag-0 AverageEffect on a temporal graph is the contrast over all observed "
                "periods; omit target_population",
                reason_code="route_not_supported",
            )
        try:
            control, active = float(query.control_level), float(query.active_level)
        except (TypeError, ValueError) as error:
            raise CausalValueError(
                "a lag-0 AverageEffect on a temporal graph takes numeric control and active levels"
            ) from error
        self._refuse_estimator_config("AverageEffect on a temporal graph (lag-0 pulse)")
        self._select_ids()
        class_graph = graph if isinstance(graph, (TemporalCpdag, TemporalPag)) else None
        lagged = [] if class_graph is not None else _lagged_edges(graph)
        native = _NativePreparedAnalysis.prepare_temporal_effect(
            self.names,
            self.columns,
            lagged,
            query.treatment,
            query.outcome,
            policy="pulse",
            treatment_lag=0,
            horizon_steps=1,
            control_level=control,
            active_level=active,
            accepted=self.accepted,
            class_graph=class_graph,
            as_average_effect=True,
            **_class_prior_kwargs(self.class_prior),
            **_max_completions_kwargs(self.max_completions),
            **self._common(),
        )
        return native, "average"

    def _average(self) -> tuple[Any, Any]:
        from .query import coerce_outcome_functional

        query, graph = self.query, self.graph
        if str(self.estimator) in ("rd.sharp", "Estimator.SHARP_RD"):
            raise CausalUnsupportedError(
                "Sharp regression discontinuity is a query, not an estimator: "
                "analyze(data, query=antecedent.quasi.SharpRegressionDiscontinuity("
                "outcome, treatment, running, cutoff=, bandwidth=)); it needs no graph.",
                reason_code="option_not_applicable",
            )
        functional = coerce_outcome_functional(getattr(query, "outcome_functional", None))
        if isinstance(graph, (TemporalDag, TemporalCpdag, TemporalPag)):
            return self._lag_zero_average(functional)
        if isinstance(graph, TieredBackground):
            if self.identifier is not None:
                raise CausalUnsupportedError(
                    "TieredBackground selects its own identifier; omit identifier",
                    reason_code="option_not_applicable",
                )
            native = _NativePreparedAnalysis.prepare_tiered(
                self.names,
                self.columns,
                [list(tier) for tier in graph.tiers],
                str(graph.within_tier),
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                estimator=self.estimator,
                estimator_config=self.estimator_config,
                outcome_functional=functional,
                **self._common(),
            )
            return native, "average"
        if isinstance(graph, (Pag, Cpdag, Admg)):
            native = _NativePreparedAnalysis.prepare_class_ate(
                self.names,
                self.columns,
                graph,
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                identifier=self.identifier,
                estimator=self.estimator,
                estimator_config=self.estimator_config,
                outcome_functional=functional,
                accepted=self.accepted,
                **self._common(),
            )
            return native, "average"
        running_variable = cutoff = bandwidth = None
        native = _NativePreparedAnalysis.prepare(
            self.names,
            self.columns,
            _static_edges(graph),
            query.treatment,
            query.outcome,
            control_level=query.control_level,
            active_level=query.active_level,
            identifier=self.identifier,
            estimator=self.estimator,
            estimator_config=self.estimator_config,
            outcome_functional=functional,
            running_variable=running_variable,
            cutoff=cutoff,
            bandwidth=bandwidth,
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    def _continuous_dose_response(self) -> tuple[Any, Literal["average"]]:
        query = cast(ConditionalDoseResponse, self.query)
        if self.graph is not None or self.discovery is not None:
            raise _not_applicable("graph/discovery", "ConditionalDoseResponse")
        self._refuse_ids("ConditionalDoseResponse")
        self._refuse_estimator_config("ConditionalDoseResponse")
        if self._explicit_refute() or self.bootstrap:
            raise _not_applicable("refute/bootstrap", "ConditionalDoseResponse")
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise _not_applicable("inference", "ConditionalDoseResponse")
        design = self.design_columns or {}
        native = _NativePreparedAnalysis.prepare_continuous_dose_response(
            self.names,
            self.columns,
            query.outcome,
            query.dose,
            query.dose_density,
            list(design["baseline_groups"]),
            list(query.target_doses),
            query.bandwidth,
            query.density_provenance,
            min_local_support=query.min_local_support,
            policy_doses=(
                None
                if query.policy_doses is None
                else list(query.policy_doses.items())
                if isinstance(query.policy_doses, Mapping)
                else list(query.policy_doses)
            ),
            reference_doses=(
                None
                if query.reference_doses is None
                else list(query.reference_doses.items())
                if isinstance(query.reference_doses, Mapping)
                else list(query.reference_doses)
            ),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    def _policy_value(self) -> tuple[Any, Literal["average"]]:
        query = cast(PolicyValue | MultiActionPolicyValue, self.query)
        if self.graph is not None or self.discovery is not None:
            raise CausalUnsupportedError(
                "PolicyValue carries its randomized design and does not accept graph= or discovery=",
                reason_code="option_not_applicable",
            )
        self._refuse_ids("PolicyValue", "bayesian.binary_policy_value")
        self._refuse_estimator_config("PolicyValue")
        if self._explicit_refute() or self.bootstrap:
            raise CausalUnsupportedError(
                "PolicyValue does not accept refutation or bootstrap options",
                reason_code="option_not_applicable",
            )
        from .inference import Frequentist

        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise CausalUnsupportedError(
                "PolicyValue supports row-score frequentist uncertainty or a Bayesian posterior",
                reason_code="option_not_applicable",
            )
        if isinstance(query, MultiActionPolicyValue):
            labels = tuple(query.policy.action_labels)
            n = len(query.assignment)
            evaluate_multi_action_policy(
                dict(zip(self.names, self.columns, strict=True)),
                outcome=query.outcome,
                assignment=query.assignment,
                propensities=query.propensities,
                policy=query.policy,
                reference=query.reference,
                available=query.available,
            )
            multi_reference = query.reference or MultiActionPolicy(
                labels,
                [labels[0]] * n,
                costs=query.policy.costs,
            )
            multi_available = query.available or [[True] * len(labels) for _ in range(n)]
            native = _NativePreparedAnalysis.prepare_multi_action_policy_value(
                self.names,
                self.columns,
                query.outcome,
                list(labels),
                [labels.index(action) for action in query.assignment],
                [float(p) for row in query.propensities for p in row],
                [labels.index(action) for action in query.policy.recommendations],
                [labels.index(action) for action in multi_reference.recommendations],
                list(query.policy.costs or [0.0] * len(labels)),
                list(multi_reference.costs or [0.0] * len(labels)),
                [bool(value) for row in multi_available for value in row],
                list(query.policy.capacities or [n] * len(labels)),
                list(multi_reference.capacities or [n] * len(labels)),
                query.policy.budget,
                multi_reference.budget,
                list(query.evaluation_subject_ids),
                cate_groups=list(query.baseline_groups or ()),
                accepted=self.accepted,
                **self._common(),
            )
            return native, "average"
        # The direct randomized evaluator owns the constraint checks. Apply
        # that same contract before freezing the retained doubly robust study.
        evaluate_policy(
            dict(zip(self.names, self.columns, strict=True)),
            outcome=query.outcome,
            assignment=resolved(query.assignment),
            propensity=resolved_scalar(query.propensity),
            policy=query.policy,
            reference=query.reference,
            available=resolved_opt(query.available),
        )
        propensity = (
            [float(query.propensity)]
            if isinstance(query.propensity, (int, float))
            else list(resolved(query.propensity))
        )
        costs = (
            [float(query.policy.costs)]
            if isinstance(query.policy.costs, (int, float))
            else list(query.policy.costs)
        )
        reference = query.reference or BinaryPolicy([False] * len(query.policy.actions))
        reference_costs = (
            [float(reference.costs)]
            if isinstance(reference.costs, (int, float))
            else list(reference.costs)
        )
        uplift_bin_ids: list[int] = []
        if query.uplift_scores is not None:
            uplift_bin_ids = [0] * len(query.policy.actions)
            ranked_rows = np.argsort(
                -np.asarray(query.uplift_scores, dtype=np.float64), kind="stable"
            )
            for rank, row in enumerate(ranked_rows):
                uplift_bin_ids[int(row)] = min(
                    query.uplift_bin_count - 1,
                    rank * query.uplift_bin_count // len(query.policy.actions),
                )
        native = _NativePreparedAnalysis.prepare_policy_value(
            self.names,
            self.columns,
            query.outcome,
            list(resolved(query.assignment)),
            propensity,
            list(resolved(query.policy.actions)),
            list(resolved(reference.actions)),
            list(resolved_opt(query.mu0) or ()),
            list(resolved_opt(query.mu1) or ()),
            costs,
            reference_costs,
            list(resolved(query.evaluation_subject_ids)),
            query._ownership == "held_out_disjoint_subject_ids",
            query._ownership == "caller_declared_cross_fitted_excluded_fold_ids",
            uplift_bins=uplift_bin_ids,
            uplift_bin_count=query.uplift_bin_count,
            uplift_training_subject_ids=list(query.uplift_training_subject_ids or ()),
            regret_candidates=[
                list(resolved(candidate.actions)) for candidate in (query.regret_candidates or ())
            ],
            regret_selected_index=(
                next(
                    (
                        index
                        for index, candidate in enumerate(query.regret_candidates)
                        if candidate.actions == query.policy.actions
                    ),
                    None,
                )
                if query.regret_candidates is not None
                else None
            ),
            regret_training_subject_ids=list(query.regret_training_subject_ids or ()),
            global_constraints_present=(
                query.policy.capacity is not None
                or query.policy.budget is not None
                or reference.capacity is not None
                or reference.budget is not None
            ),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    def _switchback_effect(self) -> tuple[Any, Literal["average"]]:
        query = cast(SwitchbackEffect, self.query)
        if self.graph is not None or self.discovery is not None:
            raise CausalUnsupportedError(
                "SwitchbackEffect carries its randomization design and does not accept graph= or discovery=",
                reason_code="option_not_applicable",
            )
        self._refuse_ids("SwitchbackEffect")
        self._refuse_estimator_config("SwitchbackEffect")
        if self._explicit_refute():
            raise CausalUnsupportedError(
                "SwitchbackEffect has no refutation route", reason_code="option_not_applicable"
            )
        design = query.design
        n = len(design.realized_assignment)
        native = _NativePreparedAnalysis.prepare_randomized_effect(
            self.names,
            self.columns,
            query.outcome,
            list(resolved(design.realized_assignment)),
            list(resolved(design.assignment_probabilities)),
            list(design.sequence_ids),
            [f"switchback-row-{i}" for i in range(n)],
            design.treatment_arms,
            "switchback",
            periods=list(design.period_ids),
            accepted=False,
            **self._common(),
        )
        return native, "average"

    def _complier_effect(self) -> tuple[Any, Literal["average"]]:
        query = cast(ComplierEffect | TreatmentOnTreated, self.query)
        if self.graph is not None or self.discovery is not None:
            raise CausalUnsupportedError(
                "ComplierEffect carries its randomization design and does not accept graph= or discovery=",
                reason_code="option_not_applicable",
            )
        self._refuse_ids("ComplierEffect")
        self._refuse_estimator_config("ComplierEffect")
        if self._explicit_refute():
            raise CausalUnsupportedError(
                "ComplierEffect has no refutation route", reason_code="option_not_applicable"
            )
        assignment = cast(BernoulliAssignment, query.design.assignment)
        n = len(query.design.realized_assignment)
        probabilities = assignment.probabilities
        if isinstance(probabilities, (float, int)):
            probability_rows = [float(probabilities)] * n
        else:
            probability_rows = list(probabilities)
            if len(probability_rows) == 1:
                probability_rows *= n
        native = _NativePreparedAnalysis.prepare_randomized_effect(
            self.names,
            self.columns,
            query.outcome,
            list(resolved(query.design.realized_assignment)),
            probability_rows,
            list(query.design.assignment_units),
            list(query.design.outcome_units),
            query.design.treatment_arms,
            "bernoulli",
            received_treatment=list(resolved(query.received_treatment)),
            treatment_on_treated=isinstance(query, TreatmentOnTreated),
            accepted=False,
            **self._common(),
        )
        return native, "average"

    def _randomized_effect(self) -> tuple[Any, Literal["average"]]:
        query = cast(RandomizedEffect, self.query)
        if self.graph is not None or self.discovery is not None:
            raise CausalUnsupportedError(
                "RandomizedEffect carries its randomization design and does not accept graph= or discovery=",
                reason_code="option_not_applicable",
            )
        if isinstance(query.design, MultiArmExperimentDesign):
            self._select_ids()
            self._refuse_estimator_config("RandomizedEffect")
            if self._explicit_refute():
                raise CausalUnsupportedError(
                    "RandomizedEffect has no refutation route", reason_code="option_not_applicable"
                )
            design = query.design
            indices = [design.action_labels.index(action) for action in design.realized_assignment]
            native = _NativePreparedAnalysis.prepare_randomized_effect(
                self.names,
                self.columns,
                query.outcome,
                [index != 0 for index in indices],
                [
                    float(design.assignment_probabilities[i][index])
                    for i, index in enumerate(indices)
                ],
                list(design.assignment_units),
                list(design.outcome_units),
                (design.action_labels[0], design.action_labels[1]),
                "multi_arm",
                multi_arm_labels=list(design.action_labels),
                multi_arm_assignment=indices,
                multi_arm_probabilities=[list(row) for row in design.assignment_probabilities],
                accepted=False,
                **self._common(),
            )
            return native, "average"
        assignment = query.design.assignment
        if not isinstance(
            assignment,
            (
                BernoulliAssignment,
                CompleteRandomization,
                StratifiedRandomization,
                ClusterRandomization,
                FactorialRandomization,
            ),
        ):
            raise CausalUnsupportedError(
                "unsupported randomized assignment design", reason_code="route_not_supported"
            )
        self._select_ids()
        self._refuse_estimator_config("RandomizedEffect")
        if self._explicit_refute():
            raise CausalUnsupportedError(
                "RandomizedEffect has no refutation route", reason_code="option_not_applicable"
            )
        n = len(query.design.realized_assignment)
        blocks: list[str] = []
        treated_per_row: list[int] = []
        treated_units: int | None = None
        treated_clusters: int | None = None
        if isinstance(assignment, BernoulliAssignment):
            design_kind = "bernoulli"
            probabilities = assignment.probabilities
            if isinstance(probabilities, (float, int)):
                probabilities = [float(probabilities)] * n
            else:
                probabilities = list(probabilities)
                if len(probabilities) == 1:
                    probabilities *= n
        elif isinstance(assignment, CompleteRandomization):
            design_kind = "complete"
            treated_units = assignment.treated
            probabilities = [treated_units / n] * n
            if query.design.blocks is not None:
                raise CausalUnsupportedError(
                    "complete design does not use block metadata; choose StratifiedRandomization",
                    reason_code="route_not_supported",
                )
        elif isinstance(assignment, ClusterRandomization):
            design_kind = "cluster"
            treated_clusters = assignment.treated_clusters
            cluster_count = len(set(query.design.assignment_units))
            probabilities = [treated_clusters / cluster_count] * n
            if query.design.blocks is not None:
                raise CausalUnsupportedError(
                    "cluster randomization does not combine with block metadata on this route",
                    reason_code="route_not_supported",
                )
        elif isinstance(assignment, FactorialRandomization):
            design_kind = "factorial_2x2"
            probabilities = [(assignment.cell_counts[1] + assignment.cell_counts[3]) / n] * n
        else:
            design_kind = "stratified"
            assert isinstance(assignment, StratifiedRandomization)
            blocks = list(query.design.blocks or ())
            treated_per_row = [assignment.treated_per_block[block] for block in blocks]
            block_sizes = {block: blocks.count(block) for block in set(blocks)}
            probabilities = [
                assignment.treated_per_block[block] / block_sizes[block] for block in blocks
            ]
        native = _NativePreparedAnalysis.prepare_randomized_effect(
            self.names,
            self.columns,
            query.outcome,
            list(resolved(query.design.realized_assignment)),
            [float(value) for value in probabilities],
            list(query.design.assignment_units),
            list(query.design.outcome_units),
            query.design.treatment_arms,
            design_kind,
            treated_units,
            blocks,
            treated_per_row,
            treated_clusters=treated_clusters,
            fixed_cuped=(query.cuped.covariate, query.cuped.coefficient) if query.cuped else None,
            ancova_covariates=list(query.ancova_covariates),
            exact_randomization_test=query.exact_randomization_test,
            second_factor_assignment=list(resolved(assignment.second_factor_assignment))
            if isinstance(assignment, FactorialRandomization)
            else None,
            factorial_cell_counts=assignment.cell_counts
            if isinstance(assignment, FactorialRandomization)
            else None,
            second_factor_arms=assignment.second_factor_arms
            if isinstance(assignment, FactorialRandomization)
            else None,
            accepted=False,
            **self._common(),
        )
        return native, "average"

    def _panel_did(self) -> tuple[Any, Literal["average"]]:
        query = cast(
            PanelDifferenceInDifferences | StaggeredAdoption | AugmentedPanelDiD, self.query
        )
        if self.graph is not None or self.discovery is not None:
            raise CausalUnsupportedError(
                "PanelDifferenceInDifferences carries its own design and does not accept graph= or discovery=",
                reason_code="option_not_applicable",
            )
        self._refuse_ids("PanelDifferenceInDifferences", "bayesian.panel_did")
        self._refuse_estimator_config("PanelDifferenceInDifferences")
        if self._explicit_refute() or self.bootstrap:
            raise CausalUnsupportedError(
                "PanelDifferenceInDifferences has no refutation or bootstrap route",
                reason_code="option_not_applicable",
            )
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise CausalUnsupportedError(
                "PanelDifferenceInDifferences supports a cluster standard error or a Bayesian posterior",
                reason_code="option_not_applicable",
            )
        names, columns = self.names, self.columns
        design = self.design_columns
        if design is None:
            raise CausalValueError("panel DiD design columns were not bound at prepare")
        if isinstance(query, AugmentedPanelDiD):
            native = _NativePreparedAnalysis.prepare_augmented_panel_did(
                names,
                columns,
                query.outcome_pre,
                query.outcome_post,
                query.propensity,
                query.untreated_change_prediction,
                list(design["treated"]),
                list(design["subjects"]),
                list(design["clusters"]),
                query.predictions_cross_fitted,
                resampling=query.resampling,
                replicates=query.replicates,
                accepted=False,
                **self._common(),
            )
        elif isinstance(query, StaggeredAdoption):
            subjects = list(design["subjects"])
            clusters = list(design["clusters"])
            periods = list(design["periods"])
            cohorts = list(design["cohorts"])
            if query.event_study:
                native = _NativePreparedAnalysis.prepare_staggered_event_study(
                    names,
                    columns,
                    query.outcome,
                    subjects,
                    clusters,
                    periods,
                    cohorts,
                    accepted=False,
                    **self._common(),
                )
            else:
                native = _NativePreparedAnalysis.prepare_staggered_group_time(
                    names,
                    columns,
                    query.outcome,
                    subjects,
                    clusters,
                    periods,
                    cohorts,
                    design["target_cohort"],
                    design["target_period"],
                    accepted=False,
                    **self._common(),
                )
        else:
            native = _NativePreparedAnalysis.prepare_panel_did(
                names,
                columns,
                query.outcome,
                list(design["treated"]),
                list(design["post"]),
                list(design["subjects"]),
                list(design["clusters"]),
                repeated_cross_section=bool(design["repeated_cross_section"]),
                accepted=False,
                **self._common(),
            )
        return native, "average"

    def _synthetic_control(self) -> tuple[Any, Literal["average"]]:
        query = cast(SyntheticControl | SyntheticDifferenceInDifferences, self.query)
        if self.graph is not None or self.discovery is not None:
            raise _not_applicable("graph/discovery", "SyntheticControl")
        self._refuse_ids("SyntheticControl", "bayesian.synthetic_control")
        self._refuse_estimator_config("SyntheticControl")
        if self._explicit_refute() or self.bootstrap:
            raise _not_applicable("refute/bootstrap", "SyntheticControl")
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise _not_applicable("inference", "SyntheticControl")
        design = self.design_columns
        if design is None:
            raise CausalValueError("synthetic-control design columns were not bound at prepare")
        native = _NativePreparedAnalysis.prepare_synthetic_control(
            self.names,
            self.columns,
            query.outcome,
            list(design["units"]),
            list(design["periods"]),
            query.treated_unit,
            query.intervention_period,
            difference_in_differences=isinstance(query, SyntheticDifferenceInDifferences),
            uniform_unit_randomization=query.uniform_unit_randomization,
            sharp_null_effect=query.sharp_null_effect,
            augmentation_ridge=query.augmentation_ridge
            if isinstance(query, SyntheticControl)
            else None,
            conformal=query.conformal if isinstance(query, SyntheticControl) else False,
            adoptions=None if query.adoptions is None else list(query.adoptions),
            placebo_replicates=query.placebo_replicates,
            observed_window_support=isinstance(query, SyntheticControl)
            and query.observed_window_support,
            accepted=False,
            **self._common(),
        )
        return native, "average"

    def _temporal_design(self) -> tuple[Any, Literal["average"]]:
        """A sequentially randomized unit-period design, or regime survival.

        The recorded assignment law is the design; the estimator follows the inference
        (the excursion or regime-survival estimator, or its Bayesian counterpart).
        """
        query = self.query
        route = type(query).__name__
        if self.graph is not None or self.discovery is not None:
            raise _not_applicable("graph/discovery", route)
        self._refuse_ids(route)
        self._refuse_estimator_config(route)
        if self._explicit_refute():
            raise CausalUnsupportedError(
                f"{route} has no refutation suite; its uncertainty is the design's own",
                reason_code="option_not_applicable",
            )
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise _not_applicable("inference", route)
        design = self.design_columns
        if design is None:
            raise CausalValueError("temporal design columns were not bound at prepare")
        declared = self.options.get("inference") or {}
        if isinstance(self.inference, Bayesian):
            if declared.get("prior_artifact") is not None or declared.get("composed_prior"):
                raise _not_applicable("prior_from / composed prior", route)
            # The design's posterior is its own; the conjugate declaration carries the draw
            # count and prior scale without selecting a regression backend.
            inference = {
                "mode": "conjugate",
                "n_draws": declared.get("n_draws"),
                "prior_scale": float(declared.get("prior_scale", 10.0)),
            }
        else:
            inference = {"mode": "frequentist", "prior_scale": 10.0}
        native = _NativePreparedAnalysis.prepare_temporal_design(
            self.names,
            self.columns,
            query.outcome,
            list(design["units"]),
            list(design["periods"]),
            design["spec"],
            **temporal_design_keywords(query),
            seed=self.seed,
            threads=self.threads,
            options={**self.options, "inference": inference},
        )
        return native, "average"

    def _iv_bounds(self) -> tuple[Any, Literal["average"]]:
        """Bounds on the average effect under a binary instrument, as an identified set.

        The estimator follows the inference (the Imbens-Manski row-bootstrap interval, or the
        Bayesian-bootstrap posterior of the endpoints); ``bootstrap=`` adds the frequentist
        interval.
        """
        query = cast(IvBounds, self.query)
        route = "IvBounds"
        if self.graph is not None or self.discovery is not None:
            raise _not_applicable("graph/discovery", route)
        self._refuse_ids(route)
        self._refuse_estimator_config(route)
        if self._explicit_refute():
            raise CausalUnsupportedError(
                f"{route} has no refutation suite; it reports an identified set",
                reason_code="option_not_applicable",
            )
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise _not_applicable("inference", route)
        declared = self.options.get("inference") or {}
        if isinstance(self.inference, Bayesian):
            if declared.get("prior_artifact") is not None or declared.get("composed_prior"):
                raise _not_applicable("prior_from / composed prior", route)
            inference = {
                "mode": "conjugate",
                "n_draws": declared.get("n_draws"),
                "prior_scale": float(declared.get("prior_scale", 10.0)),
            }
        else:
            inference = {"mode": "frequentist", "prior_scale": 10.0}
        native = _NativePreparedAnalysis.prepare_iv_bounds(
            self.names,
            self.columns,
            query.outcome,
            query.treatment,
            query.instrument,
            query.weight,
            seed=self.seed,
            threads=self.threads,
            options={**self.options, "inference": inference},
        )
        return native, "average"

    def _association(self) -> tuple[Any, Literal["average"]]:
        """A noncausal association of the observed data, picked by the declared selection rule.

        The inference selects the uncertainty: analytic or resampling standard errors, or the
        posterior (conjugate Normal-Inverse-Gamma for a Gaussian outcome on unweighted rows,
        the Bayesian bootstrap otherwise); ``bootstrap=`` adds the frequentist resampling spread.
        """
        query = cast(Association, self.query)
        route = "Association"
        if self.graph is not None or self.discovery is not None:
            raise _not_applicable("graph/discovery", route)
        self._refuse_ids(route)
        self._refuse_estimator_config(route)
        if self._explicit_refute():
            raise CausalUnsupportedError(
                f"{route} has no refutation suite; it describes the observed law",
                reason_code="option_not_applicable",
            )
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise _not_applicable("inference", route)
        declared = self.options.get("inference") or {}
        if isinstance(self.inference, Bayesian):
            if declared.get("prior_artifact") is not None or declared.get("composed_prior"):
                raise _not_applicable("prior_from / composed prior", route)
            inference = {
                "mode": "conjugate",
                "n_draws": declared.get("n_draws"),
                "prior_scale": float(declared.get("prior_scale", 10.0)),
            }
        else:
            inference = {"mode": "frequentist", "prior_scale": 10.0}
        native = _NativePreparedAnalysis.prepare_association(
            self.names,
            self.columns,
            query.outcome,
            list(query.predictors),
            list(query.conditioning),
            query.functional,
            folds=query.folds if query.functional == "predictive" else None,
            outcome_scale=None if query.outcome_scale == "schema" else query.outcome_scale,
            weight=query.weight,
            refused_after=query.refused_after,
            series=query.temporal,
            seed=self.seed,
            threads=self.threads,
            options={**self.options, "inference": inference},
        )
        return native, "average"

    def _rollout_did(self) -> tuple[Any, Literal["average"]]:
        query = cast(StaggeredRollout | CovariateAdjustedDiD, self.query)
        route = type(query).__name__
        if self.graph is not None or self.discovery is not None:
            raise _not_applicable("graph/discovery", route)
        self._refuse_ids(route, "bayesian.panel_did")
        self._refuse_estimator_config(route)
        if self._explicit_refute() or self.bootstrap:
            raise CausalUnsupportedError(
                f"{route} resamples whole clusters through resampling=; it has no refutation "
                "or bootstrap= route",
                reason_code="option_not_applicable",
            )
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise _not_applicable("inference", route)
        design = self.design_columns
        if design is None:
            raise CausalValueError("rollout DiD design columns were not bound at prepare")
        if isinstance(query, StaggeredRollout):
            native = _NativePreparedAnalysis.prepare_rollout_did(
                self.names,
                self.columns,
                query.outcome,
                list(design["subjects"]),
                list(design["clusters"]),
                list(design["periods"]),
                list(design["cohorts"]),
                estimator=query.estimator,
                comparison=query.comparison,
                nuisance=query.nuisance,
                base_period=query.base_period,
                aggregation=query.aggregation,
                anticipation=int(query.anticipation),
                covariates=list(query.covariates),
                resampling=query.resampling,
                replicates=query.replicates,
                observation=query.observation,
                attrition_shift_bound=query.attrition_shift_bound,
                accepted=False,
                **self._common(),
            )
        else:
            native = _NativePreparedAnalysis.prepare_covariate_adjusted_did(
                self.names,
                self.columns,
                query.outcome,
                list(design["treated"]),
                list(design["post"]),
                list(design["subjects"]),
                list(design["clusters"]),
                list(query.covariates),
                nuisance=query.nuisance,
                resampling=query.resampling,
                replicates=query.replicates,
                accepted=False,
                **self._common(),
            )
        return native, "average"

    def _sharp_rd(self) -> tuple[Any, Literal["average"]]:
        from .query import coerce_outcome_functional

        query = cast(SharpRegressionDiscontinuity, self.query)
        if self.graph is not None or self.discovery is not None:
            raise _not_applicable("graph/discovery", "SharpRegressionDiscontinuity")
        self._refuse_ids("SharpRegressionDiscontinuity")
        self._refuse_estimator_config("SharpRegressionDiscontinuity")
        if self.inference is not None and not isinstance(self.inference, Frequentist):
            raise _not_applicable("inference", "SharpRegressionDiscontinuity")
        # The sharp-RD design is fixed: the running variable is the treatment's only
        # cause, and both act on the outcome. Synthesize it so the caller supplies
        # neither a graph nor an estimator.
        edges = [
            (query.running, query.treatment),
            (query.treatment, query.outcome),
            (query.running, query.outcome),
        ]
        native = _NativePreparedAnalysis.prepare(
            self.names,
            self.columns,
            _static_edges(edges),
            query.treatment,
            query.outcome,
            identifier="rd.sharp",
            estimator="rd.sharp",
            estimator_config={"se_kind": query.se} if query.se is not None else None,
            outcome_functional=coerce_outcome_functional(None),
            running_variable=query.running,
            cutoff=query.cutoff,
            bandwidth=query.bandwidth,
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    def _local_polynomial_ratio(self) -> tuple[Any, Literal["average"]]:
        query = cast(FuzzyRegressionDiscontinuity | RegressionKink, self.query)
        if self.graph is not None or self.discovery is not None:
            raise _not_applicable("graph/discovery", "local polynomial ratio")
        self._refuse_ids("local polynomial ratio", "bayesian.local_ratio")
        self._refuse_estimator_config("local polynomial ratio")
        if self._explicit_refute() or self.bootstrap:
            raise _not_applicable("refute/bootstrap", "local polynomial ratio")
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise _not_applicable("inference", "local polynomial ratio")
        native = _NativePreparedAnalysis.prepare_local_polynomial_ratio(
            self.names,
            self.columns,
            query.outcome,
            query.treatment,
            query.running,
            query.cutoff,
            query.bandwidth,
            kink=isinstance(query, RegressionKink),
            bandwidth_rule=query.bandwidth_rule,
            temporal_replicates=query.temporal_replicates,
            density_test=query.density_test,
            balance_covariates=list(query.balance_covariates),
            accepted=False,
            **self._common(),
        )
        return native, "average"

    def _survival(self) -> tuple[Any, Literal["average"]]:
        from .observation import IndependentGiven

        query = cast(SurvivalOutcome | CompetingRisksOutcome, self.query)
        if self.graph is not None or self.discovery is not None:
            raise CausalUnsupportedError(
                "randomized survival carries its own design and does not accept graph= or discovery=",
                reason_code="option_not_applicable",
            )
        self._refuse_ids("randomized survival")
        self._refuse_estimator_config("randomized survival")
        if self._explicit_refute():
            raise CausalUnsupportedError(
                "randomized survival has no refutation route",
                reason_code="option_not_applicable",
            )
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise CausalUnsupportedError(
                "randomized survival supports frequentist subject-bootstrap inference or a "
                "Bayesian piecewise-exponential posterior (Bayesian(design_model=PiecewiseHazards))",
                reason_code="option_not_applicable",
            )
        if not query.randomized:
            raise CausalUnsupportedError(
                "randomized survival requires randomized=True for individual assignment",
                reason_code="route_not_supported",
            )
        assumption = query.observation_assumption
        if not isinstance(assumption, IndependentGiven) or (
            query.known_censoring is None and tuple(assumption.variables)
        ):
            raise CausalUnsupportedError(
                "survival requires IndependentGiven; conditional censoring also requires known censoring survival",
                reason_code="route_not_supported",
            )
        known = query.known_censoring
        if known is not None:
            missing = [
                name for name in (*known.columns, *assumption.variables) if name not in self.names
            ]
            if missing:
                raise CausalUnsupportedError(
                    f"known censoring or conditioning columns are missing: {missing}",
                    reason_code="route_not_supported",
                )
        event = (
            query.event_cause if isinstance(query, CompetingRisksOutcome) else query.event_observed
        )
        target_cause = query.target_cause if isinstance(query, CompetingRisksOutcome) else None
        native = _NativePreparedAnalysis.prepare_survival(
            self.names,
            self.columns,
            query.duration,
            event,
            query.treatment,
            float(query.tau),
            target_cause,
            query.delayed_entry,
            independent_given=list(assumption.variables),
            censoring_times=list(known.times) if known is not None else [],
            censoring_columns=list(known.columns) if known is not None else [],
            censoring_probability_floor=known.minimum_probability if known is not None else None,
            accepted=False,
            **self._common(),
        )
        return native, "average"

    def _design_only(self, route: str, allowed_ids: tuple[str, ...]) -> None:
        """Refuse the graph, refutation and estimator options a design-carrying query ignores.

        ``allowed_ids`` are the estimator ids the query answers; selecting one is the same route.
        """
        if self.graph is not None or self.discovery is not None:
            raise CausalUnsupportedError(
                f"{route} carries its own design and does not accept graph= or discovery=",
                reason_code="option_not_applicable",
            )
        selected = getattr(self.estimator, "value", self.estimator)
        if self.identifier is not None or (
            self.estimator is not None and selected not in allowed_ids
        ):
            raise CausalUnsupportedError(
                f"{route} selects its identifier and estimator itself; custom "
                "identifier= / estimator= are not supported",
                reason_code="option_not_applicable",
            )
        self._refuse_estimator_config(route)
        if self._explicit_refute() or self.bootstrap:
            raise CausalUnsupportedError(
                f"{route} has no refutation or bootstrap route",
                reason_code="option_not_applicable",
            )

    def _sequential_contrast(self) -> tuple[Any, Literal["average"]]:
        query = cast(SequentialContrast, self.query)
        selected = getattr(self.estimator, "value", self.estimator)
        bayesian = isinstance(self.inference, Bayesian)
        self._design_only(
            "SequentialContrast",
            ("sequential.crossfit_contrast.wald", "bayesian.sequential_crossfit_contrast"),
        )
        if selected is not None and bayesian != (
            selected == "bayesian.sequential_crossfit_contrast"
        ):
            raise CausalUnsupportedError(
                f"estimator {selected!r} does not run under this inference mode: the normal "
                "interval is Frequentist and the modular posterior is Bayesian",
                reason_code="estimator_inference_mismatch",
            )
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise CausalUnsupportedError(
                "SequentialContrast supports a normal interval or a Bayesian posterior",
                reason_code="option_not_applicable",
            )
        design = self.design_columns
        if design is None:
            raise CausalValueError("sequential contrast folds were not bound at prepare")
        native = _NativePreparedAnalysis.prepare_sequential_contrast(
            self.names,
            self.columns,
            query.outcome,
            query.first_treatment,
            query.second_treatment,
            list(query.baseline),
            list(query.intermediate),
            (int(query.active[0]), int(query.active[1])),
            (int(query.control[0]), int(query.control[1])),
            list(design["fold_of_row"]),
            int(design["fold_count"]),
            float(query.min_probability),
            accepted=False,
            **self._common(),
        )
        return native, "average"

    def _history_match(self) -> tuple[Any, Literal["average"]]:
        query = cast(PeriodHistoryMatch, self.query)
        self._design_only(
            "PeriodHistoryMatch",
            ("matching.period_history_contrast", "bayesian.period_history_outcome_model"),
        )
        # The matching contrast has no posterior; Bayesian inference reaches the model-based
        # effect on the treated only through the outcome-model estimator, which the native
        # compiler checks against the inference mode.
        selected = getattr(self.estimator, "value", self.estimator)
        if selected == "bayesian.period_history_outcome_model":
            self._select_ids()
        native = _NativePreparedAnalysis.prepare_history_match(
            self.names,
            self.columns,
            query.outcome,
            query.treated,
            list(query.history),
            query.period,
            query.weight,
            query.caliper,
            accepted=False,
            **self._common(),
        )
        return native, "average"

    def _longitudinal_regime(self) -> tuple[Any, Literal["average"]]:
        query = cast(LongitudinalRegime, self.query)
        if self.graph is not None or self.discovery is not None:
            raise CausalUnsupportedError(
                "LongitudinalRegime carries its own sequential design and does not accept graph= or discovery=",
                reason_code="option_not_applicable",
            )
        self._refuse_ids("LongitudinalRegime", "bayesian.sequential_regime_value")
        self._refuse_estimator_config("LongitudinalRegime")
        if self._explicit_refute() or self.bootstrap:
            raise CausalUnsupportedError(
                "LongitudinalRegime has no refutation or bootstrap route",
                reason_code="option_not_applicable",
            )
        if self.inference is not None and not isinstance(self.inference, (Frequentist, Bayesian)):
            raise CausalUnsupportedError(
                "LongitudinalRegime supports frequentist inference or a Bayesian sequential posterior",
                reason_code="option_not_applicable",
            )
        if not query.probabilities_known_by_design:
            raise CausalUnsupportedError(
                "retained longitudinal regime value requires known sequential randomization probabilities",
                reason_code="route_not_supported",
            )

        def flatten(matrix: Any) -> list[Any]:
            return [value for row in matrix for value in row]

        native = _NativePreparedAnalysis.prepare_longitudinal_regime(
            self.names,
            self.columns,
            query.outcome,
            query.periods,
            flatten(query.treatment_history),
            flatten(query.actions),
            flatten(query.treatment_probabilities),
            flatten(query.censoring_probabilities),
            list(resolved_opt(query.outcome_observed) or ()),
            list(resolved(query.subject_ids)),
            list(resolved_opt(query.fold_ids) or ()),
            query.excluded_fold_predictions,
            query.probabilities_known_by_design,
            query.minimum_probability,
            method=query.method,
            period_outcome_predictions=flatten(query.period_outcome_predictions)
            if query.period_outcome_predictions is not None
            else [],
            known_fixed_outcome_predictions=query.known_fixed_outcome_predictions,
            stabilizing_numerator_probabilities=list(query.stabilizing_numerator_probabilities)
            if query.stabilizing_numerator_probabilities is not None
            else [],
            q_predictions=flatten(query.q_predictions) if query.q_predictions is not None else [],
            observation_history=flatten(query.observation_history)
            if query.observation_history is not None
            else [],
            prediction_fold_ids=list(resolved(query.prediction_fold_ids))
            if query.prediction_fold_ids is not None
            else [],
            rule_id=query.rule_id,
            rule_version=query.rule_version,
            rule_provenance=query.rule_provenance,
            accepted=False,
            **self._common(),
        )
        return native, "average"

    def _conditional(self) -> tuple[Any, Any]:
        from .query import coerce_outcome_functional

        query, graph = self.query, self.graph
        self._refuse_estimator_config("ConditionalEffect (fixed conditional estimator)")
        mixed_graph = isinstance(graph, Admg)
        class_graph = isinstance(graph, (Pag, Cpdag)) or mixed_graph
        if mixed_graph:
            # A bidirected structure is read by IDC: the modifier-conditioned contrast.
            expected_identifier = "general.id"
            expected_estimator = "functional.effect"
        else:
            expected_identifier = "generalized.adjustment" if class_graph else "backdoor.adjustment"
            expected_estimator = (
                "conditional.bayesian"
                if isinstance(self.inference, Bayesian)
                else "conditional.linear.adjustment"
            )
        if self.identifier not in (None, expected_identifier) or self.estimator not in (
            None,
            expected_estimator,
        ):
            prefix = (
                "Admg ConditionalEffect"
                if mixed_graph
                else "Cpdag/Pag ConditionalEffect"
                if class_graph
                else "ConditionalEffect"
            )
            raise CausalUnsupportedError(
                f"{prefix} requires {expected_identifier} and {expected_estimator}"
            )
        native = _NativePreparedAnalysis.prepare_conditional(
            self.names,
            self.columns,
            [] if class_graph else _static_edges(graph),
            query.treatment,
            query.outcome,
            list(query.modifiers),
            graph=graph if class_graph else None,
            control_level=query.control_level,
            active_level=query.active_level,
            identifier=self.identifier,
            estimator=self.estimator,
            outcome_functional=coerce_outcome_functional(
                getattr(query, "outcome_functional", None)
            ),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    def _static_kind(self, edges: list[tuple[str, str]]) -> tuple[Any, Any]:
        query = self.query
        self._refuse_estimator_config(f"{type(query).__name__}")
        expected_id = (
            "path_specific.natural"
            if isinstance(query, (MediationEffect, NestedCounterfactual))
            else "gcm.parametric"
        )
        expected_est = (
            "mediation.linear"
            if isinstance(query, (MediationEffect, NestedCounterfactual))
            else "gcm.fit"
        )
        if self.identifier not in (None, expected_id) or self.estimator not in (None, expected_est):
            raise CausalUnsupportedError(f"{query.kind} requires {expected_id} and {expected_est}")
        if (
            isinstance(query, Counterfactual)
            and self.bootstrap
            and isinstance(self.inference, Bayesian)
        ):
            raise CausalUnsupportedError(
                "Bayesian Counterfactual publishes its mechanism-refit posterior; bootstrap= "
                "applies to the Frequentist mechanism-refit bootstrap",
                reason_code="option_not_applicable",
            )
        if isinstance(query, NestedCounterfactual) and self.bootstrap:
            raise CausalUnsupportedError(
                "NestedCounterfactualEffect is point-only until its interval route is calibrated",
                reason_code="cell_not_licensed",
            )
        native = _NativePreparedAnalysis.prepare_static_kind(
            self.names,
            self.columns,
            edges,
            query.kind,
            query.treatment,
            query.outcome,
            mediators=(
                [query.mediator]
                if isinstance(query, NestedCounterfactual)
                else list(query.mediators)
                if isinstance(query, MediationEffect)
                else []
            ),
            contrast=query.contrast if isinstance(query, MediationEffect) else "mediated",
            control_level=query.control_level,
            active_level=query.active_level,
            **(
                {
                    "extra_outcomes": list(query.also_outcomes),
                    "joint_active": list(query.active_interventions.items()),
                    "joint_control": list(query.control_interventions.items()),
                }
                if isinstance(query, Counterfactual)
                else {}
            ),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    # -- design-based cells -----------------------------------------------------
    def _refuse_design_options(self, route: str, identifier: str, estimator: str) -> None:
        """The design cells fix their identifier, estimator and interval construction."""
        self._refuse_estimator_config(route)
        if self.identifier not in (None, identifier) or self.estimator not in (None, estimator):
            raise CausalUnsupportedError(
                f"{route} is identified by {identifier} and estimated by {estimator}; "
                "another identifier= / estimator= does not apply",
                reason_code="option_not_applicable",
            )
        if self.bootstrap:
            raise _not_applicable(
                "bootstrap", f"{route} (its interval is the {estimator} analytic construction)"
            )

    def _transport(self) -> tuple[Any, Any]:
        from .transport._impl import _response_args

        query = cast(TransportQuery, self.query)
        self._refuse_design_options("TransportQuery", "transport.sid", "transport.trial_ipw")
        if not isinstance(self.graph, Admg):
            raise CausalTypeError(
                "TransportQuery reads a selection diagram, which is an Admg; pass graph=Admg(...)"
            )
        columns = query.trial_columns
        if columns is None:
            raise CausalValueError(
                "TransportQuery on analyze reads its trial columns: pass trial=, "
                "selection_probability= and treatment_probability="
            )
        response = _response_args(query.query)
        native = _NativePreparedAnalysis.prepare_transport(
            self.names,
            self.columns,
            self.graph,
            list(query.diagram.selections),
            query.diagram.source,
            query.diagram.target,
            list(query.source_experiments),
            response["kind"],
            response["treatments"],
            response["outcomes"],
            columns[0],
            columns[1],
            columns[2],
            catalog=query.catalog,
            grid=response["grid"],
            at=response["at"],
            direction=response["direction"],
            order=response["order"],
            scale=response["scale"],
            weighting=response["weighting"],
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    def _interference(self) -> tuple[Any, Any]:
        from .interference import (
            ClusterRandomization,
            ObservedExposureDesign,
            SaturationDesign,
            _assignment_args,
            _edge_values,
            _exposure_name,
            _partition,
        )

        query = cast(InterferenceQuery, self.query)
        self._refuse_design_options(
            "InterferenceQuery",
            "interference.design",
            "interference.observational_ipw"
            if isinstance(query.assignment, ObservedExposureDesign)
            else "interference.saturation_exact"
            if isinstance(query.assignment, SaturationDesign)
            else "interference.cluster_neyman"
            if isinstance(query.assignment, ClusterRandomization)
            else "interference.ht_hajek",
        )
        if query.network is None or query.realized_assignment is None:
            raise CausalValueError(
                "InterferenceQuery on analyze reads its design: pass network= (the fixed "
                "exposure edges) and realized_assignment="
            )
        if query.partial_interference is not None:
            if not isinstance(
                query.assignment, (ClusterRandomization, SaturationDesign, ObservedExposureDesign)
            ):
                raise CausalValueError(
                    "partial_interference requires cluster or saturation randomization so cluster assignment is explicit"
                )
            partial_clusters = list(query.partial_interference.clusters)
            assignment_clusters = list(query.assignment.clusters)
            if len(partial_clusters) != len(assignment_clusters) or len(partial_clusters) != len(
                self.columns[0]
            ):
                raise CausalValueError(
                    "partial-interference and assignment clusters must match data rows"
                )
            edges = _edge_values(query.network)
            if any(
                source < 0
                or target < 0
                or source >= len(partial_clusters)
                or target >= len(partial_clusters)
                for source, target, _ in edges
            ):
                raise CausalValueError("network edge index is outside data rows")
            if _partition(partial_clusters) != _partition(assignment_clusters):
                raise CausalValueError(
                    "partial-interference clusters must match the cluster-randomization partition"
                )
            if any(
                partial_clusters[source] != partial_clusters[target] for source, target, _ in edges
            ):
                raise CausalValueError(
                    "partial-interference assumption violated: network edge crosses cluster boundary"
                )
        elif isinstance(
            query.assignment, (ClusterRandomization, SaturationDesign, ObservedExposureDesign)
        ):
            raise CausalValueError(
                "cluster interference requires partial_interference=PartialInterference(clusters)"
            )
        design = _assignment_args(query.assignment)
        contrast = query.functional
        native = _NativePreparedAnalysis.prepare_interference(
            self.names,
            self.columns,
            _static_edges(self.graph),
            contrast.outcome,
            _edge_values(query.network),
            [bool(value) for value in query.realized_assignment],
            design["assignment_kind"],
            design["assignment_probabilities"],
            design["treated"],
            design["clusters"],
            design["treated_clusters"],
            _exposure_name(query.exposure),
            (contrast.from_.own, contrast.from_.neighbors),
            (contrast.to.own, contrast.to.neighbors),
            probability_draws=query.probability_draws,
            low_probability=design.get("low_probability", 0.0),
            high_probability=design.get("high_probability", 0.0),
            realized_saturation=design.get("realized_saturation", []),
            propensity_from=design.get("propensity_from", []),
            propensity_to=design.get("propensity_to", []),
            propensity_provenance=design.get("propensity_provenance", "known"),
            assume_network_exchangeability=design.get("assume_network_exchangeability", False),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    def _attribution(self) -> tuple[Any, Any]:
        query = self.query
        route = type(query).__name__
        self._refuse_ids(route)
        self._refuse_estimator_config(route)
        # A Frequentist replicate count runs the whole-procedure mechanism-refit row
        # bootstrap (unlicensed, calibration unmeasured); the Bayesian cell publishes
        # its Bayesian-bootstrap posterior instead.
        if self.bootstrap and isinstance(self.inference, Bayesian):
            raise CausalUnsupportedError(
                f"Bayesian {route} publishes its Bayesian-bootstrap posterior; bootstrap= "
                "applies to the Frequentist mechanism-refit bootstrap",
                reason_code="option_not_applicable",
            )
        # Cheap and full validation run the functional refuter state on the whole
        # attribution vector (native, unlicensed); the ATE-shaped suites do not apply.
        if isinstance(self.graph, Dag):
            edges = [(str(a), str(b)) for a, b in self.graph.edges()]
        elif isinstance(self.graph, Sequence) and not isinstance(self.graph, (str, bytes)):
            items = list(self.graph)
            if items and len(items[0]) != 2:
                raise CausalTypeError(f"{route} requires graph=Dag(...) or an edge list")
            edges = [(str(a), str(b)) for a, b in items]
        else:
            raise CausalTypeError(f"{route} requires graph=Dag(...) or an edge list")
        if isinstance(query, AnomalyAttribution):
            reference = (
                (float(query.reference.center), float(query.reference.scale))
                if query.reference is not None
                else None
            )
            native = _NativePreparedAnalysis.prepare_anomaly_attribution(
                self.names,
                self.columns,
                edges,
                list(query.targets),
                int(query.max_units),
                accepted=self.accepted,
                reference=reference,
                **self._common(),
            )
        else:
            query = cast(ChangeAttribution, query)
            native = _NativePreparedAnalysis.prepare_change_attribution(
                self.names,
                self.columns,
                edges,
                query.outcome,
                int(query.baseline_start),
                int(query.baseline_end),
                int(query.comparison_start),
                int(query.comparison_end),
                accepted=self.accepted,
                **self._common(),
            )
        return native, "average"

    # -- static responses -------------------------------------------------------
    def _refuse_response_bootstrap(self) -> None:
        """Only Frequentist temporal surfaces resample; elsewhere refuse the request."""
        if not self.bootstrap:
            return
        raise CausalUnsupportedError(
            "Bayesian responses use posterior intervals; bootstrap is unsupported"
            if isinstance(self.inference, Bayesian)
            else "prepared static responses use analytic or influence-function "
            "uncertainty; bootstrap= applies to Frequentist temporal responses only"
        )

    def _response(self) -> tuple[Any, Any]:
        query, graph = self.query, self.graph
        # A scalar Dag InterventionResponse has an ATE-shaped state, so the
        # refuter suite denotes on it (Bayesian: the unlicensed Bayesian refuter
        # state); a curve is function valued and cheap/full name nothing there; a
        # derivative runs its own unlicensed refuter state. Admg
        # InterventionResponse cheap/full is the plugin-level suite on the
        # identified mean.
        if isinstance(graph, Admg) and isinstance(query, (ResponseCurve, InterventionResponse)):
            response_options = _parse_response_estimator_config(query, self.estimator_config)
            return self._admg_response(response_options)
        # A plain edge list is a supplied Dag here, as it is for every other static route.
        edge_list = (
            isinstance(graph, Sequence)
            and not isinstance(graph, (str, bytes))
            and all(isinstance(item, Sequence) and len(item) == 2 for item in graph)
        )
        scalar_dag_response = isinstance(query, InterventionResponse) and (
            isinstance(graph, Dag) or edge_list
        )
        if self._explicit_refute() and scalar_dag_response:
            # The response executor has no validation stage; the study runs the
            # suite as the second click of every estimate.
            self.deferred_suite = "placebo" if self.refute is True else str(self.refute)
            self.options.pop("refute", None)
        derivative = isinstance(
            query,
            (
                PointDerivative,
                AverageDerivative,
                Elasticity,
                SemiElasticity,
                DirectionalDerivative,
                ResponseJacobian,
            ),
        )
        # A static derivative's cheap/full suite is the derivative refuter state
        # (unlicensed); the native study runs it, so the suite stays in the options.
        if self._explicit_refute() and not scalar_dag_response and not derivative:
            raise CausalUnsupportedError(
                "not_applicable: a function-valued response has no ATE-shaped state for the "
                "cheap/full/placebo refuter suite; prepare it with refute='none'",
                reason_code="refutation_not_applicable",
            )
        if isinstance(query, (ResponseCurve, InterventionResponse)):
            _check_quantile_functional(query, self.inference, self.estimator, graph)
        response_options = _parse_response_estimator_config(query, self.estimator_config)
        if isinstance(query, (ResponseCurve, InterventionResponse)) and isinstance(
            graph, (Pag, Cpdag)
        ):
            return self._class_response(response_options)
        if isinstance(query, InterventionResponse) and isinstance(graph, TieredBackground):
            return self._tiered_response()
        _check_response_strategy(
            query,
            graph=graph,
            identifier=self.identifier,
            estimator=self.estimator,
            inference=self.inference,
        )
        edges = _static_edges(graph)
        if isinstance(query, ResponseCurve):
            return self._response_curve(edges, response_options)
        if isinstance(query, InterventionResponse):
            return self._intervention_response(edges)
        return self._derivative(edges, response_options)

    def _admg_response(self, response_options: Mapping[str, Any]) -> tuple[Any, Any]:
        query, graph = self.query, self.graph
        if self.identifier not in (None, "general.id"):
            raise CausalUnsupportedError("Admg response requires identifier='general.id'")
        if self.estimator not in (None, "functional.effect"):
            raise CausalUnsupportedError(
                f"Admg response requires estimator='functional.effect'; got {self.estimator!r}"
            )
        if isinstance(query, ResponseCurve):
            native = _NativePreparedAnalysis.prepare_class_response(
                self.names,
                self.columns,
                graph,
                query.kind,
                [query.treatment],
                [query.outcome],
                grid=list(query.grid),
                identifier=self.identifier or "general.id",
                estimator=self.estimator or "functional.effect",
                response_options=_response_options_wire(response_options, self.seed)
                if response_options
                else None,
                accepted=self.accepted,
                **self._common(),
            )
            return native, "response_curve"
        treatments, kinds, parameters = _encode_interventions(
            query.intervention,
            route="functional.effect",
            soft_hint="require a structural/temporal model",
        )
        native = _NativePreparedAnalysis.prepare_class_response(
            self.names,
            self.columns,
            graph,
            "intervention_response",
            treatments,
            [query.outcome],
            intervention_kinds=kinds,
            intervention_parameters=parameters,
            identifier=self.identifier or "general.id",
            estimator=self.estimator or "functional.effect",
            accepted=self.accepted,
            **self._common(),
        )
        return native, "intervention_response"

    def _class_response(self, response_options: Mapping[str, Any]) -> tuple[Any, Any]:
        query, graph = self.query, self.graph
        if self.identifier not in (None, "generalized.adjustment"):
            raise CausalUnsupportedError(
                "Cpdag/Pag response requires identifier='generalized.adjustment'"
            )
        if isinstance(self.inference, Bayesian):
            expected = "response.bayesian"
        elif isinstance(query, ResponseCurve):
            expected = "response.kennedy_dr"
        else:
            expected = "response.intervention_gcomp"
        if self.estimator not in (None, expected):
            raise CausalUnsupportedError(
                f"Cpdag/Pag response requires estimator={expected!r}; got {self.estimator!r}"
            )
        observation = getattr(query, "observation", None)
        if observation is not None:
            from .observation import Complete

            if not isinstance(observation, Complete):
                raise CausalUnsupportedError(
                    "an observation mechanism is estimated on one identified adjustment set; "
                    "a PAG/CPDAG envelope mixes several completions, each needing its own "
                    "observation model",
                    reason_code="option_not_applicable",
                )
        if isinstance(query, ResponseCurve):
            native = _NativePreparedAnalysis.prepare_class_response(
                self.names,
                self.columns,
                graph,
                query.kind,
                [query.treatment],
                [query.outcome],
                grid=list(query.grid),
                identifier=self.identifier,
                estimator=self.estimator,
                response_options=_response_options_wire(response_options, self.seed)
                if response_options
                else None,
                accepted=self.accepted,
                **self._common(),
            )
            return native, "response_curve"
        treatments, kinds, parameters = _encode_interventions(
            query.intervention,
            route="response.intervention_gcomp",
            soft_hint="require a structural/temporal model",
        )
        native = _NativePreparedAnalysis.prepare_class_response(
            self.names,
            self.columns,
            graph,
            "intervention_response",
            treatments,
            [query.outcome],
            intervention_kinds=kinds,
            intervention_parameters=parameters,
            identifier=self.identifier,
            estimator=self.estimator,
            accepted=self.accepted,
            **self._common(),
        )
        return native, "intervention_response"

    def _tiered_response(self) -> tuple[Any, Any]:
        from .query import coerce_outcome_functional

        query, graph = self.query, self.graph
        bayesian = not isinstance(self.inference, Frequentist)
        if self.identifier not in (None, "generalized.adjustment"):
            raise CausalUnsupportedError(
                "CoDetermined joint cells require identifier generalized.adjustment"
            )
        # The Frequentist joint cell is cell.aipw; a Bayesian declaration runs
        # response.bayesian over the same closure set and is unlicensed.
        expected_estimator = "response.bayesian" if bayesian else "cell.aipw"
        if self.estimator not in (None, expected_estimator):
            raise CausalUnsupportedError(
                f"CoDetermined joint cells require estimator {expected_estimator}"
            )
        from . import intervention as intervention_specs

        supplied = query.intervention
        specs = (
            list(supplied)
            if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
            else [supplied]
        )
        if len(specs) < 2:
            raise CausalUnsupportedError(
                "cell-AIPW on a tiered background requires joint InterventionResponse"
            )
        if not all(isinstance(spec, intervention_specs.Set) for spec in specs):
            raise CausalUnsupportedError(
                "CoDetermined joint cells require binary Set interventions"
            )
        native = _NativePreparedAnalysis.prepare_tiered_intervention_response(
            self.names,
            self.columns,
            [list(tier) for tier in graph.tiers],
            str(graph.within_tier),
            query.outcome,
            [spec.variable for spec in specs],
            ["set"] * len(specs),
            [[spec.value] for spec in specs],
            outcome_functional=coerce_outcome_functional(query.outcome_functional),
            **self._common(),
        )
        return native, "intervention_response"

    def _response_curve(
        self, edges: list[tuple[str, str]], response_options: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        query = self.query
        from .observation import (
            Complete,
            _assumption_kwargs,
            _ensure_latent_schema_column,
            _mechanism_kwargs,
        )

        if query.observation is not None and not isinstance(query.observation, Complete):
            from ._native import prepare_observation_response

            if not isinstance(self.inference, Frequentist):
                raise CausalUnsupportedError("observation-adjusted responses require Frequentist")
            if len(query.observation_assumptions) != 1:
                raise ValueError("observation response requires exactly one explicit assumption")
            if response_options:
                raise _not_applicable("estimator_config", "an observation-adjusted response")
            if self.identifier not in (None, "response.backdoor") or self.estimator not in (
                None,
                "response.kennedy_dr",
            ):
                raise CausalUnsupportedError(
                    "observation response requires response.backdoor and response.kennedy_dr"
                )
            names, columns = _ensure_latent_schema_column(
                self.names, self.columns, query.observation
            )
            kwargs = _mechanism_kwargs(query.observation)
            kwargs.update(_assumption_kwargs(query.observation_assumptions[0]))
            native = prepare_observation_response(
                names,
                columns,
                edges,
                query.treatment,
                query.outcome,
                list(query.grid),
                accepted=self.accepted,
                **cast(dict[str, Any], kwargs),
                **self._common(),
            )
            return native, "response_curve"
        native = _NativePreparedAnalysis.prepare_response(
            self.names,
            self.columns,
            edges,
            query.treatment,
            query.outcome,
            list(query.grid),
            identifier=self.identifier,
            estimator=self.estimator,
            accepted=self.accepted,
            response_options=_response_options_wire(response_options, self.seed),
            **self._common(),
        )
        return native, "response_curve"

    def _intervention_response(self, edges: list[tuple[str, str]]) -> tuple[Any, Any]:
        from .query import coerce_outcome_functional

        query = self.query
        # A Bayesian intervention level's cheap/full suite is the Bayesian refuter
        # state (plugin-level suite on the posterior-mean level, prior and posterior
        # predictive checks, and under full level-native prior sensitivity). It runs
        # as the deferred second click and stays unlicensed.
        treatments, kinds, parameters = _encode_interventions(
            query.intervention,
            route="response.intervention_gcomp",
            soft_hint="require a temporal response cell (set horizons=...)",
        )
        native = _NativePreparedAnalysis.prepare_intervention_response(
            self.names,
            self.columns,
            edges,
            query.outcome,
            treatments,
            kinds,
            parameters,
            identifier=self.identifier,
            estimator=self.estimator,
            outcome_functional=coerce_outcome_functional(query.outcome_functional),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "intervention_response"

    def _derivative(
        self, edges: list[tuple[str, str]], response_options: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        from .observation import Complete

        query = self.query
        if getattr(query, "observation", None) is not None and not isinstance(
            query.observation, Complete
        ):
            raise CausalUnsupportedError("derivatives require complete observations")
        if query.observation_assumptions:
            raise CausalUnsupportedError("derivatives require the unweighted observed population")
        target = _derivative_target_spec(query)
        native = _NativePreparedAnalysis.prepare_derivative(
            self.names,
            self.columns,
            edges,
            target.pop("kind"),
            target.pop("treatments"),
            target.pop("outcomes"),
            **target,
            response_options=dict(response_options),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "response_curve"


def _derivative_target_spec(query: Any) -> dict[str, Any]:
    """Resolve scalar/vector coordinates once for static and temporal native targets."""
    if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
        treatments = list(query.treatments)
        outcomes = list(query.outcomes)
    else:
        treatments = [query.treatment]
        outcomes = [query.outcome]
    at = getattr(query, "at", None)
    if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
        at = [at[t] for t in treatments] if isinstance(at, Mapping) else list(query.at)
    elif at is not None:
        at = [at]
    direction = getattr(query, "direction", None)
    if direction is not None:
        direction = (
            [direction[t] for t in treatments]
            if isinstance(direction, Mapping)
            else list(direction)
        )
    scale = (
        "log_log"
        if isinstance(query, Elasticity)
        else "log_" + query.log_scale
        if isinstance(query, SemiElasticity)
        else "identity"
    )
    return {
        "kind": query.kind,
        "treatments": treatments,
        "outcomes": outcomes,
        "at": at,
        "direction": direction,
        "order": getattr(query, "order", 1),
        "scale": scale,
        "weighting": getattr(query, "weighting", None) or "observed",
    }


def _check_quantile_functional(
    query: Any, inference: Any, estimator: str | None, graph: Any
) -> None:
    from .query import coerce_outcome_functional

    functional = coerce_outcome_functional(getattr(query, "outcome_functional", None))
    if (
        functional is not None
        and functional.get("kind") == "quantile"
        and (
            not isinstance(query, InterventionResponse)
            or getattr(query, "is_temporal", False)
            or isinstance(inference, Bayesian)
            or (
                str(estimator) != "cell.aipw"
                and not (estimator is None and isinstance(graph, TieredBackground))
            )
        )
    ):
        raise CausalUnsupportedError(
            "quantiles require Frequentist AllObserved AIPW AverageEffect, binary "
            "ConditionalEffect (one modifier or a modifier set), or cell-AIPW joint response; use prepare + "
            "retarget for score-table target weights",
            reason_code="route_not_supported",
        )
