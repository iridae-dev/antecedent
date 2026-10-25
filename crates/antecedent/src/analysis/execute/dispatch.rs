// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AnalysisRoute, Arc, CausalError, CausalQuery, CheckedInterferenceOperation,
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER_ID,
    DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR_ID,
    DEFAULT_PATH_IDENTIFIER_ID, DEFAULT_RESPONSE_IDENTIFIER_ID, DataInput, DataModality,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EstimatorId, ExecutionContext, GraphClass,
    GraphPosterior, GraphRef, IdentificationResult, IdentifierId, InferenceMode, OverlapPolicy,
    PhysicalExecutionPlan, RefuteSuite, ResponseQuery, RouteInput, StaticMediationSource, Strategy,
    StudyResult, TableView, TabularData, TimeSeriesData, VariableId, admg_has_bidirected,
    admg_without_latents_to_dag, classify_analysis_route, classify_route, data_modality,
    graph_posterior_response_supported, is_class_atom, reject_dag_only_on_pag,
    response_functional_is_derivative, temporal_admg_projection,
};
use crate::analysis::prepared::ExecutionKind;
use crate::analysis::route_guards::{
    complete_mean_response, distinct_numeric_arms, mean_all_observed,
};
use crate::graph::Dag;
use crate::planner::RouteDefaults;
use antecedent_core::NestedCounterfactualQuery;

/// Refusal for a graph-posterior study whose data / query pair has no route.
pub(super) const GRAPH_POSTERIOR_QUERY_REFUSAL: &str = concat!(
    "graph-posterior analysis supports tabular average-effect, conditional-effect, static ",
    "response, transport, interference, mediation, path-specific, natural-direct or ",
    "attribution queries, and series temporal-effect, temporal-mediation, or temporal ",
    "response queries only",
);

/// The refusal for a route whose data or query kind disagrees with the one the route was
/// classified from. Classification reads both, so reaching this means a caller swapped the
/// study's data or query between planning and execution.
fn route_mismatch() -> CausalError {
    CausalError::Compile {
        message: "execution route does not match the study's data or query kind".into(),
    }
}

/// The wire names of a typed identifier/estimator pair, as plan records carry them.
pub(super) fn wire_pair(
    (identifier, estimator): (IdentifierId, EstimatorId),
) -> (Arc<str>, Arc<str>) {
    (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
}

impl super::Study {
    /// The estimator the caller selected, or `None` for a builder placeholder.
    ///
    /// A caller selects an estimator through an estimator spec. An `estimator`
    /// without one is a placeholder the builder bound on its own
    /// (`inference()` binds `BayesianGcomp` on a static average effect,
    /// `query()` binds `TemporalLinearAdjustment` on a temporal one) and the
    /// compiled route keeps its own estimator instead. Every route that
    /// resolves or advertises an estimator decides "caller choice or
    /// placeholder" here.
    pub(crate) fn caller_selected_estimator(&self) -> Option<EstimatorId> {
        self.estimator.filter(|_| self.estimator_spec.is_some())
    }

    pub(super) fn validation_suite_id(&self) -> Option<Arc<str>> {
        let family = match self.query {
            CausalQuery::PathSpecific(_) => Some("path"),
            CausalQuery::Distribution(_) => Some("distribution"),
            CausalQuery::Mediation(_) if self.graph.class() == GraphClass::TemporalDag => {
                Some("temporal.mediation")
            }
            CausalQuery::Mediation(_)
                if matches!(
                    self.graph.class(),
                    GraphClass::Admg
                        | GraphClass::TemporalAdmg
                        | GraphClass::Cpdag
                        | GraphClass::Pag
                        | GraphClass::TemporalPag
                ) =>
            {
                Some("mediation.decomposition")
            }
            _ => None,
        };
        if let Some(family) = family {
            return match self.refute {
                RefuteSuite::None => None,
                RefuteSuite::Full => Some(Arc::from(format!("{family}.full"))),
                _ => Some(Arc::from(format!("{family}.cheap"))),
            };
        }
        self.refute.validation_suite_id().map(Arc::from)
    }

    pub(super) fn ensure_supported_combination(&self) -> Result<(), CausalError> {
        if let CausalQuery::Discover(query) = &self.query {
            // A structure search reads no supplied graph; its admission is the
            // method against the data modality.
            return self.ensure_discovery_admitted(query);
        }
        if let CausalQuery::Association(query) = &self.query {
            // An association reads no supplied graph; its admission is the declared
            // sampling design against the data modality.
            return self.ensure_association_admitted(query);
        }
        let class = self.graph.class();
        self.ensure_nested_counterfactual_licensed(class)?;
        if self.estimator == Some(EstimatorId::BayesianBasisGcomp)
            && (!matches!(self.data, DataInput::Tabular(_))
                || class != GraphClass::Dag
                || !matches!(
                    self.query,
                    CausalQuery::AverageEffect(_) | CausalQuery::ConditionalEffect(_)
                )
                || !self.inference.is_bayesian())
        {
            return Err(CausalError::unsupported_inference(
                "bayesian.basis.gcomp requires Bayesian inference, tabular data, a supplied DAG, and an ATE or CATE query",
            ));
        }
        if let DataInput::Panel(panel) = &self.data {
            super::super::builder::refuse_unlicensed_panel_route(
                &self.query,
                class,
                &self.inference,
                panel,
                self.split.as_ref(),
            )?;
        }
        self.ensure_data_query_graph_cell(class)?;
        self.ensure_temporal_effect_selection()
    }

    /// Nested-counterfactual admission: the temporal twin or the fixed three-node linear
    /// mediation DAG, and nothing else.
    fn ensure_nested_counterfactual_licensed(&self, class: GraphClass) -> Result<(), CausalError> {
        if let CausalQuery::NestedCounterfactual(q) = &self.query {
            q.validate()?;
            if self.graph_posterior.is_some() && matches!(self.data, DataInput::Tabular(_)) {
                // Each atom of a tabular graph posterior answers the natural direct effect on
                // its own structure; the placeholder graph names no edge to check.
            } else if matches!(
                self.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            ) && matches!(self.data, DataInput::Temporal(_) | DataInput::Event(_))
            {
                // The temporal twin is the linear direct slope on a TemporalDag.
            } else if matches!(
                (&self.data, class),
                (
                    DataInput::Tabular(_),
                    GraphClass::Admg
                        | GraphClass::TemporalAdmg
                        | GraphClass::Cpdag
                        | GraphClass::Pag
                )
            ) {
                // The natural direct effect on a mixed graph is identified by the
                // path-specific identifier over the ADMG, on a CPDAG in each DAG
                // completion, and on a PAG over the ADMGs each MAG completion stands for;
                // the identification decides whether the graph's structure allows it.
            } else {
                let Some(graph) = self.graph.as_dag() else {
                    return Err(crate::unsupported_reason!(
                        "cross_world_not_identified",
                        "natural direct effect requires the licensed three-node DAG"
                    ));
                };
                let expected = [
                    (q.treatment.raw(), q.mediator.raw()),
                    (q.treatment.raw(), q.outcome.raw()),
                    (q.mediator.raw(), q.outcome.raw()),
                ];
                let observed = graph.directed_edge_pairs_sorted();
                let mut expected = expected.to_vec();
                expected.sort_unstable();
                if graph.node_count() != 3 || observed != expected {
                    return Err(crate::unsupported_reason!(
                        "cross_world_not_identified",
                        "query requires exactly X -> M, X -> Y, M -> Y with no other nodes or edges"
                    ));
                }
                if !matches!((&self.data, class), (DataInput::Tabular(_), GraphClass::Dag)) {
                    return Err(crate::unsupported_reason!(
                        "cross_world_not_identified",
                        "nested effect requires tabular data and the fixed DAG"
                    ));
                }
                if self.estimator.is_some_and(|id| id != EstimatorId::StaticMediationLinear)
                    || self.identifier.is_some_and(|id| id != IdentifierId::PathSpecificNatural)
                    || self.bootstrap_replicates != 0
                {
                    return Err(crate::unsupported_reason!(
                        "cross_world_not_identified",
                        "only the point-only linear Gaussian mediation route is licensed for this query"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Refusals for data / query / graph-class cells that no route covers, plus the
    /// PAG-only identifier check for tabular ATE.
    /// Response, distribution, and path-specific queries need their own data / graph-class
    /// pairings; the other cells are checked by [`Self::ensure_data_query_graph_cell`].
    fn ensure_query_graph_class_cell(&self, class: GraphClass) -> Result<(), CausalError> {
        match (&self.data, &self.query, class) {
            (_, CausalQuery::Response(q), class)
                if !((!q.is_temporal()
                    && matches!(
                        (&self.data, class),
                        (
                            DataInput::Tabular(_),
                            GraphClass::Dag
                                | GraphClass::Cpdag
                                | GraphClass::Pag
                                | GraphClass::Admg
                                | GraphClass::TemporalAdmg
                        )
                    ))
                    || (q.is_temporal()
                        && matches!(
                            (&self.data, class),
                            (
                                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::Panel(_),
                                GraphClass::TemporalDag
                                    | GraphClass::TemporalCpdag
                                    | GraphClass::TemporalPag
                                    | GraphClass::TemporalAdmg
                            )
                        ))) =>
            {
                return Err(CausalError::Unsupported {
                    message: "static CausalQuery::Response requires tabular data and a Dag, \
                              Cpdag, Pag, or Admg (or a CoDetermined tier closure); temporal \
                              response requires series/event/panel data and a temporal graph",
                });
            }
            (_, CausalQuery::Distribution(_), class)
                if !(self.graph_posterior.is_some() && class == GraphClass::Pag)
                    && !matches!(
                    (&self.data, class),
                    (
                        DataInput::Tabular(_),
                        GraphClass::Dag
                            | GraphClass::Cpdag
                            | GraphClass::Pag
                            | GraphClass::Admg
                            | GraphClass::TemporalAdmg
                    ) | (
                        DataInput::Temporal(_) | DataInput::Event(_),
                        GraphClass::TemporalDag
                            | GraphClass::TemporalCpdag
                            | GraphClass::TemporalPag
                    )
                ) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::Distribution requires tabular data and a static Dag, Cpdag, Pag or Admg",
                });
            }
            (_, CausalQuery::PathSpecific(_), class)
                if !matches!(
                    (&self.data, class),
                    (
                        DataInput::Tabular(_),
                        GraphClass::Dag
                            | GraphClass::Cpdag
                            | GraphClass::Pag
                            | GraphClass::Admg
                            | GraphClass::TemporalAdmg
                    ) | (
                        DataInput::Temporal(_) | DataInput::Event(_),
                        GraphClass::TemporalDag
                            | GraphClass::TemporalCpdag
                            | GraphClass::TemporalPag
                    )
                ) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::PathSpecific requires tabular data and a static Dag, Cpdag, Pag or Admg",
                });
            }
            _ => {}
        }
        Ok(())
    }

    fn ensure_data_query_graph_cell(&self, class: GraphClass) -> Result<(), CausalError> {
        self.ensure_query_graph_class_cell(class)?;
        match (&self.data, &self.query, class) {
            (DataInput::Tabular(_), CausalQuery::TemporalEffect(_), _) => {
                return Err(CausalError::Compile {
                    message: "temporal effect query requires temporal data".into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                CausalQuery::AverageEffect(_),
                _,
            ) => {
                return Err(CausalError::Compile {
                    message: "static ATE on temporal data is unsupported; use TemporalEffect"
                        .into(),
                });
            }
            (DataInput::Panel(_), CausalQuery::AverageEffect(_), _) => {
                return Err(CausalError::Compile {
                    message: "static ATE on panel data is unsupported; use TemporalEffect".into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                _,
                GraphClass::Pag | GraphClass::Cpdag | GraphClass::Admg,
            ) => {
                return Err(CausalError::Compile {
                    message: "static Pag/Cpdag/Admg requires tabular data and an average-effect \
                              or static response query"
                        .into(),
                });
            }
            (DataInput::MultiEnv(_), _, class) if class != GraphClass::TemporalDag => {
                return Err(CausalError::Compile {
                    message: "multi-environment data currently supports only a supplied \
                              TemporalDag"
                        .into(),
                });
            }
            (
                DataInput::Panel(_),
                CausalQuery::TemporalEffect(_) | CausalQuery::Response(_),
                GraphClass::TemporalCpdag | GraphClass::TemporalPag,
            ) => {}
            (DataInput::Panel(_), _, class) if class != GraphClass::TemporalDag => {
                return Err(CausalError::Compile {
                    message: "panel data supports only a supplied TemporalDag, TemporalCpdag, \
                              or TemporalPag with Pulse/Sustained or temporal response"
                        .into(),
                });
            }
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_), GraphClass::Pag) => {
                let (identifier, _) = self.resolve_pag_ids();
                reject_dag_only_on_pag(&self.graph, identifier)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// The temporal-effect path accepts only its linear / temporal identifier and
    /// estimator selections; any other explicit choice is refused here.
    fn ensure_temporal_effect_selection(&self) -> Result<(), CausalError> {
        // The temporal path is linear/temporal-backdoor only; refuse an explicitly
        // selected non-temporal identifier/estimator rather than silently ignoring it.
        if matches!(&self.query, CausalQuery::TemporalEffect(_)) {
            let class_aware = self.graph.class().is_incomplete_temporal();
            // A temporal ADMG contrast is identified by the complete ID algorithm on the unrolled
            // window and evaluated by the functional plug-in; the builder's placeholder
            // estimator is not a selection.
            if self.graph.class() == GraphClass::TemporalAdmg {
                let identifier_ok = self.identifier.is_none_or(|id| id == IdentifierId::GeneralId);
                let estimator_ok = self
                    .caller_selected_estimator()
                    .is_none_or(|est| est == EstimatorId::FunctionalEffect);
                if !identifier_ok || !estimator_ok {
                    return Err(crate::compile_reason!(
                        "strategy_incompatible",
                        "a contrast on a TemporalAdmg is identified by general.id on its unrolled \
                         window and evaluated by functional.effect; got identifier {:?} and \
                         estimator {:?}",
                        self.identifier,
                        self.caller_selected_estimator()
                    ));
                }
                return Ok(());
            }
            if let Some(id) = &self.identifier {
                let ok = if class_aware {
                    *id == IdentifierId::GeneralizedAdjustment
                } else {
                    *id == IdentifierId::TemporalBackdoorUnfolded
                        || (*id == IdentifierId::TemporalIvUnfolded
                            && matches!(
                                self.estimator,
                                Some(
                                    EstimatorId::TemporalLaggedInstrument
                                        | EstimatorId::BayesianTemporalLaggedInstrument
                                )
                            ))
                };
                if !ok {
                    return Err(CausalError::Compile {
                        message: format!(
                            "temporal path identifier {id:?} is not valid for this graph class"
                        ),
                    });
                }
            }
            if let Some(est) = &self.estimator {
                let multi_step = matches!(
                    &self.query,
                    CausalQuery::TemporalEffect(q) if q.is_multi_step_sustained()
                );
                let bayesian = self.inference.is_bayesian();
                let ok = *est == EstimatorId::TemporalLinearAdjustment
                    || matches!(
                        est,
                        EstimatorId::TemporalGlmLinkContrast
                            | EstimatorId::BayesianTemporalGlmLinkContrast
                            | EstimatorId::TemporalLaggedInstrument
                            | EstimatorId::BayesianTemporalLaggedInstrument
                    )
                    || (multi_step && *est == EstimatorId::TemporalSequentialGcomp)
                    || (bayesian && !multi_step && *est == EstimatorId::BayesianTemporalGcomp);
                if !ok {
                    return Err(CausalError::Compile {
                        message: format!(
                            "temporal path only supports estimator \"temporal.linear.adjustment\" \
                             (\"bayesian.temporal.gcomp\" under Bayesian inference, or \
                             \"temporal.sequential.gcomp\" for multi-step Sustained); got {est:?}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Resolve builder-selected identifier/estimator ids against modality defaults.
    pub(super) fn resolve_id_est_pair(
        &self,
        default_identifier: IdentifierId,
        default_estimator: EstimatorId,
    ) -> (Arc<str>, Arc<str>) {
        wire_pair(self.resolve_id_est_ids(default_identifier, default_estimator))
    }

    /// [`Self::resolve_id_est_pair`] as typed ids.
    fn resolve_id_est_ids(
        &self,
        default_identifier: IdentifierId,
        default_estimator: EstimatorId,
    ) -> (IdentifierId, EstimatorId) {
        (self.identifier.unwrap_or(default_identifier), self.estimator.unwrap_or(default_estimator))
    }

    /// Resolve builder-selected identifier/estimator ids, applying static-ATE defaults.
    pub(super) fn resolve_static_ids(&self) -> (IdentifierId, EstimatorId) {
        let identifier = self.identifier.unwrap_or(DEFAULT_IDENTIFIER_ID);
        let estimator = if identifier == IdentifierId::GeneralId {
            self.estimator
                .filter(|id| *id != EstimatorId::BayesianGcomp || self.estimator_spec.is_some())
                .unwrap_or(EstimatorId::FunctionalEffect)
        } else if self.inference.is_bayesian() {
            self.estimator.unwrap_or(EstimatorId::BayesianGcomp)
        } else {
            self.estimator.unwrap_or(DEFAULT_ESTIMATOR_ID)
        };
        (identifier, estimator)
    }

    /// Resolve identifier/estimator for PAG ATE (generalized adjustment).
    pub(super) fn resolve_pag_pair(&self) -> (Arc<str>, Arc<str>) {
        wire_pair(self.resolve_pag_ids())
    }

    /// [`Self::resolve_pag_pair`] as typed ids.
    pub(super) fn resolve_pag_ids(&self) -> (IdentifierId, EstimatorId) {
        let estimator = if self.inference.is_bayesian() {
            EstimatorId::BayesianGcomp
        } else {
            DEFAULT_PAG_ESTIMATOR_ID
        };
        self.resolve_id_est_ids(DEFAULT_PAG_IDENTIFIER_ID, estimator)
    }

    /// Incomplete TemporalCpdag/Pag pulse: envelope identifier + temporal linear estimator.
    pub(super) fn resolve_temporal_class_pair(&self) -> (Arc<str>, Arc<str>) {
        self.resolve_id_est_pair(
            DEFAULT_PAG_IDENTIFIER_ID,
            crate::strategy_table::EstimatorId::TemporalLinearAdjustment,
        )
    }

    /// Resolve identifier/estimator for ADMG ATE (general ID + functional effect).
    ///
    /// `CoDetermined` tier-closure is generalized adjustment + AIPW, not `general.id`.
    /// A same-tier bidirected clique must not kick a licensed AIPW study onto the
    /// The ADMG static route's identifier/estimator ids, applying defaults.
    pub(super) fn resolve_admg_ids(&self) -> (IdentifierId, EstimatorId) {
        if self
            .tiered
            .as_ref()
            .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::CoDetermined)
        {
            let default = if self.inference.is_bayesian() {
                EstimatorId::BayesianGcomp
            } else {
                EstimatorId::Aipw
            };
            self.resolve_id_est_ids(IdentifierId::GeneralizedAdjustment, default)
        } else {
            let identifier = self.identifier.unwrap_or(DEFAULT_ADMG_IDENTIFIER_ID);
            let estimator = self.caller_selected_estimator().unwrap_or(DEFAULT_ADMG_ESTIMATOR_ID);
            (identifier, estimator)
        }
    }

    /// Resolve identifier/estimator for `ConditionalEffect`.
    pub(super) fn resolve_conditional_pair(&self) -> (Arc<str>, Arc<str>) {
        wire_pair(self.resolve_conditional_ids())
    }

    /// [`Self::resolve_conditional_pair`] as typed ids.
    pub(super) fn resolve_conditional_ids(&self) -> (IdentifierId, EstimatorId) {
        let estimator = if self.inference.is_bayesian() {
            EstimatorId::BayesianConditional
        } else {
            DEFAULT_CONDITIONAL_ESTIMATOR_ID
        };
        (
            self.identifier.unwrap_or(DEFAULT_CONDITIONAL_IDENTIFIER_ID),
            self.caller_selected_estimator().unwrap_or(estimator),
        )
    }

    /// Resolve identifier/estimator for Distribution queries.
    pub(super) fn resolve_distribution_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.unwrap_or(DEFAULT_DISTRIBUTION_IDENTIFIER_ID);
        let estimator = self
            .estimator
            .filter(|id| *id != EstimatorId::BayesianGcomp || self.estimator_spec.is_some())
            .unwrap_or(DEFAULT_DISTRIBUTION_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve the response estimator from the functional when the caller did not override it.
    pub(super) fn resolve_response_pair(&self, query: &ResponseQuery) -> (Arc<str>, Arc<str>) {
        if self.inference.is_bayesian() && !response_functional_is_derivative(&query.functional) {
            return (
                Arc::from(self.identifier.unwrap_or(DEFAULT_RESPONSE_IDENTIFIER_ID).as_str()),
                Arc::from(
                    self.estimator_spec
                        .as_deref()
                        .map_or(EstimatorId::ResponseBayesian, crate::EstimatorSpec::id)
                        .as_str(),
                ),
            );
        }
        self.resolve_id_est_pair(
            DEFAULT_RESPONSE_IDENTIFIER_ID,
            EstimatorId::default_for_response(&query.functional),
        )
    }

    pub(super) fn resolve_codetermined_joint_pair(&self) -> (Arc<str>, Arc<str>) {
        // The Bayesian joint level runs response.bayesian over the same closure set; a
        // builder placeholder estimator does not override it, a caller selection does.
        if self.inference.is_bayesian() {
            return wire_pair((
                self.identifier.unwrap_or(IdentifierId::GeneralizedAdjustment),
                self.caller_selected_estimator().unwrap_or(EstimatorId::ResponseBayesian),
            ));
        }
        self.resolve_id_est_pair(IdentifierId::GeneralizedAdjustment, EstimatorId::CellAipw)
    }

    /// Resolve identifier/estimator for Cpdag/Pag response (generalized adjustment).
    pub(super) fn resolve_class_response_pair(
        &self,
        query: &ResponseQuery,
    ) -> (Arc<str>, Arc<str>) {
        if self.inference.is_bayesian() && !response_functional_is_derivative(&query.functional) {
            return (
                Arc::from(self.identifier.unwrap_or(DEFAULT_PAG_IDENTIFIER_ID).as_str()),
                Arc::from(
                    self.estimator_spec
                        .as_deref()
                        .map_or(EstimatorId::ResponseBayesian, crate::EstimatorSpec::id)
                        .as_str(),
                ),
            );
        }
        self.resolve_id_est_pair(
            DEFAULT_PAG_IDENTIFIER_ID,
            EstimatorId::default_for_response(&query.functional),
        )
    }

    /// Resolve identifier/estimator for Cpdag/Pag `ConditionalEffect`.
    pub(super) fn resolve_class_conditional_pair(&self) -> (Arc<str>, Arc<str>) {
        wire_pair(self.resolve_class_conditional_ids())
    }

    /// [`Self::resolve_class_conditional_pair`] as typed ids.
    pub(super) fn resolve_class_conditional_ids(&self) -> (IdentifierId, EstimatorId) {
        let estimator = if self.inference.is_bayesian() {
            EstimatorId::BayesianConditional
        } else {
            EstimatorId::ConditionalLinearAdjustment
        };
        (
            self.identifier.unwrap_or(DEFAULT_PAG_IDENTIFIER_ID),
            self.estimator_spec.as_deref().map_or(estimator, crate::EstimatorSpec::id),
        )
    }

    /// Resolve identifier/estimator for `PathSpecific` queries.
    pub(super) fn resolve_path_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.unwrap_or(DEFAULT_PATH_IDENTIFIER_ID);
        let estimator = self
            .estimator
            .filter(|id| *id != EstimatorId::BayesianGcomp || self.estimator_spec.is_some())
            .unwrap_or(DEFAULT_PATH_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    pub(super) fn ensure_rd_config_present(
        &self,
        estimator: EstimatorId,
    ) -> Result<(), CausalError> {
        if matches!(estimator, EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear)
            && self.rd.is_none()
        {
            return Err(CausalError::Compile {
                message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
            });
        }
        Ok(())
    }

    /// Execute a compiled physical plan.
    ///
    /// # Errors
    ///
    /// Identification / estimation / validation failures.
    ///
    /// # Panics
    ///
    /// Never in practice; see [`Study::compile_logical`]'s panics note — the same
    /// `AcceptedGraph::class`/accessor invariant applies here.
    pub fn execute(
        &self,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if crate::analysis::functional_validation::requested(self) {
            return crate::analysis::functional_validation::run_validated(self, ctx);
        }
        self.execute_shared(&std::sync::Arc::new(physical.clone()), None, ctx)
    }

    /// [`Self::execute`] for a plan the caller already holds behind an `Arc`.
    ///
    /// `execution` is the sealed operation of the prepared handle that runs the plan.
    /// `None` is a fresh run: no handle exists, so the route executes without a
    /// retained operation, inside the dispatch scope of the plan's lowered program, and
    /// its result carries no prepared-contract stamp or cached-identification claim.
    pub(crate) fn execute_shared(
        &self,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_on(&self.data, physical, execution, ctx)
    }

    /// Execute against an explicit data slot (prepare/estimate must not clone `Study`).
    pub(crate) fn execute_on(
        &self,
        data: &DataInput,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        // A requested functional validation is run by `functional_validation::run_validated`
        // (the study refitted at validation none); a route reaching here with the suite still
        // set would return an unvalidated result under a validated request.
        if crate::analysis::functional_validation::requested(self) {
            return Err(CausalError::Unsupported {
                message: "cheap and full validation of this query run through the functional \
                          refuter state: execute the study through run, a prepared estimate or \
                          refresh",
            });
        }
        let lowered =
            physical.require_lowered(crate::analysis::lowering::expected_cell(self))?.identity();
        let ctx = &ctx.enter_lowered_dispatch(lowered);
        let mut result = match data {
            DataInput::Tabular(tabular) if crate::analysis::missing_imputation::imputes(self) => {
                crate::analysis::missing_imputation::impute_fit_pool(
                    self,
                    tabular,
                    ctx,
                    |table, ctx| {
                        self.execute_on_inner(
                            &DataInput::Tabular(table.clone()),
                            physical,
                            execution,
                            ctx,
                        )
                    },
                )?
            }
            DataInput::Temporal(series)
                if crate::analysis::missing_imputation::imputes_series(self) =>
            {
                crate::analysis::missing_imputation::impute_fit_pool_series(
                    self,
                    series,
                    ctx,
                    |table, ctx| {
                        self.execute_on_inner(
                            &DataInput::Temporal(table.clone()),
                            physical,
                            execution,
                            ctx,
                        )
                    },
                )?
            }
            _ => self.execute_on_inner(data, physical, execution, ctx)?,
        };
        push_gaussian_likelihood_disclosure(&mut result, &self.inference, data);
        push_lag_zero_average_effect_disclosure(self, &mut result);
        result.custom_validator_names = self
            .custom_validators
            .iter()
            .map(|validator| std::sync::Arc::from(validator.name()))
            .collect();
        result.lowered_identity = Some(lowered);
        crate::analysis::target_validation::stamp_route_refutations(&mut result);
        Ok(result)
    }

    /// Graph-posterior dispatch over data modality, query, and posterior atom kind.
    fn execute_graph_posterior_route(
        &self,
        data: &DataInput,
        gp: &GraphPosterior,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let (DataInput::Temporal(series) | DataInput::Event(series), true) = (
            data,
            is_class_atom(gp.atom_kind)
                && super::temporal_class_point::temporal_class_point_family(&self.query),
        ) {
            return self.execute_temporal_class_graph_posterior_family(series, gp, physical, ctx);
        }
        // A one-modifier conditional effect or a cross-world contrast over `TemporalDag` atoms is
        // answered atom by atom, with the shared interval of the whole procedure.
        if let (DataInput::Temporal(series) | DataInput::Event(series), true) =
            (data, super::dbn_atoms(gp) && super::dbn_route_query(&self.query))
        {
            return self.execute_dbn_route_graph_posterior(series, gp, physical, ctx);
        }
        match (data, &self.query) {
            (DataInput::Tabular(data), _) => {
                self.execute_graph_posterior_tabular(data, gp, physical, ctx)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
            ) => match gp.atom_kind {
                antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                | antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                    self.execute_temporal_class_graph_posterior(data, gp, q, physical, ctx)
                }
                antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                    self.execute_temporal_admg_graph_posterior(data, gp, q, physical, ctx)
                }
                _ => match self.inference {
                    InferenceMode::Frequentist => {
                        self.execute_dbn_posterior_frequentist(data, gp, q, physical, ctx)
                    }
                    InferenceMode::Bayesian(_) => {
                        self.execute_dbn_posterior_bayesian(data, gp, q, physical, ctx)
                    }
                },
            },
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Mediation(q)) => {
                match gp.atom_kind {
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag => self
                        .execute_temporal_class_graph_posterior_mediation(
                            data, gp, q, physical, ctx,
                        ),
                    antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                        self.execute_temporal_pag_graph_posterior_mediation(data, gp, q, physical, ctx)
                    }
                    antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                        self.execute_temporal_admg_graph_posterior_mediation(data, gp, q, physical, ctx)
                    }
                    _ => self.execute_dbn_posterior_mediation(data, gp, q, physical, ctx),
                }
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Response(q)) => {
                match gp.atom_kind {
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                    | antecedent_discovery::GraphPosteriorAtomKind::Pag => self
                        .execute_temporal_class_graph_posterior_response(
                            data, gp, q, physical, ctx,
                        ),
                    _ => self.execute_dbn_posterior_response(data, gp, q, physical, ctx),
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
            ) => self.execute_series_attribution_graph_posterior(data, gp, physical, ctx),
            _ => Err(CausalError::Unsupported { message: GRAPH_POSTERIOR_QUERY_REFUSAL }),
        }
    }

    /// A tabular route on a prepared handle, or on a fresh run whose operation is sealed to the
    /// plan's lowered program here.
    fn execute_tabular_prepared_or_fresh(
        &self,
        route: AnalysisRoute,
        data: &DataInput,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let DataInput::Tabular(data) = data else { return Err(route_mismatch()) };
        // A fresh run holds no prepared handle: its operation is sealed to the
        // plan's lowered program here, and stamps nothing as a prepared click.
        let fresh;
        let execution = if let Some(execution) = execution {
            execution
        } else {
            fresh = super::super::prepared::PreparedExecution::FreshRun(
                super::CheckedRouteOperation::lowered(self, physical)?,
            );
            &fresh
        };
        self.execute_tabular_route(route, data, physical, execution, ctx)
    }

    fn execute_on_inner(
        &self,
        data: &DataInput,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: Option<&super::super::prepared::PreparedExecution>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let Some(gp) = &self.graph_posterior {
            return self.execute_graph_posterior_route(data, gp, physical, ctx);
        }
        if matches!(self.query, CausalQuery::Discover(_)) {
            return self.execute_discover(data, physical, ctx);
        }
        match classify_analysis_route(data, &self.query) {
            Some(route) if matches!(data_modality(data), DataModality::Tabular) => {
                self.execute_tabular_prepared_or_fresh(route, data, physical, execution, ctx)
            }
            Some(AnalysisRoute::TemporalCrossWorld) => {
                let (DataInput::Temporal(data) | DataInput::Event(data)) = data else {
                    return Err(route_mismatch());
                };
                if self.graph.class().is_incomplete_temporal() {
                    return self.execute_temporal_class_cross_world(data, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or_else(|| CausalError::Compile {
                    message: "temporal cross-world plan is missing its TemporalDag".into(),
                })?;
                self.execute_temporal_cross_world(data, graph, physical, ctx)
            }
            Some(AnalysisRoute::TemporalMediation) => {
                let (DataInput::Temporal(data) | DataInput::Event(data)) = data else {
                    return Err(route_mismatch());
                };
                let CausalQuery::Mediation(q) = &self.query else { return Err(route_mismatch()) };
                if self.graph.class() == GraphClass::TemporalCpdag {
                    return self.execute_temporal_cpdag_mediation(data, q, physical, ctx);
                }
                if self.graph.class() == GraphClass::TemporalAdmg {
                    return self.execute_temporal_admg_mediation(data, q, physical, ctx);
                }
                if self.graph.class() == GraphClass::TemporalPag {
                    return self.execute_temporal_pag_mediation(data, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal mediation plan missing resolved graph".into(),
                })?;
                self.execute_temporal_mediation(data, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::TemporalEffect) => {
                let (DataInput::Temporal(data) | DataInput::Event(data)) = data else {
                    return Err(route_mismatch());
                };
                let CausalQuery::TemporalEffect(q) = &self.query else {
                    return Err(route_mismatch());
                };
                if self.graph.class() == GraphClass::TemporalAdmg {
                    return self.execute_temporal_admg_contrast(data, q, physical, ctx);
                }
                if self.graph.class().is_incomplete_temporal() {
                    return self.execute_temporal_class(data, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal plan missing resolved graph".into(),
                })?;
                self.execute_temporal(data, graph, q, physical, ctx, None)
            }
            Some(AnalysisRoute::TemporalResponse) => {
                let (DataInput::Temporal(data) | DataInput::Event(data)) = data else {
                    return Err(route_mismatch());
                };
                let CausalQuery::Response(q) = &self.query else { return Err(route_mismatch()) };
                if self.graph.class() == GraphClass::TemporalAdmg {
                    return self.execute_temporal_admg_derivative(data, q, physical, ctx);
                }
                if self.graph.class().is_incomplete_temporal() {
                    return self.execute_temporal_class_response(data, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal-response plan missing resolved graph".into(),
                })?;
                self.execute_temporal_response(data, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::Association) => {
                let (DataInput::Temporal(series) | DataInput::Event(series)) = data else {
                    return Err(route_mismatch());
                };
                self.execute_association_series(series, physical, ctx)
            }
            Some(AnalysisRoute::PanelTemporalEffect) => {
                self.execute_panel_effect_route(data, physical, ctx)
            }
            Some(AnalysisRoute::PanelTemporalResponse) => {
                self.execute_panel_response_route(data, physical, ctx)
            }
            Some(AnalysisRoute::MultiEnvTemporalEffect) => {
                self.execute_multi_env_effect_route(data, physical, ctx)
            }
            Some(AnalysisRoute::Anomaly | AnalysisRoute::ChangeAttribution)
                if matches!(
                    self.graph.class(),
                    GraphClass::TemporalCpdag | GraphClass::TemporalPag | GraphClass::TemporalAdmg
                ) =>
            {
                let (DataInput::Temporal(series) | DataInput::Event(series)) = data else {
                    return Err(route_mismatch());
                };
                self.execute_temporal_class_attribution(series, physical, ctx)
            }
            Some(
                AnalysisRoute::Anomaly
                | AnalysisRoute::ChangeAttribution
                | AnalysisRoute::Attribute
                | AnalysisRoute::UnitChange
                | AnalysisRoute::MechanismChange
                | AnalysisRoute::Simulate,
            ) if self.graph.class() == GraphClass::TemporalDag => {
                let (DataInput::Temporal(series) | DataInput::Event(series)) = data else {
                    return Err(route_mismatch());
                };
                self.execute_series_attribution(series, physical, ctx)
            }
            _ => Err(CausalError::Unsupported {
                message: "execute path unsupported for this configuration",
            }),
        }
    }

    /// Series anomaly or change attribution on a supplied `TemporalDag`: the sealed
    /// attribution operation over the lag-aligned unfolding, built from the static
    /// identification the plan's lowered program retained (or the prepared cache).
    /// The study identifies here only when neither carries the product.
    fn execute_series_attribution(
        &self,
        series: &TimeSeriesData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let cache = if let Some((cache, _)) = self.reuse_static_identification(physical) {
            cache
        } else {
            super::report_identify_compute(ctx);
            std::sync::Arc::new(self.prepare_static_identification(physical)?.ok_or_else(|| {
                CausalError::Compile {
                    message: "series attribution identified no structural product".into(),
                }
            })?)
        };
        super::CheckedAttributionOperation::checked(self, physical, &cache)?
            .execute_series(series, ctx)
    }

    /// Panel temporal-effect execution: class graphs or the resolved `TemporalDag`.
    fn execute_panel_effect_route(
        &self,
        data: &DataInput,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let DataInput::Panel(panel) = data else { return Err(route_mismatch()) };
        let CausalQuery::TemporalEffect(q) = &self.query else { return Err(route_mismatch()) };
        if self.graph.class().is_incomplete_temporal() {
            return self.execute_panel_class(panel, q, physical, ctx);
        }
        let graph = physical.temporal_graph().ok_or(CausalError::Compile {
            message: "Ready panel plan missing resolved graph".into(),
        })?;
        if matches!(
            self.estimator,
            Some(
                EstimatorId::TemporalGlmLinkContrast | EstimatorId::BayesianTemporalGlmLinkContrast
            )
        ) {
            return self.execute_panel_link(panel, graph, q, physical, ctx);
        }
        self.execute_panel(panel, graph, q, physical, ctx)
    }

    /// Panel temporal-response execution: class graphs or the resolved `TemporalDag`.
    fn execute_panel_response_route(
        &self,
        data: &DataInput,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let DataInput::Panel(panel) = data else { return Err(route_mismatch()) };
        let CausalQuery::Response(q) = &self.query else { return Err(route_mismatch()) };
        if self.graph.class().is_incomplete_temporal() {
            return self.execute_panel_class_response(panel, q, physical, ctx);
        }
        let graph = physical.temporal_graph().ok_or(CausalError::Compile {
            message: "Ready panel-response plan missing resolved graph".into(),
        })?;
        self.execute_panel_response(panel, graph, q, physical, ctx)
    }

    /// Multi-environment temporal-effect execution, which admits only a resolved `TemporalDag`.
    fn execute_multi_env_effect_route(
        &self,
        data: &DataInput,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let DataInput::MultiEnv(multi) = data else { return Err(route_mismatch()) };
        let CausalQuery::TemporalEffect(q) = &self.query else { return Err(route_mismatch()) };
        if self.graph.class().is_incomplete_temporal() {
            return Err(CausalError::Unsupported {
                message: "multi-environment data supports only a supplied TemporalDag",
            });
        }
        let graph = physical.temporal_graph().ok_or(CausalError::Compile {
            message: "Ready multi-env plan missing resolved graph".into(),
        })?;
        self.execute_multi_env_temporal(multi, graph, q, physical, ctx)
    }

    /// Prepared estimate click: tabular data without wrapping a `DataInput`.
    pub(crate) fn execute_tabular(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let lowered =
            physical.require_lowered(crate::analysis::lowering::expected_cell(self))?.identity();
        let ctx = &ctx.enter_lowered_dispatch(lowered);
        let mut result = if let Some(gp) = &self.graph_posterior {
            self.execute_graph_posterior_tabular(data, gp, physical, ctx)?
        } else {
            let route = classify_route(DataModality::Tabular, &self.query).ok_or(
                CausalError::Unsupported {
                    message: "execute path unsupported for this configuration",
                },
            )?;
            self.execute_tabular_route(route, data, physical, execution, ctx)?
        };
        result.lowered_identity = Some(lowered);
        crate::analysis::target_validation::stamp_route_refutations(&mut result);
        Ok(result)
    }

    /// Tabular graph-posterior dispatch shared by fresh and prepared execution.
    fn execute_graph_posterior_tabular(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let query = match &self.query {
            CausalQuery::AverageEffect(q) => q,
            // A latent-confounded conditional effect, a derivative or multi-coordinate
            // response and a counterfactual are answered atom by atom through the explicit
            // executor of each atom's structure.
            CausalQuery::ConditionalEffect(_)
                if matches!(
                    gp.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Admg
                        | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
                ) =>
            {
                return self.execute_atom_route_graph_posterior(data, gp, physical, ctx);
            }
            CausalQuery::Response(q) if super::atom_route_response(q) => {
                return self.execute_atom_route_graph_posterior(data, gp, physical, ctx);
            }
            CausalQuery::Counterfactual(_) => {
                return self.execute_atom_route_graph_posterior(data, gp, physical, ctx);
            }
            CausalQuery::ConditionalEffect(q) => &q.inner,
            CausalQuery::Response(q) => match gp.atom_kind {
                antecedent_discovery::GraphPosteriorAtomKind::Dag => {
                    return self.execute_graph_posterior_response(data, gp, q, physical, ctx);
                }
                antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                    return self.execute_admg_graph_posterior_response(data, gp, q, physical, ctx);
                }
                _ => {
                    return self.execute_class_graph_posterior_response(data, gp, q, physical, ctx);
                }
            },
            CausalQuery::Transport(q) => {
                return self.execute_transport_graph_posterior(data, gp, q, physical, ctx);
            }
            CausalQuery::Interference(q) => {
                return self.execute_interference_graph_posterior(data, gp, q, physical, ctx);
            }
            CausalQuery::Mediation(_)
            | CausalQuery::PathSpecific(_)
            | CausalQuery::NestedCounterfactual(_)
            | CausalQuery::Distribution(_) => {
                return self.execute_cross_world_graph_posterior(data, gp, physical, ctx);
            }
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_) => {
                return self.execute_attribution_graph_posterior(data, gp, physical, ctx);
            }
            _ => return Err(CausalError::Unsupported { message: GRAPH_POSTERIOR_QUERY_REFUSAL }),
        };
        match gp.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Admg
            | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                return self.execute_admg_graph_posterior(data, gp, query, physical, ctx);
            }
            antecedent_discovery::GraphPosteriorAtomKind::Dag => {}
            _ => {
                return self.execute_class_graph_posterior(data, gp, query, physical, ctx);
            }
        }
        match &self.inference {
            InferenceMode::Frequentist => {
                self.execute_graph_posterior_frequentist(data, gp, query, physical, ctx)
            }
            InferenceMode::Bayesian(_) => {
                self.execute_graph_posterior_bayesian(data, gp, query, physical, ctx)
            }
        }
    }

    fn execute_tabular_route(
        &self,
        route: AnalysisRoute,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut result = self.execute_tabular_route_rows(route, data, physical, execution, ctx)?;
        self.report_missing_data(route, data, &mut result)?;
        Ok(result)
    }

    /// The attribution and change routes over a supplied static DAG.
    fn execute_tabular_attribution_route(
        &self,
        route: AnalysisRoute,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match route {
            // A structure that stands for several (a CPDAG, a PAG) or that carries latent
            // confounding (an ADMG) is attributed member by member.
            AnalysisRoute::Anomaly | AnalysisRoute::ChangeAttribution
                if self.graph.as_dag().is_none() =>
            {
                self.execute_class_attribution(data, physical, ctx)
            }
            // The Frequentist mechanism-refit row bootstrap lives on the sealed
            // attribution operation; a requested replicate count on an explicit static
            // DAG runs that operation so the interval does not depend on the entry.
            AnalysisRoute::Anomaly | AnalysisRoute::ChangeAttribution
                if self.structure_source == crate::support::StructureSource::Explicit
                    && self.inference.is_frequentist()
                    && self.bootstrap_replicates > 0
                    && self.graph.as_dag().is_some() =>
            {
                let cache = if let Some((cache, _)) = self.reuse_static_identification(physical) {
                    cache
                } else {
                    super::report_identify_compute(ctx);
                    std::sync::Arc::new(self.prepare_static_identification(physical)?.ok_or_else(
                        || CausalError::Compile {
                            message: "attribution identified no structural product".into(),
                        },
                    )?)
                };
                super::CheckedAttributionOperation::checked(self, physical, &cache)?
                    .execute(data, ctx)
            }
            AnalysisRoute::Anomaly => {
                let CausalQuery::AnomalyAttribution(q) = &self.query else {
                    return Err(route_mismatch());
                };
                let graph = self.require_execute_dag(
                    "AnomalyAttribution execute requires a supplied static DAG",
                )?;
                self.execute_anomaly(data, graph, q, physical, ctx)
            }
            AnalysisRoute::ChangeAttribution => {
                let CausalQuery::ChangeAttribution(q) = &self.query else {
                    return Err(route_mismatch());
                };
                let graph = self.require_execute_dag(
                    "ChangeAttribution execute requires a supplied static DAG",
                )?;
                self.execute_change_attribution(data, graph, q, physical, ctx)
            }
            AnalysisRoute::MechanismChange => {
                let CausalQuery::MechanismChange(q) = &self.query else {
                    return Err(route_mismatch());
                };
                let graph = self.require_execute_dag(
                    "MechanismChange execute requires a supplied static DAG",
                )?;
                // The posterior probability of a change lives on the sealed operation.
                if self.inference.is_bayesian() {
                    let cache = if let Some((cache, _)) = self.reuse_static_identification(physical)
                    {
                        cache
                    } else {
                        super::report_identify_compute(ctx);
                        std::sync::Arc::new(
                            self.prepare_static_identification(physical)?.ok_or_else(|| {
                                CausalError::Compile {
                                    message: "mechanism change identified no structural product"
                                        .into(),
                                }
                            })?,
                        )
                    };
                    return super::CheckedAttributionOperation::checked(self, physical, &cache)?
                        .execute(data, ctx);
                }
                self.execute_mechanism_change(data, graph, q, physical, ctx)
            }
            AnalysisRoute::UnitChange => {
                let CausalQuery::UnitChange(q) = &self.query else { return Err(route_mismatch()) };
                let graph =
                    self.require_execute_dag("UnitChange execute requires a supplied static DAG")?;
                self.execute_unit_change(data, graph, q, physical, ctx)
            }
            _ => Err(route_mismatch()),
        }
    }

    /// An attribution or simulation on tabular data: the sealed attribution operation when the
    /// query carries uncertainty over the mechanisms, else the fitted-model route.
    fn execute_tabular_attribute_or_simulate(
        &self,
        route: AnalysisRoute,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let graph = self.require_execute_dag(
            "Attribute and Simulate execute on a supplied or accepted static DAG",
        )?;
        // The posterior over the mechanisms, and the mechanism-refit row bootstrap of a
        // change target, live on the sealed attribution operation; running it here
        // keeps the uncertainty independent of the entry.
        if matches!(route, AnalysisRoute::Attribute | AnalysisRoute::Simulate)
            && super::attribute_runs_checked(
                &self.query,
                &self.inference,
                self.bootstrap_replicates,
            )
        {
            let cache = if let Some((cache, _)) = self.reuse_static_identification(physical) {
                cache
            } else {
                super::report_identify_compute(ctx);
                std::sync::Arc::new(self.prepare_static_identification(physical)?.ok_or_else(
                    || CausalError::Compile {
                        message: "attribution identified no structural product".into(),
                    },
                )?)
            };
            return super::CheckedAttributionOperation::checked(self, physical, &cache)?
                .execute(data, ctx);
        }
        self.execute_model_route(data, graph, physical, ctx)
    }

    fn execute_tabular_route_rows(
        &self,
        route: AnalysisRoute,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match route {
            AnalysisRoute::TemporalCrossWorld => Err(CausalError::Unsupported {
                message: "temporal cross-world effect requires series data and a TemporalDag",
            }),
            AnalysisRoute::Response => {
                self.execute_tabular_response_route(data, physical, execution, ctx)
            }
            AnalysisRoute::StaticAte => {
                self.execute_tabular_static_ate_route(data, physical, execution, ctx)
            }
            AnalysisRoute::Distribution => {
                self.execute_tabular_distribution_route(data, physical, execution, ctx)
            }
            AnalysisRoute::PathSpecific => {
                let CausalQuery::PathSpecific(q) = &self.query else {
                    return Err(route_mismatch());
                };
                match self.graph.graph_ref() {
                    GraphRef::Admg(admg) => self.execute_path_specific_mixed(
                        data,
                        admg,
                        super::PathSpecificMixedSource::PathSpecific(q),
                        physical,
                        ctx,
                    ),
                    GraphRef::TemporalAdmg(temporal) => {
                        let admg = temporal_admg_projection(temporal)?;
                        self.execute_path_specific_mixed(
                            data,
                            &admg,
                            super::PathSpecificMixedSource::PathSpecific(q),
                            physical,
                            ctx,
                        )
                    }
                    GraphRef::Cpdag(cpdag) => self.execute_class_path_specific(
                        data,
                        cpdag,
                        super::PathSpecificMixedSource::PathSpecific(q),
                        physical,
                        ctx,
                    ),
                    GraphRef::Pag(pag) => self.execute_pag_path_specific(
                        data,
                        pag,
                        super::PathSpecificMixedSource::PathSpecific(q),
                        physical,
                        ctx,
                    ),
                    _ => {
                        let graph = self.require_execute_dag(
                            "PathSpecific execute requires a supplied static Dag, Cpdag, Pag or Admg",
                        )?;
                        self.execute_path_specific(data, graph, q, physical, ctx)
                    }
                }
            }
            AnalysisRoute::Conditional => {
                self.execute_tabular_conditional_route(data, physical, ctx)
            }
            AnalysisRoute::StaticMediation => {
                self.execute_tabular_static_mediation_route(data, physical, execution, ctx)
            }
            AnalysisRoute::Counterfactual => {
                let CausalQuery::Counterfactual(q) = &self.query else {
                    return Err(route_mismatch());
                };
                if matches!(
                    self.graph.graph_ref(),
                    GraphRef::Admg(_)
                        | GraphRef::TemporalAdmg(_)
                        | GraphRef::Pag(_)
                        | GraphRef::Cpdag(_)
                ) {
                    return self.execute_counterfactual_class(data, q, physical, ctx);
                }
                let graph = self
                    .require_execute_dag("Counterfactual execute requires a supplied static DAG")?;
                self.execute_counterfactual(data, graph, q, physical, ctx)
            }
            AnalysisRoute::Anomaly
            | AnalysisRoute::ChangeAttribution
            | AnalysisRoute::MechanismChange
            | AnalysisRoute::UnitChange => {
                self.execute_tabular_attribution_route(route, data, physical, ctx)
            }
            AnalysisRoute::Transport => {
                let CausalQuery::Transport(q) = &self.query else { return Err(route_mismatch()) };
                self.execute_transport(data, q, physical, ctx)
            }
            AnalysisRoute::Interference => {
                self.execute_tabular_interference_route(data, physical, ctx)
            }
            AnalysisRoute::RandomizedEffect => self.execute_randomized(data, physical, ctx),
            AnalysisRoute::PolicyValue => self.execute_policy_value(data, physical, ctx),
            AnalysisRoute::ContinuousDoseResponse => {
                self.execute_continuous_dose_response(data, physical, ctx)
            }
            AnalysisRoute::PanelDid => self.execute_panel_did(data, physical, ctx),
            AnalysisRoute::SyntheticControl => self.execute_synthetic_control(data, physical, ctx),
            AnalysisRoute::LocalPolynomialRatio => {
                self.execute_local_polynomial_ratio(data, physical, ctx)
            }
            AnalysisRoute::Survival => self.execute_survival(data, physical, ctx),
            AnalysisRoute::LongitudinalRegime => {
                self.execute_longitudinal_regime(data, physical, ctx)
            }
            AnalysisRoute::HistoryMatch => self.execute_history_match(data, physical, ctx),
            AnalysisRoute::TemporalDesign => self.execute_temporal_design(data, physical, ctx),
            AnalysisRoute::IvBounds => self.execute_iv_bounds(data, physical, ctx),
            AnalysisRoute::Association => self.execute_association_tabular(data, physical, ctx),
            AnalysisRoute::SequentialContrast => {
                self.execute_sequential_contrast(data, physical, ctx)
            }
            AnalysisRoute::Discover => self.execute_discover_tabular(data, physical, ctx),
            AnalysisRoute::Attribute | AnalysisRoute::Simulate => {
                self.execute_tabular_attribute_or_simulate(route, data, physical, ctx)
            }
            AnalysisRoute::TemporalMediation
            | AnalysisRoute::TemporalEffect
            | AnalysisRoute::TemporalResponse
            | AnalysisRoute::PanelTemporalEffect
            | AnalysisRoute::PanelTemporalResponse
            | AnalysisRoute::MultiEnvTemporalEffect => Err(CausalError::Unsupported {
                message: "execute path unsupported for this configuration",
            }),
        }
    }

    /// Static response execution: which executor runs depends on the graph class and tier closure.
    fn execute_tabular_response_route(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let CausalQuery::Response(q) = &self.query else { return Err(route_mismatch()) };
        match self.graph.graph_ref() {
            GraphRef::Dag(graph) => {
                self.execute_response(data, graph, q, physical, execution.response_curve(), ctx)
            }
            GraphRef::Cpdag(_) | GraphRef::Pag(_) => {
                self.execute_class_response(data, q, physical, ctx)
            }
            GraphRef::Admg(_)
                if self.tiered.as_ref().is_some_and(|b| {
                    b.within_tier == antecedent_graph::WithinTier::CoDetermined
                }) =>
            {
                self.execute_codetermined_joint_response(data, q, physical, ctx)
            }
            GraphRef::Admg(admg) => {
                if admg_has_bidirected(admg) {
                    if response_functional_is_derivative(&q.functional) {
                        self.execute_admg_derivative(data, admg, q, physical, ctx)
                    } else {
                        self.execute_admg_response(data, admg, q, physical, ctx)
                    }
                } else {
                    let graph = physical.static_graph().ok_or(CausalError::Compile {
                        message: "Ready ADMG (DAG-coerced) response plan missing resolved \
                             static DAG"
                            .into(),
                    })?;
                    self.execute_response(data, graph, q, physical, None, ctx)
                }
            }
            GraphRef::TemporalAdmg(temporal) => {
                let admg = temporal_admg_projection(temporal)?;
                if admg_has_bidirected(&admg) {
                    if response_functional_is_derivative(&q.functional) {
                        self.execute_admg_derivative(data, &admg, q, physical, ctx)
                    } else {
                        self.execute_admg_response(data, &admg, q, physical, ctx)
                    }
                } else {
                    let graph = physical.static_graph().ok_or(CausalError::Compile {
                        message: "Ready TemporalAdmg (DAG-coerced) response plan missing \
                                  resolved static DAG"
                            .into(),
                    })?;
                    self.execute_response(data, graph, q, physical, None, ctx)
                }
            }
            _ => Err(CausalError::Unsupported {
                message: "static response execute requires a Dag, Cpdag, Pag, or Admg \
                          (or a CoDetermined tier closure)",
            }),
        }
    }

    /// Static average-effect execution over the graph class, including the tiered-background route.
    fn execute_tabular_static_ate_route(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let CausalQuery::AverageEffect(q) = &self.query else { return Err(route_mismatch()) };
        if let Some(background) = &self.tiered {
            if self.inference.is_bayesian() {
                return self.execute_tiered_bayesian_average(data, q, physical, ctx, background);
            }
            let (identifier, estimator) = physical.binding().resolve(RouteDefaults::STATIC_ATE)?;
            return self.execute_tiered_average(
                data,
                q,
                physical,
                ctx,
                background,
                Strategy::new(identifier, estimator),
            );
        }
        match self.graph.graph_ref() {
            GraphRef::Dag(graph) => self.execute_static(data, graph, q, physical, execution, ctx),
            GraphRef::Cpdag(cpdag) => self.execute_cpdag(data, cpdag, q, physical, ctx),
            GraphRef::Admg(admg) => {
                if admg_has_bidirected(admg) {
                    self.execute_admg(data, admg, q, physical, ctx)
                } else {
                    let graph = physical.static_graph().ok_or(CausalError::Compile {
                        message: "Ready ADMG (DAG-coerced) plan missing resolved static DAG".into(),
                    })?;
                    self.execute_static(data, graph, q, physical, execution, ctx)
                }
            }
            GraphRef::Pag(_) => {
                let pag = physical.static_pag().ok_or(CausalError::Compile {
                    message: "Ready PAG plan missing resolved static PAG".into(),
                })?;
                self.execute_pag(data, pag, q, physical, ctx)
            }
            GraphRef::TemporalAdmg(temporal) => {
                let admg = temporal_admg_projection(temporal)?;
                if admg_has_bidirected(&admg) {
                    self.execute_admg(data, &admg, q, physical, ctx)
                } else {
                    let graph = physical.static_graph().ok_or(CausalError::Compile {
                        message: "Ready TemporalAdmg (DAG-coerced) plan missing resolved \
                                  static DAG"
                            .into(),
                    })?;
                    self.execute_static(data, graph, q, physical, execution, ctx)
                }
            }
            GraphRef::TemporalDag(_) | GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_) => {
                Err(CausalError::Unsupported {
                    message: "static ATE execute requires a static graph class",
                })
            }
            GraphRef::RandomizedTrial => Err(CausalError::Unsupported {
                message: "randomized trial queries require the dedicated experiment execute route",
            }),
        }
    }

    /// Interventional-distribution execution over a Dag or an Admg projection.
    fn execute_tabular_distribution_route(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let CausalQuery::Distribution(q) = &self.query else { return Err(route_mismatch()) };
        match self.graph.graph_ref() {
            GraphRef::Dag(graph) => self.execute_distribution(
                data,
                super::static_path::DistributionGraph::Dag(graph),
                q,
                physical,
                execution.distribution(),
                ctx,
            ),
            GraphRef::Admg(admg) => self.execute_distribution(
                data,
                super::static_path::DistributionGraph::Admg(admg),
                q,
                physical,
                execution.distribution(),
                ctx,
            ),
            GraphRef::TemporalAdmg(temporal) => {
                let admg = temporal_admg_projection(temporal)?;
                self.execute_distribution(
                    data,
                    super::static_path::DistributionGraph::Admg(&admg),
                    q,
                    physical,
                    execution.distribution(),
                    ctx,
                )
            }
            GraphRef::Cpdag(cpdag) => self.execute_class_distribution(
                data,
                super::class_distribution::DistributionClass::Cpdag(cpdag),
                q,
                physical,
                ctx,
            ),
            GraphRef::Pag(pag) => self.execute_class_distribution(
                data,
                super::class_distribution::DistributionClass::Pag(pag),
                q,
                physical,
                ctx,
            ),
            _ => Err(CausalError::Unsupported {
                message: "Distribution execute requires a supplied Dag, Admg, Cpdag or Pag",
            }),
        }
    }

    /// Conditional-effect execution: Dag, the Cpdag / Pag class route, or an Admg.
    fn execute_tabular_conditional_route(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let CausalQuery::ConditionalEffect(q) = &self.query else { return Err(route_mismatch()) };
        match self.graph.class() {
            GraphClass::Dag => {
                let graph = self.require_execute_dag(
                    "ConditionalEffect execute requires a supplied static DAG",
                )?;
                self.execute_conditional(data, graph, q, physical, ctx)
            }
            GraphClass::Cpdag | GraphClass::Pag => {
                self.execute_class_conditional(data, q, physical, ctx)
            }
            GraphClass::Admg => match self.graph.graph_ref() {
                GraphRef::Admg(admg) if admg_has_bidirected(admg) => {
                    self.execute_admg_conditional(data, admg, q, physical, ctx)
                }
                GraphRef::Admg(_) => {
                    let graph = physical.static_graph().ok_or(CausalError::Compile {
                        message: "Ready ADMG (DAG-coerced) conditional plan missing resolved \
                                  static DAG"
                            .into(),
                    })?;
                    self.execute_conditional(data, graph, q, physical, ctx)
                }
                _ => Err(route_mismatch()),
            },
            GraphClass::TemporalAdmg => {
                // The tabular rows carry no time index: the effect is the conditional contrast
                // of the contemporaneous projection (the temporal conditional contrast is
                // TemporalConditionalPulse or TemporalConditionalSustained on one series).
                let GraphRef::TemporalAdmg(temporal) = self.graph.graph_ref() else {
                    return Err(route_mismatch());
                };
                let admg = temporal_admg_projection(temporal)?;
                if admg_has_bidirected(&admg) {
                    self.execute_admg_conditional(data, &admg, q, physical, ctx)
                } else {
                    let graph = physical.static_graph().ok_or(CausalError::Compile {
                        message: "Ready TemporalAdmg (DAG-coerced) conditional plan missing \
                                  resolved static DAG"
                            .into(),
                    })?;
                    self.execute_conditional(data, graph, q, physical, ctx)
                }
            }
            _ => Err(CausalError::Unsupported {
                message: "ConditionalEffect execute requires a Dag, Cpdag, Pag, Admg or \
                          TemporalAdmg",
            }),
        }
    }

    /// Static mediation execution: the total effect, or the nested counterfactual overlaid on it.
    fn execute_tabular_static_mediation_route(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let CausalQuery::NestedCounterfactual(q) = &self.query {
            match self.graph.graph_ref() {
                GraphRef::Admg(admg) => {
                    return self.execute_path_specific_mixed(
                        data,
                        admg,
                        super::PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    );
                }
                GraphRef::TemporalAdmg(temporal) => {
                    let admg = temporal_admg_projection(temporal)?;
                    return self.execute_path_specific_mixed(
                        data,
                        &admg,
                        super::PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    );
                }
                GraphRef::Cpdag(cpdag) => {
                    return self.execute_class_path_specific(
                        data,
                        cpdag,
                        super::PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    );
                }
                GraphRef::Pag(pag) => {
                    return self.execute_pag_path_specific(
                        data,
                        pag,
                        super::PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    );
                }
                _ => {}
            }
        }
        if let CausalQuery::Mediation(q) = &self.query {
            use super::mediation_mixed::ClassStructure;
            match self.graph.graph_ref() {
                GraphRef::Admg(admg) => {
                    return self.execute_mediation_mixed(data, admg, q, physical, ctx);
                }
                GraphRef::TemporalAdmg(temporal) => {
                    let admg = temporal_admg_projection(temporal)?;
                    return self.execute_mediation_mixed(data, &admg, q, physical, ctx);
                }
                GraphRef::Cpdag(cpdag) => {
                    return self.execute_mediation_class(
                        data,
                        ClassStructure::Cpdag(cpdag),
                        q,
                        physical,
                        ctx,
                    );
                }
                GraphRef::Pag(pag) => {
                    return self.execute_mediation_class(
                        data,
                        ClassStructure::Pag(pag),
                        q,
                        physical,
                        ctx,
                    );
                }
                _ => {}
            }
        }
        let graph =
            self.require_execute_dag("static Mediation execute requires a supplied static DAG")?;
        match &self.query {
            CausalQuery::Mediation(q) => {
                self.execute_static_mediation_total(data, graph, q, physical, ctx)
            }
            CausalQuery::NestedCounterfactual(q) => {
                self.execute_nested_counterfactual(data, graph, q, physical, execution, ctx)
            }
            _ => Err(route_mismatch()),
        }
    }

    /// Interference execution: checked operation for the design-based assignments, the Gaussian path otherwise.
    fn execute_tabular_interference_route(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let CausalQuery::Interference(q) = &self.query else { return Err(route_mismatch()) };
        if matches!(
            q.assignment,
            antecedent_core::AssignmentDesign::ClusterRandomization { .. }
                | antecedent_core::AssignmentDesign::TwoStageSaturation { .. }
                | antecedent_core::AssignmentDesign::ObservedExposure { .. }
        ) {
            CheckedInterferenceOperation::checked(self, data, physical)?.execute(data, ctx)
        } else {
            self.execute_interference(data, q, physical, ctx)
        }
    }

    /// Nested counterfactual over the three-node linear mediation DAG: the total-effect run
    /// overwritten with the shared-exogenous natural direct effect.
    fn execute_nested_counterfactual(
        &self,
        data: &TabularData,
        graph: &Dag,
        q: &NestedCounterfactualQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let operation = match execution.nested_counterfactual() {
            Some(operation) if operation.matches(graph, q) => operation.clone(),
            Some(_) => {
                return Err(crate::unsupported_reason!(
                    "cross_world_not_identified",
                    "prepared nested operation does not match the frozen worlds and graph"
                ));
            }
            None => crate::gcm::NestedCounterfactualOperation::compile(graph.clone(), *q)?,
        };
        let mediation = operation.mediation_query();
        let mut result =
            self.execute_static_mediation_total(data, graph, mediation, physical, ctx)?;
        let bayesian = self.inference.is_bayesian();
        // Both inference modes publish the shared-exogenous natural
        // direct effect. That plugin is the outcome-regression slope
        // the contract replays; the mediation posterior mean is not.
        let effect_point = operation.execute(data, ctx)?;
        result.posterior = None;
        if let Some(estimate) = result.estimate.as_effect_mut() {
            estimate.ate = effect_point;
            estimate.uncertainty.analytic.se = f64::NAN;
            estimate.uncertainty.bootstrap.se = None;
        }
        result.interval = None;
        if let Some(components) = result.payload.mediation_mut() {
            components.direct = Some(effect_point);
            if let (Some(total), Some(direct)) = (components.total, components.direct) {
                components.mediated = Some(total - direct);
            }
        }
        result.diagnostics.push(Diagnostic::new(
            "counterfactual.nested.shared_exogenous",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            "both nested treatment worlds used one abduced linear-Gaussian exogenous table"
                .to_string(),
        ));
        if bayesian {
            result.diagnostics.push(Diagnostic::new(
                "counterfactual.nested.posterior_mean_only",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                "published value is the natural direct effect of the outcome regression; posterior draws and the interval are withheld".to_string(),
            ));
        }
        Ok(result)
    }

    /// Compile and run.
    ///
    /// # Errors
    ///
    /// Compile / execute failures.
    pub fn run(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        self.run_observed(ctx).result
    }

    /// Execute with engine-owned work observations, including refusal and cancellation.
    #[must_use]
    pub fn run_observed(
        &self,
        ctx: &ExecutionContext,
    ) -> antecedent_core::execution_attempt::ExecutionAttempt<StudyResult, CausalError> {
        let mut observed = antecedent_core::execution_attempt::observe_with_context(ctx, || {
            antecedent_estimate::prepare::with_preparation_budget(
                crate::planner::PhysicalExecutionPlan::preparation_budget(ctx),
                || self.run_unnamed(ctx),
            )
            .map_err(|error| crate::analysis::preflight::name_run_refusal(self, error, ctx))
        });
        if let Ok(result) = &mut observed.result {
            result.performance.native_execution.clone_from(&observed.facts);
        }
        observed
    }

    fn run_unnamed(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if crate::analysis::functional_validation::requested(self) {
            return crate::analysis::functional_validation::run_validated(self, ctx);
        }
        if let Some(result) = self.run_checked_bayesian_static_mediation(ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_checked_admg_response(ctx)? {
            return Ok(*result);
        }
        self.run_other_routes(ctx)
    }

    /// Prepare the study, require that `retained` sees the sealed checked
    /// operation, then estimate on tabular data. `message` is the compile
    /// error reported when preparation did not retain the operation.
    fn run_sealed_tabular(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
        retained: impl FnOnce(&crate::PreparedStudy) -> bool,
        message: &str,
    ) -> Result<StudyResult, CausalError> {
        let prepared = self.prepare(ctx)?;
        if !retained(&prepared) {
            return Err(CausalError::Compile { message: message.into() });
        }
        prepared.estimate(data, ctx)
    }

    /// Series twin of [`Self::run_sealed_tabular`].
    fn run_sealed_series(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
        retained: impl FnOnce(&crate::PreparedStudy) -> bool,
        message: &str,
    ) -> Result<StudyResult, CausalError> {
        let prepared = self.prepare(ctx)?;
        if !retained(&prepared) {
            return Err(CausalError::Compile { message: message.into() });
        }
        prepared.estimate_series(data, ctx)
    }

    /// Estimator the caller selected, directly or through the estimator spec.
    fn selected_estimator_id(&self) -> Option<EstimatorId> {
        self.estimator.or_else(|| self.estimator_spec.as_deref().map(crate::EstimatorSpec::id))
    }

    /// [`Self::run_sealed_tabular`] boxed as a taken route, for the one-shot admission helpers.
    fn admit_sealed_tabular(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
        retained: impl FnOnce(&crate::PreparedStudy) -> bool,
        message: &str,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        let prepared = self.prepare(ctx)?;
        if !retained(&prepared) {
            // A design over the resident budget streams through a closed-form reduction and
            // retains no operation; the ordinary dispatcher runs it.
            if prepared.plan().streams_design() {
                return Ok(None);
            }
            return Err(CausalError::Compile { message: message.into() });
        }
        prepared.estimate(data, ctx).map(Box::new).map(Some)
    }

    /// Series twin of [`Self::admit_sealed_tabular`].
    fn admit_sealed_series(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
        retained: impl FnOnce(&crate::PreparedStudy) -> bool,
        message: &str,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        self.run_sealed_series(data, ctx, retained, message).map(Box::new).map(Some)
    }

    #[inline(never)]
    fn run_other_routes(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        // Shape is parsed once. Estimator, validation, and structure-source checks
        // stay here: they choose which sealed operation to commit to, and a match
        // that does not retain that operation is still a compile error.
        let route = self.route_input_classified();
        if let Some(result) = self.run_sealed_bayesian_basis_and_robust_effect(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_bayesian_dag_effect(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_mediation_and_temporal_dag(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_frequentist_graph_posterior_effect(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_static_dag_conditional_and_response(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_derivative_counterfactual_and_program(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_aipw(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_glm_rd_propensity(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_class_and_posterior_response(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_cell_aipw_and_bayesian_class_effect(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_graph_posterior_conditional_effect(ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_temporal_class_response_and_posterior(route, ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_sealed_specialist_transport_and_linear(route, ctx)? {
            return Ok(*result);
        }
        let compiled = std::sync::Arc::new(self.compile(ctx)?);
        self.execute_shared(&compiled, None, ctx)
    }

    /// Bayesian basis ATE, robust ATE, and basis CATE one-shot routes that retain a sealed checked operation.
    #[inline(never)]
    fn run_sealed_bayesian_basis_and_robust_effect(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.structure_source == crate::support::StructureSource::Explicit
            && self.inference.is_bayesian()
            && self.estimator == Some(EstimatorId::BayesianBasisGcomp)
            && self
                .identifier
                .is_none_or(|identifier| identifier == IdentifierId::BackdoorAdjustment)
            && self.refute == RefuteSuite::None
        {
            if let Some(RouteInput::StaticAteDag { data, query, .. }) = route {
                if mean_all_observed(&query.outcome_functional, &query.target_population) {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::BayesianBasisAte),
                        "one-shot Bayesian basis ATE did not retain its checked operation",
                    );
                }
            }
        }
        if self.plain_fixed_route()
            && self.structure_source == crate::support::StructureSource::Explicit
            && self.inference.is_bayesian()
            && self.estimator == Some(EstimatorId::BayesianRobustAte)
            && self.identifier.is_none_or(|id| id == IdentifierId::BackdoorAdjustment)
            && self.refute == RefuteSuite::None
        {
            if let Some(RouteInput::StaticAteDag { data, query, .. }) = route {
                if mean_all_observed(&query.outcome_functional, &query.target_population) {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::BayesianRobustAte),
                        "one-shot Bayesian robust ATE did not retain its checked operation",
                    );
                }
            }
        }
        if self.plain_fixed_route()
            && self.structure_source == crate::support::StructureSource::Explicit
            && self.inference.is_bayesian()
            && self.estimator == Some(EstimatorId::BayesianBasisGcomp)
            && self.identifier.is_none_or(|id| id == IdentifierId::BackdoorAdjustment)
            && self.refute == RefuteSuite::None
        {
            if let Some(RouteInput::ConditionalDag { data, query, .. }) = route {
                if mean_all_observed(
                    &query.inner.outcome_functional,
                    &query.inner.target_population,
                ) {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::BayesianBasisCate),
                        "one-shot Bayesian basis CATE did not retain its checked operation",
                    );
                }
            }
        }
        Ok(None)
    }

    /// Bayesian g-computation and Bayesian conditional DAG one-shot routes that retain a sealed checked operation.
    #[inline(never)]
    fn run_sealed_bayesian_dag_effect(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.fixed_structure()
            && self.inference.is_bayesian()
            && self.estimator == Some(EstimatorId::BayesianGcomp)
            && self
                .identifier
                .is_none_or(|identifier| identifier == IdentifierId::BackdoorAdjustment)
        {
            if let Some(RouteInput::StaticAteDag { data, query, .. }) = route {
                if mean_all_observed(&query.outcome_functional, &query.target_population) {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::BayesianGcomp),
                        "one-shot Bayesian DAG effect did not retain its checked operation",
                    );
                }
            }
        }
        if self.plain_fixed_route()
            && self.inference.is_bayesian()
            && self.estimator == Some(EstimatorId::BayesianConditional)
            && self
                .identifier
                .is_none_or(|identifier| identifier == IdentifierId::BackdoorAdjustment)
            && self.point_validation()
        {
            if let Some(RouteInput::ConditionalDag { data, query, .. }) = route {
                if !query.inner.effect_modifiers.is_empty() {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::BayesianConditional),
                        "one-shot Bayesian DAG conditional effect did not retain its checked operation",
                    );
                }
            }
        }
        Ok(None)
    }

    /// Static mediation, temporal DAG effect, and temporal DAG response one-shot routes that retain a sealed checked operation.
    #[inline(never)]
    fn run_sealed_mediation_and_temporal_dag(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route() && self.inference.is_frequentist() {
            if let Some(RouteInput::StaticMediationDag {
                data,
                source: StaticMediationSource::Mediation(_),
                ..
            }) = route
            {
                return self.admit_sealed_tabular(
                    data,
                    ctx,
                    |prepared| prepared.retains(ExecutionKind::StaticMediation),
                    "one-shot mediation did not retain its checked operation",
                );
            }
        }
        if self.plain_fixed_route()
            && self.inference.is_frequentist()
            && self.split.is_none()
            && self.point_validation()
        {
            if let Some(RouteInput::TemporalEffectDag { data, query, .. }) = route {
                if query.policy.is_pulse_or_sustained()
                    && self
                        .estimator
                        .is_none_or(|id| id == EstimatorId::temporal_effect_procedure(query, false))
                {
                    return self.admit_sealed_series(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::TemporalDagEffect),
                        "one-shot temporal DAG effect did not retain its checked operation",
                    );
                }
            }
        }
        if self.plain_fixed_route()
            && self.refute == RefuteSuite::None
            && self
                .estimator
                .is_none_or(|id| id == EstimatorId::temporal_response_for(&self.inference))
        {
            if let Some(RouteInput::TemporalResponseDag { data, query, .. }) = route {
                if query.temporal.is_some()
                    && query.temporal.as_ref().is_some_and(|spec| spec.policy.is_single_step())
                    && complete_mean_response(query)
                    && crate::analysis::temporal_response_is_direct(query)
                {
                    return self.admit_sealed_series(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::TemporalDagResponse),
                        "one-shot temporal DAG response did not retain its checked operation",
                    );
                }
            }
        }
        Ok(None)
    }

    /// Frequentist graph-posterior (DAG and class atom) average-effect one-shot routes.
    #[inline(never)]
    fn run_sealed_frequentist_graph_posterior_effect(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.graph_posterior.as_ref().is_some_and(|posterior| {
            posterior.atom_kind == antecedent_discovery::GraphPosteriorAtomKind::Dag
        }) && self.inference.is_frequentist()
            && self.point_validation()
            && self.custom_validators.is_empty()
            && self.estimator.is_none_or(|id| id == EstimatorId::LinearAdjustmentAte)
        {
            if let Some(RouteInput::StaticAteDag { data, .. }) = route {
                return self.admit_sealed_tabular(
                    data,
                    ctx,
                    |prepared| prepared.retains(ExecutionKind::GraphPosteriorEffect),
                    "one-shot graph-posterior effect did not retain its checked operation",
                );
            }
        }
        if self.graph_posterior.as_ref().is_some_and(|posterior| is_class_atom(posterior.atom_kind))
            && self.inference.is_frequentist()
            && self.estimator.is_none_or(|id| id == EstimatorId::LinearAdjustmentAte)
        {
            if let Some(
                RouteInput::StaticAteCpdag { data, query, .. }
                | RouteInput::StaticAtePag { data, query, .. },
            ) = route
            {
                if Self::graph_posterior_target_is_sealable(self, query) {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::ClassGraphPosteriorEffect),
                        "one-shot class graph-posterior effect did not retain its checked operation",
                    );
                }
            }
        }
        Ok(None)
    }

    /// Frequentist conditional-effect and static DAG response one-shot routes.
    #[inline(never)]
    fn run_sealed_static_dag_conditional_and_response(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.inference.is_frequentist()
            && self.point_validation()
            && self.estimator.is_none_or(|id| id == EstimatorId::ConditionalLinearAdjustment)
        {
            if let Some(RouteInput::ConditionalDag { data, query, .. }) = route {
                if query.inner.effect_modifiers.len() == 1 {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::CheckedConditional),
                        "one-shot conditional effect did not retain its checked operation",
                    );
                }
            }
        }
        if self.plain_fixed_route() {
            if let Some(RouteInput::StaticResponseDag { data, query, .. }) = route {
                if query.temporal.is_none()
                    && complete_mean_response(query)
                    && match (&query.functional, &self.inference) {
                        (
                            antecedent_core::ResponseFunctional::MeanCurve { .. },
                            InferenceMode::Frequentist,
                        ) => {
                            self.refute == RefuteSuite::None
                                && self
                                    .estimator
                                    .is_none_or(|id| id == EstimatorId::ResponseKennedyDr)
                        }
                        (
                            antecedent_core::ResponseFunctional::InterventionResponse { .. },
                            InferenceMode::Frequentist,
                        ) => self
                            .estimator
                            .is_none_or(|id| id == EstimatorId::ResponseInterventionGcomp),
                        (
                            antecedent_core::ResponseFunctional::MeanCurve { .. },
                            InferenceMode::Bayesian(_),
                        ) => {
                            self.refute == RefuteSuite::None
                                && self
                                    .estimator
                                    .is_none_or(|id| id == EstimatorId::ResponseBayesian)
                        }
                        // The Bayesian intervention level seals its cheap/full refuter
                        // state too (it executes unlicensed).
                        (
                            antecedent_core::ResponseFunctional::InterventionResponse { .. },
                            InferenceMode::Bayesian(_),
                        ) => {
                            matches!(
                                self.refute,
                                RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                            ) && self.estimator.is_none_or(|id| id == EstimatorId::ResponseBayesian)
                        }
                        _ => false,
                    }
                {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::StaticDagResponse),
                        "one-shot static DAG response did not retain its checked operation",
                    );
                }
            }
        }
        Ok(None)
    }

    /// Derivative response, counterfactual, and complete-program one-shot routes.
    #[inline(never)]
    fn run_sealed_derivative_counterfactual_and_program(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.inference.is_frequentist()
            && self.refute == RefuteSuite::None
            && self.custom_validators.is_empty()
        {
            if let Some(RouteInput::StaticResponseDag { data, query, .. }) = route {
                if query.temporal.is_none()
                    && complete_mean_response(query)
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::PointDerivative { .. }
                            | antecedent_core::ResponseFunctional::AverageDerivative { .. }
                            | antecedent_core::ResponseFunctional::DirectionalDerivative { .. }
                            | antecedent_core::ResponseFunctional::Jacobian { .. }
                    )
                {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::DerivativeResponse),
                        "one-shot derivative response did not retain its checked operation",
                    );
                }
            }
        }
        if self.graph_posterior.is_none() {
            if let Some(RouteInput::GcmDag { data, route: AnalysisRoute::Counterfactual, .. }) =
                route
            {
                return self.admit_sealed_tabular(
                    data,
                    ctx,
                    |prepared| prepared.retains(ExecutionKind::Counterfactual),
                    "one-shot counterfactual route did not retain its checked operation",
                );
            }
        }
        // These families have a complete retained expression or model operation.
        // The one-shot facade must use that operation too, so it cannot silently
        // recover its meaning from the ordinary Study dispatcher.
        if self.graph_posterior.is_none()
            && matches!(
                self.query,
                CausalQuery::Distribution(_)
                    | CausalQuery::PathSpecific(_)
                    | CausalQuery::NestedCounterfactual(_)
            )
        {
            if let DataInput::Tabular(data) = &self.data {
                // A distribution over an equivalence class is decided member by member at
                // each execution: its checked operation is the lowered route itself.
                let class_distribution = matches!(self.query, CausalQuery::Distribution(_))
                    && matches!(self.graph.graph_ref(), GraphRef::Cpdag(_) | GraphRef::Pag(_));
                // A path-specific or nested effect on a mixed graph or an equivalence
                // class is likewise answered by the lowered route itself.
                let mixed_effect = matches!(
                    self.query,
                    CausalQuery::PathSpecific(_) | CausalQuery::NestedCounterfactual(_)
                ) && matches!(
                    self.graph.graph_ref(),
                    GraphRef::Admg(_)
                        | GraphRef::TemporalAdmg(_)
                        | GraphRef::Cpdag(_)
                        | GraphRef::Pag(_)
                );
                return self.admit_sealed_tabular(
                    data,
                    ctx,
                    |prepared| {
                        prepared.has_complete_program_operation(&self.query)
                            || ((class_distribution || mixed_effect)
                                && prepared.retains(ExecutionKind::EstimatorRoute))
                    },
                    "one-shot route did not retain its complete checked operation",
                );
            }
        }
        Ok(None)
    }

    /// AIPW one-shot route: its checked operation is complete for the binary all-observed mean ATE.
    #[inline(never)]
    fn run_sealed_aipw(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        // AIPW's licensed checked operation is complete for the binary,
        // all-observed mean ATE on a supplied DAG under any diagnosed overlap
        // policy: the trim rule selects the cross-fitted or the common-support
        // procedure inside the retained operation. Route that selection through
        // preparation; nearby estimators execute through the ordinary dispatcher.
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.inference.is_frequentist()
            && self.custom_validators.is_empty()
            && !self
                .estimator_spec
                .as_deref()
                .is_some_and(crate::estimator_spec::EstimatorSpec::excludes_undeclared_levels)
            && matches!(
                super::super::prepared::checked_aipw_fitter(self).overlap,
                OverlapPolicy::RequireDiagnostics { .. }
            )
            && (self.estimator == Some(EstimatorId::Aipw)
                || (self.estimator.is_none()
                    && matches!(
                        self.estimator_spec.as_deref(),
                        Some(
                            crate::estimator_spec::EstimatorSpec::Default(EstimatorId::Aipw)
                                | crate::estimator_spec::EstimatorSpec::Aipw(_),
                        )
                    )))
        {
            if let Some(RouteInput::StaticAteDag { data, query, .. }) = route {
                if mean_all_observed(&query.outcome_functional, &query.target_population)
                    && distinct_numeric_arms(query)
                {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        crate::PreparedStudy::has_sealed_aipw_operation,
                        "one-shot AIPW route did not retain its complete checked operation",
                    );
                }
            }
        }
        Ok(None)
    }

    /// Checked GLM adjustment, sharp-RD, and propensity one-shot routes.
    #[inline(never)]
    fn run_sealed_glm_rd_propensity(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        // Checked GLM adjustment and sharp-RD operations retain the selected
        // target and fitted design through the prepared facade.
        let selected_estimator = self.selected_estimator_id();
        let checked_glm = selected_estimator == Some(EstimatorId::GlmAdjustment)
            && self.plain_fixed_route()
            && self.inference.is_frequentist()
            && matches!(
                route,
                Some(RouteInput::StaticAteDag { query, .. })
                    if matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            );
        let checked_rd = selected_estimator == Some(EstimatorId::RdSharp)
            && self.rd.is_some()
            && self.plain_fixed_route()
            && self.inference.is_frequentist()
            && matches!(route, Some(RouteInput::StaticAteDag { .. }));
        let checked_propensity = matches!(
            selected_estimator,
            Some(EstimatorId::PropensityWeighting | EstimatorId::PropensityMatching)
        ) && self.plain_fixed_route()
            && self.inference.is_frequentist()
            && !self
                .estimator_spec
                .as_deref()
                .is_some_and(crate::estimator_spec::EstimatorSpec::excludes_undeclared_levels)
            && matches!(
                route,
                Some(RouteInput::StaticAteDag { query, .. })
                    if matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                        && query.effect_modifiers.is_empty()
                        && distinct_numeric_arms(query)
            );
        if checked_rd {
            if let Some(RouteInput::StaticAteDag { data, .. }) = route {
                // A sharp-RD design that does not fit its data (assignment not
                // sharp, no rows in the bandwidth window) is not retained; the
                // ordinary dispatch below then reports that refusal at execute,
                // as the route always did.
                let prepared = self.prepare(ctx)?;
                if prepared.retains(ExecutionKind::CheckedRd) {
                    return prepared.estimate(data, ctx).map(Box::new).map(Some);
                }
            }
        } else if checked_glm || checked_propensity {
            if let Some(RouteInput::StaticAteDag { data, .. }) = route {
                return self.admit_sealed_tabular(
                    data,
                    ctx,
                    |prepared| {
                        if checked_glm {
                            prepared.has_sealed_glm_operation()
                        } else {
                            prepared.retains(ExecutionKind::CheckedPropensity)
                        }
                    },
                    "one-shot static estimator route did not retain its checked operation",
                );
            }
        }
        Ok(None)
    }

    /// Static class response and graph-posterior response one-shot routes.
    #[inline(never)]
    fn run_sealed_class_and_posterior_response(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if super::CheckedStaticClassResponse::admits(self) {
            if let Some(
                RouteInput::StaticResponseCpdag { data, .. }
                | RouteInput::StaticResponsePag { data, .. },
            ) = route
            {
                return self.admit_sealed_tabular(
                    data,
                    ctx,
                    |prepared| prepared.retains(ExecutionKind::StaticClassResponse),
                    "one-shot static class response did not retain its checked operation",
                );
            }
        }
        if self.tiered.is_none()
            && self.custom_validators.is_empty()
            && self.graph_posterior.as_ref().is_some_and(|posterior| {
                matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Dag
                        | antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                )
            })
        {
            if let Some(
                RouteInput::StaticResponseDag { data, query, .. }
                | RouteInput::StaticResponseCpdag { data, query, .. }
                | RouteInput::StaticResponsePag { data, query, .. },
            ) = route
            {
                if graph_posterior_response_supported(query).is_ok()
                    && mean_all_observed(&query.outcome_functional, &query.target_population)
                    && match &query.functional {
                        antecedent_core::ResponseFunctional::MeanCurve { .. } => {
                            self.refute == RefuteSuite::None
                        }
                        antecedent_core::ResponseFunctional::InterventionResponse { .. } => {
                            matches!(
                                self.refute,
                                RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                            )
                        }
                        _ => false,
                    }
                    && self.estimator.is_none_or(|id| {
                        id == EstimatorId::static_response_for(&query.functional, &self.inference)
                    })
                {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::GraphPosteriorResponse),
                        "one-shot graph-posterior response did not retain its checked operation",
                    );
                }
            }
        }
        Ok(None)
    }

    /// `CoDetermined` cell-AIPW response and Bayesian static class effect one-shot routes.
    #[inline(never)]
    fn run_sealed_cell_aipw_and_bayesian_class_effect(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.structure_source == crate::support::StructureSource::Explicit
            && self.inference.is_frequentist()
            && self.shared_batch_design.is_none()
            && self.continuous_cell.is_none()
            && self.estimator == Some(EstimatorId::CellAipw)
        {
            if let Some(RouteInput::CodeterminedJoint { data, query, .. }) = route {
                if query.temporal.is_none()
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population == antecedent_core::TargetPopulation::AllObserved
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::InterventionResponse { .. }
                    )
                {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::CellAipwResponse),
                        "one-shot CoDetermined joint cell response did not retain its checked operation",
                    );
                }
            }
        }
        if self.plain_fixed_route()
            && self.inference.is_bayesian()
            && self.identifier.is_none_or(|id| id == IdentifierId::GeneralizedAdjustment)
            && self.estimator.is_none_or(|id| id == EstimatorId::BayesianGcomp)
            && matches!(
                self.estimator_spec.as_deref(),
                None | Some(crate::estimator_spec::EstimatorSpec::Default(
                    EstimatorId::BayesianGcomp
                ))
            )
            && self.point_validation_or_placebo()
        {
            if let Some(
                RouteInput::StaticAteCpdag { data, query, .. }
                | RouteInput::StaticAtePag { data, query, .. },
            ) = route
            {
                if mean_all_observed(&query.outcome_functional, &query.target_population) {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::StaticClassEffect),
                        "one-shot Bayesian static class effect did not retain its checked operation",
                    );
                }
            }
        }
        Ok(None)
    }

    /// Graph-posterior conditional and Bayesian class-posterior effect one-shot route.
    #[inline(never)]
    fn run_sealed_graph_posterior_conditional_effect(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        // Graph-posterior conditional effects on DAG and class atoms, and
        // Bayesian class-posterior effects, execute from their retained
        // checked operations. Frequentist average effects are guarded above.
        if let (Some(posterior), Some(target), DataInput::Tabular(data)) = (
            self.graph_posterior.as_ref(),
            crate::analysis::GraphPosteriorEffectTarget::from_query(&self.query),
            &self.data,
        ) {
            let class_atoms = is_class_atom(posterior.atom_kind);
            let dag_atoms =
                posterior.atom_kind == antecedent_discovery::GraphPosteriorAtomKind::Dag;
            let bayesian = self.inference.is_bayesian();
            let expected =
                if bayesian { target.bayesian_estimator() } else { target.frequentist_estimator() };
            let identifier = if class_atoms {
                IdentifierId::GeneralizedAdjustment
            } else {
                IdentifierId::BackdoorAdjustment
            };
            let query = target.inner();
            if ((class_atoms && (bayesian || target.is_conditional()))
                || (dag_atoms && target.is_conditional()))
                && Self::graph_posterior_target_is_sealable(self, query)
                && Self::graph_posterior_procedure_is(self, expected)
                && self.identifier.is_none_or(|id| id == identifier)
            {
                return self.admit_sealed_tabular(
                    data,
                    ctx,
                    |prepared| {
                        if class_atoms {
                            prepared.retains(ExecutionKind::ClassGraphPosteriorEffect)
                        } else if bayesian {
                            prepared.retains(ExecutionKind::BayesianGraphPosteriorAte)
                        } else {
                            prepared.retains(ExecutionKind::GraphPosteriorEffect)
                        }
                    },
                    "one-shot graph-posterior effect did not retain its checked operation",
                );
            }
        }
        Ok(None)
    }

    /// Temporal class response and temporal class / graph-posterior one-shot routes.
    #[inline(never)]
    fn run_sealed_temporal_class_response_and_posterior(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.refute == RefuteSuite::None
            && self.observation_delayed_entry.is_none()
            && self.identifier.is_none_or(|id| id == IdentifierId::GeneralizedAdjustment)
            && self
                .estimator
                .is_none_or(|id| id == EstimatorId::temporal_response_for(&self.inference))
        {
            if let Some(RouteInput::TemporalResponseClass { data, query, .. }) = route {
                if query.temporal.as_ref().is_some_and(|spec| spec.policy.is_single_step())
                    && complete_mean_response(query)
                    && crate::analysis::temporal_response_is_direct(query)
                {
                    return self.admit_sealed_series(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::TemporalClassResponse),
                        "one-shot temporal class response did not retain its checked operation",
                    );
                }
            }
        }
        if crate::analysis::CheckedTemporalGraphPosteriorResponse::admits(self) {
            if let DataInput::Temporal(data) | DataInput::Event(data) = &self.data {
                return self.admit_sealed_series(
                    data,
                    ctx,
                    |prepared| prepared.retains(ExecutionKind::TemporalGraphPosteriorResponse),
                    "one-shot temporal graph-posterior response did not retain its checked operation",
                );
            }
        }
        if super::CheckedTemporalClassMediationOperation::admits(self) {
            if let DataInput::Temporal(data) | DataInput::Event(data) = &self.data {
                return self.admit_sealed_series(
                    data,
                    ctx,
                    |prepared| prepared.retains(ExecutionKind::TemporalClassMediation),
                    "one-shot temporal class mediation did not retain its checked operation",
                );
            }
        }
        if crate::analysis::CheckedBayesianTemporalClassEffectOperation::admits(self) {
            if let DataInput::Temporal(data) | DataInput::Event(data) = &self.data {
                return self.admit_sealed_series(
                    data,
                    ctx,
                    |prepared| prepared.retains(ExecutionKind::BayesianTemporalClassEffect),
                    "one-shot Bayesian temporal class effect did not retain its checked operation",
                );
            }
        }
        if crate::analysis::CheckedTemporalGraphPosteriorEffect::admits(self) {
            if let DataInput::Temporal(data) | DataInput::Event(data) = &self.data {
                return self.admit_sealed_series(
                    data,
                    ctx,
                    |prepared| prepared.retains(ExecutionKind::TemporalGraphPosteriorEffect),
                    "one-shot temporal graph-posterior effect did not retain its checked operation",
                );
            }
        }
        Ok(None)
    }

    /// Bayesian specialist, transport, and linear-adjustment one-shot routes.
    #[inline(never)]
    fn run_sealed_specialist_transport_and_linear(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        let selected_estimator = self.selected_estimator_id();
        // Bayesian IV and sharp-RD specialists execute from their retained
        // conjugate model; the one-shot facade must not rebuild one.
        let specialist_estimator = matches!(
            selected_estimator,
            Some(
                EstimatorId::BayesianIvJointLinear
                    | EstimatorId::BayesianRdLocalLinear
                    | EstimatorId::BayesianRandomizedQuantile
            )
        );
        if specialist_estimator
            && self.plain_fixed_route()
            && self.split.is_none()
            && self.inference.is_bayesian()
            && self.refute == RefuteSuite::None
        {
            if let Some(RouteInput::StaticAteDag { data, query, .. }) = route {
                if (matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                    || (query.outcome_functional.quantile_level().is_some()
                        && selected_estimator == Some(EstimatorId::BayesianRandomizedQuantile)))
                    && matches!(
                        query.target_population,
                        antecedent_core::TargetPopulation::AllObserved
                            | antecedent_core::TargetPopulation::LocalAtCutoff { .. }
                    )
                {
                    return self.admit_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::BayesianSpecialist),
                        "one-shot Bayesian specialist route did not retain its checked operation",
                    );
                }
            }
        }
        // Certified trial-to-target transport executes from its retained sID
        // proof and design columns even for the one-shot facade.
        if self.selection_diagram.is_some()
            && self.transport_trial.is_some()
            && self.plain_fixed_route()
            && self.split.is_none()
            && self.refute == RefuteSuite::None
        {
            if let Some(RouteInput::TransportMixed {
                data,
                class: GraphClass::Admg | GraphClass::Dag,
                ..
            }) = route
            {
                return self.admit_sealed_tabular(
                    data,
                    ctx,
                    |prepared| prepared.retains(ExecutionKind::TransportTrial),
                    "one-shot transport route did not retain its checked operation",
                );
            }
        }
        // The static mean adjustment route executes from the prepared checked
        // lowering even for the one-shot facade. A configuration no sealed
        // operation admits executes through the ordinary dispatcher below.
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.inference.is_frequentist()
            && self.identifier.is_none_or(|id| id == DEFAULT_IDENTIFIER_ID)
            && matches!(self.estimator, None | Some(EstimatorId::LinearAdjustmentAte))
            && matches!(
                self.estimator_spec.as_deref(),
                None | Some(
                    crate::estimator_spec::EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte,)
                        | crate::estimator_spec::EstimatorSpec::LinearAdjustmentAte(_),
                )
            )
        {
            if let Some(RouteInput::StaticAteDag { data, query, .. }) = route {
                if mean_all_observed(&query.outcome_functional, &query.target_population) {
                    let prepared = self.prepare(ctx)?;
                    // A design over the resident budget streams through the closed-form
                    // reduction and retains no operation; the ordinary dispatcher runs it.
                    if prepared.retains(ExecutionKind::CheckedLinear) {
                        return prepared.estimate(data, ctx).map(Box::new).map(Some);
                    }
                    if !prepared.plan().streams_design() {
                        return Err(CausalError::Compile {
                            message: "one-shot linear adjustment did not retain its checked operation"
                                .into(),
                        });
                    }
                }
            }
        }
        Ok(None)
    }

    #[inline(never)]
    fn run_checked_admg_response(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        let licensed_shape = match &self.query {
            CausalQuery::Response(query)
                if query.temporal.is_none()
                    && !response_functional_is_derivative(&query.functional)
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population
                        == antecedent_core::TargetPopulation::AllObserved
                    && query.outcome_functional == antecedent_core::OutcomeFunctional::Mean =>
            {
                matches!(
                    (&query.functional, self.refute),
                    (_, RefuteSuite::None)
                        | (
                            antecedent_core::ResponseFunctional::InterventionResponse { .. },
                            RefuteSuite::Cheap | RefuteSuite::Full,
                        )
                )
            }
            _ => false,
        };
        if licensed_shape
            && self.plain_fixed_route()
            && self.graph.class() == GraphClass::Admg
            && self.graph.as_admg().is_some_and(admg_has_bidirected)
            && self.identifier.is_none_or(|identifier| identifier == IdentifierId::GeneralId)
            && self.estimator.is_none_or(|estimator| estimator == EstimatorId::FunctionalEffect)
        {
            if let DataInput::Tabular(data) = &self.data {
                return self
                    .run_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::AdmgResponseCurve),
                        "one-shot ADMG response did not retain its checked operation",
                    )
                    .map(Box::new)
                    .map(Some);
            }
        }
        Ok(None)
    }

    fn run_checked_bayesian_static_mediation(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.graph.class() == GraphClass::Dag
            && self.inference.is_bayesian()
            && self.point_validation()
            && matches!(self.query, CausalQuery::Mediation(_))
        {
            if let DataInput::Tabular(data) = &self.data {
                return self
                    .run_sealed_tabular(
                        data,
                        ctx,
                        |prepared| prepared.retains(ExecutionKind::BayesianStaticMediation),
                        "one-shot Bayesian mediation did not retain its checked operation",
                    )
                    .map(Box::new)
                    .map(Some);
            }
        }
        Ok(None)
    }

    /// Identify only (no estimation). Supports static DAG and ADMG average-effect
    /// / related queries.
    ///
    /// The ADMG path matters for correctness, not just coverage: a DAG cannot
    /// express "this variable is unobservable", so a latent common cause
    /// flattened into one is identified by adjusting on a variable no study can
    /// measure. Routing an ADMG through general ID lets that case report
    /// `NotIdentified` instead. Mirrors `execute()`: an ADMG with no bidirected
    /// edges is just a DAG, and is coerced rather than forced down the general
    /// ID path.
    ///
    /// # Errors
    ///
    /// Missing graph structure, unsupported graph class, or identification failure.
    pub fn identify_only(&self) -> Result<IdentificationResult, CausalError> {
        use crate::strategy_table::{
            DEFAULT_IDENTIFIER_ID, DEFAULT_RESPONSE_IDENTIFIER_ID, identify_admg,
            identify_static_query,
        };

        if self.graph_posterior.is_some() {
            // `self.graph` is only the placeholder shape here (see `stub_accepted_graph_for`);
            // identification runs per-graph, against the real posterior atoms, inside `execute()`.
            return Err(CausalError::Support {
                id: crate::support::SupportRefusal::Refused,
                message: "identify_only is not a graph-posterior cell; identification \
                          runs per-graph inside execute.",
            });
        }
        let default_id = if matches!(&self.query, CausalQuery::Response(_)) {
            DEFAULT_RESPONSE_IDENTIFIER_ID
        } else {
            DEFAULT_IDENTIFIER_ID
        };
        let id = self.identifier.unwrap_or(default_id);

        if let Some(admg) = self.graph.as_admg() {
            if admg_has_bidirected(admg) {
                let CausalQuery::AverageEffect(query) = &self.query else {
                    return Err(CausalError::Support {
                        id: crate::support::SupportRefusal::Refused,
                        message: "identify_only on an ADMG supports AverageEffect only.",
                    });
                };
                // Only general ID handles bidirected structure; the default
                // identifier is a backdoor strategy that would ignore it.
                let identifier =
                    if self.identifier.is_some() { id } else { IdentifierId::GeneralId };
                return identify_admg(identifier, admg, query);
            }
            // No bidirected edges: this ADMG *is* a DAG. Coercing keeps the
            // caller's identifier choice meaningful instead of forcing general
            // ID on a graph with no latent structure to reason about.
            let coerced = admg_without_latents_to_dag(admg)?;
            return identify_static_query(id, &coerced, &self.query);
        }

        let graph = self.graph.as_dag().ok_or(CausalError::Support {
            id: crate::support::SupportRefusal::Refused,
            message: "identify_only supports static DAG and ADMG graphs only.",
        })?;
        identify_static_query(id, graph, &self.query)
    }

    /// The identification of this study's query against its structure, for every graph
    /// class: the static and temporal DAG, the ADMG, and the CPDAG and PAG classes (whose
    /// answer is an envelope over completions).
    ///
    /// Identification is a function of structure and query, so the study's rows are never
    /// read: a study over a table with no observations (see
    /// [`StudyBuilder::identify_structure`](crate::StudyBuilder::identify_structure)) answers
    /// exactly as one over a full table. The caller's identifier is honoured; without one the
    /// class default applies.
    ///
    /// # Errors
    ///
    /// A graph-posterior study (identification runs per graph inside `execute`), an
    /// unsupported graph-class/query pair, or identification failure.
    pub fn identify_structure(&self) -> Result<crate::Identification, CausalError> {
        if self.graph_posterior.is_some() {
            return Err(CausalError::Support {
                id: crate::support::SupportRefusal::Refused,
                message: "identify_structure is not a graph-posterior cell; identification \
                          runs per-graph inside execute.",
            });
        }
        crate::identify_api::identify_study(
            &self.graph,
            &self.query,
            self.identifier,
            self.structure_source,
        )
    }

    /// Inspectable compile result (logical + physical).
    ///
    /// Prefer this over `compile` when documenting plan inspection in user code.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compile`].
    pub fn plan(&self, ctx: &ExecutionContext) -> Result<PhysicalExecutionPlan, CausalError> {
        self.compile(ctx)
    }
}

/// Posterior backend of the Bayesian-bootstrap robust ATE.
pub(crate) const ROBUST_ATE_BOOTSTRAP_BACKEND: &str =
    "bayesian.robust_ate.modular_bootstrap_pushforward";
/// Posterior backend of Bayesian anomaly attribution.
pub(crate) const ANOMALY_DIRICHLET_BACKEND: &str = "gcm.attribution.shared_dirichlet_row_weights";
/// Posterior backend of Bayesian change attribution.
pub(crate) const CHANGE_DIRICHLET_BACKEND: &str =
    "gcm.attribution.shared_population_dirichlet_row_weights";
/// Posterior backend of the Bayesian GCM counterfactual.
pub(crate) const COUNTERFACTUAL_DIRICHLET_BACKEND: &str = "gcm.fit.bayesian";
/// Posterior backend of the Bayesian fitted-model attribution (intrinsic influence, change and
/// root causes): the posterior over the mechanisms under Dirichlet row weights.
pub(crate) const ATTRIBUTE_DIRICHLET_BACKEND: &str =
    "gcm.attribution.attribute_dirichlet_row_weights";

/// Posteriors drawn by Dirichlet row reweighting (a Bayesian bootstrap).
/// They fit no outcome likelihood, so no Gaussian fit is disclosed for them.
const BOOTSTRAP_LAW_BACKENDS: [&str; 5] = [
    ROBUST_ATE_BOOTSTRAP_BACKEND,
    ANOMALY_DIRICHLET_BACKEND,
    CHANGE_DIRICHLET_BACKEND,
    COUNTERFACTUAL_DIRICHLET_BACKEND,
    ATTRIBUTE_DIRICHLET_BACKEND,
];

/// Append [`gaussian_likelihood_disclosure`] to `result` once.
///
/// Fresh runs ([`Study::execute_on`]) and prepared clicks (which stamp their
/// contract after bypassing it) both call this, so it is idempotent.
pub(crate) fn push_gaussian_likelihood_disclosure(
    result: &mut StudyResult,
    inference: &InferenceMode,
    data: &DataInput,
) {
    const CODE: &str = "estimate.bayesian.gaussian_likelihood_discrete_outcome";
    if result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == CODE) {
        return;
    }
    if result.posterior.as_ref().is_some_and(|posterior| {
        BOOTSTRAP_LAW_BACKENDS.contains(&posterior.diagnostics.backend_id.as_ref())
    }) {
        return;
    }
    if let Some(disclosure) = gaussian_likelihood_disclosure(inference, data, result.outcome) {
        result.diagnostics.push(disclosure);
    }
}

/// Disclose that a requested `AverageEffect` on a temporal graph was compiled and executed as
/// its lag-0 pulse contrast, and that the result is unlicensed.
pub(crate) fn push_lag_zero_average_effect_disclosure(
    study: &crate::Study,
    result: &mut StudyResult,
) {
    const CODE: &str = "estimate.temporal_class.lag_zero_average_effect";
    let Some(average) = study.lag_zero_average_effect.as_ref() else {
        return;
    };
    if result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == CODE) {
        return;
    }
    let treatment = average.treatment.raw().to_string();
    let outcome = average.outcome.raw().to_string();
    result.diagnostics.push(
        Diagnostic::new(
            CODE,
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the AverageEffect on series data under a temporal graph denotes the lag-0 \
             contrast E[Y_t | do(X_t = active)] - E[Y_t | do(X_t = control)]; it was compiled \
             as the pulse at offset 0 read at horizon 1 and identified on every completion of \
             the graph. The AverageEffect cell on a temporal graph is unlicensed: the result \
             reports support status refused and no coverage record names it"
                .to_owned(),
        )
        .with_fields([
            ("treatment", treatment.as_str()),
            ("outcome", outcome.as_str()),
            ("compiled_as", "PulseEffect(offset=0, horizon_steps=1)"),
        ]),
    );
}

/// Disclose a Gaussian likelihood fitted to a binary or count outcome.
///
/// A Bayesian execution under the Gaussian identity-link model (the default
/// likelihood, and the only one the conjugate backend fits) is reported with
/// `estimate.bayesian.gaussian_likelihood_discrete_outcome` when the outcome is
/// declared binary or count in the schema, or when every observed value is 0/1
/// (`binary`) or a nonnegative integer (`count`). The answer is unchanged; the
/// diagnostic names the likelihood that models the outcome.
fn gaussian_likelihood_disclosure(
    inference: &InferenceMode,
    data: &DataInput,
    outcome: VariableId,
) -> Option<Diagnostic> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return None;
    };
    let gaussian = cfg.likelihood == antecedent_prob::BayesLikelihood::GaussianIdentity
        || cfg.backend == antecedent_estimate::BayesianBackendKind::ConjugateGaussian;
    if !gaussian {
        return None;
    }
    let kind = match data {
        DataInput::Tabular(table) => discrete_outcome_kind([table as &dyn TableView], outcome),
        DataInput::Temporal(series) | DataInput::Event(series) => {
            discrete_outcome_kind([series as &dyn TableView], outcome)
        }
        DataInput::Panel(panel) => {
            let views: Vec<antecedent_data::PanelUnitView<'_>> =
                panel.units().iter().map(antecedent_data::PanelUnitView::new).collect();
            discrete_outcome_kind(views.iter().map(|view| view as &dyn TableView), outcome)
        }
        DataInput::MultiEnv(_) => None,
    }?;
    let (advice, suggested) = if kind == "binary" {
        ("A Bernoulli (logit or probit) likelihood", "bernoulli_logit")
    } else {
        ("A Poisson log-link likelihood", "poisson_log")
    };
    Some(
        Diagnostic::new(
            "estimate.bayesian.gaussian_likelihood_discrete_outcome",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "the outcome is {kind}-valued but the Bayesian model was fitted with a Gaussian \
                 identity-link likelihood, so the posterior describes a linear-Gaussian outcome \
                 model. {advice} models this outcome and is available for a tabular \
                 AverageEffect on a Dag"
            ),
        )
        .with_fields([
            ("outcome_kind", kind),
            ("fitted_likelihood", "gaussian_identity"),
            ("suggested_likelihood", suggested),
        ]),
    )
}

/// `binary` / `count` when the outcome is declared or observed as such.
#[allow(
    clippy::float_cmp,
    reason = "exact comparison is the point: a coded 0/1 outcome, not values near 0 or 1"
)]
fn discrete_outcome_kind<'a>(
    tables: impl IntoIterator<Item = &'a dyn TableView>,
    outcome: VariableId,
) -> Option<&'static str> {
    let mut binary = true;
    let mut count = true;
    let mut seen = false;
    for table in tables {
        match table.schema().get(outcome).ok().map(|variable| &variable.value_type) {
            Some(antecedent_core::ValueType::Binary) => return Some("binary"),
            Some(antecedent_core::ValueType::Count) => return Some("count"),
            // An unspecified type claims nothing, so the observed values decide, as for a
            // declared-continuous variable.
            Some(
                antecedent_core::ValueType::Continuous | antecedent_core::ValueType::Unspecified,
            ) => {}
            _ => return None,
        }
        let values = table.float64_values(outcome).ok()?;
        for value in values.into_iter().filter(|value| value.is_finite()) {
            seen = true;
            binary &= value == 0.0 || value == 1.0;
            count &= value >= 0.0 && value.fract() == 0.0;
            if !binary && !count {
                return None;
            }
        }
    }
    if !seen {
        None
    } else if binary {
        Some("binary")
    } else if count {
        Some("count")
    } else {
        None
    }
}
