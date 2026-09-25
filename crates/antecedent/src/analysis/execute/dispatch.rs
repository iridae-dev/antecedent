// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Refusal for a graph-posterior study whose data / query pair has no route.
pub(super) const GRAPH_POSTERIOR_QUERY_REFUSAL: &str = concat!(
    "graph-posterior analysis supports tabular average-effect, conditional-effect, or static ",
    "response queries, and series temporal-effect, temporal-mediation, or temporal ",
    "response queries only",
);

impl super::Study {
    pub(super) fn validation_suite_id(&self) -> Option<Arc<str>> {
        let family = match self.query {
            CausalQuery::PathSpecific(_) => Some("path"),
            CausalQuery::Distribution(_) => Some("distribution"),
            CausalQuery::Mediation(_) if self.graph.class() == GraphClass::TemporalDag => {
                Some("temporal.mediation")
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
        let class = self.graph.class();
        if let CausalQuery::NestedCounterfactual(q) = &self.query {
            q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
            let Some(graph) = self.graph.as_dag() else {
                return Err(CausalError::Unsupported {
                    message: "cross_world_not_identified: natural direct effect requires the licensed three-node DAG",
                });
            };
            let expected = [
                (q.treatment.raw(), q.mediator.raw()),
                (q.treatment.raw(), q.outcome.raw()),
                (q.mediator.raw(), q.outcome.raw()),
            ];
            let mut observed: Vec<_> =
                graph.edges().map(|edge| (edge.a.raw(), edge.b.raw())).collect();
            observed.sort_unstable();
            let mut expected = expected.to_vec();
            expected.sort_unstable();
            if graph.node_count() != 3 || observed != expected {
                return Err(CausalError::Unsupported {
                    message: "cross_world_not_identified: query requires exactly X -> M, X -> Y, M -> Y with no other nodes or edges",
                });
            }
            if !matches!((&self.data, class), (DataInput::Tabular(_), GraphClass::Dag)) {
                return Err(CausalError::Unsupported {
                    message: "cross_world_not_identified: nested effect requires tabular data and the fixed DAG",
                });
            }
            if self.estimator.is_some_and(|id| id != EstimatorId::StaticMediationLinear)
                || self.identifier.is_some_and(|id| id != IdentifierId::PathSpecificNatural)
                || matches!(self.inference, InferenceMode::Bayesian(_))
                || self.bootstrap_replicates != 0
            {
                return Err(CausalError::Unsupported {
                    message: "cross_world_not_identified: only the point-only linear Gaussian mediation route is licensed for this query",
                });
            }
        }
        if self.estimator == Some(EstimatorId::BayesianBasisGcomp)
            && (!matches!(self.data, DataInput::Tabular(_))
                || class != GraphClass::Dag
                || !matches!(
                    self.query,
                    CausalQuery::AverageEffect(_) | CausalQuery::ConditionalEffect(_)
                )
                || !matches!(self.inference, InferenceMode::Bayesian(_)))
        {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp requires Bayesian inference, tabular data, a supplied DAG, and an ATE or CATE query",
            });
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
                if !matches!(
                    (&self.data, class),
                    (DataInput::Tabular(_), GraphClass::Dag | GraphClass::Admg)
                ) =>
            {
                return Err(CausalError::Unsupported {
                    message:
                        "CausalQuery::Distribution requires tabular data and a static Dag or Admg",
                });
            }
            (_, CausalQuery::PathSpecific(_), class)
                if !matches!((&self.data, class), (DataInput::Tabular(_), GraphClass::Dag)) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::PathSpecific requires tabular data and a static DAG",
                });
            }
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
                let (identifier, _) = self.resolve_pag_pair();
                reject_dag_only_on_pag(&self.graph, identifier.parse::<IdentifierId>()?)?;
            }
            _ => {}
        }
        // The temporal path is linear/temporal-backdoor only; refuse an explicitly
        // selected non-temporal identifier/estimator rather than silently ignoring it.
        if matches!(&self.query, CausalQuery::TemporalEffect(_)) {
            let class_aware = self.graph.class().is_incomplete_temporal();
            if let Some(id) = &self.identifier {
                let ok = if class_aware {
                    *id == IdentifierId::GeneralizedAdjustment
                } else {
                    *id == IdentifierId::TemporalBackdoorUnfolded
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
                    CausalQuery::TemporalEffect(q)
                        if matches!(q.policy, antecedent_core::TemporalPolicy::Sustained { from, until } if from != until)
                );
                let bayesian = matches!(self.inference, crate::InferenceMode::Bayesian(_));
                let ok = *est == EstimatorId::TemporalLinearAdjustment
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
        let identifier = self.identifier.unwrap_or(default_identifier);
        let estimator = self.estimator.unwrap_or(default_estimator);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve builder-selected identifier/estimator ids, applying static-ATE defaults.
    pub(super) fn resolve_static_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.unwrap_or(DEFAULT_IDENTIFIER_ID);
        let estimator = if identifier == IdentifierId::GeneralId {
            self.estimator
                .filter(|id| *id != EstimatorId::BayesianGcomp || self.estimator_spec.is_some())
                .unwrap_or(EstimatorId::FunctionalEffect)
        } else if matches!(self.inference, InferenceMode::Bayesian(_)) {
            self.estimator.unwrap_or(EstimatorId::BayesianGcomp)
        } else {
            self.estimator.unwrap_or(DEFAULT_ESTIMATOR_ID)
        };
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve identifier/estimator for PAG ATE (generalized adjustment).
    pub(super) fn resolve_pag_pair(&self) -> (Arc<str>, Arc<str>) {
        let estimator = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            EstimatorId::BayesianGcomp
        } else {
            DEFAULT_PAG_ESTIMATOR_ID
        };
        self.resolve_id_est_pair(DEFAULT_PAG_IDENTIFIER_ID, estimator)
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
    /// CoDetermined tier-closure is generalized adjustment + AIPW, not `general.id`.
    /// A same-tier bidirected clique must not kick a licensed AIPW study onto the
    /// functional-effect identifier.
    pub(super) fn resolve_admg_pair(&self) -> (Arc<str>, Arc<str>) {
        if self
            .tiered
            .as_ref()
            .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::CoDetermined)
        {
            self.resolve_id_est_pair(IdentifierId::GeneralizedAdjustment, EstimatorId::Aipw)
        } else {
            let identifier = self.identifier.unwrap_or(DEFAULT_ADMG_IDENTIFIER_ID);
            let estimator = self
                .estimator
                .filter(|id| *id != EstimatorId::BayesianGcomp || self.estimator_spec.is_some())
                .unwrap_or(DEFAULT_ADMG_ESTIMATOR_ID);
            (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
        }
    }

    /// Resolve identifier/estimator for ConditionalEffect.
    pub(super) fn resolve_conditional_pair(&self) -> (Arc<str>, Arc<str>) {
        let estimator = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            EstimatorId::BayesianConditional
        } else {
            DEFAULT_CONDITIONAL_ESTIMATOR_ID
        };
        (
            Arc::from(self.identifier.unwrap_or(DEFAULT_CONDITIONAL_IDENTIFIER_ID).as_str()),
            Arc::from(
                self.estimator_spec
                    .as_ref()
                    .map_or(self.estimator.unwrap_or(estimator), crate::EstimatorSpec::id)
                    .as_str(),
            ),
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
        if matches!(self.inference, InferenceMode::Bayesian(_))
            && !response_functional_is_derivative(&query.functional)
        {
            return (
                Arc::from(self.identifier.unwrap_or(DEFAULT_RESPONSE_IDENTIFIER_ID).as_str()),
                Arc::from(
                    self.estimator_spec
                        .as_ref()
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
        self.resolve_id_est_pair(IdentifierId::GeneralizedAdjustment, EstimatorId::CellAipw)
    }

    /// Resolve identifier/estimator for Cpdag/Pag response (generalized adjustment).
    pub(super) fn resolve_class_response_pair(
        &self,
        query: &ResponseQuery,
    ) -> (Arc<str>, Arc<str>) {
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            return (
                Arc::from(self.identifier.unwrap_or(DEFAULT_PAG_IDENTIFIER_ID).as_str()),
                Arc::from(
                    self.estimator_spec
                        .as_ref()
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

    /// Resolve identifier/estimator for Cpdag/Pag ConditionalEffect.
    pub(super) fn resolve_class_conditional_pair(&self) -> (Arc<str>, Arc<str>) {
        let estimator = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            EstimatorId::BayesianConditional
        } else {
            EstimatorId::ConditionalLinearAdjustment
        };
        (
            Arc::from(self.identifier.unwrap_or(DEFAULT_PAG_IDENTIFIER_ID).as_str()),
            Arc::from(
                self.estimator_spec.as_ref().map_or(estimator, crate::EstimatorSpec::id).as_str(),
            ),
        )
    }

    /// Resolve identifier/estimator for PathSpecific queries.
    pub(super) fn resolve_path_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.unwrap_or(DEFAULT_PATH_IDENTIFIER_ID);
        let estimator = self
            .estimator
            .filter(|id| *id != EstimatorId::BayesianGcomp || self.estimator_spec.is_some())
            .unwrap_or(DEFAULT_PATH_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    pub(super) fn ensure_rd_config_present(&self, estimator: &str) -> Result<(), CausalError> {
        if matches!(
            estimator.parse::<EstimatorId>()?,
            EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear
        ) && self.rd.is_none()
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
        self.execute_on(&self.data, physical, ctx)
    }

    /// Execute against an explicit data slot (prepare/estimate must not clone `Study`).
    pub(crate) fn execute_on(
        &self,
        data: &DataInput,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut result = self.execute_on_inner(data, physical, ctx)?;
        push_gaussian_likelihood_disclosure(&mut result, &self.inference, data);
        result.custom_validator_names = self
            .custom_validators
            .iter()
            .map(|validator| std::sync::Arc::from(validator.name()))
            .collect();
        Ok(result)
    }

    fn execute_on_inner(
        &self,
        data: &DataInput,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let Some(gp) = &self.graph_posterior {
            return match (data, &self.query) {
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
                        antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag => self
                            .execute_temporal_class_graph_posterior_mediation(
                                data, gp, q, physical, ctx,
                            ),
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
                _ => Err(CausalError::Unsupported { message: GRAPH_POSTERIOR_QUERY_REFUSAL }),
            };
        }
        match classify_analysis_route(data, &self.query) {
            Some(route) if matches!(data_modality(data), DataModality::Tabular) => {
                let DataInput::Tabular(data) = data else { unreachable!() };
                self.execute_tabular_route(
                    route,
                    data,
                    physical,
                    &super::super::prepared::PreparedExecution::LegacyStudyDispatch,
                    ctx,
                )
            }
            Some(AnalysisRoute::TemporalMediation) => {
                let (DataInput::Temporal(data) | DataInput::Event(data)) = data else {
                    unreachable!()
                };
                let CausalQuery::Mediation(q) = &self.query else { unreachable!() };
                if self.graph.class() == GraphClass::TemporalCpdag {
                    return self.execute_temporal_cpdag_mediation(data, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal mediation plan missing resolved graph".into(),
                })?;
                self.execute_temporal_mediation(data, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::TemporalEffect) => {
                let (DataInput::Temporal(data) | DataInput::Event(data)) = data else {
                    unreachable!()
                };
                let CausalQuery::TemporalEffect(q) = &self.query else { unreachable!() };
                if self.graph.class().is_incomplete_temporal() {
                    return self.execute_temporal_class(data, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal plan missing resolved graph".into(),
                })?;
                self.execute_temporal(data, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::TemporalResponse) => {
                let (DataInput::Temporal(data) | DataInput::Event(data)) = data else {
                    unreachable!()
                };
                let CausalQuery::Response(q) = &self.query else { unreachable!() };
                if self.graph.class().is_incomplete_temporal() {
                    return self.execute_temporal_class_response(data, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal-response plan missing resolved graph".into(),
                })?;
                self.execute_temporal_response(data, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::PanelTemporalEffect) => {
                let DataInput::Panel(panel) = data else { unreachable!() };
                let CausalQuery::TemporalEffect(q) = &self.query else { unreachable!() };
                if self.graph.class().is_incomplete_temporal() {
                    return self.execute_panel_class(panel, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready panel plan missing resolved graph".into(),
                })?;
                self.execute_panel(panel, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::PanelTemporalResponse) => {
                let DataInput::Panel(panel) = data else { unreachable!() };
                let CausalQuery::Response(q) = &self.query else { unreachable!() };
                if self.graph.class().is_incomplete_temporal() {
                    return self.execute_panel_class_response(panel, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready panel-response plan missing resolved graph".into(),
                })?;
                self.execute_panel_response(panel, graph, q, physical, ctx)
            }
            Some(AnalysisRoute::MultiEnvTemporalEffect) => {
                let DataInput::MultiEnv(multi) = data else { unreachable!() };
                let CausalQuery::TemporalEffect(q) = &self.query else { unreachable!() };
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
            _ => Err(CausalError::Unsupported {
                message: "execute path unsupported for this configuration",
            }),
        }
    }

    /// Prepared estimate click: tabular data without wrapping a `DataInput`.
    pub(crate) fn execute_tabular(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let Some(gp) = &self.graph_posterior {
            return self.execute_graph_posterior_tabular(data, gp, physical, ctx);
        }
        let route =
            classify_route(DataModality::Tabular, &self.query).ok_or(CausalError::Unsupported {
                message: "execute path unsupported for this configuration",
            })?;
        self.execute_tabular_route(route, data, physical, execution, ctx)
    }

    /// Tabular graph-posterior dispatch shared by fresh and prepared execution.
    fn execute_graph_posterior_tabular(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let query = match &self.query {
            CausalQuery::AverageEffect(q) => q,
            CausalQuery::ConditionalEffect(q) => &q.inner,
            CausalQuery::Response(q) => match gp.atom_kind {
                antecedent_discovery::GraphPosteriorAtomKind::Dag => {
                    return self.execute_graph_posterior_response(data, gp, q, physical, ctx);
                }
                antecedent_discovery::GraphPosteriorAtomKind::Admg => {
                    return self.execute_admg_graph_posterior_response(data, gp, q, physical, ctx);
                }
                _ => {
                    return self.execute_class_graph_posterior_response(data, gp, q, physical, ctx);
                }
            },
            _ => return Err(CausalError::Unsupported { message: GRAPH_POSTERIOR_QUERY_REFUSAL }),
        };
        match gp.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Admg => {
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
        physical: &PhysicalExecutionPlan,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match route {
            AnalysisRoute::Response => {
                let CausalQuery::Response(q) = &self.query else { unreachable!() };
                match self.graph.class() {
                    GraphClass::Dag => {
                        let graph = self.require_execute_dag(
                            "Response execute requires a supplied static DAG",
                        )?;
                        self.execute_response(
                            data,
                            graph,
                            q,
                            physical,
                            execution.response_curve(),
                            ctx,
                        )
                    }
                    GraphClass::Cpdag | GraphClass::Pag => {
                        self.execute_class_response(data, q, physical, ctx)
                    }
                    GraphClass::Admg
                        if self.tiered.as_ref().is_some_and(|b| {
                            b.within_tier == antecedent_graph::WithinTier::CoDetermined
                        }) =>
                    {
                        self.execute_codetermined_joint_response(data, q, physical, ctx)
                    }
                    GraphClass::Admg => {
                        let admg = self
                            .graph
                            .as_admg()
                            .expect("class() == Admg implies as_admg() is Some");
                        if admg_has_bidirected(admg) {
                            self.execute_admg_response(data, admg, q, physical, ctx)
                        } else {
                            let graph = physical.static_graph().ok_or(CausalError::Compile {
                                message: "Ready ADMG (DAG-coerced) response plan missing resolved \
                                     static DAG"
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
            AnalysisRoute::StaticAte => {
                let CausalQuery::AverageEffect(q) = &self.query else { unreachable!() };
                if let Some(background) = self.tiered.clone() {
                    let identifier =
                        physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
                    let estimator =
                        physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);
                    return self.execute_tiered_average(
                        data,
                        q,
                        physical,
                        ctx,
                        &background,
                        identifier.parse()?,
                        estimator.parse()?,
                    );
                }
                match self.graph.class() {
                    GraphClass::Dag => {
                        let graph =
                            self.graph.as_dag().expect("class() == Dag implies as_dag() is Some");
                        self.execute_static(
                            data,
                            graph,
                            q,
                            physical,
                            execution.linear_operation(),
                            execution.aipw_operation(),
                            execution.frontdoor_linear(),
                            execution.bayesian_gcomp(),
                            execution.iv(),
                            ctx,
                        )
                    }
                    GraphClass::Cpdag => {
                        let cpdag = self
                            .graph
                            .as_cpdag()
                            .expect("class() == Cpdag implies as_cpdag() is Some");
                        self.execute_cpdag(data, cpdag, q, physical, ctx)
                    }
                    GraphClass::Admg => {
                        let admg = self
                            .graph
                            .as_admg()
                            .expect("class() == Admg implies as_admg() is Some");
                        if admg_has_bidirected(admg) {
                            self.execute_admg(data, admg, q, physical, ctx)
                        } else {
                            let graph = physical.static_graph().ok_or(CausalError::Compile {
                                message:
                                    "Ready ADMG (DAG-coerced) plan missing resolved static DAG"
                                        .into(),
                            })?;
                            self.execute_static(
                                data,
                                graph,
                                q,
                                physical,
                                execution.linear_operation(),
                                execution.aipw_operation(),
                                execution.frontdoor_linear(),
                                execution.bayesian_gcomp(),
                                execution.iv(),
                                ctx,
                            )
                        }
                    }
                    GraphClass::Pag => {
                        let pag = physical.static_pag().ok_or(CausalError::Compile {
                            message: "Ready PAG plan missing resolved static PAG".into(),
                        })?;
                        self.execute_pag(data, pag, q, physical, ctx)
                    }
                    GraphClass::TemporalDag
                    | GraphClass::TemporalCpdag
                    | GraphClass::TemporalPag => Err(CausalError::Unsupported {
                        message: "static ATE execute requires a static graph class",
                    }),
                }
            }
            AnalysisRoute::Distribution => {
                let CausalQuery::Distribution(q) = &self.query else { unreachable!() };
                match self.graph.class() {
                    GraphClass::Dag => {
                        let graph = self.require_execute_dag(
                            "Distribution execute requires a supplied static DAG",
                        )?;
                        self.execute_distribution(
                            data,
                            super::static_path::DistributionGraph::Dag(graph),
                            q,
                            physical,
                            execution.distribution(),
                            ctx,
                        )
                    }
                    GraphClass::Admg => {
                        let admg = self.graph.as_admg().ok_or(CausalError::Compile {
                            message: "Distribution ADMG execute missing ADMG".into(),
                        })?;
                        self.execute_distribution(
                            data,
                            super::static_path::DistributionGraph::Admg(admg),
                            q,
                            physical,
                            execution.distribution(),
                            ctx,
                        )
                    }
                    _ => Err(CausalError::Unsupported {
                        message: "Distribution execute requires a supplied Dag or Admg",
                    }),
                }
            }
            AnalysisRoute::PathSpecific => {
                let CausalQuery::PathSpecific(q) = &self.query else { unreachable!() };
                let graph = self
                    .require_execute_dag("PathSpecific execute requires a supplied static DAG")?;
                self.execute_path_specific(data, graph, q, physical, ctx)
            }
            AnalysisRoute::Conditional => {
                let CausalQuery::ConditionalEffect(q) = &self.query else { unreachable!() };
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
                    _ => Err(CausalError::Unsupported {
                        message: "ConditionalEffect execute requires a Dag, Cpdag, or Pag",
                    }),
                }
            }
            AnalysisRoute::StaticMediation => {
                let graph = self.require_execute_dag(
                    "static Mediation execute requires a supplied static DAG",
                )?;
                match &self.query {
                    CausalQuery::Mediation(q) => {
                        self.execute_static_mediation_total(data, graph, q, physical, ctx)
                    }
                    CausalQuery::NestedCounterfactual(q) => {
                        let operation = match execution.nested_counterfactual() {
                            Some(operation) if operation.matches(graph, q) => operation.clone(),
                            Some(_) => {
                                return Err(CausalError::Unsupported {
                                    message: "cross_world_not_identified: prepared nested operation does not match the frozen worlds and graph",
                                });
                            }
                            None => crate::gcm::NestedCounterfactualOperation::compile(
                                graph.clone(),
                                *q,
                            )?,
                        };
                        let mediation = operation.mediation_query();
                        let mut result = self.execute_static_mediation_total(
                            data, graph, mediation, physical, ctx,
                        )?;
                        let effect = operation.execute(data, ctx)?;
                        result.estimate.ate = effect;
                        result.estimate.se_analytic = f64::NAN;
                        result.estimate.se_bootstrap = None;
                        result.interval = None;
                        if let Some(components) = &mut result.mediation {
                            components.direct = Some(effect);
                            if let (Some(total), Some(direct)) =
                                (components.total, components.direct)
                            {
                                components.mediated = Some(total - direct);
                            }
                        }
                        result.diagnostics.push(Diagnostic::new(
                            "counterfactual.nested.shared_exogenous",
                            DiagnosticKind::Execution,
                            DiagnosticSeverity::Info,
                            "both nested treatment worlds used one abduced linear-Gaussian exogenous table".to_string(),
                        ));
                        Ok(result)
                    }
                    _ => unreachable!(),
                }
            }
            AnalysisRoute::Counterfactual => {
                let CausalQuery::Counterfactual(q) = &self.query else { unreachable!() };
                let graph = self
                    .require_execute_dag("Counterfactual execute requires a supplied static DAG")?;
                self.execute_counterfactual(data, graph, q, physical, ctx)
            }
            AnalysisRoute::Anomaly => {
                let CausalQuery::AnomalyAttribution(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag(
                    "AnomalyAttribution execute requires a supplied static DAG",
                )?;
                self.execute_anomaly(data, graph, q, physical, ctx)
            }
            AnalysisRoute::ChangeAttribution => {
                let CausalQuery::ChangeAttribution(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag(
                    "ChangeAttribution execute requires a supplied static DAG",
                )?;
                self.execute_change_attribution(data, graph, q, physical, ctx)
            }
            AnalysisRoute::MechanismChange => {
                let CausalQuery::MechanismChange(q) = &self.query else { unreachable!() };
                let graph = self.require_execute_dag(
                    "MechanismChange execute requires a supplied static DAG",
                )?;
                self.execute_mechanism_change(data, graph, q, physical, ctx)
            }
            AnalysisRoute::UnitChange => {
                let CausalQuery::UnitChange(q) = &self.query else { unreachable!() };
                let graph =
                    self.require_execute_dag("UnitChange execute requires a supplied static DAG")?;
                self.execute_unit_change(data, graph, q, physical, ctx)
            }
            AnalysisRoute::Transport => {
                let CausalQuery::Transport(q) = &self.query else { unreachable!() };
                self.execute_transport(data, q, physical, ctx)
            }
            AnalysisRoute::Interference => {
                let CausalQuery::Interference(q) = &self.query else { unreachable!() };
                self.execute_interference(data, q, physical, ctx)
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

    /// Compile and run.
    ///
    /// # Errors
    ///
    /// Compile / execute failures.
    pub fn run(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        if let Some(result) = self.run_checked_bayesian_static_mediation(ctx)? {
            return Ok(*result);
        }
        if let Some(result) = self.run_checked_admg_response(ctx)? {
            return Ok(*result);
        }
        self.run_other_routes(ctx)
    }

    #[inline(never)]
    fn run_other_routes(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Bayesian(_))
            && self.estimator == Some(EstimatorId::BayesianGcomp)
            && self
                .identifier
                .is_none_or(|identifier| identifier == IdentifierId::BackdoorAdjustment)
            && matches!(
                &self.query,
                CausalQuery::AverageEffect(query)
                    if matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                        && query.target_population == antecedent_core::TargetPopulation::AllObserved
            )
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if prepared.checked_bayesian_gcomp_operation().is_none() {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot Bayesian DAG effect did not retain its checked operation"
                                .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Bayesian(_))
            && self.estimator == Some(EstimatorId::BayesianConditional)
            && self
                .identifier
                .is_none_or(|identifier| identifier == IdentifierId::BackdoorAdjustment)
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validators.is_empty()
            && matches!(&self.query, CausalQuery::ConditionalEffect(query)
                if !query.inner.effect_modifiers.is_empty())
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if prepared.checked_bayesian_conditional_operation().is_none() {
                    return Err(CausalError::Compile {
                        message: "one-shot Bayesian DAG conditional effect did not retain its checked operation".into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.custom_validators.is_empty()
            && matches!(self.query, CausalQuery::Mediation(_))
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if prepared.checked_static_mediation_info().is_none() {
                    return Err(CausalError::Compile {
                        message: "one-shot mediation did not retain its checked operation".into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::TemporalDag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.split.is_none()
            && self.custom_validators.is_empty()
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && matches!(&self.query, CausalQuery::TemporalEffect(query)
                if matches!(&query.policy,
                    antecedent_core::TemporalPolicy::Pulse { .. }
                    | antecedent_core::TemporalPolicy::Sustained { .. }))
            && self.estimator.is_none_or(|id| match &self.query {
                CausalQuery::TemporalEffect(query) if query.is_multi_step_sustained() => {
                    id == EstimatorId::TemporalSequentialGcomp
                }
                _ => id == EstimatorId::TemporalLinearAdjustment,
            })
        {
            if let DataInput::Temporal(data) | DataInput::Event(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if !prepared.has_checked_temporal_dag_effect_operation() {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot temporal DAG effect did not retain its checked operation"
                                .into(),
                    });
                }
                return prepared.estimate_series(data, ctx);
            }
        }
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::TemporalDag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.refute == RefuteSuite::None
            && self.custom_validators.is_empty()
            && matches!(&self.query, CausalQuery::Response(query)
                if query.temporal.is_some()
                    && query.temporal.as_ref().is_some_and(|spec| match &spec.policy {
                        antecedent_core::TemporalPolicy::Pulse { .. } => true,
                        antecedent_core::TemporalPolicy::Sustained { from, until } => from == until,
                        antecedent_core::TemporalPolicy::Dynamic { .. } => false,
                        _ => false,
                    })
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population == antecedent_core::TargetPopulation::AllObserved
                    && matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                    && matches!(query.functional, antecedent_core::ResponseFunctional::MeanCurve { .. }))
            && self.estimator.is_none_or(|id| id == EstimatorId::TemporalResponseGcomp)
        {
            if let DataInput::Temporal(data) | DataInput::Event(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if !prepared.has_checked_temporal_dag_response_operation() {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot temporal DAG response did not retain its checked operation"
                                .into(),
                    });
                }
                return prepared.estimate_series(data, ctx);
            }
        }
        if self.graph_posterior.as_ref().is_some_and(|posterior| {
            posterior.atom_kind == antecedent_discovery::GraphPosteriorAtomKind::Dag
        }) && matches!(self.inference, InferenceMode::Frequentist)
            && matches!(self.query, CausalQuery::AverageEffect(_))
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validators.is_empty()
            && self.estimator.is_none_or(|id| id == EstimatorId::LinearAdjustmentAte)
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if !prepared.has_checked_graph_posterior_effect_operation() {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot graph-posterior effect did not retain its checked operation"
                                .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        if self.graph_posterior.as_ref().is_some_and(|posterior| {
            matches!(
                posterior.atom_kind,
                antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                    | antecedent_discovery::GraphPosteriorAtomKind::Pag
            )
        }) && matches!(self.inference, InferenceMode::Frequentist)
            && matches!(&self.query, CausalQuery::AverageEffect(query)
                if matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                    && query.target_population == antecedent_core::TargetPopulation::AllObserved)
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validators.is_empty()
            && self.estimator.is_none_or(|id| id == EstimatorId::LinearAdjustmentAte)
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if !prepared.has_checked_class_graph_posterior_effect_operation() {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot class graph-posterior effect did not retain its checked operation"
                                .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Frequentist)
            && matches!(&self.query, CausalQuery::ConditionalEffect(query)
                if query.inner.effect_modifiers.len() == 1)
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validators.is_empty()
            && self.estimator.is_none_or(|id| id == EstimatorId::ConditionalLinearAdjustment)
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if prepared.checked_conditional_effect_info().is_none() {
                    return Err(CausalError::Compile {
                        message: "one-shot conditional effect did not retain its checked operation"
                            .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.custom_validators.is_empty()
            && matches!(&self.query, CausalQuery::Response(query)
                if query.temporal.is_none()
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population == antecedent_core::TargetPopulation::AllObserved
                    && matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                    && (matches!(query.functional, antecedent_core::ResponseFunctional::MeanCurve { .. })
                        && self.refute == RefuteSuite::None
                        && self.estimator.is_none_or(|id| id == EstimatorId::ResponseKennedyDr)
                        || matches!(query.functional, antecedent_core::ResponseFunctional::InterventionResponse { .. })
                            && self.estimator.is_none_or(|id| id == EstimatorId::ResponseInterventionGcomp)))
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if !prepared.has_checked_static_dag_response_operation() {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot static DAG response did not retain its checked operation"
                                .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.refute == RefuteSuite::None
            && self.custom_validators.is_empty()
            && matches!(&self.query, CausalQuery::Response(query)
                if query.temporal.is_none()
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population == antecedent_core::TargetPopulation::AllObserved
                    && matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                    && matches!(query.functional,
                        antecedent_core::ResponseFunctional::PointDerivative { .. }
                        | antecedent_core::ResponseFunctional::AverageDerivative { .. }
                        | antecedent_core::ResponseFunctional::DirectionalDerivative { .. }
                        | antecedent_core::ResponseFunctional::Jacobian { .. }))
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if !prepared.has_checked_derivative_response_operation() {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot derivative response did not retain its checked operation"
                                .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        if self.graph_posterior.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(self.query, CausalQuery::Counterfactual(_))
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if !prepared.has_checked_counterfactual_operation() {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot counterfactual route did not retain its checked operation"
                                .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        // These families have a complete retained expression or model operation.
        // The one-shot facade must use that operation too, so it cannot silently
        // recover its meaning from the ordinary Study dispatcher.
        if matches!(
            self.query,
            CausalQuery::Distribution(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::NestedCounterfactual(_)
        ) {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if !prepared.has_complete_program_operation(&self.query) {
                    return Err(CausalError::Compile {
                        message: "one-shot route did not retain its complete checked operation"
                            .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        // AIPW's licensed checked operation is complete for the binary,
        // all-observed mean ATE on a supplied DAG. Route only that exact
        // selection through preparation; nearby estimators keep legacy dispatch.
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.custom_validators.is_empty()
            && (self.estimator == Some(EstimatorId::Aipw)
                || (self.estimator.is_none()
                    && matches!(
                        &self.estimator_spec,
                        Some(crate::estimator_spec::EstimatorSpec::Default(EstimatorId::Aipw))
                            | Some(crate::estimator_spec::EstimatorSpec::Aipw(_))
                    )))
            && matches!(
                &self.query,
                CausalQuery::AverageEffect(query)
                    if matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                        && matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
                        && matches!(&query.active, antecedent_core::Intervention::Set { variable, value }
                            if *variable == query.treatment && value.as_f64() == Some(1.0))
                        && matches!(&query.control, antecedent_core::Intervention::Set { variable, value }
                            if *variable == query.treatment && value.as_f64() == Some(0.0))
            )
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if !prepared.has_sealed_aipw_operation() {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot AIPW route did not retain its complete checked operation"
                                .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        // Checked GLM adjustment and sharp-RD operations retain the selected
        // target and fitted design through the prepared facade.
        let selected_estimator =
            self.estimator.or_else(|| self.estimator_spec.as_ref().map(crate::EstimatorSpec::id));
        let checked_glm = selected_estimator == Some(EstimatorId::GlmAdjustment)
            && self.graph.class() == GraphClass::Dag
            && self.graph_posterior.is_none()
            && self.tiered.is_none()
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.custom_validators.is_empty()
            && matches!(
                &self.query,
                CausalQuery::AverageEffect(query)
                    if matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            );
        let checked_rd = selected_estimator == Some(EstimatorId::RdSharp)
            && self.rd.is_some()
            && self.graph.class() == GraphClass::Dag
            && self.graph_posterior.is_none()
            && self.tiered.is_none()
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.custom_validators.is_empty()
            && matches!(&self.query, CausalQuery::AverageEffect(_));
        let checked_propensity = matches!(
            selected_estimator,
            Some(EstimatorId::PropensityWeighting | EstimatorId::PropensityMatching)
        ) && self.graph.class() == GraphClass::Dag
            && self.graph_posterior.is_none()
            && self.tiered.is_none()
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.custom_validators.is_empty()
            && matches!(&self.query, CausalQuery::AverageEffect(query)
                if matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                    && query.effect_modifiers.is_empty()
                    && matches!(&query.active, antecedent_core::Intervention::Set { variable, value }
                        if *variable == query.treatment && value.as_f64() == Some(1.0))
                    && matches!(&query.control, antecedent_core::Intervention::Set { variable, value }
                        if *variable == query.treatment && value.as_f64() == Some(0.0)));
        if checked_glm || checked_rd || checked_propensity {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                let selected = if checked_glm {
                    prepared.has_sealed_glm_operation()
                } else if checked_rd {
                    prepared.has_checked_rd_operation()
                } else {
                    prepared.has_checked_propensity_operation()
                };
                if !selected {
                    return Err(CausalError::Compile {
                        message:
                            "one-shot static estimator route did not retain its checked operation"
                                .into(),
                    });
                }
                return prepared.estimate(data, ctx);
            }
        }
        // The migrated static mean adjustment route executes from the prepared
        // checked lowering even for the one-shot facade. Other routes retain
        // their legacy dispatch until their own lowering checkpoint lands.
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(self.inference, InferenceMode::Frequentist)
            && matches!(
                &self.query,
                CausalQuery::AverageEffect(query)
                    if matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
                        && matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
            )
            && matches!(self.estimator, None | Some(EstimatorId::LinearAdjustmentAte))
            && matches!(
                &self.estimator_spec,
                None | Some(crate::estimator_spec::EstimatorSpec::Default(
                    EstimatorId::LinearAdjustmentAte
                )) | Some(crate::estimator_spec::EstimatorSpec::LinearAdjustmentAte(_))
            )
        {
            if let DataInput::Tabular(data) = &self.data {
                return self.prepare(ctx)?.estimate(data, ctx);
            }
        }
        let compiled = self.compile(ctx)?;
        self.execute(&compiled, ctx)
    }

    #[inline(never)]
    fn run_checked_admg_response(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        let licensed_shape = match &self.query {
            CausalQuery::Response(query)
                if query.temporal.is_none()
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population
                        == antecedent_core::TargetPopulation::AllObserved
                    && query.outcome_functional == antecedent_core::OutcomeFunctional::Mean =>
            {
                match (&query.functional, self.refute) {
                    (_, RefuteSuite::None) => true,
                    (
                        antecedent_core::ResponseFunctional::InterventionResponse { .. },
                        RefuteSuite::Cheap | RefuteSuite::Full,
                    ) => true,
                    _ => false,
                }
            }
            _ => false,
        };
        if licensed_shape
            && self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Admg
            && self.graph.as_admg().is_some_and(admg_has_bidirected)
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Frequentist | InferenceMode::Bayesian(_))
            && self.custom_validators.is_empty()
            && self.identifier.is_none_or(|identifier| identifier == IdentifierId::GeneralId)
            && self.estimator.is_none_or(|estimator| estimator == EstimatorId::FunctionalEffect)
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if prepared.checked_functional_effect_response_members().is_none() {
                    return Err(CausalError::Compile {
                        message: "one-shot ADMG response did not retain its checked operation"
                            .into(),
                    });
                }
                return prepared.estimate(data, ctx).map(Box::new).map(Some);
            }
        }
        Ok(None)
    }

    fn run_checked_bayesian_static_mediation(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.graph.class() == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && matches!(self.inference, InferenceMode::Bayesian(_))
            && self.custom_validators.is_empty()
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && matches!(self.query, CausalQuery::Mediation(_))
        {
            if let DataInput::Tabular(data) = &self.data {
                let prepared = self.prepare(ctx)?;
                if prepared.checked_bayesian_static_mediation_info().is_none() {
                    return Err(CausalError::Compile {
                        message: "one-shot Bayesian mediation did not retain its checked operation"
                            .into(),
                    });
                }
                return prepared.estimate(data, ctx).map(Box::new).map(Some);
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
            identify_admg, identify_static_query, DEFAULT_IDENTIFIER_ID,
            DEFAULT_RESPONSE_IDENTIFIER_ID,
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
    if let Some(disclosure) = gaussian_likelihood_disclosure(inference, data, result.outcome) {
        result.diagnostics.push(disclosure);
    }
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
