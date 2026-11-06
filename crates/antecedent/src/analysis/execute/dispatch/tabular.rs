//! Tabular execution route dispatch.

use super::{
    AnalysisRoute, CausalError, CausalQuery, CheckedInterferenceOperation, Dag, DataInput,
    DataModality, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext,
    GRAPH_POSTERIOR_QUERY_REFUSAL, GraphClass, GraphPosterior, GraphRef, InferenceMode,
    NestedCounterfactualQuery, PhysicalExecutionPlan, ResponseQuery, RouteDefaults,
    StaticMediationSource, Strategy, StudyResult, TabularCall, TabularData, admg_has_bidirected,
    classify_route, response_functional_is_derivative, route_mismatch, route_unsupported,
    temporal_admg_projection,
};

impl super::super::Study {
    /// Prepared estimate click: tabular data without wrapping a `DataInput`.
    pub(crate) fn execute_tabular(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unread_sampling_weights_for_route(data)?;
        let lowered =
            physical.require_lowered(crate::analysis::lowering::expected_cell(self))?.identity();
        let ctx = &ctx.enter_lowered_dispatch(lowered);
        let mut result = if let Some(gp) = &self.graph_posterior {
            self.execute_graph_posterior_tabular(data, gp, physical, ctx)?
        } else {
            let route =
                classify_route(DataModality::Tabular, &self.query).ok_or_else(route_unsupported)?;
            let call = TabularCall::of(route, &self.query)?;
            self.execute_tabular_route(call, data, physical, execution, ctx)?
        };
        result.stamp_lowered_program(
            physical.require_lowered(crate::analysis::lowering::expected_cell(self))?,
        );
        result.executed_input = Some(crate::analysis::target_validation::executed_input_identity(
            &DataInput::Tabular(data.clone()),
            ctx,
        )?);
        crate::analysis::target_validation::stamp_route_refutations(&mut result);
        Ok(result)
    }

    /// Tabular graph-posterior dispatch shared by fresh and prepared execution.
    pub(super) fn execute_graph_posterior_tabular(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let result = self.execute_graph_posterior_tabular_unrefuted(data, gp, physical, ctx)?;
        let CausalQuery::Response(q) = &self.query else { return Ok(result) };
        if gp.atom_kind() != antecedent_discovery::GraphPosteriorAtomKind::Dag {
            return Ok(result);
        }
        let Some(treatment) = super::super::curve_refutation::curve_treatment(q) else {
            return Ok(result);
        };
        let placebo = super::super::curve_refutation::tabular_placebo(treatment, ctx);
        let subset = super::super::curve_refutation::tabular_subset(ctx);
        self.attach_curve_refutation(
            result,
            data,
            q,
            (&placebo, &subset),
            &|study, perturbed, rctx| {
                study.execute_graph_posterior_tabular_unrefuted(perturbed, gp, physical, rctx)
            },
            ctx,
        )
    }

    pub(super) fn execute_graph_posterior_tabular_unrefuted(
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
                    gp.atom_kind(),
                    antecedent_discovery::GraphPosteriorAtomKind::Admg
                        | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
                ) =>
            {
                return self.execute_atom_route_graph_posterior(data, gp, physical, ctx);
            }
            CausalQuery::Response(q) if super::super::atom_route_response(q) => {
                return self.execute_atom_route_graph_posterior(data, gp, physical, ctx);
            }
            CausalQuery::Counterfactual(_) => {
                return self.execute_atom_route_graph_posterior(data, gp, physical, ctx);
            }
            CausalQuery::ConditionalEffect(q) => &q.inner,
            CausalQuery::Response(q) => match gp.atom_kind() {
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
        match gp.atom_kind() {
            antecedent_discovery::GraphPosteriorAtomKind::Admg
            | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                return self.execute_admg_graph_posterior(data, gp, query, physical, ctx);
            }
            antecedent_discovery::GraphPosteriorAtomKind::Dag => {}
            _ => {
                return self.execute_class_graph_posterior(data, gp, query, physical, ctx);
            }
        }
        match &self.settings.inference {
            InferenceMode::Frequentist => {
                self.execute_graph_posterior_frequentist(data, gp, query, physical, ctx)
            }
            InferenceMode::Bayesian(_) => {
                self.execute_graph_posterior_bayesian(data, gp, query, physical, ctx)
            }
        }
    }

    pub(super) fn execute_tabular_route(
        &self,
        call: TabularCall<'_>,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut result = self.execute_tabular_route_rows(call, data, physical, execution, ctx)?;
        self.report_missing_data(call.route(), data, &mut result)?;
        Ok(result)
    }

    /// The attribution and change routes over a supplied static DAG.
    pub(super) fn execute_tabular_attribution_route(
        &self,
        call: TabularCall<'_>,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match call {
            // A structure that stands for several (a CPDAG, a PAG) or that carries latent
            // confounding (an ADMG) is attributed member by member.
            TabularCall::Anomaly(_) | TabularCall::ChangeAttribution(_)
                if self.graph.as_dag().is_none() =>
            {
                self.execute_class_attribution(data, physical, ctx)
            }
            // The Frequentist mechanism-refit row bootstrap lives on the sealed
            // attribution operation; a requested replicate count on an explicit static
            // DAG runs that operation so the interval does not depend on the entry.
            TabularCall::Anomaly(_) | TabularCall::ChangeAttribution(_)
                if self.structure_source == crate::support::StructureSource::Explicit
                    && self.settings.inference.is_frequentist()
                    && self.settings.bootstrap_replicates > 0
                    && self.graph.as_dag().is_some() =>
            {
                let cache = if let Some((cache, _)) = self.reuse_static_identification(physical) {
                    cache
                } else {
                    super::super::report_identify_compute(ctx);
                    std::sync::Arc::new(self.prepare_static_identification(physical)?.ok_or_else(
                        || CausalError::Compile {
                            message: "attribution identified no structural product".into(),
                        },
                    )?)
                };
                super::super::CheckedAttributionOperation::checked(self, physical, &cache, ctx)?
                    .execute(data, ctx)
            }
            TabularCall::Anomaly(q) => {
                let graph = self.require_execute_dag(
                    "AnomalyAttribution execute requires a supplied static DAG",
                )?;
                self.execute_anomaly(data, graph, q, physical, ctx)
            }
            TabularCall::ChangeAttribution(q) => {
                let graph = self.require_execute_dag(
                    "ChangeAttribution execute requires a supplied static DAG",
                )?;
                self.execute_change_attribution(data, graph, q, physical, ctx)
            }
            TabularCall::MechanismChange(q) => {
                let graph = self.require_execute_dag(
                    "MechanismChange execute requires a supplied static DAG",
                )?;
                // The posterior probability of a change lives on the sealed operation.
                if self.settings.inference.is_bayesian() {
                    let cache = if let Some((cache, _)) = self.reuse_static_identification(physical)
                    {
                        cache
                    } else {
                        super::super::report_identify_compute(ctx);
                        std::sync::Arc::new(
                            self.prepare_static_identification(physical)?.ok_or_else(|| {
                                CausalError::Compile {
                                    message: "mechanism change identified no structural product"
                                        .into(),
                                }
                            })?,
                        )
                    };
                    return super::super::CheckedAttributionOperation::checked(
                        self, physical, &cache, ctx,
                    )?
                    .execute(data, ctx);
                }
                self.execute_mechanism_change(data, graph, q, physical, ctx)
            }
            TabularCall::UnitChange(q) => {
                let graph =
                    self.require_execute_dag("UnitChange execute requires a supplied static DAG")?;
                self.execute_unit_change(data, graph, q, physical, ctx)
            }
            _ => Err(route_mismatch()),
        }
    }

    /// An attribution or simulation on tabular data: the sealed attribution operation when the
    /// query carries uncertainty over the mechanisms, else the fitted-model route.
    pub(super) fn execute_tabular_attribute_or_simulate(
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
            && super::super::attribute_runs_checked(
                &self.query,
                &self.settings.inference,
                self.settings.bootstrap_replicates,
            )
        {
            let cache = if let Some((cache, _)) = self.reuse_static_identification(physical) {
                cache
            } else {
                super::super::report_identify_compute(ctx);
                std::sync::Arc::new(self.prepare_static_identification(physical)?.ok_or_else(
                    || CausalError::Compile {
                        message: "attribution identified no structural product".into(),
                    },
                )?)
            };
            return super::super::CheckedAttributionOperation::checked(
                self, physical, &cache, ctx,
            )?
            .execute(data, ctx);
        }
        self.execute_model_route(data, graph, physical, ctx)
    }

    pub(super) fn execute_tabular_route_rows(
        &self,
        call: TabularCall<'_>,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match call {
            TabularCall::Response(q) => {
                self.execute_tabular_response_route(data, q, physical, execution, ctx)
            }
            TabularCall::StaticAte(q) => {
                self.execute_tabular_static_ate_route(data, q, physical, execution, ctx)
            }
            TabularCall::Distribution(q) => {
                self.execute_tabular_distribution_route(data, q, physical, execution, ctx)
            }
            TabularCall::PathSpecific(q) => {
                self.execute_tabular_path_specific_route(data, q, physical, ctx)
            }
            TabularCall::Conditional(q) => {
                self.execute_tabular_conditional_route(data, q, physical, ctx)
            }
            TabularCall::StaticMediation(source) => {
                self.execute_tabular_static_mediation_route(data, source, physical, execution, ctx)
            }
            TabularCall::Counterfactual(q) => {
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
            TabularCall::Anomaly(_)
            | TabularCall::ChangeAttribution(_)
            | TabularCall::MechanismChange(_)
            | TabularCall::UnitChange(_) => {
                self.execute_tabular_attribution_route(call, data, physical, ctx)
            }
            TabularCall::Transport(q) => self.execute_transport(data, q, physical, ctx),
            TabularCall::Interference(q) => {
                self.execute_tabular_interference_route(data, q, physical, ctx)
            }
            TabularCall::Other(route) => {
                self.execute_tabular_other_route(route, data, physical, ctx)
            }
        }
    }

    /// The tabular routes whose executors read the study's query themselves.
    pub(super) fn execute_tabular_other_route(
        &self,
        route: AnalysisRoute,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match route {
            AnalysisRoute::TemporalCrossWorld => Err(CausalError::Unsupported {
                message: "temporal cross-world effect requires series data and a TemporalDag",
            }),
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
            | AnalysisRoute::MultiEnvTemporalEffect => Err(route_unsupported()),
            // These routes arrive with their query payload, never as `Other`.
            AnalysisRoute::Response
            | AnalysisRoute::StaticAte
            | AnalysisRoute::Distribution
            | AnalysisRoute::PathSpecific
            | AnalysisRoute::Conditional
            | AnalysisRoute::StaticMediation
            | AnalysisRoute::Counterfactual
            | AnalysisRoute::Anomaly
            | AnalysisRoute::ChangeAttribution
            | AnalysisRoute::MechanismChange
            | AnalysisRoute::UnitChange
            | AnalysisRoute::Transport
            | AnalysisRoute::Interference => Err(route_mismatch()),
        }
    }

    pub(super) fn execute_tabular_path_specific_route(
        &self,
        data: &TabularData,
        q: &antecedent_core::PathSpecificEffectQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match self.graph.graph_ref() {
            GraphRef::Admg(admg) => self.execute_path_specific_mixed(
                data,
                admg,
                super::super::PathSpecificMixedSource::PathSpecific(q),
                physical,
                ctx,
            ),
            GraphRef::TemporalAdmg(temporal) => {
                let admg = temporal_admg_projection(temporal)?;
                self.execute_path_specific_mixed(
                    data,
                    &admg,
                    super::super::PathSpecificMixedSource::PathSpecific(q),
                    physical,
                    ctx,
                )
            }
            GraphRef::Cpdag(cpdag) => self.execute_class_path_specific(
                data,
                cpdag,
                super::super::PathSpecificMixedSource::PathSpecific(q),
                physical,
                ctx,
            ),
            GraphRef::Pag(pag) => self.execute_pag_path_specific(
                data,
                pag,
                super::super::PathSpecificMixedSource::PathSpecific(q),
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

    /// Static response execution: which executor runs depends on the graph class and tier closure.
    pub(super) fn execute_tabular_response_route(
        &self,
        data: &TabularData,
        q: &ResponseQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match self.graph.graph_ref() {
            GraphRef::Dag(graph) => {
                self.execute_response(data, graph, q, physical, execution.response_curve(), ctx)
            }
            GraphRef::Cpdag(_) | GraphRef::Pag(_) => {
                self.execute_class_response(data, q, physical, ctx)
            }
            GraphRef::Admg(_)
                if self.declared.tiered.as_ref().is_some_and(|b| {
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
    pub(super) fn execute_tabular_static_ate_route(
        &self,
        data: &TabularData,
        q: &antecedent_core::AverageEffectQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let Some(background) = &self.declared.tiered {
            if self.settings.inference.is_bayesian() {
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
    pub(super) fn execute_tabular_distribution_route(
        &self,
        data: &TabularData,
        q: &antecedent_core::InterventionalDistributionQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match self.graph.graph_ref() {
            GraphRef::Dag(graph) => self.execute_distribution(
                data,
                super::super::static_path::DistributionGraph::Dag(graph),
                q,
                physical,
                execution.distribution(),
                ctx,
            ),
            GraphRef::Admg(admg) => self.execute_distribution(
                data,
                super::super::static_path::DistributionGraph::Admg(admg),
                q,
                physical,
                execution.distribution(),
                ctx,
            ),
            GraphRef::TemporalAdmg(temporal) => {
                let admg = temporal_admg_projection(temporal)?;
                self.execute_distribution(
                    data,
                    super::super::static_path::DistributionGraph::Admg(&admg),
                    q,
                    physical,
                    execution.distribution(),
                    ctx,
                )
            }
            GraphRef::Cpdag(cpdag) => self.execute_class_distribution(
                data,
                super::super::class_distribution::DistributionClass::Cpdag(cpdag),
                q,
                physical,
                ctx,
            ),
            GraphRef::Pag(pag) => self.execute_class_distribution(
                data,
                super::super::class_distribution::DistributionClass::Pag(pag),
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
    pub(super) fn execute_tabular_conditional_route(
        &self,
        data: &TabularData,
        q: &antecedent_core::ConditionalEffectQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
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
    pub(super) fn execute_tabular_static_mediation_route(
        &self,
        data: &TabularData,
        source: StaticMediationSource<'_>,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if let StaticMediationSource::Nested(q) = source {
            match self.graph.graph_ref() {
                GraphRef::Admg(admg) => {
                    return self.execute_path_specific_mixed(
                        data,
                        admg,
                        super::super::PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    );
                }
                GraphRef::TemporalAdmg(temporal) => {
                    let admg = temporal_admg_projection(temporal)?;
                    return self.execute_path_specific_mixed(
                        data,
                        &admg,
                        super::super::PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    );
                }
                GraphRef::Cpdag(cpdag) => {
                    return self.execute_class_path_specific(
                        data,
                        cpdag,
                        super::super::PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    );
                }
                GraphRef::Pag(pag) => {
                    return self.execute_pag_path_specific(
                        data,
                        pag,
                        super::super::PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    );
                }
                _ => {}
            }
        }
        if let StaticMediationSource::Mediation(q) = source {
            use super::super::mediation_mixed::ClassStructure;
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
        match source {
            StaticMediationSource::Mediation(q) => {
                self.execute_static_mediation_total(data, graph, q, physical, ctx)
            }
            StaticMediationSource::Nested(q) => {
                self.execute_nested_counterfactual(data, graph, q, physical, execution, ctx)
            }
        }
    }

    /// Interference execution: checked operation for the design-based assignments, the Gaussian path otherwise.
    pub(super) fn execute_tabular_interference_route(
        &self,
        data: &TabularData,
        q: &antecedent_core::InterferenceQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
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
    pub(super) fn execute_nested_counterfactual(
        &self,
        data: &TabularData,
        graph: &Dag,
        q: &NestedCounterfactualQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        execution: &super::super::super::prepared::PreparedExecution,
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
        let bayesian = self.settings.inference.is_bayesian();
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
        result.extras.interval = None;
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
}
