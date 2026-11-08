//! Dispatch of sealed one-shot execution routes.

use super::{
    AnalysisRoute, CausalError, CausalQuery, DEFAULT_IDENTIFIER_ID, DataInput, EstimatorId,
    ExecutionContext, ExecutionKind, GraphClass, GraphRef, IdentifierId, InferenceMode,
    OverlapPolicy, RefuteSuite, RouteInput, StaticMediationSource, StudyResult,
    admg_has_bidirected, complete_mean_response, distinct_numeric_arms,
    graph_posterior_response_supported, is_class_atom, mean_all_observed,
    response_functional_is_derivative,
};

impl super::super::Study {
    /// Bayesian basis ATE, robust ATE, and basis CATE one-shot routes that retain a sealed checked operation.
    #[inline(never)]
    pub(super) fn run_sealed_bayesian_basis_and_robust_effect(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.structure_source == crate::support::StructureSource::Explicit
            && self.settings.inference.is_bayesian()
            && self.declared.estimator == Some(EstimatorId::BayesianBasisGcomp)
            && self
                .declared
                .identifier
                .is_none_or(|identifier| identifier == IdentifierId::BackdoorAdjustment)
            && self.settings.refute == RefuteSuite::None
            && let Some(RouteInput::StaticAteDag { data, query, .. }) = route
            && mean_all_observed(&query.outcome_functional, &query.target_population)
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::BayesianBasisAte),
                "one-shot Bayesian basis ATE did not retain its checked operation",
            );
        }
        if self.plain_fixed_route()
            && self.structure_source == crate::support::StructureSource::Explicit
            && self.settings.inference.is_bayesian()
            && self.declared.estimator == Some(EstimatorId::BayesianRobustAte)
            && self.declared.identifier.is_none_or(|id| id == IdentifierId::BackdoorAdjustment)
            && self.settings.refute == RefuteSuite::None
            && let Some(RouteInput::StaticAteDag { data, query, .. }) = route
            && mean_all_observed(&query.outcome_functional, &query.target_population)
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::BayesianRobustAte),
                "one-shot Bayesian robust ATE did not retain its checked operation",
            );
        }
        if self.plain_fixed_route()
            && self.structure_source == crate::support::StructureSource::Explicit
            && self.settings.inference.is_bayesian()
            && self.declared.estimator == Some(EstimatorId::BayesianBasisGcomp)
            && self.declared.identifier.is_none_or(|id| id == IdentifierId::BackdoorAdjustment)
            && self.settings.refute == RefuteSuite::None
            && let Some(RouteInput::ConditionalDag { data, query, .. }) = route
            && mean_all_observed(&query.inner.outcome_functional, &query.inner.target_population)
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::BayesianBasisCate),
                "one-shot Bayesian basis CATE did not retain its checked operation",
            );
        }
        Ok(None)
    }

    /// Bayesian g-computation and Bayesian conditional DAG one-shot routes that retain a sealed checked operation.
    #[inline(never)]
    pub(super) fn run_sealed_bayesian_dag_effect(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.graph_posterior.is_none()
            && self.declared.tiered.is_none()
            && self.fixed_structure()
            && self.settings.inference.is_bayesian()
            && self.declared.estimator == Some(EstimatorId::BayesianGcomp)
            && self
                .declared
                .identifier
                .is_none_or(|identifier| identifier == IdentifierId::BackdoorAdjustment)
            && let Some(RouteInput::StaticAteDag { data, query, .. }) = route
            && mean_all_observed(&query.outcome_functional, &query.target_population)
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::BayesianGcomp),
                "one-shot Bayesian DAG effect did not retain its checked operation",
            );
        }
        if self.plain_fixed_route()
            && self.settings.inference.is_bayesian()
            && self.declared.estimator == Some(EstimatorId::BayesianConditional)
            && self
                .declared
                .identifier
                .is_none_or(|identifier| identifier == IdentifierId::BackdoorAdjustment)
            && self.point_validation()
            && let Some(RouteInput::ConditionalDag { data, query, .. }) = route
            && !query.inner.effect_modifiers.is_empty()
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::BayesianConditional),
                "one-shot Bayesian DAG conditional effect did not retain its checked operation",
            );
        }
        Ok(None)
    }

    /// Static mediation, temporal DAG effect, and temporal DAG response one-shot routes that retain a sealed checked operation.
    #[inline(never)]
    pub(super) fn run_sealed_mediation_and_temporal_dag(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.settings.inference.is_frequentist()
            && let Some(RouteInput::StaticMediationDag {
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
        if self.plain_fixed_route()
            && self.settings.inference.is_frequentist()
            && self.declared.split.is_none()
            && self.point_validation()
            && let Some(RouteInput::TemporalEffectDag { data, query, .. }) = route
            && query.policy.is_pulse_or_sustained()
            && self
                .declared
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
        if self.plain_fixed_route()
            && self.settings.refute == RefuteSuite::None
            && self
                .declared
                .estimator
                .is_none_or(|id| id == EstimatorId::temporal_response_for(&self.settings.inference))
            && let Some(RouteInput::TemporalResponseDag { data, query, .. }) = route
            && query.temporal.is_some()
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
        Ok(None)
    }

    /// Frequentist graph-posterior (DAG and class atom) average-effect one-shot routes.
    #[inline(never)]
    pub(super) fn run_sealed_frequentist_graph_posterior_effect(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.graph_posterior.as_ref().is_some_and(|posterior| {
            posterior.atom_kind() == antecedent_discovery::GraphPosteriorAtomKind::Dag
        }) && self.settings.inference.is_frequentist()
            && self.point_validation()
            && self.declared.custom_validators.is_empty()
            && self.declared.estimator.is_none_or(|id| id == EstimatorId::LinearAdjustmentAte)
            && let Some(RouteInput::StaticAteDag { data, .. }) = route
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::GraphPosteriorEffect),
                "one-shot graph-posterior effect did not retain its checked operation",
            );
        }
        if self
            .graph_posterior
            .as_ref()
            .is_some_and(|posterior| is_class_atom(posterior.atom_kind()))
            && self.settings.inference.is_frequentist()
            && self.declared.estimator.is_none_or(|id| id == EstimatorId::LinearAdjustmentAte)
            && let Some(
                RouteInput::StaticAteCpdag { data, query, .. }
                | RouteInput::StaticAtePag { data, query, .. },
            ) = route
            && Self::graph_posterior_target_is_sealable(self, query)
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::ClassGraphPosteriorEffect),
                "one-shot class graph-posterior effect did not retain its checked operation",
            );
        }
        Ok(None)
    }

    /// Frequentist conditional-effect and static DAG response one-shot routes.
    #[inline(never)]
    pub(super) fn run_sealed_static_dag_conditional_and_response(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.settings.inference.is_frequentist()
            && self.point_validation()
            && self
                .declared
                .estimator
                .is_none_or(|id| id == EstimatorId::ConditionalLinearAdjustment)
            && let Some(RouteInput::ConditionalDag { data, query, .. }) = route
            && !query.inner.effect_modifiers.is_empty()
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::CheckedConditional),
                "one-shot conditional effect did not retain its checked operation",
            );
        }
        if self.plain_fixed_route()
            && let Some(RouteInput::StaticResponseDag { data, query, .. }) = route
            && query.temporal.is_none()
            && complete_mean_response(query)
            && match (&query.functional, &self.settings.inference) {
                (
                    antecedent_core::ResponseFunctional::MeanCurve { .. },
                    InferenceMode::Frequentist,
                ) => {
                    self.settings.refute == RefuteSuite::None
                        && self
                            .declared
                            .estimator
                            .is_none_or(|id| id == EstimatorId::ResponseKennedyDr)
                }
                (
                    antecedent_core::ResponseFunctional::InterventionResponse { .. },
                    InferenceMode::Frequentist,
                ) => self
                    .declared
                    .estimator
                    .is_none_or(|id| id == EstimatorId::ResponseInterventionGcomp),
                (
                    antecedent_core::ResponseFunctional::MeanCurve { .. },
                    InferenceMode::Bayesian(_),
                ) => {
                    self.settings.refute == RefuteSuite::None
                        && self
                            .declared
                            .estimator
                            .is_none_or(|id| id == EstimatorId::ResponseBayesian)
                }
                // The Bayesian intervention level seals its cheap/full refuter
                // state too (it executes unsupported).
                (
                    antecedent_core::ResponseFunctional::InterventionResponse { .. },
                    InferenceMode::Bayesian(_),
                ) => {
                    matches!(
                        self.settings.refute,
                        RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                    ) && self
                        .declared
                        .estimator
                        .is_none_or(|id| id == EstimatorId::ResponseBayesian)
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
        Ok(None)
    }

    /// Derivative response, counterfactual, and complete-program one-shot routes.
    #[inline(never)]
    pub(super) fn run_sealed_derivative_counterfactual_and_program(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.graph_posterior.is_none()
            && self.declared.tiered.is_none()
            && self.settings.inference.is_frequentist()
            && self.settings.refute == RefuteSuite::None
            && self.declared.custom_validators.is_empty()
            && let Some(RouteInput::StaticResponseDag { data, query, .. }) = route
            && query.temporal.is_none()
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
        if self.graph_posterior.is_none()
            && let Some(RouteInput::GcmDag { data, route: AnalysisRoute::Counterfactual, .. }) =
                route
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::Counterfactual),
                "one-shot counterfactual route did not retain its checked operation",
            );
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
            && let DataInput::Tabular(data) = &self.data
        {
            // A distribution over an equivalence class is decided member by member at
            // each execution: its checked operation is the lowered route itself.
            let class_distribution = matches!(self.query, CausalQuery::Distribution(_))
                && matches!(self.graph.graph_ref(), GraphRef::Cpdag(_) | GraphRef::Pag(_));
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| {
                    prepared.has_complete_program_operation(&self.query)
                        || (class_distribution && prepared.retains(ExecutionKind::EstimatorRoute))
                },
                "one-shot route did not retain its complete checked operation",
            );
        }
        Ok(None)
    }

    /// AIPW one-shot route: its checked operation is complete for the binary all-observed mean ATE.
    #[inline(never)]
    pub(super) fn run_sealed_aipw(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        // AIPW's supported checked operation is complete for the binary,
        // all-observed mean ATE on a supplied DAG under any diagnosed overlap
        // policy: the trim rule selects the cross-fitted or the common-support
        // procedure inside the retained operation. Route that selection through
        // preparation; nearby estimators execute through the ordinary dispatcher.
        if self.graph_posterior.is_none()
            && self.declared.tiered.is_none()
            && self.settings.inference.is_frequentist()
            && self.declared.custom_validators.is_empty()
            && !self
                .declared
                .estimator_spec
                .as_deref()
                .is_some_and(crate::estimator_spec::EstimatorSpec::excludes_undeclared_levels)
            && matches!(
                super::super::super::prepared::checked_aipw_fitter(self).overlap,
                OverlapPolicy::RequireDiagnostics { .. }
            )
            && (self.declared.estimator == Some(EstimatorId::Aipw)
                || (self.declared.estimator.is_none()
                    && matches!(
                        self.declared.estimator_spec.as_deref(),
                        Some(
                            crate::estimator_spec::EstimatorSpec::Default(EstimatorId::Aipw)
                                | crate::estimator_spec::EstimatorSpec::Aipw(_),
                        )
                    )))
            && let Some(RouteInput::StaticAteDag { data, query, .. }) = route
            && mean_all_observed(&query.outcome_functional, &query.target_population)
            && distinct_numeric_arms(query)
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                crate::PreparedStudy::has_sealed_aipw_operation,
                "one-shot AIPW route did not retain its complete checked operation",
            );
        }
        Ok(None)
    }

    /// Checked GLM adjustment, sharp-RD, and propensity one-shot routes.
    #[inline(never)]
    pub(super) fn run_sealed_glm_rd_propensity(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        // Checked GLM adjustment and sharp-RD operations retain the selected
        // target and fitted design through the prepared facade.
        let selected_estimator = self.selected_estimator_id();
        let checked_glm = selected_estimator == Some(EstimatorId::GlmAdjustment)
            && self.plain_fixed_route()
            && self.settings.inference.is_frequentist()
            && matches!(
                route,
                Some(RouteInput::StaticAteDag { query, .. })
                    if matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            );
        let checked_rd = selected_estimator == Some(EstimatorId::RdSharp)
            && self.rd.is_some()
            && self.plain_fixed_route()
            && self.settings.inference.is_frequentist()
            && matches!(route, Some(RouteInput::StaticAteDag { .. }));
        let checked_propensity = matches!(
            selected_estimator,
            Some(EstimatorId::PropensityWeighting | EstimatorId::PropensityMatching)
        ) && self.plain_fixed_route()
            && self.settings.inference.is_frequentist()
            && !self
                .declared
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
        } else if (checked_glm || checked_propensity)
            && let Some(RouteInput::StaticAteDag { data, .. }) = route
        {
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
        Ok(None)
    }

    /// Static class response and graph-posterior response one-shot routes.
    #[inline(never)]
    pub(super) fn run_sealed_class_and_posterior_response(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if super::super::CheckedStaticClassResponse::admits(self)
            && let Some(
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
        if self.declared.tiered.is_none()
            && self.declared.custom_validators.is_empty()
            && self.graph_posterior.as_ref().is_some_and(|posterior| {
                matches!(
                    posterior.atom_kind(),
                    antecedent_discovery::GraphPosteriorAtomKind::Dag
                        | antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                )
            })
            && let Some(
                RouteInput::StaticResponseDag { data, query, .. }
                | RouteInput::StaticResponseCpdag { data, query, .. }
                | RouteInput::StaticResponsePag { data, query, .. },
            ) = route
            && graph_posterior_response_supported(query).is_ok()
            && mean_all_observed(&query.outcome_functional, &query.target_population)
            && match &query.functional {
                antecedent_core::ResponseFunctional::MeanCurve { .. } => {
                    self.settings.refute == RefuteSuite::None
                }
                antecedent_core::ResponseFunctional::InterventionResponse { .. } => {
                    matches!(
                        self.settings.refute,
                        RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                    )
                }
                _ => false,
            }
            && self.declared.estimator.is_none_or(|id| {
                id == EstimatorId::static_response_for(&query.functional, &self.settings.inference)
            })
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::GraphPosteriorResponse),
                "one-shot graph-posterior response did not retain its checked operation",
            );
        }
        Ok(None)
    }

    /// `CoDetermined` cell-AIPW response and Bayesian static class effect one-shot routes.
    #[inline(never)]
    pub(super) fn run_sealed_cell_aipw_and_bayesian_class_effect(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.structure_source == crate::support::StructureSource::Explicit
            && self.settings.inference.is_frequentist()
            && self.shared_batch_design.is_none()
            && self.continuous_cell.is_none()
            && self.declared.estimator == Some(EstimatorId::CellAipw)
            && let Some(RouteInput::CodeterminedJoint { data, query, .. }) = route
            && query.temporal.is_none()
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
        if self.plain_fixed_route()
            && self.settings.inference.is_bayesian()
            && self.declared.identifier.is_none_or(|id| id == IdentifierId::GeneralizedAdjustment)
            && self.declared.estimator.is_none_or(|id| id == EstimatorId::BayesianGcomp)
            && matches!(
                self.declared.estimator_spec.as_deref(),
                None | Some(crate::estimator_spec::EstimatorSpec::Default(
                    EstimatorId::BayesianGcomp
                ))
            )
            && self.point_validation_or_placebo()
            && let Some(
                RouteInput::StaticAteCpdag { data, query, .. }
                | RouteInput::StaticAtePag { data, query, .. },
            ) = route
            && mean_all_observed(&query.outcome_functional, &query.target_population)
        {
            return self.admit_sealed_tabular(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::StaticClassEffect),
                "one-shot Bayesian static class effect did not retain its checked operation",
            );
        }
        Ok(None)
    }

    /// Graph-posterior conditional and Bayesian class-posterior effect one-shot route.
    #[inline(never)]
    pub(super) fn run_sealed_graph_posterior_conditional_effect(
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
            let class_atoms = is_class_atom(posterior.atom_kind());
            let dag_atoms =
                posterior.atom_kind() == antecedent_discovery::GraphPosteriorAtomKind::Dag;
            let bayesian = self.settings.inference.is_bayesian();
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
                && self.declared.identifier.is_none_or(|id| id == identifier)
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
    pub(super) fn run_sealed_temporal_class_response_and_posterior(
        &self,
        route: Option<RouteInput<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.settings.refute == RefuteSuite::None
            && self.observation_delayed_entry.is_none()
            && self.declared.identifier.is_none_or(|id| id == IdentifierId::GeneralizedAdjustment)
            && self
                .declared
                .estimator
                .is_none_or(|id| id == EstimatorId::temporal_response_for(&self.settings.inference))
            && let Some(RouteInput::TemporalResponseClass { data, query, .. }) = route
            && query.temporal.as_ref().is_some_and(|spec| spec.policy.is_single_step())
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
        if crate::analysis::CheckedTemporalGraphPosteriorResponse::admits(self)
            && let DataInput::Temporal(data) | DataInput::Event(data) = &self.data
        {
            return self.admit_sealed_series(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::TemporalGraphPosteriorResponse),
                "one-shot temporal graph-posterior response did not retain its checked operation",
            );
        }
        if super::super::CheckedTemporalClassMediationOperation::admits(self)
            && let DataInput::Temporal(data) | DataInput::Event(data) = &self.data
        {
            return self.admit_sealed_series(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::TemporalClassMediation),
                "one-shot temporal class mediation did not retain its checked operation",
            );
        }
        if crate::analysis::CheckedBayesianTemporalClassEffectOperation::admits(self)
            && let DataInput::Temporal(data) | DataInput::Event(data) = &self.data
        {
            return self.admit_sealed_series(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::BayesianTemporalClassEffect),
                "one-shot Bayesian temporal class effect did not retain its checked operation",
            );
        }
        if crate::analysis::CheckedTemporalGraphPosteriorEffect::admits(self)
            && let DataInput::Temporal(data) | DataInput::Event(data) = &self.data
        {
            return self.admit_sealed_series(
                data,
                ctx,
                |prepared| prepared.retains(ExecutionKind::TemporalGraphPosteriorEffect),
                "one-shot temporal graph-posterior effect did not retain its checked operation",
            );
        }
        Ok(None)
    }

    /// Bayesian specialist, transport, and linear-adjustment one-shot routes.
    #[inline(never)]
    pub(super) fn run_sealed_specialist_transport_and_linear(
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
            && self.declared.split.is_none()
            && self.settings.inference.is_bayesian()
            && self.settings.refute == RefuteSuite::None
            && let Some(RouteInput::StaticAteDag { data, query, .. }) = route
            && (matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
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
        // Certified trial-to-target transport executes from its retained sID
        // proof and design columns even for the one-shot facade.
        if self.selection_diagram.is_some()
            && self.transport_trial.is_some()
            && self.plain_fixed_route()
            && self.declared.split.is_none()
            && self.settings.refute == RefuteSuite::None
            && let Some(RouteInput::TransportMixed {
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
        // The static mean adjustment route executes from the prepared checked
        // lowering even for the one-shot facade. A configuration no sealed
        // operation admits executes through the ordinary dispatcher below.
        if self.graph_posterior.is_none()
            && self.declared.tiered.is_none()
            && self.settings.inference.is_frequentist()
            && self.declared.identifier.is_none_or(|id| id == DEFAULT_IDENTIFIER_ID)
            && matches!(self.declared.estimator, None | Some(EstimatorId::LinearAdjustmentAte))
            && matches!(
                self.declared.estimator_spec.as_deref(),
                None | Some(
                    crate::estimator_spec::EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte,)
                        | crate::estimator_spec::EstimatorSpec::LinearAdjustmentAte(_),
                )
            )
            && let Some(RouteInput::StaticAteDag { data, query, .. }) = route
            && mean_all_observed(&query.outcome_functional, &query.target_population)
        {
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
        Ok(None)
    }

    #[inline(never)]
    pub(super) fn run_checked_admg_response(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        let supported_shape = match &self.query {
            CausalQuery::Response(query)
                if query.temporal.is_none()
                    && !response_functional_is_derivative(&query.functional)
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population
                        == antecedent_core::TargetPopulation::AllObserved
                    && query.outcome_functional == antecedent_core::OutcomeFunctional::Mean =>
            {
                matches!(
                    (&query.functional, self.settings.refute),
                    (_, RefuteSuite::None)
                        | (
                            antecedent_core::ResponseFunctional::InterventionResponse { .. },
                            RefuteSuite::Cheap | RefuteSuite::Full,
                        )
                )
            }
            _ => false,
        };
        if supported_shape
            && self.plain_fixed_route()
            && self.graph.class() == GraphClass::Admg
            && self.graph.as_admg().is_some_and(admg_has_bidirected)
            && self
                .declared
                .identifier
                .is_none_or(|identifier| identifier == IdentifierId::GeneralId)
            && self
                .declared
                .estimator
                .is_none_or(|estimator| estimator == EstimatorId::FunctionalEffect)
            && let DataInput::Tabular(data) = &self.data
        {
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
        Ok(None)
    }

    pub(super) fn run_checked_bayesian_static_mediation(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<Box<StudyResult>>, CausalError> {
        if self.plain_fixed_route()
            && self.graph.class() == GraphClass::Dag
            && self.settings.inference.is_bayesian()
            && self.point_validation()
            && matches!(self.query, CausalQuery::Mediation(_))
            && let DataInput::Tabular(data) = &self.data
        {
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
        Ok(None)
    }
}
