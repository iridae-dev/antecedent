// SPDX-License-Identifier: MIT OR Apache-2.0

use super::completion_fit_memo::{CompletionFitKey, CompletionFitMemo};
use super::temporal_class_posterior_response::{PolicyResponseMix, mix_responses_under_policy};
use super::{
    Arc, AverageEffectQuery, BayesianConfig, CausalError, CausalQuery, CausalResponse,
    ClassEnvelopeRun, ContinuousResponseEstimator, Dag, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimandMethod, EstimationWorkspace, EstimatorId,
    EvalError, ExecuteRun, ExecutionContext, FunctionalDistributionWorkspace, FunctionalEffect,
    GraphClass, GraphIdentFlag, GraphPosterior, IdentificationResult, IdentificationStatus,
    IdentifiedClaim, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, IdentifierId, InferenceMode, Instant, Intervention,
    ObservationMechanismEstimator, ObservationSpec, OverlapPolicy, PhysicalExecutionPlan,
    PosteriorMassSummary, RefutationRun, RefuteSuite, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, Strategy, StudyResult, TableView,
    TabularData, VariableId, WeightedGraphSamples, bayesian_gcomp, elapsed_ns,
    envelope_se_omits_between_atom_variance, envelope_to_identification_result_for,
    functional_cell_unevaluable, identification_from_cache_or, identification_status_ok_for_case,
    identified_set_interval_diagnostics, identify_cached_diagnostic,
    interactive_subsample_graphs_accounted, provenance_ids, report_identify_compute,
    resolve_structural_aggregation, response_functional_is_derivative, scalar_identified_set,
    support_from_functional_eval,
};
use crate::planner::RouteDefaults;

/// What the joint cell's summary reads beyond the scores: the sampling weights of the analysed
/// rows (`None`: equal) and the declared dependence, both aligned to the score rows.
struct CellAipwReading {
    weights: Option<Vec<f64>>,
    dependence: Option<antecedent_estimate::ScoreDependence>,
    record: Option<antecedent_estimate::SamplingWeightRecord>,
}

/// Treatment coordinates and outcome of a cell.aipw joint-arm query.
#[derive(Clone, Copy)]
struct CellAipwTarget<'a> {
    treatments: &'a [VariableId],
    outcome: VariableId,
}

/// Point-identified scalar response of a cell.aipw joint arm.
fn cell_aipw_response(
    query: &ResponseQuery,
    identification: &IdentificationResult,
    scalar: f64,
    se: f64,
    n_treatments: usize,
    requested_arm: u32,
) -> CausalResponse {
    CausalResponse {
        estimand: query.functional.clone(),
        identification_status: identification.status,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Scalar(scalar)),
        uncertainty: ResponseUncertainty::Scalar {
            standard_error: se,
            lower: scalar - crate::result::reported_se_interval_z() * se,
            upper: scalar + crate::result::reported_se_interval_z() * se,
            level: 0.95,
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        },
        support: antecedent_core::SupportReport {
            status: antecedent_core::SupportStatus::Supported,
            query_region: antecedent_core::SupportRegion {
                minima: Arc::from(
                    (0..n_treatments)
                        .map(|j| f64::from((requested_arm >> j) & 1))
                        .collect::<Vec<_>>(),
                ),
                maxima: Arc::from(
                    (0..n_treatments)
                        .map(|j| f64::from((requested_arm >> j) & 1))
                        .collect::<Vec<_>>(),
                ),
            },
            diagnostics: Vec::new(),
            warnings: Vec::new(),
            // One scalar answers the whole joint arm: one coordinate, one label (the
            // summary itself).
            point_status: Some(Arc::from([antecedent_core::SupportStatus::Supported])),
        },
        assumptions: identification.required_assumptions.clone(),
        provenance_id: Arc::from("estimate.cell.aipw"),
        horizon_identification: None,
        interaction_structurally_zero: false,
    }
}

/// Borrowed inputs of the plugin-level refutation of a scalar response.
#[derive(Clone, Copy)]
struct ResponseRefuteInput<'a> {
    data: &'a TabularData,
    query: &'a ResponseQuery,
    estimand: &'a IdentifiedEstimand,
    estimate: &'a EffectEstimate,
    scalar: f64,
    estimator_id: EstimatorId,
    treatment: VariableId,
    outcome: VariableId,
}

/// Borrowed inputs shared by the phases of a graph-posterior response execution.
#[derive(Clone, Copy)]
struct GraphPosteriorResponseRun<'a> {
    data: &'a TabularData,
    query: &'a ResponseQuery,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &'a ExecutionContext,
}

/// Atom fits of a graph-posterior response with the masses pooling accounts for.
struct GraphPosteriorResponseFits {
    graphs: WeightedGraphSamples,
    unidentified_mass: f64,
    subsampled_out_mass: f64,
    subsample_notes: Vec<Diagnostic>,
    atoms: Vec<crate::result::StructuralResponseAtom>,
    weighted: Vec<(u64, f64, CausalResponse)>,
    /// The estimand and partial-identification flag of each weighted atom, parallel to
    /// `weighted`.
    contributing: Vec<(IdentifiedEstimand, bool)>,
    estimand: IdentifiedEstimand,
    failed_mass: f64,
    atom_scores: Vec<(f64, antecedent_estimate::ResponseInfluence)>,
    options: antecedent_estimate::ContinuousResponseOptions,
}

/// The pooled response of a graph posterior with the values its diagnostics read.
struct GraphPosteriorPooled {
    identification: IdentificationResult,
    conditional: Option<ResponseValue>,
    mixed_if_se: Option<f64>,
    response: CausalResponse,
    estimate: EffectEstimate,
    scalar: f64,
    standard_error: f64,
    identified_mass: f64,
    total_mass: f64,
    structural_set: Option<antecedent_core::ResponseEnvelope>,
    aggregation_policy: crate::result::StructuralAggregationPolicy,
}

/// Predictive checks and the posterior refutation reports added to a Bayesian level.
type PosteriorLevelChecks =
    (Vec<antecedent_validate::PredictiveCheckReport>, Vec<antecedent_validate::RefutationReport>);

/// Structural atoms of a graph posterior, one per sample, before any fit fills their values.
fn graph_posterior_structural_atoms(
    graphs: &WeightedGraphSamples,
) -> Vec<crate::result::StructuralResponseAtom> {
    // Atom statuses come from the full ensemble: a subsampled-out atom keeps
    // its identified status and carries no value.
    graphs
        .graph_keys
        .iter()
        .zip(graphs.weights.iter())
        .zip(graphs.identified.iter())
        .map(|((&graph_key, &weight), flag)| crate::result::StructuralResponseAtom {
            posterior: None,
            response: None,
            graph_key,
            weight,
            status: if *flag == GraphIdentFlag::Identified {
                IdentificationStatus::NonparametricallyIdentified
            } else {
                IdentificationStatus::NotIdentified
            },
            value: None,
        })
        .collect()
}

mod graph_posterior;

mod cell_aipw;
mod codetermined;

impl super::Study {
    /// Identify and estimate a continuous response, including supported observation correction.
    pub(super) fn execute_response(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        checked_curve: Option<&super::super::prepared::CheckedStaticResponseCurve>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identification, estimand, identify_cached, identifier_id, estimator_id) =
            self.identify_static_response(graph, query, physical, checked_curve, ctx)?;
        self.finish_static_response(
            data,
            query,
            ExecuteRun { physical, ctx, started, identify_cached },
            IdentifiedClaim { identification, estimand },
            Strategy::new(identifier_id, estimator_id),
        )
    }

    /// Derivative or Jacobian on a latent-confounded ADMG, identified by one common adjustment
    /// set of the ADMG and estimated by the continuous response estimator on it.
    ///
    /// The adjustment set is read from the ADMG's proper back-door graph, so a bidirected edge
    /// keeps its latent common cause; nothing is coerced to a DAG. Identification that adjustment
    /// cannot reach is refused with the hedge certificate or with the ID-only boundary.
    pub(super) fn execute_admg_derivative(
        &self,
        data: &TabularData,
        admg: &antecedent_graph::Admg,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults {
            identifier: IdentifierId::GeneralizedAdjustment,
            estimator: EstimatorId::static_response_for(
                &query.functional,
                &self.settings.inference,
            ),
        })?;
        report_identify_compute(ctx);
        let identification = crate::strategy_table::identify_admg_derivative(admg, query)?;
        let identify_cached = false;
        let estimand = identification.estimands.first().cloned().ok_or_else(|| {
            CausalError::Compile { message: "response identifier returned no estimand".into() }
        })?;
        if identification
            .estimands
            .iter()
            .any(|candidate| candidate.adjustment_set != estimand.adjustment_set)
        {
            return Err(CausalError::Unsupported {
                message: "multi-pair response estimation requires one common adjustment set",
            });
        }
        self.finish_static_response(
            data,
            query,
            ExecuteRun { physical, ctx, started, identify_cached },
            IdentifiedClaim { identification, estimand },
            Strategy::new(identifier_id, estimator_id),
        )
    }

    /// The derivative of a front-door-identified response: the additive-outcome front-door
    /// estimator on the estimand's mediators, with its whole-procedure bootstrap interval
    /// (Frequentist) or Bayesian-bootstrap credible interval, resampling `block` consecutive
    /// rows together.
    ///
    /// # Errors
    ///
    /// The estimator's refusals.
    pub(super) fn estimate_front_door_derivative_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        claim: FrontDoorClaim<'_>,
        block: usize,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, CausalError> {
        let mut estimator =
            ContinuousResponseEstimator::new(Arc::clone(&claim.estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            estimator.options = options.clone();
        }
        let inference = match &self.settings.inference {
            InferenceMode::Frequentist => {
                antecedent_estimate::FrontDoorDerivativeInference::Frequentist {
                    replicates: self.settings.bootstrap_replicates,
                    block,
                }
            }
            InferenceMode::Bayesian(config) => {
                antecedent_estimate::FrontDoorDerivativeInference::Bayesian {
                    draws: config.n_draws,
                    block,
                }
            }
        };
        estimator
            .front_door_derivative(
                data,
                &query.functional,
                &claim.estimand.mediators,
                claim.status,
                claim.assumptions,
                inference,
                ctx,
            )
            .map_err(CausalError::from)
    }

    /// Result of a derivative identified by the front-door criterion.
    fn finish_front_door_derivative(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;
        let front_door = FrontDoorClaim {
            estimand: &estimand,
            status: identification.status,
            assumptions: identification.required_assumptions.clone(),
        };
        let response = self.estimate_front_door_derivative_response(
            &data_est,
            query,
            front_door.clone(),
            1,
            ctx,
        )?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = identification.diagnostics.clone();
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.extend(response.support.warnings.iter().cloned());
        // A front-door derivative's cheap/full suite is its own refuter state (unsupported).
        let (refutations, refute_notes) = self.refute_front_door_derivative_response(
            (&data_est, query),
            &front_door,
            &response,
            1,
            ctx,
        )?;
        diagnostics.extend(refute_notes);
        drop(front_door);
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.response.front_door_derivative",
                    "estimate.response.front_door_derivative",
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        );
        // No evidence row names the front-door derivative route: the cell's support belongs to
        // the adjustment-identified plug-ins, so a front-door run reports its status as refused.
        result.support_status = Some(crate::support::CellStatus::NotSupported);
        Ok(result)
    }

    /// Estimates and finishes a response whose static identification is already settled.
    #[allow(
        clippy::too_many_lines,
        reason = "one continuous response execution: observation correction, estimation, refutation and finish run in sequence"
    )]
    fn finish_static_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let cell_aipw = matches!(estimator_id, EstimatorId::CellAipw);

        if response_functional_is_derivative(&query.functional)
            && estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
        {
            return self.finish_front_door_derivative(
                data,
                query,
                ExecuteRun { physical, ctx, started, identify_cached },
                IdentifiedClaim { identification, estimand },
                Strategy::new(identifier_id, estimator_id),
            );
        }

        if estimand.method_kind().ok() == Some(EstimandMethod::GeneralId)
            || estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
        {
            return self.finish_general_id_response(
                data,
                query,
                ExecuteRun { physical, ctx, started, identify_cached },
                IdentifiedClaim { identification, estimand },
                Strategy::new(identifier_id, estimator_id),
            );
        }

        if cell_aipw {
            return self.execute_cell_aipw_response(
                data,
                query,
                ExecuteRun { physical, ctx, started, identify_cached },
                IdentifiedClaim { identification, estimand },
                Strategy::new(identifier_id, estimator_id),
                None,
            );
        }

        let (_, outcome) = response_primary_pair(&query.functional)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;

        let (response, response_scores) =
            self.estimate_continuous_response(&data_est, query, &estimand, &identification, ctx)?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let mut estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        attach_response_influence(&mut estimate, response_scores.as_ref())?;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let mut predictive_checks = Vec::new();
        let (refutations, refute_diags) = match &self.settings.inference {
            inference
                if super::derivative_refutation::is_derivative(&query.functional)
                    && !matches!(self.settings.refute, RefuteSuite::None) =>
            {
                let mut fitter =
                    ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
                if let Some(options) = &self.response_options {
                    fitter.options = options.clone();
                }
                let bayesian = match inference {
                    InferenceMode::Bayesian(config) => Some(
                        super::bayesian_level_refutation::bayesian_response_estimator_in_force(
                            &fitter, &data_est, query, config, ctx,
                        )?,
                    ),
                    InferenceMode::Frequentist => None,
                };
                let (reports, checks, notes) = super::derivative_refutation::refute_derivative(
                    &super::derivative_refutation::DerivativeRefutationInput {
                        data: &data_est,
                        query,
                        estimand: &estimand,
                        fitter: &fitter,
                        inference,
                        bayesian: bayesian.as_ref(),
                        status: identification.status,
                        assumptions: &identification.required_assumptions,
                        response: &response,
                        suite: self.settings.refute,
                        latency: self.latency_mode,
                        replicates: self.settings.bootstrap_replicates,
                        sampling_design: self.declared.sampling_design,
                    },
                    ctx,
                )?;
                predictive_checks = checks;
                (reports, notes)
            }
            InferenceMode::Bayesian(config)
                if scalar.is_finite()
                    && matches!(
                        query.functional,
                        ResponseFunctional::InterventionResponse { .. }
                    )
                    && !matches!(self.settings.refute, RefuteSuite::None) =>
            {
                let mut fitter =
                    ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
                if let Some(options) = &self.response_options {
                    fitter.options = options.clone();
                }
                let estimator =
                    super::bayesian_level_refutation::bayesian_response_estimator_in_force(
                        &fitter, &data_est, query, config, ctx,
                    )?;
                let state = super::bayesian_level_refutation::refute_bayesian_intervention_level(
                    &super::bayesian_level_refutation::BayesianLevelRefutationInput {
                        data: &data_est,
                        query,
                        estimand: &estimand,
                        fitter: &fitter,
                        estimator: &estimator,
                        status: identification.status,
                        assumptions: &identification.required_assumptions,
                        level: scalar,
                        level_sd: standard_error,
                        treatment,
                        outcome,
                        suite: self.settings.refute,
                        latency: self.latency_mode,
                        plugin_level: true,
                        summarize: response_scalar_summary,
                        target_label: "intervention-level",
                    },
                    ctx,
                )?;
                predictive_checks = state.predictive_checks;
                (state.refutations, state.diagnostics)
            }
            inference
                if super::curve_refutation::is_mean_curve(&query.functional)
                    && !matches!(self.settings.refute, RefuteSuite::None) =>
            {
                let mut fitter =
                    ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
                if let Some(options) = &self.response_options {
                    fitter.options = options.clone();
                }
                let bayesian = match inference {
                    InferenceMode::Bayesian(config) => Some(
                        super::bayesian_level_refutation::bayesian_response_estimator_in_force(
                            &fitter, &data_est, query, config, ctx,
                        )?,
                    ),
                    InferenceMode::Frequentist => None,
                };
                super::curve_refutation::refute_curve(
                    &super::curve_refutation::CurveRefutationInput {
                        data: &data_est,
                        query,
                        fitter: &fitter,
                        bayesian: bayesian.as_ref(),
                        status: identification.status,
                        assumptions: &identification.required_assumptions,
                        response: &response,
                        suite: self.settings.refute,
                        replicates: self.settings.bootstrap_replicates,
                    },
                    ctx,
                )?
            }
            _ => self.refute_response_plugin_level(
                ResponseRefuteInput {
                    data: &data_est,
                    query,
                    estimand: &estimand,
                    estimate: &estimate,
                    scalar,
                    estimator_id,
                    treatment,
                    outcome,
                },
                &mut diagnostics,
                ctx,
            )?,
        };
        diagnostics.extend(refute_diags);
        if !scalar.is_finite() {
            // `result.effect` exists for every plan, but a curve, vector, Jacobian or
            // identified set has no scalar reading. The NaN is a "not applicable", not a
            // failed computation, and the caller must read `result.response` instead.
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect summary is not applicable and the result is carried by the response payload",
            ));
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }

        let estimate_provenance = if query.observation == ObservationSpec::Complete {
            provenance_ids(Arc::clone(&response.provenance_id), Arc::clone(&response.provenance_id))
        } else {
            provenance_ids(
                "estimate.response.observation_adjusted",
                "estimate.response.observation_adjusted",
            )
        };
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(estimate_provenance),
                diagnostics: Some(diagnostics),
                response: Some(response),
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Resolves the response procedure and its identified claim, refusing mixed adjustment sets.
    fn identify_static_response(
        &self,
        graph: &Dag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        checked_curve: Option<&super::super::prepared::CheckedStaticResponseCurve>,
        ctx: &ExecutionContext,
    ) -> Result<
        (IdentificationResult, IdentifiedEstimand, bool, IdentifierId, EstimatorId),
        CausalError,
    > {
        let binding = physical.binding();
        let checked_identity = checked_curve
            .map(|plan| {
                plan.checked_route(query).map(|(identification, estimand)| {
                    let (identifier, estimator) = plan.procedure();
                    (identification.clone(), estimand.clone(), identifier, estimator)
                })
            })
            .transpose()?;
        let (identifier_id, estimator_id) = match &checked_identity {
            Some((_, _, identifier, estimator)) => (*identifier, *estimator),
            None => binding.resolve(RouteDefaults::RESPONSE)?,
        };
        let (identification, estimand, identify_cached) =
            if let Some((identification, estimand, _, _)) = checked_identity {
                (identification, estimand, true)
            } else {
                let reused = self.reuse_static_identification_if(physical, |_| {
                    binding.resolves_to(RouteDefaults::STATIC_ATE, identifier_id, estimator_id)
                });
                identification_from_cache_or(ctx, reused, || {
                    let identification = crate::strategy_table::identify_static_query_in_context(
                        identifier_id,
                        graph,
                        &CausalQuery::Response(query.clone()),
                        None,
                        Some(ctx),
                    )?;
                    let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                        CausalError::Compile {
                            message: "response identifier returned no estimand".into(),
                        }
                    })?;
                    Ok((identification, estimand))
                })?
            };
        if identification
            .estimands
            .iter()
            .any(|candidate| candidate.adjustment_set != estimand.adjustment_set)
        {
            return Err(CausalError::Unsupported {
                message: "multi-pair response estimation requires one common adjustment set",
            });
        }

        Ok((identification, estimand, identify_cached, identifier_id, estimator_id))
    }

    /// Result of a response identified by general ID or the front-door, estimated as a discrete plug-in.
    fn finish_general_id_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let (response, _) = estimate_general_id_response(
            data,
            query,
            &identification,
            &estimand,
            FunctionalBootstrap::rows(0),
            ctx,
        )?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(Diagnostic::new(
            "identify.response.general_id",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "intervention mean identified by Shpitser–Pearl ID; levels are the \
             discrete functional.effect plug-in, not an adjustment g-formula",
        ));
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.response.general_id",
                    "estimate.response.general_id",
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Fits the continuous response on the projected table under the study's inference mode.
    fn estimate_continuous_response(
        &self,
        data_est: &TabularData,
        query: &ResponseQuery,
        estimand: &IdentifiedEstimand,
        identification: &IdentificationResult,
        ctx: &ExecutionContext,
    ) -> Result<(CausalResponse, Option<antecedent_estimate::ResponseInfluence>), CausalError> {
        let mut response_estimator =
            ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            response_estimator.options = options.clone();
        }
        let mut response_scores = None;
        let response = if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            if cfg.has_transferred_source() && cfg.prior_mapping.is_none() && cfg.prior.is_none() {
                return Err(CausalError::Unsupported {
                    message: "Bayesian response prior transfer requires a response-specific \
                              mapping (dose/intervention coordinate to outcome-regression \
                              coefficients); incompatible catalogs must not fall back to \
                              isotropic priors",
                });
            }
            let mut bayes = bayesian_gcomp(cfg, ctx);
            if cfg.has_transferred_source() {
                let (outcome, treatments) = response_prior_targets(&query.functional)?;
                let prep = response_estimator
                    .prepare_linear_response_problem(data_est, outcome, &treatments)
                    .map_err(CausalError::from)?;
                let (resolved, _) =
                    crate::inference::resolve_bayesian_prior_with_conflict(cfg, &prep, Some(ctx))?;
                bayes.prior = resolved;
            } else {
                bayes.prior.clone_from(&cfg.prior);
            }
            response_estimator.estimate_bayesian(
                data_est,
                query,
                identification.status,
                identification.required_assumptions.clone(),
                &bayes,
                ctx,
            )
        } else if query.observation == ObservationSpec::Complete {
            let (response, scores) = response_estimator
                .estimate_identified_scored(
                    data_est,
                    query,
                    identification.status,
                    identification.required_assumptions.clone(),
                )
                .map_err(CausalError::from)?;
            response_scores = scores;
            Ok(response)
        } else {
            ObservationMechanismEstimator::new(self.observation_options).estimate_mean_curve(
                &response_estimator,
                data_est,
                query,
                self.observation_delayed_entry,
                identification.status,
                identification.required_assumptions.clone(),
            )
        }
        .map_err(CausalError::from)?;
        let mut response = response;
        if self.graph_posterior.is_none()
            && super::derivative_refutation::is_derivative(&query.functional)
        {
            if let Some(dag) = self.graph.as_dag() {
                super::structural_zero::apply(&mut response, dag, query);
            }
        }
        Ok((response, response_scores))
    }

    /// Plugin-level refutation of a scalar intervention response; notes why it was skipped otherwise.
    fn refute_response_plugin_level(
        &self,
        input: ResponseRefuteInput<'_>,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let ResponseRefuteInput {
            data,
            query,
            estimand,
            estimate,
            scalar,
            estimator_id,
            treatment,
            outcome,
        } = input;
        let scalar_intervention = scalar.is_finite()
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let (refutations, refute_diags) = if scalar_intervention
            && (!matches!(self.settings.refute, RefuteSuite::None)
                || !self.declared.custom_validators.is_empty())
        {
            let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
            let mut refute_ws = EstimationWorkspace::default();
            if matches!(self.settings.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not supported for a plugin intervention level; cheap runs overlap only and full runs overlap plus sampling-stability of the g-comp level. cell.aipw cheap uses the cell-versus-control contrast.",
                ));
            }
            RefutationRun {
                data,
                estimand,
                query: &ate_query,
                estimate,
                suite: self.settings.refute,
                estimator: estimator_id.as_str(),
                custom: &self.declared.custom_validators,
                temporal: None,
            }
            .plugin_level(&mut refute_ws, ctx)?
        } else {
            if !matches!(self.settings.refute, RefuteSuite::None) || !scalar_intervention {
                diagnostics.push(Diagnostic::new(
                    "refute.response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "scalar ATE refuters are not applicable to a function-valued response",
                ));
            }
            (Vec::new(), Vec::new())
        };
        Ok((refutations, refute_diags))
    }
}

pub(super) fn response_scalar_summary(response: &antecedent_core::CausalResponse) -> (f64, f64) {
    let value = match &response.estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => value,
        ResponseIdentification::GraphDependent(_) | ResponseIdentification::Unidentified { .. } => {
            return (f64::NAN, f64::NAN);
        }
    };
    let scalar = match value {
        ResponseValue::Scalar(value) => *value,
        ResponseValue::Surface { .. }
        | ResponseValue::Vector(_)
        | ResponseValue::Jacobian { .. }
        | ResponseValue::Envelope(_) => f64::NAN,
    };
    let se = match response.uncertainty {
        ResponseUncertainty::Scalar { standard_error, .. } => standard_error,
        _ => f64::NAN,
    };
    (scalar, se)
}

pub(super) fn response_primary_pair(
    functional: &ResponseFunctional,
) -> Result<(VariableId, VariableId), CausalError> {
    crate::analysis::proof_support::require_primary_pair(functional)
}

/// Posterior g-computation estimator whose prior is resolved against the linear response
/// problem the ordinary route prepares. A transferred source without a response-specific
/// mapping is refused rather than replaced by an isotropic prior.
pub(super) fn resolved_bayesian_response_estimator(
    config: &BayesianConfig,
    fitter: &ContinuousResponseEstimator,
    data: &TabularData,
    query: &ResponseQuery,
    ctx: &ExecutionContext,
) -> Result<antecedent_estimate::BayesianGComputationAte, CausalError> {
    if config.has_transferred_source() && config.prior_mapping.is_none() && config.prior.is_none() {
        return Err(CausalError::Unsupported {
            message: "Bayesian response prior transfer requires a response-specific \
                      mapping (dose/intervention coordinate to outcome-regression \
                      coefficients); incompatible catalogs must not fall back to \
                      isotropic priors",
        });
    }
    let mut bayes = bayesian_gcomp(config, ctx);
    if config.has_transferred_source() {
        let (outcome, treatments) = response_prior_targets(&query.functional)?;
        let prep = fitter
            .prepare_linear_response_problem(data, outcome, &treatments)
            .map_err(CausalError::from)?;
        let (resolved, _) =
            crate::inference::resolve_bayesian_prior_with_conflict(config, &prep, Some(ctx))?;
        bayes.prior = resolved;
    } else {
        bayes.prior.clone_from(&config.prior);
    }
    Ok(bayes)
}

pub(super) fn response_prior_targets(
    functional: &ResponseFunctional,
) -> Result<(VariableId, Vec<VariableId>), CausalError> {
    match functional {
        ResponseFunctional::MeanCurve { outcome, treatment } => {
            Ok((*outcome, vec![treatment.variable]))
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => {
            let treatments = interventions
                .iter()
                .map(|intervention| {
                    intervention.primary_variable().ok_or_else(|| CausalError::Compile {
                        message: "response prior transfer requires an intervention coordinate"
                            .into(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((*outcome, treatments))
        }
        ResponseFunctional::AverageDerivative { outcome, treatment, .. }
        | ResponseFunctional::PointDerivative { outcome, treatment, .. } => {
            Ok((*outcome, vec![*treatment]))
        }
        ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. }
        | ResponseFunctional::Jacobian { outcomes, treatments, .. } => {
            let outcome = *outcomes.first().ok_or_else(|| CausalError::Compile {
                message: "derivative prior transfer requires an outcome".into(),
            })?;
            Ok((outcome, treatments.to_vec()))
        }
    }
}

pub(crate) fn class_aware_response_supported(query: &ResponseQuery) -> bool {
    query.temporal.is_none()
        && !query.functional.treatment_ids().is_empty()
        && query.observation == ObservationSpec::Complete
        && matches!(
            query.functional,
            ResponseFunctional::MeanCurve { .. }
                | ResponseFunctional::InterventionResponse { .. }
                | ResponseFunctional::PointDerivative { .. }
                | ResponseFunctional::AverageDerivative { .. }
                | ResponseFunctional::DirectionalDerivative { .. }
                | ResponseFunctional::Jacobian { .. }
        )
}

pub(crate) fn response_witness_ate(
    query: &ResponseQuery,
) -> Result<AverageEffectQuery, CausalError> {
    if !class_aware_response_supported(query) {
        return Err(CausalError::Unsupported {
            message: "class-aware response identification requires at least one intervention on complete-observation static data",
        });
    }
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    Ok(AverageEffectQuery::binary_ate(treatment, outcome))
}

mod class_response;

/// What a front-door derivative estimates from: the estimand, its status and its premises.
#[derive(Clone)]
pub(super) struct FrontDoorClaim<'a> {
    pub(super) estimand: &'a IdentifiedEstimand,
    pub(super) status: IdentificationStatus,
    pub(super) assumptions: antecedent_core::AssumptionSet,
}

/// Borrowed inputs shared by every completion of a class-aware response.
#[derive(Clone, Copy)]
struct ClassResponseInputs<'a> {
    data: &'a TabularData,
    data_est: &'a TabularData,
    query: &'a ResponseQuery,
    ctx: &'a ExecutionContext,
}

/// Normalize class masses without redistributing unidentified or unevaluable completions.
fn structural_class_response<G>(
    envelope: &antecedent_identify::IdentificationEnvelope<G>,
    structural_atoms: Vec<crate::result::StructuralResponseAtom>,
    weighted: &[(u64, f64, CausalResponse)],
    total_w: f64,
    unestimated_id_mass: f64,
) -> crate::result::StructuralResponseMixture {
    let total_mass = envelope.identified_weight.0 + envelope.unidentified_weight.0;
    crate::result::StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::CompletionEnumeration,
        atoms: structural_atoms,
        identified_mass: total_w / total_mass.max(f64::EPSILON),
        unidentified_mass: envelope.unidentified_weight.0 / total_mass.max(f64::EPSILON),
        unevaluable_mass: unestimated_id_mass / total_mass.max(f64::EPSILON),
        subsampled_out_mass: 0.0,
        identified_set: response_envelope_from_weighted(weighted),
        identified_set_interval: None,
        conditional_on_identified: None,
        full_mass_scope: envelope.truncated_completions == 0,
        truncated_atoms: envelope.truncated_completions,
        aggregation_policy: None,
    }
}

/// Per-completion results of a class-aware response, in envelope order.
struct ClassResponseAtoms {
    weighted: Vec<(u64, f64, CausalResponse)>,
    structural_atoms: Vec<crate::result::StructuralResponseAtom>,
    atom_scores: Vec<(f64, antecedent_estimate::ResponseInfluence)>,
    total_w: f64,
    unestimated_id_mass: f64,
    unevaluable_general_id: Option<CausalResponse>,
    primary_estimand: Option<IdentifiedEstimand>,
}

impl ClassResponseAtoms {
    /// Preserve unevaluable general-ID support without inventing a numeric mixture.
    fn require_estimable(&mut self, query: &ResponseQuery) -> Result<(), CausalError> {
        if !matches!(self.total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            if let Some(response) = self.unevaluable_general_id.take() {
                // Sole general-ID atom could not evaluate a required cell.
                // Publish the support report; do not invent a number.
                self.weighted.push((0, 1.0, response));
            } else if response_functional_is_derivative(&query.functional)
                && self.unestimated_id_mass > 0.0
            {
                return Err(CausalError::Unsupported {
                    message: "every identified completion identifies the derivative through a \
                              functional that is not an adjustment (front-door or general ID); \
                              no continuous-treatment estimator evaluates such a functional",
                });
            } else {
                return Err(CausalError::Compile {
                    message: "class-aware response envelope had no estimable identified cases"
                        .into(),
                });
            }
        }
        Ok(())
    }

    /// Adds a completion identified by general ID or the front-door, or accounts its lost mass.
    fn add_general_id_case(
        &mut self,
        inputs: &ClassResponseInputs<'_>,
        case_index: usize,
        result: &IdentificationResult,
        weight: f64,
        estimand: IdentifiedEstimand,
    ) -> Result<(), CausalError> {
        if response_functional_is_derivative(&inputs.query.functional) {
            // Identified through a functional that is not an adjustment (front-door or a
            // general ID expression). The discrete plug-in evaluates probabilities at
            // observed levels; it has no derivative with respect to a continuous treatment,
            // so the completion's mass is accounted as unevaluable and never mixed.
            self.unestimated_id_mass += weight;
            if self.primary_estimand.is_none() {
                self.primary_estimand = Some(estimand);
            }
            return Ok(());
        }
        match estimate_general_id_response(
            inputs.data,
            inputs.query,
            result,
            &estimand,
            FunctionalBootstrap::rows(0),
            inputs.ctx,
        ) {
            Ok((response, scores)) => {
                if matches!(response.estimate, ResponseIdentification::Unidentified { .. }) {
                    // Required CPT cell was not evaluable: do not mix a number.
                    self.unestimated_id_mass += weight;
                    if self.primary_estimand.is_none() {
                        self.primary_estimand = Some(estimand);
                    }
                    if self.unevaluable_general_id.is_none() {
                        self.unevaluable_general_id = Some(response);
                    }
                    return Ok(());
                }
                let w = weight;
                self.total_w += w;
                if self.primary_estimand.is_none() {
                    self.primary_estimand = Some(estimand);
                }
                if let Some(scores) = scores {
                    self.atom_scores.push((w, scores));
                }
                self.structural_atoms[case_index].value = response_identified_value(&response);
                self.weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), w, response));
            }
            Err(err) if is_discrete_plug_in_limit(&err) => {
                // Identified by ID, unevaluable with the discrete plug-in.
                // Keep the mass out of the mix and refuse a subset SE.
                self.unestimated_id_mass += weight;
            }
            Err(err) => return Err(err),
        }

        Ok(())
    }
}

/// Whether `err` is the discrete plug-in refusing a variable past its level cap.
///
/// A high-cardinality or continuous column fails closed as that same
/// `EstimationError::Unsupported` message. Any other refusal, including one
/// whose text merely contains "continuous", propagates.
fn is_discrete_plug_in_limit(err: &CausalError) -> bool {
    matches!(
        err.peeled(),
        CausalError::Estimate(antecedent_estimate::EstimationError::Unsupported { message })
            if *message == "variable exceeds discrete level cap for functional.distribution"
    )
}

/// The class-aware response mixed over completions, with the values its disclosures read.
struct MixedClassResponse {
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    response: CausalResponse,
    structural_response: crate::result::StructuralResponseMixture,
    identified_set_unbanded: bool,
    weighted: Vec<(u64, f64, CausalResponse)>,
}

/// The uncertainty every contributing completion shares, when they all
/// return the same estimate and the same published band; `None` when any two
/// differ or none publishes a band.
fn shared_completion_uncertainty(
    weighted: &[(u64, f64, antecedent_core::CausalResponse)],
) -> Option<ResponseUncertainty> {
    let ((_, _, first), rest) = weighted.split_first()?;
    if matches!(first.uncertainty, ResponseUncertainty::None) {
        return None;
    }
    rest.iter()
        .all(|(_, _, r)| r.estimate == first.estimate && r.uncertainty == first.uncertainty)
        .then(|| first.uncertainty.clone())
}

mod mixing;

pub(crate) use mixing::graph_posterior_response_supported;
#[cfg(test)]
use mixing::mix_response_uncertainty;
pub(super) use mixing::{
    attach_response_influence, mix_class_responses, mix_response_influences, mix_response_values,
    mix_support_reports, mixed_response_scalar_se, response_envelope_from_weighted,
    response_identified_value,
};
use mixing::{cell_contrast_for_refute, derivative_identified_set_band};

#[cfg(test)]
mod uncertainty_tests {
    use super::*;

    #[test]
    fn disjoint_atom_intervals_are_not_averaged_into_a_confidence_interval() {
        let a = ResponseUncertainty::Scalar {
            standard_error: 0.1,
            lower: -0.2,
            upper: 0.2,
            level: 0.95,
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        };
        let b = ResponseUncertainty::Scalar {
            standard_error: 0.1,
            lower: 9.8,
            upper: 10.2,
            level: 0.95,
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        };
        assert!(matches!(
            mix_response_uncertainty(&[(0.5, &a), (0.5, &b)]),
            ResponseUncertainty::None
        ));
        assert!(matches!(
            mix_response_uncertainty(&[(1.0, &a)]),
            ResponseUncertainty::Scalar { .. }
        ));
    }
}

/// Normal interval from the front-door plug-in bootstrap SE of a general-ID response, when one is supported.
fn general_id_bootstrap_interval(
    est: &FunctionalEffect,
    prepared: &antecedent_estimate::PreparedFunctionalEffect,
    estimate: &ResponseIdentification,
    support: &mut antecedent_core::SupportReport,
    bootstrap_replicates: u32,
    ctx: &ExecutionContext,
) -> ResponseUncertainty {
    // The general-ID front-door functional exposes no closed-form influence on
    // this route, so its frequentist interval is the normal interval formed from
    // the plug-in bootstrap SE (the same 199-replicate front-door resample the
    // sibling AverageEffect / InterventionalDistribution ADMG cells use). It is
    // published only when a positive-finite SE returns; otherwise the route fails
    // open to `None` rather than panicking.
    let mut uncertainty = ResponseUncertainty::None;
    if bootstrap_replicates > 0 {
        if let ResponseIdentification::PointIdentified(ResponseValue::Scalar(point)) = estimate {
            let point = *point;
            let mut ws = FunctionalDistributionWorkspace::default();
            if let Ok(effect) = est.estimate(prepared, &mut ws, ctx) {
                if let Some(se) = effect.se_bootstrap() {
                    if se.is_finite() && se > 0.0 && point.is_finite() {
                        let z = crate::result::reported_se_interval_z();
                        uncertainty = ResponseUncertainty::Scalar {
                            standard_error: se,
                            lower: point - z * se,
                            upper: point + z * se,
                            level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
                            interpretation: antecedent_core::IntervalInterpretation::Confidence,
                            draws: None,
                        };
                        support.diagnostics.push(antecedent_core::SupportDiagnostic {
                            id: Arc::from(crate::result::RESPONSE_BOOTSTRAP_SE),
                            values: Arc::from([
                                f64::from(bootstrap_replicates),
                                f64::from(effect.uncertainty.bootstrap.replicates_ok.unwrap_or(0)),
                            ]),
                            detail: Arc::from(
                                "general-ID functional.effect response interval is a normal \
                                 interval from the front-door plug-in bootstrap SE \
                                 [requested replicates, successful replicates]",
                            ),
                            scope: antecedent_core::DiagnosticScope::Global,
                        });
                    }
                }
            }
        }
    }
    uncertainty
}

/// The replicate budget and row resampling of a general-ID functional's bootstrap.
#[derive(Clone, Copy, Debug)]
pub(super) struct FunctionalBootstrap {
    pub(super) replicates: u32,
    pub(super) resampling: antecedent_estimate::RowResampling,
    pub(super) design: antecedent_estimate::util::SamplingDeclaration,
}

impl FunctionalBootstrap {
    /// Independent rows, no declared design: the static default.
    pub(super) const fn rows(replicates: u32) -> Self {
        Self {
            replicates,
            resampling: antecedent_estimate::RowResampling::Rows,
            design: antecedent_estimate::util::SamplingDeclaration::Undeclared,
        }
    }
}

impl super::Study {
    /// The functional bootstrap of this study with `replicates` replicates.
    ///
    /// # Errors
    ///
    /// The declared design refuses the draw ([`Self::functional_row_resampling`]).
    pub(super) fn functional_bootstrap(
        &self,
        replicates: u32,
    ) -> Result<FunctionalBootstrap, CausalError> {
        let (resampling, design) = self.functional_row_resampling()?;
        Ok(FunctionalBootstrap { replicates, resampling, design })
    }
}

pub(super) fn estimate_general_id_response(
    data: &TabularData,
    query: &ResponseQuery,
    identification: &antecedent_identify::IdentificationResult,
    estimand: &IdentifiedEstimand,
    bootstrap: FunctionalBootstrap,
    ctx: &ExecutionContext,
) -> Result<(CausalResponse, Option<antecedent_estimate::ResponseInfluence>), CausalError> {
    if !matches!(query.functional, ResponseFunctional::InterventionResponse { .. }) {
        return Err(CausalError::Unsupported {
            message: "general-ID MAG/PAG/DAG response atoms are supported for \
                      InterventionResponse Set means; MeanCurve stays on the \
                      adjustment envelope",
        });
    }
    if !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
        || !matches!(
            query.outcome_functional,
            antecedent_core::OutcomeFunctional::Mean
                | antecedent_core::OutcomeFunctional::Exceedance(_)
        )
    {
        return Err(CausalError::Unsupported {
            message: "general-ID response estimation supports AllObserved scalar mean or exceedance outcomes",
        });
    }
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    let data_est = super::super::helpers::apply_scalar_outcome_functional(
        data,
        outcome,
        &query.outcome_functional,
    )?;
    let bootstrap_replicates = bootstrap.replicates;
    let est = FunctionalEffect { bootstrap_replicates, ..FunctionalEffect::new() }
        .with_row_resampling(bootstrap.resampling, bootstrap.design);
    let prepared = est
        .prepare(
            &data_est,
            estimand,
            &identification.arena,
            identification.required_assumptions.clone(),
            &[treatment, outcome],
        )
        .map_err(CausalError::from)?;
    let eval = prepared.evaluate(&prepared.provider);
    let map_eval = |e: EvalError| {
        CausalError::from(antecedent_estimate::EstimationError::data_msg(e.to_string()))
    };
    let (estimate, mut support) = match eval {
        Ok(ate) => (
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(ate)),
            support_from_functional_eval(None).map_err(map_eval)?,
        ),
        Err(err) if functional_cell_unevaluable(&err) => (
            ResponseIdentification::Unidentified {
                certificate: Arc::from("estimate.response.general_id.unevaluable_cell"),
            },
            support_from_functional_eval(Some(&err)).map_err(map_eval)?,
        ),
        Err(err) => return Err(map_eval(err)),
    };
    if let ResponseFunctional::InterventionResponse { interventions, .. } = &query.functional {
        let levels = interventions
            .iter()
            .map(|intervention| match intervention {
                Intervention::Set { value, .. } => value.as_f64().ok_or(CausalError::Unsupported {
                    message: "general-ID response requires numeric Set levels",
                }),
                _ => Err(CausalError::Unsupported {
                    message: "general-ID response requires Set interventions",
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        support.query_region = antecedent_core::SupportRegion {
            minima: Arc::from(levels.clone()),
            maxima: Arc::from(levels),
        };
    }
    let uncertainty = general_id_bootstrap_interval(
        &est,
        &prepared,
        &estimate,
        &mut support,
        bootstrap_replicates,
        ctx,
    );
    Ok((
        CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate,
            uncertainty,
            support,
            assumptions: identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.response.general_id"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        },
        None,
    ))
}

#[cfg(test)]
mod influence_review_tests {
    use super::*;

    fn influence(columns: Vec<Vec<f64>>) -> antecedent_estimate::ResponseInfluence {
        antecedent_estimate::ResponseInfluence {
            columns,
            row_index: [0u64, 1, 2, 3].map(antecedent_data::GlobalId::new).into(),
        }
    }

    #[test]
    fn surface_response_influence_attaches_no_scalar_standard_error() {
        let scores = influence(vec![vec![1.0, -1.0, 2.0, -2.0], vec![0.5, 0.5, -0.5, -0.5]]);
        let mut surface = EffectEstimate::new(
            f64::NAN,
            f64::NAN,
            antecedent_core::AssumptionSet::new(),
            OverlapPolicy::ExplicitOverride,
        );
        attach_response_influence(&mut surface, Some(&scores)).unwrap();
        assert!(surface.se_analytic().is_nan(), "se {}", surface.se_analytic());
        assert!(surface.joint.covariance.is_some());
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the scalar standard error is copied from the same joint-covariance entry, so it must match bit for bit"
    )]
    fn scalar_response_influence_still_attaches_its_standard_error() {
        let scores = influence(vec![vec![1.0, -1.0, 2.0, -2.0]]);
        let mut scalar = EffectEstimate::new(
            0.3,
            f64::NAN,
            antecedent_core::AssumptionSet::new(),
            OverlapPolicy::ExplicitOverride,
        );
        attach_response_influence(&mut scalar, Some(&scores)).unwrap();
        let expected = scalar.joint.covariance.as_ref().unwrap().se(0);
        assert!(expected.is_finite() && expected > 0.0, "se {expected}");
        assert_eq!(scalar.se_analytic(), expected);
    }

    #[test]
    fn response_mixture_aligns_original_row_ids() {
        let a = antecedent_estimate::ResponseInfluence {
            columns: vec![vec![-1.0, 1.0]],
            row_index: [0u64, 2].map(antecedent_data::GlobalId::new).into(),
        };
        let b = antecedent_estimate::ResponseInfluence {
            columns: vec![vec![-2.0, 2.0]],
            row_index: [1u64, 3].map(antecedent_data::GlobalId::new).into(),
        };
        let mixed = mix_response_influences(&[(0.5, a), (0.5, b)], 4).unwrap();
        assert_eq!(mixed.row_index.iter().map(|id| id.get()).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert_eq!(mixed.columns[0], vec![-1.0, -2.0, 1.0, 2.0]);
    }
}
