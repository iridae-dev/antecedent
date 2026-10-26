// SPDX-License-Identifier: MIT OR Apache-2.0
// Sealed static derivative-response and static DAG response operations, with the Study
// executors that run them. The shared response estimators live in `response_path`.

use super::response_path::{
    attach_response_influence, response_primary_pair, response_prior_targets,
    response_scalar_summary,
};
use super::{
    Arc, AverageEffectQuery, BayesianConfig, CausalError, CausalQuery, CausalResponse,
    CheckedCellAipwResponseOperation, ContinuousResponseEstimator, Dag, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimandMethod, EstimationWorkspace, EstimatorId,
    ExecuteRun, ExecutionContext, IdentificationResult, IdentificationStatus, IdentifiedClaim,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId,
    InferenceMode, Instant, Intervention, ObservationSpec, OverlapPolicy, PhysicalExecutionPlan,
    RefutationRun, RefuteSuite, ResponseFunctional, ResponseQuery, Strategy, StudyResult,
    TabularData, bayesian_gcomp, elapsed_ns, identify_cached_diagnostic, provenance_ids,
};
#[cfg(test)]
use super::{ResponseIdentification, ResponseValue, VariableId};
use crate::analysis::prepared::Inspect;

/// Bytes per materialized derivative coordinate of the frequentist result (value, lower and
/// upper bound).
const DERIVATIVE_CELL_BYTES: u64 = 3 * size_of::<f64>() as u64;

/// Sealed static derivative response plan. It retains the scientific query,
/// identification claim, adjustment basis, selected estimator, and inference
/// procedure together so execution cannot reconstruct a different derivative
/// from a method string after preparation.
///
/// This operation covers the six scalar/vector derivative functionals. Response
/// curves and intervention levels retain their own checked family plans.
#[derive(Clone, Debug)]
pub(crate) struct CheckedDerivativeResponseOperation {
    graph: Dag,
    graph_signature: (usize, Arc<[(u32, u32)]>),
    query: ResponseQuery,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    identifier: IdentifierId,
    estimator: EstimatorId,
    fitter: ContinuousResponseEstimator,
    inference: InferenceMode,
    max_derivative_cells: usize,
}

impl CheckedDerivativeResponseOperation {
    /// Bind a derivative query to its exact identified expression and selected
    /// numerical procedure. The caller supplies the planner's selected IDs;
    /// this constructor verifies those IDs against the functional itself.
    pub(crate) fn checked(
        graph: &Dag,
        query: &ResponseQuery,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
        identifier: IdentifierId,
        estimator: EstimatorId,
        inference: InferenceMode,
        options: antecedent_estimate::ContinuousResponseOptions,
    ) -> Result<Self, CausalError> {
        let (identification, estimand) = (&cache.identification, &cache.estimand);
        query.validate()?;
        let expected_estimator = match &query.functional {
            ResponseFunctional::PointDerivative { .. } => EstimatorId::ResponseKennedyDr,
            ResponseFunctional::AverageDerivative { .. } => EstimatorId::ResponseRieszAde,
            ResponseFunctional::DirectionalDerivative { .. }
            | ResponseFunctional::Jacobian { .. } => EstimatorId::ResponseGamDerivative,
            _ => {
                return Err(CausalError::Unsupported {
                    message: "checked derivative operation requires a derivative response functional",
                });
            }
        };
        let outcome_ids = query.functional.outcome_ids();
        let treatment_ids = query.functional.treatment_ids();
        let max_derivative_cells =
            outcome_ids.len().checked_mul(treatment_ids.len()).ok_or_else(|| {
                CausalError::Compile { message: "derivative response cell count overflow".into() }
            })?;
        // The cell count is admitted against the memory budget when the operation executes
        // (see `admit_cells`); here only an empty derivative is refused.
        if max_derivative_cells == 0 {
            return Err(CausalError::Unsupported {
                message: "checked derivative response names no outcome-treatment coordinate",
            });
        }
        if query.temporal.is_some()
            || query.observation != ObservationSpec::Complete
            || estimator != expected_estimator
            || identification.query != CausalQuery::Response(query.clone())
            || identification.status != IdentificationStatus::NonparametricallyIdentified
            || identification.estimands.is_empty()
            || identification.estimands.first().is_none_or(|candidate| {
                candidate.method != estimand.method
                    || candidate.adjustment_set != estimand.adjustment_set
                    || candidate.instruments != estimand.instruments
                    || candidate.mediators != estimand.mediators
                    || candidate.functional != estimand.functional
                    || candidate.rd_design != estimand.rd_design
            })
            || !matches!(
                estimand.method_kind().ok(),
                Some(EstimandMethod::BackdoorAdjustment | EstimandMethod::BackdoorEfficient)
            )
            || identification
                .estimands
                .iter()
                .any(|candidate| candidate.adjustment_set != estimand.adjustment_set)
            || estimand.instruments.len() > 0
            || estimand.mediators.len() > 0
        {
            return Err(CausalError::Compile {
                message: "derivative response query, identification, estimand, or estimator do not form one checked static adjustment route".into(),
            });
        }
        let fitter = ContinuousResponseEstimator {
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            options,
        };
        Ok(Self {
            graph: graph.clone(),
            graph_signature: dag_signature(graph),
            query: query.clone(),
            cache: Arc::clone(cache),
            identifier,
            estimator,
            fitter,
            inference,
            max_derivative_cells,
        })
    }

    /// Retained query used for execution and artifact inspection.
    pub(crate) fn query(&self) -> &ResponseQuery {
        &self.query
    }

    /// Verify that execution still uses the graph fixed at preparation.
    pub(crate) fn matches_graph(&self, graph: &Dag) -> bool {
        dag_signature(graph) == self.graph_signature
            && dag_signature(&self.graph) == self.graph_signature
    }

    /// Checked identification result and expression target.
    pub(crate) fn target(&self) -> (&IdentificationResult, &IdentifiedEstimand) {
        (&self.cache.identification, &self.cache.estimand)
    }

    /// Selected estimator and identifier identities.
    pub(crate) const fn procedure(&self) -> (IdentifierId, EstimatorId) {
        (self.identifier, self.estimator)
    }

    /// Maximum materialized derivative coordinates for this operation.
    pub(crate) const fn max_derivative_cells(&self) -> usize {
        self.max_derivative_cells
    }

    /// Lease the bytes of the materialized derivative arrays on the context's memory budget
    /// for the duration of the returned lease. A context that declares no budget keeps the
    /// default ceiling of [`antecedent_core::MAX_TEMPORAL_RESPONSE_CELLS`] coordinates; a
    /// declared budget sizes the derivative by itself.
    fn admit_cells(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
        if ctx.memory_limit_bytes().is_none()
            && self.max_derivative_cells > antecedent_core::MAX_TEMPORAL_RESPONSE_CELLS
        {
            return Err(CausalError::Resource {
                message: format!(
                    "derivative response of {} outcome-treatment coordinates exceeds the \
                     default ceiling of {}; declare a memory limit to size the derivative by \
                     its budget",
                    self.max_derivative_cells,
                    antecedent_core::MAX_TEMPORAL_RESPONSE_CELLS
                ),
            });
        }
        let per_cell = match &self.inference {
            InferenceMode::Frequentist => DERIVATIVE_CELL_BYTES,
            InferenceMode::Bayesian(config) => {
                let draws = u64::try_from(config.n_draws).unwrap_or(u64::MAX);
                DERIVATIVE_CELL_BYTES
                    .saturating_add(draws.saturating_mul(size_of::<f64>() as u64))
            }
        };
        let bytes = u64::try_from(self.max_derivative_cells)
            .map_or(u64::MAX, |cells| cells.saturating_mul(per_cell));
        crate::retained_refresh::admit(ctx, bytes, "derivative response")
    }

    /// Execute only the retained target and procedure. This is the common point
    /// at which frequentist and Bayesian derivative procedures diverge.
    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let _admitted = self.admit_cells(ctx)?;
        let identification_status = self.cache.identification.status;
        let assumptions = self.cache.identification.required_assumptions.clone();
        let response = match &self.inference {
            InferenceMode::Frequentist => self
                .fitter
                .estimate_identified(data, &self.query, identification_status, assumptions)
                .map_err(CausalError::from)?,
            InferenceMode::Bayesian(config) => {
                let mut estimator = bayesian_gcomp(config, ctx);
                estimator.prior.clone_from(&config.prior);
                self.fitter
                    .estimate_bayesian(
                        data,
                        &self.query,
                        identification_status,
                        assumptions,
                        &estimator,
                        ctx,
                    )
                    .map_err(CausalError::from)?
            }
        };
        if response.estimand != self.query.functional {
            return Err(CausalError::Compile {
                message: "derivative estimator returned a response for a different functional"
                    .into(),
            });
        }
        Ok(response)
    }
}

fn dag_signature(graph: &Dag) -> (usize, Arc<[(u32, u32)]>) {
    (graph.node_count(), graph.directed_edge_pairs_sorted().into())
}

/// Checked execution for static DAG response curves and scalar intervention
/// g-computation, frequentist or Bayesian. Grid coordinates or intervention
/// values remain attached to the exact response query through preparation and
/// execution, and the inference procedure is fixed with them.
#[derive(Clone, Debug)]
pub(crate) struct CheckedStaticDagResponseOperation {
    graph: Dag,
    graph_signature: (usize, Arc<[(u32, u32)]>),
    query: ResponseQuery,
    curve_grid: Arc<[f64]>,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    identifier: IdentifierId,
    estimator: EstimatorId,
    fitter: ContinuousResponseEstimator,
    inference: InferenceMode,
    refute: RefuteSuite,
}

/// The frozen premises of [`CheckedStaticDagResponseOperation::checked`], carried together.
pub(crate) struct CheckedStaticDagResponseOperationPremises<'a> {
    pub(crate) graph: &'a Dag,
    pub(crate) query: &'a ResponseQuery,
    pub(crate) cache: &'a Arc<crate::analysis::prepared::CachedStaticIdentification>,
    pub(crate) identifier: IdentifierId,
    pub(crate) estimator: EstimatorId,
    pub(crate) options: antecedent_estimate::ContinuousResponseOptions,
    pub(crate) inference: InferenceMode,
    pub(crate) refute: RefuteSuite,
}

impl CheckedStaticDagResponseOperation {
    /// Check and retain an explicit or accepted static DAG response route.
    pub(crate) fn checked(
        premises: CheckedStaticDagResponseOperationPremises<'_>,
    ) -> Result<Self, CausalError> {
        let CheckedStaticDagResponseOperationPremises {
            graph,
            query,
            cache,
            identifier,
            estimator,
            options,
            inference,
            refute,
        } = premises;
        let (identification, estimand) = (&cache.identification, &cache.estimand);
        query.validate()?;
        let curve_grid = Self::checked_curve_grid(query)?;
        let expected_estimator = match (&query.functional, &inference) {
            (_, InferenceMode::Bayesian(_)) => EstimatorId::ResponseBayesian,
            (ResponseFunctional::MeanCurve { .. }, InferenceMode::Frequentist) => {
                EstimatorId::ResponseKennedyDr
            }
            (ResponseFunctional::InterventionResponse { .. }, InferenceMode::Frequentist) => {
                EstimatorId::ResponseInterventionGcomp
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "checked response execution does not cover this functional",
                });
            }
        };
        // A Bayesian intervention level publishes a posterior; its cheap/full suites run
        // the Bayesian refuter state (plugin-level suite on the posterior-mean level plus
        // the predictive checks and, under full, level-native prior sensitivity), which
        // executes unlicensed.
        let admissible_refuter = match query.functional {
            ResponseFunctional::MeanCurve { .. } => refute == RefuteSuite::None,
            ResponseFunctional::InterventionResponse { .. } => {
                matches!(
                    refute,
                    RefuteSuite::None
                        | RefuteSuite::Cheap
                        | RefuteSuite::PlaceboAndRcc
                        | RefuteSuite::Full
                )
            }
            _ => false,
        };
        let scalar_outcome =
            matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean);
        if query.temporal.is_some()
            || query.observation != ObservationSpec::Complete
            || query.target_population != antecedent_core::TargetPopulation::AllObserved
            || !scalar_outcome
            || identifier != crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER_ID
            || estimator != expected_estimator
            || !admissible_refuter
            || identification.query != CausalQuery::Response(query.clone())
            || identification.status != IdentificationStatus::NonparametricallyIdentified
            || identification.estimands.is_empty()
            || identification.estimands.first().is_none_or(|candidate| {
                candidate.method != estimand.method
                    || candidate.adjustment_set != estimand.adjustment_set
                    || candidate.instruments != estimand.instruments
                    || candidate.mediators != estimand.mediators
                    || candidate.functional != estimand.functional
                    || candidate.rd_design != estimand.rd_design
            })
            || !matches!(
                estimand.method_kind().ok(),
                Some(EstimandMethod::BackdoorAdjustment | EstimandMethod::BackdoorEfficient)
            )
            || identification
                .estimands
                .iter()
                .any(|candidate| candidate.adjustment_set != estimand.adjustment_set)
        {
            return Err(CausalError::Compile {
                message: "static DAG response query, target, estimator, or validation suite do not form one checked route".into(),
            });
        }
        let fitter = ContinuousResponseEstimator {
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            options,
        };
        Ok(Self {
            graph: graph.clone(),
            graph_signature: dag_signature(graph),
            query: query.clone(),
            curve_grid,
            cache: Arc::clone(cache),
            identifier,
            estimator,
            fitter,
            inference,
            refute,
        })
    }

    /// Curve grid a checked static DAG response retains, refusing unsupported functionals.
    fn checked_curve_grid(query: &ResponseQuery) -> Result<Arc<[f64]>, CausalError> {
        match &query.functional {
            ResponseFunctional::MeanCurve { treatment, .. } => Ok(treatment.grid.values()?.into()),
            ResponseFunctional::InterventionResponse { interventions, .. } => {
                if interventions.is_empty()
                    || interventions.iter().any(|intervention| {
                        matches!(
                            intervention,
                            Intervention::Soft { .. } | Intervention::Sequence(_)
                        )
                    })
                {
                    return Err(CausalError::Unsupported {
                        message: "checked intervention g-computation requires hard set, shift, or stochastic policies",
                    });
                }
                Ok(Arc::from([]))
            }
            _ => Err(CausalError::Unsupported {
                message: "checked static DAG response operation supports MeanCurve or InterventionResponse",
            }),
        }
    }

    /// Inference procedure fixed by preparation.
    pub(crate) fn inference(&self) -> &InferenceMode {
        &self.inference
    }

    /// Response fitter fixed to the retained adjustment set and response options.
    pub(crate) fn fitter(&self) -> &ContinuousResponseEstimator {
        &self.fitter
    }

    /// Retained response query, including full grid or intervention values.
    pub(crate) fn query(&self) -> &ResponseQuery {
        &self.query
    }

    /// Verify graph identity at dispatch and refresh boundaries.
    pub(crate) fn matches_graph(&self, graph: &Dag) -> bool {
        dag_signature(graph) == self.graph_signature
            && dag_signature(&self.graph) == self.graph_signature
    }

    /// Ordered curve coordinates; empty for an intervention query.
    pub(crate) fn curve_grid(&self) -> &[f64] {
        &self.curve_grid
    }

    /// Retained identification and selected target.
    pub(crate) fn target(&self) -> (&IdentificationResult, &IdentifiedEstimand) {
        (&self.cache.identification, &self.cache.estimand)
    }

    /// Procedure and validation identities fixed by preparation.
    pub(crate) const fn procedure(&self) -> (IdentifierId, EstimatorId, RefuteSuite) {
        (self.identifier, self.estimator, self.refute)
    }

    /// Execute the exact retained response functional.
    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<(CausalResponse, Option<antecedent_estimate::ResponseInfluence>), CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
            });
        }
        if !self.curve_grid.is_empty() {
            // A curve result materializes the grid, mean, and interval endpoints,
            // plus estimator-side work arrays. Budget five f64 values per member;
            // this deliberately overstates only the fixed output arrays and gives
            // a deterministic guard before fitting begins.
            let bytes = self
                .curve_grid
                .len()
                .checked_mul(5 * std::mem::size_of::<f64>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| CausalError::Resource {
                    message: "response curve memory estimate overflowed".into(),
                })?;
            let limit = ctx.memory_limit_bytes();
            if limit.is_some_and(|limit| bytes > limit) {
                return Err(CausalError::Resource {
                    message: format!(
                        "response curve needs at least {bytes} bytes for {0} grid members, above the configured memory budget",
                        self.curve_grid.len()
                    ),
                });
            }
        }
        let (response, scores) = match &self.inference {
            InferenceMode::Frequentist => self
                .fitter
                .estimate_identified_scored(
                    data,
                    &self.query,
                    self.cache.identification.status,
                    self.cache.identification.required_assumptions.clone(),
                )
                .map_err(CausalError::from)?,
            InferenceMode::Bayesian(config) => {
                let response = self.execute_bayesian(data, config, ctx)?;
                (response, None)
            }
        };
        if response.estimand != self.query.functional {
            return Err(CausalError::Compile {
                message: "response estimator returned a different functional than the sealed query"
                    .into(),
            });
        }
        if let ResponseFunctional::MeanCurve { treatment, .. } = &self.query.functional {
            let current = treatment.grid.values()?;
            if current.as_slice() != self.curve_grid.as_ref() {
                return Err(CausalError::Compile {
                    message: "retained response grid no longer matches the checked curve members"
                        .into(),
                });
            }
        }
        Ok((response, scores))
    }

    /// Posterior g-computation over the retained adjustment target. Prior
    /// transfer resolves against the same linear response problem the ordinary
    /// route prepares, and an artifact or catalog without a response-specific
    /// mapping is refused rather than replaced by an isotropic prior.
    fn execute_bayesian(
        &self,
        data: &TabularData,
        config: &BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, CausalError> {
        if (config.has_transferred_source())
            && config.prior_mapping.is_none()
            && config.prior.is_none()
        {
            return Err(CausalError::Unsupported {
                message: "Bayesian response prior transfer requires a response-specific \
                          mapping (dose/intervention coordinate to outcome-regression \
                          coefficients); incompatible catalogs must not fall back to \
                          isotropic priors",
            });
        }
        let mut bayes = bayesian_gcomp(config, ctx);
        if config.has_transferred_source() {
            let (outcome, treatments) = response_prior_targets(&self.query.functional)?;
            let prep = self
                .fitter
                .prepare_linear_response_problem(data, outcome, &treatments)
                .map_err(CausalError::from)?;
            let (resolved, _) =
                crate::inference::resolve_bayesian_prior_with_conflict(config, &prep, Some(ctx))?;
            bayes.prior = resolved;
        } else {
            bayes.prior.clone_from(&config.prior);
        }
        self.fitter
            .estimate_bayesian(
                data,
                &self.query,
                self.cache.identification.status,
                self.cache.identification.required_assumptions.clone(),
                &bayes,
                ctx,
            )
            .map_err(CausalError::from)
    }
}

impl super::Study {
    /// Execute a retained derivative-response operation without consulting the
    /// study query, estimator string, or a fresh identification pass.
    pub(crate) fn execute_checked_derivative_response(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        operation: &CheckedDerivativeResponseOperation,
    ) -> Result<StudyResult, CausalError> {
        if self.query != CausalQuery::Response(operation.query().clone())
            || self.graph.as_dag().is_none_or(|graph| !operation.matches_graph(graph))
        {
            return Err(CausalError::Compile {
                message: "prepared derivative operation no longer matches the study query or graph"
                    .into(),
            });
        }
        let started = Instant::now();
        let response = operation.execute(data, ctx)?;
        let (identification, estimand) = operation.target();
        let (treatment, outcome) = response_primary_pair(&operation.query().functional)?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let (identifier_id, estimator_id) = operation.procedure();
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(identify_cached_diagnostic());
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued; the scalar effect summary is not applicable and the result is carried by the response payload",
            ));
        }
        diagnostics.extend(response.support.warnings.iter().cloned());
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification.clone(),
                estimand.clone(),
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(true)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Execute a retained static DAG mean curve or intervention g-computation
    /// plan. This adapter deliberately does not resolve IDs from the builder or
    /// re-run identification.
    #[allow(
        clippy::too_many_lines,
        reason = "one retained static response plan: the curve or level fit, its uncertainty and the identified finish run in sequence"
    )]
    pub(crate) fn execute_checked_static_dag_response(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        operation: &CheckedStaticDagResponseOperation,
    ) -> Result<StudyResult, CausalError> {
        if self.query != CausalQuery::Response(operation.query().clone())
            || self.graph.as_dag().is_none_or(|graph| !operation.matches_graph(graph))
        {
            return Err(CausalError::Compile {
                message:
                    "prepared static response operation no longer matches the study query or graph"
                        .into(),
            });
        }
        let started = Instant::now();
        let (response, scores) = operation.execute(data, ctx)?;
        let (identification, estimand) = operation.target();
        let query = operation.query();
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let mut estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        attach_response_influence(&mut estimate, scores.as_ref())?;
        let (identifier_id, estimator_id, refute) = operation.procedure();
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(identify_cached_diagnostic());
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued; the scalar effect summary is not applicable and the result is carried by the response payload",
            ));
        }
        diagnostics.extend(response.support.warnings.iter().cloned());

        let scalar_intervention = scalar.is_finite()
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let mut predictive_checks = Vec::new();
        let (refutations, refute_diagnostics) = if let (true, InferenceMode::Bayesian(config)) = (
            scalar_intervention && matches!(refute, RefuteSuite::Cheap | RefuteSuite::Full),
            operation.inference(),
        ) {
            let estimator = super::bayesian_level_refutation::bayesian_response_estimator_in_force(
                operation.fitter(),
                data,
                query,
                config,
                ctx,
            )?;
            let state = super::bayesian_level_refutation::refute_bayesian_intervention_level(
                &super::bayesian_level_refutation::BayesianLevelRefutationInput {
                    data,
                    query,
                    estimand,
                    fitter: operation.fitter(),
                    estimator: &estimator,
                    status: identification.status,
                    assumptions: &identification.required_assumptions,
                    level: scalar,
                    level_sd: standard_error,
                    treatment,
                    outcome,
                    suite: refute,
                    latency: self.latency_mode,
                    plugin_level: true,
                    summarize: response_scalar_summary,
                    target_label: "intervention-level",
                },
                ctx,
            )?;
            predictive_checks = state.predictive_checks;
            (state.refutations, state.diagnostics)
        } else if scalar_intervention && matches!(refute, RefuteSuite::Cheap | RefuteSuite::Full) {
            let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
            let mut workspace = EstimationWorkspace::default();
            if matches!(refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not licensed for a plugin intervention level; cheap runs overlap only and full runs overlap plus sampling-stability of the g-computation level",
                ));
            }
            RefutationRun {
                data,
                estimand,
                query: &ate_query,
                estimate: &estimate,
                suite: refute,
                estimator: estimator_id.as_str(),
                custom: &[],
                temporal: None,
            }
            .plugin_level(&mut workspace, ctx)?
        } else {
            (Vec::new(), Vec::new())
        };
        diagnostics.extend(refute_diagnostics);
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification.clone(),
                estimand.clone(),
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(true)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    pub(crate) fn execute_checked_cell_aipw_response(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        operation: &CheckedCellAipwResponseOperation,
    ) -> Result<StudyResult, CausalError> {
        if self.query != CausalQuery::Response(operation.query().clone())
            || !operation.matches_structure(&self.graph, self.tiered.as_ref())
        {
            return Err(CausalError::Compile {
                message: "prepared cell AIPW operation no longer matches the study query, graph, or tier closure"
                    .into(),
            });
        }
        let started = Instant::now();
        let (identification, estimand) = operation.target();
        let (identifier_id, estimator_id) = operation.procedure();
        let table = operation.execute(data, ctx)?;
        self.execute_cell_aipw_response(
            data,
            operation.query(),
            ExecuteRun { physical, ctx, started, identify_cached: true },
            IdentifiedClaim { identification: identification.clone(), estimand: estimand.clone() },
            Strategy::new(identifier_id, estimator_id),
            Some(table),
        )
    }
}

/// Read-only inspection of a retained checked derivative-response plan.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedDerivativeResponseInfo {
    /// Exact response functional and evaluation coordinates.
    pub query: antecedent_core::ResponseQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Adjustment variables retained by the checked target.
    pub adjustment_set: std::sync::Arc<[antecedent_core::VariableId]>,
    /// Maximum derivative coordinates the operation may materialize.
    pub max_derivative_cells: usize,
}

impl Inspect for CheckedDerivativeResponseOperation {
    type Info = CheckedDerivativeResponseInfo;

    fn info(&self) -> CheckedDerivativeResponseInfo {
        let (_, estimand) = self.target();
        let (identifier, estimator) = self.procedure();
        CheckedDerivativeResponseInfo {
            query: self.query().clone(),
            identifier,
            estimator,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            max_derivative_cells: self.max_derivative_cells(),
        }
    }
}

/// Read-only inspection of a retained static DAG response operation.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedStaticDagResponseInfo {
    /// Exact mean-curve or intervention query, including its member values.
    pub query: antecedent_core::ResponseQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Ordered curve grid. Empty for intervention-response queries.
    pub grid_members: std::sync::Arc<[f64]>,
    /// Inference procedure label fixed during preparation
    /// (`frequentist` or `bayesian:<backend>`).
    pub inference: std::sync::Arc<str>,
}

impl Inspect for CheckedStaticDagResponseOperation {
    type Info = CheckedStaticDagResponseInfo;

    fn info(&self) -> CheckedStaticDagResponseInfo {
        let (identifier, estimator, validation) = self.procedure();
        CheckedStaticDagResponseInfo {
            query: self.query().clone(),
            identifier,
            estimator,
            validation,
            grid_members: Arc::from(self.curve_grid()),
            inference: crate::analysis::route_guards::inference_label(self.inference()),
        }
    }
}

#[cfg(test)]
mod checked_derivative_operation_tests {
    fn cached(
        identification: &IdentificationResult,
        estimand: &IdentifiedEstimand,
    ) -> Arc<crate::analysis::prepared::CachedStaticIdentification> {
        Arc::new(crate::analysis::prepared::CachedStaticIdentification {
            identification: identification.clone(),
            estimand: estimand.clone(),
        })
    }
    use super::*;
    use antecedent_core::{DerivativeScale, ResponseFunctional};
    use antecedent_graph::{Dag, DenseNodeId};

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "builds the linear truth fixture, seals the derivative operation and asserts the point, the refused rebind and the bound query"
    )]
    fn sealed_point_derivative_matches_independent_linear_truth_and_binds_query() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let functional = ResponseFunctional::PointDerivative {
            outcome,
            treatment,
            at: 0.25,
            order: 1,
            scale: DerivativeScale::Identity,
        };
        let query = ResponseQuery::new(functional);
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let identifier = crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER_ID;
        let identification = crate::strategy_table::identify_static_query(
            identifier,
            &graph,
            &CausalQuery::Response(query.clone()),
        )
        .unwrap();
        let estimand = identification.estimands.first().unwrap().clone();
        let estimator = EstimatorId::ResponseKennedyDr;
        let options = antecedent_estimate::ContinuousResponseOptions {
            bandwidth: Some(0.35),
            ..Default::default()
        };
        let operation = CheckedDerivativeResponseOperation::checked(
            &graph,
            &query,
            &cached(&identification, &estimand),
            identifier,
            estimator,
            InferenceMode::Frequentist,
            options.clone(),
        )
        .unwrap();

        let a: Vec<_> = (0..401).map(|i| -2.0 + f64::from(i) / 100.0).collect();
        let y: Vec<_> = a.iter().map(|a| 7.0 + 2.5 * a).collect();
        let data =
            TabularData::from_f64_columns([("a", a.as_slice()), ("y", y.as_slice())]).unwrap();
        let response = operation.execute(&data, &ExecutionContext::for_tests(9)).unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(estimate)) =
            response.estimate
        else {
            panic!("expected a point-identified scalar derivative")
        };
        assert!((estimate - 2.5).abs() < 0.08, "estimated derivative {estimate}");
        assert_eq!(operation.query(), &query);
        assert!(operation.matches_graph(&graph));
        assert!(!operation.matches_graph(&Dag::with_variables(2)));
        assert_eq!(operation.target().1.functional, estimand.functional);
        assert_eq!(operation.procedure(), (identifier, estimator));
        assert_eq!(operation.max_derivative_cells(), 1);

        let changed = ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome,
            treatment,
            at: 0.75,
            order: 1,
            scale: DerivativeScale::Identity,
        });
        assert!(
            CheckedDerivativeResponseOperation::checked(
                &graph,
                &changed,
                &cached(&identification, &estimand),
                identifier,
                estimator,
                InferenceMode::Frequentist,
                options.clone(),
            )
            .is_err()
        );

        assert!(
            CheckedDerivativeResponseOperation::checked(
                &graph,
                &query,
                &cached(&identification, &estimand),
                identifier,
                EstimatorId::ResponseRieszAde,
                InferenceMode::Frequentist,
                options.clone(),
            )
            .is_err()
        );

        let altered_target = IdentifiedEstimand::new(
            estimand.method.clone(),
            Arc::clone(&estimand.adjustment_set),
            Arc::clone(&estimand.instruments),
            Arc::clone(&estimand.mediators),
            antecedent_expr::ExprId::from_raw(u32::MAX),
            estimand.rd_design,
        );
        assert!(
            CheckedDerivativeResponseOperation::checked(
                &graph,
                &query,
                &cached(&identification, &altered_target),
                identifier,
                estimator,
                InferenceMode::Frequentist,
                options,
            )
            .is_err()
        );
    }

    #[test]
    fn derivative_plan_defers_cell_ceiling_to_the_memory_budget() {
        let outcomes: Arc<[VariableId]> = (0..1_001).map(VariableId::from_raw).collect();
        let treatments: Arc<[VariableId]> = (1_001..2_002).map(VariableId::from_raw).collect();
        let at: Arc<[f64]> = vec![0.0; treatments.len()].into();
        let query = ResponseQuery::new(ResponseFunctional::Jacobian {
            outcomes,
            treatments,
            at,
            scale: antecedent_core::DerivativeScale::Identity,
        });
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let identifier = crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER_ID;
        let mut identification = crate::strategy_table::identify_static_query(
            identifier,
            &graph,
            &CausalQuery::Response(ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: VariableId::from_raw(1),
                treatment: VariableId::from_raw(0),
                at: 0.0,
                order: 1,
                scale: antecedent_core::DerivativeScale::Identity,
            })),
        )
        .unwrap();
        // The grid names variables outside the two-node graph, so the identification is the
        // two-node claim restated for the grid query the operation seals.
        identification.query = CausalQuery::Response(query.clone());
        let estimand = identification.estimands.first().unwrap();
        let operation = CheckedDerivativeResponseOperation::checked(
            &graph,
            &query,
            &cached(&identification, estimand),
            identifier,
            EstimatorId::ResponseGamDerivative,
            InferenceMode::Frequentist,
            antecedent_estimate::ContinuousResponseOptions::default(),
        )
        .unwrap();
        assert_eq!(operation.max_derivative_cells(), 1_001 * 1_001);
        // With no declared budget the grid is over the default ceiling at execution.
        let data = TabularData::from_f64_columns([("a", [0.0, 1.0].as_slice())]).unwrap();
        let error = operation.execute(&data, &ExecutionContext::for_tests(9)).unwrap_err();
        assert!(error.to_string().contains("default ceiling"), "{error}");
    }
}
