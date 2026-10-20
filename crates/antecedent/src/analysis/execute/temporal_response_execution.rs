//! Full result assembly for a sealed temporal DAG mean response operation.
//!
//! A response with a declared feedback mechanism publishes the finite-horizon
//! response of the history, initial-state, and transition operators at every
//! requested horizon. That value is not the fitted dose-by-horizon surface. A
//! response with no declared feedback mechanism publishes the fitted surface,
//! whatever the action lag.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalQuery, CausalResponse, DerivativeSpec, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, ExecutionContext, IdentificationResult, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedResultContext, InferenceMode,
    Instant, OverlapPolicy, PhysicalExecutionPlan, Strategy, StudyResult, TableView,
    TimeSeriesData, bayesian_gcomp, elapsed_ns, provenance_ids, push_conflict_diagnostics,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::require_primary_pair;
use antecedent_core::{HistoryWindow, IdentificationStatus, TemporalPolicy};
use antecedent_data::TimeIndex;
use antecedent_model::history_operators::{
    DelayedAction, DelayedDynamics, DelayedFeedback, DelayedFeedbackRequest, FeedbackDelay,
    FeedbackSearchCap, HistoryOperatorError, ResponseTarget, TargetResponse, TruncationCertificate,
    horizon_ending_after,
};

/// Checked operation together with the immutable logical/physical result context.
///
/// `execute` consumes only this retained context, series data, and the execution
/// context. It never consults a `Study` builder to recover a query or procedure.
#[derive(Clone)]
pub(crate) struct CheckedTemporalResponseExecution {
    operation: crate::analysis::CheckedTemporalResponseOperation,
    result_context: IdentifiedResultContext,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    feedback: Option<FeedbackMechanism>,
}

/// Feedback mechanism declared for a response.
///
/// The delay is the feedback delay: ticks before the action re-enters the
/// state. It is not the action lag, which is the negative start of the policy.
/// The level is the dose the pulse or sustained policy holds. The dynamics are
/// the fitted transition and carry the initial law the frame binds.
#[derive(Clone, Debug)]
pub(crate) struct FeedbackMechanism {
    delay: FeedbackDelay,
    level: f64,
    dynamics: DelayedDynamics,
}

impl FeedbackMechanism {
    /// Declare a feedback delay of `delay_ticks`, held `level`, and `dynamics`.
    ///
    /// A zero delay is contemporaneous transmission and a non-finite level is
    /// not a dose; both are refused.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn declared(
        delay_ticks: u32,
        level: f64,
        dynamics: DelayedDynamics,
    ) -> Result<Self, HistoryOperatorError> {
        DelayedAction::pulse(0, level)?;
        Ok(Self { delay: FeedbackDelay::delayed(delay_ticks)?, level, dynamics })
    }
}

impl std::fmt::Debug for CheckedTemporalResponseExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedTemporalResponseExecution")
            .field("operation", &self.operation)
            .field("physical", &self.physical)
            .finish_non_exhaustive()
    }
}

impl CheckedTemporalResponseExecution {
    /// Seal a response operation to its exact public result and physical plans.
    pub(crate) fn checked(
        operation: crate::analysis::CheckedTemporalResponseOperation,
        result_context: IdentifiedResultContext,
        physical: std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let query = CausalQuery::Response(operation.query().clone());
        if result_context.query != query || physical.logical.query != query {
            return Err(CausalError::Compile {
                message: "temporal response result context does not match its retained query"
                    .into(),
            });
        }
        let (status, assumptions) = super::aggregate_temporal_horizon_evidence(
            operation
                .evidence()
                .iter()
                .map(crate::analysis::TemporalResponseHorizonEvidence::identification),
        )?;
        let mut identification = operation.primary_identification()?.clone();
        identification.status = status;
        identification.required_assumptions = assumptions;
        let estimand = operation.primary_estimand()?.clone();
        Ok(Self { operation, result_context, physical, identification, estimand, feedback: None })
    }

    /// Attach the feedback mechanism the response is declared to have.
    ///
    /// Without one the fitted dose-by-horizon surface is published.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn with_feedback(mut self, feedback: FeedbackMechanism) -> Self {
        self.feedback = Some(feedback);
        self
    }

    /// Retained response plan for refresh validation and descriptor inspection.
    #[must_use]
    pub(crate) fn operation(&self) -> &crate::analysis::CheckedTemporalResponseOperation {
        &self.operation
    }

    /// The response surface of the sealed operation: the local slope of the
    /// linear fit for a derivative functional, otherwise the operation's
    /// frequentist or Bayesian estimate with its prior-conflict summary.
    fn execute_response_surface(
        &self,
        data: &TimeSeriesData,
        temporal: &antecedent_core::TemporalResponseSpec,
        (treatment, outcome): (antecedent_core::VariableId, antecedent_core::VariableId),
        ctx: &ExecutionContext,
    ) -> Result<(CausalResponse, Option<antecedent_prob::ConflictSummary>), CausalError> {
        if super::response_functional_is_derivative(&self.operation.query().functional) {
            return Ok((self.derivative_surface(data, ctx)?, None));
        }
        Ok(match self.operation.inference() {
            InferenceMode::Frequentist => (self.operation.execute(data, ctx)?, None),
            InferenceMode::Bayesian(cfg) => {
                let mut bayes = bayesian_gcomp(cfg, ctx);
                let aligned = self
                    .operation
                    .evidence()
                    .iter()
                    .map(crate::analysis::TemporalResponseHorizonEvidence::cached_horizon)
                    .collect::<Vec<_>>();
                let aligned_refs = aligned.iter().collect::<Vec<_>>();
                let (resolved, conflict) = super::resolve_temporal_response_prior(
                    cfg,
                    data,
                    temporal,
                    &aligned_refs,
                    treatment,
                    outcome,
                    ctx,
                )?;
                bayes.prior = resolved;
                (self.operation.execute_bayesian(data, &bayes, ctx)?, conflict)
            }
        })
    }

    /// The local slope of the linear unfolded fit at the single requested
    /// horizon.
    fn derivative_surface(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, CausalError> {
        let evidence = self.operation.evidence();
        let first = evidence.first().ok_or_else(|| CausalError::Compile {
            message: "temporal derivative is missing its horizon proof".into(),
        })?;
        if evidence.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon",
            });
        }
        let prior_scale = match self.operation.inference() {
            InferenceMode::Bayesian(cfg) => {
                if cfg.prior.is_some()
                    || cfg.prior_artifact.is_some()
                    || cfg.external_compose.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "temporal derivative uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed",
                    });
                }
                Some(cfg.prior_scale)
            }
            InferenceMode::Frequentist => None,
        };
        super::temporal_linear_derivative_response(
            data,
            self.operation.query(),
            first.estimand(),
            first.indexer(),
            first.identification(),
            DerivativeSpec {
                treatment_offset: first.effect_query().try_treatment_offset()?,
                outcome_offset: first.effect_query().outcome_offset(),
                policy: &ctx.kernel_policy,
                prior_scale,
            },
        )
    }

    /// The diagnostics of the sealed response, in publication order.
    fn response_diagnostics(
        &self,
        response: &CausalResponse,
        scalar: f64,
        conflict_summary: Option<&antecedent_prob::ConflictSummary>,
    ) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        for member in self.operation.evidence() {
            diagnostics.extend(member.identification().diagnostics.iter().cloned());
        }
        if let Some(summary) = conflict_summary {
            push_conflict_diagnostics(&mut diagnostics, summary);
        }
        if horizons_use_different_adjustments(self.operation.evidence()) {
            diagnostics.push(Diagnostic::new(
                "identify.temporal_response.horizon_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "adjustment sets differ across requested horizons; each response slice uses its own horizon-specific proof",
            ));
        }
        if response.provenance_id.as_ref() == "estimate.temporal.linear_derivative" {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal.linear_derivative",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                "published value is the slope of the linear unfolded temporal fit; no interval is published",
            ));
        }
        if response.provenance_id.as_ref() == "model.delayed_feedback.finite_horizon" {
            diagnostics.push(Diagnostic::new(
                "model.delayed_feedback.finite_horizon",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                "finite-horizon response of the history, initial state, and transition",
            ));
        }
        diagnostics.push(Diagnostic::new(
            "refute.temporal_response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters do not apply to a function-valued temporal response",
        ));
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this function-valued response is carried by its dose-by-horizon response payload",
            ));
        }
        diagnostics.extend(response.support.warnings.iter().cloned());
        diagnostics
    }

    /// Execute and assemble the ordinary, portable `StudyResult` contract.
    pub(crate) fn execute(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let temporal =
            self.operation.query().temporal.as_ref().ok_or_else(|| CausalError::Compile {
                message: "checked temporal response lost its temporal specification".into(),
            })?;
        super::enforce_temporal_response_memory_budget(
            self.operation.query(),
            temporal,
            self.operation.procedure().bootstrap_replicates,
            ctx,
        )?;
        let (treatment, outcome) = require_primary_pair(&self.operation.query().functional)?;
        let (mut response, conflict_summary) = if let Some(mechanism) = &self.feedback {
            let searched_steps = self
                .operation
                .evidence()
                .iter()
                .map(|member| member.indexer().history())
                .max()
                .unwrap_or(0);
            let levels =
                finite_horizon_levels(data.time_index(), temporal, searched_steps, mechanism)?;
            let response = finite_horizon_causal_response(
                &self.operation.query().functional,
                &self.identification,
                &levels,
            );
            (response, None)
        } else {
            self.execute_response_surface(data, temporal, (treatment, outcome), ctx)?
        };
        // The estimator accepts one aggregate status; the sealed route keeps the
        // richer per-horizon proof statuses on the surface, as the fresh path does.
        if let Some(per_horizon) = response.horizon_identification.as_mut() {
            for (record, member) in
                Arc::make_mut(per_horizon).iter_mut().zip(self.operation.evidence())
            {
                record.status = member.identification().status;
            }
        }
        let (scalar, standard_error) =
            super::super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(lag_aligned_rows(data, self.operation.evidence()));
        let diagnostics = self.response_diagnostics(&response, scalar, conflict_summary.as_ref());
        let procedure = self.operation.procedure();
        let (identifier_id, estimator_id) = (procedure.identifier, procedure.estimator);
        let response_provenance = Arc::clone(&response.provenance_id);
        let args = IdentifiedExecuteFinish::new(
            &self.physical,
            self.identification.clone(),
            self.estimand.clone(),
            estimate,
            Strategy::new(identifier_id, estimator_id),
            treatment,
            outcome,
        )
        // The immutable horizon proofs are captured at prepare time and used
        // as-is by every execution/refresh of this sealed route.
        .identify_cached(true)
        .wall_time_ns(elapsed_ns(started))
        .cancelled(ctx.cancellation.is_cancelled())
        .extras(IdentifiedExecuteExtras {
            identify_provenance: Some(provenance_ids(
                "identify.temporal_backdoor",
                "identify.temporal.backdoor.unfolded",
            )),
            estimate_provenance: Some(provenance_ids(
                Arc::clone(&response_provenance),
                Arc::clone(&response_provenance),
            )),
            diagnostics: Some(diagnostics),
            response: Some(response),
            // Scalar bootstrap metadata is not emitted for response-family
            // intervals; the response carries the actual block procedure,
            // survivor count, and interval interpretation.
            bootstrap_replicates_requested: Some(None),
            ..Default::default()
        });
        Ok(super::super::finish_identified_execute_with_context(&self.result_context, None, args))
    }
}

/// Rollout level at one requested horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HorizonLevel {
    horizon_steps: u32,
    level: f64,
}

/// Retained-history certificate, computed from the sealed proofs.
///
/// An explicit history cap is certified when every requested horizon's proof
/// fit inside it, so shortening to the cap drops no identifying history. A
/// proof that needed more history leaves the cap uncertified, and a delay or
/// action lag deeper than the cap is refused.
fn truncation_certificate(
    temporal: &antecedent_core::TemporalResponseSpec,
    proven_history: u32,
) -> Option<TruncationCertificate> {
    let retained = temporal.max_history_lag?;
    (proven_history <= retained).then(|| TruncationCertificate::retaining(retained))
}

/// Admit the finite-horizon delayed frame at `steps` the response will be
/// estimated on.
///
/// The history window, initial law, and one transition come from the history
/// operators. The initial law is the one the dynamics were declared for. The
/// feedback delay is the mechanism's; the action lag stays the negative start
/// of the policy. An off-cadence horizon, an uncertified truncation, feedback
/// arriving after the horizon, and an ancestry deeper than the declared search
/// cap are refusals. The cap is not replaced by the depth the identifier walked.
fn admit_delayed_finite_horizon(
    cadence: &TimeIndex,
    temporal: &antecedent_core::TemporalResponseSpec,
    steps: u32,
    searched_steps: u32,
    mechanism: &FeedbackMechanism,
) -> Result<DelayedFeedback, HistoryOperatorError> {
    let horizon = horizon_ending_after(cadence, steps)?;
    let history = match temporal.max_history_lag {
        Some(lag) => HistoryWindow::max_lag(lag),
        None => HistoryWindow::planner_default(),
    };
    DelayedFeedback::try_new(DelayedFeedbackRequest {
        cadence: cadence.clone(),
        horizon,
        initial_law: Some(mechanism.dynamics.initial_law().clone()),
        history,
        delay: Some(mechanism.delay),
        action: delayed_action(&temporal.policy, mechanism.level)?,
        truncation: truncation_certificate(temporal, searched_steps),
        search_cap: temporal.max_history_lag.map(FeedbackSearchCap::declared),
        ancestry_steps: searched_steps,
    })
}

/// Finite-horizon rollout at every requested horizon, in request order.
fn finite_horizon_levels(
    cadence: &TimeIndex,
    temporal: &antecedent_core::TemporalResponseSpec,
    searched_steps: u32,
    mechanism: &FeedbackMechanism,
) -> Result<Vec<HorizonLevel>, CausalError> {
    temporal
        .horizons
        .iter()
        .map(|&steps| {
            let feedback =
                admit_delayed_finite_horizon(cadence, temporal, steps, searched_steps, mechanism)
                    .map_err(causal_refusal)?;
            let finite = temporal_target_response(
                &feedback,
                ResponseTarget::FiniteHorizon,
                &mechanism.dynamics,
            )?;
            Ok(HorizonLevel { horizon_steps: steps, level: finite.level() })
        })
        .collect()
}

/// Study payload for a declared feedback mechanism.
///
/// One horizon publishes its scalar rollout. Several publish a surface over the
/// requested horizons. Equilibrium, stationary, and continuous-time maps are
/// not substituted for it.
fn finite_horizon_causal_response(
    functional: &antecedent_core::ResponseFunctional,
    identification: &IdentificationResult,
    levels: &[HorizonLevel],
) -> CausalResponse {
    let horizons: Vec<f64> = levels.iter().map(|point| f64::from(point.horizon_steps)).collect();
    let means: Vec<f64> = levels.iter().map(|point| point.level).collect();
    let value = match means.as_slice() {
        [level] => antecedent_core::ResponseValue::Scalar(*level),
        _ => antecedent_core::ResponseValue::Surface {
            grid: Arc::from(horizons.as_slice()),
            dimension: 1,
            mean: Arc::from(means.as_slice()),
        },
    };
    let first = horizons.first().copied().unwrap_or(f64::NAN);
    let last = horizons.last().copied().unwrap_or(f64::NAN);
    CausalResponse {
        estimand: functional.clone(),
        identification_status: identification.status,
        estimate: antecedent_core::ResponseIdentification::PointIdentified(value),
        uncertainty: antecedent_core::ResponseUncertainty::None,
        support: antecedent_core::SupportReport {
            status: antecedent_core::SupportStatus::MissingEvidence,
            query_region: antecedent_core::SupportRegion {
                minima: Arc::from([first]),
                maxima: Arc::from([last]),
            },
            diagnostics: Vec::new(),
            warnings: Vec::new(),
            point_status: None,
        },
        assumptions: identification.required_assumptions.clone(),
        provenance_id: Arc::from("model.delayed_feedback.finite_horizon"),
        horizon_identification: None,
        interaction_structurally_zero: false,
    }
}

/// Publish the finite-horizon response of an admitted delayed frame.
///
/// A static equilibrium, a stationary law, and a continuous-time dynamic are
/// refused. The finite-horizon rollout is not returned in their place.
fn temporal_target_response(
    feedback: &DelayedFeedback,
    target: ResponseTarget,
    dynamics: &DelayedDynamics,
) -> Result<TargetResponse, CausalError> {
    if target != ResponseTarget::FiniteHorizon {
        return Err(CausalError::Compile {
            message: format!(
                "temporal response refuses {target}; a finite-horizon rollout does not replace it"
            ),
        });
    }
    let response = feedback.response(target, dynamics).map_err(causal_refusal)?;
    if response.target() != ResponseTarget::FiniteHorizon {
        return Err(CausalError::Compile {
            message: "temporal response returned a target other than the finite horizon".into(),
        });
    }
    Ok(response)
}

fn delayed_action(
    policy: &TemporalPolicy,
    level: f64,
) -> Result<DelayedAction, HistoryOperatorError> {
    match policy {
        TemporalPolicy::Pulse { at } => DelayedAction::pulse(*at, level),
        TemporalPolicy::Sustained { from, until } => DelayedAction::sustained(*from, *until, level),
        _ => Err(HistoryOperatorError::ActionNotDelayable),
    }
}

fn causal_refusal(error: HistoryOperatorError) -> CausalError {
    match error {
        HistoryOperatorError::SearchNotIdentified { .. } => CausalError::NotIdentified {
            status: IdentificationStatus::NotIdentified,
            search_capped: true,
            message: error.to_string(),
        },
        other => CausalError::Compile { message: other.to_string() },
    }
}

fn horizons_use_different_adjustments(
    members: &[crate::analysis::TemporalResponseHorizonEvidence],
) -> bool {
    let Some(first) = members.first() else {
        return false;
    };
    let normalized = |member: &crate::analysis::TemporalResponseHorizonEvidence| {
        let mut keys = member
            .estimand()
            .adjustment_set
            .iter()
            .filter_map(|variable| member.indexer().key_of(variable.raw()).ok())
            .collect::<Vec<_>>();
        keys.sort();
        keys
    };
    let baseline = normalized(first);
    members.iter().skip(1).any(|member| normalized(member) != baseline)
}

fn lag_aligned_rows(
    data: &TimeSeriesData,
    members: &[crate::analysis::TemporalResponseHorizonEvidence],
) -> u64 {
    let span = members
        .iter()
        .map(|member| member.indexer().history().saturating_add(member.indexer().horizon()))
        .max()
        .unwrap_or(1);
    u64::try_from(
        data.row_count().saturating_sub(usize::try_from(span.saturating_sub(1)).unwrap_or(0)),
    )
    .unwrap_or(u64::MAX)
}

/// Read-only inspection of a retained `TemporalDag` response operation.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedTemporalDagResponseInfo {
    /// Exact response functional, policy, and requested horizons.
    pub query: antecedent_core::ResponseQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Interval procedure and replicate count.
    pub uncertainty: std::sync::Arc<str>,
    /// Ordered response grid.
    pub grid_members: std::sync::Arc<[f64]>,
    /// Ordered horizon members.
    pub horizons: std::sync::Arc<[u32]>,
}

impl Inspect for CheckedTemporalResponseExecution {
    type Info = CheckedTemporalDagResponseInfo;

    fn info(&self) -> CheckedTemporalDagResponseInfo {
        let operation = self.operation();
        let procedure = operation.procedure();
        let (identifier, estimator, uncertainty) =
            (procedure.identifier, procedure.estimator, procedure.uncertainty);
        CheckedTemporalDagResponseInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            uncertainty: Arc::from(uncertainty),
            grid_members: Arc::from(operation.grid()),
            // The sealed evidence is one member per requested horizon, in order.
            horizons: operation
                .evidence()
                .iter()
                .map(crate::analysis::TemporalResponseHorizonEvidence::horizon)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Study;
    use crate::analysis::builder::RefuteSuite;
    use crate::strategy_table::{EstimatorId, IdentifierId};
    use antecedent_core::ResponseQuery;
    use antecedent_core::{
        ContinuousDomain, GridSpec, Lag, ResponseFunctional, TemporalEffectQuery, TemporalPolicy,
        TemporalResponseSpec, VariableId,
    };
    use antecedent_data::TimeSeriesData;
    use antecedent_graph::{TemporalDag, ensure_lagged};
    use antecedent_identify::IdentificationStatus;
    use antecedent_identify::temporal_backdoor::TemporalBackdoorIdentifier;

    #[test]
    fn full_result_is_assembled_from_the_retained_operation_and_context() {
        let n = 800;
        let treatment = (0..n).map(|i| ((i * 31 % 997) as f64 - 498.0) / 250.0).collect::<Vec<_>>();
        let outcome = std::iter::once(0.0)
            .chain(treatment.iter().take(n - 1).map(|value| 1.5 + 2.0 * value))
            .collect::<Vec<_>>();
        let data = TimeSeriesData::from_f64_columns(
            [("t", treatment.as_slice()), ("y", outcome.as_slice())],
            1,
        )
        .unwrap();
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let mut graph = TemporalDag::empty();
        let t_lag = ensure_lagged(&mut graph, t, Lag::from_raw(1)).unwrap();
        let y_now = ensure_lagged(&mut graph, y, Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t_lag, y_now).unwrap();
        let policy = TemporalPolicy::pulse(-1);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(t, GridSpec::Values(Arc::from([-0.5, 0.0, 0.5]))),
        })
        .with_temporal(TemporalResponseSpec::new(vec![1], policy.clone(), None).unwrap());
        let mut effect = TemporalEffectQuery::pulse(t, y, 1.0).with_horizon_steps(1);
        effect.policy = policy;
        let identified =
            TemporalBackdoorIdentifier::new().identify_temporal(&graph, &effect).unwrap();
        let member = crate::analysis::TemporalResponseHorizonEvidence::checked(
            1,
            effect,
            identified.result.clone(),
            identified.result.estimands[0].clone(),
            identified.indexer.clone(),
        )
        .unwrap();
        let operation = crate::analysis::CheckedTemporalResponseOperation::checked(
            &graph,
            &query,
            vec![member],
            IdentifierId::TemporalBackdoorUnfolded,
            EstimatorId::TemporalResponseGcomp,
            32,
            &InferenceMode::Frequentist,
        )
        .unwrap();

        // Freeze result metadata at preparation. Execution below receives no Study.
        let study = Study::series(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(query.clone()))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(32)
            .build()
            .unwrap();
        let context = IdentifiedResultContext::from_study(&study);
        let ctx = ExecutionContext::for_tests(41);
        let reference = study.run(&ctx).unwrap();
        let logical =
            crate::planner::compile_logical_temporal_response(&data, &graph, &query, false)
                .unwrap();
        let physical = logical.compile_physical_with_graph(&ctx, Some(graph.clone())).unwrap();
        let execution =
            CheckedTemporalResponseExecution::checked(operation, context, Arc::new(physical))
                .unwrap();
        drop(study);
        let result = execution.execute(&data, &ctx).unwrap();

        assert_eq!(result.identification.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(result.estimate.as_effect().unwrap().ate.is_nan()); // curves have no scalar contrast
        assert_eq!(result.logical_plan.plan_id.as_ref(), "temporal_response");
        assert_eq!(result.response().unwrap().estimand, query.functional);
        assert_eq!(
            result.estimate.as_effect().unwrap().provenance.n_obs,
            reference.estimate.as_effect().unwrap().provenance.n_obs
        );
        assert_eq!(result.provenance, reference.provenance);
        assert_eq!(
            result.identification.required_assumptions,
            reference.identification.required_assumptions
        );
        let antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Surface { mean, grid, .. },
        ) = &result.response().unwrap().estimate
        else {
            panic!("mean-curve result has a point-identified surface");
        };
        assert_eq!(grid.as_ref(), [-0.5, 1.0, 0.0, 1.0, 0.5, 1.0]);
        for (value, level) in mean.iter().zip([-0.5, 0.0, 0.5]) {
            assert!((value - (1.5 + 2.0 * level)).abs() < 0.08);
        }
        assert!(matches!(
            result.response().unwrap().uncertainty,
            antecedent_core::ResponseUncertainty::PointwiseBand {
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                ..
            }
        ));
        assert!(result.provenance.nodes.len() >= 2);
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "refute.temporal_response.skipped")
        );
        assert!(
            !result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "model.delayed_feedback.finite_horizon")
        );
    }

    use super::{
        DelayedDynamics, DelayedFeedback, FeedbackMechanism, HistoryOperatorError, HorizonLevel,
        ResponseTarget, admit_delayed_finite_horizon, causal_refusal,
        finite_horizon_causal_response, finite_horizon_levels, temporal_target_response,
    };
    use antecedent_core::InitialLaw;
    use antecedent_data::{SamplingRegularity, TimeIndex};

    fn cadence(interval_ns: u64, length: usize) -> TimeIndex {
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns }, length }
    }

    fn response_spec(
        policy: TemporalPolicy,
        horizons: Vec<u32>,
        max_history_lag: Option<u32>,
    ) -> TemporalResponseSpec {
        TemporalResponseSpec::new(horizons, policy, max_history_lag).unwrap()
    }

    fn dynamics_for(law: InitialLaw) -> DelayedDynamics {
        DelayedDynamics::try_new(0.5, 1.0, 1.0, law, 0.0).unwrap()
    }

    fn mechanism() -> FeedbackMechanism {
        FeedbackMechanism::declared(1, 1.0, dynamics_for(InitialLaw::contemporaneous_observed()))
            .unwrap()
    }

    fn admit_at(
        policy: TemporalPolicy,
        steps: u32,
        max_history_lag: Option<u32>,
        searched_steps: u32,
    ) -> Result<DelayedFeedback, HistoryOperatorError> {
        let spec = response_spec(policy, vec![steps], max_history_lag);
        admit_delayed_finite_horizon(&cadence(1, 8), &spec, steps, searched_steps, &mechanism())
    }

    fn rollout(feedback: &DelayedFeedback) -> f64 {
        temporal_target_response(feedback, ResponseTarget::FiniteHorizon, &mechanism().dynamics)
            .unwrap()
            .level()
    }

    #[test]
    fn temporal_response_keeps_finite_horizon_apart_from_other_targets() {
        let sustained = admit_at(TemporalPolicy::sustained(0, 1), 3, Some(2), 2).unwrap();
        let dynamics = mechanism().dynamics;
        let finite =
            temporal_target_response(&sustained, ResponseTarget::FiniteHorizon, &dynamics).unwrap();
        assert_eq!(finite.target(), ResponseTarget::FiniteHorizon);
        assert_eq!(finite.level_bits(), 1.5f64.to_bits());
        for target in [
            ResponseTarget::StaticEquilibrium,
            ResponseTarget::Stationary,
            ResponseTarget::ContinuousTime,
        ] {
            let other = sustained.response(target, &dynamics).unwrap();
            assert_ne!(other.target(), finite.target());
            assert_ne!(other, finite);
            let refused = temporal_target_response(&sustained, target, &dynamics).unwrap_err();
            assert!(matches!(refused, CausalError::Compile { .. }));
        }
    }

    #[test]
    fn temporal_response_pulse_differs_from_sustained_windows_shorter_than_the_horizon() {
        let pulse = admit_at(TemporalPolicy::pulse(0), 3, Some(2), 2).unwrap();
        let sustained = admit_at(TemporalPolicy::sustained(0, 1), 3, Some(2), 2).unwrap();
        let one_step = admit_at(TemporalPolicy::sustained(0, 0), 3, Some(2), 2).unwrap();
        assert_eq!(rollout(&pulse).to_bits(), 0.5f64.to_bits());
        assert_eq!(rollout(&sustained).to_bits(), 1.5f64.to_bits());
        assert_eq!(rollout(&one_step).to_bits(), 0.5f64.to_bits());
        assert_ne!(pulse.action(), one_step.action());
        assert_ne!(pulse.action(), sustained.action());
    }

    #[test]
    fn the_action_lag_is_not_the_feedback_delay() {
        // The same delay of one tick, with the action sitting zero and two steps back.
        let at_origin = admit_at(TemporalPolicy::pulse(0), 3, None, 0).unwrap();
        let lagged = admit_at(TemporalPolicy::pulse(-2), 3, None, 0).unwrap();
        assert_eq!(at_origin.delay(), lagged.delay());
        assert_eq!(at_origin.delay().unwrap().ticks(), 1);
        assert_eq!(rollout(&at_origin).to_bits(), 0.5f64.to_bits());
        // The lagged pulse arrives at -1, before the origin, and leaves no input in the horizon.
        assert_eq!(rollout(&lagged).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn temporal_response_refuses_feedback_arriving_after_the_horizon() {
        let error = admit_at(TemporalPolicy::sustained(0, 2), 3, None, 0).unwrap_err();
        assert!(matches!(error, HistoryOperatorError::InformationRefused { .. }));
        assert!(matches!(causal_refusal(error), CausalError::Compile { .. }));
        let negative = admit_at(TemporalPolicy::pulse(-1), 3, None, 0).unwrap();
        assert!(negative.information().is_some());
    }

    #[test]
    fn temporal_response_refuses_dynamics_for_another_initial_law() {
        let feedback = admit_at(TemporalPolicy::pulse(0), 3, None, 0).unwrap();
        assert_eq!(feedback.initial_law(), mechanism().dynamics.initial_law());
        let other = dynamics_for(InitialLaw::named("other").unwrap());
        let refused =
            temporal_target_response(&feedback, ResponseTarget::FiniteHorizon, &other).unwrap_err();
        assert!(matches!(refused, CausalError::Compile { .. }));
    }

    #[test]
    fn temporal_response_refuses_a_dynamic_policy_and_a_zero_delay() {
        let dynamic = TemporalPolicy::dynamic(antecedent_core::DynamicRuleId::from_raw(1), [0]);
        let spec = TemporalResponseSpec {
            horizons: Arc::from([3]),
            policy: dynamic,
            max_history_lag: None,
        };
        let error =
            admit_delayed_finite_horizon(&cadence(1, 8), &spec, 3, 0, &mechanism()).unwrap_err();
        assert_eq!(error, HistoryOperatorError::ActionNotDelayable);
        let zero = FeedbackMechanism::declared(
            0,
            1.0,
            dynamics_for(InitialLaw::contemporaneous_observed()),
        )
        .unwrap_err();
        assert_eq!(zero, HistoryOperatorError::ContemporaneousFeedback);
    }

    #[test]
    fn temporal_response_publishes_every_requested_horizon() {
        let spec = response_spec(TemporalPolicy::pulse(-1), vec![1, 2, 3], None);
        let levels = finite_horizon_levels(&cadence(1, 8), &spec, 0, &mechanism()).unwrap();
        assert_eq!(
            levels,
            [
                HorizonLevel { horizon_steps: 1, level: 1.0 },
                HorizonLevel { horizon_steps: 2, level: 0.5 },
                HorizonLevel { horizon_steps: 3, level: 0.25 },
            ]
        );
        let functional = ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0])),
            ),
        };
        let identification = identification_for_publication();
        let published = finite_horizon_causal_response(&functional, &identification, &levels);
        let antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Surface { grid, mean, dimension },
        ) = &published.estimate
        else {
            panic!("several horizons publish a surface over the horizons");
        };
        assert_eq!(
            (grid.as_ref(), mean.as_ref(), *dimension),
            (&[1.0, 2.0, 3.0][..], &[1.0, 0.5, 0.25][..], 1)
        );
        assert_eq!(published.support.query_region.minima.as_ref(), [1.0]);
        assert_eq!(published.support.query_region.maxima.as_ref(), [3.0]);
        let single = finite_horizon_causal_response(&functional, &identification, &levels[..1]);
        assert!(matches!(
            single.estimate,
            antecedent_core::ResponseIdentification::PointIdentified(
                antecedent_core::ResponseValue::Scalar(level)
            ) if level.to_bits() == 1.0f64.to_bits()
        ));
    }

    fn identification_for_publication() -> IdentificationResult {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let mut graph = TemporalDag::empty();
        let t_lag = ensure_lagged(&mut graph, t, Lag::from_raw(1)).unwrap();
        let y_now = ensure_lagged(&mut graph, y, Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t_lag, y_now).unwrap();
        let effect = TemporalEffectQuery::pulse(t, y, 1.0).with_horizon_steps(1);
        TemporalBackdoorIdentifier::new().identify_temporal(&graph, &effect).unwrap().result
    }

    fn published_finite_horizon(policy: TemporalPolicy) -> f64 {
        let n = 16;
        let treatment = vec![0.0; n];
        let outcome = vec![0.0; n];
        let data = TimeSeriesData::from_f64_columns(
            [("t", treatment.as_slice()), ("y", outcome.as_slice())],
            1,
        )
        .unwrap();
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let mut graph = TemporalDag::empty();
        let t_lag = ensure_lagged(&mut graph, t, Lag::from_raw(1)).unwrap();
        let y_now = ensure_lagged(&mut graph, y, Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t_lag, y_now).unwrap();
        let horizon = 1;
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(t, GridSpec::Values(Arc::from([-0.5, 0.0, 0.5]))),
        })
        .with_temporal(TemporalResponseSpec::new(vec![horizon], policy.clone(), None).unwrap());
        let mut effect = TemporalEffectQuery::pulse(t, y, 1.0).with_horizon_steps(horizon);
        effect.policy = policy;
        let identified =
            TemporalBackdoorIdentifier::new().identify_temporal(&graph, &effect).unwrap();
        let member = crate::analysis::TemporalResponseHorizonEvidence::checked(
            horizon,
            effect,
            identified.result.clone(),
            identified.result.estimands[0].clone(),
            identified.indexer.clone(),
        )
        .unwrap();
        let operation = crate::analysis::CheckedTemporalResponseOperation::checked(
            &graph,
            &query,
            vec![member],
            IdentifierId::TemporalBackdoorUnfolded,
            EstimatorId::TemporalResponseGcomp,
            32,
            &InferenceMode::Frequentist,
        )
        .unwrap();
        let study = Study::series(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(query.clone()))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(32)
            .build()
            .unwrap();
        let context = IdentifiedResultContext::from_study(&study);
        let ctx = ExecutionContext::for_tests(7);
        let logical =
            crate::planner::compile_logical_temporal_response(&data, &graph, &query, false)
                .unwrap();
        let physical = logical.compile_physical_with_graph(&ctx, Some(graph)).unwrap();
        let execution =
            CheckedTemporalResponseExecution::checked(operation, context, Arc::new(physical))
                .unwrap()
                .with_feedback(mechanism());
        let result = execution.execute(&data, &ctx).unwrap();
        let antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Scalar(level),
        ) = &result.response().unwrap().estimate
        else {
            panic!("delayed feedback published a dose-by-horizon fit");
        };
        assert_eq!(result.estimate.as_effect().unwrap().ate.to_bits(), level.to_bits());
        assert_eq!(
            result.response().unwrap().provenance_id.as_ref(),
            "model.delayed_feedback.finite_horizon"
        );
        let spec = query.temporal.as_ref().unwrap();
        let admitted = admit_delayed_finite_horizon(
            data.time_index(),
            spec,
            horizon,
            identified.indexer.history(),
            &mechanism(),
        )
        .unwrap();
        let operator = temporal_target_response(
            &admitted,
            ResponseTarget::FiniteHorizon,
            &mechanism().dynamics,
        )
        .unwrap();
        assert_eq!(level.to_bits(), operator.level_bits());
        *level
    }

    #[test]
    fn a_declared_feedback_mechanism_publishes_the_finite_horizon_rollout() {
        let pulse = published_finite_horizon(TemporalPolicy::pulse(-1));
        assert_eq!(pulse.to_bits(), 1.0f64.to_bits());
    }

    #[test]
    fn temporal_response_refuses_an_off_cadence_horizon() {
        let irregular = TimeIndex { regularity: SamplingRegularity::Irregular, length: 8 };
        let spec = response_spec(TemporalPolicy::pulse(-1), vec![1], None);
        let error =
            admit_delayed_finite_horizon(&irregular, &spec, 1, 0, &mechanism()).unwrap_err();
        assert_eq!(
            error,
            HistoryOperatorError::CadenceWithoutInterval { interval_ns: None, steps: 1 }
        );
        let past = admit_delayed_finite_horizon(&cadence(1, 8), &spec, 9, 0, &mechanism());
        assert!(matches!(past, Err(HistoryOperatorError::HorizonCrossesCadence { end_ns: 9, .. })));
    }

    #[test]
    fn temporal_response_certifies_a_truncation_the_proofs_fit_inside() {
        // The delay of one tick is deeper than the retained zero, and the proofs fit in zero.
        let admitted = admit_at(TemporalPolicy::pulse(-1), 1, Some(0), 0).unwrap();
        assert_eq!(admitted.history().history().max_lag_steps(), Some(0));
        assert_eq!(admitted.delay().unwrap().ticks(), 1);
    }

    #[test]
    fn temporal_response_refuses_a_truncation_the_proofs_do_not_fit_inside() {
        let error = admit_at(TemporalPolicy::pulse(-1), 1, Some(0), 2).unwrap_err();
        assert_eq!(
            error,
            HistoryOperatorError::UncertifiedTruncation { retained_ticks: 0, required_ticks: 1 }
        );
    }

    #[test]
    fn temporal_response_does_not_identify_a_raised_search_cap() {
        let error = admit_at(TemporalPolicy::pulse(-1), 1, Some(2), 5).unwrap_err();
        assert_eq!(error, HistoryOperatorError::SearchNotIdentified { cap: 2, ancestry_steps: 5 });
        match causal_refusal(error) {
            CausalError::NotIdentified { status, search_capped, .. } => {
                assert_eq!(status, IdentificationStatus::NotIdentified);
                assert!(search_capped);
            }
            other => panic!("a raised cap was not refused: {other:?}"),
        }
    }
}
