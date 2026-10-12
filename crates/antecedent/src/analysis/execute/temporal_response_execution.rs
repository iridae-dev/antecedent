//! Full result assembly for a sealed temporal DAG mean response operation.
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
        Ok(Self { operation, result_context, physical, identification, estimand })
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
        let (mut response, conflict_summary) =
            self.execute_response_surface(data, temporal, (treatment, outcome), ctx)?;
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
    }
}
