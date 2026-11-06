//! Temporal class cross-world fitting and execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::Study;
use super::{
    Arc, CROSS_WORLD_PRIOR_REFUSAL, CausalError, CausalQuery, ClassPoints, CrossWorldContrast,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EstimatorId, ExecutionContext, FrozenAtom,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, Instant, PhysicalExecutionPlan, RefuteSuite,
    RouteDefaults, SlopeAtom, SlopeSource, Strategy, StudyResult, TimeSeriesData, VariableId,
    class_scalar_uncertainty, collect_class_points, cross_world_endpoints, cross_world_probe,
    elapsed_ns, envelope_span, identify_cached_diagnostic, isotropic_prior_scale,
    point_uncertainty, refuse_unread_series_weights, report_identify_compute, select_estimand,
};

/// The refusal message for a cross-world query whose back-door set is not empty.
fn cross_world_backdoor_refusal(query: &CausalQuery) -> &'static str {
    match query {
        CausalQuery::NestedCounterfactual(_) => {
            "temporal nested effect is supported when the back-door set is empty"
        }
        CausalQuery::PathSpecific(_) => {
            "temporal path-specific effect is supported when the back-door set is empty"
        }
        CausalQuery::Counterfactual(_) => {
            "temporal counterfactual is supported when the back-door set is empty"
        }
        CausalQuery::Distribution(_) => {
            "temporal interventional mean is supported when the back-door set is empty"
        }
        _ => "temporal cross-world effect is supported when the back-door set is empty",
    }
}

/// Every identified completion's cross-world regressions and point.
pub(super) fn collect_class_cross_world<'a>(
    data: &'a TimeSeriesData,
    query: &CausalQuery,
    bundle: &'a antecedent_identify::TemporalClassEnvelope,
    (contrast, offsets): (CrossWorldContrast, (i32, i32)),
    policy: antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<ClassPoints<'a>, CausalError> {
    refuse_unread_series_weights(data, "the class-aware cross-world contrast")?;
    collect_class_points(bundle, |case, indexer| {
        if let CausalQuery::PathSpecific(path) = query {
            match &case.graph {
                antecedent_identify::TemporalCompletionGraph::Dag(graph) => {
                    super::super::temporal_path::require_path_chain(graph, &path.path_nodes)?;
                }
                antecedent_identify::TemporalCompletionGraph::Mag(graph) => {
                    let dense = |variable: VariableId| {
                        graph.nodes().iter().position(|node| matches!(node,
                            antecedent_graph::NodeRef::Lagged { variable: v, lag }
                            if *v == variable && *lag == antecedent_core::Lag::CONTEMPORANEOUS
                        )).map(antecedent_graph::DenseNodeId::from_index)
                    };
                    for pair in path.path_nodes.windows(2) {
                        if !matches!((dense(pair[0]), dense(pair[1])), (Some(a), Some(b)) if graph.reaches_directed(a, b))
                        {
                            return Err(CausalError::Unsupported {
                                message: "temporal path-specific path nodes must be listed in causal order, each contemporaneous node reaching the next in the graph",
                            });
                        }
                    }
                }
            }
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        if !estimand.adjustment_set.is_empty() {
            let message = cross_world_backdoor_refusal(query);
            return Err(CausalError::Unsupported { message });
        }
        let atom = SlopeAtom::new(data, &estimand, indexer, offsets, policy, prior_scale);
        let point = cross_world_from(query, &atom, contrast)?;
        Ok((atom, point))
    })
}

/// The cross-world point of `query` from the slopes `source` fits.
pub(super) fn cross_world_from(
    query: &CausalQuery,
    source: &dyn SlopeSource,
    contrast: CrossWorldContrast,
) -> Result<f64, CausalError> {
    let fit = |outcome: VariableId, treatments: &[VariableId], contemporaneous: &[VariableId]| {
        source.slopes(outcome, treatments, contemporaneous)
    };
    super::super::temporal_path::cross_world_point(query, &fit, contrast)
}

impl Study {
    #[allow(
        clippy::too_many_lines,
        reason = "one class cross-world execution: the completions, the mixture and its interval stay together"
    )]
    pub(in crate::analysis::execute) fn execute_temporal_class_cross_world(
        &self,
        data: &TimeSeriesData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.settings.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal cross-world effects are point-only; cheap/full refuters are not supported",
            });
        }
        let prior_scale =
            isotropic_prior_scale(&self.settings.inference, CROSS_WORLD_PRIOR_REFUSAL)?;
        let (treatment, outcome, mediator, active, control) = cross_world_endpoints(&self.query)?;
        let contrast = CrossWorldContrast { treatment, outcome, mediator, active, control };
        let probe = cross_world_probe(treatment, outcome);
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        // The probe is identified once, at compile, and the program retains its certificate;
        // a plan that retained none identifies here.
        let (bundle, identify_cached) =
            if let Some(crate::planner::lowered::RetainedExecutionProof::CrossWorldClass {
                envelope,
            }) = crate::analysis::lowering::plan_retained_execution_proof(self, physical)
            {
                (Arc::clone(envelope), false)
            } else {
                report_identify_compute(ctx);
                (self.identify_temporal_class(identifier_id, &probe, ctx)?.envelope.clone(), false)
            };
        let offsets = (probe.try_treatment_offset()?, probe.outcome_offset()?);
        let query = &self.query;
        let fitted = collect_class_cross_world(
            data,
            query,
            &bundle,
            (contrast, offsets),
            ctx.kernel_policy,
            prior_scale,
        )?;
        let point = fitted.point()?;
        let identification = super::super::temporal_path::template_scoped_cross_world(
            fitted.primary_identification.clone(),
            &fitted.primary_indexer,
            self.query.clone(),
            contrast,
        );
        let estimand = identification.estimands[0].clone();
        let envelope = &bundle.envelope;
        let mut diagnostics =
            vec![super::super::temporal_path::temporal_class_envelope_diagnostic(
                envelope,
                self.graph.class(),
            )];
        diagnostics.push(cross_world_note());
        diagnostics.extend(fitted.skipped.diagnostic());
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let mut structural = super::super::temporal_path::temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &fitted.values,
        );
        let frozen: Vec<FrozenAtom> = fitted.atoms.into_iter().map(SlopeAtom::freeze).collect();
        let outcome_uncertainty = class_scalar_uncertainty(
            &frozen,
            &fitted.weights,
            &fitted.points,
            point,
            identification.required_assumptions.clone(),
            envelope.truncated_completions > 0,
            self.class_point_plan(
                envelope_span(&bundle),
                (
                    point_uncertainty::CROSS_WORLD_BLOCK_STREAM,
                    point_uncertainty::CROSS_WORLD_POSTERIOR_STREAM,
                ),
            ),
            ctx,
            |_, source| cross_world_from(query, source, contrast).map(|value| vec![value]),
        )?;
        structural.identified_set_interval = outcome_uncertainty.identified_set;
        diagnostics.extend(outcome_uncertainty.diagnostics.iter().cloned());
        let posterior = outcome_uncertainty.posterior;
        let n_draws = posterior
            .as_ref()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                outcome_uncertainty.estimate,
                Strategy::new(identifier_id, EstimatorId::TemporalLinearAdjustment),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(outcome_uncertainty.replicates_ok)
            .cancelled(outcome_uncertainty.cancelled)
            .extras(IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: Arc::unwrap_or_clone(bundle),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                n_draws,
                posterior,
                ..Default::default()
            }),
        );
        Ok(result)
    }
}

pub(super) fn cross_world_note() -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal.cross_world",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "published value is the completion-weighted linear unfolded cross-world point; its \
         interval refits every completion's regressions of the contrast",
    )
}
