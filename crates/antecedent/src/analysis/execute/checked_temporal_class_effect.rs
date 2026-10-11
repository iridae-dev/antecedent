//! Builder independent execution of checked temporal class effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::temporal_path::{temporal_class_envelope_diagnostic, temporal_class_structural_mixture};
use super::*;

/// Checked class envelope together with its result and physical contracts.
#[derive(Clone)]
pub(crate) struct CheckedTemporalClassEffectExecution {
    operation: crate::analysis::CheckedTemporalClassEffectOperation,
    result_context: IdentifiedResultContext,
    physical: PhysicalExecutionPlan,
}

impl std::fmt::Debug for CheckedTemporalClassEffectExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedTemporalClassEffectExecution")
            .field("operation", &self.operation)
            .field("physical", &self.physical)
            .finish_non_exhaustive()
    }
}

impl CheckedTemporalClassEffectExecution {
    /// Replace the caller validators run by the sealed class operation.
    pub(crate) fn set_custom_validators(
        &mut self,
        validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    ) {
        self.operation.set_custom_validators(validators);
    }

    pub(crate) fn checked(
        operation: crate::analysis::CheckedTemporalClassEffectOperation,
        result_context: IdentifiedResultContext,
        physical: PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let query = CausalQuery::TemporalEffect(operation.query().clone());
        let (identifier, estimator, _, _, _, _) = operation.procedure();
        if !operation.matches_query(&result_context.query)
            || result_context.query != query
            || physical.logical.query != query
            || physical.logical.record.identifier.as_deref() != Some(identifier.as_str())
            || physical.logical.record.estimator.as_deref() != Some(estimator.as_str())
            || result_context.graph_class != operation.graph_class()
            || result_context.graph_version != operation.structure_version()
        {
            return Err(CausalError::Compile {
                message:
                    "temporal class result context disagrees with its retained proof or procedure"
                        .into(),
            });
        }
        Ok(Self { operation, result_context, physical })
    }

    #[must_use]
    pub(crate) fn operation(&self) -> &crate::analysis::CheckedTemporalClassEffectOperation {
        &self.operation
    }

    pub(crate) fn execute(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let operation = &self.operation;
        let query = operation.query();
        let bundle = operation.bundle();
        let envelope = &bundle.envelope.envelope;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "temporal class effect has no identified mass (no identified completion mass)",
            ));
        }
        let (identifier_id, estimator_id, bootstrap_replicates, split, refute, custom_validators) =
            operation.procedure();
        if !query.effect_modifiers.is_empty() {
            return self.execute_conditional(
                started,
                data,
                ctx,
                identifier_id,
                estimator_id,
                refute,
            );
        }
        if query.is_multi_step_sustained() {
            self.execute_sequential(
                started,
                data,
                ctx,
                identifier_id,
                estimator_id,
                bootstrap_replicates,
                refute,
                custom_validators,
            )
        } else {
            self.execute_linear(
                started,
                data,
                ctx,
                identifier_id,
                estimator_id,
                bootstrap_replicates,
                split,
                refute,
                custom_validators,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_linear(
        &self,
        started: Instant,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        bootstrap_replicates: u32,
        split: Option<&antecedent_data::DiscoveryEstimationSplit>,
        refute: RefuteSuite,
        custom_validators: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    ) -> Result<StudyResult, CausalError> {
        let operation = &self.operation;
        let query = operation.query();
        let envelope = &operation.bundle().envelope.envelope;
        let mut diagnostics =
            vec![temporal_class_envelope_diagnostic(envelope, operation.graph_class())];
        let fit = fit_class_linear_atoms(
            data,
            query,
            &operation.bundle().envelope,
            &operation.bundle().envelope.indexers,
            split,
            ctx,
        )?;
        let ClassLinearAtoms {
            values,
            designs,
            weights,
            indexers,
            points,
            se_items,
            refute_atoms,
            primary_estimand,
            assumptions,
            weighted_point,
            total_weight,
        } = fit;
        if total_weight <= 0.0 {
            return Err(CausalError::Compile {
                message: "temporal class envelope has no estimable identified cases".into(),
            });
        }
        let design_refs = designs.iter().collect::<Vec<_>>();
        let block = shared_circular_block_mixture_se(
            &design_refs,
            &weights,
            temporal_class_block_span(indexers.iter().copied()),
            bootstrap_replicates,
            0xC1A5_5E00,
            ctx,
        );
        let identified_set_interval = block
            .identified_set_interval(&points, IDENTIFIED_SET_INTERVAL_LEVEL)
            .map(|interval| interval.with_truncated(envelope.truncated_completions > 0));
        let se_bootstrap = block.se.is_finite().then_some(block.se);
        let estimate = class_mixture_estimate(
            weighted_point / total_weight,
            mix_weighted_analytic_se(se_items),
            &block,
            bootstrap_replicates,
            assumptions,
            ctx,
        );
        let tabular = TabularData::new(data.storage().clone());
        let ate_query = temporal_refuter_average(query);
        let (refutations, refute_diagnostics) = run_envelope_effect_refuters(
            &tabular,
            &ate_query,
            &refute_atoms,
            &mut EstimationWorkspace::default(),
            ctx,
            refute,
            estimator_id.as_str(),
            custom_validators,
            Some(query),
            split,
            Some(data.time_index()),
        )?;
        diagnostics.extend(refute_diagnostics);
        if se_bootstrap.is_some() {
            diagnostics.extend(envelope_shared_block_diagnostics(
                total_weight,
                envelope.unidentified_weight.0,
                &block,
            ));
        } else if designs.len() > 1 {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        diagnostics.extend(
            identified_set_interval
                .as_ref()
                .into_iter()
                .flat_map(identified_set_interval_diagnostics),
        );
        let mut structural = temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &values,
        );
        structural.identified_set_interval = identified_set_interval;
        Ok(self.finish(
            started,
            identifier_id,
            estimator_id,
            primary_estimand.ok_or_else(|| CausalError::Compile {
                message: "temporal class effect has no primary estimand".into(),
            })?,
            estimate,
            refutations,
            diagnostics,
            Some(structural),
            (bootstrap_replicates > 0).then_some(block.completed),
            Some(crate::Identification::TemporalEnvelope {
                envelope: operation.bundle().envelope.clone(),
                strategy: identifier_id,
                structure_version: operation.structure_version(),
            }),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_sequential(
        &self,
        started: Instant,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        bootstrap_replicates: u32,
        refute: RefuteSuite,
        custom_validators: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    ) -> Result<StudyResult, CausalError> {
        if self.operation.procedure().3.is_some() {
            return Err(CausalError::Unsupported {
                message: "class-aware multi-step sustained requires no discovery-estimation split",
            });
        }
        let operation = &self.operation;
        let query = operation.query();
        let envelope = &operation.bundle().envelope.envelope;
        let mut diagnostics =
            vec![temporal_class_envelope_diagnostic(envelope, operation.graph_class())];
        let mut values = vec![None; envelope.cases.len()];
        let mut atoms = Vec::new();
        let mut primary_estimand = None;
        let mut weighted_point = 0.0;
        let mut total_weight = 0.0;
        let mut unevaluable_weight = 0.0;
        for (index, (case, indexer)) in
            envelope.cases.iter().zip(&operation.bundle().envelope.indexers).enumerate()
        {
            let key = case.graph.fingerprint();
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let Some(graph) = case.graph.sequential_dag() else {
                unevaluable_weight += case.weight.0;
                diagnostics.push(mag_sequential_unevaluable_diagnostic(key));
                continue;
            };
            let estimand = select_estimand(&case.result, EstimatorId::TemporalSequentialGcomp)
                .or_else(|_| {
                    select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)
                })?;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
            }
            let mut assumptions = case.result.required_assumptions.clone();
            assumptions.push(sequential_linear_sem_assumption(UNFOLDED_LINEAR_SEM));
            let mut mechanisms = Vec::new();
            let (estimate, posterior) = antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
                data,
                &graph,
                indexer,
                &estimand,
                query,
                case.result.status,
                assumptions,
                bootstrap_replicates,
                None,
                ctx,
                Some(&mut mechanisms),
            )
            .map_err(CausalError::from)?;
            debug_assert!(posterior.is_none());
            values[index] = Some(estimate.ate);
            weighted_point += case.weight.0 * estimate.ate;
            total_weight += case.weight.0;
            atoms.push(super::sequential_validation::SequentialValidationAtom {
                weight: case.weight.0,
                graph,
                indexer: indexer.clone(),
                estimand,
                status: case.result.status,
                estimate,
                mechanisms,
            });
        }
        if atoms.is_empty() || total_weight <= 0.0 {
            return Err(CausalError::Compile {
                message: "temporal class sequential envelope has no evaluable directed completions"
                    .into(),
            });
        }
        let SequentialAtomBlock { block, identified_set_interval, se_analytic } =
            sequential_atom_block(
                data,
                query,
                &atoms,
                bootstrap_replicates,
                envelope.truncated_completions > 0,
                ctx,
            )?;
        if block.se.is_finite() {
            diagnostics.extend(envelope_shared_block_diagnostics(
                total_weight,
                envelope.unidentified_weight.0,
                &block,
            ));
        } else if atoms.len() > 1 {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        diagnostics.extend(
            identified_set_interval
                .as_ref()
                .into_iter()
                .flat_map(identified_set_interval_diagnostics),
        );
        let estimate = class_mixture_estimate(
            weighted_point / total_weight,
            se_analytic,
            &block,
            bootstrap_replicates,
            atoms[0].estimate.assumptions.clone(),
            ctx,
        );
        let (refutations, extra_diagnostics, _) =
            super::sequential_validation::validate_sequential(
                data,
                query,
                &atoms,
                refute,
                custom_validators,
                None,
                None,
                estimate.ate,
                ctx,
                0,
            )?;
        diagnostics.extend(extra_diagnostics);
        let mut structural = temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &values,
        );
        structural.identified_set_interval = identified_set_interval;
        let total: f64 = structural.atoms.iter().map(|atom| atom.weight).sum();
        structural.unevaluable_mass = if total > 0.0 { unevaluable_weight / total } else { 0.0 };
        structural.unidentified_mass =
            (structural.unidentified_mass - structural.unevaluable_mass).max(0.0);
        Ok(self.finish(
            started,
            identifier_id,
            estimator_id,
            primary_estimand.ok_or_else(|| CausalError::Compile {
                message: "temporal class sequential envelope has no primary estimand".into(),
            })?,
            estimate,
            refutations,
            diagnostics,
            Some(structural),
            (bootstrap_replicates > 0).then_some(block.completed),
            Some(crate::Identification::TemporalEnvelope {
                envelope: operation.bundle().envelope.clone(),
                strategy: identifier_id,
                structure_version: operation.structure_version(),
            }),
        ))
    }

    fn execute_conditional(
        &self,
        started: Instant,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        refute: RefuteSuite,
    ) -> Result<StudyResult, CausalError> {
        let operation = &self.operation;
        let query = operation.query();
        let (estimate, estimand, identification, diagnostics, structural) =
            super::temporal_class_point::assemble_temporal_class_conditional(
                data,
                query,
                &operation.bundle().envelope,
                operation.graph_class(),
                refute,
                &self.result_context.inference,
                &ctx.kernel_policy,
            )?;
        let mut result = self.finish(
            started,
            identifier_id,
            estimator_id,
            estimand,
            estimate,
            Vec::new(),
            diagnostics,
            Some(structural),
            None,
            Some(crate::Identification::TemporalEnvelope {
                envelope: operation.bundle().envelope.clone(),
                strategy: identifier_id,
                structure_version: operation.structure_version(),
            }),
        );
        let _ = identification;
        result.interval = None;
        result.posterior = None;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        started: Instant,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        estimand: IdentifiedEstimand,
        estimate: EffectEstimate,
        refutations: Vec<antecedent_validate::RefutationReport>,
        diagnostics: Vec<Diagnostic>,
        structural: Option<crate::result::StructuralResponseMixture>,
        bootstrap_replicates_ok: Option<u32>,
        certificate: Option<crate::Identification>,
    ) -> StudyResult {
        let query = self.operation.query();
        let mut identification = self.operation.identification().clone();
        let envelope = &self.operation.bundle().envelope.envelope;
        identification.status = envelope.status;
        let args = IdentifiedExecuteFinish::new(
            &self.physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            query.treatment,
            query.outcome,
        )
        .identify_cached(true)
        .refutations(refutations)
        .wall_time_ns(elapsed_ns(started))
        .bootstrap_replicates_ok(bootstrap_replicates_ok)
        .extras(IdentifiedExecuteExtras {
            certificate,
            structural_response: structural,
            diagnostics: Some(diagnostics),
            estimate_provenance: Some(provenance_ids(
                "estimate.temporal_class.envelope",
                estimator_id.as_str(),
            )),
            ..Default::default()
        });
        super::finish_identified_execute_with_context(&self.result_context, None, args)
    }
}
