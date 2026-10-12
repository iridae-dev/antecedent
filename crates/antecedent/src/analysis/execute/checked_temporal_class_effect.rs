//! Builder independent execution of checked temporal class effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
        use super::temporal_class_effect_core::{
            TemporalClassEffectInputs, fit_temporal_class_effect,
        };
        let started = Instant::now();
        let operation = &self.operation;
        let bundle = operation.bundle();
        let (identifier_id, _, bootstrap_replicates, split, refute, custom_validators) =
            operation.procedure();
        let inference = InferenceMode::Frequentist;
        let fit = fit_temporal_class_effect(
            &TemporalClassEffectInputs {
                data,
                query: operation.query(),
                bundle,
                graph_class: operation.graph_class(),
                inference: &inference,
                class_prior: None,
                bootstrap_replicates,
                split,
                refute,
                custom_validators,
                latency_mode: None,
            },
            ctx,
        )?;
        let point_only = fit.point_only;
        let mut identification = operation.identification().clone();
        identification.status = bundle.envelope.envelope.status;
        let args = fit.finish_args(
            &self.physical,
            identification,
            bundle,
            identifier_id,
            operation.query(),
            true,
            operation.structure_version(),
            elapsed_ns(started),
        );
        Ok(super::temporal_effect_core::seal_point_only(
            point_only,
            super::finish_identified_execute_with_context(&self.result_context, None, args),
        ))
    }
}
