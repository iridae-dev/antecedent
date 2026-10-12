//! Builder-independent execution of a checked temporal effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[derive(Clone)]
pub(crate) struct CheckedTemporalEffectExecution {
    operation: crate::analysis::CheckedTemporalEffectOperation,
    result_context: IdentifiedResultContext,
    physical: PhysicalExecutionPlan,
}

impl std::fmt::Debug for CheckedTemporalEffectExecution {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("CheckedTemporalEffectExecution")
            .field("operation", &self.operation)
            .field("physical", &self.physical)
            .finish_non_exhaustive()
    }
}

impl CheckedTemporalEffectExecution {
    pub(crate) fn checked(
        operation: crate::analysis::CheckedTemporalEffectOperation,
        result_context: IdentifiedResultContext,
        physical: PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        if !operation.matches_query(&result_context.query)
            || !operation.matches_query(&physical.logical.query)
        {
            return Err(CausalError::Compile {
                message: "temporal effect result context does not match its checked query".into(),
            });
        }
        let (identifier, estimator, _, _) = operation.procedure();
        if physical
            .logical
            .record
            .identifier
            .as_deref()
            .is_some_and(|name| name != identifier.as_str())
            || physical
                .logical
                .record
                .estimator
                .as_deref()
                .is_some_and(|name| name != estimator.as_str())
        {
            return Err(CausalError::Compile {
                message: "temporal effect physical plan disagrees on its checked procedure".into(),
            });
        }
        Ok(Self { operation, result_context, physical })
    }

    #[must_use]
    pub(crate) fn operation(&self) -> &crate::analysis::CheckedTemporalEffectOperation {
        &self.operation
    }

    pub(crate) fn execute(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use super::temporal_effect_core::{
            TemporalEffectInputs, fit_temporal_effect, plan_completed_to_dag, seal_point_only,
        };
        let started = Instant::now();
        let op = &self.operation;
        let (_, _, suite, bootstrap_replicates) = op.procedure();
        let inference = InferenceMode::Frequentist;
        let fit = fit_temporal_effect(
            &TemporalEffectInputs {
                data,
                graph: op.graph(),
                query: op.query(),
                identification: op.identification(),
                estimand: op.estimand(),
                indexer: op.indexer(),
                inference: &inference,
                checked_bayesian: None,
                bootstrap_replicates,
                refute: suite,
                custom_validators: &[],
                split: None,
                latency_mode: None,
                completed_to_dag: plan_completed_to_dag(&self.physical),
            },
            ctx,
        )?;
        let point_only = fit.point_only;
        let args = fit.finish_args(
            &self.physical,
            op.identification().clone(),
            op.estimand().clone(),
            op.indexer(),
            op.query(),
            true,
            self.result_context.graph_version,
            elapsed_ns(started),
        );
        Ok(seal_point_only(
            point_only,
            super::finish_identified_execute_with_context(&self.result_context, None, args),
        ))
    }
}
