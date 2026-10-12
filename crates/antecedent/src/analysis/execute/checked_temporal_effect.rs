//! Builder-independent execution of a checked temporal effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::temporal_effect_core::FitFinishMeta;
use super::{
    Arc, CausalError, ExecutionContext, IdentifiedResultContext, InferenceMode, Instant,
    PhysicalExecutionPlan, StudyResult, TimeSeriesData, elapsed_ns,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::Procedure;

#[derive(Clone)]
pub(crate) struct CheckedTemporalEffectExecution {
    operation: crate::analysis::CheckedTemporalEffectOperation,
    result_context: IdentifiedResultContext,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
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
        physical: std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        if !operation.matches_query(&result_context.query)
            || !operation.matches_query(&physical.logical.query)
        {
            return Err(CausalError::Compile {
                message: "temporal effect result context does not match its checked query".into(),
            });
        }
        let procedure = operation.procedure();
        let (identifier, estimator) = (procedure.identifier, procedure.estimator);
        if !physical.binding().names(identifier, estimator) {
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
        let procedure = op.procedure();
        let (suite, bootstrap_replicates) = (procedure.validation, procedure.bootstrap_replicates);
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
            FitFinishMeta {
                identify_cached: true,
                structure_version: self.result_context.graph_version,
                wall_time_ns: elapsed_ns(started),
            },
        );
        Ok(seal_point_only(
            point_only,
            super::finish_identified_execute_with_context(&self.result_context, None, args),
        ))
    }
}

/// Read-only target, proof, and procedure of a prepared `TemporalDag` effect.
#[derive(Clone, Debug)]
pub struct CheckedTemporalDagEffectInfo {
    /// Exact temporal policy, intervention levels, horizon, and population.
    pub query: antecedent_core::TemporalEffectQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
    /// Graph-derived adjustment variables in the unfolded proof.
    pub adjustment_set: std::sync::Arc<[antecedent_core::TemporalNodeKey]>,
    /// Canonical node and edge signature of the prepared temporal graph.
    pub graph_signature: std::sync::Arc<str>,
}

impl Inspect for CheckedTemporalEffectExecution {
    type Info = CheckedTemporalDagEffectInfo;

    fn info(&self) -> CheckedTemporalDagEffectInfo {
        let operation = self.operation();
        let Procedure { identifier, estimator, bootstrap_replicates, validation } =
            operation.procedure();
        let adjustment_set = operation
            .estimand()
            .adjustment_set
            .iter()
            .filter_map(|id| operation.indexer().key_of(id.raw()).ok())
            .collect::<Vec<_>>();
        CheckedTemporalDagEffectInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            validation,
            bootstrap_replicates,
            adjustment_set: adjustment_set.into(),
            graph_signature: Arc::from(operation.graph_signature()),
        }
    }
}
