//! Builder independent execution of checked temporal class effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::temporal_effect_core::FitFinishMeta;
use super::{
    Arc, CausalError, CausalQuery, ExecutionContext, IdentifiedResultContext, InferenceMode,
    Instant, PhysicalExecutionPlan, StudyResult, TimeSeriesData, elapsed_ns,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::Procedure;

/// Checked class envelope together with its result and physical contracts.
#[derive(Clone)]
pub(crate) struct CheckedTemporalClassEffectExecution {
    operation: crate::analysis::CheckedTemporalClassEffectOperation,
    result_context: IdentifiedResultContext,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
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
        physical: std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let query = CausalQuery::TemporalEffect(operation.query().clone());
        let procedure = operation.procedure();
        let (identifier, estimator) = (procedure.identifier, procedure.estimator);
        if !operation.matches_query(&result_context.query)
            || result_context.query != query
            || physical.logical.query != query
            || !physical.binding().names(identifier, estimator)
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
        let procedure = operation.procedure();
        let (identifier_id, bootstrap_replicates, refute) =
            (procedure.identifier, procedure.bootstrap_replicates, procedure.validation);
        let (split, custom_validators) = (procedure.split, procedure.custom_validators);
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
            FitFinishMeta {
                identify_cached: true,
                structure_version: operation.structure_version(),
                wall_time_ns: elapsed_ns(started),
            },
        );
        Ok(super::temporal_effect_core::seal_point_only(
            point_only,
            super::finish_identified_execute_with_context(&self.result_context, None, args),
        ))
    }
}

/// Read-only source graph, completion proof, and procedure for a temporal
/// Cpdag/Pag pulse or sustained effect.
#[derive(Clone, Debug)]
pub struct CheckedTemporalClassEffectInfo {
    /// Exact policy, intervention levels, horizon, and target population.
    pub query: antecedent_core::TemporalEffectQuery,
    /// Source class retained with the completion proof.
    pub graph_class: crate::accepted::GraphClass,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
    /// Number of completion atoms retained by the checked proof.
    pub completion_count: usize,
    /// Maximum number of class completions searched, when configured.
    pub completion_limit: Option<usize>,
    /// Identified and unresolved enumeration mass retained by the proof.
    pub identified_mass: f64,
    /// Unidentified enumeration mass retained by the proof.
    pub unidentified_mass: f64,
    /// Number of completions whose enumeration was cut off.
    pub truncated_completions: usize,
    /// Source graph version sealed with the proof.
    pub graph_version: u64,
}

impl Inspect for CheckedTemporalClassEffectExecution {
    type Info = CheckedTemporalClassEffectInfo;

    fn info(&self) -> CheckedTemporalClassEffectInfo {
        let operation = self.operation();
        let Procedure { identifier, estimator, bootstrap_replicates, validation } =
            *operation.procedure();
        let envelope = &operation.bundle().envelope.envelope;
        CheckedTemporalClassEffectInfo {
            query: operation.query().clone(),
            graph_class: operation.graph_class(),
            identifier,
            estimator,
            validation,
            bootstrap_replicates,
            completion_count: envelope.cases.len(),
            completion_limit: operation.max_completions(),
            identified_mass: envelope.identified_weight.0,
            unidentified_mass: envelope.unidentified_weight.0,
            truncated_completions: envelope.truncated_completions,
            graph_version: u64::from(operation.structure_version()),
        }
    }
}
