//! Checked counterfactual execution plans and inference procedures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalQuery, EstimatorId, ExecutionContext, IdentificationStatus,
    IdentifiedResultContext, IdentifierId, InferenceMode, PhysicalExecutionPlan, StudyResult,
    TabularData, bayesian_draw_count,
};
use crate::analysis::proof_support::proof_supplies_full;

/// Inference procedure over the heterogeneity best-score counterfactual fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CounterfactualProcedure {
    Point,
    DirichletRowWeights {
        draws: usize,
    },
    /// Frequentist whole-procedure multinomial row bootstrap of the mean ITE.
    RowBootstrap {
        replicates: u32,
    },
}

impl CounterfactualProcedure {
    pub(crate) fn for_inference(inference: &InferenceMode) -> Result<Self, CausalError> {
        match inference {
            InferenceMode::Frequentist => Ok(Self::Point),
            InferenceMode::Bayesian(config) => {
                if config.has_transfer() {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian counterfactuals require a declared mechanism mapping; a coefficient artifact cannot be applied as an isotropic GCM prior or hydrated onto fitted GCM mechanisms",
                    });
                }
                let draws = bayesian_draw_count(inference)?;
                if draws < 2 {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian counterfactuals require at least two posterior draws",
                    });
                }
                Ok(Self::DirichletRowWeights { draws })
            }
        }
    }

    /// The Frequentist point procedure becomes the mechanism-refit row bootstrap when a
    /// replicate count is requested and the declared design resamples subjects.
    ///
    /// # Errors
    ///
    /// A declared design that does not allow a subject resample.
    pub(crate) fn with_bootstrap(
        self,
        replicates: u32,
        design: Option<antecedent_core::DeclaredSamplingDesign>,
    ) -> Result<Self, CausalError> {
        match self {
            Self::Point if replicates > 0 => {
                super::super::gcm_bootstrap::require_subject_resampling(design)?;
                Ok(Self::RowBootstrap { replicates })
            }
            other => Ok(other),
        }
    }

    pub(super) fn matches_inference(&self, inference: &InferenceMode) -> bool {
        match (self, inference) {
            (Self::Point | Self::RowBootstrap { .. }, InferenceMode::Frequentist) => true,
            (Self::DirichletRowWeights { draws }, InferenceMode::Bayesian(config)) => {
                *draws == config.n_draws
            }
            _ => false,
        }
    }
}

#[derive(Clone)]
pub(crate) struct CheckedCounterfactualPlan {
    pub(super) operation: crate::gcm::CheckedCounterfactualOperation,
    pub(super) cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    pub(super) identify_cached: bool,
    pub(super) inference: InferenceMode,
    pub(super) procedure: CounterfactualProcedure,
    pub(super) physical: std::sync::Arc<PhysicalExecutionPlan>,
    pub(super) result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedCounterfactualPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedCounterfactualPlan")
            .field("operation", &self.operation)
            .field("identification", &self.cache.identification)
            .field("estimand", &self.cache.estimand)
            .field("identify_cached", &self.identify_cached)
            .field("inference", &self.inference)
            .field("procedure", &self.procedure)
            .field("physical", &self.physical)
            .finish_non_exhaustive()
    }
}

impl CheckedCounterfactualPlan {
    pub(crate) fn new(
        operation: crate::gcm::CheckedCounterfactualOperation,
        cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
        identify_cached: bool,
        inference: InferenceMode,
        procedure: CounterfactualProcedure,
        physical: std::sync::Arc<PhysicalExecutionPlan>,
        result_context: IdentifiedResultContext,
    ) -> Result<Self, CausalError> {
        let (identification, estimand) = (&cache.identification, &cache.estimand);
        if !matches!(&identification.query, CausalQuery::Counterfactual(query) if query == operation.query())
            || result_context.query != CausalQuery::Counterfactual(operation.query().clone())
            || result_context.inference != inference
            || !procedure.matches_inference(&inference)
            || !matches!(
                identification.status,
                IdentificationStatus::IdentifiedUnderParametricRestrictions
            )
            || !proof_supplies_full(identification, estimand)
            || !physical.binding().names(IdentifierId::GcmParametric, EstimatorId::GcmFit)
        {
            return Err(CausalError::Compile { message: "checked counterfactual plan components do not agree on target, inference, identification, or procedure".into() });
        }
        Ok(Self {
            operation,
            cache,
            identify_cached,
            inference,
            procedure,
            physical,
            result_context,
        })
    }

    pub(crate) fn operation(&self) -> &crate::gcm::CheckedCounterfactualOperation {
        &self.operation
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::execute_checked_counterfactual_plan(self, data, ctx)
    }
}
