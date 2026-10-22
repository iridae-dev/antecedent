//! Typed accessors over the sealed [`PreparedExecution`] route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CheckedAdmgResponseCurveOperation, CheckedAipwOperation, CheckedBayesianGcompOperation,
    CheckedDistributionOperation, CheckedFrontDoorOperation, CheckedFunctionalEffectOperation,
    CheckedIvOperation, CheckedLinearOperation, CheckedPathSpecificEffectOperation,
    CheckedProgramBinding, CheckedStaticResponseCurve, PreparedExecution,
};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use antecedent_data::TabularData;
use antecedent_estimate::AipwAte;
use std::sync::Arc;

impl PreparedExecution {
    pub(super) fn set_stage_sink(
        &mut self,
        sink: Option<Arc<dyn crate::analysis::stage::StageResultSink>>,
    ) {
        if let Some(operation) = self.operation_mut() {
            operation.set_stage_sink(sink);
        }
    }

    /// Push rebound caller validators into the sealed operations that run
    /// them from their own frozen copy.
    pub(super) fn set_custom_validators(
        &mut self,
        validators: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    ) {
        if let Some(operation) = self.operation_mut() {
            operation.set_custom_validators(validators);
        }
    }

    pub(crate) fn program_binding(&self) -> CheckedProgramBinding<'_> {
        self.operation().program_binding()
    }

    /// Rebind only the checked receipts that still execute through the shared
    /// tabular dispatcher; `None` for every other variant, which executes as
    /// prepared. Direct operations own their rebind at execution.
    pub(super) fn rebind_for_dispatch(
        &self,
        data: &TabularData,
    ) -> Result<Option<Self>, CausalError> {
        self.operation().rebind_for_dispatch(data)
    }

    pub(crate) fn checked_linear(
        &self,
    ) -> Option<&antecedent_estimate::CheckedLinearAdjustmentAte> {
        self.linear_operation().map(|operation| &operation.preparation)
    }
    pub(crate) fn checked_aipw(&self) -> Option<&antecedent_estimate::CheckedAipwPreparation> {
        self.aipw_operation().map(|operation| &operation.preparation)
    }
    pub(crate) fn bayesian_gcomp(&self) -> Option<&CheckedBayesianGcompOperation> {
        if let Self::BayesianGcomp(value) = self { Some(value.operation()) } else { None }
    }
}

/// Typed read access to the operation a variant retains. Each accessor is
/// `None` for every other route; the binding between a name and its variant is
/// stated once, here.
macro_rules! typed_accessors {
    ($($name:ident -> $operation:ty = $variant:ident;)+) => {
        impl PreparedExecution {
            $(
                pub(crate) fn $name(&self) -> Option<&$operation> {
                    if let Self::$variant(value) = self {
                        let operation: &$operation = value;
                        Some(operation)
                    } else {
                        None
                    }
                }
            )+
        }
    };
}

typed_accessors! {
    linear_operation -> CheckedLinearOperation = CheckedLinear;
    aipw_operation -> CheckedAipwOperation = CheckedAipw;
    nested_counterfactual -> crate::gcm::NestedCounterfactualOperation = NestedCounterfactual;
    distribution -> CheckedDistributionOperation = Distribution;
    functional_effect_operation -> CheckedFunctionalEffectOperation = FunctionalEffect;
    path_specific_effect_operation -> CheckedPathSpecificEffectOperation = PathSpecificEffect;
    admg_response_curve -> CheckedAdmgResponseCurveOperation = AdmgResponseCurve;
    response_curve -> CheckedStaticResponseCurve = StaticResponseCurve;
    frontdoor_linear -> CheckedFrontDoorOperation = FrontDoorLinear;
    iv -> CheckedIvOperation = Iv;
}

#[cfg(test)]
impl super::PreparedStudy {
    /// The static identification allocation the sealed static-ATE operation
    /// retains, for sharing assertions.
    pub(crate) fn sealed_static_identification(
        &self,
    ) -> Option<&Arc<super::CachedStaticIdentification>> {
        match &self.execution {
            PreparedExecution::CheckedLinear(operation) => Some(&operation.cache),
            PreparedExecution::CheckedGlmAdjustment(operation) => Some(&operation.cache),
            PreparedExecution::CheckedAipw(operation) => Some(&operation.cache),
            PreparedExecution::CheckedConditional(operation) => Some(&operation.cache),
            _ => None,
        }
    }
}

pub(crate) fn checked_aipw_fitter(study: &Study) -> AipwAte {
    if let Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) =
        study.estimator_spec.as_deref()
    {
        return (**config).clone();
    }
    let mut fitter = AipwAte::new();
    fitter.sampling_design = crate::estimator_spec::declared_sampling(study.sampling_design);
    fitter.bootstrap_replicates = study.bootstrap_replicates;
    if let Some(overlap) = study.overlap_policy {
        fitter.overlap = overlap;
    }
    fitter.population_registry.clone_from(&study.population_registry);
    fitter
}
