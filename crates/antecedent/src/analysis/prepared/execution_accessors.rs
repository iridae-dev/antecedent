//! Typed accessors over the sealed [`PreparedExecution`] route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CheckedAdmgResponseCurveOperation, CheckedAipwOperation, CheckedBayesianGcompOperation,
    CheckedDistributionOperation, CheckedFrontDoorOperation, CheckedFunctionalEffectOperation,
    CheckedIvOperation, CheckedLinearOperation, CheckedPathSpecificEffectOperation,
    CheckedProgramBinding, CheckedStaticResponseCurve, PreparedExecution,
};
use crate::analysis::checked_propensity::CheckedPropensityOperation;
use crate::analysis::execute::Study;
use crate::analysis::{
    CheckedAdmgGraphPosteriorResponse, CheckedClassGraphPosteriorEffect,
    CheckedConditionalOperation, CheckedGraphPosteriorEffect, CheckedStaticClassEffect,
};
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
    admg_graph_posterior_response -> CheckedAdmgGraphPosteriorResponse = AdmgGraphPosteriorResponse;
    static_class_response -> crate::analysis::execute::CheckedStaticClassResponse = StaticClassResponse;
    graph_posterior_response -> crate::analysis::execute::CheckedGraphPosteriorResponse = GraphPosteriorResponse;
    bayesian_conditional -> crate::analysis::execute::CheckedBayesianConditionalOperation = BayesianConditional;
    response_curve -> CheckedStaticResponseCurve = StaticResponseCurve;
    static_dag_response -> crate::analysis::execute::CheckedStaticDagResponseOperation = StaticDagResponse;
    static_mediation -> crate::analysis::execute::CheckedStaticMediationOperation = StaticMediation;
    bayesian_static_mediation -> crate::analysis::execute::CheckedBayesianStaticMediationOperation = BayesianStaticMediation;
    attribution -> crate::analysis::execute::CheckedAttributionOperation = Attribution;
    temporal_dag_response -> crate::analysis::execute::CheckedTemporalResponseExecution = TemporalDagResponse;
    temporal_class_response -> crate::analysis::CheckedTemporalClassResponseOperation = TemporalClassResponse;
    temporal_graph_posterior_response -> crate::analysis::CheckedTemporalGraphPosteriorResponse = TemporalGraphPosteriorResponse;
    temporal_dag_effect -> crate::analysis::execute::CheckedTemporalEffectExecution = TemporalDagEffect;
    bayesian_temporal_dag_effect -> crate::analysis::CheckedBayesianTemporalEffectOperation = BayesianTemporalDagEffect;
    temporal_mediation -> crate::analysis::execute::CheckedTemporalMediationOperation = TemporalMediation;
    temporal_class_mediation -> crate::analysis::execute::CheckedTemporalClassMediationOperation = TemporalClassMediation;
    interference -> crate::analysis::execute::CheckedInterferenceOperation = Interference;
    temporal_class_effect -> crate::analysis::execute::CheckedTemporalClassEffectExecution = TemporalClassEffect;
    propensity -> CheckedPropensityOperation = CheckedPropensity;
    conditional -> CheckedConditionalOperation = CheckedConditional;
    bayesian_class_conditional -> crate::analysis::CheckedBayesianClassConditional = BayesianClassConditional;
    static_class_effect -> CheckedStaticClassEffect = StaticClassEffect;
    graph_posterior_effect -> CheckedGraphPosteriorEffect = GraphPosteriorEffect;
    class_graph_posterior_effect -> CheckedClassGraphPosteriorEffect = ClassGraphPosteriorEffect;
    frontdoor_linear -> CheckedFrontDoorOperation = FrontDoorLinear;
    iv -> CheckedIvOperation = Iv;
    bayesian_temporal_class_effect -> crate::analysis::CheckedBayesianTemporalClassEffectOperation = BayesianTemporalClassEffect;
    temporal_graph_posterior_effect -> crate::analysis::CheckedTemporalGraphPosteriorEffect = TemporalGraphPosteriorEffect;
}

pub(crate) fn checked_aipw_fitter(study: &Study) -> AipwAte {
    if let Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) = &study.estimator_spec {
        return (**config).clone();
    }
    let mut fitter = AipwAte::new();
    fitter.bootstrap_replicates = study.bootstrap_replicates;
    if let Some(overlap) = study.overlap_policy {
        fitter.overlap = overlap;
    }
    fitter.population_registry.clone_from(&study.population_registry);
    fitter
}
