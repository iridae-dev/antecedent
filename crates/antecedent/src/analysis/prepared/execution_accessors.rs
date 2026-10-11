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
        match self {
            // These operations own the executable copy of the click controls.
            // Updating only Study would leave a sealed route unable to stream.
            Self::CheckedLinear(operation) => operation.stage_sink = sink,
            Self::BayesianGcomp(operation) => operation.set_stage_sink(sink),
            Self::BayesianConditional(operation) => operation.set_stage_sink(sink),
            _ => {}
        }
    }

    /// Push rebound caller validators into the sealed operations that run
    /// them from their own frozen copy.
    pub(super) fn set_custom_validators(
        &mut self,
        validators: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    ) {
        match self {
            Self::BayesianGcomp(operation) => operation.set_custom_validators(validators.to_vec()),
            Self::BayesianConditional(operation) => {
                operation.set_custom_validators(validators.to_vec());
            }
            Self::TemporalClassEffect(operation) => {
                operation.set_custom_validators(validators.to_vec());
            }
            _ => {}
        }
    }

    pub(crate) fn program_binding(&self) -> CheckedProgramBinding<'_> {
        match self {
            Self::CheckedAipw(operation) => CheckedProgramBinding::Aipw(&operation.preparation),
            Self::FrontDoorLinear(operation) => {
                CheckedProgramBinding::FrontDoor(&operation.preparation)
            }
            Self::Iv(operation) => CheckedProgramBinding::Iv(operation.preparation()),
            Self::CheckedLinear(operation) => CheckedProgramBinding::Linear(operation),
            Self::FunctionalEffect(operation) => {
                CheckedProgramBinding::FunctionalEffect(operation.prepared.program())
            }
            Self::PathSpecificEffect(operation) => {
                CheckedProgramBinding::FunctionalEffect(operation.prepared.program())
            }
            Self::AdmgResponseCurve(operation) => {
                CheckedProgramBinding::FunctionalResponse(&operation.members)
            }
            Self::Distribution(operation) => {
                CheckedProgramBinding::Distribution(operation.prepared.program())
            }
            Self::NestedCounterfactual(operation) => {
                CheckedProgramBinding::NestedCounterfactual(operation)
            }
            Self::OrdinaryDispatch
            | Self::Counterfactual(_)
            | Self::DerivativeResponse(_)
            | Self::CellAipwResponse(_)
            | Self::StaticDagResponse(_)
            | Self::StaticMediation(_)
            | Self::BayesianStaticMediation(_)
            | Self::Attribution(_)
            | Self::TemporalDagResponse(_)
            | Self::TemporalDagEffect(_)
            | Self::BayesianTemporalDagEffect(_)
            | Self::TemporalMediation(_)
            | Self::Interference(_)
            | Self::Randomized(_)
            | Self::PolicyValue(_)
            | Self::ContinuousDoseResponse(_)
            | Self::PanelDid(_)
            | Self::SyntheticControl(_)
            | Self::LocalPolynomialRatio(_)
            | Self::Survival(_)
            | Self::LongitudinalRegime(_)
            | Self::TemporalClassEffect(_)
            | Self::BayesianGcomp(_)
            | Self::BayesianBasisAte(_)
            | Self::BayesianBasisCate(_)
            | Self::BayesianRobustAte(_)
            | Self::BayesianGraphPosteriorAte(_)
            | Self::BayesianConditional(_)
            | Self::StaticResponseCurve(_)
            | Self::GraphPosteriorEffect(_)
            | Self::StaticClassEffect(_)
            | Self::ClassGraphPosteriorEffect(_)
            | Self::StaticClassResponse(_)
            | Self::GraphPosteriorResponse(_)
            | Self::CheckedGlmAdjustment(_)
            | Self::CheckedRd(_)
            | Self::CheckedPropensity(_)
            | Self::CheckedConditional(_)
            | Self::UnknownTieredAverage(_)
            | Self::BayesianSpecialist(_)
            | Self::TransportTrial(_)
            | Self::AdmgGraphPosteriorResponse(_)
            | Self::BayesianClassConditional(_)
            | Self::TemporalClassResponse(_)
            | Self::TemporalGraphPosteriorResponse(_)
            | Self::TemporalClassMediation(_)
            | Self::BayesianTemporalClassEffect(_)
            | Self::TemporalGraphPosteriorEffect(_) => CheckedProgramBinding::None,
        }
    }

    /// Rebind only the checked receipts that still execute through the shared
    /// tabular dispatcher; `None` for every other variant, which executes as
    /// prepared. Direct operations own their rebind at execution. Keeping the
    /// variant selection here, and only here, prevents a refreshed receipt
    /// from accidentally changing the procedure selected at preparation.
    pub(super) fn rebind_for_dispatch(
        &self,
        data: &TabularData,
    ) -> Result<Option<Self>, CausalError> {
        Ok(Some(match self {
            Self::CheckedLinear(operation) => Self::CheckedLinear(operation.rebind(data)?),
            Self::CheckedGlmAdjustment(operation) => {
                Self::CheckedGlmAdjustment(operation.rebind(data)?)
            }
            Self::CheckedRd(operation) => Self::CheckedRd(operation.rebind(data)?),
            Self::CheckedAipw(operation) => Self::CheckedAipw(operation.rebind(data)?),
            Self::FrontDoorLinear(operation) => Self::FrontDoorLinear(operation.rebind(data)?),
            Self::Iv(operation) => Self::Iv(operation.rebind(data)?),
            _ => return Ok(None),
        }))
    }

    pub(crate) fn checked_linear(
        &self,
    ) -> Option<&antecedent_estimate::CheckedLinearAdjustmentAte> {
        self.linear_operation().map(|operation| &operation.preparation)
    }
    pub(crate) fn linear_operation(&self) -> Option<&CheckedLinearOperation> {
        if let Self::CheckedLinear(value) = self { Some(value) } else { None }
    }
    pub(crate) fn checked_aipw(&self) -> Option<&antecedent_estimate::CheckedAipwPreparation> {
        self.aipw_operation().map(|operation| &operation.preparation)
    }
    pub(crate) fn aipw_operation(&self) -> Option<&CheckedAipwOperation> {
        if let Self::CheckedAipw(value) = self { Some(value) } else { None }
    }
    pub(crate) fn nested_counterfactual(
        &self,
    ) -> Option<&crate::gcm::NestedCounterfactualOperation> {
        if let Self::NestedCounterfactual(value) = self { Some(value) } else { None }
    }
    pub(crate) fn distribution(&self) -> Option<&CheckedDistributionOperation> {
        if let Self::Distribution(value) = self { Some(value) } else { None }
    }
    pub(crate) fn functional_effect_operation(&self) -> Option<&CheckedFunctionalEffectOperation> {
        if let Self::FunctionalEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn path_specific_effect_operation(
        &self,
    ) -> Option<&CheckedPathSpecificEffectOperation> {
        if let Self::PathSpecificEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn admg_response_curve(&self) -> Option<&CheckedAdmgResponseCurveOperation> {
        if let Self::AdmgResponseCurve(value) = self { Some(value) } else { None }
    }
    pub(crate) fn admg_graph_posterior_response(
        &self,
    ) -> Option<&CheckedAdmgGraphPosteriorResponse> {
        if let Self::AdmgGraphPosteriorResponse(value) = self { Some(value) } else { None }
    }
    pub(crate) fn static_class_response(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedStaticClassResponse> {
        if let Self::StaticClassResponse(value) = self { Some(value) } else { None }
    }
    pub(crate) fn graph_posterior_response(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedGraphPosteriorResponse> {
        if let Self::GraphPosteriorResponse(value) = self { Some(value) } else { None }
    }
    pub(crate) fn bayesian_gcomp(&self) -> Option<&CheckedBayesianGcompOperation> {
        if let Self::BayesianGcomp(value) = self { Some(value.operation()) } else { None }
    }
    pub(crate) fn bayesian_conditional(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedBayesianConditionalOperation> {
        if let Self::BayesianConditional(value) = self { Some(value) } else { None }
    }
    pub(crate) fn response_curve(&self) -> Option<&CheckedStaticResponseCurve> {
        if let Self::StaticResponseCurve(value) = self { Some(value) } else { None }
    }
    pub(crate) fn static_dag_response(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedStaticDagResponseOperation> {
        if let Self::StaticDagResponse(value) = self { Some(value) } else { None }
    }
    pub(crate) fn static_mediation(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedStaticMediationOperation> {
        if let Self::StaticMediation(value) = self { Some(value) } else { None }
    }
    pub(crate) fn bayesian_static_mediation(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedBayesianStaticMediationOperation> {
        if let Self::BayesianStaticMediation(value) = self { Some(value) } else { None }
    }
    pub(crate) fn attribution(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedAttributionOperation> {
        if let Self::Attribution(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_dag_response(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedTemporalResponseExecution> {
        if let Self::TemporalDagResponse(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_class_response(
        &self,
    ) -> Option<&crate::analysis::CheckedTemporalClassResponseOperation> {
        if let Self::TemporalClassResponse(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_graph_posterior_response(
        &self,
    ) -> Option<&crate::analysis::CheckedTemporalGraphPosteriorResponse> {
        if let Self::TemporalGraphPosteriorResponse(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_dag_effect(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedTemporalEffectExecution> {
        if let Self::TemporalDagEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn bayesian_temporal_dag_effect(
        &self,
    ) -> Option<&crate::analysis::CheckedBayesianTemporalEffectOperation> {
        if let Self::BayesianTemporalDagEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_mediation(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedTemporalMediationOperation> {
        if let Self::TemporalMediation(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_class_mediation(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedTemporalClassMediationOperation> {
        if let Self::TemporalClassMediation(value) = self { Some(value) } else { None }
    }
    pub(crate) fn interference(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedInterferenceOperation> {
        if let Self::Interference(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_class_effect(
        &self,
    ) -> Option<&crate::analysis::execute::CheckedTemporalClassEffectExecution> {
        if let Self::TemporalClassEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn propensity(&self) -> Option<&CheckedPropensityOperation> {
        if let Self::CheckedPropensity(value) = self { Some(value) } else { None }
    }
    pub(crate) fn conditional(&self) -> Option<&CheckedConditionalOperation> {
        if let Self::CheckedConditional(value) = self { Some(value) } else { None }
    }
    pub(crate) fn bayesian_class_conditional(
        &self,
    ) -> Option<&crate::analysis::CheckedBayesianClassConditional> {
        if let Self::BayesianClassConditional(value) = self { Some(value) } else { None }
    }
    pub(crate) fn static_class_effect(&self) -> Option<&CheckedStaticClassEffect> {
        if let Self::StaticClassEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn graph_posterior_effect(&self) -> Option<&CheckedGraphPosteriorEffect> {
        if let Self::GraphPosteriorEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn class_graph_posterior_effect(&self) -> Option<&CheckedClassGraphPosteriorEffect> {
        if let Self::ClassGraphPosteriorEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn frontdoor_linear(&self) -> Option<&CheckedFrontDoorOperation> {
        if let Self::FrontDoorLinear(value) = self { Some(value) } else { None }
    }
    pub(crate) fn iv(&self) -> Option<&CheckedIvOperation> {
        if let Self::Iv(value) = self { Some(value) } else { None }
    }
    pub(crate) fn bayesian_temporal_class_effect(
        &self,
    ) -> Option<&crate::analysis::CheckedBayesianTemporalClassEffectOperation> {
        if let Self::BayesianTemporalClassEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_graph_posterior_effect(
        &self,
    ) -> Option<&crate::analysis::CheckedTemporalGraphPosteriorEffect> {
        if let Self::TemporalGraphPosteriorEffect(value) = self { Some(value) } else { None }
    }
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
