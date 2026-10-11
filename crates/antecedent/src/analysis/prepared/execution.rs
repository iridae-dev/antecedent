//! The sealed execution route a prepared handle dispatches on.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CheckedAdmgResponseCurveOperation, CheckedAipwOperation, CheckedDistributionOperation,
    CheckedFrontDoorOperation, CheckedFunctionalEffectOperation,
    CheckedFunctionalEffectResponseMember, CheckedGlmAdjustmentOperation, CheckedIvOperation,
    CheckedLinearOperation, CheckedPathSpecificEffectOperation, CheckedRdOperation,
    CheckedStaticResponseCurve, CheckedUnknownTieredAverageOperation,
};
use crate::analysis::checked_propensity::CheckedPropensityOperation;
use crate::analysis::{
    CheckedAdmgGraphPosteriorResponse, CheckedBayesianGraphPosteriorAte,
    CheckedClassGraphPosteriorEffect, CheckedConditionalOperation, CheckedGraphPosteriorEffect,
    CheckedStaticClassEffect,
};

/// One sealed execution choice for a prepared click. The checked variants carry
/// the complete operation retained by preparation; `OrdinaryDispatch` names a
/// configuration no sealed operation admits, which executes through the
/// ordinary dispatcher.
#[derive(Clone, Debug)]
pub(crate) enum PreparedExecution {
    /// No sealed operation admits this configuration; execute through the
    /// ordinary dispatcher.
    OrdinaryDispatch,
    FunctionalEffect(CheckedFunctionalEffectOperation),
    PathSpecificEffect(CheckedPathSpecificEffectOperation),
    AdmgResponseCurve(CheckedAdmgResponseCurveOperation),
    AdmgGraphPosteriorResponse(Box<CheckedAdmgGraphPosteriorResponse>),
    CheckedLinear(CheckedLinearOperation),
    CheckedGlmAdjustment(CheckedGlmAdjustmentOperation),
    CheckedRd(CheckedRdOperation),
    CheckedPropensity(CheckedPropensityOperation),
    CheckedConditional(CheckedConditionalOperation),
    BayesianConditional(crate::analysis::execute::CheckedBayesianConditionalOperation),
    GraphPosteriorEffect(CheckedGraphPosteriorEffect),
    ClassGraphPosteriorEffect(CheckedClassGraphPosteriorEffect),
    BayesianGraphPosteriorAte(Box<CheckedBayesianGraphPosteriorAte>),
    StaticClassEffect(CheckedStaticClassEffect),
    CheckedAipw(CheckedAipwOperation),
    Counterfactual(crate::analysis::execute::CheckedCounterfactualPlan),
    NestedCounterfactual(crate::gcm::NestedCounterfactualOperation),
    DerivativeResponse(crate::analysis::execute::CheckedDerivativeResponseOperation),
    CellAipwResponse(crate::analysis::execute::CheckedCellAipwResponseOperation),
    StaticDagResponse(crate::analysis::execute::CheckedStaticDagResponseOperation),
    StaticMediation(crate::analysis::execute::CheckedStaticMediationOperation),
    BayesianStaticMediation(Box<crate::analysis::execute::CheckedBayesianStaticMediationOperation>),
    Attribution(crate::analysis::execute::CheckedAttributionOperation),
    TemporalDagResponse(crate::analysis::execute::CheckedTemporalResponseExecution),
    TemporalDagEffect(crate::analysis::execute::CheckedTemporalEffectExecution),
    BayesianTemporalDagEffect(Box<crate::analysis::CheckedBayesianTemporalEffectOperation>),
    TemporalMediation(crate::analysis::execute::CheckedTemporalMediationOperation),
    Interference(crate::analysis::execute::CheckedInterferenceOperation),
    Randomized(crate::analysis::execute::CheckedRandomizedOperation),
    PolicyValue(crate::analysis::execute::CheckedPolicyValueOperation),
    ContinuousDoseResponse(crate::analysis::execute::CheckedContinuousDoseOperation),
    PanelDid(crate::analysis::execute::CheckedPanelDidOperation),
    SyntheticControl(crate::analysis::execute::CheckedSyntheticControlOperation),
    LocalPolynomialRatio(crate::analysis::execute::CheckedLocalPolynomialRatioOperation),
    Survival(crate::analysis::execute::CheckedSurvivalOperation),
    LongitudinalRegime(crate::analysis::execute::CheckedLongitudinalRegimeOperation),
    TemporalClassEffect(crate::analysis::execute::CheckedTemporalClassEffectExecution),
    Distribution(CheckedDistributionOperation),
    BayesianGcomp(crate::analysis::execute::CheckedBayesianDagAteExecution),
    BayesianBasisAte(crate::analysis::execute::CheckedBayesianBasisAteExecution),
    BayesianBasisCate(crate::analysis::execute::CheckedBayesianBasisCateExecution),
    BayesianRobustAte(crate::analysis::execute::CheckedBayesianRobustAteExecution),
    StaticResponseCurve(CheckedStaticResponseCurve),
    FrontDoorLinear(CheckedFrontDoorOperation),
    Iv(CheckedIvOperation),
    UnknownTieredAverage(Box<CheckedUnknownTieredAverageOperation>),
    BayesianClassConditional(crate::analysis::CheckedBayesianClassConditional),
    TemporalClassResponse(Box<crate::analysis::CheckedTemporalClassResponseOperation>),
    TemporalGraphPosteriorResponse(Box<crate::analysis::CheckedTemporalGraphPosteriorResponse>),
    TemporalClassMediation(Box<crate::analysis::execute::CheckedTemporalClassMediationOperation>),
    BayesianTemporalClassEffect(Box<crate::analysis::CheckedBayesianTemporalClassEffectOperation>),
    TemporalGraphPosteriorEffect(Box<crate::analysis::CheckedTemporalGraphPosteriorEffect>),
    StaticClassResponse(Box<crate::analysis::execute::CheckedStaticClassResponse>),
    GraphPosteriorResponse(Box<crate::analysis::execute::CheckedGraphPosteriorResponse>),
    BayesianSpecialist(Box<crate::analysis::execute::CheckedBayesianSpecialistOperation>),
    TransportTrial(Box<crate::analysis::execute::CheckedTransportTrialOperation>),
}

/// Exactly one checked product may supply a prepared result contract. This is
/// deliberately a sum type: passing seven independent optional receipts to
/// contract construction could describe a procedure no prepared plan selected.
#[derive(Clone, Copy)]
pub(crate) enum CheckedProgramBinding<'a> {
    None,
    Aipw(&'a antecedent_estimate::CheckedAipwPreparation),
    FrontDoor(&'a antecedent_estimate::CheckedFrontDoorPreparation),
    Iv(&'a antecedent_estimate::CheckedIvPreparation),
    Linear(&'a CheckedLinearOperation),
    FunctionalEffect(&'a antecedent_expr::FunctionalProgram),
    FunctionalResponse(&'a [CheckedFunctionalEffectResponseMember]),
    Distribution(&'a antecedent_expr::FunctionalProgram),
    NestedCounterfactual(&'a crate::gcm::NestedCounterfactualOperation),
}
