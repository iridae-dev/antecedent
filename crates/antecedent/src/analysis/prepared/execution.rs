//! The sealed execution route a prepared handle dispatches on.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::sealed::SealedOperation;
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

/// Declares [`PreparedExecution`], its fieldless [`ExecutionKind`], and the
/// exhaustive variant-to-[`SealedOperation`] projection from one list.
///
/// Every dispatch over the sealed route goes through that projection, so a
/// variant added here cannot be reached by a match that forgot it: the new
/// payload must implement [`SealedOperation`], and every hook that route does
/// not override is stated by its implementation rather than by a wildcard arm.
macro_rules! sealed_executions {
    ($($variant:ident($payload:ty)),+ $(,)?) => {
        /// One sealed execution choice for a prepared click. Every variant carries
        /// the complete operation retained by preparation; `FreshRun` is the operation
        /// of a run that holds no prepared handle.
        #[derive(Clone, Debug)]
        pub(crate) enum PreparedExecution {
            $($variant($payload),)+
        }

        /// The variant of a [`PreparedExecution`] without its payload.
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub(crate) enum ExecutionKind {
            $($variant,)+
        }

        impl ExecutionKind {
            /// Every kind, in declaration order.
            #[cfg(test)]
            pub(crate) const ALL: &'static [Self] = &[$(Self::$variant,)+];
            /// Number of kinds a preparation can seal.
            pub(super) const SEALED_COUNT: usize = [$(Self::$variant,)+].len();
        }

        impl PreparedExecution {
            /// The payload-free variant of this route.
            pub(crate) fn kind(&self) -> ExecutionKind {
                match self {
                    $(Self::$variant(_) => ExecutionKind::$variant,)+
                }
            }

            /// The retained operation behind this route.
            pub(super) fn operation(&self) -> &dyn SealedOperation {
                match self {
                    $(Self::$variant(operation) => operation,)+
                }
            }

            /// Mutable access to the retained operation.
            pub(super) fn operation_mut(&mut self) -> Option<&mut dyn SealedOperation> {
                match self {
                    $(Self::$variant(operation) => Some(operation),)+
                }
            }
        }
    };
}

sealed_executions! {
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
    SequentialContrast(crate::analysis::execute::CheckedSequentialContrastOperation),
    HistoryMatch(crate::analysis::execute::CheckedHistoryMatchOperation),
    TemporalDesign(Box<crate::analysis::execute::CheckedTemporalDesignOperation>),
    IvBounds(crate::analysis::execute::CheckedIvBoundsOperation),
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
    EstimatorRoute(crate::analysis::execute::CheckedRouteOperation),
    FreshRun(crate::analysis::execute::CheckedRouteOperation),
    Discovery(crate::analysis::execute::CheckedDiscoveryOperation),
    PanelTemporal(crate::analysis::execute::CheckedRouteOperation),
    MultiEnvTemporal(crate::analysis::execute::CheckedRouteOperation),
    TemporalCrossWorld(crate::analysis::execute::CheckedRouteOperation),
    TemporalAdmgEffect(crate::analysis::execute::CheckedRouteOperation),
    ClassAttribution(crate::analysis::execute::CheckedRouteOperation),
    Association(crate::analysis::execute::CheckedAssociationOperation),
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

impl PreparedExecution {
    /// Whether the shared route executors run this execution from the frozen study,
    /// inside the dispatch scope of the plan's lowered program: the estimator-family
    /// and route-cell operations that retain only that program, and a fresh run.
    pub(super) const fn runs_shared_dispatch(&self) -> bool {
        matches!(
            self,
            Self::EstimatorRoute(_)
                | Self::FreshRun(_)
                | Self::Discovery(_)
                | Self::PanelTemporal(_)
                | Self::MultiEnvTemporal(_)
                | Self::TemporalCrossWorld(_)
                | Self::TemporalAdmgEffect(_)
                | Self::ClassAttribution(_)
                | Self::Association(_)
        )
    }
}
