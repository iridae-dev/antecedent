//! Sealer of the estimator families that execute from the shared route executors.
//!
//! A study the route-cell sealers decline (a double-machine-learning or doubly
//! robust learner, matching, a causal forest, the temporal and design estimators
//! whose configuration a cell sealer does not admit, and the rest) is sealed here to
//! the lowered program of its plan, by the estimator it resolves to. The table lists
//! every `EstimatorId`; the retained-route gate fails when an id is missing, so an
//! estimator added without being assigned to a sealer cannot prepare.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::analysis::execute::CheckedRouteOperation;
use crate::analysis::prepared::PreparedExecution;
use crate::strategy_table::EstimatorId;

/// Every estimator this sealer admits.
const FAMILY_ESTIMATORS: &[EstimatorId] = &[
    EstimatorId::LinearAdjustmentAte,
    EstimatorId::PropensityWeighting,
    EstimatorId::PropensityMatching,
    EstimatorId::PropensityStratification,
    EstimatorId::DistanceMatching,
    EstimatorId::Aipw,
    EstimatorId::GlmAdjustment,
    EstimatorId::FrontDoorTwoStage,
    EstimatorId::FrontDoorFunctional,
    EstimatorId::IvWald,
    EstimatorId::Iv2Sls,
    EstimatorId::BayesianIvJointLinear,
    EstimatorId::RdSharp,
    EstimatorId::BayesianRdLocalLinear,
    EstimatorId::BayesianGcomp,
    EstimatorId::BayesianBasisGcomp,
    EstimatorId::BayesianRobustAte,
    EstimatorId::BayesianConditional,
    EstimatorId::TemporalLinearAdjustment,
    EstimatorId::BayesianTemporalGcomp,
    EstimatorId::BayesianTemporalWhiteInnovation,
    EstimatorId::TemporalGlmLinkContrast,
    EstimatorId::BayesianTemporalGlmLinkContrast,
    EstimatorId::TemporalLaggedInstrument,
    EstimatorId::BayesianTemporalLaggedInstrument,
    EstimatorId::TemporalSequentialGcomp,
    EstimatorId::FunctionalDistribution,
    EstimatorId::FunctionalEffect,
    EstimatorId::StaticMediationLinear,
    EstimatorId::ConditionalLinearAdjustment,
    EstimatorId::TemporalMediation,
    EstimatorId::TemporalResponseBayesian,
    EstimatorId::ResponseBayesian,
    EstimatorId::BayesianTemporalMediation,
    EstimatorId::ResponseKennedyDr,
    EstimatorId::ResponseRieszAde,
    EstimatorId::ResponseGamDerivative,
    EstimatorId::ResponseInterventionGcomp,
    EstimatorId::CellAipw,
    EstimatorId::TemporalResponseGcomp,
    EstimatorId::GcmFit,
    EstimatorId::GcmFitBayesian,
    EstimatorId::GcmAttributionBayesian,
    EstimatorId::TransportTrialIpw,
    EstimatorId::TransportTrialBayesianBootstrap,
    EstimatorId::InterferenceHtHajek,
    EstimatorId::InterferenceClusterNeyman,
    EstimatorId::InterferenceSaturationExact,
    EstimatorId::InterferenceObservationalIpw,
    EstimatorId::InterferenceBayesianGaussian,
    EstimatorId::InterferenceClusterBayesianBootstrap,
    EstimatorId::RandomizedHt,
    EstimatorId::ContinuousDoseKernel,
    EstimatorId::RandomizedFixedCupedHt,
    EstimatorId::RandomizedAncova,
    EstimatorId::RandomizedWaldCace,
    EstimatorId::RandomizedSwitchbackHt,
    EstimatorId::RandomizedNeyman,
    EstimatorId::RandomizedDrPolicy,
    EstimatorId::RandomizedIpwPolicy,
    EstimatorId::RandomizedMultiActionIpwPolicy,
    EstimatorId::RandomizedSurvivalProductLimit,
    EstimatorId::LongitudinalIpwRegime,
    EstimatorId::LongitudinalGFormulaRegime,
    EstimatorId::LongitudinalSequentialDrRegime,
    EstimatorId::LongitudinalMarginalStructuralModel,
    EstimatorId::Dml,
    EstimatorId::DrLearner,
    EstimatorId::CausalForest,
    EstimatorId::BayesianPanelDid,
    EstimatorId::BayesianSyntheticControl,
    EstimatorId::BayesianLocalRatio,
    EstimatorId::BayesianPiecewiseSurvival,
    EstimatorId::BayesianRandomizedQuantile,
    EstimatorId::BayesianDosePolicy,
    EstimatorId::BayesianRandomizedFinitePopulation,
    EstimatorId::BayesianRandomizedSuperpopulation,
    EstimatorId::BayesianComplierEffect,
    EstimatorId::BayesianSwitchback,
    EstimatorId::BayesianBinaryPolicyValue,
    EstimatorId::BayesianSequentialRegimeValue,
    EstimatorId::SequentialCrossfitContrastWald,
    EstimatorId::BayesianSequentialCrossfitContrast,
    EstimatorId::PeriodHistoryMatching,
    EstimatorId::BayesianPeriodHistoryOutcomeModel,
    EstimatorId::TemporalDesignExcursion,
    EstimatorId::BayesianTemporalDesignExcursion,
    EstimatorId::TemporalRegimeSurvival,
    EstimatorId::BayesianTemporalRegimeSurvival,
    EstimatorId::IvBoundsSharp,
    EstimatorId::BayesianIvBounds,
];

/// Seal a study no route-cell sealer admits, by the estimator it resolves to; a study naming none runs its route's
/// default estimator, which a compiled cell always has.
pub(super) fn estimator_route(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, binding, .. } = *cx;
    // Only a study no route-cell sealer admits: the table's last two entries are the
    // program-only sealers, this one and the fresh run.
    let specialised = &super::SEALERS[..super::SEALERS.len() - 2];
    for sealer in specialised {
        if (sealer.seal)(cx)?.is_some() {
            return Ok(None);
        }
    }
    let estimator = source
        .estimator
        .or_else(|| source.estimator_spec.as_deref().map(crate::EstimatorSpec::id))
        .or_else(|| binding.estimator.named());
    let admitted = match estimator {
        Some(id) => FAMILY_ESTIMATORS.contains(&id),
        None => cx.route.is_some() || analysis.graph_posterior.is_some(),
    };
    if !admitted {
        return Ok(None);
    }
    CheckedRouteOperation::lowered(analysis, plan)
        .map(|operation| Some(PreparedExecution::EstimatorRoute(operation)))
}

/// A fresh run (`Study::execute`, one-shot `run`) holds no prepared handle, so no
/// preparation seals it; its operation is built where the run dispatches.
#[allow(
    clippy::unnecessary_wraps,
    reason = "every sealer shares the table's fallible `SealOutcome` signature"
)]
pub(super) fn fresh_run(_cx: &SealContext<'_>) -> SealOutcome {
    Ok(None)
}
