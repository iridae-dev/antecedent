//! Sealer of the estimator families that execute from the shared route executors.
//!
//! A study the route-cell sealers decline (a double-machine-learning or doubly
//! robust learner, matching, a causal forest, the temporal and design estimators
//! whose configuration a cell sealer does not admit, and the rest) is sealed here to
//! the lowered program of its plan, by the estimator it resolves to. The table lists
//! the estimators no specialised sealer serves and those whose specialised sealer
//! declines some shapes (marked in the list), and equals
//! `parity/estimator_route_residual.toml`; the retained-route gate checks both that and
//! that every other `EstimatorId` is served by a specialised sealer, so an estimator
//! added without a sealer cannot prepare.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::analysis::execute::CheckedRouteOperation;
use crate::analysis::prepared::PreparedExecution;
use crate::strategy_table::EstimatorId;

/// Every estimator this sealer admits: the residual no specialised sealer serves, and the
/// estimators a specialised sealer admits only for some shapes.
const FAMILY_ESTIMATORS: &[EstimatorId] = &[
    EstimatorId::PropensityStratification,
    EstimatorId::DistanceMatching,
    EstimatorId::FrontDoorFunctional,
    EstimatorId::BayesianIvJointLinear,
    EstimatorId::BayesianRdLocalLinear,
    EstimatorId::TemporalLinearAdjustment,
    EstimatorId::BayesianTemporalGcomp,
    EstimatorId::BayesianTemporalWhiteInnovation,
    EstimatorId::TemporalGlmLinkContrast,
    EstimatorId::BayesianTemporalGlmLinkContrast,
    EstimatorId::TemporalLaggedInstrument,
    EstimatorId::BayesianTemporalLaggedInstrument,
    EstimatorId::TemporalSequentialGcomp,
    EstimatorId::FunctionalDistribution,
    EstimatorId::StaticMediationLinear,
    EstimatorId::TemporalMediation,
    EstimatorId::TemporalResponseBayesian,
    EstimatorId::BayesianTemporalMediation,
    EstimatorId::ResponseRieszAde,
    EstimatorId::ResponseGamDerivative,
    EstimatorId::TemporalResponseGcomp,
    EstimatorId::Dml,
    EstimatorId::DrLearner,
    EstimatorId::CausalForest,
    EstimatorId::IvBoundsSharp,
    EstimatorId::BayesianIvBounds,
    EstimatorId::Aipw,
    EstimatorId::CellAipw,
    EstimatorId::ConditionalLinearAdjustment,
    EstimatorId::LinearAdjustmentAte,
    EstimatorId::RdSharp,
    EstimatorId::BayesianConditional,
    EstimatorId::BayesianGcomp,
    EstimatorId::ResponseKennedyDr,
    EstimatorId::BayesianBasisGcomp,
    EstimatorId::BayesianRobustAte,
    EstimatorId::FrontDoorTwoStage,
    EstimatorId::FunctionalEffect,
    EstimatorId::GlmAdjustment,
    EstimatorId::Iv2Sls,
    EstimatorId::IvWald,
    EstimatorId::PropensityMatching,
    EstimatorId::PropensityWeighting,
    EstimatorId::ResponseBayesian,
    EstimatorId::ResponseInterventionGcomp,
    EstimatorId::GcmFit,
    EstimatorId::TransportTrialIpw,
    EstimatorId::TransportTrialBayesianBootstrap,
    EstimatorId::InterferenceHtHajek,
    EstimatorId::InterferenceClusterNeyman,
    EstimatorId::InterferenceSaturationExact,
    EstimatorId::InterferenceObservationalIpw,
    EstimatorId::InterferenceBayesianGaussian,
    EstimatorId::InterferenceClusterBayesianBootstrap,
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
    // A graph posterior no specialised sealer claims is answered atom by atom by the compiled
    // route of its cell, whichever estimator its atoms' explicit cells name.
    let admitted = match estimator {
        Some(id) => FAMILY_ESTIMATORS.contains(&id) || analysis.graph_posterior.is_some(),
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

#[cfg(test)]
mod tests {
    use super::FAMILY_ESTIMATORS;
    use std::collections::BTreeSet;

    /// The table equals the committed residual list, one row per estimator id.
    #[test]
    fn the_family_table_is_the_committed_residual() {
        let path =
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/estimator_route_residual.toml");
        let text = std::fs::read_to_string(path).expect("the residual list is committed");
        let listed: BTreeSet<String> = text
            .lines()
            .filter_map(|line| line.strip_prefix("id = \""))
            .filter_map(|rest| rest.strip_suffix('"'))
            .map(str::to_owned)
            .collect();
        let table: BTreeSet<String> =
            FAMILY_ESTIMATORS.iter().map(|id| format!("{id:?}")).collect();
        assert_eq!(table.len(), FAMILY_ESTIMATORS.len(), "an estimator is listed twice");
        assert_eq!(table, listed);
    }
}
