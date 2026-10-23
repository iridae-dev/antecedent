//! Wire forms of the rollout fields of the difference-in-differences, synthetic-control and
//! local-ratio queries: the heterogeneity-robust specification and whole-procedure resample
//! of a rollout, the adoptions and interval construction of a synthetic panel, and the design
//! options of a local ratio.
//!
//! Every field is optional on its parent wire and absent when unset, so artifacts written
//! before these fields existed read unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DidAggregation, DidBasePeriod, DidComparisonGroup, DidNuisanceMethod, DidResampling,
    DidResamplingScheme, DidSamplingDesign, HeterogeneityRobustDid, LocalBandwidthRule,
    LocalRatioOptions, RolloutDidEstimator, SyntheticInterval, SyntheticRollout, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::error::IoError;

/// Heterogeneity-robust rollout specification.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RolloutDidWire {
    /// `staggered_heterogeneity_robust` or `covariate_adjusted_two_period`.
    pub design: String,
    /// `callaway_santanna`, `sun_abraham` or `imputation`.
    pub estimator: String,
    /// `never_treated` or `not_yet_treated`.
    pub comparison: String,
    /// `outcome_regression`, `inverse_probability_weighting` or `doubly_robust`.
    pub nuisance: String,
    /// `varying` or `universal`.
    pub base_period: String,
    /// `simple`, `dynamic`, `group` or `calendar`.
    pub aggregation: String,
    /// Anticipation periods.
    pub anticipation: u32,
    /// Baseline covariate column ids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub covariates: Vec<u32>,
}

/// Whole-procedure resample of a rollout or augmented panel.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DidResamplingWire {
    /// `cluster_bootstrap` or `cluster_multiplier`.
    pub scheme: String,
    /// Replicates.
    pub replicates: u32,
}

/// Adoptions and interval construction of a synthetic panel with intervals.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SyntheticRolloutWire {
    /// Adopting units with their first treated periods.
    pub adoptions: Vec<(String, i64)>,
    /// `conformal` or `placebo_variance`.
    pub interval: String,
    /// Placebo replicates.
    pub replicates: u32,
}

/// Design options of a local ratio.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LocalRatioOptionsWire {
    /// `declared` or `mse_optimal`.
    pub bandwidth_rule: String,
    /// Block-bootstrap replicates of a calendar-time running variable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_replicates: Option<u32>,
    /// Whether the density test runs.
    #[serde(default)]
    pub density_test: bool,
    /// Balance covariate column ids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub balance_covariates: Vec<u32>,
}

fn unknown(field: &str, value: &str) -> IoError {
    IoError::Convert(format!("unknown {field} `{value}`"))
}

fn ids(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().map(|id| VariableId::from_raw(*id)).collect()
}

/// The wire form of a rollout specification of `design`.
#[must_use]
pub fn rollout_to_wire(
    design: DidSamplingDesign,
    robust: &HeterogeneityRobustDid,
) -> RolloutDidWire {
    RolloutDidWire {
        design: if design == DidSamplingDesign::CovariateAdjustedTwoPeriod {
            "covariate_adjusted_two_period"
        } else {
            "staggered_heterogeneity_robust"
        }
        .into(),
        estimator: robust.estimator.as_str().into(),
        comparison: robust.comparison.as_str().into(),
        nuisance: robust.nuisance.as_str().into(),
        base_period: robust.base_period.as_str().into(),
        aggregation: robust.aggregation.as_str().into(),
        anticipation: robust.anticipation,
        covariates: robust.covariates.iter().map(|id| id.raw()).collect(),
    }
}

/// The design and specification of a rollout wire.
///
/// # Errors
///
/// An unknown name.
pub fn rollout_from_wire(
    wire: &RolloutDidWire,
) -> Result<(DidSamplingDesign, HeterogeneityRobustDid), IoError> {
    let design = match wire.design.as_str() {
        "staggered_heterogeneity_robust" => DidSamplingDesign::StaggeredHeterogeneityRobust,
        "covariate_adjusted_two_period" => DidSamplingDesign::CovariateAdjustedTwoPeriod,
        other => return Err(unknown("rollout design", other)),
    };
    let estimator = match wire.estimator.as_str() {
        "callaway_santanna" => RolloutDidEstimator::CallawaySantAnna,
        "sun_abraham" => RolloutDidEstimator::SunAbraham,
        "imputation" => RolloutDidEstimator::Imputation,
        other => return Err(unknown("rollout estimator", other)),
    };
    let comparison = match wire.comparison.as_str() {
        "never_treated" => DidComparisonGroup::NeverTreated,
        "not_yet_treated" => DidComparisonGroup::NotYetTreated,
        other => return Err(unknown("comparison group", other)),
    };
    let nuisance = match wire.nuisance.as_str() {
        "outcome_regression" => DidNuisanceMethod::OutcomeRegression,
        "inverse_probability_weighting" => DidNuisanceMethod::InverseProbabilityWeighting,
        "doubly_robust" => DidNuisanceMethod::DoublyRobust,
        other => return Err(unknown("nuisance method", other)),
    };
    let base_period = match wire.base_period.as_str() {
        "varying" => DidBasePeriod::Varying,
        "universal" => DidBasePeriod::Universal,
        other => return Err(unknown("base period", other)),
    };
    let aggregation = match wire.aggregation.as_str() {
        "simple" => DidAggregation::Simple,
        "dynamic" => DidAggregation::Dynamic,
        "group" => DidAggregation::Group,
        "calendar" => DidAggregation::Calendar,
        other => return Err(unknown("aggregation", other)),
    };
    Ok((
        design,
        HeterogeneityRobustDid {
            estimator,
            comparison,
            nuisance,
            base_period,
            aggregation,
            anticipation: wire.anticipation,
            covariates: ids(&wire.covariates),
        },
    ))
}

/// The wire form of a whole-procedure resample.
#[must_use]
pub fn resampling_to_wire(resampling: &DidResampling) -> DidResamplingWire {
    DidResamplingWire {
        scheme: match resampling.scheme {
            DidResamplingScheme::ClusterBootstrap => "cluster_bootstrap",
            DidResamplingScheme::ClusterMultiplier => "cluster_multiplier",
        }
        .into(),
        replicates: resampling.replicates,
    }
}

/// The resample of a wire.
///
/// # Errors
///
/// An unknown scheme.
pub fn resampling_from_wire(wire: &DidResamplingWire) -> Result<DidResampling, IoError> {
    let scheme = match wire.scheme.as_str() {
        "cluster_bootstrap" => DidResamplingScheme::ClusterBootstrap,
        "cluster_multiplier" => DidResamplingScheme::ClusterMultiplier,
        other => return Err(unknown("resampling scheme", other)),
    };
    Ok(DidResampling { scheme, replicates: wire.replicates })
}

/// The wire form of a synthetic rollout.
#[must_use]
pub fn synthetic_rollout_to_wire(rollout: &SyntheticRollout) -> SyntheticRolloutWire {
    SyntheticRolloutWire {
        adoptions: rollout
            .adoptions
            .iter()
            .map(|(unit, period)| (unit.to_string(), *period))
            .collect(),
        interval: rollout.interval.as_str().into(),
        replicates: rollout.replicates,
    }
}

/// The synthetic rollout of a wire.
///
/// # Errors
///
/// An unknown interval construction.
pub fn synthetic_rollout_from_wire(
    wire: &SyntheticRolloutWire,
) -> Result<SyntheticRollout, IoError> {
    let interval = match wire.interval.as_str() {
        "conformal" => SyntheticInterval::Conformal,
        "placebo_variance" => SyntheticInterval::PlaceboVariance,
        other => return Err(unknown("synthetic interval", other)),
    };
    Ok(SyntheticRollout {
        adoptions: wire
            .adoptions
            .iter()
            .map(|(unit, period)| (Arc::<str>::from(unit.as_str()), *period))
            .collect(),
        interval,
        replicates: wire.replicates,
    })
}

/// The wire form of local-ratio options.
#[must_use]
pub fn local_ratio_options_to_wire(options: &LocalRatioOptions) -> LocalRatioOptionsWire {
    LocalRatioOptionsWire {
        bandwidth_rule: match options.bandwidth_rule {
            LocalBandwidthRule::Declared => "declared",
            LocalBandwidthRule::MseOptimal => "mse_optimal",
        }
        .into(),
        temporal_replicates: options.temporal_replicates,
        density_test: options.density_test,
        balance_covariates: options.balance_covariates.iter().map(|id| id.raw()).collect(),
    }
}

/// The local-ratio options of a wire.
///
/// # Errors
///
/// An unknown bandwidth rule.
pub fn local_ratio_options_from_wire(
    wire: &LocalRatioOptionsWire,
) -> Result<LocalRatioOptions, IoError> {
    let bandwidth_rule = match wire.bandwidth_rule.as_str() {
        "declared" => LocalBandwidthRule::Declared,
        "mse_optimal" => LocalBandwidthRule::MseOptimal,
        other => return Err(unknown("bandwidth rule", other)),
    };
    Ok(LocalRatioOptions {
        bandwidth_rule,
        temporal_replicates: wire.temporal_replicates,
        density_test: wire.density_test,
        balance_covariates: ids(&wire.balance_covariates),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rollout_field_round_trips_through_its_wire() {
        let robust = HeterogeneityRobustDid {
            estimator: RolloutDidEstimator::SunAbraham,
            comparison: DidComparisonGroup::NeverTreated,
            nuisance: DidNuisanceMethod::InverseProbabilityWeighting,
            base_period: DidBasePeriod::Universal,
            aggregation: DidAggregation::Calendar,
            anticipation: 2,
            covariates: Arc::from([VariableId::from_raw(4)]),
        };
        let wire = rollout_to_wire(DidSamplingDesign::StaggeredHeterogeneityRobust, &robust);
        assert_eq!(
            rollout_from_wire(&wire).unwrap(),
            (DidSamplingDesign::StaggeredHeterogeneityRobust, robust)
        );
        let resampling =
            DidResampling { scheme: DidResamplingScheme::ClusterMultiplier, replicates: 99 };
        assert_eq!(resampling_from_wire(&resampling_to_wire(&resampling)).unwrap(), resampling);
        let rollout = SyntheticRollout {
            adoptions: Arc::from([(Arc::<str>::from("a"), 5_i64)]),
            interval: SyntheticInterval::PlaceboVariance,
            replicates: 50,
        };
        assert_eq!(
            synthetic_rollout_from_wire(&synthetic_rollout_to_wire(&rollout)).unwrap(),
            rollout
        );
        let options = LocalRatioOptions {
            bandwidth_rule: LocalBandwidthRule::MseOptimal,
            temporal_replicates: Some(99),
            density_test: true,
            balance_covariates: Arc::from([VariableId::from_raw(3)]),
        };
        assert_eq!(
            local_ratio_options_from_wire(&local_ratio_options_to_wire(&options)).unwrap(),
            options
        );
        let mut bad = wire;
        bad.estimator = "unknown".into();
        assert!(rollout_from_wire(&bad).is_err());
    }
}
