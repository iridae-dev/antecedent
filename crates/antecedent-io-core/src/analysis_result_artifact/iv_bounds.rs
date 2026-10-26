//! Wire section of a binary-instrument bounds result.
//!
//! The section is optional on the result body and absent on every other result, so their bodies
//! and digests are unchanged. An `IvBounds` query requires it. The result is an identified set,
//! never a point: the body carries no scalar estimate, standard error or interval beside it. A
//! loader recomputes the Balke-Pearl bounds from the retained observed law and checks that they
//! are the reported ones, that the law is a law (each instrument arm sums to one), that the
//! sampling interval, when present, is ordered and covers the identified set when it is a
//! frequentist Imbens-Manski interval, and that the result claims no licence and an unmeasured
//! calibration.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use super::AnalysisResultWire;
use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};
use crate::{CausalQueryWire, IoError};

/// Replay policy of the published bounds against the bounds recomputed from the observed law:
/// a relative `1e-9` of the recomputed value, absolute below one; non-finite is refused.
const BOUNDS_REPLAY: ReplayTolerance =
    ReplayTolerance::new(1e-9, ReplayScale::OnePlusReplayed, ReplayNonFinite::Refuse);

/// Uncertainty label of a result that reports the identified set without a sampling interval.
pub const IV_BOUNDS_POINT_UNCERTAINTY: &str = "identified_set_without_sampling_interval";
/// Frequentist interval construction.
pub const IV_BOUNDS_FREQUENTIST_METHOD: &str = "imbens_manski_row_bootstrap";
/// Bayesian interval construction.
pub const IV_BOUNDS_BAYESIAN_METHOD: &str = "posterior_endpoint_quantiles";

/// Sampling interval about the identified set.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IvBoundsIntervalWire {
    /// Nominal level.
    pub level: f64,
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Sampling standard deviation of the lower bound.
    pub lower_se: f64,
    /// Sampling standard deviation of the upper bound.
    pub upper_se: f64,
    /// Imbens-Manski critical value.
    pub critical_value: f64,
    /// Whether the estimated width passed the moment-selection threshold.
    pub width_retained: bool,
    /// Resamples or draws behind the interval.
    pub replicates: u64,
    /// Resamples or draws left out because their law violates the instrumental inequality.
    pub infeasible: u64,
    /// Construction name.
    pub method: String,
}

/// Posterior summary of the endpoints under the Bayesian bootstrap.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IvBoundsPosteriorWire {
    /// Draws requested.
    pub n_draws: u64,
    /// Draws whose law violates the instrumental inequality.
    pub infeasible_draws: u64,
    /// Posterior mean of the lower bound.
    pub lower_mean: f64,
    /// Posterior mean of the upper bound.
    pub upper_mean: f64,
    /// Posterior standard deviation of the lower bound.
    pub lower_sd: f64,
    /// Posterior standard deviation of the upper bound.
    pub upper_sd: f64,
}

/// Sharp bounds on the average effect under a binary instrument (unlicensed, unmeasured).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IvBoundsWire {
    /// Lower bound of the average effect.
    pub lower: f64,
    /// Upper bound of the average effect.
    pub upper: f64,
    /// Observed law `P(Y=y, D=d | Z=z)`, instrument arm major, cell order `(0,0) (1,0) (0,1) (1,1)`.
    pub cells: Vec<f64>,
    /// Weighted row mass in each instrument arm.
    pub arm_weights: Vec<f64>,
    /// Complete rows behind the law.
    pub rows: u64,
    /// Rows left out for a missing cell or weight.
    pub rows_dropped: u64,
    /// Sampling interval about the identified set, when one was requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<IvBoundsIntervalWire>,
    /// Posterior summary, under Bayesian inference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior: Option<IvBoundsPosteriorWire>,
    /// Uncertainty construction.
    pub uncertainty: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Always `false`.
    pub licensed: bool,
}

/// Check a bounds section against its query.
pub(in crate::analysis_result_artifact) fn validate_iv_bounds(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let fail = |message: &str| Err(IoError::Convert(format!("iv bounds section: {message}")));
    let is_query = matches!(result.query, CausalQueryWire::IvBounds(_));
    let section = match (is_query, &result.iv_bounds) {
        (false, None) => return Ok(()),
        (true, None) => return fail("a bounds artifact is missing its section"),
        (false, Some(_)) => return fail("the section is attached to another query"),
        (true, Some(section)) => section,
    };
    if section.licensed
        || section.calibration != "unmeasured"
        || result.estimate.is_some()
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail(
            "a bounds result is an identified set: it claims no licence, no calibration, and no \
             scalar estimate or interval",
        );
    }
    if section.cells.len() != 8 || section.arm_weights.len() != 2 {
        return fail("the observed law is not two arms of four cells");
    }
    if section.arm_weights.iter().any(|w| !w.is_finite() || *w <= 0.0) || section.rows == 0 {
        return fail("an instrument arm has no mass");
    }
    let near = |a: f64, b: f64| BOUNDS_REPLAY.close(a, b);
    let cells = &section.cells;
    let law = antecedent_identify::BinaryIvLaw {
        cells: [[cells[0], cells[1], cells[2], cells[3]], [cells[4], cells[5], cells[6], cells[7]]],
    };
    let Ok(bounds) = antecedent_identify::binary_iv_ate_bounds(law) else {
        return fail("the retained law is not a law that satisfies the instrumental inequality");
    };
    if !near(section.lower, bounds.lower) || !near(section.upper, bounds.upper) {
        return fail("the reported bounds are not those of the retained observed law");
    }
    match (&section.interval, &section.posterior) {
        (None, None) => {
            if section.uncertainty != IV_BOUNDS_POINT_UNCERTAINTY {
                return fail("the uncertainty label names an interval the section does not hold");
            }
        }
        (Some(interval), posterior) => {
            let finite = [
                interval.level,
                interval.lower,
                interval.upper,
                interval.lower_se,
                interval.upper_se,
                interval.critical_value,
            ]
            .iter()
            .all(|v| v.is_finite());
            if !finite
                || !(0.0..1.0).contains(&interval.level)
                || interval.lower > interval.upper
                || interval.lower_se < 0.0
                || interval.upper_se < 0.0
            {
                return fail("the sampling interval is not finite and ordered");
            }
            if section.uncertainty != interval.method {
                return fail("the uncertainty label is not the interval's construction");
            }
            match (interval.method.as_str(), posterior) {
                (IV_BOUNDS_FREQUENTIST_METHOD, None) => {
                    let slack = 1e-9 * (1.0 + section.lower.abs() + section.upper.abs());
                    if interval.lower > section.lower + slack
                        || interval.upper < section.upper - slack
                    {
                        return fail("an Imbens-Manski interval covers the identified set");
                    }
                }
                (IV_BOUNDS_BAYESIAN_METHOD, Some(posterior)) => {
                    if ![
                        posterior.lower_mean,
                        posterior.upper_mean,
                        posterior.lower_sd,
                        posterior.upper_sd,
                    ]
                    .iter()
                    .all(|v| v.is_finite())
                        || posterior.lower_sd < 0.0
                        || posterior.upper_sd < 0.0
                        || posterior.infeasible_draws > posterior.n_draws
                    {
                        return fail("the posterior summary is not finite");
                    }
                }
                _ => return fail("the interval construction and the posterior do not agree"),
            }
        }
        (None, Some(_)) => return fail("a posterior summary comes with its interval"),
    }
    Ok(())
}
