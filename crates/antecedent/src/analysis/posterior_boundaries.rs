//! Typed boundaries of two posteriors that have no same-target counterpart.
//!
//! * A trajectory survival contrast under a prescribed time-varying regime is estimated from
//!   supplied treatment probabilities and censoring weights. Resampling subjects with those
//!   weights attached gives an interval conditional on them
//!   ([`trajectory_survival_bootstrap`]); a credible interval for the survival under the
//!   regime needs a joint likelihood for the confounders, treatment, censoring and hazard,
//!   which the weights do not supply, so [`trajectory_hazard_credible_interval`] always
//!   refuses with `cell_not_licensed` and names that likelihood and the next attempt.
//! * The temporal response-curve posterior is a tempered generalized posterior. An ordinary
//!   posterior needs the innovation law of the series in the likelihood;
//!   [`require_ordinary_temporal_curve_posterior`] refuses the ordinary reading with
//!   `cell_not_licensed` and `temporal_curve.ordinary_posterior_not_licensed` unless the
//!   caller attests independent innovations, and names the next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::temporal_curve_posterior_meaning::{
    SerialErrorModel, TemporalCurvePosteriorMeaning, temporal_curve_posterior_meaning,
};
pub use antecedent_estimate::trajectory_survival::{
    TrajectoryBootstrapIntervals, TrajectoryRows, TrajectorySurvivalContrast,
};

use crate::error::CausalError;
use antecedent_estimate::EstimationError;

fn data_error(message: &str) -> CausalError {
    CausalError::Estimate(EstimationError::data_msg(message.to_owned()))
}

/// Weighted product-limit survival and restricted-mean contrast of two regimes: a point.
///
/// # Errors
/// Invalid trajectories, identical regimes, an empty weighted risk set or a treatment
/// probability outside the declared positivity floor.
pub fn trajectory_survival_contrast(
    rows: &TrajectoryRows<'_>,
    active: &[bool],
    control: &[bool],
) -> Result<TrajectorySurvivalContrast, CausalError> {
    antecedent_estimate::trajectory_survival::trajectory_survival_contrast(rows, active, control)
        .map_err(data_error)
}

/// Percentile subject-bootstrap intervals of the weighted product-limit contrast,
/// conditional on the supplied treatment probabilities and censoring weights.
///
/// # Errors
/// Every error of [`trajectory_survival_contrast`], a replicate count outside
/// `199..=100000`, or more than a tenth of the replicates lost to empty risk sets.
pub fn trajectory_survival_bootstrap(
    rows: &TrajectoryRows<'_>,
    active: &[bool],
    control: &[bool],
    replicates: u32,
    seed: u64,
) -> Result<TrajectoryBootstrapIntervals, CausalError> {
    antecedent_estimate::trajectory_survival::trajectory_survival_bootstrap(
        rows, active, control, replicates, seed,
    )
    .map_err(data_error)
}

/// Credible interval of the trajectory survival contrast: always refused.
///
/// # Errors
/// Invalid trajectories as for [`trajectory_survival_contrast`]; otherwise always a
/// `cell_not_licensed` refusal (`trajectory_survival.credible_interval_not_licensed`) naming
/// the missing joint likelihood and the next attempt.
pub fn trajectory_hazard_credible_interval(
    rows: &TrajectoryRows<'_>,
    active: &[bool],
    control: &[bool],
) -> Result<[f64; 2], CausalError> {
    let message =
        match antecedent_estimate::trajectory_survival::trajectory_hazard_credible_interval(
            rows, active, control,
        ) {
            Ok(interval) => return Ok(interval),
            Err(message) => message,
        };
    match antecedent_estimate::trajectory_survival::trajectory_survival_contrast(
        rows, active, control,
    ) {
        Err(invalid) => Err(data_error(invalid)),
        Ok(_) => Err(CausalError::Estimate(EstimationError::refused_detail(
            antecedent_core::reason_code!("cell_not_licensed"),
            "trajectory_survival.credible_interval_not_licensed",
            message,
        ))),
    }
}

/// Require the ordinary-likelihood reading of a temporal response-curve posterior.
///
/// # Errors
/// A `cell_not_licensed` refusal with detail `temporal_curve.ordinary_posterior_not_licensed`
/// when the innovation dependence is [`SerialErrorModel::Unmodeled`]; the message names the
/// missing likelihood and the next attempt.
pub fn require_ordinary_temporal_curve_posterior(
    serial: SerialErrorModel,
) -> Result<(), CausalError> {
    antecedent_estimate::temporal_curve_posterior_meaning::require_ordinary_temporal_curve_posterior(
        serial,
    )
    .map_err(CausalError::from)
}
