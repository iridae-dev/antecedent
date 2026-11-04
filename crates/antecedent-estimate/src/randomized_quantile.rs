//! Frequentist quantile treatment effect of a randomized two-arm trial.
//!
//! The target is the `tau`-quantile of each potential outcome law and their difference, the
//! same target the Bayesian randomized quantile answers. Under randomization (an empty
//! adjustment set) the arm's potential-outcome distribution function is the arm's empirical
//! distribution function, `F_a(c) = #{i in arm a: Y_i <= c} / n_a`, evaluated on a strictly
//! increasing threshold grid. The quantile is the piecewise-linear inversion of that grid
//! distribution function ([`crate::quantile::invert_cdf_quantile`], shared with the
//! cross-fitted cell), so every frequentist quantile route reads one inversion.
//!
//! Uncertainty is the delta method on the inversion. The row influence of the grid
//! distribution function is `phi_F(c)_i = (1{T_i = a} / p_a) (1{Y_i <= c} - F_a(c))` with
//! `p_a = n_a / n` (the ratio estimator's influence under independent assignment), the
//! influence of the quantile is `-phi_F(q) / f(q)` with the local finite-difference density
//! of the inversion, and the treated-minus-control influence is the row difference. The
//! standard error is the iid standard error of that influence and the interval is the 95%
//! normal interval around the difference. Inference conditions on the grid: grid selection
//! and interpolation bias are excluded, as in the cross-fitted cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::NORMAL_Q975;

use crate::error::EstimationError;
use crate::quantile::{empirical_threshold_grid, invert_cdf_quantile};

/// Fewest rows each arm must carry; the arm's distribution function is not estimated below it.
pub const MIN_RANDOMIZED_QUANTILE_ARM_ROWS: usize = 10;

/// Points of the data-driven threshold grid built from the pooled outcomes.
pub const RANDOMIZED_QUANTILE_GRID_POINTS: usize = 19;

/// Quantile treatment effect of a randomized two-arm trial with its influence-function interval.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedQuantileEffect {
    /// Quantile level.
    pub tau: f64,
    /// Treated minus control quantile.
    pub difference: f64,
    /// Control-arm quantile.
    pub control_quantile: f64,
    /// Treated-arm quantile.
    pub treated_quantile: f64,
    /// Local finite-difference densities at the control and treated quantiles.
    pub densities: [f64; 2],
    /// Centered row influence of the difference.
    pub influence: Vec<f64>,
    /// Standard error of the difference: the iid standard error of the influence.
    pub se: f64,
    /// 95% normal interval `difference -/+ NORMAL_Q975 * se`.
    pub interval_95: [f64; 2],
    /// Rows in the control arm.
    pub n_control: usize,
    /// Rows in the treated arm.
    pub n_treated: usize,
    /// The threshold grid the distribution functions were read on.
    pub thresholds: Vec<f64>,
}

/// Quantile treatment effect on a threshold grid the caller declares.
///
/// `thresholds` is strictly increasing with at least two points.
///
/// # Errors
///
/// Misaligned or non-finite rows, an arm with fewer than
/// [`MIN_RANDOMIZED_QUANTILE_ARM_ROWS`] rows, an invalid grid or level, a level outside the
/// grid distribution function of an arm, or a density too small to invert.
pub fn randomized_quantile_effect_on_grid(
    outcome: &[f64],
    treated: &[bool],
    thresholds: &[f64],
    tau: f64,
) -> Result<RandomizedQuantileEffect, EstimationError> {
    if outcome.len() != treated.len() {
        return Err(EstimationError::data_msg("outcome and treatment must align"));
    }
    if outcome.iter().any(|y| !y.is_finite()) {
        return Err(EstimationError::data_msg("the quantile outcome must be finite"));
    }
    let n = outcome.len();
    let n_treated = treated.iter().filter(|t| **t).count();
    let n_control = n - n_treated;
    if n_treated < MIN_RANDOMIZED_QUANTILE_ARM_ROWS || n_control < MIN_RANDOMIZED_QUANTILE_ARM_ROWS
    {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("positivity_violated"),
            "the randomized quantile needs at least ten rows in each arm",
        ));
    }
    let arm = |active: bool| -> Result<(f64, Vec<f64>, f64), EstimationError> {
        let size = (if active { n_treated } else { n_control }) as f64;
        let share = size / n as f64;
        let cdf: Vec<f64> = thresholds
            .iter()
            .map(|&c| {
                let count =
                    outcome.iter().zip(treated).filter(|(y, t)| **t == active && **y <= c).count();
                count as f64 / size
            })
            .collect();
        let phi: Vec<Vec<f64>> = thresholds
            .iter()
            .zip(&cdf)
            .map(|(&c, &f)| {
                outcome
                    .iter()
                    .zip(treated)
                    .map(|(y, t)| {
                        if *t == active { (f64::from(u8::from(*y <= c)) - f) / share } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        invert_cdf_quantile(thresholds, &cdf, &phi, tau)
    };
    let (control_quantile, control_influence, control_density) = arm(false)?;
    let (treated_quantile, treated_influence, treated_density) = arm(true)?;
    let influence: Vec<f64> =
        treated_influence.iter().zip(&control_influence).map(|(a, c)| a - c).collect();
    let se = antecedent_stats::iid_se(&influence);
    let difference = treated_quantile - control_quantile;
    if !difference.is_finite() || !se.is_finite() {
        return Err(EstimationError::data_msg("the randomized quantile effect is not finite"));
    }
    let radius = NORMAL_Q975 * se;
    Ok(RandomizedQuantileEffect {
        tau,
        difference,
        control_quantile,
        treated_quantile,
        densities: [control_density, treated_density],
        influence,
        se,
        interval_95: [difference - radius, difference + radius],
        n_control,
        n_treated,
        thresholds: thresholds.to_vec(),
    })
}

/// Quantile treatment effect on the empirical grid of the pooled outcomes, the grid the
/// cross-fitted quantile cell uses ([`RANDOMIZED_QUANTILE_GRID_POINTS`] points).
///
/// # Errors
///
/// As [`randomized_quantile_effect_on_grid`], or fewer than sixteen finite outcomes or a
/// collapsed grid.
pub fn randomized_quantile_effect(
    outcome: &[f64],
    treated: &[bool],
    tau: f64,
) -> Result<RandomizedQuantileEffect, EstimationError> {
    let grid = empirical_threshold_grid(outcome, RANDOMIZED_QUANTILE_GRID_POINTS)?;
    randomized_quantile_effect_on_grid(outcome, treated, &grid, tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_arm_is_refused() {
        let outcome: Vec<f64> = (0..24).map(f64::from).collect();
        let mut treated = vec![false; 24];
        for flag in treated.iter_mut().take(5) {
            *flag = true;
        }
        assert!(randomized_quantile_effect(&outcome, &treated, 0.5).is_err());
    }
}
