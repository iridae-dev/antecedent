//! Dependence-aware uncertainty of the trial-to-target IPW contrast on one serial table.
//!
//! The rows are the consecutive times of one series, so the per-row influence of the contrast
//! is serially dependent (trial and target rows may interleave, and the treatment arm may be
//! held over runs of rows). Three pieces turn the independent-row constructions into series
//! ones:
//!
//! * [`trial_to_target_series_block_length`]: the lag of the Newey-West error and the block
//!   length of the block bootstrap, sized on the influence itself;
//! * [`trial_to_target_ipw_hac_se_fixed_b`]: the Newey-West error carrying the fixed-b
//!   reference of its bandwidth;
//! * [`block_weight_reference_scale`]: the Student-t reference of block-weight draws.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ipw_influence, trial_to_target_ipw_hac_se, validate_trial_to_target_inputs};
use crate::EstimationError;

/// The per-row influence of the ratio-of-means IPW contrast at its own point estimate, with
/// the number of target rows.
fn influence_at_point(
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let target_n = validate_trial_to_target_inputs(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
    )?;
    // Target rows carry `-ipw`, so with `ipw = 0` the influence is the trial-row score alone
    // and the point estimate is its sum over the target count.
    let scores =
        ipw_influence(outcome, treatment, trial, selection_probability, treatment_probability, 0.0);
    let ipw = scores.iter().sum::<f64>() / target_n as f64;
    Ok(ipw_influence(outcome, treatment, trial, selection_probability, treatment_probability, ipw))
}

/// Block length (equivalently Newey-West lag) of the trial-to-target contrast on one series,
/// the dependence-aware length [`crate::dependence_block_length`] reads off the contrast's own
/// influence over all rows in row order.
///
/// A fixed `n^(1/3)` length ignores how long the influence stays correlated: when the
/// treatment is held over runs of rows, or the outcome noise is persistent, the influence of
/// neighbouring rows is correlated over a stretch the rule can be shorter than, and the
/// Bartlett long-run variance then misses part of the dependence and the interval
/// under-covers. Politis-White sizes the length on the influence and the `n^(1/6)` rescale
/// is the coverage-error rate under fixed-b critical values; weakly dependent influence
/// keeps the rule.
///
/// # Errors
///
/// As [`trial_to_target_ipw_hac_se`].
pub fn trial_to_target_series_block_length(
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
) -> Result<usize, EstimationError> {
    let influence = influence_at_point(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
    )?;
    Ok(crate::dependence_block_length(1, influence.len(), &[&influence]))
}

/// [`trial_to_target_ipw_hac_se`] scaled by the Kiefer-Vogelsang fixed-b ratio of its
/// bandwidth, so that `estimate +/- z * SE` is the fixed-b interval of the Bartlett
/// long-run variance at `lag` over the `n` rows.
///
/// The Newey-West variance at a bandwidth that is a fraction `b = lag / n` of the series
/// is not consistent: its studentized error is not standard normal but the fixed-b law,
/// whose critical values exceed the normal ones by the polynomial of `b`
/// ([`crate::circular_fixed_b_scale`] is the circular counterpart). Dividing the fixed-b
/// critical value by the normal one and folding it into the SE keeps the published
/// `z * SE` construction.
///
/// # Errors
///
/// As [`trial_to_target_ipw_hac_se`].
pub fn trial_to_target_ipw_hac_se_fixed_b(
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    (ipw, lag): (f64, usize),
) -> Result<f64, EstimationError> {
    let se = trial_to_target_ipw_hac_se(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
        (ipw, lag),
    )?;
    Ok(se * crate::temporal_block::fixed_b_scale(lag, outcome.len()))
}

/// Ratio that turns the spread of block-weight draws over `blocks` blocks into the 95%
/// Student-t interval of the block-mean estimate.
///
/// Draws that weight `B` consecutive blocks by `Dirichlet(1, ..., 1)` have variance
/// `(B - 1)/(B + 1)` of the batch-means variance of the estimate, whose studentized error is
/// Student-t on `B - 1` degrees of freedom, so deviations from the draw centre are scaled by
/// `t_{B-1} / z * sqrt((B + 1)/(B - 1))`. The ratio is the 95% one and serves every level
/// of the published interval, as the fixed-b ratio of the circular-block intervals does.
///
/// # Errors
///
/// Fewer than three blocks, where the Student-t reference has no variance.
pub fn block_weight_reference_scale(blocks: usize) -> Result<f64, EstimationError> {
    if blocks < 3 {
        return Err(EstimationError::data_msg(
            "block-weight draws over fewer than three blocks carry no interval; use a longer \
             series or a shorter block",
        ));
    }
    let b = blocks as f64;
    let z = antecedent_stats::normal_ppf(0.975);
    Ok(antecedent_stats::student_t_ppf(0.975, b - 1.0) / z * ((b + 1.0) / (b - 1.0)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_weight_scale_exceeds_one_and_shrinks_with_blocks() {
        let few = block_weight_reference_scale(5).unwrap();
        let many = block_weight_reference_scale(200).unwrap();
        assert!(few > many && many > 1.0 && few > 1.5);
        assert!(block_weight_reference_scale(2).is_err());
    }

    fn alternating_series(rows: usize, run: usize) -> (Vec<f64>, Vec<bool>, Vec<bool>) {
        let (mut y, mut a, mut trial) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..rows {
            let in_trial = t % 2 == 1;
            let arm = (t / run) % 2 == 0;
            trial.push(in_trial);
            a.push(in_trial && arm);
            y.push(if in_trial { f64::from(u8::from(arm)) + 0.01 * (t % 7) as f64 } else { 0.0 });
        }
        (y, a, trial)
    }

    #[test]
    fn series_length_covers_the_arm_runs_and_fixed_b_widens_the_error() {
        let rows = 500;
        let (y, a, trial) = alternating_series(rows, 10);
        let half = vec![0.5; rows];
        let block = trial_to_target_series_block_length(&y, &a, &trial, &half, &half).unwrap();
        assert!(block >= 8, "block {block} is below the n^(1/3) rule");
        let plain = trial_to_target_ipw_hac_se(&y, &a, &trial, &half, &half, (1.0, block)).unwrap();
        let scaled =
            trial_to_target_ipw_hac_se_fixed_b(&y, &a, &trial, &half, &half, (1.0, block)).unwrap();
        assert!(scaled > plain);
    }
}
