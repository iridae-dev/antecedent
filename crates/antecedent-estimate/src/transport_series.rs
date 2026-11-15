//! Dependence-aware uncertainty of the trial-to-target IPW contrast on one serial table.
//!
//! The rows are the consecutive times of one series, so the per-row influence of the contrast
//! is serially dependent (trial and target rows may interleave, and the treatment arm may be
//! held over runs of rows). Three pieces turn the independent-row constructions into series
//! ones:
//!
//! * [`trial_to_target_series_block_length`]: the block length of the block bootstrap, sized
//!   on the influence itself;
//! * [`trial_to_target_ipw_ewc_se`]: the equal-weighted cosine long-run variance of the
//!   influence, whose studentized error is Student-t at an exact degrees of freedom;
//! * [`scale_series_draws_to_ewc`]: the block-weight draws brought to that same spread.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ipw_influence, validate_trial_to_target_inputs};
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
/// Uncertified inputs, as [`crate::trial_to_target_ipw_se`].
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

/// Cosine terms of the equal-weighted cosine variance over `rows` rows: `floor(0.4 rows^(2/3))`,
/// the rule of Lazarus, Lewis, Stock and Watson (2018) that minimizes the size distortion of a
/// 5% test, at least two and below the row count.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "0.4 n^(2/3) of a series length is a non-negative count far below usize::MAX"
)]
pub fn series_ewc_terms(rows: usize) -> usize {
    let terms = (0.4 * (rows as f64).powf(2.0 / 3.0)).floor() as usize;
    terms.clamp(2, rows.saturating_sub(1).max(2))
}

/// Equal-weighted cosine standard error of the contrast from its per-row `influence`.
///
/// With `Lambda_j = sqrt(2/n) sum_t x_t cos(pi j (t + 1/2) / n)` the long-run variance is
/// estimated by the mean of `Lambda_j^2` over `B` terms; the projections on distinct cosines are
/// asymptotically independent normals, so `B * variance / truth` is chi-square on `B` degrees of
/// freedom and the studentized contrast is exactly Student-t on `B`, whatever the bandwidth
/// would have been. No lag is chosen, and the finite-sample reference is exact rather than a
/// fixed-b approximation. The standard error is `sqrt(n * variance) / target_n`.
fn ewc_standard_error(influence: &[f64], target_n: usize) -> (f64, usize) {
    let n = influence.len();
    let terms = series_ewc_terms(n);
    let nf = n as f64;
    let mut sum_sq = 0.0;
    for j in 1..=terms {
        let omega = std::f64::consts::PI * j as f64 / nf;
        let projection: f64 = influence
            .iter()
            .enumerate()
            .map(|(t, value)| value * (omega * (t as f64 + 0.5)).cos())
            .sum();
        sum_sq += 2.0 / nf * projection * projection;
    }
    let variance = sum_sq / terms as f64;
    ((nf * variance).sqrt() / target_n as f64, terms)
}

/// Standard error of the trial-to-target IPW contrast on one series from the equal-weighted
/// cosine long-run variance of its influence, with the Student-t degrees of freedom (the number
/// of cosine terms) its interval is referred to.
///
/// # Errors
///
/// As [`trial_to_target_ipw_se`](crate::trial_to_target_ipw_se).
pub fn trial_to_target_ipw_ewc_se(
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    ipw: f64,
) -> Result<(f64, usize), EstimationError> {
    let target_n = validate_trial_to_target_inputs(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
    )?;
    if !ipw.is_finite() {
        return Err(EstimationError::data_msg("transport IPW estimate must be finite"));
    }
    let influence =
        ipw_influence(outcome, treatment, trial, selection_probability, treatment_probability, ipw);
    Ok(ewc_standard_error(&influence, target_n))
}

/// Bring block-weight `draws` of the contrast to the 95% interval of the equal-weighted cosine
/// construction: deviations from the draw mean are scaled so that their 97.5% normal quantile is
/// `t_{B,0.975} * se`, with `se` and `B` those of [`trial_to_target_ipw_ewc_se`] at the
/// contrast's own point estimate. The block draws keep their shape and the posterior spread is the
/// calibrated one; the ratio is the 95% one and serves every level, as the circular-block
/// intervals' fixed-b ratio does.
///
/// # Errors
///
/// Uncertified inputs, fewer than two draws, or draws with no spread.
pub fn scale_series_draws_to_ewc(
    draws: &mut [f64],
    (outcome, treatment, trial): (&[f64], &[bool], &[bool]),
    (selection_probability, treatment_probability): (&[f64], &[f64]),
) -> Result<(), EstimationError> {
    let influence = influence_at_point(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
    )?;
    let target_n = trial.iter().filter(|&&source| !source).count();
    let (se, terms) = ewc_standard_error(&influence, target_n);
    let count = draws.len();
    if count < 2 {
        return Err(EstimationError::data_msg("block-weight draws need at least two draws"));
    }
    let centre = draws.iter().sum::<f64>() / count as f64;
    let spread =
        (draws.iter().map(|draw| (draw - centre).powi(2)).sum::<f64>() / (count - 1) as f64).sqrt();
    if !(spread.is_finite() && spread > 0.0 && se.is_finite()) {
        return Err(EstimationError::data_msg("block-weight draws have no spread to calibrate"));
    }
    let z = antecedent_stats::normal_ppf(0.975);
    let scale = se * antecedent_stats::student_t_ppf(0.975, terms as f64) / z / spread;
    for draw in draws {
        *draw = centre + (*draw - centre) * scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn terms_follow_the_two_thirds_rule() {
        assert_eq!(series_ewc_terms(500), 25);
        assert_eq!(series_ewc_terms(1000), 40);
        assert_eq!(series_ewc_terms(2000), 63);
        assert_eq!(series_ewc_terms(4), 2);
    }

    #[test]
    fn ewc_error_is_positive_and_draws_take_its_spread() {
        let rows = 500;
        let (y, a, trial) = alternating_series(rows, 10);
        let half = vec![0.5; rows];
        let block = trial_to_target_series_block_length(&y, &a, &trial, &half, &half).unwrap();
        assert!(block >= 8);
        let target_n = rows / 2;
        let ipw = y.iter().zip(&a).filter(|(_, arm)| **arm).map(|(v, _)| 2.0 * v).sum::<f64>()
            / target_n as f64;
        let (se, terms) = trial_to_target_ipw_ewc_se(&y, &a, &trial, &half, &half, ipw).unwrap();
        assert!(se > 0.0 && terms == 25);
        let mut draws: Vec<f64> = (0..200).map(|i| f64::from(i) * 0.001).collect();
        scale_series_draws_to_ewc(&mut draws, (&y, &a, &trial), (&half, &half)).unwrap();
        let centre = draws.iter().sum::<f64>() / 200.0;
        let sd = (draws.iter().map(|d| (d - centre).powi(2)).sum::<f64>() / 199.0).sqrt();
        let z = antecedent_stats::normal_ppf(0.975);
        let expected = se * antecedent_stats::student_t_ppf(0.975, 25.0) / z;
        assert!((sd - expected).abs() < 1e-9 * expected.max(1.0));
    }
}
