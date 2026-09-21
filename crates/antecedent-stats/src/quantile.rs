//! Sample quantiles, equal-tail intervals and robust scale for finite samples.
//!
//! One owner for the two quantile conventions the estimators use: exchangeable-rank
//! (Hyndman–Fan type 6) for posterior draws and interpolated (type 7) for resampling
//! percentiles and descriptive summaries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

/// Consistency constant turning a median absolute deviation into a Gaussian σ.
pub const MAD_TO_SIGMA: f64 = 1.4826;

/// Rule that maps a probability to a fractional rank of the sorted sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantileRule {
    /// Hyndman–Fan type 6: the `p`-quantile sits at one-based rank `p·(D + 1)`.
    ///
    /// When a posterior is calibrated the truth is exchangeable with its `D` draws, so
    /// the interval between order statistics `r` and `s` covers it with probability
    /// exactly `(s − r)/(D + 1)`. Rank `p·(D + 1)` makes that probability the level,
    /// whereas the type-7 ranks sit about one order statistic inside on each side and
    /// under-cover at finite `D` (0.872 at `D = 64` for a 0.90 interval). Ranks outside
    /// `[1, D]` clamp to the extreme draws. Use for posterior and predictive draws.
    ExchangeableRank,
    /// Hyndman–Fan type 7: the `p`-quantile sits at zero-based rank `p·(D − 1)`.
    /// Use for resampling (bootstrap) percentiles and descriptive quantiles.
    Interpolated,
}

/// `p`-quantile of an ascending-sorted sample under `rule`; `NaN` for an empty sample.
/// `p` is clamped to `[0, 1]`.
#[must_use]
pub fn quantile_sorted(sorted: &[f64], p: f64, rule: QuantileRule) -> f64 {
    let d = sorted.len();
    if d == 0 {
        return f64::NAN;
    }
    let p = p.clamp(0.0, 1.0);
    let zero_based = match rule {
        QuantileRule::ExchangeableRank => (p * (d + 1) as f64).clamp(1.0, d as f64) - 1.0,
        QuantileRule::Interpolated => p * (d - 1) as f64,
    };
    let lo = zero_based.floor() as usize;
    let hi = zero_based.ceil() as usize;
    let frac = zero_based - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Equal-tailed interval `[q((1 − level)/2), q((1 + level)/2)]` of an ascending-sorted sample.
#[must_use]
pub fn equal_tail_interval_sorted(sorted: &[f64], level: f64, rule: QuantileRule) -> (f64, f64) {
    (
        quantile_sorted(sorted, (1.0 - level) / 2.0, rule),
        quantile_sorted(sorted, (1.0 + level) / 2.0, rule),
    )
}

/// Median of an ascending-sorted sample; `NaN` for an empty sample.
#[must_use]
pub fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    if n % 2 == 0 { 0.5 * (sorted[mid - 1] + sorted[mid]) } else { sorted[mid] }
}

/// Gaussian-consistent MAD scale `1.4826 · median(|x − median(x)|)`; `None` for an empty sample.
#[must_use]
pub fn mad_sigma(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let center = median_sorted(&sorted);
    let mut deviations: Vec<f64> = values.iter().map(|v| (v - center).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    Some(MAD_TO_SIGMA * median_sorted(&deviations))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type7_matches_the_textbook_interpolation() {
        let s = [1.0, 2.0, 4.0, 8.0, 16.0];
        assert_eq!(quantile_sorted(&s, 0.0, QuantileRule::Interpolated), 1.0);
        assert_eq!(quantile_sorted(&s, 1.0, QuantileRule::Interpolated), 16.0);
        assert_eq!(quantile_sorted(&s, 0.5, QuantileRule::Interpolated), 4.0);
        // rank 0.25·4 = 1 exactly; rank 0.6·4 = 2.4 → 4 + 0.4·(8 − 4).
        assert_eq!(quantile_sorted(&s, 0.25, QuantileRule::Interpolated), 2.0);
        assert!((quantile_sorted(&s, 0.6, QuantileRule::Interpolated) - 5.6).abs() < 1e-12);
    }

    #[test]
    fn type6_sits_at_rank_p_times_d_plus_one() {
        // D = 9 draws 1..=9: p = 0.2 → one-based rank 2, p = 0.25 → rank 2.5.
        let s: Vec<f64> = (1..=9).map(f64::from).collect();
        assert!((quantile_sorted(&s, 0.2, QuantileRule::ExchangeableRank) - 2.0).abs() < 1e-12);
        assert!((quantile_sorted(&s, 0.25, QuantileRule::ExchangeableRank) - 2.5).abs() < 1e-12);
        // Ranks beyond the sample clamp to the extremes.
        assert_eq!(quantile_sorted(&s, 0.001, QuantileRule::ExchangeableRank), 1.0);
        assert_eq!(quantile_sorted(&s, 0.999, QuantileRule::ExchangeableRank), 9.0);
        // Level 0.6: p = 0.2 and 0.8 → ranks 2 and 8.
        let (lo, hi) = equal_tail_interval_sorted(&s, 0.6, QuantileRule::ExchangeableRank);
        assert!((lo - 2.0).abs() < 1e-12 && (hi - 8.0).abs() < 1e-12);
    }

    #[test]
    fn exchangeable_ranks_cover_at_the_nominal_level_for_an_exchangeable_truth() {
        // Truth and D draws exchangeable ⇒ the truth's rank is uniform on 1..=D+1, and
        // an interval on order statistics (r, s) covers with probability (s − r)/(D + 1).
        // D = 64, level 0.90: ranks 3.25 and 61.75 → coverage (61.75 − 3.25)/65 = 0.9,
        // where the type-7 ranks 4.15 and 60.85 give 56.7/65 = 0.872.
        let s: Vec<f64> = (1..=64).map(f64::from).collect();
        let (lo, hi) = equal_tail_interval_sorted(&s, 0.9, QuantileRule::ExchangeableRank);
        assert!((lo - 3.25).abs() < 1e-12 && (hi - 61.75).abs() < 1e-12);
        let (lo7, hi7) = equal_tail_interval_sorted(&s, 0.9, QuantileRule::Interpolated);
        assert!((lo7 - 4.15).abs() < 1e-12 && (hi7 - 60.85).abs() < 1e-12);
    }

    #[test]
    fn empty_sample_is_nan_and_mad_is_sigma_scaled() {
        assert!(quantile_sorted(&[], 0.5, QuantileRule::Interpolated).is_nan());
        assert!(median_sorted(&[]).is_nan());
        assert_eq!(mad_sigma(&[]), None);
        // |x − 3| = [2, 1, 0, 1, 2] → sorted [0, 1, 1, 2, 2], median 1.
        let mad = mad_sigma(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!((mad - MAD_TO_SIGMA).abs() < 1e-15);
    }
}
