//! Exact intervals for a binomial proportion.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::special::beta_quantile;

/// The exact two-sided Clopper–Pearson interval for a binomial proportion with `successes`
/// of `trials` at `level`: each end is a beta quantile. `None` for no trials, more successes
/// than trials, or a level outside `(0, 1)`.
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
pub fn clopper_pearson(successes: u64, trials: u64, level: f64) -> Option<(f64, f64)> {
    if trials == 0 || successes > trials || !proper_level(level) {
        return None;
    }
    let tail = (1.0 - level) / 2.0;
    let (x, n) = (successes as f64, trials as f64);
    let lower = if successes == 0 { 0.0 } else { beta_quantile(tail, x, n - x + 1.0) };
    let upper = if successes == trials { 1.0 } else { beta_quantile(1.0 - tail, x + 1.0, n - x) };
    Some((lower, upper))
}

/// The exact one-sided Clopper–Pearson upper bound for a binomial proportion with
/// `successes` of `trials` at `level`; with no successes it is `1 - (1 - level)^(1/trials)`.
/// `None` as for [`clopper_pearson`].
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
pub fn clopper_pearson_upper(successes: u64, trials: u64, level: f64) -> Option<f64> {
    if trials == 0 || successes > trials || !proper_level(level) {
        return None;
    }
    if successes == trials {
        return Some(1.0);
    }
    let (x, n) = (successes as f64, trials as f64);
    Some(beta_quantile(level, x + 1.0, n - x))
}

/// Whether `level` is a probability strictly between zero and one.
fn proper_level(level: f64) -> bool {
    level.is_finite() && level > 0.0 && level < 1.0
}
