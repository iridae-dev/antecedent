//! Confidence and credible intervals for a finite identified set (1.9, C-3).
//!
//! A class-aware scalar effect with several identified completions has an
//! identified set `{θ_g}` whose hull `[θ_l, θ_u] = [min_g θ_g, max_g θ_g]` is
//! published as `identified_set`. Its point bounds carry no sampling
//! uncertainty. The interval here covers the true effect `θ_{g*}` (whichever
//! completion is the true graph) with probability at least `level`, following
//! Imbens & Manski (2004):
//!
//! `[θ̂_l − c·σ_l, θ̂_u + c·σ_u]`, with `c` solving
//! `Φ(c + Δ̃ / max(σ_l, σ_u)) − Φ(−c) = level`.
//!
//! * `θ̂_l`, `θ̂_u` are the min / max of the completion point estimates, and
//!   `σ_l`, `σ_u` the sampling SDs of those min / max, taken from replicates in
//!   which **every** completion is refit on the same resample (Frequentist), or
//!   from per-draw min / max of the completion posteriors (Bayesian).
//! * `Δ̃` is the estimated width `Δ̂ = θ̂_u − θ̂_l`, set to zero unless it exceeds
//!   `κ·σ_Δ` with `κ = sqrt(ln n)` (the BIC-type moment-selection threshold of
//!   Andrews & Soares 2010). IM's interpolation from a two-sided (`Δ = 0`) to a
//!   one-sided (`Δ` large) critical value assumes `Δ̂` is superefficient at zero
//!   (Stoye 2009). The min / max of completion estimates is not: when completions
//!   nearly agree, `Δ̂` is biased upward, and plugging it in would shrink `c`
//!   toward the one-sided value exactly where the set is a point. Below the
//!   threshold the two-sided critical value is used.
//! * The min of estimates is biased downward and the max upward when completions
//!   are close. That bias moves each endpoint outward, so it can only raise
//!   coverage; the interval is conservative when completions nearly agree and
//!   the set is narrow relative to noise, and is nominal at an endpoint
//!   completion once completions are well separated (where `c` is one-sided and
//!   the other endpoint is far away).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use antecedent_kernels::{norm_cdf, norm_inv};

/// How an [`IdentifiedSetInterval`] was computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IdentifiedSetIntervalMethod {
    /// Frequentist Imbens–Manski interval from shared circular-block replicates
    /// (every completion refit on the same resample; SDs fixed-b scaled).
    ImbensManskiSharedBlock,
    /// Bayesian analogue from per-completion posterior draws paired by draw
    /// index: endpoint quantiles of the per-draw min / max at the IM tail level.
    ImbensManskiPosteriorDraws,
}

/// Interval for the identified set of a class-aware scalar effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentifiedSetInterval {
    /// Nominal coverage of the true (completion-specific) effect.
    pub level: f64,
    /// Lower endpoint of the interval.
    pub lower: f64,
    /// Upper endpoint of the interval.
    pub upper: f64,
    /// Estimated lower bound `min_g θ̂_g`.
    pub bound_lower: f64,
    /// Estimated upper bound `max_g θ̂_g`.
    pub bound_upper: f64,
    /// Sampling SD (or posterior SD) of the estimated lower bound.
    pub lower_se: f64,
    /// Sampling SD (or posterior SD) of the estimated upper bound.
    pub upper_se: f64,
    /// Critical value `c`: two-sided `z_{(1+level)/2}` when the width is not
    /// retained, falling toward one-sided `z_level` as the set widens.
    pub critical_value: f64,
    /// Whether the estimated width passed the `κ·σ_Δ` moment-selection threshold.
    pub width_retained: bool,
    /// Completions spanning the set.
    pub completions: usize,
    /// Replicates or posterior draws behind the SDs.
    pub replicates: usize,
    /// Construction.
    pub method: IdentifiedSetIntervalMethod,
}

/// Moment-selection threshold `κ = sqrt(ln n)`, floored at 1.
fn selection_threshold(rows: usize) -> f64 {
    (rows.max(3) as f64).ln().sqrt().max(1.0)
}

fn sample_sd(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
}

/// IM critical value: solve `Φ(c + r) − Φ(−c) = level` for `c ≥ 0` with
/// `r = Δ̃ / max(σ_l, σ_u) ≥ 0`, by bisection between the one- and two-sided values.
#[must_use]
pub fn imbens_manski_critical_value(level: f64, standardized_width: f64) -> f64 {
    let two_sided = norm_inv(0.5 + level / 2.0);
    let r = standardized_width.max(0.0);
    if r == 0.0 || !r.is_finite() {
        return if r.is_finite() { two_sided } else { norm_inv(level) };
    }
    let (mut lo, mut hi) = (norm_inv(level).min(two_sided), two_sided);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if norm_cdf(mid + r) - norm_cdf(-mid) < level {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

struct BoundSummary {
    lower: Vec<f64>,
    upper: Vec<f64>,
    width: Vec<f64>,
}

fn bound_draws(draws: impl Iterator<Item = Vec<f64>>) -> BoundSummary {
    let mut out = BoundSummary { lower: Vec::new(), upper: Vec::new(), width: Vec::new() };
    for draw in draws {
        let lo = draw.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = draw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo.is_finite() && hi.is_finite() {
            out.lower.push(lo);
            out.upper.push(hi);
            out.width.push(hi - lo);
        }
    }
    out
}

/// Frequentist IM interval from shared replicates.
///
/// `points[g]` are the completion point estimates; `draws[b][g]` completion
/// `g`'s refit on replicate `b` (every completion on the same resample).
/// `se_scale` multiplies every replicate SD (the fixed-b factor of the block
/// bootstrap); `rows` is the resampled row count (for the selection threshold).
/// `None` with fewer than two replicates or a non-finite SD.
#[must_use]
pub fn imbens_manski_shared_replicates(
    points: &[f64],
    draws: &[Vec<f64>],
    se_scale: f64,
    rows: usize,
    level: f64,
) -> Option<IdentifiedSetInterval> {
    if points.is_empty() || draws.iter().any(|draw| draw.len() != points.len()) {
        return None;
    }
    let bounds = bound_draws(draws.iter().cloned());
    let lower_se = sample_sd(&bounds.lower) * se_scale;
    let upper_se = sample_sd(&bounds.upper) * se_scale;
    let width_se = sample_sd(&bounds.width) * se_scale;
    finish(
        points,
        lower_se,
        upper_se,
        width_se,
        rows,
        level,
        bounds.lower.len(),
        IdentifiedSetIntervalMethod::ImbensManskiSharedBlock,
        None,
    )
}

/// Bayesian analogue from per-completion posterior draws.
///
/// `posterior_means[g]` are completion posterior means; `draws[g]` completion
/// `g`'s effect draws. Completions are separate models, so draws are paired by
/// index only to form per-draw min / max (the pairing carries no cross-completion
/// dependence). The critical value follows the IM rule on the posterior SDs of
/// the min / max, and each endpoint is the posterior quantile of the per-draw
/// min (max) at tail probability `1 − Φ(c)`: equal-tailed when the width is not
/// retained, one-sided per endpoint as the set widens.
#[must_use]
pub fn imbens_manski_posterior_draws(
    posterior_means: &[f64],
    draws: &[&[f64]],
    rows: usize,
    level: f64,
) -> Option<IdentifiedSetInterval> {
    if posterior_means.is_empty() || draws.len() != posterior_means.len() {
        return None;
    }
    let count = draws.iter().map(|d| d.len()).min()?;
    let bounds = bound_draws((0..count).map(|k| draws.iter().map(|d| d[k]).collect::<Vec<f64>>()));
    let lower_se = sample_sd(&bounds.lower);
    let upper_se = sample_sd(&bounds.upper);
    let width_se = sample_sd(&bounds.width);
    finish(
        posterior_means,
        lower_se,
        upper_se,
        width_se,
        rows,
        level,
        bounds.lower.len(),
        IdentifiedSetIntervalMethod::ImbensManskiPosteriorDraws,
        Some((&bounds.lower, &bounds.upper)),
    )
}

#[allow(clippy::too_many_arguments)]
fn finish(
    points: &[f64],
    lower_se: f64,
    upper_se: f64,
    width_se: f64,
    rows: usize,
    level: f64,
    replicates: usize,
    method: IdentifiedSetIntervalMethod,
    quantile_draws: Option<(&[f64], &[f64])>,
) -> Option<IdentifiedSetInterval> {
    if !(level > 0.0 && level < 1.0) || replicates < 2 {
        return None;
    }
    let bound_lower = points.iter().copied().fold(f64::INFINITY, f64::min);
    let bound_upper = points.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(bound_lower.is_finite()
        && bound_upper.is_finite()
        && lower_se.is_finite()
        && upper_se.is_finite())
    {
        return None;
    }
    let width = bound_upper - bound_lower;
    // With one completion (or identical ones) the width SD is zero and the set
    // is a point: never retain it.
    let width_retained = points.len() > 1
        && width_se.is_finite()
        && width > selection_threshold(rows) * width_se
        && width > 0.0;
    let scale = lower_se.max(upper_se);
    let standardized = if width_retained && scale > 0.0 { width / scale } else { 0.0 };
    let critical_value = imbens_manski_critical_value(level, standardized);
    let (lower, upper) = match quantile_draws {
        None => (bound_lower - critical_value * lower_se, bound_upper + critical_value * upper_se),
        Some((per_draw_min, per_draw_max)) => {
            let tail = 1.0 - norm_cdf(critical_value);
            (quantile(per_draw_min, tail), quantile(per_draw_max, 1.0 - tail))
        }
    };
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(IdentifiedSetInterval {
        level,
        lower,
        upper,
        bound_lower,
        bound_upper,
        lower_se,
        upper_se,
        critical_value,
        width_retained,
        completions: points.len(),
        replicates,
        method,
    })
}

/// Linear-interpolated empirical quantile (type 7).
fn quantile(values: &[f64], p: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted.sort_by(f64::total_cmp);
    let h = (sorted.len() - 1) as f64 * p.clamp(0.0, 1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (i, frac) = (h.floor() as usize, h - h.floor());
    let next = sorted[(i + 1).min(sorted.len() - 1)];
    sorted[i] + frac * (next - sorted[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_value_moves_from_two_sided_to_one_sided() {
        let two = imbens_manski_critical_value(0.9, 0.0);
        assert!((two - 1.644_853_6).abs() < 1e-6);
        let wide = imbens_manski_critical_value(0.9, 50.0);
        assert!((wide - 1.281_551_6).abs() < 1e-6, "{wide}");
        let mid = imbens_manski_critical_value(0.9, 1.0);
        assert!(mid < two && mid > wide);
        assert!((norm_cdf(mid + 1.0) - norm_cdf(-mid) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn single_completion_is_a_two_sided_interval() {
        let draws: Vec<Vec<f64>> =
            (0..200).map(|b| vec![1.0 + 0.01 * f64::from(b % 21) - 0.1]).collect();
        let sd = sample_sd(&draws.iter().map(|d| d[0]).collect::<Vec<_>>());
        let im = imbens_manski_shared_replicates(&[1.0], &draws, 1.1, 160, 0.9).unwrap();
        assert!(!im.width_retained);
        assert!((im.lower - (1.0 - 1.644_853_6 * sd * 1.1)).abs() < 1e-6);
        assert!((im.upper - (1.0 + 1.644_853_6 * sd * 1.1)).abs() < 1e-6);
    }

    #[test]
    fn separated_completions_use_one_sided_endpoints() {
        // Two completions 0.5 apart with replicate SD ≈ 0.05 each.
        let draws: Vec<Vec<f64>> = (0..400)
            .map(|b| {
                let e = 0.05 * (f64::from(b % 41) - 20.0) / 11.8;
                vec![0.8 + e, 1.3 + 0.5 * e]
            })
            .collect();
        let im = imbens_manski_shared_replicates(&[0.8, 1.3], &draws, 1.0, 160, 0.9).unwrap();
        assert!(im.width_retained);
        assert!((im.critical_value - 1.281_551_6).abs() < 1e-3, "{}", im.critical_value);
        assert!(im.lower < 0.8 && im.upper > 1.3);
        assert_eq!(im.completions, 2);
    }

    #[test]
    fn near_equal_completions_keep_the_two_sided_value() {
        let draws: Vec<Vec<f64>> = (0..400)
            .map(|b| {
                let e = 0.05 * (f64::from(b % 41) - 20.0) / 11.8;
                let f = 0.05 * (f64::from((b * 7) % 41) - 20.0) / 11.8;
                vec![0.8 + e, 0.8 + f]
            })
            .collect();
        let im = imbens_manski_shared_replicates(&[0.80, 0.81], &draws, 1.0, 160, 0.9).unwrap();
        assert!(!im.width_retained, "a width inside the noise must not shrink c");
        assert!((im.critical_value - 1.644_853_6).abs() < 1e-6);
    }

    #[test]
    fn posterior_draw_interval_is_equal_tailed_for_a_point() {
        let a: Vec<f64> = (0..1000).map(|k| f64::from(k) / 999.0).collect();
        let im = imbens_manski_posterior_draws(&[0.5], &[&a], 160, 0.9).unwrap();
        assert!((im.lower - 0.05).abs() < 1e-9 && (im.upper - 0.95).abs() < 1e-9);
    }
}
