//! Confidence and credible intervals for a finite identified set (C-3).
//!
//! A class-aware scalar effect with several identified completions has an
//! identified set `{θ_g}` whose hull `[θ_l, θ_u] = [min_g θ_g, max_g θ_g]` is
//! published as `identified_set`. Its point bounds carry no sampling
//! uncertainty. The Frequentist interval here covers the true effect `θ_{g*}`
//! (whichever completion is the true graph) with asymptotic probability at
//! least `level`, following Imbens & Manski (2004) with per-completion
//! endpoints:
//!
//! `[min_g (θ̂_g − c·σ_g), max_g (θ̂_g + c·σ_g)]`, with `c` solving
//! `Φ(c + Δ̃ / max_g σ_g) − Φ(−c) = level`.
//!
//! * `θ̂_g` is completion `g`'s point estimate and `σ_g` its own sampling SD,
//!   taken from replicates in which **every** completion is refit on the same
//!   resample. `σ_g` is not the SD of the min / max of estimates: when a noisy
//!   completion sits next to a precise one, the min is capped by the precise
//!   estimate and its SD falls well below the noisy completion's own, so an
//!   interval built on the SD of the min under-covers the noisy completion.
//! * Why `c` stays valid. Each completion's own interval `θ̂_g ± c·σ_g` lies
//!   inside the published one. Take the true completion at distance `d_l` above
//!   the lowest effect and `d_u` below the highest (`d_l + d_u = Δ`). The
//!   interval misses below only if the true completion's lower end and the
//!   lowest completion's lower end both pass `θ_{g*}`, which has probability at
//!   most `Φ(−c − d_l/σ)`; above, at most `Φ(−c − d_u/σ)`, with `σ = max_g σ_g`.
//!   The sum is convex in `d_l` on `[0, Δ]`, so it peaks at an endpoint
//!   completion, where it is `Φ(−c) + Φ(−c − Δ/σ) = 1 − level` by the choice
//!   of `c`. Coverage is at least `level` for every completion, interior ones
//!   included, and exact at an endpoint completion once the set is wide.
//! * `Δ̃` is the estimated width `Δ̂ = θ̂_u − θ̂_l`, set to zero unless it exceeds
//!   `κ·σ_Δ` with `κ = sqrt(ln n)` (the BIC-type moment-selection threshold of
//!   Andrews & Soares 2010). IM's interpolation from a two-sided (`Δ = 0`) to a
//!   one-sided (`Δ` large) critical value assumes `Δ̂` is superefficient at zero
//!   (Stoye 2009). The min / max of completion estimates is not: when completions
//!   nearly agree, `Δ̂` is biased upward, and plugging it in would shrink `c`
//!   toward the one-sided value exactly where the set is a point. Below the
//!   threshold the two-sided critical value is used, and every completion's own
//!   two-sided interval is inside the published one.
//! * Coverage is above nominal while completions nearly agree (the two-sided
//!   `c` with several completions' intervals in the hull), and nominal at an
//!   endpoint completion once completions are well separated.
//!
//! The Bayesian analogue ([`imbens_manski_posterior_draws`]) is a statement
//! about posteriors, not a Frequentist IM interval: its endpoints are
//! quantiles of the per-draw min / max of independent completion posteriors at
//! tail probability `1 − Φ(c)`, so every completion's posterior puts at most
//! `1 − Φ(c)` of its mass below the lower endpoint and at most `1 − Φ(c)` above
//! the upper one.
//!
//! In both constructions the set spans the completions supplied. When the
//! caller's completion enumeration (or its equivalence audit) was capped, the
//! set covers retained completions only and [`IdentifiedSetInterval::truncated`]
//! records it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use antecedent_kernels::{norm_cdf, norm_inv};

/// How an [`IdentifiedSetInterval`] was computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IdentifiedSetIntervalMethod {
    /// Frequentist Imbens–Manski interval with per-completion endpoints from
    /// shared circular-block replicates (every completion refit on the same
    /// resample; SDs fixed-b scaled).
    ImbensManskiSharedBlock,
    /// Product-posterior envelope: per-draw min / max of independent completion
    /// draws, then quantiles at Imbens–Manski tail probabilities `1 − Φ(c)`.
    /// Not a Frequentist IM interval (the SDs are posterior SDs, and `c` is not a
    /// posterior probability).
    ProductPosteriorEnvelopeQuantile,
}

impl IdentifiedSetIntervalMethod {
    /// Every construction, so serialisers can check they map each one.
    pub const ALL: [Self; 2] =
        [Self::ImbensManskiSharedBlock, Self::ProductPosteriorEnvelopeQuantile];
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
    /// Endpoint SD below the lower bound, `(bound_lower − lower) / critical_value`,
    /// so that `lower = bound_lower − critical_value · lower_se`. Frequentist: the
    /// largest reach of any completion's own interval below the lower bound,
    /// `max_g (σ_g − (θ̂_g − θ̂_l) / c)`, at least the lowest completion's SD.
    pub lower_se: f64,
    /// Endpoint SD above the upper bound: `upper = bound_upper + critical_value · upper_se`.
    pub upper_se: f64,
    /// Critical value `c`: two-sided `z_{(1+level)/2}` when the width is not
    /// retained, falling toward one-sided `z_level` as the set widens.
    pub critical_value: f64,
    /// Whether the estimated width passed the `κ·σ_Δ` moment-selection threshold.
    pub width_retained: bool,
    /// Identified completions whose effects enter the set: every fitted
    /// completion, completions with coinciding fitted models counted separately.
    pub completions: usize,
    /// Replicates or posterior draws behind the SDs.
    pub replicates: usize,
    /// Construction.
    pub method: IdentifiedSetIntervalMethod,
    /// The completion enumeration behind the set (or its equivalence audit) was
    /// capped: the set spans retained completions only, and the coverage
    /// statement holds only when the true graph is one of them.
    pub truncated: bool,
}

impl IdentifiedSetInterval {
    /// Mark whether the set spans a capped (retained-only) completion enumeration.
    #[must_use]
    pub const fn with_truncated(mut self, truncated: bool) -> Self {
        self.truncated = truncated;
        self
    }

    /// Tail probability `1 − Φ(c)` of the critical value: the posterior mass a
    /// completion may place beyond each endpoint of a product-posterior interval.
    #[must_use]
    pub fn tail_probability(&self) -> f64 {
        1.0 - norm_cdf(self.critical_value)
    }
}

/// Moment-selection threshold `κ = sqrt(ln n)`, floored at 1.
fn selection_threshold(rows: usize) -> f64 {
    (rows.max(3) as f64).ln().sqrt().max(1.0)
}

fn sample_sd<I>(values: I) -> f64
where
    I: ExactSizeIterator<Item = f64> + Clone,
{
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = values.clone().sum::<f64>() / n as f64;
    (values.map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
}

/// IM critical value: solve `Φ(c + r) − Φ(−c) = level` for `c ≥ 0` with
/// `r = Δ̃ / max_g σ_g ≥ 0`, by bisection between the one- and two-sided values.
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

/// Per-draw min / max over completions (draws with a non-finite value skipped).
struct BoundDraws {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl BoundDraws {
    fn new<D, I>(draws: D) -> Self
    where
        D: Iterator<Item = I>,
        I: Iterator<Item = f64>,
    {
        let mut out = Self { lower: Vec::new(), upper: Vec::new() };
        for draw in draws {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            let mut finite = true;
            for value in draw {
                finite &= value.is_finite();
                lo = lo.min(value);
                hi = hi.max(value);
            }
            if finite && lo.is_finite() && hi.is_finite() {
                out.lower.push(lo);
                out.upper.push(hi);
            }
        }
        out
    }

    fn width_sd(&self) -> f64 {
        sample_sd(self.lower.iter().zip(&self.upper).map(|(lo, hi)| hi - lo))
    }
}

/// Frequentist IM interval with per-completion endpoints from shared replicates.
///
/// `points[g]` are the completion point estimates; `draws[b][g]` completion
/// `g`'s refit on replicate `b` (every completion on the same resample).
/// `se_scale` multiplies every replicate SD (the fixed-b factor of the block
/// bootstrap); `rows` is the resampled row count (for the selection threshold).
/// A replicate with a non-finite refit is dropped for every completion, so all
/// SDs come from the same resamples. `None` with fewer than two replicates or a
/// non-finite SD.
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
    let complete: Vec<&[f64]> =
        draws.iter().map(Vec::as_slice).filter(|draw| draw.iter().all(|v| v.is_finite())).collect();
    let sds: Vec<f64> = (0..points.len())
        .map(|g| sample_sd(complete.iter().map(|draw| draw[g])) * se_scale)
        .collect();
    let bounds = BoundDraws::new(complete.iter().map(|draw| draw.iter().copied()));
    let width_se = bounds.width_sd() * se_scale;
    let selection = Selection::new(points, &sds, width_se, rows, level, complete.len())?;
    let c = selection.critical_value;
    let lower = points.iter().zip(&sds).map(|(p, s)| p - c * s).fold(f64::INFINITY, f64::min);
    let upper = points.iter().zip(&sds).map(|(p, s)| p + c * s).fold(f64::NEG_INFINITY, f64::max);
    selection.finish(lower, upper, IdentifiedSetIntervalMethod::ImbensManskiSharedBlock)
}

/// Bayesian analogue from per-completion posterior draws.
///
/// `posterior_means[g]` are completion posterior means; `draws[g]` completion
/// `g`'s effect draws. Completions must be fitted on independent random-number
/// streams (completions whose fitted models coincide may share one posterior,
/// which leaves the per-draw min / max unchanged), so pairing draws by index to
/// form the per-draw min / max carries no cross-completion dependence. The
/// critical value follows the IM rule on the largest completion posterior SD,
/// and each endpoint is the posterior quantile of the per-draw min (max) at tail
/// probability `1 − Φ(c)`: equal-tailed when the width is not retained,
/// one-sided per endpoint as the set widens. The per-draw min lies below every
/// completion's draw, so every completion's posterior puts at most `1 − Φ(c)` of
/// its mass below the lower endpoint (and likewise above the upper one).
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
    let sds: Vec<f64> = draws.iter().map(|d| sample_sd(d[..count].iter().copied())).collect();
    let bounds = BoundDraws::new((0..count).map(|k| draws.iter().map(move |d| d[k])));
    let selection =
        Selection::new(posterior_means, &sds, bounds.width_sd(), rows, level, bounds.lower.len())?;
    let tail = 1.0 - norm_cdf(selection.critical_value);
    let lower = quantile(&bounds.lower, tail);
    let upper = quantile(&bounds.upper, 1.0 - tail);
    selection.finish(lower, upper, IdentifiedSetIntervalMethod::ProductPosteriorEnvelopeQuantile)
}

/// Bounds, moment selection and critical value shared by both constructions.
struct Selection {
    level: f64,
    bound_lower: f64,
    bound_upper: f64,
    critical_value: f64,
    width_retained: bool,
    completions: usize,
    replicates: usize,
}

impl Selection {
    fn new(
        points: &[f64],
        sds: &[f64],
        width_se: f64,
        rows: usize,
        level: f64,
        replicates: usize,
    ) -> Option<Self> {
        if !(level > 0.0 && level < 1.0)
            || replicates < 2
            || points.is_empty()
            || points.len() != sds.len()
            || points.iter().any(|p| !p.is_finite())
        {
            return None;
        }
        let bound_lower = points.iter().copied().fold(f64::INFINITY, f64::min);
        let bound_upper = points.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !(bound_lower.is_finite()
            && bound_upper.is_finite()
            && sds.iter().all(|s| s.is_finite() && *s >= 0.0))
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
        let scale = sds.iter().copied().fold(0.0, f64::max);
        let standardized = if width_retained && scale > 0.0 { width / scale } else { 0.0 };
        Some(Self {
            level,
            bound_lower,
            bound_upper,
            critical_value: imbens_manski_critical_value(level, standardized),
            width_retained,
            completions: points.len(),
            replicates,
        })
    }

    fn finish(
        self,
        lower: f64,
        upper: f64,
        method: IdentifiedSetIntervalMethod,
    ) -> Option<IdentifiedSetInterval> {
        if !(lower.is_finite() && upper.is_finite() && lower <= upper) {
            return None;
        }
        let c = self.critical_value;
        let endpoint_se = |reach: f64| if c > 0.0 { (reach / c).max(0.0) } else { 0.0 };
        Some(IdentifiedSetInterval {
            level: self.level,
            lower,
            upper,
            bound_lower: self.bound_lower,
            bound_upper: self.bound_upper,
            lower_se: endpoint_se(self.bound_lower - lower),
            upper_se: endpoint_se(upper - self.bound_upper),
            critical_value: c,
            width_retained: self.width_retained,
            completions: self.completions,
            replicates: self.replicates,
            method,
            truncated: false,
        })
    }
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

    fn column_sd(draws: &[Vec<f64>], g: usize) -> f64 {
        sample_sd(draws.iter().map(|d| d[g]))
    }

    #[test]
    fn review_nonfinite_completion_point_cannot_be_silently_dropped() {
        let draws = vec![vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0]];
        assert!(
            imbens_manski_shared_replicates(&[1.0, f64::NAN], &draws, 1.0, 100, 0.95).is_none()
        );
        assert!(
            imbens_manski_posterior_draws(
                &[1.0, f64::NAN],
                &[&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]],
                100,
                0.95
            )
            .is_none()
        );
    }

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
        let sd = column_sd(&draws, 0);
        let im = imbens_manski_shared_replicates(&[1.0], &draws, 1.1, 160, 0.9).unwrap();
        assert!(!im.width_retained);
        assert!(!im.truncated);
        assert!((im.lower - (1.0 - 1.644_853_6 * sd * 1.1)).abs() < 1e-6);
        assert!((im.upper - (1.0 + 1.644_853_6 * sd * 1.1)).abs() < 1e-6);
        assert!((im.lower_se - sd * 1.1).abs() < 1e-9 && (im.upper_se - sd * 1.1).abs() < 1e-9);
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
        // Each endpoint is its own completion's one-sided end.
        let c = im.critical_value;
        assert!((im.lower - (0.8 - c * column_sd(&draws, 0))).abs() < 1e-12);
        assert!((im.upper - (1.3 + c * column_sd(&draws, 1))).abs() < 1e-12);
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

    /// A noisy completion next to a precise one: the min of the two is capped by
    /// the precise estimate, so its SD falls far below the noisy completion's own.
    /// The lower endpoint must reach the noisy completion's own `c·σ`.
    #[test]
    fn noisy_completion_keeps_its_own_sd_next_to_a_precise_one() {
        let draws: Vec<Vec<f64>> = (0..400)
            .map(|b| {
                let e = (f64::from(b % 41) - 20.0) / 11.8;
                let f = (f64::from((b * 13) % 41) - 20.0) / 11.8;
                vec![e, 0.4 + 0.2 * f]
            })
            .collect();
        let noisy = column_sd(&draws, 0);
        let min_sd = sample_sd(draws.iter().map(|d| d[0].min(d[1])));
        assert!(min_sd < 0.8 * noisy, "fixture: the min must be capped ({min_sd} vs {noisy})");
        let im = imbens_manski_shared_replicates(&[0.0, 0.4], &draws, 1.0, 400, 0.9).unwrap();
        let c = im.critical_value;
        assert!((im.lower - (0.0 - c * noisy)).abs() < 1e-12, "{im:?}");
        assert!((im.lower_se - noisy).abs() < 1e-12);
        // The precise completion's upper end is inside the noisy one's: the
        // upper endpoint is whichever completion reaches further.
        let reach = (0.0 + c * noisy).max(0.4 + c * column_sd(&draws, 1));
        assert!((im.upper - reach).abs() < 1e-12);
        assert!((im.bound_upper + c * im.upper_se - im.upper).abs() < 1e-12);
    }

    #[test]
    fn posterior_draw_interval_is_equal_tailed_for_a_point() {
        let a: Vec<f64> = (0..1000).map(|k| f64::from(k) / 999.0).collect();
        let im = imbens_manski_posterior_draws(&[0.5], &[&a], 160, 0.9).unwrap();
        assert!((im.lower - 0.05).abs() < 1e-9 && (im.upper - 0.95).abs() < 1e-9);
        assert_eq!(im.method, IdentifiedSetIntervalMethod::ProductPosteriorEnvelopeQuantile);
    }

    /// Completions sharing one posterior (coinciding fitted models) leave the
    /// interval unchanged; only the completion count records them.
    #[test]
    fn shared_posteriors_leave_the_posterior_interval_unchanged() {
        let a: Vec<f64> = (0..1000).map(|k| f64::from((k * 37) % 1000) / 999.0).collect();
        let b: Vec<f64> = (0..1000).map(|k| 2.0 + f64::from((k * 91) % 1000) / 999.0).collect();
        let one = imbens_manski_posterior_draws(&[0.5, 2.5], &[&a, &b], 160, 0.9).unwrap();
        let dup = imbens_manski_posterior_draws(&[0.5, 0.5, 2.5], &[&a, &a, &b], 160, 0.9).unwrap();
        assert_eq!((one.lower, one.upper), (dup.lower, dup.upper));
        assert_eq!((one.completions, dup.completions), (2, 3));
    }
}
