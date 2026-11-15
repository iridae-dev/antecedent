//! Pooled-draw intervals for coverage runs of resampling routes.
//!
//! A route whose interval is built from `B` internal resamples costs `B` fits per coverage
//! replicate. The warp-speed Monte Carlo of Giacomini, Politis and White (2013, *Econometric
//! Theory* 29) draws a small number `K` of resamples per replicate (`K = 1` in the original
//! method) and forms the resampling law each replicate's interval would use from the draws of
//! every replicate pooled together. This module forms those intervals from per-replicate
//! `(estimate, inner draws)` pairs, so the tally machinery scores them like any other interval.
//!
//! What the pooling reproduces and what it does not:
//!
//! * The pooled law is the unconditional mixture of the resampling laws, not the law conditional
//!   on one replicate's data. Each draw is therefore re-expressed as a root (`draw - estimate`)
//!   and divided by a per-replicate `scale` the caller supplies (an analytic standard error of
//!   the same replicate), then multiplied back by the scored replicate's own `scale`. With
//!   `scale = 1` the pooled root is used as it stands, which holds the interval width fixed
//!   across replicates and is only asymptotically the product's interval.
//! * A product interval built from `B` draws carries the quantile noise of `B` draws, which
//!   shortens a percentile interval and lowers its coverage at small `B`. Pooling `R * K` draws
//!   estimates the coverage of the `B = infinity` interval unless each replicate forms its
//!   interval from a random subset of `inner` pooled draws ([`Pooling::inner`]); set `inner`
//!   to the product's `B` to keep the finite-`B` behaviour.
//! * The product's multiplicative corrections (the fixed-b factor and kernel-bias factor of a
//!   block resample, a Student-t ratio) enter through [`Replicate::factor`] and
//!   [`Construction::NormalSd::critical`]; the helper never infers them.
//!
//! The quantile rules are the two the product uses: type 7 for resampling percentiles and
//! `p (n + 1)` ranks for posterior quantiles.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    dead_code,
    reason = "each integration test binary includes this module and uses a subset of it"
)]

/// The rule that turns a probability into an order-statistic position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantileRule {
    /// Type 7, `(n - 1) p`: the rule of resampling percentile intervals.
    Interpolated,
    /// Type 6, `p (n + 1)` clamped to `[1, n]`: the rule of posterior quantile intervals.
    ExchangeableRank,
}

/// The interval a route forms from its resampling law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Construction {
    /// `estimate + factor * scale * [q_lo(root), q_hi(root)]`: the equal-tailed percentile
    /// interval of the draws, with a product multiplier on the distance from the estimate.
    Percentile {
        /// Quantile rule of the product.
        rule: QuantileRule,
    },
    /// `estimate - factor * scale * [q_hi(root), q_lo(root)]`: the basic (reverse percentile)
    /// interval.
    Basic {
        /// Quantile rule of the product.
        rule: QuantileRule,
    },
    /// `estimate - error * [q_hi(pivot), q_lo(pivot)]` with pivots
    /// `(draw - estimate) / draw_error`: the bootstrap-t interval. Needs
    /// [`Replicate::draw_errors`] and [`Replicate::error`].
    Studentized {
        /// Quantile rule of the product.
        rule: QuantileRule,
    },
    /// `estimate +- critical * factor * scale * sd(root)`: the interval of a resampling standard
    /// error, with the critical value (normal or Student-t) the product uses.
    NormalSd {
        /// Multiplier of the standard error.
        critical: f64,
    },
}

/// One coverage replicate: its estimate and the inner draws it ran.
#[derive(Clone, Debug, PartialEq)]
pub struct Replicate {
    /// Point estimate of the replicate.
    pub estimate: f64,
    /// Inner resample draws of the scored statistic (one in the original method).
    pub draws: Vec<f64>,
    /// Per-draw resampling error of each draw, aligned with `draws`; studentized only.
    pub draw_errors: Vec<f64>,
    /// Data scale of the replicate (an analytic standard error), positive; 1 pools unscaled roots.
    pub scale: f64,
    /// The product's multiplicative correction of this replicate's interval; 1 for none.
    pub factor: f64,
    /// The replicate's own error of the estimate: the multiplier of studentized pivots.
    pub error: f64,
}

impl Replicate {
    /// A replicate with a single inner draw, no scale and no correction.
    #[must_use]
    pub fn single(estimate: f64, draw: f64) -> Self {
        Self::with_draws(estimate, vec![draw])
    }

    /// A replicate with several inner draws, no scale and no correction.
    #[must_use]
    pub fn with_draws(estimate: f64, draws: Vec<f64>) -> Self {
        Self { estimate, draws, draw_errors: Vec::new(), scale: 1.0, factor: 1.0, error: f64::NAN }
    }

    /// Set the per-replicate data scale.
    #[must_use]
    pub fn scaled(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Set the product's multiplicative correction.
    #[must_use]
    pub fn corrected(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }

    /// Set the resampling error of each draw and of the estimate (studentized intervals).
    #[must_use]
    pub fn studentized(mut self, draw_errors: Vec<f64>, error: f64) -> Self {
        self.draw_errors = draw_errors;
        self.error = error;
        self
    }
}

/// How the pooled draws become each replicate's interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pooling {
    /// Two-sided level of the interval.
    pub level: f64,
    /// Fewest pooled draws that form an interval; fewer give no interval (a miss).
    pub min_pool: usize,
    /// When set, each replicate forms its interval from this many pooled draws chosen at random
    /// without replacement, which restores the quantile noise of a product that uses that many
    /// draws. `None` uses the whole pool.
    pub inner: Option<usize>,
    /// Seed of the subset choices.
    pub seed: u64,
}

impl Pooling {
    /// The whole pool at `level`, at least two draws.
    #[must_use]
    pub const fn whole_pool(level: f64) -> Self {
        Self { level, min_pool: 2, inner: None, seed: 0 }
    }
}

/// Quantile of the ascending `sorted` at probability `p` under `rule`; `NaN` when empty.
#[must_use]
pub fn quantile_sorted(sorted: &[f64], p: f64, rule: QuantileRule) -> f64 {
    let n = sorted.len();
    if n == 0 || p.is_nan() {
        return f64::NAN;
    }
    let p = p.clamp(0.0, 1.0);
    let position = match rule {
        QuantileRule::Interpolated => (n - 1) as f64 * p,
        QuantileRule::ExchangeableRank => (p * (n + 1) as f64).clamp(1.0, n as f64) - 1.0,
    };
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the position is clamped to [0, n - 1]"
    )]
    let (low, high) = (position.floor() as usize, position.ceil() as usize);
    if low == high {
        return sorted[low];
    }
    let fraction = position - low as f64;
    sorted[low] + (sorted[high] - sorted[low]) * fraction
}

fn rule_of(construction: Construction) -> QuantileRule {
    match construction {
        Construction::Percentile { rule }
        | Construction::Basic { rule }
        | Construction::Studentized { rule } => rule,
        Construction::NormalSd { .. } => QuantileRule::Interpolated,
    }
}

fn usable(replicate: &Replicate, construction: Construction) -> bool {
    let studentized = matches!(construction, Construction::Studentized { .. });
    replicate.estimate.is_finite()
        && replicate.scale.is_finite()
        && replicate.scale > 0.0
        && replicate.factor.is_finite()
        && replicate.factor > 0.0
        && (!studentized
            || (replicate.error.is_finite()
                && replicate.error > 0.0
                && replicate.draw_errors.len() == replicate.draws.len()))
}

/// The pooled roots (or pivots) of every usable replicate, in replicate order.
fn pooled_roots(replicates: &[Option<Replicate>], construction: Construction) -> Vec<f64> {
    let studentized = matches!(construction, Construction::Studentized { .. });
    let mut pool = Vec::new();
    for replicate in replicates.iter().flatten().filter(|r| usable(r, construction)) {
        for (k, &draw) in replicate.draws.iter().enumerate() {
            let divisor = if studentized { replicate.draw_errors[k] } else { replicate.scale };
            let root = (draw - replicate.estimate) / divisor;
            if root.is_finite() && divisor.is_finite() && divisor > 0.0 {
                pool.push(root);
            }
        }
    }
    pool
}

/// `SplitMix64` step.
pub fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `count` entries of `pool` chosen without replacement (partial Fisher-Yates), ascending.
fn subset_sorted(pool: &[f64], count: usize, stream: u64) -> Vec<f64> {
    let mut state = stream;
    let mut indices: Vec<usize> = (0..pool.len()).collect();
    for i in 0..count {
        let span = (pool.len() - i) as u64;
        let j = i + usize::try_from(next_u64(&mut state) % span).unwrap_or(0);
        indices.swap(i, j);
    }
    let mut chosen: Vec<f64> = indices[..count].iter().map(|&i| pool[i]).collect();
    chosen.sort_by(f64::total_cmp);
    chosen
}

fn sample_sd(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// The interval of each replicate from the pooled draws of all of them.
///
/// A `None` replicate (a refused run), a replicate with an unusable estimate, scale, correction
/// or error, and a pool smaller than [`Pooling::min_pool`] give `None`, which the tally scores
/// as a miss through `record(None, truth)`. The result is aligned with `replicates`.
#[must_use]
pub fn pooled_intervals(
    replicates: &[Option<Replicate>],
    construction: Construction,
    pooling: Pooling,
) -> Vec<Option<(f64, f64)>> {
    let rule = rule_of(construction);
    let tail = (1.0 - pooling.level) / 2.0;
    let mut whole = pooled_roots(replicates, construction);
    whole.sort_by(f64::total_cmp);
    let enough = whole.len() >= pooling.min_pool.max(2);
    let subset_size = pooling.inner.filter(|&b| b >= 2 && b < whole.len());
    replicates
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            let replicate = slot.as_ref().filter(|r| usable(r, construction))?;
            if !enough {
                return None;
            }
            let drawn;
            let law: &[f64] = match subset_size {
                Some(count) => {
                    let stream = pooling.seed ^ (index as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                    drawn = subset_sorted(&whole, count, stream);
                    &drawn
                }
                None => &whole,
            };
            let at = |p: f64| quantile_sorted(law, p, rule);
            let (estimate, reach) = (replicate.estimate, replicate.factor * replicate.scale);
            let interval = match construction {
                Construction::Percentile { .. } => {
                    (estimate + reach * at(tail), estimate + reach * at(1.0 - tail))
                }
                Construction::Basic { .. } => {
                    (estimate - reach * at(1.0 - tail), estimate - reach * at(tail))
                }
                Construction::Studentized { .. } => {
                    let error = replicate.error * replicate.factor;
                    (estimate - error * at(1.0 - tail), estimate - error * at(tail))
                }
                Construction::NormalSd { critical } => {
                    let half = critical * reach * sample_sd(law);
                    (estimate - half, estimate + half)
                }
            };
            (interval.0.is_finite() && interval.1.is_finite()).then_some(interval)
        })
        .collect()
}
