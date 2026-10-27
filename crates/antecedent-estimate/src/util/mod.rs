//! Shared estimation helpers (SOLID/DRY).
//!
//! The helpers are grouped by concern and re-exported here so every caller keeps one path:
//! [`bootstrap`] (resampling standard errors and intervals, the sampling design),
//! [`scratch`] (admission of retained buffers), [`least_squares`] (column-major fits and
//! inverses), [`gather`] (resampled row gathers and arm moments) and [`refusal`] (overlap
//! and structured refusals). The scalar kernels and the small numeric helpers live below.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

mod bootstrap;
mod gather;
mod least_squares;
mod refusal;
mod scratch;

#[cfg(test)]
pub(crate) mod interval_test_support;

pub(crate) use bootstrap::{
    BOOTSTRAP_MAX_FAILURE_FRAC, BootstrapDesign, BootstrapOutcome, adaptive_bootstrap_should_stop,
    bootstrap_by_design, bootstrap_interval_by_design, bootstrap_outcome_by_design,
    bootstrap_se_with_scratch, finalize_bootstrap_se_ex, rows::ClusterRows,
};
pub use bootstrap::{
    BootstrapSeResult, IntervalRequest, ReplicatePolicy, ResamplingReport, SamplingDeclaration,
    WaveBootstrap, bootstrap_interval_with_scratch,
};
#[cfg(test)]
pub(crate) use bootstrap::{bootstrap_se, finalize_bootstrap_se};
pub(crate) use gather::{
    arm_moments, gather, gather_bootstrap_vector, gather_cols, gather_cols_into,
    gather_count_weighted, gather_into, range, resample_counts,
};
pub(crate) use least_squares::{
    delta_method_se, fit_logistic_ridge, fit_logistic_weighted, gram, invert_fixed, invert_rows,
    ls_fit, ols_sigma2, solve_spd, xtx_inverse,
};
pub(crate) use refusal::{refuse, refuse_explicit_override, require_explicit_override};
pub use scratch::admit_gather_scratch;
pub(crate) use scratch::{admit_array, admit_bound_array};

/// The overflow-free logistic function shared by every estimator; with [`dot`] and [`mean`]
/// (compensated, `NaN` for an empty slice, so a caller that must refuse emptiness tests the
/// length first) it is the crate's one scalar kernel set.
pub(crate) use antecedent_kernels::sigmoid;

/// The 64-bit identity word of a finished canonical stream: the first eight bytes of its
/// BLAKE3 digest, little-endian. For identities that are carried as a `u64` field; anything
/// stored as a whole digest uses [`CanonicalWriter::finish`] directly.
#[must_use]
pub(crate) fn canonical_word(writer: &antecedent_core::CanonicalWriter) -> u64 {
    let digest = writer.finish();
    u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}

/// `n` draws from `N(mean, variance)` on the seeded direct-draw stream of the shared
/// multivariate sampler; `None` when `variance` is not positive and finite (the sampler's
/// Cholesky refuses it) or the draw fails.
#[must_use]
pub(crate) fn normal_draws(mean: f64, variance: f64, n: usize, seed: u64) -> Option<Vec<f64>> {
    if !(variance.is_finite() && variance > 0.0) {
        return None;
    }
    antecedent_prob::sample_gaussian_mvn(
        &[mean],
        &[variance],
        n,
        seed,
        &mut antecedent_prob::LaplaceWorkspace::default(),
    )
    .ok()
    .map(|draws| draws.to_vec())
}

/// Slack of every correlation-matrix factorization, relative to its unit diagonal: a pivot
/// within it of zero is a rank-deficient direction (perfectly collinear scores, an identical
/// estimand), one below it is an indefinite matrix. Also the tolerance of the unit-diagonal
/// and symmetry checks that precede the factorization.
pub(crate) const CORRELATION_PSD_TOLERANCE: f64 = 1e-9;

/// Two-sided normal p-value `2·(1 − Φ(|z|))` of a standardized statistic; `NaN` propagates.
pub(crate) fn normal_two_sided_p(z: f64) -> f64 {
    2.0 * antecedent_kernels::norm_sf(z.abs())
}

/// A row or parameter count as `f64`.
#[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
pub(crate) const fn count(n: usize) -> f64 {
    n as f64
}

pub(crate) use antecedent_kernels::{dot, mean};

/// Unbiased sample standard deviation; `NaN` if fewer than 2 observations.
pub(crate) use antecedent_kernels::sample_std;

/// Monte Carlo critical value of a max-statistic band: the `⌈level·(B+1)⌉`-th
/// smallest of `B` ascending-sorted replicate maxima.
///
/// The `B + 1` rank treats the observed statistic as one more exchangeable draw
/// (Davison & Hinkley 1997, §4.2), so the band is conservative wherever the rank
/// exists; `⌈level·B⌉` sits one rank lower whenever `level·B` is not an integer.
/// When `⌈level·(B+1)⌉ > B` (`B < level/(1−level)`: 19 at 0.95, 99 at 0.99) the
/// exact critical value is `+∞` — the sample maximum only covers `B/(B+1) < level`
/// — so `+∞` is returned rather than a narrower, anti-conservative maximum. Every
/// simultaneous response band (multiplier, Gaussian max-t and replicate sup-t)
/// reads its critical value here. `NaN` for an empty slice.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the rank is at least 1 and the code below rejects any rank above b"
)]
#[allow(clippy::cast_sign_loss, reason = "the rank is clamped to at least 1 before the cast")]
pub(crate) fn monte_carlo_critical(sorted_maxima: &[f64], level: f64) -> f64 {
    let b = sorted_maxima.len();
    if b == 0 {
        return f64::NAN;
    }
    // The tolerance keeps a product that lands one ulp above an exact integer (e.g.
    // `0.95 · 20`) from bumping the rank past `B`.
    let rank = ((level * (b as f64 + 1.0) - 1e-9).ceil().max(1.0)) as usize;
    if rank > b {
        return f64::INFINITY;
    }
    sorted_maxima[rank - 1]
}

/// How many seeded posterior or resampling draws an estimator takes, and from which seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawSpec {
    /// Number of draws.
    pub draws: usize,
    /// Master seed of the draw stream.
    pub seed: u64,
}

impl DrawSpec {
    /// `draws` draws from `seed`.
    #[must_use]
    pub const fn new(draws: usize, seed: u64) -> Self {
        Self { draws, seed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monte_carlo_critical_is_infinite_when_the_exact_rank_does_not_exist() {
        let maxima: Vec<f64> = (1..=10).map(f64::from).collect();
        // ⌈0.95·11⌉ = 11 > 10: the sample maximum only covers 10/11 < 0.95.
        assert!(monte_carlo_critical(&maxima, 0.95).is_infinite());
        // B = 19: ⌈0.95·20⌉ = 19 ≤ 19, so the maximum is the exact critical value.
        let nineteen: Vec<f64> = (1..=19).map(f64::from).collect();
        assert!((monte_carlo_critical(&nineteen, 0.95) - 19.0).abs() < 1e-12);
        // B = 99 at 0.99: ⌈0.99·100⌉ = 99.
        let ninety_nine: Vec<f64> = (1..=99).map(f64::from).collect();
        assert!((monte_carlo_critical(&ninety_nine, 0.99) - 99.0).abs() < 1e-12);
        let ninety_eight: Vec<f64> = (1..=98).map(f64::from).collect();
        assert!(monte_carlo_critical(&ninety_eight, 0.99).is_infinite());
        // Interior rank: B = 100, ⌈0.95·101⌉ = 96.
        let hundred: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((monte_carlo_critical(&hundred, 0.95) - 96.0).abs() < 1e-12);
        assert!(monte_carlo_critical(&[], 0.95).is_nan());
    }
}
