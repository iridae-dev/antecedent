//! The conjugate posterior of a response with a large common offset, against a 60-digit
//! evaluation of the batch formulas (`mpmath`, using the same binary64 inputs):
//!
//! `Λ = Λ₀ + XᵀX`, `m = Λ⁻¹(Λ₀m₀ + Xᵀy)`, `a = a₀ + n/2`,
//! `b = b₀ + ½(|y − Xm|² + (m − m₀)ᵀΛ₀(m − m₀))`
//!
//! with a prior centred on the offset `10⁸`. The residual scale is about one while the raw sums
//! of squares are about `10¹⁷`, so a rate taken as the difference of those sums returns the prior
//! rate (`b = 2.0`) instead of `2.6265...`; the block scatter that is centred on its mean does not.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_state::{ConjugateOlsPosterior, ConjugateOlsPrior, LinearOlsSuffStats};

const X: [f64; 8] = [0.5, 1.2, -0.3, 2.0, 1.5, -1.0, 0.8, 2.2];
const Y: [f64; 8] = [
    100_000_000.66,
    100_000_000.62,
    100_000_000.05,
    100_000_000.93,
    100_000_001.14,
    100_000_000.68,
    100_000_000.44,
    100_000_001.52,
];

fn block(range: std::ops::Range<usize>) -> LinearOlsSuffStats {
    let mut stats = LinearOlsSuffStats::new(2).unwrap();
    for i in range {
        stats.append_row(&[1.0, X[i]], Y[i]).unwrap();
    }
    stats
}

fn prior() -> ConjugateOlsPrior {
    ConjugateOlsPrior::new(vec![1.0e8, 0.0], vec![2.0, 0.3, 0.3, 1.0], 3.0, 2.0).unwrap()
}

fn close(actual: f64, expected: f64, relative: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs().max(1.0),
        "{label}: {actual:.17e} vs {expected:.17e}"
    );
}

#[test]
fn rate_survives_a_response_offset_of_one_hundred_million() {
    let prior = prior();
    let mut posterior = ConjugateOlsPosterior::new(&prior);
    posterior.update(&prior, "first", &block(0..4)).unwrap();
    let first = posterior.summary(&prior).unwrap();
    close(first.mean[0], 100_000_000.167_562_058_746_881, 1e-12, "first intercept");
    close(first.mean[1], 0.339_088_554_833_278_96, 1e-7, "first slope");
    close(first.shape, 5.0, 0.0, "first shape");
    close(first.rate, 2.159_455_131_242_424_5, 1e-6, "first rate");

    posterior.update(&prior, "second", &block(4..8)).unwrap();
    let both = posterior.summary(&prior).unwrap();
    close(both.mean[0], 100_000_000.374_166_183_312_137_4, 1e-12, "pooled intercept");
    close(both.mean[1], 0.319_213_635_199_249_6, 1e-7, "pooled slope");
    close(both.shape, 7.0, 0.0, "pooled shape");
    close(both.rate, 2.626_524_009_587_208, 1e-6, "pooled rate");
}

#[test]
fn rate_of_the_second_block_alone_and_of_a_posterior_used_as_a_prior() {
    let prior = prior();
    let mut alone = ConjugateOlsPosterior::new(&prior);
    alone.update(&prior, "second", &block(4..8)).unwrap();
    close(alone.summary(&prior).unwrap().rate, 2.571_005_692_741_575, 1e-6, "second rate");

    let mut posterior = ConjugateOlsPosterior::new(&prior);
    posterior.update(&prior, "first", &block(0..4)).unwrap();
    let next = posterior.to_prior().unwrap();
    let mut chained = ConjugateOlsPosterior::new(&next);
    chained.update(&next, "second", &block(4..8)).unwrap();
    let summary = chained.summary(&next).unwrap();
    close(summary.rate, 2.626_524_009_587_208, 1e-6, "chained rate");
    close(summary.mean[0], 100_000_000.374_166_183_312_137_4, 1e-12, "chained intercept");
}

#[test]
fn no_observations_leave_the_prior_rate() {
    let prior = prior();
    let summary = ConjugateOlsPosterior::new(&prior).summary(&prior).unwrap();
    close(summary.rate, 2.0, 1e-9, "empty rate");
}
