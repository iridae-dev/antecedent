//! Streaming mean / sum-of-squared-deviations accumulator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Welford's online accumulator for the (optionally weighted) mean and the sum of
/// squared deviations from it.
///
/// The naive `Σx²/n − mean²` form loses every significant digit once
/// `mean² ≫ variance` (decision utilities in currency units, log-likelihood gaps);
/// this update never forms that difference, so the relative error of the variance
/// is independent of the location of the data.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Welford {
    mass: f64,
    count: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    /// Empty accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self { mass: 0.0, count: 0, mean: 0.0, m2: 0.0 }
    }

    /// Add one observation with unit weight.
    pub fn push(&mut self, x: f64) {
        self.push_weighted(1.0, x);
    }

    /// Add one observation with a nonnegative `weight` (zero weights are ignored).
    pub fn push_weighted(&mut self, weight: f64, x: f64) {
        if weight == 0.0 {
            return;
        }
        let total = self.mass + weight;
        let delta = x - self.mean;
        self.mean += (weight / total) * delta;
        self.m2 += weight * delta * (x - self.mean);
        self.mass = total;
        self.count += 1;
    }

    /// Number of observations added.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Total weight (`count` when every weight is 1).
    #[must_use]
    pub const fn mass(&self) -> f64 {
        self.mass
    }

    /// Running (weighted) mean; `0` when empty.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Running (weighted) sum of squared deviations from the mean.
    #[must_use]
    pub const fn m2(&self) -> f64 {
        self.m2
    }

    /// Bessel-corrected sample variance for unit weights; `None` for fewer than 2 observations.
    #[must_use]
    pub fn sample_variance(&self) -> Option<f64> {
        (self.count >= 2).then(|| (self.m2 / (self.mass - 1.0)).max(0.0))
    }

    /// Standard error of the mean for unit weights, `sqrt(s² / n)`; `+∞` for fewer than
    /// 2 observations (a single draw carries no variance information).
    #[must_use]
    pub fn stderr_of_mean(&self) -> f64 {
        self.sample_variance().map_or(f64::INFINITY, |v| (v / self.mass).sqrt())
    }
}

/// One step of the multivariate Welford scatter used by streaming covariance.
///
/// `n_before` is the count before this row and must be at least 1. `mean` has
/// length `dim` and `m2` is row-major `dim×dim`. The update uses the old mean
/// on both axes, then the mean. The integer count is the one the streaming
/// covariance already used; it is not the `f64` mass of [`Welford`], so a
/// diagonal entry can differ from [`Welford::m2`] in the last bit.
///
/// A short buffer or a zero count does nothing.
pub fn update_scatter(mean: &mut [f64], m2: &mut [f64], n_before: u64, x: &[f64]) {
    let dim = mean.len();
    let next = n_before.saturating_add(1);
    if n_before == 0 || x.len() < dim || m2.len() < dim.saturating_mul(dim) || next == 0 {
        return;
    }
    let correction = n_before as f64 / next as f64;
    for i in 0..dim {
        let delta_i = x[i] - mean[i];
        for j in i..dim {
            let value = m2[i * dim + j] + delta_i * (x[j] - mean[j]) * correction;
            m2[i * dim + j] = value;
            m2[j * dim + i] = value;
        }
    }
    for (slot, value) in mean.iter_mut().zip(x) {
        *slot += (*value - *slot) / next as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_survives_a_huge_offset() {
        // 1e8 + {-2,-1,0,1,2}: exact sample variance 2.5, exact mean 1e8.
        let mut w = Welford::new();
        for e in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            w.push(1e8 + e);
        }
        assert!((w.mean() - 1e8).abs() < 1e-6);
        assert!((w.sample_variance().unwrap() - 2.5).abs() < 1e-9);
        assert!((w.stderr_of_mean() - (2.5_f64 / 5.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn weighted_matches_repeated_unit_weights() {
        let mut a = Welford::new();
        a.push_weighted(3.0, 1.0);
        a.push_weighted(1.0, 5.0);
        let mut b = Welford::new();
        for x in [1.0, 1.0, 1.0, 5.0] {
            b.push(x);
        }
        assert!((a.mean() - b.mean()).abs() < 1e-14);
        assert!((a.m2() - b.m2()).abs() < 1e-14);
    }

    #[test]
    fn fewer_than_two_observations_have_infinite_stderr() {
        let mut w = Welford::new();
        assert!(w.stderr_of_mean().is_infinite());
        w.push(3.0);
        assert!(w.stderr_of_mean().is_infinite());
        assert!(w.sample_variance().is_none());
    }

    /// Unit-weight updates match the hand recurrence the HMC diagonal metric used
    /// before it stored one accumulator per coordinate, including a huge offset.
    #[test]
    fn unit_pushes_match_the_hand_recurrence_bit_for_bit() {
        let samples = [1e8, 1e8 + 1.0, 1e8 - 2.0, 3.5, -4.0];
        let mut acc = Welford::new();
        let (mut count, mut mean, mut m2) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &v in &samples {
            acc.push(v);
            count += 1.0;
            let delta = v - mean;
            mean += delta / count;
            m2 += delta * (v - mean);
        }
        assert_eq!(acc.mass().to_bits(), count.to_bits());
        assert_eq!(acc.mean().to_bits(), mean.to_bits());
        assert_eq!(acc.m2().to_bits(), m2.to_bits());
        let n = count;
        let weight = n / (n + 5.0);
        let from_acc = weight * (acc.m2() / (n - 1.0)) + (1.0 - weight) * 1e-3;
        let from_hand = weight * (m2 / (n - 1.0)) + (1.0 - weight) * 1e-3;
        assert_eq!(from_acc.to_bits(), from_hand.to_bits());
    }

    /// The extracted scatter is the loop `StreamingCovariance::append` used,
    /// including symmetry of the off-diagonal.
    #[test]
    fn scatter_matches_the_previous_inline_loop() {
        let rows = [[1.0, 4.0], [3.0, 0.5], [2.0, 8.0], [1e8, 1e8 + 1.0]];
        let mut mean = [0.0; 2];
        let mut m2 = [0.0; 4];
        let mut mean_ref = [0.0; 2];
        let mut m2_ref = [0.0; 4];
        mean.copy_from_slice(&rows[0]);
        mean_ref.copy_from_slice(&rows[0]);
        for (n, row) in rows.iter().enumerate().skip(1) {
            let n_before = n as u64;
            update_scatter(&mut mean, &mut m2, n_before, row);
            let next = n_before + 1;
            let correction = n_before as f64 / next as f64;
            for i in 0..2 {
                let delta_i = row[i] - mean_ref[i];
                for j in i..2 {
                    let value = m2_ref[i * 2 + j] + delta_i * (row[j] - mean_ref[j]) * correction;
                    m2_ref[i * 2 + j] = value;
                    m2_ref[j * 2 + i] = value;
                }
            }
            for i in 0..2 {
                mean_ref[i] += (row[i] - mean_ref[i]) / next as f64;
            }
        }
        assert_eq!(mean[0].to_bits(), mean_ref[0].to_bits());
        assert_eq!(mean[1].to_bits(), mean_ref[1].to_bits());
        // This is a differential against the exact former update order, so bit equality is
        // the contract rather than an approximate numerical comparison.
        assert_eq!(m2.map(f64::to_bits), m2_ref.map(f64::to_bits));
    }
}
