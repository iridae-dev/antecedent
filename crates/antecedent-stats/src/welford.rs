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
}
