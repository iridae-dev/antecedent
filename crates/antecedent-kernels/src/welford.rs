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
    /// Sum of squared weights, for the reliability-weight variance and effective sample size.
    mass_sq: f64,
    count: u64,
    mean: f64,
    m2: f64,
}

/// Why a checked ingestion call was refused. The accumulator is unchanged on error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WelfordError {
    /// The observation is `NaN` or infinite.
    NonFiniteValue,
    /// The weight is `NaN`, infinite or negative.
    InvalidWeight,
}

impl core::fmt::Display for WelfordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NonFiniteValue => "observation is not finite",
            Self::InvalidWeight => "weight is not a finite nonnegative number",
        })
    }
}

impl std::error::Error for WelfordError {}

impl Welford {
    /// Empty accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self { mass: 0.0, mass_sq: 0.0, count: 0, mean: 0.0, m2: 0.0 }
    }

    /// Add one observation with unit weight.
    pub fn push(&mut self, x: f64) {
        self.push_weighted(1.0, x);
    }

    /// Add one observation with a nonnegative `weight` (zero weights are ignored).
    ///
    /// # Preconditions
    ///
    /// `weight` must be finite and nonnegative and `x` finite. This unchecked form does not
    /// verify them: a negative weight can drive the mass to zero or below and a non-finite
    /// input poisons every later moment. Use [`Self::try_push_weighted`] when either is not
    /// already guaranteed.
    pub fn push_weighted(&mut self, weight: f64, x: f64) {
        if weight == 0.0 {
            return;
        }
        let total = self.mass + weight;
        let delta = x - self.mean;
        self.mean += (weight / total) * delta;
        self.m2 += weight * delta * (x - self.mean);
        self.mass = total;
        self.mass_sq += weight * weight;
        self.count += 1;
    }

    /// Checked ingestion of one unit-weight observation.
    ///
    /// # Errors
    ///
    /// [`WelfordError::NonFiniteValue`] when `x` is not finite; the accumulator is unchanged.
    pub fn try_push(&mut self, x: f64) -> Result<(), WelfordError> {
        self.try_push_weighted(1.0, x)
    }

    /// Checked form of [`Self::push_weighted`]: refuses a non-finite observation or a weight
    /// that is not finite and nonnegative, leaving the accumulator unchanged. A zero weight is
    /// accepted and ignored.
    ///
    /// # Errors
    ///
    /// See [`WelfordError`].
    pub fn try_push_weighted(&mut self, weight: f64, x: f64) -> Result<(), WelfordError> {
        if !weight.is_finite() || weight < 0.0 {
            return Err(WelfordError::InvalidWeight);
        }
        if !x.is_finite() {
            return Err(WelfordError::NonFiniteValue);
        }
        let total = self.mass + weight;
        let total_sq = self.mass_sq + weight * weight;
        if !total.is_finite() || !total_sq.is_finite() {
            return Err(WelfordError::InvalidWeight);
        }
        self.push_weighted(weight, x);
        Ok(())
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

    /// Sample variance `m2 / (mass − 1)` under *frequency-weight* semantics (each weight is a
    /// repeat count), which is Bessel's correction for unit weights; `None` for fewer than 2
    /// observations. For reliability (precision) weights use
    /// [`Self::reliability_sample_variance`].
    #[must_use]
    pub fn sample_variance(&self) -> Option<f64> {
        (self.count >= 2).then(|| (self.m2 / (self.mass - 1.0)).max(0.0))
    }

    /// Sum of squared weights.
    #[must_use]
    pub const fn mass_squared(&self) -> f64 {
        self.mass_sq
    }

    /// Kish effective sample size `mass² / Σw²`; `None` when empty.
    #[must_use]
    pub fn effective_sample_size(&self) -> Option<f64> {
        (self.mass_sq > 0.0).then(|| self.mass * self.mass / self.mass_sq)
    }

    /// Unbiased variance under *reliability-weight* semantics,
    /// `m2 / (mass − Σw² / mass)`; `None` for fewer than 2 observations or when the
    /// denominator is not positive (all mass on one observation). Equals
    /// [`Self::sample_variance`] when every weight is 1.
    #[must_use]
    pub fn reliability_sample_variance(&self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let denominator = self.mass - self.mass_sq / self.mass;
        (denominator > 0.0).then(|| (self.m2 / denominator).max(0.0))
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
/// A short buffer or a zero count does nothing; callers that must distinguish an invalid
/// update from a successful one use [`try_update_scatter`].
pub fn update_scatter(mean: &mut [f64], m2: &mut [f64], n_before: u64, x: &[f64]) {
    // The refusal is the documented no-op of this form.
    let _ = try_update_scatter(mean, m2, n_before, x);
}

/// Why [`try_update_scatter`] refused an update. The buffers are unchanged on error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScatterError {
    /// `n_before` is zero (or the count would overflow): there is no mean to update.
    EmptyCount,
    /// `x` is shorter than `mean`, or `m2` is shorter than `dim × dim`.
    ShapeMismatch,
}

impl core::fmt::Display for ScatterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::EmptyCount => "scatter update requires at least one prior row",
            Self::ShapeMismatch => "scatter update buffers do not match the dimension",
        })
    }
}

impl std::error::Error for ScatterError {}

/// Checked form of [`update_scatter`]: identical arithmetic, but a zero count or a short buffer
/// is reported instead of silently skipped.
///
/// # Errors
///
/// See [`ScatterError`].
pub fn try_update_scatter(
    mean: &mut [f64],
    m2: &mut [f64],
    n_before: u64,
    x: &[f64],
) -> Result<(), ScatterError> {
    let dim = mean.len();
    let next = n_before.saturating_add(1);
    if n_before == 0 || next == 0 {
        return Err(ScatterError::EmptyCount);
    }
    if x.len() < dim || m2.len() < dim.saturating_mul(dim) {
        return Err(ScatterError::ShapeMismatch);
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
    Ok(())
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
    fn checked_ingestion_refuses_bad_inputs_without_mutation() {
        let mut w = Welford::new();
        w.try_push(1.0).unwrap();
        let before = w;
        assert_eq!(w.try_push(f64::NAN), Err(WelfordError::NonFiniteValue));
        assert_eq!(w.try_push_weighted(-1.0, 2.0), Err(WelfordError::InvalidWeight));
        assert_eq!(w.try_push_weighted(f64::NAN, 2.0), Err(WelfordError::InvalidWeight));
        assert_eq!(w.try_push_weighted(f64::INFINITY, 2.0), Err(WelfordError::InvalidWeight));
        assert_eq!(w, before);
        w.try_push_weighted(0.0, 5.0).unwrap();
        assert_eq!(w, before);
    }

    #[test]
    fn reliability_variance_matches_frequency_variance_for_unit_weights() {
        let mut w = Welford::new();
        for x in [1.0, 2.0, 4.0, 8.0] {
            w.push(x);
        }
        let a = w.sample_variance().unwrap();
        let b = w.reliability_sample_variance().unwrap();
        assert!((a - b).abs() < 1e-12);
        assert!((w.effective_sample_size().unwrap() - 4.0).abs() < 1e-12);
        // Unequal weights: reliability denominator mass − Σw²/mass differs from mass − 1.
        let mut u = Welford::new();
        u.push_weighted(3.0, 1.0);
        u.push_weighted(1.0, 5.0);
        let denominator = 4.0 - 10.0 / 4.0;
        assert!((u.reliability_sample_variance().unwrap() - u.m2() / denominator).abs() < 1e-12);
        let mut single = Welford::new();
        single.push_weighted(2.0, 1.0);
        assert!(single.reliability_sample_variance().is_none());
    }

    #[test]
    fn checked_scatter_reports_invalid_updates_and_matches_the_unchecked_form() {
        let mut mean = [1.0, 2.0];
        let mut m2 = [0.0; 4];
        assert_eq!(
            try_update_scatter(&mut mean, &mut m2, 0, &[1.0, 1.0]),
            Err(ScatterError::EmptyCount)
        );
        assert_eq!(
            try_update_scatter(&mut mean, &mut m2, 1, &[1.0]),
            Err(ScatterError::ShapeMismatch)
        );
        assert_eq!(
            try_update_scatter(&mut mean, &mut [0.0; 3], 1, &[1.0, 1.0]),
            Err(ScatterError::ShapeMismatch)
        );
        assert_eq!((mean, m2), ([1.0, 2.0], [0.0; 4]));
        let (mut mean_u, mut m2_u) = (mean, m2);
        try_update_scatter(&mut mean, &mut m2, 1, &[3.0, 5.0]).unwrap();
        update_scatter(&mut mean_u, &mut m2_u, 1, &[3.0, 5.0]);
        assert_eq!((mean, m2), (mean_u, m2_u));
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
