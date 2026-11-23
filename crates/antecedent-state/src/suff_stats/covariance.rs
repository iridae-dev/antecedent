//! Streaming covariance updates and Chan merges.

use super::{chan_merge_scatter, square_len, unsupported};
use crate::error::StateError;
use crate::retention::RetentionPolicy;

/// Larger count is the Chan base. Equal counts break the tie on mean bits.
fn chan_base_is_left(left_n: u64, left_mean: &[f64], right_n: u64, right_mean: &[f64]) -> bool {
    if left_n != right_n {
        return left_n > right_n;
    }
    for (left, right) in left_mean.iter().zip(right_mean) {
        let left_bits = left.to_bits();
        let right_bits = right.to_bits();
        if left_bits != right_bits {
            return left_bits < right_bits;
        }
    }
    true
}

/// Streaming mean / covariance (Welford updates, merged by Chan's formula).
#[derive(Clone, Debug, PartialEq)]
pub struct StreamingCovariance {
    /// Dimension.
    pub(super) dim: usize,
    /// Observation count.
    pub(super) n: u64,
    /// Running mean.
    pub(super) mean: Vec<f64>,
    /// Upper-triangular packed? — store full `dim×dim` unnormalized scatter `M2`.
    pub(super) m2: Vec<f64>,
    /// Retention.
    retention: RetentionPolicy,
}

impl StreamingCovariance {
    /// Empty streaming covariance for `dim`.
    ///
    /// # Errors
    ///
    /// `dim²` does not fit in `usize`.
    pub fn new(dim: usize) -> Result<Self, StateError> {
        let cells = square_len(dim, "covariance dimension overflow")?;
        Ok(Self {
            dim,
            n: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; cells],
            retention: RetentionPolicy::SufficientStatisticsOnly,
        })
    }

    /// Observation count.
    #[must_use]
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Running mean.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Unnormalized `dim×dim` scatter `M2`.
    #[must_use]
    pub fn m2(&self) -> &[f64] {
        &self.m2
    }

    /// Append one observation.
    ///
    /// # Errors
    ///
    /// Dimension mismatch.
    pub fn append(&mut self, x: &[f64]) -> Result<(), StateError> {
        if x.len() != self.dim {
            return Err(StateError::Shape(format!("cov dim {} != {}", x.len(), self.dim)));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(StateError::Numerical("covariance observations must be finite".into()));
        }
        let next_n = self
            .n
            .checked_add(1)
            .ok_or_else(|| StateError::Numerical("covariance count overflow".into()))?;
        if self.n == 0 {
            self.mean.copy_from_slice(x);
            self.n = next_n;
            return Ok(());
        }
        // Same rank-one update this type has always applied, now shared with the
        // kernel crate. It is not `Welford::push`: the two evaluation orders can
        // differ in the last bit of a diagonal entry. Accessors expose that state.
        antecedent_kernels::update_scatter(&mut self.mean, &mut self.m2, self.n, x);
        self.n = next_n;
        Ok(())
    }

    /// Merge `other` into this scatter with Chan's formula.
    ///
    /// Merging `A` then `B` matches merging `B` then `A`. Gram entries and means match a
    /// scatter rebuilt by appending the concatenated rows, within
    /// [`merge_within_tolerance`]. A centered second moment of a shared large offset
    /// can also differ by the rounding of that offset: both the one-pass update and
    /// this merge subtract numbers whose magnitude is the offset. The larger count is
    /// the base so that offset is not cancelled twice.
    ///
    /// # Errors
    ///
    /// Dimension mismatch, a non-finite summary, or a count that would overflow.
    #[must_use = "a refused covariance merge must be recomputed from the retained rows"]
    pub fn merge(&mut self, other: &Self) -> Result<(), StateError> {
        let merged = self.merged(other)?;
        *self = merged;
        Ok(())
    }

    pub(super) fn merged(&self, other: &Self) -> Result<Self, StateError> {
        if self.dim != other.dim {
            return Err(StateError::Shape(format!("cov dim {} != {}", self.dim, other.dim)));
        }
        if !self.entries_finite() || !other.entries_finite() {
            return Err(unsupported("covariance summary is non-finite"));
        }
        if other.n == 0 {
            return Ok(self.clone());
        }
        if self.n == 0 {
            let mut out = other.clone();
            out.retention = self.retention;
            return Ok(out);
        }
        let n = self
            .n
            .checked_add(other.n)
            .ok_or_else(|| StateError::Numerical("covariance count overflow".into()))?;
        let (base, add) = if chan_base_is_left(self.n, &self.mean, other.n, &other.mean) {
            (self, other)
        } else {
            (other, self)
        };
        let (mean, m2) = chan_merge_scatter(
            self.dim,
            (base.n as f64, &base.mean, &base.m2),
            (add.n as f64, &add.mean, &add.m2),
            n as f64,
        );
        let dim = self.dim;
        Ok(Self { dim, n, mean, m2, retention: self.retention })
    }

    fn entries_finite(&self) -> bool {
        self.mean.iter().all(|value| value.is_finite())
            && self.m2.iter().all(|value| value.is_finite())
    }

    /// Sample covariance matrix (`n − 1` denominator); `None` if `n < 2`.
    #[must_use]
    pub fn sample_covariance(&self) -> Option<Vec<f64>> {
        if self.n < 2 {
            return None;
        }
        let denom = (self.n - 1) as f64;
        Some(self.m2.iter().map(|v| v / denom).collect())
    }
}
