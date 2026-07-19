//! Posterior draw reductions.
//!
//! One semantic entry point for scalar/portable mean / variance / quantile over
//! a contiguous draw column. Dispatch once per batch via [`KernelPolicy`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use causal_core::KernelPolicy;

use crate::dispatch::{KernelImpl, select_impl};
use crate::portable;
use crate::scalar;
use crate::view::F64VectorView;

/// Reduce a contiguous posterior-draw column.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PosteriorReduceOp {
    /// Arithmetic mean.
    Mean,
    /// Population variance (÷n).
    Variance,
    /// Sample standard deviation (÷(n−1), 0 if n<2).
    Std,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
}

/// Apply [`PosteriorReduceOp`] to a draw column under `policy`.
///
/// Deterministic for Mean/Variance/Std/Min/Max (no RNG). Empty input → `None`.
#[must_use]
pub fn reduce_posterior_draws(
    draws: &[f64],
    op: PosteriorReduceOp,
    policy: &KernelPolicy,
) -> Option<f64> {
    if draws.is_empty() {
        return None;
    }
    let view = F64VectorView::contiguous(draws);
    match select_impl(policy) {
        KernelImpl::Scalar => reduce_scalar(view, op),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => reduce_portable(view, op),
    }
}

fn reduce_scalar(view: F64VectorView<'_>, op: PosteriorReduceOp) -> Option<f64> {
    match op {
        PosteriorReduceOp::Mean => scalar::masked_mean(view, None),
        PosteriorReduceOp::Variance => scalar::masked_variance(view, None),
        PosteriorReduceOp::Std => {
            let v = scalar::masked_variance(view, None)?;
            // Convert population var to sample sd when n≥2.
            let n = view.len() as f64;
            if n < 2.0 {
                return Some(0.0);
            }
            Some((v * n / (n - 1.0)).sqrt())
        }
        PosteriorReduceOp::Min => {
            let mut m = f64::INFINITY;
            for i in 0..view.len() {
                m = m.min(view.get(i).unwrap_or(f64::NAN));
            }
            Some(m)
        }
        PosteriorReduceOp::Max => {
            let mut m = f64::NEG_INFINITY;
            for i in 0..view.len() {
                m = m.max(view.get(i).unwrap_or(f64::NAN));
            }
            Some(m)
        }
    }
}

fn reduce_portable(view: F64VectorView<'_>, op: PosteriorReduceOp) -> Option<f64> {
    match op {
        PosteriorReduceOp::Mean => portable::masked_mean(view, None),
        PosteriorReduceOp::Variance => portable::masked_variance(view, None),
        PosteriorReduceOp::Std => {
            let v = portable::masked_variance(view, None)?;
            let n = view.len() as f64;
            if n < 2.0 {
                return Some(0.0);
            }
            Some((v * n / (n - 1.0)).sqrt())
        }
        PosteriorReduceOp::Min | PosteriorReduceOp::Max => reduce_scalar(view, op),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::KernelPolicy;

    #[test]
    fn mean_matches_hand() {
        let d = [1.0, 2.0, 3.0, 4.0];
        let m = reduce_posterior_draws(&d, PosteriorReduceOp::Mean, &KernelPolicy::scalar_only())
            .unwrap();
        assert!((m - 2.5).abs() < 1e-12);
    }
}
