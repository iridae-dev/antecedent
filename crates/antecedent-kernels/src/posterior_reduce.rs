//! Posterior draw reductions.
//!
//! One semantic entry point for scalar/portable mean / variance / quantile over
//! a contiguous draw column. Dispatch once per batch via [`KernelPolicy`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use antecedent_core::KernelPolicy;

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
    /// Sample standard deviation (÷(n−1)); undefined (`None`) for fewer than two draws.
    Std,
    /// Minimum. NaN if any draw is NaN.
    Min,
    /// Maximum. NaN if any draw is NaN.
    Max,
}

/// Apply [`PosteriorReduceOp`] to a draw column under `policy`.
///
/// Deterministic for Mean/Variance/Std/Min/Max (no RNG). Empty input → `None`.
/// A single draw carries no spread information, so `Std` of one draw is `None`,
/// not zero. A NaN draw makes every reduction NaN, Min and Max included, so a
/// failed draw is never silently dropped from an extreme.
#[must_use]
pub fn reduce_posterior_draws(
    draws: &[f64],
    op: PosteriorReduceOp,
    policy: &KernelPolicy,
) -> Option<f64> {
    if draws.is_empty() {
        return None;
    }
    match op {
        PosteriorReduceOp::Min => return Some(extreme(draws, f64::min)),
        PosteriorReduceOp::Max => return Some(extreme(draws, f64::max)),
        _ => {}
    }
    let view = F64VectorView::contiguous(draws);
    match select_impl(policy) {
        KernelImpl::Scalar => reduce_scalar(view, op),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => reduce_portable(view, op),
    }
}

/// Fold with `pick`, returning NaN as soon as any draw is NaN (`f64::min/max` would skip it).
fn extreme(draws: &[f64], pick: fn(f64, f64) -> f64) -> f64 {
    let mut acc = draws[0];
    for &v in draws {
        if v.is_nan() {
            return f64::NAN;
        }
        acc = pick(acc, v);
    }
    acc
}

fn sample_sd(population_variance: f64, n: usize) -> Option<f64> {
    if n < 2 {
        return None;
    }
    let n = n as f64;
    Some((population_variance * n / (n - 1.0)).sqrt())
}

fn reduce_scalar(view: F64VectorView<'_>, op: PosteriorReduceOp) -> Option<f64> {
    match op {
        PosteriorReduceOp::Mean => scalar::masked_mean(view, None),
        PosteriorReduceOp::Variance => scalar::masked_variance(view, None),
        PosteriorReduceOp::Std => sample_sd(scalar::masked_variance(view, None)?, view.len()),
        PosteriorReduceOp::Min | PosteriorReduceOp::Max => {
            unreachable!("extremes are reduced on the raw slice before dispatch")
        }
    }
}

fn reduce_portable(view: F64VectorView<'_>, op: PosteriorReduceOp) -> Option<f64> {
    match op {
        PosteriorReduceOp::Mean => portable::masked_mean(view, None),
        PosteriorReduceOp::Variance => portable::masked_variance(view, None),
        PosteriorReduceOp::Std => sample_sd(portable::masked_variance(view, None)?, view.len()),
        PosteriorReduceOp::Min | PosteriorReduceOp::Max => reduce_scalar(view, op),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::KernelPolicy;

    #[test]
    fn mean_matches_hand() {
        let d = [1.0, 2.0, 3.0, 4.0];
        let m = reduce_posterior_draws(&d, PosteriorReduceOp::Mean, &KernelPolicy::scalar_only())
            .unwrap();
        assert!((m - 2.5).abs() < 1e-12);
    }

    fn reduce(draws: &[f64], op: PosteriorReduceOp) -> Option<f64> {
        reduce_posterior_draws(draws, op, &KernelPolicy::scalar_only())
    }

    #[test]
    fn nan_draws_propagate_through_min_and_max() {
        let d = [3.0, f64::NAN, 1.0];
        assert!(reduce(&d, PosteriorReduceOp::Min).unwrap().is_nan());
        assert!(reduce(&d, PosteriorReduceOp::Max).unwrap().is_nan());
        assert!(reduce(&[f64::NAN], PosteriorReduceOp::Min).unwrap().is_nan());
        assert_eq!(reduce(&[3.0, -1.0, 2.0], PosteriorReduceOp::Min), Some(-1.0));
        assert_eq!(reduce(&[3.0, -1.0, 2.0], PosteriorReduceOp::Max), Some(3.0));
    }

    #[test]
    fn std_of_one_draw_is_undefined_not_zero() {
        assert_eq!(reduce(&[5.0], PosteriorReduceOp::Std), None);
        let sd = reduce(&[1.0, 3.0], PosteriorReduceOp::Std).unwrap();
        // Sample sd of {1, 3}: sqrt(((1-2)^2 + (3-2)^2) / 1) = sqrt(2).
        assert!((sd - 2.0_f64.sqrt()).abs() < 1e-12);
        let policy = KernelPolicy::default_policy();
        let portable =
            reduce_posterior_draws(&[1.0, 3.0], PosteriorReduceOp::Std, &policy).unwrap();
        assert!((portable - 2.0_f64.sqrt()).abs() < 1e-12);
    }
}
