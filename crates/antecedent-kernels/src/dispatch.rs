//! Once-per-batch kernel dispatch.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::KernelPolicy;

use crate::view::F64VectorView;

/// Selected implementation class for a batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum KernelImpl {
    /// Scalar reference.
    Scalar,
    /// Portable optimized.
    PortableOptimized,
}

/// Resolve the implementation for a batch from policy.
///
/// `force_scalar`, or portable-optimized disallowed → [`KernelImpl::Scalar`];
/// otherwise [`KernelImpl::PortableOptimized`]. `allow_arch_simd` is part of the
/// persisted policy but no architecture-specific kernel is compiled in, so it
/// never changes the selection.
#[must_use]
pub fn select_impl(policy: &KernelPolicy) -> KernelImpl {
    if policy.force_scalar || !policy.allow_portable_optimized {
        KernelImpl::Scalar
    } else {
        KernelImpl::PortableOptimized
    }
}

/// Public semantic entry: gather.
/// # Errors
///
/// Same contract as [`crate::scalar::gather`].
pub fn gather(
    policy: &KernelPolicy,
    src: F64VectorView<'_>,
    indices: &[usize],
    out: &mut [f64],
) -> Result<(), crate::view::ViewError> {
    match select_impl(policy) {
        KernelImpl::Scalar => crate::scalar::gather(src, indices, out),
        KernelImpl::PortableOptimized => crate::portable::gather(src, indices, out),
    }
}

/// Public semantic entry: in-place standardization (sample SD).
///
/// Contract: `StableFloat`; beneficial for `n ≳ 32`.
#[must_use]
pub fn standardize_inplace(policy: &KernelPolicy, x: &mut [f64], eps: f64) -> (f64, f64) {
    match select_impl(policy) {
        KernelImpl::Scalar => crate::scalar::standardize_inplace(x, eps),
        KernelImpl::PortableOptimized => crate::portable::standardize_inplace(x, eps),
    }
}

/// Public semantic entry: contingency table accumulation.
///
/// Contract: exact integer counts as `f64`; beneficial for `n ≳ 256`.
pub fn accumulate_contingency(
    policy: &KernelPolicy,
    x_codes: &[u32],
    y_codes: &[u32],
    out: &mut [f64],
    n_y_levels: usize,
) {
    match select_impl(policy) {
        KernelImpl::Scalar => {
            crate::scalar::accumulate_contingency(x_codes, y_codes, out, n_y_levels);
        }
        KernelImpl::PortableOptimized => {
            crate::portable::accumulate_contingency(x_codes, y_codes, out, n_y_levels);
        }
    }
}

/// Public semantic entry: weighted mean.
#[must_use]
pub fn weighted_mean(policy: &KernelPolicy, x: &[f64], weights: &[f64]) -> Option<f64> {
    match select_impl(policy) {
        KernelImpl::Scalar => crate::scalar::weighted_mean(x, weights),
        KernelImpl::PortableOptimized => crate::portable::weighted_mean(x, weights),
    }
}
