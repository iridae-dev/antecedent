//! Once-per-batch kernel dispatch (DESIGN.md §23.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::KernelPolicy;

use crate::view::{BitMaskView, F64VectorView};

/// Selected implementation class for a batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum KernelImpl {
    /// Scalar reference.
    Scalar,
    /// Portable optimized.
    PortableOptimized,
}

/// Resolve the implementation for a batch from policy.
#[must_use]
pub fn select_impl(policy: &KernelPolicy) -> KernelImpl {
    if policy.force_scalar || !policy.allow_portable_optimized {
        KernelImpl::Scalar
    } else if cfg!(feature = "portable-optimized") {
        KernelImpl::PortableOptimized
    } else {
        KernelImpl::Scalar
    }
}

/// Public semantic entry: masked sum.
#[must_use]
pub fn masked_sum(
    policy: &KernelPolicy,
    x: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> f64 {
    match select_impl(policy) {
        KernelImpl::Scalar => crate::scalar::masked_sum(x, mask),
        KernelImpl::PortableOptimized => crate::portable::masked_sum(x, mask),
    }
}

/// Public semantic entry: masked mean.
#[must_use]
pub fn masked_mean(
    policy: &KernelPolicy,
    x: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Option<f64> {
    match select_impl(policy) {
        KernelImpl::Scalar => crate::scalar::masked_mean(x, mask),
        KernelImpl::PortableOptimized => crate::portable::masked_mean(x, mask),
    }
}

/// Public semantic entry: masked population variance.
#[must_use]
pub fn masked_variance(
    policy: &KernelPolicy,
    x: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Option<f64> {
    match select_impl(policy) {
        KernelImpl::Scalar => crate::scalar::masked_variance(x, mask),
        KernelImpl::PortableOptimized => crate::portable::masked_variance(x, mask),
    }
}

/// Public semantic entry: gather.
pub fn gather(policy: &KernelPolicy, src: F64VectorView<'_>, indices: &[usize], out: &mut [f64]) {
    match select_impl(policy) {
        KernelImpl::Scalar => crate::scalar::gather(src, indices, out),
        KernelImpl::PortableOptimized => crate::portable::gather(src, indices, out),
    }
}

/// Public semantic entry: copy.
pub fn copy_vec(policy: &KernelPolicy, src: F64VectorView<'_>, dst: &mut [f64]) {
    match select_impl(policy) {
        KernelImpl::Scalar => crate::scalar::copy_vec(src, dst),
        KernelImpl::PortableOptimized => crate::portable::copy_vec(src, dst),
    }
}

/// Public semantic entry: partial correlation of `x` and `y` given `z_cols`.
#[must_use]
pub fn partial_correlation(
    policy: &KernelPolicy,
    x: &[f64],
    y: &[f64],
    z_cols: &[&[f64]],
    workspace: &mut crate::parcorr::ParCorrWorkspace,
) -> Option<f64> {
    match select_impl(policy) {
        KernelImpl::Scalar => crate::parcorr::partial_correlation_scalar(x, y, z_cols, workspace),
        KernelImpl::PortableOptimized => {
            crate::parcorr::partial_correlation_portable(x, y, z_cols, workspace)
        }
    }
}
