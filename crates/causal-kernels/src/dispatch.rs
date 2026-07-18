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
    /// Architecture-specific SIMD (requires `simd-runtime` feature + CPU support).
    ArchSimd,
}

/// Whether an arch-SIMD path is compiled in and available on this CPU.
///
/// Always `false` until a justified `simd-runtime` kernel lands (DESIGN.md §23.2).
/// `KernelPolicy::allow_arch_simd` is still consulted by [`select_impl`]; when this
/// returns false, selection falls through to portable/scalar.
#[must_use]
pub fn arch_simd_available() -> bool {
    false
}

/// Resolve the implementation for a batch from policy.
///
/// Selection order (DESIGN.md §23.2 / §30):
/// 1. `force_scalar`, or neither portable nor arch allowed → [`KernelImpl::Scalar`]
/// 2. `allow_arch_simd` and compiled `simd-runtime` and CPU support → [`KernelImpl::ArchSimd`]
/// 3. `allow_portable_optimized` → [`KernelImpl::PortableOptimized`]
/// 4. else → [`KernelImpl::Scalar`]
#[must_use]
pub fn select_impl(policy: &KernelPolicy) -> KernelImpl {
    if policy.force_scalar {
        return KernelImpl::Scalar;
    }
    let arch_ok = policy.allow_arch_simd && arch_simd_available();
    let portable_ok = policy.allow_portable_optimized;
    if !arch_ok && !portable_ok {
        return KernelImpl::Scalar;
    }
    if arch_ok {
        return KernelImpl::ArchSimd;
    }
    if portable_ok {
        KernelImpl::PortableOptimized
    } else {
        KernelImpl::Scalar
    }
}

fn portable_or_scalar_reductions(policy: &KernelPolicy) -> KernelImpl {
    match select_impl(policy) {
        KernelImpl::ArchSimd => KernelImpl::PortableOptimized,
        other => other,
    }
}

/// Public semantic entry: masked sum.
#[must_use]
pub fn masked_sum(
    policy: &KernelPolicy,
    x: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> f64 {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::masked_sum(x, mask),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => crate::portable::masked_sum(x, mask),
    }
}

/// Public semantic entry: masked mean.
#[must_use]
pub fn masked_mean(
    policy: &KernelPolicy,
    x: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Option<f64> {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::masked_mean(x, mask),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::masked_mean(x, mask)
        }
    }
}

/// Public semantic entry: masked population variance.
#[must_use]
pub fn masked_variance(
    policy: &KernelPolicy,
    x: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Option<f64> {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::masked_variance(x, mask),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::masked_variance(x, mask)
        }
    }
}

/// Public semantic entry: gather.
pub fn gather(policy: &KernelPolicy, src: F64VectorView<'_>, indices: &[usize], out: &mut [f64]) {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::gather(src, indices, out),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::gather(src, indices, out)
        }
    }
}

/// Public semantic entry: copy.
pub fn copy_vec(policy: &KernelPolicy, src: F64VectorView<'_>, dst: &mut [f64]) {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::copy_vec(src, dst),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => crate::portable::copy_vec(src, dst),
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
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::parcorr::partial_correlation_scalar(x, y, z_cols, workspace),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::parcorr::partial_correlation_portable(x, y, z_cols, workspace)
        }
    }
}

/// Public semantic entry: masked population covariance.
///
/// Contract: deterministic reduction; `StableFloat` tolerance; beneficial for `n ≳ 64`.
#[must_use]
pub fn masked_covariance(
    policy: &KernelPolicy,
    x: F64VectorView<'_>,
    y: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Option<f64> {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::masked_covariance(x, y, mask),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::masked_covariance(x, y, mask)
        }
    }
}

/// Public semantic entry: in-place standardization (sample SD).
///
/// Contract: `StableFloat`; beneficial for `n ≳ 32`.
#[must_use]
pub fn standardize_inplace(policy: &KernelPolicy, x: &mut [f64], eps: f64) -> (f64, f64) {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::standardize_inplace(x, eps),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::standardize_inplace(x, eps)
        }
    }
}

/// Public semantic entry: pairwise L1 distance matrix fill.
///
/// Contract: exact for finite inputs; beneficial for `n ≳ 64`.
pub fn pairwise_l1_fill(policy: &KernelPolicy, x: &[f64], out: &mut [f64]) {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::pairwise_l1_fill(x, out),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::pairwise_l1_fill(x, out)
        }
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
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => {
            crate::scalar::accumulate_contingency(x_codes, y_codes, out, n_y_levels)
        }
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::accumulate_contingency(x_codes, y_codes, out, n_y_levels)
        }
    }
}

/// Public semantic entry: contingency accumulation over a row subset.
pub fn accumulate_contingency_rows(
    policy: &KernelPolicy,
    x_codes: &[u32],
    y_codes: &[u32],
    rows: &[usize],
    out: &mut [f64],
    n_y_levels: usize,
) {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => {
            crate::scalar::accumulate_contingency_rows(x_codes, y_codes, rows, out, n_y_levels)
        }
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::accumulate_contingency_rows(x_codes, y_codes, rows, out, n_y_levels)
        }
    }
}

/// Public semantic entry: weighted sum (bootstrap weight accumulation).
///
/// Contract: non-finite/negative weights → 0; `StableFloat`; beneficial for `n ≳ 64`.
#[must_use]
pub fn weighted_sum(policy: &KernelPolicy, x: &[f64], weights: &[f64]) -> f64 {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::weighted_sum(x, weights),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::weighted_sum(x, weights)
        }
    }
}

/// Public semantic entry: weighted mean.
#[must_use]
pub fn weighted_mean(policy: &KernelPolicy, x: &[f64], weights: &[f64]) -> Option<f64> {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::weighted_mean(x, weights),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::weighted_mean(x, weights)
        }
    }
}

/// Public semantic entry: weighted dot product.
#[must_use]
pub fn weighted_dot(policy: &KernelPolicy, x: &[f64], y: &[f64], weights: &[f64]) -> f64 {
    match portable_or_scalar_reductions(policy) {
        KernelImpl::Scalar => crate::scalar::weighted_dot(x, y, weights),
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => {
            crate::portable::weighted_dot(x, y, weights)
        }
    }
}
