//! Low-level borrowed views and numerical kernels for causal-library.
//!
//! Scalar kernels are the correctness reference. Portable-optimized and
//! architecture-specific paths must pass the same differential tests
//! (DESIGN.md §23.2, ADR 0011).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(missing_docs)]
// Unsafe is confined to validated unchecked view indexing in hot loops.
#![allow(unsafe_code)]
#![cfg_attr(test, allow(clippy::cast_precision_loss))]

pub mod dispatch;
pub mod portable;
pub mod scalar;
pub mod view;

pub use dispatch::{
    KernelImpl, copy_vec, gather, masked_mean, masked_sum, masked_variance, select_impl,
};
pub use view::{BitMaskView, F64MatrixView, F64VectorView, ViewError};

#[cfg(test)]
mod tests {
    use causal_core::KernelPolicy;

    use super::*;

    fn sample_data() -> Vec<f64> {
        (0..128).map(|i| f64::from(i) * 0.5 - 10.0).collect()
    }

    #[test]
    fn scalar_and_portable_agree_on_reductions() {
        let data = sample_data();
        let x = F64VectorView::contiguous(&data);
        let mask_bytes = vec![0b1010_1011u8; data.len().div_ceil(8)];
        let mask = BitMaskView::new(&mask_bytes, data.len()).unwrap();

        let s_sum = scalar::masked_sum(x, Some(mask));
        let p_sum = portable::masked_sum(x, Some(mask));
        assert!((s_sum - p_sum).abs() <= 1e-12 * (1.0 + s_sum.abs()));

        let s_mean = scalar::masked_mean(x, Some(mask)).unwrap();
        let p_mean = portable::masked_mean(x, Some(mask)).unwrap();
        assert!((s_mean - p_mean).abs() <= 1e-12 * (1.0 + s_mean.abs()));

        let s_var = scalar::masked_variance(x, Some(mask)).unwrap();
        let p_var = portable::masked_variance(x, Some(mask)).unwrap();
        assert!((s_var - p_var).abs() <= 1e-12 * (1.0 + s_var.abs()));
    }

    #[test]
    fn gather_differential_contiguous_and_strided() {
        let data = sample_data();
        let contiguous = F64VectorView::contiguous(&data);
        let strided = F64VectorView::strided(&data, 32, 2).unwrap();
        let indices: Vec<usize> = (0..32).collect();
        let mut out_s = vec![0.0; 32];
        let mut out_p = vec![0.0; 32];
        scalar::gather(contiguous, &indices, &mut out_s);
        portable::gather(contiguous, &indices, &mut out_p);
        assert_eq!(out_s, out_p);

        let mut out_strided = vec![0.0; 16];
        let idx: Vec<usize> = (0..16).collect();
        scalar::gather(strided, &idx, &mut out_strided);
        let mut out_strided_p = vec![0.0; 16];
        portable::gather(strided, &idx, &mut out_strided_p);
        assert_eq!(out_strided, out_strided_p);
    }

    #[test]
    fn dispatch_respects_force_scalar() {
        let policy = KernelPolicy::scalar_only();
        assert_eq!(select_impl(&policy), KernelImpl::Scalar);
        let default = KernelPolicy::default_policy();
        assert_eq!(select_impl(&default), KernelImpl::PortableOptimized);
    }

    #[test]
    fn dispatch_entries_match_scalar_under_force() {
        let data = sample_data();
        let x = F64VectorView::contiguous(&data);
        let policy = KernelPolicy::scalar_only();
        let mut out = vec![0.0; 8];
        let idx = [0usize, 1, 2, 3, 4, 5, 6, 7];
        gather(&policy, x, &idx, &mut out);
        let mut expected = vec![0.0; 8];
        scalar::gather(x, &idx, &mut expected);
        assert_eq!(out, expected);
    }

    #[test]
    fn phase0_gather_hot_path_reuses_output_buffer() {
        let n = 8_000usize;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let src = F64VectorView::contiguous(&data);
        let indices: Vec<usize> = (0..n).step_by(8).collect();
        let mut out = vec![0.0; indices.len()];
        let policy = KernelPolicy::default_policy();
        let ptr = out.as_ptr();
        let cap = out.capacity();
        for _ in 0..200 {
            gather(&policy, src, &indices, &mut out);
            assert_eq!(out.as_ptr(), ptr);
            assert_eq!(out.capacity(), cap);
        }
    }
}
