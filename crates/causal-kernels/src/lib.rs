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
pub mod parcorr;
pub mod portable;
pub mod posterior_reduce;
pub mod rng;
pub mod scalar;
pub mod special;
pub mod view;

pub use dispatch::{
    KernelImpl, accumulate_contingency, accumulate_contingency_rows, arch_simd_available, copy_vec,
    gather, masked_covariance, masked_mean, masked_sum, masked_variance, pairwise_l1_fill,
    partial_correlation, select_impl, standardize_inplace, weighted_dot, weighted_mean,
    weighted_sum,
};
pub use posterior_reduce::{PosteriorReduceOp, reduce_posterior_draws};
pub use scalar::sanitize_weight;
pub use parcorr::{ParCorrMode, ParCorrQuery, ParCorrWorkspace, partial_correlation_batch, pearson};
pub use rng::{
    categorical_from_u, fill_standard_normal, sample_categorical, shuffle, standard_normal,
    standard_normal_pair, unbiased_index,
};
pub use special::{erf, erfc, norm_cdf, norm_pdf};
pub use view::{BitMaskView, F64MatrixView, F64VectorView, ViewError};

#[cfg(test)]
mod tests {
    use causal_core::{KernelPolicy, ToleranceClass};

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
        assert!(ToleranceClass::StableFloat.close(s_sum, p_sum));

        let s_mean = scalar::masked_mean(x, Some(mask)).unwrap();
        let p_mean = portable::masked_mean(x, Some(mask)).unwrap();
        assert!(ToleranceClass::StableFloat.close(s_mean, p_mean));

        let s_var = scalar::masked_variance(x, Some(mask)).unwrap();
        let p_var = portable::masked_variance(x, Some(mask)).unwrap();
        assert!(ToleranceClass::StableFloat.close(s_var, p_var));
    }

    #[test]
    fn scalar_and_portable_agree_on_covariance() {
        let x_data = sample_data();
        let y_data: Vec<f64> = x_data.iter().map(|v| v * 0.3 + 1.0).collect();
        let x = F64VectorView::contiguous(&x_data);
        let y = F64VectorView::contiguous(&y_data);
        let mask_bytes = vec![0b1111_0000u8; x_data.len().div_ceil(8)];
        let mask = BitMaskView::new(&mask_bytes, x_data.len()).unwrap();

        let s = scalar::masked_covariance(x, y, Some(mask)).unwrap();
        let p = portable::masked_covariance(x, y, Some(mask)).unwrap();
        assert!(ToleranceClass::StableFloat.close(s, p));

        let s_full = scalar::masked_covariance(x, y, None).unwrap();
        let p_full = portable::masked_covariance(x, y, None).unwrap();
        assert!(ToleranceClass::StableFloat.close(s_full, p_full));
    }

    #[test]
    fn scalar_and_portable_agree_on_standardize() {
        let mut a = sample_data();
        let mut b = a.clone();
        let (ms, ss) = scalar::standardize_inplace(&mut a, 1e-12);
        let (mp, sp) = portable::standardize_inplace(&mut b, 1e-12);
        assert!(ToleranceClass::StableFloat.close(ms, mp));
        assert!(ToleranceClass::StableFloat.close(ss, sp));
        for (sa, sb) in a.iter().zip(b.iter()) {
            assert!(ToleranceClass::StableFloat.close(*sa, *sb));
        }
    }

    #[test]
    fn scalar_and_portable_agree_on_pairwise_l1() {
        let x = sample_data();
        let n = x.len();
        let mut out_s = vec![0.0; n * n];
        let mut out_p = vec![0.0; n * n];
        scalar::pairwise_l1_fill(&x, &mut out_s);
        portable::pairwise_l1_fill(&x, &mut out_p);
        assert_eq!(out_s, out_p);
    }

    #[test]
    fn scalar_and_portable_agree_on_contingency() {
        let x_codes: Vec<u32> = (0..64).map(|i| (i % 4) as u32).collect();
        let y_codes: Vec<u32> = (0..64).map(|i| (i % 3) as u32).collect();
        let mut out_s = vec![0.0; 4 * 3];
        let mut out_p = vec![0.0; 4 * 3];
        scalar::accumulate_contingency(&x_codes, &y_codes, &mut out_s, 3);
        portable::accumulate_contingency(&x_codes, &y_codes, &mut out_p, 3);
        assert_eq!(out_s, out_p);
    }

    #[test]
    fn scalar_and_portable_agree_on_weighted() {
        let x = sample_data();
        let y: Vec<f64> = x.iter().map(|v| v + 1.0).collect();
        let w: Vec<f64> = (0..x.len()).map(|i| if i % 5 == 0 { 0.0 } else { 1.0 + (i as f64) * 0.01 }).collect();
        let s_sum = scalar::weighted_sum(&x, &w);
        let p_sum = portable::weighted_sum(&x, &w);
        assert!(ToleranceClass::StableFloat.close(s_sum, p_sum));
        let s_mean = scalar::weighted_mean(&x, &w).unwrap();
        let p_mean = portable::weighted_mean(&x, &w).unwrap();
        assert!(ToleranceClass::StableFloat.close(s_mean, p_mean));
        let s_dot = scalar::weighted_dot(&x, &y, &w);
        let p_dot = portable::weighted_dot(&x, &y, &w);
        assert!(ToleranceClass::StableFloat.close(s_dot, p_dot));
    }

    #[test]
    fn sanitize_weight_drops_nonfinite_and_negative() {
        assert_eq!(sanitize_weight(1.5), 1.5);
        assert_eq!(sanitize_weight(0.0), 0.0);
        assert_eq!(sanitize_weight(-2.0), 0.0);
        assert_eq!(sanitize_weight(f64::NAN), 0.0);
        assert_eq!(sanitize_weight(f64::INFINITY), 0.0);
        assert_eq!(sanitize_weight(f64::NEG_INFINITY), 0.0);
        let x = [1.0, 2.0, 3.0];
        let w = [1.0, f64::NAN, -1.0];
        assert!(ToleranceClass::StableFloat.close(scalar::weighted_mean(&x, &w).unwrap(), 1.0));
        assert!(ToleranceClass::StableFloat.close(portable::weighted_mean(&x, &w).unwrap(), 1.0));
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
        // Without simd-runtime, default selects portable even when allow_arch_simd is true.
        assert_eq!(select_impl(&default), KernelImpl::PortableOptimized);
        assert!(!arch_simd_available());
    }

    #[test]
    fn dispatch_honors_allow_arch_simd_false() {
        let mut policy = KernelPolicy::default_policy();
        policy.allow_arch_simd = false;
        assert_eq!(select_impl(&policy), KernelImpl::PortableOptimized);
        policy.allow_portable_optimized = false;
        assert_eq!(select_impl(&policy), KernelImpl::Scalar);
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

        let y: Vec<f64> = data.iter().map(|v| v * 2.0).collect();
        let yv = F64VectorView::contiguous(&y);
        let cov = masked_covariance(&policy, x, yv, None).unwrap();
        let cov_s = scalar::masked_covariance(x, yv, None).unwrap();
        assert!(ToleranceClass::StableFloat.close(cov, cov_s));
    }

    /// DESIGN §28.2: scalar ≡ portable under random lengths, strides/tails, masks, NaNs.
    #[test]
    fn property_scalar_portable_random_strides_masks_nans() {
        let mut rng = causal_core::CausalRng::from_seed(28_02);
        for trial in 0..80 {
            let len = 1 + (rng.next_u64() as usize % 48); // 1..=48
            let stride = 1 + (rng.next_u64() as usize % 4); // 1..=4
            let tail = rng.next_u64() as usize % 5; // unused padding
            let need = (len - 1) * stride + 1 + tail;
            let mut x_buf = vec![0.0f64; need];
            let mut y_buf = vec![0.0f64; need];
            for i in 0..need {
                // Mix finite values with occasional NaNs (inject into logical slots).
                let u = rng.next_u64();
                let base = ((u % 1000) as f64) * 0.01 - 5.0;
                x_buf[i] = if u % 17 == 0 { f64::NAN } else { base };
                y_buf[i] = if (u / 17) % 19 == 0 {
                    f64::NAN
                } else {
                    base * 0.3 + 1.0
                };
            }
            let x = F64VectorView::strided(&x_buf, len, stride).unwrap();
            let y = F64VectorView::strided(&y_buf, len, stride).unwrap();

            let mut bits = vec![0u8; len.div_ceil(8)];
            let use_mask = rng.next_u64() % 3 != 0;
            if use_mask {
                for i in 0..len {
                    if rng.next_u64() & 1 == 1 {
                        bits[i / 8] |= 1 << (i % 8);
                    }
                }
            }
            // Ensure at least one valid bit when masked so mean/var/cov have a chance.
            if use_mask && bits.iter().all(|&b| b == 0) {
                bits[0] |= 1;
            }
            let mask = if use_mask {
                Some(BitMaskView::new(&bits, len).unwrap())
            } else {
                None
            };

            let s_sum = scalar::masked_sum(x, mask);
            let p_sum = portable::masked_sum(x, mask);
            assert!(
                floats_agree(s_sum, p_sum),
                "trial={trial} sum scalar={s_sum} portable={p_sum} len={len} stride={stride}"
            );

            let s_mean = scalar::masked_mean(x, mask);
            let p_mean = portable::masked_mean(x, mask);
            assert!(
                options_agree(s_mean, p_mean),
                "trial={trial} mean scalar={s_mean:?} portable={p_mean:?}"
            );

            let s_var = scalar::masked_variance(x, mask);
            let p_var = portable::masked_variance(x, mask);
            assert!(
                options_agree(s_var, p_var),
                "trial={trial} var scalar={s_var:?} portable={p_var:?}"
            );

            let s_cov = scalar::masked_covariance(x, y, mask);
            let p_cov = portable::masked_covariance(x, y, mask);
            assert!(
                options_agree(s_cov, p_cov),
                "trial={trial} cov scalar={s_cov:?} portable={p_cov:?}"
            );
        }
    }

    fn floats_agree(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || ToleranceClass::StableFloat.close(a, b)
    }

    fn options_agree(a: Option<f64>, b: Option<f64>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => floats_agree(x, y),
            _ => false,
        }
    }

    #[test]
    fn gather_hot_path_reuses_output_buffer() {
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

    #[test]
    fn pairwise_and_contingency_reuse_output_buffers() {
        let policy = KernelPolicy::default_policy();
        let x: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let mut out = vec![0.0; 64 * 64];
        let ptr = out.as_ptr();
        let cap = out.capacity();
        for _ in 0..50 {
            pairwise_l1_fill(&policy, &x, &mut out);
            assert_eq!(out.as_ptr(), ptr);
            assert_eq!(out.capacity(), cap);
        }

        let xc: Vec<u32> = (0..256).map(|i| (i % 5) as u32).collect();
        let yc: Vec<u32> = (0..256).map(|i| (i % 4) as u32).collect();
        let mut table = vec![0.0; 5 * 4];
        let tptr = table.as_ptr();
        let tcap = table.capacity();
        for _ in 0..50 {
            table.fill(0.0);
            accumulate_contingency(&policy, &xc, &yc, &mut table, 4);
            assert_eq!(table.as_ptr(), tptr);
            assert_eq!(table.capacity(), tcap);
        }
    }
}
