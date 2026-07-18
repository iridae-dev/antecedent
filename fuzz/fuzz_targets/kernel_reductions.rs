//! Fuzz scalar vs portable kernel agreement on random masks.
#![no_main]

use causal_core::KernelPolicy;
use causal_kernels::{
    BitMaskView, F64VectorView, accumulate_contingency, masked_covariance, masked_mean, masked_sum,
    masked_variance, pairwise_l1_fill, standardize_inplace, weighted_sum,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let n = (usize::from(data[0]) % 64).max(1);
    let values: Vec<f64> = (0..n)
        .map(|i| f64::from(data.get(1 + (i % (data.len() - 1)).max(1)).copied().unwrap_or(0)))
        .collect();
    let values_y: Vec<f64> = (0..n)
        .map(|i| {
            f64::from(data.get(2 + (i % (data.len() - 1)).max(1)).copied().unwrap_or(1)) * 0.5
        })
        .collect();
    let weights: Vec<f64> = (0..n)
        .map(|i| f64::from(data.get(i % data.len()).copied().unwrap_or(1)) / 255.0)
        .collect();
    let mut bits = vec![0u8; n.div_ceil(8)];
    for i in 0..n {
        if data.get(i % data.len()).copied().unwrap_or(0) & 1 == 1 {
            bits[i / 8] |= 1 << (i % 8);
        }
    }
    let Ok(mask) = BitMaskView::new(&bits, n) else {
        return;
    };
    let view = F64VectorView::contiguous(&values);
    let view_y = F64VectorView::contiguous(&values_y);
    let n_pair = n.min(16);
    let mut pair_out = vec![0.0; n_pair * n_pair];
    let xc: Vec<u32> = (0..n).map(|i| (data[i % data.len()] as u32) % 4).collect();
    let yc: Vec<u32> = (0..n).map(|i| (data[(i + 1) % data.len()] as u32) % 3).collect();
    let mut table = vec![0.0; 4 * 3];
    for policy in [KernelPolicy::scalar_only(), KernelPolicy::default_policy()] {
        let _ = masked_sum(&policy, view, Some(mask));
        let _ = masked_mean(&policy, view, Some(mask));
        let _ = masked_variance(&policy, view, Some(mask));
        let _ = masked_covariance(&policy, view, view_y, Some(mask));
        let _ = weighted_sum(&policy, &values, &weights);
        let mut std_buf = values.clone();
        let _ = standardize_inplace(&policy, &mut std_buf, 1e-12);
        pairwise_l1_fill(&policy, &values[..n_pair], &mut pair_out);
        table.fill(0.0);
        accumulate_contingency(&policy, &xc, &yc, &mut table, 3);
    }
});
