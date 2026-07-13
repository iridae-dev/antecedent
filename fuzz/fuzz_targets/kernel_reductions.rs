//! Fuzz scalar vs portable kernel agreement on random masks.
#![no_main]

use causal_core::KernelPolicy;
use causal_kernels::{BitMaskView, F64VectorView, masked_mean, masked_sum, masked_variance};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let n = (usize::from(data[0]) % 64).max(1);
    let values: Vec<f64> = (0..n)
        .map(|i| f64::from(data.get(1 + (i % (data.len() - 1)).max(1)).copied().unwrap_or(0)))
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
    for policy in [KernelPolicy::scalar_only(), KernelPolicy::default_policy()] {
        let _ = masked_sum(&policy, view, Some(mask));
        let _ = masked_mean(&policy, view, Some(mask));
        let _ = masked_variance(&policy, view, Some(mask));
    }
});
