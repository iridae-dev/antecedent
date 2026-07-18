//! Fuzz scalar vs portable kernel agreement on random masks / strides / NaNs.
#![no_main]

use causal_core::ToleranceClass;
use causal_kernels::{BitMaskView, F64VectorView, portable, scalar};
use libfuzzer_sys::fuzz_target;

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

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let n = (usize::from(data[0]) % 64).max(1);
    let stride = usize::from(data[1] % 4) + 1;
    let need = (n - 1).saturating_mul(stride).saturating_add(1);
    let mut values = vec![0.0f64; need];
    let mut values_y = vec![0.0f64; need];
    for i in 0..need {
        let b = data.get(1 + (i % (data.len() - 1)).max(1)).copied().unwrap_or(0);
        values[i] = if b & 0x80 != 0 { f64::NAN } else { f64::from(b) };
        let b2 = data.get(2 + (i % (data.len() - 1)).max(1)).copied().unwrap_or(1);
        values_y[i] = if b2 & 0x80 != 0 {
            f64::NAN
        } else {
            f64::from(b2) * 0.5
        };
    }
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
    let Ok(view) = F64VectorView::strided(&values, n, stride) else {
        return;
    };
    let Ok(view_y) = F64VectorView::strided(&values_y, n, stride) else {
        return;
    };
    let n_pair = n.min(16);
    let mut pair_out_s = vec![0.0; n_pair * n_pair];
    let mut pair_out_p = vec![0.0; n_pair * n_pair];
    let xc: Vec<u32> = (0..n).map(|i| u32::from(data[i % data.len()]) % 4).collect();
    let yc: Vec<u32> = (0..n).map(|i| u32::from(data[(i + 1) % data.len()]) % 3).collect();
    let mut table_s = vec![0.0; 4 * 3];
    let mut table_p = vec![0.0; 4 * 3];

    let s_sum = scalar::masked_sum(view, Some(mask));
    let p_sum = portable::masked_sum(view, Some(mask));
    assert!(floats_agree(s_sum, p_sum), "masked_sum {s_sum} vs {p_sum}");

    let s_mean = scalar::masked_mean(view, Some(mask));
    let p_mean = portable::masked_mean(view, Some(mask));
    assert!(options_agree(s_mean, p_mean), "masked_mean {s_mean:?} vs {p_mean:?}");

    let s_var = scalar::masked_variance(view, Some(mask));
    let p_var = portable::masked_variance(view, Some(mask));
    assert!(options_agree(s_var, p_var), "masked_variance {s_var:?} vs {p_var:?}");

    let s_cov = scalar::masked_covariance(view, view_y, Some(mask));
    let p_cov = portable::masked_covariance(view, view_y, Some(mask));
    assert!(options_agree(s_cov, p_cov), "masked_covariance {s_cov:?} vs {p_cov:?}");

    let x_contig: Vec<f64> = (0..n).map(|i| view.get(i).unwrap_or(0.0)).collect();
    let s_w = scalar::weighted_sum(&x_contig, &weights);
    let p_w = portable::weighted_sum(&x_contig, &weights);
    assert!(floats_agree(s_w, p_w), "weighted_sum {s_w} vs {p_w}");

    let mut std_s = x_contig.clone();
    let mut std_p = x_contig;
    let (ms, ss) = scalar::standardize_inplace(&mut std_s, 1e-12);
    let (mp, sp) = portable::standardize_inplace(&mut std_p, 1e-12);
    assert!(floats_agree(ms, mp) && floats_agree(ss, sp), "standardize moments");
    for (a, b) in std_s.iter().zip(std_p.iter()) {
        assert!(floats_agree(*a, *b), "standardize element");
    }

    let pair_x: Vec<f64> = (0..n_pair).map(|i| view.get(i).unwrap_or(0.0)).collect();
    scalar::pairwise_l1_fill(&pair_x, &mut pair_out_s);
    portable::pairwise_l1_fill(&pair_x, &mut pair_out_p);
    assert_eq!(pair_out_s, pair_out_p, "pairwise_l1");

    scalar::accumulate_contingency(&xc, &yc, &mut table_s, 3);
    portable::accumulate_contingency(&xc, &yc, &mut table_p, 3);
    assert_eq!(table_s, table_p, "contingency");
});
