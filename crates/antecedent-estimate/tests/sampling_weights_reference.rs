//! Sampling-weight normalisation, zero-weight accounting and Kish size against exact rationals.
//!
//! `oracle_data/sampling_weights_reference.py` (frozen in the JSON beside it) normalises each
//! retained weight to `s_i k / sum(s)`, counts the rows of weight zero that a mask would have
//! kept, and forms the Kish size, all in `Fraction` arithmetic. The weight vectors include
//! scales of 1e308 (the sum overflows binary64) and 5e-320 (the reciprocal of the sum overflows)
//! and a spread of 1e300 between the largest and smallest weight.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::sampling_weights::SamplingWeights;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn normalised_weights_zero_counts_and_kish_size_match_the_exact_reference() {
    let file: Value =
        serde_json::from_str(include_str!("oracle_data/sampling_weights_reference.json")).unwrap();
    for case in file["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut mask: Vec<bool> =
            case["mask"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
        let kept = SamplingWeights::new(floats(&case["weights"]))
            .unwrap()
            .restrict(&mut mask)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(kept.record.rows_used, case["rows_used"].as_u64().unwrap(), "{name}");
        assert_eq!(
            kept.record.zero_weight_rows,
            case["zero_weight_rows"].as_u64().unwrap(),
            "{name}"
        );
        let want = floats(&case["normalized"]);
        assert_eq!(kept.weights.len(), want.len(), "{name}");
        for (i, (g, w)) in kept.weights.iter().zip(&want).enumerate() {
            // Binary64 denormals carry only a few bits, so the 5e-320 scale keeps 1e-3.
            let rel = if name == "scale_5e_320" { 1e-3 } else { 1e-12 };
            assert!((g - w).abs() <= rel * w, "{name} weight {i}: got {g} want {w}");
        }
        let ess = case["ess"].as_f64().unwrap();
        let rel = if name == "scale_5e_320" { 1e-3 } else { 1e-12 };
        assert!((kept.record.effective_sample_size - ess).abs() <= rel * ess, "{name} ess");
        let mean = kept.weights.iter().sum::<f64>() / kept.weights.len() as f64;
        assert!((mean - 1.0).abs() < 1e-12, "{name}: normalised mean {mean}");
    }
}
