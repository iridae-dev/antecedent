//! The sequential contrast uncertainty against exact references.
//!
//! `oracle_data/sequential_contrast_reference.py` (frozen in the JSON beside it) takes ten
//! per-subject active and control scores, forms the contrast scores, gives the Wald interval of
//! their mean (the standard error from the exact sum of squares, the normal quantile in 40-digit
//! arithmetic) at four coverages, and evaluates the exact Dirichlet-weighted law (a B-spline law
//! for distinct scores) of the active value, the control value and their shared-weight contrast.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::score_posterior::ScorePosteriorOptions;
use antecedent_estimate::sequential_aipw_inference::{
    bayesian_sequential_contrast, sequential_contrast_wald_interval,
};
use antecedent_estimate::sequential_crossfit::SequentialContrast;
use serde_json::Value;

const DRAWS: usize = 200_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn contrast(data: &Value) -> SequentialContrast {
    let (active, control) = (floats(&data["active_scores"]), floats(&data["control_scores"]));
    let contrast_scores: Vec<f64> = active.iter().zip(&control).map(|(a, c)| a - c).collect();
    let n = active.len() as f64;
    SequentialContrast {
        contrast: contrast_scores.iter().sum::<f64>() / n,
        active_mean: active.iter().sum::<f64>() / n,
        control_mean: control.iter().sum::<f64>() / n,
        unmeasured_score_scale: 0.0,
        evaluated_rows: active.len() as u64,
        minimum_action_probability: 0.5,
        active_scores: active,
        control_scores: control,
        contrast_scores,
        fits_computed: 0,
        fits_reused: 0,
    }
}

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/sequential_contrast_reference.json")).unwrap()
}

#[test]
fn the_wald_interval_matches_the_exact_standard_error_and_normal_quantile() {
    let data = reference();
    let input = contrast(&data);
    for want in data["wald"].as_array().unwrap() {
        let coverage = want["coverage"].as_f64().unwrap();
        let got = sequential_contrast_wald_interval(&input, coverage).unwrap();
        let rel = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
        assert!(rel(got.contrast, data["centre"].as_f64().unwrap()));
        assert!(rel(got.standard_error, data["standard_error"].as_f64().unwrap()));
        assert!(rel(got.interval.0, want["lower"].as_f64().unwrap()), "{coverage}");
        assert!(rel(got.interval.1, want["upper"].as_f64().unwrap()), "{coverage}");
    }
}

#[test]
fn the_posterior_laws_of_both_regimes_and_the_shared_weight_contrast_match_the_exact_laws() {
    let data = reference();
    let input = contrast(&data);
    let fit = bayesian_sequential_contrast(
        &input,
        ScorePosteriorOptions { draws: DRAWS, coverage: 0.95 },
        &ExecutionContext::for_tests(31),
    )
    .unwrap();
    for (name, posterior) in
        [("active", &fit.active), ("control", &fit.control), ("contrast", &fit.contrast)]
    {
        let want = &data[name];
        assert!((posterior.score_mean - want["mean"].as_f64().unwrap()).abs() < 1e-12, "{name}");
        for (x, f) in floats(&want["grid"]).into_iter().zip(floats(&want["cdf"])) {
            let got = posterior.draws.iter().filter(|d| **d <= x).count() as f64 / DRAWS as f64;
            let se = (f * (1.0 - f) / DRAWS as f64).sqrt();
            assert!((got - f).abs() < 5.0 * se + 1e-9, "{name} cdf at {x}: {got} vs {f}");
        }
    }
    for k in 0..200 {
        let paired = fit.active.draws[k] - fit.control.draws[k];
        assert!((fit.contrast.draws[k] - paired).abs() < 1e-9);
    }
}
