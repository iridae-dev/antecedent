//! Independent interior multinomial information oracle; no coverage measurement.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_core::ExecutionContext;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, InferenceStanding, NestedMarkovInput, Regime, RegimeCounts,
    refuse_public_interval_route,
};
use antecedent_estimate::nested_markov_uncertainty::nested_markov_fisher_internal;

fn input(n: f64) -> NestedMarkovInput {
    // Independent generating model: X1 and X2 are fair; P(M=0|X2)=.75/.25;
    // P(Y=0|M)=.8/.4. Sum/product the structural factors directly, not via pilot.
    let mut cells = Vec::new();
    for _x1 in 0..2 {
        for x2 in 0..2 {
            for m in 0..2 {
                for y in 0..2 {
                    let p_m0 = if x2 == 0 { 0.75 } else { 0.25 };
                    let p_y0 = if m == 0 { 0.8 } else { 0.4 };
                    let p_m = if m == 0 { p_m0 } else { 1. - p_m0 };
                    let p_y = if y == 0 { p_y0 } else { 1. - p_y0 };
                    cells.push((n * 0.25 * p_m * p_y).round());
                }
            }
        }
    }
    NestedMarkovInput {
        graph: AdmgDeclaration::selected(),
        regimes: vec![RegimeCounts { regime: Regime::Observational, levels: vec![2; 4], cells }],
    }
}

#[test]
fn nested_fisher_delta_covariance_matches_independent_stratified_variance() {
    let oracle: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/transport/nested_fisher/expected.json"
    ))
    .unwrap();
    let n = oracle["sample_size"].as_f64().unwrap();
    let estimate = nested_markov_fisher_internal(
        &input(n),
        &FitOptions::default(),
        0.95,
        &ExecutionContext::for_tests(17),
    )
    .unwrap();
    assert_eq!(estimate.sample_count.to_bits(), n.to_bits());
    assert!((estimate.pilot.comparison.model_contrast - 0.2).abs() < 1e-12);
    // c0/c1 variance=.375/N each. Their scores are orthogonal to the Q block.
    // q40 is the half-weighted mean of Y across X2 inside M=0: variance
    // .25*.16/(N*.5*.75)+.25*.16/(N*.5*.25)=(32/75)/N.
    // q41 similarly has variance .64/N; their covariance is zero at this law.
    // Delta contrast=(c0-c1)(q40-q41): .4²*(.375+.375)+.5²*(32/75+.64).
    let variance = (29. / 75.) / n;
    assert!(
        (estimate.effect_covariance[2][2] - variance).abs() < 1e-12,
        "{} vs {variance}",
        estimate.effect_covariance[2][2]
    );
    let cov = estimate.effect_covariance;
    // Independent covariance of the two do-means: c-uncertainty contributes .06/N
    // to each variance; the shared q4 block contributes .28/N and (29/75)/N.
    // Their covariance is (.75*.25)*(32/75+.64)/N=.2/N.
    let expected = [
        [0.34 / n, 0.2 / n, -0.14 / n],
        [0.2 / n, (67. / 150.) / n, (37. / 150.) / n],
        [-0.14 / n, (37. / 150.) / n, (29. / 75.) / n],
    ];
    for (i, (row, expected)) in cov.iter().zip(expected).enumerate() {
        for (j, (value, truth)) in row.iter().zip(expected).enumerate() {
            let pinned = oracle["covariance_times_n"][i][j].as_f64().unwrap() / n;
            assert!((truth - pinned).abs() < 1e-14);
            assert!((value - pinned).abs() < 1e-12);
        }
    }
    assert!((cov[2][2] - (cov[0][0] + cov[1][1] - 2. * cov[0][1])).abs() < 1e-14);
    assert!(cov[0][1].abs() > 1e-8, "shared parameters imply nonzero mean covariance");
    assert!(estimate.inverse_residual < 1e-10);
    let radius = 1.959_963_984_540_054 * variance.sqrt();
    assert!((estimate.interval_candidates[2][0] - (0.2 - radius)).abs() < 1e-10);
    assert!((estimate.interval_candidates[2][1] - (0.2 + radius)).abs() < 1e-10);
    assert_eq!(
        estimate.pilot.status.inference,
        InferenceStanding::IntervalWithheldCalibrationUnmeasured
    );
    assert!(format!("{}", refuse_public_interval_route()).contains("cell_not_licensed"));
}

#[test]
fn nested_fisher_information_scales_with_actual_sample_size() {
    let ctx = ExecutionContext::for_tests(18);
    let a =
        nested_markov_fisher_internal(&input(16_000.), &FitOptions::default(), 0.9, &ctx).unwrap();
    let b =
        nested_markov_fisher_internal(&input(32_000.), &FitOptions::default(), 0.9, &ctx).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((a.effect_covariance[i][j] - 2. * b.effect_covariance[i][j]).abs() < 1e-14);
        }
    }
    assert_eq!(a.pilot.comparison, b.pilot.comparison);
}

#[test]
fn nested_fisher_refuses_pseudocounts_sparse_cells_invalid_scope_and_cancellation() {
    let ctx = ExecutionContext::for_tests(19);
    let mut fractional = input(16_000.);
    fractional.regimes[0].cells[0] += 0.5;
    let err =
        nested_markov_fisher_internal(&fractional, &FitOptions::default(), 0.95, &ctx).unwrap_err();
    assert!(err.to_string().contains("nested_markov.fisher_sampling_design"));
    let mut sparse = input(16_000.);
    sparse.regimes[0].cells[0] = 0.;
    assert!(nested_markov_fisher_internal(&sparse, &FitOptions::default(), 0.95, &ctx).is_err());
    for level in [0., 1., 0.80, 0.99, f64::NAN] {
        assert!(
            nested_markov_fisher_internal(&input(16_000.), &FitOptions::default(), level, &ctx)
                .unwrap_err()
                .to_string()
                .contains("invalid_level")
        );
    }
    let mut graph = input(16_000.);
    graph.graph.bidirected.clear();
    assert!(
        nested_markov_fisher_internal(&graph, &FitOptions::default(), 0.95, &ctx)
            .unwrap_err()
            .to_string()
            .contains("outside_binary_pilot")
    );
    ctx.cancellation.cancel();
    assert!(
        nested_markov_fisher_internal(&input(16_000.), &FitOptions::default(), 0.95, &ctx)
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
}
