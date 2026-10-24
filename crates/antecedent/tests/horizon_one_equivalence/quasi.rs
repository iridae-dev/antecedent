//! Discontinuity designs at horizon one.
//!
//! A discontinuity in calendar time with the running variable equal to time reuses the static
//! local-polynomial fit row for row, so the point and the independent-rows standard error are
//! the static ones and only the interval is replaced by a moving-block bootstrap. The sharp
//! regression-discontinuity average effect answers the same jump with a different local fit,
//! and the difference is pinned from the closed-form uniform-window fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent::IdentifierId;
use antecedent_core::{LocalPolynomialRatioQuery, LocalRatioOptions};

const TIME_ROWS: usize = 2000;

/// `time = 1..=N`, treatment switching on at 1000, `y = 0.001 t + d + e`.
fn time_discontinuity() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let time: Vec<f64> = (1..=TIME_ROWS).map(|t| t as f64).collect();
    let treated: Vec<f64> = time.iter().map(|t| if *t >= 1000.0 { 1.0 } else { 0.0 }).collect();
    let outcome: Vec<f64> = time
        .iter()
        .zip(&treated)
        .enumerate()
        .map(|(i, (t, d))| 0.001 * t + d + 0.5 * disturbance(i))
        .collect();
    (time, treated, outcome)
}

fn ratio_query(
    outcome: u32,
    treatment: u32,
    running: u32,
    cutoff: f64,
    bandwidth: f64,
    replicates: Option<u32>,
) -> LocalPolynomialRatioQuery {
    LocalPolynomialRatioQuery {
        outcome: id(outcome),
        treatment: id(treatment),
        running: id(running),
        cutoff,
        bandwidth,
        kink: false,
        options: replicates.map(|temporal_replicates| LocalRatioOptions {
            temporal_replicates: Some(temporal_replicates),
            ..LocalRatioOptions::default()
        }),
    }
}

/// outcome: equivalent point, selected rows and independent-rows standard error; the interval is
/// `not_equivalent` (moving-block bootstrap).
///
/// The calendar-time design is the static local-polynomial ratio on the same rows. Its published
/// effect is the static effect, its window counts are the static counts, and the independent-rows
/// delta-method standard error is kept as a named diagnostic. The route differs only in how the
/// published uncertainty is formed: a within-side moving-block bootstrap refits the whole ratio,
/// so the block standard error is a different object from the HC0 one.
#[test]
fn horizon_one_time_discontinuity_local_ratio_matches_the_static_fit_with_block_uncertainty() {
    let (time, treated, outcome) = time_discontinuity();
    let data = table(&[("time", &time), ("treatment", &treated), ("outcome", &outcome)]);

    let static_result = Study::tabular(data.clone())
        .query(ratio_query(2, 1, 0, 1000.0, 300.0, None))
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    let fixed = static_result.local_polynomial_ratio().expect("the static local ratio payload");
    assert!((fixed.effect - 1.0).abs() < 0.6, "the fixture must carry the jump: {}", fixed.effect);
    assert!(fixed.standard_error > 0.0);

    let temporal_result = Study::tabular(data)
        .query(ratio_query(2, 1, 0, 1000.0, 300.0, Some(199)))
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    assert_eq!(temporal_result.logical_plan.plan_id.as_ref(), "quasi.temporal_fuzzy_rd");
    let rollout = temporal_result.quasi_rollout().expect("the temporal rollout payload");
    let overall = rollout.overall().expect("the scalar summary");

    // Same fit: the point, the window rows and the independent-rows delta-method SE.
    assert!(
        (overall.estimate - fixed.effect).abs() < 1e-9,
        "{} vs {}",
        overall.estimate,
        fixed.effect
    );
    assert!(
        (rollout.diagnostic("hc0_ratio_standard_error").unwrap() - fixed.standard_error).abs()
            < 1e-12
    );
    assert_eq!(rollout.diagnostic("rows_left").unwrap() as usize, fixed.n_left);
    assert_eq!(rollout.diagnostic("rows_right").unwrap() as usize, fixed.n_right);

    // The published interval is a block-bootstrap object, not the HC0 one.
    let block_se = overall.standard_error.expect("a block-bootstrap standard error");
    assert!(block_se.is_finite() && block_se > 0.0);
    assert!(
        (block_se - fixed.standard_error).abs() > 1e-9,
        "{block_se} vs {}",
        fixed.standard_error
    );
    assert!(rollout.diagnostic("block_length").is_some(), "the block length is recorded");
    assert!(!rollout.licensed);
}

const RD_ROWS: usize = 2000;

/// `r` equispaced on `(-1.5, 1.5)`, `t = 1{r >= 0}`, `y = 3 t + 0.5 r + r^3 + noise`.
fn sharp_rd_rows() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let running: Vec<f64> =
        (0..RD_ROWS).map(|i| -1.5 + 3.0 * (i as f64 + 0.5) / RD_ROWS as f64).collect();
    let treated: Vec<f64> = running.iter().map(|r| if *r >= 0.0 { 1.0 } else { 0.0 }).collect();
    let outcome: Vec<f64> = running
        .iter()
        .zip(&treated)
        .enumerate()
        .map(|(i, (r, t))| 3.0 * t + 0.5 * r + r.powi(3) + 0.1 * disturbance(i))
        .collect();
    (running, treated, outcome)
}

/// outcome: equivalent target (the jump at the cutoff); `not_equivalent` estimator.
///
/// The sharp regression discontinuity is a uniform-window local-linear fit with a conventional
/// HC1 standard error. On a mean with cubic curvature its jump is biased by the difference of the
/// two sides' linear-fit intercept errors (`-0.4` here), which the closed-form window regression
/// reproduces. The temporal local ratio is a triangular-kernel local quadratic with a local
/// cubic bias correction, so it recovers the jump on the same rows, and its published uncertainty
/// is the moving-block bootstrap. The two answer one question with estimators that cannot agree
/// under curvature.
#[test]
fn horizon_one_sharp_rd_uniform_window_fit_is_biased_where_the_temporal_local_ratio_recovers_the_jump(
) {
    let (running, treated, outcome) = sharp_rd_rows();
    let bandwidth = 1.0;
    let data = table(&[("t", &treated), ("y", &outcome), ("r", &running)]);

    // The static route: sharp RD average effect at the cutoff.
    let static_result = run(frequentist(
        Study::tabular(data.clone())
            .graph(dag(3, &[(2, 0), (0, 1), (2, 1)]))
            .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(id(0), id(1))))
            .identifier(IdentifierId::RdSharp)
            .estimator(EstimatorId::RdSharp)
            .rd_config(id(2), 0.0, bandwidth),
    ));
    let static_effect = static_result.estimate.as_effect().unwrap();

    // Closed form: OLS of y on [1, T, R, T R] over the rows inside the window.
    let inside: Vec<usize> = (0..RD_ROWS).filter(|&i| running[i].abs() <= bandwidth).collect();
    let pick = |values: &[f64]| -> Vec<f64> { inside.iter().map(|&i| values[i]).collect() };
    let (t_window, r_window, y_window) = (pick(&treated), pick(&running), pick(&outcome));
    let interaction: Vec<f64> = t_window.iter().zip(&r_window).map(|(t, r)| t * r).collect();
    let window_jump = ols(&[&t_window, &r_window, &interaction], &y_window)[1];
    assert!(
        (static_effect.ate - window_jump).abs() < 1e-8,
        "{} vs {window_jump}",
        static_effect.ate
    );
    assert!(static_effect.se_analytic().is_finite() && static_effect.se_analytic() > 0.0);
    assert!(
        (static_effect.ate - 3.0).abs() > 0.3,
        "curvature must bias the uniform fit: {}",
        static_effect.ate
    );
    assert!((static_effect.ate - 2.6).abs() < 0.05, "{}", static_effect.ate);

    // The temporal counterpart: the local ratio of the same columns, with block uncertainty.
    let temporal = Study::tabular(data)
        .query(ratio_query(1, 0, 2, 0.0, bandwidth, Some(199)))
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    let rollout = temporal.quasi_rollout().expect("the temporal rollout payload");
    let recovered = rollout.overall().unwrap().estimate;
    assert!((recovered - 3.0).abs() < 0.05, "{recovered}");
    assert!((recovered - static_effect.ate).abs() > 0.3);
}
