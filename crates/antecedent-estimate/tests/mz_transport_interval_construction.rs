//! Independent construction check of the multi-source mz joint percentile bootstrap.
//!
//! The interval the joint bootstrap reports for a request is rebuilt in the test without the
//! interval code: the law sets of every replicate come from the public draw inspector, each is
//! pushed through the exact point evaluator, and the equal-tailed type-7 (linear
//! interpolation) percentiles of the resulting risks are computed by hand. The reported
//! mean interval and the contrast interval must equal those hand values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_core::{ExecutionContext, Value};
use antecedent_estimate::{
    evaluate_exact_mz_transport, mz_transport_bootstrap_interval, mz_transport_bootstrap_law_draws,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use antecedent_identify::{
    BoundMzTransportFunctional, MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision,
    bind_mz_transport_catalog, decide_mz_transport,
};
use common::mz_fixture::{X, Y, empirical, evidence, graph, query, sources, with_studies};
use common::z_scm::{risk_of, vid};

fn request(x: bool) -> Assignment {
    Assignment::from_pairs([(vid(X), Value::Bool(x))])
}

/// Type-7 quantile of an unsorted sample, written out from the definition
/// `h = (n - 1) p`, `x_(floor h) + (h - floor h)(x_(floor h + 1) - x_(floor h))`.
fn hand_quantile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let h = (sorted.len() as f64 - 1.0) * p;
    let k = h.floor() as usize;
    if k + 1 >= sorted.len() {
        return sorted[sorted.len() - 1];
    }
    sorted[k] + (h - h.floor()) * (sorted[k + 1] - sorted[k])
}

fn hand_interval(values: &[f64], level: f64) -> (f64, f64) {
    (hand_quantile(values, (1.0 - level) / 2.0), hand_quantile(values, (1.0 + level) / 2.0))
}

#[test]
fn reported_bootstrap_intervals_equal_hand_percentiles_of_the_replicate_point_estimates() {
    let (catalog, exact) = evidence();
    let catalog = with_studies(&catalog);
    let data = empirical(&exact, 400.0);
    let q = query(sources());
    let ctx = ExecutionContext::for_tests(11);
    let MzTransportDecision::Identified { derivation, .. } =
        decide_mz_transport(&graph(), &q, &catalog, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx).unwrap()
    else {
        panic!("the complementary catalog identifies");
    };
    let functional: BoundMzTransportFunctional =
        bind_mz_transport_catalog(&graph(), &derivation, &catalog).unwrap();
    let limits = ExactEvaluationLimits::default();
    let replicates = 99;
    let level = 0.95;

    let reported = mz_transport_bootstrap_interval(
        &functional,
        &data,
        &[request(false), request(true)],
        limits,
        replicates,
        level,
        &ctx,
    )
    .unwrap()
    .expect("declared independent studies publish a joint interval");
    assert!((reported.coverage_target - level).abs() < 1e-15);

    // Independent rebuild: replicate law sets -> exact point evaluator -> hand percentiles.
    let draws = mz_transport_bootstrap_law_draws(&functional, &data, replicates, &ctx)
        .unwrap()
        .expect("the draws exist");
    let mut risks = [Vec::new(), Vec::new()];
    for draw in draws.into_iter().flatten() {
        let mut row = [0.0; 2];
        let mut ok = true;
        for (k, x) in [false, true].into_iter().enumerate() {
            match evaluate_exact_mz_transport(&functional, draw.clone(), request(x), limits, &ctx) {
                Ok(result) => row[k] = risk_of(&result),
                Err(_) => ok = false,
            }
        }
        if ok {
            risks[0].push(row[0]);
            risks[1].push(row[1]);
        }
    }
    assert_eq!(risks[0].len() as u32, reported.replicates_ok);
    for (k, values) in risks.iter().enumerate() {
        let (lower, upper) = hand_interval(values, level);
        let &(outcome, got_lower, got_upper) = &reported.requests[k].mean_intervals[0];
        assert_eq!(outcome, vid(Y));
        assert!((got_lower - lower).abs() < 1e-12, "request {k} lower: {got_lower} vs {lower}");
        assert!((got_upper - upper).abs() < 1e-12, "request {k} upper: {got_upper} vs {upper}");
    }
    let differences: Vec<f64> = risks[1].iter().zip(&risks[0]).map(|(b, a)| b - a).collect();
    let (lower, upper) = hand_interval(&differences, level);
    let &(k, outcome, got_lower, got_upper) = &reported.contrasts[0];
    assert_eq!((k, outcome), (1, vid(Y)));
    assert!((got_lower - lower).abs() < 1e-12 && (got_upper - upper).abs() < 1e-12);
}
