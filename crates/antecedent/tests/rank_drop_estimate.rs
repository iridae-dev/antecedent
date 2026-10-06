//! `estimate_with_rank_drop`: estimation on the span-preserving reduced adjustment set.
//!
//! Each numeric expectation comes from an independent oracle in this file: a hand-built
//! reduced study over a table written out column by column, and a normal-equations OLS that
//! shares no code with the library's fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "the reduced estimate must equal the hand-built one exactly: same columns, same fits"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "the normal-equations oracle indexes a small matrix by row and column on purpose"
)]

use antecedent::{
    CausalError, ColumnPriority, EstimatorId, PreflightInput, RankDropPolicy, RefuteSuite, Study,
    estimate_with_rank_drop,
};
use antecedent_core::{AverageEffectQuery, CausalRng, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::Dag;
use antecedent_kernels::standard_normal;

const N: usize = 600;

fn stream(seed: u64) -> CausalRng {
    ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 1)
}

fn named(columns: &[(&str, &Vec<f64>)]) -> TabularData {
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (*name, values.as_slice())).collect();
    TabularData::from_f64_columns(borrowed).unwrap()
}

fn id(data: &TabularData, name: &str) -> VariableId {
    data.schema().id_of(name).unwrap()
}

struct Fixture {
    t: Vec<f64>,
    y: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    /// Exact combination `2a - 3b + 1` of the others.
    c: Vec<f64>,
}

/// Binary treatment driven by `a`, outcome linear in `t`, `a`, `b` plus noise; `c` is an exact
/// linear function of `a` and `b`.
fn fixture(seed: u64) -> Fixture {
    let mut rng = stream(seed);
    let a: Vec<f64> = (0..N).map(|_| standard_normal(&mut rng)).collect();
    let b: Vec<f64> = (0..N).map(|_| standard_normal(&mut rng)).collect();
    let c: Vec<f64> = a.iter().zip(&b).map(|(x, z)| 2.0 * x - 3.0 * z + 1.0).collect();
    let t: Vec<f64> = a
        .iter()
        .map(|x| {
            let p = 1.0 / (1.0 + (-0.8 * x).exp());
            f64::from(rng.next_f64() < p)
        })
        .collect();
    let y: Vec<f64> =
        (0..N).map(|i| 1.5 * t[i] + a[i] + 0.5 * b[i] + 0.3 * standard_normal(&mut rng)).collect();
    Fixture { t, y, a, b, c }
}

/// Estimate through the ordinary study path on a table written out by hand with the declared
/// confounder graph over exactly `covariates`.
fn hand_estimate(f: &Fixture, covariates: &[(&str, &Vec<f64>)], estimator: EstimatorId) -> f64 {
    let mut columns: Vec<(&str, &Vec<f64>)> = vec![("t", &f.t), ("y", &f.y)];
    columns.extend_from_slice(covariates);
    let data = named(&columns);
    let mut edges = vec![("t", "y")];
    for (name, _) in covariates {
        edges.push((*name, "t"));
        edges.push((*name, "y"));
    }
    let graph = Dag::from_named_edges(data.schema(), &edges).unwrap();
    Study::tabular(data.clone())
        .graph(graph)
        .query(AverageEffectQuery::with_levels(id(&data, "t"), id(&data, "y"), 0.0, 1.0))
        .estimator(estimator)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap()
        .estimate
        .ate
}

/// Coefficient on the second column of `[1 | t | z...]` by Gaussian elimination on the normal
/// equations: an oracle that shares nothing with the library's QR.
fn ols_treatment_coefficient(f: &Fixture, covariates: &[&Vec<f64>]) -> f64 {
    let k = 2 + covariates.len();
    let column = |j: usize, i: usize| match j {
        0 => 1.0,
        1 => f.t[i],
        _ => covariates[j - 2][i],
    };
    let mut m = vec![vec![0.0; k + 1]; k];
    for r in 0..k {
        for s in 0..k {
            m[r][s] = (0..N).map(|i| column(r, i) * column(s, i)).sum();
        }
        m[r][k] = (0..N).map(|i| column(r, i) * f.y[i]).sum();
    }
    for p in 0..k {
        let pivot = (p..k).max_by(|&u, &v| m[u][p].abs().total_cmp(&m[v][p].abs())).unwrap();
        m.swap(p, pivot);
        for r in (p + 1)..k {
            let factor = m[r][p] / m[p][p];
            for s in p..=k {
                m[r][s] -= factor * m[p][s];
            }
        }
    }
    let mut beta = vec![0.0; k];
    for r in (0..k).rev() {
        let tail: f64 = ((r + 1)..k).map(|s| m[r][s] * beta[s]).sum();
        beta[r] = (m[r][k] - tail) / m[r][r];
    }
    beta[1]
}

fn policy(order: ColumnPriority) -> RankDropPolicy {
    RankDropPolicy { priority: order }
}

/// An exact collinearity: the full design cannot be fit, and the reduced estimate equals both
/// the estimate on a hand-built reduced design and the normal-equations OLS oracle.
#[test]
fn reduced_estimate_equals_the_hand_built_reduced_design() {
    let f = fixture(1);
    let data = named(&[("t", &f.t), ("y", &f.y), ("a", &f.a), ("b", &f.b), ("c", &f.c)]);
    let (ia, ib, ic) = (id(&data, "a"), id(&data, "b"), id(&data, "c"));
    let input = PreflightInput::binary_effect(
        &data,
        id(&data, "t"),
        id(&data, "y"),
        &[ia, ib, ic],
        0.0,
        1.0,
    );
    let ctx = ExecutionContext::for_tests(3);

    // The full design is rank deficient: the plain study refuses it.
    let full = {
        let graph = Dag::from_named_edges(
            data.schema(),
            &[("t", "y"), ("a", "t"), ("a", "y"), ("b", "t"), ("b", "y"), ("c", "t"), ("c", "y")],
        )
        .unwrap();
        Study::tabular(data.clone())
            .graph(graph)
            .query(AverageEffectQuery::with_levels(id(&data, "t"), id(&data, "y"), 0.0, 1.0))
            .estimator(EstimatorId::LinearAdjustmentAte)
            .bootstrap_replicates(0)
            .refute(RefuteSuite::None)
            .build()
            .and_then(|study| study.run(&ctx))
    };
    assert!(full.is_err(), "the rank-deficient design must not estimate");

    let result = estimate_with_rank_drop(
        &input,
        &policy(ColumnPriority::AdjustmentOrder),
        EstimatorId::LinearAdjustmentAte,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.plan.dropped.len(), 1);
    assert_eq!(result.plan.dropped[0].column, "c");
    assert_eq!(result.plan.original_adjustment, vec!["a", "b", "c"]);
    assert_eq!(result.plan.kept_adjustment, vec!["a", "b"]);
    assert_eq!(result.plan.design_identity, "adjustment=[a,b];dropped=[c];priority=[a,b,c]");
    assert_eq!(result.span_check.original_rank, 3);
    assert_eq!(result.span_check.retained_rank, 3);
    assert!(result.span_check.max_dropped_residual_ratio <= result.span_check.tolerance);
    assert_eq!(result.projection_invariance, "exact");

    let by_hand = hand_estimate(&f, &[("a", &f.a), ("b", &f.b)], EstimatorId::LinearAdjustmentAte);
    assert_eq!(result.ate, by_hand);
    let oracle = ols_treatment_coefficient(&f, &[&f.a, &f.b]);
    assert!((result.ate - oracle).abs() < 1e-8, "{} vs {oracle}", result.ate);
    // The truth is 1.5 and the sample is large enough to see it.
    assert!((result.ate - 1.5).abs() < 0.1, "{}", result.ate);
}

/// The declared priority changes which column drops and the design identity, not the estimate:
/// the retained spans coincide.
#[test]
fn priority_changes_the_dropped_column_and_identity_but_not_the_estimate() {
    let f = fixture(2);
    let data = named(&[("t", &f.t), ("y", &f.y), ("a", &f.a), ("b", &f.b), ("c", &f.c)]);
    let (ia, ib, ic) = (id(&data, "a"), id(&data, "b"), id(&data, "c"));
    let input = PreflightInput::binary_effect(
        &data,
        id(&data, "t"),
        id(&data, "y"),
        &[ia, ib, ic],
        0.0,
        1.0,
    );
    let ctx = ExecutionContext::for_tests(3);
    let run = |order: ColumnPriority, estimator| {
        estimate_with_rank_drop(&input, &policy(order), estimator, &ctx).unwrap()
    };
    let first = run(ColumnPriority::AdjustmentOrder, EstimatorId::LinearAdjustmentAte);
    let second = run(ColumnPriority::Declared(vec![ic, ia, ib]), EstimatorId::LinearAdjustmentAte);
    assert_eq!(first.plan.dropped[0].column, "c");
    assert_eq!(second.plan.dropped[0].column, "b");
    assert_eq!(second.plan.kept_adjustment, vec!["a", "c"]);
    assert_ne!(first.plan.design_identity, second.plan.design_identity);
    assert_eq!(second.plan.design_identity, "adjustment=[a,c];dropped=[b];priority=[c,a,b]");
    assert!((first.ate - second.ate).abs() < 1e-9, "{} vs {}", first.ate, second.ate);
    // Same inputs, same result.
    assert_eq!(first, run(ColumnPriority::AdjustmentOrder, EstimatorId::LinearAdjustmentAte));

    // The cross-fitted route fits an unpenalized logistic propensity: the same invariance
    // holds up to the IRLS convergence tolerance, and it equals the hand-built reduced design.
    let aipw_first = run(ColumnPriority::AdjustmentOrder, EstimatorId::Aipw);
    let aipw_second = run(ColumnPriority::Declared(vec![ic, ia, ib]), EstimatorId::Aipw);
    assert_eq!(aipw_first.projection_invariance, "unpenalized_logistic_no_separation");
    assert!((aipw_first.ate - aipw_second.ate).abs() < 1e-4, "{aipw_first:?} {aipw_second:?}");
    let by_hand = hand_estimate(&f, &[("a", &f.a), ("b", &f.b)], EstimatorId::Aipw);
    assert_eq!(aipw_first.ate, by_hand);
    assert!((aipw_first.ate - 1.5).abs() < 0.25, "{}", aipw_first.ate);
}

/// A drop that would remove a treatment alias is refused, and so is every estimator whose
/// nuisance fit is not shown invariant, a priority that does not cover the set, and a
/// cancelled context.
#[test]
fn refusals_leave_nothing_estimated() {
    let f = fixture(3);
    let ctx = ExecutionContext::for_tests(3);
    let alias = f.t.clone();
    let data = named(&[("t", &f.t), ("y", &f.y), ("a", &f.a), ("alias", &alias)]);
    let input = PreflightInput::binary_effect(
        &data,
        id(&data, "t"),
        id(&data, "y"),
        &[id(&data, "a"), id(&data, "alias")],
        0.0,
        1.0,
    );
    let refusal = estimate_with_rank_drop(
        &input,
        &policy(ColumnPriority::AdjustmentOrder),
        EstimatorId::LinearAdjustmentAte,
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal.reason_code(), Some("rank_drop_not_licensed"));
    assert!(refusal.refusal_fields().unwrap().implicated_columns.contains(&"alias".to_string()));

    let data = named(&[("t", &f.t), ("y", &f.y), ("a", &f.a), ("b", &f.b), ("c", &f.c)]);
    let (ia, ib, ic) = (id(&data, "a"), id(&data, "b"), id(&data, "c"));
    let input = PreflightInput::binary_effect(
        &data,
        id(&data, "t"),
        id(&data, "y"),
        &[ia, ib, ic],
        0.0,
        1.0,
    );
    let order = policy(ColumnPriority::AdjustmentOrder);

    let refusal = estimate_with_rank_drop(&input, &order, EstimatorId::PropensityWeighting, &ctx)
        .unwrap_err();
    assert_eq!(refusal.reason_code(), Some("route_not_supported"));

    let partial = policy(ColumnPriority::Declared(vec![ia, ib]));
    let refusal = estimate_with_rank_drop(&input, &partial, EstimatorId::LinearAdjustmentAte, &ctx)
        .unwrap_err();
    assert_eq!(refusal.reason_code(), Some("rank_drop_not_licensed"));

    // An effect modifier the drop would remove is protected.
    let mut protected = input.clone();
    protected.protected = vec![ic];
    let refusal =
        estimate_with_rank_drop(&protected, &order, EstimatorId::LinearAdjustmentAte, &ctx)
            .unwrap_err();
    assert_eq!(refusal.reason_code(), Some("rank_drop_not_licensed"));

    ctx.cancellation.cancel();
    let cancelled = estimate_with_rank_drop(&input, &order, EstimatorId::LinearAdjustmentAte, &ctx)
        .unwrap_err();
    assert!(matches!(cancelled, CausalError::Cancelled { .. }), "{cancelled}");
}

/// A joint cell (several treatment columns) is outside this route and refuses typed.
#[test]
fn a_joint_cell_is_outside_the_route() {
    let f = fixture(4);
    let data = named(&[("t", &f.t), ("y", &f.y), ("a", &f.a), ("b", &f.b)]);
    let input = PreflightInput::joint_cell(
        &data,
        id(&data, "y"),
        &[id(&data, "a")],
        vec![(id(&data, "t"), 1.0), (id(&data, "b"), 0.0)],
    );
    let refusal = estimate_with_rank_drop(
        &input,
        &policy(ColumnPriority::AdjustmentOrder),
        EstimatorId::LinearAdjustmentAte,
        &ExecutionContext::for_tests(3),
    )
    .unwrap_err();
    assert_eq!(refusal.reason_code(), Some("route_not_supported"));
}

/// Treatment that is completely separated by a covariate: the cross-fitted route refuses
/// because a penalized refit could break the projection invariance, and the least-squares
/// route, which has no such refit, still estimates.
#[test]
fn a_separating_propensity_refuses_the_cross_fitted_route_only() {
    let mut f = fixture(5);
    f.t = f.a.iter().map(|x| f64::from(*x > 0.0)).collect();
    let data = named(&[("t", &f.t), ("y", &f.y), ("a", &f.a), ("b", &f.b), ("c", &f.c)]);
    let adjustment = [id(&data, "a"), id(&data, "b"), id(&data, "c")];
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let ctx = ExecutionContext::for_tests(3);
    let order = policy(ColumnPriority::AdjustmentOrder);
    let refusal = estimate_with_rank_drop(&input, &order, EstimatorId::Aipw, &ctx).unwrap_err();
    assert_eq!(refusal.reason_code(), Some("rank_drop_not_licensed"));
    let reason = refusal.refusal_fields().unwrap().reason.clone().unwrap();
    assert!(reason.starts_with("rank_drop_estimate.propensity_separates"), "{reason}");
    let linear =
        estimate_with_rank_drop(&input, &order, EstimatorId::LinearAdjustmentAte, &ctx).unwrap();
    assert!(linear.ate.is_finite());
}

/// The result carries its whole drop record on the wire form: original and reduced sets, the
/// dropped column with its exact relation, the identity and the span evidence. The identity is
/// recomputed here from the declared names.
#[test]
fn the_estimate_serializes_with_its_drop_record() {
    let f = fixture(6);
    let data = named(&[("t", &f.t), ("y", &f.y), ("a", &f.a), ("b", &f.b), ("c", &f.c)]);
    let adjustment = [id(&data, "a"), id(&data, "b"), id(&data, "c")];
    let input =
        PreflightInput::binary_effect(&data, id(&data, "t"), id(&data, "y"), &adjustment, 0.0, 1.0);
    let result = estimate_with_rank_drop(
        &input,
        &policy(ColumnPriority::AdjustmentOrder),
        EstimatorId::LinearAdjustmentAte,
        &ExecutionContext::for_tests(3),
    )
    .unwrap();
    let wire = serde_json::to_value(&result).unwrap();
    assert_eq!(wire["plan"]["original_adjustment"], serde_json::json!(["a", "b", "c"]));
    assert_eq!(wire["plan"]["kept_adjustment"], serde_json::json!(["a", "b"]));
    assert_eq!(wire["plan"]["dropped"][0]["column"], "c");
    // c = 2a - 3b + 1, so the relation names the intercept, a and b.
    let weight = |column: &str| {
        wire["plan"]["dropped"][0]["explained_by"]
            .as_array()
            .unwrap()
            .iter()
            .find(|w| w["column"] == column)
            .map_or(f64::NAN, |w| w["coefficient"].as_f64().unwrap())
    };
    assert!((weight("(intercept)") - 1.0).abs() < 1e-8);
    assert!((weight("a") - 2.0).abs() < 1e-8);
    assert!((weight("b") + 3.0).abs() < 1e-8);
    let expected_identity = format!(
        "adjustment=[{}];dropped=[{}];priority=[{}]",
        ["a", "b"].join(","),
        "c",
        ["a", "b", "c"].join(",")
    );
    assert_eq!(wire["plan"]["design_identity"], expected_identity.as_str());
    assert_eq!(wire["span_check"]["original_rank"], 3);
    assert_eq!(wire["estimator"], "linear.adjustment.ate");
    assert_eq!(wire["ate"].as_f64().unwrap(), result.ate);
    assert!(wire["note"].as_str().unwrap().contains("no interval"));
    // Serialization is deterministic.
    assert_eq!(wire, serde_json::to_value(&result).unwrap());
}
