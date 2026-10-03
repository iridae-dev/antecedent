//! E3: configured batches and batch retarget (joint covariance, contrasts, closed
//! simultaneous interval, score-table lifetime, tidy export).
//!
//! The independent oracles here are written from the raw score tables with plain sums: the
//! weighted mean `θ = Σ w φ / Σ w`, the weighted influence value
//! `ξ_r = sqrt(n/(n-1)) (w_r/Σw)(φ_r − θ)` and the covariance `Σ_r ξ_kr ξ_lr`, never the
//! library's own helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "configured parity and order invariance are bit-for-bit statements"
)]

use antecedent::{
    BatchRetargetError, BatchRetargetReport, BatchRetargetRequest, BatchScores, BatchStudy,
    EstimatorId, PreparedBatch, RefuteSuite, RetargetClaim, RetargetContrast, ScoreSource, Study,
    StudyResult, TidyKind,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{AipwAte, OverlapPolicy, PropensityNuisance, RidgeTuning, ScoreTable};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const T1: u32 = 0;
const T2: u32 = 1;
const Y1: u32 = 2;
const Y2: u32 = 3;
const Z: u32 = 4;

struct Fixture {
    data: TabularData,
    graph: Dag,
    z: Vec<f64>,
}

/// Two binary treatments and two outcomes, all confounded by one `z`.
fn fixture(n: usize, seed: u64) -> Fixture {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xE3);
    let (mut t1, mut t2, mut y1, mut y2, mut z) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        let p1 = 1.0 / (1.0 + (-(-0.2 + 0.8 * zi)).exp());
        let p2 = 1.0 / (1.0 + (-(0.1 - 0.6 * zi)).exp());
        t1[i] = f64::from(rng.next_f64() < p1);
        t2[i] = f64::from(rng.next_f64() < p2);
        y1[i] = 2.0 * t1[i] + 0.5 * t2[i] + zi + 0.3 * standard_normal(&mut rng);
        y2[i] = -t1[i] + 1.5 * t2[i] - 0.5 * zi + 0.3 * standard_normal(&mut rng);
    }
    let pairs = [("t1", t1), ("t2", t2), ("y1", y1), ("y2", y2), ("z", z.clone())];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(5);
    for (from, to) in [(Z, T1), (Z, T2), (Z, Y1), (Z, Y2), (T1, Y1), (T1, Y2), (T2, Y1), (T2, Y2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    Fixture { data, graph, z }
}

fn ate(treatment: u32, outcome: u32) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(treatment), VariableId::from_raw(outcome))
}

fn z_id() -> VariableId {
    VariableId::from_raw(Z)
}

fn plain_batch(fx: &Fixture) -> BatchStudy {
    BatchStudy::new(fx.data.clone(), fx.graph.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

fn prepare(fx: &Fixture, queries: &[AverageEffectQuery], seed: u64) -> PreparedBatch {
    plain_batch(fx).prepare(queries, &ExecutionContext::for_tests(seed)).unwrap()
}

/// Target weights `exp(sign · 0.4 · z)` on the snapshot rows.
fn exp_weights(fx: &Fixture, rows: &[u32], sign: f64) -> Vec<f64> {
    rows.iter().map(|&r| (sign * 0.4 * fx.z[r as usize]).exp()).collect()
}

fn retarget(
    prepared: &PreparedBatch,
    scores: &BatchScores,
    request: &BatchRetargetRequest,
) -> Result<BatchRetargetReport, BatchRetargetError> {
    prepared.retarget(scores, request, &ExecutionContext::for_tests(1))
}

fn claim(name: &str, query_index: usize, weights: Vec<f64>) -> RetargetClaim {
    RetargetClaim { name: name.into(), query_index, weights, depends_on: vec![z_id()] }
}

fn contrast(name: &str, terms: &[(&str, f64)]) -> RetargetContrast {
    RetargetContrast {
        name: name.into(),
        coefficients: terms.iter().map(|(n, c)| ((*n).to_string(), *c)).collect(),
    }
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-10 * (1.0 + a.abs().max(b.abs())), "{what}: {a} vs {b}");
}

/// `(θ, ξ)` of the arm-1 minus arm-0 contrast score of `table` under `weights`.
fn oracle(table: &ScoreTable, weights: &[f64]) -> (f64, Vec<f64>) {
    let column = |arm: u32| {
        let at = table.columns.iter().position(|c| c.arm == arm && c.threshold.is_none()).unwrap();
        table.column(at).unwrap().to_vec()
    };
    let (control, active) = (column(0), column(1));
    let phi: Vec<f64> = active.iter().zip(&control).map(|(a, c)| a - c).collect();
    let n = phi.len() as f64;
    let total: f64 = weights.iter().sum();
    let theta = phi.iter().zip(weights).map(|(f, w)| f * w).sum::<f64>() / total;
    let correction = (n / (n - 1.0)).sqrt();
    let xi = phi.iter().zip(weights).map(|(f, w)| correction * (w / total) * (f - theta)).collect();
    (theta, xi)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn solo(fx: &Fixture, query: &AverageEffectQuery, est: &AipwAte, seed: u64) -> StudyResult {
    Study::tabular(fx.data.clone())
        .graph(fx.graph.clone())
        .query(query.clone())
        .estimator(est.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap()
}

fn shared_field<'a>(result: &'a StudyResult, key: &str) -> &'a str {
    let diagnostic = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "batch.shared_design")
        .expect("every batch member records the shared-design diagnostic");
    diagnostic
        .fields
        .iter()
        .find(|(k, _)| k.as_ref() == key)
        .map_or_else(|| panic!("batch.shared_design has no {key} field"), |(_, v)| v.as_ref())
}

fn ridge(grid: &[f64], inner_folds: usize) -> AipwAte {
    AipwAte::new().with_bootstrap_replicates(0).with_propensity_nuisance(
        PropensityNuisance::ridge_logistic(RidgeTuning::new(grid, inner_folds).unwrap()),
    )
}

#[test]
fn a_configured_batch_matches_the_single_query_call_bit_for_bit() {
    let fx = fixture(500, 61);
    let queries = [ate(T1, Y1), ate(T1, Y2), ate(T2, Y1)];
    let ctx = ExecutionContext::for_tests(61);
    for est in [AipwAte::new().with_bootstrap_replicates(0), ridge(&[0.1, 1.0, 10.0], 3)] {
        let batch = BatchStudy::new(fx.data.clone(), fx.graph.clone())
            .estimator_spec(est.clone())
            .refute(RefuteSuite::None);
        let estimated = batch.estimate_many(&queries, &ctx).unwrap();
        let prepared = batch.prepare(&queries, &ctx).unwrap();
        let refreshed = prepared.estimate(&fx.data, &ctx).unwrap();
        for (i, query) in queries.iter().enumerate() {
            let alone = solo(&fx, query, &est, 61);
            for (what, batch_result) in
                [("estimate_many", &estimated[i]), ("prepared", &refreshed[i])]
            {
                assert_eq!(
                    batch_result.estimate.ate.to_bits(),
                    alone.estimate.ate.to_bits(),
                    "{what} query {i}: ate"
                );
                assert_eq!(
                    batch_result.estimate.se_analytic.to_bits(),
                    alone.estimate.se_analytic.to_bits(),
                    "{what} query {i}: se"
                );
            }
        }
    }
}

#[test]
fn a_configuration_decides_nuisance_sharing_and_has_a_canonical_fingerprint() {
    let fx = fixture(500, 62);
    let queries = [ate(T1, Y1), ate(T1, Y2), ate(T2, Y1)];
    let ctx = ExecutionContext::for_tests(62);
    let study = |est: AipwAte| {
        BatchStudy::new(fx.data.clone(), fx.graph.clone())
            .estimator_spec(est)
            .refute(RefuteSuite::None)
    };

    // The plain configuration shares the t1 propensity between its first two members;
    // the t2 member and, for different outcomes, the outcome regressions stay separate.
    let plain = study(AipwAte::new().with_bootstrap_replicates(0));
    let results = plain.estimate_many(&queries, &ctx).unwrap();
    assert_eq!(shared_field(&results[0], "shares_propensity"), "true");
    assert_eq!(shared_field(&results[1], "shares_propensity"), "true");
    assert_eq!(shared_field(&results[2], "shares_propensity"), "false");
    assert_eq!(shared_field(&results[0], "shares_outcome_residualization"), "false");

    // A penalized propensity is never shared, even between identical treatments.
    let penalized = study(ridge(&[0.1, 1.0, 10.0], 3));
    let results = penalized.estimate_many(&queries, &ctx).unwrap();
    for result in &results {
        assert_eq!(shared_field(result, "shares_propensity"), "false");
        assert_eq!(shared_field(result, "shares_outcome_residualization"), "false");
    }

    // Every declared choice changes the canonical fingerprint; equal declarations agree
    // (a penalty grid is canonical under order and duplicates).
    let print = |est: AipwAte| study(est).estimator_fingerprint();
    let base = AipwAte::new().with_bootstrap_replicates(0);
    let different = [
        print(base.clone()),
        print(ridge(&[0.1, 1.0], 3)),
        print(ridge(&[0.1, 1.0], 4)),
        print(ridge(&[0.1, 2.0], 3)),
        print(
            base.clone()
                .with_overlap(OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: None }),
        ),
        print(AipwAte::new().with_bootstrap_replicates(50)),
    ];
    for (i, a) in different.iter().enumerate() {
        for b in &different[i + 1..] {
            assert_ne!(a, b);
        }
    }
    assert_eq!(print(ridge(&[1.0, 0.1, 1.0], 3)), print(ridge(&[0.1, 1.0], 3)));
    assert_ne!(print(base), plain_batch(&fx).estimator_fingerprint());
    // The fingerprint is on every result's shared-design diagnostic.
    let results = plain.estimate_many(&queries[..1], &ctx).unwrap();
    assert_eq!(shared_field(&results[0], "estimator_config"), plain.estimator_fingerprint());
}

#[test]
fn the_joint_score_covariance_matches_an_independent_calculation() {
    let fx = fixture(600, 63);
    let queries = [ate(T1, Y1), ate(T1, Y2), ate(T2, Y1)];
    let prepared = prepare(&fx, &queries, 63);
    let scores = prepared.prepared_scores();
    assert_eq!(scores.source(), ScoreSource::Prepared);
    let rows = scores.common_rows().unwrap().unwrap();
    let (plus, minus) = (exp_weights(&fx, &rows, 1.0), exp_weights(&fx, &rows, -1.0));
    let request = BatchRetargetRequest {
        claims: vec![
            claim("a_y1", 0, plus.clone()),
            claim("b_y1", 0, minus.clone()),
            claim("a_y2", 1, plus.clone()),
            claim("c_y1", 2, plus.clone()),
        ],
        contrasts: vec![
            contrast("route_a_minus_route_b", &[("a_y1", 1.0), ("b_y1", -1.0)]),
            contrast("mixed", &[("a_y1", 2.0), ("a_y2", -1.0), ("c_y1", 0.5)]),
        ],
        expected_snapshot: scores.snapshot_id().unwrap(),
    };
    let report = retarget(&prepared, &scores, &request).unwrap();
    let covariance = report.complete_family().expect("every member succeeded with covariance");
    assert_eq!(report.scores_source, ScoreSource::Prepared);
    assert_eq!(covariance.names, ["a_y1", "b_y1", "a_y2", "c_y1"]);

    let truth: Vec<(f64, Vec<f64>)> = [(0, &plus), (0, &minus), (1, &plus), (2, &plus)]
        .iter()
        .map(|(q, w)| oracle(scores.table(*q).unwrap(), w))
        .collect();
    let names = ["a_y1", "b_y1", "a_y2", "c_y1"];
    for (i, name) in names.iter().enumerate() {
        let point = report.claims[i].outcome.as_ref().unwrap();
        close(point.value, truth[i].0, name);
        close(point.std_error.unwrap(), dot(&truth[i].1, &truth[i].1).sqrt(), name);
        for (j, other) in names.iter().enumerate() {
            let entry = covariance.get(name, other).unwrap();
            close(entry, dot(&truth[i].1, &truth[j].1), &format!("cov({name},{other})"));
            // Symmetric to the last bit.
            assert_eq!(entry.to_bits(), covariance.get(other, name).unwrap().to_bits());
            // Cauchy-Schwarz: positive semidefinite on every 2x2 principal minor.
            let bound = (covariance.get(name, name).unwrap()
                * covariance.get(other, other).unwrap())
            .sqrt();
            assert!(entry.abs() <= bound * (1.0 + 1e-12), "cov({name},{other}) breaks CS");
        }
    }
    // Positive semidefinite: v' S v >= 0 for a spread of directions.
    for k in 0..16 {
        let v: Vec<f64> = (0..4).map(|i| f64::from(k * 7 + i * 3).sin()).collect();
        let mut form = 0.0;
        for (i, a) in names.iter().enumerate() {
            for (j, b) in names.iter().enumerate() {
                form += v[i] * v[j] * covariance.get(a, b).unwrap();
            }
        }
        assert!(form >= -1e-14, "quadratic form {form} is negative");
    }

    // A one-claim retarget is the same number through the single-query route.
    let single = prepared.plans()[0].retarget(&plus, &[z_id()], &ExecutionContext::for_tests(63));
    let single = single.unwrap();
    let first = report.claims[0].outcome.as_ref().unwrap();
    close(single.estimate.ate, first.value, "single-query point");
    close(single.estimate.se_analytic, first.std_error.unwrap(), "single-query se");

    // Contrast algebra against the influence values: value c'θ, variance Σ_r (Σ_k c_k ξ_kr)².
    let route = report.contrasts[0].outcome.as_ref().unwrap();
    close(route.value, truth[0].0 - truth[1].0, "route contrast value");
    let diff: Vec<f64> = truth[0].1.iter().zip(&truth[1].1).map(|(a, b)| a - b).collect();
    close(route.std_error.unwrap(), dot(&diff, &diff).sqrt(), "route contrast se");
    let mixed = report.contrasts[1].outcome.as_ref().unwrap();
    close(mixed.value, 2.0 * truth[0].0 - truth[2].0 + 0.5 * truth[3].0, "mixed value");
    let combo: Vec<f64> = (0..rows.len())
        .map(|r| 2.0 * truth[0].1[r] - truth[2].1[r] + 0.5 * truth[3].1[r])
        .collect();
    close(mixed.std_error.unwrap(), dot(&combo, &combo).sqrt(), "mixed se");

    // The simultaneous interval is closed with a typed reason, not published.
    assert_eq!(report.simultaneous_interval.code, "cell_not_licensed");
    assert_eq!(report.simultaneous_interval.detail, "batch_retarget.simultaneous_interval_closed");
}

#[test]
fn claim_order_does_not_change_the_family_or_any_value() {
    let fx = fixture(500, 64);
    let queries = [ate(T1, Y1), ate(T2, Y1)];
    let prepared = prepare(&fx, &queries, 64);
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let (plus, minus) = (exp_weights(&fx, &rows, 1.0), exp_weights(&fx, &rows, -1.0));
    let declared = BatchRetargetRequest {
        claims: vec![
            claim("p", 0, plus.clone()),
            claim("q", 1, minus.clone()),
            claim("r", 0, minus),
        ],
        contrasts: vec![contrast("d", &[("p", 1.0), ("q", -1.0)]), contrast("e", &[("r", 1.0)])],
        expected_snapshot: None,
    };
    let reversed = BatchRetargetRequest {
        claims: declared.claims.iter().rev().cloned().collect(),
        contrasts: declared
            .contrasts
            .iter()
            .rev()
            .map(|c| RetargetContrast {
                name: c.name.clone(),
                coefficients: c.coefficients.iter().rev().cloned().collect(),
            })
            .collect(),
        expected_snapshot: None,
    };
    let a = retarget(&prepared, &scores, &declared).unwrap();
    let b = retarget(&prepared, &scores, &reversed).unwrap();
    assert_eq!(a.family_id, b.family_id);
    let find = |report: &antecedent::BatchRetargetReport, name: &str| {
        report.claims.iter().find(|c| c.name == name).unwrap().outcome.clone().unwrap()
    };
    for name in ["p", "q", "r"] {
        let (x, y) = (find(&a, name), find(&b, name));
        assert_eq!(x.value.to_bits(), y.value.to_bits(), "{name}");
        assert_eq!(x.std_error.unwrap().to_bits(), y.std_error.unwrap().to_bits(), "{name}");
    }
    let (ca, cb) = (a.covariance.as_ref().unwrap(), b.covariance.as_ref().unwrap());
    for x in ["p", "q", "r"] {
        for y in ["p", "q", "r"] {
            assert_eq!(ca.get(x, y).unwrap().to_bits(), cb.get(x, y).unwrap().to_bits());
        }
    }
    for name in ["d", "e"] {
        let get = |report: &antecedent::BatchRetargetReport| {
            report.contrasts.iter().find(|c| c.name == name).unwrap().outcome.clone().unwrap()
        };
        close(get(&a).value, get(&b).value, name);
        close(get(&a).std_error.unwrap(), get(&b).std_error.unwrap(), name);
    }
    // A different weight changes the family identity.
    let mut other = declared.clone();
    other.claims[0].weights[0] += 1e-9;
    assert_ne!(retarget(&prepared, &scores, &other).unwrap().family_id, a.family_id);
}

#[test]
fn a_failed_member_is_reported_and_the_family_is_never_complete() {
    let fx = fixture(500, 65);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let prepared = prepare(&fx, &queries, 65);
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let plus = exp_weights(&fx, &rows, 1.0);
    // Three rows carry all of the target's mass: weighted overlap cannot hold.
    let mut spike = vec![0.0; rows.len()];
    spike[..3].fill(1.0);
    let request = BatchRetargetRequest {
        claims: vec![
            claim("good", 0, plus.clone()),
            claim("spiked", 1, spike),
            claim("short", 1, plus[..10].to_vec()),
        ],
        contrasts: vec![
            contrast("good_minus_spiked", &[("good", 1.0), ("spiked", -1.0)]),
            contrast("good_only", &[("good", 1.0)]),
        ],
        expected_snapshot: None,
    };
    let report = retarget(&prepared, &scores, &request).unwrap();
    assert!(report.claims[0].outcome.is_ok());
    let spiked = report.claims[1].outcome.as_ref().unwrap_err();
    assert!(spiked.support_refused);
    assert_eq!(spiked.detail.as_deref(), Some("batch_retarget.weighted_overlap_failed"));
    assert_eq!(spiked.reason_code.as_deref(), Some("cell_not_licensed"));
    let short = report.claims[2].outcome.as_ref().unwrap_err();
    assert_eq!(short.detail.as_deref(), Some("batch_retarget.incompatible_target"));
    assert_eq!(short.reason_code.as_deref(), Some("invalid_argument"));
    let bad_contrast = report.contrasts[0].outcome.as_ref().unwrap_err();
    assert_eq!(bad_contrast.detail.as_deref(), Some("batch_retarget.contrast_member_failed"));
    assert!(report.contrasts[1].outcome.is_ok(), "a contrast of surviving members survives");
    assert_eq!(report.failed_members(), ["spiked", "short", "good_minus_spiked"]);

    // The surviving member keeps its covariance, but the family is not a complete claim.
    assert_eq!(report.covariance.as_ref().unwrap().names, ["good"]);
    let refusal = report.complete_family().unwrap_err();
    assert_eq!(refusal.detail, "batch_retarget.partial_family");
    assert_eq!(refusal.code, "cell_not_licensed");
    assert!(refusal.message.contains("spiked") && refusal.message.contains("short"));

    // The tidy export lists every member, the failed ones with their typed refusal.
    let rows = report.tidy_rows();
    assert_eq!(rows.len(), 5);
    assert!(rows.iter().all(|r| r.family_id == report.family_id
        && !r.family_complete
        && r.family_failed == 3
        && r.family_size == 5
        && r.simultaneous_interval == "closed"
        && r.scores_source == "prepared"));
    let row = |name: &str| rows.iter().find(|r| r.name == name).unwrap();
    assert_eq!(row("good").status, "ok");
    assert_eq!(row("good").support_status, "supported");
    assert_eq!(row("spiked").status, "failed");
    assert_eq!(row("spiked").support_status, "refused");
    assert_eq!(row("spiked").refusal_code.as_deref(), Some("cell_not_licensed"));
    assert_eq!(row("short").refusal_detail.as_deref(), Some("batch_retarget.incompatible_target"));
    assert_eq!(row("good_minus_spiked").kind, TidyKind::Contrast);
    assert_eq!(row("good_minus_spiked").status, "failed");
    assert_eq!(row("good_only").status, "ok");
    assert_eq!(row("good_only").value, row("good").value);
}

#[test]
fn a_malformed_family_is_refused_with_a_typed_reason() {
    let fx = fixture(400, 66);
    let prepared = prepare(&fx, &[ate(T1, Y1)], 66);
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let w = exp_weights(&fx, &rows, 1.0);
    let refuse =
        |request: BatchRetargetRequest| retarget(&prepared, &scores, &request).unwrap_err();
    let family = |claims: Vec<RetargetClaim>, contrasts: Vec<RetargetContrast>| {
        BatchRetargetRequest { claims, contrasts, expected_snapshot: None }
    };

    let empty = refuse(family(vec![], vec![]));
    assert_eq!((empty.code, empty.detail), ("invalid_argument", "batch_retarget.empty_family"));
    let duplicate =
        refuse(family(vec![claim("a", 0, w.clone()), claim("a", 0, w.clone())], vec![]));
    assert_eq!(duplicate.detail, "batch_retarget.duplicate_name");
    let clash = refuse(family(vec![claim("a", 0, w.clone())], vec![contrast("a", &[("a", 1.0)])]));
    assert_eq!(clash.detail, "batch_retarget.duplicate_name");
    let unknown_query = refuse(family(vec![claim("a", 3, w.clone())], vec![]));
    assert_eq!(unknown_query.detail, "batch_retarget.unknown_claim");
    let unknown_claim =
        refuse(family(vec![claim("a", 0, w.clone())], vec![contrast("c", &[("zzz", 1.0)])]));
    assert_eq!(unknown_claim.detail, "batch_retarget.unknown_claim");
    for bad in [
        contrast("c", &[]),
        contrast("c", &[("a", f64::NAN)]),
        contrast("c", &[("a", 1.0), ("a", -1.0)]),
    ] {
        let invalid = refuse(family(vec![claim("a", 0, w.clone())], vec![bad]));
        assert_eq!(invalid.detail, "batch_retarget.invalid_contrast");
    }
    let mut declared = family(vec![claim("a", 0, w.clone())], vec![]);
    declared.expected_snapshot = Some("not-this-snapshot".into());
    let moved = refuse(declared);
    assert_eq!(
        (moved.code, moved.detail),
        ("row_weights_bound_to_snapshot", "batch_retarget.mixed_snapshot")
    );
    // Undeclared nonconstant weights are a typed member failure, not a silent reweighting.
    let report = retarget(
        &prepared,
        &scores,
        &family(
            vec![RetargetClaim {
                name: "undeclared".into(),
                query_index: 0,
                weights: w,
                depends_on: vec![],
            }],
            vec![],
        ),
    )
    .unwrap();
    assert!(report.claims[0].outcome.is_err());
}

#[test]
fn retarget_after_estimate_reweights_that_estimates_rows() {
    let fx = fixture(500, 67);
    let queries = [ate(T1, Y1), ate(T2, Y1)];
    let prepared = prepare(&fx, &queries, 67);
    let ctx = ExecutionContext::for_tests(67);
    let prepared_scores = prepared.prepared_scores();
    let prepared_snapshot = prepared_scores.snapshot_id().unwrap().unwrap();

    // An estimate on the same table returns its own scores, equal to its results' tables.
    let (results, estimated) = prepared.estimate_scored(&fx.data, &ctx).unwrap();
    assert_eq!(estimated.source(), ScoreSource::Estimated);
    for (i, result) in results.iter().enumerate() {
        assert_eq!(estimated.table(i), result.estimate.score_table.as_ref());
    }
    let rows = estimated.common_rows().unwrap().unwrap();
    let w = exp_weights(&fx, &rows, 1.0);
    let report = retarget(
        &prepared,
        &estimated,
        &BatchRetargetRequest {
            claims: vec![claim("x", 0, w.clone()), claim("y", 1, w.clone())],
            contrasts: vec![contrast("x_minus_y", &[("x", 1.0), ("y", -1.0)])],
            expected_snapshot: Some(prepared_snapshot.clone()),
        },
    )
    .unwrap();
    assert_eq!(report.scores_source, ScoreSource::Estimated);
    let (theta, xi) = oracle(estimated.table(0).unwrap(), &w);
    let x = report.claims[0].outcome.as_ref().unwrap();
    close(x.value, theta, "estimated point");
    close(x.std_error.unwrap(), dot(&xi, &xi).sqrt(), "estimated se");

    // Fresh rows of another size: the estimate's snapshot is a different one, a family
    // declared on the prepare-time snapshot is refused, and old-length weights fail their
    // members rather than being reused on rows they were not aligned to.
    let fresh = fixture(430, 68);
    let (_, later) = prepared.estimate_scored(&fresh.data, &ctx).unwrap();
    let later_snapshot = later.snapshot_id().unwrap().unwrap();
    assert_ne!(later_snapshot, prepared_snapshot);
    let declared_before = BatchRetargetRequest {
        claims: vec![claim("x", 0, w.clone())],
        contrasts: vec![],
        expected_snapshot: Some(prepared_snapshot),
    };
    let moved = retarget(&prepared, &later, &declared_before).unwrap_err();
    assert_eq!(moved.detail, "batch_retarget.mixed_snapshot");
    let stale = retarget(
        &prepared,
        &later,
        &BatchRetargetRequest { expected_snapshot: None, ..declared_before },
    )
    .unwrap();
    let failure = stale.claims[0].outcome.as_ref().unwrap_err();
    assert_eq!(failure.detail.as_deref(), Some("batch_retarget.incompatible_target"));
    let rows = later.common_rows().unwrap().unwrap();
    assert_eq!(rows.len(), 430);
    let aligned = retarget(
        &prepared,
        &later,
        &BatchRetargetRequest {
            claims: vec![claim("x", 0, exp_weights(&fresh, &rows, 1.0))],
            contrasts: vec![],
            expected_snapshot: Some(later_snapshot),
        },
    )
    .unwrap();
    assert!(aligned.claims[0].outcome.is_ok());
}

#[test]
fn a_plan_without_scores_refuses_instead_of_losing_the_retarget_silently() {
    let fx = fixture(400, 69);
    let ctx = ExecutionContext::for_tests(69);
    let prepared = BatchStudy::new(fx.data.clone(), fx.graph.clone())
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&[ate(T1, Y1)], &ctx)
        .unwrap();
    let w = vec![1.0; 400];
    let request = BatchRetargetRequest {
        claims: vec![claim("a", 0, w)],
        contrasts: vec![],
        expected_snapshot: None,
    };
    let none_prepared = retarget(&prepared, &prepared.prepared_scores(), &request).unwrap();
    let failure = none_prepared.claims[0].outcome.as_ref().unwrap_err();
    assert_eq!(failure.reason_code.as_deref(), Some("score_table_unavailable"));
    assert_eq!(failure.detail.as_deref(), Some("batch_retarget.scores_unavailable"));

    let (_, estimated) = prepared.estimate_scored(&fx.data, &ctx).unwrap();
    let after = retarget(&prepared, &estimated, &request).unwrap();
    let failure = after.claims[0].outcome.as_ref().unwrap_err();
    assert_eq!(failure.reason_code.as_deref(), Some("score_table_unavailable"));
    assert_eq!(failure.detail.as_deref(), Some("batch_retarget.scores_unavailable_after_estimate"));
    let rows = after.tidy_rows();
    assert_eq!(rows[0].status, "failed");
    assert_eq!(rows[0].family_failed, 1);
}

#[test]
fn a_penalized_family_retargets_with_its_plug_in_score_covariance() {
    let fx = fixture(500, 70);
    let ctx = ExecutionContext::for_tests(70);
    let prepared = BatchStudy::new(fx.data.clone(), fx.graph.clone())
        .estimator_spec(ridge(&[0.1, 1.0, 10.0], 3))
        .refute(RefuteSuite::None)
        .prepare(&[ate(T1, Y1), ate(T1, Y2)], &ctx)
        .unwrap();
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let w = exp_weights(&fx, &rows, 1.0);
    let report = retarget(
        &prepared,
        &scores,
        &BatchRetargetRequest {
            claims: vec![claim("a", 0, w.clone()), claim("b", 1, w.clone())],
            contrasts: vec![contrast("a_minus_b", &[("a", 1.0), ("b", -1.0)])],
            expected_snapshot: None,
        },
    )
    .unwrap();
    // The penalized table retargets exactly like an unpenalized one: points, the plug-in
    // score covariance, contrast standard errors and a complete family.
    assert!(report.covariance.is_some());
    let a = report.claims[0].outcome.as_ref().unwrap();
    let (theta, _) = oracle(scores.table(0).unwrap(), &w);
    close(a.value, theta, "penalized point");
    assert!(a.std_error.is_some_and(|se| se.is_finite() && se > 0.0));
    assert_eq!(a.uncertainty_kind.as_str(), "plug_in_score_covariance");
    assert!(a.nuisance_provenance.contains("ridge_logistic"));
    let diff = report.contrasts[0].outcome.as_ref().unwrap();
    assert!(diff.std_error.is_some_and(|se| se.is_finite() && se > 0.0));
    assert!(report.point_only_members().is_empty());
    assert!(report.complete_family().is_ok());
    let rows = report.tidy_rows();
    assert!(rows.iter().all(|r| r.status == "ok" && r.std_error.is_some()));
    assert!(rows.iter().all(|r| r.family_complete));
}

#[test]
fn a_cancelled_retarget_is_a_stop_never_a_verdict() {
    let fx = fixture(300, 71);
    let prepared = prepare(&fx, &[ate(T1, Y1)], 71);
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let request = BatchRetargetRequest {
        claims: vec![claim("a", 0, exp_weights(&fx, &rows, 1.0))],
        contrasts: vec![],
        expected_snapshot: None,
    };
    let ctx = ExecutionContext::for_tests(71);
    ctx.cancellation.cancel();
    let stopped = prepared.retarget(&scores, &request, &ctx).unwrap_err();
    assert_eq!((stopped.code, stopped.detail), ("cancelled_no_claim", "batch_retarget.cancelled"));
    // The same family then completes under a live context.
    assert!(retarget(&prepared, &scores, &request).unwrap().complete_family().is_ok());
}

#[test]
fn the_simultaneous_interval_is_closed_with_a_typed_reason() {
    let fx = fixture(300, 72);
    let prepared = prepare(&fx, &[ate(T1, Y1), ate(T2, Y1)], 72);
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let w = exp_weights(&fx, &rows, 1.0);
    let request = BatchRetargetRequest {
        claims: vec![claim("a", 0, w.clone()), claim("b", 1, w)],
        contrasts: vec![],
        expected_snapshot: None,
    };
    let report = retarget(&prepared, &scores, &request).unwrap();
    // A complete, covariance-bearing family still has no simultaneous interval: the
    // construction is nominal asymptotic and no coverage record measures it.
    assert!(report.complete_family().is_ok());
    let closed = &report.simultaneous_interval;
    assert_eq!(closed.code, "cell_not_licensed");
    assert_eq!(closed.detail, "batch_retarget.simultaneous_interval_closed");
    assert!(closed.message.contains("no coverage record"));
    assert!(report.tidy_rows().iter().all(|r| r.simultaneous_interval == "closed"));
}
