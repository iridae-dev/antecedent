//! Cross-fitted DML scores and their retarget covariance (2.2 E3 extension, work package C).
//!
//! The DML AIPW route keeps a row-identified per-arm score table built from its own
//! out-of-fold nuisances, so a batch retarget reweights those scores and reports the joint
//! plug-in score covariance of the retargeted claims. The oracles are plain sums over the raw
//! score columns written here (`theta = sum w phi / sum w`, `xi_r = sqrt(n/(n-1)) a_r (phi_r -
//! theta)`, covariance `sum xi_k xi_l`), never the library's helpers. A DML fit without a
//! per-arm score (the partially linear score) or on a trimmed population keeps no table and
//! refuses `score_table_unavailable` instead of reweighting another construction. DR-Learner
//! and `CausalForest` keep the marginal AIPW score table of the ATE they report.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    BatchRetargetReport, BatchRetargetRequest, BatchScores, BatchStudy, EstimatorSpec, RefuteSuite,
    RetargetClaim, ScoreSource,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{CausalForest, DmlAte, DmlScore, DrLearner, OverlapPolicy, ScoreTable};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const T1: u32 = 0;
const T2: u32 = 1;
const Y: u32 = 2;
const Z: u32 = 3;

struct Fixture {
    data: TabularData,
    graph: Dag,
    z: Vec<f64>,
}

/// Two confounded binary treatments, one outcome, one covariate.
fn fixture(n: usize, seed: u64) -> Fixture {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xD31);
    let (mut t1, mut t2, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        t1[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-(-0.2 + 0.8 * zi)).exp()));
        t2[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-(0.1 - 0.6 * zi)).exp()));
        y[i] = 2.0 * t1[i] + 0.5 * t2[i] + 0.4 * t1[i] * zi + zi + 0.3 * standard_normal(&mut rng);
    }
    let pairs = [("t1", t1), ("t2", t2), ("y", y), ("z", z.clone())];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(Z, T1), (Z, T2), (Z, Y), (T1, Y), (T2, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    Fixture { data, graph, z }
}

fn ate(treatment: u32) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(treatment), VariableId::from_raw(Y))
}

fn batch(fx: &Fixture, spec: impl Into<EstimatorSpec>) -> BatchStudy {
    BatchStudy::new(fx.data.clone(), fx.graph.clone())
        .estimator_spec(spec)
        .refute(RefuteSuite::None)
}

fn claim(name: &str, query_index: usize, weights: Vec<f64>) -> RetargetClaim {
    RetargetClaim {
        name: name.into(),
        query_index,
        weights,
        depends_on: vec![VariableId::from_raw(Z)],
    }
}

fn tilt(fx: &Fixture, rows: &[u32], sign: f64) -> Vec<f64> {
    rows.iter().map(|&r| (sign * 0.4 * fx.z[r as usize]).exp()).collect()
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs())), "{what}: {a} vs {b}");
}

/// `(theta, xi)` of the arm-1 minus arm-0 contrast score of `table` under `weights`.
fn oracle(table: &ScoreTable, weights: &[f64]) -> (f64, Vec<f64>) {
    let n = table.n_rows;
    let phi: Vec<f64> = (0..n).map(|i| table.scores[n + i] - table.scores[i]).collect();
    let total: f64 = weights.iter().sum();
    let theta = phi.iter().zip(weights).map(|(f, w)| f * w).sum::<f64>() / total;
    let correction = (n as f64 / (n as f64 - 1.0)).sqrt();
    let xi = phi.iter().zip(weights).map(|(f, w)| correction * (w / total) * (f - theta)).collect();
    (theta, xi)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn run(
    prepared: &antecedent::PreparedBatch,
    scores: &BatchScores,
    claims: Vec<RetargetClaim>,
) -> BatchRetargetReport {
    let request = BatchRetargetRequest { claims, contrasts: Vec::new(), expected_snapshot: None };
    prepared.retarget(scores, &request, &ExecutionContext::for_tests(1)).unwrap()
}

/// Retarget after estimate: the estimate's own table is reweighted, uniform weights return the
/// route's point and iid standard error, tilted weights and the cross-claim covariance equal
/// the independent sums, and the prepare-time table equals the estimate's table.
#[test]
fn a_dml_batch_retargets_its_own_scores_with_the_joint_covariance() {
    let fx = fixture(500, 91);
    let ctx = ExecutionContext::for_tests(91);
    let prepared = batch(&fx, DmlAte::new()).prepare(&[ate(T1), ate(T2)], &ctx).expect("prepare");
    let (results, estimated) = prepared.estimate_scored(&fx.data, &ctx).unwrap();
    assert_eq!(estimated.source(), ScoreSource::Estimated);
    for (i, result) in results.iter().enumerate() {
        let table = result.estimate.score_table.as_ref().expect("DML keeps its scores");
        // The table names the identified estimand's ids, not the projected table's renumbering.
        assert_eq!(table.adjustment_set.to_vec(), vec![VariableId::from_raw(Z)]);
        assert_eq!(table.treatment, VariableId::from_raw(if i == 0 { T1 } else { T2 }));
        assert_eq!(estimated.table(i), Some(table));
        // Seeded replay: the prepare-time table is the estimate's table on the same seed.
        assert_eq!(prepared.prepared_scores().table(i), Some(table));
        let rows = estimated.common_rows().unwrap().unwrap();
        let uniform = run(&prepared, &estimated, vec![claim("u", i, vec![1.0; rows.len()])]);
        let point = uniform.claims[0].outcome.as_ref().unwrap();
        // Score-mean identity and the iid plug-in covariance of the route's own scores.
        close(point.value, result.estimate.ate, "uniform retarget = DML ate");
        close(point.std_error.unwrap(), result.estimate.se_analytic, "uniform se = DML se");
        assert!(point.nuisance_provenance.starts_with("dml.crossfit.v1;route=dml;"));
    }
    let rows = estimated.common_rows().unwrap().unwrap();
    let (plus, minus) = (tilt(&fx, &rows, 1.0), tilt(&fx, &rows, -1.0));
    let report = run(
        &prepared,
        &estimated,
        vec![
            claim("a", 0, plus.clone()),
            claim("b", 0, minus.clone()),
            claim("c", 1, plus.clone()),
        ],
    );
    let covariance = report.complete_family().expect("complete family with covariance");
    let truth = [
        oracle(estimated.table(0).unwrap(), &plus),
        oracle(estimated.table(0).unwrap(), &minus),
        oracle(estimated.table(1).unwrap(), &plus),
    ];
    let names = ["a", "b", "c"];
    for (i, name) in names.iter().enumerate() {
        let point = report.claims[i].outcome.as_ref().unwrap();
        close(point.value, truth[i].0, name);
        close(point.std_error.unwrap(), dot(&truth[i].1, &truth[i].1).sqrt(), name);
        for (j, other) in names.iter().enumerate() {
            close(
                covariance.get(name, other).unwrap(),
                dot(&truth[i].1, &truth[j].1),
                &format!("cov({name},{other})"),
            );
        }
    }
    // Retargeted claims differ from the uniform point: the weights do something.
    let a = report.claims[0].outcome.as_ref().unwrap().value;
    let b = report.claims[1].outcome.as_ref().unwrap().value;
    assert!((a - b).abs() > 1e-3);
}

/// A DML fit with no per-arm score (partially linear) or on a trimmed population keeps no
/// table: every member fails with `score_table_unavailable`, before and after an estimate.
#[test]
fn a_dml_fit_without_scores_refuses_the_retarget() {
    let fx = fixture(400, 92);
    let ctx = ExecutionContext::for_tests(92);
    let trimmed = {
        let mut est = DmlAte::new();
        est.overlap = OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.05) };
        est
    };
    for spec in [DmlAte::new().with_score(DmlScore::PartiallyLinear), trimmed] {
        let prepared = batch(&fx, spec).prepare(&[ate(T1)], &ctx).unwrap();
        let before = prepared.prepared_scores();
        assert!(before.table(0).is_none());
        let rows = fx.data.row_count();
        let request = |n: usize| BatchRetargetRequest {
            claims: vec![claim("a", 0, vec![1.0; n])],
            contrasts: Vec::new(),
            expected_snapshot: None,
        };
        let report = prepared.retarget(&before, &request(rows), &ctx).unwrap();
        let failure = report.claims[0].outcome.as_ref().unwrap_err();
        assert_eq!(failure.reason_code.as_deref(), Some("score_table_unavailable"));
        assert_eq!(failure.detail.as_deref(), Some("batch_retarget.scores_unavailable"));
        let (results, estimated) = prepared.estimate_scored(&fx.data, &ctx).unwrap();
        assert!(results[0].estimate.score_table.is_none());
        let report = prepared.retarget(&estimated, &request(rows), &ctx).unwrap();
        let failure = report.claims[0].outcome.as_ref().unwrap_err();
        assert_eq!(failure.reason_code.as_deref(), Some("score_table_unavailable"));
        assert_eq!(
            failure.detail.as_deref(),
            Some("batch_retarget.scores_unavailable_after_estimate")
        );
        assert!(report.complete_family().is_err());
    }
}

/// Retarget a marginal-ATE score table: the table names its route, equals the prepare-time
/// table, and a uniform retarget returns the reported ATE and the iid plug-in standard error.
fn assert_marginal_scores_retarget(spec: EstimatorSpec, route: &str) {
    let fx = fixture(300, 93);
    let ctx = ExecutionContext::for_tests(93);
    let prepared = batch(&fx, spec).prepare(&[ate(T1)], &ctx).unwrap();
    let (results, estimated) = prepared.estimate_scored(&fx.data, &ctx).unwrap();
    let table = results[0].estimate.score_table.as_ref().expect("marginal scores kept");
    assert!(table.nuisance_provenance.contains(route), "{}", table.nuisance_provenance);
    assert_eq!(prepared.prepared_scores().table(0), Some(table));
    let rows = estimated.common_rows().unwrap().unwrap();
    let report = run(&prepared, &estimated, vec![claim("u", 0, vec![1.0; rows.len()])]);
    let point = report.claims[0].outcome.as_ref().unwrap();
    close(point.value, results[0].estimate.ate, "uniform retarget = reported ATE");
    close(point.std_error.unwrap(), results[0].estimate.se_analytic, "uniform se");
}

/// `DrLearner` keeps the marginal AIPW score table of the ATE it reports, never a score
/// derived from its CATE.
#[test]
fn dr_learner_retargets_its_marginal_scores() {
    assert_marginal_scores_retarget(DrLearner::new().into(), "route=dr_learner");
}

/// `CausalForest` keeps the DML AIPW score table of its marginal ATE, never a score derived
/// from its CATE.
#[test]
fn causal_forest_retargets_its_marginal_scores() {
    assert_marginal_scores_retarget(CausalForest::new().with_n_trees(10).into(), "route=dml");
}
