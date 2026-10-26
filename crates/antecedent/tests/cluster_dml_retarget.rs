//! Cluster-DML through the facade: the published interval carries the few-cluster Student-t
//! reference, and a retarget of a cluster-DML score table (single and batch) reports the
//! covariance of the cluster-summed weighted influence columns (`G/(G-1)` sandwich) instead of
//! the iid Gram, with its `t_{G-1}` reference.
//!
//! The oracle is written from the raw score table: with `a_r = w_r / sum w`, `theta` the weighted
//! mean of the contrast score `phi_r` and `S_g = sum_{r in g} a_r (phi_r - theta)`, the claim's
//! variance is `G/(G-1) sum_g S_g^2` and two claims' covariance is `G/(G-1) sum_g S_gk S_gl`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent::{
    BatchRetargetRequest, BatchStudy, PublishedScalarUncertainty, RefuteSuite, RetargetClaim, Study,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{AipwAte, ClusterDml, DmlAte, DrLearner, SamplingWeights, ScoreTable};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use antecedent_stats::student_t_ppf;

const T: u32 = 0;
const Y: u32 = 1;
const Z0: u32 = 2;
const Z1: u32 = 3;
const GROUPS: usize = 40;
const SIZE: usize = 10;

struct Fixture {
    data: TabularData,
    graph: Dag,
    cluster: Vec<u32>,
    z0: Vec<f64>,
}

fn fixture(seed: u64) -> Fixture {
    fixture_with_groups(seed, GROUPS)
}

fn fixture_with_groups(seed: u64, groups: usize) -> Fixture {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xC1D);
    let n = groups * SIZE;
    let (mut t, mut y, mut z0, mut z1) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut cluster = vec![0_u32; n];
    for g in 0..groups {
        let shift = 0.7 * standard_normal(&mut rng);
        let shock = 1.5 * standard_normal(&mut rng);
        let effect = 2.0 + 1.5 * standard_normal(&mut rng);
        for k in 0..SIZE {
            let i = g * SIZE + k;
            z0[i] = shift + standard_normal(&mut rng);
            z1[i] = standard_normal(&mut rng);
            let eta = 0.6 * z0[i] - 0.4 * z1[i];
            t[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp()));
            y[i] = effect * t[i] + z0[i] + 0.5 * z1[i] + shock + 0.5 * standard_normal(&mut rng);
            cluster[i] = u32::try_from(g).unwrap();
        }
    }
    let columns = [("t", t), ("y", y), ("z0", z0.clone()), ("z1", z1)];
    let borrowed: Vec<(&str, &[f64])> = columns.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(Z0, T), (Z1, T), (Z0, Y), (Z1, Y), (T, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    Fixture { data, graph, cluster, z0 }
}

fn estimator(fx: &Fixture) -> AipwAte {
    AipwAte { cluster_dml: Some(ClusterDml::new(20).unwrap()), ..AipwAte::new() }
        .with_bootstrap_replicates(0)
        .with_cluster_ids(fx.cluster.clone())
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(T), VariableId::from_raw(Y))
}

fn study(fx: &Fixture) -> Study {
    Study::tabular(fx.data.clone())
        .graph(fx.graph.clone())
        .query(query())
        .estimator(estimator(fx))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn tilt(fx: &Fixture, sign: f64) -> Vec<f64> {
    fx.z0.iter().map(|z| (sign * 0.4 * z).exp()).collect()
}

/// `theta` and the per-cluster sums `S_g` of `a_r (phi_r - theta)` for the contrast score.
fn cluster_sums(table: &ScoreTable, cluster: &[u32], weights: &[f64]) -> (f64, BTreeMap<u32, f64>) {
    let n = table.n_rows;
    let phi: Vec<f64> = (0..n).map(|i| table.scores[n + i] - table.scores[i]).collect();
    let total: f64 = weights.iter().sum();
    let theta = phi.iter().zip(weights).map(|(f, w)| f * w).sum::<f64>() / total;
    let mut sums: BTreeMap<u32, f64> = BTreeMap::new();
    for ((f, w), g) in phi.iter().zip(weights).zip(cluster) {
        *sums.entry(*g).or_default() += (w / total) * (f - theta);
    }
    (theta, sums)
}

fn cov(a: &BTreeMap<u32, f64>, b: &BTreeMap<u32, f64>) -> f64 {
    let g = a.len() as f64;
    g / (g - 1.0) * a.iter().map(|(k, x)| x * b[k]).sum::<f64>()
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs())), "{what}: {a} vs {b}");
}

/// The published interval of a cluster-DML result is `estimate +- t_{G-1} * se`, not the normal
/// interval, and a result without a declared reference keeps the normal quantile.
#[test]
fn the_published_interval_uses_the_few_cluster_t_reference() {
    let fx = fixture(5);
    let study = study(&fx);
    let result = study.run(&ExecutionContext::for_tests(5)).unwrap();
    let basis = result.calibration_bases(&study.inspect().unwrap()).unwrap().remove(0);
    assert_eq!(basis.key.dependence, "cluster");
    assert_eq!(basis.key.interval_method, "analytic_se");
    let measured = *antecedent_io::coverage_records_data::RECORDS
        .iter()
        .find(|record| record.test.ends_with("::cluster_dml_t_wald_interval"))
        .expect("the unweighted Student construction was measured");
    assert_eq!(
        antecedent_io::calibration::calibration_slot_in(&basis, &[measured]).record_id.as_deref(),
        Some(measured.id),
    );
    let estimate = result.estimate.as_effect().unwrap();
    assert_eq!(estimate.uncertainty.analytic.reference_df, Some(39.0));
    let published = PublishedScalarUncertainty::select(estimate);
    let t = student_t_ppf(0.975, 39.0);
    close(published.critical_value(estimate), t, "critical value");
    assert!(t > antecedent::result::reported_se_interval_z());
    let (lo, hi) = published.interval(estimate).unwrap();
    close(lo, estimate.ate - t * estimate.se_analytic(), "lower");
    close(hi, estimate.ate + t * estimate.se_analytic(), "upper");
    // An ordinary AIPW fit declares no reference and keeps the normal interval.
    let plain = Study::tabular(fx.data.clone())
        .graph(fx.graph.clone())
        .query(query())
        .estimator(AipwAte::new().with_bootstrap_replicates(0))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap();
    let plain = plain.estimate.as_effect().unwrap();
    assert_eq!(plain.uncertainty.analytic.reference_df, None);
    let published = PublishedScalarUncertainty::select(plain);
    close(published.critical_value(plain), antecedent::result::reported_se_interval_z(), "normal");
}

/// Even where the Student evidence covers the row count, a normal interval cannot borrow it.
#[test]
fn unweighted_normal_cluster_never_borrows_student_calibration() {
    use antecedent_io::calibration::calibration_slot_in;
    use antecedent_io::coverage_records_data::RECORDS;

    let student = *RECORDS
        .iter()
        .find(|record| record.test.ends_with("::cluster_dml_t_wald_interval"))
        .unwrap();
    let normal = *RECORDS
        .iter()
        .find(|record| {
            record.test.ends_with("::practitioner_aipw_cluster_interval")
                && record.nominal.to_bits() == 0.95_f64.to_bits()
        })
        .unwrap();
    assert!(student.n_min <= 800 && student.n_max >= 800);
    for rows in [800, 1000] {
        let fx = fixture_with_groups(5, rows / SIZE);
        let study = Study::tabular(fx.data.clone())
            .graph(fx.graph.clone())
            .query(query())
            .estimator(
                AipwAte::new()
                    .with_bootstrap_replicates(0)
                    .with_se_kind(antecedent_estimate::AnalyticSeKind::Cluster)
                    .with_source_cluster_ids(fx.cluster.clone()),
            )
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let result = study.run(&ExecutionContext::for_tests(5)).unwrap();
        assert!(result.estimate.as_effect().unwrap().uncertainty.analytic.reference_df.is_none());
        let basis = result.calibration_bases(&study.inspect().unwrap()).unwrap().remove(0);
        assert_eq!(basis.key.interval_method, "analytic_se");
        assert_eq!(basis.key.se_kind, "cluster");
        assert_eq!(basis.key.dependence, "iid", "historical normal key convention");
        let selected = calibration_slot_in(&basis, &[student, normal]);
        assert_eq!(selected.record_id.as_deref(), Some(normal.id));
        assert_eq!(selected.status, if rows == 800 { "scope_not_assessed" } else { "calibrated" });
    }
}

/// These are lookup fixtures only: the stored normal row must never license Student endpoints.
#[test]
fn weighted_cluster_student_calibration_matches_its_actual_reference() {
    use antecedent::estimator_spec::EstimatorSpec;
    use antecedent_io::calibration::calibration_slot_in;
    use antecedent_io::coverage_records_data::RECORDS;

    let fx = fixture(5);
    let weights = SamplingWeights::new(tilt(&fx, 1.0)).unwrap();
    let units = ClusterDml::new(20).unwrap();
    let estimators: [EstimatorSpec; 3] = [
        estimator(&fx).with_sampling_weights(weights.clone()).into(),
        DmlAte::new()
            .with_cluster_dml(units)
            .with_source_cluster_ids(fx.cluster.clone())
            .with_sampling_weights(weights.clone())
            .into(),
        DrLearner::new()
            .with_cluster_dml(units)
            .with_source_cluster_ids(fx.cluster.clone())
            .with_sampling_weights(weights.clone())
            .into(),
    ];
    let template = *RECORDS
        .iter()
        .find(|record| {
            record.test.ends_with("::weighted_aipw_cluster_dml_interval")
                && record.interval_method == "analytic_se"
        })
        .expect("the historical normal record is retained");
    for estimator in estimators {
        let study = Study::tabular(fx.data.clone())
            .graph(fx.graph.clone())
            .query(query())
            .estimator(estimator)
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let result = study.run(&ExecutionContext::for_tests(5)).unwrap();
        let contract = study.inspect().unwrap();
        let basis = result.calibration_bases(&contract).unwrap().remove(0);
        assert_eq!(
            basis.key.interval_method,
            "student_analytic",
            "basis={:?}; estimator={:?}; binding={:?}; analytic={:?}; functional={:?}",
            basis.key,
            result.logical_plan.estimator,
            result.primary_interval_binding(false),
            result.estimate.as_effect().unwrap().uncertainty.analytic,
            contract.functional,
        );
        assert_eq!(basis.key.functional, "all_observed.mean+reference=cluster_df+weights=sampling");
        assert_eq!(basis.key.dependence, "cluster");
        let mut historical = template;
        historical.id = "cov.fixture.historical_normal_cluster";
        historical.estimator = match basis.key.estimator.as_str() {
            "aipw" => "aipw",
            "dml" => "dml",
            "dr.learner" => "dr.learner",
            other => panic!("unexpected estimator {other}"),
        };
        historical.n_min = 1;
        historical.n_max = u64::MAX;
        historical.grid = &[];
        assert!(calibration_slot_in(&basis, &[historical]).record_id.is_none());
        let mut student = historical;
        student.id = "cov.fixture.student_cluster";
        student.interval_method = "student_analytic";
        student.functional = "all_observed.mean+reference=cluster_df+weights=sampling";
        let selected = calibration_slot_in(&basis, &[historical, student]);
        assert_eq!(selected.record_id.as_deref(), Some(student.id));
    }
    let ordinary = Study::tabular(fx.data.clone())
        .graph(fx.graph.clone())
        .query(query())
        .estimator(
            AipwAte::new()
                .with_bootstrap_replicates(0)
                .with_sampling_weights(weights)
                .with_se_kind(antecedent_estimate::AnalyticSeKind::Cluster)
                .with_source_cluster_ids(fx.cluster.clone()),
        )
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = ordinary.run(&ExecutionContext::for_tests(5)).unwrap();
    let basis = result.calibration_bases(&ordinary.inspect().unwrap()).unwrap().remove(0);
    assert_eq!(basis.key.interval_method, "analytic_se");
    assert_eq!(basis.key.functional, "all_observed.mean+weights=sampling");
    assert_eq!(result.estimate.as_effect().unwrap().uncertainty.analytic.reference_df, None);
    assert_eq!(basis.key.se_kind, "cluster");
    assert_eq!(basis.key.dependence, "cluster");
    let mut normal = template;
    normal.n_min = 1;
    normal.n_max = u64::MAX;
    normal.grid = &[];
    assert_eq!(calibration_slot_in(&basis, &[normal]).record_id.as_deref(), Some(normal.id));
}

/// A single retarget of the prepared cluster-DML table reports the cluster-summed standard error
/// with its `t_{G-1}` reference; uniform weights return the fit's own standard error.
#[test]
fn a_single_retarget_reports_the_cluster_summed_se_with_its_reference() {
    let fx = fixture(6);
    let ctx = ExecutionContext::for_tests(6);
    let prepared = study(&fx).prepare(&ctx).unwrap();
    let table = prepared.score_table().expect("a cluster-DML plan prepares its table").clone();
    let fit = study(&fx).run(&ctx).unwrap();
    let n = table.n_rows;

    let uniform = prepared.retarget(&vec![1.0; n], &[], &ctx).unwrap();
    let estimate = uniform.estimate.as_effect().unwrap();
    close(estimate.ate, fit.estimate.as_effect().unwrap().ate, "uniform point");
    close(estimate.se_analytic(), fit.estimate.as_effect().unwrap().se_analytic(), "uniform se");
    assert_eq!(estimate.uncertainty.analytic.reference_df, Some(39.0));
    assert!(estimate.joint.covariance.is_some() && estimate.scores.inference.is_none());

    let weights = tilt(&fx, 1.0);
    let result = prepared.retarget(&weights, &[VariableId::from_raw(Z0)], &ctx).unwrap();
    let estimate = result.estimate.as_effect().unwrap();
    let (theta, sums) = cluster_sums(&table, &fx.cluster, &weights);
    close(estimate.ate, theta, "tilted point");
    close(estimate.se_analytic(), cov(&sums, &sums).sqrt(), "cluster se");
    // The iid plug-in would be smaller: clusters share a random effect.
    let iid = {
        let total: f64 = weights.iter().sum();
        let phi: Vec<f64> = (0..n).map(|i| table.scores[n + i] - table.scores[i]).collect();
        let xi = phi
            .iter()
            .zip(&weights)
            .map(|(f, w)| (n as f64 / (n as f64 - 1.0)).sqrt() * (w / total) * (f - theta));
        xi.map(|x| x * x).sum::<f64>().sqrt()
    };
    assert!(estimate.se_analytic() > iid, "{} vs {iid}", estimate.se_analytic());
    let published = PublishedScalarUncertainty::select(estimate);
    close(published.critical_value(estimate), student_t_ppf(0.975, 39.0), "t reference");
}

/// A batch family over a cluster-DML plan reports the cluster-summed joint covariance
/// (variances and the cross term against bilinear cluster sums) and each claim's reference.
#[test]
fn a_batch_family_reports_the_cluster_summed_joint_covariance() {
    let fx = fixture(7);
    let ctx = ExecutionContext::for_tests(7);
    let prepared = BatchStudy::new(fx.data.clone(), fx.graph.clone())
        .estimator_spec(estimator(&fx))
        .refute(RefuteSuite::None)
        .prepare(&[query()], &ctx)
        .unwrap();
    let (results, scores) = prepared.estimate_scored(&fx.data, &ctx).unwrap();
    let table = results[0].estimate.as_effect().unwrap().scores.table.as_ref().unwrap().clone();
    assert_eq!(scores.table(0), Some(&table));
    let (plus, minus) = (tilt(&fx, 1.0), tilt(&fx, -1.0));
    let claim = |name: &str, weights: &[f64]| RetargetClaim {
        name: name.into(),
        query_index: 0,
        weights: weights.to_vec(),
        depends_on: vec![VariableId::from_raw(Z0)],
    };
    let request = BatchRetargetRequest {
        claims: vec![claim("plus", &plus), claim("minus", &minus)],
        contrasts: Vec::new(),
        expected_snapshot: None,
    };
    let report = prepared.retarget(&scores, &request, &ctx).unwrap();
    let family = report.complete_family().expect("a cluster-DML family carries covariance");
    let (_, a) = cluster_sums(&table, &fx.cluster, &plus);
    let (_, b) = cluster_sums(&table, &fx.cluster, &minus);
    close(family.get("plus", "plus").unwrap(), cov(&a, &a), "var plus");
    close(family.get("minus", "minus").unwrap(), cov(&b, &b), "var minus");
    close(family.get("plus", "minus").unwrap(), cov(&a, &b), "cov plus,minus");
    for claim in &report.claims {
        let point = claim.outcome.as_ref().unwrap();
        assert_eq!(point.reference_df, Some(39.0));
        assert!(point.nuisance_provenance.contains("cluster_dml=unit=cluster"));
    }
    // The max-t band of the cluster family is formed from the cluster covariance.
    let band = report.simultaneous_interval(0.95, 3, 5_000, &ctx).unwrap();
    close(band.members[0].std_error, cov(&a, &a).sqrt(), "band se");
}
