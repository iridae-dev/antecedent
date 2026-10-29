//! Coverage of the published cluster-DML Wald interval `estimate ± t_{G-1} · se` (record
//! `estimate.clustered_dml_aipw`).
//!
//! Clustered data with a cluster-level random effect on the outcome, a covariate and the
//! treatment effect; the population average effect is 2. The scored interval is the
//! facade's own (`PublishedScalarUncertainty::interval` of the cluster-DML result: the
//! cluster-sandwich standard error with its few-cluster Student-t reference), and the sample
//! grid varies the number of clusters.
//!
//! `cluster_dml_t_wald_interval` is registered in `scripts/gate_calibration.sh` (`run_cd`) and
//! its coverage is unmeasured: the coverage record is the only missing piece. A fixed-seed test of the same construction without a record is
//! `crates/antecedent-estimate/tests/cluster_dml_known_truth.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{PublishedScalarUncertainty, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{AipwAte, ClusterDml};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim,
};

const INTERVAL: &str = "analytic_se";
const SIZE: usize = 10;
const TRUTH: f64 = 2.0;
const T: u32 = 0;
const Y: u32 = 1;
const Z0: u32 = 2;
const Z1: u32 = 3;

/// One replicate: simulate `groups` clusters, fit the cluster-DML AIPW through the facade and
/// return the published interval with the row count.
fn replicate(groups: usize, seed: u64) -> Option<((f64, f64), u64)> {
    replicate_with_estimate(groups, seed).map(|(interval, rows, _, _)| (interval, rows))
}

/// [`replicate`] with the point estimate and the cluster-sandwich standard error the interval
/// is built from.
fn replicate_with_estimate(groups: usize, seed: u64) -> Option<((f64, f64), u64, f64, f64)> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xCD1);
    let n = groups * SIZE;
    let (mut t, mut y, mut z0, mut z1) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut cluster = vec![0_u32; n];
    for g in 0..groups {
        let shift = 0.7 * standard_normal(&mut rng);
        let shock = 1.5 * standard_normal(&mut rng);
        let effect = TRUTH + 1.5 * standard_normal(&mut rng);
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
    let columns = [("t", t), ("y", y), ("z0", z0), ("z1", z1)];
    let borrowed: Vec<(&str, &[f64])> = columns.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).ok()?;
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(Z0, T), (Z1, T), (Z0, Y), (Z1, Y), (T, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).ok()?;
    }
    let estimator = AipwAte { cluster_dml: Some(ClusterDml::new(20).ok()?), ..AipwAte::new() }
        .with_bootstrap_replicates(0)
        .with_cluster_ids(cluster);
    let result = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(T), VariableId::from_raw(Y)))
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed))
        .ok()?;
    let estimate = result.estimate().as_effect()?;
    let published = PublishedScalarUncertainty::select(estimate);
    Some((published.interval(estimate)?, n as u64, estimate.ate, estimate.uncertainty.analytic.se))
}

fn construction() -> Construction {
    Construction {
        query: "AverageEffect".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "aipw".into(),
        interval_method: INTERVAL.into(),
        se_kind: "cluster".into(),
        dependence: "cluster".into(),
        posterior: String::new(),
        functional: "all_observed.mean".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

/// The scored interval is the published one: `estimate ± t_{G-1,0.975} · se` of the
/// cluster-sandwich standard error, with the Student-t quantile of 29 degrees of freedom
/// (2.045229642...) rather than the normal 1.96.
#[test]
fn the_scored_interval_is_the_published_t_interval() {
    const T_29_0975: f64 = 2.045_229_642_132_703;
    let ((lo, hi), rows, ate, se) = replicate_with_estimate(30, 90_001).expect("a cluster-DML fit");
    assert_eq!(rows, 300);
    assert!(se.is_finite() && se > 0.0, "{se}");
    assert!((lo - (ate - T_29_0975 * se)).abs() < 1e-9, "lower {lo} vs {}", ate - T_29_0975 * se);
    assert!((hi - (ate + T_29_0975 * se)).abs() < 1e-9, "upper {hi} vs {}", ate + T_29_0975 * se);
    // The cluster-level effect heterogeneity (sd 1.5) is sampled; the population average is 2.
    assert!((ate - TRUTH).abs() < 4.0 * se, "estimate {ate} se {se}");
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn cluster_dml_t_wald_interval() {
    let groups = grid_n(40);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "cluster_dml_t_wald_interval",
            dgp: "cluster_random_effect",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
    for row in map_replicates(n_sim(), |rep| replicate(groups, 61_000 + rep)) {
        let Some((interval, rows)) = row else {
            tally.skip();
            continue;
        };
        tally.bind(
            &construction(),
            ScopeFacts {
                row_count: rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(Some(interval), TRUTH);
    }
    tally.assert();
}
