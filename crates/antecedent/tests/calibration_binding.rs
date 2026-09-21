//! Every estimator-level coverage record describes a construction the facade reports.
//!
//! `crates/antecedent-estimate/src/calibration_coverage.rs` measures estimator
//! configurations directly, without a `Study`. Its records are keyed with the
//! facade construction listed in `common/estimator_level.rs`. This test runs
//! the facade with each of those configurations and checks that the runtime's
//! own calibration key is exactly that construction, so an estimator-level
//! record cannot describe an interval the facade reports differently.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

mod common;

use antecedent::{EstimatorId, IdentifierId, RdConfig, RefuteSuite, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{
    AipwAte, AnalyticSeKind, FrontDoorTwoStage, LinearAdjustmentAte, PropensityMatching,
    PropensityWeighting, TwoStageLeastSquares, WaldIv,
};
use antecedent_graph::{Dag, DenseNodeId};

use common::calibration::gaussian;
use common::calibration_bind::constructions;
use common::estimator_level::{CASES, case_for};

const N: usize = 300;

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(name, col)| (*name, *col))).unwrap()
}

/// `z -> t -> y`, `z -> y`: back-door adjustment on `z`, with 10-row clusters.
fn backdoor_data(seed: u64) -> (TabularData, Vec<u32>) {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    let mut clusters = vec![0u32; N];
    for i in 0..N {
        clusters[i] = (i / 10) as u32;
        z[i] = g();
        t[i] = f64::from(0.8 * z[i] + g() > 0.0);
        y[i] = 2.0 * t[i] + t[i] * z[i] + z[i] + 0.6 * g();
    }
    (table(&[("t", &t), ("y", &y), ("z", &z)]), clusters)
}

fn backdoor_dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(d(2), d(0)).unwrap();
    dag.insert_directed(d(2), d(1)).unwrap();
    dag.insert_directed(d(0), d(1)).unwrap();
    dag
}

/// Binary instrument `z -> t -> y` with an unmeasured confounder (absent from
/// the graph), the shape the IV route identifies.
fn iv_data(seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for i in 0..N {
        let u = g();
        z[i] = (i % 2) as f64;
        t[i] = 0.6 * z[i] + u + 0.1 * g();
        y[i] = 2.0 * t[i] + u + 0.1 * g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

fn iv_dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(d(2), d(0)).unwrap();
    dag.insert_directed(d(0), d(1)).unwrap();
    dag
}

/// `t -> m -> y` with an unmeasured `t <- u -> y` (absent from the graph).
fn frontdoor_data(seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut m) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for i in 0..N {
        let u = g();
        t[i] = u + 0.8 * g();
        m[i] = t[i] + 0.7 * g();
        y[i] = 2.0 * m[i] + 1.5 * u + 0.5 * g();
    }
    table(&[("t", &t), ("y", &y), ("m", &m)])
}

/// Binary treatment on the same graph; the mediator is binary (`discrete`) or continuous.
fn frontdoor_functional_data(seed: u64, discrete: bool) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut m) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for i in 0..N {
        let u = f64::from(g() > 0.25);
        t[i] = f64::from(g() + 0.8 * u > 0.6);
        m[i] = if discrete { f64::from(g() + t[i] > 0.5) } else { 1.0 + 0.4 * t[i] + g() };
        y[i] = 2.0 * m[i] * (0.5 + u) + 0.5 * u + 0.6 * g();
    }
    table(&[("t", &t), ("y", &y), ("m", &m)])
}

fn frontdoor_dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(d(0), d(2)).unwrap();
    dag.insert_directed(d(2), d(1)).unwrap();
    dag
}

/// Running variable `r` around the cutoff; RD derives treatment from it.
fn rd_data(seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut r) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for i in 0..N {
        let centered = -1.0 + 2.0 * (i as f64) / (N as f64);
        let treated = f64::from(centered >= 0.0);
        r[i] = centered;
        t[i] = treated;
        y[i] = 1.0 + 0.5 * centered + 2.0 * treated - 0.8 * treated * centered + 0.3 * g();
    }
    table(&[("t", &t), ("y", &y), ("r", &r)])
}

fn ate_query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(v(0), v(1))
}

fn continuous_query() -> AverageEffectQuery {
    AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0)
}

/// Run the facade with the configuration of estimator-level test `test`.
fn facade_result(test: &str) -> (Study, StudyResult) {
    let ctx = ExecutionContext::for_tests(11);
    let (data, clusters) = backdoor_data(7);
    let builder = |spec: antecedent::EstimatorSpec, query: AverageEffectQuery| {
        Study::tabular(data.clone())
            .graph(backdoor_dag())
            .query(query)
            .estimator(spec)
            .refute(RefuteSuite::None)
    };
    let study = match test {
        "linear_adjustment_analytic_ci_coverage" => builder(
            LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::default() }
                .into(),
            ate_query(),
        ),
        "linear_adjustment_hc1_ci_coverage" => builder(
            LinearAdjustmentAte {
                bootstrap_replicates: 0,
                se_kind: AnalyticSeKind::Hc1,
                ..LinearAdjustmentAte::default()
            }
            .into(),
            ate_query(),
        ),
        "ipw_hajek_bootstrap_ci_coverage" => builder(
            PropensityWeighting { bootstrap_replicates: 60, ..PropensityWeighting::new() }.into(),
            ate_query(),
        ),
        "ipw_hajek_analytic_ci_coverage" | "ipw_hajek_analytic_conformance_scm_ci_coverage" => {
            builder(
                PropensityWeighting { bootstrap_replicates: 0, ..PropensityWeighting::new() }
                    .into(),
                ate_query(),
            )
        }
        "aipw_analytic_ci_coverage" => {
            builder(AipwAte { bootstrap_replicates: 0, ..AipwAte::new() }.into(), ate_query())
        }
        "aipw_ate_hc1_ci_coverage" => builder(
            AipwAte { bootstrap_replicates: 0, se_kind: AnalyticSeKind::Hc1, ..AipwAte::new() }
                .into(),
            ate_query(),
        ),
        "aipw_att_hc1_ci_coverage" => builder(
            AipwAte { bootstrap_replicates: 0, se_kind: AnalyticSeKind::Hc1, ..AipwAte::new() }
                .into(),
            ate_query().with_target_population(TargetPopulation::Treated),
        ),
        "aipw_atc_hc1_boundary_within_band" => builder(
            AipwAte { bootstrap_replicates: 0, se_kind: AnalyticSeKind::Hc1, ..AipwAte::new() }
                .into(),
            ate_query().with_target_population(TargetPopulation::Untreated),
        ),
        "aipw_att_cluster_ci_coverage" => builder(
            AipwAte { bootstrap_replicates: 0, se_kind: AnalyticSeKind::Cluster, ..AipwAte::new() }
                .with_cluster_ids(clusters)
                .into(),
            ate_query().with_target_population(TargetPopulation::Treated),
        ),
        "matching_homoskedastic_ci_coverage" => builder(
            PropensityMatching {
                bootstrap_replicates: 0,
                se_kind: AnalyticSeKind::Homoskedastic,
                ..PropensityMatching::new()
            }
            .into(),
            ate_query().with_target_population(TargetPopulation::Treated),
        ),
        "wald_iv_analytic_ci_coverage" | "wald_iv_hc1_ci_coverage" => {
            let se_kind = if test == "wald_iv_hc1_ci_coverage" {
                AnalyticSeKind::Hc1
            } else {
                AnalyticSeKind::Homoskedastic
            };
            Study::tabular(iv_data(9))
                .graph(iv_dag())
                .query(continuous_query())
                .identifier(IdentifierId::Iv)
                .estimator(WaldIv { bootstrap_replicates: 0, se_kind, ..WaldIv::new() })
                .refute(RefuteSuite::None)
        }
        "iv_2sls_analytic_ci_coverage" | "iv_2sls_hc1_heteroskedastic_ci_coverage" => {
            let se_kind = if test == "iv_2sls_hc1_heteroskedastic_ci_coverage" {
                AnalyticSeKind::Hc1
            } else {
                AnalyticSeKind::Homoskedastic
            };
            Study::tabular(iv_data(10))
                .graph(iv_dag())
                .query(continuous_query())
                .identifier(IdentifierId::Iv)
                .estimator(TwoStageLeastSquares {
                    bootstrap_replicates: 0,
                    se_kind,
                    ..TwoStageLeastSquares::new()
                })
                .refute(RefuteSuite::None)
        }
        "frontdoor_stacked_hc0_ci_coverage" | "frontdoor_stacked_hc1_ci_coverage" => {
            let se_kind = if test == "frontdoor_stacked_hc1_ci_coverage" {
                AnalyticSeKind::Hc1
            } else {
                AnalyticSeKind::Hc0
            };
            Study::tabular(frontdoor_data(12))
                .graph(frontdoor_dag())
                .query(continuous_query())
                .identifier(IdentifierId::Frontdoor)
                .estimator(FrontDoorTwoStage {
                    bootstrap_replicates: 0,
                    se_kind,
                    ..FrontDoorTwoStage::new()
                })
                .refute(RefuteSuite::None)
        }
        "frontdoor_functional_saturated_ci_coverage"
        | "frontdoor_functional_arm_linear_ci_coverage" => {
            let discrete = test == "frontdoor_functional_saturated_ci_coverage";
            Study::tabular(frontdoor_functional_data(14, discrete))
                .graph(frontdoor_dag())
                .query(continuous_query())
                .identifier(IdentifierId::Frontdoor)
                .estimator(EstimatorId::FrontDoorFunctional)
                .bootstrap_replicates(0)
                .refute(RefuteSuite::None)
        }
        "rd_sharp_analytic_ci_coverage" | "rd_sharp_hc1_heteroskedastic_ci_coverage" => {
            let se_kind = if test == "rd_sharp_hc1_heteroskedastic_ci_coverage" {
                AnalyticSeKind::Hc1
            } else {
                AnalyticSeKind::Homoskedastic
            };
            // The sharp design as a graph: r -> t -> y and r -> y.
            let mut design = Dag::with_variables(3);
            design.insert_directed(d(2), d(0)).unwrap();
            design.insert_directed(d(0), d(1)).unwrap();
            design.insert_directed(d(2), d(1)).unwrap();
            Study::tabular(rd_data(13))
                .graph(design)
                .query(ate_query())
                .identifier(IdentifierId::RdSharp)
                .estimator(EstimatorId::RdSharp)
                .rd_design(RdConfig::new(v(2), 0.0, 1.0).with_se_kind(se_kind))
                .bootstrap_replicates(0)
                .refute(RefuteSuite::None)
        }
        other => panic!("no facade configuration for {other}"),
    };
    let study = study.build().unwrap_or_else(|err| panic!("{test}: build {err}"));
    let result = study.run(&ctx).unwrap_or_else(|err| panic!("{test}: run {err}"));
    (study, result)
}

#[test]
fn estimator_level_records_key_the_facade_construction() {
    for case in CASES {
        let (study, result) = facade_result(case.test);
        let contract = study.inspect().unwrap();
        let reported = constructions(&contract, &result);
        let expected = case_for(case.test).construction();
        assert!(
            reported.iter().any(|(construction, _)| *construction == expected),
            "{}: the facade reports {:#?}, not the construction the record claims {expected:#?}",
            case.test,
            reported.iter().map(|(c, _)| c).collect::<Vec<_>>()
        );
    }
}
