//! 1.10 repeated-sampling coverage of the `glm.adjustment` estimator
//! (`parity/estimate.toml` row `estimate.glm`) through the public Study API.
//!
//! Binary-outcome logistic law of `conformance/estimate/glm_adjustment`
//! (`estimate_conformance::glm_binary_scm`, columns `t, y, z`):
//! `z ~ N(0,1)`, `t | z ~ Bern(σ(−0.3 + 0.8z))`,
//! `y | t, z ~ Bern(σ(−0.5 + 1.2t + 0.8z))`. The outcome model is the fitted
//! logistic model, so the g-computation target is the population ATE
//! `E[σ(0.7 + 0.8z) − σ(−0.5 + 0.8z)]` (numerical integral below).
//!
//! The Study default (199 bootstrap replicates) reports the bootstrap-SE
//! interval; with `bootstrap_replicates(0)` the same fit reports the
//! delta-method analytic SE. Both constructions are scored from the same
//! replicate's data (replicate 0 checks that the two runs agree on the point
//! and on the analytic SE), at 0.95 and 0.90, each construction scored on the
//! execution that reports it. Every tally is keyed through [`keyed`], this
//! file's one emission point; the construction each record describes is read
//! from that execution's own contract rather than declared here.
//!
//! The file also measures the interval an ordinary default Frequentist
//! `AverageEffect` on a DAG reports (`linear.adjustment.ate`, the Study's
//! default 199 bootstrap replicates, `bootstrap_se`) on the
//! binary-treatment adjustment law `common::static_dgp::linear_ate_data`, the
//! same law the default Bayesian record is measured on.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

mod common;

use antecedent::{EstimatorId, RefuteSuite, Study, StudyBuilder, StudyResult};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{
    CoverageTally, RecordKey, gaussian, grid_n, n_sim, stream_seed, unit_uniform,
};
use common::calibration_bind::bind_all;
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, gate, normal_at, record_pair, scalar_normal_pair, skip_pair,
};
use common::static_dgp::linear_ate_data;

const N: usize = 800;

/// This file's single coverage-record emission point. The construction a
/// record describes is not declared here: it is read from the execution the
/// tally scores (`bind`), so the record can only name the interval the facade
/// reported.
fn keyed(
    test: &'static str,
    dgp: &'static str,
    interval_method: &'static str,
    level: f64,
) -> CoverageTally {
    CoverageTally::for_record(RecordKey { test, dgp, interval: interval_method }, level)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn glm_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        let (u_t, u_y) = (
            unit_uniform(stream_seed(seed, 2 * i as u64)),
            unit_uniform(stream_seed(seed, 2 * i as u64 + 1)),
        );
        t[i] = f64::from(u_t < sigmoid(-0.3 + 0.8 * z[i]));
        y[i] = f64::from(u_y < sigmoid(-0.5 + 1.2 * t[i] + 0.8 * z[i]));
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// `E[σ(0.7 + 0.8z) − σ(−0.5 + 0.8z)]`, `z ~ N(0,1)`, trapezoid on `[−12, 12]`.
fn glm_truth() -> f64 {
    let steps = 240_000;
    let h = 24.0 / f64::from(steps);
    let mut acc = 0.0;
    for k in 0..=steps {
        let z = -12.0 + h * f64::from(k);
        let w = if k == 0 || k == steps { 0.5 } else { 1.0 };
        let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        acc += w * (sigmoid(0.7 + 0.8 * z) - sigmoid(-0.5 + 0.8 * z)) * phi;
    }
    acc * h
}

/// The executed study beside its result: [`bind`] reads the construction from
/// the study's own contract, so a record cannot describe another one.
fn run(data: TabularData, bootstrap: Option<u32>, seed: u64) -> Option<(Study, StudyResult)> {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let mut builder = Study::tabular(data)
        .graph(dag)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(EstimatorId::GlmAdjustment)
        .refute(RefuteSuite::None);
    if let Some(b) = bootstrap {
        builder = builder.bootstrap_replicates(b);
    }
    let study = builder.build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn glm_adjustment_binary_outcome_nominal_coverage() {
    const TEST: &str = "glm_adjustment_binary_outcome_nominal_coverage";
    let truth = glm_truth();
    let mut bootstrap = [
        keyed(TEST, "glm_data", "bootstrap_se", REPORTED_LEVEL),
        keyed(TEST, "glm_data", "bootstrap_se", GATE_LEVEL),
    ];
    let mut analytic = [
        keyed(TEST, "glm_data", "analytic_se", REPORTED_LEVEL),
        keyed(TEST, "glm_data", "analytic_se", GATE_LEVEL),
    ];
    for rep in 0..u64::from(n_sim()) {
        let seed = stream_seed(0x110_0301, rep);
        let data = glm_data(grid_n(N), seed);
        let Some((study, result)) = run(data.clone(), None, seed) else {
            skip_pair(&mut bootstrap);
            skip_pair(&mut analytic);
            continue;
        };
        // The analytic interval is the one a bootstrap-free run reports, so it
        // is scored on that execution and keyed by it.
        let Some((bare_study, bare)) = run(data, Some(0), seed) else {
            skip_pair(&mut bootstrap);
            skip_pair(&mut analytic);
            continue;
        };
        let est = &result.estimate;
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("glm.adjustment"));
            assert_eq!(est.bootstrap_replicates_ok, Some(199), "the Study default bootstrap");
            assert_eq!(bare.estimate.ate, est.ate, "same point without bootstrap");
            assert_eq!(bare.estimate.se_analytic, est.se_analytic, "same analytic SE");
        }
        {
            let [reported, gated] = &mut bootstrap;
            bind_all(&mut [reported, gated], &study, &result);
            let [reported, gated] = &mut analytic;
            bind_all(&mut [reported, gated], &bare_study, &bare);
        }
        let se_boot = est.se_bootstrap.unwrap_or(f64::NAN);
        record_pair(
            &mut bootstrap,
            [normal_at(est.ate, se_boot, REPORTED_LEVEL), normal_at(est.ate, se_boot, GATE_LEVEL)],
            truth,
        );
        record_pair(
            &mut analytic,
            [
                normal_at(bare.estimate.ate, bare.estimate.se_analytic, REPORTED_LEVEL),
                normal_at(bare.estimate.ate, bare.estimate.se_analytic, GATE_LEVEL),
            ],
            truth,
        );
    }
    eprintln!("info glm.adjustment: population ATE truth {truth:.6}");
    let tallies = [bootstrap, analytic].concat();
    gate(&tallies, &[None, None, None, None]);
}

// ============================================ default Frequentist AverageEffect

const LINEAR_ATE_DATA: &str = "crates/antecedent/tests/common/static_dgp.rs::linear_ate_data";

/// Base rows of the default `AverageEffect` design (grid 250, 500, 1000).
const N_DEFAULT_ATE: usize = 500;

/// The facade's default Frequentist `AverageEffect` on a DAG: no estimator,
/// no bootstrap count and no interval method chosen by the caller, so the
/// study resolves `linear.adjustment.ate` and reports the bootstrap-SE
/// interval at its default replicate count. Scored at 0.95 (reported) and 0.90.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_dag_frequentist_default_nominal_coverage() {
    const TEST: &str = "average_effect_dag_frequentist_default_nominal_coverage";
    let mut tallies = [
        keyed(TEST, LINEAR_ATE_DATA, "bootstrap_se", REPORTED_LEVEL),
        keyed(TEST, LINEAR_ATE_DATA, "bootstrap_se", GATE_LEVEL),
    ];
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    for rep in 0..u64::from(n_sim()) {
        let seed = stream_seed(0x110_0310, rep);
        let study = Study::tabular(linear_ate_data(grid_n(N_DEFAULT_ATE), seed))
            .graph(dag.clone())
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .refute(RefuteSuite::None)
            .build()
            .expect("the default study builds");
        let Ok(result) = study.run(&ExecutionContext::for_tests(seed)) else {
            skip_pair(&mut tallies);
            continue;
        };
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("linear.adjustment.ate"));
            assert_eq!(
                result.estimate.bootstrap_replicates_ok,
                Some(StudyBuilder::OMITTED_BOOTSTRAP),
                "the Study default bootstrap"
            );
        }
        let [reported, gated] = &mut tallies;
        bind_all(&mut [reported, gated], &study, &result);
        record_pair(&mut tallies, scalar_normal_pair(&result), 2.0);
    }
    gate(&tallies, &[None, None]);
}
