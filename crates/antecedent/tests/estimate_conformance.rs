//! conformance: propensity IPW, IV/2SLS, front-door two-stage.
//!
//! Fixtures under `conformance/estimate/*` are clean-room synthetic SCMs generated inline
//! (deterministic from a fixed seed) — independent of any `pinned baseline` install or CSV fixture. Each
//! test checks `|estimate.ate - expected.true_effect| < expected.tolerance`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent::Study;
use antecedent::{EstimatorId, IdentifierId};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use serde_json::Value as JsonValue;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/estimate").join(name)
}

fn load_expected(name: &str) -> JsonValue {
    let raw = fs::read_to_string(fixture_dir(name).join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

/// Build `TabularData` from `(name, role, column)` triples; variable ids follow slice order.
fn tabular_data(vars: &[(&str, RoleHint, Vec<f64>)]) -> TabularData {
    let n = vars[0].2.len();
    let mut b = CausalSchemaBuilder::new();
    for (name, role, _) in vars {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(*role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let cols: Vec<OwnedColumn> = vars
        .iter()
        .enumerate()
        .map(|(i, (_, _, data))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(data.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TabularData::new(storage)
}

fn assert_recovers(result: &antecedent::StudyResult, expected: &JsonValue) {
    let true_effect = expected["true_effect"].as_f64().unwrap();
    let tolerance = expected["tolerance"].as_f64().unwrap();
    assert!(
        (result.estimate.ate - true_effect).abs() < tolerance,
        "ate={} expected true_effect={} tolerance={}",
        result.estimate.ate,
        true_effect,
        tolerance
    );
    assert_eq!(result.logical_plan.identifier.as_deref(), expected["identifier"].as_str());
    assert_eq!(result.logical_plan.estimator.as_deref(), expected["estimator"].as_str());
}

/// Largest accepted `|ln(SE_rust·√n_rust / SE_ref·√n_ref)|`: `ln 1.5`.
///
/// The recorded DoWhy SE (`reference.outputs.se`) comes from DoWhy's own
/// `n = 800` draw of the same SCM, so the two SEs are compared after `√n`
/// rescaling, never as raw numbers. `ln 1.5` absorbs the sampling noise of
/// two SE estimates from different draws (the DoWhy side is a bootstrap SE)
/// while still failing every constant-factor bug of 2 or more — variance
/// reported as SD, a dropped `√2`, a missing `n/(n−1)` squared, or an SE
/// computed on the wrong row count.
const SE_LOG_RATIO_TOLERANCE: f64 = 0.405_465_108_108_164_4;

/// Fixtures whose recorded DoWhy SE is not a reference for the Rust estimator.
///
/// `aipw`'s DoWhy block ran `backdoor.propensity_score_weighting`, a different
/// estimator (its point estimate is byte-identical to `propensity_ipw`'s).
/// `propensity_ipw`'s DoWhy SE (0.273 at n = 800) is about five times the
/// Monte Carlo sampling SD of any IPW estimator on the SCM this test draws
/// from (≈0.054 Hajek with a fitted logistic propensity, ≈0.11 Horvitz–Thompson
/// with the true one), so it cannot calibrate this SCM. Comparing either
/// would test DoWhy's recording, not this crate; the Rust SEs for these two
/// estimators are covered by the `antecedent-estimate` coverage gates instead.
const SE_NOT_COMPARABLE: [&str; 2] = ["propensity_ipw", "aipw"];

/// C-6: compare the reported SE against the fixture's DoWhy reference SE.
fn assert_reference_se(result: &antecedent::StudyResult, name: &str, n: usize) {
    assert!(!SE_NOT_COMPARABLE.contains(&name), "{name} has no comparable reference SE");
    let expected = load_expected(name);
    let outputs = &expected["reference"]["outputs"];
    let reference_se = outputs["se"].as_f64().expect("reference.outputs.se");
    let reference_n = outputs["n"].as_f64().expect("reference.outputs.n");
    let se = if result.estimate.se_analytic.is_finite() && result.estimate.se_analytic > 0.0 {
        result.estimate.se_analytic
    } else {
        result.estimate.se_bootstrap.expect("an analytic or bootstrap SE")
    };
    let log_ratio = (se * (n as f64).sqrt() / (reference_se * reference_n.sqrt())).ln();
    eprintln!(
        "se check {}: se={se} n={n} reference_se={reference_se} reference_n={reference_n} \
         log_ratio={log_ratio:.4}",
        expected["estimator"]
    );
    assert!(
        log_ratio.abs() <= SE_LOG_RATIO_TOLERANCE,
        "{}: SE {se} (n={n}) vs DoWhy reference {reference_se} (n={reference_n}): \
         |ln ratio| = {:.3} > ln 1.5 after sqrt(n) rescaling",
        expected["estimator"],
        log_ratio.abs()
    );
}

/// `Z ~ N(0,1)` confounder; `T ~ Bernoulli(sigmoid(-0.4 + 0.9 Z))`; `Y = 2T + Z + noise`.
/// True ATE = 2; a naive unadjusted contrast is biased by `Z`, exercising IPW.
fn propensity_ipw_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5051_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit = -0.4 + 0.9 * zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.4;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + noise;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap(); // z -> t
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap(); // z -> y
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // t -> y
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query)
}

#[test]
fn estimate_propensity_ipw_recovers_ate() {
    let expected = load_expected("propensity_ipw");
    let (data, graph, query) = propensity_ipw_scm(1200, 3);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(9);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
    // No SE comparison here: see `SE_NOT_COMPARABLE`.
    assert!(result.estimate.overlap_report.is_some(), "propensity.weighting must report overlap");
}

/// Binary instrument `Z`; unobserved confounder `U` (absent from the graph) with
/// `T = 0.6 Z + U + noise`, `Y = 2T + U + noise`. True structural effect = 2.
fn iv_2sls_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5052_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = (i % 2) as f64;
        let u = standard_normal(&mut rng);
        let ti = 0.6 * zi + u + 0.1 * standard_normal(&mut rng);
        let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
        z[i] = zi;
        t[i] = ti;
        y[i] = yi;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap(); // z -> t
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // t -> y
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    (data, dag, query)
}

#[test]
fn estimate_iv_2sls_recovers_structural_effect() {
    let expected = load_expected("iv_2sls");
    let (data, graph, query) = iv_2sls_scm(4000, 5);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(21);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
    assert_reference_se(&result, "iv_2sls", 4000);
}

/// `U -> T -> M -> Y` with `U -> Y` (no direct `T -> Y` edge; `U` unmeasured, absent from the
/// graph). `M = T + noise`, `Y = 2M + U + noise`. True mediated effect = `1 * 2 = 2`.
fn frontdoor_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5053_u64);
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let u = standard_normal(&mut rng);
        let ti = u + 0.1 * standard_normal(&mut rng);
        let mi = ti + 0.1 * standard_normal(&mut rng);
        let yi = 2.0 * mi + u + 0.1 * standard_normal(&mut rng);
        t[i] = ti;
        m[i] = mi;
        y[i] = yi;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("m", RoleHint::Context, m),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap(); // t -> m
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap(); // m -> y
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    (data, dag, query)
}

#[test]
fn estimate_frontdoor_two_stage_recovers_mediated_effect() {
    let expected = load_expected("frontdoor");
    let (data, graph, query) = frontdoor_scm(4000, 1);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(41);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
    assert_reference_se(&result, "frontdoor", 4000);
}

fn run_static(
    name: &str,
    data: TabularData,
    graph: Dag,
    query: AverageEffectQuery,
    seed: u64,
) -> antecedent::StudyResult {
    let expected = load_expected(name);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(20)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
    result
}

#[test]
fn estimate_propensity_matching_recovers_att() {
    let (data, graph, mut query) = propensity_ipw_scm(1500, 11);
    query = query.with_target_population(antecedent_core::TargetPopulation::Treated);
    run_static("propensity_matching", data, graph, query, 12);
}

#[test]
fn estimate_propensity_stratification_recovers_ate() {
    let (data, graph, query) = propensity_ipw_scm(1500, 13);
    run_static("propensity_stratification", data, graph, query, 14);
}

#[test]
fn estimate_distance_matching_recovers_att() {
    let (data, graph, mut query) = propensity_ipw_scm(1500, 15);
    query = query.with_target_population(antecedent_core::TargetPopulation::Treated);
    run_static("distance_matching", data, graph, query, 16);
}

#[test]
fn estimate_aipw_recovers_ate() {
    let (data, graph, query) = propensity_ipw_scm(1500, 17);
    // No SE comparison here: see `SE_NOT_COMPARABLE`.
    run_static("aipw", data, graph, query, 18);
}

#[test]
fn estimate_efficient_backdoor_ipw_recovers_ate() {
    let (data, graph, query) = propensity_ipw_scm(1500, 19);
    run_static("efficient_backdoor", data, graph, query, 20);
}

#[test]
fn estimate_iv_wald_recovers_structural_effect() {
    let (data, graph, query) = iv_2sls_scm(4000, 21);
    let result = run_static("iv_wald", data, graph, query, 22);
    assert_reference_se(&result, "iv_wald", 4000);
}

/// Binary outcome logistic SCM: `Y ~ Bern(sigmoid(-0.5 + 1.2 T + 0.8 Z))` with confounded T.
fn glm_binary_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5054_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit_t = -0.3 + 0.8 * zi;
        let pt = 1.0 / (1.0 + (-logit_t).exp());
        let ti = if rng.next_f64() < pt { 1.0 } else { 0.0 };
        let logit_y = -0.5 + 1.2 * ti + 0.8 * zi;
        let py = 1.0 / (1.0 + (-logit_y).exp());
        let yi = if rng.next_f64() < py { 1.0 } else { 0.0 };
        z[i] = zi;
        t[i] = ti;
        y[i] = yi;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query)
}

#[test]
fn estimate_glm_adjustment_recovers_positive_ate() {
    // Monte Carlo: logistic g-comp ATE is positive and typically ~0.2–0.3 under this SCM.
    let expected = load_expected("glm_adjustment");
    let (data, graph, query) = glm_binary_scm(2000, 23);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(20)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(24);
    let result = analysis.run(&ctx).unwrap();
    assert!(result.estimate.ate > 0.05, "ate={}", result.estimate.ate);
    assert!(
        (result.estimate.ate - expected["true_effect"].as_f64().unwrap()).abs()
            < expected["tolerance"].as_f64().unwrap(),
        "ate={}",
        result.estimate.ate
    );
}

/// Sharp RD: running variable R, T = 1{R >= 0}, Y = 3T + 0.5 R + noise.
fn rd_scm(n: usize, seed: u64) -> (TabularData, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5055_u64);
    let mut r = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let ri = standard_normal(&mut rng);
        let ti = if ri >= 0.0 { 1.0 } else { 0.0 };
        let yi = 3.0 * ti + 0.5 * ri + 0.2 * standard_normal(&mut rng);
        r[i] = ri;
        t[i] = ti;
        y[i] = yi;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("r", RoleHint::Context, r),
    ]);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, query)
}

#[test]
fn estimate_rd_sharp_recovers_jump() {
    let expected = load_expected("rd_sharp");
    let (data, query) = rd_scm(3000, 25);
    // Synthetic empty DAG; RD path does not use graph identification.
    let graph = Dag::with_variables(3);
    let analysis = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_config(VariableId::from_raw(2), 0.0, 1.5)
        .bootstrap_replicates(20)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(26);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);

    // Sharp RD is the documented identify-per-click exception on the prepared
    // handle: prepare() stores no identification cache for it.
    let prepared = analysis.prepare(&ctx).unwrap();
    let click = prepared.estimate(&data, &ctx).unwrap();
    assert!(
        !click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "sharp RD must not claim identification reuse"
    );
    assert_recovers(&click, &expected);
}
