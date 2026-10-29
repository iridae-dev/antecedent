//! Cheap and full validation of a counterfactual and of a nested natural direct effect on a
//! supplied or accepted DAG, against the closed forms of the equations that generated the data
//! (`conformance/estimate/cross_world_dag_validation`).
//!
//! Each effect is refuted by the placebo treatment (the permuted treatment removes the mechanism
//! the abducted noise is pushed through), subset stability and, under full, the placebo outcome and
//! the bootstrap stability of the whole procedure. The planted violations are a null outcome slope
//! for the counterfactual and an outcome error coupled to the mediator error for the nested
//! effect: the cross-world sensitivity refuter flags each and passes its clean twin.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "test fixtures: small exact indices and one coordinate walk per test"
)]

use std::sync::Arc;

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, NestedCounterfactualQuery,
    Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

/// Seeded standard-normal stream (xorshift64* uniforms, Box-Muller).
struct Normals(u64);

impl Normals {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cross_world_dag_validation/expected.json"
    ))
    .unwrap()
}

fn number(value: &serde_json::Value, path: &[&str]) -> f64 {
    path.iter().fold(value, |node, key| &node[*key]).as_f64().unwrap()
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn report<'a>(result: &'a StudyResult, refuter: &str) -> &'a antecedent_validate::RefutationReport {
    result.refutations.iter().find(|report| report.refuter.as_ref() == refuter).unwrap_or_else(
        || {
            let refused: Vec<String> = result
                .diagnostics
                .iter()
                .filter(|d| d.code.as_ref() == "refute.functional.refuter_refused")
                .map(|d| d.message.to_string())
                .collect();
            panic!("no `{refuter}` ran: {refused:?}")
        },
    )
}

fn inference(bayesian: bool) -> InferenceMode {
    if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))
    } else {
        InferenceMode::Frequentist
    }
}

fn dag(edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(3);
    for &(source, target) in edges {
        graph
            .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
            .unwrap();
    }
    graph
}

/// Columns `z, t, y` with `y = 1 + slope t + z` and a small deterministic wiggle.
fn outcome_table(slope: f64) -> TabularData {
    let n = 240_u32;
    let z: Vec<f64> = (0..n).map(|i| (f64::from(i) * 1.37).sin()).collect();
    let t: Vec<f64> = (0..n).zip(&z).map(|(i, z)| 0.6 * z + (f64::from(i) * 0.91).cos()).collect();
    let mut noise = Normals(20_241_029);
    let y: Vec<f64> =
        t.iter().zip(&z).map(|(t, z)| 1.0 + slope * t + z + 0.4 * noise.next()).collect();
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn counterfactual_builder(
    data: TabularData,
    accepted: bool,
    bayesian: bool,
    suite: RefuteSuite,
    active: f64,
    control: f64,
) -> antecedent::analysis::StudyBuilder {
    let graph = dag(&[(0, 1), (0, 2), (1, 2)]);
    let query =
        CounterfactualQuery::new(v(2), Arc::from([Intervention::set(v(1), Value::f64(active))]))
            .with_control_level(control);
    let base = Study::tabular(data);
    let base = if accepted { base.graph(AcceptedGraph::from(graph)) } else { base.graph(graph) };
    base.query(CausalQuery::Counterfactual(query))
        .inference(inference(bayesian))
        .refute(suite)
        .bootstrap_replicates(0)
}

/// Orthonormal centred columns (unit RMS), orthogonal to the centred `t` and to each other.
fn orthogonal_noise(n: usize, t: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let center = |v: &mut Vec<f64>| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.iter_mut().for_each(|x| *x -= mean);
    };
    let mut tc = t.to_vec();
    center(&mut tc);
    let mut gaussian = Normals(7_654_321);
    let mut a: Vec<f64> = (0..n).map(|_| gaussian.next()).collect();
    center(&mut a);
    let k = dot(&a, &tc) / dot(&tc, &tc);
    a.iter_mut().zip(&tc).for_each(|(x, y)| *x -= k * y);
    let mut b: Vec<f64> = (0..n).map(|_| gaussian.next()).collect();
    center(&mut b);
    let kt = dot(&b, &tc) / dot(&tc, &tc);
    let ka = dot(&b, &a) / dot(&a, &a);
    for i in 0..n {
        b[i] -= kt * tc[i] + ka * a[i];
    }
    let unit = |v: &mut Vec<f64>| {
        let rms = (dot(v, v) / v.len() as f64).sqrt();
        v.iter_mut().for_each(|x| *x /= rms);
    };
    unit(&mut a);
    unit(&mut b);
    (a, b)
}

/// Columns `t, m, y`: `m = 0.8 t + 0.5 a`, `y = 0.3 t + 1.2 m + 0.5 (c a + sqrt(1 - c^2) b)`.
fn mediation_table(coupling: f64) -> TabularData {
    let n = 600;
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64 + 0.3 * (i as f64 * 0.91).sin()).collect();
    let (a, b) = orthogonal_noise(n, &t);
    let m: Vec<f64> = t.iter().zip(&a).map(|(t, a)| 0.8 * t + 0.5 * a).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            0.3 * t[i]
                + 1.2 * m[i]
                + 0.5 * (coupling * a[i] + (1.0_f64 - coupling * coupling).sqrt() * b[i])
        })
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn nested_builder(
    data: TabularData,
    accepted: bool,
    bayesian: bool,
    suite: RefuteSuite,
) -> antecedent::analysis::StudyBuilder {
    let graph = dag(&[(0, 1), (0, 2), (1, 2)]);
    let base = Study::tabular(data);
    let base = if accepted { base.graph(AcceptedGraph::from(graph)) } else { base.graph(graph) };
    base.query(CausalQuery::NestedCounterfactual(
        NestedCounterfactualQuery::new(v(0), v(1), v(2)).unwrap(),
    ))
    .inference(inference(bayesian))
    .refute(suite)
    .bootstrap_replicates(0)
}

fn nde(result: &StudyResult) -> f64 {
    result.estimate().as_effect().unwrap().ate
}

#[test]
fn counterfactual_dag_cheap_and_full_validation_refutes_the_structural_slope() {
    let pin = pin();
    let truth = number(&pin, &["counterfactual", "mean_unit_effect"]);
    let null_truth = number(&pin, &["counterfactual", "null_slope_mean_unit_effect"]);
    let tolerance = number(&pin, &["counterfactual", "absolute_tolerance"]);
    let active = number(&pin, &["counterfactual", "active_level"]);
    let control = number(&pin, &["counterfactual", "control_level"]);
    let ctx = ExecutionContext::for_tests(24);
    for accepted in [false, true] {
        for bayesian in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let coordinate = format!("accepted={accepted} bayesian={bayesian} {suite:?}");
                let run = |slope: f64| {
                    counterfactual_builder(
                        outcome_table(slope),
                        accepted,
                        bayesian,
                        suite,
                        active,
                        control,
                    )
                    .build()
                    .unwrap()
                    .run(&ctx)
                    .unwrap_or_else(|error| panic!("{coordinate}: {error}"))
                };
                let clean = run(2.0);
                assert_eq!(clean.support_status(), Some(CellStatus::Licensed), "{coordinate}");
                let mean = clean.counterfactual().unwrap().mean_ite;
                assert!((mean - truth).abs() < tolerance, "{coordinate}: {mean} vs {truth}");
                // The validation never moves the answer.
                let plain = counterfactual_builder(
                    outcome_table(2.0),
                    accepted,
                    bayesian,
                    RefuteSuite::None,
                    active,
                    control,
                )
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap();
                assert!(
                    (plain.counterfactual().unwrap().mean_ite - mean).abs() < 1e-12,
                    "{coordinate}"
                );
                // The permuted treatment carries no slope: the refitted effect is centred on zero.
                let placebo = report(&clean, "target.placebo_treatment");
                assert!(placebo.passed, "{coordinate}");
                assert!(placebo.refuted_ate.abs() < 0.1, "{coordinate}: {}", placebo.refuted_ate);
                assert!(placebo.replicates >= 30, "{coordinate}");
                // The clean case is robust to the cross-world dependence; the planted null is not.
                assert!(report(&clean, "target.cross_world_sensitivity").passed, "{coordinate}");
                let null = run(0.0);
                let null_mean = null.counterfactual().unwrap().mean_ite;
                assert!((null_mean - null_truth).abs() < tolerance, "{coordinate}: {null_mean}");
                assert!(!report(&null, "target.cross_world_sensitivity").passed, "{coordinate}");
                if suite == RefuteSuite::Full {
                    let outcome = report(&clean, "target.placebo_outcome");
                    assert!(outcome.passed, "{coordinate}");
                    assert!(
                        outcome.refuted_ate.abs() < 0.1,
                        "{coordinate}: {}",
                        outcome.refuted_ate
                    );
                    assert!(outcome.replicates >= 60, "{coordinate}");
                    if !bayesian {
                        let bootstrap = report(&clean, "target.bootstrap_stability");
                        assert!(bootstrap.passed, "{coordinate}");
                        assert!((bootstrap.refuted_ate - truth).abs() < tolerance, "{coordinate}");
                    }
                    if bayesian {
                        // Gaussian disturbances: the posterior predictive check passes the clean case.
                        assert!(
                            report(&clean, "target.posterior_predictive").passed,
                            "{coordinate}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn nested_dag_validation_refutes_the_structural_direct_effect() {
    let pin = pin();
    let direct = number(&pin, &["nested", "structural_direct_effect"]);
    let total = number(&pin, &["nested", "total_effect"]);
    let read = number(&pin, &["nested", "shared_exogenous_read_under_coupling"]);
    let tolerance = number(&pin, &["nested", "absolute_tolerance"]);
    let coupling = number(&pin, &["nested", "error_coupling"]);
    let ctx = ExecutionContext::for_tests(25);
    for accepted in [false, true] {
        for bayesian in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let coordinate = format!("accepted={accepted} bayesian={bayesian} {suite:?}");
                let run = |coupling: f64| {
                    nested_builder(mediation_table(coupling), accepted, bayesian, suite)
                        .build()
                        .unwrap()
                        .run(&ctx)
                        .unwrap_or_else(|error| panic!("{coordinate}: {error}"))
                };
                let clean = run(0.0);
                assert_eq!(clean.support_status(), Some(CellStatus::Licensed), "{coordinate}");
                assert!((nde(&clean) - direct).abs() < tolerance, "{coordinate}: {}", nde(&clean));
                if suite == RefuteSuite::None {
                    assert!(clean.refutations.is_empty(), "{coordinate}");
                    continue;
                }
                // The planted coupling moves the shared-exogenous read to the pinned value.
                let planted = run(coupling);
                assert!(
                    (nde(&planted) - read).abs() < tolerance,
                    "{coordinate}: {}",
                    nde(&planted)
                );
                assert!(report(&clean, "target.cross_world_sensitivity").passed, "{coordinate}");
                assert!(!report(&planted, "target.cross_world_sensitivity").passed, "{coordinate}");
                // The treatment placebo removes the direct and indirect paths; the mediator
                // placebo returns the total effect.
                let placebo = report(&clean, "target.placebo_treatment");
                assert!(placebo.passed && placebo.refuted_ate.abs() < 0.1, "{coordinate}");
                let cross = report(&clean, "target.cross_world_placebo");
                assert!(cross.passed, "{coordinate}");
                assert!(
                    (cross.refuted_ate - total).abs() < 0.1,
                    "{coordinate}: {}",
                    cross.refuted_ate
                );
                if !bayesian {
                    let subset = report(&clean, "target.subset_stability");
                    assert!(subset.passed, "{coordinate}");
                    assert!(
                        (subset.refuted_ate - subset.original_ate).abs() < 0.05,
                        "{coordinate}"
                    );
                }
                if suite == RefuteSuite::Full {
                    let outcome = report(&clean, "target.placebo_outcome");
                    assert!(outcome.passed && outcome.refuted_ate.abs() < 0.1, "{coordinate}");
                    if !bayesian {
                        let bootstrap = report(&clean, "target.bootstrap_stability");
                        assert!(bootstrap.passed, "{coordinate}");
                        assert!(
                            (bootstrap.refuted_ate - bootstrap.original_ate).abs() < 0.05,
                            "{coordinate}"
                        );
                    }
                    if bayesian {
                        assert!(
                            report(&clean, "target.posterior_predictive").passed,
                            "{coordinate}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn counterfactual_dag_validation_executes_from_the_retained_plan_without_the_builder() {
    let ctx = ExecutionContext::for_tests(61);
    let data = outcome_table(2.0);
    for accepted in [false, true] {
        for bayesian in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let coordinate = format!("accepted={accepted} bayesian={bayesian} {suite:?}");
                let builder =
                    counterfactual_builder(data.clone(), accepted, bayesian, suite, 1.5, 0.0);
                let study = builder.clone().build().unwrap();
                let mut prepared = study.prepare(&ctx).unwrap();
                drop(builder);
                drop(study);
                let RetainedOperation::Counterfactual(plan) = prepared.retained_operation() else {
                    panic!("{coordinate}: retained checked counterfactual plan")
                };
                assert_eq!(plan.graph().node_count(), 3, "{coordinate}");
                let result = prepared.estimate(&data, &ctx).unwrap();
                assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{coordinate}");
                assert!(
                    (result.counterfactual().unwrap().mean_ite - 3.0).abs() < 0.05,
                    "{coordinate}"
                );
                assert!(
                    result
                        .refutations
                        .iter()
                        .any(|r| r.refuter.as_ref() == "target.placebo_treatment"),
                    "{coordinate}"
                );
            }
        }
    }
}

#[test]
fn nested_dag_validation_executes_from_the_retained_plan_without_the_builder() {
    let ctx = ExecutionContext::for_tests(62);
    let data = mediation_table(0.0);
    for accepted in [false, true] {
        for bayesian in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let coordinate = format!("accepted={accepted} bayesian={bayesian} {suite:?}");
                let builder = nested_builder(data.clone(), accepted, bayesian, suite);
                let study = builder.clone().build().unwrap();
                let mut prepared = study.prepare(&ctx).unwrap();
                drop(builder);
                drop(study);
                let RetainedOperation::NestedCounterfactual(plan) = prepared.retained_operation()
                else {
                    panic!("{coordinate}: retained checked nested counterfactual plan")
                };
                assert_eq!(plan.graph().node_count(), 3, "{coordinate}");
                let result = prepared.estimate(&data, &ctx).unwrap();
                assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{coordinate}");
                assert!((nde(&result) - 0.3).abs() < 0.02, "{coordinate}: {}", nde(&result));
                assert_eq!(
                    result
                        .refutations
                        .iter()
                        .any(|r| r.refuter.as_ref() == "target.placebo_treatment"),
                    suite != RefuteSuite::None,
                    "{coordinate}"
                );
            }
        }
    }
}


