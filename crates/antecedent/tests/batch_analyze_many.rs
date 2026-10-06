//! The batch (`analyze_many`) routes: shared cross-fitted AIPW nuisances, and
//! the typed boundary of what a batch covers.
//!
//! A batch shares cross-fitted AIPW nuisance fits between queries that would
//! fit the identical nuisance (same rows, treatment coding, adjustment design,
//! fold plan and learner options), and every shared result is bit-identical to
//! the per-query fit. A batch covers average effects (`estimate_many`,
//! `prepare`) and discrete joint `InterventionResponse` cells (`prepare_cells`);
//! any other query is refused with `route_not_supported`, naming `analyze`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "sharing a nuisance fit must not move a single bit of the estimate"
)]

use std::sync::Arc;

use antecedent::{BatchStudy, EstimatorId, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, DerivativeWeighting, ExecutionContext, Intervention, ResponseFunctional,
    ResponseQuery, StreamDomain, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const T1: u32 = 0;
const T2: u32 = 1;
const Y1: u32 = 2;
const Y2: u32 = 3;
const Z: u32 = 4;

/// Two binary treatments and two outcomes, all confounded by one `z`.
fn two_by_two(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xB5);
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
    let pairs = [("t1", t1), ("t2", t2), ("y1", y1), ("y2", y2), ("z", z)];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(5);
    for (from, to) in [(Z, T1), (Z, T2), (Z, Y1), (Z, Y2), (T1, Y1), (T1, Y2), (T2, Y1), (T2, Y2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn ate(treatment: u32, outcome: u32) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(treatment), VariableId::from_raw(outcome))
}

fn solo(data: &TabularData, graph: &Dag, query: &AverageEffectQuery, seed: u64) -> StudyResult {
    Study::tabular(data.clone())
        .graph(graph.clone())
        .query(query.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap()
}

fn shared_design_field<'a>(result: &'a StudyResult, key: &str) -> &'a str {
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

fn assert_bit_identical(batch: &StudyResult, solo: &StudyResult, what: &str) {
    assert_eq!(batch.estimate.ate.to_bits(), solo.estimate.ate.to_bits(), "{what}: ate");
    assert_eq!(
        batch.estimate.se_analytic.to_bits(),
        solo.estimate.se_analytic.to_bits(),
        "{what}: se_analytic"
    );
    let (Some(a), Some(b)) = (&batch.estimate.influence, &solo.estimate.influence) else {
        panic!("{what}: both runs publish the influence function");
    };
    assert_eq!(a.len(), b.len(), "{what}: influence length");
    assert!(a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()), "{what}: influence");
}

#[test]
fn batch_shares_matching_nuisances_and_matches_per_query_fits_bit_for_bit() {
    let (data, graph) = two_by_two(600, 41);
    // (t1, y1) and (t1, y2) share the t1 propensity; the repeated (t1, y1) also shares
    // its outcome regressions; (t2, y1) shares nothing with any other member.
    let queries = [ate(T1, Y1), ate(T1, Y2), ate(T2, Y1), ate(T1, Y1)];
    for threads in [1_u32, 4] {
        let mut ctx = ExecutionContext::for_tests(41);
        ctx.parallelism.max_threads = antecedent_core::NonZeroThreadCount::new(threads).unwrap();
        let results = BatchStudy::new(data.clone(), graph.clone())
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .estimate_many(&queries, &ctx)
            .unwrap();
        assert_eq!(results.len(), queries.len());
        for (i, (result, query)) in results.iter().zip(&queries).enumerate() {
            let alone = solo(&data, &graph, query, 41);
            assert_bit_identical(result, &alone, &format!("threads={threads} query {i}"));
        }
        let shares = |i: usize| {
            (
                shared_design_field(&results[i], "shares_propensity"),
                shared_design_field(&results[i], "shares_outcome_residualization"),
            )
        };
        assert_eq!(shares(0), ("true", "true"), "threads={threads}");
        assert_eq!(shares(1), ("true", "false"), "threads={threads}");
        assert_eq!(shares(2), ("false", "false"), "threads={threads}");
        assert_eq!(shares(3), ("true", "true"), "threads={threads}");
    }
}

#[test]
fn batch_never_shares_nuisances_across_different_seeds() {
    let (data, graph) = two_by_two(400, 43);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let batch = |seed: u64| {
        BatchStudy::new(data.clone(), graph.clone())
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .estimate_many(&queries, &ExecutionContext::for_tests(seed))
            .unwrap()
    };
    // Each batch is its own sharing scope: a second batch under another seed (another
    // fold plan) refits, so its members match their own per-query fits, not the first's.
    let first = batch(43);
    let second = batch(44);
    for (i, query) in queries.iter().enumerate() {
        assert_bit_identical(&first[i], &solo(&data, &graph, query, 43), "seed 43");
        assert_bit_identical(&second[i], &solo(&data, &graph, query, 44), "seed 44");
    }
    assert_ne!(first[0].estimate.ate.to_bits(), second[0].estimate.ate.to_bits());
}

#[test]
fn prepared_batch_estimate_shares_nuisances_and_matches_per_query_fits() {
    let (data, graph) = two_by_two(500, 47);
    let queries = [ate(T1, Y1), ate(T1, Y2)];
    let ctx = ExecutionContext::for_tests(47);
    let prepared = BatchStudy::new(data.clone(), graph.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&queries, &ctx)
        .unwrap();
    let results = prepared.estimate(&data, &ctx).unwrap();
    for (i, query) in queries.iter().enumerate() {
        assert_bit_identical(&results[i], &solo(&data, &graph, query, 47), "prepared");
        assert_eq!(shared_design_field(&results[i], "shares_propensity"), "true");
        assert_eq!(shared_design_field(&results[i], "shares_outcome_residualization"), "false");
    }
}

fn cells(interventions: Vec<Intervention>) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(Y1),
        interventions: Arc::from(interventions),
    })
}

fn set(variable: u32, value: f64) -> Intervention {
    Intervention::set(VariableId::from_raw(variable), Value::f64(value))
}

#[test]
fn prepare_cells_refuses_queries_outside_discrete_joint_cells_naming_analyze() {
    let (data, graph) = two_by_two(300, 53);
    let ctx = ExecutionContext::for_tests(53);
    let joint = cells(vec![set(T1, 1.0), set(T2, 1.0)]);
    let outside = [
        ("single intervention", cells(vec![set(T1, 1.0)])),
        (
            "average derivative",
            ResponseQuery::new(ResponseFunctional::AverageDerivative {
                outcome: VariableId::from_raw(Y1),
                treatment: VariableId::from_raw(T1),
                weighting: DerivativeWeighting::Observed,
            }),
        ),
    ];
    for (what, query) in outside {
        let err = BatchStudy::new(data.clone(), graph.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .prepare_cells(&[joint.clone(), query], &ctx)
            .expect_err(what);
        assert_eq!(err.reason_code(), Some("route_not_supported"), "{what}: {err}");
        assert!(err.to_string().contains("analyze"), "{what} names analyze: {err}");
    }
    // The covered family still prepares.
    BatchStudy::new(data, graph)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare_cells(&[joint, cells(vec![set(T1, 1.0), set(T2, 0.0)])], &ctx)
        .expect("discrete joint cells batch");
}
