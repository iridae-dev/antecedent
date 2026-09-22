//! Coverage of CPDAG class graph-posterior ATE intervals.
//!
//! Two identified CPDAG posterior atoms share an empty backdoor adjustment. Neither
//! atom has an edge from `Z` into `T`, so the DGP is one in which that is true:
//! `Y = 2T + 2Z ± 0.2` with `T` and `Z` independent fair draws. The empty-adjustment
//! contrast is then the causal effect 2.0 (a `Z -> T` edge would make it the
//! confounded contrast 2 + 2·(P(Z=1|T=1) − P(Z=1|T=0)), which is not the ATE),
//! and the scalar mixture carries a joint influence-function SE.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use antecedent::discovery::GraphPosterior;
use antecedent::{InferenceMode, RefuteSuite, StructuralAggregationPolicy, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{GraphPosteriorAtomKind, adjacency_mask_from_cpdag};
use antecedent_graph::{Cpdag, Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{
    CoverageTally, REPORTED_LEVEL, RecordKey, Z90, Z95, grid_n, n_sim, normal_interval,
};
use common::calibration_bind::bind_all;

const N: usize = 320;
const LEVEL: f64 = 0.9;
const WEIGHTS: [f64; 2] = [0.6, 0.4];
const TRUTH: f64 = 2.0;

const SEED_STRIDE: u64 = 7_919;
const SEED_BASE: u64 = 0x110C_0000;

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn oriented(from_z: bool) -> Cpdag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(d(0), d(1)).unwrap();
    if from_z {
        g.insert_directed(d(2), d(0)).unwrap();
        g.insert_directed(d(2), d(1)).unwrap();
    }
    Cpdag::from_dag(&g)
}

fn outcome_parent_cpdag() -> Cpdag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    Cpdag::from_dag(&g)
}

fn draw_data(seed: u64) -> TabularData {
    let mut unif = common::calibration::uniform_from_state(seed | 1);
    let rows = grid_n(N);
    let (mut t, mut y, mut z) =
        (Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let zi = f64::from(u8::from(unif() < 0.5));
        let ti = f64::from(u8::from(unif() < 0.5));
        let epsilon = if unif() < 0.5 { -0.2 } else { 0.2 };
        t.push(ti);
        z.push(zi);
        y.push(2.0 * ti + 2.0 * zi + epsilon);
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn class_posterior() -> GraphPosterior {
    let masks = [
        adjacency_mask_from_cpdag(&oriented(false)).unwrap(),
        adjacency_mask_from_cpdag(&outcome_parent_cpdag()).unwrap(),
    ];
    GraphPosterior::new(
        3,
        WEIGHTS.to_vec(),
        masks.to_vec(),
        vec![0.0; 9],
        vec![0.0; 9],
        1.0,
        InferenceDiagnostics::analytic("v110_class_posterior"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
}

fn run(seed: u64) -> (Study, StudyResult) {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let study = Study::tabular(draw_data(seed))
        .graph_posterior(class_posterior())
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(seed)).unwrap();
    (study, result)
}

fn assert_shape(result: &StudyResult) {
    assert!(result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
            && d.message.contains(StructuralAggregationPolicy::SameEstimandWeightedMean.as_str())
    }));
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.graph_posterior.joint_if_se"),
        "class posterior must publish joint IF SE"
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn class_posterior_frequentist_ate_joint_if_nominal_90_coverage() {
    let test = "class_posterior_frequentist_ate_joint_if_nominal_90_coverage";
    let key = RecordKey { test, dgp: "draw_data", interval: "analytic_se" };
    let mut tally = CoverageTally::for_record(key, LEVEL);
    let mut reported = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    for r in 0..n_sim() {
        let seed = SEED_BASE + u64::from(r) * SEED_STRIDE;
        let (study, result) = run(seed);
        if r == 0 {
            assert_shape(&result);
        }
        bind_all(&mut [&mut tally, &mut reported], &study, &result);
        tally.record(
            normal_interval(result.estimate.ate, Some(result.estimate.se_analytic), Z90),
            TRUTH,
        );
        reported.record(
            normal_interval(result.estimate.ate, Some(result.estimate.se_analytic), Z95),
            TRUTH,
        );
    }
    tally.assert();
    reported.emit();
}
