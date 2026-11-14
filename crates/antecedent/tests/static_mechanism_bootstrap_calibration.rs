//! Repeated-sampling coverage of the mechanism-refit row bootstrap on a given DAG over static
//! (tabular) data: the per-unit counterfactual's mean effect and the anomaly attribution's mean
//! score, each under `.bootstrap_replicates(n)`.
//!
//! Both publish `estimate ± z·SE` from the bootstrap standard error of a whole-procedure
//! multinomial row bootstrap (every mechanism is refitted under each replicate's row counts).
//! Static data has no block structure, so the runtime reports the interval as `bootstrap_se`;
//! the driver (`common/grid_cell.rs`) finds the interval by that name among the constructions the
//! execution reports and panics if the runtime reports another.
//!
//! A test measures one coordinate on an explicit DAG. The accepted twin of that DAG and each
//! validation mode (none, cheap, full) are sibling executions: the driver measures the first
//! listed execution on every replicate and requires every sibling to report its interval bit for
//! bit before recording it under its own runtime match key.
//!
//! Truths are the exact population values of the reported functional:
//!
//! * Counterfactual: the data law is `y = 0.5 + 1.5a + z + e` with `a, z, w, e` independent
//!   standard normals (`w` is an unused covariate). `do(a = 1)` against `do(a = 0)` shifts `y`
//!   by exactly the coefficient on `a`, so the mean effect is 1.5 for every unit.
//! * Anomaly attribution: the data law is `y = 1 + 2z + 0.7e` with `z, e` standard normal, so
//!   `y ~ N(1, 4 + 0.49)`. Scoring `y` against a fixed reference with that location and scale
//!   makes the standardized deviation standard normal, the two-sided tail probability
//!   `U = 2Φ(−|z|)` uniform on (0, 1), and the per-unit score `−log U` a unit exponential
//!   variable, whose mean is exactly 1.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "test scaffolding compares exact constants and indexes rows with small integers"
)]

mod common;
#[path = "common/grid_cell.rs"]
mod grid_cell;

use std::sync::Arc;

use antecedent::{AcceptedGraph, InferenceMode, RefuteSuite, Study, StudyBuilder, StudyResult};
use antecedent_core::{
    AnomalyAttributionQuery, AnomalyReference, CausalQuery, CounterfactualQuery, ExecutionContext,
    Intervention, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{gaussian, grid_n, stream_seed};
use common::reported::scalar_normal_pair;
use grid_cell::{Cell, Target, measure};

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn table(columns: &[&str], values: &[Vec<f64>]) -> TabularData {
    TabularData::from_f64_columns(
        columns.iter().zip(values).map(|(name, col)| (*name, col.as_slice())),
    )
    .unwrap()
}

fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(nodes);
    for &(from, to) in edges {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph
}

/// How an execution is given its (single) structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Via {
    Explicit,
    Accepted,
}

use Via::{Accepted, Explicit};

/// One execution of a coordinate: its structure route and validation mode.
type Variant = (Via, RefuteSuite);

/// The execution grid of a coordinate: explicit and accepted structure under every validation
/// mode, the validation-none explicit execution first (it is the base measured on every
/// replicate).
const VARIANTS: [Variant; 6] = [
    (Explicit, RefuteSuite::None),
    (Accepted, RefuteSuite::None),
    (Explicit, RefuteSuite::Cheap),
    (Accepted, RefuteSuite::Cheap),
    (Explicit, RefuteSuite::Full),
    (Accepted, RefuteSuite::Full),
];

fn attach(builder: StudyBuilder, graph: &Dag, via: Via) -> StudyBuilder {
    match via {
        Explicit => builder.graph(graph.clone()),
        Accepted => builder.graph(AcceptedGraph::from(graph.clone())),
    }
}

fn execute(test: &str, builder: StudyBuilder, variant: Variant, seed: u64) -> (Study, StudyResult) {
    let (via, suite) = variant;
    let study =
        builder.refute(suite).build().unwrap_or_else(|e| panic!("{test} {via:?} {suite:?}: {e}"));
    let result = study
        .run(&ExecutionContext::for_tests(seed))
        .unwrap_or_else(|e| panic!("{test} {via:?} {suite:?}: {e}"));
    (study, result)
}

// ============================================================= counterfactual

/// The outcome's coefficient on `a`: the mean effect of `do(a = 1)` against `do(a = 0)`.
const CF_EFFECT: f64 = 1.5;
/// Rows of a counterfactual design.
const N_COUNTERFACTUAL: usize = 500;
/// Row-bootstrap replicates behind the standard error.
const CF_REPLICATES: u32 = 100;

/// Columns `a, z, w, y`: `a, z, w ~ N(0,1)` independent, `y = 0.5 + 1.5a + z + e`.
fn counterfactual_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut cols: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let (a, z, w) = (g(), g(), g());
        cols[0].push(a);
        cols[1].push(z);
        cols[2].push(w);
        cols[3].push(0.5 + CF_EFFECT * a + z + g());
    }
    table(&["a", "z", "w", "y"], &cols)
}

/// Measure the frequentist counterfactual on the explicit DAG `a -> y <- z` (with `w` isolated).
fn counterfactual(test: &'static str, family: u64) {
    let graph = dag(4, &[(0, 3), (1, 3)]);
    let cell = Cell {
        test,
        dgp: "counterfactual_data",
        interval: "bootstrap_se",
        estimator: "gcm.fit",
        variants: VARIANTS.len(),
    };
    let query = CausalQuery::Counterfactual(
        CounterfactualQuery::new(v(3), Arc::from([Intervention::set(v(0), Value::f64(1.0))]))
            .with_control_level(0.0),
    );
    measure(
        cell,
        &Target::scalar(CF_EFFECT),
        |rep, variant| {
            let seed = stream_seed(family, rep);
            let data = counterfactual_data(grid_n(N_COUNTERFACTUAL), seed);
            let builder = attach(Study::tabular(data), &graph, VARIANTS[variant].0)
                .query(query.clone())
                .inference(InferenceMode::Frequentist)
                .bootstrap_replicates(CF_REPLICATES);
            execute(test, builder, VARIANTS[variant], seed)
        },
        |result| vec![scalar_normal_pair(result)],
    );
}

// ================================================================ attribution

/// Residual scale of the anomaly target.
const NOISE: f64 = 0.7;
/// The anomaly target's intercept: the fixed reference's center.
const CENTER: f64 = 1.0;
/// The anomaly mean score against the target's true location and scale: the mean of a unit
/// exponential variable.
const ANOMALY_TRUTH: f64 = 1.0;
/// Rows of an anomaly design.
const N_ANOMALY: usize = 1000;
/// Row-bootstrap replicates behind the standard error.
const ATTRIBUTION_REPLICATES: u32 = 200;

/// Columns `z, y`: `y = 1 + 2z + 0.7e`.
fn chain_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut cols: Vec<Vec<f64>> = (0..2).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let z = g();
        cols[0].push(z);
        cols[1].push(CENTER + 2.0 * z + NOISE * g());
    }
    table(&["z", "y"], &cols)
}

/// Exact variance of the target's marginal law: `4·Var z` plus the residual.
fn chain_target_variance() -> f64 {
    4.0 + NOISE * NOISE
}

/// Measure the frequentist anomaly attribution on the explicit DAG `z -> y`.
fn anomaly_attribution(test: &'static str, family: u64) {
    let graph = dag(2, &[(0, 1)]);
    let cell = Cell {
        test,
        dgp: "attribution_chain",
        interval: "bootstrap_se",
        estimator: "gcm.fit",
        variants: VARIANTS.len(),
    };
    let query = CausalQuery::AnomalyAttribution(
        AnomalyAttributionQuery::new([v(1)], 1_000_000)
            .with_reference(AnomalyReference::fixed(CENTER, chain_target_variance().sqrt())),
    );
    measure(
        cell,
        &Target::scalar(ANOMALY_TRUTH),
        |rep, variant| {
            let seed = stream_seed(family, rep);
            let data = chain_data(grid_n(N_ANOMALY), seed);
            let builder = attach(Study::tabular(data), &graph, VARIANTS[variant].0)
                .query(query.clone())
                .inference(InferenceMode::Frequentist)
                .bootstrap_replicates(ATTRIBUTION_REPLICATES);
            execute(test, builder, VARIANTS[variant], seed)
        },
        |result| vec![scalar_normal_pair(result)],
    );
}

// ===================================================================== tests

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn counterfactual_dag_explicit_frequentist_bootstrap_coverage() {
    counterfactual("counterfactual_dag_explicit_frequentist_bootstrap_coverage", 0x5342_0001);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn anomaly_attribution_dag_explicit_frequentist_bootstrap_coverage() {
    anomaly_attribution(
        "anomaly_attribution_dag_explicit_frequentist_bootstrap_coverage",
        0x5342_0002,
    );
}
