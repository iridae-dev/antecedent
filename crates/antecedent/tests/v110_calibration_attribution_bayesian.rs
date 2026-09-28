//! Repeated-sampling coverage of the Bayesian change-attribution coordinate:
//! the modular Dirichlet row-weight bootstrap of a two-population mean shift on
//! an explicit Dag.
//!
//! The test scores exactly the posterior-quantile credible interval the study
//! reports by default at 0.95, and the same construction at 0.90 re-derived from
//! the retained draws (`common::reported::posterior_pair`). Every tally is keyed
//! through [`keyed`], this file's one emission point: the key names only the
//! emitting test, its DGP and the interval method it scores. The construction
//! behind a record (query, graph class, estimator, inference, identification) is
//! never declared here — it is read from the runtime by binding every scored
//! replicate's execution with `common::calibration_bind`.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::doc_markdown)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AllocationMethod, CausalQuery, ChangeAttributionQuery, ExecutionContext, PopulationSelector,
    ShapleyConfig, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{
    CoverageTally, RecordKey, gaussian, grid_n, map_replicates, n_sim, stream_seed,
};
use common::calibration_bind::{bind_all, constructions};
use common::reported::{GATE_LEVEL, REPORTED_LEVEL, gate, posterior_pair, record_pair, skip_pair};

// ---------------------------------------------------------------- emission

/// The provenance of one attribution coordinate: the DGP its replicates are
/// drawn from and the estimator the plan must select. Everything else about the
/// record comes from the runtime.
#[derive(Clone, Copy)]
struct Cell {
    estimator: &'static str,
    dgp: &'static str,
}

/// This file's single coverage-record emission point. `test` is the name of the
/// `#[test] fn` that emits the record; the coordinate scores the
/// posterior-quantile credible interval the facade reports.
fn keyed(test: &'static str, cell: Cell, level: f64) -> CoverageTally {
    CoverageTally::for_record(
        RecordKey { test, dgp: cell.dgp, interval: "posterior_quantile" },
        level,
    )
}

/// The reported-level and gate-level tallies of one coordinate, scored on the
/// same replicates. The two records differ by level, so they need no label.
fn keyed_pair(test: &'static str, cell: Cell) -> [CoverageTally; 2] {
    [keyed(test, cell, REPORTED_LEVEL), keyed(test, cell, GATE_LEVEL)]
}

/// Bind one replicate's execution to both levels' tallies, as `record_pair`
/// scores both from that execution.
fn bind_pair(tallies: &mut [CoverageTally; 2], study: &Study, result: &StudyResult) {
    let [reported, gated] = tallies;
    bind_all(&mut [reported, gated], study, result);
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// The interval the coverage test scores must be the posterior quantile the
/// facade reports, or a later run would score a construction the study never
/// published.
fn assert_posterior_quantile(study: &Study, result: &StudyResult) {
    let contract = study.inspect().expect("inspect");
    let methods: Vec<String> = constructions(&contract, result)
        .into_iter()
        .map(|(construction, _)| construction.interval_method)
        .collect();
    assert!(
        methods.iter().any(|m| m == "posterior_quantile"),
        "the execution reported {methods:?}, not posterior_quantile"
    );
}

// ============================================================= change attribution

/// Structural coefficients of the two-period linear-Gaussian chain.
///
/// Baseline population (rows `0..n`): `x ~ N(0,1)`, `y = A1 + B·x + ε`.
/// Comparison population (rows `n..2n`): `x ~ N(0,1)` (identical law), `y = A2 +
/// B·x + ε`. Noise `ε ~ N(0,1)` is independent everywhere.
const A1: f64 = 1.0;
const A2: f64 = 6.0;
const B: f64 = 2.0;
const N_DRAWS: usize = 1_000;

/// Rows per population; two populations are stacked into `2·POP` rows.
const POP: usize = 600;

fn pop() -> usize {
    grid_n(POP)
}

/// Two stacked populations of `pop()` rows each, columns `x, y`. Baseline rows
/// carry intercept [`A1`], comparison rows carry [`A2`]; both share the slope
/// [`B`] and the same `x ~ N(0,1)` law, so any change in `E[Y]` is exactly the
/// intercept shift. `ε ~ N(0,1)` is drawn fresh for every row.
fn change_data(seed: u64) -> TabularData {
    let n = pop();
    let total = 2 * n;
    let mut g = gaussian(seed);
    let mut x = vec![0.0; total];
    let mut y = vec![0.0; total];
    for i in 0..total {
        let xi = g();
        let noise = g();
        let intercept = if i < n { A1 } else { A2 };
        x[i] = xi;
        y[i] = intercept + B * xi + noise;
    }
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

/// The study and its result, so a scored replicate can be bound to its tally.
/// The chain `x -> y` is the explicit Dag; the change-attribution query
/// contrasts the two stacked time ranges under a modular Dirichlet row-weight
/// bootstrap (exact-Shapley allocation), which the runtime routes to
/// `gcm.attribution.bayesian`.
fn run_change(data: TabularData, seed: u64) -> Option<(Study, StudyResult)> {
    let n = pop();
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ChangeAttributionQuery::new(
        v(1),
        PopulationSelector::TimeRange { start: 0, end: n },
        PopulationSelector::TimeRange { start: n, end: 2 * n },
    )
    .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
    let study = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::ChangeAttribution(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(N_DRAWS)))
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

const CHANGE_CELL: Cell = Cell { estimator: "gcm.attribution.bayesian", dgp: "change_data" };

/// Modular Dirichlet row-weight bootstrap of the two-population mean shift.
///
/// The estimand is `total_change = ΔE[Y] = E[Y_comp] − E[Y_base]`. Under the
/// linear-Gaussian chain the pushforward mean equals the sample mean, so
/// `total_change = (A2 + B·E[x]) − (A1 + B·E[x]) = A2 − A1`; the two `x`-laws
/// are identical (`N(0,1)`) so the slope term cancels exactly. Truth = `A2 −
/// A1 = 5.0`. The Bayesian bootstrap of a difference of two independent-
/// population means is first-order Normal, so `posterior_quantile` coverage is
/// asymptotically calibrated (n ≥ 600 per population).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn change_attribution_dag_bayesian_shared_dirichlet_nominal_coverage() {
    // total_change = A2 − A1 (the x-laws are identical, so B·E[x] cancels).
    let truth = A2 - A1;
    let mut tallies = keyed_pair(
        "change_attribution_dag_bayesian_shared_dirichlet_nominal_coverage",
        CHANGE_CELL,
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0301, rep);
        let (study, result) = run_change(change_data(seed), seed)?;
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some(CHANGE_CELL.estimator));
            assert_posterior_quantile(&study, &result);
            assert!(
                posterior_pair(&result, 0)[0].is_some(),
                "the change attribution must publish its posterior interval"
            );
        }
        Some((study, result))
    });
    for scored in &runs {
        let Some((study, result)) = scored else {
            skip_pair(&mut tallies);
            continue;
        };
        bind_pair(&mut tallies, study, result);
        record_pair(&mut tallies, posterior_pair(result, 0), truth);
    }
    eprintln!("info change_attribution: total_change truth {truth:.6}");
    gate(&tallies, &[None, None]);
}
