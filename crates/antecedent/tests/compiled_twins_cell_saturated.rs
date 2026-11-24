//! The cell-saturated AIPW score fits are the same computation as the compiled joint-cell
//! response route.
//!
//! A study of a joint binary intervention response with the `cell.aipw` estimator fits its cell
//! scores with `CellSaturatedAipw` seeded by the run's master seed: the one-shot route through
//! `fit_scores_with_assignment` (no shared batch design, so no shared folds or design), the
//! retained prepared route through `fit_scores`, and a table that carries observation weights
//! through `fit_scores_sampling_weighted`. Each test fits the scores directly with the
//! estimand's adjustment set, summarizes them with the kernel summary the route reads, and
//! compares the score table, the point estimate, the standard error and the published interval
//! with the compiled result, bit for bit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "the bits are compared exactly and the fixtures build columns from small counts"
)]

use std::sync::Arc;

use antecedent::result::reported_se_interval_z;
use antecedent::{EstimatorId, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseQuery,
    ResponseUncertainty, StreamDomain, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{
    CellSaturatedAipw, CellScoreSpec, ScoreTable, SharedCellInputs, summarize_functional_with,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const ROWS: usize = 1_200;
const SEED: u64 = 431;
/// The requested cell `a = 1, d = 1`: bit 0 is `a`, bit 1 is `d`.
const REQUESTED_ARM: u32 = 3;

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("compiled-twins-cell-saturated".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .expect("spawn twin thread")
        .join()
        .expect("twin body panicked");
}

/// Two binary treatments confounded by `z`, with an interaction of 1.5 on `y`.
fn table() -> TabularData {
    let mut rng = ExecutionContext::for_tests(SEED).rng.stream_for(StreamDomain::Test, 0xCE);
    let (mut a, mut d, mut y, mut z) =
        (vec![0.0; ROWS], vec![0.0; ROWS], vec![0.0; ROWS], vec![0.0; ROWS]);
    for i in 0..ROWS {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        a[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-(0.5 * zi)).exp()));
        d[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-(-0.3 * zi)).exp()));
        y[i] = 1.5 * a[i] * d[i] + 0.4 * a[i] + 0.2 * zi + 0.3 * standard_normal(&mut rng);
    }
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("d", d.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap()
}

/// Row weights for the weighted population: a deterministic pattern of positive weights.
fn row_weights() -> Vec<f64> {
    (0..ROWS).map(|row| 0.5 + (row % 4) as f64 * 0.5).collect()
}

fn graph() -> Dag {
    let mut dag = Dag::with_variables(4);
    for (from, to) in [(3, 0), (3, 1), (3, 2), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    dag
}

fn query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
        ]),
    })
}

fn study(data: TabularData) -> antecedent::analysis::Study {
    Study::tabular(data)
        .graph(graph())
        .query(CausalQuery::Response(query()))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

const TREATMENTS: [VariableId; 2] = [VariableId::from_raw(0), VariableId::from_raw(1)];

/// The spec the route builds: the joint treatments, the outcome, the identified adjustment set
/// and the query's outcome functional.
fn spec<'a>(adjustment: &'a [VariableId], query: &'a ResponseQuery) -> CellScoreSpec<'a> {
    CellScoreSpec {
        treatments: &TREATMENTS,
        outcome: VariableId::from_raw(2),
        adjustment,
        functional: &query.outcome_functional,
        continuous: None,
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The direct summary of the requested cell: its mean and standard error under `weights`.
fn direct_level(table: &ScoreTable, weights: Option<&[f64]>) -> (f64, f64) {
    let (summary, _, _) = summarize_functional_with(table, weights, None).unwrap();
    let column = table
        .columns
        .iter()
        .position(|column| column.arm == REQUESTED_ARM)
        .expect("the requested cell has a score column");
    (summary.means[column], summary.covariance.se(column))
}

fn assert_same_table(label: &str, direct: &ScoreTable, compiled: &ScoreTable) {
    assert_eq!(direct.n_rows, compiled.n_rows, "{label}: rows");
    assert_eq!(direct.row_index, compiled.row_index, "{label}: row index");
    assert_eq!(direct.fold_ids, compiled.fold_ids, "{label}: folds");
    assert_eq!(direct.n_folds, compiled.n_folds, "{label}: fold count");
    assert_eq!(bits(&direct.scores), bits(&compiled.scores), "{label}: scores");
    assert_eq!(bits(&direct.propensities), bits(&compiled.propensities), "{label}: propensities");
    assert_eq!(
        direct.columns.iter().map(|column| column.arm).collect::<Vec<_>>(),
        compiled.columns.iter().map(|column| column.arm).collect::<Vec<_>>(),
        "{label}: score columns"
    );
    assert_eq!(direct.nuisance_provenance, compiled.nuisance_provenance, "{label}: provenance");
}

/// The compiled point, standard error and published interval are the direct ones.
fn assert_same_level(label: &str, direct: (f64, f64), compiled: &StudyResult) {
    let (value, se) = direct;
    let estimate = compiled.estimate().as_effect().expect("an effect estimate");
    assert_eq!(
        value.to_bits(),
        estimate.ate.to_bits(),
        "{label}: point {value} vs {}",
        estimate.ate
    );
    assert_eq!(se.to_bits(), estimate.se_analytic().to_bits(), "{label}: standard error");
    let response = compiled.response().expect("a response result");
    let ResponseUncertainty::Scalar { standard_error, lower, upper, .. } = &response.uncertainty
    else {
        panic!("{label}: a scalar response interval");
    };
    let z = reported_se_interval_z();
    assert_eq!(se.to_bits(), standard_error.to_bits(), "{label}: response standard error");
    assert_eq!((value - z * se).to_bits(), lower.to_bits(), "{label}: lower bound");
    assert_eq!((value + z * se).to_bits(), upper.to_bits(), "{label}: upper bound");
}

fn compiled_table(result: &StudyResult) -> &ScoreTable {
    result
        .estimate()
        .as_effect()
        .unwrap()
        .scores
        .table
        .as_ref()
        .expect("the cell route publishes its score table")
}

#[test]
fn one_shot_route_fits_with_assignment_and_equals_the_direct_fit() {
    on_large_stack(|| {
        let data = table();
        let ctx = ctx();
        let compiled = study(data.clone()).run(&ctx).unwrap();
        let query = query();
        let direct = CellSaturatedAipw::new()
            .with_fold_seed(ctx.rng.master_seed())
            .fit_scores_with_assignment(
                &data,
                &spec(&compiled.estimand.adjustment_set, &query),
                SharedCellInputs::default(),
            )
            .unwrap();
        assert_same_table("one-shot", &direct, compiled_table(&compiled));
        assert_same_level("one-shot", direct_level(&direct, None), &compiled);
    });
}

#[test]
fn prepared_route_fits_scores_and_equals_the_direct_fit() {
    on_large_stack(|| {
        let data = table();
        let ctx = ctx();
        let prepared = study(data.clone()).prepare(&ctx).unwrap();
        let compiled = prepared.estimate(&data, &ctx).unwrap();
        let query = query();
        let direct = CellSaturatedAipw::new()
            .with_fold_seed(ctx.rng.master_seed())
            .fit_scores(&data, &spec(&compiled.estimand.adjustment_set, &query))
            .unwrap();
        assert_same_table("prepared", &direct, compiled_table(&compiled));
        assert_same_table(
            "prepared retained",
            &direct,
            prepared.score_table().expect("the prepared route retains its scores"),
        );
        assert_same_level("prepared", direct_level(&direct, None), &compiled);
    });
}

#[test]
fn weighted_route_fits_the_sampling_weighted_scores_and_equals_the_direct_fit() {
    on_large_stack(|| {
        let weights = row_weights();
        let data = table().with_weights(Arc::from(weights.clone())).unwrap();
        let ctx = ctx();
        let compiled = study(data.clone()).run(&ctx).unwrap();
        let query = query();
        let (direct, _retained) = CellSaturatedAipw::new()
            .with_fold_seed(ctx.rng.master_seed())
            .fit_scores_sampling_weighted(
                &data,
                &spec(&compiled.estimand.adjustment_set, &query),
                SharedCellInputs::default(),
            )
            .unwrap();
        assert_same_table("weighted", &direct, compiled_table(&compiled));
        // The route summarizes with the table's own weights of the analysed rows.
        let used: Vec<f64> =
            direct.row_index.iter().map(|id| weights[usize::try_from(id.get()).unwrap()]).collect();
        assert_same_level("weighted", direct_level(&direct, Some(&used)), &compiled);
    });
}
