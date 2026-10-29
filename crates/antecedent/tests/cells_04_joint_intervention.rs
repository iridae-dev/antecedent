//! The joint two-treatment intervention contrast (cell AIPW) against declarations it must honour
//! or refuse.
//!
//! The balanced design `t1, t2` in {0, 1} with `y = 1.5 + 2 t1 + 3 t2 + 4 t1 t2 + 0.05 sin(0.31 i)`
//! has the saturated cell means as its answer: the requested cell (1, 1) is 10.5 up to the small
//! deterministic wiggle, and the reference below is the plain mean of the rows of that cell,
//! written out from the definition, never read from the library. Each test runs the route where
//! it works against that answer and then gives it the declaration it cannot read:
//!
//! - a latent-variable or equivalence-class graph: a tiered `CoDetermined` closure runs and equals
//!   the stratified standardization written out from the definition, and an explicit ADMG or
//!   CPDAG of the cell AIPW route is refused instead of read as a DAG;
//! - missing outcomes follow the default complete-case policy, equal to the fit on the remaining
//!   rows, an explicit error policy refuses, and the policies the route cannot pool (multiple
//!   imputation, inverse probability) are refused with `missing_policy_not_applicable`;
//! - observation weights on the table are refused with `sampling_weights.route_unsupported` on a
//!   study and on a batch, never estimated as unweighted rows;
//! - an unobserved cell and a non-binary level are refused instead of extrapolated;
//! - a declared clustered panel or dependent series with the bootstrap off is refused, and runs
//!   under independent subject trajectories.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    BatchStudy, CausalError, EstimatorId, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, DeclaredSamplingDesign, ExecutionContext, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::{ImputationSpec, MissingDeclaration, MissingPolicy, TabularData};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, TieredBackground, WithinTier};

fn on_large_stack(name: &str, body: fn()) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn level_query(first: f64, second: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(0), Value::f64(first)),
            Intervention::set(VariableId::from_raw(1), Value::f64(second)),
        ]),
    })
}

const ROWS: usize = 1200;

fn cell_of(row: usize) -> (f64, f64) {
    ((row % 2) as f64, ((row / 2) % 2) as f64)
}

fn outcome_of(row: usize) -> f64 {
    let (a, b) = cell_of(row);
    1.5 + 2.0 * a + 3.0 * b + 4.0 * a * b + 0.05 * (row as f64 * 0.31).sin()
}

/// The balanced table; `missing` blanks the outcome of one row and `without_cell` leaves out the
/// rows of the cell (1, 1).
fn table(missing: Option<usize>, without_cell: bool) -> TabularData {
    let (mut first, mut second, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for row in 0..ROWS {
        let (a, b) = cell_of(row);
        if without_cell && a == 1.0 && b == 1.0 {
            continue;
        }
        first.push(a);
        second.push(b);
        outcome.push(if missing == Some(row) { f64::NAN } else { outcome_of(row) });
    }
    TabularData::from_f64_columns([
        ("t1", first.as_slice()),
        ("t2", second.as_slice()),
        ("y", outcome.as_slice()),
    ])
    .unwrap()
}

/// Mean outcome of the rows of the cell (1, 1) that are not `skipped`: the answer from the
/// definition of the saturated cell mean.
fn cell_mean(skipped: Option<usize>) -> f64 {
    let rows: Vec<usize> =
        (0..ROWS).filter(|row| cell_of(*row) == (1.0, 1.0) && Some(*row) != skipped).collect();
    rows.iter().map(|row| outcome_of(*row)).sum::<f64>() / rows.len() as f64
}

fn edges(graph: &mut impl Edges) {
    for (from, to) in [(0, 2), (1, 2)] {
        graph.edge(from, to);
    }
}

trait Edges {
    fn edge(&mut self, from: u32, to: u32);
}

impl Edges for Dag {
    fn edge(&mut self, from: u32, to: u32) {
        self.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
}

impl Edges for Admg {
    fn edge(&mut self, from: u32, to: u32) {
        self.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
}

impl Edges for Cpdag {
    fn edge(&mut self, from: u32, to: u32) {
        self.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    edges(&mut graph);
    graph
}

fn study(
    data: TabularData,
    query: ResponseQuery,
    missing: Option<MissingDeclaration>,
    design: Option<DeclaredSamplingDesign>,
    replicates: u32,
) -> Result<StudyResult, CausalError> {
    let mut builder = Study::tabular(data)
        .graph(dag())
        .query(CausalQuery::Response(query))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates);
    if let Some(declaration) = missing {
        builder = builder.missing(declaration);
    }
    if let Some(design) = design {
        builder = builder.sampling_design(design);
    }
    builder.build().and_then(|built| built.run(&ctx()))
}

fn value(result: &StudyResult) -> f64 {
    match &result.response().expect("a response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("expected a point-identified scalar, got {other:?}"),
    }
}

fn refused(result: Result<StudyResult, CausalError>, what: &str) -> CausalError {
    result
        .err()
        .unwrap_or_else(|| panic!("{what}: published a number for a declaration it ignores"))
}

// ---------------------------------------------------------------------------------------------
// Latent-variable and equivalence-class graphs
// ---------------------------------------------------------------------------------------------

/// Columns `z, t1, t2, y` on a balanced binary design:
/// `y = 0.2 + 0.4 t1 + 0.3 t2 + 0.5 t1 t2 + 0.2 z + 0.03 sin(0.31 i)`; tiers `{z} | {t1, t2} | {y}`
/// with the two treatments co-determined, so the closure ADMG carries `t1 <-> t2` and the closure
/// adjustment set is `{z}`.
fn tiered_table() -> TabularData {
    let n = 960usize;
    let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let t1: Vec<f64> = (0..n).map(|i| (i / 2 % 2) as f64).collect();
    let t2: Vec<f64> = (0..n).map(|i| (i / 4 % 2) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            0.2 + 0.4 * t1[i]
                + 0.3 * t2[i]
                + 0.5 * t1[i] * t2[i]
                + 0.2 * z[i]
                + 0.03 * (i as f64 * 0.31).sin()
        })
        .collect();
    TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

/// `sum_z P(z) mean(y | z, t1 = 1, t2 = 1)`, the standardization over the closure adjustment set.
fn standardized_cell() -> f64 {
    let n = 960usize;
    let outcome = |i: usize| {
        let (t1, t2) = ((i / 2 % 2) as f64, (i / 4 % 2) as f64);
        0.2 + 0.4 * t1
            + 0.3 * t2
            + 0.5 * t1 * t2
            + 0.2 * (i % 2) as f64
            + 0.03 * (i as f64 * 0.31).sin()
    };
    (0..2usize)
        .map(|z| {
            let share = (0..n).filter(|i| i % 2 == z).count() as f64 / n as f64;
            let cell: Vec<usize> =
                (0..n).filter(|i| i % 2 == z && i / 2 % 2 == 1 && i / 4 % 2 == 1).collect();
            share * cell.iter().map(|i| outcome(*i)).sum::<f64>() / cell.len() as f64
        })
        .sum()
}

#[test]
fn the_joint_cell_equals_the_standardization_on_a_codetermined_closure_and_refuses_an_explicit_admg_or_cpdag()
 {
    on_large_stack("cells-04-latent-class", latent_class_body);
}

fn latent_class_body() {
    let data = tiered_table();
    let background = TieredBackground::from_named(
        antecedent_data::TableView::schema(&data),
        &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(3),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(2), Value::f64(1.0)),
        ]),
    });
    let result = Study::tabular(data)
        .tiered_background(background)
        .unwrap()
        .query(CausalQuery::Response(query))
        .estimator(EstimatorId::CellAipw)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    let got = value(&result);
    let want = standardized_cell();
    assert!((got - want).abs() < 1e-4, "closure joint cell {got} vs standardization {want}");

    // An explicit latent-variable graph or equivalence class of the same two causes is not read
    // as a DAG: the route names what it needs.
    let mut admg = Admg::with_variables(3);
    edges(&mut admg);
    let mut cpdag = Cpdag::with_variables(3);
    edges(&mut cpdag);
    for (class, error) in [
        (
            "admg",
            Study::tabular(table(None, false))
                .graph(admg)
                .query(CausalQuery::Response(level_query(1.0, 1.0)))
                .estimator(EstimatorId::CellAipw)
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .and_then(|built| built.run(&ctx())),
        ),
        (
            "cpdag",
            Study::tabular(table(None, false))
                .graph(cpdag)
                .query(CausalQuery::Response(level_query(1.0, 1.0)))
                .estimator(EstimatorId::CellAipw)
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .and_then(|built| built.run(&ctx())),
        ),
    ] {
        let error = refused(error, class);
        assert!(matches!(error, CausalError::Unsupported { .. }), "{class}: {error:?}");
        assert!(error.to_string().contains("requires a static DAG"), "{class}: {error}");
    }
    // The same graph as a DAG recovers the cell mean.
    let plain = value(&study(table(None, false), level_query(1.0, 1.0), None, None, 0).unwrap());
    assert!((plain - cell_mean(None)).abs() < 1e-5, "{plain} vs {}", cell_mean(None));
}

// ---------------------------------------------------------------------------------------------
// Missing outcomes
// ---------------------------------------------------------------------------------------------

#[test]
fn a_missing_outcome_follows_complete_case_and_the_policies_the_route_cannot_pool_are_refused() {
    on_large_stack("cells-04-missing", missing_body);
}

fn missing_body() {
    // Row 3 sits in the cell (1, 1): dropping it moves the cell mean, so a row read as zero or
    // kept as a value cannot match.
    const HOLE: usize = 3;
    assert_eq!(cell_of(HOLE), (1.0, 1.0));
    let kept: Vec<usize> = (0..ROWS).filter(|row| *row != HOLE).collect();
    let remaining = TabularData::from_f64_columns([
        ("t1", kept.iter().map(|row| cell_of(*row).0).collect::<Vec<_>>().as_slice()),
        ("t2", kept.iter().map(|row| cell_of(*row).1).collect::<Vec<_>>().as_slice()),
        ("y", kept.iter().map(|row| outcome_of(*row)).collect::<Vec<_>>().as_slice()),
    ])
    .unwrap();
    let on_remaining = value(&study(remaining, level_query(1.0, 1.0), None, None, 0).unwrap());
    let reference = cell_mean(Some(HOLE));
    assert!((on_remaining - reference).abs() < 1e-5, "{on_remaining} vs {reference}");
    assert!((reference - cell_mean(None)).abs() > 1e-5, "the dropped row moves the reference");

    let declared = study(
        table(Some(HOLE), false),
        level_query(1.0, 1.0),
        Some(MissingDeclaration::new(MissingPolicy::CompleteCase)),
        None,
        0,
    )
    .unwrap();
    let default = study(table(Some(HOLE), false), level_query(1.0, 1.0), None, None, 0).unwrap();
    for (label, result) in [("declared", &declared), ("default", &default)] {
        assert!((value(result) - on_remaining).abs() < 1e-12, "{label}: {}", value(result));
        let report = result.missing_data().as_ref().expect("a missing-data report");
        assert_eq!(report.accounting.rows_kept, ROWS - 1, "{label}");
    }

    let error = refused(
        study(
            table(Some(HOLE), false),
            level_query(1.0, 1.0),
            Some(MissingDeclaration::new(MissingPolicy::ErrorOnMissing)),
            None,
            0,
        ),
        "error policy",
    );
    assert_eq!(error.reason_code(), Some("missing_policy_required"), "{error}");

    for (name, policy) in [
        ("multiple imputation", MissingPolicy::MultipleImputation(ImputationSpec::new())),
        ("inverse probability", MissingPolicy::InverseProbabilityOfObservation),
    ] {
        let error = refused(
            study(
                table(Some(HOLE), false),
                level_query(1.0, 1.0),
                Some(MissingDeclaration::new(policy)),
                None,
                0,
            ),
            name,
        );
        assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{name}: {error}");
        assert!(
            error.to_string().contains("Next attempt")
                || error.to_string().contains("next attempt"),
            "{name}: {error}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Observation weights
// ---------------------------------------------------------------------------------------------

#[test]
fn table_weights_are_refused_by_the_joint_cell_on_a_study_and_on_a_batch() {
    on_large_stack("cells-04-weights", weights_body);
}

fn weights_body() {
    let unweighted =
        value(&study(table(None, false), level_query(1.0, 1.0), None, None, 0).unwrap());
    assert!((unweighted - cell_mean(None)).abs() < 1e-5);
    let weights: Vec<f64> = (0..ROWS).map(|row| 1.0 + (row % 3) as f64).collect();
    let weighted = || table(None, false).with_weights(Arc::from(weights.clone())).unwrap();
    let check = |what: &str, error: CausalError| {
        let text = error.to_string();
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{what}: {text}");
        assert!(text.contains("sampling_weights.route_unsupported"), "{what}: {text}");
        assert!(text.contains("Next attempt"), "{what}: {text}");
    };
    check("study", refused(study(weighted(), level_query(1.0, 1.0), None, None, 0), "study"));
    let batch = |data: TabularData| {
        BatchStudy::new(data.clone(), dag())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .prepare_cells(&[level_query(1.0, 1.0), level_query(1.0, 0.0)], &ctx())
            .and_then(|prepared| prepared.estimate(&data, &ctx()))
    };
    assert_eq!(batch(table(None, false)).unwrap().len(), 2);
    check("batch", batch(weighted()).err().expect("a weighted batch was estimated"));
}

// ---------------------------------------------------------------------------------------------
// Unobserved cell and non-binary level
// ---------------------------------------------------------------------------------------------

#[test]
fn an_unobserved_cell_and_a_non_binary_level_are_refused_instead_of_extrapolated() {
    on_large_stack("cells-04-held-out", held_out_body);
}

fn held_out_body() {
    // The cell (1, 1) is never observed: its mean is not estimated from the other cells.
    let error =
        refused(study(table(None, true), level_query(1.0, 1.0), None, None, 0), "empty cell");
    assert!(format!("{error:?}").contains("Unsupported"), "{error:?}");
    assert!(error.to_string().contains("empty joint treatment combination"), "{error}");
    // A level between the observed cells is not a cell.
    let error =
        refused(study(table(None, false), level_query(0.5, 1.0), None, None, 0), "half level");
    assert!(error.to_string().contains("binary 0/1 Set levels"), "{error}");
    // The batch refuses the unobserved cell the same way.
    let data = table(None, true);
    let error = BatchStudy::new(data.clone(), dag())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare_cells(&[level_query(1.0, 1.0)], &ctx())
        .and_then(|prepared| prepared.estimate(&data, &ctx()))
        .err()
        .expect("a batch estimated an unobserved cell");
    assert!(error.to_string().contains("empty joint treatment combination"), "{error}");
}

// ---------------------------------------------------------------------------------------------
// Declared dependence with the bootstrap off
// ---------------------------------------------------------------------------------------------

#[test]
fn a_declared_panel_or_series_is_refused_by_the_joint_cell_with_the_bootstrap_off() {
    on_large_stack("cells-04-dependence", dependence_body);
}

fn dependence_body() {
    let independent = value(
        &study(
            table(None, false),
            level_query(1.0, 1.0),
            None,
            Some(DeclaredSamplingDesign::IndependentSubjectTrajectories),
            0,
        )
        .unwrap(),
    );
    assert!((independent - cell_mean(None)).abs() < 1e-5);
    for (design, phrase) in [
        (DeclaredSamplingDesign::ClusteredPanel, "clustered-panel"),
        (DeclaredSamplingDesign::DependentTimeSeries, "dependent-time-series"),
    ] {
        let error = refused(
            study(table(None, false), level_query(1.0, 1.0), None, Some(design), 0),
            phrase,
        );
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{phrase}: {error}");
        let text = error.to_string();
        assert!(text.contains(phrase), "{phrase}: {text}");
        assert!(text.contains("Next attempt"), "{phrase}: {text}");
    }
}

#[test]
fn a_declared_panel_or_series_is_refused_by_the_joint_cell_even_when_a_bootstrap_is_requested() {
    on_large_stack("cells-04-dependence-bootstrap", dependence_with_bootstrap_body);
}

fn dependence_with_bootstrap_body() {
    for (design, phrase) in [
        (DeclaredSamplingDesign::ClusteredPanel, "clustered-panel"),
        (DeclaredSamplingDesign::DependentTimeSeries, "dependent-time-series"),
    ] {
        let error = refused(
            study(table(None, false), level_query(1.0, 1.0), None, Some(design), 20),
            phrase,
        );
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{phrase}: {error}");
        let text = error.to_string();
        assert!(text.contains("do not bootstrap"), "{phrase}: {text}");
        assert!(text.contains("Next attempt"), "{phrase}: {text}");
    }
    // Independent subject trajectories read nothing and run, with the bootstrap requested or not.
    let run = |replicates| {
        value(
            &study(
                table(None, false),
                level_query(1.0, 1.0),
                None,
                Some(DeclaredSamplingDesign::IndependentSubjectTrajectories),
                replicates,
            )
            .unwrap(),
        )
    };
    assert!((run(20) - run(0)).abs() < 1e-12);
}
