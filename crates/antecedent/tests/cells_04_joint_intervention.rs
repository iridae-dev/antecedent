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
//! - observation weights on the table are read by a study (weighted nuisance fits and weighted
//!   influence variance, against an exact-rational reference) and still refused with
//!   `sampling_weights.route_unsupported` on a batch, never estimated as unweighted rows;
//! - an unobserved cell and a non-binary level are refused instead of extrapolated;
//! - a declared clustered panel or dependent series is refused unless its cluster labels or
//!   Newey-West lag are declared with `score_dependence`, which the cell reads as a cluster-robust
//!   or Newey-West influence variance (exact-rational reference); it runs under independent
//!   subject trajectories.
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
    // A study now estimates the weighted population: the weights are read, not refused. The
    // weighted cell (1, 1) is the weighted mean of its rows (the exact weighted cross-fit
    // reference is `the_joint_cell_equals_the_exact_weighted_cross_fit_aipw_...`).
    let (num, den) =
        (0..ROWS).filter(|row| cell_of(*row) == (1.0, 1.0)).fold((0.0, 0.0), |(num, den), row| {
            (num + weights[row] * outcome_of(row), den + weights[row])
        });
    let on_study = value(&study(weighted(), level_query(1.0, 1.0), None, None, 0).unwrap());
    assert!((on_study - num / den).abs() < 1e-3, "{on_study} vs {}", num / den);
    // A batch shares one unweighted design across its cells and still refuses the weights.
    let check = |what: &str, error: CausalError| {
        let text = error.to_string();
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{what}: {text}");
        assert!(text.contains("sampling_weights.route_unsupported"), "{what}: {text}");
        assert!(text.contains("Next attempt"), "{what}: {text}");
    };
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

// ---------------------------------------------------------------------------------------------
// Sampling weights and declared dependence: exact references
// ---------------------------------------------------------------------------------------------

use antecedent::ScoreDependence;

/// The frozen exact-rational reference (see `oracle_data/joint_cell_dependence_reference.py`).
fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!("oracle_data/joint_cell_dependence_reference.json")).unwrap()
}

fn column(reference: &serde_json::Value, key: &str) -> Vec<f64> {
    reference[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| match v {
            serde_json::Value::String(text) => text.parse().unwrap(),
            other => other.as_f64().unwrap(),
        })
        .collect()
}

fn reference_table(reference: &serde_json::Value, weighted: bool) -> TabularData {
    let (t1, t2, y) =
        (column(reference, "t1"), column(reference, "t2"), column(reference, "y_decimal"));
    let table = TabularData::from_f64_columns([
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    if weighted {
        table.with_weights(Arc::from(column(reference, "weights"))).unwrap()
    } else {
        table
    }
}

fn dependent_study(
    data: TabularData,
    design: Option<DeclaredSamplingDesign>,
    dependence: Option<ScoreDependence>,
) -> Result<StudyResult, CausalError> {
    let mut builder = Study::tabular(data)
        .graph(dag())
        .query(CausalQuery::Response(level_query(1.0, 1.0)))
        .estimator(EstimatorId::CellAipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    if let Some(design) = design {
        builder = builder.sampling_design(design);
    }
    if let Some(dependence) = dependence {
        builder = builder.score_dependence(dependence);
    }
    builder.build().and_then(|built| built.run(&ctx()))
}

fn fraction(pair: &serde_json::Value) -> f64 {
    // The exact rational is reduced to the nearest f64 only here, at the comparison.
    let parts = pair.as_array().unwrap();
    let (num, den) = (parts[0].to_string(), parts[1].to_string());
    num.parse::<f64>().unwrap() / den.parse::<f64>().unwrap()
}

fn assert_matches_reference(result: &StudyResult, expected: &serde_json::Value, what: &str) {
    let effect = result.estimate().as_effect().expect("an effect estimate");
    let table = effect.scores.table.as_ref().expect("the cell score table");
    // The one fixed input of the reference is the product's seeded, cell-stratified fold plan.
    let frozen: Vec<u32> = expected["__folds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| u32::try_from(f.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(
        table.fold_ids.as_ref(),
        frozen.as_slice(),
        "{what}: the fold plan moved; regenerate the reference with these folds"
    );
    let mean = fraction(&expected["mean"]);
    let variance = fraction(&expected["variance"]);
    let got = value(result);
    assert!((got - mean).abs() < 1e-7 * mean.abs(), "{what}: mean {got} vs exact {mean}");
    let se = effect.se_analytic();
    assert!(
        (se * se - variance).abs() < 1e-6 * variance,
        "{what}: variance {} vs exact {variance}",
        se * se
    );
}

#[test]
fn the_joint_cell_equals_the_exact_weighted_cross_fit_aipw_and_its_sandwich_variances() {
    on_large_stack("cells-04-weighted-reference", weighted_reference_body);
}

fn weighted_reference_body() {
    let reference = reference();
    let clusters = ScoreDependence::clusters(
        reference["clusters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| u32::try_from(c.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>(),
    );
    let lag = usize::try_from(reference["lag"].as_u64().unwrap()).unwrap();
    for (weighted, key, folds_key) in
        [(false, "unweighted", "unweighted"), (true, "weighted", "weighted")]
    {
        let folds: Vec<u32> = reference["folds"][folds_key]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| u32::try_from(f.as_u64().unwrap()).unwrap())
            .collect();
        let runs = [
            ("independent", None, None),
            ("clusters", Some(DeclaredSamplingDesign::ClusteredPanel), Some(clusters.clone())),
            (
                "newey_west",
                Some(DeclaredSamplingDesign::DependentTimeSeries),
                Some(ScoreDependence::newey_west(lag)),
            ),
        ];
        for (dependence, design, declared) in runs {
            let result =
                dependent_study(reference_table(&reference, weighted), design, declared).unwrap();
            let what = format!("{key}/{dependence}");
            let mut expected = reference[key][dependence].clone();
            expected["__folds"] = serde_json::json!(folds);
            assert_matches_reference(&result, &expected, &what);
            let record = result.estimate().as_effect().unwrap().diagnostics.sampling_weights;
            if weighted {
                let record = record.expect("a weighted study reports its weight record");
                assert_eq!(record.rows_used, 59, "{what}");
                assert_eq!(record.zero_weight_rows, 1, "{what}");
                assert!(record.effective_sample_size < 59.0 && record.effective_sample_size > 1.0);
            } else {
                assert!(record.is_none(), "{what}");
            }
        }
    }
}

#[test]
fn a_zero_weight_row_equals_the_table_with_the_row_deleted() {
    on_large_stack("cells-04-zero-weight", zero_weight_body);
}

fn zero_weight_body() {
    let reference = reference();
    let zero = usize::try_from(reference["zero_weight_row"].as_u64().unwrap()).unwrap();
    let weighted = dependent_study(reference_table(&reference, true), None, None).unwrap();
    let keep = |values: Vec<f64>| -> Vec<f64> {
        values.into_iter().enumerate().filter(|(i, _)| *i != zero).map(|(_, v)| v).collect()
    };
    let (t1, t2, y, w) = (
        keep(column(&reference, "t1")),
        keep(column(&reference, "t2")),
        keep(column(&reference, "y_decimal")),
        keep(column(&reference, "weights")),
    );
    let deleted = TabularData::from_f64_columns([
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
    .with_weights(Arc::from(w))
    .unwrap();
    let deleted = dependent_study(deleted, None, None).unwrap();
    let (a, b) =
        (weighted.estimate().as_effect().unwrap(), deleted.estimate().as_effect().unwrap());
    assert!((value(&weighted) - value(&deleted)).abs() < 1e-12);
    assert!((a.se_analytic() - b.se_analytic()).abs() < 1e-12);
}

#[test]
fn the_weights_are_scale_free_and_an_extreme_weight_is_read_not_clipped() {
    on_large_stack("cells-04-weight-scale", weight_scale_body);
}

fn weight_scale_body() {
    let reference = reference();
    let scaled = |factor: f64| {
        let weights: Vec<f64> = column(&reference, "weights").iter().map(|w| w * factor).collect();
        reference_table(&reference, false).with_weights(Arc::from(weights)).unwrap()
    };
    let base = dependent_study(scaled(1.0), None, None).unwrap();
    for factor in [1e-6, 0.37, 1e6] {
        let other = dependent_study(scaled(factor), None, None).unwrap();
        assert!((value(&base) - value(&other)).abs() < 1e-9 * value(&base).abs(), "{factor}");
        let (a, b) = (base.estimate().as_effect().unwrap(), other.estimate().as_effect().unwrap());
        assert!((a.se_analytic() - b.se_analytic()).abs() < 1e-9 * a.se_analytic(), "{factor}");
    }
}

#[test]
fn declared_score_dependence_is_read_by_the_joint_cell_and_refused_when_it_cannot_be() {
    on_large_stack("cells-04-dependence-refusals", dependence_refusal_body);
}

fn dependence_refusal_body() {
    let reference = reference();
    let labels = |len: usize| {
        ScoreDependence::clusters(
            (0..len).map(|i| u32::try_from(i % 6).unwrap()).collect::<Vec<_>>(),
        )
    };
    let code = |error: &CausalError, text: &str| {
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
        assert!(error.to_string().contains(text), "expected {text:?} in {error}");
    };
    let table = || reference_table(&reference, false);
    // Labels without the matching design, and a design that does not match the kind.
    let error =
        refused(dependent_study(table(), None, Some(labels(60))), "labels without a design");
    code(&error, "disagree");
    let error = refused(
        dependent_study(
            table(),
            Some(DeclaredSamplingDesign::DependentTimeSeries),
            Some(labels(60)),
        ),
        "labels under a series design",
    );
    code(&error, "clustered-panel");
    let error = refused(
        dependent_study(
            table(),
            Some(DeclaredSamplingDesign::ClusteredPanel),
            Some(ScoreDependence::newey_west(2)),
        ),
        "lag under a panel design",
    );
    code(&error, "dependent-time-series");
    // Labels that do not cover the table, and a single cluster.
    let short =
        dependent_study(table(), Some(DeclaredSamplingDesign::ClusteredPanel), Some(labels(59)));
    let short = refused(short, "a short label vector");
    assert!(
        short
            .to_string()
            .contains("a cluster label was declared for fewer rows than the table has"),
        "{short}"
    );
    let single = dependent_study(
        table(),
        Some(DeclaredSamplingDesign::ClusteredPanel),
        Some(ScoreDependence::clusters(vec![0u32; 60])),
    );
    let single = refused(single, "one cluster");
    assert!(
        single.to_string().contains("cluster-robust variance requires at least 2 clusters"),
        "{single}"
    );
    // A bootstrap beside the declaration is not read by an influence-function route.
    let error = refused(
        Study::tabular(table())
            .graph(dag())
            .query(CausalQuery::Response(level_query(1.0, 1.0)))
            .estimator(EstimatorId::CellAipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(20)
            .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
            .score_dependence(labels(60))
            .build()
            .and_then(|built| built.run(&ctx())),
        "bootstrap beside labels",
    );
    code(&error, "does not bootstrap");
    // The refutation suite and the quantile read unweighted independent rows.
    let error = refused(
        Study::tabular(reference_table(&reference, true))
            .graph(dag())
            .query(CausalQuery::Response(level_query(1.0, 1.0)))
            .estimator(EstimatorId::CellAipw)
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0)
            .build()
            .and_then(|built| built.run(&ctx())),
        "refutation under weights",
    );
    code(&error, "RefuteSuite::None");
    // A route that does not read the labels refuses them instead of ignoring them.
    let error = refused(
        Study::tabular(table())
            .graph(dag())
            .query(CausalQuery::Response(level_query(1.0, 1.0)))
            .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
            .score_dependence(labels(60))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .and_then(|built| built.run(&ctx())),
        "labels on a route that does not read them",
    );
    code(&error, "score_dependence is read only by the joint-cell response");
}

#[test]
fn a_prepared_joint_cell_reads_the_weights_and_the_dependence_and_stamps_them_in_its_identity() {
    on_large_stack("cells-04-prepared", prepared_body);
}

fn prepared_body() {
    let reference = reference();
    let labels = |shift: u32| {
        ScoreDependence::clusters(
            reference["clusters"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| u32::try_from(c.as_u64().unwrap()).unwrap() + shift * 0)
                .enumerate()
                .map(|(row, c)| if shift == 0 || row % 7 != 0 { c } else { (c + 1) % 12 })
                .collect::<Vec<_>>(),
        )
    };
    let study = |dependence: ScoreDependence| {
        Study::tabular(reference_table(&reference, true))
            .graph(dag())
            .query(CausalQuery::Response(level_query(1.0, 1.0)))
            .estimator(EstimatorId::CellAipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
            .score_dependence(dependence)
            .build()
            .unwrap()
    };
    let first = study(labels(0));
    let one_shot = first.run(&ctx()).unwrap();
    let mut prepared = first.prepare(&ctx()).unwrap();
    let data = reference_table(&reference, true);
    let result = prepared.estimate(&data, &ctx()).unwrap();
    let (a, b) = (one_shot.estimate().as_effect().unwrap(), result.estimate().as_effect().unwrap());
    assert!((a.ate - b.ate).abs() < 1e-12, "{} vs {}", a.ate, b.ate);
    assert!((a.se_analytic() - b.se_analytic()).abs() < 1e-12);
    assert!(b.diagnostics.sampling_weights.is_some());
    // The prepared cell keeps no score table, so no other weights can retarget it.
    let rows = b.scores.table.as_ref().unwrap().n_rows;
    let error = prepared.retarget(&vec![1.0; rows], &[], &ctx()).unwrap_err();
    assert_eq!(error.reason_code(), Some("score_table_unavailable"), "{error}");
    // Two results that differ only in the declared labels never share an identity.
    let other = study(labels(1));
    let mut other_prepared = other.prepare(&ctx()).unwrap();
    let other_result = other_prepared.estimate(&data, &ctx()).unwrap();
    let (stamp, other_stamp) = (
        result.executed_contract().expect("a prepared result is stamped"),
        other_result.executed_contract().expect("a prepared result is stamped"),
    );
    assert_ne!(stamp.identities, other_stamp.identities);
    let again = study(labels(0)).prepare(&ctx()).unwrap().estimate(&data, &ctx()).unwrap();
    assert_eq!(again.executed_contract().unwrap().identities, stamp.identities);
}

#[test]
fn a_weighted_or_dependent_joint_cell_does_not_inherit_the_unweighted_cells_license() {
    on_large_stack("cells-04-license", license_body);
}

fn license_body() {
    let reference = reference();
    let plain = dependent_study(reference_table(&reference, false), None, None).unwrap();
    let weighted = dependent_study(reference_table(&reference, true), None, None).unwrap();
    let clustered = dependent_study(
        reference_table(&reference, false),
        Some(DeclaredSamplingDesign::ClusteredPanel),
        Some(ScoreDependence::clusters((0..60u32).map(|i| i / 5).collect::<Vec<_>>())),
    )
    .unwrap();
    assert_eq!(format!("{:?}", plain.support_status()), "Some(Licensed)");
    assert_ne!(weighted.support_status(), plain.support_status());
    assert_ne!(clustered.support_status(), plain.support_status());
    assert_eq!(format!("{:?}", weighted.support_status()), "Some(Refused)");
    assert_eq!(format!("{:?}", clustered.support_status()), "Some(Refused)");
}
