//! The instrumental-variable routes on a latent-variable graph, with weights and with holes.
//!
//! The data are `z in {0, 1}` alternating, `t = 0.6 z + u`, `y = 2 t + u` with a shared
//! disturbance `u`, so the structural effect of `t` on `y` is 2 and least squares is biased.
//! Two-stage least squares recovers it on the instrument graph. An ADMG that carries the
//! unmeasured confounding as a bidirected edge is the general identification route, which runs
//! only its own functional estimator: both instrumental estimators are refused there with the
//! estimator named. Observation weights on the table are refused, because the instrumental fit
//! reads none of them, and an incomplete outcome follows the default complete-case policy with
//! the dropped rows reported, equal to the fit on the rows that remain.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{IdentifierId, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    ColumnView, Float64Column, MissingDeclaration, MissingPolicy, OwnedColumn,
    OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_estimate::{AnalyticSeKind, TwoStageLeastSquares, WaldIv};
use antecedent_graph::{Admg, Dag, DenseNodeId};

fn iv_data(y_shift: f64) -> TabularData {
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let ids = [schema.id_of("t").unwrap(), schema.id_of("y").unwrap(), schema.id_of("z").unwrap()];
    let n = 1_600;
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut instrument = Vec::with_capacity(n);
    for i in 0..n {
        let z = (i % 2) as f64;
        let u = ((i * 37 % 101) as f64 - 50.0) / 30.0;
        let t = 0.6 * z + u;
        instrument.push(z);
        treatment.push(t);
        outcome.push(y_shift + 2.0 * t + u);
    }
    let columns = [treatment, outcome, instrument]
        .into_iter()
        .zip(ids)
        .map(|(values, id)| {
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
}



fn query() -> AverageEffectQuery {
    AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0)
}

fn instrument_dag(data: &TabularData) -> Dag {
    Dag::from_named_edges(data.schema(), &[("z", "t"), ("t", "y")]).unwrap()
}

/// `z -> t -> y` with the unmeasured `t <-> y` confounding as a bidirected edge.
fn instrument_admg() -> Admg {
    let node = DenseNodeId::from_raw;
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(2), node(0)).unwrap();
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_bidirected(node(0), node(1)).unwrap();
    graph
}

fn two_stage() -> TwoStageLeastSquares {
    TwoStageLeastSquares::new().with_se_kind(AnalyticSeKind::Hc1)
}

fn run_dag(
    data: &TabularData,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, antecedent::CausalError> {
    let builder = Study::tabular(data.clone())
        .query(query())
        .identifier(IdentifierId::Iv)
        .estimator(two_stage())
        .refute(RefuteSuite::None)
        .graph(instrument_dag(data));
    let builder = match missing {
        Some(declaration) => builder.missing(declaration),
        None => builder,
    };
    builder.build().and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

fn column(data: &TabularData, raw: u32) -> Vec<f64> {
    let ColumnView::Float64(column) = data.column(VariableId::from_raw(raw)).unwrap() else {
        panic!("a continuous column");
    };
    column.values.as_slice().to_vec()
}

#[test]
fn two_stage_least_squares_recovers_the_structural_effect_on_the_instrument_graph() {
    let data = iv_data(0.0);
    let effect = run_dag(&data, None).unwrap().effect();
    assert!((effect - 2.0).abs() < 0.15, "{effect}");
}

#[test]
fn the_latent_variable_graph_runs_only_its_own_estimator_and_refuses_both_instrumental_ones() {
    let data = iv_data(0.0);
    for (name, estimator) in [
        ("iv.2sls", antecedent::EstimatorSpec::from(two_stage())),
        ("iv.wald", antecedent::EstimatorSpec::from(WaldIv::new().with_se_kind(AnalyticSeKind::Hc1))),
    ] {
        let error = Study::tabular(data.clone())
            .query(query())
            .identifier(IdentifierId::Iv)
            .estimator(estimator)
            .refute(RefuteSuite::None)
            .graph(instrument_admg())
            .build()
            .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
            .err()
            .unwrap_or_else(|| panic!("{name}: an instrumental estimate on an ADMG"));
        let text = error.to_string();
        assert!(text.contains("ADMG ATE requires estimator functional.effect"), "{name}: {text}");
        assert!(text.contains(&format!("got {name}")), "{name}: {text}");
    }
}

#[test]
fn table_weights_are_refused_by_the_instrumental_route() {
    let data = iv_data(0.0);
    let weights: Vec<f64> = (0..data.row_count()).map(|i| 1.0 + (i % 3) as f64).collect();
    let weighted = data.with_weights(Arc::from(weights)).unwrap();
    let error = run_dag(&weighted, None).err().expect("weights are not read by the fit");
    let text = error.to_string();
    assert!(text.starts_with("reason=option_not_applicable"), "{text}");
    assert!(text.contains("sampling_weights.route_unsupported"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
}

#[test]
fn an_incomplete_outcome_defaults_to_complete_case_and_equals_the_fit_on_the_remaining_rows() {
    let data = iv_data(0.0);
    let (t, y, z) = (column(&data, 0), column(&data, 1), column(&data, 2));
    let n = t.len();
    let holes: Vec<usize> = (0..n).filter(|i| i % 10 == 3).collect();
    let mut holey = y.clone();
    for &row in &holes {
        holey[row] = f64::NAN;
    }
    let incomplete = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", holey.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let keep = |v: &[f64]| -> Vec<f64> {
        v.iter().enumerate().filter(|(i, _)| i % 10 != 3).map(|(_, x)| *x).collect()
    };
    let remaining = TabularData::from_f64_columns([
        ("t", keep(&t).as_slice()),
        ("y", keep(&y).as_slice()),
        ("z", keep(&z).as_slice()),
    ])
    .unwrap();
    let want = run_dag(&remaining, None).unwrap().effect();
    let default = run_dag(&incomplete, None).unwrap();
    let declared =
        run_dag(&incomplete, Some(MissingDeclaration::new(MissingPolicy::CompleteCase))).unwrap();
    for (label, result) in [("default", &default), ("declared", &declared)] {
        assert!((result.effect() - want).abs() < 1e-9, "{label}: {} vs {want}", result.effect());
        let report = result.missing_data().as_ref().expect("a missing-data report");
        assert_eq!(report.accounting.rows_total, n, "{label}");
        assert_eq!(report.accounting.rows_kept, n - holes.len(), "{label}");
        let dropped: Vec<u64> = holes.iter().map(|&row| row as u64).collect();
        assert_eq!(report.accounting.dropped_rows.as_ref(), dropped.as_slice(), "{label}");
    }
    assert!(!default.missing_data().as_ref().unwrap().declared);
    assert!(declared.missing_data().as_ref().unwrap().declared);
    let error = run_dag(&incomplete, Some(MissingDeclaration::new(MissingPolicy::ErrorOnMissing)))
        .err()
        .expect("an explicit error policy refuses the holes");
    assert_eq!(error.reason_code(), Some("missing_policy_required"), "{error}");
}
