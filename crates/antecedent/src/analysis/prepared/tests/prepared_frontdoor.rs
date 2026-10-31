//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_estimate::{AnalyticSeKind, FrontDoorTwoStage};
use antecedent_graph::Dag;

use crate::Study;
use crate::analysis::builder::RefuteSuite;
use crate::analysis::prepared::PreparedExecution;
use crate::strategy_table::{EstimatorId, IdentifierId};

fn data(outcome_shift: f64) -> TabularData {
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("m", RoleHint::Context),
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
    let treatment_id = schema.id_of("t").unwrap();
    let outcome_id = schema.id_of("y").unwrap();
    let mediator_id = schema.id_of("m").unwrap();
    let n = 1_600;
    let mut treatment = Vec::with_capacity(n);
    let mut mediator = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    for i in 0..(n / 2) {
        let mediator_noise = ((i * 17 % 101) as f64 - 50.0) / 65.0;
        let outcome_noise = ((i * 31 % 97) as f64 - 48.0) / 42.0;
        for t in [0.0, 1.0] {
            let m = 0.8 * t + mediator_noise;
            treatment.push(t);
            mediator.push(m);
            outcome.push(outcome_shift + 2.0 * m + outcome_noise);
        }
    }
    let columns = [(treatment_id, treatment), (outcome_id, outcome), (mediator_id, mediator)];
    let columns = columns
        .into_iter()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
}

fn study(data: TabularData, fitter: FrontDoorTwoStage) -> Study {
    let graph = Dag::from_named_edges(data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
    let treatment = data.schema().id_of("t").unwrap();
    let outcome = data.schema().id_of("y").unwrap();
    Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::with_levels(treatment, outcome, 0.0, 1.0))
        .identifier(IdentifierId::Frontdoor)
        .estimator(fitter)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

#[test]
fn prepared_frontdoor_keeps_checked_procedure_through_estimate_and_refresh() {
    let context = ExecutionContext::for_tests(91);
    let fitter =
        FrontDoorTwoStage::new().with_bootstrap_replicates(0).with_se_kind(AnalyticSeKind::Hc1);
    // `build` and `prepare` consume their builders; only the retained handle is used below.
    let mut prepared = study(data(0.0), fitter.clone()).prepare(&context).unwrap();
    let PreparedExecution::FrontDoorLinear(operation) = &prepared.execution else {
        panic!("linear front-door must prepare a checked execution variant")
    };
    assert_eq!(operation.fitter.inference.se_kind, AnalyticSeKind::Hc1);
    assert_eq!(
        operation.preparation.lowering().procedure,
        antecedent_estimate::frontdoor::CheckedFrontDoorProcedure::LinearPathProduct
    );
    assert_eq!(
        operation.preparation.program().mapping().source,
        operation.preparation.target().functional
    );

    let first = prepared.estimate(&data(0.0), &context).unwrap();
    assert!(
        (first.estimate.as_effect().unwrap().ate - 1.6).abs() < 0.12,
        "estimate={}",
        first.estimate.as_effect().unwrap().ate
    );
    assert_eq!(
        first.estimate.as_effect().unwrap().uncertainty.analytic.kind,
        Some(AnalyticSeKind::Hc1)
    );
    let first_bytes =
        prepared.encode_contracted_result(&first, "checked-frontdoor", &context).unwrap();
    let first_consumed = antecedent_io::consume_analysis_result(&first_bytes).unwrap();
    assert!(!first_consumed.acceptance.accepts_as_verified_program());
    assert_eq!(
        first_consumed
            .acceptance
            .unresolved
            .iter()
            .map(|reason| reason.as_ref())
            .collect::<Vec<&str>>(),
        ["dependencies.checked_estimator_operation"]
    );
    assert_eq!(first_consumed.body.estimate, Some(first.effect()));

    let refreshed =
        crate::analysis::prepared::rebind_estimate(&mut prepared, data(0.35), &context).unwrap();
    assert!(
        (refreshed.estimate.as_effect().unwrap().ate - 1.6).abs() < 0.12,
        "estimate={}",
        refreshed.estimate.as_effect().unwrap().ate
    );
    assert_eq!(
        refreshed.estimate.as_effect().unwrap().uncertainty.analytic.kind,
        Some(AnalyticSeKind::Hc1)
    );
    let PreparedExecution::FrontDoorLinear(rebound) = &prepared.execution else {
        panic!("the rebound branch must keep the checked linear front-door operation")
    };
    assert_eq!(rebound.fitter.inference.se_kind, AnalyticSeKind::Hc1);
    let refreshed_bytes = prepared
        .encode_contracted_result(&refreshed, "checked-frontdoor-refresh", &context)
        .unwrap();
    let refreshed_consumed = antecedent_io::consume_analysis_result(&refreshed_bytes).unwrap();
    assert!(!refreshed_consumed.acceptance.accepts_as_verified_program());
    assert_eq!(
        refreshed_consumed
            .acceptance
            .unresolved
            .iter()
            .map(|reason| reason.as_ref())
            .collect::<Vec<&str>>(),
        ["dependencies.checked_estimator_operation"]
    );
    assert_eq!(refreshed_consumed.body.estimate, Some(refreshed.effect()));

    let treatment = prepared.schema.id_of("t").unwrap();
    let outcome = prepared.schema.id_of("y").unwrap();
    prepared.study_mut().unwrap().query =
        CausalQuery::AverageEffect(AverageEffectQuery::with_levels(treatment, outcome, 0.0, 2.0));
    let error = prepared.estimate(&data(0.35), &context).unwrap_err();
    assert!(error.to_string().contains("retained checked front-door operation"), "{error}");
}

#[test]
fn default_frontdoor_id_exports_a_bound_but_unreplayed_checked_result() {
    let data = data(0.0);
    let graph = Dag::from_named_edges(data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
    let treatment = data.schema().id_of("t").unwrap();
    let outcome = data.schema().id_of("y").unwrap();
    let context = ExecutionContext::for_tests(92);
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(AverageEffectQuery::with_levels(treatment, outcome, 0.0, 1.0))
        .identifier(IdentifierId::Frontdoor)
        .estimator(EstimatorId::FrontDoorTwoStage)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    assert!(matches!(prepared.execution, PreparedExecution::FrontDoorLinear(_)));
    let result = prepared.estimate(&data, &context).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "default-frontdoor", &context).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert!(!consumed.acceptance.accepts_as_verified_program());
    assert_eq!(
        consumed.acceptance.unresolved.iter().map(|reason| reason.as_ref()).collect::<Vec<&str>>(),
        ["dependencies.checked_estimator_operation"]
    );
}
