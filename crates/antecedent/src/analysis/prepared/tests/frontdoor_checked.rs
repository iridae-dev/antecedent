//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_graph::Dag;

use crate::analysis::builder::RefuteSuite;
use crate::analysis::execute::Study;
use crate::analysis::prepared::PreparedExecution;
use crate::estimator_spec::EstimatorSpec;
use crate::strategy_table::IdentifierId;

fn data(y_shift: f64) -> TabularData {
    data_rows(y_shift, 2_000)
}

fn data_rows(y_shift: f64, n: usize) -> TabularData {
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("m", RoleHint::Context),
        ("y", RoleHint::OutcomeCandidate),
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
    let mut treatment = Vec::with_capacity(n);
    let mut mediator = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i % 2) as f64;
        let u = ((i * 37 % 101) as f64 - 50.0) / 100.0;
        let m = 0.7 * t + u;
        treatment.push(t);
        mediator.push(m);
        outcome.push(y_shift + 2.0 * m + ((i * 13 % 47) as f64 - 23.0) / 100.0);
    }
    let columns = [treatment, mediator, outcome]
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(index).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
}

#[test]
fn frontdoor_refresh_publishes_the_refreshed_row_count() {
    let base_data = data(0.0);
    let graph = Dag::from_named_edges(base_data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(2), 0.0, 1.0);
    let context = ExecutionContext::for_tests(810);
    for rows in [1_200_usize, 3_000] {
        let mut prepared = Study::tabular(base_data.clone())
            .graph(graph.clone())
            .query(query.clone())
            .identifier(IdentifierId::Frontdoor)
            .estimator(antecedent_estimate::FrontDoorTwoStage::new().with_bootstrap_replicates(0))
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&context)
            .unwrap();
        let refreshed = prepared.refresh(data_rows(0.4, rows), &context).unwrap();
        let artifact =
            prepared.encode_contracted_result(&refreshed, "frontdoor", &context).unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert!(
            consumed.acceptance.accepts_as_verified_program(),
            "unresolved={:?}",
            consumed.acceptance.unresolved
        );
        let published = consumed
            .contract
            .as_ref()
            .and_then(|contract| contract.program.as_ref())
            .and_then(|program| program.checked_frontdoor_lowering.as_ref())
            .map(|lowering| lowering.complete_case_rows);
        assert_eq!(published, Some(rows as u64));
    }
}

#[test]
fn frontdoor_prepared_click_uses_retained_fit_after_study_config_changes() {
    let base_data = data(0.0);
    let refreshed_data = data(0.4);
    let graph = Dag::from_named_edges(base_data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(2), 0.0, 1.0);
    let context = ExecutionContext::for_tests(810);
    let study = Study::tabular(base_data.clone())
        .graph(graph)
        .query(query)
        .identifier(IdentifierId::Frontdoor)
        .estimator(antecedent_estimate::FrontDoorTwoStage::new().with_bootstrap_replicates(0))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut prepared = study.prepare(&context).unwrap();
    drop(study);
    let PreparedExecution::FrontDoorLinear(operation) = &prepared.execution else {
        panic!("expected retained checked front-door operation")
    };
    assert_eq!(operation.preparation.lowering().treatment, VariableId::from_raw(0));
    assert_eq!(operation.fitter.inference.bootstrap_replicates, 0);
    prepared.analysis.estimator_spec = Some(Arc::new(EstimatorSpec::LinearAdjustmentAte(
        Box::new(antecedent_estimate::LinearAdjustmentAte::new()),
    )));
    let estimate = prepared.estimate(&base_data, &context).unwrap();
    assert!((estimate.effect() - 1.4).abs() < 0.1, "estimate={}", estimate.effect());
    let refreshed = prepared.refresh(refreshed_data, &context).unwrap();
    assert!((refreshed.effect() - estimate.effect()).abs() < 0.05);
    assert!(refreshed.estimate.as_effect().unwrap().provenance.assumptions.entries.iter().any(|entry| {
        matches!(&entry.assumption, antecedent_core::Assumption::ParametricRestriction(p) if p.id.as_ref() == "frontdoor.linear_path_product")
    }));
}
