//! Coordinate-specific closure evidence for checked front-door and IV execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, IdentifierId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_estimate::{AnalyticSeKind, FrontDoorTwoStage, TwoStageLeastSquares, WaldIv};
use antecedent_graph::Dag;
use antecedent_io::consume_analysis_result;

fn frontdoor_data(y_shift: f64) -> TabularData {
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
    let n = 2_000;
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
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(
                        u32::try_from(i).expect("fixture variable index fits u32"),
                    ),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
}

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

fn suite(name: &str) -> RefuteSuite {
    match name {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        _ => unreachable!(),
    }
}

fn verify_frontdoor(accepted: bool, validation: &str) {
    let coordinate = format!(
        "AverageEffect:Dag:{}:Frequentist:{}::estimator=frontdoor.linear_two_stage",
        if accepted { "accepted" } else { "explicit" },
        validation
    );
    let data = frontdoor_data(0.0);
    let refreshed_data = frontdoor_data(0.35);
    let graph = Dag::from_named_edges(data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(2), 0.0, 1.0);
    let context = ExecutionContext::for_tests(810);
    let builder = Study::tabular(data.clone())
        .query(query)
        .identifier(IdentifierId::Frontdoor)
        .estimator(FrontDoorTwoStage::new().with_bootstrap_replicates(0))
        .refute(suite(validation));
    let builder =
        if accepted { builder.graph(AcceptedGraph::from(graph)) } else { builder.graph(graph) };
    let reference = builder.clone().build().unwrap().run(&context).unwrap();
    let study = builder.clone().build().unwrap();
    let mut prepared = study.prepare(&context).unwrap();
    drop(builder);
    drop(study);
    assert_eq!(
        prepared.plan().logical.record.estimator.as_deref(),
        Some("frontdoor.linear_two_stage"),
        "{coordinate}"
    );
    let checked = prepared.checked_frontdoor_linear().expect("checked front-door plan");
    assert_eq!(
        checked.lowering().procedure,
        antecedent_estimate::frontdoor::CheckedFrontDoorProcedure::LinearPathProduct,
        "{coordinate}"
    );
    assert_eq!(checked.program().mapping().source, checked.target().functional, "{coordinate}");
    assert_eq!(checked.lowering().treatment, VariableId::from_raw(0), "{coordinate}");
    assert_eq!(checked.lowering().outcome, VariableId::from_raw(2), "{coordinate}");
    let result = prepared.estimate(&data, &context).unwrap();
    assert!((result.effect() - 1.4).abs() < 0.1, "{coordinate}: {}", result.effect());
    assert!(
        (result.effect() - reference.effect()).abs() < 1e-12,
        "{coordinate} changed result vs fresh route"
    );
    let refreshed = prepared.refresh(refreshed_data, &context).unwrap();
    assert!((refreshed.effect() - result.effect()).abs() < 0.05, "{coordinate} refresh");
    let artifact = prepared
        .encode_contracted_result(
            &refreshed,
            &format!("frontdoor-{validation}-{accepted}"),
            &context,
        )
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed.acceptance.accepts_as_verified_program(),
        "{coordinate} artifact refusal: {:?}",
        consumed.acceptance.unresolved
    );
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{coordinate} artifact estimate");
}

fn verify_iv(accepted: bool, validation: &str, wald: bool) {
    let estimator_name = if wald { "iv.wald" } else { "iv.2sls" };
    let coordinate = format!(
        "AverageEffect:Dag:{}:Frequentist:{}::estimator={estimator_name}",
        if accepted { "accepted" } else { "explicit" },
        validation
    );
    let data = iv_data(0.0);
    let refreshed_data = iv_data(0.4);
    let graph = Dag::from_named_edges(data.schema(), &[("z", "t"), ("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let context = ExecutionContext::for_tests(811);
    let estimator: antecedent::EstimatorSpec = if wald {
        WaldIv::new().with_se_kind(AnalyticSeKind::Hc1).into()
    } else {
        TwoStageLeastSquares::new().with_se_kind(AnalyticSeKind::Hc1).into()
    };
    let builder = Study::tabular(data.clone())
        .query(query)
        .identifier(IdentifierId::Iv)
        .estimator(estimator)
        .refute(suite(validation));
    let builder =
        if accepted { builder.graph(AcceptedGraph::from(graph)) } else { builder.graph(graph) };
    let reference = builder.clone().build().unwrap().run(&context).unwrap();
    let study = builder.clone().build().unwrap();
    let mut prepared = study.prepare(&context).unwrap();
    drop(builder);
    drop(study);
    assert_eq!(
        prepared.plan().logical.record.estimator.as_deref(),
        Some(estimator_name),
        "{coordinate}"
    );
    let checked = prepared.checked_iv().expect("checked IV plan");
    assert_eq!(
        checked.lowering().procedure,
        if wald {
            antecedent_estimate::CheckedIvProcedure::Wald
        } else {
            antecedent_estimate::CheckedIvProcedure::TwoStageLeastSquares
        },
        "{coordinate}"
    );
    assert_eq!(checked.lowering().instruments.as_ref(), &[VariableId::from_raw(2)], "{coordinate}");
    assert_eq!(checked.lowering().treatment, VariableId::from_raw(0), "{coordinate}");
    assert_eq!(checked.lowering().outcome, VariableId::from_raw(1), "{coordinate}");
    let result = prepared.estimate(&data, &context).unwrap();
    assert!((result.effect() - 2.0).abs() < 0.15, "{coordinate}: {}", result.effect());
    assert!(
        (result.effect() - reference.effect()).abs() < 1e-12,
        "{coordinate} changed result vs fresh route"
    );
    assert_eq!(result.estimate.se_kind, Some(AnalyticSeKind::Hc1), "{coordinate}");
    let refreshed = prepared.refresh(refreshed_data, &context).unwrap();
    assert!(
        (refreshed.effect() - 2.0).abs() < 0.15,
        "{coordinate} refresh: {}",
        refreshed.effect()
    );
    let artifact = prepared
        .encode_contracted_result(
            &refreshed,
            &format!("{estimator_name}-{validation}-{accepted}"),
            &context,
        )
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed.acceptance.accepts_as_verified_program(),
        "{coordinate} artifact refusal: {:?}",
        consumed.acceptance.unresolved
    );
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{coordinate} artifact estimate");
}

#[test]
fn dag_accepted_frequentist_frontdoor_none() {
    verify_frontdoor(true, "none");
}

#[test]
fn dag_accepted_frequentist_frontdoor_cheap() {
    verify_frontdoor(true, "cheap");
}

#[test]
fn dag_accepted_frequentist_frontdoor_full() {
    verify_frontdoor(true, "full");
}

#[test]
fn dag_explicit_frequentist_frontdoor_none() {
    verify_frontdoor(false, "none");
}

#[test]
fn dag_explicit_frequentist_frontdoor_cheap() {
    verify_frontdoor(false, "cheap");
}

#[test]
fn dag_explicit_frequentist_frontdoor_full() {
    verify_frontdoor(false, "full");
}

#[test]
fn dag_accepted_frequentist_iv_2sls_none() {
    verify_iv(true, "none", false);
}

#[test]
fn dag_accepted_frequentist_iv_wald_none() {
    verify_iv(true, "none", true);
}

#[test]
fn dag_accepted_frequentist_iv_2sls_cheap() {
    verify_iv(true, "cheap", false);
}

#[test]
fn dag_accepted_frequentist_iv_wald_cheap() {
    verify_iv(true, "cheap", true);
}

#[test]
fn dag_accepted_frequentist_iv_2sls_full() {
    verify_iv(true, "full", false);
}

#[test]
fn dag_accepted_frequentist_iv_wald_full() {
    verify_iv(true, "full", true);
}

#[test]
fn dag_explicit_frequentist_iv_2sls_none() {
    verify_iv(false, "none", false);
}

#[test]
fn dag_explicit_frequentist_iv_wald_none() {
    verify_iv(false, "none", true);
}

#[test]
fn dag_explicit_frequentist_iv_2sls_cheap() {
    verify_iv(false, "cheap", false);
}

#[test]
fn dag_explicit_frequentist_iv_wald_cheap() {
    verify_iv(false, "cheap", true);
}

#[test]
fn dag_explicit_frequentist_iv_2sls_full() {
    verify_iv(false, "full", false);
}

#[test]
fn dag_explicit_frequentist_iv_wald_full() {
    verify_iv(false, "full", true);
}
