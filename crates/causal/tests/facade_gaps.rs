//! Facade gap closure: PAG execute, conditional/CF/attribution paths, Auto estimand selection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use causal::estimate::{identify_static_query, select_estimand};
use causal::{CausalAnalysis, CompiledAnalysis, EstimatorId, IdentifierId, RefuteSuite};
use causal_core::{
    AnomalyAttributionQuery, AverageEffectQuery, CausalQuery, CausalSchemaBuilder,
    ConditionalEffectQuery, CounterfactualQuery, ExecutionContext, IdentificationStatus,
    Intervention, MeasurementSpec, MediationContrast, MediationQuery, RoleHint, SmallRoleSet,
    Value, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_graph::{Dag, DenseNodeId, Pag};

fn chain_table(n: usize) -> (TabularData, Dag) {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let y: Vec<f64> =
        t.iter().enumerate().map(|(i, &ti)| 1.0 + 2.0 * ti + 0.01 * (i as f64)).collect();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, g)
}

#[test]
fn pag_ate_via_generalized_adjustment() {
    let (data, _) = chain_table(80);
    // Directed MAG completion of a two-node PAG is unique → full identified mass.
    let mut pag = Pag::with_variables(2);
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let analysis = CausalAnalysis::builder()
        .data(data)
        .pag(pag)
        .query(q)
        .identifier(IdentifierId::GeneralizedAdjustment)
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let result = analysis.run(&ctx).unwrap();
    assert!(result.estimate.ate.is_finite());
    assert!((result.estimate.ate - 2.0).abs() < 0.2);
    assert!(!matches!(result.identification.status, IdentificationStatus::NotIdentified));
}

#[test]
fn pag_with_circles_ate_via_generalized_adjustment() {
    let (data, _) = chain_table(80);
    let mut pag = Pag::with_variables(2);
    pag.insert_circle_arrow(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let analysis = CausalAnalysis::builder()
        .data(data)
        .pag(pag)
        .query(q)
        .identifier(IdentifierId::GeneralizedAdjustment)
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let plan = analysis.compile(&ctx).unwrap();
    assert!(matches!(plan, CompiledAnalysis::Ready(_)));
    let result = analysis.execute(&plan, &ctx).unwrap();
    assert!(result.estimate.ate.is_finite());
}

#[test]
fn conditional_effect_via_causal_analysis() {
    let n = 120usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y", "w"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let y: Vec<f64> =
        t.iter().zip(w.iter()).map(|(&ti, &wi)| 1.0 + 2.0 * ti + 0.5 * ti * wi).collect();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(w), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let inner = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_effect_modifiers([VariableId::from_raw(2)]);
    let cq = ConditionalEffectQuery::try_new(inner).unwrap();
    let analysis = CausalAnalysis::builder()
        .data(data)
        .graph(g)
        .query(CausalQuery::ConditionalEffect(cq))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(1)).unwrap();
    assert!(result.estimate.ate.is_finite());
}

#[test]
fn counterfactual_and_anomaly_via_causal_analysis() {
    let (data, g) = chain_table(60);
    let cf = CounterfactualQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    let analysis = CausalAnalysis::builder()
        .data(data.clone())
        .graph(g.clone())
        .query(CausalQuery::Counterfactual(cf))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(1)).unwrap();
    assert!(result.counterfactual.is_some());
    assert!(result.estimate.ate.is_finite());

    let an = AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100);
    let analysis = CausalAnalysis::builder()
        .data(data)
        .graph(g)
        .query(CausalQuery::AnomalyAttribution(an))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(1)).unwrap();
    assert!(result.anomaly.is_some());
}

#[test]
fn static_mediation_natural_rejected() {
    let n = 40usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "m", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let zeros = vec![0.0; n];
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(zeros.clone()),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(zeros.clone()),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(2),
                Arc::from(zeros),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let q = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::NaturalDirect,
    );
    let err = CausalAnalysis::builder()
        .data(data)
        .graph(g)
        .query(CausalQuery::Mediation(q))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .compile_logical()
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("natural") || msg.contains("Total") || msg.contains("temporal"),
        "unexpected error: {msg}"
    );
}

#[test]
fn auto_multi_estimand_requires_unique_match() {
    let (data, g) = chain_table(40);
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let id = identify_static_query(IdentifierId::Auto, &g, &CausalQuery::AverageEffect(q)).unwrap();
    if id.estimands.len() > 1 {
        let err = select_estimand(&id, EstimatorId::Other(Arc::from("unmatched"))).unwrap_err();
        assert!(err.to_string().contains("estimands"));
    }
    let _ = data;
}

#[test]
fn generalized_adjustment_on_dag_errors_clearly() {
    let (_, g) = chain_table(10);
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let err = identify_static_query(
        IdentifierId::GeneralizedAdjustment,
        &g,
        &CausalQuery::AverageEffect(q),
    )
    .unwrap_err();
    assert!(err.to_string().contains("PAG"));
}

#[test]
fn pag_compile_ready() {
    let (data, _) = chain_table(40);
    let mut pag = Pag::with_variables(2);
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let analysis = CausalAnalysis::builder()
        .data(data)
        .pag(pag)
        .query(q)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let compiled = analysis.compile(&ExecutionContext::for_tests(1)).unwrap();
    assert!(matches!(compiled, CompiledAnalysis::Ready(_)));
}
