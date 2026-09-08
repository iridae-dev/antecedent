//! Effect-suite applicability: ConditionalEffect Placebo/RCC run on the licensed
//! interaction-model scalar. Per-run `NotApplicable` skips remain pinned for temporal
//! overlap in `manufacturing_temporal.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchemaBuilder, ConditionalEffectQuery, ExecutionContext,
    MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};

/// `ConditionalEffectQuery` executes through `ConditionalLinearAdjustment`. The licensed
/// number is the interaction-model scalar at Ē[W], so Placebo / RCC refit that model
/// rather than skipping because the estimator id is not `linear.adjustment.ate`.
fn conditional_query_fixture(n: usize) -> (TabularData, Dag, ConditionalEffectQuery) {
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
    (data, g, ConditionalEffectQuery::try_new(inner).unwrap())
}

#[test]
fn conditional_effect_placebo_and_rcc_run_on_interaction_scalar() {
    let (data, g, cq) = conditional_query_fixture(120);
    let analysis = Study::tabular(data)
        .graph(g)
        .query(CausalQuery::ConditionalEffect(cq))
        .refute(RefuteSuite::PlaceboAndRcc)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(1)).unwrap();
    assert!(result.estimate.ate.is_finite());
    let names: Vec<&str> = result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
    assert!(
        names.iter().any(|n| n.contains("placebo")),
        "placebo must run on the licensed conditional scalar; got {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|n| n.contains("random.common_cause") || n.contains("random_common_cause")),
        "RCC must run on the licensed conditional scalar; got {names:?}"
    );
    assert!(
        result.diagnostics.iter().all(|d| d.code.as_ref() != "refute.validator.not_applicable"),
        "Placebo/RCC are applicable here; got {:?}",
        result.diagnostics.iter().map(|d| d.code.as_ref()).collect::<Vec<_>>()
    );
}
