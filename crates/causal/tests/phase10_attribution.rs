//! Phase 10 attribution conformance (DESIGN.md §17 / §32).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal::{
    DifferenceMeasure, DistributionChangeOptions, MechanismChangeMethod,
    attribute_distribution_change, fit_gcm, mechanism_change_detection, rank_root_causes,
};
use causal_core::{
    AllocationMethod, AttributionComponents, CachePolicy, CausalSchemaBuilder,
    ChangeAttributionQuery, ExecutionContext, MeasurementSpec, MechanismChangeQuery,
    PopulationSelector, RoleHint, ShapleyConfig, SmallRoleSet, ValueType, VariableId,
};
use causal_data::column::{Float64Column, ValidityBitmap};
use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use causal_graph::{Dag, DenseNodeId};
use causal_model::CompiledCausalModel;
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/phase10")
        .join(name)
        .join("expected.json")
}

fn two_period_chain() -> (CompiledCausalModel, TabularData) {
    let n = 80usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
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
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    for i in 0..n {
        let x = (i % 40) as f64 * 0.1;
        xv.push(x);
        yv.push(if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x });
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let fitted = fit_gcm(g, &data).unwrap();
    (fitted.model, data)
}

#[test]
fn distribution_change_y_shift_conformance() {
    let expected: Value =
        serde_json::from_str(&fs::read_to_string(fixture("distribution_change_y_shift")).unwrap())
            .unwrap();
    let (model, data) = two_period_chain();
    let query = ChangeAttributionQuery::new(
        VariableId::from_raw(1),
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
    )
    .with_components(AttributionComponents::Mechanisms)
    .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
    let opts =
        DistributionChangeOptions { measure: DifferenceMeasure::MeanDiff, n_samples: 400, seed: 2 };
    let result = attribute_distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
    let min_change = expected["total_change_min"].as_f64().unwrap();
    assert!(
        result.total_change >= min_change,
        "total_change={} < {}",
        result.total_change,
        min_change
    );
    let y = result
        .contributions
        .iter()
        .find(|c| c.component.variable() == VariableId::from_raw(1))
        .expect("y");
    let x = result
        .contributions
        .iter()
        .find(|c| c.component.variable() == VariableId::from_raw(0))
        .map_or(0.0, |c| c.contribution.abs());
    assert!(y.contribution.abs() > x, "y={} x={}", y.contribution, x);
    let ranks = rank_root_causes(&result, &ctx).unwrap();
    assert_eq!(ranks[0].component.variable(), VariableId::from_raw(1));
}

#[test]
fn mechanism_change_detect_conformance() {
    let expected: Value =
        serde_json::from_str(&fs::read_to_string(fixture("mechanism_change_detect")).unwrap())
            .unwrap();
    let (model, data) = two_period_chain();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(0), VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
        expected["significance_level"].as_f64().unwrap(),
        10,
    );
    let dets = mechanism_change_detection(
        &model,
        &data,
        &q,
        MechanismChangeMethod::MeanDiff,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
    assert!(y.changed, "{y:?}");
}
