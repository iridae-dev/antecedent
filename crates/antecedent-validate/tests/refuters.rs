//! refuter / sensitivity conformance (clean-room).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_expr::ExprId;
use antecedent_identify::IdentifiedEstimand;
use antecedent_validate::{
    BootstrapRefute, DataSubsetRefuter, DummyOutcome, EValue, GraphRefuter, LinearSensitivity,
    NonparametricSensitivity, OverlapRefuter, OverlapRuleRefuter, PartialLinearSensitivity,
    PlaceboTreatment, RandomCommonCause, RefutationProblem, ReiszSensitivity,
    UnobservedCommonCause, ValidationSuite,
};
use serde_json::Value as JsonValue;

fn fixture_expected() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/estimate/refuters/expected.json");
    let raw = fs::read_to_string(path).expect("refuters expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn toy() -> (TabularData, IdentifiedEstimand) {
    let n = 400usize;
    let mut b = CausalSchemaBuilder::new();
    for (name, role) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + z[i]).collect();
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
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    (TabularData::new(storage), estimand)
}

fn problem_setup() -> (
    TabularData,
    IdentifiedEstimand,
    AverageEffectQuery,
    antecedent_estimate::EffectEstimate,
    EstimationWorkspace,
    ExecutionContext,
) {
    let (data, estimand) = toy();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(7);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    (data, estimand, query, original, ws, ctx)
}

#[test]
fn refuters_and_sensitivity_smoke() {
    let expected = fixture_expected();
    assert_eq!(expected["tolerance_class"].as_str().unwrap(), "Exact");
    let pinned = expected["validators"].as_array().unwrap();
    assert!(pinned.len() >= 14, "expected.json must pin the validator set");

    let (data, estimand, query, original, mut ws, ctx) = problem_setup();
    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
        temporal: None,
    };

    assert!(PlaceboTreatment::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(RandomCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(BootstrapRefute::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(UnobservedCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(OverlapRefuter::new().refute(&problem).unwrap().informative);
    assert!(OverlapRuleRefuter::new().refute(&problem).unwrap().informative);
    assert!(DataSubsetRefuter::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(DummyOutcome::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(EValue::new().refute(&problem).unwrap().informative);
    assert!(GraphRefuter::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(LinearSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(PartialLinearSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(NonparametricSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(ReiszSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap().informative);

    let outcomes = ValidationSuite::full_effect().run(&problem, &mut ws, &ctx).unwrap();
    assert_eq!(outcomes.len(), 14);
    assert!(ValidationSuite::reports_only(&outcomes).len() >= 12);
}
