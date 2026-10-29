//! refuter / sensitivity conformance (clean-room).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
    PlaceboTreatment, RandomCommonCause, RefutationProblem, RieszSensitivity,
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
    // Noise in the pattern `+ - - +` per block of four rows is orthogonal to the constant, to
    // `t = i % 2` and to the linear trend `z`, so least squares still recovers the coefficient 2
    // exactly while the outcome keeps the residual variation the E-value standardizes by.
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let e = if matches!(i % 4, 0 | 3) { 0.5 } else { -0.5 };
            1.0 + 2.0 * t[i] + z[i] + e
        })
        .collect();
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
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
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
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );

    // Known answers on this fixture, derived from the construction and not from the product.
    // OLS recovers the coefficient 2 exactly (the noise is orthogonal to the design).
    assert!((original.ate - 2.0).abs() < 1e-9, "ate {}", original.ate);
    // E-value of a continuous outcome (VanderWeele and Ding): d = ATE / sigma with sigma the
    // residual SD of y on (1, t, z), RR = exp(0.91 d), E = RR + sqrt(RR (RR - 1)); the reported
    // value is the smaller of the point E-value and the E-value of the 95% interval limit nearest
    // the null, ATE -/+ 1.959964 se with se = sigma / sqrt(S_tt), S_tt the sum of squares of t
    // after removing the constant and the linear trend z.
    let n = 400.0_f64;
    let t: Vec<f64> = (0..400).map(|i| f64::from(i % 2)).collect();
    let z: Vec<f64> = (0..400).map(|i| f64::from(i) / n).collect();
    let (mean_t, mean_z) = (t.iter().sum::<f64>() / n, z.iter().sum::<f64>() / n);
    let slope = t.iter().zip(&z).map(|(a, b)| (a - mean_t) * (b - mean_z)).sum::<f64>()
        / z.iter().map(|b| (b - mean_z).powi(2)).sum::<f64>();
    let s_tt: f64 =
        t.iter().zip(&z).map(|(a, b)| ((a - mean_t) - slope * (b - mean_z)).powi(2)).sum();
    let sigma = (400.0_f64 * 0.25 / (n - 3.0)).sqrt();
    let e_value = |effect: f64| {
        let rr = (0.91 * effect / sigma).exp();
        let rr = if rr < 1.0 { 1.0 / rr } else { rr };
        rr + (rr * (rr - 1.0)).sqrt()
    };
    let limit = 2.0 - 1.959_963_984_540_054 * sigma / s_tt.sqrt();
    let want = e_value(2.0).min(e_value(limit));
    let reported = EValue::new().refute(&problem).unwrap().comparison;
    assert!((reported - want).abs() < 1e-6 * want, "e-value {reported} vs {want}");

    // OLS-gated noise/subset refuters have no power against the causal claim: they
    // must report informative=false, as does the descriptive leave-one-out check. The remaining
    // validators keep informative=true.
    assert!(!PlaceboTreatment::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(!RandomCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(BootstrapRefute::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(UnobservedCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(OverlapRefuter::new().refute(&problem).unwrap().informative);
    assert!(OverlapRuleRefuter::new().refute(&problem).unwrap().informative);
    assert!(!DataSubsetRefuter::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(!DummyOutcome::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(EValue::new().refute(&problem).unwrap().informative);
    // Leave-one-out covariate dropping is descriptive: a real confounder is supposed to move it.
    assert!(!GraphRefuter::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(LinearSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(PartialLinearSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(NonparametricSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap().informative);
    assert!(RieszSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap().informative);

    let outcomes = ValidationSuite::full_effect().run(&problem, &mut ws, &ctx).unwrap();
    assert_eq!(outcomes.len(), 14);
    assert!(ValidationSuite::reports_only(&outcomes).len() >= 12);
}
