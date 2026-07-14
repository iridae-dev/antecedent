use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use causal_expr::ExprId;
use causal_identify::IdentifiedEstimand;

use super::*;

fn toy_confounded() -> (TabularData, IdentifiedEstimand, f64) {
    // True ATE = 2; Z confounds T and Y.
    let n = 400usize;
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
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
    let t: Vec<f64> = (0..n).map(|i| if z[i] > 0.5 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 3.0 * z[i]).collect();
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
    (TabularData::new(storage), estimand, 2.0)
}

#[test]
fn placebo_near_zero_on_null() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(7);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    assert!((original.ate - 2.0).abs() < 1e-6);

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let report = PlaceboTreatment::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    // comparison is the two-sided p-value of zero under the placebo distribution.
    assert!(report.comparison >= 0.05, "p={}", report.comparison);
    assert!(report.refuted_ate.abs() < 0.25, "mean placebo ate={}", report.refuted_ate);
}

#[test]
fn rcc_preserves_ate() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(11);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let report = RandomCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    assert!((report.refuted_ate - original.ate).abs() < 0.15);
}

#[test]
fn unobserved_common_cause_is_robust_to_mild_confounding() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(13);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let report = UnobservedCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.comparison >= 0.0);
    assert!(report.passed, "{:?}", report.failure_condition);
}

#[test]
fn overlap_flags_near_deterministic_treatment_assignment() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(17);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    assert!(original.overlap_report.is_none());

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let report = OverlapRefuter::new().refute(&problem).unwrap();
    assert_eq!(report.replicates, 1);
    // T is a deterministic step function of Z (t = 1{z > 0.5}); the diagnostic propensity
    // fit should show near-degenerate propensities, failing the overlap check.
    assert!(!report.passed, "{:?}", report.failure_condition);
}

#[test]
fn data_subset_preserves_ate() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(19);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let report = DataSubsetRefuter::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    assert!((report.refuted_ate - original.ate).abs() < 0.3);
}

#[test]
fn dummy_outcome_near_zero() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(23);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let report = DummyOutcome::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    // comparison is the two-sided p-value of zero under the dummy-outcome distribution.
    assert!(report.comparison >= 0.05, "p={}", report.comparison);
    assert!(report.refuted_ate.abs() < 0.25, "mean dummy ate={}", report.refuted_ate);
}

#[test]
fn bootstrap_refute_contains_original_ate() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(29);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let mut refuter = BootstrapRefute::new();
    refuter.replicates = 100;
    let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    assert!(report.comparison > 0.0, "expected a non-degenerate CI width");
}

#[test]
fn evalue_exceeds_one_for_nonnull_effect() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(31);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let report = EValue::new().refute(&problem).unwrap();
    assert!(report.comparison > 1.0, "e_value={}", report.comparison);
    assert!(report.passed, "{:?}", report.failure_condition);
}

#[test]
fn graph_refute_flags_dropping_the_true_confounder() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(37);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let report = GraphRefuter::new().refute(&problem, &mut ws, &ctx).unwrap();
    // Z is the only, essential confounder; dropping it biases the estimate by 1.5 of
    // a true ATE of 2 — a 75% relative change.
    assert!(!report.passed, "{:?}", report.failure_condition);
    assert!(report.comparison > 0.5, "relative delta={}", report.comparison);
}

#[test]
fn linear_sensitivity_reports_a_bounded_robustness_value() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(41);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let refuter = LinearSensitivity::new();
    let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.comparison > 0.0);
    assert!(report.comparison <= *refuter.partial_r2_grid.last().unwrap());
    assert_eq!(report.replicates as usize, refuter.partial_r2_grid.len());
}

#[test]
fn partial_linear_sensitivity_reports_a_bounded_robustness_value() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(43);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let refuter = PartialLinearSensitivity::new();
    let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.comparison > 0.0);
    assert!(report.comparison <= *refuter.partial_r2_grid.last().unwrap());
    assert_eq!(report.replicates as usize, refuter.partial_r2_grid.len());
}

#[test]
fn nonparametric_sensitivity_reports_a_bounded_robustness_value() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(47);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &original,
        estimator: Some("linear.adjustment.ate"),
    };
    let refuter = NonparametricSensitivity::new();
    let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
    assert_eq!(report.refuter.as_ref(), "sensitivity.nonparametric");
    assert!(report.comparison > 0.0);
    assert!(report.comparison <= *refuter.partial_r2_grid.last().unwrap());
}
