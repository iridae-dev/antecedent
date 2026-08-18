use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec,
    RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy, ToleranceClass, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, PanelData, PanelUnit, TableView, TabularData,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte, TemporalLinearAdjustment};
use antecedent_expr::ExprId;
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::{IdentifiedEstimand, TemporalBackdoorIdentifier};

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
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../conformance/validate/refuters/expected.json"))
            .unwrap();
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(7);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    assert!((original.ate - 2.0).abs() < 1e-6);

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let report = PlaceboTreatment::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    // comparison is the two-sided p-value of zero under the placebo distribution.
    assert!(report.comparison >= 0.05, "p={}", report.comparison);
    let max = fixture["expected"]["placebo_abs_max"].as_f64().unwrap();
    assert!(report.refuted_ate.abs() < max, "mean placebo ate={}", report.refuted_ate);
}

#[test]
fn placebo_permute_near_zero_on_null() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(19);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let mut placebo = PlaceboTreatment::new();
    placebo.mode = PlaceboMode::Permute;
    placebo.replicates = 40;
    let report = placebo.refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    assert!(report.refuted_ate.abs() < 0.35, "mean placebo ate={}", report.refuted_ate);
}

#[test]
fn rcc_preserves_ate() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../conformance/validate/refuters/expected.json"))
            .unwrap();
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(11);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let report = RandomCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    let max = fixture["expected"]["random_common_cause_abs_delta_max"].as_f64().unwrap();
    assert!((report.refuted_ate - original.ate).abs() < max);
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

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
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

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
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

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let report = DataSubsetRefuter::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    assert!((report.refuted_ate - original.ate).abs() < 0.3);
}

/// A non-default `LinearAdjustmentAte` config (e.g. `se_kind = AnalyticSeKind::Hc1`) set on a
/// refuter's `estimator` field must actually reach the refit inside `refit_effect`/`fit_once`,
/// not be silently discarded in favor of a fresh default (homoskedastic) estimator. Proven by
/// calling `crate::common::refit_effect` directly with two configs that differ only in
/// `se_kind`, on the same unmutated heteroskedastic design, and asserting the resulting
/// `se_analytic` differs.
#[test]
fn refit_effect_honors_caller_se_kind() {
    use antecedent_estimate::AnalyticSeKind;

    // Heteroskedastic design: residual scale grows with z, so HC1 (heteroskedasticity-robust)
    // and homoskedastic analytic SEs are visibly different — a deterministic (noise-free) `y`
    // like `toy_confounded` would make the two indistinguishable.
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
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let ctx = ExecutionContext::for_tests(101);
    let mut noise = vec![0.0; n];
    crate::common::fill_gaussian(&mut noise, &ctx, 0x5EED_0001);
    // Residual scale grows with z: near-zero at z=0, wide at z=1.
    let y: Vec<f64> =
        (0..n).map(|i| 1.0 + 2.0 * t[i] + 3.0 * z[i] + noise[i] * (0.05 + 4.0 * z[i])).collect();
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
    let data = TabularData::new(storage);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );

    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );

    let homoskedastic = LinearAdjustmentAte::new();
    assert_eq!(homoskedastic.se_kind, AnalyticSeKind::Homoskedastic);
    let mut hc1 = LinearAdjustmentAte::new();
    hc1.se_kind = AnalyticSeKind::Hc1;

    // Same problem, same unmutated data: the only difference between these two calls is the
    // caller-configured `se_kind`, isolating whether `refit_effect` honors it.
    let home_effect =
        crate::common::refit_effect(&problem, &data, &estimand, &[], &homoskedastic, &mut ws, &ctx)
            .unwrap();
    let hc1_effect =
        crate::common::refit_effect(&problem, &data, &estimand, &[], &hc1, &mut ws, &ctx).unwrap();

    assert!(
        (home_effect.ate - hc1_effect.ate).abs() < 1e-9,
        "se_kind must not change the point estimate"
    );
    assert!(home_effect.se_analytic.is_finite() && home_effect.se_analytic > 0.0);
    assert!(hc1_effect.se_analytic.is_finite() && hc1_effect.se_analytic > 0.0);
    assert!(
        (home_effect.se_analytic - hc1_effect.se_analytic).abs() > 1e-6,
        "expected caller-configured se_kind to change the refit SE: homoskedastic={} hc1={}",
        home_effect.se_analytic,
        hc1_effect.se_analytic,
    );

    // Regression guard on the plumbing change itself: a refuter whose `estimator` field is
    // Hc1-configured must still run its full replicate loop (through `DataSubsetRefuter::refute`
    // -> `refit_effect`) without error.
    let mut refuter = DataSubsetRefuter::new();
    refuter.estimator.se_kind = AnalyticSeKind::Hc1;
    let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.informative);
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

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
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

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let mut refuter = BootstrapRefute::new();
    refuter.replicates = 100;
    let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.passed, "{:?}", report.failure_condition);
    assert!(report.comparison > 0.0, "expected a non-degenerate CI width");
}

#[test]
fn evalue_passes_moderate_threshold_for_nonnull_effect() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(31);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let report = EValue::new().refute(&problem).unwrap();
    assert!(report.comparison >= DEFAULT_EVALUE_THRESHOLD, "e_value={}", report.comparison);
    assert!(report.passed, "{:?}", report.failure_condition);
}

#[test]
fn evalue_zero_effect_fails_default_threshold() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(32);
    let mut original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    original.ate = 0.0;

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let report = EValue::new().refute(&problem).unwrap();
    // Null effect → RR = 1 → E = 1, below moderate-robustness default of 2.
    assert!((report.comparison - 1.0).abs() < 1e-12, "e_value={}", report.comparison);
    assert!(!report.passed, "null effect must fail default threshold");
}

#[test]
fn graph_refute_flags_dropping_the_true_confounder() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/validate/overlap_graph_refutation/expected.json"
    ))
    .unwrap();
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(37);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let report = GraphRefuter::new().refute(&problem, &mut ws, &ctx).unwrap();
    // Z is the only, essential confounder; dropping it biases the estimate by 1.5 of
    // a true ATE of 2 — a 75% relative change.
    assert!(!report.passed, "{:?}", report.failure_condition);
    let min = fixture["graph_refutation"]["minimum_relative_effect_change"].as_f64().unwrap();
    assert!(report.comparison > min, "relative delta={}", report.comparison);
}

#[test]
fn linear_sensitivity_reports_a_bounded_robustness_value() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/validate/confounding_sensitivity/expected.json"
    ))
    .unwrap();
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(41);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let refuter = LinearSensitivity::new();
    let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
    assert!(report.comparison > 0.0);
    assert!(report.comparison <= *refuter.partial_r2_grid.last().unwrap());
    assert_eq!(u64::from(report.replicates), fixture["expected"]["replicates"].as_u64().unwrap());
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

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
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

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let refuter = NonparametricSensitivity::new();
    let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
    assert_eq!(report.refuter.as_ref(), "sensitivity.nonparametric");
    assert!(report.comparison > 0.0);
    assert!(report.comparison <= *refuter.partial_r2_grid.last().unwrap());
}

/// The sensitivity grid is a *partial* R², so the injected confounder must be scaled by the
/// residual SD of `T` given `Z` — not its marginal SD.
///
/// Using the marginal SD calibrates against the wrong variance: the realized partial R² then
/// exceeds the nominal grid value by `Var(T)/Var(T|Z)`, so a run reported as "explained away
/// at partial R² = 0.2" actually required a far stronger confounder. In `toy_confounded`,
/// `T = 1{Z > 0.5}` is largely explained by `Z`, so the two SDs are far apart and the
/// distinction is unmissable.
#[test]
fn sensitivity_scales_by_residual_not_marginal_sd() {
    let (data, estimand, _) = toy_confounded();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(7);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );

    let ids = vec![VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let mask = data.complete_case_mask(&ids).unwrap();
    let t = data.float64_masked(VariableId::from_raw(0), &mask).unwrap();
    let z = data.float64_masked(VariableId::from_raw(2), &mask).unwrap();

    // Independent reference: simple OLS of t on z, residual SD.
    let n = t.len() as f64;
    let (mt, mz) = (t.iter().sum::<f64>() / n, z.iter().sum::<f64>() / n);
    let cov_tz: f64 = t.iter().zip(&z).map(|(&a, &b)| (a - mt) * (b - mz)).sum();
    let var_z: f64 = z.iter().map(|&b| (b - mz) * (b - mz)).sum();
    let beta = cov_tz / var_z;
    let resid: Vec<f64> = t.iter().zip(&z).map(|(&a, &b)| a - (mt + beta * (b - mz))).collect();
    let expected = crate::common::sample_sd(&resid);
    let marginal = crate::common::sample_sd(&t);

    let (got, _sd_y) = crate::sensitivity::residual_sd_pair_on_adjustment(
        &problem,
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        &mask,
    )
    .unwrap();

    assert!(
        (got - expected).abs() < 1e-9,
        "residual SD {got} != independently computed {expected}"
    );
    assert!(
        got < 0.8 * marginal,
        "Z explains most of T here, so residual SD {got} must be well below marginal {marginal}"
    );
}

fn float_payload_ptr(col: &OwnedColumn) -> *const f64 {
    match col {
        OwnedColumn::Float64(c) => c.values.as_slice().as_ptr(),
        _ => panic!("expected float64 column"),
    }
}

fn toy_panel() -> (PanelData, TabularData) {
    let u0 = TimeSeriesData::from_f64_columns(
        [
            ("t", &[0.0_f64, 1.0, 0.0, 1.0][..]),
            ("y", &[1.0, 3.0, 1.0, 3.0][..]),
            ("z", &[0.1, 0.2, 0.3, 0.4][..]),
        ],
        1,
    )
    .unwrap();
    let u1 = TimeSeriesData::from_f64_columns(
        [("t", &[1.0_f64, 0.0, 1.0][..]), ("y", &[4.0, 2.0, 4.0][..]), ("z", &[0.5, 0.6, 0.7][..])],
        1,
    )
    .unwrap();
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: u0 },
        PanelUnit { unit_id: 1, series: u1 },
    ]))
    .unwrap();
    let stacked = stack_panel_tabular(&panel).unwrap();
    (panel, stacked)
}

fn lagged_xy_series(n: usize, seed: f64) -> TimeSeriesData {
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = ((t as f64).mul_add(0.07, seed)).sin();
        y[t] = 0.8 * x[t - 1];
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn lagged_xy_graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

#[test]
fn panel_slice_template_reuses_unmutated_arcs() {
    let (panel, stacked) = toy_panel();
    let plan = PanelSliceTemplate::from_panel(&panel, &stacked).unwrap();
    let t_id = VariableId::from_raw(0);
    let n = stacked.row_count();
    let mutated_t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mutated = stacked.with_replaced_float(t_id, Arc::from(mutated_t.clone())).unwrap();
    let rebuilt = plan.apply_stacked(&mutated).unwrap();
    let copied = crate::panel_slice::copy_all_panel_from_stacked(&panel, &mutated).unwrap();

    for i in 0..2 {
        let orig = &panel.units()[i];
        let fast = &rebuilt.units()[i];
        let slow = &copied.units()[i];
        let orig_cols = orig.series.storage().columns();
        let fast_cols = fast.series.storage().columns();
        let slow_cols = slow.series.storage().columns();
        // Z (col 2) was not replaced: reuse the original unit Arc.
        assert_eq!(float_payload_ptr(&fast_cols[2]), float_payload_ptr(&orig_cols[2]));
        // Y (col 1) was not replaced either.
        assert_eq!(float_payload_ptr(&fast_cols[1]), float_payload_ptr(&orig_cols[1]));
        // T was replaced: values match the copy-all path, and do not alias the original.
        assert_ne!(float_payload_ptr(&fast_cols[0]), float_payload_ptr(&orig_cols[0]));
        match (&fast_cols[0], &slow_cols[0]) {
            (OwnedColumn::Float64(a), OwnedColumn::Float64(b)) => {
                assert_eq!(a.values.as_slice(), b.values.as_slice());
            }
            _ => panic!("expected float64 treatment"),
        }
    }
}

#[test]
fn panel_slice_template_slices_appended_column() {
    let (panel, stacked) = toy_panel();
    let plan = PanelSliceTemplate::from_panel(&panel, &stacked).unwrap();
    let extra: Vec<f64> = (0..stacked.row_count()).map(|i| i as f64 * 0.5).collect();
    let (augmented, extra_id) = stacked.with_appended_float("__rcc", Arc::from(extra)).unwrap();
    let rebuilt = plan.apply_stacked(&augmented).unwrap();
    assert_eq!(rebuilt.schema().len(), panel.schema().len() + 1);
    assert_eq!(extra_id.as_usize(), 3);
    for i in 0..2 {
        let orig_z = float_payload_ptr(&panel.units()[i].series.storage().columns()[2]);
        let new_z = float_payload_ptr(&rebuilt.units()[i].series.storage().columns()[2]);
        assert_eq!(orig_z, new_z);
        assert_eq!(rebuilt.units()[i].series.storage().columns().len(), 4);
    }
}

#[test]
fn panel_refit_effect_round_trip_matches_copy_all() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: lagged_xy_series(32, 0.1) },
        PanelUnit { unit_id: 1, series: lagged_xy_series(32, 0.4) },
    ]))
    .unwrap();
    let stacked = stack_panel_tabular(&panel).unwrap();
    let graph = lagged_xy_graph();
    let temporal_query =
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
    let id_res =
        TemporalBackdoorIdentifier::new().identify_temporal(&graph, &temporal_query).unwrap();
    let estimand = id_res.result.estimands.first().cloned().expect("identified");
    let ctx = ExecutionContext::for_tests(11);
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = 0;
    let (prep, _, _) = estimator
        .prepare_panel(
            &panel,
            &estimand,
            &temporal_query,
            &id_res.indexer,
            None,
            &ctx.kernel_policy,
        )
        .unwrap();
    let mut ws = EstimationWorkspace::default();
    let original = estimator.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    let ate_q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let temporal = TemporalRefitContext {
        indexer: &id_res.indexer,
        temporal_query: &temporal_query,
        split: None,
        kernel_policy: &ctx.kernel_policy,
        time_index: None,
        panel: Some(&panel),
    };
    let problem = RefutationProblem::new(
        &stacked,
        &estimand,
        &ate_q,
        &original,
        Some("temporal.linear.adjustment"),
        Some(temporal),
    );
    let mut prepared =
        DummyOutcome { replicates: 2, ..DummyOutcome::new() }.prepare(&problem, &ctx).unwrap();
    assert!(prepared.panel.is_some(), "prepare must compile a panel slice template");
    let attached = prepared.problem.with_panel_slices(prepared.panel.as_ref());

    let unmutated = crate::common::refit_effect(
        &attached,
        &stacked,
        &estimand,
        &[],
        &estimator.inner,
        &mut ws,
        &ctx,
    )
    .unwrap();
    assert!(
        ToleranceClass::StableFloat.close(unmutated.ate, original.ate),
        "unmutated panel refit {} != original {}",
        unmutated.ate,
        original.ate
    );

    let t_id = VariableId::from_raw(0);
    let scaled: Vec<f64> = stacked.float64_values(t_id).unwrap().iter().map(|v| v * 0.5).collect();
    let mutated = stacked.with_replaced_float(t_id, Arc::from(scaled)).unwrap();
    let fast = crate::common::refit_effect(
        &attached,
        &mutated,
        &estimand,
        &[],
        &estimator.inner,
        &mut ws,
        &ctx,
    )
    .unwrap();
    let copied = crate::panel_slice::copy_all_panel_from_stacked(&panel, &mutated).unwrap();
    let (slow_prep, _, _) = estimator
        .prepare_panel(
            &copied,
            &estimand,
            &temporal_query,
            &id_res.indexer,
            None,
            &ctx.kernel_policy,
        )
        .unwrap();
    let slow = estimator.fit(&slow_prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    assert!(
        ToleranceClass::StableFloat.close(fast.ate, slow.ate),
        "template refit {} != copy-all refit {}",
        fast.ate,
        slow.ate
    );
    assert!(
        (fast.ate - original.ate).abs() > 1e-6,
        "scaled treatment should move the panel ATE (got {})",
        fast.ate
    );

    DummyOutcome { replicates: 2, ..DummyOutcome::new() }
        .validate(&mut prepared, &mut ws, &ctx)
        .unwrap();
}

#[test]
fn prepared_refutation_compile_is_none_without_panel() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(3);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let prepared = PreparedRefutation::compile(&problem).unwrap();
    assert!(prepared.panel.is_none());
}

#[test]
fn sensitivity_gram_matches_data_pass_on_toy() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(41);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5];
    let data_ates = crate::sensitivity::grid_ates_data_pass(
        &problem,
        &mut ws,
        &ctx,
        &est,
        &grid,
        0xA7E0_000A_0000_u64,
        false,
    )
    .unwrap();
    let gram_ates = crate::sensitivity::grid_ates_gram(
        &problem,
        &est,
        &ctx,
        &grid,
        0xA7E0_000A_0000_u64,
        false,
    )
    .unwrap()
    .expect("Gram path should compile on static OLS");
    assert_eq!(data_ates.len(), gram_ates.len());
    for (i, (data_ate, gram_ate)) in data_ates.iter().zip(&gram_ates).enumerate() {
        assert!(
            ToleranceClass::BackendSensitive.close(*gram_ate, *data_ate),
            "grid[{i}]: gram={gram_ate} data-pass={data_ate}"
        );
    }
}

#[test]
fn sensitivity_gram_matches_data_pass_partial_linear_bounded_u() {
    let (data, estimand, _) = toy_confounded();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(43);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let grid = [0.01, 0.05, 0.2];
    let data_ates = crate::sensitivity::grid_ates_data_pass(
        &problem,
        &mut ws,
        &ctx,
        &est,
        &grid,
        0xA7E0_000B_0000_u64,
        true,
    )
    .unwrap();
    let gram_ates =
        crate::sensitivity::grid_ates_gram(&problem, &est, &ctx, &grid, 0xA7E0_000B_0000_u64, true)
            .unwrap()
            .expect("Gram path should compile on static OLS");
    for (i, (data_ate, gram_ate)) in data_ates.iter().zip(&gram_ates).enumerate() {
        assert!(
            ToleranceClass::BackendSensitive.close(*gram_ate, *data_ate),
            "bounded-U grid[{i}]: gram={gram_ate} data-pass={data_ate}"
        );
    }
}
