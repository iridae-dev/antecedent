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

fn with_invalid_treatment_row(data: &TabularData, row: usize) -> TabularData {
    let n = data.row_count();
    let storage = data.storage();
    let OwnedColumn::Float64(t) = &storage.columns()[0] else {
        panic!("toy treatment is float64");
    };
    let mut bytes = vec![0xFFu8; n.div_ceil(8)];
    bytes[row / 8] &= !(1 << (row % 8));
    let validity = ValidityBitmap::from_bytes(bytes, n).unwrap();
    let mut cols = storage.columns().to_vec();
    cols[0] = OwnedColumn::Float64(Float64Column::new(t.id, t.values.clone(), validity).unwrap());
    let storage = OwnedColumnarStorage::try_new(
        storage.schema().clone(),
        cols,
        storage.analysis_mask().cloned(),
        storage.weights().map(Arc::from),
    )
    .unwrap();
    TabularData::new(storage)
}

#[test]
fn placebo_ols_is_non_informative() {
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
    // OLS recovers ~0 on independent noise by construction: not a falsifier.
    assert!(!report.informative, "OLS placebo must not claim to falsify the causal claim");
    let max = fixture["expected"]["placebo_abs_max"].as_f64().unwrap();
    assert!(report.refuted_ate.abs() < max, "mean placebo ate={}", report.refuted_ate);
}

#[test]
fn placebo_permute_ols_is_non_informative() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../conformance/validate/refuters/expected.json"))
            .unwrap();
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
    assert!(!report.informative, "OLS placebo permute must not claim to falsify");
    let max = fixture["expected"]["placebo_abs_max"].as_f64().unwrap();
    assert!(report.refuted_ate.abs() < max, "mean placebo ate={}", report.refuted_ate);
}

#[test]
fn rcc_ols_is_non_informative() {
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
    assert!(!report.informative, "OLS RCC must not claim to falsify the causal claim");
    let max = fixture["expected"]["random_common_cause_abs_delta_max"].as_f64().unwrap();
    assert!((report.refuted_ate - original.ate).abs() < max);
}

#[test]
fn unobserved_common_cause_is_robust_to_mild_confounding() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../conformance/validate/refuters/expected.json"))
            .unwrap();
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
    // With strengths in residual-SD units the expected relative shift is
    // |a b - rho a^2| / (1 + a^2) / |rho| for a = b = 0.5. Here Y|Z = 2 (T|Z), so rho = 1 and the
    // expected shift is exactly 0; what remains is the O(1/sqrt(n)) sampling correlation of the
    // 20 drawn confounders with the design (n = 400), well below the failure threshold of 1.
    let bound = fixture["expected"]["unobserved_common_cause_max_relative_shift"].as_f64().unwrap();
    assert!(
        report.comparison < bound,
        "relative shift {} must stay below the analytic bound {bound}",
        report.comparison
    );
}

/// Adversarial case: a weak effect (partial correlation about 0.2) and a strong simulated
/// confounder (six residual SDs on each of treatment and outcome) must be reported as able to
/// swamp the estimate. Expected relative shift (a b - rho a^2) / (1 + a^2) / rho with
/// a = b = 6 is about 0.97 (1 - rho) / rho, roughly 4, far above the threshold of 1.
#[test]
fn unobserved_common_cause_fails_when_the_confounder_can_swamp_a_weak_effect() {
    let n = 400_usize;
    let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let w = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
            0.2 * t[i] + 0.5 * w + 0.3 * (i % 3) as f64
        })
        .collect();
    let data = crate::test_support::tabular(&[t, y, z]);
    let estimand = crate::test_support::backdoor(1);
    let query = crate::test_support::ate_query();
    let (original, mut ws, ctx) = crate::test_support::linear_original(&data, &estimand, &query);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let refuter = UnobservedCommonCause {
        effect_on_treatment: 6.0,
        effect_on_outcome: 6.0,
        ..UnobservedCommonCause::new()
    };
    let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
    assert!(!report.passed, "comparison={}", report.comparison);
    assert!(report.comparison > 2.0, "comparison={}", report.comparison);
}

/// Complete-case rows are fixed across the leave-one-out drops: a covariate with missing rows
/// must not change the estimation sample when it is dropped.
#[test]
fn graph_refute_holds_the_complete_case_sample_fixed_across_drops() {
    // z1 is missing on rows with i % 4 in {0, 1}; on exactly those rows the outcome carries an
    // extra 5 t. The full-adjustment fit sees only rows i % 4 in {2, 3}. Dropping z1 must refit on
    // those same rows, so the estimate barely moves; if the hidden rows re-entered, their +5 t
    // would shift the treatment effect by a large fraction of itself.
    let n = 400_usize;
    let z0: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let z1: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64).collect();
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let hidden = i % 4 < 2;
            1.0 + 2.0 * t[i]
                + 3.0 * z0[i]
                + 0.3 * (i % 3) as f64
                + if hidden { 5.0 * t[i] } else { 0.0 }
        })
        .collect();
    let complete = crate::test_support::tabular(&[t, y, z0, z1]);
    let storage = complete.storage();
    let OwnedColumn::Float64(z1_col) = &storage.columns()[3] else { panic!("float64") };
    let mut bytes = vec![0u8; n.div_ceil(8)];
    for i in (0..n).filter(|i| i % 4 >= 2) {
        bytes[i / 8] |= 1 << (i % 8);
    }
    let validity = ValidityBitmap::from_bytes(bytes, n).unwrap();
    let mut cols = storage.columns().to_vec();
    cols[3] = OwnedColumn::Float64(
        Float64Column::new(z1_col.id, z1_col.values.clone(), validity).unwrap(),
    );
    let data = TabularData::new(
        OwnedColumnarStorage::try_new(storage.schema().clone(), cols, None, None).unwrap(),
    );
    let estimand = crate::test_support::backdoor(2);
    let query = crate::test_support::ate_query();
    let (original, mut ws, ctx) = crate::test_support::linear_original(&data, &estimand, &query);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let report = GraphRefuter::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert_eq!(report.replicates, 2);
    assert!(
        report.comparison < 0.2,
        "dropping a covariate re-admitted its missing rows: relative change {}",
        report.comparison
    );
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

/// `t = 1{z > 0.5}` separates the arms completely, so the diagnostic logistic fit flags
/// separation. The ridge-regularized scores that fit returns would shrink exactly the extreme
/// values that evidence the violation, so every propensity-based validator reports the
/// positivity failure itself instead of judging the shrunken scores.
#[test]
#[allow(
    clippy::float_cmp,
    reason = "the asserted values are exact by construction of the fixture (a clamped or passed-through constant), so exact equality is intended"
)]
fn separated_propensity_fit_is_reported_as_a_positivity_failure() {
    let (data, estimand, _) = toy_confounded();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let (original, mut ws, ctx) = crate::test_support::linear_original(&data, &estimand, &query);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let overlap = OverlapRefuter::new().refute(&problem).unwrap();
    let rule = OverlapRuleRefuter::new().refute(&problem).unwrap();
    let riesz = RieszSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap();
    for report in [&overlap, &rule, &riesz] {
        assert!(!report.passed, "{report:?}");
        let message = report.failure_condition.as_deref().unwrap();
        assert!(message.contains("positivity violation"), "{}: {message}", report.refuter);
    }
    // Riesz reports no robustness at all: the representer is unbounded under separation.
    assert_eq!(riesz.comparison, 0.0);
}

#[test]
fn continuous_overlap_comparison_is_unsupported_mass() {
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
    // T tracks Z almost perfectly, so doses 0 and 1 lie outside the residual
    // interval on most rows. comparison must be unsupported mass (near 1),
    // matching binary overlap.assessment — not the supported fraction (near 0).
    let t: Vec<f64> = z.iter().map(|&z| z + 0.01).collect();
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
    let data = TabularData::new(storage);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let original = antecedent_estimate::EffectEstimate::new(
        2.0,
        0.1,
        AssumptionSet::new(),
        antecedent_estimate::OverlapPolicy::ExplicitOverride,
    );
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let support = OverlapRefuter::new().refute(&problem).unwrap();
    assert_eq!(support.refuter.as_ref(), "overlap.continuous_support");
    assert!(!support.passed);
    assert!(
        support.comparison > 0.5,
        "comparison={} (expected unsupported mass, not supported fraction)",
        support.comparison
    );
    let rule = OverlapRuleRefuter::new().refute(&problem).unwrap();
    assert_eq!(rule.refuter.as_ref(), "overlap.continuous_rule");
    assert!(rule.comparison > 0.5, "comparison={}", rule.comparison);
}

#[test]
fn data_subset_ols_is_non_informative() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../conformance/validate/refuters/expected.json"))
            .unwrap();
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
    assert!(!report.informative, "OLS data-subset must not claim to falsify the causal claim");
    let max = fixture["expected"]["subset_abs_delta_max"].as_f64().unwrap();
    assert!((report.refuted_ate - original.ate).abs() < max);
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
    assert!(!report.informative);
}

#[test]
fn dummy_outcome_ols_is_non_informative() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../conformance/validate/refuters/expected.json"))
            .unwrap();
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
    assert!(!report.informative, "OLS dummy-outcome must not claim to falsify the causal claim");
    let max = fixture["expected"]["dummy_outcome_abs_max"].as_f64().unwrap();
    assert!(report.refuted_ate.abs() < max, "mean dummy ate={}", report.refuted_ate);
}

#[test]
fn ols_refuters_do_not_earn_a_pass_from_p_equals_one() {
    // Historical bug: NaN replicates collapsed to p=1.0, and pass-only tests treated
    // that as earning the falsifier claim. Earn the opposite: non-finite replicates
    // fail closed, and OLS-gated refuters are non-informative even when p is large.
    // A NaN replicate is an operational error: neither the pass (p = 1) it used to become nor a
    // refutation (p = 0) it later became.
    assert!(matches!(
        crate::common::replicate_p_value(&[0.1, f64::NAN, 0.2, 0.15], 0.0),
        Err(ValidationError::Estimation(_))
    ));

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
    for (name, report) in [
        ("placebo", PlaceboTreatment::new().refute(&problem, &mut ws, &ctx).unwrap()),
        ("rcc", RandomCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap()),
        ("subset", DataSubsetRefuter::new().refute(&problem, &mut ws, &ctx).unwrap()),
        ("dummy", DummyOutcome::new().refute(&problem, &mut ws, &ctx).unwrap()),
    ] {
        assert!(
            !report.informative,
            "{name}: large p under OLS must not be presented as an informative falsifier \
             (comparison={})",
            report.comparison
        );
    }
}

#[test]
fn bootstrap_refute_contains_original_ate() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../conformance/validate/refuters/expected.json"))
            .unwrap();
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
    assert_eq!(
        report.passed,
        fixture["expected"]["bootstrap_contains_original"].as_bool().unwrap(),
        "{:?}",
        report.failure_condition
    );
    assert!(report.comparison > 0.0, "expected a non-degenerate CI width");
}

/// Continuous outcome with noise orthogonal to `[1, t, z]`: `y = 1 + 2t + 3z + e`, where
/// `e = +-0.5` in the pattern `+ - - +` over each block of four rows. That pattern sums to zero
/// against the constant, against `t = i % 2`, and against the linear trend `z = i / n` (block
/// sums `4k - (4k+1) - (4k+2) + (4k+3) = 0`), so least squares recovers the coefficient 2
/// exactly and the residual is `e`, with `RSS = n * 0.25 = 100` and `n - p = 397` residual
/// degrees of freedom for `p = 3`.
fn evalue_continuous_data() -> (TabularData, IdentifiedEstimand) {
    let n = 400_usize;
    let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let e = if matches!(i % 4, 0 | 3) { 0.5 } else { -0.5 };
            1.0 + 2.0 * t[i] + 3.0 * z[i] + e
        })
        .collect();
    (crate::test_support::tabular(&[t, y, z]), crate::test_support::backdoor(1))
}

/// VanderWeele-Ding E-value of a standardized mean difference `d`, written out independently of
/// the crate: `RR = exp(0.91 |d|)`, `E = RR + sqrt(RR (RR - 1))`.
fn evalue_of_smd(d: f64) -> f64 {
    let rr = (0.91 * d.abs()).exp();
    rr + (rr * (rr - 1.0)).sqrt()
}

#[test]
fn evalue_uses_the_residual_sd_of_the_outcome_regression() {
    let (data, estimand) = evalue_continuous_data();
    let query = crate::test_support::ate_query();
    let (mut original, _, _) = crate::test_support::linear_original(&data, &estimand, &query);
    assert!((original.ate - 2.0).abs() < 1e-9, "fixture must recover ATE 2, got {}", original.ate);
    // Point estimate only: a degenerate interval isolates the point-estimate conversion.
    original.se_analytic = 0.0;
    original.se_bootstrap = None;
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let report = EValue::new().refute(&problem).unwrap();
    // sigma = sqrt(RSS / (n - p)) = sqrt(100 / 397); the marginal SD of y (about 1.7) would give a
    // d less than half as large.
    let sigma = (100.0_f64 / 397.0).sqrt();
    let expected = evalue_of_smd(2.0 / sigma);
    assert!(
        (report.comparison - expected).abs() < 1e-7 * expected,
        "e_value={} expected={expected}",
        report.comparison
    );
    assert!(report.passed && report.informative, "{:?}", report.failure_condition);
}

#[test]
#[allow(
    clippy::float_cmp,
    reason = "the asserted values are exact by construction of the fixture (a clamped or passed-through constant), so exact equality is intended"
)]
fn evalue_gates_on_the_confidence_limit_not_only_the_point_estimate() {
    let (data, estimand) = evalue_continuous_data();
    let query = crate::test_support::ate_query();
    let (mut original, _, _) = crate::test_support::linear_original(&data, &estimand, &query);
    let sigma = (100.0_f64 / 397.0).sqrt();
    let problem_with = |original: &antecedent_estimate::EffectEstimate| {
        EValue::new()
            .refute(&RefutationProblem::new(
                &data,
                &estimand,
                &query,
                original,
                Some("linear.adjustment.ate"),
                None,
            ))
            .unwrap()
    };
    // A very imprecise estimate (se 2 against ATE 2): the 95% interval covers the null, so the
    // interval-limit E-value is 1 and the check fails although the point E-value is large.
    original.se_analytic = 2.0;
    original.se_bootstrap = None;
    let wide = problem_with(&original);
    assert_eq!(wide.comparison, 1.0);
    assert!(!wide.passed);
    // se 0.5: the lower limit 2 - 1.96 * 0.5 = 1.02 does not cover the null; the E-value is that
    // limit's, smaller than the point estimate's.
    original.se_analytic = 0.5;
    let narrow = problem_with(&original);
    let limit = 2.0 - 1.959_963_984_540_054 * 0.5;
    let expected = evalue_of_smd(limit / sigma);
    assert!(
        (narrow.comparison - expected).abs() < 1e-7 * expected,
        "e_value={} expected={expected}",
        narrow.comparison
    );
    assert!(narrow.comparison < evalue_of_smd(2.0 / sigma));
    // No usable standard error: the point E-value is reported but cannot support a pass.
    original.se_analytic = f64::NAN;
    let unknown = problem_with(&original);
    assert!(!unknown.passed && !unknown.informative);
}

#[test]
fn evalue_zero_effect_fails_default_threshold() {
    let (data, estimand) = evalue_continuous_data();
    let query = crate::test_support::ate_query();
    let (mut original, _, _) = crate::test_support::linear_original(&data, &estimand, &query);
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

/// Binary outcome: the ATE is a risk difference, so the arm risks (not `d / 0.91`) give the risk
/// ratio. Control risk 0.10 (20 of 200), treated risk 0.20 (40 of 200): RR = 2, E = 2 + sqrt(2).
#[test]
fn evalue_of_a_binary_outcome_uses_the_arm_risk_ratio() {
    let n = 400_usize;
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    // Within each arm (rows of equal parity) every 10th row (control) or 5th row (treated) is 1.
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let k = i / 2;
            let one = if i % 2 == 0 { k % 10 == 0 } else { k % 5 == 0 };
            f64::from(u8::from(one))
        })
        .collect();
    let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let data = crate::test_support::tabular(&[t, y, z]);
    let estimand = crate::test_support::backdoor(1);
    let query = crate::test_support::ate_query();
    let (mut original, _, _) = crate::test_support::linear_original(&data, &estimand, &query);
    original.ate = 0.10;
    original.se_analytic = 0.0;
    original.se_bootstrap = None;
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let report = EValue::new().refute(&problem).unwrap();
    let expected = 2.0 + 2.0_f64.sqrt();
    assert!(
        (report.comparison - expected).abs() < 1e-12,
        "e_value={} expected={expected}",
        report.comparison
    );
    // The continuous conversion would have given d = 0.1 / sd(y) ~ 0.27, RR ~ 1.28, E ~ 1.9.
    assert!(report.passed, "{:?}", report.failure_condition);
}

#[test]
fn graph_refute_reports_the_confounder_dependence_descriptively() {
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
    // a true ATE of 2 — a 75% relative change. A moving estimate is what a confounder does, so
    // the result is descriptive (`informative: false`) even though `passed` records the move.
    assert!(!report.informative, "leave-one-out sensitivity is not a falsifier");
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
    // Here Y|Z = 2 (T|Z) exactly, so the partial correlation is 1 and the analytic tipping
    // partial R2 is |rho| / (1 + |rho|) = 0.5: no smaller grid value (0.3 at most) can explain
    // the effect away. The report is the first grid value at or beyond it.
    assert!(
        report.comparison >= 0.5,
        "comparison={} is below the analytic tipping partial R2 0.5",
        report.comparison
    );
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
    // Analytic tipping partial R2 is 0.5 (partial correlation 1, see the linear test).
    assert!(
        report.comparison >= 0.5,
        "comparison={} is below the analytic tipping partial R2 0.5",
        report.comparison
    );
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
    // Kernel residualization leaves T|Z and Y|Z almost perfectly correlated here (Y|Z = 2 T|Z
    // up to the smoother's small boundary error), so the tipping partial R2 is near 0.5: at least
    // the 0.3 grid value below it does not explain the effect away.
    assert!(
        report.comparison >= 0.5,
        "comparison={} must be at or beyond the analytic tipping value ~0.5",
        report.comparison
    );
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
#[allow(clippy::too_many_lines)]
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
    let suite = ValidationSuite::new().with(ValidatorId::Overlap);
    let plain = suite.run(&problem, &mut ws, &ctx).unwrap();
    let warmed = suite
        .run_with_propensity(
            &problem,
            &mut ws,
            &mut antecedent_stats::PropensityWorkspace::default(),
            &ctx,
        )
        .unwrap();
    assert_eq!(ValidationSuite::not_applicable_only(&plain).len(), 1);
    assert_eq!(
        ValidationSuite::not_applicable_only(&plain),
        ValidationSuite::not_applicable_only(&warmed)
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

#[test]
fn sensitivity_gram_matches_data_pass_when_treatment_has_invalids() {
    // Both sensitivity paths must retain the original missingness pattern.
    let (complete, estimand, _) = toy_confounded();
    let data = with_invalid_treatment_row(&complete, 10);
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
    let grid = [0.01, 0.1, 0.3];
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
    for (i, (data_ate, gram_ate)) in data_ates.iter().zip(&gram_ates).enumerate() {
        assert!(
            ToleranceClass::BackendSensitive.close(*gram_ate, *data_ate),
            "invalid-T grid[{i}]: gram={gram_ate} data-pass={data_ate}"
        );
    }
}

fn routing_problem(
    treatment: Vec<f64>,
    outcome_valid: Option<Vec<bool>>,
) -> (TabularData, IdentifiedEstimand, AverageEffectQuery, antecedent_estimate::EffectEstimate) {
    let n = treatment.len();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * treatment[i] + 0.01 * i as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
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
    let y_validity = match outcome_valid {
        None => ValidityBitmap::all_valid(n),
        Some(flags) => {
            let mut bytes = vec![0u8; n.div_ceil(8)];
            for (i, keep) in flags.iter().enumerate() {
                if *keep {
                    bytes[i / 8] |= 1 << (i % 8);
                }
            }
            ValidityBitmap::from_bytes(bytes, n).unwrap()
        }
    };
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(treatment),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), y_validity).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let original = antecedent_estimate::EffectEstimate::new(
        2.0,
        0.1,
        AssumptionSet::new(),
        antecedent_estimate::OverlapPolicy::ExplicitOverride,
    );
    (data, estimand, query, original)
}

fn assert_treatment_route(
    name: &str,
    treatment: Vec<f64>,
    outcome_valid: Option<Vec<bool>>,
    binary: bool,
) {
    let (data, estimand, query, original) = routing_problem(treatment, outcome_valid);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    assert_eq!(
        crate::common::binary_treatment(&problem).unwrap(),
        binary,
        "{name}: binary_treatment mismatch"
    );
    let overlap = OverlapRefuter::new()
        .refute(&problem)
        .unwrap_or_else(|err| panic!("{name}: overlap refute failed ({err})"));
    if binary {
        assert_eq!(overlap.refuter.as_ref(), "overlap.assessment", "{name}");
    } else {
        assert_eq!(overlap.refuter.as_ref(), "overlap.continuous_support", "{name}");
    }
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(4);
    let outcomes = ValidationSuite::new()
        .with(ValidatorId::Riesz)
        .run(&problem, &mut ws, &ctx)
        .unwrap_or_else(|err| panic!("{name}: Riesz suite failed ({err})"));
    let riesz_na = ValidationSuite::not_applicable_only(&outcomes)
        .iter()
        .any(|(id, _)| *id == ValidatorId::Riesz);
    assert_eq!(riesz_na, !binary, "{name}: Riesz NA={riesz_na} expected for binary={binary}");
}

#[test]
fn refutation_report_mixture_weighted_is_mass_weighted_and_fail_closed() {
    let pass = RefutationReport::new("overlap.assessment", 2.0, 0.1, 0.2, true, true, None, 4);
    let fail = RefutationReport::new(
        "overlap.assessment",
        2.0,
        0.4,
        0.8,
        true,
        false,
        Some(Arc::from("bad overlap")),
        2,
    );
    let mixed = RefutationReport::mixture_weighted(&[(0.5, &pass), (0.3, &fail)]).unwrap();
    assert_eq!(mixed.refuter.as_ref(), "overlap.assessment");
    assert!((mixed.comparison - (0.5 * 0.2 + 0.3 * 0.8) / 0.8).abs() < 1e-12);
    assert!((mixed.refuted_ate - (0.5 * 0.1 + 0.3 * 0.4) / 0.8).abs() < 1e-12);
    assert!(!mixed.passed, "mixture must fail if any contributing atom fails");
    assert_eq!(mixed.replicates, 4);
    assert!(RefutationReport::mixture_weighted(&[(0.0, &pass)]).is_none());
    let only_pass = RefutationReport::mixture_weighted(&[(0.5, &pass), (0.0, &fail)]).unwrap();
    assert!(only_pass.passed);
}

#[test]
fn query_refutation_plan_mixes_mediation_atoms_fail_closed() {
    let pass =
        RefutationReport::new("mediation.placebo_mediator", 0.0, 0.01, 0.4, true, true, None, 20);
    let fail = RefutationReport::new(
        "mediation.placebo_mediator",
        0.0,
        0.2,
        0.01,
        true,
        false,
        Some(Arc::from("mediation contrast is inconsistent with the refuter target")),
        20,
    );
    let rcc = RefutationReport::new(
        "mediation.random_common_cause",
        0.44,
        0.45,
        0.6,
        true,
        true,
        None,
        20,
    );
    let mixed =
        QueryRefutationPlan::mix_weighted([(0.7, vec![pass, rcc.clone()]), (0.3, vec![fail, rcc])]);
    assert_eq!(mixed.len(), 2);
    assert_eq!(mixed[0].refuter.as_ref(), "mediation.placebo_mediator");
    assert!(!mixed[0].passed);
    assert_eq!(mixed[1].refuter.as_ref(), "mediation.random_common_cause");
    assert!(mixed[1].passed);
    assert_eq!(
        QueryRefutationPlan::temporal_mediation(true),
        QueryRefutationPlan::TemporalMediation { full: true }
    );
}

#[test]
fn average_effect_treatment_routing_matrix() {
    let n = 40;
    let two_valued: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    assert_treatment_route("two-valued 0/1", two_valued, None, true);

    let continuous: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    assert_treatment_route("continuous", continuous, None, false);

    let integer_dose: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
    assert_treatment_route("integer dosage", integer_dose, None, false);

    let categorical: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    assert_treatment_route("categorical >2", categorical, None, false);

    let non_unit_two_valued: Vec<f64> =
        (0..n).map(|i| if i % 2 == 0 { 2.0 } else { 5.0 }).collect();
    assert_treatment_route("two-valued non-unit", non_unit_two_valued, None, false);

    let degenerate: Vec<f64> = vec![1.0; n];
    let (data, estimand, query, original) = routing_problem(degenerate, None);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    assert!(crate::common::binary_treatment(&problem).unwrap());
    let overlap = match OverlapRefuter::new().refute(&problem) {
        Ok(report) => report,
        Err(err) => {
            assert!(
                crate::common::binary_treatment(&problem).unwrap(),
                "one-level treatment must stay binary even if overlap GLM fails ({err})"
            );
            return;
        }
    };
    assert_eq!(
        overlap.refuter.as_ref(),
        "overlap.assessment",
        "one-level 0/1 treatment stays on the binary overlap path, got {}",
        overlap.refuter
    );

    let mut mixed = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect::<Vec<_>>();
    mixed[3] = 2.5;
    let mut valid = vec![true; n];
    valid[3] = false;
    assert_treatment_route("complete-case remaining binary", mixed, Some(valid), true);

    let all_missing = vec![false; n];
    let empty_cases: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let (data, estimand, query, original) = routing_problem(empty_cases, Some(all_missing));
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    assert!(
        !crate::common::binary_treatment(&problem).unwrap(),
        "empty complete-case remainder must not vacuously enter the binary path"
    );
}

#[test]
fn refuter_replacement_preserves_missing_rows() {
    let (complete, _, _) = toy_confounded();
    let data = with_invalid_treatment_row(&complete, 10);
    let replaced = crate::common::with_replaced_float(
        &data,
        VariableId::from_raw(0),
        Arc::from(vec![0.0; data.row_count()]),
    )
    .unwrap();
    let mask = replaced.complete_case_mask(&[VariableId::from_raw(0)]).unwrap();
    assert!(!mask[10], "replacement must not resurrect a missing treatment");
    assert_eq!(mask.iter().filter(|&&valid| valid).count(), data.row_count() - 1);
}

#[test]
fn placebo_permutation_is_invariant_to_missing_payloads() {
    let (complete, estimand, _) = toy_confounded();
    let treatment = VariableId::from_raw(0);
    let query = AverageEffectQuery::binary_ate(treatment, VariableId::from_raw(1));
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let ctx = ExecutionContext::for_tests(47);
    let mut ws = EstimationWorkspace::default();
    let mut results = Vec::new();
    for payload in [0.0, f64::NAN] {
        let mut values = complete.float64_values(treatment).unwrap();
        values[10] = payload;
        let data = complete.with_replaced_float(treatment, Arc::from(values)).unwrap();
        let data = with_invalid_treatment_row(&data, 10);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
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
        placebo.replicates = 4;
        results.push(placebo.refute(&problem, &mut ws, &ctx).unwrap().refuted_ate);
    }
    assert!((results[0] - results[1]).abs() < 1e-12);
}

#[test]
fn sensitivity_reports_respect_treatment_contrast_units() {
    let (data, estimand, _) = toy_confounded();
    let treatment = VariableId::from_raw(0);
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let ctx = ExecutionContext::for_tests(47);
    let mut ws = EstimationWorkspace::default();
    let mut reports = Vec::new();
    for delta in [1.0, 2.0, -1.0] {
        let mut query = AverageEffectQuery::binary_ate(treatment, VariableId::from_raw(1));
        query.active =
            antecedent_core::Intervention::set(treatment, antecedent_core::Value::Float64(delta));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let nonparametric =
            NonparametricSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap();
        let unobserved = UnobservedCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap();
        reports.push((
            nonparametric.refuted_ate / delta,
            nonparametric.comparison,
            unobserved.comparison,
        ));
    }
    for report in &reports[1..] {
        assert!((report.0 - reports[0].0).abs() < 1e-12);
        assert!((report.1 - reports[0].1).abs() < 1e-12);
        assert!((report.2 - reports[0].2).abs() < 1e-12);
    }
}

/// Refitter that always returns a fixed estimate (its `ate` may be non-finite).
#[derive(Debug)]
struct ConstantRefit(antecedent_estimate::EffectEstimate);

impl crate::common::EffectRefit for ConstantRefit {
    fn refit(
        &self,
        _data: &TabularData,
        _extra_contemporaneous: &[VariableId],
        _ctx: &ExecutionContext,
    ) -> Result<antecedent_estimate::EffectEstimate, ValidationError> {
        Ok(self.0.clone())
    }
}

/// A refit that returns `NaN` is a failed computation: every replicate-based refuter and the
/// leave-one-out check must surface an error, never a pass (`p = 1`) nor a silent skip.
#[test]
fn a_non_finite_refit_effect_is_an_error_in_every_replicate_refuter() {
    let (data, estimand, _) = toy_confounded();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let (original, mut ws, ctx) = crate::test_support::linear_original(&data, &estimand, &query);
    let mut broken = original.clone();
    broken.ate = f64::NAN;
    let refit = ConstantRefit(broken);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    )
    .with_effect_refit(&refit);
    let is_operational_error = |result: Result<crate::RefutationReport, ValidationError>| {
        matches!(result, Err(ValidationError::Estimation(_)))
    };
    assert!(is_operational_error(PlaceboTreatment::new().refute(&problem, &mut ws, &ctx)));
    assert!(is_operational_error(DummyOutcome::new().refute(&problem, &mut ws, &ctx)));
    assert!(is_operational_error(RandomCommonCause::new().refute(&problem, &mut ws, &ctx)));
    assert!(is_operational_error(DataSubsetRefuter::new().refute(&problem, &mut ws, &ctx)));
    assert!(is_operational_error(UnobservedCommonCause::new().refute(&problem, &mut ws, &ctx)));
    assert!(is_operational_error(GraphRefuter::new().refute(&problem, &mut ws, &ctx)));
    let permute = PlaceboTreatment { mode: PlaceboMode::Permute, ..PlaceboTreatment::new() };
    assert!(is_operational_error(permute.refute(&problem, &mut ws, &ctx)));
    // The suite renders the same failure as `Failed`, "not evidence", rather than a verdict.
    let outcomes =
        ValidationSuite::new().with(ValidatorId::Placebo).run(&problem, &mut ws, &ctx).unwrap();
    assert!(matches!(outcomes[0], ValidationOutcome::Failed { .. }), "{:?}", outcomes[0]);
}

/// Replicate refuters centre on the same estimator's full-sample refit. A published estimate
/// that differs from the refit (a Bayesian posterior mean shrunk toward its prior, say) must not
/// make a stability check fail: here the published number is deliberately 0.3 away from the exact
/// least-squares effect 2, while every perturbed refit reproduces 2.
#[test]
#[allow(
    clippy::float_cmp,
    reason = "the asserted values are exact by construction of the fixture (a clamped or passed-through constant), so exact equality is intended"
)]
fn perturbation_refuters_centre_on_the_full_sample_refit_not_the_published_estimate() {
    let (data, estimand, _) = toy_confounded();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let (mut original, mut ws, ctx) =
        crate::test_support::linear_original(&data, &estimand, &query);
    assert!((original.ate - 2.0).abs() < 1e-9);
    original.ate = 2.3;
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("bayesian.temporal.gcomp"),
        None,
    );
    let rcc = RandomCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(rcc.passed, "{:?}", rcc.failure_condition);
    assert_eq!(rcc.comparison, 1.0);
    assert_eq!(rcc.original_ate, 2.3, "the report still names the published estimate");
    let subset = DataSubsetRefuter::new().refute(&problem, &mut ws, &ctx).unwrap();
    assert!(subset.passed, "{:?}", subset.failure_condition);
}

/// A replicate count beyond the stream-alias bound is refused rather than silently sharing
/// noise streams between refuters.
#[test]
fn replicate_counts_beyond_the_stream_alias_bound_are_refused() {
    let (data, estimand, _) = toy_confounded();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let (original, mut ws, ctx) = crate::test_support::linear_original(&data, &estimand, &query);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let placebo = PlaceboTreatment { replicates: 5000, ..PlaceboTreatment::new() };
    assert!(matches!(
        placebo.refute(&problem, &mut ws, &ctx),
        Err(ValidationError::NotApplicable { .. })
    ));
}

/// Cancellation is polled inside replicate loops.
#[test]
fn replicate_loops_stop_on_cancellation() {
    let (data, estimand, _) = toy_confounded();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let (original, mut ws, ctx) = crate::test_support::linear_original(&data, &estimand, &query);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    ctx.cancellation.cancel();
    assert_eq!(
        PlaceboTreatment::new().refute(&problem, &mut ws, &ctx).unwrap_err(),
        ValidationError::Cancelled
    );
    assert_eq!(
        RandomCommonCause::new().refute(&problem, &mut ws, &ctx).unwrap_err(),
        ValidationError::Cancelled
    );
    assert_eq!(
        BootstrapRefute::new().refute(&problem, &mut ws, &ctx).unwrap_err(),
        ValidationError::Cancelled
    );
}

/// Composed refitter whose aligned-row refit is a stand-in with a fixed block length.
#[derive(Debug)]
struct FixedBlockRefit {
    rows: usize,
    block_length: usize,
}

impl crate::common::EffectRefit for FixedBlockRefit {
    fn refit(
        &self,
        _data: &TabularData,
        _extra_contemporaneous: &[VariableId],
        _ctx: &ExecutionContext,
    ) -> Result<antecedent_estimate::EffectEstimate, ValidationError> {
        Err(ValidationError::estimation_msg("not used by bootstrap.ci_coverage"))
    }

    fn prepare_aligned(
        &self,
        _data: &TabularData,
        _ctx: &ExecutionContext,
    ) -> Option<Result<crate::common::AlignedRefit, ValidationError>> {
        let (rows, block) = (self.rows, self.block_length);
        Some(Ok(crate::common::AlignedRefit {
            rows,
            block_length: block,
            stand_in: Some(Arc::from("least-squares stand-in for the posterior mean")),
            estimate: Box::new(move |map: &[usize]| {
                // Every replicate is a run of circular blocks of the refitter's length,
                // not the n^(1/3) rule.
                for chunk in map.chunks(block) {
                    for pair in chunk.windows(2) {
                        assert_eq!(pair[1], (pair[0] + 1) % rows, "block shorter than {block}");
                    }
                }
                Some(map.iter().map(|&r| r as f64).sum::<f64>() / map.len() as f64)
            }),
        }))
    }
}

#[test]
fn composed_bootstrap_refute_uses_the_interval_block_length_and_reports_the_stand_in() {
    let series = lagged_xy_series(64, 0.2);
    let data = TabularData::new(series.storage().clone());
    let graph = lagged_xy_graph();
    let temporal_query =
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
    let id_res =
        TemporalBackdoorIdentifier::new().identify_temporal(&graph, &temporal_query).unwrap();
    let estimand = id_res.result.estimands.first().cloned().expect("identified");
    let ctx = ExecutionContext::for_tests(41);
    let temporal = TemporalRefitContext {
        indexer: &id_res.indexer,
        temporal_query: &temporal_query,
        split: None,
        kernel_policy: &ctx.kernel_policy,
        time_index: Some(series.time_index()),
        panel: None,
    };
    let ate_q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    // Far outside the replicate means (row-index averages near 31.5).
    let original = antecedent_estimate::EffectEstimate::new(
        1_000.0,
        f64::NAN,
        AssumptionSet::new(),
        antecedent_estimate::OverlapPolicy::ExplicitOverride,
    );
    let refit = FixedBlockRefit { rows: 60, block_length: 12 };
    assert!(antecedent_data::circular_block_length(2, 60) < 12);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &ate_q,
        &original,
        Some("temporal.sequential.gcomp"),
        Some(temporal),
    )
    .with_effect_refit(&refit);
    let refuter = BootstrapRefute { replicates: 40, ..BootstrapRefute::new() };
    let report = refuter.refute(&problem, &mut EstimationWorkspace::default(), &ctx).unwrap();
    assert_eq!(report.replicates, 40);
    assert!(!report.passed);
    let failure = report.failure_condition.expect("failing report");
    assert!(failure.contains("least-squares stand-in for the posterior mean"), "{failure}");
}
