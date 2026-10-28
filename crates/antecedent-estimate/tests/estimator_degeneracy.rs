//! Degenerate designs of the observational contrast estimators (linear adjustment, Gaussian
//! GLM adjustment, propensity matching, AIPW).
//!
//! A treatment column with one level has no contrast and no comparison group: every estimator
//! refuses it rather than publish a number. A covariate that is constant across rows is
//! collinear with the intercept: every estimator refuses it as rank deficient instead of
//! silently dropping the column and publishing the unadjusted contrast. A constant outcome has
//! no effect: the contrast and its standard error are exactly zero. The control runs the same
//! rows with the informative covariate against the exact arm-mean contrast.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "the degenerate fixtures are exact literals"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{
    AipwAte, AipwWorkspace, AnalyticSeKind, EffectEstimate, EstimationWorkspace,
    GlmAdjustmentAte, GlmAdjustmentWorkspace, LinearAdjustmentAte, OverlapPolicy,
    PropensityEstimationWorkspace, PropensityMatching,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_stats::GlmFamily;

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(29)
}

fn dataset(t: &[f64], y: &[f64], z: &[f64]) -> (TabularData, IdentifiedEstimand) {
    let n = t.len();
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
    let columns = [(T, t), (Y, y), (Z, z)]
        .into_iter()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap();
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([Z]), ExprId::from_raw(0));
    (TabularData::new(storage), estimand)
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(T, Y)
}

fn no_clip() -> OverlapPolicy {
    OverlapPolicy::RequireDiagnostics { clip: None, trim: None }
}

/// Deterministic rows: `z` takes four values, treatment alternates in a pattern that is
/// balanced within every `z`, and the outcome is `1 + 2 t + 0.5 z` exactly.
fn rows(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let z: Vec<f64> = (0..n).map(|i| (i % 4) as f64 - 1.5).collect();
    let t: Vec<f64> = (0..n).map(|i| f64::from((i / 4) % 2 == 0)).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 0.5 * z[i]).collect();
    (t, y, z)
}

type Run = Box<dyn Fn(&[f64], &[f64], &[f64]) -> Result<EffectEstimate, String>>;

fn runners() -> Vec<(&'static str, Run)> {
    let linear: Run = Box::new(|t, y, z| {
        let (data, estimand) = dataset(t, y, z);
        let estimator = LinearAdjustmentAte::new()
            .with_bootstrap_replicates(0)
            .with_se_kind(AnalyticSeKind::Hc1);
        let prepared = estimator.prepare(&data, &estimand, &query()).map_err(|e| e.to_string())?;
        estimator
            .fit(&prepared, &mut EstimationWorkspace::default(), &ctx(), AssumptionSet::new())
            .map_err(|e| e.to_string())
    });
    let glm: Run = Box::new(|t, y, z| {
        let (data, estimand) = dataset(t, y, z);
        let estimator = GlmAdjustmentAte::new()
            .with_bootstrap_replicates(0)
            .with_family(GlmFamily::GaussianIdentity)
            .with_se_kind(AnalyticSeKind::Hc1);
        let prepared = estimator.prepare(&data, &estimand, &query()).map_err(|e| e.to_string())?;
        estimator
            .fit(&prepared, &mut GlmAdjustmentWorkspace::default(), &ctx(), AssumptionSet::new())
            .map_err(|e| e.to_string())
    });
    let matching: Run = Box::new(|t, y, z| {
        let (data, estimand) = dataset(t, y, z);
        let estimator = PropensityMatching::new().with_overlap(no_clip());
        let prepared = estimator.prepare(&data, &estimand, &query()).map_err(|e| e.to_string())?;
        estimator
            .fit(
                &prepared,
                &mut PropensityEstimationWorkspace::default(),
                &ctx(),
                AssumptionSet::new(),
            )
            .map_err(|e| e.to_string())
    });
    let aipw: Run = Box::new(|t, y, z| {
        let (data, estimand) = dataset(t, y, z);
        let estimator = AipwAte::new().with_bootstrap_replicates(0);
        let prepared = estimator.prepare(&data, &estimand, &query()).map_err(|e| e.to_string())?;
        estimator
            .fit(&prepared, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new())
            .map_err(|e| e.to_string())
    });
    vec![
        ("linear adjustment", linear),
        ("gaussian glm adjustment", glm),
        ("propensity matching", matching),
        ("aipw", aipw),
    ]
}

#[test]
fn a_treatment_column_with_one_level_is_refused_by_every_contrast_estimator() {
    let (_, y, z) = rows(96);
    let all_treated = vec![1.0; y.len()];
    let none_treated = vec![0.0; y.len()];
    for (name, run) in runners() {
        for (label, t) in [("every row treated", &all_treated), ("no row treated", &none_treated)] {
            let outcome = run(t, &y, &z);
            assert!(
                outcome.is_err(),
                "{name} published {:?} for a treatment column with {label}",
                outcome.map(|fit| fit.ate)
            );
        }
    }
}

#[test]
fn a_constant_outcome_has_exactly_zero_effect_in_every_estimator() {
    let (t, _, z) = rows(96);
    let y = vec![3.25; t.len()];
    for (name, run) in runners() {
        let fit = run(&t, &y, &z).unwrap_or_else(|e| panic!("{name} refused a constant outcome: {e}"));
        assert!(fit.ate.abs() < 1e-9, "{name}: constant outcome gave effect {}", fit.ate);
        let se = fit.se_analytic();
        assert!(se.is_finite() && (0.0..1e-9).contains(&se), "{name}: standard error {se}");
    }
}

#[test]
fn a_constant_covariate_is_refused_as_rank_deficient_not_repaired() {
    // The adjustment column is constant, hence collinear with the intercept (or, for a
    // propensity model, with the intercept-only design): every estimator refuses, none drops
    // the column silently and publishes the unadjusted contrast.
    let (t, y, _) = rows(96);
    let constant = vec![0.7; t.len()];
    for (name, run) in runners() {
        let outcome = run(&t, &y, &constant);
        let message = outcome.as_ref().err().unwrap_or_else(|| {
            panic!("{name} published {:?} for a constant adjustment covariate", outcome.as_ref().map(|f| f.ate))
        });
        assert!(message.contains("rank deficient"), "{name}: {message}");
    }
}

#[test]
fn a_balanced_design_has_the_exact_arm_mean_contrast_as_its_effect() {
    // The control for the two tests above: on the same rows with the informative covariate
    // every estimator that fits publishes an effect near the exact arm-mean contrast 2.
    let (t, y, z) = rows(96);
    let treated: Vec<f64> = (0..t.len()).filter(|&i| t[i] == 1.0).map(|i| y[i]).collect();
    let control: Vec<f64> = (0..t.len()).filter(|&i| t[i] == 0.0).map(|i| y[i]).collect();
    let difference = treated.iter().sum::<f64>() / treated.len() as f64
        - control.iter().sum::<f64>() / control.len() as f64;
    assert!((difference - 2.0).abs() < 1e-12);
    for (name, run) in runners() {
        let fit = run(&t, &y, &z).unwrap_or_else(|e| panic!("{name} refused the control: {e}"));
        assert!((fit.ate - 2.0).abs() < 1e-6, "{name}: {}", fit.ate);
    }
}
