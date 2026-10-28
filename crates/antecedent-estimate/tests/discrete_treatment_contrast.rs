//! A regression contrast of two levels of a discrete treatment with further levels.
//!
//! The outcome is `y = f(t) + z / 2` with `f = (0, 1, 4)` at the treatment levels `0, 1, 2` and
//! the binary covariate `z` balanced within every level, so the contrast of level 1 against
//! level 0 is exactly 1 and of level 2 against level 0 exactly 4. The treatment enters a
//! regression as one numeric column, so on all rows it would be read as a dose: the fitted slope
//! is 2, and "1 against 0" would report 2 and "2 against 0" would report 4 by luck of the
//! spacing. Both regression estimators therefore refuse such a column with
//! `treatment_level_undeclared`, naming the observed levels and the next attempt, unless the
//! restriction to the rows at the two levels is declared; the declared restriction recovers the
//! contrast exactly and reports the rows it excluded. A two-valued column is untouched, and a
//! column with many distinct values stays a continuous dose.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "test fixtures index small literals; level codes are exact labels"
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
    EffectEstimate, EstimationError, EstimationWorkspace, GlmAdjustmentAte, GlmAdjustmentWorkspace,
    LinearAdjustmentAte, OtherLevels,
};
use antecedent_stats::GlmFamily;
use antecedent_expr::{ExprId, IdentifiedEstimand};

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);
const Z: VariableId = VariableId::from_raw(2);

const EFFECT_OF_LEVEL: [f64; 3] = [0.0, 1.0, 4.0];

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

/// 120 rows, 40 at each of the levels 0, 1, 2, `z` balanced within every level.
fn three_levels() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 120;
    let t: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| ((i / 3) % 2) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| EFFECT_OF_LEVEL[i % 3] + 0.5 * z[i]).collect();
    (t, y, z)
}

fn coded(error: EstimationError, what: &str) -> (&'static str, String) {
    match error {
        EstimationError::Refused { code, message } => (code, message.into_string()),
        other => panic!("{what}: expected a coded refusal, got {other:?}"),
    }
}

fn linear(
    estimator: &LinearAdjustmentAte,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
) -> Result<EffectEstimate, EstimationError> {
    let problem = estimator.prepare(data, estimand, query)?;
    estimator.fit(
        &problem,
        &mut EstimationWorkspace::default(),
        &ExecutionContext::for_tests(3),
        AssumptionSet::new(),
    )
}

fn glm(
    estimator: &GlmAdjustmentAte,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
) -> Result<EffectEstimate, EstimationError> {
    let problem = estimator.prepare(data, estimand, query)?;
    estimator.fit(
        &problem,
        &mut GlmAdjustmentWorkspace::default(),
        &ExecutionContext::for_tests(3),
        AssumptionSet::new(),
    )
}

fn gaussian_glm() -> GlmAdjustmentAte {
    GlmAdjustmentAte::new().with_family(GlmFamily::GaussianIdentity).with_bootstrap_replicates(0)
}

fn linear_estimator() -> LinearAdjustmentAte {
    LinearAdjustmentAte::new().with_bootstrap_replicates(0)
}

#[test]
fn a_three_level_treatment_is_refused_naming_its_levels_by_the_linear_and_glm_adjustments() {
    let (t, y, z) = three_levels();
    let (data, estimand) = dataset(&t, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 1.0);
    let listing = "observed treatment levels: 0 (40 rows), 1 (40 rows), 2 (40 rows)";
    let refusals = [
        ("linear", linear(&linear_estimator(), &data, &estimand, &query).unwrap_err()),
        ("gaussian glm", glm(&gaussian_glm(), &data, &estimand, &query).unwrap_err()),
        (
            "logit glm",
            glm(&GlmAdjustmentAte::new().with_bootstrap_replicates(0), &data, &estimand, &query)
                .unwrap_err(),
        ),
    ];
    for (name, error) in refusals {
        let (code, message) = coded(error, name);
        assert_eq!(code, "treatment_level_undeclared", "{name}");
        assert!(message.starts_with("treatment_levels.undeclared_level:"), "{name}: {message}");
        assert!(message.contains(listing), "{name}: {message}");
        assert!(message.contains("40 complete-case rows"), "{name}: {message}");
        assert!(message.contains("Next attempt"), "{name}: {message}");
    }
}

#[test]
fn the_declared_restriction_recovers_the_contrast_of_the_two_levels_and_counts_what_it_excluded() {
    let (t, y, z) = three_levels();
    let (data, estimand) = dataset(&t, &y, &z);
    for (active, control, truth, excluded) in
        [(1.0, 0.0, 1.0, 40_u64), (2.0, 0.0, 4.0, 40), (2.0, 1.0, 3.0, 40)]
    {
        let query = AverageEffectQuery::with_levels(T, Y, control, active);
        let estimators = [
            (
                "linear",
                linear(
                    &linear_estimator().with_other_levels(OtherLevels::ExcludeUndeclared),
                    &data,
                    &estimand,
                    &query,
                )
                .unwrap(),
            ),
            (
                "gaussian glm",
                glm(
                    &gaussian_glm().with_other_levels(OtherLevels::ExcludeUndeclared),
                    &data,
                    &estimand,
                    &query,
                )
                .unwrap(),
            ),
        ];
        for (name, effect) in estimators {
            assert!(
                (effect.ate - truth).abs() < 1e-9,
                "{name} {active} vs {control}: {} vs {truth}",
                effect.ate
            );
            let levels = effect.diagnostics.treatment_levels.expect("the exclusion is reported");
            assert_eq!((levels.active, levels.control), (active, control), "{name}");
            assert_eq!((levels.rows_at_active, levels.rows_at_control), (40, 40), "{name}");
            assert_eq!(levels.excluded_rows, excluded, "{name}");
            assert_eq!(
                levels.restricted_population(),
                Some("rows at the two declared levels"),
                "{name}"
            );
            if name == "linear" {
                assert_eq!(effect.provenance.n_obs, Some(80), "{name}");
            }
        }
    }
}

#[test]
fn a_two_valued_column_and_a_continuous_dose_are_not_restricted() {
    // Two-valued: no further level, nothing to refuse or report.
    let (t, y, z) = three_levels();
    let keep: Vec<usize> = (0..t.len()).filter(|&i| t[i] != 2.0).collect();
    let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let (data, estimand) = dataset(&pick(&t), &pick(&y), &pick(&z));
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 1.0);
    for effect in [
        linear(&linear_estimator(), &data, &estimand, &query).unwrap(),
        glm(&gaussian_glm(), &data, &estimand, &query).unwrap(),
    ] {
        assert!((effect.ate - 1.0).abs() < 1e-9, "{}", effect.ate);
        assert!(effect.diagnostics.treatment_levels.is_none());
    }
    // A continuous dose with more than ten distinct values keeps its linear-dose reading: the
    // contrast is the slope times the gap between the levels.
    let n = 120;
    let dose: Vec<f64> = (0..n).map(|i| (i % 12) as f64 / 11.0).collect();
    let z: Vec<f64> = (0..n).map(|i| ((i / 12) % 2) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| 3.0 * dose[i] + 0.5 * z[i]).collect();
    let (data, estimand) = dataset(&dose, &y, &z);
    let query = AverageEffectQuery::with_levels(T, Y, 0.0, 1.0);
    let effect = linear(&linear_estimator(), &data, &estimand, &query).unwrap();
    assert!((effect.ate - 3.0).abs() < 1e-9, "{}", effect.ate);
    assert!(effect.diagnostics.treatment_levels.is_none());
}
