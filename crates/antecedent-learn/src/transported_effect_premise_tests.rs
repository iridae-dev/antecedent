//! Premise checks of the transported cohort mean that the module's own unit tests leave open:
//! the identity-scale requirement, cohort weight validation and the closed support box.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use super::*;
use crate::fitted_effect::FeatureRange;
use crate::learner::LearnerProvenance;
use crate::portable::PortablePredictor;

/// tau(z, w) = 1 + 2 z - w on the identity scale, fitted on z in [0, 1], w in [0, 2].
fn model(logistic: bool) -> FittedEffect {
    FittedEffect {
        version: 1,
        features: vec![3, 4],
        intercept: true,
        predictor: PortablePredictor {
            version: 1,
            columns: 3,
            provenance: LearnerProvenance {
                spec: "linear".into(),
                implementation: "test".into(),
                version: "1".into(),
            },
            model: PredictionMap::Linear { coefficients: vec![1.0, 2.0, -1.0], logistic },
        },
        support: Some(EffectSupport {
            version: 1,
            rows: 400,
            ranges: vec![FeatureRange { lo: 0.0, hi: 1.0 }, FeatureRange { lo: 0.0, hi: 2.0 }],
        }),
    }
}

fn cohort(z: Vec<f64>, w: Vec<f64>, weights: Vec<f64>) -> TargetCohort {
    TargetCohort {
        population_id: "clinic_b".into(),
        row_ids: (0..z.len()).map(|i| format!("r{i}")).collect(),
        features: vec!["w".into(), "z".into()],
        columns: vec![w, z],
        weights,
    }
}

fn run(
    effect: &FittedEffect,
    cohort: &TargetCohort,
) -> Result<TransportedCohortEffect, TransportRefusal> {
    let declaration =
        SelectionShiftDeclaration::new("clinic_a", "clinic_b", "S", &["z"], &["z", "w"]);
    transport_cohort_effect(
        effect,
        &["z".to_owned(), "w".to_owned()],
        "model-1",
        "clinic_a",
        cohort,
        Some(&declaration),
        &ExecutionContext::for_tests(5),
    )
}

#[test]
fn a_probability_scale_predictor_is_not_a_treatment_contrast() {
    let c = cohort(vec![0.5], vec![1.0], vec![1.0]);
    assert!(run(&model(false), &c).is_ok());
    let refusal = run(&model(true), &c).unwrap_err();
    assert_eq!(refusal.detail, "transported_effect.link_not_identity");
}

#[test]
fn cohort_weights_must_be_nonnegative_and_normalized() {
    let unnormalized = cohort(vec![0.5, 0.6], vec![1.0, 1.0], vec![0.5, 0.6]);
    assert_eq!(
        run(&model(false), &unnormalized).unwrap_err().detail,
        "transported_effect.weights_not_normalized"
    );
    let negative = cohort(vec![0.5, 0.6], vec![1.0, 1.0], vec![1.5, -0.5]);
    assert_eq!(
        run(&model(false), &negative).unwrap_err().detail,
        "transported_effect.weights_invalid"
    );
}

#[test]
fn the_mean_is_a_weighted_mean_of_row_contrasts_and_ignores_row_order() {
    // tau(z, w) = 1 + 2 z - w at (0.25, 0.5) = 1, at (0.75, 1.5) = 1, at (1.0, 0.0) = 3.
    let forward = cohort(vec![0.25, 0.75, 1.0], vec![0.5, 1.5, 0.0], vec![0.2, 0.3, 0.5]);
    let a = run(&model(false), &forward).unwrap();
    assert!((a.mean_contrast - (0.2 * 1.0 + 0.3 * 1.0 + 0.5 * 3.0)).abs() < 1e-12);
    let reversed = cohort(vec![1.0, 0.75, 0.25], vec![0.0, 1.5, 0.5], vec![0.5, 0.3, 0.2]);
    let b = run(&model(false), &reversed).unwrap();
    assert!((a.mean_contrast - b.mean_contrast).abs() < 1e-12);
    assert_ne!(a.identity(), b.identity(), "the identity binds the row order");
}

#[test]
fn the_fitted_support_box_is_closed_and_a_zero_weight_row_outside_it_still_refuses() {
    // The box endpoints are inside the support.
    let edge = cohort(vec![0.0, 1.0], vec![0.0, 2.0], vec![0.5, 0.5]);
    assert!(run(&model(false), &edge).unwrap().overlap.fully_covered());
    // Rows are checked regardless of weight: the support check is not mass-weighted.
    let outside = cohort(vec![0.5, 1.5], vec![1.0, 1.0], vec![1.0, 0.0]);
    assert_eq!(
        run(&model(false), &outside).unwrap_err().detail,
        "transported_effect.out_of_support"
    );
}

#[test]
fn a_cohort_column_with_another_name_is_a_schema_mismatch() {
    // Features are matched to the model by name; a differently named column is refused.
    let mut c = cohort(vec![0.5], vec![1.0], vec![1.0]);
    c.features[1] = "zz".into();
    assert_eq!(
        run(&model(false), &c).unwrap_err().detail,
        "transported_effect.feature_schema_mismatch"
    );
}
