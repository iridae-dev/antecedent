//! `AverageEffect` routing: 0/1 treatments stay on binary machinery; any other
//! complete-case encoding must not enter logistic propensity / Riesz / IPW.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{EstimatorId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{LinearAdjustmentAte, OtherLevels};
use antecedent_graph::{Dag, DenseNodeId};

fn table(t: Vec<f64>, y_valid: Option<&[bool]>) -> TabularData {
    let n = t.len();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 0.01 * i as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
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
    let y_validity = match y_valid {
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
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
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
    TabularData::new(OwnedColumnarStorage::try_new(b.build().unwrap(), cols, None, None).unwrap())
}

fn confounded_dag() -> Dag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
}

fn run(data: TabularData, suite: RefuteSuite) -> antecedent::StudyResult {
    run_with(data, suite, LinearAdjustmentAte::new().with_bootstrap_replicates(0).into())
}

fn run_with(
    data: TabularData,
    suite: RefuteSuite,
    estimator: antecedent::EstimatorSpec,
) -> antecedent::StudyResult {
    Study::tabular(data)
        .graph(confounded_dag())
        .query(query())
        .estimator(estimator)
        .refute(suite)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(9))
        .unwrap()
}

fn overlap_names(result: &antecedent::StudyResult) -> Vec<String> {
    result.refutations.iter().map(|r| r.refuter.to_string()).collect()
}

#[test]
fn two_valued_float_stays_on_binary_overlap() {
    let t: Vec<f64> = (0..80).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let names = overlap_names(&run(table(t, None), RefuteSuite::Cheap));
    assert!(names.iter().any(|n| n == "overlap.assessment"), "{names:?}");
    assert!(names.iter().all(|n| n != "overlap.continuous_support"), "{names:?}");
}

#[test]
fn continuous_float_uses_residual_support_and_skips_riesz() {
    let t: Vec<f64> = (0..80).map(|i| f64::from(i) / 80.0).collect();
    let result = run(table(t, None), RefuteSuite::Full);
    let names = overlap_names(&result);
    assert!(names.iter().any(|n| n == "overlap.continuous_support"), "{names:?}");
    assert!(names.iter().all(|n| n != "overlap.assessment"), "{names:?}");
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "refute.validator.not_applicable"
            && d.message.contains("RieszSensitivity")),
        "continuous treatment must skip Riesz, diagnostics={:?}",
        result.diagnostics.iter().map(|d| (&*d.code, &*d.message)).collect::<Vec<_>>()
    );
}

// Levels beyond the declared 0 and 1 are refused unless the estimator declares the
// restriction to the two contrasted levels.
fn restricted() -> antecedent::EstimatorSpec {
    LinearAdjustmentAte::new()
        .with_bootstrap_replicates(0)
        .with_other_levels(OtherLevels::ExcludeUndeclared)
        .into()
}

#[test]
fn integer_dosage_and_multilevel_encoding_are_refused_or_restricted_to_the_binary_contrast() {
    for levels in [4, 3] {
        let values = || -> Vec<f64> { (0..80).map(|i| f64::from(i % levels)).collect() };
        let refused = Study::tabular(table(values(), None))
            .graph(confounded_dag())
            .query(query())
            .estimator(EstimatorId::LinearAdjustmentAte)
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(9))
            .err()
            .expect("extra discrete levels must not be read as a dose");
        assert_eq!(
            refused.reason_code(),
            Some("treatment_level_undeclared"),
            "{levels}: {refused}"
        );
        let names =
            overlap_names(&run_with(table(values(), None), RefuteSuite::Cheap, restricted()));
        // The overlap check classifies the encoding of the whole treatment column, which is
        // not the binary 0/1 one, so it stays on the continuous-support route.
        assert!(names.iter().any(|n| n == "overlap.continuous_support"), "{levels}: {names:?}");
        assert!(names.iter().all(|n| n != "overlap.assessment"), "{levels}: {names:?}");
    }
}

#[test]
fn degenerate_one_level_treatment_does_not_take_continuous_overlap() {
    let t = vec![1.0; 80];
    let result = Study::tabular(table(t, None))
        .graph(confounded_dag())
        .query(query())
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(9));
    // An all-ones column never visits the control level 0: the contrast is refused as
    // unpopulated (extrapolation), and it is not routed through continuous overlap.
    let err = result.err().expect("a one-level treatment has no control arm");
    assert_eq!(err.reason_code(), Some("arm_not_populated"), "{err}");
    let message = err.to_string();
    assert!(message.contains("treatment_levels.level_not_populated"), "{message}");
    assert!(!message.contains("continuous_support"), "{message}");
}

#[test]
fn complete_case_missingness_classifies_the_remaining_sample() {
    let mut t: Vec<f64> = (0..80).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    t[5] = 2.7;
    let mut keep = vec![true; 80];
    keep[5] = false;
    let names = overlap_names(&run(table(t, Some(&keep)), RefuteSuite::Cheap));
    assert!(
        names.iter().any(|n| n == "overlap.assessment"),
        "non-0/1 rows dropped by complete-case must leave the binary path, got {names:?}"
    );
}

#[test]
fn propensity_weighting_refuses_non_binary_treatment() {
    let t: Vec<f64> = (0..80).map(|i| f64::from(i) / 80.0).collect();
    let err = Study::tabular(table(t, None))
        .graph(confounded_dag())
        .query(query())
        .estimator(EstimatorId::PropensityWeighting)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(9))
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("treatment_level_undeclared"),
        "IPW must refuse continuous treatment rather than dichotomize, got {message}"
    );
}

#[test]
fn propensity_weighting_accepts_two_valued_unit_interval() {
    let t: Vec<f64> = (0..80).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let result = Study::tabular(table(t, None))
        .graph(confounded_dag())
        .query(query())
        .estimator(EstimatorId::PropensityWeighting)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(9))
        .unwrap();
    // y = 1 + 2 t + 0.01 i: the treated (odd i) rows have mean index 40 and the control rows 39,
    // so the weighted contrast is 2 + 0.01 * (40 - 39) = 2.01 (propensity ~ constant 1/2).
    let ate = result.estimate().as_effect().unwrap().ate;
    assert!((ate - 2.01).abs() < 1e-6, "{ate}");
    assert!(result.estimate().as_effect().unwrap().diagnostics.overlap_report.is_some());
}
