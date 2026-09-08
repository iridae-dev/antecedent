//! `AverageEffect` routing: 0/1 treatments stay on binary machinery; any other
//! complete-case encoding must not enter logistic propensity / Riesz / IPW.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::doc_markdown
)]

use std::sync::Arc;

use antecedent::{EstimatorId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
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
    Study::tabular(data)
        .graph(confounded_dag())
        .query(query())
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(suite)
        .bootstrap_replicates(0)
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
    let t: Vec<f64> = (0..80).map(|i| i as f64 / 80.0).collect();
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

#[test]
fn integer_dosage_and_multilevel_encoding_are_not_binary() {
    let dose: Vec<f64> = (0..80).map(|i| (i % 4) as f64).collect();
    let names = overlap_names(&run(table(dose, None), RefuteSuite::Cheap));
    assert!(names.iter().any(|n| n == "overlap.continuous_support"), "dosage {names:?}");
    let cat: Vec<f64> = (0..80).map(|i| (i % 3) as f64).collect();
    let names = overlap_names(&run(table(cat, None), RefuteSuite::Cheap));
    assert!(names.iter().any(|n| n == "overlap.continuous_support"), "categorical {names:?}");
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
    match result {
        Ok(result) => {
            let names = overlap_names(&result);
            assert!(
                names.iter().all(|n| n != "overlap.continuous_support"),
                "all-ones 0/1 coding must not be classified as continuous, got {names:?}"
            );
        }
        Err(err) => {
            let message = err.to_string();
            assert!(
                !message.contains("continuous_support"),
                "one-level treatment must not be routed through continuous overlap, got {message}"
            );
        }
    }
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
    let t: Vec<f64> = (0..80).map(|i| i as f64 / 80.0).collect();
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
        message.contains("binary treatment"),
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
    assert!(result.estimate.ate.is_finite());
    assert!(result.estimate.overlap_report.is_some());
}
