//! End-to-end subject-owned sequential randomized regime value.

use antecedent::prelude::ExecutionContext;
use antecedent::{PrimaryEstimate, Study};
use antecedent_core::{CausalQuery, LongitudinalRegimeQuery, VariableId};
use antecedent_data::TabularData;
use std::sync::Arc;

fn query() -> LongitudinalRegimeQuery {
    LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(0),
        periods: 2,
        treatment_history: Arc::from([true, true, true, false, false, true, false, false]),
        regime_actions: Arc::from([true; 8]),
        treatment_probabilities: Arc::from([0.5; 8]),
        censoring_probabilities: Arc::from([1.0; 8]),
        outcome_observed: Arc::from([true; 4]),
        subject_ids: Arc::from(["a", "b", "c", "d"].map(Arc::<str>::from)),
        fold_ids: Arc::from([0; 4]),
        excluded_fold_predictions: false,
        probabilities_known_by_design: true,
        minimum_probability: 0.01,
    }
}

#[test]
fn known_truth_value_is_retained_and_point_only_artifact_round_trips() {
    let y = [4.0, 0.0, 0.0, 0.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let study = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(query()))
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(97);
    let direct = study.run(&ctx).unwrap();
    assert_eq!(direct.longitudinal_regime.as_ref().unwrap().value, 4.0);
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let regime = first.longitudinal_regime.as_ref().unwrap();
    assert_eq!(regime.value, 4.0);
    assert_eq!(regime.effective_sample_size, 1.0);
    assert_eq!(regime.matched_observed_fraction, 0.25);
    assert_eq!(&*regime.uncertainty, "point_only_no_interval");
    assert!(matches!(first.estimate, PrimaryEstimate::NotAnEffect));
    assert!(first.identification.required_assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. }
            if id.as_ref() == "sequential_censoring_exchangeability")
    }));
    let bytes = prepared.encode_contracted_result(&first, "longitudinal-regime", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value, 4.0);
    assert_eq!(body.estimate, None);
    assert_eq!(body.standard_error, None);
    assert_eq!(body.interval_lower, None);
    let mut fabricated = body.clone();
    fabricated.interval_lower = Some(3.0);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &fabricated,
            header.variable_names,
            "fabricated"
        )
        .is_err()
    );
    let refreshed = prepared.refresh(data, &ctx).unwrap();
    assert_eq!(first.longitudinal_regime, refreshed.longitudinal_regime);
}

#[test]
fn unsupported_ownership_and_sequential_positivity_refuse() {
    let mut q = query();
    q.probabilities_known_by_design = false;
    q.excluded_fold_predictions = true;
    q.fold_ids = Arc::from([0, 1, 0, 1]);
    let y = [4.0, 0.0, 0.0, 0.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let ctx = ExecutionContext::for_tests(98);
    let study =
        Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q)).build().unwrap();
    assert!(study.prepare(&ctx).is_err());
    let mut q = query();
    q.treatment_probabilities = Arc::from([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    assert!(q.validate().is_err());
}
