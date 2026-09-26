//! End-to-end retained randomized survival and competing-risk execution.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PrimaryEstimate, Study};
use antecedent_core::{
    CausalQuery, ObservationAssumption, SurvivalFunctional, SurvivalQuery, VariableId,
};
use antecedent_data::TabularData;

fn query(functional: SurvivalFunctional) -> SurvivalQuery {
    SurvivalQuery {
        duration: VariableId::from_raw(0),
        event: VariableId::from_raw(1),
        treatment: VariableId::from_raw(2),
        tau: 2.0,
        delayed_entry: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional,
    }
}

#[test]
fn randomized_survival_rmst_is_retained_and_round_trips() {
    let duration = [1.0, 2.0, 2.0, 2.0];
    let event = [1.0, 0.0, 0.0, 0.0];
    let treatment = [0.0, 0.0, 1.0, 1.0];
    let data = TabularData::from_f64_columns([
        ("duration", &duration[..]),
        ("event", &event[..]),
        ("treatment", &treatment[..]),
    ])
    .unwrap();
    let study = Study::tabular(data.clone())
        .query(CausalQuery::Survival(query(SurvivalFunctional::SurvivalAndRmst)))
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(86);
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let curve = first.survival.as_ref().unwrap();
    assert_eq!(&*curve.times, &[0.0, 1.0, 2.0]);
    assert_eq!(&*curve.control, &[1.0, 0.5, 0.5]);
    assert_eq!(&*curve.treated, &[1.0, 1.0, 1.0]);
    assert_eq!(curve.rmst_control, Some(1.5));
    assert_eq!(curve.rmst_treated, Some(2.0));
    assert_eq!(&*curve.uncertainty, "point_only_no_interval");
    assert!(matches!(first.estimate, PrimaryEstimate::NotAnEffect));
    let bytes = prepared.encode_contracted_result(&first, "survival", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.survival.as_ref().unwrap();
    assert_eq!(wire.rmst_control, Some(1.5));
    assert_eq!(wire.uncertainty, "point_only_no_interval");
    assert!(body.estimate.is_none());
    assert!(body.standard_error.is_none());
    assert!(body.interval_lower.is_none());
    assert!(body.interval_upper.is_none());
    let mut fabricated = body.clone();
    fabricated.standard_error = Some(0.1);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &fabricated,
            header.variable_names,
            "fabricated"
        )
        .is_err()
    );
    assert!(first.identification.required_assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == "independent_censoring_and_entry")
    }));
    let refreshed = prepared.refresh(data, &ctx).unwrap();
    assert_eq!(first.survival, refreshed.survival);
}

#[test]
fn competing_risk_incidence_and_unsupported_observation_contract() {
    let duration = [1.0, 2.0, 1.0, 2.0];
    let event = [1.0, 2.0, 1.0, 1.0];
    let treatment = [0.0, 0.0, 1.0, 1.0];
    let data = TabularData::from_f64_columns([
        ("duration", &duration[..]),
        ("event", &event[..]),
        ("treatment", &treatment[..]),
    ])
    .unwrap();
    let q = query(SurvivalFunctional::CumulativeIncidence { target_cause: 1 });
    let study =
        Study::tabular(data.clone()).query(CausalQuery::Survival(q.clone())).build().unwrap();
    let ctx = ExecutionContext::for_tests(87);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival.as_ref().unwrap();
    assert_eq!(&*curve.control, &[0.0, 0.5, 0.5]);
    assert_eq!(&*curve.treated, &[0.0, 0.5, 1.0]);
    assert_eq!(curve.rmst_control, None);
    assert_eq!(curve.target_cause, Some(1));

    let mut invalid = q;
    invalid.observation_assumption =
        ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(0)]));
    let unsupported = Study::tabular(data).query(CausalQuery::Survival(invalid)).build().unwrap();
    assert!(unsupported.prepare(&ctx).is_err());
}
