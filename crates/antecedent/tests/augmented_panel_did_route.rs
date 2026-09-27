//! Retained augmented panel `DiD` known-truth, refusal, and artifact evidence.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PanelDidQuery, Study};
use antecedent_core::{Assumption, IntervalMethod, VariableId};
use antecedent_data::TabularData;

fn fixture(propensity: [f64; 6]) -> (TabularData, PanelDidQuery) {
    let pre = [10.0, 12.0, 13.0, 9.0, 15.0, 11.0];
    let post = [15.0, 17.0, 18.0, 11.0, 17.0, 13.0];
    let prediction = [2.0; 6];
    let data = TabularData::from_f64_columns([
        ("pre", pre.as_slice()), ("post", post.as_slice()),
        ("p", propensity.as_slice()), ("m0", prediction.as_slice()),
    ]).unwrap();
    let ids = (0..6).map(|i| Arc::<str>::from(format!("s{i}"))).collect::<Vec<_>>();
    let query = PanelDidQuery::augmented_panel(
        VariableId::from_raw(1), VariableId::from_raw(0),
        VariableId::from_raw(2), VariableId::from_raw(3),
        [true, true, true, false, false, false], ids.clone(), ids, true,
    );
    (data, query)
}

#[test]
fn augmented_panel_did_retains_known_truth_and_point_only_semantics() {
    let (data, query) = fixture([0.5; 6]);
    let ctx = ExecutionContext::for_tests(9);
    let study = Study::tabular(data.clone()).query(query.clone()).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.panel_did.as_ref().unwrap();
    assert!((fit.effect - 3.0).abs() < 1e-12);
    assert_eq!(fit.augmented, Some((0.5, 0.5, 3.0, true)));
    assert_eq!(fit.uncertainty.as_ref(), "point_only_no_standard_error");
    assert_eq!(result.interval.as_ref().unwrap().method, IntervalMethod::None);
    assert!(result.estimate.as_effect().unwrap().se_analytic.is_nan());
    for required in ["strict_propensity_overlap", "supplied_nuisance_predictions_valid_for_evaluation_rows"] {
        assert!(result.identification.required_assumptions.entries.iter().any(|record|
            matches!(&record.assumption, Assumption::Custom { id, .. } if id.as_ref() == required)));
    }
    assert_eq!(Study::tabular(data.clone()).query(query).build().unwrap().run(&ctx).unwrap().panel_did, result.panel_did);
    let bytes = prepared.encode_contracted_result(&result, "augmented-panel-did", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.panel_did.as_ref().unwrap().augmented, Some((0.5, 0.5, 3.0, true)));
    assert_eq!(body.panel_did.as_ref().unwrap().standard_error, None);
    assert!(body.interval_lower.is_none() && body.interval_upper.is_none());
    let mut fabricated = body.clone();
    fabricated.panel_did.as_mut().unwrap().augmented.as_mut().unwrap().2 = 99.0;
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated").is_err());
}

#[test]
fn augmented_panel_did_refuses_nonoverlap_and_repeated_subject() {
    let ctx = ExecutionContext::for_tests(9);
    let (data, query) = fixture([0.5, 0.5, 0.5, 0.0, 0.5, 0.5]);
    assert!(Study::tabular(data).query(query).build().unwrap().run(&ctx).is_err());
    let (data, mut query) = fixture([0.5; 6]);
    let mut subjects = query.subjects.to_vec();
    subjects[1] = subjects[0].clone();
    query.subjects = subjects.into();
    assert!(Study::tabular(data).query(query).build().unwrap().run(&ctx).is_err());
}
