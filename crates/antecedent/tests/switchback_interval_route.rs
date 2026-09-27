//! Retained and artifact evidence for calibrated switchback inference.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{RandomizedEffectQuery, Study};
use antecedent_core::{RandomizationDesign, VariableId};
use antecedent_data::TabularData;

fn study(sequences: usize, periods: usize, probability: f64) -> (TabularData, RandomizedEffectQuery) {
    let n = sequences * periods;
    let assigned = (0..n).map(|i| (i.wrapping_mul(137) + (i / periods) * 31) % 11 < 5)
        .collect::<Vec<_>>();
    let outcomes = (0..n).map(|i| {
        let sequence = i / periods;
        let period = i % periods;
        1.0 + 0.5 * (sequence as f64 * 0.7).sin()
            + 0.3 * (period as f64 * 0.4).cos()
            + if assigned[i] { 1.4 } else { 0.0 }
    }).collect::<Vec<_>>();
    let sequence_ids = (0..n).map(|i| Arc::<str>::from(format!("s{}", i / periods)))
        .collect::<Vec<_>>();
    let period_ids = (0..n).map(|i| Arc::<str>::from(format!("p{}", i % periods)))
        .collect::<Vec<_>>();
    let row_ids = (0..n).map(|i| Arc::<str>::from(format!("row-{i}")))
        .collect::<Vec<_>>();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Switchback { periods: period_ids.into() },
        VariableId::from_raw(0), assigned, vec![probability; n],
        sequence_ids, row_ids, ("off", "on"),
    );
    (TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap(), query)
}

#[test]
fn supported_switchback_interval_round_trips_and_rejects_forgery() {
    let (data, query) = study(30, 24, 0.2);
    let ctx = ExecutionContext::for_tests(0x5A17);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap()
        .prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.randomized_effect.as_ref().unwrap();
    let [lower, upper] = fit.interval_95.expect("supported switchback interval");
    assert!(lower < fit.effect && fit.effect < upper);
    assert!(fit.standard_error.unwrap() > 0.0);
    assert_eq!(fit.uncertainty.as_ref(), "switchback_independent_sequence_student_interval");
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::AnalyticSe);
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "switchback_no_carryover"
    )));
    let bytes = prepared.encode_contracted_result(&result, "switchback", &ctx).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(artifact.randomized_effect.as_ref().unwrap().interval_95, Some([lower, upper]));
    assert_eq!(artifact.randomized_effect.as_ref().unwrap().graphless_support_status.as_deref(), Some("licensed"));
    let mut forged = artifact.clone();
    forged.randomized_effect.as_mut().unwrap().interval_95.as_mut().unwrap()[1] += 0.5;
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names.clone(), "forged-switchback-interval").is_err());
    let mut forged = artifact;
    forged.randomized_effect.as_mut().unwrap().graphless_support_status = Some("refused".into());
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names, "forged-switchback-status").is_err());
}

#[test]
fn sparse_or_weak_switchback_remains_point_only() {
    for (sequences, probability) in [(4, 0.5), (30, 0.19)] {
        let (data, query) = study(sequences, 24, probability);
        let ctx = ExecutionContext::for_tests(0x5A17);
        let result = Study::tabular(data.clone()).query(query).build().unwrap()
            .prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
        let fit = result.randomized_effect.as_ref().unwrap();
        assert!(fit.effect.is_finite());
        assert_eq!(fit.interval_95, None);
        assert_eq!(result.support_status, None);
        assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    }
}

#[test]
fn unequal_sequence_period_counts_refuse_interval_and_license() {
    let (data, mut query) = study(30, 24, 0.5);
    let mut sequence_ids = query.assignment_units.to_vec();
    sequence_ids[0] = Arc::from("s1");
    query.assignment_units = sequence_ids.into();
    let RandomizationDesign::Switchback { periods } = &query.design else { unreachable!() };
    let mut period_ids = periods.to_vec();
    period_ids[0] = Arc::from("p24");
    query.design = RandomizationDesign::Switchback { periods: period_ids.into() };
    let ctx = ExecutionContext::for_tests(0x5A17);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap()
        .prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.randomized_effect.as_ref().unwrap().interval_95, None);
    assert_eq!(result.support_status, None);
    let bytes = prepared.encode_contracted_result(&result, "unbalanced-switchback", &ctx).unwrap();
    let (_, header, mut artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    artifact.randomized_effect.as_mut().unwrap().graphless_support_status = Some("licensed".into());
    assert!(antecedent_io::encode_analysis_result_artifact(
        &artifact, header.variable_names, "forged-unbalanced-switchback-license").is_err());
}
