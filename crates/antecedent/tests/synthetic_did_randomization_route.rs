//! Exact assignment inference for retained synthetic difference in differences.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{Study, SyntheticControlQuery};
use antecedent_core::{IntervalMethod, VariableId};
use antecedent_data::TabularData;

#[test]
fn exact_synthetic_did_assignment_round_trips_and_rejects_tampering() {
    let mut units = Vec::new();
    let mut periods = Vec::new();
    let mut outcomes = Vec::new();
    for (unit, baseline) in [("a", 1.0), ("b", 2.0), ("c", 3.0), ("treated", 2.5)] {
        for period in 1..=4 {
            units.push(Arc::<str>::from(unit));
            periods.push(period);
            outcomes.push(baseline + period as f64 * 0.4
                + if unit == "treated" && period == 4 { 5.0 } else { 0.0 });
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0), units, periods, "treated", 4,
    ).difference_in_differences().with_uniform_unit_randomization();
    let ctx = ExecutionContext::for_tests(29);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.synthetic_did.as_ref().unwrap();
    assert!((fit.effect - 5.0).abs() < 1e-8);
    assert_eq!(fit.randomization_statistics.len(), 4);
    let observed = fit.randomization_statistics.iter()
        .find(|(unit, _)| unit.as_ref() == "treated").unwrap().1;
    let extreme = fit.randomization_statistics.iter()
        .filter(|(_, statistic)| *statistic >= observed).count();
    assert_eq!(fit.randomization_p_value, Some(extreme as f64 / 4.0));
    assert_eq!(result.interval.as_ref().unwrap().method, IntervalMethod::None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record|
        format!("{:?}", record.assumption).contains("uniform_single_treated_unit_assignment")));
    let bytes = prepared.encode_contracted_result(&result, "synthetic-did-exact", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.synthetic_did.as_ref().unwrap().randomization_p_value, fit.randomization_p_value);
    let mut fabricated = body.clone();
    fabricated.synthetic_did.as_mut().unwrap().randomization_p_value = Some(0.0);
    assert!(antecedent_io::encode_analysis_result_artifact(
        &fabricated, header.variable_names, "fabricated",
    ).is_err());
}

#[test]
fn exact_synthetic_did_assignment_refuses_more_than_32_candidate_units() {
    let mut units = Vec::new();
    let mut periods = Vec::new();
    let mut outcomes = Vec::new();
    for unit in 0..33 {
        for period in 1..=4 {
            units.push(Arc::<str>::from(format!("unit-{unit}")));
            periods.push(period);
            outcomes.push(unit as f64 * 0.2 + period as f64 * 0.3);
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0), units, periods, "unit-0", 4,
    ).difference_in_differences().with_uniform_unit_randomization();
    let error = Study::tabular(data).query(query).build().unwrap()
        .run(&ExecutionContext::for_tests(31)).unwrap_err();
    assert!(error.to_string().contains("at most 32 candidate units"));
}
