//! Retained synthetic-control study evidence.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{Study, SyntheticControlQuery};
use antecedent_core::{IntervalMethod, VariableId};
use antecedent_data::TabularData;

fn fixture() -> (TabularData, SyntheticControlQuery) {
    let donors = [
        ("a", [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ("b", [0.0, 1.0, 0.0, 1.0, 2.0, 2.0]),
        ("c", [1.0, 0.0, 0.0, 1.0, 0.0, 2.0]),
    ];
    let mut y = Vec::new();
    let mut units = Vec::new();
    let mut periods = Vec::new();
    for (unit, trajectory) in donors {
        for (index, value) in trajectory.into_iter().enumerate() {
            units.push(Arc::<str>::from(unit));
            periods.push(index as i64 + 1);
            y.push(value);
        }
    }
    for index in 0..6 {
        units.push(Arc::<str>::from("treated"));
        periods.push(index as i64 + 1);
        let counterfactual = 0.2 * donors[0].1[index] + 0.3 * donors[1].1[index]
            + 0.5 * donors[2].1[index];
        y.push(counterfactual + if index >= 4 { 5.0 } else { 0.0 });
    }
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0), units, periods, Arc::<str>::from("treated"), 5,
    );
    (data, query)
}

#[test]
fn synthetic_control_retained_route_recovers_truth_and_preserves_point_only_claim() {
    let (data, query) = fixture();
    let ctx = ExecutionContext::for_tests(11);
    let result = Study::tabular(data.clone()).query(query.clone()).build().unwrap()
        .run(&ctx).unwrap();
    let fit = result.synthetic_control.as_ref().unwrap();
    assert!((fit.effect - 5.0).abs() < 1e-5);
    assert!(fit.pre_treatment_rmse < 1e-5);
    assert_eq!(fit.donor_weights.len(), 3);
    assert_eq!(fit.placebo_effects.len(), 3);
    assert_eq!(fit.n_pre_periods, 4);
    assert_eq!(fit.n_post_periods, 2);
    assert_eq!(fit.uncertainty.as_ref(), "point_only_with_unlicensed_placebo_rank");
    assert_eq!(result.interval.as_ref().unwrap().method, IntervalMethod::None);
    assert!(result.estimate.as_effect().unwrap().se_analytic.is_nan());
    assert!(result.identification.required_assumptions.entries.iter().any(|record|
        format!("{:?}", record.assumption).contains("convex_donor_combination_is_a_valid_counterfactual")));
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap()
        .prepare(&ctx).unwrap();
    let retained = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(retained.synthetic_control, result.synthetic_control);
    let bytes = prepared.encode_contracted_result(&retained, "synthetic-control", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.synthetic_control.as_ref().unwrap();
    assert!((wire.effect - 5.0).abs() < 1e-5);
    assert_eq!(wire.donor_weights.len(), 3);
    assert_eq!(wire.placebo_effects.len(), 3);
    assert!(body.standard_error.is_none());
    assert!(body.interval_lower.is_none());
    assert!(body.interval_upper.is_none());
    let mut fabricated = body.clone();
    fabricated.standard_error = Some(0.1);
    assert!(antecedent_io::encode_analysis_result_artifact(
        &fabricated, header.variable_names, "fabricated"
    ).is_err());
}

#[test]
fn synthetic_control_refuses_insufficient_donors_and_unbalanced_panel() {
    let (data, query) = fixture();
    let ctx = ExecutionContext::for_tests(1);
    let few_donors = SyntheticControlQuery::new(
        query.outcome, query.units[..18].to_vec(), query.periods[..18].to_vec(), "a", 5,
    );
    let few_data = TabularData::from_f64_columns([("outcome", [0.0; 18].as_slice())]).unwrap();
    assert!(Study::tabular(few_data).query(few_donors).build().unwrap().run(&ctx).is_err());
    let mut missing = query.clone();
    let mut periods = query.periods.to_vec();
    *periods.last_mut().unwrap() = 5;
    missing.periods = periods.into();
    assert!(Study::tabular(data).query(missing).build().unwrap().run(&ctx).is_err());
}
