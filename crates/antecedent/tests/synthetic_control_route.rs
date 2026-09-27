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

#[test]
fn retained_synthetic_did_recovers_additive_truth_and_round_trips_weights() {
    let mut units = Vec::new();
    let mut periods = Vec::new();
    let mut y = Vec::new();
    for (unit, baseline) in [("treated", 10.0), ("d0", 2.0), ("d1", 10.0), ("d2", 18.0)] {
        for (period, common) in [(1, 1.0), (2, 3.0), (3, -2.0), (4, 5.0)] {
            units.push(Arc::<str>::from(unit));
            periods.push(period);
            y.push(baseline + common + if unit == "treated" && period == 4 { 7.0 } else { 0.0 });
        }
    }
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0), units, periods, "treated", 4,
    ).difference_in_differences();
    let ctx = ExecutionContext::for_tests(13);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.synthetic_did.as_ref().unwrap();
    assert!((fit.effect - 7.0).abs() < 1e-8);
    assert_eq!((fit.n_donors, fit.n_pre_periods, fit.n_post_periods), (3, 3, 1));
    assert_eq!(fit.time_weights.len(), 3);
    assert_eq!(fit.uncertainty.as_ref(), "point_only_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, IntervalMethod::None);
    let bytes = prepared.encode_contracted_result(&result, "synthetic-did", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.synthetic_did.as_ref().unwrap().effect, 7.0);
    assert!(body.standard_error.is_none());
    let mut fabricated = body.clone();
    fabricated.interval_lower = Some(6.0);
    assert!(antecedent_io::encode_analysis_result_artifact(
        &fabricated, header.variable_names, "fabricated"
    ).is_err());
}

#[test]
fn uniform_unit_randomization_enumerates_sharp_null_and_seals_p_value() {
    let (data, query) = fixture();
    let query = query.with_uniform_unit_randomization();
    let ctx = ExecutionContext::for_tests(17);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.synthetic_control.as_ref().unwrap();
    assert_eq!(fit.randomization_statistics.len(), 4);
    let observed = fit.randomization_statistics.iter().find(|(unit, _)| unit.as_ref() == "treated").unwrap().1;
    let extreme = fit.randomization_statistics.iter().filter(|(_, statistic)| *statistic >= observed).count();
    assert_eq!(fit.randomization_p_value, Some(extreme as f64 / 4.0));
    assert_eq!(fit.uncertainty.as_ref(), "point_only_with_exact_unit_randomization_p_value_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, IntervalMethod::None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record|
        format!("{:?}", record.assumption).contains("uniform_single_treated_unit_assignment")));
    let bytes = prepared.encode_contracted_result(&result, "exact-placebo", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.synthetic_control.as_ref().unwrap().randomization_p_value, fit.randomization_p_value);
    let mut fabricated = body.clone();
    fabricated.synthetic_control.as_mut().unwrap().randomization_p_value = Some(0.0);
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated").is_err());
    let (_, invalid) = fixture();
    assert!(invalid.difference_in_differences().with_uniform_unit_randomization().validate().is_err());
}

#[test]
fn exact_unit_randomization_is_superuniform_over_all_sharp_null_assignments() {
    let mut units = Vec::new();
    let mut periods = Vec::new();
    let mut outcomes = Vec::new();
    for unit in 0..8 {
        for period in 1..=4 {
            units.push(Arc::<str>::from(format!("unit-{unit}")));
            periods.push(period);
            // Under the sharp null, this complete outcome panel is fixed
            // regardless of which unit receives the uniform assignment.
            outcomes.push(unit as f64 * 0.31 + period as f64 * 0.17
                + ((unit * 7 + period as usize * 3) % 11) as f64 * 0.13);
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(71);
    let mut p_values = Vec::new();
    for assigned in 0..8 {
        let query = SyntheticControlQuery::new(
            VariableId::from_raw(0), units.clone(), periods.clone(),
            format!("unit-{assigned}"), 4,
        ).with_uniform_unit_randomization();
        let result = Study::tabular(data.clone()).query(query).build().unwrap().run(&ctx).unwrap();
        let fit = result.synthetic_control.unwrap();
        p_values.push(fit.randomization_p_value.unwrap());
        assert_eq!(fit.randomization_statistics.len(), 8);
    }
    for alpha in [0.125, 0.25, 0.5] {
        let rejects = p_values.iter().filter(|p| **p <= alpha).count();
        assert!(rejects as f64 / 8.0 <= alpha, "sharp-null rejection rate exceeds {alpha}");
    }
}

#[test]
fn augmented_synthetic_control_recovers_outside_hull_truth_and_seals_artifact() {
    let mut units = Vec::new();
    let mut periods = Vec::new();
    let mut y = Vec::new();
    for (unit, position) in [("a", 0.0), ("b", 1.0), ("c", 2.0), ("treated", 3.0)] {
        for period in 1..=4 {
            units.push(Arc::<str>::from(unit));
            periods.push(period);
            y.push(if period < 4 { position * period as f64 }
                else { 4.0 * position + if unit == "treated" { 5.0 } else { 0.0 } });
        }
    }
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(VariableId::from_raw(0), units, periods, "treated", 4)
        .with_augmentation(1e-8);
    let ctx = ExecutionContext::for_tests(23);
    let prepared = Study::tabular(data.clone()).query(query.clone()).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.synthetic_control.as_ref().unwrap();
    assert!((fit.unadjusted_effect.unwrap() - 9.0).abs() < 1e-5);
    assert!((fit.outcome_model_correction.unwrap() - 4.0).abs() < 1e-4);
    assert!((fit.effect - 5.0).abs() < 1e-4);
    assert_eq!(fit.uncertainty.as_ref(), "point_only_augmented_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, IntervalMethod::None);
    assert!(result.estimate.as_effect().unwrap().se_analytic.is_nan());
    assert!(result.identification.required_assumptions.entries.iter().any(|record|
        format!("{:?}", record.assumption).contains("donor_ridge_outcome_model_transports")));
    let bytes = prepared.encode_contracted_result(&result, "augmented-control", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.synthetic_control.as_ref().unwrap().effect, fit.effect);
    let mut fabricated = body.clone();
    fabricated.synthetic_control.as_mut().unwrap().outcome_model_correction = Some(0.0);
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated").is_err());
    assert!(query.clone().with_uniform_unit_randomization().validate().is_err());
    assert!(query.difference_in_differences().validate().is_err());
}
