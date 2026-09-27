//! Retained fuzzy RD and regression-kink route evidence.

use antecedent::prelude::ExecutionContext;
use antecedent::{LocalPolynomialRatioQuery, Study};
use antecedent_core::{IntervalMethod, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::local_polynomial_ratio::fit_local_polynomial_ratio;

fn fixture(kink: bool) -> (TabularData, LocalPolynomialRatioQuery) {
    let mut x = Vec::new();
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    for step in 1..80 {
        for sign in [-1.0, 1.0] {
            let score = sign * f64::from(step) / 80.0;
            for replicate in 0..4 {
                let dose = if kink {
                    1.0 + 0.2 * score + 0.8 * score.max(0.0)
                } else if sign < 0.0 {
                    if replicate == 0 { 1.0 } else { 0.0 }
                } else {
                    if replicate != 3 { 1.0 } else { 0.0 }
                };
                x.push(score);
                treatment.push(dose);
                outcome.push(1.0 + 2.0 * score + 0.5 * score * score + 3.0 * dose
                    + if kink { 0.0 } else { [-0.02, -0.01, 0.01, 0.02][replicate] });
            }
        }
    }
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()), ("t", treatment.as_slice()), ("y", outcome.as_slice()),
    ]).unwrap();
    let query = LocalPolynomialRatioQuery {
        outcome: VariableId::from_raw(2), treatment: VariableId::from_raw(1),
        running: VariableId::from_raw(0), cutoff: 0.0, bandwidth: 1.0, kink,
    };
    (data, query)
}

#[test]
fn retained_fuzzy_rd_recovers_truth_and_round_trips_calibrated_interval() {
    let (data, query) = fixture(false);
    let ctx = ExecutionContext::for_tests(41);
    let prepared = Study::tabular(data.clone()).query(query.clone()).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(prepared.estimate(&data, &ctx).unwrap().local_polynomial_ratio, result.local_polynomial_ratio);
    let fit = result.local_polynomial_ratio.as_ref().unwrap();
    assert!((fit.effect - 3.0).abs() < 1e-8);
    assert_eq!((fit.n_left, fit.n_right), (316, 316));
    let direct = fit_local_polynomial_ratio(
        match data.column(query.running).unwrap() { antecedent_data::ColumnView::Float64(c) => &c.values, _ => unreachable!() },
        match data.column(query.outcome).unwrap() { antecedent_data::ColumnView::Float64(c) => &c.values, _ => unreachable!() },
        match data.column(query.treatment).unwrap() { antecedent_data::ColumnView::Float64(c) => &c.values, _ => unreachable!() },
        query.cutoff, query.bandwidth, query.kink,
    ).unwrap();
    assert_eq!(fit.standard_error, direct.standard_error);
    assert_eq!(fit.reduced_form_standard_error, direct.reduced_form_standard_error);
    assert_eq!(fit.first_stage_standard_error, direct.first_stage_standard_error);
    assert_eq!((fit.cutoff, fit.bandwidth, fit.kink), (query.cutoff, query.bandwidth, query.kink));
    assert_eq!(fit.uncertainty.as_ref(), "rbc_hc0_delta_normal_fixed_bandwidth");
    assert_eq!(result.interval.as_ref().unwrap().method, IntervalMethod::AnalyticSe);
    assert_eq!(result.estimate.as_effect().unwrap().se_analytic, fit.standard_error);
    assert!(result.identification.required_assumptions.entries.iter().any(|record|
        format!("{:?}", record.assumption).contains("local_continuity")));
    let bytes = prepared.encode_contracted_result(&result, "fuzzy-rd", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.local_polynomial_ratio.as_ref().unwrap().effect, fit.effect);
    assert_eq!(body.local_polynomial_ratio.as_ref().unwrap().first_stage_standard_error, direct.first_stage_standard_error);
    assert_eq!(body.standard_error, Some(fit.standard_error));
    assert!(body.local_polynomial_ratio.as_ref().unwrap().ci_lower.unwrap() < fit.effect);
    assert!(body.local_polynomial_ratio.as_ref().unwrap().ci_upper.unwrap() > fit.effect);
    let mut fabricated = body.clone();
    fabricated.local_polynomial_ratio.as_mut().unwrap().ci_lower = Some(2.0);
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names.clone(), "fabricated").is_err());
    let mut mismatched = body.clone();
    mismatched.local_polynomial_ratio.as_mut().unwrap().bandwidth = 0.5;
    assert!(antecedent_io::encode_analysis_result_artifact(&mismatched, header.variable_names, "mismatched").is_err());
}

#[test]
fn retained_regression_kink_recovers_truth_and_refuses_weak_first_stage() {
    let (data, query) = fixture(true);
    let ctx = ExecutionContext::for_tests(43);
    let result = Study::tabular(data).query(query).build().unwrap().run(&ctx).unwrap();
    assert!((result.local_polynomial_ratio.as_ref().unwrap().effect - 3.0).abs() < 1e-7);
    let (mut data, query) = fixture(false);
    let y = match data.column(VariableId::from_raw(2)).unwrap() {
        antecedent_data::ColumnView::Float64(column) => column.values.clone(), _ => unreachable!(),
    };
    let x = match data.column(VariableId::from_raw(0)).unwrap() {
        antecedent_data::ColumnView::Float64(column) => column.values.clone(), _ => unreachable!(),
    };
    let zero = vec![0.0; x.len()];
    data = TabularData::from_f64_columns([
        ("x", x.as_slice()), ("t", zero.as_slice()), ("y", y.as_slice()),
    ]).unwrap();
    assert!(Study::tabular(data).query(query).build().unwrap().run(&ctx).is_err());
}
