//! Retained fuzzy RD and regression-kink route evidence.

use antecedent::prelude::ExecutionContext;
use antecedent::{LocalPolynomialRatioQuery, Study};
use antecedent_core::{IntervalMethod, VariableId};
use antecedent_data::{TableView, TabularData};

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
fn retained_fuzzy_rd_recovers_truth_and_rejects_fabricated_interval() {
    let (data, query) = fixture(false);
    let ctx = ExecutionContext::for_tests(41);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(prepared.estimate(&data, &ctx).unwrap().local_polynomial_ratio, result.local_polynomial_ratio);
    let fit = result.local_polynomial_ratio.as_ref().unwrap();
    assert!((fit.effect - 3.0).abs() < 1e-8);
    assert_eq!((fit.n_left, fit.n_right), (316, 316));
    assert_eq!(fit.uncertainty.as_ref(), "point_only_with_unvalidated_hc0_standard_error");
    assert_eq!(result.interval.as_ref().unwrap().method, IntervalMethod::None);
    assert!(result.estimate.as_effect().unwrap().se_analytic.is_nan());
    assert!(result.identification.required_assumptions.entries.iter().any(|record|
        format!("{:?}", record.assumption).contains("local_continuity")));
    let bytes = prepared.encode_contracted_result(&result, "fuzzy-rd", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.local_polynomial_ratio.as_ref().unwrap().effect, fit.effect);
    assert!(body.standard_error.is_none());
    let mut fabricated = body.clone();
    fabricated.interval_lower = Some(2.0);
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated").is_err());
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
