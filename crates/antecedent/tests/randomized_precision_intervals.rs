//! Retained interval evidence for Bernoulli precision adjustment.
#![allow(clippy::unusual_byte_groupings, reason = "deterministic seed literals keep their author-chosen digit grouping")]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{RandomizedEffectQuery, Study};
use antecedent_core::VariableId;
use antecedent_data::TabularData;

fn query(n: usize) -> RandomizedEffectQuery {
    RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>(),
        vec![0.5; n],
        (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>(),
        (0..n).map(|i| Arc::<str>::from(format!("row-{i}"))).collect::<Vec<_>>(),
        ("control", "treated"),
    )
}

#[test]
fn supported_ancova_and_fixed_cuped_intervals_round_trip_and_refuse_forgery() {
    let n = 400;
    let x1 = (0..n).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
    let x2 = (0..n).map(|i| (0.11 * i as f64).cos()).collect::<Vec<_>>();
    let y = (0..n).map(|i| {
        2.0 + 1.2 * x1[i] - 0.7 * x2[i] + 0.5 * (0.29 * i as f64).sin()
            + 1.4 * f64::from(i % 2 == 0)
    }).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([
        ("outcome", y.as_slice()), ("baseline_a", x1.as_slice()),
        ("baseline_b", x2.as_slice()),
    ]).unwrap();
    let ctx = ExecutionContext::for_tests(0xA11C_0A);
    for (label, query) in [
        ("ancova", query(n).with_ancova(vec![VariableId::from_raw(1), VariableId::from_raw(2)])),
        ("cuped", query(n).with_fixed_cuped(VariableId::from_raw(1), 1.2)),
    ] {
        let prepared = Study::tabular(data.clone()).query(query).build().unwrap()
            .prepare(&ctx).unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let fit = result.randomized_effect.as_ref().unwrap();
        let [lower, upper] = fit.interval_95.expect("supported precision interval");
        assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
        assert!(lower < fit.effect && fit.effect < upper);
        assert!(fit.standard_error.unwrap() > 0.0);
        assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::AnalyticSe);
        let bytes = prepared.encode_contracted_result(&result, label, &ctx).unwrap();
        let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(artifact.randomized_effect.as_ref().unwrap().interval_95, Some([lower, upper]));
        assert_eq!(artifact.randomized_effect.as_ref().unwrap().graphless_support_status.as_deref(), Some("licensed"));
        let mut forged = artifact.clone();
        forged.randomized_effect.as_mut().unwrap().interval_95.as_mut().unwrap()[1] += 0.5;
        assert!(antecedent_io::encode_analysis_result_artifact(
            &forged, header.variable_names.clone(), "forged-interval").is_err());
        let mut forged = artifact;
        forged.randomized_effect.as_mut().unwrap().standard_error = Some(0.01);
        assert!(antecedent_io::encode_analysis_result_artifact(
            &forged, header.variable_names, "forged-se").is_err());
    }
}

#[test]
fn sparse_ancova_and_fixed_cuped_remain_point_only() {
    let n = 40;
    let x1 = (0..n).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
    let y = (0..n).map(|i| 2.0 + x1[i] + 1.4 * f64::from(i % 2 == 0)
        + 0.2 * (0.31 * i as f64).sin()).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([
        ("outcome", y.as_slice()), ("baseline", x1.as_slice()),
    ]).unwrap();
    for query in [
        query(n).with_ancova(vec![VariableId::from_raw(1)]),
        query(n).with_fixed_cuped(VariableId::from_raw(1), 1.0),
    ] {
        let result = Study::tabular(data.clone()).query(query).build().unwrap()
            .run(&ExecutionContext::for_tests(0xA11C_0A)).unwrap();
        let fit = result.randomized_effect.as_ref().unwrap();
        assert!(fit.interval_95.is_none());
        assert!(fit.standard_error.is_none());
        assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
        assert!(result.support_status.is_none());
    }
}
