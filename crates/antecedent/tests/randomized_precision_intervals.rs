//! Retained interval evidence for Bernoulli precision adjustment.
#![allow(
    clippy::unusual_byte_groupings,
    reason = "deterministic seed literals keep their author-chosen digit grouping"
)]

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
    let y = (0..n)
        .map(|i| {
            2.0 + 1.2 * x1[i] - 0.7 * x2[i]
                + 0.5 * (0.29 * i as f64).sin()
                + 1.4 * f64::from(i % 2 == 0)
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([
        ("outcome", y.as_slice()),
        ("baseline_a", x1.as_slice()),
        ("baseline_b", x2.as_slice()),
    ])
    .unwrap();
    let ctx = ExecutionContext::for_tests(0xA11C_0A);
    for (label, query) in [
        ("ancova", query(n).with_ancova(vec![VariableId::from_raw(1), VariableId::from_raw(2)])),
        ("cuped", query(n).with_fixed_cuped(VariableId::from_raw(1), 1.2)),
    ] {
        let prepared =
            Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let fit = result.randomized_effect().unwrap();
        let [lower, upper] = fit.interval_95.expect("supported precision interval");
        assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
        assert!(lower < fit.effect && fit.effect < upper);
        assert!(fit.standard_error.unwrap() > 0.0);
        assert_eq!(
            result.interval.as_ref().unwrap().method,
            antecedent_core::IntervalMethod::AnalyticSe
        );
        let bytes = prepared.encode_contracted_result(&result, label, &ctx).unwrap();
        let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(artifact.randomized_effect.as_ref().unwrap().interval_95, Some([lower, upper]));
        assert_eq!(
            artifact.randomized_effect.as_ref().unwrap().graphless_support_status.as_deref(),
            Some("licensed")
        );
        let mut forged = artifact.clone();
        forged.randomized_effect.as_mut().unwrap().interval_95.as_mut().unwrap()[1] += 0.5;
        assert!(
            antecedent_io::encode_analysis_result_artifact(
                &forged,
                header.variable_names.clone(),
                "forged-interval"
            )
            .is_err()
        );
        let mut forged = artifact;
        forged.randomized_effect.as_mut().unwrap().standard_error = Some(0.01);
        assert!(
            antecedent_io::encode_analysis_result_artifact(
                &forged,
                header.variable_names,
                "forged-se"
            )
            .is_err()
        );
    }
}

#[test]
fn sparse_ancova_and_fixed_cuped_remain_point_only() {
    let n = 40;
    let x1 = (0..n).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| 2.0 + x1[i] + 1.4 * f64::from(i % 2 == 0) + 0.2 * (0.31 * i as f64).sin())
        .collect::<Vec<_>>();
    let data =
        TabularData::from_f64_columns([("outcome", y.as_slice()), ("baseline", x1.as_slice())])
            .unwrap();
    for query in [
        query(n).with_ancova(vec![VariableId::from_raw(1)]),
        query(n).with_fixed_cuped(VariableId::from_raw(1), 1.0),
    ] {
        let result = Study::tabular(data.clone())
            .query(query)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(0xA11C_0A))
            .unwrap();
        let fit = result.randomized_effect().unwrap();
        assert!(fit.interval_95.is_none());
        assert!(fit.standard_error.is_none());
        assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
        assert!(result.support_status.is_none());
    }
}

/// A Bernoulli trial of `n` rows whose realized treated rows are the ones `treated` selects, with
/// a common declared treatment probability, and its outcome and two baseline covariates.
fn trial(
    n: usize,
    treated: impl Fn(usize) -> bool,
    probability: f64,
) -> (RandomizedEffectQuery, TabularData) {
    let x1 = (0..n).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
    let x2 = (0..n).map(|i| (0.11 * i as f64).cos()).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            2.0 + 1.2 * x1[i] - 0.7 * x2[i]
                + 0.5 * (0.29 * i as f64).sin()
                + 1.4 * f64::from(treated(i))
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([
        ("outcome", y.as_slice()),
        ("baseline_a", x1.as_slice()),
        ("baseline_b", x2.as_slice()),
    ])
    .unwrap();
    let query = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        (0..n).map(&treated).collect::<Vec<_>>(),
        vec![probability; n],
        (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>(),
        (0..n).map(|i| Arc::<str>::from(format!("row-{i}"))).collect::<Vec<_>>(),
        ("control", "treated"),
    );
    (query, data)
}

fn constructions(query: &RandomizedEffectQuery) -> [(&'static str, RandomizedEffectQuery); 3] {
    [
        ("score", query.clone()),
        (
            "ancova",
            query.clone().with_ancova(vec![VariableId::from_raw(1), VariableId::from_raw(2)]),
        ),
        ("cuped", query.clone().with_fixed_cuped(VariableId::from_raw(1), 1.2)),
    ]
}

#[test]
fn three_hundred_row_trials_publish_intervals_from_their_own_arm_support() {
    let ctx = ExecutionContext::for_tests(0xA11C_0B);
    // Balanced, and a 2:1 allocation (one third treated, 100 and 200 rows per arm).
    for (name, n, treated, probability) in [
        ("balanced", 300, (|i: usize| i % 2 == 0) as fn(usize) -> bool, 0.5),
        ("two_to_one", 300, |i: usize| i % 3 == 0, 1.0 / 3.0),
        ("probability_boundary", 300, |i: usize| i % 5 == 0, 0.2),
    ] {
        let (query, data) = trial(n, treated, probability);
        for (label, query) in constructions(&query) {
            let result = Study::tabular(data.clone())
                .query(query)
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap_or_else(|error| panic!("{name} {label}: {error}"));
            let fit = result.randomized_effect().unwrap();
            let [lower, upper] = fit
                .interval_95
                .unwrap_or_else(|| panic!("{name} {label}: 300 rows must publish an interval"));
            assert!(lower < fit.effect && fit.effect < upper, "{name} {label}");
            assert_eq!(
                result.support_status,
                Some(antecedent::support::CellStatus::Licensed),
                "{name} {label}"
            );
        }
    }
}

#[test]
fn weak_arm_support_withholds_the_interval_and_names_the_arm_rule() {
    let ctx = ExecutionContext::for_tests(0xA11C_0B);
    // 300 rows with a balanced declared probability but only 20 realized treated rows: the
    // observed-arm rule fails whatever the row count. 100 rows at probability 0.25 realizing 40
    // treated rows: the design expects only 25, so the expected-arm rule fails.
    for (name, n, treated, probability) in [
        ("realized_arm_of_20", 300, (|i: usize| i % 15 == 0) as fn(usize) -> bool, 0.5),
        ("expected_arm_of_25", 100, |i: usize| i % 5 < 2, 0.25),
    ] {
        let (query, data) = trial(n, treated, probability);
        for (label, query) in constructions(&query) {
            let result =
                Study::tabular(data.clone()).query(query).build().unwrap().run(&ctx).unwrap();
            let fit = result.randomized_effect().unwrap();
            assert!(result.support_status.is_none(), "{name} {label}");
            if label == "ancova" {
                // The ANCOVA has a small-arm construction of its own, never licensed.
                assert!(fit.interval_95.is_some(), "{name} {label}");
                assert_eq!(
                    fit.uncertainty.as_ref(),
                    "bernoulli_ancova_hc2_bell_mccaffrey_student_interval",
                    "{name} {label}"
                );
                continue;
            }
            assert!(fit.interval_95.is_none(), "{name} {label}");
            assert!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| d.message.contains("30 observed and 40 expected")),
                "{name} {label}: the refusal must name the arm-support rule"
            );
        }
    }
}
