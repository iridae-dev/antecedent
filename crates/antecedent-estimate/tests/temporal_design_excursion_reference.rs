//! Horvitz-Thompson excursion designs and their cluster-robust ratio inference against exact
//! rational references.
//!
//! `oracle_data/temporal_design_excursion_reference.py` (frozen in the JSON beside it) pairs each
//! decision with the outcome row `lag` periods later, forms the per-pair Horvitz-Thompson scores of
//! a two-arm design with probabilities as small as 0.06 and an outcome offset of 1e6, a four-arm
//! ordered design with three clusters, a two-factor factorial and a within-cluster interference
//! design (peer exposure probabilities from the other observed members), in
//! `fractions.Fraction`, and from them the ratio estimate, the linearised cluster-robust
//! covariance with the G/(G-1) factor, the normal interval and the Holm step-down adjusted
//! p-values. The product forms the same scores and sums, so this checks the pairing across
//! missing periods, the score of every coordinate, the cluster accumulation and the sandwich.
//! The Bayesian bootstrap, the simultaneous band and the quantile, dose, observational and regime
//! functionals are other routes and are not covered here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::temporal_design::contrasts::{
    factorial_scores, interference_scores, multi_arm_scores, two_arm_scores,
};
use antecedent_estimate::temporal_design::{
    ClusterScores, PanelRows, holm_adjusted, lagged_pairs, sampling_summary,
};
use serde_json::Value;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1e-12),
        "{what}: {actual} vs {expected}"
    );
}

fn column<T>(rows: &[Value], key: &str, read: impl Fn(&Value) -> T) -> Vec<T> {
    rows.iter().map(|r| read(&r[key])).collect()
}

fn count(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}

#[test]
fn excursion_designs_match_the_exact_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/temporal_design_excursion_reference.json"))
            .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rows = case["rows"].as_array().unwrap();
        let unit: Vec<usize> = column(rows, "unit", count);
        let period: Vec<i64> = column(rows, "period", |v| v.as_i64().unwrap());
        let outcome: Vec<f64> = column(rows, "y", |v| v.as_f64().unwrap());
        let lag = u32::try_from(case["lag"].as_u64().unwrap()).unwrap();
        let pairs = lagged_pairs(PanelRows { unit: &unit, period: &period }, lag).unwrap();
        assert_eq!(pairs.decision.len(), count(&case["pairs"]), "{name}: pairs");
        assert_eq!(pairs.unpaired, count(&case["unpaired"]), "{name}: unpaired");
        let design = match case["design"].as_str().unwrap() {
            "two_arm" => two_arm_scores(
                &pairs,
                &outcome,
                &column(rows, "a", |v| v.as_bool().unwrap()),
                &column(rows, "p", |v| v.as_f64().unwrap()),
                &unit,
                0.05,
            ),
            "multi_arm" => {
                let arms: Vec<String> = (0..4).map(|k| format!("arm{k}")).collect();
                let probabilities: Vec<Vec<f64>> = column(rows, "probs", |v| {
                    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
                });
                let cluster: Vec<usize> = unit.iter().map(|u| u / 3).collect();
                multi_arm_scores(
                    &pairs,
                    &outcome,
                    &column(rows, "arm", count),
                    &probabilities,
                    &arms,
                    &cluster,
                    true,
                    0.01,
                )
            }
            "factorial" => factorial_scores(
                &pairs,
                &outcome,
                &column(rows, "a", |v| v.as_bool().unwrap()),
                &column(rows, "pa", |v| v.as_f64().unwrap()),
                &column(rows, "b", |v| v.as_bool().unwrap()),
                &column(rows, "pb", |v| v.as_f64().unwrap()),
                &unit,
                0.01,
            ),
            "interference" => interference_scores(
                &pairs,
                &outcome,
                &column(rows, "a", |v| v.as_bool().unwrap()),
                &column(rows, "p", |v| v.as_f64().unwrap()),
                &column(rows, "group", count),
                &period,
                0.01,
            ),
            other => panic!("unknown design {other}"),
        }
        .unwrap();
        close(
            design.minimum_probability,
            case["minimum_probability"].as_f64().unwrap(),
            1e-12,
            &format!("{name}: smallest realized probability"),
        );
        let k = count(&case["coordinates"]);
        assert_eq!(design.coordinates.len(), k, "{name}");
        let accumulated =
            ClusterScores::accumulate(&design.scores, k, &design.cluster_of_pair).unwrap();
        assert_eq!(accumulated.clusters(), count(&case["clusters"]), "{name}: clusters");
        let summary = sampling_summary(&accumulated).unwrap();
        let floats = |key: &str| -> Vec<f64> {
            case[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
        };
        for (j, ((e, s), (lo, hi))) in summary
            .estimate
            .iter()
            .zip(&summary.standard_error)
            .zip(summary.lower.iter().zip(&summary.upper))
            .enumerate()
        {
            let at = format!("{name} coordinate {} ({})", j, design.coordinates[j]);
            close(*e, floats("estimate")[j], 1e-10, &format!("{at} estimate"));
            close(*s, floats("standard_error")[j], 1e-9, &format!("{at} se"));
            close(*lo, floats("lower")[j], 1e-9, &format!("{at} lower"));
            close(*hi, floats("upper")[j], 1e-9, &format!("{at} upper"));
        }
        for (a, row) in case["covariance"].as_array().unwrap().iter().enumerate() {
            for (b, want) in row.as_array().unwrap().iter().enumerate() {
                close(
                    summary.covariance[a * k + b],
                    want.as_f64().unwrap(),
                    1e-9,
                    &format!("{name} covariance ({a},{b})"),
                );
            }
        }
        let family: Vec<usize> = case["family"].as_array().unwrap().iter().map(count).collect();
        assert_eq!(design.family, family, "{name}: family");
        for (got, want) in holm_adjusted(&summary, &family).iter().zip(floats("holm")) {
            close(*got, want, 1e-7, &format!("{name} Holm"));
        }
    }
}
