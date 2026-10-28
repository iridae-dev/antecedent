//! Multi-arm, independent-score contrast, switchback ITT and history-mean estimators against exhaustive design enumeration.
//!
//! `oracle_data/randomized_design_enumeration.py` (frozen in the JSON beside it) lists every
//! realization of the assignment for small designs with fixed potential outcomes and dyadic
//! probabilities, with the exact rational value of each estimator (and of its variance estimate)
//! on it. While generating it asserts exact unbiasedness and conservative variance estimation
//! over the whole realization set; this test requires the product to reproduce every realization
//! and to refuse exactly the realizations the reference marks as unsupported.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::multi_arm::estimate_multi_arm;
use antecedent_estimate::randomized_scores::independent_action_contrast;
use antecedent_estimate::switchback::{
    HistoryAssignment, switchback_history_means, switchback_itt,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/randomized_design_enumeration.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1e-300),
        "{what}: got {got:e} want {want:e}"
    );
}

#[test]
fn multi_arm_matches_every_realization_of_the_assignment() {
    let reference = reference();
    for design in reference["multi_arm"].as_array().unwrap() {
        let name = design["name"].as_str().unwrap();
        let by_arm: Vec<Vec<f64>> =
            design["outcomes_by_arm"].as_array().unwrap().iter().map(floats).collect();
        let probabilities: Vec<Vec<f64>> =
            design["probabilities"].as_array().unwrap().iter().map(floats).collect();
        let mut mass = 0.0;
        for real in design["realizations"].as_array().unwrap() {
            let assignment: Vec<usize> = real["assignment"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_u64().unwrap() as usize)
                .collect();
            mass += real["prob"].as_f64().unwrap();
            let outcomes: Vec<f64> =
                assignment.iter().enumerate().map(|(i, &a)| by_arm[a][i]).collect();
            let got = estimate_multi_arm(&outcomes, &assignment, &probabilities);
            match real["arms"].as_array() {
                None => assert!(got.is_err(), "{name} {assignment:?}: unsupported arm accepted"),
                Some(arms) => {
                    let got = got.unwrap_or_else(|e| panic!("{name} {assignment:?}: {e}"));
                    for (g, w) in got.iter().zip(arms) {
                        close(g.value, w["value"].as_f64().unwrap(), 1e-12, name);
                        close(g.variance_bound, w["variance"].as_f64().unwrap(), 1e-12, name);
                        assert_eq!(g.observed_support as u64, w["support"].as_u64().unwrap());
                    }
                }
            }
        }
        assert!((mass - 1.0).abs() < 1e-12, "{name}: realization mass {mass}");
    }
}

#[test]
fn switchback_itt_matches_every_realization_including_unequal_sequences() {
    let reference = reference();
    for design in reference["switchback_itt"].as_array().unwrap() {
        let name = design["name"].as_str().unwrap();
        let ids: Vec<&str> =
            design["ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let p = floats(&design["probabilities"]);
        let (y1, y0) = (floats(&design["treated_outcomes"]), floats(&design["control_outcomes"]));
        for real in design["realizations"].as_array().unwrap() {
            let assignment: Vec<bool> = real["assignment"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_u64().unwrap() == 1)
                .collect();
            let outcomes: Vec<f64> =
                (0..p.len()).map(|i| if assignment[i] { y1[i] } else { y0[i] }).collect();
            let got = switchback_itt(&outcomes, &assignment, &p, &ids);
            match real["fit"].as_object() {
                None => assert!(got.is_none(), "{name} {assignment:?}: one-arm schedule accepted"),
                Some(fit) => {
                    let got = got.unwrap_or_else(|| panic!("{name} {assignment:?}: refused"));
                    // the 1e8 outcome offset costs the centred scores about 1e-8 absolute
                    close(got.effect, fit["effect"].as_f64().unwrap(), 1e-9, name);
                    close(got.variance, fit["variance"].as_f64().unwrap(), 1e-9, name);
                    assert_eq!(
                        got.interval_95, None,
                        "{name}: a three-sequence design is point-only"
                    );
                }
            }
        }
    }
}

#[test]
fn history_means_and_covariance_match_every_realization() {
    let reference = reference();
    for design in reference["history"].as_array().unwrap() {
        let name = design["name"].as_str().unwrap();
        let schedules: Vec<Vec<Vec<usize>>> = design["schedules"]
            .as_array()
            .unwrap()
            .iter()
            .map(|seq| {
                seq.as_array()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        s.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
                    })
                    .collect()
            })
            .collect();
        let probabilities: Vec<Vec<f64>> =
            design["probabilities"].as_array().unwrap().iter().map(floats).collect();
        let histories: Vec<Vec<usize>> = design["histories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect())
            .collect();
        for real in design["realizations"].as_array().unwrap() {
            let choice: Vec<usize> = real["choice"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sequences: Vec<HistoryAssignment> = choice
                .iter()
                .enumerate()
                .map(|(s, &c)| HistoryAssignment {
                    schedules: schedules[s].clone(),
                    probabilities: probabilities[s].clone(),
                    realized: c,
                    outcomes: floats(&real["outcomes"][s]),
                })
                .collect();
            let got = switchback_history_means(&sequences, &histories);
            match real["fit"].as_object() {
                None => assert!(got.is_err(), "{name} {choice:?}: unsupported history accepted"),
                Some(fit) => {
                    let got = got.unwrap_or_else(|e| panic!("{name} {choice:?}: {e}"));
                    for (g, w) in got.means.iter().zip(floats(&fit["means"])) {
                        close(*g, w, 1e-12, name);
                    }
                    let want_cov = floats(&fit["covariance"]);
                    // covariance entries may be exactly zero in the reference; compare with a scale
                    let scale = want_cov.iter().fold(0.0_f64, |a, v| a.max(v.abs())).max(1e-300);
                    for (g, w) in got.covariance.iter().zip(&want_cov) {
                        assert!((g - w).abs() <= 1e-12 * scale, "{name} {choice:?}: cov {g} {w}");
                    }
                    assert_eq!(
                        got.observed_support,
                        fit["support"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(got.eligible_periods as u64, fit["eligible"].as_u64().unwrap());
                }
            }
        }
    }
}

#[test]
fn independent_score_contrast_matches_every_realization() {
    let reference = reference();
    for design in reference["scores"].as_array().unwrap() {
        let name = design["name"].as_str().unwrap();
        let probabilities: Vec<Vec<f64>> =
            design["probabilities"].as_array().unwrap().iter().map(floats).collect();
        let (reference_arm, action) = (
            design["reference"].as_u64().unwrap() as usize,
            design["action"].as_u64().unwrap() as usize,
        );
        for real in design["realizations"].as_array().unwrap() {
            let assignment: Vec<usize> = real["assignment"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_u64().unwrap() as usize)
                .collect();
            let outcomes = floats(&real["outcomes"]);
            let got = independent_action_contrast(
                &outcomes,
                &assignment,
                &probabilities,
                reference_arm,
                action,
            );
            match real["fit"].as_object() {
                None => assert!(got.is_none(), "{name} {assignment:?}: unobserved arm accepted"),
                Some(fit) => {
                    let got = got.unwrap_or_else(|| panic!("{name} {assignment:?}: refused"));
                    // the 1e8 offset costs the scores about 1e-8 absolute
                    let rel = if name == "two_arm_offset_outcomes" { 1e-8 } else { 1e-12 };
                    let want = fit["effect"].as_f64().unwrap();
                    assert!(
                        (got.effect - want).abs() <= rel * want.abs().max(1.0),
                        "{name}: effect {} vs {want}",
                        got.effect
                    );
                    let want = fit["variance"].as_f64().unwrap();
                    assert!(
                        (got.variance_upper_bound - want).abs() <= rel * want.max(1e-300),
                        "{name}: variance {} vs {want}",
                        got.variance_upper_bound
                    );
                    assert_eq!(
                        got.reference_support as u64,
                        fit["reference_support"].as_u64().unwrap()
                    );
                    assert_eq!(got.action_support as u64, fit["action_support"].as_u64().unwrap());
                    assert_eq!(got.interval_95, None, "{name}: a six-row design has no interval");
                }
            }
        }
    }
}
