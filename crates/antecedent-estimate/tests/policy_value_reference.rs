//! Fixed-policy value against exact rational scores: binary augmented and inverse-probability
//! scores, multi-action Horvitz-Thompson scores, fixed-stratum action effects, uplift bins and
//! finite-class regret.
//!
//! `oracle_data/policy_value_reference.py` (frozen in the JSON beside it) writes each score from
//! its estimand in `fractions.Fraction` arithmetic over dyadic inputs, so only the square root of
//! a standard error and the 50-digit normal quantile are not rational. Cases cover scalar and
//! row-aligned propensities (including 1/1024 and 1023/1024), row-aligned costs, a reference that
//! treats, an outcome offset of 1e7, a two-row table, three actions in two strata, singleton and
//! multi-row uplift bins and a three-candidate class.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::MultiActionPolicyInputs;
use antecedent_estimate::policy_value::{
    BinaryIpwDesign, BinaryPolicyDesign, BinaryPolicyLimits, MultiActionIpwDesign,
    MultiActionLimits, evaluate_binary_ipw_policy, evaluate_fixed_candidate_regret,
    evaluate_multi_action_cate, evaluate_multi_action_ipw_policy,
    evaluate_multi_action_policy_value_scores, evaluate_policy_value_ipw_scores,
    evaluate_policy_value_scores, evaluate_uplift_bins, uplift_by_score_bin,
};
use serde_json::Value;

fn cases(kind: &str) -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/policy_value_reference.json")).unwrap()
        ["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == kind)
        .cloned()
        .collect()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn idx(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| usize::try_from(x.as_u64().unwrap()).unwrap()).collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-10 * want.abs().max(1.0), "{what}: got {got} want {want}");
}

#[test]
fn binary_scores_standard_errors_and_point_values_match_the_exact_scores() {
    for case in cases("binary") {
        let name = case["name"].as_str().unwrap();
        let (y, t) = (floats(&case["y"]), bools(&case["t"]));
        let (e, actions, reference) =
            (floats(&case["e"]), bools(&case["actions"]), bools(&case["reference"]));
        let (costs, rcosts) = (floats(&case["costs"]), floats(&case["reference_costs"]));
        let n = y.len();
        // The reference script lists one propensity or cost per row; scalar cases repeat them.
        let scalar = |v: &Vec<f64>| if v.iter().all(|x| *x == v[0]) { vec![v[0]] } else { v.clone() };
        let (e, costs, rcosts) = (scalar(&e), scalar(&costs), scalar(&rcosts));
        let design = BinaryPolicyDesign {
            assignment: &t,
            actions: &actions,
            propensity: &e,
            reference: &reference,
            costs: &costs,
            reference_costs: &rcosts,
        };
        let scores = if case["mu0"].is_null() {
            evaluate_policy_value_ipw_scores(&y, &design).unwrap()
        } else {
            evaluate_policy_value_scores(&y, &design, &floats(&case["mu0"]), &floats(&case["mu1"]))
                .unwrap()
        };
        for (got, key) in [
            (scores.policy_value, "policy_value"),
            (scores.reference_value, "reference_value"),
            (scores.incremental_value, "incremental_value"),
            (scores.policy_standard_error, "policy_se"),
            (scores.reference_standard_error, "reference_se"),
            (scores.incremental_standard_error, "incremental_se"),
            (scores.treatment_rate, "treatment_rate"),
            (scores.total_cost, "total_cost"),
            (scores.propensity_min, "propensity_min"),
            (scores.propensity_max, "propensity_max"),
        ] {
            close(got, case[key].as_f64().unwrap(), &format!("{name} {key}"));
        }
        close(scores.relative_value_gap, scores.reference_value - scores.policy_value, name);

        if case["mu0"].is_null() {
            let ipw = evaluate_binary_ipw_policy(
                &y,
                &BinaryIpwDesign {
                    assignment: &t,
                    actions: &actions,
                    propensity: &e,
                    reference: Some(&reference),
                    costs: &costs,
                    reference_costs: Some(&rcosts),
                    available: None,
                },
                &BinaryPolicyLimits::default(),
            )
            .unwrap();
            close(ipw.policy_net, case["ipw_policy_net"].as_f64().unwrap(), name);
            close(ipw.reference_net, case["ipw_reference_net"].as_f64().unwrap(), name);
            close(ipw.total_cost, case["total_cost"].as_f64().unwrap(), name);
            assert_eq!(n, y.len());
        }
    }
}

fn multi_inputs(case: &Value) -> MultiActionPolicyInputs {
    let labels: Vec<Arc<str>> =
        case["labels"].as_array().unwrap().iter().map(|s| Arc::from(s.as_str().unwrap())).collect();
    let k = labels.len();
    let n = case["y"].as_array().unwrap().len();
    let groups: Vec<Arc<str>> =
        case["groups"].as_array().unwrap().iter().map(|s| Arc::from(s.as_str().unwrap())).collect();
    MultiActionPolicyInputs {
        action_labels: Arc::from(labels),
        assignment: Arc::from(idx(&case["assignment"])),
        actions: Arc::from(idx(&case["actions"])),
        reference: Arc::from(idx(&case["reference"])),
        propensities: Arc::from(floats(&case["propensities"])),
        available: Arc::from(vec![true; n * k]),
        costs: Arc::from(floats(&case["costs"])),
        reference_costs: Arc::from(floats(&case["reference_costs"])),
        capacities: Arc::from(vec![n; k]),
        reference_capacities: Arc::from(vec![n; k]),
        budget: None,
        reference_budget: None,
        cate_groups: Arc::from(groups),
    }
}

#[test]
fn multi_action_scores_point_values_and_stratum_effects_match_the_exact_scores() {
    for case in cases("multi") {
        let name = case["name"].as_str().unwrap();
        let y = floats(&case["y"]);
        let inputs = multi_inputs(&case);
        let scores = evaluate_multi_action_policy_value_scores(&y, &inputs).unwrap();
        for (got, key) in [
            (scores.policy_value, "policy_value"),
            (scores.reference_value, "reference_value"),
            (scores.incremental_value, "incremental_value"),
            (scores.policy_standard_error, "policy_se"),
            (scores.reference_standard_error, "reference_se"),
            (scores.incremental_standard_error, "incremental_se"),
            (scores.treatment_rate, "non_control_rate"),
            (scores.total_cost, "policy_cost"),
            (scores.propensity_min, "propensity_min"),
            (scores.propensity_max, "propensity_max"),
        ] {
            close(got, case[key].as_f64().unwrap(), &format!("{name} {key}"));
        }
        let n = y.len();
        let k = inputs.action_labels.len();
        let point = evaluate_multi_action_ipw_policy(
            &y,
            &MultiActionIpwDesign {
                assigned: &inputs.assignment,
                actions: &inputs.actions,
                probabilities: &inputs.propensities,
                probability_rows: n,
                action_count: k,
                reference: Some(&inputs.reference),
                costs: Some(&inputs.costs),
                reference_costs: Some(&inputs.reference_costs),
                available: None,
            },
            &MultiActionLimits::default(),
        )
        .unwrap();
        close(point.policy_value, case["ipw_policy"].as_f64().unwrap(), name);
        close(point.reference_value, case["ipw_reference"].as_f64().unwrap(), name);
        close(point.policy_cost, case["policy_cost"].as_f64().unwrap(), name);

        let cate = evaluate_multi_action_cate(&y, &inputs).unwrap();
        let want = case["cate"].as_array().unwrap();
        assert_eq!(cate.len(), want.len(), "{name}");
        for (got, want) in cate.iter().zip(want) {
            assert_eq!(got.group, want["group"].as_str().unwrap());
            assert_eq!(got.action, want["action"].as_str().unwrap());
            close(got.effect, want["effect"].as_f64().unwrap(), "cate effect");
            close(got.standard_error, want["se"].as_f64().unwrap(), "cate se");
            assert_eq!(got.evaluation_rows, want["rows"].as_u64().unwrap() as usize);
            assert_eq!(got.observed_action_rows, want["action_rows"].as_u64().unwrap() as usize);
            assert_eq!(got.observed_control_rows, want["control_rows"].as_u64().unwrap() as usize);
            assert!(got.interval_95.is_none(), "twelve rows are below the interval floor");
        }
    }
}

#[test]
fn uplift_bins_match_the_exact_scores_and_a_singleton_bin_is_handled_as_documented() {
    for case in cases("uplift") {
        let (y, t, e) = (floats(&case["y"]), bools(&case["t"]), floats(&case["e"]));
        let bins = idx(&case["bins"]);
        let want = case["bins_expected"].as_array().unwrap();
        let point = uplift_by_score_bin(&y, &t, &bins, &e, 3).unwrap();
        let interval_form = evaluate_uplift_bins(&y, &t, &e, &bins, 3).unwrap();
        for (b, w) in want.iter().enumerate() {
            close(point[b].effect, w["effect"].as_f64().unwrap(), "uplift effect");
            close(point[b].standard_error, w["se"].as_f64().unwrap(), "uplift se");
            close(interval_form[b].effect, w["effect"].as_f64().unwrap(), "uplift effect");
            close(interval_form[b].standard_error, w["se"].as_f64().unwrap(), "uplift se");
            assert!(interval_form[b].interval_95.is_none());
        }
    }
    for case in cases("uplift_singleton") {
        let (y, t, e) = (floats(&case["y"]), bools(&case["t"]), floats(&case["e"]));
        let bins = idx(&case["bins"]);
        let point = uplift_by_score_bin(&y, &t, &bins, &e, 2).unwrap();
        close(point[0].effect, case["first_effect"].as_f64().unwrap(), "first bin");
        close(point[0].standard_error, case["first_se"].as_f64().unwrap(), "first se");
        close(point[1].effect, case["second_effect"].as_f64().unwrap(), "singleton effect");
        assert!(point[1].standard_error.abs() < 1e-300, "a singleton bin has a zero error");
        assert!(evaluate_uplift_bins(&y, &t, &e, &bins, 2).is_err());
    }
}

#[test]
fn finite_class_regret_interval_matches_the_exact_bonferroni_contrasts() {
    for case in cases("regret") {
        let (y, t, e) = (floats(&case["y"]), bools(&case["t"]), floats(&case["e"]));
        let candidates: Vec<Vec<bool>> =
            case["candidates"].as_array().unwrap().iter().map(bools).collect();
        let selected = case["selected"].as_u64().unwrap() as usize;
        let got = evaluate_fixed_candidate_regret(
            &y,
            &t,
            &e,
            &candidates,
            selected,
            &floats(&case["costs"]),
            true,
        )
        .unwrap();
        for (a, b) in got.candidate_values.iter().zip(floats(&case["values"])) {
            close(*a, b, "candidate value");
        }
        close(got.regret, case["regret"].as_f64().unwrap(), "regret");
        for (a, b) in got.contrast_standard_errors.iter().zip(floats(&case["ses"])) {
            close(*a, b, "contrast se");
        }
        let want = floats(&case["interval"]);
        close(got.interval_95[0], want[0], "regret lower");
        close(got.interval_95[1], want[1], "regret upper");
    }
}
