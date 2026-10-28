//! Doubly robust policy value against a 60-digit reference with recorded splits.
//!
//! `oracle_data/observational_policy_reference.py` (frozen in the JSON beside it) takes the data,
//! the declared policy and the training and scored rows of every split as part of the case. It
//! fits the baseline-category multinomial logit by Newton's method and each action's outcome
//! regression by least squares on the training rows, forms the augmented scores
//! `mu_c + 1{T = c}(y - mu_c)/e_c`, and writes the policy value, its standard error, every
//! action's value, the paired difference with the reference action, the policy shares and the
//! smallest propensities. The seeded split plan is a separate kernel and is not reviewed here: the
//! test asserts that the product's scored row order reproduces the recorded splits, so a changed
//! plan fails loudly instead of de-synchronising the reference. Cases cover a prespecified policy,
//! a policy learned on held-out rows, cross-fitted learning (minimising and maximising), a known
//! constant propensity, and an outcome offset of 1e6 with a covariate offset of 1e3.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::observational_policy::{
    KnownPropensity, ObservationalPolicyValue, PolicyData, PolicyEvaluation, PolicyObjective,
    PolicyOrigin, PolicyValueEstimate,
};
use serde_json::Value;

const SEED: u64 = 5;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn indices(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

fn try_run(case: &Value) -> Result<PolicyValueEstimate, antecedent_estimate::EstimationError> {
    let support: Vec<f64> = floats(&case["support"]);
    let objective = match case["objective"].as_str().unwrap() {
        "maximize" => PolicyObjective::Maximize,
        _ => PolicyObjective::Minimize,
    };
    let evaluation = match case["evaluation"].as_str().unwrap() {
        "prespecified" => PolicyEvaluation::Prespecified {
            actions: Arc::from(floats(&case["policy"])),
            origin: PolicyOrigin::Prespecified,
        },
        "held_out" => {
            PolicyEvaluation::HeldOut { evaluation_fraction: case["heldout"].as_f64().unwrap() }
        }
        _ => PolicyEvaluation::CrossFitted,
    };
    let mut request = ObservationalPolicyValue::new(support, objective, evaluation);
    if let Some(folds) = case["folds"].as_u64() {
        request.folds = folds as usize;
    }
    request.reference = case["reference"].as_f64();
    if !case["known_propensity"].is_null() {
        request.known_propensity =
            Some(KnownPropensity::Constant(Arc::from(floats(&case["known_propensity"]))));
    }
    let covariates: Vec<Vec<f64>> =
        case["covariates"].as_array().unwrap().iter().map(floats).collect();
    let columns: Vec<&[f64]> = covariates.iter().map(Vec::as_slice).collect();
    request.estimate(
        &PolicyData {
            outcome: &floats(&case["y"]),
            treatment: &floats(&case["treatment"]),
            covariates: &columns,
        },
        &ExecutionContext::for_tests(SEED),
    )
}

fn run(case: &Value) -> PolicyValueEstimate {
    try_run(case).unwrap()
}

/// The scored rows in blocks that ascend within a split, cut where the order descends.
fn blocks(rows: &[usize]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    for (i, &row) in rows.iter().enumerate() {
        if i == 0 || row < rows[i - 1] {
            out.push(Vec::new());
        }
        out.last_mut().unwrap().push(row);
    }
    out
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-12), "{what}: got {got} want {want}");
}

#[test]
fn values_scores_and_comparisons_match_the_recorded_split_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/observational_policy_reference.json"))
            .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let want = &case["expected"];
        let got = run(case);
        let rel = if name.contains("offset") { 1e-5 } else { 1e-7 };
        let order: Vec<usize> = got.row_index.iter().map(|id| id.get() as usize).collect();
        let recorded: Vec<Vec<usize>> =
            case["splits"].as_array().unwrap().iter().map(|s| indices(&s["valid"])).collect();
        assert_eq!(blocks(&order), recorded, "{name}: the product's splits are the recorded ones");
        assert_eq!(order, indices(&want["row_order"]), "{name}");
        let scores = floats(&want["scores"]);
        for (r, (g, w)) in got.scores.iter().zip(&scores).enumerate() {
            close(*g, *w, rel, &format!("{name} score {r}"));
        }
        assert_eq!(got.policy_actions, floats(&want["policy_actions"]), "{name}");
        close(got.value, want["value"].as_f64().unwrap(), rel, &format!("{name} value"));
        close(got.se, want["se"].as_f64().unwrap(), rel, &format!("{name} se"));
        let (values, ses) = (floats(&want["action_values"]), floats(&want["action_se"]));
        let (shares, lows) = (floats(&want["policy_share"]), floats(&want["propensity_min"]));
        for (a, action) in got.action_values.iter().enumerate() {
            close(action.value, values[a], rel, &format!("{name} action {a} value"));
            close(action.se, ses[a], rel, &format!("{name} action {a} se"));
            close(action.policy_share, shares[a], 1e-12, &format!("{name} share {a}"));
            close(action.propensity_min, lows[a], rel, &format!("{name} propensity min {a}"));
        }
        if let Some(reference) = got.reference {
            close(
                reference.incremental_value,
                want["incremental_value"].as_f64().unwrap(),
                rel,
                &format!("{name} incremental"),
            );
            close(
                reference.incremental_se,
                want["incremental_se"].as_f64().unwrap(),
                rel,
                &format!("{name} incremental se"),
            );
        } else {
            assert!(case["reference"].is_null(), "{name}: a declared reference action is reported");
        }
    }
}

/// Writes the recorded splits the reference script needs (the second stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit_splits() {
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("POLICY_REFERENCE_DATA").unwrap()).unwrap(),
    )
    .unwrap();
    let mut out = serde_json::Map::new();
    for case in data["cases"].as_array().unwrap() {
        let n = case["y"].as_array().unwrap().len();
        let estimate = match try_run(case) {
            Ok(estimate) => estimate,
            Err(error) => {
                eprintln!("CASE {} FAILS: {error}", case["name"]);
                continue;
            }
        };
        let order: Vec<usize> = estimate.row_index.iter().map(|id| id.get() as usize).collect();
        let splits: Vec<Value> = blocks(&order)
            .into_iter()
            .map(|valid| {
                let train: Vec<usize> = (0..n).filter(|i| !valid.contains(i)).collect();
                serde_json::json!({ "train": train, "valid": valid })
            })
            .collect();
        out.insert(case["name"].as_str().unwrap().to_owned(), Value::Array(splits));
    }
    std::fs::write(
        std::env::var("POLICY_REFERENCE_OUT").unwrap(),
        serde_json::to_string(&Value::Object(out)).unwrap(),
    )
    .unwrap();
}
