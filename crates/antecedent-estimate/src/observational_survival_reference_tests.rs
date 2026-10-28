//! Cross-fitted doubly robust survival effects against a 40-digit reference with recorded folds.
//!
//! `tests/oracle_data/observational_survival_reference.py` (frozen in the JSON beside it) takes the
//! data and the fold of every row as part of the case, fits the logistic propensity and the
//! per-arm Cox event and censoring models (Breslow ties) by Newton's method on each fold's
//! training rows, and evaluates the augmented inverse-probability and censoring-weighted scores of
//! the survival probability and the restricted mean survival time by their step-function
//! definitions. This test asserts that the product's seeded stratified plan equals the recorded
//! folds, so a changed plan fails loudly, and that every arm summary, contrast, standard error,
//! support diagnostic and per-row score equals the reference. Cases cover independent and
//! at-random censoring, ties between events and censoring, a covariate offset of 50 and a horizon
//! before most events.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use serde_json::Value;

use super::*;

const SEED: u64 = 9;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases(text: &str) -> Vec<Value> {
    serde_json::from_str::<Value>(text).unwrap()["cases"].as_array().unwrap().clone()
}

fn plan_folds(case: &Value) -> Vec<u32> {
    let treatment = floats(&case["treatment"]);
    let active = case["active"].as_f64().unwrap();
    let arm: Vec<u32> = treatment.iter().map(|&t| u32::from(t == active)).collect();
    let units: Vec<antecedent_data::GlobalId> =
        (0..arm.len() as u64).map(antecedent_data::GlobalId::new).collect();
    let ctx = ExecutionContext::for_tests(SEED);
    CrossfitPlan::derive(
        arm.len(),
        case["n_folds"].as_u64().unwrap() as usize,
        FoldSource::Stratified { strata: &arm, units: &units, seed: ctx.rng.master_seed() },
        &SHARED_FOLD_ID_ERRORS,
    )
    .unwrap()
    .fold_ids()
    .to_vec()
}

fn run(case: &Value) -> ObservationalSurvivalEstimate {
    let censoring = match case["censoring"].as_str().unwrap() {
        "independent_within_arm" => CensoringAssumption::IndependentWithinArm,
        _ => CensoringAssumption::AtRandomGivenCovariates,
    };
    let mut request = ObservationalSurvival::new(
        case["active"].as_f64().unwrap(),
        case["control"].as_f64().unwrap(),
        case["horizon"].as_f64().unwrap(),
        censoring,
    );
    request.folds = case["n_folds"].as_u64().unwrap() as usize;
    let covariates: Vec<Vec<f64>> =
        case["covariates"].as_array().unwrap().iter().map(floats).collect();
    let columns: Vec<&[f64]> = covariates.iter().map(Vec::as_slice).collect();
    request
        .estimate(
            &SurvivalData {
                duration: &floats(&case["duration"]),
                event: &floats(&case["event"]),
                treatment: &floats(&case["treatment"]),
                covariates: &columns,
            },
            &ExecutionContext::for_tests(SEED),
        )
        .unwrap()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-9), "{what}: got {got} want {want}");
}

#[test]
fn arm_summaries_contrasts_and_scores_match_the_recorded_fold_reference() {
    for case in cases(include_str!("../tests/oracle_data/observational_survival_reference.json")) {
        let name = case["name"].as_str().unwrap();
        let folds: Vec<u32> =
            case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(plan_folds(&case), folds, "{name}: the product's plan is the recorded one");
        let want = &case["expected"];
        let got = run(&case);
        let rel = 1e-6;
        for (a, arm) in [&got.control, &got.active].into_iter().enumerate() {
            let w = &want["arms"][a];
            close(arm.survival, w["survival"].as_f64().unwrap(), rel, &format!("{name} S{a}"));
            close(arm.survival_se, w["survival_se"].as_f64().unwrap(), rel, name);
            close(arm.rmst, w["rmst"].as_f64().unwrap(), rel, &format!("{name} R{a}"));
            close(arm.rmst_se, w["rmst_se"].as_f64().unwrap(), rel, name);
            close(arm.propensity_min, w["propensity_min"].as_f64().unwrap(), rel, name);
            close(
                arm.censoring_survival_min,
                w["censoring_survival_min"].as_f64().unwrap(),
                rel,
                name,
            );
        }
        close(
            got.survival_difference.estimate,
            want["survival_difference"]["estimate"].as_f64().unwrap(),
            rel,
            name,
        );
        close(
            got.survival_difference.se,
            want["survival_difference"]["se"].as_f64().unwrap(),
            rel,
            name,
        );
        close(
            got.rmst_difference.estimate,
            want["rmst_difference"]["estimate"].as_f64().unwrap(),
            rel,
            name,
        );
        close(got.rmst_difference.se, want["rmst_difference"]["se"].as_f64().unwrap(), rel, name);
        for (r, (g, w)) in got
            .survival_difference_scores
            .iter()
            .zip(floats(&want["survival_difference_scores"]))
            .enumerate()
        {
            close(*g, w, rel, &format!("{name} survival score {r}"));
        }
        for (r, (g, w)) in got
            .rmst_difference_scores
            .iter()
            .zip(floats(&want["rmst_difference_scores"]))
            .enumerate()
        {
            close(*g, w, rel, &format!("{name} rmst score {r}"));
        }
    }
}

/// Writes the plan folds the reference script needs (the second stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit_folds() {
    let text = std::fs::read_to_string(std::env::var("SURVIVAL_REFERENCE_DATA").unwrap()).unwrap();
    let mut out = serde_json::Map::new();
    for case in cases(&text) {
        out.insert(case["name"].as_str().unwrap().to_owned(), serde_json::json!(plan_folds(&case)));
    }
    std::fs::write(
        std::env::var("SURVIVAL_REFERENCE_OUT").unwrap(),
        serde_json::to_string(&Value::Object(out)).unwrap(),
    )
    .unwrap();
}
