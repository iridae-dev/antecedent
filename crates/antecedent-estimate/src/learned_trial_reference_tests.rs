//! Cross-fitted trial-to-nonparticipant AIPW and its joint outer bootstrap against a 40-digit
//! reference with recorded folds and recorded replicate draws.
//!
//! `tests/oracle_data/learned_trial_reference.py` (frozen in the JSON beside it) takes the data, the
//! fold of every row and the row indices of every bootstrap replicate as part of the case. It fits
//! the unpenalised logistic membership model and the per-arm ridge outcome models on each fold's
//! training rows, forms the AIPW terms `sign (1 - s) / (s arm) (y - mu)` on trial rows and
//! `mu1 - mu0` on target rows, refits the whole procedure on every recorded resample (each
//! duplicate keeping its original row's fold) and takes the type-7 equal-tail percentile
//! interval. This test asserts that the product's fold assignment and resampling stream equal the
//! recorded ones (draws stay inside their group and keep its size), and that the estimate, every
//! per-row score and out-of-fold nuisance, every replicate estimate and the interval equal the
//! reference, for a nested cohort and for independent samples at a covariate offset of 100.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_identify::{PopulationFactor, TransportCertificate};
use serde_json::Value;
use std::sync::Arc;

use super::*;
use crate::{LogisticSpec, RidgeSpec};

const SEED: u64 = 31;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn certificate(features: &[u32]) -> TransportIdentification {
    let factor = || PopulationFactor {
        population: Arc::from("target"),
        regime: None,
        variables: Arc::from([]),
        conditioned_on: Arc::from([]),
        interventions: Arc::from([]),
    };
    TransportIdentification::Transportable {
        formula: TransportFormula::Standardize {
            over: features.iter().map(|&f| VariableId::from_raw(f)).collect(),
            source_response: factor(),
            target_law: factor(),
        },
        certificate: TransportCertificate {
            rule: Arc::from("test"),
            selection_targets: Arc::from([]),
            premises: Arc::from([]),
        },
    }
}

fn input(case: &Value) -> TrialAipwInput {
    TrialAipwInput {
        features: case["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect(),
        covariates: case["covariates"].as_array().unwrap().iter().map(floats).collect(),
        outcome: floats(&case["outcome"]),
        treatment: bools(&case["treatment"]),
        source: bools(&case["source"]),
        randomization: floats(&case["randomization"]),
        sampling: if case["sampling"] == "nested_cohort" {
            TrialSampling::NestedCohort
        } else {
            TrialSampling::IndependentSamples
        },
    }
}

fn options(case: &Value) -> TrialAipwOptions {
    TrialAipwOptions {
        outcome: LearnerSpec::Ridge(RidgeSpec { lambda: case["ridge_lambda"].as_f64().unwrap() }),
        membership: LearnerSpec::Logistic(LogisticSpec { ridge_lambda: 0.0 }),
        folds: case["n_folds"].as_u64().unwrap() as usize,
        bootstrap: case["bootstrap"].as_u64().unwrap() as u32,
        coverage_level: case["coverage_level"].as_f64().unwrap(),
    }
}

fn plan(case: &Value) -> (Vec<u16>, Vec<Vec<usize>>) {
    let input = input(case);
    let ctx = ExecutionContext::for_tests(SEED);
    let folds = trial_fold_assignment(&input, case["n_folds"].as_u64().unwrap() as usize);
    let groups = bootstrap_groups(&input);
    let draws = (0..case["bootstrap"].as_u64().unwrap() as u32)
        .map(|r| bootstrap_rows(&groups, r, &ctx).unwrap())
        .collect();
    (folds, draws)
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-6), "{what}: got {got} want {want}");
}

#[test]
fn estimate_scores_nuisances_replicates_and_interval_match_the_recorded_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("../tests/oracle_data/learned_trial_reference.json"))
            .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let want = &case["expected"];
        let (folds, draws) = plan(case);
        let recorded_folds: Vec<u16> =
            case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u16).collect();
        assert_eq!(folds, recorded_folds, "{name}: the product's fold plan is the recorded one");
        let recorded_rows: Vec<Vec<usize>> = case["replicate_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect())
            .collect();
        assert_eq!(
            draws, recorded_rows,
            "{name}: the product's resampling stream is the recorded one"
        );
        let data = input(case);
        for draw in &draws {
            assert_eq!(draw.len(), data.source.len(), "{name}: the sample size is kept");
            if data.sampling == TrialSampling::IndependentSamples {
                let trial = draw.iter().filter(|&&i| data.source[i]).count();
                assert_eq!(trial, data.source.iter().filter(|&&s| s).count(), "{name}");
            }
        }
        let got = estimate_trial_aipw(
            &certificate(&data.features),
            &data,
            &options(case),
            &ExecutionContext::for_tests(SEED),
        )
        .unwrap();
        let rel = 1e-6;
        close(got.estimate, want["estimate"].as_f64().unwrap(), rel, &format!("{name} estimate"));
        for (label, g, w) in [
            ("scores", &got.scores, floats(&want["scores"])),
            ("membership", &got.membership, floats(&want["membership"])),
            ("mu0", &got.mu0, floats(&want["mu0"])),
            ("mu1", &got.mu1, floats(&want["mu1"])),
        ] {
            assert_eq!(g.len(), w.len(), "{name} {label}");
            for (r, (a, b)) in g.iter().zip(&w).enumerate() {
                close(*a, *b, rel, &format!("{name} {label} {r}"));
            }
        }
        assert_eq!(got.failures, 0, "{name}");
        let replicates = floats(&want["replicates"]);
        assert_eq!(got.replicates.len(), replicates.len(), "{name}");
        for ((index, value), w) in got.replicates.iter().zip(&replicates) {
            close(*value, *w, rel, &format!("{name} replicate {index}"));
        }
        let (low, high) = got.interval.expect("every replicate fits");
        let interval = floats(&want["interval"]);
        close(low, interval[0], rel, &format!("{name} lower"));
        close(high, interval[1], rel, &format!("{name} upper"));
    }
}

/// Writes the folds and replicate draws the reference script needs (second stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit_plan() {
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("LEARNED_TRIAL_REFERENCE_DATA").unwrap()).unwrap(),
    )
    .unwrap();
    let mut out = serde_json::Map::new();
    for case in data["cases"].as_array().unwrap() {
        let (folds, draws) = plan(case);
        out.insert(
            case["name"].as_str().unwrap().to_owned(),
            serde_json::json!({ "folds": folds, "replicate_rows": draws }),
        );
    }
    std::fs::write(
        std::env::var("LEARNED_TRIAL_REFERENCE_OUT").unwrap(),
        serde_json::to_string(&Value::Object(out)).unwrap(),
    )
    .unwrap();
}
