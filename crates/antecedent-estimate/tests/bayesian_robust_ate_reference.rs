//! The Bayesian-bootstrap cross-fitted orthogonal ATE against a 50-digit reference.
//!
//! The route draws one row-weight vector per posterior draw from a seeded stream and fixes the
//! cross-fit folds from the stable row ids and the treatment arm. Those two are inputs of the
//! reference, frozen in `oracle_data/bayesian_robust_ate_reference.json`: for the first three
//! draws of each case, `oracle_data/bayesian_robust_ate_reference.py` takes the weight vector and
//! the fold ids and recomputes the draw from the definitions in multiprecision arithmetic: per
//! fold, the weighted ridge-logistic propensity by Newton's method (intercept unpenalised,
//! `lambda/2` on the rest), the two per-arm weighted ridge outcome regressions, the clip, the
//! orthogonal score and its weighted mean. The test regenerates the weights from the product's
//! stream and checks they are the frozen ones, then compares each product draw with the reference
//! draw. The cases cover 12 to 60 rows, 2 to 5 folds, 20 percent treated, a covariate offset by
//! 1000, two covariates within 1e-5 of each other, a clip that binds on a strongly confounded
//! propensity and a ridge penalty of 0.01.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};
use antecedent_estimate::bayesian_robust_ate::{
    BayesianRobustAteInput, BayesianRobustAteOptions, estimate_bayesian_robust_ate,
    plan_bayesian_robust_ate_folds,
};
use serde_json::Value;

const SEED: u64 = 17;
const REFERENCE_DRAWS: usize = 3;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

fn input(case: &Value) -> BayesianRobustAteInput {
    BayesianRobustAteInput {
        row_ids: case["row_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect(),
        treatment: case["treatment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect(),
        outcome: floats(&case["outcome"]),
        covariates: case["covariates"].as_array().unwrap().iter().map(floats).collect(),
    }
}

fn options(case: &Value, draws: usize) -> BayesianRobustAteOptions {
    BayesianRobustAteOptions {
        draws,
        folds: usize::try_from(case["folds"].as_u64().unwrap()).unwrap(),
        ridge_penalty: case["ridge"].as_f64().unwrap(),
        propensity_clip: case["clip"].as_f64().unwrap(),
        coverage: 0.95,
    }
}

fn weights(draw: usize, n: usize) -> Vec<f64> {
    let ctx = ExecutionContext::for_tests(SEED);
    let mut rng = ctx.rng.stream_for(StreamDomain::Bayesian, 0xB007_0000 | draw as u64);
    antecedent_kernels::bayesian_bootstrap_weights(&mut rng, n, true)
}

/// Regenerates the frozen stream inputs: reads the case list the reference script prints with
/// `inputs` from `ROBUST_ATE_INPUTS` and writes the fold ids and weights to `ROBUST_ATE_STREAM`.
#[test]
#[ignore = "regenerates the frozen weights and folds of the reference; run by hand"]
fn dump_stream_inputs() {
    let path = std::env::var("ROBUST_ATE_INPUTS").expect("ROBUST_ATE_INPUTS");
    let out = std::env::var("ROBUST_ATE_STREAM").expect("ROBUST_ATE_STREAM");
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let dumped: Vec<Value> = cases
        .iter()
        .map(|case| {
            let data = input(case);
            let n = data.row_ids.len();
            let folds = plan_bayesian_robust_ate_folds(&data, options(case, 2).folds).unwrap();
            serde_json::json!({
                "fold_ids": folds,
                "weights": (0..REFERENCE_DRAWS).map(|d| weights(d, n)).collect::<Vec<_>>(),
            })
        })
        .collect();
    std::fs::write(out, serde_json::to_string(&dumped).unwrap()).unwrap();
}

fn reference() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/bayesian_robust_ate_reference.json"))
        .expect("frozen reference parses")
        .as_array()
        .cloned()
        .expect("array")
}

#[test]
fn each_draw_equals_the_multiprecision_weighted_cross_fitted_score() {
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let data = input(&case);
        let n = data.row_ids.len();
        // The frozen weights are the product's stream, and the folds are the product's plan.
        let frozen = case["weights"].as_array().unwrap();
        for d in 0..REFERENCE_DRAWS {
            for (got, want) in weights(d, n).iter().zip(floats(&frozen[d])) {
                assert!((got - want).abs() <= 1e-12 * want.abs().max(1e-300), "{name} weights {d}");
            }
        }
        let folds: Vec<u64> =
            case["fold_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let planned = plan_bayesian_robust_ate_folds(&data, options(&case, 2).folds).unwrap();
        assert_eq!(planned.iter().map(|f| u64::from(*f)).collect::<Vec<_>>(), folds, "{name}");
        // Every fold holds both treatment arms, which the nuisance fits need.
        for fold in 0..case["folds"].as_u64().unwrap() {
            let held: Vec<usize> = (0..n).filter(|i| folds[*i] == fold).collect();
            assert!(
                held.iter().any(|i| data.treatment[*i]) && held.iter().any(|i| !data.treatment[*i]),
                "{name}"
            );
        }
        let ctx = ExecutionContext::for_tests(SEED);
        if case["refused"].as_bool().unwrap() {
            let error = estimate_bayesian_robust_ate(&data, options(&case, REFERENCE_DRAWS), &ctx)
                .expect_err(name);
            assert!(error.to_string().contains("too few training rows"), "{name}: {error}");
            continue;
        }
        let result = estimate_bayesian_robust_ate(&data, options(&case, REFERENCE_DRAWS), &ctx)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for (d, want) in floats(&case["draws"]).iter().enumerate() {
            let got = result.draws[d];
            // The product's logistic engine stops at a score norm near 1e-8.
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "{name} draw {d}: {got} vs {want}"
            );
        }
        // Every refit of a draw records that draw's weight fingerprint, once per role and fold.
        for audit in &result.refits {
            assert_eq!(
                audit.len(),
                3 * usize::try_from(case["folds"].as_u64().unwrap()).unwrap(),
                "{name}"
            );
            assert!(
                audit.iter().all(|r| r.weight_fingerprint == audit[0].weight_fingerprint),
                "{name}"
            );
        }
    }
}

#[test]
fn declared_options_are_used_or_refused() {
    let case = reference().into_iter().find(|c| c["name"] == "moderate").unwrap();
    let data = input(&case);
    let ctx = ExecutionContext::for_tests(SEED);
    let base = estimate_bayesian_robust_ate(&data, options(&case, 8), &ctx).unwrap();
    // The ridge penalty, the clip and the fold count each change the draws.
    for changed in [
        BayesianRobustAteOptions { ridge_penalty: 50.0, ..options(&case, 8) },
        BayesianRobustAteOptions { propensity_clip: 0.3, ..options(&case, 8) },
        BayesianRobustAteOptions { folds: 3, ..options(&case, 8) },
    ] {
        let other = estimate_bayesian_robust_ate(&data, changed, &ctx).unwrap();
        assert_ne!(other.draws, base.draws, "{changed:?}");
    }
    // The coverage moves the interval and not the draws.
    let wide = estimate_bayesian_robust_ate(
        &data,
        BayesianRobustAteOptions { coverage: 0.5, ..options(&case, 8) },
        &ctx,
    )
    .unwrap();
    assert_eq!(wide.draws, base.draws);
    assert!(wide.interval.1 - wide.interval.0 < base.interval.1 - base.interval.0);
    // A reordering of the rows leaves the plan and every draw's fold membership by stable row id.
    let n = data.row_ids.len();
    let order: Vec<usize> = (0..n).rev().collect();
    let reversed = BayesianRobustAteInput {
        row_ids: order.iter().map(|i| data.row_ids[*i]).collect(),
        treatment: order.iter().map(|i| data.treatment[*i]).collect(),
        outcome: order.iter().map(|i| data.outcome[*i]).collect(),
        covariates: data.covariates.iter().map(|c| order.iter().map(|i| c[*i]).collect()).collect(),
    };
    let original = plan_bayesian_robust_ate_folds(&data, 4).unwrap();
    let permuted = plan_bayesian_robust_ate_folds(&reversed, 4).unwrap();
    for (position, source) in order.iter().enumerate() {
        assert_eq!(permuted[position], original[*source]);
    }
    // One treatment arm, a repeated row id, a non-finite outcome and a bad coverage are refused.
    let mut one_arm = data.clone();
    one_arm.treatment = vec![true; n];
    assert!(estimate_bayesian_robust_ate(&one_arm, options(&case, 8), &ctx).is_err());
    let mut repeated = data.clone();
    repeated.row_ids[1] = repeated.row_ids[0];
    assert!(estimate_bayesian_robust_ate(&repeated, options(&case, 8), &ctx).is_err());
    let mut nan = data.clone();
    nan.outcome[0] = f64::NAN;
    assert!(estimate_bayesian_robust_ate(&nan, options(&case, 8), &ctx).is_err());
    for bad in [0.0, 1.0, f64::NAN] {
        assert!(
            estimate_bayesian_robust_ate(
                &data,
                BayesianRobustAteOptions { coverage: bad, ..options(&case, 8) },
                &ctx
            )
            .is_err()
        );
    }
}
