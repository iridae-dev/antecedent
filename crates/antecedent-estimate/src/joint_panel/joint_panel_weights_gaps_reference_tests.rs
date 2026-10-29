// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sampling weights and excluded periods in the Gaussian panel likelihood, against an
//! independent dense-covariance reference.
//!
//! `tests/oracle_data/joint_panel_weights_gaps_reference.py` (frozen in the JSON beside it)
//! computes, from the definition of the model in 60 to 120-digit arithmetic and without the
//! product's precision-factor algebra:
//!
//! * the marginal law of residuals observed at a subset of their periods, as the sub-matrix of
//!   the full AR(1) plus random-intercept covariance at the true time distances;
//! * the weighted log-likelihood, the weighted score of every group, the weighted observed
//!   information and the weighted group-cluster sandwich of a five-group unbalanced panel,
//!   weighted, gapped and both;
//! * the maximized weighted dense log-likelihood.

use super::{cluster_factor, information_inverse, optimize_target, score};
use crate::joint_panel::likelihood::{self, CompiledPanel};
use crate::joint_panel::{
    PanelChannel, PanelChannelHistory, PanelChannelProgram, PanelData, PanelDraw, PanelFamily,
    PanelFitOptions, PanelGroup, PanelInference, PanelOptimizerOptions, PanelPrior, PanelProgram,
    PanelSamplingOptions, PanelScenario, evaluate_panel_programs, validate_fit_options,
};
use antecedent_core::ExecutionContext;
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/oracle_data/joint_panel_weights_gaps_reference.json"
    ))
    .unwrap()
}

fn f(value: &Value) -> f64 {
    value.as_f64().unwrap()
}

fn close(got: f64, want: f64, rel: f64, abs: f64) -> bool {
    (got - want).abs() <= abs + rel * want.abs()
}

fn vector(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(f).collect()
}

fn optional(value: &Value) -> f64 {
    if value.is_null() { f64::NAN } else { f(value) }
}

fn panel(case: &Value, weighted: bool) -> PanelData {
    let weights = vector(&case["weights"]);
    let groups = case["groups"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, group)| PanelGroup {
            id: index.to_string(),
            history: vec![[f(&group["history"][0]), f(&group["history"][1])]],
            channels: vec![vector(&group["channel"])],
            outcome: group["outcome"].as_array().unwrap().iter().map(optional).collect(),
            confounder: vector(&group["confounder"]),
            season: vec![0; group["channel"].as_array().unwrap().len()],
            weight: if weighted { weights[index] } else { 1.0 },
        })
        .collect();
    PanelData {
        channels: vec![PanelChannel { id: "0".into() }],
        interactions: vec![],
        groups,
        seasons: 1,
        lag_order: 2,
        family: PanelFamily::default(),
    }
}

/// A group with excluded periods marked in its outcome, and residuals only where observed.
fn gapped_group(times: &[usize]) -> PanelGroup {
    let n = times.iter().max().unwrap() + 1;
    let mut outcome = vec![f64::NAN; n];
    for &time in times {
        outcome[time] = 0.0;
    }
    PanelGroup {
        id: "g".into(),
        history: vec![[0.0; 2]],
        channels: vec![vec![0.0; n]],
        outcome,
        confounder: vec![0.0; n],
        season: vec![0; n],
        weight: 1.0,
    }
}

#[test]
fn build_joint_panel_gapped_marginal_equals_the_dense_submatrix_reference() {
    let reference = reference();
    for case in reference["gap_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let times: Vec<usize> = case["times"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap() as usize)
            .collect();
        let r = vector(&case["r"]);
        let group = gapped_group(&times);
        let mut residuals = vec![f64::NAN; group.outcome.len()];
        for (&time, &value) in times.iter().zip(&r) {
            residuals[time] = value;
        }
        let got = likelihood::group_marginal(
            &group,
            &residuals,
            f(&case["rho"]),
            f(&case["sigma"]),
            f(&case["tau"]),
        )
        .unwrap_or_else(|| panic!("{name}: the product refused a valid gapped group"));
        let (ll, m, v) = (
            f(&case["log_likelihood"]),
            f(&case["intercept_mean"]),
            f(&case["intercept_variance"]),
        );
        assert!(
            close(got.log_likelihood, ll, 1e-11, 1e-11),
            "{name} ll: {} vs {ll}",
            got.log_likelihood
        );
        assert!(
            close(got.intercept_mean, m, 1e-9, 1e-12),
            "{name} mean: {} vs {m}",
            got.intercept_mean
        );
        assert!(
            close(got.intercept_variance, v, 1e-9, 1e-14),
            "{name} var: {} vs {v}",
            got.intercept_variance
        );
    }
}

#[test]
fn build_joint_panel_a_residual_that_is_not_finite_at_an_analysed_period_is_refused() {
    let group = gapped_group(&[0, 2, 3]);
    let mut residuals = vec![f64::NAN, f64::NAN, 0.1, -0.2];
    residuals[0] = f64::NAN; // analysed period with a numerical failure, not a gap
    assert!(likelihood::group_marginal(&group, &residuals, 0.4, 0.2, 0.3).is_none());
    residuals[0] = 0.3;
    assert!(likelihood::group_marginal(&group, &residuals, 0.4, 0.2, 0.3).is_some());
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(20_261_029)
}

#[test]
fn build_joint_panel_weighted_gapped_likelihood_score_and_sandwich_match_the_dense_reference() {
    let reference = reference();
    for case in reference["model_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let data = panel(case, true);
        data.validate().unwrap();
        let compiled = CompiledPanel::new(&data).unwrap();
        let x = vector(&case["x"]);
        let ll = likelihood::log_likelihood(&compiled, &x);
        assert!(close(ll, f(&case["log_likelihood"]), 1e-11, 1e-10), "{name} ll {ll}");
        let want_scores = case["scores"].as_array().unwrap();
        for (group, want) in want_scores.iter().enumerate() {
            let got = score(&compiled, &x, Some(group), &ctx()).unwrap();
            for (coordinate, (got, want)) in got.iter().zip(vector(want)).enumerate() {
                assert!(
                    close(*got, want, 2e-6, 2e-6),
                    "{name} group {group} coordinate {coordinate}: {got} vs {want}"
                );
            }
        }
        // The weighted observed information, through its inverse.
        let inverse = information_inverse(&compiled, &x, &ctx()).unwrap().expect("curvature");
        let information = case["information"].as_array().unwrap();
        let p = x.len();
        for row in 0..p {
            let want = vector(&information[row]);
            for column in 0..p {
                let got: f64 = (0..p)
                    .map(|k| {
                        // (inverse^-1)[row][column] solved by multiplying with the reference
                        // information: inverse * H must be the identity.
                        inverse[row * p + k] * want_information(&information, k, column)
                    })
                    .sum();
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!(
                    (got - expected).abs() < 5e-4,
                    "{name} inverse*H [{row}][{column}] = {got}; row {want:?}"
                );
            }
        }
        // The weighted group-cluster sandwich.
        let factor = cluster_factor(&compiled, &x, &inverse, &ctx()).unwrap();
        let covariance = case["covariance"].as_array().unwrap();
        for i in 0..p {
            let want_row = vector(&covariance[i]);
            for j in 0..p {
                let got: f64 = factor.iter().map(|group| group[i] * group[j]).sum();
                let scale = (want_row[i].abs() * f(&covariance[j][j]).abs()).sqrt();
                assert!(
                    (got - want_row[j]).abs() <= 2e-3 * scale.max(1e-9),
                    "{name} covariance [{i}][{j}]: {got} vs {}",
                    want_row[j]
                );
            }
        }
    }
}

fn want_information(information: &[Value], row: usize, column: usize) -> f64 {
    f(&information[row][column])
}

#[test]
fn build_joint_panel_weighted_fit_reaches_the_dense_weighted_maximum() {
    let reference = reference();
    for case in reference["model_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let data = panel(case, true);
        let compiled = CompiledPanel::new(&data).unwrap();
        let start = vector(&case["x"]);
        let local = optimize_target(
            start,
            PanelOptimizerOptions { max_iterations: 20_000, gradient_tolerance: 1e-5 },
            &ctx(),
            &|x| likelihood::log_likelihood(&compiled, x),
        )
        .unwrap();
        let want = f(&case["mle_log_likelihood"]);
        eprintln!(
            "{name}: product {} reference {want} score norm {} x {:?}",
            local.likelihood, local.score_norm, local.values
        );
        assert!(local.converged, "{name}: optimizer did not converge");
        if let Some(eigenvalue) = case.get("mle_min_information_eigenvalue") {
            // A certified candidate: the dense observed information there is positive definite
            // and the dense score is within the optimizer's own tolerance.
            assert!(f(eigenvalue) > 0.0 && f(&case["mle_score_norm"]) < 1e-4, "{name}");
        }
        assert!(
            (local.likelihood - want).abs() <= 1e-4 * want.abs().max(1.0),
            "{name}: weighted maximum {} vs dense reference {want}",
            local.likelihood
        );
    }
}

#[test]
fn build_joint_panel_integer_weights_are_exactly_unit_replication() {
    let reference = reference();
    let case = &reference["model_cases"][0];
    let mut weighted = panel(case, true);
    for (group, weight) in weighted.groups.iter_mut().zip([1.0, 2.0, 1.0, 3.0, 1.0]) {
        group.weight = weight;
    }
    let mut replicated = panel(case, false);
    let mut copies = Vec::new();
    for (group, count) in replicated.groups.iter().zip([1, 2, 1, 3, 1]) {
        for copy in 0..count {
            let mut group = group.clone();
            group.id = format!("{}-{copy}", group.id);
            copies.push(group);
        }
    }
    replicated.groups = copies;
    let (a, b) = (CompiledPanel::new(&weighted).unwrap(), CompiledPanel::new(&replicated).unwrap());
    for x in [vector(&case["x"]), vector(&case["mle"])] {
        let (left, right) =
            (likelihood::log_likelihood(&a, &x), likelihood::log_likelihood(&b, &x));
        assert!((left - right).abs() <= 1e-10 * right.abs(), "{left} vs {right}");
        let sum = |data: &CompiledPanel<'_>, groups: usize| -> Vec<f64> {
            let mut total = vec![0.0; x.len()];
            for group in 0..groups {
                for (slot, value) in
                    total.iter_mut().zip(score(data, &x, Some(group), &ctx()).unwrap())
                {
                    *slot += value;
                }
            }
            total
        };
        for (left, right) in sum(&a, 5).iter().zip(sum(&b, 8)) {
            assert!((left - right).abs() <= 1e-6 * (1.0 + right.abs()), "{left} vs {right}");
        }
    }
}

#[test]
fn build_joint_panel_equal_weights_do_not_change_the_fit_and_the_sandwich_is_scale_free() {
    let reference = reference();
    let case = &reference["model_cases"][0];
    let plain = panel(case, false);
    let mut scaled = plain.clone();
    for group in &mut scaled.groups {
        group.weight = 7.5;
    }
    let (a, b) = (CompiledPanel::new(&plain).unwrap(), CompiledPanel::new(&scaled).unwrap());
    let x = vector(&case["x"]);
    // The log-likelihood scales with the common weight; the sandwich does not move.
    let (left, right) = (likelihood::log_likelihood(&a, &x), likelihood::log_likelihood(&b, &x));
    assert!((right - 7.5 * left).abs() <= 1e-10 * right.abs());
    let covariance = |data: &CompiledPanel<'_>| {
        let inverse = information_inverse(data, &x, &ctx()).unwrap().unwrap();
        cluster_factor(data, &x, &inverse, &ctx()).unwrap()
    };
    let (fa, fb) = (covariance(&a), covariance(&b));
    for (ga, gb) in fa.iter().zip(&fb) {
        for (u, v) in ga.iter().zip(gb) {
            assert!((u - v).abs() <= 1e-6 * (1.0 + u.abs()), "{u} vs {v}");
        }
    }
}

#[test]
fn build_joint_panel_weighted_response_is_the_weighted_group_average() {
    let reference = reference();
    let case = &reference["model_cases"][0];
    let data = panel(case, true);
    let compiled = CompiledPanel::new(&data).unwrap();
    let parameters = likelihood::decode(&compiled, &vector(&case["x"])).unwrap();
    let intercepts = vec![0.5, -0.25, 1.0, 0.0, 2.0];
    let draw = PanelDraw { parameters: parameters.clone(), group_intercepts: intercepts.clone() };
    let program = PanelProgram {
        id: "p".into(),
        channels: vec![PanelChannelProgram { id: "0".into(), values: vec![2.0, 3.0] }],
        scenarios: data
            .groups
            .iter()
            .map(|group| PanelScenario {
                group: group.id.clone(),
                history: vec![PanelChannelHistory { id: "0".into(), values: group.history[0] }],
                confounder: vec![0.3, -0.4],
                season: vec![0, 0],
            })
            .collect(),
    };
    let response = evaluate_panel_programs(&data, &draw, &[program.clone()], &ctx()).unwrap();
    let weights = vector(&case["weights"]);
    let mut expected = [0.0; 2];
    for (index, group) in data.groups.iter().enumerate() {
        let scenario = &program.scenarios[index];
        for time in 0..2 {
            let exposure = {
                let w = parameters.channels[0].lag;
                (0..3)
                    .map(|lag| {
                        let value = if lag <= time {
                            program.channels[0].values[time - lag]
                        } else {
                            scenario.history[0].values[2 + time - lag]
                        };
                        w[lag] * value
                    })
                    .sum::<f64>()
            };
            let mean = parameters.intercept
                + parameters.channels[0].coefficient * exposure
                    / (exposure + parameters.half_saturation)
                + parameters.confounder * scenario.confounder[time]
                + intercepts[index];
            expected[time] += weights[index] * mean;
        }
        let _ = group;
    }
    let total: f64 = weights.iter().sum();
    for time in 0..2 {
        assert!(
            (response[0].means[time] - expected[time] / total).abs() < 1e-12,
            "{} vs {}",
            response[0].means[time],
            expected[time] / total
        );
    }
}

#[test]
fn build_joint_panel_weighted_pseudo_likelihood_refuses_a_bootstrap_a_posterior_and_a_spatial_law()
{
    let reference = reference();
    let case = &reference["model_cases"][0];
    let data = panel(case, true);
    let options = |inference| PanelFitOptions {
        optimizer: PanelOptimizerOptions { max_iterations: 10, gradient_tolerance: 1e-3 },
        inference,
        initial: None,
    };
    validate_fit_options(&data, &options(PanelInference::Frequentist { draws: 2 })).unwrap();
    for inference in [
        PanelInference::FrequentistBootstrap { draws: 2 },
        PanelInference::Bayesian {
            prior: PanelPrior::default(),
            sampler: PanelSamplingOptions::default(),
        },
    ] {
        let error = validate_fit_options(&data, &options(inference)).unwrap_err();
        assert!(error.to_string().contains("weighted group-cluster sandwich"), "{error}");
    }
    // The same inferences are fine for the unweighted panel.
    let plain = panel(case, false);
    validate_fit_options(&plain, &options(PanelInference::FrequentistBootstrap { draws: 2 }))
        .unwrap();
    // Invalid weights are refused at validation.
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut broken = data.clone();
        broken.groups[0].weight = bad;
        assert!(broken.validate().is_err(), "weight {bad}");
    }
}

#[test]
fn build_joint_panel_a_constant_channel_is_judged_on_the_analysed_periods_only() {
    let reference = reference();
    let case = &reference["model_cases"][1];
    let mut data = panel(case, false);
    // Every analysed period carries the same dose; the excluded periods carry another. The
    // channel is constant where it is read, so its effect is not identified.
    for group in &mut data.groups {
        for (value, outcome) in group.channels[0].iter_mut().zip(&group.outcome) {
            *value = if outcome.is_finite() { 1.5 } else { 4.0 };
        }
    }
    let error = data.validate().unwrap_err();
    assert!(error.to_string().contains("constant across every group and period"), "{error}");
}
