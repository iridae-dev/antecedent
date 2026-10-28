//! The Ljung-Box residual gate of the white-innovation posterior against an exact reference.
//!
//! `oracle_data/white_innovation_ljung_box_reference.py` (frozen in the JSON beside it) solves
//! the least-squares fit in 40-digit arithmetic, forms the residual autocorrelations
//! `r_k = sum_{t>k} e_t e_{t-k} / sum e_t^2`, the statistic `Q = n (n + 2) sum r_k^2 / (n - k)` and
//! its chi-square upper-tail probability on `m` degrees of freedom. The gate is the arithmetic
//! that licenses or refuses the route, so its statistic, p-value and refusal threshold are pinned
//! here; the invalid-gate refusals are pinned as well, the full route is run against the exact
//! Normal-Inverse-Gamma posterior of the treatment coefficient, and the refusal flips at the
//! reference p-value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, IdentificationStatus, TargetPopulation, VariableId};
use antecedent_data::materialize::PreparationReceipt;
use antecedent_estimate::white_innovation_posterior::{
    InnovationWhitenessGate, WhiteInnovationDeclaration, fit_white_innovation_posterior,
    innovation_whiteness_of,
};
use antecedent_estimate::{BayesianGComputationAte, OverlapPolicy, PreparedEstimationProblem};
use antecedent_stats::CompiledDesign;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!(
        "oracle_data/white_innovation_ljung_box_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone()
}

fn design(case: &Value) -> CompiledDesign {
    let covariate = floats(&case["covariate"]);
    CompiledDesign::linear_adjustment(
        &floats(&case["treatment"]),
        &[(VariableId::from_raw(0), covariate.as_slice())],
        &floats(&case["outcome"]),
        &[],
    )
    .expect("design")
}

#[test]
fn the_ljung_box_statistic_and_p_value_match_the_exact_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let lags = case["lags"].as_u64().unwrap() as usize;
        let report = innovation_whiteness_of(
            &design(&case),
            &InnovationWhitenessGate { max_lag: lags, level: 0.05 },
        )
        .unwrap();
        let (q, p) = (case["q"].as_f64().unwrap(), case["p_value"].as_f64().unwrap());
        assert_eq!(report.lags, lags, "{name}");
        assert_eq!(report.rows, case["outcome"].as_array().unwrap().len(), "{name}");
        assert!(
            (report.q_statistic / q - 1.0).abs() < 1e-9,
            "{name} Q {} vs {q}",
            report.q_statistic
        );
        assert!((report.p_value / p - 1.0).abs() < 1e-7, "{name} p {} vs {p}", report.p_value);
    }
}

#[test]
fn invalid_gates_and_too_few_rows_are_refused() {
    let case = cases().into_iter().find(|c| c["name"] == "single_lag").unwrap();
    let design = design(&case);
    for gate in [
        InnovationWhitenessGate { max_lag: 0, level: 0.05 },
        InnovationWhitenessGate { max_lag: 1, level: 0.0 },
        InnovationWhitenessGate { max_lag: 1, level: 1.0 },
        InnovationWhitenessGate { max_lag: 40, level: 0.05 },
    ] {
        assert!(innovation_whiteness_of(&design, &gate).is_err(), "{gate:?}");
    }
}

fn prepared(case: &Value) -> PreparedEstimationProblem {
    let treatment = floats(&case["treatment"]);
    PreparedEstimationProblem {
        design: design(case),
        method: Arc::from("temporal.linear.adjustment"),
        adjustment_set: Arc::from(vec![VariableId::from_raw(0)]),
        overlap: OverlapPolicy::ExplicitOverride,
        treatment_delta: 1.0,
        target_population: TargetPopulation::AllObserved,
        treatment: Arc::from(treatment.clone()),
        active: 1.0,
        control: 0.0,
        treatment_levels: None,
        sampling_weights: None,
        preparation: PreparationReceipt::composed_finite(treatment.len()).expect("row count"),
    }
}

fn declaration(case: &Value, level: f64) -> WhiteInnovationDeclaration {
    WhiteInnovationDeclaration {
        horizon_steps: 1,
        gate: InnovationWhitenessGate { max_lag: case["lags"].as_u64().unwrap() as usize, level },
    }
}

fn estimator() -> BayesianGComputationAte {
    BayesianGComputationAte::conjugate().with_n_draws(200_000).with_prior_scale(10.0).with_seed(5)
}

#[test]
fn the_gate_refuses_exactly_when_the_reference_p_value_is_below_the_level() {
    let case = cases().into_iter().find(|c| c["name"] == "four_lags_persistent_error").unwrap();
    let p = case["p_value"].as_f64().unwrap();
    let run = |level: f64| {
        fit_white_innovation_posterior(
            &estimator().with_n_draws(1000),
            &prepared(&case),
            &declaration(&case, level),
            IdentificationStatus::NonparametricallyIdentified,
            &ExecutionContext::for_tests(1),
        )
    };
    assert!(run(p * 0.99).is_ok(), "level just below the reference p-value passes");
    assert!(run(p * 1.01).is_err(), "level just above the reference p-value refuses");
}

#[test]
fn a_passing_gate_returns_the_untempered_conjugate_posterior_and_the_reference_statistic() {
    let case = cases().into_iter().find(|c| c["name"] == "single_lag").unwrap();
    let fit = fit_white_innovation_posterior(
        &estimator(),
        &prepared(&case),
        &declaration(&case, 0.05),
        IdentificationStatus::NonparametricallyIdentified,
        &ExecutionContext::for_tests(1),
    )
    .expect("gate passes");
    assert!((fit.whiteness.q_statistic / case["q"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    let column = fit.posterior.effect_column().unwrap();
    let (mean, sd) = (fit.posterior.summaries.mean[column], fit.posterior.summaries.sd[column]);
    let (r_mean, r_sd) =
        (case["posterior_mean"].as_f64().unwrap(), case["posterior_sd"].as_f64().unwrap());
    let root = (fit.posterior.draws.n_draws as f64).sqrt();
    assert!((mean - r_mean).abs() < 5.0 * r_sd / root, "mean {mean} vs {r_mean}");
    assert!((sd / r_sd - 1.0).abs() < 8.0 / root, "sd {sd} vs {r_sd}");
}

#[test]
fn a_multi_step_horizon_is_refused() {
    let case = cases().into_iter().find(|c| c["name"] == "single_lag").unwrap();
    let mut declared = declaration(&case, 0.05);
    declared.horizon_steps = 2;
    assert!(
        fit_white_innovation_posterior(
            &estimator(),
            &prepared(&case),
            &declared,
            IdentificationStatus::NonparametricallyIdentified,
            &ExecutionContext::for_tests(1),
        )
        .is_err()
    );
}
