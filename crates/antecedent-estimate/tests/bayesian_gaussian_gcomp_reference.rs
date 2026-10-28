//! Bayesian Gaussian g-computation against the exact Normal-Inverse-Gamma posterior.
//!
//! `oracle_data/bayesian_gaussian_gcomp_reference.py` (frozen in the JSON beside it) forms the
//! posterior of the treatment coefficient of an adjusted linear outcome model in 40-digit
//! arithmetic: the Student-t marginal under the weakly informative Inverse-Gamma residual prior,
//! or the Normal posterior under a declared residual variance. For a main-effects design the
//! g-computation effect `(active - control) * b_t` is that coefficient, so its posterior moments
//! and interval ends are the reference's. The default (Laplace-labelled) estimator is routed to
//! the exact conjugate posterior for an unknown variance, the conjugate estimator is the same
//! computation, and the temporal wrapper is the same fit on a lag-aligned design.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, IdentificationStatus, VariableId};
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp, OverlapPolicy,
    PreparedBayesianProblem, SerialDependence,
};
use antecedent_prob::{GaussianCoefficientPrior, PriorSet, PriorSpec};
use antecedent_stats::CompiledDesign;
use serde_json::Value;

const DRAWS: usize = 400_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!(
        "oracle_data/bayesian_gaussian_gcomp_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone()
}

fn problem(case: &Value) -> PreparedBayesianProblem {
    let treatment = floats(&case["treatment"]);
    let outcome = floats(&case["outcome"]);
    let columns: Vec<(VariableId, Vec<f64>)> = case["covariates"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, name)| (VariableId::from_raw(i as u32), floats(&case[name.as_str().unwrap()])))
        .collect();
    let covariates: Vec<(VariableId, &[f64])> =
        columns.iter().map(|(id, c)| (*id, c.as_slice())).collect();
    PreparedBayesianProblem {
        design: CompiledDesign::linear_adjustment(&treatment, &covariates, &outcome, &[])
            .expect("design"),
        method: Arc::from("backdoor.adjustment"),
        adjustment_set: Arc::from(Vec::new()),
        active: 1.0,
        control: 0.0,
        overlap: OverlapPolicy::ExplicitOverride,
        coef_names: None,
        unit_ids: None,
        serial_dependence: SerialDependence::Iid,
        preparation: None,
    }
}

fn estimator(case: &Value, problem: &PreparedBayesianProblem) -> BayesianGComputationAte {
    let scale = case["prior_scale"].as_f64().unwrap();
    let base = match case["backend"].as_str().unwrap() {
        "conjugate" => BayesianGComputationAte::conjugate(),
        _ => BayesianGComputationAte::new(),
    };
    let base = base.with_n_draws(DRAWS).with_seed(11).with_prior_scale(scale);
    match case["known_variance"].as_f64() {
        Some(variance) => base.with_prior(PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                    problem.design.ncols,
                    scale,
                )),
                PriorSpec::KnownResidualVariance(variance),
            ],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        }),
        None => base,
    }
}

fn effect_summary(posterior: &antecedent_estimate::CausalPosterior) -> (f64, f64, f64, f64, usize) {
    let column = posterior.effect_column().expect("effect column");
    (
        posterior.summaries.mean[column],
        posterior.summaries.sd[column],
        posterior.summaries.q025[column],
        posterior.summaries.q975[column],
        posterior.draws.n_draws,
    )
}

fn fit(
    estimator: &BayesianGComputationAte,
    problem: &PreparedBayesianProblem,
) -> antecedent_estimate::CausalPosterior {
    estimator
        .fit(
            problem,
            IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .expect("fit")
}

#[test]
fn the_effect_posterior_matches_the_exact_conjugate_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let problem = problem(&case);
        let posterior = fit(&estimator(&case, &problem), &problem);
        let (mean, sd, lower, upper, n_draws) = effect_summary(&posterior);
        let root = (n_draws as f64).sqrt();
        let (r_mean, r_sd) = (case["mean"].as_f64().unwrap(), case["sd"].as_f64().unwrap());
        assert!((mean - r_mean).abs() < 5.0 * r_sd / root, "{name} mean {mean} vs {r_mean}");
        assert!((sd / r_sd - 1.0).abs() < 8.0 / root, "{name} sd {sd} vs {r_sd}");
        assert!(
            (lower - case["lower"].as_f64().unwrap()).abs() < 0.03 * r_sd,
            "{name} lower {lower}"
        );
        assert!(
            (upper - case["upper"].as_f64().unwrap()).abs() < 0.03 * r_sd,
            "{name} upper {upper}"
        );
    }
}

#[test]
fn the_temporal_wrapper_is_the_same_fit_on_the_same_design() {
    let case = cases().into_iter().find(|c| c["name"] == "adjusted_tight_prior").unwrap();
    let problem = problem(&case);
    let inner = estimator(&case, &problem);
    let direct = effect_summary(&fit(&inner, &problem));
    let wrapped = BayesianTemporalGcomp::conjugate().with_inner(inner);
    let via = wrapped
        .fit(
            &problem,
            IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .expect("temporal fit");
    assert_eq!(effect_summary(&via), direct);
}
