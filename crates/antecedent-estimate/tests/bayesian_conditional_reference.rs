//! The Bayesian conditional-effect posterior against exact Normal-Inverse-Gamma moments.
//!
//! `oracle_data/bayesian_conditional_reference.py` (frozen in the JSON beside it) forms the exact
//! Student-t covariance of the coefficients of `y ~ 1 + t + w + t (w - wbar)` and the variance the
//! empirical-modifier-distribution uncertainty adds to the average contrast: each draw averages
//! the row contrasts with Dirichlet(1, ..., 1) weights, which adds `d eps` with `E[eps] = 0` and
//! `Var(eps) = var_w / (n + 1)`. The posterior mean of the reported effect must be the treatment
//! coefficient's and its standard deviation must include that extra variance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, IdentificationStatus, VariableId};
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, OverlapPolicy, PreparedBayesianProblem,
    SerialDependence,
};
use antecedent_stats::CompiledDesign;
use serde_json::Value;

const DRAWS: usize = 300_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/bayesian_conditional_reference.json"))
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn problem(case: &Value) -> PreparedBayesianProblem {
    let (w, t, y) =
        (floats(&case["modifier"]), floats(&case["treatment"]), floats(&case["outcome"]));
    let mean = w.iter().sum::<f64>() / w.len() as f64;
    let interaction: Vec<f64> = t.iter().zip(&w).map(|(t, w)| t * (w - mean)).collect();
    let design = CompiledDesign::linear_adjustment(
        &t,
        &[
            (VariableId::from_raw(0), w.as_slice()),
            (VariableId::from_raw(1), interaction.as_slice()),
        ],
        &y,
        &[],
    )
    .unwrap();
    let names: Vec<Arc<str>> =
        ["intercept", "treatment_at_mean_modifier", "modifier_0", "treatment_centered_modifier_0"]
            .into_iter()
            .map(Arc::from)
            .collect();
    PreparedBayesianProblem {
        design,
        method: Arc::from("backdoor.adjustment"),
        adjustment_set: Arc::from(Vec::new()),
        active: 1.0,
        control: 0.0,
        overlap: OverlapPolicy::ExplicitOverride,
        coef_names: Some(Arc::from(names)),
        unit_ids: None,
        serial_dependence: SerialDependence::Iid,
        preparation: None,
    }
}

#[test]
fn the_conditional_average_contrast_carries_the_modifier_distribution_uncertainty() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let problem = problem(&case);
        let posterior = BayesianGComputationAte::conjugate()
            .with_n_draws(DRAWS)
            .with_prior_scale(case["prior_scale"].as_f64().unwrap())
            .with_seed(17)
            .fit(
                &problem,
                IdentificationStatus::NonparametricallyIdentified,
                &mut BayesianGCompWorkspace::default(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let column = posterior.effect_column().unwrap();
        let (mean, sd) = (posterior.summaries.mean[column], posterior.summaries.sd[column]);
        let (r_mean, r_sd) = (case["mean"].as_f64().unwrap(), case["sd"].as_f64().unwrap());
        let root = (DRAWS as f64).sqrt();
        assert!((mean - r_mean).abs() < 5.0 * r_sd / root, "{name} mean {mean} vs {r_mean}");
        assert!((sd / r_sd - 1.0).abs() < 8.0 / root, "{name} sd {sd} vs {r_sd}");
        // The coefficient alone is narrower: the modifier-distribution term is real.
        assert!(sd > case["coefficient_sd"].as_f64().unwrap(), "{name}");
    }
}
