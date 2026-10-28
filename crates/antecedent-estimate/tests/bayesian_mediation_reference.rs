//! The independent-product mediation posterior against exact Normal-Inverse-Gamma moments.
//!
//! `oracle_data/bayesian_mediation_reference.py` (frozen in the JSON beside it) forms the exact
//! multivariate Student-t covariance of the mediator mechanism `M ~ 1 + T` and of the outcome
//! mechanism `Y ~ 1 + T + M`, and from their independence the mean and standard deviation of the
//! mediated quantity `delta a b`, the direct quantity `delta c` and the total `delta (c + a b)`,
//! including the covariance of `c` with the product that goes through the outcome mechanism's
//! own coefficient correlation. The composed posterior's draw moments must reproduce them, which
//! checks the product, the treatment-gap scaling and the three decomposition columns. The
//! dependence-coupled composition is not exercised here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    ExecutionContext, IdentificationStatus, Intervention, MediationContrast, MediationQuery, Value,
    VariableId,
};
use antecedent_estimate::bayesian_mediation::compose_temporal_mediation;
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, OverlapPolicy,
    PreparedBayesianProblem, SerialDependence,
};
use antecedent_prob::PosteriorQuantityKind;
use antecedent_stats::CompiledDesign;
use serde_json::Value as Json;

const DRAWS: usize = 300_000;

fn floats(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Json> {
    serde_json::from_str::<Json>(include_str!("oracle_data/bayesian_mediation_reference.json"))
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn problem(design: CompiledDesign) -> PreparedBayesianProblem {
    PreparedBayesianProblem {
        design,
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

fn fit(problem: &PreparedBayesianProblem, scale: f64, seed: u64) -> CausalPosterior {
    BayesianGComputationAte::conjugate()
        .with_n_draws(DRAWS)
        .with_prior_scale(scale)
        .with_seed(seed)
        .fit(
            problem,
            IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .expect("mechanism fit")
}

fn moments(post: &CausalPosterior, name: &str) -> (f64, f64) {
    let column = post
        .draws
        .schema
        .quantities
        .iter()
        .position(|q| match q {
            PosteriorQuantityKind::Scalar { name: n } => n.as_ref() == name,
            _ => false,
        })
        .expect("decomposition column");
    (post.summaries.mean[column], post.summaries.sd[column])
}

#[test]
fn the_composed_decomposition_matches_the_exact_product_moments() {
    for case in cases() {
        let label = case["name"].as_str().unwrap();
        let (t, m, y) =
            (floats(&case["treatment"]), floats(&case["mediator"]), floats(&case["outcome"]));
        let scale = case["prior_scale"].as_f64().unwrap();
        let delta = case["delta"].as_f64().unwrap();
        let mediator_problem =
            problem(CompiledDesign::linear_adjustment(&t, &[], &m, &[]).unwrap());
        let outcome_problem = problem(
            CompiledDesign::linear_adjustment(
                &t,
                &[(VariableId::from_raw(1), m.as_slice())],
                &y,
                &[],
            )
            .unwrap(),
        );
        let (mediator, outcome) =
            (fit(&mediator_problem, scale, 21), fit(&outcome_problem, scale, 22));
        let treatment_id = VariableId::from_raw(0);
        let mut query = MediationQuery::binary(
            treatment_id,
            VariableId::from_raw(2),
            vec![VariableId::from_raw(1)],
            MediationContrast::Total,
        );
        query.active = Intervention::set(treatment_id, Value::f64(delta));
        let post = compose_temporal_mediation(
            &mediator,
            &outcome,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
        )
        .expect("composition");
        let root = (DRAWS as f64).sqrt();
        for part in ["mediated", "direct", "total"] {
            let (mean, sd) = moments(&post, part);
            let (r_mean, r_sd) = (
                case[format!("{part}_mean")].as_f64().unwrap(),
                case[format!("{part}_sd")].as_f64().unwrap(),
            );
            assert!(
                (mean - r_mean).abs() < 5.0 * r_sd / root,
                "{label} {part} mean {mean} vs {r_mean}"
            );
            assert!((sd / r_sd - 1.0).abs() < 8.0 / root, "{label} {part} sd {sd} vs {r_sd}");
        }
        // The headline effect column is the requested contrast, here the total.
        let column = post.effect_column().expect("effect column");
        let (total, _) = moments(&post, "total");
        assert!((post.summaries.mean[column] - total).abs() < 1e-12);
    }
}
