//! The GLM g-computation contrast and its delta-method standard error against 40-digit Fisher
//! scoring.
//!
//! `oracle_data/glm_adjustment_reference.py` (frozen in the JSON beside it) fits the main-effects
//! logit, probit and Poisson-log models by Fisher scoring to a 1e-35 fixed point and forms the
//! observed-row contrast `mean(m(x(1)'b) - m(x(0)'b))` with the delta-method standard error from
//! the expected information at the estimate. The estimator's IRLS fit, contrast and standard
//! error must reproduce them, and the treatment coefficient (a link-scale quantity) must not be
//! reported as the effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AssumptionSet, ExecutionContext, TargetPopulation, VariableId};
use antecedent_data::materialize::PreparationReceipt;
use antecedent_estimate::{
    GlmAdjustmentAte, GlmAdjustmentWorkspace, InferenceSpec, OverlapPolicy, PreparedGlmProblem,
};
use antecedent_stats::{CompiledDesign, GlmFamily};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn contrast_and_delta_method_standard_error_match_the_exact_reference() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/glm_adjustment_reference.json")).unwrap();
    for case in cases {
        let family_name = case["family"].as_str().unwrap();
        let family = match family_name {
            "logit" => GlmFamily::BinomialLogit,
            "probit" => GlmFamily::BinomialProbit,
            _ => GlmFamily::PoissonLog,
        };
        let (t, y) = (floats(&case["treatment"]), floats(&case["outcome"]));
        let (z1, z2) = (floats(&case["z1"]), floats(&case["z2"]));
        let problem = PreparedGlmProblem {
            design: CompiledDesign::linear_adjustment(
                &t,
                &[
                    (VariableId::from_raw(0), z1.as_slice()),
                    (VariableId::from_raw(1), z2.as_slice()),
                ],
                &y,
                &[],
            )
            .unwrap(),
            method: Arc::from("backdoor.adjustment"),
            adjustment_set: Arc::from(vec![VariableId::from_raw(0), VariableId::from_raw(1)]),
            overlap: OverlapPolicy::ExplicitOverride,
            active: 1.0,
            control: 0.0,
            family,
            target_population: TargetPopulation::AllObserved,
            treatment: Arc::from(t.clone()),
            treatment_levels: None,
            sampling: None,
            preparation: PreparationReceipt::composed_finite(t.len()).unwrap(),
        };
        let estimator = GlmAdjustmentAte {
            family,
            inference: InferenceSpec::new(0),
            ..GlmAdjustmentAte::new()
        };
        let fit = estimator
            .fit(
                &problem,
                &mut GlmAdjustmentWorkspace::default(),
                &ExecutionContext::for_tests(1),
                AssumptionSet::new(),
            )
            .unwrap();
        let (ate, se) = (case["ate"].as_f64().unwrap(), case["se"].as_f64().unwrap());
        assert!(
            (fit.ate - ate).abs() <= 1e-6 * ate.abs(),
            "{family_name} effect {} vs {ate}",
            fit.ate
        );
        assert!(
            (fit.se_analytic() - se).abs() <= 1e-6 * se,
            "{family_name} se {} vs {se}",
            fit.se_analytic()
        );
        let coefficient = case["treatment_coefficient"].as_f64().unwrap();
        assert!(
            (fit.ate - coefficient).abs() > 0.05,
            "{family_name}: the effect is not the link-scale coefficient"
        );
    }
}
