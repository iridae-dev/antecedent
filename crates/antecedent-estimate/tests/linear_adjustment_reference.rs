//! The linear-adjustment effect, its standard errors and its influence against 40-digit
//! references.
//!
//! `oracle_data/linear_adjustment_reference.py` (frozen in the JSON beside it) solves the normal
//! equations of `y ~ 1 + t + z1 + z2` in multiprecision arithmetic and forms, for a declared
//! treatment gap of two, the effect `delta b_t`, the homoskedastic and HC0-HC3 standard errors of
//! the treatment coefficient (leverage-corrected where defined), the exported influence
//! `n delta [(X'X)^-1 x_i]_t e_i` and, for two penalties, the ridge effect with the unpenalised
//! intercept and the ridge influence. Ridge publishes no analytic standard error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AssumptionSet, TargetPopulation, VariableId};
use antecedent_data::materialize::PreparationReceipt;
use antecedent_estimate::{
    AnalyticSeKind, EstimationWorkspace, InferenceSpec, LinearAdjustmentAte, LinearFitKind,
    OverlapPolicy, PreparedEstimationProblem,
};
use antecedent_stats::CompiledDesign;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/linear_adjustment_reference.json")).unwrap()
}

fn problem(reference: &Value) -> PreparedEstimationProblem {
    problem_for(reference, "outcome")
}

fn problem_for(reference: &Value, outcome: &str) -> PreparedEstimationProblem {
    let (t, y) = (floats(&reference["treatment"]), floats(&reference[outcome]));
    let (z1, z2) = (floats(&reference["z1"]), floats(&reference["z2"]));
    PreparedEstimationProblem {
        design: CompiledDesign::linear_adjustment(
            &t,
            &[(VariableId::from_raw(0), z1.as_slice()), (VariableId::from_raw(1), z2.as_slice())],
            &y,
            &[],
        )
        .unwrap(),
        method: Arc::from("backdoor.adjustment"),
        adjustment_set: Arc::from(vec![VariableId::from_raw(0), VariableId::from_raw(1)]),
        overlap: OverlapPolicy::ExplicitOverride,
        treatment_delta: reference["delta"].as_f64().unwrap(),
        target_population: TargetPopulation::AllObserved,
        treatment: Arc::from(t.clone()),
        active: 2.0,
        control: 0.0,
        treatment_levels: None,
        sampling_weights: None,
        preparation: PreparationReceipt::composed_finite(t.len()).unwrap(),
    }
}

fn estimator(kind: LinearFitKind, se_kind: AnalyticSeKind) -> LinearAdjustmentAte {
    let mut inference = InferenceSpec::new(0);
    inference.se_kind = se_kind;
    LinearAdjustmentAte { inference, fit_kind: kind, ..LinearAdjustmentAte::new() }
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-3),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn ols_effect_standard_errors_and_influence_match_the_exact_reference() {
    let reference = reference();
    let problem = problem(&reference);
    let ols = &reference["ols"];
    for (name, kind) in [
        ("homoskedastic", AnalyticSeKind::Homoskedastic),
        ("hc0", AnalyticSeKind::Hc0),
        ("hc1", AnalyticSeKind::Hc1),
        ("hc2", AnalyticSeKind::Hc2),
        ("hc3", AnalyticSeKind::Hc3),
    ] {
        let fit = estimator(LinearFitKind::Ols, kind)
            .fit_point(&problem, &mut EstimationWorkspace::default(), AssumptionSet::new())
            .unwrap();
        close(fit.ate, ols["ate"].as_f64().unwrap(), &format!("{name} effect"));
        close(fit.se_analytic(), ols["se"][name].as_f64().unwrap(), &format!("{name} se"));
        let influence = fit.scores.influence.as_ref().expect("influence");
        for (i, (a, e)) in influence.iter().zip(floats(&ols["influence"])).enumerate() {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1e-3),
                "{name} influence row {i}: {a} vs {e}"
            );
        }
    }
}

#[test]
fn ridge_effect_and_influence_match_and_no_analytic_standard_error_is_published() {
    let reference = reference();
    let problem = problem(&reference);
    for ridge in reference["ridge"].as_array().unwrap() {
        let lambda = ridge["lambda_"].as_f64().unwrap();
        let fit = estimator(LinearFitKind::Ridge { lambda }, AnalyticSeKind::Homoskedastic)
            .fit_point(&problem, &mut EstimationWorkspace::default(), AssumptionSet::new())
            .unwrap();
        close(fit.ate, ridge["ate"].as_f64().unwrap(), &format!("ridge {lambda} effect"));
        assert!(fit.se_analytic().is_nan(), "ridge publishes no analytic standard error");
        let influence = fit.scores.influence.as_ref().expect("influence");
        for (i, (a, e)) in influence.iter().zip(floats(&ridge["influence"])).enumerate() {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1e-3),
                "ridge {lambda} influence row {i}: {a} vs {e}"
            );
        }
    }
}

#[test]
fn huber_effect_and_homoskedastic_form_standard_error_match_the_fixed_point_reference() {
    let reference = reference();
    let problem = problem_for(&reference, "outcome_contaminated");
    let fit = estimator(LinearFitKind::Huber { c: 1.345 }, AnalyticSeKind::Homoskedastic)
        .fit_point(&problem, &mut EstimationWorkspace::default(), AssumptionSet::new())
        .unwrap();
    // The IRLS stops at a coefficient change of 1e-8 (relative), so agreement is to about 1e-6.
    let huber = &reference["huber"];
    let (ate, se) = (huber["ate"].as_f64().unwrap(), huber["se"].as_f64().unwrap());
    assert!((fit.ate - ate).abs() <= 1e-6 * ate.abs(), "effect {} vs {ate}", fit.ate);
    assert!((fit.se_analytic() - se).abs() <= 1e-6 * se, "se {} vs {se}", fit.se_analytic());
    // Contamination moves the least-squares effect further than the robust one.
    let ols = estimator(LinearFitKind::Ols, AnalyticSeKind::Homoskedastic)
        .fit_point(&problem, &mut EstimationWorkspace::default(), AssumptionSet::new())
        .unwrap();
    let clean = reference["ols"]["ate"].as_f64().unwrap();
    assert!((fit.ate - clean).abs() < (ols.ate - clean).abs());
}
