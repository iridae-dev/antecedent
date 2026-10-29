//! The public observational survival and policy-value routes: required declarations,
//! typed refusals with their own details, and a self-describing wire report.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::multi_level_treatment::AdjustmentColumn;
use antecedent::analysis::observational_policy::{
    ObservationalPolicyReportWire, ObservationalPolicyRequest, ObservationalPolicySpecWire,
    ObservationalPolicyValue,
};
use antecedent::analysis::observational_survival::{
    ObservationalSurvivalEffect, ObservationalSurvivalReportWire, ObservationalSurvivalRequest,
    ObservationalSurvivalSpecWire,
};
use antecedent_core::CausalRng;
use antecedent_kernels::{sigmoid, standard_normal};

fn survival_request(n: usize, spec: ObservationalSurvivalSpecWire) -> ObservationalSurvivalRequest {
    let mut rng = CausalRng::from_seed(5);
    let (mut duration, mut event, mut treatment, mut z) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let zi = standard_normal(&mut rng);
        let a = f64::from(u8::from(rng.next_f64() < sigmoid(0.7 * zi)));
        let rate = 0.3 * (-0.5 * a + 0.5 * zi).exp();
        let t = -rng.next_f64().max(1e-300).ln() / rate;
        let c = -rng.next_f64().max(1e-300).ln() / 0.12;
        duration.push(t.min(c));
        event.push(f64::from(u8::from(t <= c)));
        treatment.push(a);
        z.push(zi);
    }
    ObservationalSurvivalRequest {
        duration,
        event,
        treatment,
        adjustment: vec![AdjustmentColumn { name: "z".into(), values: z }],
        spec,
    }
}

fn survival_spec() -> ObservationalSurvivalSpecWire {
    ObservationalSurvivalSpecWire {
        active: 1.0,
        control: 0.0,
        horizon: Some(2.0),
        censoring: Some("at_random_given_covariates".into()),
        folds: None,
        min_propensity: None,
        min_censoring_survival: None,
        seed: 3,
    }
}

#[test]
fn survival_report_states_its_estimands_assumptions_and_standing() {
    let request = survival_request(1500, survival_spec());
    let report = ObservationalSurvivalEffect::evaluate(&request).unwrap();
    assert!((report.horizon - 2.0).abs() < f64::EPSILON);
    assert_eq!(report.censoring, "at_random_given_covariates");
    assert_eq!(report.calibration, "unmeasured");
    assert_eq!(report.estimands.len(), 2);
    assert!(report.assumptions.iter().any(|a| a.contains("coarsening at random")));
    assert!(report.assumptions.iter().any(|a| a.contains("exchangeability")));
    assert!(report.survival_difference.se > 0.0);
    assert_eq!(report.n_obs, 1500);
    // Truth by Gauss-Hermite over z ~ N(0, 1) of the exponential model with rate
    // 0.3 exp(-0.5 a + 0.5 z): S_1(2) = 0.67648, S_0(2) = 0.53525, difference 0.14123, and the
    // restricted mean survival difference E[(1 - exp(-2 rate)) / rate] gives 0.17509.
    // The estimates are sampling estimates from 1500 rows: each must sit within 3 of its own
    // reported standard errors of the truth.
    for (what, got, se, truth) in [
        ("S_1(2)", report.active.survival, report.active.survival_se, 0.676_479_797),
        ("S_0(2)", report.control.survival, report.control.survival_se, 0.535_250_383),
        (
            "S difference",
            report.survival_difference.estimate,
            report.survival_difference.se,
            0.141_229_414,
        ),
        (
            "RMST difference",
            report.rmst_difference.estimate,
            report.rmst_difference.se,
            0.175_086_738,
        ),
    ] {
        assert!((got - truth).abs() < 3.0 * se, "{what}: {got} vs {truth} (se {se})");
    }
    // The wire round-trips.
    let json = serde_json::to_string(&report).unwrap();
    let back: ObservationalSurvivalReportWire = serde_json::from_str(&json).unwrap();
    assert_eq!(back, report);
}

#[test]
fn a_missing_horizon_and_an_undeclared_censoring_assumption_are_refused_separately() {
    let mut spec = survival_spec();
    spec.horizon = None;
    let text = ObservationalSurvivalEffect::evaluate(&survival_request(300, spec))
        .unwrap_err()
        .to_string();
    assert!(text.contains("survival_effect.horizon_missing"), "{text}");
    assert!(text.contains("required_option_missing"), "{text}");

    let mut spec = survival_spec();
    spec.censoring = None;
    let text = ObservationalSurvivalEffect::evaluate(&survival_request(300, spec))
        .unwrap_err()
        .to_string();
    assert!(text.contains("survival_effect.censoring_assumption_missing"), "{text}");
    assert!(text.contains("Informative censoring is not identified"), "{text}");

    let mut spec = survival_spec();
    spec.censoring = Some("informative".into());
    let text = ObservationalSurvivalEffect::evaluate(&survival_request(300, spec))
        .unwrap_err()
        .to_string();
    assert!(text.contains("survival_effect.censoring_assumption_unknown"), "{text}");
}

#[test]
fn a_horizon_beyond_follow_up_is_a_positivity_refusal_on_the_public_route() {
    let mut spec = survival_spec();
    spec.horizon = Some(500.0);
    let text = ObservationalSurvivalEffect::evaluate(&survival_request(600, spec))
        .unwrap_err()
        .to_string();
    assert!(text.contains("positivity_violated"), "{text}");
    assert!(text.contains("survival_effect.horizon_beyond_follow_up"), "{text}");
}

fn policy_request(spec: ObservationalPolicySpecWire) -> ObservationalPolicyRequest {
    let mut rng = CausalRng::from_seed(6);
    let (mut outcome, mut treatment, mut z, mut policy) = (vec![], vec![], vec![], vec![]);
    for _ in 0..1200 {
        let zi = standard_normal(&mut rng);
        let a = f64::from(u8::from(rng.next_f64() < sigmoid(zi)));
        outcome.push(1.0 + zi + 1.5 * zi * a + 0.5 * standard_normal(&mut rng));
        treatment.push(a);
        policy.push(f64::from(u8::from(zi > 0.0)));
        z.push(zi);
    }
    let mut spec = spec;
    if spec.evaluation.as_deref() == Some("prespecified") {
        spec.policy_actions = Some(policy);
    }
    ObservationalPolicyRequest {
        outcome,
        treatment,
        adjustment: vec![AdjustmentColumn { name: "z".into(), values: z }],
        spec,
    }
}

fn policy_spec(evaluation: &str) -> ObservationalPolicySpecWire {
    ObservationalPolicySpecWire {
        support: Some(vec![0.0, 1.0]),
        objective: Some("maximize".into()),
        evaluation: Some(evaluation.into()),
        policy_origin: Some("prespecified".into()),
        evaluation_fraction: Some(0.4),
        seed: 2,
        ..ObservationalPolicySpecWire::default()
    }
}

#[test]
fn policy_value_reports_the_declared_support_evaluation_and_standing() {
    for (evaluation, expected_rows) in [("prespecified", 1200), ("cross_fitted_learning", 1200)] {
        let report =
            ObservationalPolicyValue::evaluate(&policy_request(policy_spec(evaluation))).unwrap();
        assert_eq!(report.evaluation, evaluation);
        assert_eq!(report.support, vec![0.0, 1.0]);
        assert_eq!(report.evaluation_rows, expected_rows);
        assert_eq!(report.calibration, "unmeasured");
        assert_eq!(report.action_values.len(), 2);
        // E[Y(a)] = 1 + E[z] + 1.5 a E[z] = 1 for both actions.
        for action in &report.action_values {
            assert!((action.value - 1.0).abs() < 3.0 * action.se, "{evaluation}: {action:?}");
        }
        // Treating exactly the units with z > 0 is worth 1 + 1.5 E[z+] = 1 + 1.5 / sqrt(2 pi).
        if evaluation == "prespecified" {
            assert!(
                (report.value - 1.598_413_421).abs() < 3.0 * report.se,
                "{} vs 1.5984 (se {})",
                report.value,
                report.se
            );
        }
        assert!(report.assumptions.iter().any(|a| a.contains("positivity")));
        let json = serde_json::to_string(&report).unwrap();
        let back: ObservationalPolicyReportWire = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }
    let held_out =
        ObservationalPolicyValue::evaluate(&policy_request(policy_spec("held_out"))).unwrap();
    assert!(held_out.evaluation_rows < 1200 && held_out.learning_rows > 0);
    assert!(held_out.estimand.contains("held-out"));
}

#[test]
fn every_undeclared_policy_input_is_refused_with_its_own_detail() {
    type Mutate = Box<dyn Fn(&mut ObservationalPolicySpecWire)>;
    let cases: [(&str, Mutate); 6] = [
        ("policy_value.support_missing", Box::new(|s| s.support = None)),
        ("policy_value.objective_missing", Box::new(|s| s.objective = None)),
        ("policy_value.evaluation_missing", Box::new(|s| s.evaluation = None)),
        ("policy_value.policy_origin_missing", Box::new(|s| s.policy_origin = None)),
        ("policy_value.policy_actions_missing", Box::new(|s| s.policy_actions = None)),
        ("policy_value.evaluation_unknown", Box::new(|s| s.evaluation = Some("in_sample".into()))),
    ];
    for (detail, edit) in cases {
        let mut request = policy_request(policy_spec("prespecified"));
        edit(&mut request.spec);
        let text = ObservationalPolicyValue::evaluate(&request).unwrap_err().to_string();
        assert!(text.contains(detail), "{detail}: {text}");
    }
    let mut request = policy_request(policy_spec("held_out"));
    request.spec.evaluation_fraction = None;
    let text = ObservationalPolicyValue::evaluate(&request).unwrap_err().to_string();
    assert!(text.contains("policy_value.evaluation_fraction_missing"), "{text}");
}

#[test]
fn reusing_the_evaluation_rows_to_learn_the_policy_and_unsupported_actions_are_refused() {
    let mut spec = policy_spec("prespecified");
    spec.policy_origin = Some("learned_on_evaluation_rows".into());
    let text = ObservationalPolicyValue::evaluate(&policy_request(spec)).unwrap_err().to_string();
    assert!(text.contains("policy_learned_on_evaluation_rows"), "{text}");

    let mut request = policy_request(policy_spec("prespecified"));
    request.spec.policy_actions.as_mut().unwrap()[0] = 7.0;
    let text = ObservationalPolicyValue::evaluate(&request).unwrap_err().to_string();
    assert!(text.contains("positivity_violated"), "{text}");
    assert!(text.contains("policy_value.action_outside_support"), "{text}");
}
