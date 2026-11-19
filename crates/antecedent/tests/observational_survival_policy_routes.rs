//! The public observational survival and policy-value routes: required declarations,
//! typed refusals with their own details, and a self-describing wire report.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::multi_level_treatment::AdjustmentColumn;
use antecedent::analysis::observational_policy::{
    ObservationalPolicyReportWire, ObservationalPolicyRequest, ObservationalPolicySpecWire,
    ObservationalPolicyValue, PolicyGraphDeclaration, compile_observational_policy,
    compile_policy_subsampling, consume_compiled_observational_policy, consume_policy_subsampling,
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

fn run_survival(
    request: &ObservationalSurvivalRequest,
) -> Result<ObservationalSurvivalReportWire, antecedent::CausalError> {
    ObservationalSurvivalEffect::evaluate(
        request,
        &antecedent::prelude::ExecutionContext::production_default(request.spec.seed),
    )
}

#[test]
fn survival_runs_under_the_callers_cancelled_context() {
    let request = survival_request(300, survival_spec());
    let ctx = antecedent::prelude::ExecutionContext::production_default(request.spec.seed);
    ctx.cancellation.cancel();
    assert!(ObservationalSurvivalEffect::evaluate(&request, &ctx).is_err());
}

#[test]
fn survival_report_states_its_estimands_assumptions_and_standing() {
    let request = survival_request(1500, survival_spec());
    let report = run_survival(&request).unwrap();
    assert!((report.horizon - 2.0).abs() < f64::EPSILON);
    assert_eq!(report.censoring, "at_random_given_covariates");
    assert_eq!(report.calibration, "not_measured");
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
    let text = run_survival(&survival_request(300, spec)).unwrap_err().to_string();
    assert!(
        text.contains(antecedent_core::reason_details::survival_effect::HORIZON_MISSING),
        "{text}"
    );
    assert!(text.contains("required_option_missing"), "{text}");

    let mut spec = survival_spec();
    spec.censoring = None;
    let text = run_survival(&survival_request(300, spec)).unwrap_err().to_string();
    assert!(
        text.contains(
            antecedent_core::reason_details::survival_effect::CENSORING_ASSUMPTION_MISSING
        ),
        "{text}"
    );
    assert!(text.contains("Informative censoring is not identified"), "{text}");

    let mut spec = survival_spec();
    spec.censoring = Some("informative".into());
    let text = run_survival(&survival_request(300, spec)).unwrap_err().to_string();
    assert!(
        text.contains(
            antecedent_core::reason_details::survival_effect::CENSORING_ASSUMPTION_UNKNOWN
        ),
        "{text}"
    );
}

#[test]
fn a_horizon_beyond_follow_up_is_a_positivity_refusal_on_the_public_route() {
    let mut spec = survival_spec();
    spec.horizon = Some(500.0);
    let text = run_survival(&survival_request(600, spec)).unwrap_err().to_string();
    assert!(text.contains("positivity_violated"), "{text}");
    assert!(
        text.contains(antecedent_core::reason_details::survival_effect::HORIZON_BEYOND_FOLLOW_UP),
        "{text}"
    );
}

fn run_policy(
    request: &ObservationalPolicyRequest,
) -> Result<ObservationalPolicyReportWire, antecedent::CausalError> {
    ObservationalPolicyValue::evaluate(
        request,
        &antecedent::prelude::ExecutionContext::production_default(request.spec.seed),
    )
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
fn learned_policy_compiles_the_split_and_independently_replays_it() {
    let request = policy_request(policy_spec("held_out"));
    let ctx = antecedent::prelude::ExecutionContext::production_default(request.spec.seed);
    let plan = compile_observational_policy(request, &ctx).expect("policy compiler");
    assert_eq!(plan.plan_view().route().as_str(), "ObservationalPolicy");
    let report = plan.execute(&ctx).expect("held-out estimate");
    assert_eq!(report.calibration, "not_measured");
    let bytes = plan.export(&report, &ctx).expect("replay artifact");
    let replay =
        consume_compiled_observational_policy(&bytes, Some(plan.lowered().identity()), &ctx)
            .expect("independent policy replay");
    assert_eq!(replay, report);
    let changed_seed = antecedent::prelude::ExecutionContext::production_default(3);
    assert!(consume_compiled_observational_policy(&bytes, None, &changed_seed).is_err());
}

#[test]
fn graph_checked_policy_binds_backdoor_proof_and_information_to_the_compiler() {
    let mut request = policy_request(policy_spec("held_out"));
    let graph = PolicyGraphDeclaration {
        nodes: vec!["z".into(), "a".into(), "y".into()],
        edges: vec![("z".into(), "a".into()), ("z".into(), "y".into()), ("a".into(), "y".into())],
        treatment: "a".into(),
        outcome: "y".into(),
        information_set: vec!["z".into()],
    };
    request.spec.causal_graph = Some(graph.clone());
    let ctx = antecedent::prelude::ExecutionContext::production_default(request.spec.seed);
    let plan = compile_observational_policy(request.clone(), &ctx).expect("backdoor compiler");
    let report = plan.execute(&ctx).expect("graph checked estimate");
    assert_eq!(
        ObservationalPolicyValue::evaluate(&request, &ctx).expect("public entry compiles first"),
        report
    );
    assert!(
        report.assumptions.iter().any(|text| text.contains("declared causally sufficient DAG"))
    );
    let bytes = plan.export(&report, &ctx).expect("versioned artifact");
    assert_eq!(
        consume_compiled_observational_policy(&bytes, Some(plan.lowered().identity()), &ctx,)
            .expect("independent graph check"),
        report
    );
    let mut confounded = request;
    confounded.adjustment.clear();
    confounded.spec.causal_graph.as_mut().expect("graph").information_set.clear();
    assert!(compile_observational_policy(confounded, &ctx).is_err());
    let mut late = policy_request(policy_spec("held_out"));
    late.spec.causal_graph = Some(graph);
    late.spec.causal_graph.as_mut().expect("graph").edges.push(("a".into(), "z".into()));
    assert!(compile_observational_policy(late, &ctx).is_err());
}

#[test]
fn graph_policy_can_learn_on_less_information_than_nuisance_adjustment() {
    let mut request = policy_request(policy_spec("held_out"));
    let rows = request.outcome.len();
    request.adjustment.push(AdjustmentColumn {
        name: "w".into(),
        values: (0..rows).map(|i| ((i as f64) * 0.31).sin()).collect(),
    });
    request.spec.causal_graph = Some(PolicyGraphDeclaration {
        nodes: vec!["z".into(), "w".into(), "a".into(), "y".into()],
        edges: vec![
            ("z".into(), "a".into()),
            ("z".into(), "y".into()),
            ("w".into(), "y".into()),
            ("a".into(), "y".into()),
        ],
        treatment: "a".into(),
        outcome: "y".into(),
        information_set: vec!["z".into()],
    });
    let ctx = antecedent::prelude::ExecutionContext::production_default(request.spec.seed);
    let plan = compile_observational_policy(request.clone(), &ctx)
        .expect("the policy can use a subset of the nuisance adjustment columns");
    let report = plan.execute(&ctx).expect("separate policy and nuisance fits");
    assert_eq!(report.calibration, "not_measured");
    let bytes = plan.export(&report, &ctx).expect("version-three policy artifact");
    assert_eq!(
        consume_compiled_observational_policy(&bytes, Some(plan.lowered().identity()), &ctx,)
            .expect("independent narrow-information replay"),
        report
    );
    let mut other = request;
    other.adjustment[1].values = (0..rows).map(|i| ((i as f64) * 0.47).cos()).collect();
    let other_report = compile_observational_policy(other, &ctx)
        .expect("second adjustment design")
        .execute(&ctx)
        .expect("second estimate");
    assert_eq!(other_report.policy_actions, report.policy_actions);
}

#[test]
fn cross_fitted_policy_subsampling_compiles_its_schedule_and_independently_replays() {
    let mut request = policy_request(ObservationalPolicySpecWire {
        support: Some(vec![0.0, 1.0]),
        objective: Some("maximize".into()),
        evaluation: Some("cross_fitted_learning".into()),
        folds: Some(3),
        seed: 2,
        ..ObservationalPolicySpecWire::default()
    });
    // A 300-row prefix keeps the hundred full refits of the schedule small.
    let rows = 300;
    request.outcome.truncate(rows);
    request.treatment.truncate(rows);
    request.adjustment[0].values.truncate(rows);
    request.adjustment.push(AdjustmentColumn {
        name: "w".into(),
        values: (0..rows).map(|i| ((i as f64) * 0.31).sin()).collect(),
    });
    request.spec.causal_graph = Some(PolicyGraphDeclaration {
        nodes: vec!["z".into(), "w".into(), "a".into(), "y".into()],
        edges: vec![
            ("z".into(), "a".into()),
            ("z".into(), "y".into()),
            ("w".into(), "y".into()),
            ("a".into(), "y".into()),
        ],
        treatment: "a".into(),
        outcome: "y".into(),
        information_set: vec!["z".into()],
    });
    let ctx = antecedent::prelude::ExecutionContext::production_default(request.spec.seed);
    let plan = compile_policy_subsampling(request.clone(), 150, 100, &ctx)
        .expect("the subsampling schedule compiles before any row is drawn");
    assert_eq!(plan.plan_view().route().as_str(), "PolicySubsampling");
    let candidate = plan.execute(&ctx).expect("one hundred full-refit subsamples");
    assert_eq!(candidate.report.evaluation_rows, rows as u64);
    assert!(candidate.interval.lower <= candidate.interval.upper);
    assert_eq!(candidate.interval.calibration, candidate.slot.status.as_str());
    assert_eq!(candidate.basis.key.interval_method, "cross_fitted_policy_subsampling_basic_v1");
    let bytes = plan.export(&candidate, &ctx).expect("subsampling artifact");
    let replay = consume_policy_subsampling(&bytes, plan.lowered().identity(), &ctx)
        .expect("independent recompilation and refits");
    assert_eq!(replay.report, candidate.report);
    assert_eq!(replay.interval, candidate.interval);
    // The artifact is bound to its split seed and to the compiled schedule.
    let other_seed = antecedent::prelude::ExecutionContext::production_default(3);
    assert!(consume_policy_subsampling(&bytes, plan.lowered().identity(), &other_seed).is_err());
    let other = compile_policy_subsampling(request.clone(), 120, 100, &ctx)
        .expect("a second subsample size compiles");
    assert_ne!(other.lowered().identity(), plan.lowered().identity());
    assert!(consume_policy_subsampling(&bytes, other.lowered().identity(), &ctx).is_err());
    assert!(compile_policy_subsampling(request.clone(), rows, 100, &ctx).is_err());
    assert!(compile_policy_subsampling(request, 150, 99, &ctx).is_err());
}

#[test]
fn policy_value_runs_under_the_callers_cancelled_context() {
    let request = policy_request(policy_spec("prespecified"));
    let ctx = antecedent::prelude::ExecutionContext::production_default(request.spec.seed);
    ctx.cancellation.cancel();
    assert!(ObservationalPolicyValue::evaluate(&request, &ctx).is_err());
}

#[test]
fn policy_value_reports_the_declared_support_evaluation_and_standing() {
    for (evaluation, expected_rows) in [("prespecified", 1200), ("cross_fitted_learning", 1200)] {
        let report = run_policy(&policy_request(policy_spec(evaluation))).unwrap();
        assert_eq!(report.evaluation, evaluation);
        assert_eq!(report.support, vec![0.0, 1.0]);
        assert_eq!(report.evaluation_rows, expected_rows);
        assert_eq!(report.calibration, "not_measured");
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
    let held_out = run_policy(&policy_request(policy_spec("held_out"))).unwrap();
    assert!(held_out.evaluation_rows < 1200 && held_out.learning_rows > 0);
    assert!(held_out.estimand.contains("held-out"));
}

#[test]
fn every_undeclared_policy_input_is_refused_with_its_own_detail() {
    type Mutate = Box<dyn Fn(&mut ObservationalPolicySpecWire)>;
    let cases: [(&str, Mutate); 6] = [
        (
            antecedent_core::reason_details::policy_value::SUPPORT_MISSING,
            Box::new(|s| s.support = None),
        ),
        (
            antecedent_core::reason_details::policy_value::OBJECTIVE_MISSING,
            Box::new(|s| s.objective = None),
        ),
        (
            antecedent_core::reason_details::policy_value::EVALUATION_MISSING,
            Box::new(|s| s.evaluation = None),
        ),
        (
            antecedent_core::reason_details::policy_value::POLICY_ORIGIN_MISSING,
            Box::new(|s| s.policy_origin = None),
        ),
        (
            antecedent_core::reason_details::policy_value::POLICY_ACTIONS_MISSING,
            Box::new(|s| s.policy_actions = None),
        ),
        (
            antecedent_core::reason_details::policy_value::EVALUATION_UNKNOWN,
            Box::new(|s| s.evaluation = Some("in_sample".into())),
        ),
    ];
    for (detail, edit) in cases {
        let mut request = policy_request(policy_spec("prespecified"));
        edit(&mut request.spec);
        let text = run_policy(&request).unwrap_err().to_string();
        assert!(text.contains(detail), "{detail}: {text}");
    }
    let mut request = policy_request(policy_spec("held_out"));
    request.spec.evaluation_fraction = None;
    let text = run_policy(&request).unwrap_err().to_string();
    assert!(
        text.contains(antecedent_core::reason_details::policy_value::EVALUATION_FRACTION_MISSING),
        "{text}"
    );
}

#[test]
fn reusing_the_evaluation_rows_to_learn_the_policy_and_unsupported_actions_are_refused() {
    let mut spec = policy_spec("prespecified");
    spec.policy_origin = Some("learned_on_evaluation_rows".into());
    let text = run_policy(&policy_request(spec)).unwrap_err().to_string();
    assert!(text.contains("policy_learned_on_evaluation_rows"), "{text}");

    let mut request = policy_request(policy_spec("prespecified"));
    request.spec.policy_actions.as_mut().unwrap()[0] = 7.0;
    let text = run_policy(&request).unwrap_err().to_string();
    assert!(text.contains("positivity_violated"), "{text}");
    assert!(
        text.contains(antecedent_core::reason_details::policy_value::ACTION_OUTSIDE_SUPPORT),
        "{text}"
    );
}

#[test]
fn many_policies_over_one_scope_match_each_request_share_fits_and_isolate_refusals() {
    use antecedent_state::retention::RetentionMode;
    let threshold = policy_request(policy_spec("prespecified"));
    let mut treat_all = threshold.clone();
    treat_all.spec.policy_actions = Some(vec![1.0; treat_all.outcome.len()]);
    let mut unsupported = threshold.clone();
    unsupported.spec.support = None;
    let requests = [threshold.clone(), unsupported, treat_all.clone()];
    let ctx = || antecedent::prelude::ExecutionContext::production_default(threshold.spec.seed);

    let results =
        ObservationalPolicyValue::evaluate_many(&requests, &ctx(), RetentionMode::FullReusable);
    assert_eq!(results.len(), 3);
    // Each result is its request's own evaluation under the same context.
    assert_eq!(
        results[0].as_ref().unwrap(),
        &ObservationalPolicyValue::evaluate(&threshold, &ctx()).unwrap()
    );
    assert_eq!(
        results[2].as_ref().unwrap(),
        &ObservationalPolicyValue::evaluate(&treat_all, &ctx()).unwrap()
    );
    // One request's refusal leaves the others untouched.
    let refusal = results[1].as_ref().unwrap_err().to_string();
    assert!(
        refusal.contains(antecedent_core::reason_details::policy_value::SUPPORT_MISSING),
        "{refusal}"
    );

    // The nuisance fits do not depend on the policy: retained state fits them once for both
    // policies, while a mode that retains nothing refits them per request.
    let fits = |retention| {
        let scope = ctx();
        let pair = [threshold.clone(), treat_all.clone()];
        let attempt = antecedent::execution::observe(&scope, || {
            let out = ObservationalPolicyValue::evaluate_many(&pair, &scope, retention);
            out.into_iter().collect::<Result<Vec<_>, _>>()
        });
        attempt.result.unwrap();
        attempt.facts.and_then(|facts| facts.observed_model_fits()).expect("fits are observed")
    };
    let shared = fits(RetentionMode::FullReusable);
    let separate = fits(RetentionMode::SummaryOnly);
    assert!(shared < separate, "retained fits {shared}, unretained fits {separate}");
}
