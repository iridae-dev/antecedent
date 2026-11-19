//! Independent structural truth for the compiled latent soft-intervention route.

use antecedent::analysis::latent_soft_intervention::{
    LatentSoftInterventionRequest, VersionEvidence, replay_forecast,
};
use antecedent::planner::PlanRoute;
use antecedent::Study;
use antecedent::prelude::ExecutionContext;
use antecedent_core::CausalRng;

fn standard_normal(rng: &mut CausalRng) -> f64 {
    let radial = (-2.0 * (1.0 - rng.next_f64()).ln()).sqrt();
    radial * (std::f64::consts::TAU * rng.next_f64()).cos()
}

fn fixture(mixture: bool) -> LatentSoftInterventionRequest {
    let mut rng = CausalRng::from_seed(0x4c53_495f_5452_5554);
    let controls = (0..1_200)
        .map(|_| {
            let x = standard_normal(&mut rng);
            let a = 0.4 * x + 0.45 * standard_normal(&mut rng);
            let b = -0.3 * x + 0.45 * standard_normal(&mut rng);
            let y = a + b + 0.6 * standard_normal(&mut rng);
            vec![x, a, b, y]
        })
        .collect();
    let action = (0..1_200)
        .map(|_| {
            let x = standard_normal(&mut rng);
            let route_a = !mixture || rng.next_f64() < 1.0 / (1.0 + (-1.8 * x).exp());
            let a = 0.4 * x + 0.45 * standard_normal(&mut rng) + if route_a { 3.0 } else { 0.0 };
            let b = -0.3 * x + 0.45 * standard_normal(&mut rng) + if route_a { 0.0 } else { 3.0 };
            let y = a + b + 0.6 * standard_normal(&mut rng);
            vec![x, a, b, y]
        })
        .collect();
    LatentSoftInterventionRequest {
        variables: ["x", "path_a", "path_b", "outcome"].map(String::from).to_vec(),
        edges: vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        context: vec![0],
        candidates: vec![1, 2, 3],
        baseline: controls,
        action,
        population: "independent_gaussian_trial".into(),
        randomized: true,
        version_exogenous_given_context: true,
        seed: 73,
        refits: 0,
        search_operations: 41_000,
    }
}

#[test]
fn equal_downstream_means_preserve_distinct_pathways_and_replay() {
    let ctx = ExecutionContext::for_tests(0);
    let prepared = Study::prepare_latent_soft_intervention(fixture(true), &ctx).unwrap();
    let program = prepared.program_identity().to_hex();
    assert_eq!(prepared.plan_view().route(), PlanRoute::LatentSoftIntervention);
    let fitted = prepared.execute(&ctx).unwrap();
    let discovery = fitted.discovery();
    assert_eq!(discovery.program, program);
    assert_eq!(discovery.evidence, VersionEvidence::TargetsPreferred);
    assert_eq!(discovery.shifts.len(), 2);
    let at_positive = fitted.forecast(&[1.0], &ctx).unwrap();
    let at_negative = fitted.forecast(&[-1.0], &ctx).unwrap();
    assert_ne!(at_positive.query, at_negative.query);
    let at = |name: &str| at_positive.variables.iter().position(|n| n == name).unwrap();
    let a = at("path_a");
    let b = at("path_b");
    let y = at("outcome");
    let find = |coordinate| {
        at_positive
            .action
            .iter()
            .position(|law| law.mean[coordinate] - at_positive.baseline.mean[coordinate] > 2.5)
            .unwrap()
    };
    let route_a = find(a);
    let route_b = find(b);
    assert_ne!(route_a, route_b);
    let true_positive_weight = 1.0 / (1.0 + (-1.8_f64).exp());
    assert!((at_positive.action[route_a].weight - true_positive_weight).abs() < 0.10);
    assert!((at_negative.action[route_a].weight - (1.0 - true_positive_weight)).abs() < 0.10);
    for law in &at_positive.action {
        assert!((law.mean[y] - at_positive.baseline.mean[y] - 3.0).abs() < 0.30);
        let k = at_positive.variables.len();
        // Independent truth: Cov(M_a,Y | X) = Var(e_a), since Y = M_a + M_b + e_y.
        assert!((law.covariance[a * k + y] - 0.45_f64.powi(2)).abs() < 0.07);
    }
    let marginal = at_positive.action_marginal("path_a", &ctx).unwrap();
    let mean = at_positive.baseline.mean[a] + at_positive.mean_difference[a];
    assert!((marginal.value("mean").unwrap() - mean).abs() < 1e-12);
    let width = at_positive.variables.len();
    let variation = at_positive
        .action
        .iter()
        .map(|law| law.weight * (law.covariance[a * width + a] + (law.mean[a] - mean).powi(2)))
        .sum::<f64>();
    assert!((marginal.value("variance").unwrap() - variation).abs() < 1e-12);
    assert!(at_positive.action_marginal("undeclared", &ctx).is_err());

    let exported = fitted.export_forecast(&[1.0], &ctx).unwrap();
    let replay = replay_forecast(&exported.bytes, &exported.expectation, &ctx).unwrap();
    assert_eq!(&replay.discovery, discovery);
    assert_eq!(replay.forecast, at_positive);
    let another = fitted.export_forecast(&[-1.0], &ctx).unwrap();
    let Err(error) = replay_forecast(&another.bytes, &exported.expectation, &ctx) else {
        panic!("expected a mismatched forecast context to refuse");
    };
    assert_eq!(error.reason_code(), Some("invalid_argument"));
    assert!(error.to_string().contains("artifact forecast context differs"));
    assert!(fitted.forecast(&[50.0], &ctx).is_err());
}

#[test]
fn one_version_leaves_a_split_unresolved() {
    let ctx = ExecutionContext::for_tests(0);
    let fitted = Study::prepare_latent_soft_intervention(fixture(false), &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert_eq!(fitted.discovery().evidence, VersionEvidence::SplitUnresolved);
    assert_eq!(fitted.forecast(&[0.0], &ctx).unwrap().action.len(), 1);
}

#[test]
fn compiler_rejects_missing_premises_and_post_action_context() {
    let ctx = ExecutionContext::for_tests(0);
    let mut request = fixture(true);
    request.version_exogenous_given_context = false;
    assert!(Study::prepare_latent_soft_intervention(request.clone(), &ctx).is_err());
    request.version_exogenous_given_context = true;
    request.context.push(3);
    request.candidates.pop();
    assert!(Study::prepare_latent_soft_intervention(request, &ctx).is_err());
}

#[test]
fn variance_only_change_does_not_release_a_single_shift_forecast() {
    let mut request = fixture(false);
    for row in &mut request.action {
        // Widen the disturbances of path A with no latent shift. Its descendants
        // receive the resulting change from the same structural edge.
        let a = row[1];
        row[1] = 0.4 * row[0] + 4.0 * (a - 0.4 * row[0] - 3.0);
        row[3] += row[1] - a;
    }
    let ctx = ExecutionContext::for_tests(0);
    let fit =
        Study::prepare_latent_soft_intervention(request, &ctx).unwrap().execute(&ctx).unwrap();
    assert_eq!(fit.discovery().evidence, VersionEvidence::SplitUnresolved);
    assert!(!fit.discovery().forecast_admissible);
    assert!(fit.forecast(&[0.0], &ctx).is_err());
}

#[test]
fn joint_refits_keep_baseline_and_weighted_action_aligned() {
    let mut request = fixture(true);
    request.refits = 4;
    let ctx = ExecutionContext::for_tests(0);
    let fit =
        Study::prepare_latent_soft_intervention(request, &ctx).unwrap().execute(&ctx).unwrap();
    let output = fit.forecast(&[0.0], &ctx).unwrap();
    let refits = output.refits.as_ref().unwrap();
    assert_eq!(refits.attempted, 4);
    assert_eq!(usize::from(refits.attempted - refits.unavailable), refits.draws.len());
    assert!(!refits.draws.is_empty());
    for draw in &refits.draws {
        for i in 0..output.variables.len() {
            let manual = draw.action.iter().map(|part| part.weight * part.mean[i]).sum::<f64>()
                - draw.baseline.mean[i];
            assert!((draw.mean_difference[i] - manual).abs() < 1e-12);
        }
    }
    let export = fit.export_forecast(&[0.0], &ctx).unwrap();
    assert_eq!(replay_forecast(&export.bytes, &export.expectation, &ctx).unwrap().forecast, output);
}

#[test]
fn compiler_meter_refuses_an_exhausted_search() {
    let ctx = ExecutionContext::for_tests(0);
    let mut request = fixture(true);
    request.search_operations = 1;
    let prepared = Study::prepare_latent_soft_intervention(request, &ctx).unwrap();
    match prepared.execute(&ctx) {
        Err(antecedent::CausalError::Resource { message }) => {
            assert!(message.contains(antecedent_core::search::SearchStop::Operations.code()));
        }
        _ => panic!("expected a refusal from the compiled search budget"),
    }
}

#[test]
fn unearned_release_route_is_inspectable_and_refuses_execution() {
    let ctx = ExecutionContext::for_tests(0);
    let prepared = Study::prepare_latent_soft_intervention(fixture(true), &ctx).unwrap();
    assert_eq!(prepared.plan_view().route(), PlanRoute::LatentSoftIntervention);
    if !prepared.plan_view().support().supported() && antecedent_core::BUILD_FLAVOR == "release" {
        let Err(error) = prepared.execute(&ctx) else {
            panic!("expected the release evidence gate to refuse");
        };
        assert_eq!(error.reason_code(), Some("not_supported"));
    }
}
