use super::*;
use antecedent_model::mechanism_expr::MechanismExpression as E;
use antecedent_model::nonlinear_sde::{
    sample_sde_paths, SdeAction, SdeBoundary, SdeBoundaryRule, SdeDiffusion, SdeInput,
    SdeInterpolation, SdeParameter, SdeScheme, SdeSystem,
};

fn c(value: f64) -> E {
    E::Constant { value }
}
fn var(variable: usize) -> E {
    E::Variable { variable }
}
fn par(parameter: usize) -> E {
    E::Parameter { parameter }
}
fn add(terms: Vec<E>) -> E {
    E::Add { terms }
}
fn mul(factors: Vec<E>) -> E {
    E::Multiply { factors }
}
fn neg(e: E) -> E {
    mul(vec![c(-1.0), e])
}
fn sqrt_of(expression: E) -> SdeDiffusion {
    SdeDiffusion { expression, square_root: true }
}
fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

/// The documented SIR system, started at the given parameters; `extra` appends parameters
/// that no expression reads.
fn epidemic(start: [f64; 3], extra: usize) -> SdeSystem {
    let flux =
        mul(vec![par(0), add(vec![c(1.0), mul(vec![par(1), var(3)])]), var(0), var(1), c(0.001)]);
    let recovery = mul(vec![par(2), var(1)]);
    let mut parameters = vec![
        SdeParameter { name: "beta".into(), value: start[0] },
        SdeParameter { name: "amplitude".into(), value: start[1] },
        SdeParameter { name: "gamma".into(), value: start[2] },
    ];
    for k in 0..extra {
        parameters.push(SdeParameter { name: format!("unused{k}"), value: 1.0 });
    }
    SdeSystem {
        variables: vec!["S".into(), "I".into(), "R".into()],
        inputs: vec![SdeInput {
            name: "season".into(),
            interpolation: SdeInterpolation::Linear,
            times: vec![0.0, 5.0, 10.0, 15.0, 20.0],
            values: vec![-1.0, 0.0, 1.0, 0.0, -1.0],
            envelope: Some([-1.0, 1.0]),
        }],
        parameters,
        drift: vec![
            neg(flux.clone()),
            add(vec![flux.clone(), neg(recovery.clone())]),
            recovery.clone(),
        ],
        diffusion: vec![
            sqrt_of(flux.clone()),
            sqrt_of(add(vec![flux, recovery.clone()])),
            sqrt_of(recovery),
        ],
        boundaries: vec![
            SdeBoundary {
                lower: Some(0.0),
                upper: Some(1000.0),
                rule: SdeBoundaryRule::Reflect
            };
            3
        ],
    }
}

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../../conformance/gcm/nonlinear_sde_ensemble/expected.json"
    ))
    .unwrap()
}

fn numbers(value: &serde_json::Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

/// Retained trajectories of the independent NumPy generator, keeping every `every`-th point.
fn reference_trajectories(every: usize) -> Vec<SdeTrajectory> {
    let document = reference();
    document["quasi_likelihood"]["units"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(k, unit)| {
            let times = numbers(&unit["times"]);
            let rows = unit["values"].as_array().unwrap();
            SdeTrajectory {
                id: format!("unit-{k}"),
                intervention: SdeWorld::default(),
                observations: times
                    .iter()
                    .zip(rows)
                    .enumerate()
                    .filter(|(index, _)| index % every == 0)
                    .map(|(_, (time, row))| SdeObservation { time: *time, values: numbers(row) })
                    .collect(),
            }
        })
        .collect()
}

fn policy() -> SdeFitPolicy {
    SdeFitPolicy {
        optimizer: CyclicGradientPolicy {
            learning_rate: 0.05,
            score_tolerance: 0.05,
            max_iterations: 300,
            l1_penalty: 0.0,
        },
        ..SdeFitPolicy::default()
    }
}

const TRUTH: [f64; 3] = [0.6, 0.3, 0.25];

#[test]
fn quasi_log_likelihood_equals_the_dense_numpy_reference() {
    let document = reference();
    let quasi = &document["quasi_likelihood"];
    let compiled = epidemic([0.5, 0.2, 0.3], 0).compile().unwrap();
    let trajectories = reference_trajectories(1);
    for (key, theta) in [
        ("log_likelihood_truth", numbers(&quasi["truth"])),
        ("log_likelihood_alternative", numbers(&quasi["alternative"])),
    ] {
        let value = sde_quasi_log_likelihood(
            &compiled,
            &theta,
            None,
            None,
            &trajectories,
            policy(),
            &ctx(1),
        )
        .unwrap();
        let expected = quasi[key].as_f64().unwrap();
        assert!((value - expected).abs() <= 1e-9 * expected.abs(), "{key}: {value} vs {expected}");
    }
}

#[test]
fn dense_trajectories_recover_the_epidemic_parameters_with_separate_evidence() {
    let compiled = epidemic([0.5, 0.2, 0.3], 0).compile().unwrap();
    let trajectories = reference_trajectories(1);
    let fit = fit_sde_quasi_likelihood(
        &compiled,
        &[0, 1, 2],
        None,
        None,
        &trajectories,
        policy(),
        &ctx(2),
    )
    .unwrap();
    assert!(fit.fitting.converged, "{:?}", fit.fitting);
    let errors = fit.standard_errors.as_ref().unwrap();
    for ((value, truth), se) in fit.values.iter().zip(TRUTH).zip(errors) {
        assert!(*se > 0.0);
        assert!((value - truth).abs() <= 5.0 * se, "{value} vs {truth} (se {se})");
    }
    // The pseudo-likelihood maximum is at least as high as at the generating parameters.
    let truth_value = reference()["quasi_likelihood"]["log_likelihood_truth"].as_f64().unwrap();
    assert!(fit.log_quasi_likelihood >= truth_value - 1e-6);
    assert_eq!(fit.names, vec!["beta", "amplitude", "gamma"]);
    // Three distinct records.
    assert_ne!(fit.identification.status, SdeIdentificationStatus::NotIdentified);
    assert_eq!(fit.discretization.status, SdeDiscretizationStatus::Adequate);
    assert_eq!(fit.discretization.likelihood, "euler_gaussian_transition");
    assert_eq!(fit.discretization.transitions, 800);
    assert!(fit.discretization.maximum_rate_interval_product < 0.1);
    assert!(!fit
        .identification
        .conditions
        .contains(&"observation_interval_coarse_for_euler".to_owned()));
}

#[test]
fn coarse_observation_is_flagged_then_refused_by_the_interval_rate_product() {
    let compiled = epidemic([0.5, 0.2, 0.3], 0).compile().unwrap();
    // Every sixth point: interval 0.3, product well above the 0.1 flag and below the refusal.
    let flagged = fit_sde_quasi_likelihood(
        &compiled,
        &[0, 1, 2],
        None,
        None,
        &reference_trajectories(6),
        policy(),
        &ctx(3),
    )
    .unwrap();
    assert_eq!(
        flagged.discretization.status,
        SdeDiscretizationStatus::ModelConditionalApproximation
    );
    assert_eq!(flagged.identification.status, SdeIdentificationStatus::ModelConditional);
    assert!(flagged
        .identification
        .conditions
        .contains(&"observation_interval_coarse_for_euler".to_owned()));
    assert!(flagged.discretization.transitions_above_flag > 0);
    // Every fortieth point: interval 2.0, product far above the 0.5 refusal.
    match fit_sde_quasi_likelihood(
        &compiled,
        &[0, 1, 2],
        None,
        None,
        &reference_trajectories(40),
        policy(),
        &ctx(4),
    ) {
        Err(SdeFitError::Unsupported { code, .. }) => {
            assert_eq!(code, "sde_fit.observation_interval_too_coarse");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn designs_without_transitions_or_with_unread_parameters_are_refused_as_unidentified() {
    let compiled = epidemic([0.5, 0.2, 0.3], 0).compile().unwrap();
    let snapshots: Vec<SdeTrajectory> = reference_trajectories(1)
        .into_iter()
        .map(|mut t| {
            t.observations.truncate(1);
            t
        })
        .collect();
    match fit_sde_quasi_likelihood(&compiled, &[0], None, None, &snapshots, policy(), &ctx(5)) {
        Err(SdeFitError::Unidentified { failure, .. }) => {
            assert_eq!(failure, SdeIdentificationFailure::NoTransitionInformation);
        }
        other => panic!("{other:?}"),
    }
    // A parameter no expression reads has a flat likelihood.
    let with_unused = epidemic([0.5, 0.2, 0.3], 1).compile().unwrap();
    match fit_sde_quasi_likelihood(
        &with_unused,
        &[0, 1, 2, 3],
        None,
        None,
        &reference_trajectories(1),
        policy(),
        &ctx(6),
    ) {
        Err(SdeFitError::Unidentified { failure, detail }) => {
            assert_eq!(failure, SdeIdentificationFailure::SingularInformation);
            assert!(detail.contains("rank"), "{detail}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_vanishing_transition_variance_is_refused() {
    let compiled = epidemic([0.5, 0.2, 0.3], 0).compile().unwrap();
    let extinct = vec![SdeTrajectory {
        id: "extinct".into(),
        intervention: SdeWorld::default(),
        observations: vec![
            SdeObservation { time: 0.0, values: vec![500.0, 0.0, 500.0] },
            SdeObservation { time: 0.05, values: vec![500.0, 0.0, 500.0] },
        ],
    }];
    match fit_sde_quasi_likelihood(&compiled, &[0], None, None, &extinct, policy(), &ctx(7)) {
        Err(SdeFitError::Unsupported { code, .. }) => {
            assert_eq!(code, "sde_fit.degenerate_transition")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn malformed_designs_are_invalid_arguments() {
    let compiled = epidemic([0.5, 0.2, 0.3], 0).compile().unwrap();
    let trajectories = reference_trajectories(1);
    assert!(matches!(
        fit_sde_quasi_likelihood(&compiled, &[], None, None, &trajectories, policy(), &ctx(8)),
        Err(SdeFitError::Invalid(_))
    ));
    assert!(matches!(
        fit_sde_quasi_likelihood(&compiled, &[0, 0], None, None, &trajectories, policy(), &ctx(8)),
        Err(SdeFitError::Invalid(_))
    ));
    assert!(matches!(
        fit_sde_quasi_likelihood(&compiled, &[9], None, None, &trajectories, policy(), &ctx(8)),
        Err(SdeFitError::Invalid(_))
    ));
    // An observation model needs a filter prior.
    let model = SdeObservationModel {
        matrix: vec![0.0, 1.0, 0.0],
        offset: vec![],
        noise_standard_deviation: vec![5.0],
    };
    assert!(matches!(
        fit_sde_quasi_likelihood(
            &compiled,
            &[0],
            Some(&model),
            None,
            &trajectories,
            policy(),
            &ctx(8)
        ),
        Err(SdeFitError::Invalid(_))
    ));
    let mut bad = policy();
    bad.coarse_flag = 1.0;
    assert!(matches!(
        fit_sde_quasi_likelihood(&compiled, &[0], None, None, &trajectories, bad, &ctx(8)),
        Err(SdeFitError::Invalid(_))
    ));
}

fn simulated(world: SdeWorld, every: usize, seed: u64, paths: usize) -> Vec<SdeTrajectory> {
    let compiled = epidemic([0.6, 0.3, 0.25], 0).compile().unwrap();
    let simulation = SdeSimulationPolicy {
        start: 0.0,
        step: 0.05,
        steps: 400,
        output_every: every,
        scheme: SdeScheme::EulerMaruyama,
        paths,
        seed,
        weak_tolerance: 0.05,
        check_step: false,
        retain_final_samples: false,
    };
    let data = sample_sde_paths(&compiled, None, &[950.0, 50.0, 0.0], &simulation, &world, &ctx(9))
        .unwrap();
    let n = data.times.len();
    (0..paths)
        .map(|p| SdeTrajectory {
            id: format!("sim-{p}"),
            intervention: world.clone(),
            observations: (0..n)
                .map(|t| SdeObservation {
                    time: data.times[t],
                    values: data.states[(p * n + t) * 3..(p * n + t) * 3 + 3].to_vec(),
                })
                .collect(),
        })
        .collect()
}

#[test]
fn trajectories_under_a_known_hold_exclude_the_held_component_from_the_likelihood() {
    let compiled = epidemic([0.5, 0.2, 0.3], 0).compile().unwrap();
    let hold = SdeWorld {
        id: "suppress".into(),
        actions: vec![SdeAction { variable: 1, times: vec![5.0], values: vec![20.0] }],
        inputs: vec![],
    };
    let trajectories = simulated(hold, 1, 17, 3);
    let fit = fit_sde_quasi_likelihood(
        &compiled,
        &[0, 1, 2],
        None,
        None,
        &trajectories,
        policy(),
        &ctx(10),
    )
    .unwrap();
    assert!(fit.fitting.converged, "{:?}", fit.fitting);
    let errors = fit.standard_errors.as_ref().unwrap();
    for ((value, truth), se) in fit.values.iter().zip(TRUTH).zip(errors) {
        assert!((value - truth).abs() <= 6.0 * se, "{value} vs {truth} (se {se})");
    }
}

#[test]
fn partial_noisy_observation_uses_the_extended_kalman_likelihood_and_says_so() {
    let compiled = epidemic([0.55, 0.3, 0.3], 0).compile().unwrap();
    let truth = simulated(SdeWorld { id: "baseline".into(), ..SdeWorld::default() }, 10, 23, 2);
    let noise = 5.0;
    let trajectories: Vec<SdeTrajectory> = truth
        .into_iter()
        .enumerate()
        .map(|(p, mut t)| {
            for (k, o) in t.observations.iter_mut().enumerate() {
                let z = counter_normal(77, p as u64, k as u64, 0);
                o.values = vec![o.values[1] + noise * z];
            }
            t
        })
        .collect();
    let model = SdeObservationModel {
        matrix: vec![0.0, 1.0, 0.0],
        offset: vec![],
        noise_standard_deviation: vec![noise],
    };
    let prior = SdeFilterPrior {
        mean: vec![950.0, 50.0, 0.0],
        covariance: vec![25.0, 0.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 1.0],
    };
    let mut filter_policy = policy();
    filter_policy.filter_step = 0.05;
    let fit = fit_sde_quasi_likelihood(
        &compiled,
        &[0, 2],
        Some(&model),
        Some(&prior),
        &trajectories,
        filter_policy,
        &ctx(11),
    )
    .unwrap();
    assert!(fit.log_quasi_likelihood.is_finite());
    assert_eq!(fit.discretization.likelihood, "extended_kalman_euler_innovation");
    assert_eq!(fit.identification.unobserved_components, vec!["S", "R"]);
    for condition in ["extended_kalman_linearization", "unobserved_state_filtered"] {
        assert!(fit.identification.conditions.contains(&condition.to_owned()), "{condition}");
    }
    assert_eq!(fit.identification.status, SdeIdentificationStatus::ModelConditional);
    // The substep controls the product, so a half-unit observation interval is admitted.
    assert!(fit.discretization.maximum_interval >= 0.5 - 1e-9);
    assert!(fit.discretization.maximum_rate_interval_product < 0.1);
}

#[test]
fn reports_keep_monte_carlo_numerical_and_parameter_uncertainty_apart() {
    let compiled = epidemic([0.5, 0.2, 0.3], 0).compile().unwrap();
    let fit = fit_sde_quasi_likelihood(
        &compiled,
        &[0, 1, 2],
        None,
        None,
        &reference_trajectories(1),
        policy(),
        &ctx(12),
    )
    .unwrap();
    let lockdown = SdeWorld {
        id: "lockdown".into(),
        actions: vec![],
        inputs: vec![SdeInput {
            name: "season".into(),
            interpolation: SdeInterpolation::PiecewiseConstant,
            times: vec![0.0],
            values: vec![-1.0],
            envelope: Some([-1.0, 1.0]),
        }],
    };
    let contrast =
        SdeContrast { id: "effect".into(), treated: "lockdown".into(), control: "baseline".into() };
    let simulation = SdeSimulationPolicy {
        start: 0.0,
        step: 0.05,
        steps: 200,
        output_every: 100,
        scheme: SdeScheme::EulerMaruyama,
        paths: 60,
        seed: 31,
        weak_tolerance: 0.05,
        check_step: true,
        retain_final_samples: false,
    };
    let initial = [950.0, 50.0, 0.0];
    let report = sde_report(
        &compiled,
        Some(fit.clone()),
        &initial,
        &simulation,
        &[lockdown.clone()],
        &[contrast.clone()],
        24,
        5,
        &ctx(13),
    )
    .unwrap();
    // The ensemble is simulated at the fitted parameters, not the declared start values.
    assert_eq!(report.simulation.parameters, fit.theta);
    let spread = report.parameter_uncertainty.as_ref().unwrap();
    assert!(spread.completed_draws >= 2);
    assert_eq!(spread.worlds.len(), 2);
    let variance = &spread.contrasts[0].mean_difference_between_draw_variance;
    assert!(variance.iter().flatten().all(|v| v.is_finite() && *v >= 0.0));
    assert!(variance[2].iter().any(|v| *v > 0.0));
    // The three sources are separate fields of one report.
    assert!(report.simulation.worlds[0].numerical.is_some());
    assert!(report.simulation.worlds[0].monte_carlo_standard_error[2][1] > 0.0);
    // Without a fit there is nothing to propagate.
    match sde_report(
        &compiled,
        None,
        &initial,
        &simulation,
        &[lockdown],
        &[contrast],
        10,
        5,
        &ctx(14),
    ) {
        Err(SdeFitError::Unsupported { code, .. }) => {
            assert_eq!(code, "sde_fit.parameter_uncertainty_unavailable");
        }
        other => panic!("{other:?}"),
    }
}
