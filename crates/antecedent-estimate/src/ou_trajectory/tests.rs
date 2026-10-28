use super::*;
use antecedent_core::{CausalRng, RngFactory};
use antecedent_kernels::standard_normal;

const DT: f64 = 0.002;
const BURN_IN_STEPS: u64 = 20_000;

fn truth() -> OuMechanisms {
    OuMechanisms {
        equations: vec!["a".into(), "b".into()],
        drift: vec![-1.0, 0.5, -0.4, -0.8],
        forcing: vec![0.6, -0.3],
        diffusion_factor: vec![0.5, 0.0, 0.2, 0.4],
    }
}

fn start() -> OuMechanisms {
    OuMechanisms {
        equations: vec!["a".into(), "b".into()],
        drift: vec![-0.7, 0.2, -0.1, -1.1],
        forcing: vec![0.3, 0.0],
        diffusion_factor: vec![0.3, 0.0, 0.1, 0.3],
    }
}

fn full_coordinates() -> Vec<OuMechanismParameter> {
    vec![
        OuMechanismParameter::Drift { equation: 0, parent: 0 },
        OuMechanismParameter::Drift { equation: 0, parent: 1 },
        OuMechanismParameter::Drift { equation: 1, parent: 0 },
        OuMechanismParameter::Drift { equation: 1, parent: 1 },
        OuMechanismParameter::Forcing { equation: 0 },
        OuMechanismParameter::Forcing { equation: 1 },
        OuMechanismParameter::Diffusion { equation: 0, source: 0 },
        OuMechanismParameter::Diffusion { equation: 1, source: 0 },
        OuMechanismParameter::Diffusion { equation: 1, source: 1 },
    ]
}

fn truth_values() -> Vec<f64> {
    vec![-1.0, 0.5, -0.4, -0.8, 0.6, -0.3, 0.5, 0.2, 0.4]
}

fn policy() -> CyclicGradientPolicy {
    CyclicGradientPolicy {
        learning_rate: 0.05,
        score_tolerance: 1e-3,
        max_iterations: 400,
        l1_penalty: 0.0,
    }
}

/// Replace one equation row by hand, independently of the library's replacement code.
fn with_row(
    model: &OuMechanisms,
    i: usize,
    drift: &[f64],
    forcing: f64,
    diffusion: &[f64],
) -> OuMechanisms {
    let n = model.equations.len();
    let mut out = model.clone();
    out.drift[i * n..(i + 1) * n].copy_from_slice(drift);
    out.diffusion_factor[i * n..(i + 1) * n].copy_from_slice(diffusion);
    out.forcing[i] = forcing;
    out
}

/// Euler–Maruyama on a fine grid: a different code path from the exact-transition fit.
fn euler(
    phases: &[(u64, OuMechanisms)],
    x0: &[f64],
    observe: &[u64],
    dt: f64,
    rng: &mut CausalRng,
) -> Vec<Vec<f64>> {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut out = Vec::new();
    let mut next = 0;
    let mut step = 0_u64;
    loop {
        while next < observe.len() && observe[next] == step {
            out.push(x.clone());
            next += 1;
        }
        if next == observe.len() {
            return out;
        }
        let model = &phases.iter().rev().find(|p| p.0 <= step).unwrap().1;
        let z: Vec<f64> = (0..n).map(|_| standard_normal(rng)).collect();
        let mut moved = x.clone();
        for i in 0..n {
            let drift: f64 =
                (0..n).map(|j| model.drift[i * n + j] * x[j]).sum::<f64>() + model.forcing[i];
            let noise: f64 = (0..n).map(|s| model.diffusion_factor[i * n + s] * z[s]).sum();
            moved[i] = x[i] + drift * dt + dt.sqrt() * noise;
        }
        x = moved;
        step += 1;
    }
}

fn stationary_mean_2d(model: &OuMechanisms) -> Vec<f64> {
    let m = &model.drift;
    let b = &model.forcing;
    let det = m[0] * m[3] - m[1] * m[2];
    vec![-(m[3] * b[0] - m[1] * b[1]) / det, -(-m[2] * b[0] + m[0] * b[1]) / det]
}

/// Irregular-interval trajectories of the truth, observed through `observe`.
fn simulate_units(
    seed: u64,
    units: usize,
    per_unit: usize,
    truth_model: &OuMechanisms,
    measure: &dyn Fn(&[f64], &mut CausalRng) -> Vec<f64>,
) -> Vec<OuTrajectory> {
    let factory = RngFactory::from_seed(seed);
    (0..units)
        .map(|u| {
            let mut rng = factory.stream(0x7261_6a00 + u as u64);
            let mut noise_rng = factory.stream(0x6e6f_6900 + u as u64);
            let mut steps = Vec::new();
            let mut cumulative = 0_u64;
            for k in 0..per_unit {
                cumulative += 150 + ((k as u64 * 7919 + (seed + u as u64) * 104_729) % 451);
                steps.push(cumulative);
            }
            let observe: Vec<u64> = steps.iter().map(|s| BURN_IN_STEPS + s).collect();
            let states = euler(
                &[(0, truth_model.clone())],
                &stationary_mean_2d(truth_model),
                &observe,
                DT,
                &mut rng,
            );
            OuTrajectory {
                id: format!("unit-{u}"),
                schedule: vec![OuScheduleSegment { start: 0.0, replacements: vec![] }],
                initial: OuInitialLaw::Stationary,
                observations: steps
                    .iter()
                    .zip(&states)
                    .map(|(s, x)| OuObservation {
                        time: *s as f64 * DT,
                        values: measure(x, &mut noise_rng),
                    })
                    .collect(),
            }
        })
        .collect()
}

fn direct_measure(x: &[f64], _: &mut CausalRng) -> Vec<f64> {
    x.to_vec()
}

fn assert_recovered(fit: &OuTrajectoryFit, coordinates: &[usize]) {
    let truth = truth_values();
    let errors = fit.standard_errors.as_ref().expect("positive-definite information");
    for (position, &index) in coordinates.iter().enumerate() {
        let (estimate, standard_error) = (fit.values[position], errors[position]);
        assert!(
            (estimate - truth[index]).abs() <= 4.0 * standard_error + 0.02,
            "coordinate {index}: estimate {estimate} truth {} se {standard_error}",
            truth[index]
        );
    }
}

#[test]
fn the_exact_transition_matches_the_scalar_closed_form() {
    let model = OuMechanisms {
        equations: vec!["x".into()],
        drift: vec![-0.7],
        forcing: vec![0.35],
        diffusion_factor: vec![0.6],
    };
    let h = 0.9;
    let step = transition(&model, h, None).unwrap();
    let phi = (-0.7_f64 * h).exp();
    assert!((step.phi[0] - phi).abs() < 1e-13);
    assert!((step.shift[0] - (0.35 / 0.7) * (1.0 - phi)).abs() < 1e-13);
    assert!((step.q[0] - 0.36 * (1.0 - phi * phi) / 1.4).abs() < 1e-13);
}

#[test]
fn the_exact_transition_handles_a_singular_drift_and_rotation() {
    // Brownian motion with drift: M = 0, so Phi = 1, shift = b h and Q = L L^T h.
    let brownian = OuMechanisms {
        equations: vec!["x".into()],
        drift: vec![0.0],
        forcing: vec![0.25],
        diffusion_factor: vec![0.5],
    };
    let step = transition(&brownian, 2.0, None).unwrap();
    assert!((step.phi[0] - 1.0).abs() < 1e-14);
    assert!((step.shift[0] - 0.5).abs() < 1e-14);
    assert!((step.q[0] - 0.5).abs() < 1e-13);
    let rotation = OuMechanisms {
        equations: vec!["a".into(), "b".into()],
        drift: vec![0.0, -1.0, 1.0, 0.0],
        forcing: vec![0.0, 0.0],
        diffusion_factor: vec![0.0, 0.0, 0.0, 0.0],
    };
    let mut residual = 0.0;
    let step = transition(&rotation, 1.3, Some(&mut residual)).unwrap();
    assert!((step.phi[0] - 1.3_f64.cos()).abs() < 1e-13);
    assert!((step.phi[2] - 1.3_f64.sin()).abs() < 1e-13);
    assert!(residual < 1e-12);
}

#[test]
fn the_kalman_likelihood_matches_the_scalar_gaussian_transition_product() {
    let ctx = ExecutionContext::for_tests(1);
    let model = OuMechanisms {
        equations: vec!["x".into()],
        drift: vec![-0.7],
        forcing: vec![0.35],
        diffusion_factor: vec![0.6],
    };
    let times = [0.4, 1.1, 1.3, 2.9, 3.0];
    let values = [0.2, 0.9, 0.35, 0.1, 0.6];
    let trajectory = OuTrajectory {
        id: "u".into(),
        schedule: vec![OuScheduleSegment { start: 0.0, replacements: vec![] }],
        initial: OuInitialLaw::Gaussian { mean: vec![0.5], covariance: vec![0.0] },
        observations: times
            .iter()
            .zip(&values)
            .map(|(t, v)| OuObservation { time: *t, values: vec![*v] })
            .collect(),
    };
    let observation = OuObservationModel::direct(1, 0.0);
    let coordinates = [OuMechanismParameter::Drift { equation: 0, parent: 0 }];
    let problem = Problem {
        template: &model,
        observation: &observation,
        coordinates: &coordinates,
        trajectories: std::slice::from_ref(&trajectory),
        n: 1,
        m: 1,
    };
    let kalman = problem.log_likelihood(&[-0.7], &ctx, None).unwrap();
    let mut expected = 0.0;
    let (mut previous, mut time) = (0.5, 0.0);
    for (t, v) in times.iter().zip(&values) {
        let h = t - time;
        let phi = (-0.7_f64 * h).exp();
        let mean = phi * previous + (0.35 / 0.7) * (1.0 - phi);
        let variance = 0.36 * (1.0 - phi * phi) / 1.4;
        expected += -0.5
            * ((2.0 * std::f64::consts::PI * variance).ln() + (v - mean) * (v - mean) / variance);
        previous = *v;
        time = *t;
    }
    assert!((kalman - expected).abs() < 1e-10, "{kalman} vs {expected}");
}

#[test]
fn a_known_intervention_schedule_changes_the_transition_at_its_switch_time() {
    let ctx = ExecutionContext::for_tests(2);
    let model = OuMechanisms {
        equations: vec!["x".into()],
        drift: vec![-1.0],
        forcing: vec![0.0],
        diffusion_factor: vec![0.5],
    };
    let schedule = vec![
        OuScheduleSegment { start: 0.0, replacements: vec![] },
        OuScheduleSegment {
            start: 1.0,
            replacements: vec![OuEquationReplacement {
                equation: "x".into(),
                drift: vec![-2.0],
                forcing: 3.0,
                diffusion_factor: vec![0.0],
            }],
        },
    ];
    let design = OuPredictionDesign {
        id: "switch".into(),
        schedule,
        initial: OuInitialLaw::Gaussian { mean: vec![2.0], covariance: vec![0.0] },
        times: vec![0.5, 1.0, 2.5],
    };
    let out = design_output(&model, &design, &ctx).unwrap();
    // Before the switch: x0 e^{-t}. After it: relax to 1.5 at rate 2 with no noise.
    let at_one = 2.0 * (-1.0_f64).exp();
    assert!((out[0] - 2.0 * (-0.5_f64).exp()).abs() < 1e-12);
    assert!((out[1] - at_one).abs() < 1e-12);
    assert!((out[2] - (1.5 + (at_one - 1.5) * (-2.0_f64 * 1.5).exp())).abs() < 1e-12);
    // Variance accrues only before the switch and then decays at twice the rate.
    let before = 0.25 * (1.0 - (-2.0_f64).exp()) / 2.0;
    assert!((out[3 + 1] - before).abs() < 1e-12);
    assert!((out[3 + 2] - before * (-4.0_f64 * 1.5).exp()).abs() < 1e-12);
}

#[test]
fn eigenvalues_match_known_spectra() {
    let mut rotation = eigenvalues(&[0.0, -2.0, 2.0, 0.0], 2).unwrap();
    rotation.sort_by(|a, b| a.1.total_cmp(&b.1));
    assert!(rotation[0].0.abs() < 1e-12 && (rotation[0].1 + 2.0).abs() < 1e-12);
    assert!(rotation[1].0.abs() < 1e-12 && (rotation[1].1 - 2.0).abs() < 1e-12);
    let mut triangular = eigenvalues(&[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0], 3).unwrap();
    triangular.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (found, wanted) in triangular.iter().zip([1.0, 4.0, 6.0]) {
        assert!((found.0 - wanted).abs() < 1e-10 && found.1.abs() < 1e-10);
    }
    let root = std::f64::consts::SQRT_2;
    let mut laplacian = eigenvalues(&[2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0], 3).unwrap();
    laplacian.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (found, wanted) in laplacian.iter().zip([2.0 - root, 2.0, 2.0 + root]) {
        assert!((found.0 - wanted).abs() < 1e-10 && found.1.abs() < 1e-10);
    }
}

#[test]
fn observability_and_aliasing_checks_follow_their_definitions() {
    let coupled = [-1.0, 0.5, 0.0, -2.0];
    let decoupled = [-1.0, 0.0, 0.0, -2.0];
    let first_only = [1.0, 0.0];
    assert_eq!(ou_observability_rank(&coupled, 2, &first_only, 1), 2);
    assert_eq!(ou_observability_rank(&decoupled, 2, &first_only, 1), 1);
    assert_eq!(ou_observability_rank(&decoupled, 2, &[1.0, 0.0, 0.0, 1.0], 2), 2);
    let oscillator = [-0.1, -4.0, 4.0, -0.1];
    let coarse = ou_aliasing_check(&oscillator, 2, Some(1.0));
    assert_eq!(coarse.aliasing_possible, Some(true));
    assert!((coarse.largest_oscillation_frequency.unwrap() - 4.0).abs() < 1e-9);
    assert_eq!(ou_aliasing_check(&oscillator, 2, Some(0.5)).aliasing_possible, Some(false));
}

#[test]
fn irregular_full_observation_recovers_the_parameters_and_predicts_held_out_interventions() {
    let ctx = ExecutionContext::for_tests(11);
    let trajectories = simulate_units(11, 4, 80, &truth(), &direct_measure);
    let observation = OuObservationModel::direct(2, 0.0);
    let coordinates = full_coordinates();
    let fit =
        fit_ou_trajectories(&start(), &observation, &coordinates, &trajectories, policy(), &ctx)
            .unwrap();
    assert!(fit.fitting.converged, "score {}", fit.fitting.score_norm);
    assert_recovered(&fit, &(0..9).collect::<Vec<_>>());
    assert_eq!(fit.identification.status, OuIdentificationStatus::Identified);
    assert_eq!(fit.identification.observability_rank, 2);
    assert!(fit.identification.unobserved_components.is_empty());
    assert_eq!(fit.numerics.drift_stable, Some(true));
    assert!(fit.numerics.expm_reliable, "residual {}", fit.numerics.expm_inverse_residual);
    assert_eq!(fit.observation_count, 320);

    // The Gauss-Newton information agrees with the central-difference observed information
    // at the optimum up to sampling variability of the zero-expectation terms.
    let problem = Problem {
        template: &start(),
        observation: &observation,
        coordinates: &coordinates,
        trajectories: &trajectories,
        n: 2,
        m: 2,
    };
    let observed = problem.information(&fit.values, fit.log_likelihood, &ctx).unwrap();
    let (_, _, analytic) = problem.score_and_information(&fit.values, &ctx).unwrap();
    for i in 0..9 {
        let ratio = analytic[i * 9 + i] / observed[i * 9 + i];
        assert!((0.5..2.0).contains(&ratio), "information diagonal {i}: ratio {ratio}");
    }

    // Held-out: replace equation a at t = 1; the fit never saw this intervention.
    let replacement = OuEquationReplacement {
        equation: "a".into(),
        drift: vec![-1.5, 0.0],
        forcing: 0.4,
        diffusion_factor: vec![0.3, 0.0],
    };
    let mean0 = vec![0.3, -0.2];
    let cov0 = vec![0.2, 0.05, 0.05, 0.1];
    let times = vec![0.5, 1.0, 1.5, 2.5, 4.0];
    let design = OuPredictionDesign {
        id: "hold-out".into(),
        schedule: vec![
            OuScheduleSegment { start: 0.0, replacements: vec![] },
            OuScheduleSegment { start: 1.0, replacements: vec![replacement] },
        ],
        initial: OuInitialLaw::Gaussian { mean: mean0.clone(), covariance: cov0.clone() },
        times: times.clone(),
    };
    let predictions = predict_ou_trajectories(&fit, &observation, &[design], &ctx).unwrap();
    let prediction = &predictions[0];
    let parameter_blocks = prediction.state_mean_parameter_covariance.as_ref().unwrap();

    let held_out_model = with_row(&truth(), 0, &[-1.5, 0.0], 0.4, &[0.3, 0.0]);
    let dt = 0.004;
    let phases = [(0_u64, truth()), (250_u64, held_out_model)];
    let observe: Vec<u64> = times.iter().map(|t| (t / dt).round() as u64).collect();
    let l00 = 0.2_f64.sqrt();
    let l10 = 0.05 / l00;
    let l11 = (0.1 - l10 * l10).sqrt();
    let replicates = 2500;
    let mut rng = RngFactory::from_seed(99).stream(0x6865_6c64);
    let mut sums = vec![vec![0.0; 2]; times.len()];
    let mut squares = vec![vec![0.0; 2]; times.len()];
    for _ in 0..replicates {
        let (z0, z1) = (standard_normal(&mut rng), standard_normal(&mut rng));
        let x0 = [mean0[0] + l00 * z0, mean0[1] + l10 * z0 + l11 * z1];
        let path = euler(&phases, &x0, &observe, dt, &mut rng);
        for (t, state) in path.iter().enumerate() {
            for c in 0..2 {
                sums[t][c] += state[c];
                squares[t][c] += state[c] * state[c];
            }
        }
    }
    for t in 0..times.len() {
        for c in 0..2 {
            let mc_mean = sums[t][c] / replicates as f64;
            let mc_var = squares[t][c] / replicates as f64 - mc_mean * mc_mean;
            let process_var = prediction.state_covariance[t][c * 2 + c];
            let parameter_var = parameter_blocks[t][c * 2 + c];
            assert!(parameter_var > 0.0, "propagated sampling variance must be positive");
            let slack = 4.0 * (parameter_var + process_var / replicates as f64).sqrt() + 0.02;
            assert!(
                (prediction.state_mean[t][c] - mc_mean).abs() <= slack,
                "t {t} c {c}: predicted {} simulated {mc_mean} slack {slack}",
                prediction.state_mean[t][c]
            );
            let covariance_sampling =
                prediction.state_covariance_parameter_variance.as_ref().unwrap()[t][c * 2 + c];
            let variance_slack = 4.0
                * (covariance_sampling.sqrt() + process_var * (2.0 / replicates as f64).sqrt())
                + 0.01;
            assert!(
                (process_var - mc_var).abs() <= variance_slack,
                "t {t} c {c}: process variance {process_var} vs simulated {mc_var} slack {variance_slack}"
            );
        }
    }
    // Observation space carries both uncertainty meanings separately.
    assert_eq!(prediction.observation_mean[0], prediction.state_mean[0]);
    assert!(prediction.observation_mean_parameter_variance.is_some());

    // A sustained replacement: the stationary law and its sampling uncertainty.
    let world = OuStationaryDesign {
        id: "sustained".into(),
        replacements: vec![OuEquationReplacement {
            equation: "a".into(),
            drift: vec![-1.5, 0.0],
            forcing: 0.4,
            diffusion_factor: vec![0.3, 0.0],
        }],
    };
    let unstable = OuStationaryDesign {
        id: "unstable".into(),
        replacements: vec![OuEquationReplacement {
            equation: "a".into(),
            drift: vec![1.0, 0.0],
            forcing: 0.0,
            diffusion_factor: vec![0.3, 0.0],
        }],
    };
    let laws = predict_ou_stationary(&fit, &[world, unstable], &ctx).unwrap();
    // a is pinned: mean 0.4 / 1.5, variance 0.09 / 3; b follows from b' = -0.4 a - 0.8 b - 0.3.
    assert!((laws[0].mean[0] - 0.4 / 1.5).abs() < 1e-9);
    assert!((laws[0].covariance[0] - 0.09 / 3.0).abs() < 1e-9);
    assert!(laws[0].mean_parameter_covariance.is_some());
    assert!(laws[1].failure.is_some());
}

#[test]
fn partial_observation_recovers_the_parameters_and_reports_the_unobserved_component() {
    let ctx = ExecutionContext::for_tests(21);
    let trajectories =
        simulate_units(21, 5, 120, &truth(), &|x, rng| vec![x[0] + 0.05 * standard_normal(rng)]);
    let observation = OuObservationModel {
        matrix: vec![1.0, 0.0],
        offset: vec![],
        noise_standard_deviation: vec![0.05],
    };
    // One observed channel of a two-state process determines four things once the diffusion
    // entry of the observed equation is known (its direct noise fixes the cusp of the
    // autocovariance): the mean, the two decay rates and one amplitude. The four drift entries
    // are identified; adding the forcing of `a` (or the diffusion entry) makes the information
    // exactly singular, which the fit refuses. Everything else is fixed design.
    let coordinates = vec![
        OuMechanismParameter::Drift { equation: 0, parent: 0 },
        OuMechanismParameter::Drift { equation: 0, parent: 1 },
        OuMechanismParameter::Drift { equation: 1, parent: 0 },
        OuMechanismParameter::Drift { equation: 1, parent: 1 },
    ];
    let mut template = start();
    template.forcing = vec![0.6, -0.3];
    template.diffusion_factor = vec![0.5, 0.0, 0.2, 0.4];
    let fit =
        fit_ou_trajectories(&template, &observation, &coordinates, &trajectories, policy(), &ctx)
            .unwrap();
    assert!(fit.fitting.converged, "score {}", fit.fitting.score_norm);
    assert_recovered(&fit, &[0, 1, 2, 3]);
    assert_eq!(fit.identification.unobserved_components, vec!["b".to_owned()]);
    assert_eq!(fit.identification.observability_rank, 2);
    assert_eq!(fit.identification.unobservable_dimension, 0);
    assert_ne!(fit.identification.status, OuIdentificationStatus::NotIdentified);
    assert_eq!(fit.filtered_states.len(), 5);
    // The unobserved component is reported with its own filtered uncertainty.
    for state in &fit.filtered_states {
        assert!(state.standard_deviation[1] > state.standard_deviation[0]);
    }
}

#[test]
fn snapshots_with_no_transition_are_refused_as_unidentified() {
    let ctx = ExecutionContext::for_tests(31);
    let mut trajectories = simulate_units(31, 6, 3, &truth(), &direct_measure);
    for trajectory in &mut trajectories {
        trajectory.observations.truncate(1);
    }
    let error = fit_ou_trajectories(
        &start(),
        &OuObservationModel::direct(2, 0.0),
        &full_coordinates(),
        &trajectories,
        policy(),
        &ctx,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OuTrajectoryError::Unidentified {
            failure: OuIdentificationFailure::NoTransitionInformation,
            ..
        }
    ));
}

#[test]
fn a_coordinate_of_a_decoupled_unobserved_state_is_refused_as_unidentified() {
    let ctx = ExecutionContext::for_tests(41);
    let decoupled = OuMechanisms {
        equations: vec!["a".into(), "b".into()],
        drift: vec![-1.0, 0.0, 0.0, -2.0],
        forcing: vec![0.5, 0.0],
        diffusion_factor: vec![0.5, 0.0, 0.0, 0.3],
    };
    let trajectories = simulate_units(41, 3, 60, &decoupled, &|x, _| vec![x[0]]);
    let observation = OuObservationModel {
        matrix: vec![1.0, 0.0],
        offset: vec![],
        noise_standard_deviation: vec![0.0],
    };
    let coordinates = vec![
        OuMechanismParameter::Drift { equation: 0, parent: 0 },
        OuMechanismParameter::Forcing { equation: 0 },
        OuMechanismParameter::Drift { equation: 1, parent: 1 },
    ];
    let mut template = decoupled.clone();
    template.drift[0] = -0.7;
    template.forcing[0] = 0.3;
    let error =
        fit_ou_trajectories(&template, &observation, &coordinates, &trajectories, policy(), &ctx)
            .unwrap_err();
    assert!(
        matches!(
            error,
            OuTrajectoryError::Unidentified {
                failure: OuIdentificationFailure::SingularInformation,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn a_fit_stopped_by_its_iteration_bound_reports_non_convergence_separately() {
    let ctx = ExecutionContext::for_tests(51);
    let trajectories = simulate_units(51, 2, 40, &truth(), &direct_measure);
    let bounded = CyclicGradientPolicy { max_iterations: 1, ..policy() };
    let fit = fit_ou_trajectories(
        &start(),
        &OuObservationModel::direct(2, 0.0),
        &full_coordinates(),
        &trajectories,
        bounded,
        &ctx,
    )
    .unwrap();
    assert!(!fit.fitting.converged);
    assert_eq!(fit.fitting.iterations, 1);
    // Numerical evidence is still computed and is a separate block from the fit status.
    assert!(fit.numerics.expm_reliable);
    assert!(fit.numerics.spectral_abscissa.is_some());
}

#[test]
fn malformed_designs_are_invalid_not_unidentified() {
    let ctx = ExecutionContext::for_tests(61);
    let mut trajectories = simulate_units(61, 1, 5, &truth(), &direct_measure);
    trajectories[0].observations[2].time = trajectories[0].observations[1].time;
    let error = fit_ou_trajectories(
        &start(),
        &OuObservationModel::direct(2, 0.0),
        &full_coordinates(),
        &trajectories,
        policy(),
        &ctx,
    )
    .unwrap_err();
    assert!(matches!(error, OuTrajectoryError::Invalid(_)));
}

fn matrix_rows(value: &serde_json::Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect()
}

fn numbers(value: &serde_json::Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

/// The committed brute-force joint Gaussian densities (NumPy/SciPy) of retained data.
#[test]
fn the_kalman_likelihood_matches_the_committed_joint_gaussian_reference() {
    let document: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/gcm/ou_trajectory_likelihood/expected.json"
    ))
    .unwrap();
    let ctx = ExecutionContext::for_tests(71);
    for case in document["cases"].as_array().unwrap() {
        let model = OuMechanisms {
            equations: vec!["a".into(), "b".into()],
            drift: matrix_rows(&case["drift"]),
            forcing: numbers(&case["forcing"]),
            diffusion_factor: matrix_rows(&case["diffusion_factor"]),
        };
        let observation = OuObservationModel {
            matrix: matrix_rows(&case["observation"]),
            offset: vec![],
            noise_standard_deviation: numbers(&case["noise_standard_deviation"]),
        };
        let trajectories: Vec<OuTrajectory> = case["units_data"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(k, unit)| OuTrajectory {
                id: format!("unit-{k}"),
                schedule: vec![OuScheduleSegment { start: 0.0, replacements: vec![] }],
                initial: OuInitialLaw::Stationary,
                observations: numbers(&unit["times"])
                    .into_iter()
                    .zip(unit["values"].as_array().unwrap())
                    .map(|(time, values)| OuObservation { time, values: numbers(values) })
                    .collect(),
            })
            .collect();
        let m = observation.validate(2).unwrap();
        let coordinates = [OuMechanismParameter::Drift { equation: 0, parent: 0 }];
        let problem = Problem {
            template: &model,
            observation: &observation,
            coordinates: &coordinates,
            trajectories: &trajectories,
            n: 2,
            m,
        };
        let got = problem.log_likelihood(&[model.drift[0]], &ctx, None).unwrap();
        let want = case["log_likelihood"].as_f64().unwrap();
        assert!(
            (got - want).abs() <= 1e-8 * want.abs().max(1.0),
            "{}: {got} vs {want}",
            case["case"]
        );
    }
}

#[test]
fn the_analytic_score_matches_the_central_difference_score_and_information_is_psd() {
    let ctx = ExecutionContext::for_tests(81);
    let trajectories = simulate_units(81, 2, 30, &truth(), &direct_measure);
    let observation = OuObservationModel::direct(2, 0.0);
    let coordinates = full_coordinates();
    // A schedule with an intervention exercises the zero derivative on replaced rows.
    let mut with_switch = trajectories.clone();
    with_switch[1].schedule.push(OuScheduleSegment {
        start: 3.0,
        replacements: vec![OuEquationReplacement {
            equation: "b".into(),
            drift: vec![0.0, -1.2],
            forcing: 0.2,
            diffusion_factor: vec![0.0, 0.3],
        }],
    });
    for (label, data) in [("stationary", &trajectories), ("switched", &with_switch)] {
        let template = start();
        let problem = Problem {
            template: &template,
            observation: &observation,
            coordinates: &coordinates,
            trajectories: data,
            n: 2,
            m: 2,
        };
        let theta = parameters(&template, &coordinates).unwrap();
        let (value, score, info) = problem.score_and_information(&theta, &ctx).unwrap();
        assert!((value - problem.log_likelihood(&theta, &ctx, None).unwrap()).abs() < 1e-9);
        let (numeric, _) = problem.gradient(&theta, &ctx).unwrap();
        for i in 0..9 {
            assert!(
                (score[i] - numeric[i]).abs() <= 1e-5 * (1.0 + numeric[i].abs()),
                "{label} coordinate {i}: analytic {} numeric {}",
                score[i],
                numeric[i]
            );
            assert!(info[i * 9 + i] >= 0.0);
            for j in 0..9 {
                assert!((info[i * 9 + j] - info[j * 9 + i]).abs() < 1e-9 * (1.0 + info[i * 9 + j].abs()));
            }
        }
    }
}
