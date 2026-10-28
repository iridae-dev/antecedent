use super::*;
use crate::mechanism_expr::MechanismExpression as E;

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
fn plain(expression: E) -> SdeDiffusion {
    SdeDiffusion { expression, square_root: false }
}
fn named(name: &str, value: f64) -> SdeParameter {
    SdeParameter { name: name.into(), value }
}
fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

/// The documented SIR system: nonlinear transmission `beta (1 + a u(t)) S I / N`, a seasonal
/// input, sqrt-type state-dependent noise and reflecting bounds on `[0, N]`.
fn epidemic() -> SdeSystem {
    let flux =
        mul(vec![par(0), add(vec![c(1.0), mul(vec![par(1), var(3)])]), var(0), var(1), c(0.001)]);
    let recovery = mul(vec![par(2), var(1)]);
    SdeSystem {
        variables: vec!["S".into(), "I".into(), "R".into()],
        inputs: vec![SdeInput {
            name: "season".into(),
            interpolation: SdeInterpolation::Linear,
            times: vec![0.0, 5.0, 10.0, 15.0, 20.0],
            values: vec![-1.0, 0.0, 1.0, 0.0, -1.0],
            envelope: Some([-1.0, 1.0]),
        }],
        parameters: vec![named("beta", 0.6), named("amplitude", 0.3), named("gamma", 0.25)],
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

fn epidemic_policy() -> SdeSimulationPolicy {
    SdeSimulationPolicy {
        start: 0.0,
        step: 0.05,
        steps: 400,
        output_every: 80,
        scheme: SdeScheme::EulerMaruyama,
        paths: 200,
        seed: 20_260_101,
        weak_tolerance: 0.05,
        check_step: true,
        retain_final_samples: false,
    }
}

fn lockdown() -> SdeWorld {
    SdeWorld {
        id: "lockdown".into(),
        actions: vec![],
        inputs: vec![SdeInput {
            name: "season".into(),
            interpolation: SdeInterpolation::PiecewiseConstant,
            times: vec![0.0],
            values: vec![-1.0],
            envelope: Some([-1.0, 1.0]),
        }],
    }
}

fn suppress() -> SdeWorld {
    SdeWorld {
        id: "suppress".into(),
        actions: vec![SdeAction { variable: 1, times: vec![5.0], values: vec![20.0] }],
        inputs: vec![],
    }
}

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../../conformance/gcm/nonlinear_sde_ensemble/expected.json"
    ))
    .unwrap()
}

fn grid(value: &serde_json::Value) -> Vec<Vec<f64>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect()
}

fn numbers(value: &serde_json::Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn assert_grid(actual: &[Vec<f64>], expected: &[Vec<f64>], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} rows");
    for (t, (a, e)) in actual.iter().zip(expected).enumerate() {
        for (i, (x, y)) in a.iter().zip(e).enumerate() {
            assert!((x - y).abs() <= 1e-8 * (1.0 + y.abs()), "{label}[{t}][{i}]: {x} vs {y}");
        }
    }
}

#[test]
fn counter_normals_match_the_independent_generator() {
    let document = reference();
    for case in document["rng"]["cases"].as_array().unwrap() {
        let value = counter_normal(
            case["seed"].as_u64().unwrap(),
            case["path"].as_u64().unwrap(),
            case["step"].as_u64().unwrap(),
            case["component"].as_u64().unwrap(),
        );
        assert!((value - case["value"].as_f64().unwrap()).abs() < 1e-12, "{case}");
    }
}

#[test]
fn boundaries_reflect_and_clamp_and_inputs_interpolate() {
    let reflect =
        SdeBoundary { lower: Some(0.0), upper: Some(10.0), rule: SdeBoundaryRule::Reflect };
    assert_eq!(reflect.apply(-3.0), 3.0);
    assert_eq!(reflect.apply(12.0), 8.0);
    assert_eq!(reflect.apply(23.0), 3.0);
    let clamp = SdeBoundary { lower: Some(0.0), upper: None, rule: SdeBoundaryRule::Clamp };
    assert_eq!(clamp.apply(-3.0), 0.0);
    assert_eq!(clamp.apply(4.0), 4.0);
    let one_sided = SdeBoundary { lower: Some(1.0), upper: None, rule: SdeBoundaryRule::Reflect };
    assert_eq!(one_sided.apply(-1.0), 3.0);
    let system = epidemic();
    let linear = &system.inputs[0];
    assert_eq!(linear.value_at(-5.0), -1.0);
    assert!((linear.value_at(2.5) + 0.5).abs() < 1e-15);
    assert_eq!(linear.value_at(99.0), -1.0);
    let steps = SdeInput {
        name: "u".into(),
        interpolation: SdeInterpolation::PiecewiseConstant,
        times: vec![0.0, 1.0],
        values: vec![3.0, 4.0],
        envelope: None,
    };
    assert_eq!(steps.value_at(0.999), 3.0);
    assert_eq!(steps.value_at(1.0), 4.0);
}

#[test]
fn the_epidemic_system_is_certified_with_a_degenerate_square_root_diffusion() {
    let compiled = epidemic().compile().unwrap();
    let certificate = compiled.certificate();
    // Row sums of |d drift / d state| over S, I in [0, N]: 0.78 + 0.78 for the S row.
    assert!(
        certificate.drift_lipschitz > 1.55 && certificate.drift_lipschitz < 1.6,
        "{}",
        certificate.drift_lipschitz
    );
    assert_eq!(certificate.diffusion_lipschitz, None);
    assert_eq!(certificate.square_root_degenerate, vec![0, 1, 2]);
    assert!(!certificate.diagonal_milstein_admissible);
}

#[test]
fn unbounded_nonlinear_growth_is_refused_with_a_typed_reason() {
    let system = SdeSystem {
        variables: vec!["x".into()],
        inputs: vec![],
        parameters: vec![named("s", 0.1)],
        drift: vec![mul(vec![var(0), var(0)])],
        diffusion: vec![plain(par(0))],
        boundaries: vec![],
    };
    match system.compile() {
        Err(SdeError::Unsupported { code, .. }) => assert_eq!(code, "sde.unbounded_growth"),
        other => panic!("{other:?}"),
    }
    // The same nonlinearity on a declared box is certified.
    let mut boxed = system;
    boxed.boundaries =
        vec![SdeBoundary { lower: Some(-2.0), upper: Some(2.0), rule: SdeBoundaryRule::Reflect }];
    let certificate = boxed.compile().unwrap().certificate().clone();
    assert!(certificate.drift_lipschitz >= 4.0 && certificate.drift_lipschitz < 4.01);
}

#[test]
fn milstein_is_refused_when_a_diffusion_depends_on_another_state() {
    let compiled = epidemic().compile().unwrap();
    let mut policy = epidemic_policy();
    policy.scheme = SdeScheme::Milstein;
    match simulate_sde(&compiled, None, &[950.0, 50.0, 0.0], &policy, &[], &[], &ctx()) {
        Err(SdeError::Unsupported { code, .. }) => assert_eq!(code, "sde.milstein_noncommutative"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn input_and_action_declarations_are_checked_against_the_certificate_and_the_grid() {
    let compiled = epidemic().compile().unwrap();
    let initial = [950.0, 50.0, 0.0];
    let mut wide = lockdown();
    wide.inputs[0].values = vec![2.0];
    wide.inputs[0].envelope = Some([-1.0, 2.0]);
    match simulate_sde(&compiled, None, &initial, &epidemic_policy(), &[wide], &[], &ctx()) {
        Err(SdeError::Unsupported { code, .. }) => {
            assert_eq!(code, "sde.input_outside_certified_envelope");
        }
        other => panic!("{other:?}"),
    }
    let mut off_grid = suppress();
    off_grid.actions[0].times = vec![5.013];
    match simulate_sde(&compiled, None, &initial, &epidemic_policy(), &[off_grid], &[], &ctx()) {
        Err(SdeError::Unsupported { code, .. }) => assert_eq!(code, "sde.action_time_off_grid"),
        other => panic!("{other:?}"),
    }
    let mut outside = suppress();
    outside.actions[0].values = vec![5000.0];
    assert!(matches!(
        simulate_sde(&compiled, None, &initial, &epidemic_policy(), &[outside], &[], &ctx()),
        Err(SdeError::Invalid(_))
    ));
    assert!(matches!(
        simulate_sde(&compiled, None, &[2000.0, 50.0, 0.0], &epidemic_policy(), &[], &[], &ctx()),
        Err(SdeError::Invalid(_))
    ));
}

#[test]
fn the_epidemic_ensemble_replays_the_independent_numpy_reference() {
    let document = reference();
    let compiled = epidemic().compile().unwrap();
    let contrasts = [
        SdeContrast {
            id: "lockdown_vs_baseline".into(),
            treated: "lockdown".into(),
            control: "baseline".into(),
        },
        SdeContrast {
            id: "suppress_vs_baseline".into(),
            treated: "suppress".into(),
            control: "baseline".into(),
        },
    ];
    let report = simulate_sde(
        &compiled,
        None,
        &[950.0, 50.0, 0.0],
        &epidemic_policy(),
        &[lockdown(), suppress()],
        &contrasts,
        &ctx(),
    )
    .unwrap();
    assert_eq!(report.times, vec![0.0, 4.0, 8.0, 12.0, 16.0, 20.0]);
    for world in &report.worlds {
        let expected = &document["epidemic"]["worlds"][world.id.as_str()];
        assert_grid(&world.mean, &grid(&expected["mean"]), &format!("{} mean", world.id));
        assert_grid(
            &world.variance,
            &grid(&expected["variance"]),
            &format!("{} variance", world.id),
        );
        let numerical = world.numerical.as_ref().unwrap();
        assert_grid(
            &numerical.weak_difference,
            &grid(&expected["weak_difference"]),
            &format!("{} weak difference", world.id),
        );
    }
    for contrast in &report.contrasts {
        let expected = &document["epidemic"]["contrasts"][contrast.id.as_str()];
        assert_grid(
            &contrast.mean_difference,
            &grid(&expected["mean_difference"]),
            &format!("{} contrast", contrast.id),
        );
        assert_grid(
            &contrast.monte_carlo_standard_error,
            &grid(&expected["standard_error"]),
            &format!("{} contrast standard error", contrast.id),
        );
    }
    // Scientific content of the interventions: a lockdown lowers the final recovered count
    // and a held low prevalence leaves more susceptibles, each by many standard errors.
    let last = report.times.len() - 1;
    let lock = &report.contrasts[0];
    assert!(
        lock.mean_difference[last][2] < -50.0 * lock.monte_carlo_standard_error[last][2].max(1.0)
    );
    let held = &report.contrasts[1];
    assert!(held.mean_difference[last][0] > 100.0);
}

#[test]
fn held_components_are_exact_and_share_noise_with_the_baseline_until_the_action() {
    let compiled = epidemic().compile().unwrap();
    let report = simulate_sde(
        &compiled,
        None,
        &[950.0, 50.0, 0.0],
        &epidemic_policy(),
        &[suppress()],
        &[],
        &ctx(),
    )
    .unwrap();
    let (base, held) = (&report.worlds[0], &report.worlds[1]);
    // Output times 0 and 4 precede the action at 5: identical bit for bit.
    assert_eq!(base.mean[1], held.mean[1]);
    assert_eq!(base.mean[0], held.mean[0]);
    for t in 2..report.times.len() {
        assert_eq!(held.mean[t][1], 20.0);
        assert_eq!(held.variance[t][1], 0.0);
    }
}

#[test]
fn simulation_replays_exactly_and_the_seed_changes_the_ensemble() {
    let compiled = epidemic().compile().unwrap();
    let run = |seed: u64| {
        let mut policy = epidemic_policy();
        policy.seed = seed;
        policy.paths = 40;
        simulate_sde(&compiled, None, &[950.0, 50.0, 0.0], &policy, &[lockdown()], &[], &ctx())
            .unwrap()
    };
    assert_eq!(run(11), run(11));
    assert_ne!(run(11).worlds[0].mean, run(12).worlds[0].mean);
}

fn cir_system() -> SdeSystem {
    SdeSystem {
        variables: vec!["x".into()],
        inputs: vec![SdeInput {
            name: "forcing".into(),
            interpolation: SdeInterpolation::PiecewiseConstant,
            times: vec![0.0, 5.0],
            values: vec![1.0, 2.0],
            envelope: None,
        }],
        parameters: vec![named("a", 1.5), named("b", 0.5), named("sigma", 0.5)],
        drift: vec![add(vec![mul(vec![par(0), var(1)]), neg(mul(vec![par(1), var(0)]))])],
        diffusion: vec![sqrt_of(mul(vec![par(2), par(2), var(0)]))],
        boundaries: vec![SdeBoundary {
            lower: Some(0.0),
            upper: None,
            rule: SdeBoundaryRule::Reflect,
        }],
    }
}

/// Exact mean and variance of the Euler-Maruyama scheme itself: the drift is linear and the
/// noise variance is linear in the state, so both obey closed recursions.
fn em_moments(dt: f64, steps: usize, every: usize) -> (Vec<f64>, Vec<f64>) {
    let (a, b, sigma) = (1.5, 0.5, 0.5);
    let (mut m, mut v) = (10.0, 0.0);
    let (mut means, mut variances) = (vec![m], vec![v]);
    for n in 0..steps {
        let t = n as f64 * dt;
        let u = if t >= 5.0 { 2.0 } else { 1.0 };
        let next_v = (1.0 - b * dt) * (1.0 - b * dt) * v + sigma * sigma * m * dt;
        m += (a * u - b * m) * dt;
        v = next_v;
        if (n + 1) % every == 0 {
            means.push(m);
            variances.push(v);
        }
    }
    (means, variances)
}

#[test]
fn square_root_diffusion_matches_the_exact_moment_odes_with_monte_carlo_tolerance() {
    let document = reference();
    let exact_mean = numbers(&document["cir"]["exact_mean"]);
    let compiled = cir_system().compile().unwrap();
    assert_eq!(compiled.certificate().square_root_degenerate, vec![0]);
    assert_eq!(compiled.certificate().diffusion_lipschitz, None);
    let policy = SdeSimulationPolicy {
        start: 0.0,
        step: 0.05,
        steps: 200,
        output_every: 40,
        scheme: SdeScheme::EulerMaruyama,
        paths: 4000,
        seed: 7,
        weak_tolerance: 0.05,
        check_step: true,
        retain_final_samples: false,
    };
    let report = simulate_sde(&compiled, None, &[10.0], &policy, &[], &[], &ctx()).unwrap();
    let world = &report.worlds[0];
    let (coarse_mean, coarse_var) = em_moments(0.05, 200, 40);
    let (fine_mean, _) = em_moments(0.025, 400, 80);
    let numerical = world.numerical.as_ref().unwrap();
    assert_eq!(numerical.refined_step, 0.025);
    for t in 0..exact_mean.len() {
        let se = world.monte_carlo_standard_error[t][0];
        // The ensemble reproduces the scheme's own mean within Monte Carlo error ...
        assert!((world.mean[t][0] - coarse_mean[t]).abs() <= 4.5 * se + 1e-12, "mean {t}");
        // ... which is within a small, step-controlled distance of the exact mean-path ODE,
        assert!((coarse_mean[t] - exact_mean[t]).abs() < 0.06, "scheme bias {t}");
        // and halving the step reduces that distance.
        assert!(
            (fine_mean[t] - exact_mean[t]).abs() <= (coarse_mean[t] - exact_mean[t]).abs() + 1e-12,
            "halving {t}"
        );
        let expected_difference = coarse_mean[t] - fine_mean[t];
        let spread = numerical.weak_difference_standard_error[t][0];
        assert!(
            (numerical.weak_difference[t][0] - expected_difference).abs() <= 4.5 * spread + 1e-12,
            "weak difference {t}"
        );
        if t > 0 {
            let tolerance = 6.0 * coarse_var[t] * (2.0 / (policy.paths as f64 - 1.0)).sqrt();
            assert!((world.variance[t][0] - coarse_var[t]).abs() <= tolerance, "variance {t}");
        }
    }
    // Monte Carlo error is reported and is far smaller than the process spread.
    assert!(world.monte_carlo_standard_error[5][0] < 0.1 * world.variance[5][0].sqrt());
    let disabled = SdeSimulationPolicy { check_step: false, ..policy };
    let plain = simulate_sde(&compiled, None, &[10.0], &disabled, &[], &[], &ctx()).unwrap();
    assert!(plain.worlds[0].numerical.is_none());
    assert_eq!(plain.worlds[0].mean, world.mean);
}

#[test]
fn paired_contrasts_have_a_smaller_standard_error_than_independent_arms() {
    let compiled = cir_system().compile().unwrap();
    let boost = SdeWorld {
        id: "boost".into(),
        actions: vec![],
        inputs: vec![SdeInput {
            name: "forcing".into(),
            interpolation: SdeInterpolation::PiecewiseConstant,
            times: vec![0.0],
            values: vec![2.0],
            envelope: Some([1.0, 2.0]),
        }],
    };
    let policy = SdeSimulationPolicy {
        start: 0.0,
        step: 0.05,
        steps: 200,
        output_every: 200,
        scheme: SdeScheme::EulerMaruyama,
        paths: 1000,
        seed: 41,
        weak_tolerance: 0.05,
        check_step: false,
        retain_final_samples: true,
    };
    let contrast =
        SdeContrast { id: "effect".into(), treated: "boost".into(), control: "baseline".into() };
    let report =
        simulate_sde(&compiled, None, &[10.0], &policy, &[boost], &[contrast], &ctx()).unwrap();
    let paired = report.contrasts[0].monte_carlo_standard_error[1][0];
    let independent = (report.worlds[0].monte_carlo_standard_error[1][0].powi(2)
        + report.worlds[1].monte_carlo_standard_error[1][0].powi(2))
    .sqrt();
    assert!(paired < 0.8 * independent, "paired {paired} vs independent {independent}");
    // The joint response law is retained path by path, in the same order in both worlds.
    let samples = report.worlds[1].final_samples.as_ref().unwrap();
    assert_eq!(samples.len(), 1000);
    let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / 1000.0;
    assert!((mean - report.worlds[1].mean[1][0]).abs() < 1e-9);
}

#[test]
fn milstein_has_lower_strong_error_than_euler_on_geometric_brownian_motion() {
    let (mu, sigma, x0) = (0.1, 0.8, 1.0);
    let system = SdeSystem {
        variables: vec!["x".into()],
        inputs: vec![],
        parameters: vec![named("mu", mu), named("sigma", sigma)],
        drift: vec![mul(vec![par(0), var(0)])],
        diffusion: vec![plain(mul(vec![par(1), var(0)]))],
        boundaries: vec![],
    };
    let compiled = system.compile().unwrap();
    assert!(compiled.certificate().diagonal_milstein_admissible);
    let steps = 50;
    let mut policy = SdeSimulationPolicy {
        start: 0.0,
        step: 0.02,
        steps,
        output_every: steps,
        scheme: SdeScheme::EulerMaruyama,
        paths: 400,
        seed: 3,
        weak_tolerance: 0.05,
        check_step: false,
        retain_final_samples: false,
    };
    let world = SdeWorld { id: "w".into(), ..SdeWorld::default() };
    let mut errors = Vec::new();
    for scheme in [SdeScheme::EulerMaruyama, SdeScheme::Milstein] {
        policy.scheme = scheme;
        let paths = sample_sde_paths(&compiled, None, &[x0], &policy, &world, &ctx()).unwrap();
        let mut squared = 0.0;
        for p in 0..policy.paths {
            // The coarse increment is the sum of the two half-step normals, scaled.
            let w: f64 = (0..steps)
                .map(|n| {
                    (policy.step / 2.0).sqrt()
                        * (counter_normal(policy.seed, p as u64, 2 * n as u64, 0)
                            + counter_normal(policy.seed, p as u64, 2 * n as u64 + 1, 0))
                })
                .sum();
            let exact = x0 * ((mu - 0.5 * sigma * sigma) * 1.0 + sigma * w).exp();
            let simulated = paths.states[p * 2 + 1];
            squared += (simulated - exact).powi(2);
        }
        errors.push((squared / policy.paths as f64).sqrt());
    }
    assert!(errors[1] < 0.4 * errors[0], "euler {} milstein {}", errors[0], errors[1]);
}

#[test]
fn work_and_memory_budgets_are_refused_before_simulating() {
    let compiled = cir_system().compile().unwrap();
    let mut policy = epidemic_policy();
    policy.steps = 1 << 19;
    policy.output_every = 1 << 19;
    policy.paths = 4000;
    assert!(matches!(
        simulate_sde(&compiled, None, &[10.0], &policy, &[], &[], &ctx()),
        Err(SdeError::Resource(_))
    ));
}
