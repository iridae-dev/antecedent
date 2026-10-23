use super::*;
use antecedent_core::RngFactory;
use antecedent_core::execution::AdmissionPool;
use antecedent_model::nonlinear_equilibrium::{Activation, Equation};

const COORDINATES: [OuMechanismParameter; 3] = [
    OuMechanismParameter::Drift { equation: 0, parent: 1 },
    OuMechanismParameter::Drift { equation: 1, parent: 0 },
    OuMechanismParameter::Forcing { equation: 1 },
];
const TRUTH: [f64; 3] = [0.3, -0.2, 0.7];
/// Stationary mean of `a` once `b` is replaced by `db = (0.4 - b)dt + 0.3 dW`.
const UNSEEN_MEAN: f64 = 1.0 + 0.3 * 0.4;
const INDEPENDENT: DeclaredSamplingDesign = DeclaredSamplingDesign::IndependentSubjectTrajectories;

fn truth() -> OuMechanisms {
    OuMechanisms {
        equations: vec!["a".into(), "b".into()],
        drift: vec![-1.0, 0.3, -0.2, -1.0],
        forcing: vec![1.0, 0.7],
        diffusion_factor: vec![0.4, 0.1, 0.0, 0.3],
    }
}

fn replacement(
    equation: &str,
    drift: [f64; 2],
    forcing: f64,
    diffusion: [f64; 2],
) -> OuEquationReplacement {
    OuEquationReplacement {
        equation: equation.into(),
        drift: drift.to_vec(),
        forcing,
        diffusion_factor: diffusion.to_vec(),
    }
}

fn unseen() -> DynamicQuantity {
    DynamicQuantity::OuMechanismMean {
        replacements: vec![replacement("b", [0.0, -1.0], 0.4, [0.0, 0.3])],
        variable: 0,
    }
}

/// Independent Gaussian rows from a known stationary law.
fn gaussian_rows(
    mean: &[f64],
    covariance: &[f64],
    count: usize,
    rng: &mut CausalRng,
) -> Vec<Vec<f64>> {
    let n = mean.len();
    let lower = cholesky(covariance, n).unwrap();
    (0..count)
        .map(|_| {
            let z: Vec<f64> = (0..n).map(|_| standard_normal(rng)).collect();
            (0..n)
                .map(|i| mean[i] + (0..=i).map(|j| lower[i * n + j] * z[j]).sum::<f64>())
                .collect::<Vec<f64>>()
        })
        .collect()
}

fn mechanism_policy() -> CyclicGradientPolicy {
    CyclicGradientPolicy { score_tolerance: 1e-4, ..CyclicGradientPolicy::default() }
}

/// An observational regime and an `a`-equation replacement, `per_regime` rows each.
fn mechanism_problem(
    seed: u64,
    per_regime: usize,
    policy: CyclicGradientPolicy,
) -> OuMechanismInferenceProblem {
    let ctx = ExecutionContext::for_tests(seed);
    let model = truth();
    let mut rng = RngFactory::from_seed(seed).stream(0x4f55_4d45);
    let regimes = [
        ("observational", vec![]),
        ("a-equation", vec![replacement("a", [-1.0, 0.0], 0.1, [0.3, 0.0])]),
    ]
    .into_iter()
    .map(|(id, replacements)| {
        let law = model.stationary(&replacements, &ctx).unwrap();
        let values = gaussian_rows(&law.mean, &law.covariance, per_regime, &mut rng);
        OuMechanismRegime { id: id.into(), replacements, values }
    })
    .collect();
    OuMechanismInferenceProblem {
        template: crate::ou_mechanisms::bind(&model, &COORDINATES, &[0.0, 0.0, 0.3]),
        parameters: COORDINATES.to_vec(),
        regimes,
        policy,
    }
}

fn fit_mechanisms(problem: &OuMechanismInferenceProblem, ctx: &ExecutionContext) -> OuMechanismFit {
    fit_ou_mechanisms(&problem.template, &problem.parameters, &problem.regimes, problem.policy, ctx)
        .unwrap()
}

fn ratio_within(value: f64, reference: f64, low: f64, high: f64) -> bool {
    let ratio = value / reference;
    ratio > low && ratio < high
}

fn column(draws: &[Vec<f64>], j: usize) -> Vec<f64> {
    draws.iter().map(|draw| draw[j]).collect()
}

fn mean_sd(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, variance.sqrt())
}

#[test]
fn shared_mechanism_information_agrees_with_both_bootstraps() {
    let ctx = ExecutionContext::for_tests(41);
    let problem = mechanism_problem(41, 300, mechanism_policy());
    let fit = fit_mechanisms(&problem, &ctx);
    assert!(fit.converged, "{fit:?}");
    let target = DynamicFitRef::OuMechanisms { problem: &problem, fit: &fit };
    let quantities = [unseen()];
    let information = dynamic_observed_information(target, INDEPENDENT, &quantities, &ctx).unwrap();
    assert_eq!(information.regularity, InformationRegularity::Regular);
    assert_eq!(information.derivative, "central_difference_of_analytic_score");
    let se = information.standard_errors.clone().unwrap();
    let quantity_se = information.quantity_standard_errors[0].unwrap();
    for (j, truth) in TRUTH.iter().enumerate() {
        assert!((information.estimate[j] - truth).abs() < 4.0 * se[j], "{information:?}");
    }
    for method in [DynamicBootstrapMethod::Nonparametric, DynamicBootstrapMethod::Parametric] {
        let request = DynamicBootstrapRequest::whole_units(
            method,
            200,
            CyclicBootstrapDesign::FixedAssignments,
        );
        let law = bootstrap_dynamic_fit(target, request, &quantities, &ctx).unwrap();
        assert_eq!(law.replicates.len(), 200);
        assert!(law.failed_mass < 0.05, "{method:?} failed mass {}", law.failed_mass);
        let spread = law.standard_errors.clone().unwrap();
        for (replicated, observed) in spread.iter().zip(&se) {
            assert!(
                ratio_within(*replicated, *observed, 0.75, 1.33),
                "{method:?} {spread:?} vs {se:?}"
            );
        }
        let replicated = law.quantity_standard_errors[0].unwrap();
        assert!(ratio_within(replicated, quantity_se, 0.75, 1.33), "{replicated} {quantity_se}");
    }
}

#[test]
fn shared_mechanism_posterior_centres_on_and_covers_truth_across_seeds() {
    let prior = DynamicPrior { mean: vec![0.0; 3], standard_deviation: vec![2.0; 3] };
    let options = DynamicPosteriorOptions {
        chains: 4,
        warmup: 300,
        draws: 800,
        proposal_scale: None,
        start_spread: 1.0,
    };
    let (mut covered, mut checks) = (0, 0);
    for seed in [3_u64, 17, 29, 101] {
        let ctx = ExecutionContext::for_tests(seed);
        let problem = mechanism_problem(seed, 300, mechanism_policy());
        let fit = fit_mechanisms(&problem, &ctx);
        let target = DynamicFitRef::OuMechanisms { problem: &problem, fit: &fit };
        let se = dynamic_observed_information(target, INDEPENDENT, &[], &ctx)
            .unwrap()
            .standard_errors
            .unwrap();
        let posterior =
            sample_dynamic_posterior(target, &prior, options, INDEPENDENT, &[unseen()], &ctx)
                .unwrap();
        assert_eq!(posterior.preconditioner, PosteriorPreconditioner::Laplace);
        assert_eq!(posterior.parameter_draws.len(), 3200);
        assert!(posterior.max_rhat.is_some_and(|r| r < 1.05), "{:?}", posterior.max_rhat);
        assert!(posterior.min_bulk_ess > 80.0 && posterior.min_tail_ess > 80.0, "{posterior:?}");
        for (j, truth) in TRUTH.iter().enumerate() {
            let mut draws = column(&posterior.parameter_draws, j);
            let (mean, sd) = mean_sd(&draws);
            assert!((mean - truth).abs() < 4.0 * sd, "seed {seed} coordinate {j}: {mean} sd {sd}");
            // With a weak prior the posterior spread tracks the observed information.
            assert!(ratio_within(sd, se[j], 0.75, 1.33), "seed {seed} coordinate {j}");
            draws.sort_by(f64::total_cmp);
            let tail = draws.len() / 40;
            checks += 1;
            if draws[tail] <= *truth && *truth <= draws[draws.len() - 1 - tail] {
                covered += 1;
            }
        }
        let unseen_draws: Vec<f64> = posterior.quantity_draws.iter().filter_map(|d| d[0]).collect();
        assert_eq!(unseen_draws.len(), 3200);
        assert!(posterior.quantity_failed_mass[0].abs() < 1e-15);
        let (mean, sd) = mean_sd(&unseen_draws);
        assert!((mean - UNSEEN_MEAN).abs() < 4.0 * sd, "seed {seed}: {mean} sd {sd}");
    }
    assert!(covered >= checks - 3, "{covered} of {checks} central 95% intervals cover");
}

#[test]
fn stationary_ou_information_matches_closed_form_and_bootstrap_spread() {
    let ctx = ExecutionContext::for_tests(77);
    let mut rng = RngFactory::from_seed(77).stream(0x4f55);
    let count = 300_u32;
    let rows = (0..count)
        .map(|_| RegimeRow { values: vec![0.5 + standard_normal(&mut rng)], clamps: vec![] })
        .collect();
    let problem = OuInferenceProblem {
        template: OrnsteinUhlenbeck { drift: vec![-1.0], center: vec![0.0], diffusion: vec![4.0] },
        parameters: vec![
            OuParameter::Drift { row: 0, column: 0 },
            OuParameter::Center { variable: 0 },
        ],
        rows,
        options: CyclicFitOptions::default(),
    };
    let fit = fit_ou_likelihood(
        &problem.template,
        &problem.parameters,
        &problem.rows,
        problem.options,
        &ctx,
    )
    .unwrap();
    assert!(fit.converged);
    let target = DynamicFitRef::Ou { problem: &problem, fit: &fit };
    let quantities =
        [DynamicQuantity::OuStationaryCovariance { clamps: vec![], variable: 0, other: 0 }];
    let information = dynamic_observed_information(target, INDEPENDENT, &quantities, &ctx).unwrap();
    assert_eq!(information.regularity, InformationRegularity::Regular);
    assert_eq!(information.derivative, "central_second_difference_of_likelihood");
    // Scalar stationary law N(c, -D/(2M)): at the MLE the information is
    // n/(2M^2) for M and n/variance for c, with no cross term.
    let drift = fit.model.drift[0];
    let variance = -4.0 / (2.0 * drift);
    let n = f64::from(count);
    let se = information.standard_errors.clone().unwrap();
    assert!(ratio_within(se[0], drift.abs() * (2.0 / n).sqrt(), 0.97, 1.03), "{se:?}");
    assert!(ratio_within(se[1], (variance / n).sqrt(), 0.97, 1.03), "{se:?}");
    let quantity_se = information.quantity_standard_errors[0].unwrap();
    assert!(ratio_within(quantity_se, variance * (2.0 / n).sqrt(), 0.95, 1.05), "{quantity_se}");
    for method in [DynamicBootstrapMethod::Nonparametric, DynamicBootstrapMethod::Parametric] {
        let request = DynamicBootstrapRequest::whole_units(
            method,
            150,
            CyclicBootstrapDesign::FixedAssignments,
        );
        let law = bootstrap_dynamic_fit(target, request, &quantities, &ctx).unwrap();
        assert!(law.failed_mass < 0.05, "{method:?} failed mass {}", law.failed_mass);
        let spread = law.standard_errors.unwrap();
        for (replicated, observed) in spread.iter().zip(&se) {
            assert!(ratio_within(*replicated, *observed, 0.75, 1.33), "{method:?} {spread:?}");
        }
    }
}

/// `dx/dt = -k x` with true `k = 2`, eight trajectories, Gaussian measurement sd 0.05.
fn decay_problem(seed: u64) -> OdeInferenceProblem {
    let mut rng = RngFactory::from_seed(seed).stream(0x004f_4445);
    let times = vec![0.0, 0.25, 0.5, 1.0];
    let data = (0..8_u32)
        .map(|i| {
            let initial = 1.0 + f64::from(i) / 4.0;
            let values = times
                .iter()
                .enumerate()
                .map(|(t, time)| {
                    let exact = initial * (-2.0_f64 * time).exp();
                    vec![if t == 0 { exact } else { exact + 0.05 * standard_normal(&mut rng) }]
                })
                .collect();
            OdeObservation {
                initial: vec![initial],
                times: times.clone(),
                values,
                variances: vec![0.0025],
                schedule: vec![],
            }
        })
        .collect();
    OdeInferenceProblem {
        template: OdeSystem {
            equations: vec![Equation {
                id: "x".into(),
                intercept: 0.0,
                weights: vec![0.0],
                activation: Activation::Identity,
                expression: None,
                parameters: vec![],
            }],
            decay: vec![1.5],
        },
        parameters: vec![OdeParameter::Decay { equation: 0 }],
        data,
        integration: OdePolicy::default(),
        policy: OdeFitPolicy { learning_rate: 1e-3, score_tolerance: 1e-4, max_iterations: 1000 },
    }
}

#[test]
fn ode_information_matches_expected_information_bootstraps_and_posterior() {
    let ctx = ExecutionContext::for_tests(5);
    let problem = decay_problem(5);
    let fit = fit_ode_likelihood(
        &problem.template,
        &problem.parameters,
        &problem.data,
        problem.integration,
        problem.policy,
        &ctx,
    )
    .unwrap();
    assert!(fit.converged, "{fit:?}");
    let k = fit.model.decay[0];
    // Expected information sum (dx/dk)^2 / variance with dx/dk = -x0 t exp(-k t).
    let expected: f64 = problem
        .data
        .iter()
        .map(|o| {
            o.times
                .iter()
                .skip(1)
                .map(|t| (o.initial[0] * t * (-k * t).exp()).powi(2) / 0.0025)
                .sum::<f64>()
        })
        .sum();
    let quantities = [DynamicQuantity::OdeState {
        initial: vec![2.0],
        start: 0.0,
        time: 0.7,
        schedule: vec![],
        variable: 0,
    }];
    let target = DynamicFitRef::Ode { problem: &problem, fit: &fit };
    let information = dynamic_observed_information(target, INDEPENDENT, &quantities, &ctx).unwrap();
    assert_eq!(information.regularity, InformationRegularity::Regular);
    let se = information.standard_errors.clone().unwrap()[0];
    assert!(ratio_within(se, expected.sqrt().recip(), 0.9, 1.1), "{se} vs {expected}");
    // d/dk of 2 exp(-0.7 k) is -1.4 exp(-0.7 k).
    let quantity_se = information.quantity_standard_errors[0].unwrap();
    assert!(ratio_within(quantity_se, 1.4 * (-0.7 * k).exp() * se, 0.98, 1.02));
    let parametric = bootstrap_dynamic_fit(
        target,
        DynamicBootstrapRequest::whole_units(
            DynamicBootstrapMethod::Parametric,
            60,
            CyclicBootstrapDesign::FixedAssignments,
        ),
        &quantities,
        &ctx,
    )
    .unwrap();
    assert!(parametric.failed_mass < 0.05);
    assert_eq!(parametric.resampling, "simulated_independent_units");
    assert!(ratio_within(parametric.standard_errors.unwrap()[0], se, 0.7, 1.4));
    let trajectories = bootstrap_dynamic_fit(
        target,
        DynamicBootstrapRequest::whole_units(
            DynamicBootstrapMethod::Nonparametric,
            60,
            CyclicBootstrapDesign::FixedAssignments,
        ),
        &quantities,
        &ctx,
    )
    .unwrap();
    assert!(trajectories.failed_mass < 0.05);
    assert_eq!(trajectories.resampling, "row_subject_iid");
    assert!(ratio_within(trajectories.standard_errors.unwrap()[0], se, 0.5, 2.0));
    let prior = DynamicPrior { mean: vec![1.0], standard_deviation: vec![5.0] };
    let options = DynamicPosteriorOptions { warmup: 200, draws: 500, ..Default::default() };
    let posterior =
        sample_dynamic_posterior(target, &prior, options, INDEPENDENT, &quantities, &ctx).unwrap();
    let (mean, sd) = mean_sd(&column(&posterior.parameter_draws, 0));
    assert!((mean - 2.0).abs() < 4.0 * sd, "{mean} sd {sd}");
    assert!(ratio_within(sd, se, 0.75, 1.33), "{sd} vs {se}");
    assert!(posterior.quantity_draws.iter().all(|draw| draw[0].is_some()));
}

#[test]
fn replicates_do_not_depend_on_worker_count_or_replicate_total() {
    let problem = mechanism_problem(7, 60, mechanism_policy());
    let serial = ExecutionContext::for_tests(7);
    let fit = fit_mechanisms(&problem, &serial);
    assert!(fit.converged);
    let mut parallel =
        ExecutionContext::for_tests(7).with_admission(AdmissionPool::fixed(8, u64::MAX));
    parallel.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).unwrap());
    let target = DynamicFitRef::OuMechanisms { problem: &problem, fit: &fit };
    let quantities = [unseen()];
    for (method, stratification) in [
        (DynamicBootstrapMethod::Nonparametric, CyclicBootstrapDesign::RandomizedWithinTargets),
        (DynamicBootstrapMethod::Parametric, CyclicBootstrapDesign::FixedAssignments),
    ] {
        let request =
            |replicates| DynamicBootstrapRequest::whole_units(method, replicates, stratification);
        let one = bootstrap_dynamic_fit(target, request(16), &quantities, &serial).unwrap();
        let four = bootstrap_dynamic_fit(target, request(16), &quantities, &parallel).unwrap();
        assert_eq!(one, four, "{method:?}");
        let prefix = bootstrap_dynamic_fit(target, request(5), &quantities, &parallel).unwrap();
        assert_eq!(prefix.replicates[..], one.replicates[..5], "{method:?}");
        assert!(one.replicates.windows(2).any(|pair| pair[0] != pair[1]));
    }
}

#[test]
fn trajectory_design_refuses_row_draws_and_dependent_series() {
    let ctx = ExecutionContext::for_tests(5);
    let problem = mechanism_problem(5, 40, mechanism_policy());
    let fit = fit_mechanisms(&problem, &ctx);
    let target = DynamicFitRef::OuMechanisms { problem: &problem, fit: &fit };
    let base = DynamicBootstrapRequest::whole_units(
        DynamicBootstrapMethod::Nonparametric,
        4,
        CyclicBootstrapDesign::FixedAssignments,
    );
    let design_refusal = |request| {
        matches!(
            bootstrap_dynamic_fit(target, request, &[], &ctx),
            Err(DynamicInferenceError::Design(SamplingDesignError::FailedAttempt { .. }))
        )
    };
    // Rows inside a trajectory are never independent draws.
    assert!(design_refusal(DynamicBootstrapRequest { resampling: ResamplingRule::RowIid, ..base }));
    let series =
        DynamicBootstrapRequest { design: DeclaredSamplingDesign::DependentTimeSeries, ..base };
    assert!(design_refusal(series));
    assert!(design_refusal(DynamicBootstrapRequest {
        method: DynamicBootstrapMethod::Parametric,
        ..series
    }));
    let clusters = DynamicBootstrapRequest {
        design: DeclaredSamplingDesign::ClusteredPanel,
        resampling: ResamplingRule::WholeCluster,
        ..base
    };
    assert!(matches!(
        bootstrap_dynamic_fit(target, clusters, &[], &ctx),
        Err(DynamicInferenceError::Invalid(_))
    ));
    assert!(matches!(
        dynamic_observed_information(
            target,
            DeclaredSamplingDesign::DependentTimeSeries,
            &[],
            &ctx
        ),
        Err(DynamicInferenceError::Design(_))
    ));
    let prior = DynamicPrior { mean: vec![0.0; 3], standard_deviation: vec![1.0; 3] };
    assert!(matches!(
        sample_dynamic_posterior(
            target,
            &prior,
            DynamicPosteriorOptions::default(),
            DeclaredSamplingDesign::ClusteredPanel,
            &[],
            &ctx,
        ),
        Err(DynamicInferenceError::Design(_))
    ));
    // A quantity of another family is refused before any work.
    let foreign = [DynamicQuantity::OuStationaryMean { clamps: vec![], variable: 0 }];
    assert!(matches!(
        bootstrap_dynamic_fit(target, base, &foreign, &ctx),
        Err(DynamicInferenceError::Invalid(_))
    ));
}

#[test]
fn unconverged_replicates_keep_their_slots_as_failed_mass() {
    let ctx = ExecutionContext::for_tests(9);
    let problem = mechanism_problem(9, 80, mechanism_policy());
    let fit = fit_mechanisms(&problem, &ctx);
    assert!(fit.converged);
    // The declared procedure is what replicates rerun; one step cannot converge.
    let truncated = OuMechanismInferenceProblem {
        policy: CyclicGradientPolicy { max_iterations: 1, ..mechanism_policy() },
        ..problem.clone()
    };
    let target = DynamicFitRef::OuMechanisms { problem: &truncated, fit: &fit };
    let request = DynamicBootstrapRequest::whole_units(
        DynamicBootstrapMethod::Nonparametric,
        12,
        CyclicBootstrapDesign::FixedAssignments,
    );
    let law = bootstrap_dynamic_fit(target, request, &[unseen()], &ctx).unwrap();
    assert_eq!(law.replicates.len(), 12);
    assert_eq!(law.weights.len(), 12);
    let failed = law.replicates.iter().filter(|r| r.failure.is_some()).count();
    assert!(failed > 0);
    assert!((law.failed_mass - failed as f64 / 12.0).abs() < 1e-12);
    assert!((law.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    for replicate in &law.replicates {
        assert_eq!(replicate.parameters.is_none(), replicate.failure.is_some());
        if replicate.failure.is_some() {
            assert_eq!(replicate.quantities, vec![None]);
        }
    }
    assert!(law.quantity_failed_mass[0] >= law.failed_mass);
    assert_eq!(law.standard_errors.is_some(), 12 - failed >= 2);
    let parametric = bootstrap_dynamic_fit(
        target,
        DynamicBootstrapRequest { method: DynamicBootstrapMethod::Parametric, ..request },
        &[],
        &ctx,
    )
    .unwrap();
    assert_eq!(parametric.replicates.len(), 12);
    assert!(
        parametric
            .replicates
            .iter()
            .all(|r| matches!(r.failure, Some(ReplicateFailure::NotConverged)))
    );
    assert!((parametric.failed_mass - 1.0).abs() < 1e-12);
    assert!(parametric.standard_errors.is_none());
}

#[test]
fn admission_and_cancellation_refuse_before_any_replicate() {
    let problem = mechanism_problem(11, 40, mechanism_policy());
    let fit = fit_mechanisms(&problem, &ExecutionContext::for_tests(11));
    let target = DynamicFitRef::OuMechanisms { problem: &problem, fit: &fit };
    let capacity = 64 * 1024;
    let pool = AdmissionPool::fixed(1, capacity);
    let small = ExecutionContext::for_tests(11).with_admission(pool.clone());
    let request = DynamicBootstrapRequest::whole_units(
        DynamicBootstrapMethod::Nonparametric,
        5_000,
        CyclicBootstrapDesign::FixedAssignments,
    );
    let refused = bootstrap_dynamic_fit(target, request, &[], &small).unwrap_err();
    assert!(matches!(refused, DynamicInferenceError::Resource(_)), "{refused:?}");
    assert_eq!(pool.available().bytes(), capacity);
    let prior = DynamicPrior { mean: vec![0.0; 3], standard_deviation: vec![1.0; 3] };
    let options = DynamicPosteriorOptions { chains: 16, draws: 100_000, ..Default::default() };
    let refused =
        sample_dynamic_posterior(target, &prior, options, INDEPENDENT, &[], &small).unwrap_err();
    assert!(matches!(refused, DynamicInferenceError::Resource(_)), "{refused:?}");
    assert_eq!(pool.available().bytes(), capacity);
    let stopped = ExecutionContext::for_tests(11);
    stopped.cancellation.cancel();
    let cancelled = DynamicInferenceError::Resource("cancelled".into());
    assert_eq!(bootstrap_dynamic_fit(target, request, &[], &stopped).unwrap_err(), cancelled);
    assert_eq!(
        dynamic_observed_information(target, INDEPENDENT, &[], &stopped).unwrap_err(),
        cancelled
    );
    assert_eq!(
        sample_dynamic_posterior(
            target,
            &prior,
            DynamicPosteriorOptions::default(),
            INDEPENDENT,
            &[],
            &stopped
        )
        .unwrap_err(),
        cancelled
    );
}

fn mismatched(target: DynamicFitRef<'_>, ctx: &ExecutionContext) -> bool {
    matches!(
        dynamic_observed_information(target, INDEPENDENT, &[], ctx),
        Err(DynamicInferenceError::FitMismatch(_))
    )
}

#[test]
fn a_fit_from_another_problem_is_refused() {
    let ctx = ExecutionContext::for_tests(13);
    let problem = mechanism_problem(13, 60, mechanism_policy());
    let fit = fit_mechanisms(&problem, &ctx);
    let other_data = mechanism_problem(14, 60, mechanism_policy());
    let ctx = &ctx;
    assert!(mismatched(DynamicFitRef::OuMechanisms { problem: &other_data, fit: &fit }, ctx));
    let mut moved = fit.clone();
    moved.model.diffusion_factor[0] += 0.1;
    assert!(mismatched(DynamicFitRef::OuMechanisms { problem: &problem, fit: &moved }, ctx));
    assert!(!mismatched(DynamicFitRef::OuMechanisms { problem: &problem, fit: &fit }, ctx));
}
