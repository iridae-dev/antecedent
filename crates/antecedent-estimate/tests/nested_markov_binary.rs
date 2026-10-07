//! 2.3A A3 (X4): binary nested-Markov likelihood pilot against an enumerated
//! latent-variable SCM. The oracle below sums the latent variable out by hand and
//! calls nothing in the code under test.
#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "binary levels and tiny fixed enumerations"
)]

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, BinaryCells, ConstraintStatus, FitMethod, FitOptions, IdentificationStanding,
    InferenceStanding, LikelihoodFitStanding, NestedMarkovInput, Regime, RegimeCounts,
    binary_cells, check_likelihood, constraint_residuals, evaluate_nested_markov_pilot,
    fit_nested_markov, refuse_public_interval_route,
};

/// `P(V = 1) = p` as a level probability.
fn b(p: f64, level: usize) -> f64 {
    if level == 1 { p } else { 1.0 - p }
}

// Structural equations, probabilities of level 1 (latent `u` confounds X2 and X4;
// `eps` is a direct X1 -> X4 effect, zero inside the selected ADMG).
fn p2(x1: usize, u: usize) -> f64 {
    0.2 + 0.3 * x1 as f64 + 0.4 * u as f64
}
fn p3(x2: usize) -> f64 {
    0.25 + 0.5 * x2 as f64
}
fn p4(x3: usize, u: usize, x1: usize, eps: f64) -> f64 {
    0.15 + 0.35 * x3 as f64 + 0.3 * u as f64 + eps * x1 as f64
}

const PU: f64 = 0.3;

/// Observed law by explicit summation over the latent variable.
fn scm_law(px1: f64, eps: f64) -> [f64; 16] {
    let mut law = [0.0; 16];
    for x1 in 0..2 {
        for x2 in 0..2 {
            for x3 in 0..2 {
                for x4 in 0..2 {
                    let mut mass = 0.0;
                    for u in 0..2 {
                        mass += b(px1, x1)
                            * b(PU, u)
                            * b(p2(x1, u), x2)
                            * b(p3(x2), x3)
                            * b(p4(x3, u, x1, eps), x4);
                    }
                    law[x1 * 8 + x2 * 4 + x3 * 2 + x4] = mass;
                }
            }
        }
    }
    law
}

/// `E[X4 | do(X2 = x2)]` in the SCM (X1 -> X4 effect zero).
fn truth_mean(x2: usize) -> f64 {
    let mut m = 0.0;
    for x3 in 0..2 {
        for u in 0..2 {
            m += b(p3(x2), x3) * b(PU, u) * p4(x3, u, 0, 0.0);
        }
    }
    m
}

fn scaled(law: &[f64; 16], n: f64) -> Vec<f64> {
    law.iter().map(|p| p * n).collect()
}

fn input_for(cells: Vec<f64>) -> NestedMarkovInput {
    NestedMarkovInput {
        graph: AdmgDeclaration::selected(),
        regimes: vec![RegimeCounts {
            regime: Regime::Observational,
            levels: vec![2, 2, 2, 2],
            cells,
        }],
    }
}

fn coded(error: &EstimationError) -> (&'static str, String) {
    match error {
        EstimationError::Refused { code, message }
        | EstimationError::RefusedWithFields { code, message, .. } => (*code, message.clone()),
        other => panic!("not a coded refusal: {other:?}"),
    }
}

fn assert_outside(error: &EstimationError) {
    let (code, message) = coded(error);
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("nested_markov.outside_binary_pilot"), "{message}");
    assert!(!message.contains("nonidentif") && !message.contains("not_identified"), "{message}");
}

#[test]
fn x4_nested_likelihood_truth_exact_counts_recover_scm() {
    let ctx = ExecutionContext::for_tests(1);
    let law = scm_law(0.4, 0.0);
    // Sixteen cells of an exact (infinite-sample) table scaled to counts.
    let input = input_for(scaled(&law, 1.0e6));
    let report =
        evaluate_nested_markov_pilot(&input, &FitOptions::default(), &ctx).expect("pilot fit");
    let fit = &report.fit;

    // The oracle law is itself normalized, and so is the model likelihood.
    assert!((law.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let check = check_likelihood(&fit.cells);
    assert!(check.normalized && check.positive, "{check:?}");

    // The fitted cells reproduce the SCM's observed law exactly.
    for (cell, truth) in fit.cells.iter().zip(law.iter()) {
        assert!((cell - truth).abs() < 1e-9, "{cell} vs {truth}");
    }

    // The fitted Mobius parameters equal the SCM's, summed over the latent by hand.
    let q2 = |x1: usize| -> f64 { (0..2).map(|u| b(PU, u) * (1.0 - p2(x1, u))).sum() };
    let q4 = |x3: usize| -> f64 { (0..2).map(|u| b(PU, u) * (1.0 - p4(x3, u, 0, 0.0))).sum() };
    let g = |x1: usize, x3: usize| -> f64 {
        (0..2).map(|u| b(PU, u) * (1.0 - p2(x1, u)) * (1.0 - p4(x3, u, 0, 0.0))).sum()
    };
    let params = &fit.parameters;
    assert!((params.a - 0.6).abs() < 1e-9);
    for k in 0..2 {
        assert!((params.c[k] - (1.0 - p3(k))).abs() < 1e-9);
        assert!((params.q2[k] - q2(k)).abs() < 1e-9);
        assert!((params.q4[k] - q4(k)).abs() < 1e-9);
        for j in 0..2 {
            assert!((params.g[k][j] - g(k, j)).abs() < 1e-9);
        }
    }

    // The equality constraints hold on the SCM's law.
    let residuals = constraint_residuals(&law).expect("residuals");
    assert!(residuals.max_abs < 1e-12, "{residuals:?}");
    assert_eq!(fit.diagnostics.constraint_status, ConstraintStatus::Satisfied);
    assert_eq!(fit.diagnostics.method, FitMethod::SaturatedFeasible);
    assert!(fit.diagnostics.converged);

    // Target contrast equals the SCM truth through the model and the plug-in.
    let truth = truth_mean(1) - truth_mean(0);
    let cmp = &report.comparison;
    assert!((cmp.model_contrast - truth).abs() < 1e-9, "{} vs {truth}", cmp.model_contrast);
    assert!((cmp.plugin_contrast - truth).abs() < 1e-9);
    assert!(cmp.difference.abs() < 1e-9);
    assert!((cmp.model_means[0] - truth_mean(0)).abs() < 1e-9);
    assert!((cmp.plugin_means[1] - truth_mean(1)).abs() < 1e-9);

    // The three standings are separate fields.
    assert_eq!(report.status.identification, IdentificationStanding::NonparametricallyIdentified);
    assert_eq!(report.status.likelihood_fit, LikelihoodFitStanding::ConvergedConstraintsSatisfied);
    assert_eq!(report.status.inference, InferenceStanding::IntervalWithheldCalibrationUnmeasured);

    // The interval route stays closed.
    let (code, message) = coded(&refuse_public_interval_route());
    assert_eq!(code, "cell_not_licensed");
    assert!(message.contains("nested_markov.route_frozen"));
}

#[test]
fn x4_nested_likelihood_truth_model_and_plugin_are_distinct_when_constraint_fails() {
    // A direct X1 -> X4 effect leaves the selected ADMG: the Verma constraint fails.
    let ctx = ExecutionContext::for_tests(2);
    let law = scm_law(0.4, 0.15);
    let residuals = constraint_residuals(&law).expect("residuals");
    assert!(residuals.max_abs > 1e-3, "{residuals:?}");
    let input = input_for(scaled(&law, 1.0e6));
    let report =
        evaluate_nested_markov_pilot(&input, &FitOptions::default(), &ctx).expect("projection");
    let d = &report.fit.diagnostics;
    // Reported, not silently fit.
    assert_eq!(d.constraint_status, ConstraintStatus::ViolatedByData);
    assert_eq!(d.method, FitMethod::CoordinateAscent);
    assert!(d.converged && d.iterations >= 1);
    assert!(d.empirical_residuals.max_abs > 1e-3);
    assert!(d.deviance.is_some_and(|dev| dev > 0.0));
    assert_eq!(
        report.status.likelihood_fit,
        LikelihoodFitStanding::ConvergedConstraintsViolatedByData
    );
    assert!(check_likelihood(&report.fit.cells).normalized);
    // The fitted model and the plug-in are different quantities here.
    assert!(report.comparison.difference.abs() > 1e-9);
    // A strict caller refuses instead of fitting.
    let strict =
        FitOptions { refuse_constraint_residual_above: Some(1e-6), ..FitOptions::default() };
    let counts = BinaryCells::new(&scaled(&law, 1.0e6)).expect("counts");
    let refusal = fit_nested_markov(&counts, &strict, &ctx).expect_err("strict refusal");
    let (code, message) = coded(&refusal.error);
    assert_eq!(code, "transport_numerical_failure");
    assert!(message.contains("nested_markov.constraint_violated"));
    assert!(refusal.diagnostics.is_some());
}

#[test]
fn x4_nested_outside_class_graph_regime_and_domain_refuse() {
    let cells = scaled(&scm_law(0.4, 0.0), 1.0e6);

    // Adjacent ADMGs: an extra directed edge, an extra bidirected edge, a missing edge.
    let mut extra_directed = AdmgDeclaration::selected();
    extra_directed.directed.push((0, 2));
    let mut extra_bidirected = AdmgDeclaration::selected();
    extra_bidirected.bidirected.push((1, 2));
    let mut missing = AdmgDeclaration::selected();
    missing.bidirected.clear();
    for graph in [extra_directed, extra_bidirected, missing] {
        let mut input = input_for(cells.clone());
        input.graph = graph;
        assert_outside(&binary_cells(&input).expect_err("adjacent graph"));
    }

    // Invalid regime and two regimes.
    let mut experimental = input_for(cells.clone());
    experimental.regimes[0].regime = Regime::Interventional(vec![1]);
    assert_outside(&binary_cells(&experimental).expect_err("regime"));
    let mut two = input_for(cells.clone());
    two.regimes.push(two.regimes[0].clone());
    assert_outside(&binary_cells(&two).expect_err("two regimes"));

    // Non-binary domain and non-binary table.
    let mut ternary = input_for(cells.clone());
    ternary.regimes[0].levels = vec![2, 3, 2, 2];
    assert_outside(&binary_cells(&ternary).expect_err("levels"));
    let wide = input_for(vec![1.0; 24]);
    assert_outside(&binary_cells(&wide).expect_err("table size"));

    // More than six observed variables.
    let mut large = input_for(cells);
    large.graph.variables = (0..7).map(|i| format!("V{i}")).collect();
    assert_outside(&binary_cells(&large).expect_err("seven variables"));
}

#[test]
fn x4_nested_outside_class_violating_law_is_reported_in_diagnostics() {
    let ctx = ExecutionContext::for_tests(3);
    let law = scm_law(0.4, 0.15);
    let counts = BinaryCells::new(&scaled(&law, 1.0e6)).expect("counts");
    let fit = fit_nested_markov(&counts, &FitOptions::default(), &ctx).expect("fit");
    let d = &fit.diagnostics;
    assert_eq!(d.constraint_status, ConstraintStatus::ViolatedByData);
    let r = &d.empirical_residuals;
    assert!(r.verma.iter().any(|v| v.abs() > 1e-3), "{r:?}");
    assert!(r.max_abs > 1e-3);
    // The fitted law is not the empirical law.
    let law_max_gap =
        fit.cells.iter().zip(law.iter()).map(|(c, p)| (c - p).abs()).fold(0.0_f64, f64::max);
    assert!(law_max_gap > 1e-6, "{law_max_gap}");
}

#[test]
fn x4_nested_fit_diagnostics_boundary_refuses_with_retained_diagnostics() {
    let ctx = ExecutionContext::for_tests(4);
    // P(X1 = 1) = 1e-11: every x1 = 1 cell sits at the boundary.
    let law = scm_law(1.0e-11, 0.0);
    let counts = BinaryCells::new(&scaled(&law, 1.0e6)).expect("counts");
    let refusal =
        fit_nested_markov(&counts, &FitOptions::default(), &ctx).expect_err("boundary fit");
    let (code, message) = coded(&refusal.error);
    assert_eq!(code, "transport_numerical_failure");
    assert!(message.contains("nested_markov.fitted_boundary"), "{message}");
    let d = refusal.diagnostics.expect("diagnostics retained");
    assert!(d.boundary_count.is_some_and(|c| c > 0));
    assert!(d.boundary_margin.is_some_and(|m| m <= 1e-9));
    assert!(d.model_log_likelihood.is_some());
    assert!(d.normalization.is_some_and(|n| n.normalized));
    let fields = refusal.error.refusal_fields().expect("structured fields");
    assert!(fields.boundary_count.is_some_and(|c| c > 0));
}

#[test]
fn x4_nested_fit_diagnostics_non_convergence_is_typed_and_keeps_state() {
    let ctx = ExecutionContext::for_tests(5);
    let counts = BinaryCells::new(&scaled(&scm_law(0.4, 0.15), 1.0e6)).expect("counts");
    let options = FitOptions { max_iterations: 1, ..FitOptions::default() };
    let refusal = fit_nested_markov(&counts, &options, &ctx).expect_err("not converged");
    let (code, message) = coded(&refusal.error);
    assert_eq!(code, "transport_numerical_failure");
    assert!(message.contains("nested_markov.fit_not_converged"), "{message}");
    let d = refusal.diagnostics.expect("diagnostics retained");
    assert_eq!(d.iterations, 1);
    assert!(!d.converged);
    assert_eq!(d.method, FitMethod::CoordinateAscent);
    assert!(d.final_change.is_some_and(|c| c > options.tolerance));
    assert_eq!(d.constraint_status, ConstraintStatus::ViolatedByData);
}

#[test]
fn x4_nested_fit_diagnostics_bounds_cancellation_and_positivity() {
    let ctx = ExecutionContext::for_tests(6);
    let counts = BinaryCells::new(&scaled(&scm_law(0.4, 0.0), 1.0e6)).expect("counts");

    let zero_bound = FitOptions { max_iterations: 0, ..FitOptions::default() };
    let refusal = fit_nested_markov(&counts, &zero_bound, &ctx).expect_err("bound");
    assert_eq!(coded(&refusal.error).0, "invalid_argument");
    let over = FitOptions { max_iterations: usize::MAX, ..FitOptions::default() };
    assert_eq!(
        coded(&fit_nested_markov(&counts, &over, &ctx).expect_err("cap").error).0,
        "invalid_argument"
    );

    let cancelled = ExecutionContext::for_tests(7);
    cancelled.cancellation.cancel();
    let refusal =
        fit_nested_markov(&counts, &FitOptions::default(), &cancelled).expect_err("cancelled");
    assert_eq!(coded(&refusal.error).0, "transport_budget_cancel");

    let mut cells = scaled(&scm_law(0.4, 0.0), 1.0e6);
    cells[3] = 0.0;
    let sparse = BinaryCells::new(&cells).expect("non-negative counts");
    let refusal = fit_nested_markov(&sparse, &FitOptions::default(), &ctx).expect_err("zero cell");
    let (code, message) = coded(&refusal.error);
    assert_eq!(code, "transport_support_failure");
    assert!(message.contains("nested_markov.positivity"));

    let negative = BinaryCells::new(&[-1.0; 16]).expect_err("negative counts");
    assert_eq!(coded(&negative).0, "invalid_argument");
}
