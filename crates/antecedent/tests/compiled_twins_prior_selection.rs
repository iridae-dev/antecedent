//! The compiled power-prior discount selections and the direct `antecedent_prob` selections
//! return the same selected discount, estimate, intervals, sensitivity table and composed prior,
//! bit for bit.
//!
//! The fixtures are those of the kernel's own tests: the Beta source 30/100 against 9/30 target
//! successes, a discrepant source 90/100 against 12/60, a 300/1000 source with a declared
//! discrepancy bound, the selected Beta prior on the grid (0, 0.5, 1), a cross-validated
//! Gamma-Poisson rate, and the two external Gaussian sources of the named-composed-prior test
//! over a 200-row linear target with a known residual variance. Neither path reads the context's
//! seed: the selection's fold assignment and critical-value draws come from the options' seed,
//! which both paths receive unchanged. Each compiled program names its route, and the route's
//! unit is the supported capability record that owns the selection.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use std::sync::Arc;

use antecedent::Study;
use antecedent::planner::PlanRoute;
use antecedent_core::ExecutionContext;
use antecedent_prob::{
    AlphaCriterion, AlphaSelectionOptions, BetaHyperparameters, BetaSource, ExternalPriorSource,
    ExternalPriorWeight, GammaHyperparameters, GammaSource, GaussianCoefficientPrior, LinearTarget,
    PriorSet, PriorSpec, SelectedAlphaReport, select_beta_power_prior, select_beta_prior,
    select_gamma_power_prior, select_gamma_prior, select_linear_power_prior, select_linear_prior,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(17)
}

fn same_pair(left: (f64, f64), right: (f64, f64), what: &str) {
    assert_eq!(left.0.to_bits(), right.0.to_bits(), "{what} lower");
    assert_eq!(left.1.to_bits(), right.1.to_bits(), "{what} upper");
}

fn same_report(direct: &SelectedAlphaReport, compiled: &SelectedAlphaReport) {
    assert_eq!(direct.criterion, compiled.criterion);
    assert_eq!(direct.selected_index, compiled.selected_index);
    assert_eq!(direct.selected_alpha.to_bits(), compiled.selected_alpha.to_bits());
    assert_eq!(direct.estimate.to_bits(), compiled.estimate.to_bits());
    assert_eq!(direct.level.to_bits(), compiled.level.to_bits());
    same_pair(direct.conditional_interval, compiled.conditional_interval, "conditional");
    same_pair(direct.interval, compiled.interval, "interval");
    same_pair(direct.uniform_interval, compiled.uniform_interval, "uniform");
    assert_eq!(
        direct.discrepancy_bound.map(f64::to_bits),
        compiled.discrepancy_bound.map(f64::to_bits)
    );
    assert_eq!(direct.sensitivity.len(), compiled.sensitivity.len());
    for (left, right) in direct.sensitivity.iter().zip(&compiled.sensitivity) {
        assert_eq!(left.alpha.to_bits(), right.alpha.to_bits());
        assert_eq!(left.score.to_bits(), right.score.to_bits());
        assert_eq!(left.estimate.to_bits(), right.estimate.to_bits());
        assert_eq!(left.lower.to_bits(), right.lower.to_bits());
        assert_eq!(left.upper.to_bits(), right.upper.to_bits());
    }
    assert_eq!(direct.interval_semantics(), compiled.interval_semantics());
    assert_eq!(direct.provenance_id(), compiled.provenance_id());
}

fn beta_cases() -> Vec<(BetaSource, Vec<bool>, AlphaSelectionOptions)> {
    let base = BetaHyperparameters { alpha: 1.0, beta: 1.0 };
    let empirical = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
    let mut bounded = empirical.clone();
    bounded.discrepancy_bound = Some(0.02);
    let mut coarse = empirical.clone();
    coarse.grid = vec![0.0, 0.5, 1.0];
    coarse.discrepancy_bound = Some(0.2);
    let mut folded = AlphaSelectionOptions::new(AlphaCriterion::CrossValidated { folds: 5 });
    folded.seed = 11;
    folded.discrepancy_bound = Some(0.05);
    vec![
        (
            BetaSource { base, successes: 30, trials: 100 },
            (0..30).map(|i| i < 9).collect(),
            empirical.clone(),
        ),
        (
            BetaSource { base, successes: 90, trials: 100 },
            (0..60).map(|i| i < 12).collect(),
            empirical,
        ),
        (
            BetaSource { base, successes: 300, trials: 1_000 },
            (0..100).map(|i| i < 30).collect(),
            bounded,
        ),
        (
            BetaSource { base, successes: 30, trials: 50 },
            (0..40).map(|i| i % 5 < 3).collect(),
            coarse,
        ),
        (
            BetaSource { base, successes: 30, trials: 100 },
            (0..40).map(|i| i % 4 == 0).collect(),
            folded,
        ),
    ]
}

#[test]
fn compiled_beta_selection_equals_the_direct_selection_bit_for_bit() {
    let ctx = ctx();
    for (source, outcomes, options) in beta_cases() {
        let compiled = Study::compile_beta_power_prior_selection(
            source,
            outcomes.clone(),
            options.clone(),
            &ctx,
        )
        .unwrap();
        let direct = select_beta_power_prior(source, &outcomes, &options).unwrap();
        same_report(&direct, &compiled.select_power_prior(&ctx).unwrap());
        let (direct_prior, direct_report) = select_beta_prior(source, &outcomes, &options).unwrap();
        let (prior, report) = compiled.select_prior(&ctx).unwrap();
        assert_eq!(direct_prior.alpha.to_bits(), prior.alpha.to_bits());
        assert_eq!(direct_prior.beta.to_bits(), prior.beta.to_bits());
        same_report(&direct_report, &report);
    }
}

#[test]
fn compiled_gamma_selection_equals_the_direct_selection_bit_for_bit() {
    let ctx = ctx();
    let source = GammaSource {
        base: GammaHyperparameters { shape: 1.0, rate: 1.0 },
        total_count: 50,
        exposure: 25.0,
    };
    let counts = vec![2_u64, 1, 3, 2, 2, 1];
    let mut bounded = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
    bounded.discrepancy_bound = Some(0.5);
    for options in [
        AlphaSelectionOptions::new(AlphaCriterion::CrossValidated { folds: 3 }),
        AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes),
        bounded,
    ] {
        let compiled = Study::compile_gamma_power_prior_selection(
            source,
            counts.clone(),
            options.clone(),
            &ctx,
        )
        .unwrap();
        let direct = select_gamma_power_prior(source, &counts, &options).unwrap();
        same_report(&direct, &compiled.select_power_prior(&ctx).unwrap());
        let (direct_prior, direct_report) = select_gamma_prior(source, &counts, &options).unwrap();
        let (prior, report) = compiled.select_prior(&ctx).unwrap();
        assert_eq!(direct_prior.shape.to_bits(), prior.shape.to_bits());
        assert_eq!(direct_prior.rate.to_bits(), prior.rate.to_bits());
        same_report(&direct_report, &report);
    }
}

const ROWS: usize = 200;

fn gaussian(mean: Vec<f64>, variance: Vec<f64>) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean),
        variance: Arc::from(variance),
    }));
    prior
}

fn source(id: &str, mean: Vec<f64>, variance: Vec<f64>) -> ExternalPriorSource {
    ExternalPriorSource {
        id: Arc::from(id),
        prior: gaussian(mean, variance),
        weight: ExternalPriorWeight::power(1.0).unwrap(),
        ess: None,
    }
}

/// The rows of the named-composed-prior test: `y = 1 + 2 t + 0.8 z` plus a deterministic wiggle.
fn linear_rows() -> (Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..ROWS).map(|i| f64::from(u8::from((i * 7) % 3 == 0))).collect();
    let z: Vec<f64> = (0..ROWS).map(|i| (i as f64 * 0.37).sin() + 0.01 * i as f64).collect();
    let y: Vec<f64> = (0..ROWS)
        .map(|i| 1.0 + 2.0 * t[i] + 0.8 * z[i] + 0.2 * ((i as f64 * 12.9898).sin()))
        .collect();
    let design: Vec<f64> = (0..ROWS).flat_map(|i| [1.0, t[i], z[i]]).collect();
    (design, y)
}

#[test]
fn compiled_linear_selection_equals_the_direct_selection_bit_for_bit() {
    let ctx = ctx();
    let (design, y) = linear_rows();
    let sources = vec![
        source("a", vec![0.8, 1.7, 0.9], vec![0.5, 0.4, 0.3]),
        source("b", vec![1.2, 2.3, 1.1], vec![0.6, 0.5, 0.4]),
    ];
    let mut known = gaussian(vec![0.0; 3], vec![100.0; 3]);
    known.push(PriorSpec::KnownResidualVariance(0.05));
    let mut empirical = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
    empirical.grid = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    empirical.discrepancy_bound = Some(0.5);
    let mut folded = AlphaSelectionOptions::new(AlphaCriterion::CrossValidated { folds: 4 });
    folded.seed = 5;
    let functional = [0.0, 1.0, 0.0];
    let target =
        LinearTarget { x: &design, y: &y, n_rows: ROWS, n_cols: 3, functional: &functional };
    for options in [empirical, folded] {
        let compiled = Study::compile_linear_power_prior_selection(
            sources.clone(),
            known.clone(),
            target,
            options.clone(),
            &ctx,
        )
        .unwrap();
        let direct = select_linear_power_prior(&sources, &known, target, &options).unwrap();
        same_report(&direct, &compiled.select_power_prior(&ctx).unwrap());
        let (direct_prior, direct_report) =
            select_linear_prior(&sources, &known, target, &options).unwrap();
        let (prior, report) = compiled.select_prior(&ctx).unwrap();
        same_report(&direct_report, &report);
        let (left, right) = (
            direct_prior.prior.gaussian_coefficients().unwrap(),
            prior.prior.gaussian_coefficients().unwrap(),
        );
        for (a, b) in left.mean.iter().zip(right.mean.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in left.variance.iter().zip(right.variance.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(format!("{direct_prior:?}"), format!("{prior:?}"));
    }
}

#[test]
fn each_selection_names_its_route_and_its_supported_record() {
    let ctx = ctx();
    let (beta_source, outcomes, options) = beta_cases().remove(0);
    let beta =
        Study::compile_beta_power_prior_selection(beta_source, outcomes, options, &ctx).unwrap();
    let gamma = Study::compile_gamma_power_prior_selection(
        GammaSource {
            base: GammaHyperparameters { shape: 1.0, rate: 1.0 },
            total_count: 5,
            exposure: 4.0,
        },
        vec![1, 2, 0, 1],
        AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes),
        &ctx,
    )
    .unwrap();
    let (design, y) = linear_rows();
    let mut known = gaussian(vec![0.0; 3], vec![100.0; 3]);
    known.push(PriorSpec::KnownResidualVariance(0.05));
    let linear = Study::compile_linear_power_prior_selection(
        vec![source("a", vec![0.8, 1.7, 0.9], vec![0.5, 0.4, 0.3])],
        known,
        LinearTarget { x: &design, y: &y, n_rows: ROWS, n_cols: 3, functional: &[0.0, 1.0, 0.0] },
        AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes),
        &ctx,
    )
    .unwrap();
    for (view, route, unit) in [
        (
            beta.plan_view(),
            PlanRoute::PowerPriorBetaSelection,
            "association.conjugate_count_posterior",
        ),
        (
            gamma.plan_view(),
            PlanRoute::PowerPriorGammaSelection,
            "association.conjugate_count_posterior",
        ),
        (
            linear.plan_view(),
            PlanRoute::PowerPriorLinearSelection,
            "prior.composed_named_coordinates",
        ),
    ] {
        assert_eq!(view.route(), route);
        let decision = view.unit_support().expect("the route is bound to a unit");
        assert_eq!(decision.unit(), unit);
        assert!(decision.supported(), "{unit} is a supported record");
    }
    // A run consults the unit: on a supported unit it passes and returns the selection.
    assert!(beta.select_power_prior(&ctx).is_ok());
    assert!(gamma.select_prior(&ctx).is_ok());
    assert!(linear.select_power_prior(&ctx).is_ok());
}
