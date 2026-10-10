//! B4 latent-class (finite mixture) regime effects.
//!
//! Oracle: a deterministic, exactly balanced two-class design whose classes are so far apart that
//! every posterior responsibility is exactly 0 or 1 at the generating parameters. At such a
//! point the EM fixed point is the within-class maximum-likelihood regression, and because the
//! four noise points of every `(a, x)` cell are `{+0.5, -0.5, +0.25, -0.25}` (zero sum inside the
//! cell, so orthogonal to every function of `(a, x)`), that regression returns the generating
//! coefficients exactly, with residual variance `(0.25 + 0.25 + 0.0625 + 0.0625) / 4 = 0.15625`.
//!
//! * class `P` (24 rows): `a in {0, 1}`, `x in {-1, 0, 1}`, `y = 0 + 1 a + 0.5 x + e`;
//! * class `Q` (16 rows): `a in {0, 1}`, `x in {-1, 1}`, `y = 20 - 2 a - 0.5 x + e`.
//!
//! So `pi = (0.6, 0.4)`, `tau = (1, -2)`, and the canonical (ascending `tau`) order is `Q, P`.
//! The mixture-average effect is `0.6 * 1 + 0.4 * (-2) = -0.2`. Bootstrap calibration is
//! deliberately NOT measured here.

use antecedent_estimate::EstimationError;
use antecedent_estimate::effect_constancy::CalibrationStatus;
use antecedent_estimate::latent_class_effects::{
    LatentClassConfig, LatentClassData, LatentClassResult, PremiseStatus, fit_latent_class_effects,
};

const NOISE: [f64; 4] = [0.5, -0.5, 0.25, -0.25];
const NOISE_VARIANCE: f64 = 0.15625;

struct Data {
    y: Vec<f64>,
    a: Vec<f64>,
    x: Vec<Vec<f64>>,
    /// 0 for class `P`, 1 for class `Q`.
    class: Vec<usize>,
}

fn push(data: &mut Data, a: f64, x: f64, mean: f64, class: usize) {
    for e in NOISE {
        data.y.push(mean + e);
        data.a.push(a);
        data.x[0].push(x);
        data.class.push(class);
    }
}

fn build() -> Data {
    let mut data = Data { y: vec![], a: vec![], x: vec![vec![]], class: vec![] };
    for a in [0.0, 1.0] {
        for x in [-1.0, 0.0, 1.0] {
            push(&mut data, a, x, a + 0.5 * x, 0);
        }
    }
    for a in [0.0, 1.0] {
        for x in [-1.0, 1.0] {
            push(&mut data, a, x, 20.0 - 2.0 * a - 0.5 * x, 1);
        }
    }
    data
}

fn reversed(data: &Data) -> Data {
    Data {
        y: data.y.iter().rev().copied().collect(),
        a: data.a.iter().rev().copied().collect(),
        x: data.x.iter().map(|c| c.iter().rev().copied().collect()).collect(),
        class: data.class.iter().rev().copied().collect(),
    }
}

fn config(seed: u64) -> LatentClassConfig {
    let mut config = LatentClassConfig::new(2, seed).declare_conditional_randomization();
    config.bootstrap_replicates = 0;
    config
}

fn fit(data: &Data, config: &LatentClassConfig) -> Result<LatentClassResult, EstimationError> {
    fit_latent_class_effects(
        &LatentClassData { outcome: &data.y, treatment: &data.a, covariates: &data.x },
        config,
    )
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance, "{actual} vs {expected} (tol {tolerance})");
}

fn refusal(result: Result<LatentClassResult, EstimationError>) -> (&'static str, String) {
    match result {
        Err(EstimationError::Refused { code, message }) => (code, message),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

fn assert_canonical_equal(left: &LatentClassResult, right: &LatentClassResult, tolerance: f64) {
    assert_eq!(left.classes.len(), right.classes.len());
    for (l, r) in left.classes.iter().zip(&right.classes) {
        close(l.effect, r.effect, tolerance);
        close(l.weight, r.weight, tolerance);
        close(l.intercept, r.intercept, tolerance);
        close(l.residual_variance, r.residual_variance, tolerance);
        for (lc, rc) in l.covariate_coefficients.iter().zip(&r.covariate_coefficients) {
            close(*lc, *rc, tolerance);
        }
    }
    close(left.mixture_average_effect, right.mixture_average_effect, tolerance);
}

#[test]
fn b4_latent_recovers_hand_derived_two_class_fixed_point() {
    let data = build();
    let result = fit(&data, &config(11)).expect("well-separated fit");
    // Canonical order is ascending tau: Q then P.
    let q = &result.classes[0];
    let p = &result.classes[1];
    close(q.effect, -2.0, 1e-8);
    close(q.weight, 0.4, 1e-12);
    close(q.intercept, 20.0, 1e-8);
    close(q.covariate_coefficients[0], -0.5, 1e-8);
    close(q.residual_variance, NOISE_VARIANCE, 1e-8);
    close(p.effect, 1.0, 1e-8);
    close(p.weight, 0.6, 1e-12);
    close(p.intercept, 0.0, 1e-8);
    close(p.covariate_coefficients[0], 0.5, 1e-8);
    close(p.residual_variance, NOISE_VARIANCE, 1e-8);
    close(result.mixture_average_effect, -0.2, 1e-8);
    close(result.classes[0].effective_n, 16.0, 1e-8);
    close(result.classes[1].effective_n, 24.0, 1e-8);
    close(result.min_effect_gap, 3.0, 1e-8);
    close(result.separation, 1.0, 1e-12);
    assert!(result.classes[0].effect_se.is_none() && result.mixture_average_se.is_none());
}

#[test]
fn b4_latent_responsibilities_are_the_exact_class_indicators() {
    let data = build();
    let result = fit(&data, &config(3)).expect("fit");
    assert_eq!(result.responsibilities.len(), data.y.len());
    for (i, row) in result.responsibilities.iter().enumerate() {
        close(row.iter().sum::<f64>(), 1.0, 1e-12);
        // true class P (0) is canonical 1; true class Q (1) is canonical 0.
        let canonical = 1 - data.class[i];
        close(row[canonical], 1.0, 1e-12);
        assert_eq!(result.hard_assignment[i], canonical);
    }
}

#[test]
fn b4_latent_log_likelihood_is_non_decreasing_across_iterations() {
    let data = build();
    // Start from a contaminated labelling (every fourth unit mislabelled) so EM has to walk.
    let labels: Vec<usize> =
        data.class.iter().enumerate().map(|(i, &c)| if i % 4 == 3 { 1 - c } else { c }).collect();
    let mut cfg = config(5);
    cfg.initial_labels = Some(labels);
    let result = fit(&data, &cfg).expect("contaminated start still converges");
    assert!(result.log_likelihood_trace.len() >= 3, "{:?}", result.log_likelihood_trace);
    for pair in result.log_likelihood_trace.windows(2) {
        assert!(pair[1] >= pair[0] - 1e-9 * (1.0 + pair[0].abs()), "{pair:?}");
    }
    close(*result.log_likelihood_trace.last().expect("trace"), result.log_likelihood, 0.0);
    close(result.classes[0].effect, -2.0, 1e-8);
    close(result.classes[1].effect, 1.0, 1e-8);
    close(result.classes[1].weight, 0.6, 1e-12);
}

#[test]
fn b4_latent_label_permutation_gives_identical_canonical_output() {
    let data = build();
    let truth = data.class.clone();
    let swapped: Vec<usize> = truth.iter().map(|&c| 1 - c).collect();
    let mut cfg_a = config(1);
    cfg_a.initial_labels = Some(truth);
    let mut cfg_b = config(1);
    cfg_b.initial_labels = Some(swapped);
    let a = fit(&data, &cfg_a).expect("fit a");
    let b = fit(&data, &cfg_b).expect("fit b");
    assert_canonical_equal(&a, &b, 1e-12);
    // The mapping is retained and does record that the raw labels were swapped.
    assert_eq!(a.class_order, vec![1, 0]);
    assert_eq!(b.class_order, vec![0, 1]);
    for (x, y) in a.responsibilities.iter().zip(&b.responsibilities) {
        close(x[0], y[0], 1e-12);
        close(x[1], y[1], 1e-12);
    }
    assert_eq!(a.hard_assignment, b.hard_assignment);
}

#[test]
fn b4_latent_seeds_and_row_order_do_not_change_the_canonical_result() {
    let data = build();
    let baseline = fit(&data, &config(1)).expect("seed 1");
    for seed in [2_u64, 3, 7, 12345] {
        let other = fit(&data, &config(seed)).expect("other seed");
        assert_canonical_equal(&baseline, &other, 1e-9);
    }
    let flipped = reversed(&data);
    let rev = fit(&flipped, &config(1)).expect("reversed rows");
    assert_canonical_equal(&baseline, &rev, 1e-9);
    let n = data.y.len();
    for i in 0..n {
        for c in 0..2 {
            close(baseline.responsibilities[i][c], rev.responsibilities[n - 1 - i][c], 1e-9);
        }
    }
}

#[test]
fn b4_latent_mixture_average_is_the_weighted_sum_of_class_effects() {
    let data = build();
    let result = fit(&data, &config(2)).expect("fit");
    let weighted: f64 = result.classes.iter().map(|c| c.weight * c.effect).sum();
    close(result.mixture_average_effect, weighted, 1e-12);
    close(result.classes.iter().map(|c| c.weight).sum::<f64>(), 1.0, 1e-12);
    close(weighted, 0.6 - 0.4 * 2.0, 1e-8);
}

#[test]
fn b4_latent_deterministic_replay_given_seed_and_bootstrap_uncertainty() {
    let data = build();
    let mut cfg = config(7);
    cfg.bootstrap_replicates = 60;
    let first = fit(&data, &cfg).expect("fit");
    let second = fit(&data, &cfg).expect("replay");
    assert_eq!(first, second, "same seed must replay bit-identically");

    let boot = first.bootstrap.expect("bootstrap summary");
    assert_eq!(boot.requested, 60);
    assert_eq!(boot.succeeded + boot.failed, 60);
    assert!(boot.succeeded >= 40, "{boot:?}");
    // Closed-form OLS standard error of tau within an exactly balanced class:
    // sigma^2 / (n_k * Var(a)) with Var(a) = 0.25 (a is orthogonal to x).
    let closed_form_q = (NOISE_VARIANCE / (16.0 * 0.25)).sqrt();
    let closed_form_p = (NOISE_VARIANCE / (24.0 * 0.25)).sqrt();
    let se_q = first.classes[0].effect_se.expect("se q");
    let se_p = first.classes[1].effect_se.expect("se p");
    assert!(se_q > 0.5 * closed_form_q && se_q < 2.0 * closed_form_q, "{se_q} vs {closed_form_q}");
    assert!(se_p > 0.5 * closed_form_p && se_p < 2.0 * closed_form_p, "{se_p} vs {closed_form_p}");
    let avg_se = first.mixture_average_se.expect("average se");
    assert!(avg_se.is_finite() && avg_se > 0.0);
    assert!(first.classes[0].weight_se.expect("weight se") > 0.0);

    // A different seed resamples differently (different SE) around the same point estimate.
    let mut other_cfg = cfg.clone();
    other_cfg.seed = 8;
    let other = fit(&data, &other_cfg).expect("other seed");
    assert!((other.classes[0].effect_se.expect("se") - se_q).abs() > 0.0);
    close(other.classes[0].effect, first.classes[0].effect, 1e-9);

    // Honest scope: calibration unmeasured, premises declared vs checked are distinguished.
    assert_eq!(first.calibration, CalibrationStatus::Unmeasured);
    let randomization = first
        .premises
        .iter()
        .find(|p| p.name == "conditional_randomization_within_class")
        .expect("premise");
    assert_eq!(randomization.status, PremiseStatus::Declared);
}

#[test]
fn b4_latent_weak_class_weight_refuses() {
    let data = build();
    let mut cfg = config(1);
    cfg.min_class_weight = 0.5; // the smaller class has weight 0.4
    let (code, message) = refusal(fit(&data, &cfg));
    assert_eq!(code, "population_not_estimable");
    assert!(message.contains("latent_class.weak_class"), "{message}");
}

#[test]
fn b4_latent_overlapping_classes_are_never_reported_as_separated() {
    // Two classes whose regression surfaces differ by 0.2 on noise of sd 0.4: not separable.
    let mut data = Data { y: vec![], a: vec![], x: vec![vec![]], class: vec![] };
    for (shift, class) in [(0.0, 0_usize), (0.2, 1)] {
        for a in [0.0, 1.0] {
            for x in [-1.0, 0.0, 1.0] {
                push(&mut data, a, x, shift + a + 0.5 * x, class);
            }
        }
    }
    let cfg = config(4);
    match fit(&data, &cfg) {
        Err(EstimationError::Refused { message, .. }) => {
            assert!(message.contains("latent_class."), "{message}");
        }
        Ok(result) => {
            // If a split is found at all it must satisfy the declared guards.
            assert!(result.separation >= cfg.min_separation);
            assert!(result.classes.iter().all(|c| c.weight >= cfg.min_class_weight));
        }
        Err(other) => panic!("unexpected error {other:?}"),
    }
}

#[test]
fn b4_latent_single_class_is_degenerate() {
    let data = build();
    let mut one = config(1);
    one.classes = 1;
    let (code, message) = refusal(fit(&data, &one));
    assert_eq!(code, "population_not_estimable");
    assert!(message.contains("latent_class.degenerate_class"), "{message}");

    // A single exact regime (no noise): every start collapses a class onto a perfect fit.
    let mut single = Data { y: vec![], a: vec![], x: vec![vec![]], class: vec![] };
    for a in [0.0, 1.0] {
        for x in [-1.0, 0.0, 1.0] {
            for _ in 0..4 {
                single.y.push(a + 0.5 * x);
                single.a.push(a);
                single.x[0].push(x);
                single.class.push(0);
            }
        }
    }
    let (code, message) = refusal(fit(&single, &config(1)));
    assert_eq!(code, "population_not_estimable");
    assert!(message.contains("latent_class.degenerate_class"), "{message}");
}

#[test]
fn b4_latent_premise_and_configuration_refusals() {
    let data = build();
    let undeclared = LatentClassConfig::new(2, 1);
    let (code, message) = refusal(fit(&data, &undeclared));
    assert_eq!(code, "required_option_missing");
    assert!(message.contains("latent_class.randomization_not_declared"), "{message}");

    let mut many = config(1);
    many.classes = 5;
    let (code, message) = refusal(fit(&data, &many));
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("latent_class.too_many_classes"), "{message}");

    let mut constant = build();
    constant.a = vec![1.0; constant.a.len()];
    let (code, message) = refusal(fit(&constant, &config(1)));
    assert_eq!(code, "effect_not_identified");
    assert!(message.contains("latent_class.treatment_constant"), "{message}");

    let mut not_converged = config(1);
    not_converged.initial_labels = Some(
        build()
            .class
            .iter()
            .enumerate()
            .map(|(i, &c)| if i % 4 == 3 { 1 - c } else { c })
            .collect(),
    );
    not_converged.max_iterations = 1;
    let (code, message) = refusal(fit(&data, &not_converged));
    assert_eq!(code, "mechanism_fit_not_converged");
    assert!(message.contains("latent_class.not_converged"), "{message}");
}
