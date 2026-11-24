//! Known-truth tests of external prior composition and of the transport of external priors.
//!
//! Every expected value is derived in the test from the closed form the composition states, by
//! hand arithmetic that does not call the code under test:
//!
//! * the Gaussian power prior of Ibrahim and Chen (2000) on Gaussian sources is the tempered
//!   product `N(m0, V0) * prod_k N(m_k, V_k)^alpha_k`, whose precision is
//!   `1/V0 + sum_k alpha_k / V_k` and whose mean is the precision-weighted mean of the means;
//!   the mixture path moment-matches `sum_k w_k N(m_k, V_k / alpha_k) + (1 - sum w) N(m0, V0)`;
//! * the Beta-Binomial count power prior is `Beta(a + sum alpha k, b + sum alpha (n - k))` and
//!   the Gamma-Poisson one is `Gamma(shape + sum alpha S, rate + sum alpha E)` in the rate
//!   parameterization (mean `shape / rate`), as `antecedent_prob::power_prior_alpha` documents;
//! * the transport reweighting (a declared reweighting of the repository, not a paper's) moves
//!   the chosen coefficient's mean to the weighted mean `sum w e / sum w` of the unit effects
//!   and widens its conjugate variance by the sampling variance of that mean,
//!   `sum w^2 (e - mean)^2 / (sum w)^2`, divided by the source's residual variance; a weighted
//!   target is checked by its support, its Kish effective sample size `(sum w)^2 / sum w^2`
//!   (floor 20 and a fraction of the rows) and its largest single-row share;
//! * the Gaussian transport kernel is `exp(-u^2 / 2)` and the Epanechnikov kernel `1 - u^2` for
//!   `u = d / h < 1`, as `antecedent_prob::KernelFamily` documents.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "exact values are compared exactly")]

use std::sync::Arc;

use antecedent_prob::{
    BetaHyperparameters, ComposedPrior, ExternalPriorSource, ExternalPriorWeight,
    GammaHyperparameters, GaussianCoefficientPrior, KernelFamily, PriorSet, PriorSpec, ProbError,
    TransportAdjustment, TransportContext, TransportKernel, TransportPolicy, WeightedTarget,
    apply_transport, apply_weighted_target_transport, compose_beta_power_priors,
    compose_external_priors, compose_gamma_power_priors, compose_with_transport,
    compose_with_weighted_target_transport, kernel_bandwidth_sensitivity,
};

fn close(got: f64, want: f64, what: &str) {
    let scale = want.abs().max(1.0);
    assert!((got - want).abs() <= 1e-12 * scale, "{what}: got {got}, want {want}");
}

fn gaussian(mean: &[f64], variance: &[f64]) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean.to_vec()),
        variance: Arc::from(variance.to_vec()),
    }));
    prior
}

fn source(
    id: &str,
    mean: &[f64],
    variance: &[f64],
    weight: ExternalPriorWeight,
    ess: Option<f64>,
) -> ExternalPriorSource {
    ExternalPriorSource { id: Arc::from(id), prior: gaussian(mean, variance), weight, ess }
}

fn power(alpha: f64) -> ExternalPriorWeight {
    ExternalPriorWeight::power(alpha).unwrap()
}

fn moments(prior: &PriorSet) -> (Vec<f64>, Vec<f64>) {
    let coef = prior.gaussian_coefficients().unwrap();
    (coef.mean.to_vec(), coef.variance.to_vec())
}

/// The precision-weighted closed form of one coefficient: `(mean, variance)` of
/// `N(m0, v0) * prod N(m_k, v_k)^alpha_k`.
fn tempered_product(m0: f64, v0: f64, parts: &[(f64, f64, f64)]) -> (f64, f64) {
    let mut precision = 1.0 / v0;
    let mut weighted = m0 / v0;
    for &(mean, variance, alpha) in parts {
        precision += alpha / variance;
        weighted += alpha * mean / variance;
    }
    (weighted / precision, 1.0 / precision)
}

fn refusal_code(error: &ProbError) -> &'static str {
    match error {
        ProbError::Refused { code, .. } => code,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn gaussian_power_composition_of_one_and_two_sources_equals_the_precision_weighted_closed_form() {
    // One source: baseline N(0, 4), source N(2, 1) at alpha 1/2. Precision 1/4 + 1/2 = 3/4,
    // mean (1/2 * 2) / (3/4) = 4/3, variance 4/3. Prior-strength ESS 1/2 * 40 = 20.
    let one = compose_external_priors(
        &[source("a", &[2.0], &[1.0], power(0.5), Some(40.0))],
        &gaussian(&[0.0], &[4.0]),
    )
    .unwrap();
    let (mean, variance) = moments(&one.prior);
    close(mean[0], 4.0 / 3.0, "one-source mean");
    close(variance[0], 4.0 / 3.0, "one-source variance");
    close(one.effective_ess[0].unwrap(), 20.0, "one-source effective ess");
    close(one.composed_ess.unwrap(), 20.0, "one-source composed ess");
    close(one.kish_ess.unwrap(), 1.0, "one-source kish ess");

    // Two sources over two coefficients: baseline N((0, 1), diag(4, 2)); source a
    // N((2, -1), diag(1, 1/2)) at alpha 1/2 with ess 40; source b N((3, 0), diag(2, 1)) at
    // alpha 1/4 with ess 80.
    // Coefficient 0: precision 1/4 + 1/2 + 1/8 = 7/8; mean (1 + 3/8) / (7/8) = 11/7.
    // Coefficient 1: precision 1/2 + 1 + 1/4 = 7/4; mean (1/2 - 1 + 0) / (7/4) = -2/7.
    let sources = [
        source("a", &[2.0, -1.0], &[1.0, 0.5], power(0.5), Some(40.0)),
        source("b", &[3.0, 0.0], &[2.0, 1.0], power(0.25), Some(80.0)),
    ];
    let two = compose_external_priors(&sources, &gaussian(&[0.0, 1.0], &[4.0, 2.0])).unwrap();
    let (mean, variance) = moments(&two.prior);
    close(mean[0], 11.0 / 7.0, "two-source mean 0");
    close(variance[0], 8.0 / 7.0, "two-source variance 0");
    close(mean[1], -2.0 / 7.0, "two-source mean 1");
    close(variance[1], 4.0 / 7.0, "two-source variance 1");
    // The same numbers from the tempered-product formula, coefficient by coefficient.
    for (i, (m0, v0)) in [(0.0, 4.0), (1.0, 2.0)].into_iter().enumerate() {
        let parts = [([2.0, -1.0][i], [1.0, 0.5][i], 0.5), ([3.0, 0.0][i], [2.0, 1.0][i], 0.25)];
        let (m, v) = tempered_product(m0, v0, &parts);
        close(mean[i], m, "tempered-product mean");
        close(variance[i], v, "tempered-product variance");
    }
    // Per-source ESS is alpha * ess; the composed ESS sums them; Kish over the applied alphas
    // (1/2, 1/4) is (3/4)^2 / (1/4 + 1/16) = 9/5.
    close(two.effective_ess[0].unwrap(), 20.0, "effective ess a");
    close(two.effective_ess[1].unwrap(), 20.0, "effective ess b");
    close(two.composed_ess.unwrap(), 40.0, "composed ess");
    close(two.kish_ess.unwrap(), 9.0 / 5.0, "kish ess");
    assert_eq!(&*two.alphas_requested, &[0.5, 0.25]);
    assert_eq!(&*two.alphas_applied, &[0.5, 0.25]);
    assert_eq!(two.source_ids.iter().map(|id| &**id).collect::<Vec<&str>>(), ["a", "b"]);

    // A discount of zero drops the source: the composition is the baseline itself.
    let dropped = compose_external_priors(
        &[source("a", &[2.0], &[1.0], power(0.0), Some(40.0))],
        &gaussian(&[0.5], &[4.0]),
    )
    .unwrap();
    let (mean, variance) = moments(&dropped.prior);
    close(mean[0], 0.5, "dropped mean");
    close(variance[0], 4.0, "dropped variance");
    close(dropped.effective_ess[0].unwrap(), 0.0, "dropped effective ess");
}

#[test]
fn beta_and_gamma_count_power_priors_add_the_discounted_counts() {
    // Beta(1, 1) with sources (30 of 100, alpha 1/2) and (12 of 40, alpha 1/4):
    // a = 1 + 15 + 3 = 19 and b = 1 + 35 + 7 = 43.
    let beta = compose_beta_power_priors(
        BetaHyperparameters { alpha: 1.0, beta: 1.0 },
        &[(30, 100, 0.5), (12, 40, 0.25)],
    )
    .unwrap();
    close(beta.alpha, 19.0, "beta a");
    close(beta.beta, 43.0, "beta b");
    close(beta.alpha / (beta.alpha + beta.beta), 19.0 / 62.0, "beta prior mean");
    // Gamma(2, 1) (shape, rate) with sources (50 events over exposure 25, alpha 0.4) and
    // (10 events over exposure 8, alpha 1): shape = 2 + 20 + 10 = 32, rate = 1 + 10 + 8 = 19.
    let gamma = compose_gamma_power_priors(
        GammaHyperparameters { shape: 2.0, rate: 1.0 },
        &[(50, 25.0, 0.4), (10, 8.0, 1.0)],
    )
    .unwrap();
    close(gamma.shape, 32.0, "gamma shape");
    close(gamma.rate, 19.0, "gamma rate");
    close(gamma.shape / gamma.rate, 32.0 / 19.0, "gamma prior mean");
    // A zero discount leaves the base; no source leaves the base.
    let base = BetaHyperparameters { alpha: 2.5, beta: 0.5 };
    let unchanged = compose_beta_power_priors(base, &[(7, 9, 0.0)]).unwrap();
    assert_eq!((unchanged.alpha, unchanged.beta), (2.5, 0.5));
    let empty =
        compose_gamma_power_priors(GammaHyperparameters { shape: 3.0, rate: 2.0 }, &[]).unwrap();
    assert_eq!((empty.shape, empty.rate), (3.0, 2.0));
}

#[test]
fn tag_transport_reweights_the_coefficient_to_the_hand_weighted_mean_and_widens_it() {
    // 25 unit effects 0..24 with equal weights: weighted mean 12; the sampling variance of
    // that mean is sum (e - 12)^2 / 25^2 = 2 * (1^2 + ... + 12^2) / 625 = 1300 / 625 = 2.08;
    // the Kish ESS is 25, above the floor of 20. The source has no residual variance, so the
    // conjugate scale is 1 and V0 widens by 2.08.
    let effects: Vec<f64> = (0..25).map(f64::from).collect();
    let adjustment = TransportAdjustment::new(effects, vec![1.0; 25]).unwrap();
    let sources = [source("s", &[0.3, 1.0], &[0.7, 0.4], power(0.5), None)];
    let context = TransportContext {
        source_populations: &[Some("source")],
        target_population: Some("target"),
        policy: Some(TransportPolicy::InvariantConditionalOutcome),
        adjustment: Some(&adjustment),
        coef_index: None,
    };
    let (moved, outcomes) = apply_transport(&sources, &context).unwrap();
    let (mean, variance) = moments(&moved[0].prior);
    // The default coefficient is the last one; the other is untouched.
    close(mean[1], 12.0, "transported mean");
    close(variance[1], 0.4 + 2.08, "transported variance");
    close(mean[0], 0.3, "untouched mean");
    close(variance[0], 0.7, "untouched variance");
    assert!(outcomes[0].required);
    assert_eq!(outcomes[0].alpha_override, None);
    // Composed with the baseline N(0, 100) at alpha 1/2: the precision-weighted closed form.
    let baseline = gaussian(&[0.0, 0.0], &[100.0, 100.0]);
    let (composed, _) = compose_with_transport(&sources, &baseline, &context).unwrap();
    let (mean, variance) = moments(&composed.prior);
    let (m1, v1) = tempered_product(0.0, 100.0, &[(12.0, 2.48, 0.5)]);
    let (m0, v0) = tempered_product(0.0, 100.0, &[(0.3, 0.7, 0.5)]);
    close(mean[1], m1, "composed transported mean");
    close(variance[1], v1, "composed transported variance");
    close(mean[0], m0, "composed untouched mean");
    close(variance[0], v0, "composed untouched variance");

    // A declared coefficient index rewrites that coefficient instead.
    let first = TransportContext { coef_index: Some(0), ..context.clone() };
    let (moved, _) = apply_transport(&sources, &first).unwrap();
    let (mean, variance) = moments(&moved[0].prior);
    close(mean[0], 12.0, "indexed mean");
    close(variance[0], 0.7 + 2.08, "indexed variance");
    close(mean[1], 1.0, "indexed untouched mean");

    // A known residual variance of 2 converts the absolute sampling variance to V0 units.
    let mut scaled = sources[0].clone();
    scaled.prior.push(PriorSpec::KnownResidualVariance(2.0));
    let (moved, _) = apply_transport(&[scaled], &context).unwrap();
    close(moments(&moved[0].prior).1[1], 0.4 + 2.08 / 2.0, "residual-scaled variance");

    // The same population needs no transport: the source is returned unchanged.
    let same = TransportContext {
        source_populations: &[Some("target")],
        target_population: Some("target"),
        policy: None,
        adjustment: None,
        coef_index: None,
    };
    let (kept, outcomes) = apply_transport(&sources, &same).unwrap();
    assert!(!outcomes[0].required);
    assert_eq!(kept[0], sources[0]);

    // Fifteen equally weighted units have Kish ESS 15 < 20: the discount is forced to zero, so
    // the composition is the baseline.
    let thin = TransportAdjustment::new(vec![1.0; 15], vec![1.0; 15]).unwrap();
    let thin_context = TransportContext { adjustment: Some(&thin), ..context.clone() };
    let (composed, outcomes) = compose_with_transport(&sources, &baseline, &thin_context).unwrap();
    assert_eq!(outcomes[0].alpha_override, Some(0.0));
    assert_eq!(&*composed.alphas_requested, &[0.5]);
    assert_eq!(&*composed.alphas_applied, &[0.0]);
    let (mean, variance) = moments(&composed.prior);
    for i in 0..2 {
        close(mean[i], 0.0, "forced-zero mean");
        close(variance[i], 100.0, "forced-zero variance");
    }
}

#[test]
fn weighted_target_transport_composes_to_the_hand_computed_moments_and_overlap() {
    // 60 rows with effect 0.5 * (i mod 6); the first 30 rows weigh 2, the rest 1.
    let effects: Vec<f64> = (0..60).map(|i| 0.5 * f64::from(i % 6)).collect();
    let weights: Vec<f64> = (0..60).map(|i| if i < 30 { 2.0 } else { 1.0 }).collect();
    let mass: f64 = weights.iter().sum();
    let mean_hand = effects.iter().zip(&weights).map(|(e, w)| w * e).sum::<f64>() / mass;
    let mean_variance_hand = effects
        .iter()
        .zip(&weights)
        .map(|(e, w)| w * w * (e - mean_hand) * (e - mean_hand))
        .sum::<f64>()
        / (mass * mass);
    let squares: f64 = weights.iter().map(|w| w * w).sum();
    // Hand totals: sum w = 90, sum w^2 = 150, so the Kish ESS is 8100 / 150 = 54; every
    // residue mod 6 appears five times in each half, so the weighted mean is 1.25.
    close(mass, 90.0, "mass");
    close(mean_hand, 1.25, "weighted mean");
    let kish_hand = mass * mass / squares;
    close(kish_hand, 54.0, "kish");

    // The source carries a known residual variance of 2, so the widening in V0 units is the
    // sampling variance over 2.
    let mut src = source("s", &[4.0], &[0.25], power(0.5), None);
    src.prior.push(PriorSpec::KnownResidualVariance(2.0));
    let target = WeightedTarget::from_row_weights(effects, weights).unwrap();
    let policy = TransportPolicy::InvariantEffectModifiers;
    let (moved, overlap) =
        apply_weighted_target_transport(std::slice::from_ref(&src), policy, &target, None).unwrap();
    let (mean, variance) = moments(&moved[0].prior);
    let v0_hand = 0.25 + mean_variance_hand / 2.0;
    close(mean[0], 1.25, "transported mean");
    close(variance[0], v0_hand, "transported variance");
    assert_eq!(overlap.n_rows, 60);
    assert_eq!(overlap.n_support, 60);
    close(overlap.support_fraction, 1.0, "support fraction");
    close(overlap.kish_ess, 54.0, "overlap kish");
    close(overlap.ess_fraction, 54.0 / 60.0, "ess fraction");
    close(overlap.max_weight_share, 2.0 / 90.0, "max share");

    let baseline = gaussian(&[0.0], &[10.0]);
    let (composed, overlap_again) = compose_with_weighted_target_transport(
        std::slice::from_ref(&src),
        &baseline,
        policy,
        &target,
        None,
    )
    .unwrap();
    assert_eq!(overlap_again, overlap);
    let (mean, variance) = moments(&composed.prior);
    let (m, v) = tempered_product(0.0, 10.0, &[(1.25, v0_hand, 0.5)]);
    close(mean[0], m, "composed mean");
    close(variance[0], v, "composed variance");
}

#[test]
fn kernel_bandwidth_sensitivity_rows_equal_hand_computed_kernel_weights() {
    // 50 rows at distance 0.1 i from the target, with effect 1 + 0.2 i.
    let distances: Vec<f64> = (0..50).map(|i| 0.1 * f64::from(i)).collect();
    let effects: Vec<f64> = (0..50).map(|i| 1.0 + 0.2 * f64::from(i)).collect();
    let bandwidths = [0.05, 0.5, 2.0];
    let floor = 0.05;
    let rows = kernel_bandwidth_sensitivity(
        KernelFamily::Gaussian,
        &bandwidths,
        &distances,
        &effects,
        floor,
    )
    .unwrap();
    assert_eq!(rows.len(), 3);
    for (row, &h) in rows.iter().zip(&bandwidths) {
        assert_eq!(row.bandwidth, h);
        // The Gaussian kernel exp(-u^2 / 2), u = d / h, by hand.
        let w: Vec<f64> = distances
            .iter()
            .map(|d| {
                let u = d / h;
                (-0.5 * u * u).exp()
            })
            .collect();
        let mass: f64 = w.iter().sum();
        let mean = w.iter().zip(&effects).map(|(w, e)| w * e).sum::<f64>() / mass;
        let mean_variance =
            w.iter().zip(&effects).map(|(w, e)| w * w * (e - mean) * (e - mean)).sum::<f64>()
                / (mass * mass);
        let kish = mass * mass / w.iter().map(|w| w * w).sum::<f64>();
        let support = w.iter().filter(|&&w| w > 0.0).count();
        let share = w.iter().copied().fold(0.0, f64::max) / mass;
        close(row.weighted_mean.unwrap(), mean, "weighted mean");
        close(row.mean_variance.unwrap(), mean_variance, "mean variance");
        let overlap = row.overlap.unwrap();
        assert_eq!(overlap.n_rows, 50);
        assert_eq!(overlap.n_support, support);
        close(overlap.kish_ess, kish, "kish");
        close(overlap.ess_fraction, kish / 50.0, "ess fraction");
        close(overlap.max_weight_share, share, "max share");
        // The overlap check: positive support, Kish ESS at least 20 and at least the fraction.
        let passes = support > 0 && kish >= 20.0 && kish / 50.0 >= floor;
        assert_eq!(row.overlap_ok, passes, "bandwidth {h}");
        assert_eq!(row.refusal.is_none(), passes, "bandwidth {h}");
    }
    // h = 0.05: weights 1, exp(-2), exp(-8), ...: Kish ESS about 1.3, so the check fails.
    // h = 2: weights exp(-i^2 / 800): Kish ESS about 35 of 50 rows, so it passes.
    assert!(!rows[0].overlap_ok);
    assert!(rows[0].overlap.unwrap().kish_ess < 2.0);
    assert!(rows[2].overlap_ok);
    assert!(rows[2].overlap.unwrap().kish_ess > 30.0);

    // A compact kernel with no row inside its support reports no mean and fails the check.
    let far: Vec<f64> = (0..50).map(|i| 1.0 + 0.1 * f64::from(i)).collect();
    let empty =
        kernel_bandwidth_sensitivity(KernelFamily::Epanechnikov, &[0.5], &far, &effects, floor)
            .unwrap();
    assert_eq!(empty[0].weighted_mean, None);
    assert_eq!(empty[0].mean_variance, None);
    assert_eq!(empty[0].overlap.unwrap().n_support, 0);
    assert!(!empty[0].overlap_ok && empty[0].refusal.is_some());

    // A declared kernel target gives the same transported mean as the hand weights.
    let kernel = TransportKernel::new(KernelFamily::Gaussian, 2.0).unwrap();
    let target = WeightedTarget::from_kernel(effects, &distances, kernel).unwrap();
    let (moved, _) = apply_weighted_target_transport(
        &[source("s", &[0.0], &[1.0], power(1.0), None)],
        TransportPolicy::InvariantConditionalOutcome,
        &target,
        None,
    )
    .unwrap();
    close(moments(&moved[0].prior).0[0], rows[2].weighted_mean.unwrap(), "kernel target mean");
}

#[test]
fn mixture_composition_and_claim_only_transport_match_their_closed_forms() {
    // Mixture: weight 0.4 on N(10, 1 / alpha) with alpha 1/2 (so variance 2), leftover 0.6 on
    // the baseline N(0, 100). Mean 0.4 * 10 = 4; second moment 0.4 * (2 + 100) + 0.6 * 100 =
    // 100.8; variance 100.8 - 16 = 84.8. The mixture path reports no composed ESS.
    let mixed = compose_external_priors(
        &[source(
            "m",
            &[10.0],
            &[1.0],
            ExternalPriorWeight::power_mixture(0.5, 0.4).unwrap(),
            Some(50.0),
        )],
        &gaussian(&[0.0], &[100.0]),
    )
    .unwrap();
    let (mean, variance) = moments(&mixed.prior);
    close(mean[0], 4.0, "mixture mean");
    close(variance[0], 84.8, "mixture variance");
    assert_eq!(mixed.composed_ess, None);
    close(mixed.effective_ess[0].unwrap(), 25.0, "mixture effective ess");

    // A claim-only transfer (populations differ, an invariance declared, no reweighting)
    // keeps the source's moments and discount, so the composition is the plain closed form.
    let sources = [source("c", &[1.5], &[0.5], power(0.8), None)];
    let context = TransportContext {
        source_populations: &[Some("us")],
        target_population: Some("eu"),
        policy: Some(TransportPolicy::InvariantEffectModifiers),
        adjustment: None,
        coef_index: None,
    };
    let baseline = gaussian(&[0.0], &[4.0]);
    let (composed, outcomes) = compose_with_transport(&sources, &baseline, &context).unwrap();
    assert!(outcomes[0].required && outcomes[0].alpha_override.is_none());
    let (mean, variance) = moments(&composed.prior);
    let (m, v) = tempered_product(0.0, 4.0, &[(1.5, 0.5, 0.8)]);
    close(mean[0], m, "claim-only mean");
    close(variance[0], v, "claim-only variance");
}

#[test]
fn composition_and_transport_refusals_name_what_is_wrong() {
    let baseline = gaussian(&[0.0], &[4.0]);
    // Mixture weights on some sources but not others are refused, not guessed.
    let mixed_modes = [
        source("a", &[1.0], &[1.0], power(1.0), None),
        source("b", &[2.0], &[1.0], ExternalPriorWeight::power_mixture(1.0, 0.5).unwrap(), None),
    ];
    assert!(matches!(
        compose_external_priors(&mixed_modes, &baseline),
        Err(ProbError::InvalidPrior { .. })
    ));
    // A source of a different dimension is a shape refusal.
    let wide = [source("w", &[1.0, 2.0], &[1.0, 1.0], power(1.0), None)];
    assert!(matches!(compose_external_priors(&wide, &baseline), Err(ProbError::Shape { .. })));
    // A count discount outside [0, 1], and more successes than trials, are refused.
    let base = BetaHyperparameters { alpha: 1.0, beta: 1.0 };
    assert!(compose_beta_power_priors(base, &[(3, 10, 1.5)]).is_err());
    assert!(compose_beta_power_priors(base, &[(11, 10, 0.5)]).is_err());
    let rate = GammaHyperparameters { shape: 1.0, rate: 1.0 };
    assert!(compose_gamma_power_priors(rate, &[(3, -1.0, 0.5)]).is_err());
    // Differing populations without a declared invariance are refused with a typed code.
    let sources = [source("s", &[1.0], &[1.0], power(1.0), None)];
    let undeclared = TransportContext {
        source_populations: &[Some("us")],
        target_population: Some("eu"),
        policy: None,
        adjustment: None,
        coef_index: None,
    };
    let error = apply_transport(&sources, &undeclared).unwrap_err();
    assert_eq!(error.code(), "transport_policy_required");
    let error = compose_with_transport(&sources, &baseline, &undeclared).unwrap_err();
    assert_eq!(refusal_code(&error), "transport_policy_required");
    // An invariant propensity without weights forces the discount to zero with its reason.
    let propensity = TransportContext {
        policy: Some(TransportPolicy::InvariantPropensity),
        ..undeclared.clone()
    };
    let (_, outcomes) = apply_transport(&sources, &propensity).unwrap();
    assert_eq!(outcomes[0].alpha_override, Some(0.0));
    assert!(outcomes[0].zero_reason.as_deref().unwrap().contains("target_weights"));
    // A weighted target with 15 rows has Kish ESS 15 < 20 and is refused with a next attempt.
    let thin = WeightedTarget::from_row_weights(vec![1.0; 15], vec![1.0; 15]).unwrap();
    let error = apply_weighted_target_transport(
        &sources,
        TransportPolicy::InvariantConditionalOutcome,
        &thin,
        None,
    )
    .unwrap_err();
    assert_eq!(refusal_code(&error), "transport_target_overlap_failed");
    assert!(error.to_string().contains("Next attempt"));
    // A coefficient index past the prior's width is refused.
    let adjustment = TransportAdjustment::new(vec![1.0; 30], vec![1.0; 30]).unwrap();
    let out_of_range = TransportContext {
        policy: Some(TransportPolicy::InvariantConditionalOutcome),
        adjustment: Some(&adjustment),
        coef_index: Some(3),
        ..undeclared
    };
    assert_eq!(
        apply_transport(&sources, &out_of_range).unwrap_err().code(),
        "transport_coef_index_out_of_range"
    );
    // An empty bandwidth set is refused.
    assert!(
        kernel_bandwidth_sensitivity(KernelFamily::Gaussian, &[], &[0.0], &[1.0], 0.05).is_err()
    );
}

fn restriction<'a>(composed: &'a ComposedPrior, id: &str) -> Vec<&'a str> {
    composed
        .prior
        .restrictions
        .iter()
        .filter(|r| r.id.as_ref() == id)
        .map(|r| r.description.as_ref())
        .collect()
}

#[test]
fn the_composed_prior_records_its_composition_transport_and_kernel_assumptions() {
    let baseline = gaussian(&[0.0], &[4.0]);
    // The composition records every source with its requested and applied discounts.
    let sources = [
        source("first", &[1.0], &[1.0], power(0.5), None),
        source("second", &[2.0], &[1.0], power(0.25), None),
    ];
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    let described = restriction(&composed, "external_composed_prior");
    assert_eq!(described.len(), 1);
    assert!(described[0].contains("first:alpha_req=0.5,alpha_app=0.5,w=none"));
    assert!(described[0].contains("second:alpha_req=0.25,alpha_app=0.25,w=none"));

    // A forced-zero transport records the requested and the applied discount apart, and the
    // transport claim with its populations.
    let context = TransportContext {
        source_populations: &[Some("us"), None],
        target_population: Some("eu"),
        policy: Some(TransportPolicy::InvariantPropensity),
        adjustment: None,
        coef_index: None,
    };
    let (transported, _) = compose_with_transport(&sources, &baseline, &context).unwrap();
    let described = restriction(&transported, "external_composed_prior");
    assert!(described[0].contains("first:alpha_req=0.5,alpha_app=0,w=none"));
    let claims = restriction(&transported, "external_transport_prior");
    assert_eq!(claims.len(), 2);
    assert!(claims[0].contains("TransportPolicy invariant_propensity for source=first"));
    assert!(claims[0].contains("source_pop=us target_pop=eu"));

    // A kernel target records the weighting and the kernel, whose id carries the exact
    // bandwidth bits, so two bandwidths give two different prior identities.
    let effects: Vec<f64> = (0..100).map(|i| 0.01 * f64::from(i)).collect();
    let distances: Vec<f64> = (0..100).map(|i| 0.02 * f64::from(i)).collect();
    for h in [0.75, 1.5] {
        let kernel = TransportKernel::new(KernelFamily::Gaussian, h).unwrap();
        let target = WeightedTarget::from_kernel(effects.clone(), &distances, kernel).unwrap();
        let (composed, _) = compose_with_weighted_target_transport(
            &sources[..1],
            &baseline,
            TransportPolicy::InvariantConditionalOutcome,
            &target,
            None,
        )
        .unwrap();
        assert_eq!(restriction(&composed, "external_transport_target_weighting").len(), 1);
        let kernel_id = format!("transport_kernel:gaussian:bandwidth_bits={:016x}", h.to_bits());
        assert_eq!(restriction(&composed, &kernel_id).len(), 1, "{kernel_id}");
    }
}
