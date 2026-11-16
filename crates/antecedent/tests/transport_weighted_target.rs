//! Known-truth tests of kernel transport weights and row-weighted target
//! populations.
//!
//! Every expected value is derived by hand (closed forms on planted rows), not
//! by calling the code under test, and the tag-defined transport path is
//! checked against the weighted path where they must agree.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_io_core::prior_set_identity;
use antecedent_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, KernelFamily, PriorSet,
    PriorSpec, ProbError, TransportAdjustment, TransportContext, TransportKernel, TransportPolicy,
    WeightedTarget, apply_transport, apply_weighted_target_transport,
    compose_with_weighted_target_transport, kernel_bandwidth_sensitivity,
};

const TOL: f64 = 1e-12;

fn source(mean: f64, variance: f64) -> ExternalPriorSource {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(1, mean, variance).unwrap(),
    ));
    ExternalPriorSource {
        id: Arc::from("src"),
        prior,
        weight: ExternalPriorWeight::power(0.5).unwrap(),
        ess: None,
    }
}

fn baseline() -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(1, 0.0, 100.0).unwrap(),
    ));
    prior
}

fn moments(source: &ExternalPriorSource) -> (f64, f64) {
    let coef = source.prior.gaussian_coefficients().unwrap();
    (coef.mean[0], coef.variance[0])
}

fn refusal_code(error: &ProbError) -> &'static str {
    match error {
        ProbError::Refused { code, .. } => *code,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

const POLICY: TransportPolicy = TransportPolicy::InvariantConditionalOutcome;

#[test]
fn kernel_weights_equal_hand_computed_values() {
    let distances = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    let h = 2.0;
    let gaussian = TransportKernel::new(KernelFamily::Gaussian, h).unwrap();
    let epanechnikov = TransportKernel::new(KernelFamily::Epanechnikov, h).unwrap();
    let tricube = TransportKernel::new(KernelFamily::Tricube, h).unwrap();
    // u = d / 2 = 0, 1/4, 1/2, 3/4, 1, 3/2.
    let gaussian_hand = [
        1.0,
        (-0.5_f64 * 0.0625).exp(),
        (-0.5_f64 * 0.25).exp(),
        (-0.5_f64 * 0.5625).exp(),
        (-0.5_f64).exp(),
        (-0.5_f64 * 2.25).exp(),
    ];
    let epanechnikov_hand = [1.0, 1.0 - 0.0625, 1.0 - 0.25, 1.0 - 0.5625, 0.0, 0.0];
    let tricube_hand = [
        1.0,
        (1.0_f64 - 1.0 / 64.0).powi(3),
        (1.0_f64 - 1.0 / 8.0).powi(3),
        (1.0_f64 - 27.0 / 64.0).powi(3),
        0.0,
        0.0,
    ];
    for (kernel, hand) in
        [(gaussian, gaussian_hand), (epanechnikov, epanechnikov_hand), (tricube, tricube_hand)]
    {
        let weights = kernel.weights(&distances).unwrap();
        for (got, want) in weights.iter().zip(hand) {
            assert!((got - want).abs() < 1e-15, "{:?}: {got} vs {want}", kernel.family());
        }
    }
}

#[test]
fn invalid_kernel_declarations_are_refused() {
    assert!(TransportKernel::new(KernelFamily::Gaussian, 0.0).is_err());
    assert!(TransportKernel::new(KernelFamily::Gaussian, -1.0).is_err());
    assert!(TransportKernel::new(KernelFamily::Gaussian, f64::NAN).is_err());
    assert!(KernelFamily::parse("boxcar").is_err());
    let kernel = TransportKernel::new(KernelFamily::Tricube, 1.0).unwrap();
    assert!(kernel.weights(&[0.5, -0.1]).is_err());
    assert!(kernel.weights(&[]).is_err());
}

#[test]
fn neighbourhood_effect_is_recovered_exactly_from_row_weights() {
    // 100 source rows: the 40 inside the neighbourhood have effect 2, the 60 outside have 5.
    // The target weights the neighbourhood rows equally, so the transported mean is 2 and,
    // with no spread among the weighted effects, adds no variance to the source's V0.
    let effects: Vec<f64> = (0..100).map(|i| if i < 40 { 2.0 } else { 5.0 }).collect();
    let weights: Vec<f64> = (0..100).map(|i| if i < 40 { 1.0 } else { 0.0 }).collect();
    let target = WeightedTarget::from_row_weights(effects, weights).unwrap();
    let (out, overlap) =
        apply_weighted_target_transport(&[source(9.0, 0.7)], POLICY, &target, None).unwrap();
    let (mean, variance) = moments(&out[0]);
    assert!((mean - 2.0).abs() < TOL);
    assert!((variance - 0.7).abs() < TOL);
    assert_eq!(overlap.n_rows, 100);
    assert_eq!(overlap.n_support, 40);
    assert!((overlap.kish_ess - 40.0).abs() < TOL);
    assert!((overlap.ess_fraction - 0.4).abs() < TOL);
    assert!((overlap.max_weight_share - 0.025).abs() < TOL);
}

#[test]
fn graded_weights_follow_the_closed_form_mean_and_sampling_variance() {
    // Rows i = 0..40 with effect i and weight i + 1.
    // sum w = 820, sum w e = sum i (i + 1) = 21320, so the mean is 26.
    // sum w^2 = 22140, so the Kish ESS is 820^2 / 22140.
    // sum w^2 (e - 26)^2 = sum (i + 1)^2 (i - 26)^2, taken exactly over integers.
    let effects: Vec<f64> = (0..40).map(f64::from).collect();
    let weights: Vec<f64> = (0..40).map(|i| f64::from(i) + 1.0).collect();
    let numerator: i64 = (0..40_i64).map(|i| (i + 1) * (i + 1) * (i - 26) * (i - 26)).sum();
    let hand_variance = numerator as f64 / (820.0 * 820.0);
    let target = WeightedTarget::from_row_weights(effects, weights).unwrap();
    let (out, overlap) =
        apply_weighted_target_transport(&[source(0.0, 0.25)], POLICY, &target, None).unwrap();
    let (mean, variance) = moments(&out[0]);
    assert!((mean - 26.0).abs() < TOL);
    assert!((variance - (0.25 + hand_variance)).abs() < 1e-10);
    assert!((overlap.kish_ess - 820.0 * 820.0 / 22140.0).abs() < 1e-9);
}

#[test]
fn weighted_target_reproduces_the_tag_path_moments_for_the_same_weights() {
    // The new variant uses the existing reweight arithmetic; with enough overlap
    // the two paths agree to the bit on the transported coefficient moments.
    let effects: Vec<f64> = (0..60).map(|i| f64::from(i % 7) + 0.25 * f64::from(i)).collect();
    let weights: Vec<f64> = (0..60).map(|i| 1.0 + f64::from(i % 5)).collect();
    let target = WeightedTarget::from_row_weights(effects.clone(), weights.clone()).unwrap();
    let (weighted, _) =
        apply_weighted_target_transport(&[source(1.0, 0.4)], POLICY, &target, None).unwrap();
    let adjustment = TransportAdjustment::new(effects, weights).unwrap();
    let ctx = TransportContext {
        source_populations: &[Some("source")],
        target_population: Some("target"),
        policy: Some(POLICY),
        adjustment: Some(&adjustment),
        coef_index: None,
    };
    let (tagged, _) = apply_transport(&[source(1.0, 0.4)], &ctx).unwrap();
    let (wm, wv) = moments(&weighted[0]);
    let (tm, tv) = moments(&tagged[0]);
    assert_eq!(wm.to_bits(), tm.to_bits());
    assert_eq!(wv.to_bits(), tv.to_bits());
}

#[test]
fn tag_targets_keep_their_decision_when_the_weights_are_thin() {
    // A tag-defined target with a thin weighting still forces alpha to 0 and does not refuse;
    // only the row-weighted target refuses. The tag path is not routed through the new check.
    let effects: Vec<f64> = (0..15).map(f64::from).collect();
    let weights = vec![1.0; 15];
    let adjustment = TransportAdjustment::new(effects.clone(), weights.clone()).unwrap();
    let ctx = TransportContext {
        source_populations: &[Some("source")],
        target_population: Some("target"),
        policy: Some(POLICY),
        adjustment: Some(&adjustment),
        coef_index: None,
    };
    let (_, outcomes) = apply_transport(&[source(1.0, 0.4)], &ctx).unwrap();
    assert_eq!(outcomes[0].alpha_override, Some(0.0));
    let target = WeightedTarget::from_row_weights(effects, weights).unwrap();
    let error =
        apply_weighted_target_transport(&[source(1.0, 0.4)], POLICY, &target, None).unwrap_err();
    assert_eq!(refusal_code(&error), "transport_target_overlap_failed");
}

#[test]
fn degenerate_overlap_is_refused_with_a_next_attempt() {
    let effects: Vec<f64> = (0..200).map(f64::from).collect();
    let distances: Vec<f64> = (0..200).map(f64::from).collect();
    // A narrow kernel centred on one row: Kish ESS ~ 1.
    let narrow = TransportKernel::new(KernelFamily::Gaussian, 0.4).unwrap();
    let target = WeightedTarget::from_kernel(effects.clone(), &distances, narrow).unwrap();
    let error = compose_with_weighted_target_transport(
        &[source(0.0, 1.0)],
        &baseline(),
        POLICY,
        &target,
        None,
    )
    .unwrap_err();
    assert_eq!(refusal_code(&error), "transport_target_overlap_failed");
    assert!(error.to_string().contains("Next attempt"));
    // A compact kernel with no row inside its support has no positivity at all.
    let far: Vec<f64> = (0..200).map(|i| 50.0 + f64::from(i)).collect();
    let compact = TransportKernel::new(KernelFamily::Epanechnikov, 5.0).unwrap();
    let error = WeightedTarget::from_kernel(effects, &far, compact).unwrap_err();
    assert_eq!(refusal_code(&error), "transport_target_overlap_failed");
}

#[test]
fn the_absolute_ess_floor_is_never_relaxed_by_the_fraction_floor() {
    let effects: Vec<f64> = (0..15).map(f64::from).collect();
    let target = WeightedTarget::from_row_weights(effects, vec![1.0; 15])
        .unwrap()
        .with_min_ess_fraction(0.0)
        .unwrap();
    assert!(apply_weighted_target_transport(&[source(0.0, 1.0)], POLICY, &target, None).is_err());
    // 1000 rows, 30 of them weighted: ESS 30 clears the absolute floor of 20 but not a
    // fraction floor of 0.05 (50 rows); declaring a lower fraction admits it.
    let effects: Vec<f64> = (0..1000).map(f64::from).collect();
    let weights: Vec<f64> = (0..1000).map(|i| if i < 30 { 1.0 } else { 0.0 }).collect();
    let default = WeightedTarget::from_row_weights(effects.clone(), weights.clone()).unwrap();
    assert!(apply_weighted_target_transport(&[source(0.0, 1.0)], POLICY, &default, None).is_err());
    let declared = WeightedTarget::from_row_weights(effects, weights)
        .unwrap()
        .with_min_ess_fraction(0.01)
        .unwrap();
    assert!(apply_weighted_target_transport(&[source(0.0, 1.0)], POLICY, &declared, None).is_ok());
}

#[test]
fn prior_identity_changes_with_kernel_family_and_bandwidth() {
    let effects: Vec<f64> = (0..400).map(|i| f64::from(i) * 0.01).collect();
    let distances: Vec<f64> = (0..400).map(|i| (f64::from(i) - 200.0).abs() * 0.01).collect();
    let identity = |family, bandwidth| {
        let kernel = TransportKernel::new(family, bandwidth).unwrap();
        let target = WeightedTarget::from_kernel(effects.clone(), &distances, kernel).unwrap();
        let (out, _) =
            apply_weighted_target_transport(&[source(0.0, 1.0)], POLICY, &target, None).unwrap();
        prior_set_identity(&out[0].prior)
    };
    let base = identity(KernelFamily::Gaussian, 0.5);
    assert_eq!(base, identity(KernelFamily::Gaussian, 0.5));
    assert_ne!(base, identity(KernelFamily::Gaussian, 0.6));
    assert_ne!(base, identity(KernelFamily::Tricube, 0.5));
    // Same weights, no kernel declared: a different identity.
    let weights =
        TransportKernel::new(KernelFamily::Gaussian, 0.5).unwrap().weights(&distances).unwrap();
    let plain = WeightedTarget::from_row_weights(effects, weights).unwrap();
    let (out, _) =
        apply_weighted_target_transport(&[source(0.0, 1.0)], POLICY, &plain, None).unwrap();
    assert_ne!(base, prior_set_identity(&out[0].prior));
}

#[test]
fn bandwidth_sensitivity_reports_the_known_mean_and_where_overlap_fails() {
    // Rows at x = -2.00, -1.99, ..., 1.99 with effect x; the target is x = 0, so each row's
    // distance is |x| and the weighted mean of the effects is (sum w x) / (sum w) exactly.
    let xs: Vec<f64> = (0..400).map(|i| -2.0 + 0.01 * f64::from(i)).collect();
    let distances: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
    let bandwidths = [0.02, 0.1, 0.5, 1.0];
    let rows = kernel_bandwidth_sensitivity(
        KernelFamily::Epanechnikov,
        &bandwidths,
        &distances,
        &xs,
        0.05,
    )
    .unwrap();
    assert_eq!(rows.len(), 4);
    for row in &rows {
        let h = row.bandwidth;
        let (mut sw, mut swx) = (0.0, 0.0);
        for &x in &xs {
            let u = x.abs() / h;
            let w = if u < 1.0 { 1.0 - u * u } else { 0.0 };
            sw += w;
            swx += w * x;
        }
        assert!((row.weighted_mean.unwrap() - swx / sw).abs() < 1e-12);
    }
    // h = 0.02 puts three rows in the support: refused. h >= 0.5 puts more than 20
    // effective rows in: supported.
    assert!(!rows[0].overlap_ok);
    assert!(rows[0].refusal.as_deref().unwrap().contains("Next attempt"));
    assert!(rows[2].overlap_ok && rows[3].overlap_ok);
    // Wider bandwidths have a larger effective sample size.
    let ess: Vec<f64> = rows.iter().map(|r| r.overlap.unwrap().kish_ess).collect();
    assert!(ess.windows(2).all(|w| w[0] < w[1]));
}
